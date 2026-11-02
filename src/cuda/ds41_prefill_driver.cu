// deepseek41 CUDA prefill driver (cuda_ds41_prefill.hpp): the DevicePrefill plan on the CUDA layer kernels, with the
// cone, the tiered routed-expert provider and the Engram rows from the model file.

#include "strix/cuda_ds41_prefill.hpp"
#include "strix/cuda_ds41_replay.hpp"
#include "strix/cuda_ds41_router_v2.hpp"

#include "ds41_expert_streamer.hpp"
#include "strix/cuda_ds41_attention.hpp"
#include "strix/cuda_ds41_core.hpp"
#include "strix/cuda_ds41_gpu_profile.hpp"
#include "strix/cuda_ds41_dense_fast.hpp"
#include "strix/cuda_ds41_fast_attention.hpp"
#include "strix/cuda_ds41_layer_core.hpp"
#include "strix/cuda_ds41_moe.hpp"
#include "strix/cuda_ds41_prefill_kernels.hpp"
#include "strix/cuda_ds41_prefill_q8.hpp"
#include "strix/cuda_ds41_moe_fast.hpp"
#include "strix/cuda_ds41_weights.hpp"
#include "strix/deepseek41_model.hpp"
#include "strix/ds41_attention.hpp"
#include "strix/ds41_engram.hpp"
#include "strix/ds41_file_reader.hpp"
#include "strix/ds41_layer.hpp"
#include "strix/ds41_prefill.hpp"
#ifdef STRIX_DS41_CUDA_TRELLIS
#include "strix/cuda_trellis.hpp"
#include "strix/ds41_moe.hpp"
#endif
#include "strix/ds41_memory_budget.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <tuple>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strix::ds41::cuda {
namespace {

namespace at = ::strix::ds41::cuda::attn;
namespace pk = ::strix::ds41::cuda::pk;

#ifdef STRIX_DS41_CUDA_TRELLIS
// One tensor of a safetensors file (the trellis packs): dtype, shape and its bytes.
struct SafetensorBlob {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::vector<std::uint8_t> bytes;
};
SafetensorBlob read_safetensor(const std::filesystem::path& path, const std::string& name) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("trellis pack: cannot open " + path.string());
    std::uint64_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 8);
    if (!f || n < 2U || n > (std::uint64_t{64} << 20U)) throw std::runtime_error("trellis pack: bad header in " + path.string());
    std::string h(static_cast<std::size_t>(n), '\0');
    f.read(h.data(), static_cast<std::streamsize>(n));
    const std::string key = "\"" + name + "\"";
    const auto at = h.find(key);
    if (at == std::string::npos) throw std::runtime_error("trellis pack: no tensor " + name + " in " + path.string());
    const auto end = h.find('}', at);
    const std::string d = h.substr(at, end - at);
    SafetensorBlob b;
    const auto field = [&](const char* k) {
        const auto p = d.find(std::string("\"") + k + "\"");
        if (p == std::string::npos) throw std::runtime_error("trellis pack: tensor " + name + " without " + k);
        return d.substr(d.find(':', p) + 1U);
    };
    {
        const auto v = field("dtype");
        const auto q0 = v.find('"'), q1 = v.find('"', q0 + 1U);
        b.dtype = v.substr(q0 + 1U, q1 - q0 - 1U);
    }
    const auto numbers = [](const std::string& v) {
        std::vector<std::uint64_t> out;
        const auto open = v.find('['), close = v.find(']');
        std::stringstream ss(v.substr(open + 1U, close - open - 1U));
        for (std::string t; std::getline(ss, t, ',');)
            if (t.find_first_of("0123456789") != std::string::npos) out.push_back(std::stoull(t));
        return out;
    };
    b.shape = numbers(field("shape"));
    const auto off = numbers(field("data_offsets"));
    if (off.size() != 2U || off[1] < off[0]) throw std::runtime_error("trellis pack: bad offsets of " + name);
    b.bytes.resize(static_cast<std::size_t>(off[1] - off[0]));
    f.seekg(static_cast<std::streamoff>(8U + n + off[0]));
    f.read(reinterpret_cast<char*>(b.bytes.data()), static_cast<std::streamsize>(b.bytes.size()));
    if (!f) throw std::runtime_error("trellis pack: short read of " + name);
    return b;
}
// The pair library's split count of a GEMV of R <= 8 rows at integer K (its vector policy for every geometry and K
// without a measured override; the overrides are K5 / K6 only): output <= 1024: 32; output <= 4096 and input >= 4096:
// 16; else 4; at most input / 16.
std::uint32_t trellis_vector_splits(std::uint32_t input, std::uint32_t output, std::uint32_t bits) {
    if (bits == 5U || bits == 6U) throw std::invalid_argument("cuda trellis shexp: K5 / K6 have measured split overrides (not supported here)");
    const std::uint32_t s = output <= 1024U ? 32U : (output <= 4096U && input >= 4096U ? 16U : 4U);
    return std::min(s, input / 16U);
}
__global__ void trellis_swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up, std::size_t n, float limit,
                                      float* __restrict__ act) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i < n) act[i] = swiglu_clamped(gate[i], up[i], limit);
}
#endif

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

std::uint64_t fnv_bytes(const void* data, std::size_t n, std::uint64_t h = 14695981039346656037ULL) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("ds41 cuda prefill: ") + what);
}

// Lazy host promotion keeps at least the service's opportunistic host-tier floor free (--host-gib-auto's default).
constexpr double kHostPromoteFloorGiB = 20.0;
// The streamer only, a full prompt per run (pass_rows' bounded host budget does not count it), real weights.
bool host_promote_effective(const CudaPrefillOptions& o) {
    return o.host_promote && o.pass_rows == 0U && o.host_alias_layers == 0U && !o.decode_only;
}

// Pinned host allocation (cudaHostAlloc), owned.
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(std::size_t bytes) { resize(bytes); }
    ~PinnedBuffer() { release(); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
    void resize(std::size_t bytes, bool mapped = false) {
        release();
        if (bytes) STRIX_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&p_), bytes, mapped ? cudaHostAllocMapped : cudaHostAllocDefault));
        n_ = bytes;
        mapped_ = mapped;
    }
    // The device address of a mapped buffer (kernels read it over PCIe, interleaved with other transfers instead of
    // queued behind them on a copy engine).
    [[nodiscard]] const std::uint8_t* device() const {
        void* d = nullptr;
        STRIX_CUDA_CHECK(cudaHostGetDevicePointer(&d, p_, 0));
        return static_cast<const std::uint8_t*>(d);
    }
    [[nodiscard]] std::uint8_t* get() const noexcept { return p_; }
    [[nodiscard]] std::size_t size() const noexcept { return n_; }
    // No exception: false when cudaHostAlloc fails (the error is cleared).
    bool try_resize(std::size_t bytes) noexcept {
        release();
        void* p = nullptr;
        if (bytes == 0U) return true;
        if (cudaHostAlloc(&p, bytes, cudaHostAllocDefault) != cudaSuccess || p == nullptr) {
            (void)cudaGetLastError();
            return false;
        }
        p_ = static_cast<std::uint8_t*>(p);
        n_ = bytes;
        mapped_ = false;
        return true;
    }

private:
    void release() noexcept {
        if (p_) cudaFreeHost(p_);
        p_ = nullptr;
        n_ = 0;
    }
    std::uint8_t* p_ = nullptr;
    std::size_t n_ = 0;
    bool mapped_ = false;
};

// Telemetry failure is explicit; optional pins decline when measurements fail.
static void system_room(double& commit, double& physical) {
    const auto r = ds41::host_memory_room();
    commit = r.known ? double(r.free_commit) : -1;
    physical = r.known ? double(r.free_physical) : -1;
}
bool pin_with_floor(PinnedBuffer& out, std::size_t n, double floor_gib,
        std::chrono::steady_clock::time_point deadline, std::uint32_t& waits,
        bool implicit, bool allow_first, HostPinReason& reason) {
    std::uint64_t before_commit = 0;
    reason = ds41::allocate_host_pin(n, floor_gib, implicit, allow_first, deadline, waits,
        [&] { auto r = ds41::host_memory_room(); before_commit = r.free_commit; return r; }, [&] { return out.try_resize(n); },
        [](double ask) {
#ifdef _WIN32
            if (void* q = VirtualAlloc(nullptr, static_cast<SIZE_T>(ask), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)) VirtualFree(q, 0, MEM_RELEASE);
#else
            (void)ask;
#endif
        }, [] { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); },
        [] { return std::chrono::steady_clock::now(); });
    if (!implicit && reason == HostPinReason::none) {
        for (int i = 0; i < 30 && std::chrono::steady_clock::now() < deadline; ++i) {
            const auto r = ds41::host_memory_room();
            if (!r.known || (before_commit >= r.free_commit && double(before_commit - r.free_commit) >= .9 * double(n))) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    return reason == HostPinReason::none;
}

struct Event {
    cudaEvent_t e = nullptr;
    explicit Event(bool timing = false) {
        STRIX_CUDA_CHECK(cudaEventCreateWithFlags(&e, timing ? cudaEventDefault : cudaEventDisableTiming));
    }
    ~Event() {
        if (e) cudaEventDestroy(e);
    }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

// ------------------------------------------------------------------------------------------------ expert provider
// Every layer's routed experts as one block: gate [384][2304][5120] IQ2_XXS, up likewise, down [384][5120][2304] Q2_K.
struct ExpertExtent {
    std::uint64_t offset[3]{}, bytes[3]{};
    [[nodiscard]] std::uint64_t total() const { return bytes[0] + bytes[1] + bytes[2]; }
};

class ExpertProvider {
public:
    ExpertProvider(FileReader& file, std::vector<ExpertExtent> extents, std::vector<ExpertTier> tiers, std::uint32_t first,
                   std::uint32_t end, std::uint32_t slots, cudaStream_t load_stream, std::uint32_t host_alias = 0,
                   double host_commit_floor_gib = 0.0, double host_pin_wait_s = 360.0, bool implicit = false)
        : file_(file), extents_(std::move(extents)), tiers_(std::move(tiers)), first_(first), end_(end), host_alias_(host_alias) {
        layer_bytes_ = 0;
        for (std::uint32_t l = first; l < end; ++l) layer_bytes_ = std::max(layer_bytes_, extents_[l].total());
        STRIX_CUDA_CHECK(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking));
        (void)load_stream;
        // pinned staging of file reads (resident layers at load, file-tier layers per run): only when something uses it
        // (copy_file_to_device). A decode-only object with no device-tier layer and no slots allocates none (0.25 GiB).
        bool uploads = false, streams = false;
        for (std::uint32_t l = first; l < end; ++l) {
            uploads = uploads || tiers_[l] == ExpertTier::device;
            streams = streams || tiers_[l] == ExpertTier::file || tiers_[l] == ExpertTier::host;
        }
        if (uploads || (streams && slots != 0U)) {
            staging_.resize(kStaging);
            for (auto& st : staging_) {
                st.buf = std::make_unique<PinnedBuffer>(kStagingBytes + 2U * FileReader::alignment);
                st.done = std::make_unique<Event>();
                pinned_bytes_ += st.buf->size();
            }
        }
        resident_.resize(end);
        host_.resize(end);
        host_view_.assign(end, nullptr);
        decode_view_.assign(end, nullptr);
        promoted_.resize(end);
        // resident layers and the slots first, the host tier last (with a commit floor it takes what the device left)
        for (std::uint32_t l = first; l < end; ++l) {
            if (tiers_[l] != ExpertTier::device) streamed_.push_back(l);
            if (tiers_[l] == ExpertTier::device) {
                resident_[l] = std::make_unique<DeviceBuffer<std::uint8_t>>(extents_[l].total());
                upload_layer(l, resident_[l]->get());
                device_bytes_ += extents_[l].total();
                resident_bytes_ += extents_[l].total();
            }
        }
        std::sort(streamed_.begin(), streamed_.end());
        // slots 0: no streaming slots (a decode-only object: steps read the routed experts of non-resident layers sparsely)
        const std::uint32_t nslots = streamed_.empty() || slots == 0U ? 0U : std::min<std::uint32_t>(slots, static_cast<std::uint32_t>(streamed_.size()));
        for (std::uint32_t k = 0; k < nslots; ++k) {
            slots_.emplace_back();
            auto& s = slots_.back();
            s.buf = std::make_unique<DeviceBuffer<std::uint8_t>>(layer_bytes_);
            s.ready = std::make_unique<Event>();
            s.free = std::make_unique<Event>();
            device_bytes_ += layer_bytes_;
        }
        const auto pin_t0 = std::chrono::steady_clock::now();
        const auto pin_deadline = host_pin_deadline(host_pin_wait_s);
        for (std::uint32_t l = first; l < end; ++l) {
            if (tiers_[l] != ExpertTier::host) continue;
            host_pin_stats_.requested_bytes += extents_[l].total();
            HostPinReason reason = HostPinReason::alloc;
            if (host_alias_ != 0U && host_layers_ >= host_alias_) {
                // timing only: this layer's copy reads another host layer's bytes (wrong weights, real transfers)
                host_view_[l] = host_view_[alias_of_[host_layers_ % host_alias_]];
            } else {
                host_[l] = std::make_unique<PinnedBuffer>();
                if ((implicit || (host_commit_floor_gib > 0.0 && host_alias_ == 0U))) {
                    if (!pin_with_floor(*host_[l], extents_[l].total(), host_commit_floor_gib, pin_deadline, host_pin_waits_, implicit, host_pin_wait_s == 0.0, reason)) {
                        // no room above the floor: this layer streams from the file (same bytes, slower)
                        host_[l].reset();
                        tiers_[l] = ExpertTier::file;
                        host_pin_stats_.demotions.push_back({l, reason});
                        ++host_pin_failed_;
                        continue;
                    }
                    double fc = 0.0, fp = 0.0;
                    system_room(fc, fp);
                    std::fprintf(stderr, "host pin layer %u: pinned %.2f GB, free commit %.2f GiB, free phys %.2f GiB, %.1f s\n", l,
                                 double(pinned_bytes_ + extents_[l].total()) / 1e9, fc / double(1ULL << 30), fp / double(1ULL << 30), seconds_since(pin_t0));
                } else {
                    host_[l]->resize(extents_[l].total());
                }
                read_layer_into(l, host_[l]->get());
                pinned_bytes_ += host_[l]->size();
                host_view_[l] = host_[l]->get();
                alias_of_.push_back(l);
            }
            host_pin_stats_.layers.push_back(l);
            host_pin_stats_.effective_bytes += extents_[l].total();
            ++host_layers_;
        }
        host_pin_s_ = seconds_since(pin_t0);
        if (!slots_.empty()) worker_ = std::thread([this] { loop(); });
    }
    ~ExpertProvider() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        if (worker_.joinable()) worker_.join();
        cudaStreamSynchronize(copy_);
        cudaStreamDestroy(copy_);
    }

    // Starts a run: the streamed layers are loaded in order, use j into slot j % S (the first S may already be there).
    void begin_run() { begin_run_uses(streamed_); }
    void begin_run_uses(std::span<const std::uint32_t> layers) {
        clear_decode();
        std::lock_guard<std::mutex> lock(mutex_);
        idle_ready_ = false;
        uses_.clear();
        for (const auto layer : layers) {
            require(layer >= first_ && layer < end_, "scheduled expert layer outside range");
            if (tiers_[layer] != ExpertTier::device) uses_.push_back(layer);
        }
        acquired_.assign(end_, ~std::size_t{0});
        if (streamed_.empty()) return;
        run_ = true;
        ++run_id_;
        next_use_ = 0;
        bytes_streamed_ = 0;
        file_bytes_streamed_ = 0;
        io_seconds_ = 0.0;
        wait_seconds_ = 0.0;
        wake_.notify_all();
    }
    // The experts of `layer` for kernels on `stream` (which waits for its slot's copy); call release() after the
    // layer's last kernel was enqueued.
    ExpertWeights acquire(std::uint32_t layer, cudaStream_t stream) {
        const std::uint8_t* base = nullptr;
        if (tiers_[layer] == ExpertTier::device) {
            base = resident_[layer]->get();
        } else if (decode_view_[layer] != nullptr) {  // decode residency (between runs, steps only)
            base = decode_view_[layer];
        } else {
            if (slots_.empty()) throw std::runtime_error("ds41 cuda prefill: a streamed expert layer without slots (options.slots 0)");
            std::size_t use = 0;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                require(next_use_ < uses_.size() && uses_[next_use_] == layer, "expert acquire differs from request schedule");
                use = acquired_[layer] = next_use_++;
            }
            auto& s = slots_[use % slots_.size()];
            const auto t0 = std::chrono::steady_clock::now();
            std::unique_lock<std::mutex> lock(mutex_);
            done_.wait(lock, [&] { return (s.loaded_run == run_id_ && s.loaded_use == use) || !error_.empty(); });
            if (!error_.empty()) throw std::runtime_error(error_);
            wait_seconds_ += seconds_since(t0);
            STRIX_CUDA_CHECK(cudaStreamWaitEvent(stream, s.ready->e, 0));
            base = s.buf->get();
        }
        const auto& x = extents_[layer];
        return {base, base + x.bytes[0], base + x.bytes[0] + x.bytes[1]};
    }
    void release(std::uint32_t layer, cudaStream_t stream) {
        if (tiers_[layer] == ExpertTier::device) return;
        const auto use = use_of(layer);
        auto& s = slots_[use % slots_.size()];
        STRIX_CUDA_CHECK(cudaEventRecord(s.free->e, stream));
        {
            std::lock_guard<std::mutex> lock(mutex_);
            s.freed_use = static_cast<std::int64_t>(use);
            s.freed_run = run_id_;
        }
        wake_.notify_all();
    }
    // After the run: the next run's first layers are loaded while idle.
    void end_run() {
        std::lock_guard<std::mutex> lock(mutex_);
        run_ = false;
        idle_ready_ = false;
        wake_.notify_all();
    }

    // ---- decode residency (CudaPrefill::decode_residency): after a run, every non-resident expert layer of [first,
    // stop) becomes usable by step(): a layer the idle preload left in a slot is used in place (the next run finds it
    // there and does not copy it again), any other is uploaded into its own device buffer (from its pinned copy, or the
    // file). `stream` waits for the uploads. Undone by begin_run() (or clear_decode()).
    void make_decode_resident(std::uint32_t first, std::uint32_t stop, cudaStream_t stream, std::uint32_t& adopted, std::uint32_t& promoted,
                              std::uint64_t& promoted_bytes) {
        adopted = promoted = 0;
        promoted_bytes = 0;
        if (!slots_.empty()) {  // the idle preload of the slots (once after end_run) must be done
            std::unique_lock<std::mutex> lock(mutex_);
            done_.wait(lock, [&] { return idle_ready_ || !error_.empty(); });
            if (!error_.empty()) throw std::runtime_error(error_);
        }
        for (std::uint32_t l = std::max(first, first_); l < std::min(stop, end_); ++l) {
            if (tiers_[l] == ExpertTier::device || decode_view_[l] != nullptr) continue;
            for (const auto& s : slots_)
                if (s.layer == static_cast<std::int32_t>(l)) decode_view_[l] = s.buf->get();
            if (decode_view_[l] != nullptr) {
                ++adopted;
                continue;
            }
            const auto& x = extents_[l];
            promoted_[l] = std::make_unique<DeviceBuffer<std::uint8_t>>(x.total());
            std::uint8_t* dst = promoted_[l]->get();
            std::uint64_t at = 0;
            for (int k = 0; k < 3; ++k) {
                if (tiers_[l] == ExpertTier::host)
                    STRIX_CUDA_CHECK(cudaMemcpyAsync(dst + at, host_view_[l] + at, x.bytes[k], cudaMemcpyHostToDevice, copy_));
                else
                    copy_file_to_device(x.offset[k], x.bytes[k], dst + at);
                at += x.bytes[k];
            }
            decode_view_[l] = dst;
            promoted_bytes += x.total();
            ++promoted;
        }
        if (!decode_event_) decode_event_ = std::make_unique<Event>();
        STRIX_CUDA_CHECK(cudaEventRecord(decode_event_->e, copy_));
        STRIX_CUDA_CHECK(cudaStreamWaitEvent(stream, decode_event_->e, 0));
    }
    void clear_decode() {
        bool any = false;
        for (const auto* v : decode_view_) any = any || v != nullptr;
        if (!any) return;
        STRIX_CUDA_CHECK(cudaDeviceSynchronize());  // steps that read the promoted buffers are done
        std::fill(decode_view_.begin(), decode_view_.end(), nullptr);
        for (auto& p : promoted_) p.reset();
    }
    [[nodiscard]] bool decode_resident(std::uint32_t layer) const { return tiers_[layer] == ExpertTier::device || decode_view_[layer] != nullptr; }
    [[nodiscard]] const std::vector<ExpertTier>& tiers() const { return tiers_; }
    [[nodiscard]] const HostPinStats& host_pin_stats() const { return host_pin_stats_; }
    [[nodiscard]] std::uint32_t host_pin_failed_layers() const { return host_pin_failed_; }
    [[nodiscard]] std::uint32_t host_pin_waits() const { return host_pin_waits_; }
    [[nodiscard]] double host_pin_seconds() const { return host_pin_s_; }

    [[nodiscard]] std::uint64_t bytes_streamed() const { return bytes_streamed_; }
    [[nodiscard]] std::uint64_t file_bytes_streamed() const { return file_bytes_streamed_; }
    [[nodiscard]] double io_seconds() const { return io_seconds_; }
    [[nodiscard]] double wait_seconds() const { return wait_seconds_; }
    [[nodiscard]] std::size_t device_bytes() const { return device_bytes_; }
    [[nodiscard]] std::size_t resident_bytes() const { return resident_bytes_; }  // resident layers (device_bytes - slots)
    [[nodiscard]] std::size_t pinned_bytes() const { return pinned_bytes_; }
    [[nodiscard]] std::size_t slot_count() const { return slots_.size(); }
    [[nodiscard]] std::uint64_t layer_bytes() const { return layer_bytes_; }

private:
    static constexpr std::size_t kStaging = 4, kStagingBytes = std::size_t{64} << 20;
    struct Slot {
        std::unique_ptr<DeviceBuffer<std::uint8_t>> buf;
        std::unique_ptr<Event> ready, free;
        std::uint64_t loaded_run = 0;   // run id and use index whose layer is (being) in the slot; ready once posted
        std::size_t loaded_use = ~std::size_t{0};
        std::int64_t freed_use = -1;    // last use whose kernels recorded `free`
        std::uint64_t freed_run = 0;
        std::int32_t layer = -1;        // layer whose bytes the slot holds (survives across runs)
    };
    struct Staging {
        std::unique_ptr<PinnedBuffer> buf;
        std::unique_ptr<Event> done;
        bool used = false;
    };

    [[nodiscard]] std::size_t use_of(std::uint32_t layer) const {
        require(layer < acquired_.size() && acquired_[layer] != ~std::size_t{0}, "expert layer has no acquired use");
        return acquired_[layer];
    }

    // file -> pinned staging -> device (the copy stream), aligned superset pieces; blocks until enqueued.
    void copy_file_to_device(std::uint64_t offset, std::uint64_t bytes, std::uint8_t* dst) {
        if (staging_.empty()) throw std::logic_error("ds41 cuda prefill: file -> device copy without staging (no device tier and no slots)");
        const std::uint64_t A = FileReader::alignment;
        std::uint64_t done = 0;
        while (done < bytes) {
            auto& st = staging_[next_staging_++ % staging_.size()];
            if (st.used) STRIX_CUDA_CHECK(cudaEventSynchronize(st.done->e));
            const std::uint64_t at = offset + done, a0 = at / A * A;
            const std::uint64_t n = std::min<std::uint64_t>(kStagingBytes, bytes - done);
            const std::uint64_t a1 = std::min<std::uint64_t>((at + n + A - 1U) / A * A, file_.size());
            const auto t0 = std::chrono::steady_clock::now();
            file_.read_range(a0, static_cast<std::size_t>(a1 - a0), st.buf->get(), std::size_t{8} << 20);
            io_seconds_ += seconds_since(t0);
            STRIX_CUDA_CHECK(cudaMemcpyAsync(dst + done, st.buf->get() + (at - a0), n, cudaMemcpyHostToDevice, copy_));
            STRIX_CUDA_CHECK(cudaEventRecord(st.done->e, copy_));
            st.used = true;
            done += n;
        }
    }
    void read_layer_into(std::uint32_t l, std::uint8_t* host) {
        // aligned bounce through a 64 MB pinned page buffer (host tier is filled once, at load)
        const std::uint64_t A = FileReader::alignment;
        PinnedBuffer page(kStagingBytes + 2U * A);
        std::uint64_t at_dst = 0;
        for (int k = 0; k < 3; ++k) {
            const auto& x = extents_[l];
            for (std::uint64_t done = 0; done < x.bytes[k];) {
                const std::uint64_t at = x.offset[k] + done, a0 = at / A * A;
                const std::uint64_t n = std::min<std::uint64_t>(kStagingBytes, x.bytes[k] - done);
                const std::uint64_t a1 = std::min<std::uint64_t>((at + n + A - 1U) / A * A, file_.size());
                file_.read_range(a0, static_cast<std::size_t>(a1 - a0), page.get());
                std::memcpy(host + at_dst, page.get() + (at - a0), n);
                at_dst += n;
                done += n;
            }
        }
    }
    void upload_layer(std::uint32_t l, std::uint8_t* dev) {
        std::uint64_t at = 0;
        for (int k = 0; k < 3; ++k) {
            copy_file_to_device(extents_[l].offset[k], extents_[l].bytes[k], dev + at);
            at += extents_[l].bytes[k];
        }
        STRIX_CUDA_CHECK(cudaStreamSynchronize(copy_));
    }

    // Loader thread: for every use j of the run, once slot j % S was freed by use j - S (or holds the layer already),
    // enqueue the layer's copy and record `ready`.
    void loop() {
        std::uint64_t run_seen = 0;
        std::size_t use = 0;
        bool idle_prefetch_done = false;
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [&] {
                if (stop_) return true;
                if (run_ && run_id_ != run_seen) return true;
                if (run_ && use < uses_.size()) {
                    const auto& s = slots_[use % slots_.size()];
                    return use < slots_.size() || (s.freed_run == run_id_ && s.freed_use == static_cast<std::int64_t>(use - slots_.size()));
                }
                return !run_ && !idle_prefetch_done;
            });
            if (stop_) return;
            if (run_ && run_id_ != run_seen) {
                run_seen = run_id_;
                use = 0;
                continue;
            }
            if (!run_) {
                // idle: preload the first S streamed layers for the next run (the previous run's kernels finished)
                idle_prefetch_done = true;
                lock.unlock();
                try {
                    STRIX_CUDA_CHECK(cudaDeviceSynchronize());
                    for (std::size_t j = 0; j < slots_.size(); ++j) {
                        auto& s = slots_[j];
                        if (s.layer == static_cast<std::int32_t>(streamed_[j])) continue;
                        load_into(s, streamed_[j]);
                        STRIX_CUDA_CHECK(cudaStreamSynchronize(copy_));
                    }
                    std::lock_guard<std::mutex> g(mutex_);
                    if (!run_) idle_ready_ = true;  // decode residency may adopt the slots now
                    done_.notify_all();
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> g(mutex_);
                    error_ = e.what();
                    done_.notify_all();
                }
                continue;
            }
            idle_prefetch_done = false;
            const std::size_t j = use++;
            auto& s = slots_[j % slots_.size()];
            const std::uint32_t layer = uses_[j];
            const bool wait_free = j >= slots_.size();
            lock.unlock();
            try {
                if (wait_free) STRIX_CUDA_CHECK(cudaStreamWaitEvent(copy_, s.free->e, 0));
                if (s.layer != static_cast<std::int32_t>(layer)) load_into(s, layer);
                STRIX_CUDA_CHECK(cudaEventRecord(s.ready->e, copy_));
                std::lock_guard<std::mutex> g(mutex_);
                s.loaded_run = run_seen;
                s.loaded_use = j;
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> g(mutex_);
                error_ = e.what();
            }
            done_.notify_all();
        }
    }
    void load_into(Slot& s, std::uint32_t layer) {
        s.layer = -1;
        const auto& x = extents_[layer];
        std::uint64_t at = 0;
        for (int k = 0; k < 3; ++k) {
            if (tiers_[layer] == ExpertTier::host)
                STRIX_CUDA_CHECK(cudaMemcpyAsync(s.buf->get() + at, host_view_[layer] + at, x.bytes[k], cudaMemcpyHostToDevice, copy_));
            else
                copy_file_to_device(x.offset[k], x.bytes[k], s.buf->get() + at);
            at += x.bytes[k];
        }
        bytes_streamed_ += x.total();
        if (tiers_[layer] == ExpertTier::file) file_bytes_streamed_ += x.total();
        s.layer = static_cast<std::int32_t>(layer);
    }

    FileReader& file_;
    std::vector<ExpertExtent> extents_;
    std::vector<ExpertTier> tiers_;
    std::uint32_t first_ = 0, end_ = 0;
    std::uint64_t layer_bytes_ = 0;
    std::vector<std::unique_ptr<DeviceBuffer<std::uint8_t>>> resident_;
    std::vector<std::unique_ptr<PinnedBuffer>> host_;
    std::vector<const std::uint8_t*> host_view_;
    std::vector<std::uint32_t> alias_of_;  // layers with their own pinned copy
    std::uint32_t host_alias_ = 0, host_layers_ = 0;
    std::vector<std::uint32_t> uses_;
    std::vector<std::size_t> acquired_;
    std::vector<std::uint32_t> streamed_;
    std::deque<Slot> slots_;
    std::vector<Staging> staging_;
    std::size_t next_staging_ = 0;
    cudaStream_t copy_ = nullptr;
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable wake_, done_;
    bool stop_ = false, run_ = false;
    std::uint64_t run_id_ = 0;
    std::size_t next_use_ = 0;
    std::string error_;
    std::atomic<std::uint64_t> bytes_streamed_{0}, file_bytes_streamed_{0};
    double io_seconds_ = 0.0, wait_seconds_ = 0.0;
    std::size_t device_bytes_ = 0, pinned_bytes_ = 0, resident_bytes_ = 0;
    // decode residency
    bool idle_ready_ = false;                    // the idle preload after the last run (or the load) is done
    std::vector<const std::uint8_t*> decode_view_;
    std::vector<std::unique_ptr<DeviceBuffer<std::uint8_t>>> promoted_;
    std::unique_ptr<Event> decode_event_;
    HostPinStats host_pin_stats_;
    std::uint32_t host_pin_failed_ = 0, host_pin_waits_ = 0;
    double host_pin_s_ = 0.0;
};

// ------------------------------------------------------------------------------------------------ small kernels
// dst[r][0..4) = src[r][0..4) with src stride 24 (the pre of an hc mix row -> a pre_carry row)
// Boundary KV entries of a step: for every listed source, entries [pos[0] / ratio, + n) of its KV and index-key caches
// into the staging buffer (KV then keys per source; n = the most a step of the call's rows can complete).
struct SectionKvJob {
    const std::uint8_t* kv;
    const std::uint8_t* idx;
    std::uint32_t ratio, n;
    std::uint8_t* out;
};
struct SectionKvJobs {
    SectionKvJob job[8];
    std::uint32_t count;
    const std::uint64_t* pos;
};
__global__ void section_kv_kernel(SectionKvJobs jobs) {
    const SectionKvJob& j = jobs.job[blockIdx.x];
    const std::uint64_t first = jobs.pos[0] / j.ratio;
    const std::size_t kvb = std::size_t{j.n} * 288U, ib = std::size_t{j.n} * 68U;
    const std::uint8_t* kv = j.kv + first * 288U;
    const std::uint8_t* ix = j.idx + first * 68U;
    for (std::size_t i = threadIdx.x; i < kvb; i += blockDim.x) j.out[i] = kv[i];
    for (std::size_t i = threadIdx.x; i < ib; i += blockDim.x) j.out[kvb + i] = ix[i];
}

// Device -> mapped pinned host memory by the kernel's own stores (no copy-engine transfer to wait for): 16-byte words,
// then the tail bytes. Used for a step's outgoing boundary (stream, mix, sections) and its greedy candidates.
__global__ void store_mapped_kernel(std::uint8_t* __restrict__ dst, const std::uint8_t* __restrict__ src, std::size_t bytes) {
    const std::size_t words = bytes / 16U, stride = std::size_t{gridDim.x} * blockDim.x;
    for (std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x; i < words; i += stride)
        reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(src)[i];
    for (std::size_t i = words * 16U + std::size_t{blockIdx.x} * blockDim.x + threadIdx.x; i < bytes; i += stride) dst[i] = src[i];
}
void store_mapped(void* host_mapped, const void* src, std::size_t bytes, cudaStream_t st) {
    if (bytes == 0U) return;
    void* d = nullptr;
    STRIX_CUDA_CHECK(cudaHostGetDevicePointer(&d, host_mapped, 0));
    const unsigned blocks = static_cast<unsigned>(std::min<std::size_t>((bytes / 16U + 255U) / 256U + 1U, 64U));
    store_mapped_kernel<<<blocks, 256, 0, st>>>(static_cast<std::uint8_t*>(d), static_cast<const std::uint8_t*>(src), bytes);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

__global__ void copy_pre_kernel(const float* __restrict__ mix, float* __restrict__ pre, std::uint32_t rows) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * 4U) pre[i] = mix[std::size_t{i / 4U} * 24U + i % 4U];
}

// Drafter input of a step's rows (CudaStepIo::draft_hidden): out[r * out_stride + c] = (((s0 + s1) + s2) + s3) * 0.25 of
// the layer's input stream row r (copies s0..s3 of [4][hidden]), the HIP target's main_hidden_kernel formula.
__global__ void draft_hidden_kernel(const float* __restrict__ stream, std::uint32_t hidden, float* __restrict__ out, std::uint32_t out_stride) {
    const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (c >= hidden) return;
    const float* s = stream + std::size_t{r} * 4U * hidden;
    out[std::size_t{r} * out_stride + c] = (((s[c] + s[hidden + c]) + s[2U * hidden + c]) + s[3U * hidden + c]) * 0.25F;
}

}  // namespace

// ================================================================================================ Impl
struct CudaPrefill::Impl {
    std::shared_ptr<attn::ExportCapture> export_capture;
    CudaPrefillOptions o;
    deepseek41::Config cfg;
    LayerCoreGeometry g;
    std::unique_ptr<TensorFile> file;
    std::filesystem::path gguf_path;
    // STRIX_DS41_CUDA_ENGRAM_CHECK=1 at construction, set_engram_check() later: re-read every Engram job (diagnostics)
    bool engram_check = [] {
        const char* v = std::getenv("STRIX_DS41_CUDA_ENGRAM_CHECK");
        return v && v[0] == '1';
    }();
    std::filesystem::path file_path;
    std::unique_ptr<FileReader> reader;         // routed-expert reads (the provider's loader thread)
    std::unique_ptr<FileReader> engram_reader;  // Engram rows (their own thread: a FileReader serves one caller at a time)
    std::unique_ptr<LayerCore> core;
    std::vector<ds41::attn::LayerAttention> plan;
    std::vector<at::LayerWeights> attn_w;
    std::unique_ptr<ExpertProvider> experts;
    std::unique_ptr<detail::ExpertStreamer> streamer;  // instead of `experts` with the transport / residency options
    DeviceBuffer<std::uint32_t> mapped_ids;            // [moe chunk][top_k] ids mapped to virtual positions
    GpuProfile gpu;
    double dense_gib = 0.0, used_before_experts_gib = 0.0, free_at_plan_gib = 0.0, expert_budget_gib = 0.0;
    std::vector<std::uint32_t> resident_experts;       // [layer] (streamer)
    std::string residency_plan;                        // how the per-expert plan was found (rows / even / even-1slot)
    std::vector<ExpertTier> tiers;
    std::unique_ptr<deepseek41::TargetModel> model;
    std::vector<EngramTable> engram_tables;
    std::unique_ptr<EngramHasher> hasher;
    MoeShape moe_shape;
    at::DecodeAllocationOptions decode_alloc_options;
    std::uint32_t split = 192;
    cudaStream_t st = nullptr;
    double load_seconds = 0.0;
    // device accounting (DeviceBuffer byte deltas around each load): dense weights (LayerCore tensor-core forms,
    // embedding / head, attention weights), row and chunk buffers, attention state (rings, caches, RoPE tables)
    std::size_t dense_bytes = 0, buffer_bytes = 0, state_bytes = 0;
    std::size_t device_in_use_at_load = 0;  // cudaMemGetInfo after the load (every process on the device)
    DeviceInfo dev;
    // decode residency (decode_residency()): the run's tiers, whether the run-only buffers are released, last stats
    std::vector<ExpertTier> run_tiers;
    bool decode_mode = false, run_buffers_released = false;
    std::size_t run_buffer_bytes = 0;  // the run-only row / chunk buffers (released between a run and the next)
    std::string last_residency;        // JSON of the last decode_residency()
    // CUDA graphs of the layer segments (options.graphs): A and B per layer, captured for one T
    struct Graphs {
        std::uint32_t T = 0;
        const void* engram = nullptr;
        std::vector<cudaGraphExec_t> a, b;
        std::vector<ExpertWeights> b_weights;
        void clear() {
            for (auto& e : a) if (e) cudaGraphExecDestroy(e);
            for (auto& e : b) if (e) cudaGraphExecDestroy(e);
            a.clear();
            b.clear();
            b_weights.clear();
            T = 0;
        }
        ~Graphs() { clear(); }
    } graphs;
    std::uint32_t runs_done = 0;
    bool idle_held = false;  // defer_idle_preload: the streamer is still in its run state after run()
    // the decode sequence is context 0 (seqs[0]: run(), step(), import_state(), export_state() / capture_state())
    std::uint32_t last_row = 0;  // row of context 0's index sources' last selection (export_state)
    std::uint32_t last_step_rows = 0;  // rows of the last step not yet committed (commit_window), 0: none
    bool fresh = false;                // reset_sequence(): steps may start at position 0
    // sequence slots (set_sequences / select_sequence): the per-sequence part of the object (attention state, token
    // history, Engram n-gram state) of every slot but the selected one, which lives in the fields above
    struct Sequence {
        std::unique_ptr<at::AttentionState> attn;
        std::unique_ptr<EngramHasher> hasher;
        std::uint32_t last_T = 0, last_next = 0, last_row = 0, last_step_rows = 0;
        std::vector<std::uint32_t> last_tokens;
        bool fresh = false;
    };
    std::vector<Sequence> sequences;  // [slot]; the selected slot's entry is empty (moved into the fields above)
    std::uint32_t sequence = 0;       // the selected slot
    at::LayerAttentionOptions attn_options;
    void swap_sequence(Sequence& q) {
        // public multi-seq tree: the attention state and the token history live in seq0() (context 0)
        std::swap(seq0().attn, q.attn);
        std::swap(hasher, q.hasher);
        std::swap(seq0().last_T, q.last_T);
        std::swap(seq0().last_next, q.last_next);
        std::swap(last_row, q.last_row);
        std::swap(last_step_rows, q.last_step_rows);
        std::swap(seq0().last_tokens, q.last_tokens);
        std::swap(fresh, q.fresh);
    }
    std::vector<ExpertExtent> extents;

    // ---- incremental decode (step)
    struct Decode;
    std::unique_ptr<Decode> dec;

    // Per-sequence context: every row buffer whose values live across ops or layers, the attention state and the
    // Engram rows of one prompt. seqs[0] holds max_positions rows (run()); seqs[1..] the batch contexts
    // (options.batch_positions), each sized to its own capacity. Weights, chunk / window scratch and workspaces are
    // shared: a closed batch runs every sequence of a layer one after the other on the one stream.
    struct Seq {
        std::uint32_t capacity = 0;
        DeviceBuffer<float> stream, mix_attn, mix_ffn, x, qkv, kvrows, qr, y, shared, route_w;
        DeviceBuffer<std::uint32_t> ids;
        A8Buffer xa, qa;
        std::unique_ptr<at::AttentionState> attn;
        PinnedBuffer engram_host;  // [engram layer][row][col][264] of the run (mapped)
        std::size_t row_bytes = 0;  // device bytes of the row buffers above
        std::uint32_t last_T = 0, last_rows = 0;
        std::vector<std::uint32_t> last_tokens;
        std::uint32_t last_next = 0;
    };
    std::vector<std::unique_ptr<Seq>> seqs;
    std::vector<std::size_t> last_batch_ctx;  // context of each item of the last run / run_batch
    std::atomic<bool> cancel_requested{false};
    std::atomic<std::uint32_t> progress_position{0}, progress_total{0};
    std::atomic<std::int64_t> request_start_ns{0};
    std::shared_ptr<CudaPrefillStateStream> state_stream;
    bool long_request = false;
    std::uint32_t pass_base = 0, request_total = 0, request_cached = 0;
    std::size_t pass_index = 0, engram_consumed = 0;
    ds41::SuffixReplayPlan request_replay;
    std::vector<ds41::PrefillPass> passes;
    std::vector<std::uint32_t> request_tokens;
    std::vector<std::uint64_t> request_ids;
    std::vector<std::uint32_t> engram_ready;
    PinnedBuffer engram_ahead;
    bool engram_cancelled = false;
    std::uint32_t row_capacity = 0;
    std::size_t batch_bytes = 0;              // device bytes of seqs[1..] (row buffers + attention state)
    void alloc_seq(Seq& q, std::uint32_t capacity, at::LayerAttentionOptions ao);
    [[nodiscard]] std::size_t engram_pinned_bytes() const {
        std::size_t n = 0;
        for (const auto& q : seqs) n += q->engram_host.size();
        return n + engram_ahead.size();
    }
    // context 0: the run() / decode sequence
    [[nodiscard]] Seq& seq0() { return *seqs[0]; }
    [[nodiscard]] const Seq& seq0() const { return *seqs[0]; }

    DeviceBuffer<std::uint64_t> pos64;
    // chunk buffers
    DeviceBuffer<float> q, o_, o_low, ckv, cscore, iq, iw, scores, partial, router_scores;
    A8Buffer oa, la;
    DeviceBuffer<std::uint8_t> qprep, erows, ws;
    DeviceBuffer<std::uint32_t> block_keys;
    DeviceBuffer<std::uint32_t> survivors;  // selection above attn::kSelDirect positions
    // fast attention (o.fast_attention): FP16 window / KV rows, FP8 index keys, chunk-wide index scores and selection
    bool fast_index = false;
    DeviceBuffer<std::uint16_t> win16, kv16;
    DeviceBuffer<std::uint8_t> k8, fqprep;
    DeviceBuffer<float> kscale, fscores;
    DeviceBuffer<std::uint32_t> fblock_keys, fsurvivors;
    // routed experts
    DeviceBuffer<float> raw;
    DeviceBuffer<float> raw1;  // expert TP (decode steps): rank 1's K-half parts
    DeviceBuffer<std::uint8_t> moe_ws;
#ifdef STRIX_DS41_CUDA_TRELLIS
    // trellis shared expert of decode steps (o.trellis_shexp): per layer of [first, end) gate (w1), up (w3), down (w2)
    struct TrellisLinear {
        DeviceBuffer<std::uint16_t> packed, suh, svh;
        trellis::Weights w;
        std::uint32_t splits = 4;
    };
    std::vector<std::array<TrellisLinear, 3>> tshexp;
    DeviceBuffer<float> tr_gate, tr_up, tr_act;
    DeviceBuffer<std::uint8_t> tr_ws;
    std::size_t tr_ws_bytes = 0;
    std::uint64_t trellis_bytes = 0;
    std::uint32_t trellis_bits = 0;
#endif
    // Engram rows of the run: in each Seq (pinned)
    std::thread engram_thread;
    std::exception_ptr engram_error;
    std::mutex engram_mutex;
    std::condition_variable engram_cv;
    std::uint32_t engram_layers_done = 0;  // Engram layers whose rows are in engram_host (in order)
    double engram_seconds = 0.0;
    // engram_upload == "stream": the rows copied to the device by a high-priority stream, an event per Engram layer
    cudaStream_t engram_st = nullptr;
    std::vector<cudaEvent_t> engram_ev;
    DeviceBuffer<std::uint8_t> engram_dev;

    [[nodiscard]] bool has_embed() const { return o.first_layer == 0U; }
    [[nodiscard]] bool has_head() const { return o.end_layer == cfg.num_hidden_layers; }
};

void CudaPrefill::Impl::alloc_seq(Seq& q, std::uint32_t capacity, at::LayerAttentionOptions ao) {
    (void)ao;
    const auto& c = cfg;
    const std::size_t T = capacity, H = c.hidden_size, QA = c.q_lora_rank, D = at::kHeadDim;
    const std::size_t b0 = device_buffer_bytes();
    q.capacity = capacity;
    q.stream.resize(T * 4U * H);
    q.mix_attn.resize(T * 24U);
    q.mix_ffn.resize(T * 24U);
    q.x.resize(T * H);
    q.qkv.resize(T * (QA + D));
    q.kvrows.resize(T * D);
    q.qr.resize(T * QA);
    q.y.resize(T * H);
    q.shared.resize(T * H);
    q.route_w.resize(T * c.num_experts_per_tok);
    q.ids.resize(T * c.num_experts_per_tok);
    q.xa.resize(static_cast<std::uint32_t>(T), c.hidden_size);
    q.qa.resize(static_cast<std::uint32_t>(T), static_cast<std::uint32_t>(QA));
    q.row_bytes = device_buffer_bytes() - b0;
    require(q.row_bytes == at::prefill_sequence_allocations(capacity, c.hidden_size, c.q_lora_rank,
                c.num_experts_per_tok).total(), "prefill sequence allocation ledger");
}

namespace {

// The decode's Engram row reads on their own thread: one job = the rows of every Engram layer of a step, read layer
// by layer (one overlapped read per record, all of a layer in flight at once); the step waits per layer. A job is
// issued at the start of a step or, for the next token, as soon as a head step knows its greedy token. With a cache
// (cache_records > 0) a record already read is copied from RAM (the file is immutable): only the misses are read.
class EngramIoThread {
public:
    explicit EngramIoThread(FileReader& reader, std::size_t cache_records = 0)
        : reader_(reader), cache_(cache_records), th_([this] { loop(); }) {}
    ~EngramIoThread() {
        {
            std::lock_guard<std::mutex> g(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        th_.join();
    }
    EngramIoThread(const EngramIoThread&) = delete;
    EngramIoThread& operator=(const EngramIoThread&) = delete;

    // offsets[i] of the records of layer i into outs[i] (the previous job must be complete: wait_idle)
    void submit(std::vector<std::vector<std::uint64_t>> offsets, std::vector<std::uint8_t*> outs) {
        std::lock_guard<std::mutex> g(mu_);
        if (busy_) throw std::logic_error("ds41 cuda decode: Engram job submitted while one is in flight");
        offsets_ = std::move(offsets);
        outs_ = std::move(outs);
        error_ = nullptr;
        layers_done_.store(0U, std::memory_order_release);
        busy_ = true;
        ++generation_;
        cv_.notify_all();
    }
    // the job's layer i is in (spin: poll, else block)
    void wait_layer(std::uint32_t i, bool spin) {
        if (spin)
            while (layers_done_.load(std::memory_order_acquire) <= i) std::this_thread::yield();
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return layers_done_.load(std::memory_order_acquire) > i; });
        if (error_) std::rethrow_exception(error_);
    }
    void wait_idle() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return !busy_; });
    }
    // wall time of the last complete job (its start to the last layer in) and the part spent issuing the reads
    [[nodiscard]] double last_seconds() {
        std::lock_guard<std::mutex> g(mu_);
        return last_seconds_;
    }
    [[nodiscard]] double last_issue_seconds() {
        std::lock_guard<std::mutex> g(mu_);
        return last_issue_seconds_;
    }

private:
    static constexpr std::size_t kRecord = 264U;
    // Engram records by file offset (clock replacement); used by the I/O thread only.
    struct RowCache {
        explicit RowCache(std::size_t n) : cap(n), rows(n * kRecord), key(n), ref(n, 0) { where.reserve(n); }
        std::size_t cap = 0, used = 0, hand = 0;
        std::vector<std::uint8_t> rows;
        std::vector<std::uint64_t> key;
        std::vector<std::uint8_t> ref;
        std::unordered_map<std::uint64_t, std::uint32_t> where;
        void insert(std::uint64_t off, const std::uint8_t* rec) {
            if (where.count(off)) return;
            std::size_t slot;
            if (used < cap) {
                slot = used++;
            } else {
                while (ref[hand]) {
                    ref[hand] = 0;
                    hand = (hand + 1U) % cap;
                }
                slot = hand;
                hand = (hand + 1U) % cap;
                where.erase(key[slot]);
            }
            std::memcpy(rows.data() + slot * kRecord, rec, kRecord);
            key[slot] = off;
            ref[slot] = 0;
            where.emplace(off, static_cast<std::uint32_t>(slot));
        }
    };
    // one layer's records: hits from the cache, the misses read in one queued batch (then cached)
    void read_layer(const std::vector<std::uint64_t>& offs, std::uint8_t* out) {
        if (cache_.cap == 0U) {
            reader_.read_records_queued(offs, kRecord, out);
            return;
        }
        miss_at_.clear();
        miss_off_.clear();
        for (std::size_t k = 0; k < offs.size(); ++k) {
            const auto it = cache_.where.find(offs[k]);
            if (it != cache_.where.end()) {
                std::memcpy(out + k * kRecord, cache_.rows.data() + std::size_t{it->second} * kRecord, kRecord);
                cache_.ref[it->second] = 1;
            } else {
                miss_at_.push_back(k);
                miss_off_.push_back(offs[k]);
            }
        }
        if (miss_off_.empty()) return;
        miss_rows_.resize(miss_off_.size() * kRecord);
        reader_.read_records_queued(miss_off_, kRecord, miss_rows_.data());
        for (std::size_t j = 0; j < miss_at_.size(); ++j) {
            std::memcpy(out + miss_at_[j] * kRecord, miss_rows_.data() + j * kRecord, kRecord);
            cache_.insert(miss_off_[j], miss_rows_.data() + j * kRecord);
        }
    }

    void loop() {
#ifdef _WIN32
        // the completions of the reads are delivered to this thread: it must not wait behind normal-priority work
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#endif
        std::uint64_t seen = 0;
        for (;;) {
            std::vector<std::vector<std::uint64_t>> offsets;
            std::vector<std::uint8_t*> outs;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
                offsets = offsets_;
                outs = outs_;
            }
            const auto t0 = std::chrono::steady_clock::now();
            const double issue0 = reader_.stats().issue_seconds;
            std::exception_ptr err;
            for (std::size_t i = 0; i < offsets.size(); ++i) {
                try {
                    if (!err && !offsets[i].empty()) read_layer(offsets[i], outs[i]);
                } catch (...) {
                    err = std::current_exception();
                }
                {
                    std::lock_guard<std::mutex> g(mu_);
                    if (err && !error_) error_ = err;
                    layers_done_.store(static_cast<std::uint32_t>(i + 1U), std::memory_order_release);
                }
                cv_.notify_all();
            }
            {
                std::lock_guard<std::mutex> g(mu_);
                last_seconds_ = seconds_since(t0);
                last_issue_seconds_ = reader_.stats().issue_seconds - issue0;
                layers_done_.store(~0U, std::memory_order_release);
                busy_ = false;
            }
            cv_.notify_all();
        }
    }

    FileReader& reader_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<std::vector<std::uint64_t>> offsets_;
    std::vector<std::uint8_t*> outs_;
    std::exception_ptr error_;
    std::atomic<std::uint32_t> layers_done_{0U};
    std::uint64_t generation_ = 0;
    bool busy_ = false, stop_ = false;
    double last_seconds_ = 0.0, last_issue_seconds_ = 0.0;
    RowCache cache_;
    std::vector<std::size_t> miss_at_;
    std::vector<std::uint64_t> miss_off_;
    std::vector<std::uint8_t> miss_rows_;
    std::thread th_;  // last: starts after the members above exist
};

// Expert ranking per layer, hottest first: a routes file (whitespace-separated "layer:id,id,..." per token) counts
// the experts each layer selected; a placement JSON (tensormeld/hot-expert-placement-v1) puts each layer's
// "rtx-pro" experts first. Ties and unlisted experts by ascending id. Empty path: ascending id.
std::vector<std::vector<std::uint32_t>> expert_ranking(const std::string& path, std::uint32_t layers, std::uint32_t experts) {
    std::vector<std::vector<double>> score(layers, std::vector<double>(experts, 0.0));
    if (!path.empty()) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("ds41 cuda prefill: cannot read the expert ranking " + path);
        std::stringstream ss;
        ss << in.rdbuf();
        const std::string t = ss.str();
        if (t.find("hot-expert-placement") != std::string::npos) {
            std::size_t at = t.find("\"layers\"");
            while (at != std::string::npos && (at = t.find("\"layer\"", at + 1)) != std::string::npos) {
                const std::uint32_t l = static_cast<std::uint32_t>(std::strtoul(t.c_str() + t.find(':', at) + 1, nullptr, 10));
                const std::size_t pro = t.find("\"rtx-pro\"", at), stop = t.find('}', at);
                if (pro == std::string::npos || pro > stop || l >= layers) continue;
                const std::size_t open = t.find('[', pro), close = t.find(']', pro);
                std::istringstream ids(t.substr(open + 1, close - open - 1));
                std::string tok;
                double w = 1e9;
                while (std::getline(ids, tok, ','))
                    if (!tok.empty() && tok.find_first_not_of(" \r\n\t") != std::string::npos) {
                        const auto e = static_cast<std::uint32_t>(std::stoul(tok));
                        if (e < experts) score[l][e] = w;
                        w -= 1.0;
                    }
            }
        } else {
            std::istringstream words(t);
            std::string w;
            while (words >> w) {
                const auto colon = w.find(':');
                if (colon == std::string::npos) continue;
                const auto l = static_cast<std::uint32_t>(std::strtoul(w.c_str(), nullptr, 10));
                if (l >= layers) continue;
                std::istringstream ids(w.substr(colon + 1));
                std::string tok;
                while (std::getline(ids, tok, ','))
                    if (!tok.empty()) {
                        const auto e = static_cast<std::uint32_t>(std::stoul(tok));
                        if (e < experts) score[l][e] += 1.0;
                    }
            }
        }
    }
    std::vector<std::vector<std::uint32_t>> rank(layers);
    for (std::uint32_t l = 0; l < layers; ++l) {
        rank[l].resize(experts);
        for (std::uint32_t e = 0; e < experts; ++e) rank[l][e] = e;
        std::stable_sort(rank[l].begin(), rank[l].end(), [&](std::uint32_t a, std::uint32_t b) { return score[l][a] > score[l][b]; });
    }
    return rank;
}

// Streamed experts per layer for a device budget of B experts (resident + S slots of the largest streamed part):
// layer l streams s_l = min(E, X * w_l / W) for the smallest total X that fits (w: the layer's rows).
std::vector<std::uint32_t> plan_streamed(std::uint64_t budget_experts, std::uint32_t slots, const std::vector<double>& w, std::uint32_t E) {
    const auto n = static_cast<std::uint32_t>(w.size());
    double W = 0.0;
    for (const double v : w) W += v;
    auto streamed = [&](std::uint64_t X) {
        std::vector<std::uint32_t> s(n);
        for (std::uint32_t l = 0; l < n; ++l) s[l] = static_cast<std::uint32_t>(std::min<double>(E, std::floor(double(X) * w[l] / W)));
        return s;
    };
    auto fits = [&](const std::vector<std::uint32_t>& s) {
        std::uint64_t res = 0;
        std::uint32_t mx = 0;
        for (const auto v : s) {
            res += E - v;
            mx = std::max(mx, v);
        }
        return res + std::uint64_t{mx == 0U ? 0U : std::min<std::uint32_t>(slots, n)} * mx <= budget_experts;
    };
    if (fits(std::vector<std::uint32_t>(n, 0U))) return std::vector<std::uint32_t>(n, 0U);
    std::uint64_t lo = 0, hi = std::uint64_t{n} * E * 4U;  // the weights' rounding may need X above n * E
    if (!fits(streamed(hi)))
        throw std::invalid_argument("ds41 cuda prefill: device expert budget of " + std::to_string(budget_experts) + " experts below " +
                                    std::to_string(std::min<std::uint32_t>(slots, n)) + " slots of a whole streamed layer (" +
                                    std::to_string(E) + " experts each)");
    while (lo + 1U < hi) {
        const std::uint64_t mid = (lo + hi) / 2U;
        (fits(streamed(mid)) ? hi : lo) = mid;
    }
    return streamed(hi);
}

}  // namespace

struct CudaPrefill::Impl::Decode {
    static constexpr std::size_t kParamsPos = 0, kParamsTokens = 64, kParamsCand = 128, kParamsBytes = 256;
    // Engram reads (o.engram == "file"); a prefetched job for the token `pf_token` at position `pf_position`
    std::unique_ptr<EngramIoThread> engram_io;
    // STRIX_DS41_CUDA_ENGRAM_CHECK=1 (diagnostics): every job's records read again through a plain buffered stream once
    // the step has consumed them, compared byte for byte (CudaStepResult::engram_checked / engram_check_mismatches)
    std::vector<std::vector<std::uint64_t>> check_offsets;
    std::vector<std::uint8_t*> check_outs;
    std::unique_ptr<std::ifstream> check_file;
    bool pf_pending = false;
    std::uint32_t pf_position = 0, pf_token = 0;
    void drain_engram() {
        if (engram_io) engram_io->wait_idle();
        pf_pending = false;
        check_offsets.clear();  // a dropped job's records are never re-read against the next job's rows
        check_outs.clear();
    }
    PinnedBuffer params;         // pos uint64[8], tokens uint32[8], candidates ArgmaxCandidate[8]
    PinnedBuffer engram;         // [engram layer][row][col][264]
    PinnedBuffer io_in, io_out;  // stream R x 4H then mix R x 24
    PinnedBuffer logits;         // R x vocab
    DeviceBuffer<std::uint64_t> pos;
    DeviceBuffer<float> kvrows;
    DeviceBuffer<std::uint32_t> router_counters;  // router v2 arrival counters: [0] the step's router, [1] route_other
    // decode_shared_fork: the shared expert's side branch and its per-layer fork / join events
    std::unique_ptr<Stream> shared_side;
    std::vector<std::unique_ptr<Event>> shared_fork, shared_join;
    std::vector<std::unique_ptr<Event>> attn_fork, attn_join;  // decode_attn_fork (on shared_side)
    // sparse experts (non-resident layers in eager steps)
    std::unique_ptr<FileReader> reader;
    std::unique_ptr<DeviceBuffer<std::uint8_t>> slot;
    std::unique_ptr<PinnedBuffer> staging;
    // graphs: per shape, two segments and the timing events they record
    struct Key {
        std::uint32_t rows, bucket, stop;
        bool stream_in, stream_out, timing;
        std::uint32_t sections_cut;
        std::uint32_t sequence;  // the slot: a captured graph reads that slot's attention state
        bool moe_site = false;   // the MoE sites are the caller's (captured with them)
        std::uint32_t moe_epoch = 0;  // CudaStepIo::moe_site_epoch
        const float* draft_hidden = nullptr;  // CudaStepIo::draft_hidden (the captured kernel writes there)
        std::uint32_t draft_first = 0;
        const float* draft_raw = nullptr;     // CudaStepIo::draft_hidden_raw
        bool operator<(const Key& k) const {
            return std::tie(rows, bucket, stop, stream_in, stream_out, timing, sections_cut, sequence, moe_site, moe_epoch, draft_hidden, draft_first, draft_raw) <
                   std::tie(k.rows, k.bucket, k.stop, k.stream_in, k.stream_out, k.timing, k.sections_cut, k.sequence, k.moe_site, k.moe_epoch,
                            k.draft_hidden, k.draft_first, k.draft_raw);
        }
    };
    struct Captured {
        std::vector<cudaGraphExec_t> exec;              // one per segment
        std::vector<std::unique_ptr<Event>> events;     // recorded by the graphs (per-layer marks)
        std::vector<std::array<cudaEvent_t, 4>> marks;  // per layer of the step: begin, attn_end, moe_begin, end
        std::uint32_t eager_steps = 0;
        Captured() = default;
        Captured(const Captured&) = delete;
        Captured& operator=(const Captured&) = delete;
        ~Captured() {
            for (auto& e : exec)
                if (e) cudaGraphExecDestroy(e);
        }
    };
    std::map<Key, std::unique_ptr<Captured>> graphs;
    std::unique_ptr<Event> start, done;
    // boundary sections (io.sections_cut): staging on the device, the pinned copy the step's last segment fills
    DeviceBuffer<std::uint8_t> sec_dev;
    PinnedBuffer sec_host;
    Decode() {
        params.resize(kParamsBytes, true);
        std::memset(params.get(), 0, kParamsBytes);
        const auto a = at::decode_allocations({});
        pos.resize(a.bytes("pos") / sizeof(std::uint64_t));
        kvrows.resize(a.bytes("kvrows") / sizeof(float));
        start = std::make_unique<Event>(true);
        done = std::make_unique<Event>(true);
    }
};

namespace {

// The row and chunk buffers only run() reads (max_positions rows / chunk rows); step() uses the layer core's own
// buffers, raw / raw1 / moe_ws and the attention state, so decode_residency() may release these between runs.
void alloc_run_buffers(CudaPrefill::Impl& m) {
    const auto& o = m.o;
    const auto& c = m.cfg;
    const std::size_t T = o.max_positions, C = o.chunk_rows, W = o.window_rows;
    const std::uint32_t QH = c.num_attention_heads * c.head_dim, OL = c.o_groups * c.o_lora_rank, D = at::kHeadDim;
    // bounded passes (pass_rows): the fast index scores query tiles of at most 256 rows
    const std::uint32_t IC = o.pass_rows ? std::min<std::uint32_t>(256U, o.chunk_rows) : o.chunk_rows;
    const std::size_t before = device_buffer_bytes();
    // context 0's row buffers (row_capacity: max_positions, or one bounded pass), its positions max_positions; the
    // batch contexts stay allocated
    // decode_only (the single-GPU replica): step() uses its own decode buffers, so none of run()'s row /
    // chunk / window buffers are allocated (they would take the arena's headroom); the fast-attention buffers stay
    m.seq0().capacity = o.max_positions;
    if (!o.decode_only) {
    m.alloc_seq(m.seq0(), m.row_capacity, {});
    {
        std::vector<std::uint64_t> pos(T + 1U);
        for (std::size_t i = 0; i <= T; ++i) pos[i] = i;
        m.pos64.resize(T + 1U);
        m.pos64.upload(pos);
    }
    m.q.resize(C * QH);
    m.o_.resize(C * QH);
    m.o_low.resize(C * OL);
    m.oa.resize(static_cast<std::uint32_t>(C), QH);
    m.la.resize(static_cast<std::uint32_t>(C), OL);
    m.ckv.resize(C * D);
    m.cscore.resize(C * D);
    m.iq.resize(C * at::kIdxHeads * at::kIdxDim);
    m.iw.resize(C * at::kIdxHeads);
    m.scores.resize(W * (T + 1U));
    m.partial.resize(W * at::kHeads * at::kPartialFloats);
    m.qprep.resize(W * at::kIdxPrepBytes);
    m.block_keys.resize(W * m.seq0().attn->cand_stride());
    m.survivors.resize(std::max<std::size_t>(1, at::select_survivor_words(static_cast<std::uint32_t>(W), static_cast<std::uint32_t>(T + 1U), at::kTopK)));
    m.router_scores.resize(C * c.n_routed_experts);
    }
    if (o.fast_attention) {
        m.fast_index = o.fast_index && at::fast_index_supported(o.device);
        m.win16.resize((std::size_t{m.row_capacity} + (o.pass_rows ? at::kWindow : 0U)) * D);
        m.kv16.resize((T + 1U) * D);
        if (m.fast_index) {
            m.k8.resize((T + 1U) * at::kIdxDim);
            m.kscale.resize(T + 1U);
            m.fqprep.resize(IC * at::kFastIdxPrepBytes);
            m.fscores.resize(IC * (T + 1U));
            m.fblock_keys.resize(IC * m.seq0().attn->cand_stride());
            m.fsurvivors.resize(std::max<std::size_t>(1, at::select_survivor_words(IC, static_cast<std::uint32_t>(T + 1U), at::kTopK)));
        }
    }
    if (!o.decode_only) m.erows.resize(C * m.g.engram_cols * 264U);
    if (!o.decode_only) m.ws.resize(std::max({hc_mix_workspace_bytes(static_cast<std::uint32_t>(C), c.hidden_size),
                          shared_expert_workspace_bytes(static_cast<std::uint32_t>(C), c.moe_intermediate_size),
                          engram_workspace_bytes(static_cast<std::uint32_t>(C), m.g.engram_cols, c.hidden_size)}));
    m.run_buffer_bytes = device_buffer_bytes() - before;
    m.run_buffers_released = false;
}
void release_run_buffers(CudaPrefill::Impl& m) {
    STRIX_CUDA_CHECK(cudaDeviceSynchronize());
    m.graphs.clear();
    if (m.dec) m.dec->graphs.clear();
    auto& q0 = m.seq0();
    for (auto* b : {&q0.stream, &q0.mix_attn, &q0.mix_ffn, &q0.x, &q0.qkv, &q0.kvrows, &q0.qr, &q0.y, &q0.shared, &q0.route_w, &m.q, &m.o_, &m.o_low, &m.ckv,
                    &m.cscore, &m.iq, &m.iw, &m.scores, &m.partial, &m.router_scores, &m.kscale, &m.fscores})
        b->resize(0);
    for (auto* b : {&q0.ids, &m.block_keys, &m.survivors, &m.fblock_keys, &m.fsurvivors}) b->resize(0);
    for (auto* b : {&m.qprep, &m.erows, &m.ws, &m.k8, &m.fqprep}) b->resize(0);
    for (auto* b : {&m.win16, &m.kv16}) b->resize(0);
    m.pos64.resize(0);
    for (auto* b : {&q0.xa, &q0.qa, &m.oa, &m.la}) b->resize(0, 0);
    m.run_buffers_released = true;
}

std::vector<ExpertTier> default_tiers(const CudaPrefillOptions& o, std::uint32_t n_layers, std::uint64_t layer_bytes) {
    std::vector<ExpertTier> t(n_layers, ExpertTier::file);
    const double gib = static_cast<double>(layer_bytes) / double(1ULL << 30);
    auto dev_left = static_cast<std::int64_t>(o.device_expert_gib / gib + 1e-9);
    auto host_left = static_cast<std::int64_t>(o.host_expert_gib / gib + 1e-9);
    // device: the decoder layers first from the top (cone: few rows each, a streamed load would not hide), then the
    // encoder layers from the last one down (the first streamed layers are preloaded between runs)
    std::vector<std::uint32_t> order;
    for (std::uint32_t l = n_layers; l-- > 0;) order.push_back(l);
    for (const auto l : order) {
        if (l < o.first_layer || l >= o.end_layer) continue;
        if (dev_left > 0) {
            t[l] = ExpertTier::device;
            --dev_left;
        }
    }
    if (o.host_pin_order != HostPinOrder::layer) {
        std::vector<std::uint64_t> bytes(n_layers, 0);
        for (std::uint32_t l = o.first_layer; l < o.end_layer; ++l)
            if (t[l] == ExpertTier::file) bytes[l] = layer_bytes;
        const auto budget = o.host_expert_gib * double(1ULL << 30);
        const auto host = o.host_pin_order == HostPinOrder::early ? early_host_layers(bytes, o.slots, budget)
                                                                  : deadline_host_layers(bytes, o.slots, budget);
        for (std::uint32_t l = o.first_layer; l < o.end_layer; ++l)
            if (host[l]) t[l] = ExpertTier::host;
        return t;
    }
    for (std::uint32_t l = o.first_layer; l < o.end_layer; ++l)
        if (t[l] == ExpertTier::file && host_left > 0) {
            t[l] = ExpertTier::host;
            --host_left;
        }
    return t;
}

}  // namespace

CudaPrefill::CudaPrefill(const std::filesystem::path& gguf, const CudaPrefillOptions& options) : impl_(std::make_unique<Impl>()) {
    auto& m = *impl_;
    const auto t0 = std::chrono::steady_clock::now();
    m.o = options;
    m.gguf_path = gguf;
    auto& o = m.o;
    require(valid_host_pin_policy(o.host_commit_floor_gib, o.host_pin_wait_s), "host floor/wait must be finite and nonnegative");
    require(o.host_pin_order == HostPinOrder::layer || o.host_pin_order == HostPinOrder::deadline || o.host_pin_order == HostPinOrder::early, "host pin order layer, deadline or early");
    if (o.decode_only) o.pass_rows = 0;
    if (o.pass_rows != 0U) {
        require(o.max_positions > 0U && o.max_positions <= 2097152U, "long context capacity 1..2M");
        require(o.pass_rows <= 131072U && o.chunk_rows > 0U && o.chunk_rows <= 65535U,
                "bounded pass capacity <=128K and chunk rows 1..65535");
        require(std::isfinite(o.vram_cap_gib) && o.vram_cap_gib >= 0.0 && o.vram_cap_gib <= 1024.0 &&
                std::isfinite(o.host_expert_gib) && o.host_expert_gib >= 0.0 && o.host_expert_gib <= 1024.0 &&
                std::isfinite(o.host_commit_floor_gib) && o.host_commit_floor_gib >= 0.0 && o.host_commit_floor_gib <= 1024.0,
                "finite memory budgets in 0..1024 GiB");
        if (o.vram_cap_gib == 0.0) o.vram_cap_gib = kDefaultLongVramCapGiB;
        o.host_commit_floor_gib = std::max(4.0, o.host_commit_floor_gib);
        const double gib = double(1ULL << 30);
        const auto rows = std::min(o.pass_rows, o.max_positions);
        const auto engram = std::uint64_t{rows} * 2U * 2U * 24U * 264U;
        const auto planned = static_cast<std::uint64_t>((o.vram_cap_gib + o.host_expert_gib) * gib) +
            engram + ds41::long_request_host_reserve(o.max_positions) + (256ULL << 20U);
        require(ds41::host_commit_fits(ds41::host_memory_room(), planned,
                    static_cast<std::uint64_t>(o.host_commit_floor_gib * gib)), "insufficient free commit for long-context load and reserve");
    }
    if (o.pass_rows != 0U) {
        cudaDeviceProp device{};
        STRIX_CUDA_CHECK(cudaGetDeviceProperties(&device, o.device));
        require(std::string_view(device.name).find("RTX PRO 6000") != std::string_view::npos,
                "long prefill requires RTX PRO 6000 before allocation or profiling");
    }
    m.dev = select_device(o.device);
    STRIX_CUDA_CHECK(cudaStreamCreateWithFlags(&m.st, cudaStreamNonBlocking));
    const bool use_streamer = o.expert_residency || o.xfer_group_experts != 0U || o.xfer_piece_mib != 0U;
    const bool budget_from_device = o.auto_budget || o.vram_cap_gib > 0.0;
    // expert TP (decode steps, and runs of at most decode_rows rows) runs on the provider's whole-layer MoE only; the
    // streamer's range kernels keep the EP order, so the two are not combined
    require(!(use_streamer && o.expert_tp), "expert_tp needs the whole-layer expert provider (no expert_residency / xfer_group_experts / xfer_piece_mib)");
    // the H2D probe (a 1 s pinned transfer when the per-GPU cache has no fresh value) only where the device budget or
    // the streamer is in use; other objects (decode / FRONT) report the cached profile without measuring
    m.gpu = query_gpu_profile(o.device, o.gpu_probe && (use_streamer || budget_from_device));
    std::size_t free0 = 0, total0 = 0;
    STRIX_CUDA_CHECK(cudaMemGetInfo(&free0, &total0));
    m.file = std::make_unique<TensorFile>(gguf.string(), o.weights_overlay);
    m.file_path = gguf;
    m.cfg = m.file->config();
    m.g = LayerCoreGeometry::from(m.cfg);
    const auto& c = m.cfg;
    require(o.first_layer < o.end_layer && o.end_layer <= c.num_hidden_layers, "layer range [first, end)");
    require(o.decode_rows >= 1U && o.decode_rows <= 8U && o.decode_bucket > 0U, "decode rows 1..8, decode bucket > 0");
    require(o.window_rows >= 1U && o.window_rows <= at::kMaxCallRows, "window rows 1..64");
    require(o.chunk_rows > 0U && o.chunk_rows % o.window_rows == 0U && o.moe_chunk_rows > 0U && o.max_positions > 0U,
            "positive chunk rows, a multiple of window rows");
    require(o.pass_rows == 0U || (o.pass_rows >= o.chunk_rows && o.pass_rows % o.chunk_rows == 0U &&
                o.batch_positions.empty() && !o.graphs && o.engram_upload == "mapped"), "pass rows need a chunk multiple and no graphs/batches");
    if (o.cone) o.cone = o.first_layer == 0U && o.end_layer == c.num_hidden_layers;
    validate_bounded_replay(o.replay_rows, o.cone, o.first_layer, o.end_layer,
                            o.head_rows, o.draft_rows, o.fast_dense, o.fast_moe);
    m.reader = std::make_unique<FileReader>(gguf, o.engram_direct, o.io_threads, o.weight_read);
    m.engram_reader = std::make_unique<FileReader>(gguf, o.engram_direct, o.engram_threads, o.weight_read);
    m.model = std::make_unique<deepseek41::TargetModel>(deepseek41::validate_target(m.file->info()));
    m.plan = ds41::attn::plan_from_contract(m.model->layers);
    m.split = c.n_routed_experts / 2U;
    m.moe_shape = {c.hidden_size, c.moe_intermediate_size, c.n_routed_experts, c.num_experts_per_tok, static_cast<float>(c.swiglu_limit)};

    // dense weights of the range (tensor-core forms), embedding and head
    const std::size_t dense0 = device_buffer_bytes();
    m.core = std::make_unique<LayerCore>(*m.file, o.first_layer, o.end_layer, m.has_embed(), m.has_head(), std::max({1U, o.head_rows, o.decode_rows}), m.st);
    m.attn_w.assign(c.num_hidden_layers, {});
    for (std::uint32_t l = o.first_layer; l < o.end_layer; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        const auto& r = m.plan[l];
        auto& w = m.attn_w[l];
        w.kv_norm = m.file->device_f32(p + "attn_kv_a_norm.weight");
        w.sinks = m.file->device_f32(p + "attn_sinks.weight");
        if (r.owns_kv) {
            w.compressor_kv = m.file->device_f16(p + "attn_compressor_kv.weight");
            if (r.ratio == 2U) w.compressor_gate = m.file->device_f16(p + "attn_compressor_gate.weight");
            w.compressor_norm = m.file->device_f32(p + "attn_compressor_norm.weight");
            w.index_k = m.file->device_f16(p + "indexer.attn_k.weight");
            w.index_k_norm = m.file->device_f32(p + "indexer.k_norm.weight");
        }
        if (r.owns_index) {
            w.index_q_b = m.file->device_f16(p + "indexer.attn_q_b.weight");
            w.index_proj = m.file->device_f16(p + "indexer.proj.weight");
        }
    }
    if (m.has_embed()) (void)m.file->device_f16("token_embd.weight");  // the embedding run() reads (LayerCore holds it)
    m.dense_bytes = device_buffer_bytes() - dense0;
    m.dense_gib = double(m.dense_bytes) / double(1ULL << 30);
    // Prefill retains its row capacity; decode-only bounds query rows independently of context history.
    at::LayerAttentionOptions ao;
    ao.first_layer = o.first_layer;
    ao.end_layer = o.end_layer;
    ao.max_positions = o.max_positions + 1U;
    m.row_capacity = o.pass_rows ? std::min(o.pass_rows, o.max_positions) : o.max_positions;
    ao.max_rows = m.row_capacity;
    ao = at::prefill_attention_options(ao, o.pass_rows, o.decode_rows);
    if (o.pass_rows != 0U && o.cone) {
        ds41::SuffixReplayGeometry geo;
        geo.draft_rows = o.draft_rows;
        const auto replay = ds41::plan_suffix_replay(geo, 0U, o.max_positions);
        const auto first = replay.layers[20].out_begin;
        const auto shift = o.head_rows > 1U ? o.head_rows - 1U : 0U;
        const auto from = first > shift ? first - shift : 0U;
        const auto aligned = from / o.window_rows * o.window_rows;
        const auto suffix_rows = o.max_positions - aligned;
        require(suffix_rows <= m.row_capacity && o.head_rows <= m.row_capacity && o.draft_rows <= m.row_capacity,
                "pass_rows too small for the head/draft cone");
        ao.decoder_rows = std::min(m.row_capacity, static_cast<std::uint32_t>(suffix_rows + o.window_rows));
        ao.candidate_rows = ao.decoder_rows;
    }
    ao.window_rows = o.window_rows;
    ao.hidden = c.hidden_size;
    ao.q_lora_rank = c.q_lora_rank;
    ao = at::decode_attention_options(ao, o.decode_only, o.lean_state, o.decode_rows);
    m.seqs.push_back(std::make_unique<Impl::Seq>());
    m.seq0().capacity = o.max_positions;
    m.seq0().attn = std::make_unique<at::AttentionState>(m.plan, ao);
    m.last_batch_ctx.assign(1, 0U);
    m.attn_options = ao;

    // routed experts
    std::vector<ExpertExtent> extents(c.num_hidden_layers);
    for (std::uint32_t l = o.first_layer; l < o.end_layer; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        const char* names[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
        for (int k = 0; k < 3; ++k) {
            const auto& t = m.file->tensor(p + names[k]);
            extents[l].offset[k] = t.absolute_offset;
            extents[l].bytes[k] = t.byte_size;
        }
        require(extents[l].bytes[0] == m.moe_shape.gate_expert_bytes() * c.n_routed_experts &&
                    extents[l].bytes[2] == m.moe_shape.down_expert_bytes() * c.n_routed_experts,
                "routed expert tensor sizes");
    }
    m.extents = extents;

    // Engram
    if (o.engram == "file") m.engram_tables = engram_tables(*m.model);
    else require(o.engram == "zero", "engram must be file or zero");
    require(o.engram_upload == "mapped" || o.engram_upload == "stream", "engram upload must be mapped or stream");
    if (o.engram_cache_mib != 0U) m.engram_reader->set_page_cache(o.engram_cache_mib << 20U);
    if (o.engram_upload == "stream") {
        int lo = 0, hi = 0;
        STRIX_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
        STRIX_CUDA_CHECK(cudaStreamCreateWithPriority(&m.engram_st, cudaStreamNonBlocking, hi));
        m.engram_ev.resize(c.engram_layer_ids.size());
        for (auto& e : m.engram_ev) STRIX_CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        m.engram_dev.resize(c.engram_layer_ids.size() * std::size_t{m.row_capacity} * m.g.engram_cols * 264U);
    }
    m.hasher = std::make_unique<EngramHasher>(m.model->engram);
    if (o.pass_rows) {
        std::size_t engram_layers = 0;
        for (std::size_t li = 0; li < c.engram_layer_ids.size(); ++li)
            if (c.engram_layer_ids[li] >= o.first_layer && c.engram_layer_ids[li] < o.end_layer) engram_layers = li + 1U;
        const auto bytes = engram_layers * std::size_t{m.row_capacity} * m.g.engram_cols * 264U;
        m.seq0().engram_host.resize(bytes, true);
        m.engram_ahead.resize(bytes, true);
    }

    // buffers
    const std::size_t H = c.hidden_size, MC = o.moe_chunk_rows;
    const std::size_t buffers0 = device_buffer_bytes();
    alloc_run_buffers(m);
    // the routed scratch: run()'s chunks, or only the decode rows (decode_only: no run buffers)
    const std::size_t MR = o.decode_only ? std::size_t{o.decode_rows} : MC;
    auto& da = m.decode_alloc_options;
    da.rows = o.decode_rows;
    da.raw_rows = static_cast<std::uint32_t>(MR);
    da.hidden = c.hidden_size;
    da.top_k = c.num_experts_per_tok;
    da.inter = c.moe_intermediate_size;
    da.cand_stride = m.seq0().attn->cand_stride();
    da.expert_tp = o.expert_tp;
    da.moe_workspace = std::max(moe_layer_workspace_bytes(static_cast<std::uint32_t>(MR), m.moe_shape),
                               o.fast_moe ? moe_fast_workspace_bytes(static_cast<std::uint32_t>(MR), m.moe_shape) : std::size_t{0});
    const auto decode_alloc = at::decode_allocations(da);
    m.raw.resize(decode_alloc.bytes("raw") / sizeof(float));
    if (o.expert_tp) m.raw1.resize(decode_alloc.bytes("raw1") / sizeof(float));
    m.moe_ws.resize(decode_alloc.bytes("moe_ws"));
    if (!o.trellis_shexp.empty()) {
#ifdef STRIX_DS41_CUDA_TRELLIS
        const std::uint32_t I = c.moe_intermediate_size, HS = c.hidden_size;
        const char* parts[3] = {"w1", "w3", "w2"};
        std::size_t ws = 0;
        m.tshexp.resize(o.end_layer - o.first_layer);
        for (std::uint32_t l = o.first_layer; l < o.end_layer; ++l) {
            const auto path = std::filesystem::path(o.trellis_shexp) / ("dense.L" + std::to_string(l) + ".safetensors");
            for (std::size_t k = 0; k < 3U; ++k) {
                const std::string p = "layers." + std::to_string(l) + ".ffn.shared_experts." + parts[k] + ".";
                auto& t = m.tshexp[l - o.first_layer][k];
                const auto marker = read_safetensor(path, p + "mul1");
                const auto packed = read_safetensor(path, p + "trellis");
                const auto suh = read_safetensor(path, p + "suh");
                const auto svh = read_safetensor(path, p + "svh");
                require(marker.bytes.size() == 4U && packed.shape.size() == 3U && packed.dtype == "I16" && suh.dtype == "F16" && svh.dtype == "F16",
                        "trellis shexp: tensor types (mul1 I32, trellis I16 [I/16][N/16][16 K], suh / svh F16)");
                std::uint32_t mark = 0;
                std::memcpy(&mark, marker.bytes.data(), 4);
                const auto input = static_cast<std::uint32_t>(suh.shape.at(0)), output = static_cast<std::uint32_t>(svh.shape.at(0));
                const auto bits = static_cast<std::uint32_t>(packed.shape[2] / 16U);
                require(k < 2U ? (input == HS && output == I) : (input == I && output == HS), "trellis shexp: gate / up [I][H], down [H][I]");
                require(m.trellis_bits == 0U || m.trellis_bits == bits, "trellis shexp: one K for gate, up and down");
                m.trellis_bits = bits;
                t.packed.resize(packed.bytes.size() / 2U);
                t.packed.upload(reinterpret_cast<const std::uint16_t*>(packed.bytes.data()), packed.bytes.size() / 2U);
                t.suh.resize(input);
                t.suh.upload(reinterpret_cast<const std::uint16_t*>(suh.bytes.data()), input);
                t.svh.resize(output);
                t.svh.upload(reinterpret_cast<const std::uint16_t*>(svh.bytes.data()), output);
                t.w.packed = t.packed.get();
                t.w.suh = t.suh.get();
                t.w.svh = t.svh.get();
                t.w.packed_bytes = packed.bytes.size();
                t.w.suh_bytes = std::size_t{input} * 2U;
                t.w.svh_bytes = std::size_t{output} * 2U;
                t.w.input = input;
                t.w.output = output;
                t.w.bits = bits;
                t.w.marker = mark;
                require(t.w.packed_bytes == trellis::packed_bytes(input, output, bits), "trellis shexp: packed bytes");
                t.splits = trellis_vector_splits(input, output, bits);
                m.trellis_bytes += packed.bytes.size() + (std::size_t{input} + output) * 2U;
                trellis::Linear v;
                v.weights = t.w;
                v.rows = o.decode_rows;
                v.splits = t.splits;
                v.mode = trellis::Mode::gemv_quality;
                v.x_stride = input;
                v.y_stride = output;
                ws = std::max(ws, trellis::workspace_bytes(v));
            }
        }
        da.trellis = true;
        da.trellis_workspace = ws;
        const auto tr_alloc = at::decode_allocations(da);
        m.tr_gate.resize(tr_alloc.bytes("tr_gate") / sizeof(float));
        m.tr_up.resize(tr_alloc.bytes("tr_up") / sizeof(float));
        m.tr_act.resize(tr_alloc.bytes("tr_act") / sizeof(float));
        m.tr_ws.resize(tr_alloc.bytes("tr_ws"));
        m.tr_ws_bytes = ws;
#else
        throw std::invalid_argument("cuda decode: trellis_shexp needs the build option STRIX_ENABLE_CUDA_TRELLIS");
#endif
    }
    // every row / chunk / workspace buffer above
    m.buffer_bytes = device_buffer_bytes() - buffers0;
    m.state_bytes = m.seq0().attn->device_bytes();
    if (use_streamer && o.expert_residency) m.mapped_ids.resize(MC * c.num_experts_per_tok);
    // closed-batch contexts (before the expert budget: their memory comes out of the resident experts)
    for (const std::uint32_t cap : o.batch_positions) {
        require(cap >= 1U && cap <= o.max_positions, "batch positions 1..max_positions");
        m.seqs.push_back(std::make_unique<Impl::Seq>());
        auto& q = *m.seqs.back();
        at::LayerAttentionOptions bo = ao;
        bo.max_positions = cap + 1U;
        bo.max_rows = cap;
        q.attn = std::make_unique<at::AttentionState>(m.plan, bo);
        m.alloc_seq(q, cap, bo);
        m.batch_bytes += q.row_bytes + q.attn->device_bytes();
    }

    // Service request buffers are mandatory: allocate every configured context's
    // worst-case Engram rows and the fast-export capture before optional experts.
    if (o.service_host_buffers && !o.pass_rows) {
        for (auto& q : m.seqs)
            q->engram_host.resize(c.engram_layer_ids.size() * std::size_t{q->capacity} * m.g.engram_cols * 264U, true);
    }
    if (o.reserve_export_capture) {
        m.export_capture = std::make_shared<attn::ExportCapture>();
        m.export_capture->reserve(m.seq0().attn->capture_bound(o.first_layer, o.end_layer, o.max_positions));
    }

    // ---- routed experts last: budgets (from the device when asked), tiers or per-expert residency, the provider
    // (device layers, slots, then the pinned host tier: with a commit floor it takes what is left) or the streamer
    STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));
    const std::uint64_t layer_bytes = extents[o.first_layer].total();
    const std::uint64_t expert_bytes = m.moe_shape.expert_bytes();
    {
        std::size_t free_b = 0, total_b = 0;
        STRIX_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        const double gib = double(1ULL << 30);
        m.used_before_experts_gib = double(free0 > free_b ? free0 - free_b : 0U) / gib;
        m.free_at_plan_gib = double(free_b) / gib;
        if (budget_from_device) {
            // Fixed allocations above already use the lean ledger; only the run's temporary token upload remains.
            const double transient = o.decode_only ? 0.0 : double(std::uint64_t{m.row_capacity} * 4U) / gib;
            double avail = m.free_at_plan_gib - o.vram_reserve_gib - transient;
            if (o.vram_cap_gib > 0.0) avail = std::min(avail, o.vram_cap_gib - m.used_before_experts_gib - o.vram_reserve_gib - transient);
            if (!(avail > 0.0))
                throw std::invalid_argument("ds41 cuda prefill: no device memory left for routed experts (used " +
                                            std::to_string(m.used_before_experts_gib) + " GiB, free " + std::to_string(m.free_at_plan_gib) +
                                            " GiB, cap " + std::to_string(o.vram_cap_gib) + " GiB, reserve " + std::to_string(o.vram_reserve_gib) + " GiB)");
            // whole layers: the slots come out of the budget; per expert: the plan below sizes them
            o.device_expert_gib = o.expert_residency ? avail : std::max(0.0, avail - double(o.slots) * double(layer_bytes) / gib);
        }
        m.expert_budget_gib = o.device_expert_gib;
    }
    if (!use_streamer) {
        m.tiers = o.tiers.empty() ? default_tiers(o, c.num_hidden_layers, layer_bytes) : o.tiers;
        require(m.tiers.size() >= o.end_layer, "tiers cover the layer range");
        m.experts = std::make_unique<ExpertProvider>(*m.reader, extents, m.tiers, o.first_layer, o.end_layer, o.slots, m.st, o.host_alias_layers,
                                                     o.host_commit_floor_gib, o.host_pin_wait_s, o.host_tier_implicit);
        m.tiers = m.experts->tiers();  // a host layer that could not be pinned streams from the file
    } else {
        const std::uint32_t E = c.n_routed_experts;
        detail::StreamerPlan sp;
        sp.first = o.first_layer;
        sp.end = o.end_layer;
        sp.experts = E;
        sp.expert_bytes[0] = sp.expert_bytes[1] = m.moe_shape.gate_expert_bytes();
        sp.expert_bytes[2] = m.moe_shape.down_expert_bytes();
        sp.extents.resize(o.end_layer);
        for (std::uint32_t l = o.first_layer; l < o.end_layer; ++l)
            for (int k = 0; k < 3; ++k) {
                sp.extents[l].offset[k] = extents[l].offset[k];
                sp.extents[l].bytes[k] = extents[l].bytes[k];
            }
        sp.order.assign(o.end_layer, {});
        sp.resident.assign(o.end_layer, 0U);
        sp.stream_tier.assign(o.end_layer, ExpertTier::host);
        sp.slots = o.slots;
        sp.group_experts = o.xfer_group_experts;
        sp.piece_bytes = std::uint64_t{o.xfer_piece_mib} << 20U;
        sp.host_alias = o.host_alias_layers;
        sp.host_commit_floor_gib = o.host_commit_floor_gib;
        sp.host_pin_wait_s = o.host_pin_wait_s;
        sp.host_tier_implicit = o.host_tier_implicit;
        sp.host_promote.enabled = host_promote_effective(o);
        sp.host_promote.floor_bytes = static_cast<std::uint64_t>(std::max(o.host_commit_floor_gib, kHostPromoteFloorGiB) * double(1ULL << 30));
        sp.host_promote.cap_bytes = static_cast<std::uint64_t>(std::max(0.0, o.host_promote_max_gib) * double(1ULL << 30));
        sp.timing = o.timing;
        if (!o.expert_residency) {
            // whole layers through the streamer: device = all resident, host / file = all streamed, identity order
            m.tiers = o.tiers.empty() ? default_tiers(o, c.num_hidden_layers, layer_bytes) : o.tiers;
            require(m.tiers.size() >= o.end_layer, "tiers cover the layer range");
            for (std::uint32_t l = o.first_layer; l < o.end_layer; ++l) {
                sp.order[l].resize(E);
                for (std::uint32_t e = 0; e < E; ++e) sp.order[l][e] = e;
                sp.resident[l] = m.tiers[l] == ExpertTier::device ? E : 0U;
                sp.stream_tier[l] = m.tiers[l] == ExpertTier::file ? ExpertTier::file : ExpertTier::host;
            }
        } else {
            // per expert: streamed counts by rows at max_positions, the hottest experts resident
            std::vector<double> w;
            {
                std::vector<std::uint32_t> o0(c.num_hidden_layers, 0U);
                if (o.cone) {
                    ds41::SuffixReplayGeometry geo;
                    geo.layers = c.num_hidden_layers;
                    geo.draft_rows = o.draft_rows;
                    const auto pl = ds41::plan_suffix_replay(geo, 0U, o.max_positions);
                    for (const auto& x : pl.layers) o0[x.layer] = static_cast<std::uint32_t>(x.out_begin / o.window_rows * o.window_rows);
                }
                for (std::uint32_t l = o.first_layer; l < o.end_layer; ++l) w.push_back(double(o.max_positions - o0[l]));
            }
            const auto budget_experts = static_cast<std::uint64_t>(o.device_expert_gib * double(1ULL << 30) / double(expert_bytes));
            // rows-weighted first; a small budget falls back to even streaming (smaller slots), then to one slot
            std::vector<std::uint32_t> s;
            std::string plan_note = "rows";
            try {
                s = plan_streamed(budget_experts, o.slots, w, E);
            } catch (const std::invalid_argument&) {
                try {
                    s = plan_streamed(budget_experts, o.slots, std::vector<double>(w.size(), 1.0), E);
                    plan_note = "even";
                } catch (const std::invalid_argument&) {
                    s = plan_streamed(budget_experts, 1U, std::vector<double>(w.size(), 1.0), E);
                    sp.slots = 1U;
                    plan_note = "even-1slot";
                }
            }
            m.residency_plan = plan_note;
            const auto rank = expert_ranking(o.expert_rank, c.num_hidden_layers, E);
            double host_left = o.host_expert_gib * double(1ULL << 30);
            for (std::uint32_t l = o.first_layer; l < o.end_layer; ++l) {
                const std::uint32_t r = E - s[l - o.first_layer];
                const double need = double(E - r) * double(expert_bytes);
                // host budget counts every streamed part, aliased or not (host_alias: timing only)
                sp.stream_tier[l] = host_left >= need ? ExpertTier::host : ExpertTier::file;
                if (sp.stream_tier[l] == ExpertTier::host) host_left -= need;
                sp.resident[l] = r;
                if (sp.stream_tier[l] == ExpertTier::file || r == E || r == 0U) {
                    // read from the file per run: the streamed experts as one id range (long reads); a prefill of
                    // thousands of rows needs every expert anyway, so which ones stay resident only matters for
                    // short prompts
                    sp.order[l].resize(E);
                    for (std::uint32_t e = 0; e < E; ++e) sp.order[l][e] = e;
                    continue;
                }
                // packed into pinned memory at load: the hottest experts resident, the rest streamed
                std::vector<std::uint32_t> res(rank[l].begin(), rank[l].begin() + r), str(rank[l].begin() + r, rank[l].end());
                std::sort(res.begin(), res.end());
                std::sort(str.begin(), str.end());
                res.insert(res.end(), str.begin(), str.end());
                sp.order[l] = std::move(res);
            }
            if (o.host_pin_order != HostPinOrder::layer) {
                // Preserve the legacy budget's resident expert IDs / virtual permutation.
                // Changing the source must not change which experts are resident, or the
                // range arithmetic. FILE can read the same non-identity permutation as HOST.
                std::vector<std::uint64_t> bytes(o.end_layer, 0);
                for (std::uint32_t l = o.first_layer; l < o.end_layer; ++l)
                    bytes[l] = std::uint64_t{E - sp.resident[l]} * expert_bytes;
                const auto budget = o.host_expert_gib * double(1ULL << 30);
                const auto host = o.host_pin_order == HostPinOrder::early ? early_host_layers(bytes, sp.slots, budget)
                                                                          : deadline_host_layers(bytes, sp.slots, budget);
                for (std::uint32_t l = o.first_layer; l < o.end_layer; ++l)
                    sp.stream_tier[l] = host[l] ? ExpertTier::host : ExpertTier::file;
            }
            m.tiers.assign(c.num_hidden_layers, ExpertTier::device);
            for (std::uint32_t l = o.first_layer; l < o.end_layer; ++l)
                m.tiers[l] = sp.resident[l] == E ? ExpertTier::device : sp.stream_tier[l];
        }
        m.resident_experts = sp.resident;
        m.streamer = std::make_unique<detail::ExpertStreamer>(*m.reader, std::move(sp));
        // host layers whose pinned buffer could not be allocated were demoted to the file tier by the streamer
        for (std::uint32_t l = o.first_layer; l < o.end_layer; ++l)
            if (m.tiers[l] == ExpertTier::host && m.streamer->plan().stream_tier[l] == ExpertTier::file) m.tiers[l] = ExpertTier::file;
    }
    m.run_tiers = m.tiers;
    STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));
    {
        std::size_t free_b = 0, total_b = 0;
        STRIX_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        m.device_in_use_at_load = total_b - free_b;
    }
    m.load_seconds = seconds_since(t0);
}

CudaPrefill::~CudaPrefill() {
    if (!impl_) return;
    impl_->dec.reset();
    if (impl_->engram_thread.joinable()) impl_->engram_thread.join();
    impl_->experts.reset();
    impl_->streamer.reset();
    for (auto e : impl_->engram_ev) cudaEventDestroy(e);
    if (impl_->engram_st) {
        cudaStreamSynchronize(impl_->engram_st);
        cudaStreamDestroy(impl_->engram_st);
    }
    if (impl_->st) {
        cudaStreamSynchronize(impl_->st);
        cudaStreamDestroy(impl_->st);
    }
}

const std::vector<ExpertTier>& CudaPrefill::tiers() const noexcept { return impl_->tiers; }
void CudaPrefill::set_graphs(bool on) noexcept { impl_->o.graphs = on; }
bool CudaPrefill::set_fast_attention(bool on) {
    auto& m = *impl_;
    const bool want = on && m.win16.get() != nullptr;  // the buffers exist only with fast_attention at construction
    if (want != m.o.fast_attention) {
        STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));
        m.graphs.clear();
        m.o.fast_attention = want;
    }
    return want;
}
double CudaPrefill::load_seconds() const noexcept { return impl_->load_seconds; }
std::size_t CudaPrefill::device_bytes() const noexcept {
    const auto& m = *impl_;
    return m.dense_bytes + m.buffer_bytes + m.state_bytes + m.batch_bytes + (m.streamer ? m.streamer->device_bytes() : m.experts->device_bytes());
}
std::shared_ptr<attn::ExportCapture> CudaPrefill::export_capture() const noexcept { return impl_->export_capture; }

const HostPinStats& CudaPrefill::host_pin_stats() const noexcept {
    return impl_->streamer ? impl_->streamer->host_pin_stats() : impl_->experts->host_pin_stats();
}

std::size_t CudaPrefill::pinned_bytes() const noexcept {
    const auto& m = *impl_;
    std::size_t decode = 0;
    if (m.dec) {
        const auto& d = *m.dec;
        decode = d.params.size() + d.engram.size() + d.io_in.size() + d.io_out.size() + d.logits.size() + d.sec_host.size();
        if (d.staging) decode += d.staging->size();
    }
    return (m.streamer ? m.streamer->pinned_bytes() : m.experts->pinned_bytes()) + m.engram_pinned_bytes() + decode;
}

double CudaPrefill::decode_residency(std::uint32_t stop, bool release_run_buffers_too) {
    auto& m = *impl_;
    const auto t0 = std::chrono::steady_clock::now();
    require(stop > m.o.first_layer && stop <= m.o.end_layer, "decode residency: stop in (first, end]");
    // The expert streamer (expert_residency / xfer_*) owns the routed experts of run() only: decode residency adopts and
    // promotes whole layers of the whole-layer provider, so the two are not combined.
    release_idle_preload();  // defer_idle_preload: the provider back in its idle state before the decode adopts its slots
    require(!m.streamer, "decode residency needs the whole-layer expert provider (no expert_residency / xfer_group_experts / xfer_piece_mib)");
    // bounded passes (pass_rows) keep one pass of row buffers and compact decoder rows: not a decode object
    require(m.o.pass_rows == 0U, "decode residency needs pass_rows 0");
    STRIX_CUDA_CHECK(cudaDeviceSynchronize());
    m.graphs.clear();
    if (m.dec) m.dec->graphs.clear();
    if (m.decode_mode) m.experts->clear_decode();
    std::size_t released = 0;
    if (release_run_buffers_too && !m.run_buffers_released) {
        released = m.run_buffer_bytes;
        release_run_buffers(m);
    }
    if (m.dec) m.dec->graphs.clear();  // the decode graphs captured the previous expert addresses
    std::uint32_t adopted = 0, promoted = 0;
    std::uint64_t promoted_bytes = 0;
    m.experts->make_decode_resident(m.o.first_layer, stop, m.st, adopted, promoted, promoted_bytes);
    m.tiers = m.run_tiers;
    for (std::uint32_t l = m.o.first_layer; l < stop; ++l)
        if (m.experts->decode_resident(l)) m.tiers[l] = ExpertTier::device;
    m.decode_mode = true;
    const double enqueue_s = seconds_since(t0);
    std::size_t free_b = 0, total_b = 0;
    STRIX_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    std::ostringstream s;
    s << "{\"stop\":" << stop << ",\"adopted_slot_layers\":" << adopted << ",\"promoted_layers\":" << promoted << ",\"promoted_gb\":" << double(promoted_bytes) / 1e9
      << ",\"released_run_buffers_gb\":" << double(released) / 1e9 << ",\"enqueue_s\":" << enqueue_s << ",\"device_in_use_gb\":" << double(total_b - free_b) / 1e9 << "}";
    m.last_residency = s.str();
    return enqueue_s;
}
std::string CudaPrefill::decode_residency_report() const { return impl_->last_residency; }

const std::uint8_t* CudaPrefill::device_tensor(const std::string& name) { return impl_->file->device(name); }
Q8Tiled CudaPrefill::head_tiled() const noexcept { return impl_->core ? impl_->core->head_tiled() : Q8Tiled{}; }

std::string CudaPrefill::weight_io_json() const {
    const auto& m = *impl_;
    // The expert and Engram openers share options and a session. Decode always opens direct: --buffered gives it
    // a separate session whose counters must also be reported, without counting shared sessions twice.
    const auto* reader = m.reader->multi() ? m.reader.get() : m.engram_reader->multi() ? m.engram_reader.get() : nullptr;
    const auto* decode = m.dec && m.dec->reader && m.dec->reader->multi() ? m.dec->reader.get() : nullptr;
    if (!reader && !decode) return "null";
    std::string json = "{\"requested_verify_mode\":\"" + m.o.weight_read.weight_verify + "\",\"session\":" +
                       (reader ? reader : decode)->multi_stats_json();
    if (reader && decode && reader->direct() != decode->direct())
        json += ",\"decode_session\":" + decode->multi_stats_json();
    return json + "}";
}

std::string CudaPrefill::weight_io_fields() const {
    const auto json = weight_io_json();
    return json == "null" ? std::string() : ",\"weight_io\":" + json;
}

std::string CudaPrefill::describe() const {
    const auto& m = *impl_;
    std::ostringstream s;
    const auto attention_alloc = at::attention_allocations(m.plan, m.attn_options);
    std::uint32_t dev = 0, host = 0, file = 0;
    for (std::uint32_t l = m.o.first_layer; l < m.o.end_layer; ++l)
        (m.tiers[l] == ExpertTier::device ? dev : m.tiers[l] == ExpertTier::host ? host : file) += 1U;
    s << "{\"layers\":[" << m.o.first_layer << "," << m.o.end_layer << "],\"max_positions\":" << m.o.max_positions << ",\"pass_rows\":" << m.o.pass_rows << ",\"row_capacity\":" << m.row_capacity
      << ",\"chunk_rows\":" << m.o.chunk_rows << ",\"moe_chunk_rows\":" << m.o.moe_chunk_rows << ",\"window_rows\":" << m.o.window_rows
      << ",\"prefill_pass_rows\":" << (m.o.decode_only ? 0U : m.row_capacity)
      << ",\"prefill_sequence_alloc_bytes\":" << at::allocation_json(m.o.decode_only ? at::AllocationLedger{} :
            at::prefill_sequence_allocations(m.row_capacity, m.cfg.hidden_size, m.cfg.q_lora_rank, m.cfg.num_experts_per_tok))
      << ",\"prefill_temporary_token_bytes\":" << (m.o.decode_only ? 0ULL : std::uint64_t{m.row_capacity} * 4U)
      << ",\"decode_only\":" << (m.o.decode_only ? "true" : "false")
      << ",\"lean_state\":" << (m.o.decode_only && m.o.lean_state ? "true" : "false")
      << ",\"attention_max_rows\":" << m.attn_options.max_rows << ",\"attention_window_rows\":" << m.attn_options.window_rows
      << ",\"ring_rows\":" << attention_alloc.ring_rows << ",\"candidate_stride\":" << attention_alloc.cand_stride
      << ",\"attention_alloc_bytes_per_sequence\":" << at::allocation_json(attention_alloc)
      << ",\"decode_driver_alloc_bytes_with_lazy_sections\":" << at::allocation_json(at::decode_allocations(m.decode_alloc_options))
      << ",\"fast_moe_wide_tiles\":" << (m.o.fast_moe_wide_tiles ? "true" : "false")
      << ",\"fast_moe_signed_lut\":" << (m.o.fast_moe_signed_lut ? "true" : "false")
      << ",\"cone\":" << (m.o.cone ? "true" : "false") << ",\"replay_rows\":" << (m.o.cone ? m.o.replay_rows : 0U) << ",\"fast_moe\":" << (m.o.fast_moe ? "true" : "false") << ",\"expert_layers\":{\"device\":" << dev << ",\"host\":" << host << ",\"file\":" << file
      << "},\"tiers\":\"";
    for (std::uint32_t l = m.o.first_layer; l < m.o.end_layer; ++l) s << "DHF"[static_cast<int>(m.tiers[l])];
    const std::size_t expert_dev = m.streamer ? m.streamer->device_bytes() : m.experts->device_bytes();
    const std::size_t expert_pin = m.streamer ? m.streamer->pinned_bytes() : m.experts->pinned_bytes();
    const std::size_t slots_dev = m.streamer ? m.streamer->slot_count() * m.streamer->slot_bytes() : m.experts->device_bytes() - m.experts->resident_bytes();
    const auto& pins = host_pin_stats();
    s << "\",\"slots\":" << (m.streamer ? m.streamer->slot_count() : m.experts->slot_count())
      << ",\"expert_layer_gb\":" << double(m.moe_shape.expert_bytes() * m.cfg.n_routed_experts) / 1e9
      << ",\"device_gb\":" << double(device_bytes()) / 1e9 << ",\"dense_gb\":" << double(m.dense_bytes) / 1e9
      << ",\"expert_gb\":" << double(expert_dev - slots_dev) / 1e9 << ",\"slots_gb\":" << double(slots_dev) / 1e9
      << ",\"buffers_gb\":" << double(m.buffer_bytes) / 1e9 << ",\"state_gb\":" << double(m.state_bytes) / 1e9
      << ",\"batch_gb\":" << double(m.batch_bytes) / 1e9 << ",\"batch_positions\":[";
    for (std::size_t k = 1; k < m.seqs.size(); ++k) s << (k > 1 ? "," : "") << m.seqs[k]->capacity;
    s << "]"
      << ",\"device_in_use_gb\":" << double(m.device_in_use_at_load) / 1e9
      << ",\"pinned_gb\":" << double(expert_pin) / 1e9
      << ",\"host_pin_failed_layers\":" << (m.streamer ? m.streamer->host_pin_failed_layers() : m.experts->host_pin_failed_layers())
      << ",\"engram_pinned_gb\":" << double(m.engram_pinned_bytes()) / 1e9 << ",\"pinned_total_gb\":" << double(pinned_bytes()) / 1e9
      << ",\"host_pin_s\":" << (m.streamer ? m.streamer->host_pin_seconds() : m.experts->host_pin_seconds())
      << ",\"host_pin_waits\":" << (m.streamer ? m.streamer->host_pin_waits() : m.experts->host_pin_waits()) << ",\"host_commit_floor_gib\":" << m.o.host_commit_floor_gib
      << ",\"run_buffers_gb\":" << double(m.run_buffer_bytes) / 1e9 << ",\"host_alias_layers\":" << m.o.host_alias_layers << ",\"engram\":\"" << m.o.engram << "\""
      << ",\"fast_dense\":" << (m.o.fast_dense ? "true" : "false") << ",\"io_direct\":" << (m.reader->direct() ? "true" : "false")
      << ",\"fast_attention\":" << (m.o.fast_attention ? "true" : "false") << ",\"fast_index\":" << (m.fast_index ? "true" : "false")
      << ",\"expert_tp\":" << (m.o.expert_tp ? "\"ds41-expert-tp2-i1152-v1\"" : "false")
      << ",\"decode_fuse_hc\":" << (m.o.decode_fuse_hc ? "true" : "false")
      << ",\"decode_fuse_shared\":" << (m.o.decode_fuse_shared ? "true" : "false")
      << ",\"decode_router_v2\":" << (m.o.decode_router_v2 ? "true" : "false")
      << ",\"decode_pdl\":" << (m.o.decode_pdl ? "true" : "false")
      << ",\"decode_ob_one\":" << (m.o.decode_ob_one ? "true" : "false")
      << ",\"decode_shared_fork\":" << (m.o.decode_shared_fork ? "true" : "false")
      << ",\"decode_attn_fork\":" << (m.o.decode_attn_fork ? "true" : "false")
      << ",\"decode_qakv_s1\":" << (m.o.decode_qakv_s1 ? "true" : "false")
      << ",\"decode_ob_expand\":" << (m.o.decode_ob_expand && m.o.decode_ob_one ? "true" : "false")
      << ",\"decode_moe_expand\":" << (m.o.decode_moe_expand ? "true" : "false")
      << ",\"overlay\":\"" << m.file->overlay_description() << "\"";
    s << ",\"host_tier_policy\":\"" << (m.o.host_tier_implicit ? "implicit" : "explicit")
      << "\",\"host_requested_bytes\":" << pins.requested_bytes
      << ",\"host_effective_bytes\":" << pins.effective_bytes
      << ",\"host_pin_wait_s\":" << m.o.host_pin_wait_s
      << ",\"host_effective_wait_s\":" << (m.o.host_tier_implicit ? 0.0 : m.o.host_pin_wait_s)
      << ",\"export_capture_pinned_bytes\":" << (m.export_capture ? m.export_capture->pinned_bytes() : 0)
      << ",\"service_pinned_bytes\":" << pinned_bytes() + (m.export_capture ? m.export_capture->pinned_bytes() : 0)
      << ",\"host_demotions\":[";
    for (std::size_t i = 0; i < pins.demotions.size(); ++i)
        s << (i ? "," : "") << "{\"layer\":" << pins.demotions[i].layer << ",\"reason\":\""
          << host_pin_reason_name(pins.demotions[i].reason) << "\"}";
    const auto room = ds41::host_memory_room();
    s << "],\"host_free_physical_bytes\":" << (room.known ? std::to_string(room.free_physical) : "\"unknown\"")
      << ",\"host_free_commit_bytes\":" << (room.known ? std::to_string(room.free_commit) : "\"unknown\"");
    s << ",\"host_pin_order\":\"" << host_pin_order_name(m.o.host_pin_order) << "\",\"host_layers\":[";
    bool first_host = true;
    for (std::uint32_t l = m.o.first_layer; l < m.o.end_layer; ++l) {
        if (m.run_tiers[l] != ExpertTier::host) continue;
        s << (first_host ? "" : ",") << l;
        first_host = false;
    }
    s << "]";
    s << ",\"residency\":\"" << (m.o.expert_residency ? "expert" : "layer") << "\",\"transport\":\"" << (m.streamer ? "streamer" : "provider") << "\"";
    {
        const auto pr = m.streamer ? m.streamer->promoted() : detail::ExpertStreamer::Promoted{};
        s << ",\"host_promote\":{\"switch\":" << (m.o.host_promote ? "true" : "false")
          << ",\"enabled\":" << (m.streamer && m.streamer->plan().host_promote.enabled ? "true" : "false")
          << ",\"floor_gib\":" << std::max(m.o.host_commit_floor_gib, kHostPromoteFloorGiB) << ",\"max_gib\":" << m.o.host_promote_max_gib
          << ",\"layers\":" << pr.layers << ",\"gb\":" << double(pr.bytes) / 1e9 << ",\"promoted\":" << pr.promoted
          << ",\"demoted\":" << pr.demoted << "}";
    }
    if (m.streamer) {
        s << ",\"residency_plan\":\"" << m.residency_plan << "\"";
        s << ",\"slot_gb\":" << double(m.streamer->slot_bytes()) / 1e9 << ",\"group_experts\":" << m.o.xfer_group_experts
          << ",\"piece_mib\":" << m.o.xfer_piece_mib << ",\"resident_experts\":[";
        for (std::uint32_t l = m.o.first_layer; l < m.o.end_layer; ++l) s << (l > m.o.first_layer ? "," : "") << m.resident_experts[l];
        s << "]";
    }
    s << ",\"budget\":{\"auto\":" << (m.o.auto_budget || m.o.vram_cap_gib > 0.0 ? "true" : "false") << ",\"vram_cap_gib\":" << m.o.vram_cap_gib
      << ",\"dense_gib\":" << m.dense_gib << ",\"used_before_experts_gib\":" << m.used_before_experts_gib << ",\"free_at_plan_gib\":" << m.free_at_plan_gib
      << ",\"device_expert_gib\":" << m.expert_budget_gib << ",\"host_expert_gib\":" << m.o.host_expert_gib << "}";
    s << ",\"engram_upload\":\"" << m.o.engram_upload << "\",\"engram_cache_mib\":" << m.o.engram_cache_mib;
    s << weight_io_fields();
    s << ",\"gpu\":" << to_json(m.gpu) << ",\"code\":\"sm_" << m.dev.binary_version << "\",\"load_s\":" << m.load_seconds << "}";
    return s.str();
}

CudaPrefillResult CudaPrefill::run(std::span<const std::uint32_t> tokens) {
    if (impl_->o.pass_rows != 0U) return run_long(tokens, false);
    auto& m = *impl_;
    const CudaBatchItem item{tokens, m.o.fast_attention};
    // the solo run: context 0, the options' fast-attention flag (set_fast_attention)
    auto r = run_contexts(std::span<const CudaBatchItem>(&item, 1), std::vector<std::size_t>{0U});
    return std::move(r[0]);
}

CudaPrefillResult CudaPrefill::extend(std::span<const std::uint32_t> tokens) {
    require(impl_->o.pass_rows != 0U && impl_->seqs[0]->last_T != 0U, "extend needs pass_rows and an existing sequence");
    return run_long(tokens, true);
}

std::uint32_t CudaPrefill::positions() const noexcept { return impl_->seqs[0]->last_T; }

void CudaPrefill::request_cancel() noexcept { impl_->cancel_requested.store(true); }
void CudaPrefill::clear_cancel() noexcept { impl_->cancel_requested.store(false); }

void CudaPrefill::set_state_stream(std::shared_ptr<CudaPrefillStateStream> stream) {
    require(!impl_->long_request && (!stream || impl_->o.pass_rows != 0U), "state stream requires idle bounded passes");
    impl_->state_stream = std::move(stream);
}

CudaPrefillProgress CudaPrefill::progress() const noexcept {
    const auto& m = *impl_;
    CudaPrefillProgress p;
    p.position = m.progress_position.load();
    p.total = m.progress_total.load();
    p.cancel_pending = m.cancel_requested.load();
    const auto start = m.request_start_ns.load();
    if (start != 0) {
        const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        p.elapsed_seconds = double(now - start) * 1e-9;
    }
    return p;
}

CudaPrefillResult CudaPrefill::run_long(std::span<const std::uint32_t> tokens, bool continuation) {
    auto& m = *impl_;
    auto& q = *m.seqs[0];
    const auto& o = m.o;
    const auto started = std::chrono::steady_clock::now();
    const auto cached = continuation ? q.last_T : 0U;
    require(!tokens.empty() && tokens.size() <= o.max_positions - cached, "long prompt exceeds context capacity");
    for (const auto t : tokens) require(t < m.cfg.vocab_size, "token outside vocabulary");
    if (!ds41::host_commit_fits(ds41::host_memory_room(),
            ds41::long_request_host_reserve(cached + static_cast<std::uint32_t>(tokens.size())),
            static_cast<std::uint64_t>(o.host_commit_floor_gib * double(1ULL << 30))))
        throw CudaPrefillInterrupted("insufficient free commit before request allocation");
    m.request_tokens = continuation ? q.last_tokens : std::vector<std::uint32_t>{};
    m.request_tokens.insert(m.request_tokens.end(), tokens.begin(), tokens.end());
    m.request_cached = cached;
    m.request_total = static_cast<std::uint32_t>(m.request_tokens.size());
    std::uint32_t tail = 0;
    if (o.cone) {
        ds41::SuffixReplayGeometry geo;
        geo.draft_rows = o.draft_rows;
        m.request_replay = ds41::plan_suffix_replay(geo, cached, m.request_total);
        const auto shift = o.head_rows > 1U ? o.head_rows - 1U : 0U;
        const auto start = m.request_replay.encoder_output_begin;
        const auto from = start > shift ? std::max<std::uint64_t>(cached, start - shift) : cached;
        const auto need = m.request_total - from / o.window_rows * o.window_rows;
        if (tokens.size() > m.row_capacity) {
            require(need + o.window_rows <= m.row_capacity, "pass_rows too small for the global decoder suffix");
            tail = std::max(static_cast<std::uint32_t>(need), std::min(8192U, m.row_capacity - o.window_rows));
        }
    }
    m.passes = ds41::plan_prefill_passes(static_cast<std::uint32_t>(tokens.size()), o.pass_rows, o.window_rows, tail);
    m.hasher->reset();
    m.request_ids = m.hasher->hash_sequence(m.request_tokens);
    m.engram_ready.assign(m.passes.size(), 0U);
    m.engram_consumed = 0;
    m.engram_cancelled = false;
    m.engram_error = nullptr;
    m.engram_seconds = 0;
    const auto engram_before = m.engram_reader->stats();
    std::vector<std::uint32_t> uses;
    for (const auto& pass : m.passes) {
        const auto end = cached + pass.begin + pass.rows;
        for (std::uint32_t l = o.first_layer; l < o.end_layer; ++l) {
            std::uint64_t from = cached;
            if (o.cone) {
                const auto shift = o.head_rows > 1U ? o.head_rows - 1U : 0U;
                const auto begin = m.request_replay.layers[l].out_begin;
                from = begin > shift ? begin - shift : 0U;
                from = from / o.window_rows * o.window_rows;
            }
            if (from < end) uses.push_back(l);
        }
    }
    const auto stop_reader = [&] {
        {
            std::lock_guard<std::mutex> lock(m.engram_mutex);
            m.engram_cancelled = true;
        }
        m.engram_cv.notify_all();
        if (m.engram_thread.joinable()) m.engram_thread.join();
    };
    bool experts_started = false;
    const auto state_stream = m.state_stream;
    constexpr std::size_t transfer_chunk_bytes = 4U << 20U;
    struct TransferSlot {
        PinnedBuffer buffer;
        std::future<void> pending;
        ~TransferSlot() { if (pending.valid()) { try { pending.wait(); } catch (...) {} } }
    };
    std::array<TransferSlot, 2> transfer_slots;
    std::size_t transfer_slot = 0;
    std::vector<std::uint32_t> transferred_entries(m.plan.size(), 0U);
    const auto wait_transfer = [](TransferSlot& slot) {
        if (slot.pending.valid()) {
            try { slot.pending.get(); }
            catch (const std::exception& e) { throw CudaPrefillTransferError(std::string("state stream: ") + e.what()); }
        }
    };
    const auto transfer_bytes = [&](const std::string& name, const std::uint8_t* source, std::uint64_t offset, std::uint64_t bytes) {
        for (std::uint64_t done = 0; done < bytes;) {
            auto& slot = transfer_slots[transfer_slot++ % transfer_slots.size()];
            wait_transfer(slot);
            const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(transfer_chunk_bytes, bytes - done));
            STRIX_CUDA_CHECK(cudaMemcpyAsync(slot.buffer.get(), source + done, count, cudaMemcpyDeviceToHost, m.st));
            STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));
            try { slot.pending = state_stream->chunk(name, offset + done, {slot.buffer.get(), count}); }
            catch (const std::exception& e) { throw CudaPrefillTransferError(std::string("state stream: ") + e.what()); }
            if (!slot.pending.valid()) throw CudaPrefillTransferError("state stream returned an invalid completion future");
            done += count;
        }
    };
    m.long_request = true;
    m.progress_position.store(cached);
    m.progress_total.store(m.request_total);
    m.request_start_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(started.time_since_epoch()).count());
    try {
        if (m.cancel_requested.load()) throw CudaPrefillInterrupted("prefill cancelled before the first pass");
        if (state_stream) {
            for (auto& slot : transfer_slots) slot.buffer.resize(transfer_chunk_bytes);
            std::vector<CudaPrefillStateSection> sections;
            for (auto l = o.first_layer; l < o.end_layer; ++l) if (m.plan[l].owns_kv) {
                const auto entries = std::uint64_t{m.request_total / m.plan[l].ratio};
                sections.push_back({"kv." + std::to_string(l), entries * at::kKvEntryBytes});
                sections.push_back({"idx." + std::to_string(l), entries * at::kIdxEntryBytes});
            }
            try { state_stream->begin(m.request_total, sections); }
            catch (const std::exception& e) { throw CudaPrefillTransferError(std::string("state stream begin: ") + e.what()); }
        }
        if (!continuation) q.attn->reset(m.st);
        m.engram_thread = std::thread([&m, &q] {
            double read_seconds = 0;
            const auto cols = m.g.engram_cols;
            const auto layers = static_cast<std::uint32_t>(m.cfg.engram_layer_ids.size());
            try {
                for (std::size_t k = 0; k < m.passes.size(); ++k) {
                    {
                        std::unique_lock<std::mutex> lock(m.engram_mutex);
                        m.engram_cv.wait(lock, [&] { return m.engram_cancelled || k < 2U || m.engram_consumed > k - 2U; });
                        if (m.engram_cancelled) return;
                    }
                    const auto& pass = m.passes[k];
                    auto* dst = (k % 2U ? m.engram_ahead : q.engram_host).get();
                    for (std::uint32_t li = 0; li < layers; ++li) {
                        if (m.cfg.engram_layer_ids[li] < m.o.end_layer) {
                            const auto begin = std::chrono::steady_clock::now();
                            auto* out = dst + std::size_t{li} * pass.rows * cols * 264U;
                            if (m.o.engram == "zero") std::memset(out, 0, std::size_t{pass.rows} * cols * 264U);
                            else {
                                std::vector<std::uint64_t> offsets(std::size_t{pass.rows} * cols);
                                for (std::uint32_t t = 0; t < pass.rows; ++t) for (std::uint32_t c = 0; c < cols; ++c) {
                                    const auto position = m.request_cached + pass.begin + t;
                                    const auto row = m.request_ids[(std::size_t{position} * layers + li) * cols + c];
                                    if (row >= m.engram_tables[li].rows) throw std::runtime_error("ds41 cuda prefill: Engram row out of range");
                                    offsets[std::size_t{t} * cols + c] = m.engram_tables[li].file_offset + row * 264U;
                                }
                                m.engram_reader->read_records(offsets, 264U, out);
                            }
                            read_seconds += seconds_since(begin);
                        }
                        {
                            std::lock_guard<std::mutex> lock(m.engram_mutex);
                            m.engram_ready[k] = li + 1U;
                        }
                        m.engram_cv.notify_all();
                    }
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(m.engram_mutex);
                m.engram_error = std::current_exception();
                m.engram_cv.notify_all();
            }
            m.engram_seconds = read_seconds;
        });
        if (m.idle_held) release_idle_preload();  // defer_idle_preload: end the last run's held state first
        if (m.streamer) m.streamer->begin_run_uses(uses);
        else m.experts->begin_run_uses(uses);
        experts_started = true;
        CudaPrefillResult result;
        result.layers.assign(m.cfg.num_hidden_layers, {});
        for (std::size_t k = 0; k < m.passes.size(); ++k) {
            if (m.cancel_requested.load()) throw CudaPrefillInterrupted("prefill cancelled at pass boundary");
            if (!ds41::host_commit_fits(ds41::host_memory_room(), ds41::long_request_host_reserve(m.request_total),
                    static_cast<std::uint64_t>(o.host_commit_floor_gib * double(1ULL << 30))))
                throw CudaPrefillInterrupted("insufficient free commit at pass boundary");
            m.pass_index = k;
            const auto& pass = m.passes[k];
            m.pass_base = cached + pass.begin;
            const CudaBatchItem item{tokens.subspan(pass.begin, pass.rows), o.fast_attention};
            auto r = run_contexts(std::span<const CudaBatchItem>(&item, 1), {0U})[0];
            m.progress_position.store(m.pass_base + pass.rows);
            if (state_stream) {
                for (auto l = o.first_layer; l < o.end_layer; ++l) if (m.plan[l].owns_kv) {
                    const auto end = (m.pass_base + pass.rows) / m.plan[l].ratio;
                    const auto begin = transferred_entries[l];
                    const auto& caches = q.attn->caches(l);
                    transfer_bytes("kv." + std::to_string(l), caches.kv_cache + std::uint64_t{begin} * at::kKvEntryBytes,
                                   std::uint64_t{begin} * at::kKvEntryBytes, std::uint64_t{end - begin} * at::kKvEntryBytes);
                    transfer_bytes("idx." + std::to_string(l), caches.idx_cache + std::uint64_t{begin} * at::kIdxEntryBytes,
                                   std::uint64_t{begin} * at::kIdxEntryBytes, std::uint64_t{end - begin} * at::kIdxEntryBytes);
                    transferred_entries[l] = end;
                }
            }
            result.row_layers += r.row_layers;
            result.enqueue_seconds += r.enqueue_seconds;
            for (const auto& [name, ms] : r.ops) result.ops[name] += ms;
            for (std::size_t l = 0; l < r.layers.size(); ++l) {
                auto& a = result.layers[l]; const auto& b = r.layers[l];
                a.rows += b.rows; a.total_ms += b.total_ms; a.attention_ms += b.attention_ms; a.moe_ms += b.moe_ms;
                a.expert_wait_ms += b.expert_wait_ms; a.expert_wait_device_ms += b.expert_wait_device_ms;
            }
            if (k + 1U == m.passes.size()) {
                result.logits = std::move(r.logits); result.head_logits = std::move(r.head_logits); result.next_token = r.next_token;
                result.stream_hashes = std::move(r.stream_hashes);
                result.expert_bytes_streamed = r.expert_bytes_streamed; result.expert_io_seconds = r.expert_io_seconds;
                result.file_bytes_streamed = r.file_bytes_streamed;
                result.host_promoted_layers = r.host_promoted_layers; result.host_demoted_layers = r.host_demoted_layers;
                result.host_promoted_total = r.host_promoted_total; result.host_promoted_bytes = r.host_promoted_bytes;
                result.expert_copy_busy_s = r.expert_copy_busy_s; result.expert_stall_ms = r.expert_stall_ms;
            }
            {
                std::lock_guard<std::mutex> lock(m.engram_mutex);
                m.engram_consumed = k + 1U;
            }
            m.engram_cv.notify_all();
        }
        stop_reader();
        for (auto& slot : transfer_slots) wait_transfer(slot);
        if (m.engram_error) std::rethrow_exception(m.engram_error);
        result.engram_seconds = m.engram_seconds;
        const auto engram_after = m.engram_reader->stats();
        result.engram_records = engram_after.records - engram_before.records;
        result.engram_pages = engram_after.pages - engram_before.pages;
        result.engram_page_hits = engram_after.page_hits - engram_before.page_hits;
        result.seconds = result.batch_seconds = seconds_since(started);
        if (m.streamer) m.streamer->end_run(); else m.experts->end_run();
        m.long_request = false;
        m.request_start_ns.store(0);
        m.cancel_requested.store(false);
        m.request_ids.clear();
        m.request_tokens.clear();
        return result;
    } catch (...) {
        stop_reader();
        if (experts_started) { if (m.streamer) m.streamer->end_run(); else m.experts->end_run(); }
        m.long_request = false;
        m.request_start_ns.store(0);
        m.cancel_requested.store(false);
        q.last_T = q.last_rows = 0;
        q.last_tokens.clear();
        m.last_batch_ctx.clear();
        throw;
    }
}

std::vector<CudaPrefillResult> CudaPrefill::run_batch(std::span<const CudaBatchItem> items) {
    auto& m = *impl_;
    require(m.o.pass_rows == 0U, "run_batch needs pass_rows 0");
    require(!items.empty() && items.size() <= m.seqs.size(), "batch of 1..contexts prompts");
    // contexts by capacity: the longest prompt takes the smallest free context that holds it (context 0: max_positions)
    std::vector<std::size_t> order(items.size()), ctx(items.size());
    for (std::size_t i = 0; i < items.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return items[a].tokens.size() > items[b].tokens.size(); });
    std::vector<bool> used(m.seqs.size(), false);
    for (const auto i : order) {
        std::size_t best = m.seqs.size();
        for (std::size_t k = 0; k < m.seqs.size(); ++k)
            if (!used[k] && m.seqs[k]->capacity >= items[i].tokens.size() &&
                (best == m.seqs.size() || m.seqs[k]->capacity < m.seqs[best]->capacity))
                best = k;
        require(best < m.seqs.size(), "no batch context large enough for the prompt (batch_positions)");
        used[best] = true;
        ctx[i] = best;
    }
    return run_contexts(items, ctx);
}

std::vector<CudaPrefillResult> CudaPrefill::run_contexts(std::span<const CudaBatchItem> items, const std::vector<std::size_t>& ctx) {
    auto& m = *impl_;
    require(!m.o.decode_only, "run() on a decode_only object (it has no prefill buffers)");
    require(m.sequence == 0U, "run() on sequence slot 0 only (its prefill graphs read slot 0's attention state)");
    if (m.dec) m.dec->drain_engram();  // a prefetch of the previous sequence's next token
    if (m.decode_mode) {  // undo decode_residency(): the run's tiers and buffers (the decode graphs held their addresses)
        STRIX_CUDA_CHECK(cudaDeviceSynchronize());
        m.graphs.clear();
        if (m.dec) m.dec->graphs.clear();
        m.experts->clear_decode();
        m.tiers = m.run_tiers;
        if (m.run_buffers_released) {
            alloc_run_buffers(m);
            m.graphs.clear();
        }
        m.decode_mode = false;
    }
    const auto& o = m.o;
    const auto& c = m.cfg;
    const auto& g = m.g;
    const auto t_run = std::chrono::steady_clock::now();
    const std::size_t B = items.size();
    const bool long_mode = m.long_request;
    const auto base = long_mode ? m.pass_base : 0U;
    const bool final_pass = !long_mode || base + items[0].tokens.size() == m.request_total;
    const std::uint32_t H = c.hidden_size, K = c.num_experts_per_tok, QA = c.q_lora_rank, D = at::kHeadDim;
    const std::uint32_t QH = c.num_attention_heads * c.head_dim, OL = c.o_groups * c.o_lora_rank;
    const std::uint32_t C = o.chunk_rows, W = o.window_rows, MC = o.moe_chunk_rows, QKV = QA + D;
    const std::size_t SR = std::size_t{4} * H;  // stream row
    cudaStream_t st = m.st;
    require(o.first_layer == 0U, "a prefill runs layers [0, end) (a later range continues with step())");
    require(B >= 1U && ctx.size() == B, "batch contexts");
    require(B == 1U || (m.engram_st == nullptr && !o.graphs), "a batch needs the mapped Engram upload and no graphs");
    std::vector<CudaPrefillResult> results(B);
    CudaPrefillResult& result = results[0];  // the batch-wide figures (timing, transport) are filled here, copied at the end
    for (auto& r : results) r.layers.assign(c.num_hidden_layers, {});
    std::vector<std::ofstream> route_files;
    if (!o.moe_routes_out.empty()) {
        route_files.resize(B);
        for (std::size_t i = 0; i < B; ++i) {
            const auto path = o.moe_routes_out + (i ? "." + std::to_string(i) : "");
            route_files[i].open(path, std::ios::trunc);
            require(static_cast<bool>(route_files[i]), "cannot open MoE route trace");
        }
    }

    // rows per layer: x0 attention input, k0 window K/V, o0 output (cone: rounded down to the attention window)
    struct Rows { std::uint32_t x0 = 0, k0 = 0, o0 = 0; };
    // one prompt of the closed batch: its context, rows, Engram ids, fast-attention flag
    struct SeqRun {
        Impl::Seq* q = nullptr;
        std::span<const std::uint32_t> tokens;
        std::uint32_t T = 0;
        bool fa = false;
        std::vector<Rows> rows;
        std::vector<std::uint64_t> ids_all;
        std::uint32_t attn_input_from = 0;
    };
    std::vector<SeqRun> runs(B);
    for (std::size_t i = 0; i < B; ++i) {
        auto& sr = runs[i];
        require(ctx[i] < m.seqs.size(), "batch context");
        for (std::size_t j = 0; j < i; ++j) require(ctx[j] != ctx[i], "one prompt per batch context");
        sr.q = m.seqs[ctx[i]].get();
        sr.tokens = items[i].tokens;
        sr.T = static_cast<std::uint32_t>(sr.tokens.size());
        require(sr.T >= 1U && sr.T <= sr.q->capacity && sr.T <= m.row_capacity, "rows 1..the context and row capacity");
        for (const auto t : sr.tokens) require(t < c.vocab_size, "token outside the vocabulary");
        sr.fa = items[i].fast_attention && m.win16.get() != nullptr;  // its buffers exist only with fast_attention at construction
        sr.attn_input_from = sr.T;
        sr.rows.assign(c.num_hidden_layers, {});
        if (o.cone) {
            ds41::SuffixReplayGeometry geo;
            geo.layers = c.num_hidden_layers;
            geo.draft_rows = o.draft_rows;
            const auto plan = long_mode ? m.request_replay : ds41::plan_suffix_replay(geo, 0U, sr.T);
            const auto down = [&](std::uint64_t v) {
                const auto aligned = v / W * W;
                return aligned <= base ? 0U : static_cast<std::uint32_t>(std::min<std::uint64_t>(sr.T, aligned - base));
            };
            // head_rows > 1: the logits of the last H rows must be exact too: the cone of the last row shifted H - 1 rows
            // earlier on every decoder layer (window dependencies are translation invariant)
            const std::uint64_t shift = o.head_rows > 1U ? o.head_rows - 1U : 0U;
            const auto back = [&](std::uint64_t v) { return v > shift ? v - shift : 0U; };
            for (const auto& pl : plan.layers) {
                auto& r = sr.rows[pl.layer];
                const bool dec = pl.out_begin > 0U || pl.kv_begin > 0U;
                r.o0 = down(dec ? back(pl.out_begin) : pl.out_begin);
                r.k0 = down(dec ? back(pl.kv_begin) : pl.kv_begin);
                if (pl.layer >= geo.first_decoder)
                    bound_decoder_replay(sr.T, o.replay_rows, W, r.o0, r.k0);
                r.x0 = m.plan[pl.layer].owns_kv ? 0U : r.k0;
            }
        }
        if (o.pass_rows != 0U && o.cone) sr.q->attn->set_decoder_row0(sr.rows[20].o0);
        for (std::uint32_t l = o.first_layer; l < o.end_layer; ++l) results[i].row_layers += sr.T - sr.rows[l].o0;
    }

    // Engram rows of every token of every prompt, read on a thread (needed at the first Engram layer)
    const std::uint32_t ecols = g.engram_cols, elayers = static_cast<std::uint32_t>(c.engram_layer_ids.size());
    if (!long_mode) for (auto& sr : runs) {
        m.hasher->reset();
        sr.ids_all = m.hasher->hash_sequence(sr.tokens);  // [token][layer][column]
    }
    if (B > 1U) {
        // the hasher keeps the Engram history of context 0, the sequence step() continues: its prompt if it is in the
        // batch, else the tokens it already holds
        const SeqRun* r0 = nullptr;
        for (const auto& sr : runs)
            if (sr.q == m.seqs[0].get()) r0 = &sr;
        m.hasher->reset();
        (void)m.hasher->hash_sequence(r0 ? r0->tokens : std::span<const std::uint32_t>(m.seq0().last_tokens));
    }
    std::vector<std::uint32_t> engram_li(c.num_hidden_layers, ~0U);
    for (std::uint32_t li = 0; li < elayers; ++li) engram_li[c.engram_layer_ids[li]] = li;
    const auto engram_stats0 = m.engram_reader->stats();
    if (!long_mode) {
        for (auto& sr : runs) {
            const std::size_t need = std::size_t{elayers} * sr.T * ecols * 264U;
            if (sr.q->engram_host.size() < need) sr.q->engram_host.resize(need, true);
        }
        if (m.engram_thread.joinable()) m.engram_thread.join();
        m.engram_error = nullptr;
        m.engram_layers_done = 0;
        // one Engram layer after the other (the first is needed at layer 1, the second only at layer 14), every prompt
        // of the batch in each
        struct EngramJob {
            std::uint8_t* host;
            const std::vector<std::uint64_t>* ids;
            std::uint32_t T;
        };
        std::vector<EngramJob> jobs;
        for (auto& sr : runs) jobs.push_back({sr.q->engram_host.get(), &sr.ids_all, sr.T});
        m.engram_thread = std::thread([&m, jobs, ecols, elayers] {
            const auto t0 = std::chrono::steady_clock::now();
            try {
                std::vector<std::uint8_t> combined;  // a batch: every prompt's rows of a layer in one read (one queue)
                for (std::uint32_t li = 0; li < elayers; ++li) {
                    if (jobs.size() > 1U && m.o.engram == "file") {
                        std::vector<std::uint64_t> offsets;
                        for (const auto& job : jobs)
                            for (std::uint32_t t = 0; t < job.T; ++t)
                                for (std::uint32_t k = 0; k < ecols; ++k) {
                                    const std::uint64_t row = (*job.ids)[(std::size_t{t} * elayers + li) * ecols + k];
                                    if (row >= m.engram_tables[li].rows) throw std::runtime_error("ds41 cuda prefill: Engram row out of range");
                                    offsets.push_back(m.engram_tables[li].file_offset + row * 264U);
                                }
                        combined.resize(offsets.size() * 264U);
                        m.engram_reader->read_records(offsets, 264U, combined.data());
                        std::size_t at = 0;
                        for (const auto& job : jobs) {
                            const std::size_t bytes = std::size_t{job.T} * ecols * 264U;
                            std::memcpy(job.host + std::size_t{li} * bytes, combined.data() + at, bytes);
                            at += bytes;
                        }
                        std::lock_guard<std::mutex> lock(m.engram_mutex);
                        m.engram_layers_done = li + 1U;
                        m.engram_cv.notify_all();
                        continue;
                    }
                    for (const auto& job : jobs) {
                        const std::uint32_t T = job.T;
                        const auto& ids_all = *job.ids;
                        std::uint8_t* out = job.host + std::size_t{li} * T * ecols * 264U;
                        if (m.o.engram == "zero") {
                            std::memset(out, 0, std::size_t{T} * ecols * 264U);
                        } else {
                            std::vector<std::uint64_t> offsets(std::size_t{T} * ecols);
                            for (std::uint32_t t = 0; t < T; ++t)
                                for (std::uint32_t k = 0; k < ecols; ++k) {
                                    const std::uint64_t row = ids_all[(std::size_t{t} * elayers + li) * ecols + k];
                                    if (row >= m.engram_tables[li].rows) throw std::runtime_error("ds41 cuda prefill: Engram row out of range");
                                    offsets[std::size_t{t} * ecols + k] = m.engram_tables[li].file_offset + row * 264U;
                                }
                            m.engram_reader->read_records(offsets, 264U, out);
                        }
                        if (m.engram_st) {
                            // the layer's rows to the device now, on their own high-priority stream (B == 1)
                            const std::size_t bytes = std::size_t{T} * ecols * 264U;
                            STRIX_CUDA_CHECK(cudaMemcpyAsync(m.engram_dev.get() + std::size_t{li} * bytes, out, bytes, cudaMemcpyHostToDevice, m.engram_st));
                            STRIX_CUDA_CHECK(cudaEventRecord(m.engram_ev[li], m.engram_st));
                        }
                    }
                    std::lock_guard<std::mutex> lock(m.engram_mutex);
                    m.engram_layers_done = li + 1U;
                    m.engram_cv.notify_all();
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(m.engram_mutex);
                m.engram_error = std::current_exception();
                m.engram_layers_done = elayers;
                m.engram_cv.notify_all();
            }
            m.engram_seconds = seconds_since(t0);
        });
    }
    bool engram_joined = false;
    const auto wait_engram = [&](std::uint32_t li) {
        std::unique_lock<std::mutex> lock(m.engram_mutex);
        m.engram_cv.wait(lock, [&] { return m.engram_error || (long_mode ? m.engram_ready[m.pass_index] > li : m.engram_layers_done > li); });
        if (m.engram_error) std::rethrow_exception(m.engram_error);
    };
    const auto join_engram = [&] {
        if (engram_joined || long_mode) return;
        m.engram_thread.join();
        engram_joined = true;
        if (m.engram_error) std::rethrow_exception(m.engram_error);
    };

    // accounting (timing): the run's first and last device events
    std::unique_ptr<Event> run_first, run_last;
    if (o.timing) {
        run_first = std::make_unique<Event>(true);
        run_last = std::make_unique<Event>(true);
        result.host_setup_s = seconds_since(t_run);
        STRIX_CUDA_CHECK(cudaEventRecord(run_first->e, st));
    }
    if (!long_mode) {  // a bounded pass continues the request's sequence and expert schedule (run_long)
        for (auto& sr : runs) sr.q->attn->reset(st);
        if (m.idle_held) release_idle_preload();
        if (m.streamer) m.streamer->begin_run();
        else m.experts->begin_run();
    }

    // The prompt the layer body below works on (bind(i)): every name the body reads per prompt.
    Impl::Seq* Q = runs[0].q;
    std::uint32_t T = runs[0].T;
    bool fa = runs[0].fa;
    float* stream = Q->stream.get();
    const std::vector<Rows>* rows_of = &runs[0].rows;
    std::uint32_t attn_input_from = runs[0].T;
    std::size_t cur = 0;
    const auto bind = [&](std::size_t i) {
        runs[cur].attn_input_from = attn_input_from;
        cur = i;
        Q = runs[i].q;
        T = runs[i].T;
        fa = runs[i].fa;
        stream = Q->stream.get();
        rows_of = &runs[i].rows;
        attn_input_from = runs[i].attn_input_from;
    };

    // graph runs: the layer segments after the warm-up runs (captured on the first graph run of a T)
    const bool use_graphs = o.graphs && m.runs_done >= o.graph_warmup_runs;
    result.graphs = use_graphs;
    auto& G = m.graphs;
    if (use_graphs) {
        const void* ebase = Q->engram_host.get();
        if (G.T != T || G.engram != ebase) {
            G.clear();
            G.T = T;
            G.engram = ebase;
            G.a.assign(c.num_hidden_layers, nullptr);
            G.b.assign(c.num_hidden_layers, nullptr);
            G.b_weights.assign(c.num_hidden_layers, {});
        }
    }
    bool capturing = false;
    // runs body() on the stream, or (graph runs) as the captured graph `exec` (captured now if absent)
    const auto segment = [&](cudaGraphExec_t& exec, auto&& body) {
        if (!use_graphs) {
            body();
            return;
        }
        if (exec == nullptr) {
            STRIX_CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            capturing = true;
            try {
                body();
            } catch (...) {
                cudaGraph_t broken = nullptr;
                cudaStreamEndCapture(st, &broken);
                if (broken) cudaGraphDestroy(broken);
                capturing = false;
                throw;
            }
            capturing = false;
            cudaGraph_t graph = nullptr;
            STRIX_CUDA_CHECK(cudaStreamEndCapture(st, &graph));
            const cudaError_t inst = cudaGraphInstantiate(&exec, graph, 0ULL);
            cudaGraphDestroy(graph);
            STRIX_CUDA_CHECK(inst);
            ++result.graph_captures;
        }
        STRIX_CUDA_CHECK(cudaGraphLaunch(exec, st));
    };

    // timing events
    std::vector<std::unique_ptr<Event>> ev;
    const auto mark = [&]() -> cudaEvent_t {
        if (!o.timing || capturing) return nullptr;
        ev.push_back(std::make_unique<Event>(true));
        STRIX_CUDA_CHECK(cudaEventRecord(ev.back()->e, st));
        return ev.back()->e;
    };
    struct LayerMarks { cudaEvent_t begin = nullptr, attn_end = nullptr, moe_begin = nullptr, wait_end = nullptr, end = nullptr; };
    // op profile (timing): an event before every op group; the time to the next event is the op's
    std::vector<std::pair<const char*, cudaEvent_t>> op_marks;
    const bool op_events = o.timing && o.timing_ops && !use_graphs;
    const auto op = [&](const char* name) {
        if (!op_events) return;
        ev.push_back(std::make_unique<Event>(true));
        STRIX_CUDA_CHECK(cudaEventRecord(ev.back()->e, st));
        op_marks.emplace_back(name, ev.back()->e);
    };
    std::vector<LayerMarks> marks(c.num_hidden_layers);
    // accounting (timing, eager runs): events around each streamed range's wait, by layer
    std::vector<std::vector<std::pair<cudaEvent_t, cudaEvent_t>>> range_waits(c.num_hidden_layers);
    const auto timed_ranges = [&](const std::vector<MoeRange>& in, std::uint32_t layer) {
        std::vector<MoeRange> out = in;
        if (!o.timing || use_graphs) return out;
        for (auto& r : out) {
            if (!r.ready || r.first >= r.end) continue;
            ev.push_back(std::make_unique<Event>(true));
            r.wait_begin = ev.back()->e;
            ev.push_back(std::make_unique<Event>(true));
            r.wait_end = ev.back()->e;
            range_waits[layer].emplace_back(r.wait_begin, r.wait_end);
        }
        return out;
    };

    // ---- embedding (layer 0's stream; pre_carry identity: the first collapse takes pre = null)
    if (m.has_embed())
        for (std::size_t i = 0; i < B; ++i) {
            bind(i);
            DeviceBuffer<std::uint32_t> dtok(T);
            STRIX_CUDA_CHECK(cudaMemcpyAsync(dtok.get(), runs[i].tokens.data(), std::size_t{T} * 4U, cudaMemcpyHostToDevice, st));
            F16Weights table{m.file->device_f16("token_embd.weight"), c.vocab_size, H};
            enqueue_embed({table, dtok.get(), T, stream, nullptr}, st);
            STRIX_CUDA_CHECK(cudaStreamSynchronize(st));
        }
    bind(0);
    const auto a8_at = [](const A8Buffer& b, std::uint32_t row, std::uint32_t rows) {
        A8Rows v = b.view();
        v.codes += std::size_t{row} * v.stride;
        v.deltas += std::size_t{row} * v.stride / 32U;
        v.sums += std::size_t{row} * v.stride / 32U;
        v.rows = rows;
        return v;
    };
    const auto idx_at = [&](const at::AttentionState::Caches& cache, std::uint32_t row) {
        require(cache.idx && row >= cache.row0 && row - cache.row0 < cache.rows, "index row outside allocation");
        return cache.idx + std::size_t{row - cache.row0} * at::kTopK;
    };
    const auto count_at = [&](const at::AttentionState::Caches& cache, std::uint32_t row) {
        require(cache.idx_count && row >= cache.row0 && row - cache.row0 < cache.rows, "index count row outside allocation");
        return cache.idx_count + row - cache.row0;
    };
    const auto cand_at = [&](std::uint32_t row) {
        if (!Q->attn->candidates()) return static_cast<std::uint8_t*>(nullptr);
        require(row >= Q->attn->candidate_row0(), "candidate row before compact origin");
        return Q->attn->candidates() + std::size_t{row - Q->attn->candidate_row0()} * Q->attn->cand_stride();
    };
    const auto chunks = [&](std::uint32_t from, std::uint32_t size, auto&& body) {
        for (std::uint32_t r0 = from; r0 < T; r0 += size) body(r0, std::min(size, T - r0));
    };
    // prefill-shape kernels for calls of many rows (same bits as the decode kernels)
    const auto pk_rows = [&](std::uint32_t n) { return o.prefill_kernels && n >= o.prefill_kernels_min_rows; };
    // many-row Q8_0 x A8 sites (same bits as the tiled kernel)
    namespace fdn = ::strix::ds41::cuda::fast;
    const bool fd = o.fast_dense;
    const auto q8 = [&](const Q8A8View& v) {
        if (fd) fdn::enqueue_q8_0_a8(v, st);
        else if (o.prefill_q8 && v.x.rows >= o.prefill_q8_min_rows) pk::enqueue_q8_0_a8_rows(v, st);
        else enqueue_q8_0_a8(v, st);
    };
    const auto f16_rows = [&](const at::F16RowsView& v) {
        if (!pk_rows(v.rows)) return at::enqueue_f16_rows(v, st);
        pk::LaneDotView p;
        p.rows = v.rows;
        p.job_count = v.job_count;
        for (std::uint32_t j = 0; j < v.job_count; ++j) {
            const auto& a = v.jobs[j];
            p.jobs[j] = pk::LaneDotJob{a.weight, a.width, a.count, a.x, a.x_stride, a.out, a.out_stride};
        }
        pk::enqueue_lane_dot(p, st);
    };

    // fast dense: one pass over rows [r0, T) of hc_post (emix set: stream = expand(y, stream, emix)), the next site's
    // collapse (pre) + RMSNorm (weight) -> x, xa, and its hc mixes (fn, base, scale) -> mixes
    const auto hc_pass = [&](std::uint32_t r0, const float* emix, const float* pre, const float* weight, const std::uint16_t* fn,
                             const float* base, const float* scale, float* mixes) {
        if (r0 >= T) return;
        fdn::HcFusedView f;
        f.mix.stream = stream + std::size_t{r0} * SR;
        f.mix.rows = T - r0;
        f.mix.hidden = H;
        f.mix.fn = fn;
        f.mix.base = base;
        f.mix.scale = scale;
        f.mix.iters = g.hc_iters;
        f.mix.hc_eps = g.hc_eps;
        f.mix.mixes = mixes + std::size_t{r0} * 24U;
        if (emix) {
            f.sub = Q->y.get() + std::size_t{r0} * H;
            f.emix = emix + std::size_t{r0} * 24U;
        }
        f.pre = pre ? pre + std::size_t{r0} * 24U : nullptr;
        f.pre_stride = 24;
        f.weight = weight;
        f.x = Q->x.get() + std::size_t{r0} * H;
        f.a8 = a8_at(Q->xa, r0, T - r0);
        fdn::enqueue_hc_fused(f, st);
    };
    // fast dense: rows [from, T) of the next layer's attention input (mix_attn, x, xa) made by the last pass of a layer
    // (attn_input_from, per prompt: bind())

    for (std::uint32_t layer = o.first_layer; layer < o.end_layer; ++layer) {
        bool any_input = false, any_output = false;
        for (const auto& sr : runs) {
            const auto& a = sr.rows[layer];
            any_input = any_input || std::min({a.x0, a.k0, a.o0}) < sr.T;
            any_output = any_output || a.o0 < sr.T;
        }
        if (!any_input) continue;
        const auto& lw = m.core->weights(layer);
        const auto& r = m.plan[layer];
        const auto& aw = m.attn_w[layer];
        auto& mk = marks[layer];
        mk.begin = mark();
        // host wait for this layer's Engram rows (outside the segment graphs; every prompt's)
        if (engram_li[layer] != ~0U) wait_engram(engram_li[layer]);
        cudaGraphExec_t no_graph = nullptr;
        // layer-major closed batch: segment A of every prompt, ONE acquire of the layer's experts, segment B of every
        // prompt, ONE release (each prompt's kernels and their order are the solo run's)
        for (std::size_t si = 0; si < B; ++si) {
        bind(si);
        const auto& caches = Q->attn->caches(layer);
        const Rows R0 = (*rows_of)[layer];
        const std::uint32_t o0 = R0.o0;
        // pre of this layer's attention collapse: the previous layer's ffn mix (identity on the first layer of a
        // fresh embedding)
        const float* pre_attn = layer == 0U ? nullptr : Q->mix_ffn.get();
        segment(use_graphs ? G.a[layer] : no_graph, [&] {

        // ---- Engram (encoder layers: every row)
        if (engram_li[layer] != ~0U) {
            op("engram");
            // rows read by the kernel from mapped host memory: a copy would queue behind the streamed experts' DMA
            const std::uint8_t* base = (long_mode && m.pass_index % 2U ? m.engram_ahead : Q->engram_host).device() + std::size_t{engram_li[layer]} * T * ecols * 264U;
            if (m.engram_st) {
                STRIX_CUDA_CHECK(cudaStreamWaitEvent(st, m.engram_ev[engram_li[layer]], 0));
                base = m.engram_dev.get() + std::size_t{engram_li[layer]} * T * ecols * 264U;
            }
            chunks(0U, C, [&](std::uint32_t r0, std::uint32_t n) {
                EngramView e;
                e.row_bytes = base + std::size_t{r0} * ecols * 264U;
                e.rows = n;
                e.cols = ecols;
                e.hidden = H;
                e.kv = {lw.engram_kv, 5U * H, ecols * 256U};
                e.q_norm = lw.engram_q_norm;
                e.k_norm = lw.engram_k_norm;
                e.stream = stream + std::size_t{r0} * SR;
                e.workspace = m.ws.get();
                if (pk_rows(n)) pk::enqueue_engram(e, st);
                else enqueue_engram(e, st);
            });
        }
        // ---- attention input: hc mix (output rows), collapse + norm (input rows from x0)
        op("hc_mix_attn");
        if (fd) {
            require(attn_input_from <= o0 || attn_input_from == T, "fast dense: attention input rows");
            if (attn_input_from > o0) hc_pass(o0, nullptr, pre_attn, lw.attn_norm, lw.hc_attn_fn, lw.hc_attn_base, lw.hc_attn_scale, Q->mix_attn.get());
            op("norm_attn");
            // input rows below the output rows: the norm only
            const std::uint32_t n_end = std::min(o0, attn_input_from);
            if (R0.x0 < n_end) {
                NormView v;
                v.hc_stream = stream + std::size_t{R0.x0} * SR;
                v.pre = pre_attn ? pre_attn + std::size_t{R0.x0} * 24U : nullptr;
                v.pre_stride = 24;
                v.weight = lw.attn_norm;
                v.rows = n_end - R0.x0;
                v.n = H;
                v.out = Q->x.get() + std::size_t{R0.x0} * H;
                v.a8 = a8_at(Q->xa, R0.x0, n_end - R0.x0);
                fdn::enqueue_norm(v, st);
            }
        } else {
        chunks(o0, C, [&](std::uint32_t r0, std::uint32_t n) {
            HcMixView v;
            v.stream = stream + std::size_t{r0} * SR;
            v.rows = n;
            v.hidden = H;
            v.fn = lw.hc_attn_fn;
            v.base = lw.hc_attn_base;
            v.scale = lw.hc_attn_scale;
            v.iters = g.hc_iters;
            v.hc_eps = g.hc_eps;
            v.mixes = Q->mix_attn.get() + std::size_t{r0} * 24U;
            v.partial = reinterpret_cast<float*>(m.ws.get());
            enqueue_hc_mixes(v, st);
        });
        op("norm_attn");
        chunks(R0.x0, C, [&](std::uint32_t r0, std::uint32_t n) {
            NormView v;
            v.hc_stream = stream + std::size_t{r0} * SR;
            v.pre = pre_attn ? pre_attn + std::size_t{r0} * 24U : nullptr;
            v.pre_stride = 24;
            v.weight = lw.attn_norm;
            v.rows = n;
            v.n = H;
            v.out = Q->x.get() + std::size_t{r0} * H;
            v.a8 = a8_at(Q->xa, r0, n);
            enqueue_norm(v, st);
        });
        }
        attn_input_from = T;
        // ---- q_a + kv (rows from k0), q_a_norm (rows from o0)
        op("qa_kv");
        chunks(R0.k0, fd ? T : C, [&](std::uint32_t r0, std::uint32_t n) {
            Q8A8View v;
            v.tiled = lw.qa_kv.view();
            v.count = QKV;
            v.split = 1;
            v.x = a8_at(Q->xa, r0, n);
            v.out = Q->qkv.get() + std::size_t{r0} * QKV;
            q8(v);
            // the window kernel reads contiguous 512-value kv rows
            STRIX_CUDA_CHECK(cudaMemcpy2DAsync(Q->kvrows.get() + std::size_t{r0} * D, D * 4U, Q->qkv.get() + std::size_t{r0} * QKV + QA, QKV * 4U,
                                               D * 4U, n, cudaMemcpyDeviceToDevice, st));
        });
        op("q_norm");
        chunks(o0, fd ? T : C, [&](std::uint32_t r0, std::uint32_t n) {
            NormView v;
            v.x = Q->qkv.get() + std::size_t{r0} * QKV;
            v.x_stride = QKV;
            v.weight = lw.q_a_norm;
            v.rows = n;
            v.n = QA;
            v.out = Q->qr.get() + std::size_t{r0} * QA;
            v.a8 = a8_at(Q->qa, r0, n);
            if (fd) fdn::enqueue_norm(v, st);
            else enqueue_norm(v, st);
        });

        // ---- attention: window K/V (rows from k0) and compressor (rows from x0) in row order
        const at::RopeRow* rope = Q->attn->rope().table(r.rope);
        const std::uint64_t* pos = m.pos64.get() + base;
        for (std::uint32_t w0 = std::min(R0.x0, R0.k0); w0 < T; w0 += C) {
            const std::uint32_t n = std::min(C, T - w0);
            if (r.owns_kv && w0 + n > R0.x0) {
                op("compress_proj");
                const std::uint32_t a0 = std::max(w0, R0.x0), an = w0 + n - a0;
                at::F16RowsView v{};
                v.rows = an;
                v.jobs[v.job_count++] = at::F16Job{aw.compressor_kv, H, D, Q->x.get() + std::size_t{a0} * H, H, m.ckv.get(), D};
                if (r.ratio == 2U) v.jobs[v.job_count++] = at::F16Job{aw.compressor_gate, H, D, Q->x.get() + std::size_t{a0} * H, H, m.cscore.get(), D};
                f16_rows(v);
            }
            op("window_compress");
            for (std::uint32_t s0 = w0; s0 < w0 + n; s0 += W) {
                const std::uint32_t sn = std::min(W, w0 + n - s0);
                if (s0 >= R0.k0)
                    at::enqueue_window_write(at::WindowView{Q->kvrows.get() + std::size_t{s0} * D, aw.kv_norm, rope, pos + s0, sn,
                                                            Q->attn->ring_capacity(), caches.ring},
                                             st);
                if (r.owns_kv && s0 >= R0.x0) {
                    at::CompressView v{};
                    v.ratio = r.ratio;
                    v.rows = sn;
                    v.kv = m.ckv.get() + std::size_t{s0 - std::max(w0, R0.x0)} * D;
                    v.score = m.cscore.get() + std::size_t{s0 - std::max(w0, R0.x0)} * D;
                    v.norm = aw.compressor_norm;
                    v.wk = aw.index_k;
                    v.knorm = aw.index_k_norm;
                    v.rope = Q->attn->rope().table(ds41::attn::RopeKind::compressed);
                    v.pos = pos + s0;
                    v.state = caches.carry;
                    v.state_valid = caches.carry_valid;
                    v.kv_cache = caches.kv_cache;
                    v.idx_cache = caches.idx_cache;
                    at::enqueue_compress(v, st);
                }
            }
        }
        // ---- fast attention: the layer's window rows, its KV source's entries (FP16) and index keys (FP8), once
        if (fa && r.ratio != 0U) {
            op("fast_kv");
            const auto& src = m.plan[static_cast<std::size_t>(r.kv_source)];
            const auto& sc = Q->attn->caches(static_cast<std::uint32_t>(r.kv_source));
            const std::uint32_t count = (base + T) / src.ratio;
            at::FastKvView kv{};
            kv.kv_cache = sc.kv_cache;
            kv.kv_count = count;
            kv.kv16 = m.kv16.get();
            at::enqueue_fast_kv(kv, st);
            if (r.owns_index && m.fast_index) at::enqueue_fast_index_keys({sc.idx_cache, count, m.k8.get(), m.kscale.get()}, st);
        }
        const std::uint32_t win_first = base + o0 >= at::kWindow ? base + o0 - (at::kWindow - 1U) : 0U;
        if (fa) {
            at::FastKvView kv{};
            kv.ring = caches.ring;
            kv.capacity = Q->attn->ring_capacity();
            kv.first = win_first;
            kv.count = base + T - win_first;
            kv.win16 = m.win16.get();
            at::enqueue_fast_kv(kv, st);
        }
        // ---- per chunk of output rows: q_b, index + selection, attention, o_a, o_b
        chunks(o0, C, [&](std::uint32_t r0, std::uint32_t n) {
            op("q_b");
            Q8A8View qb;
            qb.tiled = lw.q_b.view();
            qb.count = QH;
            qb.split = 1;
            qb.x = a8_at(Q->qa, r0, n);
            qb.out = m.q.get();
            q8(qb);
            if (r.owns_index) {
                op("index_proj");
                at::F16RowsView v{};
                v.rows = n;
                v.jobs[v.job_count++] = at::F16Job{aw.index_q_b, QA, at::kIdxHeads * at::kIdxDim, Q->qr.get() + std::size_t{r0} * QA, QA,
                                                   m.iq.get(), at::kIdxHeads * at::kIdxDim};
                v.jobs[v.job_count++] = at::F16Job{aw.index_proj, H, at::kIdxHeads, Q->x.get() + std::size_t{r0} * H, H, m.iw.get(), at::kIdxHeads};
                f16_rows(v);
            }
            if (fa) {
                // chunk-wide index scores and selection, then one attention call writing the A8 rows of o
                op("index");
                if (r.owns_index && m.fast_index) {
                    const std::uint32_t IC = o.pass_rows ? std::min<std::uint32_t>(256U, C) : C;
                    for (std::uint32_t ir = 0; ir < n; ir += IC) {
                    const std::uint32_t nr = std::min(IC, n - ir), row = r0 + ir;
                    const auto& src = m.plan[static_cast<std::size_t>(r.kv_source)];
                    const std::uint32_t max_count = (base + row + nr) / src.ratio;
                    if (max_count == 0U) {
                        STRIX_CUDA_CHECK(cudaMemsetAsync(count_at(caches, row), 0, std::size_t{nr} * 4U, st));
                    } else {
                        at::FastIndexView iv{};
                        iv.rows = nr;
                        iv.ratio = src.ratio;
                        iv.max_count = max_count;
                        iv.q = m.iq.get() + std::size_t{ir} * at::kIdxHeads * at::kIdxDim;
                        iv.w = m.iw.get() + std::size_t{ir} * at::kIdxHeads;
                        iv.rope = rope;
                        iv.pos = pos + row;
                        iv.k8 = m.k8.get();
                        iv.kscale = m.kscale.get();
                        iv.scores = m.fscores.get();
                        iv.qprep = m.fqprep.get();
                        at::enqueue_fast_index_scores(iv, st);
                        at::SelectView sv{};
                        sv.long_context = o.pass_rows != 0U;
                        sv.rows = nr;
                        sv.ratio = src.ratio;
                        sv.max_count = max_count;
                        sv.topk = at::kTopK;
                        sv.cand_blocks = at::kCandBlocks;
                        sv.block_size = at::kCandBlockSize;
                        sv.pool = r.candidate_source ? ds41::attn::Pool::candidate_source
                                  : r.candidate_user ? ds41::attn::Pool::candidate_user : ds41::attn::Pool::plain;
                        sv.pos = pos + row;
                        sv.scores = m.fscores.get();
                        sv.candidates = (r.candidate_source || r.candidate_user) ? cand_at(row) : nullptr;
                        sv.cand_stride = Q->attn->cand_stride();
                        sv.block_keys = m.fblock_keys.get();
                        sv.survivors = m.fsurvivors.get();
                        sv.idx = idx_at(caches, row);
                        sv.idx_count = count_at(caches, row);
                        at::enqueue_select(sv, st);
                    }
                    }  // index query tiles
                } else if (r.owns_index) {  // no FP8 tensor cores: the exact indexer per window
                    const auto& src = m.plan[static_cast<std::size_t>(r.kv_source)];
                    const auto& sc = Q->attn->caches(static_cast<std::uint32_t>(r.kv_source));
                    for (std::uint32_t w0 = r0; w0 < r0 + n; w0 += W) {
                        const std::uint32_t wn = std::min(W, r0 + n - w0), max_count = (base + w0 + wn) / src.ratio;
                        if (max_count == 0U) {
                            STRIX_CUDA_CHECK(cudaMemsetAsync(count_at(caches, w0), 0, std::size_t{wn} * 4U, st));
                            continue;
                        }
                        at::IndexView iv{wn, src.ratio, max_count, m.iq.get() + std::size_t{w0 - r0} * at::kIdxHeads * at::kIdxDim,
                                         m.iw.get() + std::size_t{w0 - r0} * at::kIdxHeads, rope, pos + w0, sc.idx_cache, m.scores.get(), m.qprep.get()};
                        at::enqueue_index_scores(iv, st);
                        at::SelectView sv{};
                        sv.long_context = o.pass_rows != 0U;
                        sv.rows = wn;
                        sv.ratio = src.ratio;
                        sv.max_count = max_count;
                        sv.topk = at::kTopK;
                        sv.cand_blocks = at::kCandBlocks;
                        sv.block_size = at::kCandBlockSize;
                        sv.pool = r.candidate_source ? ds41::attn::Pool::candidate_source
                                  : r.candidate_user ? ds41::attn::Pool::candidate_user : ds41::attn::Pool::plain;
                        sv.pos = pos + w0;
                        sv.scores = m.scores.get();
                        sv.candidates = (r.candidate_source || r.candidate_user) ? cand_at(w0) : nullptr;
                        sv.cand_stride = Q->attn->cand_stride();
                        sv.block_keys = m.block_keys.get();
                        sv.survivors = m.survivors.get();
                        sv.idx = idx_at(caches, w0);
                        sv.idx_count = count_at(caches, w0);
                        at::enqueue_select(sv, st);
                    }
                }
                op("attention");
                at::FastAttentionView fv{};
                fv.rows = n;
                fv.topk = at::kTopK;
                fv.q = m.q.get();
                fv.sink = aw.sinks;
                fv.rope = rope;
                fv.pos = pos + r0;
                fv.win16 = m.win16.get();
                fv.win_first = win_first;
                if (r.ratio != 0U) {
                    const auto& ic = Q->attn->caches(static_cast<std::uint32_t>(r.index_source));
                    fv.kv16 = m.kv16.get();
                    fv.idx = idx_at(ic, r0);
                    fv.idx_count = count_at(ic, r0);
                }
                fv.a8 = a8_at(m.oa, 0, n);
                fv.heads = at::kHeads;
                at::enqueue_fast_attention(fv, st);
            }
            if (!fa) op("index_attention");
            for (std::uint32_t w0 = r0; w0 < (fa ? r0 : r0 + n); w0 += W) {
                const std::uint32_t wn = std::min(W, r0 + n - w0);
                if (r.owns_index) {
                    const auto& src = m.plan[static_cast<std::size_t>(r.kv_source)];
                    const auto& sc = Q->attn->caches(static_cast<std::uint32_t>(r.kv_source));
                    const std::uint32_t max_count = (base + w0 + wn) / src.ratio;  // the last row's (pos + 1) / ratio
                    if (max_count == 0U) {
                        STRIX_CUDA_CHECK(cudaMemsetAsync(count_at(caches, w0), 0, std::size_t{wn} * 4U, st));
                    } else {
                        at::IndexView iv{};
                        iv.rows = wn;
                        iv.ratio = src.ratio;
                        iv.max_count = max_count;
                        iv.q = m.iq.get() + std::size_t{w0 - r0} * at::kIdxHeads * at::kIdxDim;
                        iv.w = m.iw.get() + std::size_t{w0 - r0} * at::kIdxHeads;
                        iv.rope = rope;
                        iv.pos = pos + w0;
                        iv.idx_cache = sc.idx_cache;
                        iv.scores = m.scores.get();
                        iv.qprep = m.qprep.get();
                        at::enqueue_index_scores(iv, st);
                        at::SelectView sv{};
                        sv.long_context = o.pass_rows != 0U;
                        sv.rows = wn;
                        sv.ratio = src.ratio;
                        sv.max_count = max_count;
                        sv.topk = at::kTopK;
                        sv.cand_blocks = at::kCandBlocks;
                        sv.block_size = at::kCandBlockSize;
                        sv.pool = r.candidate_source ? ds41::attn::Pool::candidate_source
                                  : r.candidate_user ? ds41::attn::Pool::candidate_user : ds41::attn::Pool::plain;
                        sv.pos = pos + w0;
                        sv.scores = m.scores.get();
                        sv.candidates = (r.candidate_source || r.candidate_user) ? cand_at(w0) : nullptr;
                        sv.cand_stride = Q->attn->cand_stride();
                        sv.block_keys = m.block_keys.get();
                        sv.survivors = m.survivors.get();
                        sv.idx = idx_at(caches, w0);
                        sv.idx_count = count_at(caches, w0);
                        at::enqueue_select(sv, st);
                    }
                }
                at::AttentionView av{};
                av.rows = wn;
                av.capacity = Q->attn->ring_capacity();
                av.topk = at::kTopK;
                av.q = m.q.get() + std::size_t{w0 - r0} * QH;
                av.sink = aw.sinks;
                av.rope = rope;
                av.pos = pos + w0;
                av.ring = caches.ring;
                if (r.ratio != 0U) {
                    const auto& ic = Q->attn->caches(static_cast<std::uint32_t>(r.index_source));
                    av.kv_cache = Q->attn->caches(static_cast<std::uint32_t>(r.kv_source)).kv_cache;
                    av.idx = idx_at(ic, w0);
                    av.idx_count = count_at(ic, w0);
                }
                av.partial = m.partial.get();
                av.o = m.o_.get() + std::size_t{w0 - r0} * QH;
                av.heads = at::kHeads;
                at::enqueue_attention(av, st);
            }
            op("o_a_o_b");
            auto oa = a8_at(m.oa, 0, n);
            if (!fa) (fd ? fdn::enqueue_quantize_a8 : enqueue_quantize_a8)({m.o_.get(), n, QH, 0, oa}, st);
            Q8A8View a;
            a.tiled = lw.o_a.view();
            a.count = OL;
            a.per_group = g.o_lora;
            a.width = QH / g.o_groups;
            a.split = 1;
            a.x = oa;
            a.out = m.o_low.get();
            q8(a);
            auto la = a8_at(m.la, 0, n);
            (fd ? fdn::enqueue_quantize_a8 : enqueue_quantize_a8)({m.o_low.get(), n, OL, 0, la}, st);
            const std::uint32_t half = OL / 2U;
            if (fd) {  // both K halves in one call
                Q8A8View v;
                v.tiled = lw.o_b.view();
                v.count = H;
                v.split = 1;
                v.x = la;
                v.out = Q->y.get() + std::size_t{r0} * H;
                q8(v);
            }
            for (std::uint32_t h = 0; h < (fd ? 0U : 2U); ++h) {
                Q8A8View v;
                v.tiled = lw.o_b.view();
                v.count = H;
                v.k_offset = h * half;
                v.width = half;
                v.split = 1;
                v.x = {la.codes + h * half, la.deltas + h * half / 32U, la.sums + h * half / 32U, n, half, OL};
                v.out = Q->y.get() + std::size_t{r0} * H;
                v.accumulate = h > 0;
                q8(v);
            }
        });
        // ---- hc_post attention, ffn input
        op("hc_post");
        if (fd) {
            // hc_post/attn + the ffn input (collapse with the attn mix's pre, ffn norm, A8) + hc_mix/ffn
            hc_pass(o0, Q->mix_attn.get(), Q->mix_attn.get(), lw.ffn_norm, lw.hc_ffn_fn, lw.hc_ffn_base, lw.hc_ffn_scale, Q->mix_ffn.get());
        } else {
        chunks(o0, C, [&](std::uint32_t r0, std::uint32_t n) {
            enqueue_hc_expand({Q->y.get() + std::size_t{r0} * H, 0, stream + std::size_t{r0} * SR, Q->mix_attn.get() + std::size_t{r0} * 24U, n, H,
                               stream + std::size_t{r0} * SR},
                              st);
        });
        }
        mk.attn_end = mark();
        chunks(o0, C, [&](std::uint32_t r0, std::uint32_t n) {
            HcMixView v;
            v.stream = stream + std::size_t{r0} * SR;
            v.rows = n;
            v.hidden = H;
            v.fn = lw.hc_ffn_fn;
            v.base = lw.hc_ffn_base;
            v.scale = lw.hc_ffn_scale;
            v.iters = g.hc_iters;
            v.hc_eps = g.hc_eps;
            v.mixes = Q->mix_ffn.get() + std::size_t{r0} * 24U;
            v.partial = reinterpret_cast<float*>(m.ws.get());
            if (!fd) {
            op("hc_mix_ffn");
            enqueue_hc_mixes(v, st);
            op("norm_ffn");
            NormView nv;
            nv.hc_stream = stream + std::size_t{r0} * SR;
            nv.pre = Q->mix_attn.get() + std::size_t{r0} * 24U;
            nv.pre_stride = 24;
            nv.weight = lw.ffn_norm;
            nv.rows = n;
            nv.n = H;
            nv.out = Q->x.get() + std::size_t{r0} * H;
            nv.a8 = a8_at(Q->xa, r0, n);
            enqueue_norm(nv, st);
            }
            // router + shared expert
            op("router");
            RouterView rv;
            rv.x = Q->x.get() + std::size_t{r0} * H;
            rv.rows = n;
            rv.weight = {lw.router, g.experts, H};
            rv.bias = lw.bias;
            rv.bias_image = lw.bias_image;
            rv.top_k = K;
            rv.route_scale = g.route_scale;
            rv.scores = m.router_scores.get();
            rv.ids = Q->ids.get() + std::size_t{r0} * K;
            rv.weights = Q->route_w.get() + std::size_t{r0} * K;
            if (pk_rows(n)) pk::enqueue_router(rv, st);
            else enqueue_router(rv, st);
            op("shared");
            SharedExpertView e;
            e.gate_up_tiled = lw.gate_up.view();
            e.down_tiled = lw.down.view();
            e.limit = g.swiglu_limit;
            e.x = a8_at(Q->xa, r0, n);
            e.out = Q->shared.get() + std::size_t{r0} * H;
            e.workspace = m.ws.get();
            if (fd) fdn::enqueue_shared_expert(e, st);
            else if (o.prefill_q8 && n >= o.prefill_q8_min_rows) pk::enqueue_shared_expert(e, st);
            else enqueue_shared_expert(e, st);
        });
        });  // segment A
        }  // prompts (segment A)
        // Outside graph capture: the router has selected all rows, including the actual replay cone. This trace
        // measures the union available to a possible DMA filter; timings from traced runs are diagnostic only.
        if (!route_files.empty()) {
            for (std::size_t si = 0; si < B; ++si) {
                const auto& sr = runs[si];
                const auto first = sr.rows[layer].o0;
                std::vector<std::uint32_t> ids(std::size_t{sr.T - first} * K);
                STRIX_CUDA_CHECK(cudaMemcpyAsync(ids.data(), sr.q->ids.get() + std::size_t{first} * K,
                                                ids.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, st));
                STRIX_CUDA_CHECK(cudaStreamSynchronize(st));
                for (std::size_t r = 0; r < ids.size() / K; ++r) {
                    route_files[si] << layer << ':';
                    for (std::uint32_t k = 0; k < K; ++k) route_files[si] << (k ? "," : "") << ids[r * K + k];
                    route_files[si] << '\n';
                }
                require(static_cast<bool>(route_files[si]), "cannot write MoE route trace");
            }
        }
        // ---- routed experts + combine
        mk.moe_begin = mark();
        op("expert_wait");  // the device time the stream waits for this layer's slot copy
        if (any_output) {
        if (m.streamer) {
            const auto t_wait = std::chrono::steady_clock::now();
            const auto lx = m.streamer->acquire(layer);
            result.layers[layer].expert_wait_ms = static_cast<float>(seconds_since(t_wait) * 1e3);
            mk.wait_end = mark();  // the copy waits are inside the MoE ranges (group by group)
            // the fast MoE: a single identity range (a resident or whole streamed layer, one group) runs as on the
            // provider; several ranges run range by range (enqueue_moe_fast_layer_ranges)
            const bool one_range = lx.perm == nullptr && lx.ranges.size() == 1U && lx.ranges[0].first == 0U &&
                                   lx.ranges[0].end == c.n_routed_experts;
            for (std::size_t si = 0; si < B; ++si) {
            bind(si);
            const std::uint32_t o0 = (*rows_of)[layer].o0;
            op("moe");
            if (o.fast_moe && one_range && lx.ranges[0].ready) {
                const auto tr = timed_ranges(lx.ranges, layer);
                if (tr[0].wait_begin) STRIX_CUDA_CHECK(cudaEventRecord(tr[0].wait_begin, st));
                STRIX_CUDA_CHECK(cudaStreamWaitEvent(st, lx.ranges[0].ready, 0));
                if (tr[0].wait_end) STRIX_CUDA_CHECK(cudaEventRecord(tr[0].wait_end, st));
            }
            chunks(o0, MC, [&](std::uint32_t r0, std::uint32_t n) {
                if (o.fast_moe && !one_range && n >= 64U) {
                    // per-expert ranges on the fast MoE (same arithmetic as the whole-layer fast arm)
                    MoeFastLayerView f;
                    f.shape = m.moe_shape;
                    f.signed_lut = o.fast_moe_signed_lut;
                    f.wide_tiles = o.fast_moe_wide_tiles;
                    f.rows = n;
                    f.split = m.split;
                    f.x = Q->x.get() + std::size_t{r0} * H;
                    f.xa = a8_at(Q->xa, r0, n);
                    f.ids = Q->ids.get() + std::size_t{r0} * K;
                    f.route_weights = Q->route_w.get() + std::size_t{r0} * K;
                    f.shared = Q->shared.get() + std::size_t{r0} * H;
                    f.raw = m.raw.get();
                    f.y = Q->y.get() + std::size_t{r0} * H;
                    f.workspace = m.moe_ws.get();
                    f.capacity = MC;
                    const auto tr = timed_ranges(lx.ranges, layer);
                    enqueue_moe_fast_layer_ranges(f, lx.perm, m.mapped_ids.get(), tr.data(), static_cast<std::uint32_t>(tr.size()), st);
                    return;
                }
                if (o.fast_moe && one_range && n >= 64U) {
                    MoeFastLayerView f;
                    f.shape = m.moe_shape;
                    f.signed_lut = o.fast_moe_signed_lut;
                    f.wide_tiles = o.fast_moe_wide_tiles;
                    f.weights = lx.ranges[0].weights;
                    f.rows = n;
                    f.split = m.split;
                    f.x = Q->x.get() + std::size_t{r0} * H;
                    f.xa = a8_at(Q->xa, r0, n);
                    f.ids = Q->ids.get() + std::size_t{r0} * K;
                    f.route_weights = Q->route_w.get() + std::size_t{r0} * K;
                    f.shared = Q->shared.get() + std::size_t{r0} * H;
                    f.raw = m.raw.get();
                    f.y = Q->y.get() + std::size_t{r0} * H;
                    f.workspace = m.moe_ws.get();
                    f.capacity = MC;
                    enqueue_moe_fast_layer(f, st);
                    return;
                }
                MoeRangesView rv;
                MoeLayerView& v = rv.layer;
                v.shape = m.moe_shape;
                v.rows = n;
                v.split = m.split;
                v.x = a8_at(Q->xa, r0, n);
                v.ids = Q->ids.get() + std::size_t{r0} * K;
                v.route_weights = Q->route_w.get() + std::size_t{r0} * K;
                v.shared = Q->shared.get() + std::size_t{r0} * H;
                v.raw = m.raw.get();
                v.y = Q->y.get() + std::size_t{r0} * H;
                v.workspace = m.moe_ws.get();
                v.capacity = MC;
                rv.perm = lx.perm;
                rv.mapped_ids = m.mapped_ids.get();
                const auto tr = timed_ranges(lx.ranges, layer);
                rv.ranges = tr.data();
                rv.range_count = static_cast<std::uint32_t>(tr.size());
                enqueue_moe_layer_ranges(rv, st);
            });
            op("hc_post");
            if (fd && layer + 1U < o.end_layer && engram_li[layer + 1U] == ~0U) {
                const auto& nw = m.core->weights(layer + 1U);
                hc_pass(o0, Q->mix_ffn.get(), Q->mix_ffn.get(), nw.attn_norm, nw.hc_attn_fn, nw.hc_attn_base, nw.hc_attn_scale, Q->mix_attn.get());
                attn_input_from = o0;
            } else {
                chunks(o0, C, [&](std::uint32_t r0, std::uint32_t n) {
                    enqueue_hc_expand({Q->y.get() + std::size_t{r0} * H, 0, stream + std::size_t{r0} * SR, Q->mix_ffn.get() + std::size_t{r0} * 24U,
                                       n, H, stream + std::size_t{r0} * SR},
                                      st);
                });
            }
            }  // prompts (segment B)
            m.streamer->release(layer, st);  // after hc_post, as the provider
        } else {
            const auto t_wait = std::chrono::steady_clock::now();
            const ExpertWeights w = m.experts->acquire(layer, st);
            result.layers[layer].expert_wait_ms = static_cast<float>(seconds_since(t_wait) * 1e3);
            mk.wait_end = mark();
            for (std::size_t si = 0; si < B; ++si) {
            bind(si);
            const std::uint32_t o0 = (*rows_of)[layer].o0;
            cudaGraphExec_t* gb = &no_graph;
            if (use_graphs) {
                gb = &G.b[layer];
                const auto& kw = G.b_weights[layer];
                if (*gb != nullptr && (kw.gate != w.gate || kw.up != w.up || kw.down != w.down)) {
                    cudaGraphExecDestroy(*gb);  // the slot of this layer moved: recapture
                    *gb = nullptr;
                }
                G.b_weights[layer] = w;
            }
            segment(*gb, [&] {
            op("moe");
            chunks(o0, MC, [&](std::uint32_t r0, std::uint32_t n) {
                if (o.fast_moe && n >= 64U) {
                    MoeFastLayerView f;
                    f.shape = m.moe_shape;
                    f.signed_lut = o.fast_moe_signed_lut;
                    f.wide_tiles = o.fast_moe_wide_tiles;
                    f.weights = w;
                    f.rows = n;
                    f.split = m.split;
                    f.x = Q->x.get() + std::size_t{r0} * H;
                    f.xa = a8_at(Q->xa, r0, n);
                    f.ids = Q->ids.get() + std::size_t{r0} * K;
                    f.route_weights = Q->route_w.get() + std::size_t{r0} * K;
                    f.shared = Q->shared.get() + std::size_t{r0} * H;
                    f.raw = m.raw.get();
                    f.y = Q->y.get() + std::size_t{r0} * H;
                    f.workspace = m.moe_ws.get();
                    f.capacity = MC;
                    enqueue_moe_fast_layer(f, st);
                    return;
                }
                MoeLayerView v;
                v.shape = m.moe_shape;
                v.weights = w;
                v.rows = n;
                v.split = m.split;
                v.x = a8_at(Q->xa, r0, n);
                v.ids = Q->ids.get() + std::size_t{r0} * K;
                v.route_weights = Q->route_w.get() + std::size_t{r0} * K;
                v.shared = Q->shared.get() + std::size_t{r0} * H;
                v.raw = m.raw.get();
                // expert TP: a run of at most decode_rows rows (a sequence decoded from position 0) takes the TP order
                // too; longer prompts keep the EP order of the pair's prefill
                v.raw1 = o.expert_tp && n <= o.decode_rows && T <= o.decode_rows ? m.raw1.get() : nullptr;
                v.y = Q->y.get() + std::size_t{r0} * H;
                v.workspace = m.moe_ws.get();
                v.capacity = MC;
                enqueue_moe_layer(v, st);
            });
            op("hc_post");
            if (fd && layer + 1U < o.end_layer && engram_li[layer + 1U] == ~0U) {
                // hc_post/ffn + the next layer's attention input (collapse with this mix's pre, attn norm, A8) + its
                // hc_mix/attn (host state attn_input_from is set when the body runs: eager runs and captures)
                const auto& nw = m.core->weights(layer + 1U);
                hc_pass(o0, Q->mix_ffn.get(), Q->mix_ffn.get(), nw.attn_norm, nw.hc_attn_fn, nw.hc_attn_base, nw.hc_attn_scale, Q->mix_attn.get());
                attn_input_from = o0;
            } else {
                chunks(o0, C, [&](std::uint32_t r0, std::uint32_t n) {
                    enqueue_hc_expand({Q->y.get() + std::size_t{r0} * H, 0, stream + std::size_t{r0} * SR, Q->mix_ffn.get() + std::size_t{r0} * 24U,
                                       n, H, stream + std::size_t{r0} * SR},
                                      st);
                });
            }
            });  // segment B
            }  // prompts (segment B)
            m.experts->release(layer, st);  // after hc_post (the slot frees one small kernel later than before)
        }
        } else {
            mk.wait_end = mark();
        }  // no routed weights for a compression-only layer
        op("end");
        mk.end = mark();
        for (std::size_t si = 0; si < B; ++si) {
            bind(si);
            results[si].layers[layer].rows = T - (*rows_of)[layer].o0;
            if (o.stream_hashes) {
                std::vector<float> row(SR);
                STRIX_CUDA_CHECK(cudaMemcpyAsync(row.data(), stream + std::size_t{T - 1U} * SR, SR * 4U, cudaMemcpyDeviceToHost, st));
                STRIX_CUDA_CHECK(cudaStreamSynchronize(st));
                results[si].stream_hashes.push_back(fnv_bytes(row.data(), SR * 4U));
            }
        }
    }

    result.enqueue_seconds = seconds_since(t_run);
    // ---- head on the last row of every prompt (one after the other: the head buffers are shared)
    std::vector<double> done_s(B, 0.0);
    for (std::size_t si = 0; si < B; ++si) {
        bind(si);
        auto& res = results[si];
        if (m.has_head() && final_pass) {
            // the last hr rows (every one is an output row of the last layer: hr <= T - o0 of layer 39)
            auto& b = m.core->buffers();
            const std::uint32_t hr = std::min({std::max(1U, o.head_rows), T, T - (*rows_of)[c.num_hidden_layers - 1U].o0});
            const std::uint32_t h0 = T - hr;
            STRIX_CUDA_CHECK(cudaMemcpyAsync(b.stream.get(), stream + std::size_t{h0} * SR, std::size_t{hr} * SR * 4U, cudaMemcpyDeviceToDevice, st));
            copy_pre_kernel<<<(hr * 4U + 255U) / 256U, 256, 0, st>>>(Q->mix_ffn.get() + std::size_t{h0} * 24U, b.pre_carry.get(), hr);
            STRIX_CUDA_CHECK(cudaGetLastError());
            m.core->set_rows(hr);
            m.core->head(st);
            res.logits.resize(std::size_t{c.vocab_size});
            STRIX_CUDA_CHECK(cudaMemcpyAsync(res.logits.data(), b.logits.get() + std::size_t{hr - 1U} * c.vocab_size,
                                             std::size_t{c.vocab_size} * 4U, cudaMemcpyDeviceToHost, st));
            if (hr > 1U) {
                res.head_logits.resize(std::size_t{hr} * c.vocab_size);
                STRIX_CUDA_CHECK(cudaMemcpyAsync(res.head_logits.data(), b.logits.get(), res.head_logits.size() * 4U, cudaMemcpyDeviceToHost, st));
            }
            std::uint32_t cand[3] = {0, 0, 0};
            STRIX_CUDA_CHECK(cudaMemcpyAsync(cand, b.candidates.get() + (hr - 1U), 12U, cudaMemcpyDeviceToHost, st));
            STRIX_CUDA_CHECK(cudaStreamSynchronize(st));
            if (cand[2] == 0U) throw std::runtime_error("ds41 cuda prefill: non-finite logits");
            res.next_token = cand[1];
        } else {
            STRIX_CUDA_CHECK(cudaStreamSynchronize(st));
        }
        done_s[si] = seconds_since(t_run);  // this prompt's logits are on the host
    }
    if (run_last) STRIX_CUDA_CHECK(cudaEventRecord(run_last->e, st));  // after the last head (nothing enqueued after it)
    STRIX_CUDA_CHECK(cudaStreamSynchronize(st));
    const auto t_synced = std::chrono::steady_clock::now();
    join_engram();
    result.engram_join_s = seconds_since(t_synced);
    if (!long_mode) {  // run_long ends the expert schedule after its last pass
        if (m.o.defer_idle_preload) m.idle_held = true;  // capture_state / release_idle_preload / the next run end it
        else if (m.streamer) m.streamer->end_run();
        else m.experts->end_run();
    }
    result.seconds = seconds_since(t_run);
    result.engram_seconds = long_mode ? 0.0 : m.engram_seconds;
    {
        const auto es = m.engram_reader->stats();
        result.engram_records = es.records - engram_stats0.records;
        result.engram_pages = es.pages - engram_stats0.pages;
        result.engram_page_hits = es.page_hits - engram_stats0.page_hits;
    }
    if (m.streamer && (!long_mode || final_pass)) {
        const auto ss = m.streamer->stats();
        result.expert_io_seconds = ss.io_seconds;
        result.expert_bytes_streamed = ss.bytes_streamed;
        result.expert_copy_busy_s = ss.copy_busy_ms * 1e-3;
        result.preload_bytes = ss.preload_bytes;
        result.preload_s = ss.preload_s;
        result.preload_after_begin_s = ss.preload_after_begin_s;
        result.host_promoted_layers = ss.promoted_layers;
        result.host_demoted_layers = ss.demoted_layers;
        result.host_promoted_total = ss.promoted_held;
        result.host_promoted_bytes = ss.promoted_bytes;
        result.file_bytes_streamed = ss.file_bytes;
        for (std::uint32_t l = o.first_layer; l < o.end_layer; ++l) {
            result.layers[l].copy_bytes = ss.copy_bytes[l];
            if (ss.copy_begin_ms[l] >= 0.0F) result.layers[l].copy_ms = ss.copy_end_ms[l] - ss.copy_begin_ms[l];
        }
        if (o.timing && m.streamer->copy_origin())
            for (std::uint32_t l = o.first_layer; l < o.end_layer; ++l) {
                if (ss.copy_end_ms[l] < 0.0F || !marks[l].moe_begin) continue;
                float at = 0.0F;
                STRIX_CUDA_CHECK(cudaEventElapsedTime(&at, m.streamer->copy_origin(), marks[l].moe_begin));
                const float stall = std::max(0.0F, ss.copy_end_ms[l] - at);
                result.layers[l].expert_wait_ms = stall;
                result.expert_stall_ms += stall;
            }
    } else if (!m.streamer && (!long_mode || final_pass)) {
        result.expert_io_seconds = m.experts->io_seconds();
        result.expert_bytes_streamed = m.experts->bytes_streamed();
        result.file_bytes_streamed = m.experts->file_bytes_streamed();
    }
    if (o.timing)
        for (std::uint32_t l = o.first_layer; l < o.end_layer; ++l) {
            const auto& mk = marks[l];
            if (!mk.begin) continue;
            auto& t = result.layers[l];
            STRIX_CUDA_CHECK(cudaEventElapsedTime(&t.total_ms, mk.begin, mk.end));
            if (mk.attn_end) STRIX_CUDA_CHECK(cudaEventElapsedTime(&t.attention_ms, mk.begin, mk.attn_end));
            STRIX_CUDA_CHECK(cudaEventElapsedTime(&t.moe_ms, mk.moe_begin, mk.end));
            STRIX_CUDA_CHECK(cudaEventElapsedTime(&t.expert_wait_device_ms, mk.moe_begin, mk.wait_end));
            for (const auto& [wb, we] : range_waits[l]) {
                float ms = 0.0F;
                if (cudaEventElapsedTime(&ms, wb, we) == cudaSuccess) t.range_wait_ms += ms;
                else (void)cudaGetLastError();  // a range that was never enqueued
            }
            const cudaEvent_t prev = l == o.first_layer ? run_first->e : marks[l - 1U].end;
            STRIX_CUDA_CHECK(cudaEventElapsedTime(&t.gap_before_ms, prev, mk.begin));
        }
    if (o.timing) {
        float ms = 0.0F;
        STRIX_CUDA_CHECK(cudaEventElapsedTime(&ms, run_first->e, run_last->e));
        result.device_ms = ms;
        result.embed_ms = result.layers[o.first_layer].gap_before_ms;
        result.layers[o.first_layer].gap_before_ms = 0.0F;
        STRIX_CUDA_CHECK(cudaEventElapsedTime(&ms, marks[o.end_layer - 1U].end, run_last->e));
        result.head_ms = ms;
    }
    for (std::size_t i = 0; i + 1U < op_marks.size(); ++i) {
        if (std::strcmp(op_marks[i].first, "end") == 0) continue;
        float ms = 0.0F;
        STRIX_CUDA_CHECK(cudaEventElapsedTime(&ms, op_marks[i].second, op_marks[i + 1U].second));
        result.ops[op_marks[i].first] += ms;
    }
    ++m.runs_done;
    m.last_batch_ctx = ctx;
    for (std::size_t si = 0; si < B; ++si) {
        auto& q = *runs[si].q;
        q.last_T = base + runs[si].T;
        q.last_rows = runs[si].T;
        if (long_mode) q.last_tokens.assign(m.request_tokens.begin(), m.request_tokens.begin() + q.last_T);
        else q.last_tokens.assign(runs[si].tokens.begin(), runs[si].tokens.end());
        q.last_next = results[si].next_token;
        if (ctx[si] == 0U) {  // context 0 is the decode sequence: a fresh prefill, no step pending
            m.last_row = runs[si].T - 1U;
            m.last_step_rows = 0;
        }
    }
    // batch-wide figures in every result; seconds: until this prompt's logits (the batch's wall in batch_seconds)
    for (std::size_t si = 1; si < B; ++si) {
        auto& r = results[si];
        r.enqueue_seconds = result.enqueue_seconds;
        r.engram_seconds = result.engram_seconds;
        r.expert_io_seconds = result.expert_io_seconds;
        r.expert_bytes_streamed = result.expert_bytes_streamed;
        r.file_bytes_streamed = result.file_bytes_streamed;
        r.host_promoted_layers = result.host_promoted_layers;
        r.host_demoted_layers = result.host_demoted_layers;
        r.host_promoted_total = result.host_promoted_total;
        r.host_promoted_bytes = result.host_promoted_bytes;
        r.expert_copy_busy_s = result.expert_copy_busy_s;
        r.expert_stall_ms = result.expert_stall_ms;
        r.engram_records = result.engram_records;
        r.engram_pages = result.engram_pages;
        r.engram_page_hits = result.engram_page_hits;
        r.device_ms = result.device_ms;
        r.embed_ms = result.embed_ms;
        r.head_ms = result.head_ms;
        r.host_setup_s = result.host_setup_s;
        r.engram_join_s = result.engram_join_s;
        r.preload_bytes = result.preload_bytes;
        r.preload_s = result.preload_s;
        r.preload_after_begin_s = result.preload_after_begin_s;
        r.ops = result.ops;
        r.graphs = result.graphs;
        for (std::uint32_t l = 0; l < c.num_hidden_layers; ++l) {
            const auto rows_l = r.layers[l].rows;
            r.layers[l] = result.layers[l];
            r.layers[l].rows = rows_l;
        }
    }
    for (std::size_t si = 0; si < B; ++si) {
        results[si].batch_seconds = result.seconds;
        results[si].batch_size = static_cast<std::uint32_t>(B);
        if (B > 1U) results[si].seconds = done_s[si];
        results[si].host_tail_s = seconds_since(t_synced);
    }
    return results;
}


std::vector<std::pair<std::string, std::uint64_t>> CudaPrefill::state_hashes() const {
    return export_state().hashes();
}

void CudaPrefill::release_idle_preload() {
    auto& m = *impl_;
    if (!m.idle_held) return;
    m.idle_held = false;
    if (m.streamer) m.streamer->end_run();
    else m.experts->end_run();
}

void CudaPrefill::capture_state(attn::ExportCapture& out) {
    auto& m = *impl_;
    // export_state()'s sequence: item 0 of the last run / run_batch (context 0 after a run(), a step or an import)
    const std::size_t k = m.last_batch_ctx.empty() ? 0U : m.last_batch_ctx[0];
    const auto& q = *m.seqs[k];
    if (q.last_T == 0U) throw std::logic_error("ds41 cuda prefill: capture_state before a run");
    try {
        q.attn->capture_state(m.o.first_layer, m.o.end_layer, q.last_T, k == 0U ? m.last_row : q.last_rows - 1U, q.last_tokens, q.last_next, out, m.st);  // as export_state()
    } catch (...) {
        release_idle_preload();
        throw;
    }
    release_idle_preload();
}

std::size_t CudaPrefill::capture_bound(std::uint32_t positions) const {
    const auto& m = *impl_;
    return m.seq0().attn->capture_bound(m.o.first_layer, m.o.end_layer, positions != 0U ? positions : m.o.max_positions);
}

deepseek41::AttentionState CudaPrefill::export_state() const { return export_state(0U); }

deepseek41::AttentionState CudaPrefill::export_state(std::size_t item) const {
    const auto& m = *impl_;
    if (item >= m.last_batch_ctx.size()) throw std::logic_error("ds41 cuda prefill: export_state before a run (or no such batch item)");
    const std::size_t k = m.last_batch_ctx[item];
    const auto& q = *m.seqs[k];
    if (q.last_T == 0U) throw std::logic_error("ds41 cuda prefill: export_state before a run");
    // the last row's selection: a prefill's is the one of its last row (the caches hold one selection per row of the
    // run or pass: q.last_rows); context 0 (the decode sequence) may have continued with steps: last_row
    auto state = q.attn->export_state(m.o.first_layer, m.o.end_layer, q.last_T, k == 0U ? m.last_row : q.last_rows - 1U, m.st);
    state.tokens = q.last_tokens;
    state.next_token = q.last_next;
    return state;
}

std::size_t CudaPrefill::batch_contexts() const noexcept { return impl_->seqs.size(); }
std::uint32_t CudaPrefill::context_positions(std::size_t k) const noexcept {
    return k < impl_->seqs.size() ? impl_->seqs[k]->capacity : 0U;
}

// ================================================================================================ incremental decode
// A step of R rows on the LayerCore buffers (R <= decode_rows) and the attention's decode sequence
// (AttentionState::enqueue_layer): the same kernels, in the same order, as the reference's decode step; the attention
// caches the prefill (or an import) left are continued in place (ring slots by position, compressed entries by
// position / ratio, the ratio-2 carry). The step is cut into segments at its Engram layers (one launch each): the host reads
// an Engram layer's rows for the step's tokens while the segments before it run.
namespace {

void wait_event(cudaEvent_t e, bool spin) {
    if (!spin) {
        STRIX_CUDA_CHECK(cudaEventSynchronize(e));
        return;
    }
    for (;;) {
        const cudaError_t q = cudaEventQuery(e);
        if (q == cudaSuccess) return;
        if (q != cudaErrorNotReady) STRIX_CUDA_CHECK(q);
    }
}

}  // namespace

std::uint32_t CudaPrefill::position() const noexcept { return impl_->seq0().last_T; }
std::uint32_t CudaPrefill::next_token() const noexcept { return impl_->seq0().last_next; }
std::uint32_t CudaPrefill::cand_stride() const noexcept { return impl_->seq0().attn->cand_stride(); }
void CudaPrefill::set_decode_graphs(bool on) noexcept { impl_->o.decode_graphs = on; }
void CudaPrefill::clear_decode_graphs() noexcept {
    (void)cudaDeviceSynchronize();
    if (impl_->dec) impl_->dec->graphs.clear();
}
void CudaPrefill::set_timing(bool on, bool ops) noexcept {
    impl_->o.timing = on;
    impl_->o.timing_ops = ops;
}

void CudaPrefill::set_sequences(std::uint32_t count) {
    auto& m = *impl_;
    require(count >= 1U, "at least one sequence slot");
    require(m.sequence == 0U, "set_sequences with slot 0 selected");
    if (m.dec) {
        m.dec->drain_engram();
        const auto slots = std::max<std::size_t>(1, m.sequences.size());
        if (count != slots) {
            STRIX_CUDA_CHECK(cudaDeviceSynchronize());
            m.dec->graphs.clear();
        }
    }
    if (m.sequences.empty()) m.sequences.resize(1);  // slot 0: the fields of the object
    while (m.sequences.size() > count) {
        m.state_bytes -= m.sequences.back().attn->device_bytes();
        m.sequences.pop_back();
    }
    while (m.sequences.size() < count) {
        Impl::Sequence q;
        q.attn = std::make_unique<at::AttentionState>(m.plan, m.attn_options);
        q.attn->reset(m.st);
        q.hasher = std::make_unique<EngramHasher>(m.model->engram);
        m.state_bytes += q.attn->device_bytes();
        m.sequences.push_back(std::move(q));
    }
    STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));
}
void CudaPrefill::select_sequence(std::uint32_t slot) {
    auto& m = *impl_;
    require(slot < std::max<std::size_t>(m.sequences.size(), 1U), "no such sequence slot");
    if (slot == m.sequence) return;
    if (m.dec) m.dec->drain_engram();                   // a prefetch belongs to the selected sequence's next token
    STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));      // nothing still reads the selected slot's buffers
    m.swap_sequence(m.sequences[m.sequence]);           // the selected slot back into its entry
    m.swap_sequence(m.sequences[slot]);                 // the new one into the fields
    m.sequence = slot;
}
std::uint32_t CudaPrefill::sequences() const noexcept { return static_cast<std::uint32_t>(std::max<std::size_t>(impl_->sequences.size(), 1U)); }
std::uint32_t CudaPrefill::sequence() const noexcept { return impl_->sequence; }

void CudaPrefill::reset_sequence() {
    auto& m = *impl_;
    if (m.dec) m.dec->drain_engram();
    STRIX_CUDA_CHECK(cudaDeviceSynchronize());
    m.seq0().attn->reset(m.st);
    STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));
    m.seq0().last_T = 0;
    m.seq0().last_tokens.clear();
    m.seq0().last_next = 0;
    m.last_row = 0;
    m.last_step_rows = 0;
    m.fresh = true;
    m.last_batch_ctx.assign(1, 0U);
    m.hasher->reset();
}

namespace {
// bounded passes (pass_rows): the state a later extend() continues (the compact decoder rows of the long prefill)
void import_state_long(CudaPrefill::Impl& m, const deepseek41::AttentionState& state) {
    auto& q = *m.seqs[0];
    require(m.o.pass_rows != 0U && !m.long_request, "import_state requires idle long prefill");
    require(state.position != 0U && state.position <= m.o.max_positions && state.tokens.size() == state.position,
            "import_state position and token history");
    require(state.layers.size() == m.plan.size(), "import_state layer count");
    for (const auto token : state.tokens) require(token < m.cfg.vocab_size, "import_state token outside vocabulary");
    for (auto l = m.o.first_layer; l < m.o.end_layer; ++l)
        require(!m.plan[l].owns_index || state.layers[l].topk_known, "import_state needs known producer selections");
    if (m.dec) m.dec->drain_engram();
    STRIX_CUDA_CHECK(cudaDeviceSynchronize());
    if (m.o.cone) q.attn->set_decoder_row0(0U);
    q.attn->import_state(state, m.st);
    q.last_T = state.position;
    q.last_rows = 1U;  // import stores the last known selection at physical row zero
    q.last_tokens = state.tokens;
    q.last_next = state.next_token;
    m.last_batch_ctx = {0U};
    m.last_row = 0;  // the selection row export_state() / capture_state() read for context 0
    m.last_step_rows = 0;
}
}  // namespace

void CudaPrefill::reserve_decode_buffers() {
    auto& m = *impl_;
    require(m.o.pass_rows == 0U, "decode buffers need pass_rows 0");
    if (!m.dec) m.dec = std::make_unique<Impl::Decode>();
    auto& d = *m.dec;
    const auto a = at::decode_allocations(m.decode_alloc_options);
    if (!d.router_counters.size()) {
        d.router_counters.resize(a.bytes("router_counters") / sizeof(std::uint32_t));
        d.router_counters.zero();
    }
    const auto engram_bytes = m.cfg.engram_layer_ids.size() * std::size_t{m.o.decode_rows} * m.g.engram_cols * 264U;
    if (!d.engram.size() && engram_bytes) d.engram.resize(engram_bytes);
    if (!d.sec_dev.size()) d.sec_dev.resize(a.bytes("sec_dev"));
    if (!d.sec_host.size()) d.sec_host.resize(a.bytes("sec_dev"), true);
}

void CudaPrefill::import_state(const deepseek41::AttentionState& state) {
    auto& m = *impl_;
    if (m.o.pass_rows != 0U) return import_state_long(m, state);  // a long-context object: continued by extend()
    require(state.position >= 1U && state.position < m.o.max_positions, "imported position 1..max_positions - 1");
    require(state.tokens.size() == state.position, "imported state: tokens of every position");
    if (m.dec) m.dec->drain_engram();
    STRIX_CUDA_CHECK(cudaDeviceSynchronize());
    m.seq0().attn->import_state(state, m.st);
    // the index sources' last selection is not imported: the next step selects again
    m.seq0().last_T = state.position;
    m.seq0().last_tokens = state.tokens;
    m.seq0().last_next = state.next_token;
    m.last_row = 0;
    m.last_step_rows = 0;
    m.last_batch_ctx.assign(1, 0U);
    m.hasher->reset();
    (void)m.hasher->hash_sequence(state.tokens);
}

void CudaPrefill::copy_kv_entries(std::uint32_t layer, std::uint32_t first, std::uint32_t count, std::uint8_t* kv, std::uint8_t* idx_keys) const {
    const auto& m = *impl_;
    const auto& c = m.seq0().attn->caches(layer);
    require(c.kv_cache != nullptr, "copy_kv_entries: not a KV source of this object");
    STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));
    if (count == 0U) return;
    STRIX_CUDA_CHECK(cudaMemcpy(kv, c.kv_cache + std::size_t{first} * at::kKvEntryBytes, std::size_t{count} * at::kKvEntryBytes, cudaMemcpyDeviceToHost));
    STRIX_CUDA_CHECK(
        cudaMemcpy(idx_keys, c.idx_cache + std::size_t{first} * at::kIdxEntryBytes, std::size_t{count} * at::kIdxEntryBytes, cudaMemcpyDeviceToHost));
}
void CudaPrefill::copy_topk(std::uint32_t layer, std::uint32_t rows, std::uint32_t* counts, std::uint32_t* ids) const {
    const auto& m = *impl_;
    const auto& c = m.seq0().attn->caches(layer);
    require(c.idx != nullptr, "copy_topk: not an index source of this object");
    STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));
    STRIX_CUDA_CHECK(cudaMemcpy(counts, c.idx_count, std::size_t{rows} * 4U, cudaMemcpyDeviceToHost));
    STRIX_CUDA_CHECK(cudaMemcpy(ids, c.idx, std::size_t{rows} * at::kTopK * 4U, cudaMemcpyDeviceToHost));
}
void CudaPrefill::copy_candidates(std::uint32_t rows, std::uint8_t* out) const {
    const auto& m = *impl_;
    require(m.seq0().attn->candidates() != nullptr, "copy_candidates: no candidate pool");
    STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));
    STRIX_CUDA_CHECK(cudaMemcpy(out, m.seq0().attn->candidates(), std::size_t{rows} * m.seq0().attn->cand_stride(), cudaMemcpyDeviceToHost));
}

void CudaPrefill::write_kv_entries(std::uint32_t layer, std::uint32_t first, std::uint32_t count, const std::uint8_t* kv, const std::uint8_t* idx_keys) {
    auto& m = *impl_;
    const auto& c = m.seq0().attn->caches(layer);
    require(c.kv_cache != nullptr, "write_kv_entries: not a KV source read by this object");
    STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));
    if (count == 0U) return;
    STRIX_CUDA_CHECK(cudaMemcpy(c.kv_cache + std::size_t{first} * at::kKvEntryBytes, kv, std::size_t{count} * at::kKvEntryBytes, cudaMemcpyHostToDevice));
    STRIX_CUDA_CHECK(
        cudaMemcpy(c.idx_cache + std::size_t{first} * at::kIdxEntryBytes, idx_keys, std::size_t{count} * at::kIdxEntryBytes, cudaMemcpyHostToDevice));
}
void CudaPrefill::write_topk(std::uint32_t layer, std::uint32_t rows, const std::uint32_t* counts, const std::uint32_t* ids) {
    auto& m = *impl_;
    const auto& c = m.seq0().attn->caches(layer);
    require(c.idx != nullptr, "write_topk: not an index source read by this object");
    STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));
    STRIX_CUDA_CHECK(cudaMemcpy(c.idx_count, counts, std::size_t{rows} * 4U, cudaMemcpyHostToDevice));
    STRIX_CUDA_CHECK(cudaMemcpy(c.idx, ids, std::size_t{rows} * at::kTopK * 4U, cudaMemcpyHostToDevice));
}
void CudaPrefill::write_candidates(std::uint32_t rows, const std::uint8_t* in) {
    auto& m = *impl_;
    require(m.seq0().attn->candidates() != nullptr, "write_candidates: no candidate pool");
    STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));
    STRIX_CUDA_CHECK(cudaMemcpy(m.seq0().attn->candidates(), in, std::size_t{rows} * m.seq0().attn->cand_stride(), cudaMemcpyHostToDevice));
}

CudaStepResult CudaPrefill::step(std::span<const std::uint32_t> tokens, const CudaStepIo& io) {
    auto& m = *impl_;
    const auto& o = m.o;
    const auto& c = m.cfg;
    const auto& g = m.g;
    const auto t_step = std::chrono::steady_clock::now();
    // step() reads device-tier layers through the whole-layer provider (and the rest sparsely from the file); the expert
    // streamer (expert_residency / xfer_*) serves run() only, so a streamer object refuses steps
    require(!m.streamer, "step() needs the whole-layer expert provider (no expert_residency / xfer_group_experts / xfer_piece_mib)");
    // bounded passes (pass_rows): compact decoder rows and no attention workspace; continued by extend(), not step()
    require(m.o.pass_rows == 0U, "step() needs pass_rows 0 (a long-context object continues with extend())");
    const auto R = static_cast<std::uint32_t>(tokens.size());
    const std::uint32_t P = m.seq0().last_T;
    require(P >= 1U || m.fresh, "step after a run(), an import_state() or a reset_sequence()");
    require(R >= 1U && R <= o.decode_rows, "step rows 1..decode_rows");
    require(P + R <= o.max_positions, "step positions below max_positions");
    for (const auto t : tokens) require(t < c.vocab_size, "token outside the vocabulary");
    release_idle_preload();  // defer_idle_preload: end the run state the last run() left held (as the next run would)
    if (!m.dec) m.dec = std::make_unique<Impl::Decode>();
    auto& D = *m.dec;
    if (D.router_counters.size() == 0U) {
        D.router_counters.resize(at::decode_allocations(m.decode_alloc_options).bytes("router_counters") / sizeof(std::uint32_t));
        D.router_counters.zero();
    }

    const std::uint32_t first = o.first_layer, stop = io.stop_layer ? io.stop_layer : o.end_layer;
    require(stop > first && stop <= o.end_layer, "stop layer in (first, end]");
    const bool embed = first == 0U, head = stop == c.num_hidden_layers && m.has_head();
    require(embed || (io.stream_in != nullptr && io.mix_in != nullptr), "a step of a range after layer 0 needs stream_in and mix_in");
    const std::uint32_t H = c.hidden_size, K = c.num_experts_per_tok, V = c.vocab_size;
    const std::size_t SR = std::size_t{4} * H;
    cudaStream_t st = m.st;
    auto& core = *m.core;
    auto& b = core.buffers();
    CudaStepResult result;
    result.position = P;
    result.rows = R;

    // Engram layers of the step and the first one (segment boundary)
    const std::uint32_t ecols = g.engram_cols, elayers = static_cast<std::uint32_t>(c.engram_layer_ids.size());
    std::vector<std::uint32_t> engram_li(c.num_hidden_layers, ~0U);
    for (std::uint32_t li = 0; li < elayers; ++li) engram_li[c.engram_layer_ids[li]] = li;
    const std::size_t erow_bytes = std::size_t{ecols} * 264U;
    if (D.engram.size() < std::size_t{elayers} * o.decode_rows * erow_bytes) D.engram.resize(std::size_t{elayers} * o.decode_rows * erow_bytes);

    bool resident = true;
    for (std::uint32_t l = first; l < stop; ++l) resident = resident && m.tiers[l] == ExpertTier::device;
    const bool hooked = static_cast<bool>(io.moe_site);
    // the caller's routed-expert site combines in the EP order (hot mode); the expert-TP order and the trellis
    // shared expert are not qualified with it
    require(!hooked || !o.expert_tp, "a moe_site hook does not combine with expert_tp");
    require(resident || hooked || o.decode_sparse_experts, "a step over non-resident expert layers needs decode_sparse_experts");

    // host parameters of the step (read by the step's own copies)
    auto* hpos = reinterpret_cast<std::uint64_t*>(D.params.get() + Impl::Decode::kParamsPos);
    auto* htok = reinterpret_cast<std::uint32_t*>(D.params.get() + Impl::Decode::kParamsTokens);
    auto* hcand = reinterpret_cast<std::uint32_t*>(D.params.get() + Impl::Decode::kParamsCand);
    for (std::uint32_t r = 0; r < R; ++r) {
        hpos[r] = P + r;
        htok[r] = tokens[r];
    }
    const std::size_t io_bytes = std::size_t{o.decode_rows} * (SR + 24U) * 4U;
    if (!embed) {
        if (D.io_in.size() < io_bytes) D.io_in.resize(io_bytes);
        std::memcpy(D.io_in.get(), io.stream_in, std::size_t{R} * SR * 4U);
        std::memcpy(D.io_in.get() + std::size_t{R} * SR * 4U, io.mix_in, std::size_t{R} * 24U * 4U);
    }
    const bool want_out = io.stream_out != nullptr || io.mix_out != nullptr || io.stream_view;
    if (want_out && D.io_out.size() < io_bytes) D.io_out.resize(io_bytes, true);
    const bool want_logits = head && io.logits;
    if (want_logits && D.logits.size() < std::size_t{o.decode_rows} * V * 4U) D.logits.resize(std::size_t{o.decode_rows} * V * 4U);

    // graphs
    const std::uint32_t bucket = (P + R - 1U) / o.decode_bucket;
    const std::uint64_t pos_bound = std::min<std::uint64_t>(std::uint64_t{bucket + 1U} * o.decode_bucket - 1U, o.max_positions - 1U);
    const bool timing = o.timing;
    const std::uint32_t draft_first = io.draft_hidden ? (io.draft_hidden_first != ~0U ? io.draft_hidden_first : c.num_hidden_layers - 3U) : 0U;
    require(!io.draft_hidden_raw || io.draft_hidden, "draft_hidden_raw needs draft_hidden");
    require(!io.draft_hidden || (draft_first >= first && draft_first + 3U <= stop), "draft_hidden layers inside the step's layers");
    const Impl::Decode::Key key{R, bucket, stop, !embed, want_out, timing, io.sections_cut, m.sequence, hooked, hooked ? io.moe_site_epoch : 0U,
                                io.draft_hidden, draft_first, io.draft_hidden_raw};
    // ---- boundary sections of the cut: which sources, where they land in the staging / pinned buffers
    struct SecKv {
        std::uint32_t layer, ratio, n;
        std::size_t at;
    };
    struct SecTopk {
        std::uint32_t layer;
        std::size_t at;
    };
    std::vector<SecKv> sec_kv;
    std::vector<SecTopk> sec_topk;
    std::int32_t sec_cand = -1;
    std::size_t sec_cand_at = 0, sec_bytes = 0;
    if (io.sections_cut != 0U) {
        const std::uint32_t K = io.sections_cut;
        require(K > first && K <= stop, "sections_cut inside the step's layers");
        std::vector<bool> kv_src(c.num_hidden_layers, false), ix_src(c.num_hidden_layers, false);
        bool cand_user_after = false;
        std::int32_t cand_source = -1;
        for (std::uint32_t l = 0; l < c.num_hidden_layers; ++l) {
            const auto& r = m.plan[l];
            if (r.candidate_source) cand_source = static_cast<std::int32_t>(l);
            if (l < K) continue;
            if (r.candidate_user) cand_user_after = true;
            if (r.ratio == 0U) continue;
            if (r.kv_source >= 0 && static_cast<std::uint32_t>(r.kv_source) < K) kv_src[static_cast<std::size_t>(r.kv_source)] = true;
            if (r.index_source >= 0 && static_cast<std::uint32_t>(r.index_source) < K) ix_src[static_cast<std::size_t>(r.index_source)] = true;
        }
        for (std::uint32_t l = 0; l < K; ++l) {
            if (kv_src[l]) {
                const std::uint32_t ratio = m.plan[l].ratio, n = R / ratio + 1U;
                sec_kv.push_back({l, ratio, n, sec_bytes});
                sec_bytes += std::size_t{n} * (288U + 68U);
            }
        }
        sec_bytes = (sec_bytes + 15U) / 16U * 16U;
        for (std::uint32_t l = 0; l < K; ++l)
            if (ix_src[l]) {
                sec_topk.push_back({l, sec_bytes});
                sec_bytes += std::size_t{R} * 4U + std::size_t{R} * at::kTopK * 4U;
            }
        if (cand_source >= 0 && static_cast<std::uint32_t>(cand_source) < K && cand_user_after) {
            sec_cand = cand_source;
            sec_cand_at = sec_bytes;
            sec_bytes += std::size_t{R} * m.seq0().attn->cand_stride();
        }
        require(sec_kv.size() <= 8U, "at most 8 KV sources per cut");
        const std::size_t cap = at::decode_allocations(m.decode_alloc_options).bytes("sec_dev");
        if (D.sec_dev.size() < cap) D.sec_dev.resize(cap);
        if (D.sec_host.size() < cap) D.sec_host.resize(cap, true);
        require(sec_bytes <= cap, "section staging size");
    }
    auto& cap_ptr = D.graphs[key];
    if (!cap_ptr) cap_ptr = std::make_unique<Impl::Decode::Captured>();
    auto& cap = *cap_ptr;
    const bool use_graphs = o.decode_graphs && (hooked ? io.moe_site_graphs : resident) && !o.stream_hashes && cap.eager_steps >= o.decode_graph_warmup;
    result.graphs = use_graphs;
    bool capturing = false;

    // timing events
    std::vector<std::unique_ptr<Event>> ev;
    const auto new_event = [](std::vector<std::unique_ptr<Event>>& pool) {
        pool.push_back(std::make_unique<Event>(true));
        return pool.back()->e;
    };
    // per-layer marks: eager steps record step-local events; captured segments record the Captured's (external
    // event nodes, re-recorded by every replay)
    std::vector<std::array<cudaEvent_t, 4>> marks(stop - first, std::array<cudaEvent_t, 4>{});
    const auto mark = [&](std::uint32_t layer, int which) {
        if (!timing) return;
        if (use_graphs) {
            if (!capturing) return;
            const cudaEvent_t e = new_event(cap.events);
            if (cap.marks.size() < stop - first) cap.marks.assign(stop - first, std::array<cudaEvent_t, 4>{});
            cap.marks[layer - first][static_cast<std::size_t>(which)] = e;
            STRIX_CUDA_CHECK(cudaEventRecordWithFlags(e, st, cudaEventRecordExternal));
            return;
        }
        const cudaEvent_t e = new_event(ev);
        marks[layer - first][static_cast<std::size_t>(which)] = e;
        STRIX_CUDA_CHECK(cudaEventRecord(e, st));
    };
    std::vector<std::pair<const char*, cudaEvent_t>> op_marks;
    const bool op_events = timing && o.timing_ops && !use_graphs;
    const auto op = [&](const char* name) {
        if (!op_events) return;
        op_marks.emplace_back(name, new_event(ev));
        STRIX_CUDA_CHECK(cudaEventRecord(op_marks.back().second, st));
    };

    core.set_rows(R);
    double expert_io = 0.0;
    std::vector<std::uint32_t> route_ids_host;

    // ---- one layer of the step (the reference's decode op order; LayerCore's calls, inlined for the op marks)
    const auto layer_body = [&](std::uint32_t layer) {
        const auto& w = core.weights(layer);
        const auto& aw = m.attn_w[layer];
        HcMixView hc_mix;
        mark(layer, 0);
        if (io.draft_hidden && layer >= draft_first && layer < draft_first + 3U) {
            op("draft_hidden");
            draft_hidden_kernel<<<dim3((H + 255U) / 256U, R), 256, 0, st>>>(b.stream.get(), H, io.draft_hidden + std::size_t{layer - draft_first} * H,
                                                                          3U * H);
            STRIX_CUDA_CHECK(cudaGetLastError());
            if (io.draft_hidden_raw)
                STRIX_CUDA_CHECK(cudaMemcpy2DAsync(io.draft_hidden_raw + std::size_t{layer - draft_first} * SR, 3U * SR * 4U, b.stream.get(), SR * 4U,
                                                   SR * 4U, R, cudaMemcpyDeviceToDevice, st));
        }
        if (engram_li[layer] != ~0U) {
            op("engram");
            STRIX_CUDA_CHECK(cudaMemcpyAsync(b.engram_rows.get(), D.engram.get() + std::size_t{engram_li[layer]} * o.decode_rows * erow_bytes,
                                             std::size_t{R} * erow_bytes, cudaMemcpyHostToDevice, st));
            EngramView e;
            e.row_bytes = b.engram_rows.get();
            e.rows = R;
            e.cols = ecols;
            e.hidden = H;
            e.kv = {w.engram_kv, 5U * H, ecols * 256U};
            e.q_norm = w.engram_q_norm;
            e.k_norm = w.engram_k_norm;
            e.stream = b.stream.get();
            e.workspace = b.workspace.get();
            enqueue_engram(e, st);
        }
        op(o.decode_fuse_hc ? "hc_norm_attn" : "hc_mix_attn");
        {
            auto& mv = hc_mix;
            mv.stream = b.stream.get();
            mv.rows = R;
            mv.hidden = H;
            mv.fn = w.hc_attn_fn;
            mv.base = w.hc_attn_base;
            mv.scale = w.hc_attn_scale;
            mv.iters = g.hc_iters;
            mv.hc_eps = g.hc_eps;
            mv.mixes = b.mix_attn.get();
            mv.partial = reinterpret_cast<float*>(b.workspace.get());
            if (!o.decode_fuse_hc) enqueue_hc_mixes(mv, st);
        }
        if (!o.decode_fuse_hc) op("norm_attn");
        {
            NormView n;
            n.hc_stream = b.stream.get();
            n.pre = b.pre_carry.get();
            n.pre_stride = 4;
            n.weight = w.attn_norm;
            n.rows = R;
            n.n = H;
            n.out = b.x.get();
            n.a8 = b.xa.view();
            n.a8.rows = R;
            if (o.decode_fuse_hc) enqueue_hc_mixes_norm(hc_mix, n, st);
            else enqueue_norm(n, st);
        }
        op("qa_kv");
        {
            Q8A8View v;
            v.tiled = w.qa_kv.view();
            v.count = b.qkv_stride;
            v.x = b.xa.view();
            v.x.rows = R;
            v.out = b.qkv.get();
            v.split_scratch = b.split_scratch.get();
            v.split_scratch_floats = b.split_scratch.size();
            if (o.decode_qakv_s1) v.split = 1U;
            v.pdl = o.decode_pdl;
            enqueue_q8_0_a8(v, st);
        }
        op("q_norm");
        {
            NormView n;
            n.x = b.qkv.get();
            n.x_stride = b.qkv_stride;
            n.weight = w.q_a_norm;
            n.rows = R;
            n.n = g.q_lora;
            n.out = b.qr.get();
            n.a8 = b.qa.view();
            n.a8.rows = R;
            enqueue_norm(n, st);
        }
        // decode_attn_fork: the attention preparation (reads x, qr, kv, pos) on the side branch while q_b runs
        const float* kv_rows = b.qkv.get() + b.kv_offset;  // the window kernel reads contiguous 512-value kv rows
        if (R > 1U) {
            STRIX_CUDA_CHECK(cudaMemcpy2DAsync(D.kvrows.get(), at::kHeadDim * 4U, kv_rows, std::size_t{b.qkv_stride} * 4U, at::kHeadDim * 4U, R,
                                               cudaMemcpyDeviceToDevice, st));
            kv_rows = D.kvrows.get();
        }
        at::LayerIo lio;
        lio.rows = R;
        lio.pos = D.pos.get();
        lio.pos_bound = pos_bound;
        lio.x = b.x.get();
        lio.qr = b.qr.get();
        lio.q = b.q.get();
        lio.kv = kv_rows;
        lio.o = b.o.get();
        if (o.decode_attn_fork) {
            if (!D.shared_side) D.shared_side = std::make_unique<Stream>();
            while (D.attn_fork.size() <= layer) {
                D.attn_fork.push_back(std::make_unique<Event>());
                D.attn_join.push_back(std::make_unique<Event>());
            }
            const cudaStream_t side = D.shared_side->get();
            STRIX_CUDA_CHECK(cudaEventRecord(D.attn_fork[layer]->e, st));
            STRIX_CUDA_CHECK(cudaStreamWaitEvent(side, D.attn_fork[layer]->e, 0));
            m.seq0().attn->enqueue_layer(layer, aw, lio, side, at::AttentionState::Phase::prep);
            STRIX_CUDA_CHECK(cudaEventRecord(D.attn_join[layer]->e, side));
        }
        op("q_b");
        {
            Q8A8View qb;
            qb.tiled = w.q_b.view();
            qb.count = g.q_out;
            qb.x = b.qa.view();
            qb.x.rows = R;
            qb.out = b.q.get();
            qb.split_scratch = b.split_scratch.get();
            qb.split_scratch_floats = b.split_scratch.size();
            qb.pdl = o.decode_pdl;
            enqueue_q8_0_a8(qb, st);
        }
        op("attention");
        if (o.decode_attn_fork) {
            STRIX_CUDA_CHECK(cudaStreamWaitEvent(st, D.attn_join[layer]->e, 0));
            m.seq0().attn->enqueue_layer(layer, aw, lio, st, at::AttentionState::Phase::attend);
        } else {
            m.seq0().attn->enqueue_layer(layer, aw, lio, st);
        }
        op("o_a_o_b");
        {
            const std::uint32_t ol = g.o_groups * g.o_lora, half = ol / 2U;
            auto oa = b.oa.view();
            oa.rows = R;
            enqueue_quantize_a8({b.o.get(), R, g.q_out, 0, oa}, st);
            Q8A8View a;
            a.tiled = w.o_a.view();
            a.count = ol;
            a.per_group = g.o_lora;
            a.width = g.q_out / g.o_groups;
            a.x = oa;
            a.out = b.o_low.get();
            a.split_scratch = b.split_scratch.get();
            a.split_scratch_floats = b.split_scratch.size();
            a.pdl = o.decode_pdl;
            enqueue_q8_0_a8(a, st);
            auto la = b.la.view();
            la.rows = R;
            enqueue_quantize_a8({b.o_low.get(), R, ol, 0, la}, st);
            for (std::uint32_t h = 0; h < (o.decode_ob_one ? 1U : 2U); ++h) {
                Q8A8View v;
                v.tiled = w.o_b.view();
                v.count = H;
                v.k_offset = h * half;
                v.width = half;
                v.x = {la.codes + h * half, la.deltas + h * half / 32U, la.sums + h * half / 32U, R, half, ol};
                v.out = b.y.get();
                v.accumulate = h > 0;
                if (o.decode_ob_one) v.k_halves = 2U;  // both halves, same bits
                if (o.decode_ob_one && o.decode_ob_expand) {  // the hc_expand below, in the epilogue
                    v.hc_residual = b.stream.get();
                    v.hc_mixes = b.mix_attn.get();
                    v.hc_out = b.stream.get();
                    v.hc_hidden = H;
                }
                v.split_scratch = b.split_scratch.get();
                v.split_scratch_floats = b.split_scratch.size();
                v.pdl = o.decode_pdl;
            enqueue_q8_0_a8(v, st);
            }
        }
        op("hc_post");
        if (!(o.decode_ob_one && o.decode_ob_expand)) enqueue_hc_expand({b.y.get(), 0, b.stream.get(), b.mix_attn.get(), R, H, b.stream.get()}, st);
        mark(layer, 1);
        op(o.decode_fuse_hc ? "hc_norm_ffn" : "hc_mix_ffn");
        {
            auto& mv = hc_mix;
            mv.stream = b.stream.get();
            mv.rows = R;
            mv.hidden = H;
            mv.fn = w.hc_ffn_fn;
            mv.base = w.hc_ffn_base;
            mv.scale = w.hc_ffn_scale;
            mv.iters = g.hc_iters;
            mv.hc_eps = g.hc_eps;
            mv.mixes = b.mix_ffn.get();
            mv.partial = reinterpret_cast<float*>(b.workspace.get());
            if (!o.decode_fuse_hc) enqueue_hc_mixes(mv, st);
        }
        if (!o.decode_fuse_hc) op("norm_ffn");
        {
            NormView n;
            n.hc_stream = b.stream.get();
            n.pre = b.mix_attn.get();
            n.pre_stride = 24;
            n.weight = w.ffn_norm;
            n.rows = R;
            n.n = H;
            n.out = b.x.get();
            n.a8 = b.xa.view();
            n.a8.rows = R;
            if (o.decode_fuse_hc) enqueue_hc_mixes_norm(hc_mix, n, st);
            else enqueue_norm(n, st);
        }
        // decode_shared_fork: the fork point is here, before the router (the shared expert reads only the norm's A8)
        const bool shared_fork = o.decode_shared_fork && hooked;
        if (shared_fork) {
            if (!D.shared_side) D.shared_side = std::make_unique<Stream>();
            while (D.shared_fork.size() <= layer) {
                D.shared_fork.push_back(std::make_unique<Event>());
                D.shared_join.push_back(std::make_unique<Event>());
            }
            STRIX_CUDA_CHECK(cudaEventRecord(D.shared_fork[layer]->e, st));
        }
        op("router");
        {
            RouterView rv;
            rv.x = b.x.get();
            rv.rows = R;
            rv.weight = {w.router, g.experts, H};
            rv.bias = w.bias;
            rv.bias_image = w.bias_image;
            rv.top_k = g.top_k;
            rv.route_scale = g.route_scale;
            rv.scores = b.scores.get();
            rv.ids = b.route_ids.get();
            rv.weights = b.route_w.get();
            if (o.decode_router_v2 && router_v2_supported(rv)) enqueue_router_v2(rv, D.router_counters.get(), st);
            else enqueue_router(rv, st);
        }
        const auto shared_expert = [&](cudaStream_t s) {
            SharedExpertView e;
            e.gate_up_tiled = w.gate_up.view();
            e.down_tiled = w.down.view();
            e.limit = g.swiglu_limit;
            e.x = b.xa.view();
            e.x.rows = R;
            e.out = b.shared.get();
            e.workspace = b.workspace.get();
            e.pdl = o.decode_pdl;
#ifdef STRIX_DS41_CUDA_TRELLIS
            if (!m.tshexp.empty()) {  // trellis shared expert: F32 x -> gate, up -> clamped SwiGLU (F32) -> down
                const auto& ts = m.tshexp[layer - o.first_layer];
                const auto linear = [&](const Impl::TrellisLinear& t, const float* x, float* y) {
                    trellis::Linear v;
                    v.weights = t.w;
                    v.x = x;
                    v.y = y;
                    v.rows = R;
                    v.x_stride = t.w.input;
                    v.y_stride = t.w.output;
                    v.x_bytes = std::size_t{R} * t.w.input * 4U;
                    v.y_bytes = std::size_t{R} * t.w.output * 4U;
                    v.splits = t.splits;
                    v.mode = trellis::Mode::gemv_quality;
                    v.workspace = m.tr_ws.get();
                    v.workspace_bytes = m.tr_ws_bytes;
                    trellis::enqueue_linear(v, s);
                };
                linear(ts[0], b.x.get(), m.tr_gate.get());
                linear(ts[1], b.x.get(), m.tr_up.get());
                const std::size_t n = std::size_t{R} * g.inter;
                trellis_swiglu_kernel<<<static_cast<unsigned>((n + 255U) / 256U), 256, 0, s>>>(m.tr_gate.get(), m.tr_up.get(), n, g.swiglu_limit,
                                                                                               m.tr_act.get());
                STRIX_CUDA_CHECK(cudaGetLastError());
                linear(ts[2], m.tr_act.get(), b.shared.get());
            } else
#endif
            if (o.decode_fuse_shared && R == 1U) enqueue_shared_expert_fused(e, s);
            else enqueue_shared_expert(e, s);
        };
        if (shared_fork) {
            const cudaStream_t side = D.shared_side->get();
            STRIX_CUDA_CHECK(cudaStreamWaitEvent(side, D.shared_fork[layer]->e, 0));
            shared_expert(side);
            STRIX_CUDA_CHECK(cudaEventRecord(D.shared_join[layer]->e, side));
        }
        if (hooked) {  // the caller's routed-expert site (it enqueues the shared expert, or joins its side branch)
            op("moe_site");
            mark(layer, 2);
            CudaMoeSite site;
            site.layer = layer;
            site.rows = R;
            site.stream = st;
            site.x = b.xa.view();
            site.x.rows = R;
            site.ids = b.route_ids.get();
            site.weights = b.route_w.get();
            site.shared = b.shared.get();
            site.raw = m.raw.get();
            site.y = b.y.get();
            if (shared_fork) site.shared_join = D.shared_join[layer]->e;
            else site.shared_expert = shared_expert;
            if (o.decode_moe_expand) {  // the hc_post below, in the site's combine
                site.hc_residual = b.stream.get();
                site.hc_mixes = b.mix_ffn.get();
                site.hc_out = b.stream.get();
                site.hc_pre_carry = o.decode_fuse_hc ? b.pre_carry.get() : nullptr;
            }
            site.x_f32 = b.x.get();
            site.route_other = [&core, &b, &g, &o, &D, R, H](std::uint32_t ly, std::uint32_t k, float* sc, std::uint32_t* ids, float* wts, cudaStream_t s2) {
                const auto& w2 = core.weights(ly);
                RouterView rv;
                rv.x = b.x.get();
                rv.rows = R;
                rv.weight = {w2.router, g.experts, H};
                rv.bias = w2.bias;
                rv.bias_image = w2.bias_image;
                rv.top_k = k;
                rv.route_scale = g.route_scale;
                rv.scores = sc;
                rv.ids = ids;
                rv.weights = wts;
                if (o.decode_router_v2 && router_v2_supported(rv)) enqueue_router_v2(rv, D.router_counters.get() + 1, s2);
                else enqueue_router(rv, s2);
            };
            io.moe_site(site);
        } else {
        op("shared");
        shared_expert(st);
        // ---- routed experts
        ExpertWeights ew;
        if (m.tiers[layer] == ExpertTier::device) {
            ew = m.experts->acquire(layer, st);
        } else {
            // sparse: only the routed experts' bytes of this layer, read now (synchronises the step)
            op("expert_read");
            const auto t0 = std::chrono::steady_clock::now();
            const auto& x = m.extents[layer];
            if (!D.slot) {
                std::uint64_t lb = 0;
                for (const auto& e : m.extents) lb = std::max(lb, e.total());
                D.slot = std::make_unique<DeviceBuffer<std::uint8_t>>(lb);
                D.reader = std::make_unique<FileReader>(m.file_path, true, 8, o.weight_read);
                D.staging = std::make_unique<PinnedBuffer>((std::size_t{8} << 20) + 2U * FileReader::alignment);
            }
            route_ids_host.resize(std::size_t{R} * K);
            STRIX_CUDA_CHECK(cudaMemcpyAsync(route_ids_host.data(), b.route_ids.get(), route_ids_host.size() * 4U, cudaMemcpyDeviceToHost, st));
            STRIX_CUDA_CHECK(cudaStreamSynchronize(st));
            std::vector<std::uint32_t> uniq(route_ids_host);
            std::sort(uniq.begin(), uniq.end());
            uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
            const std::uint64_t per[3] = {m.moe_shape.gate_expert_bytes(), m.moe_shape.gate_expert_bytes(), m.moe_shape.down_expert_bytes()};
            const std::uint64_t at_slot[3] = {0, x.bytes[0], x.bytes[0] + x.bytes[1]};
            const std::uint64_t A = FileReader::alignment;
            for (const auto e : uniq) {
                require(e < c.n_routed_experts, "routed expert id");
                for (int k = 0; k < 3; ++k) {
                    const std::uint64_t off = x.offset[k] + std::uint64_t{e} * per[k], a0 = off / A * A;
                    const std::uint64_t a1 = std::min<std::uint64_t>((off + per[k] + A - 1U) / A * A, D.reader->size());
                    D.reader->read_range(a0, static_cast<std::size_t>(a1 - a0), D.staging->get());
                    STRIX_CUDA_CHECK(cudaMemcpy(D.slot->get() + at_slot[k] + std::uint64_t{e} * per[k], D.staging->get() + (off - a0), per[k],
                                                cudaMemcpyHostToDevice));
                }
            }
            expert_io += seconds_since(t0);
            ew = {D.slot->get(), D.slot->get() + at_slot[1], D.slot->get() + at_slot[2]};
        }
        mark(layer, 2);  // after a sparse read: moe_ms is the device time of the routed experts + hc_post
        op("moe");
        {
            MoeLayerView v;
            v.shape = m.moe_shape;
            v.weights = ew;
            v.rows = R;
            v.split = m.split;
            v.x = b.xa.view();
            v.x.rows = R;
            v.ids = b.route_ids.get();
            v.route_weights = b.route_w.get();
            v.shared = b.shared.get();
            v.raw = m.raw.get();
            v.raw1 = o.expert_tp ? m.raw1.get() : nullptr;  // ds41-expert-tp2-i1152-v1 sum order (decode steps)
            v.y = b.y.get();
            v.workspace = m.moe_ws.get();
            v.capacity = o.decode_only ? o.decode_rows : o.moe_chunk_rows;  // the rows moe_ws / raw were sized for
            enqueue_moe_layer(v, st);
        }
        }  // hooked
        op("hc_post");
        if (!(hooked && o.decode_moe_expand))
            enqueue_hc_expand({b.y.get(), 0, b.stream.get(), b.mix_ffn.get(), R, H, b.stream.get(),
                               o.decode_fuse_hc ? b.pre_carry.get() : nullptr}, st);
        // the next attention input collapses with this site's pre
        if (!o.decode_fuse_hc) STRIX_CUDA_CHECK(cudaMemcpy2DAsync(b.pre_carry.get(), 4U * sizeof(float), b.mix_ffn.get(), 24U * sizeof(float), 4U * sizeof(float), R,
                                           cudaMemcpyDeviceToDevice, st));
        op("end");
        mark(layer, 3);
        if (o.stream_hashes && !use_graphs) {
            std::vector<float> row(SR);
            STRIX_CUDA_CHECK(cudaMemcpyAsync(row.data(), b.stream.get() + std::size_t{R - 1U} * SR, SR * 4U, cudaMemcpyDeviceToHost, st));
            STRIX_CUDA_CHECK(cudaStreamSynchronize(st));
            result.stream_hashes.push_back(fnv_bytes(row.data(), SR * 4U));
        }
    };

    // ---- segments: [first, e1), [e1, e2), ... cut at the step's Engram layers (prologue in the first, head in the last)
    std::vector<std::uint32_t> cuts{first};
    for (std::uint32_t l = first + 1U; l < stop; ++l)
        if (engram_li[l] != ~0U) cuts.push_back(l);
    cuts.push_back(stop);
    const std::size_t nseg = cuts.size() - 1U;
    if (cap.exec.size() != nseg) cap.exec.assign(nseg, nullptr);
    const auto segment_body = [&](std::size_t j) {
        if (j == 0U) {
            STRIX_CUDA_CHECK(cudaMemcpyAsync(D.pos.get(), hpos, std::size_t{R} * 8U, cudaMemcpyHostToDevice, st));
            if (embed) {
                STRIX_CUDA_CHECK(cudaMemcpyAsync(b.tokens.get(), htok, std::size_t{R} * 4U, cudaMemcpyHostToDevice, st));
                enqueue_embed({{m.file->device_f16("token_embd.weight"), V, H}, b.tokens.get(), R, b.stream.get(), b.pre_carry.get()}, st);
            } else {
                STRIX_CUDA_CHECK(cudaMemcpyAsync(b.stream.get(), D.io_in.get(), std::size_t{R} * SR * 4U, cudaMemcpyHostToDevice, st));
                STRIX_CUDA_CHECK(
                    cudaMemcpyAsync(b.mix_ffn.get(), D.io_in.get() + std::size_t{R} * SR * 4U, std::size_t{R} * 24U * 4U, cudaMemcpyHostToDevice, st));
                STRIX_CUDA_CHECK(cudaMemcpy2DAsync(b.pre_carry.get(), 4U * sizeof(float), b.mix_ffn.get(), 24U * sizeof(float), 4U * sizeof(float), R,
                                                   cudaMemcpyDeviceToDevice, st));
            }
        }
        for (std::uint32_t l = cuts[j]; l < cuts[j + 1U]; ++l) layer_body(l);
        if (j + 1U == nseg && io.sections_cut != 0U) {
            if (!sec_kv.empty()) {
                SectionKvJobs jobs{};
                jobs.count = static_cast<std::uint32_t>(sec_kv.size());
                jobs.pos = D.pos.get();
                for (std::size_t i = 0; i < sec_kv.size(); ++i) {
                    const auto& cc = m.seq0().attn->caches(sec_kv[i].layer);
                    jobs.job[i] = {cc.kv_cache, cc.idx_cache, sec_kv[i].ratio, sec_kv[i].n, D.sec_dev.get() + sec_kv[i].at};
                }
                section_kv_kernel<<<jobs.count, 256, 0, st>>>(jobs);
                STRIX_CUDA_CHECK(cudaGetLastError());
            }
            for (const auto& t : sec_topk) {
                const auto& cc = m.seq0().attn->caches(t.layer);
                STRIX_CUDA_CHECK(cudaMemcpyAsync(D.sec_dev.get() + t.at, cc.idx_count, std::size_t{R} * 4U, cudaMemcpyDeviceToDevice, st));
                STRIX_CUDA_CHECK(cudaMemcpyAsync(D.sec_dev.get() + t.at + std::size_t{R} * 4U, cc.idx, std::size_t{R} * at::kTopK * 4U,
                                                 cudaMemcpyDeviceToDevice, st));
            }
            if (sec_cand >= 0)
                STRIX_CUDA_CHECK(cudaMemcpyAsync(D.sec_dev.get() + sec_cand_at, m.seq0().attn->candidates(), std::size_t{R} * m.seq0().attn->cand_stride(),
                                                 cudaMemcpyDeviceToDevice, st));
            store_mapped(D.sec_host.get(), D.sec_dev.get(), sec_bytes, st);
        }
        if (j + 1U == nseg) {
            if (want_out) {
                store_mapped(D.io_out.get(), b.stream.get(), std::size_t{R} * SR * 4U, st);
                store_mapped(D.io_out.get() + std::size_t{R} * SR * 4U, b.mix_ffn.get(), std::size_t{R} * 24U * 4U, st);
            }
            if (head) {
                op("head");
                core.head(st);
                store_mapped(hcand, b.candidates.get(), std::size_t{R} * 12U, st);
                op("end");
            }
        }
    };
    // A missing segment graph is captured and instantiated here. Every one of the step's missing segments is captured
    // before the step's first launch (see the launch loop): instantiating while an earlier segment of the same gated
    // step runs can block on the device (a spinning device-gated wait) and deadlock the service until the wait deadline
    // (review finding: token 71 = the first R4 capture, 'a wait timed out at gated layer 1').
    const auto capture_segment = [&](std::size_t j) {
        cudaGraphExec_t& exec = cap.exec[j];
        if (exec == nullptr) {
            STRIX_CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            capturing = true;
            try {
                segment_body(j);
            } catch (...) {
                cudaGraph_t broken = nullptr;
                cudaStreamEndCapture(st, &broken);
                if (broken) cudaGraphDestroy(broken);
                capturing = false;
                throw;
            }
            capturing = false;
            cudaGraph_t graph = nullptr;
            STRIX_CUDA_CHECK(cudaStreamEndCapture(st, &graph));
            const cudaError_t inst = cudaGraphInstantiate(&exec, graph, 0ULL);
            cudaGraphDestroy(graph);
            STRIX_CUDA_CHECK(inst);
            ++result.graph_captures;
        }
    };
    const auto run_segment = [&](std::size_t j) {
        if (!use_graphs) {
            segment_body(j);
            return;
        }
        capture_segment(j);  // a no-op: the launch loop captured every missing segment first
        STRIX_CUDA_CHECK(cudaGraphLaunch(cap.exec[j], st));
    };
    // Engram rows of the step's tokens (the history follows every step): one job on the Engram I/O thread with the
    // rows of every Engram layer of the step, in layer order; the host waits for a layer's rows right before the
    // segment that starts with it, while the earlier segments run. A head step of one row issues the job of its greedy
    // token at once (prefetch); the next step uses it when it feeds that token at the next position.
    const std::size_t per = std::size_t{elayers} * ecols;
    const bool file_engram = o.engram == "file";
    // Engram layers of the step and of the object (a prefetch is issued before the next step's stop layer is known)
    std::vector<std::uint32_t> step_li, object_li;
    for (std::uint32_t l = first; l < o.end_layer; ++l)
        if (engram_li[l] != ~0U) {
            object_li.push_back(engram_li[l]);
            if (l < stop) step_li.push_back(engram_li[l]);
        }
    if (file_engram && !D.engram_io)
        D.engram_io = std::make_unique<EngramIoThread>(*m.engram_reader, std::size_t{o.engram_cache_mb} * (std::size_t{1} << 20) / 264U);
    // the job of `nrows` rows `rows` ([row][layer][column]) of the Engram layers `lis` into D.engram from row `r0` on
    // (r0 = 1: row 0 came with a prefetch; its records stay in the re-read check)
    const auto submit_engram = [&](const std::vector<std::uint64_t>& rows, std::uint32_t nrows, const std::vector<std::uint32_t>& lis,
                                   std::uint32_t r0) {
        std::vector<std::vector<std::uint64_t>> offs(lis.size());
        std::vector<std::uint8_t*> outs(lis.size());
        for (std::size_t i = 0; i < lis.size(); ++i) {
            const std::uint32_t li = lis[i];
            outs[i] = D.engram.get() + (std::size_t{li} * o.decode_rows + r0) * erow_bytes;
            offs[i].resize(std::size_t{nrows} * ecols);
            for (std::uint32_t r = 0; r < nrows; ++r)
                for (std::uint32_t k = 0; k < ecols; ++k) {
                    const std::uint64_t row = rows[std::size_t{r} * per + std::size_t{li} * ecols + k];
                    if (row >= m.engram_tables[li].rows) throw std::runtime_error("ds41 cuda decode: Engram row out of range");
                    offs[i][std::size_t{r} * ecols + k] = m.engram_tables[li].file_offset + row * 264U;
                }
        }
        if (m.engram_check) {
            if (r0 == 0U) {
                D.check_offsets = offs;
                D.check_outs = outs;
            } else {
                D.check_offsets.insert(D.check_offsets.end(), offs.begin(), offs.end());
                D.check_outs.insert(D.check_outs.end(), outs.begin(), outs.end());
            }
        }
        D.engram_io->submit(std::move(offs), std::move(outs));
    };
    bool prefetched = false, row0 = false;
    if (D.pf_pending) {
        D.pf_pending = false;
        const bool same = P == D.pf_position && tokens[0] == D.pf_token;
        prefetched = R == 1U && same;
        row0 = R > 1U && same;  // a window whose row 0 was prefetched: rows 1..R-1 follow
        if (!prefetched && !row0) {
            // another token at this position: the prefetched rows are dropped (its job ends first: same buffer)
            const auto tw = std::chrono::steady_clock::now();
            D.engram_io->wait_idle();
            const double waited = seconds_since(tw);  // before any segment: the GPU idles
            result.engram_wait_seconds += waited;
            result.engram_wait_first_seconds += waited;
            m.hasher->truncate(P);
        }
    }
    result.engram_prefetched = prefetched || row0;
    if (!prefetched) {
        const std::uint32_t r0 = row0 ? 1U : 0U;
        std::vector<std::uint64_t> all(std::size_t{R - r0} * per);
        for (std::uint32_t r = r0; r < R; ++r)
            m.hasher->push(tokens[r], std::span<std::uint64_t>(all.data() + std::size_t{r - r0} * per, per));
        if (file_engram) {
            if (D.engram_io) {
                const auto tw = std::chrono::steady_clock::now();
                D.engram_io->wait_idle();  // the previous job (row 0's prefetch: its rows are complete after this)
                const double waited = seconds_since(tw);
                result.engram_wait_seconds += waited;
                result.engram_wait_first_seconds += waited;
            }
            submit_engram(all, R - r0, step_li, r0);
        } else {
            for (const auto li : step_li) std::memset(D.engram.get() + std::size_t{li} * o.decode_rows * erow_bytes, 0, std::size_t{R} * erow_bytes);
        }
    }
    // the job's layers (a prefetched job has every Engram layer of the object, in order)
    const std::vector<std::uint32_t>& job_li = prefetched ? object_li : step_li;
    const auto wait_engram = [&](std::uint32_t layer) {
        const std::uint32_t li = engram_li[layer];
        if (li == ~0U || !file_engram) return;
        const auto at = static_cast<std::uint32_t>(std::find(job_li.begin(), job_li.end(), li) - job_li.begin());
        const auto t0 = std::chrono::steady_clock::now();
        D.engram_io->wait_layer(at, o.decode_spin);
        const double waited = seconds_since(t0);
        result.engram_wait_seconds += waited;
        if (at == 0U) result.engram_wait_first_seconds += waited;
    };

    if (use_graphs)  // capture + instantiate before anything of the step runs (nothing of the gated step spins yet)
        for (std::size_t j = 0; j < nseg; ++j) capture_segment(j);
    STRIX_CUDA_CHECK(cudaEventRecord(D.start->e, st));
    for (std::size_t j = 0; j < nseg; ++j) {
        wait_engram(cuts[j]);
        run_segment(j);
    }
    STRIX_CUDA_CHECK(cudaEventRecord(D.done->e, st));
    result.enqueue_seconds = seconds_since(t_step);
    if (io.after_launch) io.after_launch();  // the caller's host work while the step runs
    wait_event(D.done->e, o.decode_spin);
    if (!use_graphs) ++cap.eager_steps;
    if (file_engram && job_li.size() == step_li.size()) {
        // every layer of the job is in already (the segments waited for each)
        D.engram_io->wait_idle();
        result.engram_seconds = D.engram_io->last_seconds();
        result.engram_issue_seconds = D.engram_io->last_issue_seconds();
        if (!D.check_offsets.empty()) {
            if (!D.check_file) D.check_file = std::make_unique<std::ifstream>(m.gguf_path, std::ios::binary);
            std::uint8_t rec[264];
            for (std::size_t i = 0; i < D.check_offsets.size(); ++i)
                for (std::size_t k = 0; k < D.check_offsets[i].size(); ++k) {
                    D.check_file->seekg(static_cast<std::streamoff>(D.check_offsets[i][k]));
                    D.check_file->read(reinterpret_cast<char*>(rec), 264);
                    ++result.engram_checked;
                    if (!*D.check_file || std::memcmp(rec, D.check_outs[i] + k * 264U, 264U) != 0) ++result.engram_check_mismatches;
                    D.check_file->clear();
                }
            D.check_offsets.clear();
        }
    }
    // prefetch: the rows of the greedy token of a one-row head step, issued before the downloads and the caller's work
    if (head && R == 1U && file_engram && o.engram_prefetch && hcand[2] != 0U && P + 1U < o.max_positions) {
        const std::uint32_t next = hcand[1];
        std::vector<std::uint64_t> rows(per);
        m.hasher->push(next, std::span<std::uint64_t>(rows.data(), per));
        submit_engram(rows, 1U, object_li, 0U);
        D.pf_pending = true;
        D.pf_position = P + 1U;
        D.pf_token = next;
    }

    // ---- results
    if (head) {
        result.tokens.resize(R);
        for (std::uint32_t r = 0; r < R; ++r) {
            if (hcand[r * 3U + 2U] == 0U) throw std::runtime_error("ds41 cuda decode: non-finite logits");
            result.tokens[r] = hcand[r * 3U + 1U];
        }
        if (want_logits) {
            STRIX_CUDA_CHECK(cudaMemcpyAsync(D.logits.get(), b.logits.get(), std::size_t{R} * V * 4U, cudaMemcpyDeviceToHost, st));
            STRIX_CUDA_CHECK(cudaStreamSynchronize(st));
            const auto* L = reinterpret_cast<const float*>(D.logits.get());
            result.logits.assign(L, L + std::size_t{R} * V);
            for (std::uint32_t r = 0; r < R; ++r) result.logits_fnv.push_back(fnv_bytes(L + std::size_t{r} * V, std::size_t{V} * 4U));
        }
    }
    if (want_out) {
        if (io.stream_out) std::memcpy(io.stream_out, D.io_out.get(), std::size_t{R} * SR * 4U);
        if (io.mix_out) std::memcpy(io.mix_out, D.io_out.get() + std::size_t{R} * SR * 4U, std::size_t{R} * 24U * 4U);
        if (io.stream_view) {
            result.stream_view = reinterpret_cast<const float*>(D.io_out.get());
            result.mix_view = reinterpret_cast<const float*>(D.io_out.get() + std::size_t{R} * SR * 4U);
        }
    }
    STRIX_CUDA_CHECK(cudaEventElapsedTime(&result.device_ms, D.start->e, D.done->e));
    if (timing) {
        const auto& mk = use_graphs ? cap.marks : marks;
        result.layers.assign(c.num_hidden_layers, {});
        for (std::uint32_t l = first; l < stop && l - first < mk.size(); ++l) {
            const auto& a = mk[l - first];
            if (a[0] == nullptr || a[3] == nullptr) continue;
            auto& t = result.layers[l];
            t.rows = R;
            STRIX_CUDA_CHECK(cudaEventElapsedTime(&t.total_ms, a[0], a[3]));
            STRIX_CUDA_CHECK(cudaEventElapsedTime(&t.attention_ms, a[0], a[1]));
            STRIX_CUDA_CHECK(cudaEventElapsedTime(&t.moe_ms, a[2], a[3]));
        }
        for (std::size_t i = 0; i + 1U < op_marks.size(); ++i) {
            if (std::strcmp(op_marks[i].first, "end") == 0) continue;
            float ms = 0.0F;
            STRIX_CUDA_CHECK(cudaEventElapsedTime(&ms, op_marks[i].second, op_marks[i + 1U].second));
            result.ops[op_marks[i].first] += ms;
        }
    }
    result.expert_io_seconds = expert_io;
    if (io.sections_cut != 0U) {
        auto& S = result.sections;
        const std::uint8_t* h = D.sec_host.get();
        for (const auto& k : sec_kv) {
            CudaStepSections::Kv e;
            e.layer = k.layer;
            e.ratio = k.ratio;
            e.first = P / k.ratio;
            e.count = (P + R) / k.ratio - e.first;
            e.kv.assign(h + k.at, h + k.at + std::size_t{e.count} * 288U);
            e.idx_keys.assign(h + k.at + std::size_t{k.n} * 288U, h + k.at + std::size_t{k.n} * 288U + std::size_t{e.count} * 68U);
            S.kv.push_back(std::move(e));
        }
        for (const auto& t : sec_topk) {
            CudaStepSections::Topk e;
            e.layer = t.layer;
            const auto* w = reinterpret_cast<const std::uint32_t*>(h + t.at);
            e.counts.assign(w, w + R);
            e.ids.assign(w + R, w + R + std::size_t{R} * at::kTopK);
            S.topk.push_back(std::move(e));
        }
        S.candidate_source = sec_cand;
        S.cand_stride = m.seq0().attn->cand_stride();
        if (sec_cand >= 0) S.candidates.assign(h + sec_cand_at, h + sec_cand_at + std::size_t{R} * m.seq0().attn->cand_stride());
    }
    // ---- the sequence advances
    m.seq0().last_T = P + R;
    m.seq0().last_tokens.insert(m.seq0().last_tokens.end(), tokens.begin(), tokens.end());
    if (head) m.seq0().last_next = result.tokens.back();
    m.last_batch_ctx.assign(1, 0U);
    m.last_row = R - 1U;
    m.last_step_rows = R;
    result.seconds = seconds_since(t_step);
    return result;
}

void CudaPrefill::set_engram_check(bool on) noexcept { impl_->engram_check = on; }

void CudaPrefill::wait_io() {
    auto& m = *impl_;
    if (m.dec && m.dec->engram_io) m.dec->engram_io->wait_idle();
}

void CudaPrefill::prefetch_engram(std::uint32_t token) {
    auto& m = *impl_;
    const auto& o = m.o;
    const auto& c = m.cfg;
    auto* D = m.dec.get();
    if (!D || !D->engram_io || o.engram != "file" || !o.engram_prefetch) return;
    require(m.last_step_rows == 0U, "prefetch_engram after commit_window");
    require(token < c.vocab_size, "prefetch_engram: token in the vocabulary");
    const std::uint32_t P = m.seq0().last_T;
    if (P + 1U >= o.max_positions) return;
    if (D->pf_pending) {
        if (D->pf_position == P && D->pf_token == token) return;  // the head step issued it already
        D->engram_io->wait_idle();
        D->pf_pending = false;
        m.hasher->truncate(P);
    }
    const std::uint32_t ecols = m.g.engram_cols, elayers = static_cast<std::uint32_t>(c.engram_layer_ids.size());
    const std::size_t per = std::size_t{elayers} * ecols, erow_bytes = std::size_t{ecols} * 264U;
    if (D->engram.size() < std::size_t{elayers} * o.decode_rows * erow_bytes) return;  // no step has sized the buffer
    D->engram_io->wait_idle();  // one job at a time
    std::vector<std::uint32_t> lis;  // the object's Engram layers in layer order (as the head step's prefetch)
    for (std::uint32_t l = o.first_layer; l < o.end_layer; ++l)
        for (std::uint32_t li = 0; li < elayers; ++li)
            if (c.engram_layer_ids[li] == l) lis.push_back(li);
    std::vector<std::uint64_t> rows(per);
    m.hasher->push(token, std::span<std::uint64_t>(rows.data(), per));
    std::vector<std::vector<std::uint64_t>> offs(lis.size());
    std::vector<std::uint8_t*> outs(lis.size());
    for (std::size_t i = 0; i < lis.size(); ++i) {
        const std::uint32_t li = lis[i];
        outs[i] = D->engram.get() + std::size_t{li} * o.decode_rows * erow_bytes;
        offs[i].resize(ecols);
        for (std::uint32_t k = 0; k < ecols; ++k) {
            const std::uint64_t row = rows[std::size_t{li} * ecols + k];
            if (row >= m.engram_tables[li].rows) throw std::runtime_error("ds41 cuda decode: Engram row out of range");
            offs[i][k] = m.engram_tables[li].file_offset + row * 264U;
        }
    }
    if (m.engram_check) {
        D->check_offsets = offs;
        D->check_outs = outs;
    }
    D->engram_io->submit(std::move(offs), std::move(outs));
    D->pf_pending = true;
    D->pf_position = P;
    D->pf_token = token;
}

void CudaPrefill::commit_window(std::uint32_t kept, std::uint32_t next_token) {
    auto& m = *impl_;
    const std::uint32_t R = m.last_step_rows;
    require(R != 0U, "commit_window after a step");
    require(kept >= 1U && kept <= R, "commit_window: kept rows 1..R");
    require(next_token < m.cfg.vocab_size, "commit_window: next token in the vocabulary");
    m.last_step_rows = 0;
    m.seq0().last_next = next_token;
    if (kept == R) return;
    const std::uint32_t P = m.seq0().last_T - R, T = P + kept;
    auto* D = m.dec.get();
    if (D && D->pf_pending) {  // a prefetch is only issued after a one-row step (kept == R): defensive
        D->pf_pending = false;
        D->engram_io->wait_idle();
    }
    // STRIX_DS41_CUDA_COMMIT_NEGATIVE_CONTROL=1 (tests only): leave the carries of the rejected rows (must fail the gates)
    static const bool negative = std::getenv("STRIX_DS41_CUDA_COMMIT_NEGATIVE_CONTROL") != nullptr;
    if (!negative) m.seq0().attn->commit_rows(m.o.first_layer, m.o.end_layer, T - 1U, kept - 1U, m.st);
    m.seq0().last_T = T;
    m.seq0().last_tokens.resize(T);
    m.last_row = kept - 1U;
    m.hasher->truncate(T);
}

}  // namespace strix::ds41::cuda
