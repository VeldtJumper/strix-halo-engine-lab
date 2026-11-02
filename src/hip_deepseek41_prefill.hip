#include "strix/file_snapshot.hpp"
#include "strix/ds41_prefill_tuning.hpp"
#include "strix/hip_ds41_dense_fp16.hpp"
#include "strix/deepseek41_prefill.hpp"

#include "strix/deepseek41_attention_host.hpp"
#include "strix/deepseek41_plan.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/ds41_engram.hpp"
#include "strix/ds41_expert_tp.hpp"
#include "strix/hip_ds41_expert_tp.hpp"
#include "strix/ds41_moe.hpp"
#include "strix/ds41_prefill.hpp"
#include "strix/ds41_weight_overlay.hpp"
#include "strix/hip_ds41_attention.hpp"
#include "strix/hip_ds41_formats.hpp"
#include "strix/hip_ds41_layer.hpp"
#include "strix/hip_ds41_prefill.hpp"
#include "strix/hip_ds41_rn_table.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <string>
#include <condition_variable>
#include <future>
#include <mutex>
#include <map>
#include <set>
#include <stdexcept>
#include <thread>
#include <system_error>
#include <initializer_list>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace strix::deepseek41 {
namespace {

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("deepseek41 prefill: ") + what + ": " + hipGetErrorString(e));
}

std::uint64_t fnv_bytes(const void* data, std::size_t n, std::uint64_t h = 14695981039346656037ULL) {
    const auto* b = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

template <typename T>
struct Dev {
    T* p = nullptr;
    std::size_t n = 0;
    bool owned = true;  // false: memory of another object (PrefillOptions::shared_weights), never freed here
    Dev() = default;
    // 256 bytes of slack past the end: the prefill split kernels' weight tiles read whole 4-group chunks, up to 252 bytes
    // past a row's last group (never decoded); on the last row of the last expert that is past the tensor.
    explicit Dev(std::size_t count) : n(count) {
        check(hipMalloc(reinterpret_cast<void**>(&p), std::max<std::size_t>(count, 1) * sizeof(T) + 256U), "hipMalloc");
    }
    static Dev borrow(const void* data, std::size_t count) {
        Dev d;
        d.p = static_cast<T*>(const_cast<void*>(data));
        d.n = count;
        d.owned = false;
        return d;
    }
    ~Dev() {
        if (p != nullptr && owned) (void)hipFree(p);
    }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    Dev(Dev&& o) noexcept : p(std::exchange(o.p, nullptr)), n(o.n), owned(o.owned) {}
    Dev& operator=(Dev&& o) noexcept {
        std::swap(p, o.p);
        std::swap(n, o.n);
        std::swap(owned, o.owned);
        return *this;
    }
    void zero(hipStream_t s) { check(hipMemsetAsync(p, 0, std::max<std::size_t>(n, 1) * sizeof(T), s), "memset"); }
    template <typename C>
    void upload(const C& host, hipStream_t s, std::size_t at = 0) {
        check(hipMemcpyAsync(p + at, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice, s), "upload");
    }
};

template <typename T>
std::vector<T> download(const T* p, std::size_t n, hipStream_t s) {
    std::vector<T> out(n);
    check(hipMemcpyAsync(out.data(), p, n * sizeof(T), hipMemcpyDeviceToHost, s), "download");
    check(hipStreamSynchronize(s), "download");
    return out;
}

// Diagnostics (PrefillOptions::null_ulp_layer / trace): the last mantissa bit of every finite nonzero value flipped,
// and an order-independent 64-bit hash per stream row.
// seed != 0: one ulp up or down (magnitude +-1 in the bit pattern) by a hash of (seed, i), so seeds sample the noise.
__global__ void ulp_flip_kernel(float* p, std::size_t n, std::uint32_t seed) {
    const std::size_t i = blockIdx.x * 256ULL + threadIdx.x;
    if (i >= n) return;
    const float f = p[i];
    if (f == 0.0F || !isfinite(f)) return;
    if (seed == 0U) {
        p[i] = __int_as_float(__float_as_int(f) ^ 1);
        return;
    }
    std::uint64_t x = (std::uint64_t{seed} << 40) ^ i;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^= x >> 31;
    const int b = __float_as_int(f);
    p[i] = __int_as_float((x & 1U) || (b & 0x7FFFFF) == 0 ? b + 1 : b - 1);
}
__global__ void row_hash_kernel(const float* s, std::size_t row, std::uint64_t* out) {
    __shared__ std::uint64_t red[256];
    const float* r = s + blockIdx.x * row;
    std::uint64_t h = 0;
    for (std::size_t i = threadIdx.x; i < row; i += 256) {
        std::uint64_t x = (std::uint64_t{static_cast<std::uint32_t>(__float_as_int(r[i]))} | (std::uint64_t{i} << 32)) + 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        h += x ^ (x >> 31);
    }
    red[threadIdx.x] = h;
    __syncthreads();
    for (unsigned d = 128; d; d /= 2) {
        if (threadIdx.x < d) red[threadIdx.x] += red[threadIdx.x + d];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = red[0];
}

class Mapped {
public:
    explicit Mapped(const std::filesystem::path& path) {
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) throw std::runtime_error("deepseek41 prefill: cannot open " + path.string());
        strix::LoaderCapture::record(path, fd_);
        struct stat st {};
        if (::fstat(fd_, &st) != 0) throw std::runtime_error("deepseek41 prefill: cannot stat the model");
        size_ = static_cast<std::uint64_t>(st.st_size);
        void* p = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (p == MAP_FAILED) throw std::runtime_error("deepseek41 prefill: mmap failed");
        base_ = static_cast<const std::uint8_t*>(p);
    }
    ~Mapped() {
        if (base_ != nullptr) ::munmap(const_cast<std::uint8_t*>(base_), size_);
        if (fd_ >= 0) ::close(fd_);
    }
    Mapped(const Mapped&) = delete;
    Mapped& operator=(const Mapped&) = delete;
    [[nodiscard]] const std::uint8_t* at(std::uint64_t offset) const { return base_ + offset; }

private:
    int fd_ = -1;
    std::uint64_t size_ = 0;
    const std::uint8_t* base_ = nullptr;
};

// Byte ranges of the model file into device memory through two pinned staging buffers, with O_DIRECT (the weights
// never enter the page cache). The copy of one buffer runs on the loader's own stream while the other is read; a
// buffer is read into again only after its copy's event, and load() returns after the last copy completed.
class DirectLoader {
public:
    static constexpr std::uint64_t align = 4096, chunk = 128ULL << 20U;
    explicit DirectLoader(const std::filesystem::path& path) {
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd_ < 0) fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) throw std::runtime_error("deepseek41 prefill: cannot open " + path.string());
        strix::LoaderCapture::record(path, fd_);
        check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "loader stream");
        for (int k = 0; k < 2; ++k) {
            check(hipHostMalloc(reinterpret_cast<void**>(&buffer_[k]), chunk + 2U * align, hipHostMallocDefault), "staging buffer");
            check(hipEventCreateWithFlags(&copied_[k], hipEventDisableTiming), "staging event");
        }
    }
    ~DirectLoader() {
        if (stream_ != nullptr) (void)hipStreamSynchronize(stream_);
        for (int k = 0; k < 2; ++k) {
            if (copied_[k] != nullptr) (void)hipEventDestroy(copied_[k]);
            if (buffer_[k] != nullptr) (void)hipHostFree(buffer_[k]);
        }
        if (stream_ != nullptr) (void)hipStreamDestroy(stream_);
        if (fd_ >= 0) ::close(fd_);
    }
    DirectLoader(const DirectLoader&) = delete;
    DirectLoader& operator=(const DirectLoader&) = delete;
    void load(std::uint64_t offset, std::uint64_t bytes, std::uint8_t* device) {
        std::uint64_t done = 0;
        while (done < bytes) {
            const std::uint64_t at = offset + done, base = at & ~(align - 1U), lead = at - base;
            const std::uint64_t take = std::min(bytes - done, chunk - lead);
            const std::uint64_t want = (lead + take + align - 1U) & ~(align - 1U);
            std::uint8_t* buffer = buffer_[next_];
            if (pending_[next_]) check(hipEventSynchronize(copied_[next_]), "staging reuse");  // its previous copy is done
            std::uint64_t got = 0;
            while (got < lead + take) {
                const auto n = ::pread(fd_, buffer + got, want - got, static_cast<off_t>(base + got));
                if (n <= 0) throw std::runtime_error("deepseek41 prefill: short read of the weights");
                got += static_cast<std::uint64_t>(n);
            }
            check(hipMemcpyAsync(device + done, buffer + lead, take, hipMemcpyHostToDevice, stream_), "weight upload");
            check(hipEventRecord(copied_[next_], stream_), "staging event");
            pending_[next_] = true;
            next_ ^= 1;
            done += take;
        }
        check(hipStreamSynchronize(stream_), "weight upload");
        pending_[0] = pending_[1] = false;
    }

private:
    int fd_ = -1;
    hipStream_t stream_ = nullptr;
    std::uint8_t* buffer_[2] = {nullptr, nullptr};
    hipEvent_t copied_[2] = {nullptr, nullptr};
    bool pending_[2] = {false, false};
    int next_ = 0;
};

std::uint64_t row_bytes(ds41::WeightFormat f, std::uint32_t width) {
    return std::uint64_t{width} / ds41::block_elements(f) * ds41::block_bytes(f);
}
double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// Prefill routes of a dense site resident on the device (the shared expert: one expert, slot = row).
struct DeviceSiteRoutes {
    Dev<std::uint32_t> begins, entry_rows, entry_slots, batch_expert, batch_begin, batch_count, batch_total;
    ds41::PrefillRoutesView view{};
    DeviceSiteRoutes(const ds41::PrefillRoutes& r, hipStream_t s)
        : begins(r.begins.size()), entry_rows(r.entry_rows.size()), entry_slots(r.entry_slots.size()), batch_expert(r.batch_expert.size()),
          batch_begin(r.batch_begin.size()), batch_count(r.batch_count.size()), batch_total(1) {
        begins.upload(r.begins, s);
        entry_rows.upload(r.entry_rows, s);
        entry_slots.upload(r.entry_slots, s);
        batch_expert.upload(r.batch_expert, s);
        batch_begin.upload(r.batch_begin, s);
        batch_count.upload(r.batch_count, s);
        batch_total.upload(std::vector<std::uint32_t>{static_cast<std::uint32_t>(r.batch_expert.size())}, s);
        view.begins = begins.p; view.entry_rows = entry_rows.p; view.entry_slots = entry_slots.p; view.batch_expert = batch_expert.p;
        view.batch_begin = batch_begin.p; view.batch_count = batch_count.p; view.batch_total = batch_total.p;
        view.max_batches = static_cast<std::uint32_t>(r.batch_expert.size());
        view.batch_rows = r.batch_rows;
        check(hipStreamSynchronize(s), "site routes");
    }
};

// FNV of the live window of a ring (slot = position % capacity): positions max(0, T - window) .. T - 1 in order. The
// slots of older positions are never read again (a decode step or verify window reads at most window - 1 rows back),
// and a prefill on the exact cone does not write them.
std::uint64_t ring_window_hash(const std::uint8_t* ring, std::uint32_t positions) {
    namespace attn = ds41::attn;
    std::uint64_t h = 14695981039346656037ULL;
    const std::uint32_t first = positions > attn::kWindow ? positions - attn::kWindow : 0U;
    for (std::uint32_t p = first; p < positions; ++p) {
        const std::uint8_t* e = ring + std::size_t{p % AttentionHost::ring_capacity} * attn::kWindowEntryBytes;
        for (std::uint32_t i = 0; i < attn::kWindowEntryBytes; ++i) h = (h ^ e[i]) * 1099511628211ULL;
    }
    return h;
}

// Row-split exchanges with several per-row fields in one message: field k of row r (words[k] 32-bit words, rows
// consecutive in its buffer) at word off_k = words[0] + ... + words[k - 1] of packed row r (stride words).
struct PackFields {
    const std::uint32_t* src[4] = {};
    std::uint32_t* dst[4] = {};
    std::uint32_t words[4] = {};
    std::uint32_t count = 0, stride = 0;
};
__global__ void pack_rows_kernel(PackFields f, std::uint32_t* __restrict__ packed) {
    const std::size_t r = blockIdx.x;
    std::uint32_t off = 0;
    for (std::uint32_t k = 0; k < f.count; ++k) {
        for (std::uint32_t i = threadIdx.x; i < f.words[k]; i += blockDim.x) packed[r * f.stride + off + i] = f.src[k][r * f.words[k] + i];
        off += f.words[k];
    }
}
__global__ void unpack_rows_kernel(PackFields f, const std::uint32_t* __restrict__ packed) {
    const std::size_t r = blockIdx.x;
    std::uint32_t off = 0;
    for (std::uint32_t k = 0; k < f.count; ++k) {
        for (std::uint32_t i = threadIdx.x; i < f.words[k]; i += blockDim.x) f.dst[k][r * f.words[k] + i] = packed[r * f.stride + off + i];
        off += f.words[k];
    }
}

__global__ void add_halves_kernel(const float* __restrict__ h0, const float* __restrict__ h1, float* __restrict__ out, std::uint32_t n) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = h0[i] + h1[i];
}

// The live window of the prefill ring (capacity src_capacity) into a layer's decode ring (capacity dst_capacity):
// one block per position first .. first + count - 1, the 528-byte entry as 132 words.
__global__ void ring_copy_kernel(const std::uint32_t* __restrict__ src, std::uint32_t src_capacity, std::uint32_t* __restrict__ dst,
                                 std::uint32_t dst_capacity, std::uint32_t first) {
    constexpr unsigned words = ds41::attn::kWindowEntryBytes / 4U;
    const std::uint32_t p = first + blockIdx.x;
    if (threadIdx.x < words)
        dst[std::size_t{p % dst_capacity} * words + threadIdx.x] = src[std::size_t{p % src_capacity} * words + threadIdx.x];
}

// STRIX_DS41_PREFILL_FINAL_KV (P17.5): the final attention entries of rows into (Pack) or out of a packed message:
// ring entries (528 B, slot position % capacity) of ring_rows rows at ring_pos, then compressed entries (KV 288 B +
// index key 68 B, slot position / ratio) of comp_rows entries; entry j is completed by the row at comp_pos[ratio * j
// + ratio - 1] (ratio 2: windows start at even positions, so a rank's rows from a window start pair up in order).
struct FinalEntries {
    std::uint32_t* ring = nullptr;
    std::uint32_t capacity = 0;
    std::uint32_t* kv_cache = nullptr;
    std::uint32_t* idx_cache = nullptr;
    std::uint32_t ratio = 1;
    const std::uint64_t* ring_pos = nullptr;
    std::uint32_t ring_rows = 0;
    const std::uint64_t* comp_pos = nullptr;
    std::uint32_t comp_rows = 0;
    std::uint32_t* packed = nullptr;
};
constexpr std::uint32_t final_ring_words = ds41::attn::kWindowEntryBytes / 4U, final_kv_words = 288U / 4U, final_idx_words = 68U / 4U;
template <bool Pack>
__global__ void final_entries_kernel(FinalEntries f) {
    const std::uint32_t b = blockIdx.x;
    if (b < f.ring_rows) {
        std::uint32_t* e = f.ring + std::size_t{f.ring_pos[b] % f.capacity} * final_ring_words;
        std::uint32_t* q = f.packed + std::size_t{b} * final_ring_words;
        for (std::uint32_t i = threadIdx.x; i < final_ring_words; i += blockDim.x) {
            if constexpr (Pack) q[i] = e[i];
            else e[i] = q[i];
        }
        return;
    }
    const std::uint32_t j = b - f.ring_rows;
    const std::uint64_t slot = f.comp_pos[std::size_t{f.ratio} * j + f.ratio - 1U] / f.ratio;
    std::uint32_t* kv = f.kv_cache + slot * final_kv_words;
    std::uint32_t* ix = f.idx_cache + slot * final_idx_words;
    std::uint32_t* q = f.packed + std::size_t{f.ring_rows} * final_ring_words + std::size_t{j} * (final_kv_words + final_idx_words);
    for (std::uint32_t i = threadIdx.x; i < final_kv_words + final_idx_words; i += blockDim.x) {
        std::uint32_t* e = i < final_kv_words ? kv + i : ix + (i - final_kv_words);
        if constexpr (Pack) q[i] = *e;
        else *e = q[i];
    }
}
// STRIX_DS41_PREFILL_O_A8_FUSED (read per call; default on, =0 off): the attention combine writes o's A8 operands
bool prefill_o_a8_fused() {
    const char* v = std::getenv("STRIX_DS41_PREFILL_O_A8_FUSED");
    return !(v && v[0] == '0');
}
// STRIX_DS41_PREFILL_HC_FUSE (read per call; default on, =0 off): the attention's hc expansion also computes the ffn
// site's mixes (one enqueue_hc_site, no re-read of the stream), and each site's collapse is read by its RMSNorm (no x
// round trip); same kernels' arithmetic, same bits
bool prefill_hc_fuse() {
    const char* v = std::getenv("STRIX_DS41_PREFILL_HC_FUSE");
    return !(v && v[0] == '0');
}
// STRIX_DS41_PREFILL_HC_COLLAPSE (read per call; default on with HC_FUSE, =0 off): that fused expansion also writes the
// next site's collapsed rows (the collapse uses the expansion's own pre), so the site's RMSNorm reads x instead of the
// 4-copy stream; same bits
bool prefill_hc_collapse() {
    const char* v = std::getenv("STRIX_DS41_PREFILL_HC_COLLAPSE");
    return !(v && v[0] == '0');
}
// STRIX_DS41_PREFILL_FINAL_KV=all (default since P17.5: 8K prefill +1.8 %, bitwise) | none | <comma-separated layers>
bool prefill_final_kv(std::uint32_t layer) {
    static const std::vector<bool> on = [] {
        std::vector<bool> r(1024, false);
        const char* v = std::getenv("STRIX_DS41_PREFILL_FINAL_KV");
        if (!v || !*v) return std::vector<bool>(1024, true);
        const std::string s(v);
        if (s == "all") return std::vector<bool>(1024, true);
        if (s == "none" || s == "off") return r;
        std::size_t i = 0;
        while (i < s.size()) {
            const std::size_t j = s.find(',', i);
            const auto l = std::stoul(s.substr(i, j == std::string::npos ? std::string::npos : j - i));
            if (l >= r.size()) throw std::invalid_argument("STRIX_DS41_PREFILL_FINAL_KV: layer out of range");
            r[l] = true;
            if (j == std::string::npos) break;
            i = j + 1U;
        }
        return r;
    }();
    return layer < on.size() && on[layer];
}

}  // namespace

struct DevicePrefill::Impl {
    PrefillOptions options;
    TargetModel model;
    Plan plan;
    std::unique_ptr<Mapped> file;
    std::map<std::string, const ManifestTensor*> tensors;
    std::map<std::string, Dev<std::uint8_t>> weights;
    std::unique_ptr<ds41::EngramRowReader> engram_reader;
    // the routed-expert MoE of every layer (one scratch; STRIX_DS41_PREFILL_MOE_PERSIST=0: one object per layer)
    std::unique_ptr<ds41::PrefillMoe> moe;
    bool moe_persist = true;
    // resumable runs (PrefillTask): the task whose run is in progress (its worker thread runs run_from and stops at the
    // unit boundaries), and a sequence left invalid by a cancelled task (extend refused until the next run)
    PrefillTask::State* task = nullptr;
    bool broken = false, cancel_broken = false;
    // row_split all-gathers on the communication stream, overlapped with independent work (default;
    // STRIX_DS41_PREFILL_ASYNC_GATHER=0: synchronous exchanges on the main stream)
    bool async_gather = true;
    // row_split: per-row fields of one all-gather packed into one message (STRIX_DS41_PREFILL_PACK=0: one exchange per field)
    bool pack = true;
    // row_split: the peer rows' MoE half sums are written straight into the communicator's pinned send buffer and the
    // peer's are read straight from its receive buffer (no device<->host copy, which long MoE kernels on the main
    // stream would hold back until the own-row pass ends); one message per layer, sent while the own rows run.
    // Same kernels, same sum order: bitwise identical. Default on; STRIX_DS41_PREFILL_STAGED_HALF=0: the copy path.
    bool staged_half = false;
    // row_split with staged_half: the MoE input all-gather the same way (packed into the pinned send buffer right after
    // the router, crossing while the shared expert runs). Default on; STRIX_DS41_PREFILL_STAGED_GATHER=0: the copy path.
    bool staged_gather = false;
    // row_split, final entries: the first chunk's q_b runs before the host waits for the peer's entries (it reads only
    // this rank's rows), so the wait no longer leaves the GPU idle. Same kernel, same inputs: bitwise identical. Only
    // when the next unit boundary does not park (m.q is shared). Default on; STRIX_DS41_PREFILL_EARLY_QB=0: after.
    bool early_q_b = true;
    Dev<std::uint32_t> pack_send, pack_recv;
    // sites run on the dense WMMA kernel instead of the batch kernel (STRIX_DS41_PREFILL_DENSE_SITES, comma list of
    // q_a, kv, q_b, o_a; default q_a,kv, faster on the dense kernel at 512 and 2048 rows; "none" = all on the batch
    // kernel; o_a: one dense call per o-group): both give the decode's bits
    bool dense_q_a = true, dense_kv = true, dense_q_b = false, dense_o_a = false;
    // set_dense_select: the selector forced on (auto form on every exact dense call, q_b / o_a dense; default on,
    // STRIX_DS41_DENSE_SELECT=0 off); env_* = loaded
    bool dense_select = false, env_dense_q_b = false, env_dense_o_a = false;
    // q_b / o_a take the dense kernel only from this many rows per call (STRIX_DS41_PREFILL_DENSE_MIN_ROWS, default 256:
    // the selector's per-site forms start there; smaller chunks, cone tails and extends stay on the batch kernel)
    std::uint32_t dense_min_rows = 256;
#ifdef STRIX_DS41_PREFILL_FP16_LT
    ds41::DenseFp16Cache dense_fp16;
#endif
    // projection bit (PrefillOptions::fp16_projections) of a layer's call of `rows` rows on the FP16 backend
    bool fp16_at(std::uint32_t bit, std::uint32_t layer, std::uint32_t rows) const {
        // FP16 output form: calls under 128 rows stay exact (hipBLASLt loses there: 64 x 8192 x 5120 0.46-0.51 vs 0.40 ms)
        return options.fp16_dense && (options.fp16_projections & bit) != 0U && layer >= options.approx_layer_first &&
               layer < options.approx_layer_end && rows >= std::max(options.fp16_min_rows, options.fp16_fast ? 128U : 0U);
    }
    // FP16 output form: the norms' FP16 activations (x for q_a / kv, qr for q_b), sized as x / qr on first use
    Dev<std::uint16_t> x16, qr16;
    bool fast_at(std::uint32_t bits, std::uint32_t layer) const {
        return options.fp16_dense && options.fp16_fast && (options.fp16_projections & bits) != 0U && layer >= options.approx_layer_first &&
               layer < options.approx_layer_end;
    }
    bool fast_used = false;
    void fast(const ds41::Q8A8DensePrefillView* views, unsigned count, std::uint32_t rows, const std::uint16_t* a16, std::uint32_t a16_stride,
              const float* input_x, void* st, unsigned groups = 1) {
        (void)views, (void)count, (void)rows, (void)a16, (void)a16_stride, (void)input_x, (void)st, (void)groups;
#ifdef STRIX_DS41_PREFILL_FP16_LT
        dense_fp16.enqueue_fast(views, count, rows, a16, a16_stride, input_x, st, groups, options.fp16_wsplit);
        fast_used = true;
#else
        throw std::runtime_error("FP16 dense prefill requires STRIX_DS41_PREFILL_FP16_LT");
#endif
    }
    bool bf16_at(std::uint32_t layer) const {
        return options.bf16_attention && layer >= options.approx_layer_first && layer < options.approx_layer_end;
    }
    bool fast_attn_at(std::uint32_t layer) const {
        return options.fast_attention && layer >= options.approx_layer_first && layer < options.approx_layer_end;
    }
    void dense(const ds41::Q8A8DensePrefillView& v, const float* input_x, void* st, bool fp16) {
        (void)input_x;
        if (fp16) {
#ifdef STRIX_DS41_PREFILL_FP16_LT
            dense_fp16.enqueue(v,input_x,st,options.fp16_keep_a8);
#else
            throw std::runtime_error("FP16 dense prefill requires STRIX_DS41_PREFILL_FP16_LT");
#endif
        } else if (dense_select && v.tall == 0U) {
            auto w = v;
            w.tall = 1U;  // auto
            ds41::enqueue_q8_0_a8_dense_prefill(w,st);
        } else ds41::enqueue_q8_0_a8_dense_prefill(v,st);
    }

    // row_split MoE: PrefillMoe::enqueue_half_joint (one route build and expert pass over the peer's and this rank's
    // rows; opt-in, STRIX_DS41_PREFILL_JOINT=1. Default: two passes, peer rows first; P11: joint -4 % at 2K, 0 at 16K)
    bool moe_joint = false;
    // stream_experts: the rank's slice (offset, bytes) of every layer's routed-expert tensors, two layer slots and the
    // thread filling the next one
    std::filesystem::path gguf;
    std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> expert_extent;
    struct ExpertSlot {
        std::map<std::string, Dev<std::uint8_t>> kinds;  // "gate" / "up" / "down"
        std::atomic<std::int32_t> layer{-1};  // written by the prefetch thread
    };
    ExpertSlot slots[2];
    std::thread prefetch;
    std::exception_ptr prefetch_error;
    double expert_read_seconds = 0.0, expert_wait_seconds = 0.0;
    hipStream_t stream = nullptr;
    double load_seconds = 0.0;
    std::uint32_t expert_first = 0, expert_end = 0;
    bool expert_tp = false;  // PrefillOptions::expert_tp: every expert's K-half parts (ds41-expert-tp2-i1152-v1)
    ds41::VocabSlice head_slice{};
    std::array<ds41::attn::LayerAttention, 40> roles{};
    std::uint32_t rows_done = 0;  // rows of the sequence so far (every run)
    std::unique_ptr<ds41::EngramHasher> hasher;   // the sequence's Engram history (extend continues it)
    std::uint32_t pos_base = 0;                   // position of row 0 in pos64 / rope_window / rope_comp / rope_group
    // Per-run setup kept across runs (same bits; STRIX_DS41_PREFILL_OVERLAP=0: the previous per-run form): the dense-site
    // routes by (rows, groups), the token ids on the device, a pinned buffer the Engram rows are read into (on a thread,
    // while the layers before the first Engram layer run; no stream synchronisation per Engram layer).
    bool overlap = true;
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::unique_ptr<DeviceSiteRoutes>> route_cache;
    Dev<std::uint32_t> dtokens;
    std::uint8_t* engram_pinned = nullptr;
    std::size_t engram_pinned_bytes = 0;
    // Engram lookahead (set_engram_lookahead, STRIX_DS41_ENGRAM_LOOKAHEAD default on, =0 off): the next run's tokens,
    // and the thread that reads this rank's rows of them into the reader's payload cache after this run's own read.
    std::vector<std::uint32_t> engram_next;
    // decoder deferral (set_decoder_deferral): the next run's continuation rows (0: none), and whether the last run left
    // the decoder state for its continuation to complete (the state accessors refuse until then)
    std::uint32_t defer_next = 0;
    bool state_incomplete = false;
    std::vector<ds41::EngramRowRequest> engram_ahead_requests;
    std::vector<std::uint8_t> engram_ahead_out;
    std::thread engram_ahead;
    void join_engram_ahead() {
        if (engram_ahead.joinable()) engram_ahead.join();
    }
    // world 2 with row_split: runs of fewer rows than this take the tensor-parallel form instead (same bits)
    std::uint32_t split_min_rows = 0;
    // row_split MoE: runs of at most this many rows take the joint expert pass (one read of the expert half per layer
    // instead of two; same bits). 0: only STRIX_DS41_PREFILL_JOINT=1.
    std::uint32_t joint_max_rows = 0;

    // attention state per layer: ring, the KV sources' caches, the ratio-2 carry (ping-pong), the index sources'
    // selections for every row of the prompt
    struct LayerState {
        Dev<std::uint8_t> ring, kv_cache, idx_cache;
        Dev<float> carry[2];
        Dev<std::uint32_t> carry_valid[2];
        int carry_cur = 0;
        Dev<std::uint32_t> idx, idx_count;  // [rows][topk], [rows]
    };
    std::vector<LayerState> attn;
    Dev<std::uint8_t> candidates;  // per window of the candidate source: window_rows x ceil(max_count / block) bytes
    std::size_t cand_window_bytes = 0;
    // The window ring of the layer being run, one slot per position (capacity max_positions + window_rows): every
    // window K/V of the layer is written before its attention windows, which then depend only on the layer's caches and
    // run on several lanes. Its live window goes to the layer's decode ring at the end.
    Dev<std::uint8_t> prefill_ring;
    std::uint32_t prefill_capacity = 0;
    // in-run snapshots (set_snapshot_points): the points of the next run, those of the last run and their rings
    // (point-major, one ring_capacity ring per layer)
    std::vector<std::uint32_t> snap_next, snap_points;
    std::vector<Dev<std::uint8_t>> snap_rings;
    // a stream and the scratch of one attention window per lane
    struct Lane {
        hipStream_t stream = nullptr;
        hipEvent_t done = nullptr;
        Dev<float> index_scores, attn_partial;
        Dev<std::uint32_t> attn_tickets;  // W x kAttnGroupCount, zero between calls (the last chunk block combines)
        Dev<std::uint32_t> select_keys;
        Dev<std::uint8_t> index_qprep;
    };
    std::vector<Lane> lanes;
    hipEvent_t chunk_ready = nullptr;
    // world 2: the stream the MoE half sums are staged and received on, and the events ordering it with the main one
    hipStream_t comm_stream = nullptr;
    hipEvent_t half_ready[2] = {nullptr, nullptr}, half_received = nullptr;
    // row_split: producers of the gathered rows (K/V, MoE input) and their arrival, on the communication stream
    hipEvent_t gather_ready = nullptr, gathered = nullptr;
    Dev<std::uint64_t> pos64;
    Dev<ds41::attn::RopeRow> rope_window, rope_comp, rope_group;
    // positions and RoPE rows of this rank's rows (row_split; otherwise the same as the global tables)
    Dev<std::uint64_t> lpos64;
    Dev<std::uint64_t> ppos64;  // row_split: the positions of the peer's local rows (STRIX_DS41_PREFILL_FINAL_KV)
    Dev<ds41::attn::RopeRow> lrope_window, lrope_comp, lrope_group;
    // row_split: the peer's rows received (K/V for the ring and the compressor, MoE input), the peer's half sums of
    // this rank's rows, the last row's head input
    Dev<float> peer_kv, peer_ckv, peer_cscore, peer_xdeltas, peer_route_w, half_recv, h_peer;
    Dev<std::int8_t> peer_xcodes;
    Dev<std::uint32_t> peer_ids;
    std::uint32_t local_rows_done = 0;
    bool last_row_local = true;
    // draft_rows: this rank's seed rows per captured layer (compact, local order), the peer's, and all in position order
    Dev<float> seed_local[3], seed_peer[3], seed[3];
    std::uint32_t seed_rows = 0, seed_first = 0;
    // row buffers (max_positions rows unless noted)
    Dev<float> streams[2];
    int cur = 0;
    bool identity_pre = true;
    Dev<ds41::HcMix> mix_attn, mix_ffn;
    Dev<float> mix_partial;
    Dev<std::uint32_t> hc_counters;  // enqueue_hc_site's counters, back at zero after every launch
    Dev<float> x, qr, kv, ckv, cscore, iq, iw, y, shared, scores, route_w;
    Dev<std::int8_t> xcodes, qcodes;
    Dev<float> xdeltas, qdeltas;
    Dev<std::int32_t> xsums, qsums;
    Dev<std::uint32_t> ids;
    Dev<std::uint8_t> image_rows;
    // per chunk of chunk_rows rows
    Dev<float> q, o, o_low;
    Dev<std::int8_t> ocodes, olcodes;
    Dev<float> odeltas, oldeltas;
    Dev<std::int32_t> osums, olsums;
    Dev<std::uint8_t> erows;
    Dev<float> emb, ekv;
    // shared expert activations (every row)
    Dev<std::int8_t> sh_codes;
    Dev<float> sh_deltas;
    Dev<std::int32_t> sh_sums;
    // world 2: half sums of every row; TP attention: this rank's and the peer's wo_b K-half of a chunk
    Dev<float> half_own, half_other, y_own, y_other;
    // head (last row)
    Dev<float> h, logits, quality_h, quality_logits;
    Dev<std::uint64_t> trace_hash;
    Dev<std::int8_t> quality_codes;
    Dev<float> quality_deltas;
    Dev<std::int32_t> quality_sums;
    Dev<std::int8_t> hcodes;
    Dev<float> hdeltas;
    Dev<std::int32_t> hsums;
    Dev<ds41::ArgmaxCandidate> partial, cand0, cand1, best, cand_own, cand_other;

    // pos64 and the RoPE tables of rows [0, rows) at positions [base, base + rows)
    void upload_positions(std::uint32_t base, hipStream_t s) {
        namespace attn = ds41::attn;
        const std::size_t rows = pos64.n;
        std::vector<std::uint64_t> pos(rows);
        std::vector<attn::RopeRow> rw(rows), rc(rows), rg(rows);
        for (std::size_t r = 0; r < rows; ++r) {
            const std::uint64_t p = base + r;
            pos[r] = p;
            rw[r] = attn::rope_row(attn::RopeKind::window, p);
            rc[r] = attn::rope_row(attn::RopeKind::compressed, p);
            rg[r] = attn::rope_row(attn::RopeKind::compressed, p / 2U * 2U);
        }
        pos64.upload(pos, s);
        rope_window.upload(rw, s);
        rope_comp.upload(rc, s);
        rope_group.upload(rg, s);
        if (hipStreamSynchronize(s) != hipSuccess) throw std::runtime_error("deepseek41 prefill: rope tables");
        pos_base = base;
    }
    [[nodiscard]] bool distributed() const noexcept { return options.world == 2U; }
    std::unique_ptr<ds41::WeightOverlay> overlay;  // weights_overlay (Q8_0 form)
    [[nodiscard]] const std::uint8_t* dev(const std::string& name) const { return weights.at(name).p; }
    [[nodiscard]] const float* devf(const std::string& name) const { return reinterpret_cast<const float*>(weights.at(name).p); }
    [[nodiscard]] const std::uint16_t* devh(const std::string& name) const {
        return reinterpret_cast<const std::uint16_t*>(weights.at(name).p);
    }
};

DevicePrefill::DevicePrefill(const std::filesystem::path& gguf, const PrefillOptions& requested) : impl_(std::make_unique<Impl>()) {
    const auto t_load = std::chrono::steady_clock::now();
    auto& m = *impl_;
    // The routed experts follow PrefillOptions::expert_tp (or STRIX_DS41_PREFILL_EXPERT_TP=1), not STRIX_DS41_EXPERT_TP:
    // off, the expert-parallel identity (with an expert-TP decode, the declared mixed identity ds41-prefill-ep+decode-tp);
    // on, ds41-expert-tp2-i1152-v1, the decode's bits.
    PrefillOptions options = requested;
    // (the variable applies to two-rank prefills only: a world-1 prefill holds every expert whole, the expert-parallel form)
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_EXPERT_TP"); v != nullptr && *v != 0 && options.world == 2U) options.expert_tp = v[0] == '1';
    // STRIX_DS41_PREFILL_WINDOW=W: rows per attention call and row-split unit (same bits for any W)
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_WINDOW"); v != nullptr && *v != 0)
        options.window_rows = static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10));
    // An explicit WINDOW takes precedence. Selection happens before any scratch allocation
    // or row-owner map is built, and stays fixed for this DevicePrefill's lifetime.
    const bool large_slabs = ds41::prefill_tuning::enabled("STRIX_DS41_PREFILL_SLABS");
    if (large_slabs && options.window_rows == PrefillOptions{}.window_rows &&
        (std::getenv("STRIX_DS41_PREFILL_WINDOW") == nullptr || !*std::getenv("STRIX_DS41_PREFILL_WINDOW"))) {
        const auto rows = options.max_rows != 0U ? options.max_rows : options.max_positions;
        options.window_rows = ds41::prefill_tuning::attention_rows(rows, options.chunk_rows, ds41::attn::kHeads,
                                                                 options.max_positions, options.attention_lanes);
    }
    if(const char* e=std::getenv("STRIX_DS41_PREFILL_FP16_DENSE");e && std::string(e)=="1")options.fp16_dense=true;
    if(const char* e=std::getenv("STRIX_DS41_PREFILL_BF16_ATTN");e && std::string(e)=="1")options.bf16_attention=true;
    if (const char* e = std::getenv("STRIX_DS41_FAST_ATTN"); e != nullptr && std::string(e) == "1") options.fast_attention = true;
    // Per-shape selector of the FP16 backend: STRIX_DS41_PREFILL_FP16_PROJ (comma list of q_a, kv, q_b, o_a, o_b),
    // STRIX_DS41_PREFILL_FP16_MIN_ROWS (calls of at least that many rows), STRIX_DS41_PREFILL_APPROX_LAYERS=first-end,
    // STRIX_DS41_PREFILL_FP16_A8=1 (operands from the A8 codes)
    if (const char* e = std::getenv("STRIX_DS41_PREFILL_FP16_PROJ"); e != nullptr && *e != 0) {
        std::uint32_t mask = 0;
        std::stringstream ss(e);
        for (std::string t; std::getline(ss, t, ',');) {
            if (t == "q_a") mask |= 1U;
            else if (t == "kv") mask |= 2U;
            else if (t == "q_b") mask |= 4U;
            else if (t == "o_a") mask |= 8U;
            else if (t == "o_b") mask |= 16U;
            else throw std::invalid_argument("STRIX_DS41_PREFILL_FP16_PROJ: " + t);
        }
        options.fp16_projections = mask;
    }
    if (const char* e = std::getenv("STRIX_DS41_PREFILL_FP16_MIN_ROWS"); e != nullptr && *e != 0)
        options.fp16_min_rows = static_cast<std::uint32_t>(std::strtoul(e, nullptr, 10));
    if (const char* e = std::getenv("STRIX_DS41_PREFILL_APPROX_LAYERS"); e != nullptr && *e != 0) {
        char* end = nullptr;
        options.approx_layer_first = static_cast<std::uint32_t>(std::strtoul(e, &end, 10));
        if (end == nullptr || *end != '-') throw std::invalid_argument("STRIX_DS41_PREFILL_APPROX_LAYERS: first-end");
        options.approx_layer_end = static_cast<std::uint32_t>(std::strtoul(end + 1, nullptr, 10));
    }
    if (const char* e = std::getenv("STRIX_DS41_PREFILL_FP16_A8"); e != nullptr && std::string(e) == "1") options.fp16_keep_a8 = true;
    if (const char* e = std::getenv("STRIX_DS41_PREFILL_FP16_FAST"); e != nullptr && std::string(e) == "1") options.fp16_fast = true;
    if (const char* e = std::getenv("STRIX_DS41_PREFILL_FP16_WSPLIT"); e != nullptr && std::string(e) == "1") options.fp16_wsplit = true;
    m.options = options;
    if (options.world != 1U && options.world != 2U) throw std::invalid_argument("deepseek41 prefill: world must be 1 or 2");
    if (options.world == 2U && (options.communicator == nullptr || options.rank > 1U))
        throw std::invalid_argument("deepseek41 prefill: world 2 needs a communicator and rank 0 or 1");
    if (options.window_rows == 0U || options.window_rows > (large_slabs ? ds41::attn::kMaxCallRows : ds41::attn::kLegacyMaxCallRows) || options.chunk_rows == 0U ||
        options.chunk_rows % options.window_rows != 0U)
        throw std::invalid_argument("deepseek41 prefill: window_rows 1..kMaxCallRows dividing chunk_rows");
    if (options.attention_lanes == 0U || options.attention_lanes > 16U) throw std::invalid_argument("deepseek41 prefill: attention_lanes 1..16");
    if (options.moe_chunk_rows == 0U || options.moe_chunk_rows % options.chunk_rows != 0U)
        throw std::invalid_argument("deepseek41 prefill: moe_chunk_rows a multiple of chunk_rows");
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_MOE_PERSIST"); v != nullptr && v[0] == '0') m.moe_persist = false;
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_ASYNC_GATHER"); v != nullptr && v[0] == '0') m.async_gather = false;
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_PACK"); v != nullptr && v[0] == '0') m.pack = false;
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_JOINT"); v != nullptr && v[0] == '1') m.moe_joint = true;
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_OVERLAP"); v != nullptr && v[0] == '0') m.overlap = false;
    m.split_min_rows = options.split_min_rows;
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_SPLIT_MIN_ROWS"); v != nullptr && *v != 0)
        m.split_min_rows = static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10));
    m.joint_max_rows = options.joint_max_rows;
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_JOINT_MAX_ROWS"); v != nullptr && *v != 0)
        m.joint_max_rows = static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10));
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_DENSE_SITES"); v != nullptr) {
        const std::string list = std::string(",") + v + ",";
        m.dense_q_a = list.find(",q_a,") != std::string::npos;
        m.dense_kv = list.find(",kv,") != std::string::npos;
        m.dense_q_b = list.find(",q_b,") != std::string::npos;
        m.dense_o_a = list.find(",o_a,") != std::string::npos;
    }
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_DENSE_MIN_ROWS"); v != nullptr && *v != 0) m.dense_min_rows = static_cast<std::uint32_t>(std::stoul(v));
    m.env_dense_q_b = m.dense_q_b;
    m.env_dense_o_a = m.dense_o_a;
    // the exact dense selector is the default (bitwise EQUAL, 2K -5.5 % / 8K -5.1 % TTFT);
    // STRIX_DS41_DENSE_SELECT=0 = the loaded configuration above (STRIX_DS41_DENSE_TALL / PREFILL_DENSE_SITES)
    if (const char* v = std::getenv("STRIX_DS41_DENSE_SELECT"); !(v != nullptr && v[0] == '0' && v[1] == 0)) {
        m.dense_select = true;
        m.dense_q_b = true;
        m.dense_o_a = true;
    }
    m.model = validate_target(read_file(gguf));
    m.plan = build_decode_plan(m.model, 0, options.layers);
    m.file = std::make_unique<Mapped>(gguf);
    for (const auto& t : m.model.manifest.tensors) m.tensors.emplace(t.name, &t);
    {
        auto path = options.weights_overlay;
        if (const char* e = std::getenv("STRIX_DS41_OVERLAY"); path.empty() && e != nullptr && *e != 0) path = e;
        if (!path.empty()) m.overlay = std::make_unique<ds41::WeightOverlay>(path, m.model.manifest);
    }
    check(hipStreamCreateWithFlags(&m.stream, hipStreamNonBlocking), "stream");
    const auto& c = m.model.config;
    {
        const std::uint32_t per_rank = c.n_routed_experts / options.world;
        m.expert_first = options.rank * per_rank;
        m.expert_end = m.expert_first + per_rank;
        m.head_slice = m.distributed() ? ds41::vocab_slice(c.vocab_size, 2U, options.rank) : ds41::VocabSlice{0U, c.vocab_size};
        if (options.expert_tp) {
            if (options.world != 2U || options.stream_experts || c.moe_intermediate_size != ds41::expert_tp::inter)
                throw std::invalid_argument("deepseek41 prefill: expert TP needs two ranks, resident experts and I = 2304");
            if (ds41::expert_tp::packed_from_env())
                throw std::invalid_argument("deepseek41 prefill: expert TP reads the dup layout (STRIX_DS41_EXPERT_TP_LAYOUT=dup)");
            m.expert_tp = true;
            m.expert_first = 0;
            m.expert_end = c.n_routed_experts;
        }
    }
    std::set<std::string> names;
    for (const auto& s : m.plan.steps)
        for (const auto& w : s.weights)
            if (w.find("engram_embd") == std::string::npos) names.insert(w);
    m.gguf = gguf;
    // STRIX_DS41_PREFILL_MEM_GUARD_GIB=G (default off): before loading anything, the bytes this prefill would load (the
    // weights not found in shared_weights, the streamed-expert slots) must fit in MemAvailable minus G GiB, else throw
    // (a harness that failed to share the weights fails here instead of exhausting the node's memory).
    if (const char* g = std::getenv("STRIX_DS41_PREFILL_MEM_GUARD_GIB"); g != nullptr && *g != 0) {
        std::uint64_t need = 0, shared = 0;
        for (const auto& n : names) {
            std::uint64_t bytes = m.tensors.at(n)->byte_size;
            if (n.ends_with("_exps.weight")) bytes = bytes / c.n_routed_experts * (m.expert_end - m.expert_first) * (options.stream_experts ? 2U : 1U);
            else if (n == "output.weight") bytes = bytes / c.vocab_size * m.head_slice.count;
            const auto it = options.shared_weights.find(n);
            if (!options.stream_experts && it != options.shared_weights.end() && it->second.second == bytes) shared += bytes;
            else need += bytes;
        }
        std::uint64_t avail_kb = 0;
        {
            std::FILE* f = std::fopen("/proc/meminfo", "r");
            char line[256];
            while (f != nullptr && std::fgets(line, sizeof(line), f) != nullptr)
                if (std::strncmp(line, "MemAvailable:", 13) == 0) avail_kb = std::strtoull(line + 13, nullptr, 10);
            if (f != nullptr) std::fclose(f);
        }
        const double gib = 1073741824.0, margin = std::strtod(g, nullptr);
        std::fprintf(stderr, "deepseek41 prefill memory guard: loads %.2f GiB (reads %.2f GiB in place), MemAvailable %.2f GiB, margin %.1f GiB\n",
                     static_cast<double>(need) / gib, static_cast<double>(shared) / gib, static_cast<double>(avail_kb) * 1024.0 / gib, margin);
        if (static_cast<double>(need) / gib + margin > static_cast<double>(avail_kb) * 1024.0 / gib)
            throw std::runtime_error("deepseek41 prefill: memory guard: the weights to load do not fit in MemAvailable minus the margin");
    }
    {
        DirectLoader loader(gguf);
        std::unique_ptr<DirectLoader> overlay_loader;
        if (m.overlay) overlay_loader = std::make_unique<DirectLoader>(m.overlay->path());
        for (const auto& n : names) {
            const auto* t = m.tensors.at(n);
            const auto* over = m.overlay ? m.overlay->find(n) : nullptr;
            if (over != nullptr && options.stream_experts && n.ends_with("_exps.weight"))
                throw std::invalid_argument("deepseek41 prefill: streamed experts read the model file, not an overlay's experts");
            std::uint64_t offset = over != nullptr ? over->absolute_offset : t->absolute_offset, bytes = t->byte_size;
            if (m.expert_tp && n.ends_with("_exps.weight")) {  // this rank's K-half parts of every expert
                const bool down = n.ends_with("ffn_down_exps.weight");
                const std::uint64_t part = ds41::expert_tp::part_bytes(bytes, c.n_routed_experts, down, c.hidden_size);
                if (const auto it = options.shared_weights.find(n); it != options.shared_weights.end() && it->second.second == part) {
                    m.weights.emplace(n, Dev<std::uint8_t>::borrow(it->second.first, part));
                    continue;
                }
                Dev<std::uint8_t> d(part);
                auto& from = over != nullptr ? *overlay_loader : loader;
                ds41::expert_tp::load_part([&from](std::uint64_t at, std::uint64_t count, std::uint8_t* dst) { from.load(at, count, dst); },
                                           offset, bytes, c.n_routed_experts, down, c.hidden_size, options.rank, d.p, m.stream);
                m.weights.emplace(n, std::move(d));
                continue;
            }
            if (n.ends_with("_exps.weight")) {
                const std::uint64_t per = bytes / c.n_routed_experts;
                offset += per * m.expert_first;
                bytes = per * (m.expert_end - m.expert_first);
                if (options.stream_experts) {
                    m.expert_extent.emplace(n, std::make_pair(offset, bytes));
                    const std::string kind = n.find("gate") != std::string::npos ? "gate" : n.find("up") != std::string::npos ? "up" : "down";
                    for (auto& slot : m.slots) {
                        auto& d = slot.kinds[kind];
                        if (d.n < bytes) d = Dev<std::uint8_t>(bytes);
                    }
                    continue;
                }
            } else if (n == "output.weight") {
                const std::uint64_t per = bytes / c.vocab_size;
                offset += per * m.head_slice.first;
                bytes = per * m.head_slice.count;
            }
            if (const auto it = options.shared_weights.find(n); it != options.shared_weights.end() && it->second.second == bytes) {
                m.weights.emplace(n, Dev<std::uint8_t>::borrow(it->second.first, bytes));  // the same bytes, read in place
                continue;
            }
            Dev<std::uint8_t> d(bytes);
            (over != nullptr ? *overlay_loader : loader).load(offset, bytes, d.p);
            m.weights.emplace(n, std::move(d));
        }
    }
    {
        std::vector<ds41::EngramTable> tables;
        for (const auto l : m.model.engram.layer_ids) {
            const auto* t = m.tensors.at("blk." + std::to_string(l) + ".engram_embd.weight");
            tables.push_back({t->absolute_offset, t->byte_size / 264U});
        }
        m.engram_reader = std::make_unique<ds41::EngramRowReader>(gguf, std::move(tables), 96U, options.engram_direct_io);
    }

    namespace attn = ds41::attn;
    // row buffers hold max_positions rows plus four windows (row_split pads an exchange to the larger rank's rows)
    const std::size_t W = options.window_rows, C = options.chunk_rows;
    const std::size_t P = options.max_positions + 4U * W;  // positions the caches hold
    if (options.max_rows > options.max_positions) throw std::invalid_argument("deepseek41 prefill: max_rows > max_positions");
    const std::size_t T = (options.max_rows != 0U ? options.max_rows : options.max_positions) + 4U * W;  // rows of one run
    m.roles = attn::released_plan();
    std::uint32_t max_count = 1;
    for (std::uint32_t l = 0; l < options.layers; ++l) {
        const auto& r = m.roles[l];
        Impl::LayerState L;
        L.ring = Dev<std::uint8_t>(std::size_t{AttentionHost::ring_capacity} * attn::kWindowEntryBytes);
        if (r.owns_kv) {
            const std::size_t entries = P / r.ratio + 1U;
            max_count = std::max<std::uint32_t>(max_count, static_cast<std::uint32_t>(entries));
            L.kv_cache = Dev<std::uint8_t>(entries * attn::kKvEntryBytes);
            L.idx_cache = Dev<std::uint8_t>(entries * attn::kIdxEntryBytes);
            for (int k = 0; k < 2; ++k) {
                L.carry[k] = Dev<float>(2U * attn::kHeadDim);
                L.carry_valid[k] = Dev<std::uint32_t>(1);
            }
        }
        if (r.owns_index) {
            L.idx = Dev<std::uint32_t>(T * attn::kTopK);
            L.idx_count = Dev<std::uint32_t>(T);
        }
        m.attn.push_back(std::move(L));
    }
    m.cand_window_bytes = W * ((P + attn::kCandBlockSize) / attn::kCandBlockSize);
    m.candidates = Dev<std::uint8_t>((T / W + 1U) * m.cand_window_bytes);
    // a run's rows plus the window before it (restored from the layer's ring by extend) never share a slot
    m.prefill_capacity = static_cast<std::uint32_t>(T + W + attn::kWindow);
    m.prefill_ring = Dev<std::uint8_t>(std::size_t{m.prefill_capacity} * attn::kWindowEntryBytes);
    m.prefill_ring.zero(m.stream);
    check(hipEventCreateWithFlags(&m.chunk_ready, hipEventDisableTiming), "event");
    check(hipStreamCreateWithFlags(&m.comm_stream, hipStreamNonBlocking), "comm stream");
    for (auto& e : m.half_ready) check(hipEventCreateWithFlags(&e, hipEventDisableTiming), "event");
    check(hipEventCreateWithFlags(&m.half_received, hipEventDisableTiming), "event");
    check(hipEventCreateWithFlags(&m.gather_ready, hipEventDisableTiming), "event");
    check(hipEventCreateWithFlags(&m.gathered, hipEventDisableTiming), "event");
    m.lanes.resize(options.attention_lanes);
    for (auto& ln : m.lanes) {
        check(hipStreamCreateWithFlags(&ln.stream, hipStreamNonBlocking), "lane stream");
        check(hipEventCreateWithFlags(&ln.done, hipEventDisableTiming), "lane event");
        ln.index_scores = Dev<float>(W * max_count);
        ln.select_keys = Dev<std::uint32_t>(W * max_count);
        ln.index_qprep = Dev<std::uint8_t>(W * attn::hip::kIdxPrepBytes);
        ln.attn_tickets = Dev<std::uint32_t>(std::size_t{W} * attn::hip::kAttnGroupCount);
        ln.attn_tickets.zero(m.stream);
        ln.attn_partial = Dev<float>(W * attn::kHeads * attn::kMaxChunks * (2U + attn::kHeadDim));
    }
    m.pos64 = Dev<std::uint64_t>(T);
    m.dtokens = Dev<std::uint32_t>(T);
    m.rope_window = Dev<attn::RopeRow>(T);
    m.rope_comp = Dev<attn::RopeRow>(T);
    m.rope_group = Dev<attn::RopeRow>(T);
    m.lpos64 = Dev<std::uint64_t>(T);
    m.ppos64 = Dev<std::uint64_t>(T);
    m.lrope_window = Dev<attn::RopeRow>(T);
    m.lrope_comp = Dev<attn::RopeRow>(T);
    m.lrope_group = Dev<attn::RopeRow>(T);
    m.upload_positions(0U, m.stream);
    m.hasher = std::make_unique<ds41::EngramHasher>(m.model.engram);
    const std::size_t H = c.hidden_size, hc = c.hc_mult, I = c.moe_intermediate_size, K = c.num_experts_per_tok;
    const std::size_t QA = c.q_lora_rank, QH = std::size_t{c.num_attention_heads} * c.head_dim, OL = std::size_t{c.o_groups} * c.o_lora_rank;
    m.streams[0] = Dev<float>(T * hc * H);
    m.streams[1] = Dev<float>(T * hc * H);
    if (options.draft_rows != 0U) {
        if (options.layers < 4U) throw std::invalid_argument("deepseek41 prefill: draft_rows needs at least 4 layers");
        for (int j = 0; j < 3; ++j) {
            m.seed_local[j] = Dev<float>(std::size_t{options.draft_rows} * hc * H);
            m.seed_peer[j] = Dev<float>(std::size_t{options.draft_rows} * hc * H);
            m.seed[j] = Dev<float>(std::size_t{options.draft_rows} * hc * H);
        }
    }
    m.mix_attn = Dev<ds41::HcMix>(T);
    m.mix_ffn = Dev<ds41::HcMix>(T);
    m.mix_partial = Dev<float>(ds41::hc_mix_partial_floats(static_cast<std::uint32_t>(T), static_cast<std::uint32_t>(H)));
    m.hc_counters = Dev<std::uint32_t>(ds41::hc_site_counter_count(static_cast<std::uint32_t>(T)));
    check(hipMemset(m.hc_counters.p, 0, m.hc_counters.n * 4U), "hc counters");
    m.x = Dev<float>(T * H);
    m.xcodes = Dev<std::int8_t>(T * H);
    m.xdeltas = Dev<float>(T * H / 32U);
    m.xsums = Dev<std::int32_t>(T * H / 32U);
    m.qr = Dev<float>(T * QA);
    m.qcodes = Dev<std::int8_t>(T * QA);
    m.qdeltas = Dev<float>(T * QA / 32U);
    m.qsums = Dev<std::int32_t>(T * QA / 32U);
    m.kv = Dev<float>(T * attn::kHeadDim);
    m.ckv = Dev<float>(T * attn::kHeadDim);
    m.cscore = Dev<float>(T * attn::kHeadDim);
    m.iq = Dev<float>(T * attn::kIdxHeads * attn::kIdxDim);
    m.iw = Dev<float>(T * attn::kIdxHeads);
    m.y = Dev<float>(T * H);
    m.shared = Dev<float>(T * H);
    m.scores = Dev<float>(T * c.n_routed_experts);
    m.route_w = Dev<float>(T * K);
    m.ids = Dev<std::uint32_t>(T * K);
    m.trace_hash = Dev<std::uint64_t>(T);
    m.image_rows = Dev<std::uint8_t>(T);
    m.image_rows.zero(m.stream);
    m.q = Dev<float>(C * QH);
    m.o = Dev<float>(C * QH);
    m.ocodes = Dev<std::int8_t>(C * QH);
    m.odeltas = Dev<float>(C * QH / 32U);
    m.osums = Dev<std::int32_t>(C * QH / 32U);
    m.o_low = Dev<float>(C * OL);
    m.olcodes = Dev<std::int8_t>(C * OL);
    m.oldeltas = Dev<float>(C * OL / 32U);
    m.olsums = Dev<std::int32_t>(C * OL / 32U);
    const std::size_t cols = m.model.engram.rows_per_token();
    m.erows = Dev<std::uint8_t>(C * cols * 264U);
    m.emb = Dev<float>(C * cols * 256U);
    m.ekv = Dev<float>(C * H * (hc + 1U));
    m.sh_codes = Dev<std::int8_t>(T * I);
    m.sh_deltas = Dev<float>(T * I / 32U);
    m.sh_sums = Dev<std::int32_t>(T * I / 32U);
    if (m.distributed()) {
        // row_split: the half sums hold one rank's rows of a split run (every other window), or a whole run below
        // split_min_rows (no split); without the row split, every row
        const std::size_t split_rows = ((T + 2U * W - 1U) / (2U * W)) * W;
        const std::size_t half_rows = options.row_split ? std::max(split_rows, std::min<std::size_t>(T, m.split_min_rows)) : T;
        if (options.row_split) {
            bool want = true;
            if (const char* v = std::getenv("STRIX_DS41_PREFILL_STAGED_HALF"); v != nullptr && v[0] == '0') want = false;
            const std::size_t bytes = split_rows * H * 4U;
            auto* comm = options.communicator;
            if (want && comm->send_staging(std::min(bytes, comm->max_bytes())) != nullptr) {
                const std::size_t before = comm->max_bytes();
                if (bytes > before) comm->resize_bytes(bytes, false);  // the staging buffers only (one message per layer)
                m.staged_half = comm->send_staging(bytes) != nullptr;
                if (!m.staged_half && bytes > before) comm->resize_bytes(before, false);
            }
            m.staged_gather = m.staged_half;
            if (const char* v = std::getenv("STRIX_DS41_PREFILL_STAGED_GATHER"); v != nullptr && v[0] == '0') m.staged_gather = false;
            if (const char* v = std::getenv("STRIX_DS41_PREFILL_EARLY_QB"); v != nullptr && v[0] == '0') m.early_q_b = false;
        }
        m.half_own = Dev<float>(half_rows * H);
        m.half_other = Dev<float>(half_rows * H);
        if (options.row_split) {
            m.peer_kv = Dev<float>(T * attn::kHeadDim);
            m.peer_ckv = Dev<float>(T * attn::kHeadDim);
            m.peer_cscore = Dev<float>(T * attn::kHeadDim);
            m.peer_xcodes = Dev<std::int8_t>(T * H);
            m.peer_xdeltas = Dev<float>(T * H / 32U);
            m.peer_ids = Dev<std::uint32_t>(T * K);
            m.peer_route_w = Dev<float>(T * K);
            m.half_recv = Dev<float>(half_rows * H);
            m.h_peer = Dev<float>(H);
            // the widest packed row: window K/V + compressor kv + score (3 x 512 floats), or the MoE input
            const std::size_t pack_words = std::max<std::size_t>(3U * attn::kHeadDim, H / 4U + H / 32U + 2U * K);
            m.pack_send = Dev<std::uint32_t>(T * pack_words);
            m.pack_recv = Dev<std::uint32_t>(T * pack_words);
        }
        if (options.tp_attention) {
            m.y_own = Dev<float>(C * H);
            m.y_other = Dev<float>(C * H);
        }
    }
    if(options.quality_head_rows>1) {
        if(options.row_split || options.cone || options.quality_head_rows>64)
            throw std::invalid_argument("quality head rows requires no row split, no cone, <=64 rows");
        auto nr=options.quality_head_rows;
        m.quality_h=Dev<float>(std::size_t(nr)*H);
        m.quality_logits=Dev<float>(std::size_t(nr)*c.vocab_size);
        m.quality_codes=Dev<std::int8_t>(std::size_t(nr)*H);
        m.quality_deltas=Dev<float>(std::size_t(nr)*H/32);
        m.quality_sums=Dev<std::int32_t>(std::size_t(nr)*H/32);
    }
#ifndef STRIX_DS41_PREFILL_FP16_LT
    if(options.fp16_dense)throw std::invalid_argument("FP16 prefill backend not built");
#endif
    m.h = Dev<float>(H);
    m.logits = Dev<float>(c.vocab_size);
    m.hcodes = Dev<std::int8_t>(H);
    m.hdeltas = Dev<float>(H / 32U);
    m.hsums = Dev<std::int32_t>(H / 32U);
    std::uint32_t partial_blocks = 0;
    for (std::uint32_t r = 0; r < 2U; ++r)
        partial_blocks = std::max(partial_blocks, ds41::argmax_partial_blocks(ds41::vocab_slice(c.vocab_size, 2U, r).count));
    m.partial = Dev<ds41::ArgmaxCandidate>(partial_blocks);
    m.cand0 = Dev<ds41::ArgmaxCandidate>(1);
    m.cand1 = Dev<ds41::ArgmaxCandidate>(1);
    m.best = Dev<ds41::ArgmaxCandidate>(1);
    m.cand_own = Dev<ds41::ArgmaxCandidate>(1);
    m.cand_other = Dev<ds41::ArgmaxCandidate>(1);
    check(hipStreamSynchronize(m.stream), "setup");
    if (options.warmup_rows != 0U) {  // one run of a dummy prompt (first launches, scratch first touch); the next run starts fresh
        const std::uint32_t n = std::min<std::uint32_t>(options.warmup_rows, options.max_rows != 0U ? options.max_rows : options.max_positions);
        std::vector<std::uint32_t> dummy(n);
        for (std::uint32_t i = 0; i < n; ++i) dummy[i] = 1000U + i;
        (void)run(dummy);
        m.rows_done = 0;
        m.local_rows_done = 0;
        m.last_row_local = true;
    }
    m.load_seconds = seconds_since(t_load);
}

DevicePrefill::~DevicePrefill() {
    if (!impl_) return;
    if (impl_->prefetch.joinable()) impl_->prefetch.join();
    for (auto& ln : impl_->lanes) {
        if (ln.done != nullptr) (void)hipEventDestroy(ln.done);
        if (ln.stream != nullptr) (void)hipStreamDestroy(ln.stream);
    }
    if (impl_->chunk_ready != nullptr) (void)hipEventDestroy(impl_->chunk_ready);
    for (auto& e : impl_->half_ready)
        if (e != nullptr) (void)hipEventDestroy(e);
    if (impl_->half_received != nullptr) (void)hipEventDestroy(impl_->half_received);
    if (impl_->gather_ready != nullptr) (void)hipEventDestroy(impl_->gather_ready);
    if (impl_->gathered != nullptr) (void)hipEventDestroy(impl_->gathered);
    if (impl_->comm_stream != nullptr) (void)hipStreamDestroy(impl_->comm_stream);
    if (impl_->stream != nullptr) (void)hipStreamDestroy(impl_->stream);
    impl_->join_engram_ahead();
    if (impl_->engram_pinned != nullptr) (void)hipHostFree(impl_->engram_pinned);
}

std::uint64_t DevicePrefill::weight_bytes() const noexcept {
    std::uint64_t n = 0;
    for (const auto& [name, d] : impl_->weights) n += d.owned ? d.n : 0U;
    return n;
}
std::uint64_t DevicePrefill::shared_weight_bytes() const noexcept {
    std::uint64_t n = 0;
    for (const auto& [name, d] : impl_->weights) n += d.owned ? 0U : d.n;
    return n;
}
double DevicePrefill::load_seconds() const noexcept { return impl_->load_seconds; }
std::uint64_t DevicePrefill::owned_weight_bytes(const std::filesystem::path& gguf, const PrefillOptions& o) {
    const auto model = validate_target(read_file(gguf));
    const auto plan = build_decode_plan(model, 0, o.layers);
    const auto& c = model.config;
    std::map<std::string, std::uint64_t> sizes;
    for (const auto& t : model.manifest.tensors) sizes.emplace(t.name, t.byte_size);
    std::set<std::string> names;
    for (const auto& st : plan.steps)
        for (const auto& w : st.weights)
            if (w.find("engram_embd") == std::string::npos) names.insert(w);
    const std::uint32_t experts = o.expert_tp ? c.n_routed_experts : c.n_routed_experts / std::max(1U, o.world);
    const auto head = o.world == 2U ? ds41::vocab_slice(c.vocab_size, 2U, o.rank).count : c.vocab_size;
    std::uint64_t need = 0;
    for (const auto& n : names) {  // the constructor's rules: the expert range (or TP part), the vocabulary slice, shared tensors
        std::uint64_t bytes = sizes.at(n);
        const bool exps = n.ends_with("_exps.weight");
        if (exps && o.expert_tp) bytes = ds41::expert_tp::part_bytes(bytes, c.n_routed_experts, n.ends_with("ffn_down_exps.weight"), c.hidden_size);
        else if (exps) bytes = bytes / c.n_routed_experts * experts * (o.stream_experts ? 2U : 1U);
        else if (n == "output.weight") bytes = bytes / c.vocab_size * head;
        const auto it = o.shared_weights.find(n);
        if (o.stream_experts && exps) need += bytes;
        else if (it == o.shared_weights.end() || it->second.second != bytes) need += bytes;
    }
    return need;
}
std::string DevicePrefill::overlay_description() const { return impl_->overlay ? impl_->overlay->describe() : std::string{}; }

namespace {

const char* expert_kind_tensor(const std::string& kind) {
    return kind == "gate" ? "ffn_gate_exps.weight" : kind == "up" ? "ffn_up_exps.weight" : "ffn_down_exps.weight";
}

}  // namespace

DevicePrefill::DraftSeed DevicePrefill::draft_seed() const noexcept {
    const auto& m = *impl_;
    if (m.state_incomplete) return DraftSeed{};  // a deferred run leaves no seed
    return DraftSeed{{m.seed[0].p, m.seed[1].p, m.seed[2].p}, m.seed_rows, m.seed_first};
}

// ---- PrefillTask: run_from on a worker thread that stops at the unit boundaries ----
namespace {
struct TaskCancelled {};
}  // namespace

struct PrefillTask::State {
    DevicePrefill* owner = nullptr;
    DevicePrefill::Impl* m = nullptr;
    std::vector<std::uint32_t> tokens;
    std::uint32_t start = 0;
    PrefillTaskOptions opt;
    int device = 0;
    std::thread worker;
    std::mutex mu;
    std::condition_variable cv;
    std::uint32_t granted = 0;  // units the worker may still end before it parks
    bool parked = true, finished = false, cancel_req = false, cancelled = false;
    std::exception_ptr error;
    PrefillResult result;
    std::uint32_t units = 0;  // units ended
    // the unit in progress: its layer and kind, its index within the layer
    std::int32_t layer = -1;
    const char* kind = "setup";
    std::uint32_t in_layer = 0;
    std::uint32_t ordinal = 0;  // units of the same kind before it in its layer
    std::map<std::pair<std::int32_t, std::string>, std::uint32_t> kind_count;
    std::vector<std::pair<hipEvent_t, hipEvent_t>> events;  // per unit: start, end (unit_timing)
    std::uint32_t timed = 0;                                // units whose device time is read
    std::vector<PrefillUnitTiming> timings;
    std::vector<std::pair<std::int32_t, std::uint32_t>> where;  // per ended unit: (layer, index within the layer)
    std::map<std::pair<std::int32_t, std::uint32_t>, double> by_place;
    std::vector<double> walls;

    hipEvent_t new_event() {
        hipEvent_t e = nullptr;
        check(hipEventCreate(&e), "task event");
        return e;
    }
    void unit_started() {
        if (!opt.unit_timing) return;
        events.emplace_back(new_event(), nullptr);
        check(hipEventRecord(events.back().first, m->stream), "unit start");
    }
    void unit_ended() {
        timings.push_back({units, layer, kind, 0.0});
        where.emplace_back(layer, in_layer);
        if (opt.unit_timing) {
            events.back().second = new_event();
            check(hipEventRecord(events.back().second, m->stream), "unit end");
        }
        ++units;
    }
    // every stream of the prefill idle (the units' work complete); then the device times of the ended units
    void drain() {
        check(hipStreamSynchronize(m->stream), "task drain");
        if (m->comm_stream != nullptr) check(hipStreamSynchronize(m->comm_stream), "task drain");
        for (auto& ln : m->lanes) check(hipStreamSynchronize(ln.stream), "task drain");
        for (; opt.unit_timing && timed < units; ++timed) {
            float ms = 0.0F;
            check(hipEventElapsedTime(&ms, events[timed].first, events[timed].second), "unit time");
            timings[timed].device_ms = ms;
            by_place[where[timed]] = ms;
        }
    }
    // The worker, inside a unit: do the next `boundaries` boundaries at `level` all pass without parking (no other
    // work on the communicator in between)? Both ranks hold the same grant at the same unit (the scheduler's budget).
    bool runs_through(PrefillUnitLevel level, std::uint32_t boundaries) {
        if (static_cast<std::uint32_t>(level) > static_cast<std::uint32_t>(opt.level)) return true;
        std::lock_guard<std::mutex> lock(mu);
        return granted > boundaries;
    }
    // A unit boundary of run_from at `level`: the unit in progress ends; the next one runs `next_kind` in `next_layer`.
    void boundary(PrefillUnitLevel level, std::int32_t next_layer, const char* next_kind) {
        if (static_cast<std::uint32_t>(level) > static_cast<std::uint32_t>(opt.level)) return;
        unit_ended();
        in_layer = next_layer == layer ? in_layer + 1U : 0U;
        layer = next_layer;
        kind = next_kind;
        ordinal = kind_count[{layer, std::string(kind)}]++;
        std::unique_lock<std::mutex> lock(mu);
        if (--granted == 0U) {
            lock.unlock();
            drain();
            lock.lock();
            parked = true;
            cv.notify_all();
            cv.wait(lock, [&] { return granted > 0U || cancel_req; });
            if (cancel_req) throw TaskCancelled{};
            parked = false;
        }
        lock.unlock();
        unit_started();
    }
    void run_worker() {
        {
            std::unique_lock<std::mutex> lock(mu);
            cv.wait(lock, [&] { return granted > 0U || cancel_req; });
            if (cancel_req) {
                cancelled = finished = parked = true;
                m->cancel_broken = true;  // even an unstarted abandoned task needs run/import before extend
                m->defer_next = 0;        // and drops its one-shot decoder-deferral request
                m->task = nullptr;
                cv.notify_all();
                return;
            }
            parked = false;
        }
        bool ok = false;
        try {
            check(hipSetDevice(device), "task device");
            unit_started();
            result = owner->run_from(tokens, start);
            unit_ended();  // the last unit ends with the result
            drain();
            ok = true;
        } catch (const TaskCancelled&) {
            cancelled = true;
        } catch (...) {
            error = std::current_exception();
        }
        if (!ok) {
            m->cancel_broken = true;  // the sequence's state is partial
            m->defer_next = 0;
            try {
                drain();
            } catch (...) {
            }
        }
        std::lock_guard<std::mutex> lock(mu);
        m->task = nullptr;
        finished = parked = true;
        cv.notify_all();
    }
    ~State() {
        for (auto& [a, b] : events) {
            if (a != nullptr) (void)hipEventDestroy(a);
            if (b != nullptr) (void)hipEventDestroy(b);
        }
    }
};

PrefillTask::PrefillTask(std::unique_ptr<State> s) : s_(std::move(s)) {}

PrefillTask::~PrefillTask() {
    if (!s_) return;
    cancel();
    if (s_->worker.joinable()) s_->worker.join();
}

std::uint32_t PrefillTask::step(std::uint32_t max_units) {
    auto& s = *s_;
    if (max_units == 0U) throw std::invalid_argument("deepseek41 prefill task: step needs max_units >= 1");
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(s.mu);
    if (s.finished) return 0U;
    const std::uint32_t before = s.units;
    s.granted = max_units;
    s.parked = false;
    s.cv.notify_all();
    s.cv.wait(lock, [&] { return s.parked; });
    s.walls.push_back(seconds_since(t0) * 1e3);
    if (s.error) std::rethrow_exception(s.error);
    return s.units - before;
}

bool PrefillTask::done() const noexcept {
    std::lock_guard<std::mutex> lock(s_->mu);
    return s_->finished;
}

std::uint32_t PrefillTask::units_done() const noexcept { return s_->units; }

std::uint32_t PrefillTask::units_within(double budget_ms) const {
    // The layers repeat their units: the units after the one in progress are predicted by the previous layer's units
    // from the same (kind, ordinal) on, cyclically (the next layer like the previous one), each by its device time,
    // + 10 % and 0.5 ms of host overhead. Without such a template (the first layers, a kind the previous layer did not
    // have, e.g. the Engram pieces), one unit.
    const auto& s = *s_;
    std::vector<std::pair<std::string, double>> tpl;
    for (std::uint32_t i = 0; i < s.timed; ++i)
        if (s.timings[i].layer == s.layer - 1 && s.layer > 0) tpl.emplace_back(s.timings[i].kind, s.timings[i].device_ms);
    std::size_t at = tpl.size();
    for (std::size_t i = 0, seen = 0; i < tpl.size(); ++i)
        if (tpl[i].first == s.kind && seen++ == s.ordinal) {
            at = i;
            break;
        }
    if (at == tpl.size()) return 1U;
    std::uint32_t n = 0;
    double sum = 0.0;
    for (std::size_t i = at; n < 4096U; ++n, i = (i + 1U) % tpl.size()) {
        const double t = tpl[i].second * 1.1 + 0.5;
        if (n > 0U && sum + t > budget_ms) break;
        sum += t;
    }
    return std::max<std::uint32_t>(1U, n);
}

const PrefillResult& PrefillTask::result() const {
    const auto& s = *s_;
    if (s.error) std::rethrow_exception(s.error);
    if (s.cancelled) throw std::logic_error("deepseek41 prefill task: cancelled");
    if (!s.finished) throw std::logic_error("deepseek41 prefill task: not done");
    return s.result;
}

const std::vector<PrefillUnitTiming>& PrefillTask::unit_timings() const noexcept { return s_->timings; }
const std::vector<double>& PrefillTask::step_walls() const noexcept { return s_->walls; }

void PrefillTask::cancel() {
    auto& s = *s_;
    std::unique_lock<std::mutex> lock(s.mu);
    if (s.finished) return;
    s.cancel_req = true;
    s.cv.notify_all();
    s.cv.wait(lock, [&] { return s.finished; });
}

namespace {
// The global rows of `rank`'s local rows in a run of T rows: with row_split (world 2) window k of W rows belongs to rank
// k % 2 and a rank's rows are compacted in window order; otherwise local rows = global rows (the run's own mapping).
std::vector<std::uint32_t> rank_global_rows(bool split, std::uint32_t T, std::uint32_t W, std::uint32_t rank) {
    std::vector<std::uint32_t> rows;
    rows.reserve(split ? T / 2U + W : T);
    for (std::uint32_t g = 0; g < T; ++g)
        if (!split || (g / W) % 2U == rank) rows.push_back(g);
    return rows;
}
}  // namespace

std::unique_ptr<PrefillTask> DevicePrefill::begin_run(std::span<const std::uint32_t> tokens, const PrefillTaskOptions& task) {
    return begin_from(tokens, false, task);
}
std::unique_ptr<PrefillTask> DevicePrefill::begin_extend(std::span<const std::uint32_t> tokens, const PrefillTaskOptions& task) {
    return begin_from(tokens, true, task);
}

std::unique_ptr<PrefillTask> DevicePrefill::begin_from(std::span<const std::uint32_t> tokens, bool extend, const PrefillTaskOptions& task) {
    auto& m = *impl_;
    if (m.task != nullptr) throw std::logic_error("deepseek41 prefill: a task is in progress");
    if (extend && (m.broken || m.cancel_broken)) throw std::logic_error("deepseek41 prefill: the sequence was abandoned (cancelled task); run() first");
    auto s = std::make_unique<PrefillTask::State>();
    s->owner = this;
    s->m = &m;
    s->tokens.assign(tokens.begin(), tokens.end());
    s->start = extend ? m.rows_done : 0U;
    s->opt = task;
    check(hipGetDevice(&s->device), "task device");
    m.task = s.get();
    auto* raw = s.get();
    s->worker = std::thread([raw] { raw->run_worker(); });
    return std::unique_ptr<PrefillTask>(new PrefillTask(std::move(s)));
}

void DevicePrefill::set_precision_for_new_sequence(bool dense,bool attention) {
    if(impl_->task)throw std::logic_error("cannot change precision during a prefill task");
#ifndef STRIX_DS41_PREFILL_FP16_LT
    if(dense)throw std::invalid_argument("FP16 prefill backend not built");
#endif
    impl_->options.fp16_dense=dense;impl_->options.bf16_attention=attention;impl_->broken=true;
}

void DevicePrefill::set_dense_select(bool on) {
    if (impl_->task) throw std::logic_error("cannot change the dense selector during a prefill task");
    auto& m = *impl_;
    m.dense_select = on;
    m.dense_q_b = on || m.env_dense_q_b;
    m.dense_o_a = on || m.env_dense_o_a;
    m.broken = true;
}

void DevicePrefill::set_precision_for_new_sequence(const PrefillOptions& p) {
    set_precision_for_new_sequence(p.fp16_dense, p.bf16_attention);
    auto& o = impl_->options;
    o.fp16_projections = p.fp16_projections;
    o.approx_layer_first = p.approx_layer_first;
    o.approx_layer_end = p.approx_layer_end;
    o.fp16_min_rows = p.fp16_min_rows;
    o.fp16_keep_a8 = p.fp16_keep_a8;
    o.fp16_fast = p.fp16_fast;
    o.fp16_wsplit = p.fp16_wsplit;
    o.null_ulp_layer = p.null_ulp_layer;
    o.null_seed = p.null_seed;
    o.trace = p.trace;
    o.fast_attention = p.fast_attention;
    o.fast_moe = p.fast_moe;
    ds41::set_fast_moe(p.fast_moe ? 1 : 0);   // process-wide switch of the batch MoE dispatch (one sequence at a time here)
}

PrefillResult DevicePrefill::run(std::span<const std::uint32_t> tokens) {
    if (impl_->task != nullptr) throw std::logic_error("deepseek41 prefill: a task is in progress");
    return run_from(tokens, 0U);
}

PrefillResult DevicePrefill::extend(std::span<const std::uint32_t> tokens) {
    if (impl_->task != nullptr) throw std::logic_error("deepseek41 prefill: a task is in progress");
    if (impl_->broken || impl_->cancel_broken) throw std::logic_error("deepseek41 prefill: the sequence was abandoned (cancelled task); run() first");
    return run_from(tokens, impl_->rows_done);
}

std::uint32_t DevicePrefill::positions() const noexcept { return impl_->rows_done; }

PrefillResult DevicePrefill::run_from(std::span<const std::uint32_t> tokens, std::uint32_t start) {
    auto& m = *impl_;
    const ds41::RnWideOff own_kernels;  // the prefill keeps its own Q8_0 x A8 kernels (P12: the multi-row form is -1 %)
    // stream_experts: layer `layer`'s experts into slot layer % 2 (on the prefetch thread or inline)
    const auto load_experts = [&m](std::uint32_t layer) {
        const auto t0 = std::chrono::steady_clock::now();
        auto& slot = m.slots[layer % 2U];
        slot.layer = -1;
        DirectLoader loader(m.gguf);
        for (auto& [kind, d] : slot.kinds) {
            const auto& [offset, bytes] = m.expert_extent.at("blk." + std::to_string(layer) + "." + expert_kind_tensor(kind));
            loader.load(offset, bytes, d.p);
        }
        slot.layer = static_cast<std::int32_t>(layer);
        m.expert_read_seconds += seconds_since(t0);
    };
    const auto start_prefetch = [&m, load_experts](std::uint32_t layer) {
        if (layer >= m.options.layers) return;
        m.prefetch = std::thread([&m, load_experts, layer] {
            try {
                load_experts(layer);
            } catch (...) {
                m.prefetch_error = std::current_exception();
            }
        });
    };
    const auto join_prefetch = [&m] {
        if (!m.prefetch.joinable()) return;
        const auto t0 = std::chrono::steady_clock::now();
        m.prefetch.join();
        m.expert_wait_seconds += seconds_since(t0);
        if (m.prefetch_error) std::rethrow_exception(std::exchange(m.prefetch_error, nullptr));
    };
    const auto expert_weight = [&m](std::uint32_t layer, const std::string& kind) -> const std::uint8_t* {
        if (!m.options.stream_experts) return m.dev("blk." + std::to_string(layer) + "." + expert_kind_tensor(kind));
        const auto& slot = m.slots[layer % 2U];
        if (slot.layer != static_cast<std::int32_t>(layer)) throw std::logic_error("deepseek41 prefill: expert slot not loaded");
        return slot.kinds.at(kind).p;
    };
    m.expert_read_seconds = m.expert_wait_seconds = 0.0;
    if (m.options.stream_experts) {
        join_prefetch();
        if (m.slots[0].layer != 0) load_experts(0U);
        if (m.slots[1].layer != 1) start_prefetch(1U);
    }
    const auto& c = m.model.config;
    namespace attn = ds41::attn;
    const std::uint32_t T = static_cast<std::uint32_t>(tokens.size());  // rows of this run, at positions [P0, P0 + T)
    ds41::prefill_tuning::set_run_rows(T);  // the tile policy's size rule (ds41_prefill_tuning.hpp)
    const std::uint32_t P0 = start;
    const std::uint32_t max_rows = m.options.max_rows != 0U ? m.options.max_rows : m.options.max_positions;
    if (T == 0U || T > max_rows || std::uint64_t{P0} + T > m.options.max_positions)
        throw std::invalid_argument("deepseek41 prefill: rows outside 1..max_rows or positions beyond max_positions");
    if (P0 % m.options.window_rows != 0U) throw std::invalid_argument("deepseek41 prefill: extend needs the rows so far to be a multiple of window_rows");
    if (m.pos_base != P0) m.upload_positions(P0, m.stream);
    m.snap_points = std::exchange(m.snap_next, {});
    for (const auto p : m.snap_points)
        if (p <= P0 || p >= P0 + T || p % m.options.window_rows != 0U)
            throw std::invalid_argument("deepseek41 prefill: snapshot point outside the run or not a multiple of window_rows");
    while (m.snap_rings.size() < m.snap_points.size() * m.options.layers)
        m.snap_rings.emplace_back(std::size_t{AttentionHost::ring_capacity} * attn::kWindowEntryBytes);
    for (const auto t : tokens)
        if (t >= c.vocab_size) throw std::invalid_argument("deepseek41 prefill: token outside the vocabulary");
    const std::uint32_t H = c.hidden_size, hc = c.hc_mult, K = c.num_experts_per_tok, I = c.moe_intermediate_size;
    const std::uint32_t QA = c.q_lora_rank, QH = c.num_attention_heads * c.head_dim, OL = c.o_groups * c.o_lora_rank;
    const std::uint32_t W = m.options.window_rows, C = m.options.chunk_rows, D = attn::kHeadDim;
    hipStream_t st = m.stream;
    PrefillResult result;
    result.layers.assign(m.options.layers, PrefillLayerTiming{});
    if (P0 == 0U) m.broken = m.cancel_broken = false;
    // PrefillTask: a unit boundary (run() / extend(): none). Cut points only between enqueues with no exchange in
    // flight; loops whose count differs between the ranks are padded so both ranks see the same list of units.
    const auto cut = [&](PrefillUnitLevel level, std::int32_t next_layer, const char* kind) {
        if (m.task != nullptr) m.task->boundary(level, next_layer, kind);
    };
    std::int32_t cut_layer = -1;
    const auto t_run = std::chrono::steady_clock::now();

    // Rows each layer computes. Without the cone every layer runs every row. With it (the full 40 layers): encoder
    // layers every row; decoder layer L its attention input from x0 (layer 20: every row, for the compressor and the
    // index keys every decoder reads), its window K/V from k0 and its whole output from o0, as plan_suffix_replay
    // gives them, rounded down to the attention window so the windows of every layer line up (the extra rows are exact).
    // Decoder deferral (set_decoder_deferral, STRIX_DS41_DEFER_DECODER default on): admitted when the continuation's
    // own cone, as this code plans it (rounded down to W, with the draft rows), starts at least one attention window
    // after the continuation's first row in every decoder layer, so it reads no decoder ring entry of this run.
    const std::uint32_t first_decoder = ds41::SuffixReplayGeometry{}.first_decoder;
    bool defer = false;
    {
        const std::uint32_t next_rows = std::exchange(m.defer_next, 0U);
        static const bool defer_on = [] { const char* v = std::getenv("STRIX_DS41_DEFER_DECODER"); return !(v && std::strcmp(v, "0") == 0); }();
        if (next_rows != 0U && defer_on && m.options.cone && m.options.layers == c.num_hidden_layers && m.options.layers > first_decoder &&
            m.snap_points.empty() && (P0 + T) % W == 0U) {
            ds41::SuffixReplayGeometry g;
            g.layers = c.num_hidden_layers;
            g.draft_rows = m.options.draft_rows;
            defer = true;
            for (const auto& pl : ds41::plan_suffix_replay(g, 0U, next_rows).layers)
                if (pl.layer >= first_decoder && pl.out_begin / W * W < attn::kWindow) defer = false;
        }
        if (m.distributed() && m.options.cone && m.options.layers == c.num_hidden_layers) {  // both ranks: the same decision
            const std::array<std::uint32_t, 2> mine{defer ? 1U : 0U, defer ? next_rows : 0U};
            std::array<std::uint32_t, 2> other{};
            m.options.communicator->exchange_host(std::as_bytes(std::span(mine)), std::as_writable_bytes(std::span(other)),
                                                  dist::Tag{static_cast<std::uint32_t>(dist::TagKind::control), 0x64656631U, P0});
            if (other != mine)
                throw std::runtime_error("deepseek41 prefill: the ranks disagree on the decoder deferral of the run at " + std::to_string(P0));
        }
    }
    result.decoder_deferred = defer;
    if (defer) result.next_token = 0xFFFFFFFFU;
    struct Rows { std::uint32_t x0 = 0, k0 = 0, o0 = 0; };
    std::vector<Rows> rows(m.options.layers);
    if (m.options.cone && m.options.layers == c.num_hidden_layers) {
        ds41::SuffixReplayGeometry g;
        g.layers = c.num_hidden_layers;
        g.draft_rows = m.options.draft_rows;
        const auto plan = ds41::plan_suffix_replay(g, 0U, T);
        const auto down = [&](std::uint64_t v) { return static_cast<std::uint32_t>(v / W * W); };
        for (const auto& pl : plan.layers) {
            if (pl.layer >= m.options.layers) continue;
            auto& r = rows[pl.layer];
            r.o0 = down(pl.out_begin);
            r.k0 = down(pl.kv_begin);
            r.x0 = m.roles[pl.layer].owns_kv ? 0U : r.k0;
        }
        // in-run snapshot points: also the cone of the state at p (one contiguous suffix per layer: the union)
        for (const auto p : m.snap_points) {
            ds41::SuffixReplayGeometry gp = g;
            gp.draft_rows = 0;
            for (const auto& pl : ds41::plan_suffix_replay(gp, 0U, p - P0).layers) {
                if (pl.layer >= m.options.layers) continue;
                auto& r = rows[pl.layer];
                r.o0 = std::min(r.o0, down(pl.out_begin));
                r.k0 = std::min(r.k0, down(pl.kv_begin));
                r.x0 = m.roles[pl.layer].owns_kv ? 0U : r.k0;
            }
        }
        if (defer) {  // decoder layers: no output and no window rows; a compressed-KV source keeps every input row
            for (std::uint32_t l = first_decoder; l < m.options.layers; ++l) {
                auto& r = rows[l];
                r.o0 = r.k0 = T;
                r.x0 = m.roles[l].owns_kv ? 0U : T;
            }
        }
        result.row_layers = 0;
        for (const auto& r : rows) result.row_layers += T - r.o0;
        // the continuation of a deferred run: no decoder layer may read a ring entry of that run (rows before P0)
        if (m.state_incomplete && P0 != 0U)
            for (std::uint32_t l = first_decoder; l < m.options.layers; ++l)
                if (rows[l].o0 < attn::kWindow)
                    throw std::logic_error("deepseek41 prefill: this run's decoder cone reaches into the previous run, which deferred its decoder");
    } else if (m.state_incomplete && P0 != 0U) {
        throw std::logic_error("deepseek41 prefill: a run without the cone cannot continue a deferred run");
    } else {
        result.row_layers = std::uint64_t{T} * m.options.layers;
    }

    // Rows of this rank. row_split (world 2): window k of W rows belongs to rank k % 2, and a rank computes its rows
    // compacted in window order ("local rows", each with its own position); otherwise local rows = global rows.
    const bool split = m.distributed() && m.options.row_split && T >= m.split_min_rows;
    const std::uint32_t me = m.options.rank, peer = 1U - me, nw = (T + W - 1U) / W;
    const auto windows_before = [&](std::uint32_t rank, std::uint32_t k) { return split ? (k + 1U - rank) / 2U : k; };
    const auto local_of = [&](std::uint32_t rank, std::uint32_t g0) { return windows_before(rank, g0 / W) * W; };  // g0: window start
    const auto rows_of = [&](std::uint32_t rank) {
        if (!split) return T;
        std::uint32_t n = windows_before(rank, nw) * W;
        if (T % W != 0U && (nw - 1U) % 2U == rank) n -= W - T % W;
        return n;
    };
    const auto window_owner = [&](std::uint32_t k) { return split ? k % 2U : me; };
    const auto owner_window = [&](std::uint32_t k) { return split ? k / 2U : k; };  // index among its owner's windows
    const auto global_window = [&](std::uint32_t j) { return split ? 2U * j + me : j; };  // of this rank's local window j
    const std::uint32_t Tn = rows_of(me), Tp = split ? rows_of(peer) : 0U;
    const bool last_local = window_owner(nw - 1U) == me;
    std::vector<std::uint32_t> global_row(Tn);
    for (std::uint32_t l = 0; l < Tn; ++l) global_row[l] = global_window(l / W) * W + l % W;
    {
        std::vector<std::uint64_t> pos(Tn);
        std::vector<attn::RopeRow> rw(Tn), rc(Tn), rg(Tn);
        for (std::uint32_t l = 0; l < Tn; ++l) {
            const std::uint64_t p = P0 + global_row[l];
            pos[l] = p;
            rw[l] = attn::rope_row(attn::RopeKind::window, p);
            rc[l] = attn::rope_row(attn::RopeKind::compressed, p);
            rg[l] = attn::rope_row(attn::RopeKind::compressed, p / 2U * 2U);
        }
        m.lpos64.upload(pos, st);
        if (split && Tp) {
            std::vector<std::uint64_t> ppos(Tp);
            for (std::uint32_t l = 0; l < Tp; ++l) ppos[l] = P0 + (2U * (l / W) + peer) * W + l % W;
            m.ppos64.upload(ppos, st);
        }
        m.lrope_window.upload(rw, st);
        m.lrope_comp.upload(rc, st);
        m.lrope_group.upload(rg, st);
    }
    // the same ranges in this rank's local rows
    std::vector<Rows> lrows(m.options.layers);
    for (std::uint32_t l = 0; l < m.options.layers; ++l)
        lrows[l] = {local_of(me, rows[l].x0), local_of(me, rows[l].k0), local_of(me, rows[l].o0)};
    result.last_row_local = last_local;
    result.local_rows = Tn;
    // draft_rows: the seed is the run's last rows; this rank captures its own (a suffix of its local rows)
    const std::uint32_t seed_n = std::min(m.options.draft_rows, T), seed_begin = T - seed_n;
    std::uint32_t seed_l0 = Tn;
    for (std::uint32_t l = 0; l < Tn; ++l)
        if (global_row[l] >= seed_begin) {
            seed_l0 = l;
            break;
        }
    const std::uint32_t seed_own = Tn - seed_l0;
    // row_split all-gathers on the communication stream: rows of this rank's buffer `own` to the peer and the peer's
    // rows into `dst`, both padded to the larger count, pieces of <= 16 MB. The caller records gather_ready after the
    // producers (start_gather) and enqueues independent work on the main stream before exchange_rows, which blocks the
    // CPU until the pieces have crossed; join_gather() orders the main stream after them.
    std::uint64_t exchange_seq = 0;
    const auto start_gather = [&] {
        if (!m.async_gather) return;
        check(hipEventRecord(m.gather_ready, st), "gather ready");
        check(hipStreamWaitEvent(m.comm_stream, m.gather_ready, 0), "gather ready");
    };
    const auto exchange_rows = [&](const void* own, std::uint32_t own_rows, void* dst, std::uint32_t peer_rows, std::size_t row_bytes,
                                   std::uint32_t layer) {
        const std::uint32_t n = std::max(own_rows, peer_rows);
        const auto per = static_cast<std::uint32_t>(std::max<std::size_t>(1U, (std::size_t{16} << 20U) / row_bytes));
        hipStream_t gs = m.async_gather ? m.comm_stream : st;
        for (std::uint32_t p0 = 0; p0 < n; p0 += per) {
            const std::uint32_t pn = std::min(per, n - p0);
            m.options.communicator->start(gs, static_cast<const std::uint8_t*>(own) + std::size_t{p0} * row_bytes, std::size_t{pn} * row_bytes,
                                          dist::Tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), layer, ++exchange_seq});
            m.options.communicator->finish(gs, static_cast<std::uint8_t*>(dst) + std::size_t{p0} * row_bytes);
        }
    };
    // the same for several fields of the same rows (own buffers at this rank's first row, destinations at the peer's),
    // packed into one message per piece
    struct Field { const void* own; void* dst; std::size_t row_bytes; };
    const auto pack_of = [](std::initializer_list<Field> fields) {
        PackFields pf;
        for (const auto& f : fields) {
            pf.src[pf.count] = static_cast<const std::uint32_t*>(f.own);
            pf.dst[pf.count] = static_cast<std::uint32_t*>(f.dst);
            pf.words[pf.count] = static_cast<std::uint32_t>(f.row_bytes / 4U);
            pf.stride += pf.words[pf.count];
            ++pf.count;
        }
        return pf;
    };
    const auto exchange_fields = [&](std::initializer_list<Field> fields, std::uint32_t own_rows, std::uint32_t peer_rows, std::uint32_t layer) {
        if (!m.pack || fields.size() == 1U) {
            for (const auto& f : fields) exchange_rows(f.own, own_rows, f.dst, peer_rows, f.row_bytes, layer);
            return;
        }
        const std::uint32_t n = std::max(own_rows, peer_rows);
        if (n == 0U) return;
        const PackFields pf = pack_of(fields);
        // on the stream the exchange runs on (async: the communication stream, after its wait for the producers)
        hipStream_t us = m.async_gather ? m.comm_stream : st;
        hipLaunchKernelGGL(pack_rows_kernel, dim3(n), dim3(256), 0, us, pf, m.pack_send.p);
        check(hipGetLastError(), "pack rows");
        exchange_rows(m.pack_send.p, n, m.pack_recv.p, n, std::size_t{pf.stride} * 4U, layer);
        hipLaunchKernelGGL(unpack_rows_kernel, dim3(n), dim3(256), 0, us, pf, m.pack_recv.p);
        check(hipGetLastError(), "unpack rows");
    };
    const auto join_gather = [&] {
        if (!m.async_gather) return;
        check(hipEventRecord(m.gathered, m.comm_stream), "gathered");
        check(hipStreamWaitEvent(st, m.gathered, 0), "gather join");
    };
    // row_split, staged MoE input (m.staged_gather): packed on the main stream straight into the pinned send buffer and
    // put on the wire right after the router, so it crosses while the shared expert runs (a pack kernel or a copy on the
    // communication stream waits behind the shared expert's kernels); unpacked from the pinned receive buffer
    const auto moe_input_pack = [&](const Rows& R, std::uint32_t own0) {
        const std::uint32_t p0 = local_of(peer, R.o0);
        return pack_of({Field{m.xcodes.p + std::size_t{own0} * H, m.peer_xcodes.p + std::size_t{p0} * H, H},
                        Field{m.xdeltas.p + std::size_t{own0} * H / 32U, m.peer_xdeltas.p + std::size_t{p0} * H / 32U, H / 32U * 4U},
                        Field{m.ids.p + std::size_t{own0} * K, m.peer_ids.p + std::size_t{p0} * K, K * 4U},
                        Field{m.route_w.p + std::size_t{own0} * K, m.peer_route_w.p + std::size_t{p0} * K, K * 4U}});
    };
    std::uint32_t moe_input_rows = 0;  // > 0: a staged MoE input is on the wire (rows of the message)
    // both ranks take the same path (staged_gather and the buffer size are the same on both): the same messages
    const auto begin_moe_input = [&](const Rows& R, std::uint32_t own0, std::uint32_t own_rows, std::uint32_t layer) {
        const std::uint32_t n = std::max(own_rows, Tp - local_of(peer, R.o0));
        const PackFields pf = moe_input_pack(R, own0);
        const std::size_t bytes = std::size_t{n} * pf.stride * 4U;
        auto* send = m.staged_gather && m.pack && n > 0U ? static_cast<std::uint32_t*>(m.options.communicator->send_staging(bytes)) : nullptr;
        if (send == nullptr) {
            start_gather();
            return;
        }
        hipLaunchKernelGGL(pack_rows_kernel, dim3(n), dim3(256), 0, st, pf, send);
        check(hipGetLastError(), "pack rows");
        m.options.communicator->start_staged(st, bytes, dist::Tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), layer, ++exchange_seq});
        moe_input_rows = n;
    };

    // fresh sequence: every layer's attention state cleared (extend keeps it)
    for (std::uint32_t l = 0; P0 == 0U && l < m.options.layers; ++l) {
        auto& L = m.attn[l];
        L.ring.zero(st);
        if (L.kv_cache.p) {
            L.kv_cache.zero(st);
            L.idx_cache.zero(st);
            for (int k = 0; k < 2; ++k) {
                L.carry[k].zero(st);
                L.carry_valid[k].zero(st);
            }
            L.carry_cur = 0;
        }
        if (L.idx.p) L.idx_count.zero(st);
    }
    m.candidates.zero(st);

    // Engram rows of every token (hash of the whole prompt), read from the NVMe: now (overlap off) or on a thread
    // joined before the first Engram layer (overlap: into the pinned buffer)
    std::vector<std::uint8_t> engram_bytes;
    const std::uint8_t* engram_data = nullptr;
    const std::uint32_t engram_cols = m.model.engram.rows_per_token();
    std::vector<std::uint32_t> engram_layers;
    std::vector<ds41::EngramRowRequest> requests;
    std::future<double> engram_job;  // declared after `requests`: destroyed (joined) first
    // STRIX_DS41_ENGRAM_SPLIT_READ (default on, =0 off): the read thread signals once the first Engram layer's rows are
    // in, and the first Engram layer waits only for those (the other layers' rows keep arriving while it runs; same bytes)
    std::shared_future<void> engram_first;
    std::uint32_t engram_first_layer = 0xFFFFFFFFU;
    m.join_engram_ahead();  // the previous run's lookahead: its rows are in the payload cache (or it failed: read below)
    const std::vector<std::uint32_t> engram_next = std::exchange(m.engram_next, {});
    {
        const auto t0 = std::chrono::steady_clock::now();
        if (P0 == 0U) m.hasher->reset();
        if (m.hasher->positions() != P0) throw std::logic_error("deepseek41 prefill: Engram history out of step");
        const std::uint32_t per_token = m.hasher->layers() * engram_cols;
        std::vector<std::uint64_t> ids(std::size_t{T} * per_token), one(per_token);  // [row][layer][column] of this run
        for (std::uint32_t t = 0; t < T; ++t) {
            m.hasher->push(tokens[t], one);
            std::copy(one.begin(), one.end(), ids.begin() + static_cast<std::ptrdiff_t>(std::size_t{t} * per_token));
        }
        for (std::uint32_t li = 0; li < m.model.engram.layer_ids.size(); ++li) {
            if (m.model.engram.layer_ids[li] >= m.options.layers) continue;
            engram_layers.push_back(li);
            for (std::uint32_t t = 0; t < Tn; ++t)  // this rank's rows
                for (std::uint32_t k = 0; k < engram_cols; ++k)
                    requests.push_back({li, ids[std::size_t{global_row[t]} * per_token + li * engram_cols + k]});
        }
        // lookahead: this rank's Engram rows of the next run (its history continues this run's), read after this run's own
        m.engram_ahead_requests.clear();
        if (!engram_next.empty() && !requests.empty() && m.engram_reader->payload_cache()) {  // without a cache nothing is reused
            if (rank_global_rows(split, T, W, me) != global_row) throw std::logic_error("deepseek41 prefill: Engram lookahead row map");
            const auto Tx = static_cast<std::uint32_t>(engram_next.size());
            const bool split_next = m.distributed() && m.options.row_split && Tx >= m.split_min_rows;
            const auto next_rows = rank_global_rows(split_next, Tx, W, me);
            ds41::EngramHasher h = *m.hasher;
            std::vector<std::uint64_t> nids(std::size_t{Tx} * per_token);
            for (std::uint32_t t = 0; t < Tx; ++t) {
                h.push(engram_next[t], one);
                std::copy(one.begin(), one.end(), nids.begin() + static_cast<std::ptrdiff_t>(std::size_t{t} * per_token));
            }
            for (const auto li : engram_layers)
                for (const auto t : next_rows)
                    for (std::uint32_t k = 0; k < engram_cols; ++k)
                        m.engram_ahead_requests.push_back({li, nids[std::size_t{t} * per_token + li * engram_cols + k]});
        }
        static const bool ahead_log = [] { const char* v = std::getenv("STRIX_DS41_ENGRAM_LOOKAHEAD_LOG"); return v && v[0] == '1'; }();
        const auto own_stats = ahead_log ? m.engram_reader->stats() : ds41::EngramReaderStats{};
        const auto start_ahead = [&m, P0, T, own_stats] {  // after this run's own read (the reader serves one caller at a time)
            if (ahead_log) {
                const auto s = m.engram_reader->stats();
                std::fprintf(stderr, "engram lookahead: run at %u, %u rows: own read %llu requests, %llu cache hits, %llu misses, %.3f s busy; next run %zu requests\n",
                             P0, T, static_cast<unsigned long long>(s.requests - own_stats.requests),
                             static_cast<unsigned long long>(s.payload_hits - own_stats.payload_hits),
                             static_cast<unsigned long long>(s.payload_misses - own_stats.payload_misses), s.busy_seconds - own_stats.busy_seconds,
                             m.engram_ahead_requests.size());
            }
            if (m.engram_ahead_requests.empty()) return;
            try {  // no thread: the next run reads its rows itself
                m.engram_ahead = std::thread([&m] {
                    const auto t0 = std::chrono::steady_clock::now();
                    try {
                        m.engram_ahead_out.resize(m.engram_ahead_requests.size() * 264U);
                        m.engram_reader->read(m.engram_ahead_requests, m.engram_ahead_out);
                    } catch (...) {  // a warm-up only: the next run reads what the cache misses
                    }
                    if (ahead_log) std::fprintf(stderr, "engram lookahead: next run's %zu requests read in %.3f s\n", m.engram_ahead_requests.size(), seconds_since(t0));
                });
            } catch (const std::system_error&) {
            }
        };
        const std::size_t bytes = requests.size() * 264U;
        if (m.overlap) {
            if (m.engram_pinned_bytes < bytes) {
                if (m.engram_pinned != nullptr) check(hipHostFree(m.engram_pinned), "engram pinned");
                m.engram_pinned = nullptr;
                m.engram_pinned_bytes = 0;
                check(hipHostMalloc(reinterpret_cast<void**>(&m.engram_pinned), bytes, hipHostMallocDefault), "engram pinned");
                m.engram_pinned_bytes = bytes;
            }
            engram_data = m.engram_pinned;
            if (!requests.empty())
            {
                static const bool split_read = [] { const char* v = std::getenv("STRIX_DS41_ENGRAM_SPLIT_READ"); return !(v && std::strcmp(v, "0") == 0); }();
                // requests are [Engram layer][row][column]: the first layer's block is the first Tn * engram_cols of them
                const std::size_t first = split_read && engram_layers.size() >= 2U ? std::size_t{Tn} * engram_cols : 0U;
                std::promise<void> first_in;
                if (first != 0U) {
                    engram_first = first_in.get_future().share();
                    engram_first_layer = m.model.engram.layer_ids[engram_layers[0]];
                }
                engram_job = std::async(std::launch::async, [&m, &requests, bytes, first, start_ahead, first_in = std::move(first_in)]() mutable {
                    const auto t1 = std::chrono::steady_clock::now();
                    const std::span<const ds41::EngramRowRequest> all(requests);
                    if (first != 0U) {
                        try {
                            m.engram_reader->read(all.first(first), std::span<std::uint8_t>(m.engram_pinned, first * 264U));
                        } catch (...) {
                            first_in.set_exception(std::current_exception());
                            throw;
                        }
                        first_in.set_value();
                    }
                    m.engram_reader->read(all.subspan(first), std::span<std::uint8_t>(m.engram_pinned + first * 264U, bytes - first * 264U));
                    const double s = seconds_since(t1);
                    start_ahead();
                    return s;
                });
            }
        } else {
            engram_bytes.resize(bytes);
            if (!requests.empty()) m.engram_reader->read(requests, engram_bytes);
            engram_data = engram_bytes.data();
            if (!requests.empty()) start_ahead();
        }
        result.engram_seconds = seconds_since(t0);
    }
    const auto join_engram = [&](std::uint32_t layer) {  // overlap: the rows of `layer` (any other value: the whole read) in
        if (!engram_job.valid()) return;
        const auto t0 = std::chrono::steady_clock::now();
        if (layer == engram_first_layer && engram_first.valid()) {  // split read: this layer's block only
            engram_first.get();
            result.engram_wait_seconds += seconds_since(t0);
            return;
        }
        result.engram_seconds += engram_job.get();  // the read's wall time into engram_seconds, the waits into engram_wait_seconds
        result.engram_wait_seconds += seconds_since(t0);
    };
    const auto engram_slice = [&](std::uint32_t layer) -> const std::uint8_t* {  // [token][column] x 264 B of an Engram layer
        const auto& ids = m.model.engram.layer_ids;
        const auto li = static_cast<std::uint32_t>(std::find(ids.begin(), ids.end(), layer) - ids.begin());
        const auto at = static_cast<std::size_t>(std::find(engram_layers.begin(), engram_layers.end(), li) - engram_layers.begin());
        return engram_data + at * std::size_t{Tn} * engram_cols * 264U;
    };

    Dev<std::uint32_t> run_tokens;  // overlap off: allocated per run
    const std::uint32_t* dtokens = nullptr;
    {
        std::vector<std::uint32_t> local_tokens(Tn);
        for (std::uint32_t l = 0; l < Tn; ++l) local_tokens[l] = tokens[global_row[l]];
        if (m.overlap) {
            if (Tn != 0U) m.dtokens.upload(local_tokens, st);
            dtokens = m.dtokens.p;
        } else {
            run_tokens = Dev<std::uint32_t>(Tn);
            run_tokens.upload(local_tokens, st);
            dtokens = run_tokens.p;
        }
    }
    // dense-site routes by (rows, groups): one "expert" per group, slot = row * groups + group (overlap: kept across runs)
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::unique_ptr<DeviceSiteRoutes>> run_routes;
    auto& route_cache = m.overlap ? m.route_cache : run_routes;
    const auto site_routes = [&](std::uint32_t n, std::uint32_t groups) -> const ds41::PrefillRoutesView& {
        auto& slot = route_cache[{n, groups}];
        if (!slot) slot = std::make_unique<DeviceSiteRoutes>(ds41::prefill_site_routes(n, groups, 16U), st);
        return slot->view;
    };
    const auto stream_now = [&] { return m.streams[m.cur].p; };
    const std::size_t stream_row = std::size_t{hc} * H;
    std::int32_t timed_layer = -1;
    auto t_layer = std::chrono::steady_clock::now();
    double comm_at_layer = 0.0, blocked_at_layer = 0.0, attn_s = 0.0, moe_s = 0.0;
    const auto comm_seconds = [&] { return m.distributed() ? m.options.communicator->stats().seconds : 0.0; };
    const auto blocked_seconds = [&] { return m.distributed() ? m.options.communicator->stats().finish_wait : 0.0; };
    const auto sync_time = [&](double& acc, auto&& body) {
        if (!m.options.timing) {
            body();
            return;
        }
        check(hipStreamSynchronize(st), "timing");
        const auto t0 = std::chrono::steady_clock::now();
        body();
        check(hipStreamSynchronize(st), "timing");
        acc += seconds_since(t0);
    };
    const auto close_layer = [&] {
        if (!m.options.timing || timed_layer < 0) return;
        check(hipStreamSynchronize(st), "layer end");
        result.layers[static_cast<std::size_t>(timed_layer)] = {seconds_since(t_layer), attn_s, moe_s, comm_seconds() - comm_at_layer,
                                                                 blocked_seconds() - blocked_at_layer};
    };
    auto t_head = t_run;

    std::vector<std::uint32_t> trace_router;  // options.trace: this layer's router ids until the layer's end
    const bool hc_fuse = prefill_hc_fuse();
    bool hc_mixed = false;  // hc_fuse: the next site's mixes came with the expansion before it
    const bool hc_collapse = hc_fuse && prefill_hc_collapse();
    bool hc_collapsed = false;  // hc_collapse: the next site's collapsed rows are in x (written with the expansion)
    // the collapse's mixes of a site from row r0 (attn: the previous layer's ffn mixes, identity before the first)
    const auto collapse_mixes = [&](const std::string& site, std::uint32_t r0) -> const ds41::HcMix* {
        return site == "attn" ? (m.identity_pre ? nullptr : m.mix_ffn.p + r0) : m.mix_attn.p + r0;
    };
    for (const auto& s : m.plan.steps) {
        const auto layer = static_cast<std::uint32_t>(std::max(s.layer, 0));
        if (defer && (s.layer < 0 ? s.op == Op::head || s.op == Op::argmax : layer >= first_decoder)) {
            // deferred decoder: only a compressed-KV source's attention input and its compressor run (no head)
            const bool source = s.layer >= 0 && m.roles[layer].owns_kv;
            if (!source || !(s.op == Op::attention || ((s.op == Op::hc_mix || s.op == Op::hc_pre || s.op == Op::rms_norm) && s.site == "attn")))
                continue;
        }
        if (s.layer >= 0 && s.layer != cut_layer) {  // a layer starts: hc / norms / q_a / kv (and Engram)
            if (cut_layer >= 0) cut(PrefillUnitLevel::layer, s.layer, "pre");
            else if (m.task != nullptr) m.task->layer = s.layer, m.task->kind = "setup+pre";
            cut_layer = s.layer;
        }
        if (s.op == Op::head) cut(PrefillUnitLevel::layer, -1, "head");
        if (s.op == Op::hc_mix && s.site == "ffn") cut(PrefillUnitLevel::phase, s.layer, "ffn");  // ffn hc / norm, router, shared expert, MoE input
        const std::string p = "blk." + std::to_string(layer) + ".";
        const Rows R0 = s.layer >= 0 ? rows[layer] : Rows{};   // global rows
        const Rows L0 = s.layer >= 0 ? lrows[layer] : Rows{};  // this rank's rows
        const std::uint32_t o0 = L0.o0, no = Tn - o0;  // this rank's output rows of the layer
        if (s.layer >= 0 && s.layer != timed_layer) {
            close_layer();
            timed_layer = s.layer;
            t_layer = std::chrono::steady_clock::now();
            comm_at_layer = comm_seconds();
            blocked_at_layer = blocked_seconds();
            attn_s = moe_s = 0.0;
        }
        if (s.op == Op::head) {
            close_layer();
            t_head = std::chrono::steady_clock::now();
        }
        switch (s.op) {
        case Op::embed:
            if (Tn != 0U) ds41::enqueue_embedding(m.devh("token_embd.weight"), dtokens, Tn, H, stream_now(), st);  // row_split: a rank may own no row
            m.identity_pre = true;
            break;
        case Op::engram: {  // encoder layers only: every row
            join_engram(layer);
            const std::uint8_t* erow = engram_slice(layer);
            // pieces of E rows (C; a task: its engram_rows, every kernel here is per row: same bits); task units: one per
            // piece (padded to the rank with more pieces)
            const std::uint32_t E = m.task != nullptr && m.task->opt.engram_rows != 0U ? std::min(m.task->opt.engram_rows, C) : C;
            std::uint32_t engram_cut = 0;
            const std::uint32_t engram_most = (std::max(Tn, Tp) + E - 1U) / E;
            for (std::uint32_t r0 = 0; r0 < Tn; r0 += E) {
                const std::uint32_t n = std::min(E, Tn - r0);
                cut(PrefillUnitLevel::finest, s.layer, "engram");
                ++engram_cut;
                check(hipMemcpyAsync(m.erows.p, erow + std::size_t{r0} * engram_cols * 264U, std::size_t{n} * engram_cols * 264U,
                                     hipMemcpyHostToDevice, st), "engram rows");
                ds41::enqueue_engram_rows_bf16(m.erows.p, n * engram_cols, m.emb.p, st);
                ds41::F16RowsView v;
                v.weight = m.devh(p + "engram_kv.weight");
                v.width = engram_cols * 256U;
                v.count = H * (hc + 1U);
                v.x = m.emb.p;
                v.rows = n;
                v.x_stride = v.width;
                v.out_stride = v.count;
                v.out = m.ekv.p;
                ds41::enqueue_engram_f16_rows(v, st);
                ds41::enqueue_engram_inject(stream_now() + std::size_t{r0} * stream_row, m.ekv.p, m.devf(p + "engram_q_norm.weight"),
                                            m.devf(p + "engram_k_norm.weight"), n, H, 1e-20F, st);
            }
            for (; engram_cut < engram_most; ++engram_cut) cut(PrefillUnitLevel::finest, s.layer, "engram.pad");
            if (!m.overlap) check(hipStreamSynchronize(st), "engram");  // overlap: pinned rows, alive until the run's end
            break;
        }
        case Op::hc_mix: {
            if (no == 0U) break;  // row_split: none of this layer's output rows is this rank's
            if (hc_mixed) {  // computed with the expansion before it
                hc_mixed = false;
                break;
            }
            ds41::HcMixView v;
            v.stream = stream_now() + std::size_t{o0} * stream_row;
            v.fn = m.devh(s.weights[0]);
            v.base = m.devf(s.weights[1]);
            v.scale = m.devf(s.weights[2]);
            v.rows = no;
            v.hidden = H;
            v.iters = c.hc_sinkhorn_iters;
            v.hc_eps = static_cast<float>(c.hc_eps);
            v.norm_eps = 1e-20F;
            v.out = (s.site == "attn" ? m.mix_attn.p : m.mix_ffn.p) + o0;
            v.partial = m.mix_partial.p;
            ds41::enqueue_hc_mixes(v, st);
            break;
        }
        case Op::hc_pre: {
            const std::uint32_t r0 = s.site == "attn" ? L0.x0 : o0;
            if (r0 == Tn || hc_fuse) break;  // hc_fuse: read by the site's RMSNorm
            ds41::enqueue_hc_collapse(stream_now() + std::size_t{r0} * stream_row, collapse_mixes(s.site, r0), Tn - r0, H,
                                      m.x.p + std::size_t{r0} * H, st);
            break;
        }
        case Op::rms_norm: {
            // x in place (+ its A8 for q_a / kv / the experts); qr in place + its A8 for q_b
            const bool q = s.site == "q";
            const std::uint32_t r0 = s.site == "attn" ? L0.x0 : o0, n = q ? QA : H;
            if (r0 == Tn) break;
            ds41::NormView v;
            v.x = (q ? m.qr.p : m.x.p) + std::size_t{r0} * n;
            v.x_stride = n;
            v.weight = m.devf(s.weights[0]);
            v.rows = Tn - r0;
            v.n = n;
            v.eps = 1e-20F;
            v.out = const_cast<float*>(v.x);
            v.out_stride = n;
            if (!q && hc_collapsed) hc_collapsed = false;  // x holds the collapse, as hc_pre would have written it
            else if (!q && hc_fuse) {  // the collapse of the stream, as hc_pre
                v.x = nullptr;
                v.x_stride = 0;
                v.hc_stream = stream_now() + std::size_t{r0} * stream_row;
                v.mixes = collapse_mixes(s.site, r0);
            }
            v.codes = (q ? m.qcodes.p : m.xcodes.p) + std::size_t{r0} * n;
            v.deltas = (q ? m.qdeltas.p : m.xdeltas.p) + std::size_t{r0} * n / 32U;
            v.sums = (q ? m.qsums.p : m.xsums.p) + std::size_t{r0} * n / 32U;
            v.a8_stride = n;
            if (m.fast_at(q ? 4U : 3U, layer)) {
                auto& b = q ? m.qr16 : m.x16;
                if (b.n == 0U) b = Dev<std::uint16_t>(q ? m.qr.n : m.x.n);
                v.f16 = b.p + std::size_t{r0} * n;
                v.f16_stride = n;
            }
            ds41::enqueue_norm(v, st);
            break;
        }
        case Op::q_a: case Op::kv: {
            const std::uint32_t r0 = s.op == Op::q_a ? o0 : L0.k0;
            if (r0 == Tn) break;
            const bool fp16 = m.fp16_at(s.op == Op::q_a ? 1U : 2U, layer, Tn - r0);
            if (m.options.fp16_fast && fp16) {
                // q_a || kv: one GEMM over rows [min(o0, k0), Tn) when both are FP16 here (run at q_a, skipped at kv)
                const bool joint = o0 < Tn && L0.k0 < Tn && m.fp16_at(1U, layer, Tn - o0) && m.fp16_at(2U, layer, Tn - L0.k0);
                if (joint && s.op == Op::kv) break;
                ds41::Q8A8DensePrefillView v[2];
                unsigned count = 0;
                for (const Op op : {Op::q_a, Op::kv}) {
                    if (op != s.op && !joint) continue;
                    auto& d = v[count++];
                    const std::uint32_t d0 = op == Op::q_a ? o0 : L0.k0;
                    d.weight = m.dev(op == Op::q_a ? s.weights[0] : p + "attn_kv.weight");
                    d.width = H;
                    d.count = op == Op::q_a ? QA : D;
                    d.rows = Tn - d0;
                    d.x_stride = H;
                    d.out = op == Op::q_a ? m.qr.p + std::size_t{d0} * QA : m.kv.p + std::size_t{d0} * D;
                    d.out_stride = d.count;
                }
                const std::uint32_t first = joint ? std::min(o0, L0.k0) : r0;
                m.fast(v, count, Tn - first, m.x16.p + std::size_t{first} * H, H, nullptr, st);
                break;
            }
            if (fp16 || (s.op == Op::q_a ? m.dense_q_a : m.dense_kv)) {
                ds41::Q8A8DensePrefillView v;
                v.weight = m.dev(s.weights[0]);
                v.width = H;
                v.count = s.op == Op::q_a ? QA : D;
                v.codes = m.xcodes.p + std::size_t{r0} * H;
                v.deltas = m.xdeltas.p + std::size_t{r0} * H / 32U;
                v.rows = Tn - r0;
                v.x_stride = H;
                v.out = s.op == Op::q_a ? m.qr.p + std::size_t{r0} * QA : m.kv.p + std::size_t{r0} * D;
                v.out_stride = v.count;
                m.dense(v, m.x.p + std::size_t{r0} * H, st, fp16);
                break;
            }
            ds41::Q8A8PrefillView v;
            v.weight = m.dev(s.weights[0]);
            v.width = H;
            v.groups = 1;
            v.per_group = s.op == Op::q_a ? QA : D;
            v.routes = site_routes(Tn - r0, 1U);
            v.codes = m.xcodes.p + std::size_t{r0} * H;
            v.deltas = m.xdeltas.p + std::size_t{r0} * H / 32U;
            v.out = (s.op == Op::q_a ? m.qr.p + std::size_t{r0} * QA : m.kv.p + std::size_t{r0} * D);
            ds41::enqueue_q8_0_a8_prefill(v, st);
            break;
        }
        case Op::q_b: case Op::window_write: case Op::compress: case Op::index: case Op::o_a: case Op::o_b:
            break;  // inside the attention step
        case Op::attention: {
            cut(PrefillUnitLevel::phase, s.layer, "attn");  // projections, window K/V + compressor
            sync_time(attn_s, [&] {
                const auto& r = m.roles[layer];
                auto& L = m.attn[layer];
                // TP: this rank's heads [head0, head0 + heads) and o-groups [group0, group0 + groups)
                const bool tp = m.distributed() && m.options.tp_attention && !split;
                const std::uint32_t heads = tp ? c.num_attention_heads / 2U : c.num_attention_heads, head0 = tp ? m.options.rank * heads : 0U;
                const std::uint32_t groups = tp ? c.o_groups / 2U : c.o_groups, group0 = tp ? m.options.rank * groups : 0U;
                const std::uint32_t QL = heads * c.head_dim, OLL = groups * c.o_lora_rank;
                const auto q8_rows = [](const std::uint8_t* w, std::size_t first, std::uint32_t width) {
                    return w + first * (width / 32U) * 34U;  // Q8_0 rows: 34 bytes per 32 weights
                };
                auto t_part = std::chrono::steady_clock::now();
                const auto mark_part = [&](std::size_t k) {  // timing: the work since the previous mark into part k
                    if (!m.options.timing) return;
                    check(hipStreamSynchronize(st), "attention part");
                    for (auto& ln : m.lanes) check(hipStreamSynchronize(ln.stream), "attention part");
                    result.attention_parts[k] += seconds_since(t_part);
                    t_part = std::chrono::steady_clock::now();
                };
                const auto f16 = [&](const std::string& w, const float* in, std::uint32_t width, std::uint32_t count, float* out, std::uint32_t r0) {
                    if (r0 == Tn) return;
                    ds41::F16RowsView v;
                    v.weight = m.devh(w);
                    v.width = width;
                    v.count = count;
                    v.x = in + std::size_t{r0} * width;
                    v.rows = Tn - r0;
                    v.x_stride = width;
                    v.out_stride = count;
                    v.out = out + std::size_t{r0} * count;
                    ds41::enqueue_f16_rows(v, st);
                };
                if (r.owns_kv) {
                    f16(p + "attn_compressor_kv.weight", m.x.p, H, D, m.ckv.p, L0.x0);
                    if (r.ratio == 2U) f16(p + "attn_compressor_gate.weight", m.x.p, H, D, m.cscore.p, L0.x0);
                }
                // STRIX_DS41_PREFILL_FINAL_KV: the gather starts after this rank's final entries (below)
                const bool final_kv = split && prefill_final_kv(layer);
                if (split && !final_kv) start_gather();  // window K/V and compressor inputs are ready; the indexer runs while they cross
                const auto index_projections = [&] {
                    if (!r.owns_index) return;
                    // STRIX_DS41_IDX_DIRECT: every window of the run selects all its positions (below): no projections
                    if (attn::hip::index_direct() && (std::uint64_t{P0} + T) / m.roles[static_cast<std::size_t>(r.kv_source)].ratio <= attn::kTopK) return;
                    f16(p + "indexer.attn_q_b.weight", m.qr.p, QA, attn::kIdxHeads * attn::kIdxDim, m.iq.p, o0);
                    f16(p + "indexer.proj.weight", m.x.p, H, attn::kIdxHeads, m.iw.p, o0);
                };
                if (!final_kv) index_projections();
                // ring and compressor: global window w0 (positions from w0), K/V rows from the window's owner
                const auto ring_write = [&](const float* kv, std::uint32_t w0, std::uint32_t n, const attn::RopeRow* rope) {
                    attn::hip::WindowView v{};
                    v.kv = kv;
                    v.norm = m.devf(p + "attn_kv_a_norm.weight");
                    v.rope = rope;
                    v.pos = m.pos64.p + w0;
                    v.rows = n;
                    v.capacity = m.prefill_capacity;
                    v.ring = m.prefill_ring.p;
                    attn::hip::enqueue_window_write(v, st);
                };
                const auto compress = [&](const float* ckv, const float* cscore, std::uint32_t w0, std::uint32_t n) {
                    attn::hip::CompressView v{};
                    v.ratio = r.ratio;
                    v.rows = n;
                    v.kv = ckv;
                    v.score = cscore;
                    v.norm = m.devf(p + "attn_compressor_norm.weight");
                    v.wk = m.devh(p + "indexer.attn_k.weight");
                    v.knorm = m.devf(p + "indexer.k_norm.weight");
                    v.rope = (r.ratio == 2U ? m.rope_group.p : m.rope_comp.p) + w0;
                    v.pos = m.pos64.p + w0;
                    v.state_in = L.carry[L.carry_cur].p;
                    v.state_valid_in = L.carry_valid[L.carry_cur].p;
                    v.state_out = L.carry[1 - L.carry_cur].p;
                    v.state_valid_out = L.carry_valid[1 - L.carry_cur].p;
                    v.kv_cache = L.kv_cache.p;
                    v.idx_cache = L.idx_cache.p;
                    attn::hip::enqueue_compress(v, st);
                    L.carry_cur = 1 - L.carry_cur;
                };
                // q = q_b(qr) of this rank's chunk at r0 (n rows) into m.q
                const auto q_b_chunk = [&](std::uint32_t r0, std::uint32_t n) {
                    const bool fp16_q_b = m.fp16_at(4U, layer, n);
                    if (fp16_q_b && m.options.fp16_fast) {
                        ds41::Q8A8DensePrefillView v;
                        v.weight = q8_rows(m.dev(p + "attn_q_b.weight"), std::size_t{head0} * c.head_dim, QA);
                        v.width = QA;
                        v.count = QL;
                        v.rows = n;
                        v.x_stride = QA;
                        v.out = m.q.p;
                        v.out_stride = QL;
                        m.fast(&v, 1U, n, m.qr16.p + std::size_t{r0} * QA, QA, nullptr, st);
                    } else if (fp16_q_b || (m.dense_q_b && n >= m.dense_min_rows)) {
                        ds41::Q8A8DensePrefillView v;
                        v.weight = q8_rows(m.dev(p + "attn_q_b.weight"), std::size_t{head0} * c.head_dim, QA);
                        v.width = QA;
                        v.count = QL;
                        v.codes = m.qcodes.p + std::size_t{r0} * QA;
                        v.deltas = m.qdeltas.p + std::size_t{r0} * QA / 32U;
                        v.rows = n;
                        v.x_stride = QA;
                        v.out = m.q.p;
                        v.out_stride = QL;
                        m.dense(v, m.qr.p + std::size_t{r0} * QA, st, fp16_q_b);
                    } else {
                        ds41::Q8A8PrefillView v;
                        v.weight = q8_rows(m.dev(p + "attn_q_b.weight"), std::size_t{head0} * c.head_dim, QA);
                        v.width = QA;
                        v.groups = 1;
                        v.per_group = QL;
                        v.routes = site_routes(n, 1U);
                        v.codes = m.qcodes.p + std::size_t{r0} * QA;
                        v.deltas = m.qdeltas.p + std::size_t{r0} * QA / 32U;
                        v.out = m.q.p;
                        ds41::enqueue_q8_0_a8_prefill(v, st);
                    }
                };
                // m.early_q_b: the first chunk's q_b already ran while the final entries crossed (below)
                bool first_q_b_done = false;
                // Every window K/V of the layer (rows from k0) and every compressed entry (rows from x0: below k0 only a KV
                // source in the cone, layer 20, whose entries every decoder layer reads) in row order on the main stream;
                // a row's index scores, selection and attention read only entries of rows up to its own, so the windows
                // after this are independent of each other.
                if (final_kv) {
                    // This rank's windows only, in row order: the ring entries (rows from k0) and the compressed entries
                    // (rows from x0). Windows start at even positions (P0 % window_rows == 0), so no ratio-2 pair spans
                    // two owners; the pending row after the run is the carry, sent by the last window's owner.
                    const attn::RopeRow* own_rope = r.rope == attn::RopeKind::window ? m.rope_window.p : m.rope_comp.p;
                    for (std::uint32_t w0 = R0.x0; w0 < T; w0 += W) {
                        const std::uint32_t n = std::min(W, T - w0), k = w0 / W;
                        if (window_owner(k) != me) continue;
                        const std::size_t at = std::size_t{owner_window(k)} * W * D;
                        if (w0 >= R0.k0) ring_write(m.kv.p + at, w0, n, own_rope + w0);
                        if (r.owns_kv) compress(m.ckv.p + at, m.cscore.p + at, w0, n);
                    }
                    const std::uint32_t lk0 = L0.k0, lx0 = L0.x0, pk0 = local_of(peer, R0.k0), px0 = local_of(peer, R0.x0);
                    const auto comp_of = [&](std::uint32_t rows) { return r.owns_kv ? rows / r.ratio : 0U; };
                    const std::uint32_t carry_words = r.owns_kv && r.ratio == 2U ? 2U * D + 1U : 0U;
                    const std::uint32_t own_ring = Tn > lk0 ? Tn - lk0 : 0U, peer_ring = Tp > pk0 ? Tp - pk0 : 0U;
                    const std::uint32_t own_comp = comp_of(Tn > lx0 ? Tn - lx0 : 0U), peer_comp = comp_of(Tp > px0 ? Tp - px0 : 0U);
                    const auto words = [&](std::uint32_t ring, std::uint32_t comp) {
                        return ring * final_ring_words + comp * (final_kv_words + final_idx_words) + carry_words;
                    };
                    const std::uint32_t own_words = words(own_ring, own_comp), peer_words = words(peer_ring, peer_comp);
                    FinalEntries fe;
                    fe.ring = reinterpret_cast<std::uint32_t*>(m.prefill_ring.p);
                    fe.capacity = m.prefill_capacity;
                    fe.kv_cache = reinterpret_cast<std::uint32_t*>(L.kv_cache.p);
                    fe.idx_cache = reinterpret_cast<std::uint32_t*>(L.idx_cache.p);
                    fe.ratio = r.owns_kv ? r.ratio : 1U;
                    fe.ring_pos = m.lpos64.p + lk0;
                    fe.ring_rows = own_ring;
                    fe.comp_pos = m.lpos64.p + lx0;
                    fe.comp_rows = own_comp;
                    // staged (m.staged_gather): packed straight into the pinned send buffer, unpacked from the receive buffer,
                    // so the message crosses while the indexer projections run instead of after them
                    const std::size_t entry_bytes = std::size_t{std::max(own_words, peer_words)} * 4U;
                    auto* staged_send = m.staged_gather && entry_bytes > 0U
                                            ? static_cast<std::uint32_t*>(m.options.communicator->send_staging(entry_bytes)) : nullptr;
                    std::uint32_t* const send_words = staged_send ? staged_send : m.pack_send.p;
                    fe.packed = send_words;
                    if (own_ring + own_comp) {
                        final_entries_kernel<true><<<own_ring + own_comp, 128, 0, st>>>(fe);
                        check(hipGetLastError(), "final entries pack");
                    }
                    const std::size_t carry_at = std::size_t{own_words} - carry_words;
                    if (carry_words && last_local) {  // after this rank's last compress: the run's pending row (or none)
                        check(hipMemcpyAsync(send_words + carry_at, L.carry[L.carry_cur].p, std::size_t{2U * D} * 4U, hipMemcpyDefault, st),
                              "carry");
                        check(hipMemcpyAsync(send_words + carry_at + 2U * D, L.carry_valid[L.carry_cur].p, 4U, hipMemcpyDefault, st),
                              "carry");
                    }
                    const std::uint32_t* recv_words = m.pack_recv.p;
                    if (staged_send) {
                        m.options.communicator->start_staged(st, entry_bytes,
                                                             dist::Tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), layer, ++exchange_seq});
                        index_projections();  // while the entries cross
                        // the first chunk's q_b too, unless the task parks at that chunk's boundary (m.q is shared)
                        if (m.early_q_b && o0 < Tn && (m.task == nullptr || m.task->runs_through(PrefillUnitLevel::finest, 1U))) {
                            mark_part(0);  // timing: the q_b into its own part, as in the loop
                            q_b_chunk(o0, std::min(C, Tn - o0));
                            mark_part(2);
                            first_q_b_done = true;
                        }
                        recv_words = static_cast<const std::uint32_t*>(m.options.communicator->finish_staged(st));
                    } else {
                        start_gather();
                        index_projections();  // while the entries cross
                        exchange_rows(m.pack_send.p, own_words, m.pack_recv.p, peer_words, 4U, layer);
                        join_gather();
                    }
                    fe.ring_pos = m.ppos64.p + pk0;
                    fe.ring_rows = peer_ring;
                    fe.comp_pos = m.ppos64.p + px0;
                    fe.comp_rows = peer_comp;
                    fe.packed = const_cast<std::uint32_t*>(recv_words);
                    if (peer_ring + peer_comp) {
                        final_entries_kernel<false><<<peer_ring + peer_comp, 128, 0, st>>>(fe);
                        check(hipGetLastError(), "final entries unpack");
                    }
                    if (carry_words && !last_local) {
                        const std::size_t peer_carry = std::size_t{peer_words} - carry_words;
                        check(hipMemcpyAsync(L.carry[1 - L.carry_cur].p, recv_words + peer_carry, std::size_t{2U * D} * 4U,
                                             hipMemcpyDefault, st),
                              "carry");
                        check(hipMemcpyAsync(L.carry_valid[1 - L.carry_cur].p, recv_words + peer_carry + 2U * D, 4U, hipMemcpyDefault, st),
                              "carry");
                        L.carry_cur = 1 - L.carry_cur;
                    }
                    if (staged_send) m.options.communicator->release_staged(st);
                } else if (split) {  // the peer's window K/V (rows from k0) and compressor inputs (rows from x0)
                    const std::uint32_t pk0 = local_of(peer, R0.k0), px0 = local_of(peer, R0.x0);
                    const Field kv{m.kv.p + std::size_t{L0.k0} * D, m.peer_kv.p + std::size_t{pk0} * D, D * 4U};
                    const Field ckv{m.ckv.p + std::size_t{L0.x0} * D, m.peer_ckv.p + std::size_t{px0} * D, D * 4U};
                    const Field cscore{m.cscore.p + std::size_t{L0.x0} * D, m.peer_cscore.p + std::size_t{px0} * D, D * 4U};
                    if (!r.owns_kv) {
                        exchange_fields({kv}, Tn - L0.k0, Tp - pk0, layer);
                    } else if (R0.x0 == R0.k0) {  // the same rows for all three (every encoder layer)
                        if (r.ratio == 2U) exchange_fields({kv, ckv, cscore}, Tn - L0.x0, Tp - px0, layer);
                        else exchange_fields({kv, ckv}, Tn - L0.x0, Tp - px0, layer);
                    } else {  // a KV source in the cone (layer 20): the window K/V of fewer rows than the compressor inputs
                        exchange_fields({kv}, Tn - L0.k0, Tp - pk0, layer);
                        if (r.ratio == 2U) exchange_fields({ckv, cscore}, Tn - L0.x0, Tp - px0, layer);
                        else exchange_fields({ckv}, Tn - L0.x0, Tp - px0, layer);
                    }
                    join_gather();
                }
                if (P0 != 0U) {  // extend: the layer's last window positions before the run, into their slots of the run's ring
                    const std::uint32_t first = P0 > attn::kWindow ? P0 - attn::kWindow : 0U;
                    ring_copy_kernel<<<P0 - first, 160, 0, st>>>(reinterpret_cast<const std::uint32_t*>(L.ring.p), AttentionHost::ring_capacity,
                                                                 reinterpret_cast<std::uint32_t*>(m.prefill_ring.p), m.prefill_capacity, first);
                    check(hipGetLastError(), "ring restore");
                }
                mark_part(0);
                const attn::RopeRow* rope_rows = r.rope == attn::RopeKind::window ? m.rope_window.p : m.rope_comp.p;
                const attn::RopeRow* lrope_rows = r.rope == attn::RopeKind::window ? m.lrope_window.p : m.lrope_comp.p;
                for (std::uint32_t w0 = R0.x0; !final_kv && w0 < T; w0 += W) {
                    const std::uint32_t n = std::min(W, T - w0), k = w0 / W;
                    const bool mine = window_owner(k) == me;
                    const std::size_t at = std::size_t{owner_window(k)} * W * D;
                    if (w0 >= R0.k0) ring_write((mine ? m.kv.p : m.peer_kv.p) + at, w0, n, rope_rows + w0);
                    if (r.owns_kv) compress((mine ? m.ckv.p : m.peer_ckv.p) + at, (mine ? m.cscore.p : m.peer_cscore.p) + at, w0, n);
                }
                mark_part(1);
                if (defer && layer >= first_decoder) return;  // deferred: the compressed entries are in; no rows, no ring save
                // TP: the peer's K-half of the chunk at `pending` (if any), then y = h0 + h1 of its rows
                std::uint32_t pending = T;
                float* staged_half = nullptr;
                const auto close_halves = [&] {
                    if (pending == T) return;
                    const std::uint32_t n = std::min(C, Tn - pending);
                    const float* peer = m.y_other.p;
                    if (staged_half) peer = static_cast<const float*>(m.options.communicator->finish_staged(st));
                    else m.options.communicator->finish(st, m.y_other.p);
                    const float* own = staged_half ? staged_half : m.y_own.p;
                    const float* h0 = m.options.rank == 0U ? own : peer;
                    const float* h1 = m.options.rank == 0U ? peer : own;
                    const std::uint32_t count = n * H;
                    add_halves_kernel<<<(count + 255U) / 256U, 256, 0, st>>>(h0, h1, m.y.p + std::size_t{pending} * H, count);
                    check(hipGetLastError(), "add halves");
                    if (staged_half) m.options.communicator->release_staged(st);
                    staged_half = nullptr;
                    pending = T;
                };
                // task units: one per chunk (TP form: none, a chunk's wo_b halves are in flight across chunks); the
                // rank with fewer chunks pads with empty units
                const auto chunk_count = [&](std::uint32_t rows, std::uint32_t first) { return rows > first ? (rows - first + C - 1U) / C : 0U; };
                std::uint32_t chunks_cut = 0, chunks_most = chunk_count(Tn, o0);
                if (split) chunks_most = std::max(chunks_most, chunk_count(Tp, local_of(peer, R0.o0)));
                for (std::uint32_t r0 = o0; r0 < Tn; r0 += C) {  // this rank's rows
                    const std::uint32_t n = std::min(C, Tn - r0);
                    if (!tp) {
                        cut(PrefillUnitLevel::finest, s.layer, "attn.chunk");
                        ++chunks_cut;
                    }
                    if (!first_q_b_done || r0 != o0) q_b_chunk(r0, n);
                    mark_part(2);
                    check(hipEventRecord(m.chunk_ready, st), "chunk ready");
                    for (auto& ln : m.lanes) check(hipStreamWaitEvent(ln.stream, m.chunk_ready, 0), "lane wait");
                    // STRIX_DS41_PREFILL_O_A8_FUSED (default on, =0 off): the exact attention combine writes o's A8 codes,
                    // deltas and sums (enqueue_quantize_a8's bits, from the same values it stores) instead of a separate pass
                    const bool fast_o_a = m.options.fp16_fast && m.fp16_at(8U, layer, n);
                    const bool o_a8_fused = !fast_o_a && !m.bf16_at(layer) && !m.fast_attn_at(layer) && prefill_o_a8_fused();
                    for (std::uint32_t w0 = r0; w0 < r0 + n; w0 += W) {
                        // local window at w0 = global window wi (positions from wi * W)
                        const std::uint32_t Rw = std::min(W, r0 + n - w0), wi = global_window(w0 / W);
                        auto& ln = m.lanes[(w0 / W) % m.lanes.size()];
                        hipStream_t ls = ln.stream;
                        const std::uint64_t* pos = m.lpos64.p + w0;
                        const attn::RopeRow* rope = lrope_rows + w0;
                        if (r.owns_index) {
                            const std::uint32_t ratio = m.roles[static_cast<std::size_t>(r.kv_source)].ratio;
                            const std::uint32_t count = (P0 + wi * W + Rw) / ratio;  // the last row's (pos + 1) / ratio
                            if (count == 0U) {
                                check(hipMemsetAsync(L.idx_count.p + w0, 0, std::size_t{Rw} * 4U, ls), "empty top-k");
                            } else {
                                attn::hip::IndexView iv{};
                                iv.rows = Rw;
                                iv.ratio = ratio;
                                iv.max_count = count;
                                iv.q = m.iq.p + std::size_t{w0} * attn::kIdxHeads * attn::kIdxDim;
                                iv.w = m.iw.p + std::size_t{w0} * attn::kIdxHeads;
                                iv.rope = rope;
                                iv.pos = pos;
                                iv.idx_cache = m.attn[static_cast<std::size_t>(r.kv_source)].idx_cache.p;
                                iv.scores = ln.index_scores.p;
                                iv.qprep = ln.index_qprep.p;
                                attn::hip::SelectView sv{};
                                sv.rows = Rw;
                                sv.ratio = ratio;
                                sv.max_count = count;
                                sv.topk = attn::kTopK;
                                sv.cand_blocks = attn::kCandBlocks;
                                sv.block_size = attn::kCandBlockSize;
                                sv.pool = r.candidate_source ? attn::Pool::candidate_source
                                          : r.candidate_user ? attn::Pool::candidate_user : attn::Pool::plain;
                                sv.pos = pos;
                                sv.scores = ln.index_scores.p;
                                sv.candidates = m.candidates.p + std::size_t{wi} * m.cand_window_bytes;
                                sv.keys = ln.select_keys.p;
                                sv.idx = L.idx.p + std::size_t{w0} * attn::kTopK;
                                sv.idx_count = L.idx_count.p + w0;
                                if (count <= attn::kTopK && attn::hip::index_direct()) {  // every position selected
                                    attn::hip::enqueue_select_direct(sv, ls);
                                } else {
                                    if (sv.pool == attn::Pool::candidate_user && count > attn::kCandBlocks * attn::kCandBlockSize && attn::hip::index_candidate_filter()) {
                                        iv.candidates = sv.candidates;
                                        iv.block_size = sv.block_size;
                                        sv.compact = true;
                                    }
                                    attn::hip::enqueue_index_scores(iv, ls);
                                    attn::hip::enqueue_select(sv, ls);
                                }
                            }
                        }
                        attn::hip::AttentionView v{};
                        v.rows = Rw;
                        v.capacity = m.prefill_capacity;
                        v.topk = attn::kTopK;
                        v.heads = heads;
                        v.q = m.q.p + std::size_t{w0 - r0} * QL;
                        v.sink = m.devf(p + "attn_sinks.weight") + head0;
                        v.rope = rope;
                        v.pos = pos;
                        v.ring = m.prefill_ring.p;
                        if (r.ratio != 0U) {
                            const auto& src = m.attn[static_cast<std::size_t>(r.index_source)];
                            v.kv_cache = m.attn[static_cast<std::size_t>(r.kv_source)].kv_cache.p;
                            v.idx = src.idx.p + std::size_t{w0} * attn::kTopK;
                            v.idx_count = src.idx_count.p + w0;
                        }
                        v.partial = ln.attn_partial.p;
                        v.tickets = ln.attn_tickets.p;
                        v.o = m.o.p + std::size_t{w0 - r0} * QL;
                        if (o_a8_fused) {
                            v.codes = m.ocodes.p + std::size_t{w0 - r0} * QL;
                            v.deltas = m.odeltas.p + std::size_t{w0 - r0} * QL / 32U;
                            v.sums = m.osums.p + std::size_t{w0 - r0} * QL / 32U;
                        }
                        if(m.bf16_at(layer)) attn::hip::enqueue_attention_bf16_prefill(v,ls);
                        else if (m.fast_attn_at(layer)) attn::hip::enqueue_attention_f16wmma_prefill(v, ls);
                        else attn::hip::enqueue_attention(v, ls);
                    }
                    close_halves();  // the previous chunk's wo_b halves crossed meanwhile; y_own is free again
                    for (auto& ln : m.lanes) {
                        check(hipEventRecord(ln.done, ln.stream), "lane done");
                        check(hipStreamWaitEvent(st, ln.done, 0), "chunk join");
                    }
                    mark_part(3);
                    if (!fast_o_a && !o_a8_fused) ds41::enqueue_quantize_a8(m.o.p, QL, n, m.ocodes.p, m.odeltas.p, m.osums.p, st);
                    {
                        ds41::Q8A8PrefillView v;
                        v.width = QH / c.o_groups;
                        v.weight = q8_rows(m.dev(p + "attn_output_a.weight"), std::size_t{group0} * c.o_lora_rank, v.width);
                        v.groups = groups;
                        v.per_group = c.o_lora_rank;
                        v.routes = site_routes(n, groups);
                        v.codes = m.ocodes.p;
                        v.deltas = m.odeltas.p;
                        v.out = m.o_low.p;
                        if (fast_o_a) {
                            // the o-groups: one FP16 conversion of the row, one GEMM per group, one finish
                            ds41::Q8A8DensePrefillView d;
                            d.weight = v.weight;
                            d.width = v.width;
                            d.count = c.o_lora_rank;
                            d.rows = n;
                            d.x_stride = QL;
                            d.out = m.o_low.p;
                            d.out_stride = OLL;
                            m.fast(&d, 1U, n, nullptr, 0U, m.o.p, st, groups);
                        } else if(m.fp16_at(8U, layer, n)) {
                            for(unsigned g=0;g<groups;++g){
                                ds41::Q8A8DensePrefillView d;
                                d.weight=q8_rows(v.weight,std::size_t(g)*c.o_lora_rank,v.width);
                                d.width=v.width;d.count=c.o_lora_rank;d.rows=n;d.x_stride=QL;
                                d.codes=m.ocodes.p+g*v.width;d.deltas=m.odeltas.p+g*v.width/32U;
                                d.out=m.o_low.p+g*c.o_lora_rank;d.out_stride=OLL;
                                m.dense(d,m.o.p+g*v.width,st,true);
                            }
                        } else if (m.dense_o_a && n >= m.dense_min_rows) {
                            // one dense call per o-group (the same group products as the grouped batch kernel)
                            for (unsigned g = 0; g < groups; ++g) {
                                ds41::Q8A8DensePrefillView d;
                                d.weight = q8_rows(v.weight, std::size_t{g} * c.o_lora_rank, v.width);
                                d.width = v.width;
                                d.count = c.o_lora_rank;
                                d.rows = n;
                                d.x_stride = QL;
                                d.codes = m.ocodes.p + g * v.width;
                                d.deltas = m.odeltas.p + g * v.width / 32U;
                                d.out = m.o_low.p + g * c.o_lora_rank;
                                d.out_stride = OLL;
                                m.dense(d, m.o.p + g * v.width, st, false);
                            }
                        } else ds41::enqueue_q8_0_a8_prefill(v, st);
                    }
                    const bool fast_o_b = m.options.fp16_fast && m.fp16_at(16U, layer, n);
                    if (!fast_o_b) ds41::enqueue_quantize_a8(m.o_low.p, OLL, n, m.olcodes.p, m.oldeltas.p, m.olsums.p, st);
                    {
                        // wo_b as two K-halves h0 (columns 0..4095) + h1 (4096..8191): h0 into y, then h1 accumulated
                        const bool fp16_o_b = m.fp16_at(16U, layer, n);
                        ds41::Q8A8DensePrefillView v;
                        v.weight = m.dev(p + "attn_output_b.weight");
                        v.width = OL / 2U;
                        v.row_width = OL;
                        v.count = H;
                        v.rows = n;
                        v.x_stride = OLL;
                        v.out = m.y.p + std::size_t{r0} * H;
                        v.out_stride = H;
                        if (tp) {
                            // this rank's K-half, exchanged; y = h0 + h1
                            v.k_offset = m.options.rank * (OL / 2U);
                            v.codes = m.olcodes.p;
                            v.deltas = m.oldeltas.p;
                            const std::size_t bytes = std::size_t{n} * H * 4U;
                            staged_half = static_cast<float*>(m.options.communicator->send_staging(bytes));
                            v.out = staged_half ? staged_half : m.y_own.p;
                            if (fast_o_b) m.fast(&v, 1U, n, nullptr, 0U, m.o_low.p, st);
                            else m.dense(v,m.o_low.p,st,fp16_o_b);
                            const dist::Tag tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), layer, r0 / C};
                            if (staged_half) m.options.communicator->start_staged(st, bytes, tag);
                            else m.options.communicator->start(st, m.y_own.p, bytes, tag);
                            pending = r0;
                        } else if (fast_o_b) {
                            v.width = OL;  // the whole K in one GEMM
                            m.fast(&v, 1U, n, nullptr, 0U, m.o_low.p, st);
                        } else {
                            for (std::uint32_t part = 0; part < 2U; ++part) {
                                v.k_offset = part * (OL / 2U);
                                v.codes = m.olcodes.p + part * (OL / 2U);
                                v.deltas = m.oldeltas.p + part * (OL / 2U) / 32U;
                                v.accumulate = part == 1U;
                                m.dense(v,m.o_low.p+part*(OL/2U),st,fp16_o_b);
                            }
                        }
                    }
                    mark_part(4);
                }
                for (; !tp && chunks_cut < chunks_most; ++chunks_cut) cut(PrefillUnitLevel::finest, s.layer, "attn.pad");
                close_halves();
                mark_part(4);
                for (std::size_t k = 0; k < m.snap_points.size(); ++k) {  // the ring at each in-run snapshot point
                    auto& ring = m.snap_rings[k * m.options.layers + layer];
                    check(hipMemcpyAsync(ring.p, L.ring.p, L.ring.n, hipMemcpyDeviceToDevice, st), "snapshot ring");
                    const std::uint32_t end = m.snap_points[k], first = end > attn::kWindow ? end - attn::kWindow : 0U;
                    ring_copy_kernel<<<end - first, 160, 0, st>>>(reinterpret_cast<const std::uint32_t*>(m.prefill_ring.p), m.prefill_capacity,
                                                                  reinterpret_cast<std::uint32_t*>(ring.p), AttentionHost::ring_capacity, first);
                    check(hipGetLastError(), "snapshot ring copy");
                }
                {
                    const std::uint32_t end = P0 + T, first = end > attn::kWindow ? end - attn::kWindow : 0U;
                    ring_copy_kernel<<<end - first, 160, 0, st>>>(reinterpret_cast<const std::uint32_t*>(m.prefill_ring.p), m.prefill_capacity,
                                                                  reinterpret_cast<std::uint32_t*>(L.ring.p), AttentionHost::ring_capacity, first);
                    check(hipGetLastError(), "ring copy");
                }
            });
            break;
        }
        case Op::hc_post: {
            const int next = 1 - m.cur;
            // hc_fuse: the expansion + the next site's mixes when the next step is that site's hc_mix over the same rows
            // (attn: the ffn site; ffn: layer + 1's attention unless an Engram inject or the ULP diagnostic comes between)
            const auto* nx = &s + 1;
            const bool attn = s.site == "attn";
            const bool next_mix = hc_fuse && no > 0U && nx != m.plan.steps.data() + m.plan.steps.size() && nx->op == Op::hc_mix &&
                                  nx->site == (attn ? "ffn" : "attn") && nx->layer == s.layer + (attn ? 0 : 1) &&
                                  (attn || (layer + 1U < m.options.layers && lrows[layer + 1U].o0 == o0 &&
                                            m.options.null_ulp_layer != static_cast<int>(layer)));
            if (next_mix) {
                // as enqueue_hc_expand + enqueue_hc_mixes (the next hc_mix is skipped)
                ds41::HcSiteView v;
                v.rows = no;
                v.hidden = H;
                v.sub = m.y.p + std::size_t{o0} * H;
                v.residual = stream_now() + std::size_t{o0} * stream_row;
                v.expand_mixes = (attn ? m.mix_attn.p : m.mix_ffn.p) + o0;
                v.stream = m.streams[next].p + std::size_t{o0} * stream_row;
                v.fn = m.devh(nx->weights[0]);
                v.base = m.devf(nx->weights[1]);
                v.scale = m.devf(nx->weights[2]);
                v.iters = c.hc_sinkhorn_iters;
                v.hc_eps = static_cast<float>(c.hc_eps);
                v.mix_norm_eps = 1e-20F;
                v.mixes_out = (attn ? m.mix_ffn.p : m.mix_attn.p) + o0;
                v.partial = m.mix_partial.p;
                v.counters = m.hc_counters.p;
                v.launches = 1;
                v.form = 2;
                v.tile = no >= 8U ? 8U : 0U;  // 0: by the row count (1, 2 or 4)
                // the next site's collapse uses this expansion's mixes (mix_attn after attn, mix_ffn after ffn) from its
                // first row: o0 for the ffn site, the next layer's x0 for its attention
                if (hc_collapse && (attn || lrows[layer + 1U].x0 == o0)) {
                    v.collapse_out = m.x.p + std::size_t{o0} * H;
                    hc_collapsed = true;
                }
                ds41::enqueue_hc_site(v, st);
                hc_mixed = true;
            } else if (no > 0U)
                ds41::enqueue_hc_expand(m.y.p + std::size_t{o0} * H, stream_now() + std::size_t{o0} * stream_row,
                                        (s.site == "attn" ? m.mix_attn.p : m.mix_ffn.p) + o0, no, H, m.streams[next].p + std::size_t{o0} * stream_row,
                                        st);
            m.cur = next;
            if (s.site == "ffn") {
                m.identity_pre = false;
                if (m.options.null_ulp_layer == static_cast<int>(layer) && no > 0U) {
                    const std::size_t count = std::size_t{no} * stream_row;
                    ulp_flip_kernel<<<static_cast<unsigned>((count + 255U) / 256U), 256, 0, st>>>(stream_now() + std::size_t{o0} * stream_row, count,
                                                                                     m.options.null_seed);
                    check(hipGetLastError(), "ulp flip");
                }
                if (m.options.trace) {
                    PrefillLayerTrace t;
                    t.router_ids = std::move(trace_router);
                    trace_router.clear();
                    if (m.roles[layer].owns_index) {
                        const auto idx = download(m.attn[layer].idx.p, std::size_t{Tn} * attn::kTopK, st);
                        const auto cnt = download(m.attn[layer].idx_count.p, Tn, st);
                        t.index_sets.resize(Tn);
                        for (std::uint32_t r = 0; r < Tn; ++r) {
                            std::uint64_t h = 0;
                            const std::uint32_t k = std::min<std::uint32_t>(cnt[r], attn::kTopK);
                            for (std::uint32_t j = 0; j < k; ++j) {
                                std::uint64_t x = idx[std::size_t{r} * attn::kTopK + j] + 0x9E3779B97F4A7C15ULL;
                                x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
                                x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
                                h += x ^ (x >> 31);
                            }
                            t.index_sets[r] = h ^ (std::uint64_t{k} << 48);
                        }
                    }
                    if (no > 0U) {
                        row_hash_kernel<<<no, 256, 0, st>>>(stream_now() + std::size_t{o0} * stream_row, stream_row, m.trace_hash.p);
                        check(hipGetLastError(), "row hash");
                        t.row_hashes = download(m.trace_hash.p, no, st);
                    }
                    const std::uint32_t tail = std::min(std::max(m.options.quality_head_rows, 1U), Tn);
                    t.tail = download(stream_now() + std::size_t{Tn - tail} * stream_row, std::size_t{tail} * stream_row, st);
                    result.trace.push_back(std::move(t));
                }
                if (m.options.stream_hashes)  // the last row's stream, on its owner (row_split: 0 on the other rank)
                    result.stream_hashes.push_back(last_local ? fnv_bytes(download(stream_now() + std::size_t{Tn - 1U} * stream_row, stream_row, st).data(),
                                                                          stream_row * 4U)
                                                              : 0U);
                if (seed_n != 0U && layer + 1U >= m.options.layers - 3U && layer + 1U < m.options.layers && seed_own != 0U)
                    check(hipMemcpyAsync(m.seed_local[layer + 4U - m.options.layers].p, stream_now() + std::size_t{seed_l0} * stream_row,
                                         std::size_t{seed_own} * stream_row * 4U, hipMemcpyDeviceToDevice, st),
                          "seed rows");  // the input of layer + 1
            }
            break;
        }
        case Op::router: {
            if (no == 0U) {
                if (m.options.routing_loads) result.routing_loads.emplace_back(c.n_routed_experts, 0U);
                if (split) begin_moe_input(R0, o0, no, layer);  // nothing of this rank's crosses, but the gather still pairs with the peer's
                break;
            }
            ds41::RouterView v;
            v.x = m.x.p + std::size_t{o0} * H;
            v.weight = m.devf(s.weights[0]);
            v.bias = m.devf(s.weights[1]);
            v.bias_image = m.devf(s.weights[2]);
            v.image_rows = m.image_rows.p;
            v.rows = no;
            v.hidden = H;
            v.experts = c.n_routed_experts;
            v.top_k = K;
            v.route_scale = static_cast<float>(c.routed_scaling_factor);
            v.scores = m.scores.p;
            v.ids = m.ids.p + std::size_t{o0} * K;
            v.weights = m.route_w.p + std::size_t{o0} * K;
            ds41::enqueue_prefill_router(v, st);
            if (m.options.trace) trace_router = download(m.ids.p + std::size_t{o0} * K, std::size_t{no} * K, st);
            if (split) begin_moe_input(R0, o0, no, layer);  // the MoE input of this rank's rows is ready; the shared expert runs while it crosses
            if (m.options.routing_loads) {  // per-expert entries of this layer's output rows
                const auto ids = download(m.ids.p + std::size_t{o0} * K, std::size_t{no} * K, st);
                std::vector<std::uint32_t> load(c.n_routed_experts, 0U);
                for (const auto e : ids) ++load[e];
                result.routing_loads.push_back(std::move(load));
            }
            break;
        }
        case Op::shared_expert: {
            if (no == 0U) break;
            sync_time(moe_s, [&] {
                const auto gate_row = row_bytes(ds41::WeightFormat::q8_0, H), down_row = row_bytes(ds41::WeightFormat::q8_0, I);
                const auto& routes = site_routes(no, 1U);
                ds41::PrefillGateUpView gu;
                gu.gate = m.dev(s.weights[0]);
                gu.up = m.dev(s.weights[1]);
                gu.expert_stride_bytes = gate_row * I;
                gu.row_bytes = gate_row;
                gu.hidden = H;
                gu.inter = I;
                gu.routes = routes;
                gu.codes = m.xcodes.p + std::size_t{o0} * H;
                gu.deltas = m.xdeltas.p + std::size_t{o0} * H / 32U;
                gu.limit = static_cast<float>(c.swiglu_limit);
                gu.act_codes = m.sh_codes.p;
                gu.act_deltas = m.sh_deltas.p;
                gu.act_sums = m.sh_sums.p;
                ds41::enqueue_prefill_gate_up(ds41::WeightFormat::q8_0, gu, st);
                ds41::PrefillDownView dv;
                dv.weights = m.dev(s.weights[2]);
                dv.expert_stride_bytes = down_row * H;
                dv.row_bytes = down_row;
                dv.width = I;
                dv.outputs = H;
                dv.routes = routes;
                dv.codes = m.sh_codes.p;
                dv.deltas = m.sh_deltas.p;
                dv.out = m.shared.p + std::size_t{o0} * H;
                ds41::enqueue_prefill_down(ds41::WeightFormat::q8_0, dv, st);
            });
            break;
        }
        case Op::routed_experts:
            break;  // with the combine, chunk by chunk
        case Op::combine: {
            sync_time(moe_s, [&] {
                ds41::PrefillMoeWeights w;
                const auto gate_row = row_bytes(ds41::WeightFormat::iq2_xxs, H);
                // expert TP: 1152 local intermediate rows; down rows of 5 whole Q2_K blocks (dup layout) read as width 1280
                const std::uint32_t I_routed = m.expert_tp ? ds41::expert_tp::half : static_cast<std::uint32_t>(I);
                const std::uint32_t down_width = m.expert_tp ? (ds41::expert_tp::half + 255U) / 256U * 256U : 0U;
                const auto down_row = m.expert_tp ? ds41::expert_tp::down_row_bytes : row_bytes(ds41::WeightFormat::q2_k, I);
                const std::uint32_t moe_half = m.expert_tp ? 0U : m.options.rank;  // expert TP: every slot in half 0
                if (m.options.stream_experts) {
                    // this layer's slot was filled meanwhile; the other slot (layer - 1, finished) takes layer + 1
                    if (m.slots[layer % 2U].layer != static_cast<std::int32_t>(layer)) join_prefetch();
                    if (m.slots[layer % 2U].layer != static_cast<std::int32_t>(layer)) load_experts(layer);
                    join_prefetch();
                    start_prefetch(layer + 1U);
                }
                w.gate = expert_weight(layer, "gate");
                w.up = expert_weight(layer, "up");
                w.down = expert_weight(layer, "down");
                w.gate_row_bytes = gate_row;
                w.down_row_bytes = down_row;
                w.hidden = H;
                w.inter = I_routed;
                w.down_width = down_width;
                w.limit = static_cast<float>(c.swiglu_limit);
                const std::uint32_t MC = m.options.moe_chunk_rows;
                std::unique_ptr<ds41::PrefillMoe> layer_moe;
                if (!m.moe_persist || !m.moe) {
                    layer_moe = std::make_unique<ds41::PrefillMoe>(w, K, c.n_routed_experts, m.expert_tp ? c.n_routed_experts : m.plan.expert_split, MC,
                                                                   m.distributed() ? m.expert_first : 0U, m.distributed() ? m.expert_end : 0U);
                    if (m.moe_persist) m.moe = std::move(layer_moe);
                } else {
                    m.moe->set_weights(w);  // the previous layer's work on the stream keeps its own weights
                }
                ds41::PrefillMoe& moe = layer_moe ? *layer_moe : *m.moe;
                // A task's MoE chunk TMC <= MC: the same PrefillMoe (no second scratch) on views of TMC rows (chunk 0 of
                // the view; the outputs are indexed relative to the view's first row). Rows are independent in the MoE,
                // so the bits are those of the MC chunks. Without a task (or TMC = MC) the chunks of MC rows as before.
                const std::uint32_t TMC = m.task != nullptr && m.task->opt.moe_chunk_rows != 0U ? std::min(m.task->opt.moe_chunk_rows, MC) : MC;
                const auto pieces = [&](std::uint32_t n) { return TMC == MC ? moe.chunks(n) : (n + TMC - 1U) / TMC; };
                const auto view = [&](const ds41::PrefillMoeInputs& a, std::uint32_t r0) {
                    ds41::PrefillMoeInputs v = a;
                    v.rows = std::min(TMC, a.rows - r0);
                    v.ids = a.ids + std::size_t{r0} * K;
                    v.weights = a.weights + std::size_t{r0} * K;
                    v.codes = a.codes + std::size_t{r0} * H;
                    v.deltas = a.deltas + std::size_t{r0} * H / 32U;
                    if (a.shared != nullptr) v.shared = a.shared + std::size_t{r0} * H;
                    return v;
                };
                const auto half_piece = [&](const ds41::PrefillMoeInputs& a, std::uint32_t k, float* out) {
                    if (TMC == MC) moe.enqueue_half(a, k, moe_half, out, st);  // expert TP: every slot in half 0
                    else moe.enqueue_half(view(a, k * TMC), 0U, moe_half, out + std::size_t{k} * TMC * H, st);
                };
                ds41::PrefillMoeInputs in;
                in.rows = no;
                in.ids = m.ids.p + std::size_t{o0} * K;
                in.weights = m.route_w.p + std::size_t{o0} * K;
                in.codes = m.xcodes.p + std::size_t{o0} * H;
                in.deltas = m.xdeltas.p + std::size_t{o0} * H / 32U;
                in.shared = m.shared.p + std::size_t{o0} * H;
                float* y = m.y.p + std::size_t{o0} * H;
                float* own = m.half_own.p + std::size_t{o0} * H;
                float* other = m.half_other.p + std::size_t{o0} * H;
                // world 2: chunk ch's half sums cross in pieces of C rows (messages <= 21 MB) on the communication stream,
                // issued while chunk ch + 1's half runs on the main stream; then y = (shared + low) + high of chunk ch
                const auto exchange_chunk = [&](std::uint32_t ch) {
                    const std::uint32_t r0 = ch * MC, n = std::min(MC, no - r0);
                    hipStream_t cs = m.comm_stream;
                    check(hipStreamWaitEvent(cs, m.half_ready[ch % 2U], 0), "half ready");
                    for (std::uint32_t p0 = 0; p0 < n; p0 += C) {
                        const std::uint32_t pn = std::min(C, n - p0);
                        const std::size_t at = std::size_t{r0 + p0} * H;
                        m.options.communicator->start(cs, own + at, std::size_t{pn} * H * 4U,
                                                      dist::Tag{static_cast<std::uint32_t>(dist::TagKind::half_sum), layer, (r0 + p0) / C});
                        m.options.communicator->finish(cs, other + at);
                    }
                    check(hipEventRecord(m.half_received, cs), "half received");
                    check(hipStreamWaitEvent(st, m.half_received, 0), "half join");
                    const float* low = (m.options.rank == 0U ? own : other) + std::size_t{r0} * H;
                    const float* high = (m.options.rank == 0U ? other : own) + std::size_t{r0} * H;
                    ds41::enqueue_canonical_sum(in.shared + std::size_t{r0} * H, low, high, std::size_t{n} * H, y + std::size_t{r0} * H, st);
                };
                if (split) {
                    // the MoE input of the peer's output rows; this rank's half of the peer's rows first, sent while this
                    // rank's own rows run; the peer's half of this rank's rows comes back; y = (shared + low) + high
                    const std::uint32_t p0 = local_of(peer, R0.o0), np = Tp - p0;
                    if (moe_input_rows > 0U) {  // staged at the router: already on the wire
                        const auto* recv = static_cast<const std::uint32_t*>(m.options.communicator->finish_staged(st));
                        hipLaunchKernelGGL(unpack_rows_kernel, dim3(moe_input_rows), dim3(256), 0, st, moe_input_pack(R0, o0), recv);
                        check(hipGetLastError(), "unpack rows");
                        m.options.communicator->release_staged(st);
                        moe_input_rows = 0;
                    } else {
                        exchange_fields({Field{m.xcodes.p + std::size_t{o0} * H, m.peer_xcodes.p + std::size_t{p0} * H, H},
                                         Field{m.xdeltas.p + std::size_t{o0} * H / 32U, m.peer_xdeltas.p + std::size_t{p0} * H / 32U, H / 32U * 4U},
                                         Field{m.ids.p + std::size_t{o0} * K, m.peer_ids.p + std::size_t{p0} * K, K * 4U},
                                         Field{m.route_w.p + std::size_t{o0} * K, m.peer_route_w.p + std::size_t{p0} * K, K * 4U}},
                                        no, np, layer);
                        join_gather();
                    }
                    ds41::PrefillMoeInputs pin;
                    pin.rows = np;
                    pin.ids = m.peer_ids.p + std::size_t{p0} * K;
                    pin.weights = m.peer_route_w.p + std::size_t{p0} * K;
                    pin.codes = m.peer_xcodes.p + std::size_t{p0} * H;
                    pin.deltas = m.peer_xdeltas.p + std::size_t{p0} * H / 32U;
                    // both ranks send max(np, no) rows (the same message size); staged: in one message
                    const std::uint32_t nmax = std::max(np, no);
                    const std::size_t half_bytes = std::size_t{nmax} * H * 4U;
                    const dist::Tag half_tag{static_cast<std::uint32_t>(dist::TagKind::half_sum), layer, 0U};
                    // A task parks only at a boundary its grant runs out at, and the scheduler may then run other exchanges
                    // (which also write the send buffer the peer half sums would build up in): staged only if every MoE unit
                    // of this layer runs now (same unit and grant on both ranks), and if this size takes the host path
                    // (the same route table on both ranks); otherwise the copy path through half_other / half_recv.
                    float* peer_half = m.staged_half && nmax > 0U &&
                                               (m.task == nullptr || m.moe_joint || T <= m.joint_max_rows ||
                                                m.task->runs_through(PrefillUnitLevel::finest, pieces(np) + pieces(no) + 1U))
                                           ? static_cast<float*>(m.options.communicator->send_staging(half_bytes))
                                           : nullptr;
                    const bool staged = peer_half != nullptr;
                    if (!staged) peer_half = m.half_other.p;
                    if (m.moe_joint || T <= m.joint_max_rows) {  // one expert pass over [peer rows; own rows], the peer rows' half sums first
                        cut(PrefillUnitLevel::finest, s.layer, "moe.joint");
                        moe.enqueue_half_joint(pin, in, moe_half, peer_half, m.half_own.p, st, m.half_ready[0]);
                        if (staged) m.options.communicator->start_staged(st, half_bytes, half_tag);
                    } else {  // task units: one per chunk pass (chunks(np) + chunks(no): the same count on both ranks)
                        for (std::uint32_t ch = 0; ch < pieces(np); ++ch) {
                            cut(PrefillUnitLevel::finest, s.layer, "moe.peer");
                            half_piece(pin, ch, peer_half);
                        }
                        check(hipEventRecord(m.half_ready[0], st), "peer half ready");
                        // staged: on the wire as soon as the peer rows are done, while the own rows below run
                        if (staged) m.options.communicator->start_staged(st, half_bytes, half_tag);
                        for (std::uint32_t ch = 0; ch < pieces(no); ++ch) {
                            cut(PrefillUnitLevel::finest, s.layer, "moe.own");
                            half_piece(in, ch, m.half_own.p);
                        }
                    }
                    // task: the half sums' exchange and the canonical sum as their own unit (two-pass form: the joint
                    // pass is one unit already); the same place on both ranks
                    if (!(m.moe_joint || T <= m.joint_max_rows)) cut(PrefillUnitLevel::finest, s.layer, "moe.sum");
                    if (staged) {
                        const auto* recv = static_cast<const float*>(m.options.communicator->finish_staged(st));
                        const float* low = me == 0U ? m.half_own.p : recv;
                        const float* high = me == 0U ? recv : m.half_own.p;
                        if (no > 0U) ds41::enqueue_canonical_sum(in.shared, low, high, std::size_t{no} * H, y, st);
                        m.options.communicator->release_staged(st);
                        if (!m.moe_persist || m.options.stream_experts) check(hipStreamSynchronize(st), "moe");
                        return;
                    }
                    hipStream_t cs = m.comm_stream;
                    check(hipStreamWaitEvent(cs, m.half_ready[0], 0), "peer half ready");
                    for (std::uint32_t q0 = 0; q0 < nmax; q0 += C) {
                        const std::uint32_t qn = std::min(C, nmax - q0);
                        m.options.communicator->start(cs, m.half_other.p + std::size_t{q0} * H, std::size_t{qn} * H * 4U,
                                                      dist::Tag{static_cast<std::uint32_t>(dist::TagKind::half_sum), layer, q0 / C});
                        m.options.communicator->finish(cs, m.half_recv.p + std::size_t{q0} * H);
                    }
                    check(hipEventRecord(m.half_received, cs), "half received");
                    check(hipStreamWaitEvent(st, m.half_received, 0), "half join");
                    const float* low = me == 0U ? m.half_own.p : m.half_recv.p;
                    const float* high = me == 0U ? m.half_recv.p : m.half_own.p;
                    if (no > 0U) ds41::enqueue_canonical_sum(in.shared, low, high, std::size_t{no} * H, y, st);
                    // a per-layer scratch is freed here; with streamed experts the slot is overwritten next
                    if (!m.moe_persist || m.options.stream_experts) check(hipStreamSynchronize(st), "moe");
                    return;
                }
                if (!m.distributed()) {  // world 1: the combine chunk by chunk (task: pieces of TMC rows)
                    for (std::uint32_t ch = 0; ch < pieces(no); ++ch) {
                        cut(PrefillUnitLevel::finest, s.layer, "moe.chunk");
                        if (TMC == MC) moe.enqueue_combine(in, ch, y, st);
                        else moe.enqueue_combine(view(in, ch * TMC), 0U, y + std::size_t{ch} * TMC * H, st);
                    }
                    if (!m.moe_persist || m.options.stream_experts) check(hipStreamSynchronize(st), "moe");
                    return;
                }
                const std::uint32_t chunks = moe.chunks(no);
                for (std::uint32_t ch = 0; ch < chunks; ++ch) {
                    moe.enqueue_half(in, ch, moe_half, own, st);
                    check(hipEventRecord(m.half_ready[ch % 2U], st), "half ready");
                    if (ch > 0U) exchange_chunk(ch - 1U);
                }
                if (m.distributed() && chunks > 0U) exchange_chunk(chunks - 1U);
                // a per-layer scratch is freed here; with streamed experts the slot is overwritten next
                if (!m.moe_persist || m.options.stream_experts) check(hipStreamSynchronize(st), "moe");
            });
            break;
        }
        case Op::head: {
            if(m.options.quality_head_rows>1) {
                unsigned nr=std::min(m.options.quality_head_rows,Tn),first=Tn-nr;
                ds41::enqueue_hc_collapse(stream_now()+std::size_t(first)*stream_row,m.identity_pre?nullptr:m.mix_ffn.p+first,nr,H,m.quality_h.p,st);
                ds41::enqueue_rms_norm(m.quality_h.p,m.devf("output_norm.weight"),nr,H,H,1e-20F,m.quality_h.p,st);
                ds41::enqueue_quantize_a8(m.quality_h.p,H,nr,m.quality_codes.p,m.quality_deltas.p,m.quality_sums.p,st);
                ds41::Q8A8DensePrefillView q;
                q.weight=m.dev("output.weight");q.width=H;q.count=m.head_slice.count;q.codes=m.quality_codes.p;q.deltas=m.quality_deltas.p;
                q.rows=nr;q.x_stride=H;q.out_stride=q.count;q.out=m.quality_logits.p;
                ds41::enqueue_q8_0_a8_dense_prefill(q,st);
                auto own=download(q.out,std::size_t(nr)*q.count,st);
                result.teacher_logits.assign(std::size_t(nr)*c.vocab_size,0);
                std::vector<float> other;
                auto sl=ds41::vocab_slice(c.vocab_size,2U,1U-m.options.rank);
                if(m.distributed()) {
                    m.options.communicator->exchange(st,q.out,m.quality_logits.p+std::size_t(nr)*q.count,std::size_t(nr)*q.count*4U,
                        dist::Tag{static_cast<std::uint32_t>(dist::TagKind::control),4U,T-1U});
                    other=download(m.quality_logits.p+std::size_t(nr)*q.count,std::size_t(nr)*sl.count,st);
                }
                for(unsigned r=0;r<nr;++r) {
                    std::copy_n(own.data()+std::size_t(r)*q.count,q.count,result.teacher_logits.data()+std::size_t(r)*c.vocab_size+m.head_slice.first);
                    if(m.distributed())std::copy_n(other.data()+std::size_t(r)*sl.count,sl.count,result.teacher_logits.data()+std::size_t(r)*c.vocab_size+sl.first);
                }
            }

            const std::size_t last = Tn - 1U;  // the last row, local on its owner
            if (last_local) {
                ds41::enqueue_hc_collapse(stream_now() + last * stream_row, m.identity_pre ? nullptr : m.mix_ffn.p + last, 1, H, m.h.p, st);
                ds41::enqueue_rms_norm(m.h.p, m.devf("output_norm.weight"), 1, H, H, 1e-20F, m.h.p, st);
            }
            if (split)
                m.options.communicator->exchange(st, last_local ? m.h.p : m.h_peer.p, last_local ? m.h_peer.p : m.h.p, std::size_t{H} * 4U,
                                                 dist::Tag{static_cast<std::uint32_t>(dist::TagKind::control), 3U, T - 1U});
            ds41::enqueue_quantize_a8(m.h.p, H, 1, m.hcodes.p, m.hdeltas.p, m.hsums.p, st);
            ds41::Q8A8View v;
            v.weight = m.dev("output.weight");
            v.width = H;
            v.count = m.head_slice.count;
            v.per_group = m.head_slice.count;
            v.codes = m.hcodes.p;
            v.deltas = m.hdeltas.p;
            v.rows = 1;
            v.x_stride = H;
            v.out_stride = m.head_slice.count;
            check(hipMemsetAsync(m.logits.p, 0, std::size_t{c.vocab_size} * 4U, st), "logits");
            v.out = m.logits.p + m.head_slice.first;
            ds41::enqueue_q8_0_a8(v, st);
            if (m.distributed() && m.options.gather_logits) {
                const auto other = ds41::vocab_slice(c.vocab_size, 2U, 1U - m.options.rank);
                m.options.communicator->exchange(st, m.logits.p + m.head_slice.first, m.logits.p + other.first, std::size_t{other.count} * 4U,
                                                 dist::Tag{static_cast<std::uint32_t>(dist::TagKind::control), 1U, T - 1U});
            }
            break;
        }
        case Op::argmax: {
            if (m.distributed()) {
                const auto& sl = m.head_slice;
                ds41::enqueue_argmax(m.logits.p + sl.first, 1, sl.count, sl.count, sl.first, m.partial.p, m.cand_own.p, st);
                m.options.communicator->exchange(st, m.cand_own.p, m.cand_other.p, sizeof(ds41::ArgmaxCandidate),
                                                 dist::Tag{static_cast<std::uint32_t>(dist::TagKind::argmax), 0U, T - 1U});
                const auto* first = m.options.rank == 0U ? m.cand_own.p : m.cand_other.p;
                const auto* second = m.options.rank == 0U ? m.cand_other.p : m.cand_own.p;
                ds41::enqueue_combine_argmax(first, second, 1, m.best.p, st);
            } else {
                const auto r0 = ds41::vocab_slice(c.vocab_size, 2U, 0U), r1 = ds41::vocab_slice(c.vocab_size, 2U, 1U);
                ds41::enqueue_argmax(m.logits.p + r0.first, 1, r0.count, r0.count, r0.first, m.partial.p, m.cand0.p, st);
                ds41::enqueue_argmax(m.logits.p + r1.first, 1, r1.count, r1.count, r1.first, m.partial.p, m.cand1.p, st);
                ds41::enqueue_combine_argmax(m.cand0.p, m.cand1.p, 1, m.best.p, st);
            }
            const auto best = download(m.best.p, 1, st)[0];
            if (best.finite == 0U) throw std::runtime_error("deepseek41 prefill: non-finite logits");
            result.next_token = best.index;
            result.logits = download(m.logits.p, c.vocab_size, st);
            result.head_seconds = seconds_since(t_head);
            break;
        }
        }
    }
    join_engram(0xFFFFFFFFU);
    if (seed_n != 0U && !defer) {  // every seed row on both ranks, in position order (row_split: 8-row windows alternate)
        const std::size_t rb = std::size_t{hc} * H * 4U;
        for (std::uint32_t j = 0; j < 3U; ++j) {
            if (split) {
                start_gather();
                exchange_rows(m.seed_local[j].p, seed_own, m.seed_peer[j].p, seed_n - seed_own, rb, 1100U + j);
                join_gather();
            }
            std::uint32_t own = 0, other = 0;
            for (std::uint32_t g = seed_begin; g < T;) {  // runs of one window (or all rows without the split)
                const std::uint32_t end = split ? std::min(T, (g / W + 1U) * W) : T;
                const bool mine = !split || window_owner(g / W) == me;
                auto* src = mine ? m.seed_local[j].p + std::size_t{own} * hc * H : m.seed_peer[j].p + std::size_t{other} * hc * H;
                check(hipMemcpyAsync(m.seed[j].p + std::size_t{g - seed_begin} * hc * H, src, std::size_t{end - g} * rb, hipMemcpyDeviceToDevice, st),
                      "seed order");
                (mine ? own : other) += end - g;
                g = end;
            }
        }
    }
    m.seed_rows = defer ? 0U : seed_n;
    m.seed_first = P0 + seed_begin;
    check(hipStreamSynchronize(st), "prefill");
    if (m.options.stream_experts) {
        join_prefetch();
        start_prefetch(0U);  // the next run starts at layer 0 (slot 0 is free: layer 0 + 2k finished)
        result.expert_read_seconds = m.expert_read_seconds;
        result.expert_wait_seconds = m.expert_wait_seconds;
    }
    m.rows_done = P0 + T;
    m.local_rows_done = Tn;
    m.last_row_local = last_local;
    m.state_incomplete = defer;
#ifdef STRIX_DS41_PREFILL_FP16_LT
    if (m.fast_used) {
        // FP16 output form: no value of this run may have left the FP16 range (checked after the timed run)
        m.fast_used = false;
        if (const unsigned bad = m.dense_fp16.take_nonfinite(m.stream); bad != 0U)
            throw std::runtime_error("FP16 output form: " + std::to_string(bad) + " non-finite projection outputs");
    }
#endif
    result.seconds = seconds_since(t_run);
    return result;
}

std::vector<std::pair<std::string, std::uint64_t>> DevicePrefill::state_hashes() const {
    const auto& m = *impl_;
    if (m.state_incomplete) throw std::logic_error("deepseek41 prefill: the last run deferred its decoder (its continuation completes the state)");
    namespace attn = ds41::attn;
    std::vector<std::pair<std::string, std::uint64_t>> out;
    const std::uint32_t T = m.rows_done;
    for (std::uint32_t l = 0; l < m.options.layers; ++l) {
        const auto& r = m.roles[l];
        const auto& L = m.attn[l];
        const std::string at = "@" + std::to_string(l);
        const auto ring = download(L.ring.p, L.ring.n, m.stream);
        out.emplace_back("ring" + at, ring_window_hash(ring.data(), T));
        if (r.owns_kv) {
            const std::size_t entries = T / r.ratio;
            const auto kv = download(L.kv_cache.p, entries * attn::kKvEntryBytes, m.stream);
            const auto ix = download(L.idx_cache.p, entries * attn::kIdxEntryBytes, m.stream);
            out.emplace_back("kv" + at, fnv_bytes(kv.data(), kv.size()));
            out.emplace_back("idx_keys" + at, fnv_bytes(ix.data(), ix.size()));
            if (r.ratio == 2U) {
                const auto carry = download(L.carry[L.carry_cur].p, L.carry[L.carry_cur].n, m.stream);
                const auto valid = download(L.carry_valid[L.carry_cur].p, 1, m.stream);
                out.emplace_back("carry" + at, valid[0] != 0U ? fnv_bytes(carry.data(), carry.size() * 4U) : 0U);
            }
        }
        if (r.owns_index) {  // the last row's selection, on the rank that computed the row (row_split: 0 on the other)
            if (!m.last_row_local) {
                out.emplace_back("topk" + at, 0U);
            } else {
                const std::uint32_t last = m.local_rows_done - 1U;
                const auto count = download(L.idx_count.p + last, 1, m.stream)[0];
                const auto idx = download(L.idx.p + std::size_t{last} * attn::kTopK, count, m.stream);
                out.emplace_back("topk" + at, fnv_bytes(idx.data(), idx.size() * 4U, count));
            }
        }
    }
    return out;
}

void DevicePrefill::import_state(const AttentionState& s) {
    auto& m = *impl_;
    if (m.task != nullptr) throw std::logic_error("deepseek41 prefill: import while a task owns the instance");
    namespace attn = ds41::attn;
    const auto fail = [](const std::string& what) { throw std::invalid_argument("deepseek41 prefill: import_state: " + what); };
    if (s.layers.size() != m.options.layers) fail("layer count differs");
    if (s.tokens.size() != s.position || s.position == 0U) fail("token history length != position, or an empty state");
    if (s.position % m.options.window_rows != 0U || s.position >= m.options.max_positions) fail("position not a multiple of window_rows or no room left");
    for (std::uint32_t l = 0; l < m.options.layers; ++l) {
        const auto& r = m.roles[l];
        const auto& L = m.attn[l];
        const auto& S = s.layers[l];
        if (S.owns_kv != r.owns_kv || S.owns_index != r.owns_index || (r.owns_kv && S.ratio != r.ratio)) fail("roles of layer " + std::to_string(l));
        if (S.ring.size() != L.ring.n) fail("ring size of layer " + std::to_string(l));
        if (r.owns_kv) {
            const std::size_t entries = s.position / r.ratio;
            if (S.kv.size() != entries * attn::kKvEntryBytes || S.idx_keys.size() != entries * attn::kIdxEntryBytes) fail("caches of layer " + std::to_string(l));
            if (r.ratio == 2U && S.carry_valid && S.carry.size() != L.carry[0].n) fail("carry of layer " + std::to_string(l));
        }
    }
    for (const auto t : s.tokens)
        if (t >= m.model.config.vocab_size) fail("token outside the vocabulary");
    m.cancel_broken = true;
    hipStream_t st = m.stream;
    for (std::uint32_t l = 0; l < m.options.layers; ++l) {
        const auto& r = m.roles[l];
        auto& L = m.attn[l];
        const auto& S = s.layers[l];
        check(hipMemcpyAsync(L.ring.p, S.ring.data(), L.ring.n, hipMemcpyHostToDevice, st), "import ring");
        if (r.owns_kv) {
            L.kv_cache.zero(st);
            L.idx_cache.zero(st);
            check(hipMemcpyAsync(L.kv_cache.p, S.kv.data(), S.kv.size(), hipMemcpyHostToDevice, st), "import kv");
            check(hipMemcpyAsync(L.idx_cache.p, S.idx_keys.data(), S.idx_keys.size(), hipMemcpyHostToDevice, st), "import idx");
            for (int k = 0; k < 2; ++k) {
                L.carry[k].zero(st);
                L.carry_valid[k].zero(st);
            }
            L.carry_cur = 0;
            if (r.ratio == 2U && S.carry_valid) {
                check(hipMemcpyAsync(L.carry[0].p, S.carry.data(), S.carry.size() * 4U, hipMemcpyHostToDevice, st), "import carry");
                check(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(L.carry_valid[0].p), 1, 1, st), "import carry");
            }
        }
    }
    m.hasher->reset();
    std::vector<std::uint64_t> one(std::size_t{m.hasher->layers()} * m.hasher->columns());
    for (const auto t : s.tokens) m.hasher->push(t, one);
    m.rows_done = s.position;
    m.local_rows_done = 0;
    m.last_row_local = false;
    check(hipStreamSynchronize(st), "import state");
    m.cancel_broken = false;
    m.state_incomplete = false;  // a complete state replaces a deferred run's (only after the import succeeded)
    m.defer_next = 0;
}

AttentionState DevicePrefill::export_state() const {
    const auto& m = *impl_;
    if (m.state_incomplete) throw std::logic_error("deepseek41 prefill: the last run deferred its decoder (its continuation completes the state)");
    namespace attn = ds41::attn;
    AttentionState s;
    const std::uint32_t T = m.rows_done;
    s.position = T;
    s.layers.resize(m.options.layers);
    if (m.overlap) {  // the same bytes with two stream synchronisations instead of several per layer
        const auto get = [&](auto& v, const auto* p, std::size_t n) {
            v.resize(n);
            if (n != 0U) check(hipMemcpyAsync(v.data(), p, n * sizeof(*p), hipMemcpyDeviceToHost, m.stream), "export");
        };
        std::vector<std::uint32_t> valid(m.options.layers, 0U), count(m.options.layers, 0U);
        const std::uint32_t last = m.local_rows_done - 1U;
        for (std::uint32_t l = 0; l < m.options.layers; ++l) {
            const auto& r = m.roles[l];
            const auto& L = m.attn[l];
            if (r.owns_kv && r.ratio == 2U)
                check(hipMemcpyAsync(&valid[l], L.carry_valid[L.carry_cur].p, 4, hipMemcpyDeviceToHost, m.stream), "export");
            if (r.owns_index && m.last_row_local) check(hipMemcpyAsync(&count[l], L.idx_count.p + last, 4, hipMemcpyDeviceToHost, m.stream), "export");
        }
        for (std::uint32_t l = 0; l < m.options.layers; ++l) {
            const auto& r = m.roles[l];
            const auto& L = m.attn[l];
            auto& S = s.layers[l];
            S.owns_kv = r.owns_kv;
            S.owns_index = r.owns_index;
            S.ratio = r.ratio;
            get(S.ring, L.ring.p, L.ring.n);
            if (r.owns_kv) {
                const std::size_t entries = T / r.ratio;
                get(S.kv, L.kv_cache.p, entries * attn::kKvEntryBytes);
                get(S.idx_keys, L.idx_cache.p, entries * attn::kIdxEntryBytes);
                if (r.ratio == 2U) get(S.carry, L.carry[L.carry_cur].p, L.carry[L.carry_cur].n);
            }
        }
        check(hipStreamSynchronize(m.stream), "export");  // the counts and the carries' flags are on the host now
        for (std::uint32_t l = 0; l < m.options.layers; ++l) {
            const auto& r = m.roles[l];
            const auto& L = m.attn[l];
            auto& S = s.layers[l];
            if (r.owns_kv && r.ratio == 2U) S.carry_valid = valid[l] != 0U;
            if (r.owns_index) {
                S.topk_known = m.last_row_local;
                if (m.last_row_local) get(S.topk, L.idx.p + std::size_t{last} * attn::kTopK, count[l]);
            }
        }
        check(hipStreamSynchronize(m.stream), "export");
        return s;
    }
    for (std::uint32_t l = 0; l < m.options.layers; ++l) {
        const auto& r = m.roles[l];
        const auto& L = m.attn[l];
        auto& S = s.layers[l];
        S.owns_kv = r.owns_kv;
        S.owns_index = r.owns_index;
        S.ratio = r.ratio;
        S.ring = download(L.ring.p, L.ring.n, m.stream);
        if (r.owns_kv) {
            const std::size_t entries = T / r.ratio;
            S.kv = download(L.kv_cache.p, entries * attn::kKvEntryBytes, m.stream);
            S.idx_keys = download(L.idx_cache.p, entries * attn::kIdxEntryBytes, m.stream);
            if (r.ratio == 2U) {
                S.carry = download(L.carry[L.carry_cur].p, L.carry[L.carry_cur].n, m.stream);
                S.carry_valid = download(L.carry_valid[L.carry_cur].p, 1, m.stream)[0] != 0U;
            }
        }
        if (r.owns_index) {
            S.topk_known = m.last_row_local;
            if (m.last_row_local) {
                const std::uint32_t last = m.local_rows_done - 1U;
                const auto count = download(L.idx_count.p + last, 1, m.stream)[0];
                S.topk = download(L.idx.p + std::size_t{last} * attn::kTopK, count, m.stream);
            }
        }
    }
    return s;
}

void DevicePrefill::set_engram_lookahead(std::span<const std::uint32_t> next) {
    static const bool on = [] { const char* v = std::getenv("STRIX_DS41_ENGRAM_LOOKAHEAD"); return !(v && std::strcmp(v, "0") == 0); }();
    impl_->engram_next.assign(next.begin(), next.end());
    if (!on) impl_->engram_next.clear();
}

void DevicePrefill::set_decoder_deferral(std::uint32_t next_rows) {
    if (impl_->task != nullptr) throw std::logic_error("deepseek41 prefill: a task is in progress");
    impl_->defer_next = next_rows;
}

void DevicePrefill::set_snapshot_points(std::vector<std::uint32_t> positions) {
    if (impl_->task != nullptr) throw std::logic_error("deepseek41 prefill: a task is in progress");
    std::sort(positions.begin(), positions.end());
    positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
    impl_->snap_next = std::move(positions);
}

AttentionState DevicePrefill::export_state_at(std::uint32_t position) const {
    const auto& m = *impl_;
    if (m.state_incomplete) throw std::logic_error("deepseek41 prefill: the last run deferred its decoder (its continuation completes the state)");
    namespace attn = ds41::attn;
    const auto it = std::find(m.snap_points.begin(), m.snap_points.end(), position);
    if (it == m.snap_points.end() || position > m.rows_done)
        throw std::invalid_argument("deepseek41 prefill: export_state_at: not a snapshot point of the last run");
    const std::size_t k = static_cast<std::size_t>(it - m.snap_points.begin());
    AttentionState s;
    s.position = position;
    s.layers.resize(m.options.layers);
    for (std::uint32_t l = 0; l < m.options.layers; ++l) {
        const auto& r = m.roles[l];
        const auto& L = m.attn[l];
        auto& S = s.layers[l];
        S.owns_kv = r.owns_kv;
        S.owns_index = r.owns_index;
        S.ratio = r.ratio;
        S.ring = download(m.snap_rings[k * m.options.layers + l].p, L.ring.n, m.stream);
        if (r.owns_kv) {
            const std::size_t entries = position / r.ratio;
            S.kv = download(L.kv_cache.p, entries * attn::kKvEntryBytes, m.stream);
            S.idx_keys = download(L.idx_cache.p, entries * attn::kIdxEntryBytes, m.stream);
            if (r.ratio == 2U) {  // after an odd last row no group is pending: the compressor leaves zeros, invalid
                S.carry.assign(L.carry[L.carry_cur].n, 0.0F);
                S.carry_valid = false;
            }
        }
        S.topk_known = !r.owns_index;
    }
    return s;
}

}  // namespace strix::deepseek41
