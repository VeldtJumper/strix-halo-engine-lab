// deepseek41 CUDA prefill: routed experts with per-expert residency and grouped transport (ds41_expert_streamer.hpp).
#include "ds41_expert_streamer.hpp"
#include "strix/ds41_memory_budget.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strix::ds41::cuda::detail {
namespace {

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}
constexpr std::size_t kStaging = 4, kStagingBytes = std::size_t{64} << 20;

}  // namespace

struct ExpertStreamer::Event {
    cudaEvent_t e = nullptr;
    explicit Event(bool timing = false) { STRIX_CUDA_CHECK(cudaEventCreateWithFlags(&e, timing ? cudaEventDefault : cudaEventDisableTiming)); }
    ~Event() {
        if (e) cudaEventDestroy(e);
    }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

PinnedBytes::PinnedBytes(std::size_t n) : n_(n) {
    if (n) STRIX_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&p_), n, cudaHostAllocDefault));
}
std::unique_ptr<PinnedBytes> PinnedBytes::try_alloc(std::size_t n, std::uint32_t attempts, std::uint32_t retry_ms) {
    auto out = std::make_unique<PinnedBytes>();
    for (std::uint32_t a = 0; a < std::max(1U, attempts); ++a) {
        if (a) std::this_thread::sleep_for(std::chrono::milliseconds(retry_ms));
        void* p = nullptr;
        if (cudaHostAlloc(&p, n, cudaHostAllocDefault) == cudaSuccess && p) {
            out->p_ = static_cast<std::uint8_t*>(p);
            out->n_ = n;
            return out;
        }
        (void)cudaGetLastError();  // allocation failures are not sticky: clear and retry
    }
    return nullptr;
}
// Telemetry failure is explicit; optional pins decline when measurements fail.
static void system_room(double& commit, double& physical) {
    const auto r = ds41::host_memory_room();
    commit = r.known ? double(r.free_commit) : -1;
    physical = r.known ? double(r.free_physical) : -1;
}
std::unique_ptr<PinnedBytes> PinnedBytes::try_alloc_floor(std::size_t n, double floor_gib,
        std::chrono::steady_clock::time_point deadline, std::uint32_t& waits,
        bool implicit, bool allow_first, HostPinReason& reason) {
    std::unique_ptr<PinnedBytes> out;
    std::uint64_t before_commit = 0;
    reason = ds41::allocate_host_pin(n, floor_gib, implicit, allow_first, deadline, waits,
        [&] { auto r = ds41::host_memory_room(); before_commit = r.free_commit; return r; }, [&] { out = PinnedBytes::try_alloc(n, 1, 0); return bool(out); },
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
    return out;
}
PinnedBytes::~PinnedBytes() {
    if (p_) cudaFreeHost(p_);
}

ExpertStreamer::ExpertStreamer(FileReader& file, StreamerPlan plan) : file_(file), p_(std::move(plan)) {
    const std::uint32_t E = p_.experts;
    if (p_.order.size() < p_.end || p_.resident.size() < p_.end || p_.stream_tier.size() < p_.end || p_.extents.size() < p_.end)
        throw std::invalid_argument("ds41 expert streamer: plan does not cover the layers");
    STRIX_CUDA_CHECK(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking));
    staging_.resize(kStaging);
    for (auto& st : staging_) {
        st.buf = std::make_unique<PinnedBytes>(kStagingBytes + 2U * FileReader::alignment);
        st.done = std::make_unique<Event>();
        pinned_bytes_ += st.buf->size();
    }
    resident_.resize(p_.end);
    perm_.resize(p_.end);
    host_.resize(p_.end);
    host_view_.assign(p_.end, nullptr);
    copy_begin_.resize(p_.end);
    copy_end_.resize(p_.end);
    copied_.assign(p_.end, false);
    copy_bytes_.assign(p_.end, 0);
    const std::uint64_t eb = p_.expert_bytes[0] + p_.expert_bytes[1] + p_.expert_bytes[2];
    for (std::uint32_t l = p_.first; l < p_.end; ++l) {
        const auto& order = p_.order[l];
        const std::uint32_t r = p_.resident[l], s = E - r;
        if (order.size() != E || r > E) throw std::invalid_argument("ds41 expert streamer: bad order / resident count");
        bool identity = true;
        std::vector<std::uint32_t> perm(E, E);
        for (std::uint32_t v = 0; v < E; ++v) {
            if (order[v] >= E || perm[order[v]] != E) throw std::invalid_argument("ds41 expert streamer: order is not a permutation");
            perm[order[v]] = v;
            identity &= order[v] == v;
        }
        if (!identity) {
            perm_[l] = std::make_unique<DeviceBuffer<std::uint32_t>>(E);
            perm_[l]->upload(perm.data(), E);
            device_bytes_ += E * 4U;
        }
        if (s > 0U) {
            streamed_.push_back(l);
            smax_ = std::max(smax_, s);
        }
    }
    // resident experts: [gate r][up r][down r] in virtual order, read from the file run by run of consecutive ids
    for (std::uint32_t l = p_.first; l < p_.end; ++l) {
        const std::uint32_t r = p_.resident[l];
        if (r == 0U) continue;
        resident_[l] = std::make_unique<DeviceBuffer<std::uint8_t>>(std::uint64_t{r} * eb);
        device_bytes_ += std::uint64_t{r} * eb;
        std::uint8_t* at = resident_[l]->get();
        for (int k = 0; k < 3; ++k) {
            const std::uint64_t b = p_.expert_bytes[k];
            for (std::uint32_t v = 0; v < r;) {
                std::uint32_t n = 1;
                while (v + n < r && p_.order[l][v + n] == p_.order[l][v] + n) ++n;
                file_to_device(p_.extents[l].offset[k] + std::uint64_t{p_.order[l][v]} * b, std::uint64_t{n} * b, at);
                at += std::uint64_t{n} * b;
                v += n;
            }
        }
        STRIX_CUDA_CHECK(cudaStreamSynchronize(copy_));
    }
    const std::uint32_t nslots = streamed_.empty() ? 0U : std::min<std::uint32_t>(std::max(1U, p_.slots), static_cast<std::uint32_t>(streamed_.size()));
    slot_bytes_ = std::uint64_t{smax_} * eb;
    std::uint32_t max_groups = 0;
    for (const auto l : streamed_) max_groups = std::max(max_groups, groups(l));
    for (std::uint32_t k = 0; k < nslots; ++k) {
        slots_.emplace_back();
        auto& s = slots_.back();
        s.buf = std::make_unique<DeviceBuffer<std::uint8_t>>(slot_bytes_);
        for (std::uint32_t g = 0; g < max_groups; ++g) s.group_ready.push_back(std::make_unique<Event>());
        s.free = std::make_unique<Event>();
        device_bytes_ += slot_bytes_;
    }
    if (p_.timing) {
        origin_ = std::make_unique<Event>(true);
        for (const auto l : streamed_) {
            copy_begin_[l] = std::make_unique<Event>(true);
            copy_end_[l] = std::make_unique<Event>(true);
        }
    }
    // host tier: the streamed experts packed in virtual order [gate s][up s][down s]
    {
        PinnedBytes page(kStagingBytes + 2U * FileReader::alignment);
        std::vector<std::uint32_t> owners;  // host layers with their own copy
        std::uint32_t host_layers = 0;
        const auto pin_deadline = host_pin_deadline(p_.host_pin_wait_s);
        for (const auto l : streamed_) {
            if (p_.stream_tier[l] != ExpertTier::host) continue;
            const std::uint32_t r = p_.resident[l], s = E - r;
            host_pin_stats_.requested_bytes += std::uint64_t{s} * eb;
            HostPinReason reason = HostPinReason::alloc;
            if (p_.host_alias != 0U && host_layers >= p_.host_alias) {
                // timing only: another host layer's bytes (wrong weights, real transfers; owners hold smax experts)
                host_view_[l] = host_view_[owners[host_layers % p_.host_alias]];
            } else {
                const std::uint32_t cap = p_.host_alias != 0U ? smax_ : s;
                const auto t_pin = std::chrono::steady_clock::now();
                host_[l] = (p_.host_tier_implicit || (p_.host_commit_floor_gib > 0.0 && p_.host_alias == 0U))
                               ? PinnedBytes::try_alloc_floor(std::uint64_t{cap} * eb, p_.host_commit_floor_gib, pin_deadline, host_pin_waits_, p_.host_tier_implicit, p_.host_pin_wait_s == 0.0, reason)
                               : PinnedBytes::try_alloc(std::uint64_t{cap} * eb, p_.host_pin_attempts, p_.host_pin_retry_ms);
                host_pin_s_ += seconds_since(t_pin);
                if (!host_[l]) {
                    // no pinned memory left (commit limit / physical): this layer streams from the file instead;
                    // the file path reads the same virtual order run by run, so the bits do not change
                    if (p_.host_alias != 0U && !p_.host_tier_implicit) throw std::runtime_error("ds41 expert streamer: pinned host allocation failed");
                    p_.stream_tier[l] = ExpertTier::file;
                    host_pin_stats_.demotions.push_back({l, reason});
                    ++host_pin_failed_;
                    continue;
                }
                pinned_bytes_ += host_[l]->size();
                if (p_.host_commit_floor_gib > 0.0) {
                    double c = 0.0, ph = 0.0;
                    system_room(c, ph);
                    std::fprintf(stderr, "host pin layer %u: pinned %.2f GB, free commit %.2f GiB, free phys %.2f GiB, %.1f s\n", l,
                                 double(pinned_bytes_) / 1e9, c / double(1ULL << 30), ph / double(1ULL << 30), host_pin_s_);
                }
                std::uint8_t* base = host_[l]->get();
                std::uint64_t off = 0;
                for (int k = 0; k < 3; ++k) {
                    const std::uint64_t b = p_.expert_bytes[k];
                    std::uint8_t* at = base + off;
                    for (std::uint32_t v = r; v < E;) {
                        std::uint32_t n = 1;
                        while (v + n < E && p_.order[l][v + n] == p_.order[l][v] + n) ++n;
                        file_to_host(p_.extents[l].offset[k] + std::uint64_t{p_.order[l][v]} * b, std::uint64_t{n} * b, at, page);
                        at += std::uint64_t{n} * b;
                        v += n;
                    }
                    off += std::uint64_t{cap} * b;
                }
                host_view_[l] = base;
                owners.push_back(l);
            }
            host_pin_stats_.layers.push_back(l);
            host_pin_stats_.effective_bytes += std::uint64_t{s} * eb;
            ++host_layers;
        }
    }
    if (!slots_.empty()) worker_ = std::thread([this] { loop(); });
}

ExpertStreamer::~ExpertStreamer() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    if (worker_.joinable()) worker_.join();
    cudaStreamSynchronize(copy_);
    cudaStreamDestroy(copy_);
}

std::uint32_t ExpertStreamer::groups(std::uint32_t l) const {
    const std::uint32_t s = streamed(l);
    if (s == 0U) return 0U;
    return p_.group_experts == 0U ? 1U : (s + p_.group_experts - 1U) / p_.group_experts;
}

std::size_t ExpertStreamer::use_of(std::uint32_t layer) const {
    if (layer >= acquired_.size() || acquired_[layer] == ~std::size_t{0})
        throw std::logic_error("ds41 expert streamer: layer has no acquired use");
    return acquired_[layer];
}

void ExpertStreamer::begin_run() {
    begin_run_uses(streamed_);
}

void ExpertStreamer::begin_run_uses(std::span<const std::uint32_t> layers) {
    std::lock_guard<std::mutex> lock(mutex_);
    uses_.clear();
    for (const auto layer : layers) {
        if (layer < p_.first || layer >= p_.end) throw std::invalid_argument("ds41 expert streamer: scheduled layer outside range");
        if (streamed(layer) != 0U) uses_.push_back(layer);
    }
    acquired_.assign(p_.end, ~std::size_t{0});
    acquire_cursor_ = 0;
    std::fill(copied_.begin(), copied_.end(), false);
    std::fill(copy_bytes_.begin(), copy_bytes_.end(), std::uint64_t{0});
    run_begin_t_ = std::chrono::steady_clock::now();
    origin_recorded_ = false;
    if (streamed_.empty()) return;
    run_ = true;
    ++run_id_;
    bytes_streamed_ = 0;
    file_bytes_ = 0;
    promoted_mark_ = promoted_end_;
    demoted_mark_ = demoted_end_;
    io_seconds_ = 0.0;
    wait_seconds_ = 0.0;
    wake_.notify_all();
}
void ExpertStreamer::end_run() {
    std::lock_guard<std::mutex> lock(mutex_);
    promoted_end_ = promoted_cum_;
    demoted_end_ = demoted_cum_;
    run_ = false;
    wake_.notify_all();
}

ExpertStreamer::Layer ExpertStreamer::acquire(std::uint32_t layer) {
    Layer out;
    const std::uint32_t E = p_.experts, r = p_.resident[layer], s = E - r;
    out.perm = perm_[layer] ? perm_[layer]->get() : nullptr;
    const std::uint64_t gb = p_.expert_bytes[0], ub = p_.expert_bytes[1];
    if (r > 0U) {
        const std::uint8_t* b = resident_[layer]->get();
        out.ranges.push_back({0U, r, 0U, {b, b + std::uint64_t{r} * gb, b + std::uint64_t{r} * (gb + ub)}, nullptr});
    }
    if (s == 0U) return out;
    std::size_t use = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (acquire_cursor_ >= uses_.size() || uses_[acquire_cursor_] != layer)
            throw std::logic_error("ds41 expert streamer: acquire differs from request schedule");
        use = acquired_[layer] = acquire_cursor_++;
    }
    auto& sl = slots_[use % slots_.size()];
    const auto t0 = std::chrono::steady_clock::now();
    {
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [&] { return (sl.loaded_run == run_id_ && sl.loaded_use == use) || !error_.empty(); });
        if (!error_.empty()) throw std::runtime_error(error_);
        wait_seconds_ += seconds_since(t0);
    }
    const std::uint8_t* b = sl.buf->get();
    const ExpertWeights w{b, b + std::uint64_t{smax_} * gb, b + std::uint64_t{smax_} * (gb + ub)};
    const std::uint32_t G = p_.group_experts == 0U ? s : p_.group_experts;
    for (std::uint32_t g = 0, f = r; f < E; ++g, f += G) out.ranges.push_back({f, std::min(E, f + G), r, w, sl.group_ready[g]->e});
    return out;
}

void ExpertStreamer::release(std::uint32_t layer, cudaStream_t stream) {
    if (p_.resident[layer] == p_.experts) return;
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

ExpertStreamer::Stats ExpertStreamer::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Stats st;
    st.bytes_streamed = bytes_streamed_;
    st.io_seconds = io_seconds_;
    st.wait_seconds = wait_seconds_;
    st.copy_end_ms.assign(p_.end, -1.0F);
    st.copy_begin_ms.assign(p_.end, -1.0F);
    st.copy_bytes = copy_bytes_;
    if (p_.timing && origin_recorded_)
        for (const auto l : streamed_) {
            if (!copied_[l]) continue;
            float b = 0.0F, e = 0.0F;
            STRIX_CUDA_CHECK(cudaEventElapsedTime(&b, origin_->e, copy_begin_[l]->e));
            STRIX_CUDA_CHECK(cudaEventElapsedTime(&e, origin_->e, copy_end_[l]->e));
            st.copy_begin_ms[l] = b;
            st.copy_end_ms[l] = e;
            st.copy_busy_ms += e - b;
        }
    st.preload_bytes = consumed_bytes_;
    st.preload_s = consumed_s_;
    st.preload_after_begin_s = consumed_after_begin_s_;
    // after end_run: up to that end (a promotion by the next preload belongs to the next run)
    st.promoted_layers = (run_ ? promoted_cum_ : promoted_end_) - promoted_mark_;
    st.demoted_layers = (run_ ? demoted_cum_ : demoted_end_) - demoted_mark_;
    st.promoted_held = promoted_held_;
    st.promoted_bytes = promoted_bytes_;
    st.file_bytes = file_bytes_;
    return st;
}
ExpertStreamer::Promoted ExpertStreamer::promoted() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {promoted_held_, promoted_cum_, demoted_cum_, promoted_bytes_};
}
cudaEvent_t ExpertStreamer::copy_origin() const { return origin_ ? origin_->e : nullptr; }

// file -> pinned staging -> device (copy stream), aligned superset pieces; blocks until enqueued
void ExpertStreamer::file_to_device(std::uint64_t offset, std::uint64_t bytes, std::uint8_t* dst) {
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
void ExpertStreamer::file_to_host(std::uint64_t offset, std::uint64_t bytes, std::uint8_t* dst, PinnedBytes& page) {
    const std::uint64_t A = FileReader::alignment;
    for (std::uint64_t done = 0; done < bytes;) {
        const std::uint64_t at = offset + done, a0 = at / A * A;
        const std::uint64_t n = std::min<std::uint64_t>(kStagingBytes, bytes - done);
        const std::uint64_t a1 = std::min<std::uint64_t>((at + n + A - 1U) / A * A, file_.size());
        file_.read_range(a0, static_cast<std::size_t>(a1 - a0), page.get());
        std::memcpy(dst + done, page.get() + (at - a0), n);
        done += n;
    }
}
// file -> pinned host memory (lazy promotion): straight into dst when it is aligned like the file offset (the
// reader's aligned path, no bounce), else the aligned superset through the staging ring and a host copy
void ExpertStreamer::file_to_pinned(std::uint64_t offset, std::uint64_t bytes, std::uint8_t* dst) {
    const std::uint64_t A = FileReader::alignment;
    const auto t0 = std::chrono::steady_clock::now();
    if (reinterpret_cast<std::uintptr_t>(dst) % A == offset % A) {
        file_.read_range(offset, static_cast<std::size_t>(bytes), dst, std::size_t{8} << 20);
    } else {
        for (std::uint64_t done = 0; done < bytes;) {
            auto& st = staging_[next_staging_++ % staging_.size()];
            if (st.used) STRIX_CUDA_CHECK(cudaEventSynchronize(st.done->e));  // its last copy to the device is done
            const std::uint64_t at = offset + done, a0 = at / A * A;
            const std::uint64_t n = std::min<std::uint64_t>(kStagingBytes, bytes - done);
            const std::uint64_t a1 = std::min<std::uint64_t>((at + n + A - 1U) / A * A, file_.size());
            file_.read_range(a0, static_cast<std::size_t>(a1 - a0), st.buf->get(), std::size_t{8} << 20);
            std::memcpy(dst + done, st.buf->get() + (at - a0), n);
            done += n;
        }
    }
    io_seconds_ += seconds_since(t0);
}
void ExpertStreamer::copy_piecewise(std::uint8_t* dst, const std::uint8_t* src, std::uint64_t bytes) {
    const std::uint64_t piece = p_.piece_bytes == 0U ? bytes : p_.piece_bytes;
    for (std::uint64_t at = 0; at < bytes; at += piece)
        STRIX_CUDA_CHECK(cudaMemcpyAsync(dst + at, src + at, std::min(piece, bytes - at), cudaMemcpyHostToDevice, copy_));
}

// Group g of the layer's streamed experts into the slot: its gate, up and down pieces, then the group's event.
// fill (a file-tier layer being promoted): the file runs go into the new pinned copy at the host tier's packed layout,
// and the slot is copied from there (the same bytes as through the staging ring).
void ExpertStreamer::load_group(Slot& s, std::uint32_t layer, std::uint32_t g, std::uint8_t* fill) {
    const std::uint32_t E = p_.experts, r = p_.resident[layer], n_s = E - r;
    const std::uint32_t G = p_.group_experts == 0U ? n_s : p_.group_experts;
    const std::uint32_t j0 = g * G, n = std::min(n_s - j0, G);
    const bool host = p_.stream_tier[layer] == ExpertTier::host;
    const std::uint32_t cap = host && p_.host_alias != 0U ? smax_ : n_s;  // the host copy's tensor stride (experts)
    std::uint64_t src_off = 0, dst_off = 0;
    for (int k = 0; k < 3; ++k) {
        const std::uint64_t b = p_.expert_bytes[k];
        std::uint8_t* dst = s.buf->get() + dst_off + std::uint64_t{j0} * b;
        if (host) {
            copy_piecewise(dst, host_view_[layer] + src_off + std::uint64_t{j0} * b, std::uint64_t{n} * b);
        } else if (fill != nullptr) {
            std::uint8_t* const h = fill + src_off + std::uint64_t{j0} * b;
            std::uint8_t* at = h;
            for (std::uint32_t v = r + j0; v < r + j0 + n;) {
                std::uint32_t m = 1;
                while (v + m < r + j0 + n && p_.order[layer][v + m] == p_.order[layer][v] + m) ++m;
                file_to_pinned(p_.extents[layer].offset[k] + std::uint64_t{p_.order[layer][v]} * b, std::uint64_t{m} * b, at);
                at += std::uint64_t{m} * b;
                v += m;
            }
            copy_piecewise(dst, h, std::uint64_t{n} * b);
        } else {
            for (std::uint32_t v = r + j0; v < r + j0 + n;) {
                std::uint32_t m = 1;
                while (v + m < r + j0 + n && p_.order[layer][v + m] == p_.order[layer][v] + m) ++m;
                file_to_device(p_.extents[layer].offset[k] + std::uint64_t{p_.order[layer][v]} * b, std::uint64_t{m} * b, dst);
                dst += std::uint64_t{m} * b;
                v += m;
            }
        }
        src_off += std::uint64_t{cap} * b;
        dst_off += std::uint64_t{smax_} * b;
    }
    STRIX_CUDA_CHECK(cudaEventRecord(s.group_ready[g]->e, copy_));
}

std::uint64_t ExpertStreamer::load_into(Slot& s, std::uint32_t layer) {
    s.layer = -1;
    const bool timed = p_.timing && run_;
    if (timed) {
        if (!origin_recorded_) {
            STRIX_CUDA_CHECK(cudaEventRecord(origin_->e, copy_));
            origin_recorded_ = true;
        }
        STRIX_CUDA_CHECK(cudaEventRecord(copy_begin_[layer]->e, copy_));
    }
    auto fresh = promote_buffer(layer);
    try {
        for (std::uint32_t g = 0; g < groups(layer); ++g) load_group(s, layer, g, fresh ? fresh->get() : nullptr);
    } catch (...) {
        if (fresh) cudaStreamSynchronize(copy_);  // no copy may still read the buffer freed below
        throw;
    }
    if (fresh) {
        // HOST from now on, exactly as a load-time host layer (the buffer stays until a demotion or the destructor)
        const std::size_t n = fresh->size();
        host_view_[layer] = fresh->get();
        host_[layer] = std::move(fresh);
        p_.stream_tier[layer] = ExpertTier::host;
        pinned_bytes_ += n;
        promote_lru_.add(layer, n, ++promote_clock_);
        std::lock_guard<std::mutex> g(mutex_);
        ++promoted_cum_;
        promoted_held_ = static_cast<std::uint32_t>(promote_lru_.size());
        promoted_bytes_ = promote_lru_.bytes();
    } else if (promote_lru_.contains(layer)) {
        promote_lru_.touch(layer, ++promote_clock_);
    }
    if (timed) {
        STRIX_CUDA_CHECK(cudaEventRecord(copy_end_[layer]->e, copy_));
        copied_[layer] = true;
    }
    const std::uint64_t eb = p_.expert_bytes[0] + p_.expert_bytes[1] + p_.expert_bytes[2];
    s.layer = static_cast<std::int32_t>(layer);
    return std::uint64_t{streamed(layer)} * eb;
}

// Loader thread: a new pinned buffer for this load of a file-tier layer when the switch is on, the run gave nothing
// back, and the host has room for one more whole layer (the largest streamed part) above the floor and under the cap.
std::unique_ptr<PinnedBytes> ExpertStreamer::promote_buffer(std::uint32_t layer) {
    if (!p_.host_promote.enabled || promote_blocked_ || p_.host_alias != 0U || p_.stream_tier[layer] != ExpertTier::file) return nullptr;
    const auto room = host_promote_room(ds41::host_memory_room(), p_.host_promote, promote_lru_.bytes());
    if (!host_promote_admit(room, slot_bytes_)) return nullptr;
    const std::uint64_t eb = p_.expert_bytes[0] + p_.expert_bytes[1] + p_.expert_bytes[2];
    return PinnedBytes::try_alloc(std::uint64_t{streamed(layer)} * eb, 1, 0);  // nullptr: streams from the file as before
}

// Loader thread, at a run start: below the floor, the least recently used promoted layers go back to the file tier
// (never a load-time host layer) until the shortfall is covered; the copy stream is drained first (no copy still
// reads a buffer that is freed).
void ExpertStreamer::demote_under_pressure() {
    promote_blocked_ = false;
    if (promote_lru_.size() == 0U) return;
    const auto victims = promote_lru_.victims(host_promote_deficit(ds41::host_memory_room(), p_.host_promote));
    if (victims.empty()) return;
    STRIX_CUDA_CHECK(cudaStreamSynchronize(copy_));
    std::uint64_t freed = 0;
    for (const auto l : victims) {
        freed += host_[l]->size();
        pinned_bytes_ -= host_[l]->size();
        host_view_[l] = nullptr;
        host_[l].reset();
        p_.stream_tier[l] = ExpertTier::file;
        promote_lru_.remove(l);
    }
    promote_blocked_ = true;
    std::fprintf(stderr, "host promote: %zu layer(s) back to the file tier (%.2f GB) below the %.1f GiB floor\n", victims.size(),
                 double(freed) / 1e9, double(p_.host_promote.floor_bytes) / double(1ULL << 30));
    std::lock_guard<std::mutex> g(mutex_);
    demoted_cum_ += static_cast<std::uint32_t>(victims.size());
    promoted_held_ = static_cast<std::uint32_t>(promote_lru_.size());
    promoted_bytes_ = promote_lru_.bytes();
}

// Loader thread: for every use j of the run, once slot j % S was freed by use j - S (or holds the layer already),
// enqueue the layer's groups and post. Between runs: the first S streamed layers are loaded for the next run.
void ExpertStreamer::loop() {
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
            // the preload this run consumes: the one that followed the previous run (this thread finished it before)
            consumed_bytes_ = 0;
            consumed_s_ = consumed_after_begin_s_ = 0.0;
            if (run_id_ > 0U && preload_after_run_ == run_id_ - 1U && !preload_running_ && preload_bytes_ != 0U) {
                consumed_bytes_ = preload_bytes_;
                consumed_s_ = std::chrono::duration<double>(preload_t1_ - preload_t0_).count();
                const auto from = std::max(preload_t0_, run_begin_t_);
                consumed_after_begin_s_ = preload_t1_ > from ? std::chrono::duration<double>(preload_t1_ - from).count() : 0.0;
            }
            if (p_.host_promote.enabled) {
                lock.unlock();
                try {
                    demote_under_pressure();
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> g(mutex_);
                    error_ = e.what();
                    done_.notify_all();
                }
            }
            continue;
        }
        if (!run_) {
            idle_prefetch_done = true;
            preload_after_run_ = run_id_;
            preload_bytes_ = 0;
            preload_running_ = true;
            preload_t0_ = std::chrono::steady_clock::now();
            lock.unlock();
            std::uint64_t pre_bytes = 0;
            try {
                STRIX_CUDA_CHECK(cudaDeviceSynchronize());
                for (std::size_t j = 0; j < slots_.size(); ++j) {
                    auto& s = slots_[j];
                    if (s.layer == static_cast<std::int32_t>(streamed_[j])) continue;
                    pre_bytes += load_into(s, streamed_[j]);
                    STRIX_CUDA_CHECK(cudaStreamSynchronize(copy_));
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> g(mutex_);
                error_ = e.what();
                done_.notify_all();
            }
            {
                std::lock_guard<std::mutex> g(mutex_);
                preload_bytes_ = pre_bytes;
                preload_t1_ = std::chrono::steady_clock::now();
                preload_running_ = false;
            }
            continue;
        }
        idle_prefetch_done = false;
        const std::size_t j = use++;
        auto& s = slots_[j % slots_.size()];
        const std::uint32_t layer = uses_[j];
        const bool wait_free = j >= slots_.size();
        try {
            if (wait_free) STRIX_CUDA_CHECK(cudaStreamWaitEvent(copy_, s.free->e, 0));
            if (s.layer != static_cast<std::int32_t>(layer)) {
                const bool from_file = p_.stream_tier[layer] == ExpertTier::file;
                lock.unlock();
                const std::uint64_t b = load_into(s, layer);
                lock.lock();
                bytes_streamed_ += b;
                if (from_file) file_bytes_ += b;
                copy_bytes_[layer] = b;
            } else {
                // preloaded between runs: its group events fired long ago (record them again: same meaning)
                lock.unlock();
                for (std::uint32_t g = 0; g < groups(layer); ++g) STRIX_CUDA_CHECK(cudaEventRecord(s.group_ready[g]->e, copy_));
                lock.lock();
            }
            s.loaded_run = run_seen;
            s.loaded_use = j;
        } catch (const std::exception& e) {
            if (!lock.owns_lock()) lock.lock();
            error_ = e.what();
        }
        lock.unlock();
        done_.notify_all();
    }
}

}  // namespace strix::ds41::cuda::detail
