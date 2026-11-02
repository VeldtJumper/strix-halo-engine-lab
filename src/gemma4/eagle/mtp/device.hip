#include <hip/hip_runtime.h>
#include "device.hpp"
#include "gemma4/kernels/capture_guard.hpp"

#include <array>
#include <chrono>
#include <optional>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::gemma4::eagle::mtp {
namespace {
using gpu::Buffer;
void require(bool ok, const char* text) {
    if (!ok) throw std::runtime_error(std::string("device mtp: ") + text);
}
void checked(hipError_t e) {
    if (e != hipSuccess) throw std::runtime_error(std::string("device mtp: ") + hipGetErrorString(e));
}
void launched() { checked(hipGetLastError()); }

struct Result {
    unsigned tokens[8];
    float prob[8];
    float margin[8];
};
constexpr unsigned max_split_keys = (gemma4::max_positions + attention_splits - 1) / attention_splits;

__device__ void flag(float x, unsigned* failure) {
    if ((bit_cast<unsigned>(x) & 0x7f800000U) == 0x7f800000U) atomicOr(failure, 1U);
}
__device__ float wave_reduce(float a) {
    for (unsigned d = 16; d; d >>= 1U) a = a + __shfl_down(a, d, 32);
    return __shfl(a, 0, 32);
}
__device__ const float* kv_row(const KvLayer& kv, std::uint64_t p, bool value, unsigned* failure) {
    if (p >= kv.frontier) {
        if (!kv.overlay_k || p - kv.frontier >= kv.candidate_rows) {
            atomicOr(failure, 2U);
            return nullptr;
        }
        return (value ? kv.overlay_v : kv.overlay_k) + (p - kv.frontier) * kv.width;
    }
    const auto slot = kv.global ? p : p % kv.slots;
    if (slot >= kv.slots || (kv.tags && kv.tags[slot] != p)) {
        atomicOr(failure, 2U);
        return nullptr;
    }
    return (value ? kv.v : kv.k) + slot * kv.width;
}
struct Extent {
    std::uint64_t first, n, len;
};
__device__ Extent extent(std::uint64_t p, bool full, unsigned sliding_keys) {
    const std::uint64_t last = p - 1, first = full ? 0 : (p > sliding_keys ? p - sliding_keys : 0);
    const auto n = last - first + 1;
    return {first, n, (n + attention_splits - 1) / attention_splits};
}

// grid (splits, heads), 256 threads. part[(h*S + s)*(dim+2)]: acc[dim], m, l.
__global__ void attention_partial(const float* q, unsigned dim, unsigned group, const KvLayer* kvs, unsigned which,
                                  const std::uint64_t* position, unsigned sliding_keys, float* part, unsigned* failure) {
    __shared__ float score[max_split_keys];
    __shared__ float maximum;
    const unsigned s = blockIdx.x, h = blockIdx.y, t = threadIdx.x, lane = t % 32U, wave = t / 32U;
    const KvLayer kv = kvs[which];
    const bool full = which == 1;
    const auto e = extent(*position, full, sliding_keys);
    const auto b = e.first + s * e.len, end = min(e.first + e.n, b + e.len);
    float* out = part + (std::size_t(h) * attention_splits + s) * (dim + 2U);
    if (b >= end) return; // unused split: the merge recomputes the same extent and skips it
    const unsigned kvh = h / group;
    const float* qh = q + std::size_t(h) * dim;
    for (auto j = b + wave; j < end; j += 8U) {
        const float* k = kv_row(kv, j, false, failure);
        float sum = 0.0F;
        if (k)
            for (unsigned i = lane; i < dim; i += 32U) sum = sum + qh[i] * k[std::size_t(kvh) * dim + i];
        sum = wave_reduce(sum);
        if (lane == 0) { flag(sum, failure); score[j - b] = sum; }
    }
    __syncthreads();
    const auto n = unsigned(end - b);
    if (t == 0) {
        float m = bit_cast<float>(0xff800000U);
        for (unsigned j = 0; j < n; ++j) m = score[j] > m ? score[j] : m;
        maximum = m;
    }
    __syncthreads();
    for (unsigned j = t; j < n; j += blockDim.x) score[j] = declared_exp(score[j] - maximum);
    __syncthreads();
    if (t == 0) {
        float l = 0.0F;
        for (unsigned j = 0; j < n; ++j) l = l + score[j];
        out[dim] = maximum;
        out[dim + 1U] = l;
    }
    for (unsigned d = t; d < dim; d += blockDim.x) {
        float a = 0.0F;
        for (unsigned j = 0; j < n; ++j) {
            const float* v = kv_row(kv, b + j, true, failure);
            if (v) a = a + score[j] * v[std::size_t(kvh) * dim + d];
        }
        out[d] = a;
    }
}
// grid heads, 256 threads.
__global__ void attention_merge(const float* part, unsigned dim, unsigned which, const std::uint64_t* position,
                                unsigned sliding_keys, float* att, unsigned* failure) {
    const unsigned h = blockIdx.x;
    const auto e = extent(*position, which == 1, sliding_keys);
    const float* base = part + std::size_t(h) * attention_splits * (dim + 2U);
    float M = bit_cast<float>(0xff800000U);
    for (unsigned s = 0; s < attention_splits; ++s)
        if (e.first + s * e.len < e.first + e.n) M = base[s * (dim + 2U) + dim] > M ? base[s * (dim + 2U) + dim] : M;
    float f[attention_splits];
    float L = 0.0F;
    for (unsigned s = 0; s < attention_splits; ++s) {
        f[s] = 0.0F;
        if (e.first + s * e.len < e.first + e.n) {
            f[s] = declared_exp(base[s * (dim + 2U) + dim] - M);
            L = L + base[s * (dim + 2U) + dim + 1U] * f[s];
        }
    }
    for (unsigned d = threadIdx.x; d < dim; d += blockDim.x) {
        float num = 0.0F;
        for (unsigned s = 0; s < attention_splits; ++s)
            if (e.first + s * e.len < e.first + e.n) num = num + base[s * (dim + 2U) + d] * f[s];
        const float value = num / L;
        flag(value, failure);
        att[std::size_t(h) * dim + d] = value;
    }
}
// One block of 1024: k rounds of argmax over not-yet-taken clusters (higher value, lower index).
__global__ void __launch_bounds__(1024) topk_kernel(const float* logits, unsigned count, unsigned k, unsigned* top,
                                                    unsigned* failure) {
    __shared__ bool taken[4096];
    __shared__ float bv[1024];
    __shared__ unsigned bi[1024];
    const unsigned t = threadIdx.x;
    for (unsigned i = t; i < count; i += 1024U) { taken[i] = false; flag(logits[i], failure); }
    __syncthreads();
    for (unsigned r = 0; r < k; ++r) {
        float value = bit_cast<float>(0xff800000U);
        unsigned id = UINT32_MAX;
        for (unsigned i = t; i < count; i += 1024U)
            if (!taken[i] && better(logits[i], i, value, id)) { value = logits[i]; id = i; }
        bv[t] = value; bi[t] = id;
        __syncthreads();
        for (unsigned s = 512; s; s >>= 1U) {
            if (t < s && better(bv[t + s], bi[t + s], bv[t], bi[t])) { bv[t] = bv[t + s]; bi[t] = bi[t + s]; }
            __syncthreads();
        }
        if (t == 0) { top[r] = bi[0]; if (bi[0] < count) taken[bi[0]] = true; }
        __syncthreads();
    }
}
// One wave per candidate (8 per 256-thread block): z = lane_dot(a, table[id]).
__global__ void candidates_kernel(const float* a, LinearView table, const unsigned* ordering, const unsigned* top,
                                  unsigned per, unsigned width, float* z, unsigned* ids) {
    const unsigned c = blockIdx.x * 8U + threadIdx.x / 32U, lane = threadIdx.x % 32U;
    const unsigned id = ordering[std::size_t(top[c / per]) * per + c % per];
    float sum = 0.0F;
    for (unsigned i = lane; i < width; i += 32U) sum = sum + a[i] * gpu::coefficient(table, id, i);
    sum = wave_reduce(sum);
    if (lane == 0) { z[c] = sum; ids[c] = id; }
}
__global__ void __launch_bounds__(1024) select_kernel(const float* z, const unsigned* ids, unsigned n, unsigned depth,
                                                      unsigned* token, Result* result, unsigned* failure) {
    __shared__ float bv[1024], sv[1024], sum[1024];
    __shared__ unsigned bi[1024];
    const unsigned t = threadIdx.x;
    float value = bit_cast<float>(0xff800000U);
    unsigned id = UINT32_MAX;
    for (unsigned i = t; i < n; i += 1024U) {
        flag(z[i], failure);
        if (better(z[i], ids[i], value, id)) { value = z[i]; id = ids[i]; }
    }
    bv[t] = value; bi[t] = id;
    __syncthreads();
    for (unsigned s = 512; s; s >>= 1U) {
        if (t < s && better(bv[t + s], bi[t + s], bv[t], bi[t])) { bv[t] = bv[t + s]; bi[t] = bi[t + s]; }
        __syncthreads();
    }
    const float best = bv[0];
    const unsigned winner = bi[0];
    float second = bit_cast<float>(0xff800000U), acc = 0.0F;
    for (unsigned i = t; i < n; i += 1024U) {
        if (ids[i] != winner) second = z[i] > second ? z[i] : second;
        acc = acc + declared_exp(z[i] - best);
    }
    sv[t] = second; sum[t] = acc;
    __syncthreads();
    for (unsigned s = 512; s; s >>= 1U) {
        if (t < s) { sv[t] = sv[t + s] > sv[t] ? sv[t + s] : sv[t]; sum[t] = sum[t] + sum[t + s]; }
        __syncthreads();
    }
    if (t == 0) {
        *token = winner;
        result->tokens[depth] = winner;
        result->prob[depth] = 1.0F / sum[0];
        result->margin[depth] = best - sv[0];
    }
}
} // namespace

// ------------------------------------------------------------------ model
struct DeviceModel::Impl {
    Config c;
    LinearView embedding;
    std::vector<Buffer> store;
    std::size_t bytes = 0;
    struct L {
        LinearView q, o, gate, up, down, input_norm, q_norm, post_attn, pre_ffn, post_ffn, scalar;
    };
    std::array<L, layers> layer;
    LinearView pre, post, centroids, table, final_norm;
    const unsigned* ordering = nullptr;
    const float *sliding = nullptr, *global = nullptr;
    std::uint32_t positions = 0;
    const std::uint8_t* raw(const void* data, std::size_t size) {
        store.emplace_back(size);
        store.back().upload(data, size);
        bytes += size;
        return store.back().as<std::uint8_t>();
    }
    LinearView matrix(const Matrix& m) { return {raw(m.bytes.data(), m.bytes.size()), m.outputs, m.inputs, m.row_bytes(), m.format}; }
    LinearView vec(const std::vector<float>& v) {
        return {raw(v.data(), v.size() * 4), 1, std::uint32_t(v.size()), v.size() * 4, WeightFormat::f32};
    }
};
DeviceModel::DeviceModel(const Weights& w, Tables t, LinearView target_embedding) : impl_(std::make_unique<Impl>()) {
    gpu::require_device();
    w.validate();
    t.validate();
    auto& d = *impl_;
    d.c = w.config;
    require(target_embedding.data && target_embedding.inputs == d.c.backbone && target_embedding.outputs >= d.c.vocab,
            "target embedding geometry");
    d.embedding = target_embedding;
    d.pre = d.matrix(w.pre_projection); d.post = d.matrix(w.post_projection);
    d.centroids = d.matrix(w.centroids); d.table = d.matrix(w.table); d.final_norm = d.vec(w.final_norm);
    for (std::uint32_t l = 0; l < layers; ++l) {
        const auto& y = w.layer[l];
        auto& z = d.layer[l];
        z.q = d.matrix(y.q); z.o = d.matrix(y.o); z.gate = d.matrix(y.gate); z.up = d.matrix(y.up); z.down = d.matrix(y.down);
        z.input_norm = d.vec(y.input_norm); z.q_norm = d.vec(y.q_norm); z.post_attn = d.vec(y.post_attn_norm);
        z.pre_ffn = d.vec(y.pre_ffn_norm); z.post_ffn = d.vec(y.post_ffn_norm); z.scalar = d.vec({y.scalar});
    }
    d.ordering = reinterpret_cast<const unsigned*>(d.raw(w.ordering.data(), w.ordering.size() * 4));
    d.sliding = reinterpret_cast<const float*>(d.raw(t.sliding.data(), t.sliding.size() * 4));
    d.global = reinterpret_cast<const float*>(d.raw(t.global.data(), t.global.size() * 4));
    d.positions = t.positions;
}
DeviceModel::~DeviceModel() = default;
const Config& DeviceModel::config() const { return impl_->c; }
std::size_t DeviceModel::resident_bytes() const { return impl_->bytes; }
std::size_t DeviceModel::step_weight_bytes() const {
    const auto& d = *impl_;
    auto b = [](LinearView v) { return std::size_t(v.outputs) * v.row_bytes; };
    std::size_t total = b(d.pre) + b(d.post) + b(d.centroids) + std::size_t(d.c.candidates()) * d.table.row_bytes + d.embedding.row_bytes;
    for (const auto& y : d.layer) total += b(y.q) + b(y.o) + b(y.gate) + b(y.up) + b(y.down);
    return total;
}

// ------------------------------------------------------------------ drafter
struct DeviceDrafter::Impl {
    std::shared_ptr<const DeviceModel> model;
    const DeviceModel::Impl* m = nullptr;
    Config c;
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    Buffer cat, x, a, y, o, q, att, g, cl, top, z, ids, part, kv, pos, token, result, failure;
    // Graphs per depth segment [from, to): draft(k) replays [0, k), extend(k) replays [depth, k).
    std::array<std::array<hipGraphExec_t, max_depth + 1>, max_depth + 1> graphs{};
    std::array<std::array<hipGraph_t, max_depth + 1>, max_depth + 1> sources{};
    bool use_graphs = true;
    std::uint32_t chain = 0; // depth of the chain held in the device buffers (0: none / invalid)
    Timings times;
    ~Impl() {
        for (auto& row : graphs) for (auto& e : row) if (e) (void)hipGraphExecDestroy(e);
        for (auto& row : sources) for (auto& g0 : row) if (g0) (void)hipGraphDestroy(g0);
        if (start) (void)hipEventDestroy(start);
        if (stop) (void)hipEventDestroy(stop);
        if (stream) (void)hipStreamDestroy(stream);
    }
    unsigned* fail() const { return failure.as<unsigned>(); }
    void count() { ++times.launches; }
    // Depths [from, to) of the chain. The depth index only selects select_kernel's result slot; every other input of a
    // depth (token, cat backbone half, p, KV descriptors) is device state left by the previous depth, so a segment
    // [d, k) after [0, d) computes exactly what [0, k) does.
    void enqueue(std::uint32_t from, std::uint32_t to) {
        const auto H = c.hidden, B = c.backbone;
        const gpu::Queue qu = stream;
        for (std::uint32_t depth = from; depth < to; ++depth) {
            gpu::embedding(m->embedding, token.as<unsigned>(), 1, cat.as<float>(), c.embedding_scale, qu); count();
            gpu::linear(m->pre, cat.as<float>(), 1, x.as<float>(), true, qu, fail()); count();
            gpu::norm(x.as<float>(), m->layer[0].input_norm, a.as<float>(), 1, H, qu, fail()); count();
            for (std::uint32_t l = 0; l < layers; ++l) {
                const auto& L = m->layer[l];
                const bool full = c.full[l];
                const auto dim = c.head_dim(l);
                gpu::linear(L.q, a.as<float>(), 1, q.as<float>(), true, qu, fail()); count();
                gpu::head_norm_rope(q.as<float>(), L.q_norm, 1, c.heads, dim, pos.as<std::uint64_t>(), full ? m->global : m->sliding,
                                    full, qu, fail()); count();
                hipLaunchKernelGGL(attention_partial, dim3(attention_splits, c.heads), dim3(256), 0, stream, q.as<float>(), dim,
                                   c.heads / c.kv_heads, kv.as<KvLayer>(), full ? 1U : 0U, pos.as<std::uint64_t>(), c.sliding_keys,
                                   part.as<float>(), fail());
                launched(); count();
                hipLaunchKernelGGL(attention_merge, dim3(c.heads), dim3(256), 0, stream, part.as<float>(), dim, full ? 1U : 0U,
                                   pos.as<std::uint64_t>(), c.sliding_keys, att.as<float>(), fail());
                launched(); count();
                gpu::linear(L.o, att.as<float>(), 1, o.as<float>(), true, qu, fail()); count();
                gpu::norm_residual(o.as<float>(), L.post_attn, x.as<float>(), 1, H, {}, y.as<float>(), L.pre_ffn, fail(), qu); count();
                gpu::gate_up(L.gate, L.up, y.as<float>(), 1, g.as<float>(), true, qu, fail()); count();
                gpu::linear(L.down, g.as<float>(), 1, o.as<float>(), true, qu, fail()); count();
                gpu::norm_residual(o.as<float>(), L.post_ffn, x.as<float>(), 1, H, L.scalar, a.as<float>(),
                                   l + 1 < layers ? m->layer[l + 1].input_norm : m->final_norm, fail(), qu); count();
            }
            gpu::linear(m->post, a.as<float>(), 1, cat.as<float>() + B, true, qu, fail()); count();
            gpu::linear(m->centroids, a.as<float>(), 1, cl.as<float>(), true, qu, fail()); count();
            hipLaunchKernelGGL(topk_kernel, dim3(1), dim3(1024), 0, stream, cl.as<float>(), c.centroids, c.top_k, top.as<unsigned>(), fail());
            launched(); count();
            hipLaunchKernelGGL(candidates_kernel, dim3(c.candidates() / 8U), dim3(256), 0, stream, a.as<float>(), m->table, m->ordering,
                               top.as<unsigned>(), c.per_cluster(), H, z.as<float>(), ids.as<unsigned>());
            launched(); count();
            hipLaunchKernelGGL(select_kernel, dim3(1), dim3(1024), 0, stream, z.as<float>(), ids.as<unsigned>(), c.candidates(), depth,
                               token.as<unsigned>(), result.as<Result>(), fail());
            launched(); count();
        }
    }
    // Runs [from, to) on the stream (graph capture on first use, replay after) between the start/stop events, then
    // one sync and one download of all result slots.
    Draft run(std::uint32_t from, std::uint32_t to) {
        chain = 0;
        checked(hipMemsetAsync(failure.data(), 0, sizeof(unsigned), stream));
        checked(hipEventRecord(start, stream));
        times.launches = 0;
        times.graph = use_graphs;
        // A new capture waits for no prefill pass in flight (gpu::CaptureSlot); otherwise this window runs eagerly.
        std::optional<gpu::CaptureSlot> slot;
        if (use_graphs && !graphs[from][to]) {
            slot.emplace();
            if (!slot->granted()) slot.reset();
        }
        if (use_graphs && (graphs[from][to] || slot)) {
            auto& exec = graphs[from][to];
            auto& counters = gpu::capture_guard_state();
            if (!exec) {
                const auto capture_begin = std::chrono::steady_clock::now();
                checked(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
                try { enqueue(from, to); } catch (...) { hipGraph_t g0 = nullptr; (void)hipStreamEndCapture(stream, &g0); if (g0) (void)hipGraphDestroy(g0); throw; }
                checked(hipStreamEndCapture(stream, &sources[from][to]));
                slot.reset();
                checked(hipGraphInstantiate(&exec, sources[from][to], nullptr, nullptr, 0));
                counters.draft_captures.fetch_add(1);
                counters.draft_capture_ns.fetch_add(std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - capture_begin).count()));
            } else {
                counters.draft_replays.fetch_add(1);
                times.launches = 44ULL * (to - from); // replayed graph nodes
            }
            checked(hipGraphLaunch(exec, stream));
        } else {
            enqueue(from, to);
        }
        checked(hipEventRecord(stop, stream));
        Result r{};
        unsigned failed = 0;
        checked(hipMemcpyAsync(&r, result.data(), sizeof(r), hipMemcpyDeviceToHost, stream));
        checked(hipMemcpyAsync(&failed, failure.data(), sizeof(failed), hipMemcpyDeviceToHost, stream));
        checked(hipStreamSynchronize(stream));
        if (failed) throw std::runtime_error("device mtp: draft failed (nonfinite=1, missing target KV tag=2): " + std::to_string(failed));
        float ms = 0;
        checked(hipEventElapsedTime(&ms, start, stop));
        times.draft_ms = ms;
        Draft out;
        out.depth = to;
        for (std::uint32_t i = 0; i < to; ++i) { out.tokens[i] = r.tokens[i]; out.prob[i] = r.prob[i]; out.margin[i] = r.margin[i]; }
        chain = to;
        return out;
    }
};

DeviceDrafter::DeviceDrafter(std::shared_ptr<const DeviceModel> model) : impl_(std::make_unique<Impl>()) {
    require(bool(model), "missing model");
    auto& d = *impl_;
    d.model = std::move(model);
    d.m = d.model->impl_.get();
    d.c = d.m->c;
    const auto& c = d.c;
    require(c.centroids <= 4096 && c.candidates() % 8 == 0 && c.candidates() <= 65536, "head geometry");
    const auto* env = std::getenv("STRIX_G4_MTP_GRAPHS");
    d.use_graphs = !env || std::string_view(env) != "0";
    checked(hipStreamCreateWithFlags(&d.stream, hipStreamNonBlocking));
    checked(hipEventCreate(&d.start));
    checked(hipEventCreate(&d.stop));
    auto f = [](Buffer& b, std::size_t n) { b.resize(n * 4); b.zero(); };
    f(d.cat, 2ULL * c.backbone); f(d.x, c.hidden); f(d.a, c.hidden); f(d.y, c.hidden); f(d.o, c.hidden);
    f(d.q, std::size_t(c.heads) * c.full_dim); f(d.att, std::size_t(c.heads) * c.full_dim); f(d.g, c.ffn); f(d.cl, c.centroids);
    f(d.top, c.top_k); f(d.z, c.candidates()); f(d.ids, c.candidates());
    f(d.part, std::size_t(c.heads) * attention_splits * (c.full_dim + 2));
    d.kv.resize(2 * sizeof(KvLayer)); d.pos.resize(sizeof(std::uint64_t)); d.token.resize(sizeof(unsigned));
    d.result.resize(sizeof(Result)); d.failure.resize(sizeof(unsigned));
}
DeviceDrafter::~DeviceDrafter() = default;
const Config& DeviceDrafter::config() const { return impl_->c; }
const Timings& DeviceDrafter::timings() const { return impl_->times; }

Draft DeviceDrafter::draft(const Inputs& in, std::uint32_t k) {
    auto& d = *impl_;
    const auto& c = d.c;
    const auto p = in.position;
    require(k >= 1 && k <= max_depth && p >= 1 && p < d.m->positions && in.hidden.size() == c.backbone && in.anchor < c.vocab,
            "draft inputs");
    require(in.sliding.k && in.sliding.v && in.full.k && in.full.v && in.sliding.slots && in.full.slots &&
            in.sliding.width >= c.kv_heads * c.sliding_dim && in.full.width >= c.kv_heads * c.full_dim, "target KV views");
    const KvLayer kvs[2] = {in.sliding, in.full};
    const std::uint32_t anchor = in.anchor;
    d.chain = 0;
    checked(hipMemcpyAsync(d.cat.as<float>() + c.backbone, in.hidden.data(), c.backbone * 4ULL, hipMemcpyHostToDevice, d.stream));
    checked(hipMemcpyAsync(d.kv.data(), kvs, sizeof(kvs), hipMemcpyHostToDevice, d.stream));
    checked(hipMemcpyAsync(d.pos.data(), &p, sizeof(p), hipMemcpyHostToDevice, d.stream));
    checked(hipMemcpyAsync(d.token.data(), &anchor, sizeof(anchor), hipMemcpyHostToDevice, d.stream));
    return d.run(0, k);
}
Draft DeviceDrafter::extend(std::uint32_t k) {
    auto& d = *impl_;
    require(d.chain && k > d.chain && k <= max_depth, "extend needs a previous shallower draft");
    return d.run(d.chain, k);
}

} // namespace strix::gemma4::eagle::mtp
