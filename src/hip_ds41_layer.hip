#include "strix/ds41_prefill_tuning.hpp"
#include "strix/hip_ds41_layer.hpp"
#include "strix/ds41_exact_flags.hpp"
#include "strix/dist/staged_ready.hpp"
#include "strix/hip_ds41_f16_rows.hpp"
#include "strix/hip_ds41_rn_table.hpp"
#include "strix/hip_ds41_rn_wide.hpp"
#include "strix/hip_ds41_f8_rows_fast.hpp"

#include "strix/hip_ds41_formats.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// Pointers that never alias are declared __restrict__ (kernel parameters, or the parameters of the
// force-inlined bodies of the view kernels, since restrict on struct members is not used by the
// compiler): without it the loads of a later pass cannot move above the stores of an earlier one and
// small kernels serialise on load -> wait -> store round trips.

namespace strix::ds41 {
namespace {

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
hipStream_t as_stream(void* s) { return static_cast<hipStream_t>(s); }
bool overlap(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) {
    if (!a || !b || !a_bytes || !b_bytes) return false;
    const auto pa = reinterpret_cast<std::uintptr_t>(a), pb = reinterpret_cast<std::uintptr_t>(b);
    return pa < pb + b_bytes && pb < pa + a_bytes;
}

constexpr unsigned wave = 32;
constexpr unsigned max_rows = 8;

__device__ inline float butterfly(float v) {
    for (int d = 16; d; d >>= 1) v = v + __shfl_xor(v, d, wave);
    return v;
}

// ---- hyper-connections ---------------------------------------------------------------------------
constexpr unsigned chunk = dot_chunk;    // elements per wave chunk of the long dots
constexpr unsigned chunk_k = chunk / 32; // values per lane in a chunk
constexpr unsigned hc_tile = 8;          // rows per mix-partial block
constexpr unsigned mix_waves = hc_mix_count + 1;

// Copy k of element i of row r after the expansion of the previous sublayer (as hc_expand):
// post[k] * sub[i] + sum_j comb[j][k] * residual[j][i], the sum from 0 in ascending j.
__device__ __forceinline__ float copy_value(const float* __restrict__ sub, const float* __restrict__ residual,
                                            const HcMix* __restrict__ em, unsigned r, unsigned k, unsigned i, unsigned hidden) {
    const float* res = residual + static_cast<std::size_t>(r) * hc_copies * hidden;
    const HcMix& m = em[r];
    float s = 0.0F;
#pragma unroll
    for (unsigned j = 0; j < hc_copies; ++j) s = s + m.comb[j * hc_copies + k] * res[j * hidden + i];
    return m.post[k] * sub[static_cast<std::size_t>(r) * hidden + i] + s;
}
// All copies of element i of row r: expanded (sub not null) or read from the stored stream.
__device__ __forceinline__ void site_copies(const float* __restrict__ sub, const float* __restrict__ residual,
                                            const HcMix* __restrict__ em, const float* __restrict__ stream, unsigned r, unsigned i,
                                            unsigned hidden, float* e) {
    const std::size_t row = static_cast<std::size_t>(r) * hc_copies * hidden;
    if (!sub) {
#pragma unroll
        for (unsigned k = 0; k < hc_copies; ++k) e[k] = stream[row + k * hidden + i];
        return;
    }
#pragma unroll
    for (unsigned k = 0; k < hc_copies; ++k) e[k] = copy_value(sub, residual, em, r, k, i, hidden);
}

// Mix partials of chunk c for the rows of a tile: wave m < 24 the chunk's lane dot of fn row m with the
// stream, wave 24 the chunk's sum of squares; the wave's fn values stay in registers over the tile (many
// rows). partial[(r * 25 + m) * chunks + c].
__device__ __forceinline__ void mix_partial_block(const float* __restrict__ stream, const std::uint16_t* __restrict__ fn,
                                                  float* __restrict__ partial, unsigned hidden, unsigned rows, unsigned chunks,
                                                  unsigned c, unsigned r0) {
    const unsigned m = threadIdx.x / wave, l = threadIdx.x % wave, n = hc_copies * hidden;
    const unsigned c0 = c * chunk, len = c0 + chunk < n ? chunk : n - c0, k_count = len / wave;
    const bool squares = m == hc_mix_count;
    float f[chunk_k];
#pragma unroll
    for (unsigned k = 0; k < chunk_k; ++k)
        f[k] = !squares && k < k_count ? f16_to_float(fn[static_cast<std::size_t>(m) * n + c0 + k * wave + l]) : 0.0F;
    for (unsigned r = r0; r < r0 + hc_tile && r < rows; ++r) {
        const float* x = stream + static_cast<std::size_t>(r) * n + c0 + l;
        float xv[chunk_k];
#pragma unroll
        for (unsigned k = 0; k < chunk_k; ++k) xv[k] = k < k_count ? x[k * wave] : 0.0F;
        float acc = 0.0F;
#pragma unroll
        for (unsigned k = 0; k < chunk_k; ++k)
            if (k < k_count) acc = acc + (squares ? xv[k] : f[k]) * xv[k];
        acc = butterfly(acc);
        if (l == 0) partial[(static_cast<std::size_t>(r) * mix_waves + m) * chunks + c] = acc;
    }
}

__global__ __launch_bounds__(32 * mix_waves) void hc_mix_partial_kernel(const float* __restrict__ stream,
                                                                        const std::uint16_t* __restrict__ fn, unsigned hidden,
                                                                        unsigned rows, unsigned chunks, float* __restrict__ partial) {
    mix_partial_block(stream, fn, partial, hidden, rows, chunks, blockIdx.x, blockIdx.y * hc_tile);
}

// One wave per row: the ascending chunk sums (lane t <= 24 holds dot t), the RMS factor, the split and
// Sinkhorn (hc_split_sinkhorn). Lane t (and t + 16) holds comb entry (j, k) = (t / 4, t % 4); every row or
// column sum is gathered from its four lanes and added in ascending order from 0, as on the host.
__device__ __forceinline__ void mix_final_wave(const float* __restrict__ partial, const float* __restrict__ base,
                                               const float* __restrict__ scale, unsigned n, unsigned chunks, unsigned iters,
                                               float hc_eps, float norm_eps, HcMix* __restrict__ out, unsigned r) {
    const unsigned t = threadIdx.x % wave;
    float d = 0.0F;
    if (t < mix_waves) {
        const float* p = partial + (static_cast<std::size_t>(r) * mix_waves + t) * chunks;
        for (unsigned c = 0; c < chunks; ++c) d = d + p[c];
    }
    constexpr unsigned h = hc_copies;
    const float rr = declared_rsqrt(__shfl(d, static_cast<int>(hc_mix_count), wave) / static_cast<float>(n) + norm_eps);
    const unsigned e = t & 15U, j = e / h, k = e % h;
    HcMix& o = out[r];
    if (t < h) o.pre[t] = declared_sigmoid((d * rr) * scale[0] + base[t]) + hc_eps;
    else if (t < 2 * h) o.post[t - h] = 2.0F * declared_sigmoid((d * rr) * scale[1] + base[t]);
    float v = (__shfl(d, static_cast<int>(2 * h + e), wave) * rr) * scale[2] + base[2 * h + e];
    auto at = [&](float x, unsigned jj, unsigned kk) { return __shfl(x, static_cast<int>(jj * h + kk), wave); };
    float mx = at(v, j, 0);
    for (unsigned kk = 1; kk < h; ++kk) { const float y = at(v, j, kk); mx = y > mx ? y : mx; }
    v = declared_exp(v - mx);
    float sum = 0.0F;
    for (unsigned kk = 0; kk < h; ++kk) sum = sum + at(v, j, kk);
    v = v / sum + hc_eps;
    for (unsigned it = 0; it < iters; ++it) {
        if (it) {
            float rs = 0.0F;
            for (unsigned kk = 0; kk < h; ++kk) rs = rs + at(v, j, kk);
            v = v / (rs + hc_eps);
        }
        float cs = 0.0F;
        for (unsigned jj = 0; jj < h; ++jj) cs = cs + at(v, jj, k);
        v = v / (cs + hc_eps);
    }
    if (t < 16) o.comb[t] = v;
}

__global__ __launch_bounds__(32) void hc_mix_final_kernel(const float* __restrict__ partial, const float* __restrict__ base,
                                                          const float* __restrict__ scale, unsigned n, unsigned chunks, unsigned iters,
                                                          float hc_eps, float norm_eps, HcMix* __restrict__ out) {
    mix_final_wave(partial, base, scale, n, chunks, iters, hc_eps, norm_eps, out, blockIdx.x);
}

// A8 of one 32-value group held one value per lane (quantize_a8 bitwise): the order-free maximum of
// |y| (NaN ignored, from 0), delta = amax / 127 (1 for a zero or tiny group), code = rint(y / delta)
// clamped to +-127 in quantize_a8's max/min form, and the exact integer sum.
__device__ inline void a8_group(float y, unsigned l, std::int8_t* code, float* delta, std::int32_t* sum) {
    float amax = fmaxf(0.0F, fabsf(y));
    for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor(amax, d, wave));
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    const float dl = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
    float rounded = tiny ? 0.0F : __builtin_rintf(y / dl);
    rounded = -127.0F < rounded ? rounded : -127.0F;  // std::max(-127, rounded)
    rounded = rounded < 127.0F ? rounded : 127.0F;    // std::min(127, .)
    const int c = static_cast<int>(rounded);
    int s = c;
    for (int d = 16; d; d >>= 1) s = s + __shfl_xor(s, d, wave);
    *code = static_cast<std::int8_t>(c);
    if (l == 0) { *delta = dl; *sum = s; }
}

__global__ __launch_bounds__(256) void hc_collapse_kernel(const float* __restrict__ stream, const HcMix* __restrict__ mixes,
                                                          unsigned hidden, unsigned long long total, float* __restrict__ out) {
    const auto index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const auto r = index / hidden, i = index % hidden;
    const float* x = stream + r * hc_copies * hidden;
    float y = 0.0F;
    for (unsigned c = 0; c < hc_copies; ++c) {
        const float p = mixes ? mixes[r].pre[c] : (c == 0 ? 1.0F : 0.0F);
        y = y + p * x[c * hidden + i];
    }
    out[index] = y;
}

__global__ __launch_bounds__(256) void hc_expand_kernel(const float* __restrict__ sub, const float* __restrict__ residual,
                                                        const HcMix* __restrict__ mixes, unsigned hidden, unsigned long long total,
                                                        float* __restrict__ out) {
    const auto index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;  // total = rows * hc * hidden
    const auto per_row = static_cast<unsigned long long>(hc_copies) * hidden;
    const auto r = static_cast<unsigned>(index / per_row), rest = static_cast<unsigned>(index % per_row);
    out[index] = copy_value(sub, residual, mixes, r, rest / hidden, rest % hidden, hidden);
}

// ---- RMSNorm ----------------------------------------------------------------------------------------
// One block per row, one wave per chunk; every value is read before the barrier and written after it, so
// x == out (in place) is allowed: x and out are not restrict here.
__device__ __forceinline__ float norm_input(const float* x, unsigned x_stride, const float* __restrict__ hc_stream, unsigned n,
                                            unsigned r, unsigned i, const float* pre) {
    if (!hc_stream) return x[static_cast<std::size_t>(r) * x_stride + i];
    const float* s = hc_stream + static_cast<std::size_t>(r) * hc_copies * n + i;
    float v = 0.0F;
#pragma unroll
    for (unsigned j = 0; j < hc_copies; ++j) v = v + pre[j] * s[static_cast<std::size_t>(j) * n];
    return v;
}
// Normalised value i of row r: weight * (x * rr), then the optional RoPE of the tail (pairs on adjacent
// lanes), stored to out and/or quantised (the group of 32 is the wave).
__device__ __forceinline__ void norm_emit_weight(const NormView& v, float* out, std::int8_t* __restrict__ codes, float* __restrict__ deltas,
                                                 std::int32_t* __restrict__ sums, unsigned r, unsigned i, unsigned group, float x, float rr,
                                                 float weight) {
    const unsigned l = threadIdx.x % wave;
    float y = weight * (x * rr);
    if (v.pairs) {
        const float partner = __shfl_xor(y, 1, wave);
        if (i >= v.rope_offset && i < v.rope_offset + 2U * v.pairs) {
            const std::size_t pos = static_cast<std::size_t>(v.positions[r]) * v.pairs;
            const unsigned j = i - v.rope_offset, p = j / 2U;
            const float cs = v.rope_cos[pos + p], sn = v.rope_sin[pos + p];
            y = (j & 1U) ? partner * sn + y * cs : y * cs - partner * sn;
        }
    }
    if (out) out[static_cast<std::size_t>(r) * v.out_stride + i] = y;
    if (v.f16) v.f16[static_cast<std::size_t>(r) * v.f16_stride + i] = __half_as_ushort(__float2half_rn(y));
    if (codes) {
        const std::size_t g = static_cast<std::size_t>(r) * (v.a8_stride / 32U) + group;
        a8_group(y, l, codes + static_cast<std::size_t>(r) * v.a8_stride + i, deltas + g, sums + g);
    }
}
__device__ __forceinline__ void norm_emit(const NormView& v, float* out, std::int8_t* __restrict__ codes, float* __restrict__ deltas,
                                          std::int32_t* __restrict__ sums, unsigned r, unsigned i, unsigned group, float x, float rr) {
    norm_emit_weight(v, out, codes, deltas, sums, r, i, group, x, rr, v.weight[i]);
}
__device__ __forceinline__ void collapse_pre(const HcMix* __restrict__ mixes, unsigned r, float* pre) {
#pragma unroll
    for (unsigned j = 0; j < hc_copies; ++j) pre[j] = mixes ? mixes[r].pre[j] : (j == 0 ? 1.0F : 0.0F);
}

__global__ __launch_bounds__(1024) void norm_kernel(NormView v) {
    __shared__ float sums[32];
    const unsigned c = threadIdx.x / wave, l = threadIdx.x % wave, r = blockIdx.x, n = v.n;
    const unsigned chunks = blockDim.x / wave, c0 = c * chunk, len = c0 + chunk < n ? chunk : n - c0, k_count = len / wave;
    float y[chunk_k];
    float pre[hc_copies];
    collapse_pre(v.mixes, r, pre);
    float acc = 0.0F;
#pragma unroll
    for (unsigned k = 0; k < chunk_k; ++k)
        if (k < k_count) {
            y[k] = norm_input(v.x, v.x_stride, v.hc_stream, n, r, c0 + k * wave + l, pre);
            acc = acc + y[k] * y[k];
        }
    acc = butterfly(acc);
    if (l == 0) sums[c] = acc;
    __syncthreads();
    float total = 0.0F;
    for (unsigned j = 0; j < chunks; ++j) total = total + sums[j];
    const float rr = declared_rsqrt(total / static_cast<float>(n) + v.eps);
#pragma unroll
    for (unsigned k = 0; k < chunk_k; ++k)
        if (k < k_count) norm_emit(v, v.out, v.codes, v.deltas, v.sums, r, c0 + k * wave + l, c0 / 32U + k, y[k], rr);
}

// STRIX_DS41_NORM_ILP (default on, =0 off): norm_kernel's values, sums (same order) and stores, with the loads
// gathered: every x value of the lane's chunk is read before the first add and every weight before the first store (in
// place, x == out, the compiler may not move a weight load past a store). One row at q_lora_rank 1280 is two waves
// (chunk 640) and norm_kernel waited on its 20 x loads and 20 weight loads one by one: ~18 us a launch for 5 KB
// (r1fuse D1 trace), on the R1 critical path once per layer.
__global__ __launch_bounds__(1024) void norm_ilp_kernel(NormView v) {
    __shared__ float sums[32];
    const unsigned c = threadIdx.x / wave, l = threadIdx.x % wave, r = blockIdx.x, n = v.n;
    const unsigned chunks = blockDim.x / wave, c0 = c * chunk, len = c0 + chunk < n ? chunk : n - c0, k_count = len / wave;
    float y[chunk_k], w[chunk_k];
    float pre[hc_copies];
    collapse_pre(v.mixes, r, pre);
    if (k_count == chunk_k) {
#pragma unroll
        for (unsigned k = 0; k < chunk_k; ++k) y[k] = norm_input(v.x, v.x_stride, v.hc_stream, n, r, c0 + k * wave + l, pre);
#pragma unroll
        for (unsigned k = 0; k < chunk_k; ++k) w[k] = v.weight[c0 + k * wave + l];
    } else {
#pragma unroll
        for (unsigned k = 0; k < chunk_k; ++k)
            if (k < k_count) y[k] = norm_input(v.x, v.x_stride, v.hc_stream, n, r, c0 + k * wave + l, pre);
#pragma unroll
        for (unsigned k = 0; k < chunk_k; ++k)
            if (k < k_count) w[k] = v.weight[c0 + k * wave + l];
    }
    float acc = 0.0F;
#pragma unroll
    for (unsigned k = 0; k < chunk_k; ++k)
        if (k < k_count) acc = acc + y[k] * y[k];
    acc = butterfly(acc);
    if (l == 0) sums[c] = acc;
    __syncthreads();
    float total = 0.0F;
    for (unsigned j = 0; j < chunks; ++j) total = total + sums[j];
    const float rr = declared_rsqrt(total / static_cast<float>(n) + v.eps);
#pragma unroll
    for (unsigned k = 0; k < chunk_k; ++k)
        if (k < k_count) norm_emit_weight(v, v.out, v.codes, v.deltas, v.sums, r, c0 + k * wave + l, c0 / 32U + k, y[k], rr, w[k]);
}
bool norm_ilp_enabled() {
    return norm_ilp();  // default on, STRIX_DS41_NORM_ILP=0 off (ds41_exact_flags.hpp)
}

// Few rows, out not overlapping the input: block (c, r) recomputes the row's sum of squares exactly as
// norm_kernel (one wave per chunk, same order, same bits) and writes only chunk c, its 32-value groups
// spread over the block's waves; the row's work runs on `chunks` CUs instead of one.
__device__ __forceinline__ void norm_split_body(const NormView& v, const float* __restrict__ x, const float* __restrict__ hc_stream,
                                                float* __restrict__ out, std::int8_t* __restrict__ codes, float* __restrict__ deltas,
                                                std::int32_t* __restrict__ sums_out) {
    __shared__ float sums[32];
    const unsigned c = threadIdx.x / wave, l = threadIdx.x % wave, r = blockIdx.y, n = v.n, cb = blockIdx.x;
    const unsigned chunks = blockDim.x / wave, c0 = c * chunk, len = c0 + chunk < n ? chunk : n - c0, k_count = len / wave;
    float pre[hc_copies];
    collapse_pre(v.mixes, r, pre);
    float acc = 0.0F;
#pragma unroll
    for (unsigned k = 0; k < chunk_k; ++k)
        if (k < k_count) {
            const float xv = norm_input(x, v.x_stride, hc_stream, n, r, c0 + k * wave + l, pre);
            acc = acc + xv * xv;
        }
    acc = butterfly(acc);
    if (l == 0) sums[c] = acc;
    __syncthreads();
    float total = 0.0F;
    for (unsigned j = 0; j < chunks; ++j) total = total + sums[j];
    const float rr = declared_rsqrt(total / static_cast<float>(n) + v.eps);
    const unsigned b0 = cb * chunk, groups = (b0 + chunk < n ? chunk : n - b0) / wave;
    for (unsigned gg = c; gg < groups; gg += chunks) {
        const unsigned i = b0 + gg * wave + l;
        norm_emit(v, out, codes, deltas, sums_out, r, i, b0 / 32U + gg, norm_input(x, v.x_stride, hc_stream, n, r, i, pre), rr);
    }
}
__global__ __launch_bounds__(1024) void norm_split_kernel(NormView v) {
    norm_split_body(v, v.x, v.hc_stream, v.out, v.codes, v.deltas, v.sums);
}

// ---- hyper-connection site: expansion + mixes + Sinkhorn + collapse + norm + A8 in one launch -------
// 256-thread blocks of two kinds. Mix-partial block (chunk c, row r): wave w computes the chunk's lane
// dots of fn rows 3w, 3w + 1, 3w + 2 over one evaluation of the stream values, wave 0 also the sum of
// squares and, when expanding, the store of the expanded stream; the last of the row's partial blocks
// to finish (atomic counter per row) runs the row's split + Sinkhorn on wave 0 with agent-scope loads of
// the partials. Norm block (chunk cb, row r): the row's sum of squares by chunk (wave w takes chunks w,
// w + 8, ...), then chunk cb's 32-value groups over the 8 waves. Both kinds recompute the expansion they
// read. hidden is a multiple of 640, so a chunk lies in one copy.
constexpr unsigned site_waves = 8, site_mixes = hc_mix_count / site_waves, site_unroll = 10;
static_assert(site_waves * site_mixes == hc_mix_count, "three mixes per wave");

__device__ __forceinline__ void site_partial_row(const float* __restrict__ sub, const float* __restrict__ residual,
                                                 const HcMix* __restrict__ em, float* __restrict__ stream,
                                                 const std::uint16_t* __restrict__ fn, float* __restrict__ partial, unsigned hidden,
                                                 unsigned chunks, unsigned c, unsigned r) {
    const unsigned w = threadIdx.x / wave, l = threadIdx.x % wave, n = hc_copies * hidden;
    const unsigned c0 = c * chunk, kc = c0 / hidden, i0 = c0 % hidden;
    if (!fn && w != 0) return;
    float* srow = stream + static_cast<std::size_t>(r) * n + c0 + l;
    const std::uint16_t* f = fn ? fn + static_cast<std::size_t>(w * site_mixes) * n + c0 + l : nullptr;
    float acc[site_mixes] = {};
    float sq = 0.0F;
#pragma unroll site_unroll
    for (unsigned k = 0; k < chunk_k; ++k) {
        const float x = sub ? copy_value(sub, residual, em, r, kc, i0 + k * wave + l, hidden) : srow[k * wave];
        if (w == 0) {
            if (sub) srow[k * wave] = x;
            sq = sq + x * x;
        }
        if (fn)
#pragma unroll
            for (unsigned j = 0; j < site_mixes; ++j) acc[j] = acc[j] + f16_to_float(f[static_cast<std::size_t>(j) * n + k * wave]) * x;
    }
    if (!fn) return;
    const std::size_t base = static_cast<std::size_t>(r) * mix_waves;
#pragma unroll
    for (unsigned j = 0; j < site_mixes; ++j) {
        const float t = butterfly(acc[j]);
        if (l == 0) partial[(base + w * site_mixes + j) * chunks + c] = t;
    }
    if (w == 0) {
        const float t = butterfly(sq);
        if (l == 0) partial[(base + hc_mix_count) * chunks + c] = t;
    }
}

__device__ __forceinline__ void site_body(const HcSiteView& v, const float* __restrict__ sub, const float* __restrict__ residual,
                                          const HcMix* __restrict__ em, float* __restrict__ stream, const std::uint16_t* __restrict__ fn,
                                          float* __restrict__ partial, std::uint32_t* __restrict__ counters,
                                          const HcMix* __restrict__ cm, float* __restrict__ out, std::int8_t* __restrict__ codes,
                                          float* __restrict__ deltas, std::int32_t* __restrict__ sums_out, unsigned mix_chunks,
                                          unsigned partial_blocks, unsigned norm_chunks) {
    const unsigned H = v.hidden, n = hc_copies * H, w = threadIdx.x / wave, l = threadIdx.x % wave;
    if (blockIdx.x < partial_blocks) {
        const unsigned c = blockIdx.x % mix_chunks, r = blockIdx.x / mix_chunks;
        site_partial_row(sub, residual, em, stream, fn, partial, H, mix_chunks, c, r);
        if (!fn) return;
        __shared__ unsigned last;
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) last = atomicAdd(counters + r, 1U) == mix_chunks - 1U;
        __syncthreads();
        if (!last) return;
        // Acquire at agent scope (invalidates this CU's vector caches): the other blocks' partials, released
        // by their fences before their counter increments, are then read with ordinary (batched) loads.
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        if (w == 0) {
            mix_final_wave(partial, v.base, v.scale, n, mix_chunks, v.iters, v.hc_eps, v.mix_norm_eps, v.mixes_out, r);
            if (l == 0) __hip_atomic_store(counters + r, 0U, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        return;
    }
    __shared__ float sq[32];
    const unsigned b = blockIdx.x - partial_blocks, cb = b % norm_chunks, r = b / norm_chunks;
    float pre[hc_copies];
    collapse_pre(cm, r, pre);
    auto collapsed = [&](unsigned i) {
        float e[hc_copies];
        site_copies(sub, residual, em, stream, r, i, H, e);
        float x = 0.0F;
#pragma unroll
        for (unsigned j = 0; j < hc_copies; ++j) x = x + pre[j] * e[j];
        return x;
    };
    for (unsigned cc = w; cc < norm_chunks; cc += site_waves) {
        const unsigned c0 = cc * chunk;
        float acc = 0.0F;
#pragma unroll site_unroll
        for (unsigned k = 0; k < chunk_k; ++k) {
            const float x = collapsed(c0 + k * wave + l);
            acc = acc + x * x;
        }
        acc = butterfly(acc);
        if (l == 0) sq[cc] = acc;
    }
    __syncthreads();
    float total = 0.0F;
    for (unsigned j = 0; j < norm_chunks; ++j) total = total + sq[j];
    const float rr = declared_rsqrt(total / static_cast<float>(H) + v.norm_eps);
    const unsigned b0 = cb * chunk;
    for (unsigned gg = w; gg < chunk_k; gg += site_waves) {
        const unsigned i = b0 + gg * wave + l;
        const float y = v.norm_weight[i] * (collapsed(i) * rr);
        if (out) out[static_cast<std::size_t>(r) * v.out_stride + i] = y;
        if (codes) {
            const std::size_t g = static_cast<std::size_t>(r) * (v.a8_stride / 32U) + b0 / 32U + gg;
            a8_group(y, l, codes + static_cast<std::size_t>(r) * v.a8_stride + i, deltas + g, sums_out + g);
        }
    }
}
__global__ __launch_bounds__(32 * site_waves) void hc_site_kernel(HcSiteView v, unsigned mix_chunks, unsigned partial_blocks,
                                                                  unsigned norm_chunks) {
    site_body(v, v.sub, v.residual, v.expand_mixes, v.stream, v.fn, v.partial, v.counters, v.collapse_mixes, v.out, v.codes, v.deltas,
              v.sums, mix_chunks, partial_blocks, norm_chunks);
}

// ---- hyper-connection site, fast form (STRIX_DS41_HC_SITE_FAST, on unless =0) ----------------------
// The chunk partition, lane order, butterflies, chunk sums and Sinkhorn sequence of hc_site_kernel, so the
// same bits; what changes is when the values arrive and how often they are read. There, every F16 load of fn
// sat behind the branches of f16_to_float and every expanded value behind the branch on sub, so a partial block
// made ~80 dependent round trips to memory (the cold fn bytes among them) and the row's mixes waited ~20-30 us
// for its slowest block; and each row had its own partial blocks (fn read once per row) and 8 norm blocks that
// each re-evaluated the whole row. Here: a partial block covers a tile of 1, 2 or 4 rows; it issues all of its
// fn loads at once (raw halves in registers, converted by selects), its 256 threads evaluate the tile's stream
// values of the chunk once (expansion or stored stream, the branch lifted out as a template parameter) into LDS
// while those loads fly, and the waves then read them from LDS; a norm block covers 1-4 chunks of a row, runs its
// loop without a branch per value and keeps its own chunks in LDS for the second pass; the last partial block of
// a tile loads each row's 32 chunk partials at once and runs the rows' Sinkhorns side by side (one wave per row)
// with permlane16 gathers (the four values of a row or column sum, added in the same ascending order) instead of
// LDS shuffles. Butterflies use DPP / permlanex16 (the same exchanges as __shfl_xor, the same additions).
constexpr unsigned site_threads = 32 * site_waves, site_x_per_thread = (chunk + site_threads - 1) / site_threads;

template <unsigned D> __device__ __forceinline__ unsigned site_xor_bits(unsigned x) {  // lane ^ D, without LDS
    if constexpr (D == 16U) return __builtin_amdgcn_permlanex16(x, x, 0x76543210U, 0xfedcba98U, false, false);
    else return static_cast<unsigned>(__builtin_amdgcn_update_dpp(static_cast<int>(x), static_cast<int>(x), static_cast<int>(0x160U + D), 0xF,
                                                                  0xF, false));
}
template <unsigned D> __device__ __forceinline__ float site_xor(float v) {
    return __builtin_bit_cast(float, site_xor_bits<D>(__builtin_bit_cast(unsigned, v)));
}
__device__ __forceinline__ float site_butterfly(float v) {  // butterfly(v), the same additions in the same order
    v = v + site_xor<16>(v);
    v = v + site_xor<8>(v);
    v = v + site_xor<4>(v);
    v = v + site_xor<2>(v);
    v = v + site_xor<1>(v);
    return v;
}
// a8_group with the DPP exchanges (same maximum, same integer sum).
__device__ __forceinline__ void site_a8_group(float y, unsigned l, std::int8_t* code, float* delta, std::int32_t* sum) {
    float amax = fmaxf(0.0F, fabsf(y));
    amax = fmaxf(amax, site_xor<16>(amax));
    amax = fmaxf(amax, site_xor<8>(amax));
    amax = fmaxf(amax, site_xor<4>(amax));
    amax = fmaxf(amax, site_xor<2>(amax));
    amax = fmaxf(amax, site_xor<1>(amax));
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    const float dl = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
    float rounded = tiny ? 0.0F : __builtin_rintf(y / dl);
    rounded = -127.0F < rounded ? rounded : -127.0F;  // std::max(-127, rounded)
    rounded = rounded < 127.0F ? rounded : 127.0F;    // std::min(127, .)
    const int c = static_cast<int>(rounded);
    auto add = [](int s, unsigned o) { return s + static_cast<int>(o); };
    int s = c;
    s = add(s, site_xor_bits<16>(static_cast<unsigned>(s)));
    s = add(s, site_xor_bits<8>(static_cast<unsigned>(s)));
    s = add(s, site_xor_bits<4>(static_cast<unsigned>(s)));
    s = add(s, site_xor_bits<2>(static_cast<unsigned>(s)));
    s = add(s, site_xor_bits<1>(static_cast<unsigned>(s)));
    *code = static_cast<std::int8_t>(c);
    if (l == 0) { *delta = dl; *sum = s; }
}

// permlane16 selectors (lane i of each 16-lane row reads lane sel(i)) gathering comb entry (j, kk) to the lanes of
// row j, or (jj, k) to the lanes of column k; lane i holds entry (i / 4, i % 4).
constexpr unsigned long long site_row_sel(unsigned kk) {
    unsigned long long s = 0;
    for (unsigned i = 0; i < 16; ++i) s |= static_cast<unsigned long long>((i & 12U) | kk) << (4U * i);
    return s;
}
constexpr unsigned long long site_col_sel(unsigned jj) {
    unsigned long long s = 0;
    for (unsigned i = 0; i < 16; ++i) s |= static_cast<unsigned long long>(jj * 4U + (i & 3U)) << (4U * i);
    return s;
}
template <unsigned long long S> __device__ __forceinline__ float site_lane16(float x) {
    const unsigned b = __builtin_bit_cast(unsigned, x);
    return __builtin_bit_cast(float, __builtin_amdgcn_permlane16(b, b, static_cast<unsigned>(S), static_cast<unsigned>(S >> 32U), false, false));
}
template <unsigned K> __device__ __forceinline__ float site_row_at(float x) { return site_lane16<site_row_sel(K)>(x); }
template <unsigned J> __device__ __forceinline__ float site_col_at(float x) { return site_lane16<site_col_sel(J)>(x); }

// mix_final_wave with batched partial loads and permlane16 gathers: the same operations in the same order.
// The whole wave runs it.
__device__ __forceinline__ void mix_final_fast_wave(const float* __restrict__ partial, const float* __restrict__ base,
                                                    const float* __restrict__ scale, unsigned n, unsigned chunks, unsigned iters,
                                                    float hc_eps, float norm_eps, HcMix* __restrict__ out, unsigned r) {
    const unsigned t = threadIdx.x % wave;
    float d = 0.0F;
    if (t < mix_waves) {
        const float* p = partial + (static_cast<std::size_t>(r) * mix_waves + t) * chunks;
        constexpr unsigned batch = 32;
        unsigned c = 0;
        for (; c + batch <= chunks; c += batch) {
            float q[batch];
#pragma unroll
            for (unsigned u = 0; u < batch; ++u) q[u] = p[c + u];
#pragma unroll
            for (unsigned u = 0; u < batch; ++u) d = d + q[u];
        }
        for (; c < chunks; ++c) d = d + p[c];
    }
    constexpr unsigned h = hc_copies;
    const float rr = declared_rsqrt(__shfl(d, static_cast<int>(hc_mix_count), wave) / static_cast<float>(n) + norm_eps);
    const unsigned e = t & 15U;
    HcMix& o = out[r];
    if (t < h) o.pre[t] = declared_sigmoid((d * rr) * scale[0] + base[t]) + hc_eps;
    else if (t < 2 * h) o.post[t - h] = 2.0F * declared_sigmoid((d * rr) * scale[1] + base[t]);
    float v = (__shfl(d, static_cast<int>(2 * h + e), wave) * rr) * scale[2] + base[2 * h + e];
    float mx = site_row_at<0>(v);
    {
        float y = site_row_at<1>(v);
        mx = y > mx ? y : mx;
        y = site_row_at<2>(v);
        mx = y > mx ? y : mx;
        y = site_row_at<3>(v);
        mx = y > mx ? y : mx;
    }
    v = declared_exp(v - mx);
    auto row_sum = [](float x) {
        float s = 0.0F;
        s = s + site_row_at<0>(x);
        s = s + site_row_at<1>(x);
        s = s + site_row_at<2>(x);
        s = s + site_row_at<3>(x);
        return s;
    };
    auto col_sum = [](float x) {
        float s = 0.0F;
        s = s + site_col_at<0>(x);
        s = s + site_col_at<1>(x);
        s = s + site_col_at<2>(x);
        s = s + site_col_at<3>(x);
        return s;
    };
    v = v / row_sum(v) + hc_eps;
    for (unsigned it = 0; it < iters; ++it) {
        if (it) v = v / (row_sum(v) + hc_eps);
        v = v / (col_sum(v) + hc_eps);
    }
    if (t < 16) o.comb[t] = v;
}

// Value e (< chunk) of the stream chunk at c0 of row r: expanded (as copy_value) or read from the stored stream.
template <bool Expand>
__device__ __forceinline__ float site_stream_value(const float* __restrict__ sub, const float* __restrict__ residual,
                                                   const HcMix* __restrict__ em, const float* __restrict__ srow, unsigned r,
                                                   unsigned kc, unsigned i, unsigned hidden, unsigned e) {
    if constexpr (Expand) return copy_value(sub, residual, em, r, kc, i, hidden);
    else return srow[e];
}

// Partial block (chunk c, tile of G rows from r0 = tile * G; rows past the last are evaluated as the last row
// and not stored): the fn values of the chunk are loaded once for the tile, the G rows' stream values go to LDS,
// then wave w computes the chunk dots of fn rows 3w..3w+2 for every row of the tile and wave g the sum of squares
// of tile row g. The last of the tile's partial blocks (one counter per tile) runs the split + Sinkhorn of tile
// row g on wave g. Collapse (with Expand and Mix): the block takes the chunk at the same hidden offset in each of the
// hc copies in turn (each chunk as above) and also stores the collapse of the expanded values with the expansion's
// pre (as norm_input: from 0, ascending copies) to v.collapse_out.
template <bool Expand, bool Mix, unsigned G, bool Collapse = false>
__device__ __forceinline__ void site_fast_partial(const HcSiteView& v, const float* __restrict__ sub, const float* __restrict__ residual,
                                                  const HcMix* __restrict__ em, float* __restrict__ stream,
                                                  const std::uint16_t* __restrict__ fn, float* __restrict__ partial,
                                                  std::uint32_t* __restrict__ counters, unsigned mix_chunks, float* lds,
                                                  unsigned& last) {
    static_assert(G <= site_waves, "one wave per tile row");
    static_assert(!Collapse || (Expand && Mix), "the collapse comes with the expansion and the mixes");
    constexpr unsigned pad = site_x_per_thread * site_threads, copies = Collapse ? hc_copies : 1U;
    float (*xs)[pad] = reinterpret_cast<float (*)[pad]>(lds);
    const unsigned H = v.hidden, n = hc_copies * H, w = threadIdx.x / wave, l = threadIdx.x % wave, rows = v.rows;
    const unsigned per_tile = mix_chunks / copies, b = blockIdx.x % per_tile, tile = blockIdx.x / per_tile, r0 = tile * G;
    float y[G][site_x_per_thread];
#pragma unroll
    for (unsigned g = 0; g < G; ++g)
#pragma unroll
        for (unsigned u = 0; u < site_x_per_thread; ++u) y[g][u] = 0.0F;
    for (unsigned q = 0; q < copies; ++q) {
        if (q) __syncthreads();  // the previous copy's dots have read xs
        const unsigned c = q * per_tile + b, c0 = c * chunk, kc = c0 / H, i0 = c0 % H;
        float x[G][site_x_per_thread];
#pragma unroll
        for (unsigned g = 0; g < G; ++g) {
            const unsigned r = r0 + g < rows ? r0 + g : rows - 1U;
            const float* srow = stream + static_cast<std::size_t>(r) * n + c0;
#pragma unroll
            for (unsigned u = 0; u < site_x_per_thread; ++u) {  // past the chunk: a repeat of its last value, not stored (no branch)
                const unsigned e = threadIdx.x + u * site_threads < chunk ? threadIdx.x + u * site_threads : chunk - 1U;
                x[g][u] = site_stream_value<Expand>(sub, residual, em, srow, r, kc, i0 + e, H, e);
            }
            if constexpr (Collapse) {
                const float pre = em[r].pre[kc];
#pragma unroll
                for (unsigned u = 0; u < site_x_per_thread; ++u) y[g][u] = y[g][u] + pre * x[g][u];
            }
        }
        std::uint16_t raw[site_mixes][chunk_k];
        if constexpr (Mix) {
            const std::uint16_t* f = fn + static_cast<std::size_t>(w * site_mixes) * n + c0 + l;
#pragma unroll
            for (unsigned j = 0; j < site_mixes; ++j)
#pragma unroll
                for (unsigned k = 0; k < chunk_k; ++k) raw[j][k] = f[static_cast<std::size_t>(j) * n + k * wave];
        }
#pragma unroll
        for (unsigned g = 0; g < G; ++g) {
            float* srow = stream + static_cast<std::size_t>(r0 + g) * n + c0;
#pragma unroll
            for (unsigned u = 0; u < site_x_per_thread; ++u) {  // xs is padded to whole passes: its stores need no branch
                const unsigned e = threadIdx.x + u * site_threads;
                xs[g][e] = x[g][u];
                if constexpr (Expand)
                    if (r0 + g < rows && ((u + 1U) * site_threads <= chunk || e < chunk)) srow[e] = x[g][u];
            }
        }
        if constexpr (Mix) {
            __syncthreads();
            float f[site_mixes][chunk_k];
#pragma unroll
            for (unsigned j = 0; j < site_mixes; ++j)
#pragma unroll
                for (unsigned k = 0; k < chunk_k; ++k) f[j][k] = f16_to_float_select(raw[j][k]);
#pragma unroll
            for (unsigned g = 0; g < G; ++g) {
                const unsigned r = r0 + g;
                if (r >= rows) break;
                float acc[site_mixes] = {};
                float sq = 0.0F;
#pragma unroll
                for (unsigned k = 0; k < chunk_k; ++k) {
                    const float xv = xs[g][k * wave + l];
                    if (w == g) sq = sq + xv * xv;
#pragma unroll
                    for (unsigned j = 0; j < site_mixes; ++j) acc[j] = acc[j] + f[j][k] * xv;
                }
                const std::size_t base = static_cast<std::size_t>(r) * mix_waves;
#pragma unroll
                for (unsigned j = 0; j < site_mixes; ++j) {
                    const float s = site_butterfly(acc[j]);
                    if (l == 0) partial[(base + w * site_mixes + j) * mix_chunks + c] = s;
                }
                if (w == g) {
                    const float s = site_butterfly(sq);
                    if (l == 0) partial[(base + hc_mix_count) * mix_chunks + c] = s;
                }
            }
        }
    }
    if constexpr (Collapse) {
        const unsigned i0 = b * chunk;
#pragma unroll
        for (unsigned g = 0; g < G; ++g) {
            float* crow = v.collapse_out + static_cast<std::size_t>(r0 + g) * H + i0;
#pragma unroll
            for (unsigned u = 0; u < site_x_per_thread; ++u) {
                const unsigned e = threadIdx.x + u * site_threads;
                if (r0 + g < rows && ((u + 1U) * site_threads <= chunk || e < chunk)) crow[e] = y[g][u];
            }
        }
    }
    if constexpr (Mix) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");  // the partials before the increment (no acquire needed here)
        __syncthreads();
        if (threadIdx.x == 0) last = atomicAdd(counters + tile, 1U) == per_tile - 1U;
        __syncthreads();
        if (!last) return;
        // Acquire at agent scope, as in site_body.
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        if (threadIdx.x == 0) __hip_atomic_store(counters + tile, 0U, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (w < G && r0 + w < rows)
            mix_final_fast_wave(partial, v.base, v.scale, n, mix_chunks, v.iters, v.hc_eps, v.mix_norm_eps, v.mixes_out, r0 + w);
    }
}

// Norm block (part cb of row r, P = norm_part chunks from cb * P): the row's sum of squares by chunk (wave w takes
// chunks w, w + 8, ...; the waves keep the collapsed values of the part's chunks in LDS), then the part's 32-value
// groups over the 8 waves.
constexpr unsigned site_norm_part_max = 8, site_norm_groups_max = site_norm_part_max * chunk_k / site_waves;
template <bool Expand>
__device__ __forceinline__ void site_fast_norm(const HcSiteView& v, const float* __restrict__ sub, const float* __restrict__ residual,
                                               const HcMix* __restrict__ em, const float* __restrict__ stream,
                                               const HcMix* __restrict__ cm, float* __restrict__ out, std::int8_t* __restrict__ codes,
                                               float* __restrict__ deltas, std::int32_t* __restrict__ sums_out, unsigned b,
                                               unsigned norm_chunks, unsigned norm_part, float* own, float* sq) {
    const unsigned H = v.hidden, w = threadIdx.x / wave, l = threadIdx.x % wave;
    const unsigned parts = (norm_chunks + norm_part - 1U) / norm_part, cb = b % parts, r = b / parts;
    const unsigned first = cb * norm_part, count = (first + norm_part < norm_chunks ? norm_part : norm_chunks - first) * chunk_k;
    float pre[hc_copies];
    collapse_pre(cm, r, pre);
    const float* srow = stream + static_cast<std::size_t>(r) * hc_copies * H;
    auto collapsed = [&](unsigned i) {  // as site_body: copies by site_copies, then the collapse from 0 in ascending j
        float e[hc_copies];
#pragma unroll
        for (unsigned k = 0; k < hc_copies; ++k) e[k] = site_stream_value<Expand>(sub, residual, em, srow, r, k, i, H, k * H + i);
        float x = 0.0F;
#pragma unroll
        for (unsigned j = 0; j < hc_copies; ++j) x = x + pre[j] * e[j];
        return x;
    };
    for (unsigned cc = w; cc < norm_chunks; cc += site_waves) {
        const unsigned c0 = cc * chunk;
        float xv[chunk_k];
#pragma unroll
        for (unsigned k = 0; k < chunk_k; ++k) xv[k] = collapsed(c0 + k * wave + l);
        float acc = 0.0F;
#pragma unroll
        for (unsigned k = 0; k < chunk_k; ++k) acc = acc + xv[k] * xv[k];
        acc = site_butterfly(acc);
        if (l == 0) sq[cc] = acc;
        if (cc >= first && cc < first + norm_part)
#pragma unroll
            for (unsigned k = 0; k < chunk_k; ++k) own[(cc - first) * chunk + k * wave + l] = xv[k];
    }
    const unsigned b0 = first * chunk;
    float wt[site_norm_groups_max];
#pragma unroll
    for (unsigned q = 0; q < site_norm_groups_max; ++q) {  // past the part: a repeat of its last group (no branch)
        const unsigned gg = w + q * site_waves < count ? w + q * site_waves : count - 1U;
        wt[q] = v.norm_weight[b0 + gg * wave + l];
    }
    __syncthreads();
    float total = 0.0F;
    for (unsigned j = 0; j < norm_chunks; ++j) total = total + sq[j];
    const float rr = declared_rsqrt(total / static_cast<float>(H) + v.norm_eps);
#pragma unroll
    for (unsigned q = 0; q < site_norm_groups_max; ++q) {
        const unsigned gg = w + q * site_waves;
        if (gg >= count) break;
        const unsigned i = b0 + gg * wave + l;
        const float y = wt[q] * (own[gg * wave + l] * rr);
        if (out) out[static_cast<std::size_t>(r) * v.out_stride + i] = y;
        if (codes) {
            const std::size_t g = static_cast<std::size_t>(r) * (v.a8_stride / 32U) + b0 / 32U + gg;
            site_a8_group(y, l, codes + static_cast<std::size_t>(r) * v.a8_stride + i, deltas + g, sums_out + g);
        }
    }
}

template <bool Expand, bool Mix, unsigned G>
__global__ __launch_bounds__(site_threads) void hc_site_fast_kernel(HcSiteView v, unsigned mix_chunks, unsigned partial_blocks,
                                                                    unsigned norm_chunks, unsigned norm_part) {
    // One LDS buffer for either kind of block: a partial block's tile values, a norm block's own chunks.
    constexpr unsigned xs_floats = G * site_x_per_thread * site_threads, own_floats = site_norm_part_max * chunk;
    __shared__ float lds[xs_floats > own_floats ? xs_floats : own_floats];
    __shared__ float sq[32];
    __shared__ unsigned last;
    if (blockIdx.x < partial_blocks)
        site_fast_partial<Expand, Mix, G>(v, v.sub, v.residual, v.expand_mixes, v.stream, v.fn, v.partial, v.counters, mix_chunks, lds,
                                          last);
    else
        site_fast_norm<Expand>(v, v.sub, v.residual, v.expand_mixes, v.stream, v.collapse_mixes, v.out, v.codes, v.deltas, v.sums,
                               blockIdx.x - partial_blocks, norm_chunks, norm_part, lds, sq);
}
// Expansion + mixes + the collapse with the expansion's pre (HcSiteView::collapse_out): partial blocks only.
template <unsigned G>
__global__ __launch_bounds__(site_threads) void hc_site_collapse_kernel(HcSiteView v, unsigned mix_chunks) {
    __shared__ float lds[G * site_x_per_thread * site_threads];
    __shared__ unsigned last;
    site_fast_partial<true, true, G, true>(v, v.sub, v.residual, v.expand_mixes, v.stream, v.fn, v.partial, v.counters, mix_chunks, lds,
                                           last);
}
using SiteFastKernel = void (*)(HcSiteView, unsigned, unsigned, unsigned, unsigned);
template <unsigned G> SiteFastKernel site_fast_kernel(bool expand, bool mix) {
    return expand ? (mix ? hc_site_fast_kernel<true, true, G> : hc_site_fast_kernel<true, false, G>)
                  : (mix ? hc_site_fast_kernel<false, true, G> : hc_site_fast_kernel<false, false, G>);
}

// ---- inverse RoPE + A8 of the attention output (one wave per 32-value group) ------------------------
__device__ __forceinline__ void inverse_rope_a8_body(const InverseRopeA8View& v, const float* __restrict__ o,
                                                     std::int8_t* __restrict__ codes, float* __restrict__ deltas,
                                                     std::int32_t* __restrict__ sums) {
    const unsigned l = threadIdx.x % wave, r = blockIdx.y;
    const unsigned group = blockIdx.x * (blockDim.x / wave) + threadIdx.x / wave, n = v.heads * v.head_dim;
    if (group >= n / wave) return;
    const unsigned i = group * wave + l, j = i % v.head_dim, tail = v.head_dim - 2U * v.pairs;
    float y = o[static_cast<std::size_t>(r) * n + i];
    const float partner = __shfl_xor(y, 1, wave);
    if (j >= tail) {
        const unsigned q = j - tail, p = q / 2U;
        const std::size_t t = static_cast<std::size_t>(v.positions[r]) * v.pairs + p;
        const float cs = v.rope_cos[t], sn = -v.rope_sin[t];
        y = (q & 1U) ? partner * sn + y * cs : y * cs - partner * sn;
    }
    const std::size_t g = static_cast<std::size_t>(r) * (n / 32U) + group;
    a8_group(y, l, codes + static_cast<std::size_t>(r) * n + i, deltas + g, sums + g);
}
__global__ __launch_bounds__(256) void inverse_rope_a8_kernel(InverseRopeA8View v) {
    inverse_rope_a8_body(v, v.o, v.codes, v.deltas, v.sums);
}

// ---- RoPE (in place through one pointer) ---------------------------------------------------------
__global__ __launch_bounds__(256) void rope_kernel(RopeView v) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (index >= v.vectors * v.pairs) return;
    const unsigned p = index % v.pairs, vec = index / v.pairs;
    float* x = v.x + static_cast<std::size_t>(r) * v.row_stride + static_cast<std::size_t>(vec) * v.vector_stride + v.offset;
    const std::size_t t = static_cast<std::size_t>(v.positions[r]) * v.pairs + p;
    const float c = v.cos[t], s = v.inverse ? -v.sin[t] : v.sin[t];
    const float2 ab = *reinterpret_cast<const float2*>(x + 2 * p);
    float2 y;
    y.x = ab.x * c - ab.y * s;
    y.y = ab.x * s + ab.y * c;
    *reinterpret_cast<float2*>(x + 2 * p) = y;
}

// ---- Q8_0 x A8 sites ------------------------------------------------------------------------------
// One wave per W consecutive output rows (one group of the grouped site), lane l takes the 32-weight
// groups l, l + 32, ... in ascending order with the next group's weights in flight; each group term is
// float(integer dot) * (scale * delta), the integer dot by sudot4 (exact, so its order is free).
struct Q8Raw { unsigned short d; int q[8]; };
__device__ __forceinline__ Q8Raw load_q8(const std::uint8_t* __restrict__ row, unsigned g) {
    Q8Raw raw;
    const std::uint8_t* block = row + g * 34U;
    __builtin_memcpy(&raw.d, block, 2);
    __builtin_memcpy(raw.q, block + 2, 32);
    return raw;
}
// CB4: 4 words of 4-bit indices (byte j: weight j low nibble, weight j + 16 high nibble) to the 8 words of int8 codes
// of the equivalent Q8_0 block, by byte permutes of the 16-entry table (entries 0-7 in t0/t1, 8-15 in t2/t3).
__device__ __forceinline__ unsigned cb4_lookup(unsigned idx, const unsigned* t) {
    const unsigned sel = idx & 0x07070707U;
    const unsigned lo = __builtin_amdgcn_perm(t[1], t[0], sel), hi = __builtin_amdgcn_perm(t[3], t[2], sel);
    const unsigned high = ((idx >> 3U) & 0x01010101U) * 0xffU;
    return (hi & high) | (lo & ~high);
}
// U5 / U6: the high bits of 4 weights spread to bit 4 (U5) or bits 4-5 (U6) of their bytes, then the uniform value
// v (0..31 / 0..63) to the signed code v - 16 / v - 32 per byte.
__device__ __forceinline__ unsigned u5_word(unsigned nib, unsigned h4) {
    const unsigned v = nib | (((h4 & 1U) | ((h4 & 2U) << 7U) | ((h4 & 4U) << 14U) | ((h4 & 8U) << 21U)) << 4U);
    const unsigned b = v ^ 0x10101010U;
    return b | ((b & 0x10101010U) * 14U);
}
__device__ __forceinline__ unsigned u6_word(unsigned nib, unsigned h8) {
    const unsigned v = nib | (((h8 & 3U) | ((h8 & 0xcU) << 6U) | ((h8 & 0x30U) << 12U) | ((h8 & 0xc0U) << 18U)) << 4U);
    const unsigned b = v ^ 0x20202020U;
    return b | ((b & 0x20202020U) * 6U);
}
template <unsigned P> inline constexpr unsigned packed_block_bytes = P == 4U ? 18U : P == 5U ? 22U : P == 6U ? 26U : 34U;
template <unsigned P>
__device__ __forceinline__ Q8Raw load_weights(const std::uint8_t* __restrict__ row, unsigned g, const unsigned* table) {
    if constexpr (P == 0U) {
        return load_q8(row, g);
    } else if constexpr (P == 5U || P == 6U) {
        Q8Raw raw;
        const std::uint8_t* block = row + g * packed_block_bytes<P>;
        unsigned n[4];
        __builtin_memcpy(&raw.d, block, 2);
        __builtin_memcpy(n, block + 2, 16);
        if constexpr (P == 5U) {
            unsigned hi;
            __builtin_memcpy(&hi, block + 18, 4);
#pragma unroll
            for (unsigned k = 0; k < 4; ++k) {
                raw.q[k] = static_cast<int>(u5_word(n[k] & 0x0f0f0f0fU, (hi >> (4U * k)) & 0xfU));
                raw.q[4 + k] = static_cast<int>(u5_word((n[k] >> 4U) & 0x0f0f0f0fU, (hi >> (16U + 4U * k)) & 0xfU));
            }
        } else {
            unsigned hi[2];
            __builtin_memcpy(hi, block + 18, 8);
#pragma unroll
            for (unsigned k = 0; k < 4; ++k) {
                raw.q[k] = static_cast<int>(u6_word(n[k] & 0x0f0f0f0fU, (hi[0] >> (8U * k)) & 0xffU));
                raw.q[4 + k] = static_cast<int>(u6_word((n[k] >> 4U) & 0x0f0f0f0fU, (hi[1] >> (8U * k)) & 0xffU));
            }
        }
        (void)table;
        return raw;
    } else {
        Q8Raw raw;
        const std::uint8_t* block = row + g * cb4_block_bytes;
        unsigned n[4];
        __builtin_memcpy(&raw.d, block, 2);
        __builtin_memcpy(n, block + 2, 16);
#pragma unroll
        for (unsigned k = 0; k < 4; ++k) {
            raw.q[k] = static_cast<int>(cb4_lookup(n[k] & 0x0f0f0f0fU, table));
            raw.q[4 + k] = static_cast<int>(cb4_lookup((n[k] >> 4U) & 0x0f0f0f0fU, table));
        }
        return raw;
    }
}
__device__ __forceinline__ int dot32(const int* q, const int4& x0, const int4& x1) {
    int d = 0;
    d = __builtin_amdgcn_sudot4(true, q[0], true, x0.x, d, false);
    d = __builtin_amdgcn_sudot4(true, q[1], true, x0.y, d, false);
    d = __builtin_amdgcn_sudot4(true, q[2], true, x0.z, d, false);
    d = __builtin_amdgcn_sudot4(true, q[3], true, x0.w, d, false);
    d = __builtin_amdgcn_sudot4(true, q[4], true, x1.x, d, false);
    d = __builtin_amdgcn_sudot4(true, q[5], true, x1.y, d, false);
    d = __builtin_amdgcn_sudot4(true, q[6], true, x1.z, d, false);
    d = __builtin_amdgcn_sudot4(true, q[7], true, x1.w, d, false);
    return d;
}
constexpr unsigned q8_waves = 8;
template <unsigned MaxRows, unsigned W, unsigned P = 0>
__device__ __forceinline__ void q8_0_a8_body(const Q8A8View& v, const std::uint8_t* __restrict__ weight,
                                             const std::int8_t* __restrict__ codes, const float* __restrict__ deltas,
                                             float* __restrict__ out, unsigned bx) {
    const unsigned wave_index = bx * q8_waves + threadIdx.x / wave, l = threadIdx.x % wave;
    const unsigned o0 = wave_index * W;
    if (o0 >= v.count) return;
    constexpr unsigned block_bytes = packed_block_bytes<P>;
    const unsigned* table = v.table;
    const unsigned rows = v.rows, groups = v.width / 32U, row_bytes = (v.row_width ? v.row_width : v.width) / 32U * block_bytes;
    const unsigned grp = o0 / v.per_group;
    const signed char* xg = reinterpret_cast<const signed char*>(codes) + static_cast<std::size_t>(grp) * v.width;
    const float* dg = deltas + static_cast<std::size_t>(grp) * groups;
    const std::uint8_t* wbase = weight + v.k_offset / 32U * block_bytes;
    const std::uint8_t* wrow[W];
#pragma unroll
    for (unsigned w = 0; w < W; ++w) wrow[w] = wbase + static_cast<std::size_t>(v.first + min(o0 + w, v.count - 1U)) * row_bytes;
    float acc[W][MaxRows];
#pragma unroll
    for (unsigned w = 0; w < W; ++w)
#pragma unroll
        for (unsigned r = 0; r < MaxRows; ++r) acc[w][r] = 0.0F;
    Q8Raw current[W];
    if (l < groups)
#pragma unroll
        for (unsigned w = 0; w < W; ++w) current[w] = load_weights<P>(wrow[w], l, table);
    for (unsigned g = l; g < groups; g += wave) {
        const bool more = g + wave < groups;
        Q8Raw next[W];
        if (more)
#pragma unroll
            for (unsigned w = 0; w < W; ++w) next[w] = load_weights<P>(wrow[w], g + wave, table);
        int4 x0[MaxRows], x1[MaxRows];
        float delta[MaxRows];
#pragma unroll
        for (unsigned r = 0; r < MaxRows; ++r)
            if (r < rows) {
                const auto* x = reinterpret_cast<const int4*>(xg + static_cast<std::size_t>(r) * v.x_stride + 32U * g);
                x0[r] = x[0]; x1[r] = x[1];
                delta[r] = dg[static_cast<std::size_t>(r) * (v.x_stride / 32U) + g];
            }
#pragma unroll
        for (unsigned w = 0; w < W; ++w) {
            const float scale = f16_to_float(current[w].d);
#pragma unroll
            for (unsigned r = 0; r < MaxRows; ++r)
                if (r < rows) acc[w][r] = acc[w][r] + static_cast<float>(dot32(current[w].q, x0[r], x1[r])) * (scale * delta[r]);
        }
        if (more)
#pragma unroll
            for (unsigned w = 0; w < W; ++w) current[w] = next[w];
    }
    constexpr unsigned block_outputs = q8_waves * W;
    __shared__ float tile[MaxRows][block_outputs];  // the block's outputs, for their A8
#pragma unroll
    for (unsigned w = 0; w < W; ++w)
#pragma unroll
        for (unsigned r = 0; r < MaxRows; ++r) {
            const float total = butterfly(acc[w][r]);
            if (l == w * MaxRows + r && r < rows && o0 + w < v.count) {
                float* y = out + static_cast<std::size_t>(r) * v.out_stride + o0 + w;
                *y = v.accumulate ? *y + total : total;
            }
            if constexpr (block_outputs % 32U == 0U)
                if (v.a8_codes && l == 0 && r < rows) tile[r][threadIdx.x / wave * W + w] = total;
        }
    if constexpr (block_outputs % 32U == 0U) {
        if (!v.a8_codes) return;  // uniform over the launch; every wave of the block is valid (count % block_outputs == 0)
        __syncthreads();
        constexpr unsigned per_row = block_outputs / 32U;
        for (unsigned j = threadIdx.x / wave; j < rows * per_row; j += q8_waves) {
            const unsigned r = j / per_row, g = j % per_row, o = bx * block_outputs + 32U * g + l;
            const std::size_t gi = static_cast<std::size_t>(r) * (v.count / 32U) + o / 32U;
            a8_group(tile[r][32U * g + l], l, v.a8_codes + static_cast<std::size_t>(r) * v.count + o, v.a8_deltas + gi, v.a8_sums + gi);
        }
    }
}
template <unsigned MaxRows, unsigned W, unsigned P = 0>
__global__ __launch_bounds__(32 * q8_waves) void q8_0_a8_kernel(Q8A8View v) {
    q8_0_a8_body<MaxRows, W, P>(v, v.weight, v.codes, v.deltas, v.out, blockIdx.x);
}
template <unsigned MaxRows, unsigned W, unsigned P = 0>
__global__ __launch_bounds__(32 * q8_waves) void q8_0_a8_ready_kernel(Q8A8View v, dist::ProducerStage stage) {
    q8_0_a8_body<MaxRows, W, P>(v, v.weight, v.codes, v.deltas, v.out, blockIdx.x);
    // Body returns (including padding waves) reconverge here. Every lane
    // releases its own stores before the last block publishes the TX token.
    dist::publish_staged_view(stage, gridDim.x);
}
// Two sites in one launch: the first a_blocks blocks are site a's, the rest site b's (each output the bits of its own launch).
template <unsigned MaxRows, unsigned W, unsigned P = 0>
__global__ __launch_bounds__(32 * q8_waves) void q8_0_a8_pair_kernel(Q8A8View a, Q8A8View b, unsigned a_blocks) {
    if (blockIdx.x < a_blocks) q8_0_a8_body<MaxRows, W, P>(a, a.weight, a.codes, a.deltas, a.out, blockIdx.x);
    else q8_0_a8_body<MaxRows, W, P>(b, b.weight, b.codes, b.deltas, b.out, blockIdx.x - a_blocks);
}

// 5-8 rows: the block's 8 waves (q8_staged_outputs outputs of one group, q8_wave_outputs per wave)
// share the activations, staged in LDS by chunks of q8_stage_groups groups. A chunk starts at a multiple
// of 32 groups, so lane l still takes its groups l, l + 32, ... of every output in ascending order.
constexpr unsigned q8_stage_groups = 128, q8_wave_outputs = 4, q8_staged_outputs = q8_waves * q8_wave_outputs;
template <unsigned P = 0>
__device__ __forceinline__ void q8_0_a8_staged_body(const Q8A8View& v, const std::uint8_t* __restrict__ weight,
                                                    const std::int8_t* __restrict__ codes, const float* __restrict__ deltas,
                                                    float* __restrict__ out) {
    __shared__ int4 xs[max_rows * q8_stage_groups * 2];
    __shared__ float ds[max_rows * q8_stage_groups];
    const unsigned w = threadIdx.x / wave, l = threadIdx.x % wave, o0 = blockIdx.x * q8_staged_outputs;
    constexpr unsigned block_bytes = packed_block_bytes<P>;
    const unsigned* table = v.table;
    const unsigned rows = v.rows, groups = v.width / 32U, row_bytes = (v.row_width ? v.row_width : v.width) / 32U * block_bytes;
    const unsigned grp = o0 / v.per_group;
    const signed char* xg = reinterpret_cast<const signed char*>(codes) + static_cast<std::size_t>(grp) * v.width;
    const float* dg = deltas + static_cast<std::size_t>(grp) * groups;
    const std::uint8_t* wbase = weight + v.k_offset / 32U * block_bytes;
    float acc[q8_wave_outputs][max_rows];
#pragma unroll
    for (unsigned j = 0; j < q8_wave_outputs; ++j)
#pragma unroll
        for (unsigned r = 0; r < max_rows; ++r) acc[j][r] = 0.0F;
    for (unsigned c0 = 0; c0 < groups; c0 += q8_stage_groups) {
        const unsigned cg = groups - c0 < q8_stage_groups ? groups - c0 : q8_stage_groups;
        __syncthreads();
        for (unsigned i = threadIdx.x; i < rows * cg * 2U; i += blockDim.x) {
            const unsigned r = i / (cg * 2U), q = i % (cg * 2U);
            xs[r * q8_stage_groups * 2U + q] =
                reinterpret_cast<const int4*>(xg + static_cast<std::size_t>(r) * v.x_stride + 32U * c0)[q];
        }
        for (unsigned i = threadIdx.x; i < rows * cg; i += blockDim.x) {
            const unsigned r = i / cg, q = i % cg;
            ds[r * q8_stage_groups + q] = dg[static_cast<std::size_t>(r) * (v.x_stride / 32U) + c0 + q];
        }
        __syncthreads();
#pragma unroll
        for (unsigned j = 0; j < q8_wave_outputs; ++j) {
            const unsigned o = o0 + w * q8_wave_outputs + j;
            const std::uint8_t* wrow = wbase + static_cast<std::size_t>(v.first + (o < v.count ? o : v.count - 1U)) * row_bytes;
            unsigned g = c0 + l;
            Q8Raw current;
            if (g < c0 + cg) current = load_weights<P>(wrow, g, table);
            for (; g < c0 + cg; g += wave) {
                const bool more = g + wave < c0 + cg;
                Q8Raw next;
                if (more) next = load_weights<P>(wrow, g + wave, table);
                const float scale = f16_to_float(current.d);
                const unsigned q = g - c0;
#pragma unroll
                for (unsigned r = 0; r < max_rows; ++r)
                    if (r < rows) {
                        const int4 x0 = xs[r * q8_stage_groups * 2U + 2U * q], x1 = xs[r * q8_stage_groups * 2U + 2U * q + 1U];
                        acc[j][r] = acc[j][r] + static_cast<float>(dot32(current.q, x0, x1)) * (scale * ds[r * q8_stage_groups + q]);
                    }
                if (more) current = next;
            }
        }
    }
    __shared__ float tile[max_rows][q8_staged_outputs];  // the block's outputs, for their A8
#pragma unroll
    for (unsigned j = 0; j < q8_wave_outputs; ++j) {
        const unsigned o = o0 + w * q8_wave_outputs + j;
#pragma unroll
        for (unsigned r = 0; r < max_rows; ++r) {
            const float total = butterfly(acc[j][r]);
            if (l == j * max_rows + r && r < rows && o < v.count) {
                float* y = out + static_cast<std::size_t>(r) * v.out_stride + o;
                *y = v.accumulate ? *y + total : total;
            }
            if (v.a8_codes && l == 0 && r < rows) tile[r][w * q8_wave_outputs + j] = total;
        }
    }
    static_assert(q8_staged_outputs == 32U, "one A8 group per staged block");
    if (!v.a8_codes) return;
    __syncthreads();
    for (unsigned r = w; r < rows; r += q8_waves) {
        const std::size_t gi = static_cast<std::size_t>(r) * (v.count / 32U) + o0 / 32U;
        a8_group(tile[r][l], l, v.a8_codes + static_cast<std::size_t>(r) * v.count + o0 + l, v.a8_deltas + gi, v.a8_sums + gi);
    }
}
template <unsigned P = 0>
__global__ __launch_bounds__(32 * q8_waves) void q8_0_a8_staged_kernel(Q8A8View v) {
    q8_0_a8_staged_body<P>(v, v.weight, v.codes, v.deltas, v.out);
}
template <unsigned P = 0>
__global__ __launch_bounds__(32 * q8_waves) void q8_0_a8_staged_ready_kernel(Q8A8View v, dist::ProducerStage stage) {
    q8_0_a8_staged_body<P>(v, v.weight, v.codes, v.deltas, v.out);
    dist::publish_staged_view(stage, gridDim.x);
}

// ---- native FP8 x A8 sites (ds41_native_fp8.hpp) --------------------------------------------------
// The lane order of the Q8_0 x A8 site kernel (one wave per W consecutive outputs, lane l takes groups l, l + 32, ...
// in ascending order with the next group in flight, then the butterfly) with the group term
// float(D) * (2^(e - 136) * delta): D is the exact int32 dot of the integer forms N = e4m3 * 2^9 (|N| <= 229376)
// with the A8 codes, summed from 24-bit products (exact, so its order is free). gfx1151 has no FP8 dot instruction:
// each code is decoded by integer operations, once per group for every activation row.
struct Fp8Raw { uint4 q[2]; unsigned e; };
__device__ __forceinline__ Fp8Raw load_fp8(const std::uint8_t* __restrict__ row, const std::uint8_t* __restrict__ scales, unsigned g) {
    Fp8Raw raw;
    const auto* p = reinterpret_cast<const uint4*>(row + 32U * g);
    raw.q[0] = p[0];
    raw.q[1] = p[1];
    raw.e = scales[g];
    return raw;
}
// N of the code in bits 31..24 of t, branch-free: the code's sign at bit 31 and its exponent and mantissa at bits 26..20
// form the FP32 (8 + m) * 2^(e - 130) for e >= 1 and the FP32 denormal m * 2^-129 for e = 0 (denormals are kept,
// -fdenormal-fp-math=ieee), i.e. N * 2^-129 for every finite code; the exact scaling by 2^129 and conversion give N.
__device__ __forceinline__ int e4m3_int_top(unsigned t) {
    const unsigned bits = (t & 0x80000000U) | ((t >> 4U) & 0x07F00000U);
    return static_cast<int>(__builtin_amdgcn_ldexpf(__uint_as_float(bits), 129));
}
// 2^(e - 136) for e in [10, 254] (the loader's range): a normal FP32 power of two.
__device__ __forceinline__ float fp8_scale(unsigned e) { return __uint_as_float((e - 9U) << 23U); }
// The 32 integer forms of a group: n[4k + b] from byte b of word k.
__device__ __forceinline__ void fp8_decode(const Fp8Raw& raw, int (&n)[32]) {
    const unsigned w[8] = {raw.q[0].x, raw.q[0].y, raw.q[0].z, raw.q[0].w, raw.q[1].x, raw.q[1].y, raw.q[1].z, raw.q[1].w};
#pragma unroll
    for (unsigned k = 0; k < 8U; ++k)
#pragma unroll
        for (unsigned b = 0; b < 4U; ++b) n[4U * k + b] = e4m3_int_top(w[k] << (24U - 8U * b));
}
// The 32 A8 codes of a group as integers (once per activation row, shared by the wave's W outputs).
__device__ __forceinline__ void a8_unpack(const int4& x0, const int4& x1, int (&x)[32]) {
    const int q[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
    for (unsigned k = 0; k < 8U; ++k)
#pragma unroll
        for (unsigned b = 0; b < 4U; ++b) x[4U * k + b] = (q[k] << (24U - 8U * b)) >> 24;
}
__device__ __forceinline__ int fp8_dot32(const int (&n)[32], const int (&x)[32]) {
    int d = 0;
#pragma unroll
    for (unsigned j = 0; j < 32U; ++j) d += __mul24(n[j], x[j]);
    return d;
}
template <unsigned MaxRows, unsigned W>
__device__ __forceinline__ void fp8_a8_body(const Q8A8View& v, const std::uint8_t* __restrict__ weight, const std::uint8_t* __restrict__ scales,
                                            const std::int8_t* __restrict__ codes, const float* __restrict__ deltas, float* __restrict__ out) {
    const unsigned wave_index = blockIdx.x * q8_waves + threadIdx.x / wave, l = threadIdx.x % wave;
    const unsigned o0 = wave_index * W;
    if (o0 >= v.count) return;
    const unsigned rows = v.rows, groups = v.width / 32U, full = v.row_width ? v.row_width : v.width;
    const unsigned grp = o0 / v.per_group;
    const signed char* xg = reinterpret_cast<const signed char*>(codes) + static_cast<std::size_t>(grp) * v.width;
    const float* dg = deltas + static_cast<std::size_t>(grp) * groups;
    const std::uint8_t* wrow[W];
    const std::uint8_t* srow[W];
#pragma unroll
    for (unsigned w = 0; w < W; ++w) {
        const unsigned row = v.first + min(o0 + w, v.count - 1U);
        wrow[w] = weight + static_cast<std::size_t>(row) * full + v.k_offset;
        srow[w] = scales + static_cast<std::size_t>(row / 32U) * (full / 32U) + v.k_offset / 32U;
    }
    float acc[W][MaxRows];
#pragma unroll
    for (unsigned w = 0; w < W; ++w)
#pragma unroll
        for (unsigned r = 0; r < MaxRows; ++r) acc[w][r] = 0.0F;
    Fp8Raw current[W];
    if (l < groups)
#pragma unroll
        for (unsigned w = 0; w < W; ++w) current[w] = load_fp8(wrow[w], srow[w], l);
    for (unsigned g = l; g < groups; g += wave) {
        const bool more = g + wave < groups;
        Fp8Raw next[W];
        if (more)
#pragma unroll
            for (unsigned w = 0; w < W; ++w) next[w] = load_fp8(wrow[w], srow[w], g + wave);
        int4 x0[MaxRows], x1[MaxRows];
        float delta[MaxRows];
#pragma unroll
        for (unsigned r = 0; r < MaxRows; ++r)
            if (r < rows) {
                const auto* x = reinterpret_cast<const int4*>(xg + static_cast<std::size_t>(r) * v.x_stride + 32U * g);
                x0[r] = x[0]; x1[r] = x[1];
                delta[r] = dg[static_cast<std::size_t>(r) * (v.x_stride / 32U) + g];
            }
        int n[W][32];
        float scale[W];
#pragma unroll
        for (unsigned w = 0; w < W; ++w) {
            fp8_decode(current[w], n[w]);
            scale[w] = fp8_scale(current[w].e);
        }
#pragma unroll
        for (unsigned r = 0; r < MaxRows; ++r)
            if (r < rows) {
                int x[32];
                a8_unpack(x0[r], x1[r], x);
#pragma unroll
                for (unsigned w = 0; w < W; ++w) acc[w][r] = acc[w][r] + static_cast<float>(fp8_dot32(n[w], x)) * (scale[w] * delta[r]);
            }
        if (more)
#pragma unroll
            for (unsigned w = 0; w < W; ++w) current[w] = next[w];
    }
#pragma unroll
    for (unsigned w = 0; w < W; ++w)
#pragma unroll
        for (unsigned r = 0; r < MaxRows; ++r) {
            const float total = butterfly(acc[w][r]);
            if (l == w * MaxRows + r && r < rows && o0 + w < v.count) {
                float* y = out + static_cast<std::size_t>(r) * v.out_stride + o0 + w;
                *y = v.accumulate ? *y + total : total;
            }
        }
}
template <unsigned MaxRows, unsigned W>
__global__ __launch_bounds__(32 * q8_waves) void fp8_a8_kernel(Q8A8View v) {
    fp8_a8_body<MaxRows, W>(v, v.weight, v.scales, v.codes, v.deltas, v.out);
}

// One wave per 32-value group of a row: the clamped SwiGLU, then the group's A8 code, delta and sum (as quantize_a8,
// and as the epilogue of the fused gate/up kernels).
__global__ __launch_bounds__(256) void swiglu_a8_kernel(const float* __restrict__ gate, const float* __restrict__ up, unsigned group_count,
                                                        float limit, std::int8_t* __restrict__ codes, float* __restrict__ deltas,
                                                        std::int32_t* __restrict__ sums) {
    const unsigned gi = blockIdx.x * 8U + threadIdx.x / wave, l = threadIdx.x % wave;
    if (gi >= group_count) return;
    const std::size_t i = static_cast<std::size_t>(gi) * 32U + l;
    const float act = swiglu_clamped(gate[i], up[i], limit);
    float amax = fabsf(act);
    for (int stride = 16; stride; stride /= 2) amax = fmaxf(amax, __shfl_xor(amax, stride, wave));
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
    const float rounded = tiny ? 0.0F : rintf(act / delta);
    const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
    int sum = code;
    for (int stride = 16; stride; stride /= 2) sum += __shfl_xor(sum, stride, wave);
    codes[i] = static_cast<std::int8_t>(code);
    if (l == 0U) { deltas[gi] = delta; sums[gi] = sum; }
}

// ---- dense Q8_0 x A8, many rows: WMMA iu8 ------------------------------------------------------------
// Block: 16 rows x 32 outputs (two 16 x 16 output tiles), 4 waves that split the 32 lane classes: wave w
// owns classes w, w + 4, ..., w + 28 and walks the groups in stripes of 32 consecutive groups (group
// 32 s + w + 4 j adds to its class accumulator j), so the four waves read the four consecutive groups of
// every cache line together and each class still sums its groups in ascending order from 0. The class
// sums then follow the butterfly's tree: in the wave, levels 16, 8, 4 (j ^ 4, j ^ 2, j ^ 1); across the
// waves through LDS, levels 2 and 1: (S0 + S2) + (S1 + S3). C layout (wave32): lane l holds output column
// l % 16 and rows 2e + l / 16, e = 0..7; A rows and B columns are lanes 0..15 (16..31 duplicate them).
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
constexpr unsigned wide_waves = 4, wide_outs = 2, wide_classes = 32 / wide_waves;
// Staged form (STRIX_DS41_DENSE_FORM=1, default): the accumulator of each group's two WMMAs starts at the magic offset
// 1.5 * 2^23, so float(dot) is the exact subtraction bits - 1.5 * 2^23 (|dot| < 2^22) instead of a conversion that
// cannot dual-issue, and the 16 rows' deltas of a chunk of 32 groups are staged in LDS as [group][h][e], so a lane
// takes its eight row deltas with two 16-byte reads per group instead of eight cross-lane reads per output tile. The
// group term keeps its operations and order (float(dot) * (scale * delta), then the add), unfused: the same bits.
constexpr int wide_magic = 0x4b400000;
template <bool Staged, unsigned Outputs = wide_outs>
__device__ __forceinline__ void q8_wide_body(const Q8A8DensePrefillView& v, const std::uint8_t* __restrict__ weight,
                                             const std::int8_t* __restrict__ codes, const float* __restrict__ deltas,
                                             float* __restrict__ out) {
    __shared__ float part[wide_waves][Outputs][8][32];
    __shared__ float4 dsh[Staged ? 32U : 1U][2][2];  // [group of the chunk][h][e / 4] = deltas of rows 2e + h
    const unsigned w = threadIdx.x / wave, l = threadIdx.x % wave, h = l >> 4U, lo = l & 15U;
    const unsigned row = blockIdx.x * 16U + lo, o0 = blockIdx.y * (16U * Outputs);
    const unsigned groups = v.width / 32U, row_bytes = (v.row_width ? v.row_width : v.width) / 32U * 34U;
    const bool live = row < v.rows;
    const signed char* arow = reinterpret_cast<const signed char*>(codes) + static_cast<std::size_t>(live ? row : 0U) * v.x_stride;
    const float* drow = deltas + static_cast<std::size_t>(live ? row : 0U) * (v.x_stride / 32U);
    const std::uint8_t* wrow[Outputs];
#pragma unroll
    for (unsigned t = 0; t < Outputs; ++t) {
        const unsigned o = o0 + t * 16U + lo;
        wrow[t] = weight + v.k_offset / 32U * 34U + static_cast<std::size_t>(o < v.count ? o : v.count - 1U) * row_bytes;
    }
    float acc[wide_classes][Outputs][8];
#pragma unroll
    for (unsigned j = 0; j < wide_classes; ++j)
#pragma unroll
        for (unsigned t = 0; t < Outputs; ++t)
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) acc[j][t][e] = 0.0F;
    const v8i zero{0, 0, 0, 0, 0, 0, 0, 0};
    const v8i magic{wide_magic, wide_magic, wide_magic, wide_magic, wide_magic, wide_magic, wide_magic, wide_magic};
    for (unsigned s0 = 0; s0 < groups; s0 += 32U) {
        if constexpr (Staged) {
            __syncthreads();  // the previous chunk's deltas are read
            auto* d = reinterpret_cast<float*>(dsh);
            for (unsigned i = threadIdx.x; i < 32U * 16U; i += 32U * wide_waves) {
                const unsigned gl = i / 16U, rr = i % 16U, r = blockIdx.x * 16U + rr, g = s0 + gl;
                d[(gl * 2U + (rr & 1U)) * 8U + (rr >> 1U)] =
                    r < v.rows && g < groups ? deltas[static_cast<std::size_t>(r) * (v.x_stride / 32U) + g] : 0.0F;
            }
            __syncthreads();
        }
#pragma unroll
        for (unsigned j = 0; j < wide_classes; ++j) {
            const unsigned g = s0 + w + 4U * j;
            if (g >= groups) continue;
            v4i a0{0, 0, 0, 0}, a1{0, 0, 0, 0};
            float dl = 0.0F;
            if (live) {
                const auto* x = reinterpret_cast<const int4*>(arow + 32U * g);
                const int4 x0 = x[0], x1 = x[1];
                a0 = v4i{x0.x, x0.y, x0.z, x0.w};
                a1 = v4i{x1.x, x1.y, x1.z, x1.w};
                if constexpr (!Staged) dl = drow[g];
            }
            float rd[8];
            if constexpr (Staged) {
                const float4 d0 = dsh[g - s0][h][0], d1 = dsh[g - s0][h][1];
                rd[0] = d0.x; rd[1] = d0.y; rd[2] = d0.z; rd[3] = d0.w; rd[4] = d1.x; rd[5] = d1.y; rd[6] = d1.z; rd[7] = d1.w;
            }
#pragma unroll
            for (unsigned t = 0; t < Outputs; ++t) {
                const Q8Raw raw = load_q8(wrow[t], g);
                const v4i b0{raw.q[0], raw.q[1], raw.q[2], raw.q[3]}, b1{raw.q[4], raw.q[5], raw.q[6], raw.q[7]};
                const float scale = f16_to_float(raw.d);
                v8i c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, true, b0, Staged ? magic : zero, false);
                c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, true, b1, c, false);
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) {
                    if constexpr (Staged) {
                        const float dot = __fsub_rn(__int_as_float(c[e]), 0x1.8p23F);
                        acc[j][t][e] = __fadd_rn(acc[j][t][e], __fmul_rn(dot, __fmul_rn(scale, rd[e])));
                    } else {
                        const float rdl = __shfl(dl, static_cast<int>(2U * e + h), wave);
                        acc[j][t][e] = acc[j][t][e] + static_cast<float>(c[e]) * (scale * rdl);
                    }
                }
            }
        }
    }
#pragma unroll
    for (unsigned t = 0; t < Outputs; ++t)
#pragma unroll
        for (unsigned e = 0; e < 8; ++e) {
            const float s = ((acc[0][t][e] + acc[4][t][e]) + (acc[2][t][e] + acc[6][t][e])) +
                            ((acc[1][t][e] + acc[5][t][e]) + (acc[3][t][e] + acc[7][t][e]));
            part[w][t][e][l] = s;
        }
    __syncthreads();
    if constexpr (Outputs == 2U) {
        const unsigned t = w / 2U, o = o0 + t * 16U + lo;
#pragma unroll
        for (unsigned q = 0; q < 4; ++q) {
            const unsigned e = 4U * (w % 2U) + q, r = blockIdx.x * 16U + 2U * e + h;
            const float y = (part[0][t][e][l] + part[2][t][e][l]) + (part[1][t][e][l] + part[3][t][e][l]);
            if (o < v.count && r < v.rows) {
                float* dst = out + static_cast<std::size_t>(r) * v.out_stride + o;
                *dst = v.accumulate ? *dst + y : y;
            }
        }
    } else {
    // Disjoint stores of the tile. The reduction for each element is unchanged.
    for (unsigned item = threadIdx.x; item < Outputs * 8U * 32U; item += 32U * wide_waves) {
        const unsigned t = item / (8U * 32U), e = (item / 32U) % 8U, lane = item % 32U;
        const unsigned o = o0 + t * 16U + (lane & 15U), r = blockIdx.x * 16U + 2U * e + (lane >> 4U);
        const float y = (part[0][t][e][lane] + part[2][t][e][lane]) + (part[1][t][e][lane] + part[3][t][e][lane]);
        if (o < v.count && r < v.rows) {
            float* dst = out + static_cast<std::size_t>(r) * v.out_stride + o;
            *dst = v.accumulate ? *dst + y : y;
        }
    }
    }
}
template <bool Staged, unsigned Outputs = wide_outs>
__global__ __launch_bounds__(32 * wide_waves) void q8_0_a8_wide_kernel(Q8A8DensePrefillView v) {
    q8_wide_body<Staged, Outputs>(v, v.weight, v.codes, v.deltas, v.out);
}

// Tall form (STRIX_DS41_DENSE_TALL=RT OT, opt-in): the staged form's group terms and tree, bit for bit, on larger
// tiles: a block of 8 waves owns RT x OT tiles of 16 x 16 (16 RT rows x 16 OT outputs), so a weight block read by a
// wave serves RT row tiles and an activation block OT output tiles. Wave v owns classes v, v + 8, v + 16, v + 24 (group
// 32 s + v + 8 j adds to its accumulator j, ascending as before). Tree: t_v = (c_v + c_v+16) + (c_v+8 + c_v+24) in the
// wave (the staged form's levels 16 and 8), then through LDS S_w = t_w + t_w+4 (level 4) and (S0 + S2) + (S1 + S3).
constexpr unsigned tall_waves = 8;
// ColFirst (forms 112 / 114): blockIdx.x walks the output tiles and blockIdx.y the row tiles, so consecutive blocks share
// their activation rows (with many rows, row-fastest order re-reads activations larger than the L2 from DRAM once per
// column tile). Only the block order changes: every output keeps its bits.
// Lean (forms 2xx / 3xx with ColFirst): the magic accumulator init made opaque once, so it stays in eight VGPRs instead of
// being rebuilt from scalars before every WMMA pair, and the weight scale by the branchless f16_to_float_select (the
// same bits for every half, NaN payloads included). Only instruction selection changes: every output keeps its bits.
// Lean >= 2 (forms 4xx / 5xx with ColFirst): also no zeroing or predicates on the activation loads (rows past the end
// read the clamped row 0; their outputs are never stored, and every stored output depends on its own row only), the
// activation codes by raw buffer loads (scalar base, 32-bit offsets: no 64-bit address math per row and group), and the
// delta staging with the row clamped instead of predicated.
// Lean >= 3 (forms 6xx): also the weight records by raw buffer loads (32-bit offsets) and the scale by the hardware
// v_cvt_f32_f16, which equals f16_to_float for every non-NaN half (device check: --conv); a NaN scale (never in a valid
// Q8_0 tensor) would keep NaN but may lose its payload bits.
// Lean >= 4 (forms 8xx / 914 with ColFirst; 422 / 822 are the 2x2 tile at Lean 2 / 4): Lean 3 with every weight record
// of a group loaded before its first WMMA. Lean 5 (forms 15xx / 16xx with ColFirst): Lean 4 with opaque f32 scales.
// Lean 6 (forms 17xx / 18xx with ColFirst): Lean 5 with each group's weight records loaded one group ahead (the first
// group before the chunk loop, the next one before this group's WMMAs; never past the last group).
// k6: Lean 5/6 issue fewer VALU but run slower: the opaque scale consumes the loaded half at the load, so each tile waits
// vmcnt before the next tile's loads issue (the Lean 3 serialization again). Lean 7 (forms 19xx / 20xx with ColFirst) and
// Lean 8 (forms 21xx / 22xx with ColFirst) are Lean 5 / 6 with the scale made opaque at its use instead, after the
// tile's WMMA pair (the asm also takes the WMMA result, so it cannot be scheduled before it; it is not volatile, so the
// deltas are not re-read from LDS for every tile).
template <unsigned RT, unsigned OT, bool ColFirst = false, unsigned Lean = 0>
__global__ __launch_bounds__(32 * tall_waves) void q8_0_a8_tall_kernel(Q8A8DensePrefillView v) {
    __shared__ float part[tall_waves][RT * OT][8][32];
    __shared__ float4 dsh[RT][32][2][2];
    const unsigned w = threadIdx.x / wave, l = threadIdx.x % wave, h = l >> 4U, lo = l & 15U;
    const unsigned r0 = (ColFirst ? blockIdx.y : blockIdx.x) * 16U * RT, o0 = (ColFirst ? blockIdx.x : blockIdx.y) * 16U * OT;
    const unsigned groups = v.width / 32U, row_bytes = (v.row_width ? v.row_width : v.width) / 32U * 34U;
    const signed char* arow[RT];
    bool live[RT];
#pragma unroll
    for (unsigned rt = 0; rt < RT; ++rt) {
        const unsigned row = r0 + rt * 16U + lo;
        live[rt] = row < v.rows;
        arow[rt] = reinterpret_cast<const signed char*>(v.codes) + static_cast<std::size_t>(live[rt] ? row : 0U) * v.x_stride;
    }
    const std::uint8_t* wrow[OT];
#pragma unroll
    for (unsigned t = 0; t < OT; ++t) {
        const unsigned o = o0 + t * 16U + lo;
        wrow[t] = v.weight + v.k_offset / 32U * 34U + static_cast<std::size_t>(o < v.count ? o : v.count - 1U) * row_bytes;
    }
    float acc[4][RT][OT][8];
#pragma unroll
    for (unsigned j = 0; j < 4; ++j)
#pragma unroll
        for (unsigned rt = 0; rt < RT; ++rt)
#pragma unroll
            for (unsigned t = 0; t < OT; ++t)
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) acc[j][rt][t][e] = 0.0F;
    v8i magic{wide_magic, wide_magic, wide_magic, wide_magic, wide_magic, wide_magic, wide_magic, wide_magic};
    if constexpr (Lean >= 1) asm volatile("" : "+v"(magic));
    // Lean >= 2: the activation codes as a buffer (raw, stride 0, offsets in bytes) and each row's byte offset
    const std::size_t abytes = static_cast<std::size_t>(v.rows) * v.x_stride;
    [[maybe_unused]] const __amdgpu_buffer_rsrc_t abuf = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<std::int8_t*>(v.codes), static_cast<short>(0), (abytes < 0x7fffffffU ? abytes : std::size_t{0x7fffffffU}), 0x31004000);
    [[maybe_unused]] unsigned aoff[RT];
#pragma unroll
    for (unsigned rt = 0; rt < RT; ++rt) {
        const unsigned row = r0 + rt * 16U + lo;
        aoff[rt] = (row < v.rows ? row : 0U) * v.x_stride;
    }
    // Lean >= 3: the weight tensor as a buffer and each output row's byte offset (K-offset included)
    const std::size_t wbytes = static_cast<std::size_t>(v.count) * row_bytes;
    [[maybe_unused]] const __amdgpu_buffer_rsrc_t wbuf = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<std::uint8_t*>(v.weight), static_cast<short>(0), (wbytes < 0x7fffffffU ? wbytes : std::size_t{0x7fffffffU}), 0x31004000);
    [[maybe_unused]] unsigned woff[OT];
#pragma unroll
    for (unsigned t = 0; t < OT; ++t) {
        const unsigned o = o0 + t * 16U + lo;
        woff[t] = (o < v.count ? o : v.count - 1U) * row_bytes + v.k_offset / 32U * 34U;
    }
    // Lean >= 4: one group's weight records (OT tiles) as WMMA operands and f32 scales; Lean >= 5 keeps each scale an
    // opaque f32 VGPR, so the scale products are plain v_mul_f32 (dual-issuable) instead of v_fma_mix with the f16 scale.
    [[maybe_unused]] const auto load_w = [&](unsigned gg, v4i (&b0)[OT], v4i (&b1)[OT], float (&sc)[OT]) {
#pragma unroll
        for (unsigned t = 0; t < OT; ++t) {
            const int off = static_cast<int>(woff[t] + 34U * gg);
            const auto dw16 = __builtin_amdgcn_raw_buffer_load_b16(wbuf, off, 0, 0);
            const auto q0 = __builtin_amdgcn_raw_buffer_load_b128(wbuf, off + 2, 0, 0);
            const auto q1 = __builtin_amdgcn_raw_buffer_load_b128(wbuf, off + 18, 0, 0);
            b0[t] = v4i{static_cast<int>(q0[0]), static_cast<int>(q0[1]), static_cast<int>(q0[2]), static_cast<int>(q0[3])};
            b1[t] = v4i{static_cast<int>(q1[0]), static_cast<int>(q1[1]), static_cast<int>(q1[2]), static_cast<int>(q1[3])};
            sc[t] = static_cast<float>(__builtin_bit_cast(_Float16, static_cast<unsigned short>(dw16)));
            if constexpr (Lean == 5 || Lean == 6) asm volatile("" : "+v"(sc[t]));
        }
    };
    [[maybe_unused]] v4i pb0[OT], pb1[OT];
    [[maybe_unused]] float pscale[OT];
    if constexpr (Lean == 6 || Lean == 8)
        if (w < groups) load_w(w, pb0, pb1, pscale);
    for (unsigned s0 = 0; s0 < groups; s0 += 32U) {
        __syncthreads();  // the previous chunk's deltas are read
        auto* d = reinterpret_cast<float*>(dsh);
        for (unsigned i = threadIdx.x; i < RT * 32U * 16U; i += 32U * tall_waves) {
            const unsigned rt = i / 512U, gl = (i / 16U) % 32U, rr = i % 16U, r = r0 + rt * 16U + rr, g = s0 + gl;
            if constexpr (Lean >= 2)
                d[((rt * 32U + gl) * 2U + (rr & 1U)) * 8U + (rr >> 1U)] =
                    g < groups ? v.deltas[static_cast<std::size_t>(r < v.rows ? r : 0U) * (v.x_stride / 32U) + g] : 0.0F;
            else
                d[((rt * 32U + gl) * 2U + (rr & 1U)) * 8U + (rr >> 1U)] =
                    r < v.rows && g < groups ? v.deltas[static_cast<std::size_t>(r) * (v.x_stride / 32U) + g] : 0.0F;
        }
        __syncthreads();
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            const unsigned g = s0 + w + 8U * j;
            if (g >= groups) continue;
            v4i a0[RT], a1[RT];
#pragma unroll
            for (unsigned rt = 0; rt < RT; ++rt) {
                if constexpr (Lean >= 2) {
                    const auto x0 = __builtin_amdgcn_raw_buffer_load_b128(abuf, static_cast<int>(aoff[rt] + 32U * g), 0, 0);
                    const auto x1 = __builtin_amdgcn_raw_buffer_load_b128(abuf, static_cast<int>(aoff[rt] + 32U * g + 16U), 0, 0);
                    a0[rt] = v4i{static_cast<int>(x0[0]), static_cast<int>(x0[1]), static_cast<int>(x0[2]), static_cast<int>(x0[3])};
                    a1[rt] = v4i{static_cast<int>(x1[0]), static_cast<int>(x1[1]), static_cast<int>(x1[2]), static_cast<int>(x1[3])};
                    continue;
                }
                a0[rt] = v4i{0, 0, 0, 0};
                a1[rt] = v4i{0, 0, 0, 0};
                if (live[rt]) {
                    const auto* x = reinterpret_cast<const int4*>(arow[rt] + 32U * g);
                    const int4 x0 = x[0], x1 = x[1];
                    a0[rt] = v4i{x0.x, x0.y, x0.z, x0.w};
                    a1[rt] = v4i{x1.x, x1.y, x1.z, x1.w};
                }
            }
            // Lean >= 4: every weight record of the chunk issued before the first WMMA (Lean 3 left each tile's buffer loads
            // next to its WMMA pair, waiting vmcnt(0) per tile).
            [[maybe_unused]] v4i wb0[OT], wb1[OT];
            [[maybe_unused]] float wscale[OT];
            if constexpr (Lean == 6 || Lean == 8) {
                // Lean 6 / 8: this group's records were loaded one group ahead; issue the next group's now
#pragma unroll
                for (unsigned t = 0; t < OT; ++t) {
                    wb0[t] = pb0[t];
                    wb1[t] = pb1[t];
                    wscale[t] = pscale[t];
                }
                const unsigned gn = j < 3U ? g + 8U : s0 + 32U + w;
                if (gn < groups) load_w(gn, pb0, pb1, pscale);
            } else if constexpr (Lean >= 4) {
                load_w(g, wb0, wb1, wscale);
            }
#pragma unroll
            for (unsigned t = 0; t < OT; ++t) {
                v4i b0, b1;
                float scale;
                if constexpr (Lean >= 4) {
                    b0 = wb0[t];
                    b1 = wb1[t];
                    scale = wscale[t];
                } else if constexpr (Lean >= 3) {
                    const int off = static_cast<int>(woff[t] + 34U * g);
                    const auto dw16 = __builtin_amdgcn_raw_buffer_load_b16(wbuf, off, 0, 0);
                    const auto q0 = __builtin_amdgcn_raw_buffer_load_b128(wbuf, off + 2, 0, 0);
                    const auto q1 = __builtin_amdgcn_raw_buffer_load_b128(wbuf, off + 18, 0, 0);
                    b0 = v4i{static_cast<int>(q0[0]), static_cast<int>(q0[1]), static_cast<int>(q0[2]), static_cast<int>(q0[3])};
                    b1 = v4i{static_cast<int>(q1[0]), static_cast<int>(q1[1]), static_cast<int>(q1[2]), static_cast<int>(q1[3])};
                    scale = static_cast<float>(__builtin_bit_cast(_Float16, static_cast<unsigned short>(dw16)));
                } else {
                    const Q8Raw raw = load_q8(wrow[t], g);
                    b0 = v4i{raw.q[0], raw.q[1], raw.q[2], raw.q[3]};
                    b1 = v4i{raw.q[4], raw.q[5], raw.q[6], raw.q[7]};
                    scale = Lean >= 1 ? f16_to_float_select(raw.d) : f16_to_float(raw.d);
                }
#pragma unroll
                for (unsigned rt = 0; rt < RT; ++rt) {
                    const float4 d0 = dsh[rt][g - s0][h][0], d1 = dsh[rt][g - s0][h][1];
                    const float rd[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
                    v8i c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0[rt], true, b0, magic, false);
                    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1[rt], true, b1, c, false);
                    float sc = scale;
                    if constexpr (Lean >= 7) asm("" : "+v"(sc) : "v"(c[0]));  // not volatile: no barrier for the LDS reads
#pragma unroll
                    for (unsigned e = 0; e < 8; ++e) {
                        const float dot = __fsub_rn(__int_as_float(c[e]), 0x1.8p23F);
                        acc[j][rt][t][e] = __fadd_rn(acc[j][rt][t][e], __fmul_rn(dot, __fmul_rn(sc, rd[e])));
                    }
                }
            }
        }
    }
#pragma unroll
    for (unsigned rt = 0; rt < RT; ++rt)
#pragma unroll
        for (unsigned t = 0; t < OT; ++t)
#pragma unroll
            for (unsigned e = 0; e < 8; ++e)
                part[w][rt * OT + t][e][l] = (acc[0][rt][t][e] + acc[2][rt][t][e]) + (acc[1][rt][t][e] + acc[3][rt][t][e]);
    __syncthreads();
    constexpr unsigned per_thread = RT * OT * 8U * 32U / (32U * tall_waves);
    float ys[per_thread];
#pragma unroll
    for (unsigned q = 0; q < per_thread; ++q) {
        const unsigned i = threadIdx.x + q * 32U * tall_waves;
        const unsigned ln = i % 32U, e = (i / 32U) % 8U, tile = i / 256U, rt = tile / OT, t = tile % OT;
        const unsigned o = o0 + t * 16U + (ln & 15U), r = r0 + rt * 16U + 2U * e + (ln >> 4U);
        const float s0 = part[0][tile][e][ln] + part[4][tile][e][ln], s1 = part[1][tile][e][ln] + part[5][tile][e][ln];
        const float s2 = part[2][tile][e][ln] + part[6][tile][e][ln], s3 = part[3][tile][e][ln] + part[7][tile][e][ln];
        const float y = (s0 + s2) + (s1 + s3);
        ys[q] = y;
        if (o < v.count && r < v.rows) {
            float* dst = v.out + static_cast<std::size_t>(r) * v.out_stride + o;
            *dst = v.accumulate ? *dst + y : y;
        }
    }
    // Optional A8 of the outputs (OT even: the block owns whole 32-output groups). ys[q] is tile q (rt = q / OT, t = q % OT)
    // at row 2w + l / 16 and column l % 16 of the tile, so tiles t = 2g and 2g + 1 hold columns lo and lo + 16 of the same
    // row and group in one lane: each half wave quantizes its row's group with quantize_a8_kernel's operations, its first
    // maximum step (lanes L and L ^ 16) being the lane's own fmaxf(|x0|, |x1|), then the 8 / 4 / 2 / 1 steps in the half.
    if constexpr (OT % 2U == 0U) {
        if (v.a8_codes) {
#pragma unroll
            for (unsigned rt = 0; rt < RT; ++rt)
#pragma unroll
                for (unsigned g = 0; g < OT / 2U; ++g) {
                    const float x0 = ys[rt * OT + 2U * g], x1 = ys[rt * OT + 2U * g + 1U];
                    float amax = fmaxf(fabsf(x0), fabsf(x1));
                    for (int stride = 8; stride; stride /= 2) amax = fmaxf(amax, __shfl_xor(amax, stride, 16));
                    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
                    const float dl = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
                    const float rounded0 = tiny ? 0.0F : rintf(x0 / dl), rounded1 = tiny ? 0.0F : rintf(x1 / dl);
                    const int c0 = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded0)));
                    const int c1 = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded1)));
                    int sum = c0 + c1;
                    for (int stride = 8; stride; stride /= 2) sum += __shfl_xor(sum, stride, 16);
                    const unsigned r = r0 + rt * 16U + 2U * w + (l >> 4U), o = o0 + g * 32U;
                    if (r < v.rows && o < v.count) {
                        std::int8_t* code = v.a8_codes + static_cast<std::size_t>(r) * v.a8_stride + o + lo;
                        code[0] = static_cast<std::int8_t>(c0);
                        code[16] = static_cast<std::int8_t>(c1);
                        if (lo == 0U) {
                            const std::size_t gi = static_cast<std::size_t>(r) * (v.a8_stride / 32U) + o / 32U;
                            v.a8_deltas[gi] = dl;
                            v.a8_sums[gi] = sum;
                        }
                    }
                }
        }
    }
}

__device__ __forceinline__ void f16_rows_body(const F16RowsView& v, const std::uint16_t* __restrict__ weight,
                                              const float* __restrict__ x, float* __restrict__ out) {
    const unsigned o = blockIdx.x * (blockDim.x / wave) + threadIdx.x / wave, l = threadIdx.x % wave;
    if (o >= v.count) return;
    const std::uint16_t* row = weight + static_cast<std::size_t>(v.first + o) * v.width;
    float acc[max_rows] = {};
    for (unsigned i = l; i < v.width; i += wave) {
        const float w = f16_to_float(row[i]);
        for (unsigned r = 0; r < v.rows; ++r) acc[r] = acc[r] + w * x[static_cast<std::size_t>(r) * v.x_stride + i];
    }
    for (unsigned r = 0; r < v.rows; ++r) {
        const float total = butterfly(acc[r]);
        if (l == 0) out[static_cast<std::size_t>(r) * v.out_stride + o] = total;
    }
}
__global__ __launch_bounds__(256) void f16_rows_kernel(F16RowsView v) { f16_rows_body(v, v.weight, v.x, v.out); }

// The same products as f16_rows_kernel, bit for bit (lane l sums i = l, l + 32, ... in that order, then the butterfly),
// with more memory in flight: each wave owns Outputs consecutive outputs, reads their weight rows 16 bytes per lane
// (chunks of 256 values staged in LDS, so each lane still takes its strided values) and reads every activation value
// once for all of them. Widths that are multiples of 256 only.
constexpr unsigned f16_chunk = 256, f16_waves = 8;
template <unsigned Outputs>
__global__ __launch_bounds__(wave * f16_waves) void f16_rows_wide_kernel(F16RowsView v) {
    __shared__ uint4 stage[f16_waves][Outputs][f16_chunk / 8];
    const unsigned wv = threadIdx.x / wave, l = threadIdx.x % wave;
    const unsigned o0 = (blockIdx.x * f16_waves + wv) * Outputs;
    float acc[Outputs][max_rows] = {};
    for (unsigned c = 0; c < v.width; c += f16_chunk) {
#pragma unroll
        for (unsigned q = 0; q < Outputs; ++q) {
            const unsigned o = o0 + q < v.count ? o0 + q : v.count - 1U;  // past the end: a valid row, never stored
            stage[wv][q][l] = *reinterpret_cast<const uint4*>(v.weight + static_cast<std::size_t>(v.first + o) * v.width + c + 8U * l);
        }
        __syncthreads();
        float w[Outputs][f16_chunk / wave];
        const auto* h = reinterpret_cast<const std::uint16_t*>(stage[wv]);
#pragma unroll
        for (unsigned q = 0; q < Outputs; ++q)
#pragma unroll
            for (unsigned j = 0; j < f16_chunk / wave; ++j) w[q][j] = f16_to_float(h[q * f16_chunk + l + wave * j]);
        __syncthreads();
        for (unsigned r = 0; r < v.rows; ++r) {
            const float* x = v.x + static_cast<std::size_t>(r) * v.x_stride + c + l;
#pragma unroll
            for (unsigned j = 0; j < f16_chunk / wave; ++j) {
                const float xv = x[wave * j];
#pragma unroll
                for (unsigned q = 0; q < Outputs; ++q) acc[q][r] = acc[q][r] + w[q][j] * xv;
            }
        }
    }
#pragma unroll
    for (unsigned q = 0; q < Outputs; ++q)
        for (unsigned r = 0; r < v.rows; ++r) {
            const float total = butterfly(acc[q][r]);
            if (l == 0 && o0 + q < v.count) v.out[static_cast<std::size_t>(r) * v.out_stride + o0 + q] = total;
        }
}

bool f16_rows_wide_enabled = [] {
    const char* e = std::getenv("STRIX_DS41_F16_WIDE");
    return !(e && e[0] == '0');
}();
// Many rows: lanes stay the lane classes of the declared lane dot (lane l: elements l, l + 32, ... in
// ascending order, then the butterfly), and each wave keeps the class sums of 16 rows x 6 outputs, so a
// weight value is used by 16 rows and an activation value by 6 outputs; a block's 8 waves take 48
// consecutive outputs of the same 16 rows (activations shared through the caches).
constexpr unsigned f16_tile_rows = 16, f16_wave_outputs = 6, f16_block_outputs = 8 * f16_wave_outputs;
__device__ __forceinline__ void f16_many_body(const F16RowsView& v, const std::uint16_t* __restrict__ weight, const float* __restrict__ x,
                                              float* __restrict__ out) {
    const unsigned w = threadIdx.x / wave, l = threadIdx.x % wave;
    const unsigned r0 = blockIdx.y * f16_tile_rows, o0 = blockIdx.x * f16_block_outputs + w * f16_wave_outputs;
    if (o0 >= v.count) return;
    const std::uint16_t* wrow[f16_wave_outputs];
#pragma unroll
    for (unsigned j = 0; j < f16_wave_outputs; ++j)
        wrow[j] = weight + static_cast<std::size_t>(v.first + (o0 + j < v.count ? o0 + j : v.count - 1U)) * v.width;
    const float* xrow[f16_tile_rows];
#pragma unroll
    for (unsigned r = 0; r < f16_tile_rows; ++r) xrow[r] = x + static_cast<std::size_t>(r0 + r < v.rows ? r0 + r : v.rows - 1U) * v.x_stride;
    float acc[f16_tile_rows][f16_wave_outputs];
#pragma unroll
    for (unsigned r = 0; r < f16_tile_rows; ++r)
#pragma unroll
        for (unsigned j = 0; j < f16_wave_outputs; ++j) acc[r][j] = 0.0F;
    for (unsigned i = l; i < v.width; i += wave) {
        float wv[f16_wave_outputs];
#pragma unroll
        for (unsigned j = 0; j < f16_wave_outputs; ++j) wv[j] = f16_to_float(wrow[j][i]);
#pragma unroll
        for (unsigned r = 0; r < f16_tile_rows; ++r) {
            const float xv = xrow[r][i];
#pragma unroll
            for (unsigned j = 0; j < f16_wave_outputs; ++j) acc[r][j] = acc[r][j] + wv[j] * xv;
        }
    }
#pragma unroll
    for (unsigned r = 0; r < f16_tile_rows; ++r)
#pragma unroll
        for (unsigned j = 0; j < f16_wave_outputs; ++j) {
            const float t = butterfly(acc[r][j]);
            if (l == 0 && r0 + r < v.rows && o0 + j < v.count) out[static_cast<std::size_t>(r0 + r) * v.out_stride + o0 + j] = t;
        }
}
__global__ __launch_bounds__(256) void f16_many_kernel(F16RowsView v) { f16_many_body(v, v.weight, v.x, v.out); }

// Many rows, LDS tiles (width a multiple of 128, x rows 16-byte aligned): a block takes RB rows x OB outputs and
// stages slabs of 128 elements of both (weights converted to FP32 once) in LDS with 16-byte loads; each wave keeps
// the class sums of 8 rows x 16 outputs. Lane l still adds elements l, l + 32, ... in ascending order with the same
// expression, then the butterfly, so the bits are those of f16_rows_kernel whatever the tile. Blocks walk GROUP row
// tiles per output tile before the next output tile, so a weight tile is reused from the caches and the activations
// of GROUP x RB rows stay resident while every output tile passes over them.
constexpr unsigned f16t_slab = 128;
// In LDS, element i = 32 s + l of a slab sits at (s / V) * 32 V + l V + s % V: lane l reads V consecutive s values with
// one V-wide load (V 1, 2 or 4), consecutive lanes at consecutive addresses.
template <unsigned V>
__device__ __forceinline__ unsigned f16t_slot(unsigned i) {
    const unsigned s = i / wave, l = i % wave;
    return (s / V) * wave * V + l * V + s % V;
}
template <unsigned RB, unsigned OB, unsigned GROUP, unsigned V>
__global__ __launch_bounds__(RB / 8U * (OB / 16U) * 32U) void f16_tiled_kernel(F16RowsView v, unsigned row_tiles, unsigned output_tiles) {
    constexpr unsigned threads = RB / 8U * (OB / 16U) * 32U, wave_rows = RB / 8U;
    __shared__ float xs[RB][f16t_slab];
    __shared__ float ws[OB][f16t_slab];
    const unsigned b = blockIdx.x, g = b / (GROUP * output_tiles);
    const unsigned rem = b - g * GROUP * output_tiles;
    const unsigned in_group = row_tiles - g * GROUP < GROUP ? row_tiles - g * GROUP : GROUP;
    const unsigned rt = g * GROUP + rem % in_group, ot = rem / in_group;
    const unsigned r0 = rt * RB, o0 = ot * OB;
    const unsigned tid = threadIdx.x, l = tid % wave, w = tid / wave;
    const unsigned wr = (w % wave_rows) * 8U, wo = (w / wave_rows) * 16U;
    float acc[8][16];
#pragma unroll
    for (unsigned r = 0; r < 8; ++r)
#pragma unroll
        for (unsigned j = 0; j < 16; ++j) acc[r][j] = 0.0F;
    for (unsigned k0 = 0; k0 < v.width; k0 += f16t_slab) {
#pragma unroll
        for (unsigned q = tid; q < RB * f16t_slab / 4U; q += threads) {
            const unsigned r = q / (f16t_slab / 4U), c = q % (f16t_slab / 4U);
            const unsigned row = r0 + r < v.rows ? r0 + r : v.rows - 1U;
            const float4 x4 = *reinterpret_cast<const float4*>(v.x + static_cast<std::size_t>(row) * v.x_stride + k0 + c * 4U);
            if constexpr (V == 1U) {
                *reinterpret_cast<float4*>(&xs[r][c * 4U]) = x4;
            } else {
                xs[r][f16t_slot<V>(c * 4U)] = x4.x;
                xs[r][f16t_slot<V>(c * 4U + 1U)] = x4.y;
                xs[r][f16t_slot<V>(c * 4U + 2U)] = x4.z;
                xs[r][f16t_slot<V>(c * 4U + 3U)] = x4.w;
            }
        }
#pragma unroll
        for (unsigned q = tid; q < OB * f16t_slab / 8U; q += threads) {
            const unsigned o = q / (f16t_slab / 8U), c = q % (f16t_slab / 8U);
            const unsigned out = o0 + o < v.count ? o0 + o : v.count - 1U;
            const uint4 raw = *reinterpret_cast<const uint4*>(v.weight + static_cast<std::size_t>(v.first + out) * v.width + k0 + c * 8U);
            const unsigned words[4] = {raw.x, raw.y, raw.z, raw.w};
            float f[8];
#pragma unroll
            for (unsigned k = 0; k < 4; ++k) {
                f[2U * k] = f16_to_float(static_cast<std::uint16_t>(words[k] & 0xFFFFU));
                f[2U * k + 1U] = f16_to_float(static_cast<std::uint16_t>(words[k] >> 16U));
            }
            if constexpr (V == 1U) {
                *reinterpret_cast<float4*>(&ws[o][c * 8U]) = make_float4(f[0], f[1], f[2], f[3]);
                *reinterpret_cast<float4*>(&ws[o][c * 8U + 4U]) = make_float4(f[4], f[5], f[6], f[7]);
            } else {
#pragma unroll
                for (unsigned k = 0; k < 8; ++k) ws[o][f16t_slot<V>(c * 8U + k)] = f[k];
            }
        }
        __syncthreads();
#pragma unroll
        for (unsigned sb = 0; sb < f16t_slab / wave / V; ++sb) {  // V steps s = sb V .. sb V + V - 1, ascending
            float xv[8][V], wv[16][V];
#pragma unroll
            for (unsigned r = 0; r < 8; ++r)
#pragma unroll
                for (unsigned k = 0; k < V; ++k) xv[r][k] = xs[wr + r][sb * wave * V + l * V + k];
#pragma unroll
            for (unsigned j = 0; j < 16; ++j)
#pragma unroll
                for (unsigned k = 0; k < V; ++k) wv[j][k] = ws[wo + j][sb * wave * V + l * V + k];
#pragma unroll
            for (unsigned k = 0; k < V; ++k)
#pragma unroll
                for (unsigned r = 0; r < 8; ++r)
#pragma unroll
                    for (unsigned j = 0; j < 16; ++j) acc[r][j] = acc[r][j] + wv[j][k] * xv[r][k];
        }
        __syncthreads();
    }
    // every lane holds the butterfly total (pairs are added commutatively); lane j stores output j
#pragma unroll
    for (unsigned r = 0; r < 8; ++r)
#pragma unroll
        for (unsigned j = 0; j < 16; ++j) {
            const float t = butterfly(acc[r][j]);
            const unsigned row = r0 + wr + r, o = o0 + wo + j;
            if (l == j && row < v.rows && o < v.count) v.out[static_cast<std::size_t>(row) * v.out_stride + o] = t;
        }
}
template <unsigned RB, unsigned OB, unsigned GROUP, unsigned V>
void launch_f16_tiled(const F16RowsView& v, hipStream_t stream) {
    const unsigned row_tiles = (v.rows + RB - 1U) / RB, output_tiles = (v.count + OB - 1U) / OB;
    hipLaunchKernelGGL((f16_tiled_kernel<RB, OB, GROUP, V>), dim3(row_tiles * output_tiles), dim3(RB / 8U * (OB / 16U) * 32U), 0, stream, v,
                       row_tiles, output_tiles);
}

// Engram prefill tile with next-slab prefetch: f16_tiled_kernel's V = 2 slot layout and arithmetic body on RB x OB blocks of
// WR x 16 per wave, but the next 128-element slab's raw x / F16 weight words are loaded into registers before the current
// slab's arithmetic and published to LDS after its last barrier. WR = 6 (48 x 64, 1024 threads) keeps the prefetch spill-free
// (164 VGPRs; the 64 x 64 / WR 8 form needs 192 and spills). Lane l still adds elements l, l + 32, ... in ascending order with
// separate multiply and add rounding from 0.0F, then the same butterfly and storing lane: the bits of f16_rows_kernel.
template <unsigned RB, unsigned OB, unsigned GROUP, unsigned WR>
__global__ __launch_bounds__(RB / WR * (OB / 16U) * 32U) void f16_tiled_pf_kernel(F16RowsView v, unsigned row_tiles, unsigned output_tiles) {
    constexpr unsigned S = f16t_slab, WO = 16U, T = RB / WR * (OB / WO) * 32U;
    constexpr unsigned NX = (RB * S / 4U + T - 1U) / T, NW = (OB * S / 8U + T - 1U) / T;
    static_assert(RB % WR == 0 && OB % WO == 0 && T <= 1024U && (RB + OB) * S * sizeof(float) <= 65536U);
    __shared__ float xs[RB][S];
    __shared__ float ws[OB][S];
    const unsigned b = blockIdx.x, g = b / (GROUP * output_tiles);
    const unsigned rem = b - g * GROUP * output_tiles;
    const unsigned in_group = row_tiles - g * GROUP < GROUP ? row_tiles - g * GROUP : GROUP;
    const unsigned r0 = (g * GROUP + rem % in_group) * RB, o0 = (rem / in_group) * OB;
    const unsigned tid = threadIdx.x, l = tid % wave, w = tid / wave;
    const unsigned wr = (w % (RB / WR)) * WR, wo = (w / (RB / WR)) * WO;
    float acc[WR][WO];
#pragma unroll
    for (unsigned r = 0; r < WR; ++r)
#pragma unroll
        for (unsigned j = 0; j < WO; ++j) acc[r][j] = 0.0F;
    float4 px[NX];
    uint4 pw[NW];
    const auto fetch = [&](unsigned k0) {  // raw words of slab k0 (clamped rows/outputs, as f16_tiled_kernel)
#pragma unroll
        for (unsigned n = 0; n < NX; ++n) {
            const unsigned q = tid + n * T;
            if (q < RB * S / 4U) {
                const unsigned r = q / (S / 4U), c = q % (S / 4U);
                const unsigned row = r0 + r < v.rows ? r0 + r : v.rows - 1U;
                px[n] = *reinterpret_cast<const float4*>(v.x + static_cast<std::size_t>(row) * v.x_stride + k0 + c * 4U);
            }
        }
#pragma unroll
        for (unsigned n = 0; n < NW; ++n) {
            const unsigned q = tid + n * T;
            if (q < OB * S / 8U) {
                const unsigned o = q / (S / 8U), c = q % (S / 8U);
                const unsigned out = o0 + o < v.count ? o0 + o : v.count - 1U;
                pw[n] = *reinterpret_cast<const uint4*>(v.weight + static_cast<std::size_t>(v.first + out) * v.width + k0 + c * 8U);
            }
        }
    };
    const auto publish = [&]() {  // the same values and LDS slots as f16_tiled_kernel<.., V = 2>
#pragma unroll
        for (unsigned n = 0; n < NX; ++n) {
            const unsigned q = tid + n * T;
            if (q < RB * S / 4U) {
                const unsigned r = q / (S / 4U), c = q % (S / 4U);
                xs[r][f16t_slot<2>(c * 4U)] = px[n].x;
                xs[r][f16t_slot<2>(c * 4U + 1U)] = px[n].y;
                xs[r][f16t_slot<2>(c * 4U + 2U)] = px[n].z;
                xs[r][f16t_slot<2>(c * 4U + 3U)] = px[n].w;
            }
        }
#pragma unroll
        for (unsigned n = 0; n < NW; ++n) {
            const unsigned q = tid + n * T;
            if (q < OB * S / 8U) {
                const unsigned o = q / (S / 8U), c = q % (S / 8U);
                const unsigned words[4] = {pw[n].x, pw[n].y, pw[n].z, pw[n].w};
#pragma unroll
                for (unsigned k = 0; k < 4; ++k) {
                    ws[o][f16t_slot<2>(c * 8U + 2U * k)] = f16_to_float(static_cast<std::uint16_t>(words[k] & 0xFFFFU));
                    ws[o][f16t_slot<2>(c * 8U + 2U * k + 1U)] = f16_to_float(static_cast<std::uint16_t>(words[k] >> 16U));
                }
            }
        }
    };
    fetch(0);
    for (unsigned k0 = 0; k0 < v.width; k0 += S) {
        publish();
        __syncthreads();  // current slab visible
        if (k0 + S < v.width) fetch(k0 + S);  // consumed only by the next publish
#pragma unroll
        for (unsigned sb = 0; sb < S / wave / 2U; ++sb) {  // steps s = 2 sb, 2 sb + 1, ascending
            float xv[2][WR], wv[WO][2];
#pragma unroll
            for (unsigned r = 0; r < WR; ++r)
#pragma unroll
                for (unsigned t = 0; t < 2; ++t) xv[t][r] = xs[wr + r][sb * wave * 2U + l * 2U + t];
#pragma unroll
            for (unsigned j = 0; j < WO; ++j)
#pragma unroll
                for (unsigned t = 0; t < 2; ++t) wv[j][t] = ws[wo + j][sb * wave * 2U + l * 2U + t];
#pragma unroll
            for (unsigned t = 0; t < 2; ++t)
#pragma unroll
                for (unsigned r = 0; r < WR; ++r)
#pragma unroll
                    for (unsigned j = 0; j < WO; ++j) acc[r][j] = acc[r][j] + wv[j][t] * xv[t][r];
        }
        __syncthreads();  // all readers finish before the next publish
    }
#pragma unroll
    for (unsigned r = 0; r < WR; ++r)
#pragma unroll
        for (unsigned j = 0; j < WO; ++j) {
            const float t = butterfly(acc[r][j]);
            const unsigned row = r0 + wr + r, o = o0 + wo + j;
            if (l == j && row < v.rows && o < v.count) v.out[static_cast<std::size_t>(row) * v.out_stride + o] = t;
        }
}

// ---- embedding and argmax ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void embedding_kernel(const std::uint16_t* __restrict__ table, const std::uint32_t* __restrict__ tokens,
                                                        unsigned hidden, unsigned long long total, float* __restrict__ out) {
    const auto index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;  // rows * hc * hidden
    const auto per_row = static_cast<unsigned long long>(hc_copies) * hidden;
    const auto r = index / per_row;
    const unsigned i = static_cast<unsigned>(index % hidden);
    out[index] = f16_to_float(table[static_cast<std::size_t>(tokens[r]) * hidden + i]);
}

constexpr unsigned argmax_threads = 256, argmax_span = 4096;

__device__ inline ArgmaxCandidate better(const ArgmaxCandidate& a, const ArgmaxCandidate& b) {
    // Both usable: larger value, then lower index. Unusable entries (index ~0U, value -inf) never win.
    ArgmaxCandidate o = (b.value > a.value || (b.value == a.value && b.index < a.index)) ? b : a;
    o.finite = a.finite & b.finite;
    return o;
}

__global__ __launch_bounds__(argmax_threads) void argmax_partial_kernel(const float* __restrict__ logits, unsigned count,
                                                                        unsigned stride, unsigned first_index, unsigned blocks,
                                                                        ArgmaxCandidate* __restrict__ partial) {
    __shared__ alignas(ArgmaxCandidate) unsigned char storage[sizeof(ArgmaxCandidate) * argmax_threads];
    auto* s = reinterpret_cast<ArgmaxCandidate*>(storage);
    const unsigned r = blockIdx.y, b = blockIdx.x;
    const float* x = logits + static_cast<std::size_t>(r) * stride;
    ArgmaxCandidate best{-__builtin_huge_valf(), 0xffffffffU, 1};
    const unsigned begin = b * argmax_span, end = begin + argmax_span < count ? begin + argmax_span : count;
    for (unsigned i = begin + threadIdx.x; i < end; i += argmax_threads) {
        const float v = x[i];
        if (!__builtin_isfinite(v)) { best.finite = 0; continue; }
        best = better(best, ArgmaxCandidate{v, first_index + i, 1});
    }
    s[threadIdx.x] = best;
    __syncthreads();
    for (unsigned d = argmax_threads / 2; d; d >>= 1) {
        if (threadIdx.x < d) s[threadIdx.x] = better(s[threadIdx.x], s[threadIdx.x + d]);
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[static_cast<std::size_t>(r) * blocks + b] = s[0];
}

__global__ __launch_bounds__(argmax_threads) void argmax_final_kernel(const ArgmaxCandidate* __restrict__ partial, unsigned blocks,
                                                                      ArgmaxCandidate* __restrict__ out) {
    __shared__ alignas(ArgmaxCandidate) unsigned char storage[sizeof(ArgmaxCandidate) * argmax_threads];
    auto* s = reinterpret_cast<ArgmaxCandidate*>(storage);
    const unsigned r = blockIdx.x;
    ArgmaxCandidate best{-__builtin_huge_valf(), 0xffffffffU, 1};
    for (unsigned i = threadIdx.x; i < blocks; i += argmax_threads) {
        const ArgmaxCandidate* q = partial + static_cast<std::size_t>(r) * blocks + i;
        best = better(best, ArgmaxCandidate{q->value, q->index, q->finite});  // field loads: no private copy
    }
    s[threadIdx.x] = best;
    __syncthreads();
    for (unsigned d = argmax_threads / 2; d; d >>= 1) {
        if (threadIdx.x < d) s[threadIdx.x] = better(s[threadIdx.x], s[threadIdx.x + d]);
        __syncthreads();
    }
    if (threadIdx.x == 0) out[r] = s[0];
}

__global__ void combine_argmax_kernel(const ArgmaxCandidate* __restrict__ a, const ArgmaxCandidate* __restrict__ b, unsigned rows,
                                      ArgmaxCandidate* __restrict__ out) {
    const unsigned r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) out[r] = better(a[r], b[r]);
}

// ---- Engram ---------------------------------------------------------------------------------------
__global__ __launch_bounds__(256) void round_bf16_kernel(float* __restrict__ values, unsigned long long count) {
    const auto index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) values[index] = round_to_bf16(values[index]);
}

// One block per row: wave (copy cc, chunk c) computes the chunk sums of h*h, key*key and (h*(q*k))*key;
// the copy's totals are the ascending chunk sums; then every thread applies the gates (in place through
// hc_stream only).
__global__ __launch_bounds__(1024) void engram_inject_kernel(float* __restrict__ hc_stream, const float* __restrict__ kv,
                                                             const float* __restrict__ qn, const float* __restrict__ kn, unsigned hidden,
                                                             float eps, float inv_sqrt_hidden) {
    __shared__ float part[3][hc_copies][32];
    __shared__ float gates[hc_copies];
    const unsigned chunks = (hidden + chunk - 1U) / chunk;
    const unsigned w = threadIdx.x / wave, l = threadIdx.x % wave, cc = w / chunks, c = w % chunks;
    float* h = hc_stream + static_cast<std::size_t>(blockIdx.x) * hc_copies * hidden;
    const float* row_kv = kv + static_cast<std::size_t>(blockIdx.x) * (hc_copies + 1U) * hidden;
    if (cc < hc_copies) {
        const float* hcp = h + cc * hidden;
        const float* key = row_kv + cc * hidden;
        const float* q = qn + cc * hidden;
        const float* k = kn + cc * hidden;
        const unsigned c0 = c * chunk, end = c0 + chunk < hidden ? c0 + chunk : hidden;
        float hh = 0.0F, kk = 0.0F, dk = 0.0F;
        for (unsigned i = c0 + l; i < end; i += wave) {
            hh = hh + hcp[i] * hcp[i];
            kk = kk + key[i] * key[i];
            dk = dk + (hcp[i] * (q[i] * k[i])) * key[i];
        }
        hh = butterfly(hh);
        kk = butterfly(kk);
        dk = butterfly(dk);
        if (l == 0) { part[0][cc][c] = hh; part[1][cc][c] = kk; part[2][cc][c] = dk; }
    }
    __syncthreads();
    if (threadIdx.x < hc_copies) {
        const unsigned cp = threadIdx.x;
        float hh = 0.0F, kk = 0.0F, dk = 0.0F;
        for (unsigned j = 0; j < chunks; ++j) { hh = hh + part[0][cp][j]; kk = kk + part[1][cp][j]; dk = dk + part[2][cp][j]; }
        const float rh = declared_rsqrt(hh / static_cast<float>(hidden) + eps);
        const float rk = declared_rsqrt(kk / static_cast<float>(hidden) + eps);
        gates[cp] = engram_gate_of((dk * (rh * rk)) * inv_sqrt_hidden);
    }
    __syncthreads();
    const float* value = row_kv + hc_copies * hidden;
    for (unsigned i = threadIdx.x; i < hc_copies * hidden; i += blockDim.x) h[i] = h[i] + gates[i / hidden] * value[i % hidden];
}

unsigned blocks_for(unsigned long long total, unsigned threads) {
    return static_cast<unsigned>((total + threads - 1) / threads);
}
std::uint64_t q8_bytes(std::uint64_t rows, std::uint64_t width) { return rows * (width / 32U) * 34U; }

} // namespace

std::size_t hc_mix_partial_floats(std::uint32_t rows, std::uint32_t hidden) {
    const std::size_t chunks = (std::size_t{hidden} * hc_copies + chunk - 1) / chunk;
    return std::size_t{rows} * (hc_mix_count + 1) * chunks;
}

void enqueue_hc_mixes(const HcMixView& v, void* stream) {
    require(v.stream && v.fn && v.base && v.scale && v.out && v.partial && v.rows && v.hidden && v.iters >= 1, "hc mixes view");
    const unsigned n = v.hidden * hc_copies, chunks = (n + chunk - 1U) / chunk;
    hipLaunchKernelGGL(hc_mix_partial_kernel, dim3(chunks, (v.rows + hc_tile - 1U) / hc_tile), dim3(wave * mix_waves), 0,
                       as_stream(stream), v.stream, v.fn, v.hidden, v.rows, chunks, v.partial);
    hipLaunchKernelGGL(hc_mix_final_kernel, dim3(v.rows), dim3(wave), 0, as_stream(stream), v.partial, v.base, v.scale, n, chunks,
                       v.iters, v.hc_eps, v.norm_eps, v.out);
    check(hipGetLastError(), "hc mixes launch");
}

void enqueue_hc_collapse(const float* hc_stream, const HcMix* mixes, std::uint32_t rows, std::uint32_t hidden, float* out,
                         void* stream) {
    const auto total = static_cast<unsigned long long>(rows) * hidden;
    hipLaunchKernelGGL(hc_collapse_kernel, dim3(blocks_for(total, 256)), dim3(256), 0, as_stream(stream), hc_stream, mixes,
                       hidden, total, out);
    check(hipGetLastError(), "hc collapse launch");
}

void enqueue_hc_expand(const float* sub, const float* residual, const HcMix* mixes, std::uint32_t rows, std::uint32_t hidden,
                       float* out, void* stream) {
    const std::size_t n = std::size_t{rows} * hc_copies * hidden * 4;
    require(!overlap(out, n, residual, n) && !overlap(out, n, sub, n / hc_copies), "hc expand output must not alias its inputs");
    const auto total = static_cast<unsigned long long>(rows) * hc_copies * hidden;
    hipLaunchKernelGGL(hc_expand_kernel, dim3(blocks_for(total, 256)), dim3(256), 0, as_stream(stream), sub, residual, mixes,
                       hidden, total, out);
    check(hipGetLastError(), "hc expand launch");
}

std::size_t hc_site_counter_count(std::uint32_t rows) { return rows; }

void enqueue_hc_site(const HcSiteView& v, void* stream) {
    const unsigned H = v.hidden, n = H * hc_copies, mix_chunks = (n + chunk - 1U) / chunk, norm_chunks = (H + chunk - 1U) / chunk;
    const std::size_t stream_bytes = std::size_t{v.rows} * n * 4;
    require(v.rows && H && H % chunk == 0 && v.stream && norm_chunks <= 32U && (v.fn || v.norm_weight || v.sub) &&
                (v.tile == 0 || v.tile == 1 || v.tile == 2 || v.tile == 4 || v.tile == 8) && v.norm_part <= site_norm_part_max &&
                (!v.sub || (v.residual && v.expand_mixes && !overlap(v.stream, stream_bytes, v.residual, stream_bytes) &&
                            !overlap(v.stream, stream_bytes, v.sub, stream_bytes / hc_copies))) &&
                (!v.fn || (v.base && v.scale && v.mixes_out && v.partial && v.counters && v.iters >= 1 &&
                           v.mixes_out != v.expand_mixes && v.mixes_out != v.collapse_mixes)) &&
                (!v.norm_weight || ((v.out || v.codes) && (!v.out || v.out_stride >= H) &&
                                    (!v.codes || (v.deltas && v.sums && v.a8_stride >= H && v.a8_stride % 32U == 0)))) &&
                (!v.collapse_out || (v.sub && v.fn && !v.norm_weight && v.form == 2 && v.launches == 1 &&
                                     !overlap(v.stream, stream_bytes, v.collapse_out, stream_bytes / hc_copies) &&
                                     !overlap(v.residual, stream_bytes, v.collapse_out, stream_bytes / hc_copies) &&
                                     !overlap(v.sub, stream_bytes / hc_copies, v.collapse_out, stream_bytes / hc_copies))),
            "hc site view");
    // STRIX_DS41_HC_SITE_FAST=0: the previous site kernel (and its one-launch row limit); same bits either way.
    static const bool fast_env = [] { const char* e = std::getenv("STRIX_DS41_HC_SITE_FAST"); return !(e && e[0] == '0'); }();
    const bool fast = v.form == 2 || (v.form == 0 && fast_env);
    const bool one = v.launches == 1 ||
                     (v.launches == 0 && v.rows <= (fast ? hc_site_fast_one_launch_rows : hc_site_one_launch_rows));
    if (!one) {
        if (v.sub) enqueue_hc_expand(v.sub, v.residual, v.expand_mixes, v.rows, H, v.stream, stream);
        if (v.fn) {
            HcMixView m;
            m.stream = v.stream; m.fn = v.fn; m.base = v.base; m.scale = v.scale; m.rows = v.rows; m.hidden = H; m.iters = v.iters;
            m.hc_eps = v.hc_eps; m.norm_eps = v.mix_norm_eps; m.out = v.mixes_out; m.partial = v.partial;
            enqueue_hc_mixes(m, stream);
        }
        if (v.norm_weight) {
            NormView nv;
            nv.hc_stream = v.stream; nv.mixes = v.collapse_mixes; nv.weight = v.norm_weight; nv.rows = v.rows; nv.n = H; nv.eps = v.norm_eps;
            nv.out = v.out; nv.out_stride = v.out_stride; nv.codes = v.codes; nv.deltas = v.deltas; nv.sums = v.sums;
            nv.a8_stride = v.a8_stride;
            enqueue_norm(nv, stream);
        }
        return;
    }
    // Partial blocks exist when there are mixes to compute or an expansion to store.
    if (!fast) {
        const unsigned partial_blocks = v.fn || v.sub ? mix_chunks * v.rows : 0U;
        const unsigned norm_blocks = v.norm_weight ? norm_chunks * v.rows : 0U;
        hipLaunchKernelGGL(hc_site_kernel, dim3(partial_blocks + norm_blocks), dim3(wave * site_waves), 0, as_stream(stream), v, mix_chunks,
                           partial_blocks, norm_chunks);
    } else {
        // Tiles of g rows per partial block (1, 2, 4 or 8) and norm parts of `part` chunks per block (1-8).
        const unsigned g = v.tile ? v.tile : v.rows == 1 ? 1U : v.rows == 2 ? 2U : 4U;
        unsigned part = v.norm_part ? v.norm_part : v.rows == 1 ? 1U : v.rows == 2 ? 2U : 4U;  // measured (cold fn, R 1-8)
        part = part < norm_chunks ? part : norm_chunks;
        const unsigned tiles = (v.rows + g - 1U) / g;
        if (v.collapse_out) {  // one block per hidden chunk (every copy) and tile of at most 4 rows (8: ~100 VGPRs spilled)
            const unsigned gc = g < 4U ? g : 4U, ctiles = (v.rows + gc - 1U) / gc;
            const auto kernel = gc == 1 ? hc_site_collapse_kernel<1> : gc == 2 ? hc_site_collapse_kernel<2> : hc_site_collapse_kernel<4>;
            hipLaunchKernelGGL(kernel, dim3(mix_chunks / hc_copies * ctiles), dim3(site_threads), 0, as_stream(stream), v, mix_chunks);
            check(hipGetLastError(), "hc site collapse launch");
            return;
        }
        const unsigned partial_blocks = v.fn || v.sub ? mix_chunks * tiles : 0U;
        const unsigned norm_blocks = v.norm_weight ? (norm_chunks + part - 1U) / part * v.rows : 0U;
        const bool expand = v.sub != nullptr, mix = v.fn != nullptr;
        const SiteFastKernel kernel = g == 1   ? site_fast_kernel<1>(expand, mix)
                                      : g == 2 ? site_fast_kernel<2>(expand, mix)
                                      : g == 4 ? site_fast_kernel<4>(expand, mix)
                                               : site_fast_kernel<8>(expand, mix);
        hipLaunchKernelGGL(kernel, dim3(partial_blocks + norm_blocks), dim3(site_threads), 0, as_stream(stream), v, mix_chunks,
                           partial_blocks, norm_chunks, part);
    }
    check(hipGetLastError(), "hc site launch");
}

void enqueue_norm(const NormView& v, void* stream) {
    const unsigned chunks = (v.n + chunk - 1U) / chunk;
    require(v.weight && v.rows && v.n && v.n % 32U == 0 && chunks <= 32U && (v.x || v.hc_stream) && (v.out || v.codes) &&
                (!v.f16 || v.f16_stride >= v.n) &&
                (!v.out || v.out_stride >= v.n) && (!v.x || v.x_stride >= v.n) &&
                (!v.codes || (v.deltas && v.sums && v.a8_stride >= v.n && v.a8_stride % 32U == 0)) &&
                (!v.pairs || (v.rope_cos && v.rope_sin && v.positions && v.rope_offset % 2U == 0 && v.rope_offset + 2U * v.pairs <= v.n)),
            "norm view");
    // Up to 8 rows the split form spreads each row over `chunks` blocks; it needs the outputs apart from the
    // input (another block still reads the row). More rows, or in place, keep one block per row.
    const std::size_t in_bytes = v.x ? (std::size_t{v.rows - 1U} * v.x_stride + v.n) * 4 : std::size_t{v.rows} * hc_copies * v.n * 4;
    const void* in = v.x ? static_cast<const void*>(v.x) : static_cast<const void*>(v.hc_stream);
    const bool apart = !overlap(in, in_bytes, v.out, v.out ? (std::size_t{v.rows - 1U} * v.out_stride + v.n) * 4 : 0) &&
                       !overlap(in, in_bytes, v.codes, v.codes ? std::size_t{v.rows} * v.a8_stride : 0);
    if (v.rows <= 8U && chunks > 1U && apart)
        hipLaunchKernelGGL(norm_split_kernel, dim3(chunks, v.rows), dim3(wave * chunks), 0, as_stream(stream), v);
    else if (norm_ilp_enabled())
        hipLaunchKernelGGL(norm_ilp_kernel, dim3(v.rows), dim3(wave * chunks), 0, as_stream(stream), v);
    else
        hipLaunchKernelGGL(norm_kernel, dim3(v.rows), dim3(wave * chunks), 0, as_stream(stream), v);
    check(hipGetLastError(), "norm launch");
}

void enqueue_rms_norm(const float* x, const float* weight, std::uint32_t rows, std::uint32_t n, std::uint32_t stride, float eps,
                      float* out, void* stream) {
    NormView v;
    v.x = x; v.x_stride = stride; v.weight = weight; v.rows = rows; v.n = n; v.eps = eps; v.out = out; v.out_stride = stride;
    enqueue_norm(v, stream);
}

void enqueue_inverse_rope_a8(const InverseRopeA8View& v, void* stream) {
    const unsigned n = v.heads * v.head_dim;
    require(v.o && v.rows && n % 32U == 0 && v.head_dim >= 2U * v.pairs && v.rope_cos && v.rope_sin && v.positions && v.codes &&
                v.deltas && v.sums, "inverse RoPE + A8 view");
    hipLaunchKernelGGL(inverse_rope_a8_kernel, dim3((n / 32U + 7U) / 8U, v.rows), dim3(256), 0, as_stream(stream), v);
    check(hipGetLastError(), "inverse RoPE + A8 launch");
}

void enqueue_rope(const RopeView& v, void* stream) {
    require(v.x && v.cos && v.sin && v.positions && v.rows && v.vectors && v.pairs, "RoPE view");
    require(v.offset % 2U == 0 && v.row_stride % 2U == 0 && v.vector_stride % 2U == 0, "RoPE view alignment");
    hipLaunchKernelGGL(rope_kernel, dim3((v.vectors * v.pairs + 255U) / 256U, v.rows), dim3(256), 0, as_stream(stream), v);
    check(hipGetLastError(), "RoPE launch");
}

template <unsigned MaxRows, unsigned W>
void launch_q8_0_a8(const Q8A8View& v, hipStream_t stream) {
    const unsigned waves = (v.count + W - 1U) / W;
    const dim3 grid((waves + q8_waves - 1U) / q8_waves), block(32 * q8_waves);
    if (v.packed == 4U) hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_0_a8_kernel<MaxRows, W, 4>), grid, block, 0, stream, v);
    else if (v.packed == 5U) hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_0_a8_kernel<MaxRows, W, 5>), grid, block, 0, stream, v);
    else if (v.packed == 6U) hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_0_a8_kernel<MaxRows, W, 6>), grid, block, 0, stream, v);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_0_a8_kernel<MaxRows, W>), grid, block, 0, stream, v);
}

template <unsigned MaxRows, unsigned W>
void launch_q8_0_a8_pair(const Q8A8View& a, const Q8A8View& b, hipStream_t stream) {
    const unsigned a_blocks = ((a.count + W - 1U) / W + q8_waves - 1U) / q8_waves, b_blocks = ((b.count + W - 1U) / W + q8_waves - 1U) / q8_waves;
    const dim3 grid(a_blocks + b_blocks), block(32 * q8_waves);
    if (a.packed == 4U) hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_0_a8_pair_kernel<MaxRows, W, 4>), grid, block, 0, stream, a, b, a_blocks);
    else if (a.packed == 5U) hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_0_a8_pair_kernel<MaxRows, W, 5>), grid, block, 0, stream, a, b, a_blocks);
    else if (a.packed == 6U) hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_0_a8_pair_kernel<MaxRows, W, 6>), grid, block, 0, stream, a, b, a_blocks);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_0_a8_pair_kernel<MaxRows, W>), grid, block, 0, stream, a, b, a_blocks);
}

template <unsigned MaxRows, unsigned W>
void launch_fp8_a8(const Q8A8View& v, hipStream_t stream) {
    const unsigned waves = (v.count + W - 1U) / W;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(fp8_a8_kernel<MaxRows, W>), dim3((waves + q8_waves - 1U) / q8_waves), dim3(32 * q8_waves), 0, stream, v);
}
namespace {
std::uint32_t fp8_outputs_per_wave = 0;  // 0: STRIX_DS41_FP8_W or 1

void enqueue_fp8_a8(const Q8A8View& view, hipStream_t q) {
    require(view.scales && !view.planar && view.rows <= 16U, "native FP8 x A8 site: scales, interleaved rows, 1..16 rows");
    if (view.rows > max_rows) {
        launch_fp8_a8<16, 1>(view, q);
        check(hipGetLastError(), "native FP8 x A8 sixteen-row launch");
        return;
    }
    // Outputs per wave (1, 2 or 4; same bits): STRIX_DS41_FP8_W, default 1 at 1-2 rows (one contiguous row stream per
    // wave) and 2 from 3 rows (each row's unpacked A8 codes serve two outputs). 5-8 rows take at most 2 (4 spills).
    static const unsigned env_w = [] { const char* e = std::getenv("STRIX_DS41_FP8_W"); return e && (e[0] == '1' || e[0] == '2' || e[0] == '4') ? unsigned(e[0] - '0') : 0U; }();
    const unsigned set = fp8_outputs_per_wave ? fp8_outputs_per_wave : env_w;
    const unsigned want = set ? set : view.rows <= 2 ? 1U : 2U;
    const bool w4 = want >= 4U && view.per_group % 4U == 0, w2 = want >= 2U && view.per_group % 2U == 0;
    if (view.rows == 1) { if (w4) launch_fp8_a8<1, 4>(view, q); else if (w2) launch_fp8_a8<1, 2>(view, q); else launch_fp8_a8<1, 1>(view, q); }
    else if (view.rows == 2) { if (w4) launch_fp8_a8<2, 4>(view, q); else if (w2) launch_fp8_a8<2, 2>(view, q); else launch_fp8_a8<2, 1>(view, q); }
    else if (view.rows <= 4) { if (w4) launch_fp8_a8<4, 4>(view, q); else if (w2) launch_fp8_a8<4, 2>(view, q); else launch_fp8_a8<4, 1>(view, q); }
    else { if (w2) launch_fp8_a8<8, 2>(view, q); else launch_fp8_a8<8, 1>(view, q); }
    check(hipGetLastError(), "native FP8 x A8 launch");
}
}  // namespace

void set_native_fp8_outputs_per_wave(std::uint32_t w) { fp8_outputs_per_wave = w; }

void enqueue_swiglu_a8(const float* gate, const float* up, std::uint32_t rows, std::uint32_t width, float limit, std::int8_t* codes,
                       float* deltas, std::int32_t* sums, void* stream) {
    require(gate && up && codes && deltas && sums && rows && width && width % 32U == 0, "SwiGLU + A8 view");
    const unsigned groups = rows * (width / 32U);
    hipLaunchKernelGGL(swiglu_a8_kernel, dim3((groups + 7U) / 8U), dim3(256), 0, as_stream(stream), gate, up, groups, limit, codes, deltas, sums);
    check(hipGetLastError(), "SwiGLU + A8 launch");
}

static Q8A8View checked_q8_0_a8(const Q8A8View& v) {
    Q8A8View view = v;
    if (!view.per_group) view.per_group = view.count;
    require(v.packed == 0U || (v.packed >= 4U && v.packed <= 6U), "Q8_0 x A8 site: packed must be 0, 4, 5 or 6");
    require(!v.planar || v.packed != 0U, "Q8_0 x A8 site: planar needs packed rows");
    require(!v.planar, "Q8_0 x A8 site: no planar packed kernel is registered yet");
    require(v.weight && v.codes && v.deltas && v.out && v.count && v.width && v.width % 32U == 0 && v.rows >= 1 && v.rows <= 16U &&
                v.x_stride % 32U == 0 && view.count % view.per_group == 0 && reinterpret_cast<std::uintptr_t>(v.codes) % 16U == 0 &&
                (v.rows == 1 || v.x_stride >= (view.count / view.per_group) * v.width) &&
                (!v.row_width || (v.row_width % 32U == 0 && v.k_offset % 32U == 0 && v.k_offset + v.width <= v.row_width &&
                                  view.per_group == view.count)),
            "Q8_0 x A8 site view");
    require(!v.a8_codes || (v.a8_deltas && v.a8_sums && !v.accumulate && v.count % 32U == 0 && v.out_stride == v.count),
            "Q8_0 x A8 site view: output A8");
    return view;
}

void enqueue_q8_0_a8(const Q8A8View& v, void* stream) {
    if (v.packed == packed_native_fp8) {
        Q8A8View view = v;
        if (!view.per_group) view.per_group = view.count;
        require(!v.a8_codes || (v.a8_deltas && v.a8_sums && !v.accumulate && v.count % 32U == 0 && v.out_stride == v.count),
                "native FP8 x A8 site view: output A8");
        require(v.weight && v.codes && v.deltas && v.out && v.count && v.width && v.width % 32U == 0 && v.rows >= 1 &&
                    v.x_stride % 32U == 0 && view.count % view.per_group == 0 && reinterpret_cast<std::uintptr_t>(v.codes) % 16U == 0 &&
                    reinterpret_cast<std::uintptr_t>(v.weight) % 16U == 0 &&
                    (v.rows == 1 || v.x_stride >= (view.count / view.per_group) * v.width) &&
                    (!v.row_width || (v.row_width % 32U == 0 && v.k_offset % 32U == 0 && v.k_offset + v.width <= v.row_width &&
                                      view.per_group == view.count)),
                "native FP8 x A8 site view");
        enqueue_fp8_a8(view, as_stream(stream));
        if (v.a8_codes) enqueue_quantize_a8(v.out, v.count, v.rows, v.a8_codes, v.a8_deltas, v.a8_sums, stream);
        return;
    }
    const Q8A8View view = checked_q8_0_a8(v);
    // A real sixteen-row product: each weight load serves every row. Keep all <=8
    // specializations and their LDS sizes unchanged. The separate output quantizer
    // has the same per-row bytes as the fused epilogue.
    if (v.rows > max_rows) {
        Q8A8View plain = view;
        plain.a8_codes = nullptr;
        launch_q8_0_a8<16, 1>(plain, as_stream(stream));
        check(hipGetLastError(), "Q8_0 x A8 sixteen-row launch");
        if (v.a8_codes) enqueue_quantize_a8(v.out, v.count, v.rows, v.a8_codes, v.a8_deltas, v.a8_sums, stream);
        return;
    }
    // STRIX_DS41_RN_WIDE=1|2: 2-8 rows in the multi-row form (hip_ds41_rn_wide.hpp), same bits; =1 the default shape
    // for every site, =2 the per site x format x rows table (hip_ds41_rn_table.hpp; may keep this kernel). The rn form
    // does not write the output A8 (ATTN_A8): then the separate quantize of out, the declared same-bits fallback.
    if (v.rows >= 2) {
        if (const int mode = rn_wide_mode(); mode != 0) {
            RnWideShape shape;
            const bool rn = mode == 2 ? rn_table_select(view, shape) : (shape = rn_wide_default_shape(view), true);
            if (rn && rn_wide_supported(view, shape)) {
                Q8A8View plain = view;
                plain.a8_codes = nullptr;
                enqueue_q8_0_a8_rn(plain, shape, stream);
                if (v.a8_codes) enqueue_quantize_a8(v.out, v.count, v.rows, v.a8_codes, v.a8_deltas, v.a8_sums, stream);
                return;
            }
        }
    }
    const auto q = as_stream(stream);
    // W outputs per wave share every activation load; W must divide the group size.
    // Packed rows (CB4/U5/U6) at 1-2 rows: STRIX_DS41_PACKED_W=1|2|4 caps the outputs per wave (same bits,
    // W-independent). Default 1: P10 turn A, R1 greedy 46.85 -> 43.85 ms/token against 4.
    static const unsigned packed_w = [] { const char* e = std::getenv("STRIX_DS41_PACKED_W"); return e && (e[0] == '2' || e[0] == '4') ? unsigned(e[0] - '0') : 1U; }();
    const bool cap = view.packed != 0U && v.rows <= 2;
    const bool w4 = view.per_group % 4U == 0 && !(cap && packed_w < 4U), w2 = view.per_group % 2U == 0 && !(cap && packed_w < 2U);
    // The output A8 in the product kernel when one block owns whole groups: 1-2 rows at 4 outputs per wave (32 per
    // block) or the staged form; otherwise a separate quantize of out.
    const bool staged = v.rows > 4 && view.per_group % q8_staged_outputs == 0 && view.count / q8_staged_outputs >= 32U;
    const bool a8_in_kernel = v.a8_codes && ((v.rows <= 2 && w4 && view.count % (q8_waves * 4U) == 0U) || staged);
    if (v.a8_codes && !a8_in_kernel) {
        Q8A8View plain = view;
        plain.a8_codes = nullptr;
        enqueue_q8_0_a8(plain, stream);
        enqueue_quantize_a8(v.out, v.count, v.rows, v.a8_codes, v.a8_deltas, v.a8_sums, stream);
        return;
    }
    if (v.rows == 1) { if (w4) launch_q8_0_a8<1, 4>(view, q); else if (w2) launch_q8_0_a8<1, 2>(view, q); else launch_q8_0_a8<1, 1>(view, q); }
    else if (v.rows == 2) { if (w4) launch_q8_0_a8<2, 4>(view, q); else if (w2) launch_q8_0_a8<2, 2>(view, q); else launch_q8_0_a8<2, 1>(view, q); }
    else if (v.rows <= 4) { if (w2) launch_q8_0_a8<4, 2>(view, q); else launch_q8_0_a8<4, 1>(view, q); }
    // Staged form when it still fills the GPU (>= 32 blocks); small sites keep one output per wave.
    else if (view.per_group % q8_staged_outputs == 0 && view.count / q8_staged_outputs >= 32U) {
        const dim3 grid((view.count + q8_staged_outputs - 1U) / q8_staged_outputs), block(32 * q8_waves);
        if (view.packed == 4U) hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_0_a8_staged_kernel<4>), grid, block, 0, q, view);
        else if (view.packed == 5U) hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_0_a8_staged_kernel<5>), grid, block, 0, q, view);
        else if (view.packed == 6U) hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_0_a8_staged_kernel<6>), grid, block, 0, q, view);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_0_a8_staged_kernel<0>), grid, block, 0, q, view);
    }
    else launch_q8_0_a8<8, 1>(view, q);
    check(hipGetLastError(), "Q8_0 x A8 launch");
}

namespace {
template <unsigned R, unsigned W, unsigned P>
void launch_q8_ready(const Q8A8View& v, dist::ProducerStage stage, hipStream_t q) {
    const dim3 grid((v.count + q8_waves * W - 1U) / (q8_waves * W));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_0_a8_ready_kernel<R, W, P>), grid, dim3(32 * q8_waves), 0, q, v, stage);
}
template <unsigned P>
void launch_q8_ready_rows(const Q8A8View& v, dist::ProducerStage stage, hipStream_t q) {
    static const unsigned packed_w = [] {
        const char* e = std::getenv("STRIX_DS41_PACKED_W");
        return e && (e[0] == '2' || e[0] == '4') ? unsigned(e[0] - '0') : 1U;
    }();
    const bool cap = P != 0U && v.rows <= 2;
    const bool w4 = v.per_group % 4U == 0 && !(cap && packed_w < 4U);
    const bool w2 = v.per_group % 2U == 0 && !(cap && packed_w < 2U);
    if (v.rows == 1) {
        if (w4) launch_q8_ready<1, 4, P>(v, stage, q);
        else if (w2) launch_q8_ready<1, 2, P>(v, stage, q);
        else launch_q8_ready<1, 1, P>(v, stage, q);
    } else if (v.rows == 2) {
        if (w4) launch_q8_ready<2, 4, P>(v, stage, q);
        else if (w2) launch_q8_ready<2, 2, P>(v, stage, q);
        else launch_q8_ready<2, 1, P>(v, stage, q);
    } else if (v.rows <= 4) {
        if (w2) launch_q8_ready<4, 2, P>(v, stage, q);
        else launch_q8_ready<4, 1, P>(v, stage, q);
    } else if (v.per_group % q8_staged_outputs == 0 && v.count / q8_staged_outputs >= 32U) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_0_a8_staged_ready_kernel<P>), dim3((v.count + 31U) / 32U),
                           dim3(32 * q8_waves), 0, q, v, stage);
    } else launch_q8_ready<8, 1, P>(v, stage, q);
}
}

bool enqueue_q8_0_a8_ready(const Q8A8View& v, dist::ProducerStage stage, void* stream) {
    require(stage.data == v.out && stage.flag && stage.token, "Q8 x A8 ready producer view");
    // Publishing only a partial output or A8 from another launch is not supported.
    if (v.packed == packed_native_fp8 || v.rows > max_rows || v.accumulate || v.a8_codes ||
        v.a8_deltas || v.a8_sums || v.out_stride != v.count) return false;
    const auto view = checked_q8_0_a8(v);
    if (v.rows >= 2) {
        if (const int mode = rn_wide_mode(); mode != 0) {
            RnWideShape shape;
            const bool rn = mode == 2 ? rn_table_select(view, shape) : (shape = rn_wide_default_shape(view), true);
            if (rn && rn_wide_supported(view, shape))
                return enqueue_q8_0_a8_rn_ready(view, shape, stage, stream);
        }
    }
    const auto q = as_stream(stream);
    if (view.packed == 0U) launch_q8_ready_rows<0>(view, stage, q);
    else if (view.packed == 4U) launch_q8_ready_rows<4>(view, stage, q);
    else if (view.packed == 5U) launch_q8_ready_rows<5>(view, stage, q);
    else launch_q8_ready_rows<6>(view, stage, q);
    check(hipGetLastError(), "Q8 x A8 ready epilogue launch");
    return true;
}

void enqueue_q8_0_a8_pair(const Q8A8View& a, const Q8A8View& b, void* stream) {
    if (a.packed == packed_native_fp8 || b.packed == packed_native_fp8) {  // the native FP8 kernel takes one site per launch
        enqueue_q8_0_a8(a, stream);
        enqueue_q8_0_a8(b, stream);
        return;
    }
    const Q8A8View x = checked_q8_0_a8(a), y = checked_q8_0_a8(b);
    // One launch when both take the same unstaged kernel (same packing and row count, at most 4 rows); else two.
    // Lane l sums groups l, l + 32, ... in ascending order whatever W, so W is chosen for both sites at once.
    if (x.packed != y.packed || x.rows != y.rows || x.rows > 4U || x.a8_codes || y.a8_codes) {
        enqueue_q8_0_a8(a, stream);
        enqueue_q8_0_a8(b, stream);
        return;
    }
    static const unsigned packed_w = [] { const char* e = std::getenv("STRIX_DS41_PACKED_W"); return e && (e[0] == '1' || e[0] == '2') ? unsigned(e[0] - '0') : 4U; }();
    const bool cap = x.packed != 0U && x.rows <= 2;
    const bool w4 = x.per_group % 4U == 0 && y.per_group % 4U == 0 && !(cap && packed_w < 4U);
    const bool w2 = x.per_group % 2U == 0 && y.per_group % 2U == 0 && !(cap && packed_w < 2U);
    const auto q = as_stream(stream);
    if (x.rows == 1) { if (w4) launch_q8_0_a8_pair<1, 4>(x, y, q); else if (w2) launch_q8_0_a8_pair<1, 2>(x, y, q); else launch_q8_0_a8_pair<1, 1>(x, y, q); }
    else if (x.rows == 2) { if (w4) launch_q8_0_a8_pair<2, 4>(x, y, q); else if (w2) launch_q8_0_a8_pair<2, 2>(x, y, q); else launch_q8_0_a8_pair<2, 1>(x, y, q); }
    else { if (w2) launch_q8_0_a8_pair<4, 2>(x, y, q); else launch_q8_0_a8_pair<4, 1>(x, y, q); }
    check(hipGetLastError(), "Q8_0 x A8 pair launch");
}

void enqueue_f16_rows(const F16RowsView& v, void* stream) {
    require((v.weight || v.weight_f8) && v.x && v.out && v.count && v.width && v.rows >= 1, "F16 rows view");
    if (v.rows > max_rows && v.rows <= 16U && f16_rows_lanes_supported(v)) {
        enqueue_f16_rows_lanes(v, stream);
        return;
    }
    if (v.weight_f8) {  // F8 rows: only the lanes kernel reads them
        require(v.rows <= max_rows && f16_rows_lanes_supported(v), "F16 rows view: F8 rows need the lanes kernel (decode rows)");
        // STRIX_DS41_ENGRAM_FAST=1, 1 row: 16-byte F8 loads, activations staged once per 32 rows (hip_ds41_f8_rows_fast.hip; same bits)
        if (f8_rows_fast_enabled() && f8_rows_fast_supported(v)) { enqueue_f8_rows_fast(v, stream); return; }
        enqueue_f16_rows_lanes(v, stream);
        return;
    }
    if (v.rows <= max_rows) {  // decode rows: lanes kernel, else the wide or the plain site kernel (same bits)
        // Same bits, 16-byte loads (hip_ds41_f16_rows.hip); STRIX_DS41_F16_LANES=0 keeps this kernel.
        static const bool lanes = [] { const char* e = std::getenv("STRIX_DS41_F16_LANES"); return !(e && e[0] == '0'); }();
        if (lanes && f16_rows_lanes_supported(v)) { enqueue_f16_rows_lanes(v, stream); return; }
        // DSpark's wide-load kernel (same bits; superseded by the lanes kernel) when the lanes are switched off
        if (f16_rows_wide_enabled && v.width % f16_chunk == 0U && v.x_stride % 4U == 0U) {  // same bits, more bandwidth
            if (v.count >= 2048U)
                hipLaunchKernelGGL(f16_rows_wide_kernel<4>, dim3((v.count + 4U * f16_waves - 1U) / (4U * f16_waves)), dim3(wave * f16_waves), 0,
                                   as_stream(stream), v);
            else
                hipLaunchKernelGGL(f16_rows_wide_kernel<1>, dim3((v.count + f16_waves - 1U) / f16_waves), dim3(wave * f16_waves), 0,
                                   as_stream(stream), v);
            check(hipGetLastError(), "F16 rows launch");
            return;
        }
        hipLaunchKernelGGL(f16_rows_kernel, dim3((v.count + 7U) / 8U), dim3(256), 0, as_stream(stream), v);
        check(hipGetLastError(), "F16 rows launch");
        return;
    }
    // prefill rows: the LDS-tiled kernel, else the many-rows kernel (same lane classes, same bits)
    const bool tiled = v.width % f16t_slab == 0U && v.x_stride % 4U == 0U && (reinterpret_cast<std::uintptr_t>(v.x) & 15U) == 0U &&
                       (reinterpret_cast<std::uintptr_t>(v.weight) & 15U) == 0U && v.width % 8U == 0U;
    if (const char* env = std::getenv("STRIX_DS41_F16_TILED"); tiled && (env == nullptr || env[0] != '0')) {  // 0: old kernel
        // STRIX_DS41_F16_TILE: a 32 x 32 (8 waves), b 64 x 32 (16 waves), c 64 x 64 (32 waves; default); then the LDS read
        // width 1, 2 or 4 (default 2), e.g. "b4"
        const char* tile = std::getenv("STRIX_DS41_F16_TILE");
        const char t = tile != nullptr ? tile[0] : 'c';
        const char w = tile != nullptr && tile[0] != 0 && tile[1] != 0 ? tile[1] : '2';
        const auto launch = [&]<unsigned V>() {
            if (t == 'a') launch_f16_tiled<32, 32, 16, V>(v, as_stream(stream));
            else if (t == 'c') launch_f16_tiled<64, 64, 8, V>(v, as_stream(stream));
            else launch_f16_tiled<64, 32, 8, V>(v, as_stream(stream));
        };
        if (w == '1') launch.template operator()<1U>();
        else if (w == '4') launch.template operator()<4U>();
        else launch.template operator()<2U>();
    } else
        hipLaunchKernelGGL(f16_many_kernel, dim3((v.count + f16_block_outputs - 1U) / f16_block_outputs, (v.rows + f16_tile_rows - 1U) / f16_tile_rows),
                           dim3(256), 0, as_stream(stream), v);
    check(hipGetLastError(), "F16 rows launch");
}

bool engram_f16_pf_supported(const F16RowsView& v) noexcept {
    return v.weight && !v.weight_f8 && v.x && v.out && v.width == 6144U && v.count == 25600U && v.first == 0U && v.rows >= 32U &&
           v.rows <= 8192U && v.x_stride >= v.width && v.x_stride % 4U == 0U && v.out_stride >= v.count &&
           (reinterpret_cast<std::uintptr_t>(v.x) & 15U) == 0U && (reinterpret_cast<std::uintptr_t>(v.weight) & 15U) == 0U;
}

void enqueue_engram_f16_pf(const F16RowsView& v, void* stream) {
    require(engram_f16_pf_supported(v), "Engram F16 prefetch tile view");
    const unsigned row_tiles = (v.rows + 47U) / 48U, output_tiles = (v.count + 63U) / 64U;
    hipLaunchKernelGGL((f16_tiled_pf_kernel<48, 64, 8, 6>), dim3(row_tiles * output_tiles), dim3(48U / 6U * (64U / 16U) * 32U), 0,
                       as_stream(stream), v, row_tiles, output_tiles);
    check(hipGetLastError(), "Engram F16 prefetch tile launch");
}

bool engram_f16_pf_admitted(const F16RowsView& v) noexcept {
    // The 48-row tile is ~14 % faster per padded row than the 64 x 64 default; admit it while its row padding is at most
    // 9/8 of that tile's (measured: 64 rows, 96 vs 64, 1.16x slower; 128 and 256 at the bound 0.87x / 0.94x).
    return engram_f16_pf_supported(v) && (v.rows + 47U) / 48U * 48U * 8U <= (v.rows + 63U) / 64U * 64U * 9U;
}

void enqueue_engram_f16_rows(const F16RowsView& v, void* stream) {
    // Read on every call (in-process A/B); =0 restores enqueue_f16_rows with all its controls.
    const char* e = std::getenv("STRIX_DS41_ENGRAM_F16_PF");
    if (!(e && e[0] == '0') && engram_f16_pf_admitted(v)) enqueue_engram_f16_pf(v, stream);
    else enqueue_f16_rows(v, stream);
}

void enqueue_f16_rows_pair(const F16RowsView& a, const F16RowsView& b, void* stream) {
    // STRIX_DS41_F16_PAIR (on by default since P16, =0 off): one launch when both views take the lanes kernels (hip_ds41_f16_rows.hip)
    static const bool pair = [] { const char* e = std::getenv("STRIX_DS41_F16_PAIR"); return !(e && e[0] == '0'); }();
    static const bool lanes = [] { const char* e = std::getenv("STRIX_DS41_F16_LANES"); return !(e && e[0] == '0'); }();
    if (pair && lanes && a.rows <= max_rows && f16_rows_pair_supported(a, b)) {
        enqueue_f16_rows_lanes_pair(a, b, stream);
        return;
    }
    enqueue_f16_rows(a, stream);
    enqueue_f16_rows(b, stream);
}

void set_f16_rows_wide(bool on) { f16_rows_wide_enabled = on; }

void enqueue_embedding(const std::uint16_t* table, const std::uint32_t* tokens, std::uint32_t rows, std::uint32_t hidden,
                       float* stream_out, void* stream) {
    const auto total = static_cast<unsigned long long>(rows) * hc_copies * hidden;
    hipLaunchKernelGGL(embedding_kernel, dim3(blocks_for(total, 256)), dim3(256), 0, as_stream(stream), table, tokens, hidden,
                       total, stream_out);
    check(hipGetLastError(), "embedding launch");
}

std::uint32_t argmax_partial_blocks(std::uint32_t count) { return (count + argmax_span - 1U) / argmax_span; }

void enqueue_argmax(const float* logits, std::uint32_t rows, std::uint32_t count, std::uint32_t stride, std::uint32_t first_index,
                    ArgmaxCandidate* partial, ArgmaxCandidate* out, void* stream) {
    require(logits && partial && out && rows && count && stride >= count, "argmax view");
    const unsigned blocks = argmax_partial_blocks(count);
    hipLaunchKernelGGL(argmax_partial_kernel, dim3(blocks, rows), dim3(argmax_threads), 0, as_stream(stream), logits, count,
                       stride, first_index, blocks, partial);
    hipLaunchKernelGGL(argmax_final_kernel, dim3(rows), dim3(argmax_threads), 0, as_stream(stream), partial, blocks, out);
    check(hipGetLastError(), "argmax launch");
}

void enqueue_combine_argmax(const ArgmaxCandidate* a, const ArgmaxCandidate* b, std::uint32_t rows, ArgmaxCandidate* out,
                            void* stream) {
    hipLaunchKernelGGL(combine_argmax_kernel, dim3((rows + 63U) / 64U), dim3(64), 0, as_stream(stream), a, b, rows, out);
    check(hipGetLastError(), "argmax combine launch");
}

void enqueue_engram_rows_bf16(const std::uint8_t* rows, std::uint32_t row_count, float* out, void* stream) {
    enqueue_decode_engram_rows(rows, row_count, out, stream);
    const auto total = static_cast<unsigned long long>(row_count) * engram_row_elements;
    hipLaunchKernelGGL(round_bf16_kernel, dim3(blocks_for(total, 256)), dim3(256), 0, as_stream(stream), out, total);
    check(hipGetLastError(), "Engram bf16 launch");
}

PrefillRoutes prefill_site_routes(std::uint32_t rows, std::uint32_t groups, std::uint32_t batch_rows) {
    require(rows && groups, "prefill site routes");
    std::vector<std::uint32_t> ids(static_cast<std::size_t>(rows) * groups);
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<std::uint32_t>(i % groups);
    return build_prefill_routes(ids, rows, groups, groups, groups, batch_rows);
}

void enqueue_q8_0_a8_prefill(const Q8A8PrefillView& v, void* stream) {
    require(v.weight && v.codes && v.deltas && v.out && v.groups && v.per_group % 128U == 0 && v.per_group && v.width &&
                v.width % 32U == 0 && v.width <= 5120U,
            "Q8_0 x A8 prefill view");
    PrefillDownView d;
    d.weights = v.weight;
    d.row_bytes = static_cast<std::uint64_t>(v.width) / 32U * 34U;
    d.expert_stride_bytes = d.row_bytes * v.per_group;
    d.width = v.width;
    d.outputs = v.per_group;
    d.routes = v.routes;
    d.codes = v.codes;
    d.deltas = v.deltas;
    d.out = v.out;
    enqueue_prefill_down(WeightFormat::q8_0, d, stream);
}

// Post-order form (tall = 96..99, opt-in): the same group terms, class sums and tree as the staged wide form, bit for
// bit, on 64 x 64 blocks. Block: 8 waves; wave w owns rows 16 (w & 3) .. +16 and outputs 32 (w >> 2) .. +32 (two
// 16 x 16 tiles) and walks ALL groups itself, class-major: the classes in the order of the tree (T0 T4 T2 T6 T1 T5 T3
// T7, with T(v) = (c_v + c_v+16) + (c_v+8 + c_v+24)), each class's groups ascending from +0 as before, and each finished
// class folds into the tree at once (post-order): S(w) = T(w) + T(w + 4), y = (S0 + S2) + (S1 + S3). At most six partial
// sums per output are live, instead of the 32 class sums. A chunk of post_chunk groups (in that order) is staged in
// LDS for the block: activation codes (row stride 48 B), weight codes (one 34-byte block per output and group, read
// once per block, by dword loads), weight scales as f32 and row deltas, so every operand is read from global memory
// once per block. Variants (measured against each other, all the same bits):
//   Prefetch: the next chunk's operands are loaded into registers while the current chunk runs its WMMAs;
//   Direct: the classes at tree positions 4t and 4t + 2 accumulate straight into the pair partials P and Q (from +0,
//           the same adds), so a finished class needs no copy.
constexpr unsigned post_waves = 8, post_chunk = 4, post_row = 48;
static_assert(post_chunk * 64U == 32U * post_waves, "one weight block and one delta per thread and chunk");
__device__ __constant__ unsigned char post_class_order[32] = {0,  16, 8,  24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30,
                                                             1,  17, 9,  25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31};
template <bool Prefetch, bool Direct>
__global__ __launch_bounds__(32 * post_waves) void q8_0_a8_post_kernel(Q8A8DensePrefillView v) {
    __shared__ __attribute__((aligned(16))) signed char sa[post_chunk][64][post_row];  // activation codes [group][row]
    __shared__ __attribute__((aligned(16))) signed char sw[post_chunk][64][post_row];  // weight codes [group][output]
    __shared__ float ss[post_chunk][64];                                               // weight scales
    __shared__ __attribute__((aligned(16))) float sd[post_chunk][4][2][8];             // deltas [group][tile][h][e]
    __shared__ unsigned short order[256];                                              // group of flattened step f
    __shared__ unsigned char last[256];                                                // step f ends its class
    const unsigned w = threadIdx.x / wave, l = threadIdx.x % wave, h = l >> 4U, lo = l & 15U;
    const unsigned rt = w & 3U, cp = w >> 2U;
    const unsigned r0 = blockIdx.x * 64U, o0 = blockIdx.y * 64U;
    const unsigned groups = v.width / 32U, row_bytes = (v.row_width ? v.row_width : v.width) / 32U * 34U;
    if (threadIdx.x == 0) {  // the flattened class-major order (groups <= 256)
        unsigned f = 0;
        for (unsigned p = 0; p < 32U; ++p) {
            const unsigned c = post_class_order[p];
            for (unsigned g = c; g < groups; g += 32U) {
                order[f] = static_cast<unsigned short>(g);
                last[f] = g + 32U >= groups ? static_cast<unsigned char>(1U) : static_cast<unsigned char>(0U);
                ++f;
            }
        }
    }
    __syncthreads();
    // live partials of the tree: cs (class), P, Q (pairs of a T), TA (first T of an S), SA (first S of a half), L (S0+S2)
    float cs[2][8], P[2][8], Q[2][8], TA[2][8], SA[2][8], L[2][8];
#pragma unroll
    for (unsigned t = 0; t < 2; ++t)
#pragma unroll
        for (unsigned e = 0; e < 8; ++e) cs[t][e] = P[t][e] = Q[t][e] = TA[t][e] = SA[t][e] = L[t][e] = 0.0F;
    const v8i magic{wide_magic, wide_magic, wide_magic, wide_magic, wide_magic, wide_magic, wide_magic, wide_magic};
    unsigned p = 0;  // tree position of the class being summed (uniform across the block)
    // the finished class (its sum in cs, or with Direct in P / Q at positions 4t / 4t + 2) at tree position p; cs back
    // to +0, p advances. An empty class (groups < 32) folds as a +0 sum, exactly as its zero class sum did.
    const auto fold_class = [&]() {
#pragma unroll
        for (unsigned t = 0; t < 2; ++t)
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) {
                const float c_ = cs[t][e];
                switch (p & 3U) {
                    case 0: if constexpr (!Direct) P[t][e] = c_; break;
                    case 1: P[t][e] = __fadd_rn(P[t][e], c_); break;
                    case 2: if constexpr (!Direct) Q[t][e] = c_; break;
                    default: {
                        const float T = __fadd_rn(P[t][e], __fadd_rn(Q[t][e], c_));
                        if constexpr (Direct) P[t][e] = Q[t][e] = 0.0F;  // the next T's pair sums start at +0
                        if (((p >> 2U) & 1U) == 0U) TA[t][e] = T;
                        else {
                            const float S = __fadd_rn(TA[t][e], T);
                            switch (p >> 3U) {
                                case 0: SA[t][e] = S; break;
                                case 1: L[t][e] = __fadd_rn(SA[t][e], S); break;
                                case 2: SA[t][e] = S; break;
                                default: L[t][e] = __fadd_rn(L[t][e], __fadd_rn(SA[t][e], S)); break;
                            }
                        }
                    }
                }
                cs[t][e] = 0.0F;
            }
        ++p;
    };
    // Staging (thread i of 256): activation pieces i and i + 256 (group k = piece / 128, row, half), weight block i
    // (group i / 64, output i % 64), delta i (group i / 64, row i % 64).
    int4 rx[2];
    unsigned short rdw = 0;
    int rq[8];
    float rdl = 0.0F;
    const auto load_chunk = [&](unsigned f0) {
        const unsigned n = groups - f0 < post_chunk ? groups - f0 : post_chunk;
#pragma unroll
        for (unsigned j = 0; j < 2; ++j) {
            const unsigned i = threadIdx.x + j * 256U, k = i / 128U, r = (i / 2U) % 64U, half = i % 2U, row = r0 + r;
            rx[j] = int4{0, 0, 0, 0};
            if (k < n && row < v.rows)
                rx[j] = *reinterpret_cast<const int4*>(reinterpret_cast<const signed char*>(v.codes) + static_cast<std::size_t>(row) * v.x_stride +
                                                       32U * order[f0 + k] + 16U * half);
        }
        const unsigned k = threadIdx.x / 64U, o = threadIdx.x % 64U, out = o0 + o;
        if (k < n) {
            // a 34-byte block starts at 4 n (scale aligned, codes at +2: nine dwords realigned by 2 bytes) or at 4 n + 2
            // (scale alone, codes dword-aligned): only dword loads either way
            const std::uint8_t* block = v.weight + static_cast<std::size_t>(out < v.count ? out : v.count - 1U) * row_bytes +
                                        (v.k_offset / 32U + order[f0 + k]) * 34U;
            if ((reinterpret_cast<std::uintptr_t>(block) & 3U) == 0U) {
                const auto* u = reinterpret_cast<const unsigned*>(block);
                unsigned t[9];
#pragma unroll
                for (unsigned m = 0; m < 9; ++m) t[m] = u[m];
                rdw = static_cast<unsigned short>(t[0] & 0xffffU);
#pragma unroll
                for (unsigned m = 0; m < 8; ++m) rq[m] = static_cast<int>(__builtin_amdgcn_alignbyte(t[m + 1], t[m], 2U));
            } else {
                rdw = *reinterpret_cast<const unsigned short*>(block);
                const auto* u = reinterpret_cast<const unsigned*>(block + 2);
#pragma unroll
                for (unsigned m = 0; m < 8; ++m) rq[m] = static_cast<int>(u[m]);
            }
            const unsigned row = r0 + o;
            rdl = row < v.rows ? v.deltas[static_cast<std::size_t>(row) * (v.x_stride / 32U) + order[f0 + k]] : 0.0F;
        }
    };
    const auto store_chunk = [&](unsigned f0) {
        const unsigned n = groups - f0 < post_chunk ? groups - f0 : post_chunk;
#pragma unroll
        for (unsigned j = 0; j < 2; ++j) {
            const unsigned i = threadIdx.x + j * 256U, k = i / 128U, r = (i / 2U) % 64U, half = i % 2U;
            if (k < n) *reinterpret_cast<int4*>(&sa[k][r][16U * half]) = rx[j];
        }
        const unsigned k = threadIdx.x / 64U, o = threadIdx.x % 64U;
        if (k < n) {
            ss[k][o] = f16_to_float(rdw);
            *reinterpret_cast<int4*>(&sw[k][o][0]) = int4{rq[0], rq[1], rq[2], rq[3]};
            *reinterpret_cast<int4*>(&sw[k][o][16]) = int4{rq[4], rq[5], rq[6], rq[7]};
            (&sd[k][0][0][0])[(o / 16U) * 16U + (o % 2U) * 8U + (o % 16U) / 2U] = rdl;
        }
    };
    if constexpr (Prefetch) load_chunk(0);
    for (unsigned f0 = 0; f0 < groups; f0 += post_chunk) {
        __syncthreads();  // the previous chunk is consumed
        const unsigned n = groups - f0 < post_chunk ? groups - f0 : post_chunk;
        if constexpr (!Prefetch) load_chunk(f0);
        store_chunk(f0);
        __syncthreads();
        if constexpr (Prefetch)
            if (f0 + post_chunk < groups) load_chunk(f0 + post_chunk);
        // the chunk's class ends, once per chunk, as scalars
        unsigned ends = 0;
#pragma unroll
        for (unsigned k = 0; k < post_chunk; ++k)
            if (k < n) ends |= static_cast<unsigned>(__builtin_amdgcn_readfirstlane(last[f0 + k])) << k;
#pragma unroll
        for (unsigned k = 0; k < post_chunk; ++k) {
            if (k >= n) break;
            const int4 x0 = *reinterpret_cast<const int4*>(&sa[k][rt * 16U + lo][0]);
            const int4 x1 = *reinterpret_cast<const int4*>(&sa[k][rt * 16U + lo][16]);
            const v4i a0{x0.x, x0.y, x0.z, x0.w}, a1{x1.x, x1.y, x1.z, x1.w};
            const float4 d0 = *reinterpret_cast<const float4*>(&sd[k][rt][h][0]), d1 = *reinterpret_cast<const float4*>(&sd[k][rt][h][4]);
            const float rd[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
            float term[2][8];
#pragma unroll
            for (unsigned t = 0; t < 2; ++t) {
                const unsigned o = cp * 32U + t * 16U + lo;
                const int4 w0 = *reinterpret_cast<const int4*>(&sw[k][o][0]), w1 = *reinterpret_cast<const int4*>(&sw[k][o][16]);
                const v4i b0{w0.x, w0.y, w0.z, w0.w}, b1{w1.x, w1.y, w1.z, w1.w};
                const float scale = ss[k][o];
                v8i c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, true, b0, magic, false);
                c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, true, b1, c, false);
#pragma unroll
                for (unsigned e = 0; e < 8; ++e)
                    term[t][e] = __fmul_rn(__fsub_rn(__int_as_float(c[e]), 0x1.8p23F), __fmul_rn(scale, rd[e]));
            }
            // the group term into the class sum (ascending): with Direct straight into P / Q at positions 4t / 4t + 2
            const unsigned slot = Direct ? (p & 3U) : 1U;
#pragma unroll
            for (unsigned t = 0; t < 2; ++t)
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) {
                    if (slot == 0U) P[t][e] = __fadd_rn(P[t][e], term[t][e]);
                    else if (slot == 2U) Q[t][e] = __fadd_rn(Q[t][e], term[t][e]);
                    else cs[t][e] = __fadd_rn(cs[t][e], term[t][e]);
                }
            if ((ends >> k) & 1U) {
                fold_class();
                while (p < 32U && post_class_order[p] >= groups) fold_class();  // empty classes (groups < 32): +0 sums
            }
        }
    }
    while (p < 32U) fold_class();
#pragma unroll
    for (unsigned t = 0; t < 2; ++t)
#pragma unroll
        for (unsigned e = 0; e < 8; ++e) {
            const unsigned o = o0 + cp * 32U + t * 16U + lo, r = r0 + rt * 16U + 2U * e + h;
            if (o < v.count && r < v.rows) {
                float* dst = v.out + static_cast<std::size_t>(r) * v.out_stride + o;
                *dst = v.accumulate ? *dst + L[t][e] : L[t][e];
            }
        }
}

bool enqueue_q8_0_a8_dense_prefill(const Q8A8DensePrefillView& v, void* stream) {
    require(v.weight && v.codes && v.deltas && v.out && v.count && v.rows && v.width && v.width % 32U == 0 && v.x_stride >= v.width &&
                v.x_stride % 16U == 0 && v.out_stride >= v.count && reinterpret_cast<std::uintptr_t>(v.codes) % 16U == 0 &&
                (!v.row_width || (v.row_width % 32U == 0 && v.k_offset % 32U == 0 && v.k_offset + v.width <= v.row_width)) &&
                (!v.a8_codes || (v.a8_deltas && v.a8_sums && !v.accumulate && v.count % 32U == 0 && v.a8_stride >= v.count &&
                                 v.a8_stride % 32U == 0)),
            "Q8_0 x A8 dense prefill view");
    unsigned outputs = 32U;
    if (prefill_tuning::tiles())
        outputs = prefill_tuning::tile_override("STRIX_DS41_PREFILL_DENSE_TILE", prefill_tuning::dense_outputs(v.width, v.count, v.rows));
    const dim3 grid((v.rows + 15U) / 16U, (v.count + outputs - 1U) / outputs);
    static const bool staged = [] {
        const char* e = std::getenv("STRIX_DS41_DENSE_FORM");
        return !(e && e[0] == '0');
    }();
    // STRIX_DS41_DENSE_TALL=22 / 41 / 14 / 21 / 12: RT OT of the tall form for every call (staged only); auto: by shape,
    // from the gfx1151 smoke (prefill measurements): >= 3072 rows, or 2048 rows and >= 1024 outputs: 1 x 4 (q_a / kv);
    // >= 256 rows with >= 4096 outputs and width >= 4096: 1 x 2 (wo_b halves); else the wide form. Every form: the same bits.
    static const unsigned tall_env = [] {
        const char* e = std::getenv("STRIX_DS41_DENSE_TALL");
        if (e != nullptr && std::string(e) == "auto") return 1U;
        return e ? static_cast<unsigned>(std::strtoul(e, nullptr, 10)) : 0U;
    }();
    unsigned tall = v.tall ? v.tall : tall_env;
    if (tall == 1U) {
        // auto, by the production attention sites (site benchmark, gfx1151, every form bitwise): from 256 rows the Lean 4
        // forms, 1 x 4 rows-first (814) for q_a / kv up to 6143 rows and q_b (wide outputs, K <= 1536), column-first
        // (914) for q_a / kv from 6144 rows and o_a groups (K 4096 -> 1024) from 512 rows, 2 x 2 (822) for the o_b
        // K-halves (row_width 8192) and smaller o_a; else the earlier rule (14 / 12 / wide). Views past the lean
        // buffer limit take the Lean 0 form of the same tile.
        const bool lean_ok = static_cast<std::size_t>(v.rows) * v.x_stride < 0x7fffffffULL &&
                             static_cast<std::size_t>(v.count) * ((v.row_width ? v.row_width : v.width) / 32U * 34U) < 0x7fffffffULL;
        unsigned pick = 0U;
        if (v.rows >= 256U) {
            if (v.width == 5120U && v.count <= 1536U) pick = v.rows >= 6144U ? 914U : 814U;
            else if (v.width <= 1536U && v.count >= 4096U) pick = 814U;
            else if (v.width == 4096U && v.count == 1024U) pick = v.rows >= 512U ? 914U : 822U;
            else if (v.row_width == 8192U && v.width == 4096U) pick = 822U;
        }
        // STRIX_DS41_DENSE_TALL_ROTATE=1 (exactness gates only): every call takes the next of 814 / 914 / 822 in turn,
        // whatever its shape, so one --reference run covers the three selector forms on every site
        static const bool rotate = [] {
            const char* e = std::getenv("STRIX_DS41_DENSE_TALL_ROTATE");
            return e != nullptr && e[0] == '1';
        }();
        static std::atomic<unsigned> turn{0U};
        if (rotate) pick = std::array<unsigned, 3>{814U, 914U, 822U}[turn.fetch_add(1U, std::memory_order_relaxed) % 3U];
        if (pick != 0U && !lean_ok) pick = pick == 914U ? 114U : pick == 822U ? 22U : 14U;
        tall = pick != 0U ? pick
               : v.rows >= 3072U || (v.rows >= 2048U && v.count >= 1024U) ? 14U
               : v.rows >= 256U && v.count >= 4096U && v.width >= 4096U  ? 12U
                                                                          : 0U;
    }
    // STRIX_DS41_DENSE_STATS=1 (diagnostics): calls and rows per (form, width, count, row_width), printed at exit
    static const bool stats = [] {
        const char* e = std::getenv("STRIX_DS41_DENSE_STATS");
        return e != nullptr && e[0] == '1';
    }();
    if (stats) {
        struct Count { unsigned long long calls = 0, rows = 0; };
        static std::mutex lock;
        static std::map<std::array<unsigned, 4>, Count> counts;
        static const bool printer = [] {
            std::atexit([] {
                std::lock_guard<std::mutex> g(lock);
                for (const auto& [k, c] : counts)
                    std::fprintf(stderr, "dense stats: form %u width %u count %u row_width %u calls %llu rows %llu\n", k[0], k[1], k[2], k[3], c.calls, c.rows);
            });
            return true;
        }();
        (void)printer;
        std::lock_guard<std::mutex> g(lock);
        auto& c = counts[{staged ? tall : 0U, v.width, v.count, v.row_width}];
        ++c.calls;
        c.rows += v.rows;
    }
    const auto tall_grid = [&](unsigned rt, unsigned ot) { return dim3((v.rows + 16U * rt - 1U) / (16U * rt), (v.count + 16U * ot - 1U) / (16U * ot)); };
    if (tall >= 96U && tall <= 99U) {  // post-order forms: 99 prefetch + direct, 98 direct, 97 prefetch, 96 neither
        require(v.width / 32U <= 256U, "Q8_0 x A8 dense post form: at most 256 groups");
        const dim3 pg((v.rows + 63U) / 64U, (v.count + 63U) / 64U), pb(32 * post_waves);
        if (tall == 99U) hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_0_a8_post_kernel<true, true>), pg, pb, 0, as_stream(stream), v);
        else if (tall == 98U) hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_0_a8_post_kernel<false, true>), pg, pb, 0, as_stream(stream), v);
        else if (tall == 97U) hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_0_a8_post_kernel<true, false>), pg, pb, 0, as_stream(stream), v);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_0_a8_post_kernel<false, false>), pg, pb, 0, as_stream(stream), v);
    } else if (staged && tall == 22U) hipLaunchKernelGGL((q8_0_a8_tall_kernel<2, 2>), tall_grid(2, 2), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 41U) hipLaunchKernelGGL((q8_0_a8_tall_kernel<4, 1>), tall_grid(4, 1), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 14U) hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 4>), tall_grid(1, 4), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 114U)
        hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 4, true>), dim3(tall_grid(1, 4).y, tall_grid(1, 4).x), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 112U)
        hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 2, true>), dim3(tall_grid(1, 2).y, tall_grid(1, 2).x), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 214U) hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 4, false, 1U>), tall_grid(1, 4), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 212U) hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 2, false, 1U>), tall_grid(1, 2), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 314U)
        hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 4, true, 1U>), dim3(tall_grid(1, 4).y, tall_grid(1, 4).x), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 312U)
        hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 2, true, 1U>), dim3(tall_grid(1, 2).y, tall_grid(1, 2).x), dim3(32 * tall_waves), 0, as_stream(stream), v);
    // Lean >= 2 forms address the activations / weights through raw buffers of at most 0x7fffffff bytes
    else if ((tall == 414U || tall == 412U || tall == 514U || tall == 614U || tall == 612U || tall == 714U || tall == 422U ||
              tall == 814U || tall == 812U || tall == 914U || tall == 822U || tall >= 1500U) &&
             (static_cast<std::size_t>(v.rows) * v.x_stride >= 0x7fffffffULL ||
              static_cast<std::size_t>(v.count) * ((v.row_width ? v.row_width : v.width) / 32U * 34U) >= 0x7fffffffULL))
        throw std::invalid_argument("Q8_0 x A8 dense prefill: lean buffer form beyond 2 GiB");
    else if (staged && tall == 414U) hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 4, false, 2U>), tall_grid(1, 4), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 412U) hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 2, false, 2U>), tall_grid(1, 2), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 514U)
        hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 4, true, 2U>), dim3(tall_grid(1, 4).y, tall_grid(1, 4).x), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 614U) hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 4, false, 3U>), tall_grid(1, 4), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 612U) hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 2, false, 3U>), tall_grid(1, 2), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 714U)
        hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 4, true, 3U>), dim3(tall_grid(1, 4).y, tall_grid(1, 4).x), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 814U) hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 4, false, 4U>), tall_grid(1, 4), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 812U) hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 2, false, 4U>), tall_grid(1, 2), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 914U)
        hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 4, true, 4U>), dim3(tall_grid(1, 4).y, tall_grid(1, 4).x), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 422U) hipLaunchKernelGGL((q8_0_a8_tall_kernel<2, 2, false, 2U>), tall_grid(2, 2), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 822U) hipLaunchKernelGGL((q8_0_a8_tall_kernel<2, 2, false, 4U>), tall_grid(2, 2), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 1514U) hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 4, false, 5U>), tall_grid(1, 4), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 1522U) hipLaunchKernelGGL((q8_0_a8_tall_kernel<2, 2, false, 5U>), tall_grid(2, 2), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 1614U)
        hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 4, true, 5U>), dim3(tall_grid(1, 4).y, tall_grid(1, 4).x), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 1714U) hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 4, false, 6U>), tall_grid(1, 4), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 1722U) hipLaunchKernelGGL((q8_0_a8_tall_kernel<2, 2, false, 6U>), tall_grid(2, 2), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 1814U)
        hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 4, true, 6U>), dim3(tall_grid(1, 4).y, tall_grid(1, 4).x), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 1914U) hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 4, false, 7U>), tall_grid(1, 4), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 1922U) hipLaunchKernelGGL((q8_0_a8_tall_kernel<2, 2, false, 7U>), tall_grid(2, 2), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 2014U)
        hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 4, true, 7U>), dim3(tall_grid(1, 4).y, tall_grid(1, 4).x), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 2114U) hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 4, false, 8U>), tall_grid(1, 4), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 2122U) hipLaunchKernelGGL((q8_0_a8_tall_kernel<2, 2, false, 8U>), tall_grid(2, 2), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 2214U)
        hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 4, true, 8U>), dim3(tall_grid(1, 4).y, tall_grid(1, 4).x), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 21U) hipLaunchKernelGGL((q8_0_a8_tall_kernel<2, 1>), tall_grid(2, 1), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (staged && tall == 12U) hipLaunchKernelGGL((q8_0_a8_tall_kernel<1, 2>), tall_grid(1, 2), dim3(32 * tall_waves), 0, as_stream(stream), v);
    else if (outputs == 16U) {
        if (staged) hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_0_a8_wide_kernel<true, 1>), grid, dim3(32 * wide_waves), 0, as_stream(stream), v);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_0_a8_wide_kernel<false, 1>), grid, dim3(32 * wide_waves), 0, as_stream(stream), v);
    } else {
        if (staged) hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_0_a8_wide_kernel<true, 2>), grid, dim3(32 * wide_waves), 0, as_stream(stream), v);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_0_a8_wide_kernel<false, 2>), grid, dim3(32 * wide_waves), 0, as_stream(stream), v);
    }
    check(hipGetLastError(), "Q8_0 x A8 dense prefill launch");
    // the tall forms with an even OT wrote the outputs' A8 (the others ignore a8_codes)
    static constexpr unsigned a8_forms[] = {22U,  14U,   114U,  112U,  214U,  212U,  314U,  312U,  414U,  412U,  514U,  614U,
                                            612U, 714U,  814U,  812U,  914U,  422U,  822U,  1514U, 1522U, 1614U, 1714U, 1722U,
                                            1814U, 1914U, 1922U, 2014U, 2114U, 2122U, 2214U, 12U};
    bool a8 = false;
    for (const unsigned f : a8_forms) a8 = a8 || f == tall;
    return v.a8_codes != nullptr && staged && a8;
}

void enqueue_engram_inject(float* hc_stream, const float* kv, const float* q_norm, const float* k_norm, std::uint32_t rows,
                           std::uint32_t hidden, float eps, void* stream) {
    require(hc_stream && kv && q_norm && k_norm && rows && hidden, "Engram injection view");
    const float inv = static_cast<float>(1.0 / __builtin_sqrt(static_cast<double>(hidden)));
    const unsigned chunks = (hidden + chunk - 1U) / chunk;
    require(chunks * hc_copies <= 32U, "Engram injection hidden size");
    hipLaunchKernelGGL(engram_inject_kernel, dim3(rows), dim3(wave * hc_copies * chunks), 0, as_stream(stream), hc_stream, kv, q_norm,
                       k_norm, hidden, eps, inv);
    check(hipGetLastError(), "Engram injection launch");
}

// ---- launch accesses ---------------------------------------------------------------------------
// Every non-null pointer of a view is declared (ranges start at the pointer the kernel receives), whether or
// not the configuration uses it, so an audit of the kernel arguments finds each of them in a range.
LaunchAccess launch_access(const HcSiteView& v) {
    LaunchAccess a;
    const std::uint64_t H = v.hidden, n = H * hc_copies, R = v.rows;
    const std::uint64_t partial = hc_mix_partial_floats(v.rows, v.hidden) * 4, counters = hc_site_counter_count(v.rows) * 4;
    a.read(v.sub, R * H * 4);
    a.read(v.residual, R * n * 4);
    a.read(v.expand_mixes, R * sizeof(HcMix));
    a.read(v.stream, R * n * 4);
    if (v.sub) a.write(v.stream, R * n * 4);
    a.read(v.fn, hc_mix_count * n * 2);
    a.read(v.base, hc_mix_count * 4);
    a.read(v.scale, 3 * 4);
    a.read(v.partial, partial);
    a.write(v.partial, partial);
    a.read(v.counters, counters);
    a.write(v.counters, counters);
    a.write(v.mixes_out, R * sizeof(HcMix));
    a.read(v.collapse_mixes, R * sizeof(HcMix));
    a.read(v.norm_weight, H * 4);
    a.write(v.out, ((R - 1) * v.out_stride + H) * 4);
    a.write(v.codes, R * v.a8_stride);
    a.write(v.deltas, R * v.a8_stride / 32 * 4);
    a.write(v.sums, R * v.a8_stride / 32 * 4);
    a.write(v.collapse_out, R * H * 4);
    return a;
}

LaunchAccess launch_access(const HcMixView& v) {
    LaunchAccess a;
    const std::uint64_t n = std::uint64_t{v.hidden} * hc_copies;
    a.read(v.stream, v.rows * n * 4);
    a.read(v.fn, hc_mix_count * n * 2);
    a.read(v.base, hc_mix_count * 4);
    a.read(v.scale, 3 * 4);
    a.read(v.partial, hc_mix_partial_floats(v.rows, v.hidden) * 4);
    a.write(v.partial, hc_mix_partial_floats(v.rows, v.hidden) * 4);
    a.write(v.out, std::uint64_t{v.rows} * sizeof(HcMix));
    return a;
}

LaunchAccess launch_access(const NormView& v) {
    LaunchAccess a;
    const std::uint64_t R = v.rows, n = v.n;
    a.read(v.x, ((R - 1) * v.x_stride + n) * 4);
    a.read(v.hc_stream, R * hc_copies * n * 4);
    a.read(v.mixes, R * sizeof(HcMix));
    a.read(v.weight, n * 4);
    a.read(v.positions, R * 4);
    a.read(v.rope_cos, 0x7fffffffULL);  // table rows by position: the whole table may be read
    a.read(v.rope_sin, 0x7fffffffULL);
    a.write(v.out, ((R - 1) * v.out_stride + n) * 4);
    a.write(v.codes, R * v.a8_stride);
    a.write(v.deltas, R * v.a8_stride / 32 * 4);
    a.write(v.sums, R * v.a8_stride / 32 * 4);
    return a;
}

LaunchAccess launch_access(const Q8A8View& v) {
    LaunchAccess a;
    const std::uint32_t per_group = v.per_group ? v.per_group : v.count;
    const std::uint64_t R = v.rows, in = std::uint64_t{v.count / per_group} * v.width;
    const std::uint64_t packed_bytes = v.packed == 4U ? 18U : v.packed == 5U ? 22U : v.packed == 6U ? 26U : 0U;
    if (v.packed == packed_native_fp8) {  // codes [rows][row width], exponents [rows / 32][row width / 32]
        const std::uint64_t full = v.row_width ? v.row_width : v.width, end = std::uint64_t{v.first} + v.count;
        a.read(v.weight, end * full);
        a.read(v.scales, (end + 31U) / 32U * (full / 32U));
    } else {
        a.read(v.weight, packed_bytes ? (std::uint64_t{v.first} + v.count) * ((v.row_width ? v.row_width : v.width) / 32U) * packed_bytes
                                      : q8_bytes(std::uint64_t{v.first} + v.count, v.row_width ? v.row_width : v.width));
    }
    a.read(v.codes, (R - 1) * v.x_stride + in);
    a.read(v.deltas, ((R - 1) * v.x_stride + in) / 32 * 4);
    if (v.accumulate) a.read(v.out, ((R - 1) * v.out_stride + v.count) * 4);
    a.write(v.out, ((R - 1) * v.out_stride + v.count) * 4);
    return a;
}

LaunchAccess launch_access(const F16RowsView& v) {
    LaunchAccess a;
    const std::uint64_t R = v.rows;
    a.read(v.weight, (std::uint64_t{v.first} + v.count) * v.width * 2);
    a.read(v.x, ((R - 1) * v.x_stride + v.width) * 4);
    a.write(v.out, ((R - 1) * v.out_stride + v.count) * 4);
    return a;
}

LaunchAccess launch_access_expand(const float* sub, const float* residual, const HcMix* mixes, std::uint32_t rows, std::uint32_t hidden,
                                  const float* out) {
    LaunchAccess a;
    const std::uint64_t n = std::uint64_t{hidden} * hc_copies;
    a.read(sub, std::uint64_t{rows} * hidden * 4);
    a.read(residual, rows * n * 4);
    a.read(mixes, std::uint64_t{rows} * sizeof(HcMix));
    a.write(out, rows * n * 4);
    return a;
}

LaunchAccess launch_access_engram_inject(const float* hc_stream, const float* kv, const float* q_norm, const float* k_norm,
                                         std::uint32_t rows, std::uint32_t hidden) {
    LaunchAccess a;
    const std::uint64_t n = std::uint64_t{hidden} * hc_copies;
    a.read(hc_stream, rows * n * 4);
    a.write(hc_stream, rows * n * 4);
    a.read(kv, rows * (n + hidden) * 4);
    a.read(q_norm, n * 4);
    a.read(k_norm, n * 4);
    return a;
}

LaunchAccess launch_access_argmax(const float* logits, std::uint32_t rows, std::uint32_t count, std::uint32_t stride,
                                  const ArgmaxCandidate* partial, const ArgmaxCandidate* out) {
    LaunchAccess a;
    const std::uint64_t blocks = argmax_partial_blocks(count);
    a.read(logits, (std::uint64_t{rows - 1U} * stride + count) * 4);
    a.read(partial, rows * blocks * sizeof(ArgmaxCandidate));
    a.write(partial, rows * blocks * sizeof(ArgmaxCandidate));
    a.write(out, std::uint64_t{rows} * sizeof(ArgmaxCandidate));
    return a;
}

} // namespace strix::ds41
