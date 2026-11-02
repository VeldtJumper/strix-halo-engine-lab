#pragma once
// DSpark Markov chain over a candidate set: the device kernels shared by the MiniMax drafter
// (src/minimax/dspark/markov_cand.*: one logits segment, BF16 Markov matrices, 7 rows) and the DeepSeek V4.1 drafter
// (src/hip_ds41_dspark.hip: core + extra vocabulary segments, F16 matrices, up to 5 rows). Each user instantiates the
// templates with its own logits view and decode, so the MiniMax kernels keep their arithmetic (qualified bitwise against
// CPU replicas in hip_markov_cand_tests).
//
// A logits view L maps a position j in [0, L.n) of this rank's vocabulary to a logit L.at(row, j) and a token id L.id(j).
// Selection orders by value (descending, -0 == +0) and then by position (ascending); positions of ONE segment ascend with
// the ids, so for MiniMax this is the documented "ties by lower id".
//
// Payload (per rank, same size on every rank): ids [rows][C] u32 | counts [rows] u32 | scores [rows][C] f32 |
// hval [rows][hmax] f32 (the values of the H members this rank owns, zero padded to hmax).
#include <cstddef>
#include <cstdint>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace strix::dspark::markov_cand {
namespace {

constexpr std::uint32_t kRank = 256;  // Markov rank of both drafters (32 lanes x 8)
constexpr std::uint32_t kMaxRows = 8;
constexpr std::uint32_t kFallback = 0xffffffffU;
struct Cand {
    float value;
    std::uint32_t id;
};

// The Markov matrices' element formats.
struct Bf16 {
    __host__ __device__ static float f(std::uint16_t b) { return __builtin_bit_cast(float, static_cast<std::uint32_t>(b) << 16U); }
};
struct F16 {
    __device__ static float f(std::uint16_t b) { return __half2float(__ushort_as_half(b)); }
};

// One segment: logits [rows][n], id = id0 + j.
struct OneSegment {
    const float* p;
    std::uint32_t n, id0;
    __device__ float at(std::uint32_t row, std::uint32_t j) const { return p[static_cast<std::size_t>(row) * n + j]; }
    __device__ std::uint32_t id(std::uint32_t j) const { return id0 + j; }
};
// Two segments: positions [0, na) read a[row * a_stride + a_off + j] (id a_id0 + j), positions [na, n) read
// x[row * x_stride + x_off + j - na] (id x_ids[x_ids_off + j - na]).
struct TwoSegments {
    const float* a;
    std::uint32_t a_stride, a_off, na, a_id0;
    const float* x;
    std::uint32_t x_stride, x_off;
    const std::uint32_t* x_ids;
    std::uint32_t x_ids_off, n;
    __device__ float at(std::uint32_t row, std::uint32_t j) const {
        return j < na ? a[static_cast<std::size_t>(row) * a_stride + a_off + j] : x[static_cast<std::size_t>(row) * x_stride + x_off + (j - na)];
    }
    __device__ std::uint32_t id(std::uint32_t j) const { return j < na ? a_id0 + j : x_ids[x_ids_off + (j - na)]; }
};

__host__ __device__ inline bool better(float v, std::uint32_t i, float b, std::uint32_t bi) { return v > b || (v == b && i < bi); }
__device__ inline float wave_sum(float v) {
    for (int d = 16; d != 0; d >>= 1) v = v + __shfl_xor(v, d, 32);
    return v;
}
// Per-candidate arithmetic of both full Markov passes: 8 products per lane in element order, xor butterfly 16..1.
template <class Dec>
__device__ __forceinline__ float lane_dot(const std::uint16_t* w1row, const std::uint16_t* w2row, unsigned lane) {
    const uint4 mw = *reinterpret_cast<const uint4*>(w1row + 8U * lane);
    const uint4 ww = *reinterpret_cast<const uint4*>(w2row + 8U * lane);
    const std::uint32_t ma[4] = {mw.x, mw.y, mw.z, mw.w}, wa[4] = {ww.x, ww.y, ww.z, ww.w};
    float acc = 0.0F;
    for (std::uint32_t e = 0; e < 8U; ++e) {
        const float m = Dec::f(static_cast<std::uint16_t>(ma[e >> 1U] >> (16U * (e & 1U))));
        const float w = Dec::f(static_cast<std::uint16_t>(wa[e >> 1U] >> (16U * (e & 1U))));
        acc = acc + m * w;
    }
    return wave_sum(acc);
}

struct Payload {
    std::uint32_t* ids;
    std::uint32_t* counts;
    float* scores;
    float* hval;
};
__host__ __device__ inline Payload view(unsigned char* p, unsigned C, unsigned rows) {
    Payload v;
    v.ids = reinterpret_cast<std::uint32_t*>(p);
    v.counts = v.ids + std::size_t{rows} * C;
    v.scores = reinterpret_cast<float*>(v.counts + rows);
    v.hval = v.scores + std::size_t{rows} * C;
    return v;
}
__host__ __device__ inline std::size_t payload_bytes(unsigned C, unsigned rows, unsigned hmax) {
    return (std::size_t{rows} * C * 2 + rows + std::size_t{rows} * hmax) * 4;
}

__device__ __forceinline__ std::uint32_t okey(float f) {
    std::uint32_t u = __builtin_bit_cast(std::uint32_t, f);
    if (u == 0x80000000U) u = 0U;  // -0 == +0
    return (u & 0x80000000U) ? ~u : (u | 0x80000000U);
}

// ---- topc v1: exactly C entries of row blockIdx.x in canonical order, one 1024-thread CTA per row ----
template <class L>
__global__ __launch_bounds__(1024) void topc_kernel(L lg, std::uint32_t C, unsigned rows, unsigned char* payload) {
    __shared__ std::uint32_t hist[256];
    __shared__ std::uint32_t s_prefix, s_need, s_n, s_bad, s_wave[32], s_taken;
    const Payload pv = view(payload, C, rows);
    const unsigned row = blockIdx.x;
    const std::uint32_t n = lg.n;
    std::uint32_t* out = pv.ids + static_cast<std::size_t>(row) * C;
    if (threadIdx.x == 0) s_bad = 0;
    __syncthreads();
    for (std::uint32_t v = threadIdx.x; v < n; v += blockDim.x)
        if (!__builtin_isfinite(lg.at(row, v))) atomicOr(&s_bad, 1U);
    __syncthreads();
    if (s_bad || C == 0 || C > n) {
        if (threadIdx.x == 0) pv.counts[row] = kFallback;
        return;
    }
    std::uint32_t prefix = 0, mask = 0, need = C;
    for (int pass = 3; pass >= 0; --pass) {
        for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        const unsigned shift = 8U * static_cast<unsigned>(pass);
        for (std::uint32_t v = threadIdx.x; v < n; v += blockDim.x) {
            const std::uint32_t k = okey(lg.at(row, v));
            if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 255U], 1U);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            std::uint32_t acc = 0;
            int d = 255;
            for (; d > 0; --d) {
                if (acc + hist[d] >= need) break;
                acc += hist[d];
            }
            s_prefix = prefix | (static_cast<std::uint32_t>(d) << shift);
            s_need = need - acc;
        }
        __syncthreads();
        prefix = s_prefix;
        need = s_need;
        mask |= 255U << shift;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        s_n = 0;
        s_taken = 0;
    }
    __syncthreads();
    for (std::uint32_t v = threadIdx.x; v < n; v += blockDim.x)
        if (okey(lg.at(row, v)) > prefix) out[atomicAdd(&s_n, 1U)] = lg.id(v);
    __syncthreads();
    const std::uint32_t above = s_n;
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U;
    for (std::uint32_t t0 = 0; t0 < n; t0 += blockDim.x) {  // ties: the lowest positions, ascending tiles
        if (s_taken >= need) break;
        const std::uint32_t v = t0 + threadIdx.x;
        const bool eq = v < n && okey(lg.at(row, v)) == prefix;
        const std::uint32_t m = static_cast<std::uint32_t>(__ballot(eq));
        const std::uint32_t before = static_cast<std::uint32_t>(__builtin_popcount(m & ((1U << lane) - 1U)));
        if (lane == 0) s_wave[wave] = static_cast<std::uint32_t>(__builtin_popcount(m));
        __syncthreads();
        std::uint32_t off = 0;
        for (unsigned w = 0; w < wave; ++w) off += s_wave[w];
        const std::uint32_t rank = s_taken + off + before;
        if (eq && rank < need) out[above + rank] = lg.id(v);
        __syncthreads();
        if (threadIdx.x == 0) {
            std::uint32_t tot = 0;
            for (unsigned w = 0; w < blockDim.x / 32U; ++w) tot += s_wave[w];
            s_taken += tot;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) pv.counts[row] = C;
}

// ---- topc v2: the same canonical set over many CTAs per row ----
// Key k = okey(x) (order-preserving, -0 -> +0). Row threshold T (full key) and tie cut: the selected set is
// {j : k(j) > T} u {j : k(j) == T, j <= cut}, exactly C entries, i.e. the C largest keys, ties by lower position (= topc_kernel).
//   hist    (nb x rows CTAs): nonfinite flag + 2048-bin histogram of k >> 21 (global atomics, nonzero bins only)
//   compact (nb x rows CTAs): boundary bin d1 (C-th largest key's top 11 bits); per chunk the count of keys with bin > d1
//            (all selected) and the (low 21 bits, position) of the keys with bin == d1 appended to the row's candidate buffer
//   refine  (rows CTAs):      exact radix select over the candidates only: low 21 bits (11 + 10), then, if not all keys == T
//            are taken, the need-th lowest position among them (22-bit positions, 11 + 11); per-chunk output offsets;
//            counts; resets the hist / flag / candidate counter for the next draft
//   emit    (nb x rows CTAs): ids + scores of the selected keys in ascending position order (deterministic) + the owned-H values
constexpr std::uint32_t kBins = 2048, kCh = 4096, kV2Threads = 256, kPer = kCh / kV2Threads, kIdBits = 22, kLowMask = 0x1fffffU;
constexpr std::uint32_t kMaxChunks = (1U << kIdBits) / kCh;

struct Ws {  // hist [rows][2048] | bad [rows] | ccount [rows] | thr [rows] | cut [rows] | gt [rows][nb] | off [rows][nb] | (8-aligned) cbuf [rows][n]
    std::uint32_t *hist, *bad, *ccount, *thr, *cut, *gt, *off;
    uint2* cbuf;
};
__host__ __device__ inline std::size_t ws_words(std::uint32_t nb, unsigned rows) {
    const std::size_t w = std::size_t{rows} * (kBins + 4U + 2U * nb);
    return (w + 1U) & ~std::size_t{1};
}
__host__ __device__ inline std::size_t ws_bytes(std::uint32_t nb, std::uint32_t n, unsigned rows) {
    return ws_words(nb, rows) * 4U + std::size_t{rows} * n * sizeof(uint2);
}
__host__ __device__ inline Ws ws_view(unsigned char* p, std::uint32_t nb, unsigned rows) {
    Ws w;
    w.hist = reinterpret_cast<std::uint32_t*>(p);
    w.bad = w.hist + std::size_t{rows} * kBins;
    w.ccount = w.bad + rows;
    w.thr = w.ccount + rows;
    w.cut = w.thr + rows;
    w.gt = w.cut + rows;
    w.off = w.gt + std::size_t{rows} * nb;
    w.cbuf = reinterpret_cast<uint2*>(p + ws_words(nb, rows) * 4U);
    return w;
}
// Exclusive prefix (thread order) of x over the block; total -> tot. Block size a multiple of 32; s_wave >= 32 words.
__device__ inline std::uint32_t block_scan(std::uint32_t x, std::uint32_t* s_wave, std::uint32_t& tot) {
    const std::uint32_t lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U, nw = blockDim.x >> 5U;
    std::uint32_t inc = x;
    for (std::uint32_t d = 1; d < 32U; d <<= 1U) {
        const std::uint32_t y = __shfl_up(inc, static_cast<unsigned>(d), 32);
        if (lane >= d) inc += y;
    }
    if (lane == 31U) s_wave[wave] = inc;
    __syncthreads();
    std::uint32_t off = 0, t = 0;
    for (std::uint32_t w = 0; w < nw; ++w) {
        if (w < wave) off += s_wave[w];
        t += s_wave[w];
    }
    __syncthreads();
    tot = t;
    return off + inc - x;
}
// Over hist[0, nbins) (all threads read it): the digit d with sum(hist[d+1..]) < need <= sum(hist[d..]) (need >= 1, total
// >= need) -> s_out[0] = d, s_out[1] = sum(hist[d+1..]). Ends with a barrier.
__device__ inline void block_find(const std::uint32_t* hist, std::uint32_t nbins, std::uint32_t need, std::uint32_t* s_wave,
                                  std::uint32_t* s_out) {
    const std::uint32_t t = threadIdx.x, lane = t & 31U, wave = t >> 5U, nw = blockDim.x >> 5U;
    const std::uint32_t per = (nbins + blockDim.x - 1U) / blockDim.x;
    std::uint32_t s = 0;
    for (std::uint32_t j = 0; j < per; ++j)
        if (t * per + j < nbins) s += hist[t * per + j];
    std::uint32_t suf = s;  // inclusive suffix sum within the wave (lanes >= this one)
    for (std::uint32_t d = 1; d < 32U; d <<= 1U) {
        const std::uint32_t y = __shfl_down(suf, static_cast<unsigned>(d), 32);
        if (lane + d < 32U) suf += y;
    }
    if (lane == 0) s_wave[wave] = suf;
    __syncthreads();
    std::uint32_t above = suf - s;  // keys in the bins after this thread's
    for (std::uint32_t w = wave + 1U; w < nw; ++w) above += s_wave[w];
    if (above < need && above + s >= need) {
        std::uint32_t acc = above, i = t * per + per - 1U;
        for (;; --i) {
            if (i < nbins) {
                if (acc + hist[i] >= need) break;
                acc += hist[i];
            }
        }
        s_out[0] = i;
        s_out[1] = acc;
    }
    __syncthreads();
}
template <class L>
__global__ __launch_bounds__(256) void v2_hist_kernel(L lg, unsigned rows, unsigned char* ws, std::uint32_t nb) {
    __shared__ std::uint32_t h[kBins];
    __shared__ std::uint32_t s_bad;
    const Ws w = ws_view(ws, nb, rows);
    const std::uint32_t n = lg.n, row = blockIdx.y, c0 = blockIdx.x * kCh, c1 = c0 + kCh < n ? c0 + kCh : n;
    for (std::uint32_t i = threadIdx.x; i < kBins; i += blockDim.x) h[i] = 0;
    if (threadIdx.x == 0) s_bad = 0;
    __syncthreads();
    bool bad = false;
    for (std::uint32_t v = c0 + threadIdx.x; v < c1; v += blockDim.x) {
        const float x = lg.at(row, v);
        bad = bad || !__builtin_isfinite(x);
        atomicAdd(&h[okey(x) >> 21U], 1U);
    }
    if (bad) atomicOr(&s_bad, 1U);
    __syncthreads();
    for (std::uint32_t i = threadIdx.x; i < kBins; i += blockDim.x)
        if (h[i] != 0) atomicAdd(&w.hist[std::size_t{row} * kBins + i], h[i]);
    if (threadIdx.x == 0 && s_bad != 0) atomicOr(&w.bad[row], 1U);
}
template <class L>
__global__ __launch_bounds__(256) void v2_compact_kernel(L lg, std::uint32_t C, unsigned rows, unsigned char* ws, std::uint32_t nb) {
    __shared__ std::uint32_t s_wave[32], s_out[2], s_base;
    const Ws w = ws_view(ws, nb, rows);
    const std::uint32_t n = lg.n, row = blockIdx.y, chunk = blockIdx.x;
    if (w.bad[row] != 0 || C == 0 || C > n) return;  // refine reports the fallback
    block_find(w.hist + std::size_t{row} * kBins, kBins, C, s_wave, s_out);
    const std::uint32_t d1 = s_out[0];
    const std::uint32_t v0 = chunk * kCh + threadIdx.x * kPer;
    std::uint32_t key[kPer];
    std::uint32_t eq = 0, gt = 0;
    for (std::uint32_t j = 0; j < kPer; ++j) {
        key[j] = v0 + j < n ? okey(lg.at(row, v0 + j)) : 0U;
        const std::uint32_t bin = key[j] >> 21U;
        gt += (v0 + j < n && bin > d1) ? 1U : 0U;
        eq += (v0 + j < n && bin == d1) ? 1U : 0U;
    }
    std::uint32_t eq_tot = 0, gt_tot = 0;
    std::uint32_t pos = block_scan(eq, s_wave, eq_tot);
    (void)block_scan(gt, s_wave, gt_tot);
    if (threadIdx.x == 0) {
        s_base = eq_tot != 0 ? atomicAdd(&w.ccount[row], eq_tot) : 0U;
        w.gt[std::size_t{row} * nb + chunk] = gt_tot;
    }
    __syncthreads();
    pos += s_base;
    uint2* cb = w.cbuf + static_cast<std::size_t>(row) * n;
    for (std::uint32_t j = 0; j < kPer; ++j)
        if (v0 + j < n && (key[j] >> 21U) == d1) cb[pos++] = make_uint2(key[j] & kLowMask, v0 + j);
}
// Radix select among the candidates (ties == false: key = low 21 bits; ties == true: only low == tlow, key = the position
// reversed so larger = lower position): the key t with #(key > t) < need <= #(key >= t); need -> need - #(key > t); eq = #(key == t).
__device__ inline std::uint32_t cta_select(const uint2* cb, std::uint32_t m, bool ties, std::uint32_t tlow, std::uint32_t nbits,
                                           std::uint32_t& need, std::uint32_t& eq, std::uint32_t* h, std::uint32_t* s_wave,
                                           std::uint32_t* s_out) {
    std::uint32_t prefix = 0, mask = 0, shift = nbits;
    while (shift != 0) {
        const std::uint32_t wd = shift < 11U ? shift : 11U, digits = 1U << wd;
        shift -= wd;
        for (std::uint32_t i = threadIdx.x; i < digits; i += blockDim.x) h[i] = 0;
        __syncthreads();
        for (std::uint32_t i = threadIdx.x; i < m; i += blockDim.x) {
            const uint2 c = cb[i];
            if (ties && c.x != tlow) continue;
            const std::uint32_t k = ties ? ((1U << kIdBits) - 1U) - c.y : c.x;
            if ((k & mask) == prefix) atomicAdd(&h[(k >> shift) & (digits - 1U)], 1U);
        }
        __syncthreads();
        block_find(h, digits, need, s_wave, s_out);
        const std::uint32_t d = s_out[0];
        eq = h[d];
        need -= s_out[1];
        prefix |= d << shift;
        mask |= (digits - 1U) << shift;
        __syncthreads();
    }
    return prefix;
}
__global__ __launch_bounds__(1024) void v2_refine_kernel(std::uint32_t n, std::uint32_t C, unsigned rows, unsigned char* ws, std::uint32_t nb,
                                                         unsigned char* payload) {
    __shared__ std::uint32_t h[kBins], cnt[kMaxChunks], s_wave[32], s_out[2];
    const Ws w = ws_view(ws, nb, rows);
    const Payload pv = view(payload, C, rows);
    const std::uint32_t row = blockIdx.x;
    const bool fallback = w.bad[row] != 0 || C == 0 || C > n;
    if (!fallback) {
        block_find(w.hist + std::size_t{row} * kBins, kBins, C, s_wave, s_out);
        const std::uint32_t d1 = s_out[0];
        std::uint32_t need = C - s_out[1], eq = 0;
        const std::uint32_t m = w.ccount[row];
        const uint2* cb = w.cbuf + static_cast<std::size_t>(row) * n;
        const std::uint32_t tlow = cta_select(cb, m, false, 0, 21U, need, eq, h, s_wave, s_out);
        std::uint32_t cut = 0xffffffffU;  // all keys == T taken
        if (need < eq) {
            std::uint32_t e2 = 0;
            cut = ((1U << kIdBits) - 1U) - cta_select(cb, m, true, tlow, kIdBits, need, e2, h, s_wave, s_out);
        }
        for (std::uint32_t i = threadIdx.x; i < nb; i += blockDim.x) cnt[i] = 0;
        __syncthreads();
        for (std::uint32_t i = threadIdx.x; i < m; i += blockDim.x) {
            const uint2 c = cb[i];
            if (c.x > tlow || (c.x == tlow && c.y <= cut)) atomicAdd(&cnt[c.y / kCh], 1U);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            std::uint32_t acc = 0;
            for (std::uint32_t i = 0; i < nb; ++i) {
                w.off[std::size_t{row} * nb + i] = acc;
                acc += w.gt[std::size_t{row} * nb + i] + cnt[i];
            }
            w.thr[row] = (d1 << 21U) | tlow;
            w.cut[row] = cut;
            pv.counts[row] = acc == C ? C : kFallback;  // acc == C by construction
        }
    } else if (threadIdx.x == 0) {
        pv.counts[row] = kFallback;
    }
    __syncthreads();  // every read of hist / bad / ccount is done: reset for the next draft
    for (std::uint32_t i = threadIdx.x; i < kBins; i += blockDim.x) w.hist[std::size_t{row} * kBins + i] = 0;
    if (threadIdx.x == 0) {
        w.bad[row] = 0;
        w.ccount[row] = 0;
    }
}
// hpos: positions (in this rank's view) of the H members it owns.
template <class L>
__global__ __launch_bounds__(256) void v2_emit_kernel(L lg, std::uint32_t C, unsigned rows, const std::uint32_t* hpos, std::uint32_t nh_own,
                                                      std::uint32_t hmax, unsigned char* ws, std::uint32_t nb, unsigned char* payload) {
    __shared__ std::uint32_t s_wave[32];
    const Ws w = ws_view(ws, nb, rows);
    const Payload pv = view(payload, C, rows);
    const std::uint32_t n = lg.n, row = blockIdx.y, chunk = blockIdx.x;
    for (std::uint32_t i = chunk * blockDim.x + threadIdx.x; i < hmax; i += nb * blockDim.x)  // = pack_kernel's hval
        pv.hval[static_cast<std::size_t>(row) * hmax + i] = i < nh_own ? lg.at(row, hpos[i]) : 0.0F;
    if (pv.counts[row] != C) return;
    const std::uint32_t T = w.thr[row], cut = w.cut[row], v0 = chunk * kCh + threadIdx.x * kPer;
    float x[kPer];
    std::uint32_t sel = 0, cnt = 0;
    for (std::uint32_t j = 0; j < kPer; ++j) {
        const std::uint32_t v = v0 + j;
        x[j] = v < n ? lg.at(row, v) : 0.0F;
        const std::uint32_t k = okey(x[j]);
        if (v < n && (k > T || (k == T && v <= cut))) {
            sel |= 1U << j;
            ++cnt;
        }
    }
    std::uint32_t tot = 0;
    std::uint32_t pos = w.off[std::size_t{row} * nb + chunk] + block_scan(cnt, s_wave, tot);
    for (std::uint32_t j = 0; j < kPer; ++j)
        if ((sel >> j) & 1U) {
            pv.ids[static_cast<std::size_t>(row) * C + pos] = lg.id(v0 + j);
            pv.scores[static_cast<std::size_t>(row) * C + pos] = x[j];
            ++pos;
        }
}
// v1's scores of the selected ids and the values of the owned H members. id_off: one segment only (position = id - id_off).
[[maybe_unused]] __global__ void pack_kernel(const float* logits, std::uint32_t n, std::uint32_t C, unsigned rows, std::uint32_t id_off,
                            const std::uint32_t* hpos, std::uint32_t nh_own, std::uint32_t hmax, unsigned char* payload) {
    const Payload pv = view(payload, C, rows);
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x, k = blockIdx.y;
    const float* row = logits + static_cast<std::size_t>(k) * n;
    if (i < C && pv.counts[k] == C) pv.scores[static_cast<std::size_t>(k) * C + i] = row[pv.ids[static_cast<std::size_t>(k) * C + i] - id_off];
    if (i < hmax) pv.hval[static_cast<std::size_t>(k) * hmax + i] = i < nh_own ? row[hpos[i]] : 0.0F;
}
// One chain step of row k over [own top-C | peer top-C | own H | peer H]; one wave per candidate, 8 per block. Both
// payloads' counts of every chained row must be C: callers check them on the host (a nonfinite logit -> the full chain).
template <class Dec>
__global__ void cand_kernel(const std::uint16_t* w1, const std::uint16_t* w2, const std::uint32_t* prev, unsigned k, unsigned rows,
                            const unsigned char* own, const unsigned char* peer, unsigned C, const std::uint32_t* hown, unsigned nh_own,
                            const std::uint32_t* hpeer, unsigned nh_peer, unsigned hmax, Cand* cand) {
    __shared__ float best_v[8];
    __shared__ std::uint32_t best_i[8];
    const Payload a = view(const_cast<unsigned char*>(own), C, rows), b = view(const_cast<unsigned char*>(peer), C, rows);
    const std::uint32_t lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U, q = blockIdx.x * 8U + wave;
    Cand c{-__builtin_huge_valf(), 0xffffffffU};
    if (q < 2 * C + nh_own + nh_peer) {
        std::uint32_t v;
        float bv;
        if (q < C) {
            v = a.ids[std::size_t{k} * C + q];
            bv = a.scores[std::size_t{k} * C + q];
        } else if (q < 2 * C) {
            v = b.ids[std::size_t{k} * C + q - C];
            bv = b.scores[std::size_t{k} * C + q - C];
        } else if (q < 2 * C + nh_own) {
            v = hown[q - 2 * C];
            bv = a.hval[std::size_t{k} * hmax + q - 2 * C];
        } else {
            v = hpeer[q - 2 * C - nh_own];
            bv = b.hval[std::size_t{k} * hmax + q - 2 * C - nh_own];
        }
        const float acc = lane_dot<Dec>(w1 + static_cast<std::size_t>(*prev) * kRank, w2 + static_cast<std::size_t>(v) * kRank, lane);
        c = {bv + acc, v};
    }
    if (lane == 0) {
        best_v[wave] = c.value;
        best_i[wave] = c.id;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        Cand bst{best_v[0], best_i[0]};
        for (int j = 1; j < 8; ++j)
            if (better(best_v[j], best_i[j], bst.value, bst.id)) bst = {best_v[j], best_i[j]};
        cand[blockIdx.x] = bst;
    }
}
// The best of the per-block candidates, writing the next prev on device (and the pick when out != nullptr).
__global__ void pick_kernel(const Cand* cand, std::uint32_t n, Cand* out, std::uint32_t* prev_out) {
    __shared__ float rv[256];
    __shared__ std::uint32_t ri[256];
    Cand b{-__builtin_huge_valf(), 0xffffffffU};
    for (std::uint32_t i = threadIdx.x; i < n; i += 256U)
        if (better(cand[i].value, cand[i].id, b.value, b.id)) b = cand[i];
    rv[threadIdx.x] = b.value;
    ri[threadIdx.x] = b.id;
    __syncthreads();
    for (std::uint32_t k = 128; k != 0; k >>= 1U) {
        const std::uint32_t t = threadIdx.x;
        if (t < k && better(rv[t + k], ri[t + k], rv[t], ri[t])) {
            rv[t] = rv[t + k];
            ri[t] = ri[t + k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (out != nullptr) *out = Cand{rv[0], ri[0]};
        *prev_out = ri[0];
    }
}
// Sum of squares of each W2 row (one wave per row, lane chains of 8 then the butterfly): ranking only.
template <class Dec>
__global__ void norm_kernel(const std::uint16_t* w2, std::uint32_t rows, float* out) {
    const std::uint32_t lane = threadIdx.x & 31U, v = blockIdx.x * 8U + (threadIdx.x >> 5U);
    if (v >= rows) return;
    float acc = 0.0F;
    for (unsigned e = 0; e < 8; ++e) {
        const float x = Dec::f(w2[static_cast<std::size_t>(v) * kRank + 8U * lane + e]);
        acc = acc + x * x;
    }
    acc = wave_sum(acc);
    if (lane == 0) out[v] = acc;
}

}  // namespace
}  // namespace strix::dspark::markov_cand
