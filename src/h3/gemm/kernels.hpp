#pragma once
// MiniMax H3 W8A8 tensorwise + convrot projection on gfx1151 (wave32, v_wmma_i32_16x16x16_iu8).
// Arithmetic = src/h3/gemm/ref.hpp, bit for bit:
//   quantize_kernel: one block per row. Lanes in 16-lane teams, a team per 256-group, 16 consecutive features per lane
//     (feature 16 l + j, l = 4 d3 + d2, j = 4 d1 + d0). Radix-4 stages over d0 / d1 inside the lane, over d2 / d3 across
//     the team (lane xor 1, 2, 3 / 4, 8, 12), in 64-bit fixed point relative to the group's max exponent; rounded once
//     to bf16 (ref fixed_to_bf16); row absmax over the block; s = amax / 127 (IEEE), divisor bf16(s); codes
//     clamp(rint(bf16(x / sb))). Rows independent: batch invariant.
//   gemm_kernel: block tile 256 rows x BN outputs (BN 128: 8 waves, BN 256: 16 waves; each wave 64 x 64 = 4 x 4 WMMA
//     tiles), 64 K bytes per stage, two LDS stages (64-byte rows, 16-byte chunks XOR-swizzled by (row / 2) % 4), int32
//     accumulators over the full K (exact, order-free). Epilogue per element: bf16(fp32(acc) * (row_scale * ch_scale)).
//     Epilogue SwiGLU (fc1 with swiglu_permute()d weights): each wave's 16-column tiles alternate gate / up of the same
//     columns, h = bf16(bf16(silu(gate)) * up) with silu from the exact 65536-entry table (ref silu_table).
// Output: column segments (up to 4 base pointers, seg_cols columns each, row pitch ld elements): column n goes to
//   seg[n / seg_cols] + row * ld + n % seg_cols (plain row-major = 1 segment; q | k | v into three buffers = 3).

#include <hip/hip_runtime.h>

#include <cstdint>
#include <atomic>
#include <cstdlib>
#include <stdexcept>

#include "h3/gemm/ref.hpp"

namespace strix::h3::gemm::dev {

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ u32x4 g16(const void* p) {
    return *reinterpret_cast<const __attribute__((address_space(1))) u32x4*>(reinterpret_cast<std::uintptr_t>(p));
}

// ---------------------------------------------------------------------------------------------------------------- quantizer
constexpr unsigned kQThreads = 256U, kQTeams = kQThreads / 16U, kQMaxIters = 4U;  // K <= 16 * 256 * 4 = 16384

// Lane exchanges inside a 16-lane row by DPP (no LDS traffic): xor 1 / 2 / 3 = quad_perm, xor 4 / 8 / 12 = row_xmask.
template <int Mask> __device__ __forceinline__ int xor_lane(int v) {
    constexpr int ctrl = Mask == 1 ? 0xB1 : Mask == 2 ? 0x4E : Mask == 3 ? 0x1B : 0x160 + Mask;
    static_assert(Mask == 1 || Mask == 2 || Mask == 3 || Mask == 4 || Mask == 8 || Mask == 12);
    return __builtin_amdgcn_update_dpp(0, v, ctrl, 0xF, 0xF, false);
}
template <int Mask> __device__ __forceinline__ std::int64_t xor_lane(std::int64_t v) {
    const auto u = static_cast<std::uint64_t>(v);
    const int lo = xor_lane<Mask>(static_cast<int>(u & 0xffffffffU)), hi = xor_lane<Mask>(static_cast<int>(u >> 32U));
    return static_cast<std::int64_t>((static_cast<std::uint64_t>(static_cast<std::uint32_t>(hi)) << 32U) | static_cast<std::uint32_t>(lo));
}
// One radix-4 stage across lanes: this lane holds a_{d}, d = (lane / Step) % 4; y_d = S - 2 a_{3 - d}, 3 - d = d ^ 3.
template <int Step, class T> __device__ __forceinline__ T cross4(T a) {
    const T p1 = xor_lane<Step>(a), p2 = xor_lane<2 * Step>(a), p3 = xor_lane<3 * Step>(a);
    const T s = (a + p1) + (p2 + p3);  // exact integers: pairing irrelevant
    return s - 2 * p3;
}
template <class T> __device__ __forceinline__ void in4(T* v, unsigned stride) {
#pragma unroll
    for (unsigned base = 0; base < 16U; ++base) {
        if ((base / stride) % 4U) continue;
        const T a0 = v[base], a1 = v[base + stride], a2 = v[base + 2U * stride], a3 = v[base + 3U * stride];
        const T s = (a0 + a1) + (a2 + a3);
        v[base] = s - 2 * a3;
        v[base + stride] = s - 2 * a2;
        v[base + 2U * stride] = s - 2 * a1;
        v[base + 3U * stride] = s - 2 * a0;
    }
}

// Variants (identical bits; the variant only changes the path):
//   baseline <false, 4>: 256 threads (16 teams), up to 4 groups per team, int64 fixed point (shift kFixShift = 47).
//   fast <true, 1>:      16 * groups threads (one team per group, <= 1024), and per group: when the nonzero exponents of
//                        the group span <= kNarrowSpan binades (team-uniform), the transform runs in int32 with shift
//                        kNarrowShift = 15 (|sig| <= 255, 255 * 2^15 * 256 < 2^31: every partial sum fits), else the
//                        int64 path. Both feed the same value (y * 2^(emax - shift - 138)) to fixed_to_bf16, so the
//                        rounded result, the flush decision (none possible within 15 binades) and the count agree.
constexpr int kNarrowShift = 15, kNarrowSpan = 15;
template <bool Fast> constexpr unsigned quant_threads_max() { return Fast ? 1024U : kQThreads; }

// x: bf16 rows (pitch x_ld elements); q: int8 codes (pitch k); scale: fp32 row scales; flushed (optional): count of
// groups with nonzero values more than kFixShift binades below the group max (flushed: exact within 2^-47 of the max).
// A template, so the definition may appear in every translation unit including this header.
template <bool Fast = false, unsigned Iters = kQMaxIters>
__global__ __launch_bounds__(quant_threads_max<Fast>()) void quantize_kernel(const std::uint16_t* __restrict__ x, std::uint64_t x_ld,
    std::uint32_t k, std::int8_t* __restrict__ q, float* __restrict__ scale, unsigned* __restrict__ flushed) {
#pragma clang fp contract(off)
    __shared__ float red[32];
    const unsigned t = threadIdx.x, lane = t % 32U, l = t % 16U, team = t / 16U, teams = blockDim.x / 16U;
    const std::uint32_t row = blockIdx.x, groups = k / 256U;
    const std::uint16_t* xr = x + row * x_ld;
    std::uint16_t out[Iters][16];
    float amax = 0.0F;
    unsigned nflush = 0;
#pragma unroll
    for (unsigned it = 0; it < Iters; ++it) {
        const std::uint32_t g = team + teams * it;
        // a team either has the group or not (team-uniform), so the lane exchanges inside a team stay convergent
        if (g >= groups) {
#pragma unroll
            for (unsigned j = 0; j < 16U; ++j) out[it][j] = 0;
            continue;
        }
        const u32x4 w0 = g16(xr + 256U * g + 16U * l), w1 = g16(xr + 256U * g + 16U * l + 8U);
        std::uint16_t in[16];
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            in[2U * j] = static_cast<std::uint16_t>(w0[j] & 0xffffU); in[2U * j + 1U] = static_cast<std::uint16_t>(w0[j] >> 16U);
            in[8U + 2U * j] = static_cast<std::uint16_t>(w1[j] & 0xffffU); in[8U + 2U * j + 1U] = static_cast<std::uint16_t>(w1[j] >> 16U);
        }
        int emax = 1;  // ref: max ee over nonzero values (1 when the group is all zero)
        int emin = 1000;
#pragma unroll
        for (unsigned j = 0; j < 16U; ++j)
            if (in[j] & 0x7fffU) { const int e = bf16_parts(in[j]).ee; emax = max(emax, e); emin = min(emin, e); }
        emax = max(emax, xor_lane<1>(emax)); emax = max(emax, xor_lane<2>(emax));
        emax = max(emax, xor_lane<4>(emax)); emax = max(emax, xor_lane<8>(emax));
        bool narrow = false;
        if constexpr (Fast) {
            emin = min(emin, xor_lane<1>(emin)); emin = min(emin, xor_lane<2>(emin));
            emin = min(emin, xor_lane<4>(emin)); emin = min(emin, xor_lane<8>(emin));
            narrow = emax - emin <= kNarrowSpan;  // team-uniform (all-zero group: emin 1000, narrow)
        }
        bool fl = false;
        if (narrow) {
            int v[16];
#pragma unroll
            for (unsigned j = 0; j < 16U; ++j) {
                const Bf16Parts p = bf16_parts(in[j]);
                v[j] = static_cast<int>(p.sig) * (1 << (p.sig ? p.ee - emax + kNarrowShift : 0));  // shift in [0, 15]
            }
            in4(v, 1U);
            in4(v, 4U);
#pragma unroll
            for (unsigned j = 0; j < 16U; ++j) v[j] = cross4<1>(v[j]);
#pragma unroll
            for (unsigned j = 0; j < 16U; ++j) v[j] = cross4<4>(v[j]);
#pragma unroll
            for (unsigned j = 0; j < 16U; ++j) out[it][j] = fixed_to_bf16(static_cast<std::int64_t>(v[j]), emax - kNarrowShift - 138);
        } else {
            std::int64_t v[16];
#pragma unroll
            for (unsigned j = 0; j < 16U; ++j) {
                const Bf16Parts p = bf16_parts(in[j]);
                const int sh = p.ee - emax + kFixShift;
                fl |= sh < 0 && p.sig != 0;
                v[j] = sh < 0 ? 0 : p.sig * (std::int64_t{1} << sh);
            }
            in4(v, 1U);
            in4(v, 4U);
#pragma unroll
            for (unsigned j = 0; j < 16U; ++j) v[j] = cross4<1>(v[j]);
#pragma unroll
            for (unsigned j = 0; j < 16U; ++j) v[j] = cross4<4>(v[j]);
#pragma unroll
            for (unsigned j = 0; j < 16U; ++j) out[it][j] = fixed_to_bf16(v[j], emax - kFixShift - 138);
        }
#pragma unroll
        for (unsigned j = 0; j < 16U; ++j) amax = fmaxf(amax, bf16_to_f32(static_cast<std::uint16_t>(out[it][j] & 0x7fffU)));
        int any = static_cast<int>(fl);
        any |= xor_lane<1>(any); any |= xor_lane<2>(any); any |= xor_lane<4>(any); any |= xor_lane<8>(any);
        if (any && l == 0U) ++nflush;
    }
    // row absmax (max is exact, order-free)
#pragma unroll
    for (int m = 16; m >= 1; m /= 2) amax = fmaxf(amax, __shfl_xor(amax, m));
    if (lane == 0U) red[t / 32U] = amax;
    __syncthreads();
    const unsigned waves = (blockDim.x + 31U) / 32U;
    amax = red[0];
    for (unsigned w = 1; w < waves; ++w) amax = fmaxf(amax, red[w]);
    float s = __fdiv_rn(amax, 127.0F);
    if (!(s >= 1e-30F)) s = 1e-30F;
    float sb = bf16_to_f32(f32_to_bf16(s));
    if (sb == 0.0F) sb = 0x1p-126F;
    if (t == 0U) scale[row] = s;
    if (flushed && nflush) atomicAdd(flushed, nflush);
#pragma unroll
    for (unsigned it = 0; it < Iters; ++it) {
        const std::uint32_t g = team + teams * it;
        if (g >= groups) continue;
        unsigned word[4] = {0U, 0U, 0U, 0U};
#pragma unroll
        for (unsigned j = 0; j < 16U; ++j) {
            const float tq = bf16_to_f32(f32_to_bf16(__fdiv_rn(bf16_to_f32(out[it][j]), sb)));
            const float r = fminf(127.0F, fmaxf(-128.0F, rintf(tq)));
            word[j / 4U] |= (static_cast<unsigned>(static_cast<int>(r)) & 0xffU) << (8U * (j % 4U));
        }
        *reinterpret_cast<uint4*>(q + static_cast<std::uint64_t>(row) * k + 256U * g + 16U * l) = make_uint4(word[0], word[1], word[2], word[3]);
    }
}

// --------------------------------------------------------------------------------------------------------------------- GEMM
struct OutSegs {
    std::uint16_t* ptr[4] = {};
    std::uint32_t ld = 0, seg_cols = 0;
    __device__ __forceinline__ std::uint16_t* column(std::uint32_t col) const {
        const std::uint32_t seg = col / seg_cols;
        std::uint16_t* p = seg == 0U ? ptr[0] : seg == 1U ? ptr[1] : seg == 2U ? ptr[2] : ptr[3];
        return p + (col - seg * seg_cols);
    }
};
enum class Epi : unsigned { plain = 0, swiglu = 1, raw = 2 };  // raw: int32 accumulators to acc_out [m][n] (tests)
struct GemmArgs {
    const std::int8_t* a = nullptr;   // codes [m][k]
    const std::int8_t* b = nullptr;   // weights [n][k] (swiglu: swiglu_permute()d)
    const float* row_scale = nullptr; // [m]
    const float* ch_scale = nullptr;  // [n] (swiglu: permuted with the rows)
    const std::uint16_t* silu = nullptr;  // swiglu: the 65536-entry table
    OutSegs out;                      // plain: n columns; swiglu: n / 2 columns
    std::int32_t* acc_out = nullptr;  // raw
    std::uint32_t m = 0, n = 0, k = 0, raster_m = 8;
};

constexpr unsigned kBM = 256U, kWM = 64U, kWN = 64U, kStageK = 64U;
__device__ __forceinline__ unsigned lds_at(unsigned row, unsigned chunk) { return row * 64U + ((chunk ^ ((row / 2U) % 4U)) * 16U); }
__device__ __forceinline__ void lds_barrier() { __asm__ volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }

template <unsigned BN, Epi E>
__global__ __launch_bounds__(BN == 256U ? 512U : 256U) void gemm_kernel(GemmArgs g) {
#pragma clang fp contract(off)
    constexpr unsigned Threads = BN == 256U ? 512U : 256U, WavesN = BN / kWN, MT = kWM / 16U, NT = kWN / 16U;
    constexpr unsigned StageBytes = (kBM + BN) * 64U;
    constexpr unsigned AChunks = kBM * 4U / Threads, BChunks = BN * 4U / Threads;  // 16-byte chunks per thread
    __shared__ __align__(16) unsigned char lds[2U * StageBytes];
    const unsigned thread = threadIdx.x, lane = thread % 32U;
    const unsigned wave = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(thread / 32U)));
    const unsigned wm = wave / WavesN, wn = wave % WavesN;
    const std::uint32_t m = g.m, n = g.n, k = g.k;
    const std::uint32_t tiles_m = (m + kBM - 1U) / kBM, tiles_n = n / BN;
    const std::uint32_t pid = blockIdx.x, per_group = g.raster_m * tiles_n, group = pid / per_group, first_m = group * g.raster_m;
    const std::uint32_t group_m = min(g.raster_m, tiles_m - first_m), local = pid - group * per_group;
    const std::uint32_t m0 = (first_m + local % group_m) * kBM, n0 = (local / group_m) * BN;
    std::uint32_t stages = k / kStageK;
    asm volatile("" : "+s"(stages));
    // staging: chunk c = thread + Threads * i -> row c / 4, 16-byte part c % 4
    std::uint32_t aoff[AChunks], boff[BChunks];
#pragma unroll
    for (unsigned i = 0; i < AChunks; ++i) {
        const unsigned c = thread + Threads * i;
        aoff[i] = min(m0 + c / 4U, m - 1U) * k + 16U * (c % 4U);
    }
#pragma unroll
    for (unsigned i = 0; i < BChunks; ++i) {
        const unsigned c = thread + Threads * i;
        boff[i] = (n0 + c / 4U) * k + 16U * (c % 4U);
    }
    u32x4 areg[AChunks], breg[BChunks];
    const auto fetch = [&](std::uint32_t s) {
#pragma unroll
        for (unsigned i = 0; i < AChunks; ++i) areg[i] = g16(g.a + kStageK * s + aoff[i]);
#pragma unroll
        for (unsigned i = 0; i < BChunks; ++i) breg[i] = g16(g.b + kStageK * s + boff[i]);
    };
    const auto store = [&](unsigned buffer) {
        unsigned char* as = lds + buffer * StageBytes;
        unsigned char* bs = as + kBM * 64U;
#pragma unroll
        for (unsigned i = 0; i < AChunks; ++i) { const unsigned c = thread + Threads * i; *reinterpret_cast<u32x4*>(as + lds_at(c / 4U, c % 4U)) = areg[i]; }
#pragma unroll
        for (unsigned i = 0; i < BChunks; ++i) { const unsigned c = thread + Threads * i; *reinterpret_cast<u32x4*>(bs + lds_at(c / 4U, c % 4U)) = breg[i]; }
    };
    v8i acc[MT][NT];
#pragma unroll
    for (unsigned t = 0; t < MT; ++t)
#pragma unroll
        for (unsigned u = 0; u < NT; ++u) acc[t][u] = v8i{};
    const auto compute = [&](unsigned buffer) {
        const unsigned char* as = lds + buffer * StageBytes;
        const unsigned char* bs = as + kBM * 64U;
#pragma unroll
        for (unsigned kt = 0; kt < 4U; ++kt) {
            v4i b[NT];
#pragma unroll
            for (unsigned u = 0; u < NT; ++u) b[u] = *reinterpret_cast<const v4i*>(bs + lds_at(wn * kWN + 16U * u + lane % 16U, kt));
#pragma unroll
            for (unsigned t = 0; t < MT; ++t) {
                const v4i a = *reinterpret_cast<const v4i*>(as + lds_at(wm * kWM + 16U * t + lane % 16U, kt));
#pragma unroll
                for (unsigned u = 0; u < NT; ++u) acc[t][u] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b[u], acc[t][u], false);
            }
        }
    };
    fetch(0U);
    store(0U);
    lds_barrier();
    // stages = k / 64 is even (k % 128 == 0): even stages buffer 0, odd stages buffer 1
#pragma unroll 1
    for (std::uint32_t s = 0; s < stages; s += 2U) {
        fetch(s + 1U);
        compute(0U);
        store(1U);
        lds_barrier();
        const bool more = s + 2U < stages;
        if (more) fetch(s + 2U);
        compute(1U);
        if (more) store(0U);
        lds_barrier();
    }
    // epilogue
    const float* rs = g.row_scale;
#pragma unroll
    for (unsigned t = 0; t < MT; ++t) {
        float rsc[8];
        std::uint32_t rows[8];
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            rows[e] = m0 + wm * kWM + 16U * t + 2U * e + lane / 16U;
            rsc[e] = rs[min(rows[e], m - 1U)];
        }
        if constexpr (E == Epi::raw) {
#pragma unroll
            for (unsigned u = 0; u < NT; ++u) {
                const std::uint32_t col = n0 + wn * kWN + 16U * u + lane % 16U;
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e)
                    if (rows[e] < m) g.acc_out[static_cast<std::uint64_t>(rows[e]) * n + col] = acc[t][u][e];
            }
        } else if constexpr (E == Epi::plain) {
#pragma unroll
            for (unsigned u = 0; u < NT; ++u) {
                const std::uint32_t col = n0 + wn * kWN + 16U * u + lane % 16U;
                const float cs = g.ch_scale[col];
                std::uint16_t* base = g.out.column(col);
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e)
                    if (rows[e] < m) base[static_cast<std::uint64_t>(rows[e]) * g.out.ld] = epilogue(acc[t][u][e], rsc[e], cs);
            }
        } else {
#pragma unroll
            for (unsigned u = 0; u < NT; u += 2U) {
                const std::uint32_t pc = n0 + wn * kWN + 16U * u + lane % 16U;  // gate column (permuted); up = pc + 16
                const float cg = g.ch_scale[pc], cu = g.ch_scale[pc + 16U];
                const std::uint32_t col = n0 / 2U + wn * (kWN / 2U) + 8U * u + lane % 16U;  // h column
                std::uint16_t* base = g.out.column(col);
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) {
                    if (rows[e] >= m) continue;
                    const std::uint16_t gate = epilogue(acc[t][u][e], rsc[e], cg), up = epilogue(acc[t][u + 1U][e], rsc[e], cu);
                    base[static_cast<std::uint64_t>(rows[e]) * g.out.ld] = f32_to_bf16(bf16_to_f32(g.silu[gate]) * bf16_to_f32(up));
                }
            }
        }
    }
}

}  // namespace strix::h3::gemm::dev

namespace strix::h3::gemm {

// SwiGLU weight order for the fused fc1 epilogue: permuted row p (tile p / 128, wave half wn, 16-column tile u, column c)
// holds source row (u odd ? half : 0) + tile * 64 + wn * 32 + (u / 2) * 16 + c, half = n / 2 (fc1: [gate | up] halves).
// Valid for both tile widths (the epilogue pairs tiles u, u + 1 of one 64-column wave tile).
inline std::uint32_t swiglu_source_row(std::uint32_t p, std::uint32_t n) {
    const std::uint32_t tile = p / 128U, within = p % 128U, wn = within / 64U, u = (within % 64U) / 16U, c = within % 16U;
    return (u % 2U ? n / 2U : 0U) + tile * 64U + wn * 32U + (u / 2U) * 16U + c;
}

// Launchers (low level: no error collection; api.hpp checks launches). Both validate their arguments and throw
// std::invalid_argument: quantizer 0 < k <= 16384, k % 256 == 0, x 16-byte aligned with x_ld >= k and a 16-byte row pitch;
// GEMM k % 128 == 0, k <= 16384 (int32 accumulators), n % BN == 0, m * k and n * k < 2^32 (32-bit operand offsets),
// raster_m in [1, 64], segments: seg_cols > 0, at most 4 addressed segments, each pointer set, pitch >= the
// columns one segment holds.
// Quantizer variant (identical bits): 0 = baseline (256 threads, int64 transform), 1 = fast (one 16-lane team per
// group, int32 transform for groups spanning <= 15 binades). Default (-1 = auto): fast for K 5376 and 14336, where it
// measured faster (gfx1151: 1.16-1.26x at K 5376, 1.32x at K 14336, 1.96x on wide-span fc2 rows), baseline
// otherwise (K 7168: fast 0.88-0.93x). STRIX_H3_GEMM_QUANT_FAST=0/1 forces one variant for every K.
constexpr int kQuantVariantAuto = -1;
inline std::atomic<int>& quantizer_variant_slot() {
    static std::atomic<int> v{[] { const char* e = std::getenv("STRIX_H3_GEMM_QUANT_FAST"); return e && (e[0] == '0' || e[0] == '1') ? e[0] - '0' : kQuantVariantAuto; }()};
    return v;
}
inline int quantizer_variant(std::uint32_t k) {
    const int v = quantizer_variant_slot().load(std::memory_order_relaxed);
    return v >= 0 ? v : (k == 5376U || k == 14336U ? 1 : 0);
}
inline void set_quantizer_variant(int v) { quantizer_variant_slot().store(v == 0 || v == 1 ? v : kQuantVariantAuto, std::memory_order_relaxed); }  // process-wide
inline void launch_quantize(const std::uint16_t* x, std::uint64_t x_ld, std::uint32_t m, std::uint32_t k, std::int8_t* q, float* scale,
    unsigned* flushed, hipStream_t stream, int variant = -1) {
    if (!m || !k || k % 256U || k > 16384U || !x || !q || !scale || reinterpret_cast<std::uintptr_t>(x) % 16U || x_ld < k || (x_ld * 2U) % 16U)
        throw std::invalid_argument("h3 gemm: invalid quantizer arguments");
    if (variant < 0) variant = quantizer_variant(k);
    if (variant == 1)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(dev::quantize_kernel<true, 1U>), dim3(m), dim3(16U * (k / 256U)), 0, stream, x, x_ld, k, q, scale, flushed);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(dev::quantize_kernel<false, dev::kQMaxIters>), dim3(m), dim3(dev::kQThreads), 0, stream, x, x_ld, k, q, scale, flushed);
}
template <unsigned BN, dev::Epi E> inline void launch_gemm(const dev::GemmArgs& args, hipStream_t stream) {
    const auto& g = args;
    // k <= 16384 keeps |acc| <= 16384 * 128 * 128 = 2^28 inside int32; 32-bit operand offsets; raster_m in [1, 64] and
    // the tile counts small enough that raster_m * tiles_n and the grid size cannot wrap.
    const std::uint64_t tiles_m = (static_cast<std::uint64_t>(g.m) + dev::kBM - 1U) / dev::kBM, tiles_n = g.n / BN;
    bool ok = g.m && g.k && g.k % 128U == 0U && g.k <= 16384U && g.n && g.n % BN == 0U && g.a && g.b && g.row_scale &&
        g.raster_m >= 1U && g.raster_m <= 64U && static_cast<std::uint64_t>(g.raster_m) * tiles_n <= 0x7fffffffULL &&
        tiles_m * tiles_n <= 0x7fffffffULL &&
        static_cast<std::uint64_t>(g.m) * g.k <= 0xffffffffULL && static_cast<std::uint64_t>(g.n) * g.k <= 0xffffffffULL;
    if constexpr (E == dev::Epi::raw) ok = ok && g.acc_out;
    else {
        ok = ok && g.ch_scale && g.out.seg_cols;
        if constexpr (E == dev::Epi::swiglu) ok = ok && g.silu;
        const std::uint32_t cols = E == dev::Epi::swiglu ? g.n / 2U : g.n;
        const std::uint32_t segs = ok ? (cols + g.out.seg_cols - 1U) / g.out.seg_cols : 0U;
        ok = ok && segs >= 1U && segs <= 4U && g.out.ld >= (segs == 1U ? cols : g.out.seg_cols);
        for (std::uint32_t s = 0; ok && s < segs; ++s) ok = g.out.ptr[s] != nullptr;
    }
    if (!ok) throw std::invalid_argument("h3 gemm: invalid GEMM arguments");
    const std::uint32_t tiles = (args.m + dev::kBM - 1U) / dev::kBM * (args.n / BN);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(dev::gemm_kernel<BN, E>), dim3(tiles), dim3(BN == 256U ? 512U : 256U), 0, stream, args);
}

}  // namespace strix::h3::gemm
