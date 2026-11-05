#include "strix/ds41_prefill_tuning.hpp"
// gfx1151 kernels of the deepseek41 decode attention. The arithmetic of each kernel is its host reference in
// src/ds41_attention.cpp (same chains, same xor butterflies, same scalar helpers), so results match bit for bit.
#include "strix/hip_ds41_attention.hpp"
#include "hip_ds41_attention_qk.inc"

#include <cstdlib>
#include <stdexcept>

namespace strix::ds41::attn::hip {
#if defined(STRIX_DS41_ATTN_TIMING)
// Timing build only: thread 0 of every block records wall clock and shader cycles at entry and exit.
__device__ unsigned long long* g_ds41_timing;
void set_timing_buffer(void* p) {
    if (hipMemcpyToSymbol(HIP_SYMBOL(g_ds41_timing), &p, sizeof(p)) != hipSuccess) throw std::runtime_error("timing buffer");
}
#endif
namespace {
#if defined(STRIX_DS41_ATTN_TIMING)
struct BlockTimer {
    unsigned slot; unsigned long long w0, c0;
    __device__ explicit BlockTimer(unsigned s) : slot(s), w0(static_cast<unsigned long long>(wall_clock64())), c0(__builtin_readcyclecounter()) {}
    __device__ ~BlockTimer() {
        if (threadIdx.x != 0U || g_ds41_timing == nullptr) return;
        const unsigned long long b = blockIdx.x + gridDim.x * (blockIdx.y + static_cast<unsigned long long>(gridDim.y) * blockIdx.z);
        unsigned long long* t = g_ds41_timing + (static_cast<unsigned long long>(slot) * 65536ULL + b) * 4ULL;
        t[0] = w0; t[1] = static_cast<unsigned long long>(wall_clock64()); t[2] = c0; t[3] = __builtin_readcyclecounter();
    }
};
#define DS41_TIMED(s) const BlockTimer ds41_timer_(s)
// phase mark k of the block (thread 0, after its outstanding memory operations): slots 7.. of the timing buffer
#define DS41_MARK(k) do { __builtin_amdgcn_s_waitcnt(0); if (threadIdx.x == 0U && g_ds41_timing != nullptr) {     const unsigned long long b_ = blockIdx.x + gridDim.x * (blockIdx.y + static_cast<unsigned long long>(gridDim.y) * blockIdx.z);     g_ds41_timing[((7ULL + (k) / 4U) * 65536ULL + b_) * 4ULL + (k) % 4U] = static_cast<unsigned long long>(wall_clock64()); } } while (0)
// shader-cycle mark k of the block (thread 0, after its outstanding memory operations): slots 11.. (20-bit counter)
#define DS41_CMARK(k) do { __builtin_amdgcn_s_waitcnt(0); if (threadIdx.x == 0U && g_ds41_timing != nullptr) {     const unsigned long long b_ = blockIdx.x + gridDim.x * (blockIdx.y + static_cast<unsigned long long>(gridDim.y) * blockIdx.z);     g_ds41_timing[((11ULL + (k) / 4U) * 65536ULL + b_) * 4ULL + (k) % 4U] = __builtin_readcyclecounter(); } } while (0)
#else
#define DS41_TIMED(s) (void)0
#define DS41_MARK(k) (void)0
#define DS41_CMARK(k) (void)0
#endif

// xor lane exchange without LDS: DPP row_xmask inside the 16-lane rows, permlanex16 across them
template <unsigned D> __device__ __forceinline__ float xor_lane(float v) {
    const unsigned x = __builtin_bit_cast(unsigned, v);
    if constexpr (D == 16U) return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(x, x, 0x76543210U, 0xfedcba98U, false, false));
    else return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(static_cast<int>(x), static_cast<int>(x), static_cast<int>(0x160U + D), 0xF, 0xF, false));
}
// The same exchange without tying the result to the source register (DPP old operand undefined: every lane of a
// row_xmask exchange is in bounds), so the source stays live without a copy. The empty asm keeps the moved value out
// of DPP combining: a combined v_add_f32_dpp takes the partner as src0, and with two NaN operands v_add_f32 returns
// the src0 NaN (gfx1151), so own + partner keeps its operand order.
template <unsigned D> __device__ __forceinline__ float xor_lane_nt(float v) {
    const unsigned x = __builtin_bit_cast(unsigned, v);
    if constexpr (D == 16U) return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(x, x, 0x76543210U, 0xfedcba98U, false, false));
    else {
        int r = __builtin_amdgcn_mov_dpp(static_cast<int>(x), static_cast<int>(0x160U + D), 0xF, 0xF, true);
        asm("" : "+v"(r));
        return __builtin_bit_cast(float, r);
    }
}
// The value of quad lane 0 in every lane of the quad (DPP quad_perm [0, 0, 0, 0])
__device__ __forceinline__ float quad_first(float v) {
    return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x00, 0xF, 0xF, true));
}
__device__ __forceinline__ float wave_sum(float v) {
    v = v + xor_lane<16>(v);
    v = v + xor_lane<8>(v);
    v = v + xor_lane<4>(v);
    v = v + xor_lane<2>(v);
    v = v + xor_lane<1>(v);
    return v;
}
__device__ __forceinline__ float wave_max(float v, int from = 16) {
    for (int d = from; d != 0; d /= 2) v = fmaxf(v, __shfl_xor(v, d, 32));
    return v;
}
// RMS norm of the wave's 512 vector (lane: 16 dims), in place: (x * r) * w.
__device__ __forceinline__ void norm512(float (&x)[16], const float* w, unsigned lane) {
    float acc = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) acc = fmaf(x[i], x[i], acc);
    const float r = 1.0F / sqrtf(wave_sum(acc) / 512.0F + kNormEps);
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) x[i] = (x[i] * r) * w[16 * lane + i];
}
// lanes 28..31 own dims 448..511 = rope pairs 8(lane-28)..+7
__device__ __forceinline__ void rope512(float (&x)[16], const RopeRow& rope, unsigned lane, bool inverse) {
    if (lane < 28U) return;
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
        const unsigned pair = 8U * (lane - 28U) + i;
        rope_pair(x[2 * i], x[2 * i + 1], rope.c[pair], rope.s[pair], inverse);
    }
}

// One wave: the ring entry of one kv row (lane: dims 16 lane .. +15, its 16 code bytes and, even lanes, the scale byte).
__device__ __forceinline__ void window_entry(const float* kv, const float* norm, const RopeRow& rope, unsigned lane, std::uint8_t* entry) {
    float x[16];
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) x[i] = kv[16 * lane + i];
    norm512(x, norm, lane);
    rope512(x, rope, lane, false);
    float amax = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(x[i]));
    amax = fmaxf(fmaxf(amax, __shfl_xor(amax, 1, 32)), 1e-4F);
    const int e = log2_ceil(amax * (1.0F / 448.0F));
    const float s = pow2f(e);
    std::uint32_t w[4] = {0, 0, 0, 0};
#pragma unroll
    for (unsigned i = 0; i < 16; ++i)
        w[i / 4] |= static_cast<std::uint32_t>(e4m3_encode(clampf(x[i] / s, -448.0F, 448.0F))) << (8U * (i % 4));
    *reinterpret_cast<uint4*>(entry + 16 * lane) = make_uint4(w[0], w[1], w[2], w[3]);
    if ((lane & 1U) == 0U) entry[512 + lane / 2] = static_cast<std::uint8_t>(e + 127);
}
__device__ __forceinline__ void window_write_body(const WindowView& v, const uint3 bid) {
    DS41_TIMED(0U);
    const unsigned r = bid.x;
    window_entry(v.kv + r * kHeadDim, v.norm, v.rope[r], threadIdx.x, v.ring + (v.pos[r] % v.capacity) * kWindowEntryBytes);
}

// Ahead (STRIX_DS41_COMPRESS_AHEAD=1, default off): thread tid's index key row of wk is read kAheadSteps steps of 64
// weights ahead (the first ones before the latent barrier) instead of one step at a time;
// the chain of each output (512 fmaf over d ascending) and everything else are unchanged.
constexpr unsigned kAheadSteps = 3;
// Fast (STRIX_DS41_COMPRESS_FAST, default on, =0 off): the same values in fewer instructions. f16_to_float compiles to
// per-element exec-mask branches in the 512-term chains; here a finite F16 converts in the fmaf itself (v_fma_mix:
// exact, the same float for normals, subnormals with F16 denormals on and signed zeros) and a step of 64 weights with
// an exponent of 31 in any of its lanes takes f16_to_float itself; the latent is read from LDS 4 floats at a time and
// the next step's weights are requested before the current step's chain. exp_f's range tests become selects over the
// same polynomial, e2m1_encode's chain a sum of its tests, and the carry rows are read before they are stored. Every
// chain (512 fmaf over d ascending, the merge, the norms) is unchanged.
__device__ __forceinline__ float exp_f_sel(float x) {
    const float n = __builtin_rintf(x * 1.44269504088896341F);
    float r = __builtin_fmaf(n, -0.693145751953125F, x);
    r = __builtin_fmaf(n, -1.42860676533018e-06F, r);
    float q = 2.48015873e-05F;
    q = __builtin_fmaf(q, r, 1.98412698e-04F);
    q = __builtin_fmaf(q, r, 1.38888889e-03F);
    q = __builtin_fmaf(q, r, 8.33333333e-03F);
    q = __builtin_fmaf(q, r, 4.16666667e-02F);
    q = __builtin_fmaf(q, r, 1.66666667e-01F);
    q = __builtin_fmaf(q, r, 0.5F);
    q = __builtin_fmaf(q, r, 1.0F);
    q = __builtin_fmaf(q, r, 1.0F);
    // n clamped to pow2f's range before the conversion (an infinite or NaN x gives n = +-inf / NaN, whose int conversion
    // is undefined): the same n for every x the select keeps (-87 <= x <= 88), and a NaN x still gives NaN through q
    const float e = q * pow2f(static_cast<int>(__builtin_fminf(__builtin_fmaxf(n, -126.0F), 127.0F)));
    return x < -87.0F ? 0.0F : (x > 88.0F ? __builtin_inff() : e);
}
// some F16 of the pair has exponent 31 (inf / NaN)
__device__ __forceinline__ unsigned f16x2_special(std::uint32_t w) {
    return static_cast<unsigned>((~w & 0x7C00U) == 0U) | static_cast<unsigned>((~w & 0x7C000000U) == 0U);
}
__device__ __forceinline__ float f16_finite(std::uint32_t w, unsigned hi) {
    return static_cast<float>(__builtin_bit_cast(_Float16, static_cast<std::uint16_t>(w >> (16U * hi))));
}
// e2m1_encode as a count of the thresholds not below the magnitude: the first true test of its chain is the number
// of false tests before it (the thresholds ascend), and a NaN fails every test (code 7) in both forms.
__device__ __forceinline__ std::uint8_t e2m1_encode_sel(float x) {
    const std::uint8_t sign = (f2u(x) >> 31U) != 0U ? 8U : 0U;
    const float a = x < 0.0F ? -x : x;
    const unsigned m = static_cast<unsigned>(!(a <= 0.25F)) + static_cast<unsigned>(!(a < 0.75F)) + static_cast<unsigned>(!(a <= 1.25F)) +
                       static_cast<unsigned>(!(a < 1.75F)) + static_cast<unsigned>(!(a <= 2.5F)) + static_cast<unsigned>(!(a < 3.5F)) +
                       static_cast<unsigned>(!(a <= 5.0F));
    return static_cast<std::uint8_t>(m | sign);
}
// Fast takes precedence over Ahead (it requests the next step itself).
template <bool Ahead, bool Fast>
__device__ __forceinline__ void compress_body(const CompressView& v, const uint3 bid) {
    DS41_TIMED(1U);
    __shared__ float latent_s[kHeadDim];
    __shared__ float k_s[kIdxDim];
    const unsigned r = bid.x, tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U;
    const std::uint64_t p = v.pos[r];
    [[maybe_unused]] const uint4* wrow = reinterpret_cast<const uint4*>(v.wk + tid * kHeadDim);  // 8 F16 weights per uint4
    [[maybe_unused]] uint4 ahead[kAheadSteps][8];
    float x[16];
    if (v.ratio == 2U) {
        const float* kv = v.kv + r * kHeadDim + 16 * lane;
        const float* sc = v.score + r * kHeadDim + 16 * lane;
        if (wave == 0U && r + 1U == v.rows) {  // the pending group after the last row
            const bool pending = (p % 2U) == 0U;
            if constexpr (Fast) {   // both rows read unconditionally, then stored: no load-to-store round trip per element
                float a[16], b[16];
#pragma unroll
                for (unsigned i = 0; i < 16; ++i) { a[i] = kv[i]; b[i] = sc[i]; }
#pragma unroll
                for (unsigned i = 0; i < 16; ++i) {
                    v.state_out[16 * lane + i] = pending ? a[i] : 0.0F;
                    v.state_out[kHeadDim + 16 * lane + i] = pending ? b[i] : 0.0F;
                }
            } else {
                for (unsigned i = 0; i < 16; ++i) {
                    v.state_out[16 * lane + i] = pending ? kv[i] : 0.0F;
                    v.state_out[kHeadDim + 16 * lane + i] = pending ? sc[i] : 0.0F;
                }
            }
            if (lane == 0U) *v.state_valid_out = pending ? 1U : 0U;
        }
        if ((p % 2U) == 0U) return;  // uniform over the block (no barrier passed yet)
        if (wave == 0U) {
            const float* kv0 = r > 0U ? v.kv + (r - 1U) * kHeadDim + 16 * lane : v.state_in + 16 * lane;
            const float* sc0 = r > 0U ? v.score + (r - 1U) * kHeadDim + 16 * lane : v.state_in + kHeadDim + 16 * lane;
            float a0[16], b0[16], a1[16], b1[16];
#pragma unroll
            for (unsigned i = 0; i < 16; ++i) { a0[i] = kv0[i]; b0[i] = sc0[i]; a1[i] = kv[i]; b1[i] = sc[i]; }
            DS41_CMARK(0U);
#pragma unroll
            for (unsigned i = 0; i < 16; ++i) {
                const float m = fmaxf(b0[i], b1[i]);
                const float e0 = Fast ? exp_f_sel(b0[i] - m) : exp_f(b0[i] - m), e1 = Fast ? exp_f_sel(b1[i] - m) : exp_f(b1[i] - m);
                const float sum = e0 + e1;
                x[i] = a0[i] * (e0 / sum) + a1[i] * (e1 / sum);
            }
            DS41_CMARK(1U);
        }
    } else if (wave == 0U) {
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) x[i] = v.kv[r * kHeadDim + 16 * lane + i];
        DS41_CMARK(0U);
    }
    if (wave == 0U) {
        norm512(x, v.norm, lane);
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) latent_s[16 * lane + i] = x[i];
        DS41_CMARK(2U);
    }
    if constexpr (Ahead && !Fast) {  // in flight across the barrier
#pragma unroll
        for (unsigned a = 0; a < kAheadSteps; ++a)
#pragma unroll
            for (unsigned u = 0; u < 8; ++u) ahead[a][u] = wrow[8 * a + u];
    }
    __syncthreads();
    DS41_CMARK(3U);
    if constexpr (Fast) {   // the same chain: hardware conversion of finite weights, next step requested ahead
        const uint4* row = reinterpret_cast<const uint4*>(v.wk + tid * kHeadDim);
        const float4* lat = reinterpret_cast<const float4*>(latent_s);
        float acc = 0.0F;
        uint4 h[8], nh[8];
#pragma unroll
        for (unsigned u = 0; u < 8; ++u) nh[u] = row[u];
#pragma unroll 1
        for (unsigned n = 0; n < kHeadDim / 64U; ++n) {
            unsigned special = 0U;
#pragma unroll
            for (unsigned u = 0; u < 8; ++u) {
                h[u] = nh[u];
                special |= f16x2_special(h[u].x) | f16x2_special(h[u].y) | f16x2_special(h[u].z) | f16x2_special(h[u].w);
            }
            if (n + 1U < kHeadDim / 64U) {
#pragma unroll
                for (unsigned u = 0; u < 8; ++u) nh[u] = row[8 * (n + 1U) + u];
            }
            if (special != 0U) {
#pragma unroll
                for (unsigned u = 0; u < 8; ++u) {
                    const std::uint32_t hw[4] = {h[u].x, h[u].y, h[u].z, h[u].w};
#pragma unroll
                    for (unsigned j = 0; j < 8; ++j)
                        acc = fmaf(f16_to_float(static_cast<std::uint16_t>(hw[j / 2] >> (16U * (j % 2)))), latent_s[64 * n + 8 * u + j], acc);
                }
            } else {
#pragma unroll
                for (unsigned u = 0; u < 8; ++u) {
                    const std::uint32_t hw[4] = {h[u].x, h[u].y, h[u].z, h[u].w};
                    const float4 l0 = lat[16 * n + 2 * u], l1 = lat[16 * n + 2 * u + 1];
                    const float l[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
#pragma unroll
                    for (unsigned j = 0; j < 8; ++j) acc = fmaf(f16_finite(hw[j / 2], j % 2), l[j], acc);
                }
            }
        }
        k_s[tid] = acc;
    } else if constexpr (!Ahead) {   // index key output tid: one 512-term fmaf chain over d ascending; 64 weights loaded ahead per step
        const std::uint16_t* row = v.wk + tid * kHeadDim;
        float acc = 0.0F;
        for (unsigned d0 = 0; d0 < kHeadDim; d0 += 64) {
            uint4 h[8];
#pragma unroll
            for (unsigned u = 0; u < 8; ++u) h[u] = *reinterpret_cast<const uint4*>(row + d0 + 8 * u);
#pragma unroll
            for (unsigned u = 0; u < 8; ++u) {
                const std::uint32_t hw[4] = {h[u].x, h[u].y, h[u].z, h[u].w};
#pragma unroll
                for (unsigned j = 0; j < 8; ++j)
                    acc = fmaf(f16_to_float(static_cast<std::uint16_t>(hw[j / 2] >> (16U * (j % 2)))), latent_s[d0 + 8 * u + j], acc);
            }
        }
        k_s[tid] = acc;
    } else {  // the same chain; step n's weights were requested kAheadSteps steps earlier
        float acc = 0.0F;
#pragma unroll 1
        for (unsigned n = 0; n < kHeadDim / 64U; ++n) {  // not unrolled (one step's live values, as above): the queue shifts
            uint4 h[8];
#pragma unroll
            for (unsigned u = 0; u < 8; ++u) h[u] = ahead[0][u];
#pragma unroll
            for (unsigned a = 0; a + 1U < kAheadSteps; ++a)
#pragma unroll
                for (unsigned u = 0; u < 8; ++u) ahead[a][u] = ahead[a + 1U][u];
            if (n + kAheadSteps < kHeadDim / 64U) {
#pragma unroll
                for (unsigned u = 0; u < 8; ++u) ahead[kAheadSteps - 1U][u] = wrow[8 * (n + kAheadSteps) + u];
            }
#pragma unroll
            for (unsigned u = 0; u < 8; ++u) {
                const std::uint32_t hw[4] = {h[u].x, h[u].y, h[u].z, h[u].w};
#pragma unroll
                for (unsigned j = 0; j < 8; ++j)
                    acc = fmaf(f16_to_float(static_cast<std::uint16_t>(hw[j / 2] >> (16U * (j % 2)))), latent_s[64 * n + 8 * u + j], acc);
            }
        }
        k_s[tid] = acc;
    }
    DS41_CMARK(4U);
    __syncthreads();
    DS41_CMARK(5U);
    if (wave != 0U) return;
    const RopeRow& rope = v.rope[r];
    const std::uint64_t slot = p / v.ratio;
    float k[4];
#pragma unroll
    for (unsigned o = 0; o < 4; ++o) k[o] = k_s[4 * lane + o];
    float ss = 0.0F;
#pragma unroll
    for (unsigned o = 0; o < 4; ++o) ss = fmaf(k[o], k[o], ss);
    const float rk = 1.0F / sqrtf(wave_sum(ss) / 128.0F + kNormEps);
#pragma unroll
    for (unsigned o = 0; o < 4; ++o) k[o] = (k[o] * rk) * v.knorm[4 * lane + o];
    DS41_CMARK(6U);
    if (lane >= 16U) {
        rope_pair(k[0], k[1], rope.c[2 * (lane - 16U)], rope.s[2 * (lane - 16U)], false);
        rope_pair(k[2], k[3], rope.c[2 * (lane - 16U) + 1], rope.s[2 * (lane - 16U) + 1], false);
    }
    float amax = fmaxf(fmaxf(fabsf(k[0]), fabsf(k[1])), fmaxf(fabsf(k[2]), fabsf(k[3])));
    amax = fmaxf(wave_max(amax, 4), 7.05296657e-38F);
    const int e = log2_ceil(amax * (1.0F / 6.0F));
    const float s = pow2f(e);
    std::uint8_t* ie = v.idx_cache + slot * kIdxEntryBytes;
    std::uint8_t c[4];
#pragma unroll
    for (unsigned o = 0; o < 4; ++o) c[o] = Fast ? e2m1_encode_sel(clampf(k[o] / s, -6.0F, 6.0F)) : e2m1_encode(clampf(k[o] / s, -6.0F, 6.0F));
    ie[2 * lane] = static_cast<std::uint8_t>(c[0] | (c[1] << 4U));
    ie[2 * lane + 1] = static_cast<std::uint8_t>(c[2] | (c[3] << 4U));
    if ((lane & 7U) == 0U) ie[64 + lane / 8] = static_cast<std::uint8_t>(e + 127);
    DS41_CMARK(7U);
    // compressed KV: RoPE the latent, FP4 with an E4M3 scale per lane (16 dims)
    rope512(x, rope, lane, false);
    DS41_CMARK(8U);
    float am = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) am = fmaxf(am, fabsf(x[i]));
    am = fmaxf(am, 0.01171875F);
    const std::uint8_t sc = e4m3_encode(am / 6.0F);
    const float sv = e4m3_value(sc);
    std::uint32_t w[2] = {0, 0};
#pragma unroll
    for (unsigned i = 0; i < 16; ++i)
        w[i / 8] |= static_cast<std::uint32_t>(Fast ? e2m1_encode_sel(clampf(x[i] / sv, -6.0F, 6.0F)) : e2m1_encode(clampf(x[i] / sv, -6.0F, 6.0F)))
                    << (4U * (i % 8));
    std::uint8_t* ke = v.kv_cache + slot * kKvEntryBytes;
    *reinterpret_cast<uint2*>(ke + 8 * lane) = make_uint2(w[0], w[1]);
    ke[256 + lane] = sc;
    DS41_CMARK(9U);
}

// 4 E2M1 codes (16 bits: dims 4i..4i+3, low nibble first) -> 4 int8 values (2 x value).
__device__ __forceinline__ std::uint32_t e2m1x4_to_i8(std::uint32_t x) {
    const std::uint32_t sel = (x & 7U) | ((x << 4U) & 0x700U) | ((x << 8U) & 0x70000U) | ((x << 12U) & 0x7000000U);
    const std::uint32_t mag = __builtin_amdgcn_perm(0x0C080604U, 0x03020100U, sel);
    const std::uint32_t sg = ((x >> 3U) & 1U) | ((x >> 7U) & 1U) << 8U | ((x >> 11U) & 1U) << 16U | ((x >> 15U) & 1U) << 24U;
    std::uint32_t nz = mag | (mag >> 2U);
    nz = (nz | (nz >> 1U)) & 0x01010101U;
    const std::uint32_t m = (sg & nz) * 0xFFU;
    return (mag ^ m) + (m & 0x01010101U);
}

// Index queries, once per row: head tid/8, dims 16 (tid%8) .. +15; RoPE on head dims 64..127; E2M1 per 32
// (threads tid, tid^1) as packed int8 (2 x value), the block scale 2^e as float, and the head weight / 64.
struct QPrep { int qv[kIdxHeads][32]; float qs[kIdxHeads][4]; float w[kIdxHeads]; };
static_assert(sizeof(QPrep) == kIdxPrepBytes);
__device__ __forceinline__ void index_qprep_body(const IndexView& v, const uint3 bid) {
    DS41_TIMED(2U);
    const unsigned r = bid.x, tid = threadIdx.x, h = tid / 8U, part = tid % 8U;
    QPrep& out = reinterpret_cast<QPrep*>(v.qprep)[r];
    float q[16];
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) q[i] = v.q[(r * kIdxHeads + h) * kIdxDim + 16 * part + i];
    if (part >= 4U) {
        const RopeRow& rope = v.rope[r];
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) rope_pair(q[2 * i], q[2 * i + 1], rope.c[8 * (part - 4U) + i], rope.s[8 * (part - 4U) + i], false);
    }
    float amax = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(q[i]));
    amax = fmaxf(fmaxf(amax, __shfl_xor(amax, 1, 32)), 7.05296657e-38F);
    const int e = log2_ceil(amax * (1.0F / 6.0F));
    const float s = pow2f(e);
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        std::uint32_t packed = 0;
#pragma unroll
        for (unsigned j = 0; j < 4; ++j)
            packed |= (static_cast<std::uint32_t>(e2m1_int2(e2m1_encode(clampf(q[4 * i + j] / s, -6.0F, 6.0F)))) & 255U) << (8U * j);
        out.qv[h][4 * part + i] = static_cast<int>(packed);
    }
    if ((part & 1U) == 0U) out.qs[h][part / 2] = s;
    if (tid < kIdxHeads) out.w[tid] = v.w[r * kIdxHeads + tid] * kIdxWeightScale;
}
// Scores: thread = position; its key decoded once, then every row in turn (each row's arithmetic unchanged). The
// rows' quantised queries are wave-uniform and no global store precedes their reads, so they come in with scalar
// loads (the dot products take them as scalar operands); scores wait in LDS and are stored after every kMaxRows rows
// (calls of up to kMaxCallRows rows: the key is still decoded once).
// With IndexView::candidates (a candidate user) position t is scored for row r only if its block is one of the row's
// candidates, and its key is read only if some row scores it; every other score stays unwritten (select never reads it).
__device__ __forceinline__ void index_scores_body(const IndexView& v, const uint3 bid) {
    DS41_TIMED(3U);
    __shared__ float out_s[kMaxRows][256];
    const unsigned tid = threadIdx.x;
    const std::uint64_t t = static_cast<std::uint64_t>(bid.x) * 256U + tid;
    if (t >= v.max_count) return;
    const std::uint8_t* cand = v.candidates;
    const std::size_t cand_stride = cand != nullptr ? (v.max_count + v.block_size - 1U) / v.block_size : 0U;
    const auto wanted = [&](unsigned r) {
        return t < (v.pos[r] + 1U) / v.ratio && (cand == nullptr || cand[r * cand_stride + t / v.block_size] != 0U);
    };
    if (cand != nullptr) {
        bool any = false;
        for (unsigned r = 0; r < v.rows && !any; ++r) any = wanted(r);
        if (!any) return;
    }
    const QPrep* prep = static_cast<const QPrep*>(v.qprep);
    const std::uint32_t* e = reinterpret_cast<const std::uint32_t*>(v.idx_cache + t * kIdxEntryBytes);
    int kv8[32];
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) {
        const std::uint32_t w = e[i];
        kv8[2 * i] = static_cast<int>(e2m1x4_to_i8(w & 0xFFFFU));
        kv8[2 * i + 1] = static_cast<int>(e2m1x4_to_i8(w >> 16U));
    }
    const std::uint32_t sc = e[16];
    float ks[4];
#pragma unroll
    for (unsigned b = 0; b < 4; ++b) ks[b] = e8m0_value(static_cast<std::uint8_t>(sc >> (8U * b)));
#pragma unroll 1
    for (unsigned r0 = 0; r0 < v.rows; r0 += kMaxRows) {
        const unsigned nr = v.rows - r0 < kMaxRows ? v.rows - r0 : kMaxRows;
#pragma unroll 1
        for (unsigned rr = 0; rr < nr; ++rr) {
            if (cand != nullptr && !wanted(r0 + rr)) continue;
            const QPrep& q = prep[r0 + rr];
            float score = 0.0F;
#pragma unroll 2
            for (unsigned h = 0; h < kIdxHeads; ++h) {
                float term[4];
#pragma unroll
                for (unsigned b = 0; b < 4; ++b) {
                    int dot = 0;
#pragma unroll
                    for (unsigned i = 0; i < 8; ++i) dot = __builtin_amdgcn_sudot4(true, q.qv[h][8 * b + i], true, kv8[8 * b + i], dot, false);
                    term[b] = ((static_cast<float>(dot) * q.qs[h][b]) * ks[b]) * 0.25F;
                }
                const float d = ((term[0] + term[1]) + term[2]) + term[3];
                score = fmaf(fmaxf(d, 0.0F), q.w[h], score);
            }
            out_s[rr][tid] = score;
        }
        for (unsigned rr = 0; rr < nr; ++rr) {
            const unsigned r = r0 + rr;
            if (cand != nullptr ? wanted(r) : t < (v.pos[r] + 1U) / v.ratio) v.scores[r * v.max_count + t] = out_s[rr][tid];
        }
    }
}

// ---- deterministic top-k: radix threshold over order keys, then ordered compaction ----
constexpr unsigned kSelThreads = 1024;
__device__ unsigned block_exclusive_scan(unsigned x, unsigned* warp_tot, unsigned& total) {
    const unsigned lane = threadIdx.x & 31U, w = threadIdx.x >> 5U;
    unsigned incl = x;
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {
        const unsigned y = __shfl_up(incl, static_cast<unsigned>(d), 32);
        if (lane >= static_cast<unsigned>(d)) incl += y;
    }
    if (lane == 31U) warp_tot[w] = incl;
    __syncthreads();
    if (w == 0U) {
        unsigned t = warp_tot[lane], ti = t;
#pragma unroll
        for (int d = 1; d < 32; d *= 2) {
            const unsigned y = __shfl_up(ti, static_cast<unsigned>(d), 32);
            if (lane >= static_cast<unsigned>(d)) ti += y;
        }
        warp_tot[lane] = ti - t;
        if (lane == 31U) warp_tot[32] = ti;
    }
    __syncthreads();
    const unsigned r = warp_tot[w] + incl - x;
    total = warp_tot[32];
    __syncthreads();
    return r;
}
// Threshold T (the k-th largest key) and how many keys equal to T are taken (need); requires n > k. Per-wave
// histograms (integer counts, so the result does not depend on the order of the adds), merged into one total per bin by
// 256 threads in parallel (each sums its bin over the waves); wave 0 scans the 256 totals from the top, 8 per lane,
// with a lane prefix scan. A bin taken whole ends the search early: every key of the bin and above is taken and every
// key below it is not, so T = (the bin's lowest key) - 1 with need 0 selects the same keys (the k-th largest key's
// equals are all in the bin, so the rule "key > T, then the first need keys equal to T" picks them all either way).
constexpr unsigned kSelWaves = kSelThreads / 32;
__device__ void radix_threshold(const std::uint32_t* keys, std::uint32_t n, std::uint32_t k, unsigned (*hist)[256],
    std::uint32_t& thr, std::uint32_t& need) {
    __shared__ std::uint32_t prefix_s, rem_s, done_s, total[256];
    const unsigned lane = threadIdx.x & 31U, w = threadIdx.x >> 5U;
    std::uint32_t prefix = 0, mask = 0, rem = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (unsigned b = lane; b < 256U; b += 32U) hist[w][b] = 0;
        __syncthreads();
        for (std::uint32_t i = threadIdx.x; i < n; i += kSelThreads) {
            const std::uint32_t key = keys[i];
            if ((key & mask) == prefix) atomicAdd(&hist[w][(key >> shift) & 255U], 1U);
        }
        __syncthreads();
        if (threadIdx.x < 256U) {
            unsigned c = 0;
            for (unsigned ww = 0; ww < kSelWaves; ++ww) c += hist[ww][threadIdx.x];
            total[threadIdx.x] = c;
        }
        __syncthreads();
        if (w == 0U) {
            unsigned cnt[8], local = 0;   // lane owns bins 255 - 8 lane - i, i = 0..7 (descending)
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                cnt[i] = total[255U - 8U * lane - i];
                local += cnt[i];
            }
            unsigned incl = local;
#pragma unroll
            for (int d = 1; d < 32; d *= 2) {
                const unsigned y = __shfl_up(incl, static_cast<unsigned>(d), 32);
                if (lane >= static_cast<unsigned>(d)) incl += y;
            }
            const unsigned before = incl - local;
            if (before < rem && incl >= rem) {   // exactly one lane
                unsigned cum = before, i = 0;
                while (cum + cnt[i] < rem) { cum += cnt[i]; ++i; }
                const std::uint32_t low = prefix | ((255U - 8U * lane - i) << shift);
                const bool whole = cnt[i] == rem - cum && low != 0U;
                prefix_s = whole ? low - 1U : low;
                rem_s = whole ? 0U : rem - cum;
                done_s = whole ? 1U : 0U;
            }
        }
        __syncthreads();
        prefix = prefix_s; rem = rem_s;
        const bool done = done_s != 0U;
        mask |= 255U << shift;
        __syncthreads();
        if (done) break;
    }
    thr = prefix; need = rem;
}
// Visits keys in index order; f(i, selected, out_index) for every i < n. One block scan per chunk counts the keys above
// T (low 16 bits) and equal to T (high 16 bits; a chunk has at most kSelThreads of each): the selected keys before i are
// those above T plus the equal ones whose rank among the equals is below need.
template <class F>
__device__ void ordered_select(const std::uint32_t* keys, std::uint32_t n, std::uint32_t thr, std::uint32_t need,
    unsigned* warp_tot, F f) {
    static_assert(kSelThreads < 65536U, "packed counts");
    std::uint32_t eq_carry = 0, out_carry = 0;
    for (std::uint32_t base = 0; base < n; base += kSelThreads) {
        const std::uint32_t i = base + threadIdx.x;
        const std::uint32_t key = i < n ? keys[i] : 0U;
        const unsigned gt = i < n && key > thr ? 1U : 0U, eq = i < n && key == thr ? 1U : 0U;
        unsigned total;
        const unsigned ex = block_exclusive_scan(gt | eq << 16U, warp_tot, total);
        const std::uint32_t eq_rank = eq_carry + (ex >> 16U), taken = min(eq_carry, need);
        const unsigned sel = gt != 0U || (eq != 0U && eq_rank < need) ? 1U : 0U;
        const unsigned out = out_carry + (ex & 0xFFFFU) + (min(eq_rank, need) - taken);
        if (i < n) f(i, sel != 0U, out);
        const std::uint32_t eq_next = eq_carry + (total >> 16U);
        out_carry += (total & 0xFFFFU) + (min(eq_next, need) - taken);
        eq_carry = eq_next;
    }
}

// STRIX_DS41_SELECT_V1=1 (the <true> instantiations of the select kernels): the selection before the parallel histogram merge, the packed scan and
// the whole-bin end (same thresholds, same selected positions in the same order).
__device__ void radix_threshold_v1(const std::uint32_t* keys, std::uint32_t n, std::uint32_t k, unsigned (*hist)[256],
    std::uint32_t& thr, std::uint32_t& need) {
    __shared__ std::uint32_t prefix_s, rem_s;
    const unsigned lane = threadIdx.x & 31U, w = threadIdx.x >> 5U;
    std::uint32_t prefix = 0, mask = 0, rem = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (unsigned b = lane; b < 256U; b += 32U) hist[w][b] = 0;
        __syncthreads();
        for (std::uint32_t i = threadIdx.x; i < n; i += kSelThreads) {
            const std::uint32_t key = keys[i];
            if ((key & mask) == prefix) atomicAdd(&hist[w][(key >> shift) & 255U], 1U);
        }
        __syncthreads();
        if (w == 0U) {
            unsigned cnt[8], local = 0;   // lane owns bins 255 - 8 lane - i, i = 0..7 (descending)
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                const unsigned b = 255U - 8U * lane - i;
                unsigned c = 0;
                for (unsigned ww = 0; ww < kSelWaves; ++ww) c += hist[ww][b];
                cnt[i] = c; local += c;
            }
            unsigned incl = local;
#pragma unroll
            for (int d = 1; d < 32; d *= 2) {
                const unsigned y = __shfl_up(incl, static_cast<unsigned>(d), 32);
                if (lane >= static_cast<unsigned>(d)) incl += y;
            }
            const unsigned before = incl - local;
            if (before < rem && incl >= rem) {   // exactly one lane
                unsigned cum = before, i = 0;
                while (cum + cnt[i] < rem) { cum += cnt[i]; ++i; }
                prefix_s = prefix | ((255U - 8U * lane - i) << shift);
                rem_s = rem - cum;
            }
        }
        __syncthreads();
        prefix = prefix_s; rem = rem_s;
        mask |= 255U << shift;
        __syncthreads();
    }
    thr = prefix; need = rem;
}
// Visits keys in index order; f(i, selected, out_index) for every i < n.
template <class F>
__device__ void ordered_select_v1(const std::uint32_t* keys, std::uint32_t n, std::uint32_t thr, std::uint32_t need,
    unsigned* warp_tot, F f) {
    std::uint32_t eq_carry = 0, out_carry = 0;
    for (std::uint32_t base = 0; base < n; base += kSelThreads) {
        const std::uint32_t i = base + threadIdx.x;
        const std::uint32_t key = i < n ? keys[i] : 0U;
        const unsigned eq = i < n && key == thr ? 1U : 0U;
        unsigned eq_total;
        const unsigned eq_rank = block_exclusive_scan(eq, warp_tot, eq_total) + eq_carry;
        const unsigned sel = i < n && (key > thr || (eq != 0U && eq_rank < need)) ? 1U : 0U;
        unsigned sel_total;
        const unsigned out = block_exclusive_scan(sel, warp_tot, sel_total) + out_carry;
        if (i < n) f(i, sel != 0U, out);
        eq_carry += eq_total; out_carry += sel_total;
    }
}

template <bool Legacy>
__device__ __forceinline__ void select_threshold(const std::uint32_t* keys, std::uint32_t n, std::uint32_t k, unsigned (*hist)[256],
    std::uint32_t& thr, std::uint32_t& need) {
    if constexpr (Legacy) radix_threshold_v1(keys, n, k, hist, thr, need);
    else radix_threshold(keys, n, k, hist, thr, need);
}
template <bool Legacy, class F>
__device__ __forceinline__ void select_ordered(const std::uint32_t* keys, std::uint32_t n, std::uint32_t thr, std::uint32_t need,
    unsigned* warp_tot, F f) {
    if constexpr (Legacy) ordered_select_v1(keys, n, thr, need, warp_tot, f);
    else ordered_select(keys, n, thr, need, warp_tot, f);
}
template <bool Legacy>
__device__ __forceinline__ void select_core(const SelectView& v, const uint3 bid, unsigned (*hist)[256], unsigned* warp_tot) {
    const unsigned r = bid.x;
    const std::uint32_t n = static_cast<std::uint32_t>((v.pos[r] + 1U) / v.ratio);
    const float* scores = v.scores + static_cast<std::size_t>(r) * v.max_count;
    std::uint32_t* keys = v.keys + static_cast<std::size_t>(r) * v.max_count;
    std::uint8_t* cand = v.candidates + static_cast<std::size_t>(r) * ((v.max_count + v.block_size - 1) / v.block_size);
    const float ninf = -__builtin_inff();
    if (v.pool == Pool::candidate_source && n != 0U) {
        const std::uint32_t nb = (n + v.block_size - 1) / v.block_size, last = (n - 1) / v.block_size;
        for (std::uint32_t b = threadIdx.x; b < nb; b += kSelThreads) {
            float m = ninf;
            for (std::uint32_t t = b * v.block_size; t < (b + 1) * v.block_size && t < n; ++t) m = fmaxf(m, scores[t]);
            keys[b] = order_key(b == last ? __builtin_inff() : m);
        }
        __syncthreads();
        const std::uint32_t ninf_key = order_key(ninf);
        if (nb <= v.cand_blocks) {
            for (std::uint32_t b = threadIdx.x; b < nb; b += kSelThreads) cand[b] = keys[b] > ninf_key ? 1U : 0U;
        } else {
            std::uint32_t thr, need;
            select_threshold<Legacy>(keys, nb, v.cand_blocks, hist, thr, need);
            select_ordered<Legacy>(keys, nb, thr, need, warp_tot, [&](std::uint32_t b, bool s, unsigned) {
                cand[b] = s && keys[b] > ninf_key ? 1U : 0U;
            });
        }
        __syncthreads();
    }
    for (std::uint32_t t = threadIdx.x; t < n; t += kSelThreads)
        keys[t] = order_key(v.pool == Pool::candidate_user && cand[t / v.block_size] == 0U ? ninf : scores[t]);
    __syncthreads();
    std::uint32_t* idx = v.idx + static_cast<std::size_t>(r) * v.topk;
    if (n <= v.topk) {
        for (std::uint32_t t = threadIdx.x; t < n; t += kSelThreads) idx[t] = t;
        if (threadIdx.x == 0U) v.idx_count[r] = n;
        return;
    }
    std::uint32_t thr, need;
    select_threshold<Legacy>(keys, n, v.topk, hist, thr, need);
    select_ordered<Legacy>(keys, n, thr, need, warp_tot, [&](std::uint32_t t, bool s, unsigned out) { if (s) idx[out] = t; });
    if (threadIdx.x == 0U) v.idx_count[r] = v.topk;
}
template <bool Legacy>
__device__ __forceinline__ void select_body(const SelectView& v, const uint3 bid) {
    DS41_TIMED(4U);
    __shared__ unsigned hist[kSelWaves][256];
    __shared__ unsigned warp_tot[33];
    select_core<Legacy>(v, bid, hist, warp_tot);
}
// Two-level selection for long rows (max_count > kSelSlice). Level one, one block per slice of 4096 positions: the
// slice's keys in LDS, its own top-k (same rule: key, then the smaller position) written in position order and
// padded to k with key 0 (below every float's key); for the candidate source also the slice's block maxima. Every
// element of the row's top-k is in its slice's top-k (fewer than k elements of the slice beat it), so level two, the
// same selection over the slices' survivors in slice order, gives exactly the row's top-k, in ascending positions.
// Scratch (keys, per row): survivor keys [S x k], survivor positions [S x k], block-maximum keys [nb].
constexpr unsigned kSelSlice = 4096;
struct SelScratch { std::uint32_t *keys, *pos, *bmax; unsigned slices; };
__device__ __forceinline__ SelScratch sel_scratch(const SelectView& v, unsigned r) {
    const unsigned slices = (v.max_count + kSelSlice - 1) / kSelSlice;
    std::uint32_t* base = v.keys + static_cast<std::size_t>(r) * v.max_count;
    return {base, base + static_cast<std::size_t>(slices) * v.topk, base + 2 * static_cast<std::size_t>(slices) * v.topk, slices};
}
// SelectView::compact (candidate user, STRIX_DS41_IDX_CAND_FILTER): the slice keeps only its candidate positions' keys,
// in position order, and selects among them (padded with key 0 to k, as a short slice). Today's slice holds the key of
// -inf at every other position. If every candidate key of the row is above -inf's and the row has more than k candidate
// positions, the row's top-k are candidates only, every one of them is in its slice's candidate top-k, and the merge
// over the survivors (keys, then positions in order) gives today's result. The slice records its candidate count and
// whether a candidate key is at or below -inf's (in the block-maximum scratch, unused by users); the merge checks both and
// otherwise runs today's one-block selection over the whole row.
template <bool Legacy>
__device__ void select_slice_compact(const SelectView& v, unsigned s, std::uint32_t n, std::uint32_t t0, std::uint32_t m,
    const float* scores, const std::uint8_t* cand, const SelScratch& sc, std::uint32_t* skeys, unsigned (*hist)[256], unsigned* warp_tot) {
    __shared__ std::uint32_t boff[kSelSlice / 8U];
    const std::uint32_t bs = v.block_size, b0 = t0 / bs, nsb = (m + bs - 1) / bs, ninf_key = order_key(-__builtin_inff());
    const std::uint32_t b = b0 + threadIdx.x;
    const unsigned len = threadIdx.x < nsb && cand[b] != 0U ? (n - b * bs < bs ? n - b * bs : bs) : 0U;
    unsigned ms;
    const unsigned off = block_exclusive_scan(len, warp_tot, ms);
    if (threadIdx.x < nsb) boff[threadIdx.x] = off;
    bool bad = false;
    for (unsigned j = 0; j < len; ++j) {
        const std::uint32_t key = order_key(scores[b * bs + j]);
        skeys[off + j] = key;
        bad = bad || key <= ninf_key;
    }
    bad = __syncthreads_or(bad ? 1 : 0) != 0;
    if (threadIdx.x == 0U) { sc.bmax[2U * s] = ms; sc.bmax[2U * s + 1U] = bad ? 1U : 0U; }
    if (n <= v.topk) return;
    const auto position = [&](std::uint32_t i) {   // the slice block holding compacted index i: the last with boff <= i
        std::uint32_t lo = 0, hi = nsb;
        while (hi - lo > 1U) { const std::uint32_t mid = (lo + hi) / 2U; if (boff[mid] <= i) lo = mid; else hi = mid; }
        return (b0 + lo) * bs + (i - boff[lo]);
    };
    std::uint32_t* ok = sc.keys + static_cast<std::size_t>(s) * v.topk;
    std::uint32_t* op = sc.pos + static_cast<std::size_t>(s) * v.topk;
    if (ms <= v.topk) {
        for (std::uint32_t i = threadIdx.x; i < v.topk; i += kSelThreads) { ok[i] = i < ms ? skeys[i] : 0U; op[i] = i < ms ? position(i) : t0; }
        return;
    }
    std::uint32_t thr, need;
    select_threshold<Legacy>(skeys, ms, v.topk, hist, thr, need);
    select_ordered<Legacy>(skeys, ms, thr, need, warp_tot, [&](std::uint32_t i, bool sel, unsigned out) {
        if (sel) { ok[out] = skeys[i]; op[out] = position(i); }
    });
}
template <bool Legacy>
__device__ __forceinline__ void select_slice_body(const SelectView& v, const uint3 bid) {
    DS41_TIMED(4U);
    __shared__ unsigned hist[kSelWaves][256];
    __shared__ unsigned warp_tot[33];
    __shared__ std::uint32_t skeys[kSelSlice];
    const unsigned s = bid.x, r = bid.y;
    const std::uint32_t n = static_cast<std::uint32_t>((v.pos[r] + 1U) / v.ratio), t0 = s * kSelSlice;
    if (t0 >= n) return;
    const std::uint32_t m = n - t0 < kSelSlice ? n - t0 : kSelSlice;
    const float* scores = v.scores + static_cast<std::size_t>(r) * v.max_count;
    const std::uint8_t* cand = v.candidates + static_cast<std::size_t>(r) * ((v.max_count + v.block_size - 1) / v.block_size);
    const SelScratch sc = sel_scratch(v, r);
    const float ninf = -__builtin_inff();
    if (v.compact && v.pool == Pool::candidate_user) {
        select_slice_compact<Legacy>(v, s, n, t0, m, scores, cand, sc, skeys, hist, warp_tot);
        return;
    }
    for (std::uint32_t i = threadIdx.x; i < m; i += kSelThreads) {
        const std::uint32_t t = t0 + i;
        skeys[i] = order_key(v.pool == Pool::candidate_user && cand[t / v.block_size] == 0U ? ninf : scores[t]);
    }
    if (v.pool == Pool::candidate_source) {
        const std::uint32_t b0 = t0 / v.block_size, b1 = (t0 + m + v.block_size - 1) / v.block_size, last = (n - 1) / v.block_size;
        for (std::uint32_t b = b0 + threadIdx.x; b < b1; b += kSelThreads) {
            float mx = ninf;
            for (std::uint32_t t = b * v.block_size; t < (b + 1) * v.block_size && t < n; ++t) mx = fmaxf(mx, scores[t]);
            sc.bmax[b] = order_key(b == last ? __builtin_inff() : mx);
        }
    }
    __syncthreads();
    if (n <= v.topk) return;   // level two takes every position
    std::uint32_t* ok = sc.keys + static_cast<std::size_t>(s) * v.topk;
    std::uint32_t* op = sc.pos + static_cast<std::size_t>(s) * v.topk;
    if (m <= v.topk) {
        for (std::uint32_t i = threadIdx.x; i < v.topk; i += kSelThreads) { ok[i] = i < m ? skeys[i] : 0U; op[i] = t0 + i; }
        return;
    }
    std::uint32_t thr, need;
    select_threshold<Legacy>(skeys, m, v.topk, hist, thr, need);
    select_ordered<Legacy>(skeys, m, thr, need, warp_tot, [&](std::uint32_t i, bool sel, unsigned out) {
        if (sel) { ok[out] = skeys[i]; op[out] = t0 + i; }
    });
}
// Level two: block 0 of a row selects the top-k among the survivors; block 1 (candidate source only) the candidate
// blocks among the block maxima.
template <bool Legacy>
__device__ __forceinline__ void select_merge_body(const SelectView& v, const uint3 bid) {
    DS41_TIMED(4U);
    __shared__ unsigned hist[kSelWaves][256];
    __shared__ unsigned warp_tot[33];
    const unsigned r = bid.y;
    const std::uint32_t n = static_cast<std::uint32_t>((v.pos[r] + 1U) / v.ratio);
    const SelScratch sc = sel_scratch(v, r);
    if (bid.x == 1U) {
        if (n == 0U) return;
        std::uint8_t* cand = v.candidates + static_cast<std::size_t>(r) * ((v.max_count + v.block_size - 1) / v.block_size);
        const std::uint32_t nb = (n + v.block_size - 1) / v.block_size, ninf_key = order_key(-__builtin_inff());
        if (nb <= v.cand_blocks) {
            for (std::uint32_t b = threadIdx.x; b < nb; b += kSelThreads) cand[b] = sc.bmax[b] > ninf_key ? 1U : 0U;
            return;
        }
        std::uint32_t thr, need;
        select_threshold<Legacy>(sc.bmax, nb, v.cand_blocks, hist, thr, need);
        select_ordered<Legacy>(sc.bmax, nb, thr, need, warp_tot, [&](std::uint32_t b, bool s, unsigned) {
            cand[b] = s && sc.bmax[b] > ninf_key ? 1U : 0U;
        });
        return;
    }
    std::uint32_t* idx = v.idx + static_cast<std::size_t>(r) * v.topk;
    if (n <= v.topk) {
        for (std::uint32_t t = threadIdx.x; t < n; t += kSelThreads) idx[t] = t;
        if (threadIdx.x == 0U) v.idx_count[r] = n;
        return;
    }
    if (v.compact && v.pool == Pool::candidate_user) {   // the compact slices' records (select_slice_compact)
        std::uint32_t cnt = 0, bad = 0;
        for (std::uint32_t k = threadIdx.x; k < (n + kSelSlice - 1) / kSelSlice; k += kSelThreads) { cnt += sc.bmax[2U * k]; bad |= sc.bmax[2U * k + 1U]; }
        unsigned all_cnt;
        (void)block_exclusive_scan(cnt, warp_tot, all_cnt);
        if (__syncthreads_or(static_cast<int>(bad)) != 0 || all_cnt <= v.topk) {
            select_core<Legacy>(v, make_uint3(r, 0U, 0U), hist, warp_tot);
            return;
        }
    }
    const std::uint32_t total = ((n + kSelSlice - 1) / kSelSlice) * v.topk;
    std::uint32_t thr, need;
    select_threshold<Legacy>(sc.keys, total, v.topk, hist, thr, need);
    select_ordered<Legacy>(sc.keys, total, thr, need, warp_tot, [&](std::uint32_t i, bool s, unsigned out) { if (s) idx[out] = sc.pos[i]; });
    if (threadIdx.x == 0U) v.idx_count[r] = v.topk;
}

// Every row's count <= topk: the selection of select_body without its scores (every position, in order) and, for the
// candidate source, its pool: with nb <= cand_blocks select_body takes every block whose maximum is above -inf, and the
// last block always (its key is +inf); a block's maximum of finite scores is finite, so every block of the row.
__device__ __forceinline__ void select_direct_body(const SelectView& v, const uint3 bid) {
    DS41_TIMED(4U);
    const unsigned r = bid.x;
    const std::uint32_t n = static_cast<std::uint32_t>((v.pos[r] + 1U) / v.ratio);
    std::uint32_t* idx = v.idx + static_cast<std::size_t>(r) * v.topk;
    for (std::uint32_t t = threadIdx.x; t < n; t += blockDim.x) idx[t] = t;
    if (threadIdx.x == 0U) v.idx_count[r] = n;
    if (v.pool != Pool::candidate_source) return;
    std::uint8_t* cand = v.candidates + static_cast<std::size_t>(r) * ((v.max_count + v.block_size - 1) / v.block_size);
    const std::uint32_t nb = (n + v.block_size - 1) / v.block_size;
    for (std::uint32_t b = threadIdx.x; b < nb; b += blockDim.x) cand[b] = 1U;
}

// ---- attention: one block per (key chunk of 64, 8 Hpw heads, row); 8 waves of Hpw heads; lane = 16 dims of each head ----
// Tiles of 16 keys: the block dequantises the tile once into LDS (thread -> key tid/16, dims 32 (tid%16) .. +31) while
// the next tile's bytes are in flight. Scores of 8 keys at a time: the lane fmaf chains, then a transposed xor
// butterfly (DPP and permlanex16, no LDS) that leaves key lane/4 in every lane after exactly the additions of the
// per-key butterfly, so each softmax term is evaluated once. The last block of a (row, head group) combines.
// Heads per wave Hpw (1, 2 or 4; 8 waves per block, so 8 Hpw heads per block) is chosen per call so that the grid has
// enough blocks (enqueue_attention); every choice computes each head identically.
constexpr unsigned kAttnThreads = 256, kAttnMinBlockHeads = kAttnThreads / 32U, kAttnGroups = kHeads / kAttnMinBlockHeads;
static_assert(kAttnGroups == kAttnGroupCount);

struct KeyRaw { uint4 c0, c1; std::uint32_t sc; };
// One key's dims 32 part .. +31 into its LDS row (zeros for a key past the chunk's end). Row layout: float4 i of lane l
// (dims 16 l + 4 i .. +3) at i * 32 + l, so the wave's reads of one float4 index are conflict-free.
__device__ __forceinline__ unsigned key_slot(unsigned part, unsigned i) { return (i % 4U) * 32U + 2U * part + i / 4U; }
// Involutive permutation of 16-byte vectors, applied to both stores and reads.
template <bool Swizzle>
__device__ __forceinline__ unsigned lds_key_slot(unsigned slot) { return Swizzle ? slot ^ ((slot >> 3U) & 1U) : slot; }
// STRIX_DS41_ATTN_TILE_F16: the tile as f16 pairs (dim 2 t in the low half). Row layout: 16-byte vector h of lane l
// (dims 16 l + 8 h .. +7) at h * 32 + l. A dequantised key value is E4M3 x 2^k (4 significant bits) or E2M1 x E4M3
// (at most 6), so it is exact in f16 unless its scale is out of range (checked per key, see store_key_f16); the store reports any value
// that is not (or is NaN) and the block then stores that tile as today's f32 tile and runs today's tile code. The f16
// operand reaches the f32 FMA by an exact conversion (folded into v_fma_mix_f32, one rounding as fmaf), so the dot
// products see exactly the floats of today's f32 tile, without unpack instructions.
// Writes the f16 pairs of the part's 32 values (today's f32 values, packed; exact when the check passes, so the
// truncating pack keeps them); returns whether they are exact. Compressed: E2M1 x E4M3 is at most 2688 in steps of at
// least 2^-10 (exact unless the scale is NaN). Window: E4M3 x 2^(b - 127) is exact for E8M0 b in [112, 134] (at most
// 448 x 2^7 <= 65504, steps of at least 2^-24) unless a code is NaN.
__device__ __forceinline__ bool store_key_f16(uint4* row, unsigned part, const KeyRaw& k, bool win, bool valid) {
    const std::uint32_t cw[8] = {k.c0.x, k.c0.y, k.c0.z, k.c0.w, k.c1.x, k.c1.y, k.c1.z, k.c1.w};
    float x[32];
    bool exact = true;
    if (!valid) {
#pragma unroll
        for (unsigned i = 0; i < 32; ++i) x[i] = 0.0F;
    } else if (win) {
        const std::uint32_t b = k.sc & 0xFFU;
        std::uint32_t nan = 0;
        const float sv = e8m0_value(static_cast<std::uint8_t>(b));
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
            const std::uint32_t w = cw[i];
            nan |= ((w & 0x7F7F7F7FU) + 0x01010101U) & 0x80808080U;   // a byte with low 7 bits 0x7F (E4M3 NaN)
#pragma unroll
            for (unsigned t = 0; t < 4; ++t) x[4 * i + t] = e4m3_value(static_cast<std::uint8_t>(w >> (8U * t))) * sv;
        }
        exact = b >= 112U && b <= 134U && nan == 0U;
    } else {
        exact = ((k.sc & 0x7FU) != 0x7FU) && (((k.sc >> 8U) & 0x7FU) != 0x7FU);
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
            const std::uint32_t w = cw[i / 2] >> (16U * (i % 2));
            const float sv = e4m3_value(static_cast<std::uint8_t>(k.sc >> (8U * (i / 4))));
#pragma unroll
            for (unsigned t = 0; t < 4; ++t) x[4 * i + t] = e2m1_value(static_cast<std::uint8_t>((w >> (4U * t)) & 15U)) * sv;
        }
    }
#pragma unroll
    for (unsigned v = 0; v < 4; ++v) {   // vector v: lane 2 part + v / 2, half v % 2 = dims 8 v .. 8 v + 7 of the part
        std::uint32_t w[4];
#pragma unroll
        for (unsigned t = 0; t < 4; ++t)
            w[t] = __builtin_bit_cast(std::uint32_t, __builtin_amdgcn_cvt_pkrtz(x[8 * v + 2 * t], x[8 * v + 2 * t + 1]));
        row[(v % 2U) * 32U + 2U * part + v / 2U] = make_uint4(w[0], w[1], w[2], w[3]);
    }
    return exact;
}
template <bool Swizzle>
__device__ __forceinline__ void store_key(float4* row, unsigned part, const KeyRaw& k, bool win, bool valid) {
    const std::uint32_t cw[8] = {k.c0.x, k.c0.y, k.c0.z, k.c0.w, k.c1.x, k.c1.y, k.c1.z, k.c1.w};
    if (!valid) {
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) row[lds_key_slot<Swizzle>(key_slot(part, i))] = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    } else if (win) {
        const float s = e8m0_value(static_cast<std::uint8_t>(k.sc));
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
            const std::uint32_t w = cw[i];
            row[lds_key_slot<Swizzle>(key_slot(part, i))] = make_float4(e4m3_value(static_cast<std::uint8_t>(w)) * s, e4m3_value(static_cast<std::uint8_t>(w >> 8U)) * s,
                                 e4m3_value(static_cast<std::uint8_t>(w >> 16U)) * s, e4m3_value(static_cast<std::uint8_t>(w >> 24U)) * s);
        }
    } else {
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
            const std::uint32_t w = cw[i / 2] >> (16U * (i % 2));
            const float s = e4m3_value(static_cast<std::uint8_t>(k.sc >> (8U * (i / 4))));
            row[lds_key_slot<Swizzle>(key_slot(part, i))] = make_float4(e2m1_value(static_cast<std::uint8_t>(w & 15U)) * s, e2m1_value(static_cast<std::uint8_t>((w >> 4U) & 15U)) * s,
                                 e2m1_value(static_cast<std::uint8_t>((w >> 8U) & 15U)) * s, e2m1_value(static_cast<std::uint8_t>((w >> 12U) & 15U)) * s);
        }
    }
}
// STRIX_DS41_ATTN_LEAN: store_key's values through f16 bit patterns. An E4M3 code's bits (c & 0x7F) << 7 | sign << 15 read
// as f16 are exactly e4m3(c) x 2^-8 (normal and subnormal codes alike), an E2M1 code's (n & 7) << 9 | sign << 15 exactly
// e2m1(n) x 2^-14; one fma_mix(f16, S, -0) with S = the scale x 2^8 (resp. x 2^14), exact for an E8M0 byte <= 246 (resp. a
// non-NaN E4M3 scale), rounds the same real product once, as e4m3(c) * s does (-0 addend: zero signs kept). A part with a
// NaN code or another scale takes store_key's own code.
__device__ __forceinline__ float f16_scaled(std::uint32_t bits16, float s) {
    return __builtin_fmaf(static_cast<float>(__builtin_bit_cast(_Float16, static_cast<std::uint16_t>(bits16))), s, -0.0F);
}
// Lean 3: the E4M3 halves as x + (x & 0x4000) with x = byte << 7 (the sign lands on bit 14 and the add carries it to
// bit 15): the same bits as (c & 0x7F) << 7 | sign << 15 in one instruction less per pair.
// E2M1: the f16 high byte of each nibble, (n & 8) << 4 | (n & 7) << 1, for all 8 nibbles of a word in two masks and
// one byte permutation per byte (no byte extraction, no integer multiply): the same halves.
template <bool Swizzle, bool Lean3 = false>
__device__ __forceinline__ void store_key_lean(float4* row, unsigned part, const KeyRaw& k, bool win, bool valid) {
    const std::uint32_t cw[8] = {k.c0.x, k.c0.y, k.c0.z, k.c0.w, k.c1.x, k.c1.y, k.c1.z, k.c1.w};
    if (valid && win) {
        const std::uint32_t b = k.sc & 0xFFU;
        std::uint32_t nan = 0;
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) nan |= ((cw[i] & 0x7F7F7F7FU) + 0x01010101U) & 0x80808080U;
        if (b <= 246U && nan == 0U) {
            const float sv = __builtin_bit_cast(float, (b + 8U) << 23U);   // 2^(b - 127) x 2^8
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                const std::uint32_t w = cw[i];
                std::uint32_t h[2];
#pragma unroll
                for (unsigned t = 0; t < 2; ++t) {   // bytes 2t, 2t + 1 -> f16 halves
                    const std::uint32_t pz = __builtin_amdgcn_perm(0U, w, t == 0U ? 0x0C010C00U : 0x0C030C02U);
                    if constexpr (Lean3) {
                        const std::uint32_t x = pz << 7U;
                        h[t] = x + (x & 0x40004000U);
                    } else {
                        h[t] = ((pz & 0x007F007FU) << 7U) | ((pz & 0x00800080U) << 8U);
                    }
                }
                row[lds_key_slot<Swizzle>(key_slot(part, i))] = make_float4(f16_scaled(h[0], sv), f16_scaled(h[0] >> 16U, sv),
                                                                            f16_scaled(h[1], sv), f16_scaled(h[1] >> 16U, sv));
            }
            return;
        }
    } else if (valid && ((k.sc & 0x7FU) != 0x7FU) && (((k.sc >> 8U) & 0x7FU) != 0x7FU)) {
        float sv[2];
#pragma unroll
        for (unsigned g = 0; g < 2; ++g) sv[g] = e4m3_value(static_cast<std::uint8_t>(k.sc >> (8U * g))) * 16384.0F;   // exact
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
            const std::uint32_t w = cw[i / 2] >> (16U * (i % 2));
            std::uint32_t h[2];
#pragma unroll
            for (unsigned t = 0; t < 2; ++t) {   // byte t of the 16 bits: its low nibble -> low half, high nibble -> high half
                if constexpr (Lean3) {
                    // the f16 high bytes of every nibble of the word at once, (n & 8) << 4 | (n & 7) << 1 (low nibbles
                    // in xl, high nibbles in xh), then one byte permutation per byte: the same halves as below
                    const std::uint32_t c = cw[i / 2];
                    const std::uint32_t xl = ((c << 4U) & 0x80808080U) | ((c << 1U) & 0x0E0E0E0EU);
                    const std::uint32_t xh = (c & 0x80808080U) | ((c >> 3U) & 0x0E0E0E0EU);
                    const std::uint32_t kb = 2U * (i % 2U) + t;   // byte of the word
                    h[t] = __builtin_amdgcn_perm(xh, xl, 0x000C000CU | (kb << 8U) | ((4U + kb) << 24U));
                } else {
                    const std::uint32_t by = (w >> (8U * t)) & 0xFFU, pz = (by | (by << 12U)) & 0x000F000FU;
                    h[t] = ((pz & 0x00070007U) << 9U) | ((pz & 0x00080008U) << 12U);
                }
            }
            const float s = sv[i / 4];
            row[lds_key_slot<Swizzle>(key_slot(part, i))] = make_float4(f16_scaled(h[0], s), f16_scaled(h[0] >> 16U, s),
                                                                        f16_scaled(h[1], s), f16_scaled(h[1] >> 16U, s));
        }
        return;
    }
    store_key<Swizzle>(row, part, k, win, valid);
}
__device__ __forceinline__ float lane_value(float x, unsigned src) {
    return __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, x), static_cast<int>(src)));
}
// The lane's 16 dims of tile key j (dims 16 lane .. +15, in order) from the f32 tile or the f16 tile.
template <bool Swizzle, bool Half>
__device__ __forceinline__ void tile_key(const void* tile, unsigned j, unsigned lane, float (&k)[16]) {
    if constexpr (Half) {
        const auto* kb = static_cast<const uint4 (*)[kHeadDim / 8]>(tile);
#pragma unroll
        for (unsigned h = 0; h < 2; ++h) {
            const uint4 u = kb[j][32U * h + lane];
            const std::uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (unsigned t = 0; t < 4; ++t) {   // exact widening, folded into the consumers' FMAs (v_fma_mix_f32)
                k[8 * h + 2 * t] = static_cast<float>(__builtin_bit_cast(_Float16, static_cast<std::uint16_t>(w[t])));
                k[8 * h + 2 * t + 1] = static_cast<float>(__builtin_bit_cast(_Float16, static_cast<std::uint16_t>(w[t] >> 16U)));
            }
        }
    } else {
        const auto* ks = static_cast<const float4 (*)[kHeadDim / 4]>(tile);
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const float4 kk = ks[j][lds_key_slot<Swizzle>(32 * i + lane)];
            k[4 * i] = kk.x; k[4 * i + 1] = kk.y; k[4 * i + 2] = kk.z; k[4 * i + 3] = kk.w;
        }
    }
}
// One tile's online-softmax step for the wave's Hpw heads over its first cnt keys (loops over keys kept rolled: the
// kernel is instruction-fetch bound when fully unrolled).
// Lean (STRIX_DS41_ATTN_LEAN, default 3, =0 off; 2 also stores the tile through store_key_lean; 3: see the Lean 3 notes below): the same operations in fewer instructions: exp_f's range tests as
// selects over the same polynomial (exp_f_sel), the running sum l[h] = l[h] + p_j taken in the PV loop (same j order,
// after the same l[h] * sc) instead of a second loop over the tile's keys, and the PV loop split at key 8 so p_j is
// read from pr[h][0] / pr[h][1] without a per-key select.
template <unsigned kAttnHpw, bool Swizzle, bool Half = false, bool Narrow = Half, unsigned Lean = 0>
__device__ __forceinline__ void attention_tile(const void* tile, unsigned cnt, unsigned lane,
    const float (&q)[kAttnHpw][16], float (&m)[kAttnHpw], float (&l)[kAttnHpw], float (&acc)[kAttnHpw][16], [[maybe_unused]] bool mark = false) {
    if (mark) DS41_CMARK(0U);   // timing build: the first tile's phases (QK + butterfly, softmax, PV)
    const float ninf = -__builtin_inff();
    const bool b4 = (lane & 16U) != 0U, b3 = (lane & 8U) != 0U, b2 = (lane & 4U) != 0U;
    float s[kAttnHpw][2];
    if constexpr (Lean >= 3U && !Half) {
        // Lean 3: lane slot jj of each half holds key jj ^ (lane bits 4..2) (each lane's own fmaf chains, unchanged), so
        // every stage of the transposed butterfly keeps slots 0.. and sends the upper slots without selects: at each
        // stage the lane and its partner hold the same key in the kept / sent slot and the sum is the same
        // own + partner's of the per-key butterfly; slot 0 ends with key lane / 4, as before.
        const char* tb = static_cast<const char*>(tile) + 16U * lds_key_slot<Swizzle>(lane);   // + 2048 key + 512 i
        const unsigned kx = ((lane >> 2U) & 7U) << 11U;
#pragma unroll 1
        for (unsigned sb = 0; sb < 2; ++sb) {
            float pt[kAttnHpw][8];
            const unsigned ka = kx + 16384U * sb;
            if constexpr (kAttnHpw == 4U) {
                // 4 heads per wave: the same chains as one asm block with fixed registers, every FMA pair dual-issued
                // (tools/gen_ds41_qk_asm.py); the tile's LDS address as a 32-bit LDS pointer
                using f32x16 = float __attribute__((ext_vector_type(16)));
                using f32x32 = float __attribute__((ext_vector_type(32)));
                f32x16 qv[4];
#pragma unroll
                for (unsigned h = 0; h < 4; ++h)
#pragma unroll
                    for (unsigned i = 0; i < 16; ++i) qv[h][i] = q[h][i];
                using lds_cp = const __attribute__((address_space(3))) char*;
                const auto tl = static_cast<unsigned>(reinterpret_cast<std::uintptr_t>((lds_cp)tb));
                f32x32 o;
                asm volatile(DS41_QK_ASM_TEXT : "=&{v[102:133]}"(o) : "v"(ka), "v"(tl), "{v[0:15]}"(qv[0]), "{v[17:32]}"(qv[1]), "{v[34:49]}"(qv[2]),
                             "{v[51:66]}"(qv[3]) : DS41_QK_ASM_CLOBBERS, "memory");
#pragma unroll
                for (unsigned h = 0; h < 4; ++h)
#pragma unroll
                    for (unsigned jj = 0; jj < 8; ++jj) pt[h][jj] = o[8 * (jj / 2) + 2 * h + jj % 2];
            } else
#pragma unroll
            for (unsigned jj = 0; jj < 8; ++jj) {
                const float4* kp = reinterpret_cast<const float4*>(tb + (ka ^ (jj << 11U)));
                float k[16];
#pragma unroll
                for (unsigned i = 0; i < 4; ++i) {
                    const float4 kk = kp[32U * i];
                    k[4 * i] = kk.x; k[4 * i + 1] = kk.y; k[4 * i + 2] = kk.z; k[4 * i + 3] = kk.w;
                }
#pragma unroll
                for (unsigned h = 0; h < kAttnHpw; ++h) {
                    float t = 0.0F;
#pragma unroll
                    for (unsigned i = 0; i < 16; ++i) t = fmaf(q[h][i], k[i], t);
                    pt[h][jj] = t;
                }
            }
#pragma unroll
            for (unsigned h = 0; h < kAttnHpw; ++h) {
#pragma unroll
                for (unsigned i = 0; i < 4; ++i) pt[h][i] = pt[h][i] + xor_lane_nt<16>(pt[h][i + 4]);
#pragma unroll
                for (unsigned i = 0; i < 2; ++i) pt[h][i] = pt[h][i] + xor_lane_nt<8>(pt[h][i + 2]);
                pt[h][0] = pt[h][0] + xor_lane_nt<4>(pt[h][1]);
                float t = pt[h][0];
                t = t + xor_lane_nt<2>(t);
                t = t + xor_lane_nt<1>(t);
                const float sv = 8U * sb + lane / 4U < cnt ? t * kAttnScale : ninf;
                if (sb == 0U) s[h][0] = sv; else s[h][1] = sv;
            }
        }
    } else {
#pragma unroll 1
    for (unsigned sb = 0; sb < 2; ++sb) {
        float pt[kAttnHpw][8];
        // the f16 kernels at 4 heads per wave (both of their tile forms): 2 keys per unrolled step (8 spill); same order
        constexpr unsigned kKeyUnroll = Narrow && kAttnHpw == 4U ? 2U : 8U;
#pragma unroll kKeyUnroll
        for (unsigned jj = 0; jj < 8; ++jj) {
            float k[16];
            tile_key<Swizzle, Half>(tile, 8 * sb + jj, lane, k);
#pragma unroll
            for (unsigned h = 0; h < kAttnHpw; ++h) {
                float t = 0.0F;
#pragma unroll
                for (unsigned i = 0; i < 16; ++i) t = fmaf(q[h][i], k[i], t);
                pt[h][jj] = t;
            }
        }
        // transposed butterfly: at xor 16 / 8 / 4 the lane keeps the half of its keys selected by lane bit 4 / 3 / 2
        // and adds the partner's partial of the same key (v + v^d as in the per-key butterfly); then xor 2 and 1
#pragma unroll
        for (unsigned h = 0; h < kAttnHpw; ++h) {
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                const float keep = b4 ? pt[h][i + 4] : pt[h][i], send = b4 ? pt[h][i] : pt[h][i + 4];
                pt[h][i] = keep + xor_lane<16>(send);
            }
#pragma unroll
            for (unsigned i = 0; i < 2; ++i) {
                const float keep = b3 ? pt[h][i + 2] : pt[h][i], send = b3 ? pt[h][i] : pt[h][i + 2];
                pt[h][i] = keep + xor_lane<8>(send);
            }
            {
                const float keep = b2 ? pt[h][1] : pt[h][0], send = b2 ? pt[h][0] : pt[h][1];
                pt[h][0] = keep + xor_lane<4>(send);
            }
            float t = pt[h][0];
            t = t + xor_lane<2>(t);
            t = t + xor_lane<1>(t);
            const float sv = 8U * sb + lane / 4U < cnt ? t * kAttnScale : ninf;
            if (sb == 0U) s[h][0] = sv; else s[h][1] = sv;
        }
    }
    }
    if (mark) DS41_CMARK(1U);
    float pr[kAttnHpw][2];
    if constexpr (Lean) {
        // Lean 3, 4 heads per wave: the probability of head h for key j is evaluated in lane 4 j + h instead of every lane
        // of the quad, from quad lane 0's score and running max (the floats lane 4 j uses in the previous form): 2
        // exp_f_sel per lane instead of 8, read back from lane 4 j + h.
        constexpr bool kSpread = Lean >= 3U && kAttnHpw == 4U;
        [[maybe_unused]] float mnv[kAttnHpw];
#pragma unroll
        for (unsigned h = 0; h < kAttnHpw; ++h) {
            float tm = fmaxf(s[h][0], s[h][1]);
            if constexpr (Lean >= 3U) {
                tm = fmaxf(tm, xor_lane_nt<4>(tm));
                tm = fmaxf(tm, xor_lane_nt<8>(tm));
                tm = fmaxf(tm, xor_lane_nt<16>(tm));
            } else {
                tm = fmaxf(tm, xor_lane<4>(tm));
                tm = fmaxf(tm, xor_lane<8>(tm));
                tm = fmaxf(tm, xor_lane<16>(tm));
            }
            const float mn = fmaxf(m[h], tm);
            mnv[h] = mn;
            if constexpr (!kSpread) {
                pr[h][0] = exp_f_sel(s[h][0] - mn);
                pr[h][1] = exp_f_sel(s[h][1] - mn);
            }
            // Lean 3: m - mn is the same in every lane (m and the tile max are wave-wide); when it is +0 (the running
            // max did not grow) sc = exp_f_sel(+0) is exactly 1 and l * 1, acc * 1 are l, acc: the rescale is skipped
            const float d = m[h] - mn;
            if (Lean < 3U || __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, d)) != 0) {
                const float sc = exp_f_sel(d);
                l[h] = l[h] * sc;
#pragma unroll
                for (unsigned i = 0; i < 16; ++i) acc[h][i] = acc[h][i] * sc;
            }
            m[h] = mn;
        }
        if constexpr (kSpread) {
            const unsigned r = lane & 3U;
            float mq[4], sq[4][2];
#pragma unroll
            for (unsigned h = 0; h < 4; ++h) { mq[h] = quad_first(mnv[h]); sq[h][0] = quad_first(s[h][0]); sq[h][1] = quad_first(s[h][1]); }
            const float mr = r == 0U ? mq[0] : r == 1U ? mq[1] : r == 2U ? mq[2] : mq[3];
#pragma unroll
            for (unsigned x = 0; x < 2; ++x) {
                const float sr = r == 0U ? sq[0][x] : r == 1U ? sq[1][x] : r == 2U ? sq[2][x] : sq[3][x];
                pr[0][x] = exp_f_sel(sr - mr);
            }
        }
        if (mark) DS41_CMARK(2U);
        const unsigned c0 = cnt < 8U ? cnt : 8U;
        // Lean 3, f32 tile: one address per key (the lane's 4 vectors at + 512 i)
        const auto pv_key = [&](unsigned j, float (&k)[16]) {
            if constexpr (Lean >= 3U && !Half) {
                const float4* kp = reinterpret_cast<const float4*>(static_cast<const char*>(tile) + 16U * lds_key_slot<Swizzle>(lane) + 2048U * j);
#pragma unroll
                for (unsigned i = 0; i < 4; ++i) {
                    const float4 kk = kp[32U * i];
                    k[4 * i] = kk.x; k[4 * i + 1] = kk.y; k[4 * i + 2] = kk.z; k[4 * i + 3] = kk.w;
                }
            } else {
                tile_key<Swizzle, Half>(tile, j, lane, k);
            }
        };
#pragma unroll 1
        for (unsigned j = 0; j < c0; ++j) {
            float k[16];
            pv_key(j, k);
#pragma unroll
            for (unsigned h = 0; h < kAttnHpw; ++h) {
                const float pj = kSpread ? lane_value(pr[0][0], 4U * j + h) : lane_value(pr[h][0], 4U * j);
                l[h] = l[h] + pj;
#pragma unroll
                for (unsigned i = 0; i < 16; ++i) acc[h][i] = fmaf(pj, k[i], acc[h][i]);
            }
        }
#pragma unroll 1
        for (unsigned j = 8; j < cnt; ++j) {
            float k[16];
            pv_key(j, k);
#pragma unroll
            for (unsigned h = 0; h < kAttnHpw; ++h) {
                const float pj = kSpread ? lane_value(pr[0][1], 4U * (j - 8U) + h) : lane_value(pr[h][1], 4U * (j - 8U));
                l[h] = l[h] + pj;
#pragma unroll
                for (unsigned i = 0; i < 16; ++i) acc[h][i] = fmaf(pj, k[i], acc[h][i]);
            }
        }
        if (mark) DS41_CMARK(3U);
        return;
    }
#pragma unroll
    for (unsigned h = 0; h < kAttnHpw; ++h) {
        float tm = fmaxf(s[h][0], s[h][1]);   // max is exact: any order gives the tile max
        tm = fmaxf(tm, xor_lane<4>(tm));
        tm = fmaxf(tm, xor_lane<8>(tm));
        tm = fmaxf(tm, xor_lane<16>(tm));
        const float mn = fmaxf(m[h], tm), sc = exp_f(m[h] - mn);
        pr[h][0] = exp_f(s[h][0] - mn);
        pr[h][1] = exp_f(s[h][1] - mn);
        l[h] = l[h] * sc;
#pragma unroll 1
        for (unsigned j = 0; j < cnt; ++j) l[h] = l[h] + lane_value(j < 8U ? pr[h][0] : pr[h][1], 4U * (j % 8));
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) acc[h][i] = acc[h][i] * sc;
        m[h] = mn;
    }
    if (mark) DS41_CMARK(2U);
#pragma unroll 1
    for (unsigned j = 0; j < cnt; ++j) {
        float k[16];
        tile_key<Swizzle, Half>(tile, j, lane, k);
#pragma unroll
        for (unsigned h = 0; h < kAttnHpw; ++h) {
            const float pj = lane_value(j < 8U ? pr[h][0] : pr[h][1], 4U * (j % 8));
#pragma unroll
            for (unsigned i = 0; i < 16; ++i) acc[h][i] = fmaf(pj, k[i], acc[h][i]);
        }
    }
    if (mark) DS41_CMARK(3U);
}
// Combine of one (row, head) in chunk order, sink, 1 / L, inverse RoPE. Sel (STRIX_DS41_ATTN_LEAN 3): exp_f_sel, the same
// polynomial with the range tests as selects.
template <bool Sel = false>
__device__ __forceinline__ void attention_combine_head(const AttentionView& v, unsigned r, unsigned h, unsigned lane, unsigned chunks) {
    const float* part = v.partial + (static_cast<std::size_t>(r) * v.heads + h) * kMaxChunks * (2 + kHeadDim);
    // every chunk's (m, l) and the lane's 16 accumulators are loaded at once (one memory round trip), then combined
    // in chunk order
    float mc[kMaxChunks], lc[kMaxChunks], pa[kMaxChunks][16], mm = -__builtin_inff();
#pragma unroll
    for (unsigned c = 0; c < kMaxChunks; ++c) {
        if (c < chunks) {
            const float* pc = part + c * (2 + kHeadDim);
            mc[c] = pc[0]; lc[c] = pc[1];
#pragma unroll
            for (unsigned i = 0; i < 16; ++i) pa[c][i] = pc[2 + 16 * lane + i];
        }
    }
#pragma unroll
    for (unsigned c = 0; c < kMaxChunks; ++c) if (c < chunks) mm = fmaxf(mm, mc[c]);
    float lsum = 0.0F, acc[16];
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) acc[i] = 0.0F;
#pragma unroll
    for (unsigned c = 0; c < kMaxChunks; ++c) {
        if (c < chunks) {
            const float e = Sel ? exp_f_sel(mc[c] - mm) : exp_f(mc[c] - mm);
            lsum = fmaf(lc[c], e, lsum);
#pragma unroll
            for (unsigned i = 0; i < 16; ++i) acc[i] = fmaf(pa[c][i], e, acc[i]);
        }
    }
    lsum = lsum + (Sel ? exp_f_sel(v.sink[h] - mm) : exp_f(v.sink[h] - mm));
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) acc[i] = acc[i] / lsum;
    rope512(acc, v.rope[r], lane, true);
    const std::size_t at = (static_cast<std::size_t>(r) * v.heads + h) * kHeadDim + 16 * lane;
    float* out = v.o + at;
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) out[i] = acc[i];
    if (v.codes == nullptr) return;
    // A8 of the 32-value group held by lanes 2k, 2k + 1 (16 values each), quantize_a8's bits: the order-free maximum
    // of |y|, delta = amax / 127 (1 for a zero or tiny group), code = rint(y / delta) clamped to +-127, exact sum.
    // seeded from the first value (as quantize_a8's per-lane |y|): a group of NaNs keeps a NaN maximum and delta
    float amax = fabsf(acc[0]);
#pragma unroll
    for (unsigned i = 1; i < 16; ++i) amax = fmaxf(amax, fabsf(acc[i]));
    amax = fmaxf(amax, xor_lane<1>(amax));
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
    int sum = 0;
    std::int8_t* code = v.codes + at;
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) {
        const float rounded = tiny ? 0.0F : rintf(acc[i] / delta);
        const int c = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
        code[i] = static_cast<std::int8_t>(c);
        sum += c;
    }
    sum += __shfl_xor(sum, 1, 32);
    if ((lane & 1U) == 0U) {
        const std::size_t g = at / 32U;
        v.deltas[g] = delta;
        v.sums[g] = sum;
    }
}
__device__ __forceinline__ unsigned visible_keys(const AttentionView& v, unsigned r, unsigned& nw) {
    const std::uint64_t p = v.pos[r];
    nw = p + 1U < kWindow ? static_cast<unsigned>(p + 1U) : kWindow;
    return nw + (v.idx_count != nullptr ? v.idx_count[r] : 0U);
}
template <unsigned kAttnHpw, bool Swizzle, bool Half = false, bool Fold = false, unsigned Lean = 0>
__device__ __forceinline__ void attention_chunk_body(const AttentionView& v, const uint3 bid) {
    DS41_TIMED(5U);
    __shared__ float4 ks[kTile][kHeadDim / 4];   // the f32 tile, or (STRIX_DS41_ATTN_TILE_F16) the f16 tile in its first half
    auto* kb = reinterpret_cast<uint4 (*)[kHeadDim / 8]>(ks);
    __shared__ std::uint32_t cidx[kChunk];
    __shared__ unsigned last_s;
    const unsigned c = bid.x, r = bid.z, tid = threadIdx.x, lane = tid & 31U;
    const unsigned hb = (bid.y * (kAttnThreads / 32U) + (tid >> 5U)) * kAttnHpw;
    unsigned nw;
    const unsigned n = visible_keys(v, r, nw), chunks = (n + kChunk - 1) / kChunk;
    if (v.max_visible != 0U && n > v.max_visible) __builtin_trap();   // the grid covers max_visible keys only
    if (c >= chunks) return;
    const unsigned j0 = c * kChunk, end = n - j0 < kChunk ? n : j0 + kChunk;
    const std::uint64_t p = v.pos[r];
    const unsigned first_slot = static_cast<unsigned>((p + 1U - nw) % v.capacity);
    if (tid < kChunk && j0 + tid < end && j0 + tid >= nw) cidx[tid] = v.idx[static_cast<std::size_t>(r) * v.topk + j0 + tid - nw];
    const unsigned tk = tid / 16U, part = tid % 16U;
    [[maybe_unused]] const std::uint64_t call0 = v.pos[0];   // the call's first row: positions call0 .. are this call's rows
    // STRIX_DS41_ATTN_WINDOW_FOLD: the window keys of this call's rows 0..r that fall in this chunk, one wave per key
    // (wave k = row k, window_write's own per-lane code) into an LDS copy of the ring entry; the row's own entry is also
    // copied to its ring slot by head group 0 (the only writer; no block of this call reads those slots).
    [[maybe_unused]] __shared__ __attribute__((aligned(16))) std::uint8_t fold_e[Fold ? kMaxRows : 1][kWindowEntryBytes];
    if constexpr (Fold) {
        const unsigned k = tid >> 5U;   // key of row k: visible index nw - 1 - (r - k); uniform per wave
        if (k <= r && r - k < nw) {
            const unsigned j = nw - 1U - (r - k);
            if (j >= j0 && j < end) {
                window_entry(v.kv + k * kHeadDim, v.kv_norm, v.rope[k], lane, fold_e[k]);
                if (k == r && bid.y == 0U) {
                    unsigned slot = first_slot + j;
                    if (slot >= v.capacity) slot -= v.capacity;
                    std::uint8_t* e = v.ring_out + static_cast<std::size_t>(slot) * kWindowEntryBytes;
                    __builtin_amdgcn_wave_barrier();   // the wave's LDS bytes are written (same wave reads them back)
                    *reinterpret_cast<uint4*>(e + 16 * lane) = *reinterpret_cast<const uint4*>(fold_e[k] + 16 * lane);
                    if ((lane & 1U) == 0U) e[512 + lane / 2] = fold_e[k][512 + lane / 2];
                }
            }
        }
    }
    __syncthreads();
    DS41_MARK(0U);
    const auto load = [&](unsigned t0, KeyRaw& k) {
        const unsigned j = t0 + tk;
        if (j >= end) return;
        if (j < nw) {
            unsigned slot = first_slot + j;
            if (slot >= v.capacity) slot -= v.capacity;
            const std::uint64_t at = p + 1U - nw + j;   // the key's position
            if (Fold && at >= call0) {                    // STRIX_DS41_ATTN_WINDOW_FOLD: a key of this call, computed above
                const std::uint8_t* e = fold_e[at - call0];
                const uint4* cp = reinterpret_cast<const uint4*>(e + 32 * part);
                k.c0 = cp[0]; k.c1 = cp[1];
                k.sc = e[512 + part];
                return;
            }
            const std::uint8_t* e = v.ring + static_cast<std::size_t>(slot) * kWindowEntryBytes;
            const uint4* cp = reinterpret_cast<const uint4*>(e + 32 * part);
            k.c0 = cp[0]; k.c1 = cp[1];
            k.sc = e[512 + part];
        } else {
            const std::uint8_t* e = v.kv_cache + static_cast<std::size_t>(cidx[j - j0]) * kKvEntryBytes;
            k.c0 = *reinterpret_cast<const uint4*>(e + 16 * part);
            k.sc = *reinterpret_cast<const std::uint16_t*>(e + 256 + 2 * part);
        }
    };
    KeyRaw raw{};
    load(j0, raw);
    float q[kAttnHpw][16], m[kAttnHpw], l[kAttnHpw], acc[kAttnHpw][16];
#pragma unroll
    for (unsigned h = 0; h < kAttnHpw; ++h) {
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) q[h][i] = v.q[(static_cast<std::size_t>(r) * v.heads + hb + h) * kHeadDim + 16 * lane + i];
        rope512(q[h], v.rope[r], lane, false);
        m[h] = -__builtin_inff(); l[h] = 0.0F;
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) acc[h][i] = 0.0F;
    }
    DS41_MARK(1U);
    for (unsigned t0 = j0; t0 < end; t0 += kTile) {
        const unsigned cnt = end - t0 < kTile ? end - t0 : kTile;
        bool f32_tile = true;
        if constexpr (Half) {
            f32_tile = __syncthreads_or(store_key_f16(kb[tk], part, raw, t0 + tk < nw, tk < cnt) ? 0 : 1) != 0;
            if (f32_tile) {   // uniform: some value of the tile is not exact in f16: today's f32 tile over the same LDS
                if constexpr (Lean >= 2U) store_key_lean<Swizzle, (Lean >= 3U)>(ks[tk], part, raw, t0 + tk < nw, tk < cnt); else store_key<Swizzle>(ks[tk], part, raw, t0 + tk < nw, tk < cnt);
                __syncthreads();
            }
        } else {
            if constexpr (Lean >= 2U) store_key_lean<Swizzle, (Lean >= 3U)>(ks[tk], part, raw, t0 + tk < nw, tk < cnt); else store_key<Swizzle>(ks[tk], part, raw, t0 + tk < nw, tk < cnt);
            __syncthreads();
        }
        DS41_MARK(2U + 2U * ((t0 - j0) / kTile));
        if (t0 + kTile < end) load(t0 + kTile, raw);
        if (!Half || f32_tile) attention_tile<kAttnHpw, Swizzle, false, Half, Lean>(ks, cnt, lane, q, m, l, acc, t0 == j0);
        else attention_tile<kAttnHpw, Swizzle, true, true, Lean>(kb, cnt, lane, q, m, l, acc, t0 == j0);
        __syncthreads();
        DS41_MARK(3U + 2U * ((t0 - j0) / kTile));
    }
#pragma unroll
    for (unsigned h = 0; h < kAttnHpw; ++h) {
        float* pp = v.partial + ((static_cast<std::size_t>(r) * v.heads + hb + h) * kMaxChunks + c) * (2 + kHeadDim);
        if (lane == 0U) { pp[0] = m[h]; pp[1] = l[h]; }
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) pp[2 + 16 * lane + i] = acc[h][i];
    }
    DS41_MARK(10U);
    if (v.tickets == nullptr) return;
    // the last chunk block of this (row, head group) to finish combines its heads
    __threadfence();
    __syncthreads();
    std::uint32_t* ticket = v.tickets + r * kAttnGroups + bid.y;
    if (tid == 0U) last_s = atomicAdd(ticket, 1U) == chunks - 1U ? 1U : 0U;
    __syncthreads();
    if (last_s == 0U) return;
    __threadfence();
#pragma unroll 1
    for (unsigned h = 0; h < kAttnHpw; ++h) attention_combine_head<(Lean >= 3U)>(v, r, hb + h, lane, chunks);
    if (tid == 0U) *ticket = 0U;
    DS41_MARK(11U);
}
__device__ __forceinline__ void attention_combine_body(const AttentionView& v, const uint3 bid) {
    DS41_TIMED(6U);
    unsigned nw;
    const unsigned n = visible_keys(v, bid.y, nw);
    attention_combine_head(v, bid.y, bid.x, threadIdx.x, (n + kChunk - 1) / kChunk);
}


// Prefill-only BF16 hi/lo attention. Packed K/V values fit BF16 in its normal
// range; extreme underflow is lossy; queries and probabilities retain two terms.
using bf16x16 = unsigned short __attribute__((ext_vector_type(16)));
using f32x8 = float __attribute__((ext_vector_type(8)));
__device__ unsigned short bf_rn(float f) {
    unsigned u=__builtin_bit_cast(unsigned,f);
    if ((u&0x7f800000U)==0x7f800000U) return static_cast<unsigned short>((u>>16U)|((u&0x7fffffU)?64U:0U));
    return static_cast<unsigned short>((u+0x7fffU+((u>>16U)&1U))>>16U);
}
__device__ float bf_float(unsigned short b) { return __builtin_bit_cast(float,unsigned(b)<<16U); }
__global__ __launch_bounds__(256) void attention_bf16_chunk(AttentionView v) {
    __shared__ unsigned short qh[16][512],ql[16][512],keys[16][512],ph[16][16],pl[16][16];
    __shared__ float scores[16][16],ms[16],ls[16],scale[16];
    unsigned tid=threadIdx.x,lane=tid%32,wave=tid/32,lo=lane%16,half=lane/16;
    unsigned row=blockIdx.z,h0=blockIdx.y*16,c=blockIdx.x,nw;
    unsigned n=visible_keys(v,row,nw),chunks=(n+kChunk-1)/kChunk;
    if(c>=chunks)return;
    unsigned first=c*kChunk,end=min(n,first+kChunk);
    auto pos=v.pos[row];unsigned first_slot=(pos+1-nw)%v.capacity;
    for(unsigned h=wave;h<16;h+=8){
        float q[16];
        for(unsigned i=0;i<16;++i)q[i]=v.q[(std::size_t(row)*v.heads+h0+h)*512+lane*16+i];
        rope512(q,v.rope[row],lane,false);
        for(unsigned i=0;i<16;++i){unsigned d=lane*16+i;qh[h][d]=bf_rn(q[i]);ql[h][d]=bf_rn(q[i]-bf_float(qh[h][d]));}
    }
    if(tid<16){ms[tid]=-__builtin_inff();ls[tid]=0;}
    f32x8 acc[4]{};
    __syncthreads();
    for(unsigned t0=first;t0<end;t0+=16){
        unsigned tk=tid/16,part=tid%16,key_index=t0+tk,cnt=min(16U,end-t0);
        KeyRaw raw{};bool valid=key_index<end,win=key_index<nw;
        if(valid){
            if(win){unsigned slot=(first_slot+key_index)%v.capacity;auto* e=v.ring+std::size_t(slot)*kWindowEntryBytes;auto* b=reinterpret_cast<const uint4*>(e+32*part);raw.c0=b[0];raw.c1=b[1];raw.sc=e[512+part];}
            else{unsigned idx=v.idx[std::size_t(row)*v.topk+key_index-nw];auto* e=v.kv_cache+std::size_t(idx)*kKvEntryBytes;raw.c0=*reinterpret_cast<const uint4*>(e+16*part);raw.sc=*reinterpret_cast<const unsigned short*>(e+256+2*part);}
        }
        unsigned cw[8]={raw.c0.x,raw.c0.y,raw.c0.z,raw.c0.w,raw.c1.x,raw.c1.y,raw.c1.z,raw.c1.w};
        for(unsigned i=0;i<32;++i){
            float f=0;
            if(valid && win)f=e4m3_value(static_cast<unsigned char>(cw[i/4]>>(8*(i%4))))*e8m0_value(static_cast<unsigned char>(raw.sc));
            else if(valid)f=e2m1_value(static_cast<unsigned char>((cw[i/8]>>(4*(i%8)))&15))*e4m3_value(static_cast<unsigned char>(raw.sc>>(8*(i/16))));
            keys[tk][part*32+i]=bf_rn(f);
        }
        __syncthreads();
        if(wave==0){
            f32x8 dot{};
            for(unsigned k=0;k<512;k+=16){
                bf16x16 a,b,al;
                for(unsigned i=0;i<16;++i){a[i]=qh[lo][k+i];al[i]=ql[lo][k+i];b[i]=keys[lo][k+i];}
                dot=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a,b,dot);
                dot=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(al,b,dot);
            }
            for(unsigned e=0;e<8;++e)scores[2*e+half][lo]=lo<cnt?dot[e]*kAttnScale:-__builtin_inff();
        }
        __syncthreads();
        if(tid<16){
            float tm=-__builtin_inff();for(unsigned j=0;j<cnt;++j)tm=fmaxf(tm,scores[tid][j]);
            float mn=fmaxf(ms[tid],tm),sc=exp_f(ms[tid]-mn);scale[tid]=sc;float sum=ls[tid]*sc;
            for(unsigned j=0;j<16;++j){float pr=j<cnt?exp_f(scores[tid][j]-mn):0;sum=sum+pr;ph[tid][j]=bf_rn(pr);pl[tid][j]=bf_rn(pr-bf_float(ph[tid][j]));}
            ms[tid]=mn;ls[tid]=sum;
        }
        __syncthreads();
        for(unsigned tile=0;tile<4;++tile){
            for(unsigned e=0;e<8;++e)acc[tile][e]*=scale[2*e+half];
            bf16x16 a,al,b;unsigned dim=wave*64+tile*16+lo;
            for(unsigned j=0;j<16;++j){a[j]=ph[lo][j];al[j]=pl[lo][j];b[j]=keys[j][dim];}
            acc[tile]=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a,b,acc[tile]);
            acc[tile]=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(al,b,acc[tile]);
        }
        __syncthreads();
    }
    if(tid<16){auto* pp=v.partial+((std::size_t(row)*v.heads+h0+tid)*kMaxChunks+c)*(2+512);pp[0]=ms[tid];pp[1]=ls[tid];}
    for(unsigned tile=0;tile<4;++tile)for(unsigned e=0;e<8;++e){auto* pp=v.partial+((std::size_t(row)*v.heads+h0+2*e+half)*kMaxChunks+c)*(2+512);pp[2+wave*64+tile*16+lo]=acc[tile][e];}
}

// Variant 2 (STRIX_DS41_PREFILL_BF16_ATTN=2): the same hi/lo arithmetic per product, with the QK dot of each 16-key
// step split over the 8 waves (64 dimensions each, partial tiles summed in a fixed tree) instead of wave 0 alone, and
// the decoded keys written to LDS 16 bytes at a time instead of 2.
__global__ __launch_bounds__(256) void attention_bf16_chunk2(AttentionView v) {
    __shared__ __attribute__((aligned(16))) unsigned short qh[16][512], ql[16][512], keys[16][512];
    __shared__ unsigned short ph[16][16], pl[16][16];
    __shared__ float spart[8][16][16], scores[16][16], ms[16], ls[16], scale[16];
    unsigned tid=threadIdx.x,lane=tid%32,wave=tid/32,lo=lane%16,half=lane/16;
    unsigned row=blockIdx.z,h0=blockIdx.y*16,c=blockIdx.x,nw;
    unsigned n=visible_keys(v,row,nw),chunks=(n+kChunk-1)/kChunk;
    if(c>=chunks)return;
    unsigned first=c*kChunk,end=min(n,first+kChunk);
    auto pos=v.pos[row];unsigned first_slot=(pos+1-nw)%v.capacity;
    for(unsigned h=wave;h<16;h+=8){
        float q[16];
        for(unsigned i=0;i<16;++i)q[i]=v.q[(std::size_t(row)*v.heads+h0+h)*512+lane*16+i];
        rope512(q,v.rope[row],lane,false);
        for(unsigned i=0;i<16;++i){unsigned d=lane*16+i;qh[h][d]=bf_rn(q[i]);ql[h][d]=bf_rn(q[i]-bf_float(qh[h][d]));}
    }
    if(tid<16){ms[tid]=-__builtin_inff();ls[tid]=0;}
    f32x8 acc[4]{};
    __syncthreads();
    for(unsigned t0=first;t0<end;t0+=16){
        unsigned tk=tid/16,part=tid%16,key_index=t0+tk,cnt=min(16U,end-t0);
        KeyRaw raw{};bool valid=key_index<end,win=key_index<nw;
        if(valid){
            if(win){unsigned slot=(first_slot+key_index)%v.capacity;auto* e=v.ring+std::size_t(slot)*kWindowEntryBytes;auto* b=reinterpret_cast<const uint4*>(e+32*part);raw.c0=b[0];raw.c1=b[1];raw.sc=e[512+part];}
            else{unsigned idx=v.idx[std::size_t(row)*v.topk+key_index-nw];auto* e=v.kv_cache+std::size_t(idx)*kKvEntryBytes;raw.c0=*reinterpret_cast<const uint4*>(e+16*part);raw.sc=*reinterpret_cast<const unsigned short*>(e+256+2*part);}
        }
        unsigned cw[8]={raw.c0.x,raw.c0.y,raw.c0.z,raw.c0.w,raw.c1.x,raw.c1.y,raw.c1.z,raw.c1.w};
        unsigned packed[16];
        for(unsigned i=0;i<32;i+=2){
            float f0=0,f1=0;
            if(valid && win){
                const float s=e8m0_value(static_cast<unsigned char>(raw.sc));
                f0=e4m3_value(static_cast<unsigned char>(cw[i/4]>>(8*(i%4))))*s;f1=e4m3_value(static_cast<unsigned char>(cw[(i+1)/4]>>(8*((i+1)%4))))*s;
            }else if(valid){
                const float s=e4m3_value(static_cast<unsigned char>(raw.sc>>(8*(i/16))));
                f0=e2m1_value(static_cast<unsigned char>((cw[i/8]>>(4*(i%8)))&15))*s;f1=e2m1_value(static_cast<unsigned char>((cw[(i+1)/8]>>(4*((i+1)%8)))&15))*s;
            }
            packed[i/2]=unsigned(bf_rn(f0))|(unsigned(bf_rn(f1))<<16);
        }
        auto* kd=reinterpret_cast<uint4*>(&keys[tk][part*32]);
        for(unsigned i=0;i<4;++i)kd[i]=uint4{packed[4*i],packed[4*i+1],packed[4*i+2],packed[4*i+3]};
        __syncthreads();
        {
            f32x8 dot{};
            for(unsigned k=wave*64;k<wave*64+64;k+=16){
                bf16x16 a,b,al;
                const auto* qa=reinterpret_cast<const uint4*>(&qh[lo][k]);const auto* qb=reinterpret_cast<const uint4*>(&ql[lo][k]);const auto* kb=reinterpret_cast<const uint4*>(&keys[lo][k]);
                uint4 x0=qa[0],x1=qa[1],y0=qb[0],y1=qb[1],z0=kb[0],z1=kb[1];
                unsigned xa[8]={x0.x,x0.y,x0.z,x0.w,x1.x,x1.y,x1.z,x1.w},ya[8]={y0.x,y0.y,y0.z,y0.w,y1.x,y1.y,y1.z,y1.w},za[8]={z0.x,z0.y,z0.z,z0.w,z1.x,z1.y,z1.z,z1.w};
                for(unsigned i=0;i<8;++i){a[2*i]=static_cast<unsigned short>(xa[i]);a[2*i+1]=static_cast<unsigned short>(xa[i]>>16);al[2*i]=static_cast<unsigned short>(ya[i]);al[2*i+1]=static_cast<unsigned short>(ya[i]>>16);b[2*i]=static_cast<unsigned short>(za[i]);b[2*i+1]=static_cast<unsigned short>(za[i]>>16);}
                dot=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a,b,dot);
                dot=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(al,b,dot);
            }
            for(unsigned e=0;e<8;++e)spart[wave][2*e+half][lo]=dot[e];
        }
        __syncthreads();
        {
            unsigned hh=tid/16,kk=tid%16;
            float s=((spart[0][hh][kk]+spart[1][hh][kk])+(spart[2][hh][kk]+spart[3][hh][kk]))+((spart[4][hh][kk]+spart[5][hh][kk])+(spart[6][hh][kk]+spart[7][hh][kk]));
            scores[hh][kk]=kk<cnt?s*kAttnScale:-__builtin_inff();
        }
        __syncthreads();
        if(tid<16){
            float tm=-__builtin_inff();for(unsigned j=0;j<cnt;++j)tm=fmaxf(tm,scores[tid][j]);
            float mn=fmaxf(ms[tid],tm),sc=exp_f(ms[tid]-mn);scale[tid]=sc;float sum=ls[tid]*sc;
            for(unsigned j=0;j<16;++j){float pr=j<cnt?exp_f(scores[tid][j]-mn):0;sum=sum+pr;ph[tid][j]=bf_rn(pr);pl[tid][j]=bf_rn(pr-bf_float(ph[tid][j]));}
            ms[tid]=mn;ls[tid]=sum;
        }
        __syncthreads();
        for(unsigned tile=0;tile<4;++tile){
            for(unsigned e=0;e<8;++e)acc[tile][e]*=scale[2*e+half];
            bf16x16 a,al,b;unsigned dim=wave*64+tile*16+lo;
            for(unsigned j=0;j<16;++j){a[j]=ph[lo][j];al[j]=pl[lo][j];b[j]=keys[j][dim];}
            acc[tile]=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a,b,acc[tile]);
            acc[tile]=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(al,b,acc[tile]);
        }
        __syncthreads();
    }
    if(tid<16){auto* pp=v.partial+((std::size_t(row)*v.heads+h0+tid)*kMaxChunks+c)*(2+512);pp[0]=ms[tid];pp[1]=ls[tid];}
    for(unsigned tile=0;tile<4;++tile)for(unsigned e=0;e<8;++e){auto* pp=v.partial+((std::size_t(row)*v.heads+h0+2*e+half)*kMaxChunks+c)*(2+512);pp[2+wave*64+tile*16+lo]=acc[tile][e];}
}

#include "hip_ds41_attention_f16wmma.inl"  // FAST lever ds41-attn-f16wmma-v1 (not exact; prefill only)

// ---- kernels: one call (the block's own coordinates), or several segments of one kind in one launch ----
// A segment launch carries up to kMaxSegments views by value; segment k owns the linear blocks [first[k], first[k+1])
// and each of them runs exactly the block (x, y, z) of that segment's own grid (gx x gy x gz), so every row's work,
// arithmetic and order are those of the per-segment call. Segments never share state (each its own request).
template <class View>
struct SegmentLaunch {
    View v[kMaxSegments];
    unsigned first[kMaxSegments + 1];
    unsigned gx[kMaxSegments], gy[kMaxSegments];
    unsigned n;
};
template <class View>
__device__ __forceinline__ uint3 segment_block(const SegmentLaunch<View>& s, unsigned& k) {
    const unsigned b = blockIdx.x;
    k = 0;
    while (k + 1U < s.n && b >= s.first[k + 1U]) ++k;
    const unsigned l = b - s.first[k], gx = s.gx[k], gy = s.gy[k];
    return make_uint3(l % gx, (l / gx) % gy, l / (gx * gy));
}
__device__ __forceinline__ uint3 own_block() { return make_uint3(blockIdx.x, blockIdx.y, blockIdx.z); }

__global__ __launch_bounds__(32) void window_write_kernel(WindowView v) { window_write_body(v, own_block()); }
__global__ __launch_bounds__(32) void window_write_seg_kernel(SegmentLaunch<WindowView> s) {
    unsigned k;
    const uint3 b = segment_block(s, k);
    window_write_body(s.v[k], b);
}
template <bool Ahead, bool Fast>
__global__ __launch_bounds__(128) void compress_kernel(CompressView v) { compress_body<Ahead, Fast>(v, own_block()); }
template <bool Ahead, bool Fast>
__global__ __launch_bounds__(128) void compress_seg_kernel(SegmentLaunch<CompressView> s) {
    unsigned k;
    const uint3 b = segment_block(s, k);
    compress_body<Ahead, Fast>(s.v[k], b);
}
__global__ __launch_bounds__(256) void index_qprep_kernel(IndexView v) { index_qprep_body(v, own_block()); }
__global__ __launch_bounds__(256) void index_qprep_seg_kernel(SegmentLaunch<IndexView> s) {
    unsigned k;
    const uint3 b = segment_block(s, k);
    index_qprep_body(s.v[k], b);
}
__global__ __launch_bounds__(256) void index_scores_kernel(IndexView v) { index_scores_body(v, own_block()); }
__global__ __launch_bounds__(256) void index_scores_seg_kernel(SegmentLaunch<IndexView> s) {
    unsigned k;
    const uint3 b = segment_block(s, k);
    index_scores_body(s.v[k], b);
}
template <bool Legacy>
__global__ __launch_bounds__(kSelThreads) void select_kernel(SelectView v) { select_body<Legacy>(v, own_block()); }
template <bool Legacy>
__global__ __launch_bounds__(kSelThreads) void select_seg_kernel(SegmentLaunch<SelectView> s) {
    unsigned k;
    const uint3 b = segment_block(s, k);
    select_body<Legacy>(s.v[k], b);
}
template <bool Legacy>
__global__ __launch_bounds__(kSelThreads) void select_slice_kernel(SelectView v) { select_slice_body<Legacy>(v, own_block()); }
template <bool Legacy>
__global__ __launch_bounds__(kSelThreads) void select_slice_seg_kernel(SegmentLaunch<SelectView> s) {
    unsigned k;
    const uint3 b = segment_block(s, k);
    select_slice_body<Legacy>(s.v[k], b);
}
template <bool Legacy>
__global__ __launch_bounds__(kSelThreads) void select_merge_kernel(SelectView v) { select_merge_body<Legacy>(v, own_block()); }
template <bool Legacy>
__global__ __launch_bounds__(kSelThreads) void select_merge_seg_kernel(SegmentLaunch<SelectView> s) {
    unsigned k;
    const uint3 b = segment_block(s, k);
    select_merge_body<Legacy>(s.v[k], b);
}
__global__ __launch_bounds__(256) void select_direct_kernel(SelectView v) { select_direct_body(v, own_block()); }
__global__ __launch_bounds__(256) void select_direct_seg_kernel(SegmentLaunch<SelectView> s) {
    unsigned k;
    const uint3 b = segment_block(s, k);
    select_direct_body(s.v[k], b);
}
// Cap (STRIX_DS41_ATTN_WAVE_CAP, lean 3, 4 heads per wave): compiled for at least 6 waves per SIMD, i.e. at most 240 VGPRs
// (243 without it: 5 waves per SIMD). gfx1151 prefill-attention bench: -9..-14 % per call with compressed keys and at 64
// heads, while 32-head window-only calls are faster uncapped. The register budget changes the allocation only, not the
// operations: the same bits either way.
template <unsigned kAttnHpw, bool Swizzle, bool Half, bool Fold, unsigned Lean, bool Cap = false>
__global__ __launch_bounds__(kAttnThreads) __attribute__((amdgpu_waves_per_eu(Cap ? 6 : 1)))
void attention_chunk_kernel(AttentionView v) { attention_chunk_body<kAttnHpw, Swizzle, Half, Fold, Lean>(v, own_block()); }
template <unsigned kAttnHpw, bool Swizzle, bool Half, bool Fold, unsigned Lean, bool Cap = false>
__global__ __launch_bounds__(kAttnThreads) __attribute__((amdgpu_waves_per_eu(Cap ? 6 : 1)))
void attention_chunk_seg_kernel(SegmentLaunch<AttentionView> s) {
    unsigned k;
    const uint3 b = segment_block(s, k);
    attention_chunk_body<kAttnHpw, Swizzle, Half, Fold, Lean>(s.v[k], b);
}
__global__ __launch_bounds__(32) void attention_combine_kernel(AttentionView v) { attention_combine_body(v, own_block()); }
__global__ __launch_bounds__(32) void attention_combine_seg_kernel(SegmentLaunch<AttentionView> s) {
    unsigned k;
    const uint3 b = segment_block(s, k);
    attention_combine_body(s.v[k], b);
}

// Host side: segments of one kind, each with its own grid, flushed as launches of at most kMaxSegments.
template <class View>
struct SegmentBatch {
    SegmentLaunch<View> s{};
    void add(const View& v, dim3 g) {
        s.v[s.n] = v;
        s.gx[s.n] = g.x;
        s.gy[s.n] = g.y;
        s.first[s.n + 1U] = s.first[s.n] + g.x * g.y * g.z;
        ++s.n;
    }
    [[nodiscard]] bool full() const noexcept { return s.n == kMaxSegments; }
    [[nodiscard]] bool empty() const noexcept { return s.n == 0U; }
    [[nodiscard]] unsigned blocks() const noexcept { return s.first[s.n]; }
    void clear() { s = SegmentLaunch<View>{}; }
};

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
void check_rows(std::uint32_t rows) {
    const unsigned limit = strix::ds41::prefill_tuning::enabled("STRIX_DS41_PREFILL_SLABS") ? kMaxCallRows : kLegacyMaxCallRows;
    if (rows == 0U || rows > limit) throw std::invalid_argument("ds41 attention: rows exceed enabled call limit");
}

}  // namespace

static void check_window(const WindowView& v) {
    check_rows(v.rows);
    if (v.capacity < kWindow + v.rows) throw std::invalid_argument("ds41 window ring capacity");
}
void enqueue_window_write(const WindowView& v, hipStream_t stream) {
    check_window(v);
    hipLaunchKernelGGL(window_write_kernel, dim3(v.rows), dim3(32), 0, stream, v);
    check(hipGetLastError(), "ds41 window write");
}
int compress_ahead_mode = -1;  // set_compress_ahead: 0 / 1; -1 = STRIX_DS41_COMPRESS_AHEAD
void set_compress_ahead(int mode) { compress_ahead_mode = mode; }
int compress_fast_mode = -1;  // set_compress_fast: 0 / 1; -1 = STRIX_DS41_COMPRESS_FAST
void set_compress_fast(int mode) { compress_fast_mode = mode; }
static void check_compress(const CompressView& v) {
    check_rows(v.rows);
    if (v.ratio != 1U && v.ratio != 2U) throw std::invalid_argument("ds41 compress ratio");
}
// 2 fast, 1 ahead, 0 the default kernel
static int compress_form() {
    static const bool ahead_env = [] { const char* e = std::getenv("STRIX_DS41_COMPRESS_AHEAD"); return e && e[0] == '1'; }();
    static const bool fast_env = [] { const char* e = std::getenv("STRIX_DS41_COMPRESS_FAST"); return !(e && e[0] == '0'); }();  // default on
    if (compress_fast_mode < 0 ? fast_env : compress_fast_mode != 0) return 2;
    if (compress_ahead_mode < 0 ? ahead_env : compress_ahead_mode != 0) return 1;
    return 0;
}
void enqueue_compress(const CompressView& v, hipStream_t stream) {
    check_compress(v);
    const int form = compress_form();
    if (form == 2) hipLaunchKernelGGL(HIP_KERNEL_NAME(compress_kernel<false, true>), dim3(v.rows), dim3(128), 0, stream, v);
    else if (form == 1) hipLaunchKernelGGL(HIP_KERNEL_NAME(compress_kernel<true, false>), dim3(v.rows), dim3(128), 0, stream, v);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(compress_kernel<false, false>), dim3(v.rows), dim3(128), 0, stream, v);
    check(hipGetLastError(), "ds41 compress");
}
static void check_index(const IndexView& v) {
    check_rows(v.rows);
    if (v.qprep == nullptr) throw std::invalid_argument("ds41 index scores: IndexView::qprep scratch (rows x kIdxPrepBytes) is required");
}
void enqueue_index_scores(const IndexView& v, hipStream_t stream) {
    check_rows(v.rows);
    if (v.max_count == 0U) return;
    check_index(v);
    hipLaunchKernelGGL(index_qprep_kernel, dim3(v.rows), dim3(256), 0, stream, v);
    hipLaunchKernelGGL(index_scores_kernel, dim3((v.max_count + 255U) / 256U), dim3(256), 0, stream, v);
    check(hipGetLastError(), "ds41 index scores");
}
// slices of the two-level selection (0: one block per row)
static unsigned select_slices(const SelectView& v) {
    check_rows(v.rows);
    if (v.max_count <= kSelSlice) return 0U;
    const std::size_t slices = (v.max_count + kSelSlice - 1) / kSelSlice;
    if (2 * slices * v.topk + (v.max_count + v.block_size - 1) / v.block_size > v.max_count)
        throw std::invalid_argument("ds41 select: top-k or candidate block size too large for the key scratch");
    if (v.compact && v.pool == Pool::candidate_user && (v.block_size < 8U || kSelSlice % v.block_size != 0U))
        throw std::invalid_argument("ds41 select: compact selection needs a candidate block size >= 8 dividing the slice");
    return static_cast<unsigned>(slices);
}
static dim3 select_merge_grid(const SelectView& v) { return dim3(v.pool == Pool::candidate_source ? 2U : 1U, v.rows); }
bool select_v1() {
    static const bool on = [] { const char* e = std::getenv("STRIX_DS41_SELECT_V1"); return e != nullptr && e[0] == '1'; }();
    return on;
}
template <bool Legacy>
void launch_select(const SelectView& v, hipStream_t stream) {
    const unsigned slices = select_slices(v);
    if (slices == 0U) {
        hipLaunchKernelGGL(select_kernel<Legacy>, dim3(v.rows), dim3(kSelThreads), 0, stream, v);
    } else {
        hipLaunchKernelGGL(select_slice_kernel<Legacy>, dim3(slices, v.rows), dim3(kSelThreads), 0, stream, v);
        hipLaunchKernelGGL(select_merge_kernel<Legacy>, select_merge_grid(v), dim3(kSelThreads), 0, stream, v);
    }
    check(hipGetLastError(), "ds41 select");
}
void enqueue_select(const SelectView& v, hipStream_t stream) {
    if (select_v1()) launch_select<true>(v, stream);
    else launch_select<false>(v, stream);
}
namespace {
int index_candidate_filter_mode = -1, index_direct_mode = -1, attention_segment_hpw_mode = -1;
unsigned attention_min_blocks_value = 0;
// on unless the variable is "0" (exact and never slower in interleaved ABA measurements: default on, =0 turns it off)
bool env_on_unless_zero(const char* name) { const char* e = std::getenv(name); return e == nullptr || !(e[0] == '0' && e[1] == 0); }
}  // namespace
void set_index_candidate_filter(int mode) { index_candidate_filter_mode = mode; }
bool index_candidate_filter() {
    static const bool env = env_on_unless_zero("STRIX_DS41_IDX_CAND_FILTER");
    return index_candidate_filter_mode < 0 ? env : index_candidate_filter_mode != 0;
}
void set_index_direct(int mode) { index_direct_mode = mode; }
bool index_direct() {
    static const bool env = env_on_unless_zero("STRIX_DS41_IDX_DIRECT");
    return index_direct_mode < 0 ? env : index_direct_mode != 0;
}
void set_attention_segment_hpw(int mode) { attention_segment_hpw_mode = mode; }
void set_attention_min_blocks(unsigned blocks) { attention_min_blocks_value = blocks; }
// the grid size below which a call takes fewer heads per wave (STRIX_DS41_ATTN_MIN_BLOCKS, default 40: one per two CUs)
static unsigned attention_min_blocks() {
    static const unsigned env = [] { const char* e = std::getenv("STRIX_DS41_ATTN_MIN_BLOCKS"); return e != nullptr && e[0] != 0 ? static_cast<unsigned>(std::strtoul(e, nullptr, 10)) : 40U; }();
    return attention_min_blocks_value != 0U ? attention_min_blocks_value : env;
}
static bool attention_segment_hpw() {
    static const bool env = env_on_unless_zero("STRIX_DS41_ATTN_SEG_HPW");
    return attention_segment_hpw_mode < 0 ? env : attention_segment_hpw_mode != 0;
}
static void check_select_direct(const SelectView& v) {
    check_rows(v.rows);
    if (v.max_count > v.topk) throw std::invalid_argument("ds41 select direct: a row may have more positions than top-k");
    if (v.pool == Pool::candidate_source && (v.block_size == 0U || (v.max_count + v.block_size - 1) / v.block_size > v.cand_blocks))
        throw std::invalid_argument("ds41 select direct: candidate pool parameters");
}
void enqueue_select_direct(const SelectView& v, hipStream_t stream) {
    check_select_direct(v);
    hipLaunchKernelGGL(select_direct_kernel, dim3(v.rows), dim3(256), 0, stream, v);
    check(hipGetLastError(), "ds41 select direct");
}
// chunk blocks per row: all kMaxChunks, or (AttentionView::max_visible, STRIX_DS41_ATTN_LIVE_CHUNKS) the chunks of the
// caller's bound on any row's visible keys
static unsigned attention_chunks(const AttentionView& v) {
    if (v.max_visible == 0U) return kMaxChunks;
    const unsigned c = (v.max_visible + kChunk - 1) / kChunk;
    return c < kMaxChunks ? c : kMaxChunks;
}
// heads per wave of a call: the most that still gives >= attention_min_blocks() blocks, else 1
static unsigned attention_hpw(const AttentionView& v) {
    check_rows(v.rows);
    if (v.capacity < kWindow + v.rows || (v.idx_count != nullptr && v.topk > kTopK))
        throw std::invalid_argument("ds41 attention capacity / topk");
    if (v.heads == 0U || v.heads > kHeads || v.heads % kAttnMinBlockHeads != 0U) throw std::invalid_argument("ds41 attention heads: multiple of 8, at most 64");
    const auto blocks = [&](unsigned hpw) { return attention_chunks(v) * (v.heads / (kAttnMinBlockHeads * hpw)) * v.rows; };
    const auto fits = [&](unsigned hpw) { return v.heads % (kAttnMinBlockHeads * hpw) == 0U && blocks(hpw) >= attention_min_blocks(); };
    return fits(4U) ? 4U : fits(2U) ? 2U : 1U;
}
int attention_lds_swizzle_mode = -1;
void set_attention_lds_swizzle(int mode) { attention_lds_swizzle_mode = mode; }
static bool attention_lds_swizzle() {
    static const bool env = [] { const char* e = std::getenv("STRIX_DS41_ATTN_LDS_SWIZZLE"); return !(e && e[0] == '0' && e[1] == 0); }();  // default on, =0 off
    return attention_lds_swizzle_mode < 0 ? env : attention_lds_swizzle_mode != 0;
}
int attention_live_chunks_mode = -1;
void set_attention_live_chunks(int mode) { attention_live_chunks_mode = mode; }
bool attention_live_chunks() {
    static const bool env = [] { const char* e = std::getenv("STRIX_DS41_ATTN_LIVE_CHUNKS"); return !(e && e[0] == '0' && e[1] == 0); }();  // default on, =0 off
    return attention_live_chunks_mode < 0 ? env : attention_live_chunks_mode != 0;
}
int attention_window_fold_mode = -1;
void set_attention_window_fold(int mode) { attention_window_fold_mode = mode; }
bool attention_window_fold() {
    static const bool env = [] { const char* e = std::getenv("STRIX_DS41_ATTN_WINDOW_FOLD"); return !(e && e[0] == '0' && e[1] == 0); }();  // default on, =0 off
    return attention_window_fold_mode < 0 ? env : attention_window_fold_mode != 0;
}
int attention_tile_f16_mode = -1;
void set_attention_tile_f16(int mode) { attention_tile_f16_mode = mode; }
static bool attention_tile_f16() {
    static const bool env = [] { const char* e = std::getenv("STRIX_DS41_ATTN_TILE_F16"); return !(e && e[0] == '0' && e[1] == 0); }();  // default on, =0 off
    return attention_tile_f16_mode < 0 ? env : attention_tile_f16_mode != 0;
}
int attention_wave_cap_mode = -1;
void set_attention_wave_cap(int mode) { attention_wave_cap_mode = mode; }
// the 6-wave form for a call: lean 3 at 4 heads per wave with compressed keys or 64 heads (default on, =0 off)
static bool attention_wave_cap(const AttentionView& v) {
    static const bool env = [] { const char* e = std::getenv("STRIX_DS41_ATTN_WAVE_CAP"); return !(e && e[0] == '0' && e[1] == 0); }();
    return (attention_wave_cap_mode < 0 ? env : attention_wave_cap_mode != 0) && (v.idx_count != nullptr || v.heads >= 64U);
}
int attention_lean_mode = -1;
void set_attention_lean(int mode) { attention_lean_mode = mode; }
int attention_lean() {
    // default 3 (2 + permuted-key / dual-issue QK, PV addressing, rescale skip, spread exp), 2 softmax/PV + dequant, 1 softmax/PV only, 0 off
    static const int env = [] { const char* e = std::getenv("STRIX_DS41_ATTN_LEAN"); return e == nullptr || e[0] == 0 ? 3 : e[0] >= '0' && e[0] <= '3' ? e[0] - '0' : 3; }();
    const int m = attention_lean_mode < 0 ? env : attention_lean_mode;
    return m > 3 ? 3 : m;
}
static void check_fold(const AttentionView& v) {
    if ((v.kv == nullptr) != (v.ring_out == nullptr) || (v.kv != nullptr && (v.kv_norm == nullptr || v.ring_out != v.ring || v.rows > kMaxRows)))
        throw std::invalid_argument("ds41 attention: window fold needs kv, kv_norm and ring_out == ring together, at most kMaxRows rows");
}
template <bool Swizzle, bool Half, bool Fold, unsigned Lean>
static void launch_attention(const AttentionView& v, hipStream_t stream) {
    const unsigned hpw = attention_hpw(v);
    const dim3 grid(attention_chunks(v), v.heads / (kAttnMinBlockHeads * hpw), v.rows);
    // the f16 tile at 1 and 2 heads per wave only (at 4 its two tile forms exceed the register file)
    if (hpw == 4U && Lean >= 3U && attention_wave_cap(v))
        hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_chunk_kernel<4, Swizzle, false, Fold, Lean, true>), grid, dim3(kAttnThreads), 0, stream, v);
    else if (hpw == 4U) hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_chunk_kernel<4, Swizzle, false, Fold, Lean>), grid, dim3(kAttnThreads), 0, stream, v);
    else if (hpw == 2U) hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_chunk_kernel<2, Swizzle, Half, Fold, Lean>), grid, dim3(kAttnThreads), 0, stream, v);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_chunk_kernel<1, Swizzle, Half, Fold, Lean>), grid, dim3(kAttnThreads), 0, stream, v);
    if (v.tickets == nullptr) hipLaunchKernelGGL(attention_combine_kernel, dim3(v.heads, v.rows), dim3(32), 0, stream, v);
    check(hipGetLastError(), "ds41 attention");
}
void enqueue_attention_bf16_prefill(const AttentionView& v, hipStream_t stream, int variant) {
    static const int env_variant = [] { const char* e = std::getenv("STRIX_DS41_BF16_ATTN_VARIANT"); return e && e[0] == '2' ? 2 : 1; }();
    if (variant == 0) variant = env_variant;
    (void)attention_hpw(v); // retain the incumbent view validation
    if(v.heads%16U)throw std::invalid_argument("BF16 prefill requires a multiple of 16 heads");
    if (variant == 2) hipLaunchKernelGGL(attention_bf16_chunk2, dim3(kMaxChunks,v.heads/16,v.rows),dim3(256),0,stream,v);
    else hipLaunchKernelGGL(attention_bf16_chunk, dim3(kMaxChunks,v.heads/16,v.rows),dim3(256),0,stream,v);
    hipLaunchKernelGGL(attention_combine_kernel,dim3(v.heads,v.rows),dim3(32),0,stream,v);
    check(hipGetLastError(),"BF16 hi/lo prefill attention");
}
unsigned fast_attn_cpb_value = 2U, fast_attn_heads_value = 16U;
void set_fast_attn_tuning(unsigned chunks_per_block, unsigned ablate, unsigned heads_per_block) {
    if (chunks_per_block != 1U && chunks_per_block != 2U && chunks_per_block != 5U && chunks_per_block != 10U)
        throw std::invalid_argument("ds41 f16wmma attention: chunks per block 1, 2, 5 or 10");
    if (heads_per_block != 16U && heads_per_block != 32U) throw std::invalid_argument("ds41 f16wmma attention: 16 or 32 heads per block");
    fast_attn_cpb_value = chunks_per_block;
    fast_attn_heads_value = heads_per_block;
    check(hipMemcpyToSymbol(HIP_SYMBOL(g_fast_attn_ablate), &ablate, sizeof(ablate)), "ds41 f16wmma ablate");
}
template <unsigned Cpb, unsigned H>
static void launch_f16wmma(const AttentionView& v, hipStream_t stream) {
    const dim3 grid((attention_chunks(v) + Cpb - 1U) / Cpb, v.heads / H, v.rows);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_f16wmma_kernel<Cpb, H>), grid, dim3(256), 0, stream, v);
    if (v.tickets == nullptr) hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_f16wmma_combine_kernel<Cpb>), dim3(v.heads, v.rows), dim3(32), 0, stream, v);
}
template <unsigned H>
static void launch_f16wmma_cpb(const AttentionView& v, hipStream_t stream) {
    switch (fast_attn_cpb_value) {
    case 1U: launch_f16wmma<1, H>(v, stream); break;
    case 5U: launch_f16wmma<5, H>(v, stream); break;
    case 10U: launch_f16wmma<10, H>(v, stream); break;
    default: launch_f16wmma<2, H>(v, stream); break;
    }
}
void enqueue_attention_f16wmma_prefill(const AttentionView& v, hipStream_t stream) {
    (void)attention_hpw(v);  // the incumbent view checks
    if (v.heads % kFastHeads != 0U) throw std::invalid_argument("ds41 f16wmma attention: heads must be a multiple of 16");
    if (v.kv != nullptr || v.ring_out != nullptr) throw std::invalid_argument("ds41 f16wmma attention: no window fold");
    if (fast_attn_heads_value == 32U && v.heads % 32U == 0U) launch_f16wmma_cpb<32>(v, stream);
    else launch_f16wmma_cpb<16>(v, stream);
    check(hipGetLastError(), "ds41 f16wmma attention");
}
template <unsigned Lean>
static void enqueue_attention_form(const AttentionView& v, bool fold, hipStream_t stream) {
    // The swizzle permutes the f32 tile's LDS vectors (the 4-heads-per-wave tile and the f16 kernels' f32 fallback tiles) and
    // leaves the tile choice alone: with the f16 tile on, its f16 tiles are unchanged.
    if (attention_lds_swizzle() && !fold)
        attention_tile_f16() ? launch_attention<true, true, false, Lean>(v, stream) : launch_attention<true, false, false, Lean>(v, stream);
    else if (attention_tile_f16()) fold ? launch_attention<false, true, true, Lean>(v, stream) : launch_attention<false, true, false, Lean>(v, stream);
    else fold ? launch_attention<false, false, true, Lean>(v, stream) : launch_attention<false, false, false, Lean>(v, stream);
}
void enqueue_attention(const AttentionView& v, hipStream_t stream) {
    check_fold(v);
    const bool fold = v.kv != nullptr;
    const int lean = attention_lean();
    if (lean == 3) enqueue_attention_form<3>(v, fold, stream);
    else lean == 2 ? enqueue_attention_form<2>(v, fold, stream) : lean == 1 ? enqueue_attention_form<1>(v, fold, stream) : enqueue_attention_form<0>(v, fold, stream);
}

// ---- several requests' segments per launch (BatchForward): the per-call checks and grids, flushed at kMaxSegments ----
namespace {
template <class View, class Launch>
void for_segments(std::span<const View> views, Launch&& launch, const auto& grid_of) {
    SegmentBatch<View> b;
    for (const auto& v : views) {
        const dim3 g = grid_of(v);
        if (g.x * g.y * g.z == 0U) continue;
        b.add(v, g);
        if (b.full()) { launch(b); b.clear(); }
    }
    if (!b.empty()) launch(b);
}
}  // namespace

void enqueue_window_write_segments(std::span<const WindowView> views, hipStream_t stream) {
    for (const auto& v : views) check_window(v);
    for_segments(views, [&](const SegmentBatch<WindowView>& b) {
        hipLaunchKernelGGL(window_write_seg_kernel, dim3(b.blocks()), dim3(32), 0, stream, b.s);
    }, [](const WindowView& v) { return dim3(v.rows); });
    check(hipGetLastError(), "ds41 window write (segments)");
}
void enqueue_compress_segments(std::span<const CompressView> views, hipStream_t stream) {
    for (const auto& v : views) check_compress(v);
    const int form = compress_form();
    for_segments(views, [&](const SegmentBatch<CompressView>& b) {
        if (form == 2) hipLaunchKernelGGL(HIP_KERNEL_NAME(compress_seg_kernel<false, true>), dim3(b.blocks()), dim3(128), 0, stream, b.s);
        else if (form == 1) hipLaunchKernelGGL(HIP_KERNEL_NAME(compress_seg_kernel<true, false>), dim3(b.blocks()), dim3(128), 0, stream, b.s);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(compress_seg_kernel<false, false>), dim3(b.blocks()), dim3(128), 0, stream, b.s);
    }, [](const CompressView& v) { return dim3(v.rows); });
    check(hipGetLastError(), "ds41 compress (segments)");
}
void enqueue_index_scores_segments(std::span<const IndexView> views, hipStream_t stream) {
    for (const auto& v : views) if (v.max_count != 0U) check_index(v); else check_rows(v.rows);
    for_segments(views, [&](const SegmentBatch<IndexView>& b) {
        hipLaunchKernelGGL(index_qprep_seg_kernel, dim3(b.blocks()), dim3(256), 0, stream, b.s);
    }, [](const IndexView& v) { return v.max_count == 0U ? dim3(0) : dim3(v.rows); });
    for_segments(views, [&](const SegmentBatch<IndexView>& b) {
        hipLaunchKernelGGL(index_scores_seg_kernel, dim3(b.blocks()), dim3(256), 0, stream, b.s);
    }, [](const IndexView& v) { return dim3((v.max_count + 255U) / 256U); });
    check(hipGetLastError(), "ds41 index scores (segments)");
}
void enqueue_select_direct_segments(std::span<const SelectView> views, hipStream_t stream) {
    for (const auto& v : views) check_select_direct(v);
    for_segments(views, [&](const SegmentBatch<SelectView>& b) {
        hipLaunchKernelGGL(select_direct_seg_kernel, dim3(b.blocks()), dim3(256), 0, stream, b.s);
    }, [](const SelectView& v) { return dim3(v.rows); });
    check(hipGetLastError(), "ds41 select direct (segments)");
}
template <bool Legacy>
void launch_select_segments(std::span<const SelectView> views, hipStream_t stream) {
    for (const auto& v : views) (void)select_slices(v);
    for_segments(views, [&](const SegmentBatch<SelectView>& b) {
        hipLaunchKernelGGL(select_seg_kernel<Legacy>, dim3(b.blocks()), dim3(kSelThreads), 0, stream, b.s);
    }, [](const SelectView& v) { return select_slices(v) == 0U ? dim3(v.rows) : dim3(0); });
    for_segments(views, [&](const SegmentBatch<SelectView>& b) {
        hipLaunchKernelGGL(select_slice_seg_kernel<Legacy>, dim3(b.blocks()), dim3(kSelThreads), 0, stream, b.s);
    }, [](const SelectView& v) { const unsigned s = select_slices(v); return s == 0U ? dim3(0) : dim3(s, v.rows); });
    for_segments(views, [&](const SegmentBatch<SelectView>& b) {
        hipLaunchKernelGGL(select_merge_seg_kernel<Legacy>, dim3(b.blocks()), dim3(kSelThreads), 0, stream, b.s);
    }, [](const SelectView& v) { return select_slices(v) == 0U ? dim3(0) : select_merge_grid(v); });
    check(hipGetLastError(), "ds41 select (segments)");
}
void enqueue_select_segments(std::span<const SelectView> views, hipStream_t stream) {
    if (select_v1()) launch_select_segments<true>(views, stream);
    else launch_select_segments<false>(views, stream);
}
// STRIX_DS41_ATTN_SEG_HPW: one form for every segment, attention_hpw's rule over the blocks of all of them (0: off)
static unsigned shared_segment_hpw(std::span<const AttentionView> views) {
    if (!attention_segment_hpw()) return 0U;
    for (const unsigned hpw : {4U, 2U}) {
        unsigned blocks = 0;
        bool fits = true;
        for (const auto& v : views) {
            fits = fits && v.heads % (kAttnMinBlockHeads * hpw) == 0U;
            blocks += attention_chunks(v) * (v.heads / (kAttnMinBlockHeads * hpw)) * v.rows;
        }
        if (fits && blocks >= attention_min_blocks()) return hpw;
    }
    return 1U;
}
template <bool Swizzle, bool Half, bool Fold, unsigned Lean>
static void launch_attention_segments(std::span<const AttentionView> views, hipStream_t stream) {
    for (const auto& v : views) (void)attention_hpw(v);  // the per-call checks
    const unsigned shared = shared_segment_hpw(views);
    bool cap = Lean >= 3U;  // the 6-wave form when every segment takes it
    for (const auto& v : views) cap = cap && attention_wave_cap(v);
    for (const unsigned hpw : {4U, 2U, 1U}) {  // each segment keeps its own heads per wave (its kernel form), or the shared one
        const auto grid = [hpw, shared](const AttentionView& v) {
            return (shared != 0U ? shared : attention_hpw(v)) == hpw ? dim3(attention_chunks(v), v.heads / (kAttnMinBlockHeads * hpw), v.rows) : dim3(0);
        };
        for_segments(views, [&](const SegmentBatch<AttentionView>& b) {
            if (hpw == 4U && cap)
                hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_chunk_seg_kernel<4, Swizzle, false, Fold, Lean, true>), dim3(b.blocks()), dim3(kAttnThreads), 0, stream, b.s);
            else if (hpw == 4U) hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_chunk_seg_kernel<4, Swizzle, false, Fold, Lean>), dim3(b.blocks()), dim3(kAttnThreads), 0, stream, b.s);
            else if (hpw == 2U) hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_chunk_seg_kernel<2, Swizzle, Half, Fold, Lean>), dim3(b.blocks()), dim3(kAttnThreads), 0, stream, b.s);
            else hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_chunk_seg_kernel<1, Swizzle, Half, Fold, Lean>), dim3(b.blocks()), dim3(kAttnThreads), 0, stream, b.s);
        }, grid);
    }
    for_segments(views, [&](const SegmentBatch<AttentionView>& b) {
        hipLaunchKernelGGL(attention_combine_seg_kernel, dim3(b.blocks()), dim3(32), 0, stream, b.s);
    }, [](const AttentionView& v) { return v.tickets == nullptr ? dim3(v.heads, v.rows) : dim3(0); });
    check(hipGetLastError(), "ds41 attention (segments)");
}
template <unsigned Lean>
static void enqueue_attention_segments_form(std::span<const AttentionView> views, bool fold, hipStream_t stream) {
    if (attention_lds_swizzle() && !fold)  // as enqueue_attention: the f32 tiles only, the tile choice unchanged
        attention_tile_f16() ? launch_attention_segments<true, true, false, Lean>(views, stream) : launch_attention_segments<true, false, false, Lean>(views, stream);
    else if (attention_tile_f16()) fold ? launch_attention_segments<false, true, true, Lean>(views, stream) : launch_attention_segments<false, true, false, Lean>(views, stream);
    else fold ? launch_attention_segments<false, false, true, Lean>(views, stream) : launch_attention_segments<false, false, false, Lean>(views, stream);
}
void enqueue_attention_segments(std::span<const AttentionView> views, hipStream_t stream) {
    if (views.empty()) return;
    const bool fold = views[0].kv != nullptr;
    for (const auto& v : views) {
        check_fold(v);
        if ((v.kv != nullptr) != fold) throw std::invalid_argument("ds41 attention segments: window fold on some segments only");
    }
    const int lean = attention_lean();
    if (lean == 3) enqueue_attention_segments_form<3>(views, fold, stream);
    else lean == 2 ? enqueue_attention_segments_form<2>(views, fold, stream) : lean == 1 ? enqueue_attention_segments_form<1>(views, fold, stream)
                   : enqueue_attention_segments_form<0>(views, fold, stream);
}

}  // namespace strix::ds41::attn::hip
