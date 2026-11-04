#pragma once
// GLM-5.3-Flash engine: declared arithmetic shared by the host emulation (Tier A reference) and the device kernels
// (design notes section 1, approved arithmetic contract). Every function here is __host__ __device__ and uses basic IEEE
// binary32 operations only; compile with -ffp-contract=off -fno-fast-math (device: -fno-gpu-flush-denormals-to-zero).
//   declared_exp / declared_sqrt: copied verbatim from DS ds41_moe.hpp (Cody-Waite + degree-7 Horner, exp(x) = +inf
//   above 88 and +0 below -103; sqrt correctly rounded, through double on the device).
//   Reductions: R1 = ascending fmaf chain; R2 = lane-strided fmaf partials over L lanes + pairwise halving tree.
#include "../rtxpf/import/glm_dims.hpp"  // the model dimensions (shared with the RTX import state layout)
#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__)
#define STRIX_GLM_HD __host__ __device__
#else
#define STRIX_GLM_HD
#endif

namespace strix::glm {

// ---------------------------------------------------------------- model constants (checked against config.json at load)
inline constexpr float kEps = 1e-5F, kHcEps = 1e-6F, kL2Eps = 1e-6F, kLnEps = 1e-6F, kGateLow = -5.0F;
inline constexpr float kClamp = 10.0F, kRouteScale = 2.5F, kRouteDenEps = 1e-20F;
inline constexpr float kInvSqrt128 = 0.088388347648318F;   // RN32(128^-1/2): KDA q scale, indexer score scale
inline constexpr float kInvSqrt32 = 0.17677669529663687F;  // RN32(32^-1/2): indexer head weights
inline constexpr float kMlaScale = 0.0625F;                // 256^-1/2 (exact)

constexpr bool is_kda_layer(unsigned l) { return l < kLayers && l % 4 != 3; }
constexpr bool is_moe_layer(unsigned l) { return l >= kFirstMoe; }

// ---------------------------------------------------------------- transcendentals (DS ds41_moe.hpp, verbatim)
namespace detail {
STRIX_GLM_HD inline float pow2(int k) { return __builtin_bit_cast(float, static_cast<std::uint32_t>(k + 127) << 23U); }
}  // namespace detail

STRIX_GLM_HD inline float declared_exp(float x) {
    if (!(x == x)) return x;
    if (x > 88.0F) return __builtin_huge_valf();
    if (x < -103.0F) return 0.0F;
    const float n = __builtin_rintf(x * 1.44269504F);
    const float r = (x - n * 0.693359375F) - n * -2.12194440e-4F;
    float p = 1.98412698e-4F;
    p = p * r + 1.38888889e-3F;
    p = p * r + 8.33333333e-3F;
    p = p * r + 4.16666667e-2F;
    p = p * r + 1.66666667e-1F;
    p = p * r + 0.5F;
    p = p * r + 1.0F;
    p = p * r + 1.0F;
    const int k = static_cast<int>(n), k1 = k / 2, k2 = k - k1;
    return (p * detail::pow2(k1)) * detail::pow2(k2);
}
STRIX_GLM_HD inline float declared_sqrt(float x) {
#if defined(__HIP_DEVICE_COMPILE__)
    return static_cast<float>(__dsqrt_rn(static_cast<double>(x)));
#else
    return __builtin_sqrtf(x);
#endif
}
// sigmoid saturates by IEEE rules: exp(-x) = +inf -> 1 / inf = +0; exp(-x) = +0 -> 1.
STRIX_GLM_HD inline float sigmoid(float x) { return 1.0F / (1.0F + declared_exp(-x)); }
STRIX_GLM_HD inline float silu(float x) { return x * sigmoid(x); }
STRIX_GLM_HD inline float inv_sqrt(float x) { return 1.0F / declared_sqrt(x); }  // D8: no rsqrt op

// ---------------------------------------------------------------- formats
STRIX_GLM_HD inline float f16_to_float(std::uint16_t h) {  // exact (DS ds41_layer.hpp)
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16U;
    const std::uint32_t exponent = (h >> 10U) & 0x1fU, mantissa = h & 0x3ffU;
    if (exponent == 0x1fU) return __builtin_bit_cast(float, sign | 0x7f800000U | (mantissa << 13U));
    if (exponent == 0) {
        if (mantissa == 0) return __builtin_bit_cast(float, sign);
        const float v = static_cast<float>(mantissa) * 5.9604644775390625e-8F;
        return sign ? -v : v;
    }
    return __builtin_bit_cast(float, sign | ((exponent + 112U) << 23U) | (mantissa << 13U));
}
STRIX_GLM_HD inline float bf16_to_float(std::uint16_t b) { return __builtin_bit_cast(float, static_cast<std::uint32_t>(b) << 16U); }
// RNE to bfloat16 of a finite float (the cache write site C3 checks the result for +-inf afterwards).
STRIX_GLM_HD inline std::uint16_t float_to_bf16(float x) {
    const auto u = __builtin_bit_cast(std::uint32_t, x);
    return static_cast<std::uint16_t>((u + 0x7fffU + ((u >> 16U) & 1U)) >> 16U);
}
STRIX_GLM_HD inline bool is_finite(float x) { return (__builtin_bit_cast(std::uint32_t, x) & 0x7f800000U) != 0x7f800000U; }
STRIX_GLM_HD inline bool bf16_is_finite(std::uint16_t b) { return (b & 0x7f80U) != 0x7f80U; }

// R5 ordered key of a finite score: larger score -> larger key; -0 maps to +0.
STRIX_GLM_HD inline std::uint32_t ordered_key(float s) {
    std::uint32_t u = __builtin_bit_cast(std::uint32_t, s);
    if (u == 0x80000000U) u = 0;
    return (u & 0x80000000U) ? ~u : (u | 0x80000000U);
}

// ---------------------------------------------------------------- host emulation of the lane reductions
// R1: acc = +0; acc = fmaf(a(i), b(i), acc) for i ascending.
template <class A, class B>
STRIX_GLM_HD inline float chain_dot(A a, B b, unsigned n) {
    float acc = 0.0F;
    for (unsigned i = 0; i < n; ++i) acc = __builtin_fmaf(a(i), b(i), acc);
    return acc;
}
// R1b (default generated identity; default prompts retain R1): n / kR1bBlock
// contiguous blocks, block j: p_j = +0; p_j = fmaf(a(i), b(i), p_j) for i = j*256 .. j*256+255 ascending; then the
// pairwise tree p_i = p_i + p_{i+h} for h = nb/2 .. 1 (router K 4096: 16 blocks, offsets 8, 4, 2, 1). nb is a power of
// two and n % 256 == 0 (checked by the callers' allowlist).
inline constexpr unsigned kR1bBlock = 256, kR1bMaxBlocks = 64;
template <class A, class B>
STRIX_GLM_HD inline float r1b_dot(A a, B b, unsigned n) {
    float p[kR1bMaxBlocks];
    const unsigned nb = n / kR1bBlock;
    for (unsigned j = 0; j < nb; ++j) {
        float acc = 0.0F;
        for (unsigned i = j * kR1bBlock; i < (j + 1) * kR1bBlock; ++i) acc = __builtin_fmaf(a(i), b(i), acc);
        p[j] = acc;
    }
    for (unsigned h = nb / 2; h >= 1; h /= 2)
        for (unsigned i = 0; i < h; ++i) p[i] = p[i] + p[i + h];
    return p[0];
}
// Arithmetic identity of the long few-output chains (router now; mHC fn, KDA f/g/b with exchange): R1 = the declared
// arithmetic contract chain (default prompts), R1b = the blocked form above (default generated forwards). Receipts name the identity.
enum class Identity { kR1, kR1b };
inline const char* identity_name(Identity i) { return i == Identity::kR1b ? "r1b" : "r1"; }
// R2 (host form): L lanes; lane l accumulates term(i) for i = l, l + L, ... with op; then the pairwise halving tree
// s[l] = s[l] + s[l + h], h = L/2 ... 1. fma_sq: term is fmaf(x, x, s) (sum of squares); otherwise s = s + x.
template <unsigned L, class X>
inline float tree_sumsq(X x, unsigned n) {
    float s[L];
    for (unsigned l = 0; l < L; ++l) {
        float a = 0.0F;
        for (unsigned i = l; i < n; i += L) {
            const float v = x(i);
            a = __builtin_fmaf(v, v, a);
        }
        s[l] = a;
    }
    for (unsigned h = L / 2; h > 0; h >>= 1U)
        for (unsigned l = 0; l < h; ++l) s[l] = s[l] + s[l + h];
    return s[0];
}
template <unsigned L, class X>
inline float tree_sum(X x, unsigned n) {
    float s[L];
    for (unsigned l = 0; l < L; ++l) {
        float a = 0.0F;
        for (unsigned i = l; i < n; i += L) a = a + x(i);
        s[l] = a;
    }
    for (unsigned h = L / 2; h > 0; h >>= 1U)
        for (unsigned l = 0; l < h; ++l) s[l] = s[l] + s[l + h];
    return s[0];
}
// Lane-strided fmaf accumulation of p(t) * c(t) (R13 output), same tree.
template <unsigned L, class P, class C>
inline float tree_fma(P p, C c, unsigned n) {
    float s[L];
    for (unsigned l = 0; l < L; ++l) {
        float a = 0.0F;
        for (unsigned t = l; t < n; t += L) a = __builtin_fmaf(p(t), c(t), a);
        s[l] = a;
    }
    for (unsigned h = L / 2; h > 0; h >>= 1U)
        for (unsigned l = 0; l < h; ++l) s[l] = s[l] + s[l + h];
    return s[0];
}
// Lane count of R2 for an n-element norm.
constexpr unsigned norm_lanes(unsigned n) { return n >= 512 ? 256 : 128; }

}  // namespace strix::glm
