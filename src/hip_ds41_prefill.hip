#include "strix/ds41_prefill_tuning.hpp"
#pragma clang fp contract(off)
#include "strix/hip_ds41_prefill.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "strix/ds41_moe.hpp"
#include "strix/ds41_prefill.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace strix::ds41 {
namespace {

void check(hipError_t status, const char* what) {
    if (status != hipSuccess) throw std::runtime_error(std::string("ds41 prefill: ") + what + ": " + hipGetErrorString(status));
}

// ---- Format helpers (the decode kernels' integer forms, hip_ds41_formats.hip) ----

__device__ const unsigned long long prefill_iq2_grid[256] = {
#include "strix/ds41_iq2_xxs_grid.inc"
};
__device__ __forceinline__ float half_bits(unsigned short bits) { return __half2float(__ushort_as_half(bits)); }
__device__ __forceinline__ unsigned short load_u16(const unsigned char* p) {
    unsigned short v;
    __builtin_memcpy(&v, p, 2);
    return v;
}
__device__ __forceinline__ unsigned long long load_u64(const unsigned char* p) {
    unsigned long long v;
    __builtin_memcpy(&v, p, 8);
    return v;
}
__device__ __forceinline__ float e8m0_half(unsigned code) {
    if (code == 255U) return __uint_as_float(0x7fc00000U);
    return __uint_as_float(code < 2U ? 0x00200000U << code : (code - 1U) << 23U);
}
__device__ __forceinline__ unsigned byte_masks(unsigned bits4) { return ((bits4 * 0x00204081U) & 0x01010101U) * 0xffU; }
__device__ __forceinline__ unsigned sign_byte(unsigned index) {
    index &= 127U;
    return index | ((static_cast<unsigned>(__builtin_popcount(index)) & 1U) << 7U);
}
__device__ __forceinline__ int apply_signs(unsigned magnitudes, unsigned mask) {
    return static_cast<int>((magnitudes ^ mask) + (mask & 0x01010101U));
}
__device__ __forceinline__ int e2m1_bytes(unsigned nibbles) {
    const unsigned index = nibbles & 0x07070707U, negative = ((nibbles >> 3U) & 0x01010101U) * 0xffU;
    const unsigned positive_values = __builtin_amdgcn_perm(0x0c080604U, 0x03020100U, index);
    const unsigned negative_values = __builtin_amdgcn_perm(0xf4f8fafcU, 0xfdfeff00U, index);
    return static_cast<int>((positive_values & ~negative) | (negative_values & negative));
}

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
constexpr int dot_magic = 0x4b400000; // 1.5 * 2^23: bits + r are the float 1.5 * 2^23 + r for |r| < 2^22
__device__ __forceinline__ float magic_float(int bits) { return __int_as_float(bits) - 0x1.8p23F; }
// Exact int32 dot of 32 values for every (row, output) of a 16 x 16 tile, as magic-offset bits.
__device__ __forceinline__ v8i group_dot(v4i a0, v4i b0, v4i a1, v4i b1) {
    const v8i magic{dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic};
    const v8i c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, true, b0, magic, false);
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, true, b1, c, false);
}

// Raw bytes of group g of one weight row, loaded ahead of their decoding.
template <WeightFormat Format> struct Raw;
template <> struct Raw<WeightFormat::iq2_xxs> { unsigned short d; unsigned long long aux; };
template <> struct Raw<WeightFormat::q2_k> { unsigned q[8]; unsigned short scales; unsigned dm; unsigned shift; };
template <> struct Raw<WeightFormat::mxfp4> { unsigned e; unsigned q[4]; };
template <> struct Raw<WeightFormat::q8_0> { unsigned short d; unsigned q[8]; };

template <WeightFormat Format>
__device__ __forceinline__ Raw<Format> load_raw(const unsigned char* row, unsigned g) {
    Raw<Format> raw;
    if constexpr (Format == WeightFormat::iq2_xxs) {
        const auto* block = row + g / 8U * iq2_xxs_block_bytes;
        raw.d = load_u16(block);
        raw.aux = load_u64(block + 2U + 8U * (g % 8U));
    } else if constexpr (Format == WeightFormat::q2_k) {
        const auto* block = row + g / 8U * q2_k_block_bytes;
        const unsigned group = g % 8U, half = group / 4U;
        __builtin_memcpy(raw.q, block + 16U + 32U * half, 32);
        raw.scales = load_u16(block + 8U * half + 2U * (group % 4U));
        __builtin_memcpy(&raw.dm, block + 80U, 4);
        raw.shift = 2U * (group % 4U);
    } else if constexpr (Format == WeightFormat::q8_0) {
        const auto* block = row + g * q8_0_block_bytes;
        raw.d = load_u16(block);
        __builtin_memcpy(raw.q, block + 2, 32);
    } else {
        const auto* block = row + g * mxfp4_block_bytes;
        raw.e = block[0];
        __builtin_memcpy(raw.q, block + 1, 16);
    }
    return raw;
}
// B operands of one group (codes 0-15, 16-31) and its scale terms. For Q2_K the codes are
// pre-multiplied by (scale_h & 15) and m0/m1 are the half minimum factors (scale_h >> 4).
struct Operands { v4i b0, b1; float scale, d, dmin; int m0, m1; unsigned dm_bits, sc_bits; };
__device__ __forceinline__ Operands decode(const Raw<WeightFormat::iq2_xxs>& raw, const unsigned long long* grid) {
    Operands out{};
    const float d = half_bits(raw.d);
    const unsigned aux0 = static_cast<unsigned>(raw.aux), aux1 = static_cast<unsigned>(raw.aux >> 32U);
    out.scale = d * (0.5F + static_cast<float>(aux1 >> 28U)) * 0.25F;
    int w[8];
#pragma unroll
    for (unsigned l = 0; l < 4U; ++l) {
        const unsigned long long g = grid[(aux0 >> (8U * l)) & 0xffU];
        const unsigned signs = sign_byte(aux1 >> (7U * l));
        w[2U * l] = apply_signs(static_cast<unsigned>(g), byte_masks(signs & 15U));
        w[2U * l + 1U] = apply_signs(static_cast<unsigned>(g >> 32U), byte_masks(signs >> 4U));
    }
    out.b0 = v4i{w[0], w[1], w[2], w[3]};
    out.b1 = v4i{w[4], w[5], w[6], w[7]};
    return out;
}
__device__ __forceinline__ Operands decode(const Raw<WeightFormat::mxfp4>& raw, const unsigned long long*) {
    Operands out{};
    out.scale = e8m0_half(raw.e);
    out.b0 = v4i{e2m1_bytes(raw.q[0] & 0x0f0f0f0fU), e2m1_bytes(raw.q[1] & 0x0f0f0f0fU), e2m1_bytes(raw.q[2] & 0x0f0f0f0fU),
                 e2m1_bytes(raw.q[3] & 0x0f0f0f0fU)};
    out.b1 = v4i{e2m1_bytes((raw.q[0] >> 4U) & 0x0f0f0f0fU), e2m1_bytes((raw.q[1] >> 4U) & 0x0f0f0f0fU),
                 e2m1_bytes((raw.q[2] >> 4U) & 0x0f0f0f0fU), e2m1_bytes((raw.q[3] >> 4U) & 0x0f0f0f0fU)};
    return out;
}
__device__ __forceinline__ Operands decode(const Raw<WeightFormat::q8_0>& raw, const unsigned long long*) {
    Operands out{};
    out.scale = half_bits(raw.d);
    out.b0 = v4i{static_cast<int>(raw.q[0]), static_cast<int>(raw.q[1]), static_cast<int>(raw.q[2]), static_cast<int>(raw.q[3])};
    out.b1 = v4i{static_cast<int>(raw.q[4]), static_cast<int>(raw.q[5]), static_cast<int>(raw.q[6]), static_cast<int>(raw.q[7])};
    return out;
}
__device__ __forceinline__ Operands decode(const Raw<WeightFormat::q2_k>& raw, const unsigned long long*) {
    Operands out{};
    const unsigned sc0 = raw.scales & 0xffU, sc1 = raw.scales >> 8U;
    const unsigned f0 = sc0 & 15U, f1 = sc1 & 15U;
    // Codes <= 3 times factors <= 15 stay below 256 in every byte: no carries between bytes.
    out.b0 = v4i{static_cast<int>(((raw.q[0] >> raw.shift) & 0x03030303U) * f0), static_cast<int>(((raw.q[1] >> raw.shift) & 0x03030303U) * f0),
                 static_cast<int>(((raw.q[2] >> raw.shift) & 0x03030303U) * f0), static_cast<int>(((raw.q[3] >> raw.shift) & 0x03030303U) * f0)};
    out.b1 = v4i{static_cast<int>(((raw.q[4] >> raw.shift) & 0x03030303U) * f1), static_cast<int>(((raw.q[5] >> raw.shift) & 0x03030303U) * f1),
                 static_cast<int>(((raw.q[6] >> raw.shift) & 0x03030303U) * f1), static_cast<int>(((raw.q[7] >> raw.shift) & 0x03030303U) * f1)};
    out.m0 = static_cast<int>(sc0 >> 4U);
    out.m1 = static_cast<int>(sc1 >> 4U);
    out.d = half_bits(static_cast<unsigned short>(raw.dm & 0xffffU));
    out.dmin = half_bits(static_cast<unsigned short>(raw.dm >> 16U));
    out.dm_bits = raw.dm;
    out.sc_bits = raw.scales;
    return out;
}

// Lean forms of the IQ2_XXS and Q2_K decoders: the same integers with full-rate instructions instead of the
// quarter-rate v_mul_lo_u32. IQ2_XXS: the byte masks m * 255 (bytes of m are 0 or 1) come from one v_perm_b32. Q2_K: every
// byte of codes * factor is at most 3 * 15 = 45, so the packed product equals a byte lookup of the codes (0..3) in
// the table {0, f, 2f, 3f} (v_perm_b32 selector bytes 0-3 pick bytes of the second operand).
__device__ __forceinline__ unsigned byte_masks_lean(unsigned bits4) {
    // v_perm_b32 selector byte 0x0c gives 0x00 and 0x0d gives 0xff: bytes m_j + 12 select the mask byte directly.
    const unsigned m = (bits4 * 0x00204081U) & 0x01010101U;
    return __builtin_amdgcn_perm(0U, 0U, m + 0x0c0c0c0cU);
}
__device__ __forceinline__ Operands decode_lean(const Raw<WeightFormat::iq2_xxs>& raw, const unsigned long long* grid) {
    Operands out{};
    const float d = half_bits(raw.d);
    const unsigned aux0 = static_cast<unsigned>(raw.aux), aux1 = static_cast<unsigned>(raw.aux >> 32U);
    out.scale = d * (0.5F + static_cast<float>(aux1 >> 28U)) * 0.25F;
    int w[8];
#pragma unroll
    for (unsigned l = 0; l < 4U; ++l) {
        const unsigned long long g = grid[(aux0 >> (8U * l)) & 0xffU];
        const unsigned signs = sign_byte(aux1 >> (7U * l));
        w[2U * l] = apply_signs(static_cast<unsigned>(g), byte_masks_lean(signs & 15U));
        w[2U * l + 1U] = apply_signs(static_cast<unsigned>(g >> 32U), byte_masks_lean(signs >> 4U));
    }
    out.b0 = v4i{w[0], w[1], w[2], w[3]};
    out.b1 = v4i{w[4], w[5], w[6], w[7]};
    return out;
}
__device__ __forceinline__ Operands decode_lean(const Raw<WeightFormat::q2_k>& raw, const unsigned long long*) {
    Operands out{};
    const unsigned sc0 = raw.scales & 0xffU, sc1 = raw.scales >> 8U;
    const unsigned t0 = ((sc0 & 15U) * 0x00030201U) << 8U, t1 = ((sc1 & 15U) * 0x00030201U) << 8U;  // v_mul_u32_u24 (operands < 2^24)
    const auto mul = [&](unsigned q, unsigned t) {
        return static_cast<int>(__builtin_amdgcn_perm(t, t, (q >> raw.shift) & 0x03030303U));
    };
    out.b0 = v4i{mul(raw.q[0], t0), mul(raw.q[1], t0), mul(raw.q[2], t0), mul(raw.q[3], t0)};
    out.b1 = v4i{mul(raw.q[4], t1), mul(raw.q[5], t1), mul(raw.q[6], t1), mul(raw.q[7], t1)};
    out.m0 = static_cast<int>(sc0 >> 4U);
    out.m1 = static_cast<int>(sc1 >> 4U);
    out.d = half_bits(static_cast<unsigned short>(raw.dm & 0xffffU));
    out.dmin = half_bits(static_cast<unsigned short>(raw.dm >> 16U));
    out.dm_bits = raw.dm;
    out.sc_bits = raw.scales;
    return out;
}
// IQ2_XXS with a sign table (form 2): entry i (the 7-bit sign index of a grid lookup) holds the masks of
// byte_masks(sign_byte(i) & 15) and of (sign_byte(i) >> 4), each with its mask & 0x01010101, so apply_signs is one
// v_xad per word: the same integers as decode().
struct SignEntry { unsigned lo, lo_inc, hi, hi_inc; };
__device__ __forceinline__ void load_signs(SignEntry* signs) {
    for (unsigned t = threadIdx.x; t < 128U; t += blockDim.x) {
        const unsigned b = sign_byte(t), lo = byte_masks(b & 15U), hi = byte_masks(b >> 4U);
        signs[t] = SignEntry{lo, lo & 0x01010101U, hi, hi & 0x01010101U};
    }
}
__device__ __forceinline__ Operands decode_signs(const Raw<WeightFormat::iq2_xxs>& raw, const unsigned long long* grid,
                                                 const SignEntry* signs) {
    Operands out{};
    const float d = half_bits(raw.d);
    const unsigned aux0 = static_cast<unsigned>(raw.aux), aux1 = static_cast<unsigned>(raw.aux >> 32U);
    out.scale = d * (0.5F + static_cast<float>(aux1 >> 28U)) * 0.25F;
    int w[8];
#pragma unroll
    for (unsigned l = 0; l < 4U; ++l) {
        const unsigned long long g = grid[(aux0 >> (8U * l)) & 0xffU];
        const SignEntry e = signs[(aux1 >> (7U * l)) & 127U];
        w[2U * l] = static_cast<int>((static_cast<unsigned>(g) ^ e.lo) + e.lo_inc);
        w[2U * l + 1U] = static_cast<int>((static_cast<unsigned>(g >> 32U) ^ e.hi) + e.hi_inc);
    }
    out.b0 = v4i{w[0], w[1], w[2], w[3]};
    out.b1 = v4i{w[4], w[5], w[6], w[7]};
    return out;
}
// Prefill decoder forms: 0 = decode(), 1 = decode_lean(), 2 = decode_lean() with the IQ2_XXS sign table.
template <int Form, WeightFormat Format>
__device__ __forceinline__ Operands decode_form(const Raw<Format>& raw, const unsigned long long* grid, const SignEntry* signs) {
    if constexpr (Form == 2 && Format == WeightFormat::iq2_xxs) return decode_signs(raw, grid, signs);
    else if constexpr (Form >= 1 && (Format == WeightFormat::iq2_xxs || Format == WeightFormat::q2_k)) return decode_lean(raw, grid);
    else return decode(raw, grid);
}

// ---- Staged activations ----
//
// The 32 lane classes of the declared product are visited in bit-reversed order i -> l = rev5(i),
// eight per stage; a stage holds, for every batch row, the codes of the classes' groups
// l + 32 j (j < J), their deltas and (Q2_K) the sums of each 16-code half. Group slot q = u * J + j
// for the class of stage position u. Within a 16-row tile, row r sits at (r & 1) * 8 + r / 2 in the
// delta / sum arrays, so D element e of half-wave h (row 2e + h) reads position 8h + e.
#ifndef STRIX_DS41_PREFILL_STAGE_CLASSES
#define STRIX_DS41_PREFILL_STAGE_CLASSES 8
#endif
constexpr unsigned stage_classes = STRIX_DS41_PREFILL_STAGE_CLASSES;
constexpr unsigned max_groups_per_class = 5U; // widths up to 5120
__device__ __forceinline__ unsigned class_of(unsigned i) { return __builtin_bitreverse32(i) >> 27U; }
__device__ __forceinline__ unsigned tile_position(unsigned r) { return (r & ~15U) | ((r & 1U) << 3U) | ((r & 15U) >> 1U); }

struct StageLayout {
    unsigned rows, groups_per_class, slots; // rows = 16 * MT, slots = 8 * J
    __device__ __host__ unsigned code_bytes() const { return slots * 2U * rows * 16U; }
    __device__ __host__ unsigned delta_bytes() const { return slots * rows * 4U; }
    __device__ __host__ unsigned sum_bytes(bool q2) const { return q2 ? slots * 2U * rows * 4U : 0U; }
    __device__ __host__ unsigned bytes(bool q2) const { return code_bytes() + delta_bytes() + sum_bytes(q2); }
};

template <bool Q2>
__device__ __forceinline__ void stage_activations(unsigned char* lds, const StageLayout& s, unsigned stage, unsigned width,
                                                  const unsigned* row_of, unsigned count, const signed char* codes,
                                                  const float* deltas) {
    const unsigned groups = width / 32U;
    auto* a = reinterpret_cast<v4i*>(lds);
    auto* d = reinterpret_cast<float*>(lds + s.code_bytes());
    auto* sums = reinterpret_cast<int*>(lds + s.code_bytes() + s.delta_bytes());
    const unsigned items = s.slots * 2U * s.rows;
    for (unsigned item = threadIdx.x; item < items; item += blockDim.x) {
        const unsigned r = item % s.rows, kh = (item / s.rows) % 2U, q = item / (2U * s.rows);
        const unsigned u = q / s.groups_per_class, j = q % s.groups_per_class;
        const unsigned g = class_of(stage * stage_classes + u) + 32U * j;
        if (g >= groups) continue;
        v4i v{0, 0, 0, 0};
        const bool live = r < count;
        if (live) v = *reinterpret_cast<const v4i*>(codes + static_cast<unsigned long long>(row_of[r]) * width + 32U * g + 16U * kh);
        a[(2U * q + kh) * s.rows + r] = v;
        const unsigned at = tile_position(r);
        if (kh == 0U) d[q * s.rows + at] = live ? deltas[static_cast<unsigned long long>(row_of[r]) * groups + g] : 0.0F;
        if constexpr (Q2) {
            const int ones = 0x01010101;
            int sum = 0;
            sum = __builtin_amdgcn_sudot4(false, ones, true, v.x, sum, false);
            sum = __builtin_amdgcn_sudot4(false, ones, true, v.y, sum, false);
            sum = __builtin_amdgcn_sudot4(false, ones, true, v.z, sum, false);
            sum = __builtin_amdgcn_sudot4(false, ones, true, v.w, sum, false);
            sums[(2U * q + kh) * s.rows + at] = sum;
        }
    }
}

template <unsigned MT>
__device__ __forceinline__ void add_level(const float (&left)[MT][8], float (&p)[MT][8]) {
#pragma unroll
    for (unsigned t = 0; t < MT; ++t)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) p[t][e] = left[t][e] + p[t][e];
}
template <unsigned MT>
__device__ __forceinline__ void copy_level(const float (&from)[MT][8], float (&to)[MT][8]) {
#pragma unroll
    for (unsigned t = 0; t < MT; ++t)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) to[t][e] = from[t][e];
}

// The wave's 16 outputs x MT row tiles of one weight matrix: v[t][e] for (row 16t + 2e + h, output
// lane % 16). Every workgroup thread must call it (it stages and synchronizes).
template <WeightFormat Format, unsigned MT, unsigned J>
__device__ __forceinline__ void wave_products(unsigned char* lds, const StageLayout& s, const unsigned char* weights_row,
                                              unsigned width, const unsigned* row_of, unsigned count, const signed char* codes,
                                              const float* deltas, const unsigned long long* grid, float (&v)[MT][8]) {
    constexpr bool q2 = Format == WeightFormat::q2_k;
    const unsigned lane = threadIdx.x % 32U, i16 = lane & 15U, h = lane >> 4U, groups = width / 32U;
    const auto* a = reinterpret_cast<const v4i*>(lds);
    const auto* d = reinterpret_cast<const float*>(lds + s.code_bytes());
    const auto* sums = reinterpret_cast<const int*>(lds + s.code_bytes() + s.delta_bytes());
    // Carry stack of the butterfly tree: stack[k] holds a finished subtree of 2^k classes.
    float stack[5][MT][8];
#pragma unroll
    for (unsigned k = 0; k < 5U; ++k)
#pragma unroll
        for (unsigned t = 0; t < MT; ++t)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) stack[k][t][e] = 0.0F;
    // Weight pipeline: raws[j] holds group j of the next class (in visiting order) that has a
    // group j; once it is decoded, the same group of the following such class is requested into the
    // slot, so up to J loads stay in flight.
    const auto next_with = [&](unsigned i, unsigned j) {
        while (i < 32U && class_of(i) + 32U * j >= groups) ++i;
        return i;
    };
    Raw<Format> raws[J];
#pragma unroll
    for (unsigned j = 0; j < J; ++j) {
        const unsigned first = next_with(0U, j);
        if (first < 32U) raws[j] = load_raw<Format>(weights_row, class_of(first) + 32U * j);
    }
#pragma unroll 1
    for (unsigned stage = 0; stage < 32U / stage_classes; ++stage) {
        __syncthreads();
        stage_activations<q2>(lds, s, stage, width, row_of, count, codes, deltas);
        __syncthreads();
#pragma unroll 1
        for (unsigned u = 0; u < stage_classes; ++u) {
            const unsigned i = stage * stage_classes + u, l = class_of(i);
            float p[MT][8];
#pragma unroll
            for (unsigned t = 0; t < MT; ++t)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) p[t][e] = 0.0F;
            if (l < groups) {
#pragma unroll
                for (unsigned j = 0; j < J; ++j) {
                    if (l + 32U * j >= groups) continue;
                    const Operands w = decode(raws[j], grid);
                    const unsigned n = next_with(i + 1U, j);
                    if (n < 32U) raws[j] = load_raw<Format>(weights_row, class_of(n) + 32U * j);
                    const unsigned q = u * J + j;
#pragma unroll
                    for (unsigned t = 0; t < MT; ++t) {
                        if (16U * t >= count) continue;
                        const v4i a0 = a[(2U * q) * s.rows + 16U * t + i16];
                        const v4i a1 = a[(2U * q + 1U) * s.rows + 16U * t + i16];
                        const v8i r = group_dot(a0, w.b0, a1, w.b1);
                        const float* dt = d + q * s.rows + 16U * t + 8U * h;
                        if constexpr (q2) {
                            const int* s0 = sums + (2U * q) * s.rows + 16U * t + 8U * h;
                            const int* s1 = sums + (2U * q + 1U) * s.rows + 16U * t + 8U * h;
#pragma unroll
                            for (unsigned e = 0; e < 8U; ++e) {
                                const int b = __mul24(w.m0, s0[e]) + __mul24(w.m1, s1[e]);
                                const float scaled = w.d * magic_float(r[e]);
                                const float offset = w.dmin * static_cast<float>(b);
                                p[t][e] = p[t][e] + (scaled - offset) * dt[e];
                            }
                        } else {
#pragma unroll
                            for (unsigned e = 0; e < 8U; ++e) p[t][e] = p[t][e] + magic_float(r[e]) * (w.scale * dt[e]);
                        }
                    }
                }
            }
            // Fold class sum i into the tree: binary carry over the trailing one bits of i (uniform
            // branches on constant stack levels, so the stack stays in registers).
            const unsigned carries = static_cast<unsigned>(__builtin_ctz(~i));
            if (carries >= 1U) add_level(stack[0], p);
            if (carries >= 2U) add_level(stack[1], p);
            if (carries >= 3U) add_level(stack[2], p);
            if (carries >= 4U) add_level(stack[3], p);
            if (carries >= 5U) add_level(stack[4], p);
            if (carries == 0U) copy_level(p, stack[0]);
            else if (carries == 1U) copy_level(p, stack[1]);
            else if (carries == 2U) copy_level(p, stack[2]);
            else if (carries == 3U) copy_level(p, stack[3]);
            else if (carries == 4U) copy_level(p, stack[4]);
            else copy_level(p, v);
        }
    }
}

struct Batch { unsigned expert, begin, count; };
__device__ __forceinline__ bool load_batch(const PrefillRoutesView& routes, Batch& b) {
    if (blockIdx.x >= *routes.batch_total) return false;
    b.expert = routes.batch_expert[blockIdx.x];
    b.begin = routes.batch_begin[blockIdx.x];
    b.count = routes.batch_count[blockIdx.x];
    return true;
}
__device__ __forceinline__ void load_grid(unsigned long long* grid) {
    for (unsigned t = threadIdx.x; t < 256U; t += blockDim.x) grid[t] = prefill_iq2_grid[t];
}

constexpr unsigned prefill_waves = 8U;

// Gate/up: wave w runs matrix w % 2 (gate, up) for outputs 16 (w / 2) .. of the block's 64.
template <WeightFormat Format, unsigned MT, unsigned J>
__global__ __launch_bounds__(32U * prefill_waves) void prefill_gate_up_kernel(PrefillGateUpView view, StageLayout layout) {
    extern __shared__ int4 dynamic_lds[];
    auto* lds = reinterpret_cast<unsigned char*>(dynamic_lds);
    __shared__ unsigned long long grid[Format == WeightFormat::iq2_xxs ? 256 : 1];
    __shared__ unsigned row_of[32], slot_of[32];
    Batch batch;
    if (!load_batch(view.routes, batch)) return;
    if constexpr (Format == WeightFormat::iq2_xxs) load_grid(grid);
    if (threadIdx.x < batch.count) {
        row_of[threadIdx.x] = view.routes.entry_rows[batch.begin + threadIdx.x];
        slot_of[threadIdx.x] = view.routes.entry_slots[batch.begin + threadIdx.x];
    }
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U, matrix = wave & 1U, tile = wave >> 1U;
    const unsigned out = blockIdx.y * 64U + 16U * tile + (lane & 15U), h = lane >> 4U;
    const auto* weights = static_cast<const unsigned char*>(matrix == 0U ? view.gate : view.up) +
        batch.expert * view.expert_stride_bytes + static_cast<unsigned long long>(out) * view.row_bytes;
    float v[MT][8];
    wave_products<Format, MT, J>(lds, layout, weights, view.hidden, row_of, batch.count, view.codes, view.deltas, grid, v);
    __syncthreads(); // every wave is done with the stage: the tile reuses it
    auto* values = reinterpret_cast<float*>(lds); // [matrix][row][64]
#pragma unroll
    for (unsigned t = 0; t < MT; ++t)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            const unsigned r = 16U * t + 2U * e + h;
            if (r < batch.count) values[(matrix * 32U + r) * 64U + 16U * tile + (lane & 15U)] = v[t][e];
        }
    __syncthreads();
    // The decode window's epilogue for each (entry, 32-output group), one wave per pair.
    const unsigned groups = view.inter / 32U;
    for (unsigned job = wave; job < batch.count * 2U; job += prefill_waves) {
        const unsigned e = job / 2U, quad = job % 2U, at = blockIdx.y * 64U + 32U * quad + lane;
        const unsigned long long slot = slot_of[e];
        const float g = values[e * 64U + 32U * quad + lane], u = values[(32U + e) * 64U + 32U * quad + lane];
        if (view.gate_out) { view.gate_out[slot * view.inter + at] = g; view.up_out[slot * view.inter + at] = u; }
        const float act = swiglu_clamped(g, u, view.limit);
        float amax = fabsf(act);
        for (int stride = 16; stride; stride /= 2) amax = fmaxf(amax, __shfl_xor(amax, stride, 32));
        const bool tiny = amax > 0.0F && amax < 0x1p-119F;
        const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
        const float rounded = tiny ? 0.0F : rintf(act / delta);
        const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
        int sum = code;
        for (int stride = 16; stride; stride /= 2) sum += __shfl_xor(sum, stride, 32);
        view.act_codes[slot * view.inter + at] = static_cast<signed char>(code);
        const unsigned group = at / 32U;
        if (lane == 0U) { view.act_deltas[slot * groups + group] = delta; view.act_sums[slot * groups + group] = sum; }
    }
}

// Down: wave w runs outputs 16 w .. of the block's 128; raw slot vectors straight from the tile.
template <WeightFormat Format, unsigned MT, unsigned J>
__global__ __launch_bounds__(32U * prefill_waves) void prefill_down_kernel(PrefillDownView view, StageLayout layout) {
    extern __shared__ int4 dynamic_lds[];
    auto* lds = reinterpret_cast<unsigned char*>(dynamic_lds);
    __shared__ unsigned slot_of[32];
    Batch batch;
    if (!load_batch(view.routes, batch)) return;
    if (threadIdx.x < batch.count) slot_of[threadIdx.x] = view.routes.entry_slots[batch.begin + threadIdx.x];
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U, h = lane >> 4U;
    const unsigned out = blockIdx.y * 128U + 16U * wave + (lane & 15U);
    const auto* weights = static_cast<const unsigned char*>(view.weights) + batch.expert * view.expert_stride_bytes +
        static_cast<unsigned long long>(out) * view.row_bytes;
    float v[MT][8];
    wave_products<Format, MT, J>(lds, layout, weights, view.width, slot_of, batch.count, view.codes, view.deltas, nullptr, v);
#pragma unroll
    for (unsigned t = 0; t < MT; ++t)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            const unsigned r = 16U * t + 2U * e + h;
            if (r < batch.count) view.out[static_cast<unsigned long long>(slot_of[r]) * view.outputs + out] = v[t][e];
        }
}

// ---- Class-split workgroups (default for batches of 16 rows) ----
//
// The stage kernels above walk every weight row in the lane-class order, so each cache line of a row
// is touched in eight separate passes; with the working set of all resident waves beyond the L2, the
// weights come from the last-level cache about eight times. Here a workgroup is 2 x 8 waves:
// class-wave q (0..7) owns lane classes 4q .. 4q + 3, i.e. one half of block q / 2 + 4j of every
// weight row (blocks of 256 weights), and all class-waves walk the rows block by block in address
// order. The activations of the batch (16 rows) are staged one chunk of 32 groups (1024 values) at a
// time; chunk j feeds group 4q + k + 32j of class-wave q, so each class still accumulates its groups
// in ascending order. A wave covers 32 outputs (two 16-output tiles): lane i of half h decodes output
// 16h + i once and the halves trade their operands with v_permlanex16, so no group is decoded twice.
// At the end the class sums meet in the butterfly tree: levels 1-3 (class bits 4, 3, 2 = wave bits
// 2, 1, 0) pair waves q and q + 4, q + 2, q + 1 through LDS, and wave 0 finishes levels 4-5 (class
// bits 1, 0) in registers.
// Waves per SIMD the split kernels are register-allocated for (8: at most 192 VGPRs, two workgroups
// per WGP).
#ifndef STRIX_DS41_SPLIT_WAVES
#define STRIX_DS41_SPLIT_WAVES 8
#endif
#ifndef STRIX_DS41_SPLIT_DOWN_WAVES
#define STRIX_DS41_SPLIT_DOWN_WAVES 8
#endif
constexpr unsigned split_rows = 16U;
constexpr unsigned split_class_waves = 8U;
constexpr unsigned split_classes = 32U / split_class_waves;                            // 4 per wave
constexpr unsigned split_chunk_groups = 32U;
constexpr unsigned split_code_bytes = split_chunk_groups * 2U * split_rows * 16U;     // 16 KB
constexpr unsigned split_delta_bytes = split_chunk_groups * split_rows * 4U;          // 2 KB
constexpr unsigned split_sum_bytes = split_chunk_groups * split_rows * 4U;            // 2 KB
constexpr unsigned split_leaf_floats = split_classes * 8U * 32U;                      // one wave, one tile
constexpr unsigned split_exchange_bytes = 2U * 4U * split_leaf_floats * 4U;           // 32 KB
constexpr unsigned split_stage_bytes = split_code_bytes + split_delta_bytes + split_sum_bytes;      // 20 KB
// Activations double-buffered: chunk c + 1 is read from memory while chunk c is multiplied.
#ifndef STRIX_DS41_SPLIT_DOUBLE
#define STRIX_DS41_SPLIT_DOUBLE 0
#endif
// Weight chunks in flight ahead of the one being multiplied (1 or 2).
#ifndef STRIX_DS41_SPLIT_WEIGHT_DEPTH
#define STRIX_DS41_SPLIT_WEIGHT_DEPTH 2
#endif
constexpr unsigned split_weight_depth = STRIX_DS41_SPLIT_WEIGHT_DEPTH;
constexpr unsigned split_stage_buffers = STRIX_DS41_SPLIT_DOUBLE ? 2U : 1U;
constexpr unsigned split_threads = 2U * split_class_waves * 32U;

template <bool Q2>
__device__ __forceinline__ void stage_chunk(unsigned char* lds, unsigned chunk, unsigned width, const unsigned* row_of, unsigned count,
                                            const signed char* codes, const float* deltas) {
    const unsigned groups = width / 32U;
    auto* a = reinterpret_cast<v4i*>(lds);
    auto* d = reinterpret_cast<float*>(lds + split_code_bytes);
    // Q2_K half sums, packed per (group, row) as two int16: low = codes 0-15, high = codes 16-31.
    auto* sums = reinterpret_cast<short*>(lds + split_code_bytes + split_delta_bytes);
    for (unsigned item = threadIdx.x; item < split_chunk_groups * 2U * split_rows; item += blockDim.x) {
        const unsigned r = item % split_rows, kh = (item / split_rows) % 2U, gg = item / (2U * split_rows);
        const unsigned g = split_chunk_groups * chunk + gg;
        if (g >= groups) continue;
        v4i v{0, 0, 0, 0};
        const bool live = r < count;
        if (live) v = *reinterpret_cast<const v4i*>(codes + static_cast<unsigned long long>(row_of[r]) * width + 32U * g + 16U * kh);
        a[(2U * gg + kh) * split_rows + r] = v;
        const unsigned at = tile_position(r);
        if (kh == 0U) d[gg * split_rows + at] = live ? deltas[static_cast<unsigned long long>(row_of[r]) * groups + g] : 0.0F;
        if constexpr (Q2) {
            const int ones = 0x01010101;
            int sum = 0;
            sum = __builtin_amdgcn_sudot4(false, ones, true, v.x, sum, false);
            sum = __builtin_amdgcn_sudot4(false, ones, true, v.y, sum, false);
            sum = __builtin_amdgcn_sudot4(false, ones, true, v.z, sum, false);
            sum = __builtin_amdgcn_sudot4(false, ones, true, v.w, sum, false);
            sums[2U * (gg * split_rows + at) + kh] = static_cast<short>(sum); // |sum| <= 16 * 127
        }
    }
}

// The staging of stage_chunk in two steps: the global reads of a thread's two items into registers,
// then their LDS writes, so the reads of chunk c + 1 are in flight while chunk c is multiplied.
struct StagedItems { v4i v[2]; float d[2]; };
[[maybe_unused]] __device__ __forceinline__ StagedItems load_chunk_items(unsigned chunk, unsigned width, const unsigned* row_of, unsigned count,
                                                        const signed char* codes, const float* deltas) {
    const unsigned groups = width / 32U;
    StagedItems x{};
#pragma unroll
    for (unsigned i = 0; i < 2U; ++i) {
        const unsigned item = threadIdx.x + split_threads * i;
        const unsigned r = item % split_rows, kh = (item / split_rows) % 2U, gg = item / (2U * split_rows);
        const unsigned g = split_chunk_groups * chunk + gg;
        x.v[i] = v4i{0, 0, 0, 0};
        x.d[i] = 0.0F;
        if (g < groups && r < count) {
            x.v[i] = *reinterpret_cast<const v4i*>(codes + static_cast<unsigned long long>(row_of[r]) * width + 32U * g + 16U * kh);
            if (kh == 0U) x.d[i] = deltas[static_cast<unsigned long long>(row_of[r]) * groups + g];
        }
    }
    return x;
}
template <bool Q2>
__device__ __forceinline__ void store_chunk_items(unsigned char* lds, const StagedItems& x, unsigned chunk, unsigned width) {
    const unsigned groups = width / 32U;
    auto* a = reinterpret_cast<v4i*>(lds);
    auto* d = reinterpret_cast<float*>(lds + split_code_bytes);
    auto* sums = reinterpret_cast<short*>(lds + split_code_bytes + split_delta_bytes);
#pragma unroll
    for (unsigned i = 0; i < 2U; ++i) {
        const unsigned item = threadIdx.x + split_threads * i;
        const unsigned r = item % split_rows, kh = (item / split_rows) % 2U, gg = item / (2U * split_rows);
        if (split_chunk_groups * chunk + gg >= groups) continue;
        const v4i v = x.v[i];
        a[(2U * gg + kh) * split_rows + r] = v;
        const unsigned at = tile_position(r);
        if (kh == 0U) d[gg * split_rows + at] = x.d[i];
        if constexpr (Q2) {
            const int ones = 0x01010101;
            int sum = 0;
            sum = __builtin_amdgcn_sudot4(false, ones, true, v.x, sum, false);
            sum = __builtin_amdgcn_sudot4(false, ones, true, v.y, sum, false);
            sum = __builtin_amdgcn_sudot4(false, ones, true, v.z, sum, false);
            sum = __builtin_amdgcn_sudot4(false, ones, true, v.w, sum, false);
            sums[2U * (gg * split_rows + at) + kh] = static_cast<short>(sum);
        }
    }
}

// Value of lane l ^ 16 (the same lane of the other half-wave). A cross-lane read: call it only
// where every lane of the wave is active (never inside a lane-dependent branch or select).
__device__ __forceinline__ int other_half(int v) {
    const auto u = static_cast<unsigned>(v);
    return static_cast<int>(__builtin_amdgcn_permlanex16(u, u, 0x76543210U, 0xfedcba98U, false, false));
}
__device__ __forceinline__ float other_half(float v) { return __int_as_float(other_half(__float_as_int(v))); }
__device__ __forceinline__ v4i other_half(v4i v) { return v4i{other_half(v.x), other_half(v.y), other_half(v.z), other_half(v.w)}; }
__device__ __forceinline__ v4i pick(bool first, v4i a, v4i b) { return first ? a : b; }

// The four groups 4q .. 4q + 3 of one chunk (one half of a 256-weight block, or four consecutive
// MXFP4 blocks), loaded with one contiguous read per lane a whole chunk ahead of their use.
template <WeightFormat Format> struct ChunkRaw;
template <> struct ChunkRaw<WeightFormat::iq2_xxs> { unsigned short d; unsigned long long aux[4]; };
template <> struct ChunkRaw<WeightFormat::q2_k> { unsigned q[8]; unsigned scales[2]; unsigned dm; };
template <> struct ChunkRaw<WeightFormat::mxfp4> { unsigned bytes[17]; };
template <WeightFormat Format> inline constexpr bool chunk_prefetch = Format != WeightFormat::q8_0;
template <> struct ChunkRaw<WeightFormat::q8_0> { unsigned unused; };

// first = the chunk's first group of this wave (a multiple of 4).
template <WeightFormat Format>
__device__ __forceinline__ ChunkRaw<Format> load_chunk(const unsigned char* row, unsigned first) {
    ChunkRaw<Format> c{};
    if constexpr (Format == WeightFormat::iq2_xxs) {
        const auto* block = row + first / 8U * iq2_xxs_block_bytes;
        c.d = load_u16(block);
        __builtin_memcpy(c.aux, block + 2U + 8U * (first % 8U), 32);
    } else if constexpr (Format == WeightFormat::q2_k) {
        const auto* block = row + first / 8U * q2_k_block_bytes;
        const unsigned half = (first % 8U) / 4U;
        __builtin_memcpy(c.q, block + 16U + 32U * half, 32);
        __builtin_memcpy(c.scales, block + 8U * half, 8);
        __builtin_memcpy(&c.dm, block + 80U, 4);
    } else if constexpr (Format == WeightFormat::mxfp4) {
        __builtin_memcpy(c.bytes, row + first * mxfp4_block_bytes, 68);
    }
    return c;
}
[[maybe_unused]] __device__ __forceinline__ unsigned byte_at(const unsigned (&words)[17], unsigned i) { return (words[i / 4U] >> (8U * (i % 4U))) & 0xffU; }
[[maybe_unused]] __device__ __forceinline__ unsigned word_at(const unsigned (&words)[17], unsigned i) {
    // Bytes i .. i + 3 (i constant after unrolling).
    if (i % 4U == 0U) return words[i / 4U];
    return __builtin_amdgcn_alignbyte(words[i / 4U + 1U], words[i / 4U], i % 4U);
}
// Group k of the chunk as the per-group raw form the decoders take.
template <WeightFormat Format>
__device__ __forceinline__ Raw<Format> chunk_group(const ChunkRaw<Format>& c, unsigned k) {
    Raw<Format> r{};
    if constexpr (Format == WeightFormat::iq2_xxs) {
        r.d = c.d;
        r.aux = c.aux[k];
    } else if constexpr (Format == WeightFormat::q2_k) {
#pragma unroll
        for (unsigned i = 0; i < 8U; ++i) r.q[i] = c.q[i];
        r.scales = static_cast<unsigned short>((c.scales[k / 2U] >> (16U * (k % 2U))) & 0xffffU);
        r.dm = c.dm;
        r.shift = 2U * k;
    } else if constexpr (Format == WeightFormat::mxfp4) {
        r.e = byte_at(c.bytes, 17U * k);
#pragma unroll
        for (unsigned i = 0; i < 4U; ++i) r.q[i] = word_at(c.bytes, 17U * k + 1U + 4U * i);
    }
    return r;
}

// Weight tiles in LDS (default for IQ2_XXS, Q2_K, MXFP4): the workgroup's 64 weight rows (two runs of
// 32 consecutive rows: gate and up, or the two output pairs of down) are read chunk by chunk with
// coalesced dword loads (consecutive lanes, consecutive bytes of one row) into an LDS tile, and each
// lane decodes its own row's groups from there. Read per lane from global memory, every lane of a
// load touched its own cache line (32 rows per instruction). A chunk is 32 groups: 4 blocks of 256
// weights (IQ2_XXS 264 bytes, Q2_K 336) or 32 MXFP4 blocks (544), always dword-aligned in a row.
#ifndef STRIX_DS41_SPLIT_WEIGHT_LDS
#define STRIX_DS41_SPLIT_WEIGHT_LDS 1
#endif
template <WeightFormat Format> inline constexpr unsigned chunk_row_bytes =
    Format == WeightFormat::iq2_xxs ? 4U * iq2_xxs_block_bytes
    : Format == WeightFormat::q2_k  ? 4U * q2_k_block_bytes
    : Format == WeightFormat::mxfp4 ? 32U * mxfp4_block_bytes
                                    : 32U * q8_0_block_bytes;
template <WeightFormat Format> inline constexpr bool weight_lds = STRIX_DS41_SPLIT_WEIGHT_LDS != 0 && Format != WeightFormat::q8_0;
constexpr unsigned tile_rows = 64U;
template <WeightFormat Format> inline constexpr unsigned tile_words = tile_rows * chunk_row_bytes<Format> / 4U;
template <WeightFormat Format> inline constexpr unsigned tile_loads = (tile_words<Format> + split_threads - 1U) / split_threads;
// Activation stage buffers of a form: 2 (double-buffered) from form 3 on, else the build default.
template <int Form> inline constexpr unsigned stage_buffers_of = Form == 3 ? 2U : split_stage_buffers;
// Forms 6 (= 1) and 7 (= 6 with a batched activation stage): the chunk loop's barriers are LDS-only (fence
// workgroup/local + s_barrier), so the next weight tile stays in flight across them (__syncthreads drains vmcnt; barrier measurements
// thread trace: 2 x 7 % of the gate/up kernel's latency). The same loads, stores and products: the same bits.
template <int Form> __device__ __forceinline__ void split_sync() {
    if constexpr (Form >= 5) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
    } else {
        __syncthreads();
    }
}
// Form 4: one barrier before the tile store and the activation stage, the stage's reads before the next tile's (so
// the in-order load counter never makes the stage wait for the prefetch), then one barrier: two per chunk, not three.
template <WeightFormat Format, int Form> inline constexpr bool merged_stage = Form == 4 && weight_lds<Format> && stage_buffers_of<Form> == 1U;
template <WeightFormat Format, unsigned StageBuffers = split_stage_buffers>
__host__ __device__ constexpr unsigned split_lds_for() {
    const unsigned tile = weight_lds<Format> ? tile_rows * chunk_row_bytes<Format> : 0U;
    const unsigned work = StageBuffers * split_stage_bytes + tile;
    return work > split_exchange_bytes ? work : split_exchange_bytes;
}
struct TileRows { const unsigned char* base[2]; std::uint64_t row_bytes; };
template <WeightFormat Format> struct TileRegs { unsigned w[tile_loads<Format>]; };
template <WeightFormat Format>
__device__ __forceinline__ TileRegs<Format> load_tile(const TileRows& t, unsigned chunk, unsigned groups) {
    constexpr unsigned row_words = chunk_row_bytes<Format> / 4U;
    TileRegs<Format> r{};
    if (split_chunk_groups * chunk >= groups) return r;
    const std::uint64_t at = static_cast<std::uint64_t>(chunk) * chunk_row_bytes<Format>;
#pragma unroll
    for (unsigned i = 0; i < tile_loads<Format>; ++i) {
        const unsigned word = threadIdx.x + split_threads * i;
        if (word < tile_words<Format>) {
            const unsigned row = word / row_words, off = word % row_words;
            const auto* src = t.base[row / 32U] + static_cast<std::uint64_t>(row % 32U) * t.row_bytes + at;
            r.w[i] = reinterpret_cast<const unsigned*>(src)[off];
        }
    }
    return r;
}
template <WeightFormat Format>
__device__ __forceinline__ void store_tile(unsigned char* tile, const TileRegs<Format>& r) {
#pragma unroll
    for (unsigned i = 0; i < tile_loads<Format>; ++i) {
        const unsigned word = threadIdx.x + split_threads * i;
        if (word < tile_words<Format>) reinterpret_cast<unsigned*>(tile)[word] = r.w[i];
    }
}

// Leaves leaf[k][t][e]: class 4q + k, output tile t, D element e (row 2e + h, output 16t + i).
template <WeightFormat Format, int Form>
__device__ __forceinline__ void class_wave(unsigned char* lds, unsigned q, const unsigned char* weights_row, unsigned width,
                                           const unsigned* row_of, unsigned count, const signed char* codes, const float* deltas,
                                           const unsigned long long* grid, const SignEntry* signs, float (&leaf)[split_classes][2][8], const TileRows& tile_rows_src,
                                           unsigned tile_row) {
    constexpr bool q2 = Format == WeightFormat::q2_k;
    const unsigned lane = threadIdx.x % 32U, i16 = lane & 15U, h = lane >> 4U, groups = width / 32U;
    const bool upper = h != 0U;
#pragma unroll
    for (unsigned k = 0; k < split_classes; ++k)
#pragma unroll
        for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) leaf[k][t][e] = 0.0F;
    const unsigned chunks = (groups + split_chunk_groups - 1U) / split_chunk_groups;
    // Weights: the chunk's four groups arrive one chunk ahead (Q8_0, whose shared-expert rows stay
    // in the L2, keeps a one-group lookahead).
    ChunkRaw<Format> chunk_now{}, chunk_next{}, chunk_after{};
    Raw<Format> cur{};
    constexpr bool wl = weight_lds<Format>;
    constexpr unsigned SB = stage_buffers_of<Form>;
    unsigned char* tile = lds + SB * split_stage_bytes;
    TileRegs<Format> tile_next{};
    if constexpr (wl) {
        (void)chunk_now; (void)chunk_next; (void)chunk_after;
        tile_next = load_tile<Format>(tile_rows_src, 0U, groups);
    } else if constexpr (chunk_prefetch<Format>) {
        if (split_classes * q < groups) chunk_next = load_chunk<Format>(weights_row, split_classes * q);
        if constexpr (split_weight_depth > 1U)
            if (split_classes * q + split_chunk_groups < groups) chunk_after = load_chunk<Format>(weights_row, split_classes * q + split_chunk_groups);
    } else {
        if (split_classes * q < groups) cur = load_raw<Format>(weights_row, split_classes * q);
    }
#pragma unroll 1
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
        const unsigned base = split_chunk_groups * chunk + split_classes * q;
        if constexpr (merged_stage<Format, Form>) {
            split_sync<Form>();
            store_tile<Format>(tile, tile_next);
            stage_chunk<q2>(lds, chunk, width, row_of, count, codes, deltas);
            if (chunk + 1U < chunks) tile_next = load_tile<Format>(tile_rows_src, chunk + 1U, groups);
        } else if constexpr (wl) {
            // The tile of chunk c: previous readers are done (barrier), then the stage below is
            // written and the next tile's loads go out before the products.
            split_sync<Form>();
            store_tile<Format>(tile, tile_next);
            if (chunk + 1U < chunks) tile_next = load_tile<Format>(tile_rows_src, chunk + 1U, groups);
        } else if constexpr (chunk_prefetch<Format>) {
            chunk_now = chunk_next;
            if constexpr (split_weight_depth > 1U) {
                chunk_next = chunk_after;
                if (base + 2U * split_chunk_groups < groups) chunk_after = load_chunk<Format>(weights_row, base + 2U * split_chunk_groups);
            } else if (base + split_chunk_groups < groups) {
                chunk_next = load_chunk<Format>(weights_row, base + split_chunk_groups);
            }
        }
        unsigned char* buffer = lds + (chunk % SB) * split_stage_bytes;
        StagedItems ahead_items{};
        if constexpr (SB == 2U) {
            if (chunk == 0U) {
                split_sync<Form>(); // the batch's row indices (written by other threads) are visible
                store_chunk_items<q2>(buffer, load_chunk_items(0U, width, row_of, count, codes, deltas), 0U, width);
            }
            if (chunk + 1U < chunks) ahead_items = load_chunk_items(chunk + 1U, width, row_of, count, codes, deltas);
            split_sync<Form>(); // chunk's buffer written; the other buffer's readers (chunk - 1) are done
        } else if constexpr (merged_stage<Format, Form>) {
            split_sync<Form>();
        } else if constexpr (Form == 7) {
            // Form 7 (= 6): both of the thread's stage items are read before either is stored (one wait, not two).
            split_sync<Form>();
            store_chunk_items<q2>(lds, load_chunk_items(chunk, width, row_of, count, codes, deltas), chunk, width);
            split_sync<Form>();
        } else {
            split_sync<Form>();
            stage_chunk<q2>(lds, chunk, width, row_of, count, codes, deltas);
            split_sync<Form>();
        }
        const auto* a = reinterpret_cast<const v4i*>(buffer);
        const auto* d = reinterpret_cast<const float*>(buffer + split_code_bytes);
        const auto* sums = reinterpret_cast<const int*>(buffer + split_code_bytes + split_delta_bytes);
#pragma unroll
        for (unsigned k = 0; k < split_classes; ++k) {
            if (base >= groups) break; // no groups of this wave in the chunk (it still joins the barriers)
            Raw<Format> ahead{};
            (void)ahead;
            if constexpr (wl) {
                cur = load_raw<Format>(tile + tile_row * chunk_row_bytes<Format>, split_classes * q + k);
            } else if constexpr (chunk_prefetch<Format>) {
                cur = chunk_group<Format>(chunk_now, k);
            } else {
                // Next group of this wave: the following one, else the first of its next chunk.
                const unsigned next = k + 1U < split_classes ? base + k + 1U : base + split_chunk_groups;
                if (next < groups) ahead = load_raw<Format>(weights_row, next);
            }
            const Operands own = decode_form<Form>(cur, grid, signs);
            const v4i x0 = other_half(own.b0), x1 = other_half(own.b1);
            // The other half's scale terms (Q2_K: the packed d / dmin and scale bytes).
            const unsigned x_dm = q2 ? static_cast<unsigned>(other_half(static_cast<int>(own.dm_bits))) : 0U;
            const unsigned x_sc = q2 ? static_cast<unsigned>(other_half(static_cast<int>(own.sc_bits))) : 0U;
            const float x_scale = q2 ? 0.0F : other_half(own.scale);
            const unsigned gg = split_classes * q + k;
            const v4i a0 = a[(2U * gg) * split_rows + i16];
            const v4i a1 = a[(2U * gg + 1U) * split_rows + i16];
            const float* dt = d + gg * split_rows + 8U * h;
#pragma unroll
            for (unsigned t = 0; t < 2U; ++t) {
                const bool mine = (t == 1U) == upper; // this half decoded tile t
                const v4i b0 = pick(mine, own.b0, x0), b1 = pick(mine, own.b1, x1);
                const v8i r = group_dot(a0, b0, a1, b1);
                if constexpr (q2) {
                    const unsigned dmb = mine ? own.dm_bits : x_dm, scb = mine ? own.sc_bits : x_sc;
                    const float dd = half_bits(static_cast<unsigned short>(dmb & 0xffffU));
                    const float dm = half_bits(static_cast<unsigned short>(dmb >> 16U));
                    // B = (sc0 >> 4) * S0 + (sc1 >> 4) * S1 from the packed half sums (exact integers).
                    const int m0 = static_cast<int>((scb & 0xffU) >> 4U), m1 = static_cast<int>((scb >> 8U) >> 4U);
                    const int* sp = sums + gg * split_rows + 8U * h;
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) {
                        const int packed = sp[e];
                        const int s0 = static_cast<int>(static_cast<short>(packed & 0xffff)), s1 = packed >> 16;
                        const int b = __mul24(m0, s0) + __mul24(m1, s1);
                        const float scaled = dd * magic_float(r[e]);
                        const float offset = dm * static_cast<float>(b);
                        leaf[k][t][e] = leaf[k][t][e] + (scaled - offset) * dt[e];
                    }
                } else {
                    const float scale = mine ? own.scale : x_scale;
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) leaf[k][t][e] = leaf[k][t][e] + magic_float(r[e]) * (scale * dt[e]);
                }
            }
            if constexpr (!chunk_prefetch<Format>) cur = ahead;
        }
        if constexpr (SB == 2U)
            if (chunk + 1U < chunks)
                store_chunk_items<q2>(lds + ((chunk + 1U) % 2U) * split_stage_bytes, ahead_items, chunk + 1U, width);
    }
}

// Combines the leaves of the eight class-waves of one pair into the butterfly result, held by
// class-wave 0 in v[t][e]. Every thread of the workgroup calls it.
__device__ __forceinline__ void split_tree(unsigned char* lds, unsigned q, unsigned pair, float (&leaf)[split_classes][2][8],
                                           float (&v)[2][8]) {
    auto* x = reinterpret_cast<float*>(lds);
    const unsigned lane = threadIdx.x % 32U;
#pragma unroll
    for (unsigned t = 0; t < 2U; ++t) {
        // Levels 1-3: wave q + s into wave q for s = 4, 2, 1 (class bits 4, 3, 2).
#pragma unroll
        for (unsigned s = 4U; s != 0U; s /= 2U) {
            __syncthreads();
            if (q >= s && q < 2U * s) {
                float* slot = x + (4U * pair + (q - s)) * split_leaf_floats;
#pragma unroll
                for (unsigned k = 0; k < split_classes; ++k)
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) slot[(8U * k + e) * 32U + lane] = leaf[k][t][e];
            }
            __syncthreads();
            if (q < s) {
                const float* slot = x + (4U * pair + q) * split_leaf_floats;
#pragma unroll
                for (unsigned k = 0; k < split_classes; ++k)
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) leaf[k][t][e] = leaf[k][t][e] + slot[(8U * k + e) * 32U + lane];
            }
        }
        // Levels 4-5 over the low two bits of the class.
        if (q == 0U)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const float c0 = leaf[0][t][e] + leaf[2][t][e], c1 = leaf[1][t][e] + leaf[3][t][e];
                v[t][e] = c0 + c1;
            }
    }
}

// Gate/up: wave w = 2q + m (matrix m: gate, up; class-wave q); 32 outputs of both matrices.
template <WeightFormat Format, int Form>
__global__ __launch_bounds__(split_threads) __attribute__((amdgpu_waves_per_eu(STRIX_DS41_SPLIT_WAVES))) void split_gate_up_kernel(PrefillGateUpView view) {
    extern __shared__ int4 dynamic_lds[];
    auto* lds = reinterpret_cast<unsigned char*>(dynamic_lds);
    __shared__ unsigned long long grid[Format == WeightFormat::iq2_xxs ? 256 : 1];
    __shared__ SignEntry signs[Form == 2 && Format == WeightFormat::iq2_xxs ? 128 : 1];
    __shared__ unsigned row_of[split_rows], slot_of[split_rows];
    __shared__ float values[2][split_rows][32];
    Batch batch;
    if (!load_batch(view.routes, batch)) return;
    if constexpr (Format == WeightFormat::iq2_xxs) load_grid(grid);
    if constexpr (Form == 2 && Format == WeightFormat::iq2_xxs) load_signs(signs);
    if (threadIdx.x < batch.count) {
        row_of[threadIdx.x] = view.routes.entry_rows[batch.begin + threadIdx.x];
        slot_of[threadIdx.x] = view.routes.entry_slots[batch.begin + threadIdx.x];
    }
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U, m = wave & 1U, q = wave >> 1U, h = lane >> 4U;
    const unsigned own_out = blockIdx.y * 32U + lane; // lane i of half h: output 16h + i
    const auto* weights = static_cast<const unsigned char*>(m == 0U ? view.gate : view.up) + batch.expert * view.expert_stride_bytes +
        static_cast<unsigned long long>(own_out) * view.row_bytes;
    const std::uint64_t expert_at = batch.expert * view.expert_stride_bytes + static_cast<std::uint64_t>(blockIdx.y) * 32U * view.row_bytes;
    const TileRows rows{{static_cast<const unsigned char*>(view.gate) + expert_at, static_cast<const unsigned char*>(view.up) + expert_at},
                        view.row_bytes};
    float leaf[split_classes][2][8];
    class_wave<Format, Form>(lds, q, weights, view.hidden, row_of, batch.count, view.codes, view.deltas, grid, signs, leaf, rows, 32U * m + lane);
    float v[2][8];
    split_tree(lds, q, m, leaf, v);
    if (q == 0U)
#pragma unroll
        for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const unsigned r = 2U * e + h;
                if (r < batch.count) values[m][r][16U * t + (lane & 15U)] = v[t][e];
            }
    __syncthreads();
    const unsigned groups = view.inter / 32U, at = blockIdx.y * 32U + lane;
    for (unsigned e = wave; e < batch.count; e += split_threads / 32U) {
        const unsigned long long slot = slot_of[e];
        const float g = values[0][e][lane], u = values[1][e][lane];
        if (view.gate_out) { view.gate_out[slot * view.inter + at] = g; view.up_out[slot * view.inter + at] = u; }
        const float act = swiglu_clamped(g, u, view.limit);
        float amax = fabsf(act);
        for (int stride = 16; stride; stride /= 2) amax = fmaxf(amax, __shfl_xor(amax, stride, 32));
        const bool tiny = amax > 0.0F && amax < 0x1p-119F;
        const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
        const float rounded = tiny ? 0.0F : rintf(act / delta);
        const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
        int sum = code;
        for (int stride = 16; stride; stride /= 2) sum += __shfl_xor(sum, stride, 32);
        view.act_codes[slot * view.inter + at] = static_cast<signed char>(code);
        if (lane == 0U) { view.act_deltas[slot * groups + blockIdx.y] = delta; view.act_sums[slot * groups + blockIdx.y] = sum; }
    }
}

// Down: wave w = 2q + p (output pair p: outputs 32p .. of the block's 64; class-wave q).
template <WeightFormat Format, int Form>
__global__ __launch_bounds__(split_threads) __attribute__((amdgpu_waves_per_eu(STRIX_DS41_SPLIT_DOWN_WAVES))) void split_down_kernel(PrefillDownView view) {
    extern __shared__ int4 dynamic_lds[];
    auto* lds = reinterpret_cast<unsigned char*>(dynamic_lds);
    __shared__ unsigned slot_of[split_rows];
    Batch batch;
    if (!load_batch(view.routes, batch)) return;
    if (threadIdx.x < batch.count) slot_of[threadIdx.x] = view.routes.entry_slots[batch.begin + threadIdx.x];
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U, p = wave & 1U, q = wave >> 1U, h = lane >> 4U;
    const unsigned first = blockIdx.y * 64U + 32U * p, own_out = first + lane;
    const auto* weights = static_cast<const unsigned char*>(view.weights) + batch.expert * view.expert_stride_bytes +
        static_cast<unsigned long long>(own_out) * view.row_bytes;
    const auto* tile_base = static_cast<const unsigned char*>(view.weights) + batch.expert * view.expert_stride_bytes +
        static_cast<std::uint64_t>(blockIdx.y) * 64U * view.row_bytes;
    const TileRows rows{{tile_base, tile_base + 32U * view.row_bytes}, view.row_bytes};
    float leaf[split_classes][2][8];
    class_wave<Format, Form>(lds, q, weights, view.width, slot_of, batch.count, view.codes, view.deltas, nullptr, nullptr, leaf, rows, 32U * p + lane);
    float v[2][8];
    split_tree(lds, q, p, leaf, v);
    if (q != 0U) return;
#pragma unroll
    for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            const unsigned r = 2U * e + h;
            if (r < batch.count) view.out[static_cast<unsigned long long>(slot_of[r]) * view.outputs + first + 16U * t + (lane & 15U)] = v[t][e];
        }
}

#include "hip_ds41_prefill_relaxed.inl"  // FAST lever ds41-moe-relaxed-v1 (not exact)

// ---- Split kernels v2 (STRIX_DS41_PREFILL_MOE_V2, IQ2_XXS gate/up and Q2_K down) ----
//
// The same declared products as split_gate_up_kernel / split_down_kernel (forms 6/7), bit for bit:
// - Class map: class-wave q owns the lane classes q + 8k (k = 0..3), i.e. group q + 8k of every chunk, so the class bits
//   held in a wave are the tree's FIRST two levels (bits 4 and 3): each wave folds its four class sums in registers
//   ((q + q+16) + (q+8 + q+24)) and only one value per (row, output) meets the other waves in levels 3-5 (class bits 2,
//   1, 0 = wave bits), through LDS: a quarter of the exchange of split_tree, one barrier pair per level for both tiles.
//   It also balances a partial last chunk (down width 2304 = 72 groups: every wave runs 9 groups, not 12 / 8).
// - Two LDS-only barriers per chunk: the stage's global reads go out before the next weight tile's (the in-order load
//   counter never makes the stage wait for the prefetch).
// - V2_DECODE (IQ2_XXS): the sign of each grid byte is applied by one v_perm_b32 that picks the byte from the grid
//   entry or from its negation (both in the LDS grid table): the integers of apply_signs(grid, byte_masks(signs)).
// - V2_FMA (Q2_K): the minimum term's integer B = m0 * S0 + m1 * S1 (|B| <= 60960) as fma(m1, S1, fma(m0, S0, +0)) on
//   exact float operands: every intermediate is an exact integer, and +0 for B = 0 (the int-to-float of B).
// - V2_PART (Q2_K): a partial last chunk loads only the 256-weight blocks it has (Q2_K down: 1 of 4 in chunk 2).
constexpr unsigned v2_on = 1U, v2_decode = 2U, v2_fma = 4U, v2_part = 8U;
struct alignas(16) Grid2 { unsigned pos_lo, pos_hi, neg_lo, neg_hi; };
__device__ __forceinline__ void load_grid2(Grid2* grid) {
    for (unsigned t = threadIdx.x; t < 256U; t += blockDim.x) {
        const unsigned long long g = prefill_iq2_grid[t];
        const unsigned lo = static_cast<unsigned>(g), hi = static_cast<unsigned>(g >> 32U);
        // apply_signs(m, 0xffffffff) = (m ^ 0xffffffff) + 0x01010101: the all-negative word (grid bytes are 8, 25 or 43:
        // no byte carries, so each byte is the byte of apply_signs under any mask)
        grid[t] = Grid2{lo, hi, static_cast<unsigned>(apply_signs(lo, 0xffffffffU)), static_cast<unsigned>(apply_signs(hi, 0xffffffffU))};
    }
}
// v_perm_b32 selector: byte j = j (positive byte j) or 4 + j (negated byte j) for bit j of bits4.
__device__ __forceinline__ unsigned sign_select(unsigned bits4) { return ((bits4 * 0x00810204U) & 0x04040404U) | 0x03020100U; }
__device__ __forceinline__ Operands decode_v2(const Raw<WeightFormat::iq2_xxs>& raw, const Grid2* grid) {
    Operands out{};
    const float d = half_bits(raw.d);
    const unsigned aux0 = static_cast<unsigned>(raw.aux), aux1 = static_cast<unsigned>(raw.aux >> 32U);
    out.scale = d * (0.5F + static_cast<float>(aux1 >> 28U)) * 0.25F;
    int w[8];
#pragma unroll
    for (unsigned l = 0; l < 4U; ++l) {
        const Grid2 g = grid[(aux0 >> (8U * l)) & 0xffU];
        const unsigned signs = sign_byte(aux1 >> (7U * l));
        w[2U * l] = static_cast<int>(__builtin_amdgcn_perm(g.neg_lo, g.pos_lo, sign_select(signs & 15U)));
        w[2U * l + 1U] = static_cast<int>(__builtin_amdgcn_perm(g.neg_hi, g.pos_hi, sign_select(signs >> 4U)));
    }
    out.b0 = v4i{w[0], w[1], w[2], w[3]};
    out.b1 = v4i{w[4], w[5], w[6], w[7]};
    return out;
}

// V2_SIGNS (IQ2_XXS): the byte masks of both words of a grid lookup from a 128-entry LDS table indexed by the 7-bit sign
// index (entry i = byte_masks(sign_byte(i) & 15), byte_masks(sign_byte(i) >> 4)), instead of the parity, spread and
// v_perm per word: apply_signs(grid, mask) = (grid ^ mask) + (mask & 0x01010101) on the same masks, the same integers.
constexpr unsigned v2_signs = 32U, v2_signs16 = 64U; // v2_signs16: the 16-byte entries of form 2 (masks and their low bits)
__device__ __forceinline__ void load_signs2(uint2* signs) {
    for (unsigned t = threadIdx.x; t < 128U; t += blockDim.x) {
        const unsigned b = sign_byte(t);
        signs[t] = uint2{byte_masks(b & 15U), byte_masks(b >> 4U)};
    }
}
__device__ __forceinline__ Operands decode_signs2(const Raw<WeightFormat::iq2_xxs>& raw, const unsigned long long* grid, const uint2* signs) {
    Operands out{};
    const float d = half_bits(raw.d);
    const unsigned aux0 = static_cast<unsigned>(raw.aux), aux1 = static_cast<unsigned>(raw.aux >> 32U);
    out.scale = d * (0.5F + static_cast<float>(aux1 >> 28U)) * 0.25F;
    int w[8];
#pragma unroll
    for (unsigned l = 0; l < 4U; ++l) {
        const unsigned long long g = grid[(aux0 >> (8U * l)) & 0xffU];
        const uint2 m = signs[(aux1 >> (7U * l)) & 127U];
        w[2U * l] = apply_signs(static_cast<unsigned>(g), m.x);
        w[2U * l + 1U] = apply_signs(static_cast<unsigned>(g >> 32U), m.y);
    }
    out.b0 = v4i{w[0], w[1], w[2], w[3]};
    out.b1 = v4i{w[4], w[5], w[6], w[7]};
    return out;
}

// V2_PIPE (IQ2_XXS gate/up): the decode in two steps, the LDS reads of group k + 1 (its raw bytes and four grid entries)
// issued before the products of group k, the integer steps after: decode_lean's operations on the same values.
constexpr unsigned v2_pipe = 128U;
struct Iq2Pre { unsigned short d; unsigned aux1; unsigned long long g[4]; };
__device__ __forceinline__ Iq2Pre iq2_prefetch(const Raw<WeightFormat::iq2_xxs>& raw, const unsigned long long* grid) {
    Iq2Pre p;
    p.d = raw.d;
    const unsigned aux0 = static_cast<unsigned>(raw.aux);
    p.aux1 = static_cast<unsigned>(raw.aux >> 32U);
#pragma unroll
    for (unsigned l = 0; l < 4U; ++l) p.g[l] = grid[(aux0 >> (8U * l)) & 0xffU];
    return p;
}
__device__ __forceinline__ Operands iq2_finish(const Iq2Pre& p) {
    Operands out{};
    const float d = half_bits(p.d);
    out.scale = d * (0.5F + static_cast<float>(p.aux1 >> 28U)) * 0.25F;
    int w[8];
#pragma unroll
    for (unsigned l = 0; l < 4U; ++l) {
        const unsigned signs = sign_byte(p.aux1 >> (7U * l));
        w[2U * l] = apply_signs(static_cast<unsigned>(p.g[l]), byte_masks_lean(signs & 15U));
        w[2U * l + 1U] = apply_signs(static_cast<unsigned>(p.g[l] >> 32U), byte_masks_lean(signs >> 4U));
    }
    out.b0 = v4i{w[0], w[1], w[2], w[3]};
    out.b1 = v4i{w[4], w[5], w[6], w[7]};
    return out;
}

// Stage of v2: codes [2 * 32 groups][16 rows] v4i, deltas [32][16] (tile_position order), Q2_K: S0 and S1 [32][16]
// (tile order) as int16 pairs (form 6) or, with V2_FMA, two float arrays.
template <bool Q2, unsigned L> inline constexpr unsigned v2_sum_bytes = Q2 ? ((L & v2_fma) ? 2U * split_sum_bytes : split_sum_bytes) : 0U;
template <bool Q2, unsigned L> inline constexpr unsigned v2_stage_bytes = split_code_bytes + split_delta_bytes + v2_sum_bytes<Q2, L>;
template <WeightFormat Format, unsigned L> inline constexpr unsigned v2_lds =
    v2_stage_bytes<Format == WeightFormat::q2_k, L> + (Format == WeightFormat::q8_0 ? 0U : tile_rows * chunk_row_bytes<Format>);
template <bool Q2, unsigned L>
__device__ __forceinline__ void v2_store_items(unsigned char* lds, const StagedItems& x, unsigned chunk, unsigned width) {
    const unsigned groups = width / 32U;
    auto* a = reinterpret_cast<v4i*>(lds);
    auto* d = reinterpret_cast<float*>(lds + split_code_bytes);
#pragma unroll
    for (unsigned i = 0; i < 2U; ++i) {
        const unsigned item = threadIdx.x + split_threads * i;
        const unsigned r = item % split_rows, kh = (item / split_rows) % 2U, gg = item / (2U * split_rows);
        if (split_chunk_groups * chunk + gg >= groups) continue;
        const v4i v = x.v[i];
        a[(2U * gg + kh) * split_rows + r] = v;
        const unsigned at = tile_position(r);
        if (kh == 0U) d[gg * split_rows + at] = x.d[i];
        if constexpr (Q2) {
            const int ones = 0x01010101;
            int sum = 0;
            sum = __builtin_amdgcn_sudot4(false, ones, true, v.x, sum, false);
            sum = __builtin_amdgcn_sudot4(false, ones, true, v.y, sum, false);
            sum = __builtin_amdgcn_sudot4(false, ones, true, v.z, sum, false);
            sum = __builtin_amdgcn_sudot4(false, ones, true, v.w, sum, false);
            if constexpr ((L & v2_fma) != 0U) {
                auto* s = reinterpret_cast<float*>(lds + split_code_bytes + split_delta_bytes);
                s[kh * split_chunk_groups * split_rows + gg * split_rows + at] = static_cast<float>(sum); // exact (|sum| <= 2032)
            } else {
                auto* s = reinterpret_cast<short*>(lds + split_code_bytes + split_delta_bytes);
                s[2U * (gg * split_rows + at) + kh] = static_cast<short>(sum);
            }
        }
    }
}

// Weight tile loads of the v2 / R2 kernels: the two runs of 32 rows (gate and up, or the two output pairs of down) share
// one set of 32-bit per-lane offsets (row * row_bytes + 4 * word) from two uniform bases, so the loop keeps a few offsets
// instead of one 64-bit address per load (which spilled). A partial last chunk (V2_PART) loads only its blocks.
template <WeightFormat Format> inline constexpr unsigned run_words = 32U * chunk_row_bytes<Format> / 4U;
template <WeightFormat Format> inline constexpr unsigned run_loads = (run_words<Format> + split_threads - 1U) / split_threads;
template <WeightFormat Format> struct TileRegs2 { unsigned w[2][run_loads<Format>]; };
template <WeightFormat Format, unsigned L>
__device__ __forceinline__ TileRegs2<Format> load_tile2(const TileRows& t, unsigned chunk, unsigned groups) {
    constexpr unsigned row_words = chunk_row_bytes<Format> / 4U;
    constexpr unsigned block = Format == WeightFormat::q2_k ? q2_k_block_bytes : iq2_xxs_block_bytes;
    TileRegs2<Format> r{};
    const unsigned first = split_chunk_groups * chunk;
    if (first >= groups) return r;
    const unsigned left = groups - first;
    const auto* b0 = t.base[0] + static_cast<std::uint64_t>(chunk) * chunk_row_bytes<Format>;
    const auto* b1 = t.base[1] + static_cast<std::uint64_t>(chunk) * chunk_row_bytes<Format>;
    const auto rb = static_cast<unsigned>(t.row_bytes);
    if ((L & v2_part) == 0U || left >= split_chunk_groups) {
#pragma unroll
        for (unsigned i = 0; i < run_loads<Format>; ++i) {
            const unsigned word = threadIdx.x + split_threads * i;
            if (word < run_words<Format>) {
                const unsigned o = word / row_words * rb + 4U * (word % row_words);
                r.w[0][i] = *reinterpret_cast<const unsigned*>(b0 + o);
                r.w[1][i] = *reinterpret_cast<const unsigned*>(b1 + o);
            }
        }
    } else {
        const unsigned used = ((left + 7U) / 8U * block + 3U) / 4U; // words of the chunk's blocks in each row (uniform)
#pragma unroll
        for (unsigned i = 0; i < run_loads<Format>; ++i) {
            const unsigned word = threadIdx.x + split_threads * i;
            if (word < run_words<Format> && word % row_words < used) {
                const unsigned o = word / row_words * rb + 4U * (word % row_words);
                r.w[0][i] = *reinterpret_cast<const unsigned*>(b0 + o);
                r.w[1][i] = *reinterpret_cast<const unsigned*>(b1 + o);
            }
        }
    }
    return r;
}
template <WeightFormat Format>
__device__ __forceinline__ void store_tile2(unsigned char* tile, const TileRegs2<Format>& r) {
#pragma unroll
    for (unsigned i = 0; i < run_loads<Format>; ++i) {
        const unsigned word = threadIdx.x + split_threads * i;
        if (word < run_words<Format>) {
            reinterpret_cast<unsigned*>(tile)[word] = r.w[0][i];
            reinterpret_cast<unsigned*>(tile)[run_words<Format> + word] = r.w[1][i];
        }
    }
}

// v_permlanex16 without a copy of its source into the destination first: every lane is active, so the instruction's
// old-value operand is never read (an empty asm defines it).
#ifndef STRIX_DS41_V2_XCHG
#define STRIX_DS41_V2_XCHG 1
#endif
#ifndef STRIX_DS41_V2_MAGIC
#define STRIX_DS41_V2_MAGIC 1
#endif
__device__ __forceinline__ int v2_other_half(int v) {
    if constexpr (STRIX_DS41_V2_XCHG != 0) {
        int old;
        asm volatile("" : "=v"(old));
        return static_cast<int>(__builtin_amdgcn_permlanex16(static_cast<unsigned>(old), static_cast<unsigned>(v), 0x76543210U, 0xfedcba98U,
                                                             false, false));
    } else {
        return other_half(v);
    }
}
__device__ __forceinline__ v4i v2_other_half(v4i v) { return v4i{v2_other_half(v.x), v2_other_half(v.y), v2_other_half(v.z), v2_other_half(v.w)}; }
__device__ __forceinline__ float v2_other_half(float v) { return __int_as_float(v2_other_half(__float_as_int(v))); }
// group_dot with the magic accumulator kept in registers (opaque to rematerialization: no 8 moves per chain).
__device__ __forceinline__ v8i v2_group_dot(v4i a0, v4i b0, v4i a1, v4i b1, const v8i& magic) {
    const v8i c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, true, b0, magic, false);
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, true, b1, c, false);
}
// V2_PIN: the two WMMAs of v2_group_dot as inline asm on pinned registers so that A, B and C of every WMMA start in
// three different VGPR banks (bank = register % 4; gfx1151 issues a WMMA whose A, B and C share a bank in 34 cycles, not
// 32): magic accumulator v[0:7] (bank 0), A v[163:170] (bank 3), B v[154:161] (bank 2), D of output tile t v[137 + 8t]
// (bank 1, disjoint from A and B). The same two instructions on the same values (signed A and B: neg_lo:[1,1,0]).
constexpr unsigned v2_pin = 16U;
template <unsigned T>
__device__ __forceinline__ v8i v2_group_dot_pin(v4i a0, v4i b0, v4i a1, v4i b1, const v8i& magic) {
    v8i d;
    if constexpr (T == 0U) {
        asm("v_wmma_i32_16x16x16_iu8 %0, %1, %2, %3 neg_lo:[1,1,0]" : "={v[137:144]}"(d) : "{v[163:166]}"(a0), "{v[154:157]}"(b0), "{v[0:7]}"(magic));
        asm("v_wmma_i32_16x16x16_iu8 %0, %1, %2, %0 neg_lo:[1,1,0]" : "+{v[137:144]}"(d) : "{v[167:170]}"(a1), "{v[158:161]}"(b1));
    } else {
        asm("v_wmma_i32_16x16x16_iu8 %0, %1, %2, %3 neg_lo:[1,1,0]" : "={v[145:152]}"(d) : "{v[163:166]}"(a0), "{v[154:157]}"(b0), "{v[0:7]}"(magic));
        asm("v_wmma_i32_16x16x16_iu8 %0, %1, %2, %0 neg_lo:[1,1,0]" : "+{v[145:152]}"(d) : "{v[167:170]}"(a1), "{v[158:161]}"(b1));
    }
    return d;
}

// Leaves leaf[k][t][e]: class q + 8k, output tile t, D element e (row 2e + h, output 16t + i).
template <WeightFormat Format, unsigned L>
__device__ __forceinline__ void class_wave_v2(unsigned char* lds, unsigned q, unsigned width, const unsigned* row_of, unsigned count,
                                              const signed char* codes, const float* deltas, const unsigned long long* grid,
                                              const Grid2* grid2, float (&leaf)[split_classes][2][8], const TileRows& tile_rows_src,
                                              unsigned tile_row, const uint2* signs2 = nullptr) {
    constexpr bool q2 = Format == WeightFormat::q2_k;
    constexpr bool fma = q2 && (L & v2_fma) != 0U;
    const unsigned lane = threadIdx.x % 32U, i16 = lane & 15U, h = lane >> 4U, groups = width / 32U;
    const bool upper = h != 0U;
#pragma unroll
    for (unsigned k = 0; k < split_classes; ++k)
#pragma unroll
        for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) leaf[k][t][e] = 0.0F;
    const unsigned chunks = (groups + split_chunk_groups - 1U) / split_chunk_groups;
    unsigned char* tile = lds + v2_stage_bytes<q2, L>;
    // Q8_0 (the shared expert, whose rows stay in the L2 / MALL): no LDS tile; each lane reads its own row's groups
    // from memory one group ahead (as split_gate_up_kernel's Q8_0 form).
    constexpr bool direct = Format == WeightFormat::q8_0;
    [[maybe_unused]] TileRegs2<Format> tile_next{};
    [[maybe_unused]] const unsigned char* own_row = (tile_row < 32U ? tile_rows_src.base[0] : tile_rows_src.base[1]) + static_cast<std::uint64_t>(tile_row % 32U) * tile_rows_src.row_bytes;
    [[maybe_unused]] Raw<Format> ahead{};
    if constexpr (direct) ahead = load_raw<Format>(own_row, q);
    else tile_next = load_tile2<Format, L>(tile_rows_src, 0U, groups);
    v8i magic{dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic};
    if constexpr (STRIX_DS41_V2_MAGIC != 0) asm volatile("" : "+v"(magic));
#pragma unroll 1
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
        split_sync<7>(); // the previous chunk's readers are done (and, at chunk 0, row_of is visible)
        const StagedItems items = load_chunk_items(chunk, width, row_of, count, codes, deltas);
        if constexpr (!direct) {
            store_tile2<Format>(tile, tile_next);
            if (chunk + 1U < chunks) tile_next = load_tile2<Format, L>(tile_rows_src, chunk + 1U, groups);
        }
        v2_store_items<q2, L>(lds, items, chunk, width);
        split_sync<7>();
        const auto* a = reinterpret_cast<const v4i*>(lds);
        const auto* d = reinterpret_cast<const float*>(lds + split_code_bytes);
        constexpr bool pipe = Format == WeightFormat::iq2_xxs && (L & v2_pipe) != 0U;
        // two stages: the raw bytes of group k + 2 and the grid entries of group k + 1 are read during group k
        [[maybe_unused]] Iq2Pre pre{};
        [[maybe_unused]] Raw<Format> raw_next{};
        if constexpr (pipe) {
            pre = iq2_prefetch(load_raw<Format>(tile + tile_row * chunk_row_bytes<Format>, q), grid);
            raw_next = load_raw<Format>(tile + tile_row * chunk_row_bytes<Format>, q + 8U);
        }
#pragma unroll
        for (unsigned k = 0; k < split_classes; ++k) {
            const unsigned gg = q + 8U * k;
            // uniform per wave (a wave without groups still joins the barriers); also keeps the groups in separate blocks
            // (one straight-line body of the four groups spills)
            if (split_chunk_groups * chunk + gg >= groups) break;
            Raw<Format> cur;
            if constexpr (direct) {
                cur = ahead;
                const unsigned next = k + 1U < split_classes ? split_chunk_groups * chunk + gg + 8U : split_chunk_groups * (chunk + 1U) + q;
                if (next < groups) ahead = load_raw<Format>(own_row, next); // uniform per wave
            } else {
                cur = load_raw<Format>(tile + tile_row * chunk_row_bytes<Format>, gg);
            }
            Operands own;
            if constexpr (direct) {
                own = decode(cur, grid);
            } else if constexpr (pipe) {
                const Iq2Pre now = pre;
                if (k + 1U < split_classes) pre = iq2_prefetch(raw_next, grid);
                if (k + 2U < split_classes) raw_next = load_raw<Format>(tile + tile_row * chunk_row_bytes<Format>, gg + 16U);
                own = iq2_finish(now);
            } else if constexpr (Format == WeightFormat::iq2_xxs && (L & v2_decode) != 0U) own = decode_v2(cur, grid2);
            else if constexpr (Format == WeightFormat::iq2_xxs && (L & v2_signs16) != 0U) own = decode_signs(cur, grid, reinterpret_cast<const SignEntry*>(signs2));
            else if constexpr (Format == WeightFormat::iq2_xxs && (L & v2_signs) != 0U) own = decode_signs2(cur, grid, signs2);
            else own = decode_lean(cur, grid);
            const v4i x0 = v2_other_half(own.b0), x1 = v2_other_half(own.b1);
            const unsigned x_dm = q2 ? static_cast<unsigned>(v2_other_half(static_cast<int>(own.dm_bits))) : 0U;
            const unsigned x_sc = q2 ? static_cast<unsigned>(v2_other_half(static_cast<int>(own.sc_bits))) : 0U;
            const float x_scale = q2 ? 0.0F : v2_other_half(own.scale);
            const v4i a0 = a[(2U * gg) * split_rows + i16];
            const v4i a1 = a[(2U * gg + 1U) * split_rows + i16];
            const float* dt = d + gg * split_rows + 8U * h;
#pragma unroll
            for (unsigned t = 0; t < 2U; ++t) {
                const bool mine = (t == 1U) == upper; // this half decoded tile t
                const v4i b0 = pick(mine, own.b0, x0), b1 = pick(mine, own.b1, x1);
                v8i r;
                if constexpr ((L & v2_pin) != 0U) r = t == 0U ? v2_group_dot_pin<0U>(a0, b0, a1, b1, magic) : v2_group_dot_pin<1U>(a0, b0, a1, b1, magic);
                else r = STRIX_DS41_V2_MAGIC != 0 ? v2_group_dot(a0, b0, a1, b1, magic) : group_dot(a0, b0, a1, b1);
                if constexpr (q2) {
                    const unsigned dmb = mine ? own.dm_bits : x_dm, scb = mine ? own.sc_bits : x_sc;
                    const float dd = half_bits(static_cast<unsigned short>(dmb & 0xffffU));
                    const float dm = half_bits(static_cast<unsigned short>(dmb >> 16U));
                    const int m0 = static_cast<int>((scb & 0xffU) >> 4U), m1 = static_cast<int>((scb >> 8U) >> 4U);
                    if constexpr (fma) {
                        const auto* s0 = reinterpret_cast<const float*>(lds + split_code_bytes + split_delta_bytes) + gg * split_rows + 8U * h;
                        const float* s1 = s0 + split_chunk_groups * split_rows;
                        const float m0f = static_cast<float>(m0), m1f = static_cast<float>(m1);
#pragma unroll
                        for (unsigned e = 0; e < 8U; ++e) {
                            const float b = __builtin_fmaf(m1f, s1[e], __builtin_fmaf(m0f, s0[e], 0.0F));
                            const float scaled = dd * magic_float(r[e]);
                            const float offset = dm * b;
                            leaf[k][t][e] = leaf[k][t][e] + (scaled - offset) * dt[e];
                        }
                    } else {
                        const int* sp = reinterpret_cast<const int*>(lds + split_code_bytes + split_delta_bytes) + gg * split_rows + 8U * h;
#pragma unroll
                        for (unsigned e = 0; e < 8U; ++e) {
                            const int packed = sp[e];
                            const int s0 = static_cast<int>(static_cast<short>(packed & 0xffff)), s1 = packed >> 16;
                            const int b = __mul24(m0, s0) + __mul24(m1, s1);
                            const float scaled = dd * magic_float(r[e]);
                            const float offset = dm * static_cast<float>(b);
                            leaf[k][t][e] = leaf[k][t][e] + (scaled - offset) * dt[e];
                        }
                    }
                } else {
                    const float scale = mine ? own.scale : x_scale;
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) leaf[k][t][e] = leaf[k][t][e] + magic_float(r[e]) * (scale * dt[e]);
                }
            }
        }
    }
}

// The butterfly over the class sums of the eight class-waves of one pair, for the class map of class_wave_v2: levels 1-2
// (class bits 4, 3) in registers, levels 3-5 (class bits 2, 1, 0 = wave bits) through LDS; class-wave 0 holds v.
__device__ __forceinline__ void split_tree_v2(unsigned char* lds, unsigned q, unsigned pair, const float (&leaf)[split_classes][2][8],
                                              float (&v)[2][8]) {
#pragma unroll
    for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            const float c0 = leaf[0][t][e] + leaf[2][t][e], c1 = leaf[1][t][e] + leaf[3][t][e]; // classes q + q+16, q+8 + q+24
            v[t][e] = c0 + c1;
        }
    auto* x = reinterpret_cast<float*>(lds);
    const unsigned lane = threadIdx.x % 32U;
#pragma unroll
    for (unsigned s = 4U; s != 0U; s /= 2U) {
        split_sync<7>();
        if (q >= s && q < 2U * s) {
            float* slot = x + (4U * pair + (q - s)) * 16U * 32U;
#pragma unroll
            for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) slot[(8U * t + e) * 32U + lane] = v[t][e];
        }
        split_sync<7>();
        if (q < s) {
            const float* slot = x + (4U * pair + q) * 16U * 32U;
#pragma unroll
            for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) v[t][e] = v[t][e] + slot[(8U * t + e) * 32U + lane];
        }
    }
}

// Work of this block: batch x and output block y, in bands of `band` batches (band 0: x = blockIdx.x). The dispatcher
// walks blocks x fastest; within a band the linear index runs over the band's batches first, then the output blocks, so
// the blocks in flight share both the activations (a few batches) and the weights (a few output blocks).
__device__ __forceinline__ bool v2_block(const PrefillRoutesView& routes, unsigned band, Batch& b, unsigned& y, bool halves = false) {
    unsigned x = blockIdx.x;
    y = blockIdx.y;
    if (band > 1U) {
        const unsigned X = gridDim.x, Y = gridDim.y, linear = blockIdx.x + X * blockIdx.y, full = X / band;
        if (linear < full * band * Y) {
            const unsigned in = linear % (band * Y);
            x = linear / (band * Y) * band + in % band;
            y = in / band;
        } else {
            const unsigned w = X - full * band, in = linear - full * band * Y;
            x = full * band + in % w;
            y = in / w;
        }
    }
    // halves: block x runs rows 16 (x % 2) .. of the 32-row batch x / 2
    const unsigned half = halves ? x % 2U : 0U;
    if (halves) x /= 2U;
    if (x >= *routes.batch_total) return false;
    b.expert = routes.batch_expert[x];
    const unsigned count = routes.batch_count[x];
    if (count <= split_rows * half) return false;
    b.begin = routes.batch_begin[x] + split_rows * half;
    b.count = halves ? min(split_rows, count - split_rows * half) : count;
    return true;
}

template <WeightFormat Format, unsigned L>
__global__ __launch_bounds__(split_threads) __attribute__((amdgpu_waves_per_eu(STRIX_DS41_SPLIT_WAVES))) void split_gate_up_v2_kernel(PrefillGateUpView view, unsigned band) {
    extern __shared__ int4 dynamic_lds[];
    auto* lds = reinterpret_cast<unsigned char*>(dynamic_lds);
    constexpr bool dec = Format == WeightFormat::iq2_xxs && (L & v2_decode) != 0U;
    __shared__ Grid2 grid2[dec ? 256 : 1];
    __shared__ unsigned long long grid[Format == WeightFormat::iq2_xxs && !dec ? 256 : 1];
    constexpr bool sg = Format == WeightFormat::iq2_xxs && !dec && (L & (v2_signs | v2_signs16)) != 0U;
    constexpr bool sg16 = (L & v2_signs16) != 0U;
    __shared__ uint2 signs2[sg ? (sg16 ? 256 : 128) : 1];
    __shared__ unsigned row_of[split_rows], slot_of[split_rows];
    __shared__ float values[2][split_rows][32];
    Batch batch;
    unsigned yb = 0;
    if (!v2_block(view.routes, band, batch, yb)) return;
    if constexpr (dec) load_grid2(grid2);
    else if constexpr (Format == WeightFormat::iq2_xxs) load_grid(grid);
    if constexpr (sg16) load_signs(reinterpret_cast<SignEntry*>(signs2));
    else if constexpr (sg) load_signs2(signs2);
    if (threadIdx.x < batch.count) {
        row_of[threadIdx.x] = view.routes.entry_rows[batch.begin + threadIdx.x];
        slot_of[threadIdx.x] = view.routes.entry_slots[batch.begin + threadIdx.x];
    }
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U, m = wave & 1U, q = wave >> 1U, h = lane >> 4U;
    const std::uint64_t expert_at = batch.expert * view.expert_stride_bytes + static_cast<std::uint64_t>(yb) * 32U * view.row_bytes;
    const TileRows rows{{static_cast<const unsigned char*>(view.gate) + expert_at, static_cast<const unsigned char*>(view.up) + expert_at},
                        view.row_bytes};
    float leaf[split_classes][2][8];
    class_wave_v2<Format, L>(lds, q, view.hidden, row_of, batch.count, view.codes, view.deltas, grid, grid2, leaf, rows, 32U * m + lane, signs2);
    float v[2][8];
    split_tree_v2(lds, q, m, leaf, v);
    if (q == 0U)
#pragma unroll
        for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const unsigned r = 2U * e + h;
                if (r < batch.count) values[m][r][16U * t + (lane & 15U)] = v[t][e];
            }
    __syncthreads();
    const unsigned groups = view.inter / 32U, at = yb * 32U + lane;
    for (unsigned e = wave; e < batch.count; e += split_threads / 32U) {
        const unsigned long long slot = slot_of[e];
        const float g = values[0][e][lane], u = values[1][e][lane];
        if (view.gate_out) { view.gate_out[slot * view.inter + at] = g; view.up_out[slot * view.inter + at] = u; }
        const float act = swiglu_clamped(g, u, view.limit);
        float amax = fabsf(act);
        for (int stride = 16; stride; stride /= 2) amax = fmaxf(amax, __shfl_xor(amax, stride, 32));
        const bool tiny = amax > 0.0F && amax < 0x1p-119F;
        const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
        const float rounded = tiny ? 0.0F : rintf(act / delta);
        const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
        int sum = code;
        for (int stride = 16; stride; stride /= 2) sum += __shfl_xor(sum, stride, 32);
        view.act_codes[slot * view.inter + at] = static_cast<signed char>(code);
        if (lane == 0U) { view.act_deltas[slot * groups + yb] = delta; view.act_sums[slot * groups + yb] = sum; }
    }
}

template <WeightFormat Format, unsigned L>
__global__ __launch_bounds__(split_threads) __attribute__((amdgpu_waves_per_eu(STRIX_DS41_SPLIT_DOWN_WAVES))) void split_down_v2_kernel(PrefillDownView view, unsigned band) {
    extern __shared__ int4 dynamic_lds[];
    auto* lds = reinterpret_cast<unsigned char*>(dynamic_lds);
    __shared__ unsigned slot_of[split_rows];
    Batch batch;
    unsigned yb = 0;
    if (!v2_block(view.routes, band, batch, yb, view.routes.batch_rows == 32U)) return;
    if (threadIdx.x < batch.count) slot_of[threadIdx.x] = view.routes.entry_slots[batch.begin + threadIdx.x];
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U, p = wave & 1U, q = wave >> 1U, h = lane >> 4U;
    const unsigned first = yb * 64U + 32U * p;
    const auto* tile_base = static_cast<const unsigned char*>(view.weights) + batch.expert * view.expert_stride_bytes +
        static_cast<std::uint64_t>(yb) * 64U * view.row_bytes;
    const TileRows rows{{tile_base, tile_base + 32U * view.row_bytes}, view.row_bytes};
    float leaf[split_classes][2][8];
    class_wave_v2<Format, L>(lds, q, view.width, slot_of, batch.count, view.codes, view.deltas, nullptr, nullptr, leaf, rows, 32U * p + lane);
    float v[2][8];
    split_tree_v2(lds, q, p, leaf, v);
    if (q != 0U) return;
#pragma unroll
    for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            const unsigned r = 2U * e + h;
            if (r < batch.count) view.out[static_cast<unsigned long long>(slot_of[r]) * view.outputs + first + 16U * t + (lane & 15U)] = v[t][e];
        }
}

// ---- Gate/up R2 (STRIX_DS41_PREFILL_MOE_R2, opt-in): 32-row batches, each decoded weight group feeds two row tiles ----
//
// A workgroup covers a batch of up to 32 rows x 32 outputs of both matrices in two passes over the hidden width: pass p runs
// the lane classes with class bit 0 = p (the tree's LAST level), so only 16 classes are live at a time (64 accumulators
// per lane, as split_gate_up_kernel). Class-wave q (0..7) owns classes p + 2q + 16k (k = 0, 1): tree level 1 (bit 4) in
// registers, levels 2-4 (bits 3, 2, 1 = wave bits 2, 1, 0) through LDS; level 5 adds the two passes' subtrees. Each lane
// decodes its output's group once per pass and chunk (lean decoder) and both row tiles use it (half the decode per WMMA). The same
// products, chains, tree and epilogue as split_gate_up_kernel: the same bits.
#ifndef STRIX_DS41_R2_MAGIC
#define STRIX_DS41_R2_MAGIC 1
#endif
constexpr unsigned r2_rows = 32U, r2_pass_groups = 16U;
constexpr unsigned r2_code_bytes = r2_pass_groups * 2U * r2_rows * 16U;  // 16 KB
constexpr unsigned r2_delta_bytes = r2_pass_groups * r2_rows * 4U;       // 2 KB
constexpr unsigned r2_stage_bytes = r2_code_bytes + r2_delta_bytes;
constexpr unsigned r2_exchange_bytes = 2U * 4U * 32U * 32U * 4U;         // 32 KB: 2 matrices x 4 slots x 32 values x 32 lanes
template <WeightFormat Format> inline constexpr unsigned r2_lds =
    r2_stage_bytes + tile_rows * chunk_row_bytes<Format> > r2_exchange_bytes ? r2_stage_bytes + tile_rows * chunk_row_bytes<Format> : r2_exchange_bytes;

template <WeightFormat Format>
__global__ __launch_bounds__(split_threads) __attribute__((amdgpu_waves_per_eu(STRIX_DS41_SPLIT_WAVES))) void r2_gate_up_kernel(PrefillGateUpView view, unsigned band) {
    extern __shared__ int4 dynamic_lds[];
    auto* lds = reinterpret_cast<unsigned char*>(dynamic_lds);
    __shared__ unsigned long long grid[256];
    __shared__ unsigned row_of[r2_rows], slot_of[r2_rows];
    __shared__ float values[2][r2_rows][32];
    Batch batch;
    unsigned yb = 0;
    if (!v2_block(view.routes, band, batch, yb)) return;
    load_grid(grid);
    if (threadIdx.x < batch.count) {
        row_of[threadIdx.x] = view.routes.entry_rows[batch.begin + threadIdx.x];
        slot_of[threadIdx.x] = view.routes.entry_slots[batch.begin + threadIdx.x];
    }
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U, m = wave & 1U, q = wave >> 1U, h = lane >> 4U, i16 = lane & 15U;
    const bool upper = h != 0U, two = batch.count > 16U; // uniform
    const unsigned width = view.hidden, groups = width / 32U, chunks = groups / split_chunk_groups, count = batch.count;
    const std::uint64_t expert_at = batch.expert * view.expert_stride_bytes + static_cast<std::uint64_t>(yb) * 32U * view.row_bytes;
    const TileRows src{{static_cast<const unsigned char*>(view.gate) + expert_at, static_cast<const unsigned char*>(view.up) + expert_at},
                       view.row_bytes};
    const unsigned tile_row = 32U * m + lane;
    unsigned char* tile = lds + r2_stage_bytes;
    const auto* a = reinterpret_cast<const v4i*>(lds);
    const auto* d = reinterpret_cast<const float*>(lds + r2_code_bytes);
    v8i magic{dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic};
    if constexpr (STRIX_DS41_R2_MAGIC != 0) asm volatile("" : "+v"(magic));
    TileRegs2<Format> tile_next = load_tile2<Format, v2_on>(src, 0U, groups);
#pragma unroll 1
    for (unsigned pass = 0; pass < 2U; ++pass) {
        float leaf[2][2][2][8]; // [k][output tile][row tile][e]
#pragma unroll
        for (unsigned k = 0; k < 2U; ++k)
#pragma unroll
            for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
                for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) leaf[k][t][rt][e] = 0.0F;
#pragma unroll 1
        for (unsigned chunk = 0; chunk < chunks; ++chunk) {
            split_sync<7>();
            // stage: the pass's 16 groups (pass + 2 gl) x 2 halves x 32 rows, two items per thread, read before the tile prefetch
            v4i xv[2];
            float xd[2];
#pragma unroll
            for (unsigned i = 0; i < 2U; ++i) {
                const unsigned item = threadIdx.x + split_threads * i;
                const unsigned r = item % r2_rows, kh = (item / r2_rows) % 2U, gl = item / (2U * r2_rows);
                const unsigned g = split_chunk_groups * chunk + pass + 2U * gl;
                xv[i] = v4i{0, 0, 0, 0};
                xd[i] = 0.0F;
                if (r < count) {
                    xv[i] = *reinterpret_cast<const v4i*>(view.codes + static_cast<unsigned long long>(row_of[r]) * width + 32U * g + 16U * kh);
                    if (kh == 0U) xd[i] = view.deltas[static_cast<unsigned long long>(row_of[r]) * groups + g];
                }
            }
            store_tile2<Format>(tile, tile_next);
            // the next step's tile: this pass's next chunk, else the next pass's first chunk
            const unsigned next = chunk + 1U < chunks ? chunk + 1U : pass == 0U ? 0U : chunks;
            tile_next = load_tile2<Format, v2_on>(src, next, groups);
#pragma unroll
            for (unsigned i = 0; i < 2U; ++i) {
                const unsigned item = threadIdx.x + split_threads * i;
                const unsigned r = item % r2_rows, kh = (item / r2_rows) % 2U, gl = item / (2U * r2_rows);
                reinterpret_cast<v4i*>(lds)[(2U * gl + kh) * r2_rows + r] = xv[i];
                if (kh == 0U) reinterpret_cast<float*>(lds + r2_code_bytes)[gl * r2_rows + tile_position(r)] = xd[i];
            }
            split_sync<7>();
#pragma unroll
            for (unsigned k = 0; k < 2U; ++k) {
                const unsigned gl = q + 8U * k, gg = pass + 2U * gl; // class pass + 2q + 16k
                const Raw<Format> cur = load_raw<Format>(tile + tile_row * chunk_row_bytes<Format>, gg);
                const Operands own = decode_lean(cur, grid);
                const v4i x0 = v2_other_half(own.b0), x1 = v2_other_half(own.b1);
                const float x_scale = v2_other_half(own.scale);
                v4i bt0[2], bt1[2];
                float sc[2];
#pragma unroll
                for (unsigned t = 0; t < 2U; ++t) {
                    const bool mine = (t == 1U) == upper;
                    bt0[t] = pick(mine, own.b0, x0);
                    bt1[t] = pick(mine, own.b1, x1);
                    sc[t] = mine ? own.scale : x_scale;
                }
                const auto row_tile = [&](auto rt) {
                    const v4i a0 = a[(2U * gl) * r2_rows + 16U * rt + i16];
                    const v4i a1 = a[(2U * gl + 1U) * r2_rows + 16U * rt + i16];
                    const float* dt = d + gl * r2_rows + 16U * rt + 8U * h;
#pragma unroll
                    for (unsigned t = 0; t < 2U; ++t) {
                        const v8i r = STRIX_DS41_R2_MAGIC != 0 ? v2_group_dot(a0, bt0[t], a1, bt1[t], magic) : group_dot(a0, bt0[t], a1, bt1[t]);
#pragma unroll
                        for (unsigned e = 0; e < 8U; ++e) leaf[k][t][rt][e] = leaf[k][t][rt][e] + magic_float(r[e]) * (sc[t] * dt[e]);
                    }
                };
                row_tile(std::integral_constant<unsigned, 0U>{});
                if (two) row_tile(std::integral_constant<unsigned, 1U>{}); // uniform
            }
        }
        // tree: level 1 (bit 4) in registers, levels 2-4 through LDS, class-wave 0 holds the pass's subtree
        float v[2][2][8];
#pragma unroll
        for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
            for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) v[t][rt][e] = leaf[0][t][rt][e] + leaf[1][t][rt][e];
        auto* x = reinterpret_cast<float*>(lds);
        // the lane index made opaque per pass: the exchange addresses are formed here, not hoisted out of the pass loop
        // (32 live addresses through the chunk loop spill)
        unsigned xl = lane;
        asm volatile("" : "+v"(xl));
#pragma unroll
        for (unsigned s = 4U; s != 0U; s /= 2U) {
            split_sync<7>();
            if (q >= s && q < 2U * s) {
                float* slot = x + (4U * m + (q - s)) * 32U * 32U;
#pragma unroll
                for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
                    for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                        for (unsigned e = 0; e < 8U; ++e) slot[(16U * t + 8U * rt + e) * 32U + xl] = v[t][rt][e];
            }
            split_sync<7>();
            if (q < s) {
                const float* slot = x + (4U * m + q) * 32U * 32U;
#pragma unroll
                for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
                    for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                        for (unsigned e = 0; e < 8U; ++e) v[t][rt][e] = v[t][rt][e] + slot[(16U * t + 8U * rt + e) * 32U + xl];
            }
        }
        if (q == 0U)
#pragma unroll
            for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
                for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) {
                        const unsigned r = 16U * rt + 2U * e + h;
                        if (r < count) {
                            float& out = values[m][r][16U * t + (xl & 15U)];
                            out = pass == 0U ? v[t][rt][e] : out + v[t][rt][e]; // level 5: even classes + odd classes
                        }
                    }
    }
    __syncthreads();
    const unsigned at = yb * 32U + lane;
    for (unsigned e = wave; e < count; e += split_threads / 32U) {
        const unsigned long long slot = slot_of[e];
        const float g = values[0][e][lane], u = values[1][e][lane];
        if (view.gate_out) { view.gate_out[slot * view.inter + at] = g; view.up_out[slot * view.inter + at] = u; }
        const float act = swiglu_clamped(g, u, view.limit);
        float amax = fabsf(act);
        for (int stride = 16; stride; stride /= 2) amax = fmaxf(amax, __shfl_xor(amax, stride, 32));
        const bool tiny = amax > 0.0F && amax < 0x1p-119F;
        const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
        const float rounded = tiny ? 0.0F : rintf(act / delta);
        const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
        int sum = code;
        for (int stride = 16; stride; stride /= 2) sum += __shfl_xor(sum, stride, 32);
        view.act_codes[slot * view.inter + at] = static_cast<signed char>(code);
        if (lane == 0U) { view.act_deltas[slot * (view.inter / 32U) + yb] = delta; view.act_sums[slot * (view.inter / 32U) + yb] = sum; }
    }
}

// ---- Gate/up W2 (STRIX_DS41_PREFILL_MOE_R2=1 with STRIX_DS41_PREFILL_MOE_R2_GU=w, opt-in): 32-row batches in one pass ----
//
// A workgroup of 32 waves covers a batch of up to 32 rows x 32 outputs of both matrices; wave (q, m, t) = class-wave q
// (classes q + 8k, as class_wave_v2), matrix m, output tile t, and holds both row tiles (64 accumulators per lane). For
// each pair of classes (2p, 2p + 1), half-wave h decodes its output's group of class 2p + h (one decode per lane), the
// halves exchange (v_permlanex16), and both groups feed both row tiles: 8 WMMAs per decode instead of 4. Every leaf (row,
// output, class) is the chain of split_gate_up_v2_kernel (same products, epilogue and order), the tree is split_tree_v2's
// per (row, output): the same bits. Batches of up to 16 rows run the first row tile only.
constexpr unsigned w2_threads = 1024U, w2_rows = 32U;
constexpr unsigned w2_code_bytes = split_chunk_groups * 2U * w2_rows * 16U; // 32 KB
constexpr unsigned w2_delta_bytes = split_chunk_groups * w2_rows * 4U;      // 4 KB
constexpr unsigned w2_stage_bytes = w2_code_bytes + w2_delta_bytes;
constexpr unsigned w2_exchange_bytes = 4U * 4U * 16U * 32U * 4U;            // 32 KB: 4 (m, t) x 4 slots x 16 values x 32 lanes
template <WeightFormat Format> inline constexpr unsigned w2_lds =
    w2_stage_bytes + tile_rows * chunk_row_bytes<Format> > w2_exchange_bytes ? w2_stage_bytes + tile_rows * chunk_row_bytes<Format> : w2_exchange_bytes;
template <WeightFormat Format> inline constexpr unsigned w2_loads = (run_words<Format> + w2_threads - 1U) / w2_threads;
template <WeightFormat Format> struct TileRegsW { unsigned w[2][w2_loads<Format>]; };
template <WeightFormat Format>
__device__ __forceinline__ TileRegsW<Format> load_tile_w(const TileRows& t, unsigned chunk) {
    constexpr unsigned row_words = chunk_row_bytes<Format> / 4U;
    TileRegsW<Format> r{};
    const auto* b0 = t.base[0] + static_cast<std::uint64_t>(chunk) * chunk_row_bytes<Format>;
    const auto* b1 = t.base[1] + static_cast<std::uint64_t>(chunk) * chunk_row_bytes<Format>;
    const auto rb = static_cast<unsigned>(t.row_bytes);
#pragma unroll
    for (unsigned i = 0; i < w2_loads<Format>; ++i) {
        const unsigned word = threadIdx.x + w2_threads * i;
        if (word < run_words<Format>) {
            const unsigned o = word / row_words * rb + 4U * (word % row_words);
            r.w[0][i] = *reinterpret_cast<const unsigned*>(b0 + o);
            r.w[1][i] = *reinterpret_cast<const unsigned*>(b1 + o);
        }
    }
    return r;
}
template <WeightFormat Format>
__device__ __forceinline__ void store_tile_w(unsigned char* tile, const TileRegsW<Format>& r) {
#pragma unroll
    for (unsigned i = 0; i < w2_loads<Format>; ++i) {
        const unsigned word = threadIdx.x + w2_threads * i;
        if (word < run_words<Format>) {
            reinterpret_cast<unsigned*>(tile)[word] = r.w[0][i];
            reinterpret_cast<unsigned*>(tile)[run_words<Format> + word] = r.w[1][i];
        }
    }
}
template <WeightFormat Format, unsigned L>
__global__ __launch_bounds__(w2_threads) __attribute__((amdgpu_waves_per_eu(8))) void w2_gate_up_kernel(PrefillGateUpView view, unsigned band) {
    extern __shared__ int4 dynamic_lds[];
    auto* lds = reinterpret_cast<unsigned char*>(dynamic_lds);
    constexpr bool sg = (L & v2_signs) != 0U;
    __shared__ unsigned long long grid[256];
    __shared__ uint2 signs2[sg ? 128 : 1];
    __shared__ unsigned row_of[w2_rows], slot_of[w2_rows];
    __shared__ float values[2][w2_rows][32];
    Batch batch;
    unsigned yb = 0;
    if (!v2_block(view.routes, band, batch, yb)) return;
    load_grid(grid);
    if constexpr (sg) load_signs2(signs2);
    if (threadIdx.x < batch.count) {
        row_of[threadIdx.x] = view.routes.entry_rows[batch.begin + threadIdx.x];
        slot_of[threadIdx.x] = view.routes.entry_slots[batch.begin + threadIdx.x];
    }
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U, m = wave & 1U, t = (wave >> 1U) & 1U, q = wave >> 2U;
    const unsigned h = lane >> 4U, i16 = lane & 15U;
    const bool two = batch.count > 16U; // uniform
    const unsigned width = view.hidden, groups = width / 32U, chunks = groups / split_chunk_groups, count = batch.count;
    const std::uint64_t expert_at = batch.expert * view.expert_stride_bytes + static_cast<std::uint64_t>(yb) * 32U * view.row_bytes;
    const TileRows src{{static_cast<const unsigned char*>(view.gate) + expert_at, static_cast<const unsigned char*>(view.up) + expert_at},
                       view.row_bytes};
    const unsigned tile_row = 32U * m + 16U * t + i16;
    unsigned char* tile = lds + w2_stage_bytes;
    const auto* a = reinterpret_cast<const v4i*>(lds);
    const auto* d = reinterpret_cast<const float*>(lds + w2_code_bytes);
    v8i magic{dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic};
    asm volatile("" : "+v"(magic));
    float leaf[split_classes][2][8]; // [k][row tile][e]
#pragma unroll
    for (unsigned k = 0; k < split_classes; ++k)
#pragma unroll
        for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) leaf[k][rt][e] = 0.0F;
    TileRegsW<Format> tile_next = load_tile_w<Format>(src, 0U);
#pragma unroll 1
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
        split_sync<7>(); // the previous chunk's readers are done (and, at chunk 0, row_of is visible)
        v4i xv[2];
        float xd[2];
#pragma unroll
        for (unsigned i = 0; i < 2U; ++i) {
            const unsigned item = threadIdx.x + w2_threads * i;
            const unsigned r = item % w2_rows, kh = (item / w2_rows) % 2U, gg = item / (2U * w2_rows);
            const unsigned g = split_chunk_groups * chunk + gg;
            xv[i] = v4i{0, 0, 0, 0};
            xd[i] = 0.0F;
            if (r < count) {
                xv[i] = *reinterpret_cast<const v4i*>(view.codes + static_cast<unsigned long long>(row_of[r]) * width + 32U * g + 16U * kh);
                if (kh == 0U) xd[i] = view.deltas[static_cast<unsigned long long>(row_of[r]) * groups + g];
            }
        }
        store_tile_w<Format>(tile, tile_next);
        if (chunk + 1U < chunks) tile_next = load_tile_w<Format>(src, chunk + 1U);
#pragma unroll
        for (unsigned i = 0; i < 2U; ++i) {
            const unsigned item = threadIdx.x + w2_threads * i;
            const unsigned r = item % w2_rows, kh = (item / w2_rows) % 2U, gg = item / (2U * w2_rows);
            reinterpret_cast<v4i*>(lds)[(2U * gg + kh) * w2_rows + r] = xv[i];
            if (kh == 0U) reinterpret_cast<float*>(lds + w2_code_bytes)[gg * w2_rows + tile_position(r)] = xd[i];
        }
        split_sync<7>();
#pragma unroll
        for (unsigned p = 0; p < 2U; ++p) {
            // half h decodes class 2p + h of its output (a lane-dependent group: one decode per lane)
            const Raw<Format> cur = load_raw<Format>(tile + tile_row * chunk_row_bytes<Format>, q + 8U * (2U * p + h));
            Operands own;
            if constexpr (sg) own = decode_signs2(cur, grid, signs2);
            else own = decode_lean(cur, grid);
            const v4i x0 = v2_other_half(own.b0), x1 = v2_other_half(own.b1);
            const float x_scale = v2_other_half(own.scale);
#pragma unroll
            for (unsigned kk = 0; kk < 2U; ++kk) {
                const unsigned k = 2U * p + kk, gg = q + 8U * k;
                const bool mine = h == kk; // this half decoded class k
                const v4i b0 = pick(mine, own.b0, x0), b1 = pick(mine, own.b1, x1);
                const float scale = mine ? own.scale : x_scale;
#pragma unroll
                for (unsigned rt = 0; rt < 2U; ++rt) {
                    if (rt == 1U && !two) break;
                    const v4i a0 = a[(2U * gg) * w2_rows + 16U * rt + i16];
                    const v4i a1 = a[(2U * gg + 1U) * w2_rows + 16U * rt + i16];
                    const float* dt = d + gg * w2_rows + 16U * rt + 8U * h;
                    v8i r;
                    if constexpr ((L & v2_pin) != 0U) r = rt == 0U ? v2_group_dot_pin<0U>(a0, b0, a1, b1, magic) : v2_group_dot_pin<1U>(a0, b0, a1, b1, magic);
                    else r = v2_group_dot(a0, b0, a1, b1, magic);
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) leaf[k][rt][e] = leaf[k][rt][e] + magic_float(r[e]) * (scale * dt[e]);
                }
            }
        }
    }
    // tree (split_tree_v2 per (row, output)): classes q + q+16 and q+8 + q+24 in registers, then the wave bits through LDS
    float v[2][8];
#pragma unroll
    for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            const float c0 = leaf[0][rt][e] + leaf[2][rt][e], c1 = leaf[1][rt][e] + leaf[3][rt][e];
            v[rt][e] = c0 + c1;
        }
    auto* x = reinterpret_cast<float*>(lds);
    const unsigned pair = 2U * t + m;
#pragma unroll
    for (unsigned s = 4U; s != 0U; s /= 2U) {
        split_sync<7>();
        if (q >= s && q < 2U * s) {
            float* slot = x + (4U * pair + (q - s)) * 16U * 32U;
#pragma unroll
            for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) slot[(8U * rt + e) * 32U + lane] = v[rt][e];
        }
        split_sync<7>();
        if (q < s) {
            const float* slot = x + (4U * pair + q) * 16U * 32U;
#pragma unroll
            for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) v[rt][e] = v[rt][e] + slot[(8U * rt + e) * 32U + lane];
        }
    }
    if (q == 0U)
#pragma unroll
        for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const unsigned r = 16U * rt + 2U * e + h;
                if (r < count) values[m][r][16U * t + i16] = v[rt][e];
            }
    __syncthreads();
    const unsigned at = yb * 32U + lane;
    for (unsigned e = wave; e < count; e += w2_threads / 32U) {
        const unsigned long long slot = slot_of[e];
        const float g = values[0][e][lane], u = values[1][e][lane];
        if (view.gate_out) { view.gate_out[slot * view.inter + at] = g; view.up_out[slot * view.inter + at] = u; }
        const float act = swiglu_clamped(g, u, view.limit);
        float amax = fabsf(act);
        for (int stride = 16; stride; stride /= 2) amax = fmaxf(amax, __shfl_xor(amax, stride, 32));
        const bool tiny = amax > 0.0F && amax < 0x1p-119F;
        const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
        const float rounded = tiny ? 0.0F : rintf(act / delta);
        const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
        int sum = code;
        for (int stride = 16; stride; stride /= 2) sum += __shfl_xor(sum, stride, 32);
        view.act_codes[slot * view.inter + at] = static_cast<signed char>(code);
        if (lane == 0U) { view.act_deltas[slot * (view.inter / 32U) + yb] = delta; view.act_sums[slot * (view.inter / 32U) + yb] = sum; }
    }
}

// Down R2 (Q2_K, with STRIX_DS41_PREFILL_MOE_R2): a 32-row batch x 64 outputs per workgroup in two passes over the width
// (class bit 0 = pass), class-wave q owning classes pass + 2q + 16k, output pair p = wave % 2; each decoded group feeds
// both row tiles. Q2_K epilogue of split_down_v2_kernel (V2_FMA), partial last chunk tested per group (uniform per wave)
// and loaded by its blocks only. The same products, chains, tree and epilogue as split_down_kernel: the same bits.
#ifndef STRIX_DS41_R2D_MAGIC
#define STRIX_DS41_R2D_MAGIC 1
#endif
constexpr unsigned r2d_sum_bytes = 2U * r2_pass_groups * r2_rows * 4U;  // S0, S1 as floats: 4 KB
constexpr unsigned r2d_stage_bytes = r2_code_bytes + r2_delta_bytes + r2d_sum_bytes;
constexpr unsigned r2d_lds = r2d_stage_bytes + tile_rows * chunk_row_bytes<WeightFormat::q2_k> > r2_exchange_bytes
                                 ? r2d_stage_bytes + tile_rows * chunk_row_bytes<WeightFormat::q2_k> : r2_exchange_bytes;
__global__ __launch_bounds__(split_threads) __attribute__((amdgpu_waves_per_eu(STRIX_DS41_SPLIT_DOWN_WAVES))) void r2_down_kernel(PrefillDownView view, unsigned band) {
    constexpr WeightFormat Format = WeightFormat::q2_k;
    constexpr unsigned L = v2_on | v2_fma | v2_part;
    extern __shared__ int4 dynamic_lds[];
    auto* lds = reinterpret_cast<unsigned char*>(dynamic_lds);
    __shared__ unsigned slot_of[r2_rows];
    __shared__ float values[r2_rows][64];
    Batch batch;
    unsigned yb = 0;
    if (!v2_block(view.routes, band, batch, yb)) return;
    if (threadIdx.x < batch.count) slot_of[threadIdx.x] = view.routes.entry_slots[batch.begin + threadIdx.x];
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U, p = wave & 1U, q = wave >> 1U, h = lane >> 4U, i16 = lane & 15U;
    const bool upper = h != 0U, two = batch.count > 16U; // uniform
    const unsigned width = view.width, groups = width / 32U, chunks = (groups + split_chunk_groups - 1U) / split_chunk_groups, count = batch.count;
    const auto* tile_base = static_cast<const unsigned char*>(view.weights) + batch.expert * view.expert_stride_bytes +
        static_cast<std::uint64_t>(yb) * 64U * view.row_bytes;
    const TileRows src{{tile_base, tile_base + 32U * view.row_bytes}, view.row_bytes};
    const unsigned tile_row = 32U * p + lane;
    unsigned char* tile = lds + r2d_stage_bytes;
    const auto* a = reinterpret_cast<const v4i*>(lds);
    const auto* d = reinterpret_cast<const float*>(lds + r2_code_bytes);
    const auto* sums = reinterpret_cast<const float*>(lds + r2_code_bytes + r2_delta_bytes);
    v8i magic{dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic};
    if constexpr (STRIX_DS41_R2D_MAGIC != 0) asm volatile("" : "+v"(magic));
    TileRegs2<Format> tile_next = load_tile2<Format, L>(src, 0U, groups);
#pragma unroll 1
    for (unsigned pass = 0; pass < 2U; ++pass) {
        float leaf[2][2][2][8]; // [k][output tile][row tile][e]
#pragma unroll
        for (unsigned k = 0; k < 2U; ++k)
#pragma unroll
            for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
                for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) leaf[k][t][rt][e] = 0.0F;
#pragma unroll 1
        for (unsigned chunk = 0; chunk < chunks; ++chunk) {
            split_sync<7>();
            v4i xv[2];
            float xd[2];
#pragma unroll
            for (unsigned i = 0; i < 2U; ++i) {
                const unsigned item = threadIdx.x + split_threads * i;
                const unsigned r = item % r2_rows, kh = (item / r2_rows) % 2U, gl = item / (2U * r2_rows);
                const unsigned g = split_chunk_groups * chunk + pass + 2U * gl;
                xv[i] = v4i{0, 0, 0, 0};
                xd[i] = 0.0F;
                if (g < groups && r < count) {
                    xv[i] = *reinterpret_cast<const v4i*>(view.codes + static_cast<unsigned long long>(slot_of[r]) * width + 32U * g + 16U * kh);
                    if (kh == 0U) xd[i] = view.deltas[static_cast<unsigned long long>(slot_of[r]) * groups + g];
                }
            }
            store_tile2<Format>(tile, tile_next);
            const unsigned next = chunk + 1U < chunks ? chunk + 1U : pass == 0U ? 0U : chunks;
            tile_next = load_tile2<Format, L>(src, next, groups);
#pragma unroll
            for (unsigned i = 0; i < 2U; ++i) {
                const unsigned item = threadIdx.x + split_threads * i;
                const unsigned r = item % r2_rows, kh = (item / r2_rows) % 2U, gl = item / (2U * r2_rows);
                const v4i v = xv[i];
                reinterpret_cast<v4i*>(lds)[(2U * gl + kh) * r2_rows + r] = v;
                const unsigned at = tile_position(r);
                if (kh == 0U) reinterpret_cast<float*>(lds + r2_code_bytes)[gl * r2_rows + at] = xd[i];
                const int ones = 0x01010101;
                int sum = 0;
                sum = __builtin_amdgcn_sudot4(false, ones, true, v.x, sum, false);
                sum = __builtin_amdgcn_sudot4(false, ones, true, v.y, sum, false);
                sum = __builtin_amdgcn_sudot4(false, ones, true, v.z, sum, false);
                sum = __builtin_amdgcn_sudot4(false, ones, true, v.w, sum, false);
                reinterpret_cast<float*>(lds + r2_code_bytes + r2_delta_bytes)[kh * r2_pass_groups * r2_rows + gl * r2_rows + at] =
                    static_cast<float>(sum); // exact (|sum| <= 2032)
            }
            split_sync<7>();
#pragma unroll
            for (unsigned k = 0; k < 2U; ++k) {
                const unsigned gl = q + 8U * k, gg = pass + 2U * gl; // class pass + 2q + 16k
                if (split_chunk_groups * chunk + gg >= groups) break; // uniform per wave (partial last chunk)
                const Raw<Format> cur = load_raw<Format>(tile + tile_row * chunk_row_bytes<Format>, gg);
                const Operands own = decode_lean(cur, nullptr);
                const v4i x0 = v2_other_half(own.b0), x1 = v2_other_half(own.b1);
                const unsigned x_dm = static_cast<unsigned>(v2_other_half(static_cast<int>(own.dm_bits)));
                const unsigned x_sc = static_cast<unsigned>(v2_other_half(static_cast<int>(own.sc_bits)));
                v4i bt0[2], bt1[2];
                float dd[2], dm[2], m0f[2], m1f[2];
#pragma unroll
                for (unsigned t = 0; t < 2U; ++t) {
                    const bool mine = (t == 1U) == upper;
                    bt0[t] = pick(mine, own.b0, x0);
                    bt1[t] = pick(mine, own.b1, x1);
                    const unsigned dmb = mine ? own.dm_bits : x_dm, scb = mine ? own.sc_bits : x_sc;
                    dd[t] = half_bits(static_cast<unsigned short>(dmb & 0xffffU));
                    dm[t] = half_bits(static_cast<unsigned short>(dmb >> 16U));
                    m0f[t] = static_cast<float>(static_cast<int>((scb & 0xffU) >> 4U));
                    m1f[t] = static_cast<float>(static_cast<int>((scb >> 8U) >> 4U));
                }
                const auto row_tile = [&](auto rt) {
                    const v4i a0 = a[(2U * gl) * r2_rows + 16U * rt + i16];
                    const v4i a1 = a[(2U * gl + 1U) * r2_rows + 16U * rt + i16];
                    const float* dt = d + gl * r2_rows + 16U * rt + 8U * h;
                    const float* s0 = sums + gl * r2_rows + 16U * rt + 8U * h;
                    const float* s1 = s0 + r2_pass_groups * r2_rows;
#pragma unroll
                    for (unsigned t = 0; t < 2U; ++t) {
                        const v8i r = STRIX_DS41_R2D_MAGIC != 0 ? v2_group_dot(a0, bt0[t], a1, bt1[t], magic) : group_dot(a0, bt0[t], a1, bt1[t]);
#pragma unroll
                        for (unsigned e = 0; e < 8U; ++e) {
                            const float b = __builtin_fmaf(m1f[t], s1[e], __builtin_fmaf(m0f[t], s0[e], 0.0F));
                            const float scaled = dd[t] * magic_float(r[e]);
                            const float offset = dm[t] * b;
                            leaf[k][t][rt][e] = leaf[k][t][rt][e] + (scaled - offset) * dt[e];
                        }
                    }
                };
                row_tile(std::integral_constant<unsigned, 0U>{});
                if (two) row_tile(std::integral_constant<unsigned, 1U>{}); // uniform
            }
        }
        float v[2][2][8];
#pragma unroll
        for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
            for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) v[t][rt][e] = leaf[0][t][rt][e] + leaf[1][t][rt][e];
        auto* x = reinterpret_cast<float*>(lds);
        unsigned xl = lane;
        asm volatile("" : "+v"(xl));
#pragma unroll
        for (unsigned s = 4U; s != 0U; s /= 2U) {
            split_sync<7>();
            if (q >= s && q < 2U * s) {
                float* slot = x + (4U * p + (q - s)) * 32U * 32U;
#pragma unroll
                for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
                    for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                        for (unsigned e = 0; e < 8U; ++e) slot[(16U * t + 8U * rt + e) * 32U + xl] = v[t][rt][e];
            }
            split_sync<7>();
            if (q < s) {
                const float* slot = x + (4U * p + q) * 32U * 32U;
#pragma unroll
                for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
                    for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                        for (unsigned e = 0; e < 8U; ++e) v[t][rt][e] = v[t][rt][e] + slot[(16U * t + 8U * rt + e) * 32U + xl];
            }
        }
        if (q == 0U)
#pragma unroll
            for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
                for (unsigned rt = 0; rt < 2U; ++rt)
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) {
                        const unsigned r = 16U * rt + 2U * e + h;
                        if (r < count) {
                            float& out = values[r][32U * p + 16U * t + (xl & 15U)];
                            out = pass == 0U ? v[t][rt][e] : out + v[t][rt][e]; // level 5: even classes + odd classes
                        }
                    }
    }
    __syncthreads();
    for (unsigned e = wave; e < count; e += split_threads / 32U) {
        float* o = view.out + static_cast<unsigned long long>(slot_of[e]) * view.outputs + yb * 64U;
        o[lane] = values[e][lane];
        o[32U + lane] = values[e][32U + lane];
    }
}

// ---- Router of T rows ----
//
// The decode router's declared order per (row, expert): lane l accumulates acc = acc + w * x over the
// elements l, l + 32, ... in ascending order from 0, the lanes meet in the butterfly (lane 0's value),
// then router_score. Here a block of 4 waves covers 16 rows x 16 experts (a wave 8 x 8); x and the
// weights are staged in LDS 256 elements at a time and every lane keeps the 64 chains of its tile, so
// each (row, expert) is formed by exactly the decode chain and tree.
constexpr unsigned prouter_rows = 16U, prouter_experts = 16U, prouter_chunk = 256U, prouter_threads = 128U, prouter_group = 4U;
__global__ __launch_bounds__(prouter_threads) void prefill_router_scores_kernel(RouterView view) {
    __shared__ alignas(16) float xs[prouter_rows][prouter_chunk];
    __shared__ alignas(16) float ws[prouter_experts][prouter_chunk];
    __shared__ float logits[prouter_rows][prouter_experts];
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    const unsigned row0 = blockIdx.x * prouter_rows, expert0 = blockIdx.y * prouter_experts;
    const unsigned wr = (wave & 1U) * 8U, we = (wave >> 1U) * 8U;
    float acc[8][8];
#pragma unroll
    for (unsigned r = 0; r < 8U; ++r)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) acc[r][e] = 0.0F;
    for (unsigned c0 = 0; c0 < view.hidden; c0 += prouter_chunk) {
        const unsigned n = min(prouter_chunk, view.hidden - c0);
        if (c0) __syncthreads();
        if (view.wide_stage) {
            // the same values staged with 16-byte loads, prouter_group of them in flight per array before the LDS stores
            // (the element loop above waited for every load in turn: ~0.8 ms per call at any row count)
            constexpr unsigned quads = prouter_chunk / 4U, per_thread = prouter_rows * quads / prouter_threads;
            for (unsigned t0 = 0; t0 < per_thread; t0 += prouter_group) {
                float4 xv4[prouter_group], wv4[prouter_group];
#pragma unroll
                for (unsigned t = 0; t < prouter_group; ++t) {
                    const unsigned i = threadIdx.x + (t0 + t) * prouter_threads, r = i / quads, k = (i % quads) * 4U;
                    const unsigned row = row0 + r, expert = expert0 + r;
                    xv4[t] = row < view.rows && k < n ? *reinterpret_cast<const float4*>(view.x + static_cast<unsigned long long>(row) * view.hidden + c0 + k)
                                                      : float4{0.0F, 0.0F, 0.0F, 0.0F};
                    wv4[t] = expert < view.experts && k < n
                                 ? *reinterpret_cast<const float4*>(view.weight + static_cast<unsigned long long>(expert) * view.hidden + c0 + k)
                                 : float4{0.0F, 0.0F, 0.0F, 0.0F};
                }
#pragma unroll
                for (unsigned t = 0; t < prouter_group; ++t) {
                    const unsigned i = threadIdx.x + (t0 + t) * prouter_threads, r = i / quads, k = (i % quads) * 4U;
                    *reinterpret_cast<float4*>(&xs[r][k]) = xv4[t];
                    *reinterpret_cast<float4*>(&ws[r][k]) = wv4[t];
                }
            }
        } else {
            for (unsigned i = threadIdx.x; i < prouter_rows * prouter_chunk; i += prouter_threads) {
                const unsigned r = i / prouter_chunk, k = i % prouter_chunk;
                const unsigned row = row0 + r, expert = expert0 + r;
                xs[r][k] = row < view.rows && k < n ? view.x[static_cast<unsigned long long>(row) * view.hidden + c0 + k] : 0.0F;
                ws[r][k] = expert < view.experts && k < n ? view.weight[static_cast<unsigned long long>(expert) * view.hidden + c0 + k] : 0.0F;
            }
        }
        __syncthreads();
        for (unsigned j = 0; j < prouter_chunk / 32U; ++j) {
            const unsigned k = lane + 32U * j;
            if (k >= n) break; // n is a multiple of 32 whenever hidden is; the chain skips nothing
            float xv[8], wv[8];
#pragma unroll
            for (unsigned r = 0; r < 8U; ++r) xv[r] = xs[wr + r][k];
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) wv[e] = ws[we + e][k];
#pragma unroll
            for (unsigned r = 0; r < 8U; ++r)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) acc[r][e] = acc[r][e] + wv[e] * xv[r];
        }
    }
#pragma unroll
    for (unsigned r = 0; r < 8U; ++r)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            float v = acc[r][e];
#pragma unroll
            for (unsigned d = 16U; d; d /= 2U) v = v + __shfl_xor(v, static_cast<int>(d), 32);
            if (lane == 0U) logits[wr + r][we + e] = v;
        }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < prouter_rows * prouter_experts; i += prouter_threads) {
        const unsigned r = i / prouter_experts, e = i % prouter_experts, row = row0 + r, expert = expert0 + e;
        if (row < view.rows && expert < view.experts)
            view.scores[static_cast<unsigned long long>(row) * view.experts + expert] = router_score(logits[r][e]);
    }
}
// v2 of the same scores (STRIX_DS41_PREFILL_ROUTER_V2, default on): a block of 8 waves covers 32 rows x 32 experts (a
// wave 8 rows x 16 experts), x and the weights are staged 128 elements at a time and the next chunk's 16-byte loads are
// issued before the current chunk's products, so the loads overlap the FMAs instead of waiting behind two barriers.
// Lane l still owns the elements l, l + 32, ... of every (row, expert) of its tile, in ascending order from 0, and the
// lanes meet in the same butterfly: each score is formed by exactly the decode chain and tree, so every bit is equal.
// Needs hidden % 4 == 0 and 16-byte aligned x and weight (otherwise the kernel above runs).
constexpr unsigned prouter2_rows = 32U, prouter2_experts = 32U, prouter2_chunk = 128U, prouter2_threads = 256U;
constexpr unsigned prouter2_quads = prouter2_rows * prouter2_chunk / 4U / prouter2_threads; // 16-byte loads per thread per array
__global__ __launch_bounds__(prouter2_threads) void prefill_router_scores_v2_kernel(RouterView view) {
    __shared__ alignas(16) float xs[prouter2_rows][prouter2_chunk];
    __shared__ alignas(16) float ws[prouter2_experts][prouter2_chunk];
    __shared__ float logits[prouter2_rows][prouter2_experts];
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    const unsigned row0 = blockIdx.y * prouter2_rows, expert0 = blockIdx.x * prouter2_experts;
    const unsigned wr = (wave & 3U) * 8U, we = (wave >> 2U) * 16U;
    float acc[8][16];
#pragma unroll
    for (unsigned r = 0; r < 8U; ++r)
#pragma unroll
        for (unsigned e = 0; e < 16U; ++e) acc[r][e] = 0.0F;
    float4 xv4[prouter2_quads], wv4[prouter2_quads];
    const auto load = [&](unsigned c0) {
        const unsigned n = (view.hidden - c0 < prouter2_chunk ? view.hidden - c0 : prouter2_chunk);
#pragma unroll
        for (unsigned t = 0; t < prouter2_quads; ++t) {
            const unsigned i = threadIdx.x + t * prouter2_threads, r = i / (prouter2_chunk / 4U), k = (i % (prouter2_chunk / 4U)) * 4U;
            const unsigned row = row0 + r, expert = expert0 + r;
            xv4[t] = row < view.rows && k < n ? *reinterpret_cast<const float4*>(view.x + static_cast<unsigned long long>(row) * view.hidden + c0 + k)
                                              : float4{0.0F, 0.0F, 0.0F, 0.0F};
            wv4[t] = expert < view.experts && k < n
                         ? *reinterpret_cast<const float4*>(view.weight + static_cast<unsigned long long>(expert) * view.hidden + c0 + k)
                         : float4{0.0F, 0.0F, 0.0F, 0.0F};
        }
    };
    load(0U);
    for (unsigned c0 = 0; c0 < view.hidden; c0 += prouter2_chunk) {
        const unsigned n = (view.hidden - c0 < prouter2_chunk ? view.hidden - c0 : prouter2_chunk);
        if (c0) __syncthreads();
#pragma unroll
        for (unsigned t = 0; t < prouter2_quads; ++t) {
            const unsigned i = threadIdx.x + t * prouter2_threads, r = i / (prouter2_chunk / 4U), k = (i % (prouter2_chunk / 4U)) * 4U;
            *reinterpret_cast<float4*>(&xs[r][k]) = xv4[t];
            *reinterpret_cast<float4*>(&ws[r][k]) = wv4[t];
        }
        __syncthreads();
        if (c0 + prouter2_chunk < view.hidden) load(c0 + prouter2_chunk);
        for (unsigned j = 0; j < prouter2_chunk / 32U; ++j) {
            const unsigned k = lane + 32U * j;
            if (k >= n) break; // n is a multiple of 32 (hidden is); the chain skips nothing
            float xv[8], wv[16];
#pragma unroll
            for (unsigned r = 0; r < 8U; ++r) xv[r] = xs[wr + r][k];
#pragma unroll
            for (unsigned e = 0; e < 16U; ++e) wv[e] = ws[we + e][k];
#pragma unroll
            for (unsigned r = 0; r < 8U; ++r)
#pragma unroll
                for (unsigned e = 0; e < 16U; ++e) acc[r][e] = acc[r][e] + wv[e] * xv[r];
        }
    }
#pragma unroll
    for (unsigned r = 0; r < 8U; ++r)
#pragma unroll
        for (unsigned e = 0; e < 16U; ++e) {
            float v = acc[r][e];
#pragma unroll
            for (unsigned d = 16U; d; d /= 2U) v = v + __shfl_xor(v, static_cast<int>(d), 32);
            if (lane == 0U) logits[wr + r][we + e] = v;
        }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < prouter2_rows * prouter2_experts; i += prouter2_threads) {
        const unsigned r = i / prouter2_experts, e = i % prouter2_experts, row = row0 + r, expert = expert0 + e;
        if (row < view.rows && expert < view.experts)
            view.scores[static_cast<unsigned long long>(row) * view.experts + expert] = router_score(logits[r][e]);
    }
}

// Top-k of one row per block, as the decode router's selection: repeated arg-max of score + bias
// (larger value, then lower id), weights from the unbiased scores in selection order.
__global__ __launch_bounds__(256) void prefill_router_select_kernel(RouterView view) {
    __shared__ float biased[1024];
    __shared__ float best_value[256];
    __shared__ unsigned best_index[256];
    __shared__ unsigned chosen[16];
    const unsigned r = blockIdx.x, t = threadIdx.x;
    const float* scores = view.scores + static_cast<unsigned long long>(r) * view.experts;
    const float* bias = view.image_rows && view.image_rows[r] ? view.bias_image : view.bias;
    for (unsigned e = t; e < view.experts; e += 256U) biased[e] = scores[e] + bias[e];
    __syncthreads();
    for (unsigned k = 0; k < view.top_k; ++k) {
        float v = 0.0F;
        unsigned index = view.experts;
        for (unsigned e = t; e < view.experts; e += 256U) {
            bool taken = false;
            for (unsigned j = 0; j < k; ++j) taken = taken || chosen[j] == e;
            if (taken) continue;
            if (index == view.experts || biased[e] > v) { v = biased[e]; index = e; }
        }
        best_value[t] = v; best_index[t] = index;
        __syncthreads();
        for (unsigned stride = 128U; stride; stride /= 2U) {
            if (t < stride) {
                const float ov = best_value[t + stride];
                const unsigned oi = best_index[t + stride];
                const bool better = oi != view.experts &&
                    (best_index[t] == view.experts || ov > best_value[t] || (ov == best_value[t] && oi < best_index[t]));
                if (better) { best_value[t] = ov; best_index[t] = oi; }
            }
            __syncthreads();
        }
        if (t == 0U) chosen[k] = best_index[0];
        __syncthreads();
    }
    if (t == 0U) {
        float sum = 0.0F;
        for (unsigned k = 0; k < view.top_k; ++k) sum = sum + scores[chosen[k]];
        const float denominator = sum + 1e-20F;
        for (unsigned k = 0; k < view.top_k; ++k) {
            const float s = scores[chosen[k]];
            view.ids[r * view.top_k + k] = chosen[k];
            view.weights[r * view.top_k + k] = (view.top_k > 1U ? s / denominator : s) * view.route_scale;
        }
    }
}

// ---- Routes on the device ----

// Experts [first, end) only (a rank's slice); expert indices below are relative to first.
__global__ __launch_bounds__(256) void route_count_kernel(const unsigned* ids, unsigned slots, unsigned first, unsigned end,
                                                          unsigned* counts) {
    const unsigned s = blockIdx.x * 256U + threadIdx.x;
    if (s >= slots) return;
    const unsigned id = ids[s];
    if (id >= first && id < end) atomicAdd(counts + (id - first), 1U);
}
// One block, histogram split (PrefillRoutesBuffers::sparse_max > 0): sparse experts (1..sparse_max entries) first, in
// ascending expert, with batches of 8 in the sparse arrays; then the dense experts with batches of batch_rows.
__global__ __launch_bounds__(1024) void route_scan_split_kernel(const unsigned* counts, unsigned experts, unsigned batch_rows,
                                                                PrefillRoutesBuffers out) {
    __shared__ unsigned scan[4][1024];
    const unsigned e = threadIdx.x;
    const unsigned c = e < experts ? counts[e] : 0U;
    const bool sparse = c > 0U && c <= out.sparse_max;
    const unsigned sc = sparse ? c : 0U, dc = sparse ? 0U : c, snb = (sc + 7U) / 8U, dnb = (dc + batch_rows - 1U) / batch_rows;
    scan[0][e] = sc; scan[1][e] = dc; scan[2][e] = snb; scan[3][e] = dnb;
    __syncthreads();
    for (unsigned stride = 1U; stride < 1024U; stride *= 2U) {
        unsigned add[4];
        for (unsigned a = 0; a < 4U; ++a) add[a] = e >= stride ? scan[a][e - stride] : 0U;
        __syncthreads();
        for (unsigned a = 0; a < 4U; ++a) scan[a][e] += add[a];
        __syncthreads();
    }
    const unsigned sparse_entries = scan[0][1023], dense_entries = scan[1][1023];
    if (e < experts) {
        if (sparse) {
            const unsigned begin = scan[0][e] - sc, first_batch = scan[2][e] - snb;
            out.begins[e] = begin;
            for (unsigned b = 0; b < snb; ++b) {
                out.sparse_expert[first_batch + b] = e;
                out.sparse_begin[first_batch + b] = begin + b * 8U;
            }
        } else {
            const unsigned begin = sparse_entries + scan[1][e] - dc, first_batch = scan[3][e] - dnb;
            out.begins[e] = begin;
            for (unsigned b = 0; b < dnb; ++b) {
                out.batch_expert[first_batch + b] = e;
                out.batch_begin[first_batch + b] = begin + b * batch_rows;
                out.batch_count[first_batch + b] = min(batch_rows, dc - b * batch_rows);
            }
        }
    }
    if (e == 0U) {
        out.begins[experts] = sparse_entries + dense_entries;
        *out.sparse_total = scan[2][1023];
        out.sparse_begin[scan[2][1023]] = sparse_entries;
        *out.batch_total = scan[3][1023];
        out.batch_begin[scan[3][1023]] = sparse_entries + dense_entries;
    }
}
// One block: begins (exclusive scan of the counts) and the batch descriptors in ascending expert.
__global__ __launch_bounds__(1024) void route_scan_kernel(const unsigned* counts, unsigned experts, unsigned batch_rows,
                                                          PrefillRoutesBuffers out) {
    __shared__ unsigned entries[1024], batches[1024];
    const unsigned e = threadIdx.x;
    const unsigned c = e < experts ? counts[e] : 0U, nb = (c + batch_rows - 1U) / batch_rows;
    entries[e] = c; batches[e] = nb;
    __syncthreads();
    for (unsigned stride = 1U; stride < 1024U; stride *= 2U) {
        const unsigned add_e = e >= stride ? entries[e - stride] : 0U, add_b = e >= stride ? batches[e - stride] : 0U;
        __syncthreads();
        entries[e] += add_e; batches[e] += add_b;
        __syncthreads();
    }
    if (e < experts) {
        const unsigned begin = entries[e] - c, first_batch = batches[e] - nb;
        out.begins[e] = begin;
        for (unsigned b = 0; b < nb; ++b) {
            out.batch_expert[first_batch + b] = e;
            out.batch_begin[first_batch + b] = begin + b * batch_rows;
            out.batch_count[first_batch + b] = min(batch_rows, c - b * batch_rows);
        }
    }
    if (e == 0U) {
        out.begins[experts] = entries[1023];
        *out.batch_total = batches[1023];
        out.batch_begin[batches[1023]] = entries[1023]; // the batch_begin array has max_batches + 1 entries
    }
}
// One block per expert: its slots in ascending order (ordered compaction over the id array).
__global__ __launch_bounds__(256) void route_fill_kernel(const unsigned* ids, unsigned slots, unsigned top_k, unsigned first,
                                                         PrefillRoutesBuffers out) {
    __shared__ unsigned wave_counts[8];
    const unsigned local = blockIdx.x, expert = first + local, wave = threadIdx.x / 32U, lane = threadIdx.x % 32U;
    unsigned at = out.begins[local];
    for (unsigned base = 0; base < slots; base += 256U) {
        const unsigned s = base + threadIdx.x;
        const bool match = s < slots && ids[s] == expert;
        const unsigned mask = static_cast<unsigned>(__ballot(match));
        if (lane == 0U) wave_counts[wave] = static_cast<unsigned>(__builtin_popcount(mask));
        __syncthreads();
        unsigned before = 0U, total = 0U;
        for (unsigned w = 0; w < 8U; ++w) { before += w < wave ? wave_counts[w] : 0U; total += wave_counts[w]; }
        if (match) {
            const unsigned index = at + before + static_cast<unsigned>(__builtin_popcount(mask & ((1U << lane) - 1U)));
            out.entry_rows[index] = s / top_k;
            out.entry_slots[index] = s;
        }
        at += total;
        __syncthreads();
    }
}
// Per row: slots in ascending expert id and how many ids are below split.
__global__ __launch_bounds__(256) void route_order_kernel(const unsigned* ids, unsigned rows, unsigned top_k, unsigned split,
                                                          PrefillRoutesBuffers out) {
    const unsigned row = blockIdx.x * 256U + threadIdx.x;
    if (row >= rows) return;
    unsigned expert[16], slot[16], low = 0U;
    for (unsigned k = 0; k < top_k; ++k) {
        const unsigned s = row * top_k + k, id = ids[s];
        unsigned at = k;
        while (at > 0U && expert[at - 1U] > id) { expert[at] = expert[at - 1U]; slot[at] = slot[at - 1U]; --at; }
        expert[at] = id; slot[at] = s;
        low += id < split ? 1U : 0U;
    }
    for (unsigned k = 0; k < top_k; ++k) out.order[row * top_k + k] = slot[k];
    out.low_counts[row] = low;
}

StageLayout stage_layout(unsigned width, unsigned batch_rows) {
    const unsigned groups = width / 32U, j = (groups + 31U) / 32U;
    return {batch_rows, j, stage_classes * j};
}
// STRIX_DS41_PREFILL_STAGE=1 keeps 16-row batches on the stage kernels (comparison only).
bool stage_forced() {
    static const bool forced = [] {
        const char* v = std::getenv("STRIX_DS41_PREFILL_STAGE");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return forced;
}
// Prefill kernel forms (moe_kernel_tuning().prefill, STRIX_DS41_MOE_PREFILL): 0 = the original decoders; 1 = the lean
// decoders (same integers, see decode_lean); 2 = lean with the IQ2_XXS sign table (decode_signs); 3 = the lean decoders
// with the activations double-buffered; 4 = the lean decoders with the merged stage (merged_stage); 6 = 1 with LDS-only
// chunk-loop barriers; 7 (default, -1) = 6 with a batched activation stage in gate/up (down runs form 6). Measured: the
// joint MoE layer -5.7 / -3.2 / -5.1 % at T 600 / 2048 / 8192, gate/up -8.8 %; same bits. =0 keeps the original form.
int prefill_form() {
    const int f = moe_kernel_tuning().prefill;
    return f < 0 ? 7 : f;
}
// STRIX_DS41_PREFILL_MOE_V2 (read per call; default on, =0 off): the split kernels v2 (split_gate_up_v2_kernel,
// split_down_v2_kernel) for IQ2_XXS gate/up and Q2_K down under the default form 7; the same bits as forms 7 / 6.
// STRIX_DS41_PREFILL_MOE_V2_MASK (expert override, attribution): a mask of v2_decode (2), v2_fma (4), v2_part (8), default
// v2_fma | v2_part (v2_decode measured slower than the lean decoder: opt-in only; v2_part is always added). STRIX_DS41_PREFILL_MOE_BAND: blocks in bands of N batches (0 or 1: batch-fastest order as before), default 4.
unsigned prefill_v2() {
    if (prefill_form() != 7) return 0U;
    const char* v = std::getenv("STRIX_DS41_PREFILL_MOE_V2");
    if (v && std::strcmp(v, "0") == 0) return 0U;
    const char* m = std::getenv("STRIX_DS41_PREFILL_MOE_V2_MASK");
    unsigned mask = v2_on | (m && *m ? static_cast<unsigned>(std::strtoul(m, nullptr, 0)) & (v2_decode | v2_fma | v2_part | v2_pin | v2_signs | v2_signs16 | v2_pipe) : v2_fma | v2_part);
    // STRIX_DS41_PREFILL_MOE_V2_SIGNS (=1 / =16, read per call; opt-in): v2_signs / v2_signs16, the IQ2_XXS gate/up sign
    // masks from an LDS table (8- / 16-byte entries). Same bits; measured slower (8K gate/up +7 % / +13 %): -19 % VALU but
    // 5x the LDS wait cycles (random-index table reads).
    if (!(m && *m)) {
        const char* sg = std::getenv("STRIX_DS41_PREFILL_MOE_V2_SIGNS");
        if (sg && std::strcmp(sg, "1") == 0) mask |= v2_signs;
        if (sg && std::strcmp(sg, "16") == 0) mask |= v2_signs16;
        // STRIX_DS41_PREFILL_MOE_V2_PIPE=1 (opt-in): v2_pipe, the IQ2_XXS decode's LDS reads one group ahead (same bits;
        // measured +10 % at 8K)
        const char* pp = std::getenv("STRIX_DS41_PREFILL_MOE_V2_PIPE");
        if (pp && std::strcmp(pp, "1") == 0) mask |= v2_pipe;
    }
    // STRIX_DS41_PREFILL_MOE_V2_PIN (=1 on, read per call; opt-in): v2_pin, the WMMAs on bank-pinned registers. Measured
    // (rank bench, same bits): routed gate/up and down neutral (+-1 %), the Q8_0 shared expert 3-14 % slower.
    if (!(m && *m)) {
        const char* pin = std::getenv("STRIX_DS41_PREFILL_MOE_V2_PIN");
        if (pin && std::strcmp(pin, "1") == 0) mask |= v2_pin;
    }
    // Only implemented combinations: the sign-table and pipelined decoders exist in the pinned form (they imply v2_pin),
    // and a partial last chunk always loads only its blocks (v2_part: the full-chunk load would read past the Q2_K
    // tensor's last row at width 2304).
    if (mask & (v2_signs | v2_signs16 | v2_pipe)) mask |= v2_pin;
    return mask | v2_part;
}
// STRIX_DS41_PREFILL_MOE_V2_Q8 (read per call; default on, =0 off; needs v2): the Q8_0 split kernels (the shared expert's
// gate/up and down) on split_gate_up_v2_kernel / split_down_v2_kernel with direct per-lane weight reads; the same bits.
bool prefill_v2_q8() {
    const char* v = std::getenv("STRIX_DS41_PREFILL_MOE_V2_Q8");
    return !(v && std::strcmp(v, "0") == 0);
}
unsigned prefill_band() {
    const char* v = std::getenv("STRIX_DS41_PREFILL_MOE_BAND");
    return v && *v ? static_cast<unsigned>(std::strtoul(v, nullptr, 10)) : 4U;
}
// STRIX_DS41_PREFILL_MOE_R2=1 (opt-in, read per call; needs v2): 32-row batches, IQ2_XXS gate/up on r2_gate_up_kernel and
// Q2_K down on split_down_v2_kernel in 16-row halves; the same bits.
bool prefill_r2() {
    const char* v = std::getenv("STRIX_DS41_PREFILL_MOE_R2");
    return v && std::strcmp(v, "1") == 0 && (prefill_v2() & v2_on) != 0U && !prefill_tuning::tiles() && !stage_forced();
}
template <WeightFormat Format, unsigned L>
void launch_split_gate_up_v2(const PrefillGateUpView& view, hipStream_t stream) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(split_gate_up_v2_kernel<Format, L>), dim3(view.routes.max_batches, view.inter / 32U), dim3(split_threads),
                       (v2_lds<Format, L>), stream, view, prefill_band());
}
template <WeightFormat Format, unsigned L>
void launch_split_down_v2(const PrefillDownView& view, hipStream_t stream) {
    const unsigned halves = view.routes.batch_rows == 32U ? 2U : 1U; // 32-row batches (R2) run as two 16-row blocks
    hipLaunchKernelGGL(HIP_KERNEL_NAME(split_down_v2_kernel<Format, L>), dim3(halves * view.routes.max_batches, view.outputs / 64U), dim3(split_threads),
                       (v2_lds<Format, L>), stream, view, prefill_band());
}
template <WeightFormat Format, int Form>
void launch_split_gate_up_form(const PrefillGateUpView& view, hipStream_t stream) {
    constexpr unsigned lds = split_lds_for<Format, stage_buffers_of<Form>>();
    hipLaunchKernelGGL(HIP_KERNEL_NAME(split_gate_up_kernel<Format, Form>), dim3(view.routes.max_batches, view.inter / 32U), dim3(split_threads), lds, stream, view);
}
template <WeightFormat Format>
void launch_split_gate_up(const PrefillGateUpView& view, hipStream_t stream) {
    if constexpr (Format == WeightFormat::iq2_xxs) {
        if (const unsigned m = prefill_v2(); (m & v2_on) && view.hidden % (split_chunk_groups * 32U) == 0U) {
            if (m & v2_decode) launch_split_gate_up_v2<Format, v2_on | v2_decode>(view, stream);
            else if ((m & (v2_pin | v2_pipe)) == (v2_pin | v2_pipe)) launch_split_gate_up_v2<Format, v2_on | v2_pin | v2_pipe>(view, stream);
            else if ((m & (v2_pin | v2_signs16)) == (v2_pin | v2_signs16)) launch_split_gate_up_v2<Format, v2_on | v2_pin | v2_signs16>(view, stream);
            else if ((m & (v2_pin | v2_signs)) == (v2_pin | v2_signs)) launch_split_gate_up_v2<Format, v2_on | v2_pin | v2_signs>(view, stream);
            else if (m & v2_pin) launch_split_gate_up_v2<Format, v2_on | v2_pin>(view, stream);
            else launch_split_gate_up_v2<Format, v2_on>(view, stream);
            return;
        }
    } else if constexpr (Format == WeightFormat::q8_0) {
        if (const unsigned m = prefill_v2(); (m & v2_on) && prefill_v2_q8()) {
            if (m & v2_pin) launch_split_gate_up_v2<Format, v2_on | v2_pin>(view, stream);
            else launch_split_gate_up_v2<Format, v2_on>(view, stream);
            return;
        }
    }
    switch (prefill_form()) {
    case 0: launch_split_gate_up_form<Format, 0>(view, stream); break;
    case 1: launch_split_gate_up_form<Format, 1>(view, stream); break;
    case 2: launch_split_gate_up_form<Format, 2>(view, stream); break;
    case 3: launch_split_gate_up_form<Format, 3>(view, stream); break;
    case 6: launch_split_gate_up_form<Format, 6>(view, stream); break;
    case 7: launch_split_gate_up_form<Format, 7>(view, stream); break;
    default: launch_split_gate_up_form<Format, 4>(view, stream); break;
    }
}
template <WeightFormat Format, int Form>
void launch_split_down_form(const PrefillDownView& view, hipStream_t stream) {
    constexpr unsigned lds = split_lds_for<Format, stage_buffers_of<Form>>();
    hipLaunchKernelGGL(HIP_KERNEL_NAME(split_down_kernel<Format, Form>), dim3(view.routes.max_batches, view.outputs / 64U), dim3(split_threads), lds, stream, view);
}
template <WeightFormat Format>
void launch_split_down(const PrefillDownView& view, hipStream_t stream) {
    if constexpr (Format == WeightFormat::q2_k) {
        if (const unsigned m = prefill_v2(); m & v2_on) {
            switch (m & (v2_fma | v2_pin)) {  // prefill_v2() always sets v2_part
            case v2_fma | v2_pin: launch_split_down_v2<Format, v2_on | v2_fma | v2_part | v2_pin>(view, stream); break;
            case v2_fma: launch_split_down_v2<Format, v2_on | v2_fma | v2_part>(view, stream); break;
            case v2_pin: launch_split_down_v2<Format, v2_on | v2_part | v2_pin>(view, stream); break;
            default: launch_split_down_v2<Format, v2_on | v2_part>(view, stream); break;
            }
            return;
        }
    } else if constexpr (Format == WeightFormat::q8_0) {
        if (const unsigned m = prefill_v2(); (m & v2_on) && prefill_v2_q8()) {
            if (m & v2_pin) launch_split_down_v2<Format, v2_on | v2_pin>(view, stream);
            else launch_split_down_v2<Format, v2_on>(view, stream);
            return;
        }
    }
    // the down formats (Q2_K, MXFP4, Q8_0) have no IQ2_XXS sign table: forms 1 and 2 are the same kernel
    switch (prefill_form()) {
    case 0: launch_split_down_form<Format, 0>(view, stream); break;
    case 1: case 2: launch_split_down_form<Format, 1>(view, stream); break;
    case 3: launch_split_down_form<Format, 3>(view, stream); break;
    case 6: launch_split_down_form<Format, 6>(view, stream); break;
    case 7: launch_split_down_form<Format, 6>(view, stream); break;  // the batched stage spills in the down kernel
    default: launch_split_down_form<Format, 4>(view, stream); break;
    }
}
void require_routes(const PrefillRoutesView& r) {
    if (!r.begins || !r.entry_rows || !r.entry_slots || !r.batch_expert || !r.batch_begin || !r.batch_count || !r.batch_total ||
        r.max_batches == 0U || (r.batch_rows != 16U && r.batch_rows != 32U))
        throw std::invalid_argument("ds41 prefill routes view is inconsistent");
}

template <WeightFormat Format, unsigned MT, unsigned J>
void launch_gate_up_j(const PrefillGateUpView& view, const StageLayout& layout, hipStream_t stream) {
    const unsigned bytes = std::max(layout.bytes(false), 2U * 32U * 64U * 4U);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(prefill_gate_up_kernel<Format, MT, J>), dim3(view.routes.max_batches, view.inter / 64U),
                       dim3(32U * prefill_waves), bytes, stream, view, layout);
}
template <WeightFormat Format, unsigned MT>
void launch_gate_up(const PrefillGateUpView& view, hipStream_t stream) {
    const auto layout = stage_layout(view.hidden, 16U * MT);
    switch (layout.groups_per_class) {
    case 1: launch_gate_up_j<Format, MT, 1>(view, layout, stream); break;
    case 2: launch_gate_up_j<Format, MT, 2>(view, layout, stream); break;
    case 3: launch_gate_up_j<Format, MT, 3>(view, layout, stream); break;
    case 4: launch_gate_up_j<Format, MT, 4>(view, layout, stream); break;
    default: launch_gate_up_j<Format, MT, 5>(view, layout, stream); break;
    }
}
template <WeightFormat Format, unsigned MT, unsigned J>
void launch_down_j(const PrefillDownView& view, const StageLayout& layout, hipStream_t stream) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(prefill_down_kernel<Format, MT, J>), dim3(view.routes.max_batches, view.outputs / 128U),
                       dim3(32U * prefill_waves), layout.bytes(Format == WeightFormat::q2_k), stream, view, layout);
}
template <WeightFormat Format, unsigned MT>
void launch_down(const PrefillDownView& view, hipStream_t stream) {
    const auto layout = stage_layout(view.width, 16U * MT);
    switch (layout.groups_per_class) {
    case 1: launch_down_j<Format, MT, 1>(view, layout, stream); break;
    case 2: launch_down_j<Format, MT, 2>(view, layout, stream); break;
    case 3: launch_down_j<Format, MT, 3>(view, layout, stream); break;
    case 4: launch_down_j<Format, MT, 4>(view, layout, stream); break;
    default: launch_down_j<Format, MT, 5>(view, layout, stream); break;
    }
}

} // namespace

void enqueue_build_prefill_routes(const std::uint32_t* ids, std::uint32_t rows, std::uint32_t top_k, std::uint32_t experts,
                                  std::uint32_t split, std::uint32_t batch_rows, const PrefillRoutesBuffers& out, void* stream,
                                  std::uint32_t first, std::uint32_t end) {
    if (end == 0U) end = experts;
    if (first >= end || end > experts) throw std::invalid_argument("ds41 prefill route build: expert range");
    if (!ids || rows == 0U || top_k == 0U || top_k > 16U || experts == 0U || experts > 1024U ||
        (batch_rows != 8U && batch_rows != 16U && batch_rows != 32U) || !out.counts || !out.begins || !out.entry_rows || !out.entry_slots ||
        !out.batch_expert || !out.batch_begin || !out.batch_count || !out.batch_total || !out.order || !out.low_counts)
        throw std::invalid_argument("ds41 prefill route build is inconsistent");
    const auto queue = static_cast<hipStream_t>(stream);
    const unsigned slots = rows * top_k;
    const unsigned owned = end - first;
    check(hipMemsetAsync(out.counts, 0, owned * sizeof(std::uint32_t), queue), "route counts reset");
    hipLaunchKernelGGL(route_count_kernel, dim3((slots + 255U) / 256U), dim3(256), 0, queue, ids, slots, first, end, out.counts);
    if (out.sparse_max > 0U) {
        if (!out.sparse_expert || !out.sparse_begin || !out.sparse_total) throw std::invalid_argument("ds41 prefill route build: sparse arrays");
        hipLaunchKernelGGL(route_scan_split_kernel, dim3(1), dim3(1024), 0, queue, out.counts, owned, batch_rows, out);
    } else {
        hipLaunchKernelGGL(route_scan_kernel, dim3(1), dim3(1024), 0, queue, out.counts, owned, batch_rows, out);
    }
    hipLaunchKernelGGL(route_fill_kernel, dim3(owned), dim3(256), 0, queue, ids, slots, top_k, first, out);
    hipLaunchKernelGGL(route_order_kernel, dim3((rows + 255U) / 256U), dim3(256), 0, queue, ids, rows, top_k, split, out);
    check(hipGetLastError(), "route build launch");
}

PrefillRoutesView prefill_routes_view(const PrefillRoutesBuffers& b, std::uint32_t rows, std::uint32_t top_k, std::uint32_t experts,
                                      std::uint32_t batch_rows) {
    PrefillRoutesView v;
    v.begins = b.begins; v.entry_rows = b.entry_rows; v.entry_slots = b.entry_slots; v.batch_expert = b.batch_expert;
    v.batch_begin = b.batch_begin; v.batch_count = b.batch_count; v.batch_total = b.batch_total;
    v.max_batches = prefill_max_batches(rows, top_k, experts, batch_rows);
    v.batch_rows = batch_rows;
    return v;
}

void enqueue_prefill_router(const RouterView& view, void* stream) {
    if (!view.x || !view.weight || !view.bias || !view.scores || !view.ids || !view.weights || view.rows == 0U || view.hidden == 0U ||
        view.hidden % 32U != 0U || view.experts == 0U || view.experts > 1024U || view.top_k == 0U || view.top_k > 16U ||
        view.top_k > view.experts || (view.image_rows && !view.bias_image))
        throw std::invalid_argument("ds41 prefill router view is inconsistent");
    const auto queue = static_cast<hipStream_t>(stream);
    static const bool wide = [] { const char* e = std::getenv("STRIX_DS41_PREFILL_ROUTER_WIDE"); return !(e && e[0] == '0'); }();
    RouterView v = view;
    v.wide_stage = wide && view.wide_stage && view.hidden % 4U == 0U && reinterpret_cast<std::uintptr_t>(view.x) % 16U == 0U &&
                   reinterpret_cast<std::uintptr_t>(view.weight) % 16U == 0U;
    // STRIX_DS41_PREFILL_ROUTER_V2 (read per call; default on, =0 off): the v2 scores kernel, the same bits
    const char* v2_env = std::getenv("STRIX_DS41_PREFILL_ROUTER_V2");
    const bool v2 = !(v2_env && v2_env[0] == '0') && v.wide_stage; // 16-byte staging admitted (alignment, hidden % 4)
    if (v2) {
        // experts fastest: the blocks of one row tile run together and share its activations in the L2
        hipLaunchKernelGGL(prefill_router_scores_v2_kernel, dim3((view.experts + prouter2_experts - 1U) / prouter2_experts,
                           (view.rows + prouter2_rows - 1U) / prouter2_rows), dim3(prouter2_threads), 0, queue, v);
    } else {
        hipLaunchKernelGGL(prefill_router_scores_kernel, dim3((view.rows + prouter_rows - 1U) / prouter_rows,
                           (view.experts + prouter_experts - 1U) / prouter_experts), dim3(prouter_threads), 0, queue, v);
    }
    hipLaunchKernelGGL(prefill_router_select_kernel, dim3(view.rows), dim3(256), 0, queue, view);
    check(hipGetLastError(), "prefill router launch");
}

// FAST lever ds41-moe-relaxed-v1: STRIX_DS41_FAST_MOE=1 (default off) or set_fast_moe(1 / 0; -1 = the environment)
int fast_moe_mode = -1;
void set_fast_moe(int mode) { fast_moe_mode = mode; }
// STRIX_DS41_FAST_MOE_ORDER (read per call; default on, =0 off): the relaxed kernels walk output blocks fastest (same bits;
// calls of more than 65535 batches keep the batch-fastest order, the grid y limit)
static bool fast_moe_order() {
    const char* e = std::getenv("STRIX_DS41_FAST_MOE_ORDER");
    return !(e && e[0] == '0');
}
bool fast_moe() {
    static const bool env = [] { const char* e = std::getenv("STRIX_DS41_FAST_MOE"); return e != nullptr && std::strcmp(e, "1") == 0; }();
    return fast_moe_mode < 0 ? env : fast_moe_mode != 0;
}
void enqueue_prefill_gate_up(WeightFormat format, const PrefillGateUpView& view, void* stream) {
    require_routes(view.routes);
    const auto elements = block_elements(format);
    if (format == WeightFormat::q2_k || !view.gate || !view.up || !view.codes || !view.deltas || !view.act_codes ||
        !view.act_deltas || !view.act_sums || (!view.gate_out) != (!view.up_out) || view.hidden == 0U ||
        view.hidden % elements != 0U || view.hidden % 32U != 0U || view.hidden > 32U * 32U * max_groups_per_class ||
        view.inter == 0U || view.inter % 64U != 0U || view.row_bytes != view.hidden / elements * block_bytes(format) ||
        reinterpret_cast<std::uintptr_t>(view.codes) % 16U != 0U)
        throw std::invalid_argument("ds41 prefill gate/up view is inconsistent");
    const auto queue = static_cast<hipStream_t>(stream);
    if (fast_moe() && format == WeightFormat::iq2_xxs && view.hidden % 256U == 0U && view.inter % 128U == 0U &&
        view.routes.batch_rows <= relaxed_rows) {
        if (fast_moe_order() && view.routes.max_batches <= 65535U)  // grid y limit
            hipLaunchKernelGGL(HIP_KERNEL_NAME(relaxed_gate_up_kernel<WeightFormat::iq2_xxs, true>), dim3(view.inter / 128U, view.routes.max_batches),
                               dim3(relaxed_threads), 0, queue, view);
        else
            hipLaunchKernelGGL(HIP_KERNEL_NAME(relaxed_gate_up_kernel<WeightFormat::iq2_xxs, false>), dim3(view.routes.max_batches, view.inter / 128U),
                               dim3(relaxed_threads), 0, queue, view);
        check(hipGetLastError(), "prefill gate/up launch (relaxed)");
        return;
    }
    if (view.routes.batch_rows == 32U && format == WeightFormat::iq2_xxs && view.hidden % (split_chunk_groups * 32U) == 0U && prefill_r2()) {
        // STRIX_DS41_PREFILL_MOE_R2_GU=w: w2_gate_up_kernel (one pass, 32 waves) instead of r2_gate_up_kernel (same bits;
        // measured +30 % at 8K vs v2: one 32-wave workgroup per WGP, 192 VGPR with spills)
        if (const char* gu = std::getenv("STRIX_DS41_PREFILL_MOE_R2_GU"); gu && std::strcmp(gu, "w") == 0) {
            const unsigned m = prefill_v2();
            const dim3 grid(view.routes.max_batches, view.inter / 32U);
            if ((m & (v2_pin | v2_signs)) == (v2_pin | v2_signs))
                hipLaunchKernelGGL(HIP_KERNEL_NAME(w2_gate_up_kernel<WeightFormat::iq2_xxs, v2_on | v2_pin | v2_signs>), grid, dim3(w2_threads),
                                   w2_lds<WeightFormat::iq2_xxs>, queue, view, prefill_band());
            else if (m & v2_pin)
                hipLaunchKernelGGL(HIP_KERNEL_NAME(w2_gate_up_kernel<WeightFormat::iq2_xxs, v2_on | v2_pin>), grid, dim3(w2_threads),
                                   w2_lds<WeightFormat::iq2_xxs>, queue, view, prefill_band());
            else
                hipLaunchKernelGGL(HIP_KERNEL_NAME(w2_gate_up_kernel<WeightFormat::iq2_xxs, v2_on>), grid, dim3(w2_threads),
                                   w2_lds<WeightFormat::iq2_xxs>, queue, view, prefill_band());
            check(hipGetLastError(), "prefill gate/up launch");
            return;
        }
        hipLaunchKernelGGL(HIP_KERNEL_NAME(r2_gate_up_kernel<WeightFormat::iq2_xxs>), dim3(view.routes.max_batches, view.inter / 32U),
                           dim3(split_threads), r2_lds<WeightFormat::iq2_xxs>, queue, view, prefill_band());
        check(hipGetLastError(), "prefill gate/up launch");
        return;
    }
    if (view.routes.batch_rows == 16U && !stage_forced() &&
        !(prefill_tuning::tiles() && format == WeightFormat::q8_0) && view.hidden % 256U == 0U) {
        if (format == WeightFormat::iq2_xxs) launch_split_gate_up<WeightFormat::iq2_xxs>(view, queue);
        else if (format == WeightFormat::q8_0) launch_split_gate_up<WeightFormat::q8_0>(view, queue);
        else launch_split_gate_up<WeightFormat::mxfp4>(view, queue);
        check(hipGetLastError(), "prefill gate/up launch");
        return;
    }
    const bool two = view.routes.batch_rows == 32U;
    if (format == WeightFormat::iq2_xxs) two ? launch_gate_up<WeightFormat::iq2_xxs, 2>(view, queue) : launch_gate_up<WeightFormat::iq2_xxs, 1>(view, queue);
    else if (format == WeightFormat::q8_0) two ? launch_gate_up<WeightFormat::q8_0, 2>(view, queue) : launch_gate_up<WeightFormat::q8_0, 1>(view, queue);
    else two ? launch_gate_up<WeightFormat::mxfp4, 2>(view, queue) : launch_gate_up<WeightFormat::mxfp4, 1>(view, queue);
    check(hipGetLastError(), "prefill gate/up launch");
}

void enqueue_prefill_down(WeightFormat format, const PrefillDownView& view, void* stream) {
    require_routes(view.routes);
    const auto elements = block_elements(format);
    if (format == WeightFormat::iq2_xxs || !view.weights || !view.codes || !view.deltas || !view.out || view.width == 0U ||
        view.width % elements != 0U || view.width % 32U != 0U || view.width > 32U * 32U * max_groups_per_class ||
        view.outputs == 0U || view.outputs % 128U != 0U || view.row_bytes != view.width / elements * block_bytes(format) ||
        reinterpret_cast<std::uintptr_t>(view.codes) % 16U != 0U)
        throw std::invalid_argument("ds41 prefill down view is inconsistent");
    const auto queue = static_cast<hipStream_t>(stream);
    if (fast_moe() && format == WeightFormat::q2_k && view.width % 256U == 0U && view.outputs % 256U == 0U &&
        view.routes.batch_rows <= relaxed_rows) {
        // STRIX_DS41_FAST_MOE_DOWN_ROWS (read per call; default 32, =64 the previous row groups): the same bits either way
        const char* rows_env = std::getenv("STRIX_DS41_FAST_MOE_DOWN_ROWS");
        const bool rows32 = !(rows_env && std::strcmp(rows_env, "64") == 0) && view.routes.batch_rows <= 32U;
        const bool swap = fast_moe_order() && view.routes.max_batches <= 65535U;  // grid y limit
        const dim3 grid = swap ? dim3(view.outputs / 256U, view.routes.max_batches) : dim3(view.routes.max_batches, view.outputs / 256U);
        if (swap && rows32) hipLaunchKernelGGL(HIP_KERNEL_NAME(relaxed_down_kernel<WeightFormat::q2_k, true, 32U>), grid, dim3(relaxed_threads), 0, queue, view);
        else if (swap) hipLaunchKernelGGL(HIP_KERNEL_NAME(relaxed_down_kernel<WeightFormat::q2_k, true>), grid, dim3(relaxed_threads), 0, queue, view);
        else if (rows32) hipLaunchKernelGGL(HIP_KERNEL_NAME(relaxed_down_kernel<WeightFormat::q2_k, false, 32U>), grid, dim3(relaxed_threads), 0, queue, view);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(relaxed_down_kernel<WeightFormat::q2_k, false>), grid, dim3(relaxed_threads), 0, queue, view);
        check(hipGetLastError(), "prefill down launch (relaxed)");
        return;
    }
    if (view.routes.batch_rows == 32U && format == WeightFormat::q2_k && view.width % 256U == 0U && prefill_r2()) {
        // STRIX_DS41_PREFILL_MOE_R2_DOWN=0: the v2 down kernel in 16-row halves instead of r2_down_kernel
        const char* rd = std::getenv("STRIX_DS41_PREFILL_MOE_R2_DOWN");
        if (rd && std::strcmp(rd, "0") == 0) launch_split_down<WeightFormat::q2_k>(view, queue);
        else hipLaunchKernelGGL(r2_down_kernel, dim3(view.routes.max_batches, view.outputs / 64U), dim3(split_threads), r2d_lds, queue, view,
                                prefill_band());
        check(hipGetLastError(), "prefill down launch");
        return;
    }
    if (view.routes.batch_rows == 16U && !stage_forced() &&
        !(prefill_tuning::tiles() && (format == WeightFormat::q2_k || format == WeightFormat::q8_0 || prefill_tuning::down_stage(view.width, view.outputs))) && view.width % 256U == 0U) {
        if (format == WeightFormat::q2_k) launch_split_down<WeightFormat::q2_k>(view, queue);
        else if (format == WeightFormat::q8_0) launch_split_down<WeightFormat::q8_0>(view, queue);
        else launch_split_down<WeightFormat::mxfp4>(view, queue);
        check(hipGetLastError(), "prefill down launch");
        return;
    }
    const bool two = view.routes.batch_rows == 32U;
    if (format == WeightFormat::q2_k) two ? launch_down<WeightFormat::q2_k, 2>(view, queue) : launch_down<WeightFormat::q2_k, 1>(view, queue);
    else if (format == WeightFormat::q8_0) two ? launch_down<WeightFormat::q8_0, 2>(view, queue) : launch_down<WeightFormat::q8_0, 1>(view, queue);
    else two ? launch_down<WeightFormat::mxfp4, 2>(view, queue) : launch_down<WeightFormat::mxfp4, 1>(view, queue);
    check(hipGetLastError(), "prefill down launch");
}

namespace {
// Slot s's A8 activation (inter codes, inter / 32 deltas) at the down's stride: codes and deltas copied, the groups
// past inter zero (code 0, delta 0: each adds +0 to its lane class, so the product is the inter-wide one).
__global__ __launch_bounds__(256) void pad_a8_kernel(const std::int8_t* codes, const float* deltas, std::uint32_t inter,
                                                     std::uint32_t width, std::int8_t* out_codes, float* out_deltas) {
    const std::uint64_t s = blockIdx.x;
    const auto* src = codes + s * inter;
    auto* dst = out_codes + s * width;
    for (unsigned i = threadIdx.x; i < width; i += blockDim.x) dst[i] = i < inter ? src[i] : static_cast<std::int8_t>(0);
    for (unsigned g = threadIdx.x; g < width / 32U; g += blockDim.x)
        out_deltas[s * (width / 32U) + g] = g < inter / 32U ? deltas[s * (inter / 32U) + g] : 0.0F;
}
}  // namespace

void PrefillMoe::pad_activations(std::uint32_t rows, void* stream) {
    if (pad_codes_ == nullptr || rows == 0U) return;
    hipLaunchKernelGGL(pad_a8_kernel, dim3(rows * top_k_), dim3(256), 0, static_cast<hipStream_t>(stream), act_codes_, act_deltas_,
                       w_.inter, w_.down_width, pad_codes_, pad_deltas_);
    check(hipGetLastError(), "prefill MoE pad");
}

PrefillMoe::PrefillMoe(const PrefillMoeWeights& weights, std::uint32_t top_k, std::uint32_t experts, std::uint32_t split,
                       std::uint32_t chunk_rows, std::uint32_t first, std::uint32_t end)
    : w_(weights), top_k_(top_k), experts_(experts), split_(split), chunk_rows_(chunk_rows), first_(first), end_(end ? end : experts) {
    if (top_k == 0U || top_k > 16U || experts == 0U || chunk_rows == 0U || first_ >= end_ || end_ > experts || !w_.gate || !w_.up ||
        !w_.down || w_.hidden == 0U || w_.inter == 0U || (w_.down_width != 0U && (w_.down_width < w_.inter || w_.down_width % 32U != 0U)))
        throw std::invalid_argument("ds41 prefill MoE configuration is inconsistent");
    if (w_.down_width == 0U) w_.down_width = w_.inter;
    const std::uint64_t slots = static_cast<std::uint64_t>(chunk_rows) * top_k, owned = end_ - first_;
    // Batches of 8 (window path) bound the batch count of either path.
    const std::uint64_t batches = prefill_max_batches(chunk_rows, top_k, static_cast<std::uint32_t>(owned), 8U);
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_WINDOW_MEAN"); v && *v) window_mean_ = std::strtod(v, nullptr);
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_SELECT"); v && *v) select_hist_ = std::strcmp(v, "mean") != 0;
    // One allocation, 256-byte aligned parts.
    std::uint64_t offset = 0;
    const auto take = [&](std::uint64_t bytes) { const auto at = offset; offset += (bytes + 255U) / 256U * 256U; return at; };
    const auto o_counts = take(owned * 4U), o_begins = take((owned + 1U) * 4U), o_rows = take(slots * 4U), o_slots = take(slots * 4U);
    const auto o_bexp = take(batches * 4U), o_bbeg = take((batches + 1U) * 4U), o_bcnt = take(batches * 4U), o_btot = take(4U);
    const auto o_order = take(slots * 4U), o_low = take(static_cast<std::uint64_t>(chunk_rows) * 4U);
    const auto o_sexp = take(batches * 4U), o_sbeg = take((batches + 1U) * 4U), o_stot = take(4U);
    const auto o_codes = take(slots * w_.inter), o_deltas = take(slots * w_.inter / 32U * 4U), o_sums = take(slots * w_.inter / 32U * 4U);
    const auto o_raw = take(slots * w_.hidden * 4U);
    const bool pad = w_.down_width != w_.inter;
    const auto o_pcodes = pad ? take(slots * w_.down_width) : 0U, o_pdeltas = pad ? take(slots * w_.down_width / 32U * 4U) : 0U;
    scratch_bytes_ = offset;
    check(hipMalloc(&memory_, offset), "prefill MoE scratch");
    auto* base = static_cast<unsigned char*>(memory_);
    const auto u32 = [&](std::uint64_t at) { return reinterpret_cast<std::uint32_t*>(base + at); };
    routes_ = {u32(o_counts), u32(o_begins), u32(o_rows), u32(o_slots), u32(o_bexp), u32(o_bbeg), u32(o_bcnt), u32(o_btot),
               u32(o_order), u32(o_low)};
    routes_.sparse_expert = u32(o_sexp);
    routes_.sparse_begin = u32(o_sbeg);
    routes_.sparse_total = u32(o_stot);
    act_codes_ = reinterpret_cast<std::int8_t*>(base + o_codes);
    act_deltas_ = reinterpret_cast<float*>(base + o_deltas);
    act_sums_ = reinterpret_cast<std::int32_t*>(base + o_sums);
    raw_ = reinterpret_cast<float*>(base + o_raw);
    if (pad) {
        pad_codes_ = reinterpret_cast<std::int8_t*>(base + o_pcodes);
        pad_deltas_ = reinterpret_cast<float*>(base + o_pdeltas);
    }
}
PrefillMoe::~PrefillMoe() {
    (void)hipFree(memory_);
    if (joint_) (void)hipFree(joint_);
}
void PrefillMoe::set_weights(const PrefillMoeWeights& weights) {
    if (!weights.gate || !weights.up || !weights.down || weights.hidden != w_.hidden || weights.inter != w_.inter ||
        (weights.down_width != 0U ? weights.down_width : weights.inter) != w_.down_width ||
        weights.gate_up_format != w_.gate_up_format || weights.down_format != w_.down_format)
        throw std::invalid_argument("ds41 prefill MoE: set_weights needs the same geometry and formats");
    const std::uint32_t down_width = w_.down_width;
    w_ = weights;
    w_.down_width = down_width;
}

void PrefillMoe::enqueue_experts(const PrefillMoeInputs& in, std::uint32_t chunk, std::uint32_t& row0, std::uint32_t& rows, void* stream) {
    if (!in.ids || !in.weights || !in.codes || !in.deltas || in.rows == 0U || chunk >= chunks(in.rows))
        throw std::invalid_argument("ds41 prefill MoE inputs are inconsistent");
    row0 = chunk * chunk_rows_;
    rows = std::min(chunk_rows_, in.rows - row0);
    experts_rows(in, row0, rows, stream);
}
// Routes and expert products of rows [row0, row0 + rows) of `in` (at most chunk_rows), by the selected path.
unsigned PrefillMoe::selected_batch_rows(unsigned rows) const {
    if (!prefill_tuning::tiles())
        return prefill_r2() && w_.gate_up_format == WeightFormat::iq2_xxs && w_.down_format == WeightFormat::q2_k &&
                       w_.hidden % (split_chunk_groups * 32U) == 0U && w_.down_width % 256U == 0U ? 32U : 16U;
    return prefill_tuning::tile_override("STRIX_DS41_PREFILL_MOE_TILE",
        prefill_tuning::moe_rows(w_.hidden, w_.down_width, rows, top_k_, end_ - first_));
}
void PrefillMoe::experts_rows(const PrefillMoeInputs& in, std::uint32_t row0, std::uint32_t rows, void* stream) {
    const std::uint32_t owned = end_ - first_;
    const auto* ids = in.ids + static_cast<std::uint64_t>(row0) * top_k_;
    if (select_hist_) {
        enqueue_experts_split(in, row0, rows, stream);
        return;
    }
    routes_.sparse_max = 0U;
    // FAST MoE: every entry takes the relaxed batch kernels, so its arithmetic does not depend on the chunking (rows per
    // call would otherwise move small chunks onto the exact window kernels)
    const bool window = !fast_moe() && static_cast<double>(rows) * top_k_ <= window_mean_ * owned;
    const std::uint32_t batch_rows = window ? 8U : selected_batch_rows(rows);
    enqueue_build_prefill_routes(ids, rows, top_k_, experts_, split_, batch_rows, routes_, stream, first_, end_);
    if (window) {
        // Each batch (<= 8 entries of one expert) is one window block of the decode kernels.
        RoutesView gr;
        gr.experts = routes_.batch_expert; gr.begins = routes_.batch_begin; gr.inputs = routes_.entry_rows; gr.outputs = routes_.entry_slots;
        gr.expert_count = prefill_max_batches(rows, top_k_, owned, 8U); gr.max_rows = 8U; gr.counts = routes_.batch_total;
        GateUpView gu;
        gu.gate = w_.gate; gu.up = w_.up; gu.expert_stride_bytes = w_.gate_row_bytes * w_.inter; gu.row_bytes = w_.gate_row_bytes;
        gu.hidden = w_.hidden; gu.inter = w_.inter; gu.routes = gr;
        gu.codes = in.codes + static_cast<std::uint64_t>(row0) * w_.hidden;
        gu.deltas = in.deltas + static_cast<std::uint64_t>(row0) * (w_.hidden / 32U);
        gu.limit = w_.limit; gu.act_codes = act_codes_; gu.act_deltas = act_deltas_; gu.act_sums = act_sums_;
        enqueue_gate_up(w_.gate_up_format, gu, stream);
        RoutesProductView dn;
        RoutesView dr = gr;
        dr.inputs = routes_.entry_slots;
        dn.weights = w_.down; dn.expert_stride_bytes = w_.down_row_bytes * w_.hidden; dn.row_bytes = w_.down_row_bytes;
        pad_activations(rows, stream);
        dn.width = w_.down_width; dn.outputs = w_.hidden; dn.routes = dr; dn.codes = down_codes(); dn.deltas = down_deltas(); dn.out = raw_;
        enqueue_routes_product(w_.down_format, dn, stream);
        return;
    }
    const auto view = prefill_routes_view(routes_, rows, top_k_, owned, batch_rows);
    PrefillGateUpView gu;
    gu.gate = w_.gate; gu.up = w_.up; gu.expert_stride_bytes = w_.gate_row_bytes * w_.inter; gu.row_bytes = w_.gate_row_bytes;
    gu.hidden = w_.hidden; gu.inter = w_.inter; gu.routes = view;
    gu.codes = in.codes + static_cast<std::uint64_t>(row0) * w_.hidden;
    gu.deltas = in.deltas + static_cast<std::uint64_t>(row0) * (w_.hidden / 32U);
    gu.limit = w_.limit; gu.act_codes = act_codes_; gu.act_deltas = act_deltas_; gu.act_sums = act_sums_;
    enqueue_prefill_gate_up(w_.gate_up_format, gu, stream);
    PrefillDownView dn;
    dn.weights = w_.down; dn.expert_stride_bytes = w_.down_row_bytes * w_.hidden; dn.row_bytes = w_.down_row_bytes;
    pad_activations(rows, stream);
    dn.width = w_.down_width; dn.outputs = w_.hidden; dn.routes = view; dn.codes = down_codes(); dn.deltas = down_deltas(); dn.out = raw_;
    enqueue_prefill_down(w_.down_format, dn, stream);
}
// Histogram selector: one route build with the sparse experts (<= window_mean entries) first; their batches of 8 run
// the decode window kernels and the dense experts' batches of 16 the WMMA kernels (each kernel exits at once when its
// list is empty). Every entry is the declared product whichever kernel runs it.
void PrefillMoe::enqueue_experts_split(const PrefillMoeInputs& in, std::uint32_t row0, std::uint32_t rows, void* stream) {
    const std::uint32_t owned = end_ - first_;
    const auto* ids = in.ids + static_cast<std::uint64_t>(row0) * top_k_;
    routes_.sparse_max = window_mean_ > 0.0 && !fast_moe() ? static_cast<std::uint32_t>(window_mean_) : 0U;   // FAST MoE: no sparse split
    const unsigned batch_rows = selected_batch_rows(rows);
    enqueue_build_prefill_routes(ids, rows, top_k_, experts_, split_, batch_rows, routes_, stream, first_, end_);
    const auto* codes = in.codes + static_cast<std::uint64_t>(row0) * w_.hidden;
    const auto* deltas = in.deltas + static_cast<std::uint64_t>(row0) * (w_.hidden / 32U);
    if (routes_.sparse_max > 0U) {
        RoutesView gr;
        gr.experts = routes_.sparse_expert; gr.begins = routes_.sparse_begin; gr.inputs = routes_.entry_rows; gr.outputs = routes_.entry_slots;
        gr.expert_count = prefill_max_batches(rows, top_k_, owned, 8U); gr.max_rows = 8U; gr.counts = routes_.sparse_total;
        GateUpView gu;
        gu.gate = w_.gate; gu.up = w_.up; gu.expert_stride_bytes = w_.gate_row_bytes * w_.inter; gu.row_bytes = w_.gate_row_bytes;
        gu.hidden = w_.hidden; gu.inter = w_.inter; gu.routes = gr; gu.codes = codes; gu.deltas = deltas;
        gu.limit = w_.limit; gu.act_codes = act_codes_; gu.act_deltas = act_deltas_; gu.act_sums = act_sums_;
        enqueue_gate_up(w_.gate_up_format, gu, stream);
        RoutesProductView dn;
        RoutesView dr = gr;
        dr.inputs = routes_.entry_slots;
        dn.weights = w_.down; dn.expert_stride_bytes = w_.down_row_bytes * w_.hidden; dn.row_bytes = w_.down_row_bytes;
        pad_activations(rows, stream);
        dn.width = w_.down_width; dn.outputs = w_.hidden; dn.routes = dr; dn.codes = down_codes(); dn.deltas = down_deltas(); dn.out = raw_;
        enqueue_routes_product(w_.down_format, dn, stream);
    }
    const auto view = prefill_routes_view(routes_, rows, top_k_, owned, batch_rows);
    PrefillGateUpView gu;
    gu.gate = w_.gate; gu.up = w_.up; gu.expert_stride_bytes = w_.gate_row_bytes * w_.inter; gu.row_bytes = w_.gate_row_bytes;
    gu.hidden = w_.hidden; gu.inter = w_.inter; gu.routes = view; gu.codes = codes; gu.deltas = deltas;
    gu.limit = w_.limit; gu.act_codes = act_codes_; gu.act_deltas = act_deltas_; gu.act_sums = act_sums_;
    enqueue_prefill_gate_up(w_.gate_up_format, gu, stream);
    PrefillDownView dn;
    dn.weights = w_.down; dn.expert_stride_bytes = w_.down_row_bytes * w_.hidden; dn.row_bytes = w_.down_row_bytes;
    pad_activations(rows, stream);
    dn.width = w_.down_width; dn.outputs = w_.hidden; dn.routes = view; dn.codes = down_codes(); dn.deltas = down_deltas(); dn.out = raw_;
    enqueue_prefill_down(w_.down_format, dn, stream);
}
void PrefillMoe::enqueue_combine(const PrefillMoeInputs& in, std::uint32_t chunk, float* y, void* stream) {
    if (first_ != 0U || end_ != experts_ || !in.shared || !y) throw std::invalid_argument("ds41 prefill MoE combine needs every expert");
    std::uint32_t row0 = 0, rows = 0;
    enqueue_experts(in, chunk, row0, rows, stream);
    CombineView c;
    c.raw = raw_; c.shared = in.shared + static_cast<std::uint64_t>(row0) * w_.hidden; c.weights = in.weights + static_cast<std::uint64_t>(row0) * top_k_;
    c.order = routes_.order; c.low_counts = routes_.low_counts; c.rows = rows; c.top_k = top_k_; c.hidden = w_.hidden;
    c.out = y + static_cast<std::uint64_t>(row0) * w_.hidden;
    strix::ds41::enqueue_combine(c, stream);
}
// Half sums of rows [first_row, first_row + rows) of the current chunk's routes into out[row - first_row].
void PrefillMoe::half_rows(const float* weights, std::uint32_t first_row, std::uint32_t rows, std::uint32_t half, float* out, void* stream) {
    if (rows == 0U) return;
    HalfSumView v;
    v.raw = raw_; v.weights = weights; v.order = routes_.order + static_cast<std::uint64_t>(first_row) * top_k_;
    v.low_counts = routes_.low_counts + first_row; v.rows = rows; v.top_k = top_k_; v.hidden = w_.hidden; v.half = half; v.out = out;
    enqueue_half_sum(v, stream);
}
void PrefillMoe::enqueue_half_joint(const PrefillMoeInputs& peer, const PrefillMoeInputs& own, std::uint32_t half, float* out_peer,
                                    float* out_own, void* stream, void* peer_done) {
    const bool covers = half == 0U ? first_ == 0U && end_ == split_ : first_ == split_ && end_ == experts_;
    if (half > 1U || !covers || (peer.rows && (!peer.ids || !peer.weights || !peer.codes || !peer.deltas || !out_peer)) ||
        (own.rows && (!own.ids || !own.weights || !own.codes || !own.deltas || !out_own)))
        throw std::invalid_argument("ds41 prefill MoE joint half: inconsistent inputs or owned range");
    const auto queue = static_cast<hipStream_t>(stream);
    const std::uint64_t H = w_.hidden, G = w_.hidden / 32U, K = top_k_, C = chunk_rows_;
    const std::uint64_t codes_at = 0, deltas_at = (C * H + 255U) / 256U * 256U, ids_at = deltas_at + (C * G * 4U + 255U) / 256U * 256U,
                        weights_at = ids_at + (C * K * 4U + 255U) / 256U * 256U, total = weights_at + C * K * 4U;
    if (!joint_) check(hipMalloc(&joint_, total), "prefill MoE joint scratch");
    auto* base = static_cast<unsigned char*>(joint_);
    PrefillMoeInputs j;
    j.codes = reinterpret_cast<const std::int8_t*>(base + codes_at);
    j.deltas = reinterpret_cast<const float*>(base + deltas_at);
    j.ids = reinterpret_cast<const std::uint32_t*>(base + ids_at);
    j.weights = reinterpret_cast<const float*>(base + weights_at);
    const std::uint32_t np = peer.rows, rows_all = peer.rows + own.rows;
    // copies rows [from, from + n) of `src` to joint row `to`
    const auto copy = [&](const PrefillMoeInputs& src, std::uint32_t from, std::uint32_t n, std::uint32_t to) {
        if (n == 0U) return;
        check(hipMemcpyAsync(base + codes_at + to * H, src.codes + from * H, n * H, hipMemcpyDeviceToDevice, queue), "joint codes");
        check(hipMemcpyAsync(base + deltas_at + to * G * 4U, src.deltas + from * G, n * G * 4U, hipMemcpyDeviceToDevice, queue), "joint deltas");
        check(hipMemcpyAsync(base + ids_at + to * K * 4U, src.ids + from * K, n * K * 4U, hipMemcpyDeviceToDevice, queue), "joint ids");
        check(hipMemcpyAsync(base + weights_at + to * K * 4U, src.weights + from * K, n * K * 4U, hipMemcpyDeviceToDevice, queue), "joint weights");
    };
    for (std::uint32_t r0 = 0; r0 < rows_all; r0 += chunk_rows_) {
        const std::uint32_t n = std::min(chunk_rows_, rows_all - r0);
        const std::uint32_t kp = r0 < np ? std::min(n, np - r0) : 0U;  // the chunk's peer rows come first
        copy(peer, r0, kp, 0U);
        copy(own, r0 + kp - np, n - kp, kp);
        j.rows = n;
        experts_rows(j, 0U, n, stream);
        half_rows(j.weights, 0U, kp, half, out_peer + static_cast<std::uint64_t>(r0) * H, stream);
        if (peer_done && kp > 0U && r0 + kp == np) check(hipEventRecord(static_cast<hipEvent_t>(peer_done), queue), "peer done");
        half_rows(j.weights, kp, n - kp, half, out_own + static_cast<std::uint64_t>(r0 + kp - np) * H, stream);
    }
    if (peer_done && np == 0U) check(hipEventRecord(static_cast<hipEvent_t>(peer_done), queue), "peer done");
}
void PrefillMoe::enqueue_half(const PrefillMoeInputs& in, std::uint32_t chunk, std::uint32_t half, float* out, void* stream) {
    const bool covers = half == 0U ? first_ == 0U && end_ == split_ : first_ == split_ && end_ == experts_;
    if (half > 1U || !covers || !out) throw std::invalid_argument("ds41 prefill MoE half: the owned experts must be that half");
    std::uint32_t row0 = 0, rows = 0;
    enqueue_experts(in, chunk, row0, rows, stream);
    HalfSumView v;
    v.raw = raw_; v.weights = in.weights + static_cast<std::uint64_t>(row0) * top_k_; v.order = routes_.order;
    v.low_counts = routes_.low_counts; v.rows = rows; v.top_k = top_k_; v.hidden = w_.hidden; v.half = half;
    v.out = out + static_cast<std::uint64_t>(row0) * w_.hidden;
    enqueue_half_sum(v, stream);
}

} // namespace strix::ds41
