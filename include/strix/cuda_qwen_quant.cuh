// Weight decoders of the GGUF quant types used by the Qwen3.8-Flash-Next builds (UD-Q4_K_XL product file and the GSQ-RCO
// IQ3_S file), for the RTX engine. One source for host and device: every function is `QQ_HD` and keeps the float
// expressions of src/ggml_dequant.cpp (the GGUF reference decoders: same operations, same order, no contraction; the device
// side compiles with --fmad=false), so a device decode equals the host oracle bit for bit
// (tests/cuda/qwen_quant_host_test.cpp checks the host path against ggml_dequant on real rows of both files).
//
// Granularity: a 32-value group. Group g of a row is in block g / (block_elements / 32), sub-group g % (...).
// decode_group(type, block, sub, out[32]) writes the 32 values of that sub-group.
#pragma once

#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define QQ_HD __host__ __device__ __forceinline__
#else
#define QQ_HD inline
#endif

namespace strix::cuda::quant {

enum : std::uint32_t {
    t_f32 = 0, t_f16 = 1, t_q4_0 = 2, t_q5_0 = 6, t_q5_1 = 7, t_q8_0 = 8, t_q4_k = 12, t_q5_k = 13, t_q6_k = 14,
    t_iq3_xxs = 18, t_iq4_nl = 20, t_iq3_s = 21, t_iq2_s = 22, t_iq4_xs = 23, t_bf16 = 30, t_q2_0 = 42,
};

struct Geometry { std::uint32_t block_elements, block_bytes; };
// {0, 0} for a type these decoders do not cover.
QQ_HD Geometry geometry(std::uint32_t t) {
    switch (t) {
    case t_f32: return {32, 128};
    case t_f16: case t_bf16: return {32, 64};
    case t_q4_0: case t_iq4_nl: return {32, 18};
    case t_q5_0: return {32, 22};
    case t_q5_1: return {32, 24};
    case t_q8_0: return {32, 34};
    case t_q2_0: return {64, 18};
    case t_q4_k: return {256, 144};
    case t_q5_k: return {256, 176};
    case t_q6_k: return {256, 210};
    case t_iq4_xs: return {256, 136};
    case t_iq2_s: return {256, 82};
    case t_iq3_xxs: return {256, 98};
    case t_iq3_s: return {256, 110};
    default: return {0, 0};
    }
}

namespace detail {

static constexpr std::uint64_t iq2s_grid_h[1024] = {
#include "strix/qwen_iq2_s_grid.inc"
};
static constexpr std::uint32_t iq3xxs_grid_h[256] = {
#include "strix/qwen_iq3_xxs_grid.inc"
};
static constexpr std::uint32_t iq3s_grid_h[512] = {
#include "strix/qwen_iq3_s_grid.inc"
};
static constexpr std::int8_t kvalues_iq4nl_h[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
#ifdef __CUDACC__
static __device__ const std::uint64_t iq2s_grid_d[1024] = {
#include "strix/qwen_iq2_s_grid.inc"
};
static __device__ const std::uint32_t iq3xxs_grid_d[256] = {
#include "strix/qwen_iq3_xxs_grid.inc"
};
static __device__ const std::uint32_t iq3s_grid_d[512] = {
#include "strix/qwen_iq3_s_grid.inc"
};
static __device__ const std::int8_t kvalues_iq4nl_d[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
#endif


QQ_HD std::uint64_t iq2s_grid(unsigned i) {
#ifdef __CUDA_ARCH__
    return iq2s_grid_d[i];
#else
    return iq2s_grid_h[i];
#endif
}
QQ_HD std::uint32_t iq3xxs_grid(unsigned i) {
#ifdef __CUDA_ARCH__
    return iq3xxs_grid_d[i];
#else
    return iq3xxs_grid_h[i];
#endif
}
QQ_HD std::uint32_t iq3s_grid(unsigned i) {
#ifdef __CUDA_ARCH__
    return iq3s_grid_d[i];
#else
    return iq3s_grid_h[i];
#endif
}
QQ_HD int kvalue(unsigned i) {
#ifdef __CUDA_ARCH__
    return kvalues_iq4nl_d[i];
#else
    return kvalues_iq4nl_h[i];
#endif
}

QQ_HD std::uint16_t u16(const std::uint8_t* p) { return static_cast<std::uint16_t>(p[0] | (p[1] << 8)); }
QQ_HD std::uint32_t u32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) | (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}
// Exact FP16 -> FP32 (normals, subnormals, inf/NaN), the GGML_FP16_TO_FP32 value.
QQ_HD float fp16(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16U;
    std::uint32_t exp = (h >> 10U) & 0x1fU, man = h & 0x3ffU, bits;
    if (exp == 0U) {
        if (man == 0U) bits = sign;
        else {  // subnormal: normalize
            exp = 127U - 15U + 1U;
            while ((man & 0x400U) == 0U) { man <<= 1U; --exp; }
            bits = sign | (exp << 23U) | ((man & 0x3ffU) << 13U);
        }
    } else if (exp == 31U) bits = sign | 0x7f800000U | (man << 13U) | (man ? 0x00400000U : 0U);  // NaN quieted, as the oracle
    else bits = sign | ((exp + 112U) << 23U) | (man << 13U);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
QQ_HD float bf16(std::uint16_t h) {
    const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16U;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
QQ_HD std::uint8_t ksigns(unsigned i) {
    unsigned p = i;  // popcount parity of the 7 bits
    p ^= p >> 4U; p ^= p >> 2U; p ^= p >> 1U;
    return static_cast<std::uint8_t>(i | ((p & 1U) << 7U));
}
QQ_HD void scale_min_k4(int j, const std::uint8_t* q, std::uint8_t* d, std::uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = static_cast<std::uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = static_cast<std::uint8_t>((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}
QQ_HD std::uint8_t grid_byte(std::uint64_t g, unsigned j) { return static_cast<std::uint8_t>((g >> (8U * j)) & 255U); }
}  // namespace detail

// The 32 values of sub-group `sub` of one block (x = block start).
QQ_HD void decode_group(std::uint32_t t, const std::uint8_t* x, unsigned sub, float* y) {
    using namespace detail;
    switch (t) {
    case t_f32: for (int j = 0; j < 32; ++j) std::memcpy(&y[j], x + 4 * j, 4); return;
    case t_f16: for (int j = 0; j < 32; ++j) y[j] = fp16(u16(x + 2 * j)); return;
    case t_bf16: for (int j = 0; j < 32; ++j) y[j] = bf16(u16(x + 2 * j)); return;
    case t_q8_0: {
        const float d = fp16(u16(x));
        for (int j = 0; j < 32; ++j) y[j] = static_cast<std::int8_t>(x[2 + j]) * d;
        return;
    }
    case t_q4_0: {
        const float d = fp16(u16(x)); const std::uint8_t* qs = x + 2;
        for (int j = 0; j < 16; ++j) { const int x0 = (qs[j] & 0x0F) - 8, x1 = (qs[j] >> 4) - 8; y[j] = x0 * d; y[j + 16] = x1 * d; }
        return;
    }
    case t_q5_0: {
        const float d = fp16(u16(x)); const std::uint32_t qh = u32(x + 2); const std::uint8_t* qs = x + 6;
        for (int j = 0; j < 16; ++j) {
            const std::uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10, xh_1 = ((qh >> (j + 12))) & 0x10;
            const std::int32_t x0 = ((qs[j] & 0x0F) | xh_0) - 16, x1 = ((qs[j] >> 4) | xh_1) - 16;
            y[j] = x0 * d; y[j + 16] = x1 * d;
        }
        return;
    }
    case t_q5_1: {
        const float d = fp16(u16(x)), m = fp16(u16(x + 2)); const std::uint32_t qh = u32(x + 4); const std::uint8_t* qs = x + 8;
        for (int j = 0; j < 16; ++j) {
            const std::uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10, xh_1 = ((qh >> (j + 12))) & 0x10;
            const int x0 = (qs[j] & 0x0F) | xh_0, x1 = (qs[j] >> 4) | xh_1;
            const float p0 = x0 * d, p1 = x1 * d;   // no contraction: x * d, then + m
            y[j] = p0 + m; y[j + 16] = p1 + m;
        }
        return;
    }
    case t_iq4_nl: {
        const float d = fp16(u16(x)); const std::uint8_t* qs = x + 2;
        for (int j = 0; j < 16; ++j) { y[j] = d * kvalue(qs[j] & 0xf); y[j + 16] = d * kvalue(qs[j] >> 4); }
        return;
    }
    case t_q2_0: {  // fp16 d, 16 bytes of 2-bit codes, value (q - 1) * d; 64 values = 2 sub-groups
        const float d = fp16(u16(x));
        for (int k = 0; k < 32; ++k) {
            const int j = static_cast<int>(32 * sub) + k;
            const std::uint8_t q = (x[2 + j / 4] >> ((j % 4) * 2)) & 0x03;
            y[k] = (static_cast<int>(q) - 1) * d;
        }
        return;
    }
    case t_q4_k: {
        const std::uint8_t* q = x + 16 + 32 * (sub / 2);
        const float d = fp16(u16(x)), min = fp16(u16(x + 2));
        std::uint8_t sc, m;
        scale_min_k4(static_cast<int>(sub), x + 4, &sc, &m);
        const float d1 = d * sc, m1 = min * m;
        for (int l = 0; l < 32; ++l) { const float p = d1 * ((sub & 1) ? (q[l] >> 4) : (q[l] & 0xF)); y[l] = p - m1; }
        return;
    }
    case t_q5_k: {
        const std::uint8_t* ql = x + 48 + 32 * (sub / 2); const std::uint8_t* qh = x + 16;
        const float d = fp16(u16(x)), min = fp16(u16(x + 2));
        std::uint8_t sc, m;
        scale_min_k4(static_cast<int>(sub), x + 4, &sc, &m);
        const float d1 = d * sc, m1 = min * m;
        const std::uint8_t u = static_cast<std::uint8_t>(1U << sub);
        for (int l = 0; l < 32; ++l) {
            const int v = ((sub & 1) ? (ql[l] >> 4) : (ql[l] & 0xF)) + (qh[l] & u ? 16 : 0);
            const float p = d1 * v;
            y[l] = p - m1;
        }
        return;
    }
    case t_q6_k: {
        const float d = fp16(u16(x + 208));
        const unsigned h = sub / 4, k = sub % 4;
        const std::uint8_t* ql = x + 64 * h + ((k & 1) ? 32 : 0);
        const std::uint8_t* qh = x + 128 + 32 * h;
        const auto* sc = reinterpret_cast<const std::int8_t*>(x + 192 + 8 * h);
        for (int l = 0; l < 32; ++l) {
            const int lo = (k >= 2) ? (ql[l] >> 4) : (ql[l] & 0xF);
            const std::int8_t q = static_cast<std::int8_t>(static_cast<std::int8_t>(lo | (((qh[l] >> (2 * k)) & 3) << 4)) - 32);
            const float ds = d * sc[l / 16 + 2 * k];
            y[l] = ds * q;
        }
        return;
    }
    case t_iq4_xs: {
        const float d = fp16(u16(x)); const std::uint16_t scales_h = u16(x + 2); const std::uint8_t* scales_l = x + 4;
        const std::uint8_t* qs = x + 8 + 16 * sub;
        const int ls = ((scales_l[sub / 2] >> 4 * (sub % 2)) & 0xf) | (((scales_h >> 2 * sub) & 3) << 4);
        const float dl = d * (ls - 32);
        for (int j = 0; j < 16; ++j) { y[j] = dl * kvalue(qs[j] & 0xf); y[j + 16] = dl * kvalue(qs[j] >> 4); }
        return;
    }
    case t_iq2_s: {
        const float d = fp16(u16(x));
        const std::uint8_t* qs = x + 2 + 4 * sub; const std::uint8_t* qh = x + 66; const std::uint8_t* scales = x + 74;
        const std::uint8_t* signs = x + 2 + 32 + 4 * sub;
        float db[2];
        db[0] = d * (0.5f + (scales[sub] & 0xf)) * 0.25f;
        db[1] = d * (0.5f + (scales[sub] >> 4)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const float dl = db[l / 2];
            const std::uint64_t g = iq2s_grid(qs[l] | (qh[sub] << (8 - 2 * l) & 0x300));
            for (int j = 0; j < 8; ++j) { const float p = dl * grid_byte(g, j); y[8 * l + j] = p * (signs[l] & (1U << j) ? -1.f : 1.f); }
        }
        return;
    }
    case t_iq3_xxs: {
        const float d = fp16(u16(x));
        const std::uint8_t* qs = x + 2 + 8 * sub;
        const std::uint32_t aux32 = u32(x + 2 + 64 + 4 * sub);
        const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const std::uint8_t signs = ksigns((aux32 >> 7 * l) & 127);
            const std::uint32_t g1 = iq3xxs_grid(qs[2 * l + 0]), g2 = iq3xxs_grid(qs[2 * l + 1]);
            for (int j = 0; j < 4; ++j) {
                const float p1 = db * grid_byte(g1, j), p2 = db * grid_byte(g2, j);
                y[8 * l + j + 0] = p1 * (signs & (1U << (j + 0)) ? -1.f : 1.f);
                y[8 * l + j + 4] = p2 * (signs & (1U << (j + 4)) ? -1.f : 1.f);
            }
        }
        return;
    }
    case t_iq3_s: {
        const float d = fp16(u16(x));
        const std::uint8_t* qs = x + 2 + 8 * sub; const std::uint8_t qh = x[66 + sub];
        const std::uint8_t* signs = x + 74 + 4 * sub; const std::uint8_t sb = x[106 + sub / 2];
        const float db = d * (1 + 2 * ((sub & 1) ? (sb >> 4) : (sb & 0xf)));
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t g1 = iq3s_grid(qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 256));
            const std::uint32_t g2 = iq3s_grid(qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256));
            for (int j = 0; j < 4; ++j) {
                const float p1 = db * grid_byte(g1, j), p2 = db * grid_byte(g2, j);
                y[8 * l + j + 0] = p1 * (signs[l] & (1U << (j + 0)) ? -1.f : 1.f);
                y[8 * l + j + 4] = p2 * (signs[l] & (1U << (j + 4)) ? -1.f : 1.f);
            }
        }
        return;
    }
    default: for (int j = 0; j < 32; ++j) y[j] = 0.0F; return;
    }
}

// Group g (32 values) of a row of `type` starting at `row`.
QQ_HD void decode_row_group(std::uint32_t t, const std::uint8_t* row, std::uint32_t g, float* y) {
    const Geometry geo = geometry(t);
    const std::uint32_t per = geo.block_elements / 32U;
    decode_group(t, row + static_cast<std::uint64_t>(g / per) * geo.block_bytes, g % per, y);
}

}  // namespace strix::cuda::quant
