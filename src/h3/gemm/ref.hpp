#pragma once
// MiniMax H3 W8A8 tensorwise + convrot projection: host reference (plain C++, the arithmetic spec of the kernels).
//
// One projection y[M][N] = linear(x[M][K], W[N][K]) of an int8 convrot checkpoint, exactly as the reference chain:
//   1. rotation: per row, per 256-feature group g, xr[g][j] = bf16( sum_i x[g][i] * H[i][j] ), H = H4 (x) H4 (x) H4 (x) H4 / 16,
//      H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]] (regular Hadamard, symmetric), the sum taken EXACTLY and rounded
//      once to bf16 (RNE). Computed as a radix-4 fast transform in 64-bit fixed point (rotate_group256_fixed, the device
//      arithmetic); rotate_group256 (fp64) is the independent oracle of the tests.
//   2. row scale: amax = max |xr| (bf16, exact); s = fp32(amax) / 127.0f (IEEE division), s = max(s, 1e-30f);
//      the quantizer divides by sb = bf16(s) (RNE; 0 -> bf16 smallest normal), the epilogue uses the fp32 s.
//   3. codes: q = clamp(rint_even( bf16( fp32(xr) / fp32(sb) ) ), -128, 127) (IEEE fp32 division rounded to bf16 RNE, then
//      round half to even to an integer; -128 is reachable).
//   4. acc[m][n] = sum_k q[m][k] * W[n][k] in int32 (exact; |acc| <= 14336 * 128 * 128 < 2^31).
//   5. y = bf16( fp32(acc) * (s[m] * ws[n]) ) (fp32(acc) RNE, the scale product first, both fp32 RNE); with a bias,
//      y = bf16( fp32(y) + fp32(bias_bf16[n]) ).
// SwiGLU (the fc2 input of the MLP, on fc1's [gate | up] halves): h = bf16( fp32(bf16(silu(g))) * fp32(u) ), with
//   silu(g) = bf16( g / (1 + expf(-g)) ), expf correctly rounded: a 65536-entry table over the bf16 inputs.
// Every value of a row depends only on that row: batch invariant by construction.

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define H3_HD __host__ __device__
#else
#define H3_HD
#endif

namespace strix::h3::gemm {

H3_HD inline float bf16_to_f32(std::uint16_t b) { return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16U); }
H3_HD inline std::uint16_t f32_to_bf16(float f) {  // RNE; NaN kept quiet
    std::uint32_t u = std::bit_cast<std::uint32_t>(f);
    if ((u & 0x7fffffffU) > 0x7f800000U) return static_cast<std::uint16_t>((u >> 16U) | 0x40U);
    u += 0x7fffU + ((u >> 16U) & 1U);
    return static_cast<std::uint16_t>(u >> 16U);
}
// fp64 -> bf16 rounded once (RNE), finite normal-range inputs (no fp32 double rounding).
inline std::uint16_t f64_to_bf16(double d) {
    std::uint64_t u;
    std::memcpy(&u, &d, 8);
    const std::uint64_t sign = u >> 63U, mag = u & 0x7fffffffffffffffULL;
    if (mag == 0U) return static_cast<std::uint16_t>(sign << 15U);
    if (std::fabs(d) < 0x1p-126) {  // bf16 subnormal: a multiple of 2^-133, RNE (exact in fp64)
        const double r = std::nearbyint(std::fabs(d) * 0x1p133);
        return static_cast<std::uint16_t>((sign << 15U) | static_cast<std::uint16_t>(r));  // r == 128 is the smallest normal
    }
    // keep 8 significant bits: drop 45 mantissa bits with RNE (carry into the exponent is the right result)
    const std::uint64_t r = mag + ((1ULL << 44U) - 1U) + ((mag >> 45U) & 1U);
    const std::uint64_t t = r >> 45U << 45U;
    double v;
    std::memcpy(&v, &t, 8);
    const float f = static_cast<float>(v);  // exact: 8 significant bits, in fp32 range for activations
    std::uint32_t fb;
    std::memcpy(&fb, &f, 4);
    return static_cast<std::uint16_t>((fb >> 16U) | (sign << 15U));
}

// Rotation of one 256-group (bf16 bits in, bf16 bits out). Returns false when the group's nonzero magnitudes span more
// than 37 binades (fp64 sums may then round; the tests use it only inside that span).
inline bool rotate_group256(const std::uint16_t* in, std::uint16_t* out) {
    double v[256];
    int emax = -100000, emin = 100000;
    for (unsigned i = 0; i < 256U; ++i) {
        v[i] = static_cast<double>(bf16_to_f32(in[i]));
        if (v[i] != 0.0) {
            int e;
            static_cast<void>(std::frexp(v[i], &e));
            emax = e > emax ? e : emax;
            emin = e < emin ? e : emin;
        }
    }
    for (unsigned stride = 1U; stride < 256U; stride *= 4U)
        for (unsigned base = 0; base < 256U; ++base) {
            if ((base / stride) % 4U) continue;
            const double a0 = v[base], a1 = v[base + stride], a2 = v[base + 2U * stride], a3 = v[base + 3U * stride];
            const double s = (a0 + a1) + (a2 + a3);
            // y_j = sum_i a_i H4[i][j]; H4[i][j] = -1 iff i + j == 3
            v[base] = s - 2.0 * a3;
            v[base + stride] = s - 2.0 * a2;
            v[base + 2U * stride] = s - 2.0 * a1;
            v[base + 3U * stride] = s - 2.0 * a0;
        }
    for (unsigned i = 0; i < 256U; ++i) out[i] = f64_to_bf16(v[i] * 0.0625);
    return emax == -100000 || emax - emin <= 37;
}

// The same rotation in 64-bit fixed point (the arithmetic of the device quantizer, exact in integers). Every bf16 value
// is sig * 2^(ee - 134) (sig = 128 + mantissa for normals, the mantissa for subnormals, ee = max(exponent field, 1)); with
// the group's largest ee = emax, values enter as sig << (ee - emax + kFixShift) (|v| < 2^55, sums of 256 < 2^63) and
// leave as y * 2^(emax - kFixShift - 138) (the 1/16 folded into the exponent), rounded once to bf16 (RNE, subnormals
// at bf16 precision). Values more than kFixShift binades below the group maximum are flushed to zero first (reported:
// the rotation is then exact within 2^-47 of the group max; real late-block SwiGLU outputs do this). Equal to
// rotate_group256 whenever neither flags.
constexpr int kFixShift = 47;
H3_HD inline int msb64(std::uint64_t a) {  // a != 0
#if defined(__HIP_DEVICE_COMPILE__)
    return 63 - __clzll(static_cast<long long>(a));
#else
    return 63 - std::countl_zero(a);
#endif
}
struct Bf16Parts { std::int64_t sig; int ee; };  // sig signed
H3_HD inline Bf16Parts bf16_parts(std::uint16_t b) {
    const int e = (b >> 7U) & 0xff;
    const std::int64_t m = b & 0x7f;
    const std::int64_t sig = e ? 128 + m : m;
    return {(b & 0x8000U) ? -sig : sig, e ? e : 1};
}
// bf16 bits of y * 2^x (RNE), y != 0 or 0; finite results (|y * 2^x| < 2^128) only.
H3_HD inline std::uint16_t fixed_to_bf16(std::int64_t y, int x) {
    if (y == 0) return 0;
    const std::uint16_t sign = y < 0 ? 0x8000U : 0U;
    std::uint64_t a = y < 0 ? 0ULL - static_cast<std::uint64_t>(y) : static_cast<std::uint64_t>(y);
    const int p = msb64(a);  // value in [2^(p + x), 2^(p + x + 1))
    // lsb exponent of the bf16 result: 7 bits below the msb, at least the subnormal lsb 2^-133
    int lsb = p + x - 7;
    if (lsb < -133) lsb = -133;
    const int drop = lsb - x;  // bits of a below the result's lsb
    std::uint64_t r;
    if (drop <= 0) r = a << (-drop);
    else if (drop >= 64) r = 0;  // cannot happen for finite group sums (p <= 62)
    else {
        const std::uint64_t half = 1ULL << (drop - 1), rem = a & ((1ULL << drop) - 1ULL);
        r = a >> drop;
        if (rem > half || (rem == half && (r & 1ULL))) ++r;
    }
    // r * 2^lsb, r <= 256 (a carry to 256 bumps the exponent)
    if (r == 0) return sign;
    // r <= 256 (8 kept bits, or the subnormal grid); the RNE carry 255.5 -> 256 moves one binade up: 256 * 2^lsb = 128 * 2^(lsb + 1)
    if (r >> 8U) { r >>= 1U; ++lsb; }
    const int q = msb64(r);  // r in [2^q, 2^(q+1)), q <= 7
    const int exp_unb = lsb + q;  // value exponent
    if (exp_unb > 127) return static_cast<std::uint16_t>(sign | 0x7f80U);  // beyond bf16: +-inf (RNE overflow)
    std::uint32_t bits;
    if (exp_unb < -126) bits = static_cast<std::uint32_t>(r);  // subnormal: lsb == -133, r < 128
    else bits = (static_cast<std::uint32_t>(exp_unb + 127) << 7U) | static_cast<std::uint32_t>((r << (7 - q)) & 0x7fU);
    return static_cast<std::uint16_t>(sign | bits);
}
inline bool rotate_group256_fixed(const std::uint16_t* in, std::uint16_t* out) {
    int emax = 1;
    bool any = false;
    for (unsigned i = 0; i < 256U; ++i)
        if (in[i] & 0x7fffU) { const int e = bf16_parts(in[i]).ee; emax = any && emax > e ? emax : e; any = true; }
    std::int64_t v[256];
    bool flushed = false;
    for (unsigned i = 0; i < 256U; ++i) {
        const Bf16Parts p = bf16_parts(in[i]);
        const int sh = p.ee - emax + kFixShift;
        if (sh < 0) { flushed |= p.sig != 0; v[i] = 0; }
        else v[i] = p.sig * (std::int64_t{1} << sh);
    }
    for (unsigned stride = 1U; stride < 256U; stride *= 4U)
        for (unsigned base = 0; base < 256U; ++base) {
            if ((base / stride) % 4U) continue;
            const std::int64_t a0 = v[base], a1 = v[base + stride], a2 = v[base + 2U * stride], a3 = v[base + 3U * stride];
            const std::int64_t s = (a0 + a1) + (a2 + a3);
            v[base] = s - 2 * a3;
            v[base + stride] = s - 2 * a2;
            v[base + 2U * stride] = s - 2 * a1;
            v[base + 3U * stride] = s - 2 * a0;
        }
    for (unsigned i = 0; i < 256U; ++i) out[i] = fixed_to_bf16(v[i], emax - kFixShift - 138);
    return !flushed;
}

struct RowQuant {
    float scale = 0.0F;     // fp32 row scale (epilogue)
    unsigned inexact = 0;   // groups beyond the exact-rotation span
};

// Steps 2-3 on an already rotated row.
inline float quantize_rotated_row(const std::uint16_t* xr, std::uint32_t k, std::int8_t* q) {
    float amax = 0.0F;
    for (std::uint32_t i = 0; i < k; ++i) amax = std::fmax(amax, std::fabs(bf16_to_f32(xr[i])));
    float s = amax / 127.0F;
    if (!(s >= 1e-30F)) s = 1e-30F;
    float sb = bf16_to_f32(f32_to_bf16(s));
    if (sb == 0.0F) sb = 0x1p-126F;
    for (std::uint32_t i = 0; i < k; ++i) {
        const float t = bf16_to_f32(f32_to_bf16(bf16_to_f32(xr[i]) / sb));
        float r = std::nearbyint(t);
        r = r < -128.0F ? -128.0F : r > 127.0F ? 127.0F : r;
        q[i] = static_cast<std::int8_t>(r);
    }
    return s;
}

// Steps 1-3 on one row of k features (k % 256 == 0). xr_out (optional) receives the rotated bf16 row.
inline RowQuant rotate_quantize_row(const std::uint16_t* x, std::uint32_t k, std::int8_t* q, std::uint16_t* xr_out = nullptr) {
    std::vector<std::uint16_t> xr(k);
    RowQuant rq;
    for (std::uint32_t g = 0; g < k / 256U; ++g) rq.inexact += rotate_group256_fixed(x + 256U * g, xr.data() + 256U * g) ? 0U : 1U;
    rq.scale = quantize_rotated_row(xr.data(), k, q);
    if (xr_out) std::memcpy(xr_out, xr.data(), 2U * k);
    return rq;
}

inline std::int32_t dot_i8(const std::int8_t* a, const std::int8_t* b, std::uint32_t k) {
    std::int32_t s = 0;
    for (std::uint32_t i = 0; i < k; ++i) s += static_cast<std::int32_t>(a[i]) * static_cast<std::int32_t>(b[i]);
    return s;
}

// Step 5 for one element.
H3_HD inline std::uint16_t epilogue(std::int32_t acc, float row_scale, float channel_scale) {
    const float scale = row_scale * channel_scale;
    return f32_to_bf16(static_cast<float>(acc) * scale);
}
inline std::uint16_t add_bias(std::uint16_t y, std::uint16_t bias) { return f32_to_bf16(bf16_to_f32(y) + bf16_to_f32(bias)); }

// silu over every bf16 input (non-finite inputs map to 0: never produced by a finite projection).
inline const std::array<std::uint16_t, 65536>& silu_table() {
    static const std::array<std::uint16_t, 65536> table = [] {
        std::array<std::uint16_t, 65536> t{};
        for (std::uint32_t b = 0; b < 65536U; ++b) {
            const float x = bf16_to_f32(static_cast<std::uint16_t>(b));
            if (!std::isfinite(x)) { t[b] = 0; continue; }
            const float e = static_cast<float>(std::exp(-static_cast<double>(x)));  // correctly rounded expf
            t[b] = f32_to_bf16(x / (1.0F + e));
        }
        return t;
    }();
    return table;
}
inline std::uint16_t swiglu(std::uint16_t gate, std::uint16_t up) {
    return f32_to_bf16(bf16_to_f32(silu_table()[gate]) * bf16_to_f32(up));
}

// Whole projection (host reference): x [m][k] bf16 bits, w [n][k] int8, ws [n] fp32 -> y [m][n] bf16 bits.
// Optional outputs: codes [m][k], row scales [m], accumulators [m][n].
struct Projection {
    std::vector<std::int8_t> q;
    std::vector<float> row_scale;
    std::vector<std::int32_t> acc;
    std::vector<std::uint16_t> y;
    unsigned inexact_groups = 0;
};
inline Projection project(const std::uint16_t* x, std::uint32_t m, std::uint32_t k, const std::int8_t* w, const float* ws,
    std::uint32_t n, const std::uint16_t* bias = nullptr) {
    Projection p;
    p.q.resize(static_cast<std::size_t>(m) * k);
    p.row_scale.resize(m);
    p.acc.resize(static_cast<std::size_t>(m) * n);
    p.y.resize(static_cast<std::size_t>(m) * n);
    for (std::uint32_t r = 0; r < m; ++r) {
        const RowQuant rq = rotate_quantize_row(x + static_cast<std::size_t>(r) * k, k, p.q.data() + static_cast<std::size_t>(r) * k);
        p.row_scale[r] = rq.scale;
        p.inexact_groups += rq.inexact;
    }
    for (std::uint32_t r = 0; r < m; ++r)
        for (std::uint32_t c = 0; c < n; ++c) {
            const std::int32_t a = dot_i8(p.q.data() + static_cast<std::size_t>(r) * k, w + static_cast<std::size_t>(c) * k, k);
            p.acc[static_cast<std::size_t>(r) * n + c] = a;
            std::uint16_t y = epilogue(a, p.row_scale[r], ws[c]);
            if (bias) y = add_bias(y, bias[c]);
            p.y[static_cast<std::size_t>(r) * n + c] = y;
        }
    return p;
}

// fc1 + SwiGLU (host reference of linear_fc1_swiglu): w [n][k] in checkpoint order ([gate | up] halves), h [m][n / 2].
inline std::vector<std::uint16_t> project_fc1_swiglu(const std::uint16_t* x, std::uint32_t m, std::uint32_t k, const std::int8_t* w,
    const float* ws, std::uint32_t n) {
    const Projection p = project(x, m, k, w, ws, n);
    const std::uint32_t half = n / 2U;
    std::vector<std::uint16_t> h(static_cast<std::size_t>(m) * half);
    for (std::uint32_t r = 0; r < m; ++r)
        for (std::uint32_t j = 0; j < half; ++j)
            h[static_cast<std::size_t>(r) * half + j] = swiglu(p.y[static_cast<std::size_t>(r) * n + j], p.y[static_cast<std::size_t>(r) * n + half + j]);
    return h;
}

}  // namespace strix::h3::gemm
