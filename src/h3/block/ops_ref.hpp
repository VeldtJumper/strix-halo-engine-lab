#pragma once
// MiniMax H3 block glue: host reference arithmetic (plain C++, the bit-level spec of the K3/K4/K5/K6 kernels).
// Every rule below was checked bitwise against the reference capture (h3fx fixtures):
//
//  - RMSNorm (bf16 in, bf16 weight, bf16 out): ss = ref_sum_f32(x_i^2) (fp32 products, the reduction order of the
//    reference CPU library: 8-lane vectors x 4-way ILP with a 16-step cascade, then the lanes in order);
//    r = 1 / sqrt(ss / n + eps) (fp32, IEEE); y = bf16((x * r) * w) in fp32.
//  - modulation (per segment, mod row m): h = bf16(bf16(h * s1[m]) + shift[m]) with s1 = bf16(1 + bf16(scale[m])),
//    shift = bf16(shift_f32[m]) (the adaLN rows are fp32 and cast to bf16 at use).
//  - gated residual: x = bf16(x + o * bf16(gate[m])) (fp32 product rounded, fp32 add, one bf16 rounding).
//  - q/k: per-head RMSNorm (n = 128) then split-half RoPE on dims [0, 96): for i < 48 with a = y[i], b = y[48 + i],
//    c = bf16(cos), s = bf16(sin):  y'[i] = bf16(bf16(a c) - b s),  y'[48 + i] = bf16(bf16(b c) + a s)  (the products
//    a c, b c rounded to bf16 first, the cross terms fp32, one fp32 add, one bf16 rounding); dims [96, 128) unrotated.
//  - final layer: f = (fp32(norm(x)) * (1 + scale_f32)) + shift_f32 (fp32), heads in fp32.
//  - fp32 linears (adaLN, heads): fp64 dot + fp64 bias, one rounding to fp32 (the reference library order is not
//    reproducible; this differs from it in the last fp32 ulp); patch projections: fp32 k-ascending sum.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "h3/block/weights.hpp"
#include "h3/gemm/ref.hpp"

namespace strix::h3::block {

inline float bf(std::uint16_t b) { return gemm::bf16_to_f32(b); }
inline std::uint16_t tob(float f) { return gemm::f32_to_bf16(f); }

inline unsigned ceil_log2_u64(std::uint64_t x) {
    unsigned r = 0;
    while ((1ULL << r) < x) ++r;
    return r;
}

// fp32 sum in the reference library's order (inner contiguous reduction, 8-lane vectors, ILP 4, 4-level cascade).
inline float ref_sum_f32(const float* v, std::uint32_t n) {
    constexpr unsigned VW = 8, ILP = 4, LV = 4;
    const std::uint32_t nvec = n / VW;
    const std::uint32_t size = nvec / ILP;
    unsigned lp = ceil_log2_u64(size) / LV;
    if (lp < 4) lp = 4;
    const std::uint64_t step = 1ULL << lp, mask = step - 1;
    float acc[LV][ILP][VW] = {};
    std::uint64_t i = 0;
    auto add_row = [&](std::uint64_t row) {
        for (unsigned k = 0; k < ILP; ++k)
            for (unsigned l = 0; l < VW; ++l) acc[0][k][l] += v[(row * ILP + k) * VW + l];
    };
    while (i + step <= size) {
        for (std::uint64_t j = 0; j < step; ++j, ++i) add_row(i);
        for (unsigned j = 1; j < LV; ++j) {
            for (unsigned k = 0; k < ILP; ++k)
                for (unsigned l = 0; l < VW; ++l) {
                    acc[j][k][l] += acc[j - 1][k][l];
                    acc[j - 1][k][l] = 0.0F;
                }
            if ((i & (mask << (j * lp))) != 0) break;
        }
    }
    for (; i < size; ++i) add_row(i);
    for (unsigned j = 1; j < LV; ++j)
        for (unsigned k = 0; k < ILP; ++k)
            for (unsigned l = 0; l < VW; ++l) acc[0][k][l] += acc[j][k][l];
    for (std::uint32_t r = size * ILP; r < nvec; ++r)
        for (unsigned l = 0; l < VW; ++l) acc[0][0][l] += v[r * VW + l];
    for (unsigned k = 1; k < ILP; ++k)
        for (unsigned l = 0; l < VW; ++l) acc[0][0][l] += acc[0][k][l];
    float fin = 0.0F;
    for (std::uint32_t r = nvec * VW; r < n; ++r) fin += v[r];
    for (unsigned l = 0; l < VW; ++l) fin += acc[0][0][l];
    return fin;
}

// Inverse RMS of a bf16 row (fp32).
inline float rms_inv_bf16(const std::uint16_t* x, std::uint32_t n, float eps) {
    float sq[kFfn * 2];  // n <= 28672
    for (std::uint32_t i = 0; i < n; ++i) {
        const float f = bf(x[i]);
        sq[i] = f * f;
    }
    const float ms = ref_sum_f32(sq, n) / static_cast<float>(n);
    return 1.0F / std::sqrt(ms + eps);
}

inline void rmsnorm_row(const std::uint16_t* x, const std::uint16_t* w, std::uint32_t n, float eps, std::uint16_t* y) {
    const float r = rms_inv_bf16(x, n, eps);
    for (std::uint32_t i = 0; i < n; ++i) y[i] = tob((bf(x[i]) * r) * bf(w[i]));
}

// --------------------------------------------------------------------------------------------- adaLN modulation
// Per-block modulation vectors at the precision they are consumed, for every mod row (t_index * 3 + tag).
struct ModVectors {
    std::uint32_t rows = 0;
    // bf16 [rows][5376]: shift, s1 = bf16(1 + bf16(scale)), gate for the attention (msa) and MLP halves.
    std::vector<std::uint16_t> shift_msa, s1_msa, gate_msa, shift_mlp, s1_mlp, gate_mlp;
};

// fp32 linear y[m][n] = fp32(fp64 dot(x[m], w[n]) + fp64 b[n]).
inline std::vector<float> f32_linear(const F32Linear& l, const float* x, std::uint32_t m) {
    std::vector<float> y(static_cast<std::size_t>(m) * l.n);
    for (std::uint32_t r = 0; r < m; ++r)
        for (std::uint32_t c = 0; c < l.n; ++c) {
            double s = 0.0;
            const float* w = l.w.data() + static_cast<std::size_t>(c) * l.k;
            const float* xr = x + static_cast<std::size_t>(r) * l.k;
            for (std::uint32_t k = 0; k < l.k; ++k) s += static_cast<double>(xr[k]) * static_cast<double>(w[k]);
            y[static_cast<std::size_t>(r) * l.n + c] = static_cast<float>(s + static_cast<double>(l.b[c]));
        }
    return y;
}

// Patch projections: y[c] = fp32((sum_k x[k] w[c][k], fp32, k ascending) + b[c]) (matches the capture's library order
// on most rows; with bf16-valued inputs and weights the products are exact).
inline std::vector<float> patch_linear_row(const F32Linear& l, const float* x) {
    std::vector<float> y(l.n);
    for (std::uint32_t c = 0; c < l.n; ++c) {
        const float* w = l.w.data() + static_cast<std::size_t>(c) * l.k;
        float s = 0.0F;
        for (std::uint32_t k = 0; k < l.k; ++k) s = s + x[k] * w[k];
        y[c] = s + l.b[c];
    }
    return y;
}

// adaLN of one block: t_emb [n_t][basis] -> fp32 [n_t * 3][6 * 5376] (row = t_index * 3 + tag; chunk j of a row =
// columns [j * 5376, (j + 1) * 5376): shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp).
inline std::vector<float> adaln_block_f32(const BlockWeights& b, const std::vector<float>& t_emb, std::uint32_t n_t) {
    return f32_linear(b.adaln, t_emb.data(), n_t);  // [n_t][3 * 6 * 5376] is the same memory as [n_t * 3][6 * 5376]
}

inline ModVectors mod_vectors(const std::vector<float>& ada, std::uint32_t rows) {
    ModVectors mv;
    mv.rows = rows;
    const std::size_t n = static_cast<std::size_t>(rows) * kHidden;
    for (auto* v : {&mv.shift_msa, &mv.s1_msa, &mv.gate_msa, &mv.shift_mlp, &mv.s1_mlp, &mv.gate_mlp}) v->resize(n);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t c = 0; c < kHidden; ++c) {
            const float* row = ada.data() + static_cast<std::size_t>(r) * 6 * kHidden;
            const std::size_t o = static_cast<std::size_t>(r) * kHidden + c;
            mv.shift_msa[o] = tob(row[0 * kHidden + c]);
            mv.s1_msa[o] = tob(1.0F + bf(tob(row[1 * kHidden + c])));
            mv.gate_msa[o] = tob(row[2 * kHidden + c]);
            mv.shift_mlp[o] = tob(row[3 * kHidden + c]);
            mv.s1_mlp[o] = tob(1.0F + bf(tob(row[4 * kHidden + c])));
            mv.gate_mlp[o] = tob(row[5 * kHidden + c]);
        }
    return mv;
}

inline void mod_scale_shift_row(std::uint16_t* h, const std::uint16_t* s1, const std::uint16_t* shift, std::uint32_t n) {
    for (std::uint32_t i = 0; i < n; ++i) h[i] = tob(bf(tob(bf(h[i]) * bf(s1[i]))) + bf(shift[i]));
}
inline void gate_row(std::uint16_t* x, const std::uint16_t* o, const std::uint16_t* g, std::uint32_t n) {
    for (std::uint32_t i = 0; i < n; ++i) x[i] = tob(bf(x[i]) + bf(o[i]) * bf(g[i]));
}

// --------------------------------------------------------------------------------------------- RoPE
// Rotation table [S][48] (c, s) as bf16 bits, from fp64 positions [S][3]: angle = fp32(pos) * inv_freq (fp32),
// c = bf16(fp32(cos)), s = bf16(fp32(sin)) (cos/sin evaluated in fp64 and rounded to fp32).
struct RopeTable {
    std::uint32_t seq = 0;
    std::vector<std::uint16_t> c, s;  // [S][48]
};
inline std::vector<float> rope_angles(const std::vector<double>& pos, const std::vector<float>& inv_freq) {
    const std::size_t S = pos.size() / 3;
    std::vector<float> a(S * kRotDim);
    for (std::size_t r = 0; r < S; ++r)
        for (unsigned ax = 0; ax < 3; ++ax) {
            const float p = static_cast<float>(pos[r * 3 + ax]);
            for (unsigned f = 0; f < kRopeFreqs; ++f) {
                const float v = p * inv_freq[f];
                a[r * kRotDim + ax * kRopeFreqs + f] = v;
                a[r * kRotDim + 48 + ax * kRopeFreqs + f] = v;
            }
        }
    return a;
}
inline RopeTable rope_table(const std::vector<float>& angles) {
    RopeTable t;
    t.seq = static_cast<std::uint32_t>(angles.size() / kRotDim);
    t.c.resize(static_cast<std::size_t>(t.seq) * 48);
    t.s.resize(t.c.size());
    for (std::size_t r = 0; r < t.seq; ++r)
        for (unsigned i = 0; i < 48; ++i) {
            const double a = static_cast<double>(angles[r * kRotDim + i]);
            t.c[r * 48 + i] = tob(static_cast<float>(std::cos(a)));
            t.s[r * 48 + i] = tob(static_cast<float>(std::sin(a)));
        }
    return t;
}

// Static bound of the attention input contract (attn package: ||q|| ||k|| log2(e) / sqrt(128) <= 2^14 per row, key,
// head). After the per-head RMSNorm sum((x r)^2) = 128 ms / (ms + eps) <= 128, so ||y|| <= max|w| sqrt(128); the bf16
// roundings (normalised value, the RoPE products and sums, c and s) grow it by less than (1 + 2^-7)^4. Returns the bound
// of the scaled score ||q|| ||k|| log2(e) / sqrt(128) for a block's q / k norm weights. It is attained only by a one-hot
// head input, so it cannot prove the contract for every block (block 49 of the real weights: 2.17e4 > 2^14); the hard
// check is the measured one (attn_score_from_sumsq on K4's per-head maxima, DeviceForward).
inline double qk_score_bound(const std::vector<std::uint16_t>& qw, const std::vector<std::uint16_t>& kw) {
    double mq = 0, mk = 0;
    for (auto v : qw) mq = std::max(mq, static_cast<double>(std::fabs(bf(v))));
    for (auto v : kw) mk = std::max(mk, static_cast<double>(std::fabs(bf(v))));
    const double grow = std::pow(1.0 + 1.0 / 128.0, 4.0);
    return (mq * std::sqrt(128.0) * grow) * (mk * std::sqrt(128.0) * grow) * (1.4426950408889634 / std::sqrt(128.0));
}
inline constexpr double kAttnScoreLimit = 16384.0;  // 2^14
// scaled score bound of one head from the max squared norms of its q and k rows
inline double attn_score_from_sumsq(float q_ss, float k_ss) {
    return std::sqrt(static_cast<double>(q_ss)) * std::sqrt(static_cast<double>(k_ss)) * (1.4426950408889634 / std::sqrt(128.0)) *
           (1.0 + 1e-5);  // margin for the fp32 sum of squares
}

// One head vector: RMSNorm (n = 128, weight w) then split-half RoPE with the row's (c, s) [48].
inline void qk_norm_rope_head(const std::uint16_t* x, const std::uint16_t* w, const std::uint16_t* c, const std::uint16_t* s,
    std::uint16_t* y) {
    std::uint16_t xn[kHeadDim];
    rmsnorm_row(x, w, kHeadDim, kNormEps, xn);
    for (unsigned i = 0; i < 48; ++i) {
        const float a = bf(xn[i]), b = bf(xn[48 + i]), cc = bf(c[i]), ss = bf(s[i]);
        y[i] = tob(bf(tob(a * cc)) + b * (-ss));
        y[48 + i] = tob(bf(tob(b * cc)) + a * ss);
    }
    for (unsigned i = kRotDim; i < kHeadDim; ++i) y[i] = xn[i];
}

// qkv [S][21504] -> Q, K token-major [S][56][128] (normed + rotated), V^T [56][128][S] (raw). rope == nullptr: no
// rotation (refiner).
inline void qk_prepare(const std::uint16_t* qkv, std::uint32_t S, const std::uint16_t* qw, const std::uint16_t* kw,
    const RopeTable* rope, std::uint16_t* q, std::uint16_t* k, std::uint16_t* vt) {
    for (std::uint32_t r = 0; r < S; ++r) {
        const std::uint16_t* row = qkv + static_cast<std::size_t>(r) * 3 * kInner;
        for (std::uint32_t h = 0; h < kHeads; ++h) {
            const std::size_t o = (static_cast<std::size_t>(r) * kHeads + h) * kHeadDim;
            if (rope) {
                qk_norm_rope_head(row + h * kHeadDim, qw, rope->c.data() + r * 48, rope->s.data() + r * 48, q + o);
                qk_norm_rope_head(row + kInner + h * kHeadDim, kw, rope->c.data() + r * 48, rope->s.data() + r * 48, k + o);
            } else {
                rmsnorm_row(row + h * kHeadDim, qw, kHeadDim, kNormEps, q + o);
                rmsnorm_row(row + kInner + h * kHeadDim, kw, kHeadDim, kNormEps, k + o);
            }
            for (std::uint32_t d = 0; d < kHeadDim; ++d)
                vt[(static_cast<std::size_t>(h) * kHeadDim + d) * S + r] = row[2 * kInner + h * kHeadDim + d];
        }
    }
}

// --------------------------------------------------------------------------------------------- time embedding
// sigma_v as the model sees it: timestep = fp32(sigma * 1000), sigma_v = max(timestep / 1000, 1e-6) (fp32).
inline float model_sigma(float sigma) {
    const float ts = sigma * 1000.0F;
    float s = ts / 1000.0F;
    return s < 1e-6F ? 1e-6F : s;
}
// time_shift_sigma in fp32 (the reference evaluates it on fp32 tensors with exact double constants).
inline float time_shift_sigma_f32(float s, float from, float to) {
    const float base = s / (from + s * (1.0F - from));
    return to * base / (1.0F + (to - 1.0F) * base);
}
// torch-style fp32 lerp (vectorised form: fma with the |w| < 0.5 split).
inline float lerp_f32(float a, float b, float w) {
    const float d = b - a;
    return std::fabs(w) < 0.5F ? std::fma(w, d, a) : std::fma(w - 1.0F, d, b);
}
// Curve rows for the given timesteps (fp32 values): [n][basis].
inline std::vector<float> curve_t_emb(const GlobalWeights& g, const std::vector<float>& t_vals) {
    std::vector<float> e(t_vals.size() * g.basis);
    const float last = static_cast<float>(g.curve_grid - 1);
    for (std::size_t i = 0; i < t_vals.size(); ++i) {
        float t = t_vals[i];
        t = t < 0.0F ? 0.0F : (t > 1.0F ? 1.0F : t);
        const float pos = t * last;
        std::int64_t i0 = static_cast<std::int64_t>(std::floor(pos));
        if (i0 > static_cast<std::int64_t>(g.curve_grid) - 2) i0 = static_cast<std::int64_t>(g.curve_grid) - 2;
        const float w = pos - static_cast<float>(i0);
        for (std::uint32_t j = 0; j < g.basis; ++j)
            e[i * g.basis + j] = lerp_f32(g.adaln_t_table[static_cast<std::size_t>(i0) * g.basis + j],
                                          g.adaln_t_table[static_cast<std::size_t>(i0 + 1) * g.basis + j], w);
    }
    return e;
}

// --------------------------------------------------------------------------------------------- patches
// video [24][T][H][W] fp32 (H, W even) -> rows [T*(H/2)*(W/2)][96] with column c*4 + p*2 + q.
inline std::vector<float> patchify_video(const float* v, std::uint32_t T, std::uint32_t H, std::uint32_t W) {
    const std::uint32_t h = H / 2, w = W / 2;
    std::vector<float> r(static_cast<std::size_t>(T) * h * w * kVideoPatch);
    for (std::uint32_t t = 0; t < T; ++t)
        for (std::uint32_t y = 0; y < h; ++y)
            for (std::uint32_t x = 0; x < w; ++x)
                for (std::uint32_t c = 0; c < 24; ++c)
                    for (std::uint32_t p = 0; p < 2; ++p)
                        for (std::uint32_t q = 0; q < 2; ++q)
                            r[((static_cast<std::size_t>(t) * h + y) * w + x) * kVideoPatch + c * 4 + p * 2 + q] =
                                v[((static_cast<std::size_t>(c) * T + t) * H + 2 * y + p) * W + 2 * x + q];
    return r;
}
inline std::vector<float> unpatchify_video(const float* r, std::uint32_t T, std::uint32_t H, std::uint32_t W) {
    const std::uint32_t h = H / 2, w = W / 2;
    std::vector<float> v(static_cast<std::size_t>(24) * T * H * W);
    for (std::uint32_t t = 0; t < T; ++t)
        for (std::uint32_t y = 0; y < h; ++y)
            for (std::uint32_t x = 0; x < w; ++x)
                for (std::uint32_t c = 0; c < 24; ++c)
                    for (std::uint32_t p = 0; p < 2; ++p)
                        for (std::uint32_t q = 0; q < 2; ++q)
                            v[((static_cast<std::size_t>(c) * T + t) * H + 2 * y + p) * W + 2 * x + q] =
                                r[((static_cast<std::size_t>(t) * h + y) * w + x) * kVideoPatch + c * 4 + p * 2 + q];
    return v;
}
// audio [32][2][T] -> rows [2T][32] channel-major (ch0 t0..T-1, ch1 t0..T-1).
inline std::vector<float> pack_audio(const float* a, std::uint32_t T) {
    std::vector<float> r(static_cast<std::size_t>(2) * T * kAudioDim);
    for (std::uint32_t ch = 0; ch < 2; ++ch)
        for (std::uint32_t t = 0; t < T; ++t)
            for (std::uint32_t c = 0; c < kAudioDim; ++c)
                r[(static_cast<std::size_t>(ch) * T + t) * kAudioDim + c] = a[(static_cast<std::size_t>(c) * 2 + ch) * T + t];
    return r;
}
inline std::vector<float> unpack_audio(const float* r, std::uint32_t T) {
    std::vector<float> a(static_cast<std::size_t>(2) * T * kAudioDim);
    for (std::uint32_t ch = 0; ch < 2; ++ch)
        for (std::uint32_t t = 0; t < T; ++t)
            for (std::uint32_t c = 0; c < kAudioDim; ++c)
                a[(static_cast<std::size_t>(c) * 2 + ch) * T + t] = r[(static_cast<std::size_t>(ch) * T + t) * kAudioDim + c];
    return a;
}

// --------------------------------------------------------------------------------------------- attention (host)
// out[r][h*128 + d] = bf16( sum_j softmax_j(q_r . k_j / sqrt(128)) v_j[d] ), fp64 throughout. Q, K [S][56][128],
// V^T [56][128][keys]; queries rows [0, S), keys rows [0, keys).
void attention_host_f64(const std::uint16_t* q, const std::uint16_t* k, const std::uint16_t* vt, std::uint32_t S,
    std::uint32_t keys, std::uint16_t* out);

}  // namespace strix::h3::block
