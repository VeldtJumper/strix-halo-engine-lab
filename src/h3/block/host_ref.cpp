// MiniMax H3 DiT host reference forward (see host_ref.hpp / ops_ref.hpp for the arithmetic contract).
#include "h3/block/host_ref.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "h3/block/parallel.hpp"
#include "h3/gemm/ref.hpp"

#if defined(__AVX2__) && !defined(__HIP_DEVICE_COMPILE__)
#include <immintrin.h>
#endif

namespace strix::h3::block {

// --------------------------------------------------------------------------------------------- linears
namespace {

std::int32_t dot_i8_fast(const std::int8_t* a, const std::int8_t* b, std::uint32_t k) {
#if defined(__AVX2__) && !defined(__HIP_DEVICE_COMPILE__)
    __m256i acc = _mm256_setzero_si256();
    std::uint32_t i = 0;
    for (; i + 32 <= k; i += 32) {
        const __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
        const __m128i a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i + 16));
        const __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
        const __m128i b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i + 16));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_cvtepi8_epi16(a0), _mm256_cvtepi8_epi16(b0)));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_cvtepi8_epi16(a1), _mm256_cvtepi8_epi16(b1)));
    }
    alignas(32) std::int32_t lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc);
    std::int32_t s = 0;
    for (int l = 0; l < 8; ++l) s += lanes[l];
    for (; i < k; ++i) s += static_cast<std::int32_t>(a[i]) * static_cast<std::int32_t>(b[i]);
    return s;
#else
    return gemm::dot_i8(a, b, k);
#endif
}

float dot_f32_fast(const float* a, const float* b, std::uint32_t k) {
#if defined(__AVX2__) && !defined(__HIP_DEVICE_COMPILE__)
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(), s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    std::uint32_t i = 0;
    for (; i + 32 <= k; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    alignas(32) float l[8];
    _mm256_store_ps(l, _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    double s = 0.0;
    for (int j = 0; j < 8; ++j) s += l[j];
    for (; i < k; ++i) s += static_cast<double>(a[i]) * b[i];
    return static_cast<float>(s);
#else
    double s = 0.0;
    for (std::uint32_t i = 0; i < k; ++i) s += static_cast<double>(a[i]) * b[i];
    return static_cast<float>(s);
#endif
}

}  // namespace

void HostLinear::run(const QLinear& w, const std::uint16_t* x, std::uint32_t m, std::uint16_t* y) {
    const std::uint32_t k = w.k, n = w.n;
    std::vector<std::int8_t> q(static_cast<std::size_t>(m) * k);
    std::vector<float> rs(m);
    parallel_for(m, [&](std::uint64_t r) {
        const gemm::RowQuant rq = gemm::rotate_quantize_row(x + r * k, k, q.data() + r * k);
        if (rq.inexact) inexact_groups_.fetch_add(rq.inexact);
        rs[r] = rq.scale;
    });
    constexpr std::uint32_t kCB = 32;  // output channels per work item
    parallel_for((n + kCB - 1) / kCB, [&](std::uint64_t cb) {
        const std::uint32_t c0 = static_cast<std::uint32_t>(cb) * kCB, c1 = std::min(n, c0 + kCB);
        for (std::uint32_t c = c0; c < c1; ++c) {
            const std::int8_t* wr = w.w.data() + static_cast<std::size_t>(c) * k;
            for (std::uint32_t r = 0; r < m; ++r)
                y[static_cast<std::size_t>(r) * n + c] = gemm::epilogue(dot_i8_fast(q.data() + static_cast<std::size_t>(r) * k, wr, k), rs[r], w.ws[c]);
        }
    });
}

void bf16_linear_host(const Bf16Linear& l, const std::uint16_t* x, std::uint32_t m, std::uint16_t* y) {
    const std::uint32_t k = l.k, n = l.n;
    std::vector<float> xf(static_cast<std::size_t>(m) * k);
    for (std::size_t i = 0; i < xf.size(); ++i) xf[i] = bf(x[i]);
    constexpr std::uint32_t kCB = 8;
    parallel_for((n + kCB - 1) / kCB, [&](std::uint64_t cb) {
        std::vector<float> wf(k);
        const std::uint32_t c0 = static_cast<std::uint32_t>(cb) * kCB, c1 = std::min(n, c0 + kCB);
        for (std::uint32_t c = c0; c < c1; ++c) {
            const std::uint16_t* wr = l.w.data() + static_cast<std::size_t>(c) * k;
            for (std::uint32_t i = 0; i < k; ++i) wf[i] = bf(wr[i]);
            const float b = l.b.empty() ? 0.0F : bf(l.b[c]);
            std::uint32_t r = 0;
            for (; r + 8 <= m; r += 8) {  // eight independent k-ascending chains (same order as one)
                const float* x0 = xf.data() + static_cast<std::size_t>(r) * k;
                float s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                for (std::uint32_t i = 0; i < k; ++i) {
                    const float w = wf[i];
                    for (int j = 0; j < 8; ++j) s[j] = s[j] + x0[static_cast<std::size_t>(j) * k + i] * w;
                }
                for (int j = 0; j < 8; ++j) y[static_cast<std::size_t>(r + j) * n + c] = tob(s[j] + b);
            }
            for (; r < m; ++r) {
                const float* xr = xf.data() + static_cast<std::size_t>(r) * k;
                float s0 = 0.0F;
                for (std::uint32_t i = 0; i < k; ++i) s0 = s0 + xr[i] * wf[i];
                y[static_cast<std::size_t>(r) * n + c] = tob(s0 + b);
            }
        }
    });
}

void swiglu_host(const std::uint16_t* f1, std::uint32_t m, std::uint32_t ffn, std::uint16_t* out) {
    for (std::uint32_t r = 0; r < m; ++r)
        for (std::uint32_t i = 0; i < ffn; ++i)
            out[static_cast<std::size_t>(r) * ffn + i] =
                gemm::swiglu(f1[static_cast<std::size_t>(r) * 2 * ffn + i], f1[static_cast<std::size_t>(r) * 2 * ffn + ffn + i]);
}

// --------------------------------------------------------------------------------------------- attention
void attention_host_f64(const std::uint16_t* q, const std::uint16_t* k, const std::uint16_t* vt, std::uint32_t S,
    std::uint32_t keys, std::uint16_t* out) {
    // K as fp64 head-major for locality: kd[h][j][d]
    std::vector<double> kd(static_cast<std::size_t>(kHeads) * keys * kHeadDim);
    for (std::uint32_t j = 0; j < keys; ++j)
        for (std::uint32_t h = 0; h < kHeads; ++h)
            for (std::uint32_t d = 0; d < kHeadDim; ++d)
                kd[(static_cast<std::size_t>(h) * keys + j) * kHeadDim + d] = bf(k[(static_cast<std::size_t>(j) * kHeads + h) * kHeadDim + d]);
    const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
    constexpr std::uint32_t kRB = 8;
    const std::uint64_t rbs = (S + kRB - 1) / kRB;
    parallel_for(static_cast<std::uint64_t>(kHeads) * rbs, [&](std::uint64_t item) {
        const std::uint32_t h = static_cast<std::uint32_t>(item / rbs);
        const std::uint32_t r0 = static_cast<std::uint32_t>(item % rbs) * kRB, r1 = std::min(S, r0 + kRB);
        std::vector<double> p(keys);
        double qd[kHeadDim], o[kHeadDim];
        for (std::uint32_t r = r0; r < r1; ++r) {
            for (std::uint32_t d = 0; d < kHeadDim; ++d) qd[d] = bf(q[(static_cast<std::size_t>(r) * kHeads + h) * kHeadDim + d]);
            double mx = -1e300;
            for (std::uint32_t j = 0; j < keys; ++j) {
                const double* kr = kd.data() + (static_cast<std::size_t>(h) * keys + j) * kHeadDim;
                double s = 0.0;
                for (std::uint32_t d = 0; d < kHeadDim; ++d) s += qd[d] * kr[d];
                p[j] = s * scale;
                mx = std::max(mx, p[j]);
            }
            double l = 0.0;
            for (std::uint32_t j = 0; j < keys; ++j) {
                p[j] = std::exp(p[j] - mx);
                l += p[j];
            }
            for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                const std::uint16_t* vr = vt + (static_cast<std::size_t>(h) * kHeadDim + d) * keys;
                double s = 0.0;
                for (std::uint32_t j = 0; j < keys; ++j) s += p[j] * bf(vr[j]);
                o[d] = s / l;
            }
            for (std::uint32_t d = 0; d < kHeadDim; ++d)
                out[static_cast<std::size_t>(r) * kInner + h * kHeadDim + d] = gemm::f64_to_bf16(o[d]);
        }
    });
}

void HostAttention::run(const std::uint16_t* q, const std::uint16_t* k, const std::uint16_t* vt, std::uint32_t S,
    std::uint32_t keys, std::uint16_t* out) {
    attention_host_f64(q, k, vt, S, keys, out);
}

void CaptureAttention::run(const std::uint16_t* q, const std::uint16_t* k, const std::uint16_t* vt, std::uint32_t S,
    std::uint32_t keys, std::uint16_t* out) {
    std::vector<float> kf(static_cast<std::size_t>(kHeads) * keys * kHeadDim);
    for (std::uint32_t j = 0; j < keys; ++j)
        for (std::uint32_t h = 0; h < kHeads; ++h)
            for (std::uint32_t d = 0; d < kHeadDim; ++d)
                kf[(static_cast<std::size_t>(h) * keys + j) * kHeadDim + d] = bf(k[(static_cast<std::size_t>(j) * kHeads + h) * kHeadDim + d]);
    const float scale = static_cast<float>(std::pow(128.0, -0.5));
    constexpr std::uint32_t kRB = 8;
    const std::uint64_t rbs = (S + kRB - 1) / kRB;
    parallel_for(static_cast<std::uint64_t>(kHeads) * rbs, [&](std::uint64_t item) {
        const std::uint32_t h = static_cast<std::uint32_t>(item / rbs);
        const std::uint32_t r0 = static_cast<std::uint32_t>(item % rbs) * kRB, r1 = std::min(S, r0 + kRB);
        std::vector<float> p(keys);
        float qf[kHeadDim];
        for (std::uint32_t r = r0; r < r1; ++r) {
            for (std::uint32_t d = 0; d < kHeadDim; ++d) qf[d] = bf(q[(static_cast<std::size_t>(r) * kHeads + h) * kHeadDim + d]);
            float mx = -3.0e38F;
            for (std::uint32_t j = 0; j < keys; ++j) {
                const float s = dot_f32_fast(qf, kf.data() + (static_cast<std::size_t>(h) * keys + j) * kHeadDim, kHeadDim);
                p[j] = bf(tob(s * scale));
                mx = std::max(mx, p[j]);
            }
            float l = 0.0F;
            for (std::uint32_t j = 0; j < keys; ++j) {
                p[j] = static_cast<float>(std::exp(static_cast<double>(p[j] - mx)));
                l += p[j];
            }
            for (std::uint32_t j = 0; j < keys; ++j) p[j] = bf(tob(p[j] / l));
            for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                const std::uint16_t* vr = vt + (static_cast<std::size_t>(h) * kHeadDim + d) * keys;
                double s = 0.0;
                for (std::uint32_t j = 0; j < keys; ++j) s += static_cast<double>(p[j]) * bf(vr[j]);
                out[static_cast<std::size_t>(r) * kInner + h * kHeadDim + d] = tob(static_cast<float>(s));
            }
        }
    });
}

// --------------------------------------------------------------------------------------------- block
std::vector<std::uint32_t> mod_row_of(const std::vector<ModSegment>& segs, std::uint32_t S) {
    std::vector<std::uint32_t> r(S, 0xffffffffU);
    for (const auto& s : segs)
        for (std::uint32_t i = s.start; i < s.stop; ++i) r.at(i) = s.row;
    for (auto v : r)
        if (v == 0xffffffffU) throw std::runtime_error("mod segments do not cover the sequence");
    return r;
}

void block_forward_host(const BlockWeights& b, const ModVectors& mv, const std::vector<std::uint32_t>& mod_row,
    const RopeTable& rope, std::vector<std::uint16_t>& x, std::uint32_t S, LinearOp& lin, AttentionOp& att, BlockTaps* taps) {
    const std::size_t H = kHidden;
    std::vector<std::uint16_t> h(S * H);
    // norm1 + modulation
    parallel_for(S, [&](std::uint64_t r) { rmsnorm_row(x.data() + r * H, b.norm1.data(), kHidden, kNormEps, h.data() + r * H); });
    if (taps && taps->norm1_out) *taps->norm1_out = h;
    for (std::uint32_t r = 0; r < S; ++r)
        mod_scale_shift_row(h.data() + r * H, mv.s1_msa.data() + mod_row[r] * H, mv.shift_msa.data() + mod_row[r] * H, kHidden);
    if (taps && taps->attn_in) *taps->attn_in = h;
    // attention
    std::vector<std::uint16_t> qkv(static_cast<std::size_t>(S) * 3 * kInner);
    lin.run(b.qkv, h.data(), S, qkv.data());
    if (taps && taps->qkv) *taps->qkv = qkv;
    std::vector<std::uint16_t> q(static_cast<std::size_t>(S) * kInner), k(q.size()), vt(q.size());
    qk_prepare(qkv.data(), S, b.q_norm.data(), b.k_norm.data(), &rope, q.data(), k.data(), vt.data());
    qkv = {};
    if (taps && taps->q) *taps->q = q;
    if (taps && taps->k) *taps->k = k;
    if (taps && taps->vt) *taps->vt = vt;
    std::vector<std::uint16_t> ao(static_cast<std::size_t>(S) * kInner);
    att.run(q.data(), k.data(), vt.data(), S, S, ao.data());
    if (taps && taps->attn) *taps->attn = ao;
    std::vector<std::uint16_t> o(S * H);
    lin.run(b.out, ao.data(), S, o.data());
    if (taps && taps->out_proj) *taps->out_proj = o;
    for (std::uint32_t r = 0; r < S; ++r) gate_row(x.data() + r * H, o.data() + r * H, mv.gate_msa.data() + mod_row[r] * H, kHidden);
    if (taps && taps->norm2_in) *taps->norm2_in = x;
    // MLP
    parallel_for(S, [&](std::uint64_t r) { rmsnorm_row(x.data() + r * H, b.norm2.data(), kHidden, kNormEps, h.data() + r * H); });
    if (taps && taps->norm2_out) *taps->norm2_out = h;
    for (std::uint32_t r = 0; r < S; ++r)
        mod_scale_shift_row(h.data() + r * H, mv.s1_mlp.data() + mod_row[r] * H, mv.shift_mlp.data() + mod_row[r] * H, kHidden);
    if (taps && taps->mlp_in) *taps->mlp_in = h;
    std::vector<std::uint16_t> f1(static_cast<std::size_t>(S) * 2 * kFfn);
    lin.run(b.fc1, h.data(), S, f1.data());
    if (taps && taps->fc1) *taps->fc1 = f1;
    std::vector<std::uint16_t> sw(static_cast<std::size_t>(S) * kFfn);
    swiglu_host(f1.data(), S, kFfn, sw.data());
    f1 = {};
    lin.run(b.fc2, sw.data(), S, o.data());
    if (taps && taps->mlp_out) *taps->mlp_out = o;
    for (std::uint32_t r = 0; r < S; ++r) gate_row(x.data() + r * H, o.data() + r * H, mv.gate_mlp.data() + mod_row[r] * H, kHidden);
}

// --------------------------------------------------------------------------------------------- text path
std::vector<std::uint16_t> text_refine_host(const GlobalWeights& g, const std::uint16_t* text, std::uint32_t L,
    AttentionOp& att, std::vector<std::vector<std::uint16_t>>* stages) {
    const std::size_t H = kHidden;
    std::vector<std::uint16_t> x(L * H);
    bf16_linear_host(g.condition_proj, text, L, x.data());
    if (stages) stages->push_back(x);
    std::vector<std::uint16_t> h(L * H), qkv(static_cast<std::size_t>(L) * 3 * kInner), q(static_cast<std::size_t>(L) * kInner),
        k(q.size()), vt(q.size()), ao(q.size()), o(L * H), f1(static_cast<std::size_t>(L) * 2 * kFfn), sw(static_cast<std::size_t>(L) * kFfn);
    for (const auto& b : g.refiner) {
        // x = attn(norm1(x)) + x ; x = mlp(norm2(x)) + x   (bf16 adds)
        parallel_for(L, [&](std::uint64_t r) { rmsnorm_row(x.data() + r * H, b.norm1.data(), kHidden, kNormEps, h.data() + r * H); });
        bf16_linear_host(b.qkv, h.data(), L, qkv.data());
        qk_prepare(qkv.data(), L, b.q_norm.data(), b.k_norm.data(), nullptr, q.data(), k.data(), vt.data());
        att.run(q.data(), k.data(), vt.data(), L, L, ao.data());
        bf16_linear_host(b.out, ao.data(), L, o.data());
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = tob(bf(o[i]) + bf(x[i]));
        parallel_for(L, [&](std::uint64_t r) { rmsnorm_row(x.data() + r * H, b.norm2.data(), kHidden, kNormEps, h.data() + r * H); });
        bf16_linear_host(b.fc1, h.data(), L, f1.data());
        swiglu_host(f1.data(), L, kFfn, sw.data());
        bf16_linear_host(b.fc2, sw.data(), L, o.data());
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = tob(bf(o[i]) + bf(x[i]));
        if (stages) stages->push_back(x);
    }
    for (std::uint32_t r = 0; r < L; ++r) rmsnorm_row(x.data() + r * H, g.refiner_final_norm.data(), kHidden, kNormEps, h.data() + r * H);
    return h;
}

// --------------------------------------------------------------------------------------------- embedding
std::vector<float> cond_video_rows(const float* latent, std::uint32_t vt, std::uint32_t H, std::uint32_t W, double aug,
    const float* noise) {
    std::vector<float> r = patchify_video(latent, vt, H, W);
    if (aug < 1.0) {
        if (!noise) throw std::runtime_error("cond rows: noise required when aug < 1");
        const float a = static_cast<float>(aug), b = static_cast<float>(1.0 - aug);
        for (std::size_t i = 0; i < r.size(); ++i) r[i] = a * r[i] + b * noise[i];
    }
    return r;
}

namespace {
// rows [m][k] fp32 -> bf16 [m][n] through an fp32 linear.
void f32_linear_to_bf16(const F32Linear& l, const std::vector<float>& rows, std::uint32_t m, std::uint16_t* out) {
    parallel_for(m, [&](std::uint64_t r) {
        const std::vector<float> y = patch_linear_row(l, rows.data() + r * l.k);
        for (std::uint32_t c = 0; c < l.n; ++c) out[r * l.n + c] = tob(y[c]);
    });
}
}  // namespace

void embed_rows(const PackedLayout& L, const StreamInputs& in, std::vector<float>& all_v, std::vector<float>& all_a) {
    // validate every shape-derived range BEFORE any read
    if (in.T != L.latent_t || in.H != L.latent_h || in.W != L.latent_w || in.audio_t != L.audio_t)
        throw std::invalid_argument("embed: stream input shape differs from the layout");
    if (in.H % 2U || in.W % 2U) throw std::invalid_argument("embed: latent H / W must be even");
    if (in.video_x.size() != static_cast<std::size_t>(24) * in.T * in.H * in.W) throw std::invalid_argument("embed: video latent size");
    if (in.audio_x.size() != static_cast<std::size_t>(32) * 2U * in.audio_t) throw std::invalid_argument("embed: audio latent size");
    {
        std::size_t vt = 0, vc = 0, at = 0, ac = 0;
        for (auto u : L.img_update) (u ? vt : vc) += 1;
        for (auto u : L.audio_update) (u ? at : ac) += 1;
        if (vt != static_cast<std::size_t>(in.T) * (in.H / 2U) * (in.W / 2U)) throw std::invalid_argument("embed: layout video target rows");
        if (at != static_cast<std::size_t>(2) * in.audio_t) throw std::invalid_argument("embed: layout audio target rows");
        if (vc * kVideoPatch != in.cond_video.size()) throw std::invalid_argument("embed: cond video rows differ from the layout");
        if (ac * kAudioDim != in.cond_audio.size()) throw std::invalid_argument("embed: cond audio rows differ from the layout");
    }
    std::vector<float> vx(in.video_x.size()), ax(in.audio_x.size());
    for (std::size_t i = 0; i < vx.size(); ++i) vx[i] = bf(in.video_x[i]);
    for (std::size_t i = 0; i < ax.size(); ++i) ax[i] = bf(in.audio_x[i]);
    const std::vector<float> vrows = patchify_video(vx.data(), in.T, in.H, in.W);
    const std::vector<float> arows = pack_audio(ax.data(), in.audio_t);
    // all video-like rows in packed order: cond rows fill the non-update slots, target rows the update slots
    all_v.assign(L.img_update.size() * kVideoPatch, 0.0F);
    all_a.assign(L.audio_update.size() * kAudioDim, 0.0F);
    std::size_t ci = 0, ti = 0;
    for (std::size_t i = 0; i < L.img_update.size(); ++i) {
        const float* src = L.img_update[i] ? vrows.data() + (ti++) * kVideoPatch : in.cond_video.data() + (ci++) * kVideoPatch;
        std::memcpy(all_v.data() + i * kVideoPatch, src, kVideoPatch * 4);
    }
    if (ti * kVideoPatch != vrows.size() || ci * kVideoPatch != in.cond_video.size()) throw std::runtime_error("embed: video row count");
    ci = ti = 0;
    for (std::size_t i = 0; i < L.audio_update.size(); ++i) {
        const float* src = L.audio_update[i] ? arows.data() + (ti++) * kAudioDim : in.cond_audio.data() + (ci++) * kAudioDim;
        std::memcpy(all_a.data() + i * kAudioDim, src, kAudioDim * 4);
    }
    if (ti * kAudioDim != arows.size() || ci * kAudioDim != in.cond_audio.size()) throw std::runtime_error("embed: audio row count");
}

std::vector<std::uint16_t> embed_and_pack(const GlobalWeights& g, const PackedLayout& L, const StreamInputs& in) {
    std::vector<float> all_v, all_a;
    embed_rows(L, in, all_v, all_a);
    const std::uint32_t nv = static_cast<std::uint32_t>(L.img_update.size()), na = static_cast<std::uint32_t>(L.audio_update.size());
    std::vector<std::uint16_t> ve(static_cast<std::size_t>(nv) * kHidden), ae(static_cast<std::size_t>(na) * kHidden);
    f32_linear_to_bf16(g.video_patch_proj, all_v, nv, ve.data());
    f32_linear_to_bf16(g.audio_patch_proj, all_a, na, ae.data());
    std::vector<std::uint16_t> h(static_cast<std::size_t>(L.seq_len) * kHidden);
    std::size_t voff = 0, aoff = 0;
    for (const auto& s : L.segments) {
        const std::size_t n = s.stop - s.start;
        std::uint16_t* dst = h.data() + static_cast<std::size_t>(s.start) * kHidden;
        if (s.kind == SegKind::text) {
            if (in.text.size() != n * kHidden) throw std::runtime_error("embed: text size");
            std::memcpy(dst, in.text.data(), n * kHidden * 2);
        } else if (seg_is_video_like(s.kind)) {
            std::memcpy(dst, ve.data() + voff * kHidden, n * kHidden * 2);
            voff += n;
        } else {
            std::memcpy(dst, ae.data() + aoff * kHidden, n * kHidden * 2);
            aoff += n;
        }
    }
    return h;
}

// --------------------------------------------------------------------------------------------- final layer
namespace {
// python round(): half to even on doubles
double round_half_even(double v) { return std::nearbyint(v); }

// dt-weighted head of a PDD bank: weight = rows[0] + sum_{i in [first, stop)} w_i rows[i] (fp32), same for bias.
F32Linear pdd_head(const F32Linear& head, std::uint32_t n, std::uint32_t start, std::uint32_t stop, float flow_shift) {
    std::vector<double> grid(n + 1), sh(n + 1);
    for (std::uint32_t i = 0; i <= n; ++i) {
        // torch.linspace(1, 0, n + 1) in fp64: start + i * step, mirrored for the upper half
        const double step = (0.0 - 1.0) / static_cast<double>(n);
        grid[i] = i < (n + 1) / 2 ? 1.0 + step * i : 0.0 - step * static_cast<double>(n - i);
        const double fs = flow_shift;
        sh[i] = 1.0 - fs * grid[i] / (1.0 + (fs - 1.0) * grid[i]);
    }
    std::vector<double> dt;
    for (std::uint32_t i = start; i < stop; ++i) dt.push_back(sh[i + 1] - sh[i]);
    double tot = 0.0;
    for (double d : dt) tot += d;
    std::vector<float> w;
    for (double d : dt) w.push_back(static_cast<float>(d / tot));
    const std::uint32_t out = head.n / n, k = head.k;
    F32Linear h;
    h.n = out;
    h.k = k;
    h.w.assign(head.w.begin(), head.w.begin() + static_cast<std::ptrdiff_t>(out) * k);
    h.b.assign(head.b.begin(), head.b.begin() + out);
    const std::uint32_t first = std::max<std::uint32_t>(start, 1);
    std::vector<float> accw(static_cast<std::size_t>(out) * k, 0.0F), accb(out, 0.0F);
    for (std::uint32_t i = first; i < stop; ++i) {
        const float wi = w[i - start];
        for (std::size_t j = 0; j < accw.size(); ++j) accw[j] += wi * head.w[static_cast<std::size_t>(i) * out * k + j];
        for (std::uint32_t j = 0; j < out; ++j) accb[j] += wi * head.b[static_cast<std::size_t>(i) * out + j];
    }
    for (std::size_t j = 0; j < accw.size(); ++j) h.w[j] += accw[j];
    for (std::uint32_t j = 0; j < out; ++j) h.b[j] += accb[j];
    return h;
}
}  // namespace

bool select_heads(const GlobalWeights& g, const ForwardPlan& p, const PddContext* pdd, F32Linear& vh, F32Linear& ah) {
    const std::uint32_t n = g.pdd_steps;
    if (n <= 1) return false;
    if (!pdd || pdd->sample_sigmas.empty()) throw std::runtime_error("final layer: PDD heads need the sampler schedule");
    const auto& ss = pdd->sample_sigmas;
    const float sigma = p.sigma_v;
    std::size_t i = 0;
    float best = std::fabs(ss[0] - sigma);
    for (std::size_t j = 1; j < ss.size(); ++j)
        if (std::fabs(ss[j] - sigma) < best) {
            best = std::fabs(ss[j] - sigma);
            i = j;
        }
    const float sn = ss[std::min(i + 1, ss.size() - 1)];
    auto idx = [&](float s) {
        const float base = time_shift_sigma_f32(s, pdd->shift_video, 1.0F);
        return static_cast<std::int64_t>(round_half_even(static_cast<double>(1.0F - base) * n));
    };
    std::int64_t start = idx(sigma), stop = idx(sn);
    start = std::min<std::int64_t>(start, n - 1);
    stop = std::max<std::int64_t>(stop, start + 1);
    vh = pdd_head(g.video_out, n, static_cast<std::uint32_t>(start), static_cast<std::uint32_t>(stop), pdd->shift_video);
    ah = pdd_head(g.audio_out, n, static_cast<std::uint32_t>(start), static_cast<std::uint32_t>(stop), pdd->shift_audio);
    return true;
}

FinalRows final_layer_host(const GlobalWeights& g, const std::vector<std::uint16_t>& x, const PackedLayout& L,
    const ForwardPlan& p, const std::vector<float>& t_emb, const PddContext* pdd) {
    FinalRows fr;
    const std::uint32_t nt = p.n_t();
    const std::vector<float> ada = f32_linear(g.final_adaln, t_emb.data(), nt);  // [nt][2 * 5376]: shift | scale
    fr.shift.resize(static_cast<std::size_t>(nt) * kHidden);
    fr.scale.resize(fr.shift.size());
    for (std::uint32_t t = 0; t < nt; ++t)
        for (std::uint32_t c = 0; c < kHidden; ++c) {
            fr.shift[static_cast<std::size_t>(t) * kHidden + c] = ada[static_cast<std::size_t>(t) * 2 * kHidden + c];
            fr.scale[static_cast<std::size_t>(t) * kHidden + c] = ada[static_cast<std::size_t>(t) * 2 * kHidden + kHidden + c];
        }
    F32Linear vbank, abank;
    const bool bank = select_heads(g, p, pdd, vbank, abank);
    const F32Linear* vh = bank ? &vbank : &g.video_out;
    const F32Linear* ah = bank ? &abank : &g.audio_out;
    auto run = [&](const Segment& s, std::uint32_t trow, const F32Linear& head, std::vector<float>& out) {
        const std::uint32_t m = s.stop - s.start;
        out.resize(static_cast<std::size_t>(m) * head.n);
        const float* sh = fr.shift.data() + static_cast<std::size_t>(trow) * kHidden;
        const float* sc = fr.scale.data() + static_cast<std::size_t>(trow) * kHidden;
        parallel_for(m, [&](std::uint64_t r) {
            std::uint16_t nrm[kHidden];
            float f[kHidden];
            rmsnorm_row(x.data() + (s.start + r) * kHidden, g.final_norm.data(), kHidden, kNormEps, nrm);
            for (std::uint32_t c = 0; c < kHidden; ++c) f[c] = (bf(nrm[c]) * (1.0F + sc[c])) + sh[c];
            const std::vector<float> y = f32_linear(head, f, 1);
            std::memcpy(out.data() + r * head.n, y.data(), head.n * 4);
        });
    };
    run(L.find(SegKind::video), p.video_t_row, *vh, fr.video);
    run(L.find(SegKind::audio), p.audio_t_row, *ah, fr.audio);
    return fr;
}

// --------------------------------------------------------------------------------------------- model
FileBlockSource::FileBlockSource(const SafeTensors& st, std::uint32_t basis, bool cache, LoadPolicy pol)
    : st_(st), basis_(basis), n_(count_blocks(st)), cache_(cache), pol_(pol), blocks_(n_) {}

std::shared_ptr<const BlockWeights> FileBlockSource::block(std::uint32_t i) {
    if (cache_ && blocks_.at(i)) return blocks_[i];
    auto b = std::make_shared<const BlockWeights>(load_block(st_, i, basis_, pol_));
    if (cache_) blocks_[i] = b;
    return b;
}

StreamOutputs stream_forward_host(const GlobalWeights& g, BlockSource& blocks, const PackedLayout& L, const ForwardPlan& p,
    const StreamInputs& in, LinearOp& lin, AttentionOp& att, ForwardTaps* taps, const PddContext* pdd) {
    std::vector<std::uint16_t> x = embed_and_pack(g, L, in);
    if (taps && taps->h_in) *taps->h_in = x;
    const std::vector<float> t_emb = curve_t_emb(g, p.t_vals);
    if (taps && taps->t_emb) *taps->t_emb = t_emb;
    const RopeTable rope = rope_table(rope_angles(L.position_ids, g.inv_freq));
    const std::vector<std::uint32_t> mrow = mod_row_of(p.mod_segments, L.seq_len);
    for (std::uint32_t i = 0; i < blocks.num_blocks(); ++i) {
        const auto b = blocks.block(i);
        const ModVectors mv = mod_vectors(adaln_block_f32(*b, t_emb, p.n_t()), p.n_mod_rows());
        block_forward_host(*b, mv, mrow, rope, x, L.seq_len, lin, att, (i == 0 && taps) ? taps->block0 : nullptr);
        if (taps && taps->blocks) taps->blocks->push_back(x);
        if (taps && taps->after_block) taps->after_block(i, x);
    }
    FinalRows fr = final_layer_host(g, x, L, p, t_emb, pdd);
    StreamOutputs o;
    const std::vector<float> v = unpatchify_video(fr.video.data(), L.latent_t, L.latent_h, L.latent_w);
    const std::vector<float> a = unpack_audio(fr.audio.data(), L.audio_t);
    o.video_neg_v.resize(v.size());
    o.audio_neg_v.resize(a.size());
    for (std::size_t i = 0; i < v.size(); ++i) o.video_neg_v[i] = tob(-v[i]);
    for (std::size_t i = 0; i < a.size(); ++i) o.audio_neg_v[i] = tob(-a[i]);
    if (taps && taps->final_rows) *taps->final_rows = std::move(fr);
    return o;
}

std::vector<std::uint16_t> uncarry_audio(const std::vector<std::uint16_t>& audio_carried, const ForwardPlan& p) {
    const std::uint16_t carry = tob(p.sigma_a / p.sigma_v);
    std::vector<std::uint16_t> o(audio_carried.size());
    for (std::size_t i = 0; i < o.size(); ++i) o[i] = tob(bf(audio_carried[i]) * bf(carry));
    return o;
}

std::vector<std::uint16_t> carry_audio_velocity(const std::vector<std::uint16_t>& audio_carried,
    const std::vector<std::uint16_t>& out_a, const ForwardPlan& p, float audio_scale) {
    const std::vector<std::uint16_t> src = uncarry_audio(audio_carried, p);
    const float c0 = static_cast<float>(1.0 - static_cast<double>(audio_scale));
    const std::uint16_t c1 = tob(1.0F + static_cast<float>(static_cast<double>(audio_scale) - 1.0) * p.sigma_a);
    std::vector<std::uint16_t> o(out_a.size());
    for (std::size_t i = 0; i < o.size(); ++i) o[i] = tob(bf(tob(c0 * bf(src[i]))) + bf(tob(bf(c1) * bf(out_a[i]))));
    return o;
}

}  // namespace strix::h3::block
