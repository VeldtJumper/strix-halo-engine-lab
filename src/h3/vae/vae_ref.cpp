#include "h3/vae/vae_ref.hpp"
#include "h3/vae/vae_threads.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace strix::h3::vae {

namespace {
template <class F> void parallel_rows(int rows, F&& f) {
    const unsigned hw = std::max(1U, std::thread::hardware_concurrency());
    const int nt = int(std::min<unsigned>(hw, unsigned(std::max(1, rows))));
    Workers pool;
    for (int t = 0; t < nt; ++t)
        pool.run([&, t] {
            for (int r = t; r < rows; r += nt) f(r);
        });
    pool.join();
}
H16 to_h16(const std::vector<std::uint8_t>& raw) {
    H16 out(raw.size() / 2);
    std::memcpy(out.data(), raw.data(), raw.size());
    return out;
}
H16 read_f16(const SafeTensors& st, const std::string& name) {
    if (st.info(name).dtype == "f16") return to_h16(st.read(name));
    if (st.info(name).dtype == "f32") {  // the fast decoder ships FP32: cast as the module's .to(fp16)
        const auto raw = st.read(name);
        H16 out(raw.size() / 4);
        for (std::size_t i = 0; i < out.size(); ++i) {
            float f;
            std::memcpy(&f, raw.data() + 4 * i, 4);
            out[i] = f2h(f);
        }
        return out;
    }
    throw std::runtime_error("unsupported dtype for " + name);
}
}  // namespace

LinearH load_linear(const SafeTensors& st, const std::string& prefix) {
    LinearH l;
    const auto& wi = st.info(prefix + ".weight");
    l.out = int(wi.shape[0]);
    l.in = int(wi.numel() / wi.shape[0]);
    l.w = read_f16(st, prefix + ".weight");
    if (st.has(prefix + ".bias")) l.b = read_f16(st, prefix + ".bias");
    else l.b.assign(std::size_t(l.out), 0);
    return l;
}

LayerH load_layer(const SafeTensors& st, int layer, const std::string& block_prefix) {
    const std::string p = block_prefix + std::to_string(layer) + ".";
    LayerH w;
    w.norm1 = read_f16(st, p + "norm1.weight");
    w.norm2 = read_f16(st, p + "norm2.weight");
    w.scale1 = read_f16(st, p + "scale1");
    w.scale2 = read_f16(st, p + "scale2");
    w.qkv = load_linear(st, p + "attn.to_qkv");
    w.out = load_linear(st, p + "attn.to_out");
    w.w1 = load_linear(st, p + "ff.w1");
    w.w2 = load_linear(st, p + "ff.w2");
    return w;
}

HeadH load_head(const SafeTensors& st, const std::string& dec_prefix) {
    HeadH h;
    h.post_quant = load_linear(st, "post_quant_conv");
    h.x_embed = load_linear(st, dec_prefix + "x_embedder");
    h.proj = load_linear(st, dec_prefix + "proj_out");
    h.registers = read_f16(st, dec_prefix + "register_tokens");
    h.norm_out_w = read_f16(st, dec_prefix + "norm_out.weight");
    h.norm_out_b = read_f16(st, dec_prefix + "norm_out.bias");
    if (st.has("latents_mean")) {
        h.latents_mean = read_f16(st, "latents_mean");
        h.latents_std = read_f16(st, "latents_std");
    }
    return h;
}

RopeTable rope_table(int nt, int ny, int nx) {
    RopeTable r;
    r.L = nt * ny * nx + kSuffix;
    r.c.assign(std::size_t(r.L) * kRotPairs, 1.0F);
    r.s.assign(std::size_t(r.L) * kRotPairs, 0.0F);
    // inv_freq = 1 / 100 ** arange(0, 1, 6/48) in FP32, then the module cast to FP16.
    float inv[8];
    for (int f = 0; f < 8; ++f) {
        const float e = float(f) * 0.125F;
        const float p = std::pow(100.0F, e);
        inv[f] = rh(1.0F / p);
    }
    const auto coord = [](int i, int n) {  // FP16 arithmetic: ((i + 0.5) / n) * 2 - 1
        const float a = rh(float(i) + 0.5F);
        const float b = rh(a / float(n));
        const float c = rh(2.0F * b);
        return rh(c - 1.0F);
    };
    const float two_pi = float(2.0 * 3.14159265358979323846);
    for (int t = 0; t < nt; ++t)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                const int tok = (t * ny + y) * nx + x;
                const float ids[3] = {coord(t, nt), coord(y, ny), coord(x, nx)};
                for (int a = 0; a < 3; ++a)
                    for (int f = 0; f < 8; ++f) {
                        const float ang = (two_pi * ids[a]) * inv[f];
                        r.c[std::size_t(tok) * kRotPairs + a * 8 + f] = rh(std::cos(ang));
                        r.s[std::size_t(tok) * kRotPairs + a * 8 + f] = rh(std::sin(ang));
                    }
            }
    return r;
}

H16 linear(const H16& x, int rows, const LinearH& w, bool round_before_bias) {
    H16 y(std::size_t(rows) * w.out);
    std::vector<float> wf(w.w.size());
    for (std::size_t i = 0; i < wf.size(); ++i) wf[i] = h2f(w.w[i]);
    parallel_rows(rows, [&](int r) {
        std::vector<float> xr(static_cast<std::size_t>(w.in));
        for (int k = 0; k < w.in; ++k) xr[std::size_t(k)] = h2f(x[std::size_t(r) * w.in + k]);
        for (int n = 0; n < w.out; ++n) {
            const float* wr = wf.data() + std::size_t(n) * w.in;
            float acc[8] = {};
            int k = 0;
            for (; k + 8 <= w.in; k += 8)
                for (int u = 0; u < 8; ++u) acc[u] += xr[std::size_t(k + u)] * wr[k + u];
            float s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
            for (; k < w.in; ++k) s += xr[std::size_t(k)] * wr[k];
            y[std::size_t(r) * w.out + n] = round_before_bias ? f2h(rh(s) + h2f(w.b[std::size_t(n)])) : f2h(s + h2f(w.b[std::size_t(n)]));
        }
    });
    return y;
}

void linear_residual(H16& x, const H16& in, int rows, const LinearH& w, const H16& scale) {
    const H16 y = linear(in, rows, w);
    for (int r = 0; r < rows; ++r)
        for (int n = 0; n < w.out; ++n) {
            const std::size_t i = std::size_t(r) * w.out + n;
            x[i] = f2h(h2f(x[i]) + h2f(y[i]) * h2f(scale[std::size_t(n)]));
        }
}

H16 rms_norm(const H16& x, int rows, int width, const H16* weight, float eps) {
    H16 y(x.size());
    for (int r = 0; r < rows; ++r) {
        const half_bits* xr = x.data() + std::size_t(r) * width;
        double ss = 0;
        for (int k = 0; k < width; ++k) { const float v = h2f(xr[k]); ss += double(v) * v; }
        const float inv = 1.0F / std::sqrt(float(ss / width) + eps);
        for (int k = 0; k < width; ++k) {
            float v = h2f(xr[k]) * inv;
            if (weight) v *= h2f((*weight)[std::size_t(k)]);
            y[std::size_t(r) * width + k] = f2h(v);
        }
    }
    return y;
}

H16 layer_norm(const H16& x, int rows, int width, const H16& w, const H16& b, float eps) {
    H16 y(x.size());
    for (int r = 0; r < rows; ++r) {
        const half_bits* xr = x.data() + std::size_t(r) * width;
        double mu = 0;
        for (int k = 0; k < width; ++k) mu += h2f(xr[k]);
        mu /= width;
        double var = 0;
        for (int k = 0; k < width; ++k) { const double d = h2f(xr[k]) - mu; var += d * d; }
        var /= width;
        const float inv = float(1.0 / std::sqrt(var + eps));
        for (int k = 0; k < width; ++k)
            y[std::size_t(r) * width + k] = f2h(float(h2f(xr[k]) - mu) * inv * h2f(w[std::size_t(k)]) + h2f(b[std::size_t(k)]));
    }
    return y;
}

namespace {
// Per-head RMSNorm (unit weight) + split-half rope on [0, 48): FP16 rounding after each product and add.
void rms_rope_head(const half_bits* src, half_bits* dst, const float* c, const float* s) {
    double ss = 0;
    for (int d = 0; d < kHeadDim; ++d) { const float v = h2f(src[d]); ss += double(v) * v; }
    const float inv = 1.0F / std::sqrt(float(ss / kHeadDim) + 1e-5F);
    float n[kHeadDim];
    for (int d = 0; d < kHeadDim; ++d) n[d] = rh(h2f(src[d]) * inv);  // unit weight: x * inv * 1
    for (int i = 0; i < kRotPairs; ++i) {
        const float x0 = n[i], x1 = n[i + kRotPairs];
        const float o0 = rh(rh(x0 * c[i]) + x1 * -s[i]);
        const float o1 = rh(rh(x1 * c[i]) + x0 * s[i]);
        dst[i] = f2h(o0);
        dst[i + kRotPairs] = f2h(o1);
    }
    for (int d = kRotDim; d < kHeadDim; ++d) dst[d] = f2h(n[d]);
}
}  // namespace

// FP16 sanitization after attention (reference): NaN -> 0, +-Inf -> +-65504.
half_bits nan_to_num16(half_bits b) {
    if ((b & 0x7c00U) != 0x7c00U) return b;
    if (b & 0x03ffU) return 0;
    return half_bits((b & 0x8000U) | 0x7bffU);
}

H16 attention(const H16& qkv, int L, const RopeTable& rope, H16* q_out, H16* k_out) {
    H16 q(std::size_t(L) * kDim), k(std::size_t(L) * kDim), v(std::size_t(L) * kDim);
    for (int r = 0; r < L; ++r)
        for (int h = 0; h < kHeads; ++h) {
            const half_bits* row = qkv.data() + std::size_t(r) * kQkv + h * 3 * kHeadDim;
            const float* c = rope.c.data() + std::size_t(r) * kRotPairs;
            const float* s = rope.s.data() + std::size_t(r) * kRotPairs;
            rms_rope_head(row, q.data() + std::size_t(r) * kDim + h * kHeadDim, c, s);
            rms_rope_head(row + kHeadDim, k.data() + std::size_t(r) * kDim + h * kHeadDim, c, s);
            std::memcpy(v.data() + std::size_t(r) * kDim + h * kHeadDim, row + 2 * kHeadDim, kHeadDim * 2);
        }
    if (q_out) *q_out = q;
    if (k_out) *k_out = k;
    H16 o(std::size_t(L) * kDim);
    std::vector<float> kf(k.size()), vf(v.size());
    for (std::size_t i = 0; i < k.size(); ++i) { kf[i] = h2f(k[i]); vf[i] = h2f(v[i]); }
    parallel_rows(L * kHeads, [&](int job) {
        const int r = job / kHeads, h = job % kHeads;
        float qr[kHeadDim];
        for (int d = 0; d < kHeadDim; ++d) qr[d] = h2f(q[std::size_t(r) * kDim + h * kHeadDim + d]);
        std::vector<double> sc(static_cast<std::size_t>(L));
        double mx = -1e300;
        for (int j = 0; j < L; ++j) {
            const float* kr = kf.data() + std::size_t(j) * kDim + h * kHeadDim;
            double dot = 0;
            for (int d = 0; d < kHeadDim; ++d) dot += double(qr[d]) * kr[d];
            sc[std::size_t(j)] = dot * 0.125;
            mx = std::max(mx, sc[std::size_t(j)]);
        }
        double sum = 0;
        for (int j = 0; j < L; ++j) { sc[std::size_t(j)] = std::exp(sc[std::size_t(j)] - mx); sum += sc[std::size_t(j)]; }
        double acc[kHeadDim] = {};
        for (int j = 0; j < L; ++j) {
            const float* vr = vf.data() + std::size_t(j) * kDim + h * kHeadDim;
            for (int d = 0; d < kHeadDim; ++d) acc[d] += sc[std::size_t(j)] * vr[d];
        }
        for (int d = 0; d < kHeadDim; ++d) o[std::size_t(r) * kDim + h * kHeadDim + d] = nan_to_num16(f2h(float(acc[d] / sum)));
    });
    return o;
}

void layer_forward(const LayerH& w, H16& x, int L, const RopeTable& rope) {
    const H16 a = rms_norm(x, L, kDim, &w.norm1, 1e-5F);
    const H16 qkv = linear(a, L, w.qkv);
    const H16 o = attention(qkv, L, rope);
    linear_residual(x, o, L, w.out, w.scale1);
    const H16 a2 = rms_norm(x, L, kDim, &w.norm2, 1e-5F);
    const H16 h = linear(a2, L, w.w1);
    H16 g(std::size_t(L) * kFf);
    for (int r = 0; r < L; ++r)
        for (int n = 0; n < kFf; ++n) {
            const float gate = h2f(h[std::size_t(r) * kFf2 + n]);
            const float up = h2f(h[std::size_t(r) * kFf2 + kFf + n]);
            const float si = rh(gate / (1.0F + std::exp(-gate)));
            g[std::size_t(r) * kFf + n] = f2h(si * up);
        }
    linear_residual(x, g, L, w.w2, w.scale2);
}

H16 post_quant(const HeadH& h, const H16& z, int voxels) {
    H16 out(z.size());
    for (int co = 0; co < kLatentC; ++co)
        for (int p = 0; p < voxels; ++p) {
            float s = 0;
            for (int ci = 0; ci < kLatentC; ++ci) s += h2f(h.post_quant.w[std::size_t(co) * kLatentC + ci]) * h2f(z[std::size_t(ci) * voxels + p]);
            out[std::size_t(co) * voxels + p] = f2h(s + h2f(h.post_quant.b[std::size_t(co)]));
        }
    return out;
}

H16 embed(const HeadH& h, const H16& dec_in, int nt, int ny, int nx) {
    const int P = nt * ny * nx, L = P + kSuffix;
    H16 rows(std::size_t(P) * kLatentC);
    for (int p = 0; p < P; ++p)
        for (int c = 0; c < kLatentC; ++c) rows[std::size_t(p) * kLatentC + c] = dec_in[std::size_t(c) * P + p];
    // the reference feeds a transposed (non-contiguous) 3-D view: matmul rounds to FP16, then the bias add rounds again
    const H16 e = linear(rows, P, h.x_embed, true);
    H16 x(std::size_t(L) * kDim, 0);
    std::memcpy(x.data(), e.data(), e.size() * 2);
    std::memcpy(x.data() + std::size_t(P) * kDim, h.registers.data(), std::size_t(kRegisters) * kDim * 2);
    return x;  // last row: zeros
}

H16 unpatchify(const H16& pr, int nt, int ny, int nx) {
    const int F = 4 * nt, Hh = 16 * ny, Ww = 16 * nx;
    H16 out(std::size_t(3) * F * Hh * Ww);
    for (int t = 0; t < nt; ++t)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                const half_bits* row = pr.data() + std::size_t((t * ny + y) * nx + x) * kProj;
                for (int c = 0; c < 3; ++c)
                    for (int pt = 0; pt < 4; ++pt)
                        for (int py = 0; py < 16; ++py)
                            for (int px = 0; px < 16; ++px)
                                out[((std::size_t(c) * F + t * 4 + pt) * Hh + y * 16 + py) * Ww + x * 16 + px] =
                                    row[((c * 4 + pt) * 16 + py) * 16 + px];
            }
    return out;
}

H16 head_out(const HeadH& h, const H16& x, int nt, int ny, int nx, H16* proj_rows) {
    const int P = nt * ny * nx;
    const H16 xs(x.begin(), x.begin() + std::ptrdiff_t(std::size_t(P) * kDim));
    const H16 n = layer_norm(xs, P, kDim, h.norm_out_w, h.norm_out_b, 1e-5F);
    const H16 pr = linear(n, P, h.proj);
    if (proj_rows) *proj_rows = pr;
    return unpatchify(pr, nt, ny, nx);
}

// Invalid comparisons (different lengths, empty, any non-finite value on either side) score -inf PSNR and +inf
// error, so every threshold fails on them.
double Diff::psnr(double peak) const {
    if (!valid()) return -INFINITY;
    return rms_err > 0 ? 20.0 * std::log10(peak / rms_err) : INFINITY;
}
double Diff::snr_db() const {
    if (!valid()) return -INFINITY;
    return rms_err > 0 ? 20.0 * std::log10(rms_ref / rms_err) : INFINITY;
}
double Diff::rel_l2() const {
    if (!valid()) return INFINITY;
    return rms_ref > 0 ? rms_err / rms_ref : rms_err;
}

namespace {
template <class T, class ToD, class Bits>
Diff diff_impl(const std::vector<T>& got, const std::vector<T>& ref, ToD to_d, Bits bits) {
    Diff d;
    d.size_mismatch = got.size() != ref.size();
    d.n = std::min(got.size(), ref.size());
    double se = 0, sr = 0;
    for (std::size_t i = 0; i < d.n; ++i) {
        if (bits(got[i]) != bits(ref[i])) ++d.mismatches;
        const double a = to_d(got[i]), b = to_d(ref[i]);
        if (!std::isfinite(a) || !std::isfinite(b)) { ++d.nonfinite; continue; }
        d.max_abs = std::max(d.max_abs, std::fabs(a - b));
        se += (a - b) * (a - b);
        sr += b * b;
    }
    d.rms_err = std::sqrt(se / double(std::max<std::size_t>(1, d.n)));
    d.rms_ref = std::sqrt(sr / double(std::max<std::size_t>(1, d.n)));
    return d;
}
}  // namespace

Diff diff_f16(const H16& got, const H16& ref) {
    return diff_impl(got, ref, [](half_bits h) { return double(h2f(h)); }, [](half_bits h) { return unsigned(h); });
}
Diff diff_f32(const std::vector<float>& got, const std::vector<float>& ref) {
    return diff_impl(got, ref, [](float f) { return double(f); }, [](float f) { return f32_bits(f); });
}

}  // namespace strix::h3::vae
