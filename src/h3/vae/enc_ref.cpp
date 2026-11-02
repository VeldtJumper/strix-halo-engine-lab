#include "h3/vae/enc_ref.hpp"
#include "h3/vae/vae_threads.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace strix::h3::vae::enc {

namespace {
H16 rd(const SafeTensors& st, const std::string& n) {
    if (st.info(n).dtype != "f16") throw std::runtime_error("encoder: " + n + " is not f16");
    const auto raw = st.read(n);
    H16 v(raw.size() / 2);
    std::memcpy(v.data(), raw.data(), raw.size());
    return v;
}
// [cout][cin][kt][kh][kw] -> last temporal tap [cout][cin][kh][kw]
Conv2 conv2(const SafeTensors& st, const std::string& p) {
    const auto& wi = st.info(p + ".weight");
    Conv2 c;
    c.cout = int(wi.shape[0]);
    c.cin = int(wi.shape[1]);
    const int kt = int(wi.shape[2]);
    c.k = int(wi.shape[3]);
    const H16 w = rd(st, p + ".weight");
    c.w.resize(std::size_t(c.cout) * c.cin * c.k * c.k);
    for (std::size_t oc = 0; oc < std::size_t(c.cout) * c.cin; ++oc)
        std::memcpy(c.w.data() + oc * c.k * c.k, w.data() + (oc * kt + (kt - 1)) * c.k * c.k, std::size_t(c.k) * c.k * 2);
    c.b = rd(st, p + ".bias");
    c.kt = kt;
    c.w3 = w;  // all temporal taps [cout][cin][kt][k][k] (multi-frame path)
    return c;
}
Norm norm(const SafeTensors& st, const std::string& p) { return {rd(st, p + ".weight"), rd(st, p + ".bias")}; }
template <class F> void parallel(int n, F&& f) {
    const int nt = int(std::max(1U, std::min<unsigned>(std::thread::hardware_concurrency(), unsigned(std::max(1, n)))));
    Workers pool;
    for (int t = 0; t < nt; ++t) pool.run([&, t] { for (int i = t; i < n; i += nt) f(i); });
    pool.join();
}
int reflect(int i, int n) {  // single reflection (pad <= 1)
    if (i < 0) return -i;
    if (i >= n) return 2 * n - 2 - i;
    return i;
}
}  // namespace

void check_image(int H, int W, std::size_t values) {
    if (H < 32 || W < 32 || H % 16 || W % 16)
        throw std::invalid_argument("h3 vae encoder: image sides must be multiples of 16 and >= 32 (got " + std::to_string(H) + " x " + std::to_string(W) + ")");
    if (values != std::size_t(3) * H * W) throw std::invalid_argument("h3 vae encoder: image buffer length != 3 x H x W");
}
void check_tile(const Map& p) {
    if (p.C != 3 || p.H < 32 || p.W < 32 || p.H > 256 || p.W > 256 || p.H % 16 || p.W % 16)
        throw std::invalid_argument("h3 vae encoder: tile must be 3 x [32, 256] x [32, 256], multiples of 16");
    if (p.v.size() != std::size_t(3) * p.H * p.W) throw std::invalid_argument("h3 vae encoder: tile buffer length != 3 x H x W");
}

Weights load(const SafeTensors& st) {
    Weights w;
    w.conv_in = conv2(st, "encoder.conv_in");
    for (int l = 0; l < 6; ++l) {
        for (int b = 0; b < 2; ++b) {
            const std::string p = "encoder.down." + std::to_string(l) + ".block." + std::to_string(b) + ".";
            Res& r = w.blocks[l][b];
            r.n1 = norm(st, p + "norm1");
            r.n2 = norm(st, p + "norm2");
            r.c1 = conv2(st, p + "conv1");
            r.c2 = conv2(st, p + "conv2");
            r.has_nin = st.has(p + "nin_shortcut.weight");
            if (r.has_nin) r.nin = conv2(st, p + "nin_shortcut");
        }
        if (l < 4) w.down[l] = conv2(st, "encoder.down." + std::to_string(l) + ".downsample.conv");
    }
    w.norm_out = norm(st, "encoder.norm_out");
    w.conv_out = conv2(st, "encoder.conv_out");
    w.quant = conv2(st, "quant_conv");
    w.latents_mean = rd(st, "latents_mean");
    w.latents_std = rd(st, "latents_std");
    return w;
}

Map conv(const Conv2& c, const Map& x, int stride, const Map* res) {
    Map y;
    y.C = c.cout;
    const int pad = stride == 1 ? c.k / 2 : 0;  // downsample: no left/top pad, reflect 1 on the right/bottom
    const int Hp = stride == 1 ? x.H + 2 * pad : x.H + 1, Wp = stride == 1 ? x.W + 2 * pad : x.W + 1;
    y.H = (Hp - c.k) / stride + 1;
    y.W = (Wp - c.k) / stride + 1;
    y.v.resize(std::size_t(y.C) * y.H * y.W);
    std::vector<float> xf(x.v.size()), wf(c.w.size());
    for (std::size_t i = 0; i < xf.size(); ++i) xf[i] = h2f(x.v[i]);
    for (std::size_t i = 0; i < wf.size(); ++i) wf[i] = h2f(c.w[i]);
    parallel(c.cout, [&](int co) {
        std::vector<double> acc(std::size_t(y.H) * y.W, 0.0);
        for (int ci = 0; ci < c.cin; ++ci)
            for (int ky = 0; ky < c.k; ++ky)
                for (int kx = 0; kx < c.k; ++kx) {
                    const double wv = wf[((std::size_t(co) * c.cin + ci) * c.k + ky) * c.k + kx];
                    for (int oy = 0; oy < y.H; ++oy) {
                        const int iy = reflect(oy * stride + ky - pad, x.H);
                        const float* xr = xf.data() + (std::size_t(ci) * x.H + iy) * x.W;
                        double* ar = acc.data() + std::size_t(oy) * y.W;
                        for (int ox = 0; ox < y.W; ++ox) ar[ox] += wv * xr[reflect(ox * stride + kx - pad, x.W)];
                    }
                }
        const float b = h2f(c.b[std::size_t(co)]);
        for (std::size_t q = 0; q < acc.size(); ++q) {
            float v = rh(float(acc[q]) + b);
            if (res) v = rh(v + h2f(res->v[std::size_t(co) * acc.size() + q]));
            y.v[std::size_t(co) * acc.size() + q] = f2h(v);
        }
    });
    return y;
}

Map gn_silu(const Norm& n, const Map& x) {
    Map y = x;
    const int G = 32, cg = x.C / G;
    const std::size_t plane = std::size_t(x.H) * x.W;
    for (int g = 0; g < G; ++g) {
        double s = 0, q = 0;
        const std::size_t cnt = std::size_t(cg) * plane;
        for (std::size_t i = 0; i < cnt; ++i) s += h2f(x.v[std::size_t(g) * cnt + i]);
        const double mu = s / double(cnt);
        for (std::size_t i = 0; i < cnt; ++i) { const double d = h2f(x.v[std::size_t(g) * cnt + i]) - mu; q += d * d; }
        const float inv = float(1.0 / std::sqrt(q / double(cnt) + 1e-6));
        for (int c = g * cg; c < (g + 1) * cg; ++c) {
            const float w = h2f(n.w[std::size_t(c)]), b = h2f(n.b[std::size_t(c)]);
            for (std::size_t p = 0; p < plane; ++p) {
                const std::size_t i = std::size_t(c) * plane + p;
                const float nv = rh(float(h2f(x.v[i]) - mu) * inv * w + b);
                y.v[i] = f2h(nv / (1.0F + std::exp(-nv)));
            }
        }
    }
    return y;
}

Map resblock(const Res& r, const Map& x) {
    const Map h = conv(r.c1, gn_silu(r.n1, x), 1);
    const Map s = r.has_nin ? conv(r.nin, x, 1) : x;
    return conv(r.c2, gn_silu(r.n2, h), 1, &s);
}

Map normalize_pixels(const std::vector<float>& x, int H, int W) {
    const float mean[3] = {rh(0.485F), rh(0.456F), rh(0.406F)}, sd[3] = {rh(0.229F), rh(0.224F), rh(0.225F)};
    Map m;
    m.C = 3; m.H = H; m.W = W;
    m.v.resize(x.size());
    for (int c = 0; c < 3; ++c)
        for (std::size_t p = 0; p < std::size_t(H) * W; ++p) {
            const std::size_t i = std::size_t(c) * H * W + p;
            float v = rh(x[i]);
            v = rh(v + 1.0F);
            v = rh(v * 0.5F);
            v = rh(v - mean[c]);
            m.v[i] = f2h(v / sd[c]);
        }
    return m;
}

Map quant(const Weights& w, const Map& x) { return conv(w.quant, x, 1); }

Map encode_tile(const Weights& w, const Map& pixels) {
    check_tile(pixels);
    Map x = conv(w.conv_in, pixels, 1);
    for (int l = 0; l < 6; ++l) {
        x = resblock(w.blocks[l][0], x);
        x = resblock(w.blocks[l][1], x);
        if (l < 4) x = conv(w.down[l], x, 2);
    }
    return quant(w, conv(w.conv_out, gn_silu(w.norm_out, x), 1));
}

namespace {
Pix16 as_pix(const Map& m) { Pix16 p; p.C = m.C; p.F = 1; p.Hh = m.H; p.Ww = m.W; p.v = m.v; return p; }
Map as_map(const Pix16& p) { Map m; m.C = p.C; m.H = p.Hh; m.W = p.Ww; m.v = p.v; return m; }
Map crop(const Map& s, int ny, int nx) {
    Map o; o.C = s.C; o.H = ny; o.W = nx;
    o.v.resize(std::size_t(o.C) * ny * nx);
    for (int c = 0; c < s.C; ++c)
        for (int y = 0; y < ny; ++y)
            std::memcpy(o.v.data() + (std::size_t(c) * ny + y) * nx, s.v.data() + (std::size_t(c) * s.H + y) * s.W, std::size_t(nx) * 2);
    return o;
}
}  // namespace

Map blend_tiles(const std::vector<std::vector<Map>>& tiles, const Tiles1D& ty, const Tiles1D& tx) {
    const int ny = int(tiles.size()), nx = int(tiles[0].size());
    std::vector<std::vector<Map>> out(static_cast<std::size_t>(ny));
    for (int i = 0; i < ny; ++i)
        for (int j = 0; j < nx; ++j) {
            Pix16 t = as_pix(tiles[std::size_t(i)][std::size_t(j)]);
            if (i > 0) t = blend(as_pix(tiles[std::size_t(i - 1)][std::size_t(j)]), t, ty.overlap[std::size_t(i - 1)] / 16, 1);
            if (j > 0) t = blend(as_pix(tiles[std::size_t(i)][std::size_t(j - 1)]), t, tx.overlap[std::size_t(j - 1)] / 16, 2);
            Map m = as_map(t);
            m = crop(m, i < ny - 1 ? m.H - ty.overlap[std::size_t(i)] / 16 : m.H, j < nx - 1 ? m.W - tx.overlap[std::size_t(j)] / 16 : m.W);
            out[std::size_t(i)].push_back(std::move(m));
        }
    Map r;
    r.C = out[0][0].C;
    for (const auto& m : out[0]) r.W += m.W;
    for (const auto& row : out) r.H += row[0].H;
    r.v.assign(std::size_t(r.C) * r.H * r.W, 0);
    int y0 = 0;
    for (const auto& row : out) {
        int x0 = 0;
        for (const auto& m : row) {
            for (int c = 0; c < m.C; ++c)
                for (int y = 0; y < m.H; ++y)
                    std::memcpy(r.v.data() + (std::size_t(c) * r.H + y0 + y) * r.W + x0, m.v.data() + (std::size_t(c) * m.H + y) * m.W, std::size_t(m.W) * 2);
            x0 += m.W;
        }
        y0 += row[0].H;
    }
    return r;
}

std::vector<float> latent_of(const Weights& w, const Map& m) {
    const std::size_t plane = std::size_t(m.H) * m.W;
    std::vector<float> z(std::size_t(24) * plane);
    for (int c = 0; c < 24; ++c) {
        const float mu = h2f(w.latents_mean[std::size_t(c)]), sd = h2f(w.latents_std[std::size_t(c)]);
        for (std::size_t p = 0; p < plane; ++p) z[std::size_t(c) * plane + p] = (h2f(m.v[std::size_t(c) * plane + p]) - mu) / sd;
    }
    return z;
}

void check_video(int T, int H, int W, std::size_t values) {
    if (T < 2) throw std::invalid_argument("h3 vae encoder: video path needs T >= 2 frames (T == 1: image path)");
    check_image(H, W, std::size_t(3) * H * W);
    if (values != std::size_t(3) * T * H * W) throw std::invalid_argument("h3 vae encoder: video buffer length != 3 x T x H x W");
}

Vid conv3(const Conv2& c, const Vid& x, int ts, int ss, const Vid* res) {
    const int front = c.kt > 1 ? 2 : 0;
    const int pad = ss == 1 ? c.k / 2 : 0;
    const int Hp = ss == 1 ? x.H + 2 * pad : x.H + 1, Wp = ss == 1 ? x.W + 2 * pad : x.W + 1;
    Vid y;
    y.C = c.cout;
    y.T = (x.T + front - c.kt) / ts + 1;
    y.H = (Hp - c.k) / ss + 1;
    y.W = (Wp - c.k) / ss + 1;
    const std::size_t po = std::size_t(y.H) * y.W, pi = std::size_t(x.H) * x.W;
    y.v.resize(std::size_t(y.C) * y.T * po);
    std::vector<float> xf(x.v.size()), wf(c.w3.size());
    for (std::size_t i = 0; i < xf.size(); ++i) xf[i] = h2f(x.v[i]);
    for (std::size_t i = 0; i < wf.size(); ++i) wf[i] = h2f(c.w3[i]);
    parallel(c.cout, [&](int co) {
        std::vector<double> acc(std::size_t(y.T) * po, 0.0);
        for (int ci = 0; ci < c.cin; ++ci)
            for (int kt = 0; kt < c.kt; ++kt)
                for (int ky = 0; ky < c.k; ++ky)
                    for (int kx = 0; kx < c.k; ++kx) {
                        const double wv = wf[(((std::size_t(co) * c.cin + ci) * c.kt + kt) * c.k + ky) * c.k + kx];
                        for (int ot = 0; ot < y.T; ++ot) {
                            const int it = ot * ts + kt - front;
                            if (it < 0) continue;  // causal zero padding
                            for (int oy = 0; oy < y.H; ++oy) {
                                const int iy = reflect(oy * ss + ky - pad, x.H);
                                const float* xr = xf.data() + ((std::size_t(ci) * x.T + it) * x.H + iy) * x.W;
                                double* ar = acc.data() + std::size_t(ot) * po + std::size_t(oy) * y.W;
                                for (int ox = 0; ox < y.W; ++ox) ar[ox] += wv * xr[reflect(ox * ss + kx - pad, x.W)];
                            }
                        }
                    }
        (void)pi;
        const float b = h2f(c.b[std::size_t(co)]);
        for (std::size_t q = 0; q < acc.size(); ++q) {
            float v = rh(float(acc[q]) + b);
            if (res) v = rh(v + h2f(res->v[std::size_t(co) * acc.size() + q]));
            y.v[std::size_t(co) * acc.size() + q] = f2h(v);
        }
    });
    return y;
}

Vid gn_silu3(const Norm& n, const Vid& x) {  // statistics per frame
    Vid y = x;
    const std::size_t plane = std::size_t(x.H) * x.W;
    for (int t = 0; t < x.T; ++t) {
        Map f;
        f.C = x.C; f.H = x.H; f.W = x.W;
        f.v.resize(std::size_t(x.C) * plane);
        for (int c = 0; c < x.C; ++c)
            std::memcpy(f.v.data() + std::size_t(c) * plane, x.v.data() + (std::size_t(c) * x.T + t) * plane, plane * 2);
        const Map g = gn_silu(n, f);
        for (int c = 0; c < x.C; ++c)
            std::memcpy(y.v.data() + (std::size_t(c) * x.T + t) * plane, g.v.data() + std::size_t(c) * plane, plane * 2);
    }
    return y;
}

Vid resblock3(const Res& r, const Vid& x) {
    const Vid h = conv3(r.c1, gn_silu3(r.n1, x), 1, 1);
    const Vid s = r.has_nin ? conv3(r.nin, x, 1, 1) : x;
    return conv3(r.c2, gn_silu3(r.n2, h), 1, 1, &s);
}

Vid clip_pixels(const std::vector<float>& x, int T, int H, int W, int clip) {
    const float mean[3] = {rh(0.485F), rh(0.456F), rh(0.406F)}, sd[3] = {rh(0.229F), rh(0.224F), rh(0.225F)};
    Vid v;
    v.C = 3; v.T = 17; v.H = H; v.W = W;
    const std::size_t plane = std::size_t(H) * W;
    v.v.resize(std::size_t(3) * 17 * plane);
    for (int c = 0; c < 3; ++c)
        for (int f = 0; f < 17; ++f) {
            const int src = std::min(clip * 17 + f, T - 1);
            for (std::size_t p = 0; p < plane; ++p) {
                float u = rh(x[(std::size_t(c) * T + src) * plane + p]);
                u = rh(u + 1.0F);
                u = rh(u * 0.5F);
                u = rh(u - mean[c]);
                v.v[(std::size_t(c) * 17 + f) * plane + p] = f2h(u / sd[c]);
            }
        }
    return v;
}

Vid blend_tiles3(const std::vector<std::vector<Vid>>& tiles, const Tiles1D& ty, const Tiles1D& tx) {
    const auto pix = [](const Vid& m) { Pix16 p; p.C = m.C; p.F = m.T; p.Hh = m.H; p.Ww = m.W; p.v = m.v; return p; };
    const int ny = int(tiles.size()), nx = int(tiles[0].size());
    std::vector<std::vector<Pix16>> out(static_cast<std::size_t>(ny));
    for (int i = 0; i < ny; ++i)
        for (int j = 0; j < nx; ++j) {
            Pix16 t = pix(tiles[std::size_t(i)][std::size_t(j)]);
            if (i > 0) t = blend(pix(tiles[std::size_t(i - 1)][std::size_t(j)]), t, ty.overlap[std::size_t(i - 1)] / 16, 1);
            if (j > 0) t = blend(pix(tiles[std::size_t(i)][std::size_t(j - 1)]), t, tx.overlap[std::size_t(j - 1)] / 16, 2);
            const int keep_h = i < ny - 1 ? t.Hh - ty.overlap[std::size_t(i)] / 16 : t.Hh;
            const int keep_w = j < nx - 1 ? t.Ww - tx.overlap[std::size_t(j)] / 16 : t.Ww;
            Pix16 c;
            c.C = t.C; c.F = t.F; c.Hh = keep_h; c.Ww = keep_w;
            c.v.resize(std::size_t(c.C) * c.F * keep_h * keep_w);
            for (int ch = 0; ch < t.C; ++ch)
                for (int f = 0; f < t.F; ++f)
                    for (int y = 0; y < keep_h; ++y)
                        std::memcpy(c.v.data() + ((std::size_t(ch) * c.F + f) * keep_h + y) * keep_w,
                                    t.v.data() + ((std::size_t(ch) * t.F + f) * t.Hh + y) * t.Ww, std::size_t(keep_w) * 2);
            out[std::size_t(i)].push_back(std::move(c));
        }
    Vid r;
    r.C = out[0][0].C; r.T = out[0][0].F;
    for (const auto& m : out[0]) r.W += m.Ww;
    for (const auto& row : out) r.H += row[0].Hh;
    r.v.assign(std::size_t(r.C) * r.T * r.H * r.W, 0);
    int y0 = 0;
    for (const auto& row : out) {
        int x0 = 0;
        for (const auto& m : row) {
            for (int ch = 0; ch < m.C; ++ch)
                for (int f = 0; f < m.F; ++f)
                    for (int y = 0; y < m.Hh; ++y)
                        std::memcpy(r.v.data() + ((std::size_t(ch) * r.T + f) * r.H + y0 + y) * r.W + x0,
                                    m.v.data() + ((std::size_t(ch) * m.F + f) * m.Hh + y) * m.Ww, std::size_t(m.Ww) * 2);
            x0 += m.Ww;
        }
        y0 += row[0].Hh;
    }
    return r;
}

std::vector<float> video_latent(const Weights& w, const std::vector<Vid>& cm) {
    const int h = cm[0].H, wd = cm[0].W, per = cm[0].T, total = per * int(cm.size()) - 3;
    const std::size_t plane = std::size_t(h) * wd;
    std::vector<float> z(std::size_t(24) * total * plane);
    for (int c = 0; c < 24; ++c) {
        const float mu = h2f(w.latents_mean[std::size_t(c)]), sd = h2f(w.latents_std[std::size_t(c)]);
        for (int t = 0; t < total; ++t) {
            const Vid& m = cm[std::size_t(t / per)];
            const int lt = t % per;
            for (std::size_t p = 0; p < plane; ++p)
                z[(std::size_t(c) * total + t) * plane + p] = (h2f(m.v[(std::size_t(c) * m.T + lt) * plane + p]) - mu) / sd;
        }
    }
    return z;
}

Vid encode_clip_tile(const Weights& w, const Vid& px) {
    const int tdown[4] = {1, 2, 2, 1};
    Vid x = conv3(w.conv_in, px, 1, 1);
    for (int l = 0; l < 6; ++l) {
        x = resblock3(w.blocks[l][0], x);
        x = resblock3(w.blocks[l][1], x);
        if (l < 4) x = conv3(w.down[l], x, tdown[l], 2);
    }
    x = conv3(w.conv_out, gn_silu3(w.norm_out, x), 1, 1);
    return conv3(w.quant, x, 1, 1);
}

}  // namespace strix::h3::vae::enc
