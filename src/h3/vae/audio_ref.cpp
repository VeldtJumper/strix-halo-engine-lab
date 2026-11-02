#include "h3/vae/audio_ref.hpp"
#include "h3/vae/vae_threads.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace strix::h3::vae::audio {

namespace {
std::vector<float> f32(const SafeTensors& st, const std::string& name) {
    if (st.info(name).dtype != "f32") throw std::runtime_error("audio vae: " + name + " is not f32");
    const auto raw = st.read(name);
    std::vector<float> v(raw.size() / 4);
    std::memcpy(v.data(), raw.data(), raw.size());
    return v;
}
Conv load_conv(const SafeTensors& st, const std::string& p, bool transposed, int stride, int pad, int dil) {
    Conv c;
    const auto& wi = st.info(p + ".weight");
    c.transposed = transposed;
    c.cin = int(transposed ? wi.shape[0] : wi.shape[1]);
    c.cout = int(transposed ? wi.shape[1] : wi.shape[0]);
    c.k = int(wi.shape[2]);
    c.stride = stride; c.pad = pad; c.dil = dil;
    c.w = f32(st, p + ".weight");
    if (st.has(p + ".bias")) c.b = f32(st, p + ".bias");
    return c;
}
Act load_act(const SafeTensors& st, const std::string& p) {
    Act a;
    a.alpha = f32(st, p + ".act.alpha");
    a.beta = f32(st, p + ".act.beta");
    const auto up = f32(st, p + ".upsample.filter"), down = f32(st, p + ".downsample.lowpass.filter");
    if (up.size() != 12 || down.size() != 12) throw std::runtime_error("audio vae: filter taps != 12");
    std::copy(up.begin(), up.end(), a.up);
    std::copy(down.begin(), down.end(), a.down);
    return a;
}
template <class F> void parallel(int n, F&& f) {
    const int nt = int(std::max(1U, std::min<unsigned>(std::thread::hardware_concurrency(), unsigned(std::max(1, n)))));
    Workers pool;
    for (int t = 0; t < nt; ++t) pool.run([&, t] { for (int i = t; i < n; i += nt) f(i); });
    pool.join();
}
}  // namespace

Weights load(const SafeTensors& st) {
    Weights w;
    w.latents_mean = f32(st, "latents_mean");
    w.latents_std = f32(st, "latents_std");
    w.dec_in = load_conv(st, "dec_in_proj", false, 1, 0, 1);
    w.conv_pre = load_conv(st, "decoder.conv_pre", false, 1, 3, 1);
    const int rates[7] = {5, 5, 2, 2, 2, 2, 2}, ks[7] = {9, 9, 4, 4, 4, 4, 4};
    for (int i = 0; i < 7; ++i) w.ups[i] = load_conv(st, "decoder.ups." + std::to_string(i) + ".0", true, rates[i], (ks[i] - rates[i]) / 2, 1);
    const int amp_k[3] = {3, 7, 11}, dil[3] = {1, 3, 5};
    for (int r = 0; r < 21; ++r) {
        Amp& b = w.amp[r];
        b.k = amp_k[r % 3];
        const std::string p = "decoder.resblocks." + std::to_string(r) + ".";
        for (int i = 0; i < 3; ++i) {
            b.c1[i] = load_conv(st, p + "convs1." + std::to_string(i), false, 1, (b.k * dil[i] - dil[i]) / 2, dil[i]);
            b.c2[i] = load_conv(st, p + "convs2." + std::to_string(i), false, 1, (b.k - 1) / 2, 1);
        }
        for (int i = 0; i < 6; ++i) b.a[i] = load_act(st, p + "activations." + std::to_string(i));
    }
    w.act_post = load_act(st, "decoder.activation_post");
    w.conv_post = load_conv(st, "decoder.conv_post", false, 1, 3, 1);
    return w;
}

int conv_out_len(const Conv& c, int n) {
    if (c.transposed) return (n - 1) * c.stride - 2 * c.pad + c.k;
    return (n + 2 * c.pad - c.dil * (c.k - 1) - 1) / c.stride + 1;
}

Sig conv(const Conv& c, const Sig& x, int n) {
    const int m = conv_out_len(c, n);
    Sig y(std::size_t(c.cout) * m);
    parallel(c.cout, [&](int co) {
        std::vector<double> acc(std::size_t(m), c.b.empty() ? 0.0 : double(c.b[std::size_t(co)]));
        for (int ci = 0; ci < c.cin; ++ci) {
            const float* xr = x.data() + std::size_t(ci) * n;
            for (int kk = 0; kk < c.k; ++kk) {
                if (c.transposed) {
                    const double wv = c.w[(std::size_t(ci) * c.cout + co) * c.k + kk];
                    for (int s = 0; s < n; ++s) {
                        const int t = s * c.stride - c.pad + kk;
                        if (t >= 0 && t < m) acc[std::size_t(t)] += wv * xr[s];
                    }
                } else {
                    const double wv = c.w[(std::size_t(co) * c.cin + ci) * c.k + kk];
                    for (int t = 0; t < m; ++t) {
                        const int s = t * c.stride - c.pad + kk * c.dil;
                        if (s >= 0 && s < n) acc[std::size_t(t)] += wv * xr[s];
                    }
                }
            }
        }
        for (int t = 0; t < m; ++t) y[std::size_t(co) * m + t] = float(acc[std::size_t(t)]);
    });
    return y;
}

Sig activation(const Act& a, const Sig& x, int C, int N) {
    Sig y(std::size_t(C) * N);
    parallel(C, [&](int c) {
        const float* xr = x.data() + std::size_t(c) * N;
        const float al = std::exp(a.alpha[std::size_t(c)]), inv = 1.0F / (std::exp(a.beta[std::size_t(c)]) + 1e-9F);
        // up: xp[j] = x[clamp(j - 5)], j in [0, N + 10); y2[m] = sum_{2 j + k = m} xp[j] f[k]; u[i] = 2 y2[i + 15], i < 2 N
        std::vector<float> v(std::size_t(2) * N);
        for (int i = 0; i < 2 * N; ++i) {
            const int m = i + 15;
            double s = 0;
            for (int k = m % 2; k < 12; k += 2) {
                const int j = (m - k) / 2;
                if (j < 0 || j >= N + 10) continue;
                s += double(xr[std::clamp(j - 5, 0, N - 1)]) * a.up[k];
            }
            const float u = float(s) * 2.0F;
            float t = std::sin(al * u);
            t = t * t;
            t = t * inv;
            v[std::size_t(i)] = t + u;
        }
        // down: vp[j] = v[clamp(j - 5)], out[t] = sum_k vp[2 t + k] g[k]
        for (int t = 0; t < N; ++t) {
            double s = 0;
            for (int k = 0; k < 12; ++k) s += double(v[std::size_t(std::clamp(2 * t + k - 5, 0, 2 * N - 1))]) * a.down[k];
            y[std::size_t(c) * N + t] = float(s);
        }
    });
    return y;
}

Sig amp(const Amp& b, const Sig& x0, int C, int N) {
    Sig x = x0;
    for (int i = 0; i < 3; ++i) {
        Sig t = activation(b.a[2 * i], x, C, N);
        t = conv(b.c1[i], t, N);
        t = activation(b.a[2 * i + 1], t, C, N);
        t = conv(b.c2[i], t, N);
        for (std::size_t q = 0; q < x.size(); ++q) x[q] = t[q] + x[q];
    }
    return x;
}

void check_latent(const std::vector<float>& z, int T) {
    if (T < 1 || T > (1 << 20)) throw std::invalid_argument("h3 audio decoder: latent frames T must be >= 1");
    if (z.size() != std::size_t(32) * 2 * std::size_t(T)) throw std::invalid_argument("h3 audio decoder: latent length != 32 x 2 x T");
}

std::vector<float> decode(const Weights& w, const std::vector<float>& z, int T, std::vector<std::vector<Sig>>* taps) {
    check_latent(z, T);
    const int L = 800 * T;
    std::vector<float> wave(std::size_t(2) * L);
    if (taps) taps->assign(2, {});
    for (int r = 0; r < 2; ++r) {
        Sig x(std::size_t(32) * T);
        for (int c = 0; c < 32; ++c)
            for (int t = 0; t < T; ++t)
                x[std::size_t(c) * T + t] = z[(std::size_t(c) * 2 + r) * T + t] * w.latents_std[std::size_t(c)] + w.latents_mean[std::size_t(c)];
        int N = T;
        x = conv(w.dec_in, x, N);
        x = conv(w.conv_pre, x, N);
        int C = w.conv_pre.cout;
        for (int i = 0; i < 7; ++i) {
            x = conv(w.ups[i], x, N);
            N = conv_out_len(w.ups[i], N);
            C = w.ups[i].cout;
            const Sig r0 = amp(w.amp[3 * i], x, C, N), r1 = amp(w.amp[3 * i + 1], x, C, N), r2 = amp(w.amp[3 * i + 2], x, C, N);
            for (std::size_t q = 0; q < x.size(); ++q) x[q] = ((r0[q] + r1[q]) + r2[q]) / 3.0F;
            if (taps) (*taps)[std::size_t(r)].push_back(x);
        }
        x = activation(w.act_post, x, C, N);
        x = conv(w.conv_post, x, N);
        for (int t = 0; t < L; ++t) wave[std::size_t(r) * L + t] = std::clamp(x[std::size_t(t)], -1.0F, 1.0F);
    }
    return wave;
}

}  // namespace strix::h3::vae::audio
