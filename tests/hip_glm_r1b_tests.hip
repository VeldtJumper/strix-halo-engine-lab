// R1b identity experiment (approved arithmetic contract), router chain: gate + error report.
// Gate (fail closed): device r1b::router_dense == host::dense_r1b bit for bit, rows R = 1..8, each row of an R-row call
// == the same row alone; host route(..., kR1b) ids / weights == route_v2 on the device R1b logits.
// Report (informational, both identities vs float64): per-logit |error| of R1 (arithmetic contract chain) and R1b against the
// float64 dot of the same F16 weights and fp32 inputs (max / mean, relative to the row's max |logit|), and the top-8
// route flips of each identity vs the float64 route over NROWS random rows.
// usage: hip_glm_r1b_tests CASES_DIR [NROWS [R1B_CASES XROWS.f32 NREP]]
//   CASES_DIR/moe_l3: real L3 router + bias (tools/glm/export_host_case.py)
//   R1B_CASES/L<l>: the allowlist tensors of layer l (tools/glm/omath/export_r1b_cases.py); XROWS.f32: real residual
//   rows [n][16384] (e.g. a G3 output). Adds the mHC hc_pre and KDA f_a / g_a / b gates (device == host bitwise, R 1..8,
//   row independence; dense v2/v4/v5/v6 and hc v1/v2/v3/v4) and the per-layer report: every R1b chain kind, R1 and R1b vs float64 on NREP real rows
//   (|z - z64| / sum|x w| max / rms, |z - z64| / max|z64|, share of chains where R1b is closer), hc mix (pre/post/comb)
//   max |d| vs the float64 R6 chain, router top-8 flips vs the float64 route.
#include "glm/engine/glm_r1b.hpp"
#include "glm/kernels/glm_tier_a.hpp"
#include "glm/attn/glm_attn_fast.hpp"
#include "glm/host/glm_host.hpp"
#include "glm/xfer/glm_post_fast.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace strix::glm;
namespace H = strix::glm::host;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "OK  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}
void hok(hipError_t e, const char* w) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, hipGetErrorString(e));
        std::exit(3);
    }
}
template <class T> T* up(const std::vector<T>& h) {
    T* p = nullptr;
    hok(hipMalloc(&p, std::max<std::size_t>(h.size(), 1) * sizeof(T)), "malloc");
    hok(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "h2d");
    return p;
}
template <class T> std::vector<T> down(const T* p, std::size_t n) {
    std::vector<T> h(n);
    hok(hipDeviceSynchronize(), "sync");
    hok(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "d2h");
    return h;
}
std::vector<float> rnd(std::size_t n, std::mt19937& g, float scale) {
    std::normal_distribution<float> d(0.0F, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g);
    return v;
}
std::vector<char> read_bin(const std::string& dir, const std::string& name, std::size_t bytes) {
    std::vector<char> b(bytes);
    std::ifstream f(dir + "/" + name + ".bin", std::ios::binary);
    f.read(b.data(), static_cast<std::streamsize>(b.size()));
    if (static_cast<std::size_t>(f.gcount()) != b.size()) throw std::runtime_error("short " + dir + "/" + name);  // fail closed
    return b;
}

// float64 route on float64 logits (sigmoid in double, same selection key order: score desc, id asc)
std::vector<std::uint32_t> route64(const std::vector<double>& lg, const std::vector<float>& bias) {
    std::vector<std::pair<double, std::uint32_t>> c(kExperts);
    for (unsigned e = 0; e < kExperts; ++e) c[e] = {1.0 / (1.0 + std::exp(-lg[e])) + bias[e], e};
    std::partial_sort(c.begin(), c.begin() + kTopE, c.end(), [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    std::vector<std::uint32_t> ids(kTopE);
    for (unsigned k = 0; k < kTopE; ++k) ids[k] = c[k].second;
    std::sort(ids.begin(), ids.end());
    return ids;
}


// ---------------------------------------------------------------- R1b allowlist: hc / KDA gates + per-layer report
std::vector<char> read_file(const std::string& path, std::size_t bytes) {
    std::vector<char> b(bytes);
    std::ifstream f(path, std::ios::binary);
    f.read(b.data(), static_cast<std::streamsize>(b.size()));
    if (static_cast<std::size_t>(f.gcount()) != b.size()) throw std::runtime_error("short " + path);  // fail closed
    return b;
}
template <class T> std::vector<T> load(const std::string& dir, const std::string& name, std::size_t n) {
    const auto b = read_bin(dir, name, n * sizeof(T));
    std::vector<T> v(n);
    std::memcpy(v.data(), b.data(), b.size());
    return v;
}
bool exists(const std::string& path) { return std::ifstream(path).good(); }

// error statistics of one chain kind: per output e = |z - z64| / sum_i |x_i w_i|
struct Stat {
    double max1 = 0, maxb = 0, ss1 = 0, ssb = 0, rmax1 = 0, rmaxb = 0;
    unsigned long n = 0, better = 0, worse = 0;
    void add(float z1, float zb, double z64, double a, double zmax) {
        const double d1 = std::fabs(double(z1) - z64), db = std::fabs(double(zb) - z64);
        const double e1 = a > 0 ? d1 / a : 0, eb = a > 0 ? db / a : 0;
        max1 = std::max(max1, e1);
        maxb = std::max(maxb, eb);
        ss1 += e1 * e1;
        ssb += eb * eb;
        if (zmax > 0) {
            rmax1 = std::max(rmax1, d1 / zmax);
            rmaxb = std::max(rmaxb, db / zmax);
        }
        better += db < d1;
        worse += db > d1;
        ++n;
    }
    void print(const char* what, unsigned layer) const {
        std::printf("err L%u %-9s n %7lu | /sum|xw|: R1 max %.3e rms %.3e | R1b max %.3e rms %.3e | /max|z|: R1 %.3e R1b %.3e | "
                    "R1b closer %.1f %% farther %.1f %% equal %.1f %%\n",
                    layer, what, n, max1, std::sqrt(ss1 / double(n)), maxb, std::sqrt(ssb / double(n)), rmax1, rmaxb,
                    100.0 * double(better) / double(n), 100.0 * double(worse) / double(n),
                    100.0 * double(n - better - worse) / double(n));
    }
};

// one chain group: z1 (R1), zb (R1b), z64 (float64 of the same float operands), A = sum |x w|
template <class W>
void chains(const float* x, unsigned K, W w, unsigned n, std::vector<float>& z1, std::vector<float>& zb, std::vector<double>& z64,
            std::vector<double>& A) {
    z1.resize(n);
    zb.resize(n);
    z64.resize(n);
    A.resize(n);
    for (unsigned o = 0; o < n; ++o) {
        auto a = [&](unsigned i) { return x[i]; };
        auto b = [&](unsigned i) { return w(o, i); };
        z1[o] = chain_dot(a, b, K);
        zb[o] = r1b_dot(a, b, K);
        double s = 0, sa = 0;
        for (unsigned i = 0; i < K; ++i) {
            const double p = double(x[i]) * double(w(o, i));
            s += p;
            sa += std::fabs(p);
        }
        z64[o] = s;
        A[o] = sa;
    }
}
void add_all(Stat& st, const std::vector<float>& z1, const std::vector<float>& zb, const std::vector<double>& z64, const std::vector<double>& A) {
    double zm = 0;
    for (double v : z64) zm = std::max(zm, std::fabs(v));
    for (std::size_t o = 0; o < z1.size(); ++o) st.add(z1[o], zb[o], z64[o], A[o], zm);
}

// float64 R6 (same operation order as host hc_mix, in double) on float64 chain values
std::array<double, kHcMix> mix64(const std::vector<double>& z, const float* base, const float* scale) {
    auto sg = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };
    std::array<double, kHcMix> m{};
    for (unsigned j = 0; j < kHc; ++j) m[j] = sg(z[j] * scale[0] + base[j]) + double(kHcEps);
    for (unsigned j = 0; j < kHc; ++j) m[kHc + j] = 2.0 * sg(z[kHc + j] * scale[1] + base[kHc + j]);
    double* c = m.data() + 2 * kHc;
    for (unsigned k = 0; k < kHc * kHc; ++k) c[k] = z[2 * kHc + k] * scale[2] + base[2 * kHc + k];
    for (unsigned j = 0; j < kHc; ++j) {
        double mx = c[j * kHc];
        for (unsigned k = 1; k < kHc; ++k) mx = std::max(mx, c[j * kHc + k]);
        double e[kHc], sum = 0;
        for (unsigned k = 0; k < kHc; ++k) sum += e[k] = std::exp(c[j * kHc + k] - mx);
        for (unsigned k = 0; k < kHc; ++k) c[j * kHc + k] = e[k] / sum + double(kHcEps);
    }
    auto cols = [&] {
        for (unsigned k = 0; k < kHc; ++k) {
            const double s = c[k] + c[kHc + k] + c[2 * kHc + k] + c[3 * kHc + k] + double(kHcEps);
            for (unsigned j = 0; j < kHc; ++j) c[j * kHc + k] /= s;
        }
    };
    auto rows = [&] {
        for (unsigned j = 0; j < kHc; ++j) {
            const double s = c[j * kHc] + c[j * kHc + 1] + c[j * kHc + 2] + c[j * kHc + 3] + double(kHcEps);
            for (unsigned k = 0; k < kHc; ++k) c[j * kHc + k] /= s;
        }
    };
    cols();
    for (unsigned it = 1; it < kHcIters; ++it) {
        rows();
        cols();
    }
    return m;
}
std::array<float, kHcMix> pack(const H::HcMix& m) {
    std::array<float, kHcMix> a{};
    for (unsigned j = 0; j < kHc; ++j) {
        a[j] = m.pre[j];
        a[kHc + j] = m.post[j];
    }
    for (unsigned k = 0; k < kHc * kHc; ++k) a[2 * kHc + k] = m.comb[k];
    return a;
}

struct LayerCase {
    unsigned l = 0;
    bool kda = false, moe = false;
    std::vector<float> fn[2], base[2], scale[2], in_norm, post_norm, bias;
    std::vector<std::uint16_t> router, f_a, g_a, b;
};
LayerCase load_layer(const std::string& root, unsigned l) {
    LayerCase c;
    c.l = l;
    const std::string d = root + "/L" + std::to_string(l);
    const char* site[2] = {"hc_attn_", "hc_ffn_"};
    for (int s = 0; s < 2; ++s) {
        c.fn[s] = load<float>(d, std::string(site[s]) + "fn", static_cast<std::size_t>(kHcMix) * kHc * kHidden);
        c.base[s] = load<float>(d, std::string(site[s]) + "base", kHcMix);
        c.scale[s] = load<float>(d, std::string(site[s]) + "scale", 3);
    }
    c.in_norm = load<float>(d, "in_norm", kHidden);
    c.post_norm = load<float>(d, "post_norm", kHidden);
    c.moe = exists(d + "/router.bin");
    c.kda = exists(d + "/f_a.bin");
    if (c.moe) {
        c.router = load<std::uint16_t>(d, "router", static_cast<std::size_t>(kExperts) * kHidden);
        c.bias = load<float>(d, "bias", kExperts);
    }
    if (c.kda) {
        c.f_a = load<std::uint16_t>(d, "f_a", static_cast<std::size_t>(kKdaDim) * kHidden);
        c.g_a = load<std::uint16_t>(d, "g_a", static_cast<std::size_t>(kKdaDim) * kHidden);
        c.b = load<std::uint16_t>(d, "b", static_cast<std::size_t>(kKdaHeads) * kHidden);
    }
    return c;
}
// the attention / FFN input of a row: rmsnorm(collapse(X, mix(R1)), norm) (identical operands for both identities)
std::vector<float> site_input(const float* X, const LayerCase& c, int s) {
    const H::HcWeights hw{H::Mat{c.fn[s].data(), H::WType::F32, kHcMix, kHc * kHidden}, c.base[s].data(), c.scale[s].data()};
    const auto m = H::hc_mix({X, kHc * kHidden}, hw);
    std::vector<float> xc(kHidden), xn(kHidden);
    H::hc_collapse({X, kHc * kHidden}, m, xc);
    const H::Vec nw{s == 0 ? c.in_norm.data() : c.post_norm.data(), H::WType::F32, kHidden};
    H::rmsnorm(xc, &nw, kEps, xn);
    return xn;
}

template <class A, class B> bool same_bits(const A& a, const B& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0;
}

void gates_variant(const LayerCase& c, const std::vector<float>& Xall, int dense_variant, int hc_variant) {
    const H::HcWeights hw{H::Mat{c.fn[0].data(), H::WType::F32, kHcMix, kHc * kHidden}, c.base[0].data(), c.scale[0].data()};
    float* dfn = up(c.fn[0]);
    float* dbase = up(c.base[0]);
    float* dscale = up(c.scale[0]);
    bool hc_ok = true, hc_indep = true, kda_ok = true, kda_indep = true, hc_variants = true;
    const std::size_t N = kHc * kHidden;
    for (unsigned R = 1; R <= 8; ++R) {
        const auto off = static_cast<std::ptrdiff_t>(static_cast<std::size_t>(R - 1) * N);
        std::vector<float> X(Xall.begin() + off, Xall.begin() + off + static_cast<std::ptrdiff_t>(R * N));
        float* dX = up(X);
        float* dF = up(std::vector<float>(R * N, std::nanf("")));
        float* dmix = up(std::vector<float>(R * kHcMix, std::nanf("")));
        float* dxc = up(std::vector<float>(static_cast<std::size_t>(R) * kHidden, std::nanf("")));
        r1b::hc_pre(dX, dfn, dbase, dscale, dF, dmix, dxc, R, nullptr, hc_variant);
        const auto F = down(dF, R * N);
        const auto mx = down(dmix, R * kHcMix);
        const auto xc = down(dxc, static_cast<std::size_t>(R) * kHidden);
        for (int other : {1, 2, 3, 4}) for (int rep = 0; rep < 3; ++rep) {  // all finishes, repeated; compare bytes
            float* dF2 = up(std::vector<float>(R * N, std::nanf("")));
            float* dm2 = up(std::vector<float>(R * kHcMix, std::nanf("")));
            float* dx2 = up(std::vector<float>(static_cast<std::size_t>(R) * kHidden, std::nanf("")));
            r1b::hc_pre(dX, dfn, dbase, dscale, dF2, dm2, dx2, R, nullptr, other);
            hc_variants = hc_variants && same_bits(down(dF2, R * N), F) && same_bits(down(dm2, R * kHcMix), mx) &&
                          same_bits(down(dx2, static_cast<std::size_t>(R) * kHidden), xc);
            for (void* p : {static_cast<void*>(dF2), static_cast<void*>(dm2), static_cast<void*>(dx2)}) hok(hipFree(p), "free");
        }
        for (unsigned r = 0; r < R; ++r) {
            const float* Xr = X.data() + r * N;
            std::vector<float> hF(N), hxc(kHidden);
            H::rmsnorm({Xr, N}, nullptr, kEps, hF);
            const auto m = H::hc_mix({Xr, N}, hw, Identity::kR1b);
            H::hc_collapse({Xr, N}, m, hxc);
            const auto pm = pack(m);
            hc_ok = hc_ok && std::memcmp(hF.data(), F.data() + r * N, N * 4) == 0 &&
                    std::memcmp(pm.data(), mx.data() + r * kHcMix, kHcMix * 4) == 0 &&
                    std::memcmp(hxc.data(), xc.data() + static_cast<std::size_t>(r) * kHidden, kHidden * 4) == 0;
            float* dX1 = up(std::vector<float>(Xr, Xr + N));
            float* dm1 = up(std::vector<float>(kHcMix));
            float* dx1 = up(std::vector<float>(kHidden));
            float* dF1 = up(std::vector<float>(N));
            r1b::hc_pre(dX1, dfn, dbase, dscale, dF1, dm1, dx1, 1, nullptr, hc_variant);
            const auto m1 = down(dm1, kHcMix);
            const auto x1 = down(dx1, kHidden);
            hc_indep = hc_indep && same_bits(hF, down(dF1, N)) && std::memcmp(m1.data(), mx.data() + r * kHcMix, kHcMix * 4) == 0 &&
                       std::memcmp(x1.data(), xc.data() + static_cast<std::size_t>(r) * kHidden, kHidden * 4) == 0;
            for (void* p : {static_cast<void*>(dX1), static_cast<void*>(dm1), static_cast<void*>(dx1), static_cast<void*>(dF1)}) hok(hipFree(p), "free");
        }
        if (c.kda) {  // f_a, g_a (128), b rank 0 / rank 1 rows (32 each) on the attention inputs of these rows
            std::vector<float> xs;
            for (unsigned r = 0; r < R; ++r) {
                const auto xn = site_input(X.data() + r * N, c, 0);
                xs.insert(xs.end(), xn.begin(), xn.end());
            }
            float* dx = up(xs);
            struct J {
                const std::vector<std::uint16_t>* w;
                unsigned row0, n;
            };
            const J js[5] = {{&c.f_a, 0, kKdaDim}, {&c.g_a, 0, kKdaDim}, {&c.b, 0, kKdaHeads / 2},
                             {&c.b, kKdaHeads / 2, kKdaHeads / 2}, {&c.b, 0, kKdaHeads}};
            for (const J& j : js) {
                std::uint16_t* dW = up(*j.w);
                const std::uint16_t* dWr = dW + static_cast<std::size_t>(j.row0) * kHidden;
                float* dy = up(std::vector<float>(static_cast<std::size_t>(R) * j.n, std::nanf("")));
                r1b::dense_f16_k4096(dx, kHidden, dWr, j.n, dy, j.n, R, nullptr, dense_variant);
                const auto y = down(dy, static_cast<std::size_t>(R) * j.n);
                r1b::dense_f16_k4096(dx, kHidden, dWr, j.n, dy, j.n, R, nullptr, 2);
                kda_ok = kda_ok && same_bits(y, down(dy, static_cast<std::size_t>(R) * j.n));
                const H::Mat Wm{j.w->data() + static_cast<std::size_t>(j.row0) * kHidden, H::WType::F16, j.n, kHidden};
                for (unsigned r = 0; r < R; ++r) {
                    std::vector<float> h(j.n);
                    H::dense_r1b({xs.data() + static_cast<std::size_t>(r) * kHidden, kHidden}, Wm, h);
                    kda_ok = kda_ok && std::memcmp(h.data(), y.data() + static_cast<std::size_t>(r) * j.n, j.n * 4) == 0;
                    float* d1 = up(std::vector<float>(j.n));
                    r1b::dense_f16_k4096(dx + static_cast<std::size_t>(r) * kHidden, kHidden, dWr, j.n, d1, j.n, 1, nullptr, dense_variant);
                    const auto one = down(d1, j.n);
                    kda_indep = kda_indep && std::memcmp(one.data(), y.data() + static_cast<std::size_t>(r) * j.n, j.n * 4) == 0;
                    hok(hipFree(d1), "free");
                }
                hok(hipFree(dW), "free");
                hok(hipFree(dy), "free");
            }
            // Three-job calls: both rank slices and the complete b projection.
            for (unsigned shape = 0; shape < 3; ++shape) {
                std::uint16_t* dA = up(c.f_a);
                std::uint16_t* dG = up(c.g_a);
                std::uint16_t* dB = up(c.b);
                const unsigned b_rows = shape == 2 ? kKdaHeads : kKdaHeads / 2;
                const unsigned b_offset = shape == 1 ? kKdaHeads / 2 : 0U;
                const unsigned Nn[3] = {kKdaDim, kKdaDim, b_rows};
                std::vector<float*> ys, ys1;
                for (unsigned q = 0; q < 3; ++q) {
                    ys.push_back(up(std::vector<float>(static_cast<std::size_t>(R) * Nn[q], std::nanf(""))));
                    ys1.push_back(up(std::vector<float>(static_cast<std::size_t>(R) * Nn[q], std::nanf(""))));
                }
                const std::uint16_t* const W[3] = {dA, dG, dB + static_cast<std::size_t>(b_offset) * kHidden};
                float* const Y[3] = {ys[0], ys[1], ys[2]};
                r1b::dense_f16_k4096_x3(dx, kHidden, W, Nn, Y, Nn, R, nullptr, dense_variant);
                std::vector<std::vector<float>> candidate;
                for (unsigned q = 0; q < 3; ++q) candidate.push_back(down(ys[q], static_cast<std::size_t>(R) * Nn[q]));
                r1b::dense_f16_k4096_x3(dx, kHidden, W, Nn, Y, Nn, R, nullptr, 2);
                for (unsigned q = 0; q < 3; ++q) {
                    r1b::dense_f16_k4096(dx, kHidden, W[q], Nn[q], ys1[q], Nn[q], R, nullptr, 2);
                    const auto a = down(ys[q], static_cast<std::size_t>(R) * Nn[q]);
                    const auto b = down(ys1[q], static_cast<std::size_t>(R) * Nn[q]);
                    kda_ok = kda_ok && same_bits(candidate[q], a) && same_bits(candidate[q], b);
                    const auto& weights = q == 0 ? c.f_a : (q == 1 ? c.g_a : c.b);
                    const unsigned row0 = q == 2 ? b_offset : 0U;
                    const H::Mat Wm{weights.data() + static_cast<std::size_t>(row0) * kHidden, H::WType::F16, Nn[q], kHidden};
                    for (unsigned r = 0; r < R; ++r) {
                        std::vector<float> h(Nn[q]);
                        H::dense_r1b({xs.data() + static_cast<std::size_t>(r) * kHidden, kHidden}, Wm, h);
                        kda_ok = kda_ok && std::memcmp(h.data(), candidate[q].data() + static_cast<std::size_t>(r) * Nn[q],
                                                    Nn[q] * sizeof(float)) == 0;
                    }
                    hok(hipFree(ys[q]), "free");
                    hok(hipFree(ys1[q]), "free");
                }
                for (void* p : {static_cast<void*>(dA), static_cast<void*>(dG), static_cast<void*>(dB)}) hok(hipFree(p), "free");
            }
            hok(hipFree(dx), "free");
        }
        if (c.moe) {
            std::vector<float> xs;
            for (unsigned r = 0; r < R; ++r) {
                const auto xn = site_input(X.data() + r * N, c, 1);
                xs.insert(xs.end(), xn.begin(), xn.end());
            }
            float* dx = up(xs);
            auto* dW = up(c.router);
            float* dy = up(std::vector<float>(static_cast<std::size_t>(R) * kExperts));
            r1b::router_dense(dx, dW, dy, R, nullptr, dense_variant);
            const auto y = down(dy, static_cast<std::size_t>(R) * kExperts);
            r1b::router_dense(dx, dW, dy, R, nullptr, 2);
            bool exact = same_bits(y, down(dy, y.size()));
            const H::Mat Wm{c.router.data(), H::WType::F16, kExperts, kHidden};
            for (unsigned r = 0; r < R; ++r) {
                std::vector<float> h(kExperts);
                H::dense_r1b({xs.data() + static_cast<std::size_t>(r) * kHidden, kHidden}, Wm, h);
                exact = exact && std::memcmp(h.data(), y.data() + static_cast<std::size_t>(r) * kExperts, kExperts * sizeof(float)) == 0;
                r1b::router_dense(dx + static_cast<std::size_t>(r) * kHidden, dW, dy, 1, nullptr, dense_variant);
                exact = exact && same_bits(h, down(dy, kExperts));
            }
            check(exact, "R1b router real residual: variant == host == v2, row independence, R=" + std::to_string(R));
            hok(hipFree(dx), "free");
            hok(hipFree(dW), "free");
            hok(hipFree(dy), "free");
        }
        for (void* p : {static_cast<void*>(dX), static_cast<void*>(dF), static_cast<void*>(dmix), static_cast<void*>(dxc)}) hok(hipFree(p), "free");
    }
    const std::string L = " (real L" + std::to_string(c.l) + " dense v" + std::to_string(dense_variant) + " hc v" +
                          std::to_string(hc_variant) + " weights, real residual rows, R = 1..8)";
    check(hc_ok, "R1b hc_pre: device F, mix, xc == host rmsnorm / hc_mix(kR1b) / hc_collapse bitwise" + L);
    check(hc_indep, "R1b hc_pre: row r of an R-row call == the row alone" + L);
    check(hc_variants, "R1b hc_pre variants 1/2/3/4 agree bitwise, 3 repeats" + L);
    if (c.kda) {
        check(kda_ok, "R1b KDA f_a / g_a / b (rank 0 + rank 1 rows): device == host dense_r1b bitwise, 3-job call == single calls" + L);
        check(kda_indep, "R1b KDA f_a / g_a / b: row r of an R-row call == the row alone" + L);
    }
    for (void* p : {static_cast<void*>(dfn), static_cast<void*>(dbase), static_cast<void*>(dscale)}) hok(hipFree(p), "free");
}

// Make packed device copies outside the timed regions; packing is asynchronous on the same stream.
std::uint16_t* packed_copy(const std::uint16_t* w, unsigned n) {
    std::uint16_t* packed = nullptr;
    hok(hipMalloc(&packed, static_cast<std::size_t>(n) * kHidden * sizeof(std::uint16_t)), "packed malloc");
    r1b::pack_k4096(w, n, packed, nullptr);
    return packed;
}

void gate_pack_bits() {
    // All 65536 encodings, including signed zero, subnormal, infinity and NaN payloads.
    std::vector<std::uint16_t> bits(16 * kHidden);
    for (std::size_t i = 0; i < bits.size(); ++i) bits[i] = static_cast<std::uint16_t>(i);
    auto* w = up(bits);
    auto* packed = packed_copy(w, 16);
    const auto ph = down(packed, bits.size());
    bool exact = true;
    for (std::size_t v = 0; v < bits.size() / 8; ++v) {
        const std::size_t j = v % 16, o = (v / 16) % 16, c8 = v / 256;
        for (unsigned k = 0; k < 8; ++k)
            exact = exact && ph[v * 8 + k] == bits[o * kHidden + j * 256 + c8 * 8 + k];
    }
    check(exact, "R1b pack_k4096 preserves all 65536 F16 encodings");
    hok(hipFree(packed), "free");
    hok(hipFree(w), "free");
}

void gate_packed_dense(const LayerCase& c, const std::vector<float>& Xall) {
    const auto xs = site_input(Xall.data(), c, c.kda ? 0 : 1);
    float* dx = up(xs);
    bool exact = true, permutation = true;
    auto gate = [&](const std::vector<std::uint16_t>& weights, unsigned row0, unsigned n) {
        std::vector<std::uint16_t> slice(weights.begin() + static_cast<std::ptrdiff_t>(row0 * kHidden),
                                         weights.begin() + static_cast<std::ptrdiff_t>((row0 + n) * kHidden));
        auto* w = up(slice);
        auto* packed = packed_copy(w, n);
        const auto ph = down(packed, slice.size());
        for (unsigned c8 = 0; c8 < 32; ++c8)
            for (unsigned o = 0; o < n; ++o)
                for (unsigned j = 0; j < 16; ++j)
                    for (unsigned k = 0; k < 8; ++k)
                        permutation = permutation &&
                            ph[((static_cast<std::size_t>(c8) * n + o) * 16 + j) * 8 + k] ==
                            slice[static_cast<std::size_t>(o) * kHidden + j * 256 + c8 * 8 + k];
        float* y = up(std::vector<float>(n, std::nanf("")));
        r1b::dense_f16_k4096(dx, kHidden, packed, n, y, n, 1, nullptr, 5);
        const auto candidate = down(y, n);
        std::vector<float> oracle(n);
        const H::Mat wm{slice.data(), H::WType::F16, n, kHidden};
        H::dense_r1b({xs.data(), xs.size()}, wm, oracle);
        exact = exact && same_bits(candidate, oracle);
        for (int v : {2, 4}) {
            r1b::dense_f16_k4096(dx, kHidden, w, n, y, n, 1, nullptr, v);
            exact = exact && same_bits(candidate, down(y, n));
        }
        hok(hipFree(w), "free");
        hok(hipFree(packed), "free");
        hok(hipFree(y), "free");
    };
    if (c.moe) gate(c.router, 0, kExperts);
    if (c.kda) {
        gate(c.f_a, 0, kKdaDim);
        gate(c.g_a, 0, kKdaDim);
        for (unsigned shape = 0; shape < 3; ++shape) {
            const unsigned n[3] = {kKdaDim, kKdaDim, shape == 2 ? kKdaHeads : kKdaHeads / 2};
            const unsigned row0 = shape == 1 ? kKdaHeads / 2 : 0U;
            gate(c.b, row0, n[2]);
            const std::uint16_t* w[3] = {up(c.f_a), up(c.g_a), up(c.b)};
            const std::uint16_t* original[3] = {w[0], w[1], w[2] + static_cast<std::size_t>(row0) * kHidden};
            const std::uint16_t* packed[3] = {packed_copy(original[0], n[0]), packed_copy(original[1], n[1]),
                                             packed_copy(original[2], n[2])};
            float* y[3] = {up(std::vector<float>(n[0])), up(std::vector<float>(n[1])), up(std::vector<float>(n[2]))};
            r1b::dense_f16_k4096_x3(dx, kHidden, packed, n, y, n, 1, nullptr, 5);
            std::vector<float> candidate[3] = {down(y[0], n[0]), down(y[1], n[1]), down(y[2], n[2])};
            for (int v : {2, 4}) {
                r1b::dense_f16_k4096_x3(dx, kHidden, original, n, y, n, 1, nullptr, v);
                for (unsigned q = 0; q < 3; ++q) exact = exact && same_bits(candidate[q], down(y[q], n[q]));
            }
            for (unsigned q = 0; q < 3; ++q) {
                const auto& weights = q == 0 ? c.f_a : (q == 1 ? c.g_a : c.b);
                const H::Mat wm{weights.data() + (q == 2 ? static_cast<std::size_t>(row0) * kHidden : 0),
                                H::WType::F16, n[q], kHidden};
                std::vector<float> oracle(n[q]);
                H::dense_r1b({xs.data(), xs.size()}, wm, oracle);
                exact = exact && same_bits(candidate[q], oracle);
                hok(hipFree(const_cast<std::uint16_t*>(w[q])), "free");
                hok(hipFree(const_cast<std::uint16_t*>(packed[q])), "free");
                hok(hipFree(y[q]), "free");
            }
        }
        // Tail outputs still participate in all subgroup shuffles.
        gate(c.f_a, 0, 1);
        gate(c.f_a, 0, 9);
    }
    check(permutation, "R1b v5 full device permutation preserves F16 bits");
    check(exact, "R1b dense v5 == host == v2 == v4, R=1, single/x3, rank slices and tail outputs");
    hok(hipFree(dx), "free");
}

// Real model weights and site inputs derived from real residual rows. Exercise
// every R specialization, both b rank slices, and partially active subgroups.
void gate_packed_rows(const LayerCase& c, const std::vector<float>& Xall, int selected) {
    bool exact = true, independent = true, repeated = true, padding = true;
    auto gate = [&](const std::vector<std::uint16_t>* const weights[3], const unsigned row0[3],
                    const unsigned sizes[3], unsigned jobs, bool router) {
        const std::uint16_t* original[3] = {};
        const std::uint16_t* packed[3] = {};
        for (unsigned q = 0; q < jobs; ++q) {
            const auto first = weights[q]->begin() + static_cast<std::ptrdiff_t>(row0[q]) * kHidden;
            original[q] = up(std::vector<std::uint16_t>(first, first + static_cast<std::ptrdiff_t>(sizes[q]) * kHidden));
            packed[q] = packed_copy(original[q], sizes[q]);
        }
        // Router has fixed strides; single/x3 additionally test padded input/output strides.
        const unsigned ldx = kHidden + (router ? 0U : 5U);
        const unsigned ldy[3] = {sizes[0] + (router ? 0U : 3U), sizes[1] + 3U, sizes[2] + 3U};
        constexpr float sentinel = -12345.0F;
        for (unsigned R = 1; R <= 8; ++R) {
            std::vector<float> xs(static_cast<std::size_t>(R) * ldx, sentinel);
            for (unsigned r = 0; r < R; ++r) {
                const auto xn = site_input(Xall.data() + static_cast<std::size_t>(R - 1 + r) * kHc * kHidden,
                                           c, router ? 1 : 0);
                std::copy(xn.begin(), xn.end(), xs.begin() + static_cast<std::ptrdiff_t>(r) * ldx);
            }
            float* dx = up(xs);
            float* y[3] = {};
            std::vector<float> oracle[3], candidate[3];
            for (unsigned q = 0; q < jobs; ++q) {
                oracle[q].assign(static_cast<std::size_t>(R) * ldy[q], sentinel);
                const H::Mat wm{weights[q]->data() + static_cast<std::size_t>(row0[q]) * kHidden,
                                H::WType::F16, sizes[q], kHidden};
                for (unsigned r = 0; r < R; ++r) {
                    std::vector<float> h(sizes[q]);
                    H::dense_r1b({xs.data() + static_cast<std::size_t>(r) * ldx, kHidden}, wm, h);
                    std::copy(h.begin(), h.end(), oracle[q].begin() + static_cast<std::ptrdiff_t>(r) * ldy[q]);
                }
                y[q] = up(std::vector<float>(oracle[q].size(), sentinel));
            }
            auto run = [&](int variant) {
                const auto* w = variant >= 5 ? packed : original;
                if (router) r1b::router_dense(dx, w[0], y[0], R, nullptr, variant);
                else if (jobs == 3) r1b::dense_f16_k4096_x3(dx, ldx, w, sizes, y, ldy, R, nullptr, variant);
                else r1b::dense_f16_k4096(dx, ldx, w[0], sizes[0], y[0], ldy[0], R, nullptr, variant);
            };
            for (unsigned rep = 0; rep < 3; ++rep) {
                run(selected);
                for (unsigned q = 0; q < jobs; ++q) {
                    const auto got = down(y[q], oracle[q].size());
                    exact = same_bits(got, oracle[q]) && exact;
                    if (rep == 0) candidate[q] = got;
                    else repeated = same_bits(got, candidate[q]) && repeated;
                    for (unsigned r = 0; r < R; ++r)
                        for (unsigned o = sizes[q]; o < ldy[q]; ++o)
                            padding = (got[static_cast<std::size_t>(r) * ldy[q] + o] == sentinel) && padding;
                }
                for (int v : {2, 4}) {
                    run(v);
                    for (unsigned q = 0; q < jobs; ++q)
                        exact = same_bits(candidate[q], down(y[q], oracle[q].size())) && exact;
                }
                for (unsigned r = 0; r < R; ++r) {
                    float* one[3] = {};
                    for (unsigned q = 0; q < jobs; ++q) one[q] = up(std::vector<float>(ldy[q], sentinel));
                    const float* xr = dx + static_cast<std::size_t>(r) * ldx;
                    if (router) r1b::router_dense(xr, packed[0], one[0], 1, nullptr, selected);
                    else if (jobs == 3) r1b::dense_f16_k4096_x3(xr, ldx, packed, sizes, one, ldy, 1, nullptr, selected);
                    else r1b::dense_f16_k4096(xr, ldx, packed[0], sizes[0], one[0], ldy[0], 1, nullptr, selected);
                    for (unsigned q = 0; q < jobs; ++q) {
                        const auto h = down(one[q], ldy[q]);
                        independent = (std::memcmp(h.data(), candidate[q].data() + static_cast<std::size_t>(r) * ldy[q],
                                                   ldy[q] * sizeof(float)) == 0) && independent;
                        // v5's row-alone packed path must have the same bits too.
                        r1b::dense_f16_k4096(xr, ldx, packed[q], sizes[q], one[q], ldy[q], 1, nullptr, 5);
                        exact = same_bits(h, down(one[q], ldy[q])) && exact;
                        hok(hipFree(one[q]), "free");
                    }
                }
            }
            for (unsigned q = 0; q < jobs; ++q) hok(hipFree(y[q]), "free");
            hok(hipFree(dx), "free");
        }
        for (unsigned q = 0; q < jobs; ++q) {
            hok(hipFree(const_cast<std::uint16_t*>(original[q])), "free");
            hok(hipFree(const_cast<std::uint16_t*>(packed[q])), "free");
        }
    };
    const unsigned zero[3] = {};
    if (c.moe) {
        const std::vector<std::uint16_t>* const w[3] = {&c.router, nullptr, nullptr};
        const unsigned n[3] = {kExperts, 0, 0};
        gate(w, zero, n, 1, true);
    }
    if (c.kda) {
        const std::vector<std::uint16_t>* const w[3] = {&c.f_a, &c.g_a, &c.b};
        for (unsigned shape = 0; shape < 5; ++shape) {
            const unsigned row0[3] = {0, 0, shape == 1 ? kKdaHeads / 2 : 0U};
            const unsigned n[3] = {kKdaDim, kKdaDim,
                                  shape < 2 ? kKdaHeads / 2 : (shape == 2 ? kKdaHeads : (shape == 3 ? 1U : 9U))};
            gate(w, row0, n, 3, false);
        }
        for (unsigned n0 : {1U, 9U, kKdaDim}) {
            const unsigned n[3] = {n0, 0, 0};
            gate(w, zero, n, 1, false);
        }
    }
    const std::string label = " (variant " + std::to_string(selected) + ", real L" + std::to_string(c.l) + ", R=1..8, 3 repeats)";
    check(exact, "R1b dense packed auto/v6 == host == v2 == v4 == row-alone v5, router/single/x3, rank slices, b tails" + label);
    check(independent, "R1b dense packed auto/v6 row independence, including x3" + label);
    check(repeated, "R1b dense packed auto/v6 repeated calls agree bitwise" + label);
    check(padding, "R1b dense packed auto/v6 preserves output stride padding" + label);
}

// Real hc_attn and hc_ffn residuals, exact production BF16 weights, padded buffers.
void gate_hc_norm(const LayerCase& c, const std::vector<float>& Xall) {
    constexpr std::size_t N = kHc * kHidden;
    for (unsigned site = 0; site < 2; ++site) {
        auto* fn = up(c.fn[site]);
        auto* base = up(c.base[site]);
        auto* scale = up(c.scale[site]);
        const auto& nw = site == 0 ? c.in_norm : c.post_norm;
        std::vector<std::uint16_t> nb(kHidden);
        for (unsigned i = 0; i < kHidden; ++i) nb[i] = float_to_bf16(nw[i]);
        auto* norm = up(nb);
        bool exact = true, independent = true;
        for (unsigned R = 1; R <= 8; ++R) for (unsigned perm = 0; perm < 3; ++perm) {
            // Identity, reverse, rotation. Tail input and all output padding stay NaN.
            std::vector<float> X(9 * N, std::nanf(""));
            for (unsigned r = 0; r < R; ++r) {
                const unsigned src = perm == 0 ? r : perm == 1 ? R - 1 - r : (r + 1) % R;
                std::memcpy(X.data() + r * N, Xall.data() + src * N, N * sizeof(float));
            }
            auto* dX = up(X);
            const std::vector<float> pf(9 * N, std::nanf("")), pm(9 * kHcMix, std::nanf("")), px(9 * kHidden, std::nanf(""));
            auto* F = up(pf); auto* mix = up(pm); auto* xc = up(px); auto* xn = up(px);
            r1b::hc_pre(dX, fn, base, scale, F, mix, xc, R, nullptr, 4);
            dev::rmsnorm(xc, norm, dev::WType::BF16, kHidden, kEps, xn, R, nullptr);
            const auto refF = down(F, pf.size()), refM = down(mix, pm.size()), refC = down(xc, px.size()), refN = down(xn, px.size());
            // Launch modes 0 (three launches), 1 and 2 (one-launch site), three repetitions each.
            for (unsigned rep = 0; rep < 9; ++rep) {
                hok(hipMemcpy(F, pf.data(), pf.size() * 4, hipMemcpyHostToDevice), "poison F");
                hok(hipMemcpy(mix, pm.data(), pm.size() * 4, hipMemcpyHostToDevice), "poison mix");
                hok(hipMemcpy(xc, px.data(), px.size() * 4, hipMemcpyHostToDevice), "poison xc");
                hok(hipMemcpy(xn, px.data(), px.size() * 4, hipMemcpyHostToDevice), "poison xn");
                r1b::hc_pre_norm(dX, fn, base, scale, F, mix, xc, norm, xn, R, nullptr, static_cast<int>(rep % 3));
                exact = same_bits(refF, down(F, pf.size())) && same_bits(refM, down(mix, pm.size())) &&
                        same_bits(refC, down(xc, px.size())) && same_bits(refN, down(xn, px.size())) && exact;
            }
            // Arrival counter / fence check, queued: fused-site launches back-to-back on one stream (no host sync in between),
            // alternating this input with a rotated one (perm 2's rows), each result snapshotted on the device; one final
            // sync, then every snapshot == its input's mode-0 reference. R1 mode 1 (one launch), R8 mode 2 (two launches).
            if (perm == 0 && (R == 1 || R == 8)) {
                std::vector<float> X2(9 * N, std::nanf(""));
                for (unsigned r = 0; r < R; ++r) std::memcpy(X2.data() + r * N, Xall.data() + ((r + 1) % 9) * N, N * sizeof(float));
                auto* dX2 = up(X2);
                r1b::hc_pre_norm(dX2, fn, base, scale, F, mix, xc, norm, xn, R, nullptr, 0);
                const auto ref2M = down(mix, pm.size()), ref2N = down(xn, px.size());
                const unsigned reps = R == 1 ? 1000 : 250;
                const std::size_t sm = static_cast<std::size_t>(R) * kHcMix, sn = static_cast<std::size_t>(R) * kHidden;
                float* snaps = up(std::vector<float>(reps * (sm + sn), std::nanf("")));
                for (unsigned rep = 0; rep < reps; ++rep) {
                    r1b::hc_pre_norm(rep % 2 ? dX2 : dX, fn, base, scale, F, mix, xc, norm, xn, R, nullptr, R == 1 ? 1 : 2);
                    hok(hipMemcpyAsync(snaps + rep * (sm + sn), mix, sm * 4, hipMemcpyDeviceToDevice, nullptr), "snap mix");
                    hok(hipMemcpyAsync(snaps + rep * (sm + sn) + sm, xn, sn * 4, hipMemcpyDeviceToDevice, nullptr), "snap xn");
                }
                const auto all = down(snaps, reps * (sm + sn));
                for (unsigned rep = 0; rep < reps; ++rep) {
                    const float* m = all.data() + rep * (sm + sn);
                    const auto& rm = rep % 2 ? ref2M : refM;
                    const auto& rn = rep % 2 ? ref2N : refN;
                    exact = std::memcmp(m, rm.data(), sm * 4) == 0 && std::memcmp(m + sm, rn.data(), sn * 4) == 0 && exact;
                }
                hok(hipFree(snaps), "free snaps");
                hok(hipFree(dX2), "free X2");
            }
            // Each permuted row alone must match all four batched results (every launch mode).
            for (unsigned q = 0; q < 3 * R; ++q) {
                const unsigned r = q % R;
                r1b::hc_pre_norm(dX + r * N, fn, base, scale, F, mix, xc, norm, xn, 1, nullptr, static_cast<int>(q / R));
                const auto f = down(F, N), m = down(mix, kHcMix), x = down(xc, kHidden), n = down(xn, kHidden);
                independent = std::memcmp(f.data(), refF.data() + r * N, N * 4) == 0 &&
                              std::memcmp(m.data(), refM.data() + r * kHcMix, kHcMix * 4) == 0 &&
                              std::memcmp(x.data(), refC.data() + r * kHidden, kHidden * 4) == 0 &&
                              std::memcmp(n.data(), refN.data() + r * kHidden, kHidden * 4) == 0 && independent;
            }
            for (void* ptr : {static_cast<void*>(dX), static_cast<void*>(F), static_cast<void*>(mix), static_cast<void*>(xc), static_cast<void*>(xn)})
                hok(hipFree(ptr), "free hc norm");
        }
        check(exact && independent, "hc_pre_norm L" + std::to_string(c.l) + " site " + std::to_string(site) +
              ": F/mix/xc/xn == hc v4 + rmsnorm, launch modes 0/1/2, queued fused repeats (R1 x1000, R8 x250, alternating inputs), R1..8, permutations, padding, x3, row independence");
        for (void* ptr : {static_cast<void*>(fn), static_cast<void*>(base), static_cast<void*>(scale), static_cast<void*>(norm)})
            hok(hipFree(ptr), "free hc norm weights");
    }
}

void gates(const LayerCase& c, const std::vector<float>& Xall) {
    gate_hc_norm(c, Xall);
    gates_variant(c, Xall, 2, 1);
    gates_variant(c, Xall, 4, 3);
    gate_packed_dense(c, Xall);
    gate_packed_rows(c, Xall, 6);
    gate_packed_rows(c, Xall, 7);
}

void report(const LayerCase& c, const std::vector<float>& Xall, unsigned nrep) {
    const std::size_t N = kHc * kHidden;
    Stat hc[2], fa, ga, bb, rt;
    double mixe1[2] = {0, 0}, mixeb[2] = {0, 0};
    unsigned flip1 = 0, flipb = 0, diff1b = 0, mixdiff[2] = {0, 0};
    std::vector<float> z1, zb;
    std::vector<double> z64, A;
    auto f16w = [](const std::vector<std::uint16_t>& w) {
        return [&w](unsigned o, unsigned i) { return f16_to_float(w[static_cast<std::size_t>(o) * kHidden + i]); };
    };
    for (unsigned n = 0; n < nrep; ++n) {
        const float* X = Xall.data() + static_cast<std::size_t>(n) * N;
        std::vector<float> F(N);
        H::rmsnorm({X, N}, nullptr, kEps, F);
        for (int s = 0; s < 2; ++s) {
            const auto& fn = c.fn[s];
            chains(F.data(), static_cast<unsigned>(N), [&fn](unsigned o, unsigned i) { return fn[static_cast<std::size_t>(o) * kHc * kHidden + i]; },
                   kHcMix, z1, zb, z64, A);
            add_all(hc[s], z1, zb, z64, A);
            const H::HcWeights hw{H::Mat{fn.data(), H::WType::F32, kHcMix, static_cast<unsigned>(N)}, c.base[s].data(), c.scale[s].data()};
            const auto m1 = pack(H::hc_mix({X, N}, hw));
            const auto mb = pack(H::hc_mix({X, N}, hw, Identity::kR1b));
            const auto m64 = mix64(z64, c.base[s].data(), c.scale[s].data());
            for (unsigned k = 0; k < kHcMix; ++k) {
                mixe1[s] = std::max(mixe1[s], std::fabs(double(m1[k]) - m64[k]));
                mixeb[s] = std::max(mixeb[s], std::fabs(double(mb[k]) - m64[k]));
            }
            mixdiff[s] += std::memcmp(m1.data(), mb.data(), sizeof m1) != 0 ? 1U : 0U;
        }
        if (c.kda) {
            const auto x = site_input(X, c, 0);
            chains(x.data(), kHidden, f16w(c.f_a), kKdaDim, z1, zb, z64, A);
            add_all(fa, z1, zb, z64, A);
            chains(x.data(), kHidden, f16w(c.g_a), kKdaDim, z1, zb, z64, A);
            add_all(ga, z1, zb, z64, A);
            chains(x.data(), kHidden, f16w(c.b), kKdaHeads, z1, zb, z64, A);
            add_all(bb, z1, zb, z64, A);
        }
        if (c.moe) {
            const auto x = site_input(X, c, 1);
            chains(x.data(), kHidden, f16w(c.router), kExperts, z1, zb, z64, A);
            add_all(rt, z1, zb, z64, A);
            const H::Mat Wm{c.router.data(), H::WType::F16, kExperts, kHidden};
            const auto r64 = route64(z64, c.bias);
            const H::Route r1 = H::route(x, Wm, c.bias.data());
            const H::Route rb = H::route(x, Wm, c.bias.data(), Identity::kR1b);
            flip1 += !std::equal(r64.begin(), r64.end(), r1.id.begin()) ? 1U : 0U;
            flipb += !std::equal(r64.begin(), r64.end(), rb.id.begin()) ? 1U : 0U;
            diff1b += r1.id != rb.id ? 1U : 0U;
        }
    }
    hc[0].print("hc_attn", c.l);
    hc[1].print("hc_ffn", c.l);
    for (int s = 0; s < 2; ++s)
        std::printf("err L%u %s mix (pre/post/comb) max |m - m64|: R1 %.3e R1b %.3e; rows with R1 mix != R1b mix: %u of %u\n", c.l,
                    s == 0 ? "hc_attn" : "hc_ffn", mixe1[s], mixeb[s], mixdiff[s], nrep);
    if (c.kda) {
        fa.print("kda.f_a", c.l);
        ga.print("kda.g_a", c.l);
        bb.print("kda.b", c.l);
    }
    if (c.moe) {
        rt.print("router", c.l);
        std::printf("err L%u router top-8 != float64 route: R1 %u, R1b %u of %u rows; R1 route != R1b route: %u\n", c.l, flip1, flipb, nrep, diff1b);
    }
    std::fflush(stdout);
}


// ---------------------------------------------------------------- timing: R1 production kernels vs R1b (ABA x3)
// us per call: median of timed batches; three independent A B A sequences.
template <class F> double time_us(F f, unsigned warm = 5, unsigned calls = 50) {
    hipEvent_t e0{}, e1{};
    hok(hipEventCreate(&e0), "ev");
    hok(hipEventCreate(&e1), "ev");
    for (unsigned w = 0; w < warm; ++w) f();
    hok(hipEventRecord(e0, nullptr), "rec");
    for (unsigned i = 0; i < calls; ++i) f();
    hok(hipEventRecord(e1, nullptr), "rec");
    hok(hipEventSynchronize(e1), "sync");
    float ms = 0;
    hok(hipEventElapsedTime(&ms, e0, e1), "elapsed");
    hok(hipEventDestroy(e0), "ev");
    hok(hipEventDestroy(e1), "ev");
    return 1000.0 * double(ms) / double(calls);
}
template <class A, class B> void aba_measure(const char* what, unsigned R, A a, B b) {
    std::vector<double> ta, tb;
    for (int rep = 0; rep < 3; ++rep) {
        ta.push_back(a());
        tb.push_back(b());
        ta.push_back(a());
    }
    std::sort(ta.begin(), ta.end());
    std::sort(tb.begin(), tb.end());
    std::printf("bench %-22s R %u: A %.1f us [%.1f..%.1f] | B %.1f us [%.1f..%.1f] | delta %+.1f us/call\n", what, R, ta[ta.size() / 2],
                ta.front(), ta.back(), tb[1], tb.front(), tb.back(), tb[1] - ta[ta.size() / 2]);
}
template <class A, class B> void aba(const char* what, unsigned R, A a, B b) {
    aba_measure(what, R, [&] { return time_us(a); }, [&] { return time_us(b); });
}
template <class F> double time_cold_us(F f, unsigned copies) {
    // Reset the rotation for every batch; A/B/A see the same copy sequence.
    // Warm every copy twice, including every kernel/job specialization; time two full rotations.
    unsigned index = 0;
    return time_us([&] { f(index++ % copies); }, 2 * copies, 2 * copies);
}
template <class A, class B> void cold_aba(const char* what, unsigned R, unsigned copies, A a, B b) {
    aba_measure(what, R, [&] { return time_cold_us(a, copies); }, [&] { return time_cold_us(b, copies); });
}
void bench(const LayerCase& c, const std::vector<float>& Xall) {
    const std::size_t N = kHc * kHidden;
    float* dfn = up(c.fn[0]);
    float* dbase = up(c.base[0]);
    float* dscale = up(c.scale[0]);
    float* dX = up(std::vector<float>(Xall.begin(), Xall.begin() + static_cast<std::ptrdiff_t>(8 * N)));
    float* dF = up(std::vector<float>(8 * N));
    float* dmix = up(std::vector<float>(8 * kHcMix));
    float* dxc = up(std::vector<float>(8 * kHidden));
    float* dxn = up(std::vector<float>(8 * kHidden));
    std::vector<std::uint16_t> nb(kHidden);
    for (unsigned i = 0; i < kHidden; ++i) nb[i] = float_to_bf16(c.in_norm[i]);
    auto* dnorm = up(nb);
    std::vector<float> xs;
    for (unsigned r = 0; r < 8; ++r) {
        const auto xn = site_input(Xall.data() + r * N, c, 0);
        xs.insert(xs.end(), xn.begin(), xn.end());
    }
    float* dx = up(xs);
    float* dy = up(std::vector<float>(8 * 512));
    float* dy2 = up(std::vector<float>(8 * 512));
    float* dy3 = up(std::vector<float>(8 * 512));
    std::uint16_t* dfa = c.kda ? up(c.f_a) : nullptr;
    std::uint16_t* dga = c.kda ? up(c.g_a) : nullptr;
    std::uint16_t* db = c.kda ? up(c.b) : nullptr;
    std::uint16_t* drt = c.moe ? up(c.router) : nullptr;
    float* dlg = up(std::vector<float>(8 * kExperts));
    // Rotate a working set larger than 32 MiB; actual eviction/DRAM traffic needs counters.
    constexpr unsigned kCopies = 64;
    std::vector<float*> fns;
    std::vector<std::uint16_t*> rts, rts_packed;
    for (unsigned k = 0; k < kCopies; ++k) {
        fns.push_back(up(c.fn[0]));
        if (c.moe) {
            rts.push_back(up(c.router));
            rts_packed.push_back(packed_copy(rts.back(), kExperts));
        }
    }
    for (unsigned R : {1U, 8U}) {
        auto hc1 = [&](unsigned k) { r1b::hc_pre(dX, fns[k], dbase, dscale, dF, dmix, dxc, R, nullptr, 1); };
        auto hc2 = [&](unsigned k) { r1b::hc_pre(dX, fns[k], dbase, dscale, dF, dmix, dxc, R, nullptr, 2); };
        auto hc3 = [&](unsigned k) { r1b::hc_pre(dX, fns[k], dbase, dscale, dF, dmix, dxc, R, nullptr, 3); };
        auto hc4 = [&](unsigned k) { r1b::hc_pre(dX, fns[k], dbase, dscale, dF, dmix, dxc, R, nullptr, 4); };
        auto prod = [&](unsigned k) { xfer::hc_pre_v3(dX, fns[k], dbase, dscale, dF, dmix, dxc, R, nullptr, 7); };
        cold_aba("hc COLD R1|hc1", R, kCopies, prod, hc1);
        cold_aba("hc COLD R1|hc2", R, kCopies, prod, hc2);
        cold_aba("hc COLD R1|hc3", R, kCopies, prod, hc3);
        cold_aba("hc COLD current|hc3", R, kCopies, hc1, hc3);
        cold_aba("hc COLD hc2|hc3", R, kCopies, hc2, hc3);
        cold_aba("hc COLD R1|hc4", R, kCopies, prod, hc4);
        cold_aba("hc COLD hc1|hc4", R, kCopies, hc1, hc4);
        cold_aba("hc COLD hc3|hc4", R, kCopies, hc3, hc4);
        auto old_norm = [&](unsigned k) {
            hc4(k);
            dev::rmsnorm(dxc, dnorm, dev::WType::BF16, kHidden, kEps, dxn, R, nullptr);
        };
        auto fused_norm = [&](unsigned k) {
            r1b::hc_pre_norm(dX, fns[k], dbase, dscale, dF, dmix, dxc, dnorm, dxn, R, nullptr);
        };
        cold_aba("hc+norm COLD old|fused", R, kCopies, old_norm, fused_norm);
        if (c.moe) {
            auto v2 = [&](unsigned k) { r1b::router_dense(dx, rts[k], dlg, R, nullptr, 2); };
            auto v4 = [&](unsigned k) { r1b::router_dense(dx, rts[k], dlg, R, nullptr, 4); };
            auto router = [&](unsigned k) { xfer::router_dense(dx, rts[k], dlg, R, nullptr, -1); };
            cold_aba("router COLD R1|v2", R, kCopies, router, v2);
            cold_aba("router COLD R1|v4", R, kCopies, router, v4);
            cold_aba("router COLD v2|v4", R, kCopies, v2, v4);
            auto v7 = [&](unsigned k) { r1b::router_dense(dx, rts_packed[k], dlg, R, nullptr, 7); };
            cold_aba("router COLD R1|auto", R, kCopies, router, v7);
            cold_aba("router COLD v2|auto", R, kCopies, v2, v7);
            auto v6 = [&](unsigned k) { r1b::router_dense(dx, rts_packed[k], dlg, R, nullptr, 6); };
            cold_aba("router COLD R1|v6", R, kCopies, router, v6);
            cold_aba("router COLD v2|v6", R, kCopies, v2, v6);
            cold_aba("router COLD v4|v6", R, kCopies, v4, v6);
            if (R == 1) {
                auto v5 = [&](unsigned k) { r1b::router_dense(dx, rts_packed[k], dlg, R, nullptr, 5); };
                cold_aba("router COLD R1|v5", R, kCopies, router, v5);
                cold_aba("router COLD v2|v5", R, kCopies, v2, v5);
                cold_aba("router COLD v4|v5", R, kCopies, v4, v5);
                cold_aba("router COLD v5|v6", R, kCopies, v5, v6);
            }
        }
    }
    if (c.kda) {
        std::vector<std::uint16_t*> fa, ga, bw, fp, gp, bp;
        for (unsigned k = 0; k < kCopies; ++k) {
            fa.push_back(up(c.f_a));
            ga.push_back(up(c.g_a));
            bw.push_back(up(c.b));
            fp.push_back(packed_copy(fa.back(), kKdaDim));
            gp.push_back(packed_copy(ga.back(), kKdaDim));
            bp.push_back(packed_copy(bw.back(), kKdaHeads / 2));
        }
        const unsigned Hk = kKdaHeads / 2;
        for (unsigned R : {1U, 8U}) {
            auto prod = [&](unsigned k) {
                const fast::DenseJob jobs[3] = {{fa[k], dx, dy, kKdaDim, kHidden, kHidden, kKdaDim},
                                               {ga[k], dx, dy2, kKdaDim, kHidden, kHidden, kKdaDim},
                                               {bw[k], dx, dy3, Hk, kHidden, kHidden, Hk}};
                fast::dense_f16(jobs, 3, R, nullptr);
            };
            auto dense = [&](unsigned k, int variant) {
                const std::uint16_t* const W[3] = {(variant == 5 || variant == 6 || variant == 7) ? fp[k] : fa[k], (variant == 5 || variant == 6 || variant == 7) ? gp[k] : ga[k],
                                                     (variant == 5 || variant == 6 || variant == 7) ? bp[k] : bw[k]};
                const unsigned sizes[3] = {kKdaDim, kKdaDim, Hk};
                float* const Y[3] = {dy, dy2, dy3};
                r1b::dense_f16_k4096_x3(dx, kHidden, W, sizes, Y, sizes, R, nullptr, variant);
            };
            auto v2 = [&](unsigned k) { dense(k, 2); };
            auto v4 = [&](unsigned k) { dense(k, 4); };
            cold_aba("KDA COLD R1|v2", R, kCopies, prod, v2);
            cold_aba("KDA COLD R1|v4", R, kCopies, prod, v4);
            cold_aba("KDA COLD v2|v4", R, kCopies, v2, v4);
            auto v7 = [&](unsigned k) { dense(k, 7); };
            cold_aba("KDA COLD R1|auto", R, kCopies, prod, v7);
            cold_aba("KDA COLD v2|auto", R, kCopies, v2, v7);
            auto v6 = [&](unsigned k) { dense(k, 6); };
            cold_aba("KDA COLD R1|v6", R, kCopies, prod, v6);
            cold_aba("KDA COLD v2|v6", R, kCopies, v2, v6);
            cold_aba("KDA COLD v4|v6", R, kCopies, v4, v6);
            if (R == 1) {
                auto v5 = [&](unsigned k) { dense(k, 5); };
                cold_aba("KDA COLD R1|v5", R, kCopies, prod, v5);
                cold_aba("KDA COLD v2|v5", R, kCopies, v2, v5);
                cold_aba("KDA COLD v4|v5", R, kCopies, v4, v5);
                cold_aba("KDA COLD v5|v6", R, kCopies, v5, v6);
            }
        }
        for (auto* p : fa) hok(hipFree(p), "free");
        for (auto* p : ga) hok(hipFree(p), "free");
        for (auto* p : bw) hok(hipFree(p), "free");
        for (auto* p : fp) hok(hipFree(p), "free");
        for (auto* p : gp) hok(hipFree(p), "free");
        for (auto* p : bp) hok(hipFree(p), "free");
    }
    hok(hipFree(dxn), "free");
    hok(hipFree(dnorm), "free");
    for (float* p : fns) hok(hipFree(p), "free");
    for (std::uint16_t* p : rts) hok(hipFree(p), "free");
    for (std::uint16_t* p : rts_packed) hok(hipFree(p), "free");
    for (unsigned R : {1U, 8U}) {
        aba("hc_pre (v7 | r1b)", R, [&] { xfer::hc_pre_v3(dX, dfn, dbase, dscale, dF, dmix, dxc, R, nullptr, 7); },
            [&] { r1b::hc_pre(dX, dfn, dbase, dscale, dF, dmix, dxc, R, nullptr, 1); });
        if (c.moe)
            aba("router (auto | r1b)", R, [&] { xfer::router_dense(dx, drt, dlg, R, nullptr, -1); },
                [&] { r1b::router_dense(dx, drt, dlg, R, nullptr, 2); });
        if (c.kda) {
            const unsigned Hk = kKdaHeads / 2;  // TP2 rank rows
            const fast::DenseJob j3[3] = {{dfa, dx, dy, kKdaDim, kHidden, kHidden, kKdaDim},
                                          {dga, dx, dy2, kKdaDim, kHidden, kHidden, kKdaDim},
                                          {db, dx, dy3, Hk, kHidden, kHidden, Hk}};
            aba("kda f_a+g_a+b (fast|r1b)", R, [&] { fast::dense_f16(j3, 3, R, nullptr); },
                [&] {
                    const std::uint16_t* const W[3] = {dfa, dga, db};
                    const unsigned Nn[3] = {kKdaDim, kKdaDim, Hk};
                    float* const Y[3] = {dy, dy2, dy3};
                    r1b::dense_f16_k4096_x3(dx, kHidden, W, Nn, Y, Nn, R, nullptr, 2);
                });
        }
    }
    for (void* p : {static_cast<void*>(dfn), static_cast<void*>(dbase), static_cast<void*>(dscale), static_cast<void*>(dX),
                    static_cast<void*>(dF), static_cast<void*>(dmix), static_cast<void*>(dxc), static_cast<void*>(dx), static_cast<void*>(dy),
                    static_cast<void*>(dy2), static_cast<void*>(dy3), static_cast<void*>(dlg), static_cast<void*>(dfa),
                    static_cast<void*>(dga), static_cast<void*>(db), static_cast<void*>(drt)})
        if (p) hok(hipFree(p), "free");
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s CASES_DIR [NROWS]\n", argv[0]);
        return 2;
    }
    const std::string dir = std::string(argv[1]) + "/moe_l3";
    const unsigned nrows = argc > 2 ? static_cast<unsigned>(std::stoul(argv[2])) : 512;
    const auto rb = read_bin(dir, "router", static_cast<std::size_t>(kExperts) * kHidden * 2);
    const auto bb = read_bin(dir, "bias", kExperts * 4);
    std::vector<std::uint16_t> router(static_cast<std::size_t>(kExperts) * kHidden);
    std::vector<float> bias(kExperts);
    std::memcpy(router.data(), rb.data(), rb.size());
    std::memcpy(bias.data(), bb.data(), bb.size());
    const H::Mat W{router.data(), H::WType::F16, kExperts, kHidden};
    std::mt19937 g(20261025);
    gate_pack_bits();
    std::uint16_t* dW = up(router);
    std::uint16_t* dPacked = packed_copy(dW, kExperts);
    float* dbias = up(bias);

    // ---- gate: device == host, rows 1..8, row independence, route on R1b logits
    bool all = true, indep = true, routes = true;
    for (int variant : {2, 4, 5, 6, 7}) for (unsigned R = 1; R <= 8; ++R) {
        if (variant == 5 && R != 1) continue;
        const auto* weights = variant == 5 || variant == 6 || variant == 7 ? dPacked : dW;
        const auto x = rnd(static_cast<std::size_t>(R) * kHidden, g, 0.3F);
        float* dx = up(x);
        float* dl = up(std::vector<float>(static_cast<std::size_t>(R) * kExperts, std::nanf("")));
        std::uint32_t* dids = up(std::vector<std::uint32_t>(R * kTopE));
        float* dw = up(std::vector<float>(R * kTopE));
        r1b::router_dense(dx, weights, dl, R, nullptr, variant);
        xfer::route_v2(dl, dbias, dids, dw, R, nullptr);
        const auto dev = down(dl, static_cast<std::size_t>(R) * kExperts);
        r1b::router_dense(dx, dW, dl, R, nullptr, 2);
        all &= same_bits(dev, down(dl, dev.size()));
        if (variant == 5 || variant == 6 || variant == 7) {
            r1b::router_dense(dx, dW, dl, R, nullptr, 4);
            all &= same_bits(dev, down(dl, dev.size()));
        }
        const auto idd = down(dids, R * kTopE);
        const auto wd = down(dw, R * kTopE);
        for (unsigned r = 0; r < R; ++r) {
            std::vector<float> h(kExperts);
            H::dense_r1b({x.data() + static_cast<std::size_t>(r) * kHidden, kHidden}, W, h);
            all &= std::memcmp(h.data(), dev.data() + static_cast<std::size_t>(r) * kExperts, kExperts * 4) == 0;
            const H::Route hr = H::route({x.data() + static_cast<std::size_t>(r) * kHidden, kHidden}, W, bias.data(), Identity::kR1b);
            routes &= std::memcmp(hr.id.data(), idd.data() + r * kTopE, kTopE * 4) == 0 &&
                      std::memcmp(hr.w.data(), wd.data() + r * kTopE, kTopE * 4) == 0;
            float* d1 = up(std::vector<float>(kExperts));
            r1b::router_dense(dx + static_cast<std::size_t>(r) * kHidden, weights, d1, 1, nullptr, variant);
            const auto one = down(d1, kExperts);
            indep &= std::memcmp(one.data(), dev.data() + static_cast<std::size_t>(r) * kExperts, kExperts * 4) == 0;
            hok(hipFree(d1), "free");
        }
        for (void* p : {static_cast<void*>(dx), static_cast<void*>(dl), static_cast<void*>(dids), static_cast<void*>(dw)}) hok(hipFree(p), "free");
    }
    check(all, "R1b router logits: device == host dense_r1b bitwise (real L3 router, v2/v4/v6 R = 1..8, v5 R=1)");
    check(indep, "R1b router: row r of an R-row call == the row alone (R = 1..8)");
    check(routes, "R1b route: host route(kR1b) ids + weights == device route_v2 on R1b logits (R = 1..8)");

    // ---- report: R1 and R1b vs float64, logits and route flips
    double max1 = 0, maxb = 0, sum1 = 0, sumb = 0;
    unsigned flip1 = 0, flipb = 0, diff_1b = 0;
    for (unsigned n = 0; n < nrows; ++n) {
        const auto x = rnd(kHidden, g, n % 2 ? 0.3F : 1.0F);
        std::vector<float> l1(kExperts), lb(kExperts);
        std::vector<double> l64(kExperts);
        H::dense({x.data(), kHidden}, W, l1);
        H::dense_r1b({x.data(), kHidden}, W, lb);
        double mx = 0;
        for (unsigned e = 0; e < kExperts; ++e) {
            double a = 0;
            for (unsigned i = 0; i < kHidden; ++i) a += static_cast<double>(x[i]) * static_cast<double>(W.at(e, i));
            l64[e] = a;
            mx = std::max(mx, std::fabs(a));
        }
        for (unsigned e = 0; e < kExperts; ++e) {
            const double e1 = std::fabs(l1[e] - l64[e]) / mx, eb = std::fabs(lb[e] - l64[e]) / mx;
            max1 = std::max(max1, e1);
            maxb = std::max(maxb, eb);
            sum1 += e1;
            sumb += eb;
        }
        const auto r64 = route64(l64, bias);
        const H::Route r1 = H::route({x.data(), kHidden}, W, bias.data()), rb1 = H::route({x.data(), kHidden}, W, bias.data(), Identity::kR1b);
        flip1 += !std::equal(r64.begin(), r64.end(), r1.id.begin());
        flipb += !std::equal(r64.begin(), r64.end(), rb1.id.begin());
        diff_1b += r1.id != rb1.id;
    }
    const double cnt = static_cast<double>(nrows) * kExperts;
    std::printf("report: %u random rows (scale 0.3 / 1.0), real L3 router: |logit - fp64| / max|logit|: R1 max %.3e mean %.3e | "
                "R1b max %.3e mean %.3e; top-8 route != fp64 route: R1 %u, R1b %u; R1 route != R1b route: %u\n",
                nrows, max1, sum1 / cnt, maxb, sumb / cnt, flip1, flipb, diff_1b);
    hok(hipFree(dW), "free");
    hok(hipFree(dbias), "free");
    hok(hipFree(dPacked), "free");
    if (argc > 5) {  // hc / KDA gates + per-layer report on the exported allowlist tensors and real residual rows
        const std::string root = argv[3], xpath = argv[4];
        const unsigned nrep = static_cast<unsigned>(std::stoul(argv[5]));
        const std::size_t N = kHc * kHidden;
        const unsigned need = std::max(nrep, 36U);  // the gates use rows 0..35 (R = 8 starts at row 7, 8 rows)
        const auto xb = read_file(xpath, static_cast<std::size_t>(need) * N * 4);
        std::vector<float> Xall(static_cast<std::size_t>(need) * N);
        std::memcpy(Xall.data(), xb.data(), xb.size());
        std::vector<unsigned> layers;
        for (unsigned l = 0; l < 46; ++l)
            if (exists(root + "/L" + std::to_string(l) + "/manifest.txt")) layers.push_back(l);
        if (layers.empty()) throw std::runtime_error("no R1B_CASES/L<l> directories");
        bool gated = false;
        for (unsigned l : layers) {
            const LayerCase c = load_layer(root, l);
            if (!gated && c.kda) {  // device gates on the first KDA layer (hc + KDA kernels); hc again on one DSA layer below
                gates(c, Xall);
                gated = true;
            }
            report(c, Xall, nrep);
        }
        for (unsigned l : layers)
            if (l % 4 == 3) {
                gates(load_layer(root, l), Xall);
                break;
            }
        bool benched = false;
        for (unsigned l : layers) {  // timing on the first layer with KDA + MoE (hc, router, KDA small projections)
            const LayerCase c = load_layer(root, l);
            if (c.kda && c.moe && !benched) {
                bench(c, Xall);
                benched = true;
            }
        }
    }
    std::printf("%s (%d failures)\n", failures == 0 ? "R1B PASS" : "R1B FAIL", failures);
    return failures == 0 ? 0 : 1;
}
