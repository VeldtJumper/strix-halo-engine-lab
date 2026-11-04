// CPU tests of the sparse-attention threshold router host reference (src/h3/levers/sparse/router_ref.hpp, which
// holds the spec): geometry / degenerate shapes, centroids exclude padding, the 1-node path (full buffers) == the
// two-rank path (local buffers + stats exchange + remote centroids from a copy of the remote K) bit for bit,
// LUT contract (sorted / unique / in range), band / sink / dense-prefix semantics, decisions vs an fp64 evaluation
// (outside a 1e-4 margin), tau extremes.
// usage: strix-h3-levers-sparse-router-tests
#include "h3/levers/sparse/router_ref.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace strix::h3::levers::sparse;
namespace g = strix::h3::gemm;

namespace {
int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { ++failures; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

struct Rng {
    std::uint64_t s;
    std::uint64_t next() { std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL); z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL; return z ^ (z >> 31U); }
    float uni() { return static_cast<float>(next() >> 40U) * 0x1p-24F * 2.0F - 1.0F; }
};
constexpr std::uint64_t kLd = static_cast<std::uint64_t>(kHeads) * kDim;

// Q / K with structure: per-tile direction bias so scores vary by tile.
void make_qk(std::uint32_t S, std::vector<std::uint16_t>& q, std::vector<std::uint16_t>& k, std::uint64_t seed) {
    Rng r{seed};
    q.resize(S * kLd);
    k.resize(S * kLd);
    std::vector<float> bias(static_cast<std::size_t>((S + 63U) / 64U) * kLd);
    for (auto& b : bias) b = r.uni() * 2.0F;
    for (std::uint32_t row = 0; row < S; ++row)
        for (std::uint64_t e = 0; e < kLd; ++e) {
            q[row * kLd + e] = g::f32_to_bf16(r.uni() + bias[(row / 64U) * kLd + e] * 0.5F);
            k[row * kLd + e] = g::f32_to_bf16(r.uni() + bias[(row / 64U) * kLd + e]);
        }
}

struct Full {
    Lut lut;
    Stats st;
};
// 1 node: full buffers, per-block kc + partials, finalize in block order, all query tiles vs all key tiles.
Full one_node(const Geometry& gm, const Params& p, const std::vector<std::uint16_t>& q, const std::vector<std::uint16_t>& k) {
    const std::uint32_t jc = gm.cut / kTile;
    const auto kc0 = key_centroids(k.data(), kLd, gm, 0, jc), kc1 = key_centroids(k.data() + static_cast<std::size_t>(gm.cut) * kLd, kLd, gm, jc, gm.KT);
    Full f;
    f.st = stats_finalize({stats_partial(kc0, jc), stats_partial(kc1, gm.KT - jc)});
    std::vector<std::uint16_t> kc(kc0);
    kc.insert(kc.end(), kc1.begin(), kc1.end());
    const auto qb = query_centroids(q.data(), kLd, gm, 0, gm.KT);
    const auto thr = thresholds(qb, gm.KT, f.st, p.tau);
    f.lut = route(gm, p, 0, gm.QT, qb, thr, 0, gm.KT, kc);
    return f;
}

void test_geometry() {
    Geometry gm = geometry(1000, 150);
    CHECK(!gm.dense && gm.KT == 16 && gm.QT == 8 && gm.cut == 384, "geometry 1000: KT %u QT %u cut %u", gm.KT, gm.QT, gm.cut);
    gm = geometry(37910, 614);
    CHECK(!gm.dense && gm.cut == 18944 && gm.cut % 128 == 0, "geometry 37910: cut %u", gm.cut);
    CHECK(geometry(255, 10).dense && geometry(1000, 1000).dense && geometry(1000, 2000).dense && !geometry(256, 0).dense, "degenerate");
    const Params p;
    CHECK(!sparse_call(p, geometry(1000, 150), 2, 10) && !sparse_call(p, geometry(1000, 150), 3, 1) && sparse_call(p, geometry(1000, 150), 3, 2), "window");
    // dense layer mask (quality knob: layers 39-42 dense)
    Params pm;
    pm.dense_layer_mask = layer_mask("39-42");
    const Geometry gg = geometry(1000, 150);
    bool mask_ok = pm.dense_layer_mask == (0xfULL << 39U) && sparse_call(pm, gg, 3, 38) && !sparse_call(pm, gg, 3, 39) && !sparse_call(pm, gg, 3, 42) &&
                   sparse_call(pm, gg, 3, 43) && !sparse_call(pm, gg, 3, 1) && layer_mask("") == 0U && layer_mask("0,5-6") == 0x61ULL;
    int bad_lists = 0;
    for (const char* l : {"39-", "-3", "a", "3-2", "64", "1,,2", "2x"}) {
        try { layer_mask(l); } catch (const std::exception&) { ++bad_lists; }
    }
    CHECK(mask_ok && bad_lists == 7, "dense layer mask (%d of 7 bad lists rejected)", bad_lists);
    std::printf("geometry checked; dense layer mask 39-42 ok %d, %d of 7 malformed lists rejected\n", mask_ok ? 1 : 0, bad_lists);
}

void test_padding() {
    const std::uint32_t S = 1000U;
    std::vector<std::uint16_t> q, k;
    make_qk(S, q, k, 3);
    const Geometry gm = geometry(S, 100);
    const std::uint32_t last = gm.KT - 1U, n = S - last * kTile;  // 40 rows
    const auto kc = key_centroids(k.data() + static_cast<std::size_t>(last) * kTile * kLd, kLd, gm, last, gm.KT);
    std::size_t bad = 0;
    for (std::uint64_t e = 0; e < kLd; ++e) {
        float s = 0.0F;
        for (std::uint32_t r = 0; r < n; ++r) s = s + g::bf16_to_f32(k[(static_cast<std::size_t>(last) * kTile + r) * kLd + e]);
        bad += kc[e] != g::f32_to_bf16(s / static_cast<float>(n));
    }
    std::printf("padding: last tile %u rows, centroid mismatches %zu\n", n, bad);
    CHECK(bad == 0, "padding never enters centroids");
}

void test_two_rank(std::uint32_t S, std::uint32_t P, float tau, std::uint64_t seed) {
    std::vector<std::uint16_t> q, k;
    make_qk(S, q, k, seed);
    Params p;
    p.tau = tau;
    const Geometry gm = geometry(S, P);
    const Full f = one_node(gm, p, q, k);
    validate_lut(f.lut);
    const std::uint32_t jc = gm.cut / kTile, tc = gm.cut / kQTile;
    // rank r: local rows only (separate copies); the remote K arrives as a copy (RX)
    std::vector<std::uint16_t> q0(q.begin(), q.begin() + gm.cut * kLd), k0(k.begin(), k.begin() + gm.cut * kLd);
    std::vector<std::uint16_t> q1(q.begin() + gm.cut * kLd, q.end()), k1(k.begin() + gm.cut * kLd, k.end());
    const auto kc0 = key_centroids(k0.data(), kLd, gm, 0, jc), kc1 = key_centroids(k1.data(), kLd, gm, jc, gm.KT);
    const StatsPartial sp0 = stats_partial(kc0, jc), sp1 = stats_partial(kc1, gm.KT - jc);
    bool all_same = true;
    for (std::uint32_t r = 0; r < 2U; ++r) {
        const Stats st = stats_finalize({sp0, sp1});  // block order on both ranks
        const std::vector<std::uint16_t>& ql = r ? q1 : q0;
        const std::uint32_t t0 = r ? tc : 0U, t1 = r ? gm.QT : tc;
        const std::uint32_t i0 = 2U * t0, i1 = std::min(2U * t1, gm.KT);
        const auto qb = query_centroids(ql.data(), kLd, gm, i0, i1);
        const auto thr = thresholds(qb, i1 - i0, st, p.tau);
        // remote centroids recomputed from the received copy (same function, same bits)
        const auto kc_rx = r ? key_centroids(k0.data(), kLd, gm, 0, jc) : key_centroids(k1.data(), kLd, gm, jc, gm.KT);
        const Lut a = route(gm, p, t0, t1, qb, thr, 0, jc, r ? kc_rx : kc0);
        const Lut b = route(gm, p, t0, t1, qb, thr, jc, gm.KT, r ? kc1 : kc_rx);
        validate_lut(a);
        validate_lut(b);
        // concatenate block 0 then block 1 per list and compare with the 1-node LUT
        for (std::uint32_t t = t0; t < t1; ++t)
            for (std::uint32_t h = 0; h < kHeads; ++h) {
                const std::size_t l = static_cast<std::size_t>(t - t0) * kHeads + h, lf = static_cast<std::size_t>(t) * kHeads + h;
                std::vector<std::int32_t> mine(a.entries.begin() + a.offsets[l], a.entries.begin() + a.offsets[l + 1]);
                mine.insert(mine.end(), b.entries.begin() + b.offsets[l], b.entries.begin() + b.offsets[l + 1]);
                const std::vector<std::int32_t> full(f.lut.entries.begin() + f.lut.offsets[lf], f.lut.entries.begin() + f.lut.offsets[lf + 1]);
                all_same = all_same && mine == full;
            }
    }
    const double density = static_cast<double>(f.lut.entries.size()) / (static_cast<double>(gm.QT) * kHeads * gm.KT);
    const double infl = f.lut.parts_sum ? static_cast<double>(f.lut.union_sum) / static_cast<double>(f.lut.parts_sum) : 0.0;
    std::printf("two-rank vs 1 node: S %u P %u tau %g: LUT identical %d, density %.3f, dense prefix tiles %u, union / (|A| + |B|) %.3f\n", S, P,
        static_cast<double>(tau), all_same ? 1 : 0, density, f.lut.dense_tiles, infl);
    CHECK(all_same, "two-rank LUT == 1-node LUT");
}

void test_semantics() {
    const std::uint32_t S = 1000U, P = 150U;
    std::vector<std::uint16_t> q, k;
    make_qk(S, q, k, 9);
    Params p;
    const Geometry gm = geometry(S, P);
    const Full f = one_node(gm, p, q, k);
    // band, sink, dense prefix tiles always present; fp64 decision check elsewhere
    std::size_t miss = 0, far = 0, checked = 0;
    // fp64 stats from the bf16 centroids
    const auto kc = key_centroids(k.data(), kLd, gm, 0, gm.KT);
    const auto qb = query_centroids(q.data(), kLd, gm, 0, gm.KT);
    const double c = 1.4426950408889634 * 0.08838834764831845;
    for (std::uint32_t t = 0; t < gm.QT; ++t)
        for (std::uint32_t h = 0; h < kHeads; ++h) {
            const std::size_t l = static_cast<std::size_t>(t) * kHeads + h;
            std::vector<bool> sel(gm.KT, false);
            for (std::uint32_t e = f.lut.offsets[l]; e < f.lut.offsets[l + 1]; ++e) sel[static_cast<std::size_t>(f.lut.entries[e])] = true;
            for (std::uint32_t j = 0; j < gm.KT; ++j) {
                bool want = kQTile * t < P || kTile * j < P;
                bool marginal = false;
                for (std::uint32_t b = 0; b < 2U && !want; ++b) {
                    const std::uint32_t i = 2U * t + b;
                    if (i >= gm.KT) continue;
                    if ((i > j ? i - j : j - i) <= 1U) { want = true; break; }
                    double mean_dot = 0.0, var_dot = 0.0, s = 0.0;
                    for (std::uint32_t d = 0; d < kDim; ++d) {
                        double m1 = 0.0, m2 = 0.0;
                        for (std::uint32_t jj = 0; jj < gm.KT; ++jj) {
                            const double v = g::bf16_to_f32(kc[(static_cast<std::size_t>(jj) * kHeads + h) * kDim + d]);
                            m1 += v;
                            m2 += v * v;
                        }
                        m1 /= gm.KT;
                        m2 = std::max(m2 / gm.KT - m1 * m1, 0.0);
                        const double qv = qb[(static_cast<std::size_t>(i) * kHeads + h) * kDim + d];
                        mean_dot += qv * m1;
                        var_dot += qv * qv * m2;
                        s += qv * g::bf16_to_f32(kc[(static_cast<std::size_t>(j) * kHeads + h) * kDim + d]);
                    }
                    const double thr = mean_dot * c + p.tau * std::sqrt(var_dot * c * c + 1e-6);
                    s *= c;
                    if (std::fabs(s - thr) <= 1e-4 * (std::fabs(thr) + 1.0)) marginal = true;
                    if (s > thr) want = true;
                }
                ++checked;
                if (want != sel[j]) {
                    if (marginal) ++far; else ++miss;
                }
            }
        }
    std::printf("semantics vs fp64: %zu decisions checked, %zu differ outside the 1e-4 margin, %zu inside it\n", checked, miss, far);
    CHECK(miss == 0, "decisions vs fp64");
    // tau extremes: very large -> only band + sink + dense prefix; very negative -> everything
    std::vector<std::uint16_t> q2, k2;
    make_qk(S, q2, k2, 10);
    Params hi;
    hi.tau = 1e6F;
    Params lo;
    lo.tau = -1e6F;
    const Full fh = one_node(gm, hi, q2, k2), fl = one_node(gm, lo, q2, k2);
    std::size_t extra = 0;
    for (std::uint32_t t = 0; t < gm.QT; ++t)
        for (std::uint32_t h = 0; h < kHeads; ++h) {
            const std::size_t l = static_cast<std::size_t>(t) * kHeads + h;
            for (std::uint32_t e = fh.lut.offsets[l]; e < fh.lut.offsets[l + 1]; ++e) {
                const std::uint32_t j = static_cast<std::uint32_t>(fh.lut.entries[e]);
                const bool band = (2U * t > j ? 2U * t - j : j - 2U * t) <= 1U || (2U * t + 1U > j ? 2U * t + 1U - j : j - 2U * t - 1U) <= 1U;
                extra += !(kQTile * t < P || kTile * j < P || band);
            }
        }
    const bool all = fl.lut.entries.size() == static_cast<std::size_t>(gm.QT) * kHeads * gm.KT;
    std::printf("tau 1e6: %zu entries beyond band / sink / dense prefix; tau -1e6: every tile selected %d\n", extra, all ? 1 : 0);
    CHECK(extra == 0 && all, "tau extremes");
}

void test_validation() {
    Lut L;
    L.t0 = 0;
    L.t1 = 1;
    L.j0 = 0;
    L.j1 = 4;
    L.offsets.assign(kHeads + 1U, 0U);
    int rej = 0;
    auto bad = L;
    bad.entries = {1, 1};
    bad.offsets.back() = 2U;
    for (std::size_t i = 1; i < bad.offsets.size(); ++i) bad.offsets[i] = 2U;
    try { validate_lut(bad); } catch (const std::invalid_argument&) { ++rej; }
    bad.entries = {2, 1};
    try { validate_lut(bad); } catch (const std::invalid_argument&) { ++rej; }
    bad.entries = {1, 4};
    try { validate_lut(bad); } catch (const std::invalid_argument&) { ++rej; }
    bad.offsets.back() = 3U;
    try { validate_lut(bad); } catch (const std::invalid_argument&) { ++rej; }
    try { validate_lut(L); } catch (const std::invalid_argument&) { rej += 100; }
    std::printf("LUT validation: %d of 4 malformed LUTs rejected (an empty LUT accepted)\n", rej);
    CHECK(rej == 4, "LUT validation");
}
}  // namespace

int main() {
    test_geometry();
    test_padding();
    test_two_rank(1000U, 150U, 0.75F, 1);
    test_two_rank(1000U, 150U, 0.0F, 2);
    test_two_rank(1450U, 70U, 1.0F, 4);
    test_semantics();
    test_validation();
    std::printf("sparse router tests %s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
