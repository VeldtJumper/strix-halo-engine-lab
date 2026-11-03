// GPU test of the sparse-attention router kernels (src/h3/levers/sparse/router.hpp) against the host reference
// (router_ref.hpp): kc, qbar, stats partials, mean / var, thresholds, LUT offsets + entries and telemetry, bitwise,
// for the 1-node path and the rank-local path (rank 1 buffers start at the cut), several shapes incl. partial tiles.
// usage: strix-h3-levers-sparse-router-gpu-tests [--bench]
#include "h3/levers/sparse/router.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace strix::h3::levers::sparse;
namespace g = strix::h3::gemm;

namespace {
int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { ++failures; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
#define HIP(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s: %s\n", #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

struct Rng {
    std::uint64_t s;
    std::uint64_t next() { std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL); z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL; return z ^ (z >> 31U); }
    float uni() { return static_cast<float>(next() >> 40U) * 0x1p-24F * 2.0F - 1.0F; }
};
constexpr std::uint64_t kLd = static_cast<std::uint64_t>(kHeads) * kDim;

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
template <class T> T* up(const std::vector<T>& v) { T* p; HIP(hipMalloc(&p, v.size() * sizeof(T) + 16)); HIP(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
template <class T> std::vector<T> down(const T* p, std::size_t n) { std::vector<T> v(n); HIP(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return v; }
template <class T> std::size_t ndiff(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return static_cast<std::size_t>(-1);
    std::size_t n = 0;
    for (std::size_t i = 0; i < a.size(); ++i) n += std::memcmp(&a[i], &b[i], sizeof(T)) != 0;
    return n;
}

// Device LUT of query tiles [t0, t1) of the rows in dq (global row 128 t0) against key tiles [j0, j1) (dk at row 64 j0),
// with given stats; compared with the host route on the same inputs.
std::size_t compare_lut(const Geometry& gm, const Params& p, std::uint32_t t0, std::uint32_t t1, const std::uint16_t* dq, const std::vector<std::uint16_t>& hq,
    std::uint32_t j0, std::uint32_t j1, const std::uint16_t* dk, const std::vector<std::uint16_t>& hk, const float* dmean, const float* dvar, const Stats& st,
    bool* tele_ok) {
    const std::uint32_t i0 = 2U * t0, i1 = std::min(2U * t1, gm.KT);
    float *qb, *thr;
    std::uint16_t* kc;
    HIP(hipMalloc(&qb, static_cast<std::size_t>(i1 - i0) * kLd * 4)); HIP(hipMalloc(&thr, static_cast<std::size_t>(i1 - i0) * kHeads * 4));
    HIP(hipMalloc(&kc, static_cast<std::size_t>(j1 - j0) * kLd * 2));
    launch_query_centroids(dq, kLd, gm, i0, i1, qb, nullptr);
    launch_thresholds(qb, i1 - i0, dmean, dvar, p.tau, thr, nullptr);
    launch_key_centroids(dk, kLd, gm, j0, j1, kc, nullptr);
    const DeviceLut D = build_lut(gm, p, t0, t1, qb, thr, j0, j1, kc, nullptr);
    // host
    const auto hqb = query_centroids(hq.data(), kLd, gm, i0, i1);
    const auto hthr = thresholds(hqb, i1 - i0, st, p.tau);
    const auto hkc = key_centroids(hk.data(), kLd, gm, j0, j1);
    const Lut H = route(gm, p, t0, t1, hqb, hthr, j0, j1, hkc);
    std::size_t bad = ndiff(down(qb, hqb.size()), hqb) + ndiff(down(thr, hthr.size()), hthr) + ndiff(down(kc, hkc.size()), hkc);
    bad += ndiff(down(D.offsets.as<std::uint32_t>(), H.offsets.size()), H.offsets);
    if (D.entries_n != H.entries.size()) ++bad;
    else if (D.entries_n) bad += ndiff(down(D.entries.as<std::int32_t>(), H.entries.size()), H.entries);
    *tele_ok = D.union_sum == H.union_sum && D.parts_sum == H.parts_sum && D.dense_tiles == H.dense_tiles;
    HIP(hipFree(qb)); HIP(hipFree(thr)); HIP(hipFree(kc));
    return bad;
}

void test_shape(std::uint32_t S, std::uint32_t P, float tau, std::uint64_t seed) {
    std::vector<std::uint16_t> q, k;
    make_qk(S, q, k, seed);
    Params p;
    p.tau = tau;
    const Geometry gm = geometry(S, P);
    const std::uint32_t jc = gm.cut / kTile, tc = gm.cut / kQTile;
    auto *dq = up(q), *dk = up(k);
    // stats: per-block partials on the device, finalize in block order
    std::uint16_t *kc0, *kc1;
    float *s, *mean, *var;
    HIP(hipMalloc(&kc0, static_cast<std::size_t>(jc) * kLd * 2)); HIP(hipMalloc(&kc1, static_cast<std::size_t>(gm.KT - jc) * kLd * 2));
    HIP(hipMalloc(&s, 4 * kLd * 4)); HIP(hipMalloc(&mean, kLd * 4)); HIP(hipMalloc(&var, kLd * 4));
    launch_key_centroids(dk, kLd, gm, 0, jc, kc0, nullptr);
    launch_key_centroids(dk + static_cast<std::size_t>(gm.cut) * kLd, kLd, gm, jc, gm.KT, kc1, nullptr);
    launch_stats_partial(kc0, jc, s, s + kLd, nullptr);
    launch_stats_partial(kc1, gm.KT - jc, s + 2 * kLd, s + 3 * kLd, nullptr);
    launch_stats_finalize(s, s + kLd, s + 2 * kLd, s + 3 * kLd, gm.KT, mean, var, nullptr);
    HIP(hipGetLastError());
    const auto hkc0 = key_centroids(k.data(), kLd, gm, 0, jc), hkc1 = key_centroids(k.data() + static_cast<std::size_t>(gm.cut) * kLd, kLd, gm, jc, gm.KT);
    const StatsPartial p0 = stats_partial(hkc0, jc), p1 = stats_partial(hkc1, gm.KT - jc);
    const Stats st = stats_finalize({p0, p1});
    std::size_t bad_stats = ndiff(down(s, kLd), p0.s1) + ndiff(down(s + kLd, kLd), p0.s2) + ndiff(down(s + 2 * kLd, kLd), p1.s1) +
                            ndiff(down(s + 3 * kLd, kLd), p1.s2) + ndiff(down(mean, kLd), st.mean) + ndiff(down(var, kLd), st.var);
    bool t1ok = false, t2ok = false, t3ok = false;
    // 1 node: all query tiles vs all key tiles
    const std::size_t bad1 = compare_lut(gm, p, 0, gm.QT, dq, q, 0, gm.KT, dk, k, mean, var, st, &t1ok);
    // rank-local: rank 1's query tiles (buffers start at the cut) vs its own block and vs block 0
    const std::vector<std::uint16_t> q1(q.begin() + gm.cut * kLd, q.end()), k1(k.begin() + gm.cut * kLd, k.end());
    auto *dq1 = up(q1), *dk1 = up(k1);
    const std::size_t bad2 = compare_lut(gm, p, tc, gm.QT, dq1, q1, jc, gm.KT, dk1, k1, mean, var, st, &t2ok);
    const std::size_t bad3 = compare_lut(gm, p, tc, gm.QT, dq1, q1, 0, jc, dk, k, mean, var, st, &t3ok);
    std::printf("router S %u P %u tau %g: stats %zu differ, 1-node LUT path %zu, rank-1 local %zu, rank-1 remote %zu differ; telemetry %d%d%d\n", S, P,
        static_cast<double>(tau), bad_stats, bad1, bad2, bad3, t1ok, t2ok, t3ok);
    CHECK(bad_stats == 0 && bad1 == 0 && bad2 == 0 && bad3 == 0 && t1ok && t2ok && t3ok, "router S %u", S);
    for (void* ptr : {static_cast<void*>(dq), static_cast<void*>(dk), static_cast<void*>(kc0), static_cast<void*>(kc1), static_cast<void*>(s),
             static_cast<void*>(mean), static_cast<void*>(var), static_cast<void*>(dq1), static_cast<void*>(dk1)})
        HIP(hipFree(ptr));
}

void bench() {
    const std::uint32_t S = 37910U, P = 614U;
    const Geometry gm = geometry(S, P);
    Params p;
    std::vector<std::uint16_t> q, k;
    make_qk(S, q, k, 99);
    auto *dq = up(q), *dk = up(k);
    std::uint16_t* kc;
    float *s, *mean, *var, *qb, *thr;
    HIP(hipMalloc(&kc, static_cast<std::size_t>(gm.KT) * kLd * 2)); HIP(hipMalloc(&s, 2 * kLd * 4)); HIP(hipMalloc(&mean, kLd * 4)); HIP(hipMalloc(&var, kLd * 4));
    HIP(hipMalloc(&qb, static_cast<std::size_t>(gm.KT) * kLd * 4)); HIP(hipMalloc(&thr, static_cast<std::size_t>(gm.KT) * kHeads * 4));
    for (int rep = 0; rep < 2; ++rep) {
        HIP(hipDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        launch_key_centroids(dk, kLd, gm, 0, gm.KT, kc, nullptr);
        launch_query_centroids(dq, kLd, gm, 0, gm.KT, qb, nullptr);
        launch_stats_partial(kc, gm.KT, s, s + kLd, nullptr);
        launch_stats_finalize(s, s + kLd, nullptr, nullptr, gm.KT, mean, var, nullptr);
        launch_thresholds(qb, gm.KT, mean, var, p.tau, thr, nullptr);
        HIP(hipDeviceSynchronize());
        const auto t1 = std::chrono::steady_clock::now();
        const DeviceLut L = build_lut(gm, p, 0, gm.QT, qb, thr, 0, gm.KT, kc, nullptr);
        const auto t2 = std::chrono::steady_clock::now();
        const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        if (rep)
            std::printf("bench router S %u (synthetic Q/K, tau 0.75): centroids + stats + thresholds %.2f ms, LUT build %.2f ms, density %.3f, LUT %.1f MB, "
                        "union / (|A|+|B|) %.3f\n",
                S, ms(t0, t1), ms(t1, t2), static_cast<double>(L.entries_n) / (static_cast<double>(gm.QT) * kHeads * gm.KT), L.entries_n * 4.0 / 1e6,
                L.parts_sum ? static_cast<double>(L.union_sum) / static_cast<double>(L.parts_sum) : 0.0);
    }
    for (void* ptr : {static_cast<void*>(dq), static_cast<void*>(dk), static_cast<void*>(kc), static_cast<void*>(s), static_cast<void*>(mean),
             static_cast<void*>(var), static_cast<void*>(qb), static_cast<void*>(thr)})
        HIP(hipFree(ptr));
}
}  // namespace

int main(int argc, char** argv) {
    test_shape(1000U, 150U, 0.75F, 1);
    test_shape(1450U, 70U, 1.0F, 2);
    test_shape(4100U, 300U, 0.75F, 3);
    test_shape(4100U, 300U, -3.0F, 4);
    std::printf("sparse router gpu %s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    if (argc > 1 && std::string(argv[1]) == "--bench") bench();
    return failures ? 1 : 0;
}
