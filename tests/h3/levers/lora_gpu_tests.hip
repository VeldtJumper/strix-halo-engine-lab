// GPU test of the Turbo LoRA bypass branch (src/h3/levers/lora/bypass.hpp): device == host mirror bitwise (several
// shapes, M not a multiple of 64, padded pitches), batch invariance, the TurboV4Device holder, invalid calls, and
// timing (down / up+add separately) for the qkv / out / fc1 shapes.
// usage: strix-h3-levers-lora-gpu-tests [--bench]
#include "h3/levers/lora/bypass.hpp"
#include "h3/levers/lora/turbo.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace strix::h3::levers::lora;
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
std::vector<std::uint16_t> randbf(std::size_t n, float scale, std::uint64_t seed) {
    Rng r{seed};
    std::vector<std::uint16_t> v(n);
    for (auto& x : v) x = g::f32_to_bf16(r.uni() * scale);
    return v;
}
template <class T> T* up(const std::vector<T>& v) { T* p; HIP(hipMalloc(&p, v.size() * sizeof(T) + 16)); HIP(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
template <class T> std::vector<T> down(const T* p, std::size_t n) { std::vector<T> v(n); HIP(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return v; }

std::vector<std::uint16_t> run_dev(const std::vector<std::uint16_t>& x, std::uint32_t m, std::uint32_t k, std::uint32_t xld, const std::vector<std::uint16_t>& a,
    const std::vector<std::uint16_t>& b, std::uint32_t n, float s, const std::vector<std::uint16_t>& y0, std::uint32_t yld) {
    auto *dx = up(x), *da = up(a), *db = up(b), *dy = up(y0);
    std::uint16_t* dt;
    HIP(hipMalloc(&dt, static_cast<std::size_t>(m) * kRank * 2));
    add_bypass(dx, xld, m, k, da, db, n, s, dt, dy, yld, nullptr);
    HIP(hipDeviceSynchronize());
    auto out = down(dy, y0.size());
    for (void* p : {static_cast<void*>(dx), static_cast<void*>(da), static_cast<void*>(db), static_cast<void*>(dy), static_cast<void*>(dt)}) HIP(hipFree(p));
    return out;
}

void test_shape(std::uint32_t m, std::uint32_t k, std::uint32_t n, std::uint64_t seed) {
    const std::uint32_t xld = k + 8U, yld = n + 16U;
    const auto x0 = randbf(static_cast<std::size_t>(m) * k, 3.0F, seed), a = randbf(static_cast<std::size_t>(kRank) * k, 0.02F, seed + 1),
               b = randbf(static_cast<std::size_t>(n) * kRank, 0.02F, seed + 2), y00 = randbf(static_cast<std::size_t>(m) * n, 1.0F, seed + 3);
    std::vector<std::uint16_t> x(static_cast<std::size_t>(m) * xld, 0x7fc0U), y0(static_cast<std::size_t>(m) * yld, 0x1234U);
    for (std::uint32_t r = 0; r < m; ++r) {
        std::copy(x0.begin() + static_cast<std::ptrdiff_t>(r) * k, x0.begin() + static_cast<std::ptrdiff_t>(r + 1U) * k, x.begin() + static_cast<std::ptrdiff_t>(r) * xld);
        std::copy(y00.begin() + static_cast<std::ptrdiff_t>(r) * n, y00.begin() + static_cast<std::ptrdiff_t>(r + 1U) * n, y0.begin() + static_cast<std::ptrdiff_t>(r) * yld);
    }
    std::vector<std::uint16_t> yh = y00;
    bypass_host(x0.data(), m, k, a.data(), b.data(), n, 0.70F, yh.data());
    const auto yd = run_dev(x, m, k, xld, a, b, n, 0.70F, y0, yld);
    std::size_t bad = 0, pad = 0, changed = 0;
    for (std::uint32_t r = 0; r < m; ++r) {
        for (std::uint32_t c = 0; c < n; ++c) {
            bad += yd[static_cast<std::size_t>(r) * yld + c] != yh[static_cast<std::size_t>(r) * n + c];
            changed += yh[static_cast<std::size_t>(r) * n + c] != y00[static_cast<std::size_t>(r) * n + c];
        }
        for (std::uint32_t c = n; c < yld; ++c) pad += yd[static_cast<std::size_t>(r) * yld + c] != 0x1234U;
    }
    std::printf("bypass M %u K %u N %u: device vs host %zu differ, pitch padding touched %zu, outputs changed by the branch %zu of %zu\n", m, k, n, bad, pad,
        changed, yh.size());
    CHECK(bad == 0 && pad == 0 && changed > 0, "bypass M %u K %u N %u", m, k, n);
}

void test_batch_invariance() {
    constexpr std::uint32_t k = 1024U, n = 512U, ml = 500U, ms = 37U;
    const auto x = randbf(static_cast<std::size_t>(ml) * k, 2.0F, 9), a = randbf(kRank * k, 0.03F, 10), b = randbf(n * kRank, 0.03F, 11),
               y0 = randbf(static_cast<std::size_t>(ml) * n, 1.0F, 12);
    const auto yl = run_dev(x, ml, k, k, a, b, n, 0.7F, y0, n);
    const std::vector<std::uint16_t> xs(x.begin() + 100 * k, x.begin() + (100 + ms) * k), ys0(y0.begin() + 100 * n, y0.begin() + (100 + ms) * n);
    const auto ys = run_dev(xs, ms, k, k, a, b, n, 0.7F, ys0, n);
    std::size_t bad = 0;
    for (std::size_t i = 0; i < ys.size(); ++i) bad += ys[i] != yl[100 * n + i];
    std::printf("bypass batch invariance: rows 100..136 of M %u vs M %u alone: %zu differ\n", ml, ms, bad);
    CHECK(bad == 0, "batch invariance");
}

void test_holder() {
    constexpr std::uint32_t m = 130U, k = 5376U, n = 1024U;
    const auto x = randbf(static_cast<std::size_t>(m) * k, 2.0F, 21), a = randbf(kRank * k, 0.02F, 22), b = randbf(static_cast<std::size_t>(n) * kRank, 0.02F, 23),
               y0 = randbf(static_cast<std::size_t>(m) * n, 1.0F, 24);
    std::vector<std::uint16_t> yh = y0;
    bypass_host(x.data(), m, k, a.data(), b.data(), n, 0.70F, yh.data());
    TurboV4Device turbo(0.70F, 256U);
    turbo.upload(Site::kFc1, 7U, a, b, k, n);
    auto *dx = up(x), *dy = up(y0);
    turbo.apply(Site::kFc1, 7U, dx, k, m, dy, n, nullptr);
    HIP(hipDeviceSynchronize());
    const auto yd = down(dy, y0.size());
    std::size_t bad = 0;
    for (std::size_t i = 0; i < yd.size(); ++i) bad += yd[i] != yh[i];
    int rejected = 0;
    try { turbo.apply(Site::kFc1, 8U, dx, k, m, dy, n, nullptr); } catch (const std::invalid_argument&) { ++rejected; }
    try { turbo.apply(Site::kFc1, 7U, dx, k, 257U, dy, n, nullptr); } catch (const std::invalid_argument&) { ++rejected; }
    try { turbo.upload(Site::kFc1, 7U, a, b, k, n); } catch (const std::invalid_argument&) { ++rejected; }
    try { add_bypass(dx, k, m, 5370U, dx, dx, n, 0.7F, dy, dy, n, nullptr); } catch (const std::invalid_argument&) { ++rejected; }
    try { add_bypass(dx, k, m, k, dx, dx, 96U, 0.7F, dy, dy, n, nullptr); } catch (const std::invalid_argument&) { ++rejected; }
    try { add_bypass(dx + 1, k, m, k, dx, dx, n, 0.7F, dy, dy, n, nullptr); } catch (const std::invalid_argument&) { ++rejected; }
    std::printf("holder (fc1 block 7, M %u): device vs host %zu differ, %d of 6 invalid calls rejected, %zu device bytes\n", m, bad, rejected, turbo.device_bytes());
    CHECK(bad == 0 && rejected == 6, "holder");
    HIP(hipFree(dx)); HIP(hipFree(dy));
}

// Rank-generic bypass (ranks 32 .. 384, multiple rank chunks of 64 and a half chunk): device == host bitwise.
void test_rank(std::uint32_t m, std::uint32_t k, std::uint32_t n, std::uint32_t rank, std::uint64_t seed) {
    const auto x = randbf(static_cast<std::size_t>(m) * k, 2.0F, seed), a = randbf(static_cast<std::size_t>(rank) * k, 0.02F, seed + 1),
               b = randbf(static_cast<std::size_t>(n) * rank, 0.02F, seed + 2), y0 = randbf(static_cast<std::size_t>(m) * n, 1.0F, seed + 3);
    std::vector<std::uint16_t> yh = y0;
    bypass_host(x.data(), m, k, a.data(), b.data(), n, 0.0625F, yh.data(), rank);
    auto *dx = up(x), *da = up(a), *db = up(b), *dy = up(y0);
    std::uint16_t* dt;
    HIP(hipMalloc(&dt, static_cast<std::size_t>(m) * rank * 2));
    add_bypass(dx, k, m, k, da, db, n, 0.0625F, dt, dy, n, nullptr, rank);
    HIP(hipDeviceSynchronize());
    const auto yd = down(dy, y0.size());
    std::size_t bad = 0, changed = 0;
    for (std::size_t i = 0; i < yd.size(); ++i) {
        bad += yd[i] != yh[i];
        changed += yh[i] != y0[i];
    }
    std::printf("bypass rank %u M %u K %u N %u: device vs host %zu differ, outputs changed by the branch %zu of %zu\n", rank, m, k, n, bad, changed, yh.size());
    CHECK(bad == 0 && changed > 0, "rank %u", rank);
    for (void* p : {static_cast<void*>(dx), static_cast<void*>(da), static_cast<void*>(db), static_cast<void*>(dy), static_cast<void*>(dt)}) HIP(hipFree(p));
}

// Holder with per-branch scales and mixed ranks (fc2 site rank 192, qkv site rank 384) == host; rank 40 rejected.
void test_holder_ranks() {
    constexpr std::uint32_t m = 130U, k = 1024U, n = 512U;
    TurboV4Device dev(1.0F, 256U);
    std::size_t bad = 0;
    int rej = 0;
    const Site sites[2] = {Site::kFc2, Site::kQkv};
    const std::uint32_t ranks[2] = {192U, 384U};
    const float scales[2] = {0.3F, 0.0625F};
    for (int c = 0; c < 2; ++c) {
        const std::uint32_t rank = ranks[c];
        const auto x = randbf(static_cast<std::size_t>(m) * k, 2.0F, 41 + c), a = randbf(static_cast<std::size_t>(rank) * k, 0.02F, 51 + c),
                   b = randbf(static_cast<std::size_t>(n) * rank, 0.02F, 61 + c), y0 = randbf(static_cast<std::size_t>(m) * n, 1.0F, 71 + c);
        dev.upload_scaled(sites[c], 5U, a, b, k, n, scales[c]);
        std::vector<std::uint16_t> yh = y0;
        bypass_host(x.data(), m, k, a.data(), b.data(), n, scales[c], yh.data(), rank);
        auto *dx = up(x), *dy = up(y0);
        dev.apply(sites[c], 5U, dx, k, m, dy, n, nullptr);
        HIP(hipDeviceSynchronize());
        const auto yd = down(dy, y0.size());
        for (std::size_t i = 0; i < yd.size(); ++i) bad += yd[i] != yh[i];
        HIP(hipFree(dx)); HIP(hipFree(dy));
    }
    const auto a40 = randbf(40U * k, 0.02F, 81), b40 = randbf(static_cast<std::size_t>(n) * 40U, 0.02F, 82);
    try { dev.upload_scaled(Site::kOut, 5U, a40, b40, k, n, 1.0F); } catch (const std::invalid_argument&) { ++rej; }
    std::printf("holder ranks 192 (fc2) / 384 (qkv), per-branch scales: device vs host %zu differ, rank-40 upload rejected %d\n", bad, rej);
    CHECK(bad == 0 && rej == 1, "holder ranks");
}

void bench() {
    struct S { const char* name; std::uint32_t k, n; };
    for (const S sh : {S{"qkv", 5376U, 21504U}, S{"out", 7168U, 5376U}, S{"fc1", 5376U, 28672U}, S{"refiner fc2", 14336U, 5376U}})
        for (const std::uint32_t m : {19000U, 38000U}) {
            std::uint16_t *x, *a, *b, *y, *t;
            HIP(hipMalloc(&x, static_cast<std::size_t>(m) * sh.k * 2)); HIP(hipMalloc(&a, kRank * sh.k * 2)); HIP(hipMalloc(&b, sh.n * kRank * 2));
            HIP(hipMalloc(&y, static_cast<std::size_t>(m) * sh.n * 2)); HIP(hipMalloc(&t, static_cast<std::size_t>(m) * kRank * 2));
            HIP(hipMemset(x, 0x3c, static_cast<std::size_t>(m) * sh.k * 2)); HIP(hipMemset(a, 0x3c, kRank * sh.k * 2)); HIP(hipMemset(b, 0x3c, sh.n * kRank * 2));
            HIP(hipMemset(y, 0x3c, static_cast<std::size_t>(m) * sh.n * 2));
            hipEvent_t e0, e1, e2; HIP(hipEventCreate(&e0)); HIP(hipEventCreate(&e1)); HIP(hipEventCreate(&e2));
            add_bypass(x, sh.k, m, sh.k, a, b, sh.n, 0.7F, t, y, sh.n, nullptr);
            HIP(hipDeviceSynchronize());
            std::vector<float> md, mu;
            for (int r = 0; r < 5; ++r) {
                HIP(hipEventRecord(e0)); launch_down(x, sh.k, m, sh.k, a, t, nullptr); HIP(hipEventRecord(e1));
                launch_up_add(t, m, b, sh.n, 0.7F, y, sh.n, nullptr); HIP(hipEventRecord(e2)); HIP(hipEventSynchronize(e2));
                float v1, v2; HIP(hipEventElapsedTime(&v1, e0, e1)); HIP(hipEventElapsedTime(&v2, e1, e2)); md.push_back(v1); mu.push_back(v2);
            }
            HIP(hipGetLastError());
            std::sort(md.begin(), md.end()); std::sort(mu.begin(), mu.end());
            const double ybytes = 4.0 * m * sh.n;  // y read + write
            std::printf("bench bypass %-11s M %5u K %5u N %5u: %8.3f ms = down %7.3f (%.1f TF) + up+add %7.3f (%.1f TF, y r+w %.0f GB/s) | int8 GEMM ~%.0f ms at 44 TOPS\n",
                sh.name, m, sh.k, sh.n, md[2] + mu[2], md[2], 2.0 * m * sh.k * kRank / md[2] / 1e9, mu[2], 2.0 * m * sh.n * kRank / mu[2] / 1e9,
                ybytes / mu[2] / 1e6, 2.0 * m * sh.k * sh.n / 44e12 * 1e3);
            HIP(hipEventDestroy(e0)); HIP(hipEventDestroy(e1)); HIP(hipEventDestroy(e2));
            for (void* p : {static_cast<void*>(x), static_cast<void*>(a), static_cast<void*>(b), static_cast<void*>(y), static_cast<void*>(t)}) HIP(hipFree(p));
        }
}
}  // namespace

int main(int argc, char** argv) {
    test_shape(333U, 5376U, 512U, 1);
    test_shape(1U, 14336U, 256U, 2);
    test_shape(70U, 7168U, 1024U, 3);
    test_shape(129U, 5376U, 64U, 4);
    test_shape(64U, 32U, 192U, 5);
    test_batch_invariance();
    test_holder();
    test_rank(129U, 5376U, 192U, 32U, 101);
    test_rank(70U, 1024U, 256U, 128U, 102);
    test_rank(333U, 5376U, 512U, 192U, 103);
    test_rank(64U, 14336U, 128U, 384U, 104);
    test_holder_ranks();
    std::printf("correctness %s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    if (argc > 1 && std::string(argv[1]) == "--bench") bench();
    return failures ? 1 : 0;
}
