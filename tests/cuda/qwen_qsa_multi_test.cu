// Model-free device test of the multi-segment QSA core: qsa::layer_core_multi over several sessions' segments of one
// window must give exactly what per-segment qsa::layer_core gives (attention output, normed/rotated q/k/index rows, the
// stored KV / raw index rows, the pooled keys and the host pooled-valid counts), on random BF16 data, at mixed positions
// below and above the pooled-selection regime (> 2048 positions), several long sessions per window, two consecutive
// windows (the second reuses the pooled caches). Needs a CUDA device; no model.
//   qwen_qsa_multi_test <tables dir>
#include "strix/cuda_qwen_qsa.hpp"

#include "qwen_amd_math_tables.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace qsa = strix::cuda::qwen_qsa;
namespace am = strix::cuda::amdmath;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
std::mt19937 rng(20261102);
float uniform(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }
std::uint16_t to_bf16(float x) { std::uint32_t u; std::memcpy(&u, &x, 4); return static_cast<std::uint16_t>(u >> 16); }
float bf16_value(float x) { std::uint32_t u; std::memcpy(&u, &x, 4); u &= 0xffff0000U; float y; std::memcpy(&y, &u, 4); return y; }

template <class T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    ck(cudaMalloc(&d, std::max<std::size_t>(v.size(), 1) * sizeof(T)), "malloc");
    ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
template <class T> std::vector<T> download(const T* d, std::size_t n) {
    std::vector<T> v(n);
    ck(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}
std::vector<std::uint16_t> bf16_random(std::size_t n, float scale) {
    std::vector<std::uint16_t> v(n);
    for (auto& x : v) x = to_bf16(uniform(-scale, scale));
    return v;
}
std::vector<float> bf16_valued(std::size_t n, float scale) {
    std::vector<float> v(n);
    for (auto& x : v) x = bf16_value(uniform(-scale, scale));
    return v;
}

constexpr std::uint64_t P = 8192;  // positions per session

// One session's layer state (two identical copies are made per case: per-segment and multi).
struct State {
    std::uint16_t *kv = nullptr, *raw = nullptr;
    float* pooled = nullptr;
    std::uint32_t valid = 0;
};
State make_state(const std::vector<std::uint16_t>& kv, const std::vector<std::uint16_t>& raw) {
    State s;
    s.kv = upload(kv);
    s.raw = upload(raw);
    ck(cudaMalloc(&s.pooled, P / 4 * qsa::kIndexDim * sizeof(float)), "pooled");
    ck(cudaMemset(s.pooled, 0, P / 4 * qsa::kIndexDim * sizeof(float)), "pooled clear");
    return s;
}

struct Rows {
    std::uint16_t* hidden; float *q_gate, *k, *v, *iq, *ik, *out;
    qsa::RowBuffers view(std::uint32_t r) const {
        return {hidden + std::uint64_t{r} * qsa::kHidden, q_gate + std::uint64_t{r} * qsa::kQGateWidth, k + std::uint64_t{r} * qsa::kKvWidth,
                v + std::uint64_t{r} * qsa::kKvWidth, iq + std::uint64_t{r} * qsa::kIndexQWidth, ik + std::uint64_t{r} * qsa::kIndexDim,
                out + std::uint64_t{r} * qsa::kAttnWidth};
    }
};
Rows make_rows(const std::vector<std::uint16_t>& hidden, const std::vector<float>& q, const std::vector<float>& k,
               const std::vector<float>& v, std::uint32_t rows) {
    Rows r{};
    r.hidden = upload(hidden);
    r.q_gate = upload(q);
    r.k = upload(k);
    r.v = upload(v);
    ck(cudaMalloc(&r.iq, std::uint64_t{rows} * qsa::kIndexQWidth * 4), "iq");
    ck(cudaMalloc(&r.ik, std::uint64_t{rows} * qsa::kIndexDim * 4), "ik");
    ck(cudaMalloc(&r.out, std::uint64_t{rows} * qsa::kAttnWidth * 4), "out");
    return r;
}

int failures = 0;
template <class T> void expect_same(const T* a, const T* b, std::size_t n, const std::string& what) {
    const auto x = download(a, n), y = download(b, n);
    if (std::memcmp(x.data(), y.data(), n * sizeof(T)) != 0) {
        std::size_t first = 0;
        while (first < n && std::memcmp(&x[first], &y[first], sizeof(T)) == 0) ++first;
        std::printf("    DIFF %s (first at element %zu of %zu)\n", what.c_str(), first, n);
        ++failures;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: qwen_qsa_multi_test <tables dir>\n"); return 1; }
    const auto tables = am::load_tables(argv[1]);
    cudaStream_t stream;
    ck(cudaStreamCreate(&stream), "stream");
    // weights (BF16 indexer projections, F32 norms) and a random rope table (any values: the test compares two paths)
    const auto* wq = upload(bf16_random(std::uint64_t{qsa::kIndexQWidth} * qsa::kHidden, 0.02F));
    const auto* wk = upload(bf16_random(std::uint64_t{qsa::kIndexDim} * qsa::kHidden, 0.02F));
    std::vector<float> qn(qsa::kHeadDim), kn(qsa::kHeadDim), iqn(qsa::kIndexDim), ikn(qsa::kIndexDim);
    for (auto* v : {&qn, &kn, &iqn, &ikn}) for (auto& x : *v) x = uniform(0.5F, 1.5F);
    const qsa::LayerWeights w{wq, wk, upload(qn), upload(kn), upload(iqn), upload(ikn)};
    std::vector<float2> cs(P * qsa::kRotaryPairs);
    for (auto& c : cs) { const float a = uniform(0.0F, 6.2831853F); c = make_float2(std::cos(a), std::sin(a)); }
    const auto* rope = upload(cs);
    auto ws_a = qsa::allocate_workspace(8, P), ws_b = qsa::allocate_workspace(8, P);

    // cases: (first position, rows) per segment; every case runs two consecutive windows
    const std::vector<std::vector<std::pair<std::uint64_t, std::uint32_t>>> cases = {
        {{64, 1}, {2051, 2}, {4097, 3}, {2600, 1}},
        {{4096, 2}, {63, 1}, {2999, 3}},
        {{5000, 1}, {10, 1}, {2100, 1}, {3000, 1}, {64, 1}, {4500, 1}, {2048, 1}, {7000, 1}},
        {{2047, 4}, {2049, 4}},
        {{3, 2}, {5, 3}, {700, 3}},
    };
    for (std::size_t c = 0; c < cases.size(); ++c) {
        auto segs = cases[c];
        const auto n = static_cast<std::uint32_t>(segs.size());
        std::vector<State> a, b;
        for (const auto& sg : segs) {
            auto kv = bf16_random(P * 1024, 1.0F), raw = bf16_random(P * 128, 1.0F);
            a.push_back(make_state(kv, raw));
            b.push_back(make_state(kv, raw));
        }
        for (int window = 0; window < 2; ++window) {
            std::uint32_t rows = 0;
            for (const auto& sg : segs) rows += sg.second;
            const auto hidden = bf16_random(std::uint64_t{rows} * qsa::kHidden, 1.0F);
            const auto q = bf16_valued(std::uint64_t{rows} * qsa::kQGateWidth, 2.0F);
            const auto k = bf16_valued(std::uint64_t{rows} * qsa::kKvWidth, 2.0F);
            const auto v = bf16_valued(std::uint64_t{rows} * qsa::kKvWidth, 2.0F);
            const Rows ra = make_rows(hidden, q, k, v, rows), rb = make_rows(hidden, q, k, v, rows);
            // per segment
            std::uint32_t off = 0;
            for (std::uint32_t i = 0; i < n; ++i) {
                const qsa::LayerState st{a[i].kv, a[i].raw, a[i].pooled, &a[i].valid};
                qsa::layer_core(ra.view(off), segs[i].second, segs[i].first, w, st, rope, tables, ws_a, stream);
                off += segs[i].second;
            }
            // multi
            std::vector<qsa::SegmentState> ms;
            for (std::uint32_t i = 0; i < n; ++i)
                ms.push_back(qsa::SegmentState{segs[i].second, segs[i].first, qsa::LayerState{b[i].kv, b[i].raw, b[i].pooled, &b[i].valid}});
            qsa::layer_core_multi(rb.view(0), ms.data(), n, w, rope, tables, ws_b, stream);
            ck(cudaStreamSynchronize(stream), "case");
            const int before = failures;
            const std::string tag = "case " + std::to_string(c) + " window " + std::to_string(window);
            expect_same(ra.out, rb.out, std::uint64_t{rows} * qsa::kAttnWidth, tag + " attention output");
            expect_same(ra.q_gate, rb.q_gate, std::uint64_t{rows} * qsa::kQGateWidth, tag + " q");
            expect_same(ra.k, rb.k, std::uint64_t{rows} * qsa::kKvWidth, tag + " k");
            expect_same(ra.iq, rb.iq, std::uint64_t{rows} * qsa::kIndexQWidth, tag + " index q");
            expect_same(ra.ik, rb.ik, std::uint64_t{rows} * qsa::kIndexDim, tag + " index k");
            for (std::uint32_t i = 0; i < n; ++i) {
                const std::string s = tag + " segment " + std::to_string(i);
                expect_same(a[i].kv, b[i].kv, P * 1024, s + " kv");
                expect_same(a[i].raw, b[i].raw, P * 128, s + " raw index");
                expect_same(a[i].pooled, b[i].pooled, P / 4 * qsa::kIndexDim, s + " pooled keys");
                if (a[i].valid != b[i].valid) { std::printf("    DIFF %s pooled valid %u vs %u\n", s.c_str(), a[i].valid, b[i].valid); ++failures; }
            }
            std::printf("%s (%u segments, %u rows, pooled valid", tag.c_str(), n, rows);
            for (std::uint32_t i = 0; i < n; ++i) std::printf(" %u", b[i].valid);
            std::printf(") %s\n", failures == before ? "BIT-IDENTICAL" : "DIFFER");
            for (auto* p : {static_cast<void*>(ra.hidden), static_cast<void*>(ra.q_gate), static_cast<void*>(ra.k), static_cast<void*>(ra.v),
                            static_cast<void*>(ra.iq), static_cast<void*>(ra.ik), static_cast<void*>(ra.out), static_cast<void*>(rb.hidden),
                            static_cast<void*>(rb.q_gate), static_cast<void*>(rb.k), static_cast<void*>(rb.v), static_cast<void*>(rb.iq),
                            static_cast<void*>(rb.ik), static_cast<void*>(rb.out)})
                cudaFree(p);
            for (auto& sg : segs) sg.first += sg.second;  // the next window continues every session
        }
        for (auto& s : a) { cudaFree(s.kv); cudaFree(s.raw); cudaFree(s.pooled); }
        for (auto& s : b) { cudaFree(s.kv); cudaFree(s.raw); cudaFree(s.pooled); }
    }
    std::printf("%s (%d differences)\n", failures == 0 ? "QSA MULTI PASS" : "QSA MULTI FAIL", failures);
    return failures == 0 ? 0 : 3;
}
