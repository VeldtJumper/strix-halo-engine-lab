// GLM prefill per-layer component bench, one TP2 rank (design notes section 4). Synthetic weights at the
// real per-rank GLM shapes and bit widths (EXL3 K6 attention/shared, K5 dense, K4 experts; F16/BF16/F32 plain weights),
// one chunk of C rows. Tier B = strix_gemm_pinned (s = the library's split at M = 1024); Tier A = the gated kernels
// (src/glm/kernels, src/glm/prefill). Exchanges are not included; the reported per-layer sums are compute only.
// Timing only (synthetic values, synthetic routes: record, do not over-read): every component is timed with events (host
// submission, API validation and the MoE host grouping are inside the intervals), best of 3 after one warm pass; sums of
// per-component minima are optimistic estimates, not an end-to-end layer time. A warm pass and the last timed pass are
// compared on every branch output and state (determinism), all finite, no flag; the process exits nonzero otherwise.
// usage: hip_glm_pf_layer_bench C T0_DSA [C T0_DSA ...]   e.g. 1024 0 2048 6144 2048 30720
#include "glm/kernels/glm_dsa.hpp"
#include "glm/kernels/glm_tier_a.hpp"
#include "glm/prefill/glm_pf_moe.hpp"
#include "glm/prefill/glm_prefill.hpp"
#include "strix_trellis.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace strix::glm;
namespace D = strix::glm::dev;
namespace P = strix::glm::pf;

namespace {

[[noreturn]] void die(const std::string& w) {
    std::fprintf(stderr, "FATAL %s\n", w.c_str());
    std::exit(4);
}
void hok(hipError_t e, const char* w) {
    if (e != hipSuccess) die(std::string(w) + ": " + hipGetErrorString(e));
}
void sok(strix_status s, const char* w) {
    if (s != STRIX_SUCCESS) die(std::string(w) + ": " + strix_status_string(s));
}
std::mt19937 g_rng(11);
std::vector<void*> g_allocs;
template <class T> T* dmalloc(std::size_t n) {
    T* p = nullptr;
    hok(hipMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "malloc");
    g_allocs.push_back(p);
    return p;
}
template <class T> T* up(const std::vector<T>& h) {
    T* p = dmalloc<T>(h.size());
    hok(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "h2d");
    return p;
}
std::vector<float> rnd(std::size_t n, float scale) {
    std::normal_distribution<float> d(0.0F, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g_rng);
    return v;
}
float* drnd(std::size_t n, float scale) { return up(rnd(n, scale)); }
std::uint16_t* dhalf(std::size_t n, float scale, bool bf16) {
    const auto f = rnd(n, scale);
    std::vector<std::uint16_t> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (bf16) {
            v[i] = float_to_bf16(f[i]);
        } else {
            const _Float16 h = static_cast<_Float16>(f[i]);
            std::memcpy(&v[i], &h, 2);
        }
    }
    return up(v);
}

// ---------------------------------------------------------------- Tier B: pinned EXL3 linears
strix_context g_ctx{};
hipStream_t g_hs{};
strix_stream g_st{};
void* g_ws = nullptr;
std::uint64_t g_wsb = 0;

struct Lin {
    strix_weights w{};
    unsigned N = 0, K = 0, bits = 0, s = 0;
};
strix_matmul_desc mdesc(const Lin& l, const float* x, float* y, unsigned rows) {
    strix_matmul_desc d;
    strix_matmul_desc_init(&d);
    d.weights = l.w;
    d.input.data = const_cast<float*>(x);
    d.input.rank = 2;
    d.input.shape[0] = rows;
    d.input.shape[1] = l.K;
    d.input.strides[0] = l.K;
    d.input.strides[1] = 1;
    d.input.buffer_bytes = static_cast<std::uint64_t>(rows) * l.K * 4;
    d.output.data = y;
    d.output.rank = 2;
    d.output.shape[0] = rows;
    d.output.shape[1] = l.N;
    d.output.strides[0] = l.N;
    d.output.strides[1] = 1;
    d.output.buffer_bytes = static_cast<std::uint64_t>(rows) * l.N * 4;
    d.workspace = g_ws;
    d.workspace_bytes = g_wsb;
    return d;
}
Lin make_lin(unsigned N, unsigned K, unsigned bits) {
    std::vector<std::uint16_t> tr(static_cast<std::size_t>(K / 16) * (N / 16) * 16 * bits);
    for (auto& v : tr) v = static_cast<std::uint16_t>(g_rng());
    std::vector<std::uint16_t> suh(K), svh(N);
    const _Float16 one = 1.0F, mone = -1.0F, sc = 0.02F;
    for (auto& v : suh) std::memcpy(&v, (g_rng() & 1) ? &one : &mone, 2);
    for (auto& v : svh) std::memcpy(&v, &sc, 2);
    strix_exl3_parts_desc p{};
    p.struct_size = sizeof(p);
    p.abi_version = STRIX_TRELLIS_ABI_VERSION;
    p.transport_version = STRIX_TRELLIS_TRANSPORT_VERSION;
    p.bits_x2 = 2 * bits;
    p.rows = N;
    p.columns = K;
    p.codebook = STRIX_CODEBOOK_MUL1;
    p.marker = STRIX_EXL3_MARKER_MUL1;
    p.trellis = {tr.data(), tr.size() * 2};
    p.suh = {suh.data(), suh.size() * 2};
    p.svh = {svh.data(), svh.size() * 2};
    Lin l;
    sok(strix_weights_prepare_parts(g_ctx, &p, &g_st, &l.w), "prepare");
    l.N = N;
    l.K = K;
    l.bits = bits;
    // the library's split at M = 1024, read back from its workspace size
    float *qx = nullptr, *qy = nullptr;
    hok(hipMalloc(&qx, std::size_t{1024} * K * 4), "m");
    hok(hipMalloc(&qy, std::size_t{1024} * N * 4), "m");
    strix_matmul_desc d = mdesc(l, qx, qy, 1024);
    d.workspace = nullptr;
    d.workspace_bytes = 0;
    std::uint64_t ws = 0;
    sok(strix_matmul_workspace_bytes(g_ctx, &d, &ws), "ws");
    l.s = static_cast<unsigned>((ws - ((std::uint64_t{1024} * K * 2 + 255) & ~std::uint64_t{255})) / (std::uint64_t{1024} * N * 4));
    hok(hipFree(qx), "f");
    hok(hipFree(qy), "f");
    return l;
}
void ensure_ws(const Lin& l, unsigned rows) {
    const std::uint64_t need = std::uint64_t{rows} * l.K * 2 + 256 + std::uint64_t{rows} * l.N * 4 * l.s + 4096;  // >= the pinned formula
    if (need > g_wsb) {
        if (g_ws) hok(hipFree(g_ws), "f");
        hok(hipMalloc(&g_ws, need), "ws");
        g_wsb = need;
    }
}
void gemm(const Lin& l, const float* x, float* y, unsigned rows) {
    ensure_ws(l, rows);
    strix_matmul_desc d = mdesc(l, x, y, rows);
    sok(strix_gemm_pinned(g_ctx, &d, l.s, &g_st), "gemm_pinned");
}

// ---------------------------------------------------------------- timing helper
struct Timer {
    hipEvent_t a{}, b{};
    std::map<std::string, float> best;
    std::vector<std::string> order;
    Timer() {
        hok(hipEventCreate(&a), "ev");
        hok(hipEventCreate(&b), "ev");
    }
    void run(const std::string& name, const std::function<void()>& f, bool record) {
        hok(hipEventRecord(a, g_hs), "ev");
        f();
        hok(hipEventRecord(b, g_hs), "ev");
        hok(hipEventSynchronize(b), "ev");
        float ms = 0;
        hok(hipEventElapsedTime(&ms, a, b), "ev");
        if (!record) return;
        if (!best.count(name)) {
            best[name] = ms;
            order.push_back(name);
        } else {
            best[name] = std::min(best[name], ms);
        }
    }
};

constexpr unsigned kHeadsR = 32;               // KDA / MLA heads per rank
constexpr unsigned kKdaR = kHeadsR * kKdaDim;  // 4096
constexpr unsigned kE = 288, kIL = 1024;       // experts, rank-local expert I

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc % 2 == 0) die("usage: hip_glm_pf_layer_bench C T0_DSA [C T0_DSA ...]");
    for (int a = 1; a + 1 < argc; a += 2) {
        const unsigned long c = std::stoul(argv[a]), t = std::stoul(argv[a + 1]);
        if (c < 16 || c > 4096 || c % 16 || t > 65536 || t % c) die("C in [16, 4096] (multiple of 16), T0 <= 65536 and a multiple of C");
    }
    strix_context_desc cd{};
    cd.struct_size = sizeof(cd);
    cd.abi_version = STRIX_TRELLIS_ABI_VERSION;
    sok(strix_context_create(&cd, &g_ctx), "ctx");
    hok(hipStreamCreate(&g_hs), "stream");
    g_st = {sizeof(strix_stream), STRIX_TRELLIS_ABI_VERSION, g_hs, {}};

    // ---- weights (one layer of each kind; per rank)
    Lin qkv = make_lin(3 * kKdaR, kHidden, 6), o_kda = make_lin(kHidden, kKdaR, 6);
    Lin q_a = make_lin(kQLora, kHidden, 6), kv_a = make_lin(kKvLora, kHidden, 6), q_b = make_lin(kHeadsR * kMlaDim, kQLora, 6);
    Lin wq_b = make_lin(kIdxHeads * kIdxDim, kQLora, 6), o_mla = make_lin(kHidden, kHeadsR * kMlaDim, 6);
    Lin dg = make_lin(kDenseI / 2, kHidden, 5), du = make_lin(kDenseI / 2, kHidden, 5), dd = make_lin(kHidden, kDenseI / 2, 5);
    Lin sg = make_lin(kIL, kHidden, 6), su = make_lin(kIL, kHidden, 6), sd = make_lin(kHidden, kIL, 6);
    std::vector<Lin> eg, eu, ed;
    for (unsigned e = 0; e < kE; ++e) {
        eg.push_back(make_lin(kIL, kHidden, 4));
        eu.push_back(make_lin(kIL, kHidden, 4));
        ed.push_back(make_lin(kHidden, kIL, 4));
    }
    std::vector<P::MoeExpert> experts(kE);
    for (unsigned e = 0; e < kE; ++e)
        experts[e] = {P::moe_exl3_from_weights(eg[e].w, kIL, kHidden, 4), P::moe_exl3_from_weights(eu[e].w, kIL, kHidden, 4),
                      P::moe_exl3_from_weights(ed[e].w, kHidden, kIL, 4)};
    std::printf("splits: qkv %u o_kda %u q_a %u kv_a %u q_b %u wq_b %u o_mla %u dense %u/%u shared %u/%u expert %u/%u\n", qkv.s, o_kda.s, q_a.s, kv_a.s,
                q_b.s, wq_b.s, o_mla.s, dg.s, dd.s, sg.s, sd.s, eg[0].s, ed[0].s);
    const auto fa_w = dhalf(static_cast<std::size_t>(kKdaDim) * kHidden, 0.02F, true), fb_w = dhalf(static_cast<std::size_t>(kKdaR) * kKdaDim, 0.05F, true);
    const auto ga_w = dhalf(static_cast<std::size_t>(kKdaDim) * kHidden, 0.02F, true), gb_w = dhalf(static_cast<std::size_t>(kKdaR) * kKdaDim, 0.05F, true);
    const auto bp_w = dhalf(static_cast<std::size_t>(kHeadsR) * kHidden, 0.02F, true);
    const D::KdaDev kd{kHeadsR, dhalf(static_cast<std::size_t>(3) * kKdaR * kConvK, 0.5F, true), drnd(kHeadsR, 0.5F), drnd(kKdaR, 0.5F),
                       dhalf(kKdaDim, 1.0F, true)};
    std::uint16_t* kv_b = dhalf(static_cast<std::size_t>(kHeadsR) * 2 * kMlaDim * kKvLora, 0.05F, false);
    const D::IndexerDev iw{dhalf(static_cast<std::size_t>(kIdxDim) * kHidden, 0.02F, false), dhalf(static_cast<std::size_t>(kIdxDim) * kHidden, 0.02F, false),
                           dhalf(static_cast<std::size_t>(kIdxHeads) * kHidden, 0.02F, false), dhalf(kIdxDim, 1.0F, false), dhalf(kIdxDim, 0.1F, false),
                           drnd(kKpool * kIdxDim, 0.1F)};
    std::uint16_t* router = dhalf(static_cast<std::size_t>(kExperts) * kHidden, 0.02F, false);
    float* rbias = drnd(kExperts, 0.01F);
    std::uint16_t *qa_nw = dhalf(kQLora, 1.0F, true), *kv_nw = dhalf(kKvLora, 1.0F, true), *norm_w = dhalf(kHidden, 1.0F, true);
    float *hc_fn = drnd(static_cast<std::size_t>(kHcMix) * kHc * kHidden, 0.01F), *hc_base = drnd(kHcMix, 0.1F), *hc_scale = drnd(3, 0.1F);
    const std::size_t keep = g_allocs.size();

    for (int a = 1; a + 1 < argc; a += 2) {
        const unsigned C = static_cast<unsigned>(std::stoul(argv[a])), t0 = static_cast<unsigned>(std::stoul(argv[a + 1]));
        const unsigned cap = t0 + C, kcap = cap / kKpool;
        // ---- activations / state of one chunk
        float* X = drnd(static_cast<std::size_t>(C) * kHc * kHidden, 1.0F);
        float* Xo = dmalloc<float>(static_cast<std::size_t>(C) * kHc * kHidden);
        float* F = dmalloc<float>(static_cast<std::size_t>(C) * kHc * kHidden);
        float* mix = dmalloc<float>(static_cast<std::size_t>(C) * kHcMix);
        float* hz = dmalloc<float>(static_cast<std::size_t>(C) * kHcMix);
        float *xc = dmalloc<float>(static_cast<std::size_t>(C) * kHidden), *xn = dmalloc<float>(static_cast<std::size_t>(C) * kHidden);
        float* yo = dmalloc<float>(static_cast<std::size_t>(C) * kHidden);
        float *yk = dmalloc<float>(static_cast<std::size_t>(C) * kHidden), *yden = dmalloc<float>(static_cast<std::size_t>(C) * kHidden);
        unsigned* flag = dmalloc<unsigned>(1);
        float *S = drnd(static_cast<std::size_t>(kHeadsR) * kKdaDim * kKdaDim, 0.01F), *conv = drnd(static_cast<std::size_t>(3) * kKdaR * 3, 0.1F);
        float* qkvb = dmalloc<float>(static_cast<std::size_t>(C) * 3 * kKdaR);
        float *fa = dmalloc<float>(static_cast<std::size_t>(C) * kKdaDim), *fb = dmalloc<float>(static_cast<std::size_t>(C) * kKdaR);
        float *ga = dmalloc<float>(static_cast<std::size_t>(C) * kKdaDim), *gb = dmalloc<float>(static_cast<std::size_t>(C) * kKdaR);
        float* bb = dmalloc<float>(static_cast<std::size_t>(C) * kHeadsR);
        float *scr = dmalloc<float>(static_cast<std::size_t>(C) * kHeadsR * 4 * kKdaDim), *beta = dmalloc<float>(static_cast<std::size_t>(C) * kHeadsR);
        float *o = dmalloc<float>(static_cast<std::size_t>(C) * kKdaR), *y = dmalloc<float>(static_cast<std::size_t>(C) * kKdaR);
        D::DsaSlot slot{dmalloc<std::uint16_t>(static_cast<std::size_t>(cap) * kKvLora), dmalloc<std::uint16_t>(static_cast<std::size_t>(cap) * kIdxDim),
                        dmalloc<std::uint16_t>(static_cast<std::size_t>(cap) * kIdxDim), dmalloc<float>(static_cast<std::size_t>(kcap) * kIdxDim), cap};
        float *qa = dmalloc<float>(static_cast<std::size_t>(C) * kQLora), *qan = dmalloc<float>(static_cast<std::size_t>(C) * kQLora);
        float *kvl = dmalloc<float>(static_cast<std::size_t>(C) * kKvLora), *kvn = dmalloc<float>(static_cast<std::size_t>(C) * kKvLora);
        float *qb = dmalloc<float>(static_cast<std::size_t>(C) * kHeadsR * kMlaDim), *qI = dmalloc<float>(static_cast<std::size_t>(C) * kIdxHeads * kIdxDim);
        float *ik = dmalloc<float>(static_cast<std::size_t>(C) * kIdxDim), *ig = dmalloc<float>(static_cast<std::size_t>(C) * kIdxDim);
        float* wh = dmalloc<float>(static_cast<std::size_t>(C) * kIdxHeads);
        float *qt = dmalloc<float>(static_cast<std::size_t>(C) * kHeadsR * kKvLora), *e = dmalloc<float>(static_cast<std::size_t>(C) * kHeadsR * P::kEStride);
        float *ob = dmalloc<float>(static_cast<std::size_t>(C) * kHeadsR * kKvLora), *mo = dmalloc<float>(static_cast<std::size_t>(C) * kHeadsR * kMlaDim);
        std::uint32_t *keys = dmalloc<std::uint32_t>(static_cast<std::size_t>(C) * kcap), *tok = dmalloc<std::uint32_t>(static_cast<std::size_t>(C) * P::kMaxTokens);
        std::uint32_t* n = dmalloc<std::uint32_t>(C);
        hok(hipMemset(flag, 0, 4), "z");
        {  // DSA cache filled up to t0 with synthetic rows
            float *fx = drnd(static_cast<std::size_t>(C) * kHidden, 0.5F), *fl = drnd(static_cast<std::size_t>(C) * kKvLora, 1.0F);
            D::dense(fx, iw.wk, D::WType::F16, kHidden, 0, kIdxDim, ik, C, g_hs);
            D::dense(fx, iw.gate, D::WType::F16, kHidden, 0, kIdxDim, ig, C, g_hs);
            for (unsigned s0 = 0; s0 < t0; s0 += C) P::dsa_append_rows(slot, s0, fl, ik, ig, iw, flag, std::min(C, t0 - s0), g_hs);
        }
        float *g1 = dmalloc<float>(static_cast<std::size_t>(C) * kDenseI / 2), *u1 = dmalloc<float>(static_cast<std::size_t>(C) * kDenseI / 2);
        float* a1 = dmalloc<float>(static_cast<std::size_t>(C) * kDenseI / 2);
        float *logits = dmalloc<float>(static_cast<std::size_t>(C) * kExperts), *rw = dmalloc<float>(static_cast<std::size_t>(C) * kTopE);
        std::uint32_t* rids = dmalloc<std::uint32_t>(static_cast<std::size_t>(C) * kTopE);
        float* shared = dmalloc<float>(static_cast<std::size_t>(C) * kHidden);
        const std::size_t R8 = static_cast<std::size_t>(C) * kTopE;
        float *gg = dmalloc<float>(static_cast<std::size_t>(C) * kIL), *uu = dmalloc<float>(static_cast<std::size_t>(C) * kIL);
        float* aa = dmalloc<float>(static_cast<std::size_t>(C) * kIL);
        std::vector<unsigned> off(kE + 1, 0);
        const std::size_t moe_wsb = P::prefill_moe_workspace_bound(C);
        void* moe_ws = dmalloc<unsigned char>(moe_wsb);

        Timer tm;
        auto pass = [&](bool rec) {
            // ---- attention site (mHC pre + norm), shared by every layer kind
            tm.run("A hc_pre + input norm", [&] {
                P::hc_pre_rows(X, hc_fn, hc_base, hc_scale, F, hz, mix, xc, C, g_hs);
                D::rmsnorm(xc, norm_w, D::WType::BF16, kHidden, kEps, xn, C, g_hs);
            }, rec);
            // ---- KDA layer
            tm.run("K qkv gemm K6 4096->12288", [&] { gemm(qkv, xn, qkvb, C); }, rec);
            tm.run("K small R1 chains f/g/b", [&] {
                P::dense_rows(xn, fa_w, D::WType::BF16, kHidden, 0, kKdaDim, fa, C, g_hs);
                P::dense_rows(fa, fb_w, D::WType::BF16, kKdaDim, 0, kKdaR, fb, C, g_hs);
                P::dense_rows(xn, ga_w, D::WType::BF16, kHidden, 0, kKdaDim, ga, C, g_hs);
                P::dense_rows(ga, gb_w, D::WType::BF16, kKdaDim, 0, kKdaR, gb, C, g_hs);
                P::dense_rows(xn, bp_w, D::WType::BF16, kHidden, 0, kHeadsR, bb, C, g_hs);
            }, rec);
            tm.run("K kda_prefill (prep + scan + out)", [&] { P::kda_prefill(kd, qkvb, fb, gb, bb, S, conv, scr, beta, o, y, flag, C, g_hs); }, rec);
            tm.run("K o_proj gemm K6 4096->4096", [&] { gemm(o_kda, y, yk, C); }, rec);
            // ---- DSA layer
            tm.run("M q_a + kv_a gemm K6 + norms", [&] {
                gemm(q_a, xn, qa, C);
                D::rmsnorm(qa, qa_nw, D::WType::BF16, kQLora, kEps, qan, C, g_hs);
                gemm(kv_a, xn, kvl, C);
                D::rmsnorm(kvl, kv_nw, D::WType::BF16, kKvLora, kEps, kvn, C, g_hs);
            }, rec);
            tm.run("M q_b + wq_b gemm K6", [&] {
                gemm(q_b, qan, qb, C);
                gemm(wq_b, qan, qI, C);
            }, rec);
            tm.run("M indexer R1 chains wk/gate/weights_proj", [&] {
                P::dense_rows(xn, iw.wk, D::WType::F16, kHidden, 0, kIdxDim, ik, C, g_hs);
                P::dense_rows(xn, iw.gate, D::WType::F16, kHidden, 0, kIdxDim, ig, C, g_hs);
                P::dense_rows(xn, iw.weights_proj, D::WType::F16, kHidden, 0, kIdxHeads, wh, C, g_hs);
            }, rec);
            tm.run("M dsa_append_rows", [&] { P::dsa_append_rows(slot, t0, kvn, ik, ig, iw, flag, C, g_hs); }, rec);
            tm.run("M indexer_select_rows", [&] { P::indexer_select_rows(slot, t0, qI, wh, keys, kcap, tok, n, C, g_hs); }, rec);
            tm.run("M mla_attention_rows", [&] { P::mla_attention_rows(slot, tok, n, qb, kv_b, kHeadsR, qt, e, ob, mo, C, g_hs); }, rec);
            tm.run("M o_proj gemm K6 8192->4096", [&] { gemm(o_mla, mo, yo, C); }, rec);
            // ---- mHC post (after the exchange) + FFN site pre
            tm.run("F hc_expand + hc_pre + norm", [&] {
                D::hc_expand(X, mix, yo, Xo, flag, C, g_hs);
                P::hc_pre_rows(Xo, hc_fn, hc_base, hc_scale, F, hz, mix, xc, C, g_hs);
                D::rmsnorm(xc, norm_w, D::WType::BF16, kHidden, kEps, xn, C, g_hs);
            }, rec);
            // ---- dense FFN (layers 0-2), rank half
            tm.run("D dense FFN K5 4096->6144->4096", [&] {
                gemm(dg, xn, g1, C);
                gemm(du, xn, u1, C);
                D::swiglu(g1, u1, a1, static_cast<std::size_t>(C) * (kDenseI / 2), g_hs);
                gemm(dd, a1, yden, C);
            }, rec);
            // ---- MoE FFN
            tm.run("E router R1 288x4096 + route", [&] {
                P::dense_rows(xn, router, D::WType::F16, kHidden, 0, kExperts, logits, C, g_hs);
                D::route(logits, rbias, rids, rw, C, g_hs);
            }, rec);
            std::vector<std::uint32_t> ids(R8);
            std::vector<float> wts(R8);
            tm.run("E routes + weights D2H", [&] {
                hok(hipMemcpyAsync(ids.data(), rids, R8 * 4, hipMemcpyDeviceToHost, g_hs), "d2h");
                hok(hipMemcpyAsync(wts.data(), rw, R8 * 4, hipMemcpyDeviceToHost, g_hs), "d2h");
                hok(hipStreamSynchronize(g_hs), "s");
                for (unsigned r = 0; r < C; ++r)  // group sizes for the stats line
                    for (unsigned k = 0; k < kTopE; ++k)
                        if (ids[static_cast<std::size_t>(r) * kTopE + k] >= kE) die("route id out of range");
                std::fill(off.begin(), off.end(), 0U);
                for (std::size_t i = 0; i < R8; ++i) ++off[ids[i] + 1];
                for (unsigned ex = 0; ex < kE; ++ex) off[ex + 1] += off[ex];
            }, rec);
            tm.run("E shared expert K6", [&] {
                gemm(sg, xn, gg, C);
                gemm(su, xn, uu, C);
                D::swiglu(gg, uu, aa, static_cast<std::size_t>(C) * kIL, g_hs);
                gemm(sd, aa, shared, C);
            }, rec);
            tm.run("E routed experts K4 grouped core + combine", [&] {
                P::PrefillMoeView v;
                v.rows = C;
                v.x = xn;
                v.ids = ids.data();
                v.weights = wts.data();
                v.experts = experts.data();
                v.shared = shared;
                v.out = yo;
                P::enqueue_prefill_moe_glm(v, moe_ws, moe_wsb, g_hs);
            }, rec);
            tm.run("F hc_expand (FFN post)", [&] { D::hc_expand(Xo, mix, yo, X, flag, C, g_hs); }, rec);
        };
        // determinism: two passes from the same state must match on every branch output and state (KDA y + o_proj
        // output, S, conv; DSA o_proj output; dense output; final X after MoE), all finite, no flag
        std::vector<float> Sinit(static_cast<std::size_t>(kHeadsR) * kKdaDim * kKdaDim), cinit(static_cast<std::size_t>(3) * kKdaR * 3);
        std::vector<float> Xinit(static_cast<std::size_t>(C) * kHc * kHidden);
        hok(hipMemcpy(Sinit.data(), S, Sinit.size() * 4, hipMemcpyDeviceToHost), "c");
        hok(hipMemcpy(cinit.data(), conv, cinit.size() * 4, hipMemcpyDeviceToHost), "c");
        hok(hipMemcpy(Xinit.data(), X, Xinit.size() * 4, hipMemcpyDeviceToHost), "c");
        auto reset = [&] {
            hok(hipMemcpy(S, Sinit.data(), Sinit.size() * 4, hipMemcpyHostToDevice), "c");
            hok(hipMemcpy(conv, cinit.data(), cinit.size() * 4, hipMemcpyHostToDevice), "c");
            hok(hipMemcpy(X, Xinit.data(), Xinit.size() * 4, hipMemcpyHostToDevice), "c");
            hok(hipMemset(flag, 0, 4), "z");
        };
        auto snap = [&] {
            std::vector<float> v;
            auto add = [&](const float* p, std::size_t n) {
                const std::size_t o0 = v.size();
                v.resize(o0 + n);
                hok(hipMemcpy(v.data() + o0, p, n * 4, hipMemcpyDeviceToHost), "c");
            };
            add(y, static_cast<std::size_t>(C) * kKdaR);
            add(yk, static_cast<std::size_t>(C) * kHidden);
            add(S, Sinit.size());
            add(conv, cinit.size());
            add(mo, static_cast<std::size_t>(C) * kHeadsR * kMlaDim);
            add(yden, static_cast<std::size_t>(C) * kHidden);
            add(X, Xinit.size());
            unsigned f = 0;
            hok(hipMemcpy(&f, flag, 4, hipMemcpyDeviceToHost), "c");
            if (f) die("non-finite flag set in a bench pass");
            for (float x : v)
                if (!std::isfinite(x)) die("non-finite bench output");
            return v;
        };
        reset();
        pass(false);
        const auto out1 = snap();
        {  // canonical list lengths and expert group sizes of this chunk (recorded: synthetic data)
            std::vector<std::uint32_t> nh(C);
            hok(hipMemcpy(nh.data(), n, C * 4, hipMemcpyDeviceToHost), "c");
            const auto mm = std::minmax_element(nh.begin(), nh.end());
            const double mean = std::accumulate(nh.begin(), nh.end(), 0.0) / C;
            unsigned gmin = ~0U, gmax = 0, act = 0;
            for (unsigned ex = 0; ex < kE; ++ex) {
                const unsigned M = off[ex + 1] - off[ex];
                if (M) ++act;
                gmin = std::min(gmin, M);
                gmax = std::max(gmax, M);
            }
            std::printf("   lists n: min %u mean %.0f max %u; expert groups: active %u/288, size min %u max %u mean %.1f\n", *mm.first, mean, *mm.second,
                        act, gmin, gmax, static_cast<double>(C) * kTopE / kE);
        }
        for (int rep = 0; rep < 3; ++rep) {
            reset();
            pass(true);
            unsigned f = 0;  // every measured pass: no flag (checked before the next reset)
            hok(hipMemcpy(&f, flag, 4, hipMemcpyDeviceToHost), "c");
            if (f) die("non-finite flag set in a measured pass");
        }
        const auto out2 = snap();
        const bool det = out1.size() == out2.size() && std::memcmp(out1.data(), out2.data(), out1.size() * 4) == 0;
        {  // KDA split (extra calls; the state S / conv advance, timing only)
            hipEvent_t ke[5];
            for (auto& ev : ke) hok(hipEventCreate(&ev), "ev");
            // scan versions 1 / 3 / 5 interleaved per repetition (same bits)
            constexpr unsigned kSv[3] = {1, 3, 5};
            float us[3][3][4]{};
            for (unsigned rep = 0; rep < 4; ++rep)
                for (unsigned vi = 0; vi < 3; ++vi) {
                    P::set_kda_scan_version(kSv[vi]);
                    P::set_kda_profile(ke);
                    P::kda_prefill(kd, qkvb, fb, gb, bb, S, conv, scr, beta, o, y, flag, C, g_hs);
                    P::set_kda_profile(nullptr);
                    P::set_kda_scan_version(0);
                    hok(hipEventSynchronize(ke[4]), "ev");
                    if (!rep) continue;
                    for (unsigned j = 0; j < 4; ++j) {
                        float ms = 0;
                        hok(hipEventElapsedTime(&ms, ke[j], ke[j + 1]), "ev");
                        us[vi][rep - 1][j] = ms * 1000.0F;
                    }
                }
            const char* nm[4] = {"prep", "conv", "scan", "out"};
            for (unsigned vi = 0; vi < 3; ++vi) {
                std::printf("   KDA split C=%u scan v%u (us/row, median of 3):", C, kSv[vi]);
                for (unsigned j = 0; j < 4; ++j) {
                    float m[3] = {us[vi][0][j], us[vi][1][j], us[vi][2][j]};
                    std::sort(m, m + 3);
                    std::printf(" %s %.3f", nm[j], m[1] / C);
                }
                std::printf("\n");
            }
            for (auto& ev : ke) hok(hipEventDestroy(ev), "ev");
        }
        {  // indexer split: scores vs select, select v1 vs v2 (paired, alternating; wh recomputed before each call since
           // the scorer scales it in place)
            hipEvent_t ie[3];
            for (auto& ev : ie) hok(hipEventCreate(&ev), "ev");
            float us[2][3][2]{};  // v = 0: scores v2 + select v1, v = 1: scores v3 + select v2
            for (unsigned rep = 0; rep < 4; ++rep)
                for (unsigned i = 0; i < 2; ++i) {
                    const unsigned v = rep & 1U ? 1 - i : i;
                    P::dense_rows(xn, iw.weights_proj, D::WType::F16, kHidden, 0, kIdxHeads, wh, C, g_hs);
                    P::set_idx_profile(ie);
                    P::indexer_select_rows_versions(slot, t0, qI, wh, keys, kcap, tok, n, C, g_hs, v + 2, v + 1);
                    P::set_idx_profile(nullptr);
                    hok(hipEventSynchronize(ie[2]), "ev");
                    if (!rep) continue;
                    for (unsigned j = 0; j < 2; ++j) {
                        float ms = 0;
                        hok(hipEventElapsedTime(&ms, ie[j], ie[j + 1]), "ev");
                        us[v][rep - 1][j] = ms * 1000.0F;
                    }
                }
            for (unsigned v = 0; v < 2; ++v) {
                float sc[3], se[3];
                for (unsigned rep = 0; rep < 3; ++rep) {
                    sc[rep] = us[v][rep][0];
                    se[rep] = us[v][rep][1];
                }
                std::sort(sc, sc + 3);
                std::sort(se, se + 3);
                std::printf("   indexer paired median C=%u t0=%u scores v%u %.2f us/layer (%.3f us/row), select v%u %.2f us/layer (%.3f us/row)\n",
                            C, t0, v + 2, sc[1], sc[1] / C, v + 1, se[1], se[1] / C);
            }
            for (auto& ev : ie) hok(hipEventDestroy(ev), "ev");
        }
        {  // MLA stage split (one extra call on the same inputs; outside the measured passes)
            hipEvent_t pe[6];
            for (auto& ev : pe) hok(hipEventCreate(&ev), "ev");
            // Paired stage timings, independent of environment selection. Run with
            // 2048 0 2048 6144 2048 30720 to cover the three qualification positions.
            const unsigned sv[4] = {6, 7, 6, 7}, ov[4] = {7, 7, 7, 7}, gv[4] = {2, 2, 2, 2};  // scores v6 vs v7 (256-row groups), defaults otherwise
            float stage_us[4][3][5]{};
            for (unsigned rep = 0; rep < 4; ++rep) {
                // Alternate pair order to limit systematic warm-cache/clock bias.
                for (unsigned i = 0; i < 4; ++i) {
                    const unsigned v = rep & 1U ? 3 - i : i;
                    P::set_mla_profile(pe);
                    P::mla_attention_rows_versions(slot, tok, n, qb, kv_b, kHeadsR, qt, e, ob, mo, C, g_hs, sv[v], ov[v], gv[v]);
                    P::set_mla_profile(nullptr);
                    hok(hipEventSynchronize(pe[5]), "ev");
                    if (!rep) continue;
                    for (unsigned j = 0; j < 5; ++j) {
                        float ms = 0;
                        hok(hipEventElapsedTime(&ms, pe[j], pe[j + 1]), "ev");
                        stage_us[v][rep - 1][j] = ms * 1000.0F;
                    }
                }
            }
            for (unsigned v = 0; v < 4; ++v) {
                float sc[3], obar[3], total[3], other[3];
                for (unsigned rep = 0; rep < 3; ++rep) {
                    sc[rep] = stage_us[v][rep][1];
                    obar[rep] = stage_us[v][rep][3];
                    other[rep] = stage_us[v][rep][0] + stage_us[v][rep][2] + stage_us[v][rep][4];
                    total[rep] = std::accumulate(stage_us[v][rep], stage_us[v][rep] + 5, 0.0F);
                }
                std::sort(sc, sc + 3);
                std::sort(obar, obar + 3);
                std::sort(total, total + 3);
                std::sort(other, other + 3);
                float st5[5];
                for (unsigned j = 0; j < 5; ++j) {
                    float m[3] = {stage_us[v][0][j], stage_us[v][1][j], stage_us[v][2][j]};
                    std::sort(m, m + 3);
                    st5[j] = m[1] / C;
                }
                std::printf("   MLA stages us/row: absorb %.3f scores %.3f softmax %.3f obar %.3f out %.3f\n", st5[0], st5[1], st5[2], st5[3], st5[4]);
                std::printf("   MLA paired median (gemm v%u) C=%u t0=%u H=%u scores=v%u %.2f us/layer (%.3f us/row), "
                            "obar=v%u %.2f us/layer (%.3f us/row), absorb+softmax+out %.3f us/row, total %.2f us/layer (%.3f us/row)\n",
                            gv[v], C, t0, kHeadsR, sv[v], sc[1], sc[1] / C, ov[v], obar[1], obar[1] / C, other[1] / C, total[1], total[1] / C);
            }
            for (auto& ev : pe) hok(hipEventDestroy(ev), "ev");
        }
        std::printf("== C=%u t0=%u (DSA rows see P up to %u pools, n up to %u tokens); passes bitwise equal: %s\n", C, t0, cap / kKpool,
                    std::min<unsigned>(cap, P::kMaxTokens), det ? "yes" : "NO");
        double kda = 0, dsa = 0, dense = 0, moe = 0, common = 0;
        for (const auto& nm : tm.order) {
            const float ms = tm.best[nm];
            std::printf("  %-48s %9.3f ms  %8.2f us/row\n", nm.c_str(), ms, ms * 1000.0 / C);
            const char k = nm[0];
            if (k == 'K') kda += ms;
            else if (k == 'M') dsa += ms;
            else if (k == 'D') dense += ms;
            else if (k == 'E') moe += ms;
            else common += ms;
        }
        const double site = common;  // A + F rows: per layer once each (attention pre, post+ffn pre, ffn post)
        const double kda_l = kda + site + moe, dsa_l = dsa + site + moe, dense_l = kda + site + dense;
        std::printf("  per layer (compute, no exchange): KDA+MoE %.2f us/row, DSA+MoE %.2f us/row, KDA+dense %.2f us/row\n", kda_l * 1000.0 / C,
                    dsa_l * 1000.0 / C, dense_l * 1000.0 / C);
        // model: layers 0-2 KDA+dense (layer 3 is DSA), 34 KDA (31 with MoE), 11 DSA+MoE
        const double model = 3 * dense_l + 31 * kda_l + 11 * dsa_l;
        std::printf("  model sum (45 layers, this chunk position): %.1f ms per chunk = %.3f ms/token/rank\n", model, model / C);
        if (!det) die("non-deterministic passes");
        for (std::size_t i = keep; i < g_allocs.size(); ++i) hok(hipFree(g_allocs[i]), "f");
        g_allocs.resize(keep);
    }
    return 0;
}
