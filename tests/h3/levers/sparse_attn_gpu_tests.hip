// Gate of the sparse K1 variant (src/h3/levers/sparse/attn_sparse.hip) against K1 (h3/attn b0a559848):
//   1. full LUT == dense K1 bitwise (partials o / mc / l and the merged bf16 output) with identical cuts
//      (cut = 128 floor(S / 256)), several S incl. a partial last tile, V^T pitch padding filled with NaN;
//   2. reverse launch order (block 1 first, separate launches) == the same bits;
//   3. empty half: block 1's lists empty -> its partial is exactly (-inf, 0, +0);
//   4. a uniform tile subset == dense K1 over the gathered keys of those tiles (same bits: same steps, same order);
//   5. malformed LUT entries (out of range, duplicate, descending) are skipped and counted, the rest == the cleaned LUT.
// usage: strix-h3-levers-sparse-attn-gpu-tests [--bench]
#include "h3/levers/sparse/attn_sparse.hpp"
#include "h3/levers/sparse/sparse_attention.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "h3/gemm/ref.hpp"

using namespace strix::h3;
using levers::sparse::BlockLut;

namespace {
int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { ++failures; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
#define HIP(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s: %s\n", #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

constexpr unsigned H = 56U, D = 128U;
constexpr std::uint64_t W = static_cast<std::uint64_t>(H) * D;

struct Rng {
    std::uint64_t s;
    std::uint64_t next() { std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL); z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL; return z ^ (z >> 31U); }
    float uni() { return static_cast<float>(next() >> 40U) * 0x1p-24F * 2.0F - 1.0F; }
};
std::vector<std::uint16_t> randbf(std::size_t n, float scale, std::uint64_t seed) {
    Rng r{seed};
    std::vector<std::uint16_t> v(n);
    for (auto& x : v) x = gemm::f32_to_bf16(r.uni() * scale);
    return v;
}
template <class T> T* up(const std::vector<T>& v) { T* p; HIP(hipMalloc(&p, v.size() * sizeof(T) + 64)); if (!v.empty()) HIP(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
template <class T> std::vector<T> down(const T* p, std::size_t n) { std::vector<T> v(n); HIP(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return v; }

// Q / K token-major [S][H][D]; V^T [H][D][pitch] with the pitch padding filled with NaN.
struct Problem {
    std::uint32_t S = 0, cut = 0;
    std::uint64_t pitch = 0;
    std::uint16_t *q = nullptr, *k = nullptr, *vt = nullptr;
    std::vector<std::uint16_t> hq, hk, hv;  // host copies (V token-major)
    void make(std::uint32_t s, std::uint64_t seed) {
        S = s;
        cut = 128U * (S / 256U);
        pitch = (S + 63ULL) / 64ULL * 64ULL;
        hq = randbf(static_cast<std::size_t>(S) * W, 1.0F, seed);
        hk = randbf(static_cast<std::size_t>(S) * W, 1.0F, seed + 1);
        hv = randbf(static_cast<std::size_t>(S) * W, 1.0F, seed + 2);
        std::vector<std::uint16_t> vt(static_cast<std::size_t>(H) * D * pitch, 0x7fc0U);
        for (std::uint32_t j = 0; j < S; ++j)
            for (std::uint64_t e = 0; e < W; ++e) vt[e * pitch + j] = hv[j * W + e];
        q = up(hq);
        k = up(hk);
        this->vt = up(vt);
    }
    attn::KeyBlock block(std::uint32_t j0, std::uint32_t j1) const {
        return attn::KeyBlock{k + j0 * W, vt + j0, D, W, D * pitch, pitch, j1 - j0};
    }
    void free() { HIP(hipFree(q)); HIP(hipFree(k)); HIP(hipFree(vt)); }
};
attn::Queries queries(const Problem& P) { return attn::Queries{P.q, D, W, P.S, H, D}; }

struct Parts {
    float *o = nullptr, *ml = nullptr;
    std::uint32_t rows = 0, slots = 0;
    void alloc(std::uint32_t r, std::uint32_t s) {
        rows = r;
        slots = s;
        HIP(hipMalloc(&o, static_cast<std::size_t>(s) * H * r * D * 4));
        HIP(hipMalloc(&ml, static_cast<std::size_t>(s) * H * r * 2 * 4));
        HIP(hipMemset(o, 0x55, static_cast<std::size_t>(s) * H * r * D * 4));
        HIP(hipMemset(ml, 0x55, static_cast<std::size_t>(s) * H * r * 2 * 4));
    }
    attn::Partials p() const { return attn::Partials{o, ml, rows, H, D, nullptr}; }
    std::vector<float> ho() const { return down(o, static_cast<std::size_t>(slots) * H * rows * D); }
    std::vector<float> hml() const { return down(ml, static_cast<std::size_t>(slots) * H * rows * 2); }
    void free() { HIP(hipFree(o)); HIP(hipFree(ml)); }
};
std::size_t ndiff(const std::vector<float>& a, const std::vector<float>& b) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < a.size(); ++i) n += std::memcmp(&a[i], &b[i], 4) != 0;
    return n;
}

// Host LUT: every (query tile, head) list = `tiles` (global indices).
struct HostLut {
    std::vector<std::int32_t> entries;
    std::vector<std::uint32_t> offsets;
    std::int32_t* de = nullptr;
    std::uint32_t* doff = nullptr;
    void same_for_all(std::uint32_t qtiles, const std::vector<std::int32_t>& tiles) {
        entries.clear();
        offsets.assign(1, 0U);
        for (std::uint32_t l = 0; l < qtiles * H; ++l) {
            entries.insert(entries.end(), tiles.begin(), tiles.end());
            offsets.push_back(static_cast<std::uint32_t>(entries.size()));
        }
        upload();
    }
    void upload() {
        std::vector<std::int32_t> e = entries;
        if (e.empty()) e.push_back(0);
        de = up(e);
        doff = up(offsets);
    }
    BlockLut lut(std::uint32_t tile0) const { return BlockLut{de, doff, tile0}; }
    void free() { HIP(hipFree(de)); HIP(hipFree(doff)); }
};
std::vector<std::int32_t> range(std::uint32_t a, std::uint32_t b) {
    std::vector<std::int32_t> v;
    for (std::uint32_t j = a; j < b; ++j) v.push_back(static_cast<std::int32_t>(j));
    return v;
}
std::vector<std::uint16_t> merged(const Parts& pp, std::uint32_t nslots) {
    std::uint16_t* out;
    HIP(hipMalloc(&out, static_cast<std::size_t>(pp.rows) * W * 2));
    attn::launch_merge(pp.p(), nslots, attn::Output{out, nullptr, W, nullptr}, nullptr);
    HIP(hipDeviceSynchronize());
    auto v = down(out, static_cast<std::size_t>(pp.rows) * W);
    HIP(hipFree(out));
    return v;
}

void test_full_lut(std::uint32_t S, std::uint64_t seed) {
    Problem P;
    P.make(S, seed);
    const attn::Queries q = queries(P);
    const std::uint32_t qt = (S + 127U) / 128U, jc = P.cut / 64U, kt = (S + 63U) / 64U;
    const attn::KeyBlock blocks[2] = {P.block(0, P.cut), P.block(P.cut, S)};
    Parts dense, sp, rev;
    dense.alloc(S, 2);
    sp.alloc(S, 2);
    rev.alloc(S, 2);
    attn::launch_partials(q, blocks, 2, dense.p(), 0, nullptr);
    HostLut l0, l1;
    l0.same_for_all(qt, range(0, jc));
    l1.same_for_all(qt, range(jc, kt));
    const BlockLut luts[2] = {l0.lut(0), l1.lut(jc)};
    unsigned* err;
    HIP(hipMalloc(&err, 4)); HIP(hipMemset(err, 0, 4));
    levers::sparse::launch_sparse_partials(q, blocks, luts, 2, sp.p(), 0, err, nullptr);
    // reverse launch order: block 1 alone into slot 1 first, then block 0 into slot 0
    levers::sparse::launch_sparse_partials(q, &blocks[1], &luts[1], 1, rev.p(), 1, err, nullptr);
    levers::sparse::launch_sparse_partials(q, &blocks[0], &luts[0], 1, rev.p(), 0, err, nullptr);
    HIP(hipDeviceSynchronize());
    const auto d_o = dense.ho(), d_ml = dense.hml();
    const std::size_t bo = ndiff(sp.ho(), d_o), bml = ndiff(sp.hml(), d_ml), ro = ndiff(rev.ho(), d_o), rml = ndiff(rev.hml(), d_ml);
    const auto od = merged(dense, 2), os = merged(sp, 2);
    std::size_t bout = 0;
    for (std::size_t i = 0; i < od.size(); ++i) bout += od[i] != os[i];
    const unsigned errs = down(err, 1)[0];
    std::printf("full LUT == dense: S %u cut %u (block 1 last tile %u keys): partial o %zu ml %zu differ, merged out %zu differ; reverse order o %zu ml %zu; "
                "lut_errors %u\n", S, P.cut, (S - P.cut) % 64U ? (S - P.cut) % 64U : 64U, bo, bml, bout, ro, rml, errs);
    CHECK(bo == 0 && bml == 0 && bout == 0 && ro == 0 && rml == 0 && errs == 0, "full LUT S %u", S);
    // the same with a 4-entry LDS stage (entries past the stage read from global memory)
    Parts sm;
    sm.alloc(S, 2);
    levers::sparse::launch_sparse_partials_small_stage_for_test(q, blocks, luts, 2, sm.p(), 0, err, nullptr);
    HIP(hipDeviceSynchronize());
    const std::size_t so = ndiff(sm.ho(), d_o), sml = ndiff(sm.hml(), d_ml);
    const unsigned errs2 = down(err, 1)[0];
    std::printf("full LUT == dense with a 4-entry stage (global-read path): o %zu ml %zu differ, lut_errors %u\n", so, sml, errs2);
    CHECK(so == 0 && sml == 0 && errs2 == 0, "small stage S %u", S);
    sm.free();
    HIP(hipFree(err));
    l0.free(); l1.free(); dense.free(); sp.free(); rev.free(); P.free();
}

void test_empty_half() {
    Problem P;
    P.make(1000U, 21);
    const attn::Queries q = queries(P);
    const std::uint32_t qt = 8U, jc = P.cut / 64U, kt = 16U;
    const attn::KeyBlock blocks[2] = {P.block(0, P.cut), P.block(P.cut, P.S)};
    HostLut l0, l1;
    l0.same_for_all(qt, range(0, jc));
    l1.same_for_all(qt, {});
    const BlockLut luts[2] = {l0.lut(0), l1.lut(jc)};
    Parts sp;
    sp.alloc(P.S, 2);
    levers::sparse::launch_sparse_partials(q, blocks, luts, 2, sp.p(), 0, nullptr, nullptr);
    HIP(hipDeviceSynchronize());
    const auto o = sp.ho(), ml = sp.hml();
    const std::size_t half = static_cast<std::size_t>(H) * P.S;
    std::size_t bad = 0;
    for (std::size_t i = half * D; i < 2 * half * D; ++i) bad += std::memcmp(&o[i], "\0\0\0\0", 4) != 0;  // +0
    for (std::size_t r = half; r < 2 * half; ++r) bad += !(ml[2 * r] == -__builtin_inff()) || std::memcmp(&ml[2 * r + 1], "\0\0\0\0", 4) != 0;
    // block 0 partial == dense block 0
    Parts dense;
    dense.alloc(P.S, 1);
    attn::launch_partials(q, blocks, 1, dense.p(), 0, nullptr);
    HIP(hipDeviceSynchronize());
    const auto d_o = dense.ho(), d_ml = dense.hml();
    const std::size_t b0 = ndiff(std::vector<float>(o.begin(), o.begin() + static_cast<std::ptrdiff_t>(half * D)), d_o) +
                           ndiff(std::vector<float>(ml.begin(), ml.begin() + static_cast<std::ptrdiff_t>(half * 2)), d_ml);
    std::printf("empty half: block-1 partial (-inf, 0, +0) mismatches %zu; block-0 partial vs dense %zu differ (S %u, %u query tiles, %u tiles)\n", bad, b0,
        P.S, qt, kt);
    CHECK(bad == 0 && b0 == 0, "empty half");
    l0.free(); l1.free(); sp.free(); dense.free(); P.free();
}

// Subset of tiles (same for every list) == dense K1 over the gathered keys of those tiles.
void test_subset() {
    Problem P;
    P.make(1450U, 31);  // KT 23, block 1 last tile 42 keys
    const attn::Queries q = queries(P);
    const std::uint32_t qt = (P.S + 127U) / 128U, jc = P.cut / 64U, kt = (P.S + 63U) / 64U;
    const std::vector<std::int32_t> s0 = {1, 4, 5}, s1 = {static_cast<std::int32_t>(jc) + 2, static_cast<std::int32_t>(kt) - 1};
    HostLut l0, l1;
    l0.same_for_all(qt, s0);
    l1.same_for_all(qt, s1);
    const attn::KeyBlock blocks[2] = {P.block(0, P.cut), P.block(P.cut, P.S)};
    const BlockLut luts[2] = {l0.lut(0), l1.lut(jc)};
    Parts sp;
    sp.alloc(P.S, 2);
    levers::sparse::launch_sparse_partials(q, blocks, luts, 2, sp.p(), 0, nullptr, nullptr);
    // gathered keys: block 0 = tiles 1, 4, 5 (192 keys), block 1 = tile jc + 2 (64) + the partial last tile
    const auto gather = [&](const std::vector<std::int32_t>& tiles, Problem& G) {
        std::vector<std::uint32_t> rows;
        for (const std::int32_t t : tiles)
            for (std::uint32_t r = 0; r < 64U; ++r)
                if (64U * static_cast<std::uint32_t>(t) + r < P.S) rows.push_back(64U * static_cast<std::uint32_t>(t) + r);
        G.S = static_cast<std::uint32_t>(rows.size());
        G.pitch = (G.S + 63ULL) / 64ULL * 64ULL;
        std::vector<std::uint16_t> k(G.S * W), vt(static_cast<std::size_t>(H) * D * G.pitch, 0x7fc0U);
        for (std::uint32_t i = 0; i < G.S; ++i)
            for (std::uint64_t e = 0; e < W; ++e) {
                k[i * W + e] = P.hk[rows[i] * W + e];
                vt[e * G.pitch + i] = P.hv[rows[i] * W + e];
            }
        G.k = up(k);
        G.vt = up(vt);
        G.q = nullptr;
    };
    Problem G0, G1;
    gather(s0, G0);
    gather(s1, G1);
    const attn::KeyBlock gb[2] = {G0.block(0, G0.S), G1.block(0, G1.S)};
    Parts dense;
    dense.alloc(P.S, 2);
    attn::launch_partials(q, gb, 2, dense.p(), 0, nullptr);
    HIP(hipDeviceSynchronize());
    const std::size_t bo = ndiff(sp.ho(), dense.ho()), bml = ndiff(sp.hml(), dense.hml());
    std::printf("subset {1,4,5} / {%u, %u (partial %u keys)} == dense over gathered keys: o %zu ml %zu differ\n", jc + 2U, kt - 1U, P.S - 64U * (kt - 1U), bo, bml);
    CHECK(bo == 0 && bml == 0, "subset");
    HIP(hipFree(G0.k)); HIP(hipFree(G0.vt)); HIP(hipFree(G1.k)); HIP(hipFree(G1.vt));
    l0.free(); l1.free(); sp.free(); dense.free(); P.free();
}

void test_malformed() {
    Problem P;
    P.make(1000U, 41);
    const attn::Queries q = queries(P);
    const std::uint32_t qt = 8U, jc = P.cut / 64U, kt = 16U;
    const attn::KeyBlock blocks[2] = {P.block(0, P.cut), P.block(P.cut, P.S)};
    // block 0 lists: {-1, 0, 2, 2, 1, 3, 99} -> valid {0, 2, 3}; 4 errors per list
    HostLut bad0, good0, l1;
    bad0.same_for_all(qt, {-1, 0, 2, 2, 1, 3, 99});
    good0.same_for_all(qt, {0, 2, 3});
    l1.same_for_all(qt, range(jc, kt));
    const BlockLut lb[2] = {bad0.lut(0), l1.lut(jc)}, lg[2] = {good0.lut(0), l1.lut(jc)};
    Parts a, b;
    a.alloc(P.S, 2);
    b.alloc(P.S, 2);
    unsigned* err;
    HIP(hipMalloc(&err, 4)); HIP(hipMemset(err, 0, 4));
    levers::sparse::launch_sparse_partials(q, blocks, lb, 2, a.p(), 0, err, nullptr);
    levers::sparse::launch_sparse_partials(q, blocks, lg, 2, b.p(), 0, nullptr, nullptr);
    HIP(hipDeviceSynchronize());
    const unsigned errs = down(err, 1)[0], want = 4U * qt * H;
    const std::size_t d = ndiff(a.ho(), b.ho()) + ndiff(a.hml(), b.hml());
    std::printf("malformed LUT: %u errors counted (want %u), result vs cleaned LUT %zu differ\n", errs, want, d);
    CHECK(errs == want && d == 0, "malformed LUT");
    {  // the malformed entries past a 4-entry stage go through the global-read path
        Parts c;
        c.alloc(P.S, 2);
        HIP(hipMemset(err, 0, 4));
        levers::sparse::launch_sparse_partials_small_stage_for_test(q, blocks, lb, 2, c.p(), 0, err, nullptr);
        HIP(hipDeviceSynchronize());
        const unsigned e2 = down(err, 1)[0];
        const std::size_t d2 = ndiff(c.ho(), b.ho()) + ndiff(c.hml(), b.hml());
        std::printf("malformed LUT, 4-entry stage: %u errors counted (want %u), result vs cleaned LUT %zu differ\n", e2, want, d2);
        CHECK(e2 == want && d2 == 0, "malformed LUT small stage");
        c.free();
    }
    int rej = 0;
    try { levers::sparse::launch_sparse_partials(q, blocks, lb, 0, a.p(), 0, nullptr, nullptr); } catch (const std::invalid_argument&) { ++rej; }
    BlockLut nolut[2] = {BlockLut{nullptr, nullptr, 0}, l1.lut(jc)};
    try { levers::sparse::launch_sparse_partials(q, blocks, nolut, 2, a.p(), 0, nullptr, nullptr); } catch (const std::invalid_argument&) { ++rej; }
    CHECK(rej == 2, "launcher validation");
    HIP(hipFree(err));
    bad0.free(); good0.free(); l1.free(); a.free(); b.free(); P.free();
}

// Runner hook (sparse_attention.hpp): in the dense window the call == dense K1 at the lever cut (bitwise); outside it
// the call == router + sparse K1 + merge done by hand (bitwise); call / layer counting and telemetry.
void test_engine() {
    Problem P;
    P.make(1450U, 61);
    const std::uint32_t prefix = 70U;
    const levers::sparse::Geometry g = levers::sparse::geometry(P.S, prefix);
    const std::uint32_t jc = g.cut / 64U;
    std::uint16_t *o_eng, *o_ref;
    HIP(hipMalloc(&o_eng, static_cast<std::size_t>(P.S) * W * 2));
    HIP(hipMalloc(&o_ref, static_cast<std::size_t>(P.S) * W * 2));
    levers::sparse::Params prm;  // dense_calls 3, dense_layers 2
    levers::sparse::SparseAttention sa(prm);
    sa.begin_run();
    // call 0, layer 0: dense window
    sa.begin_forward(prefix);
    sa.run(P.q, P.k, P.vt, P.pitch, P.S, P.S, o_eng, nullptr);
    Parts sc;
    sc.alloc(P.S, 2);
    const std::uint32_t cuts[3] = {0U, g.cut, P.S};
    attn::AttentionCall c;
    c.q = P.q; c.k = P.k; c.vt = P.vt; c.rows = P.S; c.keys = P.S; c.vt_pitch = P.pitch; c.cuts = cuts; c.nblocks = 2; c.scratch = sc.o; c.out = o_ref;
    float* big;
    HIP(hipMalloc(&big, attn::attention_scratch_floats(P.S, H, D, 2) * 4));
    c.scratch = big;
    attn::attention(c, nullptr);
    HIP(hipDeviceSynchronize());
    std::size_t bd = 0;
    {
        const auto a = down(o_eng, static_cast<std::size_t>(P.S) * W), b = down(o_ref, static_cast<std::size_t>(P.S) * W);
        for (std::size_t i = 0; i < a.size(); ++i) bd += a[i] != b[i];
    }
    sa.end_forward(nullptr);
    sa.begin_forward(prefix);
    sa.end_forward(nullptr);
    sa.begin_forward(prefix);
    sa.end_forward(nullptr);
    // call 3: layers 0, 1 dense, layer 2 sparse
    sa.begin_forward(prefix);
    sa.run(P.q, P.k, P.vt, P.pitch, P.S, P.S, o_eng, nullptr);
    sa.run(P.q, P.k, P.vt, P.pitch, P.S, P.S, o_eng, nullptr);
    sa.run(P.q, P.k, P.vt, P.pitch, P.S, P.S, o_eng, nullptr);
    sa.end_forward(nullptr);
    // manual sparse path
    namespace s = levers::sparse;
    std::uint16_t* kc;
    float *qb, *st, *mv, *thr;
    HIP(hipMalloc(&kc, static_cast<std::size_t>(g.KT) * W * 2)); HIP(hipMalloc(&qb, static_cast<std::size_t>(g.KT) * W * 4)); HIP(hipMalloc(&st, 4 * W * 4));
    HIP(hipMalloc(&mv, 2 * W * 4)); HIP(hipMalloc(&thr, static_cast<std::size_t>(g.KT) * H * 4));
    s::launch_key_centroids(P.k, W, g, 0, jc, kc, nullptr);
    s::launch_key_centroids(P.k + static_cast<std::size_t>(g.cut) * W, W, g, jc, g.KT, kc + static_cast<std::size_t>(jc) * W, nullptr);
    s::launch_stats_partial(kc, jc, st, st + W, nullptr);
    s::launch_stats_partial(kc + static_cast<std::size_t>(jc) * W, g.KT - jc, st + 2 * W, st + 3 * W, nullptr);
    s::launch_stats_finalize(st, st + W, st + 2 * W, st + 3 * W, g.KT, mv, mv + W, nullptr);
    s::launch_query_centroids(P.q, W, g, 0, g.KT, qb, nullptr);
    s::launch_thresholds(qb, g.KT, mv, mv + W, prm.tau, thr, nullptr);
    const s::DeviceLut l0 = s::build_lut(g, prm, 0, g.QT, qb, thr, 0, jc, kc, nullptr);
    const s::DeviceLut l1 = s::build_lut(g, prm, 0, g.QT, qb, thr, jc, g.KT, kc + static_cast<std::size_t>(jc) * W, nullptr);
    const BlockLut luts[2] = {BlockLut{l0.entries.as<std::int32_t>(), l0.offsets.as<std::uint32_t>(), 0},
                              BlockLut{l1.entries.as<std::int32_t>(), l1.offsets.as<std::uint32_t>(), jc}};
    const attn::KeyBlock blocks[2] = {P.block(0, g.cut), P.block(g.cut, P.S)};
    levers::sparse::launch_sparse_partials(queries(P), blocks, luts, 2, sc.p(), 0, nullptr, nullptr);
    attn::launch_merge(sc.p(), 2, attn::Output{o_ref, nullptr, W, nullptr}, nullptr);
    HIP(hipDeviceSynchronize());
    std::size_t bs = 0;
    {
        const auto a = down(o_eng, static_cast<std::size_t>(P.S) * W), b = down(o_ref, static_cast<std::size_t>(P.S) * W);
        for (std::size_t i = 0; i < a.size(); ++i) bs += a[i] != b[i];
    }
    const auto& t = sa.telemetry();
    const bool tele_ok = t.dense_calls == 3U && t.sparse_calls == 1U && t.layer_density.size() == 3U && t.layer_density[0] < 0.0 && t.layer_density[1] < 0.0 &&
                         t.layer_density[2] > 0.0 && sa.model_call() == 4U;
    int rej = 0;
    try { sa.run(P.q, P.k, P.vt, P.pitch, P.S, P.S, o_eng, nullptr); } catch (const std::logic_error&) { ++rej; }  // outside a forward
    sa.begin_forward(prefix);
    try { sa.begin_forward(prefix); } catch (const std::logic_error&) { ++rej; }
    try { sa.run(P.q, P.k, P.vt, P.pitch, P.S - 64U, P.S, o_eng, nullptr); } catch (const std::invalid_argument&) { ++rej; }  // rows != keys
    std::printf("engine: dense-window call vs K1 at the lever cut %zu differ; sparse call (call 3, layer 2) vs manual path %zu differ; density %.3f; "
                "telemetry ok %d; %d of 3 misuse rejected\n", bd, bs, t.mean_density(), tele_ok ? 1 : 0, rej);
    CHECK(bd == 0 && bs == 0 && tele_ok && rej == 3, "engine");
    for (void* p : {static_cast<void*>(o_eng), static_cast<void*>(o_ref), static_cast<void*>(big), static_cast<void*>(kc), static_cast<void*>(qb),
             static_cast<void*>(st), static_cast<void*>(mv), static_cast<void*>(thr)})
        HIP(hipFree(p));
    sc.free();
    P.free();
}

void bench() {
    Problem P;
    P.make(37910U, 51);
    const attn::Queries q = queries(P);
    const std::uint32_t qt = (P.S + 127U) / 128U, jc = P.cut / 64U, kt = (P.S + 63U) / 64U;
    const attn::KeyBlock blocks[2] = {P.block(0, P.cut), P.block(P.cut, P.S)};
    Parts pp;
    pp.alloc(P.S, 2);
    const auto time = [&](auto f) {
        f();
        HIP(hipDeviceSynchronize());
        std::vector<double> v;
        for (int r = 0; r < 3; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            f();
            HIP(hipDeviceSynchronize());
            v.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(v.begin(), v.end());
        return v[1];
    };
    const double flops = 4.0 * P.S * static_cast<double>(P.S) * D * H;
    const double td = time([&] { attn::launch_partials(q, blocks, 2, pp.p(), 0, nullptr); });
    std::printf("bench S %u: dense K1 partials %.1f ms (%.1f TF)\n", P.S, td, flops / td / 1e9);
    for (const double dens : {1.0, 0.4, 0.25}) {
        // every list: the same evenly spread tile subset of each block (band-like structure ignored)
        std::vector<std::int32_t> t0v, t1v;
        for (std::uint32_t j = 0; j < jc; ++j) if (static_cast<std::uint32_t>(j * dens) != static_cast<std::uint32_t>((j + 1) * dens) || dens >= 1.0) t0v.push_back(static_cast<std::int32_t>(j));
        for (std::uint32_t j = jc; j < kt; ++j) if (static_cast<std::uint32_t>(j * dens) != static_cast<std::uint32_t>((j + 1) * dens) || dens >= 1.0) t1v.push_back(static_cast<std::int32_t>(j));
        HostLut l0, l1;
        l0.same_for_all(qt, t0v);
        l1.same_for_all(qt, t1v);
        const BlockLut luts[2] = {l0.lut(0), l1.lut(jc)};
        const double ts = time([&] { levers::sparse::launch_sparse_partials(q, blocks, luts, 2, pp.p(), 0, nullptr, nullptr); });
        const double d_act = static_cast<double>(t0v.size() + t1v.size()) / kt;
        std::printf("bench S %u: sparse density %.3f: %.1f ms (%.2fx dense time, %.1f TF effective on processed tiles)\n", P.S, d_act, ts, ts / td,
            flops * d_act / ts / 1e9);
        l0.free(); l1.free();
    }
    pp.free(); P.free();
}
}  // namespace

int main(int argc, char** argv) {
    test_full_lut(1000U, 1);   // cut 384, block 1: 616 keys (last tile 40)
    test_full_lut(1450U, 2);   // cut 640, block 1: 810 keys (last tile 42)
    test_full_lut(2048U, 3);   // cut 1024, full tiles
    test_full_lut(300U, 4);    // cut 128, block 1: 172 keys
    test_empty_half();
    test_subset();
    test_malformed();
    test_engine();
    std::printf("sparse attn gate %s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    if (argc > 1 && std::string(argv[1]) == "--bench") bench();
    return failures ? 1 : 0;
}
