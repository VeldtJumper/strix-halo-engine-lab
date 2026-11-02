// Tests of the fused/batched MSA decode (src/minimax/attn/msa_fused.hpp): every output bit for bit equal to the
// current K1-K4 kernels (strix/hip_minimax_msa.hpp) run once per sequence, and to the host reference on the smaller
// cases. Covers N sequences per call, verify windows, ties in the block scores, the context-parallel shard mode,
// precomputed selections, small persistent grids and graph replay (ticket reset).
//
// STRIX_MMATTN_BENCH=1: microbench, per cell (context x users/window) ABA x3 of A = current kernels (5 launches per
// sequence) vs B = fused (2 launches per call), in stream and graph mode, caches rotated over enough layers to miss
// the 32 MB MALL, plus a per-kernel breakdown. STRIX_MMATTN_BENCH_CELLS=ctx:seqs:W,... restricts the cells.
#include "msa_fused.hpp"
#include "strix/hip_minimax_msa.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace msa = strix::minimax::msa;
namespace mh = strix::minimax::msa::hip;
namespace mf = strix::minimax::msa::fused;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}
void hip_ok(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "HIP error %s: %s\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}
template <class T> T* dalloc(std::size_t n) {
    T* d = nullptr;
    hip_ok(hipMalloc(&d, std::max<std::size_t>(1, n) * sizeof(T)), "malloc");
    hip_ok(hipMemset(d, 0, std::max<std::size_t>(1, n) * sizeof(T)), "memset");
    return d;
}
template <class T> T* dev(const std::vector<T>& h) {
    T* d = dalloc<T>(h.size());
    if (!h.empty()) hip_ok(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "h2d");
    return d;
}
template <class T> std::vector<T> host(const T* d, std::size_t n) {
    std::vector<T> h(n);
    if (n != 0) hip_ok(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost), "d2h");
    return h;
}
template <class T> bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

struct Case {
    unsigned seqs, W, q_heads, kv_heads, idx_heads;
    std::vector<std::uint64_t> last_pos;  // per sequence: position of its last row
    bool ties = false;                    // index keys repeat every 5 blocks: equal block scores
    const char* name = "";
};

// Host-side inputs of one call.
struct Data {
    Case c;
    std::uint64_t cap;
    std::uint32_t max_blocks;
    std::vector<float> q, k, v, iq, ik, w;                    // all rows
    std::vector<msa::RopeRow> rope;
    std::vector<std::uint64_t> pos;
    std::vector<std::vector<std::uint16_t>> kc, vc, ikc;     // per sequence
};

Data make_data(const Case& c, std::mt19937& rng) {
    Data d;
    d.c = c;
    std::uint64_t maxp = 0;
    for (auto p : c.last_pos) maxp = std::max(maxp, p);
    d.cap = maxp + 1 + 37;  // capacity not a multiple of the block
    d.max_blocks = static_cast<std::uint32_t>(msa::block_count(d.cap));
    const unsigned R = c.seqs * c.W;
    std::normal_distribution<float> n(0.0F, 1.0F);
    d.q.resize(std::size_t(R) * c.q_heads * 128);
    d.k.resize(std::size_t(R) * c.kv_heads * 128);
    d.v.resize(d.k.size());
    d.iq.resize(std::size_t(R) * c.idx_heads * 128);
    d.ik.resize(std::size_t(R) * 128);
    for (auto* vec : {&d.q, &d.k, &d.v, &d.iq, &d.ik})
        for (auto& x : *vec) x = n(rng) * 3.0F;
    d.w.resize(4 * 128);
    for (auto& x : d.w) x = n(rng) * 0.3F;
    for (unsigned s = 0; s < c.seqs; ++s) {
        std::vector<std::uint16_t> kc(d.cap * c.kv_heads * 128), vc(kc.size()), ikc(d.cap * 128);
        std::uint64_t st = rng() | 1ULL;  // caches: fast xorshift values in [-2, 2) (normal draws are too slow at 512K)
        for (auto* vec : {&kc, &vc, &ikc})
            for (auto& x : *vec) {
                st ^= st << 13U;
                st ^= st >> 7U;
                st ^= st << 17U;
                x = msa::float_to_bf16((static_cast<float>(st >> 40U) / 16777216.0F - 0.5F) * 4.0F);
            }
        if (c.ties)
            for (std::uint64_t t = 5 * msa::kBlock; t < d.cap; ++t)
                std::memcpy(&ikc[t * 128], &ikc[(t % (5 * msa::kBlock)) * 128], 256);
        d.kc.push_back(std::move(kc));
        d.vc.push_back(std::move(vc));
        d.ikc.push_back(std::move(ikc));
        for (unsigned r = 0; r < c.W; ++r) {
            d.pos.push_back(c.last_pos[s] + 1 - c.W + r);
            d.rope.push_back(msa::rope_row(d.pos.back(), 1e7));
        }
    }
    return d;
}

// Outputs to compare (per call, rows in call order).
struct Out {
    std::vector<std::vector<std::uint16_t>> kc, vc, ikc;
    std::vector<float> scores, o;          // scores: R x idx_heads x max_blocks, only visible blocks compared
    std::vector<std::uint32_t> idx, cnt;   // R x kv x 16 (first cnt compared), R x kv
};

bool same_out(const Data& d, const Out& a, const Out& b, const char* what, bool scores = true, bool sel = true) {
    const Case& c = d.c;
    bool ok = true;
    for (unsigned s = 0; s < c.seqs; ++s) {
        ok &= same(a.kc[s], b.kc[s]);
        ok &= same(a.vc[s], b.vc[s]);
        ok &= same(a.ikc[s], b.ikc[s]);
    }
    check(ok, std::string(c.name) + " " + what + ": caches");
    const unsigned R = c.seqs * c.W;
    if (scores) {
        bool sok = true;
        for (unsigned r = 0; r < R; ++r)
            for (unsigned h = 0; h < c.idx_heads; ++h) {
                const std::size_t base = (std::size_t(r) * c.idx_heads + h) * d.max_blocks;
                const std::size_t nb = msa::block_count(d.pos[r] + 1);
                sok &= std::memcmp(&a.scores[base], &b.scores[base], nb * sizeof(float)) == 0;
            }
        check(sok, std::string(c.name) + " " + what + ": scores");
        ok &= sok;
    }
    if (sel) {
        bool iok = a.cnt == b.cnt;
        for (std::size_t rg = 0; iok && rg < a.cnt.size(); ++rg)
            for (std::uint32_t i = 0; i < a.cnt[rg]; ++i) iok &= a.idx[rg * msa::kTopK + i] == b.idx[rg * msa::kTopK + i];
        check(iok, std::string(c.name) + " " + what + ": selection");
        ok &= iok;
    }
    const bool ook = same(a.o, b.o);
    check(ook, std::string(c.name) + " " + what + ": attention output");
    return ok && ook;
}

// Host reference (the declared order), per sequence.
Out host_ref(const Data& d) {
    const Case& c = d.c;
    Out h;
    h.kc = d.kc;
    h.vc = d.vc;
    h.ikc = d.ikc;
    const unsigned R = c.seqs * c.W;
    const float *qw = d.w.data(), *kw = qw + 128, *iqw = kw + 128, *ikw = iqw + 128;
    std::vector<float> q = d.q, iq = d.iq;
    h.scores.assign(std::size_t(R) * c.idx_heads * d.max_blocks, 0.0F);
    h.idx.assign(std::size_t(R) * c.kv_heads * msa::kTopK, 0);
    h.cnt.assign(std::size_t(R) * c.kv_heads, 0);
    h.o.assign(q.size(), 0.0F);
    for (unsigned r = 0; r < R; ++r) {
        const unsigned s = r / c.W;
        const std::uint64_t p = d.pos[r];
        for (unsigned hd = 0; hd < c.q_heads; ++hd) msa::head_norm_rope(&q[(r * c.q_heads + hd) * 128], qw, d.rope[r]);
        for (unsigned g = 0; g < c.kv_heads; ++g) {
            float x[128];
            std::memcpy(x, &d.k[(r * c.kv_heads + g) * 128], sizeof x);
            msa::head_norm_rope(x, kw, d.rope[r]);
            for (unsigned i = 0; i < 128; ++i) {
                h.kc[s][(p * c.kv_heads + g) * 128 + i] = msa::float_to_bf16(x[i]);
                h.vc[s][(p * c.kv_heads + g) * 128 + i] = msa::float_to_bf16(d.v[(r * c.kv_heads + g) * 128 + i]);
            }
        }
        for (unsigned hd = 0; hd < c.idx_heads; ++hd) msa::head_norm_rope(&iq[(r * c.idx_heads + hd) * 128], iqw, d.rope[r]);
        float x[128];
        std::memcpy(x, &d.ik[r * 128], sizeof x);
        msa::head_norm_rope(x, ikw, d.rope[r]);
        for (unsigned i = 0; i < 128; ++i) h.ikc[s][p * 128 + i] = msa::float_to_bf16(x[i]);
    }
    for (unsigned r = 0; r < R; ++r) {
        const unsigned s = r / c.W;
        const std::uint64_t count = d.pos[r] + 1;
        std::vector<float> ikf(count * 128), kg(count * 128), vg(count * 128);
        for (std::size_t i = 0; i < ikf.size(); ++i) ikf[i] = msa::bf16_to_float(h.ikc[s][i]);
        for (unsigned hd = 0; hd < c.idx_heads; ++hd) {
            std::span<float> sc(&h.scores[(r * c.idx_heads + hd) * d.max_blocks], msa::block_count(count));
            msa::index_block_scores(&iq[(r * c.idx_heads + hd) * 128], ikf.data(), count, sc);
            if (hd >= c.kv_heads) continue;
            const auto sel = msa::select_blocks(sc);
            std::copy(sel.begin(), sel.end(), &h.idx[(r * c.kv_heads + hd) * msa::kTopK]);
            h.cnt[r * c.kv_heads + hd] = static_cast<std::uint32_t>(sel.size());
            for (std::uint64_t t = 0; t < count; ++t)
                for (unsigned i = 0; i < 128; ++i) {
                    kg[t * 128 + i] = msa::bf16_to_float(h.kc[s][(t * c.kv_heads + hd) * 128 + i]);
                    vg[t * 128 + i] = msa::bf16_to_float(h.vc[s][(t * c.kv_heads + hd) * 128 + i]);
                }
            msa::sparse_attention(&q[(r * c.q_heads + hd * msa::kGroup) * 128], msa::kGroup, kg.data(), vg.data(), count, sel,
                                  &h.o[(r * c.q_heads + hd * msa::kGroup) * 128]);
        }
    }
    return h;
}

// Device state of one call: shared inputs, per-sequence caches, both arms' scratch.
struct Dev {
    unsigned R = 0;
    float *q = nullptr, *k = nullptr, *v = nullptr, *iq = nullptr, *ik = nullptr, *w = nullptr;
    float *q_old = nullptr, *iq_old = nullptr;  // the current K1 rewrites q / iq in place: its own copies
    msa::RopeRow* rope = nullptr;
    std::uint64_t* pos = nullptr;
    std::vector<std::uint16_t*> kc, vc, ikc;
    std::uint16_t **kc_tab = nullptr, **vc_tab = nullptr, **ikc_tab = nullptr;
    float *scores = nullptr, *partial = nullptr, *o = nullptr;
    std::uint32_t *idx = nullptr, *cnt = nullptr, *ticket = nullptr;
};

Dev upload(const Data& d) {
    const Case& c = d.c;
    Dev x;
    x.R = c.seqs * c.W;
    x.q = dev(d.q);
    x.k = dev(d.k);
    x.v = dev(d.v);
    x.iq = dev(d.iq);
    x.ik = dev(d.ik);
    x.w = dev(d.w);
    x.q_old = dev(d.q);
    x.iq_old = dev(d.iq);
    x.rope = dev(d.rope);
    x.pos = dev(d.pos);
    for (unsigned s = 0; s < c.seqs; ++s) {
        x.kc.push_back(dev(d.kc[s]));
        x.vc.push_back(dev(d.vc[s]));
        x.ikc.push_back(dev(d.ikc[s]));
    }
    x.kc_tab = dev(x.kc);
    x.vc_tab = dev(x.vc);
    x.ikc_tab = dev(x.ikc);
    x.scores = dalloc<float>(std::size_t(x.R) * c.idx_heads * d.max_blocks);
    x.partial = dalloc<float>(std::size_t(x.R) * c.q_heads * msa::kMaxChunks * 130);
    x.o = dalloc<float>(d.q.size());
    x.idx = dalloc<std::uint32_t>(std::size_t(x.R) * c.kv_heads * msa::kTopK);
    x.cnt = dalloc<std::uint32_t>(std::size_t(x.R) * c.kv_heads);
    x.ticket = dalloc<std::uint32_t>(std::size_t(x.R) * c.kv_heads * 4);
    return x;
}
void reset_caches(const Data& d, Dev& x) {
    for (unsigned s = 0; s < d.c.seqs; ++s) {
        hip_ok(hipMemcpy(x.kc[s], d.kc[s].data(), d.kc[s].size() * 2, hipMemcpyHostToDevice), "h2d");
        hip_ok(hipMemcpy(x.vc[s], d.vc[s].data(), d.vc[s].size() * 2, hipMemcpyHostToDevice), "h2d");
        hip_ok(hipMemcpy(x.ikc[s], d.ikc[s].data(), d.ikc[s].size() * 2, hipMemcpyHostToDevice), "h2d");
    }
    hip_ok(hipMemcpy(x.q_old, d.q.data(), d.q.size() * 4, hipMemcpyHostToDevice), "h2d");
    hip_ok(hipMemcpy(x.iq_old, d.iq.data(), d.iq.size() * 4, hipMemcpyHostToDevice), "h2d");
    hip_ok(hipMemset(x.scores, 0, std::size_t(x.R) * d.c.idx_heads * d.max_blocks * 4), "memset");
    hip_ok(hipMemset(x.o, 0, d.q.size() * 4), "memset");
    hip_ok(hipMemset(x.idx, 0, std::size_t(x.R) * d.c.kv_heads * msa::kTopK * 4), "memset");
    hip_ok(hipMemset(x.cnt, 0, std::size_t(x.R) * d.c.kv_heads * 4), "memset");
}
void release(Dev& x) {
    std::vector<void*> p = {x.q, x.k, x.v, x.iq, x.ik, x.w, x.q_old, x.iq_old, x.rope, x.pos, x.kc_tab, x.vc_tab,
                            x.ikc_tab, x.scores, x.partial, x.o, x.idx, x.cnt, x.ticket};
    for (unsigned s = 0; s < x.kc.size(); ++s) {
        p.push_back(x.kc[s]);
        p.push_back(x.vc[s]);
        p.push_back(x.ikc[s]);
    }
    for (void* ptr : p) (void)hipFree(ptr);
}
Out read_out(const Data& d, const Dev& x) {
    const Case& c = d.c;
    Out o;
    for (unsigned s = 0; s < c.seqs; ++s) {
        o.kc.push_back(host(x.kc[s], d.kc[s].size()));
        o.vc.push_back(host(x.vc[s], d.vc[s].size()));
        o.ikc.push_back(host(x.ikc[s], d.ikc[s].size()));
    }
    o.scores = host(x.scores, std::size_t(x.R) * c.idx_heads * d.max_blocks);
    o.o = host(x.o, d.q.size());
    o.idx = host(x.idx, std::size_t(x.R) * c.kv_heads * msa::kTopK);
    o.cnt = host(x.cnt, std::size_t(x.R) * c.kv_heads);
    return o;
}

// Arm A: the current kernels, one call per sequence (rows = W), on the same buffers.
void enqueue_old(const Data& d, const Dev& x, hipStream_t st) {
    const Case& c = d.c;
    const unsigned W = c.W;
    for (unsigned s = 0; s < c.seqs; ++s) {
        const unsigned r0 = s * W;
        float* q = x.q_old + std::size_t(r0) * c.q_heads * 128;
        float* iq = x.iq_old + std::size_t(r0) * c.idx_heads * 128;
        const std::uint64_t* pos = x.pos + r0;
        float* sc = x.scores + std::size_t(r0) * c.idx_heads * d.max_blocks;
        std::uint32_t* idx = x.idx + std::size_t(r0) * c.kv_heads * msa::kTopK;
        std::uint32_t* cnt = x.cnt + std::size_t(r0) * c.kv_heads;
        mh::enqueue_prep({W, c.q_heads, c.kv_heads, c.idx_heads, q, x.k + std::size_t(r0) * c.kv_heads * 128,
                          x.v + std::size_t(r0) * c.kv_heads * 128, iq, x.ik + std::size_t(r0) * 128, x.w, x.w + 128,
                          x.w + 256, x.w + 384, x.rope + r0, pos, x.kc[s], x.vc[s], x.ikc[s]}, st);
        mh::enqueue_scores({W, c.idx_heads, d.max_blocks, iq, x.ikc[s], pos, sc}, st);
        mh::enqueue_select({W, c.kv_heads, d.max_blocks, pos, sc, idx, cnt}, st);  // index head g selects for KV head g
        mh::enqueue_attention({W, c.kv_heads, q, x.kc[s], x.vc[s], pos, idx, cnt,
                               x.partial + std::size_t(r0) * c.q_heads * msa::kMaxChunks * 130,
                               x.o + std::size_t(r0) * c.q_heads * 128}, st);
    }
}
mf::DecodeView new_view(const Data& d, const Dev& x) {
    const Case& c = d.c;
    mf::DecodeView v{};
    v.seqs = c.seqs;
    v.rows_per_seq = c.W;
    v.q_heads = c.q_heads;
    v.kv_heads = c.kv_heads;
    v.idx_heads = c.idx_heads;
    v.max_blocks = d.max_blocks;
    v.q = x.q;
    v.k = x.k;
    v.v = x.v;
    v.iq = x.iq;
    v.ik = x.ik;
    v.q_w = x.w;
    v.k_w = x.w + 128;
    v.iq_w = x.w + 256;
    v.ik_w = x.w + 384;
    v.rope = x.rope;
    v.pos = x.pos;
    v.k_cache = x.kc_tab;
    v.v_cache = x.vc_tab;
    v.ik_cache = x.ikc_tab;
    v.scores = x.scores;
    v.idx = x.idx;
    v.idx_count = x.cnt;
    v.partial = x.partial;
    v.ticket = x.ticket;
    v.o = x.o;
    return v;
}

void run_case(const Case& c, std::mt19937& rng) {
    const int fail0 = g_fail;
    const Data d = make_data(c, rng);
    Dev x = upload(d);
    // A: current kernels.
    reset_caches(d, x);
    enqueue_old(d, x, nullptr);
    hip_ok(hipDeviceSynchronize(), "sync A");
    const Out a = read_out(d, x);
    // B: fused, default grid.
    reset_caches(d, x);
    mf::enqueue_decode(new_view(d, x), nullptr);
    hip_ok(hipDeviceSynchronize(), "sync B");
    const Out b = read_out(d, x);
    same_out(d, a, b, "fused vs current");
    if (d.cap <= 70000) same_out(d, host_ref(d), b, "fused vs host");
    // Attend blocks of 8 and 4 query heads.
    for (unsigned pvf : {0U, 1U}) {  // version 3 with the predicated-only PV path
        if (pvf == 1U) break;
        reset_caches(d, x);
        mf::DecodeView v = new_view(d, x);
        v.pv_full_tiles = pvf;
        mf::enqueue_decode(v, nullptr);
        hip_ok(hipDeviceSynchronize(), "sync pvf0");
        same_out(d, a, read_out(d, x), "fused v3 pv_full_tiles 0");
    }
    for (unsigned hpw : {1U, 2U, 4U}) {
        reset_caches(d, x);
        mf::DecodeView v = new_view(d, x);
        v.attend_version = 4;
        v.heads_per_wave = hpw;
        mf::enqueue_decode(v, nullptr);
        hip_ok(hipDeviceSynchronize(), "sync v4");
        same_out(d, a, read_out(d, x), ("fused v4 heads/wave " + std::to_string(hpw)).c_str());
        if (std::all_of(d.pos.begin(), d.pos.end(), [](std::uint64_t p) { return p < msa::kTopK * msa::kBlock; })) {
            reset_caches(d, x);
            v.dense = true;
            mf::enqueue_decode(v, nullptr);
            hip_ok(hipDeviceSynchronize(), "sync v4 dense");
            same_out(d, a, read_out(d, x), ("dense v4 heads/wave " + std::to_string(hpw)).c_str(), false, false);
        }
    }
    for (unsigned ver : {1U, 2U, 3U})
        for (unsigned hw : {16U, 8U, 4U}) {
            reset_caches(d, x);
            mf::DecodeView v = new_view(d, x);
            v.heads_per_block = hw;
            v.attend_version = ver;
            mf::enqueue_decode(v, nullptr);
            hip_ok(hipDeviceSynchronize(), "sync hw");
            same_out(d, a, read_out(d, x), ("fused v" + std::to_string(ver) + " heads/block " + std::to_string(hw)).c_str());
            if (ver == 1) continue;  // version 1 has static LDS
            reset_caches(d, x);
            v.lds_diet = true;  // actual dynamic LDS instead of the 43008-byte floor: allocation only
            mf::enqueue_decode(v, nullptr);
            hip_ok(hipDeviceSynchronize(), "sync diet");
            same_out(d, a, read_out(d, x), ("fused v" + std::to_string(ver) + " lds_diet heads/block " + std::to_string(hw)).c_str());
        }
    // Dense single launch (every row below 2048 positions): caches and output (no scores / selection written).
    if (std::all_of(d.pos.begin(), d.pos.end(), [](std::uint64_t p) { return p < msa::kTopK * msa::kBlock; }))
        for (const auto& [ver, hw] : {std::pair{1U, 16U}, std::pair{1U, 8U}, std::pair{1U, 4U}, std::pair{2U, 16U},
                                      std::pair{2U, 8U}, std::pair{2U, 4U}, std::pair{3U, 16U}, std::pair{3U, 8U},
                                      std::pair{3U, 4U}}) {
            reset_caches(d, x);
            mf::DecodeView v = new_view(d, x);
            v.heads_per_block = hw;
            v.attend_version = ver;
            v.dense = true;
            mf::enqueue_decode(v, nullptr);
            hip_ok(hipDeviceSynchronize(), "sync dense");
            same_out(d, a, read_out(d, x), ("dense v" + std::to_string(ver) + " heads/block " + std::to_string(hw)).c_str(), false, false);
        }
    // B with a tiny persistent grid (every block loops over many items, prefetch across sequences).
    {
        const unsigned saved = mf::index_grid_cap();
        mf::set_index_grid_cap(3);
        reset_caches(d, x);
        mf::enqueue_decode(new_view(d, x), nullptr);
        hip_ok(hipDeviceSynchronize(), "sync B3");
        same_out(d, a, read_out(d, x), "fused grid 3");
        mf::set_index_grid_cap(saved);
    }
    // Precomputed selection (select = false): idx/cnt from A as inputs.
    if (c.idx_heads == c.kv_heads) {
        reset_caches(d, x);
        mf::enqueue_index(new_view(d, x), nullptr);
        hip_ok(hipMemcpy(x.idx, a.idx.data(), a.idx.size() * 4, hipMemcpyHostToDevice), "h2d");
        hip_ok(hipMemcpy(x.cnt, a.cnt.data(), a.cnt.size() * 4, hipMemcpyHostToDevice), "h2d");
        mf::enqueue_attend(new_view(d, x), false, nullptr);
        hip_ok(hipDeviceSynchronize(), "sync sel0");
        same_out(d, a, read_out(d, x), "fused select=false");
    }
    // Graph capture, replayed 3 times from the same initial caches (tickets must come back to zero).
    {
        hipStream_t st;
        hip_ok(hipStreamCreateWithFlags(&st, hipStreamNonBlocking), "stream");
        hipGraph_t g;
        hipGraphExec_t ge;
        hip_ok(hipStreamBeginCapture(st, hipStreamCaptureModeThreadLocal), "capture");
        mf::enqueue_decode(new_view(d, x), st);
        hip_ok(hipStreamEndCapture(st, &g), "end capture");
        hip_ok(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0), "instantiate");
        bool ok = true;
        for (int rep = 0; rep < 3; ++rep) {
            reset_caches(d, x);
            hip_ok(hipGraphLaunch(ge, st), "graph launch");
            hip_ok(hipStreamSynchronize(st), "graph sync");
            ok &= same_out(d, a, read_out(d, x), "graph replay");
        }
        const auto t = host(x.ticket, std::size_t(x.R) * c.kv_heads);
        check(std::all_of(t.begin(), t.end(), [](std::uint32_t y) { return y == 0; }), std::string(c.name) + " tickets reset");
        (void)ok;
        hip_ok(hipGraphExecDestroy(ge), "destroy");
        hip_ok(hipGraphDestroy(g), "destroy");
        hip_ok(hipStreamDestroy(st), "stream destroy");
    }
    // Context-parallel shards: the fused index launch with shard (rank, 2) equals the current scores kernel's shard.
    if (c.seqs == 1) {
        for (std::uint32_t sh = 0; sh < 2; ++sh) {
            const std::size_t n = std::size_t(x.R) * c.idx_heads * d.max_blocks;
            reset_caches(d, x);
            // current: prep then scores (shard)
            mh::enqueue_prep({c.W, c.q_heads, c.kv_heads, c.idx_heads, x.q_old, x.k, x.v, x.iq_old, x.ik, x.w, x.w + 128,
                              x.w + 256, x.w + 384, x.rope, x.pos, x.kc[0], x.vc[0], x.ikc[0]}, nullptr);
            mh::ScoreView sv{c.W, c.idx_heads, d.max_blocks, x.iq_old, x.ikc[0], x.pos, x.scores};
            sv.shard_rank = sh;
            sv.shard_count = 2;
            mh::enqueue_scores(sv, nullptr);
            hip_ok(hipDeviceSynchronize(), "sync shard A");
            const auto sa = host(x.scores, n);
            const auto ka = host(x.kc[0], d.kc[0].size()), va = host(x.vc[0], d.vc[0].size()),
                       ia = host(x.ikc[0], d.ikc[0].size());
            reset_caches(d, x);
            mf::DecodeView v = new_view(d, x);
            v.shard_rank = sh;
            v.shard_count = 2;
            mf::enqueue_index(v, nullptr);
            hip_ok(hipDeviceSynchronize(), "sync shard B");
            const auto sb = host(x.scores, n);
            bool ok = same(ka, host(x.kc[0], d.kc[0].size())) && same(va, host(x.vc[0], d.vc[0].size())) &&
                      same(ia, host(x.ikc[0], d.ikc[0].size()));
            for (unsigned r = 0; r < x.R; ++r) {
                const std::uint64_t nb = msa::block_count(d.pos[r] + 1);
                const std::uint64_t local = nb > sh ? (nb - sh + 1) / 2 : 0;
                for (unsigned h = 0; h < c.idx_heads; ++h)
                    ok &= std::memcmp(&sa[(r * c.idx_heads + h) * d.max_blocks], &sb[(r * c.idx_heads + h) * d.max_blocks],
                                      local * 4) == 0;
            }
            check(ok, std::string(c.name) + " shard " + std::to_string(sh));
        }
    }
    std::printf("case %-14s seqs %u W %u heads %u/%u/%u last_pos %llu..: %s\n", c.name, c.seqs, c.W, c.q_heads, c.kv_heads,
                c.idx_heads, static_cast<unsigned long long>(c.last_pos.back()), g_fail == fail0 ? "bitwise equal" : "MISMATCH");
    release(x);
}

// Prefill-shaped windows: compare v3 directly with v4/two heads per wave.
void prefill_attend4_cases(std::mt19937& rng) {
    for (unsigned m = 1; m <= 16; ++m) {
        for (std::uint64_t pos0 : {127ULL, 128ULL, 129ULL, 2040ULL, 2047ULL, 2048ULL}) {
            const Data d = make_data({1, m, 32, 2, 2, {pos0 + m - 1}, false, "prefill-attend4"}, rng);
            Dev x = upload(d);
            auto v = new_view(d, x);
            v.dense = pos0 + m - 1 < msa::kTopK * msa::kBlock;
            reset_caches(d, x);
            mf::enqueue_decode(v, nullptr);
            hip_ok(hipDeviceSynchronize(), "prefill v3");
            const Out a = read_out(d, x);
            reset_caches(d, x);
            v.attend_version = 4;
            v.heads_per_wave = 2;
            mf::enqueue_decode(v, nullptr);
            hip_ok(hipDeviceSynchronize(), "prefill v4");
            same_out(d, a, read_out(d, x), "prefill v3 vs v4/hpw2", !v.dense, !v.dense);
            release(x);
        }
    }
}

// ---------------------------------------------------------------- prefill (one sequence, T rows from position 0)

double median(std::vector<double> v);
bool g_prefill_dense = true;  // prefill windows below 2048 positions use the dense single launch

// A = the G4 harness schedule (current K1 over all T rows, then K2-K4 per 8-row window); B = fused launches per
// window of up to kMaxRows rows. Same outputs bit for bit; bench = true times both (ABA x3) instead of checking.
struct Prefill {
    std::uint64_t T;
    std::uint32_t max_blocks;
    std::vector<float> q, k, v, iq, ik, w;
    std::vector<msa::RopeRow> rope;
    std::vector<std::uint64_t> pos;
    float *dq, *dq_old, *dk, *dv, *diq, *diq_old, *dik, *dw, *scores, *partial, *o;
    msa::RopeRow* drope;
    std::uint64_t* dpos;
    std::uint16_t *kc, *vc, *ikc, **kc_tab, **vc_tab, **ikc_tab;
    std::uint32_t *idx, *cnt, *ticket;
};
Prefill make_prefill(std::uint64_t T, std::mt19937& rng) {
    Prefill p{};
    p.T = T;
    p.max_blocks = static_cast<std::uint32_t>(msa::block_count(T));
    std::normal_distribution<float> n(0.0F, 1.0F);
    p.q.resize(T * 32 * 128);
    p.k.resize(T * 2 * 128);
    p.v.resize(p.k.size());
    p.iq.resize(T * 2 * 128);
    p.ik.resize(T * 128);
    p.w.resize(512);
    for (auto* vec : {&p.q, &p.k, &p.v, &p.iq, &p.ik, &p.w})
        for (auto& x : *vec) x = n(rng) * 2.0F;
    for (std::uint64_t t = 0; t < T; ++t) {
        p.pos.push_back(t);
        p.rope.push_back(msa::rope_row(t, 1e7));
    }
    p.dq = dev(p.q);
    p.dq_old = dev(p.q);
    p.dk = dev(p.k);
    p.dv = dev(p.v);
    p.diq = dev(p.iq);
    p.diq_old = dev(p.iq);
    p.dik = dev(p.ik);
    p.dw = dev(p.w);
    p.drope = dev(p.rope);
    p.dpos = dev(p.pos);
    p.kc = dalloc<std::uint16_t>(T * 2 * 128);
    p.vc = dalloc<std::uint16_t>(T * 2 * 128);
    p.ikc = dalloc<std::uint16_t>(T * 128);
    p.kc_tab = dev(std::vector<std::uint16_t*>{p.kc});
    p.vc_tab = dev(std::vector<std::uint16_t*>{p.vc});
    p.ikc_tab = dev(std::vector<std::uint16_t*>{p.ikc});
    p.scores = dalloc<float>(std::size_t(mf::kMaxRows) * 2 * p.max_blocks);
    p.partial = dalloc<float>(std::size_t(mf::kMaxRows) * 32 * msa::kMaxChunks * 130);
    p.o = dalloc<float>(p.q.size());
    p.idx = dalloc<std::uint32_t>(std::size_t(mf::kMaxRows) * 2 * msa::kTopK);
    p.cnt = dalloc<std::uint32_t>(std::size_t(mf::kMaxRows) * 2);
    p.ticket = dalloc<std::uint32_t>(std::size_t(mf::kMaxRows) * 2 * 4);
    return p;
}
void release_prefill(Prefill& p) {
    for (void* ptr : {static_cast<void*>(p.dq), static_cast<void*>(p.dq_old), static_cast<void*>(p.dk), static_cast<void*>(p.dv),
                      static_cast<void*>(p.diq), static_cast<void*>(p.diq_old), static_cast<void*>(p.dik), static_cast<void*>(p.dw),
                      static_cast<void*>(p.drope), static_cast<void*>(p.dpos), static_cast<void*>(p.kc), static_cast<void*>(p.vc),
                      static_cast<void*>(p.ikc), static_cast<void*>(p.kc_tab), static_cast<void*>(p.vc_tab),
                      static_cast<void*>(p.ikc_tab), static_cast<void*>(p.scores), static_cast<void*>(p.partial),
                      static_cast<void*>(p.o), static_cast<void*>(p.idx), static_cast<void*>(p.cnt), static_cast<void*>(p.ticket)})
        (void)hipFree(ptr);
}
void prefill_old(Prefill& p, hipStream_t st) {
    const auto T = static_cast<std::uint32_t>(p.T);
    hip_ok(hipMemcpyAsync(p.dq_old, p.dq, p.q.size() * 4, hipMemcpyDeviceToDevice, st), "d2d");
    hip_ok(hipMemcpyAsync(p.diq_old, p.diq, p.iq.size() * 4, hipMemcpyDeviceToDevice, st), "d2d");
    mh::enqueue_prep({T, 32, 2, 2, p.dq_old, p.dk, p.dv, p.diq_old, p.dik, p.dw, p.dw + 128, p.dw + 256, p.dw + 384, p.drope,
                      p.dpos, p.kc, p.vc, p.ikc}, st);
    for (std::uint32_t r0 = 0; r0 < T; r0 += 8) {
        const std::uint32_t n = std::min<std::uint32_t>(8, T - r0);
        mh::enqueue_scores({n, 2, p.max_blocks, p.diq_old + std::size_t(r0) * 256, p.ikc, p.dpos + r0, p.scores}, st);
        mh::enqueue_select({n, 2, p.max_blocks, p.dpos + r0, p.scores, p.idx, p.cnt}, st);
        mh::enqueue_attention({n, 2, p.dq_old + std::size_t(r0) * 32 * 128, p.kc, p.vc, p.dpos + r0, p.idx, p.cnt, p.partial,
                               p.o + std::size_t(r0) * 32 * 128}, st);
    }
}
mf::DecodeView prefill_view(Prefill& p, std::uint32_t r0, std::uint32_t n, unsigned ver = 0, unsigned hw = 0, unsigned pvf = 1) {
    mf::DecodeView v{};
    v.seqs = 1;
    v.rows_per_seq = n;
    v.q_heads = 32;
    v.kv_heads = 2;
    v.idx_heads = 2;
    v.max_blocks = p.max_blocks;
    v.q = p.dq + std::size_t(r0) * 32 * 128;
    v.k = p.dk + std::size_t(r0) * 256;
    v.v = p.dv + std::size_t(r0) * 256;
    v.iq = p.diq + std::size_t(r0) * 256;
    v.ik = p.dik + std::size_t(r0) * 128;
    v.q_w = p.dw;
    v.k_w = p.dw + 128;
    v.iq_w = p.dw + 256;
    v.ik_w = p.dw + 384;
    v.rope = p.drope + r0;
    v.pos = p.dpos + r0;
    v.k_cache = p.kc_tab;
    v.v_cache = p.vc_tab;
    v.ik_cache = p.ikc_tab;
    v.scores = p.scores;
    v.idx = p.idx;
    v.idx_count = p.cnt;
    v.partial = p.partial;
    v.ticket = p.ticket;
    v.o = p.o + std::size_t(r0) * 32 * 128;
    v.dense = g_prefill_dense && p.pos[r0 + n - 1] < msa::kTopK * msa::kBlock;
    v.attend_version = ver;
    if (ver == 4) v.heads_per_wave = hw;
    else v.heads_per_block = hw;
    v.pv_full_tiles = pvf;
    return v;
}
void prefill_fused(Prefill& p, unsigned wrows, hipStream_t st, unsigned ver = 0, unsigned hw = 0, unsigned pvf = 1) {
    const auto T = static_cast<std::uint32_t>(p.T);
    for (std::uint32_t r0 = 0; r0 < T; r0 += wrows) {
        const std::uint32_t n = std::min<std::uint32_t>(wrows, T - r0);
        mf::enqueue_decode(prefill_view(p, r0, n, ver, hw, pvf), st);
    }
}
// Count physical blocks separately in each workspace-local R16 window and KV
// group. Dense IDs/counts are implicit; their untouched sentinels are not IDs.
void report_r16_reuse(std::uint64_t pos0, unsigned n, unsigned cap, unsigned pvf,
                      const std::vector<std::uint32_t>& ids, const std::vector<std::uint32_t>& counts) {
    std::uint64_t items[2]{}, distinct[2]{}, logical_bytes[2]{}, visible_bytes[2]{};
    std::array<std::uint64_t, 17> histogram{};
    std::uint64_t common = 0;
    for (unsigned first = 0; first < n; first += 16) {
        const unsigned nr = std::min(16U, n - first);
        const bool dense = pos0 + first + nr - 1 < msa::kTopK * msa::kBlock;
        const unsigned blocks = static_cast<unsigned>((pos0 + first + nr + 127) / 128);
        for (unsigned g = 0; g < 2; ++g) {
            std::vector<unsigned> membership(blocks, 0U), extent(blocks, 0U);
            for (unsigned r = first; r < first + nr; ++r) {
                const std::uint64_t count = pos0 + r + 1;
                const unsigned nb = static_cast<unsigned>((count + 127) / 128);
                const unsigned cnt = dense ? nb : counts[r * 2 + g];
                items[g] += cnt;
                for (unsigned c = 0; c < cnt; ++c) {
                    const unsigned b = dense ? c : ids[(std::size_t{r} * 2 + g) * msa::kTopK + c];
                    check(b < blocks, "reuse counter selected block in range");
                    if (b >= blocks) continue;
                    const unsigned nk = static_cast<unsigned>(std::min<std::uint64_t>(128, count - std::uint64_t{b} * 128));
                    ++membership[b];
                    extent[b] = std::max(extent[b], nk);
                    logical_bytes[g] += std::uint64_t{nk} * 512;  // BF16 K + V, 128 dimensions
                }
            }
            for (unsigned b = 0; b < blocks; ++b) {
                if (membership[b] == 0) continue;
                ++distinct[g];
                ++histogram[membership[b]];
                if (membership[b] == nr) ++common;
                visible_bytes[g] += std::uint64_t{extent[b]} * 512;
            }
        }
    }
    for (unsigned g = 0; g < 2; ++g)
        std::printf("R16 batch pos0=%llu n=%u scratch=%u pv_full=%u g=%u I=%llu D16=%llu reuse=%.3f logical_KV_bytes=%llu visible_KV_bytes=%llu\n",
                    static_cast<unsigned long long>(pos0), n, cap, pvf, g,
                    static_cast<unsigned long long>(items[g]), static_cast<unsigned long long>(distinct[g]),
                    distinct[g] == 0 ? 0.0 : static_cast<double>(items[g]) / static_cast<double>(distinct[g]),
                    static_cast<unsigned long long>(logical_bytes[g]), static_cast<unsigned long long>(visible_bytes[g]));
    std::printf("R16 batch common_blocks=%llu membership_histogram=", static_cast<unsigned long long>(common));
    for (unsigned i = 1; i <= 16; ++i)
        std::printf("%s%u:%llu", i == 1 ? "" : ",", i, static_cast<unsigned long long>(histogram[i]));
    std::printf("\n");
}

// Random finite initial cache bits include stale future slots, restored before every arm.
void prefill_batch_case(std::uint32_t pos0, std::uint32_t n, std::mt19937& rng, bool ties = false, bool quick = false) {
    Prefill p = make_prefill(std::uint64_t{pos0} + n, rng);
    if (ties) hip_ok(hipMemset(p.diq, 0, p.iq.size() * sizeof(float)), "tied index queries");
    const std::size_t nkv = p.T * 256, nik = p.T * 128;
    std::vector<std::uint16_t> initial_k(nkv), initial_v(nkv), initial_i(nik);
    for (auto* vec : {&initial_k, &initial_v, &initial_i})
        for (auto& x : *vec) x = static_cast<std::uint16_t>(0x3e00U + (rng() & 0x3ffU));
    const unsigned windows = (n + 15) / 16;
    const unsigned max_scratch = static_cast<unsigned>((512ULL * 1024 * 1024) / mf::prefill_window_bytes(p.max_blocks));
    const unsigned scratch_windows = std::min(windows, max_scratch);
    const std::size_t rows = std::size_t{scratch_windows} * 16;
    float* scores = dalloc<float>(rows * 2 * p.max_blocks);
    float* partial = dalloc<float>(rows * 32 * msa::kMaxChunks * 130);
    auto* idx = dalloc<std::uint32_t>(rows * 2 * msa::kTopK);
    auto* cnt = dalloc<std::uint32_t>(rows * 2);
    auto* ticket = dalloc<std::uint32_t>(rows * 2 * 4);
    hipEvent_t begin, end;
    hip_ok(hipEventCreate(&begin), "batch begin");
    hip_ok(hipEventCreate(&end), "batch end");
    const auto reset = [&] {
        hip_ok(hipMemcpy(p.kc, initial_k.data(), nkv * 2, hipMemcpyHostToDevice), "batch reset K");
        hip_ok(hipMemcpy(p.vc, initial_v.data(), nkv * 2, hipMemcpyHostToDevice), "batch reset V");
        hip_ok(hipMemcpy(p.ikc, initial_i.data(), nik * 2, hipMemcpyHostToDevice), "batch reset IK");
        hip_ok(hipMemset(p.o, 0xa5, p.q.size() * sizeof(float)), "batch reset O");
        hip_ok(hipMemset(scores, 0xa5, rows * 2 * p.max_blocks * sizeof(float)), "batch reset scores");
        hip_ok(hipMemset(idx, 0xa5, rows * 2 * msa::kTopK * sizeof(std::uint32_t)), "batch reset ids");
        hip_ok(hipMemset(cnt, 0xa5, rows * 2 * sizeof(std::uint32_t)), "batch reset counts");
        hip_ok(hipMemset(ticket, 0, rows * 2 * 4 * sizeof(std::uint32_t)), "batch reset tickets");
        hip_ok(hipMemset(partial, 0xa5, rows * 32 * msa::kMaxChunks * 130 * sizeof(float)), "batch reset partials");
    };
    const auto elapsed = [&] {
        hip_ok(hipEventRecord(end, nullptr), "batch end record");
        hip_ok(hipEventSynchronize(end), "batch end sync");
        float ms = 0;
        hip_ok(hipEventElapsedTime(&ms, begin, end), "batch elapsed");
        return ms;
    };
    if (!quick) for (unsigned ver : {3U, 4U}) {
        reset();
        hip_ok(hipEventRecord(begin, nullptr), "old begin record");
        for (unsigned r0 = 0; r0 < n; r0 += 16) {
            const unsigned m = std::min(16U, n - r0);
            mf::enqueue_decode(prefill_view(p, pos0 + r0, m, ver, ver == 4 ? 2U : 0U), nullptr);
        }
        const float old_ms = elapsed();
        const auto oa = host(p.o, p.q.size());
        const auto ka = host(p.kc, nkv), va = host(p.vc, nkv), ia = host(p.ikc, nik);
        // One-window scratch forces reuse; the bounded full allocation exercises simultaneous windows.
        for (unsigned cap : {1U, scratch_windows}) {
            std::vector<float> score_ref, partial_ref, batch_output_ref;
            std::vector<std::uint32_t> idx_ref, count_ref;
            for (unsigned query_rows : {0U, 16U, 32U, 64U}) {
                for (unsigned rework : {0U, 1U, 2U, 3U, 4U, 5U, 6U}) {
                    reset();
                    auto v = prefill_view(p, pos0, std::min(n, 16U), ver, ver == 4 ? 2U : 0U);
                    v.scores = scores;
                    v.partial = partial;
                    v.idx = idx;
                    v.idx_count = cnt;
                    v.ticket = ticket;
                    hip_ok(hipEventRecord(begin, nullptr), "new begin record");
                    mf::enqueue_prefill({v, n, pos0, cap, 0, query_rows != 0, query_rows == 0 ? 32U : query_rows, rework}, nullptr);
                    const float new_ms = elapsed();
                    const auto tickets = host(ticket, rows * 2 * 4);
                    const auto score_out = host(scores, rows * 2 * p.max_blocks);
                    const auto idx_out = host(idx, rows * 2 * msa::kTopK), count_out = host(cnt, rows * 2);
                    const auto partial_out = host(partial, rows * 32 * msa::kMaxChunks * 130);
                    if (query_rows == 0 && rework == 0) {
                        score_ref = score_out; idx_ref = idx_out; count_ref = count_out; partial_ref = partial_out;
                        batch_output_ref = host(p.o, p.q.size());
                    }
                    const bool ok = same(batch_output_ref, host(p.o, p.q.size())) && same(partial_ref, partial_out) && same(score_ref, score_out) && same(idx_ref, idx_out) && same(count_ref, count_out) &&
                                    same(oa, host(p.o, p.q.size())) && same(ka, host(p.kc, nkv)) &&
                                    same(va, host(p.vc, nkv)) && same(ia, host(p.ikc, nik)) &&
                                    std::all_of(tickets.begin(), tickets.end(), [](std::uint32_t x) { return x == 0; });
                    check(ok, "batched prefill pos0=" + std::to_string(pos0) + " n=" + std::to_string(n) +
                              " v=" + std::to_string(ver) + " scratch=" + std::to_string(cap) +
                              " rework=" + std::to_string(rework) + " idx_rows=" + std::to_string(query_rows) + " ties=" + std::to_string(ties));
                    std::printf("prefill chunk pos0=%u n=%u v%u scratch=%u rework=%u old %.3f ms new %.3f ms: %s\n",
                                pos0, n, ver, cap, rework, old_ms, new_ms, ok ? "bitwise equal" : "MISMATCH");
                    if (ties && cap == 1 && ver == 4 && query_rows == 0 && rework != 0) {
                        hipStream_t st;
                        hipGraph_t graph;
                        hipGraphExec_t exec;
                        hip_ok(hipStreamCreate(&st), "rework graph stream");
                        hip_ok(hipStreamBeginCapture(st, hipStreamCaptureModeGlobal), "rework capture");
                        mf::enqueue_prefill({v, n, pos0, cap, 0, false, 32, rework}, st);
                        hip_ok(hipStreamEndCapture(st, &graph), "rework end capture");
                        hip_ok(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "rework instantiate");
                        for (unsigned replay = 0; replay < 3; ++replay) {
                            reset();
                            hip_ok(hipGraphLaunch(exec, st), "rework replay");
                            hip_ok(hipStreamSynchronize(st), "rework graph sync");
                            check(same(partial_ref, host(partial, rows * 32 * msa::kMaxChunks * 130)) &&
                                  same(score_ref, host(scores, rows * 2 * p.max_blocks)) &&
                                  same(idx_ref, host(idx, rows * 2 * msa::kTopK)) && same(count_ref, host(cnt, rows * 2)) &&
                                  same(oa, host(p.o, p.q.size())) && same(ka, host(p.kc, nkv)) &&
                                  same(va, host(p.vc, nkv)) && same(ia, host(p.ikc, nik)) &&
                                  same(tickets, host(ticket, rows * 2 * 4)), "rework graph replay bytes");
                        }
                        hip_ok(hipGraphExecDestroy(exec), "rework destroy exec");
                        hip_ok(hipGraphDestroy(graph), "rework destroy graph");
                        hip_ok(hipStreamDestroy(st), "rework destroy stream");
                    }
                }
            }
        }
    }
    // Compare the same batched call in both arms, before any scratch reuse.
    // Isolated producers use the same cap/grid batching in all six modes;
    // selection, cache preparation, tickets and combine are outside the timing.
    if (quick) reset();
    const std::vector<unsigned> producer_caps = quick && n > 129 ? std::vector<unsigned>{scratch_windows}
                                                               : std::vector<unsigned>{1U, scratch_windows};
    for (unsigned cap : producer_caps) for (unsigned pvf : {0U, 1U}) {
        const std::size_t batch_rows = std::size_t{cap} * 16;
        float producer_ms[7] = {}, complete_ms[7] = {};
        float selection_ms = 0.0F;
        std::uint64_t partial_count = 0;
        for (unsigned r0 = 0; r0 < n; r0 += cap * 16) {
            const unsigned m = std::min(cap * 16, n - r0);
            auto v = prefill_view(p, pos0 + r0, std::min(m, 16U), 4, 2, pvf);
            v.scores = scores; v.partial = partial; v.idx = idx; v.idx_count = cnt; v.ticket = ticket;
            std::vector<float> score_ref, partial_ref, output_ref;
            std::vector<std::uint32_t> ids_ref, counts_ref;
            std::vector<std::uint16_t> k_ref, v_ref, ik_ref;
            for (unsigned mode : {0U, 1U, 2U, 3U, 4U, 5U, 6U}) {
                hip_ok(hipMemset(scores, 0xa5, batch_rows * 2 * p.max_blocks * sizeof(float)), "producer scores sentinel");
                hip_ok(hipMemset(partial, 0xa5, batch_rows * 32 * msa::kMaxChunks * 130 * sizeof(float)), "producer partial sentinel");
                hip_ok(hipMemset(idx, 0xa5, batch_rows * 2 * msa::kTopK * sizeof(std::uint32_t)), "producer ids sentinel");
                hip_ok(hipMemset(cnt, 0xa5, batch_rows * 2 * sizeof(std::uint32_t)), "producer count sentinel");
                mf::PrefillView batch{v, m, pos0 + r0, cap, 0, false, 32, mode};
                hip_ok(hipEventRecord(begin, nullptr), "complete prefill begin");
                mf::enqueue_prefill(batch, nullptr);
                complete_ms[mode] += elapsed();
                const auto sc = host(scores, batch_rows * 2 * p.max_blocks);
                const auto pt = host(partial, batch_rows * 32 * msa::kMaxChunks * 130);
                const auto ids = host(idx, batch_rows * 2 * msa::kTopK), counts = host(cnt, batch_rows * 2);
                const auto output = host(v.o, std::size_t{m} * 32 * 128);
                if (mode == 0) {
                    score_ref = sc; partial_ref = pt; ids_ref = ids; counts_ref = counts; output_ref = output;
                    k_ref = host(p.kc, nkv); v_ref = host(p.vc, nkv); ik_ref = host(p.ikc, nik);
                    report_r16_reuse(std::uint64_t{pos0} + r0, m, cap, pvf, ids, counts);
                    hip_ok(hipEventRecord(begin, nullptr), "shared selection begin");
                    mf::enqueue_prefill_selection(batch, nullptr);
                    selection_ms += elapsed();
                    check(same(ids_ref, host(idx, batch_rows * 2 * msa::kTopK)) && same(counts_ref, host(cnt, batch_rows * 2)),
                          "isolated selection IDs/counts/sentinels");
                }
                else check(same(score_ref, sc) && same(partial_ref, pt) && same(ids_ref, ids) &&
                           same(counts_ref, counts) && same(output_ref, output), "rework every-batch bytes mode=" +
                           std::to_string(mode) + " pos=" + std::to_string(pos0 + r0) + " pv=" + std::to_string(pvf));
                hip_ok(hipMemset(partial, 0xa5, batch_rows * 32 * msa::kMaxChunks * 130 * sizeof(float)), "timed partial sentinel");
                hip_ok(hipEventRecord(begin, nullptr), "producer begin");
                mf::enqueue_prefill_partials(batch, nullptr);
                producer_ms[mode] += elapsed();
                check(same(partial_ref, host(partial, batch_rows * 32 * msa::kMaxChunks * 130)), "isolated batched producer partial bytes mode=" +
                      std::to_string(mode) + " pos=" + std::to_string(pos0 + r0) + " pv=" + std::to_string(pvf));
                const auto tickets = host(ticket, batch_rows * 2 * 4);
                check(same(score_ref, host(scores, batch_rows * 2 * p.max_blocks)) &&
                      same(ids_ref, host(idx, batch_rows * 2 * msa::kTopK)) && same(counts_ref, host(cnt, batch_rows * 2)) &&
                      same(output_ref, host(v.o, std::size_t{m} * 32 * 128)) &&
                      same(k_ref, host(p.kc, nkv)) && same(v_ref, host(p.vc, nkv)) && same(ik_ref, host(p.ikc, nik)) &&
                      std::all_of(tickets.begin(), tickets.end(), [](std::uint32_t x) { return x == 0; }),
                      "isolated producer preserves scores/IDs/counts/output/caches/zero tickets mode=" + std::to_string(mode));
            }
            for (unsigned r = 0; r < m; ++r)
                partial_count += 2 * std::min<std::uint64_t>(msa::kTopK, (std::uint64_t{pos0} + r0 + r + 128) / 128);
        }
        std::printf("prefill batched producers pos0=%u n=%u scratch=%u pv_full=%u valid_chunks=%llu modes 0/1/2/3/4/5/6 %.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f ms; ns/chunk %.1f/%.1f/%.1f/%.1f/%.1f/%.1f/%.1f (prepared IDs, no combine)\n",
                    pos0, n, cap, pvf, static_cast<unsigned long long>(partial_count),
                    producer_ms[0], producer_ms[1], producer_ms[2], producer_ms[3], producer_ms[4], producer_ms[5], producer_ms[6],
                    1e6 * producer_ms[0] / static_cast<double>(partial_count),
                    1e6 * producer_ms[1] / static_cast<double>(partial_count),
                    1e6 * producer_ms[2] / static_cast<double>(partial_count),
                    1e6 * producer_ms[3] / static_cast<double>(partial_count),
                    1e6 * producer_ms[4] / static_cast<double>(partial_count),
                    1e6 * producer_ms[5] / static_cast<double>(partial_count),
                    1e6 * producer_ms[6] / static_cast<double>(partial_count));
        std::printf("prefill complete pos0=%u n=%u scratch=%u pv_full=%u modes 0/1/2/3/4/5/6 %.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f ms; shared-selection scheduling %.3f ms (R16 membership included in producers)\n",
                    pos0, n, cap, pvf, complete_ms[0], complete_ms[1], complete_ms[2], complete_ms[3], complete_ms[4], complete_ms[5], complete_ms[6], selection_ms);
    }
    // Index gates also run in quick mode. Compare default, Q32 and real Q64 in
    // each reused scratch batch, including causal holes and unused tail rows.
    // The complete producer checks above left every cache row prepared.
    // Small cases force one-window reuse; large cases reuse the bounded full
    // scratch allocation without copying that entire allocation per 16 rows.
    const std::vector<unsigned> index_caps = n > 129 ? std::vector<unsigned>{scratch_windows}
        : n >= 65 ? std::vector<unsigned>{1U, 4U, scratch_windows} : std::vector<unsigned>{1U, scratch_windows};
    for (unsigned cap : index_caps) {
        for (unsigned r0 = 0; r0 < n; r0 += cap * 16) {
            const unsigned m = std::min(n - r0, cap * 16);
            auto v = prefill_view(p, pos0 + r0, std::min(m, 16U));
            v.scores = scores; v.partial = partial; v.idx = idx; v.idx_count = cnt; v.ticket = ticket;
            std::vector<float> reference, complete_scores_ref, partial_ref, output_ref;
            std::vector<std::uint32_t> ids_ref, counts_ref, tickets_ref;
            std::vector<std::uint16_t> k_ref, v_ref, ik_ref;
            for (unsigned query_rows : {0U, 16U, 32U, 64U}) {
                const mf::PrefillView batch{v, m, pos0 + r0, cap, 0, query_rows != 0,
                                           query_rows == 0 ? 32U : query_rows};
                hip_ok(hipMemset(scores, 0xa5, rows * 2 * p.max_blocks * sizeof(float)), "index sentinel");
                mf::enqueue_prefill_index(batch, nullptr);
                const auto result = host(scores, rows * 2 * p.max_blocks);
                if (query_rows == 0) reference = result;
                else check(same(reference, result), "prefill index scores/sentinel pos0=" + std::to_string(pos0 + r0) +
                           " n=" + std::to_string(m) + " Q=" + std::to_string(query_rows));
                // Fresh sentinels expose extra writes; buffers are reused for the
                // next batch without reallocating (including a short final batch).
                hip_ok(hipMemset(scores, 0xa5, rows * 2 * p.max_blocks * sizeof(float)), "complete index sentinel");
                hip_ok(hipMemset(partial, 0xa5, rows * 32 * msa::kMaxChunks * 130 * sizeof(float)), "index partial sentinel");
                hip_ok(hipMemset(idx, 0xa5, rows * 2 * msa::kTopK * sizeof(std::uint32_t)), "index batch ids");
                hip_ok(hipMemset(cnt, 0xa5, rows * 2 * sizeof(std::uint32_t)), "index batch counts");
                hip_ok(hipMemset(ticket, 0, rows * 2 * 4 * sizeof(std::uint32_t)), "index batch tickets");
                hip_ok(hipMemset(p.o, 0xa5, p.q.size() * sizeof(float)), "index output sentinel");
                mf::enqueue_prefill(batch, nullptr);
                const auto sc = host(scores, rows * 2 * p.max_blocks);
                const auto pt = host(partial, rows * 32 * msa::kMaxChunks * 130);
                const auto ids = host(idx, rows * 2 * msa::kTopK), counts = host(cnt, rows * 2);
                const auto output = host(p.o, p.q.size());
                const auto tickets = host(ticket, rows * 2 * 4);
                if (query_rows == 0) {
                    complete_scores_ref = sc; partial_ref = pt; output_ref = output;
                    ids_ref = ids; counts_ref = counts; tickets_ref = tickets;
                    k_ref = host(p.kc, nkv); v_ref = host(p.vc, nkv); ik_ref = host(p.ikc, nik);
                } else check(same(complete_scores_ref, sc) && same(partial_ref, pt) &&
                             same(ids_ref, ids) && same(counts_ref, counts) && same(output_ref, output) &&
                             same(tickets_ref, tickets) && same(k_ref, host(p.kc, nkv)) &&
                             same(v_ref, host(p.vc, nkv)) && same(ik_ref, host(p.ikc, nik)),
                             "prefill index complete bytes pos0=" + std::to_string(pos0 + r0) +
                             " n=" + std::to_string(m) + " scratch=" + std::to_string(cap) +
                             " Q=" + std::to_string(query_rows) + " ties=" + std::to_string(ties));
                check(std::all_of(tickets.begin(), tickets.end(), [](std::uint32_t x) { return x == 0; }),
                      "prefill index zero tickets");
            }
        }
    }
    // Reject unsupported views before launching any work.
    auto invalid = mf::PrefillView{prefill_view(p, pos0, std::min(n, 16U)), 0, pos0, scratch_windows};
    bool rejected = false;
    try { mf::enqueue_prefill(invalid, nullptr); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "batched prefill rejects empty chunk");
    invalid.rows = n;
    invalid.window.attend_version = 2;
    rejected = false;
    try { mf::enqueue_prefill(invalid, nullptr); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "batched prefill rejects unsupported attend version");
    invalid.window.attend_version = 4;
    invalid.window.heads_per_wave = 4;
    rejected = false;
    try { mf::enqueue_prefill(invalid, nullptr); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "batched prefill rejects unsupported heads per wave");
    invalid.window.heads_per_wave = 2;
    invalid.index_rows = 48;
    rejected = false;
    try { mf::enqueue_prefill(invalid, nullptr); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "batched prefill rejects unsupported index rows");
    invalid.index_rows = 32;
    invalid.msa_rework = 7;
    rejected = false;
    try { mf::enqueue_prefill(invalid, nullptr); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "batched prefill rejects unsupported rework mode");
    auto invalid_producer = mf::PrefillView{prefill_view(p, pos0, std::min(n, 16U), 4, 2),
                                          std::min(n, 16U), pos0, 1, 0, false, 32, 7};
    rejected = false;
    try { mf::enqueue_prefill_partials(invalid_producer, nullptr); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "prefill producer diagnostic rejects unsupported rework mode");
    (void)hipEventDestroy(begin);
    (void)hipEventDestroy(end);
    for (void* ptr : {static_cast<void*>(scores), static_cast<void*>(partial), static_cast<void*>(idx),
                      static_cast<void*>(cnt), static_cast<void*>(ticket)}) (void)hipFree(ptr);
    release_prefill(p);
}

void prefill_batch_cases(std::mt19937& rng) {
    for (std::uint32_t pos0 : {2047U, 2048U, 2049U}) prefill_batch_case(pos0, 65, rng, true);
    for (std::uint32_t pos0 : {0U, 1000U, 2040U, 4096U, 8190U, 30000U})
        for (std::uint32_t n : {1U, 15U, 16U, 17U, 63U, 64U, 65U, 100U, 127U, 128U, 129U, 1024U, 4096U}) prefill_batch_case(pos0, n, rng);
    for (std::uint32_t pos0 : {127U, 128U, 129U, 2047U, 2048U, 2049U, 4095U, 4096U, 4097U})
        prefill_batch_case(pos0, 17, rng, true);
    for (std::uint32_t pos0 : {0U, 2040U, 4096U, 8190U, 30000U})
        prefill_batch_case(pos0, 100, rng, true);
    if (std::getenv("STRIX_MMATTN_REWORK_LONG") != nullptr) {
        prefill_batch_case(0, 8192, rng);
        prefill_batch_case(0, 32768, rng);
    }
    // 8K at pos0=0: 128 dense windows + 384 sparse windows, 60 layers.
    const unsigned cap = static_cast<unsigned>((512ULL * 1024 * 1024) / mf::prefill_window_bytes(64));
    const auto launch_count = [cap](unsigned chunk_windows) {
        unsigned launches = 0;
        for (unsigned chunk = 0; chunk < 512; chunk += chunk_windows) {
            ++launches;  // prep
            const unsigned end = std::min(512U, chunk + chunk_windows);
            for (unsigned first = chunk; first < end; first += cap) {
                const unsigned last = std::min(end, first + cap);
                if (first < 128) ++launches;
                if (last > 128) launches += 2;
            }
        }
        return launches;
    };
    std::printf("8K prompt/60 layers: old 896 launches/layer (53760 total); new %u/layer (%u total) "
                "with 4096-row chunks, %u/layer (%u total) with one chunk; scratch cap %u at capacity 8192\n",
                launch_count(256), launch_count(256) * 60, launch_count(512), launch_count(512) * 60, cap);
}

void prefill_case(std::uint64_t T, std::mt19937& rng, bool bench) {
    Prefill p = make_prefill(T, rng);
    const std::size_t nkv = T * 2 * 128;
    if (!bench) {
        prefill_old(p, nullptr);
        hip_ok(hipDeviceSynchronize(), "prefill A");
        const auto oa = host(p.o, p.q.size());
        const auto ka = host(p.kc, nkv), va = host(p.vc, nkv), ia = host(p.ikc, T * 128);
        for (const auto& [wrows, dense] : {std::pair{16U, true}, std::pair{5U, true}, std::pair{16U, false}}) {
            g_prefill_dense = dense;
            for (auto* ptr : {p.kc, p.vc}) hip_ok(hipMemset(ptr, 0, nkv * 2), "memset");
            hip_ok(hipMemset(p.ikc, 0, T * 256), "memset");
            hip_ok(hipMemset(p.o, 0, p.q.size() * 4), "memset");
            prefill_fused(p, wrows, nullptr);
            hip_ok(hipDeviceSynchronize(), "prefill B");
            g_prefill_dense = true;
            const bool ok = same(oa, host(p.o, p.q.size())) && same(ka, host(p.kc, nkv)) && same(va, host(p.vc, nkv)) &&
                            same(ia, host(p.ikc, T * 128));
            check(ok, "prefill T " + std::to_string(T) + " windows of " + std::to_string(wrows));
            std::printf("prefill T %llu fused windows of %u (dense below 2048: %d) vs current 8-row windows: %s\n",
                        static_cast<unsigned long long>(T), wrows, dense ? 1 : 0, ok ? "bitwise equal" : "MISMATCH");
        }
    } else {
        hipStream_t st;
        hip_ok(hipStreamCreateWithFlags(&st, hipStreamNonBlocking), "stream");
        hipEvent_t e0, e1;
        hip_ok(hipEventCreate(&e0), "event");
        hip_ok(hipEventCreate(&e1), "event");
        const unsigned pver[4] = {0, 3, 3, 4}, phw[4] = {0, 16, 16, 2};  // v4: phw = heads per wave; arm 1: predicated PV
        auto run = [&](int arm) {
            hip_ok(hipEventRecord(e0, st), "rec");
            if (arm == 0) prefill_old(p, st);
            else prefill_fused(p, mf::kMaxRows, st, pver[arm], phw[arm], arm == 1 ? 0U : 1U);
            hip_ok(hipEventRecord(e1, st), "rec");
            hip_ok(hipEventSynchronize(e1), "sync");
            float ms = 0;
            hip_ok(hipEventElapsedTime(&ms, e0, e1), "elapsed");
            return double(ms);
        };
        for (int arm = 0; arm < 4; ++arm) run(arm);
        std::vector<double> t[4];
        std::string rounds;
        for (int round = 0; round < 3; ++round) {  // A v3-pred v3-full v4-h2 A, x3
            rounds += " [";
            for (int k = 0; k <= 4; ++k) {
                const int arm = k == 4 ? 0 : k;
                const double ms = run(arm);
                t[arm].push_back(ms);
                char buf[24];
                std::snprintf(buf, sizeof buf, k == 0 ? "%.2f" : " %.2f", ms);
                rounds += buf;
            }
            rounds += "]";
        }
        const double ma = median(t[0]);
        std::printf("prefill T %5llu ms/layer (16-row fused windows, dense below 2048): A(current 8-row) %.2f v3-pred %.2f (%.3f) "
                    "v3-full %.2f (%.3f) v4-h2 %.2f (%.3f)  runs:%s\n", static_cast<unsigned long long>(T), ma, median(t[1]),
                    median(t[1]) / ma, median(t[2]), median(t[2]) / ma, median(t[3]), median(t[3]) / ma, rounds.c_str());
        std::fflush(stdout);
        (void)hipEventDestroy(e0);
        (void)hipEventDestroy(e1);
        (void)hipStreamDestroy(st);
    }
    release_prefill(p);
}

// ---------------------------------------------------------------- microbench

__global__ void fill_bf16(std::uint16_t* p, std::size_t n, std::uint32_t seed) {
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x) {
        std::uint32_t h = static_cast<std::uint32_t>(i) * 2654435761U ^ seed;
        h ^= h >> 15U;
        h *= 2246822519U;
        h ^= h >> 13U;
        const float f = (static_cast<float>(h & 0xFFFFU) / 65536.0F - 0.5F) * 4.0F;
        p[i] = static_cast<std::uint16_t>(msa::f2u(f) >> 16U);
    }
}

bool g_exp = false;  // STRIX_MMATTN_EXP=1: experiment arms (see bench_cell)  // heads per block of the dense arm (STRIX_MMATTN_DENSE_HW)

__global__ void empty_kernel(int* p) {
    if (p != nullptr && threadIdx.x == 1024) p[0] = 1;
}
// Launch cost: n empty launches back to back, stream and graph (per launch).
void launch_probe() {
    hipStream_t st;
    hip_ok(hipStreamCreateWithFlags(&st, hipStreamNonBlocking), "stream");
    hipEvent_t e0, e1;
    hip_ok(hipEventCreate(&e0), "event");
    hip_ok(hipEventCreate(&e1), "event");
    const int n = 200;
    double us[2] = {0, 0};
    hipGraph_t g;
    hipGraphExec_t ge;
    hip_ok(hipStreamBeginCapture(st, hipStreamCaptureModeThreadLocal), "capture");
    for (int i = 0; i < n; ++i) hipLaunchKernelGGL(empty_kernel, dim3(32), dim3(128), 0, st, nullptr);
    hip_ok(hipStreamEndCapture(st, &g), "end capture");
    hip_ok(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0), "instantiate");
    for (int mode = 0; mode < 2; ++mode)
        for (int rep = 0; rep < 3; ++rep) {
            hip_ok(hipEventRecord(e0, st), "rec");
            if (mode == 0)
                for (int i = 0; i < n; ++i) hipLaunchKernelGGL(empty_kernel, dim3(32), dim3(128), 0, st, nullptr);
            else
                hip_ok(hipGraphLaunch(ge, st), "graph");
            hip_ok(hipEventRecord(e1, st), "rec");
            hip_ok(hipEventSynchronize(e1), "sync");
            float ms = 0;
            hip_ok(hipEventElapsedTime(&ms, e0, e1), "elapsed");
            if (rep > 0) us[mode] += 1000.0 * ms / n / 2;
        }
    std::printf("launch probe: empty kernel back to back %.2f us/launch (stream), %.2f us/launch (graph)\n", us[0], us[1]);
    (void)hipGraphExecDestroy(ge);
    (void)hipGraphDestroy(g);
    (void)hipEventDestroy(e0);
    (void)hipEventDestroy(e1);
    (void)hipStreamDestroy(st);
}

struct BenchLayer {
    Data d;  // shapes and host inputs (caches not filled on the host)
    Dev x;
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

void bench_cell(std::uint64_t ctx, unsigned seqs, unsigned W, std::mt19937& rng) {
    Case c{seqs, W, 32, 2, 2, std::vector<std::uint64_t>(seqs, ctx - 1), false, "bench"};
    const std::uint64_t per_layer = std::uint64_t(seqs) * ctx * (128 + 2 * 2 * 128) * 2;
    const unsigned L = static_cast<unsigned>(std::clamp<std::uint64_t>((256ULL << 20U) / per_layer + 1, 2, 64));
    // Small host data (inputs); caches filled on the device.
    std::vector<BenchLayer> layers(L);
    for (unsigned li = 0; li < L; ++li) {
        Case cs = c;
        BenchLayer& bl = layers[li];
        bl.d.c = cs;
        bl.d.cap = ctx;
        bl.d.max_blocks = static_cast<std::uint32_t>(msa::block_count(ctx));
        std::normal_distribution<float> n(0.0F, 1.0F);
        const unsigned R = seqs * W;
        bl.d.q.resize(std::size_t(R) * 32 * 128);
        bl.d.k.resize(std::size_t(R) * 2 * 128);
        bl.d.v.resize(bl.d.k.size());
        bl.d.iq.resize(std::size_t(R) * 2 * 128);
        bl.d.ik.resize(std::size_t(R) * 128);
        bl.d.w.resize(512);
        for (auto* vec : {&bl.d.q, &bl.d.k, &bl.d.v, &bl.d.iq, &bl.d.ik, &bl.d.w})
            for (auto& y : *vec) y = n(rng);
        for (unsigned s = 0; s < seqs; ++s)
            for (unsigned r = 0; r < W; ++r) {
                bl.d.pos.push_back(ctx - W + r);
                bl.d.rope.push_back(msa::rope_row(bl.d.pos.back(), 1e7));
            }
        Dev& x = bl.x;
        x.R = R;
        x.q = dev(bl.d.q);
        x.k = dev(bl.d.k);
        x.v = dev(bl.d.v);
        x.iq = dev(bl.d.iq);
        x.ik = dev(bl.d.ik);
        x.w = dev(bl.d.w);
        x.q_old = dev(bl.d.q);
        x.iq_old = dev(bl.d.iq);
        x.rope = dev(bl.d.rope);
        x.pos = dev(bl.d.pos);
        for (unsigned s = 0; s < seqs; ++s) {
            for (auto* vec : {&x.kc, &x.vc}) {
                vec->push_back(dalloc<std::uint16_t>(ctx * 2 * 128));
                hipLaunchKernelGGL(fill_bf16, dim3(1024), dim3(256), 0, nullptr, vec->back(), ctx * 2 * 128, li * 977U + s * 31U + (vec == &x.kc ? 1U : 2U));
            }
            x.ikc.push_back(dalloc<std::uint16_t>(ctx * 128));
            hipLaunchKernelGGL(fill_bf16, dim3(1024), dim3(256), 0, nullptr, x.ikc.back(), ctx * 128, li * 977U + s * 31U + 3U);
        }
        x.kc_tab = dev(x.kc);
        x.vc_tab = dev(x.vc);
        x.ikc_tab = dev(x.ikc);
        x.scores = dalloc<float>(std::size_t(R) * 2 * bl.d.max_blocks);
        x.partial = dalloc<float>(std::size_t(R) * 32 * msa::kMaxChunks * 130);
        x.o = dalloc<float>(bl.d.q.size());
        x.idx = dalloc<std::uint32_t>(std::size_t(R) * 2 * msa::kTopK);
        x.cnt = dalloc<std::uint32_t>(std::size_t(R) * 2);
        x.ticket = dalloc<std::uint32_t>(std::size_t(R) * 2 * 4);
    }
    hip_ok(hipDeviceSynchronize(), "fill");
    hipStream_t st;
    hip_ok(hipStreamCreateWithFlags(&st, hipStreamNonBlocking), "stream");
    // Arms: 0 = current kernels (5 launches per sequence); 1..3 = fused with 16 / 8 / 4 heads per attend block;
    // 4 = fused dense single launch (only when every row is below 2048 positions).
    const bool dense_ok = ctx <= msa::kTopK * msa::kBlock;
    const int n_arms = g_exp ? 5 : 6;
    // Arms: 0 current, 1 v1-16, 2 v2-16, 3 v3-16, 4 = dense v3-16 when every row is below 2048 positions, else v3-8
    // (experiment mode: v3-16 with the selection precomputed).
    // Arms: 0 current, 1 v3-16 (default), 2 v4 2 heads/wave, 3 v4 4 heads/wave, 4 dense v3 (<= 2K) else v4 1 head/wave,
    // 5 dense v4 2 heads/wave (<= 2K) else v3-8. Experiment mode keeps its own meaning for arm 4 (v3 presel).
    // STRIX_MMATTN_BENCH_DIET=1 (sparse cells): arm 5 = v3-16 with lds_diet instead of v3-8, so v3 42 KB (arm 2),
    // v3 32 KB (5), v4-h1 (4) and v4-h2 (3) share one process, cache rotation and bracket order.
    const char* diet_env = std::getenv("STRIX_MMATTN_BENCH_DIET");
    const bool diet_arm = !g_exp && !dense_ok && diet_env != nullptr && std::string(diet_env) == "1";
    const unsigned arm_hw[6] = {0, 16, 16, 16, 16, diet_arm ? 16U : 8U}, arm_ver[6] = {0, 3, 3, 4, dense_ok ? 3U : 4U, dense_ok ? 4U : 3U};
    const unsigned arm_hpw[6] = {0, 0, 0, 2, dense_ok ? 0U : 1U, dense_ok ? 2U : 0U};
    // Experiment mode: arm 3 = v2-16 again (repeatability), arm 4 = v2-16 with the selection precomputed (index launch
    // + attend select=false; idx from the previous select runs).
    auto arm_view = [&](BenchLayer& bl, int arm) {
        mf::DecodeView v = new_view(bl.d, bl.x);
        v.heads_per_block = arm_hw[arm];
        v.attend_version = g_exp && arm == 4 ? 3U : arm_ver[arm];
        v.heads_per_wave = arm_hpw[arm];
        v.dense = (arm == 4 || arm == 5) && !g_exp && dense_ok;
        v.pv_full_tiles = arm == 1 ? 0U : 1U;  // arm 1 = version 3 as measured (predicated PV only)
        v.lds_diet = diet_arm && arm == 5;
        return v;
    };
    auto enqueue_arm = [&](int arm) {
        for (auto& bl : layers) {
            if (arm == 0) {
                enqueue_old(bl.d, bl.x, st);
            } else if (g_exp && arm == 4) {
                mf::enqueue_index(arm_view(bl, arm), st);
                mf::enqueue_attend(arm_view(bl, arm), false, st);
            } else {
                mf::enqueue_decode(arm_view(bl, arm), st);
            }
        }
    };
    hipEvent_t e0, e1;
    hip_ok(hipEventCreate(&e0), "event");
    hip_ok(hipEventCreate(&e1), "event");
    // Calibrate reps so one run of arm A takes ~25 ms.
    for (int arm = 0; arm < n_arms; ++arm) enqueue_arm(arm);
    hip_ok(hipStreamSynchronize(st), "warm");
    hip_ok(hipEventRecord(e0, st), "rec");
    enqueue_arm(0);
    hip_ok(hipEventRecord(e1, st), "rec");
    hip_ok(hipEventSynchronize(e1), "sync");
    float ms0 = 0;
    hip_ok(hipEventElapsedTime(&ms0, e0, e1), "elapsed");
    const int reps = std::clamp(static_cast<int>(25.0F / std::max(ms0, 0.01F)), 2, 400);
    hipGraphExec_t gexec[6];
    hipGraph_t graph[6];
    for (int arm = 0; arm < n_arms; ++arm) {
        hip_ok(hipStreamBeginCapture(st, hipStreamCaptureModeThreadLocal), "capture");
        enqueue_arm(arm);
        hip_ok(hipStreamEndCapture(st, &graph[arm]), "end capture");
        hip_ok(hipGraphInstantiate(&gexec[arm], graph[arm], nullptr, nullptr, 0), "instantiate");
    }
    auto run = [&](int arm, bool graph_mode) {
        if (graph_mode) hip_ok(hipGraphLaunch(gexec[arm], st), "graph");  // warm
        else enqueue_arm(arm);
        hip_ok(hipEventRecord(e0, st), "rec");
        for (int i = 0; i < reps; ++i) {
            if (graph_mode) hip_ok(hipGraphLaunch(gexec[arm], st), "graph");
            else enqueue_arm(arm);
        }
        hip_ok(hipEventRecord(e1, st), "rec");
        hip_ok(hipEventSynchronize(e1), "sync");
        float ms = 0;
        hip_ok(hipEventElapsedTime(&ms, e0, e1), "elapsed");
        return 1000.0 * ms / (double(reps) * L);
    };
    const char* names[6] = {"A(current)", "v3-pred", "v3-full", "v4-h2", g_exp ? "v3-presel" : dense_ok ? "D3(dense)" : "v4-h1",
                            dense_ok ? "D4-h2(dense)" : diet_arm ? "v3-diet" : "v3-8"};
    for (int gm = 0; gm < 2; ++gm) {
        std::vector<double> t[6];
        std::string rounds;
        for (int round = 0; round < 3; ++round) {  // A B... A, x3 (every fused arm between two A runs)
            rounds += " [";
            for (int k = 0; k <= n_arms; ++k) {
                const int arm = k == n_arms ? 0 : k;
                const double us = run(arm, gm != 0);
                t[arm].push_back(us);
                char buf[24];
                std::snprintf(buf, sizeof buf, k == 0 ? "%.1f" : " %.1f", us);
                rounds += buf;
            }
            rounds += "]";
        }
        const double ma = median(t[0]);
        std::string line;
        for (int arm = 0; arm < n_arms; ++arm) {
            char buf[64];
            std::snprintf(buf, sizeof buf, " %s %.1f (%.3f)", names[arm], median(t[arm]), median(t[arm]) / ma);
            line += buf;
        }
        std::printf("cell ctx %6llu seqs %u W %u %-6s L %2u reps %3d us/layer:%s  runs:%s\n", static_cast<unsigned long long>(ctx),
                    seqs, W, gm != 0 ? "graph" : "stream", L, reps, line.c_str(), rounds.c_str());
    }
    // Per-kernel breakdown (stream, events between launches; A's kernels grouped by kind over the sequences).
    {
        const int n = std::max(1, reps / 2);
        std::vector<hipEvent_t> ev(8);
        for (auto& e : ev) hip_ok(hipEventCreate(&e), "event");
        double ta[4] = {0, 0, 0, 0}, tb[2] = {0, 0};
        for (int i = 0; i < n; ++i)
            for (auto& bl : layers) {
                const Case& cc = bl.d.c;
                hip_ok(hipEventRecord(ev[0], st), "rec");
                for (unsigned s = 0; s < cc.seqs; ++s) {
                    const unsigned r0 = s * W;
                    mh::enqueue_prep({W, 32, 2, 2, bl.x.q_old + std::size_t(r0) * 32 * 128, bl.x.k + std::size_t(r0) * 256,
                                      bl.x.v + std::size_t(r0) * 256, bl.x.iq_old + std::size_t(r0) * 256, bl.x.ik + std::size_t(r0) * 128,
                                      bl.x.w, bl.x.w + 128, bl.x.w + 256, bl.x.w + 384, bl.x.rope + r0, bl.x.pos + r0, bl.x.kc[s],
                                      bl.x.vc[s], bl.x.ikc[s]}, st);
                }
                hip_ok(hipEventRecord(ev[1], st), "rec");
                for (unsigned s = 0; s < cc.seqs; ++s) {
                    const unsigned r0 = s * W;
                    mh::enqueue_scores({W, 2, bl.d.max_blocks, bl.x.iq_old + std::size_t(r0) * 256, bl.x.ikc[s], bl.x.pos + r0,
                                        bl.x.scores + std::size_t(r0) * 2 * bl.d.max_blocks}, st);
                }
                hip_ok(hipEventRecord(ev[2], st), "rec");
                for (unsigned s = 0; s < cc.seqs; ++s) {
                    const unsigned r0 = s * W;
                    mh::enqueue_select({W, 2, bl.d.max_blocks, bl.x.pos + r0, bl.x.scores + std::size_t(r0) * 2 * bl.d.max_blocks,
                                        bl.x.idx + std::size_t(r0) * 2 * 16, bl.x.cnt + std::size_t(r0) * 2}, st);
                }
                hip_ok(hipEventRecord(ev[3], st), "rec");
                for (unsigned s = 0; s < cc.seqs; ++s) {
                    const unsigned r0 = s * W;
                    mh::enqueue_attention({W, 2, bl.x.q_old + std::size_t(r0) * 32 * 128, bl.x.kc[s], bl.x.vc[s], bl.x.pos + r0,
                                           bl.x.idx + std::size_t(r0) * 2 * 16, bl.x.cnt + std::size_t(r0) * 2,
                                           bl.x.partial + std::size_t(r0) * 32 * 16 * 130, bl.x.o + std::size_t(r0) * 32 * 128}, st);
                }
                hip_ok(hipEventRecord(ev[4], st), "rec");
                mf::enqueue_index(new_view(bl.d, bl.x), st);
                hip_ok(hipEventRecord(ev[5], st), "rec");
                mf::enqueue_attend(arm_view(bl, 3), true, st);
                hip_ok(hipEventRecord(ev[6], st), "rec");
                hip_ok(hipEventSynchronize(ev[6]), "sync");
                float ms[6];
                for (std::size_t k = 0; k < 6; ++k) hip_ok(hipEventElapsedTime(&ms[k], ev[k], ev[k + 1]), "elapsed");
                for (std::size_t k = 0; k < 4; ++k) ta[k] += ms[k];
                tb[0] += ms[4];
                tb[1] += ms[5];
            }
        const double div = n * double(L) / 1000.0;
        std::printf("  breakdown (us/layer, stream, events incl. launch gaps): A prep %.1f scores %.1f select %.1f attention %.1f"
                    " | B16 index %.1f attend %.1f\n", ta[0] / div, ta[1] / div, ta[2] / div, ta[3] / div, tb[0] / div, tb[1] / div);
        for (auto& e : ev) (void)hipEventDestroy(e);
    }
    // Experiment: per-kernel time inside the back-to-back fused sequence (events after every launch), per arm.
    if (g_exp) {
        std::vector<hipEvent_t> ev(2 * L + 1);
        for (auto& e : ev) hip_ok(hipEventCreate(&e), "event");
        for (int round = 0; round < 2; ++round)
            for (int arm = 1; arm < n_arms; ++arm) {
                double ti = 0, ta = 0;
                for (int i = 0; i < reps; ++i) {
                    hip_ok(hipEventRecord(ev[0], st), "rec");
                    for (unsigned li = 0; li < L; ++li) {
                        mf::DecodeView v = arm_view(layers[li], arm);
                        mf::enqueue_index(v, st);
                        hip_ok(hipEventRecord(ev[2 * li + 1], st), "rec");
                        mf::enqueue_attend(v, !(arm == 4), st);
                        hip_ok(hipEventRecord(ev[2 * li + 2], st), "rec");
                    }
                    hip_ok(hipEventSynchronize(ev[2 * L]), "sync");
                    for (unsigned k = 0; k < 2 * L; ++k) {
                        float ms = 0;
                        hip_ok(hipEventElapsedTime(&ms, ev[k], ev[k + 1]), "elapsed");
                        (k % 2 == 0 ? ti : ta) += ms;
                    }
                }
                std::printf("  seqbreak %-9s (us/layer, back to back, events between launches): index %.1f attend %.1f total %.1f\n",
                            names[arm], 1000.0 * ti / (reps * double(L)), 1000.0 * ta / (reps * double(L)),
                            1000.0 * (ti + ta) / (reps * double(L)));
            }
        for (auto& e : ev) (void)hipEventDestroy(e);
    }
    // In-kernel phase profile of the fused arms on layer 0 (wall clock 100 MHz; mean over blocks that reached the
    // phase; span = last stamp - first stamp of the launch).
    {
        BenchLayer& bl = layers[0];
        const std::size_t n_att = std::size_t(bl.x.R) * 2 * 4 * msa::kMaxChunks * 16, n_idx = 4 * 1024;
        std::uint64_t* ph = dalloc<std::uint64_t>(n_att);
        std::uint64_t* phi = dalloc<std::uint64_t>(n_idx);
        for (int rep_arm = 0; rep_arm < (g_exp ? 5 : 1) * (n_arms - 1); ++rep_arm) {
            const int arm = 1 + rep_arm % (n_arms - 1);
            mf::DecodeView v = arm_view(bl, arm);
            v.phase = ph;
            v.phase_index = arm == 4 ? nullptr : phi;
            hip_ok(hipMemsetAsync(ph, 0, n_att * 8, st), "memset");  // on st: the null stream does not order it
            hip_ok(hipMemsetAsync(phi, 0, n_idx * 8, st), "memset");
            if (g_exp && arm == 4) {
                mf::enqueue_index(v, st);
                mf::enqueue_attend(v, false, st);
            } else {
                mf::enqueue_decode(v, st);
            }
            hip_ok(hipStreamSynchronize(st), "phase sync");
            const auto a = host(ph, n_att);
            const auto b = host(phi, n_idx);
            const unsigned hw = mf::attend_heads(v), blocks = bl.x.R * 2 * (16 / hw) * msa::kMaxChunks;
            double sum[7] = {0, 0, 0, 0, 0, 0, 0}, cyc[7] = {0, 0, 0, 0, 0, 0, 0};
            unsigned cntp[7] = {0, 0, 0, 0, 0, 0, 0};
            std::uint64_t lo = ~0ULL, hi = 0;
            for (unsigned blk = 0; blk < blocks; ++blk) {
                const std::uint64_t* p = &a[std::size_t(blk) * 16];
                if (p[0] == 0) continue;
                lo = std::min(lo, p[0]);
                for (int k = 0; k < 8; ++k)
                    if (p[k] != 0) hi = std::max(hi, p[k]);
                for (int k = 0; k < 7; ++k)
                    if (p[k] != 0 && p[k + 1] != 0) {
                        sum[k] += double(p[k + 1] - p[k]);
                        cyc[k] += double((p[9 + k] - p[8 + k]) & 0xFFFFFULL);
                        ++cntp[k];
                    }
            }
            auto us = [&](int k) { return cntp[k] != 0 ? sum[k] / cntp[k] / 100.0 : 0.0; };
            std::printf("  phases %-8s attend (us, mean per block): select %.2f loads+qnorm+K %.2f scores %.2f V %.2f PV+partial %.2f"
                        " ticket %.2f combine %.2f (last blocks %u) | span %.2f", names[arm], us(0), us(1), us(2), us(3), us(4), us(5),
                        us(6), cntp[6], hi > lo ? double(hi - lo) / 100.0 : 0.0);
            {
                double tw = 0, tc = 0;
                for (int k = 0; k < 7; ++k) {
                    tw += sum[k];
                    tc += cyc[k];
                }
                std::printf(" | shader MHz %.0f (PV phase %.0f)", tw > 0 ? tc / tw * 100.0 : 0.0, sum[4] > 0 ? cyc[4] / sum[4] * 100.0 : 0.0);
            }
            if (arm != 4 || g_exp) {
                const unsigned cap = mf::index_grid_cap();
                double setup = 0, loop = 0, items = 0;
                unsigned nb = 0;
                std::uint64_t ilo = ~0ULL, ihi = 0;
                for (unsigned k = 0; k < cap && k < 1024; ++k) {
                    const std::uint64_t* p = &b[std::size_t(k) * 4];
                    if (p[0] == 0) continue;
                    ++nb;
                    setup += double(p[1] - p[0]);
                    loop += double(p[2] - p[1]);
                    items += double(p[3]);
                    ilo = std::min(ilo, p[0]);
                    ihi = std::max(ihi, p[2]);
                }
                if (nb != 0)
                    std::printf(" || index blocks %u: setup %.2f loop %.2f us, items/block %.1f, span %.2f, index->attend gap %.2f", nb,
                                setup / nb / 100.0, loop / nb / 100.0, items / nb, double(ihi - ilo) / 100.0,
                                lo > ihi ? double(lo - ihi) / 100.0 : -1.0);
            }
            std::printf("\n");
        }
        (void)hipFree(ph);
        (void)hipFree(phi);
    }
    for (int arm = 0; arm < n_arms; ++arm) {
        (void)hipGraphExecDestroy(gexec[arm]);
        (void)hipGraphDestroy(graph[arm]);
    }
    (void)hipEventDestroy(e0);
    (void)hipEventDestroy(e1);
    (void)hipStreamDestroy(st);
    for (auto& bl : layers) release(bl.x);
    std::fflush(stdout);
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);  // line-buffered: a killed process keeps its log
    std::mt19937 rng(20261024);
    // Focused producer/index byte checks; omit the unrelated decode suite
    // and the exhaustive complete-prefill matrix. Optional list: pos0:n,pos0:n.
    if (const char* quick = std::getenv("STRIX_MSA_TEST_QUICK"); quick != nullptr && std::strcmp(quick, "1") == 0) {
        if (const char* list = std::getenv("STRIX_MSA_TEST_CASES")) {
            const std::string cells = list;
            for (std::size_t at = 0; at < cells.size();) {
                const std::size_t end = cells.find(',', at);
                const std::string cell = cells.substr(at, end == std::string::npos ? end : end - at);
                unsigned pos0 = 0, n = 0;
                char extra = 0;
                if (std::sscanf(cell.c_str(), "%u:%u%c", &pos0, &n, &extra) != 2 || n == 0)
                    throw std::invalid_argument("STRIX_MSA_TEST_CASES must be pos0:n,pos0:n with n > 0");
                prefill_batch_case(pos0, n, rng, false, true);
                if (end == std::string::npos) break;
                at = end + 1;
            }
        } else {
            for (unsigned pos0 : {0U, 4096U}) {
                prefill_batch_case(pos0, 4096, rng, false, true);
                for (unsigned n : {63U, 64U, 65U}) prefill_batch_case(pos0, n, rng, false, true);
            }
            for (unsigned pos0 : {127U, 128U, 129U, 2047U, 2048U, 2049U})
                prefill_batch_case(pos0, 17, rng, true, true);
        }
        // Mandatory real four-subtile Q64 + tail, even with a custom case list.
        // Cross the dense boundary and exercise equal scores/ID tie-breaking.
        for (unsigned pos0 : {2047U, 2048U, 2049U}) prefill_batch_case(pos0, 65, rng, true, true);
        std::printf("strix-mmattn-fused-tests: quick producer/index cases, %d failure(s)\n", g_fail);
        return g_fail != 0 ? 1 : 0;
    }
    const char* only_bench = std::getenv("STRIX_MMATTN_BENCH_ONLY");
    if (only_bench == nullptr || std::strcmp(only_bench, "1") != 0) {
        const std::vector<Case> cases = {
            {1, 1, 32, 2, 2, {0}, false, "pos0"},
            {1, 1, 32, 2, 2, {100}, false, "r1-100"},
            {1, 3, 32, 2, 2, {2047}, false, "w3-2047"},
            {1, 2, 32, 2, 2, {2048}, false, "w2-2048"},
            {1, 8, 32, 2, 2, {4095}, false, "w8-4095"},
            {1, 8, 64, 4, 4, {4100}, false, "full-w8"},
            {1, 1, 64, 4, 4, {9000}, false, "full-9000"},
            {1, 5, 32, 2, 2, {20000}, false, "w5-20000"},
            {1, 8, 32, 2, 2, {70000}, false, "w8-70000"},
            {1, 1, 32, 2, 2, {131071}, false, "r1-128K"},
            {1, 2, 32, 2, 2, {32767}, false, "w2-32767"},
            {1, 2, 32, 2, 2, {32895}, false, "w2-32895"},
            {1, 1, 32, 2, 2, {524287}, false, "r1-512K"},
            {1, 4, 32, 2, 2, {40000}, true, "ties-w4"},
            {1, 1, 32, 2, 2, {3000}, true, "ties-small"},
            {2, 1, 32, 2, 2, {500, 9000}, false, "n2"},
            {8, 1, 32, 2, 2, {0, 1, 127, 128, 1000, 2046, 2047, 1500}, false, "n8-dense"},
            {2, 8, 32, 2, 2, {130, 2047}, false, "n2w8-dense"},
            {4, 1, 32, 2, 2, {0, 2047, 2048, 33000}, false, "n4-mixed"},
            {8, 1, 32, 2, 2, {8191, 8191, 100, 70000, 4096, 129, 1, 20000}, false, "n8-mixed"},
            {8, 1, 32, 2, 2, {131071, 131071, 131071, 131071, 131071, 131071, 131071, 131071}, false, "n8-128K"},
            {4, 2, 32, 2, 2, {300, 5000, 16383, 16384}, true, "n4w2-ties"},
            {2, 8, 32, 2, 2, {8200, 65000}, false, "n2w8"},
        };
        // STRIX_MSA_TEST_DECODE_ONLY=1: the decode cases only (the prefill matrices take several minutes)
        const char* decode_only = std::getenv("STRIX_MSA_TEST_DECODE_ONLY");
        const bool prefill = decode_only == nullptr || std::strcmp(decode_only, "1") != 0;
        if (prefill) {
            prefill_batch_cases(rng);
            prefill_attend4_cases(rng);
        }
        for (const Case& c : cases) run_case(c, rng);
        if (prefill)
            for (std::uint64_t T : {300ULL, 3000ULL}) prefill_case(T, rng, false);
        if (g_fail != 0) {
            std::fprintf(stderr, "%d failure(s)\n", g_fail);
            return 1;
        }
        std::printf("strix-mmattn-fused-tests: all passed\n");
    }
    const char* b = std::getenv("STRIX_MMATTN_BENCH");
    if (b != nullptr && std::strcmp(b, "1") == 0) {
        launch_probe();
        if (const char* e = std::getenv("STRIX_MMATTN_EXP"); e != nullptr && std::strcmp(e, "1") == 0) g_exp = true;
        std::vector<std::array<unsigned long long, 3>> cells;
        if (const char* sel = std::getenv("STRIX_MMATTN_BENCH_CELLS"); sel != nullptr) {
            std::string s(sel);
            std::size_t at = 0;
            while (at < s.size()) {
                const std::size_t end = s.find(',', at) == std::string::npos ? s.size() : s.find(',', at);
                unsigned long long cx = 0, ns = 0, w = 0;
                if (std::sscanf(s.substr(at, end - at).c_str(), "%llu:%llu:%llu", &cx, &ns, &w) == 3) cells.push_back({cx, ns, w});
                at = end + 1;
            }
        } else {
            for (unsigned long long cx : {1024ULL, 8192ULL, 32768ULL, 131072ULL}) {
                for (unsigned long long ns : {1ULL, 2ULL, 4ULL, 8ULL}) cells.push_back({cx, ns, 1});
                cells.push_back({cx, 1, 8});
            }
        }
        for (const auto& cell : cells) bench_cell(cell[0], static_cast<unsigned>(cell[1]), static_cast<unsigned>(cell[2]), rng);
        if (std::getenv("STRIX_MMATTN_BENCH_CELLS") == nullptr && !g_exp)
            for (std::uint64_t T : {1024ULL, 8192ULL}) prefill_case(T, rng, true);
    }
    return g_fail != 0 ? 1 : 0;
}
