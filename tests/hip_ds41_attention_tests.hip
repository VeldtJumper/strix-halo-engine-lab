// deepseek41 decode attention on gfx1151: every kernel against its host reference, bit for bit, on synthetic data
// (window ring and compressed caches written over many verify windows of 1..8 rows, index scores, top-k with the
// candidate pool, attention with window-only and compressed visible sets) and the compress kernel's fast form against
// its default, and the segment launches against the per-segment calls. STRIX_DS41_ATTN_BENCH=1 adds timings.
#include "strix/hip_ds41_attention.hpp"
#include "strix/hip_ds41_formats.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace a = strix::ds41::attn;
namespace g = strix::ds41::attn::hip;

namespace {
int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL %s\n", what.c_str()); }
}
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
template <class T> struct Dev {
    T* p = nullptr; std::size_t n = 0;
    explicit Dev(std::size_t count) : n(count) { check(hipMalloc(&p, std::max<std::size_t>(count, 1) * sizeof(T))); check(hipMemset(p, 0, std::max<std::size_t>(count, 1) * sizeof(T))); }
    Dev(const std::vector<T>& h) : Dev(h.size()) { put(h); }
    ~Dev() { (void)hipFree(p); }
    Dev(const Dev&) = delete;
    void put(const std::vector<T>& h) { check(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); }
    std::vector<T> get() const { std::vector<T> h(n); check(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }
};
std::mt19937 rng(4141);
float uni(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }
template <class T> bool same(const std::vector<T>& x, const std::vector<T>& y) {
    return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(T)) == 0;
}
std::vector<a::RopeRow> ropes(a::RopeKind k, const std::vector<std::uint64_t>& pos, unsigned ratio = 0) {
    std::vector<a::RopeRow> r;
    for (const auto p : pos) r.push_back(a::rope_row(k, ratio == 0 ? p : p / ratio * ratio));
    return r;
}
std::vector<std::uint16_t> random_f16(std::size_t n) {
    std::vector<std::uint16_t> v(n);
    for (auto& h : v) h = static_cast<std::uint16_t>(0x1C00U + (rng() & 0xFFFU) + ((rng() & 1U) << 15U));
    return v;
}
std::vector<float> random_f(std::size_t n, float lo, float hi) { std::vector<float> v(n); for (auto& x : v) x = uni(lo, hi); return v; }

// Window ring and compression over 0..npos-1 in verify windows of 1..8 rows; host and device caches equal.
void caches(unsigned ratio, unsigned npos, int compress_fast = 0, unsigned call_rows = 8) {
    g::set_compress_fast(compress_fast);
    const unsigned cap = a::kWindow + call_rows;
    const auto norm = random_f(512, 0.5F, 1.5F), cnorm = random_f(512, 0.5F, 1.5F), knorm = random_f(128, 0.5F, 1.5F);
    const auto wk = random_f16(128 * 512);
    std::vector<std::uint8_t> ring(cap * a::kWindowEntryBytes), kv(npos * a::kKvEntryBytes), ix(npos * a::kIdxEntryBytes);
    Dev<std::uint8_t> dring(ring.size()), dkv(kv.size()), dix(ix.size());
    Dev<float> dnorm(norm), dcnorm(cnorm), dknorm(knorm), dstate_a(1024), dstate_b(1024);
    Dev<std::uint16_t> dwk(wk);
    Dev<std::uint32_t> dvalid_a(1), dvalid_b(1);
    a::CompressState state;
    std::uint64_t p = 0;
    bool flip = false;
    for (unsigned step = 0; p < npos; ++step) {
        const unsigned rows = std::min<unsigned>(call_rows == 8U ? 1U + (step * 5U) % 8U : call_rows, static_cast<unsigned>(npos - p));
        std::vector<std::uint64_t> pos(rows);
        for (unsigned r = 0; r < rows; ++r) pos[r] = p + r;
        const auto kvp = random_f(rows * 512, -3, 3), sc = random_f(rows * 512, -4, 4);
        for (unsigned first = 0; first < rows; first += a::kMaxRows) {
            const unsigned count = std::min(a::kMaxRows, rows - first);
            const auto k = std::span<const float>(kvp).subspan(std::size_t{first} * 512U, count * 512U);
            const auto score = std::span<const float>(sc).subspan(std::size_t{first} * 512U, count * 512U);
            const auto positions = std::span<const std::uint64_t>(pos).subspan(first, count);
            a::window_write({k, norm, positions, cap, a::RopeKind::compressed}, ring);
            a::compress({ratio, k, score, cnorm, wk, knorm, positions}, state, kv, ix);
        }
        Dev<float> dkvp(kvp), dsc(sc);
        Dev<std::uint64_t> dpos(pos);
        Dev<a::RopeRow> drope(ropes(a::RopeKind::compressed, pos)), dcrope(ropes(a::RopeKind::compressed, pos, ratio));
        g::enqueue_window_write({dkvp.p, dnorm.p, drope.p, dpos.p, rows, cap, dring.p}, nullptr);
        auto& sin = flip ? dstate_b : dstate_a; auto& sout = flip ? dstate_a : dstate_b;
        auto& vin = flip ? dvalid_b : dvalid_a; auto& vout = flip ? dvalid_a : dvalid_b;
        g::enqueue_compress({ratio, rows, dkvp.p, dsc.p, dcnorm.p, dwk.p, dknorm.p, dcrope.p, dpos.p, sin.p, vin.p, sout.p, vout.p, dkv.p, dix.p}, nullptr);
        check(hipDeviceSynchronize());
        if (ratio == 2) {
            const auto vs = vout.get();
            expect((vs[0] != 0U) == state.valid, "compress state valid flag");
        }
        flip = !flip;
        p += rows;
    }
    expect(same(dring.get(), ring), "window ring bytes, ratio " + std::to_string(ratio));
    expect(same(dkv.get(), kv), "compressed KV bytes, ratio " + std::to_string(ratio));
    expect(same(dix.get(), ix), "index key bytes, ratio " + std::to_string(ratio));
    g::set_compress_fast(-1);
    std::printf("caches ratio %u, %u positions%s: ring/KV/index keys host == device\n", ratio, npos, compress_fast != 0 ? " (compress fast)" : "");
}

// Compress fast (set_compress_fast(1)) against the default kernel, bit for bit (KV and index caches, carry state and
// flag): ratio 1 and 2, rows 1..8, several first positions and carry flags; weights with F16 subnormals and signed
// zeros, and in one set infinities and NaNs (quiet, signalling, negative) with far-apart, infinite and NaN scores.
// STRIX_DS41_ATTN_BENCH=1 adds us per call at rows 1, 2, 6, 8: cold (a 256 MB read before each call, median of 15)
// and warm (back to back, the mean of 50).
__global__ void flush_kernel(const uint4* p, std::size_t n, unsigned* sink) {
    uint4 acc{0, 0, 0, 0};
    for (std::size_t i = blockIdx.x * 256ULL + threadIdx.x; i < n; i += 256ULL * gridDim.x) {
        const uint4 v = p[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if ((acc.x & acc.y & acc.z & acc.w) == 0xFFFFFFFFU) *sink = 1U;
}
template <class F> double cold_us(F call) {
    Dev<std::uint8_t> flush(std::size_t{256} << 20U);
    Dev<unsigned> sink(1);
    hipEvent_t e0, e1;
    check(hipEventCreate(&e0)); check(hipEventCreate(&e1));
    std::vector<double> t;
    for (int i = 0; i < 15; ++i) {
        hipLaunchKernelGGL(flush_kernel, dim3(1024), dim3(256), 0, nullptr, reinterpret_cast<const uint4*>(flush.p), flush.n / 16U, sink.p);
        check(hipEventRecord(e0)); call(); check(hipEventRecord(e1)); check(hipEventSynchronize(e1));
        float ms; check(hipEventElapsedTime(&ms, e0, e1)); t.push_back(1e3 * ms);
    }
    check(hipEventDestroy(e0)); check(hipEventDestroy(e1));
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}
template <class F> double warm_us(F call) {
    hipEvent_t e0, e1;
    check(hipEventCreate(&e0)); check(hipEventCreate(&e1));
    call();
    check(hipEventRecord(e0));
    for (int i = 0; i < 50; ++i) call();
    check(hipEventRecord(e1)); check(hipEventSynchronize(e1));
    float ms; check(hipEventElapsedTime(&ms, e0, e1));
    check(hipEventDestroy(e0)); check(hipEventDestroy(e1));
    return 1e3 * ms / 50.0;
}
template <class T> std::vector<std::uint8_t> bytes_of(const std::vector<T>& v) {
    std::vector<std::uint8_t> b(v.size() * sizeof(T));
    std::memcpy(b.data(), v.data(), b.size());
    return b;
}
void compress_fast(bool bench) {
    const auto norm = random_f(512, 0.5F, 1.5F), knorm = random_f(128, 0.5F, 1.5F);
    Dev<float> dnorm(norm), dknorm(knorm);
    auto wk = random_f16(128 * 512);
    for (std::size_t i = 0; i < wk.size(); i += 37) wk[i] = static_cast<std::uint16_t>((rng() & 0x3FFU) | ((rng() & 1U) << 15U));  // subnormals, +-0
    wk[5] = 0x0000U; wk[9] = 0x8000U; wk[700] = 0x0001U; wk[701] = 0x83FFU;
    auto wk_special = wk;   // rows 3, 40, 41, 77, 127: +inf, -inf, quiet NaN, signalling NaN, negative NaN
    wk_special[3 * 512 + 100] = 0x7C00U; wk_special[40 * 512 + 511] = 0xFC00U; wk_special[41 * 512 + 0] = 0x7E00U;
    wk_special[77 * 512 + 300] = 0x7C01U; wk_special[127 * 512 + 64] = 0xFE01U;
    Dev<std::uint16_t> dwk(wk), dwk_special(wk_special);
    unsigned cases = 0;
    for (const unsigned set : {0U, 1U})
        for (const unsigned ratio : {1U, 2U})
            for (std::uint32_t rows = 1; rows <= 8U; ++rows)
                for (const std::uint64_t first : {0ULL, 1ULL, 1000ULL, 1001ULL, 65535ULL, 500000ULL}) {
                    std::vector<std::uint64_t> pos(rows);
                    for (std::uint32_t r = 0; r < rows; ++r) pos[r] = first + r;
                    auto kvp = random_f(std::size_t{rows} * 512, -3.0F, 3.0F), sc = random_f(std::size_t{rows} * 512, -4.0F, 4.0F);
                    auto state = random_f(1024, -3.0F, 3.0F);
                    if (set == 1U) {   // exp underflow below -87, infinite and NaN scores
                        sc[3] = 95.0F; state[512 + 3] = -20.0F; sc[17] = -1e30F; state[512 + 17] = 0.0F;
                        sc[40] = std::numeric_limits<float>::infinity(); sc[41] = std::numeric_limits<float>::quiet_NaN();
                        if (rows > 1U) { sc[512 + 3] = -100.0F; sc[512 + 5] = 88.5F; sc[5] = -0.5F; }
                    }
                    const std::size_t entries = (first + rows) / ratio + 1U;
                    Dev<float> dkvp(kvp), dsc(sc), state_in(state), state_out(1024);
                    Dev<std::uint32_t> valid_in(std::vector<std::uint32_t>{static_cast<std::uint32_t>(first % 3U != 0U)}), valid_out(1);
                    Dev<std::uint64_t> dpos(pos);
                    Dev<a::RopeRow> drope(ropes(a::RopeKind::compressed, pos, ratio));
                    Dev<std::uint8_t> kvc(entries * a::kKvEntryBytes), idxc(entries * a::kIdxEntryBytes);
                    const g::CompressView v{ratio, rows, dkvp.p, dsc.p, dnorm.p, set == 1U ? dwk_special.p : dwk.p, dknorm.p, drope.p, dpos.p,
                                            state_in.p, valid_in.p, state_out.p, valid_out.p, kvc.p, idxc.p};
                    const auto run = [&](int mode) {
                        check(hipMemset(kvc.p, 0xA5, kvc.n)); check(hipMemset(idxc.p, 0xA5, idxc.n));
                        check(hipMemset(state_out.p, 0xFF, 4096)); check(hipMemset(valid_out.p, 0xFF, 4));
                        g::set_compress_fast(mode);
                        g::enqueue_compress(v, nullptr);
                        check(hipDeviceSynchronize());
                        return std::vector<std::vector<std::uint8_t>>{kvc.get(), idxc.get(), bytes_of(state_out.get()), bytes_of(valid_out.get())};
                    };
                    const auto base = run(0), got = run(1);
                    expect(base == got, "compress fast == default, set " + std::to_string(set) + " ratio " + std::to_string(ratio) + " rows " +
                                            std::to_string(rows) + " first " + std::to_string(first));
                    ++cases;
                    if (bench && set == 0U && first == 1001U && (rows == 1U || rows == 2U || rows == 6U || rows == 8U)) {
                        const auto call = [&] { g::enqueue_compress(v, nullptr); };
                        g::set_compress_fast(0);
                        const double c0 = cold_us(call), w0 = warm_us(call);
                        g::set_compress_fast(1);
                        const double c1 = cold_us(call), w1 = warm_us(call);
                        std::printf("  bench compress ratio %u R%u cold/warm us: default %.1f/%.1f -> fast %.1f/%.1f\n", ratio, rows, c0, w0, c1, w1);
                    }
                }
    g::set_compress_fast(-1);
    std::printf("compress fast == default kernel (KV, index keys, carry state), %u cases: ratio 1 and 2, rows 1..8, F16 specials\n", cases);
}

std::vector<std::uint8_t> random_idx_cache(std::size_t n) {
    std::vector<std::uint8_t> c(n * a::kIdxEntryBytes);
    for (std::size_t t = 0; t < n; ++t) {
        for (unsigned i = 0; i < 64; ++i) c[t * 68 + i] = static_cast<std::uint8_t>(rng());
        for (unsigned b = 0; b < 4; ++b) c[t * 68 + 64 + b] = static_cast<std::uint8_t>(122 + rng() % 6);
    }
    return c;
}

// Index scores and selection for rows ending at `last`, every pool mode; host == device.
void index_and_select(std::uint64_t last, unsigned rows, unsigned ratio, unsigned topk, unsigned cand, unsigned bs, bool bench) {
    std::vector<std::uint64_t> pos(rows);
    for (unsigned r = 0; r < rows; ++r) pos[r] = last - (rows - 1) + r;
    const std::uint32_t max_count = static_cast<std::uint32_t>((last + 1) / ratio);
    const auto cache = random_idx_cache(max_count);
    const auto q = random_f(rows * 32 * 128, -2, 2), w = random_f(rows * 32, -1, 1);
    std::vector<std::vector<float>> scores;
    for (unsigned first = 0; first < rows; first += a::kMaxRows) {
        const unsigned count = std::min(a::kMaxRows, rows - first);
        std::vector<std::vector<float>> part;
        a::index_scores({std::span<const float>(q).subspan(std::size_t{first}*32U*128U,count*32U*128U),
                         std::span<const float>(w).subspan(std::size_t{first}*32U,count*32U),
                         std::span<const std::uint64_t>(pos).subspan(first,count),ratio,a::RopeKind::compressed},cache,part);
        scores.insert(scores.end(),part.begin(),part.end());
    }
    Dev<std::uint8_t> dcache(cache);
    Dev<float> dq(q), dw(w), dscores(std::size_t{rows} * max_count);
    Dev<std::uint64_t> dpos(pos);
    Dev<a::RopeRow> drope(ropes(a::RopeKind::compressed, pos));
    Dev<std::uint8_t> dprep(std::size_t{rows} * g::kIdxPrepBytes);
    const g::IndexView iv{rows, ratio, max_count, dq.p, dw.p, drope.p, dpos.p, dcache.p, dscores.p, dprep.p};
    g::enqueue_index_scores(iv, nullptr);
    check(hipDeviceSynchronize());
    const auto ds = dscores.get();
    bool ok = true;
    for (unsigned r = 0; r < rows; ++r) ok = ok && std::memcmp(&ds[std::size_t{r} * max_count], scores[r].data(), scores[r].size() * 4) == 0;
    expect(ok, "index scores rows " + std::to_string(rows) + " count " + std::to_string(max_count));
    const std::uint32_t nb = (max_count + bs - 1) / bs;
    Dev<std::uint8_t> dcand(std::size_t{rows} * nb);
    Dev<std::uint32_t> dkeys(std::size_t{rows} * max_count), didx(std::size_t{rows} * topk), dcnt(rows);
    std::vector<std::vector<std::uint8_t>> cands(rows, std::vector<std::uint8_t>(nb, 0));
    for (const auto pool : {a::Pool::candidate_source, a::Pool::candidate_user, a::Pool::plain}) {
        const g::SelectView sv{rows, ratio, max_count, topk, cand, bs, pool, dpos.p, dscores.p, dcand.p, dkeys.p, didx.p, dcnt.p};
        g::enqueue_select(sv, nullptr);
        check(hipDeviceSynchronize());
        const auto di = didx.get(), dc = dcnt.get();
        const auto dm = dcand.get();
        bool sel_ok = true, cand_ok = true;
        for (unsigned r = 0; r < rows; ++r) {
            const auto mine = a::select_topk(scores[r], pool, cands[r], topk, cand, bs);
            sel_ok = sel_ok && dc[r] == mine.size() && std::equal(mine.begin(), mine.end(), &di[std::size_t{r} * topk]);
            if (pool == a::Pool::candidate_source) {
                const std::uint32_t nbr = (static_cast<std::uint32_t>(scores[r].size()) + bs - 1) / bs;
                cand_ok = cand_ok && std::equal(cands[r].begin(), cands[r].begin() + nbr, &dm[std::size_t{r} * nb]);
            }
        }
        const char* name = pool == a::Pool::plain ? "plain" : pool == a::Pool::candidate_source ? "source" : "user";
        expect(sel_ok, std::string("select ") + name + " count " + std::to_string(max_count));
        if (pool == a::Pool::candidate_source) expect(cand_ok, "candidate blocks count " + std::to_string(max_count));
    }
    std::printf("index scores + select: rows %u, count up to %u, topk %u, candidates %u x %u: host == device\n", rows, max_count, topk, cand, bs);
    if (!bench) return;
    hipEvent_t e0, e1;
    check(hipEventCreate(&e0)); check(hipEventCreate(&e1));
    const auto time = [&](auto f) {
        f(); check(hipDeviceSynchronize());
        std::vector<float> ms;
        for (int i = 0; i < 30; ++i) { check(hipEventRecord(e0)); f(); check(hipEventRecord(e1)); check(hipEventSynchronize(e1)); float t; check(hipEventElapsedTime(&t, e0, e1)); ms.push_back(t); }
        std::sort(ms.begin(), ms.end()); return ms[15];
    };
    const float ts = time([&] { g::enqueue_index_scores(iv, nullptr); });
    const g::SelectView sv{rows, ratio, max_count, topk, cand, bs, a::Pool::candidate_source, dpos.p, dscores.p, dcand.p, dkeys.p, didx.p, dcnt.p};
    const float tsel = time([&] { g::enqueue_select(sv, nullptr); });
    const double bytes = double(max_count) * a::kIdxEntryBytes;  // keys read once per call (rows share them via L2)
    std::printf("  bench count %u rows %u: index scores %.1f us (%.0f GB/s of index keys), select (source) %.1f us\n",
        max_count, rows, ts * 1e3, bytes / (ts * 1e-3) / 1e9, tsel * 1e3);
    check(hipEventDestroy(e0)); check(hipEventDestroy(e1));
}

// STRIX_DS41_IDX_CAND_FILTER and STRIX_DS41_IDX_DIRECT against today's path on the same inputs. Filter: the candidate
// Selection over scores with few distinct values (levels; 1 = every score equal), so long runs of equal keys meet the
// threshold and whole radix bins are taken: every pool mode, host == device.
void select_ties(std::uint32_t count, unsigned rows, unsigned levels, unsigned topk, unsigned cand, unsigned bs) {
    std::vector<std::uint64_t> pos(rows);
    for (unsigned r = 0; r < rows; ++r) pos[r] = count - rows + r;
    std::mt19937 gen(levels * 7919U + count);
    std::vector<std::vector<float>> scores(rows);
    std::vector<float> flat(std::size_t{rows} * count, 0.0F);
    for (unsigned r = 0; r < rows; ++r) {
        scores[r].resize(static_cast<std::size_t>(pos[r] + 1U));
        for (std::size_t t = 0; t < scores[r].size(); ++t) {
            scores[r][t] = static_cast<float>(gen() % levels) * 0.25F - 1.0F;
            flat[std::size_t{r} * count + t] = scores[r][t];
        }
    }
    Dev<float> dscores(flat);
    Dev<std::uint64_t> dpos(pos);
    const std::uint32_t nb = (count + bs - 1) / bs;
    Dev<std::uint8_t> dcand(std::size_t{rows} * nb);
    Dev<std::uint32_t> dkeys(std::size_t{rows} * count), didx(std::size_t{rows} * topk), dcnt(rows);
    std::vector<std::vector<std::uint8_t>> cands(rows, std::vector<std::uint8_t>(nb, 0));
    for (const auto pool : {a::Pool::candidate_source, a::Pool::candidate_user, a::Pool::plain}) {
        g::enqueue_select(g::SelectView{rows, 1, count, topk, cand, bs, pool, dpos.p, dscores.p, dcand.p, dkeys.p, didx.p, dcnt.p}, nullptr);
        check(hipDeviceSynchronize());
        const auto di = didx.get(), dc = dcnt.get();
        bool ok = true;
        for (unsigned r = 0; r < rows; ++r) {
            const auto mine = a::select_topk(scores[r], pool, cands[r], topk, cand, bs);
            ok = ok && dc[r] == mine.size() && std::equal(mine.begin(), mine.end(), &di[std::size_t{r} * topk]);
        }
        expect(ok, "select ties pool " + std::to_string(static_cast<int>(pool)) + " levels " + std::to_string(levels) + " count " + std::to_string(count));
    }
    std::printf("select with ties: count %u rows %u levels %u topk %u: host == device\n", count, rows, levels, topk);
}

// user's scores over a poisoned buffer (NaN bytes) with the source's pool, then select == the unfiltered selection, and
// every candidate position's score == the unfiltered one. Direct (count <= topk): idx, counts and pool bytes == select.
void index_filter_direct(std::uint64_t last, unsigned rows, unsigned ratio, unsigned topk, unsigned cand, unsigned bs, bool bench) {
    std::vector<std::uint64_t> pos(rows);
    for (unsigned r = 0; r < rows; ++r) pos[r] = last - (rows - 1) + r;
    const std::uint32_t max_count = static_cast<std::uint32_t>((last + 1) / ratio), nb = (max_count + bs - 1) / bs;
    const auto cache = random_idx_cache(max_count);
    const auto q = random_f(rows * 32 * 128, -2, 2), w = random_f(rows * 32, -1, 1);
    Dev<std::uint8_t> dcache(cache), dprep(std::size_t{rows} * g::kIdxPrepBytes), dcand(std::size_t{rows} * nb), dcand2(std::size_t{rows} * nb);
    Dev<float> dq(q), dw(w), dscores(std::size_t{rows} * max_count), dscores2(std::size_t{rows} * max_count);
    Dev<std::uint64_t> dpos(pos);
    Dev<a::RopeRow> drope(ropes(a::RopeKind::compressed, pos));
    Dev<std::uint32_t> dkeys(std::size_t{rows} * max_count), didx(std::size_t{rows} * topk), dcnt(rows), didx2(std::size_t{rows} * topk), dcnt2(rows);
    const auto poison = [&] {
        for (auto* d : {&didx, &dcnt, &didx2, &dcnt2, &dkeys}) check(hipMemset(d->p, 0xA5, d->n * 4));
        check(hipMemset(dscores2.p, 0xFF, dscores2.n * 4));
        check(hipMemset(dcand2.p, 0xA5, dcand2.n));
    };
    const g::IndexView iv{rows, ratio, max_count, dq.p, dw.p, drope.p, dpos.p, dcache.p, dscores.p, dprep.p};
    const auto sel = [&](a::Pool pool, float* scores, std::uint8_t* c, std::uint32_t* idx, std::uint32_t* cnt) {
        return g::SelectView{rows, ratio, max_count, topk, cand, bs, pool, dpos.p, scores, c, dkeys.p, idx, cnt};
    };
    poison();
    check(hipMemset(dcand.p, 0xA5, dcand.n));
    g::enqueue_index_scores(iv, nullptr);
    g::enqueue_select(sel(a::Pool::candidate_source, dscores.p, dcand.p, didx.p, dcnt.p), nullptr);   // publishes the pool
    g::enqueue_select(sel(a::Pool::candidate_user, dscores.p, dcand.p, didx.p, dcnt.p), nullptr);
    check(hipDeviceSynchronize());
    const auto ref_idx = didx.get(), ref_cnt = dcnt.get();
    const auto ref_scores = dscores.get();
    const auto pool_bytes = dcand.get();
    g::IndexView fv = iv;
    fv.scores = dscores2.p;
    fv.candidates = dcand.p;
    fv.block_size = bs;
    g::enqueue_index_scores(fv, nullptr);
    g::enqueue_select(sel(a::Pool::candidate_user, dscores2.p, dcand.p, didx2.p, dcnt2.p), nullptr);
    check(hipDeviceSynchronize());
    const auto fs = dscores2.get();
    bool scores_ok = true;
    std::size_t scored = 0;
    for (unsigned r = 0; r < rows; ++r) {
        const std::uint32_t n = static_cast<std::uint32_t>((pos[r] + 1) / ratio);
        for (std::uint32_t t = 0; t < n; ++t)
            if (pool_bytes[std::size_t{r} * nb + t / bs] != 0U) {
                ++scored;
                scores_ok = scores_ok && std::memcmp(&fs[std::size_t{r} * max_count + t], &ref_scores[std::size_t{r} * max_count + t], 4) == 0;
            }
    }
    expect(scores_ok, "filtered candidate scores == unfiltered, count " + std::to_string(max_count));
    expect(didx2.get() == ref_idx && dcnt2.get() == ref_cnt, "filtered user selection == unfiltered, count " + std::to_string(max_count));
    for (auto* d : {&didx2, &dcnt2, &dkeys}) check(hipMemset(d->p, 0xA5, d->n * 4));
    auto compact = sel(a::Pool::candidate_user, dscores2.p, dcand.p, didx2.p, dcnt2.p);
    compact.compact = true;
    g::enqueue_select(compact, nullptr);
    check(hipDeviceSynchronize());
    expect(didx2.get() == ref_idx && dcnt2.get() == ref_cnt, "compact user selection == unfiltered, count " + std::to_string(max_count));
    for (auto* d : {&didx2, &dcnt2, &dkeys}) check(hipMemset(d->p, 0xA5, d->n * 4));
    const std::vector<g::SelectView> compact_one{compact};
    g::enqueue_select_segments(compact_one, nullptr);
    check(hipDeviceSynchronize());
    expect(didx2.get() == ref_idx && dcnt2.get() == ref_cnt, "compact user selection (segments) == unfiltered, count " + std::to_string(max_count));
    if (max_count > 4096U) {   // a candidate score of -inf: the compact path must fall back to today's selection
        std::vector<float> poked = dscores2.get();
        for (std::uint32_t t = 0; t < max_count; ++t)
            if (pool_bytes[t / bs] != 0U) { poked[t] = -std::numeric_limits<float>::infinity(); break; }
        dscores2.put(poked);
        auto plain_user = compact;
        plain_user.compact = false;
        g::enqueue_select(plain_user, nullptr);
        check(hipDeviceSynchronize());
        const auto want_idx = didx2.get(), want_cnt = dcnt2.get();
        for (auto* d : {&didx2, &dcnt2, &dkeys}) check(hipMemset(d->p, 0xA5, d->n * 4));
        g::enqueue_select(compact, nullptr);
        check(hipDeviceSynchronize());
        expect(didx2.get() == want_idx && dcnt2.get() == want_cnt, "compact user selection with a -inf candidate == today, count " + std::to_string(max_count));
    }
    if (max_count <= topk) {
        for (const auto pool : {a::Pool::candidate_source, a::Pool::candidate_user, a::Pool::plain}) {
            poison();
            check(hipMemset(dcand.p, 0xA5, dcand.n));
            g::enqueue_index_scores(iv, nullptr);
            g::enqueue_select(sel(pool, dscores.p, dcand.p, didx.p, dcnt.p), nullptr);
            g::enqueue_select_direct(sel(pool, dscores2.p, dcand2.p, didx2.p, dcnt2.p), nullptr);
            check(hipDeviceSynchronize());
            expect(didx2.get() == didx.get() && dcnt2.get() == dcnt.get() && dcand2.get() == dcand.get(),
                   "direct selection == select, pool " + std::to_string(static_cast<int>(pool)) + " count " + std::to_string(max_count));
            check(hipMemset(didx2.p, 0xA5, didx2.n * 4)); check(hipMemset(dcnt2.p, 0xA5, dcnt2.n * 4)); check(hipMemset(dcand2.p, 0xA5, dcand2.n));
            const std::vector<g::SelectView> one{sel(pool, dscores2.p, dcand2.p, didx2.p, dcnt2.p)};
            g::enqueue_select_direct_segments(one, nullptr);
            check(hipDeviceSynchronize());
            expect(didx2.get() == didx.get() && dcnt2.get() == dcnt.get() && dcand2.get() == dcand.get(),
                   "direct selection (segments) == select, pool " + std::to_string(static_cast<int>(pool)));
        }
    }
    std::printf("index filter / direct: rows %u, count up to %u, topk %u, candidates %u x %u: %zu of %zu row positions scored, == today\n",
                rows, max_count, topk, cand, bs, scored, std::size_t{rows} * max_count);
    if (!bench) return;
    hipEvent_t e0, e1;
    check(hipEventCreate(&e0)); check(hipEventCreate(&e1));
    static Dev<std::uint8_t> sweep(std::size_t{512} << 20);
    const auto time = [&](auto f) {   // cold: index keys and scores evicted by a 512 MB write before each run
        std::vector<float> ms;
        for (int i = 0; i < 15; ++i) {
            check(hipMemsetAsync(sweep.p, i, sweep.n, nullptr));
            check(hipEventRecord(e0)); f(); check(hipEventRecord(e1)); check(hipEventSynchronize(e1));
            float t; check(hipEventElapsedTime(&t, e0, e1)); ms.push_back(t);
        }
        std::sort(ms.begin(), ms.end()); return ms[7];
    };
    const auto user = sel(a::Pool::candidate_user, dscores.p, dcand.p, didx.p, dcnt.p);
    const float t_off = time([&] { g::enqueue_index_scores(iv, nullptr); }), t_on = time([&] { g::enqueue_index_scores(fv, nullptr); });
    const float t_sel = time([&] { g::enqueue_select(user, nullptr); });
    auto user_compact = user;
    user_compact.compact = true;
    const float t_csel = time([&] { g::enqueue_select(user_compact, nullptr); });
    std::printf("  bench count %u rows %u (cold): user scores %.1f -> %.1f us filtered, user select %.1f -> %.1f us compact\n", max_count, rows,
                t_off * 1e3, t_on * 1e3, t_sel * 1e3, t_csel * 1e3);
    if (max_count <= topk) {
        const auto src = sel(a::Pool::candidate_source, dscores.p, dcand.p, didx.p, dcnt.p);
        const float t_today = time([&] { g::enqueue_index_scores(iv, nullptr); g::enqueue_select(src, nullptr); });
        const float t_direct = time([&] { g::enqueue_select_direct(src, nullptr); });
        std::printf("  bench count %u rows %u (cold): source scores + select %.1f us, direct %.1f us (projections excluded)\n", max_count, rows, t_today * 1e3, t_direct * 1e3);
    }
    check(hipEventDestroy(e0)); check(hipEventDestroy(e1));
}

// Attention with random caches: window-only and compressed visible sets, 1..8 rows.
unsigned ring_scale_lo = 118;   // window E8M0 scale bytes (1..4: f32-subnormal key values, the f16 tile's fallback)
void attention(std::uint64_t last, unsigned rows, unsigned nc, bool bench) {
    // compressed keys: 4096, or every position of a long context (selected indices up to the last position, ratio 1)
    const unsigned cap = a::kWindow + a::kMaxRows, ncache = last >= 65536U ? static_cast<unsigned>(last + 1U) : 4096U;
    std::vector<std::uint8_t> ring(cap * a::kWindowEntryBytes), kv(ncache * a::kKvEntryBytes);
    for (std::size_t s = 0; s < cap; ++s) {
        for (unsigned d = 0; d < 512; ++d) { std::uint8_t c; do c = static_cast<std::uint8_t>(rng()); while ((c & 0x7FU) == 0x7FU); ring[s * 528 + d] = c; }
        for (unsigned b = 0; b < 16; ++b) ring[s * 528 + 512 + b] = static_cast<std::uint8_t>(ring_scale_lo + rng() % 4);
    }
    for (std::size_t t = 0; t < ncache; ++t) {
        for (unsigned i = 0; i < 256; ++i) kv[t * 288 + i] = static_cast<std::uint8_t>(rng());
        for (unsigned g2 = 0; g2 < 32; ++g2) kv[t * 288 + 256 + g2] = static_cast<std::uint8_t>(0x30 + rng() % 16);
    }
    std::vector<std::uint64_t> pos(rows);
    for (unsigned r = 0; r < rows; ++r) pos[r] = last - (rows - 1) + r;
    const auto q = random_f(std::size_t{rows} * 64 * 512, -0.5F, 0.5F), sink = random_f(64, -2, 2);
    std::vector<std::uint32_t> idx(std::size_t{rows} * a::kTopK), cnt(rows, nc);
    for (unsigned r = 0; r < rows; ++r) {
        std::vector<std::uint32_t> pick(ncache);
        for (std::uint32_t i = 0; i < ncache; ++i) pick[i] = i;
        std::shuffle(pick.begin(), pick.end(), rng);
        std::sort(pick.begin(), pick.begin() + nc);
        std::copy(pick.begin(), pick.begin() + nc, &idx[std::size_t{r} * a::kTopK]);
    }
    std::vector<float> o(std::size_t{rows} * 64 * 512);
    a::attention({q, sink, pos, cap, a::RopeKind::compressed, idx, nc != 0 ? std::span<const std::uint32_t>(cnt) : std::span<const std::uint32_t>{}}, ring, kv, o);
    Dev<std::uint8_t> dring(ring), dkv(kv);
    Dev<float> dq(q), dsink(sink), dpart(std::size_t{rows} * 64 * a::kMaxChunks * 514), dout(o.size());
    Dev<std::uint64_t> dpos(pos);
    Dev<std::uint32_t> didx(idx), dcnt(cnt);
    Dev<a::RopeRow> drope(ropes(a::RopeKind::compressed, pos));
    const g::AttentionView av{rows, cap, a::kTopK, dq.p, dsink.p, drope.p, dpos.p, dring.p, dkv.p, didx.p, nc != 0 ? dcnt.p : nullptr, dpart.p, dout.p};
    g::enqueue_attention(av, nullptr);
    check(hipDeviceSynchronize());
    expect(same(dout.get(), o), "attention rows " + std::to_string(rows) + " pos " + std::to_string(last) + " compressed " + std::to_string(nc));
    // fused combine (tickets): twice, the second call relies on the tickets the first left at zero
    Dev<std::uint32_t> dtick(std::vector<std::uint32_t>(std::size_t{rows} * g::kAttnGroupCount, 0U));
    g::AttentionView avf = av;
    avf.tickets = dtick.p;
    for (int rep = 0; rep < 2; ++rep) {
        check(hipMemset(dout.p, 0xFF, o.size() * sizeof(float)));
        g::enqueue_attention(avf, nullptr);
        check(hipDeviceSynchronize());
        expect(same(dout.get(), o), "attention (fused combine) rows " + std::to_string(rows) + " pos " + std::to_string(last) + " compressed " + std::to_string(nc));
    }
    expect(dtick.get() == std::vector<std::uint32_t>(std::size_t{rows} * g::kAttnGroupCount, 0U), "attention tickets left at zero");
    {  // the combine's A8 of o: quantize_a8's bits (int8 per 32: delta = amax / 127, rint, clamp, exact sum)
        Dev<std::int8_t> dcodes(o.size());
        Dev<float> ddeltas(o.size() / 32U);
        Dev<std::int32_t> dsums(o.size() / 32U);
        g::AttentionView avq = avf;
        avq.codes = dcodes.p; avq.deltas = ddeltas.p; avq.sums = dsums.p;
        g::enqueue_attention(avq, nullptr);
        check(hipDeviceSynchronize());
        std::vector<std::int8_t> codes(o.size());
        std::vector<float> deltas(o.size() / 32U);
        std::vector<std::int32_t> sums(o.size() / 32U);
        for (std::size_t gi = 0; gi < deltas.size(); ++gi) {
            float amax = 0.0F;
            for (unsigned i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(o[gi * 32U + i]));
            const bool tiny = amax > 0.0F && amax < 0x1p-119F;
            const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
            int sum = 0;
            for (unsigned i = 0; i < 32; ++i) {
                const float rounded = tiny ? 0.0F : std::nearbyint(o[gi * 32U + i] / delta);
                const int c = static_cast<int>(std::min(127.0F, std::max(-127.0F, rounded)));
                codes[gi * 32U + i] = static_cast<std::int8_t>(c);
                sum += c;
            }
            deltas[gi] = delta;
            sums[gi] = sum;
        }
        expect(same(dout.get(), o) && same(dcodes.get(), codes) && same(ddeltas.get(), deltas) && same(dsums.get(), sums),
               "attention combine A8 = quantize_a8 of o, rows " + std::to_string(rows));
    }
    // the combine's A8 = the device enqueue_quantize_a8 of the o it stores, also with one NaN head (16 all-NaN groups):
    // codes and sums bitwise, deltas bitwise or both NaN
    for (const bool poison : {false, true}) {
        auto qn = q;
        if (poison) std::fill(qn.begin(), qn.begin() + 512, std::numeric_limits<float>::quiet_NaN());
        Dev<float> dqn(qn), dout2(o.size()), fdeltas(o.size() / 32U), qdeltas(o.size() / 32U);
        Dev<std::int8_t> fcodes(o.size()), qcodes(o.size());
        Dev<std::int32_t> fsums(o.size() / 32U), qsums(o.size() / 32U);
        g::AttentionView avn = avf;
        avn.q = dqn.p; avn.o = dout2.p; avn.codes = fcodes.p; avn.deltas = fdeltas.p; avn.sums = fsums.p;
        g::enqueue_attention(avn, nullptr);
        strix::ds41::enqueue_quantize_a8(dout2.p, 64U * 512U, rows, qcodes.p, qdeltas.p, qsums.p, nullptr);
        check(hipDeviceSynchronize());
        const auto fd = fdeltas.get(), qd = qdeltas.get();
        bool deltas_same = fd.size() == qd.size();
        for (std::size_t i = 0; deltas_same && i < fd.size(); ++i)
            deltas_same = std::isnan(fd[i]) ? std::isnan(qd[i]) : std::bit_cast<std::uint32_t>(fd[i]) == std::bit_cast<std::uint32_t>(qd[i]);
        expect(same(fcodes.get(), qcodes.get()) && same(fsums.get(), qsums.get()) && deltas_same,
               std::string("attention combine A8 = device quantize_a8 of o") + (poison ? " (NaN head)" : "") + ", rows " + std::to_string(rows));
    }
    // tensor-parallel rank 1: heads 32..63 only (q, sink and o of those heads, contiguous)
    {
        std::vector<float> qh(std::size_t{rows} * 32 * 512), oh(qh.size());
        for (unsigned r = 0; r < rows; ++r) {
            std::copy_n(&q[(std::size_t{r} * 64 + 32) * 512], 32 * 512, &qh[std::size_t{r} * 32 * 512]);
            std::copy_n(&o[(std::size_t{r} * 64 + 32) * 512], 32 * 512, &oh[std::size_t{r} * 32 * 512]);
        }
        Dev<float> dqh(qh), douth(oh.size());
        g::AttentionView avh = avf;
        avh.q = dqh.p; avh.sink = dsink.p + 32; avh.o = douth.p; avh.heads = 32;
        g::enqueue_attention(avh, nullptr);
        check(hipDeviceSynchronize());
        expect(same(douth.get(), oh), "attention heads 32..63 rows " + std::to_string(rows));
    }
    std::printf("attention rows %u last pos %llu compressed %u: host == device (combine kernel and fused)\n", rows, static_cast<unsigned long long>(last), nc);
    if (!bench) return;
    hipEvent_t e0, e1;
    check(hipEventCreate(&e0)); check(hipEventCreate(&e1));
    g::enqueue_attention(avf, nullptr); check(hipDeviceSynchronize());
    std::vector<float> ms;
    for (int i = 0; i < 30; ++i) { check(hipEventRecord(e0)); g::enqueue_attention(avf, nullptr); check(hipEventRecord(e1)); check(hipEventSynchronize(e1)); float t; check(hipEventElapsedTime(&t, e0, e1)); ms.push_back(t); }
    std::sort(ms.begin(), ms.end());
    const unsigned nw = static_cast<unsigned>(std::min<std::uint64_t>(last + 1, a::kWindow));
    const double bytes = double(rows) * (nw * double(a::kWindowEntryBytes) + nc * double(a::kKvEntryBytes)) + double(rows) * 64 * 512 * 4 * 2;
    std::printf("  bench: %.1f us per call (%.0f GB/s of keys + q + o)\n", ms[15] * 1e3, bytes / (ms[15] * 1e-3) / 1e9);
    check(hipEventDestroy(e0)); check(hipEventDestroy(e1));
}

// Segment launches (enqueue_*_segments, BatchForward STRIX_DS41_BATCH_SEGMENTS=1) against the per-segment calls, bit
// for bit: segments of different rows, positions, ratios and selection paths (one block per row and two-level), every
// compress form (default, ahead, fast) and pool, attention with the combine kernel and with tickets (+ A8, 32 heads);
// every output buffer starts from the same pattern in both runs. More segments than kMaxSegments split the launches.
struct Seg { std::uint64_t last; unsigned rows, ratio; };
void segments() {
    const std::vector<Seg> segs = {{5, 1, 1}, {1000, 3, 2}, {20007, 8, 1}, {4600, 2, 1}, {9001, 1, 2}, {130, 4, 1},
                                   {64, 1, 2}, {3001, 2, 1}, {700, 5, 2}, {41, 1, 1}};   // 10 > kMaxSegments
    const unsigned cap = a::kWindow + a::kMaxRows;
    const auto norm = random_f(512, 0.5F, 1.5F), knorm = random_f(128, 0.5F, 1.5F);
    const auto wk = random_f16(128 * 512);
    Dev<float> dnorm(norm), dknorm(knorm);
    Dev<std::uint16_t> dwk(wk);
    struct Bufs {
        std::vector<std::uint64_t> pos;
        std::unique_ptr<Dev<std::uint64_t>> dpos;
        std::unique_ptr<Dev<a::RopeRow>> drope, dcrope;
        std::unique_ptr<Dev<float>> kv, sc, sin, sout, q, w, scores, aq, part, out, deltas;
        std::unique_ptr<Dev<std::uint32_t>> vin, vout, keys, idx, cnt, tick, aidx, acnt;
        std::unique_ptr<Dev<std::uint8_t>> ring, kvc, ixc, prep, cand, akv, aring;
        std::unique_ptr<Dev<std::int8_t>> codes;
        std::unique_ptr<Dev<std::int32_t>> sums;
        std::uint32_t max_count = 0;
    };
    std::vector<Bufs> b(segs.size());
    for (std::size_t k = 0; k < segs.size(); ++k) {
        const auto& s = segs[k];
        auto& x = b[k];
        for (unsigned r = 0; r < s.rows; ++r) x.pos.push_back(s.last - (s.rows - 1) + r);
        x.dpos = std::make_unique<Dev<std::uint64_t>>(x.pos);
        x.drope = std::make_unique<Dev<a::RopeRow>>(ropes(a::RopeKind::compressed, x.pos));
        x.dcrope = std::make_unique<Dev<a::RopeRow>>(ropes(a::RopeKind::compressed, x.pos, s.ratio));
        x.kv = std::make_unique<Dev<float>>(random_f(std::size_t{s.rows} * 512, -3, 3));
        x.sc = std::make_unique<Dev<float>>(random_f(std::size_t{s.rows} * 512, -4, 4));
        x.sin = std::make_unique<Dev<float>>(random_f(1024, -3, 3));
        x.sout = std::make_unique<Dev<float>>(1024);
        x.vin = std::make_unique<Dev<std::uint32_t>>(std::vector<std::uint32_t>{static_cast<std::uint32_t>(k % 2U)});
        x.vout = std::make_unique<Dev<std::uint32_t>>(1);
        x.ring = std::make_unique<Dev<std::uint8_t>>(std::size_t{cap} * a::kWindowEntryBytes);
        const std::size_t entries = (s.last + 1U) / s.ratio + 1U;
        x.kvc = std::make_unique<Dev<std::uint8_t>>(entries * a::kKvEntryBytes);
        x.max_count = static_cast<std::uint32_t>((s.last + 1U) / s.ratio);
        x.ixc = std::make_unique<Dev<std::uint8_t>>(random_idx_cache(x.max_count + 1U));
        x.q = std::make_unique<Dev<float>>(random_f(std::size_t{s.rows} * 32 * 128, -2, 2));
        x.w = std::make_unique<Dev<float>>(random_f(std::size_t{s.rows} * 32, -1, 1));
        x.scores = std::make_unique<Dev<float>>(std::size_t{s.rows} * x.max_count);
        x.prep = std::make_unique<Dev<std::uint8_t>>(std::size_t{s.rows} * g::kIdxPrepBytes);
        const std::uint32_t nb = (x.max_count + a::kCandBlockSize - 1) / a::kCandBlockSize;
        x.cand = std::make_unique<Dev<std::uint8_t>>(std::size_t{s.rows} * nb);
        x.keys = std::make_unique<Dev<std::uint32_t>>(std::size_t{s.rows} * x.max_count);
        x.idx = std::make_unique<Dev<std::uint32_t>>(std::size_t{s.rows} * a::kTopK);
        x.cnt = std::make_unique<Dev<std::uint32_t>>(s.rows);
        // attention inputs: random ring and compressed keys, selected indices below the cache size
        std::vector<std::uint8_t> ring(std::size_t{cap} * a::kWindowEntryBytes), kv(4096U * a::kKvEntryBytes);
        for (std::size_t e = 0; e < cap; ++e) {
            for (unsigned d = 0; d < 512; ++d) { std::uint8_t c; do c = static_cast<std::uint8_t>(rng()); while ((c & 0x7FU) == 0x7FU); ring[e * 528 + d] = c; }
            for (unsigned c = 0; c < 16; ++c) ring[e * 528 + 512 + c] = static_cast<std::uint8_t>(118 + rng() % 4);
        }
        for (std::size_t t = 0; t < 4096U; ++t) {
            for (unsigned i = 0; i < 256; ++i) kv[t * 288 + i] = static_cast<std::uint8_t>(rng());
            for (unsigned c = 0; c < 32; ++c) kv[t * 288 + 256 + c] = static_cast<std::uint8_t>(0x30 + rng() % 16);
        }
        const unsigned nc = std::min<unsigned>(a::kTopK, static_cast<unsigned>(k * 97U % 600U));
        std::vector<std::uint32_t> idx(std::size_t{s.rows} * a::kTopK), cnt(s.rows, nc);
        for (unsigned r = 0; r < s.rows; ++r) {
            std::vector<std::uint32_t> pick(4096);
            for (std::uint32_t i = 0; i < 4096U; ++i) pick[i] = i;
            std::shuffle(pick.begin(), pick.end(), rng);
            std::sort(pick.begin(), pick.begin() + nc);
            std::copy(pick.begin(), pick.begin() + nc, &idx[std::size_t{r} * a::kTopK]);
        }
        x.aring = std::make_unique<Dev<std::uint8_t>>(ring);
        x.akv = std::make_unique<Dev<std::uint8_t>>(kv);
        x.aidx = std::make_unique<Dev<std::uint32_t>>(idx);
        x.acnt = std::make_unique<Dev<std::uint32_t>>(cnt);
        x.aq = std::make_unique<Dev<float>>(random_f(std::size_t{s.rows} * 64 * 512, -0.5F, 0.5F));
        x.part = std::make_unique<Dev<float>>(std::size_t{s.rows} * 64 * a::kMaxChunks * 514);
        x.out = std::make_unique<Dev<float>>(std::size_t{s.rows} * 64 * 512);
        x.tick = std::make_unique<Dev<std::uint32_t>>(std::size_t{s.rows} * g::kAttnGroupCount);
        x.codes = std::make_unique<Dev<std::int8_t>>(std::size_t{s.rows} * 64 * 512);
        x.deltas = std::make_unique<Dev<float>>(std::size_t{s.rows} * 64 * 16);
        x.sums = std::make_unique<Dev<std::int32_t>>(std::size_t{s.rows} * 64 * 16);
    }
    const auto sink = random_f(64, -2, 2);
    Dev<float> dsink(sink);
    const auto reset = [&] {   // every output to the same pattern
        for (auto& x : b) {
            for (auto* d : {x.ring.get(), x.kvc.get(), x.cand.get()}) check(hipMemset(d->p, 0xA5, d->n));
            for (auto* d : {x.sout.get(), x.scores.get(), x.out.get(), x.deltas.get(), x.part.get()}) check(hipMemset(d->p, 0xA5, d->n * 4));
            for (auto* d : {x.vout.get(), x.keys.get(), x.idx.get(), x.cnt.get()}) check(hipMemset(d->p, 0xA5, d->n * 4));
            check(hipMemset(x.codes->p, 0xA5, x.codes->n)); check(hipMemset(x.sums->p, 0xA5, x.sums->n * 4));
            check(hipMemset(x.prep->p, 0xA5, x.prep->n));
            check(hipMemset(x.tick->p, 0, x.tick->n * 4));
        }
        check(hipDeviceSynchronize());
    };
    const auto snapshot = [&] {
        std::vector<std::vector<std::uint8_t>> out;
        for (auto& x : b) {
            out.push_back(x.ring->get()); out.push_back(x.kvc->get()); out.push_back(x.ixc->get()); out.push_back(x.cand->get());
            out.push_back(bytes_of(x.sout->get())); out.push_back(bytes_of(x.vout->get())); out.push_back(bytes_of(x.scores->get()));
            out.push_back(bytes_of(x.idx->get())); out.push_back(bytes_of(x.cnt->get())); out.push_back(bytes_of(x.out->get()));
            out.push_back(bytes_of(x.codes->get())); out.push_back(bytes_of(x.deltas->get())); out.push_back(bytes_of(x.sums->get()));
            out.push_back(bytes_of(x.tick->get()));
        }
        return out;
    };
    unsigned cases = 0;
    for (const int seg_hpw : {0, 1})     // STRIX_DS41_ATTN_SEG_HPW: one heads-per-wave form for all segments
    for (const int form : {0, 1, 2}) {   // compress default, ahead, fast
        g::set_attention_segment_hpw(seg_hpw);
        g::set_compress_ahead(form == 1 ? 1 : 0);
        g::set_compress_fast(form == 2 ? 1 : 0);
        for (const auto pool : {a::Pool::candidate_source, a::Pool::candidate_user, a::Pool::plain})
            for (const int att : {0, 1, 2, 3}) {   // 0 combine kernel, 1 tickets, 2 tickets + A8, 3 tickets, 32 heads
                std::vector<g::WindowView> wv;
                std::vector<g::CompressView> cv;
                std::vector<g::IndexView> iv;
                std::vector<g::SelectView> sv;
                std::vector<g::AttentionView> av;
                for (std::size_t k = 0; k < segs.size(); ++k) {
                    const auto& s = segs[k];
                    auto& x = b[k];
                    wv.push_back({x.kv->p, dnorm.p, x.drope->p, x.dpos->p, s.rows, cap, x.ring->p});
                    cv.push_back({s.ratio, s.rows, x.kv->p, x.sc->p, dnorm.p, dwk.p, dknorm.p, x.dcrope->p, x.dpos->p, x.sin->p, x.vin->p,
                                  x.sout->p, x.vout->p, x.kvc->p, x.ixc->p});
                    iv.push_back({s.rows, s.ratio, x.max_count, x.q->p, x.w->p, x.drope->p, x.dpos->p, x.ixc->p, x.scores->p, x.prep->p});
                    sv.push_back({s.rows, s.ratio, x.max_count, a::kTopK, a::kCandBlocks, a::kCandBlockSize, pool, x.dpos->p, x.scores->p,
                                  x.cand->p, x.keys->p, x.idx->p, x.cnt->p});
                    g::AttentionView v{s.rows, cap, a::kTopK, x.aq->p, dsink.p, x.drope->p, x.dpos->p, x.aring->p, x.akv->p, x.aidx->p,
                                       k % 3U == 0U ? nullptr : x.acnt->p, x.part->p, x.out->p};
                    if (att >= 1) v.tickets = x.tick->p;
                    if (att == 2) { v.codes = x.codes->p; v.deltas = x.deltas->p; v.sums = x.sums->p; }
                    if (att == 3) { v.heads = 32; v.sink = dsink.p + 32; }
                    av.push_back(v);
                }
                // the compress writes index keys that the scores then read: both runs restore them first
                std::vector<std::vector<std::uint8_t>> ixc0;
                for (auto& x : b) ixc0.push_back(x.ixc->get());
                const auto run = [&](bool seg) {
                    reset();
                    for (std::size_t k = 0; k < b.size(); ++k) b[k].ixc->put(ixc0[k]);
                    if (seg) {
                        g::enqueue_window_write_segments(wv, nullptr);
                        g::enqueue_compress_segments(cv, nullptr);
                        g::enqueue_index_scores_segments(iv, nullptr);
                        g::enqueue_select_segments(sv, nullptr);
                        g::enqueue_attention_segments(av, nullptr);
                    } else {
                        for (const auto& v : wv) g::enqueue_window_write(v, nullptr);
                        for (const auto& v : cv) g::enqueue_compress(v, nullptr);
                        for (const auto& v : iv) g::enqueue_index_scores(v, nullptr);
                        for (const auto& v : sv) g::enqueue_select(v, nullptr);
                        for (const auto& v : av) g::enqueue_attention(v, nullptr);
                    }
                    check(hipDeviceSynchronize());
                    return snapshot();
                };
                const auto loop = run(false), seg = run(true);
                expect(loop == seg, "segments == per-segment calls, seg hpw " + std::to_string(seg_hpw) + " compress form " + std::to_string(form) + " pool " +
                                        std::to_string(static_cast<int>(pool)) + " attention arm " + std::to_string(att));
                ++cases;
            }
    }
    g::set_compress_ahead(-1);
    g::set_compress_fast(-1);
    g::set_attention_segment_hpw(-1);
    std::printf("segment launches == per-segment calls (window, compress, index, select, attention), %zu segments, %u cases\n", segs.size(), cases);
}

// STRIX_DS41_ATTN_SEG_HPW on the N8 TP decode shape: `segs` requests of `rows` rows, `heads` heads each, a full visible
// set (window 128 + top-k 512 at position 5000), one segment launch; outputs == per-call, then warm median times.
void segment_hpw_bench(unsigned segs, unsigned rows, unsigned heads) {
    const unsigned cap = a::kWindow + a::kMaxRows;
    struct One {
        std::unique_ptr<Dev<std::uint8_t>> ring, kv;
        std::unique_ptr<Dev<std::uint32_t>> idx, cnt, tick;
        std::unique_ptr<Dev<float>> q, part, out;
        std::unique_ptr<Dev<std::uint64_t>> pos;
        std::unique_ptr<Dev<a::RopeRow>> rope;
    };
    std::vector<One> x(segs);
    const auto sink = random_f(heads, -2, 2);
    Dev<float> dsink(sink);
    std::vector<g::AttentionView> views;
    for (unsigned k = 0; k < segs; ++k) {
        std::vector<std::uint64_t> pos;
        for (unsigned r = 0; r < rows; ++r) pos.push_back(5000U + 37U * k + r);
        std::vector<std::uint8_t> ring(std::size_t{cap} * a::kWindowEntryBytes), kv(4096U * a::kKvEntryBytes);
        for (std::size_t e = 0; e < cap; ++e) {
            for (unsigned d = 0; d < 512; ++d) { std::uint8_t c; do c = static_cast<std::uint8_t>(rng()); while ((c & 0x7FU) == 0x7FU); ring[e * 528 + d] = c; }
            for (unsigned c = 0; c < 16; ++c) ring[e * 528 + 512 + c] = static_cast<std::uint8_t>(118 + rng() % 4);
        }
        for (std::size_t t = 0; t < 4096U; ++t) {
            for (unsigned i = 0; i < 256; ++i) kv[t * 288 + i] = static_cast<std::uint8_t>(rng());
            for (unsigned c = 0; c < 32; ++c) kv[t * 288 + 256 + c] = static_cast<std::uint8_t>(0x30 + rng() % 16);
        }
        std::vector<std::uint32_t> idx(std::size_t{rows} * a::kTopK), cnt(rows, a::kTopK);
        for (unsigned r = 0; r < rows; ++r)
            for (unsigned i = 0; i < a::kTopK; ++i) idx[std::size_t{r} * a::kTopK + i] = 8U * i + (rng() % 8U);
        auto& o = x[k];
        o.ring = std::make_unique<Dev<std::uint8_t>>(ring);
        o.kv = std::make_unique<Dev<std::uint8_t>>(kv);
        o.idx = std::make_unique<Dev<std::uint32_t>>(idx);
        o.cnt = std::make_unique<Dev<std::uint32_t>>(cnt);
        o.tick = std::make_unique<Dev<std::uint32_t>>(std::size_t{rows} * g::kAttnGroupCount);
        o.q = std::make_unique<Dev<float>>(random_f(std::size_t{rows} * heads * 512, -0.5F, 0.5F));
        o.part = std::make_unique<Dev<float>>(std::size_t{rows} * heads * a::kMaxChunks * 514);
        o.out = std::make_unique<Dev<float>>(std::size_t{rows} * heads * 512);
        o.pos = std::make_unique<Dev<std::uint64_t>>(pos);
        o.rope = std::make_unique<Dev<a::RopeRow>>(ropes(a::RopeKind::compressed, pos));
        g::AttentionView v{rows, cap, a::kTopK, o.q->p, dsink.p, o.rope->p, o.pos->p, o.ring->p, o.kv->p, o.idx->p, o.cnt->p, o.part->p, o.out->p};
        v.tickets = o.tick->p;
        v.heads = heads;
        views.push_back(v);
    }
    const auto outputs = [&] { std::vector<float> all; for (auto& o : x) { const auto v = o.out->get(); all.insert(all.end(), v.begin(), v.end()); } return all; };
    for (const auto& v : views) g::enqueue_attention(v, nullptr);
    check(hipDeviceSynchronize());
    const auto ref = outputs();
    hipEvent_t e0, e1;
    check(hipEventCreate(&e0)); check(hipEventCreate(&e1));
    float t[2];
    for (const int mode : {0, 1}) {
        g::set_attention_segment_hpw(mode);
        for (auto& o : x) check(hipMemset(o.out->p, 0xA5, o.out->n * 4));
        g::enqueue_attention_segments(views, nullptr);
        check(hipDeviceSynchronize());
        expect(outputs() == ref, "segment hpw " + std::to_string(mode) + " == per-call attention, " + std::to_string(segs) + " x " + std::to_string(rows) + " rows");
        std::vector<float> ms;
        for (int i = 0; i < 200; ++i) {
            check(hipEventRecord(e0)); g::enqueue_attention_segments(views, nullptr); check(hipEventRecord(e1)); check(hipEventSynchronize(e1));
            float m; check(hipEventElapsedTime(&m, e0, e1)); ms.push_back(m);
        }
        std::sort(ms.begin(), ms.end());
        t[mode] = ms[100];
    }
    g::set_attention_segment_hpw(-1);
    std::printf("  bench attention segments %u x %u rows, %u heads, 640 keys: per-segment hpw %.1f us, shared hpw %.1f us\n", segs, rows, heads,
                t[0] * 1e3, t[1] * 1e3);
    check(hipEventDestroy(e0)); check(hipEventDestroy(e1));
}

// STRIX_DS41_ATTN_WINDOW_FOLD: window write + attention (today) vs attention alone with the fold, the call's ring slots
// poisoned first: ring bytes and every output equal; `heads` per call, `segs` requests in one segment launch.
bool fold_live = false;   // window_fold's fold arm also passes max_visible (STRIX_DS41_ATTN_LIVE_CHUNKS)
// kv mode 1: adversarial rows (zero row, tiny, huge, one dominant element, values at E4M3 rounding ties); segments get
// rows - k % rows rows each (heterogeneous); kind: the layer's RoPE table (window or compressed).
void window_fold(std::uint64_t last, unsigned rows, unsigned nc, unsigned heads, unsigned segs, a::RopeKind kind = a::RopeKind::compressed,
                 int kv_mode = 0) {
    const unsigned cap = a::kWindow + a::kMaxRows;
    const auto norm = random_f(512, 0.5F, 1.5F);
    Dev<float> dnorm(norm);
    struct One {
        std::vector<std::uint64_t> pos;
        std::unique_ptr<Dev<std::uint8_t>> ring, kvc;
        std::unique_ptr<Dev<std::uint32_t>> idx, cnt, tick;
        std::unique_ptr<Dev<float>> q, part, out, kv;
        std::unique_ptr<Dev<std::uint64_t>> dpos;
        std::unique_ptr<Dev<a::RopeRow>> rope;
        std::vector<std::uint8_t> ring0;
    };
    std::vector<One> x(segs);
    const auto sink = random_f(heads, -2, 2);
    Dev<float> dsink(sink);
    for (unsigned k = 0; k < segs; ++k) {
        auto& o = x[k];
        const unsigned rk = segs > 1U ? rows - k % rows : rows;
        for (unsigned r = 0; r < rk; ++r) o.pos.push_back(last + 3U * k - (rk - 1) + r);
        o.ring0.resize(std::size_t{cap} * a::kWindowEntryBytes);
        for (std::size_t e = 0; e < cap; ++e) {
            for (unsigned d = 0; d < 512; ++d) { std::uint8_t c; do c = static_cast<std::uint8_t>(rng()); while ((c & 0x7FU) == 0x7FU); o.ring0[e * 528 + d] = c; }
            for (unsigned c = 0; c < 16; ++c) o.ring0[e * 528 + 512 + c] = static_cast<std::uint8_t>(118 + rng() % 4);
        }
        std::vector<std::uint8_t> kv(4096U * a::kKvEntryBytes);
        for (std::size_t t = 0; t < 4096U; ++t) {
            for (unsigned i = 0; i < 256; ++i) kv[t * 288 + i] = static_cast<std::uint8_t>(rng());
            for (unsigned c = 0; c < 32; ++c) kv[t * 288 + 256 + c] = static_cast<std::uint8_t>(0x30 + rng() % 16);
        }
        std::vector<std::uint32_t> idx(std::size_t{rk} * a::kTopK), cnt(rk, nc);
        for (unsigned r = 0; r < rk; ++r)
            for (unsigned i = 0; i < nc; ++i) idx[std::size_t{r} * a::kTopK + i] = 7U * i + (rng() % 7U);
        o.ring = std::make_unique<Dev<std::uint8_t>>(o.ring0);
        o.kvc = std::make_unique<Dev<std::uint8_t>>(kv);
        o.idx = std::make_unique<Dev<std::uint32_t>>(idx);
        o.cnt = std::make_unique<Dev<std::uint32_t>>(cnt);
        o.tick = std::make_unique<Dev<std::uint32_t>>(std::size_t{rk} * g::kAttnGroupCount);
        o.q = std::make_unique<Dev<float>>(random_f(std::size_t{rk} * heads * 512, -0.5F, 0.5F));
        auto kvrows = random_f(std::size_t{rk} * 512, -3, 3);
        if (kv_mode == 1)
            for (unsigned r = 0; r < rk; ++r) {
                float* row = &kvrows[std::size_t{r} * 512];
                switch ((r + k) % 5U) {
                case 0: for (unsigned d = 0; d < 512; ++d) row[d] = 0.0F; break;                       // zero row
                case 1: for (unsigned d = 0; d < 512; ++d) row[d] *= 1e-30F; break;                     // tiny
                case 2: for (unsigned d = 0; d < 512; ++d) row[d] *= 1e18F; break;                      // huge (sum of squares overflows)
                case 3: row[(7U * r + 3U) % 512U] = 1e6F; break;                                        // one dominant element
                default: for (unsigned d = 0; d < 512; ++d) row[d] = static_cast<float>(d % 17U) * 0.0625F - 0.5F; break;  // ties
                }
            }
        o.kv = std::make_unique<Dev<float>>(kvrows);
        o.part = std::make_unique<Dev<float>>(std::size_t{rk} * heads * a::kMaxChunks * 514);
        o.out = std::make_unique<Dev<float>>(std::size_t{rk} * heads * 512);
        o.dpos = std::make_unique<Dev<std::uint64_t>>(o.pos);
        o.rope = std::make_unique<Dev<a::RopeRow>>(ropes(kind, o.pos));
    }
    const auto run = [&](bool fold) {
        std::vector<g::WindowView> wv;
        std::vector<g::AttentionView> av;
        for (auto& o : x) {
            o.ring->put(o.ring0);
            if (fold) {   // poison the call's slots: the fold must write them
                std::vector<std::uint8_t> r = o.ring0;
                for (const auto p : o.pos) std::memset(&r[(p % cap) * a::kWindowEntryBytes], 0x5A, a::kWindowEntryBytes);
                o.ring->put(r);
            }
            check(hipMemset(o.out->p, 0xA5, o.out->n * 4));
            const unsigned rk = static_cast<unsigned>(o.pos.size());
            wv.push_back({o.kv->p, dnorm.p, o.rope->p, o.dpos->p, rk, cap, o.ring->p});
            g::AttentionView v{rk, cap, a::kTopK, o.q->p, dsink.p, o.rope->p, o.dpos->p, o.ring->p, o.kvc->p, o.idx->p,
                               nc != 0U ? o.cnt->p : nullptr, o.part->p, o.out->p, o.tick->p};
            v.heads = heads;
            if (fold) { v.kv = o.kv->p; v.kv_norm = dnorm.p; v.ring_out = o.ring->p; }
            if (fold && fold_live) v.max_visible = static_cast<std::uint32_t>(std::min<std::uint64_t>(o.pos.back() + 1, a::kWindow) + nc);
            av.push_back(v);
        }
        if (segs > 1U) {
            if (!fold) g::enqueue_window_write_segments(wv, nullptr);
            g::enqueue_attention_segments(av, nullptr);
        } else {
            if (!fold) g::enqueue_window_write(wv[0], nullptr);
            g::enqueue_attention(av[0], nullptr);
        }
        check(hipDeviceSynchronize());
        std::vector<std::vector<std::uint8_t>> res;
        for (auto& o : x) { res.push_back(o.ring->get()); res.push_back(bytes_of(o.out->get())); }
        return res;
    };
    const auto today = run(false), folded = run(true);
    const std::string tag = "last " + std::to_string(last) + " rows " + std::to_string(rows) + " compressed " + std::to_string(nc) + " heads " +
                            std::to_string(heads) + " segments " + std::to_string(segs) + (kind == a::RopeKind::window ? " window RoPE" : "") +
                            (kv_mode == 1 ? " adversarial kv" : "") + (fold_live ? " live chunks" : "");
    expect(today == folded, "window fold == window write + attention, " + tag);
    std::printf("window fold: %s: ring bytes and outputs == window write + attention\n", tag.c_str());
}

// STRIX_DS41_ATTN_TILE_F16 against today's f32 tile, device against device (same arithmetic after the tile, so outputs,
// NaN payloads included, must match byte for byte). mode 0: ring entries cover every E4M3 code x every E8M0 scale byte
// and compressed entries every E2M1 code x every E4M3 scale byte (zeros, NaN, infinities, subnormal products); mode 1:
// normal scales except one exceptional value in every other 16-key tile (alternating fast / low-plane tiles).
void tile_f16_sweep(int mode, unsigned rows, unsigned heads) {
    const unsigned cap = a::kWindow + a::kMaxRows;
    std::vector<std::uint8_t> ring(std::size_t{cap} * a::kWindowEntryBytes), kv(4096U * a::kKvEntryBytes);
    unsigned code = 0, scale = 0;
    for (std::size_t e = 0; e < cap; ++e) {
        for (unsigned d = 0; d < 512; ++d) ring[e * 528 + d] = static_cast<std::uint8_t>(mode == 0 ? (code++ & 255U) : (rng() & 0x7EU));
        for (unsigned c = 0; c < 16; ++c) ring[e * 528 + 512 + c] = static_cast<std::uint8_t>(mode == 0 ? (scale++ & 255U) : 118U + rng() % 4U);
        if (mode == 1 && (e / 16U) % 2U == 1U && e % 16U == 5U) ring[e * 528 + 512 + 3] = 2U;   // one subnormal group in the tile
    }
    for (std::size_t t = 0; t < 4096U; ++t) {
        for (unsigned i = 0; i < 256; ++i) kv[t * 288 + i] = static_cast<std::uint8_t>(mode == 0 ? (code++ & 255U) : rng());
        for (unsigned c = 0; c < 32; ++c) kv[t * 288 + 256 + c] = static_cast<std::uint8_t>(mode == 0 ? (scale++ & 255U) : 0x30U + rng() % 16U);
    }
    std::vector<std::uint64_t> pos(rows);
    for (unsigned r = 0; r < rows; ++r) pos[r] = 5000U - (rows - 1) + r;
    std::vector<std::uint32_t> idx(std::size_t{rows} * a::kTopK), cnt(rows, a::kTopK);
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned i = 0; i < a::kTopK; ++i) idx[std::size_t{r} * a::kTopK + i] = 8U * i + r;
    Dev<std::uint8_t> dring(ring), dkv(kv);
    Dev<std::uint32_t> didx(idx), dcnt(cnt), dtick(std::size_t{rows} * g::kAttnGroupCount);
    Dev<float> dq(random_f(std::size_t{rows} * heads * 512, -0.5F, 0.5F)), dsink(random_f(heads, -2, 2));
    Dev<float> dpart(std::size_t{rows} * heads * a::kMaxChunks * 514), dout(std::size_t{rows} * heads * 512);
    Dev<std::uint64_t> dpos(pos);
    Dev<a::RopeRow> drope(ropes(a::RopeKind::compressed, pos));
    g::AttentionView v{rows, cap, a::kTopK, dq.p, dsink.p, drope.p, dpos.p, dring.p, dkv.p, didx.p, dcnt.p, dpart.p, dout.p, dtick.p};
    v.heads = heads;
    std::vector<std::vector<float>> outs;
    for (const int bf : {0, 1}) {
        g::set_attention_tile_f16(bf);
        check(hipMemset(dout.p, 0xA5, dout.n * 4));
        g::enqueue_attention(v, nullptr);
        check(hipDeviceSynchronize());
        outs.push_back(dout.get());
    }
    g::set_attention_tile_f16(-1);
    expect(bytes_of(outs[0]) == bytes_of(outs[1]), "f16 tile == f32 tile, sweep mode " + std::to_string(mode) + " rows " + std::to_string(rows) +
                                                     " heads " + std::to_string(heads));
    std::printf("f16 tile sweep mode %d (%s), rows %u, heads %u: outputs byte-identical to the f32 tile\n", mode,
                mode == 0 ? "every code x every scale byte" : "alternating fast / low-plane tiles", rows, heads);
}

// Large slabs vs legacy 64-row calls over identical input/state, including a
// non-multiple tail. No change to the key/chunk/softmax sequence of any row.
void slab_attention(unsigned rows, unsigned heads, bool fused) {
    const unsigned cap=a::kWindow+rows, nc=37U, ncache=4096U;
    std::vector<std::uint8_t> ring(cap*a::kWindowEntryBytes), kv(ncache*a::kKvEntryBytes);
    for (unsigned slot=0;slot<cap;++slot) {
        for(unsigned i=0;i<512;++i) ring[slot*528+i]=static_cast<std::uint8_t>(rng()%120U);
        for(unsigned i=0;i<16;++i) ring[slot*528+512+i]=120U;
    }
    for(unsigned t=0;t<ncache;++t) {
        for(unsigned i=0;i<256;++i) kv[t*288+i]=static_cast<std::uint8_t>(rng());
        for(unsigned i=0;i<32;++i) kv[t*288+256+i]=0x30U;
    }
    std::vector<std::uint64_t> pos(rows);for(unsigned r=0;r<rows;++r)pos[r]=4096U+r;
    std::vector<std::uint32_t> idx(std::size_t{rows}*a::kTopK), cnt(rows,nc);
    for(unsigned r=0;r<rows;++r)for(unsigned i=0;i<nc;++i)idx[std::size_t{r}*a::kTopK+i]=i*7U;
    Dev<std::uint8_t> dring(ring),dkv(kv);
    Dev<float> dq(random_f(std::size_t{rows}*heads*512,-0.5F,0.5F)), dsink(random_f(heads,-2,2));
    Dev<std::uint64_t> dpos(pos);Dev<std::uint32_t> didx(idx),dcnt(cnt);
    Dev<a::RopeRow> drope(ropes(a::RopeKind::compressed,pos));
    Dev<float> partial(std::size_t{rows}*heads*a::kMaxChunks*514),out(std::size_t{rows}*heads*512);
    Dev<std::uint32_t> tickets(std::vector<std::uint32_t>(std::size_t{rows}*g::kAttnGroupCount,0));
    Dev<std::int8_t> codes(out.n);Dev<float> deltas(out.n/32);Dev<std::int32_t> sums(out.n/32);
    const auto run=[&](unsigned slab) {
        for(unsigned r=0;r<rows;r+=slab) {
            g::AttentionView v{std::min(slab,rows-r),cap,a::kTopK,dq.p+std::size_t{r}*heads*512,dsink.p,
                drope.p+r,dpos.p+r,dring.p,dkv.p,didx.p+std::size_t{r}*a::kTopK,dcnt.p+r,
                partial.p+std::size_t{r}*heads*a::kMaxChunks*514,out.p+std::size_t{r}*heads*512};
            v.heads=heads;v.tickets=fused?tickets.p+std::size_t{r}*g::kAttnGroupCount:nullptr;
            v.codes=codes.p+std::size_t{r}*heads*512;v.deltas=deltas.p+std::size_t{r}*heads*16;
            v.sums=sums.p+std::size_t{r}*heads*16;
            g::enqueue_attention(v,nullptr);
        }
        check(hipDeviceSynchronize());
        std::vector<std::uint8_t> bytes;
        const auto append=[&](const auto& dev) { const auto h=dev.get();const auto* p=reinterpret_cast<const std::uint8_t*>(h.data());bytes.insert(bytes.end(),p,p+h.size()*sizeof(h[0])); };
        append(out);append(codes);append(deltas);append(sums);append(tickets);
        return bytes;
    };
    const auto ref=run(64),got=run(rows);
    expect(ref==got,"slab == 64-row calls rows="+std::to_string(rows)+" heads="+std::to_string(heads)+" fused="+std::to_string(fused));
    std::printf("slab rows=%u heads=%u fused=%u bytes=%zu equal=%u\n",rows,heads,fused,ref.size(),ref==got);
}

// Prefill-shaped attention windows (the prefill's calls: `rows` consecutive positions per call, `heads` heads, one
// call per window spread over 4 streams with per-stream partials and tickets). mode 0: ring entries cover every E4M3
// code x every E8M0 scale byte and compressed entries every E2M1 code x every E4M3 scale byte (NaN, infinities,
// subnormal products); mode 1: the normal scales of real caches.
struct PrefillWindows {
    static constexpr unsigned kLanes = 4, kCache = 4096;
    unsigned rows, heads, nwin, cap;
    std::vector<std::uint64_t> pos;
    std::unique_ptr<Dev<std::uint8_t>> ring, kv;
    std::unique_ptr<Dev<float>> q, sink, out, part;
    std::unique_ptr<Dev<std::uint64_t>> dpos;
    std::unique_ptr<Dev<a::RopeRow>> rope;
    std::unique_ptr<Dev<std::uint32_t>> idx, cnt, tick;
    std::unique_ptr<Dev<std::int8_t>> codes;
    std::unique_ptr<Dev<float>> deltas;
    std::unique_ptr<Dev<std::int32_t>> sums;
    hipStream_t st[kLanes]{};
    bool window_only = false, a8 = false, live = false;   // live: AttentionView::max_visible from the window's last row
    std::vector<std::uint32_t> hcount;
    // count of row at position p: every compressed position up to p / ratio (ratio 0: window-only layer), at most top-k
    PrefillWindows(int mode, unsigned rows_, unsigned heads_, unsigned nwin_, std::uint64_t p0, unsigned ratio, bool a8_)
        : rows(rows_), heads(heads_), nwin(nwin_), cap(a::kWindow + rows_ * nwin_ + 8U), window_only(ratio == 0U), a8(a8_) {
        std::vector<std::uint8_t> hr(std::size_t{cap} * a::kWindowEntryBytes), hk(std::size_t{kCache} * a::kKvEntryBytes);
        unsigned code = 0, scale = 0;
        for (std::size_t e = 0; e < cap; ++e) {
            for (unsigned d = 0; d < 512; ++d) hr[e * 528 + d] = static_cast<std::uint8_t>(mode == 0 ? (code++ & 255U) : (rng() & 0x7EU));
            for (unsigned c = 0; c < 16; ++c) hr[e * 528 + 512 + c] = static_cast<std::uint8_t>(mode == 0 ? (scale++ & 255U) : 116U + rng() % 6U);
        }
        for (std::size_t t = 0; t < kCache; ++t) {
            for (unsigned i = 0; i < 256; ++i) hk[t * 288 + i] = static_cast<std::uint8_t>(mode == 0 ? (code++ & 255U) : rng());
            for (unsigned c = 0; c < 32; ++c) hk[t * 288 + 256 + c] = static_cast<std::uint8_t>(mode == 0 ? (scale++ & 255U) : 0x30U + rng() % 16U);
        }
        const unsigned n = rows * nwin;
        pos.resize(n);
        for (unsigned r = 0; r < n; ++r) pos[r] = p0 + r;
        std::vector<std::uint32_t> hidx(std::size_t{n} * a::kTopK), hcnt(n);
        for (unsigned r = 0; r < n; ++r) {
            const auto c = static_cast<std::ptrdiff_t>(ratio == 0U ? 0U : std::min<std::uint64_t>((pos[r] + 1U) / ratio, a::kTopK));
            hcnt[r] = static_cast<std::uint32_t>(c);
            std::vector<std::uint32_t> pick(kCache);
            for (std::uint32_t i = 0; i < kCache; ++i) pick[i] = i;
            std::shuffle(pick.begin(), pick.end(), rng);
            std::sort(pick.begin(), pick.begin() + c);
            std::copy(pick.begin(), pick.begin() + c, &hidx[std::size_t{r} * a::kTopK]);
        }
        ring = std::make_unique<Dev<std::uint8_t>>(hr);
        kv = std::make_unique<Dev<std::uint8_t>>(hk);
        auto hq = random_f(std::size_t{n} * heads * 512, -0.5F, 0.5F);
        if (mode == 2) {   // NaN payloads in q (other payloads and signs in lanes that meet in the butterfly): operand order
            const auto nan = [](std::uint32_t u) { return std::bit_cast<float>(u); };
            for (std::size_t rh = 0; rh < std::size_t{n} * heads; ++rh) {
                float* x = hq.data() + rh * 512;
                if (rh % 4 == 1) { x[16 * 3] = nan(0x7FC12345U); x[16 * 19 + 2] = nan(0xFFD00001U); }   // xor-16 partners
                if (rh % 4 == 2) { x[16 * 5 + 1] = nan(0xFFC00000U); x[16 * 6] = nan(0x7FC00000U); x[16 * 13] = nan(0x7FC0BEEFU); }
                if (rh % 4 == 3) x[16 * 28 + 7] = nan(0xFFC00F00U);   // one lane (a RoPE lane)
            }
        }
        q = std::make_unique<Dev<float>>(hq);
        sink = std::make_unique<Dev<float>>(random_f(heads, -2, 2));
        out = std::make_unique<Dev<float>>(std::size_t{n} * heads * 512);
        part = std::make_unique<Dev<float>>(std::size_t{kLanes} * rows * heads * a::kMaxChunks * 514);
        dpos = std::make_unique<Dev<std::uint64_t>>(pos);
        rope = std::make_unique<Dev<a::RopeRow>>(ropes(a::RopeKind::compressed, pos));
        hcount = hcnt;
        idx = std::make_unique<Dev<std::uint32_t>>(hidx);
        cnt = std::make_unique<Dev<std::uint32_t>>(hcnt);
        tick = std::make_unique<Dev<std::uint32_t>>(std::size_t{kLanes} * rows * g::kAttnGroupCount);
        codes = std::make_unique<Dev<std::int8_t>>(std::size_t{n} * heads * 512);
        deltas = std::make_unique<Dev<float>>(std::size_t{n} * heads * 16);
        sums = std::make_unique<Dev<std::int32_t>>(std::size_t{n} * heads * 16);
        for (auto& s : st) check(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    }
    ~PrefillWindows() { for (auto s : st) (void)hipStreamDestroy(s); }
    PrefillWindows(const PrefillWindows&) = delete;
    PrefillWindows& operator=(const PrefillWindows&) = delete;
    g::AttentionView view(unsigned w) const {
        const std::size_t r0 = std::size_t{w} * rows;
        const unsigned lane = w % kLanes;
        g::AttentionView v{rows, cap, a::kTopK, q->p + r0 * heads * 512, sink->p, rope->p + r0, dpos->p + r0, ring->p,
                           window_only ? nullptr : kv->p, idx->p + r0 * a::kTopK, window_only ? nullptr : cnt->p + r0,
                           part->p + std::size_t{lane} * rows * heads * a::kMaxChunks * 514, out->p + r0 * heads * 512,
                           tick->p + std::size_t{lane} * rows * g::kAttnGroupCount};
        v.heads = heads;
        if (a8) { v.codes = codes->p + r0 * heads * 512; v.deltas = deltas->p + r0 * heads * 16; v.sums = sums->p + r0 * heads * 16; }
        if (live) {
            const std::size_t last = r0 + rows - 1U;
            v.max_visible = static_cast<std::uint32_t>(std::min<std::uint64_t>(pos[last] + 1U, a::kWindow) + (window_only ? 0U : hcount[last]));
        }
        return v;
    }
    void run() const {
        for (unsigned w = 0; w < nwin; ++w) g::enqueue_attention(view(w), st[w % kLanes]);
        check(hipDeviceSynchronize());
    }
    std::vector<std::uint8_t> bytes() const {
        std::vector<std::uint8_t> b = bytes_of(out->get());
        if (a8) { const auto c = bytes_of(codes->get()), d = bytes_of(deltas->get()), s = bytes_of(sums->get()); b.insert(b.end(), c.begin(), c.end()); b.insert(b.end(), d.begin(), d.end()); b.insert(b.end(), s.begin(), s.end()); }
        const auto t = bytes_of(tick->get());
        b.insert(b.end(), t.begin(), t.end());
        return b;
    }
    void clear() const { check(hipMemset(out->p, 0xA5, out->n * 4)); check(hipMemset(codes->p, 0x5A, codes->n)); check(hipDeviceSynchronize()); }
};

// STRIX_DS41_ATTN_LEAN against the previous kernel form, device against device on prefill-shaped calls (outputs, NaN
// payloads included, A8 codes and the tickets must match byte for byte), at every tile choice the call can take.
void lean_equal(int mode, unsigned rows, unsigned heads, unsigned nwin, std::uint64_t p0, unsigned ratio, bool a8) {
    const PrefillWindows pw(mode, rows, heads, nwin, p0, ratio, a8);
    for (const int swz : {0, 1}) for (const int f16 : {0, 1}) {
        g::set_attention_lds_swizzle(swz);
        g::set_attention_tile_f16(f16);
        std::vector<std::uint8_t> ref;
        for (const int lean : {0, 1, 2, 3}) {
            g::set_attention_lean(lean);
            pw.clear();
            pw.run();
            if (lean == 0) ref = pw.bytes();
            else expect(pw.bytes() == ref, "lean " + std::to_string(lean) + " == previous form, mode " + std::to_string(mode) + " rows " + std::to_string(rows) + " heads " +
                        std::to_string(heads) + " windows " + std::to_string(nwin) + " p0 " + std::to_string(p0) + " ratio " + std::to_string(ratio) +
                        " a8 " + std::to_string(a8) + " swizzle " + std::to_string(swz) + " f16 " + std::to_string(f16));
        }
        // lean 3 with the 6-wave register budget off (STRIX_DS41_ATTN_WAVE_CAP=0) and on: the same bytes
        for (const int cap : {0, 1}) {
            g::set_attention_wave_cap(cap);
            g::set_attention_lean(3);
            pw.clear();
            pw.run();
            expect(pw.bytes() == ref, "lean 3 wave cap " + std::to_string(cap) + " == previous form, mode " + std::to_string(mode) + " rows " +
                   std::to_string(rows) + " heads " + std::to_string(heads) + " ratio " + std::to_string(ratio) + " swizzle " + std::to_string(swz) +
                   " f16 " + std::to_string(f16));
        }
        g::set_attention_wave_cap(-1);
    }
    g::set_attention_lean(-1);
    g::set_attention_tile_f16(-1);
    g::set_attention_lds_swizzle(-1);
    std::printf("lean == previous form: mode %d, %u windows x %u rows from position %llu, %u heads, ratio %u%s\n", mode, nwin, rows,
                static_cast<unsigned long long>(p0), heads, ratio, a8 ? ", A8" : "");
}

// Prefill attention bench: `nwin` windows over 4 streams, the forms alternated (STRIX_DS41_PF_BENCH_FORMS, digits of
// STRIX_DS41_ATTN_LEAN levels, default 0123), median wall time per window.
void prefill_bench(unsigned rows, unsigned heads, unsigned nwin, std::uint64_t p0, unsigned ratio, unsigned reps) {
    PrefillWindows pw(1, rows, heads, nwin, p0, ratio, true);
    hipEvent_t e0, e1;
    check(hipEventCreate(&e0)); check(hipEventCreate(&e1));
    std::vector<int> forms;
    { const char* e = std::getenv("STRIX_DS41_PF_BENCH_FORMS"); for (const char* c = e ? e : "0123"; *c; ++c) if (*c >= '0' && *c <= '9') forms.push_back(*c - '0'); }
    const unsigned nf = static_cast<unsigned>(forms.size());
    std::vector<std::vector<float>> t(nf);
    for (unsigned k = 0; k < nf * reps + nf; ++k) {
        const int lean = forms[k % nf];
        g::set_attention_lean(lean);
        pw.run();   // warm
        check(hipEventRecord(e0));
        for (unsigned w = 0; w < pw.nwin; ++w) g::enqueue_attention(pw.view(w), pw.st[w % PrefillWindows::kLanes]);
        for (auto s : pw.st) { hipEvent_t d; check(hipEventCreateWithFlags(&d, hipEventDisableTiming)); check(hipEventRecord(d, s)); check(hipStreamWaitEvent(nullptr, d, 0)); check(hipEventDestroy(d)); }
        check(hipEventRecord(e1));
        check(hipEventSynchronize(e1));
        float ms; check(hipEventElapsedTime(&ms, e0, e1));
        if (k >= nf) t[k % nf].push_back(ms * 1e3F * 8.0F / static_cast<float>(nwin * rows));   // us per 8 rows
    }
    g::set_attention_lean(-1);
    for (auto& v : t) std::sort(v.begin(), v.end());
    std::printf("  prefill bench %u windows x %u rows, %u heads, positions %llu.., ratio %u: us per 8 rows (median, min)", nwin, rows, heads,
                static_cast<unsigned long long>(p0), ratio);
    for (unsigned f = 0; f < nf; ++f)
        std::printf("%s lean %d %.1f (%+.1f %%) [%.1f]", f ? "," : "", forms[f], t[f][reps / 2], 100.0 * (t[f][reps / 2] / t[0][reps / 2] - 1.0), t[f][0]);
    std::printf("\n");
    if (std::getenv("STRIX_DS41_PF_BENCH_EXTRA") == nullptr) { check(hipEventDestroy(e0)); check(hipEventDestroy(e1)); return; }
    // lean 2 with AttentionView::max_visible (the grid's live chunks) at the default heads-per-wave rule and at Hpw 4 always
    float tl[2] = {0.0F, 0.0F};
    {
        pw.live = true;
        for (const unsigned mb : {0U, 1U}) {
            g::set_attention_min_blocks(mb);
            std::vector<float> tv;
            for (unsigned k = 0; k <= reps; ++k) {
                check(hipEventRecord(e0));
                for (unsigned w = 0; w < pw.nwin; ++w) g::enqueue_attention(pw.view(w), pw.st[w % PrefillWindows::kLanes]);
                for (auto s : pw.st) { hipEvent_t d; check(hipEventCreateWithFlags(&d, hipEventDisableTiming)); check(hipEventRecord(d, s)); check(hipStreamWaitEvent(nullptr, d, 0)); check(hipEventDestroy(d)); }
                check(hipEventRecord(e1));
                check(hipEventSynchronize(e1));
                float ms; check(hipEventElapsedTime(&ms, e0, e1));
                if (k > 0) tv.push_back(ms * 1e3F * 8.0F / static_cast<float>(nwin * rows));
            }
            std::sort(tv.begin(), tv.end());
            tl[mb] = tv[tv.size() / 2];
        }
        g::set_attention_min_blocks(0U);
        pw.live = false;
    }
    float tb[2] = {0.0F, 0.0F};   // reference only: the approximate BF16 hi/lo prefill attention (other bits), variants 1 and 2
    if (heads % 16U == 0U && ratio != 0U) {
        for (const int variant : {1, 2}) {
            std::vector<float> tv;
            for (unsigned k = 0; k <= reps; ++k) {
                check(hipEventRecord(e0));
                for (unsigned w = 0; w < pw.nwin; ++w) g::enqueue_attention_bf16_prefill(pw.view(w), pw.st[w % PrefillWindows::kLanes], variant);
                for (auto s : pw.st) { hipEvent_t d; check(hipEventCreateWithFlags(&d, hipEventDisableTiming)); check(hipEventRecord(d, s)); check(hipStreamWaitEvent(nullptr, d, 0)); check(hipEventDestroy(d)); }
                check(hipEventRecord(e1));
                check(hipEventSynchronize(e1));
                float ms; check(hipEventElapsedTime(&ms, e0, e1));
                if (k > 0) tv.push_back(ms * 1e3F * 8.0F / static_cast<float>(nwin * rows));
            }
            std::sort(tv.begin(), tv.end());
            tb[variant - 1] = tv[tv.size() / 2];
        }
    }
    std::printf("    lean 2 + max_visible: %.1f us (default heads-per-wave rule), %.1f us (Hpw 4 always) per 8 rows\n", tl[0], tl[1]);
    if (tb[0] > 0.0F) std::printf("    (approximate BF16 hi/lo prefill attention, not this identity: variant 1 %.1f us, variant 2 %.1f us per 8 rows; the combine kernel launch included)\n", tb[0], tb[1]);
    check(hipEventDestroy(e0)); check(hipEventDestroy(e1));
}

}  // namespace

int main() {
    try {
        if (std::getenv("STRIX_DS41_PF_ATTN_BENCH")) {   // prefill-shaped attention: lean equality + bench only
            for (const unsigned heads : {32U, 64U}) {
                lean_equal(0, 8, heads, 8, 5000, 4, true);
                lean_equal(2, 8, heads, 8, 9000, 4, true);     // NaN payloads and signs in q
                lean_equal(2, 3, heads, 4, 333, 2, false);
                lean_equal(1, 8, heads, 8, 0, 4, false);       // the first rows: windows < 128, counts < 512
                lean_equal(1, 8, heads, 8, 60, 2, true);       // across the 64-key chunk boundary, ratio 2
                lean_equal(1, 8, heads, 4, 120, 1, false);     // window 121..128 keys, ratio 1 counts 121..
                lean_equal(1, 8, heads, 6, 8000, 128, true);   // ratio 128 (62 compressed keys)
                lean_equal(1, 8, heads, 6, 8000, 0, false);    // window-only layer
                lean_equal(1, 8, heads, 8, 9000, 4, true);     // full top-k
                lean_equal(1, 1, heads, 4, 9000, 4, false);    // one row per call (other heads-per-wave forms)
                lean_equal(1, 3, heads, 4, 333, 2, true);
            }
            const unsigned reps = 7;
            for (const unsigned heads : {32U, 64U}) {
                prefill_bench(8, heads, 64, 9000, 4, reps);
                prefill_bench(8, heads, 64, 9000, 128, reps);
                prefill_bench(8, heads, 64, 9000, 0, reps);
                prefill_bench(8, heads, 64, 1000, 4, reps);
                prefill_bench(64, heads, 8, 9000, 4, reps);   // STRIX_DS41_PREFILL_WINDOW=64: the same rows in 64-row calls
            }
            std::puts(failures == 0 ? "prefill attention lean checks pass" : "FAILED");
            return failures ? 1 : 0;
        }
        if (std::getenv("STRIX_DS41_SLAB_TESTS_ONLY")) {
            for (unsigned rows : {65U, 128U, 256U})
                for (unsigned heads : {32U, 64U})
                    for (bool fused : {false, true}) slab_attention(rows,heads,fused);
            for (unsigned rows : {65U, 128U, 256U}) { caches(1,519,0,rows); caches(2,519,0,rows); }
            index_and_select(90,65,1,64,8,8,false);
            std::puts(failures==0?"large attention slab checks PASS":"FAILED");
            return failures?1:0;
        }
        if (std::getenv("STRIX_DS41_DSATTN_BENCH")) {   // attention benchmarks only
            for (const std::uint64_t n : {511ULL, 1023ULL, 8191ULL, 32767ULL, 131071ULL}) for (const unsigned rows : {1U, 6U, 8U})
                index_filter_direct(n, rows, 1, a::kTopK, a::kCandBlocks, a::kCandBlockSize, true);
            for (const unsigned segs : {1U, 2U, 4U, 8U}) for (const unsigned heads : {32U, 64U}) segment_hpw_bench(segs, 1U, heads);
            segment_hpw_bench(8U, 2U, 32U);
            for (const unsigned mb : {40U, 20U, 10U}) {   // one call (R1 / R6 / R8, TP and full heads) per grid threshold
                g::set_attention_min_blocks(mb);
                std::printf("  min blocks %u:\n", mb);
                for (const unsigned rows : {1U, 6U, 8U}) for (const unsigned heads : {32U, 64U}) segment_hpw_bench(1U, rows, heads);
            }
            g::set_attention_min_blocks(0U);
            std::puts(failures == 0 ? "attention benchmarks pass" : "FAILED");
            return failures ? 1 : 0;
        }
        const bool bench = std::getenv("STRIX_DS41_ATTN_BENCH") != nullptr;
        caches(2, 333);
        caches(1, 300);
        caches(2, 333, 1);
        caches(1, 300, 1);
        compress_fast(bench);
        g::set_compress_ahead(1);  // STRIX_DS41_COMPRESS_AHEAD: the same caches
        caches(2, 333, 0);
        caches(1, 300, 0);
        g::set_compress_ahead(-1);
        index_and_select(99, 1, 1, a::kTopK, a::kCandBlocks, a::kCandBlockSize, false);       // count <= topk
        index_and_select(1401, 3, 2, a::kTopK, a::kCandBlocks, a::kCandBlockSize, false);     // ratio 2, 701 positions
        index_and_select(20007, 8, 1, a::kTopK, a::kCandBlocks, a::kCandBlockSize, false);    // 2501 blocks > 2048
        index_and_select(4000, 2, 1, 64, 8, 8, false);                                          // small pool parameters
        index_and_select(9000, 2, 1, 64, 8, 8, false);                                          // two-level selection, small k
        index_and_select(4600, 3, 1, a::kTopK, a::kCandBlocks, a::kCandBlockSize, false);     // two slices, the last short
        index_filter_direct(99, 1, 1, a::kTopK, a::kCandBlocks, a::kCandBlockSize, false);    // direct, one row
        index_filter_direct(1023, 8, 2, a::kTopK, a::kCandBlocks, a::kCandBlockSize, false);  // direct, ratio 2, count 512
        index_filter_direct(511, 8, 1, a::kTopK, a::kCandBlocks, a::kCandBlockSize, false);   // direct at count = topk
        index_filter_direct(70, 64, 1, a::kTopK, a::kCandBlocks, a::kCandBlockSize, false);   // direct, 64 rows
        index_filter_direct(20007, 8, 1, a::kTopK, a::kCandBlocks, a::kCandBlockSize, false); // filter: 2501 blocks > 2048
        index_filter_direct(40001, 64, 1, a::kTopK, a::kCandBlocks, a::kCandBlockSize, false); // filter, 64 rows, two-level select
        index_filter_direct(4000, 2, 1, 64, 8, 8, false);                                        // filter, small pool
        index_filter_direct(9000, 3, 1, 64, 8, 8, false);                                        // filter, small pool, two-level
        index_filter_direct(50, 5, 1, 64, 8, 8, false);                                          // direct, small pool
        for (const unsigned levels : {1U, 2U, 7U, 300U}) {                                     // equal keys at the threshold
            select_ties(3000, 2, levels, a::kTopK, a::kCandBlocks, a::kCandBlockSize);          // one block
            select_ties(40000, 3, levels, a::kTopK, a::kCandBlocks, a::kCandBlockSize);         // two-level selection
            select_ties(9000, 2, levels, 64, 8, 8);                                             // two-level, small pool
        }
        attention(5, 1, 0, false);       // window only, partial window
        attention(1000, 3, 0, false);
        attention(1000, 1, 300, false);
        attention(20, 2, 37, false);     // partial window and compressed keys in one tile
        attention(5000, 8, a::kTopK, false);
        segments();
        g::set_attention_lds_swizzle(1);
        for (const int bf : {0, 1}) {   // the swizzle with either tile choice (it permutes the f32 tiles only)
            g::set_attention_tile_f16(bf);
            attention(5, 1, 0, false);
            attention(20, 2, 37, false);
            attention(8191, 8, a::kTopK, false);
            attention(32767, 8, a::kTopK, false);
            attention(8191, 1, a::kTopK, false);
            attention(32767, 1, a::kTopK, false);
            segments();
        }
        g::set_attention_tile_f16(-1);
        g::set_attention_lds_swizzle(-1);
        for (const int bf : {0, 1}) {     // STRIX_DS41_ATTN_WINDOW_FOLD, f32 and f16 tiles
            g::set_attention_tile_f16(bf);
            window_fold(0, 1, 0, 64, 1);       // the first position: window of one key
            window_fold(7, 8, 0, 32, 1);       // rows 0..7, window-only layer
            window_fold(100, 1, 101, 32, 1);
            window_fold(140, 6, 141, 32, 1);   // ring wrap (capacity 136)
            window_fold(5000, 8, a::kTopK, 32, 1);
            window_fold(5000, 6, a::kTopK, 64, 1);
            window_fold(5001, 1, a::kTopK, 32, 8);   // N8 segment launch
            window_fold(301, 2, 302, 32, 3);
            window_fold(130, 1, 0, 64, 1, a::RopeKind::window);        // window-only layer, window RoPE table
            window_fold(6000, 7, 0, 32, 1, a::RopeKind::window);
            window_fold(63, 3, 64, 32, 1, a::RopeKind::compressed, 1); // adversarial rows across a chunk boundary
            window_fold(2000, 8, a::kTopK, 32, 1, a::RopeKind::compressed, 1);
            window_fold(4000, 5, 400, 32, 8, a::RopeKind::compressed, 1);   // heterogeneous segments (5, 4, 3, 2, 1 rows)
        }
        fold_live = true;                 // + STRIX_DS41_ATTN_LIVE_CHUNKS (grid and heads per wave from the bound)
        for (const int bf : {0, 1}) {
            g::set_attention_tile_f16(bf);
            window_fold(0, 1, 0, 64, 1);
            window_fold(100, 1, 101, 32, 1);
            window_fold(140, 6, 141, 32, 1);
            window_fold(100, 8, 101, 32, 1);
            window_fold(5000, 6, a::kTopK, 32, 1);
            window_fold(301, 2, 302, 32, 3);
            window_fold(4000, 5, 400, 32, 8, a::RopeKind::compressed, 1);
        }
        g::set_attention_tile_f16(-1);
        fold_live = false;
        for (const unsigned rows : {1U, 6U, 8U})
            for (const unsigned heads : {32U, 64U}) { tile_f16_sweep(0, rows, heads); tile_f16_sweep(1, rows, heads); }
        for (const unsigned heads : {32U, 64U}) {   // STRIX_DS41_ATTN_LEAN == the previous form on prefill-shaped calls
            lean_equal(0, 8, heads, 4, 5000, 4, true);
            lean_equal(2, 8, heads, 4, 9000, 4, true);
            lean_equal(1, 8, heads, 4, 0, 4, false);
            lean_equal(1, 8, heads, 4, 60, 2, true);
            lean_equal(1, 8, heads, 4, 8000, 128, true);
            lean_equal(1, 8, heads, 4, 8000, 0, false);
            lean_equal(1, 1, heads, 4, 9000, 4, false);
        }
        g::set_attention_tile_f16(1);   // STRIX_DS41_ATTN_TILE_F16: the same host references
        attention(5, 1, 0, false);
        attention(1000, 3, 0, false);
        attention(1000, 1, 300, false);
        attention(20, 2, 37, false);
        attention(5000, 8, a::kTopK, false);
        attention(8191, 6, a::kTopK, false);
        attention(32767, 1, a::kTopK, false);
        ring_scale_lo = 1;               // subnormal window values: those tiles fall back to f32
        attention(1000, 3, 0, false);
        attention(5000, 6, a::kTopK, false);
        ring_scale_lo = 118;
        segments();
        g::set_attention_tile_f16(-1);
        // long context: 512K positions (ratio 1: 512K index keys, 128 select slices; ratio 2: 256K), attention at 1M
        index_and_select(524287, 1, 1, a::kTopK, a::kCandBlocks, a::kCandBlockSize, false);
        index_and_select(524287, 2, 2, a::kTopK, a::kCandBlocks, a::kCandBlockSize, false);
        attention(1048575, 3, a::kTopK, false);
        if (std::getenv("STRIX_DS41_ATTN_LONG") != nullptr) {  // minutes of host reference: 1M positions, 8-row windows
            index_and_select(1048575, 8, 1, a::kTopK, a::kCandBlocks, a::kCandBlockSize, false);
            index_and_select(1048575, 8, 2, a::kTopK, a::kCandBlocks, a::kCandBlockSize, false);
            attention(1048575, 8, a::kTopK, false);
        }
        if (bench) {
            for (const std::uint64_t n : {32767ULL, 131071ULL, 262143ULL, 524287ULL}) for (const unsigned rows : {1U, 8U})
                index_and_select(n, rows, 1, a::kTopK, a::kCandBlocks, a::kCandBlockSize, true);
            for (const unsigned rows : {1U, 4U, 8U}) attention(100000, rows, a::kTopK, true);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    std::printf("%s\n", failures == 0 ? "ds41 attention device tests pass" : "FAILED");
    return failures == 0 ? 0 : 1;
}
