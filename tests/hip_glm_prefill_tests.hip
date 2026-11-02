// Gate PF-K: GLM chunked-prefill Tier A kernels (src/glm/prefill) == the decode kernels applied row by row in position
// order, bit for bit (decode == host emulation is gate K: hip_glm_tier_a_tests / hip_glm_dsa_tests). Fail-closed:
// fixtures are checked against an exact spec (names, types, counts, file lengths; their SHA-256 is checked by the validation
// wrapper against the fixture SHA-256 manifest), every compared output must be finite, every token list is
// validated before use, and timing runs only when every check passed.
//   KDA (three weight sets: real L4 rank-0 H=32, synthetic H=32 "rank-1" stand-in, synthetic H=64 single-node):
//     chunks incl. C = 1, 2, 3, 15, 16, 17 (scan staging), 64, 1000, 2048; signed-zero / subnormal / all-zero rows;
//     per row: prep scratch (q, k, v, g) + beta, raw scan output o, y; S + conv after every chunk (incl. the first);
//     positive C2 test (+inf input sets the flag on both paths).
//   DSA (real L3 indexer + kv_b): 6000 rows, chunks crossing pool / 512-pool / 2048 boundaries: caches (+ untouched
//     poisoned guard region), per-row scaled wh and ordered keys (P > 512 rows), validated token lists, MLA outputs for
//     rank-0 heads (H=32), rank-1 heads (H=32) and all heads (H=64); per-row absorbed q, probabilities and o-bar of
//     representative rows vs the host R13 expressions; positive C3/C4 test.
//   Max-P: 33,000 rows with pooled keys repeating every 3 pools (score ties across every 1024-pool scan block), lists
//     + MLA of the last 96 rows (P up to 8250) incl. all-tie queries.
//   Timing (informational, only after a clean gate).
// usage: hip_glm_prefill_tests CASES_DIR [--no-timing]
#include "glm/kernels/glm_dsa.hpp"
#include "glm/kernels/glm_tier_a.hpp"
#include "glm/prefill/glm_prefill.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace strix::glm;
namespace D = strix::glm::dev;
namespace P = strix::glm::pf;

namespace {
// Explicit MLA {scores, o-bar, gemm} versions gated bitwise against the 3/5/1 reference (intermediates, decode outputs).
constexpr unsigned kMlaPairs = 11;
constexpr unsigned kMlaPair[kMlaPairs][3] = {{4, 5, 1}, {3, 6, 1}, {4, 6, 1}, {5, 5, 1}, {4, 7, 1}, {5, 7, 1},
                                             {5, 7, 2}, {4, 5, 2}, {4, 6, 2}, {6, 7, 2}, {6, 5, 1}};

int failures = 0, checks = 0;
void check(bool ok, const std::string& what) {
    ++checks;
    std::printf("%s %s\n", ok ? "OK  " : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++failures;
}
[[noreturn]] void die(const std::string& w) {
    std::fprintf(stderr, "FATAL %s\n", w.c_str());
    std::exit(4);
}
void hok(hipError_t e, const char* w) {
    if (e != hipSuccess) die(std::string(w) + ": " + hipGetErrorString(e));
}
std::vector<void*> g_allocs;
template <class T> T* dalloc(std::size_t n, int fill) {
    T* p = nullptr;
    hok(hipMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "malloc");
    hok(hipMemset(p, fill, std::max<std::size_t>(n, 1) * sizeof(T)), "memset");
    g_allocs.push_back(p);
    return p;
}
template <class T> T* zalloc(std::size_t n) { return dalloc<T>(n, 0); }
template <class T> T* poison(std::size_t n) { return dalloc<T>(n, 0xFF); }  // NaN floats / all-ones ints
template <class T> T* up(const std::vector<T>& h) {
    T* p = dalloc<T>(h.size(), 0);
    hok(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "h2d");
    return p;
}
template <class T> std::vector<T> down(const T* p, std::size_t n) {
    std::vector<T> h(n);
    hok(hipDeviceSynchronize(), "sync");
    hok(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "d2h");
    return h;
}
void d2d(void* dst, const void* src, std::size_t bytes) { hok(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice), "d2d"); }
void free_all() {
    for (void* p : g_allocs) hok(hipFree(p), "free");
    g_allocs.clear();
}
bool bits_eq(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}
bool all_finite(const std::vector<float>& v) {
    for (float x : v)
        if (!is_finite(x)) return false;
    return true;
}
std::vector<float> rnd(std::size_t n, std::mt19937& g, float scale) {
    std::normal_distribution<float> d(0.0F, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g);
    return v;
}
std::vector<std::uint16_t> rnd_half(std::size_t n, std::mt19937& g, float scale, bool bf16) {
    const auto f = rnd(n, g, scale);
    std::vector<std::uint16_t> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (bf16) {
            v[i] = float_to_bf16(f[i]);
        } else {  // f16 by truncation of an exactly representable value: x = k / 1024, |k| < 2048
            const int k = std::max(-2047, std::min(2047, static_cast<int>(f[i] * 1024.0F)));
            const _Float16 h = static_cast<_Float16>(static_cast<float>(k) / 1024.0F);
            std::memcpy(&v[i], &h, 2);
        }
    }
    return v;
}

// ---------------------------------------------------------------- fail-closed fixture loading
struct Spec {
    const char* name;
    const char* dt;
    std::size_t count;
};
struct CaseDir {
    std::map<std::string, std::vector<char>> raw;
    CaseDir(const std::string& dir, std::initializer_list<Spec> want) {
        std::ifstream m(dir + "/manifest.txt");
        if (!m) die("no manifest in " + dir);
        std::map<std::string, std::pair<std::string, std::size_t>> man;
        std::string line;
        while (std::getline(m, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::string name, dt;
            std::size_t count = 0;
            if (!(ss >> name >> dt >> count)) die("bad manifest line in " + dir + ": " + line);
            if (!man.emplace(name, std::make_pair(dt, count)).second) die("duplicate manifest entry " + name);
        }
        for (const Spec& s : want) {
            const auto it = man.find(s.name);
            if (it == man.end()) die(dir + ": missing " + s.name);
            if (it->second.first != s.dt || it->second.second != s.count)
                die(dir + ": " + s.name + " is " + it->second.first + " x " + std::to_string(it->second.second) + ", expected " + s.dt + " x " +
                    std::to_string(s.count));
            const std::size_t es = std::string(s.dt) == "u16" ? 2 : (std::string(s.dt) == "f64" ? 8 : 4);
            if (s.count > (std::size_t{1} << 40) / es) die("count overflow " + std::string(s.name));
            const std::size_t bytes = s.count * es;
            std::ifstream f(dir + "/" + s.name + ".bin", std::ios::binary | std::ios::ate);
            if (!f) die(dir + ": no file " + s.name);
            if (static_cast<std::size_t>(f.tellg()) != bytes) die(dir + ": " + s.name + " length mismatch");
            f.seekg(0);
            std::vector<char> b(bytes);
            f.read(b.data(), static_cast<std::streamsize>(bytes));
            if (static_cast<std::size_t>(f.gcount()) != bytes) die("short read " + std::string(s.name));
            raw.emplace(s.name, std::move(b));
        }
    }
    template <class T> std::vector<T> v(const std::string& n) const {
        const auto& b = raw.at(n);
        if (b.size() % sizeof(T)) die("element size mismatch " + n);
        std::vector<T> out(b.size() / sizeof(T));
        std::memcpy(out.data(), b.data(), b.size());
        return out;
    }
};

float ms_between(hipEvent_t a, hipEvent_t b) {
    float ms = 0;
    hok(hipEventSynchronize(b), "ev");
    hok(hipEventElapsedTime(&ms, a, b), "ev");
    return ms;
}

// ---------------------------------------------------------------- KDA
struct KdaW {
    std::string label;
    unsigned H;
    std::vector<std::uint16_t> conv, onorm;
    std::vector<float> A_log, dt;
    std::vector<float> qkv, f, gate, b;  // optional real leading rows
};

void kda_case(const KdaW& kw, const std::vector<unsigned>& chunks, std::mt19937& g, bool special_rows, bool inf_test) {
    const unsigned H = kw.H, CH = H * kKdaDim;
    unsigned T = 0;
    for (unsigned x : chunks) T += x;
    const D::KdaDev kd{H, up(kw.conv), up(kw.A_log), up(kw.dt), up(kw.onorm)};
    std::vector<float> qkv = kw.qkv, f = kw.f, gate = kw.gate, b = kw.b;
    const unsigned real = static_cast<unsigned>(b.size() / H);
    for (unsigned t = real; t < T; ++t) {
        auto a = rnd(3 * CH, g, 1.0F), bb = rnd(CH, g, 1.0F), cc = rnd(CH, g, 1.0F), dd = rnd(H, g, 1.0F);
        if (special_rows && t % 97 == 5) {  // signed zeros, subnormals, an all-zero q/k head
            for (unsigned i = 0; i < 3 * CH; i += 3) a[i] = (i / 3) % 2 ? -0.0F : 1e-40F;
            for (unsigned i = 0; i < kKdaDim; ++i) a[i] = a[CH + i] = 0.0F;
            for (unsigned i = 0; i < CH; i += 5) bb[i] = -1e-41F;
            dd[0] = -0.0F;
        }
        qkv.insert(qkv.end(), a.begin(), a.end());
        f.insert(f.end(), bb.begin(), bb.end());
        gate.insert(gate.end(), cc.begin(), cc.end());
        b.insert(b.end(), dd.begin(), dd.end());
    }
    if (inf_test) qkv[static_cast<std::size_t>(T - 2) * 3 * CH + 2 * CH + 7] = __builtin_huge_valf();  // a v input -> C2
    float *dq = up(qkv), *df = up(f), *dg = up(gate), *db = up(b);
    const std::size_t SN = static_cast<std::size_t>(H) * kKdaDim * kKdaDim, CN = static_cast<std::size_t>(3) * CH * 3;
    const std::size_t SCR = static_cast<std::size_t>(H) * 4 * kKdaDim;
    // decode reference, one row per call; per-row intermediates captured
    float *rS = zalloc<float>(SN), *rconv = zalloc<float>(CN), *rscr = zalloc<float>(SCR), *rbeta = zalloc<float>(H);
    float *rO = zalloc<float>(CH), *rY = poison<float>(static_cast<std::size_t>(T) * CH);
    float *rSCR = poison<float>(static_cast<std::size_t>(T) * SCR), *rB = poison<float>(static_cast<std::size_t>(T) * H);
    float* rOall = poison<float>(static_cast<std::size_t>(T) * CH);
    unsigned* rflag = zalloc<unsigned>(1);
    std::vector<float*> rS_at, rconv_at;
    std::vector<unsigned> ends;
    {
        unsigned e = 0;
        for (unsigned x : chunks) ends.push_back(e += x);
    }
    for (unsigned t = 0; t < T; ++t) {
        D::kda_step(kd, dq + static_cast<std::size_t>(t) * 3 * CH, df + static_cast<std::size_t>(t) * CH, dg + static_cast<std::size_t>(t) * CH,
                    db + static_cast<std::size_t>(t) * H, rS, rconv, rscr, rbeta, rO, rY + static_cast<std::size_t>(t) * CH, rflag, 1, nullptr);
        d2d(rSCR + static_cast<std::size_t>(t) * SCR, rscr, SCR * 4);
        d2d(rB + static_cast<std::size_t>(t) * H, rbeta, H * 4);
        d2d(rOall + static_cast<std::size_t>(t) * CH, rO, CH * 4);
        if (std::find(ends.begin(), ends.end(), t + 1) != ends.end()) {
            float *s = zalloc<float>(SN), *cv = zalloc<float>(CN);
            d2d(s, rS, SN * 4);
            d2d(cv, rconv, CN * 4);
            rS_at.push_back(s);
            rconv_at.push_back(cv);
        }
    }
    // prefill, every scan version (same bits)
    for (unsigned sv : {1U, 3U, 5U}) {
        P::set_kda_scan_version(sv);
        const unsigned CMAX = *std::max_element(chunks.begin(), chunks.end());
        float *pS = zalloc<float>(SN), *pconv = zalloc<float>(CN);
        float *pscr = poison<float>(static_cast<std::size_t>(CMAX) * SCR), *pbeta = poison<float>(static_cast<std::size_t>(CMAX) * H);
        float *pO = poison<float>(static_cast<std::size_t>(CMAX) * CH), *pY = poison<float>(static_cast<std::size_t>(T) * CH);
        unsigned* pflag = zalloc<unsigned>(1);
        bool states = true, inter = true;
        unsigned t0 = 0;
        for (std::size_t k = 0; k < chunks.size(); ++k) {
            const unsigned C = chunks[k];
            P::kda_prefill(kd, dq + static_cast<std::size_t>(t0) * 3 * CH, df + static_cast<std::size_t>(t0) * CH, dg + static_cast<std::size_t>(t0) * CH,
                           db + static_cast<std::size_t>(t0) * H, pS, pconv, pscr, pbeta, pO, pY + static_cast<std::size_t>(t0) * CH, pflag, C, nullptr);
            if (!inf_test) {
                const auto s1 = down(pS, SN), s2 = down(rS_at[k], SN);
                const auto c1 = down(pconv, CN);
                states &= bits_eq(s1, s2) && all_finite(s1) && bits_eq(c1, down(rconv_at[k], CN)) && all_finite(c1);
                const auto a1 = down(pscr, static_cast<std::size_t>(C) * SCR), a2 = down(rSCR + static_cast<std::size_t>(t0) * SCR, static_cast<std::size_t>(C) * SCR);
                const auto b1 = down(pbeta, static_cast<std::size_t>(C) * H), b2 = down(rB + static_cast<std::size_t>(t0) * H, static_cast<std::size_t>(C) * H);
                const auto o1 = down(pO, static_cast<std::size_t>(C) * CH), o2 = down(rOall + static_cast<std::size_t>(t0) * CH, static_cast<std::size_t>(C) * CH);
                inter &= bits_eq(a1, a2) && bits_eq(b1, b2) && bits_eq(o1, o2) && all_finite(a1) && all_finite(b1) && all_finite(o1);
            }
            t0 += C;
        }
        std::string cl;
        for (unsigned x : chunks) cl += (cl.empty() ? "" : "/") + std::to_string(x);
        const std::string tag = "KDA " + kw.label + " H=" + std::to_string(H) + " T=" + std::to_string(T) + " chunks " + cl + " scan v" + std::to_string(sv);
        if (inf_test) {
            check(down(rflag, 1)[0] != 0 && down(pflag, 1)[0] != 0, tag + ": positive C2 (+inf v input sets the flag on decode and prefill)");
            continue;
        }
        const auto y1 = down(pY, static_cast<std::size_t>(T) * CH), y2 = down(rY, static_cast<std::size_t>(T) * CH);
        check(bits_eq(y1, y2) && all_finite(y1), tag + ": outputs y of every row == decode, finite");
        check(inter, tag + ": prep scratch (q, k, v, g) + beta and raw scan output o of every row == decode");
        check(states, tag + ": S + conv state after every chunk == decode (incl. the first), finite");
        check(down(rflag, 1)[0] == 0 && down(pflag, 1)[0] == 0, tag + ": no C2 flag on either path");
    }
    P::set_kda_scan_version(0);
}

void test_kda(const std::string& cases, std::mt19937& g) {
    constexpr unsigned H = 32, CH = H * kKdaDim;
    CaseDir c(cases + "/kda_l4_r0", {{"conv", "u16", 3 * CH * kConvK}, {"A_log", "f32", H}, {"dt_bias", "f32", CH}, {"o_norm", "u16", kKdaDim},
                                     {"qkv", "f32", 12 * 3 * CH}, {"f", "f32", 12 * CH}, {"gate", "f32", 12 * CH}, {"b", "f32", 12 * H}});
    KdaW real{"real L4 rank-0", H, c.v<std::uint16_t>("conv"), c.v<std::uint16_t>("o_norm"), c.v<float>("A_log"), c.v<float>("dt_bias"),
              c.v<float>("qkv"), c.v<float>("f"), c.v<float>("gate"), c.v<float>("b")};
    kda_case(real, {12, 1, 2, 3, 15, 16, 17, 64, 1000, 2048, 906}, g, true, false);
    free_all();
    for (unsigned h : {32U, 64U}) {
        const unsigned C = h * kKdaDim;
        KdaW syn{h == 32 ? "synthetic rank-1 stand-in" : "synthetic single-node", h, rnd_half(static_cast<std::size_t>(3) * C * kConvK, g, 0.5F, true),
                 rnd_half(kKdaDim, g, 1.0F, true), rnd(h, g, 1.0F), rnd(C, g, 1.0F), {}, {}, {}, {}};
        kda_case(syn, {5, 16, 300, 1, 2}, g, true, false);
        free_all();
        if (h == 32) {
            kda_case(syn, {3, 17}, g, false, true);
            free_all();
        }
    }
}

// ---------------------------------------------------------------- DSA
struct SlotBufs {
    D::DsaSlot s;
    unsigned cap;
    explicit SlotBufs(unsigned c)
        : s{poison<std::uint16_t>(static_cast<std::size_t>(c) * kKvLora), poison<std::uint16_t>(static_cast<std::size_t>(c) * kIdxDim),
            poison<std::uint16_t>(static_cast<std::size_t>(c) * kIdxDim), poison<float>(static_cast<std::size_t>(c / kKpool) * kIdxDim), c},
          cap(c) {}
};
// Guard: positions [T, cap) and pools [T/4, cap/4) still all-ones.
bool guards_intact(const SlotBufs& b, unsigned T) {
    auto ones16 = [](const std::vector<std::uint16_t>& v, std::size_t from) {
        for (std::size_t i = from; i < v.size(); ++i)
            if (v[i] != 0xFFFF) return false;
        return true;
    };
    const auto lat = down(b.s.latent, static_cast<std::size_t>(b.cap) * kKvLora), key = down(b.s.ikey, static_cast<std::size_t>(b.cap) * kIdxDim);
    const auto gt = down(b.s.igate, static_cast<std::size_t>(b.cap) * kIdxDim);
    const auto pl = down(reinterpret_cast<const std::uint32_t*>(b.s.pooled), static_cast<std::size_t>(b.cap / kKpool) * kIdxDim);
    bool ok = ones16(lat, static_cast<std::size_t>(T) * kKvLora) && ones16(key, static_cast<std::size_t>(T) * kIdxDim) &&
              ones16(gt, static_cast<std::size_t>(T) * kIdxDim);
    for (std::size_t i = static_cast<std::size_t>(T / kKpool) * kIdxDim; i < pl.size(); ++i) ok &= pl[i] == 0xFFFFFFFFU;
    return ok;
}
// Canonical shape of the list of the query at position t (arithmetic contract section 1.4): n = 4 * min(P, 512) + (t + 1) % 4 with
// P = (t + 1) / 4; selected pools as complete ascending 4-token groups (pool-aligned), all < 4P; then the exact tail 4P .. t.
bool list_ok(const std::uint32_t* tok, std::uint32_t n, unsigned t) {
    const unsigned Pt = (t + 1) / kKpool, k = Pt < kIdxPools ? Pt : kIdxPools, tail = (t + 1) % kKpool;
    if (n != kKpool * k + tail || n < 1 || n > P::kMaxTokens) return false;
    for (unsigned j = 0; j < k; ++j) {
        const std::uint32_t b = tok[kKpool * j];
        if (b % kKpool || b >= kKpool * Pt || (j && b <= tok[kKpool * j - 1])) return false;
        for (unsigned m = 1; m < kKpool; ++m)
            if (tok[kKpool * j + m] != b + m) return false;
    }
    for (unsigned m = 0; m < tail; ++m)
        if (tok[kKpool * k + m] != kKpool * Pt + m) return false;
    return true;
}

struct DsaFix {
    std::vector<std::uint16_t> wk, gate, wp, knw, knb, kvb;
    std::vector<float> ape;
    D::IndexerDev dw{};
};

// Host R13 intermediates (glm_host.cpp mla_attention expressions) for one row and head: p[n] and ob[512].
void host_r13(const std::vector<std::uint16_t>& latent_cache, const std::vector<std::uint32_t>& tok, const float* qt, std::vector<float>& p,
              std::vector<float>& ob) {
    const unsigned n = static_cast<unsigned>(tok.size());
    p.assign(n, 0.0F);
    for (unsigned t = 0; t < n; ++t) {
        const std::uint16_t* lat = &latent_cache[static_cast<std::size_t>(tok[t]) * kKvLora];
        p[t] = chain_dot([&](unsigned l) { return qt[l]; }, [&](unsigned l) { return bf16_to_float(lat[l]); }, kKvLora) * kMlaScale;
    }
    float m = p[0];
    for (unsigned t = 1; t < n; ++t) m = p[t] > m ? p[t] : m;
    for (unsigned t = 0; t < n; ++t) p[t] = declared_exp(p[t] - m);
    const float sum = tree_sum<256>([&](unsigned t) { return p[t]; }, n);
    for (unsigned t = 0; t < n; ++t) p[t] = p[t] / sum;
    ob.assign(kKvLora, 0.0F);
    for (unsigned l = 0; l < kKvLora; ++l)
        ob[l] = tree_fma<256>([&](unsigned t) { return p[t]; }, [&](unsigned t) { return bf16_to_float(latent_cache[static_cast<std::size_t>(tok[t]) * kKvLora + l]); }, n);
}

struct HeadCfg {
    const char* label;
    unsigned H, h0;  // heads [h0, h0 + H) of the full 64-head kv_b
};
const HeadCfg kHeadCfgs[] = {{"rank-0 heads", 32, 0}, {"rank-1 heads", 32, 32}, {"all heads", 64, 0}};

// Runs decode (per row) and prefill (per chunk) over T rows; compares everything for the rows in `check_rows`
// (all rows when empty) of the chunks listed in `query_chunks` (all chunks when empty).
void dsa_case(const DsaFix& fx, const std::string& label, unsigned T, const std::vector<unsigned>& chunks, const std::vector<float>& x,
              const std::vector<float>& lat, const std::vector<float>& qI, std::mt19937& g, unsigned query_from, const std::set<unsigned>& rep_rows) {
    const unsigned cap = (T + 4095) / 4096 * 4096, kcap = cap / kKpool;
    const unsigned CMAX = *std::max_element(chunks.begin(), chunks.end());
    float *dx = up(x), *dlat = up(lat), *dqI = up(qI);
    std::vector<std::uint16_t*> kvb_dev;
    for (const HeadCfg& hc : kHeadCfgs)
        kvb_dev.push_back(up(std::vector<std::uint16_t>(fx.kvb.begin() + static_cast<std::size_t>(hc.h0) * 2 * kMlaDim * kKvLora,
                                                        fx.kvb.begin() + static_cast<std::size_t>(hc.h0 + hc.H) * 2 * kMlaDim * kKvLora)));
    std::vector<float> q[3];
    float* dqm[3];
    for (int c = 0; c < 3; ++c) {
        q[c] = rnd(static_cast<std::size_t>(T - query_from) * kHeadCfgs[c].H * kMlaDim, g, 1.0F);
        dqm[c] = up(q[c]);
    }
    const unsigned NQ = T - query_from;
    // ---- decode reference
    SlotBufs rs(cap);
    D::DsaSlot* drs = up(std::vector<D::DsaSlot>{rs.s});
    unsigned* dt = zalloc<unsigned>(1);
    unsigned* rflag = zalloc<unsigned>(1);
    float *dks = zalloc<float>(kIdxDim), *dgs = zalloc<float>(kIdxDim), *dwh = zalloc<float>(kIdxHeads), *dsc = zalloc<float>(kcap);
    float* dqt = zalloc<float>(64 * kKvLora);
    std::uint32_t* dwork = zalloc<std::uint32_t>(kcap);
    std::uint32_t *rtok = poison<std::uint32_t>(static_cast<std::size_t>(NQ) * P::kMaxTokens), *rn = poison<std::uint32_t>(NQ);
    float* rO[3];
    for (int c = 0; c < 3; ++c) rO[c] = poison<float>(static_cast<std::size_t>(NQ) * kHeadCfgs[c].H * kMlaDim);
    std::map<unsigned, std::vector<float>> r_wh, r_qt;  // per representative / scored row
    std::map<unsigned, std::vector<std::uint32_t>> r_keys;
    for (unsigned t = 0; t < T; ++t) {
        hok(hipMemcpy(dt, &t, 4, hipMemcpyHostToDevice), "t");
        D::dsa_append(drs, dt, dlat + static_cast<std::size_t>(t) * kKvLora, dx + static_cast<std::size_t>(t) * kHidden, fx.dw, dks, dgs, rflag, 1, nullptr);
        if (t < query_from) continue;
        const unsigned qi = t - query_from;
        std::uint32_t* tk = rtok + static_cast<std::size_t>(qi) * P::kMaxTokens;
        D::indexer_select(rs.s, t, dqI + static_cast<std::size_t>(t) * kIdxHeads * kIdxDim, dx + static_cast<std::size_t>(t) * kHidden, fx.dw, dwh, dsc,
                          dwork, tk, rn + qi, nullptr);
        {
            const std::uint32_t nn = down(rn + qi, 1)[0];
            if (nn < 1 || nn > P::kMaxTokens || !list_ok(down(tk, nn).data(), nn, t)) die("decode reference list malformed at row " + std::to_string(t));
        }
        const unsigned Pt = (t + 1) / kKpool;
        if (Pt > kIdxPools && (rep_rows.count(t) || t % 13 == 0)) {
            r_keys[t] = down(dwork, Pt);
            r_wh[t] = down(dwh, kIdxHeads);
            if (!all_finite(r_wh[t])) die("decode wh non-finite at row " + std::to_string(t));
        }
        for (int c = 0; c < 3; ++c) {
            D::mla_attention(rs.s, tk, rn + qi, dqm[c] + static_cast<std::size_t>(qi) * kHeadCfgs[c].H * kMlaDim, kvb_dev[c], kHeadCfgs[c].H, dqt,
                             rO[c] + static_cast<std::size_t>(qi) * kHeadCfgs[c].H * kMlaDim, nullptr);
            if (c == 2 && rep_rows.count(t)) {
                r_qt[t] = down(dqt, 64 * kKvLora);
                if (!all_finite(r_qt[t])) die("decode absorbed q non-finite at row " + std::to_string(t));
            }
        }
    }
    // ---- prefill
    SlotBufs ps(cap);
    unsigned* pflag = zalloc<unsigned>(1);
    float *pkey = poison<float>(static_cast<std::size_t>(CMAX) * kIdxDim), *pgate = poison<float>(static_cast<std::size_t>(CMAX) * kIdxDim);
    float* pwh = poison<float>(static_cast<std::size_t>(CMAX) * kIdxHeads);
    std::uint32_t* pkeys = poison<std::uint32_t>(static_cast<std::size_t>(CMAX) * kcap);
    std::uint32_t *ptok = poison<std::uint32_t>(static_cast<std::size_t>(NQ) * P::kMaxTokens), *pn = poison<std::uint32_t>(NQ);
    std::uint32_t *ptok2 = poison<std::uint32_t>(static_cast<std::size_t>(NQ) * P::kMaxTokens), *pn2 = poison<std::uint32_t>(NQ);
    float* pwh_in = poison<float>(static_cast<std::size_t>(CMAX) * kIdxHeads);
    bool sel2_ok = true;
    float *pqt = poison<float>(static_cast<std::size_t>(CMAX) * 64 * kKvLora), *pe = poison<float>(static_cast<std::size_t>(CMAX) * 64 * P::kEStride);
    float* pob = poison<float>(static_cast<std::size_t>(CMAX) * 64 * kKvLora);
    float* pO[3];
    for (int c = 0; c < 3; ++c) pO[c] = poison<float>(static_cast<std::size_t>(NQ) * kHeadCfgs[c].H * kMlaDim);
    bool wh_ok = true, keys_ok = true, inter_ok = true, lists_ok = true;
    bool versions_ok[kMlaPairs];
    for (bool& v : versions_ok) v = true;
    std::string first_bad;
    std::vector<std::uint16_t> lat_host;  // decode latent cache, for the host R13 intermediates
    unsigned t0 = 0;
    for (unsigned C : chunks) {
        const float* xr = dx + static_cast<std::size_t>(t0) * kHidden;
        P::dense_rows(xr, fx.dw.wk, D::WType::F16, kHidden, 0, kIdxDim, pkey, C, nullptr);
        P::dense_rows(xr, fx.dw.gate, D::WType::F16, kHidden, 0, kIdxDim, pgate, C, nullptr);
        P::dsa_append_rows(ps.s, t0, dlat + static_cast<std::size_t>(t0) * kKvLora, pkey, pgate, fx.dw, pflag, C, nullptr);
        if (t0 + C > query_from) {
            const unsigned a = t0 >= query_from ? t0 : query_from, Cq = t0 + C - a, qa = a - query_from;
            P::dense_rows(dx + static_cast<std::size_t>(a) * kHidden, fx.dw.weights_proj, D::WType::F16, kHidden, 0, kIdxHeads, pwh, Cq, nullptr);
            hok(hipMemcpy(pwh_in, pwh, static_cast<std::size_t>(Cq) * kIdxHeads * 4, hipMemcpyDeviceToDevice), "wh copy");
            P::indexer_select_rows(ps.s, a, dqI + static_cast<std::size_t>(a) * kIdxHeads * kIdxDim, pwh, pkeys, kcap,
                                   ptok + static_cast<std::size_t>(qa) * P::kMaxTokens, pn + qa, Cq, nullptr);
            const bool scored = (a + Cq) / kKpool > kIdxPools;  // indexer_select_rows scaled wh in place
            const auto whh = down(pwh, static_cast<std::size_t>(Cq) * kIdxHeads);
            const auto kh = down(pkeys, static_cast<std::size_t>(Cq) * kcap);
            for (const auto& [t, ref] : r_keys) {
                if (t < a || t >= a + Cq) continue;
                const unsigned r = t - a, Pt = (t + 1) / kKpool;
                if (scored) wh_ok &= std::memcmp(whh.data() + static_cast<std::size_t>(r) * kIdxHeads, r_wh[t].data(), kIdxHeads * 4) == 0 &&
                                     all_finite(std::vector<float>(whh.begin() + static_cast<std::size_t>(r) * kIdxHeads, whh.begin() + static_cast<std::size_t>(r + 1) * kIdxHeads));
                else wh_ok = false;
                keys_ok &= std::equal(ref.begin(), ref.end(), kh.begin() + static_cast<std::size_t>(r) * kcap) && ref.size() == Pt;
            }
            const auto nh = down(pn + qa, Cq), th = down(ptok + static_cast<std::size_t>(qa) * P::kMaxTokens, static_cast<std::size_t>(Cq) * P::kMaxTokens);
            for (const auto& vv : {std::pair{2U, 1U}, std::pair{3U, 1U}, std::pair{2U, 2U}}) {  // explicit pairs == default (3/2)
                const auto keys_all = down(pkeys, static_cast<std::size_t>(Cq) * kcap);
                hok(hipMemcpy(pwh, pwh_in, static_cast<std::size_t>(Cq) * kIdxHeads * 4, hipMemcpyDeviceToDevice), "wh restore");
                P::indexer_select_rows_versions(ps.s, a, dqI + static_cast<std::size_t>(a) * kIdxHeads * kIdxDim, pwh, pkeys, kcap,
                                                ptok2 + static_cast<std::size_t>(qa) * P::kMaxTokens, pn2 + qa, Cq, nullptr, vv.first, vv.second);
                const auto n2 = down(pn2 + qa, Cq), t2 = down(ptok2 + static_cast<std::size_t>(qa) * P::kMaxTokens, static_cast<std::size_t>(Cq) * P::kMaxTokens);
                sel2_ok &= n2 == nh && keys_all == down(pkeys, keys_all.size()) && bits_eq(whh, down(pwh, whh.size()));
                for (unsigned r = 0; r < Cq && sel2_ok; ++r)
                    sel2_ok &= std::equal(th.begin() + static_cast<std::size_t>(r) * P::kMaxTokens, th.begin() + static_cast<std::size_t>(r) * P::kMaxTokens + nh[r],
                                          t2.begin() + static_cast<std::size_t>(r) * P::kMaxTokens);
            }
            for (unsigned r = 0; r < Cq; ++r)
                if (!list_ok(th.data() + static_cast<std::size_t>(r) * P::kMaxTokens, nh[r], a + r)) {
                    lists_ok = false;
                    if (first_bad.empty()) first_bad = " first malformed list at row " + std::to_string(a + r);
                }
            if (!lists_ok) break;  // never traverse a malformed list
            for (int c = 0; c < 3; ++c) {
                const unsigned H = kHeadCfgs[c].H;
                // Explicit paired gates in bounded batches: every queried row, both TP2 ranks and H=64.
                // Compare all intermediates and outputs; canonicalize only the probability layout.
                for (unsigned rb = 0; rb < Cq; rb += 32) {
                    const unsigned rows = std::min(32U, Cq - rb);
                    const auto* tk = ptok + static_cast<std::size_t>(qa + rb) * P::kMaxTokens;
                    const float* qm = dqm[c] + static_cast<std::size_t>(qa + rb) * H * kMlaDim;
                    float* out = pO[c] + static_cast<std::size_t>(qa + rb) * H * kMlaDim;
                    auto call = [&](unsigned sv, unsigned ov, unsigned gv = 1) {
                        P::mla_attention_rows_versions(ps.s, tk, pn + qa + rb, qm, kvb_dev[c], H,
                                                       pqt, pe, pob, out, rows, nullptr, sv, ov, gv);
                    };
                    call(3, 5);
                    const auto qt_ref = down(pqt, static_cast<std::size_t>(rows) * H * kKvLora);
                    const auto ob_ref = down(pob, static_cast<std::size_t>(rows) * H * kKvLora);
                    const auto e_ref = down(pe, static_cast<std::size_t>(rows) * H * P::kEStride);
                    const auto o_ref = down(out, static_cast<std::size_t>(rows) * H * kMlaDim);
                    const auto decode = down(rO[c] + static_cast<std::size_t>(qa + rb) * H * kMlaDim, o_ref.size());
                    for (unsigned v = 0; v < kMlaPairs; ++v) {
                        const unsigned sv = kMlaPair[v][0], ov = kMlaPair[v][1];
                        call(sv, ov, kMlaPair[v][2]);
                        bool same = bits_eq(qt_ref, down(pqt, qt_ref.size())) && bits_eq(ob_ref, down(pob, ob_ref.size())) &&
                                    bits_eq(o_ref, down(out, o_ref.size())) && bits_eq(o_ref, decode) && all_finite(o_ref);
                        const auto ep = down(pe, e_ref.size());
                        for (unsigned rr = 0; rr < rows; ++rr)
                            for (unsigned h = 0; h < H; ++h)
                                for (unsigned j = 0; j < nh[rb + rr]; ++j) {
                                    unsigned rev = 0;
                                    for (unsigned bit = 0; bit < 8; ++bit) rev |= ((j >> bit) & 1U) << (7 - bit);
                                    const unsigned lane = P::kObRounds * rev + j / 256;
                                    const std::size_t base = (static_cast<std::size_t>(rr) * H + h) * P::kEStride;
                                    same &= std::memcmp(&e_ref[base + (H == 32 ? lane : j)],
                                                        &ep[base + (H == 32 && (ov == 5 || ov == 7) ? lane : j)], 4) == 0;
                                }
                        versions_ok[v] &= same;
                    }
                }
                P::mla_attention_rows(ps.s, ptok + static_cast<std::size_t>(qa) * P::kMaxTokens, pn + qa, dqm[c] + static_cast<std::size_t>(qa) * H * kMlaDim,
                                      kvb_dev[c], H, pqt, pe, pob, pO[c] + static_cast<std::size_t>(qa) * H * kMlaDim, Cq, nullptr);
                if (c == 0 && P::mla_e_lane_major(H)) {  // o-bar v5 intermediates (32 heads): lane-major p, padding +0, o-bar
                    bool any0 = false;
                    for (unsigned t : rep_rows) any0 |= t >= a && t < a + Cq;
                    if (any0) {
                        lat_host = down(rs.s.latent, static_cast<std::size_t>(T) * kKvLora);
                        const auto qt0 = down(pqt, static_cast<std::size_t>(Cq) * H * kKvLora);
                        const auto e0 = down(pe, static_cast<std::size_t>(Cq) * H * P::kEStride), ob0 = down(pob, static_cast<std::size_t>(Cq) * H * kKvLora);
                        for (unsigned t : rep_rows) {
                            if (t < a || t >= a + Cq) continue;
                            const unsigned r = t - a;
                            const std::vector<std::uint32_t> tok(th.begin() + static_cast<std::size_t>(r) * P::kMaxTokens,
                                                                 th.begin() + static_cast<std::size_t>(r) * P::kMaxTokens + nh[r]);
                            for (unsigned h : {0U, 31U}) {
                                std::vector<float> p, ob;  // host R13 from the device qt of this config (absorb is checked on c = 2)
                                host_r13(lat_host, tok, qt0.data() + (static_cast<std::size_t>(r) * H + h) * kKvLora, p, ob);
                                const float* lane = e0.data() + (static_cast<std::size_t>(r) * H + h) * P::kEStride;
                                for (unsigned s = 0; s < P::kEStride; ++s) {
                                    const unsigned k = s / P::kObRounds, m = s % P::kObRounds;
                                    unsigned l = 0;
                                    for (unsigned bit = 0; bit < 8; ++bit) l |= ((k >> bit) & 1U) << (7 - bit);
                                    const unsigned tt = l + 256 * m;
                                    const float want = tt < p.size() ? p[tt] : 0.0F;  // padding must be +0 (bit pattern 0)
                                    inter_ok &= std::memcmp(&lane[s], &want, 4) == 0;
                                }
                                inter_ok &= std::memcmp(ob0.data() + (static_cast<std::size_t>(r) * H + h) * kKvLora, ob.data(), kKvLora * 4) == 0;
                                inter_ok &= all_finite(p) && all_finite(ob);
                            }
                        }
                    }
                }
                if (c != 2) continue;
                bool any = false;
                for (unsigned t : rep_rows) any |= t >= a && t < a + Cq;
                if (!any) continue;
                lat_host = down(rs.s.latent, static_cast<std::size_t>(T) * kKvLora);
                const auto qth = down(pqt, static_cast<std::size_t>(Cq) * 64 * kKvLora);
                const auto eh = down(pe, static_cast<std::size_t>(Cq) * 64 * P::kEStride), obh = down(pob, static_cast<std::size_t>(Cq) * 64 * kKvLora);
                for (unsigned t : rep_rows) {
                    if (t < a || t >= a + Cq) continue;
                    const unsigned r = t - a;
                    inter_ok &= std::memcmp(qth.data() + static_cast<std::size_t>(r) * 64 * kKvLora, r_qt[t].data(), 64 * kKvLora * 4) == 0 &&
                                all_finite(std::vector<float>(qth.begin() + static_cast<std::size_t>(r) * 64 * kKvLora, qth.begin() + static_cast<std::size_t>(r + 1) * 64 * kKvLora));
                    const std::vector<std::uint32_t> tok(th.begin() + static_cast<std::size_t>(r) * P::kMaxTokens,
                                                         th.begin() + static_cast<std::size_t>(r) * P::kMaxTokens + nh[r]);
                    for (unsigned h : {0U, 31U, 32U, 63U}) {
                        std::vector<float> p, ob;
                        host_r13(lat_host, tok, r_qt[t].data() + static_cast<std::size_t>(h) * kKvLora, p, ob);
                        inter_ok &= std::memcmp(eh.data() + (static_cast<std::size_t>(r) * 64 + h) * P::kEStride, p.data(), p.size() * 4) == 0;
                        inter_ok &= std::memcmp(obh.data() + (static_cast<std::size_t>(r) * 64 + h) * kKvLora, ob.data(), kKvLora * 4) == 0;
                        inter_ok &= all_finite(p) && all_finite(ob);
                    }
                }
            }
        }
        t0 += C;
    }
    const std::string tag = "DSA " + label + " T=" + std::to_string(T);
    check(lists_ok, tag + ": prefill token lists canonical-shaped (n = 4 min(P, 512) + (t+1) % 4, whole ascending pools, exact tail)" + first_bad);
    if (!lists_ok) return;
    const bool caches = down(ps.s.latent, static_cast<std::size_t>(T) * kKvLora) == down(rs.s.latent, static_cast<std::size_t>(T) * kKvLora) &&
                        down(ps.s.ikey, static_cast<std::size_t>(T) * kIdxDim) == down(rs.s.ikey, static_cast<std::size_t>(T) * kIdxDim) &&
                        down(ps.s.igate, static_cast<std::size_t>(T) * kIdxDim) == down(rs.s.igate, static_cast<std::size_t>(T) * kIdxDim) &&
                        bits_eq(down(ps.s.pooled, static_cast<std::size_t>(T / kKpool) * kIdxDim), down(rs.s.pooled, static_cast<std::size_t>(T / kKpool) * kIdxDim));
    check(caches && all_finite(down(ps.s.pooled, static_cast<std::size_t>(T / kKpool) * kIdxDim)),
          tag + ": latent / key / gate / pooled caches == decode dsa_append, pooled finite");
    check(guards_intact(ps, T) && guards_intact(rs, T), tag + ": cache guard regions beyond T untouched (both paths)");
    const auto rnh = down(rn, NQ), pnh = down(pn, NQ);
    const auto rth = down(rtok, static_cast<std::size_t>(NQ) * P::kMaxTokens), pth = down(ptok, static_cast<std::size_t>(NQ) * P::kMaxTokens);
    bool lists = true;
    unsigned bad_row = T;
    for (unsigned i = 0; i < NQ && lists; ++i) {
        if (!list_ok(rth.data() + static_cast<std::size_t>(i) * P::kMaxTokens, rnh[i], query_from + i) || rnh[i] != pnh[i] ||
            !std::equal(rth.begin() + static_cast<std::size_t>(i) * P::kMaxTokens, rth.begin() + static_cast<std::size_t>(i) * P::kMaxTokens + rnh[i],
                        pth.begin() + static_cast<std::size_t>(i) * P::kMaxTokens)) {
            lists = false;
            bad_row = query_from + i;
        }
    }
    check(lists, tag + ": token lists + n of rows " + std::to_string(query_from) + ".." + std::to_string(T - 1) + " == decode indexer_select" +
                     (bad_row < T ? " (first bad row " + std::to_string(bad_row) + ")" : std::string()));
    check(sel2_ok, tag + ": indexer scores/select 2/1, 3/1, 2/2 == default 3/2 (token lists, counts), keys and scaled wh bitwise, every chunk");
    check(wh_ok && keys_ok && !r_keys.empty(), tag + ": per-row scaled wh + ordered keys (" + std::to_string(r_keys.size()) + " scored rows) == decode");
    if (!rep_rows.empty()) check(inter_ok, tag + ": absorbed q (64 heads) == decode; probabilities + o-bar (heads 0/31/32/63) == host R13, finite");
    for (int c = 0; c < 3; ++c) {
        const std::size_t n = static_cast<std::size_t>(NQ) * kHeadCfgs[c].H * kMlaDim;
        const auto ro = down(rO[c], n), po = down(pO[c], n);
        std::size_t bad = 0;
        for (std::size_t i = 0; i < n; ++i) bad += std::memcmp(&ro[i], &po[i], 4) != 0;
        check(bad == 0 && all_finite(po), tag + ": MLA outputs, " + kHeadCfgs[c].label + " (H=" + std::to_string(kHeadCfgs[c].H) + "), == decode, finite (mismatches " +
                                              std::to_string(bad) + ")");
    }
    for (unsigned v = 0; v < kMlaPairs; ++v)
        check(versions_ok[v], tag + ": explicit scores/obar/gemm " + std::to_string(kMlaPair[v][0]) + "/" + std::to_string(kMlaPair[v][1]) +
                                  "/" + std::to_string(kMlaPair[v][2]) +
              " == 3/5 intermediates and decode outputs, all queried rows, H=32 ranks 0/1 and H=64");
    check(down(rflag, 1)[0] == 0 && down(pflag, 1)[0] == 0, tag + ": no C3/C4 flag on either path");
}

// R1 chains over rows: pf::dense_rows == dev::dense (the decode kernel) bit for bit on every GLM plain-weight shape and
// type, with row counts around the 64-row tile and o0 offsets; poisoned outputs beyond n must stay untouched.
void test_dense_rows(const std::string& cases, std::mt19937& g) {
    CaseDir c(cases + "/moe_l3", {{"router", "u16", static_cast<std::size_t>(kExperts) * kHidden}});
    const auto router = c.v<std::uint16_t>("router");
    struct DS {
        const char* what;
        unsigned out, in, o0, n;
        D::WType t;
        bool real;
    };
    const DS shapes[] = {{"router (real L3, F16) 288x4096", kExperts, kHidden, 0, kExperts, D::WType::F16, true},
                         {"KDA f_a/g_a BF16 128x4096", 128, kHidden, 0, 128, D::WType::BF16, false},
                         {"KDA f_b/g_b BF16 4096x128 (rank rows 4096..8191 of 8192)", 8192, 128, 4096, 4096, D::WType::BF16, false},
                         {"KDA b_proj BF16 32x4096 (rank rows 32..63 of 64)", 64, kHidden, 32, 32, D::WType::BF16, false},
                         {"indexer wk/gate F16 128x4096", 128, kHidden, 0, 128, D::WType::F16, false},
                         {"indexer weights_proj F16 32x4096", 32, kHidden, 0, 32, D::WType::F16, false},
                         {"F32 weights 100x256", 100, 256, 0, 100, D::WType::F32, false}};
    for (const DS& d : shapes) {
        const std::size_t wn = static_cast<std::size_t>(d.out) * d.in;
        void* dw = nullptr;
        if (d.real) {
            dw = up(router);
        } else if (d.t == D::WType::F32) {
            dw = up(rnd(wn, g, 0.05F));
        } else {
            dw = up(rnd_half(wn, g, 0.05F, d.t == D::WType::BF16));
        }
        bool ok = true;
        for (unsigned R : {1U, 3U, 63U, 64U, 65U, 1000U, 2048U}) {
            auto x = rnd(static_cast<std::size_t>(R) * d.in, g, 1.0F);
            if (R > 2) {  // signed zeros and a subnormal in the inputs
                x[0] = -0.0F;
                x[1] = 1e-40F;
            }
            float* dx = up(x);
            float *ya = poison<float>(static_cast<std::size_t>(R) * d.n + 64), *yb = poison<float>(static_cast<std::size_t>(R) * d.n + 64);
            D::dense(dx, dw, d.t, d.in, d.o0, d.n, ya, R, nullptr);
            P::dense_rows(dx, dw, d.t, d.in, d.o0, d.n, yb, R, nullptr);
            const auto a = down(ya, static_cast<std::size_t>(R) * d.n + 64), b = down(yb, static_cast<std::size_t>(R) * d.n + 64);
            const std::vector<float> body(b.begin(), b.begin() + static_cast<std::size_t>(R) * d.n);
            ok &= bits_eq(a, b) && all_finite(body);
            for (std::size_t i = static_cast<std::size_t>(R) * d.n; i < b.size(); ++i) {
                std::uint32_t u;
                std::memcpy(&u, &b[i], 4);
                ok &= u == 0xFFFFFFFFU;
            }
        }
        check(ok, std::string("R1 dense_rows == decode dense, ") + d.what + ", rows 1/3/63/64/65/1000/2048, guard intact");
        free_all();
    }
}

// mHC pre for many rows: pf::hc_pre_rows == dev::hc_pre bit for bit (F, mix, collapsed x), real L3 hc fn/base/scale;
// row 0 = the real case stream, other rows random.
void test_hc_rows(const std::string& cases, std::mt19937& g) {
    constexpr unsigned n = kHc * kHidden;
    CaseDir c(cases + "/hc_l3", {{"fn", "f32", static_cast<std::size_t>(kHcMix) * n}, {"base", "f32", kHcMix}, {"scale", "f32", 3}, {"X", "f32", n}});
    float *fn = up(c.v<float>("fn")), *base = up(c.v<float>("base")), *scale = up(c.v<float>("scale"));
    const auto X0 = c.v<float>("X");
    bool ok = true;
    for (unsigned R : {1U, 63U, 64U, 65U, 2048U}) {
        auto X = rnd(static_cast<std::size_t>(R) * n, g, 1.0F);
        std::copy(X0.begin(), X0.end(), X.begin());
        float* dX = up(X);
        float *Fa = poison<float>(static_cast<std::size_t>(R) * n), *Fb = poison<float>(static_cast<std::size_t>(R) * n);
        float *ma = poison<float>(static_cast<std::size_t>(R) * kHcMix), *mb = poison<float>(static_cast<std::size_t>(R) * kHcMix);
        float *xa = poison<float>(static_cast<std::size_t>(R) * kHidden), *xb = poison<float>(static_cast<std::size_t>(R) * kHidden);
        float* z = poison<float>(static_cast<std::size_t>(R) * kHcMix);
        D::hc_pre(dX, fn, base, scale, Fa, ma, xa, R, nullptr);
        P::hc_pre_rows(dX, fn, base, scale, Fb, z, mb, xb, R, nullptr);
        {  // the 24 chains directly: z == decode dense over the decode F
            float* zr = poison<float>(static_cast<std::size_t>(R) * kHcMix);
            D::dense(Fa, fn, D::WType::F32, n, 0, kHcMix, zr, R, nullptr);
            const auto z1 = down(z, static_cast<std::size_t>(R) * kHcMix);
            ok &= bits_eq(z1, down(zr, static_cast<std::size_t>(R) * kHcMix)) && all_finite(z1);
        }
        const auto f1 = down(Fb, static_cast<std::size_t>(R) * n), m1 = down(mb, static_cast<std::size_t>(R) * kHcMix), x1 = down(xb, static_cast<std::size_t>(R) * kHidden);
        ok &= bits_eq(f1, down(Fa, static_cast<std::size_t>(R) * n)) && bits_eq(m1, down(ma, static_cast<std::size_t>(R) * kHcMix)) &&
              bits_eq(x1, down(xa, static_cast<std::size_t>(R) * kHidden)) && all_finite(f1) && all_finite(m1) && all_finite(x1);
    }
    check(ok, "mHC pre rows (real L3 fn/base/scale): F, mix, collapsed x == decode hc_pre; z == decode dense chains, rows 1/63/64/65/2048, finite");
    free_all();
}

// V2 gate uses explicit versions, so STRIX_GLM_PF_HC cannot make the reference select v2.
// Keep norm squares finite even on the large rows. Exercise tiny/all-subnormal/all-zero
// norms as well as mixed magnitudes and both signs of zero within a single row (also R=1).
std::vector<float> hc_stress_input(unsigned rows, std::mt19937& g) {
    constexpr unsigned n = kHc * kHidden;
    auto x = rnd(static_cast<std::size_t>(rows) * n, g, 1.0F);
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned i = 0; i < n; ++i) {
            float& v = x[static_cast<std::size_t>(r) * n + i];
            switch (r % 6) {
                case 1: v = v * 1.0e15F; break;
                case 2: v = v * 1.0e-20F; break;
                case 3: {
                    const std::uint32_t u = (i % 2 ? 0x80000000U : 0U) | (1U + i % 0x007FFFFFU);
                    std::memcpy(&v, &u, sizeof(v));
                    break;
                }
                case 4: v = i % 2 ? -0.0F : 0.0F; break;
                default: break;
            }
            if (r % 6 == 0 || r % 6 == 5) {
                switch (i % 257) {
                    case 0: v = 1.0e15F; break;
                    case 1: v = -1.0e15F; break;
                    case 2: v = 1.0e-20F; break;
                    case 3: v = -0.0F; break;
                    case 4: v = 0.0F; break;
                    case 5: {
                        const std::uint32_t u = 0x00000001U;
                        std::memcpy(&v, &u, sizeof(v));
                        break;
                    }
                    case 6: {
                        const std::uint32_t u = 0x80012345U;
                        std::memcpy(&v, &u, sizeof(v));
                        break;
                    }
                    default: break;
                }
            }
        }
    }
    return x;
}
void test_hc_rows_v2(const std::string& cases, std::mt19937& g) {
    constexpr unsigned n = kHc * kHidden;
    constexpr std::size_t guard = 17;
    CaseDir c(cases + "/hc_l3", {{"fn", "f32", static_cast<std::size_t>(kHcMix) * n}, {"base", "f32", kHcMix},
                                 {"scale", "f32", 3}, {"X", "f32", n}});
    for (unsigned R : {1U, 2U, 7U, 63U, 64U, 65U, 127U, 128U, 129U, 1000U, 2048U}) {
        for (unsigned stress = 0; stress < 2; ++stress) {
            float *fn = up(c.v<float>("fn")), *base = up(c.v<float>("base")), *scale = up(c.v<float>("scale"));
            auto X = stress ? hc_stress_input(R, g) : rnd(static_cast<std::size_t>(R) * n, g, 1.0F);
            if (!stress) {
                const auto x0 = c.v<float>("X");
                std::copy(x0.begin(), x0.end(), X.begin());
            }
            float* dX = up(X);
            const std::size_t nf = static_cast<std::size_t>(R) * n, nz = static_cast<std::size_t>(R) * kHcMix;
            const std::size_t nx = static_cast<std::size_t>(R) * kHidden;
            float *fa = poison<float>(nf + guard), *fb = poison<float>(nf + guard);
            float *za = poison<float>(nz + guard), *zb = poison<float>(nz + guard);
            float *ma = poison<float>(nz + guard), *mb = poison<float>(nz + guard);
            float *xa = poison<float>(nx + guard), *xb = poison<float>(nx + guard);
            P::hc_pre_rows_v1(dX, fn, base, scale, fa, za, ma, xa, R, nullptr);
            P::hc_pre_rows_v2(dX, fn, base, scale, fb, zb, mb, xb, R, nullptr);
            bool ok = true;
            auto compare = [&](const float* a, const float* b, std::size_t count, const char* name) {
                const auto ha = down(a, count + guard), hb = down(b, count + guard);
                bool same = bits_eq(ha, hb);
                for (std::size_t i = 0; i < count; ++i) same &= is_finite(ha[i]) && is_finite(hb[i]);
                for (std::size_t i = count; i < count + guard; ++i) {
                    std::uint32_t ua = 0, ub = 0;
                    std::memcpy(&ua, &ha[i], sizeof(ua));
                    std::memcpy(&ub, &hb[i], sizeof(ub));
                    same &= ua == 0xFFFFFFFFU && ub == 0xFFFFFFFFU;
                }
                if (!same) {
                    std::printf("FAIL mHC v2 %s R=%u input=%s: bits/finite/guard mismatch\n", name, R, stress ? "stress" : "normal");
                    for (std::size_t i = 0; i < count + guard; ++i) {
                        std::uint32_t ua = 0, ub = 0;
                        std::memcpy(&ua, &ha[i], sizeof(ua));
                        std::memcpy(&ub, &hb[i], sizeof(ub));
                        if (ua != ub) {
                            std::printf("  first mismatch index=%zu old=%08x v2=%08x\n", i,
                                        static_cast<unsigned>(ua), static_cast<unsigned>(ub));
                            break;
                        }
                    }
                }
                ok &= same;
            };
            compare(fa, fb, nf, "F");
            compare(za, zb, nz, "z");
            compare(ma, mb, nz, "mix");
            compare(xa, xb, nx, "xc");
            check(ok, "mHC v2 == v1 bitwise F/z/mix/xc, finite, guard intact, rows=" + std::to_string(R) +
                      (stress ? " large/tiny/signed-zero/subnormal" : " real L3 + random"));
            free_all();  // Bound peak allocation across the larger row-count matrix.
        }
    }
    // Zero rows must not launch or dereference any pointer, including on the explicit hooks.
    P::hc_pre_rows_v1(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr);
    P::hc_pre_rows_v2(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr);
    hok(hipGetLastError(), "mHC zero rows");
}

void test_dsa(const std::string& cases, std::mt19937& g) {
    CaseDir c(cases + "/mla_l3_r0", {{"kv_b_proj", "u16", static_cast<std::size_t>(64) * 2 * kMlaDim * kKvLora}, {"indexer.wk", "u16", kIdxDim * kHidden},
                                     {"indexer.index_kpool_compress_gate", "u16", kIdxDim * kHidden}, {"indexer.weights_proj", "u16", kIdxHeads * kHidden},
                                     {"indexer.k_norm", "u16", kIdxDim}, {"indexer.k_norm.bias", "u16", kIdxDim},
                                     {"indexer.index_kpool_compress_ape", "f32", kKpool * kIdxDim}});
    DsaFix fx{c.v<std::uint16_t>("indexer.wk"), c.v<std::uint16_t>("indexer.index_kpool_compress_gate"), c.v<std::uint16_t>("indexer.weights_proj"),
              c.v<std::uint16_t>("indexer.k_norm"), c.v<std::uint16_t>("indexer.k_norm.bias"), c.v<std::uint16_t>("kv_b_proj"),
              c.v<float>("indexer.index_kpool_compress_ape"), {}};
    auto upload_w = [&]() { fx.dw = D::IndexerDev{up(fx.wk), up(fx.gate), up(fx.wp), up(fx.knw), up(fx.knb), up(fx.ape)}; };
    {  // 1) 6000 rows, every row queried
        upload_w();
        constexpr unsigned T = 6000;
        const std::vector<unsigned> chunks = {1, 2, 3, 4, 120, 1000, 917, 2048, 1, 1, 7, 1896};
        const auto x = rnd(static_cast<std::size_t>(T) * kHidden, g, 0.5F), lat = rnd(static_cast<std::size_t>(T) * kKvLora, g, 1.0F);
        auto qI = rnd(static_cast<std::size_t>(T) * kIdxHeads * kIdxDim, g, 1.0F);
        for (unsigned t = 4096; t < 4105; ++t) std::fill_n(qI.begin() + static_cast<std::size_t>(t) * kIdxHeads * kIdxDim, kIdxHeads * kIdxDim, 0.0F);
        dsa_case(fx, "6000 rows (pools <= 1500, all-tie rows 4096..4104)", T, chunks, x, lat, qI, g, 0,
                 {0, 1, 2, 3, 5, 10, 129, 1129, 2046, 2047, 2048, 2051, 2052, 4095, 4096, 4100, 5999});
        free_all();
    }
    {  // 2) max-P: 33,000 rows; x periodic with 12 rows -> pooled keys repeat every 3 pools -> score ties across all blocks
        upload_w();
        constexpr unsigned T = 33000, period = 12;
        std::vector<unsigned> chunks;
        for (unsigned s = 0; s < 32768; s += 2048) chunks.push_back(2048);
        chunks.push_back(136);
        chunks.push_back(96);
        const auto xp = rnd(static_cast<std::size_t>(period) * kHidden, g, 0.5F), lp = rnd(static_cast<std::size_t>(period) * kKvLora, g, 1.0F);
        std::vector<float> x(static_cast<std::size_t>(T) * kHidden), lat(static_cast<std::size_t>(T) * kKvLora);
        for (unsigned t = 0; t < T; ++t) {
            std::copy_n(xp.begin() + static_cast<std::size_t>(t % period) * kHidden, kHidden, x.begin() + static_cast<std::size_t>(t) * kHidden);
            std::copy_n(lp.begin() + static_cast<std::size_t>(t % period) * kKvLora, kKvLora, lat.begin() + static_cast<std::size_t>(t) * kKvLora);
        }
        auto qI = rnd(static_cast<std::size_t>(T) * kIdxHeads * kIdxDim, g, 1.0F);
        for (unsigned t = T - 8; t < T; ++t) std::fill_n(qI.begin() + static_cast<std::size_t>(t) * kIdxHeads * kIdxDim, kIdxHeads * kIdxDim, 0.0F);
        dsa_case(fx, "max-P 33000 rows (pooled keys periodic: ties across every 1024-pool block; P up to 8250; last 8 all-tie)", T, chunks, x, lat, qI, g,
                 T - 96, {T - 96, T - 50, T - 9, T - 1});
        free_all();
    }
    {  // 3) positive C3/C4: a latent that rounds to bf16 +inf, and a gate +inf (C3, and C4 through its pool)
        upload_w();
        constexpr unsigned T = 8;
        auto x = rnd(static_cast<std::size_t>(T) * kHidden, g, 0.5F);
        auto lat = rnd(static_cast<std::size_t>(T) * kKvLora, g, 1.0F);
        lat[static_cast<std::size_t>(5) * kKvLora + 3] = 3.4e38F;
        float* dx = up(x);
        float* dl = up(lat);
        SlotBufs rs(4096), ps(4096);
        D::DsaSlot* drs = up(std::vector<D::DsaSlot>{rs.s});
        unsigned *dt = zalloc<unsigned>(1), *rflag = zalloc<unsigned>(1), *pflag = zalloc<unsigned>(1);
        float *dks = zalloc<float>(kIdxDim), *dgs = zalloc<float>(kIdxDim), *pk = zalloc<float>(T * kIdxDim), *pg = zalloc<float>(T * kIdxDim);
        for (unsigned t = 0; t < T; ++t) {
            hok(hipMemcpy(dt, &t, 4, hipMemcpyHostToDevice), "t");
            D::dsa_append(drs, dt, dl + static_cast<std::size_t>(t) * kKvLora, dx + static_cast<std::size_t>(t) * kHidden, fx.dw, dks, dgs, rflag, 1, nullptr);
        }
        D::dense(dx, fx.dw.wk, D::WType::F16, kHidden, 0, kIdxDim, pk, T, nullptr);
        D::dense(dx, fx.dw.gate, D::WType::F16, kHidden, 0, kIdxDim, pg, T, nullptr);
        P::dsa_append_rows(ps.s, 0, dl, pk, pg, fx.dw, pflag, T, nullptr);
        check(down(rflag, 1)[0] != 0 && down(pflag, 1)[0] != 0, "DSA positive C3: a latent rounding to bf16 +inf sets the flag on decode and prefill");
        // C4 alone: positions 4..6 preseeded (bypassing the store) with a +inf bf16 gate in channel 9, then the finite
        // row 7 appended on each path: its store is finite (no C3); its pool 1 is non-finite -> C4 only.
        SlotBufs rs4(4096), ps4(4096);
        D::DsaSlot* drs4 = up(std::vector<D::DsaSlot>{rs4.s});
        std::vector<std::uint16_t> g3(static_cast<std::size_t>(3) * kIdxDim, 0x3F80), k3(static_cast<std::size_t>(3) * kIdxDim, 0x3F80);
        g3[9] = 0x7F80;
        for (const SlotBufs* sb : {&rs4, &ps4}) {
            hok(hipMemcpy(sb->s.igate + static_cast<std::size_t>(4) * kIdxDim, g3.data(), g3.size() * 2, hipMemcpyHostToDevice), "seed");
            hok(hipMemcpy(sb->s.ikey + static_cast<std::size_t>(4) * kIdxDim, k3.data(), k3.size() * 2, hipMemcpyHostToDevice), "seed");
        }
        unsigned *rf4 = zalloc<unsigned>(1), *pf4 = zalloc<unsigned>(1);
        const unsigned t7 = 7;
        hok(hipMemcpy(dt, &t7, 4, hipMemcpyHostToDevice), "t");
        D::dsa_append(drs4, dt, dl, dx, fx.dw, dks, dgs, rf4, 1, nullptr);  // row 0's finite inputs at position 7
        P::dsa_append_rows(ps4.s, 7, dl, pk, pg, fx.dw, pf4, 1, nullptr);
        check(down(rf4, 1)[0] != 0 && down(pf4, 1)[0] != 0, "DSA positive C4 alone: finite row 7 completing a pool with a preseeded +inf gate sets the flag on decode and prefill");
        free_all();
    }
    {  // Direct MLA stress: tile/list boundaries and IEEE values without the append finiteness gate.
        const std::vector<std::uint32_t> counts = {1, 31, 32, 33, 63, 64, 65, 255, 256, 257, 2048, P::kMaxTokens};
        const unsigned C = static_cast<unsigned>(counts.size()), cap = 4096;
        SlotBufs slot(cap);
        std::vector<std::uint32_t> tokens(static_cast<std::size_t>(C) * P::kMaxTokens);
        for (unsigned r = 0; r < C; ++r)
            for (unsigned j = 0; j < counts[r]; ++j) tokens[static_cast<std::size_t>(r) * P::kMaxTokens + j] = (17 * j) % cap;
        auto* tk = up(tokens);
        auto* nn = up(counts);
        for (unsigned H : {32U, 64U}) {
            auto* kv = up(std::vector<std::uint16_t>(fx.kvb.begin(), fx.kvb.begin() + static_cast<std::size_t>(H) * 2 * kMlaDim * kKvLora));
            auto qm = rnd(static_cast<std::size_t>(C) * H * kMlaDim, g, 1.0F);
            auto* q = up(qm);
            const std::size_t nq = static_cast<std::size_t>(C) * H * kKvLora;
            const std::size_t ne = static_cast<std::size_t>(C) * H * P::kEStride, no = static_cast<std::size_t>(C) * H * kMlaDim;
            float *qt = poison<float>(nq + 17), *ep = poison<float>(ne + 17), *ob = poison<float>(nq + 17), *out = poison<float>(no + 17);
            for (unsigned ieee = 0; ieee < 2; ++ieee) {
                std::vector<std::uint16_t> lat(static_cast<std::size_t>(cap) * kKvLora);
                const std::uint16_t finite[] = {0x3F80, 0xBF80, 0x0000, 0x8000, 0x0001, 0x8001, 0x3EAB, 0xBEAB};
                for (std::size_t i = 0; i < lat.size(); ++i) lat[i] = finite[(i + i / kKvLora) % 8];
                if (ieee) { lat[3] = 0x7F80; lat[9] = 0xFF80; lat[17 * kKvLora + 5] = 0x7FC1; }
                hok(hipMemcpy(slot.s.latent, lat.data(), lat.size() * 2, hipMemcpyHostToDevice), "MLA seed");
                auto call = [&](unsigned sv, unsigned ov, unsigned gv = 1) {
                    P::mla_attention_rows_versions(slot.s, tk, nn, q, kv, H, qt, ep, ob, out, C, nullptr, sv, ov, gv);
                };
                call(3, 5);
                const auto qr = down(qt, nq + 17), er = down(ep, ne + 17), obr = down(ob, nq + 17), oo = down(out, no + 17);
                bool same = true;
                auto guard_ok = [](const std::vector<float>& a, std::size_t count) {
                    bool intact = true;
                    for (std::size_t j = count; j < a.size(); ++j) {
                        std::uint32_t u;
                        std::memcpy(&u, &a[j], 4);
                        intact &= u == 0xFFFFFFFFU;
                    }
                    return intact;
                };
                same &= guard_ok(qr, nq) && guard_ok(er, ne) && guard_ok(obr, nq) && guard_ok(oo, no);
                for (unsigned v = 0; v < kMlaPairs; ++v) {
                    const unsigned sv = kMlaPair[v][0], ov = kMlaPair[v][1];
                    call(sv, ov, kMlaPair[v][2]);
                    const auto eo = down(ep, ne + 17);
                    same &= bits_eq(qr, down(qt, nq + 17)) && bits_eq(obr, down(ob, nq + 17)) && bits_eq(oo, down(out, no + 17));
                    for (unsigned r = 0; r < C; ++r)
                        for (unsigned h = 0; h < H; ++h)
                            for (unsigned j = 0; j < counts[r]; ++j) {
                                unsigned rev = 0;
                                for (unsigned bit = 0; bit < 8; ++bit) rev |= ((j >> bit) & 1U) << (7 - bit);
                                const unsigned lane = P::kObRounds * rev + j / 256;
                                const std::size_t base = (static_cast<std::size_t>(r) * H + h) * P::kEStride;
                                same &= std::memcmp(&er[base + (H == 32 ? lane : j)], &eo[base + (H == 32 && (ov == 5 || ov == 7) ? lane : j)], 4) == 0;
                            }
                    for (unsigned j = 0; j < 17; ++j) {
                        std::uint32_t u;
                        std::memcpy(&u, &eo[ne + j], 4);
                        same &= u == 0xFFFFFFFFU;
                    }
                }
                // Byte comparisons deliberately include NaN payloads and the output guard regions.
                check(same, "MLA explicit scores/obar/gemm (9 combinations incl. 5/7/2, 4/6/2) == 3/5/1 bitwise intermediates/output/guards H=" + std::to_string(H) +
                      (ieee ? " NaN/+Inf/-Inf latent" : " signed-zero/subnormal latent") + ", n=1..2051 tile boundaries");
            }
        }
        free_all();
    }
    P::mla_attention_rows_versions({}, nullptr, nullptr, nullptr, nullptr, 32, nullptr, nullptr, nullptr, nullptr, 0, nullptr, 4, 6);
    hok(hipGetLastError(), "MLA zero rows");
}

// ---------------------------------------------------------------- timing (informational, after a clean gate)
bool g_only_hc = false;  // STRIX_GLM_PF_TEST_ONLY=hc: dense/hc gates + the mHC timing only
void timing(const std::string& cases, std::mt19937& g) {
    CaseDir c(cases + "/kda_l4_r0", {{"conv", "u16", 3 * 32 * kKdaDim * kConvK}, {"A_log", "f32", 32}, {"dt_bias", "f32", 32 * kKdaDim}, {"o_norm", "u16", kKdaDim}});
    CaseDir m(cases + "/mla_l3_r0", {{"kv_b_proj", "u16", static_cast<std::size_t>(64) * 2 * kMlaDim * kKvLora}, {"indexer.wk", "u16", kIdxDim * kHidden},
                                     {"indexer.index_kpool_compress_gate", "u16", kIdxDim * kHidden}, {"indexer.weights_proj", "u16", kIdxHeads * kHidden},
                                     {"indexer.k_norm", "u16", kIdxDim}, {"indexer.k_norm.bias", "u16", kIdxDim},
                                     {"indexer.index_kpool_compress_ape", "f32", kKpool * kIdxDim}});
    constexpr unsigned H = 32, CH = H * kKdaDim, C = 2048;
    hipEvent_t ev[4];
    for (auto& e : ev) hok(hipEventCreate(&e), "ev");
    if (!g_only_hc) {
    {
        const D::KdaDev kd{H, up(c.v<std::uint16_t>("conv")), up(c.v<float>("A_log")), up(c.v<float>("dt_bias")), up(c.v<std::uint16_t>("o_norm"))};
        float *dq = up(rnd(static_cast<std::size_t>(C) * 3 * CH, g, 1.0F)), *df = up(rnd(static_cast<std::size_t>(C) * CH, g, 1.0F));
        float *dg = up(rnd(static_cast<std::size_t>(C) * CH, g, 1.0F)), *db = up(rnd(static_cast<std::size_t>(C) * H, g, 1.0F));
        float *S = zalloc<float>(static_cast<std::size_t>(H) * kKdaDim * kKdaDim), *cv = zalloc<float>(static_cast<std::size_t>(3) * CH * 3);
        float *scr = zalloc<float>(static_cast<std::size_t>(C) * H * 4 * kKdaDim), *bt = zalloc<float>(static_cast<std::size_t>(C) * H);
        float *o = zalloc<float>(static_cast<std::size_t>(C) * CH), *y = zalloc<float>(static_cast<std::size_t>(C) * CH);
        unsigned* fl = zalloc<unsigned>(1);
        for (int rep = 0; rep < 4; ++rep) {
            hok(hipEventRecord(ev[0], nullptr), "ev");
            P::kda_prefill(kd, dq, df, dg, db, S, cv, scr, bt, o, y, fl, C, nullptr);
            hok(hipEventRecord(ev[1], nullptr), "ev");
            if (rep) std::printf("TIME kda_prefill C=2048 H=32: %.3f ms = %.3f us/row/layer\n", ms_between(ev[0], ev[1]), ms_between(ev[0], ev[1]) * 1000.0 / C);
        }
    }
    free_all();
    {
        const auto kvb = m.v<std::uint16_t>("kv_b_proj");
        for (unsigned t0 : {0U, 6144U, 30720U}) {
            D::IndexerDev dw{up(m.v<std::uint16_t>("indexer.wk")), up(m.v<std::uint16_t>("indexer.index_kpool_compress_gate")),
                             up(m.v<std::uint16_t>("indexer.weights_proj")), up(m.v<std::uint16_t>("indexer.k_norm")),
                             up(m.v<std::uint16_t>("indexer.k_norm.bias")), up(m.v<float>("indexer.index_kpool_compress_ape"))};
            std::uint16_t* dkvb = up(std::vector<std::uint16_t>(kvb.begin(), kvb.begin() + static_cast<std::size_t>(H) * 2 * kMlaDim * kKvLora));
            const unsigned cap = t0 + C;
            SlotBufs ts(cap);
            float *k = zalloc<float>(static_cast<std::size_t>(C) * kIdxDim), *gt = zalloc<float>(static_cast<std::size_t>(C) * kIdxDim);
            float *dx = up(rnd(static_cast<std::size_t>(C) * kHidden, g, 0.5F)), *dl = up(rnd(static_cast<std::size_t>(C) * kKvLora, g, 1.0F));
            float *dqi = up(rnd(static_cast<std::size_t>(C) * kIdxHeads * kIdxDim, g, 1.0F)), *dqm = up(rnd(static_cast<std::size_t>(C) * H * kMlaDim, g, 1.0F));
            float* wh = zalloc<float>(static_cast<std::size_t>(C) * kIdxHeads);
            std::uint32_t *keys = zalloc<std::uint32_t>(static_cast<std::size_t>(C) * (cap / kKpool)), *tok = zalloc<std::uint32_t>(static_cast<std::size_t>(C) * P::kMaxTokens);
            std::uint32_t* n = zalloc<std::uint32_t>(C);
            float *qt = zalloc<float>(static_cast<std::size_t>(C) * H * kKvLora), *e = zalloc<float>(static_cast<std::size_t>(C) * H * P::kEStride);
            float *ob = zalloc<float>(static_cast<std::size_t>(C) * H * kKvLora), *o = zalloc<float>(static_cast<std::size_t>(C) * H * kMlaDim);
            unsigned* fl = zalloc<unsigned>(1);
            D::dense(dx, dw.wk, D::WType::F16, kHidden, 0, kIdxDim, k, C, nullptr);
            D::dense(dx, dw.gate, D::WType::F16, kHidden, 0, kIdxDim, gt, C, nullptr);
            for (unsigned s0 = 0; s0 < t0; s0 += C) P::dsa_append_rows(ts.s, s0, dl, k, gt, dw, fl, C, nullptr);  // filler
            for (int rep = 0; rep < 4; ++rep) {
                D::dense(dx, dw.weights_proj, D::WType::F16, kHidden, 0, kIdxHeads, wh, C, nullptr);
                hok(hipEventRecord(ev[0], nullptr), "ev");
                P::dsa_append_rows(ts.s, t0, dl, k, gt, dw, fl, C, nullptr);
                hok(hipEventRecord(ev[1], nullptr), "ev");
                P::indexer_select_rows(ts.s, t0, dqi, wh, keys, cap / kKpool, tok, n, C, nullptr);
                hok(hipEventRecord(ev[2], nullptr), "ev");
                P::mla_attention_rows(ts.s, tok, n, dqm, dkvb, H, qt, e, ob, o, C, nullptr);
                hok(hipEventRecord(ev[3], nullptr), "ev");
                if (rep)
                    std::printf("TIME DSA t0=%u C=2048 H=32: append %.3f ms, indexer %.3f ms, MLA %.3f ms (%.2f us/row/layer)\n", t0, ms_between(ev[0], ev[1]),
                                ms_between(ev[1], ev[2]), ms_between(ev[2], ev[3]), ms_between(ev[0], ev[3]) * 1000.0 / C);
            }
            free_all();
        }
    }
    }
    {
        constexpr unsigned n = kHc * kHidden;
        CaseDir hc(cases + "/hc_l3", {{"fn", "f32", static_cast<std::size_t>(kHcMix) * n}, {"base", "f32", kHcMix},
                                      {"scale", "f32", 3}, {"X", "f32", n}});
        float *fn = up(hc.v<float>("fn")), *base = up(hc.v<float>("base")), *scale = up(hc.v<float>("scale"));
        float* X = up(rnd(static_cast<std::size_t>(C) * n, g, 1.0F));
        float* F = zalloc<float>(static_cast<std::size_t>(C) * n);
        float *z = zalloc<float>(static_cast<std::size_t>(C) * kHcMix), *mix = zalloc<float>(static_cast<std::size_t>(C) * kHcMix);
        float* xc = zalloc<float>(static_cast<std::size_t>(C) * kHidden);
        // Identical buffers and stream, explicit hooks: A=old, B=new, independent of env.
        auto call = [&](bool v2) {
            if (v2) P::hc_pre_rows_v2(X, fn, base, scale, F, z, mix, xc, C, nullptr);
            else P::hc_pre_rows_v1(X, fn, base, scale, F, z, mix, xc, C, nullptr);
        };
        call(false);
        call(true);
        hok(hipDeviceSynchronize(), "mHC warmup");
        std::vector<float> old_us, new_us;
        for (unsigned rep = 0; rep < 3; ++rep) {
            for (unsigned version = 0; version < 2; ++version) {
                hok(hipEventRecord(ev[0], nullptr), "ev");
                call(version == 1);
                hok(hipEventRecord(ev[1], nullptr), "ev");
                const float us = ms_between(ev[0], ev[1]) * 1000.0F;
                (version ? new_us : old_us).push_back(us);
                std::printf("TIME mHC %c C=2048 pair=%u: %.3f us/call = %.3f us/row\n",
                            version ? 'B' : 'A', rep + 1, static_cast<double>(us), static_cast<double>(us) / C);
            }
        }
        std::sort(old_us.begin(), old_us.end());
        std::sort(new_us.begin(), new_us.end());
        std::printf("TIME mHC C=2048 median A(old): %.3f us/call = %.3f us/row; B(v2): %.3f us/call = %.3f us/row\n",
                    static_cast<double>(old_us[1]), static_cast<double>(old_us[1]) / C,
                    static_cast<double>(new_us[1]), static_cast<double>(new_us[1]) / C);
        free_all();
    }
    for (auto& e : ev) hok(hipEventDestroy(e), "ev");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s CASES_DIR [--no-timing]\n", argv[0]);
        return 2;
    }
    const bool want_timing = !(argc > 2 && std::string(argv[2]) == "--no-timing");
    std::mt19937 g(20261025);
    test_dense_rows(argv[1], g);
    test_hc_rows(argv[1], g);
    test_hc_rows_v2(argv[1], g);
    if (const char* only = std::getenv("STRIX_GLM_PF_TEST_ONLY"); only != nullptr && std::string(only) == "hc") g_only_hc = true;
    if (!g_only_hc) {
        test_kda(argv[1], g);
        test_dsa(argv[1], g);
    }
    std::printf("checks: %d failures: %d\n", checks, failures);
    if (failures == 0 && want_timing) timing(argv[1], g);
    else if (want_timing) std::printf("timing skipped: gate failed\n");
    return failures ? 1 : 0;
}
