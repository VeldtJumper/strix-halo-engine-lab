// CPU tests of the H3 W8A8 convrot host reference (src/h3/gemm/ref.hpp).
//  1. the radix-4 rotation equals the explicit Hadamard product (fp64, exact) on random groups;
//  2. embedded goldens: FNV-1a hashes of every step (rotated row, codes, row scales, int32 accumulators, bf16 outputs)
//     produced by the python reference chain on splitmix64 inputs, and the hash of the reference silu over all bf16;
//  3. optional dumps (STRIX_H3_GEMM_DUMPS=<dir>, one sub-directory per case): every step bitwise against the
//     exact-rotation chain (B_*), and the mismatch counts against the verbatim chain (A_*, bf16 matmul rotation).
#include "h3/gemm/ref.hpp"
#include "g1_gate.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <string>

using namespace strix::h3::gemm;

namespace {

int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { ++failures; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

struct SplitMix {
    std::uint64_t s;
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31U);
    }
};
std::uint64_t fnv(const void* p, std::size_t n, std::uint64_t h = 0xcbf29ce484222325ULL) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 0x100000001b3ULL;
    return h;
}

void test_rotation_matches_matrix() {
    // H[i][j] = prod over base-4 digits of H4[i_d][j_d] / 16
    static const int h4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};
    SplitMix g{77};
    unsigned bad = 0;
    for (unsigned t = 0; t < 64; ++t) {
        std::uint16_t x[256], y[256];
        for (auto& v : x) {
            const std::uint64_t u = g.next();
            v = static_cast<std::uint16_t>(((u >> 63U) << 15U) | ((110U + (u >> 40U) % 30U) << 7U) | ((u >> 8U) & 127U));
        }
        CHECK(rotate_group256(x, y), "span flag on a 30-binade group");
        for (unsigned j = 0; j < 256; ++j) {
            double s = 0.0;
            for (unsigned i = 0; i < 256; ++i) {
                int sign = 1;
                for (unsigned d = 0, ii = i, jj = j; d < 4; ++d, ii /= 4, jj /= 4) sign *= h4[ii % 4][jj % 4];
                s += sign * static_cast<double>(bf16_to_f32(x[i]));  // exact: <= 30 + 16 bits
            }
            bad += f64_to_bf16(s / 16.0) != y[j];
        }
    }
    CHECK(bad == 0, "radix-4 rotation differs from the Hadamard product in %u values", bad);
}

// Fixed-point rotation (the device arithmetic) against the fp64 oracle: random spans 0..37 binades, subnormal inputs,
// zero groups, sparse groups, -0, and large magnitudes.
void test_fixed_rotation() {
    SplitMix g{4242};
    unsigned bad = 0, cases = 0;
    for (unsigned t = 0; t < 4000; ++t) {
        std::uint16_t x[256], a[256], b[256];
        const unsigned span = static_cast<unsigned>(g.next() % 38U);
        const unsigned lo = t % 7U == 0U ? 0U : static_cast<unsigned>(1U + g.next() % (250U - span));
        const unsigned density = 1U + static_cast<unsigned>(g.next() % 8U);
        for (auto& v : x) {
            const std::uint64_t u = g.next();
            if ((u & 7U) >= density) { v = (u & 8U) ? 0x8000U : 0U; continue; }
            const unsigned e = lo + static_cast<unsigned>((u >> 40U) % (span + 1U));
            v = static_cast<std::uint16_t>(((u >> 63U) << 15U) | ((e > 254U ? 254U : e) << 7U) | ((u >> 8U) & 127U));
        }
        if (t % 11U == 0U) for (auto& v : x) v &= 0x80ffU;  // tiny exponents incl. subnormals
        if (lo + span > 250U) continue;
        const bool exact = rotate_group256(x, a);
        CHECK(rotate_group256_fixed(x, b), "fixed rotation flushed a %u-binade group", span);
        if (!exact) continue;
        ++cases;
        for (unsigned i = 0; i < 256; ++i) { if (a[i] != b[i] && bad < 6) std::printf("  t %u span %u lo %u i %u fp64 %04x fixed %04x\n", t, span, lo, i, a[i], b[i]); bad += a[i] != b[i]; }
    }
    CHECK(bad == 0, "fixed-point rotation differs from fp64 in %u values (%u groups)", bad, cases);
    // flush report beyond the window
    std::uint16_t x[256] = {}, y[256];
    x[0] = 0x7f00U; x[1] = 0x0080U;  // 2^127 and 2^-126
    CHECK(!rotate_group256_fixed(x, y), "flush not reported");
}

// RNE carry into the next binade (r -> 256 in fixed_to_bf16): the review's group [255, 256, 0, ...] (first rotated value
// 31.9375 -> 32) and a sweep of near-binade-boundary groups, fixed point vs the fp64 oracle; counts the carry cases hit.
void test_rounding_carry() {
    std::uint16_t x[256] = {}, a[256], b[256];
    x[0] = 0x437fU; x[1] = 0x4380U;  // 255, 256
    CHECK(rotate_group256_fixed(x, b) && rotate_group256(x, a), "carry group flagged");
    CHECK(b[0] == 0x4200U && a[0] == 0x4200U, "carry group: first value %04x (fixed) %04x (fp64), expected 4200 (32)", b[0], a[0]);
    std::size_t bad = 0;
    for (unsigned i = 0; i < 256; ++i) bad += a[i] != b[i];
    CHECK(bad == 0, "carry group: %zu values differ from fp64", bad);
    // sweep: two or three nonzero values with mantissas near 0x7f / 0x00 over many exponents (incl. subnormal results)
    static const unsigned mant[] = {0x00U, 0x01U, 0x3fU, 0x40U, 0x7eU, 0x7fU};
    unsigned carries = 0, cases = 0;
    bad = 0;
    for (unsigned e0 = 1; e0 < 250; e0 += 3)
        for (unsigned d = 0; d < 10; ++d)
            for (unsigned m0 : mant)
                for (unsigned m1 : mant)
                    for (unsigned pos = 1; pos < 4; ++pos)
                        for (unsigned sg = 0; sg < 4; ++sg) {
                            std::uint16_t g[256] = {};
                            const unsigned e1 = e0 + d > 254U ? 254U : e0 + d;
                            g[0] = static_cast<std::uint16_t>(((sg & 1U) << 15U) | (e0 << 7U) | m0);
                            g[pos * pos * 17U % 256U] = static_cast<std::uint16_t>(((sg >> 1U) << 15U) | (e1 << 7U) | m1);
                            if (sg == 3U) g[255] = static_cast<std::uint16_t>((e0 << 7U) | m1);
                            if (!rotate_group256(g, a)) continue;
                            static_cast<void>(rotate_group256_fixed(g, b));
                            ++cases;
                            for (unsigned i = 0; i < 256; ++i) {
                                bad += a[i] != b[i];
                                carries += (a[i] & 0x7fU) == 0U && (a[i] & 0x7f80U) != 0U;  // results on a power of two
                            }
                        }
    CHECK(bad == 0, "carry sweep: %zu values differ from fp64 over %u groups", bad, cases);
    std::printf("rounding carry: regression group ok; sweep %u groups, %u power-of-two results, %zu differ\n", cases, carries, bad);
}

void test_conversions() {
    CHECK(f32_to_bf16(1.0F) == 0x3f80, "bf16(1)");
    CHECK(f32_to_bf16(bf16_to_f32(0x3f80) + 0x1p-8F) == 0x3f80, "tie to even down");
    CHECK(f32_to_bf16(1.0F + 3 * 0x1p-8F) == 0x3f82, "tie to even up");
    // fp64 -> bf16 must not double-round through fp32: 1 + 2^-8 + 2^-30 rounds up once, but via fp32 (1 + 2^-8) it would tie down.
    CHECK(f64_to_bf16(1.0 + 0x1p-8 + 0x1p-30) == 0x3f81, "f64 -> bf16 single rounding");
    CHECK(f64_to_bf16(-(1.0 + 0x1p-8)) == 0xbf80, "f64 -> bf16 negative tie");
    CHECK(f64_to_bf16(255.9999) == 0x4380, "carry into exponent");
}

struct Golden { std::uint32_t m, k, n, seed; std::uint64_t h[5]; };
// From the python chain (exact rotation, verbatim quantizer/int mm/epilogue).
const Golden kGoldens[] = {
    {37U, 768U, 48U, 1U, {0xca8215d85bfacda7ULL, 0x7b5245ec95b672dcULL, 0xcd15008b4d90d4f6ULL, 0xf0a2858a601f7cfeULL, 0xc3af16869d52401dULL}},
    {5U, 5376U, 16U, 2U, {0xeb3f45fe0284670cULL, 0x41f4268155fdce54ULL, 0xa9695c33c0205810ULL, 0x4e6a6eeb4a439229ULL, 0xa6cffce966ab9849ULL}},
};
constexpr std::uint64_t kSiluHash = 0x641a7bcd6a73c5a0ULL;

void test_goldens() {
    for (const auto& gd : kGoldens) {
        SplitMix g{gd.seed};
        std::vector<std::uint16_t> x(static_cast<std::size_t>(gd.m) * gd.k);
        for (auto& v : x) {
            const std::uint64_t u = g.next();
            v = static_cast<std::uint16_t>(((u >> 63U) << 15U) | ((119U + (u >> 40U) % 16U) << 7U) | ((u >> 8U) & 127U));
        }
        std::vector<std::int8_t> w(static_cast<std::size_t>(gd.n) * gd.k);
        for (auto& v : w) v = static_cast<std::int8_t>(static_cast<std::uint8_t>(g.next() & 255U));
        std::vector<float> ws(gd.n);
        for (auto& v : ws) { const std::uint32_t b = 0x3a000000U | static_cast<std::uint32_t>(g.next() & 0x7fffffU); std::memcpy(&v, &b, 4); }
        std::vector<std::uint16_t> xr(x.size());
        std::vector<std::int8_t> q(x.size());
        for (std::uint32_t r = 0; r < gd.m; ++r) rotate_quantize_row(x.data() + r * gd.k, gd.k, q.data() + r * gd.k, xr.data() + r * gd.k);
        const Projection p = project(x.data(), gd.m, gd.k, w.data(), ws.data(), gd.n);
        const std::uint64_t h[5] = {fnv(xr.data(), 2 * xr.size()), fnv(p.q.data(), p.q.size()), fnv(p.row_scale.data(), 4 * p.row_scale.size()),
            fnv(p.acc.data(), 4 * p.acc.size()), fnv(p.y.data(), 2 * p.y.size())};
        static const char* names[5] = {"rotated", "codes", "row scales", "accumulators", "outputs"};
        for (unsigned i = 0; i < 5; ++i)
            CHECK(h[i] == gd.h[i], "golden %ux%ux%u: %s hash %016llx != %016llx", gd.m, gd.k, gd.n, names[i],
                static_cast<unsigned long long>(h[i]), static_cast<unsigned long long>(gd.h[i]));
    }
    const auto& t = silu_table();
    CHECK(fnv(t.data(), 2 * t.size()) == kSiluHash, "silu table hash %016llx", static_cast<unsigned long long>(fnv(t.data(), 2 * t.size())));
}

template <class T> std::vector<T> load(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    const auto bytes = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<T> v(bytes / sizeof(T));
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
    return v;
}
std::map<std::string, std::vector<std::size_t>> shapes(const std::filesystem::path& p) {
    std::map<std::string, std::vector<std::size_t>> out;
    std::ifstream f(p);
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream is(line);
        std::string name;
        is >> name;
        std::size_t d;
        while (is >> d) out[name].push_back(d);
    }
    return out;
}
template <class T> std::size_t diff(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return ~std::size_t{0};
    std::size_t n = 0;
    for (std::size_t i = 0; i < a.size(); ++i) n += std::memcmp(&a[i], &b[i], sizeof(T)) != 0;
    return n;
}


std::set<std::string> seen;
using g1::RotGate;
using g1::rotation_gate;
using g1::flips_ok;

void test_dumps(const std::filesystem::path& root) {
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        const auto dir = entry.path();
        if (std::filesystem::exists(dir / "silu_bf16.bin")) {
            // the python table is indexed by bf16 bit pattern
            const auto tab = load<std::uint16_t>(dir / "silu_bf16.bin");
            std::size_t bad = 0;
            for (std::uint32_t b = 0; b < 65536U && tab.size() == 65536U; ++b) bad += tab[b] != silu_table()[b];
            CHECK(tab.size() == 65536U && bad == 0, "silu table: %zu of 65536 differ", bad);
            std::printf("dump silu: %zu of 65536 bf16 inputs differ\n", bad);
            seen.insert("dump/silu");
            continue;
        }
        if (!std::filesystem::exists(dir / "meta.txt")) continue;
        const auto sh = shapes(dir / "meta.txt");
        const std::uint32_t m = static_cast<std::uint32_t>(sh.at("x")[0]), k = static_cast<std::uint32_t>(sh.at("x")[1]);
        const std::uint32_t n = static_cast<std::uint32_t>(sh.at("w")[0]);
        const auto x = load<std::uint16_t>(dir / "x.bin");
        const auto w = load<std::int8_t>(dir / "w.bin");
        const auto ws = load<float>(dir / "ws.bin");
        std::vector<std::uint16_t> xr(x.size());
        std::vector<std::int8_t> q(x.size());
        for (std::uint32_t r = 0; r < m; ++r) rotate_quantize_row(x.data() + std::size_t{r} * k, k, q.data() + std::size_t{r} * k, xr.data() + std::size_t{r} * k);
        const Projection p = project(x.data(), m, k, w.data(), ws.data(), n);
        const std::string name = dir.filename().string();
        const bool g1_only = std::filesystem::exists(dir / "g1_only");
        std::printf("dump %s (M %u K %u N %u, inexact groups %u):\n", name.c_str(), m, k, n, p.inexact_groups);
        for (const char* pre : {"B_", "A_"}) {
            const std::size_t d[5] = {diff(xr, load<std::uint16_t>(dir / (std::string(pre) + "xrot.bin"))),
                diff(p.q, load<std::int8_t>(dir / (std::string(pre) + "q.bin"))), diff(p.row_scale, load<float>(dir / (std::string(pre) + "xs.bin"))),
                diff(p.acc, load<std::int32_t>(dir / (std::string(pre) + "acc.bin"))), diff(p.y, load<std::uint16_t>(dir / (std::string(pre) + "out.bin")))};
            std::printf("  vs %s rotated %zu / %zu, codes %zu, row scales %zu / %u, acc %zu / %zu, out %zu\n", pre[0] == 'B' ? "exact-rotation chain"
                : "verbatim chain     ", d[0], x.size(), d[1], d[2], m, d[3], p.acc.size(), d[4]);
            if (pre[0] == 'B') {
                // g1_only cases have groups beyond the exact window: our rotation is exact within 2^-47 of the group max
                // (smaller values flushed and counted), so the fp64 exact chain is reported, not gated
                if (!g1_only)
                    for (unsigned i = 0; i < 5; ++i) CHECK(d[i] == 0, "%s: step %u differs from the exact-rotation chain in %zu values", name.c_str(), i, d[i]);
            } else {
                if (g1_only) {  // steps 2-8 from the verbatim chain's rotated rows are bitwise
                    const auto xa = load<std::uint16_t>(dir / "A_xrot.bin");
                    std::vector<std::int8_t> qa(xa.size());
                    std::vector<float> sa(m);
                    for (std::uint32_t r = 0; r < m; ++r) sa[r] = quantize_rotated_row(xa.data() + std::size_t{r} * k, k, qa.data() + std::size_t{r} * k);
                    std::vector<std::uint16_t> ya(std::size_t{m} * n);
                    for (std::uint32_t r = 0; r < m; ++r)
                        for (std::uint32_t c = 0; c < n; ++c)
                            ya[std::size_t{r} * n + c] = epilogue(dot_i8(qa.data() + std::size_t{r} * k, w.data() + std::size_t{c} * k, k), sa[r], ws[c]);
                    const std::size_t e[3] = {diff(qa, load<std::int8_t>(dir / "A_q.bin")), diff(sa, load<float>(dir / "A_xs.bin")), diff(ya, load<std::uint16_t>(dir / "A_out.bin"))};
                    std::printf("  from the verbatim chain's rotated rows: codes %zu, row scales %zu, out %zu\n", e[0], e[1], e[2]);
                    CHECK(e[0] == 0 && e[1] == 0 && e[2] == 0, "%s: steps 2-8 from the verbatim rotation differ", name.c_str());
                }
                const RotGate gt = rotation_gate(x, xr, load<std::uint16_t>(dir / "A_xrot.bin"));
                std::printf("  G1 vs verbatim chain: rotation max %u ulp, %zu values > 1 ulp, worst / fp32-order bound %.3f; code flips %zu (budget %zu)\n",
                    gt.max_ulp, gt.over_1ulp, gt.worst, d[1], (q.size() + 500000U) / 250000U);
                CHECK(gt.ok && flips_ok(d[1], q.size()), "%s: G1 rotation gate (worst %.3f, %zu code flips)", name.c_str(), gt.worst, d[1]);
            }
        }
        seen.insert("dump/" + name);
        if (sh.count("h")) {  // swiglu of the synthesized fc1 output equals the python h
            const auto f1 = load<std::uint16_t>(dir / "fc1out.bin");
            const auto h = load<std::uint16_t>(dir / "h.bin");
            const std::size_t half = sh.at("fc1out")[1] / 2, rows = sh.at("fc1out")[0];
            std::size_t bad = 0;
            for (std::size_t r = 0; r < rows; ++r)
                for (std::size_t j = 0; j < half; ++j) bad += swiglu(f1[r * 2 * half + j], f1[r * 2 * half + half + j]) != h[r * half + j];
            CHECK(bad == 0, "swiglu differs in %zu values", bad);
            std::printf("  swiglu: %zu of %zu differ\n", bad, rows * half);
        }
    }
}


// Independent quantiser fixture set: <root>/K<k>/<family>/{x,x_rot,row_scale,row_scale_bf16,q,w_q,w_scale,acc,out}.bin.
// Gate: steps 2-8 from the set's x_rot are bitwise (codes, fp32 scales, acc, out); from x through our exact rotation, the
// rotated values / codes / outputs that differ (the reference's matmul order) are reported.
void test_quant_set(const std::filesystem::path& root) {
    for (const std::uint32_t k : {5376U, 14336U}) {
        const auto kdir = root / ("K" + std::to_string(k));
        if (!std::filesystem::is_directory(kdir)) continue;
        for (const auto& entry : std::filesystem::directory_iterator(kdir)) {
            const auto dir = entry.path();
            const auto x = load<std::uint16_t>(dir / "x.bin"), xr_ref = load<std::uint16_t>(dir / "x_rot.bin");
            const auto w = load<std::int8_t>(dir / "w_q.bin");
            const auto ws = load<float>(dir / "w_scale.bin");
            const auto s_ref = load<float>(dir / "row_scale.bin");
            const auto sb_ref = load<std::uint16_t>(dir / "row_scale_bf16.bin");
            const auto q_ref = load<std::int8_t>(dir / "q.bin");
            const auto acc_ref = load<std::int32_t>(dir / "acc.bin");
            const auto y_ref = load<std::uint16_t>(dir / "out.bin");
            const auto m = static_cast<std::uint32_t>(x.size() / k), n = static_cast<std::uint32_t>(ws.size());
            // (a) from the set's rotated rows
            std::vector<std::int8_t> q(x.size());
            std::vector<float> s(m);
            std::size_t dsb = 0;
            for (std::uint32_t r = 0; r < m; ++r) {
                s[r] = quantize_rotated_row(xr_ref.data() + std::size_t{r} * k, k, q.data() + std::size_t{r} * k);
                dsb += f32_to_bf16(s[r]) != sb_ref[r];
            }
            std::vector<std::int32_t> acc(std::size_t{m} * n);
            std::vector<std::uint16_t> y(acc.size());
            for (std::uint32_t r = 0; r < m; ++r)
                for (std::uint32_t c = 0; c < n; ++c) {
                    acc[std::size_t{r} * n + c] = dot_i8(q.data() + std::size_t{r} * k, w.data() + std::size_t{c} * k, k);
                    y[std::size_t{r} * n + c] = epilogue(acc[std::size_t{r} * n + c], s[r], ws[c]);
                }
            const std::size_t a[5] = {diff(s, s_ref), dsb, diff(q, q_ref), diff(acc, acc_ref), diff(y, y_ref)};
            // (b) the whole chain from x (our exact rotation)
            std::vector<std::uint16_t> xr(x.size());
            std::vector<std::int8_t> q2(x.size());
            for (std::uint32_t r = 0; r < m; ++r) rotate_quantize_row(x.data() + std::size_t{r} * k, k, q2.data() + std::size_t{r} * k, xr.data() + std::size_t{r} * k);
            const Projection p = project(x.data(), m, k, w.data(), ws.data(), n);
            const std::size_t b[5] = {diff(xr, xr_ref), diff(p.row_scale, s_ref), diff(p.q, q_ref), diff(p.acc, acc_ref), diff(p.y, y_ref)};
            const std::string name = "K" + std::to_string(k) + "/" + dir.filename().string();
            std::printf("quant set %-22s M %3u N %u: from x_rot: scales %zu, bf16 scales %zu, codes %zu, acc %zu, out %zu | from x: rotated %zu / %zu, scales %zu, codes %zu, acc %zu, out %zu\n",
                name.c_str(), m, n, a[0], a[1], a[2], a[3], a[4], b[0], x.size(), b[1], b[2], b[3], b[4]);
            CHECK(a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0 && a[4] == 0 && !x.empty(), "quant set %s: steps 2-8 differ", name.c_str());
            const RotGate gt = rotation_gate(x, xr, xr_ref);
            CHECK(gt.ok && flips_ok(b[2], q.size()), "quant set %s: G1 rotation gate (worst %.3f, %zu code flips)", name.c_str(), gt.worst, b[2]);
            std::printf("  G1: rotation max %u ulp, %zu values > 1 ulp, worst / fp32-order bound %.3f; code flips %zu (budget %zu)\n", gt.max_ulp,
                gt.over_1ulp, gt.worst, b[2], (q.size() + 500000U) / 250000U);
            seen.insert("quant/" + name);
        }
    }
}

}  // namespace

// Qualification mode (--qualify): the dumps and the quantiser fixture set must be given and complete; a skip is a failure.
const char* const kDumpCases[] = {"qkv_real", "out_real", "fc1_real", "fc2_real", "random_wide", "fc2_wide_synth", "silu"};
const char* const kQuantFamilies[] = {"normal", "heavy", "edge"};

int main(int argc, char** argv) {
    bool qualify = false;
    std::vector<const char*> pos;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--qualify") qualify = true;
        else pos.push_back(argv[i]);
    }
    test_conversions();
    test_rotation_matches_matrix();
    test_fixed_rotation();
    test_rounding_carry();
    test_goldens();
    const char* env = std::getenv("STRIX_H3_GEMM_DUMPS");
    const char* dumps = pos.size() > 0 ? pos[0] : env;
    if (dumps && std::filesystem::is_directory(dumps)) test_dumps(dumps);
    else std::printf("dumps: skipped (pass a directory or set STRIX_H3_GEMM_DUMPS)\n");
    const char* qset = pos.size() > 1 ? pos[1] : std::getenv("STRIX_H3_QUANT_SET");
    if (qset && std::filesystem::is_directory(qset)) test_quant_set(qset);
    else std::printf("quant set: skipped (second argument or STRIX_H3_QUANT_SET)\n");
    if (qualify) {
        std::vector<std::string> need;
        for (const char* c : kDumpCases) need.push_back(std::string("dump/") + c);
        for (const char* f : kQuantFamilies) {
            need.push_back(std::string("quant/K5376/") + f);
            need.push_back(std::string("quant/K14336/") + f);
        }
        need.push_back("quant/K5376/real_b0_qkv_in");
        need.push_back("quant/K14336/real_b0_fc2_in");
        std::size_t present = 0;
        for (const auto& nm : need) {
            present += seen.count(nm);
            CHECK(seen.count(nm) == 1, "qualification: %s missing", nm.c_str());
        }
        std::printf("qualification: %zu of %zu required cases present\n", present, need.size());
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
