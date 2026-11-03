// CPU tests of the FirstBlockCache host mirror (src/h3/levers/fbc/fbc_ref.hpp): decision arithmetic (single bf16
// rounding of the means, 1e-8 floor, bf16 quotient, <=, non-finite), fp64 -> bf16 rounding vs brute force,
// independent full-buffer vs two-rank decisions (incl. rank swap), the state machine (cold, reuse keeps the anchor,
// refresh, non-finite recovery, reset / shape change), API validation, adversarial magnitudes, and the reference's
// own decisions on boundary fixtures (tools/h3/levers/fbc/capture_fbc_ref.py).
// usage: strix-h3-levers-fbc-tests [--fixtures <dir>]   (with --fixtures every case must match)
#include "h3/levers/fbc/fbc_ref.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace strix::h3::levers::fbc;
namespace g = strix::h3::gemm;

namespace {
int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { ++failures; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

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
template <class F> bool throws(F f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}

void test_f64_to_bf16() {
    Rng r{77};
    std::size_t bad = 0;
    for (int i = 0; i < 200000; ++i) {
        const int e = static_cast<int>(r.next() % 260U) - 130;
        const double x = std::ldexp(static_cast<double>(r.next() >> 11U) * 0x1p-53, e) * ((r.next() & 1U) ? -1.0 : 1.0);
        const std::uint16_t got = f64_to_bf16(x);
        // brute force: the two bf16 neighbours around fp32-truncated x, nearest in exact (double) arithmetic, ties even
        std::uint16_t lo = g::f32_to_bf16(static_cast<float>(x));
        double best = INFINITY;
        std::uint16_t want = lo;
        for (int d = -2; d <= 2; ++d) {
            const std::uint16_t c = static_cast<std::uint16_t>(lo + d);
            if (std::isnan(g::bf16_to_f32(c))) continue;
            const double err = std::fabs(static_cast<double>(g::bf16_to_f32(c)) - x);
            if (err < best || (err == best && (c & 1U) == 0U)) { best = err; want = c; }
        }
        bad += got != want && !(std::isinf(g::bf16_to_f32(got)));
    }
    // exact midpoints between bf16 neighbours (ties to even) and one-ulp-of-fp64 perturbations of them
    for (std::uint16_t b = 0x3d00U; b < 0x3e00U; ++b) {
        const double m = (static_cast<double>(g::bf16_to_f32(b)) + static_cast<double>(g::bf16_to_f32(static_cast<std::uint16_t>(b + 1U)))) / 2.0;
        bad += f64_to_bf16(m) != ((b & 1U) ? static_cast<std::uint16_t>(b + 1U) : b);
        bad += f64_to_bf16(std::nextafter(m, 0.0)) != b;
        bad += f64_to_bf16(std::nextafter(m, 1.0)) != static_cast<std::uint16_t>(b + 1U);
    }
    std::printf("f64_to_bf16: %zu mismatches (random + ties + one-ulp neighbours of ties)\n", bad);
    CHECK(bad == 0, "f64_to_bf16");
}

void test_decide() {
    // mean 0.08 exactly -> bf16 0.0800781 > 0.08: no reuse (the score is the bf16 value)
    Decision d = decide({Sums{8.0, 100.0}}, 100, 0.08);
    CHECK(!d.reuse && d.finite && d.score == g::bf16_to_f32(g::f32_to_bf16(0.08F)), "bf16 rounding of the mean: %g", static_cast<double>(d.score));
    d = decide({Sums{7.0, 100.0}}, 100, 0.08);
    CHECK(d.reuse, "0.07 reuses");
    d = decide({Sums{8.0, 100.0}}, 100, 0.080078125);  // score == threshold -> reuse (<=)
    CHECK(d.reuse, "score == threshold reuses");
    d = decide({Sums{1.0, 0.0}}, 100, 0.08);  // D = 0 -> floor
    CHECK(!d.reuse && d.finite, "floor");
    d = decide({Sums{0.0, 0.0}}, 100, 0.08);
    CHECK(d.reuse && d.score == 0.0F, "0 / floor = 0 reuses");
    d = decide({Sums{NAN, 1.0}}, 100, 0.08);
    CHECK(!d.reuse && !d.finite, "NaN");
    d = decide({Sums{1.0, NAN}}, 100, 0.08);
    CHECK(!d.reuse && !d.finite, "NaN denominator");
    d = decide({Sums{INFINITY, 1.0}}, 100, 0.08);
    CHECK(!d.reuse && !d.finite, "inf");
    CHECK(throws([] { decide({Sums{}}, 100, -0.1); }) && throws([] { decide({Sums{}}, 100, NAN); }) && throws([] { decide({Sums{}}, 0, 0.08); }) &&
              throws([] { decide({}, 10, 0.08); }),
        "decide validation");
    std::printf("decide checked\n");
}

// Independent decisions: (a) full buffer split at cut, (b) two "ranks" holding [0, cut) and [cut, S) as their own
// buffers, computed in rank order and in swapped order; all must give the same sums bits and decision.
void test_two_rank(float mag_lo, float mag_hi, std::uint64_t seed) {
    constexpr std::uint32_t rows = 333U, w = 512U, cut = 128U;
    const std::size_t n = static_cast<std::size_t>(rows) * w;
    auto r = randbf(n, 1.0F, seed), rr = randbf(n, 1.0F, seed + 1);
    Rng sel{seed + 2};
    for (std::size_t i = 0; i < n; ++i) {  // adversarial: mixed magnitudes
        const float m = (sel.next() & 1U) ? mag_lo : mag_hi;
        r[i] = g::f32_to_bf16(g::bf16_to_f32(r[i]) * m);
        rr[i] = g::f32_to_bf16(g::bf16_to_f32(rr[i]) * m * ((sel.next() & 7U) ? 1.0F : 1.03F));
    }
    const auto full = split_block_sums(chunk_sums_all(r, rr, rows, w), rows, cut);
    const std::vector<std::uint16_t> r0(r.begin(), r.begin() + cut * w), rr0(rr.begin(), rr.begin() + cut * w), r1(r.begin() + cut * w, r.end()),
        rr1(rr.begin() + cut * w, rr.end());
    // swapped: "rank 0" computes block 1 first
    const auto c1 = chunk_sums_all(r1, rr1, rows - cut, w);
    const auto c0 = chunk_sums_all(r0, rr0, cut, w);
    const Sums b0 = block_sums(c0, 0, c0.size()), b1 = block_sums(c1, 0, c1.size());
    const Decision df = decide(full, n, 0.08), dr = decide({b0, b1}, n, 0.08);
    const bool same = full.size() == 2 && full[0].n == b0.n && full[0].d == b0.d && full[1].n == b1.n && full[1].d == b1.d &&
                      std::memcmp(&df.score, &dr.score, 4) == 0 && df.reuse == dr.reuse;
    std::printf("two-rank vs full buffer (magnitudes %g / %g): score %g / %g, sums identical %d\n", static_cast<double>(mag_lo), static_cast<double>(mag_hi),
        static_cast<double>(df.score), static_cast<double>(dr.score), same ? 1 : 0);
    CHECK(same, "two-rank decision");
}

void test_state_machine() {
    constexpr std::uint32_t rows = 200U, w = 256U, cut = 64U;
    const std::size_t n = static_cast<std::size_t>(rows) * w;
    FbcHost f(0.08, w);
    const auto base = randbf(n, 1.0F, 5), delta = randbf(n, 0.5F, 6), noise = randbf(n, 0.01F, 7), big = randbf(n, 0.5F, 8), tailv = randbf(n, 2.0F, 9);
    const auto run = [&](const std::vector<std::uint16_t>& h0, const std::vector<std::uint16_t>& pert, Decision* d) {
        std::vector<std::uint16_t> h(n);
        for (std::size_t i = 0; i < n; ++i) h[i] = add_bf16(add_bf16(h0[i], delta[i]), pert[i]);
        const Action a = f.after_block0(h0, h, rows, cut, d);
        if (a != Action::kReuse) {
            for (std::size_t i = 0; i < n; ++i) h[i] = add_bf16(h[i], tailv[i]);
            if (a == Action::kFullCommit) f.commit(h);
        }
        return std::make_pair(a, h);
    };
    const std::vector<std::uint16_t> zero(n, 0);
    Decision d;
    auto c1 = run(base, zero, &d);
    CHECK(c1.first == Action::kFullCommit && f.ready(), "cold start = full + commit");
    auto c2 = run(base, noise, &d);
    CHECK(c2.first == Action::kReuse, "reuse expected, score %g", static_cast<double>(d.score));
    std::size_t bad = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint16_t h1_1 = add_bf16(add_bf16(base[i], delta[i]), 0), hl1 = add_bf16(h1_1, tailv[i]);
        const std::uint16_t h1_2 = add_bf16(add_bf16(base[i], delta[i]), noise[i]);
        bad += c2.second[i] != add_bf16(h1_2, sub_bf16(hl1, h1_1));
    }
    CHECK(bad == 0, "reuse output");
    const float s2 = d.score;
    CHECK(throws([&] { f.commit(c2.second); }), "commit after reuse rejected");
    auto c3 = run(base, noise, &d);
    CHECK(c3.first == Action::kReuse && d.score == s2, "reuse keeps the anchor");
    auto c4 = run(base, big, &d);
    CHECK(c4.first == Action::kFullCommit, "refresh expected, score %g", static_cast<double>(d.score));
    // non-finite: NaN in block 0's output -> kFullNoCommit, cache cleared; next call cold
    std::vector<std::uint16_t> nanp = zero;
    nanp[17] = 0x7fc0U;
    auto c5 = run(base, nanp, &d);
    CHECK(c5.first == Action::kFullNoCommit && !f.ready() && !d.finite, "non-finite -> full, no commit");
    CHECK(throws([&] { f.commit(c5.second); }), "commit after non-finite rejected");
    auto c6 = run(base, noise, &d);
    CHECK(c6.first == Action::kFullCommit && f.ready(), "after non-finite: cold again");
    // shape change without reset rejected; after reset accepted (cold)
    {
        std::vector<std::uint16_t> h0s(static_cast<std::size_t>(128U) * w, 0), hs(h0s.size(), 0x3f80U);
        CHECK(throws([&] { f.after_block0(h0s, hs, 128U, 64U, nullptr); }), "row change without reset rejected");
        f.reset();
        CHECK(f.after_block0(h0s, hs, 128U, 64U, nullptr) == Action::kFullCommit, "after reset: cold");
    }
    std::printf("state machine: cold, reuse (score %g), commit-after-reuse rejected, same anchor, refresh, non-finite recovery, reset; reuse mismatches %zu\n",
        static_cast<double>(s2), bad);
}

void test_validation() {
    constexpr std::uint32_t w = 256U;
    std::vector<std::uint16_t> h0(64U * w, 0), h(64U * w, 0), small(10, 0);
    int rej = 0;
    rej += throws([] { FbcHost f(NAN, 256U); });
    rej += throws([] { FbcHost f(-1.0, 256U); });
    rej += throws([] { FbcHost f(0.08, 0U); });
    FbcHost f(0.08, w);
    rej += throws([&] { f.after_block0(small, h, 64U, 64U, nullptr); });   // h0 size
    rej += throws([&] { f.after_block0(h0, small, 64U, 64U, nullptr); });  // h size
    rej += throws([&] { f.after_block0(h0, h, 64U, 0U, nullptr); });       // cut 0
    rej += throws([&] { f.after_block0(h0, h, 64U, 32U, nullptr); });      // cut unaligned
    rej += throws([&] { f.after_block0(h0, h, 64U, 128U, nullptr); });     // cut > rows
    rej += throws([&] { f.after_block0(h0, h, 0U, 64U, nullptr); });       // rows 0
    rej += throws([&] { f.commit(h); });                                    // commit before block 0
    std::vector<Sums> ch(3);
    rej += throws([&] { block_sums(ch, 2, 2); });
    rej += throws([&] { block_sums(ch, 4, 0); });
    rej += throws([&] { split_block_sums(ch, 200U, 64U); });  // 200 rows = 4 chunks != 3
    rej += throws([&] { split_block_sums(ch, 150U, 160U); });
    rej += throws([] { checked_elems(1ULL << 40U, 1ULL << 40U); });
    rej += throws([] { chunk_sums_host(nullptr, nullptr, 1U, 1U, 1U); });
    std::printf("validation: %d of 16 invalid calls rejected\n", rej);
    CHECK(rej == 16, "validation");
    // rows boundary (chunk counts use rows + 63): UINT32_MAX - 63 accepted, one more rejected, 0 rejected
    const bool ok_max = !throws([] { check_rows(kMaxRows); });
    const bool rej_rows = throws([] { check_rows(static_cast<std::uint64_t>(kMaxRows) + 1U); }) && throws([] { check_rows(0U); }) &&
                          throws([] { checked_elems(static_cast<std::uint64_t>(kMaxRows) + 1U, 1U); });
    // phase: after_block0 while the previous call awaits its commit -> rejected; reset() abandons it explicitly
    FbcHost g(0.08, w);
    std::vector<std::uint16_t> hh(64U * w, 0x3f80U);
    const bool first = g.after_block0(h0, hh, 64U, 64U, nullptr) == Action::kFullCommit;
    const bool rej_phase = throws([&] { g.after_block0(h0, hh, 64U, 64U, nullptr); });
    g.reset();
    const bool after_reset = g.after_block0(h0, hh, 64U, 64U, nullptr) == Action::kFullCommit;
    std::printf("rows bound: max accepted %d, max + 1 / 0 rejected %d; after_block0 while awaiting commit rejected %d, reset abandons %d\n", ok_max ? 1 : 0,
        rej_rows ? 1 : 0, rej_phase ? 1 : 0, (first && after_reset) ? 1 : 0);
    CHECK(ok_max && rej_rows && first && rej_phase && after_reset, "rows bound / phase");
}

std::vector<std::uint16_t> load(const std::string& p, std::size_t n) {
    std::ifstream f(p, std::ios::binary);
    std::vector<std::uint16_t> v(n);
    if (!f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 2))) throw std::runtime_error("cannot read " + p);
    return v;
}

// The required case inventory of capture_fbc_ref.py (all must be present).
const char* const kFixtureCases[] = {"grid_20864", "grid_20926", "grid_20927", "grid_20928", "grid_20929", "grid_20992", "grid_21000",
    "grid_20000", "eq_threshold_20992", "eq_threshold_20993", "zero_all", "floor_tiny_ref", "floor_zero_ref", "big", "mixed_mag", "random_0",
    "random_1", "random_2", "random_3", "nan_in_call2", "inf_ref"};

void test_fixtures(const std::string& dir) {
    std::vector<std::string> seen;
    std::ifstream man(dir + "/manifest.txt");
    if (!man) {
        std::printf("FAIL: no %s/manifest.txt\n", dir.c_str());
        ++failures;
        return;
    }
    std::string line;
    int cases = 0, ok = 0;
    while (std::getline(man, line)) {
        if (line.empty()) continue;
        std::istringstream is(line);
        std::string name, thr_s, sbits_s;
        std::uint32_t rows = 0, w = 0;
        int reuse = 0;
        is >> name >> rows >> w >> thr_s >> reuse >> sbits_s;
        seen.push_back(name);
        const double thr = std::stod(thr_s);
        const std::uint32_t sbits = static_cast<std::uint32_t>(std::stoul(sbits_s, nullptr, 16));
        const std::size_t n = static_cast<std::size_t>(rows) * w;
        const auto h0a = load(dir + "/" + name + "_h0a.bin", n), h1a = load(dir + "/" + name + "_h1a.bin", n), hla = load(dir + "/" + name + "_hla.bin", n),
                   h0b = load(dir + "/" + name + "_h0b.bin", n), h1b = load(dir + "/" + name + "_h1b.bin", n), outb = load(dir + "/" + name + "_outb.bin", n);
        FbcHost f(thr, w);
        auto h = h1a;
        const Action a1 = f.after_block0(h0a, h, rows, 64U, nullptr);
        f.commit(hla);
        h = h1b;
        Decision d;
        const Action a2 = f.after_block0(h0b, h, rows, 64U, &d);
        float want_score;
        std::memcpy(&want_score, &sbits, 4);
        const bool score_ok = std::isnan(want_score) ? !d.finite : std::memcmp(&d.score, &want_score, 4) == 0;
        const bool good = a1 == Action::kFullCommit && (a2 == Action::kReuse) == (reuse != 0) && score_ok && h == outb;
        ++cases;
        ok += good;
        std::printf("fixture %-22s thr %-12s ref reuse %d score %-12g ours reuse %d score %-12g %s\n", name.c_str(), thr_s.c_str(), reuse,
            static_cast<double>(want_score), a2 == Action::kReuse ? 1 : 0, static_cast<double>(d.score), good ? "ok" : "MISMATCH");
    }
    std::size_t missing = 0;
    for (const char* c : kFixtureCases) missing += std::find(seen.begin(), seen.end(), std::string(c)) == seen.end();
    const std::size_t required = sizeof(kFixtureCases) / sizeof(kFixtureCases[0]);
    std::printf("reference fixtures: %d of %d match; required inventory %zu cases, %zu missing\n", ok, cases, required, missing);
    CHECK(cases > 0 && ok == cases && missing == 0 && static_cast<std::size_t>(cases) == required, "reference fixtures");
}
}  // namespace

int main(int argc, char** argv) {
    test_f64_to_bf16();
    test_decide();
    test_two_rank(1.0F, 1.0F, 31);
    test_two_rank(0x1p-60F, 0x1p60F, 41);
    test_two_rank(0x1p-120F, 1.0F, 51);
    test_state_machine();
    test_validation();
    if (argc > 2 && std::string(argv[1]) == "--fixtures") test_fixtures(argv[2]);
    else std::printf("reference fixtures: skipped (pass --fixtures <dir>)\n");
    std::printf("fbc host tests %s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
