// H3 text encoder GPU gate: layer-50 hidden states vs the reference fixtures.
//   strix-h3-te-gpu-tests PACK_DIR FIXTURE_DIR [name ...]
// Gate (per fixture): rel L2 over rows 1.. <= 2e-3 and min row cosine (rows 1..) >= 0.99999; token 0
// (the attention-sink row) is reported separately. Also checks determinism (two runs bitwise equal)
// and row-count invariance (a prefix run reproduces the first rows bitwise).
//   strix-h3-te-gpu-tests --layers N PACK_DIR FIXTURE_DIR   first N layers vs p0_short.layers.f32 (debug)
//   strix-h3-te-gpu-tests --vision PACK_DIR VISION_DIR name...  LM part of vision prompts: reference layer-0
//       embeddings, MRoPE positions and DeepStack features in, layer-50 states vs the fixture (same gate)
//   strix-h3-te-gpu-tests --vision-full PACK_DIR VISION_DIR MANIFEST name...  whole prompt from the presented
//       ids and the input frames: vision tower per block (merged / DeepStack vs fixture), then the LM (gate)
#include "h3/te/te_gpu.hpp"
#include "h3/te/te_pack_reader.hpp"
#include "te_fixture_io.hpp"

#include "strix/serve/json.hpp"

#include <cmath>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace strix::h3::te;

namespace {

using fixture_io::read_bin;

struct Stats {
    double rel_tail = 0, rel_row0 = 0, min_cos = 1, max_abs = 0, max_row_rel = 0;
    bool finite = true;
    std::vector<double> row_rel;   // per row (row 0 included)
    std::vector<double> ref_norm;  // per row |ref|
    std::vector<double> row_cos;
};

std::vector<std::vector<float>> g_sites;  // single-site drifts of the last row_limits() call

// Per-row limits: limit[r] = max(5e-3, 2 x median over the three single-site
// FP16 drifts of the fp32 reference forward (GEMM inputs, q/k, v), 2 x the pure fp32 reorder drift of the same
// forward), read from $TE_SENS_DIR/<name>.f16-{gin,qk,v}.rowrel.f32 and (optional) <name>.fp32.rowrel.f32;
// without the f16 files the limit is 5e-3.
std::vector<double> row_limits(const std::string& name, std::size_t rows, bool* have) {
    std::vector<double> lim(rows, 5e-3);
    *have = false;
    g_sites.clear();
    const char* dir = std::getenv("TE_SENS_DIR");
    if (!dir || !*dir) return lim;
    std::vector<std::vector<float>> site;
    for (const char* t : {"gin", "qk", "v"}) {
        std::ifstream f(std::string(dir) + "/" + name + ".f16-" + t + ".rowrel.f32", std::ios::binary | std::ios::ate);
        if (!f) return lim;
        const auto n = static_cast<std::size_t>(f.tellg());
        std::vector<float> v(n / 4);
        f.seekg(0);
        f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n));
        if (v.size() != rows) return lim;
        site.push_back(std::move(v));
    }
    g_sites = site;
    std::vector<float> reorder;
    {
        std::ifstream f(std::string(dir) + "/" + name + ".fp32.rowrel.f32", std::ios::binary | std::ios::ate);
        if (f) {
            const auto n = static_cast<std::size_t>(f.tellg());
            reorder.resize(n / 4);
            f.seekg(0);
            f.read(reinterpret_cast<char*>(reorder.data()), static_cast<std::streamsize>(n));
            if (reorder.size() != rows) reorder.clear();
        }
    }
    std::printf("  %s limits: 3 FP16 sites%s\n", name.c_str(), reorder.empty() ? " (no fp32 reorder file)" : " + fp32 reorder");
    for (std::size_t r = 0; r < rows; ++r) {
        float a = site[0][r], b = site[1][r], c = site[2][r];
        const float med = std::max(std::min(a, b), std::min(std::max(a, b), c));
        lim[r] = std::max(5e-3, 2.0 * double(med));
        if (!reorder.empty()) lim[r] = std::max(lim[r], 2.0 * double(reorder[r]));
    }
    *have = true;
    return lim;
}

// Aggregate rule: rows 1.. rel L2 <= max(2e-3, 2 x median over the 3 sites of the reference's own
// aggregate drift); the aggregate over rows 1.. EXCLUDING amplified rows (limit > 5e-3) must be <= 2e-3, and so
// must the min cosine (>= 0.99999) over those rows. Returns true when the aggregate part passes.
bool check_aggregate(const Stats& s, const std::vector<double>& lim, bool have, const std::string& name) {
    double agg_lim = 2e-3;
    if (have && g_sites.size() == 3) {
        double a[3];
        for (int t = 0; t < 3; ++t) {
            double num = 0, den = 0;
            for (std::size_t r = 1; r < s.ref_norm.size(); ++r) {
                const double e = double(g_sites[std::size_t(t)][r]) * s.ref_norm[r];
                num += e * e;
                den += s.ref_norm[r] * s.ref_norm[r];
            }
            a[t] = den > 0 ? std::sqrt(num / den) : 0.0;
        }
        std::sort(a, a + 3);
        agg_lim = std::max(2e-3, 2.0 * a[1]);
        std::printf("  %s reference single-site aggregate drift (gin/qk/v sorted) %.3e %.3e %.3e -> aggregate limit %.3e\n",
                    name.c_str(), a[0], a[1], a[2], agg_lim);
    }
    double num = 0, den = 0, cmin = 1;
    std::size_t kept = 0;
    for (std::size_t r = 1; r < s.row_rel.size(); ++r) {
        if (lim[r] > 5e-3) continue;
        const double e = s.row_rel[r] * s.ref_norm[r];
        num += e * e;
        den += s.ref_norm[r] * s.ref_norm[r];
        cmin = std::min(cmin, s.row_cos[r]);
        ++kept;
    }
    const double excl = den > 0 ? std::sqrt(num / den) : 0.0;
    const bool ok = s.rel_tail <= agg_lim && excl <= 2e-3 && cmin >= 0.99999;
    std::printf("  %s aggregate rows1+ %.3e (limit %.3e); excluding amplified rows: %.3e over %zu rows, min cos %.7f (limits 2e-3, "
                "0.99999) %s\n", name.c_str(), s.rel_tail, agg_lim, excl, kept, cmin, ok ? "ok" : "FAIL");
    return ok;
}

// rows 1..: row_rel <= limit; returns the number of violating rows and prints the amplified rows
std::size_t check_rows(const Stats& s, const std::vector<double>& lim, bool have, const std::string& name) {
    std::size_t bad = 0, amplified = 0;
    for (std::size_t r = 1; r < s.row_rel.size(); ++r) {
        if (s.row_rel[r] > lim[r]) {
            ++bad;
            std::printf("  %s row %zu rel %.3e > limit %.3e\n", name.c_str(), r, s.row_rel[r], lim[r]);
        }
        if (lim[r] > 5e-3) ++amplified;
    }
    if (have) {
        std::printf("  %s amplified rows (limit > 5e-3): %zu", name.c_str(), amplified);
        std::size_t shown = 0;
        for (std::size_t r = 1; r < s.row_rel.size() && shown < 12; ++r)
            if (lim[r] > 5e-3) { std::printf(" [%zu lim %.2e got %.2e]", r, lim[r], s.row_rel[r]); ++shown; }
        std::printf("\n");
    }
    return bad;
}

// got and ref must both be exactly [rows][width] with rows, width > 0 (no vacuous passes).
Stats compare(const std::vector<float>& got, const std::vector<float>& ref, std::size_t rows, std::size_t width) {
    fixture_io::check_dims(got.size(), ref.size(), rows, width);
    Stats s;
    double num = 0, den = 0;
    for (std::size_t r = 0; r < rows; ++r) {
        double d2 = 0, r2 = 0, g2 = 0, gr = 0;
        for (std::size_t c = 0; c < width; ++c) {
            const double g = got[r * width + c], e = ref[r * width + c];
            if (!std::isfinite(g)) s.finite = false;
            d2 += (g - e) * (g - e);
            r2 += e * e;
            g2 += g * g;
            gr += g * e;
            s.max_abs = std::fmax(s.max_abs, std::fabs(g - e));
        }
        s.row_rel.push_back(std::sqrt(d2 / r2));
        s.ref_norm.push_back(std::sqrt(r2));
        s.row_cos.push_back(gr / std::sqrt(g2 * r2));
        if (r == 0) { s.rel_row0 = std::sqrt(d2 / r2); continue; }
        num += d2;
        den += r2;
        s.max_row_rel = std::fmax(s.max_row_rel, std::sqrt(d2 / r2));
        s.min_cos = std::fmin(s.min_cos, gr / std::sqrt(g2 * r2));
    }
    s.rel_tail = rows > 1 ? std::sqrt(num / den) : 0.0;
    return s;
}

void worst_rows(const std::vector<float>& got, const std::vector<float>& ref, const std::vector<std::int32_t>& tags,
                std::size_t width, int n) {
    const std::size_t rows = ref.size() / width;
    std::vector<std::pair<double, std::size_t>> v;
    for (std::size_t r = 1; r < rows; ++r) {
        double d2 = 0, r2 = 0;
        for (std::size_t c = 0; c < width; ++c) {
            const double d = double(got[r * width + c]) - ref[r * width + c];
            d2 += d * d;
            r2 += double(ref[r * width + c]) * ref[r * width + c];
        }
        v.emplace_back(std::sqrt(d2 / r2), r);
    }
    std::sort(v.begin(), v.end(), [](auto a, auto b) { return a.first > b.first; });
    std::printf("  worst rows:");
    for (int i = 0; i < n && i < int(v.size()); ++i) {
        const auto r = v[std::size_t(i)].second;
        double r2 = 0;
        for (std::size_t c = 0; c < width; ++c) r2 += double(ref[r * width + c]) * ref[r * width + c];
        std::printf(" [row %zu tag %d rel %.2e |ref| %.1f]", r, r < tags.size() ? tags[r] : -1, v[std::size_t(i)].first, std::sqrt(r2));
    }
    std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 5 && std::string(argv[1]) == "--layers") {
        const int n = std::stoi(argv[2]);
        auto te = TextEncoder::load(argv[3], n);
        const std::string fx = argv[4];
        const auto ids = read_bin<std::int32_t>(fx + "/p0_short.ids.i32");
        const auto all = read_bin<float>(fx + "/p0_short.layers.f32");
        const std::size_t per = ids.size() * 5120;
        const std::vector<float> ref(all.begin() + std::ptrdiff_t(per * std::size_t(n - 1)), all.begin() + std::ptrdiff_t(per * std::size_t(n)));
        const auto got = te->encode(ids);
        const auto s = compare(got, ref, ids.size(), 5120);
        std::printf("layers %d: rel_l2_rows1+ %.3e min_cos %.7f max_row_rel %.3e row0_rel %.3e max_abs %.3g\n", n, s.rel_tail,
                    s.min_cos, s.max_row_rel, s.rel_row0, s.max_abs);
        return 0;
    }
    if (argc >= 6 && std::string(argv[1]) == "--named") {
        // --named PACK FIXTURES TOKDIR EMBDIR name...: whole presentation in C++ (tokenizer with the
        // embedding directory), ids vs the fixture, spliced user embedding rows, gate vs the fixture
        auto te = TextEncoder::load(argv[2]);
        const std::string fx = argv[3], tokdir = argv[4], embdir = argv[5];
        const auto tok = Tokenizer::load(tokdir);
        int bad = 0;
        std::ifstream mf(fx + "/manifest.json");
        std::stringstream ms;
        ms << mf.rdbuf();
        const auto man = strix::serve::parse_json(ms.str());
        for (int a = 6; a < argc; ++a) {
            const std::string name = argv[a];
            std::string text;
            for (const auto& e : man.find("fixtures")->as_array())
                if (e.find("name")->as_string() == name) text = e.find("text")->as_string();
            const EmbeddingLookup lookup = [&](std::string_view n) {
                std::ifstream f(embdir + "/" + std::string(n) + ".safetensors", std::ios::binary);
                if (!f) return -1;
                return read_prompt_embedding(embdir + "/" + std::string(n) + ".safetensors").rows;
            };
            const auto pres = present(*tok, text, 0, {}, &lookup);
            const auto want_ids = read_bin<std::int32_t>(fx + "/" + name + ".ids.i32");
            const bool ids_ok = pres.ids == want_ids;
            bad += ids_ok ? 0 : 1;
            std::vector<float> spliced;
            for (const auto& n : pres.embeddings) {
                const auto e = read_prompt_embedding(embdir + "/" + n + ".safetensors");
                spliced.insert(spliced.end(), e.data.begin(), e.data.end());
            }
            EncodeTimings t;
            const auto got = encode_presented(*te, nullptr, pres, {}, spliced, &t);
            const auto ref = read_bin<float>(fx + "/" + name + ".hidden.f32");
            const auto s = compare(got, ref, ref.size() / 5120, 5120);
            bool have = false;
            const auto lim = row_limits(name, ref.size() / 5120, &have);
            const bool pass = ids_ok && s.finite && got.size() == ref.size() && check_aggregate(s, lim, have, name) &&
                              check_rows(s, lim, have, name) == 0 && s.rel_row0 <= 2e-3;
            bad += pass ? 0 : 1;
            std::printf("%s named embeddings %zu (%zu rows) ids %s; S=%zu rel_l2_rows1+ %.3e min_cos %.7f max_row_rel %.3e row0 %.3e "
                        "forward %.1f ms %s\n", name.c_str(), pres.embeddings.size(), spliced.size() / 5120, ids_ok ? "equal" : "DIFFER",
                        ref.size() / 5120, s.rel_tail, s.min_cos, s.max_row_rel, s.rel_row0, t.forward_ms, pass ? "PASS" : "FAIL");
        }
        return bad ? 1 : 0;
    }
    if (argc >= 7 && std::string(argv[1]) == "--row-tap") {
        // --row-tap PACK FIXTURES NAME ROW OUT: residual row ROW after every layer -> OUT ([50][5120] fp32)
        auto te = TextEncoder::load(argv[2]);
        const auto ids = read_bin<std::int32_t>(std::string(argv[3]) + "/" + argv[4] + ".ids.i32");
        const char* sub = std::getenv("TE_TAP_SUB");
        te->set_row_tap(std::stoi(argv[5]), sub && *sub == '1');
        (void)te->encode(ids);
        const auto& taps = te->row_taps();
        std::ofstream(argv[6], std::ios::binary).write(reinterpret_cast<const char*>(taps.data()), std::streamsize(taps.size() * 4));
        std::printf("row tap %s row %s: %zu floats written (%s)\n", argv[4], argv[5], taps.size(),
                    te->identity().c_str());
        return 0;
    }
    if (argc >= 7 && std::string(argv[1]) == "--replay") {
        // --replay PACK RDIR L0 L1 ROW: shared-input sublayer replay. RDIR holds the fp32 reference's
        // x_in / x_attn / x_ffn.<L>.f32 ([S][5120]) and its single-site local drifts
        // (attn.<L>.{gin,qk,v}.rowrel.f32, ffn.<L>.gin.rowrel.f32), from te_ref.py --replay-dir. For each
        // layer, our attention block runs on the reference x_in and our FFN block on the reference x_attn;
        // the per-row relative error of each output is our local error of that sublayer.
        auto te = TextEncoder::load(argv[2]);
        const std::string dir = argv[3];
        const int l0 = std::stoi(argv[4]), l1 = std::stoi(argv[5]);
        const std::size_t row = std::stoul(argv[6]);
        const std::string cls = te->identity();
        std::printf("replay %s layers %d..%d row %zu\n", cls.c_str(), l0, l1, row);
        std::printf("L  block  gpu@row  median(site)@row [gin qk v]@row | gpu rows1+ agg  max(row)  median-row ratio "
                    "max(gpu/median(site))(row)\n");
        const auto median = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            return v.empty() ? 0.0 : v[v.size() / 2];
        };
        for (int L = l0; L <= l1; ++L) {
            char tag[16];
            std::snprintf(tag, sizeof tag, "%02d", L);
            const auto xin = read_bin<float>(dir + "/x_in." + tag + ".f32");
            const auto xat = read_bin<float>(dir + "/x_attn." + tag + ".f32");
            const auto xff = read_bin<float>(dir + "/x_ffn." + tag + ".f32");
            const std::size_t S = xin.size() / 5120;
            if (row >= S) throw std::runtime_error("replay: row out of range");
            for (int blk = 0; blk < 2; ++blk) {
                EncodeInput in;
                in.embeds = blk == 0 ? xin : xat;
                te->set_replay(L, blk == 1);
                te->set_debug_stop(L, blk == 0 ? 5 : 0);
                const auto got = te->encode(in);
                const auto& ref = blk == 0 ? xat : xff;
                const auto s = compare(got, ref, S, 5120);
                std::vector<std::vector<float>> site;
                std::vector<std::string> sn = blk == 0 ? std::vector<std::string>{"gin", "qk", "v"} : std::vector<std::string>{"gin"};
                for (const auto& n : sn) {
                    site.push_back(read_bin<float>(dir + "/" + (blk == 0 ? "attn." : "ffn.") + tag + "." + n + ".rowrel.f32"));
                    if (site.back().size() != S) throw std::runtime_error("replay: site rowrel size");
                }
                const auto med_site = [&](std::size_t r) {
                    std::vector<double> v;
                    for (const auto& x : site) v.push_back(x[r]);
                    return median(v);
                };
                std::vector<double> ratio;
                double worst = 0;
                std::size_t worst_r = 0;
                for (std::size_t r = 1; r < S; ++r) {
                    const double q = s.row_rel[r] / std::max(med_site(r), 1e-30);
                    ratio.push_back(q);
                    if (q > worst) { worst = q; worst_r = r; }
                }
                std::printf("%2d %-5s %.2e %.2e [", L, blk == 0 ? "attn" : "ffn", s.row_rel[row], med_site(row));
                for (const auto& x : site) std::printf(" %.2e", x[row]);
                std::printf(" ] | %.2e %.2e %.2f %.2f(%zu)\n", s.rel_tail, s.max_row_rel, median(ratio), worst, worst_r);
                const std::vector<float> rr(s.row_rel.begin(), s.row_rel.end());
                const std::string out = dir + "/gpu." + (cls.find("split") != std::string::npos ? "dense" : "fast") + "." +
                                        (blk == 0 ? "attn." : "ffn.") + tag + ".rowrel.f32";
                std::ofstream(out, std::ios::binary).write(reinterpret_cast<const char*>(rr.data()), std::streamsize(rr.size() * 4));
            }
        }
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--probe") {
        // row-count invariance of the kernels on random data
        std::uint32_t st = 12345;
        auto rnd = [&]() { st = st * 1664525U + 1013904223U; return float(int(st >> 8) % 20001 - 10000) / 10000.0F; };
        int bad = 0;
        {
            std::vector<float> a(256), b(256);
            for (auto& x : a) x = rnd();
            for (auto& x : b) x = rnd();
            const auto d1 = probe::wmma_rows(a, b);
            for (int i = 5 * 16; i < 256; ++i) a[std::size_t(i)] = 0.0F;  // zero rows 5..15
            const auto d2 = probe::wmma_rows(a, b);
            const bool ok = std::memcmp(d1.data(), d2.data(), 5 * 16 * 4) == 0;
            bad += ok ? 0 : 1;
            std::printf("probe wmma rows 0..4 independent of rows 5..15: %s\n", ok ? "PASS" : "FAIL");
        }
        for (const bool split : {false, true}) {
            const unsigned n = 512, k = 256, r1 = 10, r2 = 5;
            std::vector<float> x(std::size_t(r1) * k), w(std::size_t(n) * k);
            for (auto& v : x) v = rnd();
            for (auto& v : w) v = rnd();
            const auto g1 = probe::gemm_rows(x, w, r1, n, k, split);
            const std::vector<float> x2(x.begin(), x.begin() + std::ptrdiff_t(r2 * k));
            const auto g2 = probe::gemm_rows(x2, w, r2, n, k, split);
            const bool ok = std::memcmp(g1.data(), g2.data(), std::size_t(r2) * n * 4) == 0;
            bad += ok ? 0 : 1;
            std::printf("probe gemm split=%d rows 0..4 of S=10 vs S=5: %s\n", int(split), ok ? "PASS" : "FAIL");
        }
        {
            // WMMA row independence with extreme other rows: subnormal, huge, mixed
            for (int mode = 0; mode < 4; ++mode) {
                std::vector<float> a(256), b(256);
                for (auto& x : a) x = rnd();
                for (auto& x : b) x = rnd() * (mode == 3 ? 1e-5F : 1.0F);
                const auto d1 = probe::wmma_rows(a, b);
                for (int i = 5 * 16; i < 256; ++i)
                    a[std::size_t(i)] = mode == 0 ? rnd() * 3e-6F : mode == 1 ? rnd() * 3e4F : mode == 2 ? (i % 2 ? 0.0F : rnd() * 1e-7F) : rnd();
                const auto d2 = probe::wmma_rows(a, b);
                const bool ok = std::memcmp(d1.data(), d2.data(), 5 * 16 * 4) == 0;
                bad += ok ? 0 : 1;
                std::printf("probe wmma mode %d (other rows %s): %s\n", mode,
                            mode == 0 ? "subnormal" : mode == 1 ? "huge" : mode == 2 ? "sparse tiny" : "B tiny", ok ? "PASS" : "FAIL");
            }
        }
        {
            // WMMA column independence: output columns 0..4 vs the B rows (= output columns) 5..15, with C != 0
            for (int mode = 0; mode < 4; ++mode) {
                std::vector<float> a(256), b(256), c(256);
                for (auto& x : a) x = rnd();
                for (auto& x : b) x = rnd() * (mode == 3 ? 4e-4F : 1.0F);
                for (auto& x : c) x = rnd() * 10;
                const auto d1 = probe::wmma_rows(a, b, c.data());
                for (int i = 5 * 16; i < 256; ++i)
                    b[std::size_t(i)] = mode == 0 ? 0.0F : mode == 1 ? rnd() * 3e4F : mode == 2 ? rnd() * 3e-6F : rnd() * 4e-4F;
                const auto d2 = probe::wmma_rows(a, b, c.data());
                bool ok = true;
                for (int r = 0; r < 16; ++r)
                    for (int col = 0; col < 5; ++col) ok &= std::memcmp(&d1[std::size_t(r * 16 + col)], &d2[std::size_t(r * 16 + col)], 4) == 0;
                bad += ok ? 0 : 1;
                std::printf("probe wmma column independence mode %d (other B rows %s, C != 0): %s\n", mode,
                            mode == 0 ? "zero" : mode == 1 ? "huge" : mode == 2 ? "subnormal" : "tiny lo-like", ok ? "PASS" : "FAIL");
            }
        }
        for (const bool split : {false, true})
            for (const unsigned r1 : {10U, 43U}) {
                const unsigned r2 = r1 / 2;
                std::vector<float> q(std::size_t(64) * r1 * 128), kk(std::size_t(8) * r1 * 128), vv(std::size_t(8) * r1 * 128);
                for (auto& v : q) v = rnd() * 3;
                for (auto& v : kk) v = rnd() * 3;
                for (auto& v : vv) v = rnd();
                const auto take = [&](const std::vector<float>& src, unsigned heads) {
                    std::vector<float> o;
                    for (unsigned h = 0; h < heads; ++h)
                        o.insert(o.end(), src.begin() + std::ptrdiff_t(std::size_t(h) * r1 * 128),
                                 src.begin() + std::ptrdiff_t(std::size_t(h) * r1 * 128 + std::size_t(r2) * 128));
                    return o;
                };
                const auto a1 = probe::attention_rows(q, kk, vv, r1, split);
                const auto a2 = probe::attention_rows(take(q, 64), take(kk, 8), take(vv, 8), r2, split);
                if (split) {
                    // which tail matters: zero the query rows >= r2 (keys real) or the key/value rows >= r2
                    const auto zero_tail = [&](std::vector<float> v, unsigned heads) {
                        for (unsigned h = 0; h < heads; ++h)
                            for (std::size_t i = std::size_t(h) * r1 * 128 + std::size_t(r2) * 128; i < std::size_t(h + 1) * r1 * 128; ++i) v[i] = 0.0F;
                        return v;
                    };
                    const auto aq = probe::attention_rows(zero_tail(q, 64), kk, vv, r1, true);
                    const auto akv = probe::attention_rows(q, zero_tail(kk, 8), zero_tail(vv, 8), r1, true);
                    std::size_t nq = 0, nkv = 0;
                    for (std::size_t i = 0; i < a2.size(); ++i) {
                        nq += std::memcmp(&aq[i], &a2[i], 4) != 0;
                        nkv += std::memcmp(&akv[i], &a2[i], 4) != 0;
                    }
                    std::printf("probe attention split S=%u: q-tail zeroed -> %zu differ from prefix; kv-tail zeroed -> %zu differ\n",
                                r1, nq, nkv);
                    {
                        // taps of the last prefix row (head 0): which quantity first differs
                        const unsigned row = r2 - 1;
                        const auto tf = probe::attention_taps(q, kk, vv, r1, true, row);
                        const auto tp = probe::attention_taps(take(q, 64), take(kk, 8), take(vv, 8), r2, true, row);
                        const char* names[] = {"masked scores", "hi scores", "lo chain", "P", "m,l", "o"};
                        const unsigned lo_[] = {0, 32, 64, 96, 128, 256}, hi_[] = {32, 64, 96, 128, 130, 384};
                        for (unsigned kb = 0; kb < (row / 32 + 1); ++kb)
                            for (int q2 = 0; q2 < 6; ++q2) {
                                std::size_t nd = 0, first = SIZE_MAX;
                                for (unsigned i = lo_[q2]; i < hi_[q2]; ++i)
                                    if (std::memcmp(&tf[kb * 512 + i], &tp[kb * 512 + i], 4) != 0) { ++nd; if (first == SIZE_MAX) first = i; }
                                std::printf("  taps S=%u row %u block %u %-13s: %zu differ", r1, row, kb, names[q2], nd);
                                if (nd) std::printf(" (first idx %zu: full %.9g prefix %.9g)", first - lo_[q2], tf[kb * 512 + first], tp[kb * 512 + first]);
                                std::printf("\n");
                            }
                    }
                    for (const int var : {1, 2}) {
                        const auto f = probe::attention_rows(q, kk, vv, r1, true, var);
                        const auto g = probe::attention_rows(take(q, 64), take(kk, 8), take(vv, 8), r2, true, var);
                        std::size_t nv = 0;
                        for (std::size_t i = 0; i < g.size(); ++i) nv += std::memcmp(&f[i], &g[i], 4) != 0;
                        std::printf("probe attention variant %s S=%u vs %u: %zu differ\n", var == 1 ? "split-scores-only" : "split-PV-only",
                                    r1, r2, nv);
                    }
                }
                std::size_t nd = 0, first = SIZE_MAX;
                for (std::size_t i = 0; i < a2.size(); ++i)
                    if (std::memcmp(&a1[i], &a2[i], 4) != 0) { ++nd; if (first == SIZE_MAX) first = i; }
                bad += nd ? 1 : 0;
                std::printf("probe attention split=%d S=%u vs %u: %zu of %zu differ (first row %zd) %s\n", int(split), r1, r2, nd,
                            a2.size(), first == SIZE_MAX ? -1 : std::ptrdiff_t(first / 8192), nd ? "FAIL" : "PASS");
            }
        return bad ? 1 : 0;
    }
    if (argc >= 5 && std::string(argv[1]) == "--prefix-debug") {
        // prefix invariance localisation: full vs first-half run, per (layer, stage)
        auto te = TextEncoder::load(argv[2]);
        const std::string fx = argv[3];
        for (int a = 4; a < argc; ++a) {
            const auto ids = read_bin<std::int32_t>(fx + "/" + std::string(argv[a]) + ".ids.i32");
            const std::vector<std::int32_t> pre(ids.begin(), ids.begin() + ids.size() / 2);
            // first layer whose output differs, then every stage of that layer
            int first_layer = -1;
            for (int layer = 0; layer < 50 && first_layer < 0; ++layer) {
                te->set_debug_stop(layer, 0);
                const auto f = te->encode(ids), p = te->encode(pre);
                if (std::memcmp(f.data(), p.data(), p.size() * 4) != 0) first_layer = layer;
            }
            std::printf("%s first differing layer output: %d\n", argv[a], first_layer);
            if (first_layer < 0) continue;
            for (const int layer : {first_layer})
                for (const int stage : {1, 2, 3, 4, 5, 0}) {
                    te->set_debug_stop(layer, stage);
                    const auto f = te->encode(ids), p = te->encode(pre);
                    const std::size_t w = f.size() / ids.size();
                    const std::size_t planes = (stage == 1 || stage == 4) && w > 0 ? std::size_t(f.size() / ids.size() / w) : 1;
                    (void)planes;
                    // panel stages hold [hi][lo] planes of S rows each; compare row by row inside each plane
                    const std::size_t fw = (stage == 1 ? 5120 : stage == 4 ? 8192 : w);
                    const std::size_t np = f.size() / (ids.size() * fw);
                    std::size_t nd = 0, first = SIZE_MAX;
                    double mx = 0;
                    for (std::size_t pl = 0; pl < np; ++pl)
                        for (std::size_t i = 0; i < pre.size() * fw; ++i) {
                            const float a = p[pl * pre.size() * fw + i], b = f[pl * ids.size() * fw + i];
                            if (std::memcmp(&a, &b, 4) != 0) {
                                ++nd;
                                if (first == SIZE_MAX) first = pl * 1000000 + i;
                                mx = std::fmax(mx, std::fabs(double(a) - b));
                            }
                        }
                    std::printf("%s layer %d stage %d width %zu planes %zu: %zu of %zu differ, max_abs %.3g, first plane %zd row %zd col %zd\n",
                                argv[a], layer, stage, fw, np, nd, p.size(), mx,
                                first == SIZE_MAX ? -1 : std::ptrdiff_t(first / 1000000),
                                first == SIZE_MAX ? -1 : std::ptrdiff_t(first % 1000000 / fw),
                                first == SIZE_MAX ? -1 : std::ptrdiff_t(first % 1000000 % fw));
                }
            te->set_debug_stop(-1, 0);
        }
        return 0;
    }
    if (argc >= 6 && std::string(argv[1]) == "--vision-full") {
        auto te = TextEncoder::load(argv[2]);
        auto vt = VisionTower::load(argv[2]);
        std::printf("vision tower device %.2f GB\n", double(vt->device_bytes()) / 1e9);
        const std::string dir = argv[3];
        std::ifstream mf(argv[4]);
        std::stringstream ss;
        ss << mf.rdbuf();
        const auto man = strix::serve::parse_json(ss.str());
        int bad = 0;
        for (int a = 5; a < argc; ++a) {
            const std::string name = argv[a];
            const strix::serve::Json* c = nullptr;
            for (const auto& e : man.find("cases")->as_array()) if (e.find("name")->as_string() == name) c = &e;
            if (!c) throw std::runtime_error("no case " + name);
            Presented pres;
            pres.ids = read_bin<std::int32_t>(dir + "/" + name + ".ids.i32");
            std::vector<Frames> blocks;
            const auto nb = c->find("vision")->as_array().size();
            for (std::size_t b = 0; b < nb; ++b) {
                const auto key = "vis" + std::to_string(b) + ".image.f32";
                const auto& shape = c->find("files")->find(key)->find("shape")->as_array();
                Frames f;
                f.t = int(shape.at(0).as_int());
                f.h = int(shape.at(1).as_int());
                f.w = int(shape.at(2).as_int());
                f.rgb = read_bin<float>(dir + "/" + name + "." + key);
                blocks.push_back(std::move(f));
            }
            // vision tower alone vs the fixture, per block
            for (std::size_t b = 0; b < nb; ++b) {
                const auto base = dir + "/" + name + ".vis" + std::to_string(b);
                const auto pt = make_patches(blocks[b]);
                double ms = 0;
                const auto vo = vt->run(pt, &ms);
                const auto rm = read_bin<float>(base + ".merged.f32");
                const auto sm = compare(vo.merged, rm, rm.size() / 5120, 5120);
                // tower gate: rel L2 <= 1e-3, worst row <= 5e-3, row 0 <= 1e-3, min cos >= 0.99999
                const auto tower_ok = [](const Stats& t) {
                    return t.finite && t.rel_tail <= 1e-3 && t.max_row_rel <= 5e-3 && t.rel_row0 <= 1e-3 && t.min_cos >= 0.99999;
                };
                bad += tower_ok(sm) ? 0 : 1;
                std::printf("%s vis%zu merged rows %zu rel_l2(rows1+) %.3e row0 %.3e min_cos %.7f max_row_rel %.3e (%.1f ms) %s\n",
                            name.c_str(), b, rm.size() / 5120, sm.rel_tail, sm.rel_row0, sm.min_cos, sm.max_row_rel, ms,
                            tower_ok(sm) ? "PASS" : "FAIL");
                for (int d = 0; d < 3; ++d) {
                    const auto rd = read_bin<float>(base + ".deep" + std::to_string(d) + ".f32");
                    const auto sd = compare(vo.deep[d], rd, rd.size() / 5120, 5120);
                    bad += tower_ok(sd) ? 0 : 1;
                    std::printf("%s vis%zu deep%d rel_l2(rows1+) %.3e row0 %.3e min_cos %.7f max_row_rel %.3e %s\n", name.c_str(), b,
                                d, sd.rel_tail, sd.rel_row0, sd.min_cos, sd.max_row_rel, tower_ok(sd) ? "PASS" : "FAIL");
                }
            }
            EncodeTimings t;
            double vms = 0;
            const auto got = encode_presented(*te, vt.get(), pres, blocks, {}, &t, &vms);
            const auto ref = read_bin<float>(dir + "/" + name + ".hidden.f32");
            const auto s = compare(got, ref, ref.size() / 5120, 5120);
            bool have = false;
            const auto lim = row_limits(name, ref.size() / 5120, &have);
            const bool pass = s.finite && got.size() == ref.size() && check_aggregate(s, lim, have, name) &&
                              check_rows(s, lim, have, name) == 0 && s.rel_row0 <= 2e-3;
            bad += pass ? 0 : 1;
            std::printf("%s full S=%zu rel_l2_rows1+ %.3e min_cos_rows1+ %.7f max_row_rel_rows1+ %.3e row0_rel %.3e "
                        "vision %.1f ms LM %.1f ms %s\n", name.c_str(), ref.size() / 5120, s.rel_tail, s.min_cos, s.max_row_rel,
                        s.rel_row0, vms, t.forward_ms, pass ? "PASS" : "FAIL");
            worst_rows(got, ref, read_bin<std::int32_t>(dir + "/" + name + ".tags.i32"), 5120, 6);
        }
        return bad ? 1 : 0;
    }
    if (argc >= 5 && std::string(argv[1]) == "--vision") {
        auto te = TextEncoder::load(argv[2]);
        const std::string dir = argv[3];
        int bad = 0;
        for (int a = 4; a < argc; ++a) {
            const std::string name = argv[a];
            EncodeInput in;
            in.embeds = read_bin<float>(dir + "/" + name + ".embeds.f32");
            in.positions = read_bin<float>(dir + "/" + name + ".positions.f32");
            const auto tags = read_bin<std::int32_t>(dir + "/" + name + ".tags.i32");
            // visual rows: every run of tag 0 without its first (vision start) and last (vision end) row
            for (std::size_t i = 0; i < tags.size();) {
                if (tags[i] != 0) { ++i; continue; }
                std::size_t j = i;
                while (j < tags.size() && tags[j] == 0) ++j;
                for (std::size_t r = i + 1; r + 1 < j; ++r) in.visual_rows.push_back(std::uint32_t(r));
                i = j;
            }
            in.deepstack.resize(3);
            for (int b = 0;; ++b) {
                const std::string base = dir + "/" + name + ".vis" + std::to_string(b);
                std::ifstream probe(base + ".deep0.f32");
                if (!probe) break;
                for (int d = 0; d < 3; ++d) {
                    const auto v = read_bin<float>(base + ".deep" + std::to_string(d) + ".f32");
                    in.deepstack[d].insert(in.deepstack[d].end(), v.begin(), v.end());
                }
            }
            const auto ref = read_bin<float>(dir + "/" + name + ".hidden.f32");
            EncodeTimings t;
            const auto got = te->encode(in, &t);
            const std::size_t rows = tags.size();
            const auto s = compare(got, ref, rows, 5120);
            bool have = false;
            const auto lim = row_limits(name, rows, &have);
            const bool pass = s.finite && check_aggregate(s, lim, have, name) && check_rows(s, lim, have, name) == 0 &&
                              s.rel_row0 <= 2e-3;
            bad += pass ? 0 : 1;
            std::printf("%s S=%zu visual_rows %zu rel_l2_rows1+ %.3e min_cos_rows1+ %.7f max_row_rel_rows1+ %.3e row0_rel %.3e "
                        "forward %.1f ms %s\n", name.c_str(), rows, in.visual_rows.size(), s.rel_tail, s.min_cos, s.max_row_rel,
                        s.rel_row0, t.forward_ms, pass ? "PASS" : "FAIL");
            worst_rows(got, ref, tags, 5120, 6);
        }
        return bad ? 1 : 0;
    }
    if (argc < 3) {
        std::printf("usage: strix-h3-te-gpu-tests PACK_DIR FIXTURE_DIR [name ...]\n");
        return 2;
    }
    const std::string pack = argv[1], fixtures = argv[2];
    std::vector<std::string> names;
    for (int i = 3; i < argc; ++i) names.emplace_back(argv[i]);
    if (names.empty()) names = {"p0_short", "p1_scene", "p2_mixed"};
    int failures = 0;
    try {
        auto te = TextEncoder::load(pack);
        std::printf("identity %s; pack %s (%s)\n", te->identity().c_str(), te->pack_digest().c_str(),
                    pack_hash_verification_enabled() ? "sha256 verified" : "hashes skipped");
        std::printf("load %.1f s, device %.2f GB\n", te->load_seconds(),
                    double(te->device_bytes()) / 1e9);
        std::printf("expand all 50 layers: %.1f ms\n", te->expand_all_ms());
        std::vector<float> first;
        std::vector<std::int32_t> first_ids;
        for (const auto& name : names) {
            const auto ids = read_bin<std::int32_t>(fixtures + "/" + name + ".ids.i32");
            const auto ref = read_bin<float>(fixtures + "/" + name + ".hidden.f32");
            EncodeTimings t;
            (void)te->encode(ids, &t);  // warm
            const auto got = te->encode(ids, &t);
            const auto s = compare(got, ref, ids.size(), 5120);
            // gate: aggregate rows 1.. (rel L2 <= 2e-3, min cos >= 0.99999), worst single row 1.. (rel <= 5e-3),
            // row 0 (attention sink, |x| ~ 1.6e4) rel <= 2e-3; rows 1..: per-row limits (row_limits)
            if (const char* dd = std::getenv("TE_DUMP_ROWS"); dd && *dd) {
                std::vector<float> rr(s.row_rel.begin(), s.row_rel.end()), nn(s.ref_norm.begin(), s.ref_norm.end()),
                    cc(s.row_cos.begin(), s.row_cos.end());
                std::ofstream(std::string(dd) + "/" + name + ".gpu.rowrel.f32", std::ios::binary)
                    .write(reinterpret_cast<const char*>(rr.data()), std::streamsize(rr.size() * 4));
                std::ofstream(std::string(dd) + "/" + name + ".refnorm.f32", std::ios::binary)
                    .write(reinterpret_cast<const char*>(nn.data()), std::streamsize(nn.size() * 4));
                std::ofstream(std::string(dd) + "/" + name + ".gpu.rowcos.f32", std::ios::binary)
                    .write(reinterpret_cast<const char*>(cc.data()), std::streamsize(cc.size() * 4));
            }
            bool have = false;
            const auto lim = row_limits(name, ids.size(), &have);
            const std::size_t row_bad = check_rows(s, lim, have, name);
            const bool agg_ok = check_aggregate(s, lim, have, name);
            const bool pass = s.finite && agg_ok && row_bad == 0 && s.rel_row0 <= 2e-3;
            failures += pass ? 0 : 1;
            std::printf("%s S=%zu rel_l2_rows1+ %.3e min_cos_rows1+ %.7f max_row_rel_rows1+ %.3e row0_rel %.3e max_abs %.3g "
                        "forward %.1f ms (upload %.1f, download %.1f) %s\n",
                        name.c_str(), ids.size(), s.rel_tail, s.min_cos, s.max_row_rel, s.rel_row0, s.max_abs, t.forward_ms,
                        t.upload_ms, t.download_ms, pass ? "PASS" : "FAIL");
            if (first.empty()) { first = got; first_ids = ids; }
            if (!pass) worst_rows(got, ref, read_bin<std::int32_t>(fixtures + "/" + name + ".tags.i32"), 5120, 6);
            const auto again = te->encode(ids);
            const bool det = again.size() == got.size() && std::memcmp(again.data(), got.data(), got.size() * 4) == 0;
            failures += det ? 0 : 1;
            std::printf("%s determinism (2 runs bitwise): %s\n", name.c_str(), det ? "PASS" : "FAIL");
            if (ids.size() > 4) {
                const std::vector<std::int32_t> prefix(ids.begin(), ids.begin() + ids.size() / 2);
                const auto p = te->encode(prefix);
                const bool inv = std::memcmp(p.data(), got.data(), p.size() * 4) == 0;
                failures += inv ? 0 : 1;
                std::printf("%s prefix %zu rows bitwise equal: %s\n", name.c_str(), prefix.size(), inv ? "PASS" : "FAIL");
            }
        }
        // throughput at longer sequences (synthetic ids, timing only)
        const char* nt = std::getenv("TE_NO_TIMING");
        for (const std::size_t n : {300U, 1000U, 2500U, 5000U}) {
            if (nt && *nt == '1') break;
            std::vector<std::int32_t> ids(n);
            for (std::size_t i = 0; i < n; ++i) ids[i] = first_ids[i % first_ids.size()];
            EncodeTimings t;
            (void)te->encode(ids, &t);
            const double flop = 2.0 * 24.4e9 * double(n);
            std::printf("timing S=%zu forward %.1f ms (%.1f TF/s GEMM-equivalent)\n", n, t.forward_ms, flop / (t.forward_ms * 1e9));
        }
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf("te gpu: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
