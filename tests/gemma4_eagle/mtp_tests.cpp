// MTP assistant drafter host tests. usage: strix-gemma4-eagle-mtp-tests [fixture_dir assistant_dir]
// Without arguments the HF fixture comparison is skipped (it needs the real checkpoint).
#include "mtp_common.hpp"
#include "gemma4/eagle/mtp/session.hpp"

#include <cmath>
#include <limits>

using namespace mtp_test;

namespace {
void recipes() {
    Rng r{3};
    TargetKv kv(300, 5);
    std::vector<float> q(512);
    for (auto& x : q) x = r.uniform();
    const std::pair<std::uint64_t, std::uint64_t> ranges[] = {{0, 0}, {0, 31}, {5, 299}, {0, 299}};
    for (const auto& [first, last] : ranges) {
        std::vector<float> out(512);
        mt::attend_head(q.data(), 512, kv.full(), 1, first, last, out.data());
        std::vector<double> s;
        double m = -1e300, den = 0;
        for (auto j = first; j <= last; ++j) {
            double d = 0;
            for (std::size_t i = 0; i < 512; ++i) d += double(q[i]) * kv.kf[j * 1024 + 512 + i];
            s.push_back(d); m = std::max(m, d);
        }
        for (auto& x : s) { x = std::exp(x - m); den += x; }
        double worst = 0;
        for (std::size_t i = 0; i < 512; ++i) {
            double acc = 0;
            for (std::size_t j = 0; j < s.size(); ++j) acc += s[j] * kv.vf[(first + j) * 1024 + 512 + i];
            worst = std::max(worst, std::fabs(acc / den - out[i]));
        }
        EAGLE_CHECK(worst < 1e-5, "attention [%llu,%llu] max err %g", (unsigned long long)first, (unsigned long long)last, worst);
    }
    std::vector<float> c(2048, 0.0F);
    c[7] = 3; c[5] = 3; c[100] = 2;
    const auto top = mt::top_clusters(c, 4);
    EAGLE_CHECK(top[0] == 5 && top[1] == 7 && top[2] == 100 && top[3] == 0, "top-k order/ties %u %u %u %u", top[0], top[1], top[2], top[3]);
}

void fixture(const std::filesystem::path& dir, const std::filesystem::path& model, const Table& table) {
    Fixture f;
    if (!Fixture::load(dir, f) || !std::filesystem::exists(model / "model.safetensors")) {
        std::printf("SKIP fixture (no %s or no checkpoint)\n", dir.string().c_str());
        return;
    }
    auto w = mt::Weights::load(model);
    w.config.sliding_keys = f.sliding_keys;
    w.config.top_k = 32; // HF reference selects config.centroid_intermediate_top_k = 32
    Rope rope(1024);
    mt::HostDrafter d(w, rope.view(), table.view());
    // Teacher-forced single steps: each depth's HF input (token, h) at position p, one engine depth, same argmax.
    std::size_t same = 0, total = 0, ties = 0, chain = 0, chain_total = 0;
    for (std::size_t ci = 0; ci < f.cases.size(); ++ci) {
        const auto& c = f.cases[ci];
        for (std::size_t i = 0; i < c.tokens.size(); ++i) {
            const auto h = std::span<const float>(f.step_hidden).subspan((ci * f.K + i) * 2560, 2560);
            const auto out = d.draft({c.inputs[i], c.p, h, f.kv.full_sliding(), f.kv.full()}, 1);
            ++total;
            if (out.tokens[0] == c.tokens[i]) { ++same; continue; }
            if (c.margins[i] < 1e-3) { ++ties; continue; }
            EAGLE_CHECK(false, "p %llu depth %zu engine %u HF %u (HF margin %g)", (unsigned long long)c.p, i + 1, out.tokens[0],
                        c.tokens[i], c.margins[i]);
        }
        // Free-running chain (informational: random synthetic KV makes long chains chaotic).
        const auto run = d.draft({c.anchor, c.p, std::span<const float>(f.hidden).subspan((c.p - 1) * 2560, 2560), f.kv.full_sliding(),
                                  f.kv.full()}, std::uint32_t(c.tokens.size()));
        for (std::size_t i = 0; i < c.tokens.size() && run.tokens[i] == c.tokens[i]; ++i) ++chain;
        chain_total += c.tokens.size();
    }
    std::printf("HF fixture (real weights): %zu/%zu teacher-forced steps identical (%zu near ties); free-running chain prefix %zu/%zu\n",
                same, total, ties, chain, chain_total);
    // The engine's 512-slot tagged ring equals the full array when sliding_keys = 512.
    w.config.sliding_keys = 512;
    mt::HostDrafter d512(w, rope.view(), table.view());
    for (const std::uint64_t p : {100ULL, 512ULL, 513ULL, 650ULL}) {
        mt::Inputs a{77, p, std::span<const float>(f.hidden).subspan((p - 1) * 2560, 2560), f.kv.full_sliding(), f.kv.full()};
        auto b = a;
        b.sliding = f.kv.ring(p);
        const auto x = d512.draft(a, 4), y = d512.draft(b, 4);
        EAGLE_CHECK(x.tokens == y.tokens && x.prob == y.prob, "ring view differs at p %llu", (unsigned long long)p);
    }
}

void session(const Table& table) {
    const auto w = mt::Weights::random(mt::Config{}, 9);
    Rope rope(1024);
    TargetKv kv(600, 4);
    mt::Session s(std::make_unique<mt::HostDrafter>(w, rope.view(), table.view()), {22, 23});
    std::vector<float> hidden(3 * 2560, 0.5F), taps(3 * 5 * 2560, 0.0F);
    const std::uint32_t next[] = {10, 11, 12};
    const std::uint64_t positions[] = {497, 498, 499};
    EAGLE_CHECK(s.propose_window(12, 500, 3) == (std::array<std::uint32_t, 7>{}), "no hidden yet: empty proposal");
    EAGLE_CHECK(!s.wants_taps(), "MTP sessions never read target taps (prompt prefill may skip them)");
    s.stage_rows(taps, hidden, next, positions);
    s.publish();
    EAGLE_CHECK(s.propose_window(12, 500, 3) == (std::array<std::uint32_t, 7>{}), "no KV source: empty proposal");
    s.set_source([&](std::uint32_t layer, mt::KvLayer& out) {
        out = layer == 22 ? kv.full_sliding() : layer == 23 ? kv.full() : mt::KvLayer{};
        return layer == 22 || layer == 23;
    });
    EAGLE_CHECK(s.propose_window(13, 500, 3) == (std::array<std::uint32_t, 7>{}), "anchor mismatch: empty proposal");
    const auto p = s.propose_window(12, 500, 3);
    EAGLE_CHECK(p[0] == s.last_draft().tokens[0] && s.last_draft().depth == 3 && p[3] == 0, "proposal window 3");
    EAGLE_CHECK(throws([&] { s.stage(taps, positions); }), "legacy stage refuses");
    ModelConfig t;
    t.layer.resize(42);
    for (std::uint32_t l = 0; l < 42; ++l) {
        t.layer[l].attention = (l % 6 == 5) ? AttentionKind::global : AttentionKind::sliding;
        t.layer[l].producer = l < 24 ? l : (t.layer[l].attention == AttentionKind::global ? 23 : 22);
    }
    const auto layers = mt::shared_kv_layers(t);
    EAGLE_CHECK(layers.first == 22 && layers.second == 23, "E4B shared KV layers %u %u", layers.first, layers.second);
}
bool same_bits(float a, float b) { return bit_cast<std::uint32_t>(a) == bit_cast<std::uint32_t>(b); }
// Drafter::extend(k) continues the last draft() to depth k: bit-identical to draft(in, k) from scratch.
void extend(const Table& table) {
    const auto w = mt::Weights::random(mt::Config{}, 9);
    Rope rope(1024);
    TargetKv kv(600, 4);
    std::vector<float> hidden(2560);
    Rng r{12};
    for (auto& x : hidden) x = 2 * r.uniform();
    mt::HostDrafter d(w, rope.view(), table.view()), fresh(w, rope.view(), table.view());
    const mt::Inputs in{77, 513, hidden, kv.ring(513), kv.full()};
    EAGLE_CHECK(throws([&] { (void)d.extend(2); }), "extend without a previous draft refuses");
    const auto equal = [](const mt::Draft& a, const mt::Draft& b, std::uint32_t k) {
        bool ok = a.depth == k && b.depth == k;
        for (std::uint32_t i = 0; i < k; ++i)
            ok = ok && a.tokens[i] == b.tokens[i] && same_bits(a.prob[i], b.prob[i]) && same_bits(a.margin[i], b.margin[i]);
        return ok;
    };
    std::size_t checked = 0;
    for (const auto& chain : {std::vector<std::uint32_t>{1, 3, 7}, std::vector<std::uint32_t>{2, 5}, std::vector<std::uint32_t>{2, 3, 4}}) {
        auto x = d.draft(in, chain[0]);
        EAGLE_CHECK(equal(x, fresh.draft(in, chain[0]), chain[0]), "draft(%u) repeatable", chain[0]);
        for (std::size_t i = 1; i < chain.size(); ++i) {
            x = d.extend(chain[i]);
            EAGLE_CHECK(equal(x, fresh.draft(in, chain[i]), chain[i]), "extend(%u) after %u == draft(in, %u)", chain[i], chain[i - 1], chain[i]);
            ++checked;
        }
        EAGLE_CHECK(throws([&] { (void)d.extend(chain.back()); }), "extend to the same depth refuses");
    }
    std::printf("mtp extend == draft from scratch: %zu extensions bit-identical (tokens, prob, margin)\n", checked);
#if defined(STRIX_G4_DRAFT_CONFIDENCE_API)
    // Session::propose_confident: chunked drafting through extend(); tokens are a prefix of the full chain.
    mt::Session s(std::make_unique<mt::HostDrafter>(w, rope.view(), table.view()), {22, 23});
    std::vector<float> taps(5 * 2560, 0.0F);
    const std::uint32_t next[] = {77};
    const std::uint64_t positions[] = {512};
    s.set_source([&](std::uint32_t layer, mt::KvLayer& out) {
        out = layer == 22 ? kv.ring(513) : kv.full();
        return true;
    });
    s.stage_rows(taps, hidden, next, positions);
    s.publish();
    const auto full = fresh.draft(in, 7);
    strix::gemma4::engine::ConfidenceConfig c;
    c.tau = 1e-30; c.max_k = 7;
    auto out = s.propose_confident(77, 513, 6, c);
    EAGLE_CHECK(out.drafted == 6 && out.window == 6, "tiny tau: drafted %u window %u", out.drafted, out.window);
    for (std::uint32_t i = 0; i < 6; ++i) EAGLE_CHECK(out.tokens[i] == full.tokens[i], "confident token %u", i);
    EAGLE_CHECK(out.tokens[6] == 0, "beyond the window: zero");
    c.tau = 1; // never reached by a random-weight drafter: first chunk only, window 1
    out = s.propose_confident(77, 513, 6, c);
    EAGLE_CHECK(out.drafted == 2 && out.window == 1 && out.tokens[0] == full.tokens[0] && out.tokens[1] == 0,
                "tau 1: drafted %u window %u", out.drafted, out.window);
    c.min1 = 1; c.tau = 1e-30;
    out = s.propose_confident(77, 513, 6, c);
    EAGLE_CHECK(out.window == 0 && out.tokens[0] == 0, "min1: no verified draft");
    EAGLE_CHECK(s.propose_confident(78, 513, 6, c).drafted == 0, "anchor mismatch: nothing drafted");
#endif
}
// r5b device kernels, host emulation of their changed reductions against the unchanged reference recipes.
//   top-k: the bitonic network of topk_kernel (mtp::bitonic_pair, the exact device comparator) == top_clusters
//          (repeated argmax), ids and order, with heavy ties, +-0 and non-power-of-two counts;
//   attention max: attention_partial's strided (value, row) argmax + tree (256 threads) returns the bits of the
//          serial "x > m" scan of the legacy kernel / attend_head, including +-0 ties;
//   statistics: set_statistics never changes tokens; prob/margin are bit-identical when requested, 0 when not.
void r5b_kernels(const Table& table) {
    Rng r{31};
    std::size_t lists = 0;
    for (const std::uint32_t count : {2048U, 1500U, 64U, 3U})
        for (int trial = 0; trial < 12; ++trial) {
            std::vector<float> logits(count);
            for (auto& x : logits) {
                const auto u = r.below(8);
                x = trial % 3 == 0 ? float(int(r.below(5)) - 2) // many exact ties
                    : u == 0 ? 0.0F : u == 1 ? -0.0F : r.uniform();
            }
            const auto k = std::min<std::uint32_t>(count, trial % 2 ? 64U : 32U);
            const auto want = mt::top_clusters(logits, k);
            std::uint32_t N = 2;
            while (N < count) N <<= 1U;
            std::vector<float> value(N, -std::numeric_limits<float>::infinity());
            std::vector<std::uint32_t> id(N);
            for (std::uint32_t i = 0; i < N; ++i) { id[i] = i; if (i < count) value[i] = logits[i]; }
            for (std::uint32_t size = 2; size <= N; size <<= 1U)
                for (std::uint32_t stride = size >> 1U; stride; stride >>= 1U)
                    for (std::uint32_t p = 0; p < N / 2U; ++p) mt::bitonic_pair(value.data(), id.data(), size, stride, p);
            bool ok = true;
            for (std::uint32_t i = 0; i < k; ++i) ok = ok && id[i] == want[i];
            EAGLE_CHECK(ok, "bitonic top-%u of %u (trial %d) differs from repeated argmax", k, count, trial);
            ++lists;
        }
    std::size_t maxima = 0;
    for (const std::uint32_t n : {1U, 2U, 7U, 255U, 256U, 257U, 1024U, 2048U})
        for (int trial = 0; trial < 20; ++trial) {
            std::vector<float> score(n);
            for (auto& x : score) {
                const auto u = r.below(6);
                x = trial % 2 ? (u < 3 ? (u == 0 ? 0.0F : -0.0F) : -r.uniform()) : float(int(r.below(4)) - 3) + (u == 0 ? -0.0F : 0.0F);
            }
            float serial = -std::numeric_limits<float>::infinity();
            for (const float x : score) serial = x > serial ? x : serial;
            constexpr std::uint32_t T = 256;
            std::vector<float> bv(T, -std::numeric_limits<float>::infinity());
            std::vector<std::uint32_t> bi(T, UINT32_MAX);
            for (std::uint32_t t = 0; t < T; ++t)
                for (std::uint32_t j = t; j < n; j += T)
                    if (score[j] > bv[t]) { bv[t] = score[j]; bi[t] = j; }
            for (std::uint32_t st = T / 2; st; st >>= 1U)
                for (std::uint32_t t = 0; t < st; ++t)
                    if (better(bv[t + st], bi[t + st], bv[t], bi[t])) { bv[t] = bv[t + st]; bi[t] = bi[t + st]; }
            EAGLE_CHECK(same_bits(bv[0], serial), "attention max n %u trial %d: tree %a serial %a", n, trial, double(bv[0]), double(serial));
            ++maxima;
        }
    // Statistics: tokens identical for every setting; requested statistics bit-identical to the full head.
    const auto w = mt::Weights::random(mt::Config{}, 9);
    Rope rope(1024);
    TargetKv kv(600, 4);
    std::vector<float> row(2560);
    for (auto& x : row) x = 2 * r.uniform();
    mt::HostDrafter d(w, rope.view(), table.view());
    std::size_t depths = 0;
    for (const std::uint64_t p : {1ULL, 64ULL, 513ULL, 599ULL}) {
        const mt::Inputs in{77, p, row, kv.ring(p), kv.full()};
        d.set_statistics(mt::statistic_all);
        const auto all = d.draft(in, 7);
        for (const std::uint32_t bits : {0U, mt::statistic_probability, mt::statistic_margin}) {
            d.set_statistics(bits);
            const auto x = d.draft(in, 7);
            for (std::uint32_t i = 0; i < 7; ++i) {
                EAGLE_CHECK(x.tokens[i] == all.tokens[i], "statistics %u p %llu depth %u token", bits, (unsigned long long)p, i + 1);
                EAGLE_CHECK(same_bits(x.prob[i], bits & mt::statistic_probability ? all.prob[i] : 0.0F) &&
                                same_bits(x.margin[i], bits & mt::statistic_margin ? all.margin[i] : 0.0F),
                            "statistics %u p %llu depth %u prob %a margin %a", bits, (unsigned long long)p, i + 1,
                            double(x.prob[i]), double(x.margin[i]));
            }
            depths += 7;
        }
    }
    std::printf("mtp r5b kernels (host emulation): %zu bitonic top-k lists == repeated argmax, %zu tree maxima == serial "
                "(+-0 ties), %zu depths: statistics off/partial keep tokens, requested statistics bit-identical\n",
                lists, maxima, depths);
}
// Release layout: <model dir>/drafter-mtp/model.safetensors; absent drafter -> empty path (decode runs exact).
void layout() {
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() / "e4b-mtp-layout-test";
    fs::remove_all(root);
    fs::create_directories(root / "model" / mt::drafter_subdirectory);
    fs::create_directories(root / "empty");
    std::ofstream(root / "model" / mt::drafter_subdirectory / "model.safetensors") << "x";
    std::ofstream(root / "model" / "Nemotron.gguf") << "x";
    const auto want = root / "model" / mt::drafter_subdirectory;
    EAGLE_CHECK(mt::resolve_drafter(want) == want, "drafter dir itself");
    EAGLE_CHECK(mt::resolve_drafter(root / "model") == want, "model dir");
    EAGLE_CHECK(mt::resolve_drafter(root / "model" / "Nemotron.gguf") == want, "model file");
    EAGLE_CHECK(mt::resolve_drafter(root / "empty").empty() && mt::resolve_drafter(root / "missing").empty(), "absent drafter");
    EAGLE_CHECK(throws([&] { (void)mt::Weights::load(root / "empty"); }), "load refuses an absent drafter");
    fs::remove_all(root);
}
// Confidence-window defaults per drafter: config.json keys (validated) and the builtin tau of known weights.
void confidence_defaults() {
    const auto k = mt::confidence_keys(R"({"model_type":"gemma4_assistant","confidence_tau":0.35,"confidence_max_k":6,"confidence_first":1})");
    EAGLE_CHECK(k.tau == 0.35F && k.max_k == 6 && k.first == 1, "config keys");
    const auto none = mt::confidence_keys(R"({"model_type":"gemma4_assistant"})");
    EAGLE_CHECK(none.tau == 0 && none.max_k == 0 && none.first == 0, "absent keys");
    for (const char* bad : {R"({"confidence_tau":0})", R"({"confidence_tau":1.5})", R"({"confidence_tau":"x"})",
                            R"({"confidence_max_k":8})", R"({"confidence_first":0})", R"({"confidence_max_k":2.5})"})
        EAGLE_CHECK(throws([&] { (void)mt::confidence_keys(bad); }), "rejects %s", bad);
    EAGLE_CHECK(mt::builtin_confidence_tau("12875062fc25c51e8fa9b62abd2de7ad48b7d63f8559d5d604fbd5a3d6bcff16") > 0, "untuned builtin");
    EAGLE_CHECK(mt::builtin_confidence_tau("19594909271ef877aa") > 0, "ft1 builtin");
    EAGLE_CHECK(mt::builtin_confidence_tau("deadbeef") == 0, "unknown weights");
#if defined(STRIX_G4_DRAFT_CONFIDENCE_API)
    strix::gemma4::engine::ConfidenceConfig env; env.tau = 0.6; env.per_drafter = true; env.source = strix::gemma4::engine::ConfidenceConfig::Source::builtin;
    const auto r = env.resolved(k.tau, k.max_k, k.first, 0.3);
    EAGLE_CHECK(r.tau == double(0.35F) && r.max_k == 6 && r.first == 1, "config keys win");
#endif
    std::printf("mtp: confidence defaults (config keys, builtin per-weights tau, precedence) OK\n");
}
} // namespace

int main(int argc, char** argv) {
    try {
        recipes();
        layout();
        confidence_defaults();
        const Table table;
        if (argc > 2) fixture(argv[1], argv[2], table);
        else std::printf("SKIP fixture (usage: fixture_dir assistant_dir)\n");
        session(table);
        extend(table);
        r5b_kernels(table);
        for (int i = 3; i < argc; ++i) { // optional extra checkpoints (e.g. fine-tuned exports): load + validate
            const auto w = mt::Weights::load(argv[i]);
            std::printf("loaded %s: hidden %u ffn %u full=%d%d%d%d scalars %.4g %.4g %.4g %.4g\n", argv[i], w.config.hidden,
                        w.config.ffn, w.config.full[0], w.config.full[1], w.config.full[2], w.config.full[3], double(w.layer[0].scalar),
                        double(w.layer[1].scalar), double(w.layer[2].scalar), double(w.layer[3].scalar));
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf(eagle_test::failures ? "FAILED %d checks\n" : "ALL OK\n", eagle_test::failures);
    return eagle_test::failures ? 1 : 0;
}
