// Prompt lookup (STRIX_G4_LOOKUP): the n-gram index against a brute-force reference, and the hybrid session through the
// real engine::Coordinator on a toy target whose greedy continuation repeats (so lookup proposals are accepted):
// ids == OFF at C1 (confidence path and AutoPolicy path) and at 3 streams.
#include "common.hpp"
#include "gemma4/engine/coordinator.hpp"
#include "gemma4/engine/lookup.hpp"
#include "gemma4/eagle/mtp/session.hpp"
#include "mock.hpp"

#include <cstdio>
#include <cstdlib>

#if !defined(STRIX_G4_DRAFT_ROWS_API)
int main() {
    std::printf("SKIP: staged-rows draft API is not applied\n");
    return 0;
}
#else
using namespace eagle_test;
namespace en = strix::gemma4::engine;
namespace lk = strix::gemma4::engine::lookup;

namespace {
void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}
#define LK_CHECK(c, ...) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); std::exit(1); } } while (0)

// Reference: largest level with an earlier occurrence (most recent), then exact backward extension.
lk::Match brute(const std::vector<std::uint32_t>& s, std::uint32_t min_length, std::uint32_t want, std::uint32_t max_match,
                std::uint64_t max_end = ~0ULL) {
    lk::Match m;
    if (s.size() < 2) return m;
    const std::uint64_t cur = s.size() - 1;
    for (const std::uint32_t L : {16U, 12U, 8U, 6U, 4U, 3U, 2U, 1U}) {
        if (L < min_length || cur + 1 < L) continue;
        for (std::uint64_t e = max_end >= cur ? cur : max_end + 1; e-- > 0;) {
            if (e + 1 < L) break;
            bool same = true;
            for (std::uint32_t j = 0; j < L && same; ++j) same = s[e - j] == s[cur - j];
            if (!same) continue;
            std::uint32_t len = L;
            while (len < max_match && e >= len && s[e - len] == s[cur - len]) ++len;
            m.length = len; m.distance = cur - e;
            for (std::uint64_t i = e + 1; i <= cur && m.available < want; ++i) m.next[m.available++] = s[i];
            return m;
        }
    }
    return m;
}
void index_tests() {
    Rng r{77};
    std::uint64_t queries = 0, hits = 0;
    for (const std::uint32_t vocab : {3U, 7U, 40U}) {
        auto index = lk::make_ngram_index({4096, 64});
        std::vector<std::uint32_t> s;
        for (int i = 0; i < 1500; ++i) {
            const auto t = std::uint32_t(r.uniform() * float(vocab)) % vocab;
            s.push_back(t);
            index->append(t);
            for (const std::uint32_t minl : {1U, 2U, 3U}) {
                const auto a = index->longest({}, minl, 7), b = brute(s, minl, 7, 64);
                LK_CHECK(a.length == b.length && a.distance == b.distance && a.available == b.available,
                         "vocab %u pos %d min %u: index L %u d %llu n %u vs brute L %u d %llu n %u", vocab, i, minl, a.length,
                         (unsigned long long)a.distance, a.available, b.length, (unsigned long long)b.distance, b.available);
                for (std::uint32_t k = 0; k < a.available; ++k) LK_CHECK(a.next[k] == b.next[k], "continuation");
                ++queries; hits += a.length ? 1 : 0;
            }
            // extra (hypothetical) tokens: suffix of stream + extra
            if (i % 50 == 49) {
                const std::uint32_t extra[2] = {s[s.size() - 3], s[s.size() - 2]};
                auto t2 = s; t2.push_back(extra[0]); t2.push_back(extra[1]);
                const auto a = index->longest(extra, 2, 7), b = brute(t2, 2, 7, 64, s.size() - 2);
                LK_CHECK(a.length == b.length && a.available == b.available, "extra tokens");
            }
        }
    }
    // Bounded memory: positions older than the capacity are forgotten, never proposed.
    auto small = lk::make_ngram_index({64, 64});
    for (std::uint32_t i = 0; i < 200; ++i) small->append(i % 150);
    LK_CHECK(small->longest({}, 1, 7).length == 0, "evicted occurrence must not match");
    std::printf("lookup: index == brute force on %llu queries (%llu with a match), extra-token queries, bounded capacity OK\n",
                (unsigned long long)queries, (unsigned long long)hits);
}

// Toy target with a repeating greedy continuation: greedy(row) depends on the last two tokens only.
struct LocalTarget final : Target {
    std::uint32_t hidden = 64, vocab = 512;
    std::uint32_t greedy(std::uint32_t a, std::uint32_t b) const { return 2 + (b * 7 + (a % 3)) % 23; }
    ForwardResult forward_rows(std::span<const RowPlan> plans) override {
        ForwardResult f;
        f.rows = plans.size(); f.vocab = vocab; f.tap_width = 5 * hidden;
        f.logits.assign(f.rows * vocab, 0.0F); f.taps.assign(f.rows * f.tap_width, 0.5F);
        bool want = false;
        for (const auto& p : plans) want = want || p.export_hidden;
        if (want) f.hidden.assign(f.rows * hidden, 0.25F);
        for (const auto& p : plans) {
            auto* tx = static_cast<MockTxn*>(p.transaction);
            if (p.position != tx->start + tx->tokens.size()) throw std::runtime_error("rows must be contiguous");
            tx->tokens.push_back(p.token);
            std::vector<std::uint32_t> all = tx->s->tokens;
            all.insert(all.end(), tx->tokens.begin(), tx->tokens.end());
            const auto a = all.size() >= 2 ? all[all.size() - 2] : 0U, b = all.back();
            f.logits[p.output_row * vocab + greedy(a, b)] = 1.0F;
        }
        return f;
    }
};
struct State final : en::StreamState {
    MockStream s;
    std::unique_ptr<KvTxn> begin() override { return s.begin(); }
    std::uint64_t frontier() const override { return s.tokens.size(); }
    std::string digest() const override { return std::to_string(s.hashes.back()); }
    bool kv_view(std::uint32_t, const KvTxn*, KvView& out) const override { out = {}; return true; }
};
struct Backend final : en::Backend {
    LocalTarget target;
    ModelConfig c;
    Backend() { c.hidden = target.hidden; c.vocab = target.vocab; }
    std::unique_ptr<en::StreamState> create_stream() override { return std::make_unique<State>(); }
    ForwardResult forward_rows(std::span<const RowPlan> rows) override { return target.forward_rows(rows); }
    const ModelConfig& config() const override { return c; }
    strix::serve::Json identity() const override { return strix::serve::Json::object(); }
};
// A confidence-capable inner drafter that always proposes one wrong token: the hybrid must learn that lookup beats it.
struct WeakInner final : en::DraftSession {
    std::array<std::uint32_t, 7> propose(std::uint32_t, std::uint64_t) override { return {}; }
    void append(std::span<const float>, std::span<const std::uint64_t>) override {}
    void stage_rows(std::span<const float>, std::span<const float>, std::span<const std::uint32_t>, std::span<const std::uint64_t>) override {}
    void publish() override {}
    void discard() noexcept override {}
    [[nodiscard]] bool confidence_capable() const override { return true; }
    Confident propose_confident(std::uint32_t, std::uint64_t, std::uint32_t, const en::ConfidenceConfig&) override {
        Confident c; c.tokens[0] = 0; c.window = c.drafted = 1; return c;
    }
};
// A real MTP session (its drafter never runs here): the shipped default wraps it.
struct NoDraft final : strix::gemma4::eagle::mtp::Drafter {
    strix::gemma4::eagle::mtp::Config c;
    strix::gemma4::eagle::mtp::Timings t;
    const strix::gemma4::eagle::mtp::Config& config() const override { return c; }
    const strix::gemma4::eagle::mtp::Timings& timings() const override { return t; }
    strix::gemma4::eagle::mtp::Draft draft(const strix::gemma4::eagle::mtp::Inputs&, std::uint32_t) override { return {}; }
};
// A taps-consuming inner session (DFlash / e4b-eagle shape).
struct TapsInner final : en::DraftSession {
    std::array<std::uint32_t, 7> propose(std::uint32_t, std::uint64_t) override { return {}; }
    void append(std::span<const float>, std::span<const std::uint64_t>) override {}
    void publish() override {}
    void discard() noexcept override {}
};
// Prompt prefill skips the target-tap copy when the session does not want taps: lookup never reads them, so the
// hybrid forwards the inner drafter's answer (MTP: false) and answers false on its own.
void taps_tests() {
    lk::LookupConfig cfg;
    LK_CHECK(!lk::hybrid_factory({}, cfg)()->wants_taps(), "lookup alone reads no taps");
    const en::DraftFactory mtp = [] {
        return std::make_unique<strix::gemma4::eagle::mtp::Session>(std::make_unique<NoDraft>(), std::pair<std::uint32_t, std::uint32_t>{22, 23});
    };
    LK_CHECK(!mtp()->wants_taps(), "MTP session reads no taps");
    LK_CHECK(!lk::hybrid_factory(mtp, cfg)()->wants_taps(), "lookup + MTP reads no taps (prefill tap copy stays skipped)");
    LK_CHECK(lk::hybrid_factory(mtp, cfg)()->wants_hidden(), "lookup + MTP keeps the MTP hidden rows");
    const en::DraftFactory taps = [] { return std::make_unique<TapsInner>(); };
    LK_CHECK(lk::hybrid_factory(taps, cfg)()->wants_taps(), "a wrapped taps consumer keeps its taps");
    // Runtime admission: no drafter / MTP wrapped; DFlash and e4b-eagle never wrapped (their sessions, wants_taps
    // included, are unchanged); drafting off or STRIX_G4_LOOKUP=0 never wraps.
    LK_CHECK(lk::wraps(cfg, false, false, false) && lk::wraps(cfg, true, true, false), "wraps no drafter and MTP");
    LK_CHECK(!lk::wraps(cfg, true, false, false), "DFlash / e4b-eagle not wrapped");
    LK_CHECK(!lk::wraps(cfg, false, false, true) && !lk::wraps(cfg, true, true, true), "drafting off: not wrapped");
    lk::LookupConfig off; off.enabled = false;
    LK_CHECK(!lk::wraps(off, false, false, false) && !lk::wraps(off, true, true, false), "STRIX_G4_LOOKUP=0: not wrapped");
}
// A strong confidence-window drafter (MTP-like): every cycle a full window of 7 whose first 3 tokens are right
// (4 tokens per cycle). On the repeating toy stream lookup is right through all 7 depths, so the selector must
// learn to prefer it within one short reply.
struct GoodInner final : en::DraftSession {
    std::vector<std::uint32_t> committed, staged;
    std::array<std::uint32_t, 7> propose(std::uint32_t, std::uint64_t) override { return {}; }
    void append(std::span<const float>, std::span<const std::uint64_t>) override {}
    void stage_rows(std::span<const float>, std::span<const float>, std::span<const std::uint32_t> next,
                    std::span<const std::uint64_t>) override { staged.assign(next.begin(), next.end()); }
    void publish() override { committed.insert(committed.end(), staged.begin(), staged.end()); staged.clear(); }
    void discard() noexcept override { staged.clear(); }
    [[nodiscard]] bool confidence_capable() const override { return true; }
    Confident propose_confident(std::uint32_t anchor, std::uint64_t, std::uint32_t kmax, const en::ConfidenceConfig&) override {
        Confident c;
        if (committed.size() < 2 || kmax < 7) return c;
        std::uint32_t a = committed[committed.size() - 2], b = anchor;
        const LocalTarget t;
        for (std::uint32_t d = 0; d < 7; ++d) {
            const auto g = t.greedy(a, b);
            c.tokens[d] = d < 3 ? g : 1;
            a = b; b = g;
        }
        c.window = c.drafted = 7;
        return c;
    }
};
struct Run { std::vector<std::vector<std::uint32_t>> ids; std::uint64_t cycles = 0, accepted = 0, proposed = 0; };
Run decode(const en::DraftFactory& f, const std::vector<std::vector<std::uint32_t>>& prompts, std::size_t streams, en::DraftMode mode) {
    Run run;
    for (std::size_t i = 0; i < prompts.size(); i += streams) {
        Backend b;
        en::Coordinator coordinator(b, f);
        std::vector<std::size_t> slots;
        for (std::size_t j = i; j < std::min(prompts.size(), i + streams); ++j) {
            en::Request req;
            req.prompt = prompts[j];
            req.max_tokens = 160;
            req.dflash = mode;
            req.stop_ids.clear();
            slots.push_back(coordinator.submit(req));
        }
        coordinator.run();
        for (const auto slot : slots) {
            run.ids.push_back(coordinator.result(slot).ids);
            const auto receipt = coordinator.receipt(slot);
            if (const auto* p = receipt.find("prompt_lookup")) {
                run.cycles += std::uint64_t(p->find("cycles")->as_int());
                run.accepted += std::uint64_t(p->find("accepted")->as_int());
                run.proposed += std::uint64_t(p->find("proposed")->as_int());
            }
        }
    }
    return run;
}
void coordinator_tests() {
    Rng r{5};
    std::vector<std::vector<std::uint32_t>> prompts;
    for (const std::size_t n : {std::size_t(1), std::size_t(2), std::size_t(9), std::size_t(17), std::size_t(40)}) {
        std::vector<std::uint32_t> p;
        for (std::size_t i = 0; i < n; ++i) p.push_back(2 + std::uint32_t(r.uniform() * 300) % 300);
        prompts.push_back(p);
    }
    set_env("STRIX_G4_LOOKUP", "1");
    const auto cfg = lk::LookupConfig::from_environment();
    LK_CHECK(cfg.enabled && cfg.min_length == 2, "config");
    set_env("STRIX_G4_LOOKUP", nullptr);
    const auto builtin = lk::LookupConfig::from_environment();
    LK_CHECK(builtin.enabled && builtin.min_length == 2, "default: on");
    set_env("STRIX_G4_LOOKUP", "0");
    LK_CHECK(!lk::LookupConfig::from_environment().enabled, "STRIX_G4_LOOKUP=0 off");
    set_env("STRIX_G4_LOOKUP", nullptr);
    const auto off = decode({}, prompts, 1, en::DraftMode::off);
    const en::DraftFactory lookup_only = lk::hybrid_factory({}, cfg);
    for (const char* conf : {static_cast<const char*>(nullptr), "0"}) { // confidence path (default) and AutoPolicy path
        set_env("STRIX_G4_MTP_CONF", conf);
        const auto on = decode(lookup_only, prompts, 1, en::DraftMode::automatic);
        for (std::size_t i = 0; i < prompts.size(); ++i) LK_CHECK(on.ids[i] == off.ids[i], "C1 lookup ids == OFF (prompt %zu)", i);
        if (!conf) LK_CHECK(on.cycles > 0 && on.accepted > 0, "lookup drafted and accepted (%llu cycles)", (unsigned long long)on.cycles);
        std::printf("lookup: C1 %s: ids == OFF, %llu lookup cycles, %llu/%llu accepted\n", conf ? "STRIX_G4_MTP_CONF=0 (AutoPolicy)" : "confidence path",
                    (unsigned long long)on.cycles, (unsigned long long)on.accepted, (unsigned long long)on.proposed);
    }
    set_env("STRIX_G4_MTP_CONF", nullptr);
    const en::DraftFactory weak = [] { return std::make_unique<WeakInner>(); };
    const auto hybrid = decode(lk::hybrid_factory(weak, cfg), prompts, 1, en::DraftMode::automatic);
    for (std::size_t i = 0; i < prompts.size(); ++i) LK_CHECK(hybrid.ids[i] == off.ids[i], "hybrid ids == OFF (prompt %zu)", i);
    LK_CHECK(hybrid.cycles > 0 && hybrid.accepted > 0, "hybrid chose lookup over a useless inner drafter");
    std::printf("lookup: hybrid over a weak inner drafter: ids == OFF, %llu lookup cycles, %llu/%llu accepted\n",
                (unsigned long long)hybrid.cycles, (unsigned long long)hybrid.accepted, (unsigned long long)hybrid.proposed);
    const en::DraftFactory good = [] { return std::make_unique<GoodInner>(); };
    const auto strong = decode(lk::hybrid_factory(good, cfg), prompts, 1, en::DraftMode::automatic);
    for (std::size_t i = 0; i < prompts.size(); ++i) LK_CHECK(strong.ids[i] == off.ids[i], "hybrid (strong inner) ids == OFF (prompt %zu)", i);
    LK_CHECK(strong.cycles > 0 && strong.accepted >= 6 * strong.cycles,
             "hybrid chose full-length lookup windows over a strong 4-token inner drafter (%llu cycles, %llu accepted)",
             (unsigned long long)strong.cycles, (unsigned long long)strong.accepted);
    std::printf("lookup: hybrid over a strong inner drafter: ids == OFF, %llu lookup cycles, %llu/%llu accepted\n",
                (unsigned long long)strong.cycles, (unsigned long long)strong.accepted, (unsigned long long)strong.proposed);
    const auto multi = decode(lookup_only, prompts, 3, en::DraftMode::automatic);
    for (std::size_t i = 0; i < prompts.size(); ++i) LK_CHECK(multi.ids[i] == off.ids[i], "3 streams ids == OFF (prompt %zu)", i);
    std::printf("lookup: 3 streams ids == OFF\n");

    // STRIX_SPEC_ARBITER=1: the shared strix::spec proposer + arbiter decide the C1 cycles; ids == OFF everywhere.
    set_env("STRIX_SPEC_ARBITER", "1");
    const auto spec_cfg = lk::LookupConfig::from_environment();
    set_env("STRIX_SPEC_ARBITER", nullptr);
    LK_CHECK(spec_cfg.spec_arbiter && !cfg.spec_arbiter, "STRIX_SPEC_ARBITER parsed, default off");
    const struct { const char* name; en::DraftFactory inner; std::size_t streams; bool strong; } arms[] = {
        {"lookup only", {}, 1, false}, {"weak inner", weak, 1, false}, {"strong inner", good, 1, true},
        {"lookup only, 3 streams", {}, 3, false}};
    for (const auto& arm : arms) {
        const auto s = decode(lk::hybrid_factory(arm.inner, spec_cfg), prompts, arm.streams, en::DraftMode::automatic);
        for (std::size_t i = 0; i < prompts.size(); ++i)
            LK_CHECK(s.ids[i] == off.ids[i], "spec arbiter (%s) ids == OFF (prompt %zu)", arm.name, i);
        if (arm.streams == 1 && !arm.strong) LK_CHECK(s.cycles > 0 && s.accepted > 0, "spec arbiter (%s) used lookup", arm.name);
        std::printf("lookup: spec arbiter, %s: ids == OFF, %llu lookup cycles, %llu/%llu accepted\n", arm.name,
                    (unsigned long long)s.cycles, (unsigned long long)s.accepted, (unsigned long long)s.proposed);
    }
}
} // namespace

int main() {
    try {
        index_tests();
        taps_tests();
        coordinator_tests();
        std::printf("lookup tests OK\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "lookup tests: %s\n", e.what());
        return 1;
    }
}
#endif
