// e4b-eagle through the real engine::Coordinator (needs the staged-rows draft API). A row-invariant mock backend decodes the
// same prompts OFF and with e4b-eagle sessions (oracle drafter for acceptance, host reference drafter for the real
// code path), single and concurrent streams, DraftMode::on with k = 1..7; ids must equal OFF exactly.
#include "common.hpp"
#include "gemma4/eagle/session.hpp"
#include "gemma4/eagle/mtp/session.hpp"
#include "gemma4/engine/coordinator.hpp"
#include "mock.hpp"

#if !defined(STRIX_G4_DRAFT_ROWS_API)
int main() {
    std::printf("SKIP: staged-rows draft API (DraftSession propose_window/wants_hidden/stage_rows) is not applied\n");
    return 0;
}
#else
using namespace eagle_test;
namespace eg = strix::gemma4::eagle;
namespace en = strix::gemma4::engine;

namespace {
struct State final : en::StreamState {
    MockStream s;
    std::unique_ptr<KvTxn> begin() override { return s.begin(); }
    std::uint64_t frontier() const override { return s.tokens.size(); }
    std::string digest() const override { return std::to_string(s.hashes.back()); }
    // MTP sessions need a KV view to draft; the scripted MTP drafter below never reads it (eagle sessions ignore it).
    bool kv_view(std::uint32_t, const KvTxn*, KvView& out) const override { out = {}; return true; }
};
struct Backend final : en::Backend {
    MockTarget target;
    ModelConfig c;
    Backend() { c.hidden = target.hidden; c.vocab = target.vocab; }
    State* last = nullptr; // the coordinator creates the stream right before calling the draft factory
    std::unique_ptr<en::StreamState> create_stream() override {
        auto state = std::make_unique<State>();
        last = state.get();
        return state;
    }
    ForwardResult forward_rows(std::span<const RowPlan> rows) override { return target.forward_rows(rows); }
    const ModelConfig& config() const override { return c; }
    strix::serve::Json identity() const override { return strix::serve::Json::object(); }
};
struct Fixture {
    Config c = small_config();
    Weights w = Weights::random(c, 31);
    eagle::Matrix emb = embedding_table(c, 32);
    std::vector<float> rope = eg::make_rope_table(1024);
};

#if defined(STRIX_G4_DRAFT_MULTI_API)
void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}
// Multi-stream: forwards every hook to an eagle::Session and declares (or not) a per-depth-cost drafter, as the MTP session
// does (candidate) and DFlash does not (default false).
struct MultiCandidate final : en::DraftSession {
    std::unique_ptr<en::DraftSession> inner;
    bool candidate;
    MultiCandidate(std::unique_ptr<en::DraftSession> s, bool c) : inner(std::move(s)), candidate(c) {}
    std::array<std::uint32_t, 7> propose(std::uint32_t a, std::uint64_t p) override { return inner->propose(a, p); }
    std::array<std::uint32_t, 7> propose_window(std::uint32_t a, std::uint64_t p, std::uint32_t k) override {
        return inner->propose_window(a, p, k);
    }
    void append(std::span<const float> t, std::span<const std::uint64_t> p) override { inner->append(t, p); }
    void attach(en::StreamState& state) override { inner->attach(state); }
    bool wants_hidden() const override { return inner->wants_hidden(); }
    void stage_rows(std::span<const float> t, std::span<const float> h, std::span<const std::uint32_t> n,
                    std::span<const std::uint64_t> p) override { inner->stage_rows(t, h, n, p); }
    void stage(std::span<const float> t, std::span<const std::uint64_t> p) override { inner->stage(t, p); }
    void publish() override { inner->publish(); }
    void discard() noexcept override { inner->discard(); }
    bool multi_stream_candidate() const override { return candidate; }
};
#endif
#if defined(STRIX_G4_DRAFT_CONFIDENCE_API) && defined(STRIX_G4_DRAFT_MULTI_API)
namespace mt = strix::gemma4::eagle::mtp;
// MTP-like scripted drafter: the mock target's greedy continuation with injected errors and scripted top-1
// probabilities, all functions of the chain hash (deterministic, so extend() == draft() from scratch). Logs every
// draft/extend call (depth after the call; extend uses mt::Drafter's default re-run).
struct ScriptedMtp final : mt::Drafter {
    const MockTarget& t;
    MockStream& s;
    mt::Config c;
    mt::Timings times;
    std::vector<std::pair<bool, std::uint32_t>>* log;
    ScriptedMtp(const MockTarget& target, MockStream& stream, std::vector<std::pair<bool, std::uint32_t>>* calls)
        : t(target), s(stream), log(calls) { c.backbone = target.hidden; }
    const mt::Config& config() const override { return c; }
    const mt::Timings& timings() const override { return times; }
    mt::Draft draft(const mt::Inputs& in, std::uint32_t k) override {
        remember(in, k);
        mt::Draft d;
        d.depth = k;
        std::uint64_t h = MockStream::mix(s.hashes.at(in.position), in.anchor);
        static constexpr float script[] = {0.97F, 0.92F, 0.85F, 0.7F, 0.45F, 0.2F};
        for (std::uint32_t i = 0; i < k; ++i) {
            const auto g = t.greedy(h);
            d.prob[i] = script[(h >> 20U) % 6];
            d.tokens[i] = d.prob[i] < 0.5F && (h >> 9U) % 3 == 0 ? (g + 7) % t.vocab : g; // low confidence: some errors
            h = MockStream::mix(h, d.tokens[i]);
        }
        if (log) log->push_back({false, k});
        return d;
    }
    mt::Draft extend(std::uint32_t k) override {
        auto d = mt::Drafter::extend(k); // the default re-runs draft(), which logs; re-tag that entry as an extension
        if (log) log->back().first = true;
        return d;
    }
};
struct ConfidentRecord {
    std::array<float, 7> prob{};
    std::uint32_t kmax = 0, window = 0, drafted = 0, first = 0;
    double tau = 0, verify_tau = 0;
    bool rule = true;
};
// Forwards to an mtp::Session and records each confident proposal with the drafter's probabilities.
struct Recording final : en::DraftSession {
    std::unique_ptr<mt::Session> inner;
    std::vector<ConfidentRecord>* records;
    // > 0: every lookup_every-th proposal is reported as not chosen by the confidence rule (as a prompt-lookup
    // proposal is), so it must not feed the measured cost ratio nor count as a long-context cycle.
    std::uint32_t lookup_every = 0, calls = 0;
    Recording(std::unique_ptr<mt::Session> s, std::vector<ConfidentRecord>* r) : inner(std::move(s)), records(r) {}
    std::array<std::uint32_t, 7> propose(std::uint32_t a, std::uint64_t p) override { return inner->propose(a, p); }
    std::array<std::uint32_t, 7> propose_window(std::uint32_t a, std::uint64_t p, std::uint32_t k) override {
        return inner->propose_window(a, p, k);
    }
    void append(std::span<const float> t, std::span<const std::uint64_t> p) override { inner->append(t, p); }
    void attach(en::StreamState& state) override { inner->attach(state); }
    bool wants_hidden() const override { return inner->wants_hidden(); }
    void stage_rows(std::span<const float> t, std::span<const float> h, std::span<const std::uint32_t> n,
                    std::span<const std::uint64_t> p) override { inner->stage_rows(t, h, n, p); }
    void stage(std::span<const float> t, std::span<const std::uint64_t> p) override { inner->stage(t, p); }
    void publish() override { inner->publish(); }
    void discard() noexcept override { inner->discard(); }
    bool multi_stream_candidate() const override { return inner->multi_stream_candidate(); }
    bool confidence_capable() const override { return inner->confidence_capable(); }
    Confident propose_confident(std::uint32_t a, std::uint64_t p, std::uint32_t kmax, const en::ConfidenceConfig& c) override {
        auto out = inner->propose_confident(a, p, kmax, c);
        if (lookup_every && ++calls % lookup_every == 0) out.rule = false;
        ConfidentRecord r;
        r.rule = out.rule;
        r.kmax = kmax; r.window = out.window; r.drafted = out.drafted;
        r.first = c.first; r.tau = c.tau; r.verify_tau = c.verify_tau;
        for (std::uint32_t i = 0; i < out.drafted; ++i) r.prob[i] = inner->last_draft().prob[i];
        records->push_back(r);
        return out;
    }
};
#endif
std::vector<std::vector<std::uint32_t>> decode(Backend& backend, en::DraftFactory drafts, const std::vector<std::vector<std::uint32_t>>& prompts,
                                               std::uint32_t k, en::DraftMode mode, dflash::Acceptance* total) {
    en::Coordinator coordinator(backend, std::move(drafts));
    std::vector<std::size_t> slots;
    for (const auto& p : prompts) {
        en::Request r;
        r.prompt = p;
        r.max_tokens = 150;
        r.dflash = mode;
        r.draft_k = k;
        slots.push_back(coordinator.submit(r));
    }
    coordinator.run();
    std::vector<std::vector<std::uint32_t>> out;
    for (const auto slot : slots) {
        const auto& r = coordinator.result(slot);
        EAGLE_CHECK(r.status == en::Status::stopped || r.status == en::Status::length, "status %s error %s", en::status_name(r.status),
                    r.error.c_str());
        out.push_back(r.ids);
        if (total) { total->attempted += r.acceptance.attempted; total->accepted += r.acceptance.accepted; }
    }
    return out;
}
} // namespace

int main() {
    try {
        // The confidence window is DEFAULT ON for MTP sessions; every section except the confidence one below checks the
        // AutoPolicy path, so it runs with the OFF switch.
        set_env("STRIX_G4_MTP_CONF", "0");
        Fixture fx;
        Rng r{41};
        // Prompt lengths at the prefill-window edges (the coordinator prefills prompt[0..n-2] in <= 8-row windows; the
        // last prompt token is the first decode anchor): 1, 2, 8, 9, 16, 17 plus two random lengths.
        std::vector<std::vector<std::uint32_t>> prompts;
        for (const std::size_t n : {std::size_t(1 + r.below(40)), std::size_t(1), std::size_t(2), std::size_t(8), std::size_t(9),
                                    std::size_t(16), std::size_t(17), std::size_t(1 + r.below(40))}) {
            std::vector<std::uint32_t> p(n);
            for (auto& t : p) t = 2 + r.below(1000);
            prompts.push_back(p);
        }
        Backend off_backend;
        std::vector<std::vector<std::uint32_t>> off;
        for (const auto& p : prompts) off.push_back(decode(off_backend, {}, {p}, 0, en::DraftMode::off, nullptr)[0]);
        dflash::Acceptance oracle_stats, real_stats;
        for (std::uint32_t k = 1; k <= 7; k += 3) {
            // Oracle drafter inside an eagle::Session: exercises acceptance, bonus and next-token staging.
            Backend b1;
            en::DraftFactory oracle = [&b1]() -> std::unique_ptr<en::DraftSession> {
                return std::make_unique<eg::Session>(std::make_unique<Oracle>(b1.target, b1.last->s, small_config()));
            };
            const auto exact = decode(b1, oracle, {prompts[1]}, k, en::DraftMode::on, &oracle_stats);
            EAGLE_CHECK(exact[0] == off[1], "k %u oracle ids differ from OFF", k);
            // Real code path: host reference eagle drafter (random weights; acceptance near zero but every hook runs).
            Backend b2;
            auto factory = eg::host_drafts(fx.w, {fx.rope, 1024}, fx.emb.view());
            for (std::size_t i = 0; i < prompts.size(); ++i) {
                const auto single = decode(b2, factory, {prompts[i]}, k, en::DraftMode::on, &real_stats);
                EAGLE_CHECK(single[0] == off[i], "k %u prompt length %zu single-stream ids differ from OFF", k, prompts[i].size());
                Backend b4;
                const auto exact1 = decode(b4, [&b4]() -> std::unique_ptr<en::DraftSession> {
                    return std::make_unique<eg::Session>(std::make_unique<Oracle>(b4.target, b4.last->s, small_config()));
                }, {prompts[i]}, k, en::DraftMode::on, &oracle_stats);
                EAGLE_CHECK(exact1[0] == off[i], "k %u prompt length %zu oracle ids differ from OFF", k, prompts[i].size());
            }
            Backend b3;
            const std::vector<std::vector<std::uint32_t>> three{prompts[2], prompts[4], prompts[6]};
            const auto multi = decode(b3, factory, three, k, en::DraftMode::on, &real_stats);
            const std::size_t idx[] = {2, 4, 6};
            for (std::size_t i = 0; i < 3; ++i) EAGLE_CHECK(multi[i] == off[idx[i]], "k %u stream %zu ids differ from OFF", k, i);
        }
#if defined(STRIX_G4_DRAFT_MULTI_API)
        // multi-stream auto through the real coordinator: 4 concurrent auto streams, ids == OFF always; a
        // per-depth-cost drafter drafts at >1 stream with window <= 1 (probes at least), a DFlash-like drafter and
        // STRIX_G4_DRAFT_MULTI=0 never draft while >1 stream is live.
        for (const bool candidate : {true, false})
            for (const char* env : {static_cast<const char*>(nullptr), "0"}) {
                set_env("STRIX_G4_DRAFT_MULTI", env);
                Backend b5;
                en::DraftFactory f = [&b5, candidate]() -> std::unique_ptr<en::DraftSession> {
                    return std::make_unique<MultiCandidate>(
                        std::make_unique<eg::Session>(std::make_unique<Oracle>(b5.target, b5.last->s, small_config())), candidate);
                };
                en::Coordinator coordinator(b5, f);
                const std::size_t idx[] = {0, 2, 4, 6};
                std::vector<std::size_t> slots;
                for (const auto i : idx) {
                    en::Request req;
                    req.prompt = prompts[i];
                    req.max_tokens = 150;
                    req.dflash = en::DraftMode::automatic;
                    slots.push_back(coordinator.submit(req));
                }
                coordinator.run();
                std::uint64_t proposals = 0, cycles = 0;
                for (std::size_t j = 0; j < slots.size(); ++j) {
                    const auto& res = coordinator.result(slots[j]);
                    EAGLE_CHECK(res.ids == off[idx[j]], "multi auto (candidate %d, MULTI=%s) stream %zu ids differ from OFF",
                                int(candidate), env ? env : "unset", j);
                    EAGLE_CHECK(res.multistream_auto_max_window <= 1, "multi auto window %u above the k=1 cap",
                                unsigned(res.multistream_auto_max_window));
                    proposals += res.multistream_auto_proposals; cycles += res.multistream_auto_cycles;
                }
                EAGLE_CHECK(cycles > 0, "multi-stream cycles happened");
                if (candidate && !env) EAGLE_CHECK(proposals > 0, "per-depth drafter drafts at >1 stream");
                else EAGLE_CHECK(proposals == 0, "candidate %d MULTI=%s: auto must stay k=0 at >1 stream", int(candidate),
                                 env ? env : "unset");
                std::printf("coordinator: multi-stream auto candidate=%d STRIX_G4_DRAFT_MULTI=%s: 4 streams == OFF, %llu draft rows in %llu multi-stream cycles\n",
                            int(candidate), env ? env : "unset", (unsigned long long)proposals, (unsigned long long)cycles);
            }
        set_env("STRIX_G4_DRAFT_MULTI", nullptr);
#endif
#if defined(STRIX_G4_DRAFT_CONFIDENCE_API) && defined(STRIX_G4_DRAFT_MULTI_API)
        // Drafter-confidence-gated C1 window through the real coordinator and mtp::Session (chunked draft/extend).
        // ON: ids == OFF, no probes, every cycle's window/drafted depth follows the rule (recomputed here from the
        // recorded probabilities), drafting starts with the first chunk and extends one depth at a time.
        // Unset = DEFAULT ON per drafter (scripted weights: no config key, unknown hash -> builtin tau 0.6, max_k 7).
        // OFF ("0"/"off"): the confident hook is never called, no confidence counters/receipt fields; ids == OFF.
        set_env("STRIX_G4_MTP_CONF_LONG", "0"); // the long-context rule is checked separately below
        for (const char* tau : {"0.5", static_cast<const char*>(nullptr), "off", "0"}) {
            set_env("STRIX_G4_MTP_CONF", tau);
            std::vector<ConfidentRecord> records;
            std::vector<std::pair<bool, std::uint32_t>> calls;
            std::array<std::uint64_t, 8> windows{};
            std::uint64_t probes = 0, cycles = 0, drafted = 0, unverified = 0;
            bool receipt_field = false, receipt_counters = false;
            std::string source;
            for (std::size_t i = 0; i < prompts.size(); ++i) {
                Backend b6;
                en::DraftFactory f = [&b6, &records, &calls]() -> std::unique_ptr<en::DraftSession> {
                    return std::make_unique<Recording>(
                        std::make_unique<mt::Session>(std::make_unique<ScriptedMtp>(b6.target, b6.last->s, &calls),
                                                      std::pair<std::uint32_t, std::uint32_t>{22, 23}), &records);
                };
                en::Coordinator coordinator(b6, f);
                en::Request req;
                req.prompt = prompts[i];
                req.max_tokens = 150;
                req.dflash = en::DraftMode::automatic;
                const auto slot = coordinator.submit(req);
                coordinator.run();
                const auto& res = coordinator.result(slot);
                EAGLE_CHECK(res.ids == off[i], "confidence %s prompt %zu ids differ from OFF", tau ? tau : "unset", i);
                for (std::size_t k = 0; k < 8; ++k) windows[k] += res.windows[k];
                probes += res.probes; cycles += res.confidence_cycles; drafted += res.confidence_drafted;
                unverified += res.confidence_unverified;
                const auto receipt = coordinator.receipt(slot);
                receipt_field = receipt_field || receipt.find("mtp_confidence") || receipt.find("mtp_confidence_resolved");
                if (const auto* resolved = receipt.find("mtp_confidence_resolved")) {
                    source = resolved->find("source")->as_string();
                    EAGLE_CHECK(resolved->find("tau")->as_double() == (tau ? 0.5 : 0.6), "resolved tau");
                }
                receipt_counters = receipt_counters || receipt.find("acceptance")->find("confidence_cycles");
            }
            const bool on = !tau || std::string(tau) == "0.5";
            const double rule_tau = tau ? 0.5 : 0.6;
            if (!on) {
                EAGLE_CHECK(records.empty() && cycles == 0 && drafted == 0 && !receipt_field && !receipt_counters,
                            "confidence %s: disabled path must be untouched", tau ? tau : "unset");
                std::printf("coordinator: STRIX_G4_MTP_CONF=%s: confident hook never called, no receipt fields, ids == OFF (%llu probes)\n",
                            tau ? tau : "unset", (unsigned long long)probes);
                continue;
            }
            EAGLE_CHECK(probes == 0, "confidence window: no calibration probes (%llu)", (unsigned long long)probes);
            EAGLE_CHECK(receipt_field && receipt_counters, "confidence receipt fields");
            EAGLE_CHECK(source == (tau ? "env" : "builtin"), "resolved tau source %s", source.c_str());
            EAGLE_CHECK(cycles == records.size() && cycles > 0, "confidence cycles %llu records %zu", (unsigned long long)cycles,
                        records.size());
            std::array<std::uint64_t, 8> expected_windows{};
            std::uint64_t expected_drafted = 0, expected_unverified = 0, extends = 0;
            std::size_t call = 0;
            for (const auto& rec : records) {
                // Independent restatement of the rule (tau 0.5 env / 0.6 builtin, K 7, first 2, step 1, min1 0).
                EAGLE_CHECK(rec.kmax >= 1 && rec.kmax <= 7, "kmax %u outside 1..STRIX_G4_MTP_CONF_K default 7", rec.kmax);
                const auto kmax = rec.kmax;
                std::uint32_t d = 0, wnd = 0;
                if (rec.drafted) {
                    const auto cum = [&](std::uint32_t n) { double p = 1; for (std::uint32_t j = 0; j < n; ++j) p *= double(rec.prob[j]); return p; };
                    d = std::min(2U, kmax);
                    while (d < kmax && cum(d) >= rule_tau) ++d;
                    while (wnd < d && cum(wnd + 1) >= rule_tau) ++wnd;
                    wnd = std::max(wnd, 1U);
                    // Drafter calls: draft(first chunk), then extend(+1) per extension.
                    EAGLE_CHECK(call < calls.size() && !calls[call].first && calls[call].second == std::min(2U, kmax), "first chunk");
                    ++call;
                    for (std::uint32_t e = std::min(2U, kmax) + 1; e <= d; ++e, ++call, ++extends)
                        EAGLE_CHECK(call < calls.size() && calls[call].first && calls[call].second == e, "extend to %u", e);
                }
                EAGLE_CHECK(rec.drafted == d && rec.window == wnd, "rule: drafted %u (want %u) window %u (want %u)", rec.drafted, d, rec.window, wnd);
                ++expected_windows[rec.window];
                expected_drafted += rec.drafted; expected_unverified += rec.drafted - rec.window;
            }
            EAGLE_CHECK(call == calls.size(), "no other drafter calls (%zu of %zu)", call, calls.size());
            for (std::size_t k = 1; k < 8; ++k)
                EAGLE_CHECK(windows[k] == expected_windows[k], "window %zu histogram %llu want %llu", k, (unsigned long long)windows[k],
                            (unsigned long long)expected_windows[k]);
            EAGLE_CHECK(drafted == expected_drafted && unverified == expected_unverified, "drafted/unverified counters");
            std::size_t distinct = 0;
            for (std::size_t k = 1; k < 8; ++k) distinct += expected_windows[k] ? 1 : 0;
            EAGLE_CHECK(distinct >= 3 && expected_windows[3] + expected_windows[4] + expected_windows[5] > 0 && extends > 0,
                        "scripted probabilities exercise several windows");
            std::printf("coordinator: STRIX_G4_MTP_CONF=%s: ids == OFF, 0 probes, %llu cycles, windows k1..k5 %llu/%llu/%llu/%llu/%llu, %llu drafted (%llu unverified), %llu extends\n",
                        tau ? tau : "unset (builtin 0.6)", (unsigned long long)cycles, (unsigned long long)windows[1], (unsigned long long)windows[2],
                        (unsigned long long)windows[3], (unsigned long long)windows[4], (unsigned long long)windows[5],
                        (unsigned long long)drafted, (unsigned long long)unverified, (unsigned long long)extends);
        }
        // Cost-aware long-context rule (STRIX_G4_MTP_CONF_LONG): with a threshold of ~0 the rule starts once the
        // measured ratio exists (four rule-chosen cycles at decay 0.95) and stays on: first chunk 1, extend while
        // P >= 0.6, verify P >= 0.15 (>= 1); ids == OFF; the receipt counts the long cycles. Pass 2 reports every third
        // proposal as not chosen by the rule (prompt lookup): those neither feed the ratio nor count as long cycles.
        for (const std::uint32_t lookup_every : {0U, 3U}) {
            set_env("STRIX_G4_MTP_CONF", nullptr);
            set_env("STRIX_G4_MTP_CONF_LONG", "0.000000001");
            std::vector<ConfidentRecord> records;
            std::uint64_t long_cycles = 0;
            for (std::size_t i = 0; i < prompts.size(); ++i) {
                Backend b9;
                const auto first_record = records.size();
                en::DraftFactory f = [&b9, &records, lookup_every]() -> std::unique_ptr<en::DraftSession> {
                    auto recording = std::make_unique<Recording>(
                        std::make_unique<mt::Session>(std::make_unique<ScriptedMtp>(b9.target, b9.last->s, nullptr),
                                                      std::pair<std::uint32_t, std::uint32_t>{22, 23}), &records);
                    recording->lookup_every = lookup_every;
                    return recording;
                };
                en::Coordinator coordinator(b9, f);
                en::Request req;
                req.prompt = prompts[i]; req.max_tokens = 150; req.dflash = en::DraftMode::automatic;
                const auto slot = coordinator.submit(req);
                coordinator.run();
                const auto& res = coordinator.result(slot);
                EAGLE_CHECK(res.ids == off[i], "long-context rule prompt %zu ids differ from OFF", i);
                long_cycles += res.confidence_long_cycles;
                const auto receipt = coordinator.receipt(slot);
                const auto* acc = receipt.find("acceptance");
                EAGLE_CHECK(acc && acc->find("confidence_long_cycles") && acc->find("confidence_draft_ratio") &&
                            receipt.find("mtp_confidence_resolved")->find("long_ratio"), "long-context receipt fields");
                // The ratio needs four cycles with a measurable draft time; the scripted host drafter can take less
                // than one timer tick, so a request may finish without one (then it ran no long cycle).
                EAGLE_CHECK(std::uint64_t(acc->find("confidence_long_cycles")->as_int()) == res.confidence_long_cycles &&
                            acc->find("confidence_draft_ratio")->as_double() == res.confidence_draft_ratio,
                            "prompt %zu receipt long cycles %lld / ratio %.17g, result %llu / %.17g", i,
                            (long long)acc->find("confidence_long_cycles")->as_int(),
                            acc->find("confidence_draft_ratio")->as_double(),
                            (unsigned long long)res.confidence_long_cycles, res.confidence_draft_ratio);
                EAGLE_CHECK(res.confidence_long_cycles == 0 || res.confidence_draft_ratio > 0,
                            "prompt %zu: %llu long cycles without a measured ratio", i,
                            (unsigned long long)res.confidence_long_cycles);
                // Per request: rule-chosen cycles 1..4 run the shipped rule (no ratio yet); once the long rule starts it
                // stays on (threshold ~0); not-rule cycles never count and never advance the warm-up.
                std::uint64_t rule_cycles = 0, request_long = 0;
                bool started = false;
                for (std::size_t record_index = first_record; record_index < records.size(); ++record_index) {
                    const auto& rec = records[record_index];
                    const bool is_long = rec.first == 1;
                    if (rec.rule) {
                        EAGLE_CHECK(!is_long || rule_cycles >= 4, "long rule before four measured cycles (prompt %zu)", i);
                        EAGLE_CHECK(!started || is_long, "long rule left at threshold ~0 (prompt %zu)", i);
                        request_long += is_long ? 1 : 0;
                        ++rule_cycles;
                    }
                    started = started || is_long;
                }
                EAGLE_CHECK(request_long == res.confidence_long_cycles, "prompt %zu long cycles %llu, recorded %llu", i,
                            (unsigned long long)res.confidence_long_cycles, (unsigned long long)request_long);
            }
            std::uint64_t long_records = 0;
            for (const auto& rec : records) {
                const bool is_long = rec.first == 1;
                long_records += is_long && rec.rule ? 1 : 0;
                const double ext = is_long ? 0.6 : 0.6, ver = is_long ? 0.15 : 0.6; // builtin tau 0.6 (scripted drafter)
                EAGLE_CHECK(is_long ? (rec.tau == 0.6 && rec.verify_tau == 0.15) : (rec.first == 2 && rec.verify_tau == 0),
                            "rule parameters");
                if (!rec.drafted) continue;
                const auto cum = [&](std::uint32_t n) { double p = 1; for (std::uint32_t j = 0; j < n; ++j) p *= double(rec.prob[j]); return p; };
                std::uint32_t d = std::min(rec.first, rec.kmax), wnd = 0;
                while (d < rec.kmax && cum(d) >= ext) ++d;
                while (wnd < d && cum(wnd + 1) >= ver) ++wnd;
                wnd = std::max(wnd, 1U);
                EAGLE_CHECK(rec.drafted == d && rec.window == wnd, "%s rule: drafted %u (want %u) window %u (want %u)",
                            is_long ? "long" : "shipped", rec.drafted, d, rec.window, wnd);
            }
            EAGLE_CHECK(long_records > 0 && long_records == long_cycles, "long cycles %llu records %llu",
                        (unsigned long long)long_cycles, (unsigned long long)long_records);
            std::printf("coordinator: STRIX_G4_MTP_CONF_LONG~0%s: ids == OFF, %llu of %zu cycles on the long-context rule (first 1, extend 0.6, verify 0.15)\n",
                        lookup_every ? " (every 3rd proposal not rule-chosen)" : "", (unsigned long long)long_cycles, records.size());
            set_env("STRIX_G4_MTP_CONF_LONG", nullptr);
            set_env("STRIX_G4_MTP_CONF", "0");
        }
        {   // Enabled, but DraftMode::on and non-capable (eagle) sessions keep their paths.
            set_env("STRIX_G4_MTP_CONF", "0.5");
            Backend b7;
            std::vector<ConfidentRecord> records;
            en::DraftFactory f = [&b7, &records]() -> std::unique_ptr<en::DraftSession> {
                return std::make_unique<Recording>(
                    std::make_unique<mt::Session>(std::make_unique<ScriptedMtp>(b7.target, b7.last->s, nullptr),
                                                  std::pair<std::uint32_t, std::uint32_t>{22, 23}), &records);
            };
            const auto forced = decode(b7, f, {prompts[0]}, 4, en::DraftMode::on, nullptr);
            EAGLE_CHECK(forced[0] == off[0] && records.empty(), "DraftMode::on ignores the confidence window");
            Backend b8;
            const auto eagle = decode(b8, [&b8]() -> std::unique_ptr<en::DraftSession> {
                return std::make_unique<eg::Session>(std::make_unique<Oracle>(b8.target, b8.last->s, small_config()));
            }, {prompts[0]}, 7, en::DraftMode::automatic, nullptr);
            EAGLE_CHECK(eagle[0] == off[0], "non-capable session under STRIX_G4_MTP_CONF: ids == OFF");
            set_env("STRIX_G4_MTP_CONF", "0");
        }
#endif
        std::printf("coordinator: eagle sessions == OFF for k in {1,4,7}, prompt lengths 1/2/8/9/16/17/random, 1 and 3 streams; oracle accepted %llu/%llu, host drafter (1 and 3 streams) %llu/%llu\n",
                    (unsigned long long)oracle_stats.accepted, (unsigned long long)oracle_stats.attempted,
                    (unsigned long long)real_stats.accepted, (unsigned long long)real_stats.attempted);
        EAGLE_CHECK(oracle_stats.accepted > 0, "oracle acceptance through the coordinator");
        EAGLE_CHECK(real_stats.attempted > 0, "drafting happened through the coordinator");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf(eagle_test::failures ? "FAILED %d checks\n" : "ALL OK\n", eagle_test::failures);
    return eagle_test::failures ? 1 : 0;
}
#endif
