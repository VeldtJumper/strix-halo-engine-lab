// Shared drafting layer (strix/spec/drafting.hpp): index source/continuation vs brute force (ring wrap included), the
// copy cursor on a copy-with-one-edit stream, the arbiter's choices on synthetic acceptance/costs, and the stats line.
// A simulated greedy target checks that whatever the arbiter picks, the emitted ids equal the serial ids.
#include "strix/spec/drafting.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace sp = strix::spec;
namespace lk = strix::lookup;

namespace {
#define SP_CHECK(c, ...) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); std::exit(1); } } while (0)

struct Rng {
    std::uint64_t s;
    std::uint32_t next(std::uint32_t n) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return std::uint32_t((s >> 33) % n); }
};

// Every reported match is real: the suffix of length `length` ends at source_end, continuation = the stream after it.
void index_tests() {
    Rng r{11};
    std::uint64_t hits = 0;
    for (const std::uint32_t vocab : {3U, 7U, 40U}) {
        for (const std::uint32_t cap : {64U, 4096U}) {
            auto index = lk::make_ngram_index({cap, 64});
            std::vector<std::uint32_t> s;
            for (int i = 0; i < 3000; ++i) {
                const auto t = r.next(vocab);
                s.push_back(t);
                index->append(t);
                SP_CHECK(index->at(s.size() - 1) == t, "at(last)");
                SP_CHECK(index->oldest() == (s.size() > cap ? s.size() - cap : 0), "oldest");
                const auto m = index->longest({}, 2, 7);
                if (m.length == 0) continue;
                ++hits;
                const std::uint64_t cur = s.size() - 1, e = m.source_end;
                SP_CHECK(e < cur && cur - e == m.distance, "source_end/distance");
                SP_CHECK(e + 1 >= m.length + index->oldest(), "match inside the ring");
                for (std::uint32_t j = 0; j < m.length; ++j) SP_CHECK(s[e - j] == s[cur - j], "suffix at source_end");
                std::vector<std::uint32_t> out(40);
                const auto n = index->continuation(e, {}, out);
                SP_CHECK(n == std::min<std::size_t>(40, cur - e), "continuation length %zu", n);
                for (std::size_t j = 0; j < n; ++j) SP_CHECK(out[j] == s[e + 1 + j], "continuation token");
                for (std::uint32_t j = 0; j < m.available; ++j) SP_CHECK(m.next[j] == out[j], "next == continuation");
            }
            // extra: continuation crosses from the stream into the hypothetical tokens
            const std::vector<std::uint32_t> extra{1, 2, 3};
            std::vector<std::uint32_t> out(8);
            const auto n = index->continuation(index->tokens() - 2, extra, out);
            SP_CHECK(n == 4 && out[0] == s.back() && out[1] == 1 && out[3] == 3, "continuation into extra");
            SP_CHECK(index->continuation(0, {}, out) == (cap >= s.size() ? 8U : 0U), "positions older than the ring");
        }
    }
    SP_CHECK(hits > 1000, "too few matches %llu", static_cast<unsigned long long>(hits));
}

// Speculative loop against a known greedy truth: the arbiter decides, the "target" accepts the prefix equal to truth
// and emits the correction. Returns passes; checks emitted == truth (exactness of the protocol, any decision).
struct LoopResult { std::uint64_t passes = 0, lookup_passes = 0, resumed = 0; std::uint32_t max_k = 0; };
LoopResult run_loop(const std::vector<std::uint32_t>& prompt, const std::vector<std::uint32_t>& truth,
                    sp::ArbiterConfig config, double row_ms, sp::SpecStats* stats = nullptr) {
    sp::ContextProposer proposer;
    proposer.commit(prompt);
    sp::Arbiter arbiter(config);
    for (const auto w : arbiter.config().windows) arbiter.seed_cost(w, 19.4 + row_ms * double(w - 1));
    std::vector<std::uint32_t> emitted;
    LoopResult res;
    while (emitted.size() < truth.size()) {
        const auto proposal = proposer.propose({}, sp::max_proposal);
        sp::StepInput in;
        in.lookup = &proposal;
        in.remaining = std::uint32_t(truth.size() - emitted.size());
        in.context = prompt.size() + emitted.size();
        const auto d = arbiter.decide(in);
        SP_CHECK(d.k + 1 <= d.rows && d.k < in.remaining, "decision exceeds window/budget");
        SP_CHECK(d.source == sp::Source::serial || d.source == sp::Source::lookup, "no model offered");
        std::uint32_t accepted = 0;
        const auto base = emitted.size();
        if (d.source == sp::Source::lookup)
            while (accepted < d.k && proposal.tokens[accepted] == truth[base + accepted]) ++accepted;
        std::vector<std::uint32_t> now(truth.begin() + std::ptrdiff_t(base), truth.begin() + std::ptrdiff_t(base + accepted + 1));
        emitted.insert(emitted.end(), now.begin(), now.end());
        proposer.commit(now);
        sp::Outcome o{d.source, proposal.bucket(), d.k, accepted, d.rows, 19.4 + row_ms * double(d.rows - 1), 0.0, in.context};
        arbiter.observe(o, now);
        if (stats) stats->record(o, std::uint32_t(now.size()));
        ++res.passes;
        res.lookup_passes += d.source == sp::Source::lookup;
        res.resumed += proposal.resumed;
        res.max_k = std::max(res.max_k, d.k);
    }
    SP_CHECK(emitted == truth, "emitted ids differ from the serial truth");
    return res;
}

void copy_tests() {
    Rng r{5};
    std::vector<std::uint32_t> file(3000);
    for (auto& t : file) t = r.next(5000);
    std::vector<std::uint32_t> prompt{7, 8, 9};
    prompt.insert(prompt.end(), file.begin(), file.end());
    prompt.push_back(4999);
    auto truth = file;  // the answer: the whole file with one edit (3 tokens replaced)
    truth[1500] = 1; truth[1501] = 2; truth[1502] = 3;
    sp::ArbiterConfig config;
    config.windows = {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64};
    sp::SpecStats stats;
    const auto res = run_loop(prompt, truth, config, 2.7, &stats);
    std::printf("copy 3000 tokens: %llu passes (%.1f tokens/pass), lookup %llu, resumed %llu, max k %u\n",
                static_cast<unsigned long long>(res.passes), 3000.0 / double(res.passes),
                static_cast<unsigned long long>(res.lookup_passes), static_cast<unsigned long long>(res.resumed), res.max_k);
    SP_CHECK(res.passes < 3000 / 20, "copy should commit > 20 tokens/pass, got %llu passes", static_cast<unsigned long long>(res.passes));
    SP_CHECK(res.max_k >= 31, "long windows unused (max k %u)", res.max_k);
    // Cheap rows (0.5 ms/row): the 64-row window wins.
    const auto resc = run_loop(prompt, truth, config, 0.5);
    std::printf("copy, 0.5 ms/row: %llu passes, max k %u\n", static_cast<unsigned long long>(resc.passes), resc.max_k);
    SP_CHECK(resc.max_k == 63 && resc.passes <= res.passes, "cheap rows: max k %u", resc.max_k);
    const auto line = stats.line();
    SP_CHECK(line.rfind("strix-spec-stats {\"passes\":", 0) == 0, "stats line prefix: %s", line.c_str());
    SP_CHECK(line.find("\"emitted\":3000") != std::string::npos, "stats emitted");
    SP_CHECK(line.find("\"64\":{\"passes\":") != std::string::npos, "rows histogram has the 64-row window");

    // Eight-row engines: same stream, capped windows.
    const auto res8 = run_loop(prompt, truth, sp::ArbiterConfig{}, 2.7);
    SP_CHECK(res8.max_k == 7 && res8.passes < 3000 / 6, "8-row copy: max k %u passes %llu", res8.max_k,
             static_cast<unsigned long long>(res8.passes));

    // New text (no repeats in the answer or the prompt): lookup never wins, serial throughout after the priors.
    std::vector<std::uint32_t> fresh(2000);
    for (auto& t : fresh) t = 10000 + r.next(50000);
    const auto resn = run_loop(prompt, fresh, config, 2.7);
    SP_CHECK(resn.lookup_passes < 20, "lookup used on new text: %llu passes", static_cast<unsigned long long>(resn.lookup_passes));

    // Expensive rows (12 ms/row on a 19.4 ms base): windows must stay shorter than with cheap rows.
    const auto rese = run_loop(prompt, truth, config, 12.0);
    SP_CHECK(rese.passes > res.passes, "expensive rows should cost passes (%llu vs %llu)",
             static_cast<unsigned long long>(rese.passes), static_cast<unsigned long long>(res.passes));
}

void arbiter_tests() {
    sp::ArbiterConfig config;
    config.windows = {1, 2, 4, 8};
    sp::Arbiter a(config);
    for (const std::uint32_t w : {1U, 2U, 4U, 8U}) a.seed_cost(w, 20.0 + 3.0 * double(w - 1));
    SP_CHECK(a.verify_ms(3) > a.verify_ms(2) && a.verify_ms(3) < a.verify_ms(4), "interpolation");
    SP_CHECK(a.verify_ms(6) > a.verify_ms(4) && a.verify_ms(6) < a.verify_ms(8), "interpolation 6");
    // Model drafter p = 0.8 at every depth, 0.5 ms/draft token: the arbiter picks the model over serial.
    const std::vector<double> p(7, 0.8), draft{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5};
    a.seed_model_survival(p);
    sp::StepInput in;
    in.model_max_k = 7;
    in.model_draft_ms = draft;
    auto d = a.decide(in);
    SP_CHECK(d.source == sp::Source::model && d.k >= 3 && d.rows >= d.k + 1, "model chosen (k %u)", d.k);
    // Budget: two tokens left -> at most one draft.
    in.remaining = 2;
    d = a.decide(in);
    SP_CHECK(d.k <= 1, "budget");
    in.remaining = 1;
    d = a.decide(in);
    SP_CHECK(d.source == sp::Source::serial && d.k == 0, "last token is serial");
    // Repeated first-row misses: the model's survival collapses and serial takes over.
    in.remaining = ~0U;
    for (int i = 0; i < 40; ++i) a.observe({sp::Source::model, 0, 7, 0, 8, 41.0, 3.5}, {});
    d = a.decide(in);
    SP_CHECK(d.source == sp::Source::serial || d.probe, "retreat after misses");
    // Probe: after probe_every passes without the model it is tried once.
    bool probed = false;
    for (int i = 0; i < 40 && !probed; ++i) {
        d = a.decide(in);
        probed = d.probe && d.source == sp::Source::model;
        a.observe({d.source, 0, d.k, 0, d.rows, 20.0, 0.0}, {});
    }
    SP_CHECK(probed, "model re-probed");
    SP_CHECK(d.k == 7, "probe runs the deepest legal k (%u)", d.k);
    // Per-step legal rows: max_rows = 3 keeps every window <= 3; zero budget / zero rows = nothing to do.
    in.max_rows = 3;
    for (int i = 0; i < 8; ++i) {
        d = a.decide(in);
        SP_CHECK(d.rows >= 1 && d.rows <= 3 && d.k + 1 <= d.rows, "max_rows (rows %u k %u)", d.rows, d.k);
    }
    in.max_rows = ~0U;
    in.remaining = 0;
    SP_CHECK(a.decide(in).rows == 0, "zero budget");
    in.remaining = ~0U;
    // A drafter must come with its measured cost.
    bool threw = false;
    sp::StepInput no_cost;
    no_cost.model_max_k = 3;
    try { (void)a.decide(no_cost); } catch (const std::invalid_argument&) { threw = true; }
    SP_CHECK(threw, "drafter without draft cost");
    // Validation.
    threw = false;
    try { sp::Arbiter bad(sp::ArbiterConfig{{2, 4}}); } catch (const std::invalid_argument&) { threw = true; }
    SP_CHECK(threw, "windows must start at 1");

    // Costs: monotone in rows even when measurements invert; separate per context bin.
    sp::Arbiter c(sp::ArbiterConfig{{1, 2, 4, 8, 16}});
    c.seed_cost(1, 20.0);
    c.seed_cost(4, 30.0);
    c.seed_cost(8, 28.0);  // inverted measurement
    SP_CHECK(c.verify_ms(8) >= c.verify_ms(4) && c.verify_ms(6) >= c.verify_ms(4), "monotone cost");
    SP_CHECK(c.verify_ms(16) >= c.verify_ms(8), "extrapolation stays monotone");
    c.observe({sp::Source::serial, 0, 0, 0, 1, 80.0, 0.0, 100000}, {});  // a 100K-context wall
    SP_CHECK(c.verify_ms(1, 100000) > c.verify_ms(1, 1000), "context bins are separate");

    // Lookup needs evidence first: a proposal is not run before min_lookup_evidence depth-1 observations.
    sp::Arbiter l(sp::ArbiterConfig{{1, 2, 4, 8}});
    for (const std::uint32_t w : {1U, 2U, 4U, 8U}) l.seed_cost(w, 20.0 + 1.0 * double(w - 1));
    sp::Proposal prop;
    prop.tokens = {5, 6, 7, 8};
    prop.match_length = 16;
    sp::StepInput lin;
    lin.lookup = &prop;
    SP_CHECK(l.decide(lin).source == sp::Source::serial, "lookup before evidence");

    // Black-box drafter (picks its own window): scored on its measured rate, no per-depth cost table needed.
    sp::Arbiter bb(sp::ArbiterConfig{});
    bb.seed_cost(1, 18.0);
    sp::StepInput bin;
    bin.model_max_k = 7;
    bin.model_pass_tokens = 2.6;
    bin.model_pass_ms = 24.0;  // 0.108 tok/ms > serial 1/18 = 0.056
    auto bd = bb.decide(bin);
    SP_CHECK(bd.source == sp::Source::model && bd.k == 7, "black-box drafter chosen (k %u)", bd.k);
    bin.model_pass_tokens = 1.1;  // 0.046 tok/ms < serial; after the 3 forced measurements serial wins
    for (int i = 0; i < 3; ++i) bb.observe({sp::Source::model, 0, 7, 1, 8, 20.0, 4.0}, {});
    bd = bb.decide(bin);
    SP_CHECK(bd.source == sp::Source::serial, "weak black-box drafter loses to serial");

    // Black-box drafter with ~50 % first-row acceptance whose rate beats serial: no retreat, it keeps running.
    sp::Arbiter bb2(sp::ArbiterConfig{});
    bb2.seed_cost(1, 18.0);
    sp::StepInput bin2;
    bin2.model_max_k = 7;
    bin2.model_pass_tokens = 1.7;
    bin2.model_pass_ms = 24.0;  // 0.071 tok/ms > serial 0.056
    std::uint32_t model_runs = 0;
    for (int i = 0; i < 200; ++i) {
        const auto d2 = bb2.decide(bin2);
        model_runs += d2.source == sp::Source::model;
        const std::uint32_t acc = (i % 4 < 2) ? 0U : 2U;  // two misses in a row every 4 passes
        bb2.observe({d2.source, 0, d2.k, d2.source == sp::Source::model ? acc : 0U, d2.rows, 20.0, 4.0}, {});
    }
    SP_CHECK(model_runs == 200, "black-box drafter barred by the retreat rule (%u of 200 passes)", model_runs);

    // Stats: lookup time quantiles.
    sp::SpecStats st;
    for (int i = 0; i < 99; ++i) st.lookup_time(3.0);
    st.lookup_time(500.0);
    SP_CHECK(st.lookup_us_quantile(0.5) <= 4.0 && st.lookup_us_quantile(0.999) >= 256.0, "lookup quantiles");
    SP_CHECK(st.json().find("\"lookup_us\":{\"count\":100") != std::string::npos, "lookup_us json");
}
} // namespace

int main() {
    index_tests();
    copy_tests();
    arbiter_tests();
    std::printf("PASS strix-spec-drafting-tests\n");
    return 0;
}
