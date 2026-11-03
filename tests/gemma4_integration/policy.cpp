#include "gemma4/engine/auto_policy.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <cstdlib>
#include <string>
#include <vector>

namespace e = strix::gemma4::engine;
namespace df = strix::gemma4::dflash;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)
namespace {
df::Acceptance outcome(unsigned k, bool accepted) {
    df::Acceptance a;
    a.cycles = 1; a.attempted = k; a.emitted = accepted ? k+1 : 1;
    for (unsigned d = 0; d < k && (accepted || d == 0); ++d) {
        ++a.depth_eligible[d]; ++a.eligible;
        if (accepted) { ++a.depth_accepted[d]; ++a.accepted; }
    }
    return a;
}
void simulate(bool accepts, bool expensive) {
    e::AutoPolicy policy;
    CHECK(policy.choose(7,0).k == 0);
    unsigned probes = 0, admitted = 0, plain = 0;
    for (unsigned c = 0; c < 512; ++c) {
        const auto choice = policy.choose(7,c%400);
        CHECK(choice.k <= 7);
        if (choice.k && !choice.probe) {
            CHECK(choice.gain >= 1.05 && choice.expected > 1 && choice.cost_ms > 0);
            CHECK(accepts && !expensive); ++admitted;
        }
        probes += choice.probe ? 1 : 0;
        plain += !choice.k ? 1 : 0;
        const double draft = choice.k ? (expensive ? 500.0 : 11.5) : 0;
        const double verify = choice.k ? 24+4.0*choice.k : 24;
        policy.observe(c%400,choice.k,draft+verify+0.7,draft,verify,0.7,outcome(choice.k,accepts),choice.probe);
    }
    CHECK(probes >= 6 && probes < 80 && plain >= 32);
    CHECK(accepts && !expensive ? admitted > 100 : admitted == 0);
    // Length bins do not borrow faster short-context costs.
    CHECK(policy.choose(7,8192).k == 0);
    CHECK(policy.choose(0,0).k == 0);
}
std::uint64_t lcg = 12345;
df::Acceptance chained(unsigned k, double q) {
    df::Acceptance a;
    a.cycles = 1; a.attempted = k;
    unsigned accepted = 0;
    for (unsigned d = 0; d < k; ++d) {
        ++a.depth_eligible[d]; ++a.eligible;
        lcg = lcg*6364136223846793005ULL+1442695040888963407ULL;
        if (double(lcg >> 11)*0x1p-53 >= q) break;
        ++a.depth_accepted[d]; ++a.accepted; ++accepted;
    }
    a.emitted = accepted+1;
    return a;
}
// Measured MTP C1 EN profile (forced-k sweep): verify ms per k, draft ~0.56 ms per depth, survival ~0.69.
void measured_profile() {
    const double verify[8] = {20.0, 21.6, 23.1, 25.4, 29.6, 31.5, 33.4, 35.3};
    e::AutoPolicy policy;
    std::array<unsigned,8> admitted{};
    for (unsigned c = 0; c < 2000; ++c) {
        const auto choice = policy.choose(7,c%500);
        if (choice.k && !choice.probe) ++admitted[choice.k];
        const double draft = 0.56*choice.k;
        policy.observe(c%500,choice.k,draft+verify[choice.k]+0.2,draft,verify[choice.k],0.2,chained(choice.k,0.69),choice.probe);
    }
    for (unsigned k = 4; k < 8; ++k) CHECK(admitted[k] == 0);
    CHECK(admitted[2]+admitted[3] > 1000);
    const auto bin = policy.receipt().as_array()[0];
    const double k1 = bin.find("draft_fit_k1_ms")->as_double(), k7 = bin.find("draft_fit_k7_ms")->as_double();
    CHECK(k1 > 0.4 && k1 < 0.8 && k7 > 3.5 && k7 < 4.4); // per-depth fit, not a constant
}
void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name,value ? value : "");
#else
    if (value) setenv(name,value,1); else unsetenv(name);
#endif
}
// multi-stream batch model. Step = plain batch t0 + per extra row + every stream's own draft time (+ noise).
// Defaults reproduce the measured one-node C4 EN point: OFF 30.5 ms/step (131.0 tok/s), forced k=1 ~47 ms/step
// (144.0 tok/s at 0.695 acceptance), MTP draft 0.56 ms/depth.
struct BatchModel {
    std::size_t streams = 4;
    bool candidates = true; // false: DFlash/no per-depth drafter -> StreamCandidate::policy null
    double t0 = 30.5, per_row = 3.59, fixed = 0, per_depth = 0.56, q = 0.695;
};
struct BatchRun { unsigned steps = 0, admitted = 0, probes = 0, plain = 0, max_k = 0; double gain = 0; };
BatchRun run_batch(e::MultiStreamPolicy& multi, const BatchModel& m, unsigned steps, std::vector<e::AutoPolicy>* keep = nullptr) {
    std::vector<e::AutoPolicy> local(m.streams);
    auto& policies = keep ? *keep : local;
    if (policies.size() != m.streams) policies.assign(m.streams,e::AutoPolicy{});
    BatchRun r;
    for (unsigned step = 0; step < steps; ++step) {
        const std::uint64_t position = 16+step%400;
        std::vector<e::StreamCandidate> c(m.streams);
        for (std::size_t i = 0; i < m.streams; ++i) c[i] = {m.candidates ? &policies[i] : nullptr,position,7};
        const auto plan = multi.choose(c);
        double ms = m.t0;
        std::uint32_t extra = 0;
        for (std::size_t i = 0; i < m.streams; ++i) {
            const auto k = plan.k[i];
            CHECK(k <= multi.config().max_k);
            r.max_k = std::max(r.max_k,unsigned(k));
            if (!m.candidates) CHECK(k == 0);
            if (!k) continue;
            const double draft = m.fixed+m.per_depth*k;
            policies[i].observe_multi(position,k,draft,chained(k,m.q));
            ms += draft+m.per_row*k; extra += k;
        }
        lcg = lcg*6364136223846793005ULL+1442695040888963407ULL;
        ms += 0.3*(double(lcg >> 11)*0x1p-53-0.5);
        CHECK(extra == plan.extra);
        if (extra && !plan.probe) { ++r.admitted; CHECK(plan.gain >= multi.config().min_gain); r.gain = plan.gain; }
        r.probes += plan.probe ? 1 : 0;
        r.plain += extra ? 0 : 1;
        multi.observe(m.streams,extra,ms,plan.probe);
        ++r.steps;
    }
    return r;
}
void multi_stream() {
    set_env("STRIX_G4_DRAFT_MULTI",nullptr); set_env("STRIX_G4_DRAFT_MULTI_K",nullptr);
    set_env("STRIX_G4_DRAFT_MULTI_COST",nullptr); set_env("STRIX_G4_DRAFT_MULTI_GAIN",nullptr);
    const auto defaults = e::MultiStreamConfig::from_environment();
    CHECK(defaults.enabled && defaults.max_k == 1 && defaults.max_draft_share == 0.25 && defaults.min_gain == 1.05);
    {   // Measured C4 point: admitted, and the cap holds (never k>1 while >1 stream) although limit is 7.
        e::MultiStreamPolicy multi(defaults);
        const auto r = run_batch(multi,{},600);
        std::cout << "multi C4 measured point: admitted " << r.admitted << " probes " << r.probes << " plain " << r.plain << " gain " << r.gain << '\n';
        CHECK(r.max_k == 1 && r.admitted > 300 && r.probes >= 6 && r.probes < 90 && r.plain >= 30);
        CHECK(r.gain > 1.05 && r.gain < 1.15); // ~x1.09 modelled vs x1.10 measured
        // Cap holds even when deeper windows would be nearly free and always accepted.
        e::MultiStreamPolicy greedy(defaults);
        BatchModel cheap; cheap.per_row = 0.5; cheap.per_depth = 0.05; cheap.q = 0.99;
        const auto g = run_batch(greedy,cheap,400);
        CHECK(g.max_k == 1 && g.admitted > 300);
        for (std::size_t c = 2; c <= 7; ++c) { // every C that has spare rows: still capped
            e::MultiStreamPolicy each(defaults); cheap.streams = c;
            CHECK(run_batch(each,cheap,200).max_k == 1);
        }
        // C8 has no spare row (8-row cap): plain, no probes.
        e::MultiStreamPolicy full(defaults); cheap.streams = 8;
        const auto f = run_batch(full,cheap,200);
        CHECK(f.max_k == 0 && f.probes == 0 && f.admitted == 0);
        // Tunable: the same planner with cap 2 may use k=2 (not the default).
        auto two = defaults; two.max_k = 2;
        e::MultiStreamPolicy wider(two); cheap.streams = 4;
        CHECK(run_batch(wider,cheap,400).max_k == 2);
    }
    {   // Measured costs decide: a k=1 that loses at C>1 is declined after bounded probes.
        e::MultiStreamPolicy multi(defaults);
        BatchModel slow; slow.per_row = 6.0;
        const auto r = run_batch(multi,slow,600);
        CHECK(r.admitted == 0 && r.probes >= 6 && r.probes < 90);
    }
    {   // Per-depth cost gate: a drafter whose fitted k=1 cost is above share x plain/C never drafts, even where
        // its acceptance would make the batch win; at most the bootstrap probe that measured it.
        e::MultiStreamPolicy multi(defaults);
        BatchModel heavy; heavy.fixed = 4.0; heavy.per_row = 0.2; heavy.q = 0.9;
        const auto r = run_batch(multi,heavy,600);
        CHECK(r.admitted == 0 && r.probes <= 1);
    }
    {   // DFlash (no multi-stream candidate): unchanged at C>1 - never planned, never probed.
        e::MultiStreamPolicy multi(defaults);
        BatchModel dflash; dflash.candidates = false;
        const auto r = run_batch(multi,dflash,600);
        CHECK(r.max_k == 0 && r.probes == 0 && r.admitted == 0 && r.plain == 600);
    }
    {   // Single stream: the batch planner never acts on C1; C1 timing evidence is untouched by C>1 steps.
        e::MultiStreamPolicy multi(defaults);
        e::AutoPolicy p;
        std::vector<e::StreamCandidate> one{{&p,0,7}};
        for (unsigned i = 0; i < 64; ++i) { CHECK(!multi.choose(one).extra); }
        std::vector<e::AutoPolicy> kept;
        run_batch(multi,{},200,&kept);
        CHECK(kept[0].receipt().as_array().empty()); // no C1 cycles: no cycle/verify/plain evidence was written
        CHECK(kept[0].draft_cost(16,1) > 0.4 && kept[0].draft_cost(16,1) < 0.8);
        CHECK(kept[0].choose(7,16).k == 0); // no C1 plain rows yet -> the C1 policy is still cold
    }
    {   // STRIX_G4_DRAFT_MULTI=0 restores the previous behaviour: auto off whenever >1 stream is live.
        set_env("STRIX_G4_DRAFT_MULTI","0");
        e::MultiStreamPolicy off;
        CHECK(!off.config().enabled);
        const auto r = run_batch(off,{},400);
        CHECK(r.max_k == 0 && r.probes == 0 && r.admitted == 0 && r.plain == 400);
        CHECK(off.receipt().as_array().empty());
        set_env("STRIX_G4_DRAFT_MULTI","1");
        CHECK(e::MultiStreamConfig::from_environment().enabled);
        set_env("STRIX_G4_DRAFT_MULTI_K","2"); set_env("STRIX_G4_DRAFT_MULTI_COST","0.1");
        const auto tuned = e::MultiStreamConfig::from_environment();
        CHECK(tuned.max_k == 2 && tuned.max_draft_share == 0.1);
        bool rejected = false;
        set_env("STRIX_G4_DRAFT_MULTI_K","1.5");
        try { (void)e::MultiStreamConfig::from_environment(); } catch (const std::invalid_argument&) { rejected = true; }
        CHECK(rejected);
        set_env("STRIX_G4_DRAFT_MULTI",nullptr); set_env("STRIX_G4_DRAFT_MULTI_K",nullptr); set_env("STRIX_G4_DRAFT_MULTI_COST",nullptr);
    }
}
// Drafter-confidence window config (STRIX_G4_MTP_CONF*): default ON per drafter, OFF switch, bounds, resolution
// precedence (env > drafter config > drafter builtin > builtin_tau), and the chunked window rule.
void confidence() {
    const char* names[] = {"STRIX_G4_MTP_CONF","STRIX_G4_MTP_CONF_K","STRIX_G4_MTP_CONF_FIRST","STRIX_G4_MTP_CONF_STEP","STRIX_G4_MTP_CONF_MIN1","STRIX_G4_MTP_CONF_LONG"};
    for (const auto* n : names) set_env(n,nullptr);
    const auto rejected = [] { try { (void)e::ConfidenceConfig::from_environment(); } catch (const std::invalid_argument&) { return true; } return false; };
    const auto dflt = e::ConfidenceConfig::from_environment();
    CHECK(dflt.enabled() && dflt.per_drafter && dflt.tau == e::ConfidenceConfig::builtin_tau && dflt.max_k == 7 &&
          dflt.first == 2 && dflt.step == 1 && dflt.min1 == 0 && dflt.source == e::ConfidenceConfig::Source::builtin);
    set_env("STRIX_G4_MTP_CONF","auto"); CHECK(e::ConfidenceConfig::from_environment().per_drafter);
    set_env("STRIX_G4_MTP_CONF",nullptr);
    {   // Resolution precedence.
        const auto a = dflt.resolved(0.3,5,1,0.6);            // config keys win over the drafter builtin
        CHECK(a.tau == 0.3 && a.max_k == 5 && a.first == 1 && a.source == e::ConfidenceConfig::Source::drafter_config && !a.per_drafter);
        const auto b = dflt.resolved(0,0,0,0.3);              // no keys: the drafter's builtin tau
        CHECK(b.tau == 0.3 && b.max_k == 7 && b.first == 2 && b.source == e::ConfidenceConfig::Source::builtin);
        const auto c0 = dflt.resolved(0,0,0,0);               // unknown drafter: builtin_tau
        CHECK(c0.tau == e::ConfidenceConfig::builtin_tau);
        set_env("STRIX_G4_MTP_CONF_K","4");
        const auto k = e::ConfidenceConfig::from_environment().resolved(0.3,6,0,0); // env K wins over config max_k
        CHECK(k.max_k == 4 && k.tau == 0.3);
        set_env("STRIX_G4_MTP_CONF_K",nullptr);
        set_env("STRIX_G4_MTP_CONF","0.45");
        const auto env = e::ConfidenceConfig::from_environment();
        const auto r = env.resolved(0.3,5,1,0.6);              // explicit env tau ignores the drafter entirely
        CHECK(r.tau == 0.45 && r.max_k == 7 && r.first == 2 && r.source == e::ConfidenceConfig::Source::env);
        CHECK(std::string(r.json().find("source")->as_string()) == "env");
        set_env("STRIX_G4_MTP_CONF",nullptr);
    }
    for (const char* v : {"0","off","0.0"}) { set_env("STRIX_G4_MTP_CONF",v); CHECK(!e::ConfidenceConfig::from_environment().enabled()); }
    set_env("STRIX_G4_MTP_CONF_K","9"); // ignored while disabled
    set_env("STRIX_G4_MTP_CONF","off"); CHECK(!e::ConfidenceConfig::from_environment().enabled());
    set_env("STRIX_G4_MTP_CONF","0.4"); CHECK(rejected());
    set_env("STRIX_G4_MTP_CONF_K","0"); CHECK(rejected());
    set_env("STRIX_G4_MTP_CONF_K","2.5"); CHECK(rejected());
    set_env("STRIX_G4_MTP_CONF_K","7"); set_env("STRIX_G4_MTP_CONF_FIRST","1"); set_env("STRIX_G4_MTP_CONF_STEP","3");
    set_env("STRIX_G4_MTP_CONF_MIN1","0.2");
    const auto tuned = e::ConfidenceConfig::from_environment();
    CHECK(tuned.enabled() && tuned.tau == 0.4 && tuned.max_k == 7 && tuned.first == 1 && tuned.step == 3 && tuned.min1 == 0.2);
    CHECK(tuned.json().find("enabled")->as_bool() && tuned.json().find("max_k")->as_int() == 7);
    set_env("STRIX_G4_MTP_CONF_FIRST","8"); CHECK(rejected());
    set_env("STRIX_G4_MTP_CONF_FIRST",nullptr); set_env("STRIX_G4_MTP_CONF_STEP","0"); CHECK(rejected());
    set_env("STRIX_G4_MTP_CONF_STEP",nullptr); set_env("STRIX_G4_MTP_CONF_MIN1","1.5"); CHECK(rejected());
    set_env("STRIX_G4_MTP_CONF_MIN1",nullptr);
    set_env("STRIX_G4_MTP_CONF","1.01"); CHECK(rejected());
    set_env("STRIX_G4_MTP_CONF","-0.1"); CHECK(rejected());
    set_env("STRIX_G4_MTP_CONF","x"); CHECK(rejected());
    set_env("STRIX_G4_MTP_CONF","1"); CHECK(e::ConfidenceConfig::from_environment().tau == 1);
    for (const auto* n : names) set_env(n,nullptr);
    {   // Long-context rule switch (STRIX_G4_MTP_CONF_LONG): unset / empty / auto = 0.04; 0 / 0.0 / off = OFF; other
        // values must be numbers in [0,1]; ignored while the window itself is off; kept through per-drafter resolution.
        const auto ratio = [] { return e::ConfidenceConfig::from_environment().long_ratio; };
        CHECK(ratio() == 0.04);
        for (const char* v : {"","auto"}) { set_env("STRIX_G4_MTP_CONF_LONG",v); CHECK(ratio() == 0.04); }
        for (const char* v : {"0","0.0","off"}) { set_env("STRIX_G4_MTP_CONF_LONG",v); CHECK(ratio() == 0); }
        set_env("STRIX_G4_MTP_CONF_LONG","0.1"); CHECK(ratio() == 0.1);
        CHECK(std::fabs(e::ConfidenceConfig::from_environment().json().find("long_ratio")->as_double() - 0.1) < 1e-12);
        for (const char* v : {"1.5","-0.1","x"}) { set_env("STRIX_G4_MTP_CONF_LONG",v); CHECK(rejected()); }
        set_env("STRIX_G4_MTP_CONF","0"); CHECK(!e::ConfidenceConfig::from_environment().enabled()); // not parsed
        set_env("STRIX_G4_MTP_CONF",nullptr);
        for (const char* v : {"0","0.2"}) {
            set_env("STRIX_G4_MTP_CONF_LONG",v);
            const auto env = e::ConfidenceConfig::from_environment();
            CHECK(env.per_drafter && env.resolved(0.3,5,1,0.6).long_ratio == env.long_ratio &&
                  env.resolved(0,0,0,0.4).long_ratio == env.long_ratio);
        }
        set_env("STRIX_G4_MTP_CONF_LONG",nullptr);
        // long_context(): one depth first, extend while P >= 0.6, verify P >= 0.15; the base config is not changed.
        e::ConfidenceConfig b; b.tau = 0.4;
        const auto l = b.long_context();
        CHECK(l.first == 1 && l.step == 1 && l.tau == 0.6 && l.verify_tau == 0.15 && b.verify_tau == 0 && b.first == 2);
        const std::vector<float> q{0.9F,0.8F,0.5F,0.5F}; // P = 0.9, 0.72, 0.36, 0.18
        const auto qs = [&](std::uint32_t n) { return std::span<const float>(q.data(),n); };
        CHECK(l.next_depth({},7) == 1 && l.next_depth(qs(1),7) == 2 && l.next_depth(qs(2),7) == 3 && l.next_depth(qs(3),7) == 3);
        CHECK(l.window(qs(4)) == 4 && b.window(qs(4)) == 2); // verify at 0.15 vs tau 0.4
        const std::vector<float> lo{0.1F};
        CHECK(l.window(lo) == 1); // at least one verified depth
    }
    {   // Measured cost ratio: none before four observations (n >= 3 at decay 0.95), crossing up and down, k == 0 rows.
        e::DraftCostRatio m;
        e::ConfidenceConfig c; // long_ratio 0.04
        const double r3 = 10.0 * (1 + 5.74 / 17.89); // a 4-row verify whose one-row estimate is 10 ms
        for (int i = 0; i < 3; ++i) { m.observe(2.0,2,r3,3); CHECK(m.ratio() == 0 && !m.long_rule(c)); }
        m.observe(1.0,1,10.0,0);                     // k == 0: the verify is the one-row pass
        CHECK(std::fabs(m.ratio() - 0.1) < 1e-12 && m.long_rule(c));
        for (int i = 0; i < 80; ++i) m.observe(0.1,1,10.0,0); // cheap drafting: the ratio decays toward 0.01
        CHECK(m.ratio() < 0.04 && !m.long_rule(c));
        for (int i = 0; i < 80; ++i) m.observe(3.0,1,10.0,0); // dear drafting again: back on
        CHECK(m.long_rule(c));
        c.long_ratio = 0; CHECK(!m.long_rule(c));     // the OFF switch
    }
    // Rule: tau 0.5, first 2, step 1. P = 0.9, 0.72, 0.504, 0.252 -> extend while P(last) >= 0.5, window = 3.
    e::ConfidenceConfig c; c.tau = 0.5;
    const std::vector<float> p{0.9F,0.8F,0.7F,0.5F,0.9F,0.9F,0.9F};
    const auto sub = [&](std::uint32_t n) { return std::span<const float>(p.data(),n); };
    CHECK(c.next_depth({},5) == 2 && c.next_depth({},1) == 1);
    CHECK(c.next_depth(sub(2),5) == 3 && c.next_depth(sub(3),5) == 4 && c.next_depth(sub(4),5) == 4);
    CHECK(c.next_depth(sub(3),3) == 3); // capped by kmax
    CHECK(c.window(sub(4)) == 3 && c.window(sub(2)) == 2 && c.window({}) == 0);
    const std::vector<float> low{0.3F,0.9F};
    CHECK(c.window(low) == 1 && c.next_depth(low,5) == 2); // at least one verified depth; no extension below tau
    c.min1 = 0.4; CHECK(c.window(low) == 0);                 // MIN1: no draft verified
    c.min1 = 0; c.step = 2; CHECK(c.next_depth(sub(2),5) == 4 && c.next_depth(sub(4),7) == 4);
    c.tau = 1; CHECK(c.window(sub(4)) == 1 && c.next_depth(sub(2),7) == 2);
}
}
int main() {
    try {
        simulate(true,false); simulate(false,false); simulate(true,true);
        measured_profile();
        multi_stream();
        confidence();
        e::AutoPolicy changing;
        for (unsigned c = 0; c < 512; ++c) {
            const unsigned k = c%2 ? 2 : 0;
            changing.observe(0,k,k ? 41.0 : 24.7,k ? 11.5 : 0,k ? 28.8 : 24,0.7,outcome(k,c < 128),false);
        }
        CHECK(!changing.choose(7,0).k || changing.choose(7,0).probe);
        e::AutoPolicy policy;
        // A censored window reports only the compared prefix; its unexamined
        // suffix remains unobserved rather than becoming six extra failures.
        auto a = outcome(7,false);
        policy.observe(0,7,60,12,47,0.7,a,true);
        const auto receipt = policy.receipt();
        const auto& eligible = receipt.as_array()[0].find("decayed_eligible")->as_array();
        CHECK(eligible[0].as_double() == 1 && eligible[1].as_double() == 0);
        bool rejected = false;
        try { policy.observe(0,0,std::numeric_limits<double>::quiet_NaN(),0,1,0,outcome(0,true),false); }
        catch (const std::invalid_argument&) { rejected = true; }
        CHECK(rejected);
        std::cout << "auto policy: cold/length admission, bounded probes, survival censoring, decay, cost reversals, measured MTP profile and multi-stream (k<=1 cap, measured decline, per-depth cost gate, DFlash C1-only, STRIX_G4_DRAFT_MULTI=0) pass\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
