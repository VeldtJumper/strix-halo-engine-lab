#include "glm/spec/glm_spec_run.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace strix::glm::spec {
namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
struct IdentityScope {
    engine::Engine& e;
    engine::IdentityMode saved;
    explicit IdentityScope(engine::Engine& engine) : e(engine), saved(e.identity_mode()) {
        e.set_identity_mode(engine::IdentityMode::kR1);
    }
    ~IdentityScope() { e.set_identity_mode(saved); }
};
}
engine::Exchange::MergeCtl ControlState::next(unsigned kind) {
    engine::Exchange::MergeCtl c;
    c.cycle = cycle;
    c.step = step++;
    c.kind = kind;
    c.table_version = publisher.version;
    store_cost_table(publisher, c.table);
    return c;
}
void ControlState::consume(const engine::Exchange::MergeCtl& c) {
    table_valid = load_cost_table(c.table, c.table_version, merged);
}
void ControlState::observe(unsigned index, double elapsed) {
    if (rank == 0) observe_cost(publisher, index, elapsed);
}
Seed bootstrap(engine::Engine& e, std::uint32_t slot, std::span<const std::uint32_t> prompt,
               BootstrapPolicy policy, ControlState& control) {
    if (prompt.empty()) throw std::invalid_argument("spec bootstrap: empty prompt");
    IdentityScope scope(e);
    control.calibration = Calibration{};  // every new prompt starts with the same virtual observations
    control.cycle = control.step = 0;
    Seed seed;
    seed.p = prompt.size();
    if (policy == BootstrapPolicy::Chunked) {
        auto c = control.next(2);
        const auto b = e.prefill_mtp(slot, prompt, 0, c);
        control.consume(c);
        seed.y0 = b.y0;
        seed.draft = b.mtp.id.at(0);
        seed.confidence = b.mtp.confidence;
        seed.hp = b.hp;
        seed.flagged = b.mtp.flagged;
    } else {
        for (std::size_t j = 0; j < prompt.size(); ++j) {
            const engine::Row row{slot, prompt[j], j};
            const auto nx = e.step({&row, 1}).at(0);
            const bool last = j + 1 == prompt.size();
            if (last) e.save_trunk_logits();
            const auto token = last ? nx : prompt[j + 1];
            auto c = control.next(2);
            const auto m = e.mtp(slot, e.last_hidden(), {&token, 1}, j, c);
            control.consume(c);
            seed.flagged = seed.flagged || m.flagged;
            if (last) {
                seed.y0 = nx;
                seed.draft = m.id.at(0);
                seed.confidence = m.confidence;
                seed.hp = e.mtp_hidden();
                e.restore_trunk_logits();
            }
        }
    }
    return seed;
}
GreedyDecoder::GreedyDecoder(engine::Engine& e, std::uint32_t slot, Seed seed, const DecodePolicy& policy, unsigned budget,
                             ControlState& control)
    : e_(e), slot_(slot), policy_(policy), budget_(budget), control_(control) {
    if (budget == 0) { done_ = true; return; }
    if (seed.p > policy.capacity || std::uint64_t{budget - 1} > policy.capacity - seed.p)
        throw std::out_of_range("spec decode: output exceeds capacity");
    out_.tokens.reserve(budget);
    out_.tokens.push_back(seed.y0);
    disabled_ = seed.flagged;
    if (!disabled_) { drafts_.push_back(seed.draft); confidences_.push_back(seed.confidence); }
    hp_ = seed.hp;
    p_ = seed.p;
    start_ = Clock::now();
    done_ = out_.tokens.size() >= budget_;
}
bool GreedyDecoder::cycle() {
    if (done_) return false;
    auto& e = e_;
    auto& control = control_;
    auto& out = out_;
    auto& drafts = drafts_;
    auto& confidences = confidences_;
    auto& disabled = disabled_;
    auto& hp = hp_;
    auto& p = p_;
    auto& pending = pending_;
    const auto slot = slot_;
    const auto& policy = policy_;
    const unsigned budget = budget_;
    auto set_seed = [&](std::uint32_t token, float confidence) {
        drafts.clear(); confidences.clear();
        if (!disabled) { drafts.push_back(token); confidences.push_back(confidence); }
    };
    const auto tc = Clock::now();
    CycleRecord rec;
    rec.p = p;
    rec.replay = pending;
    // Freeze the last returned rank-0 table for this entire cycle. Local timings never feed a decision.
    const auto table = control.merged;
    const bool adaptive = policy.adaptive && control.table_valid;
    rec.table_version = table.version;
    const unsigned remaining = budget - static_cast<unsigned>(out.tokens.size());
    const unsigned limit = draft_limit(adaptive ? policy.kmax : std::min(policy.kmax, policy.fixed_k),
                                       remaining, p, policy.capacity);
    if (drafts.size() > limit) { drafts.resize(limit); confidences.resize(limit); }
    if (adaptive && !drafts.empty() && !window_gate_open(table, control.calibration.probability(1, confidences[0]))) {
        drafts.clear(); confidences.clear();
    }
    WindowEstimate estimate;
    auto add_estimate = [&](float confidence) {
        ++estimate.k;
        estimate.survival *= control.calibration.probability(estimate.k, confidence);
        estimate.expected += estimate.survival;
        estimate.cycle_ms = table.verify_ms[estimate.k + 1] +
                            static_cast<float>(estimate.k - 1) * table.step_ms + table.catchup_ms;
    };
    if (!drafts.empty()) add_estimate(confidences[0]);
    while (!drafts.empty() && drafts.size() < limit) {
        if (adaptive && !window_extend(table, estimate,
            control.calibration.probability(estimate.k + 1, confidences.back()), limit)) break;
        const auto token = drafts.back();
        auto c = control.next(3);
        const auto a = Clock::now();
        const auto m = e.mtp(slot, hp, {&token, 1}, p + drafts.size() - 1, c);
        const auto elapsed = ms(a, Clock::now());
        control.consume(c);
        control.observe(9, elapsed);
        rec.extend += elapsed;
        if (control.on_sample) control.on_sample({2, 1, 0, elapsed});
        if (m.flagged) { disabled = true; break; }
        drafts.push_back(m.id.at(0));
        confidences.push_back(m.confidence);
        hp = e.mtp_hidden();
        add_estimate(m.confidence);
    }
    rec.k = static_cast<unsigned>(drafts.size());
    std::vector<std::uint32_t> rows{out.tokens.back()};
    rows.insert(rows.end(), drafts.begin(), drafts.end());
    auto cv = control.next(1);
    const auto tv = Clock::now();
    const auto v = e.verify(slot, rows, p, cv);
    rec.verify = ms(tv, Clock::now());
    control.consume(cv);
    control.observe(rec.k + 1, rec.verify);
    if (control.on_sample) control.on_sample({1, rec.k + 1, pending, rec.verify});
    unsigned first_flag = kNoFlag;
    for (unsigned j = 0; j <= rec.k; ++j)
        if (v.flag.at(j) != 0 && first_flag == kNoFlag) first_flag = j;
    if (v.replay_flag) throw std::runtime_error("non-finite committed KDA row (replay)");
    const auto acc = accept_greedy(drafts, v.id, first_flag, Terminal{kNoFlag, remaining});
    if (acc.abort) throw std::runtime_error("non-finite required row: request aborted (as plain greedy)");
    for (unsigned d = 1; d <= rec.k; ++d) {
        auto& depth = out.depths[d];
        ++depth.offered;
        if (d <= acc.m + 1 && d <= remaining) {
            ++depth.reached;
            control.calibration.observe(d, confidences[d - 1], d <= acc.m);
        }
        if (d <= acc.m) ++depth.accepted;
    }
    out.tokens.insert(out.tokens.end(), drafts.begin(), drafts.begin() + acc.m);
    if (acc.has_bonus) out.tokens.push_back(acc.bonus);
    const auto tm = Clock::now();
    e.commit(slot, acc.rows, p);
    rec.commit = ms(tm, Clock::now());
    pending = acc.rows;
    rec.accepted = acc.m;
    rec.emitted = acc.m + (acc.has_bonus ? 1U : 0U);
    rec.rows = acc.rows;
    out.accepted += acc.m;
    out.offered += rec.k;
    ++out.histogram[rec.k];
    // Stable qualification trace of merged decisions; excludes timings and rank-local estimates.
    rec.checksum = 2166136261U;
    for (const auto x : {control.cycle, rec.k, rec.table_version, acc.m, acc.rows})
        rec.checksum = (rec.checksum ^ x) * 16777619U;
    if (!acc.stop) {
        if (!acc.has_bonus) throw std::logic_error("nonterminal acceptance without bonus");
        std::vector<std::uint32_t> cu(drafts.begin(), drafts.begin() + acc.m);
        cu.push_back(acc.bonus);
        auto c = control.next(2);
        const auto a = Clock::now();
        const auto m = e.mtp(slot, e.last_hidden(), cu, p, c);
        rec.catchup = ms(a, Clock::now());
        control.consume(c);
        control.observe(10, rec.catchup);
        if (control.on_sample) control.on_sample({3, acc.m + 1, 0, rec.catchup});
        disabled = disabled || m.flagged;
        set_seed(m.id.at(acc.m), m.confidence);
        hp = e.mtp_hidden() + static_cast<std::size_t>(acc.m) * 4096;
    }
    p += acc.rows;
    ++control.cycle;
    ++out.cycles;
    rec.other = ms(tc, Clock::now()) - rec.extend - rec.verify - rec.commit - rec.catchup;
    if (control.on_cycle) control.on_cycle(rec);
    last_ = rec;
    if (acc.stop || out.tokens.size() >= budget) done_ = true;
    return !done_;
}
void GreedyDecoder::finish() {
    if (finished_) return;
    finished_ = true;
    const auto tail = Clock::now();
    out_.loop_ms = ms(start_, tail);
    e_.flush(slot_);
    out_.tail_ms = ms(tail, Clock::now());
}
RunResult decode_greedy(engine::Engine& e, std::uint32_t slot, Seed seed, const DecodePolicy& policy,
                        unsigned budget, ControlState& control) {
    if (budget == 0) { e.flush(slot); return {}; }
    GreedyDecoder d(e, slot, seed, policy, budget, control);
    while (d.cycle()) {}
    d.finish();
    return d.result();
}
}  // namespace strix::glm::spec
