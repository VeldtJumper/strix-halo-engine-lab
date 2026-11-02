#pragma once
// Rank-0 production scheduler. Engine/runner/clock are compile-time seams for host control traces.
#include "minimax_sched.hpp"
#include "minimax_control.hpp"
#include "minimax_api.hpp"
#include "minimax_session.hpp"
#include "minimax/dspark/minimax_dspark.hpp"
#include "minimax/dspark/minimax_auto.hpp"
#include "strix/lookup/index.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
namespace strix::minimax::serve::lead {
namespace mm = strix::minimax::engine;
namespace sv = strix::serve;
namespace ms = strix::minimax::serve;
namespace ds = strix::minimax::dspark;
using Clock = std::chrono::steady_clock;
using ms::Command;
using ms::Control;
using ms::exchange;
constexpr std::uint32_t kStepRows = 8;
inline double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b-a).count(); }
struct Drafts {
    std::array<std::uint32_t, ds::block> tokens{};
    std::array<float, ds::block> confidence{};
};
inline std::uint64_t fnv(const void* p, std::size_t bytes, std::uint64_t h = 14695981039346656037ULL) {
    const auto* b = static_cast<const std::uint8_t*>(p);
    for (std::size_t i = 0; i < bytes; ++i) h = (h ^ b[i]) * 1099511628211ULL;
    return h;
}
inline std::string hex(std::uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
    return b;
}
// ---- rank 0: scheduler policy (runtime switchable) and counters ----
using Sched = ms::Sched;
struct Policy : ms::RequestPolicy {
    bool nest = true;
    bool short_first = true;
    ds::Mode dspark_mode = ds::Mode::automatic;
    ds::Prior dspark_prior = ds::Prior::structural_v1;
    std::uint32_t inflight = 1;   // DSpark contexts kept while <= this many requests are in flight (measurement: C2 slower)
    std::uint32_t draft_slots = 8;  // at most this many slots draft in one cycle
    bool lookup = true;             // prompt-lookup windows (STRIX_MM_LOOKUP at startup; /admin/sched "lookup")
    std::uint64_t epoch = 0;      // bumped by every change (the lead resets its cost tables / gates)
};
struct Counters {
    double step_s = 0, prefill_s = 0, draft_s = 0, idle_s = 0, stall_s = 0;
    unsigned max_unit_layers = 0;
    std::uint64_t cancels = 0;  // open prefills abandoned (client gone)
    // DSpark coverage: cycles with >= 2 windows (k >= 1), of which with unequal k; windows with a rejected draft
    std::uint64_t multi_windows = 0, unequal_windows = 0, rejections = 0;
    // prompt lookup: windows drafted from the request's own context, their rows and accepted tokens; host time
    std::uint64_t lookup_windows = 0, lookup_drafted = 0, lookup_accepted = 0;
    double lookup_s = 0;
    std::uint64_t steps = 0, step_rows = 0, units = 0, prefills = 0, drafts = 0, drafted = 0, accepted = 0, plain_cycles = 0,
                  draft_cycles = 0;
    std::array<std::uint64_t, kStepRows + 1> rows_hist{};
};
// Per-request extras beyond sv::GenerationStats, for the reply's timings and the request log.
using Extra = ms::RequestExtra;
struct Shared {
    std::mutex m;
    Policy policy;
    Counters counters;
    std::string unlimited_reason, nest_reason;
    unsigned nest_rows = 0;
    std::uint64_t active_id = 0, parked_id = 0;
    bool nest_taps = false;
    std::uint32_t dspark_live = 0, dspark_contexts = 0;
    std::unordered_map<const sv::GenerationTicket*, Extra> extras;
};

struct Running {
    sv::TakenJob job;
    std::uint32_t slot = 0;
    std::uint64_t pos = 0;  // position of `last` (the next input token)
    std::uint32_t last = 0;
    sv::GenerationStats stats;
    Extra x;
    ms::RequestPolicy policy;  // effective operation policy, retained for prefill, cancellation and decoder residency
    Clock::time_point decode_start, first_emit, last_emit;
    std::uint64_t request_id = 0;  // stable across slot reuse, for the optional timeline
    Clock::time_point prefill_start;
    ms::NestQuota nest_quota;
    bool prefilling = false;
    bool context = false;  // DSpark: the drafter context of this slot is complete
    ds::AutoRequest automatic;
    ds::Prior dspark_prior = ds::Prior::structural_v1;
    ds::SurvivalController survival{16};
    double tau = 1.0;      // EMA of emitted tokens per drafting cycle (which slots draft first)
    // Prompt lookup (rank 0 only, strix/lookup/index.hpp): an index over the prompt and every committed token; each
    // cycle's longest-match candidate is shadow-scored token by token on what the request commits (per match-length
    // bucket and depth, decayed). Exact: verification decides every token.
    struct LookupPending {
        std::uint64_t first = 0;  // stream position of the candidate's first token
        std::uint32_t count = 0, done = 0, bucket = 0;
        std::array<std::uint32_t, ds::block> token{};
    };
    std::unique_ptr<strix::lookup::Index> lk;
    std::uint64_t lk_next = 0;  // stream position of the next committed token
    std::vector<LookupPending> lk_pending;
    std::array<std::array<double, ds::block>, 5> lk_eligible{}, lk_accepted{};
    std::array<double, 5> lk_evidence{};
    double ds_tokens = 0, ds_cost = 0, ds_n = 0;  // this request's DSpark windows (decayed): emitted tokens, cycle seconds
};

inline bool is_stop(const Running& r, std::uint32_t id) {
    return std::find(r.job.job.stop_tokens.begin(), r.job.job.stop_tokens.end(), id) != r.job.job.stop_tokens.end();
}

// Cycle costs measured by rank 0 (seconds), by total rows and step shape: plain (one row per slot), one slot's window
// (fused MSA), mixed (several slots, at least one window: per-row MSA). Unmeasured row counts are scaled from the
// measured ones of the same shape by 1 + 0.12 (R - r) (the measurement verify slope), then from plain.
struct Costs {
    std::array<std::array<double, kStepRows + 1>, 3> v{};
    double draft = 0;  // one slot's draft (block + head + Markov chain)
    void reset() { *this = Costs{}; }
    static void ema(double& c, double x) { c = c <= 0 ? x : 0.8 * c + 0.2 * x; }
    void observe(unsigned shape, std::uint32_t rows, double s) { ema(v[shape][rows], s); }
    [[nodiscard]] bool measured(unsigned shape, std::uint32_t rows) const { return v[shape][rows] > 0; }
    [[nodiscard]] double at(unsigned shape, std::uint32_t rows) const {
        for (unsigned sh : {shape, 0U}) {
            double base = 0;
            int n = 0;
            for (std::uint32_t r = 1; r <= kStepRows; ++r)
                if (v[sh][r] > 0) {
                    base += v[sh][r] / (1.0 + 0.12 * (r - 1));
                    ++n;
                }
            if (v[sh][rows] > 0) return v[sh][rows];
            if (n > 0) return base / n * (1.0 + 0.12 * (rows - 1));
        }
        return 0.05 * (1.0 + 0.12 * (rows - 1));
    }
};

inline void log_request(const Running& r, const char* finish) {
    std::fprintf(stderr, "mmserve-dspark mode=%s reason=%s prior=%s\n", r.x.dspark_mode.c_str(), r.x.dspark_reason.c_str(), ds::prior_name(r.dspark_prior));
    std::fprintf(stderr, "mmserve-policy sched=%s service_ms=%.1f max_unit_layers=%u short_first=%u fallback=%s nested=%u parent_request=%llu task_id=%llu\n",
                 r.x.sched.c_str(), r.x.service_ms, r.x.max_unit_layers, r.x.short_first ? 1U : 0U, r.x.fallback_reason.c_str(), r.x.nested ? 1U : 0U,
                 static_cast<unsigned long long>(r.x.parent_request), static_cast<unsigned long long>(r.request_id));
    std::fprintf(stderr,
                 "{\"event\":\"mmserve-req\",\"slot\":%u,\"prompt_tokens\":%llu,\"prompt_fnv\":\"%s\",\"tokens\":%llu,\"ids_fnv\":\"%s\","
                 "\"finish\":\"%s\",\"queue_ms\":%.1f,\"ttft_ms\":%.1f,\"prefill_ms\":%.1f,\"prefill_units\":%llu,\"decode_ms\":%.1f,"
                 "\"task_id\":%llu,\"parent_task_id\":%llu,\"nest_count\":%llu,\"park_wall_ms\":%.1f,\"nest_service_ms\":%.1f,\"nest_switch_ms\":%.1f,\"max_gap_ms\":%.1f,\"stall_ms\":%.1f,\"cycles\":%llu,\"drafted\":%llu,\"accepted\":%llu,\"context\":%s}\n",
                 r.slot, static_cast<unsigned long long>(r.stats.prompt_tokens), hex(r.x.prompt_fnv).c_str(),
                 static_cast<unsigned long long>(r.stats.generated_tokens), hex(r.x.ids_fnv).c_str(), finish,
                 std::chrono::duration<double, std::milli>(r.job.started - r.job.submitted).count(), r.x.ttft_ms, r.x.prefill_wall_ms,
                 static_cast<unsigned long long>(r.x.units), r.stats.decode_ms,
                 static_cast<unsigned long long>(r.request_id), static_cast<unsigned long long>(r.x.parent_request),
                 static_cast<unsigned long long>(r.x.nest_count), r.x.park_wall_ms, r.x.nest_service_ms, r.x.nest_switch_ms, r.x.max_gap_ms, r.x.stall_ms,
                 static_cast<unsigned long long>(r.x.cycles), static_cast<unsigned long long>(r.x.drafted),
                 static_cast<unsigned long long>(r.x.accepted), r.x.context ? "true" : "false");
}

// ---- rank 0: the scheduler ----
template<class Engine, class Runner, class Time = std::chrono::steady_clock>
class Lead {
public:
    Lead(Engine& eng, Runner& run, sv::GenerationQueue& queue, Shared& shared, std::uint64_t max_rows, bool taps, std::atomic<bool>& stop,
         std::function<std::string(const std::vector<std::uint32_t>&)> prompt_bytes = {}, ms::SessionService* session = nullptr)
        : eng_(eng), run_(run), queue_(queue), sh_(shared), slots_(eng.slots()), capacity_(eng.capacity()), max_rows_(max_rows),
          run_slots_(slots_), stop_(stop), session_(session), taps_(taps) {
        prompt_bytes_ = std::move(prompt_bytes);
        round_ = run.control_round;
        const char* v = std::getenv("STRIX_MM_SERVE_TIMELINE");
        timeline_on_ = v != nullptr && std::strcmp(v, "1") == 0;
        if (timeline_on_) timeline_origin_ = Time::now();
        // prompt lookup: on / off is the policy (STRIX_MM_LOOKUP at startup); STRIX_MM_LOOKUP_MARGIN (1..2),
        // STRIX_MM_LOOKUP_TOKENS (index capacity)
        if (const char* m = std::getenv("STRIX_MM_LOOKUP_MARGIN"); m != nullptr && *m) lookup_margin_ = std::clamp(std::atof(m), 1.0, 2.0);
        if (const char* t = std::getenv("STRIX_MM_LOOKUP_TOKENS"); t != nullptr && *t)
            lookup_tokens_ = static_cast<std::uint32_t>(std::clamp(std::atoll(t), 1024LL, 1LL << 20));
    }

    void loop() try {
        auto last_round = Time::now();
        while (!stop_.load() && !queue_.stopping()) {
            refresh_policy();
            sync_taps();
            bool worked = false;
            // a client gone while its prefill is open: the task is abandoned at this unit boundary (both ranks)
            for (std::uint32_t slot = 0; slot < slots_; ++slot) {
                auto& r = run_slots_[slot];
                if (!r || !r->prefilling || !r->job.ticket->cancelled()) continue;
                Control c;
                c.command = static_cast<std::uint32_t>(Command::cancel);
                c.task_id = r->request_id;
                c.round = round_++;
                (void)exchange(eng_, c);
                const auto progress = eng_.prefill_progress(eng_.prefill_progress(true).id == c.task_id);
                run_.prefill_cancel(c.task_id);
                if (timeline_on_) timeline("cancel", Time::now(), "ticket", slot, progress.layer, progress.layer, timeline_task(progress));
                layer_.reset_pending();
                r->decode_start = Time::now();
                r->stats.prefill_ms = elapsed(r->prefill_start);
                finish(r, "cancelled");
                if (slot == parked_slot_) parked_slot_ = slots_;
                add([](Counters& k) { ++k.cancels; });
            }
            if (!eng_.prefill_active() && eng_.prefill_progress(true).open) unpark();
            if (eng_.prefill_active() && !eng_.prefill_progress(true).open && pol_.nest) {
                if (!admit(true) && nested_needs_decode_ && decoding_count() > 0) {
                    cycle();
                    (void)admit(true);  // re-evaluate real ages after the complete cycle
                }
            }
            {
                std::lock_guard<std::mutex> l(sh_.m);
                sh_.active_id = eng_.prefill_active() ? eng_.prefill_progress().id : 0;
                sh_.parked_id = eng_.prefill_progress(true).open ? eng_.prefill_progress(true).id : 0;
            }
            for (auto& r : run_slots_)
                if (r && r->policy.sched == Sched::interleave && !r->prefilling && r->job.ticket->cancelled()) finish(r, "cancelled");
            // admission: one open prefill at a time
            if (!eng_.prefill_open()) worked |= admit();
            if (eng_.prefill_open()) {
                const auto& policy = run_slots_[prefill_slot_]->policy;
                const bool off = run_slots_[prefill_slot_]->policy.sched == Sched::off;
                const bool fresh = run_slots_[prefill_slot_]->x.units == 0;  // a started task keeps interleaving
                // Whole admissions share a deadline; started tasks retain layer boundaries.
                const auto n = decoding_count();
                const bool whole = run_slots_[prefill_slot_]->x.nested || off || ms::runs_whole(fresh, predicted_rest_ms(), policy.service_ms, n > 0,
                    live_gap_ms(policy.gap_ms), cycle_ms(std::min(n + 1, kStepRows)), longest_wait_ms(), ms::control_margin_ms);
                if (off || whole) {
                    prefill_unit(~0U);
                    last_round = Time::now();
                    // Reconsider the next candidate against all last-emission deadlines. Never wait for arrivals.
                    continue;
                }
            }
            if (decoding_count() > 0 && (!eng_.prefill_open() || !deadline_serviced_)) {
                cycle();
                worked = true;
            }
            if (eng_.prefill_open()) {
                // A cycle just serviced every decoder. If a single layer still cannot fit, advance once;
                // no schedule can meet that target without changing the numerical unit.
                prefill_unit(std::max(1U, unit_layers()));
                worked = true;
            }
            if (worked) {
                last_round = Time::now();
                continue;
            }
            if (std::chrono::duration<double>(Time::now() - last_round).count() > 1.0) {  // keep rank 1 and the link alive
                Control hb;
                hb.command = static_cast<std::uint32_t>(Command::heartbeat);
                hb.round = round_++;
                const auto hb0 = timeline_on_ ? Time::now() : Clock::time_point{};
                (void)exchange(eng_, hb);
                if (timeline_on_) timeline("heartbeat", hb0, "deadline");
                last_round = Time::now();
            }
            const auto idle0 = Time::now();
            idle(std::chrono::milliseconds(2));
            add([&](Counters& c) { c.idle_s += secs(idle0, Time::now()); });
            if (timeline_on_) timeline("idle", idle0, "defer");
            flush_timeline();
        }
        for (const bool parked : {false, true}) {
            const auto p = eng_.prefill_progress(parked);
            if (!p.open) continue;
            Control c;
            c.command = static_cast<std::uint32_t>(Command::cancel);
            c.task_id = p.id;
            c.round = round_++;
            (void)exchange(eng_, c);
            run_.prefill_cancel(p.id);
        }
        for (auto& r : run_slots_)
            if (r) finish(r, "cancelled");
        if (pending_) {
            sv::GenerationStats stats;
            stats.finish = "cancelled";
            queue_.finish(*pending_, stats);
        }
        flush_timeline();
        Control bye;
        bye.command = static_cast<std::uint32_t>(Command::shutdown);
        bye.round = round_++;
        (void)exchange(eng_, bye);
    } catch (...) {
        eng_.abort("control channel: protocol failure");
        throw;
    }

private:
    static double elapsed(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Time::now() - start).count(); }
    static void idle(std::chrono::milliseconds duration) {
        if constexpr (requires { Time::sleep_for(duration); }) Time::sleep_for(duration);
        else std::this_thread::sleep_for(duration);
    }
    template <class F> void add(F f) {
        std::lock_guard<std::mutex> l(sh_.m);
        f(sh_.counters);
    }
    // The engine's draft taps follow DSpark on / off (taps cost capture work in every call): switched on both ranks with
    // no prefill open; requests admitted while they were off have no drafter context.
    void sync_taps() {
        bool active = false;
        for (const auto& r : run_slots_) active |= r && r->context;
        const bool want = run_.dd != nullptr && (pol_.dspark_mode == ds::Mode::on ||
                          (pol_.dspark_mode == ds::Mode::automatic && active));
        if (want == taps_ || eng_.prefill_open() || parked_slot_ < slots_) return;
        Control c;
        c.command = static_cast<std::uint32_t>(Command::taps);
        c.round = round_++;
        c.pflags = want ? 1U : 0U;
        (void)exchange(eng_, c);
        eng_.set_taps(want);
        taps_ = want;
    }
    void refresh_policy() {
        std::lock_guard<std::mutex> l(sh_.m);
        if (sh_.policy.epoch != pol_.epoch) {  // reset optimization gates, retain deadline safety estimates
            std::uint32_t retiring = 0;
            for (const auto& r : run_slots_) if (r && r->context) retiring |= 1U << r->slot;
            if (retiring != 0) {
                Control c;
                c.command = static_cast<std::uint32_t>(Command::retire);
                c.round = round_++;
                c.append = retiring;
                (void)exchange(eng_, c);
                run_.retire_contexts(retiring);
            }
            for (auto& r : run_slots_) if (r) {
                r->context = false;
                r->x.context = false;
                r->automatic.retire("plain:policy_epoch");
                r->x.dspark_reason = r->automatic.reason;
            }
            costs_.reset();
            // Deadline safety samples survive policy/counter epochs.
            gates_.assign(kStepRows + 1, ds::DraftGate());
            explore_ = {};
        }
        pol_ = sh_.policy;
        sh_.dspark_live = inflight();
        sh_.dspark_contexts = 0;
        for (const auto& r : run_slots_) if (r && r->context) ++sh_.dspark_contexts;
    }
    std::uint32_t decoding_count() const {
        std::uint32_t n = 0;
        for (const auto& r : run_slots_) n += r && !r->prefilling ? 1U : 0U;
        return n;
    }
    std::uint32_t inflight() const {
        std::uint32_t n = 0;
        for (const auto& r : run_slots_) n += r ? 1U : 0U;
        return n;
    }

    void finish(std::optional<Running>& r, const std::string& why) {
        if (r->slot == parked_slot_) {
            r->x.park_wall_ms += elapsed(park_start_);
            const double switch_ms = ms::nest_switch_ms(elapsed(nest_start_), nest_service_ms_);
            r->x.nest_switch_ms += switch_ms;
            nest_switch_upper_ms_ = std::max(nest_switch_upper_ms_, switch_ms);
        }
        r->stats.finish = why;
        r->stats.decode_ms = elapsed(r->decode_start);
        r->stats.cycles = r->x.cycles;
        r->stats.proposed = r->x.drafted;
        r->stats.accepted = r->x.accepted;
        // armed instrumentation (requests the queue marked observed at submission): submit -> first emitted token, the
        // time emit() already took right before that token's publish; none when no token was emitted
        if (r->job.job.observed && r->stats.generated_tokens > 0) r->stats.first_token_ms = r->x.ttft_ms;
        log_request(*r, why.c_str());
        {
            std::lock_guard<std::mutex> l(sh_.m);
            sh_.extras[r->job.ticket.get()] = r->x;
        }
        queue_.finish(r->job, r->stats);
        r.reset();
    }

    // Emits `ids` of request r in order until a stop token, a cancellation or its token limit; returns false when r ended.
    bool emit(std::optional<Running>& r, std::span<const std::uint32_t> ids, std::uint32_t* emitted = nullptr) {
        for (const auto id : ids) {
            if (emitted != nullptr) ++*emitted;
            const auto now = Time::now();
            if (r->stats.generated_tokens == 0) {
                r->first_emit = now;
                r->x.ttft_ms = std::chrono::duration<double, std::milli>(now - r->job.submitted).count();
            } else r->x.max_gap_ms = std::max(r->x.max_gap_ms, std::chrono::duration<double, std::milli>(now - r->last_emit).count());
            r->last_emit = now;
            r->x.ids_fnv = fnv(&id, 4, r->x.ids_fnv);
            ++r->stats.generated_tokens;
            const bool more = r->job.job.on_tokens(std::span(&id, 1));
            if (is_stop(*r, id)) {
                finish(r, "stop");
                return false;
            }
            if (!more) {
                finish(r, "cancelled");
                return false;
            }
            if (r->stats.generated_tokens >= r->job.job.max_tokens) {
                finish(r, "length");
                return false;
            }
        }
        return true;
    }

    bool protected_decoder(Clock::time_point submitted) const {
        for (const auto& r : run_slots_)
            if (r && !r->prefilling && !r->job.ticket->cancelled() && ms::protects(submitted, r->first_emit, r->stats.generated_tokens > 0)) return true;
        return false;
    }
    bool waiting_child(const sv::GenerationJob& job, Clock::time_point submitted,
                       Clock::time_point parent_submitted, Clock::time_point started) const {
        return (!session_ || job.session.empty()) && job.prompt.size() <= std::min(512U, eng_.nested_rows()) && job.prompt.size() > 0 &&
            job.prompt.size() + 1 < capacity_ && ms::waiting_short(submitted, parent_submitted, started);
    }
    bool nested_candidate(const sv::GenerationJob& job, Clock::time_point submitted) {
        const auto reject = [&](const char* reason) {
            if (timeline_on_) timeline("nest-reject", Time::now(), reason, prefill_slot_, 0, 0,
                "\"candidate_rows\":" + std::to_string(job.prompt.size()));
            return false;
        };
        if (!eng_.prefill_active()) return reject("no boundary");
        const auto& parent = *run_slots_[prefill_slot_];
        if (!ms::nest_timing(parent.policy.sched == Sched::off, protected_decoder(submitted),
            waiting_child(job, submitted, parent.job.submitted, parent.prefill_start))) return reject("timing");
        const auto n = decoding_count();
        const double gap = live_gap_ms(pol_.gap_ms);
        const double predicted = predicted_prompt_ms(job.prompt.size());
        const auto fits = [&](double wait) {
            return job.prompt.size() + 1 < capacity_ && ms::nest_fits(job.prompt.size(), eng_.nested_rows(), pol_.nest,
                parent.policy.sched == Sched::interleave, pol_.sched == Sched::interleave,
                *eng_.prefill_unlimited_reason() == 0 && parent.x.units > 0,
                eng_.prefill_progress(true).open, inflight() < slots_, parent.nest_quota,
                predicted, gap, wait, cycle_ms(std::min(n + 1, kStepRows)), nest_switch_upper_ms_, n > 0);
        };
        const bool ready = fits(longest_wait_ms());
        if (ready) return true;
        if (fits(0)) { nested_needs_decode_ = true; return reject("decode-needed"); }
        if (timeline_on_) {
            if (parent.x.units == 0 || *eng_.prefill_unlimited_reason() != 0) return reject("no boundary");
            if (eng_.prefill_progress(true).open || inflight() >= slots_ || eng_.nested_rows() == 0) return reject("bank");
            if (!parent.nest_quota.eligible() || parent.nest_quota.wall_ms >= 2 * std::min(1300.0, gap)) return reject("quota");
            return reject("fit");
        }
        return false;
    }
    void unpark() {
        const auto slot = parked_slot_;
        if (slot >= slots_ || !run_slots_[slot]) throw std::runtime_error("nested parent reservation lost");
        Control c;
        c.command = static_cast<std::uint32_t>(Command::resume_parked);
        c.round = round_++;
        c.task_id = run_slots_[slot]->request_id;
        const auto start = Time::now();
        (void)exchange(eng_, c);
        eng_.prefill_resume_parked(c.task_id);
        if (timeline_on_) timeline("unpark", start, "nested", slot, eng_.prefill_layer(), eng_.prefill_layer(),
            timeline_task(eng_.prefill_progress()));
        const double wall = elapsed(park_start_);
        run_slots_[slot]->nest_quota.wall_ms += wall;
        run_slots_[slot]->x.park_wall_ms += wall;
        prefill_slot_ = slot;
        parked_slot_ = slots_;
        layer_.reset_pending();
        // The child's first emission starts a real deadline; decode only if the next parent unit cannot fit.
        deadline_serviced_ = unit_layers() > 0;
        const double switch_ms = ms::nest_switch_ms(elapsed(nest_start_), nest_service_ms_);
        run_slots_[slot]->x.nest_switch_ms += switch_ms;
        nest_switch_upper_ms_ = std::max(nest_switch_upper_ms_, switch_ms);
    }

    bool admit(bool nested = false) {
        nested_needs_decode_ = false;
        const auto a0 = Time::now();
        std::uint32_t free_slot = slots_;
        for (std::uint32_t s = 0; s < slots_; ++s)
            if (!run_slots_[s]) {
                free_slot = s;
                break;
            }
        if (free_slot >= slots_) return false;
        if (session_ && !nested && inflight() == 0 && session_->tasks.pending.load(std::memory_order_relaxed)) {
            if (auto task = session_->tasks.take()) {
                try { task->result.set_value(dispatch_session(task->body)); }
                catch (...) { task->result.set_value(sv::session_error("unavailable", "session execution stopped")); throw; }
                return true;
            }
        }
        // Reserve one selected candidate without opening device state; it remains ahead of later arrivals.
        if (!pending_) {
            pending_short_first_ = false;
            pending_ = queue_.take(false, pol_.sched == Sched::interleave && pol_.short_first ?
                sv::QueueSelection::short_first : sv::QueueSelection::fifo, {},
                [&](Clock::time_point submitted) {
                    pending_short_first_ = ms::reorder_prompts(pol_.sched, pol_.short_first, protected_decoder(submitted));
                    return pending_short_first_;
                }, nested ? std::function<bool(const sv::GenerationJob&, Clock::time_point)>(
                    [&](const sv::GenerationJob& j, Clock::time_point submitted) { return nested_candidate(j, submitted); }) :
                    std::function<bool(const sv::GenerationJob&, Clock::time_point)>{},
                nested && timeline_on_ ? std::function<void()>([&] {
                    timeline("nest-reject", Time::now(), "overtake");
                }) : std::function<void()>{});
        }
        if (!pending_) return false;
        // Nested take is an atomic eligibility-aware reservation. Consume it in this admission;
        // never hold a rejected/oversized nested candidate ahead of later eligible queue entries.
        if (pending_->ticket->cancelled()) {
            queue_.finish(*pending_, {});
            pending_.reset();
            return true;
        }
        if (session_ && !pending_->job.session.empty()) {
            if (nested || inflight() != 0) return false;
            auto t = std::move(pending_);
            pending_.reset();
            t->started = Time::now();
            auto command = sv::Json::object(); command.set("op", sv::Json("generate")); command.set("id", sv::Json(t->job.session));
            auto tokens = sv::Json::array(), stops = sv::Json::array();
            for (auto id : t->job.prompt) tokens.push(sv::Json(std::uint64_t(id)));
            for (auto id : t->job.stop_tokens) stops.push(sv::Json(std::uint64_t(id)));
            command.set("tokens", std::move(tokens)); command.set("stops", std::move(stops));
            command.set("count", sv::Json(std::uint64_t(t->job.max_tokens)));
            const auto begin = Time::now();
            sv::GenerationStats stats; stats.prompt_tokens = t->job.prompt.size();
            auto publish = [&](std::span<const std::uint32_t> ids) {
                if (stats.generated_tokens == 0 && t->job.observation_complete()) stats.first_token_ms = elapsed(t->submitted);
                stats.generated_tokens += ids.size();
                return !t->ticket->cancelled() && (!t->job.on_tokens || t->job.on_tokens(ids));
            };
            auto reply = dispatch_session(command.dump(), publish);
            stats.decode_ms = elapsed(begin);
            if (const auto* error = reply.find("error")) queue_.finish(*t, stats, error->find("code")->as_string() + ": " + error->find("message")->as_string());
            else {
                std::vector<std::uint32_t> ids;
                for (const auto& id : reply.find("token_ids")->as_array()) ids.push_back(static_cast<std::uint32_t>(id.as_int()));
                stats.generated_tokens = ids.size(); stats.cached_tokens = reply.find("hit")->as_bool() ? stats.prompt_tokens : 0;
                const auto* finish = reply.find("finish");
                stats.finish = t->ticket->cancelled() ? "cancelled" : finish ? finish->as_string() :
                               ids.size() < t->job.max_tokens ? "stop" : "length";
                queue_.finish(*t, stats);
            }
            return true;
        }
        const auto& candidate = pending_->job.prompt;
        const auto n = decoding_count();
        const bool nesting = !nested && pending_->reordered && pol_.nest && pol_.sched == Sched::interleave &&
            candidate.size() > 512 && inflight() + 1 < slots_ && *eng_.prefill_unlimited_reason() == 0 &&
            queue_.has_waiting([&](const sv::GenerationJob& j, Clock::time_point submitted) {
                return waiting_child(j, submitted, pending_->submitted, a0) &&
                    ms::nest_fits(j.prompt.size(), eng_.nested_rows(), true, true, true, true, false, true,
                        ms::NestQuota{}, predicted_prompt_ms(j.prompt.size()), live_gap_ms(pol_.gap_ms),
                        longest_wait_ms(), cycle_ms(std::min(n + 1, kStepRows)), nest_switch_upper_ms_, n > 0);
            });
        const bool off = ms::off_admission(pol_.sched, protected_decoder(pending_->submitted), nested || nesting);
        if (!nested && !off && n > 0) {
            const double gap = live_gap_ms(pol_.gap_ms), next = cycle_ms(std::min(n + 1, kStepRows));
            const double predicted = predicted_prompt_ms(candidate.size());
            const bool whole = ms::runs_whole(true, predicted, pol_.service_ms, true, gap, next,
                                              longest_wait_ms(), ms::control_margin_ms);
            if (!whole && !deadline_serviced_) {
                step_reason_ = "defer";
                return false;
            }
        }
        auto t = std::move(pending_);
        pending_.reset();
        t->started = Time::now();
        auto& prompt = t->job.prompt;
        if (prompt.empty() || prompt.size() + 1 >= capacity_) {
            queue_.finish(*t, {}, "prompt of " + std::to_string(prompt.size()) + " tokens does not fit the context (" +
                                      std::to_string(capacity_) + ")");
            return true;
        }
        Running r;
        r.policy = pol_;
        r.dspark_prior = pol_.dspark_prior;
        if (!nested && off) r.policy.sched = Sched::off;  // effective operation policy, including cancellation/DSpark residency
        r.request_id = ++request_id_;
        r.x.task_id = r.request_id;
        r.slot = free_slot;
        r.prefilling = true;
        r.stats.prompt_tokens = prompt.size();
        r.x.prompt_fnv = fnv(prompt.data(), prompt.size() * 4);
        r.x.ids_fnv = 14695981039346656037ULL;
        // DSpark context: a drafter, DSpark on, few enough requests in flight, a one-chunk prompt (longer ones keep the
        // grouped prefill), room for the 7-row block
        // A second admission retires the original auto context permanently before any newcomer work.
        if (pol_.dspark_mode == ds::Mode::automatic && inflight() != 0) {
            std::uint32_t retiring = 0;
            for (auto& old : run_slots_) if (old && old->context) {
                retiring |= 1U << old->slot;
                old->context = old->x.context = false;
                old->automatic.retire("plain:concurrency");
                old->x.dspark_reason = old->automatic.reason;
            }
            if (retiring) {
                Control c; c.command = static_cast<std::uint32_t>(Command::retire);
                c.round = round_++; c.append = retiring;
                (void)exchange(eng_, c); run_.retire_contexts(retiring);
                sync_taps();
            }
        }
        const bool competing = nested || parked_slot_ < slots_ ||
            queue_.has_waiting([](const sv::GenerationJob&, Clock::time_point) { return true; });
        ds::Features structure;
        if (pol_.dspark_mode == ds::Mode::automatic && pol_.dspark_prior == ds::Prior::structural_v1 &&
            run_.dd && inflight() == 0 && !competing &&
            prompt.size() <= max_rows_ && prompt.size() < ds::long_prompt_tokens && prompt_bytes_)
            structure = ds::features(prompt_bytes_(prompt));
        r.automatic.admit({pol_.dspark_mode, run_.dd != nullptr && pol_.inflight != 0, inflight(), competing,
                           prompt.size(), max_rows_, t->job.max_tokens, prompt.size() + ds::block + 1 < capacity_,
                           pol_.dspark_prior, structure});
        const bool promoted = pol_.dspark_mode == ds::Mode::automatic && r.automatic.state == ds::AutoRequest::State::active;
        if (promoted && !taps_ && !eng_.prefill_open() && parked_slot_ >= slots_) {
            Control c; c.command = static_cast<std::uint32_t>(Command::taps); c.round = round_++; c.pflags = 1;
            (void)exchange(eng_, c); eng_.set_taps(true); taps_ = true;
        }
        r.context = run_.dd != nullptr && (pol_.dspark_mode == ds::Mode::on || promoted) && taps_ &&
                    inflight() + 1 <= pol_.inflight && prompt.size() <= max_rows_ && prompt.size() + ds::block + 1 < capacity_;
        r.x.dspark_mode = ds::mode_name(pol_.dspark_mode);
        r.x.dspark_reason = pol_.dspark_mode == ds::Mode::on ? (r.context ? "legacy" : "plain:legacy_ineligible") : r.automatic.reason;
        r.x.context = r.context;
        r.x.sched = pol_.sched == Sched::off ? "off" : "interleave";
        r.x.service_ms = pol_.service_ms;
        r.x.short_first = pending_short_first_;
        r.x.fallback_reason = eng_.prefill_unlimited_reason();
        if (off) r.x.fallback_reason = pol_.sched == Sched::off ? "sched_off" : "no_protected_decoder";
        const bool admission_whole = timeline_on_ && (r.policy.sched == Sched::off || ms::runs_whole(true, predicted_prompt_ms(prompt.size()),
            pol_.service_ms, n > 0, live_gap_ms(pol_.gap_ms), cycle_ms(std::min(n + 1, kStepRows)), longest_wait_ms(), ms::control_margin_ms));
        const auto parent_id = nested ? run_slots_[prefill_slot_]->request_id : 0;
        if (nested) {
            auto& parent = *run_slots_[prefill_slot_];
            parent.nest_quota.admitted();
            ++parent.x.nest_count;
            parked_slot_ = prefill_slot_;
            nest_start_ = a0;
            nest_service_ms_ = 0;
            park_start_ = Time::now();
            Control park;
            park.command = static_cast<std::uint32_t>(Command::park);
            park.round = round_++;
            park.task_id = parent_id;
            (void)exchange(eng_, park);
            const auto progress = eng_.prefill_progress();
            eng_.prefill_park(parent_id);
            if (timeline_on_) timeline("park", park_start_, "nested", parent.slot, progress.layer, progress.layer,
                timeline_task(progress, r.request_id));
            r.x.nested = true;
            r.x.parent_request = parent_id;
        }
        Control c;
        c.task_id = r.request_id;
        c.parent_task_id = parent_id;
        c.bank = nested ? 1U : 0U;
        c.slot_generation = ++slot_generation_[free_slot];
        c.command = static_cast<std::uint32_t>(nested ? Command::nest : Command::prefill);
        c.round = round_++;
        c.pslot = free_slot;
        c.plen = static_cast<std::uint32_t>(prompt.size());
        c.pflags = r.context ? 1U : 0U;
        (void)exchange(eng_, c);
        run_.send_prompt(prompt);
        const auto begin0 = timeline_on_ ? Time::now() : Clock::time_point{};
        run_.prefill_begin(free_slot, prompt, r.context, c.task_id, parent_id, c.slot_generation);
        layer_.reset_pending();
        r.job = std::move(*t);
        r.prefill_start = Time::now();
        run_slots_[free_slot] = std::move(r);
        prefill_slot_ = free_slot;
        add([](Counters& k) { ++k.prefills; });
        if (timeline_on_) timeline(nested ? "nest" : "admit", a0, admission_whole ? "whole" : "deadline", prefill_slot_, 0, 0, timeline_prefill(prefill_slot_, true) + "," + timeline_task(eng_.prefill_progress()), begin0);
        return true;
    }

    // predicted rest of the open prefill (ms): the open unit's remaining layers, then 60 layers of each later unit (groups
    // of up to two max_rows chunks, the engine's default pf_group)
    double predicted_rest_ms() const {
        const std::uint64_t unit = eng_.prefill_unit_rows();
        double ms = layer_.fixed_ms + static_cast<double>(60 - eng_.prefill_layer()) * layer_.upper_layer_ms(unit);
        for (std::uint64_t left = eng_.prefill_rows_left() - unit; left > 0;) {
            const std::uint64_t u = std::min<std::uint64_t>(left, 2 * max_rows_);
            ms += layer_.fixed_ms + 60.0 * layer_.upper_layer_ms(u);
            left -= u;
        }
        return ms;
    }
    // decode cycle cost (ms) with n decoding requests: the measured complete cycle, else the step-cost estimate; 0 for none
    double cycle_ms(std::uint32_t n) const {
        if (n == 0) return 0;
        double upper = costs_.at(0, n);
        for (unsigned r = 1; r <= kStepRows; ++r)
            upper = std::max(upper, cycle_cost_[r] * (1.0 + 0.12 * (n - 1)) / (1.0 + 0.12 * (r - 1)));
        return 1e3 * upper;
    }
    // longest time a decoding request has waited since its last token (ms)
    double longest_wait_ms() const {
        double w = 0;
        for (const auto& r : run_slots_)
            if (r && !r->prefilling) w = std::max(w, elapsed(r->last_emit));
        return w;
    }
    double live_gap_ms(double candidate) const {
        for (const auto& r : run_slots_)
            if (r) candidate = std::min(candidate, r->policy.gap_ms);
        return candidate;
    }
    double predicted_prompt_ms(std::uint64_t rows) const {
        double predicted = 0;
        while (rows > 0) {
            const auto u = std::min<std::uint64_t>(rows, 2 * max_rows_);
            predicted += layer_.fixed_ms + 60 * layer_.upper_layer_ms(u);
            rows -= u;
        }
        return predicted;
    }
    unsigned unit_layers() const {
        const auto& policy = run_slots_[prefill_slot_]->policy;
        if (run_slots_[prefill_slot_]->policy.sched == Sched::off) return ~0U;
        // Preserve the observed-gap goal rather than spending all of the default 1500 ms.
        return ms::unit_layers(policy.service_ms, decoding_count() > 0, std::min(live_gap_ms(policy.gap_ms), 1300.0),
            cycle_ms(decoding_count()), layer_.fixed_ms,
            layer_.upper_layer_ms(std::max<std::uint64_t>(1, eng_.prefill_unit_rows())), longest_wait_ms(), ms::control_margin_ms);
    }

    void prefill_unit(unsigned layers) {
        deadline_serviced_ = false;
        const auto u0 = timeline_on_ ? Time::now() : Clock::time_point{};
        if (*eng_.prefill_unlimited_reason() != 0) layers = ~0U;
        Control c;
        c.command = static_cast<std::uint32_t>(Command::unit);
        c.round = round_++;
        c.layers = layers;
        c.bank = eng_.prefill_progress().bank;
        c.task_id = run_slots_[prefill_slot_]->request_id;
        (void)exchange(eng_, c);
        const std::uint64_t unit_rows = eng_.prefill_unit_rows(), left0 = eng_.prefill_rows_left();
        const unsigned layer0 = eng_.prefill_layer();
        const auto summary = timeline_on_ ? timeline_prefill(prefill_slot_, layers == ~0U) : std::string{};
        const auto t0 = Time::now();
        std::uint32_t next = 0;
        const bool done = run_.prefill_unit(layers, &next);
        const auto returned = timeline_on_ ? Time::now() : Clock::time_point{};
        const double ms = elapsed(t0);
        if (parked_slot_ < slots_) {
            run_slots_[parked_slot_]->x.nest_service_ms += ms;
            nest_service_ms_ += ms;
        }
        // layer time of units that stayed inside one chunk / group (or ended the last one)
        if (!done && eng_.prefill_rows_left() == left0 && eng_.prefill_layer() > layer0) layer_.observe(unit_rows, eng_.prefill_layer() - layer0, ms);
        else if (done && left0 == unit_rows) layer_.observe(unit_rows, 60 - layer0, ms);
        else layer_.reset_pending();  // a resume spanning groups has no single-bucket sample
        if (done) layer_.reset_pending();
        auto& r = run_slots_[prefill_slot_];
        ++r->x.units;
        r->x.max_unit_layers = std::max(r->x.max_unit_layers,
            layers == ~0U || *eng_.prefill_unlimited_reason() != 0 ? ~0U : layers);
        // the decoding requests stalled for this unit
        double stalled = 0;
        for (auto& o : run_slots_)
            if (o && !o->prefilling) {
                o->x.stall_ms += ms;
                stalled += ms / 1e3;
            }
        add([&](Counters& k) {
            k.prefill_s += ms / 1e3;
            k.stall_s += stalled;
            ++k.units;
            k.max_unit_layers = std::max(k.max_unit_layers, r->x.max_unit_layers);
        });
        if (!r->x.nested) r->nest_quota.progressed = true;
        if (!done) {
            if (timeline_on_) timeline("unit", u0, layers == ~0U ? "whole" : "deadline", prefill_slot_, layer0, eng_.prefill_rows_left() != left0 ? 60 : eng_.prefill_layer(), summary, t0, returned);
            return;
        }
        if (r->x.nested) {
            Control check;
            check.command = static_cast<std::uint32_t>(Command::check);
            check.round = round_++;
            check.task_id = r->request_id;
            check.plen = next;
            const auto checked = timeline_on_ ? Time::now() : Clock::time_point{};
            const auto reply = exchange(eng_, check);
            if (timeline_on_) timeline("check", checked, "nested", prefill_slot_, 60, 60, timeline_task(eng_.prefill_progress()));
            if (reply.plen != next) {
                eng_.abort("nested completion token mismatch");
                throw std::runtime_error("nested completion token mismatch");
            }
        }
        r->prefilling = false;
        r->stats.prefill_ms = elapsed(r->prefill_start);
        r->x.prefill_wall_ms = r->stats.prefill_ms;
        r->pos = r->stats.prompt_tokens;
        r->last = next;
        {
            const auto l0 = Time::now();
            lookup_start(*r, next);
            const double ls = secs(l0, Time::now());
            add([&](Counters& k2) { k2.lookup_s += ls; });
        }
        r->decode_start = Time::now();
        if (!r->job.ticket->cancelled()) (void)emit(r, std::span(&next, 1));
        else finish(r, "cancelled");
        if (parked_slot_ < slots_) unpark();
        if (timeline_on_) timeline("unit", u0, layers == ~0U ? "whole" : "deadline", prefill_slot_, layer0, 60, summary, t0, returned);
    }

    // One decode cycle over every decoding request: a plain batch step, or DSpark drafts + verify windows within 8 rows.
    void cycle() {
        const auto op0 = Time::now();
        std::vector<std::uint32_t> order;
        for (std::uint32_t s = 0; s < slots_; ++s)
            if (run_slots_[s] && !run_slots_[s]->prefilling) {
                const auto action = ms::decode_action(run_slots_[s]->policy.sched == Sched::interleave && run_slots_[s]->job.ticket->cancelled(),
                                                       run_slots_[s]->pos + 1 >= capacity_);
                if (action != ms::DecodeAction::step) {
                    finish(run_slots_[s], action == ms::DecodeAction::cancel ? "cancelled" : "context");
                    continue;
                }
                order.push_back(s);
            }
        if (order.empty()) return;
        const auto n = static_cast<std::uint32_t>(order.size());
        // DSpark contexts: kept only while few requests are in flight (each append reads the 94 MB fc)
        const bool keep = run_.dd != nullptr && taps_ &&
            ((pol_.dspark_mode == ds::Mode::on && inflight() <= pol_.inflight) ||
             (pol_.dspark_mode == ds::Mode::automatic && inflight() == 1 && parked_slot_ >= slots_));
        std::uint32_t append = 0;
        for (auto s : order) {
            auto& r = run_slots_[s];
            if (!keep) {
                r->context = false;  // stale for good (acceptance only)
                r->x.context = false;
            }
            if (r->context) append |= 1U << s;
        }
        // drafting slots: contexts, room for the block, more than one token to go; the budget is the step's free rows
        const std::uint32_t budget = kStepRows - n;
        // DSpark drafts this cycle for a slot with a context, room for its block and tokens to go, while the gate is open
        // (asked once per cycle: the gate's draft() spends its backoff)
        const auto dspark_eligible = [&](std::uint32_t s) {
            const auto& r = run_slots_[s];
            return keep && r->context && r->pos + ds::block + 1 < capacity_ && r->stats.generated_tokens + 1 < r->job.job.max_tokens;
        };
        bool any_eligible = false;
        if (budget > 0)
            for (auto s : order) any_eligible = any_eligible || dspark_eligible(s);
        const bool gate_open = any_eligible && gates_[n].draft();
        // prompt lookup first: a slot drafts its window from its own context where the lower-bound rate (tokens per
        // priced cycle, rows only, no draft) beats its alternative (its measured DSpark windows while it may draft,
        // else a plain row) by the margin; DSpark then plans the other slots in the rows left
        std::map<std::uint32_t, std::array<std::uint32_t, ds::block>> lk_tok;
        std::map<std::uint32_t, std::uint32_t> lk_k;
        std::uint32_t lk_rows = 0;
        double lk_extra = 0;  // expected tokens beyond one per slot from the lookup windows chosen this cycle
        // with several requests in the step, lookup runs only in cycles where DSpark drafts no slot: their alternatives
        // are not comparable per slot (DSpark's draft-slot ranking and its batch yield are decided later)
        if (pol_.lookup && (n == 1 || !gate_open)) {
            const auto l0 = Time::now();
            const unsigned shape = n == 1 ? 1U : 2U;
            const double base = costs_.at(0, n);
            // a cycle of `rows` rows: its measured cost, else at least the nearest measured smaller cycle plus ~0.3 of a
            // one-row step per extra row (measurement: N4 84.8 vs N1 45.3 ms; the table's own extrapolation assumes 0.12)
            const double row = kLookupRowCost * costs_.at(0, 1);
            const auto cycle = [&](std::uint32_t rows) {
                const double c = costs_.at(shape, rows);
                if (costs_.measured(shape, rows)) return c;
                double ref = base;  // the nearest measured smaller cycle of this shape (else the plain one) plus a row each
                std::uint32_t ref_rows = n;
                for (std::uint32_t q = rows - 1; q > n; --q)
                    if (costs_.measured(shape, q)) {
                        ref = costs_.at(shape, q);
                        ref_rows = q;
                        break;
                    }
                return std::max(c, ref + row * (rows - ref_rows));
            };
            for (auto s : order) {
                auto& r = run_slots_[s];
                if (!r->lk || r->lk_next != r->pos + 1) continue;  // not current
                const auto m = r->lk->longest({}, kLookupMin, ds::block);
                if (m.length < kLookupMin || m.available == 0) continue;
                const std::uint32_t b = m.length < 4U ? 1U : m.length < 8U ? 2U : m.length < 16U ? 3U : 4U;
                for (auto& x : r->lk_eligible[b]) x *= kLookupDecay;
                for (auto& x : r->lk_accepted[b]) x *= kLookupDecay;
                const bool enough = r->lk_evidence[b] >= kLookupEvidence;
                r->lk_evidence[b] = r->lk_evidence[b] * kLookupDecay + 1.0;
                if (r->lk_pending.size() < kLookupPending) {
                    Running::LookupPending p;
                    p.first = r->pos + 1;
                    p.count = std::min<std::uint32_t>(m.available, ds::block);
                    p.bucket = b;
                    for (std::uint32_t j = 0; j < p.count; ++j) p.token[j] = m.next[j];
                    r->lk_pending.push_back(p);
                }
                if (!enough || budget <= lk_rows) continue;
                const bool dspark_may = gate_open && dspark_eligible(s);  // DSpark would draft it this cycle
                if (dspark_may && r->ds_n < 3.0) continue;  // the alternative is measured first
                const double ds_rate = r->ds_cost > 0 ? r->ds_tokens / r->ds_cost : 0.0;
                // tokens per second of the whole cycle: every decoding slot pays for the window's rows
                const double now_rate = (n + lk_extra) / (lk_rows != 0 ? cycle(n + lk_rows) : base);
                const double alternative = std::max(now_rate, dspark_may ? ds_rate : 0.0) * lookup_margin_;
                const std::uint64_t room = std::min<std::uint64_t>(r->job.job.max_tokens - r->stats.generated_tokens - 1, capacity_ - r->pos - 1);
                const auto depth = static_cast<std::uint32_t>(
                    std::min<std::uint64_t>({m.available, ds::block, room, budget - lk_rows}));
                double best = alternative, survival = 1.0, yield = 1.0, best_yield = 1.0;
                std::uint32_t cap = 0;
                for (std::uint32_t j = 0; j < depth; ++j) {
                    // Beta(1,2) minus one posterior standard deviation per depth (the Nemotron / DeepSeek selector)
                    const double cnt = r->lk_eligible[b][j] + 3.0, q = (r->lk_accepted[b][j] + 1.0) / cnt;
                    survival *= std::max(0.0, q - std::sqrt(q * (1.0 - q) / (cnt + 1.0)));
                    yield += survival;
                    const double rate = (n + lk_extra + yield - 1.0) / cycle(n + lk_rows + j + 1);
                    if (rate > best) best = rate, cap = j + 1, best_yield = yield;
                }
                if (cap == 0) continue;
                auto& t = lk_tok[s];
                for (std::uint32_t j = 0; j < cap; ++j) t[j] = m.next[j];
                lk_k[s] = cap;
                lk_rows += cap;
                lk_extra += best_yield - 1.0;
            }
            const double ls = secs(l0, Time::now());
            add([&](Counters& k2) { k2.lookup_s += ls; });
        }
        std::vector<std::uint32_t> cand;
        if (keep && budget > lk_rows)
            for (auto s : order)
                if (lk_k.count(s) == 0 && dspark_eligible(s)) cand.push_back(s);
        bool drafted = false;
        std::map<std::uint32_t, Drafts> drafts;
        const auto c0 = Time::now();
        double t_draft = 0;
        if (!cand.empty() && gate_open) {
            std::sort(cand.begin(), cand.end(), [&](auto a, auto b) { return run_slots_[a]->tau > run_slots_[b]->tau; });
            cand.resize(std::min<std::size_t>({cand.size(), static_cast<std::size_t>(budget - lk_rows), pol_.draft_slots}));
            Control c;
            c.command = static_cast<std::uint32_t>(Command::draft);
            c.round = round_++;
            c.dslots = static_cast<std::uint32_t>(cand.size());
            for (std::size_t i = 0; i < cand.size(); ++i) {
                c.dslot[i] = cand[i];
                c.danchor[i] = run_slots_[cand[i]]->last;
                c.dpos[i] = run_slots_[cand[i]]->pos;
            }
            (void)exchange(eng_, c);
            for (auto s : cand) drafts[s] = run_.draft(s, run_slots_[s]->last, run_slots_[s]->pos);
            drafted = true;
            t_draft = secs(c0, Time::now());
            Costs::ema(costs_.draft, t_draft / static_cast<double>(cand.size()));
        }
        // windows: every feasible total of draft rows j = 0..budget from the per-slot survival increments (each slot's
        // increments do not increase with depth, so the j best increments are the best j-row allocation); the j with the
        // most expected tokens per cycle cost, lowest on ties; every 16th drafting cycle per n takes the largest
        std::map<std::uint32_t, std::uint32_t> k;
        if (drafted) {
            struct Inc {
                double gain;
                std::uint32_t slot, depth;
            };
            std::vector<Inc> inc;
            for (auto& [s, d] : drafts) {
                const auto& r = run_slots_[s];
                const std::uint64_t room = std::min<std::uint64_t>(r->job.job.max_tokens - r->stats.generated_tokens - 1,
                                                                   capacity_ - r->pos - 1);
                double surv = 1.0;
                for (std::uint32_t i = 0; i < ds::block && i < room; ++i) {
                    surv *= r->survival.probability(i, d.confidence[i]);
                    inc.push_back({surv, s, i});
                }
            }
            std::stable_sort(inc.begin(), inc.end(), [](const Inc& a, const Inc& b) { return a.gain > b.gain; });
            const std::size_t jmax = std::min<std::size_t>(inc.size(), budget - lk_rows);
            const unsigned shape = n == 1 ? 1U : 2U;
            std::size_t best = 0;
            double best_rate = (static_cast<double>(n) + lk_extra) / costs_.at(lk_rows != 0 ? shape : 0U, n + lk_rows), g = 0;
            for (std::size_t j = 1; j <= jmax; ++j) {
                g += inc[j - 1].gain;
                const double cost = costs_.draft * static_cast<double>(drafts.size()) + costs_.at(shape, n + lk_rows + static_cast<std::uint32_t>(j));
                const double rate = (static_cast<double>(n) + lk_extra + g) / cost;
                if (rate > best_rate) {
                    best_rate = rate;
                    best = j;
                }
            }
            if (++explore_[n] % 16 == 1) best = jmax;
            for (std::size_t j = 0; j < best; ++j) ++k[inc[j].slot];
        }
        // the step rows: every decoding slot's anchor, then its window's drafts
        Control c;
        c.command = static_cast<std::uint32_t>(Command::step);
        c.round = round_++;
        c.append = append;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> span_of;  // slot -> (first row, rows)
        for (auto s : order) {
            auto& r = run_slots_[s];
            const std::uint32_t first = c.rows;
            c.row[c.rows++] = mm::Row{s, r->last, r->pos};
            if (lk_k.count(s) != 0) {  // a lookup window
                for (std::uint32_t i = 0; i < lk_k[s]; ++i) c.row[c.rows++] = mm::Row{s, lk_tok[s][i], r->pos + 1 + i};
            } else {
                const std::uint32_t ks = k.count(s) ? k[s] : 0U;
                for (std::uint32_t i = 0; i < ks; ++i) c.row[c.rows++] = mm::Row{s, drafts[s].tokens[i], r->pos + 1 + i};
            }
            span_of.push_back({first, c.rows - first});
        }
        const auto summary = timeline_on_ ? timeline_step(c) : std::string{};
        const auto t0 = Time::now();
        (void)exchange(eng_, c);
        const auto control_end = timeline_on_ ? Time::now() : Clock::time_point{};
        const auto next = run_.step(std::span(c.row, c.rows), append);
        if (append != 0) run_.synchronize();  // the context appends are part of this cycle's work
        const auto t_done = Time::now();
        const double t_step = secs(t0, t_done);  // control exchange + step + appends, completed
        const bool windows = c.rows > n;
        costs_.observe(windows ? (n == 1 ? 1U : 2U) : 0U, c.rows, t_step);
        std::uint32_t emitted_total = 0, accepted_total = 0, drafted_total = 0, windows_n = 0, rejected_n = 0;
        std::uint32_t lk_windows = 0, lk_drafted = 0, lk_accepted = 0;
        double lk_commit_s = 0;
        std::uint32_t k_min = ~0U, k_max = 0;
        for (std::size_t i = 0; i < order.size(); ++i) {
            const std::uint32_t s = order[i];
            auto& r = run_slots_[s];
            const auto [first, rows] = span_of[i];
            std::vector<std::uint32_t> d;
            for (std::uint32_t j = 1; j < rows; ++j) d.push_back(c.row[first + j].token);
            const auto v = ds::verify_greedy(d, std::span<const std::uint32_t>(next.data() + first, rows));
            if (drafted && drafts.count(s)) {
                const std::uint32_t ks = rows - 1;
                r->survival.observe(std::span<const float>(drafts[s].confidence.data(), ks), ks, v.accepted);
                r->x.drafted += ks;
                r->x.accepted += v.accepted;
                Costs::ema(r->tau, static_cast<double>(v.emitted.size()));
                drafted_total += ks;
                accepted_total += v.accepted;
                if (ks > 0) {
                    ++windows_n;
                    k_min = std::min(k_min, ks);
                    k_max = std::max(k_max, ks);
                    rejected_n += v.accepted < ks ? 1U : 0U;
                    r->ds_tokens = r->ds_tokens * kLookupDecay + static_cast<double>(v.emitted.size());
                    r->ds_cost = r->ds_cost * kLookupDecay + t_draft / static_cast<double>(drafts.size()) + t_step;
                    r->ds_n = r->ds_n * kLookupDecay + 1.0;
                }
            }
            if (lk_k.count(s) != 0) {
                ++lk_windows;
                lk_drafted += rows - 1;
                lk_accepted += v.accepted;
            }
            if (r->lk) {
                const auto l0 = Time::now();
                for (std::size_t e = 0; e < v.emitted.size(); ++e) lookup_commit(*r, v.emitted[e], r->pos + 1 + e);
                lk_commit_s += secs(l0, Time::now());
            }
            ++r->x.cycles;
            r->pos += v.accepted + 1;
            r->last = v.emitted.back();
            (void)emit(r, v.emitted, &emitted_total);  // the tokens actually emitted (a stop / limit drops the rest)
        }
        // the gate on one boundary for both kinds of cycle: from the cycle's start (before any draft) to its completed
        // step + appends; plain cycles as seconds per token of the n rows
        const double t_cycle = secs(c0, t_done);
        const double complete = secs(op0, Time::now());
        cycle_cost_[n] = std::max(complete, cycle_cost_[n] * 0.98);
        if (drafted) gates_[n].observe(true, emitted_total, t_cycle, costs_.at(0, n) / n);
        else if (lk_rows == 0 && (!cand.empty() || budget > 0)) gates_[n].observe(false, n, t_cycle / n, 0.0);
        add([&](Counters& k2) {
            k2.step_s += t_step;
            k2.draft_s += t_draft;
            ++k2.steps;
            k2.step_rows += c.rows;
            ++k2.rows_hist[c.rows];
            if (windows_n >= 2) ++k2.multi_windows;
            if (windows_n >= 2 && k_min != k_max) ++k2.unequal_windows;
            k2.rejections += rejected_n;
            k2.lookup_windows += lk_windows;
            k2.lookup_drafted += lk_drafted;
            k2.lookup_accepted += lk_accepted;
            k2.lookup_s += lk_commit_s;
            if (drafted) {
                ++k2.draft_cycles;
                k2.drafts += drafts.size();
                k2.drafted += drafted_total;
                k2.accepted += accepted_total;
            } else {
                ++k2.plain_cycles;
            }
        });
        deadline_serviced_ = true;
        cycle_cost_[n] = std::max(cycle_cost_[n], secs(op0, Time::now()));
        if (timeline_on_) timeline("step", op0, step_reason_, 0, 0, 60, summary, control_end, t_done);
        step_reason_ = "deadline";
    }

    // Only the enabled branch reads clocks or builds/queues records. No device synchronization.
    std::string timeline_task(const mm::PfProgress& p, std::uint64_t child = 0) const {
        return "\"task\":" + std::to_string(p.id) + ",\"parent_task\":" + std::to_string(p.parent) +
            ",\"child_task\":" + std::to_string(child) + ",\"bank\":" + std::to_string(p.bank) +
            ",\"slot_generation\":" + std::to_string(p.generation) + ",\"rows\":" + std::to_string(p.rows) +
            ",\"checkpoint\":" + std::to_string(p.layer) + ",\"predicted_ms\":" + std::to_string(predicted_prompt_ms(p.rows));
    }
    std::string timeline_prefill(std::uint32_t slot, bool whole = false) const {
        const auto& r = *run_slots_[slot];
        auto left = whole ? eng_.prefill_rows_left() : eng_.prefill_unit_rows();
        std::string out = "\"rows\":" + std::to_string(left) + ",\"members\":[";
        for (bool first = true; left > 0; first = false) {
            const auto rows = std::min(left, max_rows_);
            if (!first) out += ",";
            out += std::to_string(rows);
            left -= rows;
        }
        out += "],\"slots\":[{\"slot\":" + std::to_string(slot) + ",\"request\":" + std::to_string(r.request_id) +
               ",\"pos\":" + std::to_string(r.stats.prompt_tokens - eng_.prefill_rows_left()) + ",\"token_index\":0}]";
        return out;
    }
    std::string timeline_step(const Control& c) const {
        std::string out = "\"rows\":" + std::to_string(c.rows) + ",\"slots\":[";
        for (unsigned i = 0; i < c.rows; ++i) {
            const auto& r = *run_slots_[c.row[i].slot];
            if (i > 0) out += ",";
            out += "{\"slot\":" + std::to_string(c.row[i].slot) + ",\"request\":" + std::to_string(r.request_id) +
                   ",\"pos\":" + std::to_string(c.row[i].pos) + ",\"token_index\":" + std::to_string(r.stats.generated_tokens + c.row[i].pos - r.pos) + "}";
        }
        return out + "]";
    }
    void timeline(const char* kind, Clock::time_point start, const char* reason, std::uint32_t slot = 0,
                  unsigned layer0 = 0, unsigned layer1 = 0, const std::string& summary = {},
                  Clock::time_point control_end = {}, Clock::time_point run_end = {}) {
        if (!timeline_on_) return;
        const auto end = Time::now();
        const auto op = timeline_op_++;
        if (timeline_records_.size() == 65536) { ++timeline_dropped_; return; }
        const auto us = [&](Clock::time_point t) { return std::chrono::duration_cast<std::chrono::microseconds>(t - timeline_origin_).count(); };
        if (control_end == Clock::time_point{}) control_end = end;
        if (run_end == Clock::time_point{}) run_end = end;
        const bool prefill = std::strcmp(kind, "admit") == 0 || std::strcmp(kind, "unit") == 0;
        std::string row = summary.empty() ? prefill ? timeline_prefill(slot) : "\"rows\":0,\"slots\":[]" : summary;
        timeline_records_.push_back("{\"event\":\"mmserve-timeline\",\"op\":" + std::to_string(op) + ",\"epoch\":" + std::to_string(pol_.epoch) +
            ",\"kind\":\"" + kind + "\",\"t_start_us\":" + std::to_string(us(start)) + ",\"t_end_us\":" + std::to_string(us(end)) +
            ",\"control_end_us\":" + std::to_string(us(control_end)) + ",\"run_end_us\":" + std::to_string(us(run_end)) +
            ",\"round\":" + std::to_string(round_ - 1) + ",\"layer0\":" + std::to_string(layer0) + ",\"layer1\":" + std::to_string(layer1) + ",\"reason\":\"" + reason + "\",\"age_ms\":" +
            std::to_string(longest_wait_ms()) + ",\"slack_ms\":" + std::to_string(ms::remaining_ms(live_gap_ms(pol_.gap_ms), cycle_ms(decoding_count()), longest_wait_ms())) + "," + row + "}");
    }
    void flush_timeline() {
        if (!timeline_on_) return;
        for (const auto& record : timeline_records_) std::fprintf(stderr, "%s\n", record.c_str());
        timeline_records_.clear();
        if (timeline_dropped_ > 0) {
            std::fprintf(stderr, "{\"event\":\"mmserve-timeline-drops\",\"count\":%llu}\n", static_cast<unsigned long long>(timeline_dropped_));
            timeline_dropped_ = 0;
        }
    }

    sv::Json dispatch_session(const std::string& body, const ms::SessionService::Publish& publish = {}) {
        Control c; c.command = static_cast<std::uint32_t>(Command::session); c.plen = static_cast<std::uint32_t>(body.size());
        c.round = round_++;
        (void)exchange(eng_, c);
        const auto wire = ms::session_wire(eng_, body, body.size());
        const auto available = eng_.peer(session_ && session_->execute ? 1U : 0U);
        auto reply = available ? session_->execute(wire, publish) : sv::session_error("session_budget", "peer session precondition failed", 507);
        eng_.set_taps(taps_);
        return reply;
    }

    std::function<std::string(const std::vector<std::uint32_t>&)> prompt_bytes_;
    Engine& eng_;
    Runner& run_;
    sv::GenerationQueue& queue_;
    Shared& sh_;
    const std::uint32_t slots_;
    const std::uint64_t capacity_;
    const std::uint64_t max_rows_;
    std::vector<std::optional<Running>> run_slots_;
    std::optional<sv::TakenJob> pending_;
    bool pending_short_first_ = false;
    bool deadline_serviced_ = false;
    const char* step_reason_ = "deadline";
    bool timeline_on_ = false;
    Clock::time_point timeline_origin_;
    std::uint64_t timeline_op_ = 0, timeline_dropped_ = 0, request_id_ = 0;
    std::vector<std::string> timeline_records_;
    std::uint64_t round_ = 0;
    std::uint32_t prefill_slot_ = 0;
    std::array<std::uint64_t, kStepRows> slot_generation_{};
    std::uint32_t parked_slot_ = slots_;
    Clock::time_point park_start_;
    Clock::time_point nest_start_;
    double nest_switch_upper_ms_ = 20, nest_service_ms_ = 0;
    bool nested_needs_decode_ = false;
    Policy pol_;
    Costs costs_;
    // prompt lookup (the Nemotron / DeepSeek selector's constants): shortest match, pending candidates per request,
    // evidence decay, candidates a bucket needs
    static constexpr std::uint32_t kLookupMin = 2, kLookupPending = 16;
    static constexpr double kLookupDecay = 0.98, kLookupEvidence = 4.0, kLookupRowCost = 0.3;
    double lookup_margin_ = 1.0;
    std::uint32_t lookup_tokens_ = 1U << 16;
    // the request's index: its prompt (last lookup_tokens_ tokens), then the first generated token at the prompt's end
    void lookup_start(Running& r, std::uint32_t first) {
        if (!pol_.lookup) return;
        r.lk = strix::lookup::make_ngram_index({lookup_tokens_, 64});
        const auto& prompt = r.job.job.prompt;
        const std::size_t from = prompt.size() > lookup_tokens_ ? prompt.size() - lookup_tokens_ : 0;
        for (std::size_t i = from; i < prompt.size(); ++i) r.lk->append(prompt[i]);
        r.lk_next = prompt.size();
        lookup_commit(r, first, prompt.size());
    }
    // the token the request committed at stream `position`: scores the pending candidates waiting for it, then joins the
    // index; a position out of sequence ends lookup for the request
    void lookup_commit(Running& r, std::uint32_t token, std::uint64_t position) {
        if (!r.lk) return;
        if (position != r.lk_next) {
            r.lk.reset();
            r.lk_pending.clear();
            return;
        }
        ++r.lk_next;
        for (auto it = r.lk_pending.begin(); it != r.lk_pending.end();) {
            auto& c = *it;
            if (c.first + c.done != position) {
                ++it;
                continue;
            }
            r.lk_eligible[c.bucket][c.done] += 1.0;
            const bool hit = c.token[c.done] == token;
            if (hit) r.lk_accepted[c.bucket][c.done] += 1.0;
            ++c.done;
            if (!hit || c.done == c.count) it = r.lk_pending.erase(it);
            else ++it;
        }
        r.lk->append(token);
    }
    std::array<double, kStepRows + 1> cycle_cost_{};
    ms::LayerModel layer_;
    std::atomic<bool>& stop_;
    ms::SessionService* session_ = nullptr;
    bool taps_ = false;  // the engine's taps (set to the startup policy before READY, then sync_taps)
    std::vector<ds::DraftGate> gates_ = std::vector<ds::DraftGate>(kStepRows + 1);  // one per decoding count
    std::array<std::uint64_t, kStepRows + 1> explore_{};
};

} // namespace strix::minimax::serve::lead
