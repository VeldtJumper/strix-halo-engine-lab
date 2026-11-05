#pragma once
// GLM TP2 server: the lead loop (rank 0) and the follower (rank 1), host-only templates over the engine calls
// (Eng: reset(slot), prefill(slot, tokens, pos0) -> next, step(span<const WireRow>) -> tokens), so the scheduling and
// the protocol run against a fake engine in the CPU tests.
//
// Lead (rank 0): a slot table of up to `slots` requests. Each iteration runs ONE decode step with one row {slot, last
// token, position} per decoding request (rows in ascending slot order), then one chunk of the prompt in prefill (a
// queued request is admitted FIFO into the lowest free slot when no prompt is in prefill; --prefill-chunk 0 prefills
// it whole, stalling decode for the whole prompt). Tokens are emitted per request (stop ids, max_tokens, context capacity, cancellation); a
// finished request frees its slot at once. Serial mode (the OFF switch) admits only when nothing runs.
// Exactness: Engine::step is batch invariant per row and prefill is per slot, so each request's tokens equal its run
// alone; the self-test checks it.
#include "glm/serve/glm_serve_control.hpp"
#include "glm/spec/glm_spec_host.hpp"
#include "glm/serve/glm_serve_import.hpp"
#include "strix/serve/engine.hpp"
#include "strix/serve/glm_chat.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace strix::glm::serve {

// GLM-5 generation_config.json stops: <|endoftext|>, <|user|>, <|observation|>.
inline const std::vector<std::uint32_t>& default_stop_ids() {
    static const std::vector<std::uint32_t> ids(strix::serve::glm_stop_tokens.begin(), strix::serve::glm_stop_tokens.end());
    return ids;
}

enum class Next { decode, stop, length, context };
inline const char* finish_name(Next n) {
    switch (n) {
        case Next::stop: return "stop";
        case Next::length: return "length";
        case Next::context: return "context";
        case Next::decode: break;
    }
    return "length";
}
// After token `t` (the request's `generated`-th) was produced: its row would be at `next_pos` in the next step.
inline Next decide(std::uint32_t t, std::uint32_t generated, std::uint32_t max_tokens, std::uint64_t next_pos,
                   std::uint64_t capacity, std::span<const std::uint32_t> stops) {
    if (std::find(stops.begin(), stops.end(), t) != stops.end()) return Next::stop;
    if (generated >= max_tokens) return Next::length;
    if (next_pos >= capacity) return Next::context;
    return Next::decode;
}

// Lowest-free slot allocation keeps live slots compact while requests finish in order; a request finishing ahead of a
// higher slot leaves a gap until the next admission fills it (counted: gap steps take the per-row KDA path).
class SlotTable {
public:
    explicit SlotTable(std::uint32_t slots) : used_(slots, false) {}
    std::optional<std::uint32_t> alloc() {
        for (std::uint32_t s = 0; s < used_.size(); ++s)
            if (!used_[s]) {
                used_[s] = true;
                return s;
            }
        return std::nullopt;
    }
    void release(std::uint32_t s) { used_.at(s) = false; }
    [[nodiscard]] bool used(std::uint32_t s) const { return used_.at(s); }
    [[nodiscard]] std::uint32_t live() const { return static_cast<std::uint32_t>(std::count(used_.begin(), used_.end(), true)); }
    [[nodiscard]] bool full() const { return live() == used_.size(); }
    [[nodiscard]] std::uint32_t size() const { return static_cast<std::uint32_t>(used_.size()); }
private:
    std::vector<bool> used_;
};

struct LeadConfig {
    std::uint32_t slots = 8;
    std::uint64_t capacity = 8192;
    std::uint64_t prefix_cache_bytes = 0, prefix_identity = 0;
    bool batch = true;               // false: serial admission (one request at a time)
    double heartbeat_seconds = 1.0;  // idle: one heartbeat round per interval keeps rank 1 and the link alive
    std::vector<std::uint32_t> stops = default_stop_ids();
    // C1 MTP self-speculation: with exactly one request in flight (and none waiting in batched mode) it decodes through
    // the engine's single-slot verify loop, `mtp_cycles` verify cycles per control round; prompts are prefilled with
    // the MTP bootstrap so any request can take the path. Off: plain step() at every concurrency.
    bool mtp = false;
    // Import gate: right after each prompt's prefill (or import) both ranks digest the slot's state (state_digest);
    // Lead::take_digests() returns them per request.
    bool state_digest = false;
    unsigned mtp_cycles = 1;
    // Decode-first prefill interleave: while at least one request decodes, a new prompt is prefilled in chunks of
    // `prefill_chunk` tokens (Engine::prefill at pos0 > 0), one chunk per iteration after the decode step; with nothing
    // decoding the whole remaining prompt goes in one call (small calls are slower). 0: always the whole prompt.
    unsigned prefill_chunk = 512;
    bool prefill_chunk_mtp = true;   // false: MTP-bootstrap prompts are prefilled whole (only plain prompts chunk)
    // Batched self-speculation (G2): with MTP on and more than one request decoding (or one while a prompt is in
    // prefill), every decoding slot verifies a window (anchor + adaptive drafts) in ONE verify call of <= 8 rows; window
    // sizes from rank 0's cost table (expected tokens per cycle cost, the MiniMax rule). Off: batched plain steps.
    // Handover: with more decoding slots than leave room for drafts (2 x slots > 8 rows), or when the window sizes come
    // out without any draft, the cycle is a plain step (no drafter calls); the drafters are re-seeded (one anchor-only
    // window cycle with catch-ups) when the slot count changes or after `g2_retry` plain steps.
    bool g2 = false;
    unsigned kmax = 3;
    unsigned g2_retry = 16;
    // RTX prefill route (off when unset): a prompt of >= import_min_tokens is announced to both ranks (import_expect)
    // and handed to rtx_prefill, which asks the RTX producer to send each rank its state (false = not reachable). One
    // such prompt waits at a time, beside the prefill place (other prompts are admitted and prefilled, other slots keep
    // decoding); it skips the chunked prefill: once both ranks report it staged, import_state restores the slot and
    // runs the last prompt token. On a failure or after import_timeout_s it is prefilled in chunks as usual.
    std::function<bool(std::span<const std::uint32_t> prompt, std::uint32_t id)> rtx_prefill;
    // Optional: true once the RTX request of `id` failed (its sender gave up), so the slot falls back at once.
    std::function<bool(std::uint32_t id)> rtx_failed;
    double import_listen_grace_s = 0.0;  // both ranks report their endpoint bound -> this long more before the RTX call
    std::uint64_t import_min_tokens = 4096;
    double import_timeout_s = 120.0;
    // 'DSA once' hand-off (Begin.version 2; both ranks' receivers set alike): import_state relays rank 0's DSA components
    // to rank 1 over the pair link before the restore.
    bool import_dsa_once = false;
};


// Self-test trace (Lead::set_trace): what happened when, per request (key = its ticket). first: the request's prefill
// returned its first token; token: one emitted token (the first included); finish: the request ended; step: one plain
// step (rows, slots not consecutive); mtp: one MTP round (tokens); prefill: one prefill call [start, t] (rows = tokens).
struct TraceEvent {
    enum Kind : std::uint8_t { first, token, finish, step, mtp, prefill };
    std::chrono::steady_clock::time_point t;
    const void* key = nullptr;
    Kind kind = token;
    std::uint32_t rows = 0;
    bool gap = false;
    std::chrono::steady_clock::time_point start{};
};

struct LeadStats {
    std::uint64_t steps = 0, rows = 0, prefills = 0, prefill_tokens = 0, prefill_chunks = 0, gap_steps = 0, rounds = 0, heartbeats = 0;
    std::uint64_t prefix_hits = 0, prefix_misses = 0, prefix_stores = 0, prefix_evictions = 0, prefix_bytes = 0;
    std::uint64_t finished = 0, active = 0;
    std::uint64_t rows_hist[kMaxRows + 1]{};  // steps by row count
    double step_seconds = 0, prefill_seconds = 0;
    std::uint64_t mtp_rounds = 0, mtp_tokens = 0, mtp_enters = 0, mtp_leaves = 0, mtp_bootstraps = 0;
    double mtp_seconds = 0;
    std::uint64_t g2_cycles = 0, g2_rows = 0, g2_windows = 0, g2_drafts = 0, g2_accepted = 0, g2_tokens = 0, g2_draft_calls = 0;
    std::uint64_t g2_plain = 0, g2_reseeds = 0;  // plain steps chosen by the G2 policy (handover), re-seeding cycles
    double g2_seconds = 0;
    std::uint64_t imports = 0, import_fallbacks = 0;  // slots restored from the RTX / prefilled after a failed import
    double import_wait_seconds = 0;                   // announce -> both ranks staged (imported prompts)
    std::uint64_t imported_tokens = 0;                // prompt tokens whose state came from the RTX
};

// The follower (rank 1): hello, then pure replay until shutdown.
template<class Eng>
void follow(Eng& eng, Channel& ch, const Hello& hello, std::uint32_t slots, std::uint64_t capacity) {
    ch.hello(hello);
    std::optional<PrefixCache> cache;
    if (hello.prefix_cache_bytes) cache.emplace(hello.prefix_cache_bytes);
    for (;;) {
        Control reply;
        reply.cmd = static_cast<std::uint32_t>(Cmd::reply);
        if constexpr (requires { eng.import_report(); }) {  // this rank's staging status of the pending import
            const ImportReport r = eng.import_report();
            reply.import_id = r.id;
            reply.import_status = r.status;
            reply.import_token = r.token;
        }
        const Control c = ch.exchange(reply);
        if (static_cast<Cmd>(c.cmd) == Cmd::shutdown) return;
        std::vector<std::uint32_t> prompt;
        (void)replay(eng, ch, c, prompt, false, slots, capacity, cache ? &*cache : nullptr);
    }
}

template<class Eng>
class Lead {
public:
    using Clock = std::chrono::steady_clock;
    Lead(Eng& eng, Channel& ch, strix::serve::GenerationQueue& queue, LeadConfig cfg, std::atomic<bool>& stop)
        : eng_(eng), ch_(ch), queue_(queue), cfg_(std::move(cfg)), stop_(stop), table_(cfg_.slots), active_(cfg_.slots),
          mtp_on_(cfg_.mtp), chunk_(cfg_.prefill_chunk), g2_on_(cfg_.g2 && cfg_.mtp), calib_(cfg_.slots) {
        if (cfg_.prefix_cache_bytes) prefix_cache_.emplace(cfg_.prefix_cache_bytes);
        for (unsigned n = 0; n <= kMaxRows; ++n)
            for (unsigned r = 0; r <= kMaxRows; ++r) cost_[n][r] = 30.0 + 11.0 * r + 2.5 * n;  // ms, measured EMA from here
    }

    // Runs until `stop` (then cancels what runs) or the queue shuts down with nothing running; sends shutdown last.
    // On an engine / protocol failure every running request finishes with the error, then it rethrows (no shutdown
    // round: the peer's state is unknown).
    void loop() {
        try {
            run();
        } catch (const std::exception& e) {
            failing_ = true;
            fail_all(std::string("engine failure: ") + e.what());
            throw;
        }
        fail_all("server shutting down");
        Control bye;
        bye.cmd = static_cast<std::uint32_t>(Cmd::shutdown);
        (void)ch_.exchange(bye);
        count_round();
    }

    [[nodiscard]] LeadStats stats() const {
        std::lock_guard<std::mutex> l(m_);
        return stats_;
    }
    // Decode rounds so far (plain steps + MTP rounds), for the self-test's staggered arrivals.
    [[nodiscard]] std::uint64_t progress() const {
        std::lock_guard<std::mutex> l(m_);
        return stats_.steps + stats_.mtp_rounds;
    }

    // Runtime switch of the C1 MTP path (LeadConfig::mtp; any thread). Prompts admitted while it is off are prefilled
    // plainly and never take the path (the engine refuses a drafter on them).
    void set_mtp(bool on) { mtp_on_.store(on && cfg_.mtp); }
    // Batched self-speculation on/off (LeadConfig::g2; any thread; takes effect at the next cycle).
    void set_g2(bool on) { g2_on_.store(on && cfg_.g2 && cfg_.mtp); }
    // Selftest driver: clear on the engine thread before admitting its next request.
    void clear_prefix_cache() {
        prefix_clear_.store(true);
        while (prefix_clear_.load() && !stop_.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // RTX route for prompts admitted from now on (any thread; the gate's own-prefill arm turns it off).
    void set_route(bool on) { route_on_.store(on); }
    // Import gate digests (LeadConfig::state_digest): per request (key = its ticket), rank 0's then rank 1's words.
    struct StateDigest {
        const void* key = nullptr;
        std::uint64_t frontier = 0;
        std::vector<std::uint32_t> rank0, rank1;
    };
    std::vector<StateDigest> take_digests() {
        std::lock_guard<std::mutex> l(m_);
        std::vector<StateDigest> out;
        out.swap(digests_);
        return out;
    }
    // Prefill chunk for prompts admitted from now on (any thread; 0 = whole prompt).
    void set_prefill_chunk(unsigned n) { chunk_.store(n); }
    // Self-test trace on/off (any thread); take_trace() returns the events since the last call.
    void set_trace(bool on) { trace_on_.store(on); }
    std::vector<TraceEvent> take_trace() {
        std::lock_guard<std::mutex> l(m_);
        std::vector<TraceEvent> out;
        out.swap(trace_);
        return out;
    }

private:
    struct Active {
        strix::serve::TakenJob job;
        std::uint32_t last = 0, generated = 0;
        std::uint64_t pos = 0;
        std::vector<std::uint32_t> stops;
        Clock::time_point t_admit, t_first;
        double prefill_ms = 0, restore_ms = 0;
        std::uint64_t hit_tokens = 0;
        std::optional<double> first_token_ms;
        bool mtp_ok = false;  // prefilled with the MTP bootstrap
        std::uint64_t imported = 0;  // prompt tokens restored from the RTX state (GenerationStats::imported_tokens)
        std::string route;           // RTX route receipt (JSON object; empty: the route was not tried)
    };
    // The prompt being prefilled (at most one): its slot is allocated, `done` tokens are in the engine.
    struct Prefilling {
        std::uint32_t slot = 0;
        std::size_t done = 0;
        unsigned chunk = 0;  // fixed at admission
        Active a;
        // RTX route: the import this prompt waits for (0: none, or fallen back to the chunked prefill)
        std::uint32_t import_id = 0;
        Clock::time_point import_t0, import_t_req;
        bool import_requested = false;  // the RTX was asked to send
    };

    void run() {
        auto last_round = Clock::now();
        for (;;) {
            if (stop_.load()) return;
            if (prefix_clear_.exchange(false) && cfg_.prefix_cache_bytes) {
                Control c;
                c.cmd = static_cast<std::uint32_t>(Cmd::prefix_clear);
                (void)control_round(c);
                std::lock_guard<std::mutex> l(m_);
                stats_.prefix_bytes = 0;
            }
            // cancellations between steps
            for (std::uint32_t s = 0; s < cfg_.slots; ++s)
                if (active_[s] && active_[s]->job.ticket->cancelled()) finish(s, "cancelled");
            if (pf_ && pf_->a.job.ticket->cancelled()) drop_prefill(pf_);
            if (imp_ && imp_->a.job.ticket->cancelled()) {
                if (imp_->import_id != 0) drop_import(imp_->import_id);  // both receivers stop at once
                drop_prefill(imp_);
            }
            // a fallen-back import takes the prefill place once it is free
            if (imp_ && imp_->import_id == 0 && !pf_) {
                pf_ = std::move(imp_);
                imp_.reset();
            }
            // admission: one prompt in prefill at a time (+ at most one waiting for its RTX state)
            if (!pf_ && !table_.full() && (cfg_.batch || table_.live() == 0)) {
                if (auto j = queue_.take(false)) {
                    admit(std::move(*j));
                    last_round = Clock::now();
                }
            }
            const unsigned decoding = decoding_count();
            if (decoding == 0 && !pf_ && !imp_) {
                if (table_.live() != 0) continue;
                if (queue_.has_queued_work()) continue;
                if (queue_.stopping()) return;
                if (std::chrono::duration<double>(Clock::now() - last_round).count() >= cfg_.heartbeat_seconds) {
                    Control hb;
                    hb.cmd = static_cast<std::uint32_t>(Cmd::heartbeat);
                    note_peer(ch_.exchange(hb));
                    {
                        std::lock_guard<std::mutex> l(m_);
                        ++stats_.heartbeats;
                    }
                    count_round();
                    last_round = Clock::now();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            // decode first: the running requests' round, then the import poll of a prompt waiting for its RTX state
            // (the C1 MTP path stays open meanwhile), then one chunk of the prompt in prefill
            const bool importing = imp_ && imp_->import_id != 0;
            if (decoding > 0 && !(decoding == 1 && !pf_ && mtp_round())) {
                if (g2_ready()) g2_cycle();
                else step_all();
            }
            if (importing && decoding == 0 && !pf_) {  // nothing else runs: a heartbeat keeps the follower's status coming
                Control hb;
                hb.cmd = static_cast<std::uint32_t>(Cmd::heartbeat);
                (void)control_round(hb);
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            if (importing) poll_import();
            if (pf_) prefill_chunk();
            last_round = Clock::now();
        }
    }

    unsigned decoding_count() const {
        unsigned n = 0;
        for (const auto& a : active_) n += a ? 1U : 0U;
        return n;
    }

    void count_round() {
        std::lock_guard<std::mutex> l(m_);
        ++stats_.rounds;
    }

    // Request-local rejection: never reaches the engine or the peer.
    void reject(strix::serve::TakenJob& job, int status, const std::string& code, const std::string& msg) {
        strix::serve::GenerationStats st;
        st.error_status = status;
        st.error_code = code;
        st.prompt_tokens = job.job.prompt.size();
        queue_.finish(job, std::move(st), msg);
    }

    void admit(strix::serve::TakenJob job) {
        const std::size_t n = job.job.prompt.size();
        if (n == 0) return reject(job, 400, "invalid_request", "empty prompt");
        if (n >= cfg_.capacity)
            return reject(job, 400, "context_length_exceeded",
                          "prompt of " + std::to_string(n) + " tokens does not fit the context of " + std::to_string(cfg_.capacity) + " positions");
        const auto slot = table_.alloc();
        if (!slot) throw std::logic_error("admit without a free slot");
        Prefilling p;
        p.slot = *slot;
        p.a.job = std::move(job);
        p.a.stops = cfg_.stops;
        for (const auto t : p.a.job.job.stop_tokens)
            if (std::find(p.a.stops.begin(), p.a.stops.end(), t) == p.a.stops.end()) p.a.stops.push_back(t);
        p.a.t_admit = Clock::now();
        p.a.mtp_ok = mtp_on_.load();
        p.chunk = p.a.mtp_ok && !cfg_.prefill_chunk_mtp ? 0U : chunk_.load();
        p.a.job.ticket->report_progress({strix::serve::GenerationPhase::prefill, 0, 0, n});
        // Preserve the optional FAST import route's identity selection; native snapshots never replace an eligible import.
        const bool import_route = cfg_.rtx_prefill && route_on_.load() && !imp_ && n >= 2 && n >= cfg_.import_min_tokens;
        if (cfg_.prefix_cache_bytes) {
            if (const auto* e = import_route ? nullptr : prefix_cache_->find(p.a.job.job.prompt, p.a.mtp_ok, cfg_.prefix_identity)) {
                // Close C1 before restoring any device state; restored requests bootstrap fresh host control.
                Control c;
                c.cmd = static_cast<std::uint32_t>(Cmd::prefix_restore);
                c.pslot = p.slot;
                c.ppos = e->frontier;
                c.prefix_id = e->id;
                c.prefix_identity = cfg_.prefix_identity;
                c.pflags = p.a.mtp_ok ? kPrefillMtp : 0U;
                pf_ = std::move(p);  // a failed paired restore must finish this ticket through fail_all too
                if (mtp_slot_ >= 0) leave();
                const auto t0 = Clock::now();
                (void)engine_round(c);
                pf_->done = c.ppos;
                pf_->a.hit_tokens = c.ppos;
                pf_->a.restore_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                { std::lock_guard<std::mutex> l(m_); ++stats_.prefix_hits; }
                return;
            }
            std::lock_guard<std::mutex> l(m_);
            ++stats_.prefix_misses;
        }
        if (import_route) {
            // RTX route: both ranks start receiving (off their decode path); poll_import() asks the RTX once both
            // listen, then commits or falls back to the chunked prefill. Not an engine command: seeds stay valid.
            p.import_id = next_import_id_++;
            if (next_import_id_ == 0) next_import_id_ = 1;
            p.import_t0 = Clock::now();
            Control e;
            e.cmd = static_cast<std::uint32_t>(Cmd::import_expect);
            e.import_id = p.import_id;
            e.plen = static_cast<std::uint32_t>(n);
            (void)control_round(e, &p.a.job.job.prompt);
            imp_ = std::move(p);  // waits off the prefill place: the next prompt is admitted meanwhile
            return;
        }
        pf_ = std::move(p);
    }

    // Both ranks forget the hand-off staged (or still being received) under `id`: a fallback or a cancelled request.
    void drop_import(std::uint32_t id) {
        Control d;
        d.cmd = static_cast<std::uint32_t>(Cmd::import_drop);
        d.import_id = id;
        (void)control_round(d);
    }
    // A cancelled prompt in prefill or waiting for its import: its slot is freed (the next prefill there resets it).
    void drop_prefill(std::optional<Prefilling>& p) {
        strix::serve::GenerationStats st;
        st.finish = "cancelled";
        st.prompt_tokens = p->a.job.job.prompt.size();
        queue_.finish(p->a.job, std::move(st));
        table_.release(p->slot);
        p.reset();
    }

    // One prefill call: the next chunk of the prompt in prefill (the first chunk resets its slot), or all of the rest
    // when no request decodes (adaptive: chunking only bounds the decode stall of running requests). The last chunk
    // returns the first token and moves the request to decoding.
    void prefill_chunk() {
        if (mtp_slot_ >= 0) leave();
        Prefilling& p = *pf_;
        const auto& prompt = p.a.job.job.prompt;
        const std::size_t n = prompt.size();
        const bool whole = p.chunk == 0 || decoding_count() == 0;
        const std::size_t m = whole ? n - p.done : std::min<std::size_t>(p.chunk, n - p.done);
        const bool last = p.done + m == n;
        Control c;
        c.cmd = static_cast<std::uint32_t>(Cmd::prefill);
        c.resets = p.done == 0 ? 1U << p.slot : 0U;
        c.pslot = p.slot;
        c.plen = static_cast<std::uint32_t>(m);
        c.ppos = p.done;
        c.pflags = (p.a.mtp_ok ? kPrefillMtp : 0U) | (last ? kPrefillLast : 0U);
        std::vector<std::uint32_t> chunk(prompt.begin() + static_cast<std::ptrdiff_t>(p.done),
                                         prompt.begin() + static_cast<std::ptrdiff_t>(p.done + m));
        const auto t0 = Clock::now();
        const auto out = engine_round(c, &chunk);
        const auto t1 = Clock::now();
        p.done += m;
        {
            std::lock_guard<std::mutex> l(m_);
            ++stats_.prefill_chunks;
            stats_.prefill_tokens += m;
            stats_.prefill_seconds += std::chrono::duration<double>(t1 - t0).count();
            if (trace_on_.load()) {
                TraceEvent e;
                e.t = t1;
                e.start = t0;
                e.key = p.a.job.ticket.get();
                e.kind = TraceEvent::prefill;
                e.rows = static_cast<std::uint32_t>(m);
                trace_.push_back(e);
            }
            if (last) {
                ++stats_.prefills;
                if (p.a.mtp_ok) ++stats_.mtp_bootstraps;
                ++stats_.active;
            }
        }
        if (!last) {
            p.a.job.ticket->report_progress({strix::serve::GenerationPhase::prefill, p.done, 0, n});
            return;
        }
        if (cfg_.prefix_cache_bytes) store_prefix(p, out.at(0));
        const auto ready = cfg_.prefix_cache_bytes ? Clock::now() : t1;
        const std::uint32_t slot = p.slot;
        Active a = std::move(p.a);
        pf_.reset();
        a.prefill_ms = std::chrono::duration<double, std::milli>(ready - a.t_admit).count();
        a.t_first = ready;
        a.pos = n;
        calib_[slot] = spec::Calibration{};  // a new request: fresh acceptance statistics
        const void* key = a.job.ticket.get();
        const bool mtp_ok = a.mtp_ok;
        active_[slot] = std::move(a);
        if (cfg_.state_digest) digest_state(slot, n);
        if (active_[slot]->job.job.max_tokens == 0) return finish(slot, "length");
        emit(slot, out.at(0));
        trace(key, TraceEvent::first);  // after its first token: the steady window counts decode tokens only
        if (mtp_ok) seed_slot_ = static_cast<int>(slot);
    }

    void store_prefix(Prefilling& p, std::uint32_t y0) {
        Control c;
        c.cmd = static_cast<std::uint32_t>(Cmd::prefix_prepare);
        c.pslot = p.slot;
        c.ppos = p.a.job.job.prompt.size();
        c.prefix_id = next_prefix_id_++;
        c.prefix_identity = cfg_.prefix_identity;
        c.pflags = p.a.mtp_ok ? kPrefillMtp : 0U;
        auto key = p.a.job.job.prompt;
        if (p.a.mtp_ok) key.push_back(y0);  // key length differs from the restore frontier
        c.plen = static_cast<std::uint32_t>(key.size());
        const auto vote = control_round(c, &key);
        if (vote.at(0)) {
            while (const auto id = prefix_cache_->replacement_id()) {
                Control ev;
                ev.cmd = static_cast<std::uint32_t>(Cmd::prefix_evict);
                ev.prefix_id = id;
                (void)control_round(ev);  // replace an equal key only after both new snapshots succeeded
            }
            while (prefix_cache_->bytes() > prefix_cache_->budget() - prefix_cache_->pending_bytes()) {
                Control ev;
                ev.cmd = static_cast<std::uint32_t>(Cmd::prefix_evict);
                ev.prefix_id = prefix_cache_->oldest();
                (void)control_round(ev);  // rank 0 chooses each victim; rank 1 only replays it
            }
        }
        c.cmd = static_cast<std::uint32_t>(Cmd::prefix_commit);
        c.prefix_ok = vote.at(0);
        (void)control_round(c);  // both prepared, or both discard; never invalidate the final prefill seed
        std::lock_guard<std::mutex> l(m_);
        stats_.prefix_stores = prefix_cache_->stores;
        stats_.prefix_evictions = prefix_cache_->evictions;
        stats_.prefix_bytes = prefix_cache_->bytes();
    }

    // Import gate: both ranks digest the slot's state at frontier n (no engine change: seeds stay valid); the decode
    // clock of the request restarts after it.
    void digest_state(std::uint32_t slot, std::size_t n) {
        Control d;
        d.cmd = static_cast<std::uint32_t>(Cmd::state_digest);
        d.pslot = slot;
        d.ppos = n;
        auto words = control_round(d);
        StateDigest g;
        g.key = active_[slot]->job.ticket.get();
        g.frontier = n;
        const std::size_t half = words.size() / 2;
        g.rank0.assign(words.begin(), words.begin() + static_cast<std::ptrdiff_t>(half));
        g.rank1.assign(words.begin() + static_cast<std::ptrdiff_t>(half), words.end());
        {
            std::lock_guard<std::mutex> l(m_);
            digests_.push_back(std::move(g));
        }
        active_[slot]->t_first = Clock::now();
    }

    void step_all() {
        if (mtp_slot_ >= 0) leave();
        std::vector<WireRow> rows;
        for (std::uint32_t s = 0; s < cfg_.slots; ++s)
            if (active_[s]) rows.push_back({s, active_[s]->last, active_[s]->pos});
        Control c;
        c.cmd = static_cast<std::uint32_t>(Cmd::step);
        c.nrows = static_cast<std::uint32_t>(rows.size());
        std::copy(rows.begin(), rows.end(), c.row);
        const auto t0 = Clock::now();
        const auto out = engine_round(c);
        last_rows_ = rows;
        {
            std::lock_guard<std::mutex> l(m_);
            ++stats_.steps;
            stats_.rows += rows.size();
            ++stats_.rows_hist[rows.size()];
            if (!consecutive_slots(rows)) ++stats_.gap_steps;
            if (trace_on_.load())
                trace_.push_back({Clock::now(), nullptr, TraceEvent::step, static_cast<std::uint32_t>(rows.size()), !consecutive_slots(rows)});
            stats_.step_seconds += std::chrono::duration<double>(Clock::now() - t0).count();
        }
        for (std::size_t r = 0; r < rows.size(); ++r) {
            Active& a = *active_[rows[r].slot];
            ++a.pos;
            emit(rows[r].slot, out[r]);
        }
    }

    // Publishes token t of the slot's request (stop tokens included: the API strips them) and decides what follows.
    void emit(std::uint32_t s, std::uint32_t t) {
        Active& a = *active_[s];
        a.last = t;
        ++a.generated;
        if (a.generated == 1 && a.job.job.observed)
            a.first_token_ms = std::chrono::duration<double, std::milli>(Clock::now() - a.job.submitted).count();
        trace(a.job.ticket.get(), TraceEvent::token);
        const std::uint32_t one[1] = {t};
        const bool open = a.job.job.on_tokens ? a.job.job.on_tokens(std::span<const std::uint32_t>(one, 1)) : true;
        const Next n = decide(t, a.generated, a.job.job.max_tokens, a.pos, cfg_.capacity, a.stops);
        if (n != Next::decode) return finish(s, finish_name(n));
        if (!open) return finish(s, "cancelled");
    }

    void finish(std::uint32_t s, const char* why, const std::string& error = {}) {
        if (mtp_slot_ == static_cast<int>(s)) {
            if (failing_) mtp_slot_ = -1;
            else leave();
        }
        Active& a = *active_[s];
        trace(a.job.ticket.get(), TraceEvent::finish);
        strix::serve::GenerationStats st;
        st.finish = why;
        st.prompt_tokens = a.job.job.prompt.size();
        st.generated_tokens = a.generated;
        st.prefill_ms = a.prefill_ms;
        st.decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - a.t_first).count();
        st.first_token_ms = a.first_token_ms;
        st.imported_tokens = a.imported;
        if (cfg_.prefix_cache_bytes) {
            st.engine_extra.emplace_back("prefix_cache", "{\"hit_tokens\":" + std::to_string(a.hit_tokens) +
                ",\"prefill_tokens\":" + std::to_string(st.prompt_tokens - a.imported - a.hit_tokens) +
                ",\"restore_ms\":" + std::to_string(a.restore_ms) + "}");
        }
        if (!a.route.empty()) st.engine_extra.emplace_back("prefill_route", a.route);
        queue_.finish(a.job, std::move(st), error);
        active_[s].reset();
        table_.release(s);
        std::lock_guard<std::mutex> l(m_);
        ++stats_.finished;
        --stats_.active;
    }

    // One engine command: the control round, then the same engine calls as the follower. Every engine command ends
    // the validity of a prefill seed and of the last step's rows (the drafter reads the engine's last hidden rows).
    std::vector<std::uint32_t> engine_round(Control& c, std::vector<std::uint32_t>* prompt = nullptr) {
        seed_slot_ = -1;
        last_rows_.clear();
        return control_round(c, prompt);
    }
    // A control round without an engine command (import_expect, heartbeat): seeds and last rows stay valid.
    std::vector<std::uint32_t> control_round(Control& c, std::vector<std::uint32_t>* prompt = nullptr) {
        const auto before = ch_.round();
        note_peer(ch_.exchange(c));
        std::vector<std::uint32_t> none;
        auto out = replay(eng_, ch_, c, prompt ? *prompt : none, true, cfg_.slots, cfg_.capacity, prefix_cache_ ? &*prefix_cache_ : nullptr);
        { std::lock_guard<std::mutex> l(m_); stats_.rounds += ch_.round() - before; }
        return out;
    }
    // The follower's import status from its last reply block (one round behind; every round updates it).
    void note_peer(const Control& r) {
        peer_import_id_ = r.import_id;
        peer_import_status_ = r.import_status;
        peer_import_token_ = r.import_token;
    }

    // The prompt in prefill waits for its RTX state: commit it once both ranks staged it; on a failure or the timeout
    // the prompt falls back to its own chunked prefill (whose first chunk resets the slot on both ranks).
    void poll_import() {
        Prefilling& p = *imp_;
        std::uint32_t mine = static_cast<std::uint32_t>(ImportStatus::failed);
        if constexpr (requires { eng_.import_report(); }) {
            const ImportReport r = eng_.import_report();
            mine = r.id == p.import_id ? r.status : static_cast<std::uint32_t>(ImportStatus::pending);
        }
        const std::uint32_t peer = peer_import_id_ == p.import_id ? peer_import_status_ : static_cast<std::uint32_t>(ImportStatus::pending);
        const auto ready = static_cast<std::uint32_t>(ImportStatus::ready), failed = static_cast<std::uint32_t>(ImportStatus::failed);
        const double waited = std::chrono::duration<double>(Clock::now() - p.import_t0).count();
        bool rtx_down = false;
        if (!p.import_requested && mine != failed && peer != failed) {
            // The RTX sender connects once, without retry: ask it only after both ranks report their endpoint bound.
            const auto listening = [](std::uint32_t st) { return st == static_cast<std::uint32_t>(ImportStatus::listening) || st == static_cast<std::uint32_t>(ImportStatus::ready); };
            if (listening(mine) && listening(peer) && waited >= cfg_.import_listen_grace_s) {
                p.import_requested = true;
                p.import_t_req = Clock::now();
                rtx_down = !cfg_.rtx_prefill(std::span<const std::uint32_t>(p.a.job.job.prompt), p.import_id);
            } else if (waited < cfg_.import_timeout_s) {
                return;
            }
        }
        if (p.import_requested && cfg_.rtx_failed && cfg_.rtx_failed(p.import_id)) rtx_down = true;
        const bool ok = mine == ready && peer == ready;
        if (!ok && !rtx_down && mine != failed && peer != failed && waited < cfg_.import_timeout_s) return;
        if (!ok)
            return import_fallback(false, mine == failed ? "rank0_staging_failed" : peer == failed ? "rank1_staging_failed"
                                          : rtx_down    ? "rtx_unavailable"
                                                        : "timeout");
        commit_import(waited);
    }

    // The import is abandoned: the prompt takes the chunked prefill from position 0 once the prefill place is free
    // (decode-first as any prompt). After an import_state round (`now`) the slot is reset on both ranks at once.
    void import_fallback(bool now, const char* why) {
        {
            std::lock_guard<std::mutex> l(m_);
            ++stats_.import_fallbacks;
        }
        imp_->a.route = std::string("{\"route\":\"pair\",\"fallback\":\"") + why + "\"}";
        const std::uint32_t id = imp_->import_id;
        imp_->import_id = 0;
        if (!now && id != 0) drop_import(id);  // the receivers may still run: free them on both ranks now
        if (now) {
            Control r;
            r.cmd = static_cast<std::uint32_t>(Cmd::heartbeat);
            r.resets = 1U << imp_->slot;
            (void)engine_round(r);
        }
        if (!pf_) {
            pf_ = std::move(imp_);
            imp_.reset();
        }
    }

    // import_state: the slot restored from the staged state on both ranks, then its last prompt token through the
    // same prefill / MTP bootstrap call as its own prefill; the request then decodes. No chunked prefill.
    void commit_import(double waited) {
        if (mtp_slot_ >= 0) leave();
        Prefilling& p = *imp_;
        const std::size_t n = p.a.job.job.prompt.size();
        Control c;
        c.cmd = static_cast<std::uint32_t>(Cmd::import_state);
        c.resets = 1U << p.slot;
        c.pslot = p.slot;
        c.plen = static_cast<std::uint32_t>(n);
        c.pflags = (p.a.mtp_ok ? kPrefillMtp : 0U) | (cfg_.import_dsa_once ? kImportDsaOnce : 0U);
        c.import_id = p.import_id;
        const auto t0 = Clock::now();
        const std::uint32_t y0 = engine_round(c, &p.a.job.job.prompt).at(0);
        // Both ranks' outcome: the follower reports it in the reply of the next round (a heartbeat; seeds stay valid).
        // Either rank failed (its slot is reset), or the first tokens differ: the prompt prefills itself instead.
        Control hb;
        hb.cmd = static_cast<std::uint32_t>(Cmd::heartbeat);
        (void)control_round(hb);
        const auto restored = static_cast<std::uint32_t>(ImportStatus::restored);
        bool ok = y0 != kImportFailed && peer_import_id_ == p.import_id && peer_import_status_ == restored && peer_import_token_ == y0;
        if constexpr (requires { eng_.import_report(); }) {
            const ImportReport r = eng_.import_report();
            ok = ok && r.id == p.import_id && r.status == restored;
        }
        // (a failure in the prepare / restored votes aborts on both ranks: "both_restore_failed")
        if (!ok) return import_fallback(true, y0 == kImportFailed && peer_import_status_ != restored ? "both_restore_failed"
                                              : y0 == kImportFailed                                  ? "rank0_restore_failed"
                                              : peer_import_status_ != restored                      ? "rank1_restore_failed"
                                                                                                     : "first_token_mismatch");
        const auto t1 = Clock::now();
        p.a.imported = n - 1;
        {
            std::uint64_t relay_bytes = 0;
            double relay_ms = 0;
            if constexpr (requires { eng_.last_relay.bytes; }) {  // rank 0's side of the DSA relay (dsa_once)
                relay_bytes = eng_.last_relay.bytes;
                relay_ms = eng_.last_relay.ms;
            }
            char buf[384];
            std::snprintf(buf, sizeof buf,
                          "{\"route\":\"rtx\",\"identity\":\"glm-pf-rtx-native-v1 (FAST)\",\"imported_tokens\":%llu,\"wait_ms\":%.1f,\"listen_ms\":%.1f,\"staged_ms\":%.1f,\"commit_ms\":%.1f,"
                          "\"dsa_once\":%s,\"relay_bytes\":%llu,\"relay_ms\":%.1f}",
                          static_cast<unsigned long long>(n - 1), waited * 1000.0,
                          std::chrono::duration<double, std::milli>(p.import_t_req - p.import_t0).count(),
                          std::chrono::duration<double, std::milli>(t0 - p.import_t_req).count(),
                          std::chrono::duration<double, std::milli>(t1 - t0).count(), cfg_.import_dsa_once ? "true" : "false",
                          static_cast<unsigned long long>(cfg_.import_dsa_once ? relay_bytes : 0), cfg_.import_dsa_once ? relay_ms : 0.0);
            p.a.route = buf;
        }
        {
            std::lock_guard<std::mutex> l(m_);
            ++stats_.imports;
            stats_.import_wait_seconds += waited;
            stats_.imported_tokens += n - 1;
            ++stats_.prefills;
            ++stats_.prefill_chunks;
            stats_.prefill_tokens += 1;
            stats_.prefill_seconds += std::chrono::duration<double>(t1 - t0).count();
            if (p.a.mtp_ok) ++stats_.mtp_bootstraps;
            ++stats_.active;
        }
        const std::uint32_t slot = p.slot;
        Active a = std::move(p.a);
        imp_.reset();
        a.prefill_ms = std::chrono::duration<double, std::milli>(t1 - a.t_admit).count();
        a.t_first = t1;
        a.pos = n;
        calib_[slot] = spec::Calibration{};  // a new request: fresh acceptance statistics (as after its own prefill)
        const void* key = a.job.ticket.get();
        const bool mtp_ok = a.mtp_ok;
        active_[slot] = std::move(a);
        if (cfg_.state_digest) digest_state(slot, n);
        if (active_[slot]->job.job.max_tokens == 0) return finish(slot, "length");
        emit(slot, y0);
        trace(key, TraceEvent::first);
        if (mtp_ok) seed_slot_ = static_cast<int>(slot);
    }

    // Close the open MTP decoder (its deferred KDA rows are applied) at a cycle boundary.
    void leave() {
        Control c;
        c.cmd = static_cast<std::uint32_t>(Cmd::mtp_leave);
        c.pslot = static_cast<std::uint32_t>(mtp_slot_);
        mtp_slot_ = -1;
        (void)engine_round(c);
        std::lock_guard<std::mutex> l(m_);
        ++stats_.mtp_leaves;
    }

    // The C1 path: the only request decodes through MTP verify cycles (no prompt in prefill). Entry needs a fresh
    // drafter seed: the request's MTP prefill (the last engine command), or the row the last plain step produced for
    // it (mtp_enter). Otherwise a plain step runs first. Returns false to take the plain step.
    bool mtp_round() {
        if (!mtp_on_.load()) return false;
        if (cfg_.batch && queue_.has_queued_work()) return false;  // admission first, then batched steps
        std::uint32_t s = 0;
        while (!active_[s]) ++s;
        Active& a = *active_[s];
        if (!a.mtp_ok) return false;
        if (mtp_slot_ != static_cast<int>(s)) {
            const SlotSpec& st = eng_.spec_state(s);
            if (seed_slot_ != static_cast<int>(s) && st.seeded && !st.disabled && st.drafts.size() == 1 && st.p == a.pos) {
                // from batched self-speculation: the slot's drafter seed of its last window
                Control e;
                e.cmd = static_cast<std::uint32_t>(Cmd::mtp_enter);
                e.pflags = kEnterSeed;
                e.row[0] = {s, a.last, a.pos};
                (void)engine_round(e);
                seed_slot_ = static_cast<int>(s);
                std::lock_guard<std::mutex> l(m_);
                ++stats_.mtp_enters;
            }
            if (seed_slot_ != static_cast<int>(s)) {
                std::size_t r = 0;
                while (r < last_rows_.size() && last_rows_[r].slot != s) ++r;
                if (r == last_rows_.size() || a.pos == 0 || a.pos >= cfg_.capacity) return false;
                Control e;
                e.cmd = static_cast<std::uint32_t>(Cmd::mtp_enter);
                e.pslot = static_cast<std::uint32_t>(r);
                e.row[0] = {s, a.last, a.pos - 1};  // the step row that produced a.last
                (void)engine_round(e);
                seed_slot_ = static_cast<int>(s);
                std::lock_guard<std::mutex> l(m_);
                ++stats_.mtp_enters;
            }
            mtp_slot_ = static_cast<int>(s);
            mtp_budget_ = 1U + static_cast<std::uint32_t>(std::min<std::uint64_t>(a.job.job.max_tokens - a.generated,
                                                                                  cfg_.capacity - a.pos));
        }
        Control c;
        c.cmd = static_cast<std::uint32_t>(Cmd::mtp_run);
        c.pslot = s;
        c.budget = mtp_budget_;
        c.cycles = std::clamp(cfg_.mtp_cycles, 1U, kMaxMtpCycles);
        const auto t0 = Clock::now();
        const auto out = engine_round(c);
        {
            std::lock_guard<std::mutex> l(m_);
            ++stats_.mtp_rounds;
            stats_.mtp_tokens += out.size();
            stats_.mtp_seconds += std::chrono::duration<double>(Clock::now() - t0).count();
            if (trace_on_.load()) trace_.push_back({Clock::now(), nullptr, TraceEvent::mtp, static_cast<std::uint32_t>(out.size()), false});
        }
        for (const auto t : out) {
            if (!active_[s]) break;  // finished (stop / cancel): the rest of the cycle is discarded
            ++active_[s]->pos;
            emit(s, t);
        }
        if (out.empty() && active_[s]) throw std::logic_error("MTP round produced no token for an unfinished request");
        return true;
    }

    // ---- batched self-speculation (G2)
    bool g2_ready() const {
        if (!g2_on_.load() || !mtp_on_.load()) return false;
        unsigned n = 0;
        for (const auto& a : active_) {
            if (a && !a->mtp_ok) return false;
            n += a ? 1U : 0U;
        }
        return n > 0;
    }
    // The G2 policy's plain step (no drafter calls): the slots' drafter seeds lapse with it.
    void g2_plain_step() {
        step_all();
        ++g2_plain_streak_;
        std::lock_guard<std::mutex> l(m_);
        ++stats_.g2_plain;
    }
    double cost_ms(unsigned n, unsigned rows) const { return cost_[std::min(n, kMaxRows)][std::min(rows, kMaxRows)]; }

    // One more draft for slot s (an MTP extension step of its drafter chain).
    void draft_round(std::uint32_t s) {
        Control c;
        c.cmd = static_cast<std::uint32_t>(Cmd::spec_draft);
        c.pslot = s;
        const auto t0 = Clock::now();
        (void)engine_round(c);
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        draft_ms_ = 0.8 * draft_ms_ + 0.2 * ms;
        std::lock_guard<std::mutex> l(m_);
        ++stats_.g2_draft_calls;
    }

    // One cycle of every decoding slot: windows sized greedily by expected tokens per cycle cost (each step takes the
    // slot whose next draft has the largest survival-weighted acceptance probability, while the rate improves; a draft
    // not yet proposed costs one drafter step), then ONE spec_verify; tokens emitted per slot.
    void g2_cycle() {
        if (mtp_slot_ >= 0) leave();
        struct Win {
            std::uint32_t s = 0;
            unsigned k = 0, kcap = 0;
            float surv = 1;
            std::uint32_t remaining = 0;
            std::vector<float> conf;
        };
        std::vector<Win> w;
        for (std::uint32_t s = 0; s < cfg_.slots; ++s) {
            if (!active_[s]) continue;
            const Active& a = *active_[s];
            Win x;
            x.s = s;
            x.remaining = static_cast<std::uint32_t>(std::min<std::uint64_t>(a.job.job.max_tokens - a.generated, cfg_.capacity - a.pos));
            const SlotSpec& st = eng_.spec_state(s);
            const bool usable = st.seeded && !st.disabled && st.p == a.pos && !st.drafts.empty();
            x.kcap = usable ? std::min(spec::draft_limit(cfg_.kmax, x.remaining, a.pos, cfg_.capacity), kMaxRows - 1U) : 0U;
            w.push_back(std::move(x));
        }
        const auto n = static_cast<unsigned>(w.size());
        if (n * 2 > kMaxRows) {  // no room for a draft per window: plain steps until the slot count drops
            g2_last_n_ = n;
            return g2_plain_step();
        }
        bool any_usable = false;
        for (const auto& x : w) any_usable = any_usable || x.kcap > 0;
        const bool reseed = !any_usable;
        if (reseed) {
            // no drafter seed to draft from: a plain step, unless the slot set changed or the plain run is long enough to
            // try again (an anchor-only window cycle: plain rows + the catch-ups that seed every slot)
            if (n == g2_last_n_ && g2_plain_streak_ < cfg_.g2_retry) return g2_plain_step();
        }
        g2_last_n_ = n;
        unsigned rows = n;
        double gain = 0, cost = cost_ms(n, rows);
        while (rows < kMaxRows) {
            int best = -1;
            float bg = 0;
            for (std::size_t i = 0; i < w.size(); ++i) {
                if (w[i].k >= w[i].kcap) continue;
                const SlotSpec& st = eng_.spec_state(w[i].s);
                const float c = w[i].k < st.conf.size() ? st.conf[w[i].k] : st.conf.back();
                const float g = w[i].surv * calib_[w[i].s].probability(w[i].k + 1, c);
                if (g > bg) {
                    bg = g;
                    best = static_cast<int>(i);
                }
            }
            if (best < 0) break;
            Win& x = w[static_cast<std::size_t>(best)];
            const bool need = x.k >= eng_.spec_state(x.s).drafts.size();
            const double extra = cost_ms(n, rows + 1) - cost_ms(n, rows) + (need ? draft_ms_ : 0.0);
            if ((n + gain + bg) / (cost + extra) <= (n + gain) / cost) break;
            if (need) {
                draft_round(x.s);
                cost += draft_ms_;
                const SlotSpec& st = eng_.spec_state(x.s);
                if (st.disabled || st.drafts.size() <= x.k) {
                    x.kcap = x.k;
                    continue;
                }
                bg = x.surv * calib_[x.s].probability(x.k + 1, st.conf[x.k]);
            }
            cost += cost_ms(n, rows + 1) - cost_ms(n, rows);
            x.conf.push_back(eng_.spec_state(x.s).conf[x.k]);
            ++x.k;
            ++rows;
            x.surv = bg;
            gain += bg;
        }
        if (!reseed && rows == n) return g2_plain_step();  // the cost model gives no draft: skip the drafter entirely
        g2_plain_streak_ = 0;
        if (reseed) {
            std::lock_guard<std::mutex> l(m_);
            ++stats_.g2_reseeds;
        }
        Control c;
        c.cmd = static_cast<std::uint32_t>(Cmd::spec_verify);
        for (std::size_t i = 0; i < w.size(); ++i) {
            const Active& a = *active_[w[i].s];
            c.row[c.nrows++] = {w[i].s, a.last, a.pos};
            const SlotSpec& st = eng_.spec_state(w[i].s);
            for (unsigned j = 0; j < w[i].k; ++j) c.row[c.nrows++] = {w[i].s, st.drafts[j], a.pos + 1 + j};
            c.sbudget[i] = w[i].remaining;
        }
        const auto t0 = Clock::now();
        const auto out = engine_round(c);
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        cost_[std::min(n, kMaxRows)][c.nrows] = 0.8 * cost_[std::min(n, kMaxRows)][c.nrows] + 0.2 * ms;
        std::size_t at = 0;
        std::uint64_t drafts = 0, accepted = 0, emitted = 0;
        for (std::size_t i = 0; i < w.size(); ++i) {
            if (at >= out.size()) throw std::logic_error("spec_verify: short output");
            const std::uint32_t cnt = out[at++];
            if (cnt == 0 || at + cnt > out.size()) throw std::logic_error("spec_verify: bad window output");
            const Win& x = w[i];
            const unsigned m = cnt - 1;  // accepted drafts when the window ended with a bonus
            if (cnt < x.remaining && m <= x.k)  // not terminal: a bonus ended it
                for (unsigned d = 1; d <= x.k && d <= m + 1; ++d) calib_[x.s].observe(d, x.conf[d - 1], d <= m);
            drafts += x.k;
            accepted += std::min<unsigned>(m, x.k);
            emitted += cnt;
            for (std::uint32_t j = 0; j < cnt; ++j) {
                const std::uint32_t t = out[at++];
                if (!active_[x.s]) continue;  // finished (stop / cancel): the rest of its window is discarded
                ++active_[x.s]->pos;
                emit(x.s, t);
            }
        }
        std::lock_guard<std::mutex> l(m_);
        ++stats_.g2_cycles;
        stats_.g2_rows += c.nrows;
        stats_.g2_windows += n;
        stats_.g2_drafts += drafts;
        stats_.g2_accepted += accepted;
        stats_.g2_tokens += emitted;
        stats_.g2_seconds += ms / 1000.0;
        ++stats_.rows_hist[c.nrows];
        if (trace_on_.load()) trace_.push_back({Clock::now(), nullptr, TraceEvent::step, c.nrows, false});
    }

    void trace(const void* key, TraceEvent::Kind kind) {
        if (!trace_on_.load()) return;
        std::lock_guard<std::mutex> l(m_);
        trace_.push_back({Clock::now(), key, kind, 0, false});
    }

    void fail_all(const std::string& error) {
        for (std::uint32_t s = 0; s < cfg_.slots; ++s)
            if (active_[s]) finish(s, "length", error);
        for (auto* p : {&pf_, &imp_})
            if (*p) {
                strix::serve::GenerationStats st;
                st.prompt_tokens = (*p)->a.job.job.prompt.size();
                queue_.finish((*p)->a.job, std::move(st), error);
                table_.release((*p)->slot);
                p->reset();
            }
    }

    Eng& eng_;
    Channel& ch_;
    strix::serve::GenerationQueue& queue_;
    LeadConfig cfg_;
    std::atomic<bool>& stop_;
    SlotTable table_;
    std::vector<std::optional<Active>> active_;
    std::optional<Prefilling> pf_;
    std::optional<Prefilling> imp_;  // the prompt waiting for its RTX state (its slot allocated), off the prefill place
    std::uint32_t next_import_id_ = 1, peer_import_id_ = 0, peer_import_status_ = 0, peer_import_token_ = 0;
    std::atomic<bool> mtp_on_;
    std::atomic<unsigned> chunk_;
    std::atomic<bool> g2_on_;
    std::vector<spec::Calibration> calib_;  // per slot (rank 0: window sizes)
    double cost_[kMaxRows + 1][kMaxRows + 1]{};  // ms of a spec_verify round by [windows][rows]
    double draft_ms_ = 2.5;
    unsigned g2_last_n_ = 0, g2_plain_streak_ = 0;
    std::atomic<bool> trace_on_{false};
    std::vector<TraceEvent> trace_;
    std::vector<StateDigest> digests_;
    std::atomic<bool> route_on_{true};
    bool failing_ = false;
    int mtp_slot_ = -1;   // slot of the open MTP decoder
    int seed_slot_ = -1;  // slot whose drafter seed is fresh (MTP prefill / mtp_enter was the last engine command)
    std::uint32_t mtp_budget_ = 0;
    std::vector<WireRow> last_rows_;  // rows of the last engine command when it was a step
    mutable std::mutex m_;
    LeadStats stats_;
    std::optional<PrefixCache> prefix_cache_;
    std::uint64_t next_prefix_id_ = 1;
    std::atomic<bool> prefix_clear_{false};
};

}  // namespace strix::glm::serve
