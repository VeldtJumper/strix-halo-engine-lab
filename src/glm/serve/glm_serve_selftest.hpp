#pragma once
// GLM server self-test (rank 0, no HTTP): submits prompts to the lead's GenerationQueue, first N at once, then each alone
// (C=1, the reference), then the first C together for C = 2, 4, 8 (<= N); every stream must equal its alone run token
// for token. Prints EXACT/FAIL per level with aggregate and per-stream decode tok/s. Host-only (CPU tests use it too).
#include "glm/serve/glm_serve_lead.hpp"
#include "strix/serve/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace strix::glm::serve {

struct SelftestStream {
    std::vector<std::uint32_t> tokens;
    strix::serve::GenerationStats stats;
    std::string error;
    const void* key = nullptr;  // the ticket (trace key)
};
// Steady-state window of a barrier-started level: from the last stream's first token (every stream past its prefill,
// all rows decoding) to the first stream's finish.
struct SelftestWindow {
    bool valid = false;
    double seconds = 0;
    std::uint64_t tokens = 0, steps = 0, rows = 0, gap_steps = 0, mtp_rounds = 0;
    std::vector<std::uint64_t> per_stream;
    std::vector<double> per_stream_seconds;  // C>1: the window; C=1: each stream's own decode window
};
struct SelftestLevel {
    unsigned c = 0;
    std::string label;  // C=4, C=1 mtp, C=1 plain
    std::vector<std::size_t> idx;
    std::vector<SelftestStream> streams;
    double wall = 0;
    unsigned equal = 0;
    SelftestWindow window;  // C=1 levels: one window per stream, summed
};
struct SelftestResult {
    bool exact = false;  // set by a completed run
    std::vector<SelftestLevel> levels;
    double stall_whole_ms = 0, stall_chunked_ms = 0;  // interleave levels: decode stall before / after
};

// Submits the prompts of `idx` and collects every stream. Barrier start (all at once), or staggered when `progress` is
// set: the next request is submitted once the server ran `every` more decode rounds (or nothing it was given runs).
inline std::vector<SelftestStream> selftest_batch(strix::serve::GenerationQueue& queue,
                                                  const std::vector<std::vector<std::uint32_t>>& prompts,
                                                  const std::vector<std::size_t>& idx, std::uint32_t max_tokens,
                                                  const std::vector<std::uint32_t>& stops, double& wall,
                                                  const std::function<std::uint64_t()>& progress = {}, unsigned every = 0) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::shared_ptr<strix::serve::GenerationTicket>> tickets;
    std::vector<SelftestStream> out(idx.size());
    std::vector<char> done(idx.size(), 0);
    auto drain = [&](std::size_t k, std::chrono::milliseconds wait) {
        bool d = false;
        const auto got = tickets[k]->wait(wait, d);
        out[k].tokens.insert(out[k].tokens.end(), got.begin(), got.end());
        if (d) done[k] = 1;
    };
    for (std::size_t k = 0; k < idx.size(); ++k) {
        if (k > 0 && progress) {
            const std::uint64_t base = progress();
            for (;;) {
                bool running = false;
                for (std::size_t j = 0; j < k; ++j) {
                    if (!done[j]) drain(j, std::chrono::milliseconds(0));
                    running = running || !done[j];
                }
                if (!running || progress() >= base + every) break;
                std::this_thread::sleep_for(std::chrono::microseconds(500));
            }
        }
        strix::serve::GenerationJob job;
        job.prompt = prompts.at(idx[k]);
        job.max_tokens = max_tokens;
        job.stop_tokens = stops;
        auto t = queue.submit(std::move(job));
        if (!t)
            throw std::runtime_error(queue.stopping() ? "selftest: the server is stopping (signal or failure): queue refused a job"
                                                      : "selftest: queue refused a job (queue full)");
        tickets.push_back(std::move(t));
    }
    for (std::size_t k = 0; k < tickets.size(); ++k) {
        while (!done[k]) drain(k, std::chrono::milliseconds(200));
        out[k].stats = tickets[k]->stats();
        out[k].error = tickets[k]->error();
        out[k].key = tickets[k].get();
    }
    wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

// The decode stall: the longest gap between two consecutive tokens of one stream that overlaps a prefill call of
// another request (ms; 0 when no gap overlaps a prefill).
inline double max_prefill_stall_ms(const std::vector<SelftestStream>& streams, const std::vector<TraceEvent>& ev) {
    double worst = 0;
    for (const auto& st : streams) {
        bool have = false;
        std::chrono::steady_clock::time_point prev{};
        for (const auto& e : ev) {
            if (e.key != st.key || e.kind != TraceEvent::token) continue;
            if (have) {
                for (const auto& p : ev)
                    if (p.kind == TraceEvent::prefill && p.key != st.key && p.start < e.t && p.t > prev) {
                        worst = std::max(worst, std::chrono::duration<double, std::milli>(e.t - prev).count());
                        break;
                    }
            }
            prev = e.t;
            have = true;
        }
    }
    return worst;
}

inline double stream_decode_tok_s(const SelftestStream& s) {
    return s.stats.generated_tokens > 1 && s.stats.decode_ms > 0
               ? 1000.0 * static_cast<double>(s.stats.generated_tokens - 1) / s.stats.decode_ms
               : 0.0;
}

inline SelftestWindow steady_window(const std::vector<SelftestStream>& streams, const std::vector<TraceEvent>& ev) {
    SelftestWindow w;
    using TP = std::chrono::steady_clock::time_point;
    TP t0 = TP::min(), t1 = TP::max();
    for (const auto& st : streams) {
        bool got_first = false, got_finish = false;
        for (const auto& e : ev) {
            if (e.key != st.key) continue;
            if (e.kind == TraceEvent::first) { t0 = std::max(t0, e.t); got_first = true; }
            if (e.kind == TraceEvent::finish) { t1 = std::min(t1, e.t); got_finish = true; }
        }
        if (!got_first || !got_finish) return w;
    }
    if (!(t1 > t0)) return w;
    w.valid = true;
    w.seconds = std::chrono::duration<double>(t1 - t0).count();
    w.per_stream.assign(streams.size(), 0);
    w.per_stream_seconds.assign(streams.size(), w.seconds);
    for (const auto& e : ev) {
        if (!(e.t > t0 && e.t <= t1)) continue;
        if (e.kind == TraceEvent::step) {
            ++w.steps;
            w.rows += e.rows;
            if (e.gap) ++w.gap_steps;
        } else if (e.kind == TraceEvent::mtp) {
            ++w.mtp_rounds;
        } else if (e.kind == TraceEvent::token) {
            for (std::size_t k = 0; k < streams.size(); ++k)
                if (e.key == streams[k].key) { ++w.per_stream[k]; ++w.tokens; }
        }
    }
    return w;
}

inline void print_window(const SelftestLevel& l, const std::function<void(const std::string&)>& out) {
    const auto& w = l.window;
    if (!w.valid || w.seconds <= 0) {
        out("selftest " + l.label + " steady window: EMPTY (a stream finished before the last prefill ended)");
        return;
    }
    std::string per;
    double mn = 1e300, sum = 0;
    for (std::size_t k = 0; k < w.per_stream.size(); ++k) {
        const double sec = w.per_stream_seconds.at(k);
        const double r = sec > 0 ? static_cast<double>(w.per_stream[k]) / sec : 0.0;
        char b[32];
        std::snprintf(b, sizeof b, "%s%.2f", per.empty() ? "" : " ", r);
        per += b;
        mn = std::min(mn, r);
        sum += r;
    }
    char line[640];
    std::snprintf(line, sizeof line,
                  "selftest %s steady window (all rows decoding): %.2f s, %llu tokens, aggregate %.2f tok/s, per-stream tok/s mean %.2f "
                  "min %.2f [%s], steps %llu rows/step %.2f gap steps %llu, mtp rounds %llu",
                  l.label.c_str(), w.seconds, static_cast<unsigned long long>(w.tokens),
                  static_cast<double>(w.tokens) / w.seconds, w.per_stream.empty() ? 0.0 : sum / static_cast<double>(w.per_stream.size()),
                  w.per_stream.empty() ? 0.0 : mn, per.c_str(), static_cast<unsigned long long>(w.steps),
                  w.steps ? static_cast<double>(w.rows) / static_cast<double>(w.steps) : 0.0,
                  static_cast<unsigned long long>(w.gap_steps), static_cast<unsigned long long>(w.mtp_rounds));
    out(line);
}

inline void print_level(const SelftestLevel& l, const std::function<void(const std::string&)>& out) {
    std::uint64_t gen = 0;
    double sum = 0, mn = 1e300;
    std::string per;
    for (const auto& s : l.streams) {
        gen += s.stats.generated_tokens;
        const double r = stream_decode_tok_s(s);
        sum += r;
        mn = std::min(mn, r);
        char b[32];
        std::snprintf(b, sizeof b, "%s%.2f", per.empty() ? "" : " ", r);
        per += b;
    }
    char line[512];
    std::snprintf(line, sizeof line,
                  "selftest %s streams %zu: %s (%u/%zu equal to C=1 plain) wall %.2f s, %llu tokens, aggregate %.2f tok/s, "
                  "per-stream decode tok/s mean %.2f min %.2f [%s]",
                  l.label.c_str(), l.streams.size(), l.equal == l.streams.size() ? "EXACT" : "FAIL", l.equal, l.streams.size(), l.wall,
                  static_cast<unsigned long long>(gen), l.wall > 0 ? static_cast<double>(gen) / l.wall : 0.0,
                  l.streams.empty() ? 0.0 : sum / static_cast<double>(l.streams.size()), l.streams.empty() ? 0.0 : mn, per.c_str());
    out(line);
}

// Server hooks of the self-test (all optional).
struct SelftestHooks {
    std::function<void(bool)> set_mtp;                  // the C1 MTP switch: C=1 runs with MTP on and off
    std::function<std::vector<TraceEvent>()> take_trace;  // steady windows, prefill stalls
    std::function<void(unsigned)> set_prefill_chunk;    // with long_prompts: chunked vs whole-prompt prefill levels
    std::function<std::uint64_t()> progress;            // decode rounds (staggered arrivals)
    unsigned prefill_chunk = 0;                         // the server's chunk (the "after" of the stall comparison)
    std::vector<unsigned> extra_chunks;                 // further chunk sizes for the staggered arm (default choice)
    bool skip_basic = false;                            // only the long-prompt reference and the staggered arms
    std::function<void(bool)> set_g2;                   // batched self-speculation: C=2 / C=4 also run with it off
    unsigned stagger_every = 16;                        // decode rounds between staggered arrivals
    std::vector<std::vector<std::uint32_t>> long_prompts;  // multi-chunk prompts for the interleave levels
};

// prompts.size() = N >= 1; a level above max_c (the server's concurrency) still runs (queued) and is noted.
// set_mtp: C=1 runs twice, MTP on and off; the reference is C=1 plain, and C=1 MTP and every batched level must equal
// it token for token (MTP on for the batched levels).
// long_prompts + set_prefill_chunk + progress: the decode-first interleave levels on multi-chunk prompts, all compared
// to their whole-prompt plain runs alone: alone with chunked prefill (plain, and MTP when available), then staggered
// arrivals (one every stagger_every decode rounds while the others decode) with whole-prompt prefill (before) and
// chunked prefill (after), each with its decode stall (longest token gap of a running stream across a prefill).
inline SelftestResult run_selftest(strix::serve::GenerationQueue& queue, const std::vector<std::vector<std::uint32_t>>& prompts,
                                   std::uint32_t max_tokens, const std::vector<std::uint32_t>& stops, unsigned max_c,
                                   const std::function<void(const std::string&)>& out, const SelftestHooks& hooks = {}) {
    SelftestResult res;
    res.exact = true;
    const auto& set_mtp = hooks.set_mtp;
    const auto& take_trace = hooks.take_trace;
    // one batch: the trace of exactly its events gives its steady window and stall
    std::vector<TraceEvent> last_trace;
    auto run = [&](const std::vector<std::vector<std::uint32_t>>& set, const std::vector<std::size_t>& idx, double& wall,
                   SelftestWindow* win, bool stagger) {
        if (take_trace) (void)take_trace();
        auto st = selftest_batch(queue, set, idx, max_tokens, stops, wall, stagger ? hooks.progress : std::function<std::uint64_t()>(),
                                 hooks.stagger_every);
        last_trace.clear();
        if (take_trace) last_trace = take_trace();
        if (win) *win = steady_window(st, last_trace);
        return st;
    };
    auto alone_run = [&](const std::vector<std::vector<std::uint32_t>>& set, const std::string& label) {
        SelftestLevel l;
        l.c = 1;
        l.label = label;
        for (std::size_t i = 0; i < set.size(); ++i) {
            double w = 0;
            SelftestWindow one;
            auto st = run(set, {i}, w, &one, false);
            l.idx.push_back(i);
            l.streams.push_back(std::move(st.at(0)));
            l.wall += w;
            // C=1: the decode phases of the streams one after the other
            l.window.valid = (i == 0 || l.window.valid) && one.valid;
            l.window.seconds += one.seconds;
            l.window.tokens += one.tokens;
            l.window.steps += one.steps;
            l.window.rows += one.rows;
            l.window.gap_steps += one.gap_steps;
            l.window.mtp_rounds += one.mtp_rounds;
            l.window.per_stream.push_back(one.per_stream.empty() ? 0 : one.per_stream[0]);
            l.window.per_stream_seconds.push_back(one.seconds);
        }
        return l;
    };
    auto compare = [&](SelftestLevel& l, const SelftestLevel& reference) {
        l.equal = 0;
        for (std::size_t k = 0; k < l.streams.size(); ++k) {
            const auto& ref = reference.streams.at(l.idx[k]);
            if (l.streams[k].error.empty() && ref.error.empty() && !ref.tokens.empty() && l.streams[k].tokens == ref.tokens) ++l.equal;
            else {
                std::size_t d = 0;
                while (d < ref.tokens.size() && d < l.streams[k].tokens.size() && ref.tokens[d] == l.streams[k].tokens[d]) ++d;
                out("selftest " + l.label + " stream " + std::to_string(l.idx[k]) + " DIFFERS at token " + std::to_string(d) +
                    " (" + std::to_string(l.streams[k].tokens.size()) + " tokens" +
                    (l.streams[k].error.empty() ? "" : ", error: " + l.streams[k].error) + "; reference " + std::to_string(ref.tokens.size()) +
                    " tokens" + (ref.error.empty() ? "" : ", error: " + ref.error) + ")");
            }
        }
    };
    auto valid_reference = [](SelftestLevel& l) {
        l.equal = 0;
        for (const auto& st : l.streams)
            if (st.error.empty() && !st.tokens.empty()) ++l.equal;  // the reference: valid when it ran without an error
    };
    auto report = [&](const SelftestLevel& l) {
        print_level(l, out);
        if (take_trace) print_window(l, out);
        if (l.equal != l.streams.size()) res.exact = false;
        if (l.c > max_c && l.c > 1) out("selftest note: C=" + std::to_string(l.c) + " exceeds the server's concurrency " + std::to_string(max_c) + " (queued)");
    };

    // ---- short prompts: barrier-started levels
    std::vector<SelftestLevel> levels;
    if (!hooks.skip_basic) {
    const unsigned n = static_cast<unsigned>(prompts.size());
    std::vector<std::size_t> all(n);
    for (unsigned i = 0; i < n; ++i) all[i] = i;
    if (set_mtp) set_mtp(true);
    SelftestLevel first;  // all N at once (before the alone runs: a cold server)
    first.c = n;
    first.label = "C=" + std::to_string(n) + (hooks.set_g2 ? " g2" : "");
    first.idx = all;
    first.streams = run(prompts, all, first.wall, &first.window, false);
    std::optional<SelftestLevel> mtp;
    if (set_mtp) mtp = alone_run(prompts, "C=1 mtp");
    if (set_mtp) set_mtp(false);
    SelftestLevel alone = alone_run(prompts, set_mtp ? "C=1 plain" : "C=1");
    if (set_mtp) set_mtp(true);
    valid_reference(alone);
    levels.push_back(alone);
    if (mtp) {
        compare(*mtp, alone);
        levels.push_back(std::move(*mtp));
    }
    const std::string g2 = hooks.set_g2 ? " g2" : "";
    for (const unsigned c : {2U, 4U, 6U, 8U}) {
        if (c > n || c == n) continue;
        SelftestLevel l;
        l.c = c;
        l.label = "C=" + std::to_string(c) + g2;
        for (unsigned i = 0; i < c; ++i) l.idx.push_back(i);
        l.streams = run(prompts, l.idx, l.wall, &l.window, false);
        compare(l, alone);
        levels.push_back(std::move(l));
    }
    if (hooks.set_g2) {  // the same levels with batched self-speculation off (batched plain steps): before / after
        hooks.set_g2(false);
        for (const unsigned c : {2U, 4U, 6U, 8U}) {
            if (c > n) continue;
            SelftestLevel l;
            l.c = c;
            l.label = "C=" + std::to_string(c) + " g2-off";
            for (unsigned i = 0; i < c; ++i) l.idx.push_back(i);
            l.streams = run(prompts, l.idx, l.wall, &l.window, false);
            compare(l, alone);
            levels.push_back(std::move(l));
        }
        hooks.set_g2(true);
    }
    if (n > 1) {
        compare(first, alone);
        levels.push_back(std::move(first));
    }
    std::stable_sort(levels.begin(), levels.end(), [](const SelftestLevel& x, const SelftestLevel& y) { return x.c < y.c; });
    for (const auto& l : levels) report(l);
    }

    // ---- long prompts: chunked prefill == whole-prompt prefill, staggered arrivals, decode stall before / after
    const auto& lp = hooks.long_prompts;
    std::string arms_summary;
    if (!lp.empty() && hooks.set_prefill_chunk && hooks.progress && hooks.prefill_chunk > 0) {
        const std::string chunk = std::to_string(hooks.prefill_chunk);
        const unsigned c = static_cast<unsigned>(lp.size());
        if (set_mtp) set_mtp(false);
        hooks.set_prefill_chunk(0);
        SelftestLevel ref = alone_run(lp, "long C=1 plain whole-prompt");
        valid_reference(ref);
        report(ref);
        hooks.set_prefill_chunk(hooks.prefill_chunk);
        if (!hooks.skip_basic) {
            SelftestLevel chunked = alone_run(lp, "long C=1 plain chunk=" + chunk + " (alone: adaptive, whole prompt)");
            compare(chunked, ref);
            report(chunked);
        }
        if (set_mtp) set_mtp(true);
        if (set_mtp && !hooks.skip_basic) {
            SelftestLevel mc = alone_run(lp, "long C=1 mtp chunk=" + chunk + " (alone: adaptive, whole prompt)");
            compare(mc, ref);
            report(mc);
        }
        std::vector<std::size_t> idx(c);
        for (unsigned i = 0; i < c; ++i) idx[i] = i;
        // staggered arms: whole prompt (before), the server's chunk (after), then the extra chunk sizes
        std::vector<unsigned> arms{0U, hooks.prefill_chunk};
        for (const unsigned x : hooks.extra_chunks)
            if (x != 0 && std::find(arms.begin(), arms.end(), x) == arms.end()) arms.push_back(x);
        std::string summary;
        for (std::size_t a = 0; a < arms.size(); ++a) {
            hooks.set_prefill_chunk(arms[a]);
            SelftestLevel l;
            l.c = c;
            l.label = "long staggered C=" + std::to_string(c) + " every " + std::to_string(hooks.stagger_every) + " rounds, " +
                      (arms[a] ? "chunk=" + std::to_string(arms[a]) : std::string("whole-prompt"));
            l.idx = idx;
            l.streams = run(lp, idx, l.wall, &l.window, true);
            compare(l, ref);
            report(l);
            const double stall = take_trace ? max_prefill_stall_ms(l.streams, last_trace) : 0.0;
            if (a == 0) res.stall_whole_ms = stall;
            if (a == 1) res.stall_chunked_ms = stall;
            std::uint64_t gen = 0;
            for (const auto& st : l.streams) gen += st.stats.generated_tokens;
            char line[320];
            std::snprintf(line, sizeof line, "selftest %s: max decode stall across a prefill %.2f ms, whole run %.2f s %.2f tok/s",
                          l.label.c_str(), stall, l.wall, l.wall > 0 ? static_cast<double>(gen) / l.wall : 0.0);
            out(line);
            char part[96];
            std::snprintf(part, sizeof part, "%s%s %.2f ms / %.2f s", summary.empty() ? "" : ", ",
                          arms[a] ? ("chunk=" + std::to_string(arms[a])).c_str() : "whole-prompt", stall, l.wall);
            summary += part;
        }
        hooks.set_prefill_chunk(hooks.prefill_chunk);
        arms_summary = "selftest staggered arms (max decode stall across another prompt's prefill / whole run): " + summary;
    }
    res.levels = std::move(levels);
    if (!arms_summary.empty()) out(arms_summary);
    out(std::string("selftest RESULT ") + (res.exact ? "EXACT" : "FAIL"));
    return res;
}

}  // namespace strix::glm::serve
