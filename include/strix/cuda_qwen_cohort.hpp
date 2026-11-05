// One multi-session decode step on the CUDA Qwen engine (host code over the public Prefill API): MTP draft chains per
// session (depth capped by the session's window share), cohort packing (cuda_qwen_cohort_plan.hpp), then for every
// window verify_cohort + greedy acceptance + commit_cohort. Every committed token and logits row equals the session
// decoding alone (the engine's cohort windows are batch-invariant; drafts only choose what the verify checks).
// Used by strix-cuda-qwen-serve (slot scheduler), qwen_cohort_e2e (gate + bench).
#pragma once

#include "strix/cuda_qwen_cohort_plan.hpp"
#include "strix/cuda_qwen_prefill.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::cuda::qwen_cohort {

namespace qp = strix::cuda::qwen_prefill;

// A decoding session: `pending` is the next input token (the last verified one), at position `position` (= the
// session's length).
struct Stream {
    qp::SessionId session = 0;
    std::uint32_t pending = 0;
    std::uint64_t position = 0;
    std::uint32_t depth_max = 0;  // per-request draft cap (0: the step's depth)
};

inline std::uint32_t window_rows_default() {
    static const std::uint32_t rows = [] {
        const char* v = std::getenv("STRIX_QRTX_WINDOW_ROWS");
        const long n = v ? std::strtol(v, nullptr, 10) : static_cast<long>(kWindowRows);
        return static_cast<std::uint32_t>(std::clamp<long>(n, 1, static_cast<long>(kMaxWindowRows)));
    }();
    return rows;
}

struct StepOptions {
    bool mtp = false;              // draft chains (needs the sidecar)
    std::uint32_t depth = 7;       // draft depth of a lone session
    float min_p = 0.7F;
    DepthPolicy policy = DepthPolicy::fair;
    // window rows: STRIX_QRTX_WINDOW_ROWS (default 8; up to 16, windows of 9..16 rows run the prefill-shaped families);
    // clamped to 1..kMaxWindowRows
    std::uint32_t budget = window_rows_default();
    std::uint64_t max_positions = 0;      // 0: the engine's Config::max_positions
    bool want_logits = false;      // gates: the FP32 rows of the kept tokens
    // Draft chains of several sessions batched by step (Prefill::draft_cohort, same tokens as per-session chains):
    // -1 = default on unless STRIX_QRTX_DRAFT_BATCH=0; 0 = per session; 1 = batched.
    int draft_batch = -1;
};
inline bool draft_batch_default() {
    static const bool on = [] { const char* v = std::getenv("STRIX_QRTX_DRAFT_BATCH"); return !(v && v[0] == '0'); }();
    return on;
}

struct StreamStep {
    std::vector<std::uint32_t> tokens;   // the verified tokens of this step (argmax of the kept rows), >= 1
    std::vector<float> logits;           // [tokens][vocab] when StepOptions::want_logits
    std::uint32_t rows = 0;              // window rows this session used (1 + drafts verified)
    std::uint32_t drafted = 0;           // draft tokens computed (before packing)
    std::uint32_t window_sessions = 0;   // sessions sharing its window
    std::uint32_t window_rows = 0;       // rows of its window
    // The session's rows failed (non-finite logits, or a routed-MoE failure that it raised alone): no tokens; its window
    // was undone, so the session state is still consistent, but the request should end with `error`.
    bool failed = false;
    std::string error;
    bool retried = false;                // its shared window failed and it was re-run alone (isolation)
};

struct StepTiming {
    double draft_ms = 0, verify_ms = 0;
    std::uint32_t windows = 0, rows = 0;
};

// Every stream must have room for at least one row (position < max_positions) and its position must equal the
// engine session's length. Streams are advanced in place. Failure isolation: when a window with several sessions comes
// back with a failed entry (a routed-MoE failure word taints the whole window), every entry is undone (keep 0, exact) and
// each session is re-run alone in its own window; only a session that also fails alone is marked failed. So a healthy
// session's tokens never depend on another session's rows.
inline std::vector<StreamStep> step(qp::Prefill& p, std::vector<Stream>& streams, const StepOptions& opt, StepTiming* timing = nullptr) {
    using Clock = std::chrono::steady_clock;
    StepOptions o = opt;
    o.budget = std::clamp<std::uint32_t>(o.budget, 1U, kMaxWindowRows);
    if (o.max_positions == 0) o.max_positions = p.max_positions();
    const auto n = static_cast<std::uint32_t>(streams.size());
    std::vector<StreamStep> out(n);
    std::vector<std::vector<std::uint32_t>> rows(n);
    std::vector<std::uint32_t> want(n);
    for (const auto& s : streams)
        if (p.length(s.session) != s.position)
            throw std::logic_error("cohort step: stream position " + std::to_string(s.position) + " != session length " +
                                   std::to_string(p.length(s.session)));
    const auto t0 = Clock::now();
    const bool batch = o.draft_batch < 0 ? draft_batch_default() : o.draft_batch != 0;
    std::vector<qp::DraftRequest> requests(n);
    std::uint32_t chains = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
        auto& s = streams[i];
        rows[i].push_back(s.pending);
        const auto m = window_of(i, n, o.budget);
        out[i].window_sessions = m;
        std::uint32_t depth = o.mtp ? draft_depth_cap(m, o.depth, o.policy, o.budget) : 0U;
        if (s.depth_max) depth = std::min(depth, s.depth_max);
        requests[i] = {s.session, s.pending, depth, o.min_p};
        chains += depth > 0;
    }
    // drafts: batched by step across sessions (<= 8 chains per call), or one chain per session
    std::vector<qp::Draft> drafts(n);
    if (batch && chains > 1) {
        for (std::uint32_t first = 0; first < n; first += kWindowRows) {
            const auto last = std::min<std::uint32_t>(n, first + kWindowRows);
            const auto part = p.draft_cohort(std::vector<qp::DraftRequest>(requests.begin() + first, requests.begin() + last));
            for (std::uint32_t i = first; i < last; ++i) drafts[i] = part[i - first];
        }
    } else {
        for (std::uint32_t i = 0; i < n; ++i)
            if (requests[i].depth > 0) drafts[i] = p.draft(requests[i].session, requests[i].pending, requests[i].depth, o.min_p);
    }
    for (std::uint32_t i = 0; i < n; ++i) {
        auto& s = streams[i];
        const auto& d = drafts[i];
        rows[i].insert(rows[i].end(), d.tokens.begin(), d.tokens.end());
        out[i].drafted = static_cast<std::uint32_t>(d.tokens.size());
        const std::uint64_t room = o.max_positions > s.position ? o.max_positions - s.position : 0U;
        want[i] = static_cast<std::uint32_t>(std::min<std::uint64_t>(rows[i].size(), std::max<std::uint64_t>(room, 1U)));
    }
    const auto t1 = Clock::now();
    std::uint32_t windows_run = 0, rows_run = 0;
    // one window: verify, greedy acceptance, commit; false when some entry failed (then every entry was undone)
    const auto run_window = [&](const std::vector<Placement>& entries, bool isolate) {
        std::vector<qp::CohortRows> cohort;
        std::uint32_t total = 0;
        for (const auto& e : entries) {
            cohort.push_back({streams[e.index].session, rows[e.index].data(), e.rows, streams[e.index].position});
            total += e.rows;
        }
        ++windows_run;
        rows_run += total;
        const auto r = p.verify_cohort(cohort, o.want_logits);
        bool all_ok = true;
        for (const auto& x : r) all_ok = all_ok && x.ok;
        if (!all_ok) {
            p.commit_cohort(std::vector<std::uint32_t>(entries.size(), 0U));  // undo every entry
            if (!isolate) {
                for (std::size_t j = 0; j < entries.size(); ++j) {
                    auto& st = out[entries[j].index];
                    st.failed = true;
                    st.error = r[j].ok ? "decode step failed" : r[j].error;
                }
            }
            return false;
        }
        std::vector<std::uint32_t> keep(entries.size());
        for (std::size_t j = 0; j < entries.size(); ++j) {
            const auto& e = entries[j];
            const auto& row = rows[e.index];
            std::uint32_t k = 1;
            while (k < e.rows && row[k] == r[j].argmax[k - 1]) ++k;
            keep[j] = k;
            auto& st = out[e.index];
            st.rows = e.rows;
            st.window_rows = total;
            st.tokens.assign(r[j].argmax.begin(), r[j].argmax.begin() + k);
            if (o.want_logits) st.logits.assign(r[j].logits.begin(), r[j].logits.begin() + static_cast<std::ptrdiff_t>(r[j].logits.size() / e.rows * k));
        }
        p.commit_cohort(keep);
        for (std::size_t j = 0; j < entries.size(); ++j) {
            auto& s = streams[entries[j].index];
            s.position += keep[j];
            s.pending = out[entries[j].index].tokens.back();
        }
        return true;
    };
    for (const auto& w : pack(want, o.budget)) {
        if (run_window(w.entries, w.entries.size() > 1)) continue;
        if (w.entries.size() == 1) continue;  // already marked failed
        for (const auto& e : w.entries) {     // isolation: each session alone
            out[e.index].retried = true;
            (void)run_window({e}, false);
        }
    }
    if (timing) {
        timing->draft_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        timing->verify_ms += std::chrono::duration<double, std::milli>(Clock::now() - t1).count();
        timing->windows += windows_run;
        timing->rows += rows_run;
    }
    return out;
}

}  // namespace strix::cuda::qwen_cohort
