// Cohort window packing for multi-session decode on the CUDA Qwen engine (host only, no CUDA).
//
// Every decode step gives every live session at least one row (a session is never dropped from a step). The sessions are
// split, in order, into ceil(n / budget) windows of near-equal session counts (budget = the verify window's row cap, 8).
// Inside a window the spare rows (budget - sessions) go to the sessions' wanted extra rows (MTP drafts) round-robin, one
// row per session per round, lowest index first: max-min fair, decode first. A session's rows are a prefix of what it
// wanted (its pending token, then its first drafts), so shrinking only shortens a draft chain.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace strix::cuda::qwen_cohort {

inline constexpr std::uint32_t kWindowRows = 8;       // the default window budget (rows)
inline constexpr std::uint32_t kMaxWindowRows = 16;   // the engine's window cap (9..16: prefill-shaped kernels)
inline constexpr std::uint32_t kMaxWindowSessions = 8;  // sessions per window (engine segments)

struct Placement {
    std::uint32_t index = 0;  // the session's index in the input
    std::uint32_t rows = 0;   // 1 .. want[index]
};
struct Window {
    std::vector<Placement> entries;  // session-major row order
    std::uint32_t rows = 0;
};

// Session counts of the windows of one step: max(ceil(n / budget), ceil(n / 8)) windows (every session >= 1 row, at most 8
// sessions per window), sizes differing by at most one, larger first.
inline std::vector<std::uint32_t> window_sizes(std::uint32_t sessions, std::uint32_t budget = kWindowRows) {
    std::vector<std::uint32_t> sizes;
    if (sessions == 0 || budget == 0) return sizes;
    const std::uint32_t k = std::max((sessions + budget - 1U) / budget, (sessions + kMaxWindowSessions - 1U) / kMaxWindowSessions);
    for (std::uint32_t w = 0; w < k; ++w) sizes.push_back(sessions / k + (w < sessions % k ? 1U : 0U));
    return sizes;
}

// want[i] >= 1: rows session i would use (1 + its drafts); 0 is read as 1.
inline std::vector<Window> pack(const std::vector<std::uint32_t>& want, std::uint32_t budget = kWindowRows) {
    std::vector<Window> out;
    if (budget == 0) return out;
    std::uint32_t next = 0;
    for (const auto size : window_sizes(static_cast<std::uint32_t>(want.size()), budget)) {
        Window w;
        for (std::uint32_t j = 0; j < size; ++j) w.entries.push_back({next + j, 1U});
        std::uint32_t spare = budget - size;
        for (bool grew = true; spare > 0 && grew;) {
            grew = false;
            for (auto& e : w.entries) {
                if (spare == 0) break;
                if (e.rows < std::max<std::uint32_t>(want[e.index], 1U)) { ++e.rows; --spare; grew = true; }
            }
        }
        for (const auto& e : w.entries) w.rows += e.rows;
        next += size;
        out.push_back(std::move(w));
    }
    return out;
}

// Draft depth worth computing for a session whose window holds `window_sessions` sessions: "fair" = its round-robin
// share of the spare rows, ceil((budget - m) / m) (m = 1: budget - 1, the single-session depth); "max" = everything a
// lone drafter could use, budget - m. Never above `depth`.
enum class DepthPolicy { fair, max };
inline DepthPolicy depth_policy_of(const std::string& name) { return name == "max" ? DepthPolicy::max : DepthPolicy::fair; }
inline std::uint32_t draft_depth_cap(std::uint32_t window_sessions, std::uint32_t depth, DepthPolicy policy = DepthPolicy::fair,
                                     std::uint32_t budget = kWindowRows) {
    if (window_sessions == 0 || window_sessions >= budget) return 0;
    const std::uint32_t spare = budget - window_sessions;
    const std::uint32_t cap = policy == DepthPolicy::max ? spare : (spare + window_sessions - 1U) / window_sessions;
    return std::min(cap, depth);
}

// The window size (sessions) of the session at `index` among `sessions` (pack's split).
inline std::uint32_t window_of(std::uint32_t index, std::uint32_t sessions, std::uint32_t budget = kWindowRows) {
    std::uint32_t first = 0;
    for (const auto size : window_sizes(sessions, budget)) {
        if (index < first + size) return size;
        first += size;
    }
    return 0;
}

}  // namespace strix::cuda::qwen_cohort
