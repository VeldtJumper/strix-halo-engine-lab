// Host test of the cohort window packing (include/strix/cuda_qwen_cohort_plan.hpp); no GPU.
#include "strix/cuda_qwen_cohort_plan.hpp"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace qc = strix::cuda::qwen_cohort;

static int failures = 0;
#define EXPECT(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

static std::vector<std::uint32_t> rows_of(const std::vector<qc::Window>& ws, std::size_t n) {
    std::vector<std::uint32_t> r(n, 0);
    for (const auto& w : ws) for (const auto& e : w.entries) r[e.index] += e.rows;
    return r;
}

// Invariants of any packing: every session exactly once, 1 <= rows <= max(want, 1), window rows <= budget, windows in input
// order, session-major, and no spare row left while some session in that window still wants more.
static void check_invariants(const std::vector<std::uint32_t>& want, std::uint32_t budget) {
    const auto ws = qc::pack(want, budget);
    std::vector<int> seen(want.size(), 0);
    std::uint32_t expect_index = 0;
    for (const auto& w : ws) {
        std::uint32_t sum = 0;
        bool wants_more = false;
        for (const auto& e : w.entries) {
            EXPECT(e.index == expect_index);
            ++expect_index;
            ++seen[e.index];
            EXPECT(e.rows >= 1);
            EXPECT(e.rows <= std::max<std::uint32_t>(want[e.index], 1U));
            wants_more = wants_more || e.rows < std::max<std::uint32_t>(want[e.index], 1U);
            sum += e.rows;
        }
        EXPECT(sum == w.rows);
        EXPECT(w.rows <= budget);
        EXPECT(w.entries.size() <= qc::kMaxWindowSessions);
        if (wants_more) EXPECT(w.rows == budget);
    }
    for (const int s : seen) EXPECT(s == 1);
    // fairness: inside a window, two sessions that both want more than they got differ by at most one row
    for (const auto& w : ws)
        for (const auto& a : w.entries)
            for (const auto& b : w.entries)
                if (a.rows < want[a.index] && b.rows < want[b.index]) EXPECT(a.rows + 1 >= b.rows && b.rows + 1 >= a.rows);
}

int main() {
    // single session: the whole window
    {
        const auto ws = qc::pack({8});
        EXPECT(ws.size() == 1 && ws[0].rows == 8 && ws[0].entries[0].rows == 8);
        const auto w3 = qc::pack({3});
        EXPECT(w3.size() == 1 && w3[0].rows == 3);
    }
    // two sessions both drafting 7: 4 + 4
    {
        const auto r = rows_of(qc::pack({8, 8}), 2);
        EXPECT(r[0] == 4 && r[1] == 4);
    }
    // three sessions: 3 + 3 + 2 (round-robin, lowest index first)
    {
        const auto r = rows_of(qc::pack({8, 8, 8}), 3);
        EXPECT(r[0] == 3 && r[1] == 3 && r[2] == 2);
    }
    // a short chain frees rows for the others
    {
        const auto r = rows_of(qc::pack({1, 8, 8}), 3);
        EXPECT(r[0] == 1 && r[1] == 4 && r[2] == 3);
        const auto r2 = rows_of(qc::pack({2, 2, 8, 1}), 4);
        EXPECT(r2[0] == 2 && r2[1] == 2 && r2[2] == 3 && r2[3] == 1);
    }
    // eight sessions: one row each, one window
    {
        const auto ws = qc::pack(std::vector<std::uint32_t>(8, 8));
        EXPECT(ws.size() == 1 && ws[0].rows == 8);
        for (const auto& e : ws[0].entries) EXPECT(e.rows == 1);
    }
    // nine sessions: two windows of 5 + 4 sessions, nobody dropped
    {
        const auto ws = qc::pack(std::vector<std::uint32_t>(9, 8));
        EXPECT(ws.size() == 2 && ws[0].entries.size() == 5 && ws[1].entries.size() == 4);
        EXPECT(ws[0].rows == 8 && ws[1].rows == 8);
        const auto sizes = qc::window_sizes(17);
        EXPECT(sizes.size() == 3 && sizes[0] == 6 && sizes[1] == 6 && sizes[2] == 5);
        EXPECT(qc::window_of(0, 9) == 5 && qc::window_of(5, 9) == 4 && qc::window_of(8, 9) == 4);
    }
    // want 0 is one row; empty input is no window
    {
        const auto r = rows_of(qc::pack({0, 0}), 2);
        EXPECT(r[0] == 1 && r[1] == 1);
        EXPECT(qc::pack({}).empty());
    }
    // draft depth caps
    {
        using qc::DepthPolicy;
        EXPECT(qc::draft_depth_cap(1, 7) == 7);
        EXPECT(qc::draft_depth_cap(2, 7) == 3);
        EXPECT(qc::draft_depth_cap(3, 7) == 2);
        EXPECT(qc::draft_depth_cap(4, 7) == 1);
        EXPECT(qc::draft_depth_cap(5, 7) == 1);
        EXPECT(qc::draft_depth_cap(8, 7) == 0);
        EXPECT(qc::draft_depth_cap(2, 7, DepthPolicy::max) == 6);
        EXPECT(qc::draft_depth_cap(1, 3) == 3);
        EXPECT(qc::depth_policy_of("max") == DepthPolicy::max && qc::depth_policy_of("fair") == DepthPolicy::fair);
    }
    // 16-row windows (STRIX_QRTX_WINDOW_ROWS=16): C8 with one draft each in one window; at most 8 sessions per window
    {
        const auto ws = qc::pack(std::vector<std::uint32_t>(8, 2), 16);
        EXPECT(ws.size() == 1 && ws[0].rows == 16);
        for (const auto& e : ws[0].entries) EXPECT(e.rows == 2);
        const auto w10 = qc::pack(std::vector<std::uint32_t>(10, 1), 16);
        EXPECT(w10.size() == 2 && w10[0].entries.size() == 5 && w10[1].entries.size() == 5);
        const auto r4 = rows_of(qc::pack({8, 8, 8, 8}, 16), 4);
        EXPECT(r4[0] == 4 && r4[1] == 4 && r4[2] == 4 && r4[3] == 4);
        EXPECT(qc::draft_depth_cap(8, 7, qc::DepthPolicy::fair, 16) == 1);
        EXPECT(qc::draft_depth_cap(4, 7, qc::DepthPolicy::fair, 16) == 3);
        EXPECT(qc::window_of(9, 12, 16) == 6);
    }
    // randomized invariants
    std::mt19937 rng(12345);
    for (int t = 0; t < 20000; ++t) {
        const std::uint32_t n = rng() % 20;
        const std::uint32_t budget = 1 + rng() % 16;
        std::vector<std::uint32_t> want(n);
        for (auto& w : want) w = rng() % 10;
        check_invariants(want, budget);
    }
    std::printf("%s (%d failures)\n", failures == 0 ? "COHORT PLAN PASS" : "COHORT PLAN FAIL", failures);
    return failures == 0 ? 0 : 1;
}
