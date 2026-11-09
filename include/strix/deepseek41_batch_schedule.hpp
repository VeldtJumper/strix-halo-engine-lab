#pragma once

#include "strix/deepseek41_batch_forward.hpp"

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

// Row planning of BatchForward steps for continuous batching: every request that is decoding contributes one row
// (its last token); the rows left go to requests still reading their prompt, oldest first, in chunks of at most
// `prompt_rows` consecutive prompt tokens. A request at position 0 reads its first token alone (a multi-row segment
// may not start before the request's first compressed entry). The tokens of a row never depend on how the rows were
// planned (see BatchForward), so any plan gives every request its tokens alone.
namespace strix::deepseek41 {

struct BatchSequence {
    std::uint32_t slot = 0;
    std::uint32_t position = 0;           // tokens in the slot's state
    std::vector<std::uint32_t> prompt;
    std::size_t fed = 0;                  // prompt tokens already fed
    std::vector<std::uint32_t> generated;  // selected tokens; the last one is the next row's input
    bool sampled = false;                // sampled requests use target-only batches, never greedy draft verification
    [[nodiscard]] bool reading_prompt() const noexcept { return fed < prompt.size(); }
};

struct PlannedStep {
    std::vector<BatchRow> rows;
    std::vector<std::uint32_t> owner;  // per row: index into the sequences given to plan_step
    std::vector<bool> emits;           // per row: its token is the sequence's next generated token
};

inline PlannedStep plan_step(std::span<BatchSequence* const> sequences, std::uint32_t prompt_rows,
                             std::uint32_t max_rows = BatchForward::max_rows) {
    PlannedStep out;
    for (std::uint32_t i = 0; i < sequences.size() && out.rows.size() < max_rows; ++i) {
        const auto& s = *sequences[i];
        if (s.reading_prompt() || s.generated.empty()) continue;
        out.rows.push_back({s.slot, s.generated.back()});
        out.owner.push_back(i);
        out.emits.push_back(true);
    }
    for (std::uint32_t i = 0; i < sequences.size() && out.rows.size() < max_rows; ++i) {
        const auto& s = *sequences[i];
        if (!s.reading_prompt()) continue;
        std::size_t take = std::min<std::size_t>({prompt_rows, BatchForward::max_rows, max_rows - out.rows.size(), s.prompt.size() - s.fed});
        if (s.position == 0U) take = std::min<std::size_t>(take, 1U);
        for (std::size_t k = 0; k < take; ++k) {
            out.rows.push_back({s.slot, s.prompt[s.fed + k]});
            out.owner.push_back(i);
            out.emits.push_back(s.fed + k + 1U == s.prompt.size());  // the last prompt row gives the first token
        }
    }
    return out;
}

// After the step: advances every planned sequence and appends the emitted tokens.
inline void apply_step(std::span<BatchSequence* const> sequences, const PlannedStep& plan, std::span<const std::uint32_t> tokens) {
    for (std::size_t r = 0; r < plan.rows.size(); ++r) {
        auto& s = *sequences[plan.owner[r]];
        ++s.position;
        if (s.reading_prompt()) ++s.fed;
        if (plan.emits[r]) s.generated.push_back(tokens[r]);
    }
}

}  // namespace strix::deepseek41
