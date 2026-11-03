#pragma once

#include "strix/ple_lookup.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace strix::ple_detail {

struct StageSource {
    std::uint64_t first = 0, rows = 0, row_bytes = 0, generation = 0;
    const void* runtime = nullptr;
    const void* manifest = nullptr;
    std::uintptr_t address = 0;
};

[[nodiscard]] inline bool same_stage_source(const StageSource& staged, const StageSource& needed) noexcept {
    return needed.rows && staged.first == needed.first && staged.rows >= needed.rows &&
        staged.row_bytes == needed.row_bytes && staged.runtime == needed.runtime && staged.runtime &&
        staged.manifest == needed.manifest && staged.address == needed.address && staged.generation == needed.generation;
}

enum class ServeStageAction { keep, discard, consume };

// A different slot belongs to a later launch. The selected slot must be joined
// before any inline write, even when its source cannot be consumed.
[[nodiscard]] inline ServeStageAction serve_stage_action(const StageSource& staged, const StageSource& needed,
    std::uint32_t stage_slot, std::uint32_t launch_slot) noexcept {
    if(stage_slot!=launch_slot) return ServeStageAction::keep;
    return same_stage_source(staged,needed)?ServeStageAction::consume:ServeStageAction::discard;
}

// Worker 0 is the calling thread on each side. Leave at least one hardware
// thread for the inline gather; zero disables overlap on single-thread hosts.
[[nodiscard]] inline unsigned serve_stage_workers(unsigned hardware) noexcept {
    const auto half=hardware/2U;
    return half<8U?half:8U;
}

// The caller checks worker success, slot, position, source and byte extent before
// calling. Only a successfully finished independent row-0 lookup may be omitted
// from deferred. Rejection leaves the inline gather's work list untouched.
[[nodiscard]] inline std::size_t accept_staged_gather(
    std::span<const PleHashResult> hashes, std::uint32_t heads,
    std::span<const std::uint32_t> staged_ids, bool first_row_consumed,
    std::vector<std::size_t>& deferred) {
    if (hashes.empty() || !heads || heads > PleHashContract::maximum_heads ||
        hashes.size() > staged_ids.size() / heads)
        return 0;
    const std::size_t first_deferred = first_row_consumed ? 1U : 0U;
    if (deferred.size() != hashes.size() - first_deferred) return 0;
    for (std::size_t i = 0; i < deferred.size(); ++i)
        if (deferred[i] != i + first_deferred) return 0;
    // Compare every row, including row 0 even if it was consumed independently.
    for (std::size_t r = 0; r < hashes.size(); ++r) {
        if (hashes[r].row_count != heads) return 0;
        for (std::uint32_t h = 0; h < heads; ++h)
            if (hashes[r].row_ids[h] != staged_ids[r * heads + h]) return 0;
    }
    deferred.clear();
    return hashes.size();
}

[[nodiscard]] inline std::size_t accept_serve_staged_gather(const StageSource& staged, const StageSource& needed,
    std::span<const PleHashResult> hashes, std::uint32_t heads, std::span<const std::uint32_t> ids,
    bool first_row_consumed, std::vector<std::size_t>& deferred) {
    if(!same_stage_source(staged,needed) || needed.rows!=hashes.size()) return 0;
    return accept_staged_gather(hashes,heads,ids,first_row_consumed,deferred);
}

}  // namespace strix::ple_detail
