#pragma once

// Prompt-lookup (n-gram) drafting for the gemma4 engine: the index is the engines' shared one (strix/lookup/index.hpp).
#include "strix/lookup/index.hpp"
#include "strix/spec/drafting.hpp"

namespace strix::gemma4::engine::lookup {
using strix::lookup::Index;
using strix::lookup::IndexConfig;
using strix::lookup::make_ngram_index;
using strix::lookup::Match;
using strix::lookup::max_draft;
} // namespace strix::gemma4::engine::lookup

// ------------------------------------------------------------------------------------------------ hybrid session
#include "gemma4/engine/coordinator.hpp"

namespace strix::gemma4::engine::lookup {

// STRIX_G4_LOOKUP: auto (default; also 1/on) = lookup drafting beside the MTP drafter (per-request selector) and on
// its own when no drafter is loaded | 0/off. Read once per runtime.
struct LookupConfig {
    bool enabled = true;
    std::uint32_t min_length = 2;    // STRIX_G4_LOOKUP_MIN (1..16): shortest match proposed
    double margin = 1.0;             // STRIX_G4_LOOKUP_MARGIN (1..2): lookup must beat the alternative by this factor
    std::uint32_t evidence = 4;      // shadow-scored candidates a match bucket needs before it may be used
    bool spec_arbiter = false;       // STRIX_SPEC_ARBITER=1: the shared strix::spec proposer + arbiter decide (C1)
    IndexConfig index;               // STRIX_G4_LOOKUP_TOKENS (capacity)
    [[nodiscard]] static LookupConfig from_environment();
    [[nodiscard]] strix::serve::Json json() const;
};

// Wraps an MTP-style session (or none): every committed token feeds a per-stream Index. At C1 (the confidence-window
// path) each cycle picks ONE linear draft: the inner window (MTP) or a lookup continuation of the best cap, or (no inner
// drafter) lookup vs a plain step. Selection is exact-safe and conservative: the lookup side uses lower-bound prefix
// survival per match-length bucket learned by shadow-scoring every cycle's candidate against the committed stream;
// the alternative uses this stream's measured yield and cycle time (upper bound). Multi-stream / forced windows go to
// the inner session unchanged. Greedy verification decides every token: ids == drafting off.
[[nodiscard]] DraftFactory hybrid_factory(DraftFactory inner, LookupConfig config);
// Runtime admission: lookup wraps the MTP drafter or stands alone without a drafter; DFlash and e4b-eagle sessions are
// never wrapped, and drafting off (--dflash off / STRIX_G4_DFLASH=0) leaves it out entirely.
[[nodiscard]] inline bool wraps(const LookupConfig& config, bool drafter_loaded, bool mtp_drafter, bool drafting_off) {
    return config.enabled && !drafting_off && (!drafter_loaded || mtp_drafter);
}

} // namespace strix::gemma4::engine::lookup
