#pragma once

#include "strix/serial_request.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace strix {

// Physical unit receipt; independent of storage and produced by the incumbent cursor.
struct KnownPromptChunkUnit {
    std::uint64_t begin=0,end=0;
    std::uint32_t owner_rows=0;
    bool t_owner=false,balanced=false,merged=false,deferred=false,headless=false,state_only=false;
    bool operator==(const KnownPromptChunkUnit&) const = default;
};
inline bool known_prompt_full_t_prefix(std::span<const KnownPromptChunkUnit> trace,
    std::uint64_t boundary,std::uint32_t rows) {
    std::uint64_t pos=0;
    for(const auto& u:trace) {
        if(pos==boundary) return boundary!=0;
        if(u.begin!=pos || u.end>boundary || !u.t_owner || u.owner_rows!=rows ||
            u.end-u.begin!=rows || u.balanced || u.merged || u.deferred) return false;
        pos=u.end;
    }
    return boundary && pos==boundary;
}
struct KnownPromptChunkRequest {
    // Borrowed immutable known inputs, including any literal interior EOS.
    // Both backends must retain and use the same actual target session/owners.
    std::span<const std::uint32_t> prompt_tokens;
    std::uint64_t start_position = 0;
    std::uint64_t vocab_size = 0;
    std::uint64_t context_capacity = 0;
    std::uint64_t max_new_tokens = 0;
    std::uint32_t eos_token_id = 0;
    std::uint32_t logical_rows = 2;
    KnownWindowOutputPolicy output_policy = KnownWindowOutputPolicy::all_rows;
    // Explicit separate T16..8192 typed prefill capability. Default2/4/8
    // window contracts and callbacks retain their existing closed widths.
    bool tiled_prefill = false;
    // Tiled all-rows prefill whose every row's logits are read (teacher-forced scoring): no block
    // runs headless. The window backend delivers the non-terminal rows itself; the cursor still
    // keeps only the terminal frontier.
    bool all_row_heads = false;
    // Enable the incumbent deferred/merged/balanced schedule. Observers must decline
    // unsupported boundaries rather than change this numerical plan.
    bool schedule_extensions = false;
    // STRIX_QPF2K_SCHED bit 2 (deferred block completion) for this request; false drops it (qpf2k::cli_sched_defers).
    bool schedule_defer = true;
    // The configured tail ladder (descending) for the S3 eligibility check; empty: the set tail backends' sizes.
    std::vector<std::uint32_t> ladder_tail_rows{};
    GreedyEmissionHook emission_hook{};
    // Set by the cursor at the first unit: S3 balanced blocks apply to this prompt (bitwise equal to the ladder).
};
struct KnownPromptChunkProgress {
    std::uint64_t next_position = 0;
    std::uint64_t consumed_prompt_tokens = 0;
    std::uint64_t restored_prompt_tokens = 0;
    std::uint64_t consumed_generation_tokens = 0;
    std::uint64_t windows_executed = 0;
    std::uint64_t serial_tail_steps = 0;
    std::uint64_t emitted_tokens = 0;
    // Last token actually emitted by generation, never consumed as an input.
    // A merely computed prompt-frontier selection is not an emitted bonus.
    std::optional<std::uint32_t> bonus_pending;
    bool prefill_complete = false;
    bool generation_complete = false;
    bool cancelled = false;
    bool failed = false;
    std::uint64_t state_only_windows = 0;
    // Tail blocks of the optional smaller tiled backend (counted in windows_executed too).
    std::uint64_t tail_windows = 0;
};
struct KnownPromptChunkFrontier {
    // False before any completed chunk and after an opaque backend failure.
    // State-only chunks have a valid state frontier with empty logits/selection.
    bool valid = false;
    std::uint64_t state_position = 0;
    std::optional<std::uint64_t> last_input_position;
    // Retained across pauses. Borrowed until the next mutating cursor call.
    // Empty after generation consumed new inputs: the old prompt logits are
    // deliberately not relabelled as the new target frontier.
    std::span<const float> logits;
    std::optional<std::uint32_t> unconsumed_selection;
    bool selection_emitted = false;
};

// Incremental known-input prefill with complete configured 2/4/8-row windows
// and unpadded serial R1 suffix rows. State-only windows are used only while
// more than R inputs remain; the final <=R inputs use serial calls, preserving
// a real full-output terminal frontier without replaying a consumed input.
// Tiled T16..8192 requires an explicit capable backend; state-only T with
// positive generation budget is rejected, so its terminal frontier is never
// synthesized or replayed. All-row/terminal-row T keeps an actual final head.
// Explicit state-only with max_new_tokens==0 omits every head and consumes
// complete final windows without reserving vocabulary-sized presentation data.
// An advance is one complete chunk; yielding
// between calls pauses without recomputing a prompt or discarding its logits.
// Native shared/mixed-GDN rows4/8 and all-Q8 GDN pair tiling are backend-specific; the cursor
// does not infer a projection amortization from the logical window width.
//
// Backends are borrowed for the entire cursor lifetime. The caller retains all
// backend/session/resource owners and serializes access. Their methods must
// fence/validate completed work and contain errors before returning. cancel()
// is sticky, is called only between chunks and does not destroy those owners.
// On a backend error the cursor cannot resume; its counters describe only
// completed calls, not rollback of an opaque possibly poisoned target.
class KnownPromptChunkCursor final {
public:
    KnownPromptChunkCursor(KnownPromptChunkRequest request,
        KnownPromptWindowBackend& windows, SerialGreedyStepBackend& serial);
    ~KnownPromptChunkCursor();
    KnownPromptChunkCursor(const KnownPromptChunkCursor&) = delete;
    KnownPromptChunkCursor& operator=(const KnownPromptChunkCursor&) = delete;
    // True iff one complete configured window or serial suffix row completed. False after prefill/cancel.
    [[nodiscard]] bool advance();
    // Optional smaller tiled backends for the remainder: after the last complete T-row block, the
    // remaining rows run as complete blocks of the largest tail that fits, and only the rest serially.
    // Same output policy, explicitly tiled, each smaller than T and than the previous tail. Called
    // once per tail size, in descending order, before the first advance.
    void set_tail_backend(KnownPromptWindowBackend& tail);
    // Walk the identical whole-prompt planner without calling a backend or writing state.
    [[nodiscard]] std::vector<KnownPromptChunkUnit> cold_plan() const;
    [[nodiscard]] bool can_resume_at(std::uint64_t boundary) const;
    // Only before execution, after all tail owners are installed; never replan a remainder.
    void resume_at(std::uint64_t boundary);
    [[nodiscard]] const KnownPromptChunkUnit& last_unit() const noexcept {return last_unit_;}
    void cancel() noexcept;
    // Explicit one-shot generation after complete prefill. Delegates the
    // existing greedy continuation's EOS/budget/capacity/bonus policy unchanged.
    // No cancellation is observed inside that existing synchronous call.
    [[nodiscard]] const SerialGreedyRequestResult& finish_greedy();
    [[nodiscard]] const KnownPromptChunkProgress& progress() const noexcept { return progress_; }
    [[nodiscard]] KnownPromptChunkFrontier frontier() const noexcept;
    [[nodiscard]] const std::optional<SerialGreedyRequestResult>& generation() const noexcept { return generation_; }
private:
    KnownPromptChunkRequest request_;
    KnownPromptWindowBackend& windows_;
    std::vector<KnownPromptWindowBackend*> tails_;
    SerialGreedyStepBackend& serial_;
    KnownPromptChunkProgress progress_;
    bool planning_only_ = false;
    KnownPromptChunkUnit last_unit_;
    bool deferred_ = false;
    int balanced_ = -1;     // STRIX_QPF2K_SCHED bit 4: -1 undecided, 0/1 eligibility of this prompt // STRIX_QPF2K_SCHED bit 2: the last block of windows_ awaits its completion checks
    std::vector<float> logits_;
    std::optional<std::uint32_t> selection_;
    std::optional<SerialGreedyRequestResult> generation_;
    bool has_logits_ = false;
};
} // namespace strix
