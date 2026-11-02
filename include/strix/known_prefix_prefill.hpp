#pragma once

#include "strix/compiled_speculative_verify.hpp"
#include "strix/speculative_verify_runtime.hpp"
#include <stdexcept>

namespace strix {
struct KnownPrefillBudgetExceeded : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};
struct KnownPrefillRequest {
    std::uint32_t logical_rows=16;
    bool r2_shared_expert=false,r2_gdn=false,r2_gdn_f32_gates=false,routed_expert_union=false;
    KnownWindowOutputPolicy output_policy=KnownWindowOutputPolicy::all_rows;
    // Explicit declared storage limits, not inferred free/physical memory.
    // The native allocation consumer separately admits full owner growth.
    std::uint64_t workspace_budget_bytes=0,host_metadata_payload_budget_bytes=0;
    bool router_rows=false,combine_inject_rows=false;
    // Rows HC mixers and QSA rows in 8-row tiles.
    bool rows_hyper_mix=false,qsa_rows=false;
    // PLE rows from host-staged lookups (all T rows chained, see HipPleKnownWindow prefill).
    bool rows_ple=false;
    // Multi-row LM head (packs of eight rows, the R1 head's leaf and reduction tree per row) for an
    // all-rows request whose every row's logits are read (teacher-forced scoring over the prefill).
    bool lm_head_rows=false;
    // Segmented owners and replay/capture clients can retain the complete legacy layout.
    bool exact_memory=true;
    // Compile-time promise: the owner installs the full-T ring-binding QSA entry.
    // Optional-entry clients retain embedded scores unless they make this promise.
    bool full_qsa_score_ring_entry=false;
    // Zero retains the environment/default contract for existing callers.
    std::uint32_t qsa_score_ring_rows=0;
    // Owner may preserve the complete legacy stage-image ABI.
    bool compact_stage_images=true;
};
struct CompiledKnownPrefixPrefill {
    std::uint32_t logical_rows=0;
    std::uint64_t prefill_execution_identity=0,prefill_kernel_plan_identity=0;
    KnownWindowArena arena;
    std::uint64_t external_qsa_score_bytes=0,qsa_score_ring_workspace_bytes=0;
    std::uint32_t qsa_score_ring_rows=0;
    bool terminal_logits_arena=false;
    std::uint64_t compact_qsa_output_savings=0;
    // Closed selection/format contract only. Its row plans are never relabelled
    // as a T-row window or executed as complete eight-row forwards.
    CompiledKnownPrefixWindow tile8;
    KnownWindowOutputPolicy output_policy=KnownWindowOutputPolicy::all_rows;
    std::vector<std::uint32_t> logit_row_indices;
    std::uint64_t workspace_budget_bytes=0,host_metadata_payload_budget_bytes=0;
    std::uint64_t host_metadata_payload_bytes=0;
};
[[nodiscard]] CompiledKnownPrefixPrefill compile_known_prefix_prefill(
    const ModelContract&,const CompiledSerialExecution&,const ModelManifest&,
    const ExecutionTensorAccessPlan&,const KnownPrefillRequest&);
void validate_compiled_known_prefix_prefill(const CompiledKnownPrefixPrefill&,
    const CompiledSerialExecution&);

// Whole T step-major execution. Every promoted callback still receives exactly
// its original closed 8/4/2 rows. Odd suffixes use full R1 launchers. All T-row
// preflight completes before PLE/mutation; contexts use fixed packs of eight.
// One completed transaction publishes T rows; no rollback is implied on error.
[[nodiscard]] KnownWindowResult execute_known_prefix_prefill_eager(
    const CompiledKnownPrefixPrefill&,const CompiledSerialExecution&,
    const ExecutionRuntimeResources&,const ExecutionMemoryBinding&,SerialSessionContext&,
    std::span<const std::uint32_t>,const CompiledLaunchPlan&,
    std::span<const KnownWindowR2Entry>,std::span<const KnownWindowRowsEntry>,
    const KnownWindowExecutionHooks&,
    // Optional entries that take all T rows of a step in one call (selected by the step's variant).
    std::span<const KnownWindowRowsEntry> full={});

// Segmented tiled prefill: the blocks of several independent sessions in one step-major run over one
// T-row arena. Segment s owns arena rows [offset_s, offset_s + rows_s) (offsets in segment order) with its
// own session, state, position, tokens, PLE history and lineage. A step the caller marks shared (its
// arithmetic per row depends only on that row and the weights) runs once for the rows of every segment
// through its full-T entry; every other step runs per segment exactly as that segment's block alone
// (execute_known_prefix_prefill_eager on a rows_s-row plan): the same entries, row counts and order.
struct KnownPrefillSegment {
    SerialSessionContext* session=nullptr;
    // The session's own resources (state memory and layout, PLE lookup, model objects); the block arena
    // replaces their scratch for the rows of this block.
    const ExecutionRuntimeResources* resources=nullptr;
    std::span<const std::uint32_t> tokens; // 8..T rows, a multiple of 8
    bool head=false;                       // the terminal head of the segment's last row (terminal-row plans)
    // Producer recorded in the session's lineage: the tiled plan of the session's own block of rows_s rows,
    // whose per-row arithmetic the segment repeats (0: this plan's identities).
    std::uint64_t lineage_execution_identity=0,lineage_kernel_plan_identity=0;
};
struct KnownPrefillSegmentHooks {
    // True: step i runs once for the rows of every segment (requires a full-T entry).
    std::function<bool(std::size_t)> shared;
    // After step i of segment s is enqueued (also for shared steps, once per segment): resolver of the
    // segment's own row r (0..rows_s-1).
    std::function<void(std::size_t,std::size_t,const std::function<ExecutionDispatchContext(std::uint32_t)>&)> after_step;
    // HTTP prompt groups only: stage the successor before launching the current
    // segment. One pending stage pairs inline and staged gathers; never verify
    // or reclaim windows.
    bool stage_ple = false;
};
// The same model through possibly distinct owner objects (manifest layout, access plan, tensor bindings).
[[nodiscard]] bool same_model(const ExecutionRuntimeResources&,const ExecutionRuntimeResources&);
// One result per segment, in segment order (logits: the segment's last row when head). Any failure after
// admission poisons every segment's session.
[[nodiscard]] std::vector<KnownWindowResult> execute_known_prefix_prefill_segments_eager(
    const CompiledKnownPrefixPrefill&,const CompiledSerialExecution&,const ExecutionMemoryBinding&,
    std::span<const KnownPrefillSegment>,const CompiledLaunchPlan&,
    std::span<const KnownWindowR2Entry>,std::span<const KnownWindowRowsEntry>,
    const KnownWindowExecutionHooks&,std::span<const KnownWindowRowsEntry> full,const KnownPrefillSegmentHooks&);
} // namespace strix
