#pragma once
#include "strix/hip_mtp_runtime.hpp"
#include "strix/mtp_selection_transaction.hpp"
#include "strix/mtp_context_draft.hpp"
#include "strix/mtp_ngram_draft.hpp"
#include "strix/mtp_frame_draft.hpp"
#include "strix/mtp_shared_execution_witness.hpp"
namespace strix {
class HipMtpWindowVerifier;
class HipKnownPrefixWindow;
struct HipMtpVerifiedTargetRows;
struct HipMtpGreedyCycleResult {
    std::vector<std::uint32_t> emitted_tokens;
    std::shared_ptr<MtpSelectionReceipt> selection_receipt;
    std::optional<std::string> selection_failure;
    std::vector<std::optional<ModelContentIdentity>> output_logits_digests; // emitted rows, diagnostic only
    std::uint32_t proposed_candidates=0,accepted_candidates=0;
    std::uint64_t consumed_target_rows=0;
    bool eos=false,terminal_budget=false;
    HipMtpTargetHiddenTicket final_target;
    bool attempted_target_window=false,accepted_target_window=false,target_checkpoint_restored=false;
    std::uint32_t target_r2_dispatches=0,target_projection_launches_saved=0;
    std::uint32_t target_window_rows=0; // verifier choice: owner rows, 1 = serial; 0 without verifier
    MtpSharedExecutionCycle shared; // cohort cycles: shared verification window / batched draft launches (witness)
};
// Explicit qualified native MTP consumer, with serial or prepared closed-window target verification.
// Native execution/performance qualification is separate from source admission.
// Each returned token comes from a completed target head. Accepted candidates
// only decide how far to continue; they never replace actual target selection.
// The caller retains exclusive target session/dispatcher/resource ownership.
class HipMtpGreedyRuntime final {
public:
    HipMtpGreedyRuntime(std::shared_ptr<HipMtpRuntime>,
        QualifiedMtpDraftPlan,ActivatedMtpDraftPlan,
        std::shared_ptr<HipMtpTargetHidden>,SerialSessionContext&,
        const ExecutionRuntimeResources&,DirectLaunchDispatcher&,
        std::shared_ptr<const void> actual_target_resource_owner);
    ~HipMtpGreedyRuntime();
    HipMtpGreedyRuntime(const HipMtpGreedyRuntime&)=delete;
    HipMtpGreedyRuntime& operator=(const HipMtpGreedyRuntime&)=delete;
    // Explicit prepared closed-window target verification. Must be configured
    // before the first cycle; absent configuration preserves serial verification.
    void use_window_verifier(std::shared_ptr<HipMtpWindowVerifier>);
    // Draft chains stop after the first draft below this max softmax probability
    // (0 = full depth); the verifier receives every draft probability.
    void set_draft_min_probability(float);
    // At most this many predictor drafts per cycle (0 = no cap; default STRIX_MTP_DRAFT_DEPTH_MAX): drafts only.
    void set_draft_depth_max(std::uint32_t);
    [[nodiscard]] std::uint32_t draft_depth_max() const noexcept;
    // Context n-gram extension of the draft chain (strix/mtp_ngram_draft.hpp): `prompt` is the request's
    // prompt. Needs the window verifier and greedy selection; set before the first cycle. Cohort cycles
    // (begin_cycle_draft without the cost model) do not use it.
    void enable_ngram_draft(std::span<const std::uint32_t> prompt,MtpNgramDraftConfig config={});
    [[nodiscard]] const MtpNgramDraftStats* ngram_stats() const noexcept;
    // Opt-in context copy/n-gram drafter (strix/mtp_context_draft.hpp) instead of the extension above:
    // same prerequisites; it may also replace the chain with a gated context span before any predictor forward.
    // observe_only (after enable_ngram_draft): ordinary cycles keep the extension above unchanged; the context
    // drafter only learns from host shadows of each cycle and supplies propose_wide.
    void enable_context_draft(std::span<const std::uint32_t> prompt,MtpContextDraftConfig config={},bool observe_only=false);
    [[nodiscard]] const MtpContextDraftStats* context_stats() const noexcept;
    // Opt-in strict tool-call frame proposals (strix/mtp_frame_draft.hpp): inside a tool call, the common
    // continuation of the request's earlier frames extends the confident chain like the n-gram extension (and
    // takes precedence over it in that cycle). Same prerequisites as enable_ngram_draft; set before the first cycle.
    void enable_frame_draft(std::span<const std::uint32_t> prompt,MtpFrameDraftConfig config);
    [[nodiscard]] const MtpFrameDraftStats* frame_stats() const noexcept;
    // Prime source positions in order with actual next prompt tokens. The last
    // prompt row is reserved for begin/cycle with its actual target prediction.
    void prime_prompt(std::span<const HipMtpReplayRow>);
    // Fresh serial prompt ingestion retains only the adjacent target tickets;
    // each known next prompt token primes the preceding predictor row. Returns
    // the final prompt row and its actual first generated pending selection.
    [[nodiscard]] HipMtpTargetHiddenTicket prefill_prompt(std::span<const std::uint32_t>);
    // initial_target.predicted_token is an already emitted, unconsumed pending
    // token. Consume it and return only subsequent newly verified outputs.
    // EOS pending consumes nothing; terminal EOS/budget leaves the last output
    // pending. Continue with final_target without emitting that token twice.
    [[nodiscard]] HipMtpGreedyCycleResult cycle(const HipMtpTargetHiddenTicket& initial_target,
        std::uint32_t maximum_candidates,std::uint32_t token_budget,
        std::span<const std::uint32_t> eos_tokens={},MtpSelectionTransaction* transaction=nullptr);
    [[nodiscard]] bool failed() const noexcept;
    [[nodiscard]] const HipMtpRuntime& native_runtime() const noexcept;
    // cycle() in two phases, for a caller that verifies the drafts of several sessions at once
    // (hip_mtp_greedy_cohort_cycle). begin_cycle_draft drafts for initial_target exactly as cycle
    // does (cost_model = false drafts up to the minimum probability only) and returns the draft
    // depth limit; drafting() is false afterwards only when the pending token is EOS (nothing to
    // verify). finish_cycle publishes the target-verified rows as cycle does. An exception between
    // the phases must be followed by fail_cycle().
    std::uint32_t begin_cycle_draft(const HipMtpTargetHiddenTicket& initial_target,
        std::uint32_t maximum_candidates,std::uint32_t token_budget,
        std::span<const std::uint32_t> eos_tokens,bool cost_model);
    // begin_cycle_draft in phases (HipMtpRuntime::propose_chain_begin and the rest), so that a cohort
    // enqueues draft step k of every session before waiting for any: draft_begin (same arguments and
    // result; drafting() false: nothing to draft), then while draft_launch_step() and draft_after_step(),
    // then draft_end(). Each session's launches, waits and drafts are those of begin_cycle_draft.
    std::uint32_t draft_begin(const HipMtpTargetHiddenTicket& initial_target,
        std::uint32_t maximum_candidates,std::uint32_t token_budget,
        std::span<const std::uint32_t> eos_tokens,bool cost_model);
    [[nodiscard]] bool draft_launch_step();
    [[nodiscard]] bool draft_after_step();
    void draft_end();
    [[nodiscard]] bool drafting() const noexcept;
    // Multi-session draft batch (STRIX_MTP_DRAFT_BATCH=1): the native runtime of an open fast greedy predictor chain
    // (after draft_begin), or null; draft_after_step/draft_end keep their meaning for a batched chain.
    [[nodiscard]] HipMtpRuntime* draft_batch_runtime() noexcept;
    // Why draft_batch_runtime() is null (none when it is not); diagnostics only (shared-execution witness).
    [[nodiscard]] MtpDraftBatchDecline draft_batch_decline() const noexcept;
    [[nodiscard]] std::span<const std::uint32_t> drafted_candidates() const noexcept;
    [[nodiscard]] std::span<const float> drafted_probabilities() const noexcept;
    // Candidates from this index on came from a context source (the n-gram extension or the context drafter),
    // SIZE_MAX when none did: their probabilities are learned acceptances, not predictor probabilities.
    [[nodiscard]] std::size_t drafted_external_first() const noexcept;
    // Cohort cycles (opt-in, after draft_end): the context source of the request (enable_ngram_draft or
    // enable_context_draft) continues the confident predictor prefix up to `rows`-1 candidates, keeping only the
    // leading context rows whose learned acceptance is at least `gate` (the chain gate). Returns the context rows
    // added (0: drafts unchanged). The verifier treats them as external rows; the target verifies every row, so
    // emitted tokens never depend on it.
    std::size_t extend_cohort_drafts(std::uint32_t rows,double gate);
    // Cross-request lookup (opt-in): the n-gram extension also continues from the committed context of these
    // other requests' n-gram drafters when its own context has no match (strix/mtp_ngram_draft.hpp set_peers).
    // Pointers are borrowed until the next call; the caller refreshes them before each cycle.
    void set_ngram_peers(std::span<const HipMtpGreedyRuntime* const> peers);
    // Sampled decoding: the distribution of every drafted candidate (empty when the target selects first-max).
    [[nodiscard]] std::span<const TokenDraft> drafted_distributions() const noexcept;
    [[nodiscard]] HipMtpGreedyCycleResult finish_cycle(const HipMtpTargetHiddenTicket& initial_target,
        HipMtpVerifiedTargetRows verified,std::uint32_t token_budget,std::span<const std::uint32_t> eos_tokens);
    void fail_cycle() noexcept;
    // Wide copy verify (opt-in, --mtp-copy-wide-rows; apps/hip_mtp_wide_copy.hpp): the caller verifies rows-1
    // context tokens after initial's pending token with one external rows-row known forward. propose_wide returns
    // them (the context drafter's gated wide span, or `forced` when given: diagnostics only) or empty, and then
    // nothing changed. A non-empty proposal opens the cycle: only finish_wide_cycle, abandon_wide_cycle (or
    // fail_cycle) may follow. It needs an idle healthy owner with the window verifier, greedy selection, a non-EOS
    // pending token, at least `rows` output budget and `rows` free target and predictor positions, and is empty
    // during the cooldown after an abandoned wide cycle (until the committed frontier passes its rejected row).
    [[nodiscard]] std::vector<std::uint32_t> propose_wide(const HipMtpTargetHiddenTicket& initial,std::uint32_t rows,
        std::uint32_t token_budget,std::span<const std::uint32_t> eos_tokens,std::span<const std::uint32_t> forced={});
    // Publishes the externally verified, fully accepted wide cycle: the target session already committed rows 0..k
    // (inputs[0] is initial's pending token, inputs[r] the accepted proposal row r) and final_target is row k's
    // actual ticket (its prediction is emitted.back()). hidden[0] is initial.view() and hidden[r] row r-1's wide
    // tap: the predictor primes the k+1 teacher rows (hidden[r], inputs[r]) as the tiled prompt does; the context
    // drafter finishes the cycle (verified proposal rows, accepted rows) and the draft context observes the outputs.
    [[nodiscard]] HipMtpGreedyCycleResult finish_wide_cycle(const HipMtpTargetHiddenTicket& initial,
        HipMtpTargetHiddenTicket final_target,std::span<const MtpDeviceView> hidden,std::span<const std::uint32_t> inputs,
        std::vector<std::uint32_t> emitted,std::uint32_t verified,std::uint32_t rows,
        std::uint32_t token_budget,std::span<const std::uint32_t> eos_tokens);
    // A rejected wide cycle whose target state the caller restored to the proposal's frontier: nothing is
    // committed and the runtime returns to its state before propose_wide (proposal state, candidates, predictor
    // untouched). `truth` is the target's selections up to and including the rejected row: the context drafter
    // learns from it, and no wide cycle is proposed again until the committed frontier passes that row.
    void abandon_wide_cycle(std::span<const std::uint32_t> truth);
    [[nodiscard]] HipMtpWindowVerifier* window_verifier() const noexcept;
    [[nodiscard]] SerialSessionContext& target_session() const noexcept;
    [[nodiscard]] DirectLaunchDispatcher& target_dispatcher() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// One generation cycle of several independent sessions, each with its own window-verified greedy
// runtime. Every session drafts as in cycle() (up to the minimum probability; no cost-model stop)
// and gets the prefix owner verify() would size for its drafts; sessions are packed into cohorts
// of 4 or 8 rows (a smaller window may be rounded up with padding rows, which are never accepted)
// verified by one cohort window each. A session whose owner does not fit, or that is left alone,
// verifies with its own verify(). Each result is the session's own cycle result: every emitted
// token is that session's target greedy token. An EOS pending token emits nothing (result eos).
struct HipMtpGreedyCohortMember {
    HipMtpGreedyRuntime* runtime=nullptr;
    const HipMtpTargetHiddenTicket* initial=nullptr;
    std::uint32_t maximum_candidates=0,budget=0;
    std::span<const std::uint32_t> eos;
    std::uint32_t slot=0;
    MtpSelectionTransaction* transaction=nullptr;
};
struct HipMtpGreedyCohortStats {std::uint64_t cohort_windows=0,cohort_sessions=0,cohort_rows=0,solo_windows=0,cohort_chunks=0;
    MtpSharedExecutionStats shared;};
class MtpBatchWindowController;
[[nodiscard]] std::vector<HipMtpGreedyCycleResult> hip_mtp_greedy_cohort_cycle(
    std::span<const HipMtpGreedyCohortMember> members,HipMtpGreedyCohortStats* stats=nullptr,
    MtpBatchWindowController* joint=nullptr,MtpBatchWindowController* policy_joint=nullptr);
} // namespace strix
