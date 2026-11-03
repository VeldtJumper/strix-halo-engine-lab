#pragma once
#include <memory>
#include <string>
#include "strix/hip_mtp_target_hidden.hpp"
#include "strix/mtp_depth_window_controller.hpp"
#include "strix/hip_mtp_target_window.hpp"
#include "strix/hip_mtp_target_checkpoint.hpp"
#include "strix/hip_known_prefix_window.hpp"
namespace strix {
struct HipMtpWindowVerificationPolicy {
    HipMtpTargetWindowCapturePolicy capture;
    SessionSnapshotPolicy snapshot_policy=SessionSnapshotPolicy::occupied_kv_v1;
    std::uint64_t checkpoint_retained_byte_budget=0,total_retained_byte_budget=0;
    bool bounded_window_rollback=false; // explicit candidate, full fallback remains default
    // Adaptive owner choice (prefix owners only): 0 keeps the largest fitting owner.
    float min_probability=0.0F;
    // Owner choice and draft continuation from draft probabilities only (no measured
    // cycle times), so a sampled run is reproducible from its seed.
    bool timing_independent_schedule=false;
    // With STRIX_COST_TABLE_DIR set: the resolved configuration (route, slots, numerical identities) that keys the
    // min-p controller's persisted cost tables (mtp_cost_table_session.hpp); read at the first decode cycle, empty or
    // null = no table. Controller input only.
    std::shared_ptr<const std::string> cost_table_configuration;
};
struct HipMtpVerifiedTargetRows {
    std::vector<HipMtpTargetHiddenTicket> committed_rows;
    std::vector<std::uint32_t> emitted_tokens;
    std::shared_ptr<MtpSelectionReceipt> selection_receipt;
    std::optional<std::string> selection_failure;
    // Candidate inputs actually evaluated (window payload excluding padding, or serial comparisons).
    std::uint32_t verified_candidates=0,accepted_candidates=0;
    bool attempted_window=false,accepted_window=false,restored_checkpoint=false;
    std::uint32_t r2_dispatches=0,projection_launches_saved=0;
    std::uint32_t chosen_rows=1; // window owner rows, 1 = serial target steps
};
// Min-p controller cost samples per owner (rows 1 = serial, per emitted token); cycle_ms_ema is the current
// estimate (EMA with STRIX_MTP_WINDOW_COST=legacy, else robust, D-042); probes = re-probe choices.
struct HipMtpWindowOwnerStats {std::uint32_t rows=0;std::uint64_t cycles=0;double cycle_ms_ema=0.0;std::uint64_t probes=0;};
// Survival controller cost samples: serial rows (1,1), each owner/active-rows option (cycle time minus the
// draft chain), and the draft step (0,0).
struct HipMtpWindowOptionStats {std::uint32_t rows=0,active=0;std::uint64_t samples=0;double ms_ema=0.0;};
// STRIX_MTP_CONTROLLER=survival: the window controller of HipMtpWindowVerifier (prefix owners) chooses
// owner and active rows by calibrated prefix survival instead of the min-p cut (see the verifier source).
[[nodiscard]] bool hip_mtp_survival_controller() noexcept;
class HipMtpWindowVerifier;
// One session of a cohort verification: its verifier, committed frontier and drafts, and the rows
// of the prefix owner it verifies them with (2, 4 or 8; see HipMtpWindowVerifier::cohort_rows).
struct HipMtpCohortVerifyRequest {
    HipMtpWindowVerifier* verifier=nullptr;
    SerialSessionContext* session=nullptr;
    const HipMtpTargetHiddenTicket* initial=nullptr;
    std::span<const std::uint32_t> candidates;
    std::span<const float> probabilities;
    std::uint32_t budget=0;
    std::span<const std::uint32_t> eos;
    std::uint32_t rows=0;
    // Sampled decoding: one draft distribution per candidate (empty when the target selects first-max).
    std::span<const TokenDraft> drafts;
    MtpSelectionTransaction* transaction=nullptr;
};
// One actual closed known-row window per cycle, with retained capture pools per owner.
// Several prefix owners (distinct 2/4/8 rows over one session) share one deferred
// GDN record; per cycle T = 1 + leading drafts with probability >= min_probability,
// rounded up to an owner, then the owner (or serial at T = 1) with the best
// expected tokens per ms under an EMA of measured cycle time.
// Every speculative input except the pending seed must match the preceding
// finite target prediction (with the target's token sampler: the preceding
// row's sampled decision, which uses that input's draft distribution). A rejected/EOS-truncated window is restored before
// serial greedy replay; no provisional hidden row reaches MTP reconciliation.
// STRIX_MTP_DIAG_FULL_CHAIN=1 (diagnostic, never timed): every cycle drafts the full depth without
// a cut and verifies one serial target row, so each emitted position starts a complete draft chain.
[[nodiscard]] bool hip_mtp_full_chain_diagnostic() noexcept;
class HipMtpWindowVerifier final {
public:
    HipMtpWindowVerifier(std::shared_ptr<HipMtpTargetHidden>,std::shared_ptr<HipKnownPrefixWindow>,
        const ExecutionRuntimeResources&,std::shared_ptr<const void> actual_target_resource_owner,
        HipMtpWindowVerificationPolicy,AllocationGenerationSource&);
    HipMtpWindowVerifier(std::shared_ptr<HipMtpTargetHidden>,std::vector<std::shared_ptr<HipKnownPrefixWindow>>,
        const ExecutionRuntimeResources&,std::shared_ptr<const void> actual_target_resource_owner,
        HipMtpWindowVerificationPolicy,AllocationGenerationSource&);
    ~HipMtpWindowVerifier();
    HipMtpWindowVerifier(const HipMtpWindowVerifier&)=delete;
    HipMtpWindowVerifier& operator=(const HipMtpWindowVerifier&)=delete;
    [[nodiscard]] std::shared_ptr<HipMtpTargetHidden> target_owner() const noexcept;
    // Largest owner; logical_rows() is its row count.
    [[nodiscard]] std::shared_ptr<HipKnownPrefixWindow> window_owner() const noexcept;
    [[nodiscard]] std::vector<std::shared_ptr<HipKnownPrefixWindow>> window_owners() const;
    [[nodiscard]] std::vector<HipMtpWindowOwnerStats> owner_stats() const;
    [[nodiscard]] std::uint32_t logical_rows() const noexcept;
    [[nodiscard]] bool failed() const noexcept;
    [[nodiscard]] std::uint64_t retained_payload_bytes() const noexcept;
    [[nodiscard]] HipMtpVerifiedTargetRows verify(SerialSessionContext&,DirectLaunchDispatcher&,
        const HipMtpTargetHiddenTicket& initial,std::span<const std::uint32_t> candidates,
        std::uint32_t output_budget,std::span<const std::uint32_t> eos_tokens,
        std::span<const float> candidate_probabilities={},std::span<const TokenDraft> sampled_drafts={},
        std::span<const float> candidate_runner_ups={},MtpSelectionTransaction* transaction=nullptr);
    [[nodiscard]] bool policy_selection_admitted() const noexcept;
    // Whether one more draft can still pay off. Min-p controller: the next larger window, with the remaining
    // drafts assumed as likely as the last one, beats the best window the current drafts already allow.
    // Survival controller: some longer chain, at the calibrated lookahead acceptance, beats the best option
    // of the drafts so far (plus one exploration draft after a weak first draft).
    [[nodiscard]] bool worth_drafting(std::span<const float> probabilities,std::span<const float> runner_ups={}) const;
    [[nodiscard]] bool survival_controller() const noexcept;
    [[nodiscard]] bool depth_controller() const noexcept;
    [[nodiscard]] std::vector<MtpDepthWindowController::CostStats> depth_cost_stats() const;
    [[nodiscard]] std::vector<MtpDepthWindowController::CostStats> draft_depth_stats() const;
    // Min-p controller cost model: robust estimates with re-probes (default) or the legacy EMA.
    [[nodiscard]] bool robust_cost_model() const noexcept;
    // Start of a decode cycle, before its first draft decision (worth_drafting): resolves the persisted cost prior
    // of this session (sampled or greedy, context bucket). No effect without STRIX_COST_TABLE_DIR.
    void prepare_cycle(const SerialSessionContext& session);
    // The current cycle's draft chain (steps, host wall ms), recorded before verify for the cost split.
    void note_draft(std::uint32_t steps,double ms) noexcept;
    [[nodiscard]] std::vector<HipMtpWindowOptionStats> option_stats() const;
    // The next verify(): candidates from `first` on come from another proposal source with these
    // per-row acceptance estimates (conditional on the earlier rows). They size the window like draft
    // rows but never enter the draft calibration (min-p controller only). Cleared by verify().
    void set_external_rows(std::size_t first,std::span<const double> acceptance);
    // Clears external rows a cohort verification left (verify_cohort calibrates with them but does not clear).
    void clear_external_rows() noexcept;
    // Setup only: every owner's window admission at the session's current frontier, so the
    // first window of each owner does not pay it inside the generation loop.
    void prepare_admission(const SerialSessionContext& session);
    // Cohort verification (prefix owners only). cohort_rows: the owner verify() would size for these
    // drafts without its cost model (1 + leading drafts at or above the minimum probability, at
    // least 2, rounded up to an owner), 0 when no owner fits the budget, context room or lineage
    // history (the session then verifies alone). cohort_fits: whether the owner of `rows` fits.
    [[nodiscard]] std::uint32_t cohort_rows(const SerialSessionContext&,std::span<const std::uint32_t> candidates,
        std::span<const float> probabilities,std::uint32_t output_budget) const;
    [[nodiscard]] bool cohort_fits(const SerialSessionContext&,std::uint32_t rows,std::size_t candidates,
        std::uint32_t output_budget) const;
    [[nodiscard]] std::shared_ptr<HipKnownPrefixWindow> owner_with_rows(std::uint32_t rows) const;
    // Verifies the drafts of several sessions with one cohort window
    // (HipKnownPrefixWindow::execute_mtp_cohort) on primary, an idle owner of the requests' total
    // rows. Per session the inputs, accepted prefix, commit, frontier rewind and published rows are
    // those of verify() with the same owner, so every emitted token is the session's own target
    // greedy token. Owner cost samples are not taken from cohort cycles. Results in request order.
    [[nodiscard]] static std::vector<HipMtpVerifiedTargetRows> verify_cohort(
        std::span<const HipMtpCohortVerifyRequest> requests,HipKnownPrefixWindow& primary);
    // The same over chunks of consecutive requests (HipKnownPrefixWindow::execute_mtp_cohort_chunks): chunk c
    // has chunks[c] requests and runs on primaries[c] (null for a lone request, which runs its own window).
    [[nodiscard]] static std::vector<HipMtpVerifiedTargetRows> verify_cohort(
        std::span<const HipMtpCohortVerifyRequest> requests,std::span<HipKnownPrefixWindow* const> primaries,
        std::span<const std::size_t> chunks);
    // A new session over the same owners after the target state was reset: pending deferred rows
    // of the previous session are dropped. Adaptive policy state starts from its
    // configured prior, retaining the device owners. STRIX_MTP_REQUEST_RESET=0
    // retains the previous session's learned policy for controlled comparisons.
    void begin_session();
    // Known prompt rows (STRIX_PROMPT_WINDOW, the serial suffix of a tiled MTP prompt): `tokens` (2, 4 or 8 known
    // inputs at the session frontier, greedy target) run on the prefix owner of that many rows and every row is
    // committed, as rows accepted by verify() are; each row's arithmetic is the serial row's. Returns the rows'
    // tickets, or none (nothing ran) when no owner, capture pool, context room or lineage history admits them.
    [[nodiscard]] std::vector<HipMtpTargetHiddenTicket> prompt_rows(SerialSessionContext&,std::span<const std::uint32_t> tokens);
    // The known prompt rows of several sessions in one stream of steps (STRIX_PROMPT_WINDOW_COHORT): each session's
    // window is the one prompt_rows() would run alone (same owner, inputs and per-row arithmetic; chunks of two or
    // more sessions share the weight-only steps on an idle owner of their total rows, as verify_cohort does).
    // Results in request order; a session that prompt_rows() would not admit gets no tickets (nothing ran for it).
    struct PromptRowsRequest {HipMtpWindowVerifier* verifier=nullptr;SerialSessionContext* session=nullptr;std::span<const std::uint32_t> tokens;};
    [[nodiscard]] static std::vector<std::vector<HipMtpTargetHiddenTicket>> prompt_rows_cohort(std::span<const PromptRowsRequest> requests);
    // Host-only query: this session has deferred GDN rows to materialize.
    [[nodiscard]] bool pending_window() const noexcept;
    // Wide copy verify (an external known forward over this session, apps/hip_mtp_wide_copy.hpp): before it, apply
    // the pending deferred GDN rows on the owner that executed them, complete its PLE commit stream and invalidate
    // every owner's persistent QSA pools; after it, skip_cycle_sample() keeps the external forward out of the next
    // cycle's measured time. Idle and healthy only; owners, policy state and calibration are retained.
    void flush_pending_window();
    void skip_cycle_sample();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix
