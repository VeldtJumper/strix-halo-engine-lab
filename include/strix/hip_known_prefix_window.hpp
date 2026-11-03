#pragma once

#include "strix/hip_launcher_catalog.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/speculative_verify_runtime.hpp"
#include "strix/q51_known_rows.hpp"
#include "strix/known_prefix_prefill.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>

namespace strix {
struct HipGreedyRowsReply;
class HipMtpTargetWindowCapture;
struct HipMtpTargetWindowCapturePolicy;
struct HipMtpTargetWindowExecution;
class HipKnownPrefixWindow;
// One session's window in a cohort: its own idle closed owner of the session's row count, its
// session, input tokens and a capture prepared on that owner.
struct HipKnownPrefixCohortMember {
    HipKnownPrefixWindow* window = nullptr;
    SerialSessionContext* session = nullptr;
    std::span<const std::uint32_t> tokens;
    std::shared_ptr<HipMtpTargetWindowCapture> capture;
};
struct HipKnownPrefixUnionObservation {
    std::uint64_t first_position = 0;
    std::uint32_t layer = 0;
    q51_known_rows::Summary group;
};

// Native eager owner for 2/4/8 KNOWN input tokens, not speculative generation.
// The caller's actual_owner retains every borrowed model/state/arena/stream
// allocation and descriptor. Metadata and allocation contents are exclusively
// owned from construction through destruction; no concurrent calls or aliases
// may write them. State changes only through the prescribed execution/PLE path.
//
// If deferred status is enabled, caller initializes the separate eight-byte
// device record to zero before construction. This owner never clears it. The
// constructor checks zero during guarded setup; every successful execute checks
// it again before session publication. A runtime/admission exception permanently
// disables this owner, even when the core correctly leaves a rejected session
// ready. Restore/recovery requires a newly constructed owner and a fresh zero
// status setup. There is no implicit single-token shadow or token counter.
//
// Construction/admission is not compiler, native or whole-model qualification.
class HipKnownPrefixWindow final {
public:
    HipKnownPrefixWindow(CompiledKnownPrefixWindow window,
        CompiledSerialExecution serial_snapshot, ModelContract model,
        HipLauncherCatalogConfig config,
        const ExecutionRuntimeResources& serial_resources,
        ExecutionMemoryBinding window_scratch, HipLaunchContext scope,
        std::shared_ptr<const void> actual_owner,
        q51_known_rows::Workspace routed_union_workspace = {});
    // Explicit T16..16384 (known_prefill_max_rows) step-major owner. Closed window construction above
    // retains its original guards and identities. Every projection tile still
    // uses a closed native 8/2 entry; this is not a general GEMM backend.
    HipKnownPrefixWindow(CompiledKnownPrefixPrefill prefill,
        CompiledSerialExecution serial_snapshot,ModelContract model,HipLauncherCatalogConfig config,
        const ExecutionRuntimeResources& serial_resources,ExecutionMemoryBinding prefill_scratch,HipLaunchContext scope,
        std::shared_ptr<const void> actual_owner,q51_known_rows::Workspace routed_union_workspace = {});
    ~HipKnownPrefixWindow();
    HipKnownPrefixWindow(const HipKnownPrefixWindow&) = delete;
    HipKnownPrefixWindow& operator=(const HipKnownPrefixWindow&) = delete;
    HipKnownPrefixWindow(HipKnownPrefixWindow&&) = delete;
    HipKnownPrefixWindow& operator=(HipKnownPrefixWindow&&) = delete;

    [[nodiscard]] KnownWindowResult execute(SerialSessionContext& session,
        std::array<std::uint32_t, 2> tokens);
    [[nodiscard]] KnownWindowResult execute(SerialSessionContext& session,
        std::span<const std::uint32_t> tokens);
    // Setup-time admission: runs the owner's complete admission and launcher preflights for
    // its own plans at the session's current frontier, enqueues nothing and changes nothing
    // but the owner's preflight memo. Moves the first window's ~10 ms of host admission out of
    // the generation loop; later plans of this owner (MTP capture copies) then re-check only
    // the steps whose launcher differs.
    void preflight_admission(const SerialSessionContext& session, std::span<const std::uint32_t> tokens);
    // Tiled setup only: no prompt reads, launches, PLE lookup or session/state mutation.
    // The owner retains static admission; live token/position checks remain in execute().
    void prepare_prefill_geometry(const SerialSessionContext& session);
    // Tiled prefill block whose logits are not needed: runs every step but the terminal head.
    [[nodiscard]] KnownWindowResult execute_without_head(SerialSessionContext& session,
        std::span<const std::uint32_t> tokens);
    // STRIX_QPF2K_SCHED bit 2: the same headless block without the host fence at its end; its completion checks run when
    // the next block of this owner completes (which must follow before any other use of the session's state) or at
    // finish_pending_completion().
    [[nodiscard]] KnownWindowResult execute_without_head_deferred(SerialSessionContext& session,
        std::span<const std::uint32_t> tokens);
    [[nodiscard]] bool completion_pending() const noexcept;
    [[nodiscard]] hipStream_t producer_stream() const noexcept;
    [[nodiscard]] std::shared_ptr<const void> producer_lifetime() const noexcept;
    // STRIX_QPF2K_SCHED bit 4: a block of the first tokens.size() rows of this tiled owner's T arena (16..T rows, a
    // multiple of 8): with its head (terminal row) or headless; `defer` as execute_without_head_deferred (headless only).
    [[nodiscard]] KnownWindowResult execute_partial(SerialSessionContext& session, std::span<const std::uint32_t> tokens,
        bool head, bool defer = false);
    void finish_pending_completion();
    // A failure while a deferred block is unchecked: the owner fails, its stream is drained (recover or fail-stop).
    void abandon_pending_completion();
    // Optional serving delivery validation AFTER producer/D1 completion but
    // BEFORE publishing any session state/generations. Callback is synchronous
    // on the owned eager scope; throws follow the existing poison/drain path.
    [[nodiscard]] KnownWindowResult execute_validated(SerialSessionContext&,
        std::span<const std::uint32_t>,const std::function<void()>& before_publish);
    // Typed opt-in target capture, closed R2/4/8 all-head windows only.
    // Preparation allocates all snapshot/selector payload before target work.
    // Returned batch proves only the ACTUAL committed whole chunk, not accepted
    // speculative rows or an invented intermediate SerialSessionContext.
    [[nodiscard]] std::shared_ptr<HipMtpTargetWindowCapture> prepare_mtp_target_capture(
        const HipMtpTargetWindowCapturePolicy&);
    [[nodiscard]] HipMtpTargetWindowExecution execute_mtp_captured(SerialSessionContext&,
        std::span<const std::uint32_t>,std::shared_ptr<HipMtpTargetWindowCapture>,
        const std::function<void()>& before_publish = {},std::uint32_t active_rows = 0);
    // active_rows (2..logical_rows, 0 = all): rows from active_rows on are padding. They run the dense
    // projections but stay out of the routed expert union (zero routed output), so the caller must not
    // accept them; every committed prefix must end before active_rows.
    // Opt-in device greedy selection enqueued right after the multi-row terminal
    // head (KnownWindowRequest::lm_head_rows); logits stay materialised. After each
    // successful execute, replies[r] (host-mapped coherent, logical_rows entries)
    // holds row r's first-max finite token/status with no extra copy or sync.
    // partials is retained device scratch of hip_greedy_rows_scratch_bytes(T, vocab).
    void attach_greedy_rows(ExecutionBufferAddress partials, HipGreedyRowsReply* replies);
    // Tiled prefill owners only: observer called after each step of every later block is enqueued
    // (see KnownWindowExecutionHooks::after_step). Set while idle; an empty function removes it.
    // STRIX_BATCH_SERVICE_MS / STRIX_BATCH_CANCEL_POINTS: idle-only scheduling hook. prepare drains/checks
    // current D1 and prior deferred completion without publishing this block.
    using PrefillServicePoint = std::function<void(hipStream_t, const std::function<void()>&)>;
    void set_prefill_service_point(PrefillServicePoint);
    void reset_cancelled_prefill();
    [[nodiscard]] hipStream_t prefill_stream() const noexcept;
    [[nodiscard]] ExecutionMemoryBinding execution_scratch() const noexcept;
    [[nodiscard]] bool aliases_prefill_storage(const ExecutionMemoryBinding&) const noexcept;
    [[nodiscard]] bool aliases_prefill_storage(const HipKnownPrefixWindow&) const noexcept;
    using PrefillStepObserver = std::function<void(std::size_t, const std::function<ExecutionDispatchContext(std::uint32_t)>&)>;
    // observes (optional): the steps the observer reads (it must return at once for the others); empty = every step.
    // STRIX_QPF2K_HC prefill folds stay off for observed steps only.
    void set_prefill_step_observer(PrefillStepObserver observer, std::function<bool(std::size_t)> observes = {});
    // Diagnostics only: the same observer for verify windows (every step after it was enqueued). Set while idle.
    void set_window_step_observer(PrefillStepObserver observer);
    [[nodiscard]] std::uint32_t logical_rows() const noexcept;
    [[nodiscard]] bool supports_tiled_prefill() const noexcept;
    [[nodiscard]] bool gsq_model() const noexcept;
    [[nodiscard]] RotaryIdentity rotary_identity() const;
    // Immutable compiler-selected intent; verification consumers require all_rows.
    [[nodiscard]] KnownWindowOutputPolicy output_policy() const noexcept;
    // Last eligible layer in the last SUCCESSFULLY completed window only.
    // Not aggregate reuse, DRAM traffic, timing or model qualification.
    [[nodiscard]] std::optional<HipKnownPrefixUnionObservation> latest_routed_union_observation() const noexcept;
    // Prefix acceptance (deferred GDN / PLE rows windows). The window executed all rows but only
    // [0, n_keep) are committed; the caller rewinds its session frontier to match.
    [[nodiscard]] bool defers_state() const noexcept;
    [[nodiscard]] bool policy_selection_admitted() const noexcept;
    [[nodiscard]] std::string arithmetic_witness() const;
    void commit_prefix(std::uint32_t n_keep);
    // Host-only query of the deferred GDN record shared by this session's owners.
    [[nodiscard]] bool deferred_pending() const noexcept;
    // Applies pending GDN rows before any ordinary serial token touches GDN state.
    void flush_deferred();
    // Drops pending GDN rows without applying them: the session state they belong to was
    // replaced (a new request starts from a fresh state).
    void discard_deferred();
    // Before either owner executes: adopt primary's deferred GDN rows/control so
    // windows of different sizes over one session continue one pending protocol.
    // flush_deferred() must then run on an owner that has executed a window.
    void share_deferred(const HipKnownPrefixWindow& primary);
    // Cohort of independent sessions: each member's captured window exactly as
    // execute_mtp_captured would run it alone (same admission, same per-row arithmetic, same
    // capture and greedy replies), with the steps that read only weights and per-row activations
    // enqueued once for the rows of all members on primary, an idle owner of another session
    // whose row count is the members' total. Every owner must use one stream. Results are in
    // member order; any failure poisons every member session and disables every owner involved.
    [[nodiscard]] static std::vector<HipMtpTargetWindowExecution> execute_mtp_cohort(HipKnownPrefixWindow& primary,
        std::span<const HipKnownPrefixCohortMember> members);
    // Several cohorts in one stream of steps: chunk c is chunk_members[c] consecutive members, run as a
    // cohort on primaries[c] (an idle owner of the chunk's total rows) when it has two or more members, or
    // as the member's own window (primaries[c] null) when it has one. Step i of every chunk is enqueued
    // before step i + 1 of any; each member's arithmetic is exactly its window alone.
    [[nodiscard]] static std::vector<HipMtpTargetWindowExecution> execute_mtp_cohort_chunks(
        std::span<HipKnownPrefixWindow* const> primaries, std::span<const std::size_t> chunk_members,
        std::span<const HipKnownPrefixCohortMember> members);
    // Segmented tiled prefill (several requests' prompt blocks in one run): segment s is a block of
    // members[s].tokens rows of members[s].session, which must equal the rows of members[s].own (that
    // session's own idle tiled owner of the block, whose resources, profile and lineage identity the segment
    // uses). The segments occupy consecutive rows of this owner's arena (an idle tiled prefill owner of the
    // same execution whose T covers them all). Row-independent steps (STRIX_PROMPT_BATCH_SHARED) run once
    // for every segment's rows; every other step runs per segment exactly as the segment's block alone.
    // head: the segment's last row computes its logits (results[s].logits). observer: like
    // set_prefill_step_observer, with the segment's own rows. Any failure poisons every member session
    // and disables this owner.
    struct PrefillSegmentRequest {
        HipKnownPrefixWindow* own = nullptr;
        SerialSessionContext* session = nullptr;
        std::span<const std::uint32_t> tokens;
        bool head = false;
        PrefillStepObserver observer;
    };
    [[nodiscard]] std::vector<KnownWindowResult> execute_prefill_segments(std::span<const PrefillSegmentRequest> members,
        bool stage_ple = false);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strix
