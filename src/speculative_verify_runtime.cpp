#include "strix/host_phase_timing.hpp"
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include "strix/execution_telemetry.hpp"
#include "strix/speculative_verify_runtime.hpp"

#include "strix/dispatch_resources.hpp"
#include "strix/execution_arena.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/runtime_launchers.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <utility>
#include <vector>
#include <stdexcept>

namespace strix {
namespace {

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right,
    const char* message) {
    if (right >
        std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(message);
    }
    return left + right;
}

}  // namespace

SpeculativeVerifyExecutionResult
execute_compiled_speculative_verify_eager(
    const CompiledSpeculativeVerifyExecution& compiled,
    const ExecutionRuntimeResources& resources,
    SerialSessionContext& session,
    std::span<const std::uint32_t> candidate_tokens,
    EagerExecutionDispatcher& dispatcher) {
    if (compiled.execution_identity == 0 ||
        compiled.candidate_depth == 0 ||
        compiled.request_concurrency != 1) {
        throw std::invalid_argument(
            "speculative verify execution is not initialized");
    }
    if (candidate_tokens.size() !=
        compiled.candidate_depth) {
        throw std::invalid_argument(
            "candidate token count does not match compiled verify depth");
    }
    if (compiled.vocab_size == 0) {
        throw std::invalid_argument(
            "speculative verify vocabulary is not initialized");
    }
    for (const auto token : candidate_tokens) {
        if (static_cast<std::uint64_t>(token) >=
            compiled.vocab_size) {
            throw std::invalid_argument(
                "speculative verify candidate token is outside compiled vocabulary");
        }
    }
    if (session.health !=
        SerialSessionHealth::ready) {
        throw std::logic_error(
            "speculative verify requires ready serial session");
    }
    if (session.state_layout_identity == 0 ||
        resources.session_state_layout == nullptr ||
        session.state_layout_identity !=
            resources.session_state_layout->layout_identity) {
        throw std::invalid_argument(
            "speculative verify state layout does not match serial session");
    }

    validate_execution_runtime_resources(
        compiled.execution_identity,
        compiled.arena,
        resources,
        session.next_position);

    const auto depth =
        static_cast<std::uint64_t>(
            compiled.candidate_depth);
    const auto post_position =
        checked_add(
            session.next_position,
            depth,
            "speculative verify position overflow");
    const auto post_committed =
        checked_add(
            session.committed_tokens,
            depth,
            "speculative verify committed-token overflow");

    if (post_position >
        resources.session_state_layout->context_tokens) {
        throw std::out_of_range(
            "speculative verify exceeds allocated session-state capacity");
    }

    EagerExecutionRequest request;
    request.logical_rows =
        compiled.candidate_depth;
    request.request_concurrency = 1;
    request.input_token_id =
        candidate_tokens.front();
    request.input_tokens =
        candidate_tokens;
    request.position =
        session.next_position;
    request.state =
        session.state;

    // All deterministic shape/resource/state validation happens before the
    // first dispatch. After dispatch begins, a failing verify may have mutated
    // target persistent state and therefore poisons the session until restored.
    validate_eager_execution_preflight(
        compiled.topology,
        compiled.bindings,
        compiled.io,
        compiled.arena,
        request,
        resources.session_state_layout,
        resources.session_state_memory,
        &resources);

    SpeculativeVerifyExecutionResult result;
    result.base_position =
        session.next_position;
    result.base_committed_tokens =
        session.committed_tokens;
    result.post_position =
        post_position;
    result.post_committed_tokens =
        post_committed;

    try {
        result.execution =
            execute_serial_eager(
                compiled.topology,
                compiled.bindings,
                compiled.io,
                compiled.arena,
                request,
                dispatcher,
                resources.scratch,
                resources.session_state_layout,
                resources.session_state_memory,
                &resources,
                compiled.execution_identity);
    } catch (...) {
        session.health =
            SerialSessionHealth::poisoned;
        session.arithmetic_lineage = {};
        session.arithmetic_lineage.tainted = true;
        throw;
    }

    session.state =
        result.execution.state;
    session.next_position =
        post_position;
    session.committed_tokens =
        post_committed;
    {
        auto correlation = current_execution_correlation();
        correlation.execution_identity = session.execution_identity;
        correlation.position = result.base_position;
        correlation.row = UINT32_MAX; correlation.rows = compiled.candidate_depth;
        correlation.step_id = UINT32_MAX; correlation.phase = ExecutionPhase::commit;
        ExecutionCorrelationScope commit_scope(correlation);
        observe_execution_activity(ExecutionActivityKind::token_commit, 0, compiled.candidate_depth);
    }
    // This window producer has no qualified canonical lineage recorder yet.
    // Do not retain or relabel the earlier serial history after window mutation.
    session.arithmetic_lineage = {};

    return result;
}

namespace {
constexpr std::array<PersistentState, 6> window_state_flags{
    PersistentState::attention_kv, PersistentState::recurrent_matrix,
    PersistentState::convolution_history, PersistentState::sparse_index,
    PersistentState::ngram_history, PersistentState::position_history};

bool window_reads_position(PersistentState state) noexcept {
    return (static_cast<std::uint32_t>(state) &
        static_cast<std::uint32_t>(PersistentState::position_history)) != 0;
}

void validate_window_launch_plan(const CompiledSerialExecution& row,
    const CompiledLaunchPlan* launch) {
    if (!launch || !launch->plan_identity || launch->execution_identity != row.execution_identity ||
        launch->steps.size() != row.execution.steps.size())
        throw std::invalid_argument("known window requires a fresh launch plan for each row view");
    for (std::size_t i = 0; i < launch->steps.size(); ++i) {
        const auto& entry = launch->steps[i];
        const auto& step = row.execution.steps[i];
        if (entry.step_id != step.id || entry.op != step.op || !entry.function ||
            !entry.implementation_identity || entry.runtime_step != row.bindings.bindings[i].host_or_runtime_step)
            throw std::invalid_argument("known window row launcher does not match the compiled step");
    }
}
struct ActualWindowRecipe {std::uint64_t reduction=0,state_update=0;};
ActualWindowRecipe actual_window_recipe(const CompiledKnownPrefixWindow& window,
    const CompiledSerialExecution& serial,std::span<const CompiledLaunchPlan* const> row_launch,
    const std::vector<const KnownWindowR2Entry*>& selected,
    const std::vector<const KnownWindowRowsEntry*>& selected_rows) {
    const auto word=[](std::uint64_t& hash,std::uint64_t value) {
        for(unsigned shift=0;shift<64U;shift+=8U) {hash^=(value>>shift)&255U;hash*=1099511628211ULL;}
    };
    std::uint64_t reduction=14695981039346656037ULL,state=14695981039346656037ULL;
    word(reduction,0x4b57494e52454331ULL);word(state,0x4b57494e53544131ULL);
    const auto both=[&](std::uint64_t value) {word(reduction,value);word(state,value);};
    both(window.window_execution_identity);both(window.window_kernel_plan_identity);
    both(serial.execution_identity);both(serial.kernel_plan_identity);both(window.logical_rows);
    both(static_cast<std::uint64_t>(window.output_policy));both(window.logit_row_indices.size());
    for(const auto row:window.logit_row_indices) both(row);
    both(row_launch.size());
    for(const auto* launch:row_launch) {
        both(launch->execution_identity);both(launch->plan_identity);both(launch->steps.size());
        for(const auto& step:launch->steps) {
            both(step.step_id);both(static_cast<std::uint64_t>(step.op));both(step.runtime_step);
            both(step.implementation_identity);
        }
    }
    for(std::size_t i=0;i<serial.execution.steps.size();++i) {
        const auto& step=serial.execution.steps[i];
        both(step.id);both(static_cast<std::uint64_t>(step.op));both(step.layer?static_cast<std::uint64_t>(*step.layer)+1U:0U);
        both(static_cast<std::uint64_t>(step.reads_state));both(static_cast<std::uint64_t>(step.writes_state));
        both(static_cast<std::uint64_t>(window.mode[i]));both(static_cast<std::uint64_t>(window.r2_variant[i]));
        const auto implementation=selected_rows[i]?selected_rows[i]->implementation_identity:
            selected[i]?selected[i]->implementation_identity:0U;
        if((selected_rows[i] || selected[i]) && !implementation)
            throw std::invalid_argument("known window lineage requires every selected native recipe identity");
        both(implementation);
        // Bind each source kernel key as well as the actual row implementation.
        const auto& binding=serial.bindings.bindings[i];
        both(binding.host_or_runtime_step);both(binding.kernel_id.has_value());
        if(binding.kernel_id) {both(binding.kernel_id->size());for(const char ch:*binding.kernel_id) both(static_cast<unsigned char>(ch));}
    }
    if(!reduction || !state) throw std::overflow_error("known window actual recipe identity is zero");
    return {reduction,state};
}
}  // namespace

namespace {
// STRIX_WINDOW_TIMING=1: host-side phase totals of every eager known window, printed at exit.
struct WindowHostTiming {
    double admit = 0, preflight = 0, plan = 0, launch = 0, complete = 0; std::uint64_t windows = 0;
    double known_admit = 0, known_preflight = 0, known_plan = 0; std::uint64_t known = 0;
    bool enabled = std::getenv("STRIX_WINDOW_TIMING") != nullptr;
    ~WindowHostTiming() {
        if (enabled && windows)
            std::fprintf(stderr, "window host timing: %llu windows, per window ms: admit %.3f preflight %.3f plan %.3f launch %.3f complete %.3f\n",
                static_cast<unsigned long long>(windows), admit / static_cast<double>(windows), preflight / static_cast<double>(windows), plan / static_cast<double>(windows), launch / static_cast<double>(windows), complete / static_cast<double>(windows));
        if (enabled && known)
            std::fprintf(stderr, "window host timing (repeat windows): %llu, per window ms: admit %.3f preflight %.3f plan %.3f\n",
                static_cast<unsigned long long>(known), known_admit / static_cast<double>(known), known_preflight / static_cast<double>(known), known_plan / static_cast<double>(known));
    }
};
WindowHostTiming& window_host_timing() { static WindowHostTiming t; return t; }
double window_ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}
}  // namespace

struct KnownWindowRun::State {
    const CompiledKnownPrefixWindow& window;
    const CompiledSerialExecution& serial;
    const ExecutionRuntimeResources& serial_resources;
    const ExecutionMemoryBinding& window_scratch;
    SerialSessionContext& session;
    KnownWindowExecutionHooks hooks;
    std::chrono::steady_clock::time_point timing_mark;
    bool admission_only = false, completed = false, failed = false;
    std::uint32_t rows = 0;
    std::size_t steps = 0;
    std::uint64_t post_position = 0, post_committed = 0;
    ArithmeticLineage next_lineage;
    std::optional<ValidatedModelBindingScope> binding_scope;
    std::optional<ArenaLookupScope> arena_lookup;
    std::vector<ExecutionRuntimeResources> resources;
    std::vector<EagerExecutionRequest> requests;
    std::vector<std::vector<ExecutionDispatchContext>> contexts;
    std::vector<const KnownWindowR2Entry*> selected;
    std::vector<const KnownWindowRowsEntry*> selected_rows;
    std::vector<DirectLaunchDispatcher> dispatchers;
    KnownWindowResult result;
    SessionStateAddress position_state{};

    State(const CompiledKnownPrefixWindow& w, const CompiledSerialExecution& s, const ExecutionRuntimeResources& r,
        const ExecutionMemoryBinding& scratch, SerialSessionContext& session_ref, const KnownWindowExecutionHooks& h)
        : window(w), serial(s), serial_resources(r), window_scratch(scratch), session(session_ref), hooks(h) {}

    void materialize_position(const ExecutionDispatchContext& context, std::uint64_t value) const {
        const auto current = resolve_dispatch_state(context, SessionStateComponent::position_history);
        if (current.storage != position_state.storage || current.address != position_state.address ||
            current.bytes != position_state.bytes || current.allocation_generation != position_state.allocation_generation)
            throw std::invalid_argument("known window host position binding changed after admission");
        std::memcpy(reinterpret_cast<void*>(current.address), &value, sizeof(value));
    }
    // Counters and published writes of one promoted step group, whoever enqueued its rows.
    void record_promoted(std::size_t i, std::uint32_t width) {
        const auto& step = serial.execution.steps[i];
        if (selected_rows[i]) {
            if (window.mode[i] == KnownWindowStepMode::shared_rows) ++result.shared_rows_dispatches;
            else if (window.mode[i] == KnownWindowStepMode::gdn_rows) ++result.gdn_rows_dispatches;
            else if (window.mode[i] == KnownWindowStepMode::hyper_mix_rows) ++result.hyper_mix_rows_dispatches;
            else if (window.mode[i] == KnownWindowStepMode::qsa_rows) ++result.qsa_rows_dispatches;
            else if (window.mode[i] == KnownWindowStepMode::lm_head_rows) ++result.lm_head_rows_dispatches;
            else if (window.mode[i] == KnownWindowStepMode::routed_expert_union) ++result.routed_expert_union_dispatches;
        } else ++result.r2_dispatches;
        ++result.execution.kernel_steps;
        ++result.r2_variant_dispatches[static_cast<std::uint32_t>(window.r2_variant[i])];
        ++result.execution.completed_steps; result.execution.completion_order.push_back(step.id);
        for (std::uint32_t r = 0; r < width; ++r) result.execution.state.publish_write(step.writes_state);
    }
};

KnownWindowRun::KnownWindowRun(
    const CompiledKnownPrefixWindow& window, const CompiledSerialExecution& serial,
    const ExecutionRuntimeResources& serial_resources,
    const ExecutionMemoryBinding& window_scratch, SerialSessionContext& session,
    std::span<const std::uint32_t> tokens,
    std::span<const CompiledLaunchPlan* const> row_launch,
    std::span<const KnownWindowR2Entry> r2_entries,
    std::span<const KnownWindowRowsEntry> rows_entries,
    const KnownWindowExecutionHooks& hooks_source, bool caller_scopes)
    : state_(std::make_unique<State>(window, serial, serial_resources, window_scratch, session, hooks_source)) {
    auto& st = *state_;
    const auto& hooks = st.hooks;
    host_phase_mark("5 captured window prologue");
    const auto timing_begin = std::chrono::steady_clock::now();
    auto& timing_mark = st.timing_mark;
    timing_mark = timing_begin;
    // A repeat window of the same owner and launch plans re-checks only what changes per
    // window (tokens, positions, capacity, session); its static admission already passed.
    const void* preflight_key = row_launch.empty() ? nullptr : static_cast<const void*>(row_launch.front());
    const bool preflight_known = hooks.preflight_memo && preflight_key &&
        std::find(hooks.preflight_memo->begin(), hooks.preflight_memo->end(), preflight_key) != hooks.preflight_memo->end();
    const auto rows = window.logical_rows;
    st.rows = rows;
    const auto slot_count = serial.execution.steps.size() * rows;
    // Another plan of this owner already passed the full admission (see preflight_slots).
    const bool owner_known = !preflight_known && hooks.preflight_slots && hooks.preflight_slots->size() == slot_count;
    const bool static_known = preflight_known || owner_known;
    // Every allocation, clone, mode and identity is checked before callbacks.
    if (!static_known) validate_compiled_known_prefix_window(window, serial);
    if(tokens.size()!=rows || row_launch.size()!=rows)
        throw std::invalid_argument("known window token/launch row count differs from its compiled geometry");
    if (!hooks.preflight_scope || !hooks.complete || !hooks.quiesce_or_fail_stop)
        throw std::invalid_argument("known window requires deep admission, owned scope, completion and fail-stop hooks");
    if (session.health != SerialSessionHealth::ready)
        throw std::logic_error("known window requires a ready serial session");
    if (session.execution_identity != serial.execution_identity || !session.execution_identity)
        throw std::invalid_argument("known window serial session execution identity mismatch");
    if (!static_known) validate_execution_runtime_resources(serial.execution_identity, serial.arena,
        serial_resources, session.next_position);
    const auto identities = static_known ? RuntimeExecutionIdentities{} : build_runtime_execution_identities(
        serial.execution_identity, serial.arena, serial_resources);
    if (!session.runtime_compatibility_identity ||
        (!static_known && session.runtime_compatibility_identity != identities.compatibility_identity) ||
        session.state_layout_identity != serial_resources.session_state_layout->layout_identity ||
        window.manifest_layout_identity != serial_resources.model_manifest->layout_identity ||
        window.tensor_access_plan_identity != serial_resources.tensor_access->plan_identity)
        throw std::invalid_argument("known window runtime/session/manifest identity mismatch");
    for (const auto token : tokens)
        if (!serial.vocab_size || static_cast<std::uint64_t>(token) >= serial.vocab_size)
            throw std::invalid_argument("known window input token is outside the compiled vocabulary");
    const auto post_position = checked_add(session.next_position, rows, "known window position overflow");
    const auto post_committed = checked_add(session.committed_tokens, rows, "known window committed-token overflow");
    st.post_position = post_position; st.post_committed = post_committed;
    if (post_position > serial.max_position_embeddings ||
        post_position > serial_resources.session_state_layout->context_tokens)
        throw std::out_of_range("known window exceeds allocated context capacity");
    auto& next_lineage = st.next_lineage;
    if(!hooks.record_actual_window_lineage)
        next_lineage=prepare_prefill_chunk_lineage(session.arithmetic_lineage,
            session.next_position,post_position,window.window_execution_identity,
            window.window_kernel_plan_identity,window.window_execution_identity);
    if (!static_known) {
    validate_execution_memory_binding(window.window_execution_identity, window.arena.physical, window_scratch);
    const auto& old_scratch = *serial_resources.scratch;
    if (window_scratch.arena_bytes > std::numeric_limits<std::uintptr_t>::max() - window_scratch.arena_base ||
        old_scratch.arena_bytes > std::numeric_limits<std::uintptr_t>::max() - old_scratch.arena_base)
        throw std::invalid_argument("known window full scratch owner extent overflows");
    if (window_scratch.arena_base < old_scratch.arena_base + old_scratch.arena_bytes &&
        old_scratch.arena_base < window_scratch.arena_base + window_scratch.arena_bytes)
        throw std::invalid_argument("known window and live serial scratch arenas must be disjoint");
    const auto disjoint_from_window = [&](std::uintptr_t base, std::uint64_t bytes) {
        if (!bytes) return;
        if (!base || bytes > std::numeric_limits<std::uintptr_t>::max() - base)
            throw std::invalid_argument("known window borrowed owner extent overflows");
        if (window_scratch.arena_base < base + bytes && base < window_scratch.arena_base + window_scratch.arena_bytes)
            throw std::invalid_argument("known window scratch aliases persistent state or model weights");
    };
    const auto& persistent = *serial_resources.session_state_memory;
    disjoint_from_window(persistent.device_base, persistent.device_bytes);
    disjoint_from_window(persistent.host_base, persistent.host_bytes);
    for (const auto& weight : serial_resources.model_weights->tensors)
        if (weight.address) disjoint_from_window(weight.address, weight.bytes);
    }
    if (serial_resources.ple_prefetch && serial_resources.ple_prefetch->active)
        throw std::invalid_argument("known window cannot borrow an active PLE lookup");
    // Every row context binds these model objects, unchanged until the window returns: its
    // admission and launch lookups skip re-validating the whole binding (about 7 ms of host time
    // per window before), and row arena plans are answered from an index instead of searched
    // linearly per buffer (the owner's persistent index when it has one). A cohort caller opens
    // both scopes once around all of its windows instead (caller_scopes).
    if (!caller_scopes) {
        if (!serial_resources.distributed_stage)
            st.binding_scope.emplace(*serial_resources.model_manifest, *serial_resources.model_weights);
        if (hooks.arena_index) st.arena_lookup.emplace(*hooks.arena_index); else st.arena_lookup.emplace();
    }
    auto& resources = st.resources;
    auto& requests = st.requests;
    resources.reserve(rows);
    for (std::size_t r = 0; r < rows; ++r) resources.push_back(serial_resources);
    requests.resize(rows);
    for (std::size_t r = 0; r < rows; ++r) {
        resources[r].scratch = &window_scratch;
        requests[r].input_token_id = tokens[r];
        requests[r].position = session.next_position + r;
        requests[r].state = session.state;
        if (preflight_known) continue;
        if (!owner_known) validate_execution_runtime_resources(window.window_execution_identity, window.arena.row_view[r],
            resources[r], session.next_position + r);
        validate_window_launch_plan(window.row_plan[r], row_launch[r]);
        for (std::size_t i = 0; i < serial.execution.steps.size(); ++i) {
            const auto& step = serial.execution.steps[i];
            if (!window_reads_position(step.reads_state)) continue;
            const auto implementation = row_launch[r]->steps[i].implementation_identity;
            const bool position = step.id == 1 && step.op == SerialOp::position_prepare &&
                row_launch[r]->steps[i].runtime_step && implementation == position_prepare_launcher_identity &&
                row_launch[r]->steps[i].function == &launch_position_prepare;
            const bool qsa = step.op == SerialOp::indexed_attention && !row_launch[r]->steps[i].runtime_step &&
                implementation != position_prepare_launcher_identity &&
                implementation != position_verify_window_launcher_identity &&
                std::find(hooks.position_reader_implementations.begin(),
                    hooks.position_reader_implementations.end(), implementation) != hooks.position_reader_implementations.end();
            if (!position && !qsa)
                throw std::invalid_argument("known window position-history reader implementation is not admitted");
        }
        if (!owner_known) validate_eager_execution_preflight(window.row_plan[r].execution, window.row_plan[r].bindings,
            window.row_plan[r].io, window.row_plan[r].arena, requests[r],
            resources[r].session_state_layout, resources[r].session_state_memory, &resources[r]);
    }
    std::array<std::uint64_t, 6> pending_writes{};
    const auto steps = serial.execution.steps.size();
    st.steps = steps;
    if (steps > std::numeric_limits<std::uint32_t>::max() / rows)
        throw std::overflow_error("known window logical row-step count overflow");
    auto& contexts = st.contexts;
    contexts.resize(steps);
    for (auto& step_rows : contexts) step_rows.resize(rows);
    auto& selected = st.selected;
    auto& selected_rows = st.selected_rows;
    selected.assign(steps, nullptr);
    selected_rows.assign(steps, nullptr);
    std::array<bool, known_window_r2_variant_slots> seen{};
    for (const auto& entry : r2_entries) {
        const auto id = static_cast<std::uint32_t>(entry.variant);
        if (id == 0 || id > 3U || seen[id] || !entry.config || !entry.preflight || !entry.launch)
            throw std::invalid_argument("known window R2 entry table is invalid or ambiguous");
        seen[id] = true;
    }
    for(const auto& entry:rows_entries) {
        const auto id=static_cast<std::uint32_t>(entry.variant);
        if(id<4U || id>=seen.size() || seen[id] || !entry.config || !entry.preflight || !entry.launch)
            throw std::invalid_argument("known window wide entry table is invalid or ambiguous");
        seen[id]=true;
    }
    auto& result = st.result;
    result.logits.resize(window.logit_row_indices.size());
    result.logit_row_indices=window.logit_row_indices;
    result.execution.state = session.state;
    result.execution.completion_order.reserve(rows * steps);
    result.logical_row_steps = static_cast<std::uint32_t>(rows * steps);
    result.omitted_lm_heads=rows-static_cast<std::uint32_t>(window.logit_row_indices.size());
    result.logical_row_steps-=result.omitted_lm_heads;
    result.logical_projection_weight_bytes_avoided = window.logical_projection_weight_bytes_avoided;
    result.projection_launches_saved = window.projection_launches_saved;
    for (std::size_t i = 0; i < steps; ++i) {
        const auto& step = serial.execution.steps[i];
        for (std::size_t flag = 0; flag < window_state_flags.size(); ++flag)
            if ((static_cast<std::uint32_t>(step.writes_state) &
                 static_cast<std::uint32_t>(window_state_flags[flag])) != 0)
                pending_writes[flag] = checked_add(pending_writes[flag], rows, "known window state-write count overflow");
        for (std::size_t r = 0; r < rows; ++r) {
            auto& c = contexts[i][r];
            const auto& plan = window.row_plan[r];
            const auto& io = plan.io.steps[i];
            const auto& binding = plan.bindings.bindings[i];
            c.execution_identity = window.window_execution_identity;
            c.step_id = step.id; c.op = step.op; c.layer = step.layer;
            c.reads = io.reads; c.writes = io.writes; c.temporaries = io.temporaries;
            c.arena = &plan.arena; c.memory = &window_scratch;
            c.session_state_layout = resources[r].session_state_layout;
            c.session_state_memory = resources[r].session_state_memory;
            c.resources = &resources[r]; c.reads_state = step.reads_state; c.writes_state = step.writes_state;
            c.position = requests[r].position; c.input_token_id = tokens[r];
            if (!binding.host_or_runtime_step) c.kernel_id = std::string_view(*binding.kernel_id);
        }
        if (window.mode[i] != KnownWindowStepMode::r1_row_loop) {
            // QSA rows read each row's own context position, never the host word.
            if (window_reads_position(step.reads_state) && window.mode[i]!=KnownWindowStepMode::qsa_rows)
                throw std::invalid_argument("known window promoted entries cannot consume host position history");
            if(known_window_rows_mode(window.mode[i])) {
                const auto found=std::find_if(rows_entries.begin(),rows_entries.end(),
                    [&](const KnownWindowRowsEntry& e){return e.variant==window.r2_variant[i];});
                if(found==rows_entries.end()) throw std::invalid_argument("known window wide implementation is missing");
                selected_rows[i]=&*found;
            } else {
                const auto found = std::find_if(r2_entries.begin(), r2_entries.end(),
                    [&](const KnownWindowR2Entry& e) { return e.variant == window.r2_variant[i]; });
                if (found == r2_entries.end()) throw std::invalid_argument("known window R2 implementation is missing");
                selected[i] = &*found;
            }
        }
    }
    for (std::size_t flag = 0; flag < window_state_flags.size(); ++flag)
        if (pending_writes[flag] > std::numeric_limits<std::uint64_t>::max() - session.state.generations[flag])
            throw std::overflow_error("known window persistent-state generation overflow");
    if (steps == 0 || serial.execution.steps.back().op != SerialOp::lm_head)
        throw std::invalid_argument("known window requires every configured terminal logit row");
    for (std::size_t r = 0; r < rows; ++r) {
        if (contexts.back()[r].writes.size() != 1)
            throw std::invalid_argument("known window terminal head requires one output");
        // Admission remains complete even for a deliberately omitted output.
        const auto logits = resolve_dispatch_write(contexts.back()[r], 0);
        if (logits.bytes < serial.vocab_size * sizeof(float))
            throw std::invalid_argument("known window terminal logit row is undersized");
        const auto selected_logit_row=std::find(window.logit_row_indices.begin(),window.logit_row_indices.end(),r);
        if(selected_logit_row!=window.logit_row_indices.end())result.logits[static_cast<std::size_t>(selected_logit_row-window.logit_row_indices.begin())]=logits;
    }
    st.position_state = resolve_dispatch_state(contexts.front()[0], SessionStateComponent::position_history);
    const auto& position_state = st.position_state;
    if (position_state.storage != SessionStateStorage::host || !position_state.address ||
        position_state.bytes < sizeof(std::uint64_t) ||
        position_state.bytes > std::numeric_limits<std::uintptr_t>::max() - position_state.address)
        throw std::invalid_argument("known window requires writable bounded host position history");
    { const auto ms = window_ms_since(timing_mark); window_host_timing().admit += ms; if (preflight_known) window_host_timing().known_admit += ms; }
    timing_mark = std::chrono::steady_clock::now();
    host_phase_mark("6a admit");
    hooks.preflight_scope();
    // A missing callback is not an implicit permit. Every row-loop launcher
    // proves its current metadata before PLE can start or any word is written.
    std::vector<std::array<std::uintptr_t, 2>> proven;
    if (!preflight_known && hooks.preflight_slots) proven.assign(slot_count, {0U, 0U});
    // True when slot (i, r) needs its preflight: unproven for this owner, or a different launcher.
    const auto needs = [&](std::size_t i, std::size_t r, auto function, const void* data) {
        const std::array<std::uintptr_t, 2> key{reinterpret_cast<std::uintptr_t>(function), reinterpret_cast<std::uintptr_t>(data)};
        if (!proven.empty()) proven[i * rows + r] = key;
        return !owner_known || (*hooks.preflight_slots)[i * rows + r] != key;
    };
    for (std::size_t i = 0; i < steps && !preflight_known; ++i) {
        if(selected_rows[i]) {
            if (needs(i, 0, selected_rows[i]->preflight, selected_rows[i]->config))
                selected_rows[i]->preflight(contexts[i],selected_rows[i]->config);
        } else if (selected[i]) {
            for(std::size_t r=0;r<rows;r+=2U)
                if (needs(i, r, selected[i]->preflight, selected[i]->config))
                    selected[i]->preflight(std::span<const ExecutionDispatchContext,2>(contexts[i].data()+r,2U),selected[i]->config);
        } else for (std::size_t r = 0; r < rows; ++r) {
            const auto& launcher = row_launch[r]->steps[i];
            if (!launcher.preflight)
                throw std::invalid_argument("known window row launcher lacks pure metadata preflight");
            if (needs(i, r, launcher.preflight, launcher.user_data)) launcher.preflight(contexts[i][r], launcher.user_data);
        }
    }
    { const auto ms = window_ms_since(timing_mark); window_host_timing().preflight += ms; if (preflight_known) window_host_timing().known_preflight += ms;
      if (window_host_timing().enabled && !preflight_known)
          std::fprintf(stderr, "window admission rows=%u owner_known=%d setup=%d: admit+preflight %.3f ms (preflight %.3f)\n", rows,
              owner_known ? 1 : 0, hooks.preflight_only ? 1 : 0, window_ms_since(timing_begin), ms); }
    host_phase_mark("6b row preflight");
    timing_mark = std::chrono::steady_clock::now();
    // execute() validated the owner metadata just before this call; with every preflight
    // skipped nothing ran since, so a memoized window does not repeat it.
    if (hooks.preflight_plan && !preflight_known) hooks.preflight_plan();
    if (hooks.preflight_memo && !preflight_known) hooks.preflight_memo->push_back(preflight_key);
    // Slots this admission proved; a later plan of the owner compares against the newest ones.
    if (!proven.empty()) *hooks.preflight_slots = std::move(proven);
    if (hooks.preflight_only) { st.admission_only = true; return; }
    { const auto ms = window_ms_since(timing_mark); window_host_timing().plan += ms;
      if (preflight_known) { window_host_timing().known_plan += ms; ++window_host_timing().known; } }
    timing_mark = std::chrono::steady_clock::now();
    host_phase_mark("6c preflight plan");
    if(hooks.record_actual_window_lineage) {
        const std::pair<const void*,std::array<std::uint64_t,2>>* memo=nullptr;
        if(preflight_known && hooks.recipe_memo) for(const auto& entry:*hooks.recipe_memo) if(entry.first==preflight_key) memo=&entry;
        ActualWindowRecipe recipe;
        if(memo) recipe={memo->second[0],memo->second[1]};
        else {
            recipe=actual_window_recipe(window,serial,row_launch,selected,selected_rows);
            if(hooks.recipe_memo && preflight_key) hooks.recipe_memo->push_back({preflight_key,{recipe.reduction,recipe.state_update}});
        }
        next_lineage=prepare_known_window_chunk_lineage(session.arithmetic_lineage,session.next_position,post_position,
            window.window_execution_identity,recipe.reduction,recipe.state_update);
        if(!arithmetic_lineage_valid(next_lineage,post_position))
            throw std::invalid_argument("recorded known window requires complete actual preceding producer history");
    }
    host_phase_mark("6d lineage recipe");
    auto& dispatchers = st.dispatchers;
    dispatchers.reserve(rows);
    for(const auto* plan:row_launch) dispatchers.emplace_back(*plan);
    // As in the serial session API, a lookup start that returns no handle or
    // throws before host/device mutation is an admission/start failure, not a
    // poisoned target. The backend owns cancellation of any failed start.
    host_phase_mark("6 window admission");
    if (serial_resources.ple_lookup && serial_resources.ple_hash_contract && serial_resources.ple_prefetch) {
            const auto ple = std::find_if(serial.execution.steps.begin(), serial.execution.steps.end(),
                [](const ExecutionStep& step) { return step.op == SerialOp::ple; });
            if (ple == serial.execution.steps.end()) throw std::logic_error("known window has PLE resources without a PLE step");
            start_ple_lookup_prefetch(*serial_resources.ple_hash_contract, ple->id, tokens[0], session.next_position,
                *serial_resources.model_manifest, *serial_resources.model_weights, *serial_resources.tensor_access,
                *serial_resources.session_state_layout, *serial_resources.session_state_memory,
                *serial_resources.ple_lookup, *serial_resources.ple_prefetch, false);
    }
}

KnownWindowRun::~KnownWindowRun() = default;
bool KnownWindowRun::admission_only() const noexcept { return state_->admission_only; }
std::size_t KnownWindowRun::steps() const noexcept { return state_->steps; }
std::uint32_t KnownWindowRun::rows() const noexcept { return state_->rows; }
std::span<const ExecutionDispatchContext> KnownWindowRun::step_rows(std::size_t i) const { return state_->contexts.at(i); }
bool KnownWindowRun::wide_step(std::size_t i) const { return state_->selected_rows.at(i) != nullptr; }
bool KnownWindowRun::promoted_step(std::size_t i) const {
    return state_->selected_rows.at(i) != nullptr || state_->selected.at(i) != nullptr;
}

void KnownWindowRun::launch_step(std::size_t i) {
    auto& st = *state_;
    if (st.admission_only || st.completed || st.failed || i >= st.steps)
        throw std::logic_error("known window step launch outside an admitted window");
    const auto rows = st.rows;
    const auto steps = st.steps;
    const auto& window = st.window;
    const auto& serial = st.serial;
    const auto& serial_resources = st.serial_resources;
    const auto& hooks = st.hooks;
    auto& contexts = st.contexts;
    auto& selected = st.selected;
    auto& selected_rows = st.selected_rows;
    auto& result = st.result;
    const auto& step = serial.execution.steps[i];
    if (!result.execution.state.contains(step.reads_state))
        throw std::logic_error("known window persistent state disappeared after admission");
    // Diagnostic observer (KnownWindowExecutionHooks::after_step): called after the step was enqueued with
    // its per-row contexts. Absent in production.
    const auto observe = [&] {
        if (!hooks.after_step) return;
        hooks.after_step(i, [&](std::uint32_t row) {
            if (row >= rows) throw std::invalid_argument("known window observer row is outside the window");
            return contexts[i][row]; });
    };
    if (hooks.before_step && (selected[i] || selected_rows[i]) && hooks.before_step(i, contexts)) {
        const auto width = selected_rows[i] ? rows : 2U;
        for (std::uint32_t first = 0; first < rows; first += width) st.record_promoted(i, width);
        observe();
        return;
    }
    if (hooks.before_step && !selected[i] && !selected_rows[i]) (void)hooks.before_step(i, contexts);
    if (selected[i] || selected_rows[i]) {
        const auto width=selected_rows[i] ? rows : 2U;
        for(std::uint32_t first=0;first<rows;first+=width) {
            std::optional<ExecutionCorrelationScope> joint_scope;
            std::optional<ExecutionPhaseScope> joint_elapsed;
            if (current_execution_telemetry()) {
                auto correlation=current_execution_correlation();
                correlation.execution_identity=serial.execution_identity;
                correlation.position=contexts[i][first].position;
                correlation.rows=width; correlation.row=UINT32_MAX;
                correlation.step_id=step.id; correlation.op=step.op;
                correlation.layer=step.layer.value_or(UINT32_MAX);
                correlation.phase=ExecutionPhase::operator_step;
                joint_scope.emplace(correlation);
                joint_elapsed.emplace(ExecutionPhase::operator_step);
            }
            if(selected_rows[i]) selected_rows[i]->launch(contexts[i],selected_rows[i]->config);
            else selected[i]->launch(std::span<const ExecutionDispatchContext,2>(contexts[i].data()+first,2U),selected[i]->config);
            st.record_promoted(i, width);
        }
        observe();
        return;
    }
    for (std::size_t r = 0; r < rows; ++r) {
        const auto& context = contexts[i][r];
        // The compiler proved this is the sole terminal, state-free
        // head. Its full launcher preflight already ran above for
        // every row, before PLE or any model/state mutation.
        if(i+1U==steps && step.op==SerialOp::lm_head &&
            !known_window_produces_logits(window.output_policy,static_cast<std::uint32_t>(r),rows))continue;
        std::optional<ExecutionCorrelationScope> row_scope;
        if (current_execution_telemetry()) {
            auto correlation = current_execution_correlation();
            correlation.row = static_cast<std::uint32_t>(r); correlation.rows = 1;
            row_scope.emplace(correlation);
        }
        // Host position history is one word. Materialise the row's
        // own value before every declared reader; this changes no
        // semantic write count and never exposes the future row.
        if (window_reads_position(step.reads_state))
            st.materialize_position(context, context.position);
        if (r > 0 && step.op == SerialOp::ple && serial_resources.ple_prefetch && serial_resources.ple_prefetch->active)
            throw std::logic_error("known window row0 did not consume its sole PLE lookup");
        if (window.row_plan[r].bindings.bindings[i].host_or_runtime_step) {
            st.dispatchers[r].dispatch_runtime_step(context); ++result.execution.runtime_steps;
        } else {
            st.dispatchers[r].dispatch_kernel(context); ++result.execution.kernel_steps;
        }
        ++result.row_loop_dispatches; ++result.execution.completed_steps;
        result.execution.completion_order.push_back(step.id);
        result.execution.state.publish_write(step.writes_state);
    }
    observe();
}

void KnownWindowRun::record_external_step(std::size_t i) {
    auto& st = *state_;
    if (st.admission_only || st.completed || st.failed || i >= st.steps || (!st.selected_rows[i] && !st.selected[i]))
        throw std::logic_error("known window external step must be an admitted promoted step");
    if (!st.result.execution.state.contains(st.serial.execution.steps[i].reads_state))
        throw std::logic_error("known window persistent state disappeared after admission");
    // Counted as the window's own entry would have been: one wide group or one pair per two rows.
    const auto width = st.selected_rows[i] ? st.rows : 2U;
    for (std::uint32_t first = 0; first < st.rows; first += width) st.record_promoted(i, width);
}

void KnownWindowRun::complete() {
    auto& st = *state_;
    if (st.admission_only || st.completed || st.failed) throw std::logic_error("known window completion outside an admitted window");
    if (st.serial_resources.ple_prefetch && st.serial_resources.ple_prefetch->active)
        throw std::logic_error("known window completed without consuming its PLE lookup");
    window_host_timing().launch += window_ms_since(st.timing_mark); st.timing_mark = std::chrono::steady_clock::now();
    st.hooks.complete();
    window_host_timing().complete += window_ms_since(st.timing_mark); ++window_host_timing().windows;
    st.materialize_position(st.contexts.back()[st.rows-1U], st.post_position - 1);
    std::uint64_t final_position_word = 0;
    std::memcpy(&final_position_word, reinterpret_cast<const void*>(st.position_state.address), sizeof(final_position_word));
    if (final_position_word != st.post_position - 1)
        throw std::logic_error("known window final host position differs from the serial row frontier");
    st.completed = true;
}

void KnownWindowRun::fail() {
    auto& st = *state_;
    if (st.failed) return;
    st.failed = true;
    if (st.serial_resources.ple_lookup && st.serial_resources.ple_prefetch)
        cancel_ple_lookup_prefetch(*st.serial_resources.ple_lookup, *st.serial_resources.ple_prefetch);
    st.session.health = SerialSessionHealth::poisoned;
    st.session.arithmetic_lineage.tainted = true;
    try { st.hooks.quiesce_or_fail_stop(); } catch (...) { std::terminate(); }
}

KnownWindowResult KnownWindowRun::publish() {
    auto& st = *state_;
    if (!st.completed || st.failed) throw std::logic_error("known window publication requires a completed window");
    auto& session = st.session;
    const auto rows = st.rows;
    session.state = st.result.execution.state;
    session.arithmetic_lineage = std::move(st.next_lineage);
    session.next_position = st.post_position; session.committed_tokens = st.post_committed;
    {
        auto correlation = current_execution_correlation();
        correlation.execution_identity = session.execution_identity;
        correlation.position = st.post_position - rows;
        correlation.row = UINT32_MAX; correlation.rows = rows; correlation.step_id = UINT32_MAX;
        correlation.phase = ExecutionPhase::commit;
        ExecutionCorrelationScope commit_scope(correlation);
        observe_execution_activity(ExecutionActivityKind::token_commit, 0, rows);
    }
    return std::move(st.result);
}

KnownWindowResult execute_known_prefix_window_eager(
    const CompiledKnownPrefixWindow& window, const CompiledSerialExecution& serial,
    const ExecutionRuntimeResources& serial_resources,
    const ExecutionMemoryBinding& window_scratch, SerialSessionContext& session,
    std::span<const std::uint32_t> tokens,
    std::span<const CompiledLaunchPlan* const> row_launch,
    std::span<const KnownWindowR2Entry> r2_entries,
    std::span<const KnownWindowRowsEntry> rows_entries,
    const KnownWindowExecutionHooks& hooks) {
    KnownWindowRun run(window, serial, serial_resources, window_scratch, session, tokens, row_launch,
        r2_entries, rows_entries, hooks);
    if (run.admission_only()) return {};
    try {
        for (std::size_t i = 0; i < run.steps(); ++i) run.launch_step(i);
        run.complete();
    } catch (...) {
        run.fail();
        throw;
    }
    return run.publish();
}

KnownWindowResult execute_known_prefix_window_eager(
    const CompiledKnownPrefixWindow& window, const CompiledSerialExecution& serial,
    const ExecutionRuntimeResources& serial_resources, const ExecutionMemoryBinding& scratch,
    SerialSessionContext& session, std::array<std::uint32_t,2> tokens,
    std::array<const CompiledLaunchPlan*,2> launch,
    std::span<const KnownWindowR2Entry> entries, const KnownWindowExecutionHooks& hooks) {
    return execute_known_prefix_window_eager(window,serial,serial_resources,scratch,session,
        std::span<const std::uint32_t>(tokens),std::span<const CompiledLaunchPlan* const>(launch),entries,{},hooks);
}

}  // namespace strix
