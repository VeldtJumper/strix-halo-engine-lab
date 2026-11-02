#include "strix/batch_nested_prefill.hpp"
#include "strix/hip_known_prefix_window.hpp"
#include "strix/hip_qwen_rotary_table.hpp"
#include "strix/hip_moe_q4k_a8_known_rows.hpp"
#include "hip_prefill_pool.hpp"
#include "strix/dispatch_fuse.hpp"
#include "strix/qpf2k_defaults.hpp"
#include "strix/hip_mtp_target_window.hpp"
#include "strix/hip_ple_launcher.hpp"

#include "strix/deferred_moe_status.hpp"
#include "strix/hip_hyper_mix_launcher.hpp"
#include "strix/lane_chain_hyper_output.hpp"
#include "strix/hip_indexed_attention_launcher.hpp"
#include "strix/dispatch_resources.hpp"
#include "strix/hip_launch_context.hpp"
#include <functional>
#include "strix/hip_generation_output.hpp"
#include "strix/hip_moe_route_launcher.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/prepared_tensor_views.hpp"
#include "strix/known_window_cohort.hpp"
#include "strix/hip_mtp_target_window.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <tuple>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <cstdint>
#include <exception>
#include <deque>

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// CMake route clients: strix-hip-decode-token (also HTTP serving and MTP
// cohorts through strix_serve), strix-hip-mtp-model-qualification,
// strix-hip-known-window-model-qualification,
// strix-hip-known-prompt-model-qualification, and
// strix-hip-serving-greedy-model-qualification. Keep startup defaults shared
// with any future runtime client that pulls this translation unit into its link.
namespace {
const bool qpf2k_default_env = [] {
    strix::qpf2k::initialize_before_main(
        [](const char* name) { return std::getenv(name); },
        [](const char* name, const char* value) { return ::setenv(name, value, 0) == 0; });
    return true;
}();
} // namespace

namespace {
// STRIX_QRTX_STEP_DUMP=<file> (diagnostic, off by default): after every step of a tiled prefill block, append every row's
// read and write buffers to <file> (fixtures for the bit-identical CUDA prefill). Record, little-endian: u32 magic
// 0x31445351 (writes) or 0x31445352 (reads), u32 step id, u32 op, u32 layer (UINT32_MAX when none), u32 row, u32 buffer
// index, u32 rows, u32 kernel id length,
// u64 position, u64 bytes, the kernel id, then the bytes. Runs on the execution stream, so the copies follow the step's
// kernels; the host waits for each step's copies before writing. The values themselves are unchanged.
strix::HipKnownPrefixWindow::PrefillStepObserver qrtx_step_dump_observer(strix::HipKnownPrefixWindow::PrefillStepObserver inner,
                                                                         std::uint32_t rows) {
    static const char* path=std::getenv("STRIX_QRTX_STEP_DUMP");
    if(!path || !*path) return inner;
    return [inner=std::move(inner),rows](std::size_t step,const std::function<strix::ExecutionDispatchContext(std::uint32_t)>& row) {
        if(inner) inner(step,row);
        static std::FILE* file=[]{std::FILE* f=std::fopen(path,"ab");if(!f)throw std::runtime_error("STRIX_QRTX_STEP_DUMP cannot open");return f;}();
        static std::vector<std::uint8_t> host;
        const auto stream=strix::hip_execution_stream();
        for(std::uint32_t r=0;r<rows;++r) {
            const auto c=row(r);
            for(std::size_t k=0;k<c.reads.size()+c.writes.size();++k) {
                const bool is_read=k<c.reads.size();const std::size_t w=is_read?k:k-c.reads.size();
                const auto a=is_read?strix::resolve_dispatch_read(c,w):strix::resolve_dispatch_write(c,w);
                host.resize(static_cast<std::size_t>(a.bytes));
                if(a.bytes&&hipMemcpyAsync(host.data(),reinterpret_cast<const void*>(a.address),static_cast<std::size_t>(a.bytes),
                    hipMemcpyDeviceToHost,stream)!=hipSuccess) throw std::runtime_error("STRIX_QRTX_STEP_DUMP copy failed");
                if(hipStreamSynchronize(stream)!=hipSuccess) throw std::runtime_error("STRIX_QRTX_STEP_DUMP sync failed");
                const std::string_view id=c.kernel_id.value_or(std::string_view{});
                const std::uint32_t h[8]={is_read?0x31445352U:0x31445351U,c.step_id,static_cast<std::uint32_t>(c.op),c.layer.value_or(UINT32_MAX),r,
                    static_cast<std::uint32_t>(w),rows,static_cast<std::uint32_t>(id.size())};
                const std::uint64_t h2[2]={c.position,static_cast<std::uint64_t>(a.bytes)};
                if(std::fwrite(h,sizeof(h),1,file)!=1||std::fwrite(h2,sizeof(h2),1,file)!=1||
                   (id.size()&&std::fwrite(id.data(),1,id.size(),file)!=id.size())||
                   (a.bytes&&std::fwrite(host.data(),1,host.size(),file)!=host.size())) throw std::runtime_error("STRIX_QRTX_STEP_DUMP write failed");
            }
        }
        std::fflush(file);
    };
}
}  // namespace

namespace strix {
namespace {
void check(hipError_t error, const char* operation) {
    if (error != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
}
bool empty(const ExecutionMemoryBinding& binding) {
    return !binding.execution_identity && !binding.arena_base && !binding.arena_bytes &&
        !binding.allocation_generation;
}
bool same_layout(const SerialSessionStateLayout& a, const SerialSessionStateLayout& b) {
    if (a.layout_identity != b.layout_identity || a.context_tokens != b.context_tokens ||
        a.device_alignment != b.device_alignment || a.host_alignment != b.host_alignment ||
        a.device_total_bytes != b.device_total_bytes || a.host_total_bytes != b.host_total_bytes ||
        a.recurrent_storage != b.recurrent_storage ||
        a.allocations.size() != b.allocations.size()) return false;
    for (std::size_t i = 0; i < a.allocations.size(); ++i) {
        const auto& x = a.allocations[i]; const auto& y = b.allocations[i];
        if (x.component != y.component || x.state_flag != y.state_flag || x.storage != y.storage ||
            x.offset != y.offset || x.bytes != y.bytes || x.provisional != y.provisional) return false;
    }
    return true;
}
std::uint32_t u32(std::uint64_t value) {
    if (!value || value > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("known-window launcher shape exceeds uint32");
    return static_cast<std::uint32_t>(value);
}
ExecutionBindingRequest binding_request(const HipLauncherCatalogConfig& config) {
    ExecutionBindingRequest request;
    request.minimum_qualification = kernels::QualificationStatus::experimental;
    request.device_graph_inputs = config.enable_device_graph_inputs;
    request.prefetch_q8_lm_head = config.prefetch_q8_lm_head;
    request.wave8_q8_lm_head = config.wave8_q8_lm_head;
    request.wave8_q8_gdn_input = config.wave8_q8_gdn_input;
    request.wave8_q8_gdn_output = config.wave8_q8_gdn_output;
    request.wave8_q8_qsa_output = config.wave8_q8_qsa_output;
    request.parallel_moe_router = config.parallel_moe_router;
    request.staged_moe_router_logits = config.staged_moe_router_logits;
    request.ranked_moe_router_topk = config.ranked_moe_router_topk;
    request.wave_moe_router_topk = config.wave_moe_router_topk;
    request.prefetch_q8_qsa_input = config.prefetch_q8_qsa_input;
    request.qsa_input_prefetch_layers = config.qsa_input_prefetch_layers;
    request.q4k_a8g32_gate_up = config.q4k_a8g32_gate_up;
    request.batched_moe_activation = config.batched_moe_activation;
    request.grouped_exact_q51_down = config.grouped_exact_q51_down;
    request.q51_slot_parallel_down = config.q51_slot_parallel_down;
    request.a8_down_q51 = config.a8_down_q51;
    request.a8_down_q80 = config.a8_down_q80;
    request.q5k_a8g32_gate_up = config.q5k_a8g32_gate_up;
    request.dense_a8 = config.dense_a8;
    request.lane_chain_hyper_output = config.lane_chain_hyper_output;
    request.prefetch_hc_projection_next_tile = config.prefetch_hc_projection_next_tile;
    request.deferred_moe_status = config.deferred_moe_status;
    request.prefetch_q8_shared_expert = config.prefetch_q8_shared_expert;
    request.shared_expert_prefetch_layers = config.shared_expert_prefetch_layers;
    return request;
}
} // namespace

struct HipKnownPrefixWindow::Impl {
    // Member order keeps all retained owners alive through destructor recovery.
    std::shared_ptr<const void> actual_owner;
    HipSetupQuiescenceGuard guard;
    CompiledKnownPrefixWindow window;
    std::optional<CompiledKnownPrefixPrefill> prefill;
    std::optional<CompiledSerialExecution> prefill_launcher_source;
    CompiledSerialExecution serial;
    ModelContract model;
    const bool serial_rows_required;
    const KnownWindowFolds allowed_folds;
    HipLauncherCatalogConfig config;
    ExecutionRuntimeResources resources;
    ExecutionMemoryBinding scratch;
    HipLaunchContext scope;
    RuntimeExecutionIdentities original_identities;
    SerialSessionStateLayout original_state_layout;
    std::unique_ptr<const PreparedTensorViews> metadata;
    std::vector<ExecutionRuntimeResources> row_resources;
    std::vector<std::unique_ptr<HipLauncherCatalog>> catalogs;
    std::vector<ExecutionLauncherRegistry> registries;
    std::vector<CompiledLaunchPlan> launch_plans;
    std::vector<const CompiledLaunchPlan*> launch_pointers;
    std::vector<std::uint64_t> position_readers;
    HipSharedExpertLauncherConfig shared_config, shared_fallback;
    HipGdnLauncherConfig gdn_config;
    HipIndexedAttentionLauncherConfig qsa_config, qsa_fallback;
    HipMoeExpertsLauncherConfig routed_config;
    HipHyperMixLauncherConfig hyper_config, hyper_fallback;
    HipLmHeadLauncherConfig head_config;
    ExecutionBufferAddress greedy_partials{};
    HipGreedyRowsReply* greedy_replies = nullptr;
    HipMoeRouteLauncherConfig route_config;
    HipElementwiseLauncherConfig elementwise_config;
    q51_known_rows::Workspace routed_workspace;
    q51_known_rows::Workspace prefill_routed_workspace; // all T rows of a prefill step (STRIX_PREFILL_FULL_MOE)
    // Segmented prefill blocks of fewer than T rows: the CSR of their full union per row count (allocated at admission).
    std::map<std::uint32_t, q51_known_rows::Workspace> prefill_union_workspace;
    std::uint32_t prefill_union_rows = 0; // rows of the last full prefill union (0: T)
    std::map<std::uint32_t, q51_known_rows::Workspace> cohort_union_workspace; // wide cohort unions per row count (on first use)
    q51_known_rows::Summary pending_union_summary;
    std::uint32_t last_union_layer = 0;
    std::optional<HipKnownPrefixUnionObservation> last_union_observation;
    std::array<KnownWindowR2Entry, 3> r2_entries;
    std::vector<KnownWindowRowsEntry> rows_entries;
    std::vector<KnownWindowRowsEntry> prefill_entries;
    // Prefix acceptance: deferred GDN rows (control word in host-mapped coherent memory, so the
    // host publishes {n_keep, slot} without a copy) and per-row PLE snapshots.
    struct HipFree { void operator()(void* p) const noexcept { if (p) (void)hipFree(p); } };
    struct HipHostFree { void operator()(void* p) const noexcept { if (p) (void)hipHostFree(p); } };
    // Rows of any window size share the [2 slots][8 rows] layout, so owners of one
    // session may share this record (share_deferred): pending rows of one owner are
    // applied by the next window of any owner or by flush_deferred.
    struct DeferredGdn {
        std::unique_ptr<void, HipFree> retained;
        std::unique_ptr<void, HipHostFree> control;
        std::uint32_t slot = 0;
        bool pending = false;
    };
    std::shared_ptr<DeferredGdn> gdn_deferred;
    HipGdnWindowConfig gdn_window_config;
    std::vector<ExecutionDispatchContext> gdn_commit_rows; // one row context per GDN layer (stable plan views)
    std::deque<ExecutionRuntimeResources> gdn_commit_resources; // owned copies: the window's are call-local
    std::unique_ptr<HipPleKnownWindow> ple_window;
    std::vector<const void*> preflight_memo, static_preflight_memo;
    std::vector<std::pair<const void*, std::array<std::uint64_t, 2>>> recipe_memo;
    std::vector<std::array<std::uintptr_t, 2>> preflight_slots;
    ArenaLookupIndex arena_index; // the immutable compiled window's arena views
    // Verify windows run each layer's shared expert on a second stream next to the routed experts (see
    // detach_shared). The shared expert reads only the MoE input and writes only its output,
    // which the MoE combine reads; the owner's stream waits for it there. Its workspace is owned here,
    // since the arena may give the step's workspace to the routed kernels that now run alongside it.
    struct SharedDetach {
        hipStream_t side = nullptr;
        // Two event sets used by alternate windows: re-recording an event whose cross-stream wait of the
        // previous window may still be queued can block the host.
        std::vector<hipEvent_t> forked, joined;
        std::vector<unsigned char> pending;
        std::uint32_t parity = 0;
        void* workspace = nullptr;
        std::uint64_t workspace_bytes = 0;
        // Tiled prefill (STRIX_QPF2K_SCHED bit 1): compact workspace rows of a detached T-row shared-expert step.
        void* prefill_workspace = nullptr;
        std::uint64_t prefill_workspace_bytes = 0;
        ~SharedDetach() {
            if (side) (void)hipStreamSynchronize(side);
            for (auto e : forked) if (e) (void)hipEventDestroy(e);
            for (auto e : joined) if (e) (void)hipEventDestroy(e);
            if (workspace) (void)hipFree(workspace);
            if (prefill_workspace) (void)hipFree(prefill_workspace);
            if (side) (void)hipStreamDestroy(side);
        }
    };
    std::unique_ptr<SharedDetach> shared_detach;
    DeferredMoeFirstError completion_status{};
    HipKnownPrefixWindow::PrefillServicePoint prefill_service;
    bool failed = false;
    std::uint32_t active_rows = 0; // execute_mtp_captured: rows in the expert union (0 = every row)
    HipKnownPrefixWindow::PrefillStepObserver prefill_observer,window_observer;
    std::function<bool(std::size_t)> prefill_observes;  // empty: the prefill observer reads every step
    // Set while a cohort executes this owner's window: its routed union ran on the cohort primary.
    bool cohort_member = false;

    Impl(CompiledKnownPrefixWindow source, CompiledSerialExecution serial_source,
        ModelContract model_source, HipLauncherCatalogConfig config_source,
        const ExecutionRuntimeResources& serial_resources, ExecutionMemoryBinding memory,
        HipLaunchContext native_scope, std::shared_ptr<const void> owner,
        q51_known_rows::Workspace union_workspace,std::optional<CompiledKnownPrefixPrefill> tiled = std::nullopt)
        : actual_owner(std::move(owner)), window(std::move(source)), serial(std::move(serial_source)),
          model(std::move(model_source)),
          serial_rows_required(!tiled && model.rotary_identity().descriptor().mode()==RotaryMode::yarn),
          allowed_folds(known_window_folds_for_route(serial_rows_required,{true,true,true})),
          config(std::move(config_source)), resources(serial_resources),
          scratch(memory), scope(native_scope), routed_workspace(std::move(union_workspace)) {
        require_qwen_rotary_device_payload(model.rotary_identity(),model.qwen_rotary_table.get());
        if(model.qwen_rotary_table) validate_qwen_rotary_reader(*model.qwen_rotary_table,scope.stream);
        scope.gsq_model = resources.model_manifest && qpf2k::has_gsq_gate(*resources.model_manifest);
        const qpf2k::ModelScope defaults_scope(scope.gsq_model);
        prefill=std::move(tiled);
        resources.qsa_pool_speculative=resources.qsa_committed_pool && !prefill;
        if (!actual_owner) throw std::invalid_argument("known-window native owner requires retained actual owners");
        if (const auto error = model.validate(); !error.empty()) throw std::invalid_argument(error);
        if (config.gdn_recurrent_storage != GdnRecurrentStorage::key_value ||
            (resources.session_state_layout && resources.session_state_layout->recurrent_storage != GdnRecurrentStorage::key_value))
            throw std::invalid_argument("known-window native owner requires key-value recurrent storage");
        if (serial.device_graph_inputs || config.enable_device_graph_inputs || serial.logical_rows != 1U ||
            serial.request_concurrency != 1U || config.activation != HipActivationStorage::bf16)
            throw std::invalid_argument("known-window native owner supports eager BF16 C1 plans only");
        validate_compiled_known_prefix_window(window, serial);
        if(serial_rows_required && !known_window_policy_selection_admitted(window,false,scope.gsq_model,{}))
            throw std::invalid_argument("YaRN verify window has no serial-row implementation; refused during setup");
        if(prefill) {
            validate_compiled_known_prefix_prefill(*prefill,serial);
            if(prefill->tile8.window_execution_identity!=window.window_execution_identity||
                prefill->tile8.window_kernel_plan_identity!=window.window_kernel_plan_identity)
                throw std::invalid_argument("known prefill owner selection tile differs");
            prefill_launcher_source=serial;
            prefill_launcher_source->execution_identity=prefill->prefill_execution_identity;
            // Launcher construction admits the complete R1 head against the one capable row.
            // Live contexts still resolve every non-head row against its own view.
            prefill_launcher_source->arena=prefill->terminal_logits_arena?prefill->arena.row_view.back():prefill->arena.row_view[0];
            prefill_launcher_source->whole_plan_graph_safe=false;
        }
        if (!resources.model_manifest || !resources.tensor_access)
            throw std::invalid_argument("known-window native owner requires manifest and tensor access resources");
        validate_known_prefix_window_formats(window, serial, *resources.model_manifest, *resources.tensor_access);
        kernels::KernelRegistry admitted;
        register_complete_hip_kernel_catalog(model, binding_request(config), config, admitted);
        for (const auto& binding : serial.bindings.bindings)
            if (!binding.host_or_runtime_step && (!binding.kernel_id || !admitted.find(*binding.kernel_id)))
                throw std::invalid_argument("known-window serial kernel is not in the configured native catalog");
        original_identities = build_runtime_execution_identities(serial.execution_identity, serial.arena, resources);
        original_state_layout = *resources.session_state_layout;
        metadata = std::make_unique<const PreparedTensorViews>(serial.execution_identity, serial.execution,
            *resources.model_manifest, *resources.model_weights, *resources.tensor_access, actual_owner);
        validate_execution_memory_binding(execution_identity(), physical_arena(), scratch);
        validate_scope();
        if (prefill) {
            // The caller constructs the largest compiled T owner before tails. The
            // device arena identifies the owner; never reserve on an unrelated device.
            hipPointerAttribute_t attributes{};
            check(hipPointerGetAttributes(&attributes, reinterpret_cast<const void*>(scratch.arena_base)),
                "query prefill arena device");
            if (attributes.device != hip_prefill_device())
                throw std::invalid_argument("prefill setup requires the window's owning device");
#if defined(STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME)
            moe_q4k_a8::known_rows::reserve_prefill_pools(prefill->logical_rows);
#endif
            reserve_hyper_mix_prefill_pools(prefill->logical_rows);
            reserve_indexed_attention_prefill_pools(prefill->logical_rows,
                original_state_layout.context_tokens, u32(model.indexer_compress_ratio), prefill->qsa_score_ring_workspace_bytes != 0U);
        }
        const auto launcher_rows=prefill?1U:window.logical_rows;
        row_resources.resize(launcher_rows);
        catalogs.resize(launcher_rows);
        registries.resize(launcher_rows);
        launch_plans.resize(launcher_rows);
        launch_pointers.reserve(launcher_rows);
        for (std::size_t row = 0; row < launcher_rows; ++row) {
            const auto& row_source=prefill?*prefill_launcher_source:window.row_plan[row];
            row_resources[row] = resources;
            row_resources[row].scratch = &scratch;
            catalogs[row] = std::make_unique<HipLauncherCatalog>(model, config);
            catalogs[row]->register_for_compiled_execution(row_source, registries[row], &row_resources[row]);
            if (catalogs[row]->audit(row_source, registries[row]).missing_steps)
                throw std::invalid_argument("known-window native row catalog coverage is incomplete");
            launch_plans[row] = compile_launch_plan(row_source, registries[row]);
            for (const auto& launcher : launch_plans[row].steps)
                if (!launcher.preflight)
                    throw std::invalid_argument("known-window native row contains a launcher without deep preflight");
            launch_pointers.push_back(&launch_plans[row]);
            const auto readers = catalogs[row]->known_window_position_readers(row_source, registries[row]);
            position_readers.insert(position_readers.end(), readers.begin(), readers.end());
        }
        std::sort(position_readers.begin(), position_readers.end());
        position_readers.erase(std::unique(position_readers.begin(), position_readers.end()), position_readers.end());
        if(!prefill) {
            std::vector<const ExecutionArenaPlan*> plans{&window.arena.physical};
            for(const auto& row:window.row_plan) plans.push_back(&row.arena);
            for(const auto& view:window.arena.row_view) plans.push_back(&view);
            arena_index=ArenaLookupIndex(plans);
        } else {
            // STRIX_PREFILL_PIPE bit 128: one persistent index for every row view of the T arena.
            const char* v=std::getenv("STRIX_PREFILL_PIPE");
            if(!qpf2k::suppressed(qpf2k::Item::pipe)&&v&&(std::strtoul(v,nullptr,0)&128UL)!=0UL) arena_index=ArenaLookupIndex::uniform_views(prefill->arena.row_view);
        }
        setup_r2();
        setup_rows();
        setup_routed_union();
        if(prefill) setup_prefill_entries();
        setup_hyper_rows();
        setup_deferred();
        setup_head_rows();
        setup_router_elementwise_rows();
        // Setup only: no execution callback clears or reads this record before
        // all deep admission. Exclusive status ownership preserves zero across
        // successful calls; any exception disables the owner permanently.
        if (config.deferred_moe_status) guard.run([&] {
            check(hipMemcpy(&completion_status, reinterpret_cast<const void*>(scope.deferred_moe_status.arena_base),
                sizeof(completion_status), hipMemcpyDeviceToHost), "read initial known-window deferred status");
            check_deferred_moe_status(completion_status);
        });
    }
    ~Impl() {
        guard.recover_or_fail_stop();
        // A deferred block nobody completed (no current caller leaves one): still check it; a failed check fail-stops
        // (its results were already committed to the session).
        if (deferred_pending >= 0 && !failed) {
            try { finish_deferred_completion(); }
            catch (const std::exception& e) { std::fprintf(stderr, "known prefill abandoned deferred block failed its check: %s\n", e.what()); std::terminate(); }
        } else if (deferred_pending >= 0) (void)hipEventSynchronize(deferred_events[static_cast<unsigned>(deferred_pending)]);
        for (auto e : deferred_events) if (e) (void)hipEventDestroy(e);
        if (deferred_slots) (void)hipHostFree(deferred_slots);
    }
    bool full_prefill_union() const noexcept {return prefill&&prefill_routed_workspace.owner;}
    // The CSR workspace of a wide cohort union of 9..32 rows enqueued on this owner's stream.
    const q51_known_rows::Workspace& cohort_union(std::uint32_t rows) {
        auto& slot = cohort_union_workspace[rows];
        if (slot.owner) return slot;
        const auto bytes = q51_known_rows::required_scratch_bytes(rows, routed_config.expert_count, routed_config.top_k);
        void* memory = nullptr;
        guard.run([&] { check(hipMalloc(&memory, bytes), "allocate known-window cohort routed CSR"); });
        std::shared_ptr<void> owner(memory, [](void* p) { if (p) (void)hipFree(p); });
        const auto base = reinterpret_cast<std::uintptr_t>(memory);
        if (base % 256U) throw std::runtime_error("known-window cohort routed CSR allocation is not 256-byte aligned");
        status_disjoint(base, bytes);
        slot = {{execution_identity(), base, bytes, routed_workspace.memory.allocation_generation ^ (static_cast<std::uint64_t>(rows) << 56U)}, owner};
        return slot;
    }
    // The CSR workspace of a full prefill union of `rows` rows: the owner's own for T, else one per row count.
    const q51_known_rows::Workspace& prefill_union(std::uint32_t rows) {
        if (!prefill || rows == prefill->logical_rows) return prefill_routed_workspace;
        if (!rows || rows % 8U || rows > prefill->logical_rows)
            throw std::invalid_argument("known prefill union rows must be a multiple of 8 within T");
        auto& slot = prefill_union_workspace[rows];
        if (slot.owner) return slot;
        const auto bytes = q51_known_rows::prefill_scratch_bytes(rows, u32(model.num_experts), u32(model.num_experts_per_token));
        void* memory = nullptr;
        guard.run([&] { check(hipMalloc(&memory, bytes), "allocate known prefill segmented routed CSR"); });
        std::shared_ptr<void> owner(memory, [](void* p) { if (p) (void)hipFree(p); });
        const auto base = reinterpret_cast<std::uintptr_t>(memory);
        if (base % 256U) throw std::runtime_error("known prefill segmented routed CSR allocation is not 256-byte aligned");
        status_disjoint(base, bytes);
        slot = {{execution_identity(), base, bytes,
            routed_workspace.memory.allocation_generation ^ (1ULL << 63U) ^ (static_cast<std::uint64_t>(rows) << 40U)}, owner};
        return slot;
    }
    std::uint64_t execution_identity() const noexcept {return prefill?prefill->prefill_execution_identity:window.window_execution_identity;}
    const ExecutionArenaPlan& physical_arena() const noexcept {return prefill?prefill->arena.physical:window.arena.physical;}
    std::uint32_t logical_rows() const noexcept {return prefill?prefill->logical_rows:window.logical_rows;}

    void status_disjoint(std::uintptr_t address, std::uint64_t bytes) const {
        if (!bytes) return;
        if (!address || bytes > std::numeric_limits<std::uintptr_t>::max() - address)
            throw std::invalid_argument("known-window deferred alias owner range is invalid");
        const auto& status = scope.deferred_moe_status;
        if (status.arena_base < address + bytes && address < status.arena_base + status.arena_bytes)
            throw std::invalid_argument("known-window deferred status aliases a retained resource");
    }
    void validate_scope() const {
        guard.require_original();
        if (scope.graph_capture || !scope.stream || scope.stream == hipStreamLegacy ||
            scope.stream == hipStreamPerThread || scope.arguments != GraphDynamicArguments{} ||
            !empty(scope.status) || !empty(scope.ngram_input) || !empty(scope.ngram_output) ||
            scope.context_tokens || scope.vocabulary_size)
            throw std::invalid_argument("known-window native owner requires empty graph bindings and one eager owned stream");
        hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
        check(hipStreamIsCapturing(scope.stream, &capture), "query known-window native capture");
        if (capture != hipStreamCaptureStatusNone)
            throw std::invalid_argument("known-window native stream is being captured");
        const auto& status = scope.deferred_moe_status;
        if (!config.deferred_moe_status) {
            if (!empty(status)) throw std::invalid_argument("known-window deferred configuration/scope parity differs");
            return;
        }
        if (status.execution_identity != execution_identity() || !status.arena_base ||
            status.arena_base % alignof(std::uint64_t) || status.arena_bytes != sizeof(DeferredMoeFirstError) ||
            !status.allocation_generation || status.arena_bytes > std::numeric_limits<std::uintptr_t>::max() - status.arena_base)
            throw std::invalid_argument("known-window deferred status has stale identity, generation or extent");
        status_disjoint(scratch.arena_base, scratch.arena_bytes);
        status_disjoint(resources.scratch->arena_base, resources.scratch->arena_bytes);
        status_disjoint(resources.session_state_memory->device_base, resources.session_state_memory->device_bytes);
        status_disjoint(resources.session_state_memory->host_base, resources.session_state_memory->host_bytes);
        for (const auto& tensor : resources.model_weights->tensors)
            if (tensor.storage != RuntimeTensorStorage::remote) status_disjoint(tensor.address, tensor.bytes);
    }
    void validate_metadata() const {
        guard.require_original();
        metadata->validate_source_metadata(*resources.model_manifest, *resources.model_weights, *resources.tensor_access);
        const auto current = build_runtime_execution_identities(serial.execution_identity, serial.arena, resources);
        if (current.compatibility_identity != original_identities.compatibility_identity ||
            current.binding_identity != original_identities.binding_identity ||
            !same_layout(original_state_layout, *resources.session_state_layout))
            throw std::invalid_argument("known-window retained runtime bindings or state layout changed");
        validate_scope();
    }
    // STRIX_SHARED_DETACH=0: the shared expert stays on the owner's stream between router and routed experts.
    static bool shared_detach_enabled() {
        static const bool on = [] { const char* v = std::getenv("STRIX_SHARED_DETACH"); return !v || v[0] != '0'; }();
        return on;
    }
    SharedDetach& detach_state() {
        if (shared_detach) return *shared_detach;
        auto d = std::make_unique<SharedDetach>();
        const auto layers = static_cast<std::size_t>(model.hidden_layers);
        d->forked.assign(2U * layers, nullptr); d->joined.assign(2U * layers, nullptr); d->pending.assign(layers, 0U);
        // STRIX_QPF2K_SCHED bit 32 (tiled prefill owners only): the side stream at the lowest priority, so the detached
        // shared expert fills what the routed kernels leave instead of competing with them (same kernels and bits).
        if (prefill && (prefill_schedule() & 32U)) {
            int least = 0, greatest = 0;
            check(hipDeviceGetStreamPriorityRange(&least, &greatest), "known prefill stream priority range");
            check(hipStreamCreateWithPriority(&d->side, hipStreamNonBlocking, least), "create known prefill low-priority shared stream");
        } else
        check(hipStreamCreateWithFlags(&d->side, hipStreamNonBlocking), "create known-window shared stream");
        for (auto* list : {&d->forked, &d->joined})
            // Both streams are on this device: no system-scope fence (an L2 writeback per event otherwise).
            for (auto& e : *list) check(hipEventCreateWithFlags(&e, hipEventDisableTiming | hipEventDisableSystemFence),
                "create known-window shared event");
        // Rows keep the arena's row stride (one row view apart), so this covers eight rows of any window.
        d->workspace_bytes = 8ULL << 20U;
        check(hipMalloc(&d->workspace, d->workspace_bytes), "allocate known-window shared workspace");
        shared_detach = std::move(d);
        return *shared_detach;
    }
    bool detachable_shared(std::size_t i) const {
        const auto variant = window.r2_variant[i];
        return serial.execution.steps[i].op == SerialOp::shared_expert &&
            (variant == KnownWindowR2Variant::shared_q8_virtual256_r2_v1 ||
             variant == KnownWindowR2Variant::shared_q8_virtual256_r4_v1 || variant == KnownWindowR2Variant::shared_q8_virtual256_r8_v1);
    }
    // Before step i of a launched verify window. A shared-expert rows step is enqueued on the side stream
    // after everything queued so far on the owner's stream (the router included: forking before the router
    // also overlapped it but measured slower, the staged router needs the bandwidth itself), and the
    // layer's MoE combine waits for it. Returns true for the step this enqueued in place of the loop.
    bool detach_shared(std::size_t i, const std::vector<std::vector<ExecutionDispatchContext>>& contexts) {
        const auto& step = serial.execution.steps[i];
        if (!step.layer || *step.layer >= model.hidden_layers) return false;
        const auto layer = *step.layer;
        if (detachable_shared(i)) {
            auto& d = detach_state();
            if (d.pending[layer]) throw std::logic_error("known-window shared expert detached twice before its combine");
            const auto& rows = contexts[i];
            const auto e = layer + d.parity * d.pending.size();
            check(hipEventRecord(d.forked[e], scope.stream), "record known-window shared fork");
            check(hipStreamWaitEvent(d.side, d.forked[e], 0), "order known-window shared stream");
            shared_expert_known_window_rows_launch_detached(rows, shared_for(rows), d.side,
                reinterpret_cast<std::uintptr_t>(d.workspace), d.workspace_bytes);
            check(hipEventRecord(d.joined[e], d.side), "record known-window shared join");
            d.pending[layer] = 1U;
            return true;
        }
        if (step.op == SerialOp::moe_combine) join_shared(layer);
        return false;
    }
    // STRIX_QPF2K_SCHED (bit mask, default 0; tiled prefill only): bit 1 = each layer's full-T shared-expert step runs on
    // the side stream next to the routed experts (forked after the router, like detach_shared for verify windows),
    // with its workspace rows in an owner-held compact buffer; the MoE combine waits for it. Same kernels, launches
    // and per-row arithmetic; only the stream and the workspace address differ.
    unsigned prefill_schedule() const {
        static const unsigned v = [] { const char* e = std::getenv("STRIX_QPF2K_SCHED"); return e ? static_cast<unsigned>(std::strtoul(e, nullptr, 0)) : 0U; }();
        return qpf2k::suppressed(qpf2k::Item::sched, scope.gsq_model) ? 0U : v;
    }
    bool prefill_schedule_step(std::size_t i, std::span<const ExecutionDispatchContext> rows) {
        const auto& step = serial.execution.steps[i];
        if (prefill_join_due) throw std::logic_error("known prefill detached step was not observed before the next step");
        if (!prefill || !step.layer || *step.layer >= model.hidden_layers) return false;
        const auto layer = *step.layer;
        if ((prefill_schedule() & 1U) && !rows.empty() && step.op == SerialOp::shared_expert &&
            prefill->tile8.r2_variant[i] == KnownWindowR2Variant::shared_q8_virtual256_r8_v1) {
            auto& d = detach_state();
            if (d.pending[layer]) throw std::logic_error("known prefill shared expert detached twice before its combine");
            const auto row_bytes = resolve_dispatch_temporary(rows[0], 0).bytes;
            const auto stride = (row_bytes + 255U) / 256U * 256U;
            if (!row_bytes || stride > UINT64_MAX / rows.size()) throw std::invalid_argument("known prefill shared workspace rows overflow");
            const auto need = stride * rows.size();
            if (d.prefill_workspace_bytes < need) {
                // Once per owner (T rows): nothing of this owner is outstanding on the side stream between blocks.
                check(hipStreamSynchronize(d.side), "drain known prefill shared stream");
                if (d.prefill_workspace) check(hipFree(d.prefill_workspace), "free known prefill shared workspace");
                d.prefill_workspace = nullptr; d.prefill_workspace_bytes = 0;
                check(hipMalloc(&d.prefill_workspace, need), "allocate known prefill shared workspace");
                d.prefill_workspace_bytes = need;
            }
            const auto e = layer + d.parity * d.pending.size();
            check(hipEventRecord(d.forked[e], scope.stream), "record known prefill shared fork");
            check(hipStreamWaitEvent(d.side, d.forked[e], 0), "order known prefill shared stream");
            shared_expert_known_prefill_rows_launch_detached(rows, shared_for(rows.first(8U)), d.side,
                reinterpret_cast<std::uintptr_t>(d.prefill_workspace), d.prefill_workspace_bytes);
            // With a step observer the join is recorded after the observer's copies of this step (enqueued on the
            // side stream, see the after_step wrapper); without one, here.
            if (prefill_observed) prefill_join_due = e + 1U;
            else check(hipEventRecord(d.joined[e], d.side), "record known prefill shared join");
            d.pending[layer] = 1U;
            ++prefill_detached_shared;
            return true;
        }
        // Every step after the fork other than the routed experts (the MoE combine first) runs after the shared expert.
        if (step.op != SerialOp::moe_experts) join_all_shared();
        return false;
    }
    void join_all_shared() {
        if (!shared_detach) return;
        for (std::uint32_t l = 0; l < shared_detach->pending.size(); ++l) join_shared(l);
    }
    // The observer of a detached step runs with the side stream as the execution stream (its copies follow the
    // step there), then the join is recorded.
    void prefill_observe(std::size_t i, const std::function<ExecutionDispatchContext(std::uint32_t)>& row,
        const HipKnownPrefixWindow::PrefillStepObserver& inner) {
        if (!prefill_join_due) { inner(i, row); return; }
        auto& d = *shared_detach;
        const auto e = prefill_join_due - 1U;
        prefill_join_due = 0U;
        const auto* saved = active_hip_launch_context;
        HipLaunchContext side = *saved;
        side.stream = d.side;
        active_hip_launch_context = &side;
        try { inner(i, row); } catch (...) { active_hip_launch_context = saved; throw; }
        active_hip_launch_context = saved;
        check(hipEventRecord(d.joined[e], d.side), "record known prefill shared join");
    }
    std::uint64_t prefill_detached_shared = 0, prefill_join_due = 0;
    bool partial_current = false;
    std::uint32_t block_rows = 0;
    bool prefill_observed = false;
    // The owner's stream waits for the layer's detached shared expert (at its MoE combine, or before the
    // routed commit when the combine is folded into it).
    void join_shared(std::uint32_t layer) {
        if (!shared_detach || !shared_detach->pending[layer]) return;
        auto& d = *shared_detach;
        check(hipStreamWaitEvent(scope.stream, d.joined[layer + d.parity * d.pending.size()], 0), "join known-window shared stream");
        d.pending[layer] = 0U;
    }
    // Window HC folds (each opt-in; see hyper_mix_known_window_rows_launch_folded):
    // STRIX_WINDOW_INJECT_NORM=1: an inject rows step followed by its HC rows step runs as inject+norm, split-K,
    //   output; STRIX_WINDOW_COMBINE_INJECT=1: a MoE-combine rows step, the MLP inject and the next HC rows step
    //   run as combine+inject+norm, split-K, output. The folded steps are recorded as completed without a launch.
    static bool env_on(const char* name) { const char* v = std::getenv(name); return v && v[0] == '1'; }
    bool inject_norm_enabled() const { return allowed_folds.inject_norm && strix_window_inject_norm_enabled(); }
    bool combine_inject_enabled() const {
        if(!allowed_folds.combine_inject) return false;
        static const bool on = env_on("STRIX_WINDOW_COMBINE_INJECT"); return on;
    }
    bool commit_combine_enabled() const { return allowed_folds.commit_combine && strix_window_commit_combine_enabled(); }
    std::vector<std::size_t> folded_steps;
    bool hc_rows_step(std::size_t i) const {
        const auto op = serial.execution.steps[i].op;
        const auto v = window.r2_variant[i];
        return (op == SerialOp::attn_hyper_mix || op == SerialOp::mlp_hyper_mix || op == SerialOp::final_hyper_mix) &&
            (v == KnownWindowR2Variant::hyper_mix_rows2_v1 || v == KnownWindowR2Variant::hyper_mix_rows4_v1 ||
             v == KnownWindowR2Variant::hyper_mix_rows8_v1);
    }
    bool inject_rows_step(std::size_t i) const {
        const auto op = serial.execution.steps[i].op;
        return (op == SerialOp::attn_inject || op == SerialOp::mlp_inject) && window.r2_variant[i] == KnownWindowR2Variant::inject_rows_v1;
    }
    bool fold_window_steps(std::size_t i, const std::vector<std::vector<ExecutionDispatchContext>>& contexts) {
        if (!folded_steps.empty() && folded_steps.front() == i) { folded_steps.erase(folded_steps.begin()); return true; }
        if (!folded_steps.empty()) throw std::logic_error("known-window folded step was not reached in order");
        const auto steps = serial.execution.steps.size();
        const auto same_rows = [&](std::size_t a, std::size_t b) { return contexts[a].size() == contexts[b].size(); };
        // STRIX_WINDOW_COMMIT_COMBINE=1: the routed rows step and the MoE combine after it; the shared stream is
        // joined just before the commit.
        if (commit_combine_enabled() && serial.execution.steps[i].op == SerialOp::moe_experts && i + 1U < steps &&
            serial.execution.steps[i + 1U].op == SerialOp::moe_combine &&
            window.r2_variant[i + 1U] == KnownWindowR2Variant::moe_combine_rows_v1 && same_rows(i, i + 1U) &&
            serial.execution.steps[i].layer && (window.r2_variant[i] == KnownWindowR2Variant::routed_a8_union_rows2_v1 ||
             window.r2_variant[i] == KnownWindowR2Variant::routed_a8_union_rows4_v1 ||
             window.r2_variant[i] == KnownWindowR2Variant::routed_a8_union_rows8_v1 ||
             window.r2_variant[i] == KnownWindowR2Variant::routed_q51_union_rows2_v1 ||
             window.r2_variant[i] == KnownWindowR2Variant::routed_q51_union_rows4_v1 ||
             window.r2_variant[i] == KnownWindowR2Variant::routed_q51_union_rows8_v1)) {
            const auto combine = moe_combine_known_window_rows_views(contexts[i + 1U], elementwise_config);
            const auto n = contexts[i].size();
            const auto layer = *serial.execution.steps[i].layer;
            if (elementwise_config.activation == HipActivationStorage::bf16 &&
                moe_experts_known_window_rows_launch_combined(contexts[i], routed_config, routed_workspace,
                    std::span<const void* const>(combine.first, n), std::span<const void* const>(combine.second, n),
                    std::span<void* const>(combine.output, n), [&, layer] { join_shared(layer); }, active_rows)) {
                folded_steps = {i + 1U};
                return true;
            }
        }
        if (combine_inject_enabled() && serial.execution.steps[i].op == SerialOp::moe_combine &&
            window.r2_variant[i] == KnownWindowR2Variant::moe_combine_rows_v1 && i + 2U < steps &&
            inject_rows_step(i + 1U) && hc_rows_step(i + 2U) && same_rows(i, i + 1U) && same_rows(i, i + 2U)) {
            const auto combine = moe_combine_known_window_rows_views(contexts[i], elementwise_config);
            const auto inject = inject_known_window_rows_views(contexts[i + 1U], elementwise_config);
            if (hyper_mix_known_window_rows_launch_folded(&combine, inject, contexts[i + 2U], hyper_for(contexts[i + 2U]),
                    elementwise_config)) {
                folded_steps = {i + 1U, i + 2U};
                return true;
            }
        }
        if (inject_norm_enabled() && inject_rows_step(i) && i + 1U < steps && hc_rows_step(i + 1U) && same_rows(i, i + 1U)) {
            const auto inject = inject_known_window_rows_views(contexts[i], elementwise_config);
            if (hyper_mix_known_window_rows_launch_folded(nullptr, inject, contexts[i + 1U], hyper_for(contexts[i + 1U]),
                    elementwise_config)) {
                folded_steps = {i + 1U};
                return true;
            }
        }
        return false;
    }
    // STRIX_QPF2K_HC bits 1 / 2 (tiled prefill, full-T entries): an inject step followed by the HC step that reads its
    // streams (bit 1), or a MoE combine, the MLP inject and that HC step (bit 2), run as the folded inject + norm kernel
    // and the HC projection / output at the first step; the later steps are then recorded without a launch.
    std::vector<std::size_t> prefill_folded;
    bool prefill_fold(std::size_t i, std::span<const ExecutionDispatchContext> rows, const std::function<bool(std::size_t)>& full,
        const std::function<std::vector<ExecutionDispatchContext>(std::size_t)>& contexts_of) {
        if (!prefill_folded.empty() && prefill_folded.front() == i) { prefill_folded.erase(prefill_folded.begin()); return true; }
        if (!prefill_folded.empty()) throw std::logic_error("known prefill folded step was not reached in order");
        const auto& st = serial.execution.steps;
        const auto inject_step = [&](std::size_t s) {
            return s < st.size() && (st[s].op == SerialOp::attn_inject || st[s].op == SerialOp::mlp_inject) && full(s);
        };
        const auto hc_step = [&](std::size_t s) {
            return s < st.size() && (st[s].op == SerialOp::attn_hyper_mix || st[s].op == SerialOp::mlp_hyper_mix ||
                st[s].op == SerialOp::final_hyper_mix) && full(s);
        };
        if (st[i].op == SerialOp::moe_combine && hyper_mix_prefill_fold_enabled(true) && inject_step(i + 1U) && hc_step(i + 2U)) {
            const auto inject = contexts_of(i + 1U), hc = contexts_of(i + 2U);
            if (hyper_mix_known_prefill_rows_launch_folded(rows, inject, hc, hyper_for(hc), elementwise_config)) {
                prefill_folded = {i + 1U, i + 2U};
                return true;
            }
        }
        if (inject_step(i) && hyper_mix_prefill_fold_enabled(false) && hc_step(i + 1U)) {
            const auto hc = contexts_of(i + 1U);
            if (hyper_mix_known_prefill_rows_launch_folded({}, rows, hc, hyper_for(hc), elementwise_config)) {
                prefill_folded = {i + 1U};
                return true;
            }
        }
        return false;
    }
    // STRIX_HC_PERSISTENT=1 STRIX_HC_PERSISTENT_INJECT=1: an inject rows step followed by its HC rows step
    // (attn inject -> MLP mix, MLP inject -> next attention mix or the final mix) runs as one persistent HC
    // launch at the inject step; the HC step is then recorded as completed without a launch.
    static bool hc_inject_fusion_enabled() {
        return strix_hc_inject_fusion_enabled();
    }
    std::size_t fused_hc_step = SIZE_MAX;
    bool fuse_hc_inject(std::size_t i, const std::vector<std::vector<ExecutionDispatchContext>>& contexts) {
        if (fused_hc_step == i) { fused_hc_step = SIZE_MAX; return true; }
        if (fused_hc_step != SIZE_MAX) throw std::logic_error("known-window fused HC step was not reached");
        const auto& step = serial.execution.steps[i];
        if ((step.op != SerialOp::attn_inject && step.op != SerialOp::mlp_inject) ||
            window.r2_variant[i] != KnownWindowR2Variant::inject_rows_v1 || i + 1U >= serial.execution.steps.size())
            return false;
        const auto next = serial.execution.steps[i + 1U].op;
        const auto variant = window.r2_variant[i + 1U];
        if ((next != SerialOp::attn_hyper_mix && next != SerialOp::mlp_hyper_mix && next != SerialOp::final_hyper_mix) ||
            (variant != KnownWindowR2Variant::hyper_mix_rows2_v1 && variant != KnownWindowR2Variant::hyper_mix_rows4_v1 &&
             variant != KnownWindowR2Variant::hyper_mix_rows8_v1) || contexts[i].size() != contexts[i + 1U].size())
            return false;
        if (!hyper_mix_known_window_rows_launch_injected(contexts[i], elementwise_config, contexts[i + 1U],
                hyper_for(contexts[i + 1U])))
            return false;
        fused_hc_step = i + 1U;
        return true;
    }
    void require_shared_joined() {
        if (!shared_detach) return;
        if (std::any_of(shared_detach->pending.begin(), shared_detach->pending.end(), [](auto p) { return p != 0U; }))
            throw std::logic_error("known-window shared expert was detached without its combine");
        shared_detach->parity ^= 1U;
    }
    const HipSharedExpertLauncherConfig& shared_for(std::span<const ExecutionDispatchContext> rows) const {
        if (!rows[0].kernel_id) throw std::invalid_argument("known-window shared row has no serial kernel id");
        return rows[0].kernel_id->ends_with(shared_expert_prefetch::kernel_suffix) ? shared_config : shared_fallback;
    }
    const HipIndexedAttentionLauncherConfig& qsa_for(std::span<const ExecutionDispatchContext> rows) const {
        if (!rows[0].kernel_id) throw std::invalid_argument("known-window QSA row has no serial kernel id");
        return rows[0].kernel_id->ends_with(qsa_input_prefetch::kernel_suffix) ? qsa_config : qsa_fallback;
    }
    void setup_r2() {
        shared_config.hidden_size = u32(model.hidden_size);
        shared_config.intermediate_size = u32(model.shared_expert_intermediate_size);
        shared_config.activation = config.activation;
        shared_config.prefetch_q8 = config.prefetch_q8_shared_expert;
        shared_config.prefetch_eligible_layers = config.shared_expert_prefetch_layers;
        shared_config.prefetch_stats = config.shared_expert_prefetch_stats;
        shared_config.dense_a8_projections = config.dense_a8;
        // Fast prefill profile (prefill catalog only): FP16 WMMA dense projections in 32..256-row prompt launches.
        shared_config.dense_f16_prefill = (config.fast_prefill_components & kFastPrefillDense) != 0U;
        shared_config.dense_g64_prefill = (config.fast_prefill_components & kFastPrefillDense8) != 0U;
        shared_fallback = shared_config;
        shared_fallback.prefetch_q8 = false; shared_fallback.prefetch_eligible_layers.clear();
        gdn_config.model = model; gdn_config.activation = config.activation;
        gdn_config.virtual_q8_input_projections = config.virtual_q8_gdn_input;
        gdn_config.prefetch_q8_input_projections = config.prefetch_q8_gdn_input;
        gdn_config.wave8_q8_input_projections = config.wave8_q8_gdn_input;
        gdn_config.virtual_q8_output_projection = config.virtual_q8_gdn_output;
        gdn_config.wave8_q8_output_projection = config.wave8_q8_gdn_output;
        gdn_config.dense_a8_projections = config.dense_a8;
        gdn_config.dense_trellis_projections = numerical_profile_dense_trellis(config.numerical_profile);
        gdn_config.dense_f16_prefill = (config.fast_prefill_components & kFastPrefillDense) != 0U;
        gdn_config.dense_g64_prefill = (config.fast_prefill_components & kFastPrefillDense8) != 0U;
        r2_entries[0] = {KnownWindowR2Variant::shared_q8_virtual256_r2_v1, this,
            [](std::span<const ExecutionDispatchContext, 2> rows, void* object) {
                const auto& self = *static_cast<Impl*>(object);
                shared_expert_known_window_r2_preflight(rows, self.shared_for(rows));
            },
            [](std::span<const ExecutionDispatchContext, 2> rows, void* object) {
                const auto& self = *static_cast<Impl*>(object);
                shared_expert_known_window_r2_launch(rows, self.shared_for(rows));
            }};
        r2_entries[1] = {KnownWindowR2Variant::gdn_q8_virtual256_r2_v1, this,
            [](std::span<const ExecutionDispatchContext, 2> rows, void* object) {
                gdn_known_window_r2_preflight(rows, static_cast<Impl*>(object)->gdn_config);
            },
            [](std::span<const ExecutionDispatchContext, 2> rows, void* object) {
                gdn_known_window_r2_launch(rows, static_cast<Impl*>(object)->gdn_config);
            }};
        r2_entries[2] = {KnownWindowR2Variant::gdn_q8_rows2_f32_gates_r1_v1, this,
            [](std::span<const ExecutionDispatchContext, 2> rows, void* object) {
                gdn_known_window_r2_f32_gates_preflight(rows, static_cast<Impl*>(object)->gdn_config);
            },
            [](std::span<const ExecutionDispatchContext, 2> rows, void* object) {
                gdn_known_window_r2_f32_gates_launch(rows, static_cast<Impl*>(object)->gdn_config);
            }};
    }
    void setup_prefill_entries() {
        // STRIX_PREFILL_FULL_HC=0: keep HC rows in separate 8-row launches.
        static const bool hc=[]{const char* v=std::getenv("STRIX_PREFILL_FULL_HC");return !v||v[0]!='0';}();
        if(hc&&window.hyper_mix_rows_steps)
            prefill_entries.push_back({KnownWindowR2Variant::hyper_mix_rows8_v1,this,
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    const auto& self=*static_cast<Impl*>(object);
                    hyper_mix_known_prefill_rows_preflight(rows,self.hyper_for(rows));
                },
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    const auto& self=*static_cast<Impl*>(object);
                    hyper_mix_known_prefill_rows_launch(rows,self.hyper_for(rows));
                }});
        // STRIX_PREFILL_FULL_PLE=0: PLE lookups and projections per 8-row tile (each tile still chains the history).
        static const bool ple=[]{const char* v=std::getenv("STRIX_PREFILL_FULL_PLE");return !v||v[0]!='0';}();
        if(ple&&window.ple_rows_steps)
            prefill_entries.push_back({KnownWindowR2Variant::ple_rows_host_staged_v1,this,
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    static_cast<Impl*>(object)->ple_window->preflight(rows);
                },
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    static_cast<Impl*>(object)->ple_window->launch(rows);
                },0,false,[](std::span<const ExecutionDispatchContext> rows,void* object) {
                    static_cast<Impl*>(object)->ple_window->stage_serve(rows);
                }});
        // STRIX_PREFILL_FULL_QSA=0: keep QSA rows in 8-row tiles.
        static const bool qsa=[]{const char* v=std::getenv("STRIX_PREFILL_FULL_QSA");return !v||v[0]!='0';}();
        if(prefill->external_qsa_score_bytes && !qsa)
            throw std::invalid_argument("compact QSA owner requires the full-prefill ring entry");
        if(qsa&&window.qsa_rows_steps)
            for(std::size_t i=0;i<window.mode.size();++i) {
                if(window.mode[i]!=KnownWindowStepMode::qsa_rows) continue;
                const auto variant=window.r2_variant[i];
                if(std::any_of(prefill_entries.begin(),prefill_entries.end(),[&](const auto& e){return e.variant==variant;})) continue;
                prefill_entries.push_back({variant,this,
                    [](std::span<const ExecutionDispatchContext> rows,void* object) {
                        const auto& self=*static_cast<Impl*>(object);
                        indexed_attention_known_prefill_rows_preflight(rows,self.qsa_for(rows));
                    },
                    [](std::span<const ExecutionDispatchContext> rows,void* object) {
                        const auto& self=*static_cast<Impl*>(object);
                        indexed_attention_known_prefill_rows_launch(rows,self.qsa_for(rows));
                    },0,true});
            }
        // STRIX_PREFILL_FULL_SHARED=0: keep the shared expert in 8-row tiles.
        static const bool shared=[]{const char* v=std::getenv("STRIX_PREFILL_FULL_SHARED");return !v||v[0]!='0';}();
        if(shared)
            prefill_entries.push_back({KnownWindowR2Variant::shared_q8_virtual256_r8_v1,this,
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    const auto& self=*static_cast<Impl*>(object);
                    shared_expert_known_prefill_rows_preflight(rows,self.shared_for(rows.first(8U)));
                },
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    const auto& self=*static_cast<Impl*>(object);
                    shared_expert_known_prefill_rows_launch(rows,self.shared_for(rows.first(8U)));
                }});
        // STRIX_PREFILL_FULL_ELEMENTWISE=0: keep inject / MoE-combine rows in 8-row launches.
        static const bool elementwise=[]{const char* v=std::getenv("STRIX_PREFILL_FULL_ELEMENTWISE");return !v||v[0]!='0';}();
        if(elementwise) {
            prefill_entries.push_back({KnownWindowR2Variant::inject_rows_v1,this,
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    inject_known_prefill_rows_preflight(rows,static_cast<Impl*>(object)->elementwise_config);
                },
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    inject_known_prefill_rows_launch(rows,static_cast<Impl*>(object)->elementwise_config);
                }});
            prefill_entries.push_back({KnownWindowR2Variant::moe_combine_rows_v1,this,
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    moe_combine_known_prefill_rows_preflight(rows,static_cast<Impl*>(object)->elementwise_config);
                },
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    moe_combine_known_prefill_rows_launch(rows,static_cast<Impl*>(object)->elementwise_config);
                }});
        }
        // STRIX_PREFILL_FULL_ROUTER=0: keep router rows in separate 8-row launches.
        static const bool router=[]{const char* v=std::getenv("STRIX_PREFILL_FULL_ROUTER");return !v||v[0]!='0';}();
        if(router)
            prefill_entries.push_back({KnownWindowR2Variant::moe_route_rows_v1,this,
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    moe_route_known_prefill_rows_preflight(rows,static_cast<Impl*>(object)->route_config);
                },
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    moe_route_known_prefill_rows_launch(rows,static_cast<Impl*>(object)->route_config);
                }});
        // STRIX_PREFILL_FULL_GDN=0 / STRIX_PREFILL_FULL_MOE=0: keep that step in 8-row tiles.
        static const bool gdn=[]{const char* v=std::getenv("STRIX_PREFILL_FULL_GDN");return !v||v[0]!='0';}();
        static const bool moe=[]{const char* v=std::getenv("STRIX_PREFILL_FULL_MOE");return !v||v[0]!='0';}();
        if(q51_known_rows::prefill_csr_enabled(moe,window.routed_expert_union_steps,
            config.a8_down_q51,config.a8_down_q80,prefill->logical_rows)) {
            const auto bytes=q51_known_rows::prefill_csr_inventory(prefill->logical_rows,
                u32(model.num_experts),u32(model.num_experts_per_token)).fixed;
            void* memory=nullptr;
            guard.run([&] { check(hipMalloc(&memory,bytes),"allocate known prefill routed CSR"); });
            std::shared_ptr<void> owner(memory,[](void* p){ if(p) (void)hipFree(p); });
            const auto base=reinterpret_cast<std::uintptr_t>(memory);
            if(base%256U) throw std::runtime_error("known prefill routed CSR allocation is not 256-byte aligned");
            status_disjoint(base,bytes);
            prefill_routed_workspace={{execution_identity(),base,bytes,routed_workspace.memory.allocation_generation^(1ULL<<63U)},owner};
            prefill_entries.push_back({KnownWindowR2Variant::routed_a8_union_rows8_v1,this,
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    auto& self=*static_cast<Impl*>(object);
                    moe_experts_known_prefill_rows_preflight(rows,self.routed_config,self.routed_workspace,
                        self.prefill_union(static_cast<std::uint32_t>(rows.size())));
                },
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    auto& self=*static_cast<Impl*>(object);
                    const auto count=static_cast<std::uint32_t>(rows.size());
                    moe_experts_known_prefill_rows_launch(rows,self.routed_config,self.routed_workspace,self.prefill_union(count));
                    self.prefill_union_rows=count==self.prefill->logical_rows?0U:count;
                }});
        }
        if(!gdn) return;
        prefill_entries.push_back({KnownWindowR2Variant::gdn_q8_rows8_f32_gates_r1_v1,this,
            [](std::span<const ExecutionDispatchContext> rows,void* object) {
                gdn_known_prefill_rows_preflight(rows,static_cast<Impl*>(object)->gdn_config);
            },
            [](std::span<const ExecutionDispatchContext> rows,void* object) {
                gdn_known_prefill_rows_launch(rows,static_cast<Impl*>(object)->gdn_config);
            }});
    }
    void setup_rows() {
        rows_entries.resize(4);
        for(std::size_t i=0;i<2U;++i) {
            rows_entries[i]={i==0U ? KnownWindowR2Variant::shared_q8_virtual256_r4_v1 :
                KnownWindowR2Variant::shared_q8_virtual256_r8_v1,this,
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    const auto& self=*static_cast<Impl*>(object);
                    shared_expert_known_window_rows_preflight(rows,self.shared_for(rows));
                },
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    const auto& self=*static_cast<Impl*>(object);
                    shared_expert_known_window_rows_launch(rows,self.shared_for(rows));
                }};
            rows_entries[i+2U]={i==0U ? KnownWindowR2Variant::gdn_q8_rows4_f32_gates_r1_v1 :
                KnownWindowR2Variant::gdn_q8_rows8_f32_gates_r1_v1,this,
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    gdn_known_window_rows_preflight(rows,static_cast<Impl*>(object)->gdn_config);
                },
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    gdn_known_window_rows_launch(rows,static_cast<Impl*>(object)->gdn_config);
                }};
        }
        if(!window.qsa_rows_steps) return;
        qsa_config.model=model; qsa_config.activation=config.activation;
        qsa_config.prefetch_q8_input=config.prefetch_q8_qsa_input;
        qsa_config.prefetch_eligible_layers=config.qsa_input_prefetch_layers;
        qsa_config.prefetch_stats=config.qsa_input_prefetch_stats;
        qsa_config.wave8_q8_output=config.wave8_q8_qsa_output;
        qsa_config.dense_a8_projections=config.dense_a8;
        qsa_config.dense_trellis_projections=numerical_profile_dense_trellis(config.numerical_profile);
        qsa_config.dense_f16_prefill=(config.fast_prefill_components & kFastPrefillDense) != 0U;
        qsa_config.dense_g64_prefill=(config.fast_prefill_components & kFastPrefillDense8) != 0U;
        qsa_config.qsa_fast_prefill=(config.fast_prefill_components & kFastPrefillQsa) != 0U;
        if(!prefill) {
            qsa_config.committed_pool=resources.qsa_committed_pool;
            const char* pool=std::getenv("STRIX_QKERN_QSA_PERSISTENT");
            if(!qsa_config.committed_pool && pool && pool[0]=='1') qsa_config.window_pool=make_hip_qsa_window_pool();
        }
        if(prefill && prefill->external_qsa_score_bytes) {
            qsa_config.exact_score_owner=make_hip_qsa_exact_score_owner(model.effective_max_positions(),
                prefill->qsa_score_ring_workspace_bytes,scope.stream,scratch.allocation_generation,prefill->qsa_score_ring_rows);
        }
        qsa_fallback=qsa_config;
        qsa_fallback.prefetch_q8_input=false; qsa_fallback.prefetch_eligible_layers.clear();
        // One entry per QSA variant actually selected (layers may differ in indexer format).
        for(std::size_t i=0;i<window.mode.size();++i) {
            if(window.mode[i]!=KnownWindowStepMode::qsa_rows) continue;
            const auto variant=window.r2_variant[i];
            if(std::any_of(rows_entries.begin(),rows_entries.end(),[&](const auto& e){return e.variant==variant;})) continue;
            rows_entries.push_back({variant,this,
            [](std::span<const ExecutionDispatchContext> rows,void* object) {
                const auto& self=*static_cast<Impl*>(object);
                indexed_attention_known_window_rows_preflight(rows,self.qsa_for(rows));
            },
            [](std::span<const ExecutionDispatchContext> rows,void* object) {
                const auto& self=*static_cast<Impl*>(object);
                indexed_attention_known_window_rows_launch(rows,self.qsa_for(rows));
            }});
        }
    }
    void setup_routed_union() {
        const auto& workspace=routed_workspace.memory;
        if(!window.routed_expert_union_steps) {
            if(routed_workspace.owner || !empty(workspace))
                throw std::invalid_argument("unused known-window routed union workspace");
            return;
        }
        if(!config.q4k_a8g32_gate_up || !config.grouped_exact_q51_down ||
            !config.q51_slot_parallel_down || !config.deferred_moe_status || !routed_workspace.owner ||
            workspace.execution_identity!=execution_identity() ||
            !workspace.arena_base || workspace.arena_base%256U || !workspace.allocation_generation ||
            workspace.arena_bytes!=q51_known_rows::required_scratch_bytes(window.logical_rows,
                u32(model.num_experts),u32(model.num_experts_per_token)) ||
            workspace.arena_bytes>std::numeric_limits<std::uintptr_t>::max()-workspace.arena_base)
            throw std::invalid_argument("known-window routed union requires its exact retained extra workspace and configured serial policy");
        status_disjoint(workspace.arena_base,workspace.arena_bytes);
        routed_config.hidden_size=u32(model.hidden_size);
        routed_config.intermediate_size=u32(model.moe_intermediate_size);
        routed_config.expert_count=u32(model.num_experts);
        routed_config.top_k=u32(model.num_experts_per_token);
        routed_config.activation=config.activation;
        routed_config.q4k_a8g32_gate_up=config.q4k_a8g32_gate_up;
        routed_config.batched_slot_activation=config.batched_moe_activation;
        routed_config.grouped_exact_q51_down=config.grouped_exact_q51_down;
        routed_config.q51_slot_parallel_down=config.q51_slot_parallel_down;
        routed_config.deferred_moe_status=config.deferred_moe_status;
        routed_config.a8_down_q51=config.a8_down_q51;
        routed_config.a8_down_q80=config.a8_down_q80;
        routed_config.q5k_a8g32_gate_up=config.q5k_a8g32_gate_up;
        // fast-prefill-f16wmma-v1 moe component: T-row prefill blocks run the routed MoE in FP16 WMMA.
        routed_config.prefill_f16wmma=(config.fast_prefill_components & kFastPrefillMoe) != 0U;
        routed_config.prefill_moe_route_bf16=(config.fast_prefill_components & kFastPrefillMoeRouteBF16) != 0U;
        routed_config.q4k_a8_stats=config.q4k_a8_stats;
        routed_config.q4k_a8_quiescence=std::make_shared<HipSetupQuiescenceGuard>();
        for(std::size_t i=0;i<window.mode.size();++i)
            if(window.mode[i]==KnownWindowStepMode::routed_expert_union)
                last_union_layer=serial.execution.steps[i].layer.value();
        const bool a8=config.a8_down_q51 || config.a8_down_q80;
        const auto variant=a8 ? (window.logical_rows==2U?KnownWindowR2Variant::routed_a8_union_rows2_v1:
            window.logical_rows==4U?KnownWindowR2Variant::routed_a8_union_rows4_v1:KnownWindowR2Variant::routed_a8_union_rows8_v1) :
            window.logical_rows==2U?KnownWindowR2Variant::routed_q51_union_rows2_v1:
            window.logical_rows==4U?KnownWindowR2Variant::routed_q51_union_rows4_v1:KnownWindowR2Variant::routed_q51_union_rows8_v1;
        rows_entries.push_back({variant,this,
            [](std::span<const ExecutionDispatchContext> rows,void* object) {
                const auto& self=*static_cast<Impl*>(object);
                moe_experts_known_window_rows_preflight(rows,self.routed_config,self.routed_workspace);
            },
            [](std::span<const ExecutionDispatchContext> rows,void* object) {
                const auto& self=*static_cast<Impl*>(object);
                moe_experts_known_window_rows_launch(rows,self.routed_config,self.routed_workspace,self.active_rows);
            }});
    }
    const HipHyperMixLauncherConfig& hyper_for(std::span<const ExecutionDispatchContext> rows) const {
        if (!rows[0].kernel_id) throw std::invalid_argument("known-window HC row has no serial kernel id");
        return rows[0].kernel_id->ends_with(lane_chain_hyper_output_suffix) ? hyper_config : hyper_fallback;
    }
    void setup_hyper_rows() {
        if(!window.hyper_mix_rows_steps) return;
        hyper_config.hidden_size=u32(model.hidden_size); hyper_config.hc_count=u32(model.hc_count);
        hyper_config.lowrank=u32(model.hc_lowrank); hyper_config.rms_epsilon=static_cast<float>(model.rms_norm_epsilon);
        hyper_config.activation=config.activation; hyper_config.parallel=config.parallel_hyper_mix;
        hyper_config.staged_projections=config.staged_hyper_projections; hyper_config.fused_injection=config.fused_staged_hyper_mix;
        hyper_config.typed_projection=config.typed_hyper_projection; hyper_config.prefetch_rows=config.prefetch_typed_hyper_rows;
        hyper_config.prefetch_q8_output=config.prefetch_q8_hyper_output; hyper_config.lane_chain_output=config.lane_chain_hyper_output;
        hyper_config.splitk_projection=config.splitk_hyper_projection; hyper_config.splitk_projection_fp64=config.splitk_hyper_projection_fp64;
        hyper_config.splitk_deferred_norm=config.splitk_deferred_norm;
        hyper_config.prefetch_hc_projection_next_tile=config.prefetch_hc_projection_next_tile;
        hyper_config.arithmetic_stats=config.arithmetic_stats;
        hyper_config.fast_prefill_wmma=(config.fast_prefill_components & kFastPrefillHc) != 0U;
        hyper_fallback=hyper_config; hyper_fallback.lane_chain_output=false;
        rows_entries.push_back({window.logical_rows==2U?KnownWindowR2Variant::hyper_mix_rows2_v1:
            window.logical_rows==4U?KnownWindowR2Variant::hyper_mix_rows4_v1:KnownWindowR2Variant::hyper_mix_rows8_v1,this,
            [](std::span<const ExecutionDispatchContext> rows,void* object) {
                const auto& self=*static_cast<Impl*>(object);
                hyper_mix_known_window_rows_preflight(rows,self.hyper_for(rows));
            },
            [](std::span<const ExecutionDispatchContext> rows,void* object) {
                const auto& self=*static_cast<Impl*>(object);
                hyper_mix_known_window_rows_launch(rows,self.hyper_for(rows));
            }});
    }
    GdnWindowControl* gdn_control() const noexcept { return static_cast<GdnWindowControl*>(gdn_deferred->control.get()); }
    void bind_deferred() {
        void* device_control=nullptr;
        check(hipHostGetDevicePointer(&device_control,gdn_deferred->control.get(),0),"deferred GDN control device view");
        gdn_window_config.retained=gdn_deferred->retained.get();
        gdn_window_config.control=static_cast<const GdnWindowControl*>(device_control);
    }
    void setup_deferred() {
        if(window.gdn_deferred_rows_steps) {
            const auto per_layer=gdn_window_retained_bytes(model);
            const auto bytes=per_layer*static_cast<std::uint64_t>(model.gdn_layers);
            gdn_deferred=std::make_shared<DeferredGdn>();
            void* retained=nullptr; void* control=nullptr;
            guard.run([&] {
                check(hipMalloc(&retained,bytes),"allocate deferred GDN retained rows");
                gdn_deferred->retained.reset(retained);
                check(hipHostMalloc(&control,sizeof(GdnWindowControl),hipHostMallocMapped|hipHostMallocCoherent),
                    "allocate deferred GDN control");
                gdn_deferred->control.reset(control);
            });
            *gdn_control()=GdnWindowControl{};
            gdn_window_config.gdn=gdn_config; gdn_window_config.retained_bytes=bytes;
            bind_deferred();
            const auto variant=window.logical_rows==2U?KnownWindowR2Variant::gdn_deferred_rows2_v1:
                window.logical_rows==4U?KnownWindowR2Variant::gdn_deferred_rows4_v1:KnownWindowR2Variant::gdn_deferred_rows8_v1;
            rows_entries.push_back({variant,this,
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    gdn_window_rows_preflight(rows,static_cast<Impl*>(object)->gdn_window_config);
                },
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    auto& self=*static_cast<Impl*>(object);
                    gdn_window_rows_launch(rows,self.gdn_window_config);
                    self.remember_gdn_layer(rows[0]);
                }});
        }
        if(window.ple_rows_steps) {
            ple_window=prefill ? std::make_unique<HipPleKnownWindow>(model,config.activation,prefill->logical_rows,true,&resources) :
                std::make_unique<HipPleKnownWindow>(model,config.activation,window.logical_rows);
            rows_entries.push_back({KnownWindowR2Variant::ple_rows_host_staged_v1,this,
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    static_cast<Impl*>(object)->ple_window->preflight(rows);
                },
                [](std::span<const ExecutionDispatchContext> rows,void* object) {
                    static_cast<Impl*>(object)->ple_window->launch(rows);
                }});
        }
    }
    void remember_gdn_layer(const ExecutionDispatchContext& row) {
        for(const auto& known:gdn_commit_rows) if(known.layer==row.layer) return;
        gdn_commit_resources.push_back(*row.resources);
        auto copy=row; copy.resources=&gdn_commit_resources.back();
        gdn_commit_rows.push_back(copy);
    }
    // After a deferred window the target keeps rows [0, n_keep): GDN applies them in the next
    // window (or flush_deferred), PLE restores row n_keep-1. Host work only, no device sync.
    void commit_prefix(std::uint32_t n_keep) {
        if(!n_keep || n_keep>window.logical_rows) throw std::invalid_argument("known-window prefix commit outside the window");
        if(window.gdn_deferred_rows_steps) {
            auto& d=*gdn_deferred;
            d.slot^=1U;
            *gdn_control()=GdnWindowControl{n_keep,d.slot};
            d.pending=true;
        }
        if(ple_window) ple_window->commit(n_keep,scope.stream);
    }
    // Applies pending GDN rows to the session state before any serial GDN token.
    void flush_deferred() {
        if(!gdn_deferred || !gdn_deferred->pending) return;
        if(gdn_commit_rows.size()!=static_cast<std::size_t>(model.gdn_layers))
            throw std::logic_error("deferred GDN flush has not observed every GDN layer");
        {
            const HipLaunchScope active(scope);
            for(const auto& row:gdn_commit_rows) gdn_window_commit_launch(row,gdn_window_config);
        }
        check(hipStreamSynchronize(scope.stream),"complete deferred GDN commit");
        gdn_control()->pending_rows=0; gdn_deferred->pending=false;
    }
    void discard_deferred() {
        invalidate_hip_qsa_window_pool(qsa_config.window_pool);
        if(!gdn_deferred || !gdn_deferred->pending) return;
        check(hipStreamSynchronize(scope.stream),"discard deferred GDN rows fence");
        gdn_control()->pending_rows=0; gdn_deferred->pending=false;
    }
    void share_deferred(const Impl& primary) {
        if(!window.gdn_deferred_rows_steps || !primary.gdn_deferred || failed || primary.failed ||
            !gdn_commit_rows.empty() || gdn_deferred->pending ||
            gdn_window_config.retained_bytes!=primary.gdn_window_config.retained_bytes ||
            primary.resources.session_state_memory!=resources.session_state_memory)
            throw std::invalid_argument("known-window deferred sharing requires fresh deferred owners of one session");
        if(primary.gdn_deferred==gdn_deferred) return;
        gdn_deferred=primary.gdn_deferred;
        bind_deferred();
        preflight_memo.clear(); static_preflight_memo.clear(); recipe_memo.clear(); preflight_slots.clear();
    }
    void setup_head_rows() {
        if(!window.lm_head_rows_steps) return;
        head_config.hidden_size=u32(model.hidden_size); head_config.vocab_size=u32(model.vocab_size);
        head_config.input=config.activation; head_config.virtual_q8_wave=config.virtual_q8_lm_head;
        head_config.prefetch_q8_wave=config.prefetch_q8_lm_head; head_config.wave8_q8_wave=config.wave8_q8_lm_head;
        // STRIX_LM_HEAD_ROWS_LEAN=1|2: lean rows head (1 or 2 logits per wave) with the greedy
        // partials fused (STRIX_LM_HEAD_ROWS_GREEDY=0 keeps the separate greedy pass).
        static const std::uint32_t lean=[]{const char* v=std::getenv("STRIX_LM_HEAD_ROWS_LEAN");
            return v&&(v[0]=='1'||v[0]=='2')?static_cast<std::uint32_t>(v[0]-'0'):0U;}();
        head_config.rows_lean=qpf2k::suppressed(qpf2k::Item::rows_lean)?0U:lean;
        const auto variant=window.logical_rows==2U?KnownWindowR2Variant::lm_head_q8_prefetch8_rows2_v1:
            window.logical_rows==4U?KnownWindowR2Variant::lm_head_q8_prefetch8_rows4_v1:KnownWindowR2Variant::lm_head_q8_prefetch8_rows8_v1;
        rows_entries.push_back({variant,this,
            [](std::span<const ExecutionDispatchContext> rows,void* object) {
                lm_head_known_window_rows_preflight(rows,static_cast<Impl*>(object)->head_config);
            },
            [](std::span<const ExecutionDispatchContext> rows,void* object) {
                const auto& self=*static_cast<Impl*>(object);
                static const bool fused=[]{const char* v=std::getenv("STRIX_LM_HEAD_ROWS_GREEDY");return !v||v[0]!='0';}();
                if(self.greedy_replies && self.head_config.rows_lean && fused) {
                    lm_head_known_window_rows_launch_greedy(rows,self.head_config,self.greedy_partials,self.greedy_replies);
                    return;
                }
                lm_head_known_window_rows_launch(rows,self.head_config);
                if(!self.greedy_replies) return;
                std::array<ExecutionBufferAddress,hip_greedy_rows_max> logits{};
                for(std::size_t r=0;r<rows.size();++r) logits[r]=resolve_dispatch_write(rows[r],0);
                enqueue_hip_greedy_rows(std::span<const ExecutionBufferAddress>(logits.data(),rows.size()),
                    self.head_config.vocab_size,self.greedy_partials,self.greedy_replies,hip_execution_stream());
            }});
    }
    void attach_greedy_rows(ExecutionBufferAddress partials,HipGreedyRowsReply* replies) {
        if(executing || failed || prefill || !window.lm_head_rows_steps || !replies ||
            partials.bytes<hip_greedy_rows_scratch_bytes(window.logical_rows,u32(model.vocab_size)))
            throw std::invalid_argument("known-window greedy rows require an idle multi-row head window and sized scratch");
        status_disjoint(partials.address,partials.bytes);
        greedy_partials=partials; greedy_replies=replies;
    }
    void setup_router_elementwise_rows() {
        route_config.hidden_size=u32(model.hidden_size); route_config.expert_count=u32(model.num_experts);
        route_config.top_k=u32(model.num_experts_per_token); route_config.activation=config.activation;
        route_config.normalize_selected=model.norm_topk_probability;
        route_config.parallel_projection=config.parallel_moe_router; route_config.staged_logits=config.staged_moe_router_logits;
        route_config.ranked_topk=config.ranked_moe_router_topk; route_config.wave_topk=config.wave_moe_router_topk;
        route_config.wave_manifest_admission=config.wave_router_manifest_admission;
        route_config.router_fast_prefill=(config.fast_prefill_components & kFastPrefillRouter) != 0U;
        elementwise_config.hidden_size=u32(model.hidden_size); elementwise_config.hc_count=u32(model.hc_count);
        elementwise_config.activation=config.activation;
        rows_entries.push_back({KnownWindowR2Variant::moe_route_rows_v1,this,
            [](std::span<const ExecutionDispatchContext> rows,void* object) {
                moe_route_known_window_rows_preflight(rows,static_cast<Impl*>(object)->route_config);
            },
            [](std::span<const ExecutionDispatchContext> rows,void* object) {
                moe_route_known_window_rows_launch(rows,static_cast<Impl*>(object)->route_config);
            }});
        rows_entries.push_back({KnownWindowR2Variant::moe_combine_rows_v1,this,
            [](std::span<const ExecutionDispatchContext> rows,void* object) {
                moe_combine_known_window_rows_preflight(rows,static_cast<Impl*>(object)->elementwise_config);
            },
            [](std::span<const ExecutionDispatchContext> rows,void* object) {
                moe_combine_known_window_rows_launch(rows,static_cast<Impl*>(object)->elementwise_config);
            }});
        rows_entries.push_back({KnownWindowR2Variant::inject_rows_v1,this,
            [](std::span<const ExecutionDispatchContext> rows,void* object) {
                inject_known_window_rows_preflight(rows,static_cast<Impl*>(object)->elementwise_config);
            },
            [](std::span<const ExecutionDispatchContext> rows,void* object) {
                inject_known_window_rows_launch(rows,static_cast<Impl*>(object)->elementwise_config);
            }});
    }
    // STRIX_QPF2K_SCHED bit 2 (tiled prefill, pf route): a headless block followed by another block of this owner ends
    // without a host fence. Its completion reads land in a pinned slot behind an event; their checks run when the next
    // block of this owner completes (after waiting for that event), before that block publishes anything. The last
    // block of a prompt always fences. One deferred block at most.
    struct DeferredCompletion {
        q51_known_rows::Summary summary{};
        DeferredMoeFirstError status{};
        std::uint32_t rows_done = 0;
    };
    DeferredCompletion* deferred_slots = nullptr;
    std::array<hipEvent_t, 2> deferred_events{};
    int deferred_pending = -1;
    unsigned deferred_next = 0;
    bool defer_current = false;
    std::uint64_t deferred_blocks = 0, deferred_checked = 0;
    void enqueue_completion_reads(q51_known_rows::Summary* summary, DeferredMoeFirstError* status, std::uint32_t prefill_rows_done) {
        if(window.routed_expert_union_steps && !cohort_member)
            check(hipMemcpyAsync(summary,
                reinterpret_cast<const void*>(full_prefill_union()?prefill_union(prefill_rows_done).memory.arena_base:routed_workspace.memory.arena_base),
                sizeof(*summary),hipMemcpyDeviceToHost,scope.stream),"read latest routed expert union metadata");
        if (config.deferred_moe_status)
            check(hipMemcpyAsync(status,
                reinterpret_cast<const void*>(scope.deferred_moe_status.arena_base), sizeof(*status),
                hipMemcpyDeviceToHost, scope.stream), "read completed known-window deferred status");
    }
    void finish_deferred_completion() {
        if (deferred_pending < 0) return;
        const auto slot = static_cast<unsigned>(deferred_pending);
        check(hipEventSynchronize(deferred_events[slot]), "complete deferred known prefill block");
        deferred_pending = -1; ++deferred_checked;
        guard.require_original();
        pending_union_summary = deferred_slots[slot].summary; completion_status = deferred_slots[slot].status;
        check_completion(deferred_slots[slot].rows_done);
    }
    // STRIX_BATCH_SERVICE_MS: the eager loop retains a layer end until every native fold has closed.
    bool prefill_bundle_closed() const {
        if(!prefill_folded.empty() || !folded_steps.empty() || fused_hc_step!=SIZE_MAX || prefill_join_due) return false;
        if(shared_detach && std::any_of(shared_detach->pending.begin(),shared_detach->pending.end(),[](auto p){return p!=0U;})) return false;
#ifdef STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC
        if(!qsa_prefill_bundle_closed()) throw std::logic_error("prefill service boundary has an open QSA batch");
#endif
        return true;
    }
    void service_boundary() {
        prefill_service(scope.stream,[&] {
            check(hipStreamSynchronize(scope.stream),"drain prefill service producer");
            finish_deferred_completion();
            if(config.deferred_moe_status) {
                check(hipMemcpy(&completion_status,reinterpret_cast<const void*>(scope.deferred_moe_status.arena_base),sizeof(completion_status),hipMemcpyDeviceToHost),"read service D1");
                check_deferred_moe_status(completion_status);
            }
        });
    }
    void complete() {
        guard.require_original();
        const auto prefill_rows_done=full_prefill_union()?(prefill_union_rows?prefill_union_rows:prefill->logical_rows):0U;
        const bool defer=defer_current && prefill;
        defer_current=false;
        if(defer) {
            if(!deferred_slots) {
                void* p=nullptr;
                check(hipHostMalloc(&p,2U*sizeof(DeferredCompletion),hipHostMallocDefault),"allocate deferred known prefill slots");
                deferred_slots=static_cast<DeferredCompletion*>(p);
                for(auto& e:deferred_events) check(hipEventCreateWithFlags(&e,hipEventDisableTiming),"create deferred known prefill event");
            }
            // The previous deferred block's checks first (its event precedes this block on the stream).
            finish_deferred_completion();
            const auto slot=deferred_next;deferred_next^=1U;
            deferred_slots[slot].rows_done=prefill_rows_done;
            enqueue_completion_reads(&deferred_slots[slot].summary,&deferred_slots[slot].status,prefill_rows_done);
            check(hipEventRecord(deferred_events[slot],scope.stream),"record deferred known prefill completion");
            deferred_pending=static_cast<int>(slot);++deferred_blocks;
            return;
        }
        enqueue_completion_reads(&pending_union_summary,&completion_status,prefill_rows_done);
        check(hipStreamSynchronize(scope.stream), "complete known-window native stream");
        guard.require_original();
        // A deferred previous block of this owner: its event precedes this block, so it is complete; check it first.
        if (deferred_pending >= 0) {
            const auto mine_summary = pending_union_summary; const auto mine_status = completion_status;
            finish_deferred_completion();
            pending_union_summary = mine_summary; completion_status = mine_status;
        }
        check_completion(prefill_rows_done);
    }
    void check_completion(std::uint32_t prefill_rows_done) {
        if (config.deferred_moe_status) check_deferred_moe_status(completion_status);
        if(full_prefill_union()) {
            // One CSR over all T rows: histogram buckets saturate at 8, batches hold <= 64 assignments.
            const auto& s=pending_union_summary;
            const auto assignments=prefill_rows_done*u32(model.num_experts_per_token);
            std::uint64_t experts=0,batches=0;
            for(std::uint32_t i=0;i<8U;++i){experts+=s.rows_per_expert_histogram[i];batches+=s.batch_size_histogram[i];}
            if(s.status || s.version!=1U || s.logical_rows!=prefill_rows_done || s.assignments!=assignments ||
                !s.touched || s.touched>assignments || s.touched>model.num_experts ||
                s.batches<s.touched || s.batches>assignments || !s.maximum_assignments ||
                s.maximum_assignments>assignments || s.duplicate_assignments>assignments ||
                experts!=s.touched || batches!=s.batches ||
                std::any_of(s.reserved,s.reserved+8U,[](auto v){return v!=0U;}))
                throw std::runtime_error("routed expert union metadata disagrees with its completed prefill");
        } else if(window.routed_expert_union_steps && !cohort_member) {
            const auto& s=pending_union_summary;
            const auto union_rows=active_rows?active_rows:window.logical_rows;
            const auto assignments=union_rows*u32(model.num_experts_per_token);
            std::uint64_t experts=0,batches=0,batch_assignments=0,unique_row_assignments=0;
            for(std::uint32_t i=0;i<8U;++i) {
                experts+=s.rows_per_expert_histogram[i];
                unique_row_assignments+=static_cast<std::uint64_t>(i+1U)*s.rows_per_expert_histogram[i];
                batches+=s.batch_size_histogram[i];
                batch_assignments+=static_cast<std::uint64_t>(i+1U)*s.batch_size_histogram[i];
                if(i>=window.logical_rows && (s.rows_per_expert_histogram[i] || s.batch_size_histogram[i]))
                    throw std::runtime_error("routed expert union histogram exceeds its admitted row count");
            }
            if(s.status || s.version!=1U || s.logical_rows!=union_rows || s.assignments!=assignments ||
                !s.touched || s.touched>assignments || s.touched>model.num_experts ||
                s.batches<s.touched || s.batches>assignments || !s.maximum_assignments ||
                s.maximum_assignments>assignments || s.duplicate_assignments>assignments ||
                experts!=s.touched || batches!=s.batches || batch_assignments!=assignments ||
                unique_row_assignments+s.duplicate_assignments!=assignments ||
                std::any_of(s.reserved,s.reserved+8U,[](auto v){return v!=0U;}))
                throw std::runtime_error("routed expert union metadata disagrees with its completed window");
        }
    }
    static std::uint64_t capture_identity(std::string_view tag,std::uint64_t previous) {
        std::uint64_t hash=14695981039346656037ULL;
        for(const auto ch:tag){hash^=static_cast<unsigned char>(ch);hash*=1099511628211ULL;}
        for(unsigned shift=0;shift<64U;shift+=8U){hash^=(previous>>shift)&255U;hash*=1099511628211ULL;}
        if(!hash)throw std::logic_error("known-window capture identity is zero");return hash;
    }
    static std::uint64_t captured_recipe(KnownWindowR2Variant variant) {
        std::string_view recipe;
        switch(variant){
        case KnownWindowR2Variant::shared_q8_virtual256_r2_v1: recipe="hip-shared-known-r2-q8virtual256-3proj-f32scalar-r1act-v1";break;
        case KnownWindowR2Variant::gdn_q8_virtual256_r2_v1: recipe="hip-gdn-known-r2-5q8-staged-ordered-core-r1-v1";break;
        case KnownWindowR2Variant::gdn_q8_rows2_f32_gates_r1_v1: recipe="hip-gdn-known-r2-3q8-smallf32-r1-ordered-core-v1";break;
        case KnownWindowR2Variant::shared_q8_virtual256_r4_v1:
        case KnownWindowR2Variant::shared_q8_virtual256_r8_v1: recipe="hip-shared-known-wide-q8virtual256-3proj-f32scalar-r1act-v1";break;
        case KnownWindowR2Variant::gdn_q8_rows4_f32_gates_r1_v1:
        case KnownWindowR2Variant::gdn_q8_rows8_f32_gates_r1_v1: recipe="hip-gdn-known-wide-3q8-smallf32-r1-ordered-core-v1";break;
        case KnownWindowR2Variant::routed_q51_union_rows2_v1:
        case KnownWindowR2Variant::routed_q51_union_rows4_v1:
        case KnownWindowR2Variant::routed_q51_union_rows8_v1: recipe="hip-routed-known-q4ka8-q51-union-dot-orderedslot-v1";break;
        case KnownWindowR2Variant::routed_a8_union_rows2_v1:
        case KnownWindowR2Variant::routed_a8_union_rows4_v1:
        case KnownWindowR2Variant::routed_a8_union_rows8_v1: recipe="hip-routed-known-q4k-q5k-a8-q51-q80-a8down-union-orderedslot-v1";break;
        case KnownWindowR2Variant::hyper_mix_rows2_v1:
        case KnownWindowR2Variant::hyper_mix_rows4_v1:
        case KnownWindowR2Variant::hyper_mix_rows8_v1: recipe="hip-hc-known-rows-typed-q8-r1-ordered-v1";break;
        case KnownWindowR2Variant::qsa_q8_rows2_r1_v1:
        case KnownWindowR2Variant::qsa_q8_rows4_r1_v1:
        case KnownWindowR2Variant::qsa_q8_rows8_r1_v1: recipe="hip-qsa-known-wide-q8-append-all-r1-attend-v1";break;
        case KnownWindowR2Variant::gdn_deferred_rows2_v1:
        case KnownWindowR2Variant::gdn_deferred_rows4_v1:
        case KnownWindowR2Variant::gdn_deferred_rows8_v1: recipe="hip-gdn-window-deferred-commit-r1-tree-v1";break;
        case KnownWindowR2Variant::ple_rows_host_staged_v1: recipe="hip-ple-window-host-staged-conv-snapshots-v1";break;
        case KnownWindowR2Variant::qsa_q8_bf16idx_rows2_r1_v1:
        case KnownWindowR2Variant::qsa_q8_bf16idx_rows4_r1_v1:
        case KnownWindowR2Variant::qsa_q8_bf16idx_rows8_r1_v1: recipe="hip-qsa-known-wide-q8-bf16idx-append-all-r1-attend-v1";break;
        case KnownWindowR2Variant::lm_head_q8_prefetch8_rows2_v1:
        case KnownWindowR2Variant::lm_head_q8_prefetch8_rows4_v1:
        case KnownWindowR2Variant::lm_head_q8_prefetch8_rows8_v1: recipe="hip-lm-head-known-rows-q8prefetch8-r1tree-v1";break;
        case KnownWindowR2Variant::moe_route_rows_v1: recipe="hip-route-known-rows-tiled-ascending-chain-r1select-v1";break;
        case KnownWindowR2Variant::moe_combine_rows_v1: recipe="hip-moe-combine-known-rows-r1element-v1";break;
        case KnownWindowR2Variant::inject_rows_v1: recipe="hip-inject-known-rows-r1element-v1";break;
        default:throw std::invalid_argument("known-window capture has no closed native recipe");
        }
        return capture_identity(recipe,static_cast<std::uint32_t>(variant));
    }
    struct CapturedLauncher {
        CompiledStepLauncher delegate;
        HipMtpTargetWindowCapture* capture=nullptr;
    };
    static void capture_preflight(const ExecutionDispatchContext& context,void* object){
        const auto& wrapper=*static_cast<CapturedLauncher*>(object);
        wrapper.delegate.preflight(context,wrapper.delegate.user_data);wrapper.capture->preflight_row(context);
    }
    static void capture_launch(const ExecutionDispatchContext& context,void* object){
        const auto& wrapper=*static_cast<CapturedLauncher*>(object);
        wrapper.capture->preflight_row(context);
        wrapper.delegate.function(context,wrapper.delegate.user_data);
        wrapper.capture->record_row(context);
    }
    // Heap entries never move: observed launcher userdata and plan spans remain stable.
    // A weak key retains only the capture control block, never its GPU owner.
    struct PreparedCapture {
        std::weak_ptr<HipMtpTargetWindowCapture> key;
        HipMtpTargetWindowCapture* address=nullptr;
        std::vector<CompiledLaunchPlan> plans;
        std::vector<const CompiledLaunchPlan*> pointers;
        std::vector<CapturedLauncher> launchers;
        std::array<KnownWindowR2Entry,3> recorded_r2;
        std::vector<KnownWindowRowsEntry> recorded_rows;
    };
    std::vector<std::unique_ptr<PreparedCapture>> prepared_captures;
    bool executing=false;
    void prepare_capture(const std::shared_ptr<HipMtpTargetWindowCapture>& capture) {
        if(executing)throw std::logic_error("MTP target capture preparation requires an idle known-window owner");
        // Expired entries are reclaimed only during idle setup, never on token execution.
        std::erase_if(prepared_captures,[](const auto& entry){return entry->key.expired();});
        auto entry=std::make_unique<PreparedCapture>();
        entry->key=capture;entry->address=capture.get();entry->plans=launch_plans;
        entry->launchers.resize(logical_rows());entry->pointers.reserve(logical_rows());
        entry->recorded_r2=r2_entries;entry->recorded_rows=rows_entries;
        for(auto& recipe:entry->recorded_r2)recipe.implementation_identity=captured_recipe(recipe.variant);
        for(auto& recipe:entry->recorded_rows)recipe.implementation_identity=captured_recipe(recipe.variant);
        const auto step=capture->tap_step();
        for(std::uint32_t row=0;row<logical_rows();++row){
            auto& plan=entry->plans[row];
            if(step>=plan.steps.size() || plan.steps[step].op!=SerialOp::mlp_inject ||
               window.mode[step]!=KnownWindowStepMode::r1_row_loop ||
               !plan.steps[step].function || !plan.steps[step].preflight)
                throw std::invalid_argument("MTP observed target launcher is not its actual last MLP injection");
            auto& wrapper=entry->launchers[row];wrapper.delegate=plan.steps[step];wrapper.capture=capture.get();
            plan.steps[step].function=&capture_launch;plan.steps[step].preflight=&capture_preflight;plan.steps[step].user_data=&wrapper;
            plan.steps[step].implementation_identity=capture_identity("hip-mtp-post-mlpinject-wide-observer-v1",wrapper.delegate.implementation_identity);
            plan.plan_identity=capture_identity("hip-mtp-observed-host-launch-plan-v1",plan.plan_identity);
            entry->pointers.push_back(&plan);
        }
        prepared_captures.push_back(std::move(entry));
    }
    const PreparedCapture& prepared_capture(const std::shared_ptr<HipMtpTargetWindowCapture>& capture) const {
        for(const auto& entry:prepared_captures)
            if(entry->address==capture.get() && !entry->key.owner_before(capture) &&
               !capture.owner_before(entry->key))return *entry;
        throw std::invalid_argument("MTP target capture has no prepared plans on this actual known-window owner");
    }
    // Setup-time admission of the owner's own launch plans (see preflight_admission()).
    void preflight_admission(SerialSessionContext session, std::span<const std::uint32_t> tokens, bool static_only=false) {
        if (failed || (static_only ? (!prefill || !tokens.empty()) : (prefill || tokens.size() != logical_rows())) || executing || current_hip_launch_context())
            throw std::invalid_argument("known-window setup admission requires an idle closed window owner");
        struct ExecutionLease {
            bool& active;
            explicit ExecutionLease(bool& value):active(value){active=true;}
            ~ExecutionLease(){active=false;}
        } execution_lease(executing);
        try {
            validate_metadata();
            const HipLaunchScope active(scope);
            KnownWindowExecutionHooks hooks;
            hooks.preflight_scope = [&] { validate_scope(); };
            hooks.preflight_plan = [&] { validate_metadata(); };
            hooks.complete = [] { throw std::logic_error("known-window setup admission never completes a window"); };
            hooks.quiesce_or_fail_stop = [&] { failed = true; guard.recover_or_fail_stop(); };
            hooks.position_reader_implementations = position_readers;
            hooks.preflight_memo=&preflight_memo;hooks.recipe_memo=&recipe_memo;hooks.preflight_slots=&preflight_slots;
            hooks.arena_index=&arena_index;
            hooks.preflight_only = true;
            if(static_only) {
                hooks.static_preflight_only=true;
                hooks.static_preflight_memo=&static_preflight_memo;
                (void)execute_known_prefix_prefill_eager(*prefill,serial,resources,scratch,session,{},
                    launch_plans.front(),r2_entries,rows_entries,hooks,prefill_entries);
                return;
            }
            (void)execute_known_prefix_window_eager(window, serial, resources, scratch, session, tokens,
                std::span<const CompiledLaunchPlan* const>(launch_pointers), std::span<const KnownWindowR2Entry>(r2_entries),
                std::span<const KnownWindowRowsEntry>(rows_entries), hooks);
        } catch (...) { failed = true; throw; }
    }
    // ---- Cohort: the closed windows of several sessions in one stream of steps ----
    // Every member is the session's own idle closed owner (its rows, arena, captures, deferred GDN,
    // PLE snapshots and greedy replies); the primary is another idle owner whose row count equals
    // the members' total and whose routed-union workspace and stream run the steps shared by all
    // rows. All owners share one stream. Each member window is admitted exactly as on its own and
    // its row-loop, PLE, shared-expert, combine and inject steps are its own launches. The shared
    // steps (routed expert union, HC mixers, router, LM head and the projections of GDN and QSA)
    // run once for all rows; GDN recurrence and QSA attention run per session inside them.
    struct CohortMember {
        Impl* impl = nullptr;
        SerialSessionContext* session = nullptr;
        std::span<const std::uint32_t> tokens;
        std::shared_ptr<HipMtpTargetWindowCapture> capture;
        const PreparedCapture* observed = nullptr;
        KnownWindowExecutionHooks hooks;
        std::unique_ptr<KnownWindowRun> run;
        bool begun = false, launched = false;
    };
    static bool cohort_shared_mode(KnownWindowStepMode mode) noexcept {
        return mode == KnownWindowStepMode::routed_expert_union || mode == KnownWindowStepMode::hyper_mix_rows ||
            mode == KnownWindowStepMode::moe_route_rows || mode == KnownWindowStepMode::lm_head_rows ||
            mode == KnownWindowStepMode::gdn_deferred_rows || mode == KnownWindowStepMode::qsa_rows;
    }
    // STRIX_COHORT_SHARED=0: every step of a cohort runs per session (the same launches as alone,
    // on one stream); only the scheduling differs. STRIX_COHORT_SHARED_EXPERT=0 keeps only the
    // shared expert per session.
    static bool cohort_shared_enabled() {
        static const bool on = [] { const char* v = std::getenv("STRIX_COHORT_SHARED"); return !v || v[0] != '0'; }();
        return on;
    }
    static bool cohort_shared_expert_enabled() {
        static const bool on = [] { const char* v = std::getenv("STRIX_COHORT_SHARED_EXPERT"); return !v || v[0] != '0'; }();
        return on;
    }
    // Step i of a cohort runs once for every member's rows on the primary.
    bool cohort_step_shared(std::size_t i, std::span<CohortMember> members) const {
        if (!cohort_shared_enabled()) return false;
        const auto mode = window.mode[i];
        if (mode == KnownWindowStepMode::shared_rows) {
            if (!cohort_shared_expert_enabled()) return false;
            for (const auto& m : members) {
                const auto own = m.impl->window.mode[i];
                if ((own != KnownWindowStepMode::shared_rows && own != KnownWindowStepMode::r2_shared_q8) || !m.run->promoted_step(i) ||
                    !m.impl->detachable_shared(i)) return false;
            }
            return detachable_shared(i);
        }
        if (!cohort_shared_mode(mode)) return false;
        for (const auto& m : members) {
            if (!m.run->wide_step(i) || m.impl->window.mode[i] != mode) return false;
            if (mode == KnownWindowStepMode::qsa_rows && m.run->rows() != 2U && m.run->rows() != 4U && m.run->rows() != 8U) return false;
        }
        return true;
    }
    // STRIX_COHORT_WIDE=gdn,qsa,head,moe,combine,inject (comma list, or "all"): with two or more chunks, the
    // listed steps run once for the rows of every chunk (9..32 rows) instead of once per chunk. Per-row
    // arithmetic as in the chunk.
    static bool cohort_wide_enabled(const char* family) {
        static const std::string list = [] { const char* v = std::getenv("STRIX_COHORT_WIDE"); return std::string(v ? v : ""); }();
        if (known_window_cohort_feature_off(family)) return false;
        if (list == "all") return true;
        std::size_t at = 0;
        while (at <= list.size()) {
            const auto end = std::min(list.find(',', at), list.size());
            if (list.compare(at, end - at, family) == 0) return true;
            at = end + 1U;
        }
        return false;
    }
    static bool cohort_wide_step(std::size_t i, std::span<CohortMember> members, bool all_chunked) {
        const auto mode = members[0].impl->window.mode[i];
        const bool shared=mode==KnownWindowStepMode::shared_rows || mode==KnownWindowStepMode::r2_shared_q8;
        std::uint32_t total = 0;
        for (const auto& m : members) {
            const auto own=m.impl->window.mode[i];
            if(shared) {
                if(!m.run->promoted_step(i) || (own!=KnownWindowStepMode::shared_rows && own!=KnownWindowStepMode::r2_shared_q8)) return false;
            } else if (!m.run->wide_step(i) || own != mode) return false;
            total += m.run->rows();
        }
        if (total < 9U || total > 32U) return false;
        static const bool joint=[] {const char* v=std::getenv("STRIX_MTP_BATCH_JOINT");return v && v[0]=='1';}();
        switch (mode) {
        case KnownWindowStepMode::moe_route_rows:
            if(!joint) return false;
            for(const auto& m:members) if(!m.impl->route_config.wave_topk || m.impl->route_config.hidden_size%64U) return false;
            return cohort_wide_enabled("router");
        case KnownWindowStepMode::hyper_mix_rows:
            if(!joint) return false;
            for(const auto& m:members) {
                const auto& config=m.impl->hyper_for(m.run->step_rows(i));
                if(!config.splitk_projection || !config.lane_chain_output) return false;
            }
            return cohort_wide_enabled("hyper");
        case KnownWindowStepMode::r2_shared_q8:
        case KnownWindowStepMode::shared_rows:
            if(!joint) return false;
            if(!cohort_shared_expert_enabled() || !cohort_wide_enabled("shared")) return false;
            for(const auto& m:members) if(!m.impl->shared_for(m.run->step_rows(i)).dense_a8_projections) return false;
            return true;
        case KnownWindowStepMode::gdn_deferred_rows:
            // 16 / 32 rows: dense A8 projections only (gdn_window_cohort_enqueue).
            for (const auto& m : members) if (!m.impl->gdn_window_config.gdn.dense_a8_projections) return false;
            return cohort_wide_enabled("gdn");
        case KnownWindowStepMode::lm_head_rows: return cohort_wide_enabled("head");
        case KnownWindowStepMode::inject_rows: return cohort_wide_enabled("inject");
        // Every chunk must have a primary: a lone session joins its own detached shared expert inside its step.
        case KnownWindowStepMode::moe_combine_rows: return all_chunked && cohort_wide_enabled("combine");
        case KnownWindowStepMode::routed_expert_union: return cohort_wide_union(members);
        case KnownWindowStepMode::qsa_rows:
            // 16 / 32 rows: dense A8 only (the launcher also requires a Q8_0 output projection).
            if (!cohort_wide_enabled("qsa")) return false;
            for (const auto& m : members)
                if ((m.run->rows() != 2U && m.run->rows() != 4U && m.run->rows() != 8U) ||
                    !m.impl->qsa_for(m.run->step_rows(i)).dense_a8_projections) return false;
            return true;
        default: return false;
        }
    }
    // The routed union of every member's rows on the lead member (A8 profile, 16 or 32 rows).
    static bool cohort_wide_union(std::span<CohortMember> members) {
        if (!cohort_wide_enabled("moe")) return false;
        std::uint32_t total = 0;
        for (const auto& m : members) {
            const auto& c = m.impl->routed_config;
            if (!m.impl->window.routed_expert_union_steps || !(c.a8_down_q51 || c.a8_down_q80) || m.impl->active_rows) return false;
            total += m.impl->logical_rows();
        }
        return total >= 9U && total <= 32U;
    }
    static void launch_cohort_wide(std::size_t i, std::span<CohortMember> members) {
        const KnownWindowCohortScope cohort;
        switch (members[0].impl->window.mode[i]) {
        case KnownWindowStepMode::moe_route_rows: {
            std::vector<std::span<const ExecutionDispatchContext>> segments;
            for(const auto& m:members) segments.push_back(m.run->step_rows(i));
            moe_route_known_window_cohort_launch(segments,members[0].impl->route_config);
            break;
        }
        case KnownWindowStepMode::hyper_mix_rows: {
            std::vector<std::span<const ExecutionDispatchContext>> segments;
            for(const auto& m:members) segments.push_back(m.run->step_rows(i));
            hyper_mix_known_window_cohort_launch(segments,members[0].impl->hyper_for(segments[0]));
            break;
        }
        case KnownWindowStepMode::r2_shared_q8:
        case KnownWindowStepMode::shared_rows: {
            std::vector<std::span<const ExecutionDispatchContext>> segments;
            for(const auto& m:members) segments.push_back(m.run->step_rows(i));
            shared_expert_known_window_cohort_launch(segments,members[0].impl->shared_for(segments[0]),
                members[0].impl->scope.stream,0U,0U);
            break;
        }
        case KnownWindowStepMode::routed_expert_union: {
            std::vector<ExecutionDispatchContext> all;
            for (const auto& m : members) { const auto rows = m.run->step_rows(i); all.insert(all.end(), rows.begin(), rows.end()); }
            auto& lead = *members[0].impl;
            moe_experts_known_window_rows_launch(all, lead.routed_config, lead.cohort_union(static_cast<std::uint32_t>(all.size())), 0U);
            break;
        }
        case KnownWindowStepMode::gdn_deferred_rows: {
            std::vector<HipGdnWindowCohortSegment> segments;
            for (const auto& m : members) segments.push_back({m.run->step_rows(i), &m.impl->gdn_window_config});
            gdn_window_cohort_rows_launch(segments);
            for (const auto& m : members) m.impl->remember_gdn_layer(m.run->step_rows(i)[0]);
            break;
        }
        case KnownWindowStepMode::qsa_rows: {
            std::vector<HipQsaCohortSegment> segments;
            for (const auto& m : members) segments.push_back({m.run->step_rows(i), &m.impl->qsa_for(m.run->step_rows(i))});
            indexed_attention_known_window_cohort_launch(segments);
            break;
        }
        case KnownWindowStepMode::inject_rows:
        case KnownWindowStepMode::moe_combine_rows: {
            std::vector<std::span<const ExecutionDispatchContext>> segments;
            for (const auto& m : members) segments.push_back(m.run->step_rows(i));
            if (members[0].impl->window.mode[i] == KnownWindowStepMode::inject_rows)
                inject_known_window_cohort_rows_launch(segments, members[0].impl->elementwise_config);
            else moe_combine_known_window_cohort_rows_launch(segments, members[0].impl->elementwise_config);
            break;
        }
        case KnownWindowStepMode::lm_head_rows: {
            std::vector<ExecutionDispatchContext> all;
            for (const auto& m : members) { const auto rows = m.run->step_rows(i); all.insert(all.end(), rows.begin(), rows.end()); }
            lm_head_known_window_rows_launch(all, members[0].impl->head_config);
            // Each session's argmax rows land in its own owner's replies, read by its own capture.
            for (const auto& m : members) {
                const auto own = m.run->step_rows(i);
                std::array<ExecutionBufferAddress, hip_greedy_rows_max> logits{};
                for (std::size_t r = 0; r < own.size(); ++r) logits[r] = resolve_dispatch_write(own[r], 0);
                enqueue_hip_greedy_rows(std::span<const ExecutionBufferAddress>(logits.data(), own.size()),
                    members[0].impl->head_config.vocab_size, m.impl->greedy_partials, m.impl->greedy_replies, hip_execution_stream());
            }
            break;
        }
        default: throw std::logic_error("known-window cohort step has no wide implementation");
        }
    }
    std::vector<std::pair<std::vector<const void*>, std::unique_ptr<ArenaLookupIndex>>> cohort_indexes;
    const ArenaLookupIndex& cohort_index(std::span<CohortMember> members) {
        std::vector<const void*> key;
        for (const auto& m : members) key.push_back(m.impl);
        for (const auto& entry : cohort_indexes) if (entry.first == key) return *entry.second;
        std::vector<const ExecutionArenaPlan*> plans;
        for (const auto& m : members) {
            plans.push_back(&m.impl->window.arena.physical);
            for (const auto& row : m.impl->window.row_plan) plans.push_back(&row.arena);
            for (const auto& view : m.impl->window.arena.row_view) plans.push_back(&view);
        }
        cohort_indexes.emplace_back(std::move(key), std::make_unique<ArenaLookupIndex>(plans));
        return *cohort_indexes.back().second;
    }
    void launch_cohort_shared(std::size_t i, std::span<CohortMember> members) {
        std::vector<ExecutionDispatchContext> all;
        all.reserve(window.logical_rows);
        for (const auto& m : members) { const auto rows = m.run->step_rows(i); all.insert(all.end(), rows.begin(), rows.end()); }
        const std::span<const ExecutionDispatchContext> rows(all);
        const KnownWindowCohortScope cohort;
        switch (window.mode[i]) {
        case KnownWindowStepMode::shared_rows: {
            // As in one window: on the side stream after everything queued so far, joined before the
            // layer's MoE combine (join_cohort_shared), with every segment's workspace rows in the
            // owner's workspace.
            std::vector<std::span<const ExecutionDispatchContext>> segments;
            for (const auto& m : members) segments.push_back(m.run->step_rows(i));
            const auto& shared = shared_for(segments[0]);
            const auto layer = *serial.execution.steps[i].layer;
            if (!shared_detach_enabled()) {
                shared_expert_known_window_cohort_launch(segments, shared, scope.stream, 0U, 0U);
                break;
            }
            auto& d = detach_state();
            if (d.pending[layer]) throw std::logic_error("known-window cohort shared expert detached twice before its combine");
            const auto e = layer + d.parity * d.pending.size();
            check(hipEventRecord(d.forked[e], scope.stream), "record known-window cohort shared fork");
            check(hipStreamWaitEvent(d.side, d.forked[e], 0), "order known-window cohort shared stream");
            shared_expert_known_window_cohort_launch(segments, shared, d.side, reinterpret_cast<std::uintptr_t>(d.workspace),
                d.workspace_bytes);
            check(hipEventRecord(d.joined[e], d.side), "record known-window cohort shared join");
            d.pending[layer] = 1U;
            break;
        }
        case KnownWindowStepMode::routed_expert_union:
            moe_experts_known_window_rows_launch(rows, routed_config, routed_workspace); break;
        case KnownWindowStepMode::hyper_mix_rows:
            hyper_mix_known_window_rows_launch(rows, hyper_for(rows)); break;
        case KnownWindowStepMode::moe_route_rows:
            moe_route_known_window_rows_launch(rows, route_config); break;
        case KnownWindowStepMode::lm_head_rows: {
            lm_head_known_window_rows_launch(rows, head_config);
            // Each session's argmax rows land in its own owner's replies, read by its own capture.
            for (const auto& m : members) {
                const auto own = m.run->step_rows(i);
                std::array<ExecutionBufferAddress, hip_greedy_rows_max> logits{};
                for (std::size_t r = 0; r < own.size(); ++r) logits[r] = resolve_dispatch_write(own[r], 0);
                enqueue_hip_greedy_rows(std::span<const ExecutionBufferAddress>(logits.data(), own.size()),
                    head_config.vocab_size, m.impl->greedy_partials, m.impl->greedy_replies, hip_execution_stream());
            }
            break;
        }
        case KnownWindowStepMode::gdn_deferred_rows: {
            std::vector<HipGdnWindowCohortSegment> segments;
            for (const auto& m : members) segments.push_back({m.run->step_rows(i), &m.impl->gdn_window_config});
            gdn_window_cohort_rows_launch(segments);
            for (const auto& m : members) m.impl->remember_gdn_layer(m.run->step_rows(i)[0]);
            break;
        }
        case KnownWindowStepMode::qsa_rows: {
            std::vector<HipQsaCohortSegment> segments;
            for (const auto& m : members) segments.push_back({m.run->step_rows(i), &m.impl->qsa_for(m.run->step_rows(i))});
            indexed_attention_known_window_cohort_launch(segments);
            break;
        }
        default: throw std::logic_error("known-window cohort step has no shared implementation");
        }
    }
    // Before a step that is each member's own: the layer's MoE combine waits for the cohort shared expert.
    void join_cohort_shared(std::size_t i) {
        const auto& step = serial.execution.steps[i];
        if (step.op != SerialOp::moe_combine || !step.layer || !shared_detach || !shared_detach->pending[*step.layer]) return;
        auto& d = *shared_detach;
        check(hipStreamWaitEvent(scope.stream, d.joined[*step.layer + d.parity * d.pending.size()], 0), "join known-window cohort shared stream");
        d.pending[*step.layer] = 0U;
    }
    static std::vector<HipMtpTargetWindowExecution> execute_cohort(Impl& primary, std::span<CohortMember> members) {
        if (members.size() < 2U) throw std::invalid_argument("known-window cohort requires two or more sessions");
        Impl* const primaries[] = {&primary};
        const std::size_t sizes[] = {members.size()};
        return execute_cohort_chunks(primaries, sizes, members);
    }
    // Chunks of consecutive members, each either several members on its own primary (an idle owner of the
    // chunk's total rows; its shared steps run once for the chunk's rows, as in a one-chunk cohort) or a
    // lone member (null primary) whose steps are all its own launches. The chunks run step by step on one
    // stream: step i of every chunk before step i + 1 of any.
    static bool& cohort_last_wide() { thread_local bool wide = false; return wide; }
    static std::vector<HipMtpTargetWindowExecution> execute_cohort_chunks(std::span<Impl* const> primaries,
        std::span<const std::size_t> sizes, std::span<CohortMember> members) {
        if (primaries.empty() || primaries.size() != sizes.size() || members.size() < 2U)
            throw std::invalid_argument("known-window cohort requires two or more sessions in one or more chunks");
        struct Chunk { Impl* primary; std::span<CohortMember> members; };
        std::vector<Chunk> chunks;
        std::size_t offset = 0;
        for (std::size_t c = 0; c < sizes.size(); ++c) {
            if (!sizes[c] || offset + sizes[c] > members.size() || (sizes[c] == 1U) != (primaries[c] == nullptr))
                throw std::invalid_argument("known-window cohort chunk needs a primary exactly when it has two or more members");
            chunks.push_back({primaries[c], members.subspan(offset, sizes[c])});
            offset += sizes[c];
        }
        if (offset != members.size()) throw std::invalid_argument("known-window cohort chunks do not cover the members");
        const auto stream = members[0].impl ? members[0].impl->scope.stream : nullptr;
        for (std::size_t a = 0; a < members.size(); ++a) {
            const auto& m = members[a];
            const auto& model = *members[0].impl;
            if (!m.impl || !m.session || !m.capture || m.impl->failed || m.impl->prefill ||
                m.impl->executing || m.tokens.size() != m.impl->logical_rows() || !m.impl->greedy_replies ||
                m.impl->scope.stream != stream || m.impl->serial.execution_identity != model.serial.execution_identity ||
                m.impl->window.mode.size() != model.window.mode.size() ||
                m.impl->resources.model_manifest != model.resources.model_manifest ||
                m.impl->resources.model_weights != model.resources.model_weights)
                throw std::invalid_argument("known-window cohort members must be idle closed owners of one model on the primary's stream");
            for (std::size_t b = 0; b < a; ++b)
                if (members[b].impl == m.impl || members[b].session == m.session ||
                    members[b].impl->resources.session_state_memory == m.impl->resources.session_state_memory)
                    throw std::invalid_argument("known-window cohort members must be distinct sessions");
        }
        for (std::size_t c = 0; c < chunks.size(); ++c) {
            auto* primary = chunks[c].primary;
            if (!primary) continue;
            std::uint32_t total = 0;
            for (const auto& m : chunks[c].members) {
                if (m.impl == primary) throw std::invalid_argument("known-window cohort primary is one of its members");
                total += m.impl->logical_rows();
            }
            for (std::size_t o = 0; o < c; ++o) if (chunks[o].primary == primary)
                throw std::invalid_argument("known-window cohort chunks share a primary");
            for (const auto& m : members) if (m.impl == primary)
                throw std::invalid_argument("known-window cohort primary is a member of another chunk");
            const auto& model = *members[0].impl;
            if (primary->failed || primary->prefill || primary->executing || total != primary->window.logical_rows ||
                primary->scope.stream != stream || primary->serial.execution_identity != model.serial.execution_identity ||
                primary->window.mode.size() != model.window.mode.size() ||
                primary->resources.model_manifest != model.resources.model_manifest ||
                primary->resources.model_weights != model.resources.model_weights || current_hip_launch_context())
                throw std::invalid_argument("known-window cohort primary must be an idle closed owner of the cohort's rows");
        }
        struct Leases {
            std::vector<bool*> held;
            ~Leases() { for (auto* l : held) *l = false; }
        } leases;
        for (auto& chunk : chunks) if (chunk.primary) { chunk.primary->executing = true; leases.held.push_back(&chunk.primary->executing); }
        for (auto& m : members) { m.impl->executing = true; leases.held.push_back(&m.impl->executing); }
        struct CohortFlags {
            std::span<CohortMember> members;
            ~CohortFlags() { for (auto& m : members) m.impl->cohort_member = false; }
        } flags{members};
        std::vector<char> primary_launched(chunks.size(), 0), union_ran(chunks.size(), 0);
        try {
            for (auto& chunk : chunks) if (chunk.primary) chunk.primary->validate_metadata();
            // A lone member runs its routed union itself, unless the cohort's union runs wide for all rows.
            const bool wide_union = chunks.size() > 1U && cohort_wide_union(members);
            for (auto& chunk : chunks) for (auto& m : chunk.members) {
                auto& self = *m.impl;
                self.cohort_member = chunk.primary != nullptr || wide_union;
                self.validate_metadata();
                m.observed = &self.prepared_capture(m.capture);
                m.capture->begin(&self, *m.session, m.tokens);
                m.begun = true;
                auto& hooks = m.hooks;
                hooks.preflight_scope = [&self] { self.validate_scope(); };
                hooks.preflight_plan = [&self] { self.validate_metadata(); };
                auto* capture = m.capture.get();
                hooks.complete = [&self, capture] { self.complete(); capture->completed_before_publish(); };
                hooks.quiesce_or_fail_stop = [&self] {
                    self.failed = true;
                    if (self.shared_detach && self.shared_detach->side) (void)hipStreamSynchronize(self.shared_detach->side);
                    self.guard.recover_or_fail_stop();
                };
                hooks.position_reader_implementations = self.position_readers;
                hooks.record_actual_window_lineage = true;
                if (shared_detach_enabled())
                    hooks.before_step = [&self](std::size_t i, const std::vector<std::vector<ExecutionDispatchContext>>& contexts) {
                        return self.detach_shared(i, contexts);
                    };
                hooks.preflight_memo = &self.preflight_memo; hooks.recipe_memo = &self.recipe_memo;
                hooks.preflight_slots = &self.preflight_slots; hooks.arena_index = &self.arena_index;
                hooks.after_step=self.window_observer;
            }
            auto& lead = *members[0].impl;
            std::optional<ValidatedModelBindingScope> binding;
            if (!lead.resources.distributed_stage) binding.emplace(*lead.resources.model_manifest, *lead.resources.model_weights);
            // One lookup index over every member's plans (keyed on the first chunk's primary, or the lead member).
            auto& indexer = chunks[0].primary ? *chunks[0].primary : lead;
            const ArenaLookupScope lookups(indexer.cohort_index(members));
            for (auto& m : members) {
                auto& self = *m.impl;
                const HipLaunchScope active(self.scope);
                m.run = std::make_unique<KnownWindowRun>(self.window, self.serial, self.resources, self.scratch, *m.session,
                    m.tokens, std::span<const CompiledLaunchPlan* const>(m.observed->pointers),
                    std::span<const KnownWindowR2Entry>(m.observed->recorded_r2),
                    std::span<const KnownWindowRowsEntry>(m.observed->recorded_rows), m.hooks, true);
                m.launched = true;
            }
            const auto steps = lead.serial.execution.steps.size();
            // STRIX_COHORT_WIDE_LOG=1: per call (first 4, then every 256th) the steps run wide, shared per chunk
            // and per member, with the mode of every step that ran per member in some chunk.
            static const bool wide_log = std::getenv("STRIX_COHORT_WIDE_LOG") != nullptr;
            static std::atomic<std::uint64_t> wide_calls{0};
            const auto call = wide_log ? wide_calls.fetch_add(1U) : 0U;
            const bool log_call = wide_log && (std::getenv("STRIX_COHORT_WIDE_LOG")[0]=='2' || call < 4U || call % 256U == 0U);
            const auto observe=[](CohortMember& m,std::size_t step) {
                if(m.impl->window_observer) m.impl->window_observer(step,[&](std::uint32_t row) {
                    const auto rows=m.run->step_rows(step);
                    if(row>=rows.size()) throw std::invalid_argument("cohort observer row is outside its session");
                    return rows[row];
                });
            };
            std::uint32_t wide_count = 0, shared_count = 0, own_count = 0;
            std::uint32_t wide_union_rows = 0; // rows of the last wide union (set per union step, all equal)
            bool all_chunked = true;
            for (const auto& chunk : chunks) all_chunked = all_chunked && chunk.primary != nullptr;
            std::map<int, std::uint32_t> own_modes;
            for (std::size_t i = 0; i < steps; ++i) {
                if (chunks.size() > 1U && cohort_wide_step(i, members, all_chunked)) {
                    ++wide_count;
                    // Every chunk primary first joins its detached shared expert of an earlier layer, as before
                    // a step of its own.
                    for (auto& chunk : chunks) if (chunk.primary && chunk.primary->shared_detach) {
                        const HipLaunchScope active(chunk.primary->scope);
                        chunk.primary->join_cohort_shared(i);
                    }
                    {
                        const HipLaunchScope active(lead.scope);
                        launch_cohort_wide(i, members);
                    }
                    if (lead.window.mode[i] == KnownWindowStepMode::routed_expert_union) {
                        wide_union_rows = 0;
                        for (const auto& m : members) wide_union_rows += m.run->rows();
                    }
                    for (auto& m : members) {m.run->record_external_step(i);observe(m,i);}
                    continue;
                }
                for (std::size_t c = 0; c < chunks.size(); ++c) {
                    auto& chunk = chunks[c];
                    auto* primary = chunk.primary;
                    if (primary && primary->cohort_step_shared(i, chunk.members)) {
                        {
                            const HipLaunchScope active(primary->scope);
                            primary_launched[c] = 1;
                            if (primary->window.mode[i] == KnownWindowStepMode::routed_expert_union) union_ran[c] = 1;
                            primary->launch_cohort_shared(i, chunk.members);
                        }
                        for (auto& m : chunk.members) {m.run->record_external_step(i);observe(m,i);}
                        ++shared_count;
                        continue;
                    }
                    ++own_count;
                    if (log_call) ++own_modes[static_cast<int>(chunk.members[0].impl->window.mode[i])];
                    if (primary && primary->shared_detach) {
                        const HipLaunchScope active(primary->scope);
                        primary->join_cohort_shared(i);
                    }
                    for (auto& m : chunk.members) {
                        const HipLaunchScope active(m.impl->scope);
                        m.run->launch_step(i);
                    }
                }
            }
            cohort_last_wide() = wide_count != 0U;
            if (log_call) {
                std::uint32_t total = 0;
                for (const auto& m : members) total += m.run->rows();
                std::fprintf(stderr, "cohort call %llu: %zu chunks, %zu sessions, %u rows, %zu steps: wide %u, chunk-shared %u, own %u (modes",
                    static_cast<unsigned long long>(call), chunks.size(), members.size(), total, steps, wide_count, shared_count, own_count);
                for (const auto& [mode, n] : own_modes) std::fprintf(stderr, " %d:%u", mode, n);
                std::fprintf(stderr, ")\n");
            }
            for (std::size_t c = 0; c < chunks.size(); ++c) {
                auto* primary = chunks[c].primary;
                if (!primary) continue;
                primary->require_shared_joined();
                // The primary checks its D1 record and, when the union ran on it, the union summary.
                const HipLaunchScope active(primary->scope);
                primary->cohort_member = !union_ran[c];
                struct Reset { bool& flag; ~Reset() { flag = false; } } reset{primary->cohort_member};
                primary->complete();
            }
            for (auto& m : members) {
                const HipLaunchScope active(m.impl->scope);
                m.run->complete();
                m.impl->require_shared_joined();
            }
            if (wide_union_rows) {
                // The last wide union's summary (the stream is complete): its rows, assignments and status.
                q51_known_rows::Summary s{};
                const auto& ws = lead.cohort_union(wide_union_rows);
                check(hipMemcpy(&s, reinterpret_cast<const void*>(ws.memory.arena_base), sizeof(s), hipMemcpyDeviceToHost),
                    "read wide cohort routed union metadata");
                const auto assignments = wide_union_rows * lead.routed_config.top_k;
                if (s.status || s.version != 1U || s.logical_rows != wide_union_rows || s.assignments != assignments ||
                    !s.touched || s.touched > assignments || s.batches < s.touched || s.batches > assignments ||
                    s.duplicate_assignments > assignments || std::any_of(s.reserved, s.reserved + 8U, [](auto v) { return v != 0U; }))
                    throw std::runtime_error("wide cohort routed union metadata disagrees with its completed window");
            }
        } catch (...) {
            for (auto& m : members) {
                if (m.begun) m.capture->abort();
                if (m.run) m.run->fail();
                else if (m.launched || m.begun) {
                    m.impl->failed = true;
                    if (m.begun) m.impl->guard.recover_or_fail_stop();
                }
                m.impl->failed = true;
            }
            for (std::size_t c = 0; c < chunks.size(); ++c) {
                if (!chunks[c].primary) continue;
                chunks[c].primary->failed = true;
                if (primary_launched[c]) chunks[c].primary->guard.recover_or_fail_stop();
            }
            throw;
        }
        std::vector<HipMtpTargetWindowExecution> out(members.size());
        try {
            for (std::size_t a = 0; a < members.size(); ++a) {
                auto& m = members[a];
                out[a].execution = m.run->publish();
                out[a].batch = m.capture->seal(*m.session);
            }
        } catch (...) {
            for (auto& m : members) {
                m.capture->abort();
                m.session->health = SerialSessionHealth::poisoned; m.session->arithmetic_lineage.tainted = true;
                m.impl->failed = true; m.impl->guard.recover_or_fail_stop();
            }
            for (auto& chunk : chunks) if (chunk.primary) chunk.primary->failed = true;
            throw;
        }
        for (std::size_t c = 0; c < chunks.size(); ++c)
            if (union_ran[c]) {
                auto& primary = *chunks[c].primary;
                const auto& first = chunks[c].members[0];
                primary.last_union_observation = HipKnownPrefixUnionObservation{first.session->next_position - first.impl->logical_rows(),
                    primary.last_union_layer, primary.pending_union_summary};
            }
        return out;
    }
    KnownWindowResult execute(SerialSessionContext& session, std::span<const std::uint32_t> tokens,
        const std::function<void()>& before_publish = {},
        const std::shared_ptr<HipMtpTargetWindowCapture>& capture = {},HipMtpTargetWindowBatch* batch=nullptr,
        bool skip_head=false) {
        // STRIX_QPF2K_SCHED bit 2: any exception of this call while a previous block of this owner is still unchecked
        // poisons the session (that block's results were published before their checks) and drains the stream.
        struct PendingGuard {
            Impl& self; SerialSessionContext& session; int uncaught;
            ~PendingGuard() {
                if (std::uncaught_exceptions() > uncaught && self.deferred_pending >= 0) {
                    session.health = SerialSessionHealth::poisoned; session.arithmetic_lineage.tainted = true;
                    self.failed = true; self.deferred_pending = -1;
                    try { self.guard.recover_or_fail_stop(); } catch (...) { std::terminate(); }
                }
            }
        } pending_guard{*this, session, std::uncaught_exceptions()};
        if (failed) throw std::logic_error("known-window native owner cannot be reused after an exception");
        if (skip_head && (!prefill || capture || before_publish))
            throw std::invalid_argument("known-window headless execution is a plain tiled prefill block only");
        if(executing)throw std::logic_error("known-window native owner cannot execute recursively");
        struct ExecutionLease {
            bool& active;
            explicit ExecutionLease(bool& value):active(value){active=true;}
            ~ExecutionLease(){active=false;}
        } execution_lease(executing);
        bool owned_eager_scope_admitted=false,actual_window_committed=false;
        const PreparedCapture* observed=nullptr;
        // STRIX_PREFILL_TIMING=1: host milliseconds of the owner's own phases around a tiled block (stderr).
        static const bool timing=[]{const char* v=std::getenv("STRIX_PREFILL_TIMING");return v&&v[0]=='1';}();
        auto mark=std::chrono::steady_clock::now();
        const auto lap=[&](const char* name) {
            if(!timing || !prefill) return;
            const auto now=std::chrono::steady_clock::now();
            std::fprintf(stderr,"prefill owner %-10s %8.2f ms\n",name,std::chrono::duration<double,std::milli>(now-mark).count());mark=now;
        };
        try {
            // STRIX_QPF2K_SCHED bit 4: a partial tiled block (the first rows of the T arena).
            const bool partial=partial_current;
            partial_current=false;
            if(partial ? (!prefill || capture || tokens.size()<16U || tokens.size()%8U || tokens.size()>logical_rows() ||
                    prefill->output_policy==KnownWindowOutputPolicy::all_rows) : tokens.size()!=logical_rows())
                throw std::invalid_argument("known-window native input row count differs from its compiled window");
            block_rows=static_cast<std::uint32_t>(tokens.size());
            if (current_hip_launch_context())
                throw std::invalid_argument("known-window native owner cannot nest a HIP launch scope");
            validate_metadata();
            lap("metadata");
            if(capture){
                if(prefill || !batch)throw std::invalid_argument("MTP typed target capture supports closed windows only");
                observed=&prepared_capture(capture);
                capture->begin(this,session,tokens);
            }
            owned_eager_scope_admitted=true;
            const HipLaunchScope active(scope);
            lap("scope");
            KnownWindowExecutionHooks hooks;
            hooks.preflight_scope = [&] { validate_scope(); };
            hooks.preflight_plan = [&] { validate_metadata(); };
            if(prefill_service) {
                hooks.bundle_closed=[this]{return prefill_bundle_closed();};
                hooks.yield_point=[this](std::size_t){service_boundary();};
            }
            hooks.complete = [&] { complete(); if(capture) capture->completed_before_publish(); if(before_publish) before_publish(); };
            hooks.quiesce_or_fail_stop = [&] {
                failed = true;
                if (shared_detach && shared_detach->side) (void)hipStreamSynchronize(shared_detach->side);
                guard.recover_or_fail_stop();
            };
            hooks.position_reader_implementations = position_readers;
            hooks.record_actual_window_lineage=static_cast<bool>(capture);
            fused_hc_step=SIZE_MAX;folded_steps.clear();
            const bool detach=!prefill && shared_detach_enabled(), fuse_hc=!prefill && hc_inject_fusion_enabled(),
                fold=!prefill && (inject_norm_enabled() || combine_inject_enabled() || commit_combine_enabled());
            if(detach || fuse_hc || fold)
                hooks.before_step=[&,detach,fuse_hc,fold](std::size_t i,const std::vector<std::vector<ExecutionDispatchContext>>& contexts) {
                    // Folded steps first; the shared-expert join (at the MoE combine) precedes a fold started there.
                    // A window fold that already took an inject step leaves nothing for the persistent HC fusion.
                    if(fold && !folded_steps.empty() && folded_steps.front()==i) return fold_window_steps(i,contexts);
                    if(fuse_hc && fuse_hc_inject(i,contexts)) return true;
                    if(detach && detach_shared(i,contexts)) return true;
                    return fold && fold_window_steps(i,contexts);
                };
            hooks.static_preflight_memo=&static_preflight_memo;
            hooks.skip_head=skip_head;
            hooks.partial_rows=partial;
            prefill_folded.clear();
            if(prefill&&(hyper_mix_prefill_fold_enabled(false)||hyper_mix_prefill_fold_enabled(true)))
                hooks.prefill_before_step=[this](std::size_t i,std::span<const ExecutionDispatchContext> rows,
                    const std::function<bool(std::size_t)>& full,const std::function<std::vector<ExecutionDispatchContext>(std::size_t)>& contexts_of) {
                    return prefill_fold(i,rows,full,contexts_of);
                };
            hooks.preflight_memo=&preflight_memo;hooks.recipe_memo=&recipe_memo;hooks.preflight_slots=&preflight_slots;
            hooks.arena_index=&arena_index;
            if(prefill) {
                hooks.after_step=qrtx_step_dump_observer(prefill_observer,block_rows);
                // The step dump (STRIX_QRTX_STEP_DUMP) reads every step: no step predicate then.
                const char* dump=std::getenv("STRIX_QRTX_STEP_DUMP");
                if(hooks.after_step&&!(dump&&*dump)) hooks.after_step_observes=prefill_observes;
                // STRIX_QPF2K_SCHED: prefill schedule hook. A step observer (MTP tap copies, step dumps) of a detached step
                // runs on the side stream before that step's join is recorded (prefill_observe).
                if(prefill_schedule()) {
                    prefill_observed=static_cast<bool>(hooks.after_step);prefill_join_due=0U;
                    const auto detached_before=prefill_detached_shared;
                    const auto deferred_before=deferred_blocks,checked_before=deferred_checked;
                    hooks.prefill_step=[this](std::size_t i,std::span<const ExecutionDispatchContext> rows){return prefill_schedule_step(i,rows);};
                    if(prefill_observed)
                        hooks.after_step=[this,inner=hooks.after_step](std::size_t i,const std::function<ExecutionDispatchContext(std::uint32_t)>& row){
                            prefill_observe(i,row,inner);};
                    hooks.complete=[&,inner=hooks.complete,detached_before,deferred_before,checked_before]{
                        if(prefill_join_due) throw std::logic_error("known prefill detached step left without its join");
                        join_all_shared();inner();
                        // Execution witness (STRIX_QPF2K_SCHED): detached shared-expert steps of this block.
                        // S2: whether this block was deferred, deferred blocks checked in its completion, still pending.
                        std::fprintf(stderr,"qpf2k sched block rows %u detached_shared %llu deferred %llu checked %llu pending %d\n",block_rows,
                            static_cast<unsigned long long>(prefill_detached_shared-detached_before),
                            static_cast<unsigned long long>(deferred_blocks-deferred_before),
                            static_cast<unsigned long long>(deferred_checked-checked_before),deferred_pending>=0?1:0);
                    };
                }
            }
            // a window step observer only for windows without a tiled prefill (as 0998a0e5: never over the prefill observer)
            if(!prefill && window_observer) hooks.after_step=window_observer;
            const auto first_position=session.next_position;
            auto result=prefill ? execute_known_prefix_prefill_eager(*prefill,serial,resources,scratch,session,tokens,
                launch_plans.front(),r2_entries,rows_entries,hooks,prefill_entries) :
                execute_known_prefix_window_eager(window, serial, resources, scratch, session, tokens,
                    capture?std::span<const CompiledLaunchPlan* const>(observed->pointers):std::span<const CompiledLaunchPlan* const>(launch_pointers),
                    capture?std::span<const KnownWindowR2Entry>(observed->recorded_r2):std::span<const KnownWindowR2Entry>(r2_entries),
                    capture?std::span<const KnownWindowRowsEntry>(observed->recorded_rows):std::span<const KnownWindowRowsEntry>(rows_entries),hooks);
            actual_window_committed=true;
            lap("eager");
            require_shared_joined();
            if(capture)*batch=capture->seal(session);
            if(window.routed_expert_union_steps)
                last_union_observation=HipKnownPrefixUnionObservation{first_position+(prefill&&!full_prefill_union()?(prefill->logical_rows/8U-1U)*8U:0U),last_union_layer,pending_union_summary};
            lap("after");
            return result;
        } catch(const KnownPrefillCancelled&) {
            defer_current=false;partial_current=false;prefill_service={};prefill_observer={};prefill_observes={};
            last_union_observation.reset();throw;
        } catch (...) {
            if(capture)capture->abort();
            if(actual_window_committed){session.health=SerialSessionHealth::poisoned;session.arithmetic_lineage.tainted=true;}
            failed = true;
            // A rejected foreign capture/scope must never be drained here.
            if(owned_eager_scope_admitted) guard.recover_or_fail_stop();
            throw;
        }
    }
    // ---- Segmented prefill: the blocks of several sessions in one run over this owner's arena ----
    // STRIX_PROMPT_BATCH_SHARED (comma list of hc, router, moe, elementwise, shared, or "all"; default
    // "hc,router,moe,elementwise"; "none" runs every step per segment): the families whose prefill step runs once
    // for the rows of every segment. Each of them computes a row from that row's input and the weights only.
    static bool segment_family_shared(std::string_view family) {
        static const std::string list = [] {
            const char* v = std::getenv("STRIX_PROMPT_BATCH_SHARED");
            return std::string(v ? v : "hc,router,moe,elementwise");
        }();
        if (list == "all") return true;
        std::size_t at = 0;
        while (at <= list.size()) {
            const auto end = std::min(list.find(',', at), list.size());
            if (std::string_view(list).substr(at, end - at) == family) return true;
            at = end + 1U;
        }
        return false;
    }
    bool segment_shared_step(std::size_t i) const {
        switch (window.mode[i]) {
        case KnownWindowStepMode::hyper_mix_rows: return segment_family_shared("hc");
        case KnownWindowStepMode::moe_route_rows: return segment_family_shared("router");
        case KnownWindowStepMode::routed_expert_union: return full_prefill_union() && segment_family_shared("moe");
        case KnownWindowStepMode::inject_rows:
        case KnownWindowStepMode::moe_combine_rows: return segment_family_shared("elementwise");
        case KnownWindowStepMode::shared_rows: return segment_family_shared("shared");
        default: return false;
        }
    }
    struct PrefillSegment {
        Impl* own = nullptr;
        SerialSessionContext* session = nullptr;
        std::span<const std::uint32_t> tokens;
        bool head = false;
        HipKnownPrefixWindow::PrefillStepObserver observer;
    };
    std::vector<KnownWindowResult> execute_segments(std::span<const PrefillSegment> requests, bool stage_ple) {
        if (failed) throw std::logic_error("known-window native owner cannot be reused after an exception");
        if (executing) throw std::logic_error("known-window native owner cannot execute recursively");
        if (!prefill || requests.empty() || prefill->output_policy == KnownWindowOutputPolicy::all_rows)
            throw std::invalid_argument("segmented prefill runs on a terminal-row or state-only tiled prefill owner");
        struct ExecutionLease {
            bool& active;
            explicit ExecutionLease(bool& value):active(value){active=true;}
            ~ExecutionLease(){active=false;}
        } execution_lease(executing);
        bool owned_eager_scope_admitted = false, committed = false;
        std::vector<KnownPrefillSegment> segments(requests.size());
        try {
            if (current_hip_launch_context()) throw std::invalid_argument("known-window native owner cannot nest a HIP launch scope");
            std::uint32_t total = 0;
            for (std::size_t s = 0; s < requests.size(); ++s) {
                const auto& r = requests[s];
                if (!r.own || !r.session || !r.own->prefill || r.own->failed || (r.own->executing && r.own != this) ||
                    r.tokens.size() != r.own->logical_rows() || r.own->serial.execution_identity != serial.execution_identity ||
                    !same_model(r.own->resources, resources) || r.own->config.fast_prefill_components != config.fast_prefill_components ||
                    r.own->config.dense_a8 != config.dense_a8 || (r.head && r.own->prefill->output_policy == KnownWindowOutputPolicy::state_only))
                    throw std::invalid_argument("segmented prefill member needs its own idle tiled owner of its rows, of this owner's execution and profile");
                if (r.own != this) r.own->validate_metadata();
                total += static_cast<std::uint32_t>(r.tokens.size());
                segments[s].session = r.session;
                segments[s].resources = &r.own->resources;
                segments[s].tokens = r.tokens;
                segments[s].head = r.head;
                segments[s].lineage_execution_identity = r.own->prefill->prefill_execution_identity;
                segments[s].lineage_kernel_plan_identity = r.own->prefill->prefill_kernel_plan_identity;
            }
            if (total > prefill->logical_rows) throw std::invalid_argument("segmented prefill rows exceed this owner's T");
            validate_metadata();
            // The full union's CSR for these rows is allocated before anything is enqueued.
            if (full_prefill_union()) (void)prefill_union(total);
            owned_eager_scope_admitted = true;
            const HipLaunchScope active(scope);
            KnownWindowExecutionHooks hooks;
            hooks.preflight_scope = [&] { validate_scope(); };
            hooks.preflight_plan = [&] { validate_metadata(); };
            if(prefill_service) {
                hooks.bundle_closed=[this]{return prefill_bundle_closed();};
                hooks.yield_point=[this](std::size_t){service_boundary();};
            }
            hooks.complete = [&] { complete(); };
            hooks.quiesce_or_fail_stop = [&] {
                failed = true;
                if (shared_detach && shared_detach->side) (void)hipStreamSynchronize(shared_detach->side);
                guard.recover_or_fail_stop();
            };
            hooks.position_reader_implementations = position_readers;
            hooks.preflight_memo = &preflight_memo; hooks.recipe_memo = &recipe_memo; hooks.preflight_slots = &preflight_slots;
            hooks.arena_index = &arena_index;
            KnownPrefillSegmentHooks segment_hooks;
            const char* stage_env=std::getenv("STRIX_PLE_STAGE_SERVE");
            segment_hooks.stage_ple = stage_ple && !(stage_env && stage_env[0]=='0');
            segment_hooks.shared = [this](std::size_t i) { return segment_shared_step(i); };
            if (std::any_of(requests.begin(), requests.end(), [](const auto& r) { return static_cast<bool>(r.observer); }))
                segment_hooks.after_step = [&](std::size_t step, std::size_t s, const std::function<ExecutionDispatchContext(std::uint32_t)>& row) {
                    if (requests[s].observer) requests[s].observer(step, row);
                };
            auto results = execute_known_prefix_prefill_segments_eager(*prefill, serial, scratch, segments, launch_plans.front(),
                r2_entries, rows_entries, hooks, prefill_entries, segment_hooks);
            committed = true;
            require_shared_joined();
            return results;
        } catch(const KnownPrefillCancelled&) {
            defer_current=false;partial_current=false;prefill_service={};
            last_union_observation.reset();throw;
        } catch (...) {
            if (committed)
                for (auto& g : segments) if (g.session) { g.session->health = SerialSessionHealth::poisoned; g.session->arithmetic_lineage.tainted = true; }
            failed = true;
            if (owned_eager_scope_admitted) guard.recover_or_fail_stop();
            throw;
        }
    }
};

HipKnownPrefixWindow::HipKnownPrefixWindow(CompiledKnownPrefixWindow window,
    CompiledSerialExecution serial_snapshot, ModelContract model, HipLauncherCatalogConfig config,
    const ExecutionRuntimeResources& serial_resources, ExecutionMemoryBinding window_scratch,
    HipLaunchContext scope, std::shared_ptr<const void> actual_owner,
    q51_known_rows::Workspace routed_union_workspace)
    : impl_(std::make_unique<Impl>(std::move(window), std::move(serial_snapshot), std::move(model),
          std::move(config), serial_resources, window_scratch, scope, std::move(actual_owner),
          std::move(routed_union_workspace))) {}
HipKnownPrefixWindow::~HipKnownPrefixWindow() = default;
HipKnownPrefixWindow::HipKnownPrefixWindow(CompiledKnownPrefixPrefill prefill,
    CompiledSerialExecution serial,ModelContract model,HipLauncherCatalogConfig config,
    const ExecutionRuntimeResources& resources,ExecutionMemoryBinding memory,HipLaunchContext scope,
    std::shared_ptr<const void> owner,q51_known_rows::Workspace workspace) {
    auto tile=prefill.tile8;
    impl_=std::make_unique<Impl>(std::move(tile),std::move(serial),std::move(model),std::move(config),resources,memory,
        scope,std::move(owner),std::move(workspace),std::move(prefill));
}
KnownWindowResult HipKnownPrefixWindow::execute(SerialSessionContext& session, std::array<std::uint32_t, 2> tokens) {
    return impl_->execute(session, std::span<const std::uint32_t>(tokens));
}
KnownWindowResult HipKnownPrefixWindow::execute(SerialSessionContext& session, std::span<const std::uint32_t> tokens) {
    return impl_->execute(session,tokens);
}
void HipKnownPrefixWindow::prepare_prefill_geometry(const SerialSessionContext& session) {
    impl_->preflight_admission(session, {}, true);
}
void HipKnownPrefixWindow::preflight_admission(const SerialSessionContext& session, std::span<const std::uint32_t> tokens) {
    impl_->preflight_admission(session, tokens);
}
KnownWindowResult HipKnownPrefixWindow::execute_without_head(SerialSessionContext& session, std::span<const std::uint32_t> tokens) {
    return impl_->execute(session,tokens,{},{},nullptr,true);
}
KnownWindowResult HipKnownPrefixWindow::execute_without_head_deferred(SerialSessionContext& session, std::span<const std::uint32_t> tokens) {
    if(!impl_->prefill || impl_->executing || impl_->failed) throw std::invalid_argument("deferred headless block requires an idle tiled prefill owner");
    struct Reset { bool& v; ~Reset(){v=false;} } reset{impl_->defer_current};
    impl_->defer_current=true;
    return impl_->execute(session,tokens,{},{},nullptr,true);
}
std::shared_ptr<const void> HipKnownPrefixWindow::producer_lifetime() const noexcept {return impl_->actual_owner;}
hipStream_t HipKnownPrefixWindow::producer_stream() const noexcept {return impl_->scope.stream;}
bool HipKnownPrefixWindow::completion_pending() const noexcept {return impl_->deferred_pending>=0;}
void HipKnownPrefixWindow::abandon_pending_completion() {
    if(impl_->deferred_pending<0) return;
    impl_->failed=true;impl_->deferred_pending=-1;
    impl_->guard.recover_or_fail_stop();
}
KnownWindowResult HipKnownPrefixWindow::execute_partial(SerialSessionContext& session, std::span<const std::uint32_t> tokens,
    bool head, bool defer) {
    if(!impl_->prefill || impl_->executing || impl_->failed) throw std::invalid_argument("partial tiled block requires an idle tiled prefill owner");
    struct Reset { bool& a; bool& b; ~Reset(){a=false;b=false;} } reset{impl_->partial_current,impl_->defer_current};
    impl_->partial_current=true;impl_->defer_current=defer&&!head;
    return impl_->execute(session,tokens,{},{},nullptr,!head);
}
void HipKnownPrefixWindow::finish_pending_completion() {
    if(impl_->executing) throw std::logic_error("deferred completion during execution");
    try {impl_->finish_deferred_completion();}
    catch(...) {impl_->failed=true;impl_->deferred_pending=-1;try{impl_->guard.recover_or_fail_stop();}catch(...){std::terminate();}throw;}
}
KnownWindowResult HipKnownPrefixWindow::execute_validated(SerialSessionContext& session,
    std::span<const std::uint32_t> tokens,const std::function<void()>& before_publish) {
    return impl_->execute(session,tokens,before_publish);
}
std::shared_ptr<HipMtpTargetWindowCapture> HipKnownPrefixWindow::prepare_mtp_target_capture(const HipMtpTargetWindowCapturePolicy& policy){
    if(impl_->failed || impl_->prefill || impl_->executing || current_hip_launch_context())throw std::invalid_argument("MTP target capture requires a live closed eager window");
    impl_->validate_metadata();
    auto capture=HipMtpTargetWindowCapture::create(impl_.get(),impl_->model,impl_->serial,impl_->window,impl_->resources,
        impl_->scratch,impl_->scope,impl_->actual_owner,policy);
    impl_->prepare_capture(capture);
    capture->use_device_greedy(impl_->greedy_replies);
    return capture;
}
HipMtpTargetWindowExecution HipKnownPrefixWindow::execute_mtp_captured(SerialSessionContext& session,std::span<const std::uint32_t> tokens,
    std::shared_ptr<HipMtpTargetWindowCapture> capture,const std::function<void()>& before_publish,std::uint32_t active_rows){
    if(!capture)throw std::invalid_argument("MTP target capture owner is absent");
    if(active_rows && (active_rows<2U || active_rows>logical_rows() || impl_->prefill))
        throw std::invalid_argument("known-window active rows must be 2..rows of a closed window");
    // Only the routed expert union reads it; without a union every row runs its experts (padding cost).
    struct ActiveRows {
        std::uint32_t& value;
        ActiveRows(std::uint32_t& v,std::uint32_t a):value(v){value=a;}
        ~ActiveRows(){value=0;}
    } scoped(impl_->active_rows,active_rows==logical_rows()?0U:active_rows);
    HipMtpTargetWindowExecution out;out.execution=impl_->execute(session,tokens,before_publish,capture,&out.batch);return out;
}
ExecutionMemoryBinding HipKnownPrefixWindow::execution_scratch() const noexcept {return impl_->scratch;}
bool HipKnownPrefixWindow::aliases_prefill_storage(const ExecutionMemoryBinding& other) const noexcept {
    const auto overlap=[&](const ExecutionMemoryBinding& x) {return batch_storage_overlap(x,other);};
    const auto& s=*impl_;
    if(overlap(s.scratch) || overlap(s.scope.deferred_moe_status) || overlap(s.routed_workspace.memory) || overlap(s.prefill_routed_workspace.memory)) return true;
    for(const auto& entry:s.prefill_union_workspace) if(overlap(entry.second.memory)) return true;
    return false;
}
bool HipKnownPrefixWindow::aliases_prefill_storage(const HipKnownPrefixWindow& other) const noexcept {
    const auto& s=*other.impl_;
    if(aliases_prefill_storage(s.scratch) || aliases_prefill_storage(s.scope.deferred_moe_status) ||
       aliases_prefill_storage(s.routed_workspace.memory) || aliases_prefill_storage(s.prefill_routed_workspace.memory)) return true;
    for(const auto& entry:s.prefill_union_workspace) if(aliases_prefill_storage(entry.second.memory)) return true;
    return false;
}
hipStream_t HipKnownPrefixWindow::prefill_stream() const noexcept {return impl_->scope.stream;}
void HipKnownPrefixWindow::set_prefill_service_point(PrefillServicePoint point) {
    if(impl_->executing) throw std::logic_error("prefill service hook requires an idle owner");
    impl_->prefill_service=std::move(point);
}
void HipKnownPrefixWindow::reset_cancelled_prefill() {
    auto& s=*impl_;
    if(s.executing || s.failed || s.deferred_pending>=0) throw std::logic_error("cancelled prefill reset requires a drained healthy owner");
    const auto clear=[&](const q51_known_rows::Workspace& w) {
        if(w.owner) check(hipMemsetAsync(reinterpret_cast<void*>(w.memory.arena_base),0,static_cast<std::size_t>(w.memory.arena_bytes),s.scope.stream),"reset private prefill union");
    };
    clear(s.prefill_routed_workspace);for(const auto& entry:s.prefill_union_workspace) clear(entry.second);
    check(hipStreamSynchronize(s.scope.stream),"complete cancelled prefill reset");
    s.prefill_union_rows=0;s.last_union_observation.reset();s.pending_union_summary={};s.completion_status={};
}
void HipKnownPrefixWindow::set_window_step_observer(PrefillStepObserver observer) {
    impl_->window_observer=std::move(observer);
}
void HipKnownPrefixWindow::set_prefill_step_observer(PrefillStepObserver observer, std::function<bool(std::size_t)> observes) {
    // Clearing is always allowed (the callers' unwind path clears after a failed block; throwing here would mask
    // the block's own exception). Clearing drops the step predicate too.
    if(!observer) { impl_->prefill_observes = {}; impl_->prefill_observer = {}; return; }
    if(!impl_->prefill || impl_->executing || impl_->failed)
        throw std::invalid_argument("prefill step observer requires an idle tiled prefill owner");
    impl_->prefill_observes=observer?std::move(observes):std::function<bool(std::size_t)>{};
    impl_->prefill_observer=std::move(observer);
}
std::vector<HipMtpTargetWindowExecution> HipKnownPrefixWindow::execute_mtp_cohort(HipKnownPrefixWindow& primary,
    std::span<const HipKnownPrefixCohortMember> members) {
    std::vector<Impl::CohortMember> owned(members.size());
    for (std::size_t m = 0; m < members.size(); ++m) {
        if (!members[m].window || !members[m].session || !members[m].capture)
            throw std::invalid_argument("known-window cohort member is incomplete");
        owned[m].impl = members[m].window->impl_.get();
        owned[m].session = members[m].session;
        owned[m].tokens = members[m].tokens;
        owned[m].capture = members[m].capture;
    }
    return Impl::execute_cohort(*primary.impl_, owned);
}
std::vector<HipMtpTargetWindowExecution> HipKnownPrefixWindow::execute_mtp_cohort_chunks(
    std::span<HipKnownPrefixWindow* const> primaries, std::span<const std::size_t> chunk_members,
    std::span<const HipKnownPrefixCohortMember> members) {
    std::vector<Impl::CohortMember> owned(members.size());
    for (std::size_t m = 0; m < members.size(); ++m) {
        if (!members[m].window || !members[m].session || !members[m].capture)
            throw std::invalid_argument("known-window cohort member is incomplete");
        owned[m].impl = members[m].window->impl_.get();
        owned[m].session = members[m].session;
        owned[m].tokens = members[m].tokens;
        owned[m].capture = members[m].capture;
    }
    std::vector<Impl*> impls;
    for (auto* p : primaries) impls.push_back(p ? p->impl_.get() : nullptr);
    // STRIX_COHORT_WIDE_LOG=1: at exit, calls and mean host wall time (launch to completion) per total rows,
    // chunk count and whether any step ran wide.
    struct Timing {
        std::map<std::tuple<std::uint32_t, std::size_t, bool>, std::pair<std::uint64_t, double>> calls;
        ~Timing() {
            for (const auto& [key, value] : calls)
                std::fprintf(stderr, "cohort timing rows %u chunks %zu wide %d: %llu calls x %.3f ms\n", std::get<0>(key), std::get<1>(key),
                    std::get<2>(key) ? 1 : 0, static_cast<unsigned long long>(value.first), value.second / static_cast<double>(value.first));
        }
    };
    static const bool timed = std::getenv("STRIX_COHORT_WIDE_LOG") != nullptr;
    const auto start = std::chrono::steady_clock::now();
    auto out = Impl::execute_cohort_chunks(impls, chunk_members, owned);
    if (timed) {
        static Timing timing;
        std::uint32_t rows = 0;
        for (const auto& m : members) rows += m.window->logical_rows();
        auto& entry = timing.calls[{rows, chunk_members.size(), Impl::cohort_last_wide()}];
        ++entry.first;
        entry.second += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    return out;
}
std::vector<KnownWindowResult> HipKnownPrefixWindow::execute_prefill_segments(std::span<const PrefillSegmentRequest> members,
    bool stage_ple) {
    std::vector<Impl::PrefillSegment> owned(members.size());
    for (std::size_t s = 0; s < members.size(); ++s) {
        if (!members[s].own) throw std::invalid_argument("segmented prefill member has no own tiled owner");
        owned[s].own = members[s].own->impl_.get();
        owned[s].session = members[s].session;
        owned[s].tokens = members[s].tokens;
        owned[s].head = members[s].head;
        owned[s].observer = members[s].observer;
    }
    return impl_->execute_segments(owned,stage_ple);
}
void HipKnownPrefixWindow::attach_greedy_rows(ExecutionBufferAddress partials,HipGreedyRowsReply* replies) {
    impl_->attach_greedy_rows(partials,replies);
}
std::uint32_t HipKnownPrefixWindow::logical_rows() const noexcept { return impl_->logical_rows(); }
bool HipKnownPrefixWindow::supports_tiled_prefill() const noexcept {return impl_->prefill.has_value();}
RotaryIdentity HipKnownPrefixWindow::rotary_identity() const { return impl_->model.rotary_identity(); }
bool HipKnownPrefixWindow::gsq_model() const noexcept {return impl_->scope.gsq_model;}
KnownWindowOutputPolicy HipKnownPrefixWindow::output_policy() const noexcept { return impl_->prefill?impl_->prefill->output_policy:impl_->window.output_policy; }
std::optional<HipKnownPrefixUnionObservation> HipKnownPrefixWindow::latest_routed_union_observation() const noexcept {
    return impl_->failed?std::nullopt:impl_->last_union_observation;
}
std::string HipKnownPrefixWindow::arithmetic_witness() const {
    const auto& i=*impl_;
    return known_window_arithmetic_witness(i.window,i.serial_rows_required,!!i.prefill,i.scope.gsq_model,
        {i.inject_norm_enabled(),i.combine_inject_enabled(),i.commit_combine_enabled()});
}
bool HipKnownPrefixWindow::policy_selection_admitted() const noexcept {
    const auto& i=*impl_;
    // Preserve native lazy-reader ordering as well as its selected arithmetic.
    if(i.prefill || i.scope.gsq_model || i.inject_norm_enabled() || i.combine_inject_enabled() || i.commit_combine_enabled()) return false;
    return known_window_policy_selection_admitted(i.window,false,false,{});
}
bool HipKnownPrefixWindow::defers_state() const noexcept {
    return impl_->window.gdn_deferred_rows_steps || impl_->window.ple_rows_steps;
}
void HipKnownPrefixWindow::commit_prefix(std::uint32_t n_keep) { impl_->commit_prefix(n_keep); }
bool HipKnownPrefixWindow::deferred_pending() const noexcept {
    return impl_->gdn_deferred && impl_->gdn_deferred->pending;
}
void HipKnownPrefixWindow::flush_deferred() { impl_->flush_deferred(); }
void HipKnownPrefixWindow::discard_deferred() { impl_->discard_deferred(); }
void HipKnownPrefixWindow::share_deferred(const HipKnownPrefixWindow& primary) { impl_->share_deferred(*primary.impl_); }

} // namespace strix
