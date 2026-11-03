#include "strix/startup_markers.hpp"
#include "strix/mtp_cost_table_session.hpp"
#include "http_slot_capacity.hpp"
#include "strix/hip_mtp_native_validation.hpp"
#include "qwen_prefill_plan.hpp"
#include "ple_resident_fallback.hpp"
#include "qwen_http_memory.hpp"
#include "strix/preflight.hpp"
#include "deferred_moe_receipt.hpp"
#include "strix/lane_chain_hyper_output.hpp"
#include "qsa_input_receipt.hpp"
#include "shared_prefetch_receipt.hpp"
#include "strix/gdn_output_admission.hpp"
#include "strix/compiled_execution.hpp"
#include "strix/teacher_logit_spool.hpp"
#include "strix/hip_q6p_pack.hpp"
#include "strix/hip_dense_trellis_pack.hpp"
#include "strix/q6p_registry.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/hip_launcher_catalog.hpp"
#include "strix/hip_gdn_islands.hpp"
#include "strix/hip_known_prefix_window.hpp"
#include "strix/hip_known_prompt_backend.hpp"
#include "strix/hip_prompt_lookup_runtime.hpp"
#include "strix/hip_serving_greedy_backend.hpp"
#include "strix/hip_serving_cohort.hpp"
#include "strix/hip_serving_prefix_cache.hpp"
#include "strix/greedy_serving_scheduler.hpp"
#include "strix/known_prompt_chunks.hpp"
#include "strix/session_executor_handoff.hpp"
#include "strix/known_prefix_prefill.hpp"
#include "strix/generation_output.hpp"
#include "strix/q51_known_rows.hpp"
#include "strix/offline_tuning_preparation.hpp"
#include "strix/offline_compile_provenance.hpp"
#include "offline_tuning_build_context.hpp"
#include "strix/hip_model_residency.hpp"
#include "strix/linux_uma_observation.hpp"
#include "strix/hip_event_queue_trace.hpp"
#include "strix/hip_runtime_allocator.hpp"
#include "strix/hip_session_snapshot_transfer.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/memory_plan.hpp"
#include "strix/mapped_model_file.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/model_manifest.hpp"
#include "strix/wave_router_contract.hpp"
#include "strix/model_tensor_index.hpp"
#include "strix/ple_lookup.hpp"
#include "strix/ple_serve_policy.hpp"
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif
#include "strix/runtime_allocator.hpp"
#include "strix/raw_baseline_recording.hpp"
#include "strix/final_generation_state_export.hpp"
#include "strix/qsa_input_prefetch.hpp"
#include "strix/shared_expert_prefetch.hpp"
#include "strix/decode_execution_options.hpp"
#include "strix/decode_integer_head.hpp"
#include "strix/hip_integer_projection_runtime.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/hip_generation_output.hpp"
#include "strix/generation_output_dispatch.hpp"
#include "strix/prepared_serial_session.hpp"
#include "strix/moe_fork_join_arena.hpp"
#include "strix/hip_moe_overlap_dispatcher.hpp"
#include "strix/moe_token_cost_backend.hpp"
#include "strix/moe_q4k_a8_contract.hpp"
#include "hip_decode_experiment.hpp"
#include "hip_decode_options.hpp"
#include "bitcalc_measurement.hpp"
#include "hip_mtp_decode_bridge.hpp"
#include "hip_http_serve.hpp"
#include "hip_prefix_state_cache.hpp"
#include "strix/mtp_step_logits_digest.hpp"
#include "qwen_cache_family_contract.hpp"
#include "hip_mtp_cohort_bridge.hpp"
#include "hip_mtp_cohort_tiled.hpp"
#if defined(STRIX_MTP_MODEL_QUALIFICATION_RUNNER)
#include "../tests/hip_mtp_model_qualification.hpp"
#endif
#include "strix/q8_projection_organization.hpp"
#include "strix/q8_projection_manifest.hpp"
#include "strix/hc_projection_prefetch.hpp"
#include "hip_deferred_moe_status.hpp"
#include "hip_decode_receipt_json.hpp"
#include "execution_inventory_receipt.hpp"
#include "runtime_owner_inventory_receipt.hpp"
#include "strix/serial_decode_reference.hpp"
#include "strix/serial_decode_replay.hpp"
#include "strix/serial_prefix_cache.hpp"
#include "strix/serial_request.hpp"
#include "strix/serial_session.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/ple_reclaim.hpp"
#include "known_prefill_reservation.hpp"
#include "ple_reclaim_mapping.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <memory>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
// File-backed tensors (the PLE n-gram table) are read row by row on the host during decode.
// Another process can evict them, turning those reads into random disk I/O while the GPU waits
// (and every fault pulls a whole readahead window for a ~100-byte row). Mark them random-access
// and populate them once, after the model registration. STRIX_PLE_WARM=0 skips it.
std::uint64_t warm_file_backed_tensors(const strix::MappedModelFile& mapped, const strix::ModelManifest& manifest,
    bool before_listening = false) {
#if defined(__linux__)
    const char* v = std::getenv("STRIX_PLE_WARM");
    if (v && v[0] == '0') return 0;
    const auto page = static_cast<std::uintptr_t>(sysconf(_SC_PAGESIZE));
    const auto t0 = std::chrono::steady_clock::now();
    std::uint64_t bytes = 0, reserved = 0;
    // Cold page cache (another model's turn evicted the table): one random-access populate reads 4 KiB per fault,
    // 0.2 GB/s for the 28.8 GB table. First read the table from several threads with normal readahead; the
    // random-access pass below then normally finds the pages resident. Same bytes, same state.
    // STRIX_PLE_WARM_PARALLEL=0 keeps the single random-access pass.
    const char* parallel_env = std::getenv("STRIX_PLE_WARM_PARALLEL");
    if (!before_listening && !strix::hip_model_copy_all_mode() && !(parallel_env && parallel_env[0] == '0')) {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
        for (const auto& tensor : manifest.tensors)
            if (tensor.residency == strix::ResidencyPolicy::file_backed_candidate && tensor.byte_size)
                ranges.emplace_back(tensor.absolute_offset, tensor.byte_size);
        const auto populated = mapped.prefault_parallel(ranges);
        std::fprintf(stderr, "file-backed tensors: parallel prefault of %zu 64 MiB chunks in %.1f s\n", populated,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    for (const auto& tensor : manifest.tensors) {
        if (tensor.residency != strix::ResidencyPolicy::file_backed_candidate || !tensor.byte_size) continue;
        if (tensor.absolute_offset > mapped.bytes() || tensor.byte_size > mapped.bytes() - tensor.absolute_offset) continue;
        const auto begin = (mapped.base_address() + tensor.absolute_offset) & ~(page - 1U);
        const auto end = mapped.base_address() + tensor.absolute_offset + tensor.byte_size;
        auto* address = reinterpret_cast<void*>(begin);
        const auto length = static_cast<std::size_t>(end - begin);
        // STRIX_MODEL_COPY_ALL: the arena allocation can push the table out of the page cache; populate it with
        // readahead (sequential) before marking it random-access, instead of one 4 KiB read per fault.
        if (!before_listening && strix::hip_model_copy_all_mode()) {
            (void)madvise(address, length, MADV_SEQUENTIAL);
#if defined(MADV_POPULATE_READ)
            (void)madvise(address, length, MADV_POPULATE_READ);
#endif
        }
        if (!before_listening) (void)madvise(address, length, MADV_RANDOM);
#if defined(MADV_POPULATE_READ)
        if (madvise(address, length, MADV_POPULATE_READ) != 0)
#endif
        if (!before_listening) {  // never force pages in after the serving allocations
            (void)madvise(address, length, MADV_WILLNEED);
            volatile std::uint8_t sink = 0;
            for (auto at = begin; at < end; at += page) sink = sink + *reinterpret_cast<const volatile std::uint8_t*>(at);
            (void)sink;
        }
        bytes += tensor.byte_size;
        reserved += ((end + page - 1U) & ~(page - 1U)) - begin;
    }
    if (bytes)
        std::fprintf(stderr, before_listening ? "file-backed tensors: re-warmed %.1f GB in %.1f s before listening\n" :
            "warmed %.1f GB of file-backed tensors in %.1f s\n", static_cast<double>(bytes) / 1e9,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return reserved;
#else
    (void)mapped; (void)manifest; (void)before_listening;
    return 0;
#endif
}
// Bytes of the file-backed tensors not resident in the page cache now (mincore); nullopt when unknown.
std::optional<std::uint64_t> file_backed_missing_bytes(const strix::MappedModelFile& mapped, const strix::ModelManifest& manifest) {
#if defined(__linux__)
    const auto page = static_cast<std::uintptr_t>(sysconf(_SC_PAGESIZE));
    std::uint64_t missing = 0;
    std::vector<unsigned char> resident;
    for (const auto& tensor : manifest.tensors) {
        if (tensor.residency != strix::ResidencyPolicy::file_backed_candidate || !tensor.byte_size) continue;
        if (tensor.absolute_offset > mapped.bytes() || tensor.byte_size > mapped.bytes() - tensor.absolute_offset) continue;
        const auto begin = (mapped.base_address() + tensor.absolute_offset) & ~(page - 1U);
        const auto end = mapped.base_address() + tensor.absolute_offset + tensor.byte_size;
        resident.assign(static_cast<std::size_t>((end - begin + page - 1U) / page), 0);
        if (mincore(reinterpret_cast<void*>(begin), static_cast<std::size_t>(end - begin), resident.data()) != 0) return std::nullopt;
        for (const auto r : resident) if (!(r & 1U)) missing += page;
    }
    return missing;
#else
    (void)mapped; (void)manifest;
    return std::nullopt;
#endif
}
// Capped re-warm: populate the not-resident 64 MiB chunks of the file-backed tensors, in file order, until their
// missing bytes reach `budget`, with normal readahead (parallel), then restore random-access advice. Returns the
// missing bytes populated. Page contents are the file's; only residency changes.
std::uint64_t rewarm_file_backed_capped(const strix::MappedModelFile& mapped, const strix::ModelManifest& manifest,
    std::uint64_t budget) {
#if defined(__linux__)
    const auto page = static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
    constexpr std::uint64_t chunk = std::uint64_t{64} << 20;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> selected, tables;
    std::vector<unsigned char> resident;
    std::uint64_t taken = 0;
    for (const auto& tensor : manifest.tensors) {
        if (tensor.residency != strix::ResidencyPolicy::file_backed_candidate || !tensor.byte_size) continue;
        if (tensor.absolute_offset > mapped.bytes() || tensor.byte_size > mapped.bytes() - tensor.absolute_offset) continue;
        const auto first = tensor.absolute_offset / page * page, last = tensor.absolute_offset + tensor.byte_size;
        tables.emplace_back(first, last - first);
        for (auto at = first; at < last && taken < budget; at += chunk) {
            const auto length = std::min(chunk, last - at);
            resident.assign(static_cast<std::size_t>((length + page - 1U) / page), 0);
            if (mincore(reinterpret_cast<void*>(mapped.base_address() + at), static_cast<std::size_t>(length), resident.data()) != 0) return 0;
            std::uint64_t missing = 0;
            for (const auto r : resident) if (!(r & 1U)) missing += page;
            if (!missing) continue;
            if (taken + missing > budget) { taken = budget; break; }
            selected.emplace_back(at, length);
            taken += missing;
        }
    }
    std::uint64_t populated = 0;
    if (!selected.empty()) {
        (void)mapped.prefault_parallel(selected);
        for (const auto& [at, length] : selected) {
            resident.assign(static_cast<std::size_t>((length + page - 1U) / page), 0);
            if (mincore(reinterpret_cast<void*>(mapped.base_address() + at), static_cast<std::size_t>(length), resident.data()) == 0)
                for (const auto r : resident) if (r & 1U) populated += page;
        }
    }
    for (const auto& [at, length] : tables) (void)madvise(reinterpret_cast<void*>(mapped.base_address() + at), static_cast<std::size_t>(length), MADV_RANDOM);
    return populated;
#else
    (void)mapped; (void)manifest; (void)budget;
    return 0;
#endif
}
using strix::decode_cli::parse_options;

// One optional stderr JSON record; stdout keeps its existing decoder receipt.
// Runs during failure unwinding too. Observability must not mask an execution
// error or change publication/cleanup ordering.
struct HostTelemetryReport {
    strix::HostDispatchCounters counters;
    bool enabled = false;
    ~HostTelemetryReport() noexcept {
        if (!enabled) return;
        try {
            strix::write_host_dispatch_telemetry(std::cerr, counters);
            std::cerr << '\n';
        } catch (...) {}
    }
};

void check_hip(
    hipError_t error,
    const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) +
            ": " +
            hipGetErrorString(error));
    }
}

void check_raw_copy(hipError_t error, const char* operation, const std::function<void()>& drain) {
    if (error != hipSuccess && drain) drain();
    check_hip(error, operation);
}

strix::PersistentState all_persistent_state() {
    using strix::PersistentState;
    return
        PersistentState::attention_kv |
        PersistentState::recurrent_matrix |
        PersistentState::convolution_history |
        PersistentState::sparse_index |
        PersistentState::ngram_history |
        PersistentState::position_history;
}

class MappedModelIoSource final
    : public strix::SparseLookupIoSource {
public:
    explicit MappedModelIoSource(
        const strix::MappedModelFile& mapped)
        : mapped_(&mapped) {}
    explicit MappedModelIoSource(std::shared_ptr<const strix::MappedModelFile> mapped)
        : mapped_(mapped.get()), mapped_owner_(std::move(mapped)) {
        if (!mapped_) throw std::invalid_argument("cached PLE source requires a retained mapping");
    }

    void read(
        std::uint64_t byte_offset,
        std::span<std::byte> destination) override {
        mapped_->read_at(
            byte_offset,
            destination);
    }

private:
    const strix::MappedModelFile* mapped_ = nullptr;
    std::shared_ptr<const strix::MappedModelFile> mapped_owner_;
};


// Actual completed backend operations, never cache-derived timing estimates.
std::string ple_row_cache_json(const strix::PlannedPleCacheStats& stats) {
    return "{\"scope\":\"per-planned-backend-actual-row-cache-operations\",\"hits\":" + std::to_string(stats.hits) +
        ",\"misses\":" + std::to_string(stats.misses) + ",\"inserted\":" + std::to_string(stats.inserted) +
        ",\"evicted\":" + std::to_string(stats.evicted) + ",\"invalidations\":" + std::to_string(stats.invalidations) +
        ",\"reserved_bytes\":" + std::to_string(stats.reserved_bytes) + ",\"resident_rows\":" + std::to_string(stats.resident_rows) +
        ",\"saturated\":" + (stats.saturated ? "true" : "false") + "}";
}

struct RankedLogit {
    std::uint32_t token = 0;
    float logit = 0.0F;
};

std::vector<RankedLogit> top_logits(
    const std::vector<float>& logits,
    std::size_t count) {
    std::vector<std::uint32_t> order(
        logits.size());
    for (std::size_t index = 0;
         index < order.size();
         ++index) {
        order[index] =
            static_cast<std::uint32_t>(
                index);
    }

    count =
        std::min(
            count,
            order.size());
    std::partial_sort(
        order.begin(),
        order.begin() +
            static_cast<
                std::ptrdiff_t>(
                count),
        order.end(),
        [&](std::uint32_t left,
            std::uint32_t right) {
            if (logits[left] !=
                logits[right]) {
                return logits[left] >
                       logits[right];
            }
            return left < right;
        });

    std::vector<RankedLogit> result;
    result.reserve(count);
    for (std::size_t index = 0;
         index < count;
         ++index) {
        const auto token =
            order[index];
        result.push_back(
            RankedLogit{
                token,
                logits[token],
            });
    }
    return result;
}

class HipSerialGreedyStepBackend final
    : public strix::SerialGreedyStepBackend {
public:
    HipSerialGreedyStepBackend(
        const strix::CompiledSerialExecution& compiled,
        strix::SerialSessionContext& session,
        const strix::ExecutionRuntimeResources& resources,
        strix::EagerExecutionDispatcher& dispatcher,
        const strix::ExecutionMemoryBinding& scratch,
        const strix::ExecutionStepIo& lm_io,
        std::uint64_t vocab_size,
        std::function<void(std::uint32_t, std::uint64_t, std::span<const float>)> recording = {},
        std::function<void()> recording_binding = {}, std::function<void()> recording_drain = {},
        std::function<void(std::uint32_t)> execute_override = {}, bool track_logit_hashes = false,
        bool prepublication_completion = false, strix::HipGreedyOutput* greedy_output = nullptr,
        strix::GenerationOutputDispatcher* output_dispatcher = nullptr)
        : compiled_(&compiled),
          session_(&session),
          resources_(&resources),
          dispatcher_(&dispatcher),
          scratch_(&scratch),
          lm_io_(&lm_io),
          vocab_size_(vocab_size), recording_(std::move(recording)), recording_binding_(std::move(recording_binding)), recording_drain_(std::move(recording_drain)), execute_override_(std::move(execute_override)), track_logit_hashes_(track_logit_hashes), prepublication_completion_(prepublication_completion), greedy_output_(greedy_output), output_dispatcher_(output_dispatcher) {
        if(prepublication_completion_&&!execute_override_)
            throw std::invalid_argument("checked completion requires the owned native token executor");
        if (output_dispatcher_ && (recording_ || track_logit_hashes_))
            throw std::invalid_argument("state-only output dispatcher cannot omit requested full-logit evidence");
        if (greedy_output_ && !output_dispatcher_)
            throw std::invalid_argument("scalar greedy output requires its prefix output dispatcher");
    }

    void execute_state(std::uint32_t input_token, std::uint64_t position) override {
        if (!output_dispatcher_) {
            strix::SerialGreedyStepBackend::execute_state(input_token, position);
            return;
        }
        if (position != session_->next_position || session_->health != strix::SerialSessionHealth::ready)
            throw std::invalid_argument("state-only prefix differs from live session");
        const auto start = std::chrono::steady_clock::now();
        strix::GenerationOutputDispatcher::StateOnlyScope state_only(*output_dispatcher_);
        const auto committed_before = session_->committed_tokens;
        try {
            if (execute_override_) execute_override_(input_token);
            else static_cast<void>(strix::execute_serial_token(
                *compiled_, *session_, *resources_, input_token, *dispatcher_));
            if (!prepublication_completion_)
                check_hip(hipDeviceSynchronize(), "state-only prefix producer completion");
            prefix_state_elapsed_ms_.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
        } catch (...) {
            if (session_->committed_tokens != committed_before) {
                session_->health = strix::SerialSessionHealth::poisoned;
                session_->arithmetic_lineage.tainted = true;
            }
            if (recording_drain_) recording_drain_();
            throw;
        }
    }

    std::uint32_t execute_greedy(std::uint32_t input_token, std::uint64_t position,
                                 std::uint64_t vocab_size) override {
        if (!greedy_output_) {
            auto logits = execute(input_token, position);
            auto correlation = strix::current_execution_correlation();
            correlation.position = position; correlation.row = 0; correlation.rows = 1;
            strix::ExecutionCorrelationScope select_correlation(correlation);
            strix::ExecutionPhaseScope selection(strix::ExecutionPhase::selection);
            return strix::select_greedy_token(logits, vocab_size);
        }
        auto correlation = strix::current_execution_correlation();
        correlation.position = position; correlation.execution_identity = compiled_->execution_identity;
        correlation.row = 0; correlation.rows = 1; correlation.step_id = UINT32_MAX;
        strix::ExecutionCorrelationScope token_scope(correlation);
        if (vocab_size != vocab_size_ || position != session_->next_position ||
            session_->health != strix::SerialSessionHealth::ready)
            throw std::invalid_argument("scalar greedy request differs from live session");
        if (recording_binding_) recording_binding_();
        const auto start = std::chrono::steady_clock::now();
        const auto committed_before = session_->committed_tokens;
        try {
            if (execute_override_) execute_override_(input_token);
            else static_cast<void>(strix::execute_serial_token(
                *compiled_, *session_, *resources_, input_token, *dispatcher_));
            // The output owner uses its own stream. The producer must have
            // completed, including deferred status, before it may read logits.
            if (!prepublication_completion_)
                check_hip(hipDeviceSynchronize(), "scalar greedy producer completion");
            const auto logits = strix::resolve_execution_buffer(compiled_->execution_identity,
                compiled_->arena, *scratch_, lm_io_->writes[0]);
            if (recording_binding_) recording_binding_();
            const auto token = [&] {
                strix::ExecutionPhaseScope selection(strix::ExecutionPhase::selection);
                return greedy_output_->select(logits, static_cast<std::uint32_t>(vocab_size_),
                    position, session_->committed_tokens);
            }();
            if (recording_binding_) recording_binding_();
            elapsed_ms_.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
            return token;
        } catch (...) {
            // State may already be consumed. Do not roll counters back or
            // allow another token after an output failure. The result owner
            // drains its own work before throwing; retain the original path's
            // producer drain as well.
            if (session_->committed_tokens != committed_before) {
                session_->health = strix::SerialSessionHealth::poisoned;
                session_->arithmetic_lineage.tainted = true;
            }
            if (recording_drain_) recording_drain_();
            throw;
        }
    }

    std::vector<float> execute(
        std::uint32_t input_token,
        std::uint64_t position) override {
        auto correlation = strix::current_execution_correlation();
        correlation.position = position; correlation.execution_identity = compiled_->execution_identity;
        correlation.row = 0; correlation.rows = 1; correlation.step_id = UINT32_MAX;
        strix::ExecutionCorrelationScope token_scope(correlation);
        if (position != session_->next_position) {
            throw std::invalid_argument(
                "greedy request position does not match serial session");
        }
        if (recording_binding_) recording_binding_();

        const auto start =
            std::chrono::steady_clock::now();

        try {
        if (execute_override_) execute_override_(input_token);
        else
        static_cast<void>(
            strix::execute_serial_token(
                *compiled_,
                *session_,
                *resources_,
                input_token,
                *dispatcher_));

        if (!prepublication_completion_) check_hip(
            hipDeviceSynchronize(),
            "hipDeviceSynchronize(greedy serial token)");
        } catch (...) { if (recording_drain_) recording_drain_(); throw; }

        const auto stop =
            std::chrono::steady_clock::now();
        elapsed_ms_.push_back(
            std::chrono::duration<double, std::milli>(
                stop - start).count());

        const auto logits_address =
            strix::resolve_execution_buffer(
                compiled_->execution_identity,
                compiled_->arena,
                *scratch_,
                lm_io_->writes[0]);

        if (vocab_size_ >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::overflow_error(
                "greedy vocabulary size does not fit size_t");
        }
        std::vector<float> logits(
            static_cast<std::size_t>(
                vocab_size_));
        if (logits_address.bytes <
            logits.size() * sizeof(float)) {
            throw std::logic_error(
                "compiled logits allocation is undersized");
        }

        if (recording_binding_) recording_binding_();
        strix::ExecutionPhaseScope delivery(strix::ExecutionPhase::delivery);
        check_raw_copy(
            hipMemcpy(
                logits.data(),
                reinterpret_cast<const void*>(
                    logits_address.address),
                logits.size() * sizeof(float),
                hipMemcpyDeviceToHost),
            "hipMemcpy(greedy logits D2H)", recording_drain_);
        if (recording_binding_) recording_binding_();

        strix::validate_serial_decode_engine_logits(
            logits,
            vocab_size_);
        if (track_logit_hashes_) logit_hashes_.push_back(strix::model_content_identity_hex(
            strix::sha256_bytes(std::as_bytes(std::span<const float>(logits)))));
        if (recording_) recording_(input_token, position, logits);
        if (logits_observer_) logits_observer_(position,logits);
        return logits;
    }

    void set_logits_observer(std::function<void(std::uint64_t,std::span<const float>)> observer) {
        logits_observer_=std::move(observer);
    }
    [[nodiscard]] const std::vector<double>&
    elapsed_ms() const noexcept {
        return elapsed_ms_;
    }
    [[nodiscard]] const std::vector<std::string>& logit_hashes() const noexcept {
        return logit_hashes_;
    }
    [[nodiscard]] const std::vector<double>& prefix_state_elapsed_ms() const noexcept {
        return prefix_state_elapsed_ms_;
    }

private:
    const strix::CompiledSerialExecution* compiled_ = nullptr;
    strix::SerialSessionContext* session_ = nullptr;
    const strix::ExecutionRuntimeResources* resources_ = nullptr;
    strix::EagerExecutionDispatcher* dispatcher_ = nullptr;
    const strix::ExecutionMemoryBinding* scratch_ = nullptr;
    const strix::ExecutionStepIo* lm_io_ = nullptr;
    std::uint64_t vocab_size_ = 0;
    std::vector<double> elapsed_ms_;
    std::vector<double> prefix_state_elapsed_ms_;
    std::vector<std::string> logit_hashes_;
    std::function<void(std::uint32_t, std::uint64_t, std::span<const float>)> recording_;
    std::function<void(std::uint64_t,std::span<const float>)> logits_observer_;
    std::function<void()> recording_binding_;
    std::function<void()> recording_drain_;
    std::function<void(std::uint32_t)> execute_override_;
    bool track_logit_hashes_ = false;
    bool prepublication_completion_ = false;
    strix::HipGreedyOutput* greedy_output_ = nullptr;
    strix::GenerationOutputDispatcher* output_dispatcher_ = nullptr;
};

// One stream shared by the window owners of every request of an MTP cohort (a cohort window
// interleaves their steps); destroyed after the last owner that borrows it.
class SharedWindowStream final {
public:
    SharedWindowStream() { check_hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create cohort window stream"); }
    ~SharedWindowStream() { if (stream && hipStreamDestroy(stream) != hipSuccess) std::_Exit(3); }
    SharedWindowStream(const SharedWindowStream&) = delete;
    SharedWindowStream& operator=(const SharedWindowStream&) = delete;
    hipStream_t stream = nullptr;
};

// A separate physical window arena and completion record. Serial D1 keeps its
// existing shadow transaction; known pairs never call its begin/complete path.
class KnownPromptMemory final {
public:
    KnownPromptMemory(const strix::CompiledKnownPrefixWindow& plan, std::uint64_t arena_generation,
        std::uint64_t status_generation,
        std::shared_ptr<const void> retained, std::shared_ptr<const strix::HipSetupQuiescenceGuard> guard,
        std::uint64_t serial_identity = 0, std::uint64_t serial_status_generation = 0,
        std::uint64_t union_bytes = 0, std::uint64_t union_generation = 0,
        std::shared_ptr<const SharedWindowStream> shared_stream = {})
        : KnownPromptMemory(plan.window_execution_identity, plan.arena.physical.total_bytes, arena_generation,
            status_generation, std::move(retained), std::move(guard), serial_identity, serial_status_generation,
            union_bytes, union_generation, std::move(shared_stream)) {}
    KnownPromptMemory(const strix::CompiledKnownPrefixPrefill& plan, std::uint64_t arena_generation,
        std::uint64_t status_generation,
        std::shared_ptr<const void> retained, std::shared_ptr<const strix::HipSetupQuiescenceGuard> guard,
        std::uint64_t union_bytes = 0, std::uint64_t union_generation = 0)
        : KnownPromptMemory(plan.prefill_execution_identity, plan.arena.physical.total_bytes, arena_generation,
            status_generation, std::move(retained), std::move(guard), 0, 0, union_bytes, union_generation) {}
    KnownPromptMemory(std::uint64_t window_identity, std::uint64_t window_bytes,
        std::uint64_t arena_generation, std::uint64_t status_generation,
        std::shared_ptr<const void> retained, std::shared_ptr<const strix::HipSetupQuiescenceGuard> guard,
        std::uint64_t serial_identity, std::uint64_t serial_status_generation,
        std::uint64_t union_bytes, std::uint64_t union_generation,
        std::shared_ptr<const SharedWindowStream> shared_stream = {})
        : retained_(std::move(retained)), guard_(std::move(guard)), shared_stream_(std::move(shared_stream)) {
        if (!retained_ || !guard_) throw std::invalid_argument("known prompt requires retained original resources");
        scratch = {window_identity, 0, window_bytes, arena_generation};
        if ((union_bytes == 0) != (union_generation == 0) || union_bytes > 65536U)
            throw std::invalid_argument("known prompt union workspace exceeds its separate 64KiB setup budget");
        union_scratch = {union_bytes ? window_identity : 0, 0, union_bytes, union_generation};
        if ((serial_identity == 0) != (serial_status_generation == 0))
            throw std::invalid_argument("known memory serial status requires both identity and generation");
        if (!scratch.execution_identity || !scratch.arena_bytes || scratch.arena_bytes > SIZE_MAX || !arena_generation || !status_generation)
            throw std::invalid_argument("known prompt memory requires current nonempty bindings");
        try {
            guard_->run([&] {
                if (shared_stream_) scope.stream = shared_stream_->stream;
                else check_hip(hipStreamCreateWithFlags(&scope.stream, hipStreamNonBlocking), "create known-prompt stream");
                check_hip(hipMalloc(&arena_, static_cast<std::size_t>(scratch.arena_bytes)), "allocate known-prompt arena");
                scratch.arena_base = reinterpret_cast<std::uintptr_t>(arena_);
                check_hip(hipMalloc(&status_, 8), "allocate known-prompt completion record");
                check_hip(hipMemsetAsync(arena_, 0, static_cast<std::size_t>(scratch.arena_bytes), scope.stream), "initialize known-prompt arena");
                check_hip(hipMemsetAsync(status_, 0, 8, scope.stream), "initialize known-prompt completion record");
                if (union_bytes) {
                    check_hip(hipMalloc(&union_, static_cast<std::size_t>(union_bytes)), "allocate known-prompt expert union workspace");
                    union_scratch.arena_base = reinterpret_cast<std::uintptr_t>(union_);
                    if (union_scratch.arena_base % 256U) throw std::runtime_error("known-prompt expert union allocation is not 256-byte aligned");
                    check_hip(hipMemsetAsync(union_, 0, static_cast<std::size_t>(union_bytes), scope.stream), "initialize known-prompt expert union workspace");
                }
                if (serial_identity) {
                    check_hip(hipMalloc(&serial_status_, 8), "allocate prompt-lookup serial completion record");
                    check_hip(hipMemsetAsync(serial_status_, 0, 8, scope.stream), "initialize prompt-lookup serial completion record");
                }
            });
            // Eager window scope carries only its owned stream and D1 record.
            // Context capacity and vocabulary belong to the retained model/state.
            scope.deferred_moe_status = {window_identity,
                reinterpret_cast<std::uintptr_t>(status_), 8, status_generation};
            if (serial_identity) {
                serial_scope.stream = scope.stream;
                serial_scope.deferred_moe_status = {serial_identity,
                    reinterpret_cast<std::uintptr_t>(serial_status_), 8, serial_status_generation};
            }
        } catch (...) { release(); throw; }
    }
    // STRIX_BATCH_CANCEL_POINTS: preserve physical addresses/generations after a drained interruption.
    void reset_cancelled_prefill() {
        guard_->run([&] {
            check_hip(hipMemsetAsync(arena_,0,static_cast<std::size_t>(scratch.arena_bytes),scope.stream),"reset cancelled prompt arena");
            check_hip(hipMemsetAsync(status_,0,8,scope.stream),"reset cancelled prompt status");
            if(union_) check_hip(hipMemsetAsync(union_,0,static_cast<std::size_t>(union_scratch.arena_bytes),scope.stream),"reset cancelled prompt union");
            if(serial_status_) check_hip(hipMemsetAsync(serial_status_,0,8,scope.stream),"reset cancelled prompt serial status");
            check_hip(hipStreamSynchronize(scope.stream),"complete cancelled prompt memory reset");
        });
    }
    ~KnownPromptMemory() { release(); }
    KnownPromptMemory(const KnownPromptMemory&) = delete;
    KnownPromptMemory& operator=(const KnownPromptMemory&) = delete;
    strix::ExecutionMemoryBinding scratch;
    strix::ExecutionMemoryBinding union_scratch;
    strix::HipLaunchContext scope;
    strix::HipLaunchContext serial_scope;
private:
    void release() noexcept {
        guard_->recover_or_fail_stop();
        if (union_ && hipFree(union_) != hipSuccess) std::_Exit(3);
        union_ = nullptr;
        if (arena_ && hipFree(arena_) != hipSuccess) std::_Exit(3);
        arena_ = nullptr;
        if (serial_status_ && hipFree(serial_status_) != hipSuccess) std::_Exit(3);
        serial_status_ = nullptr;
        if (status_ && hipFree(status_) != hipSuccess) std::_Exit(3);
        status_ = nullptr;
        if (scope.stream && !shared_stream_ && hipStreamDestroy(scope.stream) != hipSuccess) std::_Exit(3);
        scope.stream = nullptr;
    }
    std::shared_ptr<const void> retained_;
    std::shared_ptr<const strix::HipSetupQuiescenceGuard> guard_;
    std::shared_ptr<const SharedWindowStream> shared_stream_;
    void* arena_ = nullptr; void* status_ = nullptr; void* serial_status_ = nullptr; void* union_ = nullptr;
};

// CPU construction precedes scheduler reservation; GPU storage is initialized
// only inside the accepted factory. Immutable model weights are never cloned.
class ServingSessionOwner final {
public:
    ServingSessionOwner(std::shared_ptr<const strix::MappedModelFile> mapped,
        std::shared_ptr<strix::HipModelResidency> weights,
        std::shared_ptr<const strix::HipSetupQuiescenceGuard> guard,
        strix::ModelContract model, strix::CompiledSerialExecution compiled,
        strix::ModelManifest manifest, std::shared_ptr<const strix::WaveRouterManifestAdmission> admitted_manifest,
        strix::ExecutionTensorAccessPlan access, strix::SerialSessionStateLayout layout, strix::PlannedPleLookupPolicy ple_policy,
        bool planned, std::uint64_t epoch, std::uint64_t request_id,
        bool telemetry, bool trace)
        : mapped_(std::move(mapped)), weights_(std::move(weights)), guard_(std::move(guard)),
          compiled_(std::move(compiled)), manifest_(std::move(manifest)), admitted_manifest_(std::move(admitted_manifest)),
          access_(std::move(access)),
          layout_(std::move(layout)), generations_(epoch), mapped_io_(*mapped_),
          ple_policy_(ple_policy),
          planned_lookup_(mapped_io_, {ple_policy.max_rows_per_io, ple_policy.max_inflight_ios, ple_policy.cpu_helper_threads}),
          hash_(strix::build_ple_hash_contract(model)),
          planned_(planned), epoch_(epoch), request_id_(request_id), telemetry_(telemetry) {
        if (!mapped_ || !weights_ || !guard_ || !epoch_) throw std::invalid_argument("serving requires retained model and fresh epoch");
        counters.enable_trace(trace);
        session.execution_identity = compiled_.execution_identity;
        session.state_layout_identity = layout_.layout_identity;
        session.state.valid = all_persistent_state();
    }
    ~ServingSessionOwner() {
        if (!touched_) return;
        guard_->recover_or_fail_stop();
        if (status_ && hipFree(status_) != hipSuccess) std::_Exit(3);
        if (scope.stream && hipStreamDestroy(scope.stream) != hipSuccess) std::_Exit(3);
        // memory_ then releases its own state/scratch before allocator_ dies.
    }
    void initialize(const std::function<void(const strix::ExecutionMemoryBinding&)>& admit) {
        if (touched_) throw std::logic_error("serving storage cannot be initialized twice");
        touched_ = true;
        guard_->run([&] {
            // initialize is called ONLY by an admitted scheduler factory. The
            // constructor must not allocate cache payload before reservation.
            if (ple_policy_.cache_capacity_rows) {
                cache_source_ = std::make_shared<MappedModelIoSource>(mapped_);
                cached_lookup_.emplace(*cache_source_, ple_policy_, cache_source_);
            }
            memory_ = strix::RuntimeExecutionMemoryOwner::allocate(compiled_, layout_, allocator_, generations_);
            if (memory_.scratch().allocation_generation != epoch_) throw std::logic_error("serving admission epoch changed");
            admit(memory_.scratch());
            strix::initialize_hip_fresh_runtime_memory(memory_.scratch(), memory_.state());
            check_hip(hipStreamCreateWithFlags(&scope.stream, hipStreamNonBlocking), "create serving stream");
            check_hip(hipMalloc(&status_, sizeof(strix::DeferredMoeFirstError)), "allocate serving status");
            check_hip(hipMemsetAsync(status_, 0, sizeof(strix::DeferredMoeFirstError), scope.stream), "initialize serving status");
        });
        scope.deferred_moe_status = {compiled_.execution_identity, reinterpret_cast<std::uintptr_t>(status_),
            sizeof(strix::DeferredMoeFirstError), generations_.next()};
        resources.scratch = &memory_.scratch();
        resources.model_manifest = admitted_manifest_ ? &admitted_manifest_->manifest() : &manifest_;
        resources.model_weights = &weights_->binding(); resources.tensor_access = &access_;
        resources.session_state_layout = &layout_; resources.session_state_memory = &memory_.state();
        resources.ple_lookup = planned_ ? (cached_lookup_ ? static_cast<strix::PleLookupBackend*>(&*cached_lookup_) :
            static_cast<strix::PleLookupBackend*>(&planned_lookup_)) : &mapped_lookup_;
        resources.ple_hash_contract = &hash_; resources.ple_prefetch = &prefetch_;
        session.runtime_compatibility_identity = strix::build_runtime_execution_identities(
            compiled_.execution_identity, compiled_.arena, resources).compatibility_identity;
    }
    strix::ExecutionCorrelation correlation(std::uint64_t position) const noexcept {
        strix::ExecutionCorrelation result;
        result.request_id = request_id_; result.session_epoch = memory_.valid() ? memory_.state().device_allocation_generation : epoch_;
        result.execution_identity = compiled_.execution_identity; result.position = position;
        result.row = 0; result.rows = 1; result.phase = strix::ExecutionPhase::token;
        return result;
    }
    [[nodiscard]] strix::PlannedPleCacheStats ple_cache_stats() const noexcept {
        return cached_lookup_ ? cached_lookup_->cache_stats() : strix::PlannedPleCacheStats{};
    }
    bool telemetry() const noexcept { return telemetry_; }
    strix::HostDispatchCounters counters;
    strix::SerialSessionContext session;
    strix::ExecutionRuntimeResources resources;
    strix::HipLaunchContext scope;
private:
    std::shared_ptr<const strix::MappedModelFile> mapped_;
    std::shared_ptr<strix::HipModelResidency> weights_;
    std::shared_ptr<const strix::HipSetupQuiescenceGuard> guard_;
    strix::CompiledSerialExecution compiled_;
    strix::ModelManifest manifest_;
    std::shared_ptr<const strix::WaveRouterManifestAdmission> admitted_manifest_;
    strix::ExecutionTensorAccessPlan access_;
    strix::SerialSessionStateLayout layout_;
    strix::HipRuntimeAllocator allocator_{strix::HipHostAllocationMode::pageable_aligned};
    strix::AllocationGenerationSource generations_;
    strix::RuntimeExecutionMemoryOwner memory_;
    MappedModelIoSource mapped_io_;
    strix::FileMappedPleLookupBackend mapped_lookup_;
    strix::PlannedPleLookupPolicy ple_policy_;
    std::shared_ptr<MappedModelIoSource> cache_source_;
    strix::PlannedPleLookupBackend planned_lookup_;
    std::optional<strix::PlannedPleLookupBackend> cached_lookup_;
    strix::PleHashContract hash_;
    strix::PleLookupPrefetchState prefetch_;
    bool planned_ = false;
    std::uint64_t epoch_ = 0, request_id_ = 0;
    bool telemetry_ = false, touched_ = false;
    void* status_ = nullptr;
};

// Presentation-only consumer. One target call per consumed input, with the
// denominator over every finite vocabulary logit. The ledger is bounded before
// inference and does not reinterpret requested IDs as a sampling support.
class RequestedLogprobBackend final : public strix::SerialGreedyStepBackend {
public:
    struct Row { std::uint32_t input; std::uint64_t position; std::vector<strix::RequestedLogprob> values; };
    RequestedLogprobBackend(strix::SerialGreedyStepBackend& target, std::span<const std::uint32_t> ids,
        std::uint64_t vocabulary, std::uint64_t capacity, strix::SerialSessionContext& session,
        std::shared_ptr<const strix::HipSetupQuiescenceGuard> guard,
        std::span<const std::uint32_t> teacher_inputs = {}, std::string teacher_logits_path = {},
        std::uint64_t effective_max_positions = 262144U)
        : target_(target), ids_(ids.begin(), ids.end()), teacher_inputs_(teacher_inputs.begin(), teacher_inputs.end()),
          vocabulary_(vocabulary), maximum_rows_(teacher_inputs.empty() ? capacity : teacher_inputs.size() - 1U),
          session_(session), guard_(std::move(guard)) {
        if (!guard_ || !capacity || capacity > effective_max_positions ||
            (teacher_inputs_.empty() && (ids_.empty() || ids_.size() > 64U || capacity > 1048576U / ids_.size())) ||
            (!teacher_inputs_.empty() && (!ids_.empty() || teacher_inputs_.size() < 2U || teacher_inputs_.size() > capacity)))
            throw std::invalid_argument("invalid bounded requested logprob ledger");
        for (const auto id : ids_) if (id >= vocabulary_) throw std::invalid_argument("requested logprob ID is outside vocabulary");
        for (const auto id : teacher_inputs_) if (id >= vocabulary_) throw std::invalid_argument("teacher input is outside vocabulary");
        rows_.reserve(static_cast<std::size_t>(maximum_rows_));
        if (!teacher_logits_path.empty()) {
            if (teacher_inputs_.empty()) throw std::invalid_argument("teacher logits file requires retained teacher inputs");
            spool_ = std::make_unique<strix::TeacherLogitSpool>(teacher_logits_path,vocabulary_,teacher_inputs_);
        }
    }
    std::vector<float> execute(std::uint32_t token, std::uint64_t position) override {
        drain_block_spool();
        if (!teacher_inputs_.empty() && (position >= teacher_inputs_.size() ||
            teacher_inputs_[static_cast<std::size_t>(position)] != token || rows_.size() != position))
            throw std::invalid_argument("teacher-forced score input or position differs from its retained sequence");
        const bool terminal_teacher = !teacher_inputs_.empty() && position + 1U == teacher_inputs_.size();
        if (!terminal_teacher && rows_.size() == maximum_rows_) throw std::length_error("requested logprob ledger capacity exhausted");
        const auto committed = session_.committed_tokens;
        try {
            // Consume the final teacher input once, with ordinary full logits;
            // it has no supplied next-token label and creates no ledger row.
            if (terminal_teacher) {
                auto logits = target_.execute(token, position);
                if (spool_) spool_->finish();
                return logits;
            }
            const auto requested = teacher_inputs_.empty() ? std::span<const std::uint32_t>{ids_} :
                std::span<const std::uint32_t>{teacher_inputs_}.subspan(static_cast<std::size_t>(position) + 1U, 1U);
            const strix::GenerationOutputRequest request{strix::GenerationOutputNeed::full_logits |
                strix::GenerationOutputNeed::requested_logprobs, token, position, vocabulary_, requested};
            auto result = strix::execute_generation_output(request, target_);
            if (spool_) spool_->append(token,position,result.logits);
            rows_.push_back({token, position, std::move(result.requested_logprobs)});
            return std::move(result.logits);
        } catch (...) {
            if (session_.committed_tokens != committed) {
                session_.health = strix::SerialSessionHealth::poisoned;
                session_.arithmetic_lineage.tainted = true;
            }
            guard_->recover_or_fail_stop();
            throw;
        }
    }
    void execute_state(std::uint32_t token, std::uint64_t position) override {
        if (!teacher_inputs_.empty()) static_cast<void>(execute(token, position));
        else target_.execute_state(token, position);
    }
    // Teacher-forced rows of one committed tiled prefill block: `logits` holds tokens.size() full
    // FP32 rows in row order, computed by the block (no target call here). Each row is scored with
    // the same validation and requested-logprob arithmetic as execute() and appended to the score
    // ledger in position order before this returns. Rows are scored in parallel (independent; each
    // row's arithmetic is sequential and unchanged). The spool append of the block (and the footer
    // when the final teacher input is in it) runs on a background thread while the next block
    // computes: `logits` must stay valid until drain_block_spool(), which the window backend calls
    // before overwriting its staging and every later consumer call makes before touching the spool.
    void record_block(std::span<const std::uint32_t> tokens, std::uint64_t first_position, std::span<const float> logits) {
        drain_block_spool();
        const auto started_at = std::chrono::steady_clock::now();
        const auto count = tokens.size();
        if (teacher_inputs_.empty() || !count || logits.size() != count * vocabulary_ || rows_.size() != first_position ||
            first_position > teacher_inputs_.size() || count > teacher_inputs_.size() - first_position)
            throw std::invalid_argument("teacher-forced block differs from its retained sequence");
        for (std::size_t r = 0; r < count; ++r)
            if (teacher_inputs_[static_cast<std::size_t>(first_position) + r] != tokens[r])
                throw std::invalid_argument("teacher-forced block input differs from its retained sequence");
        const bool terminal = first_position + count == teacher_inputs_.size();
        const auto scored = terminal ? count - 1U : count;
        if (rows_.size() + scored > maximum_rows_) throw std::length_error("requested logprob ledger capacity exhausted");
        const auto vocabulary = static_cast<std::size_t>(vocabulary_);
        const auto row_logits = [logits, vocabulary](std::size_t r) { return logits.subspan(r * vocabulary, vocabulary); };
        std::vector<std::array<strix::RequestedLogprob, 1>> values(count);
        std::vector<std::exception_ptr> errors(count);
        const auto score = [&](std::size_t r) {
            try {
                strix::validate_serial_decode_engine_logits(row_logits(r), vocabulary_);
                if (r < scored) {
                    const std::array<std::uint32_t, 1> next{teacher_inputs_[static_cast<std::size_t>(first_position) + r + 1U]};
                    strix::compute_requested_logprobs(row_logits(r), next, values[r]);
                }
            } catch (...) { errors[r] = std::current_exception(); }
        };
        const auto workers = std::min<std::size_t>(count, std::min(16U, std::max(1U, std::thread::hardware_concurrency())));
        {
            std::vector<std::thread> threads;
            threads.reserve(workers);
            for (std::size_t w = 0; w < workers; ++w)
                threads.emplace_back([&, w] { for (std::size_t r = w; r < count; r += workers) score(r); });
            for (auto& thread : threads) thread.join();
        }
        for (const auto& error : errors) if (error) std::rethrow_exception(error);
        for (std::size_t r = 0; r < scored; ++r)
            rows_.push_back({tokens[r], first_position + r, {values[r].begin(), values[r].end()}});
        tiled_rows_ += count;
        // STRIX_PREFILL_TIMING=1: host milliseconds of scoring and spooling each block on stderr.
        static const bool timing = [] { const char* v = std::getenv("STRIX_PREFILL_TIMING"); return v && v[0] == '1'; }();
        if (timing)
            std::fprintf(stderr, "teacher block %llu rows: score %.1f ms\n", static_cast<unsigned long long>(count),
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started_at).count());
        if (!spool_) return;
        std::vector<std::uint32_t> inputs(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(scored));
        pending_spool_ = std::async(std::launch::async, [this, inputs = std::move(inputs), first_position, row_logits, terminal] {
            const auto spool_start = std::chrono::steady_clock::now();
            for (std::size_t r = 0; r < inputs.size(); ++r) spool_->append(inputs[r], first_position + r, row_logits(r));
            if (terminal) spool_->finish();
            if (timing)
                std::fprintf(stderr, "teacher block spool %.1f ms (overlapped)\n",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - spool_start).count());
        });
    }
    // Completes the background spool of the last block and reports its failure. A failed spool
    // after committed rows poisons the session like a failed serial-row spool.
    void drain_block_spool() {
        if (!pending_spool_.valid()) return;
        try { pending_spool_.get(); }
        catch (...) {
            session_.health = strix::SerialSessionHealth::poisoned;
            session_.arithmetic_lineage.tainted = true;
            guard_->recover_or_fail_stop();
            throw;
        }
    }
    std::uint64_t tiled_rows() const noexcept { return tiled_rows_; }
    const std::vector<Row>& rows() const noexcept { return rows_; }
private:
    strix::SerialGreedyStepBackend& target_;
    std::vector<std::uint32_t> ids_;
    std::vector<std::uint32_t> teacher_inputs_;
    std::uint64_t vocabulary_;
    std::uint64_t maximum_rows_;
    strix::SerialSessionContext& session_;
    std::shared_ptr<const strix::HipSetupQuiescenceGuard> guard_;
    std::vector<Row> rows_;
    std::unique_ptr<strix::TeacherLogitSpool> spool_;
    std::uint64_t tiled_rows_ = 0;
    std::future<void> pending_spool_; // declared last: destroyed (joined) before the spool it writes
};

class ObservedServingBackend final : public strix::SerialGreedyStepBackend, public strix::ServingCohortObservation {
public:
    ObservedServingBackend(std::unique_ptr<strix::SerialGreedyStepBackend> backend,
        std::shared_ptr<ServingSessionOwner> owner) : backend_(std::move(backend)), owner_(std::move(owner)) {}
    strix::HipServingGreedyBackend* native_backend() noexcept {
        return dynamic_cast<strix::HipServingGreedyBackend*>(backend_.get());
    }
    void observe_serving_target(const std::function<void()>& body) override {
        strix::ExecutionTelemetryBinding bind(&owner_->counters, owner_->correlation(owner_->session.next_position));
        body();
    }
    std::vector<float> execute(std::uint32_t token, std::uint64_t position) override {
        strix::ExecutionTelemetryBinding bind(&owner_->counters, owner_->correlation(position));
        return backend_->execute(token, position);
    }
    std::uint32_t execute_greedy(std::uint32_t token, std::uint64_t position, std::uint64_t vocabulary) override {
        strix::ExecutionTelemetryBinding bind(&owner_->counters, owner_->correlation(position));
        return backend_->execute_greedy(token, position, vocabulary);
    }
    std::uint32_t prefill_chunk_rows() const noexcept override { return backend_->prefill_chunk_rows(); }
    std::optional<std::uint32_t> execute_prefill_chunk(std::span<const std::uint32_t> tokens,
        std::uint64_t position,std::uint64_t vocabulary,bool terminal) override {
        auto correlation=owner_->correlation(position);correlation.phase=strix::ExecutionPhase::prefill;
        correlation.rows=static_cast<std::uint32_t>(tokens.size());
        strix::ExecutionTelemetryBinding bind(&owner_->counters,correlation);
        return backend_->execute_prefill_chunk(tokens,position,vocabulary,terminal);
    }
    void execute_state(std::uint32_t token, std::uint64_t position) override {
        auto correlation = owner_->correlation(position); correlation.phase = strix::ExecutionPhase::prefill;
        strix::ExecutionTelemetryBinding bind(&owner_->counters, correlation);
        backend_->execute_state(token, position);
    }
    std::optional<strix::GreedyPrefixRestore> try_restore_greedy_prefix(const strix::SerialGreedyRequest& request) override {
        auto correlation = owner_->correlation(owner_->session.next_position); correlation.phase = strix::ExecutionPhase::prefill;
        strix::ExecutionTelemetryBinding bind(&owner_->counters, correlation);
        return backend_->try_restore_greedy_prefix(request);
    }
    void capture_greedy_prefix(std::span<const std::uint32_t> prefix, std::uint32_t pending, std::uint64_t vocabulary) override {
        auto correlation = owner_->correlation(owner_->session.next_position); correlation.phase = strix::ExecutionPhase::prefill;
        strix::ExecutionTelemetryBinding bind(&owner_->counters, correlation);
        backend_->capture_greedy_prefix(prefix, pending, vocabulary);
    }
private:
    std::unique_ptr<strix::SerialGreedyStepBackend> backend_;
    std::shared_ptr<ServingSessionOwner> owner_;
};

std::vector<std::uint32_t> read_serving_prefix(const std::string& path, std::uint64_t capacity, std::uint64_t vocabulary) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const auto end = file.tellg();
    if (!file || end <= 0) throw std::invalid_argument("serving prefix file is empty or unavailable");
    const auto bytes = static_cast<std::uint64_t>(end);
    if (bytes % 4 || bytes / 4 > capacity || bytes / 4 > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("serving prefix must contain bounded little-endian uint32 tokens");
    std::vector<std::uint32_t> tokens(static_cast<std::size_t>(bytes / 4));
    file.seekg(0);
    for (auto& token : tokens) {
        std::array<unsigned char, 4> raw{};
        if (!file.read(reinterpret_cast<char*>(raw.data()), 4)) throw std::runtime_error("serving prefix changed while reading");
        token = static_cast<std::uint32_t>(raw[0]) | (static_cast<std::uint32_t>(raw[1]) << 8U) |
            (static_cast<std::uint32_t>(raw[2]) << 16U) | (static_cast<std::uint32_t>(raw[3]) << 24U);
        if (token >= vocabulary) throw std::invalid_argument("serving prefix token is outside vocabulary");
    }
    if (file.peek() != std::char_traits<char>::eof()) throw std::runtime_error("serving prefix grew while reading");
    return tokens;
}

// Drain inside the transfer call before the caller's snapshot owner can unwind.
class GuardedHipSnapshotTransfer final : public strix::SessionSnapshotTransferBackend {
public:
    GuardedHipSnapshotTransfer(std::function<void()> current, std::function<void()> drain)
        : current_(std::move(current)), drain_(std::move(drain)) {}
    void capture_device(std::uintptr_t source, std::uintptr_t destination, std::uint64_t bytes) override { call([&] { backend_.capture_device(source, destination, bytes); }); }
    void capture_host(std::uintptr_t source, std::uintptr_t destination, std::uint64_t bytes) override { call([&] { backend_.capture_host(source, destination, bytes); }); }
    void restore_device(std::uintptr_t source, std::uintptr_t destination, std::uint64_t bytes) override { call([&] { backend_.restore_device(source, destination, bytes); }); }
    void restore_host(std::uintptr_t source, std::uintptr_t destination, std::uint64_t bytes) override { call([&] { backend_.restore_host(source, destination, bytes); }); }
private:
    template<class Function> void call(Function&& function) {
        try { if (current_) current_(); function(); if (current_) current_(); }
        catch (...) { if (drain_) drain_(); throw; }
    }
    strix::HipSessionSnapshotTransferBackend backend_;
    std::function<void()> current_, drain_;
};

std::string_view stop_reason_name(
    strix::SerialRequestStopReason reason) noexcept {
    switch (reason) {
    case strix::SerialRequestStopReason::eos:
        return "eos";
    case strix::SerialRequestStopReason::max_new_tokens:
        return "max_new_tokens";
    case strix::SerialRequestStopReason::context_capacity:
        return "context_capacity";
    }
    return "unknown";
}

void write_logit_hashes(std::span<const std::string> hashes) {
    std::cout << '[';
    for (std::size_t index = 0; index < hashes.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << '"' << hashes[index] << '"';
    }
    std::cout << ']';
}

// The prefix cache's segment / checkpoint unit: --prefix-cache-block-rows, else T.
template <class O>
std::uint32_t prefix_cache_unit(const O& options) {
    return static_cast<std::uint32_t>(options.prefix_cache_block_rows ? options.prefix_cache_block_rows : options.known_prefix_prefill_rows);
}

}  // namespace

// Whether the route owning a prefix-state namespace can execute STRIX_QPF2K_SCHED bit 2. The MTP bridge (CLI --mtp,
// HTTP, cohorts) never defers; the CLI cursor drops a defaulted bit 2 under a prefix-state cache, but a session/RTX
// registry without one defers, so a registry keeps the conservative answer (a miss, never a shared key).
static bool prefix_namespace_sched_defers(const strix::decode_cli::Options& options, bool rtx_state_registry) {
    if (options.mtp.enabled || options.http_serve) return false;
    return strix::qpf2k::cli_sched_defers(!rtx_state_registry);
}

#ifndef STRIX_DECODE_ENTRY
#define STRIX_DECODE_ENTRY main
#endif
// Qualification uses a canonical capacity layout; actual context remains an explicit
// range predicate. This never replaces the storage/certificate identity.
static strix::decode_cli::CacheFamilyIdentity resolved_cache_family(
    const strix::decode_cli::Options& options, const strix::ModelContract& source_model,
    const strix::ModelManifest& manifest, const strix::MappedModelFile& mapped, strix::decode_cli::CacheRunner runner) {
    // The schedule this runner executes (STRIX_QPF2K_SCHED bit 2 is inert off the CLI cursor), in the scheduler field
    // and in the numerical contract alike.
    const bool sched_defers=strix::decode_cli::cache_runner_defers(runner);
    auto model=source_model;
    model.attention_kv_storage=options.attention_kv_storage;
    strix::SerialSessionStateLayoutRequest family_layout;
    family_layout.context_tokens=16384;
    family_layout.recurrent_storage=options.gdn_recurrent_storage;
    strix::decode_cli::CacheFamilyIdentity cache_family;
    using namespace strix::decode_cli;
    cache_family.mtp_contract_qualified=cache_mtp_contract_qualified(options) && cache_mtp_vocabulary_qualified(options);
    cache_family.model=strix::model_content_identity_hex(mapped.file_sha256());
    cache_family.layout=std::to_string(manifest.layout_identity)+":"+std::to_string(strix::build_serial_session_state_layout(model, family_layout).layout_identity);
    cache_family.tokenizer=cache_family.model; // tokenizer is in the same full GGUF
    cache_family.sidecar=strix::model_content_identity_hex(strix::session_cached_file_hash(options.mtp.sidecar));
    cache_family.golden=cache_family.replay=strix::model_content_identity_hex(strix::session_cached_file_hash(options.mtp.golden));
    PrefillCapabilities cache_caps;
    const auto cache_observation=observe_prefill_memory(cache_caps);
    if(cache_observation.gfx_target) cache_family.gfx="gfx"+std::to_string(*cache_observation.gfx_target/10000)+std::to_string((*cache_observation.gfx_target/100)%100)+std::to_string(*cache_observation.gfx_target%100)+":"+std::to_string(*cache_observation.gfx_target);
    int runtime=0,driver=0;
    if(hipRuntimeGetVersion(&runtime)==hipSuccess && hipDriverGetVersion(&driver)==hipSuccess)
        cache_family.build_family=std::string(__VERSION__)+":"+std::to_string(runtime)+":"+std::to_string(driver);
    cache_family.kv_precision=std::to_string(static_cast<unsigned>(options.attention_kv_storage));
    cache_family.recurrent_precision=std::to_string(static_cast<unsigned>(options.gdn_recurrent_storage));
    const auto profiles=numerical_profile_pair(options);
    cache_family.prefill_profile=numerical_profile_name(profiles.prefill);
    cache_family.decode_profile=numerical_profile_name(profiles.decode);
    cache_family.profile_pair=std::to_string(strix::numerical_profile_pair_identity(profiles));
    cache_family.output_policy=std::to_string(static_cast<unsigned>(options.known_prefix_output_policy));
    cache_family.numerical_contract=strix::model_content_identity_hex(cache_family_execution_signature(
        cache_family_execution_cli(options),options.baseline,strix::qpf2k::has_gsq_gate(manifest),model.rotary_identity(),sched_defers));
    cache_family.tails=options.known_prefix_tail_rows;
    const auto number=[](const char* name,std::uint64_t fallback) {
        const char* value=std::getenv(name); return value?static_cast<std::uint64_t>(std::strtoull(value,nullptr,0)):fallback;
    };
    auto& c=cache_family.scheduler;
    c.schedule=strix::qpf2k::effective_sched(number("STRIX_QPF2K_SCHED",0),sched_defers); c.pipe=number("STRIX_PREFILL_PIPE",0);
    c.groups=strix::decode_cli::MtpBatchRunner::prompt_batching(); c.group_rows=strix::decode_cli::MtpBatchRunner::prompt_group_rows(); c.group_owner=(number("STRIX_PROMPT_BATCH_ROWS",0)>options.known_prefix_prefill_rows);
    c.cohort=strix::decode_cli::MtpBatchRunner::prompt_window_cohort(); c.small=number("STRIX_BATCH_PROMPT_SMALL",0)!=0;
    c.windows=number("STRIX_PROMPT_WINDOW",1)!=0;
    c.nest_rows=number("STRIX_BATCH_NEST_ROWS",2048); c.short_rows=number("STRIX_BATCH_SHORT_FIRST_ROWS",2048);
    c.lookahead=number("STRIX_BATCH_SERVICE_LOOKAHEAD",2); c.cancel=number("STRIX_BATCH_CANCEL_POINTS",1)!=0;
    c.interleave=number("STRIX_BATCH_INTERLEAVE",options.http_serve?1:0);
    const char* service=std::getenv("STRIX_BATCH_SERVICE_MS"); c.service_ms=service?std::max(0.0,std::atof(service)):250;
    const char* share=std::getenv("STRIX_BATCH_PROMPT_SHARE"); c.prompt_share=share?std::max(0.0,std::atof(share)):0;
    const char* shared=std::getenv("STRIX_PROMPT_BATCH_SHARED"); c.shared=shared?shared:"hc,router,moe,elementwise";
    c.priming=(c.pipe&2)!=0; c.worker=(c.pipe&32)!=0;
    c.tiled_start=number("STRIX_MTP_TILED_START",1)!=0;
    c.merge=(c.schedule&16)!=0; c.balance=(c.schedule&4)!=0;
    return cache_family;
}

static int run_decode_entry(
    int argc,
    char** argv, strix::decode_cli::MtpBundleSetupRetry bundle_retry = {}) try {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "Usage: strix-hip-decode-token MODEL.gguf TOKEN_ID ... [options]\n"
            "  --context N --generate N --eos-token ID --residency register|copy\n"
            "  --profile default|fast|quality (default: current qualified settings)\n"
            "  --allow-pending (benchmark runnable pending profile rows; unqualified)\n"
            "  --optimization-profile auto|manual (default auto, qualified checkpoint only)\n"
            "  --baseline / STRIX_BASELINE=1 (historical resolution, controlled defaults off)\n"
            "  --print-effective-config (deterministic host JSON; exits before HIP setup)\n"
            "  --mtp-bundle FILE (YaRN MTP pins/policy/budgets; otherwise discover MODEL.gguf.mtp-bundle.json)\n"
            "  --print-cache-family (resolved qualification identity; hashes assets, no inference)\n"
            "  --prefill-numerical-profile dense-a8|fast-prefill-f16wmma-v1 (automatic tiled default: FP16)\n"
            "  --prefill-exact-fp16 / STRIX_PREFILL_EXACT_FP16=1 (plain FP16 prefill identity)\n"
            "  --prefill-disable NAME[,NAME...] (disable bit-changing options; server-level only)\n"
            "  --no-early-setup --no-eager-code-load (exact startup policies; default on)\n"
            "  --config FILE (prefill_exact_fp16 = true; prefill_disable = [\"hc_output_wmma\"])\n"
            "  STRIX_PREFILL_F16=0 restores historical resolution of the same argv\n"
            "  --known-prefix-prefill-rows 16..16384; route/full-rows/union/router/prefill-rows accept =0\n"
            "  --host-dispatch-telemetry (host API/commit counters on stderr; profiling only)\n"
            "  --execution-telemetry-trace (bounded 256-event correlated trace; profiling only)\n"
            "  --execution-inventory (include selected host launcher and declared variant metadata)\n"
            "  --gpu-greedy-token (experimental eager generation with small GPU selection result)\n"
            "  --batched-moe-activation (experimental eager A8 slot activation batching)\n"
            "  --qsa-gqa-heads-per-block 1|2|4 (experimental eager grouped attention; default1)\n"
            "  --qsa-hierarchical-selection (experimental exact top512, eager D1, context<=131072)\n"
            "  --gdn-recurrent-layout baseline|stride32|stride33|transposed (experimental eager LDS organization)\n"
            "  --gdn-recurrent-storage key-value|value-key (experimental physical state layout, default key-value)\n"
            "  --known-prefix-window [--known-prefix-rows 2|4|8] [--known-prefix-output-policy all-rows|terminal-row|state-only] (experimental known prompt)\n"
            "  --prompt-lookup [--prompt-lookup-depth 0|1|2 | --prompt-lookup-adaptive] (experimental greedy target verification, default off)\n"
            "  --ple-cache-rows N --ple-cache-budget-bytes N (optional bounded raw-row cache; requires planned PLE)\n"
            "  --serve-prefix-u32le FILE (repeat per request; independent C1 state, shared weights)\n"
            "  --serve-prefix-cache-entries N --serve-prefix-cache-snapshot-bytes N --serve-prefix-cache-capture-peak-bytes N --serve-prefix-cache-max-tokens N (all explicit, optional bounded shared pool)\n"
            "  --serve-concurrency 1|2|4 --serve-memory-budget-bytes N --serve-request-budget-bytes N --serve-headroom-bytes N\n"
            "  --device-queue-trace (explicit profiling: requires host telemetry/eager D1; launcher queue spans, not kernel busy time)\n"
            "  --gdn-islands (experimental eager D1 GDN launch organization; paired wave8 GDN supported)\n"
            "  --prepared-eager (experimental immutable static submission plan)\n"
            "  --moe-shared-overlap [--moe-overlap-arena-budget-bytes N] (experimental eager D1)\n"
            "  --virtual-q8-gdn-output (exact original256-tree Q8 output)\n"
            "  --wave8-q8-gdn-output (eager wave8 output; requires resolved virtual Q8 GDN output)\n"
            "  --wave8-q8-qsa-output (eager all-layer Q8 output organization)\n"
            "  --prefetch-hc-projection-next-tile (canonical eager typed HC projection overlay)\n"
            "  --q51-slot-parallel-down (experimental eager grouped Q5_1 raw-dot overlay)\n"
            "  --lane-chain-hyper-output (requires --prefetch-q8-hyper-output)\n"
            "  --record-raw ABS_NEW_PRIVATE_DIRECTORY --record-max-bytes BYTES\n"
            "  --record-state-policy final|each --prefix-cache-roundtrip\n"
            "  --session-cache on|off --session-disk on|off (HTTP defaults on when admitted)\n"
            "  --session-ram-bytes N --session-max-parked N --session-dir DIR --session-disk-bytes N\n"
            "  --session-save DIR | --session-load DIR | --session-live-control [--session-then-prompt FILE] [--session-logits FILE]\n"
            "  --session-first-generate N (first turn only); --session-capture-host-bytes N (default 8 GiB); --session-validate-only (CPU preflight)\n"
            "  --prefix-batch-cert 0|1|capture|l0|l1 (default 0; 1=l0; l1 capture-only until qualified)\n"
            "      Automatic policy requests L0 only without explicit storage controls; unqualified families decline.\n"
            "  --prefix-cache-staging-bytes N (0: one T-block plus fixed state; STRIX_PREFIX_CACHE_STAGING_BYTES)\n"
            "  --prefix-cache-host-bytes N [--prefix-cache-dir DIR [--prefix-cache-disk-bytes N]] [--prefix-cache-every N]\n"
            "      [--prefix-cache-then-prompt FILE ...] (block-boundary prefix state cache, tiled known-prefix prefill only)\n"
            "  --http-serve [--http-bind ADDR] [--http-port N] [--http-model-id NAME] [--http-api-key-file FILE]\n"
            "      [--http-sampling greedy|reject] [--http-default-max-tokens N] [--http-max-queue N] [--http-max-connections N]\n"
            "      (OpenAI/Anthropic-compatible HTTP server on the tiled MTP route; the prompt is the calibration prompt)\n"
            "  --release-manifest FILE (HTTP only, no multi-input modes; verify assets rooted beside FILE)\n"
            "  --prefetch-q8-qsa-input (opt-in exact-tree Q8 input projections)\n"
            "  --prefetch-q8-shared-expert (opt-in exact-tree Q8 gate/up/down)\n"
            "  --execution-mode eager|graph (default eager)\n"
            "  --deferred-moe-status (owned eager C1 completion before token publication)\n"
            "  --graph-workspace-bytes BYTES --graph-executable-bytes BYTES\n"
            "  --graph-capture-peak-bytes BYTES --graph-budget-bytes BYTES\n"
            "  --integer-lm-head a8-iu8|a4-iu8|a4-iu4\n"
            "  --integer-internal-bytes BYTES --integer-retained-bytes BYTES\n"
            "Graph requires --residency copy, including the complete PLE table.\n"
            "Integer mode changes only the eager LM head; IQ4_XS requires IU8.\n"
            "Graph and integer modes require explicit memory ceilings.\n"
            "All paths remain experimental; RAW recording is not performance evidence.\n";
        return 0;
    }
    auto options =
        parse_options(
            argc,
            argv, true);
    bundle_retry.selected=options.mtp.enabled && !options.mtp_bundle_path.empty();
#if defined(STRIX_MTP_MODEL_QUALIFICATION_RUNNER)
    options.mtp_cli_phase_retirement=false;
#endif
    // Before any device/runtime observation. Respect even an empty user value.
    // Inspection predicts this policy without changing the process environment.
    // A value set here is already too late for this image, so restart it once with the value in its
    // environment; the restarted image finds the value set and continues without restarting again.
    if (!options.print_effective_config && !options.print_cache_family &&
        strix::decode_cli::apply_eager_code_loading(options.eager_code_load && !options.baseline &&
            !(std::getenv("STRIX_QPF2K_DEFAULTS") && std::getenv("STRIX_QPF2K_DEFAULTS")[0] == '0'),
            [](const char* name) { return std::getenv(name); },
            [](const char* name, const char* value) { return ::setenv(name, value, 0) == 0; })) {
        std::fflush(nullptr);
        std::ifstream cmdline("/proc/self/cmdline", std::ios::binary);
        const auto original = strix::decode_cli::split_process_arguments(
            std::string((std::istreambuf_iterator<char>(cmdline)), std::istreambuf_iterator<char>()));
        std::vector<char*> restart;
        for (const auto& argument : original) restart.push_back(const_cast<char*>(argument.c_str()));
        restart.push_back(nullptr);
        ::execv("/proc/self/exe", original.empty() ? argv : restart.data());
        std::fprintf(stderr, "eager code loading: restart failed (%s); continuing with deferred loading\n",
            std::strerror(errno));
    }
    strix::set_prefill_identity_override(options.prefill_exact_fp16, options.prefill_disabled);
    const auto startup_preflight = strix::emit_qwen_startup_diagnostics(
        options.print_effective_config || options.print_cache_family || strix::decode_cli::yarn_bundle_serial_retry);
    const auto release_assets = strix::decode_cli::verify_http_release_assets(options);
    // Bind supplied numerical settings; operational paths, switches and budgets are excluded.
    const std::set<std::string_view> session_io_options{"--session-save", "--session-load", "--session-then-prompt", "--session-logits",
        "--session-cache", "--session-disk", "--session-disk-bytes", "--session-first-generate", "--session-capture-host-bytes", "--session-dir", "--session-ram-bytes", "--session-max-parked", "--http-port",
        "--generate", "--tokens-file", "--record-raw", "--reference", "--teacher-logits-file",
        "--prefix-cache-inflight", "--prefix-cache-disk", "--prefix-cache-durable"};
    // Not part of the window-cost table key: prompts, per-run outputs and HTTP front-end settings.
    const std::set<std::string_view> cost_table_skip{"--prompt-ids", "--seqs-file", "--json", "--qual-dump", "--session-dump",
        "--state-out", "--logprob-token", "--chat-logits", "--http-api-key-file", "--http-bind", "--http-model-id",
        "--http-max-connections", "--http-max-queue", "--http-sse-keepalive", "--http-sse-progress-interval",
        "--http-stream-receipts"};
    for (int i = 2; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (!arg.starts_with("--")) continue; // positional token IDs
        if (strix::qwen::profile::policy_option(arg.substr(0, arg.find('=')))) {
            if (arg == "--profile") ++i;
            continue;
        }
        if (arg == "--no-early-setup" || arg == "--no-eager-code-load") continue;
        if (arg == "--session-live-control" || arg == "--session-validate-only") continue;
        if (session_io_options.count(arg)) { ++i; continue; }
        options.session_effective_cli += std::to_string(arg.size()) + ":" + std::string(arg);
        const bool cost_key = !cost_table_skip.count(arg.substr(0, arg.find('=')));
        if (cost_key) options.cost_table_cli += std::to_string(arg.size()) + ":" + std::string(arg);
        if (i + 1 < argc && !std::string_view(argv[i + 1]).starts_with("--")) {
            const std::string_view value = argv[++i];
            options.session_effective_cli += std::to_string(value.size()) + ":" + std::string(value);
            if (cost_key) options.cost_table_cli += std::to_string(value.size()) + ":" + std::string(value);
        }
    }
    bool session_cache_enabled = !options.session_save.empty() || !options.session_load.empty() || options.session_live_control;

    strix::decode_cli::BitcalcMeasurement bitcalc;
#if defined(STRIX_MTP_MODEL_QUALIFICATION_RUNNER)
    const bool target_input_export = strix::tests::mtp_target_input_export_requested();
    if(target_input_export)strix::tests::validate_mtp_target_input_export(options,false);
#else
    constexpr bool target_input_export = false;
#endif

    // Bind before model residency/setup; retain through destruction of every
    // subsequently constructed GPU owner. Reporting happens after that cleanup.
    HostTelemetryReport host_telemetry;
    host_telemetry.enabled = options.host_dispatch_telemetry;
    host_telemetry.counters.enable_trace(options.execution_telemetry_trace);
    strix::ExecutionCorrelation request_correlation;
    request_correlation.request_id = 1; // one request per decoder process
    request_correlation.phase = strix::ExecutionPhase::admission;
    strix::ExecutionTelemetryBinding telemetry_binding(
        host_telemetry.enabled ? &host_telemetry.counters : nullptr, request_correlation);
    std::optional<strix::ExecutionPhaseScope> setup_phase;
    if (host_telemetry.enabled) setup_phase.emplace(strix::ExecutionPhase::admission);

    auto model =
        strix::ModelContract::
            qwen38_flash_next();
    // Opt-in KV identity: every state size, layout/execution identity and launcher derives from it.
    model.attention_kv_storage = options.attention_kv_storage;

    const auto generation_eos =
        options.eos_token.value_or(
            static_cast<std::uint32_t>(
                model.ple_eos_token_id));
    if (static_cast<std::uint64_t>(
            generation_eos) >=
        model.vocab_size) {
        throw std::invalid_argument(
            "generation EOS token is outside target model vocabulary");
    }

    std::optional<strix::SerialDecodeReference>
        reference;
    std::uint64_t required_reference_logit_values = 0;
    if (!options.reference_path.empty()) {
        reference =
            strix::read_serial_decode_reference(
                options.reference_path);
        strix::validate_serial_decode_reference_vocabulary(
            *reference,
            model.vocab_size);
        if (reference->steps.size() < 2) {
            throw std::invalid_argument(
                "serial replay qualification requires at least two reference steps");
        }
        for (const auto& step : reference->steps) {
            const auto probes =
                static_cast<std::uint64_t>(
                    step.logits.size());
            if (probes >
                std::numeric_limits<std::uint64_t>::max() -
                    required_reference_logit_values) {
                throw std::overflow_error(
                    "reference logit probe count overflow");
            }
            required_reference_logit_values += probes;
        }
        if (required_reference_logit_values == 0) {
            throw std::invalid_argument(
                "serial replay reference requires logit probes");
        }
        if (options.tokens.empty()) {
            options.tokens.reserve(
                reference->steps.size());
            for (const auto& step :
                 reference->steps) {
                options.tokens.push_back(
                    step.input_token);
            }
        } else {
            if (options.tokens.size() !=
                reference->steps.size()) {
                throw std::invalid_argument(
                    "CLI token sequence length does not match replay reference");
            }
            for (std::size_t index = 0;
                 index < options.tokens.size();
                 ++index) {
                if (options.tokens[index] !=
                    reference->steps[index].input_token) {
                    throw std::invalid_argument(
                        "CLI token sequence does not match replay reference");
                }
            }
        }
    }

    for (const auto token : options.tokens) {
        if (static_cast<std::uint64_t>(token) >=
            model.vocab_size) {
            throw std::invalid_argument(
                "input token is outside target model vocabulary");
        }
    }


    const auto info =
        strix::gguf::read_file(
            options.model_path);
    const auto manifest =
        strix::build_model_manifest(
            info);
    const strix::qpf2k::ModelScope defaults_scope(strix::qpf2k::has_gsq_gate(manifest));
    strix::AllocationGenerationSource
        generations;
    std::shared_ptr<const strix::MappedModelFile> mapped_owner = std::make_shared<strix::MappedModelFile>(
        strix::MappedModelFile::open_read_only(options.model_path, "model", generations));
    const auto& mapped = *mapped_owner;
    // Bind the existing full digest to its file stamp before the long model upload.
    const auto session_model_digest = session_cache_enabled
        ? strix::session_cached_file_hash(options.model_path, mapped.file_sha256())
        : strix::ModelContentIdentity{};

#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    constexpr bool has_auto_a8_runtime = true;
#else
    constexpr bool has_auto_a8_runtime = false;
#endif
    // A dense Q6P32 pack built from the validated checkpoint admits its emulated
    // derivative (the pinned file) under the validated content's profiles.
    auto admitted_content = strix::model_content_identity_hex(mapped.content_identity());
    if (!options.dense_q6p_pack_path.empty()) {
        const auto header = strix::read_q6p_pack_header(options.dense_q6p_pack_path);
        if (header.format != "Q6P32-v1" || header.target_content != admitted_content ||
            header.target_file_sha256 != strix::model_content_identity_hex(mapped.file_sha256()))
            throw std::invalid_argument("dense Q6P32 pack does not pin this model file");
        if (header.source_content == strix::decode_cli::validated_model_content_identity) {
            admitted_content = std::string(strix::decode_cli::validated_model_content_identity);
            unsigned char identity[32];
            std::memcpy(identity, mapped.content_identity().bytes.data(), 32U);
            strix::q6p::admit_derived_content(identity);
        }
    }
    // Resolve actual host coefficient bytes in bounded chunks without a device.
    // The immutable identity is replaced by an owning HIP table only after admission.
    const bool yarn = options.rotary.mode() == strix::RotaryMode::yarn;
    if (yarn) {
        strix::Sha256Fast hash;
        strix::generate_qwen_yarn_rows(options.rotary,0,options.rotary.effective_max(),
            [&](auto, auto rows) { hash.update(std::as_bytes(rows)); });
        model.resolved_rotary = strix::RotaryIdentity(options.rotary,hash.finalize());
    }
    std::unique_ptr<strix::ple_reclaim::Prefetcher> yarn_preflight_reclaimer;
    strix::decode_cli::PrefillCapabilities prompt_capabilities;
    prompt_capabilities.a8_runtime = has_auto_a8_runtime;
    // TODO(R3): automatic prefix budgets remain gated off. Qualifying them requires
    // a post-weight-residency observation after lane plans, shared prefix/capture
    // peak admission and reserved host headroom; this early observation is insufficient.
    auto prompt_memory = strix::decode_cli::observe_prefill_memory(prompt_capabilities);
    prompt_memory.estimate = [&](const strix::decode_cli::Options& candidate) {
        return strix::decode_cli::compile_prompt_requirements(candidate, model, manifest, mapped.bytes());
    };
    // Hash/map the sidecar and calibration once, before any HIP allocation.
    // Keep witnesses alive and revalidate at every candidate and runtime check.
    std::optional<strix::MappedModelFile> serving_sidecar, serving_golden;
    std::optional<strix::HipModelResidencyReservation> serving_sidecar_reservation;
    prompt_memory.serving = [&](const strix::decode_cli::Options& candidate) {
        using namespace strix;
        using namespace strix::decode_cli;
        if (!candidate.mtp.enabled) return HttpServingMemory{};
        const auto& m=candidate.mtp;
        if (!serving_sidecar || !serving_golden || !serving_sidecar_reservation) {
            serving_sidecar.emplace(MappedModelFile::open_read_only(m.sidecar,m.sidecar_logical_name,generations));
            serving_golden.emplace(MappedModelFile::open_read_only(m.golden,"calibration",generations));
            verify_mtp_serving_assets(m,serving_sidecar->file_sha256(),serving_sidecar->bytes(),
                serving_golden->file_sha256(),serving_golden->bytes());
            MtpSidecarAssetEvidence evidence;
            evidence.repository=m.repository; evidence.revision=m.revision; evidence.relative_path=m.source_path;
            evidence.declared_file_sha256=mtp_cli_digest(m.sidecar_sha256); evidence.declared_file_bytes=m.sidecar_bytes;
            evidence.verified_file_sha256=serving_sidecar->file_sha256();
            evidence.verified_model_content_identity=serving_sidecar->content_identity();
            evidence.conversion=MtpSidecarConversionProfile::effective_norm_gain_embedding_first_packed_v1;
            evidence.converter_reference_sha256=mtp_cli_digest(m.converter_sha256);
            evidence.norm_converter_reference_sha256=mtp_cli_digest(m.norm_converter_sha256);
            evidence.converter_reference_revision=m.converter_revision;
            const auto sidecar_manifest=build_mtp_sidecar_manifest(model,gguf::read_file(m.sidecar),evidence);
            HipModelResidencyPolicy policy;
            policy.mode=m.sidecar_residency=="copy"?HipModelResidencyMode::resident_copy:HipModelResidencyMode::registered_file_mapping;
            policy.reservation_limits={m.sidecar_retained_bytes,m.sidecar_setup_bytes};
            if (hip_model_copy_all_mode() && policy.mode==HipModelResidencyMode::registered_file_mapping) {
                policy.copy_tier=HipModelCopyTier::all;
                policy.copy_payload_bytes=selective_copy_payload_bytes(sidecar_manifest,HipModelCopyTier::all);
            }
            serving_sidecar_reservation=plan_hip_model_residency_reservation(*serving_sidecar,sidecar_manifest,policy);
        }
        serving_sidecar->validate_unchanged(); serving_golden->validate_unchanged();
        if (!serving_sidecar_reservation) throw std::invalid_argument("one-slot:asset verification unavailable");
        const auto golden=read_mtp_forward_golden_file(m.golden,mtp_cli_digest(m.golden_sha256),m.golden_max_bytes);
        if (golden.rows.size()!=mtp_calibration_rows(candidate))
            throw std::invalid_argument("one-slot:calibration row geometry differs");
        const auto requests=prompt_compiler_requests(candidate,model,manifest);
        kernels::KernelRegistry registry;
        register_complete_hip_kernel_catalog(model,requests.binding,requests.catalog,registry);
        const auto serial=compile_serial_execution(model,registry,requests.binding,requests.buffers,256);
        const auto index=build_model_tensor_index(manifest);
        const auto access=compile_execution_tensor_access_plan(serial.execution,manifest,index,
            build_serial_execution_tensor_requirements(serial.execution));
        const auto& reservation=*serving_sidecar_reservation;
        return http_mtp_memory(candidate,model,serial,manifest,access,reservation.setup_peak_reserved_bytes,
            prefill_checked_add(reservation.registered_alias_bytes,reservation.copied_reserved_bytes),serving_golden->bytes());
    };
    if (yarn && options.mtp.enabled) prompt_memory.estimate_with_score_ring = [&](const strix::decode_cli::Options& candidate, std::uint32_t ring_rows) {
        return strix::decode_cli::compile_prompt_requirements(candidate, model, manifest, mapped.bytes(), nullptr, false, ring_rows);
    };
    if (yarn && !options.print_effective_config) prompt_memory.verify_ple_credit = [&](const strix::decode_cli::Options& candidate,
        const strix::decode_cli::Environment& environment) -> std::uint64_t {
        return strix::decode_cli::verify_fresh_ple_preflight(yarn_preflight_reclaimer,[&]() -> std::uint64_t {
        auto config = strix::decode_cli::prefill_reclaim_config(candidate,model,manifest,environment,prompt_memory.ple_reclaim_page);
        if (!config) {
            if (strix::decode_cli::env_is(environment,"STRIX_PLE_RECLAIMABLE","1"))
                throw std::invalid_argument("YaRN required PLE reclaim route unavailable before device setup");
            return 0;
        }
        strix::decode_cli::bind_ple_reclaim_mapping(*config,mapped_owner);
        config->credit_only = true; // probes and verified credit only; runtime activates prompt work
        try {
            yarn_preflight_reclaimer = std::make_unique<strix::ple_reclaim::Prefetcher>(*config,
                strix::build_ple_hash_contract(model),candidate.tokens,true);
            return yarn_preflight_reclaimer->credit_bytes();
        } catch (const std::exception& error) {
            yarn_preflight_reclaimer.reset();
            const bool required = strix::decode_cli::env_is(environment,"STRIX_PLE_RECLAIMABLE","1");
            if (required) throw std::runtime_error(std::string("YaRN required PLE preflight refused: ") + error.what());
            std::fprintf(stderr,"ple-rolling: YaRN preflight failed (%s); zero-credit resident admission required\n",error.what());
            return 0; // an automatic unverified drop earns no credit
        }
        });
    };
    if (target_input_export) options.route_owned = options.route_owned_beyond_mtp = true;
    auto requested_environment = strix::decode_cli::requested_prefill_environment();
#if defined(__linux__)
    if (yarn) for (auto entry = ::environ; entry && *entry; ++entry) {
        const std::string_view text(*entry);
        const auto equal = text.find('=');
        if (text.starts_with("STRIX_") && equal != std::string_view::npos) {
            const auto name = text.substr(0, equal);
            if (strix::qwen::profile::metadata_environment(name)) continue;
            // The pre-injection marker is startup provenance (it survives an image restart), never a request.
            const bool injected = name == strix::qpf2k::pre_injected_env ||
                (name == "STRIX_QPF2K_MOE_DEFAULTED" && strix::qpf2k::pre_injected_moe_marker) ||
                std::any_of(strix::qpf2k::defaults.begin(), strix::qpf2k::defaults.end(), [&](const auto& item) {
                    return name == item.name && (strix::qpf2k::pre_injected & strix::qpf2k::bit(item.item));
                });
            if (!injected) requested_environment.emplace(std::string(name),std::string(text.substr(equal+1U)));
        }
    }
#endif
    auto effective = strix::decode_cli::resolve_effective_config(
        {options, requested_environment, !target_input_export}, model, info, manifest,
        admitted_content, prompt_capabilities, prompt_memory);
    options = effective.options;
    // Export only after the resolved route has admitted the profile proposals.
    strix::qwen::profile::withdraw_declined(options.product_profile);
    strix::qwen::profile::apply(options.product_profile);
    for (const auto& notice : options.product_profile.notices()) std::cerr << notice << '\n';
#if defined(STRIX_MTP_MODEL_QUALIFICATION_RUNNER)
    strix::tests::set_mtp_product_profile(options.product_profile);
#endif
    strix::set_prefill_identity_override(options.prefill_exact_fp16, options.prefill_disabled);
    session_cache_enabled = !options.session_save.empty() || !options.session_load.empty() || options.session_live_control || options.session_management_requested;
    strix::decode_cli::require_bitcalc_session_route(options, bitcalc.enabled());
    if (bitcalc.enabled() && (options.mtp.enabled || options.mtp.supplied || options.prompt_lookup ||
        options.http_serve || !options.serving_prefix_files.empty() || options.prefix_cache_roundtrip ||
        strix::decode_cli::prefix_state_cache_requested(options) || options.teacher_forced_score ||
        options.gpu_greedy_token || !options.generate_tokens))
        throw std::invalid_argument("STRIX_BITCALC hooks require one single-request full-logits greedy generation; "
            "omit --mtp, prompt lookup, serving, prefix cache, teacher scoring and --gpu-greedy-token");
    if (!options.session_modes_suppression.empty() &&
        (options.session_tiled_suffix_requested || options.session_mtp_fixed_requested))
        std::cerr << "session numerical modes suppressed: " << options.session_modes_suppression << '\n';
    if (options.session_management_requested && (!options.http_serve || !options.session_mtp_fixed ||
        !options.session_ram_bytes || !options.session_max_parked))
        throw std::invalid_argument("HTTP sessions require --session-mtp-fixed, positive RAM/count budgets");
    if (options.session_mtp_fixed && (!options.mtp.enabled || !options.mtp.draft_chain ||
        !options.mtp.draft_vocab.empty() || options.mtp.draft_context_vocab || options.mtp.draft_q4 || options.mtp.sample_temperature > 0 ||
        !options.mtp.cohort.empty() || !options.mtp.copy_wide_rows.empty() || options.mtp.ngram_draft ||
        options.mtp.sample_repeats != 1 || options.mtp.metadata_only || !options.mtp.golden_dump.empty()))
        throw std::invalid_argument("fixed MTP sessions require MTP draft chain, full draft vocabulary and greedy selection");
    if ((options.session_tiled_suffix || options.session_mtp_fixed) && !session_cache_enabled && !options.http_serve)
        throw std::invalid_argument("session route options require save/load/live-control or HTTP management budgets");
    if (session_cache_enabled || !options.session_then_prompt.empty() || !options.session_logits.empty() ||
        options.session_first_generate || options.session_validate_only) {
        if (options.session_live_control && (!options.session_save.empty() || !options.session_load.empty()))
            throw std::invalid_argument("live control cannot save or load");
        if (options.session_validate_only && options.session_load.empty())
            throw std::invalid_argument("session validation requires --session-load");
        if (options.session_first_generate && (!options.session_load.empty() || options.session_then_prompt.empty()))
            throw std::invalid_argument("first-turn budget requires same-process suffix continuation");
        if (!session_cache_enabled || ((options.mtp.enabled || options.mtp.supplied || options.http_serve) && !options.session_mtp_fixed) ||
            !options.known_prefix_window || !options.known_prefix_full_rows || options.known_prefix_prefill_rows < 1024 ||
            (options.prefix_cache_block_rows && options.prefix_cache_block_rows < 1024) || options.prompt_lookup ||
            options.prefix_cache_roundtrip || options.gpu_greedy_token || options.teacher_forced_score || options.gdn_islands ||
            !options.raw_recording_path.empty() || !options.validation_final_state_path.empty() || !options.prefix_cache_then_prompts.empty() ||
            !options.generate_tokens || options.execution.mode != strix::DecodeExecutionMode::eager ||
            options.execution.integer_head != strix::DecodeIntegerHead::none)
            throw std::invalid_argument("session cache: requires greedy tiled full-row route with >=1024-row cache units; MTP/HTTP require --session-mtp-fixed; speculation/RAW unsupported");
        if (!options.session_then_prompt.empty() && options.session_save.empty() && !options.session_live_control)
            throw std::invalid_argument("--session-then-prompt requires --session-save or --session-live-control");
        if (!options.session_logits.empty() && options.session_load.empty() && options.session_then_prompt.empty() &&
            !(options.http_serve && options.session_mtp_fixed))
            throw std::invalid_argument("--session-logits requires a loaded or same-process suffix continuation");
    }

    model.attention_kv_storage = options.attention_kv_storage;
    std::cerr << effective.serve_prefill_notice << strix::decode_cli::effective_config_notice(effective);
    if (options.http_serve || !options.print_effective_config)
        strix::decode_cli::validate_effective_startup(effective);
    strix::decode_cli::validate_http_serving_memory(options,prompt_memory);
    if (options.print_effective_config) {
        std::cout << strix::decode_cli::effective_config_json(effective);
        return 0;
    }
    strix::decode_cli::require_effective_prefill_execution(effective);
    std::unique_ptr<strix::decode_cli::YarnMemoryWatchdog> yarn_watchdog;
    if (yarn) yarn_watchdog = std::make_unique<strix::decode_cli::YarnMemoryWatchdog>();
    strix::decode_cli::YarnMemoryWatchdogScope yarn_watchdog_scope(yarn_watchdog.get());
    { strix::startup::Phase marker("preflight_reset");
      yarn_preflight_reclaimer.reset(); } // runtime owns a fresh verified drop after registration
    strix::qpf2k::undo_pre_injected([](const char* name) { ::unsetenv(name); });
    strix::qpf2k::baseline = options.baseline;
    for (const auto& [name, value] : effective.environment) {
        const bool controlled = options.baseline;
        if (!std::getenv(name.c_str()) || controlled) ::setenv(name.c_str(), value.c_str(), 1);
    }
    strix::qpf2k::defaulted_items = 0U;
    for (const auto& entry : strix::qpf2k::defaults) {
        const auto source = effective.sources.find(entry.name);
        if (source != effective.sources.end() && source->second == strix::decode_cli::OptionSource::automatic)
            strix::qpf2k::defaulted_items |= strix::qpf2k::bit(entry.item);
    }
    // Validate after the resolver injected its environment: the saved key was computed with it (CPU only, no upload).
    if (options.session_validate_only) {
        if (options.session_load.empty()) throw std::invalid_argument("session validation requires --session-load");
        strix::preflight_session_checkpoint(options.session_load, strix::session_cached_file_hash("/proc/self/exe"),
            strix::session_numerical_config(options.session_effective_cli,model.rotary_identity()), options.session_capture_host_bytes);
        std::cerr << "session preflight PASS (CPU envelope/executable/supplied settings only; no model upload)\n";
        return 0;
    }
    auto optimization_profile = effective.organization;
    const auto base_optimization_profile = optimization_profile.selected;
    strix::decode_cli::apply_numerical_profile(options, model, manifest,
        admitted_content, has_auto_a8_runtime, optimization_profile);
    strix::decode_cli::apply_model_copy_profile(options, optimization_profile);
    const bool hc_splitk_f32 = strix::numerical_profile_hc_splitk(options.numerical_profile);
    const bool hc_deferred_norm = strix::numerical_profile_hc_deferred_norm(options.numerical_profile);
    const bool q5k_a8_gateup = strix::numerical_profile_q5k_a8_gateup(options.numerical_profile);
    const bool a8_down_q51 = strix::numerical_profile_q51_a8_down(options.numerical_profile);
    const bool a8_down_q80 = strix::numerical_profile_q80_a8_down(options.numerical_profile);
    const bool dense_a8 = strix::numerical_profile_dense_a8(options.numerical_profile);
    strix::decode_cli::validate_resolved_mtp_cli(options);
#if defined(STRIX_MTP_MODEL_QUALIFICATION_RUNNER)
    if(target_input_export)strix::tests::validate_mtp_target_input_export(options,true);
#endif
    strix::decode_cli::validate_resolved_prefix_roundtrip(options);
    strix::decode_cli::validate_resolved_q8_projection_organization(options);
    strix::decode_cli::validate_resolved_hc_projection_next_tile(options);
    strix::decode_cli::validate_resolved_q51_slot_parallel_down(options);
    strix::decode_cli::validate_resolved_gdn_islands(options);
    strix::decode_cli::validate_resolved_gdn_recurrent_layout(options);
    strix::decode_cli::validate_resolved_gdn_recurrent_storage(options);
    strix::decode_cli::validate_resolved_attention_kv_storage(options);
    strix::decode_cli::validate_resolved_qsa_hierarchical_selection(options);
    strix::decode_cli::validate_resolved_qsa_gqa(options);
    strix::decode_cli::validate_resolved_known_prefix_window(options);
    strix::decode_cli::validate_moe_token_cost_options(options);
    strix::decode_cli::validate_resolved_prompt_lookup(options);
    strix::decode_cli::validate_resolved_serving(options);
    strix::decode_cli::validate_resolved_ple_cpu_helpers(options);
    strix::decode_cli::validate_resolved_ple_cache(options);
    strix::decode_cli::validate_resolved_device_queue_trace(options);
    strix::decode_cli::validate_resolved_requested_logprobs(options, model.vocab_size);
    strix::decode_cli::validate_resolved_teacher_forced_score(options, model.vocab_size);
    if (options.batched_moe_activation && !options.q4k_a8g32_gate_up)
        throw std::invalid_argument("batched MoE activation requires resolved Q4K A8 gate/up execution");
    if (options.moe_shared_overlap && !options.deferred_moe_status)
        throw std::invalid_argument("MoE shared overlap requires the resolved eager deferred-status owner");
    if (options.print_cache_family) {
        // The schedule the route executes (STRIX_QPF2K_SCHED bit 2 is inert on HTTP/MTP), as qualification compares it.
        const auto identity=resolved_cache_family(options,model,manifest,mapped,
            options.http_serve?strix::decode_cli::CacheRunner::single_mtp:strix::decode_cli::CacheRunner::cli);
        std::cout << strix::decode_cli::cache_family_identity_json(identity).dump() << '\n';
        return 0;
    }
    std::optional<strix::LoadedOfflineTuningCache> loaded_offline_tuning;
    if (!options.offline_tuning_cache_path.empty())
        loaded_offline_tuning.emplace(strix::load_offline_tuning_cache_file(options.offline_tuning_cache_path));
    std::optional<strix::OfflineCompileProvenance> loaded_compile_provenance;
    std::string provenance_executable_sha;
    if (!options.offline_compile_provenance_path.empty()) {
#if defined(__linux__)
        std::error_code executable_error;
        const auto executable = std::filesystem::canonical("/proc/self/exe", executable_error);
        if (!executable_error && !executable.empty())
            provenance_executable_sha = strix::model_content_identity_hex(strix::sha256_file_content("/proc/self/exe"));
#endif
        // All file/dependency reads are bounded setup work BEFORE GPU setup.
        // A supplied but incomplete/stale description fails closed; absence
        // leaves the previous unknown flags key and invalid_key fallback.
        loaded_compile_provenance.emplace(strix::load_offline_compile_provenance(
            options.offline_compile_provenance_path, provenance_executable_sha,
            strix::decode_cli::offline_tuning_declared_build_context_hex));
    }
    strix::PlannedPleLookupPolicy requested_ple_policy{options.ple_max_rows_per_io, options.ple_max_inflight_ios,
        options.ple_cpu_helper_threads, options.ple_cache_rows, options.ple_cache_budget_bytes, mapped.content_identity()};
    const auto requested_ple_cache_bytes = strix::PlannedPleLookupBackend::cache_reservation_bytes(requested_ple_policy);
    const std::string ple_cpu_helper_receipt = options.ple_cpu_helper_threads
        ? "  \"ple_cpu_helper_budget_requested\":" + std::to_string(options.ple_cpu_helper_threads) + ",\n" +
          "  \"ple_cpu_helper_budget_scope\":\"per-planned-backend-including-coordinator-excluding-custom-source-internal-threads\",\n" +
          "  \"ple_cpu_helpers_actual\":null,\n"
        : std::string{};
    std::string optimization_profile_receipt =
        "  \"profile\":" + options.product_profile.json() + ",\n" +
        strix::decode_cli::prefill_identity_fields(effective.prefill_option_mask) + ",\n" +
        "  \"optimization_profile_requested\": \"" + optimization_profile.requested + "\",\n" +
        "  \"optimization_profile_selected\": \"" + optimization_profile.selected + "\",\n" +
        "  \"optimization_profile_base_selected\": \"" + base_optimization_profile + "\",\n" +
        "  \"numerical_profile\": \"" + std::string(strix::numerical_profile_name(options.numerical_profile)) + "\",\n" +
        "  \"numerical_profile_qualification\": \"" +
        (options.numerical_profile == strix::DecodeNumericalProfile::exact_v5 ? std::string("unchanged-base-status") : std::string("not_assessed")) + "\",\n" +
        "  \"prefill_numerical_profile\": \"" +
        std::string(strix::numerical_profile_name(strix::decode_cli::numerical_profile_pair(options).prefill)) + "\",\n" +
        "  \"prefill_numerical_profile_source\": \"" +
        (options.prefill_profile_automatic ? "automatic" : options.prefill_numerical_profile ? "explicit" : "numerical-profile") + "\",\n" +
        "  \"numerical_profile_pair_identity\": " +
        std::to_string(strix::numerical_profile_pair_identity(strix::decode_cli::numerical_profile_pair(options))) + ",\n" +
        "  \"optimization_profile_reason\": \"" + optimization_profile.reason + "\",\n" +
        "  \"optimization_profile_automatic_applied\": " +
        (optimization_profile.automatic_applied ? "true" : "false") + ",\n" + ple_cpu_helper_receipt;
    if (yarn) optimization_profile_receipt += "  \"rotary\":" + model.rotary_identity().json() + ",\n";
    optimization_profile_receipt += strix::decode_cli::yarn_mtp_storage_receipt(effective);
    if (options.ple_cache_rows)
        optimization_profile_receipt += "  \"ple_row_cache_requested_rows\":" + std::to_string(options.ple_cache_rows) + ",\n" +
            "  \"ple_row_cache_payload_budget_bytes\":" + std::to_string(options.ple_cache_budget_bytes) + ",\n" +
            "  \"ple_row_cache_setup_reservation_bytes_per_backend\":" + std::to_string(requested_ple_cache_bytes) + ",\n" +
            "  \"ple_row_cache_qualification\":\"not_assessed\",\n";
    const std::string eager_submission_receipt = options.prepared_eager
        ? "  \"eager_submission_mode\":\"prepared-static-plan-experimental-v1\",\n"
        : std::string{};
    if (options.context_tokens >
        model.effective_max_positions()) {
        throw std::invalid_argument(
            "requested context exceeds model maximum");
    }
    if (options.tokens.size() >
        options.context_tokens) {
        throw std::invalid_argument(
            "input token sequence exceeds allocated context");
    }

    if (options.wave_moe_router_topk) strix::validate_wave_router_model_inputs(info);
    // Admission validates all names and the actual layout hash once, before
    // any GPU setup or enqueue, then retains its immutable snapshot.
    const auto wave_router_admission = options.wave_moe_router_topk ?
        strix::WaveRouterManifestAdmission::create(manifest) : nullptr;
    const auto q4k_a8_layers = options.q4k_a8g32_gate_up ?
        strix::moe_q4k_a8::classify_runtime_manifest(model,manifest) :
        strix::moe_q4k_a8::RuntimeLayerClassification{};
    // A GSQ-RCO manifest (STRIX_GSQ_Q20_A8=1) has no Q5_1 down layer: every routed layer runs the GSQ A8 down.
    if(options.grouped_exact_q51_down && q4k_a8_layers.q4k_with_q51_down_layers.empty() &&
        !strix::moe_q4k_a8::q20_manifest(q4k_a8_layers))
        throw std::invalid_argument("--grouped-exact-q51-down requires a Q4_K gate/up layer with Q5_1 down");
    if (options.q51_slot_parallel_down)
        strix::moe_q4k_a8::validate_slot_parallel_manifest(model, q4k_a8_layers);
    if (a8_down_q51 || a8_down_q80) {
        strix::moe_q4k_a8::validate_a8_down_content(mapped.content_identity());
        strix::moe_q4k_a8::validate_a8_down_manifest(model, q4k_a8_layers, a8_down_q51, a8_down_q80);
    }
    if (q5k_a8_gateup) strix::moe_q4k_a8::validate_q5k_a8_manifest(model,q4k_a8_layers);
    auto selected_q80_layers = q4k_a8_layers.q4k_with_q80_down_layers;
    if (q5k_a8_gateup) {
        selected_q80_layers.insert(selected_q80_layers.end(),q4k_a8_layers.q5k_with_q80_down_layers.begin(),
            q4k_a8_layers.q5k_with_q80_down_layers.end());
        std::sort(selected_q80_layers.begin(),selected_q80_layers.end());
    }
    const auto q4k_a8_stats = options.q4k_a8g32_gate_up ?
        std::make_shared<strix::moe_q4k_a8::RuntimeDispatchStats>() : nullptr;
    const auto hyper_arithmetic_stats = options.numerical_profile != strix::DecodeNumericalProfile::exact_v5 ?
        std::make_shared<strix::HipHyperMixArithmeticStats>() : nullptr;
    std::uint64_t hc_splitk_selected_steps = 0;
    std::uint64_t hc_deferred_selected_steps = 0;
    const auto emit_q51_slot_receipt = [&] {
        if (options.q51_slot_parallel_down)
            std::cout << "  \"q51_slot_parallel_down\": true,\n"
                         "  \"q51_slot_parallel_down_qualification\": \"not_assessed\",\n";
    };
    const auto emit_q51_slot_counts = [&] {
        if (options.q51_slot_parallel_down) std::cout << "  \"slot_parallel_q51_submissions\": "
            << (q4k_a8_stats ? q4k_a8_stats->slot_parallel_q51_submissions.load(std::memory_order_relaxed) : 0) << ",\n";
    };
    const auto emit_q4k_a8_receipt = [&] {
        emit_q51_slot_receipt();
        std::cout << "  \"grouped_exact_q51_down\": " << (options.grouped_exact_q51_down ? "true" : "false") << ",\n"
            << "  \"grouped_q51_numerical_identity\": \""
            << (a8_down_q51 ? std::string_view{"replaced-by-q51-a8-down-v1"} :
                options.grouped_exact_q51_down ? strix::moe_q4k_a8::grouped_q51_numerical_identity : std::string_view{"legacy"}) << "\",\n"
            << "  \"q4k_a8g32_gate_up\": " << (options.q4k_a8g32_gate_up ? "true" : "false") << ",\n"
            << "  \"q4k_a8g32_numerical_identity\": \""
            << ((a8_down_q51 || a8_down_q80) ? strix::moe_q4k_a8::a8_down_gate_up_numerical_identity :
                options.q4k_a8g32_gate_up ? strix::moe_q4k_a8::runtime_numerical_identity : std::string_view{"legacy"}) << "\",\n"
            << "  \"q4k_a8g32_quantized_layers\": [";
        for (std::size_t i=0;i<q4k_a8_layers.quantized_layers.size();++i)
            std::cout << (i ? "," : "") << q4k_a8_layers.quantized_layers[i];
        std::cout << "],\n  \"q4k_a8g32_legacy_q5k_layers\": [";
        for (std::size_t i=0;!q5k_a8_gateup && i<q4k_a8_layers.legacy_q5k_layers.size();++i)
            std::cout << (i ? "," : "") << q4k_a8_layers.legacy_q5k_layers[i];
        std::cout << "],\n  \"grouped_q51_eligible_layers\": [";
        for(std::size_t i=0;i<q4k_a8_layers.q4k_with_q51_down_layers.size();++i)
            std::cout << (i ? "," : "") << q4k_a8_layers.q4k_with_q51_down_layers[i];
        std::cout << "],\n";
        if (options.numerical_profile != strix::DecodeNumericalProfile::exact_v5) {
            std::cout << "  \"hc_deferred_norm_configured\":" << (hc_deferred_norm ? "true" : "false") << ",\n"
                << "  \"hc_deferred_norm_selected_model_steps\":" << hc_deferred_selected_steps << ",\n"
                << "  \"hc_projection_numerical_identity\":\"" << (hc_deferred_norm ? strix::deferred_hc_numerical_identity :
                    hc_splitk_f32 ? strix::splitk_hc_fp32_numerical_identity : std::string_view{"incumbent"}) << "\",\n"
                << "  \"q5k_a8g32_gate_up_configured\":" << (q5k_a8_gateup ? "true" : "false") << ",\n"
                << "  \"q5k_a8g32_gate_up_layers\":" << (q5k_a8_gateup ? "[2]" : "[]") << ",\n"
                << "  \"q5k_a8g32_numerical_identity\":\"" << (q5k_a8_gateup ? strix::moe_q4k_a8::q5k_numerical_identity : std::string_view{"none"}) << "\",\n"
                << "  \"dense_a8_configured\":" << (dense_a8 ? "true" : "false") << ",\n"
                << "  \"dense_a8_steps\":\"" << (dense_a8 ? "gated_delta_net,indexed_attention,shared_expert" : "") << "\",\n"
                << "  \"dense_a8_numerical_identity\":\"" << (dense_a8 ? strix::q8_projection_organization::dense_a8_numerical_identity : std::string_view{"none"}) << "\",\n";
            std::cout << "  \"hc_splitk_f32_selected_model_steps\":" << hc_splitk_selected_steps << ",\n"
                << "  \"a8_down_q51_configured\":" << (a8_down_q51 ? "true" : "false") << ",\n"
                << "  \"a8_down_q80_configured\":" << (a8_down_q80 ? "true" : "false") << ",\n"
                << "  \"q51_a8_down_eligible_layers\":[";
            if (a8_down_q51) for (std::size_t i=0;i<q4k_a8_layers.q4k_with_q51_down_layers.size();++i)
                std::cout << (i ? "," : "") << q4k_a8_layers.q4k_with_q51_down_layers[i];
            std::cout << "],\n  \"q80_a8_down_eligible_layers\":[";
            if (a8_down_q80) for (std::size_t i=0;i<selected_q80_layers.size();++i)
                std::cout << (i ? "," : "") << selected_q80_layers[i];
            std::cout << "],\n  \"a8_down_legacy_fallback_layers\":[";
            bool comma = false;
            if (a8_down_q51) for (std::uint32_t layer=0;layer<model.hidden_layers;++layer) {
                const auto& q51=q4k_a8_layers.q4k_with_q51_down_layers;
                const auto& q80=selected_q80_layers;
                if (std::find(q51.begin(),q51.end(),layer)!=q51.end() ||
                    (a8_down_q80 && std::find(q80.begin(),q80.end(),layer)!=q80.end())) continue;
                std::cout << (comma ? "," : "") << layer;
                comma = true;
            }
            std::cout << "],\n";
        }
    };
    const auto emit_q4k_a8_counts = [&] {
        std::cout << "  \"q4k_a8g32_host_quantized_submissions\": "
            << (q4k_a8_stats ? q4k_a8_stats->quantized_submissions.load(std::memory_order_relaxed) : 0)
            << ",\n  \"q4k_a8g32_host_legacy_q5k_submissions\": "
            << (q4k_a8_stats ? q4k_a8_stats->legacy_q5k_submissions.load(std::memory_order_relaxed) : 0)
            << ",\n  \"grouped_q51_host_submissions\": "
            << (q4k_a8_stats ? q4k_a8_stats->grouped_q51_submissions.load(std::memory_order_relaxed) : 0) << ",\n";
        emit_q51_slot_counts();
        if (options.numerical_profile != strix::DecodeNumericalProfile::exact_v5)
            std::cout << "  \"hc_materialized_norm_submissions\":"
                << (hyper_arithmetic_stats ? hyper_arithmetic_stats->materialized_norm_submissions.load(std::memory_order_relaxed) : 0) << ",\n"
                << "  \"hc_deferred_norm_submissions\":"
                << (hyper_arithmetic_stats ? hyper_arithmetic_stats->deferred_norm_submissions.load(std::memory_order_relaxed) : 0) << ",\n"
                << "  \"q5k_a8_gate_up_submissions\":"
                << (q4k_a8_stats ? q4k_a8_stats->q5k_a8_gate_up_submissions.load(std::memory_order_relaxed) : 0) << ",\n";
        if (options.prefill_numerical_profile && *options.prefill_numerical_profile == strix::DecodeNumericalProfile::fast_prefill_f16wmma_v1)
            std::cout << "  \"prefill_moe_f16wmma_rows\":"
                << (q4k_a8_stats ? q4k_a8_stats->prefill_f16wmma_submissions.load(std::memory_order_relaxed) : 0) << ",\n";
        // GSQ-RCO routed experts (STRIX_GSQ_Q20_A8=1): prompt rows x layers per prefill kernel form (only when any ran).
        if (q4k_a8_stats) {
            const auto gf = q4k_a8_stats->prefill_gsq_f16wmma_submissions.load(std::memory_order_relaxed);
            const auto gi = q4k_a8_stats->prefill_gsq_iu8_submissions.load(std::memory_order_relaxed);
            const auto gd = q4k_a8_stats->prefill_gsq_dot4_submissions.load(std::memory_order_relaxed);
            const auto gb = q4k_a8_stats->prefill_gsq_f16wmma_batch128_submissions.load(std::memory_order_relaxed);
            if (gf || gi || gd)
                std::cout << "  \"prefill_moe_gsq_rows\": {\"f16wmma\": " << gf << ", \"f16wmma_batch128\": " << gb
                          << ", \"iu8wmma\": " << gi << ", \"dot4\": " << gd << "},\n";
        }
        if (options.numerical_profile != strix::DecodeNumericalProfile::exact_v5)
            std::cout
                 << "  \"hc_splitk_f32_submissions\":"
                << (hyper_arithmetic_stats ? hyper_arithmetic_stats->splitk_f32_submissions.load(std::memory_order_relaxed) : 0) << ",\n"
                << "  \"q51_a8_down_submissions\":"
                << (q4k_a8_stats ? q4k_a8_stats->a8_down_q51_submissions.load(std::memory_order_relaxed) : 0) << ",\n"
                << "  \"q80_a8_down_submissions\":"
                << (q4k_a8_stats ? q4k_a8_stats->a8_down_q80_submissions.load(std::memory_order_relaxed) : 0) << ",\n"
                << "  \"q80_a8_down_legacy_fallback_submissions\":"
                << (q4k_a8_stats ? q4k_a8_stats->a8_down_q80_legacy_submissions.load(std::memory_order_relaxed) : 0) << ",\n"
                << "  \"numerical_counter_scope\":\"actual-host-enqueues-not-graph-replays-or-performance\",\n";
        if (options.batched_moe_activation) std::cout << "  \"batched_moe_activation\":true,\n"
            << "  \"batched_moe_activation_launches\":"
            << (q4k_a8_stats ? q4k_a8_stats->batched_activation_launches.load(std::memory_order_relaxed) : 0) << ",\n"
            << "  \"batched_moe_activation_slots\":"
            << (q4k_a8_stats ? q4k_a8_stats->batched_activation_slots.load(std::memory_order_relaxed) : 0) << ",\n";

    };
    strix::HipModelResidencyPolicy lane_chain_policy;lane_chain_policy.mode=options.residency;lane_chain_policy.coalesce_gap_bytes=options.coalesce_gap_bytes;lane_chain_policy.promote_file_backed_candidates=options.execution.mode==strix::DecodeExecutionMode::graph;
    std::uint32_t lane_chain_total=0;
    const auto lane_chain_steps=options.lane_chain_hyper_output?strix::lane_chain_hyper_output_steps(model,manifest,lane_chain_policy,&lane_chain_total):std::vector<std::uint32_t>{};
    if (hc_deferred_norm && (lane_chain_total != 97U || lane_chain_steps.size() != 97U))
        throw std::invalid_argument("HC deferred normalization requires all 97 metadata-eligible lane-chain mixers before allocation");
    const auto tensor_index =
        strix::build_model_tensor_index(
            manifest);
    const auto q8_organization_plan = strix::q8_projection_organization::build_manifest_plan(
        model, manifest, options.prefetch_q8_lm_head, options.wave8_q8_gdn_input,
        options.wave8_q8_gdn_output, options.wave8_q8_qsa_output);
    if (options.virtual_q8_lm_head) {
        std::size_t output_heads = 0;
        for (const auto& tensor : manifest.tensors) {
            if (tensor.role != strix::TensorRole::output_head) continue;
            ++output_heads;
            if (tensor.ggml_type != 8U || model.hidden_size % 32U != 0U)
                throw std::invalid_argument("--virtual-q8-lm-head requires an aligned Q8_0 output head");
        }
        if (output_heads != 1U)
            throw std::invalid_argument("--virtual-q8-lm-head requires exactly one output head");
    }
    if (options.virtual_q8_gdn_input) {
        if (model.full_attention_interval == 0U) throw std::invalid_argument("invalid GDN layer interval");
        bool meaningful_q8 = false;
        for (const auto& tensor : manifest.tensors) {
            if (tensor.role != strix::TensorRole::attention_projection || !tensor.layer.has_value() ||
                ((*tensor.layer + 1U) % model.full_attention_interval) == 0U) continue;
            if (tensor.ggml_type == 8U &&
                (tensor.name.ends_with(".attn_qkv.weight") || tensor.name.ends_with(".attn_gate.weight")))
                meaningful_q8 = true;
        }
        // GSQ-RCO (STRIX_GSQ_Q20_A8=1): GDN inputs may all be typed (dense A8 segments); the virtual Q8 view then
        // covers only the Q8_0 ones.
        if ((!meaningful_q8 && !strix::moe_q4k_a8::q20_a8_enabled()) || model.hidden_size % 32U != 0U)
            throw std::invalid_argument("--virtual-q8-gdn-input requires an aligned Q8_0 GDN input projection");
    }
    if(options.virtual_q8_gdn_output) strix::validate_virtual_q8_gdn_output_manifest(model,manifest);
    const std::string lane_chain_mode_receipt=
        std::string("  \"lane_chain_hyper_output\": ") + (options.lane_chain_hyper_output ? "true" : "false") + ",\n" +
        "  \"lane_chain_hyper_output_eligible_steps\": " + std::to_string(lane_chain_steps.size()) + ",\n" +
        "  \"lane_chain_hyper_output_fallback_steps\": " + std::to_string(options.lane_chain_hyper_output ? lane_chain_total-lane_chain_steps.size() : 0U) + ",\n" +
        (options.prefetch_hc_projection_next_tile ?
            "  \"prefetch_hc_projection_next_tile\": true,\n"
            "  \"hc_projection_next_tile_qualification\": \"not_assessed\",\n" : "");
    const auto qsa_prefetch_plan=options.prefetch_q8_qsa_input ? strix::qsa_input_prefetch::build_plan(model,manifest) : strix::qsa_input_prefetch::Plan{};
    std::string qsa_input_mode_receipt="  \"prefetch_q8_qsa_input\": "+std::string(options.prefetch_q8_qsa_input?"true":"false")+",\n";
    if(options.prefetch_q8_qsa_input) qsa_input_mode_receipt+="  \"prefetch_q8_qsa_input_eligible_steps\": "+std::to_string(qsa_prefetch_plan.eligible_steps)+",\n";
    if(options.prefetch_q8_qsa_input) qsa_input_mode_receipt+="  \"prefetch_q8_qsa_input_fallback_steps\": "+std::to_string(qsa_prefetch_plan.fallback_steps)+",\n";
    if(options.prefetch_q8_qsa_input) qsa_input_mode_receipt+="  \"prefetch_q8_qsa_input_eligible_projections\": "+std::to_string(qsa_prefetch_plan.eligible_projections)+",\n";
    if(options.prefetch_q8_qsa_input) qsa_input_mode_receipt+="  \"prefetch_q8_qsa_input_fallback_projections\": "+std::to_string(qsa_prefetch_plan.fallback_projections)+",\n";
    if(options.prefetch_q8_qsa_input) qsa_input_mode_receipt+="  \"prefetch_q8_qsa_input_projection_choices\": [";
    for(std::size_t i=0;i<qsa_prefetch_plan.projections.size();++i){const auto& choice=qsa_prefetch_plan.projections[i];if(i)qsa_input_mode_receipt+=",";qsa_input_mode_receipt+="{\"layer\":"+std::to_string(choice.layer)+",\"projection\":"+std::to_string(i%5U)+",\"tensor_name\":\""+strix::decode_receipt::escape_json_string(choice.tensor_name)+"\",\"projection_kernel\":\""+std::string(choice.eligible?strix::qsa_input_prefetch::projection_prefetch_id:strix::qsa_input_prefetch::projection_base_id)+"\"}";}
    if(options.prefetch_q8_qsa_input) qsa_input_mode_receipt+="],\n";
    const auto qsa_prefetch_stats=options.prefetch_q8_qsa_input ?
        std::make_shared<strix::qsa_input_prefetch::RuntimeDispatchStats>() : nullptr;
    const auto qsa_gqa_stats = options.qsa_gqa_heads_per_block != 1 ?
        std::make_shared<strix::QsaHierarchicalSelectionStats>() : nullptr;
    const auto qsa_selection_stats = options.qsa_hierarchical_selection ?
        std::make_shared<strix::QsaHierarchicalSelectionStats>() : nullptr;
    const auto shared_prefetch_plan=options.prefetch_q8_shared_expert?strix::shared_expert_prefetch::build_plan(model,manifest):strix::shared_expert_prefetch::Plan{};
    std::string shared_mode_receipt="  \"prefetch_q8_shared_expert\": "+std::string(options.prefetch_q8_shared_expert?"true":"false")+",\n";
    if(options.prefetch_q8_shared_expert) shared_mode_receipt+="  \"prefetch_q8_shared_expert_eligible_steps\": "+std::to_string(shared_prefetch_plan.eligible_steps)+",\n";
    if(options.prefetch_q8_shared_expert) shared_mode_receipt+="  \"prefetch_q8_shared_expert_fallback_steps\": "+std::to_string(shared_prefetch_plan.fallback_steps)+",\n";
    if(options.prefetch_q8_shared_expert) shared_mode_receipt+="  \"prefetch_q8_shared_expert_eligible_projections\": "+std::to_string(shared_prefetch_plan.eligible_projections)+",\n";
    if(options.prefetch_q8_shared_expert) shared_mode_receipt+="  \"prefetch_q8_shared_expert_fallback_projections\": "+std::to_string(shared_prefetch_plan.fallback_projections)+",\n";
    if(options.prefetch_q8_shared_expert) shared_mode_receipt+="  \"prefetch_q8_shared_expert_projection_choices\": [";
    for(std::size_t i=0;i<shared_prefetch_plan.projections.size();++i){const auto& c=shared_prefetch_plan.projections[i];if(i)shared_mode_receipt+=",";shared_mode_receipt+="{\"layer\":"+std::to_string(c.layer)+",\"projection\":"+std::to_string(i%3U)+",\"tensor_name\":\""+strix::decode_receipt::escape_json_string(c.tensor_name)+"\",\"projection_kernel\":\""+std::string(c.eligible?strix::shared_expert_prefetch::projection_prefetch_id:strix::shared_expert_prefetch::projection_base_id)+"\"}";}
    if(options.prefetch_q8_shared_expert) shared_mode_receipt+="],\n";
    const auto shared_prefetch_stats=options.prefetch_q8_shared_expert ?
        std::make_shared<strix::shared_expert_prefetch::RuntimeDispatchStats>() : nullptr;
    // Static mode/plan metadata may be emitted before execution. Dispatch
    // counts must be read separately after the selected workload completes.
    const auto gdn_output_mode_receipt=[&] {
        std::string receipt=(options.virtual_q8_gdn_output ? "  \"virtual_q8_gdn_output\": true,\n" : "")+shared_mode_receipt+qsa_input_mode_receipt+lane_chain_mode_receipt;
        if (options.prefetch_q8_lm_head || options.wave8_q8_lm_head || options.wave8_q8_gdn_input ||
            options.wave8_q8_gdn_output || options.wave8_q8_qsa_output) {
            receipt += "  \"prefetch_q8_lm_head\": " + std::string(options.prefetch_q8_lm_head ? "true" : "false") + ",\n";
            receipt += "  \"wave8_q8_lm_head\": " + std::string(options.wave8_q8_lm_head ? "true" : "false") + ",\n";
            receipt += "  \"wave8_q8_gdn_input\": " + std::string(options.wave8_q8_gdn_input ? "true" : "false") + ",\n";
            receipt += "  \"wave8_q8_gdn_output\": " + std::string(options.wave8_q8_gdn_output ? "true" : "false") + ",\n";
            receipt += "  \"wave8_q8_qsa_output\": " + std::string(options.wave8_q8_qsa_output ? "true" : "false") + ",\n";
            receipt += "  \"q8_projection_organization_qualification\": \"not_assessed\",\n";
        }
        return receipt;
    };
    const std::string deferred_moe_mode_receipt = "  \"deferred_moe_status\": " +
        std::string(options.deferred_moe_status ? "true" : "false") + ",\n";
    const strix::decode_detail::SharedPrefetchDispatchReceipt shared_dispatch_receipt(shared_prefetch_stats);
    strix::startup::Phase first_hip_marker("first_hip_guard");
    const auto original_work_device = std::make_shared<strix::HipSetupQuiescenceGuard>();
    first_hip_marker.end();
    if (yarn) {
        model.qwen_rotary_table = std::make_shared<const strix::HipQwenRotaryTable>(options.rotary);
        if (model.qwen_rotary_table->identity() != model.resolved_rotary)
            throw std::runtime_error("YaRN uploaded table differs from admitted host identity");
        strix::validate_qwen_rotary_reader(*model.qwen_rotary_table,nullptr);
    }
    auto target_architecture = original_work_device->architecture();
    if (const auto suffix = target_architecture.find(':'); suffix != std::string::npos)
        target_architecture.resize(suffix);
    if (target_architecture != "gfx1151")
        throw std::invalid_argument("native decoder requires observed gfx1151 without architecture override");

    strix::ExecutionBindingRequest
        binding_request;
    binding_request.logical_rows = 1;
    binding_request.request_concurrency = 1;
    binding_request.minimum_qualification =
        strix::kernels::
            QualificationStatus::
                experimental;

    const bool graph_mode = options.execution.mode == strix::DecodeExecutionMode::graph;
    binding_request.require_graph_safe = graph_mode;
    binding_request.device_graph_inputs = graph_mode;
    strix::HipLauncherCatalogConfig catalog_config;
    catalog_config.activation = strix::HipActivationStorage::bf16;
    catalog_config.enable_device_graph_inputs = graph_mode;
    catalog_config.parallel_hyper_mix = options.parallel_hyper_mix;
    catalog_config.staged_hyper_projections = options.staged_hyper_projections;
    catalog_config.fused_staged_hyper_mix = options.fused_staged_hyper_mix;
    catalog_config.typed_hyper_projection = options.typed_hyper_projection;
    catalog_config.prefetch_typed_hyper_rows = options.prefetch_typed_hyper_rows;
    catalog_config.prefetch_hc_projection_next_tile = options.prefetch_hc_projection_next_tile;
    catalog_config.numerical_profile = options.numerical_profile;
    catalog_config.splitk_hyper_projection = hc_splitk_f32;
    catalog_config.splitk_deferred_norm = hc_deferred_norm;
    catalog_config.splitk_hyper_projection_fp64 = false;
    catalog_config.arithmetic_stats = hyper_arithmetic_stats;
    binding_request.prefetch_hc_projection_next_tile = options.prefetch_hc_projection_next_tile;
    catalog_config.prefetch_q8_hyper_output = options.prefetch_q8_hyper_output;
    catalog_config.lane_chain_hyper_output=options.lane_chain_hyper_output;
    binding_request.lane_chain_hyper_output=options.lane_chain_hyper_output;
    binding_request.lane_chain_hyper_output_eligible_steps=lane_chain_steps;
    catalog_config.virtual_q8_lm_head = options.virtual_q8_lm_head;
    catalog_config.prefetch_q8_lm_head = options.prefetch_q8_lm_head;
    catalog_config.wave8_q8_lm_head = options.wave8_q8_lm_head;
    catalog_config.wave8_q8_gdn_input = options.wave8_q8_gdn_input;
    catalog_config.gdn_recurrent_layout = binding_request.gdn_recurrent_layout = options.gdn_recurrent_layout;
    catalog_config.gdn_recurrent_storage = binding_request.gdn_recurrent_storage = options.gdn_recurrent_storage;
    catalog_config.wave8_q8_gdn_output = options.wave8_q8_gdn_output;
    catalog_config.wave8_q8_qsa_output = options.wave8_q8_qsa_output;
    binding_request.prefetch_q8_lm_head = options.prefetch_q8_lm_head;
    binding_request.wave8_q8_lm_head = options.wave8_q8_lm_head;
    binding_request.wave8_q8_gdn_input = options.wave8_q8_gdn_input;
    binding_request.wave8_q8_gdn_output = options.wave8_q8_gdn_output;
    binding_request.wave8_q8_qsa_output = options.wave8_q8_qsa_output;
    catalog_config.batched_moe_activation = options.batched_moe_activation;
    binding_request.batched_moe_activation = options.batched_moe_activation;
    catalog_config.q4k_a8g32_gate_up = options.q4k_a8g32_gate_up;
    catalog_config.grouped_exact_q51_down = options.grouped_exact_q51_down;
    catalog_config.q51_slot_parallel_down = options.q51_slot_parallel_down;
    catalog_config.qsa_hierarchical_selection = options.qsa_hierarchical_selection;
    // STRIX_QSA_DECODE_POOL (default on; =0 or the baseline off): persistent pooled index keys for eager serial decode,
    // only where one process appends to one session (no MTP windows, serving, prompt lookup, prefix/session restores).
    catalog_config.qsa_decode_pool = strix::decode_cli::qsa_decode_pool_route(options, strix::qpf2k::gsq_model);
    catalog_config.qsa_decode_pool_bf16 = strix::decode_cli::qsa_decode_pool_bf16(options, strix::qpf2k::gsq_model);
    catalog_config.qsa_selection_stats = qsa_selection_stats;
    catalog_config.qsa_gqa_heads_per_block = binding_request.qsa_gqa_heads_per_block = options.qsa_gqa_heads_per_block;
    catalog_config.qsa_gqa_stats = qsa_gqa_stats;
    binding_request.qsa_hierarchical_selection = options.qsa_hierarchical_selection;
    catalog_config.q4k_a8_stats = q4k_a8_stats;
    binding_request.q4k_a8g32_gate_up = options.q4k_a8g32_gate_up;
    binding_request.grouped_exact_q51_down = options.grouped_exact_q51_down;
    binding_request.q51_slot_parallel_down = options.q51_slot_parallel_down;
    catalog_config.a8_down_q51 = binding_request.a8_down_q51 = a8_down_q51;
    catalog_config.a8_down_q80 = binding_request.a8_down_q80 = a8_down_q80;
    catalog_config.q5k_a8g32_gate_up = binding_request.q5k_a8g32_gate_up = q5k_a8_gateup;
    catalog_config.dense_a8 = binding_request.dense_a8 = dense_a8;
    catalog_config.deferred_moe_status = options.deferred_moe_status;
    binding_request.deferred_moe_status = options.deferred_moe_status;
    catalog_config.ranked_moe_router_topk = options.ranked_moe_router_topk;
    binding_request.ranked_moe_router_topk = options.ranked_moe_router_topk;
    catalog_config.wave_moe_router_topk = options.wave_moe_router_topk;
    catalog_config.wave_router_manifest_admission = wave_router_admission;
    binding_request.wave_moe_router_topk = options.wave_moe_router_topk;
    catalog_config.staged_moe_router_logits = options.staged_moe_router_logits;
    binding_request.staged_moe_router_logits = options.staged_moe_router_logits;
    catalog_config.parallel_moe_router = options.parallel_moe_router;
    binding_request.parallel_moe_router = options.parallel_moe_router;
    catalog_config.virtual_q8_gdn_input = options.virtual_q8_gdn_input;
    catalog_config.prefetch_q8_gdn_input = options.prefetch_q8_gdn_input;
    catalog_config.prefetch_q8_qsa_input=options.prefetch_q8_qsa_input;
    catalog_config.qsa_input_prefetch_layers=qsa_prefetch_plan.layers;
    catalog_config.qsa_input_prefetch_stats=qsa_prefetch_stats;
    binding_request.prefetch_q8_qsa_input=options.prefetch_q8_qsa_input;
    binding_request.qsa_input_prefetch_layers=qsa_prefetch_plan.layers;
    catalog_config.prefetch_q8_shared_expert=options.prefetch_q8_shared_expert;
    catalog_config.shared_expert_prefetch_layers=shared_prefetch_plan.layers;
    catalog_config.shared_expert_prefetch_stats=shared_prefetch_stats;
    binding_request.prefetch_q8_shared_expert=options.prefetch_q8_shared_expert;
    binding_request.shared_expert_prefetch_layers=shared_prefetch_plan.layers;
    catalog_config.virtual_q8_gdn_output = options.virtual_q8_gdn_output;
    std::optional<strix::integer_projection::RuntimeConfig> integer_config;
    if (options.execution.integer_head != strix::DecodeIntegerHead::none) {
        integer_config = strix::make_decode_integer_head_config(model, manifest, options.execution,
            {original_work_device->device_index(), original_work_device->observed_binding_identity(), target_architecture});
    }

    strix::kernels::KernelRegistry
        kernel_registry;
    strix::register_complete_hip_kernel_catalog(
        model,
        binding_request,
        catalog_config,
        kernel_registry, integer_config ? &*integer_config : nullptr);

    strix::ExecutionBufferRequest
        buffer_request;
    buffer_request.logical_rows = 1;
    buffer_request.parallel_moe_router = options.parallel_moe_router;
    buffer_request.q4k_a8g32_gate_up = options.q4k_a8g32_gate_up;
    buffer_request.grouped_exact_q51_down = options.grouped_exact_q51_down;
    buffer_request.q51_slot_parallel_down = options.q51_slot_parallel_down;
    buffer_request.a8_down_q51 = a8_down_q51;
    buffer_request.a8_down_q80 = a8_down_q80;
    buffer_request.q5k_a8g32_gate_up = q5k_a8_gateup;
    buffer_request.dense_a8 = dense_a8;
    buffer_request.qsa_hierarchical_selection = options.qsa_hierarchical_selection;
    buffer_request.qsa_gqa_heads_per_block = options.qsa_gqa_heads_per_block;
    buffer_request.activation_element_bytes =
        2;
    buffer_request.control_element_bytes =
        4;
    buffer_request.index_element_bytes =
        4;
    buffer_request.logit_element_bytes =
        4;
    buffer_request.mlp_intermediate_element_bytes =
        4;

    strix::startup::Phase compilation_marker("plan_compilation");
    auto compiled =
        strix::compile_serial_execution(
            model,
            kernel_registry,
            binding_request,
            buffer_request,
            256);
    if (hc_splitk_f32) {
        for (const auto& step : compiled.bindings.bindings)
            if (step.kernel_id && step.kernel_id->find(".splitk-hc-proj-fp32-v1") != std::string::npos)
                ++hc_splitk_selected_steps;
        if (hc_deferred_norm) {
            for (const auto& step : compiled.bindings.bindings)
                if (step.kernel_id && step.kernel_id->find(".hc-deferred-norm-v1") != std::string::npos &&
                    step.kernel_id->ends_with(strix::lane_chain_hyper_output_suffix)) ++hc_deferred_selected_steps;
            if (hc_deferred_selected_steps != 97U)
                throw std::invalid_argument("HC deferred normalization rejects partial or fallback compiled mixers");
        }
        if (hc_splitk_selected_steps != 97)
            throw std::invalid_argument("HC numerical profile requires all 97 compiled splitk FP32 mixer steps");
    }
    // Split prefill numerical profile: the prompt prefill runs a sibling serial execution whose
    // dense Q8_0 projections follow the prefill profile (kernel ids, binding/buffer identities and,
    // through them, the known-prefix prefill ids and launcher configs); decode keeps `compiled`.
    const auto profile_pair = strix::decode_cli::numerical_profile_pair(options);
    std::optional<strix::CompiledSerialExecution> compiled_prefill;
    auto prefill_catalog_config = catalog_config;
    if (profile_pair.split()) {
        const bool prefill_dense_a8 = strix::numerical_profile_dense_a8(profile_pair.prefill);
        // A fast prefill profile keeps its decode sibling's kernel plan (dense A8 equal) and differs in the
        // arithmetic of its prefill launchers; the dense A8 pairs differ in the dense projections' plan.
        const bool fast = strix::numerical_profile_fast_prefill(profile_pair.prefill);
        if (strix::numerical_profile_hc_splitk(profile_pair.prefill) != hc_splitk_f32 ||
            strix::numerical_profile_hc_deferred_norm(profile_pair.prefill) != hc_deferred_norm ||
            strix::numerical_profile_q5k_a8_gateup(profile_pair.prefill) != q5k_a8_gateup ||
            strix::numerical_profile_q51_a8_down(profile_pair.prefill) != a8_down_q51 ||
            strix::numerical_profile_q80_a8_down(profile_pair.prefill) != a8_down_q80 ||
            (fast ? prefill_dense_a8 != dense_a8 : prefill_dense_a8 == dense_a8) || integer_config)
            throw std::invalid_argument("prefill numerical profile may differ from the decode profile only in dense A8 "
                "or in the prefill-only arithmetic of a fast prefill profile");
        auto prefill_binding_request = binding_request;
        auto prefill_buffer_request = buffer_request;
        prefill_binding_request.dense_a8 = prefill_buffer_request.dense_a8 = prefill_dense_a8;
        std::string gsq_moe_receipt;
        if (fast) {
            const auto identity = strix::prefill_arithmetic_identity(profile_pair,
                effective.prefill_option_mask, strix::qpf2k::has_gsq_gate(manifest));
            if (strix::qpf2k::has_gsq_gate(manifest) && (profile_pair.fast_components & strix::kFastPrefillMoe))
                gsq_moe_receipt = "  \"prefill_gsq_moe_arithmetic\": \"" + std::string(strix::moe_q4k_a8::gsq_fast_prefill_moe_name()) + "\",\n";
            prefill_binding_request.prefill_arithmetic_identity = identity;
        }
        prefill_catalog_config.numerical_profile = profile_pair.prefill;
        prefill_catalog_config.dense_a8 = prefill_dense_a8;
        prefill_catalog_config.fast_prefill_components = profile_pair.fast_components;
        strix::kernels::KernelRegistry prefill_kernel_registry;
        strix::register_complete_hip_kernel_catalog(model, prefill_binding_request, prefill_catalog_config,
            prefill_kernel_registry, nullptr);
        compiled_prefill = strix::compile_serial_execution(model, prefill_kernel_registry,
            prefill_binding_request, prefill_buffer_request, 256);
        strix::validate_sibling_serial_executions(*compiled_prefill, compiled);
        optimization_profile_receipt += gsq_moe_receipt;
        optimization_profile_receipt +=
            "  \"prefill_dense_a8_configured\": " + std::string(prefill_dense_a8 ? "true" : "false") + ",\n" +
            "  \"prefill_fast_components\": " + std::to_string(profile_pair.fast_components) + ",\n" +
            "  \"prefill_moe_route_bf16\": " + std::string(options.prefill_moe_route_bf16 ? "true" : "false") + ",\n" +
            "  \"prefill_storage_identity\": " + ((profile_pair.fast_components & strix::kFastPrefillMoe) ?
                "\"" + std::string(strix::moe_prefill_storage_name(options.prefill_moe_route_bf16)) + "\"" : "null") + ",\n" +
            "  \"prefill_storage_identity_hash\": " + ((profile_pair.fast_components & strix::kFastPrefillMoe) ?
                std::to_string(strix::moe_prefill_storage_identity(options.prefill_moe_route_bf16)) : "null") + ",\n" +
            "  \"prefill_serial_execution_identity\": " + std::to_string(compiled_prefill->execution_identity) + ",\n" +
            "  \"prefill_serial_kernel_plan_identity\": " + std::to_string(compiled_prefill->kernel_plan_identity) + ",\n" +
            "  \"prefill_serial_scratch\": \"shared-sibling-view-of-decode-serial-arena\",\n";
    }
    std::optional<strix::PreparedOfflineTuning> offline_tuning_owner;
    std::optional<strix::kernels::OfflineTuningPlan> offline_tuning_plan;
    if (loaded_offline_tuning) {
        original_work_device->require_original();
        hipDeviceProp_t properties{};
        check_hip(hipGetDeviceProperties(&properties, original_work_device->device_index()), "observe offline tuning device properties");
        int runtime_version = 0, driver_version = 0;
        check_hip(hipRuntimeGetVersion(&runtime_version), "observe offline tuning runtime version");
        check_hip(hipDriverGetVersion(&driver_version), "observe offline tuning driver version");
        original_work_device->require_original();
        if (!std::memchr(properties.name, '\0', sizeof(properties.name)))
            throw std::invalid_argument("offline tuning runtime SKU is not bounded");
        const auto declared = strix::decode_cli::offline_tuning_declared_build_context_hex;
        const auto declared_sha = strix::model_content_identity_hex(strix::sha256_bytes(
            std::as_bytes(std::span(declared.data(), declared.size()))));
        strix::OfflineTuningLiveContext live;
        live.gfx_target = original_work_device->architecture();
        live.device_sku = std::string(properties.name) + ";processor-count-domain=HIP-reported-multiprocessorCount";
        if (properties.multiProcessorCount > 0)
            live.runtime_reported_processor_count = static_cast<std::uint32_t>(properties.multiProcessorCount);
        live.hip_identity = "runtime:" + std::to_string(runtime_version) + ";driver:" + std::to_string(driver_version) +
            ";declared-cmake-context:" + declared_sha;
        std::string binary_identity_source = "unavailable";
        if (loaded_compile_provenance) {
            live.binary_sha256 = provenance_executable_sha;
            binary_identity_source = "canonical-proc-self-exe-inode-setup-once";
        }
#if defined(__linux__)
        std::error_code executable_error;
        const auto executable = std::filesystem::canonical("/proc/self/exe", executable_error);
        if (!loaded_compile_provenance && !executable_error && !executable.empty()) {
            // Hash the process inode, not a pathname that could be replaced
            // after canonicalization. No model bytes are rehashed here.
            live.binary_sha256 = strix::model_content_identity_hex(strix::sha256_file_content("/proc/self/exe"));
            binary_identity_source = "canonical-proc-self-exe-inode-setup-once";
        }
#endif
        live.model_content_sha256 = strix::model_content_identity_hex(mapped.content_identity());
        live.compiler_flags_sha256 = loaded_compile_provenance ? loaded_compile_provenance->compiler_flags_sha256 :
            std::string(strix::decode_cli::offline_tuning_live_compiler_flags_sha256);
        live.context_tokens = options.context_tokens;
        strix::SerialSessionStateLayoutRequest tuning_state_request;
        tuning_state_request.context_tokens = options.context_tokens;
        tuning_state_request.recurrent_storage = options.gdn_recurrent_storage;
        live.state_layout_identity = strix::build_serial_session_state_layout(model, tuning_state_request).layout_identity;
        live.memory_regime = std::string(options.residency == strix::HipModelResidencyMode::registered_file_mapping ?
            "configured-registered-mapping" : "configured-copy") + ";coalesce:" + std::to_string(options.coalesce_gap_bytes) +
            ";planned-PLE:" + std::to_string(options.planned_ple_io) + ";PLE-helper-cap:" + std::to_string(options.ple_cpu_helper_threads) +
            ";PLE-cache-rows:" + std::to_string(options.ple_cache_rows) + ";PLE-cache-bytes:" + std::to_string(options.ple_cache_budget_bytes);
        live.resolved_configuration = "baseline-kernel-plan:" + std::to_string(compiled.kernel_plan_identity) +
            ";baseline-execution:" + std::to_string(compiled.execution_identity) + ";prepared:" + std::to_string(options.prepared_eager) +
            ";overlap:" + std::to_string(options.moe_shared_overlap) + ";GPU-scalar:" + std::to_string(options.gpu_greedy_token);
        for (const auto* name : {"GPU_MAX_HW_QUEUES", "HSA_ENABLE_SDMA", "HSA_OVERRIDE_GFX_VERSION", "AMD_SERIALIZE_KERNEL",
                "AMD_SERIALIZE_COPY", "HIP_LAUNCH_BLOCKING", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"}) {
            live.resolved_configuration += ';'; live.resolved_configuration += name; live.resolved_configuration += '=';
            if (const auto* value = std::getenv(name)) {
                std::size_t size = 0;
                while (size <= 4096U && value[size]) ++size;
                if (size > 4096U) throw std::invalid_argument("offline tuning runtime setting is oversized");
                live.resolved_configuration.append(value,size);
            } else live.resolved_configuration += "<absent>";
        }
        const auto tuning_access = strix::compile_execution_tensor_access_plan(compiled.execution, manifest, tensor_index,
            strix::build_serial_execution_tensor_requirements(compiled.execution));
        offline_tuning_owner.emplace(std::move(*loaded_offline_tuning), live, model, compiled, manifest, tuning_access,
            kernel_registry, binding_request);
        loaded_offline_tuning.reset();
        offline_tuning_plan = offline_tuning_owner->plan();
        binding_request.offline_tuning = &*offline_tuning_plan;
        // The only tuned query is preparation. All owners/spans outlive binding
        // and execution; no file load, model hash or tuning search per token.
        compiled = strix::compile_serial_execution(model, kernel_registry, binding_request, buffer_request, 256);
        std::array<std::uint64_t,5> reasons{};
        for (const auto& binding : compiled.bindings.bindings)
            if (binding.tuning_selection_reason) {
                const auto index = static_cast<std::size_t>(*binding.tuning_selection_reason);
                if (index >= reasons.size()) throw std::logic_error("unknown offline tuning selection reason");
                ++reasons[index];
            }
        if (loaded_compile_provenance)
            optimization_profile_receipt += "  \"offline_compile_provenance_files_checked\":" +
                std::to_string(loaded_compile_provenance->files_checked) + ",\n" +
                "  \"offline_compile_provenance_units\":" + std::to_string(loaded_compile_provenance->units) + ",\n" +
                "  \"offline_compile_provenance_targets\":" + std::to_string(loaded_compile_provenance->targets) + ",\n" +
                "  \"offline_compile_provenance_referenced_bytes\":" + std::to_string(loaded_compile_provenance->referenced_bytes) + ",\n";
        optimization_profile_receipt += "  \"offline_tuning_cache_sha256\":\"" + offline_tuning_owner->payload_sha256() + "\",\n" +
            "  \"offline_tuning_cache_bytes\":" + std::to_string(offline_tuning_owner->payload_bytes()) + ",\n" +
            "  \"offline_tuning_cache_records\":" + std::to_string(offline_tuning_owner->record_count()) + ",\n" +
            "  \"offline_tuning_declared_cmake_context_sha256\":\"" + declared_sha + "\",\n" +
            "  \"offline_tuning_build_context_provenance\":\"DECLARED-CMake-not-observed-or-attested-argv\",\n" +
            "  \"offline_tuning_compiler_flags_provenance\":\"" +
                (loaded_compile_provenance ? std::string("evaluated-per-TU-argv-and-declared-inputs-binary-bound-not-driver-runtime-closure-or-attestation") :
                 std::string(strix::decode_cli::offline_tuning_flags_provenance)) + "\",\n" +
            "  \"offline_tuning_compiler_flags_sha256\":" +
                (loaded_compile_provenance ? "\"" + loaded_compile_provenance->compiler_flags_sha256 + "\"" : std::string("null")) + ",\n" +
            "  \"offline_tuning_runtime_reported_processor_count\":" + std::to_string(live.runtime_reported_processor_count) + ",\n" +
            "  \"offline_tuning_processor_count_scope\":\"HIP-multiprocessorCount-not-physical-CU-proof\",\n" +
            "  \"offline_tuning_binary_sha256\":" + (live.binary_sha256.empty() ? "null" : "\"" + live.binary_sha256 + "\"") + ",\n" +
            "  \"offline_tuning_binary_identity_source\":\"" + binary_identity_source + "\",\n" +
            "  \"offline_tuning_selection_reason_counts\":{\"qualified_record\":" + std::to_string(reasons[0]) +
            ",\"invalid_key\":" + std::to_string(reasons[1]) + ",\"ambiguous_records\":" + std::to_string(reasons[2]) +
            ",\"no_qualified_record\":" + std::to_string(reasons[3]) + ",\"selection_policy_mismatch\":" + std::to_string(reasons[4]) + "},\n";
    }
    std::optional<strix::MoeForkJoinArenaPlan> overlap_arena;
    if (options.moe_shared_overlap) {
        overlap_arena.emplace(strix::prepare_moe_fork_join_arena(
            compiled, options.moe_overlap_arena_budget_bytes));
        compiled = overlap_arena->rebind_execution(compiled);
    }

    std::uint32_t wave_eligible_steps = 0;
    for (const auto& binding : compiled.bindings.bindings)
        if(binding.op == strix::SerialOp::moe_route && binding.kernel_id && binding.kernel_id->ends_with(".wave.parallel"))
            ++wave_eligible_steps;
    if (wave_eligible_steps > compiled.execution.moe_route_steps ||
        (options.wave_moe_router_topk && (!wave_eligible_steps || wave_eligible_steps != compiled.execution.moe_route_steps)))
        throw std::invalid_argument("wave router compiled bindings do not cover every route step");
    const auto wave_fallback_steps = compiled.execution.moe_route_steps - wave_eligible_steps;
    const std::string wave_router_mode_receipt =
        "  \"staged_moe_router_logits\": " + std::string(options.staged_moe_router_logits ? "true" : "false") + ",\n" +
        "  \"staged_moe_router_logits_admission\": \"runtime-F32-H2560-E512-weight16B-otherwise-original-logits\",\n" +
        "  \"staged_moe_router_logits_qualification\": \"experimental-not-assessed\",\n" +
        "  \"wave_moe_router_topk\": " + std::string(options.wave_moe_router_topk ? "true" : "false") + ",\n" +
        "  \"wave_moe_router_topk_eligible_steps\": " + std::to_string(wave_eligible_steps) + ",\n" +
        "  \"wave_moe_router_topk_fallback_steps\": " + std::to_string(wave_fallback_steps) + ",\n";
    const auto tensor_access = strix::compile_execution_tensor_access_plan(
        compiled.execution, manifest, tensor_index,
        strix::build_serial_execution_tensor_requirements(compiled.execution));
    strix::qsa_input_prefetch::validate_tensor_access(qsa_prefetch_plan,manifest,compiled.execution,tensor_access);
    strix::shared_expert_prefetch::validate_tensor_access(shared_prefetch_plan,manifest,compiled.execution,tensor_access);
    strix::decode_cli::validate_prompt_tensor_contract(options,manifest,compiled,tensor_access);
    const auto hc_next_tile_plan = strix::hc_projection_prefetch::build_plan(
        model, manifest, compiled, tensor_access, options.prefetch_hc_projection_next_tile);

    compilation_marker.end();
    strix::SerialSessionStateLayoutRequest
        state_request;
    state_request.context_tokens =
        options.context_tokens;
    state_request.recurrent_storage = options.gdn_recurrent_storage;
    const auto state_layout =
        strix::build_serial_session_state_layout(
            model,
            state_request);


    if (!options.validation_final_state_path.empty())
        strix::validate_final_generation_state_budget(state_layout, {options.validation_final_state_maximum_bytes});
    strix::HipModelResidencyPolicy
        residency_policy;
    residency_policy.mode =
        options.residency;
    residency_policy.coalesce_gap_bytes =
        options.coalesce_gap_bytes;
    // Graph mode explicitly opts into a completely resident model, including
    // its PLE table; registered/file-backed views cannot enter capture.
    residency_policy.promote_file_backed_candidates = graph_mode;
    residency_policy.copy_tier = options.model_copy_tier;
    residency_policy.copy_payload_bytes = options.model_copy_payload_bytes;
    residency_policy.reservation_limits.retained_bytes = options.model_retained_budget;
    residency_policy.reservation_limits.setup_peak_bytes = options.model_setup_peak_budget;
    // STRIX_MODEL_COPY_ALL (opt-in): the registered model becomes one page-congruent copied arena, without any
    // userptr registration (see hip_model_residency.hpp). Only replaces the default registered backing.
    if (strix::hip_model_copy_all_mode() && options.residency == strix::HipModelResidencyMode::registered_file_mapping &&
        options.model_copy_tier == strix::HipModelCopyTier::none && !graph_mode) {
        residency_policy.copy_tier = strix::HipModelCopyTier::all;
        residency_policy.coalesce_gap_bytes = 0; // nothing is registered: the registration gap has no ranges to merge
        residency_policy.copy_payload_bytes = strix::selective_copy_payload_bytes(manifest, strix::HipModelCopyTier::all);
        residency_policy.reservation_limits.retained_bytes = std::numeric_limits<std::uint64_t>::max();
        residency_policy.reservation_limits.setup_peak_bytes = std::numeric_limits<std::uint64_t>::max();
        std::cerr << "model residency: STRIX_MODEL_COPY_ALL copies " << *residency_policy.copy_payload_bytes
                  << " payload bytes into one arena (no registration)\n";
    }

    // STRIX_LOAD_BALLOON_BYTES=N (opt-in, default 0): reserve N bytes of device memory BEFORE the model is registered
    // and release them after registration and the PLE warm. The page-cache reclaim that the later engine reservations
    // (KV/state, workspaces, arenas, MTP sidecar) would cause then happens while nothing is registered, and those
    // reservations find the released memory free, so the load reclaim never invalidates registered model pages
    // (an invalidated userptr range evicts every queue of the process until it is revalidated).
    void* load_balloon = nullptr;
    if (const char* v = std::getenv("STRIX_LOAD_BALLOON_BYTES"); v && std::strtoull(v, nullptr, 10) &&
        options.residency == strix::HipModelResidencyMode::registered_file_mapping) {
        const auto bytes = static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
        if (hipMalloc(&load_balloon, bytes) != hipSuccess) { load_balloon = nullptr; (void)hipGetLastError(); }
        else if (hipMemset(load_balloon, 0, bytes) != hipSuccess || hipDeviceSynchronize() != hipSuccess) (void)hipGetLastError();
        std::cerr << "load balloon: " << (load_balloon ? "reserved " : "could not reserve ") << bytes << " bytes before registration\n";
    }
    auto model_owner = std::make_shared<strix::HipModelResidency>(
        strix::HipModelResidency::create(mapped_owner, manifest, generations, residency_policy));
    auto& model_residency = *model_owner;
    std::shared_ptr<const strix::HipQ6pPack> dense_q6p_pack;
    if (!options.dense_q6p_pack_path.empty()) {
        dense_q6p_pack = strix::load_hip_q6p_pack(options.dense_q6p_pack_path, mapped, manifest,
            model_residency.binding(), generations);
        std::cerr << "dense Q6P32 pack: " << dense_q6p_pack->q6p_tensors << " Q6P32 + " << dense_q6p_pack->bf16_tensors
                  << " BF16 tensors, " << dense_q6p_pack->q6p_bytes << " bytes replace " << dense_q6p_pack->replaced_bytes
                  << (strix::q6p::kernels_enabled() ? "" : " (STRIX_Q6P=0: not read)") << '\n';
    }
    std::shared_ptr<const strix::HipDenseTrellisPack> dense_trellis_pack;
    std::string dense_trellis_configuration_receipt;
#if defined(STRIX_HAS_EXPERIMENTAL_DENSE_TRELLIS)
    if (!options.dense_trellis_pack_path.empty()) {
        dense_trellis_pack = strix::load_hip_dense_trellis_pack(options.dense_trellis_pack_path, mapped, manifest,
            model_residency.binding(), generations);
        dense_trellis_configuration_receipt = "  \"dense_trellis_pack_sha256\":\"" + dense_trellis_pack->sha256 + "\",\n" +
            "  \"dense_trellis_recipe\":\"fp32-transform-fma-split4-v1\",\n" +
            "  \"dense_trellis_scope\":\"decode-r1-r2-r4-r8\",\n" +
            "  \"dense_trellis_tensors\":" + std::to_string(dense_trellis_pack->tensors) + ",\n" +
            "  \"dense_trellis_packed_bytes\":" + std::to_string(dense_trellis_pack->packed_bytes) + ",\n" +
            "  \"dense_trellis_replaced_q8_bytes\":" + std::to_string(dense_trellis_pack->replaced_bytes) + ",\n" +
            "  \"dense_trellis_workspace_bytes\":" + std::to_string(dense_trellis_pack->workspace_bytes) + ",\n" +
            "  \"dense_trellis_qualification\":\"experimental-not-assessed\",\n";
        optimization_profile_receipt += dense_trellis_configuration_receipt;
    }
#endif
    const auto dense_trellis_completed_receipt = [&] {
        if (!dense_trellis_pack) return std::string{};
        return "  \"dense_trellis_used_layer_mask\":" + std::to_string(strix::hip_dense_trellis_used_layers()) + ",\n" +
            "  \"dense_trellis_usage_scope\":\"host-entry-coverage-including-capture-not-cycle-counts\",\n";
    };
    // Auto tries the same admission as =1, falling back silently to the warm path if setup cannot establish it.
    // Admitted runs retain =1's data path, credit, diagnostics and receipt (src/ple_reclaim.cpp).
    const auto ple_reclaim_mode = strix::decode_cli::ple_reclaim_mode(options, residency_policy, graph_mode);
    const bool ple_reclaim_requested = ple_reclaim_mode != strix::ple_reclaim::Mode::off;
    if (!ple_reclaim_requested && strix::ple_reclaim::requested() != strix::ple_reclaim::Mode::off) {
        const char* rolling = std::getenv("STRIX_PLE_ROLLING"), *exact = std::getenv("STRIX_CTX_EXACT_MEMORY");
        if ((!rolling || rolling[0] != '0') && (!exact || exact[0] != '0')) {
            const auto witness = strix::decode_cli::ple_reclaim_inactive_witness(options, residency_policy, graph_mode);
            std::fputs(witness.c_str(), stderr);
        }
    }
    // The leased MTP route completes sidecar hashing and registration before
    // setup drop/credit, then reuses these actual immutable owners for repeats.
    auto ple_mtp_assets = ple_reclaim_requested && options.mtp.enabled
        ? strix::decode_cli::load_mtp_cli_assets(options.mtp, model, mapped_owner, manifest, model_owner, generations)
        : std::shared_ptr<const strix::HipMtpAssets>{};
    // After registration: pinning the model may evict anything populated before it.
    std::uint64_t warmed_file_backed_bytes = 0;
    if (!ple_reclaim_requested) warmed_file_backed_bytes = warm_file_backed_tensors(mapped, manifest);
    if (load_balloon) { (void)hipFree(load_balloon); load_balloon = nullptr; std::cerr << "load balloon: released\n"; }
    // Keep the actual registration/copy owner and its exact map control block
    // pinned in the optional inventory; this is virtual backing, not RSS/PFN evidence.
    std::optional<strix::UmaOwnerInventory> model_owner_inventory;
    if (options.execution_inventory || options.known_prefix_prefill_rows)
        model_owner_inventory.emplace(model_residency.owner_inventory(model_owner, mapped_owner));
    std::shared_ptr<strix::ple_reclaim::Prefetcher> ple_reclaimer;
    if (ple_reclaim_requested) {
        const auto rolling_h = strix::decode_cli::ple_rolling_prompt_supported(options) && strix::ctx_exact_memory_enabled("STRIX_PLE_ROLLING") ? strix::ple_reclaim::rolling_positions(std::getenv("STRIX_PLE_ROLLING"),
            std::getenv("STRIX_CTX_EXACT_MEMORY"), std::getenv("STRIX_PLE_ROLLING_POSITIONS")) : 0U;
        const auto cap = rolling_h
            ? strix::ple_reclaim::rolling_cap_bytes(std::getenv("STRIX_PLE_ROLLING_CAP_BYTES")) : 0U;
        ple_reclaimer = strix::decode_cli::setup_ple_or_resident(ple_reclaim_mode, [&] {
            const strix::ManifestTensor* table = nullptr;
            for (const auto& tensor : manifest.tensors) {
                if (tensor.residency != strix::ResidencyPolicy::file_backed_candidate) continue;
                if (table || tensor.role != strix::TensorRole::ngram_table || tensor.dimensions.size() != 2 ||
                    (tensor.ggml_type != 20 && tensor.ggml_type != 30) || (tensor.ggml_type == 20 && tensor.dimensions[0] % 32U))
                    throw std::invalid_argument("STRIX_PLE_RECLAIMABLE=1 supports exactly one file-backed IQ4_NL/BF16 n-gram table");
                table = &tensor;
            }
            if (!table || !model_owner_inventory) throw std::invalid_argument("STRIX_PLE_RECLAIMABLE=1: no file-backed PLE table");
#if defined(__linux__)
            const auto page = static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
#else
            const std::uint64_t page = 4096; // Explicit =1 still fails in Prefetcher with its Linux requirement.
#endif
            std::vector<strix::ple_reclaim::Interval> registered;
            for (const auto& view : model_owner_inventory->pinned_views)
                if (view.category == "weights.registered-host")
                    registered.push_back({view.view_offset, view.view_offset + view.view_bytes});
            strix::ple_reclaim::Config config;
            config.mapping_base = mapped.base_address();
            config.mapping_bytes = mapped.bytes();
            config.page = page;
            config.table_offset = table->absolute_offset;
            config.table_rows = table->dimensions[1];
            config.stored_row_bytes = table->ggml_type == 20 ? table->dimensions[0] / 32U * 18U : table->dimensions[0] * 2U;
            if (config.stored_row_bytes * config.table_rows != table->byte_size)
                throw std::invalid_argument("STRIX_PLE_RECLAIMABLE=1: n-gram table extent does not match its rows");
            config.reclaim = strix::ple_reclaim::reclaim_intervals(table->absolute_offset, table->byte_size, page, std::move(registered));
            config.rolling_positions = rolling_h;
            config.rolling_cap_bytes = cap;
            // A mapper covering any interior PLE page retains the old contract.
            std::uint64_t reclaim_bytes = 0;
            for (const auto& r : config.reclaim) reclaim_bytes += r.end - r.begin;
            const auto interior = (table->absolute_offset + table->byte_size) / page * page -
                (table->absolute_offset + page - 1U) / page * page;
            if (config.rolling_positions && reclaim_bytes != interior) {
                if (options.mtp.enabled) throw std::runtime_error("PLE MTP rolling refuses mapper-protected interior pages");
                config.rolling_positions = 0;
                std::fprintf(stderr, "ple-rolling: prompt-wide mode retained: mapper protects interior PLE pages\n");
            }
            const auto contract = strix::build_ple_hash_contract(model);
            config.bounded_turns = options.mtp.enabled;
            strix::decode_cli::bind_ple_reclaim_mapping(config,mapped_owner);
            // The leased route charges a fixed replaceable source cache. Target
            // replay/verification reads the table; the predictor has no PLE op.
            // The historical route retains its G-dependent allowance.
            const std::uint64_t lookups_per_token = options.mtp.enabled
                ? (options.mtp.candidate_depth ? options.mtp.candidate_depth + 1ULL : 9ULL) : 1ULL;
            const char* decode_bytes = std::getenv("STRIX_PLE_RECLAIM_DECODE_BYTES");
            config.decode_allowance_bytes = strix::ple_reclaim::decode_allowance(config.rolling_positions ? 0U : options.generate_tokens, config.rolling_positions ? 1U : lookups_per_token,
                contract.head_count, page, decode_bytes ? std::strtoull(decode_bytes, nullptr, 10) : 0U);
            config.ahead_rows = 2U * options.known_prefix_prefill_rows;
            if (config.rolling_positions)
                config.critical_rows = options.known_prefix_prefill_rows;
            if (const char* v = std::getenv("STRIX_PLE_RECLAIM_AHEAD_ROWS"); v && std::strtoull(v, nullptr, 10))
                config.ahead_rows = std::strtoull(v, nullptr, 10);
            if (const char* v = std::getenv("STRIX_PLE_RECLAIM_STRICT")) config.strict = v[0] != '0';   // default strict
            if (const char* v = std::getenv("STRIX_PLE_STAGE_PREFETCH")) config.stage_prefetch = v[0] != '0';
            return std::make_shared<strix::ple_reclaim::Prefetcher>(config, contract,
                std::span<const std::uint32_t>(options.tokens.data(), options.tokens.size()),
                ple_reclaim_mode == strix::ple_reclaim::Mode::automatic);
        }, [&] {
            strix::ple_reclaim::set_active(nullptr);
            ple_reclaimer.reset();
            yarn_preflight_reclaimer.reset();
            effective.ple_reclaim_credit=0;
            effective.ple_rolling_h=effective.ple_rolling_bound=effective.ple_rolling_cap=0;
            effective.environment["STRIX_PLE_RECLAIMABLE"]="0";
        }, [&] {
            auto caps=prompt_capabilities;
            const auto fresh=strix::decode_cli::observe_prefill_memory(caps);
            const auto required=strix::decode_cli::compile_prompt_requirements(options,model,manifest,mapped.bytes(),nullptr,true);
            const auto admitted=strix::decode_cli::resident_prefill_admission(options,required,fresh);
            for(const auto& [name,bytes]:admitted) effective.admission[name]=bytes;
        }, [&] { warmed_file_backed_bytes = warm_file_backed_tensors(mapped, manifest); });
        if (ple_reclaimer) strix::ple_reclaim::set_active(ple_reclaimer.get());
    }
    if (yarn) {
        if ((ple_reclaimer ? ple_reclaimer->credit_bytes() : 0U) < effective.ple_reclaim_credit)
            throw std::runtime_error("YaRN verified PLE credit lost after model registration; refusing state allocation");
        yarn_preflight_reclaimer.reset();
        const auto observed = strix::observe_linux_uma_memory();
        if (!observed.memory.mem_total_bytes || !observed.memory.mem_available_bytes ||
            !strix::decode_cli::prefill_physical_margin_fits(*observed.memory.mem_total_bytes,
                *observed.memory.mem_available_bytes,effective.admission.at("required_bytes"),
                effective.admission.at("incremental_bytes") - options.rotary.table_bytes()))
            throw std::runtime_error("YaRN available headroom lost after model registration; refusing state allocation");
        auto post_capabilities = prompt_capabilities;
        const auto post = strix::decode_cli::observe_prefill_memory(post_capabilities);
        if (!post.gtt_limit || !post.gtt_used || *post.gtt_used > *post.gtt_limit ||
            effective.admission.at("device_growth_bytes") - options.rotary.table_bytes() > *post.gtt_limit - *post.gtt_used)
            throw std::runtime_error("YaRN GTT headroom lost after model registration; refusing state allocation");
    }
    const auto model_owner_inventory_receipt = [&]() -> std::string {
        if (!model_owner_inventory) return {};
        const auto& inventory = *model_owner_inventory;
        const auto& reservation = model_residency.reservation();
        return "  \"model_backing_inventory_kind\":\"retained-owner-virtual-extents-not-physical-residency\",\n" +
            std::string("  \"model_backing_inventory_unique_count\":") + std::to_string(inventory.known_backings.size()) + ",\n" +
            "  \"model_backing_inventory_reservation_upper_bound_bytes\":" + std::to_string(inventory.reservation_upper_bound_bytes) + ",\n" +
            "  \"model_backing_inventory_retained_reserved_bytes\":" + std::to_string(reservation.retained_reserved_bytes) + ",\n" +
            "  \"model_backing_inventory_setup_peak_reserved_bytes\":" + std::to_string(reservation.setup_peak_reserved_bytes) + ",\n";
    }();
    const auto model_copy_receipt = [&]() -> std::string {
        const auto& reservation = model_residency.reservation();
        return std::string("  \"model_copy_tier\":\"") + std::string(strix::hip_model_copy_tier_name(model_residency.copy_tier())) + "\",\n" +
            "  \"model_copy_qualification\":\"not_assessed\",\n" +
            "  \"model_copy_payload_bytes\":" + std::to_string(reservation.copy_payload_bytes) + ",\n" +
            "  \"model_copy_arena_reserved_bytes\":" + std::to_string(model_residency.copy_tier() == strix::HipModelCopyTier::none ? 0U : reservation.copied_reserved_bytes) + ",\n" +
            "  \"model_copy_tensor_count\":" + std::to_string(reservation.copy_tensor_count) + ",\n" +
            "  \"model_copy_range_count\":" + std::to_string(reservation.copy_range_count) + ",\n" +
            "  \"model_registered_alias_bytes\":" + std::to_string(reservation.registered_alias_bytes) + ",\n" +
            "  \"model_retained_reserved_bytes\":" + std::to_string(reservation.retained_reserved_bytes) + ",\n" +
            "  \"model_setup_peak_reserved_bytes\":" + std::to_string(reservation.setup_peak_reserved_bytes) + ",\n" +
            "  \"model_copy_reservation_scope\":\"declared-virtual-backing-not-physical-residency\",\n";
    }();
    strix::q8_projection_organization::validate_resident_weights(q8_organization_plan, model_residency.binding());
    // Admission of actual device views happens before any token can enqueue work.
    for(const auto& projection:qsa_prefetch_plan.projections) {
        if(!projection.eligible) continue;
        const auto found=std::find_if(manifest.tensors.begin(),manifest.tensors.end(),
            [&](const auto& tensor){return tensor.name==projection.tensor_name;});
        if(found==manifest.tensors.end()) throw std::logic_error("QSA admitted tensor disappeared");
        const auto index=static_cast<std::uint32_t>(found-manifest.tensors.begin());
        const auto* runtime=strix::find_runtime_tensor_binding(model_residency.binding(),index);
        const auto bytes=strix::qsa_input_prefetch::required_weight_bytes(found->dimensions[0],found->dimensions[1]);
        if(!runtime || runtime->address==0 || runtime->address%16U!=0 || runtime->bytes<bytes)
            throw std::invalid_argument("QSA prefetch resident weight alignment/extent invalid");
    }
    // Admit all actual eligible device views before the first token can enqueue.
    for(const auto& projection:shared_prefetch_plan.projections) {
        if(!projection.eligible) continue;
        const auto tensor=std::find_if(manifest.tensors.begin(),manifest.tensors.end(),
            [&](const auto& t){return t.name==projection.tensor_name;});
        if(tensor==manifest.tensors.end())throw std::logic_error("shared admitted tensor disappeared");
        const auto index=static_cast<std::uint32_t>(tensor-manifest.tensors.begin());
        const auto* runtime=strix::find_runtime_tensor_binding(model_residency.binding(),index);
        const auto bytes=strix::shared_expert_prefetch::required_weight_bytes(tensor->dimensions[0],tensor->dimensions[1]);
        if(!runtime || runtime->address==0 || runtime->address%16U || runtime->bytes<bytes)
            throw std::invalid_argument("shared prefetch resident weight alignment/extent invalid");
    }


    if (reference.has_value() &&
        reference->model_content_identity !=
            model_residency.binding().
                model_content_identity) {
        throw std::invalid_argument(
            "reference checkpoint SHA-256 does not match loaded model");
    }

    if (!options.serving_prefix_files.empty()) {
        // No ordinary single-session arena is allocated on this route.
        std::vector<std::vector<std::uint32_t>> prefixes;
        prefixes.reserve(options.serving_prefix_files.size());
        for (const auto& path : options.serving_prefix_files)
            prefixes.push_back(read_serving_prefix(path, options.context_tokens, model.vocab_size));
        const auto checked_add = [](std::uint64_t left, std::uint64_t right) {
            if (right > UINT64_MAX - left) throw std::overflow_error("serving reservation overflows");
            return left + right;
        };
        std::shared_ptr<const strix::HipServingPrefillPlan> serving_prefill;
        if(options.serving_prefill_rows) serving_prefill=std::make_shared<const strix::HipServingPrefillPlan>(
            strix::HipServingGreedyBackend::prepare_prefill(model,compiled,manifest,tensor_access,
                options.serving_prefill_rows,options.serving_prefill_workspace_budget,options.serving_prefill_metadata_budget));
        auto minimum = checked_add(compiled.arena.total_bytes, compiled.arena.alignment - 1U);
        minimum = checked_add(minimum, checked_add(state_layout.device_total_bytes, state_layout.device_alignment - 1U));
        minimum = checked_add(minimum, state_layout.host_total_bytes);
        minimum = checked_add(minimum, sizeof(strix::DeferredMoeFirstError));
        if (model.vocab_size > UINT32_MAX) throw std::invalid_argument("serving GPU selector vocabulary exceeds uint32");
        minimum = checked_add(minimum, strix::HipServingGreedyBackend::output_reservation_bytes(static_cast<std::uint32_t>(model.vocab_size)));
        minimum = checked_add(minimum, sizeof(ServingSessionOwner));
        minimum = checked_add(minimum, requested_ple_cache_bytes);
        if(serving_prefill) minimum=checked_add(minimum,serving_prefill->per_request_reservation_bytes);
        std::optional<strix::HipServingPrefixCachePolicy> prefix_cache_policy;
        std::uint64_t prefix_cache_reservation = 0;
        if (options.serving_cache_entries_explicit) {
            prefix_cache_policy = strix::HipServingPrefixCachePolicy{
                static_cast<std::size_t>(options.serving_cache_entries), options.serving_cache_snapshot_bytes,
                options.serving_cache_capture_peak_bytes, options.serving_cache_max_prefix_tokens};
            if(serving_prefill) {
                const auto recipe=serving_prefill->state_tiled ?
                    strix::SerialPrefixProducerRecipe::known_prefill(compiled,*serving_prefill->state_tiled,*serving_prefill->terminal_tiled) :
                    strix::SerialPrefixProducerRecipe::known_window(compiled,serving_prefill->state_window,serving_prefill->terminal_window);
                prefix_cache_policy->max_history_intervals=recipe.maximum_history_intervals(options.serving_cache_max_prefix_tokens);
            }
            prefix_cache_reservation = strix::HipServingPrefixCache::requested_reservation_bytes(*prefix_cache_policy);
        }
        const auto cohort_reservation = options.serving_tensor_cohort_rows == 1 ? 0U :
            strix::HipServingCohort::requested_reservation_bytes(options.serving_tensor_cohort_rows);
        const auto shared_bytes = checked_add(checked_add(checked_add(checked_add(model_residency.reservation().retained_reserved_bytes,
            options.serving_headroom), prefix_cache_reservation), cohort_reservation),
            serving_prefill?serving_prefill->metadata_payload_bytes:0U);
        if (shared_bytes > options.serving_memory_budget)
            throw std::invalid_argument("serving model reservation plus headroom exceeds total budget");
        // Charge retained snapshots AND capture peak before pool construction or
        // any request factory allocation. The pool owns independent snapshots.
        std::shared_ptr<strix::HipServingPrefixCache> prefix_cache;
        if (prefix_cache_policy) {
            original_work_device->require_original();
            prefix_cache = std::make_shared<strix::HipServingPrefixCache>(*prefix_cache_policy,
                mapped.content_identity(), static_cast<std::uint32_t>(model.vocab_size));
        }
        std::shared_ptr<strix::HipServingCohort> cohort;
        if (options.serving_tensor_cohort_rows != 1) {
            original_work_device->require_original();
            cohort = std::make_shared<strix::HipServingCohort>(model_owner, options.serving_tensor_cohort_rows,
                [](strix::SerialGreedyStepBackend& target) -> strix::HipServingGreedyBackend* {
                    if (auto* observed = dynamic_cast<ObservedServingBackend*>(&target)) return observed->native_backend();
                    return dynamic_cast<strix::HipServingGreedyBackend*>(&target);
                });
        }
        strix::GreedyServingPolicy policy;
        policy.concurrency = options.serving_concurrency;
        policy.tensor_cohort_rows = options.serving_tensor_cohort_rows;
        if(serving_prefill) policy.prefill_chunk_rows=serving_prefill->rows;
        policy.max_retained_requests = prefixes.size();
        policy.reservation_budget_bytes = options.serving_memory_budget;
        policy.shared_reservation_bytes = shared_bytes;
        strix::ServingCompatibility compatibility{model_residency.binding().model_content_identity,
            compiled.kernel_plan_identity, original_work_device->device_index(), model.rotary_identity()};
        strix::GreedyServingScheduler scheduler(policy, compatibility, model_owner, cohort);
        std::vector<std::shared_ptr<ServingSessionOwner>> owners;
        std::vector<strix::ServingRequestHandle> handles;
        owners.reserve(prefixes.size()); handles.reserve(prefixes.size());
        // Borrowed while retained request handles own their backend; read only
        // after run_until_idle, before either handles or scheduler are destroyed.
        std::vector<strix::HipServingGreedyBackend*> native_targets(prefixes.size(),nullptr);
        setup_phase.reset();
        const auto serving_start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < prefixes.size(); ++i) {
            const auto maximum_emissions = std::min(options.generate_tokens,
                options.context_tokens - static_cast<std::uint64_t>(prefixes[i].size()) + 1U);
            if (maximum_emissions > UINT64_MAX / 8 || prefixes[i].size() > UINT64_MAX / 8)
                throw std::overflow_error("serving token reservation overflows");
            const auto token_bytes = checked_add(maximum_emissions * 8U, static_cast<std::uint64_t>(prefixes[i].size()) * 8U);
            if (checked_add(minimum, token_bytes) > options.serving_request_budget)
                throw std::invalid_argument("serving request budget is below known state/scratch/selector/token payload");
            // Reserve an exclusive four-generation range: scratch, device state,
            // host state, D1 record. The factory proves the first actual epoch.
            const auto epoch = generations.next();
            for (unsigned reserved = 0; reserved < 3; ++reserved) static_cast<void>(generations.next());
            auto owner = std::make_shared<ServingSessionOwner>(mapped_owner, model_owner, original_work_device,
                model, compiled, manifest, wave_router_admission, tensor_access, state_layout,
                requested_ple_policy,
                options.planned_ple_io, epoch, static_cast<std::uint64_t>(i) + 1U,
                options.host_dispatch_telemetry, options.execution_telemetry_trace);
            strix::SerialGreedyRequest serve_request{prefixes[i], model.vocab_size, options.context_tokens,
                options.generate_tokens, generation_eos};
            const auto submission = scheduler.submit(serve_request, compatibility,
                reinterpret_cast<std::uintptr_t>(&owner->session), epoch, options.serving_request_budget, owner,
                [&, owner, i]() -> std::unique_ptr<strix::SerialGreedyStepBackend> {
                    std::optional<strix::ExecutionTelemetryBinding> observed;
                    if (owner->telemetry()) observed.emplace(&owner->counters, owner->correlation(0));
                    owner->initialize([&](const strix::ExecutionMemoryBinding& scratch) {
                        strix::q8_projection_organization::validate_output_steps(q8_organization_plan,
                            model, compiled, tensor_access, scratch, model_residency.binding());
                        strix::hc_projection_prefetch::validate_runtime(hc_next_tile_plan,
                            model, manifest, compiled, tensor_access, scratch, model_residency.binding());
                    });
                    auto native = std::make_unique<strix::HipServingGreedyBackend>(
                        compiled, model, catalog_config, owner->resources, owner->session, owner->scope,
                        std::static_pointer_cast<const void>(owner), options.prepared_eager, prefix_cache, serving_prefill);
                    native_targets[i]=native.get();
                    std::unique_ptr<strix::SerialGreedyStepBackend> target=std::move(native);
                    if (owner->telemetry()) return std::make_unique<ObservedServingBackend>(std::move(target), owner);
                    return target;
                });
            if (submission.admission != strix::ServingAdmission::accepted)
                throw std::invalid_argument("serving request was rejected by bounded admission");
            owners.push_back(std::move(owner)); handles.push_back(submission.request);
        }
        scheduler.run_until_idle();
        const auto serving_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - serving_start).count();
        bool passed = true;
        std::cout << "{\n" << optimization_profile_receipt << model_owner_inventory_receipt
            << "  \"serving\":true,\n  \"serving_qualification\":\"not_assessed\",\n"
            << "  \"serving_execution\":\"" << (cohort ? "independent-C1-targets-shared-projection-tensor-cohorts-rest-R1" : "independent-C1-sessions-and-owned-streams-not-tensor-batching") << "\",\n"
            << "  \"gdn_recurrent_storage\":\"" << (options.gdn_recurrent_storage == strix::GdnRecurrentStorage::key_value ? "key-value" : "value-key") << "\",\n"
            << "  \"serving_concurrency\":" << options.serving_concurrency << ",\n"
            << "  \"serving_prepared_eager\":" << (options.prepared_eager ? "true" : "false") << ",\n"
            << "  \"serving_request_budget_bytes\":" << options.serving_request_budget << ",\n"
            << "  \"serving_shared_and_headroom_reservation_bytes\":" << shared_bytes << ",\n"
            << "  \"serving_reserved_bytes_with_live_handles\":" << scheduler.reserved_bytes() << ",\n"
            << "  \"serving_reservation_scope\":\"declared-backing-and-request-budgets-not-physical-RSS-or-fit-proof\",\n"
            << "  \"serving_admission_to_completion_elapsed_ms\":" << serving_ms << ",\n"
            << "  \"serving_prefix_cache_requested\":" << (prefix_cache ? "true" : "false") << ",\n"
            << "  \"serving_prefix_cache_reservation_bytes\":" << prefix_cache_reservation << ",\n";
        if(serving_prefill) {
            std::cout << "  \"serving_prefill\":{\"rows\":" << serving_prefill->rows
                << ",\"qualification\":\"not_assessed\",\"execution\":\"known-input-native-chunks-with-R1-tail\""
                << ",\"workspace_payload_bytes\":" << serving_prefill->workspace_bytes
                << ",\"known_plan_metadata_payload_bytes\":" << serving_prefill->metadata_payload_bytes
                << ",\"runtime_payload_reservation_bytes\":" << serving_prefill->runtime_payload_bytes
                << ",\"per_request_payload_reservation_bytes\":" << serving_prefill->per_request_reservation_bytes
                << ",\"scope\":\"declared-payload-plus-explicit-serving-headroom-not-RSS-or-fit\"},\n";
        }
        if (cohort) {
            const auto observed = cohort->stats();
            std::cout << "  \"serving_tensor_cohort\":{\"requested_rows\":" << options.serving_tensor_cohort_rows
                << ",\"qualification\":\"not_assessed\",\"scope\":\"three-shared-Q8-projections-tensor-rows-rest-R1\""
                << ",\"staging_reservation_bytes\":" << cohort_reservation
                << ",\"prepared_execution\":\"" << (options.prepared_eager ? "metadata-admitted-cohort-layer-major-direct" : "cohort-layer-major-direct") << "\""
                << ",\"cohorts\":" << observed.cohorts << ",\"targets\":" << observed.targets
                << ",\"shared_tensor_steps\":" << observed.shared_tensor_steps
                << ",\"fallback_r1_steps\":" << observed.fallback_r1_steps
                << ",\"saturated\":" << (observed.saturated ? "true" : "false") << "},\n";
        }
        if (prefix_cache) {
            const auto cache = prefix_cache->stats();
            std::cout << "  \"serving_prefix_cache\":{\"scope\":\"shared-pool-actual-operations-not-speed-or-traffic\","
                << "\"max_entries\":" << options.serving_cache_entries
                << ",\"snapshot_budget_bytes\":" << options.serving_cache_snapshot_bytes
                << ",\"capture_peak_bytes\":" << options.serving_cache_capture_peak_bytes
                << ",\"max_prefix_tokens\":" << options.serving_cache_max_prefix_tokens
                << ",\"max_history_intervals\":" << prefix_cache_policy->max_history_intervals
                << ",\"matching\":\"" << (serving_prefill?"exact-prefix-and-compiled-producer-history":"longest-serial-prefix") << "\""
                << ",\"lookups\":" << cache.lookups << ",\"hits\":" << cache.hits
                << ",\"restored_prefix_tokens\":" << cache.restored_prefix_tokens
                << ",\"capture_attempts\":" << cache.capture_attempts << ",\"inserted\":" << cache.inserted
                << ",\"skipped\":" << cache.skipped << ",\"failures\":" << cache.failures
                << ",\"entries\":" << cache.entries << ",\"snapshot_bytes\":" << cache.snapshot_bytes
                << ",\"saturated\":" << (cache.saturated ? "true" : "false") << "},\n";
        }
        std::cout << "  \"requests\":[";
        for (std::size_t i = 0; i < handles.size(); ++i) {
            const auto state = handles[i].state(); const auto metrics = handles[i].metrics();
            const auto& result = handles[i].result();
            passed = passed && state == strix::ServingRequestState::completed;
            if (i) std::cout << ',';
            std::cout << "{\"request_id\":" << i + 1U << ",\"state\":" << static_cast<unsigned>(state)
                << ",\"execution_identity\":" << owners[i]->session.execution_identity
                << ",\"runtime_compatibility_identity\":" << owners[i]->session.runtime_compatibility_identity
                << ",\"committed_tokens\":" << owners[i]->session.committed_tokens
                << ",\"restored_prefix_tokens\":" << result.restored_prefix_tokens
                << ",\"final_position\":" << result.final_position
                << ",\"steps_executed\":" << result.steps_executed
                << ",\"stop_reason\":\"" << stop_reason_name(result.stop_reason) << "\""
                << ",\"queue_wait_ns\":" << metrics.queue_wait_ns << ",\"service_ns\":" << metrics.service_ns
                << ",\"wall_ns\":" << metrics.wall_ns << ",\"chunks\":" << metrics.chunks
                << ",\"emitted_tokens\":" << metrics.emitted_tokens
                << (serving_prefill && native_targets[i] ? [&] {
                    const auto observed=native_targets[i]->prefill_stats();
                    return std::string(",\"native_prefill\":{\"chunks\":")+std::to_string(observed.chunks)+
                        ",\"consumed_rows\":"+std::to_string(observed.consumed_rows)+
                        ",\"state_only_chunks\":"+std::to_string(observed.state_only_chunks)+
                        ",\"terminal_chunks\":"+std::to_string(observed.terminal_chunks)+
                        ",\"r2_dispatches\":"+std::to_string(observed.r2_dispatches)+
                        ",\"shared_rows_dispatches\":"+std::to_string(observed.shared_rows_dispatches)+
                        ",\"gdn_rows_dispatches\":"+std::to_string(observed.gdn_rows_dispatches)+
                        ",\"omitted_lm_heads\":"+std::to_string(observed.omitted_lm_heads)+
                        ",\"saturated\":"+(observed.saturated?"true":"false")+"}";
                }() : std::string{})
                << (options.ple_cache_rows ? ",\"ple_row_cache\":" + ple_row_cache_json(owners[i]->ple_cache_stats()) : std::string{})
                << ",\"inter_token_intervals\":" << metrics.inter_token_intervals
                << ",\"inter_token_ns_total\":" << metrics.inter_token_ns_total
                << ",\"inter_token_ns_max\":" << metrics.inter_token_ns_max
                << ",\"time_to_first_token_ns\":";
            if (metrics.time_to_first_token_ns) std::cout << *metrics.time_to_first_token_ns; else std::cout << "null";
            std::cout << ",\"inter_token_log2_ns_histogram\":[";
            for (std::size_t bin = 0; bin < metrics.inter_token_ns_histogram.size(); ++bin) {
                if (bin) std::cout << ',';
                std::cout << metrics.inter_token_ns_histogram[bin];
            }
            std::cout << "],\"metrics_saturated\":" << (metrics.saturated ? "true" : "false") << ",\"prefix_tokens\":[";
            for (std::size_t j = 0; j < prefixes[i].size(); ++j) { if (j) std::cout << ','; std::cout << prefixes[i][j]; }
            std::cout << "],\"generated_tokens\":[";
            for (std::size_t j = 0; j < result.generated_tokens.size(); ++j) { if (j) std::cout << ','; std::cout << result.generated_tokens[j]; }
            std::cout << "]}";
            if (options.host_dispatch_telemetry) {
                std::cerr << "{\"serving_request_id\":" << i + 1U << ",\"scope\":\"factory-and-worker-calls-before-owner-teardown\",\"telemetry\":";
                strix::write_host_dispatch_telemetry(std::cerr, owners[i]->counters); std::cerr << "}\n";
            }
        }
        std::cout << "],\n\"all_completed\":" << (passed ? "true" : "false") << "\n}\n";
        return passed ? 0 : 2;
    }
    setup_phase.reset();
    if (host_telemetry.enabled) setup_phase.emplace(strix::ExecutionPhase::prepare);
    auto allocator_owner = std::make_shared<strix::HipRuntimeAllocator>(
        strix::HipHostAllocationMode::pageable_aligned);
    auto& allocator = *allocator_owner;
    auto memory_owner = std::make_shared<strix::RuntimeExecutionMemoryOwner>(
        strix::RuntimeExecutionMemoryOwner::allocate(compiled, state_layout, allocator_owner, generations));
    auto& memory = *memory_owner;
    // These snapshots pin actual allocator-issued live pools, independently of
    // optional future request workspace estimates. No per-token inventory work.
    std::optional<strix::UmaOwnerInventory> runtime_memory_inventory,base_runtime_owner_inventory;
    if(options.execution_inventory || options.known_prefix_prefill_rows) {
        if(!model_owner_inventory) throw std::logic_error("base runtime inventory lacks its actual model owner");
        runtime_memory_inventory.emplace(strix::hip_runtime_memory_owner_inventory(memory_owner));
        auto views=model_owner_inventory->pinned_views;
        views.insert(views.end(),runtime_memory_inventory->pinned_views.begin(),runtime_memory_inventory->pinned_views.end());
        base_runtime_owner_inventory.emplace(strix::build_uma_owner_inventory(views));
    }
    {
        auto correlation = strix::current_execution_correlation();
        correlation.execution_identity = compiled.execution_identity;
        correlation.session_epoch = memory.state().device_allocation_generation;
        correlation.phase = strix::ExecutionPhase::prepare;
        strix::set_execution_correlation(correlation);
    }
    strix::q8_projection_organization::validate_output_steps(
        q8_organization_plan, model, compiled, tensor_access, memory.scratch(), model_residency.binding());
    strix::hc_projection_prefetch::validate_runtime(
        hc_next_tile_plan, model, manifest, compiled, tensor_access, memory.scratch(), model_residency.binding());

    strix::initialize_hip_fresh_runtime_memory(
        memory.scratch(),
        memory.state());

    strix::FileMappedPleLookupBackend
        mapped_ple_lookup;
    // Only the explicit MTP request needs an independently retained PLE parent.
    auto mtp_mapped_ple_owner = options.mtp.enabled
        ? std::make_shared<strix::FileMappedPleLookupBackend>() : std::shared_ptr<strix::FileMappedPleLookupBackend>{};
    auto ple_turn = ple_reclaimer && options.mtp.enabled ? ple_reclaimer->begin_turn() : nullptr;
    if (mtp_mapped_ple_owner) mtp_mapped_ple_owner->bind_residency(ple_turn);
    MappedModelIoSource
        mapped_model_io(
            mapped);
    const auto planned_ple_policy = requested_ple_policy;
    auto retained_ple_source = options.ple_cache_rows || options.known_prefix_prefill_rows || options.mtp.enabled
        ? std::make_shared<MappedModelIoSource>(mapped_owner) : std::shared_ptr<MappedModelIoSource>{};
    std::optional<strix::PlannedPleLookupBackend> local_planned_ple_lookup;
    std::shared_ptr<strix::PlannedPleLookupBackend> prefill_ple_owner;
    if (options.known_prefix_prefill_rows || options.mtp.enabled)
        prefill_ple_owner = std::make_shared<strix::PlannedPleLookupBackend>(*retained_ple_source,
            planned_ple_policy, retained_ple_source);
    else local_planned_ple_lookup.emplace(retained_ple_source ? *retained_ple_source : mapped_model_io,
        planned_ple_policy, retained_ple_source);
    auto& planned_ple_lookup = prefill_ple_owner ? *prefill_ple_owner : *local_planned_ple_lookup;
    const auto ple_cache_completed_receipt = [&]() -> std::string {
        return options.ple_cache_rows ? "  \"ple_row_cache\":" + ple_row_cache_json(planned_ple_lookup.cache_stats()) + ",\n" : std::string{};
    };
    strix::PleLookupBackend* ple_lookup =
        options.planned_ple_io
            ? static_cast<strix::PleLookupBackend*>(
                  &planned_ple_lookup)
            : static_cast<strix::PleLookupBackend*>(
                  mtp_mapped_ple_owner ? mtp_mapped_ple_owner.get() : &mapped_ple_lookup);

    const auto ple_hash_contract =
        strix::build_ple_hash_contract(
            model);
    strix::PleLookupPrefetchState
        ple_prefetch;

    strix::ExecutionRuntimeResources
        resources;
    resources.scratch =
        &memory.scratch();
    resources.model_manifest =
        &manifest;
    resources.model_weights =
        &model_residency.binding();
    resources.tensor_access =
        &tensor_access;
    resources.session_state_layout =
        &state_layout;
    resources.session_state_memory =
        &memory.state();
    resources.ple_lookup =
        ple_lookup;
    resources.ple_hash_contract =
        &ple_hash_contract;
    resources.ple_prefetch =
        &ple_prefetch;

    auto experiment_owner = std::make_shared<HipDecodeExperimentResources>(
        mapped_owner, model_owner, allocator_owner, memory_owner, manifest, tensor_access, state_layout);
    if (graph_mode) resources = experiment_owner->resources;
    if(wave_router_admission) {
        resources.model_manifest = &wave_router_admission->manifest();
        experiment_owner->resources.model_manifest = &wave_router_admission->manifest();
    }
    // Split prefill: the sibling prefill execution runs on the decode serial arena (identical
    // layout, never concurrent) under its own execution identity, with the same model, state
    // and PLE bindings. No additional device memory.
    std::optional<strix::ExecutionMemoryBinding> prefill_scratch;
    strix::ExecutionRuntimeResources prefill_resources;
    if (compiled_prefill) {
        if (graph_mode) throw std::logic_error("split prefill profile reached graph execution");
        prefill_scratch = strix::sibling_scratch_binding(*compiled_prefill, compiled, memory.scratch());
        prefill_resources = resources;
        prefill_resources.scratch = &*prefill_scratch;
    }
    std::shared_ptr<strix::integer_projection::HipIntegerRuntimeAllocationOwner> arena_pin, weight_pin;
    std::shared_ptr<strix::integer_projection::HipIntegerProjectionRuntimeBundle> integer_bundle;
    if (integer_config) {
        const auto head = std::find_if(manifest.tensors.begin(), manifest.tensors.end(),
            [](const auto& tensor) { return tensor.role == strix::TensorRole::output_head; });
        if (head == manifest.tensors.end()) throw std::logic_error("integer output head disappeared");
        const auto index = static_cast<std::uint32_t>(head - manifest.tensors.begin());
        const auto* binding = strix::find_runtime_tensor_binding(model_residency.binding(), index);
        if (!binding) throw std::logic_error("integer output head has no actual allocation binding");
        arena_pin = strix::integer_projection::HipIntegerRuntimeAllocationOwner::pin(
            integer_config->device, memory.scratch().arena_base,
            memory.scratch().allocation_generation, experiment_owner);
        weight_pin = strix::integer_projection::HipIntegerRuntimeAllocationOwner::pin(
            integer_config->device, binding->address, binding->allocation_generation, experiment_owner);
        const strix::integer_projection::RuntimeOwners pins{arena_pin, weight_pin};
        const auto plan = strix::integer_projection::RuntimePlan::bind(compiled, resources, *integer_config, pins);
        integer_bundle = std::make_shared<strix::integer_projection::HipIntegerProjectionRuntimeBundle>(
            plan, pins, hipStreamLegacy);
    }
    std::shared_ptr<HipDeferredMoeStatus> deferred_moe;
    if(options.deferred_moe_status && !options.mtp.enabled && !target_input_export) {
        const auto deferred_status_generation=generations.next();
        const auto deferred_shadow_generation=generations.next();
        deferred_moe=std::make_shared<HipDeferredMoeStatus>(resources,compiled.execution_identity,
            deferred_status_generation,deferred_shadow_generation,experiment_owner,original_work_device);
    }
    auto launcher_catalog = std::make_shared<strix::HipLauncherCatalog>(model, catalog_config, integer_bundle);
    std::shared_ptr<strix::HipGdnIslands> gdn_islands;
    if (options.gdn_islands) {
        if (!deferred_moe) throw std::logic_error("GDN islands are missing their D1 owner");
        gdn_islands = strix::HipGdnIslands::create(compiled, launcher_catalog->gdn_config(),
            deferred_moe->resources(), deferred_moe->scope(), model_owner,
            std::static_pointer_cast<const void>(deferred_moe), original_work_device);
    }

    strix::ExecutionLauncherRegistry launchers;
    launcher_catalog->register_for_compiled_execution(compiled, launchers,
        gdn_islands ? &deferred_moe->resources() : &resources, gdn_islands);
    if (launcher_catalog->audit(compiled, launchers).missing_steps)
        throw std::logic_error("HIP launcher catalog is incomplete for actual compiled execution");
    const auto launch_plan = strix::compile_launch_plan(compiled, launchers);
    std::string uma_setup_observation_receipt,base_runtime_owner_inventory_receipt;
    if (options.execution_inventory) {
        const auto observed = strix::observe_linux_uma_memory();
        uma_setup_observation_receipt = "  \"uma_setup_observation_boundary\":\"after-base-runtime-D1-catalog-before-optional-request-setup\",\n"
            "  \"uma_setup_observation\":" +
            strix::linux_uma_observation_json(observed) + ",\n";
        if (model_owner_inventory)
            model_owner_inventory = strix::build_uma_owner_inventory(model_owner_inventory->pinned_views, observed.memory);
        if(!base_runtime_owner_inventory || !runtime_memory_inventory)
            throw std::logic_error("runtime inventory snapshots are missing actual retained pools");
        base_runtime_owner_inventory = strix::build_uma_owner_inventory(base_runtime_owner_inventory->pinned_views,observed.memory);
        // This remains the original pre-optional-request snapshot. MTP owns
        // its D1 later; its completion receipt reports that private owner under
        // a distinct boundary rather than changing this historical snapshot.
        base_runtime_owner_inventory_receipt = strix::decode_receipt::runtime_owner_inventory_receipt(
            *base_runtime_owner_inventory,*runtime_memory_inventory,static_cast<bool>(deferred_moe));
    }
    const auto execution_inventory_receipt = options.execution_inventory
        ? strix::decode_receipt::execution_inventory_receipt(compiled, launch_plan, kernel_registry) +
            strix::decode_receipt::hip_runtime_environment_receipt() + model_owner_inventory_receipt + base_runtime_owner_inventory_receipt + uma_setup_observation_receipt
        : std::string{};
    std::unique_ptr<strix::HipEventQueueTrace> device_queue_trace;
    std::optional<strix::ExecutionTelemetryBinding> device_queue_binding;
    strix::ExecutionTelemetry* active_observer = host_telemetry.enabled ? &host_telemetry.counters : nullptr;
    if (options.device_queue_trace) {
        if (!deferred_moe) throw std::logic_error("device queue trace is missing its owned D1 stream");
        device_queue_trace = std::make_unique<strix::HipEventQueueTrace>(host_telemetry.counters,
            std::static_pointer_cast<const void>(deferred_moe), original_work_device);
        active_observer = device_queue_trace.get();
        device_queue_binding.emplace(active_observer, strix::current_execution_correlation());
    }
    // Export only after the existing execution completion; this adds no fence.
    // Binding dies before trace, while trace retains D1/main allocation owners.
    const auto emit_device_queue_trace = [&] {
        if (!device_queue_trace) return;
        device_queue_trace->write_after_completion(std::cerr);
        std::cerr << '\n';
        device_queue_trace->require_healthy_before_commit();
    };
    strix::DirectLaunchDispatcher direct_dispatcher(launch_plan, active_observer);
    // Serial MTP prompt rows under a split prefill profile: launchers of the sibling prefill execution
    // (the tiled MTP prompt runs only its tiled blocks on the prefill profile, through their owners).
    std::shared_ptr<strix::HipLauncherCatalog> prefill_launcher_catalog;
    strix::ExecutionLauncherRegistry prefill_launchers;
    std::optional<strix::CompiledLaunchPlan> prefill_launch_plan;
    std::optional<strix::DirectLaunchDispatcher> prefill_dispatcher;
    if (compiled_prefill && options.mtp.enabled && !strix::decode_cli::mtp_tiled_prompt(options)) {
        prefill_launcher_catalog = std::make_shared<strix::HipLauncherCatalog>(model, prefill_catalog_config);
        prefill_launcher_catalog->register_for_compiled_execution(*compiled_prefill, prefill_launchers, &prefill_resources);
        if (prefill_launcher_catalog->audit(*compiled_prefill, prefill_launchers).missing_steps)
            throw std::logic_error("HIP launcher catalog is incomplete for the prefill serial execution");
        prefill_launch_plan.emplace(strix::compile_launch_plan(*compiled_prefill, prefill_launchers));
        prefill_dispatcher.emplace(*prefill_launch_plan, active_observer);
    }
    std::unique_ptr<strix::HipMoeOverlapDispatcher> moe_overlap;
    if (overlap_arena) {
        if (!deferred_moe) throw std::logic_error("MoE overlap is missing its main stream owner");
        moe_overlap = std::make_unique<strix::HipMoeOverlapDispatcher>(compiled, *overlap_arena,
            launch_plan, deferred_moe->resources(), deferred_moe->scope().stream,
            std::static_pointer_cast<const void>(deferred_moe),
            std::static_pointer_cast<const void>(launcher_catalog), original_work_device,
            active_observer);
    }
    strix::EagerExecutionDispatcher& native_dispatcher = moe_overlap
        ? static_cast<strix::EagerExecutionDispatcher&>(*moe_overlap)
        : static_cast<strix::EagerExecutionDispatcher&>(direct_dispatcher);
    std::unique_ptr<strix::GenerationOutputDispatcher> output_dispatcher;
    if (strix::decode_cli::needs_generation_output_dispatcher(options))
        output_dispatcher = std::make_unique<strix::GenerationOutputDispatcher>(native_dispatcher, compiled);
    strix::EagerExecutionDispatcher& dispatcher = output_dispatcher
        ? static_cast<strix::EagerExecutionDispatcher&>(*output_dispatcher)
        : native_dispatcher;
    std::unique_ptr<strix::HipSerialGraph> graph;
    if (graph_mode) {
        strix::GraphMemoryRequest graph_memory;
        graph_memory.workspace_bytes_per_instance = options.execution.graph_workspace_bytes;
        graph_memory.executable_bytes_per_instance = options.execution.graph_executable_bytes;
        graph_memory.capture_peak_bytes = options.execution.graph_capture_peak_bytes;
        graph_memory.budget_bytes = options.execution.graph_budget_bytes;
        graph = std::make_unique<strix::HipSerialGraph>(compiled, launcher_catalog, experiment_owner, graph_memory);
    }

    const auto identities =
        strix::build_runtime_execution_identities(
            compiled.execution_identity,
            compiled.arena,
            resources);

    strix::SerialSessionContext
        session;
    session.execution_identity =
        compiled.execution_identity;
    session.state_layout_identity =
        state_layout.layout_identity;
    session.runtime_compatibility_identity =
        identities.compatibility_identity;
    session.state.valid =
        all_persistent_state();

    const auto& lm_io =
        compiled.io.steps.back();
    if (lm_io.op !=
            strix::SerialOp::lm_head ||
        lm_io.writes.size() != 1) {
        throw std::logic_error(
            "compiled execution does not end in one logits write");
    }

    // Known-prefix windows and the tiled prefill (also the MTP prompt with --known-prefix-window):
    // plans, arenas, native owners and backends, and the optional prefix state cache.
    std::optional<strix::CompiledKnownPrefixWindow> known_prefix_plan;
    std::shared_ptr<strix::CompiledKnownPrefixPrefill> known_prefill_plan;
    // Optional smaller tiled block for the remainder (--known-prefix-tail-rows).
    std::vector<std::shared_ptr<strix::CompiledKnownPrefixPrefill>> known_tail_plans;
    std::vector<std::shared_ptr<KnownPromptMemory>> known_tail_memories;
    std::vector<std::shared_ptr<strix::HipKnownPrefixWindow>> known_tail_owners;
    std::vector<std::unique_ptr<strix::HipKnownPromptBackend>> known_tail_backends;
    // --mtp-copy-wide-rows with a split prefill numerical profile: the wide copy verify owners run decode rows, so
    // they are compiled on the DECODE execution (decode catalog and resources), apart from the prompt tails, which
    // keep the prefill profile. Empty with one profile (the tails themselves verify the wide cycles).
    std::vector<std::shared_ptr<strix::CompiledKnownPrefixPrefill>> known_wide_plans;
    std::vector<std::shared_ptr<KnownPromptMemory>> known_wide_memories;
    std::vector<std::shared_ptr<strix::HipKnownPrefixWindow>> known_wide_owners;
    std::optional<strix::UmaReservationEstimate> known_prefill_reservation;
    std::uint64_t known_prefill_runtime_payload = 0;
    std::shared_ptr<KnownPromptMemory> known_prefix_memory;
    std::shared_ptr<strix::HipKnownPrefixWindow> known_prefix_owner;
    std::unique_ptr<strix::HipKnownPromptBackend> known_prefix_backend;
    std::optional<strix::KnownPromptGreedyResult> known_prefix_result;
    std::optional<strix::KnownPromptChunkProgress> known_prefix_progress;
    // Block-boundary prefix state cache (--prefix-cache-host-bytes): store and live-session view.
    std::shared_ptr<strix::PrefixStateCache> prefix_store;
    std::unique_ptr<strix::decode_cli::HipPrefixStateCacheSession> prefix_state;
    // --rtx-state-import / STRIX_RTX_EXPORT_STATE without a prefix cache: the same component registry over no store
    // (no capture, no lookup, no disk tier).
    std::unique_ptr<strix::decode_cli::HipPrefixStateCacheSession> rtx_state_session;
    strix::ModelContentIdentity prefix_cache_identity;
    std::uint64_t prefix_cache_every = 0;
    std::function<void(bool)> enable_prefix_store;
    std::function<void()> prepare_prefix_state;
    // OFF commitments are fixed before a cache budget is fitted. Until all
    // native owners are present, reserve their declared setup upper bounds.
    std::optional<std::uint64_t> cache_future_commitment, cache_ready_commitment;
    std::optional<std::uint64_t> cache_slot_retained, cache_slot_native;
    std::uint64_t cache_prompt_transient=0;
    const auto cache_fit_observation = [&](bool ready) {
        using namespace strix::decode_cli;
        PrefillCapabilities caps;
        const auto observed=observe_prefill_memory(caps,true);
        CacheLateFit fit;
        fit.observation_id=observed.observation_id;
        const auto now=static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
        fit.age_ns=now>=observed.end_steady_ns?now-observed.end_steady_ns:UINT64_MAX;
        fit.U=observed.usable_ceiling; fit.A=observed.cache_available; fit.L=observed.resident;
        fit.limits_consistent=observed.cache_limits_known;
        fit.cgroup_ancestry_unproven=!observed.cache_limits_known;
        fit.aliasing_known=base_runtime_owner_inventory && base_runtime_owner_inventory->unique_backing_reserved_bytes.has_value();
        if(fit.L && fit.aliasing_known) fit.L=std::max(*fit.L,*base_runtime_owner_inventory->unique_backing_reserved_bytes);
        intersect_cache_capacity(fit,options.known_prefill_uma_budget);
        fit.F=cache_unpaid_commitment(ready,cache_future_commitment,cache_ready_commitment);
        fit.headroom=options.known_prefill_host_headroom;
        fit.context_off=fit.context=options.context_tokens;
        fit.slots_off=fit.slots=options.http_batch?options.http_batch:1;
        if(options.cache_activation->source=="explicit") fit.explicit_budget=options.prefix_cache_host_bytes;
        // Metadata/certificates charged by the payload ledger stay in B.
        // E below bounds token copies, index/control nodes and lookup scratch.
        fit.S=0; fit.device_unpaid=fit.F;
        if(observed.gtt_limit && observed.gtt_used && *observed.gtt_used<=*observed.gtt_limit)
            fit.device_free=*observed.gtt_limit-*observed.gtt_used;
        try {
            auto checkpoint=prefill_checked_add(state_layout.device_total_bytes,state_layout.host_total_bytes);
            if(options.mtp.enabled) {
                const auto mul=[](std::uint64_t a,std::uint64_t b) {
                    if(b && a>UINT64_MAX/b) throw std::overflow_error("cache checkpoint overflow");
                    return a*b;
                };
                const auto predictor_row=prefill_checked_add(mul(mul(model.kv_heads,model.head_dim),4),mul(model.indexer_head_dim,2));
                checkpoint=prefill_checked_add(checkpoint,prefill_checked_add(16U<<20,mul(options.context_tokens,predictor_row)));
            }
            std::uint64_t rows=0,fixed=0;
            for(const auto& allocation:state_layout.allocations) {
                if(allocation.component==strix::SessionStateComponent::attention_kv || allocation.component==strix::SessionStateComponent::sparse_index) {
                    if(!state_layout.context_tokens || allocation.bytes%state_layout.context_tokens) { fit.aliasing_known=false; break; }
                    rows=prefill_checked_add(rows,allocation.bytes/state_layout.context_tokens);
                } else fixed=prefill_checked_add(fixed,prefill_checked_add(allocation.bytes,255)&~UINT64_C(255));
            }
            if(options.mtp.enabled) {
                const auto predictor_row=prefill_checked_mul(model.mtp_full_attention_layers,
                    prefill_checked_add(prefill_checked_mul(prefill_checked_mul(model.kv_heads,model.head_dim),4),prefill_checked_mul(model.indexer_head_dim,2)));
                // One process-wide K=1 slab, never multiplied by the slot count.
                fit.S=prefill_checked_add(qwen_batch_staging_cap(options.prefix_cache_staging_bytes,rows,fixed,predictor_row,
                    prefill_checked_mul(prefill_checked_mul(model.hidden_size,model.hc_count),2),options.known_prefix_prefill_rows),1U<<20);
            }
            const auto cap=options.cache_activation->source=="explicit"?options.prefix_cache_host_bytes:UINT64_C(4)<<30;
            const auto unit=prefix_cache_unit(options);
            const auto segment=prefill_checked_mul(rows,unit);
            if(!fixed || !segment || !unit) fit.E.reset();
            else {
                const auto segments=prefill_checked_add(cap/segment,1);
                const auto checkpoints=prefill_checked_add(cap/fixed,1);
                // 4 KiB per object/control node, 16 bounded conversation keys
                // (128 bytes each), geometric vector growth, plus one request's
                // whole-context lookup token copy. Encoded metadata is in B.
                const auto per_segment=prefill_checked_add(4096,prefill_checked_mul(unit,8));
                const auto per_checkpoint=prefill_checked_add(4096+16*256,
                    prefill_checked_mul(options.context_tokens/unit+1,32));
                fit.E=prefill_checked_add(prefill_checked_mul(segments,per_segment),
                    prefill_checked_add(prefill_checked_mul(checkpoints,per_checkpoint),prefill_checked_mul(options.context_tokens,8)));
                if(!options.prefix_cache_dir.empty()) fit.E=prefill_checked_add(*fit.E,prefill_checked_mul(checkpoint,2));
            }
            const auto tile=options.known_prefix_prefill_rows;
            const auto cadence=prefix_capture_cadence(tile,options.prefix_cache_every);
            const auto first=minimum_prefix_capture_boundary(tile,options.context_tokens,cadence);
            auto checkpoint_rows=rows, checkpoint_fixed=fixed;
            if(options.mtp.enabled) {
                checkpoint_rows=prefill_checked_add(checkpoint_rows,prefill_checked_mul(model.mtp_full_attention_layers,
                    prefill_checked_add(prefill_checked_mul(prefill_checked_mul(model.kv_heads,model.head_dim),4),prefill_checked_mul(model.indexer_head_dim,2))));
                checkpoint_fixed=prefill_checked_add(checkpoint_fixed,16U<<20);
            }
            fit.checkpoint=first && first<options.context_tokens?
                prefill_checked_add(checkpoint_fixed,prefill_checked_mul(checkpoint_rows,first)):0;
        } catch(const std::overflow_error&) { fit.F=UINT64_MAX; }
        if(ready && prefix_store) {
            const auto live=prefix_store->counters();
            fit.explicit_budget=options.cache_activation->B; // recheck the already fitted cap
            fit.resident_payload=live.host_bytes; fit.resident_staging=std::max(live.staging_bytes,live.pool_staging_bytes);
        }
        return fit;
    };
    const auto apply_cache_fit = [&](bool ready) {
        using namespace strix::decode_cli;
        const auto fit=cache_fit_observation(ready);
        return prepare_automatic_cache_fit(options,fit,ready);
    };
    std::uint64_t known_prefix_union_bytes = 0; // routed-union scratch of the tiled prefill owners (0: none)
    // Prompt groups of the batched tiled MTP prompt (STRIX_PROMPT_BATCH_ROWS above T): a tiled owner of that many rows
    // with its own arena, on which the blocks of several requests run as one segmented prefill.
    std::shared_ptr<strix::CompiledKnownPrefixPrefill> known_group_plan;
    std::shared_ptr<KnownPromptMemory> known_group_memory;
    std::shared_ptr<strix::HipKnownPrefixWindow> known_group_owner;
    std::unique_ptr<strix::HipKnownPromptBackend> known_group_backend;
    if (options.known_prefix_window || options.prompt_lookup) {
        if (options.known_prefix_prefill_rows) {
            // The tiled prefill derives from the prefill serial execution (its dense projection
            // kernel ids select the prefill verify ids and launchers).
            known_prefill_plan = std::make_shared<strix::CompiledKnownPrefixPrefill>(strix::compile_known_prefix_prefill(model,
                compiled_prefill ? *compiled_prefill : compiled, manifest, tensor_access,
                {options.known_prefix_prefill_rows, true, true, true, options.known_prefix_expert_union,
                 options.known_prefix_output_policy, options.known_prefill_workspace_budget, options.known_prefill_metadata_budget / 2U,
                 options.known_window_router_rows || options.known_prefix_full_rows,
                 options.known_window_router_rows || options.known_prefix_full_rows,
                 options.known_prefix_full_rows, options.known_prefix_full_rows, options.known_prefix_full_rows,
                 options.teacher_forced_score, strix::decode_cli::compact_prompt_memory(options), true}));
            // --mtp-copy-wide-rows N[,M]: each listed size's tail owner also verifies wide copy cycles, so it computes
            // every row's head (all-rows policy; its prompt blocks that need no logits still run without heads).
            for (const auto tail_rows : options.known_prefix_tail_rows) {
                const bool wide_tail = !compiled_prefill &&
                    std::find(options.mtp.copy_wide_rows.begin(), options.mtp.copy_wide_rows.end(), tail_rows) !=
                    options.mtp.copy_wide_rows.end();
                known_tail_plans.push_back(std::make_shared<strix::CompiledKnownPrefixPrefill>(strix::compile_known_prefix_prefill(model,
                    compiled_prefill ? *compiled_prefill : compiled, manifest, tensor_access,
                    {tail_rows, true, true, true, options.known_prefix_expert_union,
                     wide_tail ? strix::KnownWindowOutputPolicy::all_rows : options.known_prefix_output_policy,
                     options.known_prefill_workspace_budget, options.known_prefill_metadata_budget / 2U,
                     true, true, true, true, true,
                     // and its multi-row head (eight-row packs, the one-row head's leaf and reduction tree per row)
                     wide_tail, strix::decode_cli::compact_prompt_memory(options), true})));
            }
            if (compiled_prefill)
                for (const auto wide_rows : options.mtp.copy_wide_rows)
                    known_wide_plans.push_back(std::make_shared<strix::CompiledKnownPrefixPrefill>(strix::compile_known_prefix_prefill(model,
                        compiled, manifest, tensor_access,
                        {wide_rows, true, true, true, options.known_prefix_expert_union,
                         strix::KnownWindowOutputPolicy::all_rows,
                         options.known_prefill_workspace_budget, options.known_prefill_metadata_budget / 2U,
                         true, true, true, true, true, true})));
            known_prefix_plan = known_prefill_plan->tile8; // selection ledger, never a relabelled T forward
        } else {
            const bool full = options.known_prefix_full_rows;
            const bool rows_ops = full || options.known_window_router_rows;
            known_prefix_plan = strix::compile_known_prefix_window(model, compiled, manifest, tensor_access,
                {true, true, true, options.known_prefix_window ? options.known_prefix_rows : 2U, options.known_prefix_expert_union,
                 options.known_prefix_output_policy, rows_ops, rows_ops, full, full, false, full, full});
            if (full && (known_prefix_plan->gdn_deferred_rows_steps != model.gdn_layers || !known_prefix_plan->ple_rows_steps))
                throw std::invalid_argument("known-prefix full rows did not admit deferred GDN and PLE rows for every layer");
        }
        if (!strix::known_window_has_promoted_steps(*known_prefix_plan))
            throw std::invalid_argument("known prompt has no admitted known-window operators");
        if (model.vocab_size > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("known-prompt vocabulary exceeds uint32");
        std::uint64_t union_bytes = 0;
        if (options.known_prefix_expert_union && known_prefix_plan->routed_expert_union_steps) {
            if (model.num_experts > UINT32_MAX || model.num_experts_per_token > UINT32_MAX)
                throw std::invalid_argument("known-prefix expert union geometry exceeds uint32");
            union_bytes = strix::q51_known_rows::required_scratch_bytes(known_prefill_plan ? 8U : known_prefix_plan->logical_rows,
                static_cast<std::uint32_t>(model.num_experts), static_cast<std::uint32_t>(model.num_experts_per_token));
            if (!union_bytes || union_bytes > 65536U)
                throw std::invalid_argument("known-prefix expert union exceeds separate 64KiB setup budget");
        }
        known_prefix_union_bytes = union_bytes;
        const auto arena_generation = generations.next();
        const auto status_generation = generations.next();
        if (known_prefill_plan) {
            const auto add = [](std::uint64_t a, std::uint64_t b) {
                if (b > UINT64_MAX - a) throw std::overflow_error("known prefill reservation overflow");
                return a + b;
            };
            if (!base_runtime_owner_inventory || !prefill_ple_owner || !retained_ple_source || (!deferred_moe && !options.mtp.enabled))
                throw std::logic_error("known prefill lacks actual retained setup owners");
            auto views = base_runtime_owner_inventory->pinned_views;
            std::uint64_t view_id = 0;
            for (const auto& view : views) view_id = std::max(view_id, view.view_identity);
            const auto reserve = [&](std::string category, std::uint64_t bytes, std::shared_ptr<const void> owner) {
                if (!bytes) return;
                if (view_id == UINT64_MAX) throw std::overflow_error("known prefill inventory IDs exhausted");
                // No allocator-issued backing key is exposed for these views.
                // Count declared storage conservatively; do not invent alias/PFN proof.
                strix::UmaOwnerView view;
                view.view_identity = ++view_id; view.category = std::move(category);
                view.allocation_bytes = bytes; view.view_bytes = bytes;
                view.allocation_pin = std::move(owner); views.push_back(std::move(view));
            };
            const std::shared_ptr<const void> d1_owner = deferred_moe
                ? std::static_pointer_cast<const void>(deferred_moe) : std::static_pointer_cast<const void>(experiment_owner);
            reserve("D1-shadow-host-state", add(state_layout.host_total_bytes, state_layout.host_alignment - 1U), d1_owner);
            reserve("serial-D1-record", 8U, d1_owner);
            reserve("PLE-row-cache", requested_ple_cache_bytes, prefill_ple_owner);
            const auto multiply = [](std::uint64_t a, std::uint64_t b) {
                if (b && a > UINT64_MAX / b) throw std::overflow_error("known prefill runtime payload overflow");
                return a * b;
            };
            const auto map_bytes = multiply(known_prefill_plan->logit_row_indices.size(), sizeof(std::uint32_t));
            reserve("prefill-caller-plan-vector-payload", add(known_prefill_plan->host_metadata_payload_bytes, map_bytes), known_prefill_plan);
            // Result payloads coexist with both plans and the full arena.
            // Two terminal host rows cover a retained cursor frontier plus
            // the next adapter return buffer; allocator capacity overhead is headroom.
            known_prefill_runtime_payload = add(multiply(multiply(known_prefill_plan->logical_rows,
                compiled.execution.steps.size()), sizeof(std::uint32_t)), map_bytes);
            known_prefill_runtime_payload = add(known_prefill_runtime_payload,
                multiply(known_prefill_plan->logit_row_indices.size(), sizeof(strix::ExecutionBufferAddress)));
            if (known_prefill_plan->output_policy != strix::KnownWindowOutputPolicy::state_only)
                known_prefill_runtime_payload = add(known_prefill_runtime_payload, multiply(model.vocab_size, 2U * sizeof(float)));
            // Teacher-forced scoring: pinned staging of one block's T full rows (the spool streams them out).
            if (options.teacher_forced_score)
                known_prefill_runtime_payload = add(known_prefill_runtime_payload,
                    multiply(multiply(known_prefill_plan->logical_rows, model.vocab_size), sizeof(float)));
            auto growth = add(known_prefill_plan->arena.physical.total_bytes, add(8U, union_bytes));
            growth = add(growth, add(known_prefill_plan->host_metadata_payload_bytes, map_bytes)); // native owner's second copy
            growth = add(growth, known_prefill_plan->qsa_score_ring_workspace_bytes);
            if (options.http_serve && options.rotary.mode() == strix::RotaryMode::native)
                growth = add(growth, strix::decode_cli::qwen_prefill_external_memory(options,model).host);
            growth = add(growth, options.known_prefill_host_headroom); // explicit allocator/driver/fixed-plan allowance
            for (const auto& tail : known_tail_plans) { // arena, completion record and metadata copy of each tail owner
                growth = add(growth, add(tail->arena.physical.total_bytes, add(8U, union_bytes)));
                growth = add(growth, tail->host_metadata_payload_bytes);
                growth = add(growth, tail->qsa_score_ring_workspace_bytes);
            }
            for (const auto& wide : known_wide_plans) { // same for each decode-profile wide owner (split profile)
                growth = add(growth, add(wide->arena.physical.total_bytes, add(8U, union_bytes)));
                growth = add(growth, wide->host_metadata_payload_bytes);
                growth = add(growth, wide->qsa_score_ring_workspace_bytes);
            }
            // Batched MTP slots (cohort requests on the tiled prompt, --http-batch N): each further slot's block
            // and tail owners share the arenas above but hold their own metadata copy and completion records
            // (about 286 MB per slot at T = 8192).
            const std::uint64_t extra_slots = !strix::decode_cli::mtp_tiled_prompt(options) ? 0U
                : !options.mtp.cohort.empty() ? options.mtp.cohort.size()
                : 0U; // HTTP capacity is measured after the primary stack
            for (std::uint64_t slot = 0; slot < extra_slots; ++slot) {
                growth = add(growth, add(add(known_prefill_plan->host_metadata_payload_bytes, map_bytes), 8U));
                for (const auto& tail : known_tail_plans) growth = add(growth, add(tail->host_metadata_payload_bytes, 8U));
            }
            auto rolling_metadata = ple_reclaimer && ple_reclaimer->rolling() ? ple_reclaimer->owner_bytes() : 0U;
            strix::decode_cli::PrefillReservationGrowth mtp_growth;
            if (ple_turn) {
                const auto& m = options.mtp;
                mtp_growth = strix::decode_cli::mtp_prefill_reservation({m.sidecar_retained_bytes, m.runtime_bytes,
                    m.target_capture_bytes, m.replay_host_bytes, m.golden_max_bytes, m.window_arena_bytes,
                    m.window_capture_bytes, m.window_checkpoint_bytes, m.window_total_bytes,
                    m.window_rows ? static_cast<std::uint32_t>(m.window_rows_set.empty() ? 1U : m.window_rows_set.size()) : 0U,
                    m.window_prefix});
            }
            if(yarn && options.mtp.enabled) {
                // Same closed owner plan as inspection/pre-device admission.
                // No credit or source-cache quota is appended here a second time.
                const auto requirement=strix::decode_cli::compile_prompt_requirements(options,model,manifest,mapped.bytes());
                // Sidecar setup is complete; post-setup phases take their maximum,
                // with each shared M1 owner counted once in this growth estimate.
                mtp_growth={requirement.mtp_post_setup_peak,0};
                for(const auto* name:{"mtp_m1_keys","mtp_m1_scratch_metadata"})
                    if(requirement.owners.contains(name)) mtp_growth.retained=add(mtp_growth.retained,requirement.owners.at(name));
            }
            const auto prefill_live = strix::build_uma_owner_inventory(views);
            if(strix::decode_cli::mtp_prompt_phase_separation(options)) {
                const auto requirement=strix::decode_cli::compile_prompt_requirements(options,model,manifest,mapped.bytes());
                if(!requirement.owner_error.empty()) throw std::invalid_argument(requirement.owner_error);
                auto owners=requirement.owners;owners["rotary_table"]=options.rotary.table_bytes();
                if(rolling_metadata) owners["ple_rolling_owner"]=rolling_metadata;
                const auto peaks=strix::decode_cli::prefill_owner_phase_peaks(owners,requirement.owner_phases,
                    options.known_prefill_host_headroom);
                // Setup is complete. The live inventory pins actual current backing;
                // its future commitment is the common calibration/prompt/decode max.
                const auto peak=*std::max_element(peaks.begin()+1,peaks.end());
                if(prefill_live.reservation_upper_bound_bytes>peak)
                    throw std::runtime_error("MTP actual live inventory exceeds phased owner envelope");
                growth=peak-prefill_live.reservation_upper_bound_bytes;
                known_prefill_runtime_payload=0;mtp_growth={}; // already in the phase maximum
                rolling_metadata=0; // already in the same phase envelope used by inspection
            }
            if (ple_turn) std::fprintf(stderr,
                "ple-rolling MTP admission: live=%llu growth=%llu metadata=%llu mtp-retained=%llu "
                "temporary=%llu verified-credit=%llu retained-budget=%llu setup-budget=%llu B\n",
                static_cast<unsigned long long>(prefill_live.reservation_upper_bound_bytes),
                static_cast<unsigned long long>(growth), static_cast<unsigned long long>(rolling_metadata),
                static_cast<unsigned long long>(mtp_growth.retained),
                static_cast<unsigned long long>(add(known_prefill_runtime_payload, mtp_growth.temporary)),
                static_cast<unsigned long long>(ple_reclaimer->credit_bytes()),
                static_cast<unsigned long long>(options.known_prefill_uma_budget),
                static_cast<unsigned long long>(options.known_prefill_uma_budget));
            known_prefill_reservation = strix::decode_cli::plan_known_prefill_reservation(
                prefill_live, growth, known_prefill_runtime_payload,
                {options.known_prefill_uma_budget, options.known_prefill_uma_budget},
                ple_reclaimer ? ple_reclaimer->credit_bytes() : 0U, rolling_metadata, mtp_growth);
            // Physical checks charge only positive growth; never add reclaim to
            // MemAvailable or deduct it from the independent GTT observation.
            growth = add(growth, add(rolling_metadata, mtp_growth.retained));
            const auto setup_growth = add(known_prefill_runtime_payload, mtp_growth.temporary);
            if (ple_turn || yarn) {
                // Credit reduces the gross model envelope once. MemAvailable
                // already includes reclaimable cache and never receives credit.
                const auto observed = strix::observe_linux_uma_memory().memory;
                if (!observed.mem_total_bytes || !observed.mem_available_bytes ||
                    !strix::decode_cli::prefill_physical_margin_fits(*observed.mem_total_bytes,*observed.mem_available_bytes,
                        known_prefill_reservation->setup_peak_reserved_bytes,add(growth,setup_growth)))
                    throw std::runtime_error("prefill physical admission: total/incremental envelope must retain 4 GiB margin");
                strix::decode_cli::PrefillCapabilities caps;
                const auto device = strix::decode_cli::observe_prefill_memory(caps);
                // PLE is not registered: no credit is subtracted from GTT.
                if (!device.gtt_limit || !device.gtt_used || *device.gtt_used > *device.gtt_limit ||
                    add(growth, setup_growth) > *device.gtt_limit - *device.gtt_used)
                    throw std::runtime_error("prefill physical admission: GTT observation unavailable or growth exceeds remaining capacity");
            }
            struct PrefillRetained {
                std::shared_ptr<const void> original, d1, ple, source, plan;
            };
            const std::shared_ptr<const void> d1_pin = deferred_moe
                ? std::static_pointer_cast<const void>(deferred_moe) : std::static_pointer_cast<const void>(experiment_owner);
            const auto retained = std::make_shared<PrefillRetained>(PrefillRetained{
                experiment_owner, d1_pin, prefill_ple_owner, retained_ple_source, known_prefill_plan});
            known_prefix_memory = std::make_shared<KnownPromptMemory>(*known_prefill_plan, arena_generation,
                status_generation, retained, original_work_device, union_bytes,
                union_bytes ? generations.next() : 0);
            known_prefix_owner = std::make_shared<strix::HipKnownPrefixWindow>(*known_prefill_plan,
                compiled_prefill ? *compiled_prefill : compiled, model,
                compiled_prefill ? prefill_catalog_config : catalog_config,
                compiled_prefill ? prefill_resources : resources, known_prefix_memory->scratch, known_prefix_memory->scope,
                std::static_pointer_cast<const void>(known_prefix_memory),
                union_bytes ? strix::q51_known_rows::Workspace{known_prefix_memory->union_scratch,
                    std::static_pointer_cast<const void>(known_prefix_memory)} : strix::q51_known_rows::Workspace{});
            for (const auto& known_tail_plan : known_tail_plans) {
                const auto tail_retained = std::make_shared<PrefillRetained>(PrefillRetained{
                    experiment_owner, d1_pin, prefill_ple_owner, retained_ple_source, known_tail_plan});
                const auto known_tail_memory = std::make_shared<KnownPromptMemory>(*known_tail_plan, generations.next(),
                    generations.next(), tail_retained, original_work_device, union_bytes,
                    union_bytes ? generations.next() : 0);
                known_tail_memories.push_back(known_tail_memory);
                known_tail_owners.push_back(std::make_shared<strix::HipKnownPrefixWindow>(*known_tail_plan,
                    compiled_prefill ? *compiled_prefill : compiled, model,
                    compiled_prefill ? prefill_catalog_config : catalog_config,
                    compiled_prefill ? prefill_resources : resources, known_tail_memory->scratch, known_tail_memory->scope,
                    std::static_pointer_cast<const void>(known_tail_memory),
                    union_bytes ? strix::q51_known_rows::Workspace{known_tail_memory->union_scratch,
                        std::static_pointer_cast<const void>(known_tail_memory)} : strix::q51_known_rows::Workspace{}));
            }
            for (const auto& known_wide_plan : known_wide_plans) {
                const auto wide_retained = std::make_shared<PrefillRetained>(PrefillRetained{
                    experiment_owner, d1_pin, prefill_ple_owner, retained_ple_source, known_wide_plan});
                const auto known_wide_memory = std::make_shared<KnownPromptMemory>(*known_wide_plan, generations.next(),
                    generations.next(), wide_retained, original_work_device, union_bytes,
                    union_bytes ? generations.next() : 0);
                known_wide_memories.push_back(known_wide_memory);
                known_wide_owners.push_back(std::make_shared<strix::HipKnownPrefixWindow>(*known_wide_plan,
                    compiled, model, catalog_config, resources, known_wide_memory->scratch, known_wide_memory->scope,
                    std::static_pointer_cast<const void>(known_wide_memory),
                    union_bytes ? strix::q51_known_rows::Workspace{known_wide_memory->union_scratch,
                        std::static_pointer_cast<const void>(known_wide_memory)} : strix::q51_known_rows::Workspace{}));
            }
        } else {
            known_prefix_memory = std::make_shared<KnownPromptMemory>(*known_prefix_plan, arena_generation,
                status_generation,
                std::static_pointer_cast<const void>(experiment_owner), original_work_device,
                options.prompt_lookup ? compiled.execution_identity : 0,
                options.prompt_lookup ? generations.next() : 0,
                union_bytes, union_bytes ? generations.next() : 0);
            known_prefix_owner = std::make_shared<strix::HipKnownPrefixWindow>(*known_prefix_plan, compiled, model,
                catalog_config, resources, known_prefix_memory->scratch, known_prefix_memory->scope,
                std::static_pointer_cast<const void>(known_prefix_memory),
                union_bytes ? strix::q51_known_rows::Workspace{known_prefix_memory->union_scratch,
                    std::static_pointer_cast<const void>(known_prefix_memory)} : strix::q51_known_rows::Workspace{});
        }
        if (options.known_prefix_window)
            known_prefix_backend = std::make_unique<strix::HipKnownPromptBackend>(known_prefix_owner, session,
                known_prefix_memory->scratch, model.vocab_size, original_work_device);
        // Each tiled block runs on the prefill execution and hands the session back to decode.
        if (compiled_prefill) {
            if (!known_prefill_plan || !known_prefix_backend || options.prompt_lookup)
                throw std::logic_error("split prefill profile reached a non-tiled known-prefix route");
            known_prefix_backend->set_executor_handoff({&compiled, &resources}, {&*compiled_prefill, &prefill_resources});
        }
        for (std::size_t t = 0; t < known_tail_owners.size(); ++t) {
            known_tail_backends.push_back(std::make_unique<strix::HipKnownPromptBackend>(known_tail_owners[t], session,
                known_tail_memories[t]->scratch, model.vocab_size, original_work_device));
            if (compiled_prefill)
                known_tail_backends.back()->set_executor_handoff({&compiled, &resources}, {&*compiled_prefill, &prefill_resources});
        }
        if (const char* group_env = std::getenv("STRIX_PROMPT_BATCH_ROWS");
            group_env && known_prefill_plan && strix::decode_cli::mtp_tiled_prompt(options) &&
            std::strtoull(group_env, nullptr, 10) > options.known_prefix_prefill_rows) {
            const auto group_rows = std::strtoull(group_env, nullptr, 10);
            if (group_rows % 8U || group_rows > strix::known_prefill_max_rows || known_prefill_plan->output_policy == strix::KnownWindowOutputPolicy::all_rows)
                throw std::invalid_argument("STRIX_PROMPT_BATCH_ROWS must be a multiple of 8 up to " + strix::known_prefill_rows_range() +
                    " on a terminal-row or state-only tiled prefill");
            known_group_plan = std::make_shared<strix::CompiledKnownPrefixPrefill>(strix::compile_known_prefix_prefill(model,
                compiled_prefill ? *compiled_prefill : compiled, manifest, tensor_access,
                {static_cast<std::uint32_t>(group_rows), true, true, true, options.known_prefix_expert_union,
                 options.known_prefix_output_policy, options.known_prefill_workspace_budget, options.known_prefill_metadata_budget / 2U,
                 true, true, true, true, true, false, false}));
            struct GroupRetained { std::shared_ptr<const void> original, ple, source, plan; };
            const auto retained = std::make_shared<GroupRetained>(GroupRetained{experiment_owner, prefill_ple_owner, retained_ple_source,
                known_group_plan});
            known_group_memory = std::make_shared<KnownPromptMemory>(*known_group_plan, generations.next(), generations.next(),
                retained, original_work_device, known_prefix_union_bytes, known_prefix_union_bytes ? generations.next() : 0);
            known_group_owner = std::make_shared<strix::HipKnownPrefixWindow>(*known_group_plan,
                compiled_prefill ? *compiled_prefill : compiled, model, compiled_prefill ? prefill_catalog_config : catalog_config,
                compiled_prefill ? prefill_resources : resources, known_group_memory->scratch, known_group_memory->scope,
                std::static_pointer_cast<const void>(known_group_memory),
                known_prefix_union_bytes ? strix::q51_known_rows::Workspace{known_group_memory->union_scratch,
                    std::static_pointer_cast<const void>(known_group_memory)} : strix::q51_known_rows::Workspace{});
            known_group_backend = std::make_unique<strix::HipKnownPromptBackend>(known_group_owner, session,
                known_group_memory->scratch, model.vocab_size, original_work_device);
        }
        // Sessions use the registry without legacy prefix lookup/capture in every arm.
        if(!options.cache_activation->automatic_path) {
        const bool rtx_state_registry = session_cache_enabled || (!strix::decode_cli::prefix_state_cache_requested(options) &&
            (options.known_prefix_prefill_rows && options.known_prefix_window) &&
                (options.rtx_state_import || strix::decode_cli::rtx_state_export_request() ||
                    strix::decode_cli::prompt_state_digest_request(options.prefix_batch_cert) || std::getenv("STRIX_FINAL_STATE_DIGEST")));
        if (strix::decode_cli::prefix_state_cache_requested(options) || rtx_state_registry) {
            if (!known_prefill_plan || !known_prefix_backend)
                throw std::logic_error("prefix state cache reached a route other than the tiled known-prefix prefill");
            const auto pair = strix::decode_cli::numerical_profile_pair(options);
            strix::decode_cli::PrefixStateCacheIdentityInputs in;
            in.model_content = model_residency.binding().model_content_identity;
            std::string execution_cli;
            for(const auto& arg : strix::decode_cli::effective_prefill_argv(options))
                execution_cli += std::to_string(arg.size())+":"+arg;
            in.effective_execution = strix::decode_cli::prefix_effective_execution_signature(
                execution_cli, options.baseline, strix::qpf2k::has_gsq_gate(manifest), model.rotary_identity(), effective.prefill_option_mask,
                prefix_namespace_sched_defers(options, rtx_state_registry));
            if (!options.prefix_cache_dir.empty()) in.engine_build = strix::decode_cli::prefix_state_cache_engine_build();
            in.state_layout_identity = state_layout.layout_identity;
            in.runtime_compatibility_identity = identities.compatibility_identity;
            in.decode_execution_identity = compiled.execution_identity;
            in.decode_kernel_plan_identity = compiled.kernel_plan_identity;
            in.prefill_execution_identity = compiled_prefill ? compiled_prefill->execution_identity : compiled.execution_identity;
            in.prefill_kernel_plan_identity = compiled_prefill ? compiled_prefill->kernel_plan_identity : compiled.kernel_plan_identity;
            in.tiled_execution_identity = known_prefill_plan->prefill_execution_identity;
            in.tiled_kernel_plan_identity = known_prefill_plan->prefill_kernel_plan_identity;
            in.prefill_profile_identity = strix::numerical_profile_identity(pair.prefill);
            in.decode_profile_identity = strix::numerical_profile_identity(pair.decode);
            in.profile_pair_identity = strix::numerical_profile_pair_identity(pair);
            in.block_rows = options.known_prefix_prefill_rows;
            in.tail_rows = options.known_prefix_tail_rows;
            in.context_tokens = options.context_tokens;
            prefix_cache_identity = strix::decode_cli::prefix_state_cache_identity(in);
            const std::uint64_t rows = options.known_prefix_prefill_rows;
            if (rtx_state_registry) {
                rtx_state_session = std::make_unique<strix::decode_cli::HipPrefixStateCacheSession>(nullptr,
                    prefix_cache_identity, options.known_prefix_prefill_rows, rows, model, state_layout, memory.state());
            } else {
                prefix_cache_every = options.prefix_cache_every ? options.prefix_cache_every
                    : std::max<std::uint64_t>(rows, (4096U + rows - 1U) / rows * rows);
                prefix_store = std::make_shared<strix::PrefixStateCache>(strix::PrefixStateCachePolicy{
                    options.prefix_cache_host_bytes, options.prefix_cache_disk ? options.prefix_cache_dir : "", options.prefix_cache_disk_bytes,
                    options.prefix_cache_retention == "fair" ? strix::PrefixCacheRetention::fair : strix::PrefixCacheRetention::lru, options.baseline,
                    !options.baseline, options.prefix_cache_durable, options.prefix_fair_reclaim_idle},
                    options.baseline ? strix::decode_cli::hip_pinned_prefix_buffer : strix::decode_cli::hip_pinned_prefix_capture_buffer);
                options.state_cache_stats->publish(strix::decode_cli::prefix_snapshot(*prefix_store));
                prefix_state = std::make_unique<strix::decode_cli::HipPrefixStateCacheSession>(prefix_store,
                    prefix_cache_identity, prefix_cache_unit(options), prefix_cache_every, model, state_layout,
                    memory.state());
                prefix_state->set_staging_cap(options.prefix_cache_staging_bytes);
            }
        }
        } else {
        const bool rtx_state_registry = session_cache_enabled ||
            strix::decode_cli::cache_startup_observer_required(options,
                options.rtx_state_import || strix::decode_cli::rtx_state_export_request(),
                strix::decode_cli::prompt_state_digest_request(strix::decode_cli::prefix_batch_cert_execution_mode(options)));
        prepare_prefix_state=[&,rtx_state_registry] {

            if (options.prefix_cache_host_bytes || (options.cache_activation->prefix_requested &&
                options.cache_activation->automatic_qualified && options.cache_activation->suppression.empty() &&
                options.cache_activation->source!="off") || rtx_state_registry) {
                if (!known_prefill_plan || !known_prefix_backend)
                    throw std::logic_error("prefix state cache reached a route other than the tiled known-prefix prefill");
                const auto pair = strix::decode_cli::numerical_profile_pair(options);
                strix::decode_cli::PrefixStateCacheIdentityInputs in;
                in.model_content = model_residency.binding().model_content_identity;
                std::string execution_cli;
                for(const auto& arg : strix::decode_cli::effective_prefill_argv(options))
                    execution_cli += std::to_string(arg.size())+":"+arg;
                in.effective_execution = strix::decode_cli::prefix_effective_execution_signature(
                    execution_cli, options.baseline, strix::qpf2k::has_gsq_gate(manifest), model.rotary_identity(),
                    strix::prefill_options_active(), prefix_namespace_sched_defers(options, rtx_state_registry));
                if (!options.prefix_cache_dir.empty()) in.engine_build = strix::decode_cli::prefix_state_cache_engine_build();
                in.state_layout_identity = state_layout.layout_identity;
                in.runtime_compatibility_identity = identities.compatibility_identity;
                in.decode_execution_identity = compiled.execution_identity;
                in.decode_kernel_plan_identity = compiled.kernel_plan_identity;
                in.prefill_execution_identity = compiled_prefill ? compiled_prefill->execution_identity : compiled.execution_identity;
                in.prefill_kernel_plan_identity = compiled_prefill ? compiled_prefill->kernel_plan_identity : compiled.kernel_plan_identity;
                in.tiled_execution_identity = known_prefill_plan->prefill_execution_identity;
                in.tiled_kernel_plan_identity = known_prefill_plan->prefill_kernel_plan_identity;
                in.prefill_profile_identity = strix::numerical_profile_identity(pair.prefill);
                in.decode_profile_identity = strix::numerical_profile_identity(pair.decode);
                in.profile_pair_identity = strix::numerical_profile_pair_identity(pair);
                in.block_rows = options.known_prefix_prefill_rows;
                in.tail_rows = options.known_prefix_tail_rows;
                in.context_tokens = options.context_tokens;
                prefix_cache_identity = strix::decode_cli::prefix_state_cache_identity(in);
                const std::uint64_t rows = options.known_prefix_prefill_rows;
                if (rtx_state_registry && !rtx_state_session) {
                    rtx_state_session = std::make_unique<strix::decode_cli::HipPrefixStateCacheSession>(nullptr,
                        prefix_cache_identity, options.known_prefix_prefill_rows, rows, model, state_layout, memory.state());
                }
                {
                    enable_prefix_store = [&,rows](bool ready) {
                        if(!options.cache_activation->prefix_requested ||
                            (!options.prefix_cache_host_bytes && (!options.cache_activation->automatic_qualified ||
                                !options.cache_activation->suppression.empty() || options.cache_activation->source=="off")) ||
                            !apply_cache_fit(ready)) return;
                        try {
                            prefix_cache_every = options.prefix_cache_every ? options.prefix_cache_every
                                : strix::decode_cli::prefix_capture_cadence(rows,0);
                            prefix_store = std::make_shared<strix::PrefixStateCache>(strix::PrefixStateCachePolicy{
                                options.prefix_cache_host_bytes, options.prefix_cache_disk ? options.prefix_cache_dir : "", options.prefix_cache_disk_bytes,
                                options.prefix_cache_retention == "fair" ? strix::PrefixCacheRetention::fair : strix::PrefixCacheRetention::lru, options.baseline,
                                !options.baseline, options.prefix_cache_durable, options.prefix_fair_reclaim_idle},
                                options.baseline ? strix::decode_cli::hip_pinned_prefix_buffer : strix::decode_cli::hip_pinned_prefix_capture_buffer);
                            options.state_cache_stats->publish(strix::decode_cli::prefix_snapshot(*prefix_store));
                            prefix_state = std::make_unique<strix::decode_cli::HipPrefixStateCacheSession>(prefix_store,
                                prefix_cache_identity, prefix_cache_unit(options), prefix_cache_every, model, state_layout,
                                memory.state());
                            prefix_state->set_staging_cap(options.prefix_cache_staging_bytes);
                        } catch(const std::exception&) {
                            // One rollback for every partial setup, including failure
                            // after publishing the store but before registry completion.
                            strix::decode_cli::decline_cache_activation(*options.cache_activation,"cache-setup-allocation-failed");
                            options.prefix_cache_reason=options.cache_activation->reason;
                            options.prefix_cache_host_bytes=0; prefix_state.reset(); prefix_store.reset();
                            options.state_cache_stats->publish({});
                        }
                    };
                }
            }
        };
        if(rtx_state_registry) prepare_prefix_state();
        }
    }
    if (!options.mtp.enabled) {
        strix::decode_cli::finalize_cache_family(options,options.http_serve?strix::decode_cli::CacheRunner::target:strix::decode_cli::CacheRunner::cli,1);
        if(prepare_prefix_state) prepare_prefix_state();
        if(strix::decode_cli::automatic_cache_work(options)) {
            if(enable_prefix_store) enable_prefix_store(false);
        }
        std::cerr << "qwen cache policy: " << strix::decode_cli::cache_activation_json(options).dump() << '\n';
    }

#if defined(STRIX_MTP_MODEL_QUALIFICATION_RUNNER)
    if(target_input_export){
        setup_phase.reset();
        strix::tests::export_mtp_target_inputs(options,model,experiment_owner,compiled,resources,session,direct_dispatcher,
            generations,catalog_config.deferred_moe_status);
        return 0;
    }
#endif
    if (options.mtp.enabled) {
        std::shared_ptr<strix::PleLookupBackend> owned_ple = options.planned_ple_io
            ? std::static_pointer_cast<strix::PleLookupBackend>(prefill_ple_owner)
            : std::static_pointer_cast<strix::PleLookupBackend>(mtp_mapped_ple_owner);
        auto parents = std::make_shared<strix::decode_cli::MtpDecodeTargetParents>(
            experiment_owner, launcher_catalog, std::move(owned_ple), ple_hash_contract);
        parents->assets = ple_mtp_assets;
        if (compiled_prefill && !strix::decode_cli::mtp_tiled_prompt(options)) {
            if (!prefill_dispatcher || !prefill_scratch) throw std::logic_error("MTP split prefill lacks its prompt launchers");
            parents->prompt.compiled = &*compiled_prefill;
            parents->prompt.dispatcher = &*prefill_dispatcher;
            parents->prompt.resources = parents->resources;
            parents->prompt.resources.scratch = &*prefill_scratch;
        }
        if (strix::decode_cli::mtp_tiled_prompt(options)) {
            // The prompt runs through the tiled prefill set up above (see run_tiled_mtp_prompt).
            if (!known_prefix_backend || !known_prefix_owner || !known_prefill_plan)
                throw std::logic_error("tiled MTP prompt lacks its tiled prefill owners");
            auto& tiled = parents->tiled;
            tiled.blocks = known_prefix_backend.get(); tiled.block_owner = known_prefix_owner;
            for (std::size_t t = 0; t < known_tail_backends.size(); ++t)
                tiled.tails.push_back({known_tail_backends[t].get(), known_tail_owners[t], options.known_prefix_tail_rows[t]});
            for (std::size_t w = 0; w < known_wide_owners.size(); ++w)
                tiled.wide.push_back({nullptr, known_wide_owners[w], options.mtp.copy_wide_rows[w]});
            tiled.rows = options.known_prefix_prefill_rows;
            tiled.numerical_plan.block.execution=known_prefill_plan->prefill_execution_identity;
            tiled.numerical_plan.block.kernel=known_prefill_plan->prefill_kernel_plan_identity;
            for(const auto& tail:known_tail_plans) {
                strix::decode_cli::QwenPlanOwner owner;
                owner.execution=tail->prefill_execution_identity;owner.kernel=tail->prefill_kernel_plan_identity;
                tiled.numerical_plan.tails.push_back(owner);
            }
            tiled.numerical_plan.decode_execution=compiled.execution_identity;
            tiled.numerical_plan.decode_kernel=compiled.kernel_plan_identity;
            tiled.numerical_plan.prefill_execution=compiled_prefill?compiled_prefill->execution_identity:compiled.execution_identity;
            tiled.numerical_plan.prefill_kernel=compiled_prefill?compiled_prefill->kernel_plan_identity:compiled.kernel_plan_identity;
            const auto plan_profiles=strix::decode_cli::numerical_profile_pair(options);
            tiled.numerical_plan.prefill_profile=strix::numerical_profile_identity(plan_profiles.prefill);
            tiled.numerical_plan.decode_profile=strix::numerical_profile_identity(plan_profiles.decode);
            tiled.numerical_plan.profile_pair=strix::numerical_profile_pair_identity(plan_profiles);
            // Split prefill profile: the block backends above hand the session to the prefill execution.
            tiled.prefill_execution_identity = compiled_prefill ? compiled_prefill->execution_identity : 0U;
            tiled.cache = prefix_state ? prefix_state.get() : rtx_state_session.get();
            tiled.cache_tails = prefix_state && prefix_cache_unit(options) < options.known_prefix_prefill_rows;
            tiled.ladder_tail_rows = options.known_prefix_ladder_tail_rows;
            if (options.rtx_state_import)
                tiled.rtx_import = std::make_shared<strix::RtxStateImportRegistry>(strix::decode_cli::hip_pinned_prefix_buffer, 2,
                    options.rtx_state_import_decode_threads);
            tiled.reset_state = [&] {
                check_hip(hipDeviceSynchronize(), "tiled MTP prompt reset fence");
                if(prefix_state) prefix_state->before_target_reset();
                strix::initialize_hip_fresh_runtime_memory(memory.scratch(), memory.state());
            };
            tiled.reset_prefill_memory = [known_prefix_memory,known_tail_memories] {
                known_prefix_memory->reset_cancelled_prefill();
                for(const auto& arena:known_tail_memories) arena->reset_cancelled_prefill();
            };
            tiled.fresh = session;
            tiled.group = known_group_backend.get();
            tiled.group_owner = known_group_owner;
        }
        // Prepare the real closed native owner BEFORE any prompt mutation.
        // It retains all serial parents and its separate D1/arena ownership.
        // A row set builds one owner per size; owners share one deferred
        // GDN record and the arena/total budgets cover all of them. Cohort
        // requests build their owners on one shared window stream.
        const auto prepare_mtp_pool = [&](strix::decode_cli::MtpDecodeTargetParents* request_parents) -> std::uint64_t {
              const auto row_set = options.mtp.window_rows_set.empty()
                  ? std::vector<std::uint32_t>{options.mtp.window_rows} : options.mtp.window_rows_set;
              std::uint64_t pool_reservation=0;
              if(request_parents->resources.qsa_committed_pool)
                  return strix::mtp_qsa_pool_plan(options.context_tokens).reserved_bytes;
              if (strix::decode_cli::qsa_mtp_pool_bytes(options,model,strix::qpf2k::has_gsq_gate(manifest))) {
                  const auto pool_plan=strix::mtp_qsa_pool_plan(options.context_tokens);
                  // Preserve the existing independent budgets and physical headroom. The service
                  // is charged once outside the R2/R4/R8 loop, including the predictor entry.
                  std::uint64_t window_bytes=0;
                  for(const auto wr:row_set) {
                      const auto wp=strix::compile_known_prefix_window(model,compiled,manifest,tensor_access,
                          {true,true,true,wr,options.mtp.window_expert_union,strix::KnownWindowOutputPolicy::all_rows,
                           options.mtp.window_hyper_mix,options.mtp.window_qsa_rows,options.mtp.window_prefix,options.mtp.window_prefix,
                           options.mtp.window_prefix,options.mtp.window_prefix || options.known_window_router_rows,
                           options.mtp.window_prefix || options.known_window_router_rows,
                           options.rotary.mode()==strix::RotaryMode::yarn});
                      const auto ub=options.mtp.window_expert_union?strix::q51_known_rows::required_scratch_bytes(wr,
                          static_cast<std::uint32_t>(model.num_experts),static_cast<std::uint32_t>(model.num_experts_per_token)):0U;
                      window_bytes=strix::mtp_native::sum(window_bytes,strix::mtp_native::sum(wp.arena.physical.total_bytes,8U+ub));
                  }
                  std::uint64_t captures=0;
                  for(const auto wr:row_set) captures=strix::mtp_native::sum(captures,strix::mtp_native::product(4U,
                      strix::mtp_payload::window_capture(model,wr,options.mtp.window_history_intervals,manifest.tensors.size())));
                  const auto checkpoint=options.mtp.window_prefix?0U:strix::plan_hip_mtp_target_window_rollback(model,*request_parents->resources.session_state_layout,
                      0,options.mtp.window_rows,true).reservation.retained_bytes;
                  const bool fits=strix::mtp_qsa_pool_fits(pool_plan,options.mtp.window_total_bytes,window_bytes,captures,checkpoint);
                  if (fits) {
                      request_parents->resources.qsa_committed_pool=strix::HipQsaCommittedPool::create(pool_plan);
                      if(request_parents->resources.qsa_committed_pool) {
                          request_parents->resources.qsa_committed_pool->bind_domain(false,
                              request_parents->resources.model_weights->model_content_identity,compiled.execution_identity);
                          pool_reservation=pool_plan.reserved_bytes;
                      }
                  }
                  if (!pool_reservation && yarn)
                      throw std::runtime_error("YaRN MTP M1 admission/allocation refused; historical owners require a separate run with STRIX_QSA_MTP_POOL=0");
                  if (!pool_reservation) std::cerr << "qsa-mtp-pool requested=1 enabled=0 reason=admission-or-allocation\n";
              } else if(options.mtp.enabled) {
                  const char* reason=strix::decode_cli::qsa_mtp_pool_decline(options,strix::qpf2k::has_gsq_gate(manifest));
                  if(reason && std::string_view(reason)!="disabled")
                      std::cerr << "qsa-mtp-pool requested=1 enabled=0 reason=" << reason << '\n';
              }
              return pool_reservation;
        };
        // Key of the persisted window-cost tables: filled after the HTTP slot capacity is final.
        const auto cost_table_config = std::make_shared<std::string>();
        const auto window_factory_for = [&](std::shared_ptr<strix::decode_cli::MtpDecodeTargetParents> request_parents,
            std::shared_ptr<const SharedWindowStream> window_stream) -> strix::decode_cli::MtpDecodeWindowFactory {
            if (!options.mtp.window_rows || (options.session_mtp_fixed && !options.http_serve)) return {};
            return [&, request_parents, window_stream](std::shared_ptr<strix::HipMtpTargetHidden> target) {
              const auto row_set = options.mtp.window_rows_set.empty()
                  ? std::vector<std::uint32_t>{options.mtp.window_rows} : options.mtp.window_rows_set;
              std::vector<std::shared_ptr<strix::HipKnownPrefixWindow>> owners;
              const auto pool_reservation=prepare_mtp_pool(request_parents.get());
              std::uint64_t retained_total = pool_reservation, union_total = 0, union_steps = 0, union_generation = 0;
              for (const auto window_rows : row_set) {
                const auto plan = strix::compile_known_prefix_window(model, compiled, manifest, tensor_access,
                    {true, true, true, window_rows, options.mtp.window_expert_union, strix::KnownWindowOutputPolicy::all_rows,
                     options.mtp.window_hyper_mix, options.mtp.window_qsa_rows, options.mtp.window_prefix, options.mtp.window_prefix,
                     options.mtp.window_prefix, options.mtp.window_prefix || options.known_window_router_rows,
                     options.mtp.window_prefix || options.known_window_router_rows,
                     options.rotary.mode()==strix::RotaryMode::yarn});
                if (!strix::known_window_has_promoted_steps(plan) ||
                    (options.mtp.window_expert_union && !plan.routed_expert_union_steps) ||
                    (options.mtp.window_hyper_mix && !plan.hyper_mix_rows_steps) ||
                    (options.mtp.window_qsa_rows && !plan.qsa_rows_steps) ||
                    (options.mtp.window_prefix && (plan.gdn_deferred_rows_steps != model.gdn_layers || !plan.ple_rows_steps)))
                    throw std::invalid_argument("MTP target window has no admitted requested operators: union " +
                        std::to_string(plan.routed_expert_union_steps) + " hc " + std::to_string(plan.hyper_mix_rows_steps) +
                        " qsa " + std::to_string(plan.qsa_rows_steps) + " gdn-deferred " + std::to_string(plan.gdn_deferred_rows_steps) +
                        " ple " + std::to_string(plan.ple_rows_steps));
                std::uint64_t union_bytes = 0;
                if (options.mtp.window_expert_union) {
                    if (model.num_experts > UINT32_MAX || model.num_experts_per_token > UINT32_MAX)
                        throw std::invalid_argument("MTP window expert union geometry exceeds uint32");
                    union_bytes = strix::q51_known_rows::required_scratch_bytes(window_rows,
                        static_cast<std::uint32_t>(model.num_experts), static_cast<std::uint32_t>(model.num_experts_per_token));
                    if (!union_bytes) throw std::logic_error("MTP expert union has no required scratch");
                }
                const auto reservation = strix::decode_cli::reserve_mtp_window_memory(plan.arena.physical.total_bytes,
                    options.mtp.window_arena_bytes - std::min(retained_total-pool_reservation, options.mtp.window_arena_bytes),
                    options.mtp.window_total_bytes - std::min(retained_total, options.mtp.window_total_bytes),
                    union_bytes, options.mtp.window_expert_union_bytes - std::min(union_total, options.mtp.window_expert_union_bytes));
                auto window_memory = std::make_shared<KnownPromptMemory>(plan, generations.next(), generations.next(),
                    std::static_pointer_cast<const void>(request_parents), original_work_device, 0, 0,
                    union_bytes, union_bytes ? generations.next() : 0, window_stream);
                owners.push_back(std::make_shared<strix::HipKnownPrefixWindow>(plan, compiled, model, catalog_config,
                    request_parents->resources, window_memory->scratch, window_memory->scope, std::static_pointer_cast<const void>(window_memory),
                    union_bytes ? strix::q51_known_rows::Workspace{window_memory->union_scratch,
                        std::static_pointer_cast<const void>(window_memory)} : strix::q51_known_rows::Workspace{}));
                retained_total += reservation.retained_window_bytes; union_total += union_bytes;
                union_steps += plan.routed_expert_union_steps;
                if (union_bytes) union_generation = window_memory->union_scratch.allocation_generation;
              }
                strix::HipMtpWindowVerificationPolicy policy;
                policy.bounded_window_rollback = true;
                policy.capture = {options.mtp.window_capture_bytes, options.mtp.window_history_intervals};
                policy.checkpoint_retained_byte_budget = options.mtp.window_checkpoint_bytes;
                policy.total_retained_byte_budget = options.mtp.window_total_bytes - retained_total;
                policy.min_probability = strix::decode_cli::mtp_draft_min_probability(options.mtp);
                policy.timing_independent_schedule =
                    strix::decode_cli::mtp_sampling(options.mtp) && !options.mtp.sample_timing_schedule;
                // Persisted window costs (cost_table_directory_from_env; OFF: STRIX_COST_TABLE=0): the
                // configuration is filled once the route and HTTP capacity are resolved (below).
                policy.cost_table_configuration = cost_table_config;
                policy.cost_table_baseline = options.baseline;
                auto verifier = std::make_shared<strix::HipMtpWindowVerifier>(std::move(target), std::move(owners),
                    request_parents->resources, request_parents, policy, generations);
                request_parents->window_setup = {union_steps, union_total, union_generation, retained_total};
                return verifier;
            };
        };
        // Each further request of the cohort and batched routes: its own state memory and serial scratch, PLE
        // backend and prefetch, launch plan and session over the same mapped model, manifest and layout; on the
        // tiled prompt also its own tiled prefill backends (over the shared prefill arenas) and prefix cache view.
        struct CohortStack {
            std::shared_ptr<strix::decode_cli::MtpDecodeTargetParents> parents;
            std::shared_ptr<strix::RuntimeExecutionMemoryOwner> memory;
            strix::ExecutionLauncherRegistry launchers;
            std::optional<strix::CompiledLaunchPlan> plan;
            std::optional<strix::DirectLaunchDispatcher> dispatcher;
            strix::SerialSessionContext session;
            std::shared_ptr<strix::HipKnownPrefixWindow> block_owner;
            std::unique_ptr<strix::HipKnownPromptBackend> blocks;
            std::vector<std::shared_ptr<strix::HipKnownPrefixWindow>> tail_owners;
            std::vector<std::unique_ptr<strix::HipKnownPromptBackend>> tails;
            std::unique_ptr<strix::decode_cli::HipPrefixStateCacheSession> cache;
            // Split prefill profile: the sibling prefill execution's view of this slot's serial scratch and
            // its resources (this slot's state, PLE and model bindings), the tiled owners' executor.
            std::optional<strix::ExecutionMemoryBinding> prefill_scratch;
            strix::ExecutionRuntimeResources prefill_resources;
        };
        const auto make_stack = [&] {
            auto stack = std::make_unique<CohortStack>();
            stack->memory = std::make_shared<strix::RuntimeExecutionMemoryOwner>(
                strix::RuntimeExecutionMemoryOwner::allocate(compiled, state_layout, allocator_owner, generations));
            strix::initialize_hip_fresh_runtime_memory(stack->memory->scratch(), stack->memory->state());
            auto request_experiment = std::make_shared<HipDecodeExperimentResources>(
                mapped_owner, model_owner, allocator_owner, stack->memory, manifest, tensor_access, state_layout);
            // One manifest, access plan and layout object for every request: cohort windows
            // compare these owners across the rows of all requests.
            request_experiment->resources.model_manifest = experiment_owner->resources.model_manifest;
            request_experiment->resources.tensor_access = experiment_owner->resources.tensor_access;
            request_experiment->resources.session_state_layout = experiment_owner->resources.session_state_layout;
            std::shared_ptr<strix::PleLookupBackend> request_ple = options.planned_ple_io
                ? std::static_pointer_cast<strix::PleLookupBackend>(std::make_shared<strix::PlannedPleLookupBackend>(
                    *retained_ple_source, planned_ple_policy, retained_ple_source))
                : std::static_pointer_cast<strix::PleLookupBackend>(std::make_shared<strix::FileMappedPleLookupBackend>());
            auto request_catalog = std::make_shared<strix::HipLauncherCatalog>(model, catalog_config);
            stack->parents = std::make_shared<strix::decode_cli::MtpDecodeTargetParents>(
                request_experiment, request_catalog, std::move(request_ple), ple_hash_contract);
            request_catalog->register_for_compiled_execution(compiled, stack->launchers, &stack->parents->resources);
            if (request_catalog->audit(compiled, stack->launchers).missing_steps)
                throw std::logic_error("HIP launcher catalog is incomplete for a cohort request");
            stack->plan.emplace(strix::compile_launch_plan(compiled, stack->launchers));
            stack->dispatcher.emplace(*stack->plan, active_observer);
            const auto request_identities = strix::build_runtime_execution_identities(
                compiled.execution_identity, compiled.arena, stack->parents->resources);
            stack->session.execution_identity = compiled.execution_identity;
            stack->session.state_layout_identity = state_layout.layout_identity;
            stack->session.runtime_compatibility_identity = request_identities.compatibility_identity;
            stack->session.state.valid = all_persistent_state();
            return stack;
        };
        // The tiled prompt of a further request: owners over its own state, PLE and launchers and the shared
        // prefill arenas (one prompt runs at a time), backends bound to its session, its own cache view. Under
        // a split prefill profile the owners run the sibling prefill execution with the prefill launcher
        // configuration over this slot's resources, and each backend hands this slot's session to it and back.
        const auto add_tiled = [&](CohortStack& stack) {
            if (!known_prefill_plan || !known_prefix_memory)
                throw std::logic_error("batched MTP requests need the tiled prefill");
            const auto workspace = [&](const std::shared_ptr<KnownPromptMemory>& arena) {
                return known_prefix_union_bytes ? strix::q51_known_rows::Workspace{arena->union_scratch,
                    std::static_pointer_cast<const void>(arena)} : strix::q51_known_rows::Workspace{};
            };
            if (compiled_prefill) {
                stack.prefill_scratch = strix::sibling_scratch_binding(*compiled_prefill, compiled, stack.memory->scratch());
                stack.prefill_resources = stack.parents->resources;
                stack.prefill_resources.scratch = &*stack.prefill_scratch;
            }
            const auto& owner_compiled = compiled_prefill ? *compiled_prefill : compiled;
            const auto& owner_catalog = compiled_prefill ? prefill_catalog_config : catalog_config;
            const auto& owner_resources = compiled_prefill ? stack.prefill_resources : stack.parents->resources;
            const auto hand_off = [&](strix::HipKnownPromptBackend& backend) {
                if (compiled_prefill)
                    backend.set_executor_handoff({&compiled, &stack.parents->resources}, {&*compiled_prefill, &stack.prefill_resources});
            };
            stack.block_owner = std::make_shared<strix::HipKnownPrefixWindow>(*known_prefill_plan, owner_compiled, model, owner_catalog,
                owner_resources, known_prefix_memory->scratch, known_prefix_memory->scope,
                std::static_pointer_cast<const void>(known_prefix_memory), workspace(known_prefix_memory));
            stack.blocks = std::make_unique<strix::HipKnownPromptBackend>(stack.block_owner, stack.session,
                known_prefix_memory->scratch, model.vocab_size, original_work_device);
            hand_off(*stack.blocks);
            for (std::size_t t = 0; t < known_tail_plans.size(); ++t) {
                stack.tail_owners.push_back(std::make_shared<strix::HipKnownPrefixWindow>(*known_tail_plans[t], owner_compiled, model,
                    owner_catalog, owner_resources, known_tail_memories[t]->scratch, known_tail_memories[t]->scope,
                    std::static_pointer_cast<const void>(known_tail_memories[t]), workspace(known_tail_memories[t])));
                stack.tails.push_back(std::make_unique<strix::HipKnownPromptBackend>(stack.tail_owners.back(), stack.session,
                    known_tail_memories[t]->scratch, model.vocab_size, original_work_device));
                hand_off(*stack.tails.back());
            }
            if (prefix_store)
                stack.cache = std::make_unique<strix::decode_cli::HipPrefixStateCacheSession>(prefix_store, prefix_cache_identity,
                    prefix_cache_unit(options), prefix_cache_every, model, state_layout, stack.memory->state());
            else if (rtx_state_session)   // STRIX_PROMPT_STATE_DIGEST without a prefix cache: each slot gets its own registry
                stack.cache = std::make_unique<strix::decode_cli::HipPrefixStateCacheSession>(nullptr, prefix_cache_identity,
                    options.known_prefix_prefill_rows, options.known_prefix_prefill_rows, model, state_layout, stack.memory->state());
            if(stack.cache) stack.cache->set_staging_cap(options.prefix_cache_staging_bytes);
            auto& tiled = stack.parents->tiled;
            tiled.blocks = stack.blocks.get(); tiled.block_owner = stack.block_owner;
            for (std::size_t t = 0; t < stack.tails.size(); ++t)
                tiled.tails.push_back({stack.tails[t].get(), stack.tail_owners[t], options.known_prefix_tail_rows[t]});
            tiled.rows = options.known_prefix_prefill_rows;
            tiled.numerical_plan.block.execution=known_prefill_plan->prefill_execution_identity;
            tiled.numerical_plan.block.kernel=known_prefill_plan->prefill_kernel_plan_identity;
            for(const auto& tail:known_tail_plans) {
                strix::decode_cli::QwenPlanOwner owner;
                owner.execution=tail->prefill_execution_identity;owner.kernel=tail->prefill_kernel_plan_identity;
                tiled.numerical_plan.tails.push_back(owner);
            }
            tiled.numerical_plan.decode_execution=compiled.execution_identity;
            tiled.numerical_plan.decode_kernel=compiled.kernel_plan_identity;
            tiled.numerical_plan.prefill_execution=compiled_prefill?compiled_prefill->execution_identity:compiled.execution_identity;
            tiled.numerical_plan.prefill_kernel=compiled_prefill?compiled_prefill->kernel_plan_identity:compiled.kernel_plan_identity;
            const auto plan_profiles=strix::decode_cli::numerical_profile_pair(options);
            tiled.numerical_plan.prefill_profile=strix::numerical_profile_identity(plan_profiles.prefill);
            tiled.numerical_plan.decode_profile=strix::numerical_profile_identity(plan_profiles.decode);
            tiled.numerical_plan.profile_pair=strix::numerical_profile_pair_identity(plan_profiles);
            tiled.prefill_execution_identity = compiled_prefill ? compiled_prefill->execution_identity : 0U;
            tiled.cache = stack.cache.get();
            tiled.cache_tails = prefix_store && prefix_cache_unit(options) < options.known_prefix_prefill_rows;
            tiled.reset_state = [slot_memory = stack.memory,cache=stack.cache.get()] {
                check_hip(hipDeviceSynchronize(), "batched MTP slot reset fence");
                if(cache) cache->before_target_reset();
                strix::initialize_hip_fresh_runtime_memory(slot_memory->scratch(), slot_memory->state());
            };
            tiled.reset_prefill_memory = [known_prefix_memory,known_tail_memories] {
                known_prefix_memory->reset_cancelled_prefill();
                for(const auto& arena:known_tail_memories) arena->reset_cancelled_prefill();
            };
            tiled.fresh = stack.session;
        };
        const bool tiled_route = strix::decode_cli::mtp_tiled_prompt(options);
        if (!options.mtp.cohort.empty() && !tiled_route) {
            const auto window_stream = std::make_shared<const SharedWindowStream>();
            std::vector<std::unique_ptr<CohortStack>> stacks;
            std::vector<strix::decode_cli::MtpCohortRequest> requests;
            requests.push_back({parents, &session, &direct_dispatcher, options.tokens, options.mtp.golden,
                options.mtp.golden_sha256, window_factory_for(parents, window_stream)});
            for (const auto& request : options.mtp.cohort) {
                auto stack = make_stack();
                requests.push_back({stack->parents, &stack->session, &*stack->dispatcher, request.tokens, request.golden,
                    request.golden_sha256, window_factory_for(stack->parents, window_stream)});
                stacks.push_back(std::move(stack));
            }
            setup_phase.reset();
            strix::decode_cli::run_mtp_cohort_bridge(options, model, compiled, requests, generations,
                catalog_config.deferred_moe_status, optimization_profile.selected, std::cout);
            return 0;
        }
        // Batched decoding: cohort requests on the tiled prompt (one slot per request) or --http-batch N.
        std::size_t batch_slots = tiled_route && !options.mtp.cohort.empty() ? 1U + options.mtp.cohort.size()
            : strix::decode_cli::ordinary_http_batch_slots(options);
        std::shared_ptr<const SharedWindowStream> batch_stream;
        std::vector<std::unique_ptr<CohortStack>> batch_stacks;
        std::function<bool()> batch_memory_ok;
        if (options.http_serve && tiled_route) {
            using namespace strix::decode_cli;
            const auto add = [](std::uint64_t a, std::uint64_t b) {
                if (b > UINT64_MAX - a) throw std::overflow_error("HTTP slot cost overflow");
                return a + b;
            };
            // Resident unlocked registered model pages count as reclaimable in MemAvailable,
            // but reclaiming them evicts every GPU queue of this process: not available to slots.
            const auto unlocked_registered = model_residency.registered_bytes() - model_residency.registered_locked_bytes();
            const auto reserve_env = std::getenv("STRIX_PLE_SERVE_RESERVE");
            // Opt-in: reserving the table can cost T8192 and slots on a full node.
            const bool reserve_ple = !strix::hip_model_copy_all_mode() && reserve_env && reserve_env[0] == '1';
            std::uint64_t reserved_table_bytes = 0;
            if (reserve_ple && warmed_file_backed_bytes) {
                strix::HipModelResidencyPolicy policy;
                policy.coalesce_gap_bytes = options.coalesce_gap_bytes;
                const auto page = static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
                reserved_table_bytes = strix::decode_cli::file_backed_unregistered_bytes(manifest,
                    strix::build_registered_residency_ranges(manifest, policy, mapped.bytes(), page), mapped.bytes(), page);
            }
            const auto observe = [unlocked_registered, reserved_table_bytes, reserve_ple] {
                PrefillCapabilities capabilities;
                auto observation = observe_prefill_memory(capabilities);
                if (observation.available)
                    observation.available = strix::ple_detail::serve_available(*observation.available,
                        unlocked_registered, reserved_table_bytes, reserve_ple);
                return observation;
            };
            const auto gtt_free = [](const PrefillMemory& observation) -> std::optional<std::uint64_t> {
                if (!observation.gtt_limit || !observation.gtt_used || *observation.gtt_used > *observation.gtt_limit) return {};
                return *observation.gtt_limit - *observation.gtt_used;
            };
            const auto serving = prompt_memory.serving(options);
            // Future commitments exclude the already resident primary stack and tiled arenas.
            auto estimate = add(add(state_layout.device_total_bytes, state_layout.host_total_bytes), compiled.arena.total_bytes);
            estimate = add(estimate, known_prefill_plan->host_metadata_payload_bytes);
            for (const auto& tail : known_tail_plans) estimate = add(estimate, tail->host_metadata_payload_bytes);
            // Per-slot owner buffers are independent even though arenas are shared.
            const auto external=qwen_prefill_external_memory(options,model);
            estimate=add(estimate,std::max(external.host,external.device));
            const auto fixed_headroom = options.known_prefill_headroom_explicit ? options.known_prefill_host_headroom : UINT64_C(4) << 30U;
            // Shared bridge assets and the cache's future payload are not resident
            // at this observation boundary. Reserve them once, never per slot.
            // The automatic cache budget is fitted after residency; reserve only explicit payload here.
            auto headroom = add(fixed_headroom, options.cache_activation->automatic_path ? UINT64_C(0) : options.prefix_cache_host_bytes);
            // UMA: K=1 device staging and pinned host payload share physical RAM.
            // Initial sizing reserves the peak; growth reserves only its outstanding fraction.
            const auto staging_reserve=qwen_http_batch_staging_reserve(options,model,state_layout);
            headroom=add(headroom,staging_reserve);
            const auto before = observe();
            auto retained = estimate;
            // Observations of the last fit, reused if the route falls back to the single runner.
            auto fit_host = before.available;
            auto fit_device = gtt_free(before);
            auto capacity = resolve_http_bridge_capacity(estimate, serving, fit_host, fit_device, headroom, headroom, options.http_batch_cap);
            if (!capacity.slots) throw std::runtime_error("HTTP first slot does not fit: " + capacity.limiting);
            const bool single_route=!ordinary_http_batch_slots(options);
            if (single_route) capacity.slots=1;
            if (capacity.slots >= 2U) {
                auto probe = make_stack();
                add_tiled(*probe);
                check_hip(hipDeviceSynchronize(), "HTTP slot measurement fence");
                const auto after = observe();
                const auto delta = [](std::optional<std::uint64_t> a, std::optional<std::uint64_t> b) {
                    return a && b && *a > *b ? *a - *b : UINT64_C(0);
                };
                const auto measured = std::max({estimate, delta(before.available, after.available), delta(gtt_free(before), gtt_free(after))});
                retained = measured;
                fit_host = after.available ? before.available : std::nullopt;
                fit_device = gtt_free(after) ? gtt_free(before) : std::nullopt;
                capacity = resolve_http_bridge_capacity(measured, serving, fit_host, fit_device, headroom, headroom, options.http_batch_cap);
                if (capacity.slots >= 2U) batch_stacks.push_back(std::move(probe));
            }
            if (!capacity.slots) throw std::runtime_error("HTTP first slot does not fit after probe: " + capacity.limiting);
            batch_memory_ok = make_http_bridge_growth_check(retained,serving,fixed_headroom,
                options.prefix_cache_host_bytes,staging_reserve,[observe,gtt_free,prefix_store] {
                    // Allocated payload includes in-flight capture, not unallocated credits.
                    const auto stats=prefix_store ? prefix_store->stats() : strix::PrefixStateCacheStats{};
                    const auto pressure=observe();
                    return HttpBridgeGrowthObservation{pressure.available,gtt_free(pressure),
                        stats.host_bytes,stats.staging_bytes};
                });
            if(options.cache_activation->automatic_path) {
                cache_slot_retained=retained;
                const auto mul=strix::decode_cli::prefill_checked_mul;
                // Native owner budgets bound the geometry charged in serving.slot_*.
                cache_slot_native=add(add(options.mtp.runtime_bytes,options.mtp.window_total_bytes),options.mtp.target_capture_bytes);
                // Request admission builds taps, draft/greedy heads and scratch.
                // Keep all of them unpaid even when the lane's native owners exist.
                const auto width=mul(mul(model.hidden_size,model.hc_count),2U);
                const bool subset=!options.mtp.draft_vocab.empty() || options.mtp.draft_context_vocab;
                const auto row=subset ? mul(model.hidden_size/32U,options.mtp.draft_q4?18U:34U) : 0U;
                cache_prompt_transient=add(mul(mul(2U,options.known_prefix_prefill_rows),width),mul(2U,width));
                cache_prompt_transient=add(cache_prompt_transient,mul(model.vocab_size,add(row,77U)));
                cache_prompt_transient=add(cache_prompt_transient,add(mul((model.vocab_size+255U)/256U,32U),4096U));
                cache_prompt_transient=add(cache_prompt_transient,4U*(11U*640U+1U)+2U*4U*10240U*4U);
            }
            options.http_slot_reason = "limited by " + capacity.limiting;
            batch_slots = !single_route && (capacity.slots > 1U || (options.http_batch_explicit && options.http_batch_cap == 1U))
                ? capacity.slots : 0U;
            const bool pool_fallback = !single_route && !batch_slots &&
                qsa_mtp_pool_single_fallback_pending(options, strix::qpf2k::has_gsq_gate(manifest));
            options.http_batch = static_cast<std::uint32_t>(batch_slots);
            if (pool_fallback) {
                // Batch sizing excluded the pool; the single runner keeps it only if its one slot still fits with it.
                const auto single = resolve_http_bridge_capacity(retained, prompt_memory.serving(options),
                    fit_host, fit_device, headroom, headroom, 1U);
                options.qsa_mtp_pool_fallback_unfit = !single.slots;
                std::cerr << "qsa-mtp-pool single-fallback enabled=" << (single.slots ? 1 : 0) << " reserved_bytes="
                          << strix::mtp_qsa_pool_plan(options.context_tokens).reserved_bytes << " (" << single.limiting << ")\n";
            }
            if (!batch_slots && options.http_sampling == "sample")
                throw std::runtime_error("HTTP sampling requires memory for the batched route");
        }
        std::size_t eager_slots = batch_slots;
        const auto rewarm_before_listen = [&] {
            // Slot allocation after the startup warm reclaims part of the table. STRIX_PLE_REWARM=0 skips it.
            const auto rewarm = std::getenv("STRIX_PLE_REWARM");
            if(!warmed_file_backed_bytes || ple_reclaimer || strix::hip_model_copy_all_mode() || (rewarm && rewarm[0]=='0')) return;
            // Repopulating may reclaim other pages: only with every registered model byte locked (a reclaimed
            // registration evicts the process's GPU queues) and only if the missing pages fit with 4 GiB to spare.
            const char* skip = nullptr;
            const auto missing = file_backed_missing_bytes(mapped, manifest);
            strix::decode_cli::PrefillCapabilities caps;
            const auto now = strix::decode_cli::observe_prefill_memory(caps);
            if (model_residency.registered_locked_bytes() != model_residency.registered_bytes()) skip = "registered model pages are not locked";
            else if (!missing || !now.available) skip = "residency or MemAvailable unknown";
            else if (*missing == 0) skip = "table resident";
            else if (*now.available <= (UINT64_C(4) << 30)) skip = "no MemAvailable above the 4 GiB headroom";
            else if (*missing > *now.available - (UINT64_C(4) << 30)) {
                // Rows are hashed n-grams spread over the whole table: any resident fraction helps every prompt.
                // Fill what fits; STRIX_PLE_REWARM=full keeps the previous all-or-nothing rule.
                if (rewarm && std::string_view(rewarm) == "full") skip = "missing pages do not fit MemAvailable with 4 GiB headroom";
                else {
                    const auto t0 = std::chrono::steady_clock::now();
                    const auto budget = *now.available - (UINT64_C(4) << 30);
                    const auto populated = rewarm_file_backed_capped(mapped, manifest, budget);
                    std::fprintf(stderr, "file-backed tensors: re-warmed %.1f of %.1f GB not resident (capped to MemAvailable - 4 GiB) in %.1f s before listening\n",
                        static_cast<double>(populated) / 1e9, static_cast<double>(*missing) / 1e9,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                    return;
                }
            }
            if (skip) {
                if (missing && *missing) std::fprintf(stderr, "file-backed tensors: re-warm skipped (%s; %.1f GB not resident)\n", skip,
                    static_cast<double>(*missing) / 1e9);
                return;
            }
            (void)warm_file_backed_tensors(mapped,manifest,true);
        };
        if (options.http_serve && batch_slots) {
            const char* eager = std::getenv("STRIX_HTTP_EAGER_SLOTS");
            std::uint32_t count = 4;
            if (eager) {
                const auto parsed = strix::decode_cli::parse_u64(eager, "STRIX_HTTP_EAGER_SLOTS");
                if (parsed > UINT32_MAX) throw std::invalid_argument("STRIX_HTTP_EAGER_SLOTS exceeds uint32");
                count = static_cast<std::uint32_t>(parsed);
            }
            eager_slots = strix::decode_cli::http_eager_slots(static_cast<std::uint32_t>(batch_slots), count);
            // The measurement probe is retained only if it belongs to the eager part.
            if (eager_slots == 1U) batch_stacks.clear();
        }
        {
            // Resolved route and numerical identities of the persisted window-cost tables (verifiers read it at their
            // first decode cycle). The automatic slot capacity depends on free memory at startup and is not keyed
            // (mtp_cost_table_configuration). The device key adds the HIP runtime and driver versions.
            const auto profiles = strix::decode_cli::numerical_profile_pair(options);
            strix::MtpCostTableRoute route;
            route.tiled = tiled_route;
            route.http = options.http_serve;
            route.batched = batch_slots != 0U;
            route.batch_explicit = options.http_batch_explicit;
            route.batch_cap = options.http_batch_cap;
            route.cohort = options.mtp.cohort.size();
            route.kernel_plan = compiled.kernel_plan_identity;
            route.execution = compiled.execution_identity;
            route.profiles = strix::numerical_profile_pair_identity(profiles);
            route.optimization = optimization_profile.selected;
            route.cli = options.cost_table_cli;
            route.observed_slots = batch_slots;
            *cost_table_config = strix::mtp_cost_table_configuration(route);
        }
        using strix::decode_cli::prefill_checked_add;
        using strix::decode_cli::prefill_checked_mul;
        if(options.cache_activation->automatic_path && std::any_of(
            std::begin(strix::decode_cli::automatic_prefix_auto_route_qualifications),
            std::end(strix::decode_cli::automatic_prefix_auto_route_qualifications),[](const auto& row) {return row.qualified();})) try {
            const auto shared=prefill_checked_add(std::max(options.mtp.sidecar_retained_bytes,options.mtp.sidecar_setup_bytes),
                prefill_checked_add(options.mtp.golden_max_bytes,options.mtp.replay_host_bytes));
            if(batch_slots && cache_slot_retained && cache_slot_native) {
                cache_future_commitment=prefill_checked_add(strix::decode_cli::cache_unpaid_slot_bytes(
                    *cache_slot_retained,*cache_slot_native,cache_prompt_transient,batch_slots,1U+batch_stacks.size(),0),shared);
                cache_ready_commitment=strix::decode_cli::cache_unpaid_slot_bytes(
                    *cache_slot_retained,*cache_slot_native,cache_prompt_transient,batch_slots,eager_slots,eager_slots);
            } else {
                auto future=prefill_checked_add(options.mtp.runtime_bytes,options.mtp.window_total_bytes);
                future=prefill_checked_add(future,options.mtp.target_capture_bytes);
                const auto resident_native=future;
                const auto width=prefill_checked_mul(prefill_checked_mul(model.hidden_size,model.hc_count),2);
                const bool subset=!options.mtp.draft_vocab.empty() || options.mtp.draft_context_vocab;
                const auto row=subset?prefill_checked_mul(model.hidden_size/32U,options.mtp.draft_q4?18U:34U):0U;
                future=prefill_checked_add(future,prefill_checked_mul(width,2));
                future=prefill_checked_add(future,prefill_checked_mul(model.vocab_size,prefill_checked_add(row,77U)));
                future=prefill_checked_add(future,prefill_checked_add(prefill_checked_mul((model.vocab_size+255U)/256U,32U),4096U));
                future=prefill_checked_add(future,4U*(11U*640U+1U)+2U*4U*10240U*4U);
                const char* pipe=std::getenv("STRIX_PREFILL_PIPE");
                const auto copies=pipe && (std::strtoul(pipe,nullptr,0)&1U)?2U:1U;
                cache_prompt_transient=prefill_checked_mul(prefill_checked_mul(prefill_checked_mul(
                    options.known_prefix_prefill_rows,model.hidden_size),model.hc_count),2U*copies);
                cache_prompt_transient=prefill_checked_add(cache_prompt_transient,future-resident_native);
                cache_future_commitment=prefill_checked_add(prefill_checked_add(resident_native,shared),cache_prompt_transient);
                // CLI retains the full unpaid bound until its native setup completes.
                cache_ready_commitment=cache_prompt_transient;
            }
        } catch(const std::overflow_error&) { cache_future_commitment=cache_ready_commitment=UINT64_MAX; }
        strix::decode_cli::CacheFamilyIdentity cache_family;
        const auto cache_runner=!options.http_serve?strix::decode_cli::CacheRunner::cli:
            batch_slots?strix::decode_cli::CacheRunner::batch_mtp:strix::decode_cli::CacheRunner::single_mtp;
        const bool inspect_qualified_family=options.cache_activation->automatic_path && options.cache_activation->prefix_requested &&
            options.cache_activation->suppression.empty() && options.cache_activation->source!="off" &&
            std::any_of(std::begin(strix::decode_cli::automatic_prefix_auto_route_qualifications),
                std::end(strix::decode_cli::automatic_prefix_auto_route_qualifications),[&](const auto& row) {
                    return row.qualified() && row.runner==cache_runner && (batch_slots?batch_slots:1)>=row.n_min &&
                        (batch_slots?batch_slots:1)<=row.n_max && options.known_prefix_prefill_rows==row.tile;
                });
        if(inspect_qualified_family) {
            // Hash work is deferred until a release row exists. Paths never serve
            // as asset identity; unknown assets leave the family unmatched.
            cache_family=resolved_cache_family(options,model,manifest,mapped,cache_runner);
        }
        strix::decode_cli::finalize_cache_family(options,cache_runner,
            static_cast<std::uint32_t>(batch_slots?batch_slots:1),cache_family);
        if(prepare_prefix_state) prepare_prefix_state();
        if(strix::decode_cli::automatic_cache_work(options)) {
            if(enable_prefix_store) enable_prefix_store(false);
        }
        if(rtx_state_session && !prefix_state) parents->tiled.cache=rtx_state_session.get();
        if(prefix_state) {
            parents->tiled.cache=prefix_state.get();
            parents->tiled.cache_tails=prefix_cache_unit(options)<options.known_prefix_prefill_rows;
        }
        if(options.cache_activation->automatic_path && prefix_store) try {
            for(auto& stack:batch_stacks) {
                // The OFF measurement probe predates store admission. Attach only
                // its optional cache registry now; retain its original state/owners.
                stack->cache=std::make_unique<strix::decode_cli::HipPrefixStateCacheSession>(prefix_store,
                    prefix_cache_identity,prefix_cache_unit(options),prefix_cache_every,model,state_layout,stack->memory->state());
                stack->cache->set_staging_cap(options.prefix_cache_staging_bytes);
                stack->parents->tiled.cache=stack->cache.get();
                stack->parents->tiled.cache_tails=prefix_cache_unit(options)<options.known_prefix_prefill_rows;
            }
        } catch(const std::exception&) {
            // No cache copies have been submitted during startup registry setup.
            strix::decode_cli::decline_cache_activation(*options.cache_activation,"cache-setup-allocation-failed");
            options.prefix_cache_reason=options.cache_activation->reason; options.prefix_cache_host_bytes=0;
            parents->tiled.cache=rtx_state_session.get();
            for(auto& stack:batch_stacks) {
                if(stack->cache) stack->cache->disable_store_before_ready();
                stack->parents->tiled.cache=stack->cache.get();
            }
            prefix_state.reset(); prefix_store.reset(); options.state_cache_stats->publish({});
        }
        if(options.cache_activation->automatic_path && options.cache_activation->source=="auto" &&
            cache_runner==strix::decode_cli::CacheRunner::single_mtp &&
            options.cache_activation->automatic_qualified &&
            options.cache_activation->reason=="pending-post-residency") {
            options.cache_attach_ready=[&,parents] {
                if(enable_prefix_store) enable_prefix_store(true);
                if(prefix_state) {
                    parents->tiled.cache=prefix_state.get();
                    parents->tiled.cache_tails=prefix_cache_unit(options)<options.known_prefix_prefill_rows;
                }
            };
            options.cache_attach_rollback=[&,parents] {
                parents->tiled.cache=rtx_state_session.get();
                parents->tiled.cache_tails=false;
                prefix_state.reset(); prefix_store.reset();
                const auto reason=options.cache_activation->reason=="capture-pool-allocation-failed"?
                    "capture-pool-allocation-failed":"cache-setup-allocation-failed";
                strix::decode_cli::decline_cache_activation(*options.cache_activation,reason);
                options.prefix_cache_reason=options.cache_activation->reason;
                options.prefix_cache_host_bytes=0; options.state_cache_stats->publish({});
            };
        }
        options.cache_readiness_recheck=[&] {
            if(options.cache_activation->decision_phase!="post-residency-ready" &&
                strix::decode_cli::automatic_cache_work(options) && options.cache_activation->prefix && !apply_cache_fit(true) && prefix_state) {
                prefix_state->disable_store_before_ready();
                for(auto& stack:batch_stacks) if(stack->cache) stack->cache->disable_store_before_ready();
                prefix_store.reset();
                options.state_cache_stats->publish({});
            }
            std::cerr << "qwen cache policy: " << strix::decode_cli::cache_activation_json(options).dump() << '\n';
        };

        if (options.http_serve) {
            std::cerr << "http slots " << (batch_slots ? batch_slots : 1U) << " (";
            if (options.http_batch_explicit && !options.http_batch_cap) std::cerr << "explicit single route";
            else std::cerr << (options.http_batch_explicit ? "cap " + std::to_string(options.http_batch_cap) : "auto")
                << ": " << options.http_slot_reason;
            std::cerr << ") (" << (eager_slots ? eager_slots : 1U) << " built, "
                << (batch_slots - eager_slots) << " on demand)\n";
        }
        if (batch_slots) {
            batch_stream = std::make_shared<const SharedWindowStream>();
            const auto build_slot = [&, batch_stream]() -> strix::decode_cli::MtpBatchSlotSetup {
                const auto slot = batch_stacks.size() + 1U;
                const auto built = std::chrono::steady_clock::now();
                auto stack = make_stack();
                const auto stacked = std::chrono::steady_clock::now();
                add_tiled(*stack);
                batch_stacks.push_back(std::move(stack));
                // STRIX_HTTP_SLOT_TIMING=1: setup time of every extra slot (stack, tiled owners).
                static const bool timing = [] { const char* v = std::getenv("STRIX_HTTP_SLOT_TIMING"); return v && v[0] == '1'; }();
                if (timing) std::fprintf(stderr, "http slot %zu built: stack %.0f ms tiled %.0f ms\n", slot,
                    std::chrono::duration<double, std::milli>(stacked - built).count(),
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stacked).count());
                const auto& stack_owner = batch_stacks.back();
                return {stack_owner->parents, &stack_owner->session, &*stack_owner->dispatcher,
                    window_factory_for(stack_owner->parents, batch_stream)};
            };
            for (const auto& stack : batch_stacks)
                parents->batch_slots.push_back({stack->parents, &stack->session, &*stack->dispatcher,
                    window_factory_for(stack->parents, batch_stream)});
            while (batch_stacks.size() + 1U < eager_slots) parents->batch_slots.push_back(build_slot());
            parents->batch_capacity = batch_slots;
            parents->batch_memory_ok = batch_memory_ok;
            parents->grow_batch_slot = build_slot;
            parents->serve_batch = options.http_serve
                ? strix::decode_cli::make_http_batch_serve(options, "native MTP batched decode (" + std::to_string(batch_slots) +
                    " slots), profile " + std::string(optimization_profile.selected) + ", context " + std::to_string(options.context_tokens),
                    startup_preflight, release_assets, {}, rewarm_before_listen)
                : strix::decode_cli::make_cohort_tiled_run(options, model, std::cout);
        }
        if(strix::decode_cli::mtp_prompt_phase_separation(options)) {
            parents->prepare_prompt_pool = [&, p=parents.get()] { (void)prepare_mtp_pool(p); };
            parents->retire_prompt = [&, p=parents.get()] {
                // run_tiled_mtp_prompt has destroyed its tap pipeline and completed
                // priming/adoption. Clear every caller/parent pin, then the globals.
                p->tiled.reset_prefill_memory={};
                p->tiled.blocks=nullptr;p->tiled.block_owner.reset();p->tiled.tails.clear();
                known_prefix_backend.reset();known_tail_backends.clear();
                known_prefix_owner.reset();known_tail_owners.clear();
                known_prefix_memory.reset();known_tail_memories.clear();
                known_prefill_plan.reset();known_tail_plans.clear();known_prefix_plan.reset();
                strix::release_hip_qsa_prefill_storage();
            };
        }
        auto make_window = window_factory_for(parents, batch_stream);
        if (options.http_serve && !batch_slots)
            parents->serve = strix::decode_cli::make_http_serve(options, "native MTP greedy decode, profile " +
                std::string(optimization_profile.selected) + ", context " + std::to_string(options.context_tokens),
                parents->tiled.rtx_import, startup_preflight, release_assets, rewarm_before_listen);
        setup_phase.reset();
        // --mtp-sample-repeats N: N independent requests (seeds seed..seed+N-1), each from a fresh
        // session over the same loaded model; one receipt per request.
        const auto fresh_session = session;
        auto request_options = options;
        if (!options.mtp_bundle_path.empty()) {
            try {
                const auto identity=strix::decode_cli::mtp_bundle_golden_identity(options,model);
                const auto golden=strix::decode_cli::ensure_mtp_bundle_golden(options.mtp_bundle_path,identity,
                    [&](const std::filesystem::path& temp) {
                        auto dump_options=options;
                        dump_options.mtp_bundle_path.clear();
                        dump_options.mtp.golden.clear();dump_options.mtp.golden_sha256.clear();
                        dump_options.mtp.golden_dump=temp.string();
                        std::ostringstream ignored;
                        strix::decode_cli::run_mtp_decode_bridge(dump_options,model,compiled,parents,session,direct_dispatcher,
                            generations,catalog_config.deferred_moe_status,optimization_profile.selected,ignored,make_window);
                        // Restore precisely the fresh target state before generation.
                        if (parents->tiled.reset_state) parents->tiled.reset_state();
                        else strix::initialize_hip_fresh_runtime_memory(memory.scratch(),memory.state());
                        session=fresh_session;
                        if (ple_turn) {
                            ple_turn->finish();
                            ple_turn=ple_reclaimer->begin_turn();
                            mtp_mapped_ple_owner->bind_residency(ple_turn);
                        }
                    });
                request_options.mtp.golden=golden.string();
                request_options.mtp.golden_sha256=strix::model_content_identity_hex(
                    strix::sha256_file_content(golden,strix::FileHashPolicy::cached));
            } catch (const strix::decode_cli::YarnMemoryPressure&) { throw; }
            catch (const std::exception& error) {
                throw strix::decode_cli::YarnBundleCalibrationFailure(error.what());
            }
        }
        for (std::uint32_t repeat = 0; repeat < options.mtp.sample_repeats; ++repeat) {
            if (repeat) {
                if (ple_reclaimer) {
                    ple_turn = ple_reclaimer->begin_turn();
                    mtp_mapped_ple_owner->bind_residency(ple_turn);
                }
                strix::initialize_hip_fresh_runtime_memory(memory.scratch(), memory.state());
                session = fresh_session;
            }
            request_options.mtp.sample_seed = options.mtp.sample_seed + repeat;
            strix::decode_cli::run_mtp_decode_bridge(request_options, model, compiled, parents, session, direct_dispatcher,
                generations, catalog_config.deferred_moe_status, optimization_profile.selected, std::cout, make_window,
                [&] { return execution_inventory_receipt + ple_cache_completed_receipt() + dense_trellis_configuration_receipt + dense_trellis_completed_receipt() +
                    (ple_turn ? "  \"ple_reclaim\":" + ple_reclaimer->receipt_json() + ",\n" : std::string{}); }
#if defined(STRIX_MTP_MODEL_QUALIFICATION_RUNNER)
                , strix::tests::mtp_component_diagnostic_requested()
                    ? strix::decode_cli::MtpDecodeQualification{} : strix::decode_cli::MtpDecodeQualification{strix::tests::run_mtp_model_qualification}
                , strix::tests::mtp_component_diagnostic_requested()
                    ? strix::decode_cli::MtpDecodeComponentDiagnostic{strix::tests::run_mtp_component_diagnostic}
                    : strix::decode_cli::MtpDecodeComponentDiagnostic{}
                , strix::tests::mtp_prefix_gate_requested()
#else
                , {}, {}, false
#endif
                , [&] { bundle_retry.target_mutated=true; });
            if (ple_turn) ple_turn->finish();
            std::cout.flush();
        }
        if (yarn_watchdog) yarn_watchdog->complete();
        if (ple_reclaimer) ple_reclaimer->finish();
        if (prefix_store) prefix_store->flush();
        return 0;
    }

    std::unique_ptr<strix::PreparedEagerExecution> prepared_eager;
    if (options.prepared_eager) {
        strix::EagerExecutionRequest prototype;
        prototype.input_token_id = options.tokens.empty() ? 0U : options.tokens.front();
        prototype.position = session.next_position;
        prototype.state = session.state;
        const auto& current_resources = deferred_moe ? deferred_moe->resources() : resources;
        const auto owner = deferred_moe ? std::static_pointer_cast<const void>(deferred_moe)
                                        : std::static_pointer_cast<const void>(experiment_owner);
        prepared_eager = std::make_unique<strix::PreparedEagerExecution>(
            compiled, current_resources, prototype, owner);
    }
    const auto gdn_recurrent_layout_receipt = [&]() -> std::string {
        std::string storage;
        if (model.attention_kv_storage != strix::AttentionKvStorage::bf16)
            storage = std::string("  \"attention_kv_storage\":\"") + strix::attention_kv_storage_name(model.attention_kv_storage) +
                "\",\n  \"attention_kv_bytes_per_token\":" + std::to_string(strix::base_kv_bytes_per_token(model)) +
                ",\n  \"attention_kv_storage_qualification\":\"not_assessed\",\n";
        if (options.gdn_recurrent_storage != strix::GdnRecurrentStorage::key_value)
            storage += "  \"gdn_recurrent_storage\":\"value-key\",\n  \"gdn_recurrent_storage_qualification\":\"not_assessed\",\n";
        if (options.gdn_recurrent_layout == strix::HipGdnRecurrentLayout::baseline) return storage;
        return storage + "  \"gdn_recurrent_layout\": \"" +
            std::string(strix::gdn_recurrent_layout_marker(options.gdn_recurrent_layout)) + "\",\n"
            "  \"gdn_recurrent_layout_qualification\": \"not_assessed\",\n";
    };
    const auto qsa_gqa_receipt = [&]() -> std::string {
        if (!qsa_gqa_stats) return {};
        std::string result = "  \"qsa_gqa_heads_per_block\": " + std::to_string(options.qsa_gqa_heads_per_block) + ",\n"
            "  \"qsa_gqa_qualification\": \"not_assessed\",\n"
            "  \"qsa_gqa_submissions\": " + std::to_string(qsa_gqa_stats->submissions.load(std::memory_order_relaxed)) + ",\n";
        result += "  \"qsa_gqa_layer_submissions\": [";
        for (std::size_t layer = 0; layer < qsa_gqa_stats->layer_submissions.size(); ++layer) {
            if (layer) result += ',';
            result += std::to_string(qsa_gqa_stats->layer_submissions[layer].load(std::memory_order_relaxed));
        }
        return result + "],\n";
    };
    const auto qsa_selection_receipt = [&]() -> std::string {
        if (!qsa_selection_stats) return {};
        std::string result = "  \"qsa_hierarchical_selection\": true,\n"
            "  \"qsa_hierarchical_selection_qualification\": \"not_assessed\",\n"
            "  \"qsa_hierarchical_selection_submissions\": " +
            std::to_string(qsa_selection_stats->submissions.load(std::memory_order_relaxed)) + ",\n";
        result += "  \"qsa_hierarchical_selection_layer_submissions\": [";
        for (std::size_t layer = 0; layer < qsa_selection_stats->layer_submissions.size(); ++layer) {
            if (layer) result += ',';
            result += std::to_string(qsa_selection_stats->layer_submissions[layer].load(std::memory_order_relaxed));
        }
        return result + "],\n";
    };
    const auto gdn_islands_receipt = [&]() -> std::string {
        if (!gdn_islands) return {};
        const auto observed = gdn_islands->receipt();
        return "  \"gdn_islands\": true,\n  \"gdn_islands_qualification\": \"not_assessed\",\n" +
            std::string("  \"gdn_islands_organization_identity\": ") + std::to_string(observed.organization_identity) + ",\n" +
            "  \"gdn_islands_count\": " + std::to_string(observed.islands) + ",\n" +
            "  \"gdn_islands_kernel_nodes\": " + std::to_string(observed.kernel_nodes) + ",\n" +
            "  \"gdn_islands_setup_memory_observed\": " + (observed.memory_observed ? "true" : "false") + ",\n" +
            "  \"gdn_islands_setup_free_delta_bytes\": " + std::to_string(observed.observed_setup_free_delta_bytes) + ",\n" +
            "  \"gdn_islands_replays\": " + std::to_string(observed.replays) + ",\n";
    };
    const auto overlap_receipt = [&]() -> std::string {
        if (!moe_overlap) return {};
        const auto stats = moe_overlap->stats();
        return "  \"moe_shared_overlap_mode\":\"eager-sibling-streams-experimental-v1\",\n"
            "  \"moe_shared_overlap_layout_identity\":" + std::to_string(overlap_arena->layout_identity()) + ",\n" +
            "  \"moe_shared_overlap_source_execution_identity\":" + std::to_string(overlap_arena->source_execution_identity()) + ",\n" +
            "  \"moe_shared_overlap_arena_bytes\":" + std::to_string(compiled.arena.total_bytes) + ",\n" +
            "  \"moe_shared_overlap_additional_arena_bytes\":" + std::to_string(overlap_arena->additional_bytes()) + ",\n" +
            "  \"moe_shared_overlap_fork_records\":" + std::to_string(stats.fork_records) + ",\n" +
            "  \"moe_shared_overlap_join_waits\":" + std::to_string(stats.join_waits) + ",\n" +
            "  \"moe_shared_overlap_verified_tokens\":" + std::to_string(stats.verified_tokens) + ",\n" +
            "  \"moe_shared_overlap_committed_tokens\":" + std::to_string(stats.committed_tokens) + ",\n" +
            "  \"moe_shared_overlap_committed_layers\":" + std::to_string(stats.committed_layers) + ",\n" +
            "  \"moe_shared_overlap_aborted_tokens\":" + std::to_string(stats.aborted_tokens) + ",\n";
    };
    auto moe_token_schedule = strix::MoeTokenSchedule::fork_join;
    std::uint64_t input_access_epoch = 1;
    const std::function<void(std::uint32_t)> execute_native_token = [&](std::uint32_t token) {
        strix::decode_cli::check_yarn_memory();
        auto correlation = strix::current_execution_correlation();
        correlation.position = session.next_position;
        correlation.execution_identity = compiled.execution_identity;
        correlation.row = 0; correlation.rows = 1; correlation.step_id = UINT32_MAX;
        strix::ExecutionCorrelationScope token_correlation(correlation);
        strix::ExecutionPhaseScope token_phase(session.next_position < options.tokens.size() || !options.generate_tokens
            ? strix::ExecutionPhase::prefill : strix::ExecutionPhase::token);
        original_work_device->require_original();
        if (arena_pin) {
            if (input_access_epoch == UINT64_MAX) throw std::overflow_error("integer input epochs exhausted");
            arena_pin->advance_access(++input_access_epoch);
        }
        if (deferred_moe) {
            const auto committed_before = session.committed_tokens;
            deferred_moe->begin();
            try {
                strix::HipLaunchScope scope(deferred_moe->scope());
                if (moe_overlap) moe_overlap->begin_token(session, moe_token_schedule);
                const auto complete_before_commit = [&] {
                    if (moe_overlap) moe_overlap->verify_before_commit(
                        output_dispatcher && output_dispatcher->state_only_active());
                    if (device_queue_trace) device_queue_trace->require_healthy_before_commit();
                    deferred_moe->complete_before_commit();
                };
                if (prepared_eager) static_cast<void>(strix::execute_prepared_serial_token(
                    *prepared_eager, session, deferred_moe->resources(), token, dispatcher, complete_before_commit));
                else static_cast<void>(strix::execute_serial_token(compiled, session, deferred_moe->resources(),
                    token, dispatcher, complete_before_commit));
                if (moe_overlap) moe_overlap->mark_committed(session);
            } catch (...) {
                if (moe_overlap) moe_overlap->abort();
                deferred_moe->abort();
                if (session.committed_tokens != committed_before) {
                    session.health = strix::SerialSessionHealth::poisoned;
                    session.arithmetic_lineage.tainted = true;
                }
                throw;
            }
        } else if (graph) static_cast<void>(graph->execute(session, token));
        else if (prepared_eager) static_cast<void>(strix::execute_prepared_serial_token(
            *prepared_eager, session, resources, token, dispatcher));
        else static_cast<void>(strix::execute_serial_token(compiled, session, resources, token, dispatcher));
        // Deferred mode completed its owned stream inside the session's
        // prepublication boundary. Keep the legacy completion path unchanged.
        if (!deferred_moe) {
            check_hip(hipDeviceSynchronize(), "native token original-device completion");
            original_work_device->require_original();
        }
        strix::decode_cli::check_yarn_memory();
    };

    std::unique_ptr<strix::RawBaselineRecording> raw_recording;
    strix::RawBaselineStateReader raw_state_reader;
    std::function<void()> raw_require_identity = [original_work_device] {
        original_work_device->require_original();
    };
    std::function<void()> raw_drain_or_fail = [original_work_device] {
        original_work_device->recover_or_fail_stop();
    };
    std::string raw_model_sha, raw_executable_sha;
    std::filesystem::path raw_executable;
    std::optional<strix::FinalGenerationStateMetadata> final_state_metadata;
    if (!options.raw_recording_path.empty() || !options.validation_final_state_path.empty()) {
        if (const char* override_arch = std::getenv("HSA_OVERRIDE_GFX_VERSION"); override_arch && *override_arch) throw std::invalid_argument("RAW native target recording rejects architecture override");
#if defined(__linux__)
        raw_executable = "/proc/self/exe";
#else
        raw_executable = std::filesystem::canonical(argv[0]);
#endif
        raw_model_sha = strix::model_content_identity_hex(strix::sha256_file_content(options.model_path));
        raw_executable_sha = strix::model_content_identity_hex(strix::sha256_file_content(raw_executable));
        int device = 0, runtime_version = 0, driver_version = 0;
        hipDeviceProp_t properties{}; hipUUID uuid{};
        check_hip(hipGetDevice(&device), "hipGetDevice(raw recording)");
        check_hip(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties(raw recording)");
        check_hip(hipDeviceGetUuid(&uuid, device), "hipDeviceGetUuid(raw recording)");
        check_hip(hipRuntimeGetVersion(&runtime_version), "hipRuntimeGetVersion(raw recording)");
        check_hip(hipDriverGetVersion(&driver_version), "hipDriverGetVersion(raw recording)");
        if (runtime_version <= 0 || driver_version <= 0) throw std::runtime_error("RAW HIP runtime/driver version unavailable");
        constexpr char hex[] = "0123456789abcdef"; std::string gpu_uuid;
        for (const auto byte : uuid.bytes) { const auto value = static_cast<unsigned char>(byte); gpu_uuid += hex[value >> 4]; gpu_uuid += hex[value & 15]; }
        const auto expected_state_owner = memory.state();
        const std::string expected_architecture = properties.gcnArchName;
        raw_require_identity = [&, device, runtime_version, driver_version, gpu_uuid, expected_architecture, expected_state_owner] {
            int actual_device = 0, actual_runtime = 0, actual_driver = 0; hipUUID actual_uuid{}; hipDeviceProp_t actual_properties{};
            check_hip(hipGetDevice(&actual_device), "hipGetDevice(raw binding)");
            check_hip(hipRuntimeGetVersion(&actual_runtime), "hipRuntimeGetVersion(raw binding)");
            check_hip(hipDriverGetVersion(&actual_driver), "hipDriverGetVersion(raw binding)");
            check_hip(hipDeviceGetUuid(&actual_uuid, actual_device), "hipDeviceGetUuid(raw binding)");
            check_hip(hipGetDeviceProperties(&actual_properties, actual_device), "hipGetDeviceProperties(raw binding)");
            std::string actual_hex; constexpr char digits[] = "0123456789abcdef";
            for (auto byte : actual_uuid.bytes) { const auto value = static_cast<unsigned char>(byte); actual_hex += digits[value >> 4]; actual_hex += digits[value & 15]; }
            const auto& actual_owner = memory.state();
            if (actual_device != device || actual_runtime != runtime_version || actual_driver != driver_version || actual_hex != gpu_uuid ||
                std::string(actual_properties.gcnArchName) != expected_architecture || actual_owner.layout_identity != expected_state_owner.layout_identity ||
                actual_owner.device_base != expected_state_owner.device_base || actual_owner.device_bytes != expected_state_owner.device_bytes ||
                actual_owner.device_allocation_generation != expected_state_owner.device_allocation_generation || actual_owner.host_base != expected_state_owner.host_base ||
                actual_owner.host_bytes != expected_state_owner.host_bytes || actual_owner.host_allocation_generation != expected_state_owner.host_allocation_generation)
                throw std::runtime_error("RAW native device/runtime/state owner identity changed");
            if (const char* override_arch = std::getenv("HSA_OVERRIDE_GFX_VERSION"); override_arch && *override_arch) throw std::runtime_error("RAW architecture override changed");
        };
        raw_drain_or_fail = [device, require_original_binding = raw_require_identity] {
            try {
                check_hip(hipSetDevice(device), "hipSetDevice(raw exact drain)");
                require_original_binding();
                check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(raw exact drain)");
                require_original_binding();
            } catch (...) {
                std::fputs("RAW capture failed with unproved original GPU quiescence; fail-stop, no finalized recording\n", stderr);
                std::fflush(nullptr); std::_Exit(3);
            }
        };
        strix::RawBaselineMetadata metadata;
        metadata.model_sha256 = raw_model_sha;
        metadata.model_content_identity = strix::model_content_identity_hex(model_residency.binding().model_content_identity);
        metadata.executable_sha256 = raw_executable_sha;
        metadata.gpu_identity = gpu_uuid; metadata.gpu_architecture = properties.gcnArchName;
        metadata.hip_runtime_version = static_cast<std::uint32_t>(runtime_version); metadata.hip_driver_version = static_cast<std::uint32_t>(driver_version);
        metadata.execution_identity = compiled.execution_identity; metadata.kernel_plan_identity = compiled.kernel_plan_identity;
        metadata.runtime_compatibility_identity = identities.compatibility_identity; metadata.vocabulary_size = model.vocab_size;
        metadata.supplied_tokens = options.tokens; metadata.maximum_generated_tokens = options.generate_tokens;
        metadata.mode = options.generate_tokens ? strix::RawBaselineMode::greedy : strix::RawBaselineMode::teacher_forced;
        if (options.generate_tokens) metadata.eos_token = generation_eos;
        metadata.evidence = graph_mode ? strix::RawBaselineEvidence::native_graph : integer_config
            ? strix::RawBaselineEvidence::native_integer_eager : strix::RawBaselineEvidence::native_eager;
        if (buffer_request.activation_element_bytes != 2) throw std::logic_error("RAW BF16 catalog/storage binding changed");
        metadata.activation_storage = strix::RawBaselineActivationStorage::bf16;
        metadata.execution_mode = graph_mode ? strix::RawBaselineExecutionMode::graph_float : integer_config
            ? strix::RawBaselineExecutionMode::eager_integer_lm_head : strix::RawBaselineExecutionMode::eager_float;
        metadata.numerical_identity = integer_config ? integer_config->approved_numerical_identity
            : strix::raw_floating_lm_head_numerical_identity(metadata.activation_storage);
        if (options.q4k_a8g32_gate_up)
            metadata.numerical_identity += ";" + std::string((a8_down_q51 || a8_down_q80)
                ? strix::moe_q4k_a8::a8_down_gate_up_raw_identity : strix::moe_q4k_a8::runtime_numerical_identity);
        if(options.grouped_exact_q51_down && !a8_down_q51)
            metadata.numerical_identity += ";" + std::string(strix::moe_q4k_a8::grouped_q51_raw_identity);
        if (a8_down_q51)
            metadata.numerical_identity += ";" + std::string(strix::moe_q4k_a8::q51_a8_down_raw_identity);
        if (a8_down_q80)
            metadata.numerical_identity += ";" + std::string(strix::moe_q4k_a8::q80_a8_down_raw_identity);
        if (q5k_a8_gateup)
            metadata.numerical_identity += ";" + std::string(strix::moe_q4k_a8::q5k_gate_up_raw_identity);
        if (dense_a8)
            metadata.numerical_identity += ";dense-a8-q80-v1";
        if (catalog_config.splitk_hyper_projection)
            metadata.numerical_identity += ";" + std::string(hc_deferred_norm ? strix::deferred_hc_numerical_identity :
                catalog_config.splitk_hyper_projection_fp64 ? strix::splitk_hc_fp64_numerical_identity : strix::splitk_hc_fp32_numerical_identity);
        if (metadata.numerical_identity.size() > 512U)
            throw std::invalid_argument("RAW numerical identity exceeds the bounded descriptor ABI");
        strix::RawBaselinePolicy policy;
        policy.maximum_bytes = options.raw_maximum_bytes;
        policy.maximum_steps = options.tokens.size();
        if (options.generate_tokens) policy.maximum_steps = options.generate_tokens - 1 >= options.context_tokens - options.tokens.size()
            ? options.context_tokens : options.tokens.size() + options.generate_tokens - 1;
        policy.state_policy = options.raw_state_policy;
        policy.allow_prefix_restore = options.prefix_cache_roundtrip;
        raw_state_reader = [&](const strix::SessionStateAllocation& allocation, std::uint64_t offset, std::span<std::byte> destination) {
            raw_require_identity();
            const auto address = strix::resolve_session_state(state_layout, memory.state(), allocation.component);
            if (offset > address.bytes || destination.size() > address.bytes - offset || address.address > std::numeric_limits<std::uintptr_t>::max() - offset)
                throw std::out_of_range("RAW copy exceeds actual state allocation");
            const auto* source = reinterpret_cast<const void*>(address.address + offset);
            if (address.storage == strix::SessionStateStorage::device)
                check_raw_copy(hipMemcpy(destination.data(), source, destination.size(), hipMemcpyDeviceToHost), "hipMemcpy(raw state D2H)", raw_drain_or_fail);
            else std::memcpy(destination.data(), source, destination.size());
            raw_require_identity();
        };
        if (!options.validation_final_state_path.empty()) {
            final_state_metadata.emplace();
            auto& final = *final_state_metadata;
            final.model_sha256 = metadata.model_sha256;
            final.model_content_identity = metadata.model_content_identity;
            final.executable_sha256 = metadata.executable_sha256;
            final.numerical_identity = metadata.numerical_identity;
            if (compiled_prefill) final.numerical_identity += ";prefill=" +
                std::to_string(compiled_prefill->execution_identity) + ";" +
                strix::prefill_identity_name(effective.prefill_option_mask);
            final.output_mode = options.gpu_greedy_token ? std::string(strix::gpu_greedy_output_identity) : "full-logits-cpu-firstmax";
            final.gpu_identity = metadata.gpu_identity;
            final.gpu_architecture = metadata.gpu_architecture;
            final.hip_runtime_version = metadata.hip_runtime_version;
            final.hip_driver_version = metadata.hip_driver_version;
            final.kernel_plan_identity = compiled.kernel_plan_identity;
            final.launch_plan_identity = launch_plan.plan_identity;
            final.runtime_binding_identity = identities.binding_identity;
            final.prepared_eager = options.prepared_eager;
            final.shared_overlap = options.moe_shared_overlap;
            final.batched_moe_activation = options.batched_moe_activation;
        }
        if (!options.raw_recording_path.empty()) {
            raw_recording = std::make_unique<strix::RawBaselineRecording>(options.raw_recording_path, std::move(metadata), policy, state_layout);
            raw_recording->capture_initial(session, raw_state_reader);
        }
    }
    const auto finalize_raw = [&](std::span<const std::uint32_t> generated = {}, std::string stop = "raw_input_exhausted") {
        if (!raw_recording) return;
        raw_recording->finalize(session, raw_state_reader, generated, std::move(stop), [&] {
            raw_require_identity();
            if (strix::model_content_identity_hex(strix::sha256_file_content(options.model_path)) != raw_model_sha ||
                strix::model_content_identity_hex(strix::sha256_file_content(raw_executable)) != raw_executable_sha)
                throw std::runtime_error("RAW model/executable identity changed during capture");
            raw_require_identity();
        });
    };

    setup_phase.reset();
    if (options.generate_tokens != 0 || options.known_prefix_window) {
        if (options.prefix_cache_roundtrip) {
            HipSerialGreedyStepBackend
                prefill_backend(
                    compiled,
                    session,
                    resources,
                    dispatcher,
                    memory.scratch(),
                    lm_io,
                    model.vocab_size,
                    [&](std::uint32_t token, std::uint64_t position, std::span<const float> values) {
                        if (raw_recording) raw_recording->record_token(token, position, values, session, raw_state_reader);
                    }, raw_require_identity, raw_drain_or_fail, execute_native_token, false, options.deferred_moe_status);

            std::vector<float> prefix_logits;
            for (std::size_t index = 0;
                 index < options.tokens.size();
                 ++index) {
                prefix_logits =
                    prefill_backend.execute(
                        options.tokens[index],
                        index);
            }

            GuardedHipSnapshotTransfer snapshot_transfer(raw_require_identity, raw_drain_or_fail);
            // Admit the retained snapshot budget BEFORE capture allocates it.
            // This route holds one entry, so no second capture can overlap it.
            strix::SerialSessionSnapshotRequest snapshot_request;
            snapshot_request.compiled_execution_identity = session.execution_identity;
            snapshot_request.runtime_compatibility_identity = session.runtime_compatibility_identity;
            snapshot_request.next_position = session.next_position;
            snapshot_request.committed_tokens = session.committed_tokens;
            snapshot_request.state = session.state;
            snapshot_request.alignment = 64;
            const auto snapshot_plan = strix::build_serial_session_snapshot_plan(model, state_layout, snapshot_request);
            if (snapshot_plan.device_bytes > UINT64_MAX - snapshot_plan.host_bytes)
                throw std::overflow_error("prefix snapshot retained budget overflows");
            const auto snapshot_budget = snapshot_plan.device_bytes + snapshot_plan.host_bytes;
            strix::SerialPrefixCache prefix_pool(1, snapshot_budget);
            auto captured_prefix =
                strix::SerialPrefixCacheEntry::capture(
                    model,
                    options.tokens,
                    session,
                    state_layout,
                    memory.state(),
                    allocator,
                    generations,
                    snapshot_transfer);
            check_raw_copy(hipDeviceSynchronize(), "hipDeviceSynchronize(prefix capture)", raw_drain_or_fail);
            raw_require_identity();
            if (prefix_pool.insert(std::move(captured_prefix), experiment_owner) != strix::SerialPrefixCacheAdmission::inserted)
                throw std::logic_error("prefix snapshot failed bounded-cache admission");
            auto prefix_handle = prefix_pool.find(options.tokens, session.execution_identity,
                session.runtime_compatibility_identity, session.state_layout_identity, &session.arithmetic_lineage);
            if (!prefix_handle) throw std::logic_error("fresh bounded prefix snapshot did not match its source");
            // The aliasing handle retains snapshot, allocator and original backing
            // owners until every guarded transfer has completed or drained.
            const auto& prefix_cache = *prefix_handle;
            if (raw_recording) {
                raw_recording->capture_prefix_restore_source(prefix_cache.prefix(), session, raw_state_reader);
            }

            strix::SerialGreedyContinuationRequest
                continuation;
            continuation.initial_logits =
                prefix_logits;
            continuation.start_position =
                session.next_position;
            continuation.vocab_size =
                model.vocab_size;
            continuation.context_capacity =
                options.context_tokens;
            continuation.max_new_tokens =
                options.generate_tokens;
            continuation.eos_token_id =
                generation_eos;

            HipSerialGreedyStepBackend
                baseline_backend(
                    compiled,
                    session,
                    resources,
                    dispatcher,
                    memory.scratch(),
                    lm_io,
                    model.vocab_size, {}, raw_require_identity, raw_drain_or_fail, execute_native_token, true, options.deferred_moe_status);
            const auto baseline =
                strix::execute_serial_greedy_continuation(
                    continuation,
                    baseline_backend);

            const auto mutated_position =
                session.next_position;
            if (mutated_position <= options.tokens.size())
                throw std::runtime_error("prefix-cache roundtrip exercised no mutation before restore");
            const auto restore_start =
                std::chrono::steady_clock::now();
            prefix_cache.restore(
                session,
                model,
                state_layout,
                memory.state(),
                snapshot_transfer,
                options.tokens);
            check_raw_copy(
                hipDeviceSynchronize(),
                "hipDeviceSynchronize(prefix restore)", raw_drain_or_fail);
            const auto restore_stop =
                std::chrono::steady_clock::now();

            if (session.next_position !=
                    options.tokens.size() ||
                session.committed_tokens !=
                    options.tokens.size() ||
                session.health !=
                    strix::SerialSessionHealth::ready) {
                throw std::logic_error(
                    "prefix cache restore did not recover exact session metadata");
            }
            if (raw_recording) raw_recording->record_prefix_restoration(prefix_cache.prefix(), session, raw_state_reader, mutated_position);

            HipSerialGreedyStepBackend
                restored_backend(
                    compiled,
                    session,
                    resources,
                    dispatcher,
                    memory.scratch(),
                    lm_io,
                    model.vocab_size,
                    [&](std::uint32_t token, std::uint64_t position, std::span<const float> values) {
                        if (raw_recording) raw_recording->record_token(token, position, values, session, raw_state_reader);
                    }, raw_require_identity, raw_drain_or_fail, execute_native_token, true, options.deferred_moe_status);
            const auto restored =
                strix::execute_serial_greedy_continuation(
                    continuation,
                    restored_backend);

            const bool equivalent =
                baseline.generated_tokens ==
                    restored.generated_tokens &&
                baseline.stop_reason ==
                    restored.stop_reason &&
                baseline.final_position ==
                    restored.final_position &&
                baseline_backend.logit_hashes() == restored_backend.logit_hashes();

            if (!equivalent) {
                throw std::runtime_error(
                    "prefix cache round-trip changed complete logits or greedy continuation");
            }
            finalize_raw(restored.generated_tokens, std::string(stop_reason_name(restored.stop_reason)));

            double prefill_ms = 0.0;
            for (const auto value :
                 prefill_backend.elapsed_ms()) {
                prefill_ms += value;
            }
            double baseline_ms = 0.0;
            for (const auto value :
                 baseline_backend.elapsed_ms()) {
                baseline_ms += value;
            }
            double restored_ms = 0.0;
            for (const auto value :
                 restored_backend.elapsed_ms()) {
                restored_ms += value;
            }
            const auto restore_ms =
                std::chrono::duration<double, std::milli>(
                    restore_stop -
                    restore_start).count();

            std::cout << "{\n";
            emit_q51_slot_receipt();
            emit_q51_slot_counts();
            std::cout
                << optimization_profile_receipt << dense_trellis_completed_receipt() << eager_submission_receipt << overlap_receipt() << execution_inventory_receipt << gdn_output_mode_receipt() << wave_router_mode_receipt << deferred_moe_mode_receipt
                << strix::decode_detail::qsa_input_dispatch_receipt(qsa_prefetch_stats.get())
                << shared_dispatch_receipt.completed_counts()
                << ple_cache_completed_receipt() << qsa_gqa_receipt() << gdn_recurrent_layout_receipt() << qsa_selection_receipt() << gdn_islands_receipt() << strix::decode_detail::deferred_moe_completed_tokens_receipt(deferred_moe.get())
                << "  \"model_content_identity\": \""
                << strix::model_content_identity_hex(
                       model_residency.binding().
                           model_content_identity)
                << "\",\n"
                << "  \"execution_identity\": "
                << compiled.execution_identity
                << ",\n"
                << "  \"runtime_compatibility_identity\": "
                << identities.compatibility_identity
                << ",\n"
                << "  \"prefix_cache_max_entries\":1,\n"
                << "  \"prefix_cache_snapshot_byte_budget\":" << snapshot_budget << ",\n"
                << "  \"prefix_cache_retained_snapshot_bytes\":" << prefix_pool.snapshot_bytes() << ",\n"
                << "  \"prefix_identity\": "
                << prefix_cache.prefix().
                       prefix_identity
                << ",\n"
                << "  \"prefix_tokens\": [";

            for (std::size_t index = 0;
                 index < options.tokens.size();
                 ++index) {
                if (index != 0) {
                    std::cout << ',';
                }
                std::cout
                    << options.tokens[index];
            }

            std::cout
                << "],\n"
                << "  \"baseline_generated_tokens\": [";
            for (std::size_t index = 0;
                 index <
                    baseline.generated_tokens.size();
                 ++index) {
                if (index != 0) {
                    std::cout << ',';
                }
                std::cout
                    << baseline.generated_tokens[index];
            }
            std::cout
                << "],\n"
                << "  \"restored_generated_tokens\": [";
            for (std::size_t index = 0;
                 index <
                    restored.generated_tokens.size();
                 ++index) {
                if (index != 0) {
                    std::cout << ',';
                }
                std::cout
                    << restored.generated_tokens[index];
            }

            std::cout
                << "],\n"
                << "  \"continuation_equivalent\": true,\n"
                << "  \"equivalence_scope\": \"tokens_stop_position_and_complete_continuation_logits\",\n"
                << "  \"baseline_continuation_logits_sha256\": ";
            write_logit_hashes(baseline_backend.logit_hashes());
            std::cout << ",\n  \"restored_continuation_logits_sha256\": ";
            write_logit_hashes(restored_backend.logit_hashes());
            std::cout << ",\n"
                << "  \"mutated_position_before_restore\": "
                << mutated_position
                << ",\n"
                << "  \"restored_prefix_position\": "
                << options.tokens.size()
                << ",\n"
                << "  \"stop_reason\": \""
                << stop_reason_name(
                       restored.stop_reason)
                << "\",\n"
                << "  \"prefill_elapsed_ms\": "
                << prefill_ms
                << ",\n"
                << "  \"baseline_continuation_elapsed_ms\": "
                << baseline_ms
                << ",\n"
                << "  \"restore_elapsed_ms\": "
                << restore_ms
                << ",\n"
                << "  \"restored_continuation_elapsed_ms\": "
                << restored_ms
                << "\n"
                << "}\n";
            emit_device_queue_trace();
            return 0;
        }

        std::unique_ptr<strix::HipGreedyOutput> greedy_output;
        if (options.gpu_greedy_token) {
            if (model.vocab_size > std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("GPU greedy vocabulary exceeds uint32");
            greedy_output = std::make_unique<strix::HipGreedyOutput>(
                static_cast<std::uint32_t>(model.vocab_size), memory.scratch(),
                std::static_pointer_cast<const void>(memory_owner),
                original_work_device);
        }
        std::function<void(std::uint32_t, std::uint64_t, std::span<const float>)> greedy_recording;
        const char* prefill_digest_env=std::getenv("STRIX_PREFILL_LOGITS_DIGEST");
        const bool prefill_digest=prefill_digest_env && prefill_digest_env[0]=='1';
        if (raw_recording || prefill_digest) greedy_recording = [&](std::uint32_t token, std::uint64_t position,
                                                   std::span<const float> values) {
            if(raw_recording) raw_recording->record_token(token, position, values, session, raw_state_reader);
            // Include unpadded serial suffixes in the full-vocabulary prefill gate.
            if(prefill_digest && position<options.tokens.size()) {
                std::uint64_t hash=14695981039346656037ULL;
                for(const auto byte:std::as_bytes(values)) {hash^=static_cast<unsigned char>(byte);hash*=1099511628211ULL;}
                std::fprintf(stderr,"prefill terminal logits fnv64 %016llx position %llu\n",
                    static_cast<unsigned long long>(hash),static_cast<unsigned long long>(position));
            }
        };
        HipSerialGreedyStepBackend backend(
            compiled,
            session,
            resources,
            dispatcher,
            memory.scratch(),
            lm_io,
            model.vocab_size,
            std::move(greedy_recording), raw_require_identity, raw_drain_or_fail,
            execute_native_token, false, options.deferred_moe_status, greedy_output.get(), output_dispatcher.get());

        strix::SerialStepLogitsDiagnostic serial_step_ledger(options.tokens.size()-1U,model.vocab_size);
        if(strix::mtp_step_logits_digest_enabled()) backend.set_logits_observer(
            [&](std::uint64_t p,std::span<const float> values){serial_step_ledger.row(p,values);});
        strix::SerialGreedyRequest request;
        request.prefix_tokens =
            options.tokens;
        request.vocab_size =
            model.vocab_size;
        request.context_capacity =
            options.context_tokens;
        request.max_new_tokens =
            options.session_first_generate ? options.session_first_generate : options.generate_tokens;
        request.eos_token_id =
            generation_eos;
        bitcalc.prepare(model.vocab_size, options.tokens.size(), options.generate_tokens);
        request.emission_hook = bitcalc.hook();

        std::unique_ptr<RequestedLogprobBackend> requested_logprobs;
        if (!options.logprob_token_ids.empty() || options.teacher_forced_score)
            requested_logprobs = std::make_unique<RequestedLogprobBackend>(backend, options.logprob_token_ids,
                model.vocab_size, options.context_tokens, session, original_work_device,
                options.teacher_forced_score ? std::span<const std::uint32_t>{options.tokens} : std::span<const std::uint32_t>{},
                options.teacher_logits_path, model.effective_max_positions());
        strix::SerialGreedyStepBackend* generation_backend = requested_logprobs
            ? static_cast<strix::SerialGreedyStepBackend*>(requested_logprobs.get())
            : static_cast<strix::SerialGreedyStepBackend*>(&backend);
        std::unique_ptr<strix::MoeTokenCostBackend> token_cost_backend;
        if (options.moe_token_cost_policy) {
            token_cost_backend = std::make_unique<strix::MoeTokenCostBackend>(*generation_backend,
                [&]() {
                    strix::MoeTokenCostContext context;
                    context.model_content = model_residency.binding().model_content_identity;
                    context.execution_identity = compiled.execution_identity;
                    context.kernel_plan_identity = compiled.kernel_plan_identity;
                    context.runtime_compatibility_identity = session.runtime_compatibility_identity;
                    // Actual immutable resolved catalog/config plan, not a cache-provided identity.
                    context.resolved_profile_identity = compiled.kernel_plan_identity;
                    context.position = session.next_position; context.context_capacity = options.context_tokens;
                    context.committed_tokens = session.committed_tokens; context.vocab_size = model.vocab_size;
                    context.eos_token_id = generation_eos; context.cancelled = false;
                    context.ready = session.health == strix::SerialSessionHealth::ready && !session.arithmetic_lineage.tainted;
                    return context;
                }, [&](strix::MoeTokenSchedule mode) { moe_token_schedule = mode; });
            generation_backend = token_cost_backend.get();
        }
        std::unique_ptr<strix::HipPromptLookupRuntime> lookup_runtime;
        struct LookupStats {
            std::uint64_t cycles = 0, proposed = 0, accepted = 0, discarded = 0, replayed = 0;
            std::uint64_t snapshots = 0, restores = 0, consumed_bonuses = 0;
            bool saturated = false;
            void add(std::uint64_t& counter, std::uint64_t value) noexcept {
                if (value > UINT64_MAX - counter) { counter = UINT64_MAX; saturated = true; }
                else counter += value;
            }
        } lookup_stats;
        if (options.known_prefix_window || options.prompt_lookup) {
            // Teacher-forced scoring: every row of every block is scored and spooled as it commits.
            if (options.teacher_forced_score) {
                if (!known_prefill_plan || !known_prefix_backend || !requested_logprobs ||
                    known_prefill_plan->output_policy != strix::KnownWindowOutputPolicy::all_rows)
                    throw std::logic_error("teacher-forced score reached a known-prefix route other than the tiled all-rows prefill");
                known_prefix_backend->set_all_rows_sink([target = requested_logprobs.get()](std::span<const std::uint32_t> tokens,
                    std::uint64_t position, std::span<const float> logits) { target->record_block(tokens, position, logits); },
                    [target = requested_logprobs.get()] { target->drain_block_spool(); });
            }
            if (options.prompt_lookup)
                lookup_runtime = std::make_unique<strix::HipPromptLookupRuntime>(compiled, model, catalog_config,
                    resources, *known_prefix_plan, known_prefix_owner, known_prefix_memory->scratch,
                    known_prefix_memory->serial_scope, known_prefix_memory->scope,
                    std::static_pointer_cast<const void>(known_prefix_memory), original_work_device, allocator, generations,
                    strix::HipPromptLookupAdaptivePolicy{.enabled = options.prompt_lookup_adaptive},
                    strix::HipPromptLookupSnapshotPolicy{options.prompt_lookup_snapshot, options.prompt_lookup_snapshot_bytes});
        }

        auto* session_registry = prefix_state ? prefix_state.get() : rtx_state_session.get();
        strix::SessionCompatibility session_compatibility;
        std::vector<std::uint32_t> session_history;
        std::optional<std::uint32_t> session_pending;
        std::uint64_t session_emitted = 0;
        std::vector<std::pair<std::filesystem::path, strix::ModelContentIdentity>> session_files;
        if (session_cache_enabled) {
            if (!session_registry) throw std::logic_error("session component registry missing");
            session_compatibility.model = strix::session_cached_file_hash(options.model_path);
            if (session_compatibility.model != session_model_digest)
                throw std::runtime_error("session model changed during model setup");
            session_compatibility.executable = strix::session_cached_file_hash("/proc/self/exe");
            session_files.emplace_back(options.model_path, session_compatibility.model);
            std::cerr << "session model_hash=reused_mapped_full_digest\n";
            session_files.emplace_back("/proc/self/exe", session_compatibility.executable);
            std::string session_assets = "target=" + strix::model_content_identity_hex(session_compatibility.model);
            for (const auto& path : {options.dense_q6p_pack_path, options.dense_trellis_pack_path,
                    options.offline_tuning_cache_path, options.offline_compile_provenance_path})
                if (!path.empty()) {
                    const auto digest = strix::session_cached_file_hash(path);
                    session_files.emplace_back(path, digest);
                    session_assets += ";asset=" + path + ":" + strix::model_content_identity_hex(digest);
                }
            session_compatibility.model = strix::sha256_bytes(std::as_bytes(std::span(session_assets.data(), session_assets.size())));
            hipDeviceProp_t properties{}; int runtime = 0, driver = 0;
            check_hip(hipGetDeviceProperties(&properties, original_work_device->device_index()), "session device properties");
            check_hip(hipRuntimeGetVersion(&runtime), "session runtime version");
            check_hip(hipDriverGetVersion(&driver), "session driver version");
            const std::string observed = ";gpu-arch=" + std::string(properties.gcnArchName) +
                ";gpu-name=" + properties.name + ";runtime=" + std::to_string(runtime) + ";driver=" + std::to_string(driver);
            session_compatibility.numerical_config = strix::session_numerical_config(options.session_effective_cli,model.rotary_identity()) + session_assets + observed +
                ";compiled-prefix-identity=" + strix::model_content_identity_hex(prefix_cache_identity) + ";rank=0;world=1;bf16-kv";
        }
        // Diagnostic rows are raw native FP32 vocabulary rows, beginning at the suffix's terminal input.
        class SessionLogitsBackend final : public strix::SerialGreedyStepBackend {
        public:
            strix::SerialGreedyStepBackend& inner_;
            std::ofstream file;
            bool recording = false;
            std::uint64_t rows_written = 0;
            explicit SessionLogitsBackend(strix::SerialGreedyStepBackend& b) : inner_(b) {}
            std::vector<float> execute(std::uint32_t token, std::uint64_t position) override {
                auto row = inner_.execute(token, position);
                if (recording && file.is_open()) {
                    file.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)));
                    if (!file) throw std::runtime_error("session logits write failed");
                    ++rows_written;
                }
                return row;
            }
        } session_logit_backend(*generation_backend);
        if (session_cache_enabled && !options.session_logits.empty()) {
            session_logit_backend.file.open(options.session_logits, std::ios::binary | std::ios::trunc);
            if (!session_logit_backend.file) throw std::runtime_error("cannot open --session-logits");
        }
        const auto run_session_suffix = [&](std::span<const std::uint32_t> suffix) {
            std::vector<std::uint32_t> inputs;
            if (session_pending) inputs.push_back(*session_pending);
            inputs.insert(inputs.end(), suffix.begin(), suffix.end());
            if (inputs.empty()) throw std::invalid_argument("session continuation requires a nonempty suffix or pending input");
            if (inputs.size() > options.context_tokens - session.next_position ||
                options.generate_tokens > options.context_tokens - session.next_position - inputs.size())
                throw std::invalid_argument("session suffix and generation exceed context");
            for (const auto token : inputs) if (token >= model.vocab_size)
                throw std::invalid_argument("session token exceeds vocabulary");
            const auto start = std::chrono::steady_clock::now();
            std::vector<float> logits;
            std::size_t i = 0;
            const auto rows = options.known_prefix_prefill_rows;
            const auto suffix_plan = strix::plan_session_suffix(session.next_position, inputs.size(), rows,
                options.session_tiled_suffix && static_cast<bool>(known_prefill_plan));
            const bool tiled = suffix_plan.tiled;
            const auto alignment = suffix_plan.serial_rows;
            const auto serial = static_cast<std::size_t>(alignment);
            for (; i < serial; ++i) {
                session_logit_backend.recording = i + 1 == inputs.size();
                logits = session_logit_backend.execute(inputs[i], session.next_position);
            }
            if (tiled) {
                strix::KnownPromptChunkRequest paired;
                paired.prompt_tokens = std::span<const std::uint32_t>(inputs).subspan(i);
                paired.start_position = session.next_position;
                paired.vocab_size = model.vocab_size; paired.context_capacity = options.context_tokens;
                paired.max_new_tokens = options.generate_tokens; paired.eos_token_id = generation_eos;
                paired.logical_rows = rows; paired.tiled_prefill = true;
                paired.output_policy = options.known_prefix_output_policy;
                paired.ladder_tail_rows = options.known_prefix_ladder_tail_rows;
                // Same boundary cursor as prefix restore; no asynchronous scheduling across observations.
                session_logit_backend.recording = false;
                strix::KnownPromptChunkCursor chunks(paired, *known_prefix_backend, session_logit_backend);
                for (auto& tail : known_tail_backends) chunks.set_tail_backend(*tail);
                while (chunks.advance()) {}
                const auto frontier = chunks.frontier();
                if (!frontier.valid || frontier.logits.size() != model.vocab_size)
                    throw std::logic_error("session tiled suffix lacks a completed terminal head");
                logits.assign(frontier.logits.begin(), frontier.logits.end());
                if (session_logit_backend.file.is_open()) {
                    session_logit_backend.file.write(reinterpret_cast<const char*>(logits.data()),
                        static_cast<std::streamsize>(logits.size() * sizeof(float)));
                    if (!session_logit_backend.file) throw std::runtime_error("session tiled logits write failed");
                    ++session_logit_backend.rows_written;
                }
                session_logit_backend.recording = true;
            }
            session_history.insert(session_history.end(), inputs.begin(), inputs.end());
            std::cerr << "session suffix_route=" << (tiled ? "aligned-tiled" : "serial")
                << " alignment_steps=" << (tiled ? alignment : 0U) << '\n';
            std::cerr << "session suffix_ms=" << std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count() << '\n';
            strix::SerialGreedyContinuationRequest continuation{logits, session.next_position, model.vocab_size,
                options.context_tokens, options.generate_tokens, generation_eos};
            const auto generation_start = std::chrono::steady_clock::now();
            auto generated = strix::execute_serial_greedy_continuation(continuation, session_logit_backend);
            std::cerr << "session generation_ms=" << std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - generation_start).count() << '\n';
            return generated;
        };
        const auto remember_session_generation = [&](const strix::SerialGreedyRequestResult& generated) {
            // Every emitted token except the last has been consumed by the ordinary greedy controller.
            if (!generated.generated_tokens.empty()) {
                session_history.insert(session_history.end(), generated.generated_tokens.begin(), generated.generated_tokens.end() - 1);
                session_pending = generated.generated_tokens.back();
            } else session_pending.reset();
            session_emitted += generated.generated_tokens.size();
            if (session_history.size() != session.next_position)
                throw std::runtime_error("session capture refused: generation frontier is not represented exactly");
        };
        // One tiled known-prefix request: restore the longest cached block-boundary prefix (prefix
        // state cache), prefill the rest in T-row blocks (capturing selected boundaries), generate.
        const auto run_known_prompt = [&](std::span<const std::uint32_t> prompt) -> strix::SerialGreedyRequestResult {
            strix::KnownPromptChunkRequest paired;
            paired.prompt_tokens = prompt;
            paired.start_position = session.next_position;
            paired.vocab_size = request.vocab_size; paired.context_capacity = request.context_capacity;
            paired.max_new_tokens = request.max_new_tokens; paired.eos_token_id = request.eos_token_id;
            paired.logical_rows = options.known_prefix_prefill_rows ? options.known_prefix_prefill_rows : options.known_prefix_rows;
            paired.tiled_prefill = static_cast<bool>(known_prefill_plan);
            paired.output_policy = options.known_prefix_output_policy;
            // Teacher-forced: no headless block, and the serial suffix rows are scored like serial rows.
            paired.all_row_heads = options.teacher_forced_score;
            paired.schedule_extensions = true;
            // A prefix-state cache with a defaulted schedule drops bit 2 (deferred blocks are no save/resume points).
            paired.schedule_defer = strix::qpf2k::cli_sched_defers(static_cast<bool>(prefix_state));
            paired.ladder_tail_rows = options.known_prefix_ladder_tail_rows;
            paired.emission_hook = request.emission_hook;
            strix::KnownPromptChunkCursor chunks(paired, *known_prefix_backend,
                options.teacher_forced_score ? *generation_backend : static_cast<strix::SerialGreedyStepBackend&>(backend));
            for (auto& tail : known_tail_backends) chunks.set_tail_backend(*tail);
            if(prefix_state) {
                strix::decode_cli::QwenTiledPlanConfig plan;
                plan.rows=paired.logical_rows;
                const char* sched=std::getenv("STRIX_QPF2K_SCHED");
                plan.schedule=static_cast<std::uint32_t>(strix::qpf2k::effective_sched(
                    sched?std::strtoul(sched,nullptr,0):0UL,paired.schedule_defer));
                plan.suppressed=strix::qpf2k::suppressed(strix::qpf2k::Item::sched,known_prefix_backend->gsq_model());
                const auto policy=[](strix::KnownWindowOutputPolicy p) {
                    return p==strix::KnownWindowOutputPolicy::state_only?strix::decode_cli::QwenPlanOutput::state_only:
                        p==strix::KnownWindowOutputPolicy::all_rows?strix::decode_cli::QwenPlanOutput::all_rows:
                        strix::decode_cli::QwenPlanOutput::terminal;
                };
                plan.block={plan.rows,policy(paired.output_policy),known_prefill_plan->prefill_execution_identity,
                    known_prefill_plan->prefill_kernel_plan_identity};
                for(std::size_t k=0;k<known_tail_plans.size();++k)
                    plan.tails.push_back({known_tail_backends[k]->logical_rows(),policy(known_tail_backends[k]->output_policy()),
                        known_tail_plans[k]->prefill_execution_identity,known_tail_plans[k]->prefill_kernel_plan_identity});
                plan.ladder=paired.ladder_tail_rows;
                plan.decode_execution=compiled.execution_identity;plan.decode_kernel=compiled.kernel_plan_identity;
                plan.prefill_execution=compiled_prefill?compiled_prefill->execution_identity:compiled.execution_identity;
                plan.prefill_kernel=compiled_prefill?compiled_prefill->kernel_plan_identity:compiled.kernel_plan_identity;
                const auto profiles=strix::decode_cli::numerical_profile_pair(options);
                plan.prefill_profile=strix::numerical_profile_identity(profiles.prefill);
                plan.decode_profile=strix::numerical_profile_identity(profiles.decode);
                plan.profile_pair=strix::numerical_profile_pair_identity(profiles);
                plan.initial_generations=session.state.generations;
                plan.valid_state=static_cast<std::uint32_t>(session.state.valid);
                for(const auto& step:compiled.execution.steps) {
                    const auto flags=static_cast<std::uint32_t>(step.writes_state);
                    plan.valid_state|=flags;
                    for(std::size_t f=0;f<plan.generations_per_row.size();++f)
                        if(flags&(1U<<f)) ++plan.generations_per_row[f];
                }
                prefix_state->set_serial_plan(plan,prompt.size(),chunks.cold_plan(),
                    [&](std::uint64_t p) {return chunks.can_resume_at(p);},!paired.all_row_heads);
                const auto restored=prefix_state->begin_request(session,prompt);
                if(restored) chunks.resume_at(restored);
            }
            // One complete configured window or one unpadded serial suffix row per call.
            const auto prefill_start = std::chrono::steady_clock::now();
            for (;;) {
                strix::decode_cli::check_yarn_memory();
                if (!chunks.advance()) break;
                strix::decode_cli::check_yarn_memory();
                if (prefix_state) {
                    prefix_state->observe_serial_unit(chunks.last_unit());
                    prefix_state->after_block(session,prompt);
                }
            }
            if (session_cache_enabled) std::cerr << "session initial_prefill_ms=" << std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - prefill_start).count() << '\n';
            const auto initial_generation_start = std::chrono::steady_clock::now();
            if(strix::mtp_step_logits_digest_enabled() && serial_step_ledger.empty())
                serial_step_ledger.row(session.next_position-1U,chunks.frontier().logits);
            if(session_registry && strix::decode_cli::prompt_state_digest_request()) {
                const auto frontier=chunks.frontier();
                const auto pending=strix::select_greedy_token(frontier.logits,model.vocab_size);
                std::fprintf(stderr,"qwen-target-state phase=prompt position=%llu sha256=%s\n",
                    static_cast<unsigned long long>(session.next_position),
                    strix::model_content_identity_hex(session_registry->target_state_digest(session.next_position,prompt,pending)).c_str());
            }
            const auto& generated = chunks.finish_greedy();
            if (session_cache_enabled) std::cerr << "session initial_generation_ms=" << std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - initial_generation_start).count() << '\n';
            known_prefix_progress = chunks.progress();
            known_prefix_result.emplace();
            known_prefix_result->consumed_prompt_tokens = known_prefix_progress->consumed_prompt_tokens -
                known_prefix_progress->restored_prompt_tokens;
            known_prefix_result->windows_executed = known_prefix_progress->windows_executed;
            known_prefix_result->serial_prompt_steps = known_prefix_progress->serial_tail_steps;
            known_prefix_result->generation = generated;
            if (!known_prefix_result->generation) throw std::logic_error("known prompt unexpectedly stopped before generation");
            auto total = *known_prefix_result->generation;
            if (known_prefix_result->consumed_prompt_tokens > std::numeric_limits<std::uint64_t>::max() - total.steps_executed)
                throw std::overflow_error("known-prompt total execution count overflow");
            total.steps_executed += known_prefix_result->consumed_prompt_tokens;
            return total;
        };
        const bool generation_request_timing_enabled =
            !raw_recording && !final_state_metadata && !options.teacher_forced_score;
        // Plans, streams, arenas, prepared tensor maps and pool reservations above already
        // precede this timer. Move only the remaining immutable first-block admission here;
        // this is startup work, not removed work. No prompt is read or kernel launched.
        if (options.early_setup && known_prefill_plan && known_prefix_owner &&
            !options.mtp.enabled && !options.http_serve && !session_cache_enabled && !prefix_state &&
            !rtx_state_session && options.tokens.size()>=known_prefill_plan->logical_rows &&
            session.next_position==0 && generation_request_timing_enabled) {
            const auto setup_start=std::chrono::steady_clock::now();
            // Use the existing sibling handoff on a copy; the live decode session and
            // persistent state stay at their original frontier until the real block.
            auto setup_session=session;
            if(compiled_prefill)
                strix::hand_off_serial_session(setup_session,{&compiled,&resources},{&*compiled_prefill,&prefill_resources});
            known_prefix_owner->prepare_prefill_geometry(setup_session);
            const char* timing=std::getenv("STRIX_PREFILL_TIMING");
            if(timing && timing[0]=='1')
                std::fprintf(stderr,"prefill startup static-setup %.2f ms (outside generation request)\n",
                    std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-setup_start).count());
        }
        const auto generation_request_start = generation_request_timing_enabled
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        auto result = [&]() -> strix::SerialGreedyRequestResult {
            if (session_cache_enabled && !options.session_load.empty()) {
                const auto restore_start = std::chrono::steady_clock::now();
                strix::SessionIoStats io;
                strix::SessionCheckpoint checkpoint;
                // Fresh availability includes retained model, prefill and live state backing.
                const auto available = strix::observe_linux_uma_memory();
                const auto host_limit = strix::session_checkpoint_envelope_limit(options.session_capture_host_bytes,
                    available.memory.observation_identity ? available.memory.mem_available_bytes.value_or(0U) : 0U);
                const auto maximum = std::min(host_limit, options.context_tokens *
                    (session_registry->row_bytes() + 8U) + session_registry->fixed_bytes() + (16U << 20));
                (void)strix::read_session_checkpoint(options.session_load, session_compatibility,
                    strix::decode_cli::hip_pinned_prefix_buffer, maximum, &io, &checkpoint);
                for (const auto token : checkpoint.frontier.tokens) if (token >= model.vocab_size)
                    throw std::runtime_error("session restore token exceeds vocabulary");
                if (checkpoint.frontier.pending && *checkpoint.frontier.pending >= model.vocab_size)
                    throw std::runtime_error("session restore pending token exceeds vocabulary");
                session_registry->restore_session(session, checkpoint);
                session_history = checkpoint.frontier.tokens; session_pending = checkpoint.frontier.pending;
                session_emitted = checkpoint.frontier.emitted;
                std::cerr << "session restore_ms=" << std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - restore_start).count() << " bytes=" << io.bytes << '\n';
                return run_session_suffix(options.tokens);
            }
            if (lookup_runtime) {
                // Keep ordinary inclusive-EOS / unconsumed-last-token semantics.
                // Only the existing speculation planner chooses target/accepted
                // tokens; this loop owns the user output budget, not a new verifier.
                if (request.prefix_tokens.empty() || session.next_position != 0 ||
                    request.prefix_tokens.size() > request.context_capacity ||
                    request.prefix_tokens.size() > compiled.max_position_embeddings)
                    throw std::invalid_argument("prompt lookup requires a fresh, in-capacity known prefix");
                const auto committed_before_lookup = session.committed_tokens;
                try {
                for (const auto token : request.prefix_tokens) execute_native_token(token);
                lookup_runtime->prime_from_serial_logits(session);
                strix::PromptLookupDraftBackend lookup({3, 4096}, request.prefix_tokens);
                auto draft = lookup.make_state(session);
                const std::array<std::uint32_t, 1> eos{request.eos_token_id};
                strix::SerialGreedyRequestResult generated;
                while (generated.generated_tokens.size() < request.max_new_tokens) {
                    strix::GreedyDraftExecutionRequest cycle_request;
                    cycle_request.max_candidate_depth = options.prompt_lookup_depth;
                    cycle_request.token_budget = static_cast<std::uint32_t>(std::min<std::uint64_t>(
                        request.max_new_tokens - generated.generated_tokens.size(), UINT32_MAX));
                    cycle_request.eos_tokens = eos;
                    const auto cycle = lookup_runtime->cycle(session, lookup, draft, cycle_request);
                    const auto& plan = cycle.target_cycle.plan;
                    if (plan.emitted_tokens.empty() || plan.emitted_tokens.size() > cycle_request.token_budget ||
                        !lookup_runtime->pending_bonus() || *lookup_runtime->pending_bonus() != plan.emitted_tokens.back())
                        throw std::logic_error("prompt lookup returned inconsistent pending output");
                    lookup_stats.add(lookup_stats.cycles, 1);
                    lookup_stats.add(lookup_stats.proposed, cycle.proposed_tokens.size());
                    lookup_stats.add(lookup_stats.accepted, plan.verification.accepted_matches);
                    lookup_stats.add(lookup_stats.discarded, plan.verification.candidate_states_to_crop);
                    lookup_stats.add(lookup_stats.replayed, cycle.target_cycle.replayed_tokens);
                    lookup_stats.add(lookup_stats.snapshots, cycle.target_cycle.snapshot_captured ? 1U : 0U);
                    lookup_stats.add(lookup_stats.restores, cycle.target_cycle.restored_pre_window ? 1U : 0U);
                    generated.generated_tokens.insert(generated.generated_tokens.end(), plan.emitted_tokens.begin(), plan.emitted_tokens.end());
                    generated.final_position = session.next_position;
                    generated.steps_executed = session.committed_tokens; // final logical input history, not transient verify work
                    if (plan.verification.stopped_on_eos) {
                        generated.stop_reason = strix::SerialRequestStopReason::eos;
                        lookup_runtime->finish_adaptive_cycle(session);
                        break;
                    }
                    if (session.next_position >= request.context_capacity) {
                        generated.stop_reason = strix::SerialRequestStopReason::context_capacity;
                        lookup_runtime->finish_adaptive_cycle(session);
                        break;
                    }
                    if (generated.generated_tokens.size() >= request.max_new_tokens) {
                        lookup_runtime->finish_adaptive_cycle(session);
                        break;
                    }
                    // Never consume the final emitted bonus on EOS, output-budget
                    // exhaustion or context exhaustion. History already contains it.
                    lookup_runtime->consume_pending_bonus(session, lookup, draft);
                    lookup_stats.add(lookup_stats.consumed_bonuses, 1);
                    lookup_runtime->finish_adaptive_cycle(session);
                }
                return generated;
                } catch (...) {
                    if (session.committed_tokens != committed_before_lookup) {
                        session.health = strix::SerialSessionHealth::poisoned;
                        session.arithmetic_lineage.tainted = true;
                    }
                    original_work_device->recover_or_fail_stop();
                    throw;
                }
            }
            if (!known_prefix_backend) return options.gpu_greedy_token
                ? strix::execute_serial_greedy_token_request(request, *generation_backend)
                : strix::execute_serial_greedy_request(request, *generation_backend);
            return run_known_prompt(options.tokens);
        }();
        serial_step_ledger.finish(session.next_position);
        if(session_registry && strix::decode_cli::prompt_state_digest_request()) {
            auto consumed=options.tokens;
            if(!result.generated_tokens.empty()) consumed.insert(consumed.end(),result.generated_tokens.begin(),result.generated_tokens.end()-1);
            std::optional<std::uint32_t> pending;
            if(!result.generated_tokens.empty()) pending.emplace(result.generated_tokens.back());
            std::fprintf(stderr,"qwen-target-state phase=final position=%llu sha256=%s\n",
                static_cast<unsigned long long>(session.next_position),
                strix::model_content_identity_hex(session_registry->target_state_digest(session.next_position,consumed,pending)).c_str());
        }
        if (session_cache_enabled) {
            if (options.session_load.empty()) session_history = options.tokens;
            remember_session_generation(result);
            if (!options.session_save.empty()) {
                for (const auto& [path, digest] : session_files)
                    if (strix::session_cached_file_hash(path) != digest)
                        throw std::runtime_error("session checkpoint refused: model/asset/executable changed during run");
                // Conservative peak: captured sections + vector capacity (up to 2x) + final envelope,
                // with 16 MiB metadata allowance per copy. Refuse before any host capture allocation.
                const auto units = session_registry->row_bytes() + 8U;
                const auto fixed = session_registry->fixed_bytes() + (16U << 20);
                const auto available = strix::observe_linux_uma_memory();
                const auto quarter_budget = strix::session_checkpoint_envelope_limit(options.session_capture_host_bytes,
                    available.memory.observation_identity ? available.memory.mem_available_bytes.value_or(0U) : 0U);
                if (fixed > quarter_budget || session.next_position > (quarter_budget - fixed) / units)
                    throw std::runtime_error("session capture exceeds host peak budget");
                std::cerr << "session capture_peak_bound_bytes=" << 4U * (fixed + session.next_position * units)
                    << " host_budget_bytes=" << options.session_capture_host_bytes << '\n';
                const auto start = std::chrono::steady_clock::now();
                strix::SessionFrontier frontier;
                frontier.consumed = session.next_position; frontier.accepted = session.next_position;
                frontier.tokens = session_history; frontier.pending = session_pending; frontier.emitted = session_emitted;
                auto checkpoint = session_registry->capture_session(session, session_compatibility, std::move(frontier));
                auto envelope = strix::encode_session_checkpoint(checkpoint, strix::decode_cli::hip_pinned_prefix_buffer);
                strix::SessionIoStats io;
                strix::publish_session_checkpoint(options.session_save, envelope, &io);
                std::cerr << "session save_stall_ms=" << std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count() << " write_fsync_ms=" << io.write_fsync_ms << " bytes=" << io.bytes
                    << " consumed=" << session.next_position << " pending="
                    << (session_pending ? std::to_string(*session_pending) : "none") << '\n';
            }
            if (!options.session_then_prompt.empty()) {
                const auto suffix = strix::decode_cli::read_prefix_cache_prompt(options.session_then_prompt);
                result = run_session_suffix(suffix);
                remember_session_generation(result);
            }
            if (session_logit_backend.file.is_open()) {
                session_logit_backend.file.flush();
                if (!session_logit_backend.file) throw std::runtime_error("session logits flush failed");
                std::cerr << "session logits_rows=" << session_logit_backend.rows_written << " vocabulary=" << model.vocab_size << '\n';
            }
            strix::SessionFrontier final_frontier;
            final_frontier.consumed = session.next_position; final_frontier.accepted = session.next_position;
            final_frontier.tokens = session_history; final_frontier.pending = session_pending; final_frontier.emitted = session_emitted;
            std::cerr << "session final_digest=" << strix::model_content_identity_hex(
                session_registry->complete_session_digest(session, final_frontier)) << '\n';
        }
        // The spool of a final tiled block (with its footer) completes before anything is reported.
        if (requested_logprobs) requested_logprobs->drain_block_spool();
        if (options.teacher_forced_score && (!requested_logprobs ||
            requested_logprobs->rows().size() != options.tokens.size() - 1U)) {
            session.health = strix::SerialSessionHealth::poisoned;
            session.arithmetic_lineage.tainted = true;
            original_work_device->recover_or_fail_stop();
            throw std::logic_error("teacher-forced score did not produce exactly N-1 next-token rows");
        }
        const auto generation_request_stop = generation_request_timing_enabled
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        const auto generation_request_elapsed_ms =
            std::chrono::duration<double, std::milli>(
                generation_request_stop - generation_request_start).count();
        // STRIX_PLE_RECLAIMABLE: last residency sample after the timed request (decode lookups included); strict throws.
        if (ple_reclaimer) ple_reclaimer->finish();
        finalize_raw(result.generated_tokens, std::string(stop_reason_name(result.stop_reason)));

        std::string tail_rows_text;
        for (const auto rows : options.known_prefix_tail_rows)
            tail_rows_text += (tail_rows_text.empty() ? "" : ",") + std::to_string(rows);
        double elapsed_total = 0.0;
        for (const auto value :
             backend.elapsed_ms()) {
            elapsed_total += value;
        }
        // Prefix state cache: the main request's statistics, then the follow-up requests of
        // --prefix-cache-then-prompt in this process (fresh session each, same cache).
        std::string prefix_cache_receipt;
        std::optional<strix::HipKnownPromptBackendStats> known_prefix_main_stats;
        if (prefix_state) {
            const auto request_json = [&](const strix::decode_cli::PrefixStateCacheRequestStats& r) {
                std::ostringstream out;
                out << "\"prompt_tokens\":" << r.prompt_tokens << ",\"restored_position\":" << r.restored_position
                    << ",\"lookup_ms\":" << r.lookup_ms << ",\"restore_ms\":" << r.restore_ms
                    << ",\"restored_bytes\":" << r.restored_bytes << ",\"captured_checkpoints\":" << r.captured_checkpoints
                    << ",\"captured_segment_bytes\":" << r.captured_segment_bytes
                    << ",\"captured_fixed_bytes\":" << r.captured_fixed_bytes << ",\"capture_ms\":" << r.capture_ms;
                strix::serve::Json declines=strix::serve::Json::object();
                for(const auto& [reason,count]:r.certificate_declines) declines.set(reason,strix::serve::Json(count));
                out << ",\"cache_refusal\":\"" << r.refusal << "\",\"certificate_declines\":" << declines.dump();
                return out.str();
            };
            const auto main_stats = prefix_state->request();
            known_prefix_main_stats = known_prefix_backend->stats();
            const auto main_progress = known_prefix_progress;
            const auto main_known_result = known_prefix_result;
            std::ostringstream followups;
            const auto initial_session = [&] {
                strix::SerialSessionContext fresh;
                fresh.execution_identity = compiled.execution_identity;
                fresh.state_layout_identity = state_layout.layout_identity;
                fresh.runtime_compatibility_identity = identities.compatibility_identity;
                fresh.state.valid = all_persistent_state();
                return fresh;
            };
            for (std::size_t i = 0; i < options.prefix_cache_then_prompts.size(); ++i) {
                const auto prompt = strix::decode_cli::read_prefix_cache_prompt(options.prefix_cache_then_prompts[i]);
                if (prompt.size() > options.context_tokens || options.generate_tokens > options.context_tokens - prompt.size())
                    throw std::invalid_argument("prefix cache follow-up prompt and generation exceed the context");
                for (const auto token : prompt)
                    if (token >= model.vocab_size) throw std::invalid_argument("prefix cache follow-up token exceeds vocabulary");
                // Fresh state, exactly as after setup.
                check_hip(hipDeviceSynchronize(), "prefix cache follow-up fence");
                strix::initialize_hip_fresh_runtime_memory(memory.scratch(), memory.state());
                session = initial_session();
                const auto serial_before = backend.elapsed_ms().size();
                const auto start = std::chrono::steady_clock::now();
                const auto followup = run_known_prompt(prompt);
                const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                double serial_ms = 0.0; // serial suffix rows and generated tokens of this request
                for (auto k = serial_before; k < backend.elapsed_ms().size(); ++k) serial_ms += backend.elapsed_ms()[k];
                followups << (i ? "," : "") << "{\"file\":\"" << strix::decode_receipt::escape_json_string(options.prefix_cache_then_prompts[i])
                    << "\"," << request_json(prefix_state->request()) << ",\"request_elapsed_ms\":" << elapsed << ",\"serial_steps_elapsed_ms\":" << serial_ms
                    << ",\"consumed_prompt_tokens\":" << known_prefix_progress->consumed_prompt_tokens
                    << ",\"windows_executed\":" << known_prefix_progress->windows_executed
                    << ",\"serial_prompt_steps\":" << known_prefix_progress->serial_tail_steps
                    << ",\"tail_windows\":" << known_prefix_progress->tail_windows
                    << ",\"stop_reason\":\"" << stop_reason_name(followup.stop_reason) << "\",\"final_position\":" << followup.final_position
                    << ",\"generated_tokens\":[";
                for (std::size_t t = 0; t < followup.generated_tokens.size(); ++t) followups << (t ? "," : "") << followup.generated_tokens[t];
                followups << "]}";
            }
            known_prefix_progress = main_progress;
            known_prefix_result = main_known_result;
            const auto flush_start = std::chrono::steady_clock::now();
            prefix_store->flush();
            const auto flush_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - flush_start).count();
            const auto store = prefix_store->stats();
            std::ostringstream out;
            out << "  \"prefix_state_cache\":{\"identity\":\"" << strix::model_content_identity_hex(prefix_cache_identity)
                << "\",\"block_rows\":" << prefix_cache_unit(options) << ",\"capture_every_tokens\":" << prefix_cache_every
                << ",\"row_bytes_per_position\":" << prefix_state->row_bytes() << ",\"fixed_bytes_per_checkpoint\":" << prefix_state->fixed_bytes()
                << ",\"host_byte_budget\":" << options.prefix_cache_host_bytes
                << ",\"disk_directory\":\"" << strix::decode_receipt::escape_json_string(options.prefix_cache_dir)
                << "\",\"disk_byte_budget\":" << options.prefix_cache_disk_bytes
                << ",\"main_request\":{" << request_json(main_stats) << "},\"followups\":[" << followups.str() << "]"
                << ",\"store\":{\"lookups\":" << store.lookups << ",\"host_hits\":" << store.host_hits
                << ",\"disk_hits\":" << store.disk_hits << ",\"misses\":" << store.misses << ",\"inserted\":" << store.inserted
                << ",\"evicted\":" << store.evicted << ",\"rejected_over_budget\":" << store.rejected_over_budget
                << ",\"host_bytes\":" << store.host_bytes << ",\"host_checkpoints\":" << store.host_checkpoints
                << ",\"host_segments\":" << store.host_segments << ",\"disk_writes\":" << store.disk_writes
                << ",\"disk_write_bytes\":" << store.disk_write_bytes << ",\"disk_write_failures\":" << store.disk_write_failures
                << ",\"disk_write_ms\":" << store.disk_write_ms << ",\"disk_evictions\":" << store.disk_evictions
                << ",\"disk_reads\":" << store.disk_reads << ",\"disk_read_bytes\":" << store.disk_read_bytes
                << ",\"disk_read_failures\":" << store.disk_read_failures << ",\"disk_read_ms\":" << store.disk_read_ms
                << ",\"exit_flush_ms\":" << flush_ms << "}},\n";
            prefix_cache_receipt = out.str();
        }

        // Validation-only export is after the request and its per-step timing
        // collection. Never enable this for wall-clock throughput measurements.
        if (final_state_metadata) {
            strix::export_final_generation_state(options.validation_final_state_path,
                {options.validation_final_state_maximum_bytes}, model, state_layout, memory.state(),
                session, request, result, *final_state_metadata,
                std::static_pointer_cast<const void>(memory_owner), raw_state_reader,
                raw_require_identity, raw_drain_or_fail, [&] {
                    raw_require_identity();
                    if (strix::model_content_identity_hex(strix::sha256_file_content(options.model_path)) != raw_model_sha ||
                        strix::model_content_identity_hex(strix::sha256_file_content(raw_executable)) != raw_executable_sha)
                        throw std::runtime_error("final state model/executable identity changed during validation");
                    raw_require_identity();
                });
        }
        bitcalc.finish();
        const auto yarn_memory_completed_receipt = strix::decode_cli::yarn_memory_receipt(true);
        std::cout << "{\n" << yarn_memory_completed_receipt << optimization_profile_receipt << dense_trellis_completed_receipt() << eager_submission_receipt << overlap_receipt() << execution_inventory_receipt << gdn_output_mode_receipt() << wave_router_mode_receipt << deferred_moe_mode_receipt << model_copy_receipt;
        if (token_cost_backend) {
            const auto observed = token_cost_backend->stats();
            const strix::ObservedMoeTokenCostPolicy policy{};
            const auto schedule = moe_overlap->stats();
            const auto previous_precision = std::cout.precision(17);
            std::cout << "  \"moe_token_cost_policy\":{\"enabled\":true,\"qualification\":\"not_assessed\""
                << ",\"scope\":\"complete-host-wall-input-through-native-completion-output-delivery-and-commit-not-GPU-time\""
                << ",\"cost_comparison\":\"online-observed-not-paired-benchmark\""
                << ",\"intent_controllers\":\"separate-full-greedy-state\""
                << ",\"resolved_profile_identity\":" << compiled.kernel_plan_identity
                << ",\"bootstrap_samples_per_arm\":" << policy.bootstrap_samples_per_arm
                << ",\"reexplore_interval\":" << policy.reexplore_interval
                << ",\"switch_hysteresis_fraction\":" << policy.switch_hysteresis_fraction
                << ",\"context_bucket_tokens\":" << policy.context_bucket_tokens
                << ",\"selection_changes\":" << observed.selection_changes << ",\"key_resets\":" << observed.key_resets
                << ",\"last_selected\":\"" << (observed.last_selected == strix::MoeTokenSchedule::serial_expanded ? "serial-expanded" : "fork-join")
                << "\",\"serial_schedule_tokens\":" << schedule.serial_tokens
                << ",\"fork_join_schedule_tokens\":" << schedule.fork_join_tokens
                << ",\"saturated\":" << (observed.saturated ? "true" : "false") << ",\"arms\":[";
            for (std::size_t arm = 0; arm < 2U; ++arm) {
                if (arm) std::cout << ',';
                const auto& value = observed.arms[arm];
                std::cout << "{\"schedule\":\"" << (arm ? "fork-join" : "serial-expanded")
                    << "\",\"selected\":" << value.selected << ",\"observed\":" << value.observed
                    << ",\"censored\":" << value.censored << ",\"complete_wall_seconds\":" << value.complete_wall_seconds << '}';
            }
            std::cout << "]},\n";
            std::cout.precision(previous_precision);
        }
        if (lookup_runtime) {
            if (options.prompt_lookup_snapshot_explicit || options.prompt_lookup_snapshot_bytes_explicit) {
                const auto& snapshot = lookup_runtime->snapshot_policy();
                std::cout << "  \"prompt_lookup_snapshot\":{\"policy\":\"" << strix::session_snapshot_policy_name(snapshot.format)
                    << "\",\"max_payload_bytes\":" << snapshot.max_payload_bytes
                    << ",\"latest_completed_snapshot_payload_bytes\":" << lookup_runtime->latest_snapshot_payload_bytes()
                    << ",\"payload_scope\":\"latest-completed-cycle-not-total-or-peak\""
                    << ",\"state_comparison_scope\":\"" <<
                        (snapshot.format == strix::SessionSnapshotPolicy::occupied_kv_v1 ?
                            "occupied-KV-plus-full-other-components-tail-KV-indeterminate" : "full-components") << "\""
                    << ",\"qualification\":\"not_assessed\"},\n";
            }
            if (options.prompt_lookup_adaptive) {
                const auto& adaptive = lookup_runtime->adaptive_stats();
                const strix::HipPromptLookupAdaptivePolicy adaptive_policy{};
                const auto previous_precision = std::cout.precision(17);
                std::cout << "  \"prompt_lookup_adaptive\":{\"enabled\":" << (adaptive.enabled ? "true" : "false")
                    << ",\"scope\":\"completed-host-wall-cycle-including-continuing-bonus-and-publication-not-GPU-time\","
                    << "\"qualification\":\"not_assessed\",\"measurement_eligibility\":\"completed-consumed-cycle\","
                    << "\"totals_scope\":\"entire-request-across-context-resets\","
                    << "\"bootstrap_samples_per_depth\":" << adaptive_policy.controller.bootstrap_samples_per_depth
                    << ",\"context_bucket_tokens\":" << adaptive_policy.context_bucket_tokens
                    << ",\"reexplore_interval\":" << adaptive_policy.controller.reexplore_interval
                    << ",\"switch_hysteresis_fraction\":" << adaptive_policy.controller.switch_hysteresis_fraction
                    << ",\"selected_depth\":" << adaptive.selected_depth
                    << ",\"completed_cycles\":" << adaptive.completed_cycles << ",\"censored_cycles\":" << adaptive.censored_cycles
                    << ",\"context_resets\":" << adaptive.context_resets << ",\"depths\":[";
                for (std::size_t depth = 0; depth < 3U; ++depth) {
                    if (depth) std::cout << ',';
                    std::cout << "{\"depth\":" << depth << ",\"selections\":" << adaptive.depth_selections[depth]
                        << ",\"samples\":" << adaptive.samples[depth] << ",\"emitted_tokens\":" << adaptive.emitted_tokens[depth]
                        << ",\"committed_tokens\":" << adaptive.committed_tokens[depth]
                        << ",\"complete_cycle_seconds\":" << adaptive.complete_cycle_seconds[depth] << '}';
                }
                std::cout << "]},\n";
                std::cout.precision(previous_precision);
            }
            std::cout << "  \"prompt_lookup\":true,\n"
                << "  \"prompt_lookup_qualification\":\"not_assessed\",\n"
                << "  \"prompt_lookup_max_depth\":" << options.prompt_lookup_depth << ",\n"
                << "  \"prompt_lookup_ngram_size\":3,\n"
                << "  \"prompt_lookup_history_capacity_tokens\":4096,\n"
                << "  \"prompt_lookup_cycles\":" << lookup_stats.cycles << ",\n"
                << "  \"prompt_lookup_proposed_tokens\":" << lookup_stats.proposed << ",\n"
                << "  \"prompt_lookup_accepted_candidates\":" << lookup_stats.accepted << ",\n"
                << "  \"prompt_lookup_discarded_candidate_states\":" << lookup_stats.discarded << ",\n"
                << "  \"prompt_lookup_replayed_tokens\":" << lookup_stats.replayed << ",\n"
                << "  \"prompt_lookup_snapshots\":" << lookup_stats.snapshots << ",\n"
                << "  \"prompt_lookup_restores\":" << lookup_stats.restores << ",\n"
                << "  \"prompt_lookup_consumed_bonuses\":" << lookup_stats.consumed_bonuses << ",\n"
                << "  \"prompt_lookup_stats_saturated\":" << (lookup_stats.saturated ? "true" : "false") << ",\n"
                << "  \"prompt_lookup_final_bonus_pending\":" << (lookup_runtime->pending_bonus() ? "true" : "false") << ",\n"
                << "  \"prompt_lookup_full_logits_d2h_bytes\":0,\n"
                << "  \"prompt_lookup_serial_status_execution_identity\":" << known_prefix_memory->serial_scope.deferred_moe_status.execution_identity << ",\n"
                << "  \"prompt_lookup_window_status_execution_identity\":" << known_prefix_memory->scope.deferred_moe_status.execution_identity << ",\n";
        }
        if (requested_logprobs && options.teacher_forced_score) {
            // Route of the scored rows: rows of complete tiled blocks carry the prefill profile's
            // arithmetic, serial rows (all of them on the serial route, the unpadded suffix on the
            // tiled one) the decode profile's; both identities are in this receipt.
            const auto pair = strix::decode_cli::numerical_profile_pair(options);
            const auto tiled_rows = requested_logprobs->tiled_rows();
            std::cout << "  \"teacher_forced_route\":\"" << (known_prefill_plan ? "tiled-known-prefix-prefill" : "serial-eager") << "\",\n"
                << "  \"teacher_forced_tiled_input_rows\":" << tiled_rows << ",\n"
                << "  \"teacher_forced_serial_input_rows\":" << options.tokens.size() - tiled_rows << ",\n"
                << "  \"teacher_forced_tiled_rows_numerical_profile\":\""
                << (tiled_rows ? strix::numerical_profile_name(pair.prefill) : std::string_view("none")) << "\",\n"
                << "  \"teacher_forced_serial_rows_numerical_profile\":\""
                << (options.tokens.size() > tiled_rows ? strix::numerical_profile_name(pair.decode) : std::string_view("none")) << "\",\n";
            if (known_prefill_plan)
                std::cout << "  \"teacher_forced_tiled_lm_head\":\""
                    << (known_prefill_plan->tile8.lm_head_rows_steps ? "rows8-r1-reduction-tree" : "r1-row-loop") << "\",\n";
        }
        if (requested_logprobs) {
            if (!options.teacher_logits_path.empty())
                std::cout << "  \"teacher_logits_file\":\"" << strix::decode_receipt::escape_json_string(options.teacher_logits_path)
                    << "\",\n  \"teacher_logits_scope\":\"complete-finite-full-vocabulary-F32-scored-N-minus-1-rows-with-teacher-IDs-and-completion-footer\",\n";
            const auto previous_precision = std::cout.precision(17);
            std::cout << "  \"requested_logprobs_scope\":\""
                      << (options.teacher_forced_score ? "teacher-forced-next-input-diagnostic-N-minus-1-rows-not-a-speed-run" :
                          "next-token-full-vocabulary-distribution-after-each-consumed-input") << "\",\n"
                      << "  \"requested_logprobs_denominator\":\"full-vocabulary-fp64-logsumexp-ascending-id\",\n"
                      << "  \"requested_logprobs_qualification\":\"not_assessed\",\n"
                      << "  \"requested_logprobs_rows\":[";
            bool first = true;
            for (const auto& row : requested_logprobs->rows()) {
                if (!first) std::cout << ',';
                first = false;
                std::cout << "{\"input_token\":" << row.input << ",\"position\":" << row.position << ",\"values\":[";
                for (std::size_t i = 0; i < row.values.size(); ++i) {
                    if (i) std::cout << ',';
                    std::cout << "{\"token_id\":" << row.values[i].token_id << ",\"log_probability\":" << row.values[i].log_probability << '}';
                }
                std::cout << "]}";
            }
            std::cout << "],\n";
            std::cout.precision(previous_precision);
        }
        if (known_prefix_result) {
            if (known_prefill_plan) {
                std::cout << "  \"known_prefix_tiled_prefill\":true,\n"
                    << "  \"known_prefix_tiled_prefill_qualification\":\"not_assessed\",\n"
                    << "  \"known_prefix_source_ledger_scope\":\"closed-eight-row-selection-not-per-T-execution\",\n"
                    << "  \"known_prefix_prefill_workspace_reserved_bytes\":" << known_prefill_plan->arena.physical.total_bytes << ",\n"
                    << "  \"qsa_score_ring_rows\":" << known_prefill_plan->qsa_score_ring_rows << ",\n"
                    << "  \"qsa_score_ring_workspace_bytes\":" << known_prefill_plan->qsa_score_ring_workspace_bytes << ",\n"
                    << "  \"known_prefix_prefill_metadata_payload_copies\":2,\n"
                    << "  \"known_prefix_prefill_metadata_payload_bytes\":" << known_prefill_plan->host_metadata_payload_bytes * 2U << ",\n"
                    << "  \"known_prefix_prefill_metadata_budget_bytes\":" << options.known_prefill_metadata_budget << ",\n"
                    << "  \"known_prefix_prefill_runtime_known_payload_upper_bound_bytes\":" << known_prefill_runtime_payload << ",\n"
                    << "  \"known_prefix_prefill_host_headroom_bytes\":" << options.known_prefill_host_headroom << ",\n"
                    << "  \"known_prefix_prefill_uma_budget_bytes\":" << options.known_prefill_uma_budget << ",\n"
                    << "  \"known_prefix_prefill_retained_reservation_upper_bound_bytes\":" << known_prefill_reservation->retained_reserved_bytes << ",\n"
                    << "  \"known_prefix_prefill_setup_reservation_upper_bound_bytes\":" << known_prefill_reservation->setup_peak_reserved_bytes << ",\n"
                    << (ple_reclaimer ? "  \"known_prefix_prefill_ple_reclaim_credit_bytes\":" +
                        std::to_string(known_prefill_reservation->reclaimable_file_mapping_credit_bytes) + ",\n" +
                        "  \"ple_reclaim\":" + ple_reclaimer->receipt_json() + ",\n" : std::string{})
                    << "  \"known_prefix_prefill_reservation_scope\":\"declared-storage-upper-bound-not-physical-fit-or-RSS\",\n";
            }
            if (const auto latest = known_prefix_owner->latest_routed_union_observation()) {
                const auto& group = latest->group;
                std::cout << "  \"known_prefix_expert_union_latest_group\":{\"scope\":\""
                    << (known_prefill_plan ? (group.logical_rows == 8U ? "latest-completed-prefill-last-internal-tile-last-eligible-layer" :
                        "latest-completed-prefill-all-rows-last-eligible-layer") : "latest-completed-window-last-eligible-layer")
                    << "\","
                    << "\"first_position\":" << latest->first_position << ",\"layer\":" << latest->layer
                    << ",\"status\":" << group.status << ",\"version\":" << group.version
                    << ",\"logical_rows\":" << group.logical_rows << ",\"touched_experts\":" << group.touched
                    << ",\"batches\":" << group.batches << ",\"assignments\":" << group.assignments
                    << ",\"duplicate_assignments\":" << group.duplicate_assignments
                    << ",\"maximum_assignments\":" << group.maximum_assignments << ",\"rows_per_expert_histogram\":[";
                for (std::size_t i = 0; i < 8U; ++i) std::cout << (i ? "," : "") << group.rows_per_expert_histogram[i];
                std::cout << "],\"batch_size_histogram\":[";
                for (std::size_t i = 0; i < 8U; ++i) std::cout << (i ? "," : "") << group.batch_size_histogram[i];
                std::cout << "],\"histogram_bucket_rows\":\"index-plus-one\",\"whole-model-aggregation\":false},\n";
            }
            std::cout << "  \"known_prefix_output_policy\":\"" << strix::known_window_output_policy_name(known_prefix_plan->output_policy)
                << "\",\n  \"known_prefix_output_policy_qualification\":\"not_assessed\",\n";
            // This dispatcher sees only the serial suffix/generation path.
            // Native window omissions are reported separately by its owner.
            std::cout << "  \"known_prefix_serial_lm_heads_omitted\":" << (output_dispatcher ? output_dispatcher->omitted_heads() : 0U)
                << ",\n  \"known_prefix_serial_state_only_steps\":" << backend.prefix_state_elapsed_ms().size() << ",\n";
            const auto observed = known_prefix_main_stats ? *known_prefix_main_stats : known_prefix_backend->stats();
            std::cout << prefix_cache_receipt;
            std::cout << "  \"known_prefix_omitted_lm_heads\":" << observed.omitted_lm_heads
                << ",\n  \"known_prefix_state_only_windows\":" << observed.state_only_windows << ",\n";
            std::array<std::uint64_t, strix::known_window_r2_variant_slots> compiled_variants{};
            for (const auto variant : known_prefix_plan->r2_variant) {
                const auto index = static_cast<std::uint32_t>(variant);
                if (index >= compiled_variants.size()) throw std::logic_error("unknown prompt variant in source receipt");
                if (index) ++compiled_variants[index];
            }
            std::cout << "  \"known_prefix_expert_union_requested\":" << (options.known_prefix_expert_union ? "true" : "false") << ",\n"
                      << "  \"known_prefix_expert_union_source_steps\":" << known_prefix_plan->routed_expert_union_steps << ",\n"
                      << "  \"known_prefix_expert_union_workspace_budget_bytes\":" << (options.known_prefix_expert_union ? 65536U : 0U) << ",\n"
                      << "  \"known_prefix_expert_union_workspace_reserved_bytes\":" << known_prefix_memory->union_scratch.arena_bytes << ",\n"
                      << "  \"known_prefix_expert_union_workspace_generation\":" << known_prefix_memory->union_scratch.allocation_generation << ",\n"
                      << "  \"known_prefix_expert_union_workspace_lifetime\":\"setup-allocation-retained-until-original-device-quiescence-no-per-call-allocation\",\n"
                      << "  \"known_prefix_expert_union_reason\":\"" << (!options.known_prefix_expert_union ? "not-requested" :
                          known_prefix_plan->routed_expert_union_steps ? "eligible-closed-variants" : "no-eligible-routed-q51-steps-fallback") << "\",\n"
                      << "  \"known_prefix_logical_rows\":" << (known_prefill_plan ? known_prefill_plan->logical_rows : known_prefix_plan->logical_rows) << ",\n"
                      << "  \"known_prefix_suffix_policy\":\"unpadded-serial-R1\",\n"
                      << "  \"known_prefix_projection_dispatch_accounting\":\"actual-closed-variant-table-not-logical-row-inference\",\n"
                      << "  \"known_prefix_chunk_cursor\":true,\n"
                      << "  \"known_prefix_generation_input_tokens_consumed\":" << known_prefix_progress->consumed_generation_tokens << ",\n"
                      << "  \"known_prefix_final_bonus_pending\":" << (known_prefix_progress->bonus_pending ? "true" : "false") << ",\n";
            std::cout << "  \"known_prefix_completed_windows\":" << observed.windows << ",\n"
                << "  \"known_prefix_stats_saturated\":" << (observed.saturated ? "true" : "false") << ",\n"
                << "  \"known_prefix_native_completed_input_rows\":" << observed.completed_input_rows << ",\n"
                << "  \"known_prefix_native_shared_rows_dispatches\":" << observed.shared_rows_dispatches << ",\n"
                << "  \"known_prefix_native_gdn_rows_dispatches\":" << observed.gdn_rows_dispatches << ",\n"
                << "  \"known_prefix_routed_expert_union_dispatches\":" << observed.routed_expert_union_dispatches << ",\n"
                << "  \"known_prefix_r2_dispatches\":" << observed.r2_dispatches << ",\n"
                << "  \"known_prefix_row_loop_dispatches\":" << observed.row_loop_dispatches << ",\n"
                << "  \"known_prefix_logical_projection_weight_bytes_avoided\":" << observed.logical_projection_weight_bytes_avoided << ",\n"
                << "  \"known_prefix_projection_launches_saved\":" << observed.projection_launches_saved << ",\n"
                << "  \"known_prefix_r2_variant_dispatches\":[";
            for (std::size_t i = 0; i < observed.r2_variant_dispatches.size(); ++i)
                std::cout << (i ? "," : "") << observed.r2_variant_dispatches[i];
            std::cout << "],\n  \"known_prefix_variant_names\":[";
            // One name per KnownWindowR2Variant, in enum order.
            constexpr std::array<std::string_view, strix::known_window_r2_variant_slots> variant_names{
                "none", "shared-q8-rows2", "gdn-all-q8-pair-tiles", "gdn-q8-rows2-f32-gates-r1",
                "shared-q8-rows4", "shared-q8-rows8", "gdn-q8-rows4-f32-gates-r1", "gdn-q8-rows8-f32-gates-r1",
                "routed-q51-union-rows2", "routed-q51-union-rows4", "routed-q51-union-rows8",
                "hyper-mix-rows2", "hyper-mix-rows4", "hyper-mix-rows8",
                "qsa-q8-rows2-r1", "qsa-q8-rows4-r1", "qsa-q8-rows8-r1",
                "gdn-deferred-rows2", "gdn-deferred-rows4", "gdn-deferred-rows8", "ple-rows-host-staged",
                "qsa-q8-bf16idx-rows2-r1", "qsa-q8-bf16idx-rows4-r1", "qsa-q8-bf16idx-rows8-r1",
                "lm-head-q8-prefetch8-rows2", "lm-head-q8-prefetch8-rows4", "lm-head-q8-prefetch8-rows8",
                "moe-route-rows", "moe-combine-rows", "inject-rows",
                "routed-a8-union-rows2", "routed-a8-union-rows4", "routed-a8-union-rows8"};
            for (std::size_t i = 0; i < compiled_variants.size(); ++i) {
                if (i >= variant_names.size()) throw std::logic_error("unknown known-prefix variant name");
                std::cout << (i ? "," : "") << '"' << variant_names[i] << '"';
            }
            std::cout << "],\n  \"known_prefix_source_variant_steps\":[";
            for (std::size_t i = 0; i < compiled_variants.size(); ++i) std::cout << (i ? "," : "") << compiled_variants[i];
            std::cout << "],\n  \"known_prefix_window\":true,\n"
                << "  \"known_prefix_window_qualification\":\"not_assessed\",\n"
                << "  \"known_prefix_consumed_prompt_tokens\":" << known_prefix_result->consumed_prompt_tokens << ",\n"
                << "  \"known_prefix_windows_executed\":" << known_prefix_result->windows_executed << ",\n"
                << "  \"known_prefix_serial_prompt_steps\":" << known_prefix_result->serial_prompt_steps << ",\n"
                << "  \"known_prefix_tail_rows\":[" << tail_rows_text << "],\n"
                << "  \"known_prefix_tail_windows\":" << known_prefix_progress->tail_windows << ",\n"
                << "  \"known_prefix_window_execution_identity\":" << (known_prefill_plan ? known_prefill_plan->prefill_execution_identity : known_prefix_plan->window_execution_identity) << ",\n"
                << "  \"known_prefix_window_kernel_plan_identity\":" << (known_prefill_plan ? known_prefill_plan->prefill_kernel_plan_identity : known_prefix_plan->window_kernel_plan_identity) << ",\n"
                << "  \"known_prefix_source_r2_steps\":" << known_prefix_plan->r2_steps << ",\n"
                << "  \"known_prefix_source_projection_launches_saved_per_window\":" << (known_prefill_plan ? 0U : known_prefix_plan->projection_launches_saved) << ",\n"
                << "  \"known_prefix_source_logical_weight_bytes_avoided_per_window\":" << (known_prefill_plan ? 0U : known_prefix_plan->logical_projection_weight_bytes_avoided) << ",\n";
        }
        if (generation_request_timing_enabled)
            std::cout << "  \"generation_request_elapsed_ms\":" << generation_request_elapsed_ms << ",\n"
                      << "  \"generation_request_output_tokens\":" << result.generated_tokens.size() << ",\n"
                      << "  \"generation_request_output_tokens_per_second\":"
                      << (generation_request_elapsed_ms > 0.0
                              ? static_cast<double>(result.generated_tokens.size()) * 1000.0 /
                                    generation_request_elapsed_ms
                              : 0.0) << ",\n"
                      << "  \"generation_request_timing_scope\":\"full-request-including-prefix-execution-readback-token-selection-and-controller\",\n";
        if (final_state_metadata)
            std::cout << "  \"validation_final_state_timing_invalid\":true,\n"
                      << "  \"validation_final_state_manifest\":\""
                      << strix::decode_receipt::escape_json_string(
                             (std::filesystem::path(options.validation_final_state_path) / "manifest.json").string())
                      << "\",\n";
        if (greedy_output) {
            const auto stats = greedy_output->stats();
            double prefix_state_total = 0.0;
            for (const auto duration : backend.prefix_state_elapsed_ms()) prefix_state_total += duration;
            std::cout << "  \"generation_output_mode\":\"" << strix::gpu_greedy_output_identity << "\",\n"
                << "  \"prefix_state_only_steps\":" << backend.prefix_state_elapsed_ms().size() << ",\n"
                << "  \"prefix_state_only_elapsed_ms_total\":" << prefix_state_total << ",\n"
                << "  \"prefix_state_only_elapsed_ms_per_step\":"
                << (backend.prefix_state_elapsed_ms().empty() ? 0.0 : prefix_state_total /
                    static_cast<double>(backend.prefix_state_elapsed_ms().size())) << ",\n"
                << "  \"generation_output_completed\":" << stats.completed << ",\n"
                << "  \"generation_output_vocabulary_size\":" << model.vocab_size << ",\n"
                << "  \"generation_output_calls\":" << stats.calls << ",\n"
                << "  \"generation_output_failures\":" << stats.failures << ",\n"
                << "  \"generation_output_kernel_submissions\":" << stats.reduction_kernel_submissions << ",\n"
                << "  \"generation_output_poison_h2d_bytes\":" << stats.poison_h2d_bytes << ",\n"
                << "  \"generation_output_stream_completions\":" << stats.stream_completions << ",\n"
                << "  \"generation_output_full_logits_d2h_bytes\":0,\n"
                << "  \"generation_output_result_d2h_bytes\":" << stats.result_d2h_bytes << ",\n"
                << "  \"generation_output_pool_requested_bytes\":" << greedy_output->scratch_bytes() << ",\n"
                << "  \"prefill_lm_heads_omitted\":" << output_dispatcher->omitted_heads() << ",\n"
                << "  \"generation_output_timing_scope\":\"full-output-steps-producer-through-selection-and-readback\",\n";
        }
        emit_q4k_a8_receipt();
        emit_q4k_a8_counts();
        std::cout << strix::decode_detail::qsa_input_dispatch_receipt(qsa_prefetch_stats.get());
        std::cout << shared_dispatch_receipt.completed_counts();
        std::cout << ple_cache_completed_receipt() << qsa_gqa_receipt() << gdn_recurrent_layout_receipt() << qsa_selection_receipt() << gdn_islands_receipt() << strix::decode_detail::deferred_moe_completed_tokens_receipt(deferred_moe.get());
        std::cout
            << "  \"model_content_identity\": \""
            << strix::model_content_identity_hex(
                   model_residency.binding().
                       model_content_identity)
            << "\",\n"
            << "  \"execution_identity\": "
            << compiled.execution_identity
            << ",\n"
            << "  \"runtime_compatibility_identity\": "
            << identities.compatibility_identity
            << ",\n"
            << "  \"ple_lookup\": \""
            << (options.planned_ple_io
                    ? "planned-pread"
                    : "mapped")
            << "\",\n"
            << "  \"prefix_tokens\": [";

        for (std::size_t index = 0;
             index < options.tokens.size();
             ++index) {
            if (index != 0) {
                std::cout << ',';
            }
            std::cout << options.tokens[index];
        }

        std::cout
            << "],\n"
            << "  \"generated_tokens\": [";

        for (std::size_t index = 0;
             index <
                result.generated_tokens.size();
             ++index) {
            if (index != 0) {
                std::cout << ',';
            }
            std::cout
                << result.generated_tokens[index];
        }

        std::cout << "],\n";
        if (bitcalc.enabled()) {
            std::cout << "  \"bitcalc_forced_steps\": " << bitcalc.forced_steps() << ",\n";
            if (!bitcalc.force_tokens_sha256().empty())
                std::cout << "  \"bitcalc_force_tokens_sha256\": \"" << bitcalc.force_tokens_sha256() << "\",\n";
        }
        std::cout
            << "  \"stop_reason\": \""
            << stop_reason_name(
                   result.stop_reason)
            << "\",\n"
            << "  \"steps_executed\": "
            << result.steps_executed
            << ",\n"
            << "  \"final_position\": "
            << result.final_position
            << ",\n"
            << (lookup_runtime ? "  \"prompt_lookup_timing_scope\":\"generation_request_elapsed_ms_only\"\n" : "");
        if (!lookup_runtime) std::cout
            << (known_prefix_backend ? "  \"known_prefix_serial_tail_and_generation_elapsed_ms_total\": " :
                greedy_output ? "  \"generation_step_elapsed_ms_total\": " : "  \"decode_elapsed_ms_total\": ")
            << elapsed_total
            << ",\n"
            << (known_prefix_backend ? "  \"known_prefix_serial_tail_and_generation_elapsed_ms_per_step\": " :
                greedy_output ? "  \"generation_step_elapsed_ms_per_step\": " : "  \"decode_elapsed_ms_per_step\": ")
            << (backend.elapsed_ms().empty()
                    ? 0.0
                    : elapsed_total /
                          static_cast<double>(
                              backend.elapsed_ms().size()))
            << "\n";
        std::cout << "}\n";
        emit_device_queue_trace();
        return 0;
    }

    std::vector<float> logits(
        static_cast<std::size_t>(
            model.vocab_size));
    std::vector<double> token_elapsed_ms;
    token_elapsed_ms.reserve(
        options.tokens.size());
    double decode_elapsed_ms_total = 0.0;

    strix::SerialDecodeReplayObservation
        replay_observation;
    if (reference.has_value()) {
        replay_observation.model_content_identity =
            model_residency.binding().
                model_content_identity;
        replay_observation.compiled_execution_identity =
            compiled.execution_identity;
        replay_observation.runtime_compatibility_identity =
            identities.compatibility_identity;
        replay_observation.independent_reference_identity =
            reference->independent_reference_identity;
        replay_observation.input_tokens =
            options.tokens;
        replay_observation.reference_argmax_tokens.reserve(
            reference->steps.size());
        replay_observation.engine_argmax_tokens.reserve(
            reference->steps.size());
    }

    std::cout << '{' << '\n' << optimization_profile_receipt << dense_trellis_completed_receipt() << eager_submission_receipt << execution_inventory_receipt << gdn_output_mode_receipt() << wave_router_mode_receipt << deferred_moe_mode_receipt;
    emit_q4k_a8_receipt();
    std::cout << "  \"model_content_identity\": \""
              << strix::model_content_identity_hex(
                     model_residency.binding().model_content_identity)
              << "\"," << '\n'
              << "  \"execution_identity\": "
              << compiled.execution_identity
              << ',' << '\n'
              << "  \"ranked_moe_router_topk\": " << (options.ranked_moe_router_topk ? "true" : "false") << ",\n"
              << "  \"parallel_moe_router\": " << (options.parallel_moe_router ? "true" : "false") << ",\n"
              << "  \"launch_plan_identity\": "
              << launch_plan.plan_identity
              << ',' << '\n'
              << "  \"runtime_compatibility_identity\": "
              << identities.compatibility_identity
              << ',' << '\n'
              << "  \"context_tokens\": "
              << options.context_tokens
              << ',' << '\n'
              << "  \"context_qualified\": "
              << (options.context_tokens == 8192 ? "true" : "false")
              << ',' << '\n'
              << "  \"residency\": \""
              << (options.residency ==
                          strix::HipModelResidencyMode::
                              registered_file_mapping
                      ? "register"
                      : "copy")
              << "\"," << '\n'
              << model_copy_receipt
              << "  \"registered_bytes\": "
              << model_residency.registered_bytes()
              << ',' << '\n'
              << "  \"copied_bytes\": "
              << model_residency.copied_bytes()
              << ',' << '\n'
              << "  \"ple_lookup\": \""
              << (options.planned_ple_io
                      ? "planned-pread"
                      : "mapped")
              << "\"," << '\n'
              << "  \"ple_max_rows_per_io\": "
              << options.ple_max_rows_per_io
              << ',' << '\n'
              << "  \"ple_max_inflight_ios\": "
              << options.ple_max_inflight_ios
              << ',' << '\n'
              << "  \"ple_lookup_identity\": "
              << ple_lookup->implementation_identity()
              << ',' << '\n'
              << "  \"tokens\": [" << '\n';

    for (std::size_t input_index = 0;
         input_index <
            options.tokens.size();
         ++input_index) {
        const auto token =
            options.tokens[input_index];

        const auto token_start =
            std::chrono::steady_clock::now();

        try {
        execute_native_token(token);

        if (!deferred_moe) check_hip(
            hipDeviceSynchronize(),
            "hipDeviceSynchronize(serial token)");
        } catch (...) { if (raw_drain_or_fail) raw_drain_or_fail(); throw; }

        const auto token_stop =
            std::chrono::steady_clock::now();
        const auto elapsed_ms =
            std::chrono::duration<double, std::milli>(
                token_stop - token_start).count();
        token_elapsed_ms.push_back(
            elapsed_ms);
        decode_elapsed_ms_total +=
            elapsed_ms;

        const auto logits_address =
            strix::resolve_execution_buffer(
                compiled.execution_identity,
                compiled.arena,
                memory.scratch(),
                lm_io.writes[0]);
        if (logits_address.bytes <
            logits.size() *
                sizeof(float)) {
            throw std::logic_error(
                "compiled logits allocation is undersized");
        }

        if (raw_require_identity) raw_require_identity();
        check_raw_copy(
            hipMemcpy(
                logits.data(),
                reinterpret_cast<
                    const void*>(
                        logits_address.address),
                logits.size() *
                    sizeof(float),
                hipMemcpyDeviceToHost),
            "hipMemcpy(logits D2H)", raw_drain_or_fail);
        if (raw_require_identity) raw_require_identity();

        strix::validate_serial_decode_engine_logits(
            logits,
            model.vocab_size);

        const auto ranked =
            top_logits(
                logits,
                5);
        if (ranked.empty()) {
            throw std::logic_error(
                "LM head produced no logits");
        }
        if (raw_recording) raw_recording->record_token(token, input_index, logits, session, raw_state_reader,
            input_index + 1 < options.tokens.size() ? std::optional{options.tokens[input_index + 1]} : std::nullopt);

        if (reference.has_value()) {
            const auto& expected =
                reference->steps[input_index];
            replay_observation.reference_argmax_tokens.push_back(
                expected.expected_argmax_token);
            replay_observation.engine_argmax_tokens.push_back(
                ranked.front().token);

            for (const auto& probe :
                 expected.logits) {
                if (probe.index >=
                    logits.size()) {
                    throw std::out_of_range(
                        "reference logit probe index exceeds model vocabulary");
                }
                const auto error =
                    std::abs(
                        static_cast<double>(
                            logits[probe.index]) -
                        static_cast<double>(
                            probe.value));
                replay_observation.maximum_logit_abs_error =
                    std::max(
                        replay_observation.maximum_logit_abs_error,
                        error);
                ++replay_observation.logit_values_compared;
            }
        }

        std::cout
            << "    {\"input_token\":"
            << token
            << ",\"position\":"
            << (session.next_position - 1ULL)
            << ",\"argmax_token\":"
            << ranked.front().token
            << ",\"argmax_logit\":"
            << ranked.front().logit
            << ",\"elapsed_ms\":"
            << elapsed_ms
            << ",\"top5\":[";

        for (std::size_t rank = 0;
             rank < ranked.size();
             ++rank) {
            if (rank != 0) {
                std::cout << ',';
            }
            std::cout
                << "{\"token\":"
                << ranked[rank].token
                << ",\"logit\":"
                << ranked[rank].logit
                << '}';
        }
        std::cout << "]}";
        if (input_index + 1U !=
            options.tokens.size()) {
            std::cout << ',';
        }
        std::cout << '\n';
    }

    std::optional<strix::SerialDecodeReplayAssessment>
        replay_assessment;
    finalize_raw();
    if (reference.has_value()) {
        strix::SerialDecodeReplayPolicy policy;
        policy.expected_model_content_identity =
            model_residency.binding().
                model_content_identity;
        policy.expected_compiled_execution_identity =
            compiled.execution_identity;
        policy.expected_runtime_compatibility_identity =
            identities.compatibility_identity;
        policy.policy_identity =
            reference->independent_reference_identity;
        policy.max_logit_abs_error =
            options.max_logit_abs_error;
        policy.minimum_steps =
            static_cast<std::uint64_t>(
                reference->steps.size());
        policy.minimum_logit_values_compared =
            required_reference_logit_values;
        policy.require_exact_argmax = true;

        replay_assessment =
            strix::assess_serial_decode_replay(
                policy,
                replay_observation);
    }

    std::cout << "  ]," << '\n';
    emit_q4k_a8_counts();
    // Dynamic counters belong after the raw token loop, not the JSON prefix.
    std::cout << overlap_receipt();
    std::cout << strix::decode_detail::qsa_input_dispatch_receipt(qsa_prefetch_stats.get());
    std::cout << shared_dispatch_receipt.completed_counts();
    std::cout << ple_cache_completed_receipt() << qsa_gqa_receipt() << gdn_recurrent_layout_receipt() << qsa_selection_receipt() << gdn_islands_receipt() << strix::decode_detail::deferred_moe_completed_tokens_receipt(deferred_moe.get());
    std::cout
        << "  \"decode_elapsed_ms_total\": "
        << decode_elapsed_ms_total
        << ',' << '\n'
        << "  \"decode_elapsed_ms_per_token\": "
        << (token_elapsed_ms.empty()
                ? 0.0
                : decode_elapsed_ms_total /
                      static_cast<double>(
                          token_elapsed_ms.size()))
        << ',' << '\n'
        << "  \"committed_tokens\": "
        << session.committed_tokens
        << ',' << '\n'
        << "  \"next_position\": "
        << session.next_position;

    if (replay_assessment.has_value()) {
        std::cout
            << ',' << '\n'
            << "  \"replay\": {" << '\n'
            << "    \"reference_id\": "
            << reference->independent_reference_identity
            << ',' << '\n'
            << "    \"logit_values_compared\": "
            << replay_observation.logit_values_compared
            << ',' << '\n'
            << "    \"max_logit_abs_error\": "
            << replay_observation.maximum_logit_abs_error
            << ',' << '\n'
            << "    \"argmax_match\": "
            << (replay_assessment->argmax_match
                    ? "true"
                    : "false")
            << ',' << '\n'
            << "    \"passed\": "
            << (replay_assessment->passes_declared_policy
                    ? "true"
                    : "false")
            << '\n'
            << "  }";
    }

    std::cout
        << '\n'
        << '}' << '\n';

    emit_device_queue_trace();
    return replay_assessment.has_value() &&
                   !replay_assessment->passes_declared_policy
               ? 2
               : 0;
} catch (const strix::decode_cli::YarnBundleCalibrationFailure& error) {
    // All owners above have unwound. Retry from a fresh serial target session;
    // no generated output has been published by the calibration-only pass.
    bundle_retry.restore();
    return strix::decode_cli::retry_mtp_bundle_serial(error,[&] {return run_decode_entry(argc,argv);});
} catch (const strix::decode_cli::YarnMemoryPressure& error) {
    if (const auto status=strix::decode_cli::retry_mtp_bundle_setup(bundle_retry,error,[&] {return run_decode_entry(argc,argv);}))
        return *status;
    std::cerr << "strix-hip-decode-token: " << error.what() << '\n';
    std::cout << "{\"stop_reason\":\"yarn-memory-watchdog\"," << error.receipt() << "}\n";
    return 1;
} catch (const strix::ple_reclaim::DirectError& error) {
    if (const auto status=strix::decode_cli::retry_mtp_bundle_setup(bundle_retry,error,[&] {return run_decode_entry(argc,argv);}))
        return *status;
    std::cerr << "strix-hip-decode-token: PLE runtime read failed after source admission; canonical state may be mutated; "
        "resident restart requires a fresh invocation: " << error.what() << '\n';
    return 1;
} catch (const std::exception& error) {
    if (const auto status=strix::decode_cli::retry_mtp_bundle_setup(bundle_retry,error,[&] {return run_decode_entry(argc,argv);}))
        return *status;
    std::cerr
        << "strix-hip-decode-token: "
        << error.what() << '\n';
    return 1;
}

int STRIX_DECODE_ENTRY(int argc,char** argv) {
    return run_decode_entry(argc,argv);
}
