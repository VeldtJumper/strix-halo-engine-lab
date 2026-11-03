#pragma once

#include "strix/model_contract.hpp"
#include "strix/ple_rolling.hpp"
#include "strix/ple_direct.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/session_state_memory.hpp"
#include "strix/sparse_lookup_executor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix {

struct PleHashContract {
    static constexpr std::size_t maximum_heads = 32;
    static constexpr std::size_t maximum_ngram = 8;

    std::uint32_t ngram_size = 0;
    std::uint32_t heads_per_ngram = 0;
    std::uint32_t head_count = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t eos_token_id = 0;
    std::uint64_t row_bytes = 0;
    std::array<std::uint64_t, maximum_ngram> multipliers{};
    std::array<std::uint64_t, maximum_heads> head_offsets{};
    std::array<std::uint64_t, maximum_heads> head_vocab_sizes{};
    std::uint64_t minimum_table_rows = 0;
};

[[nodiscard]] PleHashContract build_ple_hash_contract(
    const ModelContract& model);

struct PleHashResult {
    std::array<std::uint64_t, PleHashContract::maximum_heads> row_ids{};
    std::uint32_t row_count = 0;
    std::array<std::uint32_t, PleHashContract::maximum_ngram - 1> next_context{};
    std::uint32_t context_count = 0;
};

[[nodiscard]] PleHashResult hash_ple_token(
    const PleHashContract& contract,
    std::uint32_t token_id,
    std::span<const std::uint32_t> previous_context);

void hash_ple_tokens(const PleHashContract& contract, std::span<const std::uint32_t> tokens,
    std::span<const std::uint32_t> previous, std::span<PleHashResult> out);

using PleLookupHandle = std::uint64_t;

struct PleLookupRequest {
    std::span<const std::uint64_t> row_ids;
    std::uint64_t row_bytes = 0;
    const ManifestTensor* table_manifest = nullptr;
    const RuntimeTensorBinding* table_runtime = nullptr;
};

class PleLookupBackend {
public:
    virtual ~PleLookupBackend() = default;

    [[nodiscard]] virtual std::uint64_t implementation_identity() const noexcept = 0;

    [[nodiscard]] virtual PleLookupHandle start(
        const PleLookupRequest& request) = 0;

    virtual void finish(
        PleLookupHandle handle,
        std::span<std::byte> destination) = 0;

    virtual void cancel(
        PleLookupHandle handle) noexcept = 0;
};

class FileMappedPleLookupBackend final
    : public PleLookupBackend {
public:
    FileMappedPleLookupBackend();   // out of line: the helper type is complete only in ple_lookup.cpp
    // Drains the pre-issue helper before any member or captured source can go away.
    ~FileMappedPleLookupBackend() override;
    // Exclusive helper ownership: a copy or move would share or orphan an in-flight pre-issue.
    FileMappedPleLookupBackend(const FileMappedPleLookupBackend&) = delete;
    FileMappedPleLookupBackend& operator=(const FileMappedPleLookupBackend&) = delete;
    FileMappedPleLookupBackend(FileMappedPleLookupBackend&&) = delete;
    FileMappedPleLookupBackend& operator=(FileMappedPleLookupBackend&&) = delete;
    void bind_residency(std::shared_ptr<ple_reclaim::ResidencyManager::Turn> turn) {
        if (pending_.active) throw std::logic_error("PLE binding requires cancelled/drained lookup");
        residency_ = std::move(turn);
    }
    [[nodiscard]] auto residency() const { return residency_; }
    void bind_prompt_source(std::shared_ptr<const ple_reclaim::DirectPageCache::View> view,
        std::uintptr_t table, std::uint64_t offset) {
        if (pending_.active) throw std::logic_error("PLE prompt view requires an idle helper");
        prompt_view_ = std::move(view); prompt_table_ = table; prompt_offset_ = offset;
    }
    [[nodiscard]] std::uint64_t implementation_identity() const noexcept override;

    // STRIX_PLE_PREISSUE (default on where enable_preissue() admitted the route; =0, STRIX_QPF2K_DEFAULTS=0 or
    // STRIX_BASELINE=1 turn it off): after start(),
    // issue the reads of `rows` now on a parked helper through the same source the gather will use (bound
    // residency turn or active rolling source), with an empty copy. No bytes are handed over; the later
    // finish() finds the pages ready or joins them in flight. Joined at the end of finish(), in cancel() and
    // on destruction. Only O_DIRECT-owned sources; otherwise and on any error it does nothing.
    void preissue(std::span<const std::uint64_t> rows) noexcept;
    // Route gate (off by default): only the plain eager CLI decode turns it on, after its prompt is consumed and
    // never for an MTP session. Every other owner (MTP, serving, cohorts, distributed stages) keeps the plain path.
    void enable_preissue(bool on) noexcept { preissue_route_ = on; }
    [[nodiscard]] static bool preissue_enabled() noexcept;
    // Master switches first: STRIX_BASELINE=1 or STRIX_QPF2K_DEFAULTS=0 turn it off even with an explicit =1.
    [[nodiscard]] static bool preissue_enabled_from(const char* preissue, const char* defaults, const char* baseline) noexcept;

    [[nodiscard]] PleLookupHandle start(
        const PleLookupRequest& request) override;

    void finish(
        PleLookupHandle handle,
        std::span<std::byte> destination) override;

    void cancel(
        PleLookupHandle handle) noexcept override;

private:
    struct Pending {
        PleLookupHandle handle = 0;
        std::uintptr_t table_address = 0;
        std::uint64_t table_bytes = 0;
        std::uint64_t row_bytes = 0;
        std::uint64_t stored_row_bytes = 0;
        std::uint32_t ggml_type = 0;
        std::array<std::uint64_t, PleHashContract::maximum_heads> row_ids{};
        std::uint32_t row_count = 0;
        bool active = false;
        std::uint64_t mapping_generation = 0, residency_epoch = 0;
        std::uintptr_t rolling_source = 0;
        std::shared_ptr<ple_reclaim::ResidencyManager::Turn> residency;
    };

    struct PreissueWorker;
    void join_preissue() noexcept;

    PleLookupHandle next_handle_ = 1;
    Pending pending_;
    std::unique_ptr<PreissueWorker> preissue_;
    bool preissue_route_ = false;
    std::shared_ptr<ple_reclaim::ResidencyManager::Turn> residency_;
    std::shared_ptr<const ple_reclaim::DirectPageCache::View> prompt_view_;
    std::uintptr_t prompt_table_ = 0;
    std::uint64_t prompt_offset_ = 0;
};

struct PlannedPleLookupPolicy {
    std::uint64_t max_rows_per_io = 4;
    std::uint64_t max_inflight_ios = 4;
    // Zero preserves the legacy executor. Explicit 1..8 counts the outer
    // asynchronous lookup coordinator as a reader within this total budget.
    // Per backend/active lookup only; excludes threads internal to IoSource.
    std::uint32_t cpu_helper_threads = 0;
    // Explicit independent payload cache. Both zero preserve the legacy path.
    // Storage and tags are reserved once in the backend constructor, before
    // any lookup. Budget excludes existing async I/O operation payloads.
    std::uint64_t cache_capacity_rows = 0;
    std::uint64_t cache_byte_budget = 0;
    // Source bytes are immutable for this exact content identity throughout
    // the retained source owner's lifetime. Cache never crosses this binding.
    ModelContentIdentity cache_model_content{};
};

struct PlannedPleCacheStats {
    std::uint64_t hits = 0, misses = 0, inserted = 0, evicted = 0, invalidations = 0;
    std::uint64_t reserved_bytes = 0, resident_rows = 0;
    bool saturated = false;
};

class PlannedPleLookupBackend final
    : public PleLookupBackend {
public:
    // Source/config callers retain their owners until finish/cancel or backend
    // destruction has joined every submitted helper. No interruptible read
    // primitive exists: cancellation drains and discards, never releases early.
    // Single active lookup / externally serialized instance, including stats.
    // With caching enabled source_owner must retain this exact source and its
    // immutable model bytes. Cached payloads are raw BF16 rows, not decoded IQ4.
    explicit PlannedPleLookupBackend(
        SparseLookupIoSource& source,
        PlannedPleLookupPolicy policy = {},
        std::shared_ptr<const void> source_owner = {});

    [[nodiscard]] std::uint64_t implementation_identity() const noexcept override;
    [[nodiscard]] PlannedPleCacheStats cache_stats() const noexcept;
    [[nodiscard]] static std::uint64_t cache_reservation_bytes(PlannedPleLookupPolicy policy);

    [[nodiscard]] PleLookupHandle start(
        const PleLookupRequest& request) override;

    void finish(
        PleLookupHandle handle,
        std::span<std::byte> destination) override;

    void cancel(
        PleLookupHandle handle) noexcept override;

private:
    struct Pending {
        PleLookupHandle handle = 0;
        std::uint64_t row_bytes = 0;
        std::array<std::uint64_t, PleHashContract::maximum_heads> row_ids{};
        std::uint32_t row_count = 0;
        std::vector<SparseLookupIoOperation> io_plan;
        std::future<SparseLookupIoExecution> future;
        std::array<std::size_t, PleHashContract::maximum_heads> cached_slots{};
        bool cache_enabled = false;
        bool active = false;
    };

    SparseLookupIoSource* source_ = nullptr;
    PlannedPleLookupPolicy policy_{};
    std::shared_ptr<const void> source_owner_;
    struct CacheSlot { std::uint64_t row = 0, age = 0; bool valid = false; };
    std::vector<CacheSlot> cache_slots_;
    std::vector<std::byte> cache_bytes_;
    std::uint64_t cache_row_bytes_ = 0, cache_rows_ = 0;
    const ManifestTensor* cache_manifest_ = nullptr;
    const RuntimeTensorBinding* cache_runtime_ = nullptr;
    RuntimeTensorBinding cache_binding_{};
    std::uint64_t cache_offset_ = 0, cache_table_bytes_ = 0, cache_table_rows_ = 0;
    PlannedPleCacheStats cache_counters_{};
    void cache_count(std::uint64_t& counter) noexcept;
    void cache_touch(std::size_t slot) noexcept;
    void cache_admit(const PleLookupRequest& request);
    void cache_insert(std::uint64_t row, const std::byte* bytes) noexcept;
    PleLookupHandle next_handle_ = 1;
    Pending pending_;
};

struct PleLookupSessionState {
    PleLookupHandle handle = 0;
    bool active = false;
    std::uint64_t sequence = 0;
    std::array<std::uint64_t, PleHashContract::maximum_heads> row_ids{};
    std::uint32_t row_count = 0;
};

struct PleLookupPrefetchState {
    PleLookupHandle handle = 0;
    std::uint64_t backend_identity = 0;
    std::uint32_t input_token_id = 0;
    std::uint64_t position = 0;
    std::uint64_t lookup_bytes = 0;
    PleHashResult hash;
    bool active = false;
};

[[nodiscard]] std::uint64_t ple_hash_contract_identity(
    const PleHashContract& contract) noexcept;

void start_ple_lookup_prefetch(
    const PleHashContract& contract,
    std::uint32_t ple_step_id,
    std::uint32_t input_token_id,
    std::uint64_t position,
    const ModelManifest& manifest,
    const ModelRuntimeBinding& weights,
    const ExecutionTensorAccessPlan& tensor_access,
    const SerialSessionStateLayout& state_layout,
    const SessionStateMemoryBinding& state_memory,
    PleLookupBackend& backend,
    PleLookupPrefetchState& prefetch,
    // False skips the read pre-issue: verify windows (their rows are warmed during the draft phase; an
    // admission-time issue of the whole window measured no gain) and prompt blocks (staged ahead already).
    bool preissue = true);

[[nodiscard]] PleHashResult finish_ple_lookup_prefetch(
    std::uint32_t input_token_id,
    std::uint64_t position,
    PleLookupBackend& backend,
    PleLookupPrefetchState& prefetch,
    std::span<std::byte> destination);

void cancel_ple_lookup_prefetch(
    PleLookupBackend& backend,
    PleLookupPrefetchState& prefetch) noexcept;

}  // namespace strix
