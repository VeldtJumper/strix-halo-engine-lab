#pragma once

#include "strix/ple_lookup.hpp"
#include "strix/ple_rolling.hpp"
#include "strix/ple_direct.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace strix::ple_reclaim {

// STRIX_PLE_RECLAIMABLE unset/auto: try the admitted route. Plain CLI setup can fall back to warm;
// leased MTP setup fails closed until a zero-credit fallback envelope is admitted. =1 requires
// admission (fail closed); =0 disables it. QPF2K defaults track auto provenance and exclude it for GSQ owners.
// When admitted, the file-backed PLE n-gram table is not kept warm in the page cache.
// Its unregistered interior pages are dropped once at setup (before the large allocations), the known-prefill UMA
// admission credits them minus a bound on owned O_DIRECT buffers (plus decode allowance and margin).
// Plain and leased MTP prompt phases use capped async lookahead/cache and H-sized
// CPU source leases. MTP calibration/decode use the fixed owned decode LRU.
// The PLE data path (hash, gather, staging, kernels) reads the same file bytes.
enum class Mode { off, automatic, required };
[[nodiscard]] Mode requested() noexcept;

// Half-open page-aligned byte interval of the model file (offsets from the mapping base).
struct Interval {
    std::uint64_t begin = 0, end = 0;
};

// Interior full pages of [table_offset, table_offset + table_bytes) minus every registered interval rounded outward
// to pages. Sorted, disjoint, page aligned. Pure; throws on overflow or a zero page size.
[[nodiscard]] std::vector<Interval> reclaim_intervals(std::uint64_t table_offset, std::uint64_t table_bytes,
    std::uint64_t page, std::vector<Interval> registered);

struct Config {
    std::uintptr_t mapping_base = 0;
    std::uint64_t mapping_bytes = 0;
    std::uint64_t page = 4096;
    std::uint64_t table_offset = 0;      // file offset of the table's first stored row
    std::uint64_t table_rows = 0;
    std::uint64_t stored_row_bytes = 0;  // 90 for IQ4_NL [160, rows]
    std::vector<Interval> reclaim;       // from reclaim_intervals()
    std::uint64_t prompt_begin = 0;       // absolute first position requiring PLE work
    std::uint64_t ahead_rows = 0;        // prefetch window beyond the block just gathered
    std::uint64_t critical_rows = 0;     // GPU prompt block size; ignored by disabled routes
    std::uint64_t decode_allowance_bytes = 0;
    bool credit_only = false; // verified pre-device setup; no prompt readers, hints or staging
    bool bounded_turns = false; // explicit leased MTP CLI source path
    std::string backing_identity;
    std::shared_ptr<const void> backing_owner;
    std::uint64_t mapping_generation = 0;
    std::uint64_t rolling_positions = 0; // zero preserves the historical planner and receipts
    std::uint64_t rolling_cap_bytes = 0; // zero selects the shared 3 GiB owned-buffer default
    bool strict = true;                  // throw on a bound violation (STRIX_PLE_RECLAIM_STRICT=0: log only, diagnosis)
    bool stage_prefetch = true;          // c1: gather the next block into the idle staging slot (STRIX_PLE_STAGE_PREFETCH=0 off)
    // Drops clean unmapped page cache of a file byte range; returns 0 or an errno (MappedModelFile::drop_page_cache).
    std::function<int(std::uint64_t, std::uint64_t)> drop_page_cache;
    std::function<int()> open_direct_read_only; // returns an owned descriptor of this backing
    DirectPageCache::Read test_direct_read; // CPU-only filesystem/fault injection seam
    // Largest resident interior PLE allowed right after the drop (verified with mincore before any credit).
    std::uint64_t drop_residual_bytes = 64ULL << 20U;
};

struct Stats {
    std::uint64_t reclaimable_bytes = 0, touched_bytes = 0, admitted_bytes = 0, credit_bytes = 0;
    std::uint64_t resident_before_drop_bytes = 0, resident_after_drop_bytes = 0;
    std::uint64_t blocks = 0, gathered_rows = 0, gather_ns_total = 0, gather_ns_max = 0;
    std::uint64_t minflt_total = 0, majflt_total = 0, majflt_max = 0;
    std::uint64_t prefetch_pages_considered = 0, prefetch_pages_issued = 0, prefetch_ns_total = 0;
    std::uint64_t resident_samples = 0, resident_max_bytes = 0, resident_last_bytes = 0;
    std::uint64_t hash_mismatches = 0, rows_outside_prompt = 0, prefetched_until = 0;
    // c1 staging prefetch: block k+1 gathered into the idle staging slot while block k runs.
    std::uint64_t stage_hits = 0, stage_misses = 0, stage_rows = 0, stage_ns_total = 0, stage_wait_ns_total = 0;
    std::uint64_t stage_majflt_total = 0, stage_errors = 0;
    // KFD per-process queue eviction time (sum of /sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms; -1 unreadable).
    std::int64_t kfd_evicted_ms_before_drop = -1, kfd_evicted_ms_after_drop = -1, kfd_evicted_ms_final = -1;
    bool final_sampled = false;
    bool kfd_evicted_during_drop = false;   // evicted_ms rose across the setup drop (queues were evicted)
    bool bound_exceeded = false, mincore_failed = false;
};

// Pure prompt accounting, shared by pre-HIP admission and verified runtime setup.
struct PromptPlan {
    std::vector<std::uint32_t> rows;
    Stats stats;
};
[[nodiscard]] PromptPlan plan_prompt(const Config& config, const PleHashContract& contract,
    std::span<const std::uint32_t> prompt);
[[nodiscard]] std::uint64_t decode_allowance(std::uint64_t generate_tokens, std::uint64_t lookups_per_token,
    std::uint64_t heads, std::uint64_t page, std::uint64_t requested_bytes = 0);

// Shared by prediction, verified admission and the source-page allocator.
struct RollingBudget { std::uint64_t cap, margin, admitted; };
[[nodiscard]] std::uint64_t rolling_cap_bytes(const char* value);
[[nodiscard]] RollingBudget rolling_budget(const Config& config, std::uint64_t heads);
// Shared prediction/runtime equation. Capacities include prompt row IDs and the
// issued-page bitmap; source bytes themselves are withheld from reclaim credit.
[[nodiscard]] std::uint64_t owner_bytes(const Config& config, std::uint64_t heads,
    std::uint64_t row_capacity, std::uint64_t issued_capacity);
[[nodiscard]] std::uint64_t planned_owner_bytes(const Config& config, std::uint64_t heads,
    std::uint64_t prompt_positions);

class Prefetcher : public std::enable_shared_from_this<Prefetcher> {
public:
    // credit_only verifies the same credit without activating prompt work. Destroy it
    // before constructing the runtime owner, which must perform a fresh verified drop.
    // Hashes [prompt_begin, prompt.size()) with the original prefix context and builds the touched-page set,
    // then drops the reclaim intervals (MADV_RANDOM, MADV_DONTNEED on this mapping's PTEs, page-cache drop) and starts
    // the helper, which prefetches the first ahead_rows suffix positions.
    // Automatic setup defers diagnostics until success so a rejected attempt can fall back to the warm path silently.
    Prefetcher(Config config, const PleHashContract& contract, std::span<const std::uint32_t> prompt,
        bool automatic = false);
    ~Prefetcher();
    Prefetcher(const Prefetcher&) = delete;
    Prefetcher& operator=(const Prefetcher&) = delete;

    [[nodiscard]] std::uint64_t credit_bytes() const { return (manager_ && !manager_->healthy()) || (rolling_ && !rolling_->healthy()) || (direct_ && !direct_->healthy()) ? 0U : credit_; }
    [[nodiscard]] Stats stats() const;
    [[nodiscard]] std::uint64_t owner_bytes() const {
        return ple_reclaim::owner_bytes(config_, heads_, rows_.capacity(), issued_.capacity());
    }
    [[nodiscard]] std::shared_ptr<ResidencyManager::Turn> begin_turn(ResidencyManager::PromptObserver observer = {});
    [[nodiscard]] bool rolling() const noexcept { return rolling_ != nullptr || manager_ != nullptr; }
    [[nodiscard]] std::uint64_t prompt_begin() const noexcept { return config_.prompt_begin; }
    [[nodiscard]] std::uint64_t gather_positions() const noexcept { return config_.rolling_positions; }
    struct PromptSource {
        std::shared_ptr<const DirectPageCache::View> view;
        std::uintptr_t table = 0;
        std::uint64_t offset = 0;
    };
    [[nodiscard]] PromptSource prompt_source(std::uint64_t first, std::uint64_t count) const;
    [[nodiscard]] bool owns_prompt_source(const DirectPageCache::View& view, std::uintptr_t table,
        std::uint64_t offset) const noexcept {
        return rolling_ && direct_ && direct_->owns(view) && table == config_.mapping_base + config_.table_offset &&
            offset == config_.table_offset;
    }
    [[nodiscard]] bool block_stats() const noexcept;
    void gather_window(std::uint64_t first, std::uint64_t count, const std::function<void()>& copy);
    void gather_range(std::uint64_t first, std::uint64_t count,
        const std::function<void(std::uint64_t, std::uint64_t)>& copy);
    // All file-mapped source readers, including decode, retain their rows until
    // the CPU copy completes. Unknown table bindings refuse on rolling routes.
    void read_row(std::uintptr_t table, std::uint64_t row, std::span<std::byte> out) const;
    void source_copy(std::uintptr_t table, std::span<const std::uint64_t> rows, const std::function<void()>& copy);
    // End of the prefill/run: one last resident sample (decode lookups included) and the KFD eviction reading;
    // throws in strict mode if the residency bound or the prompt hash check failed at any point.
    void finish();
    [[nodiscard]] std::string receipt_json() const;

    // Prefill PLE gather of rows [first_position, first_position + hashes.size()) finished (host thread, gather
    // threads joined). Checks the launcher's row ids against the precomputed ones, records the gather receipts and
    // moves the prefetch window. Throws only in strict mode after a bound violation.
    struct GatherReceipt {
        std::uint64_t bytes_read = 0, read_wait_ns = 0, copy_ns = 0, hash_ns = 0;
        std::uint64_t gather_ns = 0, minflt = 0, majflt = 0;     // exposed: hash + any gather on the launch thread
        std::uint64_t staged_rows = 0;                           // rows whose staging was filled by the c1 worker (hit)
        bool stage_attempted = false;                            // a c1 worker targeted this launch
        bool stage_error = false;                                // the c1 worker failed (inline gather used)
        std::uint64_t stage_ns = 0, stage_wait_ns = 0, stage_majflt = 0;  // worker time (hidden), join wait (exposed)
    };
    void on_gather(std::uint64_t first_position, std::span<const PleHashResult> hashes, const GatherReceipt& receipt);
    // c1: the precomputed row ids of prompt positions [first, first + n) with n = min(max_count, prompt - first)
    // (n x heads() ids in out); returns n, 0 if first is past the prompt or the staging prefetch is off. The caller
    // passes its window capacity, so the next block hits whatever size the plan gives it (T, a merged or balanced
    // partial block of STRIX_QPF2K_SCHED) as long as it starts at first and fits the window.
    [[nodiscard]] std::uint64_t stage_rows(std::uint64_t first, std::uint64_t max_count, std::vector<std::uint32_t>& out) const;
    [[nodiscard]] std::uint32_t heads() const noexcept { return heads_; }
    // Register before publishing the c1 thread; finish/destruction drain even
    // a final task whose staging is never consumed by another tiled launch.
    [[nodiscard]] bool claim_stage_prime();
    void stage_begin();
    void stage_end(std::exception_ptr error = {}) noexcept;

private:
    void report_setup() const noexcept;
    void helper();
    std::exception_ptr drain_stages();
    void prefetch_until(std::uint64_t target);
    std::uint64_t resident_bytes(bool* failed) const;
    bool in_reclaim(std::uint64_t page_index) const noexcept;

    Config config_;
    std::uint32_t heads_ = 0;
    std::uint64_t positions_ = 0;
    std::vector<std::uint32_t> rows_;     // positions_ x heads_ row ids
    std::shared_ptr<DirectPageCache> direct_; // outlives pipeline/manager callbacks
    std::unique_ptr<RollingPipeline> rolling_;
    std::shared_ptr<ResidencyManager> manager_;
    std::vector<std::uint64_t> issued_;   // bitmap over interior page indices [first_page_, last_page_)
    std::uint64_t first_page_ = 0, last_page_ = 0;
    std::uint64_t credit_ = 0, admitted_ = 0;
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::uint64_t target_ = 0, issued_until_ = 0;
    bool stop_ = false, sample_ = false;
    bool finishing_ = false, stage_primed_ = false;
    std::uint64_t stages_ = 0;
    std::exception_ptr stage_error_;
    Stats stats_;
    std::thread thread_;
};

// Process-wide hook used by the prefill PLE launcher; null unless the flag is active.
void set_active(Prefetcher* prefetcher) noexcept;
[[nodiscard]] Prefetcher* active() noexcept;

}  // namespace strix::ple_reclaim
