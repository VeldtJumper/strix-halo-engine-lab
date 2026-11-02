#pragma once

// deepseek41 CUDA MoE site of the streamed-expert replica on tierstream's device-gated decode mode (the one weight cache
// of every engine; design notes). Per layer, in the step's (captured) graph:
//
//   [layer 0]   ts_dg_enqueue_begin
//   [predict]   layer l+1's router on layer l's F32 MoE input on a forked branch, then ts_dg_enqueue_predict(l)
//   ts_dg_enqueue_export(l)   the window's ids (and the promised prediction count)
//   the shared expert
//   ts_dg_enqueue_wait(l)
//   resident routes (id -> slot through the library's table) + routed gate/up + down on the library's arena, combine,
//   join of the predictor branch
//   [last layer] ts_dg_enqueue_end
//
// The host brackets every step with begin_token / end_token (ts_begin / ts_end); end_token throws when the library
// refused the token (the outputs must be discarded). The expert bytes and the GEMVs are the engine's: only where the
// slots live and who fills them is the library's. It replaced the replica's own cache (StreamCache + StreamSite, removed
// after bitwise and A/B/A checks) bit for bit.

#include "strix/cuda_ds41_moe.hpp"
#include "strix/ds41_tier_stats.hpp"
#include "strix/ds41_weight_read_options.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strix::ds41::cuda {

class GgufExpertReader;

// Window prefetch knobs (tierstream ss_dg_winpf_params; design notes).
struct TierWinpf {
    bool enable = false;
    std::uint32_t l1_cap = 1;
    float l1_pmin = 0.4F;
    std::uint32_t prior_cap = 1;
    float prior_pmin = 0.7F;
    bool revert_unused = true;
};

// --disk-queue auto: the library's resolved queue (tierstream SS_DG_DISK_QUEUE_AUTO), 4 on a library without it
inline constexpr std::uint32_t kDiskQueueAuto = 0xFFFFFFFFu;

struct TierSiteOptions {
    WeightReadOptions weight_read;
    std::string model;                     // the GGUF the library reads the experts from
    std::uint32_t layers = 40;
    std::uint32_t slots = 0;               // persistent device slots, spares included
    std::uint32_t spare_slots = 96;        // the library's spare slots D, one global count (dg_spare_slots)
    std::uint32_t spare_target = 96;       // free slots each boundary keeps (<= spare_slots)
    std::uint32_t host_spares = 48;        // spare host blocks (>= 1)
    std::vector<std::uint32_t> seed;       // keys (layer * experts + id) by priority: the first slots - spares start resident
    std::uint32_t split = 192;
    std::uint32_t max_rows = 1;
    std::uint32_t max_predict = 8;         // predicted experts per row at most (set_predict)
    std::uint32_t prefetch_per_layer = 4;
    std::uint32_t prefetch_per_token = 48;
    std::uint64_t device_bytes = 0;        // the library's device budget (the arena + its controls)
    std::uint64_t host_bytes = 0;          // the library's pinned budget
    bool diag = true;                      // keep the per-layer timeline (and the demand copies' dma_ms)
    bool test_hooks = false;               // open with dg_test_hooks (pressure phases only)
    bool flush_ready = false;              // dg_flush_ready: flush the demand stream after each ready signal (WDDM batching)
    bool moe_v1b3 = false;                 // routed experts: v1b gate/up from 2 rows + v3 down (cuda_ds41_moe_v2.hpp; same bits)
    std::vector<std::uint8_t> prior;       // tierstream/hint-prior-v1 bytes for window prefetch (empty = none)
    bool wb_defer = false;                 // tierstream dg_wb_defer (deferred write-back scheduling)
    bool ce_pacing = false;                // one-expert background CE debt (the replica turns it on by default)
    // Three-tier placement (tierstream SS_DG_DISK_TIER; rtx::tier_plan): after the residents, the next `pinned_units`
    // keys by seed priority (then key order) take the pinned tier and the rest stay in the model file (read on demand,
    // through the weight_read stripes when given). UINT32_MAX = every non-resident key pinned (the two-tier cache).
    std::uint32_t pinned_units = 0xFFFFFFFFU;
    bool disk_prefetch = false;            // predictions may read file-tier experts too (off: demand reads only)
    std::uint32_t disk_staging = 8;        // pinned staging buffers of the file reads
    std::uint32_t disk_queue = 0;          // file reads in flight at once (0 = the library default, 1 = one at a time,
                                           // kDiskQueueAuto = the library resolves it from the read locations; 4 without that)
    std::uint32_t inclusive_units = 0;     // device experts that may keep their pinned block as a clean copy (0 = exclusive)
    // tierstream dg_wb_stream: deferred write-backs on the library's own D2H stream behind the layer's ready (true; the
    // replica's default) or on the demand stream (false: queued ahead of the next layer's demand copies). Transport
    // only: the same parts, bytes, destinations, pacing and completion events.
    bool wb_stream = false;
};

class TierSite {
public:
    TierSite(const GgufExpertReader& reader, const TierSiteOptions& options);
    ~TierSite();
    TierSite(const TierSite&) = delete;
    TierSite& operator=(const TierSite&) = delete;

    // Layer `layer`'s MoE site on `stream` (the step's first layer is layer 0; route_other launches another layer's
    // router on this layer's input). shared_join (CudaMoeSite::shared_join): the shared expert runs on the engine's side
    // branch, joined before the combine; hc_* (CudaMoeSite::hc_*): the FFN hc_expand folded into the combine.
    void enqueue(std::uint32_t layer, std::uint32_t rows, const A8Rows& x, const std::uint32_t* ids, const float* weights, float* shared, float* raw, float* y,
                 const std::function<void(cudaStream_t)>& shared_expert,
                 const std::function<void(std::uint32_t, std::uint32_t, float*, std::uint32_t*, float*, cudaStream_t)>& route_other, cudaStream_t stream,
                 cudaEvent_t shared_join = nullptr, const float* hc_residual = nullptr, const float* hc_mixes = nullptr, float* hc_out = nullptr,
                 float* hc_pre_carry = nullptr);
    void set_predict(std::uint32_t k);  // the caller must start a new graph epoch
    // Test knob (pressure phases): demand copies take at most `free_per_token` free slots, the boundary evicts the
    // `evict_recent` most recent (0, 0 = off).
    void set_pressure(std::uint32_t free_per_token, std::uint32_t evict_recent);
    // Between tokens (from the next boundary / token): the spare target and the prefetch budgets.
    void set_budgets(std::uint32_t spare_target, std::uint32_t prefetch_per_layer, std::uint32_t prefetch_per_token);
    // Between tokens: window prefetch on / off (placement only: the same bytes and arithmetic), and the next window's
    // rows (token ids; weights NULL = 1) before begin_token.
    void set_winpf(const TierWinpf& w);
    void hint_window(const std::uint32_t* tokens, std::uint32_t rows);
    [[nodiscard]] bool winpf() const noexcept;

    void begin_token();
    // After the step returned: ts_end (service, boundary, write-backs). Throws if the token was refused.
    void end_token(std::uint32_t rows = 1);
    // After end_token: the token's stats and per-layer timeline (kept out of the timed boundary, as the engine arm's).
    std::vector<TierLayer> collect();
    [[nodiscard]] const TierTokenStats& stats() const noexcept;  // of the last collect

    // Between tokens, quiescent (persistent worker, --reset-cache open): the library's cache back to the placement and
    // policy state ss_open produced (ts_dg_reset_open); the per-token stats baseline is re-read. Returns false when the
    // linked tierstream has no reset API (SS_DG_RESET_OPEN).
    static bool can_reset_cache() noexcept;
    // Between tokens: dg_wb_defer / dg_flush_ready at runtime (ts_dg_set_sched; placement / submission only). Returns
    // false (and changes nothing) when the linked tierstream has no runtime setter (SS_DG_SET_SCHED).
    static bool can_set_sched() noexcept;
    // Between tokens: the device-resident experts (layer * experts + id) in recency order, most recent first
    // (ts_dg_residency; empty when the linked tierstream has none). Cold-start fix #3: written at shutdown, the next
    // load's seed order.
    [[nodiscard]] std::vector<std::uint32_t> residency() const;
    static bool can_set_policy() noexcept;
    static bool can_auto_evict() noexcept;
    void set_policy(std::uint32_t policy);
    void set_sched(bool wb_defer, bool flush_ready);
    void reset_cache();
    // Between tokens: checks `count` random units against the file (ts_check_unit); returns mismatches.
    std::uint32_t verify(std::uint32_t count, std::uint64_t seed);
    [[nodiscard]] std::uint32_t device_experts() const noexcept;  // resident at open
    // Three-tier / inclusive counters since open as a JSON object body ("" when both are off): file reads and their host
    // time, pinned drops, victim discards, clean copies kept / dropped, file-tier and clean-copy units now.
    [[nodiscard]] std::string tier3_json() const;
    static bool can_disk_tier() noexcept;  // the linked tierstream has SS_DG_DISK_TIER
    [[nodiscard]] std::uint64_t device_bytes() const noexcept;
    [[nodiscard]] std::uint64_t host_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::ds41::cuda
