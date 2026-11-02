#pragma once
#include <cstdint>
namespace strix::ds41::cuda {
// One token, from the library's counters (stats since open, differenced per token).
struct TierTokenStats {
    std::uint32_t lookups = 0;          // true (layer, id) of the token (unique per layer)
    std::uint32_t misses = 0;           // true ids not on the device at their layer's export
    std::uint32_t miss_layers = 0;
    std::uint32_t covered = 0;          // ... of which already fetched by a prefetch (copy in flight at the export)
    std::uint32_t fetched = 0;          // copies issued for true misses
    std::uint32_t prefetched = 0;       // copies issued for predictions
    std::uint32_t resurrected = 0;
    std::uint32_t ring = 0;             // misses served from the demand banks
    std::uint32_t stalled = 0;          // misses that waited for a write-back
    std::uint32_t collected_mid = 0;    // write-backs retired inside the token
    std::uint32_t predicted = 0;        // predicted ids (sum over layers)
    std::uint32_t predicted_true = 0;   // true ids of a layer that were in the previous layer's prediction
    std::uint32_t evicted = 0;          // at the end of the token
    std::uint32_t writebacks_done = 0;
    std::uint32_t free_slots = 0;       // after the boundary
    // window prefetch (tierstream winpf; 0 when off): prior / l+1 prefetch copies issued, used by their token, late (a
    // demand met the copy in flight), reverted (unused, back to the host block); l+1 priced out / skipped (device ahead)
    std::uint32_t prior_issued = 0, prior_used = 0, prior_late = 0, prior_reverted = 0;
    std::uint32_t l1_issued = 0, l1_used = 0, l1_late = 0, l1_reverted = 0, l1_priced_out = 0, l1_skipped_ahead = 0;
    std::uint32_t prior_budget_stops = 0;
    std::uint64_t wb_completed_bytes = 0, wb_cancelled_bytes = 0, wb_boundary_flushed_bytes = 0;
    std::uint64_t wb_outstanding_start_bytes = 0, wb_outstanding_bytes = 0, wb_forced_wait_ns = 0;
    std::uint64_t wb_deferred = 0, wb_cancelled = 0, wb_forced = 0, wb_boundary_flushed = 0;
    std::uint64_t device_timeouts = 0;
    double boundary_sync_us = 0, boundary_select_us = 0, boundary_upload_us = 0, boundary_issue_us = 0;
    std::uint64_t wb_issued_bytes = 0;  // write-back bytes submitted in the token (deferred parts included)
    // three-tier file reads of the token (0 without a file tier): reads / bytes, the service's wall wait for them,
    // stage waits, pinned units dropped to their file, prefetch reads carried into a later service call
    std::uint64_t disk_reads = 0, disk_read_bytes = 0, disk_wait_ns = 0, disk_stage_waits = 0, disk_drops = 0, disk_carried = 0;
};

// One layer of the last token (the library's timeline, dg_diag).
struct TierLayer {
    std::uint32_t ids = 0, misses = 0, covered = 0, fetched = 0, prefetched = 0, predicted_true = 0;
    std::uint32_t covered_waits = 0;
    double gpu_demand_wait_us = -1, export_to_ready_us = -1, dma_first_ms = -1, wb_left_ms = -1;
    double seen_us = 0;       // host: previous layer serviced -> this layer's flag seen
    double serve_us = 0;      // host: flag seen -> the layer served (demand + prefetches)
    double demand_us = 0;     // host: flag seen -> the true misses' copies + ready enqueued
    double pred_wait_us = 0;  // host: waiting for the side branch's prediction flag
    double prefetch_us = 0;   // host: enqueueing the prefetches
    double dma_ms = -1;       // device time of this layer's demand copies + ready (-1: no miss)
};

}
