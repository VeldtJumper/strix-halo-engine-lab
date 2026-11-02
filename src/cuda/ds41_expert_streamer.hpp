#pragma once
// deepseek41 CUDA prefill: routed experts with residency per EXPERT and grouped transport (internal to the driver).
//
// Every layer numbers its 384 experts by a virtual position: its resident experts first (ascending id), then the
// streamed ones (ascending id). Resident experts live in one device buffer per layer ([gate r][up r][down r]); the
// streamed ones are copied, per run, from pinned host memory (packed in virtual order: [gate s][up s][down s]) or
// from the model file into device slots, in groups of G experts (each group's gate, up and down pieces, then an
// event). The MoE of a layer runs range by range (enqueue_moe_layer_ranges): the resident range first, then each
// group once its event fired; its combine is the layer's, so the bits do not depend on where an expert lives.
//
// The whole-layer tiers are the special cases r = 384 (device) and r = 0 (host / file, identity order).
#include "strix/cuda_ds41_common.hpp"
#include "strix/cuda_ds41_moe.hpp"
#include "strix/cuda_ds41_prefill.hpp"
#include "strix/ds41_file_reader.hpp"
#include "strix/ds41_host_promote.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace strix::ds41::cuda::detail {

struct TensorExtents {
    std::uint64_t offset[3]{}, bytes[3]{};  // gate, up, down tensors of a layer in the file
};

struct StreamerPlan {
    std::uint32_t first = 0, end = 0, experts = 384;
    std::uint64_t expert_bytes[3]{};               // one expert of gate, up, down
    std::vector<TensorExtents> extents;            // [layer]
    std::vector<std::vector<std::uint32_t>> order; // [layer] virtual position -> expert id (resident first)
    std::vector<std::uint32_t> resident;           // [layer] experts resident on the device
    std::vector<ExpertTier> stream_tier;           // [layer] where the streamed experts come from (host / file)
    std::uint32_t slots = 2;
    std::uint32_t group_experts = 0;               // experts per transport group (0: one group per layer)
    std::uint64_t piece_bytes = 0;                 // largest single copy (0: a group's tensor piece at once)
    std::uint32_t host_alias = 0;                  // TIMING ONLY: host layers beyond this many reuse earlier copies
    std::uint32_t host_pin_attempts = 3;           // tries per host layer's pinned buffer (the OS may grow the pagefile)
    std::uint32_t host_pin_retry_ms = 1500;        // wait between tries; a layer that still fails streams from the file
    double host_commit_floor_gib = 0.0;            // > 0: commit-floor policy instead of the tries (see PrefillOptions)
    double host_pin_wait_s = 360.0;                // commit-floor policy: total wait for the whole host tier
    bool host_tier_implicit = false;
    // lazy host promotion: a file-tier layer read for a run is kept in a new pinned buffer (the host tier's packed
    // layout) and streams from it in later runs; the least recently used ones go back to the file when the host drops
    // below the floor at a run start (ds41_host_promote.hpp; never with host_alias)
    HostPromotePolicy host_promote;
    bool timing = false;                           // copy start / end events per streamed layer
};

class PinnedBytes {
public:
    PinnedBytes() = default;
    explicit PinnedBytes(std::size_t n);
    // nullptr when cudaHostAlloc keeps failing after `attempts` tries `retry_ms` apart (no exception)
    static std::unique_ptr<PinnedBytes> try_alloc(std::size_t n, std::uint32_t attempts, std::uint32_t retry_ms);
    // nullptr once `deadline` passed (or free physical memory is short) without room for n bytes above the commit floor
    static std::unique_ptr<PinnedBytes> try_alloc_floor(std::size_t n, double floor_gib, std::chrono::steady_clock::time_point deadline,
                                                        std::uint32_t& waits, bool implicit, bool allow_first, HostPinReason& reason);
    ~PinnedBytes();
    PinnedBytes(const PinnedBytes&) = delete;
    PinnedBytes& operator=(const PinnedBytes&) = delete;
    [[nodiscard]] std::uint8_t* get() const noexcept { return p_; }
    [[nodiscard]] std::size_t size() const noexcept { return n_; }

private:
    std::uint8_t* p_ = nullptr;
    std::size_t n_ = 0;
};

class ExpertStreamer {
public:
    ExpertStreamer(FileReader& file, StreamerPlan plan);
    ~ExpertStreamer();
    ExpertStreamer(const ExpertStreamer&) = delete;
    ExpertStreamer& operator=(const ExpertStreamer&) = delete;

    void begin_run();
    // Explicit request schedule, including repeated layer uses across passes. Resident layers are filtered out.
    // Keeps the loader active between passes instead of triggering idle preloads after every pass.
    void begin_run_uses(std::span<const std::uint32_t> layers);
    void end_run();
    struct Layer {
        const std::uint32_t* perm = nullptr;  // device [experts] id -> virtual position (null: identity)
        std::vector<MoeRange> ranges;         // resident range, then the streamed groups (each with its event)
    };
    // The layer's ranges for kernels on `stream`; blocks until the layer's copies are enqueued (not until they
    // landed: the ranges carry the events). release() after the layer's last kernel was enqueued.
    Layer acquire(std::uint32_t layer);
    void release(std::uint32_t layer, cudaStream_t stream);

    struct Stats {
        std::uint64_t bytes_streamed = 0;
        double io_seconds = 0.0, wait_seconds = 0.0;
        double copy_busy_ms = 0.0;               // timing: sum over streamed layers of copy end - copy start
        std::vector<float> copy_end_ms;          // timing: [layer] copy end relative to the run's first copy (-1: none)
        std::vector<float> copy_begin_ms;        // timing: [layer] copy start, same origin (-1: none)
        std::vector<std::uint64_t> copy_bytes;   // [layer] bytes copied into its slot during the run (0: none / preloaded)
        // The between-runs preload this run consumed (the first slots, loaded after the previous run ended; not in
        // bytes_streamed): bytes, host wall time, and the part of it still running after this run began (s).
        std::uint64_t preload_bytes = 0;
        double preload_s = 0.0, preload_after_begin_s = 0.0;
        // lazy host promotion: layers promoted / given back since the previous run ended (its between-runs preload
        // included), promoted layers and bytes held now, and the file-tier bytes this run read (in bytes_streamed)
        std::uint32_t promoted_layers = 0, demoted_layers = 0, promoted_held = 0;
        std::uint64_t promoted_bytes = 0, file_bytes = 0;
    };
    [[nodiscard]] Stats stats() const;           // after the run's work finished
    [[nodiscard]] cudaEvent_t copy_origin() const;  // timing: the event the copy times are relative to
    [[nodiscard]] std::size_t device_bytes() const noexcept { return device_bytes_; }
    [[nodiscard]] const HostPinStats& host_pin_stats() const { return host_pin_stats_; }
    [[nodiscard]] std::size_t pinned_bytes() const noexcept { return pinned_bytes_; }  // the promoted layers included
    struct Promoted {
        std::uint32_t layers = 0, promoted = 0, demoted = 0;  // held now; promoted / given back since the load
        std::uint64_t bytes = 0;
    };
    [[nodiscard]] Promoted promoted() const;
    [[nodiscard]] std::uint32_t host_pin_failed_layers() const noexcept { return host_pin_failed_; }
    [[nodiscard]] double host_pin_seconds() const noexcept { return host_pin_s_; }
    [[nodiscard]] std::uint32_t host_pin_waits() const noexcept { return host_pin_waits_; }
    [[nodiscard]] std::size_t slot_count() const noexcept { return slots_.size(); }
    [[nodiscard]] std::uint64_t slot_bytes() const noexcept { return slot_bytes_; }
    [[nodiscard]] std::uint32_t max_streamed() const noexcept { return smax_; }
    [[nodiscard]] const StreamerPlan& plan() const noexcept { return p_; }

private:
    struct Event;
    struct Slot {
        std::unique_ptr<DeviceBuffer<std::uint8_t>> buf;
        std::vector<std::unique_ptr<Event>> group_ready;
        std::unique_ptr<Event> free;
        std::uint64_t loaded_run = 0;
        std::size_t loaded_use = ~std::size_t{0};
        std::int64_t freed_use = -1;
        std::uint64_t freed_run = 0;
        std::int32_t layer = -1;
    };
    struct Staging {
        std::unique_ptr<PinnedBytes> buf;
        std::unique_ptr<Event> done;
        bool used = false;
    };

    [[nodiscard]] std::uint32_t streamed(std::uint32_t l) const { return p_.experts - p_.resident[l]; }
    [[nodiscard]] std::uint32_t groups(std::uint32_t l) const;
    [[nodiscard]] std::size_t use_of(std::uint32_t layer) const;
    void file_to_device(std::uint64_t offset, std::uint64_t bytes, std::uint8_t* dst);
    void file_to_host(std::uint64_t offset, std::uint64_t bytes, std::uint8_t* dst, PinnedBytes& page);
    void file_to_pinned(std::uint64_t offset, std::uint64_t bytes, std::uint8_t* dst);
    void copy_piecewise(std::uint8_t* dst, const std::uint8_t* src, std::uint64_t bytes);
    void load_group(Slot& s, std::uint32_t layer, std::uint32_t g, std::uint8_t* fill);
    std::unique_ptr<PinnedBytes> promote_buffer(std::uint32_t layer);  // nullptr: no promotion of this load
    void demote_under_pressure();
    std::uint64_t load_into(Slot& s, std::uint32_t layer);  // returns the bytes enqueued
    void loop();

    FileReader& file_;
    StreamerPlan p_;
    std::uint32_t smax_ = 0;
    std::uint64_t slot_bytes_ = 0;
    std::vector<std::unique_ptr<DeviceBuffer<std::uint8_t>>> resident_;
    std::vector<std::unique_ptr<DeviceBuffer<std::uint32_t>>> perm_;
    std::vector<std::unique_ptr<PinnedBytes>> host_;
    std::vector<const std::uint8_t*> host_view_;
    std::vector<std::uint32_t> streamed_;  // layers with streamed experts, ascending
    std::vector<std::uint32_t> uses_;      // active request's streamed uses, in execution order
    std::vector<std::size_t> acquired_;
    std::size_t acquire_cursor_ = 0;
    std::deque<Slot> slots_;
    std::vector<Staging> staging_;
    std::size_t next_staging_ = 0;
    cudaStream_t copy_ = nullptr;
    std::thread worker_;
    mutable std::mutex mutex_;
    std::condition_variable wake_, done_;
    bool stop_ = false, run_ = false;
    std::uint64_t run_id_ = 0;
    std::string error_;
    std::uint64_t bytes_streamed_ = 0;
    double io_seconds_ = 0.0, wait_seconds_ = 0.0;
    std::size_t device_bytes_ = 0;
    std::atomic<std::size_t> pinned_bytes_{0};  // grows / shrinks with the promoted layers (loader thread)
    HostPinStats host_pin_stats_;
    std::uint32_t host_pin_failed_ = 0;   // host layers demoted to the file tier (pinned allocation failed)
    double host_pin_s_ = 0.0;             // time spent allocating the host tier (incl. retries)
    std::uint32_t host_pin_waits_ = 0;    // commit-floor policy: waits for the pagefile to grow
    // timing
    std::unique_ptr<Event> origin_;
    bool origin_recorded_ = false;
    std::vector<std::unique_ptr<Event>> copy_begin_, copy_end_;  // [layer]
    std::vector<bool> copied_;                                   // [layer] copied during this run
    std::vector<std::uint64_t> copy_bytes_;                      // [layer] bytes copied during this run
    // between-runs preload accounting (host clock)
    std::uint64_t preload_after_run_ = 0;   // run id the last preload followed (it serves run id + 1)
    std::uint64_t preload_bytes_ = 0;
    std::chrono::steady_clock::time_point preload_t0_{}, preload_t1_{}, run_begin_t_{};
    bool preload_running_ = false;
    // the preload the current run consumed (set when the loader starts the run: the preload, if any, finished before)
    std::uint64_t consumed_bytes_ = 0;
    double consumed_s_ = 0.0, consumed_after_begin_s_ = 0.0;
    // lazy host promotion: the LRU, its clock and the block flag belong to the loader thread; the counters are read
    // under mutex_ (cumulative; a run reports the difference to the mark taken when it began, which is the count at
    // the previous run's end)
    HostPromoteLru promote_lru_;
    std::uint64_t promote_clock_ = 0;
    bool promote_blocked_ = false;  // a run that gave layers back promotes none
    std::uint32_t promoted_cum_ = 0, demoted_cum_ = 0, promoted_end_ = 0, demoted_end_ = 0, promoted_mark_ = 0, demoted_mark_ = 0;
    std::uint32_t promoted_held_ = 0;
    std::uint64_t promoted_bytes_ = 0, file_bytes_ = 0;
};

}  // namespace strix::ds41::cuda::detail
