#pragma once

#include "strix/ple_direct.hpp"

#include <cstdint>
#include <condition_variable>
#include <memory>
#include <string>
#include <functional>
#include <map>
#include <list>
#include <mutex>
#include <set>
#include <span>
#include <vector>
#include <thread>
#include <deque>
#include <exception>

namespace strix::ple_reclaim {
class DirectPageCache;

// Zero selects the historical prompt-wide planner. Invalid tuning fails closed.
[[nodiscard]] std::uint64_t rolling_positions(const char* rolling, const char* exact_memory, const char* positions);

struct RollingStats {
    std::uint64_t peak_owned_bytes = 0, lookahead_positions = 0, throttles = 0, split_windows = 0;
    std::uint64_t epochs = 0, prefetched_pages = 0, evicted_pages = 0, peak_resident_bytes = 0, faulted_pages = 0;
};

// Page numbers are file-relative. The callbacks must only operate on reclaimable
// interior pages. Source leases and window references share the same refcounts.
// The synchronous copy callback returns only after all CPU readers have joined;
// staging/DMA consumers subsequently read their own immutable copied bytes.
class RollingPages {
public:
    using Pages = std::vector<std::uint64_t>;
    struct Io {
        std::function<std::uint64_t(std::span<const std::uint64_t>)> prefetch;
        std::function<void(std::span<const std::uint64_t>)> evict;
        std::function<std::uint64_t()> resident;
        std::function<bool(std::uint64_t)> reclaimable;
        // Optional diagnostics: nonresident source pages immediately before CPU demand.
        std::function<std::uint64_t(std::span<const std::uint64_t>)> missing;
        std::function<void(std::span<const std::uint64_t>)> prepare;
        std::shared_ptr<DirectPageCache> owned;
        std::function<void()> finish; // drained request boundary only
        std::function<std::uint64_t(std::span<const std::uint64_t>)> critical = {};
    };
    RollingPages(std::uint64_t h, std::uint64_t page, std::uint64_t offset, std::uint64_t row_bytes,
        std::uint32_t heads, std::span<const std::uint32_t> rows, std::uint64_t admitted,
        std::uint64_t decode_allowance, Io io);
    void gather(std::uint64_t first, std::uint64_t count, const std::function<void()>& copy);
    void source(std::span<const std::uint64_t> rows, const std::function<void()>& copy);
    void finish();
    [[nodiscard]] RollingStats stats() const;
private:
    Pages pages(std::span<const std::uint64_t> rows) const;
    Pages window(std::uint64_t first, std::uint64_t count) const;
    void retain(const Pages& pages);
    void release(const Pages& pages);
    void sample();
    void check() const;
    std::uint64_t h_, page_, offset_, row_bytes_, admitted_, decode_allowance_, frontier_ = 0;
    std::uint32_t heads_;
    std::span<const std::uint32_t> rows_;
    Io io_;
    mutable std::mutex mutex_;
    std::map<std::uint64_t, std::uint32_t> refs_;
    std::set<std::uint64_t> decode_;
    Pages current_, next_;
    std::uint64_t copies_ = 0;
    bool failed_ = false, finished_ = false, gathering_ = false;
    RollingStats stats_;
};

// Plain CLI pipeline. Reservations include hints still in flight. A demand slot
// is withheld from lookahead, so background I/O never consumes copy capacity.
// The helper issues callbacks outside the ledger lock; all CPU copies hold refs.
class RollingPipeline {
public:
    using Pages = RollingPages::Pages;
    RollingPipeline(std::uint64_t h, std::uint64_t page, std::uint64_t offset,
        std::uint64_t row_bytes, std::uint32_t heads, std::span<const std::uint32_t> rows,
        std::uint64_t admitted, std::uint64_t decode, std::uint64_t cap,
        std::uint64_t ahead, RollingPages::Io io, std::uint64_t block_rows = 0);
    ~RollingPipeline();
    void gather(std::uint64_t first, std::uint64_t count, const std::function<void()>& copy);
    void gather_range(std::uint64_t first, std::uint64_t count,
        const std::function<void(std::uint64_t, std::uint64_t)>& copy);
    void source(std::span<const std::uint64_t> rows, const std::function<void()>& copy);
    void advance(std::uint64_t gpu_end);
    [[nodiscard]] bool healthy() const;
    void finish();
    [[nodiscard]] RollingStats stats() const;
    struct BlockIo { DirectActivity critical; std::uint64_t lookahead_bytes = 0; };
    [[nodiscard]] BlockIo block_io(std::uint64_t first, std::uint64_t count) const;
private:
    struct Window { std::uint64_t first, count; Pages pages; };
    Pages pages(std::span<const std::uint64_t> rows) const;
    Pages window(std::uint64_t first, std::uint64_t count) const;
    std::uint64_t fit(std::uint64_t first, std::uint64_t count) const;
    void check() const;
    void sample();
    void retain(const Pages& pages);
    void release(const Pages& pages);
    void consumed();
    bool make_room(const Pages& pages, std::uint64_t limit);
    void helper() noexcept;
    void stop();
    const std::uint64_t h_, page_, offset_, row_bytes_, admitted_, decode_allowance_, cap_, reserve_, ahead_, block_rows_;
    const std::uint32_t heads_;
    const std::span<const std::uint32_t> rows_;
    RollingPages::Io io_;
    mutable std::mutex mutex_;
    mutable std::condition_variable wake_;
    std::map<std::uint64_t, std::uint32_t> refs_;
    std::set<std::uint64_t> decode_;
    std::list<std::uint64_t> decode_lru_;
    std::map<std::uint64_t, std::list<std::uint64_t>::iterator> decode_lru_index_;
    std::deque<Window> lookahead_;
    std::set<std::uint64_t> cold_; // unreferenced owned pages; retained while the cap has space
    std::uint64_t frontier_ = 0, planned_ = 0, target_ = 0, copies_ = 0, gpu_end_ = 0;
    bool gathering_ = false, stopping_ = false, finished_ = false;
    std::exception_ptr error_;
    RollingStats stats_;
    struct IoRecord { std::uint64_t first, count; DirectActivity activity; bool complete = true; };
    std::list<IoRecord> io_records_;
    std::thread thread_;
};

// One eviction authority per immutable backing. Turns share physical page
// references, one nonblocking prefill permit and fixed per-slot decode quotas.
// No callbacks which read table bytes run under the manager mutex.
class ResidencyManager : public std::enable_shared_from_this<ResidencyManager> {
public:
    using PromptObserver = std::function<void(std::uint64_t, std::span<const std::uint64_t>, std::uint64_t, std::uint64_t, std::uint64_t)>;
    class Turn {
    public:
        ~Turn();
        void source(std::uintptr_t table, std::uint64_t generation,
            std::span<const std::uint64_t> rows, const std::function<void()>& copy, std::uint64_t epoch = 0);
        [[nodiscard]] bool owned_sources() const noexcept { return manager_->io_.owned != nullptr; }
        void read_row(std::uint64_t row, std::span<std::byte> out) const;
        // Every R2/R4/R8 owner shares this maximum horizon. Pin all physical
        // rows (including min-p padding) until the complete CPU gather joins.
        // GPU work then uses staging; rejection/replay acquires fresh sources.
        static constexpr std::uint64_t maximum_verify_positions = 8;
        void verify_window(std::uintptr_t table, std::uint64_t generation,
            std::uint64_t positions, std::span<const std::uint64_t> rows, const std::function<void()>& copy);
        // Decode cache warming: admits and reads the pages of `rows` into this turn's decode cache like a verify
        // source, without a copy (no bytes leave the cache; later lookups copy the same immutable bytes). Returns
        // the pages that were not yet in the decode cache. Any thread; serialised by the ledger like source().
        std::uint64_t warm(std::uintptr_t table, std::uint64_t generation, std::span<const std::uint64_t> rows);
        [[nodiscard]] std::uint64_t epoch() const;
        void gather(std::span<const std::uint64_t> rows, const std::function<void()>& copy);
        // Known prompt only: the same ordered, capped I/O pipeline as plain CLI.
        void gather_prompt(std::uint64_t first, std::uint64_t count,
            const std::function<void(std::uint64_t, std::uint64_t)>& copy, std::uint64_t staged_epoch = 0);
        [[nodiscard]] std::shared_ptr<const DirectPageCache::View> prompt_view(std::uint64_t first, std::uint64_t count) const;
        [[nodiscard]] bool owns_prompt_view(const DirectPageCache::View&, std::uintptr_t table,
            std::uint64_t generation, std::uint64_t epoch, std::uint64_t offset) const;
        void stage_begin();
        void stage_end(std::exception_ptr error = {}) noexcept;
        void observe(std::uint64_t first, std::span<const std::uint64_t> rows,
            std::uint64_t ns, std::uint64_t minflt, std::uint64_t majflt);
        void phase(bool prompt); // drained calibration -> prompt -> decode
        void finish();
        void cancel() noexcept;
        [[nodiscard]] bool prompt() const;
        [[nodiscard]] std::uint64_t gather_positions() const noexcept { return manager_->h_; }
        [[nodiscard]] std::uint64_t id() const noexcept { return id_; }
    private:
        friend class ResidencyManager;
        Turn(std::shared_ptr<ResidencyManager> manager, std::uint64_t id, std::shared_ptr<const void> owner)
            : manager_(std::move(manager)), owner_(std::move(owner)), id_(id) {}
        std::shared_ptr<ResidencyManager> manager_;
        std::shared_ptr<const void> owner_; // retains mapping and setup certificate owner
        std::uint64_t id_;
    };
    struct PromptConfig {
        std::span<const std::uint32_t> rows;
        std::uint64_t cap = 0, ahead = 0, block_rows = 0, first_position = 0;
    };
    void configure_prompt(PromptConfig config);
    ResidencyManager(std::string backing, std::uintptr_t table, std::uint64_t generation,
        std::uint64_t h, std::uint64_t page, std::uint64_t offset, std::uint64_t row_bytes,
        std::uint64_t row_count, std::uint32_t heads, std::uint64_t admitted,
        std::uint64_t quota, std::uint32_t slots, RollingPages::Io io);
    ~ResidencyManager();
    [[nodiscard]] std::shared_ptr<Turn> begin(std::shared_ptr<const void> owner = {}, PromptObserver observer = {});
    [[nodiscard]] bool healthy() const;
    void invalidate() noexcept;
    [[nodiscard]] RollingStats stats() const;
private:
    using Pages = RollingPages::Pages;
    struct State {
        bool closing = false, prompt = false;
        std::uint64_t epoch = 1;
        PromptObserver observer;
        std::uint64_t readers = 0, completions = 0, stages = 0;
        std::shared_ptr<RollingPipeline> pipeline;
        // Cache refs and in-flight source refs are separate, even on hits.
        std::list<std::uint64_t> lru;
        std::map<std::uint64_t, std::list<std::uint64_t>::iterator> cache;
        std::map<std::uint64_t, std::uint32_t> sources;
        Pages window;
    };
    State& state(std::uint64_t id);
    Pages pages(std::span<const std::uint64_t> rows) const;
    void retain(const Pages& pages);
    void release(const Pages& pages);
    void clear(State& state);
    void retire_prompt(State& state);
    void sample();
    void source(std::uint64_t id, std::uintptr_t table, std::uint64_t generation,
        std::span<const std::uint64_t> rows, const std::function<void()>& copy, std::uint64_t epoch,
        std::uint64_t* uncached = nullptr);
    void gather(std::uint64_t id, std::span<const std::uint64_t> rows, const std::function<void()>& copy);
    void phase(std::uint64_t id, bool prompt);
    void close(std::uint64_t id, bool cancelled);
    std::string backing_;
    std::uintptr_t table_;
    std::uint64_t generation_, h_, page_, offset_, row_bytes_, row_count_, admitted_, quota_;
    std::uint32_t heads_, slots_;
    RollingPages::Io io_;
    PromptConfig prompt_config_;
    mutable std::mutex mutex_;
    std::condition_variable drained_;
    std::map<std::uint64_t, State> turns_;
    std::map<std::uint64_t, std::uint32_t> refs_;
    std::uint64_t sequence_ = 0, permit_ = 0;
    bool failed_ = false;
    RollingStats stats_;
};
} // namespace strix::ple_reclaim
