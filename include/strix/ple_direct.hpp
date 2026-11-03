#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>
#include <utility>

namespace strix::ple_reclaim {

class DirectError : public std::runtime_error { public: using std::runtime_error::runtime_error; };
class DirectUnsupported : public DirectError { public: using DirectError::DirectError; };

struct DirectActivity {
    std::uint64_t bytes = 0, wait_ns = 0, reads = 0;
    std::uint64_t begin_ns = 0, end_ns = 0;
};

class DirectReadScope {
public:
    explicit DirectReadScope(bool background, DirectActivity* activity = nullptr);
    ~DirectReadScope();
    DirectReadScope(const DirectReadScope&) = delete;
    DirectReadScope& operator=(const DirectReadScope&) = delete;
    [[nodiscard]] static bool background();
private:
    bool previous_;
    DirectActivity* previous_activity_;
};

struct DirectStats {
    std::uint64_t reads = 0, bytes = 0, hits = 0;
    std::uint64_t hidden_wait_ns = 0, exposed_wait_ns = 0, read_ns = 0;
    std::uint64_t peak_owned_bytes = 0, peak_used_bytes = 0, used_bytes = 0;
    std::uint64_t peak_inflight = 0;
    std::uint64_t pending_critical = 0;
    std::uint64_t inflight_hits = 0;   // part of hits: pages found still in flight (the caller joins their read)
};

// Owns its descriptor and a fixed, aligned anonymous arena. The page authority
// reserves source/horizon references before ensure(), and retires them before
// erase(). Neither the reader nor copy() has a model-mapping pointer.
class DirectPageCache {
    struct Impl;
public:
    // Immutable page-to-slot snapshot. Pins already completed pages; copies on
    // joined gather workers need no cache/ledger locks. Retirement refuses while
    // a snapshot lives, and its owner keeps the anonymous arena alive.
    class View {
    public:
        ~View();
        View(const View&) = delete;
        View& operator=(const View&) = delete;
        void copy(std::uint64_t file_offset, std::span<std::byte> out) const;
    private:
        friend class DirectPageCache;
        explicit View(std::shared_ptr<Impl> owner) : owner_(std::move(owner)) {}
        std::shared_ptr<Impl> owner_;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> slots_;
    };
    [[nodiscard]] std::shared_ptr<const View> view(std::span<const std::uint64_t> pages) const;
    [[nodiscard]] bool owns(const View& view) const noexcept { return view.owner_ == impl_; }
    // CPU fault seam uses pread semantics (including errno). Emulation exercises
    // the same completion/retry state machine without requiring kernel support.
    struct TestRing { int setup_errno; bool emulate; };
    using Read = std::function<std::int64_t(int, void*, std::size_t, std::uint64_t)>;
    DirectPageCache(int owned_fd, std::uint64_t file_bytes, std::uint64_t page,
        std::uint64_t capacity, Read test_read = {}, TestRing test_ring = {0, false}, bool credit_only = false);
    ~DirectPageCache();
    DirectPageCache(const DirectPageCache&) = delete;
    DirectPageCache& operator=(const DirectPageCache&) = delete;
    std::uint64_t ensure(std::span<const std::uint64_t> pages, bool background, bool critical = true);
    void erase(std::span<const std::uint64_t> pages);
    void copy(std::uint64_t file_offset, std::span<std::byte> out) const;
    [[nodiscard]] DirectStats stats() const;
    [[nodiscard]] bool healthy() const;
    [[nodiscard]] const char* reader() const noexcept;
    [[nodiscard]] int fallback_errno() const noexcept;
private:
    std::shared_ptr<Impl> impl_;
};

} // namespace strix::ple_reclaim
