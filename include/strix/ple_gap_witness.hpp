#pragma once

// PLE GPU-gap witness (diagnostic only; STRIX_PLE_GAP_WITNESS=1 or STRIX_PLE_ROLLING_STATS=1).
// Two timing events on the execution stream bracket a host PLE gather. The stream reaches the first event when
// the work queued before the gather is done and the second when the host has finished the gather, so their
// elapsed time is the stream interval the gather left uncovered (event overhead included). Pairs are read only
// after an existing completion point (Api::query, never a synchronization).
//
// Api: static bool create(Event&); static void destroy(Event&); static bool record(Event&, Stream);
//      static bool query(Event&) (true = complete); static bool elapsed(float& ms, Event& a, Event& b).

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace strix::ple_gap {

inline const char* environment(const char* name) noexcept {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv: read-only diagnostics switch
#endif
    return std::getenv(name);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
}

[[nodiscard]] inline bool enabled_from(const char* gap, const char* stats) noexcept {
    return (gap && std::strcmp(gap, "1") == 0) || (stats && std::strcmp(stats, "1") == 0);
}
[[nodiscard]] inline bool enabled() noexcept {
    static const bool on = enabled_from(environment("STRIX_PLE_GAP_WITNESS"), environment("STRIX_PLE_ROLLING_STATS"));
    return on;
}

template <class Api, std::size_t Slots = 16>
class Site {
public:
    using Event = typename Api::Event;
    using Stream = typename Api::Stream;
    explicit Site(const char* name, bool on = enabled()) noexcept : name_(name), on_(on) {}
    Site(const Site&) = delete;
    Site& operator=(const Site&) = delete;
    ~Site() {
        // No event calls here: the runtime may already be torn down. Unread pairs are reported as pending.
        unsigned long long pending = 0;
        for (const auto& p : ring_) pending += p.state == State::live ? 1U : 0U;
        if (!n_ && !dropped_ && !abandoned_ && !pending) return;
        const auto avg = [](double v, unsigned long long n) { return n ? v / static_cast<double>(n) : 0.0; };
        std::fprintf(stderr, "ple-gap %s: n=%llu gpu-gap-ms=%.3f (%.4f/call, max %.3f) host-gather-ms=%.3f (%.4f/call) "
            "demand-read-wait-ms=%.3f fence-ms=%.3f | miss n=%llu gpu-gap-ms=%.3f (%.4f/call) host-gather-ms=%.3f | "
            "hit n=%llu gpu-gap-ms=%.3f (%.4f/call) | dropped=%llu pending=%llu abandoned=%llu\n",
            name_, n_, gap_ms_, avg(gap_ms_, n_), gap_max_, host_ms_, avg(host_ms_, n_), wait_ms_, fence_ms_,
            n_miss_, gap_miss_ms_, avg(gap_miss_ms_, n_miss_), host_miss_ms_,
            n_ - n_miss_, gap_ms_ - gap_miss_ms_, avg(gap_ms_ - gap_miss_ms_, n_ - n_miss_), dropped_, pending, abandoned_);
    }
    // Reserves a distinct slot and records its first event; returns a ticket (slot + generation) or -1 (off,
    // ring busy, event error). A slot reserved by a begin() without end() (exception path) is reclaimed when the
    // ring comes back to it; a late end() with the old ticket is then ignored.
    int begin(Stream stream) noexcept {
        if (!on_) return -1;
        const std::lock_guard lock(mutex_);
        collect_locked();
        const auto index = next_;
        auto& p = ring_[index];
        if (p.state == State::live) { ++dropped_; return -1; }
        if (p.state == State::reserved) ++abandoned_;
        p.state = State::idle;
        if (!p.created) {
            if (!Api::create(p.a)) return -1;
            if (!Api::create(p.b)) { Api::destroy(p.a); return -1; }
            p.created = true;
        }
        if (!Api::record(p.a, stream)) return -1;
        p.state = State::reserved;
        generation_ = generation_ >= max_generation ? 1U : generation_ + 1U;
        p.generation = generation_;
        next_ = (next_ + 1U) % Slots;
        return static_cast<int>(generation_ * Slots + index);
    }
    void end(int ticket, Stream stream, double host_ms, std::uint64_t wait_ns, double fence_ms) noexcept {
        if (ticket < 0) return;
        const std::lock_guard lock(mutex_);
        auto& p = ring_[static_cast<std::size_t>(ticket) % Slots];
        if (p.state != State::reserved || p.generation != static_cast<std::size_t>(ticket) / Slots) return;   // reclaimed
        if (!Api::record(p.b, stream)) { p.state = State::idle; ++dropped_; return; }
        p.state = State::live; p.host_ms = host_ms; p.wait_ms = static_cast<double>(wait_ns) / 1e6; p.fence_ms = fence_ms;
    }
    // Reads every completed pair; call after an existing synchronization (begin() also does it).
    void collect() noexcept {
        if (!on_) return;
        const std::lock_guard lock(mutex_);
        collect_locked();
    }
    [[nodiscard]] unsigned long long samples() const noexcept { const std::lock_guard lock(mutex_); return n_; }
    [[nodiscard]] unsigned long long abandoned() const noexcept { const std::lock_guard lock(mutex_); return abandoned_; }
    [[nodiscard]] unsigned long long dropped() const noexcept { const std::lock_guard lock(mutex_); return dropped_; }
    [[nodiscard]] double gap_ms() const noexcept { const std::lock_guard lock(mutex_); return gap_ms_; }
private:
    enum class State : std::uint8_t { idle, reserved, live };
    struct Pair {
        Event a{}, b{};
        bool created = false;
        State state = State::idle;
        std::size_t generation = 0;
        double host_ms = 0, wait_ms = 0, fence_ms = 0;
    };
    void collect_locked() noexcept {
        for (auto& p : ring_) {
            if (p.state != State::live || !Api::query(p.b)) continue;
            float gap = 0.0F;
            p.state = State::idle;
            if (!Api::elapsed(gap, p.a, p.b)) { ++dropped_; continue; }
            const auto g = static_cast<double>(gap);
            ++n_; gap_ms_ += g; host_ms_ += p.host_ms; wait_ms_ += p.wait_ms; fence_ms_ += p.fence_ms;
            gap_max_ = std::max(gap_max_, g);
            if (p.wait_ms > 0.0) { ++n_miss_; gap_miss_ms_ += g; host_miss_ms_ += p.host_ms; }
        }
    }
    const char* name_;
    bool on_;
    mutable std::mutex mutex_;
    std::array<Pair, Slots> ring_{};
    std::size_t next_ = 0, generation_ = 0;
    static constexpr std::size_t max_generation = static_cast<std::size_t>(0x7fffffff) / Slots - 1U;
    unsigned long long n_ = 0, n_miss_ = 0, dropped_ = 0, abandoned_ = 0;
    double gap_ms_ = 0, gap_max_ = 0, host_ms_ = 0, wait_ms_ = 0, fence_ms_ = 0, gap_miss_ms_ = 0, host_miss_ms_ = 0;
};

}  // namespace strix::ple_gap
