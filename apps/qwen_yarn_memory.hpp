#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace strix::decode_cli {
// These are host admission/stop policies, independent of rotary and kernel plans.
inline constexpr std::uint64_t yarn_legacy_host_reserve = UINT64_C(8) << 30U;
inline constexpr std::uint64_t prefill_physical_margin = UINT64_C(4) << 30U;
struct PrefillPhysicalShortfall { std::uint64_t retained, incremental; };
inline PrefillPhysicalShortfall prefill_physical_shortfall(std::uint64_t total,std::uint64_t available,
    std::uint64_t retained,std::uint64_t incremental) {
    const auto deficit=[](std::uint64_t have,std::uint64_t need) {
        if (have < prefill_physical_margin)
            return need > UINT64_MAX-(prefill_physical_margin-have) ? UINT64_MAX : need+(prefill_physical_margin-have);
        return need > have-prefill_physical_margin ? need-(have-prefill_physical_margin) : 0U;
    };
    return {deficit(total,retained),deficit(available,incremental)};
}
inline bool prefill_physical_margin_fits(std::uint64_t total,std::uint64_t available,
    std::uint64_t retained,std::uint64_t incremental) {
    const auto shortfall=prefill_physical_shortfall(total,available,retained,incremental);
    return !shortfall.retained && !shortfall.incremental;
}
// The inputs of a refused margin check and how far each side is short (0 = that side fits).
inline std::string prefill_physical_margin_detail(std::uint64_t total,std::uint64_t available,
    std::uint64_t retained,std::uint64_t incremental) {
    const auto shortfall=prefill_physical_shortfall(total,available,retained,incremental);
    return "mem_total=" + std::to_string(total) + " mem_available=" + std::to_string(available) +
        " retained=" + std::to_string(retained) + " incremental=" + std::to_string(incremental) +
        " margin=" + std::to_string(prefill_physical_margin) +
        " retained_shortfall=" + std::to_string(shortfall.retained) +
        " incremental_shortfall=" + std::to_string(shortfall.incremental);
}
inline constexpr std::uint64_t yarn_memory_stop_bytes = UINT64_C(2) << 30U;

inline std::optional<std::uint64_t> yarn_parse_mem_available(std::istream& input) {
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        std::string name, unit;
        std::uint64_t kib = 0;
        if (row >> name && name == "MemAvailable:") {
            if (!(row >> kib >> unit) || unit != "kB" ||
                kib > std::numeric_limits<std::uint64_t>::max() / 1024U) return {};
            return kib * 1024U;
        }
    }
    return {};
}
inline std::optional<std::uint64_t> yarn_mem_available() {
    std::ifstream input("/proc/meminfo");
    return yarn_parse_mem_available(input);
}

class YarnMemoryPressure final : public std::runtime_error {
public:
    explicit YarnMemoryPressure(std::string receipt)
        : std::runtime_error("YaRN MemAvailable watchdog stopped at a safe boundary; "
              "MemAvailable below 2 GiB or observation unavailable"), receipt_(std::move(receipt)) {}
    const std::string& receipt() const noexcept { return receipt_; }
private:
    std::string receipt_;
};

// Only the sampler reads meminfo. Boundaries normally pay one atomic load; they
// never synchronize a device or change a launch/commit plan. A trip is latched.
class YarnMemoryWatchdog final {
public:
    using Source = std::function<std::optional<std::uint64_t>()>;
    explicit YarnMemoryWatchdog(Source source = yarn_mem_available,
        std::chrono::milliseconds interval = std::chrono::milliseconds(100))
        : source_(std::move(source)), interval_(interval) {
        if (interval_.count() <= 0) throw std::invalid_argument("watchdog interval must be positive");
        sample();
        check();
        worker_ = std::thread([this] {
            std::unique_lock lock(mutex_);
            while (!wake_.wait_for(lock, interval_, [this] { return stopping_; })) {
                lock.unlock();
                sample();
                lock.lock();
                if (tripped_.load(std::memory_order_acquire)) break;
            }
        });
    }
    ~YarnMemoryWatchdog() { stop(); }
    YarnMemoryWatchdog(const YarnMemoryWatchdog&) = delete;
    YarnMemoryWatchdog& operator=(const YarnMemoryWatchdog&) = delete;
    void stop() noexcept {
        { std::lock_guard lock(mutex_); stopping_ = true; }
        wake_.notify_all();
        if (worker_.joinable()) worker_.join();
    }
    void check() const {
        if (tripped_.load(std::memory_order_acquire)) {
            std::lock_guard lock(mutex_);
            throw YarnMemoryPressure(receipt_locked());
        }
    }
    void complete() {
        stop();
        check();
        sample(); // final boundary must observe pressure even on a short run
        check();
    }
    // Publication and the sampler's trip latch share the same lock. An in-flight
    // unit can drain, but it cannot append committed outputs after a latched trip.
    template<class Publish> void publish(Publish&& action) {
        std::lock_guard lock(mutex_);
        if (tripped_.load(std::memory_order_acquire)) throw YarnMemoryPressure(receipt_locked());
        std::forward<Publish>(action)();
    }
    std::string receipt() const {
        std::lock_guard lock(mutex_);
        return receipt_locked();
    }
    bool tripped() const noexcept { return tripped_.load(std::memory_order_acquire); }
private:
    void sample() noexcept {
        const auto start = std::chrono::steady_clock::now();
        std::optional<std::uint64_t> available;
        try { available = source_(); } catch (...) {} // lost protection fails closed
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count();
        std::lock_guard lock(mutex_);
        ++samples_;
        read_ns_ += static_cast<std::uint64_t>(ns);
        if (available) minimum_ = std::min(minimum_, *available);
        if (!available || *available < yarn_memory_stop_bytes) {
            unavailable_ = !available;
            trip_available_ = available.value_or(0);
            tripped_.store(true, std::memory_order_release);
        }
    }
    std::string receipt_locked() const {
        return "\"yarn_memory_watchdog\":{\"threshold_bytes\":" + std::to_string(yarn_memory_stop_bytes) +
            ",\"poll_ms\":" + std::to_string(interval_.count()) + ",\"tripped\":" +
            (tripped_.load(std::memory_order_acquire) ? "true" : "false") +
            ",\"observation_unavailable\":" + (unavailable_ ? "true" : "false") +
            ",\"trip_available_bytes\":" + std::to_string(trip_available_) +
            ",\"minimum_available_bytes\":" + (minimum_ == UINT64_MAX ? "null" : std::to_string(minimum_)) +
            ",\"samples\":" + std::to_string(samples_) + ",\"read_ns\":" + std::to_string(read_ns_) + "}";
    }
    Source source_;
    std::chrono::milliseconds interval_;
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::thread worker_;
    std::atomic<bool> tripped_{false};
    bool stopping_ = false, unavailable_ = false;
    std::uint64_t minimum_ = UINT64_MAX, trip_available_ = 0, samples_ = 0, read_ns_ = 0;
};

inline thread_local YarnMemoryWatchdog* active_yarn_memory_watchdog = nullptr;
class YarnMemoryWatchdogScope final {
public:
    explicit YarnMemoryWatchdogScope(YarnMemoryWatchdog* watchdog)
        : previous_(std::exchange(active_yarn_memory_watchdog, watchdog)) {}
    ~YarnMemoryWatchdogScope() { active_yarn_memory_watchdog = previous_; }
    YarnMemoryWatchdogScope(const YarnMemoryWatchdogScope&) = delete;
    YarnMemoryWatchdogScope& operator=(const YarnMemoryWatchdogScope&) = delete;
private:
    YarnMemoryWatchdog* previous_;
};
inline void check_yarn_memory() {
    if (active_yarn_memory_watchdog) active_yarn_memory_watchdog->check();
}
template<class Publish> void publish_yarn_output(Publish&& action) {
    if (active_yarn_memory_watchdog) active_yarn_memory_watchdog->publish(std::forward<Publish>(action));
    else std::forward<Publish>(action)();
}
inline std::string yarn_memory_receipt(bool complete = false) {
    if (complete && active_yarn_memory_watchdog) active_yarn_memory_watchdog->complete();
    check_yarn_memory();
    return active_yarn_memory_watchdog ? "  " + active_yarn_memory_watchdog->receipt() + ",\n" : std::string{};
}
} // namespace strix::decode_cli
