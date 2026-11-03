#pragma once
// Host thread fan-out for the reference code (results never depend on the thread count: every work item is
// computed by exactly one thread in a fixed order).
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace strix::h3::block {

inline unsigned host_threads() {
    if (const char* e = std::getenv("H3_HOST_THREADS")) {
        const int v = std::atoi(e);
        if (v > 0) return static_cast<unsigned>(v);
    }
    const unsigned hw = std::thread::hardware_concurrency();
    return hw ? hw : 4U;
}

// fn(i) for i in [0, n), dynamically scheduled over the host threads.
template <class F> void parallel_for(std::uint64_t n, F&& fn, unsigned threads = 0) {
    if (threads == 0) threads = host_threads();
    if (n == 0) return;
    if (threads <= 1 || n == 1) {
        for (std::uint64_t i = 0; i < n; ++i) fn(i);
        return;
    }
    std::atomic<std::uint64_t> next{0};
    std::exception_ptr err;
    std::mutex err_mu;
    auto work = [&] {
        try {
            for (;;) {
                const std::uint64_t i = next.fetch_add(1);
                if (i >= n) return;
                fn(i);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lk(err_mu);
            if (!err) err = std::current_exception();
            next.store(n);
        }
    };
    const unsigned t = static_cast<unsigned>(std::min<std::uint64_t>(threads, n));
    std::vector<std::thread> pool;
    pool.reserve(t - 1);
    for (unsigned j = 1; j < t; ++j) pool.emplace_back(work);
    work();
    for (auto& th : pool) th.join();
    if (err) std::rethrow_exception(err);
}

}  // namespace strix::h3::block
