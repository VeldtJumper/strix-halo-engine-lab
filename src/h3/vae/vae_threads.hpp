#pragma once
// Exception-safe worker threads for the VAE host code: every task runs inside a try block, the first exception is
// kept, all threads are joined on every path (destructor included), and join() rethrows the kept exception in the
// caller's thread. A worker's exception therefore surfaces as an ordinary exception, never std::terminate.
// Fault injection (tests only): inject_failure(stage, index) makes the named stage throw for that item once.

#include <atomic>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace strix::h3::vae {

class Workers {
public:
    Workers() = default;
    Workers(const Workers&) = delete;
    Workers& operator=(const Workers&) = delete;
    ~Workers() { join_quiet(); }

    template <class F> void run(F&& f) {
        threads_.emplace_back([this, fn = std::forward<F>(f)]() mutable {
            try {
                fn();
            } catch (...) {
                const std::lock_guard<std::mutex> lock(mu_);
                if (!error_) error_ = std::current_exception();
            }
        });
    }
    std::size_t size() const { return threads_.size(); }
    // join the oldest thread (bounded pipelines); its exception is kept for join()
    void join_oldest() {
        if (threads_.empty()) return;
        if (threads_.front().joinable()) threads_.front().join();
        threads_.erase(threads_.begin());
    }
    void join() {
        join_quiet();
        if (error_) {
            std::exception_ptr e = error_;
            error_ = nullptr;
            std::rethrow_exception(e);
        }
    }
    bool failed() {
        const std::lock_guard<std::mutex> lock(mu_);
        return error_ != nullptr;
    }

private:
    void join_quiet() {
        for (auto& t : threads_)
            if (t.joinable()) t.join();
        threads_.clear();
    }
    std::vector<std::thread> threads_;
    std::mutex mu_;
    std::exception_ptr error_;
};

// Test-only fault injection. Stages: 0 call input worker, 1 unpatchify worker, 2 GPU loop (before the call),
// 3 chunk assembly worker. index = call or chunk index within the decode; -1 disarms.
enum class FaultStage : int { call_input = 0, unpatchify = 1, gpu_loop = 2, chunk_assembly = 3 };
inline std::atomic<int>& fault_slot(FaultStage s) {
    static std::atomic<int> slots[4] = {-1, -1, -1, -1};
    return slots[static_cast<int>(s)];
}
inline void inject_failure(FaultStage s, int index) { fault_slot(s).store(index); }
inline void maybe_fail(FaultStage s, int index) {
    int expected = index;
    if (index >= 0 && fault_slot(s).compare_exchange_strong(expected, -1))
        throw std::runtime_error("h3 vae: injected failure (stage " + std::to_string(static_cast<int>(s)) + ", item " + std::to_string(index) + ")");
}

}  // namespace strix::h3::vae
