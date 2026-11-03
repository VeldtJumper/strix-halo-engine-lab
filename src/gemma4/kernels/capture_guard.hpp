#pragma once
// Graph capture guard (host only, no HIP types).
//
// A HIP graph capture on one queue is invalidated when another host thread
// issues legacy null-stream work or a synchronous copy while it runs, and the
// capturing queue is then left in an error state. A many-row prefill pass can
// run on another thread beside the decode pass (prompt chunks beside decode),
// so a decode pass (or a drafter) must not start a NEW capture while a prefill
// pass is in flight, and a prefill pass must not start while a capture runs.
//
// STRIX_G4_CAPTURE_GUARD (default on, read per call): a capture that is not
// granted runs the same kernels with the same arguments eagerly on its queue
// (bitwise identical) and caches no graph; cached graphs still replay.
// STRIX_G4_CAPTURE_GUARD=0: every capture is granted (previous behaviour).
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string_view>

namespace strix::gemma4::gpu {
struct CaptureGuardState {
    std::atomic<unsigned> prefills{0};
    std::atomic<std::uint64_t> eager_passes{0};
    // Graph counters (process-wide, monotonic): new captures (host ns from begin capture to instantiate) and
    // replays of cached graphs, for the decode target and the MTP drafter.
    std::atomic<std::uint64_t> target_captures{0}, target_capture_ns{0}, target_replays{0};
    std::atomic<std::uint64_t> draft_captures{0}, draft_capture_ns{0}, draft_replays{0};
    std::mutex capture;
};
inline CaptureGuardState& capture_guard_state() {
    static CaptureGuardState state;
    return state;
}
inline bool capture_guard_enabled() {
    const auto* p = std::getenv("STRIX_G4_CAPTURE_GUARD");
    return !p || std::string_view(p) != "0";
}
// Held by a many-row prefill pass for its whole duration. It announces itself
// first, then waits for a capture already running to end (that capture saw no
// prefill and is allowed to finish).
class PrefillInFlight {
  public:
    PrefillInFlight() : on_(capture_guard_enabled()) {
        if (!on_) return;
        auto& s = capture_guard_state();
        s.prefills.fetch_add(1);
        std::lock_guard<std::mutex> wait(s.capture);
    }
    ~PrefillInFlight() {
        if (on_) capture_guard_state().prefills.fetch_sub(1);
    }
    PrefillInFlight(const PrefillInFlight&) = delete;
    PrefillInFlight& operator=(const PrefillInFlight&) = delete;

  private:
    bool on_;
};
// Asked for by a graph capture; while granted it holds the capture lock, so no
// prefill pass starts until release() (call it right after the capture ends).
class CaptureSlot {
  public:
    CaptureSlot() {
        if (!capture_guard_enabled()) { granted_ = true; return; }
        auto& s = capture_guard_state();
        lock_ = std::unique_lock<std::mutex>(s.capture);
        if (s.prefills.load() != 0) {
            lock_.unlock();
            s.eager_passes.fetch_add(1);
            return;
        }
        granted_ = true;
    }
    [[nodiscard]] bool granted() const { return granted_; }
    void release() {
        if (lock_.owns_lock()) lock_.unlock();
    }
    CaptureSlot(const CaptureSlot&) = delete;
    CaptureSlot& operator=(const CaptureSlot&) = delete;

  private:
    std::unique_lock<std::mutex> lock_;
    bool granted_ = false;
};
} // namespace strix::gemma4::gpu
