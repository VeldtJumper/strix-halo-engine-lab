#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace strix {
class RegisteredPageLockError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
// Preserve the automatic lock/off policy. Only an explicit =1 requires success.
struct RegisteredPageLockPolicy {
    bool enabled;
    bool required;
    explicit RegisteredPageLockPolicy(const char* setting)
        : enabled(!(setting && setting[0] == '0')),
          required(setting && std::string_view(setting) == "1") {}

    template<class Warn>
    void failed(const std::string& reason, Warn warn) {
        if (required) throw RegisteredPageLockError(reason);
        warn();
        enabled = false; // As before, stop locking subsequent ranges after failure.
    }
};
// Track a successful registration before querying its alias. An alias failure
// must unwind through the owner, never release an untracked live registration.
template<class Register, class Track, class Alias>
void acquire_registered_backing(Register reg, Track track, Alias alias) {
    reg();
    track();
    alias(); // A new registration always obtains a fresh device alias.
}
// Consumers must already be closed to new work. A failed drain/unregister
// forbids unlocking/discarding the affected backing; the owner retains it.
template<class Ranges, class Drain, class Unregister, class Unlock>
bool retire_registered_backings(const Ranges& ranges, Drain drain, Unregister unreg, Unlock unlock) {
    if (ranges.empty()) return true;
    drain();
    bool complete = true;
    for (const auto& range : ranges) {
        if (unreg(range)) unlock(range);
        else complete = false;
    }
    return complete;
}
} // namespace strix
