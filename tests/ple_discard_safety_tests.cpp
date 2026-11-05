#include "strix/ple_discard_safety.hpp"
#include "strix/registered_backing_lifetime.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strix::ple_reclaim;
namespace {
void expect(bool ok) { if (!ok) throw std::runtime_error("discard safety assertion failed"); }
template<class F> void refuses(F f) {
    bool refused = false;
    try { f(); } catch (const std::exception&) { refused = true; }
    expect(refused);
}
template<class F> void safety_refuses(F f) {
    bool refused = false;
    try { f(); } catch (const DiscardSafetyError&) { refused = true; }
    expect(refused); // Generic errors could be swallowed by automatic fallback.
}
void disjointness() {
    const std::array drop{DiscardRange{0x10000,8192,7,19,0x4000}};
    std::vector<RegisteredBacking> regs{{"weight",0x80000,4096,7,19,0x6000}};
    const auto validate = [&] { validate_discard_set(4096,drop,drop,regs); };
    validate(); // Exact touching endpoints, same file, different virtual mappings.
    regs[0].file_offset = 0x5fff; // One byte on the final discarded page.
    safety_refuses(validate);
    regs[0].inode = 20;
    validate(); // Same offsets in another inode are independent.
    regs[0].inode = 19; regs[0].device = 8;
    validate(); // Same inode number on another device is independent.
    regs[0].address = 0x11fff;
    safety_refuses(validate); // Even different files may not overlap in virtual pages.
    regs[0] = {"alias",0x90000,1,7,19,0x3fff};
    validate();
    regs[0].bytes = 2;
    safety_refuses(validate); // Registration rounds both ends outward.
    regs.clear();
    auto advice = drop;
    advice[0].file_offset = 0x4001;
    safety_refuses([&] { validate_discard_set(4096,{},advice,regs); });
    advice = drop; advice[0].bytes = 0;
    safety_refuses([&] { validate_discard_set(4096,drop,advice,regs); }); // No whole-file fadvise.
    advice = drop; advice[0].file_offset = UINT64_MAX - 4095;
    safety_refuses([&] { validate_discard_set(4096,{},advice,regs); });
    safety_refuses([&] { validate_discard_set(0,drop,drop,regs); });
    safety_refuses([&] { validate_discard_set(3000,drop,drop,regs); });
    // A disjoint madvise never licenses an overlapping file-cache discard.
    advice = drop; advice[0].file_offset = 0;
    regs = {{"other-mapping",0x90000,4096,7,19,0}};
    safety_refuses([&] { validate_discard_set(4096,drop,advice,regs); });
    advice[0].file_offset = 0x8000;
    regs[0].file_offset = 0x4000;
    safety_refuses([&] { validate_discard_set(4096,drop,advice,regs); });
    // Failure is unconditional: diagnostic/strict-off paths cannot continue.
    bool discarded = false;
    safety_refuses([&] { validate_discard_set(4096,drop,drop,regs); discarded = true; });
    expect(!discarded);
}
void lock_failure_policy() {
    // Keep main's automatic/off parsing; only the exact explicit =1 is mandatory.
    for (const char* setting : {static_cast<const char*>(nullptr), "", "1", "0", "01", "true", "10"}) {
        const strix::RegisteredPageLockPolicy policy(setting);
        expect(policy.enabled == !(setting && setting[0] == '0'));
        expect(policy.required == (setting && std::string(setting) == "1"));
    }
    for (const auto* failure : {"unreadable RLIMIT_MEMLOCK", "insufficient RLIMIT_MEMLOCK", "mlock2 failed"}) {
        const bool limit_failure = std::string(failure) != "mlock2 failed";
        for (const char* setting : {static_cast<const char*>(nullptr), "", "0", "1"}) {
            strix::RegisteredPageLockPolicy policy(setting);
            unsigned warnings = 0, lock_attempts = 0;
            bool refused = false;
            const auto fail = [&] { policy.failed(failure, [&] { ++warnings; }); };
            try {
                if (policy.enabled && limit_failure) fail();
                for (unsigned range = 0; range < 3U; ++range) {
                    if (policy.enabled) { ++lock_attempts; fail(); }
                }
            } catch (const strix::RegisteredPageLockError& error) {
                refused = true;
                expect(std::string(error.what()) == failure);
            }
            const bool off = setting && setting[0] == '0';
            const bool explicit_request = setting && std::string(setting) == "1";
            expect(refused == explicit_request);
            expect(warnings == (off || explicit_request ? 0U : 1U));
            expect(lock_attempts == (off || limit_failure ? 0U : 1U));
            if (!explicit_request) expect(!policy.enabled); // No later lock attempts.
        }
    }
}
void lifetime() {
    std::vector<std::string> events;
    const auto acquire = [&](bool alias_fails) {
        strix::acquire_registered_backing(
            [&] { events.emplace_back("register"); },
            [&] { events.emplace_back("track"); },
            [&] { events.emplace_back("fresh-alias"); if (alias_fails) throw std::runtime_error("alias"); });
    };
    const auto retire = [&](bool fail, bool drain_fails) {
        return strix::retire_registered_backings(std::array{1},
            [&] { events.emplace_back("drain"); if (drain_fails) throw std::runtime_error("drain"); },
            [&](int) { events.emplace_back("unregister"); return !fail; },
            [&](int) { events.emplace_back("munlock"); });
    };
    acquire(false);
    expect(retire(false,false)); events.emplace_back("discard");
    acquire(false); // Reuse gets a fresh alias, never the old alias.
    expect(events == std::vector<std::string>{"register","track","fresh-alias","drain","unregister","munlock","discard",
        "register","track","fresh-alias"});
    events.clear();
    expect(!retire(true,false));
    expect(events == std::vector<std::string>{"drain","unregister"}); // Owner must retain its mapping.
    events.clear();
    refuses([&] { (void)retire(false,true); });
    expect(events == std::vector<std::string>{"drain"});
    events.clear();
    refuses([&] { acquire(true); });
    expect(retire(false,false));
    expect(events == std::vector<std::string>{"register","track","fresh-alias","drain","unregister","munlock"});
    events.clear();
    expect(!strix::retire_registered_backings(std::array{1,2,3},
        [&] { events.emplace_back("drain"); },
        [&](int range) { events.emplace_back("unregister" + std::to_string(range)); return range != 2; },
        [&](int range) { events.emplace_back("munlock" + std::to_string(range)); }));
    expect(events == std::vector<std::string>{"drain","unregister1","munlock1","unregister2","unregister3","munlock3"});
    // The failed range stays locked even when its neighbors retire successfully.
    events.clear();
    expect(strix::retire_registered_backings(std::vector<int>{},
        [&] { events.emplace_back("drain"); },
        [&](int) { events.emplace_back("unregister"); return true; },
        [&](int) { events.emplace_back("munlock"); }));
    expect(events.empty());
}
}
int main() {
    try { disjointness(); lock_failure_policy(); lifetime(); std::cout << "PLE discard safety and registered backing lifetime tests passed\n"; }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
