// CPU tests of the lazy host promotion policy (strix/ds41_host_promote.hpp): room above the floor and under the cap,
// promotion only with room for one whole layer, the OFF switch, unknown telemetry, the deficit at a request start, and
// the LRU order of demotions (only promoted layers: the load-time host tier is never in the LRU).
#include "strix/ds41_host_promote.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

using namespace strix::ds41;
namespace {
void check(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}
constexpr std::uint64_t GiB = 1ULL << 30;
HostMemoryRoom room(std::uint64_t physical, std::uint64_t commit) { return {commit, physical, true}; }
}  // namespace

int main() {
    try {
        HostPromotePolicy on{true, 20 * GiB, 0};
        const std::uint64_t layer = 4 * GiB;  // one whole layer (the largest streamed part)
        // room = min(physical - floor - 4 GiB, (commit - floor) / 2, cap - promoted)
        check(host_promote_room(room(60 * GiB, 200 * GiB), on, 0) == 36 * GiB, "physical bound");
        check(host_promote_room(room(200 * GiB, 60 * GiB), on, 0) == 20 * GiB, "commit bound");
        check(host_promote_room(room(24 * GiB, 200 * GiB), on, 0) == 0, "physical at the margin");
        check(host_promote_room(room(10 * GiB, 200 * GiB), on, 0) == 0, "physical below the margin");
        check(host_promote_room(room(200 * GiB, 10 * GiB), on, 0) == 0, "commit below the floor");
        // promotion only when the room holds one whole layer (inclusive)
        check(host_promote_admit(host_promote_room(room(28 * GiB, 200 * GiB), on, 0), layer), "room == one layer");
        check(!host_promote_admit(host_promote_room(room(28 * GiB - 1, 200 * GiB), on, 0), layer), "room one byte short");
        check(!host_promote_admit(host_promote_room(room(200 * GiB, 28 * GiB - 1), on, 0), layer), "commit short (2n + floor)");
        check(host_promote_admit(host_promote_room(room(200 * GiB, 28 * GiB), on, 0), layer), "commit exact (2n + floor)");
        check(!host_promote_admit(1ULL << 62, 0), "a zero-byte layer is never promoted");
        // the cap counts the promoted bytes already held
        HostPromotePolicy capped{true, 20 * GiB, 10 * GiB};
        check(host_promote_room(room(200 * GiB, 200 * GiB), capped, 0) == 10 * GiB, "cap");
        check(host_promote_admit(host_promote_room(room(200 * GiB, 200 * GiB), capped, 6 * GiB), layer), "cap leaves one layer");
        check(!host_promote_admit(host_promote_room(room(200 * GiB, 200 * GiB), capped, 7 * GiB), layer), "cap leaves less than a layer");
        check(host_promote_room(room(200 * GiB, 200 * GiB), capped, 12 * GiB) == 0, "over the cap");
        // OFF switch and unknown telemetry: never
        HostPromotePolicy off = on;
        off.enabled = false;
        check(host_promote_room(room(200 * GiB, 200 * GiB), off, 0) == 0, "off");
        check(!host_promote_admit(host_promote_room(room(200 * GiB, 200 * GiB), off, 0), layer), "off never admits");
        check(host_promote_room(HostMemoryRoom{}, on, 0) == 0, "unknown telemetry declines");
        // deficit at a request start: below floor + 4 GiB physical, or below the floor in commit
        check(host_promote_deficit(room(30 * GiB, 30 * GiB), on) == 0, "above the floor");
        check(host_promote_deficit(room(24 * GiB, 30 * GiB), on) == 0, "at the floor");
        check(host_promote_deficit(room(21 * GiB, 30 * GiB), on) == 3 * GiB, "physical deficit");
        check(host_promote_deficit(room(100 * GiB, 15 * GiB), on) == 5 * GiB, "commit deficit");
        check(host_promote_deficit(room(18 * GiB, 15 * GiB), on) == 6 * GiB, "the larger deficit");
        check(host_promote_deficit(HostMemoryRoom{}, on) == 0, "unknown telemetry demotes nothing");
        // the floor is reached again only with room for another layer: no promotion right after a demotion boundary
        check(!host_promote_admit(host_promote_room(room(24 * GiB, 200 * GiB), on, 0), layer), "hysteresis of one layer");

        // LRU: layers 30, 31, 32 promoted in order, 30 used again -> 31 is the least recently used
        HostPromoteLru lru;
        std::uint64_t clock = 0;
        lru.add(30, 2 * GiB, ++clock);
        lru.add(31, 2 * GiB, ++clock);
        lru.add(32, 2 * GiB, ++clock);
        check(lru.size() == 3 && lru.bytes() == 6 * GiB, "lru size / bytes");
        lru.touch(30, ++clock);
        check(lru.victims(0).empty(), "no deficit, no victims");
        check(lru.victims(1) == std::vector<std::uint32_t>{31}, "one victim: the least recently used");
        check(lru.victims(2 * GiB) == std::vector<std::uint32_t>{31}, "a layer covers its own bytes");
        check((lru.victims(2 * GiB + 1) == std::vector<std::uint32_t>{31, 32}), "two victims in LRU order");
        check((lru.victims(100 * GiB) == std::vector<std::uint32_t>{31, 32, 30}), "every promoted layer at most");
        lru.touch(7, ++clock);  // a load-time host layer is never added: touching it changes nothing
        check(!lru.contains(7) && lru.size() == 3, "load-time host layers are not in the LRU");
        for (const auto l : lru.victims(100 * GiB)) check(l >= 30U, "only promoted layers are demoted");
        lru.remove(31);
        check(!lru.contains(31) && lru.bytes() == 4 * GiB, "remove");
        check(lru.victims(1) == std::vector<std::uint32_t>{32}, "next least recently used");
        lru.add(31, 2 * GiB, ++clock);  // promoted again: the most recent
        check((lru.victims(100 * GiB) == std::vector<std::uint32_t>{32, 30, 31}), "re-promotion is the most recent");
        lru.add(31, 2 * GiB, ++clock);  // the same layer twice is one entry
        check(lru.size() == 3 && lru.bytes() == 6 * GiB, "no duplicate entries");
        HostPromoteLru ties;
        ties.add(12, GiB, 5);
        ties.add(11, GiB, 5);
        check((ties.victims(2 * GiB) == std::vector<std::uint32_t>{11, 12}), "ties: lower layer first");
    } catch (const std::exception& e) {
        std::cerr << "FAIL ds41_host_promote_tests: " << e.what() << "\n";
        return 1;
    }
    std::cout << "PASS DS41 host promotion: room (floor, commit, cap), one whole layer, OFF switch, telemetry, deficit, LRU demotion order\n";
    return 0;
}
