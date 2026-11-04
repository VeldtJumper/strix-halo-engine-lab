// --drafter auto|on|off: the selection (strix/ds41_rtx_drafter_select.hpp), CPU only.
#include "strix/ds41_rtx_drafter_select.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

using namespace strix::ds41::rtx;

namespace {

int checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) throw std::runtime_error(std::string("drafter select: ") + what);
}

constexpr std::uint64_t GiB = 1ULL << 30;

DrafterSelectIn capped(std::uint64_t vram, std::uint64_t ram, std::uint32_t locations) {
    DrafterSelectIn in;
    in.draft_given = in.wanted = true;
    in.vram_room = drafter_room(vram * GiB, GiB + GiB / 2, 10 * GiB, 0);
    in.ram_room = drafter_room(ram * GiB, GiB, 0, 0);
    in.expert_bytes = 142 * GiB;  // the 40-layer target's routed experts (about)
    in.read_locations = locations;
    return in;
}

}  // namespace

int main() {
    try {
        DrafterMode m = DrafterMode::on;
        check(parse_drafter_mode("auto", m) && m == DrafterMode::automatic, "parse auto");
        check(parse_drafter_mode("off", m) && m == DrafterMode::off && parse_drafter_mode("on", m) && m == DrafterMode::on, "parse on/off");
        check(!parse_drafter_mode("yes", m) && m == DrafterMode::on, "bad value refused");
        // rooms: cap - margin - fixed, an explicit tier only lowers it, nothing = unbounded, a fixed part above the cap = 0
        check(drafter_room(24 * GiB, GiB, 3 * GiB, 0) == 20 * GiB, "room cap");
        check(drafter_room(24 * GiB, GiB, 3 * GiB, 8 * GiB) == 8 * GiB && drafter_room(0, 0, 0, 8 * GiB) == 8 * GiB, "room tier");
        check(drafter_room(0, 0, 5 * GiB, 0) == kRoomUnbounded && drafter_room(4 * GiB, GiB, 5 * GiB, 0) == 0, "room none / overfull");

        // 24+32 and 32+32 on one drive: a file tier is certain -> off
        for (const auto v : {24ULL, 32ULL}) {
            const auto s = drafter_select(capped(v, 32, 1));
            check(!s.on && !s.loaded && s.file_tier_expected && s.room_bytes < s.expert_bytes, "caps, one location: off");
            check(s.reason.find("1 read location") != std::string::npos, "reason names the location");
        }
        // the same caps with stripes on a second drive -> on
        auto s = drafter_select(capped(24, 32, 2));
        check(s.on && s.loaded && s.file_tier_expected && s.read_locations == 2, "caps, two locations: on");
        // no caps (full budget) -> on, no file tier expected
        DrafterSelectIn full;
        full.draft_given = full.wanted = true;
        full.expert_bytes = 142 * GiB;
        s = drafter_select(full);
        check(s.on && s.loaded && !s.file_tier_expected && s.room_bytes == kRoomUnbounded, "no caps: on");
        // one side capped only -> not certain -> on
        auto half = capped(24, 32, 1);
        half.ram_room = kRoomUnbounded;
        check(drafter_select(half).on, "RAM not capped: on");
        // caps that hold every expert -> on
        check(drafter_select(capped(96, 96, 1)).on && !drafter_select(capped(96, 96, 1)).file_tier_expected, "everything fits: on");
        // unknown expert bytes, --no-file-tier -> on
        auto unknown = capped(24, 32, 1);
        unknown.expert_bytes = 0;
        check(drafter_select(unknown).on && !drafter_select(unknown).file_tier_expected, "unknown bytes: on");
        auto no_file = capped(24, 32, 1);
        no_file.file_tier = false;
        check(drafter_select(no_file).on, "--no-file-tier: on");
        // explicit modes win over the rule
        auto on = capped(24, 32, 1);
        on.mode = DrafterMode::on;
        check(drafter_select(on).loaded && drafter_select(on).file_tier_expected, "--drafter on");
        full.mode = DrafterMode::off;
        check(!drafter_select(full).on && !drafter_select(full).loaded && drafter_select(full).reason == "--drafter off", "--drafter off");
        // explicit tier caps alone bound the room
        DrafterSelectIn tiers;
        tiers.draft_given = tiers.wanted = true;
        tiers.vram_room = drafter_room(0, 0, 0, 8 * GiB);
        tiers.ram_room = drafter_room(0, 0, 0, 40 * GiB);
        tiers.expert_bytes = 142 * GiB;
        check(!drafter_select(tiers).on && drafter_select(tiers).room_bytes == 48 * GiB, "explicit tier caps: off");
        // not wanted: never loaded, the rule still reported
        auto idle = capped(24, 32, 2);
        idle.wanted = false;
        s = drafter_select(idle);
        check(s.on && !s.loaded && s.reason.find("not loaded") != std::string::npos, "not wanted");
        const std::string j = drafter_select(capped(24, 32, 1)).json();
        for (const char* k : {"\"mode\":\"auto\"", "\"loaded\":false", "\"reason\":", "\"file_tier_expected\":true", "\"read_locations\":1",
                              "\"expert_bytes\":", "\"room_bytes\":"})
            check(j.find(k) != std::string::npos, "json member");
        check(drafter_select(full).json().find("\"room_bytes\":null") != std::string::npos, "unbounded room = null");

        // plain equivalents of the speculative phases
        check(plain_phase("schats0") == "chats0" && plain_phase("schats8w") == "chats8" && plain_phase("schats0f3") == "chats0", "schats");
        check(plain_phase("schats0b2f1w") == "chats0b2" && plain_phase("spec8") == "exact8" && plain_phase("spec0w") == "exact0", "spec");
        check(plain_phase("chats0").empty() && plain_phase("exact8").empty() && plain_phase("wforce4").empty(), "plain phases stay");

        // read locations: the model's own directory, duplicates and directories without the manifest do not count
        namespace fs = std::filesystem;
        const fs::path root = fs::temp_directory_path() / ("ds41-drafter-select-" + std::to_string(std::rand()));
        fs::create_directories(root / "a");
        fs::create_directories(root / "b");
        fs::create_directories(root / "c");
        const fs::path model = root / "a" / "m.gguf";
        std::ofstream(model).put('x');
        std::ofstream(root / "b" / "m.gguf.stripe.json").put('x');
        std::ofstream(root / "a" / "m.gguf.stripe.json").put('x');
        const std::string b = (root / "b").string(), c = (root / "c").string(), a = (root / "a").string();
        check(read_locations(model, "") == 1, "no weight dirs");
        check(read_locations(model, c) == 1, "a directory without the manifest");
        check(read_locations(model, b) == 2, "one stripe directory");
        check(read_locations(model, b + ";" + b + ";" + a + ";" + c + ";") == 2, "duplicates / own directory ignored");
        std::ofstream(root / "c" / "m.gguf.stripe.json").put('x');
        check(read_locations(model, b + ";" + c) == 3, "two stripe directories");
        std::error_code ec;
        fs::remove_all(root, ec);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
    std::printf("PASS %d checks: modes, rooms, caps x locations, no caps, explicit tiers, json, plain phases, read locations\n", checks);
    return 0;
}
