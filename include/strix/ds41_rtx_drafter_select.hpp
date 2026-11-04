#pragma once

// --drafter auto|on|off (STRIX_DS41_DRAFTER; default auto; --baseline = on): whether the replica loads the DSpark
// drafter. Exact either way: only whether the drafter is loaded and whether requests run DSpark windows or plain decode
// changes; the committed tokens are the same by construction.
//
// auto = OFF iff a file tier is certain AND the expert reads have fewer than 2 read locations; else ON. Measured on the
// PRO (one NVMe drive): 24+32 plain 8.2 tok/s without the drafter vs DSpark 5.9-7.8; 32+32 plain 10.9 vs DSpark 10.2;
// full budget (no file tier) DSpark 105-108 vs plain 26.7; 24+32 with the stripes on 2 drives DSpark 9.2-9.3 (above plain
// on one drive). "A file tier is certain" = the expert room is bounded on both sides (VRAM: --vram-total-gib or an
// explicit --vram-expert-gib; RAM: --ram-total-gib or an explicit --ram-expert-gib) and VRAM room + RAM room < every
// routed expert's bytes. The room is a pre-load estimate that never undercounts: the caps minus their margins minus the
// declared fixed parts known before loading (the trunk's GGUF bytes on the device; nothing on the host), so a doubtful
// case keeps the drafter (today's behaviour). Pure CPU (no CUDA).
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace strix::ds41::rtx {

enum class DrafterMode { automatic, on, off };

inline const char* drafter_mode_name(DrafterMode m) {
    return m == DrafterMode::on ? "on" : m == DrafterMode::off ? "off" : "auto";
}

inline bool parse_drafter_mode(const std::string& v, DrafterMode& out) {
    if (v == "auto") out = DrafterMode::automatic;
    else if (v == "on") out = DrafterMode::on;
    else if (v == "off") out = DrafterMode::off;
    else return false;
    return true;
}

constexpr std::uint64_t kRoomUnbounded = UINT64_MAX;

// one side's expert room: min(cap - margin - fixed, explicit tier) (kRoomUnbounded when neither bounds it; 0 when the
// fixed part does not fit). cap / tier = 0: absent.
inline std::uint64_t drafter_room(std::uint64_t cap, std::uint64_t margin, std::uint64_t fixed, std::uint64_t tier) {
    std::uint64_t r = kRoomUnbounded;
    if (cap) r = cap > margin + fixed ? cap - margin - fixed : 0U;
    if (tier) r = std::min(r, tier);
    return r;
}

struct DrafterSelectIn {
    DrafterMode mode = DrafterMode::automatic;
    bool draft_given = false;   // --draft
    bool wanted = false;        // the drafter would load without the selection (--serve with --draft, or a speculative phase)
    bool file_tier = true;      // the file tier is allowed (--no-file-tier: never a file tier)
    std::uint64_t vram_room = kRoomUnbounded, ram_room = kRoomUnbounded;  // drafter_room of each side
    std::uint64_t expert_bytes = 0;  // every routed expert (layers x experts x one expert); 0 = unknown
    std::uint32_t read_locations = 1;
};

struct DrafterSelect {
    DrafterMode mode = DrafterMode::automatic;
    bool on = true;                  // the selection (a drafter that is wanted loads)
    bool loaded = false;             // wanted && on
    bool file_tier_expected = false;
    std::uint32_t read_locations = 1;
    std::uint64_t expert_bytes = 0;
    std::uint64_t room_bytes = kRoomUnbounded;  // VRAM room + RAM room (kRoomUnbounded: not bounded)
    std::string reason;

    // the members (no braces): mode, loaded, reason, file_tier_expected, read_locations, expert_bytes, room_bytes
    [[nodiscard]] std::string json() const {
        std::ostringstream os;
        std::string r;
        for (const char c : reason) {
            if (c == '"' || c == '\\') r += '\\';
            r += c;
        }
        os << "\"mode\":\"" << drafter_mode_name(mode) << "\",\"loaded\":" << (loaded ? "true" : "false") << ",\"reason\":\"" << r
           << "\",\"file_tier_expected\":" << (file_tier_expected ? "true" : "false") << ",\"read_locations\":" << read_locations
           << ",\"expert_bytes\":" << expert_bytes << ",\"room_bytes\":";
        if (room_bytes == kRoomUnbounded) os << "null";
        else os << room_bytes;
        return os.str();
    }
};

inline DrafterSelect drafter_select(const DrafterSelectIn& in) {
    DrafterSelect s;
    s.mode = in.mode;
    s.read_locations = in.read_locations;
    s.expert_bytes = in.expert_bytes;
    const bool bounded = in.vram_room != kRoomUnbounded && in.ram_room != kRoomUnbounded;
    if (bounded) s.room_bytes = in.vram_room > kRoomUnbounded - in.ram_room ? kRoomUnbounded - 1U : in.vram_room + in.ram_room;
    s.file_tier_expected = in.file_tier && bounded && in.expert_bytes && s.room_bytes < in.expert_bytes;
    if (in.mode == DrafterMode::on) {
        s.reason = "--drafter on";
    } else if (in.mode == DrafterMode::off) {
        s.on = false;
        s.reason = "--drafter off";
    } else if (!bounded) {
        s.reason = "auto: no file tier certain (VRAM or RAM expert room not capped)";
    } else if (!in.file_tier) {
        s.reason = "auto: no file tier (--no-file-tier)";
    } else if (!in.expert_bytes) {
        s.reason = "auto: routed expert bytes unknown (model header not read)";
    } else if (!s.file_tier_expected) {
        s.reason = "auto: every routed expert fits the VRAM + RAM room (no file tier)";
    } else if (in.read_locations >= 2U) {
        s.reason = "auto: file tier with " + std::to_string(in.read_locations) + " read locations (DSpark ahead with 2 drives)";
    } else {
        s.on = false;
        s.reason = "auto: file tier certain on 1 read location (plain decode measured faster)";
    }
    s.loaded = in.wanted && s.on;
    if (!in.wanted) s.reason += in.draft_given ? "; not loaded: no speculative phase" : "; not loaded: no --draft";
    return s;
}

// The read locations of the expert reads: the model file plus each --weight-dirs / STRIX_WEIGHT_DIRS directory (';'
// separated; the model's own directory and duplicates ignored) holding "<model file name>.stripe.json". The same count
// as tierstream's stripe_locations (a stat per directory, nothing opened; directories, not volumes); tierstream has no
// public call for it before a session opens.
inline std::uint32_t read_locations(const std::filesystem::path& primary, const std::string& weight_dirs) {
    if (weight_dirs.empty()) return 1U;
    namespace fs = std::filesystem;
    const auto canonical = [](const fs::path& p) {
        std::error_code ec;
        auto r = fs::weakly_canonical(fs::absolute(p, ec), ec);
        auto s = (ec ? p : r).generic_string();
#ifdef _WIN32
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
#endif
        while (s.size() > 1 && s.back() == '/') s.pop_back();
        return s;
    };
    const fs::path parent = primary.parent_path().empty() ? fs::path(".") : primary.parent_path();
    const std::string own = canonical(parent);
    const fs::path manifest = fs::path(primary.filename().string() + ".stripe.json");
    std::vector<std::string> seen;
    std::uint32_t n = 1;
    for (std::size_t begin = 0; begin <= weight_dirs.size();) {
        auto end = weight_dirs.find(';', begin);
        if (end == std::string::npos) end = weight_dirs.size();
        if (end != begin) {
            const fs::path dir(weight_dirs.substr(begin, end - begin));
            const std::string c = canonical(dir);
            if (c != own && std::find(seen.begin(), seen.end(), c) == seen.end()) {
                seen.push_back(c);
                std::error_code ec;
                if (fs::is_regular_file(dir / manifest, ec)) ++n;
            }
        }
        if (end == weight_dirs.size()) break;
        begin = end + 1;
    }
    return n;
}

// The plain equivalent of a speculative phase when the drafter is not loaded ("" = the phase does not draft):
// schats<K>[b<B>][f<N>][w] -> chats<K>[b<B>], spec<K>[w] -> exact<K>. The committed tokens are the same.
inline std::string plain_phase(const std::string& ph) {
    if (ph.rfind("schats", 0) == 0) {
        std::string s = ph.substr(6);
        if (!s.empty() && s.back() == 'w') s.pop_back();
        if (const auto f = s.find('f'); f != std::string::npos) s.erase(f);
        return "chats" + s;
    }
    if (ph.rfind("spec", 0) == 0) {
        std::string s = ph.substr(4);
        if (!s.empty() && s.back() == 'w') s.pop_back();
        return "exact" + s;
    }
    return {};
}

}  // namespace strix::ds41::rtx
