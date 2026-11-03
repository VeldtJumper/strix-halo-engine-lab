#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Product profiles: one name selects every lever's switch.
//   default  exact g4-v1 decode, MTP drafter (auto), exact prefill
//   fast     named bit-changing identities (int8 activations, WMMA prefill) + drafter, once qualified
//   quality  EXL3 pack route + drafter
// Each lever is one table row (profile.cpp). A profile column that is not qualified yet
// (`ready = false`: owner branch not merged, gate not passed, promotion open) is pending: it is never
// applied and never makes the profile refuse; the builtin keeps running, the receipt lists the row with
// its declared target and one stderr line says so. The profile applies every ready row. `--allow-pending`
// (or STRIX_G4_PROFILE_ALLOW_PENDING=1) applies a pending column whose implementation exists in this
// build (`runnable`) for benchmarking only, and stamps receipts "unqualified: <rows>". An explicit
// environment variable or CLI option always wins over the profile and is recorded as explicit (or `off`).
// Refusals: a required asset that is missing (QUALITY without --exl3-arm), an explicit request for a
// value whose code is not in this build, and a bench whose only fallback is itself unqualified.
namespace strix::gemma4::profile {

enum class Name : unsigned { standard = 0, fast = 1, quality = 2 }; // `standard` prints as "default"
enum class App : unsigned { engine = 1, decode_bench = 2, prefill_bench = 4 };
enum class Kind { exact, bit_changing, pack };

struct Lever {
    std::string_view name{};               // receipt key
    std::string_view env{};                // switch the profile sets (or reads only, see `off`)
    std::string_view option{};             // CLI option that overrides the lever
    std::array<std::string_view, 3> value{}; // per profile (default, fast, quality); empty: left alone
    std::string_view builtin{};            // behaviour of this build with the switch unset
    std::array<bool, 3> required{};        // asset levers: refuse when the asset is missing
    std::array<bool, 3> ready{};           // qualified and present; false: pending
    std::array<bool, 3> runnable{};        // a pending column's implementation exists in this build
    Kind kind = Kind::exact;
    unsigned apps = 0;                     // mask of App values the lever concerns
    bool asset = false;                    // the value is a path the app supplies (option or discovery)
    bool builtin_unqualified = false;      // the build's fallback is itself unqualified: a pending column refuses
    std::string_view target{};             // declared target of a pending column (receipt, stderr); empty: the value
    std::string_view off{};                // value of `env` (or the option) that switches the lever off;
                                           // such an env is only read, never exported
    std::string_view source{};             // owner commit/branch and, when pending, what it waits for
};
[[nodiscard]] std::span<const Lever> levers();

enum class Status { set, explicit_env, explicit_cli, builtin, pending, unqualified, absent, off };
[[nodiscard]] const char* status_name(Status status);
[[nodiscard]] const char* kind_name(Kind kind);

struct Entry {
    const Lever* lever = nullptr;
    std::string value{};  // effective value (explicit, profile, discovered asset or builtin)
    Status status = Status::set;
    std::string target{}; // pending: the declared target that is not applied yet
};
struct Resolution {
    Name name = Name::standard;
    App app = App::engine;
    bool allow_pending = false;
    std::vector<Entry> entries;
    [[nodiscard]] const Entry* find(std::string_view lever) const;
    // True when the lever is on with a non-builtin value from the profile or an explicit setting.
    [[nodiscard]] bool active(std::string_view lever) const;
    // {"profile":...,"app":...,"qualification":...,"drafter":...,"levers":{...},"pending":[...],"unqualified":[...]}
    [[nodiscard]] std::string json() const;
    // Receipt summary of the `draft` lever: "on (<path>)", "absent", "off (switch)", "pending (...)".
    [[nodiscard]] std::string drafter() const;
    // One line per pending row: "profile fast: prefill_route pending g4-wmma-pf-v2 (exact runs)".
    [[nodiscard]] std::vector<std::string> notices() const;
    // "qualified", or "unqualified: <rows>" when --allow-pending ran pending columns.
    [[nodiscard]] std::string qualification() const;
};

[[nodiscard]] Name parse(std::string_view value); // default|fast|quality, else throws
[[nodiscard]] const char* name(Name profile);
[[nodiscard]] const char* app_name(App app);

using Lookup = std::function<const char*(const char*)>;
using Pairs = std::vector<std::pair<std::string, std::string>>;
struct Inputs {
    Pairs cli;                  // levers set on the command line: {lever name, value}
    Pairs assets;               // discovered assets for asset levers: {lever name, path}
    Lookup env;                 // environment lookup; empty: the process environment
    bool allow_pending = false; // --allow-pending; STRIX_G4_PROFILE_ALLOW_PENDING=1 also enables it
};
// Throws std::invalid_argument naming the pending piece when a required column is unavailable.
[[nodiscard]] Resolution resolve(Name profile, App app, const Inputs& inputs = {}, std::span<const Lever> table = levers());
// Exports the switches of entries with status `set` or `unqualified` to the process environment
// (never asset or off-switch levers). Call before anything reads (and caches) those switches.
void apply(const Resolution& resolution);

// Prompt prefill route for STRIX_G4_PREFILL_IDENTITY (exact|v1|v2|fast) and the pack: "fast" (the fast
// profile's value) runs g4-wmma-pf-v1 only on a GGUF pack whose general.file_type is 15 (MOSTLY_Q4_K_M), the
// pack type its held-out qualification covers, and prefill-exact on any other pack (Q4_K_S, BF16, ...); EXL3
// packs keep their own route. Returns "exact", "v1" or "v2"; throws on an unknown value.
inline constexpr std::int32_t gguf_file_type_q4_k_m = 15;
[[nodiscard]] std::string_view prefill_route(std::string_view requested, std::int32_t gguf_file_type, bool exl3_pack);

// Argument helper for apps whose legacy bare `--profile` means kernel profiling:
// `--profile default|fast|quality` consumes the value and returns true; a bare
// `--profile` (followed by another option or nothing) returns false.
[[nodiscard]] bool take_option(int argc, char** argv, int& i, std::optional<Name>& out);

} // namespace strix::gemma4::profile
