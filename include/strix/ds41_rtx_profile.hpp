#pragma once

#include <array>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// DS RTX product profiles: one switch, --profile default|fast|quality (STRIX_DS41_PROFILE), selects every lever.
//   default  today's replica: every exact lever that is the replica's default (lean state, q8-r1, dsrtxk / fuse kernels,
//            auto eviction, controller window + draft-keep + draft-vocab auto, wb-defer, CE pacing, wb-stream, the file
//            tier, residency auto seed / save, warmup graphs in serve). No --profile = default = today, byte for byte.
//            The layer-major exact prompt path (PRO-qualified) is wanted in every profile but is pending: not in this build.
//   fast     default + every further exact lever and every bit-changing lever that has passed its quality qualification. Today none has:
//            the fast-only rows are pending (listed in the receipt and --print-config, never silently used).
//   quality  = default for DS on the RTX (no higher-precision expert pack exists for the replica).
// The prefill service rows (App::prefill_service; the service resolves the default column, no --profile yet): host_promote
// (lazy host promotion of file-tier expert layers, exact) on, --baseline / STRIX_DS41_RTX_BASELINE=1 turn it off.
// One table row per lever (ds41_rtx_profile.cpp): a value per profile, ready (qualified and present) and runnable (the
// implementation is in this build) per profile, kind exact | bit-changing, the CLI options and the environment switch
// that override it, and the source (owner commit, or what a pending column waits for).
// Precedence: explicit CLI (the last of the lever's own options and --baseline) > explicit environment
// (STRIX_DS41_RTX_BASELINE=1 over the lever's own variable, as the replica always did) > profile > builtin.
// A pending column is refused (the builtin form runs, the row is listed as pending); --allow-pending
// (STRIX_DS41_PROFILE_ALLOW_PENDING=1) runs a pending column whose implementation is in this build, for benchmarks only,
// and stamps the receipt "unqualified: <rows>".
namespace strix::ds41::rtx::profile {

enum class Name : unsigned { standard = 0, fast = 1, quality = 2 };  // `standard` prints as "default"
enum class App : unsigned { replica = 1, prefill_service = 2 };
enum class Kind { exact, bit_changing };

struct Option {
    std::string_view flag;   // e.g. "--no-wb-stream"
    std::string_view value;  // what the flag sets the lever to; "*" = the option's argument
};

struct Lever {
    std::string_view name;                  // receipt key
    std::string_view env;                   // the lever's own environment switch (empty: none)
    std::array<Option, 6> options{};        // CLI options that set the lever explicitly (unused entries empty)
    std::array<std::string_view, 3> value;  // per profile (default, fast, quality); empty: the profile leaves it alone
    std::string_view builtin;               // this build with no profile, option or variable
    std::array<bool, 3> ready{};            // qualified and present; false: pending
    std::array<bool, 3> runnable{};         // a pending column's implementation exists in this build
    Kind kind = Kind::exact;
    unsigned apps = 0;                      // mask of App values the lever concerns
    bool baseline = true;                   // --baseline / STRIX_DS41_RTX_BASELINE=1 turn it off
    std::string_view source;                // owner commit / branch; for a pending column, what it waits for
};
[[nodiscard]] std::span<const Lever> levers();

enum class Status { profile, explicit_cli, explicit_env, baseline_cli, baseline_env, pending, unqualified };
[[nodiscard]] const char* status_name(Status status);
[[nodiscard]] const char* kind_name(Kind kind);

struct Entry {
    const Lever* lever = nullptr;
    std::string value;   // the value the resolution asks for ("on", "off", an option's argument, ...)
    Status status = Status::profile;
    std::string wanted;  // pending / unqualified: the profile's value
};

struct Resolution {
    Name name = Name::standard;
    App app = App::replica;
    bool allow_pending = false;
    std::vector<Entry> entries;
    [[nodiscard]] const Entry* find(std::string_view lever) const;
    // "qualified", or "unqualified: <rows>" when --allow-pending ran pending columns
    [[nodiscard]] std::string qualification() const;
    [[nodiscard]] std::vector<std::string> pending() const;
    // {"profile","app","qualification","allow_pending","levers":{name:{value,status,switch,kind,source[,wanted][,effective]}},
    //  "pending":[...],"unqualified":[...]}; `effective` (optional) gives the app's resolved value of a lever ("" = omit)
    [[nodiscard]] std::string json(const std::function<std::string(std::string_view)>& effective = {}) const;
};

[[nodiscard]] Name parse(std::string_view value);  // default|fast|quality, else throws std::invalid_argument
[[nodiscard]] const char* name(Name profile);
[[nodiscard]] const char* app_name(App app);

using Lookup = std::function<const char*(const char*)>;
struct Inputs {
    std::vector<std::string> args;  // the launch options (argv[2..])
    Lookup env;                     // environment lookup; empty: the process environment
};
// --profile NAME (else STRIX_DS41_PROFILE, else default) and --allow-pending (or STRIX_DS41_PROFILE_ALLOW_PENDING=1) out
// of the options; throws std::invalid_argument on a bad name or a missing value.
struct Selection {
    Name name = Name::standard;
    bool allow_pending = false;
};
[[nodiscard]] Selection select(const Inputs& inputs);
// Throws only through select(); a pending column never refuses the run (it is refused as a lever and listed).
[[nodiscard]] Resolution resolve(App app, const Inputs& inputs, std::span<const Lever> table = levers());
[[nodiscard]] Resolution resolve(Name profile, bool allow_pending, App app, const Inputs& inputs,
                                 std::span<const Lever> table = levers());

}  // namespace strix::ds41::rtx::profile
