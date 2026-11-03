#include "profile.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace strix::gemma4::profile {
namespace {
constexpr unsigned engine = unsigned(App::engine), decode_bench = unsigned(App::decode_bench),
                   prefill_bench = unsigned(App::prefill_bench), all_apps = engine | decode_bench | prefill_bench;
// Profile columns: default, fast, quality. One row per lever; an owner merge (or a passed gate) flips its
// own `runnable` / `ready` column (and, if its switch name differs, `env`) on its own row.
constexpr Lever table[] = {
    {.name = "decode_identity", .env = "STRIX_G4_IDENTITY", .value = {"g4-v1", "g4-a8-v1", ""}, .builtin = "g4-v1",
     .required = {false, true, false}, .ready = {true, true, true}, .runnable = {true, true, true},
     .kind = Kind::bit_changing, .apps = engine | decode_bench,
     .source = "g4-a8-v1 (A8G32 activations, integer dots; byte gates exact); "
               "qualified: held-out fidelity bands met"},
    // Drafter: <model dir>/drafter-mtp when present, --draft-mtp DIR overrides, --draft off or
    // STRIX_G4_DRAFT=0 switches it off. Missing weights never fail: decode runs exact without it.
    {.name = "draft", .env = "STRIX_G4_DRAFT", .option = "--draft-mtp", .value = {"mtp-auto", "mtp-auto", "mtp-auto"},
     .builtin = "off", .required = {false, false, false}, .ready = {true, true, true}, .runnable = {true, true, true},
     .kind = Kind::exact, .apps = engine, .asset = true, .off = "0",
     .source = "MTP assistant drafter: ids == OFF on g4-v1, g4-a8-v1 and g4-exl3-v3; C2..C8 auto k<=1 (STRIX_G4_DRAFT_MULTI=0 off)"},
    {.name = "draft_mode", .env = "STRIX_G4_DFLASH", .option = "--dflash", .value = {"auto", "auto", "auto"},
     .builtin = "auto", .required = {false, false, false}, .ready = {true, true, true}, .runnable = {true, true, true},
     .kind = Kind::exact, .apps = engine, .off = "0", .source = "draft admission auto: C1 drafts, C2..C8 plain"},
    // C1 MTP drafter-confidence window: the window only selects how many drafted tokens are verified (ids == OFF).
    // Unset = per drafter (drafter-mtp/config.json confidence_tau, else the builtin tau for known weights, else 0.6);
    // a number forces tau; STRIX_G4_MTP_CONF=0 switches it off (AutoPolicy windows as before).
    {.name = "draft_window", .env = "STRIX_G4_MTP_CONF", .value = {"auto", "auto", "auto"}, .builtin = "auto",
     .required = {false, false, false}, .ready = {true, true, true}, .runnable = {true, true, true},
     .kind = Kind::exact, .apps = engine, .off = "0",
     .source = "gemma4/e4b-mtp2: confidence-gated C1 MTP window, ids == OFF; gate EN untuned 85.6 -> 97.6 tok/s"},
    // Cost-aware long-context window: when the stream's measured draft cost per depth is >= 0.04 x its measured one-row
    // pass (long contexts), start at one depth, extend while P >= 0.6, verify P >= 0.15. Exact. 0 = off.
    {.name = "draft_window_long", .env = "STRIX_G4_MTP_CONF_LONG", .value = {"auto", "auto", "auto"}, .builtin = "auto",
     .required = {false, false, false}, .ready = {true, true, true}, .runnable = {true, true, true},
     .kind = Kind::exact, .apps = engine, .off = "0",
     .source = "cost-aware long-context MTP window (measured draft cost per depth vs one-row pass), ids == OFF"},
    // Prompt lookup (n-gram) drafting, exact: auto = per-request selector beside the MTP drafter (drafts only on a strong
    // match, backs off on misses; measured equal to the drafter alone) and lookup drafting when no drafter is loaded
    // (agent prompts 48.2 -> 114.9, 49.3 -> 122.9 tok/s); 0 = off.
    {.name = "draft_lookup", .env = "STRIX_G4_LOOKUP", .value = {"auto", "auto", "auto"}, .builtin = "auto",
     .required = {false, false, false}, .ready = {true, true, true}, .runnable = {true, true, true},
     .kind = Kind::exact, .apps = engine, .off = "0",
     .source = "prompt-lookup drafting, ids == OFF; beside the MTP drafter equal, without a drafter agent 48.2 -> 114.9 tok/s"},
    // HTTP serve: up to eight chat requests share one coordinator and decode together; each request's ids equal the
    // request served alone. STRIX_G4_SERVE_BATCH=0 serves one chat job at a time.
    {.name = "serve_batch", .env = "STRIX_G4_SERVE_BATCH", .value = {"continuous", "continuous", "continuous"},
     .builtin = "continuous", .required = {false, false, false}, .ready = {true, true, true}, .runnable = {true, true, true},
     .kind = Kind::exact, .apps = engine, .off = "0",
     .source = "continuous batching in serve: concurrent requests == each alone (greedy, seeded sampling, thinking "
               "budget, streaming) on g4-v1, g4-a8-v1 and g4-exl3-v3"},
    {.name = "exl3_route", .option = "--exl3-arm", .value = {"", "", "pack"}, .builtin = "off",
     .required = {false, false, true}, .ready = {true, true, true}, .runnable = {true, true, true}, .kind = Kind::pack,
     .apps = all_apps, .asset = true, .source = "EXL3 pack route (--exl3-arm)"},
    // Fixed per pack and session from the arm and --streams (crossover table in exl3/format.hpp);
    // STRIX_G4_EXL3_IDENTITY=v1|v2|v3 forces it.
    {.name = "exl3_identity", .env = "STRIX_G4_EXL3_IDENTITY", .value = {"", "", "auto"}, .builtin = "auto",
     .required = {false, false, false}, .ready = {false, false, true}, .runnable = {false, false, true},
     .kind = Kind::pack, .apps = engine,
     .source = "auto-select per pack and --streams: M4/M6 g4-exl3-v3 always, M5 v3 at >= 8 streams, else v1"},
    // FAST targets WMMA prefill. g4-wmma-pf-v1 failed its frozen held-out gate, so the FAST column is pending
    // g4-wmma-pf-v2 qualification; --allow-pending runs the v1 implementation, unqualified.
    {.name = "prefill_route", .env = "STRIX_G4_PREFILL", .value = {"exact", "wmma", "exact"}, .builtin = "exact",
     .required = {false, true, false}, .ready = {true, false, true}, .runnable = {true, false, true},
     .kind = Kind::bit_changing, .apps = engine, .target = "g4-wmma-pf-v2",
     .source = "pending: g4-wmma-pf-v2 qualification; the engine runs prefill-exact as many-row chunks "
               "(STRIX_G4_PREFILL_EXACT_MANYROW=0 = eight-row passes) until then"},
    // EXL3 packs: the pack's held-out bands were measured on g4-exl3-wmma-pf-v1 prompts (panel WMMA linears,
    // canonical attention); QUALITY targets that identity once its receipt lands, exact runs until then.
    {.name = "exl3_prefill", .env = "STRIX_G4_EXL3_PREFILL", .value = {"", "", "wmma"}, .builtin = "exact",
     .required = {false, false, false}, .ready = {false, false, false}, .runnable = {false, false, true},
     .kind = Kind::bit_changing, .apps = engine, .target = "g4-exl3-wmma-pf-v1",
     .source = "pending: g4-exl3-wmma-pf-v1 qualification receipt (EXL3 FP16 panels on the WMMA prefill GEMMs); "
               "exact runs the many-row EXL3 core until then"},
    // The prefill bench (DevicePrefill) implements only g4-wmma-pf-v1 linears: its builtin is unqualified, so the
    // bench refuses every profile without --allow-pending (measure exact prefill through the decode app's prompt).
    {.name = "prefill_linears", .env = "STRIX_G4_PREFILL", .value = {"exact", "wmma", "exact"}, .builtin = "wmma",
     .required = {true, true, true}, .ready = {false, false, false}, .runnable = {false, true, false},
     .kind = Kind::bit_changing, .apps = prefill_bench, .builtin_unqualified = true,
     .target = "g4-wmma-pf-v2",
     .source = "the prefill bench runs only g4-wmma-pf-v1 linears (unqualified; FAST target g4-wmma-pf-v2): measure "
               "prefill-exact through strix-gemma4-decode prompt prefill"},
    {.name = "prefill_attention", .env = "STRIX_G4_PREFILL_ATTN", .value = {"canonical", "wmma", "canonical"},
     .builtin = "canonical", .required = {false, true, false}, .ready = {true, false, true}, .runnable = {true, true, true},
     .kind = Kind::bit_changing, .apps = prefill_bench, .target = "g4-wmma-pf-v2",
     .source = "pending: g4-wmma-pf-v2 qualification (WMMA attention)"},
    {.name = "prefill_canonical_kernel", .env = "STRIX_G4_PREFILL_CANON_KERNEL", .value = {"1", "1", "1"},
     .builtin = "1", .required = {false, false, false}, .ready = {true, true, true}, .runnable = {true, true, true},
     .kind = Kind::exact, .apps = prefill_bench,
     .source = "whole-chunk canonical attention, byte-equal to the 8-row loop"},
};
const char* environment(const Lookup& env, const char* name) { return env ? env(name) : std::getenv(name); }
void escape(std::string& out, std::string_view text) {
    out += '"';
    for (const char c : text) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (static_cast<unsigned char>(c) < 0x20) {
            constexpr char digits[] = "0123456789abcdef";
            out += "\\u00"; out += digits[(c >> 4) & 15]; out += digits[c & 15];
        } else out += c;
    }
    out += '"';
}
const std::string* lookup(const Pairs& pairs, std::string_view key) {
    const std::string* found = nullptr;
    for (const auto& [k, v] : pairs) if (k == key) found = &v;
    return found;
}
} // namespace

std::span<const Lever> levers() { return table; }

const char* status_name(Status status) {
    switch (status) {
    case Status::set: return "profile";
    case Status::explicit_env: return "explicit-env";
    case Status::explicit_cli: return "explicit-cli";
    case Status::builtin: return "builtin";
    case Status::pending: return "pending";
    case Status::unqualified: return "unqualified";
    case Status::absent: return "absent";
    case Status::off: return "off";
    }
    return "unknown";
}
const char* kind_name(Kind kind) {
    switch (kind) {
    case Kind::exact: return "exact";
    case Kind::bit_changing: return "bit-changing";
    case Kind::pack: return "pack";
    }
    return "unknown";
}
Name parse(std::string_view value) {
    if (value == "default") return Name::standard;
    if (value == "fast") return Name::fast;
    if (value == "quality") return Name::quality;
    throw std::invalid_argument("profile must be default, fast or quality");
}
const char* name(Name profile) {
    switch (profile) {
    case Name::standard: return "default";
    case Name::fast: return "fast";
    case Name::quality: return "quality";
    }
    return "unknown";
}
const char* app_name(App app) {
    switch (app) {
    case App::engine: return "engine";
    case App::decode_bench: return "decode-bench";
    case App::prefill_bench: return "prefill-bench";
    }
    return "unknown";
}
const Entry* Resolution::find(std::string_view lever) const {
    for (const auto& entry : entries) if (entry.lever->name == lever) return &entry;
    return nullptr;
}
bool Resolution::active(std::string_view lever) const {
    const auto* entry = find(lever);
    return entry && (entry->status == Status::set || entry->status == Status::unqualified || entry->status == Status::explicit_env ||
                     entry->status == Status::explicit_cli) &&
           entry->value != entry->lever->builtin;
}
std::string Resolution::drafter() const {
    const auto* entry = find("draft");
    if (!entry) return "n/a";
    switch (entry->status) {
    case Status::off: return "off (switch)";
    case Status::absent: return "absent";
    case Status::builtin: return "off";
    case Status::pending: return "pending (" + std::string(entry->lever->source) + ")";
    case Status::unqualified: return "on (" + entry->value + ", unqualified)";
    default: return "on (" + entry->value + ")";
    }
}
std::string Resolution::qualification() const {
    std::string rows;
    for (const auto& entry : entries)
        if (entry.status == Status::unqualified) rows += (rows.empty() ? "" : ",") + std::string(entry.lever->name);
    return rows.empty() ? "qualified" : "unqualified: " + rows;
}
std::string Resolution::json() const {
    std::string out = "{\"profile\":";
    escape(out, profile::name(name));
    out += ",\"app\":";
    escape(out, app_name(app));
    out += ",\"qualification\":";
    escape(out, qualification());
    if (find("draft")) {
        out += ",\"drafter\":";
        escape(out, drafter());
    }
    out += ",\"levers\":{";
    std::string pending, unqualified;
    bool first = true;
    for (const auto& entry : entries) {
        if (!first) out += ',';
        first = false;
        escape(out, entry.lever->name);
        out += ":{\"value\":";
        escape(out, entry.value);
        out += ",\"status\":";
        escape(out, status_name(entry.status));
        out += ",\"switch\":";
        escape(out, entry.lever->env.empty() ? entry.lever->option : entry.lever->env);
        out += ",\"kind\":";
        escape(out, kind_name(entry.lever->kind));
        out += ",\"source\":";
        escape(out, entry.lever->source);
        out += '}';
        if (entry.status == Status::pending) {
            if (!pending.empty()) pending += ',';
            pending += "{\"lever\":";
            escape(pending, entry.lever->name);
            pending += ",\"target\":";
            escape(pending, entry.target);
            pending += ",\"runs\":";
            escape(pending, entry.value);
            pending += '}';
        }
        if (entry.status == Status::unqualified) {
            if (!unqualified.empty()) unqualified += ',';
            escape(unqualified, entry.lever->name);
        }
    }
    out += "},\"pending\":[" + pending + "],\"unqualified\":[" + unqualified + "]}";
    return out;
}

Resolution resolve(Name profile, App app, const Inputs& inputs, std::span<const Lever> levers) {
    Resolution out;
    out.name = profile;
    out.app = app;
    const auto* allow = environment(inputs.env, "STRIX_G4_PROFILE_ALLOW_PENDING");
    out.allow_pending = inputs.allow_pending || (allow && std::string_view(allow) == "1");
    const auto column = unsigned(profile);
    for (const auto& lever : levers) {
        if (!(lever.apps & unsigned(app))) continue;
        const auto& wanted = lever.value[column];
        const bool required = lever.required[column] && !wanted.empty();
        const auto refuse = [&](std::string_view why) {
            throw std::invalid_argument(std::string("profile '") + profile::name(profile) + "' requires " +
                                        std::string(lever.name) + " = " + std::string(wanted) + ": " + std::string(why));
        };
        const auto switched_off = [&](std::string_view v) { return !lever.off.empty() && (v == lever.off || v == "off"); };
        const auto* given = lookup(inputs.cli, lever.name);
        const auto* asset = lever.asset ? lookup(inputs.assets, lever.name) : nullptr;
        const char* exported = lever.env.empty() ? nullptr : environment(inputs.env, std::string(lever.env).c_str());
        Entry entry{&lever, std::string(wanted), Status::set, {}};
        if ((given && switched_off(*given)) || (!given && exported && switched_off(exported))) {
            // Off switches win over the profile and over asset discovery.
            entry.value = "off";
            entry.status = Status::off;
        } else if (given) {
            entry.value = *given;
            entry.status = Status::explicit_cli;
        } else if (exported && !lever.asset) {
            entry.value = exported;
            entry.status = Status::explicit_env;
        } else if (exported) {
            // A non-off env on an asset lever asks for it; the path still comes from discovery.
            if (asset) { entry.value = *asset; entry.status = Status::explicit_env; }
            else { entry.value = std::string(lever.builtin); entry.status = Status::absent; }
        } else if (wanted.empty()) {
            continue;
        } else if (!lever.ready[column]) {
            if (out.allow_pending && lever.runnable[column]) {
                // Benchmarking only: run the pending implementation and stamp the receipt.
                entry.status = Status::unqualified;
                if (lever.asset) {
                    if (asset) entry.value = *asset;
                    else { entry.value = std::string(lever.builtin); entry.status = Status::absent; }
                }
            } else if (lever.builtin_unqualified) {
                // A bench tool whose fallback is itself unqualified: refuse rather than run it silently.
                refuse(std::string(lever.source) + (lever.runnable[column] ?
                    " (--allow-pending runs the unqualified implementation, benchmarking only)" : ""));
            } else {
                // Never refuse for a pending row: it is not applied, the builtin keeps running, the
                // receipt lists the declared target and apply() prints one stderr line.
                entry.target = std::string(lever.target.empty() ? wanted : lever.target);
                entry.value = std::string(lever.builtin);
                entry.status = Status::pending;
            }
        } else if (wanted == lever.builtin) {
            entry.status = Status::builtin;
        } else if (lever.asset) {
            if (asset) entry.value = *asset;
            else if (required) refuse("pass " + std::string(lever.option) + (app == App::engine ? "" : " (not available in this app)"));
            else { entry.value = std::string(lever.builtin); entry.status = Status::absent; }
        }
        if ((entry.status == Status::explicit_env || entry.status == Status::explicit_cli) && !lever.asset &&
            entry.value != lever.builtin) {
            // An explicit request for a value that a column declares but no column can run here is refused.
            bool declared = false, runnable = false;
            for (unsigned p = 0; p < 3; ++p)
                if (lever.value[p] == entry.value) { declared = true; runnable |= lever.runnable[p]; }
            if (declared && !runnable)
                throw std::invalid_argument(std::string(lever.env.empty() ? lever.option : lever.env) + "=" + entry.value +
                                            " is not in this build (" + std::string(lever.source) + ")");
        }
        out.entries.push_back(std::move(entry));
    }
    return out;
}

std::vector<std::string> Resolution::notices() const {
    std::vector<std::string> out;
    for (const auto& entry : entries)
        if (entry.status == Status::pending)
            out.push_back(std::string("profile ") + profile::name(name) + ": " + std::string(entry.lever->name) + " pending " +
                          entry.target + " (" + entry.value + " runs)");
    return out;
}

void apply(const Resolution& resolution) {
    for (const auto& line : resolution.notices()) std::cerr << line << '\n';
    for (const auto& entry : resolution.entries) {
        const auto& lever = *entry.lever;
        if ((entry.status != Status::set && entry.status != Status::unqualified) || lever.env.empty() || lever.asset ||
            !lever.off.empty())
            continue;
        const std::string name(lever.env);
#ifdef _WIN32
        if (_putenv_s(name.c_str(), entry.value.c_str()) != 0)
#else
        if (setenv(name.c_str(), entry.value.c_str(), 0) != 0)
#endif
            throw std::runtime_error("cannot export profile switch " + name);
    }
}

bool take_option(int argc, char** argv, int& i, std::optional<Name>& out) {
    if (std::string_view(argv[i]) != "--profile" || i + 1 >= argc) return false;
    const std::string_view next = argv[i + 1];
    if (next != "default" && next != "fast" && next != "quality") return false;
    out = parse(next);
    ++i;
    return true;
}

} // namespace strix::gemma4::profile
