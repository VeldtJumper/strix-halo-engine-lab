#include "strix/ds41_rtx_profile.hpp"

#include <cstdlib>
#include <stdexcept>

namespace strix::ds41::rtx::profile {
namespace {
constexpr unsigned replica = unsigned(App::replica), service = unsigned(App::prefill_service);
constexpr std::array<bool, 3> all{true, true, true}, none{false, false, false};
// Profile columns: default, fast, quality. One row per lever; a qualification (GPU screen, quality band, owner decision)
// flips its own row's `ready` (and `runnable` once the implementation is merged into this build).
// Every exact row below is ON in all three profiles: they are the replica's defaults today, so `default` (and no
// --profile) is today's configuration byte for byte, and `quality` = `default` (no higher-precision expert pack on the
// RTX). --baseline / STRIX_DS41_RTX_BASELINE=1 still restore every reference form (an explicit setting).
constexpr Lever kTable[] = {
    {.name = "lean_state", .options = {{{"--lean-state", "on"}, {"--no-lean-state", "off"}}},
     .value = {"on", "on", "on"}, .builtin = "on", .ready = all, .runnable = all, .kind = Kind::exact, .apps = replica,
     .source = "8965204e: lean decode state by default (gates PASS, session gate 4051b58d)"},
    {.name = "q8_r1", .options = {{{"--q8-r1", "on"}, {"--q8-r1-off", "off"}, {"--q8-r1-pad", "pad"}, {"--q8-r1-nopad", "on"}}},
     .value = {"on", "on", "on"}, .builtin = "on", .ready = all, .runnable = all, .kind = Kind::exact, .apps = replica,
     .source = "daf741e7: unpadded q8-r1 dense GEMVs (bitwise)"},
    {.name = "dsrtxk", .options = {{{"--dsrtxk", "on"}, {"--dsrtxk-off", "off"}}},
     .value = {"on", "on", "on"}, .builtin = "on", .ready = all, .runnable = all, .kind = Kind::exact, .apps = replica,
     .source = "daf741e7: router v2 form 2, ob-one, shared/attn fork, ob/moe expand, v1b3, hc cluster, attn-h1 (bitwise)"},
    {.name = "fuse", .options = {{{"--fuse", "on"}, {"--fuse-off", "off"}}},
     .value = {"on", "on", "on"}, .builtin = "on", .ready = all, .runnable = all, .kind = Kind::exact, .apps = replica,
     .source = "daf741e7: fused hc / shared-expert decode (cuda-decode-fuse gates, bitwise)"},
    {.name = "decode_forms",
     .options = {{{"--records-deferred", "on"}, {"--records-inline", "off"}, {"--router-pf", "on"}, {"--router-pf0", "off"},
                  {"--qakv-s1", "on"}, {"--qakv-auto", "off"}}},
     .value = {"on", "on", "on"}, .builtin = "on", .ready = all, .runnable = all, .kind = Kind::exact, .apps = replica,
     .source = "daf741e7: deferred step records, router form 2, qa_kv S=1 (bitwise)"},
    {.name = "evict_auto", .options = {{{"--evict", "*"}}},
     .value = {"on", "on", "on"}, .builtin = "on", .ready = all, .runnable = all, .kind = Kind::exact, .apps = replica,
     .source = "bac05355 (engine #77 50bd5ce4 + tierstream #4 b02fa419): scored victims after verify windows, +6.4 %"},
    {.name = "controller",
     .options = {{{"--window", "*"}, {"--controller-prior", "*"}, {"--controller-costs", "*"}, {"--controller-scope", "*"}}},
     .value = {"on", "on", "on"}, .builtin = "on", .ready = all, .runnable = all, .kind = Kind::exact, .apps = replica,
     .source = "daf741e7: controller windows, built-in survival prior, seeded costs, chat scope (committed tokens = greedy)"},
    {.name = "draft_keep", .options = {{{"--draft-keep", "*"}}},
     .value = {"on", "on", "on"}, .builtin = "on", .ready = all, .runnable = all, .kind = Kind::exact, .apps = replica,
     .source = "daf741e7: drafter keep-64 experts per stage (drafts only; committed tokens = greedy)"},
    {.name = "draft_vocab_auto", .options = {{{"--draft-vocab", "*"}}},
     .value = {"on", "on", "on"}, .builtin = "on", .ready = all, .runnable = all, .kind = Kind::exact, .apps = replica,
     .source = "daf741e7: drafter vocabulary auto directory (STRIX_DS41_DRAFT_VOCAB / beside --draft), else full"},
    {.name = "wb_defer", .options = {{{"--wb-defer", "on"}, {"--no-wb-defer", "off"}}},
     .value = {"on", "on", "on"}, .builtin = "on", .ready = all, .runnable = all, .kind = Kind::exact, .apps = replica,
     .source = "daf741e7: tierstream deferred write-backs (dg_wb_defer)"},
    {.name = "ce_pacing", .env = "STRIX_DS41_CE_PACING", .options = {{{"--ce-pacing", "on"}, {"--no-ce-pacing", "off"}}},
     .value = {"on", "on", "on"}, .builtin = "on", .ready = all, .runnable = all, .kind = Kind::exact, .apps = replica,
     .source = "#115: CE write-back pacing (PRO A/B: identical tokens, not slower)"},
    {.name = "wb_stream", .env = "STRIX_DS41_WB_STREAM", .options = {{{"--wb-stream", "on"}, {"--no-wb-stream", "off"}}},
     .value = {"on", "on", "on"}, .builtin = "on", .ready = all, .runnable = all, .kind = Kind::exact, .apps = replica,
     .source = "cbcbab05 (tierstream 719cf5f dg_wb_stream 1): write-backs on their own D2H stream (PRO: not slower)"},
    {.name = "file_tier", .env = "STRIX_DS41_FILE_TIER", .options = {{{"--file-tier", "on"}, {"--no-file-tier", "off"}}},
     .value = {"on", "on", "on"}, .builtin = "on", .ready = all, .runnable = all, .kind = Kind::exact, .apps = replica,
     .source = "c81968ff: three-tier placement, the model file instead of the pinned-tier refusal"},
    {.name = "residency_auto", .options = {{{"--residency-seed", "*"}, {"--residency-save", "*"}}},
     .value = {"on", "on", "on"}, .builtin = "on", .ready = all, .runnable = all, .kind = Kind::exact, .apps = replica,
     .source = "daf741e7: MODEL.residency auto seed (and save in serve; STRIX_DS41_RESIDENCY)"},
    {.name = "warmup_graphs", .options = {{{"--warmup-graphs", "*"}}},
     .value = {"on", "on", "on"}, .builtin = "on", .ready = all, .runnable = all, .kind = Kind::exact, .apps = replica,
     .source = "daf741e7: serve captures the K 0,8 graphs before ready"},
    {.name = "engram_cache", .options = {{{"--engram-cache-mb", "*"}}},
     .value = {"on", "on", "on"}, .builtin = "on", .ready = all, .runnable = all, .kind = Kind::exact, .apps = replica,
     .source = "daf741e7: decode Engram rows from RAM after the first read (512 MB)"},
    // exact and PRO-qualified (about -5 % prompt time), wanted in every profile, but not on this branch yet: pending
    {.name = "prompt_layer_major", .value = {"on", "on", "on"}, .builtin = "off", .ready = none, .runnable = none,
     .kind = Kind::exact, .apps = replica, .baseline = false,
     .source = "pending: owner review (layer-major exact prompt path, PRO-qualified exact, -5 %); not in this build"},
    // prefill service, exact (the same file bytes from pinned memory): file-tier layers stay pinned between requests
    {.name = "host_promote", .env = "STRIX_DS41_HOST_PROMOTE", .options = {{{"--host-promote", "on"}, {"--no-host-promote", "off"}}},
     .value = {"on", "on", "on"}, .builtin = "on", .ready = all, .runnable = all, .kind = Kind::exact, .apps = service,
     .source = "lazy host promotion of streamed file-tier expert layers (exact: the same file bytes; above the host floor)"},
    // fast-only, pending
    {.name = "batch_long", .options = {{{"--batch-max", "*"}}}, .value = {"", "on", ""}, .builtin = "off", .ready = none,
     .runnable = {false, true, false}, .kind = Kind::exact, .apps = service, .baseline = false,
     .source = "pending: decision on the first member's TTFT (prefill-service closed batches, --batch-max)"},
    {.name = "fp16_dense_prefill", .value = {"", "on", ""}, .builtin = "off", .ready = none, .runnable = none,
     .kind = Kind::bit_changing, .apps = service, .baseline = false,
     .source = "pending: quality qualification (DS FP16 dense prefill bands); the service does not take --profile yet"},
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

bool takes_value(std::string_view flag, std::span<const Lever> table) {
    for (const auto& l : table)
        for (const auto& o : l.options)
            if (!o.flag.empty() && o.flag == flag) return o.value == "*";
    return false;
}

std::string switches(const Lever& l) {
    std::string s;
    for (const auto& o : l.options) {
        if (o.flag.empty()) continue;
        if (!s.empty()) s += ' ';
        s += o.flag;
    }
    if (!l.env.empty()) s += (s.empty() ? "" : " ") + std::string(l.env);
    return s;
}
}  // namespace

std::span<const Lever> levers() { return kTable; }

const char* status_name(Status status) {
    switch (status) {
    case Status::profile: return "profile";
    case Status::explicit_cli: return "explicit-cli";
    case Status::explicit_env: return "explicit-env";
    case Status::baseline_cli: return "baseline-cli";
    case Status::baseline_env: return "baseline-env";
    case Status::pending: return "pending";
    case Status::unqualified: return "unqualified";
    }
    return "unknown";
}
const char* kind_name(Kind kind) { return kind == Kind::exact ? "exact" : "bit-changing"; }

Name parse(std::string_view value) {
    if (value == "default") return Name::standard;
    if (value == "fast") return Name::fast;
    if (value == "quality") return Name::quality;
    throw std::invalid_argument("--profile must be default, fast or quality (got '" + std::string(value) + "')");
}
const char* name(Name profile) {
    switch (profile) {
    case Name::standard: return "default";
    case Name::fast: return "fast";
    case Name::quality: return "quality";
    }
    return "unknown";
}
const char* app_name(App app) { return app == App::replica ? "replica" : "prefill-service"; }

const Entry* Resolution::find(std::string_view lever) const {
    for (const auto& e : entries) if (e.lever->name == lever) return &e;
    return nullptr;
}
std::string Resolution::qualification() const {
    std::string rows;
    for (const auto& e : entries)
        if (e.status == Status::unqualified) rows += (rows.empty() ? "" : ",") + std::string(e.lever->name);
    return rows.empty() ? "qualified" : "unqualified: " + rows;
}
std::vector<std::string> Resolution::pending() const {
    std::vector<std::string> out;
    for (const auto& e : entries) if (e.status == Status::pending) out.emplace_back(e.lever->name);
    return out;
}
std::string Resolution::json(const std::function<std::string(std::string_view)>& effective) const {
    std::string out = "{\"profile\":";
    escape(out, profile::name(name));
    out += ",\"app\":";
    escape(out, app_name(app));
    out += ",\"qualification\":";
    escape(out, qualification());
    out += allow_pending ? ",\"allow_pending\":true" : ",\"allow_pending\":false";
    out += ",\"levers\":{";
    std::string pend, unq;
    bool first = true;
    for (const auto& e : entries) {
        if (!first) out += ',';
        first = false;
        escape(out, e.lever->name);
        out += ":{\"value\":";
        escape(out, e.value);
        out += ",\"status\":";
        escape(out, status_name(e.status));
        if (!e.wanted.empty()) {
            out += ",\"wanted\":";
            escape(out, e.wanted);
        }
        if (effective) {
            const auto v = effective(e.lever->name);
            if (!v.empty()) {
                out += ",\"effective\":";
                escape(out, v);
            }
        }
        out += ",\"switch\":";
        escape(out, switches(*e.lever));
        out += ",\"kind\":";
        escape(out, kind_name(e.lever->kind));
        out += ",\"source\":";
        escape(out, e.lever->source);
        out += '}';
        std::string& list = e.status == Status::pending ? pend : unq;
        if (e.status == Status::pending || e.status == Status::unqualified) {
            if (!list.empty()) list += ',';
            escape(list, e.lever->name);
        }
    }
    out += "},\"pending\":[" + pend + "],\"unqualified\":[" + unq + "]}";
    return out;
}

Selection select(const Inputs& inputs) {
    Selection s;
    std::optional<Name> cli;
    for (std::size_t i = 0; i < inputs.args.size(); ++i) {
        const auto& a = inputs.args[i];
        if (a == "--profile") {
            if (i + 1 >= inputs.args.size()) throw std::invalid_argument("missing value of --profile (default, fast or quality)");
            cli = parse(inputs.args[++i]);
        } else if (a == "--allow-pending") {
            s.allow_pending = true;
        }
    }
    if (cli) s.name = *cli;
    else if (const char* v = environment(inputs.env, "STRIX_DS41_PROFILE"); v && *v) s.name = parse(v);
    if (const char* v = environment(inputs.env, "STRIX_DS41_PROFILE_ALLOW_PENDING"); v && std::string_view(v) == "1")
        s.allow_pending = true;
    return s;
}

Resolution resolve(App app, const Inputs& inputs, std::span<const Lever> table) {
    const auto s = select(inputs);
    return resolve(s.name, s.allow_pending, app, inputs, table);
}

Resolution resolve(Name profile, bool allow_pending, App app, const Inputs& inputs, std::span<const Lever> table) {
    Resolution out;
    out.name = profile;
    out.app = app;
    out.allow_pending = allow_pending;
    const auto column = unsigned(profile);
    const char* baseline_env = environment(inputs.env, "STRIX_DS41_RTX_BASELINE");
    const bool env_baseline = baseline_env && std::string_view(baseline_env) == "1";
    for (const auto& lever : table) {
        if (!(lever.apps & unsigned(app))) continue;
        Entry entry{&lever, {}, Status::profile, {}};
        // the last of the lever's own options and --baseline decides (the replica parses them in order)
        bool cli = false, cli_baseline = false;
        for (std::size_t i = 0; i < inputs.args.size(); ++i) {
            const auto& a = inputs.args[i];
            if (a == "--profile") { ++i; continue; }  // its argument
            if (a == "--baseline") {
                if (lever.baseline) {
                    cli = cli_baseline = true;
                    entry.value = "off";
                }
                continue;
            }
            const Option* hit = nullptr;
            for (const auto& o : lever.options)
                if (!o.flag.empty() && a == o.flag) { hit = &o; break; }
            if (!hit) {
                if (takes_value(a, table)) ++i;  // another lever's option: skip its argument
                continue;
            }
            cli = true;
            cli_baseline = false;
            if (hit->value == "*") entry.value = ++i < inputs.args.size() ? inputs.args[i] : std::string();
            else entry.value = std::string(hit->value);
        }
        const char* own = lever.env.empty() ? nullptr : environment(inputs.env, std::string(lever.env).c_str());
        const auto& wanted = lever.value[column];
        if (cli) {
            entry.status = cli_baseline ? Status::baseline_cli : Status::explicit_cli;
        } else if (env_baseline && lever.baseline) {
            entry.value = "off";
            entry.status = Status::baseline_env;
        } else if (own && *own) {
            entry.value = std::string_view(own) == "0" ? "off" : "on";
            entry.status = Status::explicit_env;
        } else if (wanted.empty()) {
            continue;  // the profile leaves it alone
        } else if (!lever.ready[column]) {
            entry.wanted = std::string(wanted);
            if (allow_pending && lever.runnable[column]) {
                entry.value = std::string(wanted);  // benchmarks only: the receipt is stamped unqualified
                entry.status = Status::unqualified;
            } else {
                entry.value = std::string(lever.builtin);  // refused: the builtin form runs
                entry.status = Status::pending;
            }
        } else {
            entry.value = std::string(wanted);
            entry.status = Status::profile;
        }
        out.entries.push_back(std::move(entry));
    }
    return out;
}

}  // namespace strix::ds41::rtx::profile
