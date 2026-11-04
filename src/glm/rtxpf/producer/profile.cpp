#include "profile.hpp"

#include <cstdlib>
#include <optional>
#include <stdexcept>

namespace strix::glm::rtxpf::profile {
namespace {
constexpr unsigned service = unsigned(App::decode_service);
constexpr std::array<bool, 3> all{true, true, true}, none{false, false, false};
// Profile columns: default, fast, quality. One row per lever; a qualification (PRO window, qualification band, maintainer decision)
// flips its own row's `ready` (and `runnable` once the implementation is merged into this build).
// Every exact row below is ON in all three profiles: they are the service's defaults today, so `default` (and no
// --profile) is today's configuration byte for byte, and `quality` = `default` (no higher-precision expert pack on the RTX).
constexpr Lever kTable[] = {
    {.name = "expert_tier", .options = {{{"--expert-tier", "*"}}}, .value = {"tierstream", "tierstream", "tierstream"},
     .builtin = "tierstream", .ready = all, .runnable = all, .kind = Kind::exact, .apps = service,
     .source = "the tierstream device-gated expert cache (prefill through the cache, one copy of each expert)"},
    {.name = "placement", .options = {{{"--expert-residency", "*"}, {"--expert-routes", "*"}}}, .value = {"on", "on", "on"},
     .builtin = "on", .ready = all, .runnable = all, .kind = Kind::exact, .apps = service,
     .source = "DS-style placement: residency seed (<unit-file>.residency), routes census, expert-major round-robin "
               "(dg_seed + tier array; placement only)"},
    {.name = "eviction_policy", .options = {{{"--tier-policy", "*"}}}, .value = {"auto", "auto", "auto"}, .builtin = "auto",
     .ready = all, .runnable = all, .kind = Kind::exact, .apps = service,
     .source = "SCORED_AUTO boundary victims as DS (victims only; outputs unchanged)"},
    {.name = "spare_share", .options = {{{"--tier-spare-percent", "*"}, {"--tier-spare-slots", "*"}}},
     .value = {"5", "5", "5"}, .builtin = "5", .ready = all, .runnable = all, .kind = Kind::exact, .apps = service,
     .source = "arena spares 5% of the arena (retained misses; placement only)"},
    {.name = "row_tile", .options = {{{"--row-tile", "*"}}}, .value = {"auto", "auto", "auto"}, .builtin = "auto",
     .ready = all, .runnable = all, .kind = Kind::exact, .apps = service,
     .source = "largest tile <= 8192, growth over 2048 <= 5% of its expert arena; extra file units require "
               "2048 file share <= 15% and increase <= 5% of routed units; explicit --row-tile bypasses these bounds"},
    {.name = "prefill_tile_crossover", .options = {{{"--prefill-tile-crossover", "*"}}}, .value = {"0", "0", "0"},
     .builtin = "0", .ready = all, .runnable = all, .kind = Kind::exact, .apps = service,
     .source = "per-request prefill tile inside the allocation: prefill rows below the crossover run at min(4096, allocated), "
               "at and above it at the allocated tile (exact: the tile only changes row chunking; the tile sweeps showed equal "
               "prefill state digests); 0 = always the allocated tile: the one-load tile sweep (allocation 8192) had 8192 fastest at 8K/16K/32K (29.2/36.7/53.6 s vs 30.2/38.6/56.6 at 4096); --prefill-tile-crossover N re-enables the rule"},
    {.name = "cuda_graphs", .options = {{{"--no-graphs", "off"}}}, .value = {"on", "on", "on"}, .builtin = "on", .ready = all,
     .runnable = all, .kind = Kind::exact, .apps = service, .source = "prefill CUDA graph templates (full tile + exact tail)"},
    {.name = "restore_decode_residency", .env = "STRIX_GLM_RTX_RESTORE_RESIDENCY", .options = {{{"--restore-decode-residency", "*"}}},
     .value = {"on", "on", "on"}, .builtin = "on", .ready = all, .runnable = all, .kind = Kind::exact, .apps = service,
     .source = "placement only: after every prefill through the cache the open decode placement is made resident again (held "
               "make_resident sets, then released); without it the last prefilled layers own the arena and decode misses everywhere"},
    {.name = "flush_ready", .env = "STRIX_GLM_RTX_FLUSH_READY", .options = {{{"--tier-flush-ready", "*"}}},
     .value = {"on", "on", "on"}, .builtin = "on", .ready = all, .runnable = all, .kind = Kind::exact, .apps = service,
     .source = "tierstream dg_flush_ready as the DS tier site: the demand stream is flushed after each ready signal so WDDM "
               "submits the ready write at once (submission only; outputs unchanged)"},
    // pending in every profile until a PRO window proves it exact
    {.name = "predicted_prefetch", .options = {{{"--tier-predict", "*"}}}, .value = {"1", "1", "1"}, .builtin = "0",
     .ready = none, .runnable = all, .kind = Kind::exact, .apps = service,
     .source = "pending: exact (tokens and every logits digest equal with --tier-predict 0/1, plain and self-speculation; "
               "the predicted branch reads its own input copy and only prefetches, up to 4 rows); decode 1K full card "
               "3.5 -> 4.8 tok/s warm, no change at 24 GiB VRAM + 32 GiB RAM (disk bound); run-to-run spread not yet bounded"},
    // exact decode levers, pending in every profile until a window shows equal digests
    {.name = "skip_empty_expert_groups", .env = "STRIX_GLM_RTX_SKIP_EMPTY_GROUPS", .options = {{{"--skip-empty-expert-groups", "*"}}},
     .value = {"1", "1", "1"}, .builtin = "0", .ready = none, .runnable = all, .kind = Kind::exact, .apps = service,
     .source = "pending: the bitwise digest compare in a PRO window (exact by construction: the routed experts are compacted on "
               "the device and only the 16-expert groups that can hold routed rows run; per-expert arithmetic unchanged)"},
    {.name = "persistent_dense_fp16", .env = "STRIX_GLM_RTX_PERSISTENT_DENSE_FP16", .options = {{{"--persistent-dense-fp16", "*"}}},
     .value = {"1", "1", "1"}, .builtin = "0", .ready = none, .runnable = all, .kind = Kind::exact, .apps = service,
     .source = "pending: the bitwise digest compare in a PRO window (exact by construction: shared experts and dense FFNs keep "
               "the FP16 operands the decode window writes, decoded once; ~2.8 GiB in the whole-device plan, off when it does not fit)"},
    // exact prefill lever: equal digests and a faster TTFT in PRO windows, on in every profile
    {.name = "prefill_overlap", .options = {{{"--prefill-overlap", "*"}}}, .value = {"1", "1", "1"}, .builtin = "1",
     .ready = all, .runnable = all, .kind = Kind::exact, .apps = service,
     .source = "the next routed layer's make_resident runs while this layer's kernels are in flight (held layers without a "
               "token, released after their completion event); when only the boundary spares block two held layers the "
               "prefill lowers the spare target and restores it before decode; off when the arena cannot keep two layers. "
               "Prefill state digests, tokens and logits equal with it off. TTFT full card 8K 44.7 -> 25.1 s, 32K 71.1 -> "
               "46.7 s; 24 GiB VRAM + 32 GiB RAM 1K 63.4 -> 60.3 s, 8K 73.7 -> 62.4 s. OFF: --prefill-overlap 0"},
    {.name = "decode_row_kernels", .env = "STRIX_GLM_RTX_ROW_KERNELS", .options = {{{"--decode-row-kernels", "*"}}},
     .value = {"1", "1", "1"}, .builtin = "1", .ready = all, .runnable = all, .kind = Kind::exact, .apps = service,
     .source = "few-row decode kernels, each output's arithmetic unchanged: MoE / KDA / DSA / MTP trellis GEMMs as "
               "register-decoded m16n8k16 chains in the CUTLASS K order (no per-token FP16 decode), hyper-connection and "
               "router chains from shared memory, the router's top 8 by rank. Full card, 1K/8K: tokens, every logits "
               "digest and the final state equal with it off; qualify PASS; decode 2.4 -> 6.0-6.2 tok/s warm. OFF: 0; "
               "a subset such as moe+hc+router+kda+dsa (round one exactly) selects kernels; 1 includes r2pre (A loads before "
               "the weight decode: same bits, +8 % warm decode at 1K in an ABBA pair); 2 (or +r2, or one of "
               "+r2dense/r2idx/r2hc/r2par) adds the rest of round two, opt-in: bitwise equal "
               "but not faster in one-load A/Bs (dense strips -8 %)"},
    {.name = "mtp_self_spec", .env = "STRIX_GLM_RTX_MTP_SPEC", .options = {{{"--mtp-self-spec", "*"}}},
     .value = {"1", "1", "1"}, .builtin = "0", .ready = none, .runnable = all, .kind = Kind::exact, .apps = service,
     .source = "pending (not faster at every length): greedy verify of [pending, MTP draft] with the plain step's per-row "
               "kernels; rejected rows rebuilt from the KDA block checkpoint / rewritten in DSA; up to 32 exported ids "
               "per layer. Tokens, every logits digest and the final state equal to plain decode (full card and 24 GiB "
               "VRAM + 32 GiB RAM, 1K/8K prompts). Full card, 24-step benches: 8K +27 % warm / +15 % cold, a cold "
               "48-token generate +16 %, 1K warm about equal and cold -6 %. Adaptive: measured ms per emitted token picks "
               "verify cycles or plain steps (runs of 32, probes of 4); always = every step. Tierstream only and not "
               "with --routes-out (otherwise off)"},
    {.name = "lookup_drafts", .env = "STRIX_GLM_RTX_SPEC_LOOKUP", .options = {{{"--spec-lookup", "*"}}},
     .value = {"1", "1", "1"}, .builtin = "0", .ready = none, .runnable = all, .kind = Kind::exact, .apps = service,
     .source = "pending: exact (tokens and every logits digest equal to plain decode in PRO windows; prompt-lookup "
               "drafts, up to 3, only propose tokens for the mtp_self_spec greedy verify) but not faster than MTP drafts "
               "alone on the measured prompts"},
    // fast-only, pending
    {.name = "prefill_tile_large", .value = {"", "8192", ""}, .builtin = "off", .ready = none,
     .runnable = {false, true, false}, .kind = Kind::bit_changing, .apps = service,
     .source = "pending: the --tile-sweep window; BITWISE-EQUAL prefill state across tiles makes a larger tile an exact "
               "default candidate, DIFFERENT makes this a fast row that needs qualification bands (--row-tile 8192 when run)"},
    {.name = "dsa_index_mma", .value = {"", "mma", ""}, .builtin = "fp32", .ready = none, .runnable = none,
     .kind = Kind::bit_changing, .apps = service,
     .source = "pending: DSA index MMA variants (dsa.cuh: mma, mma_proj_fp32_score, fp32_proj_mma_score, tf32x3) exist for "
               "the prefill producer only; not wired into decode sessions; needs qualification bands"},
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
const char* app_name(App) { return "decode-service"; }

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
std::vector<std::string> Resolution::notices() const {
    std::vector<std::string> out;
    const std::string p = profile::name(name);
    if (name == Name::quality) out.push_back("profile quality: no higher-precision expert pack for GLM on the RTX; running default");
    for (const auto& e : entries)
        if (e.status == Status::pending)
            out.push_back("profile " + p + ": " + std::string(e.lever->name) + " pending, not used (" + std::string(e.lever->source) + ")");
    if (qualification() != "qualified") out.push_back("profile " + p + ": " + qualification() + " (--allow-pending: benchmarks only)");
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
    if (name == Name::quality) out += ",\"note\":\"quality = default: no higher-precision expert pack for GLM on the RTX\"";
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
    else if (const char* v = environment(inputs.env, "STRIX_GLM_RTX_PROFILE"); v && *v) s.name = parse(v);
    if (const char* v = environment(inputs.env, "STRIX_GLM_RTX_PROFILE_ALLOW_PENDING"); v && std::string_view(v) == "1")
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
    for (const auto& lever : table) {
        if (!(lever.apps & unsigned(app))) continue;
        Entry entry{&lever, {}, Status::profile, {}};
        // the last of the lever's own options decides (the service parses them in order)
        bool cli = false;
        for (std::size_t i = 0; i < inputs.args.size(); ++i) {
            const auto& a = inputs.args[i];
            if (a == "--profile") { ++i; continue; }  // its argument
            const Option* hit = nullptr;
            for (const auto& o : lever.options)
                if (!o.flag.empty() && a == o.flag) { hit = &o; break; }
            if (!hit) {
                if (takes_value(a, table)) ++i;  // another lever's option: skip its argument
                continue;
            }
            cli = true;
            if (hit->value == "*") entry.value = ++i < inputs.args.size() ? inputs.args[i] : std::string();
            else entry.value = std::string(hit->value);
        }
        const char* own = lever.env.empty() ? nullptr : environment(inputs.env, std::string(lever.env).c_str());
        const auto& wanted = lever.value[column];
        if (cli) {
            entry.status = Status::explicit_cli;
        } else if (own && *own) {
            const std::string_view v(own);
            entry.value = v == "0" || v == "off" ? "off" : std::string(v);
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

}  // namespace strix::glm::rtxpf::profile
