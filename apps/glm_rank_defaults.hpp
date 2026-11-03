// strix-glm5-rank defaults: optimizations are ON by default in code and switched OFF per item, or all at
// once with the baseline master switch). Host-only, no HIP: the rank app resolves its route here, --print-effective-config
// prints the resolution without touching a GPU or a peer, and tests/glm_rank_defaults_cpu_tests.cpp diffs the plain
// default invocation against the measured decode, batch, and host-path prefill configuration. Eligible normal decode
// enables self-speculation when the model has a complete MTP inventory; explicit-off retains the plain route.
//
// Rules: explicit flags and environment values have the per-item precedence below;
// --baseline / STRIX_GLM_BASELINE=1 restores the old path of every item (except where noted) unless that item is set
// explicitly; a default whose precondition fails falls back with one notice line (prefill auto at world 1, see the app).
#pragma once

#include "glm_model_facts.hpp"

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace strix::glm::rank_defaults {

using EnvGet = std::function<const char*(const char*)>;

inline EnvGet process_env() {
    return [](const char* k) -> const char* { return std::getenv(k); };
}

// One environment-switched item. on = what the default resolves to, off = the old path. app_sets: the app applies `on`
// (setenv if unset) because the code's own default is the old path; otherwise `on` is the engine's built-in default and is
// only reported. in_baseline: the baseline switch sets `off` (if unset).
struct EnvItem {
    const char* item;
    const char* env;
    const char* on;
    const char* off;
    bool app_sets;
    bool in_baseline;
    const char* receipt;
};

// MoE: STRIX_GLM_MOE=ref (DefaultMoe) stays a per-item switch but is not in the baseline: its prepared experts do not
// fit the TP2 rank's GTT on the real model (measured OOM: 36,288 x 4 MiB granules), so it is no runnable old path.
inline const std::vector<EnvItem>& env_items() {
    static const std::vector<EnvItem> items = {
        {"hp-crc-parallel", "STRIX_DIST_CRC_PARALLEL", "1", "0", true, true,
         "Measurement: hp host path, exact TTFT 1K -8.5 %, 8K -3.1 %, logits fnv matches reference 16/16"},
        {"hp-crc-overlap", "STRIX_DIST_HOST_CRC_OVERLAP", "1", "0", true, true, "Measurement: hp host path, exact TTFT 1K -8.5 %, 8K -3.1 %, logits fnv matches reference 16/16"},
        {"hp-chunk-bytes", "STRIX_DIST_HOST_CHUNK_BYTES", "1048576", "0", true, true, "Measurement: hp host path, exact TTFT 1K -8.5 %, 8K -3.1 %, logits fnv matches reference 16/16"},
        {"pf-pipeline-2", "STRIX_GLM_PF_PIPELINE", "2", "1", true, true,
         "Measurement: N=2 whole-layer prefill pipeline + async start, prefill gate bitwise, 8K -4.0 % (needs dist-async-start)"},
        {"dist-async-start", "STRIX_DIST_ASYNC_START", "1", "0", true, true,
         "Measurement: async exchange start (full-host chunked path; only the pipeline uses it)"},
        {"hsa-no-fragment-allocator", "HSA_DISABLE_FRAGMENT_ALLOCATOR", "1", "0", true, true,
         "Measurement: production decode and prefill configuration; allocator granules fit GTT"},
        {"r1b-dense-auto", "STRIX_GLM_R1B_DENSE", "7", "2", false, true,
         "Measurement: R1b-fast kernels -7.8 % R1 / -5.4 % R8, same tokens"},
        {"r1b-hc-v4", "STRIX_GLM_R1B_HC", "4", "1", false, true, "Measurement: with r1b-dense-auto, R1b-fast kernels -7.8 % R1 / -5.4 % R8, same tokens"},
        {"hc-pre-v7", "STRIX_GLM_HC_PRE_V", "7", "4", false, true,
         "Measurement: hc_pre 77.4-77.9 vs 102.1-102.8 us, gates 0-7 bitwise; tokens match reference"},
        {"post-exchange-fast", "STRIX_GLM_POST", "fast", "ref", false, true,
         "Measurement: fast post-exchange consumers match STRIX_GLM_POST=ref and reference (real model, 6 layers)"},
        {"attn-fast", "STRIX_GLM_ATTN", "fast", "ref", false, true,
         "Measurement: FastAttn matches STRIX_GLM_ATTN=ref (G3), Tier A 12-40x"},
        {"attn-tierb-fused", "STRIX_GLM_TIERB", "fused", "lib", false, true,
         "Measurement: fused Tier B -2.46 ms/token vs strix_gemv, tokens match reference"},
        {"spec-xfer", "STRIX_GLM_SPEC_XCH", "xfer", "comm", false, true,
         "Measurement: verify + MTP drafter on XferExchange, token vectors match --mtp 0, SP1a PASS"},
        {"hc-norm-fused", "STRIX_GLM_HC_NORM", "1", "0", true, true,
         "Measurement: R1b decode hc_fin + RMSNorm fused, tokens and R8 match reference; pair ABA x2 -2.3 % C1 MTP, -3.1 % R8"},
        {"hc-site-launch", "STRIX_GLM_HC_SITE", "1", "0", false, true,
         "Measurement: one launch per one-row decode mHC site, same bits; pair ABBA C1 -1.45 %, R8 unchanged (G-02)"},
        {"spec-kda-tree-2", "STRIX_GLM_SPEC_KDA_TREE", "2", "1", true, true,
         "Measurement: two-barrier KDA verify tree, bitwise == 9-barrier tree, token vectors match --mtp 0, +2.8 %"},
        {"xfer-boundary-fused", "STRIX_GLM_BOUNDARY", "1", "0", false, true,
         "Measurement: Xfer reduce +1.41 -> fused boundary ~+1.10 ms/token over local, bitwise"},
        {"moe-fast", "STRIX_GLM_MOE", "fast", "ref", false, false,
         "Measurement: FastMoe matches DefaultMoe bitwise (real TP2 slices); raw-only experts (fits GTT)"},
        // =7 restores the previous large-call core (X2), =3 v3 for every call; the baseline restores v1.
        {"pf-exact-moe-auto", "STRIX_GLM_PF_EXACTMOE_V", "auto", "1", true, true,
         "Measurement: exact MoE auto (v14 through the X3 core with row-bucketed tiles and folded remainders for calls >= 128 rows, v3 with row buckets below), outputs bitwise; 8K TTFT -19.5 % and 1K -15 % vs the X2 core; MoE call -3..-6 % vs v11"},
        {"pf-exact-moe-plan", "STRIX_GLM_PF_EXACTMOE_PLAN", "device", "host", false, true,
         "Measurement: exact MoE device plan (pairs/tiles/groups built on the GPU, byte-equal to the host plan; no route download or host sync per MoE call)"},
        {"pf-exact-x", "STRIX_GLM_PF_EXACT_X", "auto", "0", true, true,
         "Measurement: exact X3 prefill linear core (scalar activations, eight rows per wave) for calls >= 2048 rows (TM16 below), bitwise; 8K TTFT -14.3 % vs X2"},
        {"pf-obar-v7", "STRIX_GLM_PF_OBAR", "7", "2", false, true,
         "Measurement: MLA o-bar v7 bitwise == v5 (96/0), 9.0-10.2 vs 16.6-17.3 us/row per DSA layer at t0 4096..28672 (=5 restores v5)"},
        {"pf-scores-v6", "STRIX_GLM_PF_SCORES", "6", "2", false, true,
         "Measurement: MLA scores v6 (hand-paired VOPD) bitwise == v5 (119/0), 3.40 / 8.27 / 10.38 vs 4.57 / 9.27 / 11.39 us/row per DSA layer at t0 0 / 4096 / 28672 (=5 restores v5)"},
        {"pf-mla-gemm-v2", "STRIX_GLM_PF_MLA_GEMM", "2", "1", false, true,
         "Measurement: MLA absorb / out GEMM v2 bitwise == v1 (106/0), absorb 6.1 -> 4.6 and out 4.1 -> 3.3 us/row per DSA layer"},
        {"pf-kda-scan-v5", "STRIX_GLM_PF_KDA_SCAN", "5", "1", false, true,
         "Measurement: KDA scan v5 (quad-DPP tree, batch fill off the serial path) bitwise == v3 / v1 and decode, 1.84-1.86 -> 1.18-1.20 us/row per KDA layer at t0 0..28672 (v3 1.23-1.25; =3 restores v3)"},
        {"pf-indexer-v3", "STRIX_GLM_PF_IDX", "3", "1", false, true,
         "Measurement: indexer scores v3 same keys as v2, 5.3 vs 9.6 us/row per DSA layer at t0 28672 (=2 restores v2)"},
        {"pf-select-v2", "STRIX_GLM_PF_SELECT", "2", "1", false, true,
         "Measurement: indexer select v2 same token lists as v1, 0.38 vs 0.99 us/row at t0 28672, 0.08 vs 0.50 at t0 4096"},
        {"max-rows-auto", "STRIX_GLM_MAX_ROWS_AUTO", "1", "0", false, true,
         "Measurement: exact 8192 vs 4096 rows, 8K TTFT -4 %, 32K 116.3 -> 109.9 s (-5.5 %); +7.3 GiB per rank"},
        {"pf-staged", "STRIX_GLM_PF_STAGED", "1", "0", false, true,
         "Measurement: pipelined prefill exchanges through the coherent staging views; pair ABBA -0.8 % 1K / -0.9 % 8K TTFT, logits bitwise"},
        {"pf-hc-fuse", "STRIX_GLM_PF_HC_FUSE", "1", "0", false, true,
         "Measurement: mHC expand + next-site norm fused (F not materialised), logits bitwise; pair ABBA x2 -1.1 % 1K / -1.3 % 8K TTFT"},
    };
    return items;
}

// Options with no value (everything else the rank app accepts takes exactly one value).
inline bool is_value_free(const std::string& a) {
    return a == "--g3-sweep" || a == "--pf-install-before-load" || a == "--pf-swap-moe" || a == "--mem-plan" ||
           a == "--profile" || a == "--tf-check" || a == "--baseline" || a == "--print-effective-config";
}

// Gate / measurement modes that install their own prefill path (prefill auto -> ref under them).
inline bool is_reference_gate(const std::string& a) {
    return a == "--ab" || a == "--aba" || a == "--g3-sweep" || a == "--aba-attn3" || a == "--aba-xch" || a == "--g4" ||
           a == "--tf" || a == "--spec-oracle" || a == "--spec-run" || a == "--state-in" || a == "--state-out" ||
           a == "--mtp-prefill-gate" || a == "--pf-gate" || a == "--pf-ttft" || a == "--pf-null-study";
}

// The part of the command line the defaults depend on (args = argv[2..], i.e. without program and MODEL_DIR).
struct Request {
    std::uint32_t world = 2;
    std::optional<std::string> identity, prefill, decode_xch;
    std::optional<bool> mtp, mtp_spec;
    std::optional<unsigned> max_rows;
    bool decode = false;
    unsigned mtp_k = 3;
    bool adaptive = true;
    bool spec = false, g3 = false, decode_or_batch = false, mem_plan = false, reference_gate = false;
    unsigned idcheck = 0;
    std::string xch = "comm";
    bool baseline_flag = false, print = false;
};

inline Request scan(const std::vector<std::string>& args) {
    Request r;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (is_reference_gate(a)) r.reference_gate = true;
        if (is_value_free(a)) {
            if (a == "--mem-plan") r.mem_plan = true;
            else if (a == "--baseline") r.baseline_flag = true;
            else if (a == "--print-effective-config") r.print = true;
            continue;
        }
        if (i + 1 >= args.size()) break;  // the app's own parser reports the missing value
        const std::string& v = args[++i];
        if (a == "--world") r.world = static_cast<std::uint32_t>(std::stoul(v));
        else if (a == "--identity") r.identity = v;
        else if (a == "--prefill") r.prefill = v;
        else if (a == "--mtp") r.mtp = std::stoul(v) != 0;
        else if (a == "--max-rows") r.max_rows = static_cast<unsigned>(std::stoul(v));
        else if (a == "--mtp-spec") {
            if (v != "0" && v != "1") throw std::invalid_argument("--mtp-spec 0|1");
            r.mtp_spec = v == "1";
        }
        else if (a == "--mtp-k") {
            const auto k = std::stoul(v);
            if (k > 7) throw std::invalid_argument("--mtp-k 0..7");
            r.mtp_k = static_cast<unsigned>(k);
        }
        else if (a == "--mtp-adaptive") {
            if (v != "0" && v != "1") throw std::invalid_argument("--mtp-adaptive 0|1");
            r.adaptive = v == "1";
        }
        else if (a == "--spec") r.spec = std::stoul(v) != 0;
        else if (a == "--decode-xch") r.decode_xch = v;
        else if (a == "--xch") r.xch = v;
        else if (a == "--idcheck") r.idcheck = static_cast<unsigned>(std::stoul(v));
        else if (a == "--g3") r.g3 = !v.empty();
        else if (a == "--decode" || a == "--batch") {
            const bool on = std::stoul(v) != 0;
            r.decode_or_batch = r.decode_or_batch || on;
            if (a == "--decode") r.decode = on;
        }
    }
    return r;
}

inline bool env_is(const EnvGet& env, const char* k, const char* v) {
    const char* e = env(k);
    return e != nullptr && std::string(e) == v;
}

inline bool baseline_requested(const Request& r, const EnvGet& env) {
    return r.baseline_flag || env_is(env, "STRIX_GLM_BASELINE", "1");
}

// Exact chunked prefill is also available with MTP fill for normal self-speculation. Gate and legacy MTP-only modes
// keep reference prefill; r1b-all, G3/smoke/mem-plan and --idcheck retain their existing exclusions.
inline std::string resolve_prefill_auto(bool mtp, bool spec, bool g3_only, bool smoke_head, bool mem_plan, bool r1b_all,
                                        unsigned idcheck, bool reference_gate, bool mtp_spec = false) {
    return (mtp_spec || (!mtp && !spec)) && !g3_only && !smoke_head && !mem_plan && !r1b_all && idcheck == 0 && !reference_gate ? "gemv" : "ref";
}

struct Value {
    std::string value;
    std::string source;  // default | explicit | baseline | auto-8192 | auto-4096-memory
};

struct Effective {
    bool baseline = false;
    Value identity, prefill_requested, mtp, mtp_spec, spec, decode_xch, max_rows;
    ModelFacts head;
    bool mtp_spec_candidate = false;
    unsigned kmax = 3;
    bool adaptive = true;
    std::string prefill;  // resolved: gemv | ref | wmma
    std::string xch;
    std::map<std::string, Value> env;  // env name -> effective value
};

inline Effective resolve(const Request& r, const EnvGet& env, const ModelFacts& facts = {}) {
    Effective e;
    e.baseline = baseline_requested(r, env);
    const char* src_b = e.baseline ? "baseline" : "default";
    // identity: STRIX_GLM_IDENTITY overrides --identity (engine rule)
    if (const char* v = env("STRIX_GLM_IDENTITY")) e.identity = {v, "explicit"};
    else if (r.identity) e.identity = {*r.identity, "explicit"};
    else e.identity = {e.baseline ? "r1" : "r1b", src_b};
    e.head = facts;
    e.kmax = r.mtp_k;
    e.adaptive = r.adaptive;
    std::optional<bool> requested = r.mtp_spec;
    if (!requested) if (const char* v = env("STRIX_GLM_MTP_SPEC")) {
        if (std::string(v) != "0" && std::string(v) != "1") throw std::invalid_argument("STRIX_GLM_MTP_SPEC=0|1");
        requested = std::string(v) == "1";
    }
    const bool normal_decode = r.decode && !r.reference_gate && !r.mem_plan && r.idcheck == 0;
    const std::string requested_prefill = r.prefill.value_or(env("STRIX_GLM_PREFILL") ? env("STRIX_GLM_PREFILL") : "auto");
    const bool eligible = !env_is(env, "STRIX_GLM_SMOKE_HEAD", "1") && e.identity.value != "r1b-all" &&
                          r.xch == "comm" && requested_prefill != "wmma" && (r.world == 1 || r.world == 2);
    e.mtp_spec_candidate = normal_decode && requested.value_or(!e.baseline) && r.mtp.value_or(true) && r.xch == "comm";
    const bool enabled = e.mtp_spec_candidate && eligible && facts.mtp_complete;
    e.mtp_spec = {enabled ? "1" : "0", requested ? "explicit" : src_b};
    if (r.mtp) e.mtp = {*r.mtp ? "1" : "0", "explicit"};
    else e.mtp = {enabled ? "1" : "0", src_b};
    if (e.mtp_spec_candidate && !facts.mtp_complete) e.mtp = {"0", "fallback"};
    e.spec = {enabled || r.spec ? "1" : "0", r.spec ? "explicit" : src_b};
    // prefill: --prefill wins over STRIX_GLM_PREFILL
    if (r.prefill) e.prefill_requested = {*r.prefill, "explicit"};
    else if (const char* v = env("STRIX_GLM_PREFILL")) e.prefill_requested = {v, "explicit"};
    else e.prefill_requested = {e.baseline ? "ref" : "auto", src_b};
    // decode Xfer exchange: STRIX_GLM_DECODE_XCH overrides --decode-xch (engine rule)
    if (const char* v = env("STRIX_GLM_DECODE_XCH")) e.decode_xch = {v, "explicit"};
    else if (r.decode_xch) e.decode_xch = {*r.decode_xch, "explicit"};
    else e.decode_xch = {e.baseline ? "comm" : "xfer", src_b};
    e.xch = r.xch;
    const bool g3_only = r.g3 && !r.decode_or_batch && r.idcheck == 0;
    const bool smoke_head = env_is(env, "STRIX_GLM_SMOKE_HEAD", "1");
    e.prefill = e.prefill_requested.value != "auto"
                    ? e.prefill_requested.value
                    : resolve_prefill_auto(e.mtp.value == "1", r.spec, g3_only, smoke_head, r.mem_plan,
                                           e.identity.value == "r1b-all", r.idcheck, r.reference_gate, enabled);
    // Reference prefill does not use chunk rows. Otherwise startup admits the host-independent
    // 8192 candidate only when both ranks have enough usable memory.
    if (r.max_rows) e.max_rows = {std::to_string(*r.max_rows), "explicit"};
    else if (e.baseline) e.max_rows = {"1024", "baseline"};
    else if (env_is(env, "STRIX_GLM_MAX_ROWS_AUTO", "0")) e.max_rows = {"4096", "explicit"};
    else if (r.xch == "xfer") e.max_rows = {"4096", "auto-4096-xfer"};
    else if (e.prefill == "ref") e.max_rows = {"4096", "auto-4096-ref"};
    else e.max_rows = {"8192", "auto-8192"};
    for (const auto& it : env_items()) {
        if (const char* v = env(it.env)) e.env[it.env] = {v, "explicit"};
        else if (e.baseline && it.in_baseline) e.env[it.env] = {it.off, "baseline"};
        else e.env[it.env] = {it.on, "default"};
    }
    return e;
}

inline constexpr unsigned kAutoMaxRows = 8192, kFallbackMaxRows = 4096;
inline constexpr std::uint64_t kMaxRowsSafetyBytes = std::uint64_t{8} << 30;
inline bool max_rows_auto(const Effective& e) {
    return e.max_rows.source == "auto-8192" || e.max_rows.source == "auto-4096-memory";
}

// The predicate dry-runs the load and prefill admissions, including their current margins.
template<class Fits>
inline void select_max_rows(Effective& e, Fits&& fits) {
    if (!max_rows_auto(e) || e.max_rows.value == "4096") return;
    e.max_rows = fits() ? Value{"8192", "auto-8192"} : Value{"4096", "auto-4096-memory"};
}

inline void agree_max_rows(Effective& e, unsigned rank_min) {
    if (max_rows_auto(e) && rank_min < kAutoMaxRows) e.max_rows = {"4096", "auto-4096-memory"};
}

// Process environment for the resolved items (setenv if unset: an explicit value always wins). Default: the app-set
// items' `on`; baseline: every baseline item's `off` (engine items included, they read the environment at load).
inline void apply_env(const Effective& e, const std::function<void(const char*, const char*)>& setenv_if_unset) {
    for (const auto& it : env_items()) {
        const Value& v = e.env.at(it.env);
        if (v.source == "explicit") continue;
        if (v.source == "baseline" || it.app_sets) setenv_if_unset(it.env, v.value.c_str());
    }
}

inline std::string print(const Effective& e) {
    std::string s;
    auto line = [&](const std::string& k, const Value& v) { s += "effective " + k + "=" + v.value + " (" + v.source + ")\n"; };
    s += std::string("effective baseline=") + (e.baseline ? "1" : "0") + "\n";
    line("identity", e.identity);
    line("mtp", e.mtp);
    line("mtp-spec", e.mtp_spec);
    line("spec", e.spec);
    s += "effective mtp-head=" + e.head.reason + "\n";
    s += "effective mtp-kmax=" + std::to_string(e.kmax) + " adaptive=" + (e.adaptive ? "1" : "0") + "\n";
    s += "effective mtp-bootstrap=" + std::string(e.mtp_spec.value == "1" ? (e.prefill == "gemv" ? "chunked" : "ref") : "plain") + "\n";
    s += "effective prefill=" + e.prefill + " (requested " + e.prefill_requested.value + ", " + e.prefill_requested.source + ")\n";
    line("decode-xch", e.decode_xch);
    line("max-rows", e.max_rows);
    if (max_rows_auto(e)) s += "effective max-rows-admission=startup-admission-and-rank-min (8192 candidate, 4096 fallback)\n";
    s += "effective xch=" + e.xch + "\n";
    for (const auto& it : env_items()) {
        const Value& v = e.env.at(it.env);
        s += std::string("effective env ") + it.env + "=" + v.value + " (" + v.source + ", item " + it.item + ")\n";
    }
    return s;
}

}  // namespace strix::glm::rank_defaults
