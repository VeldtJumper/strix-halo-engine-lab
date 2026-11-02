#pragma once

#include "strix/gsq_a8_groups.hpp"
#include "strix/model_manifest.hpp"

#include <array>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <string_view>

namespace strix::qpf2k {

enum class Item : unsigned {
    sched, dense, gdn, gdn_gates_batch, gdn_core16, qsa, qsa_mask,
    hc, hc_mask, hc_output3, hc_output3_band, moe_da16, moe_route, ple_reclaim, dense_rows, pipe, attend_batch, score_park,
    norm_group, index_rows, cohort_lean, rows_lean, dispatch_fuse, score_rows, ple_batch, shared_pre, moe_mc, moe_hdr,
    moe_xor64, hc_lowband,
};
struct Default {
    Item item;
    const char* name;
    const char* value;
};
inline constexpr std::array defaults{
    Default{Item::sched, "STRIX_QPF2K_SCHED", "14"},
    Default{Item::dense, "STRIX_QPF2K_DENSE", "1"},
    Default{Item::gdn, "STRIX_QPF2K_GDN", "1"},
    Default{Item::gdn_gates_batch, "STRIX_QPF2K_GDN_GATES_BATCH", "1"},
    Default{Item::gdn_core16, "STRIX_QPF2K_GDN_CORE16", "1"},
    Default{Item::qsa, "STRIX_QPF2K_QSA", "1"},
    Default{Item::qsa_mask, "STRIX_QPF2K_QSA_MASK", "0x2257"},
    Default{Item::hc, "STRIX_QPF2K_HC", "1"},
    Default{Item::hc_mask, "STRIX_QPF2K_HC_MASK", "14"},
    Default{Item::hc_output3, "STRIX_QPF2K_HC_OUTPUT3", "1"},
    Default{Item::hc_output3_band, "STRIX_QPF2K_HC_OUTPUT3_BAND", "16"},
    Default{Item::moe_da16, "STRIX_QPF2K_MOE_DA16", "1"},
    Default{Item::moe_route, "STRIX_QPF2K_MOE_ROUTE", "14"},
    Default{Item::ple_reclaim, "STRIX_PLE_RECLAIMABLE", "auto"},
    Default{Item::dense_rows, "STRIX_INTPREFILL_DENSE_ROWS", "512"},
    Default{Item::pipe, "STRIX_PREFILL_PIPE", "255"},
    Default{Item::attend_batch, "STRIX_QSA_ATTEND_BATCH", "1"},
    Default{Item::score_park, "STRIX_QSA_SCORE_PARK", "1"},
    Default{Item::norm_group, "STRIX_QSA_NORM_GROUP", "1"},
    Default{Item::index_rows, "STRIX_QSA_INDEX_ROWS", "1"},
    Default{Item::cohort_lean, "STRIX_LM_HEAD_COHORT_LEAN", "1"},
    Default{Item::rows_lean, "STRIX_LM_HEAD_ROWS_LEAN", "2"},
    Default{Item::dispatch_fuse, "STRIX_DISPATCH_FUSE", "1"},
    Default{Item::score_rows, "STRIX_QPF2K_SCORE_ROWS", "auto"},
    Default{Item::ple_batch, "STRIX_PREFILL_PLE_BATCH", "1"},
    Default{Item::shared_pre, "STRIX_QPF2K_SHARED_PRE", "1"},
    Default{Item::moe_mc, "STRIX_QPF2K_MOE_MC", "1"},
    Default{Item::moe_hdr, "STRIX_QPF2K_MOE_HDR", "1"},
    Default{Item::moe_xor64, "STRIX_QPF2K_MOE_XOR64", "1"},
    Default{Item::hc_lowband, "STRIX_QPF2K_HC_LOWBAND", "3"},
};
static_assert(defaults.size() < 32U);
constexpr unsigned bit(Item item) noexcept { return 1U << static_cast<unsigned>(item); }

// Only injected values are excluded from GSQ execution. Keep parsed environment values
// immutable, and filter AFTER their lazy cache: owners/threads may alternate models.
inline unsigned defaulted_items = 0U;
inline unsigned pre_injected = 0U;
inline bool pre_injected_moe_marker = false;
inline bool baseline = false;
inline thread_local bool gsq_model = false;
inline bool suppressed(Item item, bool gsq = gsq_model) noexcept {
    return baseline || (gsq && (defaulted_items & bit(item)) != 0U);
}
// STRIX_QPF2K_SCHED bit 2 (a headless tiled block ends without its host fence; the next block of the owner runs its
// completion checks) executes only on the main CLI known-prompt cursor (KnownPromptChunkRequest::schedule_extensions).
// Elsewhere (MTP bridge, HTTP serving, cohorts) it is inert: a route compares and records the schedule it actually
// runs, so the default 14 is 12 there, and an explicit 14 is too.
inline constexpr std::uint64_t sched_defer_bit = 2U;
constexpr std::uint64_t effective_sched(std::uint64_t value, bool defers) noexcept {
    return defers ? value : value & ~sched_defer_bit;
}
// A deferred block is never a prefix-cache save/resume boundary. With a CLI prefix-state cache and a DEFAULTED
// schedule, bit 2 is dropped for the run (the cache keeps its boundaries); an explicit value keeps its meaning.
inline bool cli_sched_defers(bool prefix_cache) noexcept {
    return !(prefix_cache && (defaulted_items & bit(Item::sched)) != 0U);
}
// Code-default exact long-context items (STRIX_QSA_UNION_MERGE, STRIX_QSA_SCORE_ROWS64, STRIX_QSA_DECODE_POOL,
// STRIX_QSA_DECODE_SELECT, STRIX_QSA_DECODE_POOL_BF16, STRIX_PREFILL_MOE_MERGE_VEC): on unless their variable is "0", the baseline is requested, STRIX_QPF2K_DEFAULTS=0, or the
// model is a GSQ checkpoint (the same exclusions as the injected defaults above).
// Why such an item is off (reported by --print-effective-config), or null when it is on.
inline const char* long_context_item_suppression(const char* name, bool gsq = gsq_model) noexcept {
    if (baseline) return "baseline";
    if (gsq) return "gsq";
    if (const char* d = std::getenv("STRIX_QPF2K_DEFAULTS"); d && d[0] == '0') return "STRIX_QPF2K_DEFAULTS=0";
    if (const char* v = std::getenv(name); v && v[0] == '0') return "explicit-off";
    return nullptr;
}
inline bool long_context_item_enabled(const char* name, bool gsq = gsq_model) noexcept {
    return !long_context_item_suppression(name, gsq);
}
// Exact cross-group projection batching adds no rotary arithmetic.
template<class Get>
inline bool index_batch_enabled(Get get, bool base, bool gsq) {
    if (base || gsq) return false;
    if (get("STRIX_QPF2K_QSA_INDEX_SHAPE")) return false;
    for (const char* name : {"STRIX_QPF2K_INDEX_ROWS", "STRIX_QSA_INDEX_ROWS"})
        if (const char* value = get(name); value && value[0] == '0') return false;
    if (const char* mask = get("STRIX_QPF2K_QSA_MASK"); mask && (std::strtoul(mask, nullptr, 0) & 0x100U)) return false;
    for (const char* name : {"STRIX_QPF2K_DEFAULTS", "STRIX_QPF2K_INDEX_BATCH"})
        if (const char* value = get(name); value && value[0] == '0') return false;
    return true;
}
inline bool index_batch_enabled() {
    return index_batch_enabled([](const char* name) { return std::getenv(name); }, baseline, gsq_model);
}
class ModelScope {
public:
    explicit ModelScope(bool gsq) noexcept : previous_(gsq_model) { gsq_model = gsq; }
    ~ModelScope() { gsq_model = previous_; }
    ModelScope(const ModelScope&) = delete;
    ModelScope& operator=(const ModelScope&) = delete;
private:
    bool previous_;
};
inline bool has_gsq_gate(const ModelManifest& manifest) noexcept {
    for (const auto& tensor : manifest.tensors)
        if (tensor.name.ends_with(".ffn_gate_exps.weight") && gsq_a8::gate_type(tensor.ggml_type)) return true;
    return false;
}

// Shared by startup and the host regression test; put returns true only for an
// actually injected value. QSA/HC masks retain the parent-dependent old defaults.
template<class Get, class Put>
unsigned initialize(Get get, Put put, unsigned allowed = ~0U) {
    if (baseline) return 0U;
    if (const char* b = get("STRIX_BASELINE"); b && b[0] == '1' && b[1] == '\0') return 0U;
    if (const char* d = get("STRIX_QPF2K_DEFAULTS"); d && d[0] == '0') return 0U;
    unsigned injected = 0U;
    for (const auto& entry : defaults) {
        if (!(allowed & bit(entry.item))) continue;
        if (entry.item == Item::qsa_mask && !(injected & bit(Item::qsa))) continue;
        if (entry.item == Item::hc_mask && !(injected & bit(Item::hc))) continue;
        if (!get(entry.name) && put(entry.name, entry.value)) injected |= bit(entry.item);
    }
    const unsigned moe = ((injected & bit(Item::moe_da16)) ? 1U : 0U) |
        ((injected & bit(Item::moe_route)) ? 2U : 0U);
    if (moe && !get("STRIX_QPF2K_MOE_DEFAULTED"))
        put("STRIX_QPF2K_MOE_DEFAULTED", moe == 3U ? "3" : moe == 1U ? "1" : "2");
    return injected;
}
// The historical set is restricted to known-input tiled kernels/scheduling:
// SCHED selects tile/tail owners; DENSE batches tiled projections; GDN,
// GATES_BATCH and CORE16 group tiled recurrence/gates; QSA/MASK select tiled
// attention; HC/MASK/OUTPUT3/BAND select tiled hyper-mix; MOE_DA16/ROUTE
// select tiled expert work. Each reader filters injected values per GSQ owner.
inline constexpr unsigned historical_items = bit(Item::moe_route) * 2U - 1U;
// Exact prefill alternatives keep admission when explicitly disabled one at a time.
inline constexpr unsigned exact_prefill_items = bit(Item::score_rows) | bit(Item::ple_batch) |
    bit(Item::shared_pre) | bit(Item::moe_mc) | bit(Item::moe_hdr) | bit(Item::moe_xor64) | bit(Item::hc_lowband);
// L1 has its own registered-mapping, single-session tiled admission (including
// serving/MTP/cohort exclusions). Other new wrapper items are decoder-only:
// PIPE also affects MTP priming; COHORT_LEAN affects active serving heads;
// DISPATCH_FUSE affects verification windows. Do not inject those at startup.
// DENSE_ROWS, ATTEND_BATCH, SCORE_PARK, NORM_GROUP, INDEX_ROWS and ROWS_LEAN
// are also deferred to decoder admission so non-decoder clients keep old defaults.
// SCORE_ROWS, PLE_BATCH, SHARED_PRE, MOE_MC, MOE_HDR and MOE_XOR64 share those prefill
// launchers (including serving/MTP priming); defer them too, preserving fixture defaults.
// HC_LOWBAND also waits for decoder admission; its route guard retains unsupported layouts.
inline constexpr unsigned startup_items = historical_items | bit(Item::ple_reclaim);
// Startup injection writes into the real environment, so a process that re-executes itself before
// undo_pre_injected (eager code loading restarts the image) hands its injected values to the new image, which
// would find them set and treat them as user values. This marker carries the injected set (items, plus
// pre_injected_marker_moe for STRIX_QPF2K_MOE_DEFAULTED) across execv, like STRIX_QPF2K_MOE_DEFAULTED carries
// the MoE provenance; the new image restores provenance only for items whose value still equals the default.
// Internal provenance, not an execution control: identities and environment sweeps skip it, and
// undo_pre_injected erases it with the values.
inline constexpr const char* pre_injected_env = "STRIX_QPF2K_PRE_INJECTED";
inline constexpr unsigned pre_injected_marker_moe = 1U << 31;
static_assert(defaults.size() < 31U);
template<class Get>
unsigned carried_pre_injected(Get get, unsigned allowed) {
    const char* text = get(pre_injected_env);
    if (!text || !*text) return 0U;
    char* end = nullptr;
    const auto value = std::strtoul(text, &end, 10);
    if (!end || *end || value > 0xffffffffUL) return 0U;
    unsigned carried = 0U;
    for (const auto& entry : defaults) {
        if (!(value & bit(entry.item)) || !(allowed & bit(entry.item))) continue;
        const char* current = get(entry.name);
        if (current && std::string_view(current) == entry.value) carried |= bit(entry.item);
    }
    if ((value & pre_injected_marker_moe) && get("STRIX_QPF2K_MOE_DEFAULTED")) carried |= pre_injected_marker_moe;
    return carried;
}
template<class Get, class Put>
void initialize_before_main(Get get, Put put) {
    const bool had_marker = get("STRIX_QPF2K_MOE_DEFAULTED") != nullptr;
    // Read before initialize: a marker counts only where defaults are enabled (initialize returns early otherwise).
    const unsigned carried = carried_pre_injected(get, startup_items);
    pre_injected = initialize(get, put, startup_items);
    pre_injected_moe_marker = !had_marker && get("STRIX_QPF2K_MOE_DEFAULTED");
    const bool enabled = !baseline && !(get("STRIX_BASELINE") && std::string_view(get("STRIX_BASELINE")) == "1") &&
        !(get("STRIX_QPF2K_DEFAULTS") && get("STRIX_QPF2K_DEFAULTS")[0] == '0');
    if (enabled) {
        pre_injected |= carried & ~pre_injected_marker_moe;
        pre_injected_moe_marker = pre_injected_moe_marker || (carried & pre_injected_marker_moe) != 0U;
    }
    defaulted_items |= pre_injected;
    if (pre_injected || pre_injected_moe_marker)
        put(pre_injected_env, std::to_string(pre_injected | (pre_injected_moe_marker ? pre_injected_marker_moe : 0U)).c_str());
}
template<class Erase>
void undo_pre_injected(Erase erase) {
    for (const auto& entry : defaults)
        if (pre_injected & bit(entry.item)) erase(entry.name);
    if (pre_injected_moe_marker) erase("STRIX_QPF2K_MOE_DEFAULTED");
    erase(pre_injected_env);
    pre_injected = defaulted_items = 0U;
    pre_injected_moe_marker = false;
    baseline = true;
}
} // namespace strix::qpf2k
