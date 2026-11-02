#pragma once

// Whole-device budgets (--vram-total-gib / --ram-total-gib): "as if the user had an N GiB GPU and M GiB of RAM".
// Glue between the replica's measurements and tierstream's one budget planner (strix_stream/ss.h: ss_budget_plan,
// ss_budget_plan_json; the peaks against the caps are the replica's ss_budget_ledger). Pure CPU arithmetic; no CUDA.
//
// The replica measures what the non-streamed parts took on the device (CUDA context, trunk / dense weights + attention
// state for --max-positions + its buffers, drafter, workspaces) and the process commit on the host, then:
//   arena slots  = min(today's arena, (cap - context - fixed - margin - arena reserves) / expert)
//   pinned tier  = min(today's RAM ceiling, (cap - commit - engine pinned - load staging - margin) / expert), the rest
//                  in the file
// A cap never grows a tier: caps that cover everything give today's placement. No cap = today's code path.
#include "strix/ds41_rtx_schedule.hpp"
#include "strix_stream/ss.h"

#if !defined(SS_BUDGET_MARGINS) || !defined(SS_BUDGET_JSON) || !defined(SS_BUDGET_OBSERVE)
#error "the tierstream tree has no ss_budget margins / plan JSON / observe (update tierstream)"
#endif

#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix::ds41::rtx {

constexpr double kBudgetGiB = 1073741824.0;

// a tierstream budget call that must succeed (the ledger's observe / stats): the library's message otherwise
inline void budget_check(ss_status st) {
    if (st != SS_OK) throw std::runtime_error(std::string("budget: ") + ss_last_error());
}

inline std::uint64_t budget_bytes(double gib) {
    return gib > 0.0 ? static_cast<std::uint64_t>(std::llround(gib * kBudgetGiB)) : 0U;
}

struct BudgetCaps {
    double vram_total_gib = -1.0;  // --vram-total-gib / STRIX_DS41_VRAM_TOTAL_GIB (< 0: no cap)
    double ram_total_gib = -1.0;   // --ram-total-gib / STRIX_DS41_RAM_TOTAL_GIB (< 0: no cap; process commit, pinned incl.)
    double vram_margin_gib = 1.5;  // --vram-margin-gib: device allocations after the plan (graphs, warmup, requests)
    double ram_margin_gib = 1.0;   // --ram-margin-gib: host commit after the plan
    bool receipt = false;          // --budget-receipt: measure and print the budget events without caps
    [[nodiscard]] bool device_capped() const { return vram_total_gib > 0.0; }
    [[nodiscard]] bool host_capped() const { return ram_total_gib > 0.0; }
    [[nodiscard]] bool capped() const { return device_capped() || host_capped(); }
    [[nodiscard]] bool measuring() const { return capped() || receipt; }
};

using BudgetComponents = std::vector<std::pair<std::string, std::uint64_t>>;

struct BudgetPlanIn {
    BudgetCaps caps;
    BudgetComponents device_fixed;  // measured: context (named "context": ss_budget_in.device_context_bytes), trunk, drafter, workspaces
    BudgetComponents host_pageable; // measured: process commit before the expert tiers, ...
    BudgetComponents host_pinned;   // the engine's own pinned buffers
    std::uint64_t units = 0;        // routed experts (layers x experts)
    std::uint64_t unit_bytes = 0;   // one expert (device slot = pinned block stride)
    std::uint64_t request_slots = 0;     // today's arena (spares included)
    std::uint32_t spare_slots = 0;       // the arena's spares (as tier_plan's residents = slots - spares)
    std::uint32_t min_residents = 1;     // fewer refuses
    std::uint64_t arena_extra_bytes = 0; // bank ring + site workspaces + graph reserve (a named device component)
    std::uint64_t host_request_blocks = UINT64_MAX;  // tier_plan's cap_units for today's input (UINT64_MAX: none)
    std::uint32_t host_spares = 0, staging = 0;
    std::uint64_t host_extra_bytes = 0;  // load staging
    std::uint64_t inclusive_bytes = 0;   // --ram-inclusive-gib (SS_BUDGET_AUTO = auto, 0 = off)
    std::uint64_t inclusive_auto_cap_bytes = 16ULL << 30;
    bool file_tier = true;
};

struct BudgetPlan {
    ss_budget_in in{};    // the component pointers are bound to the vectors below by bind() (never kept across copies)
    ss_budget_out out{};
    ss_status status = SS_OK;
    BudgetComponents device_fixed, host_pinned, host_pageable;
    std::uint64_t host_request_blocks = UINT64_MAX;

    [[nodiscard]] bool refused() const { return out.refused != 0; }
    [[nodiscard]] bool device_refused() const { return refused() && std::string(out.refuse).rfind("VRAM cap", 0) == 0; }
    [[nodiscard]] bool host_refused() const { return refused() && !device_refused(); }
    // ss_budget_in with its component arrays pointing into the given storage
    [[nodiscard]] ss_budget_in bind(std::vector<ss_budget_component>& dev, std::vector<ss_budget_component>& pin,
                                    std::vector<ss_budget_component>& page) const {
        const auto fill = [](const BudgetComponents& from, std::vector<ss_budget_component>& to) {
            to.clear();
            for (const auto& [n, v] : from) to.push_back({n.c_str(), v});
        };
        fill(device_fixed, dev);
        fill(host_pinned, pin);
        fill(host_pageable, page);
        ss_budget_in b = in;
        b.device_fixed = dev.empty() ? nullptr : dev.data();
        b.n_device_fixed = static_cast<std::uint32_t>(dev.size());
        b.pinned_fixed = pin.empty() ? nullptr : pin.data();
        b.n_pinned_fixed = static_cast<std::uint32_t>(pin.size());
        b.pageable_fixed = page.empty() ? nullptr : page.data();
        b.n_pageable_fixed = static_cast<std::uint32_t>(page.size());
        return b;
    }
    [[nodiscard]] std::string json() const {  // the planner's members (no braces)
        std::vector<ss_budget_component> d, p, g;
        const ss_budget_in b = bind(d, p, g);
        std::string s(ss_budget_plan_json(&b, &out, nullptr, 0) + 1, '\0');
        ss_budget_plan_json(&b, &out, s.data(), s.size());
        s.resize(s.size() - 1);
        return s;
    }
    // the startup refusal: which cap, why, and the smallest value that loads (0.01 GiB rounded up, and whole GiB)
    [[nodiscard]] std::string refusal() const {
        const auto up = [](std::uint64_t b) { return std::ceil(static_cast<double>(b) / kBudgetGiB * 100.0 - 1e-9) / 100.0; };
        const auto whole = [](std::uint64_t b) { return static_cast<long long>(b / (1ULL << 30) + (b % (1ULL << 30) ? 1 : 0)); };
        std::ostringstream os;
        os.setf(std::ios::fixed);
        os.precision(2);
        if (device_refused()) {
            os << "--vram-total-gib " << static_cast<double>(in.vram_total_bytes) / kBudgetGiB
               << ": the trunk and the fixed device parts do not fit (" << out.refuse << "); the minimum is --vram-total-gib "
               << up(out.min_vram_total_bytes) << " (" << whole(out.min_vram_total_bytes) << " rounded up)";
        } else if (refused()) {
            os << "--ram-total-gib " << static_cast<double>(in.ram_total_bytes) / kBudgetGiB << ": " << out.refuse;
            if (in.ram_total_bytes)
                os << "; the minimum is --ram-total-gib " << up(out.min_ram_total_bytes) << " (" << whole(out.min_ram_total_bytes) << " rounded up)";
        }
        return os.str();
    }
};

inline BudgetPlan budget_plan(const BudgetPlanIn& b) {
    if (b.request_slots < b.spare_slots) throw std::runtime_error("budget: the arena request is below its spares");
    BudgetPlan r;
    ss_budget_in_defaults(&r.in);
    r.in.vram_total_bytes = b.caps.device_capped() ? budget_bytes(b.caps.vram_total_gib) : 0U;
    r.in.ram_total_bytes = b.caps.host_capped() ? budget_bytes(b.caps.ram_total_gib) : 0U;
    r.in.vram_margin_bytes = b.caps.device_capped() ? budget_bytes(b.caps.vram_margin_gib) : 0U;
    r.in.ram_margin_bytes = b.caps.host_capped() ? budget_bytes(b.caps.ram_margin_gib) : 0U;
    for (const auto& [n, v] : b.device_fixed) {
        if (n == "context") r.in.device_context_bytes += v;
        else r.device_fixed.emplace_back(n, v);
    }
    if (b.arena_extra_bytes) r.device_fixed.emplace_back("arena reserves", b.arena_extra_bytes);
    r.host_pinned = b.host_pinned;
    r.host_pageable = b.host_pageable;
    r.in.keys = b.units;
    r.in.slot_bytes = b.unit_bytes;
    r.in.unit_bytes = b.unit_bytes;
    r.in.arena_units = b.request_slots - b.spare_slots;
    r.in.arena_spares = b.spare_slots;
    r.in.min_resident = b.min_residents;
    r.host_request_blocks = b.host_request_blocks;
    r.in.ram_expert_bytes = b.host_request_blocks == UINT64_MAX ? SS_BUDGET_AUTO : b.host_request_blocks * b.unit_bytes;
    r.in.host_spares = b.host_spares;
    r.in.staging = b.staging;
    r.in.load_staging_bytes = b.host_extra_bytes;
    r.in.inclusive_bytes = b.inclusive_bytes;
    r.in.inclusive_auto_cap_bytes = b.inclusive_auto_cap_bytes;
    r.in.file_tier = b.file_tier ? 1U : 0U;
    r.out.struct_size = static_cast<std::uint32_t>(sizeof r.out);
    std::vector<ss_budget_component> d, p, g;
    const ss_budget_in bound = r.bind(d, p, g);
    r.status = ss_budget_plan(&bound, &r.out);
    if (r.status != SS_OK && r.status != SS_E_CAPACITY) throw std::runtime_error(std::string("budget planner: ") + ss_last_error());
    return r;
}

// tier_plan's input under the plan: the planner's residents, and the RAM budget lowered to the planner's pinned
// ceiling when the RAM cap binds (+0.5 block: tier_plan floors GiB / unit back to the same count). Otherwise today's.
inline TierPlanIn apply_budget(TierPlanIn tin, const BudgetPlan& bp) {
    tin.residents = bp.out.resident;
    if (bp.in.ram_total_bytes && bp.out.host_cap_units < bp.host_request_blocks)
        tin.ram_gib = (static_cast<double>(bp.out.host_cap_units) + 0.5) * static_cast<double>(tin.unit_bytes) / kBudgetGiB;
    return tin;
}

// the planner (tierstream) decides; tier_plan must agree on every count it hands to TierSite
inline void check_budget_parity(const TierPlan& tp, const BudgetPlan& bp) {
    if (tp.refuse) return;  // tier_plan's own refusal is reported as before
    if (tp.resident != bp.out.resident || tp.pinned != bp.out.pinned || tp.disk != bp.out.disk ||
        tp.three_tier != (bp.out.three_tier != 0) || (tp.three_tier && tp.staging != bp.out.staging)) {
        std::ostringstream os;
        os << "budget: tier_plan and the tierstream planner disagree (resident " << tp.resident << "/" << bp.out.resident << ", pinned "
           << tp.pinned << "/" << bp.out.pinned << ", file " << tp.disk << "/" << bp.out.disk << ")";
        throw std::runtime_error(os.str());
    }
}

}  // namespace strix::ds41::rtx
