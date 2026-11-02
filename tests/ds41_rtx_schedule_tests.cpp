#include "strix/ds41_rtx_schedule.hpp"
#undef NDEBUG
#include <cassert>
#include <cstdint>
#include <iostream>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

#if __has_include("strix_stream/ss.h")
#include "strix_stream/ss.h"
#endif
#if defined(SS_BUDGET_MARGINS) && defined(SS_BUDGET_JSON) && defined(SS_BUDGET_OBSERVE)
#include "strix/ds41_rtx_budget.hpp"
// Whole-device budgets: tierstream's planner (ss_budget_plan) sizes the arena / pinned tier from the caps minus the measured
// fixed parts; tier_plan agrees on every count; no cap or covering caps = today's tier_plan exactly.
static void budget_tests() {
    using namespace strix::ds41::rtx;
    constexpr double gib = 1073741824.0;
    constexpr std::uint64_t unit = 9953280, down = 3870720, MiB = 1ULL << 20;
    const auto same = [](const TierPlan& a, const TierPlan& b) {
        return a.resident == b.resident && a.pinned == b.pinned && a.disk == b.disk && a.inclusive == b.inclusive && a.spare_blocks == b.spare_blocks &&
               a.staging == b.staging && a.three_tier == b.three_tier && a.need_host_gib == b.need_host_gib && a.pinned_peak_gib == b.pinned_peak_gib &&
               a.cap_units == b.cap_units && (a.refuse == nullptr) == (b.refuse == nullptr);
    };
    // the production launch: arena 74.8 GiB + 96 spares, 48 host spares, 8 staging, max rows 6 x top-k 6
    const std::uint32_t spares = 96;
    const std::uint32_t slots = static_cast<std::uint32_t>(74.8 * gib / static_cast<double>(unit)) + spares;
    TierPlanIn tin;
    tin.keys = 40ULL * 384ULL;
    tin.unit_bytes = unit;
    tin.residents = slots - spares;
    tin.host_spares = 48;
    tin.host_margin_gib = 1.0;
    tin.max_pinned_gib = 74.0;
    tin.host_avail_gib = 110.0;
    tin.extra_gib = 16.0 * down / gib;
    tin.staging = 8;
    // the measured fixed parts of a PRO load (context + trunk 13.11 GB at the trunk's end incl. the context,
    // drafter 4.42 GB, workspaces): the planner's input as the replica builds it
    const auto input = [&](double vram, double ram, std::uint32_t req_slots, const TierPlanIn& t) {
        BudgetPlanIn bi;
        bi.caps.vram_total_gib = vram;
        bi.caps.ram_total_gib = ram;
        bi.caps.receipt = true;
        bi.device_fixed = {{"context", 520 * MiB}, {"trunk", 11980 * MiB}, {"drafter", 4215 * MiB}, {"workspaces", 180 * MiB}};
        bi.host_pageable = {{"process_commit", 2600 * MiB}};
        bi.host_pinned = {{"engine_pinned", 0}};
        bi.units = t.keys;
        bi.unit_bytes = unit;
        bi.request_slots = req_slots;
        bi.spare_slots = spares;
        bi.min_residents = 36;
        bi.arena_extra_bytes = 73ULL * unit + 256 * MiB + 64 * MiB;
        bi.host_request_blocks = tier_plan(t).cap_units;
        bi.host_spares = t.host_spares;
        bi.staging = t.staging;
        bi.host_extra_bytes = 16ULL * down;
        bi.file_tier = t.file_tier;
        return bi;
    };
    // (a) --budget-receipt without caps: the planner returns today's request; tier_plan unchanged in every field
    {
        const auto bp = budget_plan(input(-1.0, -1.0, slots, tin));
        assert(!bp.refused() && bp.out.arena_slots == slots && !bp.out.device_capped && !bp.out.host_capped);
        const auto t0 = tier_plan(tin), t1 = tier_plan(apply_budget(tin, bp));
        assert(same(t0, t1));
        check_budget_parity(t1, bp);
        // the receipt JSON (ss_budget_plan_json): members only, the measured components by name
        const std::string j = bp.json();
        assert(j.front() == '"' && j.find("\"device_fixed\":{\"trunk\":") != std::string::npos &&
               j.find("\"arena reserves\":") != std::string::npos && j.find("\"device_context_bytes\":545259520") != std::string::npos &&
               j.find("\"pageable_fixed\":{\"process_commit\":") != std::string::npos && j.find("\"refused\":false") != std::string::npos);
    }
    // (b) caps that cover everything (96 GiB card, 128 GiB host): today's plan exactly
    {
        const auto bp = budget_plan(input(96.0, 128.0, slots, tin));
        assert(!bp.refused() && bp.out.arena_slots == slots && !bp.out.device_capped && !bp.out.host_capped);
        const auto t0 = tier_plan(tin), t1 = tier_plan(apply_budget(tin, bp));
        assert(same(t0, t1));
        assert(bp.out.device_planned_bytes + bp.in.vram_margin_bytes <= bp.in.vram_total_bytes);
        check_budget_parity(t1, bp);
    }
    // (c) the whole-device 24 + 32 target: arena = what the 24 GiB leave; pinned = what 32 GiB leave; the rest in the file
    {
        const auto bp = budget_plan(input(24.0, 32.0, slots, tin));
        assert(!bp.refused() && bp.out.device_capped && bp.out.host_capped && bp.out.three_tier);
        assert(bp.in.vram_margin_bytes == 1536 * MiB && bp.in.ram_margin_bytes == 1024 * MiB);
        const std::uint64_t dev_used = bp.out.device_planned_bytes + bp.in.vram_margin_bytes;
        assert(dev_used <= 24ULL << 30 && dev_used + unit > 24ULL << 30);
        assert(bp.out.host_planned_bytes + bp.in.ram_margin_bytes <= 32ULL << 30);
        assert(bp.out.resident > 36 && bp.out.resident < 1000);  // ~3.5 GiB of experts on a 24 GiB card
        const auto t1 = tier_plan(apply_budget(tin, bp));
        check_budget_parity(t1, bp);
        assert(t1.pinned + t1.spare_blocks + t1.staging == bp.out.host_cap_units);  // the pinned tier fills the cap
        assert(t1.need_host_gib - tin.host_margin_gib <= (32.0 * gib - 2600.0 * MiB - 1.0 * gib) / gib + 1e-9);
        std::cout << "  24+32: arena " << bp.out.arena_slots << " slots (" << bp.out.resident << " experts), pinned " << t1.pinned << ", file " << t1.disk
                  << ", min caps " << bp.out.min_vram_total_bytes / gib << " / " << bp.out.min_ram_total_bytes / gib << " GiB\n";
    }
    // (d) a cap below the trunk + drafter refuses at startup and names the minimum
    {
        const auto bp = budget_plan(input(16.0, -1.0, slots, tin));
        assert(bp.refused() && bp.device_refused() && bp.status == SS_E_CAPACITY);
        const std::string m = bp.refusal();
        assert(m.find("--vram-total-gib 16.00: the trunk and the fixed device parts do not fit") != std::string::npos);
        assert(m.find("the minimum is --vram-total-gib ") != std::string::npos && m.find("trunk 11.70") != std::string::npos);
        assert(m.find("margin 1.50") != std::string::npos && m.find("arena reserves") != std::string::npos);
        const double min_gib = static_cast<double>(bp.out.min_vram_total_bytes) / gib;
        assert(min_gib > 19.0 && min_gib < 22.0);
        assert(!budget_plan(input(std::ceil(min_gib * 100.0) / 100.0, -1.0, slots, tin)).refused());
        const auto hb = budget_plan(input(-1.0, 3.0, slots, tin));
        assert(hb.refused() && hb.host_refused() && hb.refusal().find("the minimum is --ram-total-gib ") != std::string::npos);
        const double hmin = static_cast<double>(hb.out.min_ram_total_bytes) / gib;
        assert(!budget_plan(input(-1.0, std::ceil(hmin * 100.0) / 100.0, slots, tin)).refused());
    }
    // (e) explicit --vram-expert-gib / --ram-expert-gib below the caps stay the bound (a cap only shrinks)
    {
        TierPlanIn t = tin;
        t.ram_gib = 16.0;
        const std::uint32_t small = static_cast<std::uint32_t>(4.0 * gib / static_cast<double>(unit)) + spares;
        t.residents = small - spares;
        const auto bp = budget_plan(input(24.0, 32.0, small, t));
        assert(!bp.refused() && bp.out.arena_slots == small && !bp.out.device_capped && !bp.out.host_capped);
        const auto t0 = tier_plan(t), t1 = tier_plan(apply_budget(t, bp));
        assert(same(t0, t1));
        check_budget_parity(t1, bp);
    }
    // (f) parity under random caps / inputs: whenever the planner accepts, tier_plan agrees on every count
    std::mt19937_64 rng(7);
    int accepted = 0;
    for (int it = 0; it < 4000; ++it) {
        TierPlanIn t = tin;
        t.host_avail_gib = 8.0 + static_cast<double>(rng() % 120);
        t.max_pinned_gib = 8.0 + static_cast<double>(rng() % 100);
        t.ram_gib = rng() % 3 ? -1.0 : static_cast<double>(rng() % 90);
        t.inclusive_gib = rng() % 4 ? 0.0 : -1.0;
        t.file_tier = rng() % 6 != 0;
        t.staging = 1 + static_cast<std::uint32_t>(rng() % 16);
        t.host_spares = static_cast<std::uint32_t>(rng() % 96);
        const std::uint32_t req = spares + 1 + static_cast<std::uint32_t>(rng() % 9000);
        t.residents = req - spares;
        const double vram = rng() % 4 ? 17.0 + static_cast<double>(rng() % 80) : -1.0;
        const double ram = rng() % 4 ? 4.0 + static_cast<double>(rng() % 120) : -1.0;
        const auto bp = budget_plan(input(vram, ram, req, t));
        if (bp.refused()) continue;
        ++accepted;
        const auto tp = tier_plan(apply_budget(t, bp));
        check_budget_parity(tp, bp);  // throws on disagreement
        if (bp.in.vram_total_bytes) assert(bp.out.device_planned_bytes + bp.in.vram_margin_bytes <= bp.in.vram_total_bytes);
        if (bp.in.ram_total_bytes && !tp.refuse) assert(bp.out.host_planned_bytes + bp.in.ram_margin_bytes <= bp.in.ram_total_bytes);
        if (vram < 0 && ram < 0) assert(same(tier_plan(t), tp));
    }
    assert(accepted > 1000);
    std::cout << "PASS whole-device budgets: tierstream ss_budget_plan = tier_plan counts, no cap / covering caps = today, 24+32 split, refusal with the minimum ("
              << accepted << " random plans)\n";
}
#else
static void budget_tests() { std::cout << "SKIP whole-device budgets: the tierstream tree has no ss_budget margins / plan JSON / observe\n"; }
#endif

int main() {
    using namespace strix::ds41::rtx;
    constexpr std::uint32_t gate = 3041280, down = 3870720;
    constexpr auto p = ce_pacing(gate, down);
    static_assert(p.background_limit == 9953280);
    static_assert(p.outstanding_threshold == 6082561);
    // Any permitted part, including after a partial victim was resurrected,
    // stays under the background bound. Both threshold endpoints matter.
    for (auto outstanding : {0U, gate, 2U * gate, p.outstanding_threshold - 1, p.outstanding_threshold}) {
        for (auto part : {gate, down}) {
            if (outstanding < p.outstanding_threshold)
                assert(std::uint64_t{outstanding} + part <= p.background_limit);
        }
    }
    // A full victim can finish without a forced wait (all three parts).
    std::uint64_t outstanding = 0;
    for (auto part : {gate, gate, down}) {
        assert(outstanding < p.outstanding_threshold);
        outstanding += part;
    }
    assert(outstanding == p.background_limit);
    assert(outstanding >= p.outstanding_threshold);
    std::cout << "PASS CE pacing: background bound and partial-victim progress\n";

    // Three-tier plan: the production shape (40 x 384 experts of 9953280 bytes, arena 74.8 GiB + 96 spares).
    constexpr double gib = 1073741824.0;
    TierPlanIn in;
    in.keys = 40ULL * 384ULL;
    in.unit_bytes = 9953280ULL;
    in.residents = static_cast<std::uint64_t>(74.8 * gib / 9953280.0);
    in.host_spares = 48;
    in.host_margin_gib = 6.0;
    in.engine_pinned_gib = 0.6;
    in.max_pinned_gib = 74.0;
    in.extra_gib = 16.0 * down / gib;
    const std::uint64_t nonres = in.keys - in.residents;
    // (a) enough RAM: today's placement exactly (every non-resident expert pinned, no file tier, today's need)
    in.host_avail_gib = 90.0;
    auto a = tier_plan(in);
    assert(!a.three_tier && a.refuse == nullptr && a.pinned == nonres && a.disk == 0 && a.staging == 0 && a.inclusive == 0);
    assert(a.spare_blocks == 48);
    const double today_need = (static_cast<double>(nonres + 48) * 9953280.0 + 16.0 * down) / gib + 6.0;
    assert(a.need_host_gib > today_need - 1e-9 && a.need_host_gib < today_need + 1e-9);
    assert(a.pinned_peak_gib <= in.max_pinned_gib);
    // (b) not enough RAM (today: refusal): auto fills what fits, the rest goes to the file tier; never above the room
    in.host_avail_gib = 40.0;
    auto b = tier_plan(in);
    assert(b.three_tier && b.refuse == nullptr && b.disk > 0 && b.pinned + b.disk == nonres && b.staging == 8);
    assert(b.need_host_gib <= in.host_avail_gib + 1e-9 && b.pinned_peak_gib <= in.max_pinned_gib + 1e-9);
    assert(b.need_host_gib > in.host_avail_gib - 2.0 * 9953280.0 / gib);  // uses the room (floor of one unit)
    // (c) without the file tier the plan stays two-tier (the budget checks refuse as before)
    in.file_tier = false;
    auto c = tier_plan(in);
    assert(!c.three_tier && c.pinned == nonres && c.need_host_gib > in.host_avail_gib);
    in.file_tier = true;
    // (d) explicit budgets: 48 GiB VRAM + 64 GiB RAM (the RAM figure includes spares and staging)
    in.host_avail_gib = 120.0;
    in.max_pinned_gib = 100.0;
    in.residents = static_cast<std::uint64_t>(48.0 * gib / 9953280.0);
    in.ram_gib = 64.0;
    auto d = tier_plan(in);
    assert(d.three_tier && !d.ram_auto && d.pinned + 48 + 8 == static_cast<std::uint64_t>(64.0 * gib / 9953280.0));
    assert(d.resident + d.pinned + d.disk == in.keys && d.file_gib > 25.0 && d.file_gib < 35.0);
    // (e) an explicit RAM budget above what the host can pin now is clamped (never refused)
    in.ram_gib = 500.0;
    auto e = tier_plan(in);
    assert(!e.three_tier || e.need_host_gib <= in.host_avail_gib + 1e-9);
    // (f) inclusive share: only from RAM left over, never displacing a home; auto is capped
    in.ram_gib = -1.0;
    in.residents = static_cast<std::uint64_t>(74.8 * gib / 9953280.0);
    in.host_avail_gib = 200.0;
    in.max_pinned_gib = 200.0;
    in.inclusive_gib = 8.0;
    auto f = tier_plan(in);
    assert(!f.three_tier && f.pinned == nonres && f.inclusive == static_cast<std::uint64_t>(8.0 * gib / 9953280.0));
    assert(f.spare_blocks == 48 + f.inclusive);
    in.inclusive_gib = -1.0;
    auto g = tier_plan(in);
    assert(g.inclusive == static_cast<std::uint64_t>(16.0 * gib / 9953280.0));
    in.host_avail_gib = 40.0;
    auto h = tier_plan(in);  // RAM short: no clean copies, homes first
    assert(h.three_tier && h.inclusive == 0);
    // (g) a host that cannot even hold the spares: refused with a message
    in.host_avail_gib = 6.2;
    auto r = tier_plan(in);
    assert(r.refuse != nullptr);
    std::cout << "PASS three-tier plan: today's placement when it fits, file tier instead of refusal, explicit budgets, inclusive share\n";
    budget_tests();
}
