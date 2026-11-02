#pragma once

// Host/device-independent scheduling arithmetic; never changes tensor math.
#include <cstdint>

namespace strix::ds41::rtx {
struct CePacing {
    std::uint32_t layer_bytes;
    std::uint32_t outstanding_threshold;
    std::uint64_t background_limit;
};
// tierstream tests outstanding < threshold BEFORE submitting a whole part.
// Subtract the largest part so its one-part overshoot stays <= one expert.
// Keep enough room to finish an expert: a threshold of one part would strand
// partial victims (completion events are recorded only after the last part).
constexpr CePacing ce_pacing(std::uint32_t gate, std::uint32_t down) {
    const std::uint64_t unit = 2ULL * gate + down;
    const std::uint32_t largest = gate > down ? gate : down;
    return {gate, static_cast<std::uint32_t>(unit - largest + 1), unit};
}

// Three-tier expert placement (VRAM arena, pinned RAM, NVMe file tier): how many routed experts each tier holds. Pure
// arithmetic on the budgets; the replica prints the result as its "tier_plan" event and hands the counts to TierSite
// (the seed priority decides WHICH experts: the first `resident` device, the next `pinned` RAM, the rest file).
// GiB = 2^30 bytes. Every count is in units of one expert's pinned host stride (unit_bytes).
struct TierPlanIn {
    std::uint64_t keys = 0;        // routed experts (layers x experts)
    std::uint64_t unit_bytes = 0;  // one expert's host stride
    std::uint64_t residents = 0;   // device experts (the arena's slots minus its spares)
    std::uint32_t host_spares = 0; // free pinned blocks for write-backs (--host-spares)
    double ram_gib = -1.0;         // --ram-expert-gib: < 0 = auto (what fits: available - margin, max pinned)
    double inclusive_gib = 0.0;    // --ram-inclusive-gib: < 0 = auto (RAM left over, at most inclusive_auto_cap_gib), 0 = off
    double inclusive_auto_cap_gib = 16.0;
    double host_avail_gib = 0.0, host_margin_gib = 0.0;  // available host memory now; the safety margin kept free
    double engine_pinned_gib = 0.0;                      // the engine's own pinned buffers (outside the expert tiers)
    double max_pinned_gib = 0.0;                         // --max-pinned-gib (engine + experts)
    double extra_gib = 0.0;                              // load staging the budget always adds (16 x down part)
    bool file_tier = true;         // false: no file tier (the pinned tier must hold every non-resident expert)
    std::uint32_t staging = 8;     // file reads' pinned staging buffers (three-tier only)
};
struct TierPlan {
    std::uint64_t resident = 0, pinned = 0, disk = 0;  // experts per tier
    std::uint64_t inclusive = 0;     // pinned blocks device experts may keep as clean copies (out of spare_blocks)
    std::uint64_t spare_blocks = 0;  // host_spares + inclusive (the library's host_spare_units)
    std::uint64_t staging = 0;       // staging buffers (three-tier)
    double vram_gib = 0, ram_gib = 0, file_gib = 0, inclusive_gib = 0;
    double room_gib = 0;             // pinned bytes the host can give the expert tiers now (auto's ceiling)
    double need_host_gib = 0;        // available host memory the load needs (blocks + extra + margin), as before
    double pinned_peak_gib = 0;      // engine + expert pinned bytes at peak, as before
    std::uint64_t cap_units = 0;     // the pinned-tier ceiling in blocks this plan used (budget or room, whichever is lower)
    bool three_tier = false;         // disk > 0: the library's file tier is on
    bool ram_auto = true;
    const char* refuse = nullptr;    // set: the load cannot go on (the message)
};
inline TierPlan tier_plan(const TierPlanIn& in) {
    constexpr double gib = 1073741824.0;
    TierPlan p;
    p.ram_auto = in.ram_gib < 0.0;
    const double unit = static_cast<double>(in.unit_bytes) / gib;
    p.resident = in.residents < in.keys ? in.residents : in.keys;
    const std::uint64_t nonres = in.keys - p.resident;
    const auto units_of = [&](double g) -> std::uint64_t { return g > 0.0 && unit > 0.0 ? static_cast<std::uint64_t>(g / unit) : 0U; };
    const double room_host = in.host_avail_gib - in.host_margin_gib - in.extra_gib;
    const double room_pin = in.max_pinned_gib - in.engine_pinned_gib - in.extra_gib;
    p.room_gib = room_host < room_pin ? room_host : room_pin;
    const std::uint64_t room = units_of(p.room_gib);
    std::uint64_t cap = p.ram_auto ? room : units_of(in.ram_gib);
    if (in.file_tier && cap > room) cap = room;  // never more than the host can pin now
    p.cap_units = cap;
    const std::uint64_t want = nonres + in.host_spares;
    if (want <= cap || !in.file_tier) {
        p.pinned = nonres;  // the two-tier placement (without the file tier the budget checks refuse as before)
    } else {
        p.three_tier = true;
        p.staging = in.staging;
        if (cap < std::uint64_t{in.host_spares} + p.staging + 1U) {
            p.refuse = "not enough host memory even for the spare blocks and the file tier's staging buffers";
            p.pinned = 0;
        } else {
            p.pinned = cap - in.host_spares - p.staging;
            if (p.pinned > nonres) p.pinned = nonres;
        }
        p.disk = nonres - p.pinned;
        if (p.disk == 0) p.three_tier = false, p.staging = 0;
    }
    const std::uint64_t used = p.pinned + in.host_spares + p.staging;
    const std::uint64_t left = cap > used ? cap - used : 0U;
    std::uint64_t inc = 0;
    if (in.inclusive_gib < 0.0) inc = units_of(in.inclusive_auto_cap_gib);
    else if (in.inclusive_gib > 0.0) inc = units_of(in.inclusive_gib);
    if (inc > left) inc = left;
    if (in.host_spares == 0) inc = 0;  // the library keeps at least one spare block for write-backs
    p.inclusive = inc;
    p.spare_blocks = std::uint64_t{in.host_spares} + inc;
    p.vram_gib = static_cast<double>(p.resident) * unit;
    p.ram_gib = static_cast<double>(p.pinned) * unit;
    p.file_gib = static_cast<double>(p.disk) * unit;
    p.inclusive_gib = static_cast<double>(p.inclusive) * unit;
    p.need_host_gib = static_cast<double>(p.pinned + p.spare_blocks + p.staging) * unit + in.extra_gib + in.host_margin_gib;
    p.pinned_peak_gib = p.need_host_gib - in.host_margin_gib + in.engine_pinned_gib;
    return p;
}
}  // namespace strix::ds41::rtx
