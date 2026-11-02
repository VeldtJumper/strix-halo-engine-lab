#pragma once
// Concrete TierCacheApi over the shared tierstream device-gated library
// (tierstream, include/tierstream/ts.h), plus the shared budget
// planner and allocation ledger (ss_budget_plan / ss_budget_ledger). Only tierstream_binding.cpp sees tierstream types.
// Open: desc-v1 of the GLM unit file (one x-glm-exl3-unit-v1 part, gated
// ordinals 0..42), ts_plan_from_tiers with per-key tiers (RESIDENT / PINNED /
// DISK), EXCLUSIVE tier mode, device-gated forward, dg_disk_tier when any unit
// is file-homed. Per token: ts_begin, per layer export + wait (+ guarded-slot
// descriptor bind in the runtime), enqueue_end after the last gated layer,
// ts_dg_end_rows(1). Victims: VRAM -> pinned RAM -> drop (file-homed units).
#include "expert_tier.hpp"
#include <memory>
#include <string>
#include <vector>
namespace strix::glm::rtxpf::producer {
struct TierOpenConfig {
    std::string descriptor_json;     // tierstream_descriptor(catalog, unit_file, model)
    std::vector<std::uint8_t> tiers; // per key, ss_tier byte, layer-major
    TierBudget budget;               // tier_budget(): slots, residents, pinned, disk, spares, staging
    std::uint64_t unit_bytes=0;      // == unit stride (one opaque part)
    std::uint32_t max_ids=routed_topk,disk_prefetch=0;
    std::uint32_t expected_gated=43,expected_units=288;
    bool cuda=true;                  // false: CPU backend admission probe only
    // Planner billing: the arena's spares are ONE global count (dg_spare_slots,
    // budgeted in TierBudget); per-layer dynamic billing stays 0. Nonzero only
    // in the regression test reproducing the per-layer over-billing refusal.
    std::uint32_t dynamic_per_layer=0;
    // Whole-device budgets: the library charges its own device/pinned allocations
    // to this ledger by name (ss_options.budget); nullptr = not charged.
    BudgetLedger* ledger=nullptr;
    // Bound open: the identity the descriptor's source must have (the unit file
    // authenticated by our held handle: volume serial, file index, size); the
    // library opens it through held ancestor handles and refuses any other file
    // (ss_options.source_identities). All zero = not checked.
    std::uint64_t source_volume=0,source_index=0,source_size=0;
    // Placement, the DS way: the priority list (expert_tier.hpp seed_order) becomes
    // ss_options.dg_seed (which units take the device slots, initial recency); the
    // tier array above must put its first `resident` keys on the device. Empty: none.
    std::vector<std::uint32_t> seed;
    // Boundary victims: 0 LRU, 1 SCORED, 2 SCORED_AUTO (DS default; online learning,
    // no profile file; scored victims for multi-row windows, LRU tail for one row).
    int policy=2;
    // ss_options.dg_flush_ready (DS TierSite flush_ready): flush the demand stream
    // after each ready signal (WDDM submission; outputs unchanged).
    bool flush_ready=true;
    // ss_options.weight_dirs / drive_readers: stripe copies of the unit file on other drives
    // (placement of reads only; every chunk is checked against the stripe manifest). Empty: primary only.
    std::string weight_dirs;std::uint32_t drive_readers=0;
    // ss_options.dg_disk_queue (1..64, <= staging) / dg_clean_evict; three-tier opens only.
    std::uint32_t disk_queue=1;bool clean_evict=false;
};
// The tierstream commit this build compiled (CMake pin / snapshot STAMP), for health.
const char* tierstream_commit();
// Whether the linked library has the reader/resident API (make_resident, kept
// units resolving in any step, single-layer steps): prefill then reads experts
// through the cache; otherwise tierstream mode keeps the interim prefill rings.
bool tierstream_supports_make_resident();
// Library budget derived from our plan: arena + unit table + control slack on
// the device; pinned home blocks, spare blocks, file staging, bounce and the
// startup load ring on the host. Margins are already applied by the caller.
struct TierLibraryBudget {std::uint64_t device_bytes=0,host_bytes=0;std::uint32_t ring_slots=0;};
inline TierLibraryBudget library_budget(const TierBudget& b,std::uint64_t unit,std::uint64_t keys,std::uint32_t max_ids){
    TierLibraryBudget l;const std::uint64_t host_stride=align_up(unit,4096),slack=64ULL<<20;
    l.ring_slots=2*max_ids+1;l.device_bytes=(b.slots+2ULL*max_ids+1)*unit+keys*4+slack;
    // +1 staging unit, +4 startup load buffers, + one bounce per disk reader (dg_disk_queue > 1).
    l.host_bytes=(b.pinned+b.host_spares+b.staging+1+4+b.disk_readers)*host_stride+(16ULL<<20)+slack;return l;
}
std::unique_ptr<TierCacheApi> open_tierstream(const TierOpenConfig&);
// CPU backend: runs every library admission rule (descriptor, plan, budgets,
// options, dg layout) and stops at "needs the CUDA backend". Returns
// {status, message}; status 8 (SS_E_UNSUPPORTED) means everything admitted.
std::pair<int,std::string> tierstream_admission_probe(const TierOpenConfig&);
}
