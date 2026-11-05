#pragma once
#include "qwen_effective_config.hpp"
#include <cstdio>

namespace strix::decode_cli {
// Retire the preceding admission trial before even probing the next route.
// Assignment alone evaluates construction while the old authority is still live.
template<class Owner, class Verify>
auto verify_fresh_ple_preflight(Owner& owner, Verify verify) {
    owner.reset();
    return verify();
}
// Setup has not mutated canonical state. Unwind every provisional rolling owner
// before observing memory or touching the resident mapping. Required mode never
// substitutes a different route. Runtime read failures do not enter this seam.
template<class Setup, class Teardown, class Admit, class Warm>
std::shared_ptr<ple_reclaim::Prefetcher> setup_ple_or_resident(ple_reclaim::Mode mode,
    Setup setup, Teardown teardown, Admit admit, Warm warm) {
    try { return setup(); }
    catch (const ple_reclaim::DiscardSafetyError&) { throw; } // Never substitute after a safety refusal.
    catch (const std::exception& error) {
        if (mode == ple_reclaim::Mode::required)
            throw std::runtime_error(std::string("required PLE setup refused before prompt mutation: ") + error.what());
        teardown();
        std::fprintf(stderr,"ple-rolling: setup failed (%s); dropped all credit; checking resident envelope\n",error.what());
        admit(); // May refuse; never warm or start consumers before this passes.
        warm();
        std::fprintf(stderr,"ple-rolling: resident fallback admitted; MTP policy unchanged; credit=0\n");
        return {};
    }
}
inline std::map<std::string,std::uint64_t> resident_prefill_admission(const Options& options,
    const PrefillRequirements& required, const PrefillMemory& memory) {
    const auto refuse=[] { throw std::runtime_error("PLE resident fallback refused before prompt mutation: zero-credit owner envelope exceeds observed memory/workspace/metadata/GTT limits"); };
    if (!required.owner_error.empty()) throw std::runtime_error("PLE resident fallback refused before prompt mutation: " + required.owner_error);
    if (!memory.total || !memory.available || !memory.gtt_limit || !memory.gtt_used || required.owners.empty()) refuse();
    auto owners=required.owners;
    owners["rotary_table"]=options.rotary.table_bytes();
    const auto peaks=prefill_owner_phase_peaks(owners,required.owner_phases,options.known_prefill_host_headroom);
    const auto gross=*std::max_element(peaks.begin(),peaks.end());
    if (gross<required.model_backing) refuse();
    const auto incremental=gross-required.model_backing;
    const auto device=prefill_checked_add(required.device_base,
        prefill_checked_add(required.workspace,prefill_checked_add(required.tails,options.rotary.table_bytes())));
    const auto limit=std::min(*memory.total,options.known_prefill_uma_budget);
    if (!prefill_physical_margin_fits(*memory.total,*memory.available,gross,incremental) ||
        gross>limit || incremental>*memory.available || *memory.gtt_used>*memory.gtt_limit ||
        device>*memory.gtt_limit-*memory.gtt_used || !required.workspace ||
        prefill_checked_add(required.workspace,required.tails)>options.known_prefill_workspace_budget ||
        required.metadata>options.known_prefill_metadata_budget) refuse();
    return {{"gross_required_bytes",gross},{"required_bytes",gross},{"incremental_bytes",incremental},
        {"device_growth_bytes",device},{"verified_ple_credit_bytes",0},{"resident_fallback",1}};
}
} // namespace strix::decode_cli
