#pragma once
// The most file reads in flight tierstream's auto queue can pick (dg_auto_disk_queue in dg_layout.cpp):
// clamp(3 x read locations, 4, 8), at most the staging buffers. The locations are the unit file plus each
// --weight-dirs entry, an upper bound (the library counts only directories that hold the stripe manifest),
// so the readers billed for it are never fewer than the library starts. Shared by the planner and the binding.
#include <algorithm>
#include <cstdint>
#include <string>

namespace strix::glm::rtxpf::producer {
inline constexpr unsigned disk_queue_regime=~0u; // unset; 0 remains explicit auto
inline std::string disk_queue_requested(unsigned asked){return asked==disk_queue_regime?"unset: regime":asked?std::to_string(asked):"auto";}
inline unsigned resolved_disk_queue(unsigned asked,std::uint64_t billed_readers){return asked&&asked!=disk_queue_regime?asked:billed_readers>1?0u:1u;}
// Auto was never slower in measured cell means at 76 % file share; 61 % has a slower
// cold-1K cell. Full card (~1-4 % at tile 2048) ties. The cut sits between 61 % and 76 %.
inline constexpr unsigned disk_queue_regime_min_file_percent=70;
inline bool disk_queue_regime_admits(std::uint64_t file_units,std::uint64_t keys){return keys&&file_units*100>=keys*disk_queue_regime_min_file_percent;}
inline unsigned auto_disk_queue_bound(const std::string& weight_dirs,unsigned staging){
    unsigned locations=1;
    if(!weight_dirs.empty()){locations=2;for(char ch:weight_dirs)if(ch==';')++locations;}
    return std::min(std::min(8u,std::max(4u,3*locations)),staging);
}
}
