#pragma once
// The most file reads in flight tierstream's auto queue can pick (dg_auto_disk_queue in dg_layout.cpp):
// clamp(3 x read locations, 4, 8), at most the staging buffers. The locations are the unit file plus each
// --weight-dirs entry, an upper bound (the library counts only directories that hold the stripe manifest),
// so the readers billed for it are never fewer than the library starts. Shared by the planner and the binding.
#include <algorithm>
#include <string>

namespace strix::glm::rtxpf::producer {
inline unsigned auto_disk_queue_bound(const std::string& weight_dirs,unsigned staging){
    unsigned locations=1;
    if(!weight_dirs.empty()){locations=2;for(char ch:weight_dirs)if(ch==';')++locations;}
    return std::min(std::min(8u,std::max(4u,3*locations)),staging);
}
}
