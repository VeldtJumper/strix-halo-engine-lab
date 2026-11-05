#pragma once
#include "strix/ple_reclaim.hpp"
#include <stdexcept>

namespace strix::decode_cli {
// Pre-device verification and post-registration setup must open the same backing
// and retain its mapping authority. Neither an address nor a predicted plan is
// sufficient to construct a rolling reader (including bounded MTP turn leases).
template<class Mapping>
void bind_ple_reclaim_mapping(ple_reclaim::Config& config,
    std::shared_ptr<const Mapping> mapping) {
    if (!mapping) throw std::invalid_argument("PLE reclaim requires a live mapping");
    config.mapping_base = mapping->base_address();
    config.mapping_bytes = mapping->bytes();
    const auto backing = mapping->file_backing_identity();
    config.file_device = backing[0];
    config.file_inode = backing[1];
    config.backing_identity = std::to_string(backing[0]) + ":" +
        std::to_string(backing[1]) + ":" + std::to_string(config.table_offset);
    config.backing_owner = mapping;
    config.mapping_generation = mapping->allocation_generation();
    config.open_direct_read_only = [mapping] { return mapping->open_direct_read_only(); };
    config.drop_page_cache = [mapping](auto offset, auto bytes) {
        return mapping->drop_page_cache(offset, bytes);
    };
}
} // namespace strix::decode_cli
