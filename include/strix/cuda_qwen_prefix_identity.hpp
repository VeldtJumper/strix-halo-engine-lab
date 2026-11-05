#pragma once

#include "strix/cuda_qwen_prefill.hpp"
#include "strix/model_content_hash.hpp"

#include <string>
#include <utility>
#include <vector>

namespace strix::cuda::qwen_prefix {

using IdentityFields = std::vector<std::pair<std::string, std::string>>;
// Each name and value has a decimal byte length followed by ':'; delimiters inside
// paths or values cannot alias field boundaries. Field order is part of the contract.
std::string canonical_identity(const IdentityFields& fields);
ModelContentIdentity identity_key(const IdentityFields& fields);
IdentityFields identity_fields(const qwen_prefill::Config& config, std::uint32_t block);
std::uint64_t default_host_budget();

}  // namespace strix::cuda::qwen_prefix
