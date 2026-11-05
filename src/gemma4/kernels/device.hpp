#pragma once

#include "gemma4/ref/target.hpp"
#include "gemma4/kv/storage.hpp"
#include <map>

namespace strix::gemma4::gpu {
struct DeviceOptions {
    bool shared_weights = true, fusion = true, profile = false;
    bool graphs = true;
    bool graph_timing = false;
    // Numerical identity name ("g4-v1" or the opt-in "g4-a8-v1"); null: STRIX_G4_IDENTITY, else g4-v1.
    const char* identity = nullptr;
    bool a8_ffn_ple_fuse = true; // g4-a8-v1 only; STRIX_G4_A8_FFN_PLE_FUSE=0 switches it off
};
struct DeviceStats {
    std::size_t resident_weight_bytes = 0, pass_weight_bytes = 0;
    bool shared_weights = false, fusion = false, graphs = false;
    bool norm_group = false, chunk_decode = false, norm_search = false;
    bool kv_pool = false;
    bool output_tile = false;
    bool kv_publish = false;
    bool linear_finite = false;
    bool norm_prefetch = false;
    bool norm_finite = false;
    bool pinned_io = false;
    bool norm_bounds = false;
    bool chunk_prefetch = false;
    bool chunk_prefetch_bf16_only = false;
    bool packed_header = false;
    bool streaming_weights = false;
    bool norm_stage = false;
    bool head_registers = false;
    bool input_cache = false;
    bool attention_prefetch = false;
    bool norm_intervals = false;
    bool one_chunk_attention = false;
    bool streaming_weights_auto = false;
    bool streaming_weights_bf16_only = false;
    bool layer_norm_fusion = false;
    bool fused_attention = false;
    bool store_fusion = false;
    bool multi_linear = false;
    bool a8 = false;
    bool a8_ffn_ple_fuse = false;
    // Cumulative successful eager/captured fused layer launches; cached graph replays are excluded.
    std::size_t a8_ffn_ple_launches = 0;
    std::string identity = arithmetic_id;
    double gpu_pass_ms = 0.0;
    std::map<std::string, double> kernel_ms;
};
class DeviceTarget final : public Target {
  public:
    // Copies packed weights and immutable tables once into hipMalloc allocations.
    // Host mappings can be released after construction. No managed/pageable demand migration.
    DeviceTarget(const HostWeights& weights, RopeTables tables, DeviceOptions options = {});
    ~DeviceTarget() override;
    ForwardResult forward_rows(std::span<const RowPlan> rows) override;
    [[nodiscard]] const DeviceStats& stats() const;
    [[nodiscard]] const std::vector<Candidate>& selected() const;
    // Read-only original packed bytes used by both target embedding and head.
    // The view is valid until this target is destroyed; no rounding/repacking.
    [[nodiscard]] LinearView resident_embedding() const;
    [[nodiscard]] LinearView resident_head() const;
    // Borrow for a prefill backend accepting device residency. This target
    // owns every view and must outlive that borrower; never read on the host.
    [[nodiscard]] const HostWeights& borrow_device_weights() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix::gemma4::gpu
