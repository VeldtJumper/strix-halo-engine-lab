#pragma once

#include <cstdint>

namespace strix {

// Test-only direct production-kernel seam. Storage is caller-owned, padded to
// 64 rows; all arrays use the production 24 query / 2 KV heads, dimension 256.
struct HipCompactQsaProbe {
    const float* query = nullptr;
    const std::uint16_t* kv = nullptr;
    const std::uint32_t* selected = nullptr;
    float* scores = nullptr;
    void* image = nullptr;
    const void* weights = nullptr;
    std::uint16_t* projected = nullptr;
    // Two words: nonfinite status and FP16 finite-overflow count.
    std::uint32_t* status = nullptr;
    void* quantized = nullptr; // 8*6144 int8 codes, then 192*8 FP32 deltas (tail probe).
    std::uint32_t rows = 0, maximum_selected = 2052, count = 32;
    // 0 multi, 1 ordered, 2 occupancy, 3 shadow, 4 split/combine,
    // 5 rows, 6 grouped, 7 prefetch, 8 tail+quantizer, 9 status body,
    // 10 int8 fast, 11 int8 rows, 12 int8 prefetched group.
    unsigned mode = 0;
    bool compact = false;
    bool tiled_projection = true;
    bool scalar_projection = false, dense_a8_projection = false, fast_f16_projection = true;
};

void launch_hip_compact_qsa_probe(const HipCompactQsaProbe&, void* stream);

// Test-only seam for the verify-window value/gate kernels (24 query / 2 KV heads, dimension 256, BF16 KV):
// group 0 runs the per-head kernel (dense A8 when codes is set), 2|3|4|6|12 the window group kernel with each
// head's features over `split` blocks (1|2|4|8).
struct HipQsaWindowOutGroupProbe {
    const float* query = nullptr;
    const std::uint16_t* kv = nullptr;
    const std::uint32_t* selected = nullptr;
    float* scores = nullptr;
    float* image = nullptr;
    std::int8_t* codes = nullptr;
    float* deltas = nullptr;
    std::uint32_t rows = 8, group = 0, split = 1, maximum_selected = 2052, count[8]{};
};

void launch_hip_qsa_window_out_group_probe(const HipQsaWindowOutGroupProbe&, void* stream);

} // namespace strix
