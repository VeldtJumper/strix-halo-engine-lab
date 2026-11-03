#pragma once
// Grouped chunk MoE preserving moe::enqueue_experts R=1 arithmetic.
#include "glm/moe/glm_moe.hpp"
#include <vector>

namespace strix::glm::pf::exact_moe {
// Pair index = row * 9 + slot (0..7 routed, 8 shared). No padded pair stores.
struct Tile { unsigned expert, first, count; }; // expert 288 is shared
struct Plan {
    explicit Plan(unsigned max_rows);
    unsigned capacity;
    std::vector<unsigned> pairs;
    std::vector<Tile> tiles;
};
// Includes device expert table, plan, gate/up, transformed down input, outputs,
// expert group ranges, and V2/V3 matrix-specific transformed gate/up inputs (9 * rows * 2 * 4096 * 2 bytes).
// No initialization required. Calls using the same plan/workspace must be ordered.
std::size_t workspace_bytes(unsigned max_rows);
struct View {
    unsigned rows = 0;
    const moe::ExpertSlices* experts = nullptr; // host [288], raw device parts
    moe::ExpertSlices shared{};
    const float* x = nullptr;
    const std::uint32_t* ids = nullptr; // host [rows][8], strictly ascending
    const float* host_weights = nullptr; // host [rows][8], optional (not consumed)
    const float* weights = nullptr; // device [rows][8]
    float* out = nullptr;
};
// STRIX_GLM_PF_EXACTMOE_V: 1, 2 (cached inputs), 3 (default, dense operand schedule,
// one CTA per expert/output block looping over TM16 row tiles).
// V3 linear kernels always use TM16; TM still controls transform/activation work tiles.
// V6 opts into X (32-row linear tiles); V7 selects X2 with register prefetch; V8/V9 the X3 core (scalar activations,
// 32/64-row tiles), V10 X3 with the tile chosen per expert group, V11 X3 with whole 64-row tiles plus 1/2/4-row
// tail buckets, V14 = V11 with each group's remainder above 32 rows run as one padded 64-row tile (one weight decode
// instead of two 32-row passes); auto = V14 for calls >= x_auto_rows(), v3 below.
hipFuncAttributes x_resources(unsigned bits, bool down, bool shared, unsigned variant = 6);
// TM 0 reads STRIX_GLM_PF_EXACTMOE_TM (16 default; 16 or 32).
// Plan/table uploads finish before return; kernels remain asynchronous on stream.
// No device or host allocations during enqueue after Plan construction.
unsigned x_auto_rows();  // STRIX_GLM_PF_EXACTMOE_V=auto threshold: STRIX_GLM_PF_EXACTMOE_X_MIN_ROWS, default 128
bool v3_row_buckets();  // STRIX_GLM_PF_EXACTMOE_V3_RCAP (default 1): v3 tiles use the smallest row bucket
unsigned selected_variant(unsigned rows);  // what enqueue runs for a call of `rows` rows (1, 2, 3, 6..11 or 14)
void enqueue(const View& v, Plan& plan, void* workspace, std::size_t bytes,
             hipStream_t stream, unsigned tm = 0);
// Device plan: the same pairs/tiles/groups bytes as enqueue() built on the GPU from device ids, without the route
// download, host grouping or plan upload (no host synchronization). table: device [289] (entry 288 = shared expert).
// Invalid ids (not strictly ascending or >= 288) set *flag and that row gets no routed pairs.
struct DeviceView {
    unsigned rows = 0;
    const moe::ExpertSlices* table = nullptr;  // device
    moe::ExpertSlices shared{};                 // host copy of table[288] (its bits select the shared kernels)
    const float* x = nullptr;
    const std::uint32_t* ids = nullptr;        // device [rows][8]
    const float* weights = nullptr;            // device [rows][8]
    float* out = nullptr;
};
unsigned tm_env();  // STRIX_GLM_PF_EXACTMOE_TM (16 default, 32)
moe::ExpertSlices* workspace_table(void* workspace);  // the workspace's [289] device table slot
// Plan buffers inside a workspace sized for max_rows (diagnostics: compare device and host plans).
const unsigned* plan_pairs(void* workspace, unsigned max_rows);
const Tile* plan_tiles(void* workspace, unsigned max_rows);
const Tile* plan_groups(void* workspace, unsigned max_rows);  // [289]
// Whether a call of `rows` rows takes the device plan: grouped variants with TM16; STRIX_GLM_PF_EXACTMOE_PLAN=host
// (default device) keeps the host plan.
bool device_plan(unsigned rows, unsigned tm = 0);
void enqueue_device(const DeviceView& v, Plan& plan, void* workspace, std::size_t bytes, hipStream_t stream, unsigned* flag);
const std::uint16_t* pair_down_input(void* workspace, unsigned max_rows, unsigned pair);
const float* pair_gate_up(void* workspace, unsigned max_rows, unsigned pair); // [2][1024]
const std::uint16_t* pair_gate_up_input(void* workspace, unsigned max_rows, unsigned pair); // [2][4096], V2+
const float* pair_output(void* workspace, unsigned max_rows, unsigned pair);
} // namespace strix::glm::pf::exact_moe
