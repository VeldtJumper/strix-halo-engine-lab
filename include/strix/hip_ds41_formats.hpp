#pragma once

// Device side of the deepseek41 formats (gfx1151, wave32): block decoders for fixtures, the A8
// activation quantizer, scalar format tables and the decode products of routed-expert rows. Every
// result is defined by the host reference in ds41_formats.hpp, operation by operation.

#include "strix/ds41_formats.hpp"

namespace strix::dist { struct ProducerStage; struct ConsumerStage; }

#include <cstddef>
#include <cstdint>

namespace strix::ds41 {

inline constexpr std::uint32_t product_max_rows = 8;

// Kernel forms of the decode MoE and head launches, for A/B; every form gives the same bits. 0 is the
// original form, -1 the default. Read once from STRIX_DS41_MOE_ROUTED / _SHARED / _HEAD / _PREFILL; fixtures may
// change the returned object between launches.
struct MoeKernelTuning {
    int routed = -1;  // routed-expert gate/up and down (routes built on the device)
    int shared = -1;  // shared expert (Q8_0, one expert of R rows)
    int head = -1;    // output head (Q8_0 x A8 over the vocabulary slice)
    int prefill = -1; // prefill experts (WMMA kernels of hip_ds41_prefill.hip)
    // Routed experts with routes built on the device (STRIX_DS41_EXPERT_SLOTS, default 1 since P16, =0 off): the grids
    // cover slots_y(R) union slots instead of rows * top_k and each block loops over slots y, y + slots_y, ...
    // below counts[0] (fewer padded blocks; every slot's products unchanged). STRIX_DS41_EXPERT_SLOTS_Y=N fixes it.
    int slots = 1;
    int slots_y = 0;
    // Routed IQ2_XXS gate/up and Q2_K down with every weight group of a wave loaded at once and held in registers
    // across the staged-row parts (STRIX_DS41_EXPERT_ROWS, default 1 = on, =0 off; same bits). STRIX_DS41_EXPERT_ROWS_FORM
    // overrides the shape for A/B (hip_ds41_formats.hip, rows_shape).
    int rows = 1;
    int rows_form = -1;
    // The expert-TP down (enqueue_routes_product_groups at width 1152) with held groups and its four tail groups spread
    // over the lanes (STRIX_DS41_TP_DOWN, default 1 since P17.2, =0 off; same bits). STRIX_DS41_TP_DOWN_FORM overrides the
    // shape.
    int tp_down = 1;
    int tp_down_form = -1;
};
MoeKernelTuning& moe_kernel_tuning();

// Decode block_count consecutive blocks into FP32 (block_elements(format) values each).
void enqueue_decode_blocks(WeightFormat format, const void* blocks, std::size_t block_count, float* out, void* stream);
// Decode row_count Engram rows (256 values each).
void enqueue_decode_engram_rows(const void* rows, std::size_t row_count, float* out, void* stream);
// All 256 codes of E4M3, E8M0 and E8M0/2, then the 16 E2M1 nibbles: 784 floats.
void enqueue_decode_scalar_tables(float* out, void* stream);
// A8 quantization of rows x width values (width a multiple of 32), as quantize_a8 per row.
void enqueue_quantize_a8(const float* values, std::uint32_t width, std::uint32_t rows, std::int8_t* codes, float* deltas,
                         std::int32_t* sums, void* stream);

// out[(r * expert_count + e) * outputs + o] = row o of expert experts[e] times activation row r,
// as product_iq2_xxs / product_q2_k / product_mxfp4. Activation row r: codes + r * width,
// deltas + r * width / 32; rows = 1..8 (every row on every expert).
struct ProductView {
    const void* weights = nullptr;
    std::uint64_t expert_stride_bytes = 0;
    std::uint64_t row_bytes = 0;
    std::uint32_t width = 0;
    std::uint32_t outputs = 0;
    const std::uint32_t* experts = nullptr;
    std::uint32_t expert_count = 0;
    const std::int8_t* codes = nullptr;
    const float* deltas = nullptr;
    std::uint32_t rows = 1;
    float* out = nullptr;
    std::uint32_t table[4] = {0, 0, 0, 0};  // WeightFormat::cb4: the 16-entry int8 table
};
void enqueue_product(WeightFormat format, const ProductView& view, void* stream);

// Window routes in device memory (WindowRoutes): expert slot j = blockIdx.y owns entries
// begins[j] .. begins[j + 1]; entry i reads activation row inputs[i] and writes output row outputs[i].
// max_rows bounds the entries of any expert (1..16), or with device routes the entries staged at a time.
struct RoutesView {
    const std::uint32_t* experts = nullptr;
    const std::uint32_t* begins = nullptr;
    const std::uint32_t* inputs = nullptr;
    const std::uint32_t* outputs = nullptr;
    std::uint32_t expert_count = 0;
    std::uint32_t max_rows = 0;
    // Routes built on the device: counts[0] is the expert count (expert_count is then an upper bound
    // used for the launch, max_rows the entries staged at a time; experts with more are done in parts).
    const std::uint32_t* counts = nullptr;
};
// out[outputs[i] * outputs_per_row + o] = row o of expert experts[j] times activation row inputs[i]
// (the declared product), e.g. the down projection writing every (row, slot)'s unweighted vector.
struct RoutesProductView {
    const void* weights = nullptr;
    std::uint64_t expert_stride_bytes = 0;
    std::uint64_t row_bytes = 0;
    std::uint32_t width = 0;
    std::uint32_t outputs = 0;
    RoutesView routes;
    const std::int8_t* codes = nullptr;
    const float* deltas = nullptr;
    float* out = nullptr;
    std::uint32_t table[4] = {0, 0, 0, 0};  // WeightFormat::cb4: the 16-entry int8 table
};
void enqueue_routes_product(WeightFormat format, const RoutesProductView& view, void* stream);
// The routed Q2_K down over a part of the K dimension (expert tensor parallelism, ds41_expert_tp.hpp): the same kernel and
// per-lane order as enqueue_routes_product, width any multiple of 32 (groups of the stored row in order) and row_bytes
// the whole Q2_K blocks that hold them. half_tail: the row is width / 256 whole blocks then a packed half block of 44 bytes
// (scales 8, codes 32, d, dmin) holding the last 4 groups (width % 256 == 128); the same group values and order.
void enqueue_routes_product_groups(WeightFormat format, const RoutesProductView& view, void* stream, bool half_tail = false);
// Fused gate/up (IQ2_XXS or MXFP4) with the clamped SwiGLU and A8 quantization of every (row, slot):
// act_codes[slot * inter], act_deltas/act_sums[slot * inter / 32]; gate_out/up_out (optional, both
// or neither) receive the FP32 gate and up products.
struct GateUpView {
    const void* gate = nullptr;
    const void* up = nullptr;
    std::uint64_t expert_stride_bytes = 0;
    std::uint64_t row_bytes = 0;
    std::uint32_t hidden = 0;
    std::uint32_t inter = 0;
    RoutesView routes;
    const std::int8_t* codes = nullptr;
    const float* deltas = nullptr;
    float limit = 10.0F;
    float* gate_out = nullptr;
    float* up_out = nullptr;
    std::int8_t* act_codes = nullptr;
    float* act_deltas = nullptr;
    std::int32_t* act_sums = nullptr;
    std::uint32_t table[4] = {0, 0, 0, 0};  // WeightFormat::cb4: the 16-entry int8 table
};
void enqueue_gate_up(WeightFormat format, const GateUpView& view, void* stream);
// y[row] = (shared + low) + high over the row's slots in `order` (ascending expert id), the first
// low_counts[row] of them being the low half (ds41_moe.hpp).
struct CombineView {
    const float* raw = nullptr;
    const float* shared = nullptr;
    const float* weights = nullptr;
    const std::uint32_t* order = nullptr;
    const std::uint32_t* low_counts = nullptr;
    std::uint32_t rows = 0;
    std::uint32_t top_k = 0;
    std::uint32_t hidden = 0;
    float* out = nullptr;
};
void enqueue_combine(const CombineView& view, void* stream);

// Router of a window (reference_router): scores [rows][experts], ids and weights [rows][top_k].
// weight is [experts][hidden] FP32; image_rows (optional) selects bias_image per row.
struct RouterView {
    const float* x = nullptr;
    const float* weight = nullptr;
    // Optional BF16 copy of `weight` (used instead when set): only for routers whose F32 values are all BF16 values,
    // so widening gives the same F32 weights and the same bits.
    const std::uint16_t* weight_bf16 = nullptr;
    const float* bias = nullptr;
    const float* bias_image = nullptr;
    const std::uint8_t* image_rows = nullptr;
    std::uint32_t rows = 0;
    std::uint32_t hidden = 0;
    std::uint32_t experts = 0;
    std::uint32_t top_k = 0;
    float route_scale = 1.0F;
    // true: the block stages the rows of x in LDS (default); false: the previous one-wave-per-expert
    // form reading x from global memory, kept for A/B. Same order and bits.
    bool staged = true;
    // true (or STRIX_DS41_ROUTER_ROWS=1): one wave per (expert, row), loads 16 elements ahead per lane, no LDS and no
    // barriers (hidden % 512 == 0, else the forms above). Same per-lane order and butterfly, same bits.
    bool row_waves = false;
    // Prefill router (enqueue_prefill_router): x and the weights staged into LDS with 16-byte loads, several in flight
    // (default; needs 16-byte aligned x / weight and hidden % 4 == 0, else the element loop; STRIX_DS41_PREFILL_ROUTER_WIDE=0
    // keeps the element loop). The same staged values, same bits. With 16-byte staging the v2 scores kernel runs (32 rows
    // x 32 experts per block, next chunk loaded during the current one; STRIX_DS41_PREFILL_ROUTER_V2=0 keeps v1), same bits.
    bool wide_stage = true;
    // STRIX_DS41_ROUTER_FAST=1 (default off) or fast = 1: the row-wave scores, then the selection by one wave per row in
    // the select block's comparison tree. fast = 4 / 8 / 16 (A/B; slower at every row count measured): wide scores, one
    // wave per expert over every row with 16-byte BF16 loads through LDS (BF16 copy, hidden % 256 == 0; other values
    // above 1: 8 waves). Same scores, ids and weights.
    std::uint32_t fast = 0;
    // Optional zeroed device word for the fast form: the scores kernel's last block then selects (no second launch)
    // and leaves it zero. One call in flight per word.
    std::uint32_t* tickets = nullptr;
    float* scores = nullptr;
    std::uint32_t* ids = nullptr;
    float* weights = nullptr;
};
void enqueue_router(const RouterView& view, void* stream);
// out[row] = the half-`half` sum of reference_half_sum (rows of hidden values).
struct HalfSumView {
    const float* raw = nullptr;
    const float* weights = nullptr;
    const std::uint32_t* order = nullptr;
    const std::uint32_t* low_counts = nullptr;
    std::uint32_t rows = 0;
    std::uint32_t top_k = 0;
    std::uint32_t hidden = 0;
    std::uint32_t half = 0;
    float* out = nullptr;
};
void enqueue_half_sum(const HalfSumView& view, void* stream);
// out = (shared + low) + high elementwise: the combine from the two half sums.
void enqueue_canonical_sum(const float* shared, const float* low, const float* high, std::size_t count, float* out, void* stream);
// Ready forms of the MoE half-sum exchange (STRIX_DS41_MOE_READY=1, staged_ready.hpp lean releases): the half sum
// written into the coherent TX view, every block publishing it; the combine reading the coherent RX view, every
// block releasing it (reads only: out stays GPU-owned on the same stream). Same arithmetic as the two above.
void enqueue_half_sum_ready(const HalfSumView& view, dist::ProducerStage stage, void* stream);
void enqueue_canonical_sum_ready(const float* shared, const float* low, const float* high, std::size_t count, float* out,
                                 dist::ConsumerStage peer, void* stream);
// The same combine with the shared expert's output split by rows over two ranks: row r of shared is
// shared_lo[r * hidden / 2 ...] then shared_hi[r * hidden / 2 ...]. shared_out (optional) receives the
// assembled [rows][hidden] shared output.
void enqueue_canonical_sum_split(const float* shared_lo, const float* shared_hi, const float* low, const float* high,
                                 std::uint32_t rows, std::uint32_t hidden, float* out, float* shared_out, void* stream);
// A8 rows of width 2 * half from two halves ([rows][half] codes, [rows][half / 32] deltas each; half a
// multiple of 32): first holds columns [0, half), second [half, 2 * half).
void enqueue_assemble_a8_halves(const std::int8_t* first_codes, const float* first_deltas, const std::int8_t* second_codes,
                                const float* second_deltas, std::uint32_t rows, std::uint32_t half, std::int8_t* codes,
                                float* deltas, void* stream);

// Window routes from device ids (rows * top_k <= 64), as build_window_routes restricted to the experts in
// [first, end): experts/begins (union + 1 entries), rows_out/slots, order/low_counts for every row, and
// counts[0] = union experts, counts[1] = most entries of one expert. Arrays sized for rows * top_k.
struct BuildRoutesView {
    const std::uint32_t* ids = nullptr;
    std::uint32_t rows = 0;
    std::uint32_t top_k = 0;
    std::uint32_t split = 0;
    std::uint32_t first = 0;
    std::uint32_t end = 0;
    // experts[] holds id - base (base = first lets a rank that keeps only its expert slice resident index
    // its own weights); ids, order and low_counts stay global.
    std::uint32_t base = 0;
    std::uint32_t* experts = nullptr;
    std::uint32_t* begins = nullptr;
    std::uint32_t* rows_out = nullptr;
    std::uint32_t* slots = nullptr;
    std::uint32_t* order = nullptr;
    std::uint32_t* low_counts = nullptr;
    std::uint32_t* counts = nullptr;
};
void enqueue_build_routes(const BuildRoutesView& view, void* stream);
// Select routes and build the same integer CSR in one block after the ordinary router scores.
// Returns false without enqueueing for unsupported shapes or an explicitly selected fast router.
// The caller owns the opt-in flag and skips enqueue_build_routes only when this returns true.
[[nodiscard]] bool enqueue_router_routes(const RouterView& router, const BuildRoutesView& routes, void* stream);

} // namespace strix::ds41
