#pragma once

namespace strix::dist { struct ProducerStage; }

// Device side of the deepseek41 layer sites (gfx1151, wave32). Every kernel reproduces the host
// reference of ds41_layer.hpp bitwise; rows are independent tokens of a decode window or a prefill
// block. All pointers are device pointers; every call only enqueues on `stream`.
//
// Aliasing: outputs must not overlap inputs unless a call says otherwise (the kernels declare their
// pointers __restrict__). In place is allowed for enqueue_rms_norm / enqueue_norm with x == out, the
// Engram injection and RoPE, as documented at each call.

#include "strix/ds41_layer.hpp"
#include "strix/ds41_prefill.hpp"
#include "strix/hip_ds41_prefill.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace strix::ds41 {

// One HcMix (24 floats: pre, post, comb) per row from rows x (hc * hidden) streams.
struct HcMixView {
    const float* stream = nullptr;
    const std::uint16_t* fn = nullptr;  // F16 [24][hc * hidden]
    const float* base = nullptr;        // [24]
    const float* scale = nullptr;       // [3]
    std::uint32_t rows = 0, hidden = 0, iters = 20;
    float hc_eps = 1e-6F, norm_eps = 1e-20F;
    HcMix* out = nullptr;
    float* partial = nullptr;           // scratch: hc_mix_partial_floats(rows, hidden)
};
// Two launches: chunk partials of the 24 dots and the sum of squares spread over (chunk, row) blocks,
// then per row the ascending chunk sums and the split + Sinkhorn.
[[nodiscard]] std::size_t hc_mix_partial_floats(std::uint32_t rows, std::uint32_t hidden);
void enqueue_hc_mixes(const HcMixView& view, void* stream);
// out[r][i] = sum_c pre[r][c] * stream[r][c][i]; pre from mixes[r].pre (or the identity when mixes is null).
void enqueue_hc_collapse(const float* hc_stream, const HcMix* mixes, std::uint32_t rows, std::uint32_t hidden, float* out,
                         void* stream);
// out[r][k][i] = post[k] * sub[r][i] + sum_j comb[j][k] * residual[r][j][i] (out may not alias residual).
void enqueue_hc_expand(const float* sub, const float* residual, const HcMix* mixes, std::uint32_t rows, std::uint32_t hidden,
                       float* out, void* stream);

// One launch per hyper-connection site boundary (the decode chain between two sublayers), bitwise the
// composition of the separate calls:
//  1. stream: the expansion of the previous sublayer, expand(sub, residual, expand_mixes), written to
//     `stream` (when sub is null, `stream` is read as it is);
//  2. this site's mixes (when fn is not null): HcMix rows into mixes_out, as enqueue_hc_mixes;
//  3. the sublayer input (when norm_weight is not null): collapse of the stream with collapse_mixes[r].pre
//     (identity when null), RMSNorm, FP32 to out and/or A8 to codes/deltas/sums, as enqueue_norm.
// Mix-partial blocks (chunk, row) and norm blocks (chunk, row) run side by side; the last partial block
// of a row to finish (an atomic counter per row) runs its split + Sinkhorn after an agent-scope acquire
// (the writers release with a fence before their increment). hidden a multiple of 640. counters: hc_site_counter_count(rows) values, zero
// before the first launch (hipMemset once); every launch leaves them zero. partial: hc_mix_partial_floats(rows, hidden). mixes_out must differ from
// expand_mixes and collapse_mixes; stream must not overlap residual or sub.
struct HcSiteView {
    std::uint32_t rows = 0, hidden = 0;
    const float* sub = nullptr;
    const float* residual = nullptr;
    const HcMix* expand_mixes = nullptr;
    float* stream = nullptr;
    const std::uint16_t* fn = nullptr;
    const float* base = nullptr;
    const float* scale = nullptr;
    std::uint32_t iters = 20;
    float hc_eps = 1e-6F, mix_norm_eps = 1e-20F;
    HcMix* mixes_out = nullptr;
    float* partial = nullptr;
    std::uint32_t* counters = nullptr;
    const HcMix* collapse_mixes = nullptr;
    const float* norm_weight = nullptr;
    float norm_eps = 1e-20F;
    float* out = nullptr;
    std::uint32_t out_stride = 0;
    std::int8_t* codes = nullptr;
    float* deltas = nullptr;
    std::int32_t* sums = nullptr;
    std::uint32_t a8_stride = 0;
    // 0: by the row count (one launch up to hc_site_one_launch_rows / hc_site_fast_one_launch_rows rows, else the separate kernels:
    // expansion, mixes, norm, same bits); 1: one launch; 2: separate kernels.
    std::uint32_t launches = 0;
    // 0: by STRIX_DS41_HC_SITE_FAST (the fast one-launch kernel unless =0); 1: the previous one-launch kernel; 2: the
    // fast one (hip_ds41_layer.hip: loads batched ahead of their uses, same bits). The separate kernels are the same in all.
    std::uint32_t form = 0;
    // Fast form: rows per partial block (1, 2, 4, 8) and chunks per norm block (1-8); 0 by the row count. Same bits for all.
    std::uint32_t tile = 0, norm_part = 0;
    // Not null (with sub and fn, no norm_weight, form 2, one launch): also the collapse of the expanded stream with
    // expand_mixes[r].pre to collapse_out[r * hidden + i] (as the collapse a NormView with hc_stream reads), same bits;
    // tiles of at most 4 rows.
    float* collapse_out = nullptr;
};
inline constexpr std::uint32_t hc_site_one_launch_rows = 4;  // previous kernel, measured: one launch wins at 1-4 rows, loses at 8
inline constexpr std::uint32_t hc_site_fast_one_launch_rows = 8;  // fast kernel
[[nodiscard]] std::size_t hc_site_counter_count(std::uint32_t rows);
void enqueue_hc_site(const HcSiteView& view, void* stream);
// Fused RMSNorm (one wave per dot_chunk of the row, n a multiple of 32, n <= 32 * dot_chunk):
//  input  x[r * x_stride + i], or the collapse of hc_stream (rows x (hc * n)) with mixes[r].pre
//         (identity pre when mixes is null);
//  then   y = weight * (x * r) as rms_norm (chunked declaration);
//  then   optionally RoPE on the tail [rope_offset, rope_offset + 2 * pairs) of y (as apply_rope);
//  output y to out[r * out_stride + i] (optional) and its A8 activation (optional: codes/deltas/sums
//         at r * a8_stride, quantize_a8 bitwise).
struct NormView {
    const float* x = nullptr;
    std::uint32_t x_stride = 0;
    const float* hc_stream = nullptr;
    const HcMix* mixes = nullptr;
    const float* weight = nullptr;
    std::uint32_t rows = 0, n = 0;
    float eps = 1e-20F;
    float* out = nullptr;
    std::uint32_t out_stride = 0;
    std::int8_t* codes = nullptr;
    float* deltas = nullptr;
    std::int32_t* sums = nullptr;
    std::uint32_t a8_stride = 0;
    const float* rope_cos = nullptr;
    const float* rope_sin = nullptr;
    const std::uint32_t* positions = nullptr;
    std::uint32_t rope_offset = 0, pairs = 0;
    // Optional FP16 copy of the normalised values (round to nearest, row stride f16_stride), for the approximate FP16
    // prefill backend (STRIX_DS41_PREFILL_FP16_FAST); nullptr: none (every other output keeps its bits)
    std::uint16_t* f16 = nullptr;
    std::uint32_t f16_stride = 0;
};
void enqueue_norm(const NormView& view, void* stream);
// Plain RMSNorm of rows x n (row stride `stride`), in place allowed.
void enqueue_rms_norm(const float* x, const float* weight, std::uint32_t rows, std::uint32_t n, std::uint32_t stride, float eps,
                      float* out, void* stream);
// Attention output to the A8 input of attn_output_a: inverse RoPE of every head's tail then A8 of each
// row (heads x head_dim values, row stride heads * head_dim), as inverse_rope_a8. o is not modified.
struct InverseRopeA8View {
    const float* o = nullptr;
    std::uint32_t rows = 0, heads = 0, head_dim = 0, pairs = 32;
    const float* rope_cos = nullptr;
    const float* rope_sin = nullptr;
    const std::uint32_t* positions = nullptr;
    std::int8_t* codes = nullptr;
    float* deltas = nullptr;
    std::int32_t* sums = nullptr;
};
void enqueue_inverse_rope_a8(const InverseRopeA8View& view, void* stream);
// RoPE of `vectors` per row: element (r, v, d) of the rope tail at x[r * row_stride + v * vector_stride + offset + d];
// table rows cos/sin [table_rows][pairs], row r uses table row positions[r].
struct RopeView {
    float* x = nullptr;
    std::uint32_t rows = 0, vectors = 1, row_stride = 0, vector_stride = 0, offset = 0, pairs = 32;
    const float* cos = nullptr;
    const float* sin = nullptr;
    const std::uint32_t* positions = nullptr;
    bool inverse = false;
};
void enqueue_rope(const RopeView& view, void* stream);
// Q8_0 x A8 site (product_q8_0, bitwise): out[r * out_stride + o] = row (first + o) . x_g, where x_g is
// the A8 activation codes + r * x_stride + g * width (deltas likewise / 32) and g = o / per_group
// (per_group = count for a plain projection, 1024 for the grouped attn_output_a). rows 1..16; x_stride
// and width multiples of 32; codes 16-byte aligned.
// K-split sites (q8_0_a8_split_rows): one call per part with row_width = the full weight row (values),
// k_offset = the part's first column, width = the part's width, codes/deltas pointing at the part's
// activation slice, and accumulate = true from the second part on (out = out + part, in part order).
struct Q8A8View {
    const std::uint8_t* weight = nullptr;
    std::uint32_t width = 0, first = 0, count = 0, per_group = 0;
    const std::int8_t* codes = nullptr;
    const float* deltas = nullptr;
    std::uint32_t rows = 1, x_stride = 0, out_stride = 0;
    float* out = nullptr;
    std::uint32_t row_width = 0, k_offset = 0;  // 0 = width, 0: the whole row
    bool accumulate = false;
    // CB4 weights (requantization candidates): 18-byte blocks {f16 d, 16 bytes of 4-bit indices: byte j holds
    // weight j in its low nibble and weight j + 16 in its high nibble} into the 16-entry int8 table `table`
    // (4 words, little endian). Each block is the Q8_0 block {d, table[index]}, decoded to those int8 codes
    // in registers, so every output has the bits of the Q8_0 form of the same weights.
    // packed = 4: CB4 rows with `table`; 5 / 6: U5 / U6 rows (ds41_formats.hpp), same products; 0: Q8_0 rows.
    // packed = packed_native_fp8: native FP8 rows (ds41_native_fp8.hpp), a different product: `weight` holds the E4M3
    // codes [rows][row_width] and `scales` the UE8M0 exponents [rows / 32][row_width / 32] of the tensor's 32 x 32
    // blocks (rows counted from the tensor's first row, so first and k_offset also select the block row and column);
    // every output is product_fp8 bitwise (identity fp8_product_identity). first a multiple of 32 is not required.
    std::uint32_t packed = 0;
    std::uint32_t table[4] = {0, 0, 0, 0};
    // Packed rows in the planar layout (ds41_packed.hpp: nibble, high-bit and scale planes per row) instead of
    // interleaved blocks. Reserved for the one-row-per-wave planar kernels; rejected until they are registered.
    bool planar = false;
    // Optional A8 of the outputs (enqueue_quantize_a8's bits; accumulate off, out_stride = count = a8_stride, count a
    // multiple of 32): written by the product kernel when one block owns whole 32-output groups, else by a following
    // enqueue_quantize_a8 of out.
    std::int8_t* a8_codes = nullptr;
    float* a8_deltas = nullptr;
    std::int32_t* a8_sums = nullptr;
    const std::uint8_t* scales = nullptr;  // packed_native_fp8 only
};
inline constexpr std::uint32_t packed_native_fp8 = 8;
void enqueue_q8_0_a8(const Q8A8View& view, void* stream);
// Optional coherent producer epilogue. Returns false WITHOUT launching when the
// selected kernel has no epilogue; the caller retains its ordinary fallback.
// No output A8/accumulation: stage.data == view.out, tightly packed GPU output.
[[nodiscard]] bool enqueue_q8_0_a8_ready(const Q8A8View& view, dist::ProducerStage stage, void* stream);
// Two sites over the same activation rows in one launch (e.g. q_a and kv on the attention input), every output
// the bits of enqueue_q8_0_a8 for its own view; two launches when the views do not share one kernel form.
void enqueue_q8_0_a8_pair(const Q8A8View& a, const Q8A8View& b, void* stream);
// Outputs per wave of the native FP8 site kernel (1, 2 or 4, at most 2 from 5 rows; every choice gives the same bits); 0
// restores the default (STRIX_DS41_FP8_W=1|2|4, else 1 at 1-2 rows and 2 from 3 rows). For tests and A/B runs.
void set_native_fp8_outputs_per_wave(std::uint32_t w);
// Shared-expert epilogue of the native FP8 form (fp8_swiglu_identity): act = swiglu_clamped(gate, up, limit) of rows x
// width values (row stride width), then its A8 activation as quantize_a8 of each row (codes [r * width], deltas and sums
// [r * width / 32]). width a multiple of 32.
void enqueue_swiglu_a8(const float* gate, const float* up, std::uint32_t rows, std::uint32_t width, float limit, std::int8_t* codes,
                       float* deltas, std::int32_t* sums, void* stream);
// Prefill form of the Q8_0 x A8 sites (T rows): the WMMA iu8 batch kernel of the prefill experts,
// which gives every (row, output) the bits of product_q8_0 (the decode form above), whatever T.
// A site of `groups` groups is routed as `groups` experts of per_group rows each with entry slot
// t * groups + g, so the activation is codes[(t * groups + g) * width] (the [t][g][width] layout of
// the grouped attention output; x_stride = width for a dense site) and out[t * groups * per_group +
// g * per_group + o]. Routes come from prefill_site_routes(rows, groups, batch_rows), uploaded by the
// caller. width <= 5120 (the batch kernel's bound), per_group a multiple of 128.
[[nodiscard]] PrefillRoutes prefill_site_routes(std::uint32_t rows, std::uint32_t groups, std::uint32_t batch_rows = 32);
struct Q8A8PrefillView {
    const std::uint8_t* weight = nullptr;
    std::uint32_t width = 0, groups = 1, per_group = 0;
    PrefillRoutesView routes;
    const std::int8_t* codes = nullptr;
    const float* deltas = nullptr;
    float* out = nullptr;
};
void enqueue_q8_0_a8_prefill(const Q8A8PrefillView& view, void* stream);
// Prefill form of a dense Q8_0 x A8 site of any width (attn_output_b: 8,192), own WMMA iu8 kernel:
// out[t * out_stride + o] = product_q8_0(row o, codes/deltas of row t) bitwise for every t < rows. Each
// 32-weight group's integer dot of a (16 rows x 16 outputs) tile comes from two 16x16x16 iu8 WMMAs; the
// group term is float(dot) * (scale * delta); lane class c (groups c, c + 32, ...) is summed in ascending
// order from 0 (the four waves of a block own the classes c mod 4 and read consecutive groups together),
// and the 32 class sums are combined in the butterfly's addition tree (levels 16, 8, 4 inside a wave,
// 2 and 1 across waves). width a multiple of 32, x_stride a multiple of 16.
struct Q8A8DensePrefillView {
    const std::uint8_t* weight = nullptr;
    std::uint32_t width = 0, count = 0;
    const std::int8_t* codes = nullptr;
    const float* deltas = nullptr;
    std::uint32_t rows = 0, x_stride = 0;
    float* out = nullptr;
    std::uint32_t out_stride = 0;
    std::uint32_t row_width = 0, k_offset = 0;  // K-split parts, as Q8A8View
    bool accumulate = false;
    std::uint32_t tall = 0;  // tall form RT * 10 + OT, 1 auto (the same bits); 0: STRIX_DS41_DENSE_TALL (default off)
    // Optional A8 of the outputs (enqueue_quantize_a8's bits; accumulate off, count a multiple of 32): codes at
    // r * a8_stride + o, deltas / sums at r * a8_stride / 32 + o / 32. Written by the tall forms whose blocks own whole
    // 32-output groups; the enqueue returns whether it was written (else the caller quantizes out).
    std::int8_t* a8_codes = nullptr;
    float* a8_deltas = nullptr;
    std::int32_t* a8_sums = nullptr;
    std::uint32_t a8_stride = 0;
    // 2: this K part and the next (k_offset + width, codes + width, deltas + width / 32; row_width set), every output the
    // first part's sum + the second's, the bits of two launches with the second accumulated (one launch in the 2 x 2
    // Lean 4 form, auto, row_width 8192 / width 4096, from 896 rows; else the two launches). No A8 output.
    std::uint32_t k_parts = 1;
};
// Returns true when the outputs' A8 (a8_codes) was written.
bool enqueue_q8_0_a8_dense_prefill(const Q8A8DensePrefillView& view, void* stream);
// F16 projection (weight [count rows of width]): out[r * out_stride + o] = f16_row_dot(row first + o, x_r),
// any number of rows (above 8 a tiled kernel with the same lane classes and butterfly, so the same bits).
struct F16RowsView {
    const std::uint16_t* weight = nullptr;
    std::uint32_t width = 0, first = 0, count = 0;
    const float* x = nullptr;
    std::uint32_t rows = 1, x_stride = 0, out_stride = 0;
    float* out = nullptr;
    // Instead of `weight`: F8 rows of an MX-FP8-valued F16 tensor (ds41_packed.hpp: width E4M3 codes, then width / 32
    // E8M0 exponents per row). Each weight widens to its exact F16 value, so the products are the F16 ones bit for
    // bit. Lanes kernel only (hip_ds41_f16_rows.hpp).
    const std::uint8_t* weight_f8 = nullptr;
};
void enqueue_f16_rows(const F16RowsView& view, void* stream);
// Engram prefill projection (F16 6144 -> 25600, first 0, 32..8192 rows (default admission below), 16-byte aligned x / weights, valid strides): a 48 x 64
// LDS tile that prefetches the next slab into registers (the bits of enqueue_f16_rows for finite F16 weights; a NaN weight
// gives NaN in both, possibly with another payload). Default on; STRIX_DS41_ENGRAM_F16_PF=0
// (read on every call) restores enqueue_f16_rows, which also serves every other view.
void enqueue_engram_f16_rows(const F16RowsView& view, void* stream);
// Explicit entry for tests / microbench: rejects unsupported views, ignores the switch.
[[nodiscard]] bool engram_f16_pf_supported(const F16RowsView& view) noexcept;
// Default admission (enqueue_engram_f16_rows): supported and the 48-row padding at most 9/8 of the 64-row tile's.
[[nodiscard]] bool engram_f16_pf_admitted(const F16RowsView& view) noexcept;
void enqueue_engram_f16_pf(const F16RowsView& view, void* stream);
// enqueue_f16_rows(a) then enqueue_f16_rows(b); with STRIX_DS41_F16_PAIR=1 (default off) one launch for both when both
// take the decode lanes kernels (F16 weights, the same 1..8 rows): the same outputs bit for bit.
void enqueue_f16_rows_pair(const F16RowsView& a, const F16RowsView& b, void* stream);
// Widths that are multiples of 256 use a wide-load kernel with the same products bit for bit; tests switch it here
// (default on; STRIX_DS41_F16_WIDE=0 turns it off).
void set_f16_rows_wide(bool on);
// Embedding rows of `tokens` into rows x (hc * hidden) streams.
void enqueue_embedding(const std::uint16_t* table, const std::uint32_t* tokens, std::uint32_t rows, std::uint32_t hidden,
                       float* stream_out, void* stream);
// Greedy candidate of each row's slice of `count` logits (row stride `stride`), indices first_index + i.
// partial must hold rows * argmax_partial_blocks(count) candidates.
[[nodiscard]] std::uint32_t argmax_partial_blocks(std::uint32_t count);
void enqueue_argmax(const float* logits, std::uint32_t rows, std::uint32_t count, std::uint32_t stride, std::uint32_t first_index,
                    ArgmaxCandidate* partial, ArgmaxCandidate* out, void* stream);
// out[r] = combine_argmax(a[r], b[r]).
void enqueue_combine_argmax(const ArgmaxCandidate* a, const ArgmaxCandidate* b, std::uint32_t rows, ArgmaxCandidate* out,
                            void* stream);
// Engram: raw rows (row_count x 264 B) to bfloat16-rounded FP32 values (row_count x 256).
void enqueue_engram_rows_bf16(const std::uint8_t* rows, std::uint32_t row_count, float* out, void* stream);
// Gated injection: stream rows x (hc * hidden) updated in place from kv rows x (hidden * (hc + 1)).
void enqueue_engram_inject(float* hc_stream, const float* kv, const float* q_norm, const float* k_norm, std::uint32_t rows,
                           std::uint32_t hidden, float eps, void* stream);

// ---- launch accesses ---------------------------------------------------------------------------
// The device bytes a call may read or write, for launchers that track cache lines across dependent
// kernels (an own AQL dispatch path drops the per-packet acquire only when it can prove no line a kernel
// reads is stale). Over-approximations; weights are included (read only). One entry per pointer; a
// call with two launches (mixes, argmax) is covered as a whole.
struct DeviceRange {
    const void* pointer = nullptr;
    std::uint64_t bytes = 0;
    bool write = false;
};
struct LaunchAccess {
    std::array<DeviceRange, 24> ranges{};
    std::uint32_t count = 0;
    void read(const void* p, std::uint64_t bytes) { add(p, bytes, false); }
    void write(const void* p, std::uint64_t bytes) { add(p, bytes, true); }

private:
    void add(const void* p, std::uint64_t bytes, bool w) {
        if (p && bytes && count < ranges.size()) ranges[count++] = DeviceRange{p, bytes, w};
    }
};
[[nodiscard]] LaunchAccess launch_access(const HcSiteView& view);
[[nodiscard]] LaunchAccess launch_access(const HcMixView& view);
[[nodiscard]] LaunchAccess launch_access(const NormView& view);
[[nodiscard]] LaunchAccess launch_access(const Q8A8View& view);
[[nodiscard]] LaunchAccess launch_access(const F16RowsView& view);
[[nodiscard]] LaunchAccess launch_access_expand(const float* sub, const float* residual, const HcMix* mixes, std::uint32_t rows,
                                                std::uint32_t hidden, const float* out);
[[nodiscard]] LaunchAccess launch_access_engram_inject(const float* hc_stream, const float* kv, const float* q_norm,
                                                       const float* k_norm, std::uint32_t rows, std::uint32_t hidden);
[[nodiscard]] LaunchAccess launch_access_argmax(const float* logits, std::uint32_t rows, std::uint32_t count, std::uint32_t stride,
                                                const ArgmaxCandidate* partial, const ArgmaxCandidate* out);

} // namespace strix::ds41
