#pragma once

#include "gemma4/kv/fp16_mirror.hpp"
#include "gemma4/kv/storage.hpp"

namespace strix::gemma4::gpu {
// FAST decode attention of the GLOBAL layers (bit-changing, --profile fast only once qualified). Two forms differing
// only in the K/V operands: g4-dattn-f1 reads FP16 halves of the committed FP32 KV, g4-dattn-f2 reads KV8 groups
// (int8 codes, one FP16 scale per 16 features; kv/kv8_device.hpp) - from the per-stream mirror, or converted in
// registers from the FP32 row for rows the mirror does not hold yet (the same values either way). The query is the
// exact path's normed and rotated query. Split-K over fixed 128-key tiles of absolute positions: per tile and query
// head an FP32 maximum, FP32 exponentials and denominator, FP32 value sums; then an ordered merge over the tiles
// (M = max m_t, w_t = exp(m_t - M), out = sum_t w_t O_t / sum_t w_t l_t, tiles in position order). The tiling, every
// reduction order and the result depend only on the row's position and the KV bytes: not on the number of rows in
// the pass, the stream's other rows, the mirror's sync state, timing or the run.
// STRIX_G4_DECODE_ATTN = g4-dattn-f1 | g4-dattn-f2 turns it on (the FAST profile sets its form once qualified;
// --allow-pending runs it unqualified); unset or exact = the exact decode attention (OFF). Sliding layers always
// run the exact path.
inline constexpr char fast_decode_attention_f16[] = "g4-dattn-f1";
inline constexpr char fast_decode_attention_kv8[] = "g4-dattn-f2";
[[nodiscard]] MirrorFormat fast_decode_attention_format(); // from STRIX_G4_DECODE_ATTN, read once per process
[[nodiscard]] const char* fast_decode_attention_name(MirrorFormat format);
// Scratch bytes for `rows` rows whose attention stride (positions rounded up to the 64-position bucket) is `stride`.
[[nodiscard]] std::size_t fast_decode_attention_scratch_bytes(std::uint32_t rows, std::uint32_t stride);
// Global layers only (dim 512, heads a multiple of the group). mirrors holds one descriptor per row (that row's
// stream mirror for this layer's producer). Returns false, enqueueing nothing, when the geometry is not admitted
// or scratch is too small; the caller then runs the exact path.
bool fast_decode_attention(const float* q, LinearView q_norm, const std::uint64_t* positions, const float* table,
                           const KvRead* reads, const Fp16MirrorRead* mirrors, std::uint32_t rows,
                           std::uint32_t head_count, std::uint32_t dim, std::uint32_t stride, float* out,
                           float* scratch, std::size_t scratch_bytes, std::uint32_t* failure, Queue queue,
                           bool parallel, bool bounded, bool intervals, MirrorFormat format, A8Rows quant = {});
} // namespace strix::gemma4::gpu
