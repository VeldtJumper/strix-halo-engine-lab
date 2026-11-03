#pragma once
// MiniMax H3 block glue kernels (gfx1151, wave32): K3 (RMSNorm + adaLN modulation per segment, gated residual),
// K4 (per-head q/k RMSNorm + split-half 3D RoPE -> Q, K token-major [S][56][128]; V^T [56][128][S]), K5 (SwiGLU,
// for linears without the fused epilogue), K6 (adaLN mod vectors, patch projections, final norm + modulation, fp32
// heads). Every kernel is row-local and bitwise equal to the host reference in ops_ref.hpp / host_ref.cpp (same
// operation order; the RMSNorm reduction is the reference library's 32-lane order, which maps onto one wave).
// All pointers are device pointers; bf16 values are raw uint16 bits. Launchers enqueue on `stream` (hipStream_t).
#include <cstdint>

namespace strix::h3::block::gpu {

// K3: h[r] = mod(rmsnorm(x[r], w)) with mod row m = mod_row[r]:  h = bf16(bf16(y * s1[m]) + shift[m]).
// s1 / shift == nullptr: plain RMSNorm. n % 32 == 0. x and h may not alias.
void launch_rmsnorm_mod(const std::uint16_t* x, const std::uint16_t* w, const std::uint32_t* mod_row, const std::uint16_t* s1,
    const std::uint16_t* shift, std::uint32_t rows, std::uint32_t n, float eps, std::uint16_t* h, void* stream);
// K3: x[r] = bf16(x[r] + o[r] * g[mod_row[r]]) (in place).
void launch_gate_residual(std::uint16_t* x, const std::uint16_t* o, const std::uint32_t* mod_row, const std::uint16_t* g,
    std::uint32_t rows, std::uint32_t n, void* stream);
// K4: qkv [S][3 * 7168] -> q, k [S][56][128] (RMSNorm with qw / kw, then split-half RoPE on [0, 96) with the rotation
// table c, s [S][48] bf16; c == nullptr: no rotation) and vt [56][128][S] (raw V).
void launch_qk_norm_rope(const std::uint16_t* qkv, std::uint32_t S, const std::uint16_t* qw, const std::uint16_t* kw,
    const std::uint16_t* c, const std::uint16_t* s, float eps, std::uint16_t* q, std::uint16_t* k, void* stream,
    std::uint32_t* head_sumsq_max = nullptr);
// head_sumsq_max (optional, [112] float bits, zeroed by the caller): max over rows of ||y||^2 per q head (0 .. 55) and
// k head (56 .. 111) of the written q / k (fp32 sum, any order) - the attention input contract check.
// V^T [56][128][vt_pitch] (keys 0 .. S-1; entries S .. vt_pitch-1 are not written).
void launch_transpose_v(const std::uint16_t* qkv, std::uint32_t S, std::uint16_t* vt, std::uint64_t vt_pitch, void* stream);
// x[i] = bf16(o[i] + x[i]) (the refiner's in-place residual add).
void launch_add_residual(std::uint16_t* x, const std::uint16_t* o, std::uint64_t n, void* stream);
// bf16 linear (refiner, condition_proj): y[m][n] = bf16(s + b), s = fp32 k-ascending sum of the (exact) products, b may be
// null (bitwise == bf16_linear_host).
void launch_bf16_linear(const std::uint16_t* x, std::uint32_t m, const std::uint16_t* w, const std::uint16_t* b, std::uint32_t n,
    std::uint32_t k, std::uint16_t* y, void* stream);
// K5: out[r][i] = bf16(bf16(silu(g)) * u), g = f1[r][i], u = f1[r][ffn + i]; silu = the 65536-entry bf16 table.
void launch_swiglu(const std::uint16_t* f1, std::uint32_t rows, std::uint32_t ffn, const std::uint16_t* silu_table,
    std::uint16_t* out, void* stream);
// K6: y[m][n] = fp32(sum_k fp64(x[m][k]) * fp64(w[n][k]) (k ascending) + fp64(b[n])) (adaLN, final adaLN, heads).
// x may be read with a row index map (x row = xrow ? xrow[m] : m) and a row stride.
void launch_linear_f64(const float* x, std::uint32_t x_rs, std::uint32_t m, const float* w, const float* b, std::uint32_t n,
    std::uint32_t k, float* y, void* stream);
// K6: adaLN fp32 rows [rows][6 * 5376] -> bf16 shift_msa, s1_msa, gate_msa, shift_mlp, s1_mlp, gate_mlp [rows][5376].
void launch_mod_vectors(const float* ada, std::uint32_t rows, std::uint16_t* const out[6], void* stream);
// K6: patch projection rows (fp32 [m][k], k-ascending fp32 sum, + bias, bf16) written to h[dst_row[i]] ([.][5376]).
void launch_patch_proj(const float* rows, std::uint32_t m, std::uint32_t k, const float* w, const float* b,
    const std::int32_t* dst_row, std::uint16_t* h, void* stream);
// K6: final norm + modulation: f[i] = (fp32(rmsnorm(x[row0 + i], w)) * (1 + scale)) + shift  (fp32 [m][5376]).
void launch_final_mod(const std::uint16_t* x, std::uint32_t row0, std::uint32_t m, const std::uint16_t* w, const float* shift,
    const float* scale, float eps, float* f, void* stream);

// Test hook: per-row reference-order sum of squares (ss[r]) and inverse RMS (inv[r]) as the K3/K4 kernels compute them.
void launch_row_rms_debug(const std::uint16_t* x, std::uint32_t rows, std::uint32_t n, float eps, float* ss, float* inv, void* stream);

}  // namespace strix::h3::block::gpu
