// R1b generated identity (default): device kernels of the blocked chains.
// Default prompts retain R1 (arithmetic contract R1); r1b-all also applies R1b to reference prompts. Scope (allowlist, nothing else changes):
//   router logits      288 x 4,096 F16  -> 16 blocks of 256 + tree 8, 4, 2, 1
//   mHC hc_attn/ffn fn  24 x 16,384 F32 -> 64 blocks of 256 + tree 32 .. 1 (inputs = the R2/R3-normalized streams)
//   KDA f_a / g_a       128 x 4,096 F16 -> 16 blocks of 256 + tree 8 .. 1
//   KDA b_proj          32 (rank rows) x 4,096 F16 -> same
//   KDA f_b / g_b       K = 128: UNCHANGED (R1 chain; shorter than one block, not a long chain)
// Every other operation keeps its R-rule. The order is fixed independent of rank, batch, occupancy.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace strix::glm::r1b {

// Process-wide identity switch for the seams that do not see engine Options (FastAttn): set by the engine at
// construction / runtime mode switch; default prefill scopes it to false (R1).
void set_active(bool on);
bool active();

// Cached process option, shared by load planning, actual load and dispatch (default 7).
int dense_variant();
// Pure device data movement, asynchronous on s. Distinct, 16-byte aligned device buffers,
// each n*4096 halves. packed[c8][output][block_j][8], c8=0..31, block_j=0..15.
void pack_k4096(const std::uint16_t* W, unsigned n, std::uint16_t* packed, hipStream_t s);

// logits[r][288] = r1b_dot(x[r], router[e]) for rows 1..8 (router F16 [288][4096]).
void router_dense(const float* x, const std::uint16_t* router, float* logits, unsigned rows, hipStream_t s, int variant = -1);

// y[r * ldy + o] = r1b_dot(x[r * ldx + 0..4095], W[o][0..4095]) for o < n, rows 1..8, W F16 [n][4096], rows * n <= 4096.
// Variant 2: two launches (block partials, tree). Variant 4: one launch for rows == 1,
// otherwise variant 2. Variant 5 requires rows == 1 and W in pack_k4096 layout;
// callers must supply original weights with explicit variant 2 for multi-row fallback.
// Variant 6 requires packed weights for rows 1..8, one launch with weight reuse across rows.
// Variant 7 (auto): packed v5 for one row, packed v6 for rows 2..8.
// -1 selects STRIX_GLM_R1B_DENSE (default 7).
// _x3: three weights on the same input in one call (KDA f_a, g_a, b_proj).
void dense_f16_k4096(const float* x, unsigned ldx, const std::uint16_t* W, unsigned n, float* y, unsigned ldy, unsigned rows,
                     hipStream_t s, int variant = -1);
void dense_f16_k4096_x3(const float* x, unsigned ldx, const std::uint16_t* const W[3], const unsigned n[3], float* const y[3],
                        const unsigned ldy[3], unsigned rows, hipStream_t s, int variant = -1);

// hc_pre in the R1b identity: same outputs / buffers as dev::hc_pre (F = normalized streams, mix [24], xc collapse),
// the 24 fn chains blocked (64 x 256 + tree). Rows 1..8. Bitwise == host::hc_mix(..., Identity::kR1b) + hc_collapse.
// Three launches (row norm, 64 block partials per chain, tree + R6 + collapse); partials live in a module-static device
// array (no allocation; one stream at a time, like every other engine scratch).
// variant 1: rr, partials, finish (3 launches); 2: rr + partials with a last-arriving-workgroup finish (2 launches);
// 3: distributed exact tree, one-row partial LDS specialization; unchanged serial R6 per row.
// 4: variant 3 plus production parallel wave32 R6 (same per-value arithmetic).
// -1: STRIX_GLM_R1B_HC (default 4). Same bits.
void hc_pre(const float* X, const float* fn, const float* base, const float* scale, float* F, float* mix, float* xc,
            unsigned rows, hipStream_t s, int variant = -1);

// Opt-in process option (read once, default false). Fusion uses the v4 hc finish,
// followed by the exact dev::rmsnorm<256> BF16 arithmetic; xc remains materialized.
bool hc_norm_enabled();
// site: launches per call (same bits). 0: rr, partials, finish + norm (3); 1: every workgroup recomputes rr, the last one
// to arrive finishes (1 launch) at one row, 0 at 2..8 rows; 2: as 1 at one row, rr + partials then finish + norm (2) at 2..8.
// -1: STRIX_GLM_HC_SITE (default 1).
int hc_site_mode();
void hc_pre_norm(const float* X, const float* fn, const float* base, const float* scale, float* F, float* mix, float* xc,
                 const std::uint16_t* norm, float* xn, unsigned rows, hipStream_t s, int site = -1);

}  // namespace strix::glm::r1b
