# H3 int8 GEMM activation quantiser: exact spec (portable reference path)

Scope: the 200 quantised linears (blocks.{0..49}.attn.qkv_proj / attn.out_proj / mlp.fc1 / mlp.fc2),
the per-layer quant config tensor = `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}` on every
one (inventory/minimax_h3_fl2va_pruned_int8_convrot.json). Weights: int8 `[N,K]` + fp32
`weight_scale [N,1]` (per output channel), already rotated offline (`W_rot = W @ H^T` per 256-group
of K, then per-row absmax/127 quantisation). Compute dtype bf16. The steps below are the portable
(eager) path of the quantised-linear library, re-derived independently and
checked bitwise against it on 6 families x 64 rows x {K=5376, K=14336} (`quant` set, meta.checks).

## Chain, per activation row x (bf16, length K, K % 256 == 0)

0. (fc2 only) input activation: x = swiglu(fc1_out) with fc1_out = [gate | up] (first half gate):
   `silu(gate)` in bf16 (computed in fp32 internally, rounded to bf16), then `* up` rounded to bf16.
   Two bf16 roundings. (The module calls `linear_input_act(fc2, fc1(x), "swiglu")`.)
1. Rotation: for each group g of 256 consecutive elements, `r_g = x_g @ H` with
   `H = kron(H4,H4,H4,H4) / 16`, `H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]` ("regular"
   Hadamard, symmetric, entries +-1/16, NOT the Sylvester/Walsh ordering). H is cast to bf16 (exact),
   the product is a bf16 matmul: fp32 accumulation, ONE rounding to bf16 (RNE) at the end.
   Normalisation is the single 1/16 inside H (exact power of two, so it commutes with any order).
   The rotated intermediate is bf16. Accumulation order: unspecified (library matmul); the last-ulp
   bf16 result is order-sensitive: vs an exact fp64 sum rounded to fp32 then bf16, 4/344064 (K 5376,
   heavy-tailed rows) and 15/917504 (K 14336) values differed by 1 ulp; none of them changed an int8 code
   in our samples; on the real captured fc2 input (400 x 14336) 155 rotated values differ by 1 ulp, again with
   zero code changes. An engine that wants bit-exact codes must either match the order or accept these rare
   1-ulp rotation differences (they flip a code only if the quotient lands on a rounding boundary).
2. Row absmax: `a = max_i |r_i|` (exact, over the bf16 values).
3. Row scale (fp32): `s = max(fp32(a) / 127.0f, 1e-30)` (fp32 RNE division).
4. Division scale: `s_b = bf16(s)` (RNE); if `s_b == 0` use bf16 tiny. NOTE: the division uses the
   bf16-rounded scale, the epilogue uses the fp32 scale.
5. Quotient: `t = bf16(fp32(r_i) / fp32(s_b))` (fp32 division, RNE to bf16 - a second rounding; in
   [64,128) bf16 resolution is 0.5, so .5 ties are common).
6. Code: `q_i = int8(clamp(round_half_to_even(t), -128, 127))`. Because s_b can be below s, |t| can
   exceed 127.5: -128 occurs (14 codes in 917504 heavy-tailed K=14336 values, 8 / 20 in the real qkv / fc2 inputs; 127 is reached by the
   absmax element and often by its neighbours). Asymmetric saturation: +127 / -128.
7. GEMM: `acc[m,n] = sum_k q[m,k] * w_q[n,k]` in int32 (exact; |acc| < 2^31).
8. Epilogue: `y = bf16( fp32(acc) * (s[m] * w_scale[n]) )` - the two fp32 scales are multiplied FIRST,
   then one fp32 multiply with the accumulator, then RNE to bf16. `(acc*s)*w_scale` differs in a few
   outputs (1 of 4096 in the K=14336 edge family). No bias on these layers. Output dtype bf16.
9. Residual: the block adds `gate * y` into the residual stream with `addcmul_` in bf16 (outside the GEMM).

## The fp32-staged variant (what an accelerated fused kernel may do)

The reference GPU's fused quantiser (used for 256 <= K <= 16384 with
group 256 and the swiglu fused in) is shipped only as a binary; its shared-memory budget,
`(K + groups_in_flight*2*256) * 4` bytes, suggests the row and the butterfly scratch are fp32. If the
rotated row stays fp32 (no bf16 rounding in steps 0, 1, 4, 5), the codes differ from the portable path in
~7-8 % of positions on random rows (all by +-1) and 8.5 % / 7.2 % on the real block-0 qkv / fc2 inputs
(|diff| <= 2), and 68-80 % of the bf16 outputs differ (quant set,
`meta.stats.*fp32_staged`, and the `fp32staged_*` tensors). The engine uses the portable path as its
reference (bitwise gate); this set provides both. The integer GEMM and the epilogue are the same.

## Files (set `quant`)

`hadamard256_f32`; per `K{5376,14336}/{normal,heavy,edge,real_b0_*}/`: `x` bf16 [64,K], `x_rot` bf16,
`row_absmax` f32, `row_scale` f32 (step 3), `row_scale_bf16` bf16 (step 4), `q` i8, `w_q` i8 [64,K]
(first 64 output channels of block 0 qkv_proj resp. fc2), `w_scale` f32 [64], `acc` i32 [64,64],
`out` bf16 [64,64], `fp32staged_q`, `fp32staged_row_scale`, `fp32staged_out`.
`real_b0_qkv_in` = the captured block-0 qkv input (modulated norm1 output, 400 rows of dit_t2va_small),
`real_b0_fc2_in` = swiglu of the captured block-0 fc1 output (400 rows).
