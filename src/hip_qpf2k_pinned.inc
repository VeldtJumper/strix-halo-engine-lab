// Operand-pinned FP32 operations, shared by the separate launches and the fused forms that must equal them.
// IEEE results do not depend on operand order except for which NaN payload survives when two NaN operands meet;
// the compiler commutes fadd / fmul / fma multiplicands freely, so every site that a fused form must reproduce is
// written through these helpers on both sides, each in the operand order (src0 first) the separate kernel had in
// the gfx1151 ISA before pinning, so default bits are unchanged:
//   combine (combine_element)         v_add_f32  shared, routed
//   inject (inject_element)           v_fmac_f32 hyper += gate * hidden        (src0 gate)
//   norm leaves                       v_fmac_f32 sum += value * value
//   norm LDS tree                     v_add_f32  r[l + s], r[l]                (upper first, every level)
//   normalized                        v_mul_f32  inverse, value (BF16 rows; FP32 rows: value, inverse); v_mul_f32 (that), weight
//   split-K lane reduction            v_add_f32  sum, xor(sum, d)              (own value first, d = 16 .. 1)
//   split-K 16-tile sum               v_add_f32  total, red[k]                 (running total first, k ascending)
//   low value (split-K, both forms)   v_mul_f32  scaled, sigmoid(scaled)      (separate: is_inject ? 2 : scaled)
//   output stream mix (lane chain)    v_fmac_f32 mixed += normalized * sig    (rows chain, split and regs forms, fused)
//   output stream mix (inline gate)   v_fmac_f32 mixed += sigmoid(gate) * normalized   (staged / Q8 batched forms)
// Included inside an anonymous namespace of each translation unit that uses them.
__device__ __forceinline__ float qpf2k_add(float src0, float src1) {
    float d;
    asm volatile("v_add_f32 %0, %1, %2" : "=v"(d) : "v"(src0), "v"(src1));
    return d;
}
__device__ __forceinline__ float qpf2k_mul(float src0, float src1) {
    float d;
    asm volatile("v_mul_f32 %0, %1, %2" : "=v"(d) : "v"(src0), "v"(src1));
    return d;
}
__device__ __forceinline__ float qpf2k_fmac(float acc, float src0, float src1) {
    asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(acc) : "v"(src0), "v"(src1));
    return acc;
}
// The normalisation's first product in the norm kernel's operand order for the activation format.
template <typename Storage> __device__ __forceinline__ float qpf2k_scale(float inverse, float value) {
    if constexpr (std::is_same_v<Storage, float>) return qpf2k_mul(value, inverse);
    else return qpf2k_mul(inverse, value);
}
