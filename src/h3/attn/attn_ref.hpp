#pragma once
// MiniMax H3 dense attention (head dim 128, non-causal, no mask): host references.
//
// Arithmetic contract of the split-KV form (the GPU kernel follows it; see attn.hpp for the memory layouts):
//   - keys are partitioned into globally indexed key blocks b = 0, 1, ... (contiguous key ranges);
//   - per (query row, head, block) the kernel returns an fp32 partial (mc, l, o[128]):
//       mc = (max over the block's keys of the raw score q.k) * c, with c = fp32(log2(e) / sqrt(128)),
//       l  = sum over the block's keys of p_j,  p_j = exp2(fma(s_j, c, -mc_running)) (online, rescaled),
//       o  = sum over the block's keys of bf16(p_j) * v_j (unnormalised, rescaled online);
//   - the canonical merge combines blocks in the fixed order 0, 1, ..., nb - 1:
//       M = max_b mc_b,  e_b = exp2(mc_b - M),  L = sum_b l_b e_b,  O = sum_b o_b e_b,  out = O / L.
// The 1-node path runs the same per-block computation on every block and the same merge, so the result for a
// query row does not depend on which node computes it or in which order the blocks' partials are produced.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strix::h3::attn {

inline constexpr unsigned kHeadDim = 128;
inline constexpr unsigned kKeyStep = 64;  // keys per online-softmax step of the kernel
// log2(e) / sqrt(128) rounded once to fp32: scores are scaled into the exp2 domain by one fma.
inline constexpr float kScaleLog2 = static_cast<float>(1.4426950408889634073599 / 11.313708498984760390231);

// bf16 as raw bits.
std::uint16_t bf16_from_float(float x);  // round to nearest even (NaN stays NaN)
float bf16_to_float(std::uint16_t b);

// Dense test tensors in the reference layout: x[h][row][d] (head-major, rows contiguous), bf16 bits.
struct HostQkv {
    unsigned heads = 0, q_rows = 0, k_rows = 0;
    std::vector<std::uint16_t> q, k, v;  // [heads][rows][128]
    std::uint16_t qv(unsigned h, unsigned r, unsigned d) const { return q[(std::size_t(h) * q_rows + r) * kHeadDim + d]; }
    std::uint16_t kv(unsigned h, unsigned r, unsigned d) const { return k[(std::size_t(h) * k_rows + r) * kHeadDim + d]; }
    std::uint16_t vv(unsigned h, unsigned r, unsigned d) const { return v[(std::size_t(h) * k_rows + r) * kHeadDim + d]; }
};
// Deterministic pseudo-random normal values (scaled by q_scale for Q, 1 for K and V) rounded to bf16.
HostQkv make_qkv(unsigned heads, unsigned q_rows, unsigned k_rows, std::uint64_t seed, float q_scale);

// fp64 reference: out[d] = sum_j softmax_j(q.k_j / sqrt(128)) v_j[d] over all keys, inputs as bf16 values.
void reference_row_f64(const HostQkv& x, unsigned head, unsigned row, double* out);

// One block's fp32 partial, in the kernel's order of operations (64-key steps, the lane split of the sums).
struct Partial {
    float mc = 0.0F, l = 0.0F;
    float o[kHeadDim] = {};
};
// Keys [k0, k1) of x (k1 - k0 may be 0: mc = -inf, l = 0, o = 0).
Partial partial_row_f32(const HostQkv& x, unsigned head, unsigned row, unsigned k0, unsigned k1);
// Canonical merge of nb partials (in index order) into out (fp32).
void merge_f32(const Partial* parts, unsigned nb, float* out);
// Merge into fp64 from fp64-exact partials (to test the merge identity itself).
struct PartialF64 {
    double m = 0.0, l = 0.0;
    double o[kHeadDim] = {};
};
PartialF64 partial_row_f64(const HostQkv& x, unsigned head, unsigned row, unsigned k0, unsigned k1);
void merge_f64(const PartialF64* parts, unsigned nb, double* out);

// G1 error of one output row: max_d |got - ref| / max_d |ref|.
double row_error(const float* got, const double* ref);

}  // namespace strix::h3::attn
