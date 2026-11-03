#pragma once
// Round two of the few-row decode kernels (opt-in, same values): one bit per change, so each can be switched
// on alone (--decode-row-kernels tokens r2pre, r2dense, r2idx, r2hc, r2par; r2 = all of them).
namespace strix::glm::rtxpf {
enum RowR2 : unsigned {
    r2_pre = 1u,    // A fragments loaded before each tile's weight decode (strips: MoE, KDA, DSA, MTP)
    r2_dense = 2u,  // dense FP16 [n][k] strips for the ordinary projections (KDA; DSA when its index is not FP32)
    r2_idx = 4u,    // FP32 index projections for up to four rows as one ascending fmaf chain per output
    r2_hc = 8u,     // hyper-connection mix chains with four-wide shared-memory reads
    r2_par = 16u,   // routed MoE prefix sum and expert compaction with one thread per expert
    r2_all = 31u,
};
}  // namespace strix::glm::rtxpf
