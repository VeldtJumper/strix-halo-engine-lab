#pragma once
#include "trellis.cuh"
#include "row_r2.hpp"
#include <cuda_runtime.h>
#include <cstddef>

namespace strix::glm::rtxpf {
inline constexpr const char* identity = "glm-pf-rtx-native-v1";
inline constexpr const char* cutlass_identity = "glm-pf-rtx-native-v1-c1";
inline constexpr unsigned group_size=16, topk=8;
struct Matrix {
    const U16 *packed, *suh, *svh;
    unsigned input, output, bits;
};
struct Expert { Matrix gate, up, down; };
// Device pointers. Matrices have full (not TP-sharded) dimensions. No biases.
// Trusted caller contract: each expert has gate/up H->I and down I->H;
// all rates are K3..K6, mul1 marker 0x83dcd12d, complete H128 suh/svh.
// Input, descriptors, packed data, output and workspace must not alias.
struct Call {
    unsigned rows, experts, hidden, inter;
    const float* x;
    const U32* ids;             // [rows,8], unique IDs per row, in router slot order
    const float* weights;       // [rows,8], finite, already normalized/scaled by router
    const Expert* matrices;
    float* y;
    // Expert batches (CUTLASS path only): false skips the final fixed-order merge.
    // Routes with ids >= experts are skipped and leave their per-(row, slot)
    // contribution untouched, so batches of WHOLE experts over the same rows and
    // workspace, the last with combine=true, give the one-call result bitwise.
    bool combine=true;
    // CUTLASS path, routed (experts > 1): launch only the groups that can hold routed rows;
    // the device lists the routed experts first (c1_schedule.hpp). Per-expert arithmetic unchanged.
    bool compact_groups=false;
    // CUTLASS path, one expert (shared / dense): its FP16 B operands decoded once by
    // decode_expert_fp16 (the decode window's exact values); the per-call decode is skipped.
    const U16 *fp16_gate=nullptr,*fp16_up=nullptr,*fp16_down=nullptr;
    // CUTLASS path, rows <= 4 (decode, verify): the C1 GEMMs as register-decoded m16n8k16 chains over the
    // packed tiles (no FP16 decode window, no grouped GEMM, routed experts only). Same per-output
    // accumulation as the 32-row CUTLASS kernel, so the same bits; the FP16 operands are not used.
    bool row_gemv=false;
    unsigned row_r2=0; // RowR2 bits (opt-in): r2_pre A loads before the decode, r2_par parallel prefix / compaction
};
struct Profile {
    cudaEvent_t begin[18], decoded[18], gate_up[18], down[18];
    // C1: timestamps on compute stream, except decode_start/decoded.
    cudaEvent_t decode_start[18], ready[18], input[18], gu_start[18], gu_gemm[18],
        swiglu[18], down_input[18], down_start[18], down_gemm[18], merge_begin, merge_end;
};
// Caller creates a nonblocking decode stream and timing-enabled events. No
// concurrent calls may share these events or the workspace. Retired[g] prevents
// reuse of the parity buffer until all three GEMMs consuming it have finished.
struct CutlassExecution {
    cudaStream_t decode_stream;
    cudaEvent_t dispatch, ready[18], retired[18];
    int decode_blocks; // total CTA budget derived from decoder residency
    int hot_blocks, cold_blocks; // residency-derived, populated by prepare_cutlass
};
cudaError_t prepare_cutlass(CutlassExecution&); // host setup, before timing
std::size_t workspace_bytes_cutlass(unsigned rows,unsigned experts,unsigned hidden,unsigned inter);
cudaError_t enqueue_cutlass(const Call&,void*,std::size_t,cudaStream_t,
    const CutlassExecution&,Profile* = nullptr);
// One expert's gate/up (hidden x inter) and down (inter x hidden) FP16 B operands, exactly as the
// C1 decode window writes them; `one` is a device word holding 1. Caller owns the buffers.
cudaError_t decode_expert_fp16(const Expert* e,unsigned hidden,unsigned inter,U16* qg,U16* qu,U16* qd,const U32* one,
    const CutlassExecution&,cudaStream_t);
// Caller owns workspace and keeps matrices/inputs alive through stream completion.
// No allocation, synchronization, device discovery or environment reading inside enqueue.
std::size_t workspace_bytes(unsigned rows,unsigned experts,unsigned hidden,unsigned inter);
std::size_t workspace_bytes_dense(unsigned rows,unsigned hidden,unsigned inter);
cudaError_t enqueue(const Call&,void* workspace,std::size_t bytes,cudaStream_t,Profile* = nullptr);
// Diagnostic SIMT reference: same exact decoded operands, chronological scalar
// FP32 FMA chain, no tensor cores. One-expert calls use dense/unit routing.
cudaError_t enqueue_reference(const Call&,void* workspace,std::size_t bytes,cudaStream_t);
// Standalone decode diagnostic, also used by the benchmark stage timing.
cudaError_t decode(Matrix,U16* decoded,cudaStream_t);
// Dense/shared use the exact same kernels with one route per row, unit weights.
cudaError_t enqueue_dense(unsigned rows,unsigned hidden,unsigned inter,const float* x,
    const Expert* matrix,float* y,void* workspace,std::size_t bytes,cudaStream_t,Profile* = nullptr);
}
