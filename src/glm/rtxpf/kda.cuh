#pragma once
#include "native.cuh"
namespace strix::glm::rtxpf {
inline constexpr const char* kda_identity="glm-pf-rtx-native-v1-kda32";
// One full, unsharded attention sublayer. x is already input-normalized;
// residual/hyperconnection and MoE are outside this API. Ordinary F16 matrices
// are [output,input], convolution/norm BF16, A_log/dt_bias FP32.
struct KdaWeights {
    unsigned hidden,heads;
    Matrix qkv,out;
    const U16 *b,*fa,*fb,*ga,*gb,*conv,*norm;
    const float *a_log,*dt;
};
struct KdaCall {
    unsigned rows;
    KdaWeights w;
    const float* x;
    float *y,*state,*conv; // state [H,128,128], conv [3*H*128,3], in/out
    // rows <= 32 (decode replay): the qkv/out trellis projections as row_mma.cuh strips from the
    // packed tiles (no FP16 layer decode); the same bits as the CUTLASS projection.
    bool row_kernels=false;
    bool row_r2=false; // row-kernels round 2 (opt-in): A loads before the decode, dense FP16 strips for the ordinary projections
};
// No allocation, device discovery or synchronization. Single caller-owned
// stream, nonaliasing buffers; initial state/conv may be nonzero.
cudaError_t prepare_kda(); // device setup once, before enqueue/timing
std::size_t kda_workspace_bytes(unsigned rows,unsigned hidden,unsigned heads);
// Optional open-block checkpoint (decode replay). row is a multiple of 32 in
// [0,rows): before that block's recurrence the incoming state is copied to
// state, and the convolution history at that row (last three projected rows
// before it, or the incoming history) to conv. Reads only; arithmetic unchanged.
struct KdaCheckpoint {unsigned row;float *state,*conv;};
cudaError_t enqueue_kda(const KdaCall&,void*,std::size_t,cudaStream_t,bool reference=false,const KdaCheckpoint* checkpoint=nullptr);
}
