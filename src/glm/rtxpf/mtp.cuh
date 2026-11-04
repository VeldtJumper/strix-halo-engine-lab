#pragma once
#include "dsa.cuh"
namespace strix::glm::rtxpf {
inline constexpr const char* mtp_identity="glm-pf-rtx-native-v1-mtp45-f32idx";
struct MtpWeights {
    unsigned layer=45;
    Matrix eh;
    DsaWeights attention;
    const U16 *enorm,*hnorm,*input_norm,*post_norm,*head_norm,*router;
    const float* router_bias;
    const Expert *experts,*shared; // 288 and 1 descriptors, full H4096/I2048
};
// Embedding rows are supplied for the next-token IDs paired with hidden rows.
// Output is shared_head.norm(z + attn + MoE), before the vocabulary head.
struct MtpCall {
    unsigned rows,first;MtpWeights w;DsaState state;
    const float *embedding,*hidden;float* y;unsigned* nonfinite;
    DsaIndex index=DsaIndex::fp32;
    bool legacy_fp32_index=false;
    // 0: complete call (prefill). Decode splits the same sequence around the
    // routed-expert acquisition: 1 runs through router selection and copies
    // the [rows,8] IDs to route_ids; 2 runs routed/shared FFN and the finish on
    // the same workspace, with w.experts rebound to the acquired descriptors.
    unsigned phase=0;
    U32* route_ids=nullptr;
    // rows <= 32 (decode): row_mma.cuh strips for the eh / attention trellis projections (row_proj)
    // and the one-row routed/shared MoE (row_moe); the same bits as the CUTLASS paths.
    bool row_proj=false,row_moe=false;unsigned row_r2=0; // row_r2: RowR2 bits (opt-in)
    bool dsa_short_skip=false;
    DsaIndexQueryProfile* index_query_profile=nullptr;
};
std::size_t mtp_workspace_bytes(unsigned rows,unsigned capacity);
// Reuses DSA and grouped trellis MoE. prepare_dsa / prepare_cutlass first.
// Both attention and expert references are scalar FP32, using raw matrices.
cudaError_t enqueue_mtp(const MtpCall&,void*,std::size_t,cudaStream_t,const CutlassExecution&,bool reference=false);
}
