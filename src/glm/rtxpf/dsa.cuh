#pragma once
#include "native.cuh"
#include "dsa_selection.hpp"
namespace strix::glm::rtxpf {
inline constexpr const char* dsa_identity="glm-pf-rtx-native-v1-dsa64-f32idx";
inline constexpr const char* dsa_mma_index_identity="glm-pf-rtx-native-v1-dsa64-mmaidx";
enum class DsaIndex { fp32, mma, mma_proj_fp32_score, fp32_proj_mma_score, tf32x3 };
inline constexpr const char* dsa_index_identity(DsaIndex v) {
    switch(v) {
    case DsaIndex::mma:return dsa_mma_index_identity;
    case DsaIndex::mma_proj_fp32_score:return "glm-pf-rtx-native-v2-dsa64-mmap-f32s";
    case DsaIndex::fp32_proj_mma_score:return "glm-pf-rtx-native-v2-dsa64-f32p-mmas";
    case DsaIndex::tf32x3:return "glm-pf-rtx-native-v2-dsa64-tf32x3p-bf16x3s";
    default:return dsa_identity;
    }
}
inline constexpr unsigned dsa_chunk=256,dsa_splits=(2051+dsa_chunk-1)/dsa_chunk;
inline constexpr unsigned dsa_tile=64,dsa_max_tokens=2051,dsa_phases=10,dsa_max_windows=512;
inline constexpr const char* dsa_phase_names[dsa_phases]={"decode","state_projections_store","pool","query_projections","q_absorb","index_scores","select","sparse_attention","v_absorb","output_projection"};
struct DsaWeights {
    Matrix qa,qb,kva,out,iq;
    const U16 *qa_norm,*kv_norm,*kvb,*key,*gate,*head_weight,*key_norm,*key_bias;
    const float* ape;
};
// Single sequence, full 64-head geometry. x is already normalized. Node layout:
// latent [capacity,512], index key/gate [capacity,128], pool [capacity/4,128].
// No expanded K/V, whole-prefix score matrix, allocation or synchronization.
struct DsaState {U16 *latent,*key,*gate;float* pooled;unsigned capacity;};
struct DsaDiagnostics {
    unsigned long long differing_rows, differing_pools, near_tie_pools, outside_tolerance_pools;
};
// Three disjoint intervals: iq weight decode, iq/head projections, score launch.
// Decode profiling only; preallocated by the caller, never part of model state.
struct DsaIndexQueryProfile {cudaEvent_t begin[3],end[3];};
struct DsaCall {
    unsigned rows,first;
    DsaWeights w; DsaState state;
    const float* x;float* y;unsigned* nonfinite;
    U32 *tokens,*counts; // optional [rows,2051], [rows], diagnostic export
    const U32 *compare_tokens,*compare_counts; // reference arm compares native list
    DsaDiagnostics* diagnostics; // caller clears; near tie: abs + rel*max(|cut|,|score|)
    DsaIndex index=DsaIndex::fp32;
    bool legacy_fp32_index=false; // a6d126c4 scalar projection/score comparison arm
    // Diagnostic IEEE model of the split, four-lane paired-half projection.
    // index_only leaves y untouched and skips attention and unrelated projections.
    bool node_mimic=false,index_only=false;
    const U32 *compare2_tokens=nullptr,*compare2_counts=nullptr;
    DsaDiagnostics* diagnostics2=nullptr;
    bool mma_index=false; // opt-in BF16 scorer and reduced-precision index projections
    // rows <= 32 (decode): the CUTLASS trellis projections (kva, qb, out; qa outside the FP32 index) as
    // row_mma.cuh strips from the packed tiles, without their layer decode; the same bits.
    bool row_kernels=false;
    unsigned row_r2=0; // RowR2 bits (opt-in): r2_pre A loads before the decode, r2_idx FP32 index row chains, r2_dense ordinary strips
    float near_tie_abs=1.e-3f,near_tie_rel=1.e-3f;
    bool short_skip=false; // decode caller opts in; prefill and diagnostics retain their existing work
    DsaIndexQueryProfile* index_query_profile=nullptr; // few-row FP32 decode only
};
// Caller creates timing events for phase*dsa_max_windows+window. decode/pool
// use only window zero. Every other phase has ceil(rows/64) event pairs.
struct DsaProfile {cudaEvent_t *begin,*end;};
// Output must not alias state. Contiguous node snapshot order: latent, index key, index gate, complete pools.
std::size_t dsa_export_bytes(unsigned frontier);
cudaError_t enqueue_dsa_export(DsaState,unsigned frontier,void* output,std::size_t bytes,cudaStream_t);
cudaError_t prepare_dsa();
std::size_t dsa_workspace_bytes(unsigned rows,unsigned capacity);
cudaError_t enqueue_dsa(const DsaCall&,void*,std::size_t,cudaStream_t,bool reference=false,DsaProfile* =nullptr);
struct DsaExactReport {std::uint64_t projection_elements=0,projection_differing=0,score_elements=0,score_differing=0,key_differing=0,memcmp_failures=0;};
// Diagnostic only: synchronizes/downloads bounded windows and uses host memcmp
// plus per-element bit counts. Reuses real layer weights, input and pooled state.
cudaError_t self_check_dsa_index(const DsaCall&,void*,std::size_t,cudaStream_t,DsaExactReport&);
}
