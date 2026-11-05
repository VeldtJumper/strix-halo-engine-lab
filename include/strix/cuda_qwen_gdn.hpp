// CUDA port of the Qwen3.8-Flash-Next GDN (gated delta net) prefill core, bit-identical to the HIP tiled prefill
// (src/hip_gdn_launcher.hip: gdn_prefill_gates_kernel<8,2> :2750, gdn_prefill_conv_kernel<8,32> :2807,
// gdn_prefill_lanes_kernel<key_value,4,8,8> :2896, gdn_prefill_norm_rows_kernel<8,A8> :3000).
//
// Host launch API on device pointers; no engine types. Shapes are the production ones (hidden 2560, 16 key heads and 48 value
// heads of 128, conv kernel 4). All small weights are FP32 (the production GGUF stores ssm_a/ssm_dt/ssm_conv1d/ssm_norm as F32;
// an F16/BF16 tensor can be widened on the host first, the widening is exact and is what HIP load_small does).
//
// Row workspace (one per prompt row, uniform stride in floats, like GdnPrefillRows): qkv[10240] (BF16-valued FP32 from the
// in projection; q 0..2047, k 2048..4095, v 4096..10239), z[6144] (BF16-valued FP32), beta[48], gate[48], core[6144].
// Row r of a field is field + r * stride. Kernels run in stream order; nothing is atomic.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "qwen_amd_math.cuh"

namespace strix::cuda::qwen_gdn {

constexpr std::uint32_t kHidden = 2560, kKeyHeads = 16, kValueHeads = 48, kHeadDim = 128, kConvKernel = 4;
constexpr std::uint32_t kKeyWidth = kKeyHeads * kHeadDim;                          // 2048
constexpr std::uint32_t kConvWidth = 2U * kKeyWidth + kValueHeads * kHeadDim;      // 10240
constexpr std::uint32_t kValueWidth = kValueHeads * kHeadDim;                      // 6144
constexpr std::uint64_t kMatrixFloats = std::uint64_t{kValueHeads} * kHeadDim * kHeadDim;  // per layer, key_value
constexpr std::uint64_t kConvStateHalfs = std::uint64_t{kConvWidth} * kConvKernel;          // per layer, BF16 [ch][4]
constexpr std::uint32_t kConvChunk = 32;                                           // rows per conv block
constexpr std::uint32_t kA8Pitch = 256;                                            // fused A8 delta pitch (kDenseA8WideRows)

// A device pointer given directly, or read at kernel start from a device word (`slot`, e.g. an entry of a per-segment
// descriptor table written before a graph replay) plus `offset` elements. Only where the pointer comes from changes; the
// arithmetic is the same.
template <class T> struct DevRef {
    T* direct = nullptr;
    T* const* slot = nullptr;
    std::uint64_t offset = 0;
    __host__ __device__ T* get() const { return slot ? *slot + offset : direct; }
};

struct Rows {
    float* qkv; float* z; float* beta; float* gate; float* core;
    std::uint64_t stride;  // floats between consecutive rows (same for every field)
};

struct Weights {
    const float* beta;    // ssm_beta  [48][2560]
    const float* alpha;   // ssm_alpha [48][2560]
    const float* conv;    // ssm_conv1d [10240][4]
    const float* dt;      // ssm_dt bias [48]
    const float* a;       // ssm_a [48]
    const float* norm;    // ssm_norm [128]
    float qk_l2_epsilon = 1e-6F;    // model_contract gdn_qk_l2_epsilon
    float rms_epsilon = 1e-6F;      // model_contract rms_norm_epsilon
};

// Fused dense-A8 quantizer of the out projection's input (gdn_prefill_norm_rows_kernel<8,true>): codes int8 [row][6144],
// deltas FP32 at [row / 256][192][row % 256] (row relative to the first row of the call, which must be a multiple of 256
// for the slot layout to match a 256-row projection launch).
struct A8Out { std::int8_t* codes; float* deltas; };

// Recurrence mapping: G lanes per state column (lane `sub` owns keys sub, sub+G, ...), blocks of 32 columns (32*G threads).
// Any G in {2,4,8,16} gives the same bits (the HIP tree pairs keys i and i+d, d = 64..1); default 4 (fastest at T=8192).
struct Tuning { std::uint32_t lanes = 4; };

// Scratch: conv = the 3 BF16 rows before each 32-row chunk; decay = expf(gate) per (row, head); core = both (256-aligned).
[[nodiscard]] std::size_t conv_scratch_bytes(std::uint32_t count);
[[nodiscard]] std::size_t decay_scratch_bytes(std::uint32_t count);
[[nodiscard]] std::size_t core_scratch_bytes(std::uint32_t count);

// K3: beta = round_bf16(1/(1+expf(-round_bf16(x.w_beta)))), gate = a * softplus(round_bf16(x.w_alpha) + dt).
// input: BF16 hidden rows, row r at input + r * input_stride (elements).
void launch_gates(const std::uint16_t* input, std::uint64_t input_stride, const Rows& rows, std::uint32_t count,
    const Weights& w, const amdmath::Tables& t, cudaStream_t stream);
// K4: causal conv1d(4) over the conv history, SiLU, BF16, q/k L2 norm (q also * rsqrtf(128)); qkv rewritten in place;
// conv_state BF16 [10240][4] read (taps 1..3) and replaced by the last 4 raw inputs. scratch: conv_scratch_bytes(count).
void launch_conv(const Rows& rows, std::uint32_t count, std::uint16_t* conv_state, void* scratch, const Weights& w,
    const amdmath::Tables& t, cudaStream_t stream);
// K5: delta rule over the rows in ascending order; state FP32 [48][128 key][128 value] read once and written once.
// scratch: decay_scratch_bytes(count).
void launch_recurrence(const Rows& rows, std::uint32_t count, float* state, void* scratch, const amdmath::Tables& t,
    cudaStream_t stream, Tuning tuning = {});
// K5 reading the state from state_in and writing the final state to state_out (may alias state_in; same bits). Decode
// windows keep S0 so a partial accept can replay the kept rows from it.
void launch_recurrence_split(const Rows& rows, std::uint32_t count, const float* state_in, float* state_out, void* scratch,
    const amdmath::Tables& t, cudaStream_t stream, Tuning tuning = {});
// K6: gated RMSNorm of core in place (FP32), optionally also the fused A8 codes/deltas.
void launch_norm(const Rows& rows, std::uint32_t count, const Weights& w, const amdmath::Tables& t, cudaStream_t stream,
    const A8Out* a8 = nullptr);
// K3..K6 in order. scratch: core_scratch_bytes(count).
void launch_core(const std::uint16_t* input, std::uint64_t input_stride, const Rows& rows, std::uint32_t count,
    std::uint16_t* conv_state, float* state, void* scratch, const Weights& w, const amdmath::Tables& t, cudaStream_t stream,
    const A8Out* a8 = nullptr, Tuning tuning = {});

// launch_core with the split recurrence; raw_qkv (optional, [count][10240] FP32) receives the conv inputs (the rows'
// qkv before K4 rewrites them in place) for a later conv-state rollback.
void launch_core_split(const std::uint16_t* input, std::uint64_t input_stride, const Rows& rows, std::uint32_t count,
    std::uint16_t* conv_state, const float* state_in, float* state_out, float* raw_qkv, void* scratch, const Weights& w,
    const amdmath::Tables& t, cudaStream_t stream, const A8Out* a8 = nullptr, Tuning tuning = {});

// launch_core_split with every session pointer as a DevRef (decode graphs over a per-segment descriptor table).
void launch_core_split_ref(const std::uint16_t* input, std::uint64_t input_stride, const Rows& rows, std::uint32_t count,
    DevRef<std::uint16_t> conv_state, DevRef<float> state_in, DevRef<float> state_out, float* raw_qkv, void* scratch,
    const Weights& w, const amdmath::Tables& t, cudaStream_t stream, const A8Out* a8 = nullptr, Tuning tuning = {});

// Multi-segment core (decode cohort windows): the window's rows hold several sessions' segments back to back
// (segment i = rows [row, row + count), count <= 32, segments ascending from row 0). Row-wise steps (K3 gates, decay,
// K6 norm) run once over all rows; the steps that carry a session's history (K4 conv prepass + conv, K5 recurrence)
// run with one grid slice per segment, each slice exactly the single-segment launch_core_split_ref of its segment (same
// per-row and per-head arithmetic, its own conv history and recurrent state). raw_qkv receives all rows' conv inputs.
// scratch: multi_scratch_bytes().
inline constexpr std::uint32_t kMaxSegments = 8;
struct Segment {
    std::uint32_t row = 0, count = 0;
    DevRef<std::uint16_t> conv;
    DevRef<float> state_in, state_out;
};
struct Segments {
    std::uint32_t n = 0;
    Segment s[kMaxSegments];
};
[[nodiscard]] std::size_t multi_scratch_bytes();
void launch_core_multi(const std::uint16_t* input, std::uint64_t input_stride, const Rows& rows, const Segments& segments,
    float* raw_qkv, void* scratch, const Weights& w, const amdmath::Tables& t, cudaStream_t stream, Tuning tuning = {});

// conv_state = the history after the first `keep` rows of a window whose pre-window history was `before` and whose raw
// conv inputs are raw_qkv [rows][10240] (launch_core_split). Same values the window's K4 prepass writes for count = keep.
void launch_conv_rollback(const float* raw_qkv, std::uint32_t keep, const std::uint16_t* before, std::uint16_t* conv_state,
    cudaStream_t stream);

}  // namespace strix::cuda::qwen_gdn
