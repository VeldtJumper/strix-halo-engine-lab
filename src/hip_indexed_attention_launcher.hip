#include "strix/qsa_prefill_stage_plan.hpp"
#include "strix/prefill_identity.hpp"
#include "strix/known_prefill_limits.hpp"
#include "hip_prefill_pool.hpp"
#include "strix/qpf2k_defaults.hpp"
#include "strix/qsa_cache_only.hpp"
#include "strix/qsa_committed_pool.hpp"
#include "strix/hip_qwen_rotary_table.hpp"
#include "strix/hip_qwen_rotary_diagnostic.hpp"
#if defined(STRIX_BUILD_QWEN_ROTARY_DIAGNOSTIC)
#include "../tests/hip_qwen_rotary_watchdog.hpp"
#endif
#include "strix/moe_q4k_a8_contract.hpp"
#include "strix/known_window_cohort.hpp"
#include "strix/hip_ggml_weight_formats.hpp"
#include "strix/hip_indexed_attention_launcher.hpp"
#include "hip_q8_rows_gemv.hpp"
#include "hip_dense_f16_tiled.hpp"
#include "hip_rounded_stage_image.hpp"
#include "strix/hip_compact_qsa_diagnostic.hpp"
#include "strix/hip_mtp_attention.hpp"
#include "strix/qpersist.hpp"
#include "strix/hip_qpersist_diagnostic.hpp"
#include "strix/hip_mtp_fast_draft.hpp"
#include "strix/hip_mtp_native_validation.hpp"

#include "strix/dispatch_resources.hpp"
#include "strix/prepared_tensor_views.hpp"
#include "strix/indexed_attention_state.hpp"
#include "strix/indexed_attention_workspace.hpp"
#include "strix/ctx_exact_memory.hpp"
#include "strix/qsa_score_scratch.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/session_state_memory.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/graph_operator_control.hpp"
#include "strix/q8_projection_organization.hpp"
#include "strix/hip_qsa_selection_diagnostic.hpp"
#include "strix/deferred_moe_status.hpp"
#include "strix/hip_qsa_gqa_reuse_diagnostic.hpp"
#include "strix/execution_plan.hpp"

#include "strix/hip_qwen_rotary_diagnostic.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <hip/hip_fp16.h>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <span>
#include <string_view>
#include <tuple>
#include <variant>
#include <type_traits>
#include <utility>

namespace strix {
namespace {
struct HipQsaScratchBackend {
    using Event=hipEvent_t;
    static void check(hipError_t e) {
        if(e!=hipSuccess) throw std::runtime_error(std::string("QSA scratch: ")+hipGetErrorString(e));
    }
    int device() const { int d=-1;check(hipGetDevice(&d));return d; }
    void activate(int d) { check(hipSetDevice(d)); }
    void* allocate(std::uint64_t bytes) { void* p=nullptr;check(hipMalloc(&p,bytes));return p; }
    Event event() { Event e=nullptr;check(hipEventCreateWithFlags(&e,hipEventDisableTiming));return e; }
    void wait(Event e) { check(hipEventSynchronize(e)); }
    void record(Event e,std::uintptr_t stream) { check(hipEventRecord(e,reinterpret_cast<hipStream_t>(stream))); }
    void drain(std::uintptr_t stream) { check(hipStreamSynchronize(reinterpret_cast<hipStream_t>(stream))); }
    void destroy(Event e) { check(hipEventDestroy(e)); }
    void free(void* p) { check(hipFree(p)); }
};
struct HipQsaSubmissionCache {
    QsaScratchSerialGate gate;
    QsaScratchHandoff handoff;
    hipEvent_t done=nullptr;
    int device=-1;
    auto enter(int d) {
        auto lock=gate.enter(d);
        if(!done) { done=HipQsaScratchBackend{}.event();device=d; }
        return lock;
    }
    ~HipQsaSubmissionCache() {
        if(!done) return;
        try {
            HipQsaScratchBackend backend;
            const auto prior=backend.device();backend.activate(device);
            if(handoff.pending()) backend.wait(done);
            backend.destroy(done);backend.activate(prior);
        } catch(...) { std::terminate(); }
    }
};
HipQsaSubmissionCache& qsa_scratch_gate() { static HipQsaSubmissionCache cache;return cache; }
}
class HipQsaExactScoreOwner {
    HipSetupQuiescenceGuard guard_;
    std::uint64_t generation_;
public:
    QsaScoreScratch<HipQsaScratchBackend> scratch;
    HipQsaExactScoreOwner(std::uint64_t maximum,std::uint64_t bytes,void* stream,std::uint64_t generation,std::uint32_t rows)
        :generation_(generation),scratch(HipQsaScratchBackend{}, {guard_.device_index(),reinterpret_cast<std::uintptr_t>(stream),generation,maximum},bytes,rows) {}
    void validate_generation(std::uint64_t generation) const {
        if(generation!=generation_) throw std::invalid_argument("QSA scratch arena generation differs from setup");
    }
    void validate(std::uint64_t maximum,std::uint64_t generation) const {
        guard_.require_original();
        scratch.validate({guard_.device_index(),reinterpret_cast<std::uintptr_t>(hip_execution_stream()),generation,maximum});
    }
};
std::shared_ptr<HipQsaExactScoreOwner> make_hip_qsa_exact_score_owner(
    std::uint64_t maximum,std::uint64_t bytes,void* stream,std::uint64_t generation,std::uint32_t rows) {
    const auto native_stream=static_cast<hipStream_t>(stream);
    if(!native_stream || native_stream==hipStreamLegacy || native_stream==hipStreamPerThread)
        throw std::invalid_argument("QSA scratch setup requires an owned eager stream");
    hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
    HipQsaScratchBackend::check(hipStreamIsCapturing(native_stream,&capture));
    if(capture!=hipStreamCaptureStatusNone) throw std::invalid_argument("QSA scratch setup cannot run during capture");
    const auto lock=qsa_scratch_gate().enter(HipQsaScratchBackend{}.device());
    return std::make_shared<HipQsaExactScoreOwner>(maximum,bytes,stream,generation,rows);
}
namespace {

// An empty Source pack preserves the native kernel ABI and arithmetic. Only
// table instances carry this additional argument; no native runtime branch.
struct QwenRotarySource {
    const float2* cs;
    std::uint32_t extent;
    std::uint32_t* status = nullptr;
    __device__ bool check(std::uint64_t position) const {
        // Uniform workgroup failure before any output. Never substitute row zero.
        if (position >= extent) {
            if (status) { atomicExch(status, 1U); return false; }
            __builtin_trap();
        }
        return true;
    }
};
template <class F>
void with_qwen_rotary(const ModelContract& model, hipStream_t stream, F&& launch) {
    require_qwen_rotary_device_payload(model.rotary_identity(),model.qwen_rotary_table.get());
    if (const auto* table = model.qwen_rotary_table.get()) {
        if (model.rotary_dim() != 64 || model.rope_theta != 10000000.0F || !table->sha256().valid())
            throw std::invalid_argument("YaRN device table/model mismatch");
        validate_qwen_rotary_reader(*table, stream);
        // Arm captured graph ownership before recording any kernel. Eager
        // event leases must follow all readers, including a partially failed launch.
        hipStreamCaptureStatus status{};
        if (hipStreamIsCapturing(stream, &status) != hipSuccess)
            throw std::runtime_error("cannot inspect rotary reader stream capture");
        if (status == hipStreamCaptureStatusActive) retain_qwen_rotary_reader(model.qwen_rotary_table, stream);
        try {
            launch(QwenRotarySource{static_cast<const float2*>(table->data()), table->extent()});
        } catch (...) {
            if (status != hipStreamCaptureStatusActive) retain_qwen_rotary_reader(model.qwen_rotary_table, stream);
            throw;
        }
        if (status != hipStreamCaptureStatusActive) retain_qwen_rotary_reader(model.qwen_rotary_table, stream);
    } else launch();
}
void check_qwen_rotary_position(const ModelContract& model, std::uint64_t position) {
    if (model.qwen_rotary_table) model.qwen_rotary_table->check_position(position);
}

constexpr std::uint32_t kThreads = 256;
constexpr std::uint64_t kLauncherTag =
    0x5141534154544e32ULL;
constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

// Attention audit items, each bitwise and ON by default (=0 turns one off):
// STRIX_QSA_SKIP_INDEX_Q (item 6): no index query projection / norm when no row scores blocks
// (every row's complete blocks <= block_topk: the selection takes every visible token);
// STRIX_QSA_COHORT_POOL (item 3): cohort windows score blocks from each owner's persistent pool
// (only with STRIX_QSA_POOL_SLAB: per-layer pool allocations during decode lose 4-15 % at N8);
// STRIX_QSA_COHORT_TAIL (item 8): cohort windows run the fused softmax + value/gate tail.
bool qsa_env_default_on(const char* name) { const char* v = std::getenv(name); return !v || v[0] != '0'; }
bool qsa_skip_index_q_enabled() { static const bool on = qsa_env_default_on("STRIX_QSA_SKIP_INDEX_Q"); return on; }
bool qsa_cohort_pool_enabled() { static const bool on = qsa_env_default_on("STRIX_QSA_COHORT_POOL"); return on; }
bool qsa_cohort_tail_enabled() { static const bool on = qsa_env_default_on("STRIX_QSA_COHORT_TAIL"); return on; }

struct Q8Block {
    __half scale;
    std::int8_t values[32];
};

struct IQ4XSBlock {
    __half d;
    std::uint16_t scales_h;
    std::uint8_t scales_l[4];
    std::uint8_t qs[128];
};

static_assert(sizeof(Q8Block) == 34);
static_assert(sizeof(IQ4XSBlock) == 136);

__device__ __constant__
std::int8_t iq4_codebook[16] = {
    -127, -104, -83, -65,
    -49, -35, -22, -10,
       1,   13,  25,  38,
      53,   69,  89, 113,
};

enum class WeightKind {
    f32,
    f16,
    bf16,
    q8_0,
    iq4_xs,
#define STRIX_GGML_KIND_ENUM(kind, type) kind,
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_ENUM) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_ENUM) STRIX_GGML_KIND_ENUM(iq4_nl, 20)
#undef STRIX_GGML_KIND_ENUM
};

enum class SmallWeightKind {
    f32,
    f16,
    bf16,
};

struct GraphPositionKernelArguments {
    const std::uint64_t* position = nullptr;
    std::uint64_t context_tokens = 0;
    std::uint32_t* status = nullptr;
};

struct GraphSelectionArguments {
    GraphPositionKernelArguments position;
    std::uint32_t ratio = 0;
    std::uint32_t block_topk = 0;
};

void check_hip(hipError_t status, const char* operation);

GraphPositionKernelArguments graph_position_arguments() {
    const auto* scope = current_hip_launch_context();
    if (scope == nullptr || !scope->graph_capture) return {};
    return {hip_graph_position(), scope->context_tokens, hip_graph_status()};
}

__device__ bool graph_visible(GraphSelectionArguments dynamic, std::uint32_t& visible) {
    if (dynamic.position.position == nullptr) return true;
    const auto position = *dynamic.position.position;
    // Other blocks can publish arithmetic failure concurrently. Broadcast once
    // per block so a status transition cannot split threads around barriers.
    __shared__ std::uint32_t reject;
    if (threadIdx.x == 0) reject = atomicAdd(dynamic.position.status, 0U) != 0 || position >= dynamic.position.context_tokens;
    __syncthreads();
    if (reject) return false;
    visible = static_cast<std::uint32_t>(position + 1ULL);
    return true;
}

__device__ bool graph_selected_count(GraphSelectionArguments dynamic, std::uint32_t& count) {
    if (dynamic.position.position == nullptr) return true;
    std::uint32_t visible = 0;
    if (!graph_visible(dynamic, visible)) return false;
    const auto blocks = visible / dynamic.ratio;
    count = min(blocks, dynamic.block_topk) * dynamic.ratio + visible % dynamic.ratio;
    return true;
}

__global__ void validate_indexed_graph_control(GraphPositionKernelArguments dynamic,
    const std::uint16_t* input, std::uint32_t hidden) {
    if (atomicAdd(dynamic.status, 0U) != 0) return;
    if (*dynamic.position >= dynamic.context_tokens) atomicOr(dynamic.status, 2U);
    for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        if ((input[i] & 0x7f80U) == 0x7f80U) atomicOr(dynamic.status, 4U);
    }
}

__global__ void validate_indexed_graph_values(const float* values, std::uint32_t count,
    std::uint32_t* status) {
    if (atomicAdd(status, 0U) != 0) return;
    for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        if (!isfinite(values[i])) atomicOr(status, 8U);
    }
}

void validate_graph_values(const float* values, std::uint32_t count) {
    if (hip_graph_status() == nullptr) return;
    hipLaunchKernelGGL(validate_indexed_graph_values, dim3(1), dim3(kThreads), 0,
        hip_execution_stream(), values, count, hip_graph_status());
    check_hip(hipGetLastError(), "HIP indexed-attention finite intermediate validation");
}

struct TensorPair {
    const ManifestTensor* manifest = nullptr;
    const RuntimeTensorBinding* runtime = nullptr;
};

struct IndexedTensors {
    TensorPair q;
    TensorPair k;
    TensorPair v;
    TensorPair out;
    TensorPair q_norm;
    TensorPair k_norm;
    TensorPair index_q;
    TensorPair index_k;
    TensorPair index_q_norm;
    TensorPair index_k_norm;
};

struct WorkspacePointers {
    float* index_query = nullptr;
    float* current_index_key = nullptr;
    float* block_scores = nullptr;
    std::uint32_t* selected_indices = nullptr;
    float* q_gate = nullptr;
    float* current_k = nullptr;
    float* current_v = nullptr;
    float* attention_scores = nullptr;
    float* attention_output = nullptr;
    qsa_selection_diagnostic::Item* selection_ping = nullptr;
    qsa_selection_diagnostic::Item* selection_pong = nullptr;
    std::uint32_t* selection_status = nullptr;
    std::uint64_t* attention_position = nullptr;
    bool compact_attention_output = false;
};

void hash_u64(
    std::uint64_t& hash,
    std::uint64_t value) noexcept {
    for (unsigned shift = 0;
         shift < 64;
         shift += 8) {
        hash ^= (value >> shift) & 0xffULL;
        hash *= fnv_prime;
    }
}

void hash_float(
    std::uint64_t& hash,
    float value) noexcept {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    hash_u64(hash, bits);
}

void check_hip(
    hipError_t error,
    const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) +
            ": " +
            hipGetErrorString(error));
    }
}

std::uint32_t checked_u32(
    std::uint64_t value,
    const char* name) {
    if (value == 0 ||
        value >
            std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            std::string(name) +
            " does not fit uint32");
    }
    return static_cast<std::uint32_t>(value);
}

void validate_config(
    const HipIndexedAttentionLauncherConfig& config) {
    if (config.dense_trellis_projections && (!config.dense_a8_projections || config.enable_device_graph_inputs))
        throw std::invalid_argument("dense trellis QSA requires its eager dense-A8 parent recipe");
#if !defined(STRIX_HAS_EXPERIMENTAL_DENSE_TRELLIS)
    if (config.dense_trellis_projections) throw std::invalid_argument("dense trellis runtime is disabled in this build");
#endif
    if (const auto error =
            config.model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }
    qsa_input_prefetch::validate_layers(config.model,config.prefetch_q8_input,config.prefetch_eligible_layers);
    (void)qsa_gqa_organization_marker(config.qsa_gqa_heads_per_block);
    if (config.qsa_gqa_heads_per_block != 1 && (config.enable_device_graph_inputs || config.model.head_dim != 256 ||
        config.model.attention_heads > 64 || config.model.hidden_layers > 48 ||
        config.model.indexer_budget != 2048 || config.model.indexer_compress_ratio != 4))
        throw std::invalid_argument("QSA GQA organization requires eager head256/budget2048/compression4");
    if (config.qsa_gqa_heads_per_block != 1 && config.model.attention_kv_storage != AttentionKvStorage::bf16)
        throw std::invalid_argument("QSA GQA organization reads the BF16 KV cache only");
    if (config.qsa_hierarchical_selection && (config.enable_device_graph_inputs ||
        config.model.indexer_budget != 2048 || config.model.indexer_compress_ratio != 4 ||
        config.model.hidden_layers > 48))
        throw std::invalid_argument("hierarchical QSA requires eager budget2048/compression4 and at most48 layers");
    if (config.activation !=
        HipActivationStorage::bf16) {
        throw std::invalid_argument(
            "HIP indexed-attention v1 targets canonical BF16 activations only");
    }
    if (config.model.attention_heads %
            config.model.kv_heads !=
        0) {
        throw std::invalid_argument(
            "attention heads must divide evenly across KV heads");
    }
    if (config.model.head_dim != kThreads ||
        config.model.indexer_head_dim >
            kThreads ||
        config.model.rotary_dim() >
            config.model.indexer_head_dim ||
        config.model.rotary_dim() >
            config.model.head_dim) {
        throw std::invalid_argument(
            "HIP indexed-attention v1 requires pinned/tileable head geometry");
    }
    if (config.wave8_q8_output) {
        if (config.enable_device_graph_inputs)
            throw std::invalid_argument("wave8 QSA output supports eager execution only");
        if (config.model.attention_heads > 10240U / config.model.head_dim)
            throw std::invalid_argument("wave8 QSA output width exceeds admitted bounds");
        q8_projection_organization::validate_geometry(
            config.model.attention_heads * config.model.head_dim, config.model.hidden_size);
    }
}

WeightKind weight_kind(
    std::uint32_t type) {
    switch (type) {
    case 0: return WeightKind::f32;
    case 1: return WeightKind::f16;
    case 8: return WeightKind::q8_0;
    case 23: return WeightKind::iq4_xs;
#define STRIX_GGML_KIND_MAP(kind, type) case type: return WeightKind::kind;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_MAP) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_MAP) STRIX_GGML_KIND_MAP(iq4_nl, 20)
#undef STRIX_GGML_KIND_MAP
    case 30: return WeightKind::bf16;
    default:
        throw std::invalid_argument(
            "HIP indexed-attention projection supports F32, F16, BF16, Q8_0 and IQ4_XS");
    }
}

SmallWeightKind small_weight_kind(
    std::uint32_t type) {
    switch (type) {
    case 0: return SmallWeightKind::f32;
    case 1: return SmallWeightKind::f16;
    case 30: return SmallWeightKind::bf16;
    default:
        throw std::invalid_argument(
            "HIP indexed-attention norm supports F32, F16 and BF16");
    }
}

__device__ float bf16_to_float(
    std::uint16_t bits) {
    return __uint_as_float(
        static_cast<std::uint32_t>(bits) << 16U);
}

__device__ std::uint16_t float_to_bf16(
    float value) {
    const auto bits = __float_as_uint(value);
    const auto rounded =
        bits +
        0x7fffU +
        ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(
        rounded >> 16U);
}

__device__ float round_bf16(
    float value) {
    return bf16_to_float(
        float_to_bf16(value));
}

// kv-int8-tok-v1 (AttentionKvStorage::int8_token). The KV row of one token and layer is
// [K codes kv_width | V codes kv_width | FP32 scales: K heads, then V heads]. Per (token, KV
// head, half) the codes are rint(x / s) clamped to [-127, 127] of the BF16-rounded values x,
// with s = max|x| / 127 (s = 0 and codes 0 for an all-zero head). Every reader uses
// float(code) * s exactly where the BF16 cache uses the BF16 value, so each product, tree and
// chain keeps its order and all score/value kernel forms stay equal to one another.
__host__ __device__ constexpr std::uint64_t kv_int8_row_bytes(std::uint32_t kv_width, std::uint32_t kv_heads) {
    return 2ULL * kv_width + 8ULL * kv_heads;
}

__device__ __forceinline__ float kv_int8_load(const void* state, std::uint64_t token, std::uint32_t kv_width,
    std::uint32_t kv_heads, std::uint32_t half, std::uint32_t kv_head, std::uint32_t head_dim, std::uint32_t feature) {
    const auto* row = static_cast<const std::uint8_t*>(state) + token * kv_int8_row_bytes(kv_width, kv_heads);
    const auto code = reinterpret_cast<const std::int8_t*>(row)[half * kv_width + kv_head * head_dim + feature];
    const auto scale = reinterpret_cast<const float*>(row + 2ULL * kv_width)[half * kv_heads + kv_head];
    return static_cast<float>(code) * scale;
}

// One block of kThreads threads (thread = feature, head_dim <= kThreads) quantizes one KV
// head half of `source` (that half's kv_width FP32 values) into its token row. The maximum
// is order-independent, so every writer (decode, rows, prefill) stores the same bytes.
__device__ __forceinline__ void kv_int8_store_head(const float* source, void* state, std::uint64_t token,
    std::uint32_t kv_width, std::uint32_t kv_heads, std::uint32_t half, std::uint32_t kv_head, std::uint32_t head_dim) {
    __shared__ float maximum[kThreads];
    const auto feature = static_cast<std::uint32_t>(threadIdx.x);
    const float x = feature < head_dim ? round_bf16(source[kv_head * head_dim + feature]) : 0.0F;
    maximum[feature] = fabsf(x);
    __syncthreads();
    for (std::uint32_t stride = kThreads / 2U; stride > 0U; stride >>= 1U) {
        if (feature < stride) maximum[feature] = fmaxf(maximum[feature], maximum[feature + stride]);
        __syncthreads();
    }
    const float scale = maximum[0] / 127.0F;
    auto* row = static_cast<std::uint8_t*>(state) + token * kv_int8_row_bytes(kv_width, kv_heads);
    if (feature < head_dim) {
        const float code = scale > 0.0F ? fminf(fmaxf(rintf(x / scale), -127.0F), 127.0F) : 0.0F;
        reinterpret_cast<std::int8_t*>(row)[half * kv_width + kv_head * head_dim + feature] = static_cast<std::int8_t>(code);
    }
    if (feature == 0U) reinterpret_cast<float*>(row + 2ULL * kv_width)[half * kv_heads + kv_head] = scale;
}

__device__ float load_activation(
    const std::uint16_t* value) {
    return bf16_to_float(*value);
}


__device__ std::uint8_t iq4_local_scale(
    const IQ4XSBlock& block,
    std::uint32_t subblock) {
    const auto low_byte =
        block.scales_l[subblock / 2U];
    const auto low =
        static_cast<std::uint8_t>(
            (low_byte >>
             (4U * (subblock % 2U))) &
            0x0fU);
    const auto high =
        static_cast<std::uint8_t>(
            (block.scales_h >>
             (2U * subblock)) &
            0x03U);
    return static_cast<std::uint8_t>(
        low |
        static_cast<std::uint8_t>(
            high << 4U));
}

template <WeightKind Kind>
__device__ float load_weight(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature);

template <>
__device__ float load_weight<WeightKind::f32>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    return static_cast<const float*>(weights)[
        row * row_width + feature];
}

template <>
__device__ float load_weight<WeightKind::f16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const __half*>(weights);
    return __half2float(
        values[row * row_width + feature]);
}

template <>
__device__ float load_weight<WeightKind::bf16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const std::uint16_t*>(weights);
    return bf16_to_float(
        values[row * row_width + feature]);
}

template <>
__device__ float load_weight<WeightKind::q8_0>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const Q8Block*>(weights);
    const auto blocks_per_row =
        row_width / 32U;
    const auto& block =
        blocks[
            row * blocks_per_row +
            feature / 32U];
    return __half2float(block.scale) *
           static_cast<float>(
               block.values[
                   feature % 32U]);
}

template <>
__device__ float load_weight<WeightKind::iq4_xs>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const IQ4XSBlock*>(weights);
    const auto blocks_per_row =
        row_width / 256U;
    const auto& block =
        blocks[
            row * blocks_per_row +
            feature / 256U];

    const auto within =
        feature % 256U;
    const auto subblock =
        within / 32U;
    const auto suboffset =
        within % 32U;
    const auto packed =
        block.qs[
            subblock * 16U +
            suboffset % 16U];
    const auto code =
        suboffset < 16U
            ? static_cast<std::uint8_t>(
                  packed & 0x0fU)
            : static_cast<std::uint8_t>(
                  packed >> 4U);
    const int local =
        static_cast<int>(
            iq4_local_scale(
                block,
                subblock)) -
        32;

    return __half2float(block.d) *
           static_cast<float>(local) *
           static_cast<float>(
               iq4_codebook[code]);
}

template <SmallWeightKind Kind>
__device__ float load_small_weight(
    const void* weights,
    std::uint32_t index);

template <>
__device__ float load_small_weight<
    SmallWeightKind::f32>(
    const void* weights,
    std::uint32_t index) {
    return static_cast<const float*>(
        weights)[index];
}

template <>
__device__ float load_small_weight<
    SmallWeightKind::f16>(
    const void* weights,
    std::uint32_t index) {
    return __half2float(
        static_cast<const __half*>(
            weights)[index]);
}

template <>
__device__ float load_small_weight<
    SmallWeightKind::bf16>(
    const void* weights,
    std::uint32_t index) {
    return bf16_to_float(
        static_cast<
            const std::uint16_t*>(
                weights)[index]);
}

#define STRIX_GGML_KIND_LOAD(kind, type)                                                          \
    template <>                                                                                   \
    __device__ float load_weight<WeightKind::kind>(                                               \
        const void* weights, std::uint32_t row_width, std::uint64_t row, std::uint32_t feature) { \
        return hip_ggml::element<type>(weights, row_width, row, feature);                        \
    }
STRIX_GGML_KINDS_K(STRIX_GGML_KIND_LOAD) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_LOAD) STRIX_GGML_KIND_LOAD(iq4_nl, 20)
#undef STRIX_GGML_KIND_LOAD

std::uint32_t ggml_type_of(WeightKind kind) {
    switch (kind) {
#define STRIX_GGML_KIND_TYPE(k, type) case WeightKind::k: return type;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_TYPE) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_TYPE) STRIX_GGML_KIND_TYPE(iq4_nl, 20)
#undef STRIX_GGML_KIND_TYPE
    default: return 0;
    }
}

std::uint64_t required_weight_bytes(
    WeightKind kind,
    std::uint32_t input,
    std::uint64_t output) {
    switch (kind) {
    case WeightKind::f32:
        return static_cast<std::uint64_t>(
                   input) *
               output * 4ULL;
    case WeightKind::f16:
    case WeightKind::bf16:
        return static_cast<std::uint64_t>(
                   input) *
               output * 2ULL;
    case WeightKind::q8_0:
        if (input % 32U != 0) {
            throw std::invalid_argument(
                "Q8_0 indexed-attention input width must divide 32");
        }
        return static_cast<std::uint64_t>(
                   input / 32U) *
               output *
               sizeof(Q8Block);
    case WeightKind::iq4_xs:
        if (input % 256U != 0) {
            throw std::invalid_argument(
                "IQ4_XS indexed-attention input width must divide 256");
        }
        return static_cast<std::uint64_t>(
                   input / 256U) *
               output *
               sizeof(IQ4XSBlock);
    default: {
        const auto g = hip_ggml::geometry(ggml_type_of(kind));
        if (g.elements == 0 || input % g.elements != 0) {
            throw std::invalid_argument("weight width is not a multiple of its GGUF block");
        }
        return static_cast<std::uint64_t>(input / g.elements) * output * g.bytes;
    }
    }
    throw std::logic_error(
        "unreachable indexed-attention weight kind");
}

std::uint64_t required_small_weight_bytes(
    SmallWeightKind kind,
    std::uint64_t elements) {
    switch (kind) {
    case SmallWeightKind::f32:
        return elements * 4ULL;
    case SmallWeightKind::f16:
    case SmallWeightKind::bf16:
        return elements * 2ULL;
    }
    throw std::logic_error(
        "unreachable indexed-attention norm kind");
}

template <WeightKind Kind>
__global__ void linear_to_float_kernel(
    const std::uint16_t* input,
    const void* weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    float* output) {
    const auto out =
        static_cast<std::uint32_t>(
            blockIdx.x);
    if (out >= output_width) {
        return;
    }

    __shared__ float scratch[kThreads];
    float sum = 0.0F;
    for (std::uint32_t feature =
             static_cast<std::uint32_t>(
                 threadIdx.x);
         feature < input_width;
         feature += kThreads) {
        sum +=
            load_activation(
                &input[feature]) *
            load_weight<Kind>(
                weights,
                input_width,
                out,
                feature);
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();

    for (std::uint32_t stride =
             kThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] +=
                scratch[
                    threadIdx.x +
                    stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[out] =
            round_bf16(
                scratch[0]);
    }
}

// Preserve the actual legacy LDS-tree operand priority: upper/partner src0,
// lower/self src1. Ordinary commutative lowering can select another NaN payload.
// Only the two new QSA input trees call this helper; old kernels remain literal.
__device__ __forceinline__ float qsa_input_legacy_tree_add(float lower, float upper) {
    float result;
    asm volatile("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(upper), "v"(lower));
    return result;
}

// Eight ascending virtual leaves per physical lane preserve the original
// 256-leaf tree, including its operand order and BF16-in-FP32 epilogue.
__global__ __launch_bounds__(32)
void linear_virtual256_q8_qsa_input_kernel(
    const std::uint16_t* input, const void* weights,
    std::uint32_t input_width, std::uint32_t output_width, float* output) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x);
    if (row >= output_width) return;
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    float partial[8]{};
    for (std::uint64_t chunk = 0; chunk < input_width; chunk += 256U) {
#pragma unroll
        for (unsigned local = 0; local < 8; ++local) {
            const auto feature = chunk + 8U * lane + local;
            if (feature < input_width)
                partial[local] += bf16_to_float(input[feature]) *
                    load_weight<WeightKind::q8_0>(weights, input_width, row,
                        static_cast<std::uint32_t>(feature));
        }
    }
#pragma unroll
    for (unsigned distance = 16; distance > 0; distance /= 2) {
#pragma unroll
        for (unsigned local = 0; local < 8; ++local) {
            const auto partner = __shfl_down(partial[local], distance, 32);
            if (lane < distance) partial[local] = qsa_input_legacy_tree_add(partial[local], partner);
        }
    }
    if (lane == 0) {
        partial[0] = qsa_input_legacy_tree_add(partial[0], partial[4]); partial[1] = qsa_input_legacy_tree_add(partial[1], partial[5]);
        partial[2] = qsa_input_legacy_tree_add(partial[2], partial[6]); partial[3] = qsa_input_legacy_tree_add(partial[3], partial[7]);
        partial[0] = qsa_input_legacy_tree_add(partial[0], partial[2]); partial[1] = qsa_input_legacy_tree_add(partial[1], partial[3]);
        partial[0] = qsa_input_legacy_tree_add(partial[0], partial[1]);
        output[row] = round_bf16(partial[0]);
    }
}

// Distinct opt-in load organization; existing virtual and legacy kernels
// remain unchanged. Actual ISA, not these arrays, qualifies load batching.
__global__ __launch_bounds__(32)
void linear_prefetch8_q8_qsa_input_kernel(
    const std::uint16_t* input, const void* weights,
    std::uint32_t input_width, std::uint32_t output_width, float* output) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x);
    if (row >= output_width) return;
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    float partial[8]{};
    const auto* blocks = static_cast<const Q8Block*>(weights) +
        static_cast<std::uint64_t>(row) * (input_width / 32U);
    for (std::uint64_t chunk = 0; chunk < input_width; chunk += 256U) {
        const auto base = chunk + 8U * lane;
        // The prelaunch Q8 width%32 gate proves all eight leaves are valid
        // together and remain inside one block. No tail load is speculative.
        if (base < input_width) {
            const auto feature = static_cast<std::uint32_t>(base);
            const auto& block = blocks[feature / 32U];
            std::uint16_t bits[8];
            std::int8_t codes[8];
#pragma unroll
            for (unsigned local = 0; local < 8; ++local) {
                bits[local] = input[feature + local];
                codes[local] = block.values[feature % 32U + local];
            }
            const float scale = __half2float(block.scale);
#pragma unroll
            for (unsigned local = 0; local < 8; ++local)
                partial[local] += bf16_to_float(bits[local]) *
                    (scale * static_cast<float>(codes[local]));
        }
    }
#pragma unroll
    for (unsigned distance = 16; distance > 0; distance /= 2) {
#pragma unroll
        for (unsigned local = 0; local < 8; ++local) {
            const auto partner = __shfl_down(partial[local], distance, 32);
            if (lane < distance) partial[local] = qsa_input_legacy_tree_add(partial[local], partner);
        }
    }
    if (lane == 0) {
        partial[0] = qsa_input_legacy_tree_add(partial[0], partial[4]); partial[1] = qsa_input_legacy_tree_add(partial[1], partial[5]);
        partial[2] = qsa_input_legacy_tree_add(partial[2], partial[6]); partial[3] = qsa_input_legacy_tree_add(partial[3], partial[7]);
        partial[0] = qsa_input_legacy_tree_add(partial[0], partial[2]); partial[1] = qsa_input_legacy_tree_add(partial[1], partial[3]);
        partial[0] = qsa_input_legacy_tree_add(partial[0], partial[1]);
        output[row] = round_bf16(partial[0]);
    }
}

template <WeightKind Kind>
__global__ void linear_float_to_bf16_kernel(
    const float* input,
    const void* weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    std::uint16_t* output) {
    const auto out =
        static_cast<std::uint32_t>(
            blockIdx.x);
    if (out >= output_width) {
        return;
    }

    __shared__ float scratch[kThreads];
    float sum = 0.0F;
    for (std::uint32_t feature =
             static_cast<std::uint32_t>(
                 threadIdx.x);
         feature < input_width;
         feature += kThreads) {
        sum +=
            input[feature] *
            load_weight<Kind>(
                weights,
                input_width,
                out,
                feature);
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();

    for (std::uint32_t stride =
             kThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] +=
                scratch[
                    threadIdx.x +
                    stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[out] =
            float_to_bf16(
                scratch[0]);
    }
}

// Actual original Q8 output LDS tree uses upper/partner src0 and lower/self
// src1 at every 128..1 stage. Keep this output-specific evidence and helper
// separate from the input projection tree and leave the old body unchanged.
__device__ __forceinline__ float qsa_output_legacy_tree_add(float lower, float upper) {
    float result;
    asm volatile("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(upper), "v"(lower));
    return result;
}

__global__ __launch_bounds__(256)
void linear_prefetch8_wave8_q8_float_to_bf16_kernel(
    const float* input, const void* weights, std::uint32_t input_width,
    std::uint32_t output_width, std::uint16_t* output) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x) * 8U +
        static_cast<std::uint32_t>(threadIdx.x) / 32U;
    if (row >= output_width) return; // Whole inactive wave; no block barriers.
    const auto lane = static_cast<std::uint32_t>(threadIdx.x) % 32U;
    float partial[8]{};
    const auto* blocks = static_cast<const Q8Block*>(weights) +
        static_cast<std::uint64_t>(row) * (input_width / 32U);
    for (std::uint64_t chunk = 0; chunk < input_width; chunk += 256U) {
        const auto base = chunk + 8U * lane;
        if (base < input_width) {
            const auto feature = static_cast<std::uint32_t>(base);
            const auto& block = blocks[feature / 32U];
            float values[8];
            std::int8_t codes[8];
#pragma unroll
            for (unsigned local = 0; local < 8; ++local) {
                values[local] = input[feature + local];
                codes[local] = block.values[feature % 32U + local];
            }
            const float scale = __half2float(block.scale);
#pragma unroll
            for (unsigned local = 0; local < 8; ++local)
                partial[local] += values[local] * (scale * static_cast<float>(codes[local]));
        }
    }
    // The shuffle reads partners before upper lanes are overwritten. Lane0's
    // 8 independent leaves preserve the old 128,64,32,16,8 stages exactly.
#pragma unroll
    for (unsigned distance = 16; distance > 0; distance /= 2) {
#pragma unroll
        for (unsigned local = 0; local < 8; ++local)
            partial[local] = qsa_output_legacy_tree_add(
                partial[local], __shfl_down(partial[local], distance, 32));
    }
    if (lane == 0) {
        partial[0] = qsa_output_legacy_tree_add(partial[0], partial[4]);
        partial[1] = qsa_output_legacy_tree_add(partial[1], partial[5]);
        partial[2] = qsa_output_legacy_tree_add(partial[2], partial[6]);
        partial[3] = qsa_output_legacy_tree_add(partial[3], partial[7]);
        partial[0] = qsa_output_legacy_tree_add(partial[0], partial[2]);
        partial[1] = qsa_output_legacy_tree_add(partial[1], partial[3]);
        partial[0] = qsa_output_legacy_tree_add(partial[0], partial[1]);
        output[row] = float_to_bf16(partial[0]);
    }
}

// Shared by the C1 kernel and its known-window rows wrapper (one row per grid.y).
template <SmallWeightKind Kind, class... Source>
__device__ __forceinline__ void norm_rope_body(
    float* values,
    std::uint32_t head_count,
    std::uint32_t head_dim,
    std::uint32_t head_stride,
    const void* norm_weights,
    float epsilon,
    std::uint32_t rotary_dim,
    float theta,
    std::uint64_t position,
    GraphPositionKernelArguments dynamic, Source... rotary_source) {
    if (dynamic.position != nullptr) position = *dynamic.position;
    if (!(rotary_source.check(position) && ...)) return;
    // Another head can report arithmetic failure while this block starts.
    // Broadcast a single atomic observation before any block-wide barriers;
    // per-thread early exits could otherwise diverge around __syncthreads.
    __shared__ unsigned graph_reject;
    if (dynamic.status != nullptr) {
        if (threadIdx.x == 0) graph_reject =
            atomicAdd(dynamic.status, 0U) != 0 || position >= dynamic.context_tokens ? 1U : 0U;
        __syncthreads();
        if (graph_reject != 0) return;
    }
    const auto head =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto feature =
        static_cast<std::uint32_t>(
            threadIdx.x);
    if (head >= head_count ||
        feature >= head_dim) {
        return;
    }

    extern __shared__ float shared[];
    auto* source = shared;
    auto* reduction =
        shared + head_dim;

    const auto index =
        static_cast<std::uint64_t>(
            head) *
            head_stride +
        feature;
    const float input =
        values[index];
    source[feature] = input;
    reduction[feature] =
        input * input;
    __syncthreads();

    for (std::uint32_t stride =
             head_dim / 2U;
         stride > 0;
         stride >>= 1U) {
        if (feature < stride) {
            reduction[feature] +=
                reduction[
                    feature +
                    stride];
        }
        __syncthreads();
    }

    const float mean = reduction[0] / static_cast<float>(head_dim) + epsilon;
    const float inv_rms = rsqrtf(mean);
    if (dynamic.status != nullptr && (!isfinite(reduction[0]) || !isfinite(mean) ||
        mean <= 0 || !isfinite(inv_rms))) {
        // The shared reduction and scalar are identical for every thread in
        // this head, so this return is uniform before the remaining barrier.
        if (feature == 0) atomicOr(dynamic.status, 8U);
        return;
    }
    source[feature] =
        round_bf16(
            input *
            inv_rms *
            load_small_weight<Kind>(
                 norm_weights,
                 feature));
    if (dynamic.status != nullptr && !isfinite(source[feature])) atomicOr(dynamic.status, 8U);
    __syncthreads();

    float result =
        source[feature];
    if constexpr (sizeof...(Source) == 0) {
        if (feature < rotary_dim) {
            const auto half =
                rotary_dim / 2U;
            const auto pair =
                feature % half;
            const float exponent =
                2.0F *
                static_cast<float>(pair) /
                static_cast<float>(
                    rotary_dim);
            const float angle =
                static_cast<float>(
                    position) *
                powf(
                    theta,
                    -exponent);
            const float cosine =
                cosf(angle);
            const float sine =
                sinf(angle);

            if (feature < half) {
                result =
                    source[feature] *
                        cosine -
                    source[
                        feature +
                        half] *
                        sine;
            } else {
                result =
                    source[feature] *
                        cosine +
                    source[
                        feature -
                        half] *
                        sine;
            }
        }
    } else {
        if (feature < rotary_dim) {
            const auto half =
                rotary_dim / 2U;
            const auto pair =
                feature % half;
            const float2 cs = (rotary_source.cs[position * 32ULL + pair], ...);
            const float cosine = cs.x;
            const float sine = cs.y;

            if (feature < half) {
                result =
                    source[feature] *
                        cosine -
                    source[
                        feature +
                        half] *
                        sine;
            } else {
                result =
                    source[feature] *
                        cosine +
                    source[
                        feature -
                        half] *
                        sine;
            }
        }
    }
    const float rounded = round_bf16(result);
    if (dynamic.status != nullptr && (!isfinite(result) || !isfinite(rounded))) {
        atomicOr(dynamic.status, 8U);
        return;
    }
    values[index] = rounded;
}

template <SmallWeightKind Kind, class... Source>
__global__ void norm_rope_kernel(float* values, std::uint32_t head_count, std::uint32_t head_dim,
    std::uint32_t head_stride, const void* norm_weights, float epsilon, std::uint32_t rotary_dim,
    float theta, std::uint64_t position, GraphPositionKernelArguments dynamic, Source... rotary_source) {
    norm_rope_body<Kind, Source...>(values, head_count, head_dim, head_stride, norm_weights, epsilon,
        rotary_dim, theta, position, dynamic, rotary_source...);
}

__global__ void initialize_graph_attention_status(
    std::uint32_t* status, std::uint64_t scalar_position, GraphPositionKernelArguments dynamic) {
    const auto position = dynamic.position == nullptr ? scalar_position : *dynamic.position;
    *status = position >= dynamic.context_tokens ? 2U : 0U;
}

template <SmallWeightKind Kind>
__global__ void validate_graph_attention_sources(
    const float* query, const float* key, const float* value,
    const void* query_norm, const void* key_norm, std::uint64_t query_elements,
    std::uint64_t kv_elements, std::uint32_t norm_elements, std::uint32_t query_stride, std::uint32_t* status) {
    const auto maximum = query_elements > kv_elements ? query_elements : kv_elements;
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < maximum || i < norm_elements; i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        bool invalid = false;
        if (i < query_elements && i % query_stride < norm_elements) invalid = !isfinite(query[i]);
        if (i < kv_elements) invalid = invalid || !isfinite(key[i]) || !isfinite(value[i]) || !isfinite(round_bf16(value[i]));
        if (i < norm_elements) invalid = invalid || !isfinite(load_small_weight<Kind>(query_norm, static_cast<std::uint32_t>(i))) ||
            !isfinite(load_small_weight<Kind>(key_norm, static_cast<std::uint32_t>(i)));
        if (invalid) atomicOr(status, 4U);
    }
}

__global__ void store_raw_index_key_kernel(
    const float* current,
    std::uint16_t* state,
    std::uint32_t width,
    std::uint64_t position, GraphPositionKernelArguments dynamic) {
    if (dynamic.position != nullptr) position = *dynamic.position;
    if (dynamic.status != nullptr && (*dynamic.status != 0 || position >= dynamic.context_tokens)) return;
    const auto feature =
        static_cast<std::uint32_t>(
            blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    if (feature < width) {
        state[
            position *
                static_cast<std::uint64_t>(
                    width) +
            feature] =
            float_to_bf16(
                current[feature]);
    }
}

__global__ void store_kv_kernel(
    const float* current_k,
    const float* current_v,
    std::uint16_t* state,
    std::uint32_t kv_width,
    std::uint64_t position,
    GraphPositionKernelArguments dynamic) {
    if (dynamic.position != nullptr) position = *dynamic.position;
    if (dynamic.status != nullptr && (*dynamic.status != 0 || position >= dynamic.context_tokens)) return;
    const auto feature =
        static_cast<std::uint32_t>(
            blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    if (feature >= kv_width) {
        return;
    }
    const auto row =
        position *
        static_cast<std::uint64_t>(
            kv_width) *
        2ULL;
    state[row + feature] =
        float_to_bf16(
            current_k[feature]);
    state[
        row +
        kv_width +
        feature] =
        float_to_bf16(
            current_v[feature]);
}

// Grid (kv_heads, 2): block (h, half) quantizes KV head h of K (half 0) or V (half 1).
__global__ void store_kv_int8_kernel(
    const float* current_k,
    const float* current_v,
    void* state,
    std::uint32_t kv_width,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    std::uint64_t position,
    GraphPositionKernelArguments dynamic) {
    if (dynamic.position != nullptr) position = *dynamic.position;
    if (dynamic.status != nullptr && (*dynamic.status != 0 || position >= dynamic.context_tokens)) return;
    const auto half = static_cast<std::uint32_t>(blockIdx.y);
    kv_int8_store_head(half ? current_v : current_k, state, position, kv_width, kv_heads, half,
        static_cast<std::uint32_t>(blockIdx.x), head_dim);
}

template <SmallWeightKind Kind, class... Source>
__global__ void block_score_kernel(
    const float* index_query,
    const std::uint16_t* raw_keys,
    std::uint32_t index_heads,
    std::uint32_t head_dim,
    std::uint32_t compress_ratio,
    const void* key_norm,
    float epsilon,
    std::uint32_t rotary_dim,
    float theta,
    float* block_scores, GraphSelectionArguments dynamic, Source... rotary_source) {
    const auto block =
        static_cast<std::uint32_t>(
            blockIdx.x);
    if (dynamic.position.position != nullptr) {
        std::uint32_t visible = 0;
        if (!graph_visible(dynamic, visible) || visible / compress_ratio <= dynamic.block_topk ||
            block >= visible / compress_ratio) return;
    }
    const auto feature =
        static_cast<std::uint32_t>(
            threadIdx.x);
    if (feature >= head_dim) {
        return;
    }

    extern __shared__ float shared[];
    auto* pooled = shared;
    auto* scratch =
        shared + head_dim;

    float mean = 0.0F;
    const auto token_start =
        static_cast<std::uint64_t>(
            block) *
        compress_ratio;
    if (!(rotary_source.check(token_start) && ...)) return;
    for (std::uint32_t offset = 0;
         offset < compress_ratio;
         ++offset) {
        mean +=
            bf16_to_float(
                raw_keys[
                    (token_start +
                     offset) *
                        head_dim +
                    feature]);
    }
    mean /=
        static_cast<float>(
            compress_ratio);

    // Upstream casts the FP32 mean back to raw-key dtype before k_norm.
    mean =
        round_bf16(mean);
    pooled[feature] = mean;
    scratch[feature] =
        mean * mean;
    __syncthreads();

    for (std::uint32_t stride =
             head_dim / 2U;
         stride > 0;
         stride >>= 1U) {
        if (feature < stride) {
            scratch[feature] +=
                scratch[
                    feature +
                    stride];
        }
        __syncthreads();
    }

    const float inv_rms =
        rsqrtf(
            scratch[0] /
                static_cast<float>(
                    head_dim) +
            epsilon);
    if (dynamic.position.status != nullptr && (!isfinite(scratch[0]) || !isfinite(inv_rms))) {
        if (feature == 0) atomicOr(dynamic.position.status, 8U);
        return;
    }
    pooled[feature] =
        round_bf16(
            mean *
            inv_rms *
            load_small_weight<Kind>(
                 key_norm,
                 feature));
    __syncthreads();

    float rotated =
        pooled[feature];
    if constexpr (sizeof...(Source) == 0) {
        if (feature < rotary_dim) {
            const auto half =
                rotary_dim / 2U;
            const auto pair =
                feature % half;
            const float exponent =
                2.0F *
                static_cast<float>(pair) /
                static_cast<float>(
                    rotary_dim);
            const float angle =
                static_cast<float>(
                    token_start) *
                powf(
                    theta,
                    -exponent);
            const float cosine =
                cosf(angle);
            const float sine =
                sinf(angle);
            if (feature < half) {
                rotated =
                    pooled[feature] *
                        cosine -
                    pooled[
                        feature +
                        half] *
                        sine;
            } else {
                rotated =
                    pooled[feature] *
                        cosine +
                    pooled[
                        feature -
                        half] *
                        sine;
            }
        }
    } else {
        if (feature < rotary_dim) {
            const auto half =
                rotary_dim / 2U;
            const auto pair =
                feature % half;
            const float2 cs = (rotary_source.cs[token_start * 32ULL + pair], ...);
            const float cosine = cs.x;
            const float sine = cs.y;
            if (feature < half) {
                rotated =
                    pooled[feature] *
                        cosine -
                    pooled[
                        feature +
                        half] *
                        sine;
            } else {
                rotated =
                    pooled[feature] *
                        cosine +
                    pooled[
                        feature -
                        half] *
                        sine;
            }
        }
    }
    // Every lane must finish reading its RoPE partner before any lane writes
    // the rotated value back into the shared pooled vector.
    __syncthreads();
    pooled[feature] =
        round_bf16(rotated);
    if (dynamic.position.status != nullptr && !isfinite(pooled[feature])) atomicOr(dynamic.position.status, 8U);
    __syncthreads();

    __shared__ float total;
    if (feature == 0) {
        total = 0.0F;
    }
    __syncthreads();

    for (std::uint32_t head = 0;
         head < index_heads;
         ++head) {
        scratch[feature] =
            index_query[
                static_cast<std::uint64_t>(
                    head) *
                    head_dim +
                feature] *
            pooled[feature];
        __syncthreads();

        for (std::uint32_t stride =
                 head_dim / 2U;
             stride > 0;
             stride >>= 1U) {
            if (feature < stride) {
                scratch[feature] +=
                    scratch[
                        feature +
                        stride];
            }
            __syncthreads();
        }
        if (feature == 0) {
            if (dynamic.position.status != nullptr && !isfinite(scratch[0])) atomicOr(dynamic.position.status, 8U);
            total +=
                fmaxf(
                    scratch[0],
                    0.0F);
        }
        __syncthreads();
    }

    if (feature == 0) {
        if (dynamic.position.status != nullptr && !isfinite(total)) atomicOr(dynamic.position.status, 8U);
        block_scores[block] =
            total /
            sqrtf(
                static_cast<float>(
                    head_dim));
    }
}
// block_score_kernel's body for the draft batch's rows kernel; the one-token kernel above stays as it was.
template <SmallWeightKind Kind, class... Source>
__device__ __forceinline__ void block_score_body(
    const float* index_query,
    const std::uint16_t* raw_keys,
    std::uint32_t index_heads,
    std::uint32_t head_dim,
    std::uint32_t compress_ratio,
    const void* key_norm,
    float epsilon,
    std::uint32_t rotary_dim,
    float theta,
    float* block_scores, GraphSelectionArguments dynamic, Source... rotary_source) {
    const auto block =
        static_cast<std::uint32_t>(
            blockIdx.x);
    if (dynamic.position.position != nullptr) {
        std::uint32_t visible = 0;
        if (!graph_visible(dynamic, visible) || visible / compress_ratio <= dynamic.block_topk ||
            block >= visible / compress_ratio) return;
    }
    const auto feature =
        static_cast<std::uint32_t>(
            threadIdx.x);
    if (feature >= head_dim) {
        return;
    }

    extern __shared__ float shared[];
    auto* pooled = shared;
    auto* scratch =
        shared + head_dim;

    float mean = 0.0F;
    const auto token_start =
        static_cast<std::uint64_t>(
            block) *
        compress_ratio;
    if (!(rotary_source.check(token_start) && ...)) return;
    for (std::uint32_t offset = 0;
         offset < compress_ratio;
         ++offset) {
        mean +=
            bf16_to_float(
                raw_keys[
                    (token_start +
                     offset) *
                        head_dim +
                    feature]);
    }
    mean /=
        static_cast<float>(
            compress_ratio);

    // Upstream casts the FP32 mean back to raw-key dtype before k_norm.
    mean =
        round_bf16(mean);
    pooled[feature] = mean;
    scratch[feature] =
        mean * mean;
    __syncthreads();

    for (std::uint32_t stride =
             head_dim / 2U;
         stride > 0;
         stride >>= 1U) {
        if (feature < stride) {
            scratch[feature] +=
                scratch[
                    feature +
                    stride];
        }
        __syncthreads();
    }

    const float inv_rms =
        rsqrtf(
            scratch[0] /
                static_cast<float>(
                    head_dim) +
            epsilon);
    if (dynamic.position.status != nullptr && (!isfinite(scratch[0]) || !isfinite(inv_rms))) {
        if (feature == 0) atomicOr(dynamic.position.status, 8U);
        return;
    }
    pooled[feature] =
        round_bf16(
            mean *
            inv_rms *
            load_small_weight<Kind>(
                 key_norm,
                 feature));
    __syncthreads();

    float rotated =
        pooled[feature];
    if constexpr (sizeof...(Source) == 0) {
        if (feature < rotary_dim) {
            const auto half =
                rotary_dim / 2U;
            const auto pair =
                feature % half;
            const float exponent =
                2.0F *
                static_cast<float>(pair) /
                static_cast<float>(
                    rotary_dim);
            const float angle =
                static_cast<float>(
                    token_start) *
                powf(
                    theta,
                    -exponent);
            const float cosine =
                cosf(angle);
            const float sine =
                sinf(angle);
            if (feature < half) {
                rotated =
                    pooled[feature] *
                        cosine -
                    pooled[
                        feature +
                        half] *
                        sine;
            } else {
                rotated =
                    pooled[feature] *
                        cosine +
                    pooled[
                        feature -
                        half] *
                        sine;
            }
        }
    } else {
        if (feature < rotary_dim) {
            const auto half =
                rotary_dim / 2U;
            const auto pair =
                feature % half;
            const float2 cs = (rotary_source.cs[token_start * 32ULL + pair], ...);
            const float cosine = cs.x;
            const float sine = cs.y;
            if (feature < half) {
                rotated =
                    pooled[feature] *
                        cosine -
                    pooled[
                        feature +
                        half] *
                        sine;
            } else {
                rotated =
                    pooled[feature] *
                        cosine +
                    pooled[
                        feature -
                        half] *
                        sine;
            }
        }
    }
    // Every lane must finish reading its RoPE partner before any lane writes
    // the rotated value back into the shared pooled vector.
    __syncthreads();
    pooled[feature] =
        round_bf16(rotated);
    if (dynamic.position.status != nullptr && !isfinite(pooled[feature])) atomicOr(dynamic.position.status, 8U);
    __syncthreads();

    __shared__ float total;
    if (feature == 0) {
        total = 0.0F;
    }
    __syncthreads();

    for (std::uint32_t head = 0;
         head < index_heads;
         ++head) {
        scratch[feature] =
            index_query[
                static_cast<std::uint64_t>(
                    head) *
                    head_dim +
                feature] *
            pooled[feature];
        __syncthreads();

        for (std::uint32_t stride =
                 head_dim / 2U;
             stride > 0;
             stride >>= 1U) {
            if (feature < stride) {
                scratch[feature] +=
                    scratch[
                        feature +
                        stride];
            }
            __syncthreads();
        }
        if (feature == 0) {
            if (dynamic.position.status != nullptr && !isfinite(scratch[0])) atomicOr(dynamic.position.status, 8U);
            total +=
                fmaxf(
                    scratch[0],
                    0.0F);
        }
        __syncthreads();
    }

    if (feature == 0) {
        if (dynamic.position.status != nullptr && !isfinite(total)) atomicOr(dynamic.position.status, 8U);
        block_scores[block] =
            total /
            sqrtf(
                static_cast<float>(
                    head_dim));
    }
}

__global__ void fill_visible_indices_kernel(
    std::uint32_t* selected,
    std::uint32_t visible) {
    const auto index =
        static_cast<std::uint32_t>(
            blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    if (index < visible) {
        selected[index] =
            index;
    }
}

// Device-owned control storage avoids a borrowed host async-copy lifetime.
__global__ void prepare_qsa_gqa_control(std::uint64_t* position, std::uint32_t* status,
    std::uint64_t scalar_position, bool preserve_selection_error) {
    if (threadIdx.x == 0) {
        *position = scalar_position;
        if (!preserve_selection_error) *status = 0;
    }
}
__global__ void publish_qsa_attention_status(const std::uint32_t* local, DeferredMoeFirstError* first_error) {
    if (threadIdx.x == 0) merge_deferred_moe_first_error(*first_error, *local, DeferredMoeDomain::qsa_attention);
}

// Local status belongs to this QSA workspace, not to the D1 domain word.
// Preserve the complete first D1 error and make indices safe even on rejection;
// the D1 completion fence rejects the token before host publication.
__global__ void publish_qsa_selection_status(const std::uint32_t* local,
    DeferredMoeFirstError* first_error, std::uint32_t* selected, std::uint32_t count) {
    // The existing block-score producer reports nonfinite intermediates as
    // graph arithmetic bit 8, before ReLU can mask NaNs. This eager selector's
    // public status contract names that same failure nonfinite_score (bit 1).
    // Preserve any other selector bits and the complete earlier D1 first error.
    const auto producer_bits = *local;
    const auto bits = (producer_bits & ~8U) |
        ((producer_bits & 8U) ? qsa_selection_diagnostic::nonfinite_score : 0U);
    const auto index = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (bits && index < count) selected[index] = 0;
    if (index == 0) merge_deferred_moe_first_error(*first_error, bits, DeferredMoeDomain::qsa_selection);
}

// Single-thread reference selection (diagnostic only; production runs the parallel body below).
[[maybe_unused]] __device__ __forceinline__ void select_blocks_body(
    float* block_scores,
    std::uint32_t num_blocks,
    std::uint32_t select_blocks,
    std::uint32_t compress_ratio,
    std::uint32_t visible_tokens,
    std::uint32_t* selected, GraphSelectionArguments dynamic) {

    if (dynamic.position.position != nullptr) {
        if (!graph_visible(dynamic, visible_tokens)) return;
        num_blocks = visible_tokens / compress_ratio;
        if (num_blocks <= dynamic.block_topk) {
            for (std::uint32_t token = 0; token < visible_tokens; ++token) selected[token] = token;
            return;
        }
        select_blocks = dynamic.block_topk;
    }

    for (std::uint32_t slot = 0;
         slot < select_blocks;
         ++slot) {
        float best =
            -INFINITY;
        std::uint32_t best_block = 0;
        for (std::uint32_t block = 0;
             block < num_blocks;
             ++block) {
            const float value =
                block_scores[block];
            if (value > best ||
                (value == best &&
                 block < best_block)) {
                best = value;
                best_block = block;
            }
        }
        selected[slot] =
            best_block;
        block_scores[best_block] =
            -INFINITY;
    }

    // The reference applies a mask over the full KV sequence. Sort selected
    // blocks by position so the sparse gather preserves chronological order.
    for (std::uint32_t i = 1;
         i < select_blocks;
         ++i) {
        const auto value =
            selected[i];
        std::uint32_t j = i;
        while (j > 0 &&
               selected[j - 1] >
                   value) {
            selected[j] =
                selected[j - 1];
            --j;
        }
        selected[j] =
            value;
    }

    // Expand backwards in place from block ids to token ids.
    for (std::uint32_t reverse = select_blocks;
         reverse > 0;
         --reverse) {
        const auto block =
            selected[
                reverse - 1U];
        const auto base =
            (reverse - 1U) *
            compress_ratio;
        for (std::uint32_t offset = 0;
             offset < compress_ratio;
             ++offset) {
            selected[
                base +
                offset] =
                block *
                    compress_ratio +
                offset;
        }
    }

    const auto complete_tokens =
        num_blocks *
        compress_ratio;
    const auto selected_complete =
        select_blocks *
        compress_ratio;
    for (std::uint32_t token =
             complete_tokens;
         token < visible_tokens;
         ++token) {
        selected[
            selected_complete +
            token -
            complete_tokens] =
            token;
    }
}

#ifdef STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC // reference selection of the block-select probe only
__global__ void select_blocks_serial_kernel(float* block_scores, std::uint32_t num_blocks,
    std::uint32_t select_blocks, std::uint32_t compress_ratio, std::uint32_t visible_tokens,
    std::uint32_t* selected, GraphSelectionArguments dynamic) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    select_blocks_body(block_scores, num_blocks, select_blocks, compress_ratio, visible_tokens, selected, dynamic);
}
#endif

// Workgroup-parallel equivalent of select_blocks_body (every thread calls it).
// The serial loop takes, slot by slot, the lowest-index maximal score > -inf
// (NaN never wins) and falls back to block 0 once none is left; it then emits
// the ascending sort of that multiset. Here a radix select over the ordered
// key (11/11/10 bits, +0 == -0) finds the k-th largest key and an ordered
// compaction keeps every greater key plus the lowest-index equal keys, so
// slots come out ascending. Selected scores (and block 0 on fallback) become
// -inf exactly as the serial loop leaves them.
constexpr std::uint32_t kSelectThreads = 1024;
constexpr std::uint32_t kSelectWaves = kSelectThreads / 32U;
static_assert(kSelectWaves == 32U);
__device__ __forceinline__ std::uint32_t select_key(float value) {
    auto bits = __float_as_uint(value);
    if ((bits & 0x7fffffffU) == 0) bits = 0;
    return bits & 0x80000000U ? ~bits : bits | 0x80000000U;
}
// Inclusive workgroup prefix sum; total is the workgroup sum.
__device__ __forceinline__ std::uint32_t select_scan(std::uint32_t value, std::uint32_t* waves, std::uint32_t& total) {
    const auto lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    for (unsigned d = 1; d < 32U; d <<= 1U) { const auto up = __shfl_up(value, d, 32); if (lane >= d) value += up; }
    if (lane == 31U) waves[wave] = value;
    __syncthreads();
    if (wave == 0) {
        auto sum = waves[lane];
        for (unsigned d = 1; d < 32U; d <<= 1U) { const auto up = __shfl_up(sum, d, 32); if (lane >= d) sum += up; }
        waves[lane] = sum;
    }
    __syncthreads();
    total = waves[kSelectWaves - 1U];
    value += wave ? waves[wave - 1U] : 0U;
    __syncthreads();
    return value;
}
__device__ void select_blocks_parallel_body(float* block_scores, std::uint32_t num_blocks, std::uint32_t select_blocks,
    std::uint32_t compress_ratio, std::uint32_t visible_tokens, std::uint32_t* selected, GraphSelectionArguments dynamic) {
    const auto tid = static_cast<std::uint32_t>(threadIdx.x);
    if (dynamic.position.position != nullptr) {
        if (!graph_visible(dynamic, visible_tokens)) return;
        num_blocks = visible_tokens / compress_ratio;
        if (num_blocks <= dynamic.block_topk) {
            for (auto token = tid; token < visible_tokens; token += kSelectThreads) selected[token] = token;
            return;
        }
        select_blocks = dynamic.block_topk;
    }
    __shared__ std::uint32_t histogram[2048];
    __shared__ std::uint32_t waves[kSelectWaves];
    __shared__ std::uint32_t found_digit, found_need, found_count;
    std::uint32_t prefix = 0, mask = 0, need = select_blocks, zeros = 0;
    for (unsigned pass = 0; pass < 3U; ++pass) {
        const unsigned shift = pass == 0 ? 21U : pass == 1 ? 10U : 0U;
        const auto digits = pass == 2 ? 1024U : 2048U;
        for (auto i = tid; i < 2048U; i += kSelectThreads) histogram[i] = 0;
        __syncthreads();
        for (std::uint32_t base = 0; base < num_blocks; base += kSelectThreads) {
            const auto block = base + tid;
            const float value = block < num_blocks ? block_scores[block] : -INFINITY;
            const auto key = select_key(value);
            const bool counted = value > -INFINITY && (key & mask) == prefix;
            const auto digit = (key >> shift) & (digits - 1U);
            // One atomic for the lanes sharing the first counted lane's digit.
            const auto votes = __ballot(counted);
            if (votes) {
                const auto leader = static_cast<unsigned>(__builtin_ctzll(votes));
                const auto lead = __shfl(digit, static_cast<int>(leader), 32);
                const auto same = __ballot(counted && digit == lead);
                if (counted && digit != lead) atomicAdd(&histogram[digit], 1U);
                else if (tid % 32U == leader) atomicAdd(&histogram[lead], static_cast<std::uint32_t>(__popcll(same)));
            }
        }
        __syncthreads();
        const auto high = 2U * tid + 1U < digits ? histogram[2U * tid + 1U] : 0U;
        const auto low = 2U * tid < digits ? histogram[2U * tid] : 0U;
        std::uint32_t counted_total = 0;
        const auto above = [&] { const auto inclusive = select_scan(high + low, waves, counted_total); return counted_total - inclusive; }();
        if (pass == 0 && counted_total <= need) { zeros = need - counted_total; need = counted_total; break; }
        if (above < need && need <= above + high) { found_digit = 2U * tid + 1U; found_need = need - above; found_count = high; }
        else if (above + high < need && need <= above + high + low) { found_digit = 2U * tid; found_need = need - above - high; found_count = low; }
        __syncthreads();
        prefix |= found_digit << shift; mask |= (digits - 1U) << shift; need = found_need;
        if (found_count == need) break;
    }
    for (auto token = tid; token < zeros * compress_ratio; token += kSelectThreads) selected[token] = token % compress_ratio;
    std::uint32_t greater_before = 0, equal_before = 0;
    for (std::uint32_t base = 0; base < num_blocks; base += 4U * kSelectThreads) {
        const auto first = base + 4U * tid;
        unsigned kind[4], greater = 0, equal = 0;
        for (unsigned j = 0; j < 4U; ++j) {
            const auto block = first + j;
            const float value = block < num_blocks ? block_scores[block] : -INFINITY;
            const auto masked = select_key(value) & mask;
            kind[j] = !(value > -INFINITY) ? 0U : masked > prefix ? 1U : masked == prefix ? 2U : 0U;
            greater += kind[j] == 1U; equal += kind[j] == 2U;
        }
        std::uint32_t tile = 0;
        const auto packed = select_scan((greater << 16U) | equal, waves, tile);
        auto g = greater_before + (packed >> 16U) - greater, e = equal_before + (packed & 0xffffU) - equal;
        for (unsigned j = 0; j < 4U; ++j) {
            if (kind[j] == 1U || (kind[j] == 2U && e < need)) {
                const auto block = first + j, slot = zeros + g + min(e, need);
                for (std::uint32_t offset = 0; offset < compress_ratio; ++offset)
                    selected[slot * compress_ratio + offset] = block * compress_ratio + offset;
                block_scores[block] = -INFINITY;
            }
            g += kind[j] == 1U; e += kind[j] == 2U;
        }
        greater_before += tile >> 16U; equal_before += tile & 0xffffU;
    }
    if (zeros && tid == 0) block_scores[0] = -INFINITY;
    const auto complete_tokens = num_blocks * compress_ratio, selected_complete = select_blocks * compress_ratio;
    for (auto token = complete_tokens + tid; token < visible_tokens; token += kSelectThreads)
        selected[selected_complete + token - complete_tokens] = token;
}

__global__ __launch_bounds__(kSelectThreads) void select_blocks_kernel(float* block_scores, std::uint32_t num_blocks,
    std::uint32_t select_blocks, std::uint32_t compress_ratio, std::uint32_t visible_tokens,
    std::uint32_t* selected, GraphSelectionArguments dynamic) {
    select_blocks_parallel_body(block_scores, num_blocks, select_blocks, compress_ratio, visible_tokens, selected, dynamic);
}

template <bool Int8 = false>
__device__ __forceinline__ void attention_score_body(
    const float* q_gate,
    const std::uint16_t* kv_state,
    const std::uint32_t* selected,
    std::uint32_t selected_count,
    std::uint32_t maximum_selected,
    std::uint32_t attention_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    float* scores, GraphSelectionArguments dynamic) {
    if (!graph_selected_count(dynamic, selected_count)) return;
    const auto flat =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto head =
        flat /
        (dynamic.position.position == nullptr ? selected_count : maximum_selected);
    const auto selected_slot =
        flat %
        (dynamic.position.position == nullptr ? selected_count : maximum_selected);
    if (head >= attention_heads || selected_slot >= selected_count) {
        return;
    }

    const auto feature =
        static_cast<std::uint32_t>(
            threadIdx.x);
    __shared__ float scratch[kThreads];

    const auto token =
        selected[selected_slot];
    const auto kv_group =
        attention_heads /
        kv_heads;
    const auto kv_head =
        head /
        kv_group;
    const auto kv_width =
        kv_heads *
        head_dim;
    const auto row_base =
        static_cast<std::uint64_t>(
            token) *
        kv_width *
        2ULL;

    float product = 0.0F;
    if constexpr (Int8) {
        if (feature < head_dim)
            product = q_gate[static_cast<std::uint64_t>(head) * head_dim * 2ULL + feature] *
                kv_int8_load(kv_state, token, kv_width, kv_heads, 0U, kv_head, head_dim, feature);
    } else if (feature < head_dim) {
        product =
            q_gate[
                static_cast<std::uint64_t>(
                    head) *
                    head_dim *
                    2ULL +
                feature] *
            bf16_to_float(
                kv_state[
                    row_base +
                    static_cast<std::uint64_t>(
                        kv_head) *
                        head_dim +
                    feature]);
    }
    scratch[feature] =
        product;
    __syncthreads();

    for (std::uint32_t stride =
             kThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (feature < stride) {
            scratch[feature] +=
                scratch[
                    feature +
                    stride];
        }
        __syncthreads();
    }

    if (feature == 0) {
        if (dynamic.position.status != nullptr && !isfinite(scratch[0])) atomicOr(dynamic.position.status, 8U);
        scores[
            static_cast<std::uint64_t>(
                head) *
                maximum_selected +
            selected_slot] =
            scratch[0] /
            sqrtf(
                static_cast<float>(
                    head_dim));
    }
}

__global__ void attention_score_kernel(const float* q_gate, const std::uint16_t* kv_state,
    const std::uint32_t* selected, std::uint32_t selected_count, std::uint32_t maximum_selected,
    std::uint32_t attention_heads, std::uint32_t kv_heads, std::uint32_t head_dim,
    float* scores, GraphSelectionArguments dynamic) {
    attention_score_body(q_gate, kv_state, selected, selected_count, maximum_selected,
        attention_heads, kv_heads, head_dim, scores, dynamic);
}

__global__ void attention_score_int8_kernel(const float* q_gate, const std::uint16_t* kv_state,
    const std::uint32_t* selected, std::uint32_t selected_count, std::uint32_t maximum_selected,
    std::uint32_t attention_heads, std::uint32_t kv_heads, std::uint32_t head_dim,
    float* scores, GraphSelectionArguments dynamic) {
    attention_score_body<true>(q_gate, kv_state, selected, selected_count, maximum_selected,
        attention_heads, kv_heads, head_dim, scores, dynamic);
}

// Publish (Q7 p32 only): each rounded probability is also stored, unchanged, to
// published[index] when published is not null; the default form is the incumbent body.
template <bool Publish = false>
__device__ __forceinline__ void attention_softmax_body(
    float* scores,
    std::uint32_t selected_count,
    std::uint32_t maximum_selected, GraphSelectionArguments dynamic, float* published = nullptr) {
    if (!graph_selected_count(dynamic, selected_count)) return;
    const auto head =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto lane =
        static_cast<std::uint32_t>(
            threadIdx.x);
    __shared__ float scratch[kThreads];

    float local_max =
        -INFINITY;
    for (std::uint32_t index = lane;
         index < selected_count;
         index += kThreads) {
        local_max =
            fmaxf(
                local_max,
                scores[
                    static_cast<std::uint64_t>(
                        head) *
                        maximum_selected +
                    index]);
    }
    scratch[lane] =
        local_max;
    __syncthreads();

    for (std::uint32_t stride =
             kThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            scratch[lane] =
                fmaxf(
                    scratch[lane],
                    scratch[
                        lane +
                        stride]);
        }
        __syncthreads();
    }
    const float maximum =
        scratch[0];
    if (dynamic.position.status != nullptr && !isfinite(maximum)) {
        if (lane == 0) atomicOr(dynamic.position.status, 8U);
        return;
    }
    // Preserve the reduced maximum in each lane before reusing scratch for
    // the sum reduction (including waves with no active selected element).
    __syncthreads();

    float local_sum = 0.0F;
    for (std::uint32_t index = lane;
         index < selected_count;
         index += kThreads) {
        local_sum +=
            expf(
                scores[
                    static_cast<std::uint64_t>(
                        head) *
                        maximum_selected +
                    index] -
                maximum);
    }
    scratch[lane] =
        local_sum;
    __syncthreads();

    for (std::uint32_t stride =
             kThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            scratch[lane] +=
                scratch[
                    lane +
                    stride];
        }
        __syncthreads();
    }
    const float denominator =
        scratch[0];
    if (dynamic.position.status != nullptr && (!isfinite(denominator) || denominator <= 0.0F)) {
        if (lane == 0) atomicOr(dynamic.position.status, 8U);
        return;
    }

    for (std::uint32_t index = lane;
         index < selected_count;
         index += kThreads) {
        const auto location =
            static_cast<std::uint64_t>(
                head) *
                maximum_selected +
            index;
        if constexpr (Publish) {
            const float probability = round_bf16(expf(scores[location] - maximum) / denominator);
            scores[location] = probability;
            if (published != nullptr) published[index] = probability;
        } else
        // Eager attention casts softmax back to query dtype.
        scores[location] =
            round_bf16(
                expf(
                    scores[location] -
                    maximum) /
                denominator);
    }
}

__global__ void attention_softmax_kernel(float* scores, std::uint32_t selected_count,
    std::uint32_t maximum_selected, GraphSelectionArguments dynamic) {
    attention_softmax_body(scores, selected_count, maximum_selected, dynamic);
}

template <bool Int8 = false>
__device__ __forceinline__ void attention_output_body(
    const float* q_gate,
    const std::uint16_t* kv_state,
    const std::uint32_t* selected,
    const float* scores,
    std::uint32_t selected_count,
    std::uint32_t maximum_selected,
    std::uint32_t attention_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    RoundedStageImage output, GraphSelectionArguments dynamic) {
    if (!graph_selected_count(dynamic, selected_count)) return;
    const auto head =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto feature =
        static_cast<std::uint32_t>(
            threadIdx.x);
    if (head >= attention_heads ||
        feature >= head_dim) {
        return;
    }

    const auto kv_group =
        attention_heads /
        kv_heads;
    const auto kv_head =
        head /
        kv_group;
    const auto kv_width =
        kv_heads *
        head_dim;

    // Sequential sum += score * value in slot order; eight slots' token indices and
    // values are requested before they join the chain (latency, not arithmetic).
    const auto* head_scores = scores + static_cast<std::uint64_t>(head) * maximum_selected;
    const auto offset = static_cast<std::uint64_t>(kv_width) + static_cast<std::uint64_t>(kv_head) * head_dim + feature;
    const auto value_at = [&](std::uint32_t token) {
        if constexpr (Int8) return kv_int8_load(kv_state, token, kv_width, kv_heads, 1U, kv_head, head_dim, feature);
        else return bf16_to_float(kv_state[static_cast<std::uint64_t>(token) * kv_width * 2ULL + offset]);
    };
    float sum = 0.0F;
    std::uint32_t slot = 0;
    for (; slot + 8U <= selected_count; slot += 8U) {
        std::uint32_t token[8];
#pragma unroll
        for (unsigned i = 0; i < 8U; ++i) token[i] = selected[slot + i];
        float value[8], score[8];
#pragma unroll
        for (unsigned i = 0; i < 8U; ++i) { value[i] = value_at(token[i]); score[i] = head_scores[slot + i]; }
#pragma unroll
        for (unsigned i = 0; i < 8U; ++i) sum += score[i] * value[i];
    }
    for (; slot < selected_count; ++slot) {
        const float value = value_at(selected[slot]);
        sum += head_scores[slot] * value;
    }

    // Preserve the model's BF16 attention output and BF16 sigmoid gate before
    // the final output projection.
    const float attn =
        round_bf16(sum);
    const float gate =
        round_bf16(
            1.0F /
            (1.0F +
             expf(
                 -q_gate[
                     static_cast<std::uint64_t>(
                         head) *
                         head_dim *
                         2ULL +
                     head_dim +
                     feature])));
    output[
        static_cast<std::uint64_t>(
            head) *
            head_dim +
        feature] =
        round_bf16(
            attn * gate);
    if (dynamic.position.status != nullptr && !isfinite(round_bf16(attn * gate))) atomicOr(dynamic.position.status, 8U);
}

__global__ void attention_output_kernel(const float* q_gate, const std::uint16_t* kv_state,
    const std::uint32_t* selected, const float* scores, std::uint32_t selected_count,
    std::uint32_t maximum_selected, std::uint32_t attention_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, float* output, GraphSelectionArguments dynamic) {
    attention_output_body(q_gate, kv_state, selected, scores, selected_count, maximum_selected,
        attention_heads, kv_heads, head_dim, output, dynamic);
}

// One workgroup owns a head through softmax and the chronological value chain.
// The barrier replaces the launch boundary; both incumbent arithmetic bodies,
// the BF16 probability stores and the slot order remain unchanged.
template <bool Int8 = false, bool Dense = false>
__device__ __forceinline__ void attention_tail_body(const float* q_gate, const std::uint16_t* kv_state,
    const std::uint32_t* selected, float* scores, std::uint32_t selected_count,
    std::uint32_t maximum_selected, std::uint32_t attention_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, RoundedStageImage output, GraphSelectionArguments dynamic,
    std::int8_t* codes = nullptr, float* deltas = nullptr, std::uint32_t row = 0U) {
    attention_softmax_body(scores, selected_count, maximum_selected, dynamic);
    __syncthreads();
    attention_output_body<Int8>(q_gate, kv_state, selected, scores, selected_count, maximum_selected,
        attention_heads, kv_heads, head_dim, output, dynamic);
    if constexpr (Dense) if (threadIdx.x < head_dim) {
        const auto index = static_cast<std::uint64_t>(blockIdx.x) * head_dim + threadIdx.x;
        q8_rows::dense_a8_quantize_lane(output[index],
            codes + static_cast<std::uint64_t>(row) * attention_heads * head_dim + index,
            deltas + (index / 32U) * q8_rows::kDenseA8Rows + row);
    }
}
__global__ void attention_tail_kernel(const float* q_gate, const std::uint16_t* kv_state,
    const std::uint32_t* selected, float* scores, std::uint32_t selected_count,
    std::uint32_t maximum_selected, std::uint32_t attention_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, float* output, GraphSelectionArguments dynamic) {
    attention_tail_body(q_gate, kv_state, selected, scores, selected_count, maximum_selected,
        attention_heads, kv_heads, head_dim, output, dynamic);
}

__global__ void attention_output_int8_kernel(const float* q_gate, const std::uint16_t* kv_state,
    const std::uint32_t* selected, const float* scores, std::uint32_t selected_count,
    std::uint32_t maximum_selected, std::uint32_t attention_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, float* output, GraphSelectionArguments dynamic) {
    attention_output_body<true>(q_gate, kv_state, selected, scores, selected_count, maximum_selected,
        attention_heads, kv_heads, head_dim, output, dynamic);
}

bool contains(
    std::string_view text,
    std::string_view needle) {
    return text.find(needle) !=
        std::string_view::npos;
}

TensorPair tensor_pair(
    const ExecutionDispatchContext& context,
    std::uint32_t index) {
    if (context.resources == nullptr ||
        context.resources->model_manifest ==
            nullptr) {
        throw std::invalid_argument(
            "HIP indexed attention requires model manifest");
    }
    if (index >=
        context.resources->
            model_manifest->
            tensors.size()) {
        throw std::out_of_range(
            "indexed-attention tensor index exceeds manifest");
    }
    return TensorPair{
        &context.resources->
            model_manifest->
            tensors[index],
        &resolve_dispatch_tensor(
            context,
            index),
    };
}

void assign_unique(
    TensorPair& destination,
    std::uint32_t& count,
    const TensorPair& value) {
    destination = value;
    ++count;
}

IndexedTensors identify_tensors(
    const ExecutionDispatchContext& context) {
    if(const auto* cached=context.prepared_tensor_views) {
        if(!context.resources) throw std::invalid_argument("prepared attention roles require retained resources");
        cached->validate_stage_scope(context.resources->distributed_stage,context.step_id);
        cached->validate_source_handles(context.resources->model_manifest,context.resources->model_weights,context.resources->tensor_access);
        const auto& r=cached->indexed_roles(context.execution_identity,context.step_id);
        return {tensor_pair(context,r.q),tensor_pair(context,r.k),tensor_pair(context,r.v),tensor_pair(context,r.out),
            tensor_pair(context,r.q_norm),tensor_pair(context,r.k_norm),tensor_pair(context,r.index_q),tensor_pair(context,r.index_k),
            tensor_pair(context,r.index_q_norm),tensor_pair(context,r.index_k_norm)};
    }

    const auto& attention =
        resolve_dispatch_tensor_access(
            context, 0);
    const auto& indexer =
        resolve_dispatch_tensor_access(
            context, 1);
    const auto& norms =
        resolve_dispatch_tensor_access(
            context, 2);

    IndexedTensors result;
    std::uint32_t q = 0;
    std::uint32_t k = 0;
    std::uint32_t v = 0;
    std::uint32_t out = 0;
    std::uint32_t qn = 0;
    std::uint32_t kn = 0;
    std::uint32_t iq = 0;
    std::uint32_t ik = 0;
    std::uint32_t iqn = 0;
    std::uint32_t ikn = 0;

    for (const auto index :
         attention.tensor_indices) {
        const auto pair =
            tensor_pair(
                context,
                index);
        const auto name =
            std::string_view(
                pair.manifest->name);
        if (contains(
                name,
                "attn_output") ||
            contains(
                name,
                "o_proj")) {
            assign_unique(
                result.out,
                out,
                pair);
        } else if (contains(
                       name,
                       "attn_q") ||
                   contains(
                       name,
                       "q_proj")) {
            assign_unique(
                result.q,
                q,
                pair);
        } else if (contains(
                       name,
                       "attn_k") ||
                   contains(
                       name,
                       "k_proj")) {
            assign_unique(
                result.k,
                k,
                pair);
        } else if (contains(
                       name,
                       "attn_v") ||
                   contains(
                       name,
                       "v_proj")) {
            assign_unique(
                result.v,
                v,
                pair);
        }
    }

    for (const auto index :
         indexer.tensor_indices) {
        const auto pair =
            tensor_pair(
                context,
                index);
        const auto name =
            std::string_view(
                pair.manifest->name);
        if (contains(
                name,
                "indexer.q_proj") ||
            contains(
                name,
                "indexer_q_proj")) {
            assign_unique(
                result.index_q,
                iq,
                pair);
        } else if (contains(
                       name,
                       "indexer.k_proj") ||
                   contains(
                       name,
                       "indexer_k_proj")) {
            assign_unique(
                result.index_k,
                ik,
                pair);
        } else if (contains(
                       name,
                       "indexer.q_norm") ||
                   contains(
                       name,
                       "indexer_q_norm")) {
            assign_unique(
                result.index_q_norm,
                iqn,
                pair);
        } else if (contains(
                       name,
                       "indexer.k_norm") ||
                   contains(
                       name,
                       "indexer_k_norm")) {
            assign_unique(
                result.index_k_norm,
                ikn,
                pair);
        }
    }

    for (const auto index :
         norms.tensor_indices) {
        const auto pair =
            tensor_pair(
                context,
                index);
        const auto name =
            std::string_view(
                pair.manifest->name);
        if (contains(
                name,
                "attn_q_norm")) {
            assign_unique(
                result.q_norm,
                qn,
                pair);
        } else if (contains(
                       name,
                       "attn_k_norm")) {
            assign_unique(
                result.k_norm,
                kn,
                pair);
        }
    }

    if (q != 1 || k != 1 ||
        v != 1 || out != 1 ||
        qn != 1 || kn != 1 ||
        iq != 1 || ik != 1 ||
        iqn != 1 || ikn != 1) {
        throw std::invalid_argument(
            "indexed-attention tensor family is missing or ambiguous");
    }
    return result;
}

void validate_runtime_tensor(
    const TensorPair& pair,
    std::uint64_t required,
    const char* name) {
    if (pair.manifest == nullptr ||
        pair.runtime == nullptr) {
        throw std::invalid_argument(
            std::string(name) +
            " tensor is missing");
    }
    if (!runtime_tensor_gpu_visible(pair.runtime->storage) ||
        pair.runtime->address == 0 ||
        pair.runtime->bytes < required) {
        throw std::invalid_argument(
            std::string(name) +
            " runtime tensor is not GPU-visible or is undersized");
    }
}

void validate_matrix(
    const TensorPair& pair,
    std::uint32_t input,
    std::uint32_t output,
    const char* name) {
    if (pair.manifest->dimensions.size() < 2 ||
        pair.manifest->dimensions[0] != input ||
        pair.manifest->dimensions[1] != output) {
        throw std::invalid_argument(
            std::string(name) +
            " tensor geometry does not match contract");
    }
    const auto kind =
        weight_kind(
            pair.manifest->ggml_type);
    validate_runtime_tensor(
        pair,
        required_weight_bytes(
            kind,
            input,
            output),
        name);
}

void validate_norm(
    const TensorPair& pair,
    std::uint32_t width,
    const char* name) {
    if (pair.manifest->dimensions.empty() ||
        pair.manifest->dimensions[0] !=
            width) {
        throw std::invalid_argument(
            std::string(name) +
            " norm geometry does not match contract");
    }
    const auto kind =
        small_weight_kind(
            pair.manifest->ggml_type);
    validate_runtime_tensor(
        pair,
        required_small_weight_bytes(
            kind,
            width),
        name);
}

template <WeightKind Kind>
void launch_linear_to_float_typed(
    const ExecutionBufferAddress& input,
    const RuntimeTensorBinding& weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    float* output) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            linear_to_float_kernel<Kind>),
        dim3(output_width),
        dim3(kThreads),
        0,
        hip_execution_stream(),
        reinterpret_cast<
            const std::uint16_t*>(
                input.address),
        reinterpret_cast<
            const void*>(
                weights.address),
        input_width,
        output_width,
        output);
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention projection launch");
}

// One-row q8_rows sites over a registered Q6P32 copy (bitwise the R1 prefetch8 kernels);
// false when the weights have none.
bool qsa_q6_r1_input(const std::uint16_t* input, const void* weights, std::uint32_t width, std::uint32_t outputs,
    float* output, hipStream_t stream);
bool qsa_q6_r1_output(const float* input, const void* weights, std::uint32_t width, std::uint32_t outputs,
    std::uint16_t* output, hipStream_t stream);

void launch_linear_to_float(
    const TensorPair& tensor,
    const ExecutionBufferAddress& input,
    std::uint32_t input_width,
    std::uint32_t output_width,
    float* output, bool prefetch = false, qsa_input_prefetch::RuntimeDispatchStats* stats = nullptr) {
    if(prefetch && qsa_input_prefetch::eligible(*tensor.manifest) &&
       qsa_q6_r1_input(reinterpret_cast<const std::uint16_t*>(input.address),reinterpret_cast<const void*>(tensor.runtime->address),
           input_width,output_width,output,hip_execution_stream())) return;
    if(prefetch && qsa_input_prefetch::eligible(*tensor.manifest)){
        if(tensor.runtime->address%16U!=0 || tensor.manifest->absolute_offset%16U!=0 || tensor.runtime->bytes<qsa_input_prefetch::required_weight_bytes(input_width,output_width))throw std::invalid_argument("QSA prefetch launch weight alignment/extent invalid");
        hipLaunchKernelGGL(linear_prefetch8_q8_qsa_input_kernel,dim3(output_width),dim3(32),0,hip_execution_stream(),reinterpret_cast<const std::uint16_t*>(input.address),reinterpret_cast<const void*>(tensor.runtime->address),input_width,output_width,output);
        check_hip(hipGetLastError(),"HIP indexed-attention prefetch8 Q8 input projection launch");
        if(stats)stats->dispatched_projections.fetch_add(1,std::memory_order_relaxed);
        return;
    }
    switch (weight_kind(
                tensor.manifest->
                    ggml_type)) {
    case WeightKind::f32:
        launch_linear_to_float_typed<
            WeightKind::f32>(
                input, *tensor.runtime,
                input_width, output_width,
                output);
        return;
    case WeightKind::f16:
        launch_linear_to_float_typed<
            WeightKind::f16>(
                input, *tensor.runtime,
                input_width, output_width,
                output);
        return;
    case WeightKind::bf16:
        launch_linear_to_float_typed<
            WeightKind::bf16>(
                input, *tensor.runtime,
                input_width, output_width,
                output);
        return;
    case WeightKind::q8_0:
        launch_linear_to_float_typed<
            WeightKind::q8_0>(
                input, *tensor.runtime,
                input_width, output_width,
                output);
        return;
    case WeightKind::iq4_xs:
        launch_linear_to_float_typed<
            WeightKind::iq4_xs>(
                input, *tensor.runtime,
                input_width, output_width,
                output);
        return;
#define STRIX_GGML_KIND_CASE(KIND, type) \
    case WeightKind::KIND: \
        launch_linear_to_float_typed< \
            WeightKind::KIND>( \
                input, *tensor.runtime, \
                input_width, output_width, \
                output); \
        return;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_CASE) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_CASE) STRIX_GGML_KIND_CASE(iq4_nl, 20)
#undef STRIX_GGML_KIND_CASE
    }
    throw std::logic_error(
        "unreachable indexed projection kind");
}

template <WeightKind Kind>
void launch_linear_float_to_bf16_typed(
    const float* input,
    const RuntimeTensorBinding& weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    const ExecutionBufferAddress& output) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            linear_float_to_bf16_kernel<Kind>),
        dim3(output_width),
        dim3(kThreads),
        0,
        hip_execution_stream(),
        input,
        reinterpret_cast<
            const void*>(
                weights.address),
        input_width,
        output_width,
        reinterpret_cast<
            std::uint16_t*>(
                output.address));
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention output projection launch");
}

void launch_linear_float_to_bf16(
    const TensorPair& tensor,
    const float* input,
    std::uint32_t input_width,
    std::uint32_t output_width,
    const ExecutionBufferAddress& output, bool wave8 = false) {
    if (wave8) {
        if (tensor.manifest->ggml_type != 8U || input_width % 32U)
            throw std::invalid_argument("wave8 QSA output requires aligned Q8_0 weights");
        q8_projection_organization::validate_geometry(input_width, output_width);
        if (qsa_q6_r1_output(input, reinterpret_cast<const void*>(tensor.runtime->address), input_width, output_width,
                reinterpret_cast<std::uint16_t*>(output.address), hip_execution_stream())) return;
        hipLaunchKernelGGL(linear_prefetch8_wave8_q8_float_to_bf16_kernel,
            dim3((output_width + 7U) / 8U), dim3(256), 0, hip_execution_stream(), input,
            reinterpret_cast<const void*>(tensor.runtime->address), input_width, output_width,
            reinterpret_cast<std::uint16_t*>(output.address));
        check_hip(hipGetLastError(), "HIP indexed-attention wave8 Q8 output projection launch");
        return;
    }
    switch (weight_kind(
                tensor.manifest->
                    ggml_type)) {
    case WeightKind::f32:
        launch_linear_float_to_bf16_typed<
            WeightKind::f32>(
                input, *tensor.runtime,
                input_width, output_width,
                output);
        return;
    case WeightKind::f16:
        launch_linear_float_to_bf16_typed<
            WeightKind::f16>(
                input, *tensor.runtime,
                input_width, output_width,
                output);
        return;
    case WeightKind::bf16:
        launch_linear_float_to_bf16_typed<
            WeightKind::bf16>(
                input, *tensor.runtime,
                input_width, output_width,
                output);
        return;
    case WeightKind::q8_0:
        launch_linear_float_to_bf16_typed<
            WeightKind::q8_0>(
                input, *tensor.runtime,
                input_width, output_width,
                output);
        return;
    case WeightKind::iq4_xs:
        launch_linear_float_to_bf16_typed<
            WeightKind::iq4_xs>(
                input, *tensor.runtime,
                input_width, output_width,
                output);
        return;
#define STRIX_GGML_KIND_CASE(KIND, type) \
    case WeightKind::KIND: \
        launch_linear_float_to_bf16_typed< \
            WeightKind::KIND>( \
                input, *tensor.runtime, \
                input_width, output_width, \
                output); \
        return;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_CASE) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_CASE) STRIX_GGML_KIND_CASE(iq4_nl, 20)
#undef STRIX_GGML_KIND_CASE
    }
    throw std::logic_error(
        "unreachable indexed output projection kind");
}

template <SmallWeightKind Kind>
void launch_norm_rope_typed(
    float* values,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t head_stride,
    const RuntimeTensorBinding& norm,
    const ModelContract& model,
    std::uint64_t position,
    GraphPositionKernelArguments dynamic = {}, hipStream_t stream = hipStreamLegacy) {
    check_qwen_rotary_position(model, position);
    if (model.qwen_rotary_table && dynamic.position && dynamic.context_tokens > model.qwen_rotary_table->extent())
        throw std::out_of_range("YaRN graph capacity exceeds table extent");
    with_qwen_rotary(model, stream, [&](auto... rotary_source) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                norm_rope_kernel<Kind, decltype(rotary_source)...>),
            dim3(heads),
            dim3(head_dim),
            2ULL *
                head_dim *
                sizeof(float),
            stream,
            values,
            heads,
            head_dim,
            head_stride,
            reinterpret_cast<
                const void*>(
                    norm.address),
            model.rms_norm_epsilon,
            checked_u32(
                model.rotary_dim(),
                "rotary_dim"),
            model.rope_theta,
            position, dynamic, rotary_source...);
    });
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention norm/rope launch");
}

void launch_norm_rope(
    float* values,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t head_stride,
    const TensorPair& norm,
    const ModelContract& model,
    std::uint64_t position) {
    switch (small_weight_kind(
                norm.manifest->
                    ggml_type)) {
    case SmallWeightKind::f32:
        launch_norm_rope_typed<
            SmallWeightKind::f32>(
                values, heads, head_dim,
                head_stride, *norm.runtime,
                model, position, graph_position_arguments(), hip_execution_stream());
        return;
    case SmallWeightKind::f16:
        launch_norm_rope_typed<
            SmallWeightKind::f16>(
                values, heads, head_dim,
                head_stride, *norm.runtime,
                model, position, graph_position_arguments(), hip_execution_stream());
        return;
    case SmallWeightKind::bf16:
        launch_norm_rope_typed<
            SmallWeightKind::bf16>(
                values, heads, head_dim,
                head_stride, *norm.runtime,
                model, position, graph_position_arguments(), hip_execution_stream());
        return;
    }
    throw std::logic_error(
        "unreachable indexed norm kind");
}

template <SmallWeightKind Kind>
void launch_block_scores_typed(
    const WorkspacePointers& workspace,
    const SessionStateAddress& raw_state,
    const TensorPair& k_norm,
    const ModelContract& model,
    std::uint32_t complete_blocks, GraphSelectionArguments dynamic) {
    with_qwen_rotary(model, hip_execution_stream(), [&](auto... rotary_source) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                block_score_kernel<Kind, decltype(rotary_source)...>),
            dim3(complete_blocks),
            dim3(
                checked_u32(
                    model.indexer_head_dim,
                    "indexer_head_dim")),
            2ULL *
                model.indexer_head_dim *
                sizeof(float),
            hip_execution_stream(),
            workspace.index_query,
            reinterpret_cast<
                const std::uint16_t*>(
                    raw_state.address),
            checked_u32(
                model.indexer_n_heads,
                "indexer_n_heads"),
            checked_u32(
                model.indexer_head_dim,
                "indexer_head_dim"),
            checked_u32(
                model.indexer_compress_ratio,
                "indexer_compress_ratio"),
            reinterpret_cast<
                const void*>(
                    k_norm.runtime->address),
            model.rms_norm_epsilon,
            checked_u32(
                model.rotary_dim(),
                "rotary_dim"),
            model.rope_theta,
            workspace.block_scores, dynamic, rotary_source...);
    });
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention block-score launch");
}

void launch_block_scores(
    const WorkspacePointers& workspace,
    const SessionStateAddress& raw_state,
    const TensorPair& k_norm,
    const ModelContract& model,
    std::uint32_t complete_blocks, GraphSelectionArguments dynamic = {}) {
    switch (small_weight_kind(
                k_norm.manifest->
                    ggml_type)) {
    case SmallWeightKind::f32:
        launch_block_scores_typed<
            SmallWeightKind::f32>(
                workspace, raw_state,
                k_norm, model,
                complete_blocks, dynamic);
        return;
    case SmallWeightKind::f16:
        launch_block_scores_typed<
            SmallWeightKind::f16>(
                workspace, raw_state,
                k_norm, model,
                complete_blocks, dynamic);
        return;
    case SmallWeightKind::bf16:
        launch_block_scores_typed<
            SmallWeightKind::bf16>(
                workspace, raw_state,
                k_norm, model,
                complete_blocks, dynamic);
        return;
    }
    throw std::logic_error(
        "unreachable indexed k-norm kind");
}

const SerialSessionStateLayout&
state_layout(
    const ExecutionDispatchContext& context) {
    if (context.resources != nullptr &&
        context.resources->
            session_state_layout !=
            nullptr) {
        return *context.resources->
            session_state_layout;
    }
    if (context.session_state_layout !=
        nullptr) {
        return *context.session_state_layout;
    }
    throw std::invalid_argument(
        "HIP indexed attention requires session-state layout");
}

const SessionStateMemoryBinding&
state_memory(
    const ExecutionDispatchContext& context) {
    if (context.resources != nullptr &&
        context.resources->
            session_state_memory !=
            nullptr) {
        return *context.resources->
            session_state_memory;
    }
    if (context.session_state_memory !=
        nullptr) {
        return *context.session_state_memory;
    }
    throw std::invalid_argument(
        "HIP indexed attention requires session-state memory");
}

WorkspacePointers workspace_pointers(
    const ExecutionBufferAddress& temporary,
    const IndexedAttentionWorkspaceLayout& layout) {
    const auto base =
        temporary.address;
    return WorkspacePointers{
        reinterpret_cast<float*>(
            base +
            layout.index_query_offset),
        reinterpret_cast<float*>(
            base +
            layout.current_index_key_offset),
        layout.block_scores_bytes ? reinterpret_cast<float*>(base + layout.block_scores_offset) : nullptr,
        reinterpret_cast<std::uint32_t*>(
            base +
            layout.selected_indices_offset),
        reinterpret_cast<float*>(
            base +
            layout.q_gate_offset),
        reinterpret_cast<float*>(
            base +
            layout.current_k_offset),
        reinterpret_cast<float*>(
            base +
            layout.current_v_offset),
        reinterpret_cast<float*>(
            base +
            layout.attention_scores_offset),
        reinterpret_cast<float*>(
            base +
            layout.attention_output_offset),
        layout.selection_ping_bytes ? reinterpret_cast<qsa_selection_diagnostic::Item*>(base + layout.selection_ping_offset) : nullptr,
        layout.selection_pong_bytes ? reinterpret_cast<qsa_selection_diagnostic::Item*>(base + layout.selection_pong_offset) : nullptr,
        layout.selection_status_bytes ? reinterpret_cast<std::uint32_t*>(base + layout.selection_status_offset) : nullptr,
        layout.attention_position_bytes ? reinterpret_cast<std::uint64_t*>(base + layout.attention_position_offset) : nullptr,
        layout.compact_attention_output,
    };
}

// Complete metadata-only preparation. GPU input values are never read here.
struct QsaPrefillPool {
    std::uint32_t mtp_role=13U;
    float* keys = nullptr;
    // STRIX_QSA_POOL_PACKED=1 (prefill pool, opt-in; the baseline keeps it off): the same keys as BF16 bits, written by
    // the pool kernels together with keys and read by qsa_flash::score_kernel instead of them (half the key reads, same
    // bits). Off by default: with 32/64 rows per CTA the scorer is arithmetic-bound and the widening costs more than the
    // reads save (tests/qsa_score_rows.hip: +3-4 % at <= 32768 blocks, -3 % at 129024 / 262144 vs 32 rows, +5 % vs 64).
    std::uint16_t* packed = nullptr;
    std::uint32_t capacity = 0, valid = 0;
    std::shared_ptr<const HipQwenRotaryTable> rotary_table;
};
std::mutex& qsa_prefill_pool_mutex() { static std::mutex mutex;return mutex; }
QsaPrefillPool& qsa_prefill_pool_storage() { static QsaPrefillPool pool;return pool; }
} // namespace
#include "hip_qsa_committed_pool.inc"
namespace {

// Diagnostic replay uses captured score bytes already checked against raw keys.
// It restores both score mutations and selected IDs before any downstream consumer.
class MtpPoolSelectionCheck {
    std::shared_ptr<HipQsaCommittedPool> owner_;
    HipQsaPoolRowDescriptor row_{};
    hipStream_t stream_=nullptr;
    std::vector<float> before_;
public:
    MtpPoolSelectionCheck(std::shared_ptr<HipQsaCommittedPool> owner,HipQsaPoolRowDescriptor row,hipStream_t stream)
        :owner_(std::move(owner)),row_(row),stream_(stream) {
        const char* v=std::getenv("STRIX_QSA_MTP_POOL_CHECK");
        if (!owner_ || !v || v[0]!='1' || row.complete_blocks<=row.topk) {owner_.reset();return;}
        before_.resize(row.complete_blocks);
        check_hip(hipMemcpyAsync(before_.data(),row.scores,before_.size()*sizeof(float),hipMemcpyDeviceToHost,stream_),"MTP selection check before");
        check_hip(hipStreamSynchronize(stream_),"MTP selection check before completion");
    }
    void finish() {
        if (!owner_) return;
        const auto count=std::min(row_.complete_blocks,row_.topk)*4U+row_.visible%4U;
        std::vector<float> after(before_.size()),reference(before_.size());
        std::vector<std::uint32_t> selected(count),ref_selected(count);
        const auto pull=[&](auto& scores,auto& ids) {
            check_hip(hipMemcpyAsync(scores.data(),row_.scores,scores.size()*sizeof(float),hipMemcpyDeviceToHost,stream_),"MTP selection check scores");
            check_hip(hipMemcpyAsync(ids.data(),row_.selected,ids.size()*sizeof(std::uint32_t),hipMemcpyDeviceToHost,stream_),"MTP selection check indices");
            check_hip(hipStreamSynchronize(stream_),"MTP selection check result completion");
        };
        pull(after,selected);
        check_hip(hipMemcpyAsync(row_.scores,before_.data(),before_.size()*sizeof(float),hipMemcpyHostToDevice,stream_),"MTP selection check input restore");
        hipLaunchKernelGGL(select_blocks_kernel,dim3(1),dim3(kSelectThreads),0,stream_,row_.scores,row_.complete_blocks,row_.topk,4U,
            row_.visible,row_.selected,GraphSelectionArguments{});
        check_hip(hipGetLastError(),"MTP selection check reference launch");pull(reference,ref_selected);
        const bool equal=selected==ref_selected && std::memcmp(after.data(),reference.data(),after.size()*sizeof(float))==0;
        check_hip(hipMemcpyAsync(row_.scores,after.data(),after.size()*sizeof(float),hipMemcpyHostToDevice,stream_),"MTP selection check output restore");
        check_hip(hipMemcpyAsync(row_.selected,selected.data(),selected.size()*sizeof(std::uint32_t),hipMemcpyHostToDevice,stream_),"MTP selection check indices restore");
        check_hip(hipStreamSynchronize(stream_),"MTP selection check restore completion");
        owner_->checked(count+before_.size(),equal?0U:1U);
    }
};
struct PreparedIndexedLaunch {
    ExecutionBufferAddress input, output, temporary;
    IndexedAttentionWorkspaceLayout workspace_layout;
    WorkspacePointers workspace;
    const SerialSessionStateLayout* session_layout;
    GraphPositionKernelArguments dynamic_position;
    bool device_graph;
    SessionStateAddress kv_state, raw_state;
    IndexedTensors tensors;
    std::uint32_t hidden, attention_heads, kv_heads, head_dim, attention_width, kv_width;
    std::uint32_t q_gate_width, index_q_width, index_dim;
};
struct PreparedIndexedSelection {
    std::uint32_t visible, ratio, block_topk, complete_blocks, selected_count;
    GraphSelectionArguments dynamic_selection;
};
// Eager serial step with HipIndexedAttentionLauncherConfig::decode_pool: block scores from the persistent pooled keys
// (defined after HipQsaWindowPool). False when the pool cannot serve this step; the caller then runs block_score_kernel.
bool qsa_decode_pool_scores(const PreparedIndexedLaunch& a, const HipIndexedAttentionLauncherConfig& config, std::uint64_t position,
    std::uint32_t complete_blocks, std::uint32_t ratio, QsaPrefillPool* borrowed = nullptr,
    const std::shared_ptr<HipQsaCommittedPool>& committed_owner = {}, bool dynamic_position = true,
    const float** fused_maxima = nullptr);
// After qsa_decode_pool_scores: the selection from the scores' 4-block group maxima (qsa_flash::select_row_body, the
// selected set, order and -inf writes of select_blocks_kernel). False when not enabled or no maxima scratch.
bool qsa_decode_pool_select(const PreparedIndexedLaunch& a, const HipIndexedAttentionLauncherConfig& config,
    const PreparedIndexedSelection& selection, const float* fused_maxima = nullptr);

PreparedIndexedLaunch prepare_indexed_attention_launch(
    const ExecutionDispatchContext& context, const HipIndexedAttentionLauncherConfig* config_ptr) {
    if (context.op !=
        SerialOp::indexed_attention) {
        throw std::invalid_argument(
            "HIP indexed-attention launcher attached to wrong operator");
    }
    if (context.reads.size() != 1 ||
        context.writes.size() != 1 ||
        context.temporaries.size() != 1 ||
        !context.layer.has_value()) {
        throw std::invalid_argument(
            "HIP indexed attention requires one read/write/temp and a layer");
    }
    if (context.logical_rows != 1 ||
        context.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP indexed-attention v1 supports only C1 one-row execution");
    }
    if (config_ptr == nullptr) {
        throw std::invalid_argument(
            "HIP indexed attention requires configuration");
    }

    const auto& config = *config_ptr;
    validate_config(config);
    if (config.wave8_q8_output) {
        const auto* scope = current_hip_launch_context();
        if (scope && scope->graph_capture)
            throw std::invalid_argument("wave8 QSA output supports eager execution only");
        hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
        check_hip(hipStreamIsCapturing(hip_execution_stream(), &capture),
            "HIP wave8 QSA output capture query");
        if (capture != hipStreamCaptureStatusNone)
            throw std::invalid_argument("wave8 QSA output supports eager execution only");
    }
    if (config.qsa_hierarchical_selection || config.qsa_gqa_heads_per_block != 1) {
        const auto* scope = current_hip_launch_context();
        if (!scope || scope->graph_capture || !scope->stream || scope->stream == hipStreamLegacy ||
            scope->stream == hipStreamPerThread || !hip_deferred_moe_status() ||
            scope->deferred_moe_status.execution_identity != context.execution_identity ||
            scope->deferred_moe_status.arena_bytes != 8 || !scope->deferred_moe_status.allocation_generation ||
            scope->deferred_moe_status.arena_base % alignof(DeferredMoeFirstError) ||
            scope->deferred_moe_status.arena_base > UINTPTR_MAX - 8U)
            throw std::invalid_argument("hierarchical QSA requires its current owned eager D1 scope");
        hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
        check_hip(hipStreamIsCapturing(scope->stream, &capture), "HIP hierarchical QSA capture query");
        if (capture != hipStreamCaptureStatusNone)
            throw std::invalid_argument("hierarchical QSA supports eager execution only");
    }
    const auto& model =
        config.model;
    check_qwen_rotary_position(model, context.position);

    const auto input =
        resolve_dispatch_read(
            context, 0);
    const auto output =
        resolve_dispatch_write(
            context, 0);
    const auto temporary =
        resolve_dispatch_temporary(
            context, 0);

    auto workspace_layout = build_indexed_attention_workspace_layout(
        model, 256, config.qsa_hierarchical_selection, config.qsa_gqa_heads_per_block);
    if (!config.qsa_hierarchical_selection && config.qsa_gqa_heads_per_block==1 && !config.enable_device_graph_inputs) {
        const auto compact=build_indexed_attention_workspace_layout(model,256,false,1,true);
        if(temporary.bytes==compact.total_bytes) workspace_layout=compact;
        const auto images=build_indexed_attention_workspace_layout(model,256,false,1,true,true);
        if(images.total_bytes<compact.total_bytes && temporary.bytes==images.total_bytes) workspace_layout=images;
    }
    if (temporary.bytes <
        workspace_layout.total_bytes) {
        throw std::invalid_argument(
            "indexed-attention temporary is undersized");
    }
    const auto workspace =
        workspace_pointers(
            temporary,
            workspace_layout);

    const auto& session_layout =
        state_layout(context);
    const auto dynamic_position = graph_position_arguments();
    const bool device_graph = dynamic_position.position != nullptr;
    if (!device_graph && context.position >=
        session_layout.context_tokens) {
        throw std::out_of_range(
            "indexed-attention position exceeds session capacity");
    }
    const auto& session_memory =
        state_memory(context);
    const auto kv_state =
        resolve_indexed_attention_layer_state(
            session_layout,
            session_memory,
            model,
            *context.layer,
            SessionStateComponent::
                attention_kv);
    const auto raw_state =
        resolve_indexed_attention_layer_state(
            session_layout,
            session_memory,
            model,
            *context.layer,
            SessionStateComponent::
                sparse_index);
    if (kv_state.storage !=
            SessionStateStorage::device ||
        raw_state.storage !=
            SessionStateStorage::device) {
        throw std::invalid_argument(
            "indexed-attention KV/indexer state must be device resident");
    }

    const auto state_contract =
        build_indexed_attention_state_layout(
            model);
    if (kv_state.bytes <
            session_layout.context_tokens *
                state_contract.kv_row_bytes ||
        raw_state.bytes <
            session_layout.context_tokens *
                state_contract.raw_index_row_bytes) {
        throw std::invalid_argument(
            "indexed-attention state slices are undersized");
    }

    const auto tensors =
        identify_tensors(context);

    const auto hidden =
        checked_u32(
            model.hidden_size,
            "hidden_size");
    const auto attention_heads =
        checked_u32(
            model.attention_heads,
            "attention_heads");
    const auto kv_heads =
        checked_u32(
            model.kv_heads,
            "kv_heads");
    const auto head_dim =
        checked_u32(
            model.head_dim,
            "head_dim");
    const auto attention_width =
        checked_u32(
            model.attention_heads *
                model.head_dim,
            "attention_width");
    const auto kv_width =
        checked_u32(
            model.kv_heads *
                model.head_dim,
            "kv_width");
    const auto q_gate_width =
        checked_u32(
            model.attention_heads *
                model.head_dim *
                2ULL,
            "q_gate_width");
    const auto index_q_width =
        checked_u32(
            model.indexer_n_heads *
                model.indexer_head_dim,
            "index_q_width");
    const auto index_dim =
        checked_u32(
            model.indexer_head_dim,
            "indexer_head_dim");

    validate_matrix(
        tensors.q,
        hidden,
        q_gate_width,
        "attention q/gate");
    validate_matrix(
        tensors.k,
        hidden,
        kv_width,
        "attention k");
    validate_matrix(
        tensors.v,
        hidden,
        kv_width,
        "attention v");
    validate_matrix(
        tensors.out,
        attention_width,
        hidden,
        "attention output");
    validate_matrix(
        tensors.index_q,
        hidden,
        index_q_width,
        "indexer q");
    validate_matrix(
        tensors.index_k,
        hidden,
        index_dim,
        "indexer k");
    validate_norm(
        tensors.q_norm,
        head_dim,
        "attention q norm");
    validate_norm(
        tensors.k_norm,
        head_dim,
        "attention k norm");
    validate_norm(
        tensors.index_q_norm,
        index_dim,
        "indexer q norm");
    validate_norm(
        tensors.index_k_norm,
        index_dim,
        "indexer k norm");

    const auto input_required =
        static_cast<std::uint64_t>(
            hidden) *
        model.bytes_bf16;
    if (input.bytes <
            input_required ||
        output.bytes <
            input_required) {
        throw std::invalid_argument(
            "indexed-attention hidden buffers are undersized");
    }

    // GSQ-RCO (STRIX_GSQ_Q20_A8=1): a non-Q8 output projection runs the typed dense A8 sites instead of wave8.
    const bool gsq_out = config.dense_a8_projections && tensors.out.manifest->ggml_type != 8U &&
        q8_rows::dense_a8_type(tensors.out.manifest->ggml_type);
    if (config.wave8_q8_output && !gsq_out) {
        if (tensors.out.manifest->ggml_type != 8U)
            throw std::invalid_argument("wave8 QSA output requires aligned Q8_0 weights");
        q8_projection_organization::validate_fp32_bf16_views(attention_width, hidden,
            {reinterpret_cast<std::uintptr_t>(workspace.attention_output),
             static_cast<std::uint64_t>(attention_width) * (workspace.compact_attention_output ? 2U : 4U)},
            {tensors.out.runtime->address, tensors.out.runtime->bytes},
            {output.address, output.bytes}, workspace.compact_attention_output);
    }
    if(config.prefetch_q8_input){
        if(!qsa_input_prefetch::selected(true,config.prefetch_eligible_layers,context.layer))throw std::invalid_argument("QSA prefetch launch layer differs from admitted plan");
        if(input.address%2U!=0 || output.address%2U!=0 || temporary.address%4U!=0 || kv_state.address%2U!=0 || raw_state.address%2U!=0)throw std::invalid_argument("QSA prefetch owner alignment invalid");
        unsigned eligible=0;
        for(const auto* tensor:{&tensors.q,&tensors.k,&tensors.v,&tensors.index_q,&tensors.index_k})if(qsa_input_prefetch::eligible(*tensor->manifest)){
            ++eligible;const auto bytes=qsa_input_prefetch::required_weight_bytes(tensor->manifest->dimensions[0],tensor->manifest->dimensions[1]);
            if(tensor->manifest->absolute_offset%16U!=0 || tensor->runtime->address%16U!=0 || tensor->manifest->byte_size<bytes || tensor->runtime->bytes<bytes)throw std::invalid_argument("QSA prefetch weight alignment/extent invalid");
        }
        if(eligible==0)throw std::invalid_argument("QSA prefetch step has no eligible projection");
        struct Range{std::uintptr_t address;std::uint64_t bytes;};
        std::array<Range,11> reads{};reads[0]={input.address,input.bytes};unsigned read_index=1;
        for(const auto* tensor:{&tensors.q,&tensors.k,&tensors.v,&tensors.out,&tensors.q_norm,&tensors.k_norm,&tensors.index_q,&tensors.index_k,&tensors.index_q_norm,&tensors.index_k_norm})reads[read_index++]={tensor->runtime->address,tensor->runtime->bytes};
        const std::array<Range,4> writes{{{temporary.address,temporary.bytes},{output.address,output.bytes},{kv_state.address,kv_state.bytes},{raw_state.address,raw_state.bytes}}};
        auto valid=[](Range view){return view.address!=0 && view.bytes!=0 && view.bytes<=UINTPTR_MAX-view.address;};
        auto overlap=[](Range x,Range y){return x.address<y.address+y.bytes && y.address<x.address+x.bytes;};
        for(auto view:reads)if(!valid(view))throw std::invalid_argument("QSA prefetch invalid read owner");
        for(unsigned i=0;i<writes.size();++i){if(!valid(writes[i]))throw std::invalid_argument("QSA prefetch invalid writable owner");for(auto view:reads)if(overlap(writes[i],view))throw std::invalid_argument("QSA prefetch writable/read owner overlap");for(unsigned j=0;j<i;++j)if(overlap(writes[i],writes[j]))throw std::invalid_argument("QSA prefetch writable owner overlap");}
    }

    if (config.qsa_hierarchical_selection || config.qsa_gqa_heads_per_block != 1) {
        if ((config.qsa_hierarchical_selection && session_layout.context_tokens > qsa_hierarchical_selection_max_visible) ||
            session_layout.context_tokens > model.effective_max_positions() || temporary.address % 8U ||
            !context.layer || *context.layer >= 48 || device_graph)
            throw std::invalid_argument("hierarchical QSA context, layer or workspace alignment is outside admission");
        const auto& record = current_hip_launch_context()->deferred_moe_status;
        struct Range { std::uintptr_t address; std::uint64_t bytes; };
        const auto valid = [](Range x) { return x.address && x.bytes && x.bytes <= UINTPTR_MAX - x.address; };
        const auto overlaps = [](Range x, Range y) { return x.address < y.address + y.bytes && y.address < x.address + x.bytes; };
        const Range status{record.arena_base, record.arena_bytes};
        std::array<Range, 11> reads{};
        reads[0] = {input.address, input.bytes};
        std::size_t read_index = 1;
        for (const auto* tensor : {&tensors.q, &tensors.k, &tensors.v, &tensors.out, &tensors.q_norm,
            &tensors.k_norm, &tensors.index_q, &tensors.index_k, &tensors.index_q_norm, &tensors.index_k_norm})
            reads[read_index++] = {tensor->runtime->address, tensor->runtime->bytes};
        const std::array<Range, 4> writes{{{temporary.address, temporary.bytes}, {output.address, output.bytes},
            {kv_state.address, kv_state.bytes}, {raw_state.address, raw_state.bytes}}};
        for (const auto read : reads)
            if (!valid(read) || overlaps(status, read)) throw std::invalid_argument("hierarchical QSA invalid or status-aliasing read owner");
        for (std::size_t i = 0; i < writes.size(); ++i) {
            if (!valid(writes[i]) || overlaps(status, writes[i]))
                throw std::invalid_argument("hierarchical QSA invalid or status-aliasing write owner");
            for (const auto read : reads) if (overlaps(writes[i], read))
                throw std::invalid_argument("hierarchical QSA writable/read owner overlap");
            for (std::size_t j = 0; j < i; ++j) if (overlaps(writes[i], writes[j]))
                throw std::invalid_argument("hierarchical QSA writable owner overlap");
        }
        hipPointerAttribute_t attributes{};
        check_hip(hipPointerGetAttributes(&attributes, reinterpret_cast<const void*>(record.arena_base)),
            "HIP hierarchical QSA status device");
        int device = -1;
        check_hip(hipGetDevice(&device), "HIP hierarchical QSA current device");
        if (attributes.type != hipMemoryTypeDevice || attributes.isManaged || attributes.device != device)
            throw std::invalid_argument("hierarchical QSA D1 record is not on the current device");
        hipDeviceptr_t allocation{}; std::size_t bytes = 0;
        check_hip(hipMemGetAddressRange(&allocation, &bytes, reinterpret_cast<hipDeviceptr_t>(record.arena_base)),
            "HIP hierarchical QSA status extent");
        const auto base = reinterpret_cast<std::uintptr_t>(allocation);
        if (record.arena_base < base || record.arena_base - base > bytes || 8U > bytes - (record.arena_base - base))
            throw std::invalid_argument("hierarchical QSA D1 record exceeds physical owner");
    }
    if (device_graph) {
        const auto& scope = *current_hip_launch_context();
        if (!config.enable_device_graph_inputs || scope.context_tokens != session_layout.context_tokens ||
            scope.context_tokens > model.effective_max_positions() ||
            scope.context_tokens > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("indexed-attention device graph context/configuration is invalid");
        }
        GraphOperatorControlRequest request;
        request.execution_identity = context.execution_identity;
        request.context_tokens = scope.context_tokens;
        request.arguments = scope.arguments;
        request.status = scope.status;
        for (const auto* view : {&input, &output}) request.resources.push_back(
            {view->address, view->bytes, view->allocation_generation, 2});
        request.resources.push_back({temporary.address, temporary.bytes, temporary.allocation_generation, 4});
        for (const auto* state : {&kv_state, &raw_state}) request.resources.push_back(
            {state->address, state->bytes, state->allocation_generation, 2});
        for (const auto* tensor : {&tensors.q, &tensors.k, &tensors.v, &tensors.out,
            &tensors.q_norm, &tensors.k_norm, &tensors.index_q, &tensors.index_k,
            &tensors.index_q_norm, &tensors.index_k_norm}) request.resources.push_back(
            {tensor->runtime->address, tensor->runtime->bytes, tensor->runtime->allocation_generation,
             tensor->manifest->ggml_type == 0 ? 4U : 2U});
        (void)prepare_graph_operator_control(request);
    }
    if (config.qsa_hierarchical_selection || config.qsa_gqa_heads_per_block != 1)
        check_hip(hipPeekAtLastError(), "HIP hierarchical QSA preexisting error");
    return {input, output, temporary, workspace_layout, workspace, &session_layout,
            dynamic_position, device_graph, kv_state, raw_state, tensors, hidden,
            attention_heads, kv_heads, head_dim, attention_width, kv_width,
            q_gate_width, index_q_width, index_dim};
}

// A prefill row of the same step as `head_context`, admitted by prepare_indexed_attention_launch:
// the configuration, the step's tensors, the layer's KV/indexer state slices and the session
// layout are those of the head row (checked by context identity below), so they are copied;
// only the row's own views (input, output, temporary and its workspace) are resolved, and
// every check that reads them, or the row's position, is repeated as the full admission does.
PreparedIndexedLaunch prepare_indexed_attention_launch_like(const PreparedIndexedLaunch& head,
    const ExecutionDispatchContext& head_context, const ExecutionDispatchContext& context,
    const HipIndexedAttentionLauncherConfig& config) {
    const auto& a = head_context;
    if (context.op != SerialOp::indexed_attention || context.op != a.op || context.step_id != a.step_id ||
        context.layer != a.layer || !context.layer || context.execution_identity != a.execution_identity ||
        context.kernel_id != a.kernel_id || context.resources != a.resources || context.arena == nullptr ||
        context.session_state_layout != a.session_state_layout || context.session_state_memory != a.session_state_memory ||
        context.memory != a.memory || context.prepared_tensor_views != a.prepared_tensor_views ||
        context.prepared_arena_views != a.prepared_arena_views || context.logical_rows != 1U || context.request_concurrency != 1U ||
        context.reads.size() != 1U || context.writes.size() != 1U || context.temporaries.size() != 1U ||
        context.reads[0] != a.reads[0] || context.writes[0] != a.writes[0] || context.temporaries[0] != a.temporaries[0] ||
        context.reads_state != a.reads_state || context.writes_state != a.writes_state || head.device_graph ||
        config.qsa_hierarchical_selection || config.qsa_gqa_heads_per_block != 1)
        throw std::invalid_argument("QSA prefill row differs from its head row");
    check_qwen_rotary_position(config.model, context.position);
    PreparedIndexedLaunch p = head;
    // Rows sharing a head are row views of one known-window/prefill arena (unique uses, same order):
    // the previous row's allocation indices admit this row without searching its arena.
    thread_local std::array<std::size_t, 3> hint{};
    p.input = resolve_dispatch_read_hinted(context, 0, hint[0]);
    p.output = resolve_dispatch_write_hinted(context, 0, hint[1]);
    p.temporary = resolve_dispatch_temporary_hinted(context, 0, hint[2]);
    if (p.temporary.bytes < p.workspace_layout.total_bytes)
        throw std::invalid_argument("indexed-attention temporary is undersized");
    p.workspace = workspace_pointers(p.temporary, p.workspace_layout);
    if (context.position >= p.session_layout->context_tokens)
        throw std::out_of_range("indexed-attention position exceeds session capacity");
    const auto input_required = static_cast<std::uint64_t>(p.hidden) * config.model.bytes_bf16;
    if (p.input.bytes < input_required || p.output.bytes < input_required)
        throw std::invalid_argument("indexed-attention hidden buffers are undersized");
    const auto& tensors = p.tensors;
    if (config.wave8_q8_output && !(config.dense_a8_projections && tensors.out.manifest->ggml_type != 8U &&
            q8_rows::dense_a8_type(tensors.out.manifest->ggml_type)))
        q8_projection_organization::validate_fp32_bf16_views(p.attention_width, p.hidden,
            {reinterpret_cast<std::uintptr_t>(p.workspace.attention_output),
             static_cast<std::uint64_t>(p.attention_width) * (p.workspace.compact_attention_output ? 2U : 4U)},
            {tensors.out.runtime->address, tensors.out.runtime->bytes},
            {p.output.address, p.output.bytes}, p.workspace.compact_attention_output);
    if (config.prefetch_q8_input) {
        if (p.input.address % 2U != 0 || p.output.address % 2U != 0 || p.temporary.address % 4U != 0)
            throw std::invalid_argument("QSA prefetch owner alignment invalid");
        struct Range { std::uintptr_t address; std::uint64_t bytes; };
        std::array<Range, 11> reads{}; reads[0] = {p.input.address, p.input.bytes}; unsigned read_index = 1;
        for (const auto* tensor : {&tensors.q, &tensors.k, &tensors.v, &tensors.out, &tensors.q_norm, &tensors.k_norm,
            &tensors.index_q, &tensors.index_k, &tensors.index_q_norm, &tensors.index_k_norm})
            reads[read_index++] = {tensor->runtime->address, tensor->runtime->bytes};
        const std::array<Range, 4> writes{{{p.temporary.address, p.temporary.bytes}, {p.output.address, p.output.bytes},
            {p.kv_state.address, p.kv_state.bytes}, {p.raw_state.address, p.raw_state.bytes}}};
        const auto valid = [](Range view) { return view.address != 0 && view.bytes != 0 && view.bytes <= UINTPTR_MAX - view.address; };
        const auto overlap = [](Range x, Range y) { return x.address < y.address + y.bytes && y.address < x.address + x.bytes; };
        for (const auto view : reads) if (!valid(view)) throw std::invalid_argument("QSA prefetch invalid read owner");
        for (unsigned i = 0; i < writes.size(); ++i) {
            if (!valid(writes[i])) throw std::invalid_argument("QSA prefetch invalid writable owner");
            for (const auto view : reads) if (overlap(writes[i], view)) throw std::invalid_argument("QSA prefetch writable/read owner overlap");
            for (unsigned j = 0; j < i; ++j) if (overlap(writes[i], writes[j])) throw std::invalid_argument("QSA prefetch writable owner overlap");
        }
    }
    return p;
}

PreparedIndexedSelection prepare_indexed_attention_selection(const ExecutionDispatchContext& context,
    const PreparedIndexedLaunch& prepared, const ModelContract& model) {
    const auto& session_layout = *prepared.session_layout;
    const auto& workspace_layout = prepared.workspace_layout;
    const auto dynamic_position = prepared.dynamic_position;
    const auto device_graph = prepared.device_graph;
    const auto visible64 =
        device_graph ? session_layout.context_tokens : context.position + 1ULL;
    if (visible64 >
        std::numeric_limits<
            std::uint32_t>::max()) {
        throw std::overflow_error(
            "indexed-attention visible token count exceeds uint32");
    }
    const auto visible =
        static_cast<std::uint32_t>(
            visible64);
    const auto ratio =
        checked_u32(
            model.indexer_compress_ratio,
            "indexer_compress_ratio");
    const auto budget =
        checked_u32(
            model.indexer_budget,
            "indexer_budget");
    const auto complete_blocks =
        visible / ratio;
    const auto block_topk =
        budget / ratio;
    const auto selected_blocks =
        std::min(
            complete_blocks,
            block_topk);
    const auto tail =
        visible -
        complete_blocks *
            ratio;
    const auto selected_count = device_graph ? checked_u32(workspace_layout.maximum_selected_tokens, "maximum_selected_tokens") :
        selected_blocks *
            ratio +
        tail;

    if (selected_count == 0 ||
        selected_count >
            workspace_layout.
                maximum_selected_tokens) {
        throw std::logic_error(
            "indexed-attention selected-token count exceeds workspace");
    }

    const GraphSelectionArguments dynamic_selection{dynamic_position, ratio, block_topk};
    return {visible, ratio, block_topk, complete_blocks, selected_count, dynamic_selection};
}

std::uint64_t prepare_indexed_attention_score_grid(std::uint32_t attention_heads,
    std::uint32_t selected_count) {
    const auto score_blocks64 =
        static_cast<std::uint64_t>(
            attention_heads) *
        selected_count;
    if (score_blocks64 >
        std::numeric_limits<
            std::uint32_t>::max()) {
        throw std::overflow_error(
            "indexed-attention score grid exceeds uint32");
    }
    return score_blocks64;
}

void preflight_indexed_attention(const ExecutionDispatchContext& context, void* user_data) {
    const auto prepared = prepare_indexed_attention_launch(context,
        static_cast<const HipIndexedAttentionLauncherConfig*>(user_data));
    require_qsa_score_source(!prepared.workspace_layout.block_scores_bytes,prepared.workspace.block_scores!=nullptr);
    const auto& model = static_cast<const HipIndexedAttentionLauncherConfig*>(user_data)->model;
    // These guards otherwise occur in nested launch helpers after preceding
    // projections. Preflight evaluates metadata only; the normal launch keeps
    // each original check at its original phase.
    static_cast<void>(checked_u32(model.indexer_n_heads, "indexer_n_heads"));
    static_cast<void>(checked_u32(model.rotary_dim(), "rotary_dim"));
    const auto selection = prepare_indexed_attention_selection(context, prepared, model);
    static_cast<void>(prepare_indexed_attention_score_grid(prepared.attention_heads, selection.selected_count));
    static_cast<void>(checked_u32(prepared.workspace_layout.maximum_selected_tokens, "maximum_selected_tokens"));
}

// Dense A8 R1 projections (defined with the rows sites below).
void launch_qsa_dense_a8_inputs(const std::array<const TensorPair*, 5>& matrices, const std::array<std::uint32_t, 5>& outputs,
    const std::array<float*, 5>& targets, const ExecutionBufferAddress& input, std::uint32_t hidden, hipStream_t stream, unsigned skip = 0U);
void launch_qsa_dense_a8_output(const TensorPair& weights, const float* input, const ExecutionBufferAddress& output,
    std::uint32_t width, std::uint32_t hidden, hipStream_t stream);
// Plain R1 attention score with four query heads per K row (defined with the draft v2 score below); false = not
// admitted, the caller launches attention_score_kernel.
bool launch_decode_r1_scores_v2(const float* q_gate, const std::uint16_t* kv, const std::uint32_t* selected, float* scores,
    std::uint32_t count, std::uint32_t maximum, std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t dim,
    const GraphSelectionArguments& selection, bool device_graph, hipStream_t stream);
// Plain R1 value/gate on the softmax's probabilities with the draft step's two-launch value kernel (defined with it);
// false = not admitted, the caller launches attention_output_kernel.
bool launch_decode_r1_value_v2(const float* q_gate, const std::uint16_t* kv, const std::uint32_t* selected, const float* scores,
    std::uint32_t count, std::uint32_t maximum, std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t dim, float* output,
    const GraphSelectionArguments& selection, bool device_graph, hipStream_t stream);

void launch_indexed_attention(const ExecutionDispatchContext& context, void* user_data) {
    const auto prepared = prepare_indexed_attention_launch(context,
        static_cast<const HipIndexedAttentionLauncherConfig*>(user_data));
    require_qsa_score_source(!prepared.workspace_layout.block_scores_bytes,prepared.workspace.block_scores!=nullptr);
    const auto& config = *static_cast<const HipIndexedAttentionLauncherConfig*>(user_data);
    const auto& committed_pool=(context.resources && context.resources->qsa_committed_pool)
        ? context.resources->qsa_committed_pool : config.committed_pool;
    std::unique_ptr<HipQsaCommittedAccess> committed_access;
    if (committed_pool) committed_access=committed_pool->before_rewrite(false,
        prepared.raw_state.address,prepared.raw_state.bytes,prepared.raw_state.allocation_generation,
        *prepared.tensors.index_k_norm.runtime,prepared.tensors.index_k_norm.manifest->ggml_type,config.model,
        context.position,1U,hip_execution_stream(),!(context.resources && context.resources->qsa_pool_speculative));
#if defined(STRIX_HAS_EXPERIMENTAL_DENSE_TRELLIS)
    HipDenseTrellisScope arithmetic(config.dense_trellis_projections);
#endif
    const auto& model = config.model;
    const auto& input = prepared.input; const auto& output = prepared.output;
    const auto& workspace_layout = prepared.workspace_layout; const auto& workspace = prepared.workspace;
    const auto dynamic_position = prepared.dynamic_position;
    const auto device_graph = prepared.device_graph;
    const auto& kv_state = prepared.kv_state; const auto& raw_state = prepared.raw_state;
    const auto& tensors = prepared.tensors;
    const auto hidden = prepared.hidden, attention_heads = prepared.attention_heads, kv_heads = prepared.kv_heads;
    const auto head_dim = prepared.head_dim, attention_width = prepared.attention_width, kv_width = prepared.kv_width;
    const auto q_gate_width = prepared.q_gate_width, index_q_width = prepared.index_q_width, index_dim = prepared.index_dim;
    if (device_graph) {
        hipLaunchKernelGGL(validate_indexed_graph_control, dim3(1), dim3(kThreads), 0,
            hip_execution_stream(), dynamic_position,
            reinterpret_cast<const std::uint16_t*>(input.address), hidden);
        check_hip(hipGetLastError(), "HIP indexed-attention device control validation");
    }

    // Dense A8: every Q8_0 input matrix (indexer and main) in one launch up front; the
    // other formats keep their R1 kernels at their usual places.
    const bool dense = config.dense_a8_projections;
    // STRIX_QSA_SKIP_INDEX_Q=1: eager steps whose selection takes every visible token (complete
    // blocks <= block_topk, the fill_visible_indices branch below) never read the index query.
    bool skip_index_q = false;
    if (qsa_skip_index_q_enabled() && !device_graph) {
        const auto early = prepare_indexed_attention_selection(context, prepared, model);
        skip_index_q = early.complete_blocks <= early.block_topk;
    }
    if (dense) {
        if (device_graph) throw std::invalid_argument("HIP indexed-attention dense A8 projections run eager");
        launch_qsa_dense_a8_inputs({&tensors.index_q, &tensors.index_k, &tensors.q, &tensors.k, &tensors.v},
            {index_q_width, index_dim, q_gate_width, kv_width, kv_width},
            {workspace.index_query, workspace.current_index_key, workspace.q_gate, workspace.current_k, workspace.current_v},
            input, hidden, hip_execution_stream(), skip_index_q ? 1U : 0U);
    }
    const auto keep = [&](const TensorPair& tensor) { return !dense || !q8_rows::dense_a8_type(tensor.manifest->ggml_type); };

    // Indexer projections. Outputs are rounded to BF16 values represented in
    // FP32 workspace so subsequent FP32 normalization starts from model dtype.
    if (!skip_index_q && keep(tensors.index_q)) launch_linear_to_float(
        tensors.index_q,
        input,
        hidden,
        index_q_width,
        workspace.index_query, config.prefetch_q8_input, config.prefetch_stats.get());
    if (keep(tensors.index_k)) launch_linear_to_float(
        tensors.index_k,
        input,
        hidden,
        index_dim,
        workspace.current_index_key, config.prefetch_q8_input, config.prefetch_stats.get());

    if (!skip_index_q) launch_norm_rope(
        workspace.index_query,
        checked_u32(
            model.indexer_n_heads,
            "indexer_n_heads"),
        index_dim,
        index_dim,
        tensors.index_q_norm,
        model,
        context.position);

    const auto raw_blocks =
        (index_dim +
         kThreads - 1U) /
        kThreads;
    validate_graph_values(workspace.current_index_key, index_dim);
    hipLaunchKernelGGL(
        store_raw_index_key_kernel,
        dim3(raw_blocks),
        dim3(kThreads),
        0, hip_execution_stream(),
        workspace.current_index_key,
        reinterpret_cast<
            std::uint16_t*>(
                raw_state.address),
        index_dim,
        context.position, dynamic_position);
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention raw-index state update");

    const auto selection = prepare_indexed_attention_selection(context, prepared, model);
    const auto visible = selection.visible, ratio = selection.ratio, block_topk = selection.block_topk;
    const auto complete_blocks = selection.complete_blocks, selected_count = selection.selected_count;
    const auto dynamic_selection = selection.dynamic_selection;
    if (!device_graph && complete_blocks <=
        block_topk) {
        const auto blocks =
            (visible +
             kThreads - 1U) /
            kThreads;
        hipLaunchKernelGGL(
            fill_visible_indices_kernel,
            dim3(blocks),
            dim3(kThreads),
            0, hip_execution_stream(),
            workspace.selected_indices,
            visible);
        check_hip(
            hipGetLastError(),
            "HIP indexed-attention full-visible selection");
    } else {
        auto score_selection = dynamic_selection;
        if (config.qsa_hierarchical_selection) {
            // Reset before the producer, not between production and selection.
            // A null position retains eager scalar-position semantics while
            // enabling its existing nonfinite-intermediate guards. In
            // particular fmaxf(NaN, 0) must not silently admit a finite score.
            check_hip(hipMemsetAsync(workspace.selection_status, 0, sizeof(std::uint32_t), hip_execution_stream()),
                "HIP hierarchical QSA local status reset");
            score_selection.position.status = workspace.selection_status;
        }
        const float* fused_maxima = nullptr;
        const bool decode_pooled = !device_graph && !config.qsa_hierarchical_selection && (config.decode_pool || committed_access) &&
            qsa_decode_pool_scores(prepared, config, context.position, complete_blocks, ratio,
                committed_access ? static_cast<QsaPrefillPool*>(committed_access->storage()) : nullptr,committed_pool,
                dynamic_selection.position.position != nullptr, &fused_maxima);
        if (!decode_pooled)
        launch_block_scores(
            workspace,
            raw_state,
            tensors.index_k_norm,
            model,
            std::max(complete_blocks, 1U), score_selection);
        MtpPoolSelectionCheck selection_check(committed_pool,{workspace.index_query,workspace.block_scores,workspace.selected_indices,
            visible,complete_blocks,block_topk,0U,nullptr},hip_execution_stream());
        if (config.qsa_hierarchical_selection) {
            using namespace qsa_selection_diagnostic;
            const Shape shape{complete_blocks, visible - complete_blocks * ratio};
            const auto items = scratch_items(shape);
            enqueue(shape, {workspace.block_scores, complete_blocks}, {workspace.selected_indices, selected_count},
                {workspace.selection_ping, items}, {workspace.selection_pong, items},
                {workspace.selection_status, 1}, hip_execution_stream());
            hipLaunchKernelGGL(publish_qsa_selection_status, dim3((selected_count + kThreads - 1U) / kThreads),
                dim3(kThreads), 0, hip_execution_stream(), workspace.selection_status,
                reinterpret_cast<DeferredMoeFirstError*>(hip_deferred_moe_status()), workspace.selected_indices, selected_count);
            check_hip(hipGetLastError(), "HIP hierarchical QSA deferred status publication");
            if (config.selection_stats) {
                config.selection_stats->submissions.fetch_add(1, std::memory_order_relaxed);
                config.selection_stats->layer_submissions[*context.layer].fetch_add(1, std::memory_order_relaxed);
            }
        } else if (!decode_pooled || !qsa_decode_pool_select(prepared, config, selection, fused_maxima)) {
        hipLaunchKernelGGL(
            select_blocks_kernel,
            dim3(1),
            dim3(kSelectThreads),
            0, hip_execution_stream(),
            workspace.block_scores,
            complete_blocks,
            block_topk,
            ratio,
            visible,
            workspace.selected_indices, dynamic_selection);
        check_hip(
            hipGetLastError(),
            "HIP indexed-attention top-k block selection");
        }
        selection_check.finish();
    }

    // Main Q/K/V projections.
    if (keep(tensors.q)) launch_linear_to_float(
        tensors.q,
        input,
        hidden,
        q_gate_width,
        workspace.q_gate, config.prefetch_q8_input, config.prefetch_stats.get());
    if (keep(tensors.k)) launch_linear_to_float(
        tensors.k,
        input,
        hidden,
        kv_width,
        workspace.current_k, config.prefetch_q8_input, config.prefetch_stats.get());
    if (keep(tensors.v)) launch_linear_to_float(
        tensors.v,
        input,
        hidden,
        kv_width,
        workspace.current_v, config.prefetch_q8_input, config.prefetch_stats.get());

    launch_norm_rope(
        workspace.q_gate,
        attention_heads,
        head_dim,
        head_dim * 2U,
        tensors.q_norm,
        model,
        context.position);
    launch_norm_rope(
        workspace.current_k,
        kv_heads,
        head_dim,
        head_dim,
        tensors.k_norm,
        model,
        context.position);

    const auto kv_blocks =
        (kv_width +
         kThreads - 1U) /
        kThreads;
    validate_graph_values(workspace.current_v, kv_width);
    const bool kv_int8 = model.attention_kv_storage == AttentionKvStorage::int8_token;
    if (kv_int8) hipLaunchKernelGGL(store_kv_int8_kernel, dim3(kv_heads, 2U), dim3(kThreads), 0, hip_execution_stream(),
        workspace.current_k, workspace.current_v, reinterpret_cast<void*>(kv_state.address), kv_width, kv_heads, head_dim,
        context.position, dynamic_position);
    else hipLaunchKernelGGL(
        store_kv_kernel,
        dim3(kv_blocks),
        dim3(kThreads),
        0, hip_execution_stream(),
        workspace.current_k,
        workspace.current_v,
        reinterpret_cast<
            std::uint16_t*>(
                kv_state.address),
        kv_width,
        context.position, dynamic_position);
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention KV state update");

    if (config.qsa_gqa_heads_per_block != 1) {
        hipLaunchKernelGGL(prepare_qsa_gqa_control, dim3(1), dim3(1), 0, hip_execution_stream(),
            workspace.attention_position, workspace.selection_status, context.position,
            config.qsa_hierarchical_selection && complete_blocks > block_topk);
        check_hip(hipGetLastError(), "HIP QSA grouped attention control");
        using namespace qsa_gqa_reuse_diagnostic;
        const Shape shape{attention_heads, kv_heads, head_dim,
            checked_u32(prepared.session_layout->context_tokens, "QSA GQA context"),
            checked_u32(config.model.effective_max_positions(), "QSA GQA maximum")};
        const auto maximum = maximum_selected_tokens(shape);
        const Bindings views{
            {reinterpret_cast<std::uintptr_t>(workspace.q_gate), static_cast<std::uint64_t>(q_gate_width) * sizeof(float)},
            {kv_state.address, static_cast<std::uint64_t>(shape.context_tokens) * kv_width * 2U * sizeof(std::uint16_t)},
            {reinterpret_cast<std::uintptr_t>(workspace.selected_indices), static_cast<std::uint64_t>(maximum) * sizeof(std::uint32_t)},
            {reinterpret_cast<std::uintptr_t>(workspace.attention_position), sizeof(std::uint64_t)},
            {reinterpret_cast<std::uintptr_t>(workspace.selection_status), sizeof(std::uint32_t)},
            {reinterpret_cast<std::uintptr_t>(workspace.attention_scores), static_cast<std::uint64_t>(attention_heads) * maximum * sizeof(float)},
            {reinterpret_cast<std::uintptr_t>(workspace.attention_output), static_cast<std::uint64_t>(attention_width) * sizeof(float)}};
        enqueue(shape, views, config.qsa_gqa_heads_per_block == 2 ? Mode::heads2 : Mode::heads4, hip_execution_stream());
        hipLaunchKernelGGL(publish_qsa_attention_status, dim3(1), dim3(1), 0, hip_execution_stream(),
            workspace.selection_status, reinterpret_cast<DeferredMoeFirstError*>(hip_deferred_moe_status()));
        check_hip(hipGetLastError(), "HIP QSA grouped attention deferred publication");
        if (config.gqa_stats) {
            config.gqa_stats->submissions.fetch_add(1, std::memory_order_relaxed);
            config.gqa_stats->layer_submissions[*context.layer].fetch_add(1, std::memory_order_relaxed);
        }
    } else {
    const auto score_blocks64 = prepare_indexed_attention_score_grid(attention_heads, selected_count);
    if (kv_int8) hipLaunchKernelGGL(attention_score_int8_kernel, dim3(static_cast<std::uint32_t>(score_blocks64)), dim3(kThreads), 0,
        hip_execution_stream(), workspace.q_gate, reinterpret_cast<const std::uint16_t*>(kv_state.address), workspace.selected_indices,
        selected_count, checked_u32(workspace_layout.maximum_selected_tokens, "maximum_selected_tokens"), attention_heads, kv_heads,
        head_dim, workspace.attention_scores, dynamic_selection);
    else if (!launch_decode_r1_scores_v2(workspace.q_gate, reinterpret_cast<const std::uint16_t*>(kv_state.address),
            workspace.selected_indices, workspace.attention_scores, selected_count,
            checked_u32(workspace_layout.maximum_selected_tokens, "maximum_selected_tokens"), attention_heads, kv_heads, head_dim,
            dynamic_selection, device_graph, hip_execution_stream())) hipLaunchKernelGGL(
        attention_score_kernel,
        dim3(
            static_cast<std::uint32_t>(
                score_blocks64)),
        dim3(kThreads),
        0, hip_execution_stream(),
        workspace.q_gate,
        reinterpret_cast<
            const std::uint16_t*>(
                kv_state.address),
        workspace.selected_indices,
        selected_count,
        checked_u32(
            workspace_layout.
                maximum_selected_tokens,
            "maximum_selected_tokens"),
        attention_heads,
        kv_heads,
        head_dim,
        workspace.attention_scores, dynamic_selection);
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention score launch");

    hipLaunchKernelGGL(
        attention_softmax_kernel,
        dim3(attention_heads),
        dim3(kThreads),
        0, hip_execution_stream(),
        workspace.attention_scores,
        selected_count,
        checked_u32(
            workspace_layout.
                maximum_selected_tokens,
            "maximum_selected_tokens"), dynamic_selection);
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention softmax launch");

    if (kv_int8) hipLaunchKernelGGL(attention_output_int8_kernel, dim3(attention_heads), dim3(head_dim), 0, hip_execution_stream(),
        workspace.q_gate, reinterpret_cast<const std::uint16_t*>(kv_state.address), workspace.selected_indices,
        workspace.attention_scores, selected_count, checked_u32(workspace_layout.maximum_selected_tokens, "maximum_selected_tokens"),
        attention_heads, kv_heads, head_dim, workspace.attention_output, dynamic_selection);
    else if (!launch_decode_r1_value_v2(workspace.q_gate, reinterpret_cast<const std::uint16_t*>(kv_state.address),
            workspace.selected_indices, workspace.attention_scores, selected_count,
            checked_u32(workspace_layout.maximum_selected_tokens, "maximum_selected_tokens"), attention_heads, kv_heads, head_dim,
            workspace.attention_output, dynamic_selection, device_graph, hip_execution_stream())) hipLaunchKernelGGL(
        attention_output_kernel,
        dim3(attention_heads),
        dim3(head_dim),
        0, hip_execution_stream(),
        workspace.q_gate,
        reinterpret_cast<
            const std::uint16_t*>(
                kv_state.address),
        workspace.selected_indices,
        workspace.attention_scores,
        selected_count,
        checked_u32(
            workspace_layout.
                maximum_selected_tokens,
            "maximum_selected_tokens"),
        attention_heads,
        kv_heads,
        head_dim,
        workspace.attention_output, dynamic_selection);
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention value/gate launch");

    }

    if (!keep(tensors.out))
        launch_qsa_dense_a8_output(tensors.out, workspace.attention_output, output, attention_width, hidden, hip_execution_stream());
    else launch_linear_float_to_bf16(
        tensors.out,
        workspace.attention_output,
        attention_width,
        hidden,
        output, config.wave8_q8_output);
    if (device_graph) {
        hipLaunchKernelGGL(validate_indexed_graph_control, dim3(1), dim3(kThreads), 0,
            hip_execution_stream(), dynamic_position,
            reinterpret_cast<const std::uint16_t*>(output.address), hidden);
        check_hip(hipGetLastError(), "HIP indexed-attention finite output validation");
    }
}

// ---- Known-prefix window rows (T = 2/4/8 consecutive positions of one step) ----
// Each row keeps the C1 kernel's exact per-row arithmetic and summation order;
// weights (and pooled indexer block keys) are read once for all rows.
// One launch per serial load organization (prefetch8 Q8, or legacy256 per format).
template <unsigned Rows> struct QsaRowsInputViews {
    const std::uint16_t* input[Rows];
    const void* weights[5];
    std::uint32_t first[6]; // output-row prefix of the group's matrices
    float* output[5][Rows];
    std::uint32_t count;
};
template <unsigned Rows>
__device__ __forceinline__ unsigned qsa_rows_matrix(const QsaRowsInputViews<Rows>& views, std::uint32_t block, std::uint32_t& row) {
    unsigned m = 0;
    while (m + 1U < views.count && block >= views.first[m + 1U]) ++m;
    row = block - views.first[m];
    return m;
}

// q8_rows sites of the C1 prefetch8 input and wave8 output kernels: the same
// C++ coefficient/FMA expressions and upper-src0 tree additions, per row.
__device__ __forceinline__ float qsa_rows_tree_add(float lower, float upper) {
    float result;
    asm("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(upper), "v"(lower));
    return result;
}
__device__ __forceinline__ float qsa_rows_coefficient(unsigned scale, std::int8_t code) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(scale))) * static_cast<float>(code);
}
// index_q, index_k, q, k, v (five segments) from BF16 rows to BF16-rounded FP32.
template <unsigned Rows> struct QsaInputRowsSite {
    static constexpr bool dense_trellis_inputs = true;
    using Input = std::uint16_t;
    static constexpr unsigned rows = Rows;
    static constexpr bool fused_fma = true;  // Site::fma compiles to v_fmac_f32 (wide VOPD rows)
    static constexpr bool tree_upper_src0 = true;  // add(lower, upper) reads upper as src0 (lane VOPD tree)
    const void* input[Rows];
    q8_rows::Segments segments;
    std::uint32_t width, outputs;
    float* output[5][Rows];
    __device__ static float coefficient(unsigned scale, std::int8_t code) { return qsa_rows_coefficient(scale, code); }
    __device__ static void fma(float& acc, float c, float x) { acc += x * c; }
    __device__ static float add(float lower, float upper) { return qsa_rows_tree_add(lower, upper); }
    __device__ void store(std::uint32_t out, unsigned r, float value) const {
        unsigned m; std::uint32_t row;
        static_cast<void>(q8_rows::segment_row(segments, out, width, m, row));
        output[m][r][row] = round_bf16(value);
    }
};
template <unsigned Rows, class InputType = float> struct QsaOutputRowsSite {
    using Input = InputType;
    static constexpr unsigned rows = Rows;
    static constexpr bool fused_fma = true;  // Site::fma compiles to v_fmac_f32 (wide VOPD rows)
    static constexpr bool tree_upper_src0 = true;  // add(lower, upper) reads upper as src0 (lane VOPD tree)
    const void* input[Rows];
    q8_rows::Segments segments;
    std::uint32_t width, outputs;
    std::uint16_t* output[Rows];
    __device__ static float coefficient(unsigned scale, std::int8_t code) { return qsa_rows_coefficient(scale, code); }
    __device__ static void fma(float& acc, float c, float x) { acc += x * c; }
    __device__ static float add(float lower, float upper) { return qsa_rows_tree_add(lower, upper); }
    __device__ void store(std::uint32_t out, unsigned r, float value) const { output[r][out] = float_to_bf16(value); }
};
// STRIX_QPF2K_DENSE=1 (q8_rows::tiled): the Q8_0 input / output projections of all uniformly strided prefill rows in one
// launch; the same store arithmetic as QsaInputRowsSite / QsaOutputRowsSite.
struct QsaInputTiledSite {
    using Input = std::uint16_t;
    const unsigned char* input0;
    std::uint64_t input_stride;
    q8_rows::Segments segments;
    std::uint32_t width, outputs;
    unsigned char* output0[5];
    std::uint64_t output_stride;
    __device__ void store(std::uint32_t out, std::uint32_t r, float value) const {
        unsigned m; std::uint32_t row;
        static_cast<void>(q8_rows::segment_row(segments, out, width, m, row));
        reinterpret_cast<float*>(output0[m] + r * output_stride)[row] = round_bf16(value);
    }
};
template <class InputType = float> struct QsaOutputTiledSite {
    using Input = InputType;
    const unsigned char* input0;
    std::uint64_t input_stride;
    q8_rows::Segments segments;
    std::uint32_t width, outputs;
    unsigned char* output0;
    std::uint64_t output_stride;
    __device__ void store(std::uint32_t out, std::uint32_t r, float value) const {
        reinterpret_cast<std::uint16_t*>(output0 + r * output_stride)[out] = float_to_bf16(value);
    }
};

void launch_qsa_dense_a8_inputs(const std::array<const TensorPair*, 5>& matrices, const std::array<std::uint32_t, 5>& outputs,
    const std::array<float*, 5>& targets, const ExecutionBufferAddress& input, std::uint32_t hidden, hipStream_t stream, unsigned skip) {
    QsaInputRowsSite<1U> site{};
    for (unsigned i = 0; i < 5U; ++i) {
        if (!q8_rows::dense_a8_type(matrices[i]->manifest->ggml_type) || ((skip >> i) & 1U)) continue;
        const auto m = site.segments.count++;
        site.segments.types[m] = q8_rows::segment_type(matrices[i]->manifest->ggml_type);
        site.segments.weights[m] = reinterpret_cast<const unsigned char*>(matrices[i]->runtime->address);
        site.segments.first[m + 1U] = site.segments.first[m] + outputs[i];
        site.output[m][0] = targets[i];
    }
    if (!site.segments.count) return;
    site.input[0] = reinterpret_cast<const void*>(input.address);
    site.width = hidden; site.outputs = site.segments.first[site.segments.count];
    q8_rows::launch(site, true, stream);
    check_hip(hipGetLastError(), "HIP indexed-attention dense A8 input projection launch");
}
void launch_qsa_dense_a8_output(const TensorPair& weights, const float* input, const ExecutionBufferAddress& output,
    std::uint32_t width, std::uint32_t hidden, hipStream_t stream) {
    QsaOutputRowsSite<1U> site{};
    site.segments = q8_rows::segments({{reinterpret_cast<const void*>(weights.runtime->address), hidden}});
    site.segments.types[0] = q8_rows::segment_type(weights.manifest->ggml_type);
    site.input[0] = input;
    site.output[0] = reinterpret_cast<std::uint16_t*>(output.address);
    site.width = width; site.outputs = hidden;
    q8_rows::launch(site, true, stream);
    check_hip(hipGetLastError(), "HIP indexed-attention dense A8 output projection launch");
}
bool q6_r1_registered(const void* weights, std::uint32_t width, std::uint32_t outputs) {
    const auto* q6 = q6p::find(weights);
    return q6 && q6->kind == q6p::Kind::q6p32 && q6->rows == outputs && q6->cols == width;
}
bool qsa_q6_r1_input(const std::uint16_t* input, const void* weights, std::uint32_t width, std::uint32_t outputs,
    float* output, hipStream_t stream) {
    if (!q6_r1_registered(weights, width, outputs)) return false;
    QsaInputRowsSite<1> site{};
    site.segments = q8_rows::segments({{weights, outputs}});
    site.width = width; site.outputs = outputs; site.input[0] = input; site.output[0][0] = output;
    q8_rows::launch(site, false, stream);
    check_hip(hipGetLastError(), "HIP indexed-attention Q6P32 input projection launch");
    return true;
}
bool qsa_q6_r1_output(const float* input, const void* weights, std::uint32_t width, std::uint32_t outputs,
    std::uint16_t* output, hipStream_t stream) {
    if (!q6_r1_registered(weights, width, outputs)) return false;
    QsaOutputRowsSite<1> site{};
    site.segments = q8_rows::segments({{weights, outputs}});
    site.width = width; site.outputs = outputs; site.input[0] = input; site.output[0] = output;
    q8_rows::launch(site, false, stream);
    check_hip(hipGetLastError(), "HIP indexed-attention Q6P32 output projection launch");
    return true;
}

template <WeightKind Kind, unsigned Rows>
__global__ void linear_to_float_rows_kernel(QsaRowsInputViews<Rows> views, std::uint32_t input_width) {
    if (blockIdx.x >= views.first[views.count]) return;
    std::uint32_t row = 0;
    const auto m = qsa_rows_matrix(views, blockIdx.x, row);
    __shared__ float scratch[Rows][kThreads];
    float sum[Rows]{};
    for (std::uint32_t feature = threadIdx.x; feature < input_width; feature += kThreads) {
        const float weight = load_weight<Kind>(views.weights[m], input_width, row, feature);
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) sum[r] += load_activation(&views.input[r][feature]) * weight;
    }
#pragma unroll
    for (unsigned r = 0; r < Rows; ++r) scratch[r][threadIdx.x] = sum[r];
    __syncthreads();
    for (std::uint32_t stride = kThreads / 2U; stride > 0; stride >>= 1U) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (unsigned r = 0; r < Rows; ++r) scratch[r][threadIdx.x] += scratch[r][threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) views.output[m][r][row] = round_bf16(scratch[r][0]);
    }
}

template <unsigned Rows> struct QsaRowsOutputViews {
    const float* input[Rows];
    std::uint16_t* output[Rows];
    bool compact = false;
};

template <WeightKind Kind, unsigned Rows>
__global__ void linear_float_to_bf16_rows_kernel(QsaRowsOutputViews<Rows> views, const void* weights,
    std::uint32_t input_width, std::uint32_t output_width) {
    const auto out = static_cast<std::uint32_t>(blockIdx.x);
    if (out >= output_width) return;
    __shared__ float scratch[Rows][kThreads];
    float sum[Rows]{};
    for (std::uint32_t feature = threadIdx.x; feature < input_width; feature += kThreads) {
        const float weight = load_weight<Kind>(weights, input_width, out, feature);
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) sum[r] += (views.compact ? bf16_to_float(reinterpret_cast<const std::uint16_t*>(views.input[r])[feature]) : views.input[r][feature]) * weight;
    }
#pragma unroll
    for (unsigned r = 0; r < Rows; ++r) scratch[r][threadIdx.x] = sum[r];
    __syncthreads();
    for (std::uint32_t stride = kThreads / 2U; stride > 0; stride >>= 1U) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (unsigned r = 0; r < Rows; ++r) scratch[r][threadIdx.x] += scratch[r][threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) views.output[r][out] = float_to_bf16(scratch[r][0]);
    }
}

template <unsigned Rows> struct QsaRowsNormViews {
    float* values[Rows];
    std::uint64_t position[Rows];
};
template <SmallWeightKind Kind, unsigned Rows, class... Source>
__global__ void norm_rope_rows_kernel(QsaRowsNormViews<Rows> views, std::uint32_t head_count,
    std::uint32_t head_dim, std::uint32_t head_stride, const void* norm_weights, float epsilon,
    std::uint32_t rotary_dim, float theta, Source... rotary_source) {
    norm_rope_body<Kind, Source...>(views.values[blockIdx.y], head_count, head_dim, head_stride, norm_weights,
        epsilon, rotary_dim, theta, views.position[blockIdx.y], GraphPositionKernelArguments{}, rotary_source...);
}

#include "hip_qsa_prefill_stages.inc"

// kv-int8-tok-v1 rows append: grid (kv_heads, 2, Rows), block (h, half, r) quantizes KV head
// h of row r's K (half 0) or V (half 1) with the decode writer's statements.
template <unsigned Rows>
__global__ void store_rows_kv_int8_kernel(QsaRowsStoreViews<Rows> views, void* kv_state, std::uint32_t kv_width,
    std::uint32_t kv_heads, std::uint32_t head_dim) {
    const auto r = blockIdx.z, half = static_cast<std::uint32_t>(blockIdx.y);
    kv_int8_store_head(half ? views.v[r] : views.k[r], kv_state, views.position[r], kv_width, kv_heads, half,
        static_cast<std::uint32_t>(blockIdx.x), head_dim);
}

// One pooled/normalised/rotated block key serves every row whose visible
// prefix contains that complete block; each row's head reduction is unchanged.
template <unsigned Rows> struct QsaRowsScoreViews {
    const float* index_query[Rows];
    float* block_scores[Rows];
    std::uint32_t blocks[Rows]; // 0 when the row selects all visible tokens
};
template <SmallWeightKind Kind, unsigned Rows, class... Source>
__global__ void block_score_rows_kernel(QsaRowsScoreViews<Rows> views, const std::uint16_t* raw_keys,
    std::uint32_t index_heads, std::uint32_t head_dim, std::uint32_t compress_ratio,
    const void* key_norm, float epsilon, std::uint32_t rotary_dim, float theta, Source... rotary_source) {
    const auto block = static_cast<std::uint32_t>(blockIdx.x);
    const auto feature = static_cast<std::uint32_t>(threadIdx.x);
    if (feature >= head_dim) return;
    extern __shared__ float shared[];
    auto* pooled = shared;
    auto* scratch = shared + head_dim;
    float mean = 0.0F;
    const auto token_start = static_cast<std::uint64_t>(block) * compress_ratio;
    if (!(rotary_source.check(token_start) && ...)) return;
    for (std::uint32_t offset = 0; offset < compress_ratio; ++offset)
        mean += bf16_to_float(raw_keys[(token_start + offset) * head_dim + feature]);
    mean /= static_cast<float>(compress_ratio);
    mean = round_bf16(mean);
    pooled[feature] = mean;
    scratch[feature] = mean * mean;
    __syncthreads();
    for (std::uint32_t stride = head_dim / 2U; stride > 0; stride >>= 1U) {
        if (feature < stride) scratch[feature] += scratch[feature + stride];
        __syncthreads();
    }
    const float inv_rms = rsqrtf(scratch[0] / static_cast<float>(head_dim) + epsilon);
    pooled[feature] = round_bf16(mean * inv_rms * load_small_weight<Kind>(key_norm, feature));
    __syncthreads();
    float rotated = pooled[feature];
    if constexpr (sizeof...(Source) == 0) {
        if (feature < rotary_dim) {
            const auto half = rotary_dim / 2U;
            const auto pair = feature % half;
            const float exponent = 2.0F * static_cast<float>(pair) / static_cast<float>(rotary_dim);
            const float angle = static_cast<float>(token_start) * powf(theta, -exponent);
            const float cosine = cosf(angle);
            const float sine = sinf(angle);
            if (feature < half) rotated = pooled[feature] * cosine - pooled[feature + half] * sine;
            else rotated = pooled[feature] * cosine + pooled[feature - half] * sine;
        }
    } else {
        if (feature < rotary_dim) {
            const auto half = rotary_dim / 2U;
            const auto pair = feature % half;
            const float2 cs = (rotary_source.cs[token_start * 32ULL + pair], ...);
            const float cosine = cs.x;
            const float sine = cs.y;
            if (feature < half) rotated = pooled[feature] * cosine - pooled[feature + half] * sine;
            else rotated = pooled[feature] * cosine + pooled[feature - half] * sine;
        }
    }
    __syncthreads();
    pooled[feature] = round_bf16(rotated);
    __syncthreads();
    __shared__ float total;
    constexpr unsigned Span = Rows <= 8U ? Rows : 8U;
    const unsigned base = Rows <= 8U ? 0U : static_cast<unsigned>(blockIdx.y) * 8U;
    for (unsigned rr = 0; rr < Span; ++rr) {
        const unsigned r = base + rr;
        if (block >= views.blocks[r]) continue; // uniform per block
        if (feature == 0) total = 0.0F;
        __syncthreads();
        for (std::uint32_t head = 0; head < index_heads; ++head) {
            scratch[feature] = views.index_query[r][static_cast<std::uint64_t>(head) * head_dim + feature] * pooled[feature];
            __syncthreads();
            for (std::uint32_t stride = head_dim / 2U; stride > 0; stride >>= 1U) {
                if (feature < stride) scratch[feature] += scratch[feature + stride];
                __syncthreads();
            }
            if (feature == 0) total += fmaxf(scratch[0], 0.0F);
            __syncthreads();
        }
        if (feature == 0) views.block_scores[r][block] = total / sqrtf(static_cast<float>(head_dim));
    }
}

template <unsigned Rows> struct QsaRowsSelectViews {
    float* block_scores[Rows];
    std::uint32_t* selected[Rows];
    std::uint32_t visible[Rows];
};
template <unsigned Rows>
__global__ __launch_bounds__(kSelectThreads) void select_rows_kernel(QsaRowsSelectViews<Rows> views, std::uint32_t compress_ratio, std::uint32_t block_topk) {
    const auto r = blockIdx.x;
    const auto visible = views.visible[r];
    const auto complete = visible / compress_ratio;
    if (complete <= block_topk) {
        for (std::uint32_t index = threadIdx.x; index < visible; index += blockDim.x) views.selected[r][index] = index;
        return;
    }
    select_blocks_parallel_body(views.block_scores[r], complete, block_topk, compress_ratio, visible,
        views.selected[r], GraphSelectionArguments{});
}

template <unsigned Rows> struct QsaRowsAttentionViews {
    const float* q_gate[Rows];
    const std::uint32_t* selected[Rows];
    float* scores[Rows];
    float* output[Rows];
    std::uint32_t count[Rows];
    bool compact_output = false;
    __device__ __forceinline__ RoundedStageImage image(unsigned r) const { return {output[r], compact_output}; }
};
template <unsigned Rows, bool Int8 = false>
__global__ void attention_score_rows_kernel(QsaRowsAttentionViews<Rows> views, const std::uint16_t* kv_state,
    std::uint32_t maximum_selected, std::uint32_t attention_heads, std::uint32_t kv_heads, std::uint32_t head_dim) {
    const auto r = blockIdx.y;
    attention_score_body<Int8>(views.q_gate[r], kv_state, views.selected[r], views.count[r], maximum_selected,
        attention_heads, kv_heads, head_dim, views.scores[r], GraphSelectionArguments{});
}
// One wave per (row, head, slot) instead of one 256-thread block: lane l holds
// features l + 32j, so strides 128/64/32 of the block tree are lane-local adds
// and 16..1 are shuffles. Same products, same pairings, same order as the LDS
// tree in attention_score_body, without contraction, so the scores match it.
// PerWave > 1: a wave runs PerWave consecutive slots of one (row, head) with q held in
// registers; every slot keeps the exact single-slot products and tree.
template <unsigned Rows, unsigned PerWave = 1U, bool Int8 = false>
__global__ __launch_bounds__(256) void attention_score_rows_wave_kernel(QsaRowsAttentionViews<Rows> views,
    const std::uint16_t* kv_state, std::uint32_t maximum_selected, std::uint32_t attention_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t slots) {
#pragma clang fp contract(off)
    static_assert(kThreads == 256U);
    const auto lane = threadIdx.x % 32U;
    const auto wave = blockIdx.x * 8U + threadIdx.x / 32U;
    const auto groups = (slots + PerWave - 1U) / PerWave;
    const auto r = wave / (attention_heads * groups);
    if (r >= Rows) return;
    const auto head = wave / groups % attention_heads, first = wave % groups * PerWave;
    if (first >= views.count[r]) return;
    const auto kv_head = head / (attention_heads / kv_heads);
    const auto* q = views.q_gate[r] + static_cast<std::uint64_t>(head) * head_dim * 2ULL;
    float qv[8];
#pragma unroll
    for (unsigned j = 0; j < 8U; ++j) { const auto feature = lane + 32U * j; qv[j] = feature < head_dim ? q[feature] : 0.0F; }
    float p[PerWave][8];
#pragma unroll
    for (unsigned i = 0; i < PerWave; ++i) {
        const auto slot = first + i < views.count[r] ? first + i : first;
        const auto token = views.selected[r][slot];
        const auto* k = kv_state + static_cast<std::uint64_t>(token) * kv_heads * head_dim * 2ULL +
            static_cast<std::uint64_t>(kv_head) * head_dim;
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) {
            const auto feature = lane + 32U * j;
            float product = 0.0F;
            if constexpr (Int8) {
                if (feature < head_dim)
                    product = qv[j] * kv_int8_load(kv_state, token, kv_heads * head_dim, kv_heads, 0U, kv_head, head_dim, feature);
            } else if (feature < head_dim) product = qv[j] * bf16_to_float(k[feature]);
            p[i][j] = product;
        }
    }
#pragma unroll
    for (unsigned i = 0; i < PerWave; ++i) {
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) p[i][j] += p[i][j + 4U];
        p[i][0] += p[i][2]; p[i][1] += p[i][3];
        float sum = p[i][0] + p[i][1];
#pragma unroll
        for (unsigned stride = 16U; stride > 0U; stride >>= 1U) sum += __shfl_down(sum, stride, 32);
        if (lane == 0U && first + i < views.count[r])
            views.scores[r][static_cast<std::uint64_t>(head) * maximum_selected + first + i] = sum / sqrtf(static_cast<float>(head_dim));
    }
}
template <unsigned Rows>
__global__ void attention_softmax_rows_kernel(QsaRowsAttentionViews<Rows> views, std::uint32_t maximum_selected) {
    const auto r = blockIdx.y;
    attention_softmax_body(views.scores[r], views.count[r], maximum_selected, GraphSelectionArguments{});
}
template <unsigned Rows, bool Int8 = false>
__global__ void attention_output_rows_kernel(QsaRowsAttentionViews<Rows> views, const std::uint16_t* kv_state,
    std::uint32_t maximum_selected, std::uint32_t attention_heads, std::uint32_t kv_heads, std::uint32_t head_dim) {
    const auto r = blockIdx.y;
    attention_output_body<Int8>(views.q_gate[r], kv_state, views.selected[r], views.scores[r], views.count[r],
        maximum_selected, attention_heads, kv_heads, head_dim, views.image(r), GraphSelectionArguments{});
}
template <unsigned Rows, bool Int8 = false, bool Dense = false>
__global__ __launch_bounds__(kThreads) void attention_tail_rows_kernel(QsaRowsAttentionViews<Rows> views,
    const std::uint16_t* kv_state, std::uint32_t maximum_selected, std::uint32_t attention_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim, std::int8_t* codes, float* deltas) {
    const auto r = static_cast<std::uint32_t>(blockIdx.y);
    attention_tail_body<Int8, Dense>(views.q_gate[r], kv_state, views.selected[r], views.scores[r],
        views.count[r], maximum_selected, attention_heads, kv_heads, head_dim, views.image(r),
        GraphSelectionArguments{}, codes, deltas, r);
}
// Dense A8 (block = one head, thread = feature, every lane present): attention_output_rows_kernel
// whose waves also quantize the 32 features they just wrote for the o projection (separate
// kernel so the default one compiles as before).
template <unsigned Rows, bool Int8 = false>
__global__ void attention_output_rows_dense_a8_kernel(QsaRowsAttentionViews<Rows> views, const std::uint16_t* kv_state,
    std::uint32_t maximum_selected, std::uint32_t attention_heads, std::uint32_t kv_heads, std::uint32_t head_dim,
    std::int8_t* a8_codes, float* a8_deltas) {
    const auto r = blockIdx.y;
    attention_output_body<Int8>(views.q_gate[r], kv_state, views.selected[r], views.scores[r], views.count[r],
        maximum_selected, attention_heads, kv_heads, head_dim, views.image(r), GraphSelectionArguments{});
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * head_dim + threadIdx.x;
    q8_rows::dense_a8_quantize_lane(views.image(r)[index], a8_codes + r * static_cast<std::uint64_t>(attention_heads) * head_dim + index,
        a8_deltas + (index / 32U) * q8_rows::kDenseA8Rows + r);
}


// Prefill groups (Rows > 8): the G query heads that share a KV head read the same K / V
// rows, so one wave (score) or one block (output) serves G heads of one row and loads each
// K / V element once for all of them. Per (row, head, slot) the arithmetic is that of
// attention_score_rows_wave_kernel / attention_output_body: the same products, the same
// lane-local pairs and the same 16..1 lane tree (lane 0's partner at each step holds the
// same partial sum under xor as under shfl_down, and one addition commutes), the same
// division; the value chain is the same ascending-slot fma chain the window kernel's
// contracted sum += score * value compiles to.
#include "hip_qsa_scatter.inc"
template <unsigned Rows, unsigned G, unsigned Slots, bool Scatter, bool Int8 = false>
__global__ __launch_bounds__(256) void attention_score_rows_group_kernel(QsaRowsAttentionViews<Rows> views,
    const std::uint16_t* kv_state, std::uint32_t maximum_selected, std::uint32_t attention_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t slots) {
#pragma clang fp contract(off)
    const auto lane = threadIdx.x % 32U;
    const auto wave = blockIdx.x * 8U + threadIdx.x / 32U;
    const auto groups = (slots + Slots - 1U) / Slots, head_groups = attention_heads / G;
    const auto r = wave / (head_groups * groups);
    if (r >= Rows) return;
    const auto head0 = wave / groups % head_groups * G, first = wave % groups * Slots;
    if (first >= views.count[r]) return;
    const auto kv_head = head0 / (attention_heads / kv_heads);
    float qv[G][8];
#pragma unroll
    for (unsigned h = 0; h < G; ++h) {
        const auto* q = views.q_gate[r] + static_cast<std::uint64_t>(head0 + h) * head_dim * 2ULL;
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) { const auto feature = lane + 32U * j; qv[h][j] = feature < head_dim ? q[feature] : 0.0F; }
    }
    const float scale = sqrtf(static_cast<float>(head_dim));
    const auto last = min(first + Slots, views.count[r]);
    for (auto slot = first; slot < last; ++slot) {
        const auto token = views.selected[r][slot];
        const auto* k = kv_state + static_cast<std::uint64_t>(token) * kv_heads * head_dim * 2ULL + static_cast<std::uint64_t>(kv_head) * head_dim;
        float kf[8];
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) {
            const auto feature = lane + 32U * j;
            if constexpr (Int8)
                kf[j] = feature < head_dim ? kv_int8_load(kv_state, token, kv_heads * head_dim, kv_heads, 0U, kv_head, head_dim, feature) : 0.0F;
            else kf[j] = feature < head_dim ? bf16_to_float(k[feature]) : 0.0F;
        }
        constexpr unsigned P = G <= 2U ? 2U : G <= 4U ? 4U : G <= 8U ? 8U : 16U;
        float partial[P];
#pragma unroll
        for (unsigned h = 0; h < P; ++h) {
            if (h >= G) { partial[h] = 0.0F; continue; }
            float p[8];
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) p[j] = lane + 32U * j < head_dim ? qv[h][j] * kf[j] : 0.0F;
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) p[j] += p[j + 4U];
            p[0] += p[2]; p[1] += p[3];
            partial[h] = p[0] + p[1];
        }
        if constexpr (Scatter) {
            qsa_scatter<P>(partial, lane);
            const auto h = qsa_scatter_head<P>(lane);
            if (h < G && (lane & ((32U / P) - 1U)) == 0U)
                views.scores[r][static_cast<std::uint64_t>(head0 + h) * maximum_selected + slot] = partial[0] / scale;
        } else {
#pragma unroll
            for (unsigned h = 0; h < G; ++h) {
                float sum = partial[h];
                sum += qsa_lane_xor<16U>(sum); sum += qsa_lane_xor<8U>(sum); sum += qsa_lane_xor<4U>(sum);
                sum += qsa_lane_xor<2U>(sum); sum += qsa_lane_xor<1U>(sum);
                if (lane == 0U) views.scores[r][static_cast<std::uint64_t>(head0 + h) * maximum_selected + slot] = sum / scale;
            }
        }
    }
}
// head_dim 256 form of attention_score_rows_group_kernel<Rows, G, Slots, true>: every feature
// guard is true, the wave's Slots token indices are loaded once (lane i holds slot first + i),
// and the next slot's K row is requested before the current slot's products. Per slot the
// statements are those of the generic kernel.
template <unsigned Rows, unsigned G, unsigned Slots, bool Int8 = false>
__global__ __launch_bounds__(256) void attention_score_rows_group256_kernel(QsaRowsAttentionViews<Rows> views,
    const std::uint16_t* kv_state, std::uint32_t maximum_selected, std::uint32_t attention_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t slots) {
#pragma clang fp contract(off)
    static_assert(Slots <= 32U);
    constexpr unsigned Dim = 256U; // head_dim == Dim (checked by the caller)
    const auto lane = threadIdx.x % 32U;
    const auto wave = blockIdx.x * 8U + threadIdx.x / 32U;
    const auto groups = (slots + Slots - 1U) / Slots, head_groups = attention_heads / G;
    const auto r = wave / (head_groups * groups);
    if (r >= Rows) return;
    const auto head0 = wave / groups % head_groups * G, first = wave % groups * Slots;
    if (first >= views.count[r]) return;
    const auto kv_head = head0 / (attention_heads / kv_heads);
    float qv[G][8];
#pragma unroll
    for (unsigned h = 0; h < G; ++h) {
        const auto* q = views.q_gate[r] + static_cast<std::uint64_t>(head0 + h) * Dim * 2ULL;
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) qv[h][j] = q[lane + 32U * j];
    }
    const float scale = sqrtf(static_cast<float>(head_dim));
    const auto last = min(first + Slots, views.count[r]);
    const auto tokens = lane < last - first ? views.selected[r][first + lane] : 0U;
    float* const scores = views.scores[r]; // loaded once: a store could alias the argument block
    const auto* kbase = kv_state + static_cast<std::uint64_t>(kv_head) * Dim + lane;
    const auto row_elements = static_cast<std::uint64_t>(kv_heads) * Dim * 2ULL;
    // Raw BF16 bits travel to the next slot; they become floats where the products use them
    // (int8 KV: the dequantized values travel instead).
    using Raw = std::conditional_t<Int8, float, std::uint16_t>;
    Raw raw[8];
    const auto load = [&](std::uint32_t index, Raw (&k)[8]) {
        const auto token = static_cast<std::uint32_t>(__builtin_amdgcn_readlane(static_cast<int>(tokens), static_cast<int>(index)));
        if constexpr (Int8) {
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) k[j] = kv_int8_load(kv_state, token, kv_heads * Dim, kv_heads, 0U, kv_head, Dim, lane + 32U * j);
        } else {
            const auto* row = kbase + static_cast<std::uint64_t>(token) * row_elements;
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) k[j] = row[32U * j];
        }
    };
    load(0U, raw);
    constexpr unsigned P = G <= 2U ? 2U : G <= 4U ? 4U : G <= 8U ? 8U : 16U;
    for (auto slot = first; slot < last; ++slot) {
        float kf[8];
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) {
            if constexpr (Int8) kf[j] = raw[j];
            else kf[j] = bf16_to_float(raw[j]);
        }
        if (slot + 1U < last) load(slot + 1U - first, raw);
        float partial[P];
#pragma unroll
        for (unsigned h = 0; h < P; ++h) {
            if (h >= G) { partial[h] = 0.0F; continue; }
            float p[8];
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) p[j] = qv[h][j] * kf[j];
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) p[j] += p[j + 4U];
            p[0] += p[2]; p[1] += p[3];
            partial[h] = p[0] + p[1];
        }
        qsa_scatter<P>(partial, lane);
        const auto h = qsa_scatter_head<P>(lane);
        if (h < G && (lane & ((32U / P) - 1U)) == 0U)
            scores[static_cast<std::uint64_t>(head0 + h) * maximum_selected + slot] = partial[0] / scale;
    }
}
template <unsigned Rows, unsigned G, bool Int8 = false>
__global__ __launch_bounds__(256) void attention_output_rows_group_kernel(QsaRowsAttentionViews<Rows> views,
    const std::uint16_t* kv_state, std::uint32_t maximum_selected, std::uint32_t attention_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim) {
    const auto head0 = static_cast<std::uint32_t>(blockIdx.x) * G, r = static_cast<std::uint32_t>(blockIdx.y);
    const auto feature = static_cast<std::uint32_t>(threadIdx.x);
    if (head0 >= attention_heads || feature >= head_dim) return;
    const auto count = views.count[r];
    const auto kv_head = head0 / (attention_heads / kv_heads);
    const auto kv_width = kv_heads * head_dim;
    const auto offset = static_cast<std::uint64_t>(kv_width) + static_cast<std::uint64_t>(kv_head) * head_dim + feature;
    const auto* selected = views.selected[r];
    const float* scores = views.scores[r] + static_cast<std::uint64_t>(head0) * maximum_selected;
    const auto value_at = [&](std::uint32_t token) {
        if constexpr (Int8) return kv_int8_load(kv_state, token, kv_width, kv_heads, 1U, kv_head, head_dim, feature);
        else return bf16_to_float(kv_state[static_cast<std::uint64_t>(token) * kv_width * 2ULL + offset]);
    };
    float sum[G];
#pragma unroll
    for (unsigned h = 0; h < G; ++h) sum[h] = 0.0F;
    std::uint32_t slot = 0;
    for (; slot + 4U <= count; slot += 4U) {
        float value[4];
#pragma unroll
        for (unsigned i = 0; i < 4U; ++i)
            value[i] = value_at(selected[slot + i]);
#pragma unroll
        for (unsigned i = 0; i < 4U; ++i)
#pragma unroll
            for (unsigned h = 0; h < G; ++h)
                sum[h] = __builtin_fmaf(scores[static_cast<std::uint64_t>(h) * maximum_selected + slot + i], value[i], sum[h]);
    }
    for (; slot < count; ++slot) {
        const float value = value_at(selected[slot]);
#pragma unroll
        for (unsigned h = 0; h < G; ++h) sum[h] = __builtin_fmaf(scores[static_cast<std::uint64_t>(h) * maximum_selected + slot], value, sum[h]);
    }
#pragma unroll
    for (unsigned h = 0; h < G; ++h) {
        const auto head = head0 + h;
        const float attn = round_bf16(sum[h]);
        const float gate = round_bf16(1.0F / (1.0F + expf(-views.q_gate[r][static_cast<std::uint64_t>(head) * head_dim * 2ULL + head_dim + feature])));
        views.image(r)[static_cast<std::uint64_t>(head) * head_dim + feature] = round_bf16(attn * gate);
    }
}

// attention_output_rows_group_kernel with the value loads of the next 8 slots requested before
// the current 8 slots join the chains. Each head's chain is the same ascending-slot fma
// sequence (its blocking into 4 or 8 slots does not enter the arithmetic), with the same
// BF16 roundings and gate.
template <unsigned Rows, unsigned G, bool Int8 = false>
__global__ __launch_bounds__(256) void attention_output_rows_group_prefetch_kernel(QsaRowsAttentionViews<Rows> views,
    const std::uint16_t* kv_state, std::uint32_t maximum_selected, std::uint32_t attention_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim) {
    const auto head0 = static_cast<std::uint32_t>(blockIdx.x) * G, r = static_cast<std::uint32_t>(blockIdx.y);
    const auto feature = static_cast<std::uint32_t>(threadIdx.x);
    if (head0 >= attention_heads || feature >= head_dim) return;
    const auto count = views.count[r];
    const auto kv_head = head0 / (attention_heads / kv_heads);
    const auto kv_width = kv_heads * head_dim;
    const auto offset = static_cast<std::uint64_t>(kv_width) + static_cast<std::uint64_t>(kv_head) * head_dim + feature;
    const auto* selected = views.selected[r];
    const float* scores = views.scores[r] + static_cast<std::uint64_t>(head0) * maximum_selected;
    const auto* values = kv_state + offset;
    const auto row_elements = static_cast<std::uint64_t>(kv_width) * 2ULL;
    float sum[G];
#pragma unroll
    for (unsigned h = 0; h < G; ++h) sum[h] = 0.0F;
    constexpr unsigned B = 8U;
    const auto full = count / B * B;
    using Raw = std::conditional_t<Int8, float, std::uint16_t>;
    const auto raw_at = [&](std::uint32_t token) -> Raw {
        if constexpr (Int8) return kv_int8_load(kv_state, token, kv_width, kv_heads, 1U, kv_head, head_dim, feature);
        else return values[static_cast<std::uint64_t>(token) * row_elements];
    };
    const auto to_float = [](Raw bits) -> float {
        if constexpr (Int8) return bits;
        else return bf16_to_float(bits);
    };
    Raw raw[B];
    if (full) {
#pragma unroll
        for (unsigned i = 0; i < B; ++i) raw[i] = raw_at(selected[i]);
    }
    for (std::uint32_t slot = 0; slot < full; slot += B) {
        float value[B];
#pragma unroll
        for (unsigned i = 0; i < B; ++i) value[i] = to_float(raw[i]);
        // Unconditional (the last block reloads itself), so the loads stay ahead of this block's chains.
        const auto next = slot + B < full ? slot + B : slot;
#pragma unroll
        for (unsigned i = 0; i < B; ++i) raw[i] = raw_at(selected[next + i]);
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (unsigned i = 0; i < B; ++i)
#pragma unroll
            for (unsigned h = 0; h < G; ++h)
                sum[h] = __builtin_fmaf(scores[static_cast<std::uint64_t>(h) * maximum_selected + slot + i], value[i], sum[h]);
    }
    for (auto slot = full; slot < count; ++slot) {
        const float value = to_float(raw_at(selected[slot]));
#pragma unroll
        for (unsigned h = 0; h < G; ++h) sum[h] = __builtin_fmaf(scores[static_cast<std::uint64_t>(h) * maximum_selected + slot], value, sum[h]);
    }
#pragma unroll
    for (unsigned h = 0; h < G; ++h) {
        const auto head = head0 + h;
        const float attn = round_bf16(sum[h]);
        const float gate = round_bf16(1.0F / (1.0F + expf(-views.q_gate[r][static_cast<std::uint64_t>(head) * head_dim * 2ULL + head_dim + feature])));
        views.image(r)[static_cast<std::uint64_t>(head) * head_dim + feature] = round_bf16(attn * gate);
    }
}

// Verify windows (Rows <= 8, STRIX_QSA_WINDOW_OUT_GROUP=2|3|4|6|12): attention_output_rows_group_prefetch_kernel
// (one block per G query heads sharing a KV head, so each value row is loaded once for the G heads) with,
// when a8_codes is set, attention_output_rows_dense_a8_kernel's epilogue run for each of the G heads. Per
// (row, head, feature) the chain is attention_output_body's: the same `sum += score * value` expression per slot in
// ascending slot order (so it takes this file's contraction exactly as the body does), the same BF16 roundings and
// gate; the quantizer is wave-local (32 features of one head).
template <unsigned Rows, unsigned G, bool Int8 = false>
__global__ __launch_bounds__(256) void attention_output_rows_window_group_kernel(QsaRowsAttentionViews<Rows> views,
    const std::uint16_t* kv_state, std::uint32_t maximum_selected, std::uint32_t attention_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::int8_t* a8_codes, float* a8_deltas) {
    const auto head0 = static_cast<std::uint32_t>(blockIdx.x) * G, r = static_cast<std::uint32_t>(blockIdx.y);
    const auto feature = static_cast<std::uint32_t>(threadIdx.x);
    if (head0 >= attention_heads || feature >= head_dim) return;
    const auto count = views.count[r];
    const auto kv_head = head0 / (attention_heads / kv_heads);
    const auto kv_width = kv_heads * head_dim;
    const auto offset = static_cast<std::uint64_t>(kv_width) + static_cast<std::uint64_t>(kv_head) * head_dim + feature;
    const auto* selected = views.selected[r];
    const float* scores = views.scores[r] + static_cast<std::uint64_t>(head0) * maximum_selected;
    const auto* values = kv_state + offset;
    const auto row_elements = static_cast<std::uint64_t>(kv_width) * 2ULL;
    float sum[G];
#pragma unroll
    for (unsigned h = 0; h < G; ++h) sum[h] = 0.0F;
    constexpr unsigned B = 8U;
    const auto full = count / B * B;
    using Raw = std::conditional_t<Int8, float, std::uint16_t>;
    const auto raw_at = [&](std::uint32_t token) -> Raw {
        if constexpr (Int8) return kv_int8_load(kv_state, token, kv_width, kv_heads, 1U, kv_head, head_dim, feature);
        else return values[static_cast<std::uint64_t>(token) * row_elements];
    };
    const auto to_float = [](Raw bits) -> float {
        if constexpr (Int8) return bits;
        else return bf16_to_float(bits);
    };
    Raw raw[B];
    if (full) {
#pragma unroll
        for (unsigned i = 0; i < B; ++i) raw[i] = raw_at(selected[i]);
    }
    for (std::uint32_t slot = 0; slot < full; slot += B) {
        float value[B];
#pragma unroll
        for (unsigned i = 0; i < B; ++i) value[i] = to_float(raw[i]);
        const auto next = slot + B < full ? slot + B : slot;
#pragma unroll
        for (unsigned i = 0; i < B; ++i) raw[i] = raw_at(selected[next + i]);
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (unsigned i = 0; i < B; ++i)
#pragma unroll
            for (unsigned h = 0; h < G; ++h)
                sum[h] += scores[static_cast<std::uint64_t>(h) * maximum_selected + slot + i] * value[i];
    }
    for (auto slot = full; slot < count; ++slot) {
        const float value = to_float(raw_at(selected[slot]));
#pragma unroll
        for (unsigned h = 0; h < G; ++h) sum[h] += scores[static_cast<std::uint64_t>(h) * maximum_selected + slot] * value;
    }
#pragma unroll
    for (unsigned h = 0; h < G; ++h) {
        const auto head = head0 + h;
        const auto index = static_cast<std::uint64_t>(head) * head_dim + feature;
        const float attn = round_bf16(sum[h]);
        const float gate = round_bf16(1.0F / (1.0F + expf(-views.q_gate[r][static_cast<std::uint64_t>(head) * head_dim * 2ULL + head_dim + feature])));
        views.image(r)[index] = round_bf16(attn * gate);
        if (a8_codes)
            q8_rows::dense_a8_quantize_lane(views.image(r)[index], a8_codes + r * static_cast<std::uint64_t>(attention_heads) * head_dim + index,
                a8_deltas + (index / 32U) * q8_rows::kDenseA8Rows + r);
    }
}

// Q7 arm E, MTP draft only (STRIX_QPERSIST_DRAFT_ATTN, default on; draft_exact, teacher forward and the
// _sessions_ batch kernels keep their kernels): attention_tail_kernel with more value loads in
// flight. One workgroup per head as before (no feature split). Softmax is attention_softmax_body
// itself: the same trees, the same in-place BF16-rounded probabilities in the scores workspace.
// The value chain is attention_output_body's: one contracted fma per slot in ascending slot order
// in its 8-slot loop and in its residual loop (checked in the ISA), written here as
// __builtin_fmaf with the same operands in the same order, then the same round_bf16 epilogue.
// Only the issue order of the loads changes:
// - Stage: after softmax each lane copies the token indices and probabilities it owns
//   (slot = lane + 256k, the slots its softmax lane wrote) to LDS unchanged, so a value gather
//   no longer waits on an index load;
// - Depth: the value gathers of the next Depth slots are requested (into a second register
//   buffer) before the current Depth slots join the chain; padded or clamped slots past the
//   count are gathered from valid tokens and never join it.
// The caller admits head_dim == kThreads and selected_count <= kDraftAttentionSlots.
constexpr std::uint32_t kDraftAttentionSlots = 2051U;
// Publish (p32, needs Stage): softmax also stores each rounded probability, unchanged, into
// staged_probability (the workspace store stays), so staging skips the reload of probabilities.
// Only without a graph position/status word: there the softmax body always completes; otherwise
// (status early return, raw scores stay in the workspace) the probabilities are reloaded as in s<D>.
template <bool Stage, unsigned Depth, bool Publish = false>
__global__ __launch_bounds__(kThreads) void attention_tail_draft_kernel(const float* q_gate,
    const std::uint16_t* kv_state, const std::uint32_t* selected, float* scores, std::uint32_t selected_count,
    std::uint32_t maximum_selected, std::uint32_t attention_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, float* output, GraphSelectionArguments dynamic) {
    static_assert(Stage || !Publish);
    // Tokens padded to a whole number of Depth blocks with a valid token (slot 0), so a staged
    // block is always read whole; padded slots never join the chain.
    constexpr std::uint32_t Padded = (kDraftAttentionSlots + 63U) / 64U * 64U;
    static_assert(Depth <= 64U && 64U % Depth == 0U);
    __shared__ std::uint32_t staged_token[Stage ? Padded : 1U];
    __shared__ float staged_probability[Stage ? Padded : 1U];
    [[maybe_unused]] const bool published = Publish && dynamic.position.position == nullptr && dynamic.position.status == nullptr;
    if constexpr (Publish) attention_softmax_body<true>(scores, selected_count, maximum_selected, dynamic,
        published ? staged_probability : nullptr);
    else attention_softmax_body(scores, selected_count, maximum_selected, dynamic);
    __syncthreads();
    if (!graph_selected_count(dynamic, selected_count)) return;
    const auto head = static_cast<std::uint32_t>(blockIdx.x);
    const auto feature = static_cast<std::uint32_t>(threadIdx.x);
    const float* head_scores = scores + static_cast<std::uint64_t>(head) * maximum_selected;
    if constexpr (Stage) {
        const auto padded = (selected_count + Depth - 1U) / Depth * Depth;
        for (std::uint32_t index = feature; index < padded; index += kThreads) {
            staged_token[index] = selected[index < selected_count ? index : 0U];
            if (!published && index < selected_count) staged_probability[index] = head_scores[index];
        }
        __syncthreads();
    }
    if (head >= attention_heads || feature >= head_dim) return;
    const auto kv_head = head / (attention_heads / kv_heads);
    const auto kv_width = kv_heads * head_dim;
    const auto* values = kv_state + static_cast<std::uint64_t>(kv_width) + static_cast<std::uint64_t>(kv_head) * head_dim + feature;
    const auto row_elements = static_cast<std::uint64_t>(kv_width) * 2ULL;
    // Raw BF16 bits widened to 32 bits, double-buffered: a block's gathers never share a register
    // with the block in the chain, so the chain does not wait for the next block's gathers.
    std::uint32_t raw[2][Depth];
    [[maybe_unused]] float probability[2][Stage ? 1U : Depth];
    // Gathers block [first, first + Depth) into buffer b; unstaged slots past the count reload slot
    // `first` (staged slots are padded).
    const auto fetch = [&](unsigned b, std::uint32_t first) {
#pragma unroll
        for (unsigned i = 0; i < Depth; ++i) {
            if constexpr (Stage) raw[b][i] = values[static_cast<std::uint64_t>(staged_token[first + i]) * row_elements];
            else {
                const auto slot = first + i < selected_count ? first + i : first;
                raw[b][i] = values[static_cast<std::uint64_t>(selected[slot]) * row_elements];
                probability[b][i] = head_scores[slot];
            }
        }
    };
    const auto weight_at = [&](unsigned b, std::uint32_t slot, unsigned i) -> float {
        if constexpr (Stage) return staged_probability[slot + i];
        else return probability[b][i];
    };
    float sum = 0.0F;
    const auto full = selected_count / Depth * Depth;
    // One whole block from buffer b (the next block is requested into the other buffer before
    // this one joins the chain).
    const auto block = [&](unsigned b, std::uint32_t slot) {
        float value[Depth], weight[Depth];
#pragma unroll
        for (unsigned i = 0; i < Depth; ++i) { value[i] = __uint_as_float(raw[b][i] << 16U); weight[i] = weight_at(b, slot, i); }
        // Unconditional (the last block reloads itself into the other buffer): a branch here
        // would merge two wait states and make the chain wait for the new gathers.
        fetch(b ^ 1U, slot + Depth < selected_count ? slot + Depth : slot);
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (unsigned i = 0; i < Depth; ++i) sum = __builtin_fmaf(weight[i], value[i], sum);
    };
    // Last partial block from buffer b (already requested): its first count - full slots join the chain.
    const auto rest = [&](unsigned b) {
        if (full == selected_count) return;
        const auto count = selected_count - full;
#pragma unroll
        for (unsigned i = 0; i < Depth; ++i) {
            const float product_sum = __builtin_fmaf(weight_at(b, full, i), __uint_as_float(raw[b][i] << 16U), sum);
            sum = i < count ? product_sum : sum;
        }
    };
    fetch(0U, 0U);
    for (std::uint32_t slot = 0;; slot += 2U * Depth) {
        if (slot >= full) { rest(0U); break; }
        block(0U, slot);
        if (slot + Depth >= full) { rest(1U); break; }
        block(1U, slot + Depth);
    }

    // attention_output_body's epilogue.
    const float attn =
        round_bf16(sum);
    const float gate =
        round_bf16(
            1.0F /
            (1.0F +
             expf(
                 -q_gate[
                     static_cast<std::uint64_t>(
                         head) *
                         head_dim *
                         2ULL +
                     head_dim +
                     feature])));
    output[
        static_cast<std::uint64_t>(
            head) *
            head_dim +
        feature] =
        round_bf16(
            attn * gate);
    if (dynamic.position.status != nullptr && !isfinite(round_bf16(attn * gate))) atomicOr(dynamic.position.status, 8U);
}

// Q7 v<G>: value half of a two-launch draft tail. attention_softmax_kernel (the incumbent 256-lane
// softmax, unchanged) has published the BF16-rounded probabilities in the workspace; this launch
// only reads them (no workspace write, no in-place race). One workgroup = G query heads of one KV
// head x one feature slice of head_dim / F; thread = feature, one chain per head: the same
// ascending-slot fma per slot as attention_output_body (each value load feeds the G chains), the
// next Depth slots' tokens, values and probabilities requested into a second register buffer
// before the current block joins the chains, and attention_output_body's epilogue per head.
// The caller admits head_dim == kThreads, (heads / kv_heads) % G == 0 and selected_count >= 1.
template <unsigned G, unsigned F, unsigned Depth>
__global__ __launch_bounds__(kThreads / F) void attention_value_draft_kernel(const float* __restrict__ q_gate,
    const std::uint16_t* __restrict__ kv_state, const std::uint32_t* __restrict__ selected,
    const float* __restrict__ scores, std::uint32_t selected_count, std::uint32_t maximum_selected,
    std::uint32_t attention_heads, std::uint32_t kv_heads, std::uint32_t head_dim, float* __restrict__ output,
    GraphSelectionArguments dynamic) {
    if (!graph_selected_count(dynamic, selected_count)) return;
    constexpr unsigned Threads = kThreads / F;
    const auto head0 = static_cast<std::uint32_t>(blockIdx.x) / F * G;
    const auto feature = static_cast<std::uint32_t>(blockIdx.x) % F * Threads + static_cast<std::uint32_t>(threadIdx.x);
    if (head0 >= attention_heads || feature >= head_dim) return;
    const auto kv_head = head0 / (attention_heads / kv_heads);
    const auto kv_width = kv_heads * head_dim;
    const auto* values = kv_state + static_cast<std::uint64_t>(kv_width) + static_cast<std::uint64_t>(kv_head) * head_dim + feature;
    const auto row_elements = static_cast<std::uint64_t>(kv_width) * 2ULL;
    const float* head_scores = scores + static_cast<std::uint64_t>(head0) * maximum_selected;
    std::uint32_t raw[2][Depth];
    float probability[2][G][Depth];
    // Block [first, first + Depth) into buffer b; slots past the count reload slot `first`.
    const auto fetch = [&](unsigned b, std::uint32_t first) {
#pragma unroll
        for (unsigned i = 0; i < Depth; ++i) {
            const auto slot = first + i < selected_count ? first + i : first;
            raw[b][i] = values[static_cast<std::uint64_t>(selected[slot]) * row_elements];
#pragma unroll
            for (unsigned h = 0; h < G; ++h) probability[b][h][i] = head_scores[static_cast<std::uint64_t>(h) * maximum_selected + slot];
        }
    };
    float sum[G];
#pragma unroll
    for (unsigned h = 0; h < G; ++h) sum[h] = 0.0F;
    const auto full = selected_count / Depth * Depth;
    const auto block = [&](unsigned b, std::uint32_t slot) {
        float value[Depth], weight[G][Depth];
#pragma unroll
        for (unsigned i = 0; i < Depth; ++i) {
            value[i] = __uint_as_float(raw[b][i] << 16U);
#pragma unroll
            for (unsigned h = 0; h < G; ++h) weight[h][i] = probability[b][h][i];
        }
        fetch(b ^ 1U, slot + Depth < selected_count ? slot + Depth : slot); // unconditional, as in the tail kernel
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (unsigned i = 0; i < Depth; ++i)
#pragma unroll
            for (unsigned h = 0; h < G; ++h) sum[h] = __builtin_fmaf(weight[h][i], value[i], sum[h]);
    };
    const auto rest = [&](unsigned b) {
        if (full == selected_count) return;
        const auto count = selected_count - full;
#pragma unroll
        for (unsigned i = 0; i < Depth; ++i)
#pragma unroll
            for (unsigned h = 0; h < G; ++h) {
                const float product_sum = __builtin_fmaf(probability[b][h][i], __uint_as_float(raw[b][i] << 16U), sum[h]);
                sum[h] = i < count ? product_sum : sum[h];
            }
    };
    fetch(0U, 0U);
    for (std::uint32_t slot = 0;; slot += 2U * Depth) {
        if (slot >= full) { rest(0U); break; }
        block(0U, slot);
        if (slot + Depth >= full) { rest(1U); break; }
        block(1U, slot + Depth);
    }
#pragma unroll
    for (unsigned h = 0; h < G; ++h) {
        const auto head = head0 + h;
        // attention_output_body's epilogue.
        const float attn =
            round_bf16(sum[h]);
        const float gate =
            round_bf16(
                1.0F /
                (1.0F +
                 expf(
                     -q_gate[
                         static_cast<std::uint64_t>(
                             head) *
                             head_dim *
                             2ULL +
                         head_dim +
                         feature])));
        output[
            static_cast<std::uint64_t>(
                head) *
                head_dim +
            feature] =
            round_bf16(
                attn * gate);
        if (dynamic.position.status != nullptr && !isfinite(round_bf16(attn * gate))) atomicOr(dynamic.position.status, 8U);
    }
}

// STRIX_QSA_WINDOW_ATTN_V2 score: attention_score_rows_group256_kernel<Rows, G, Slots, Int8 = false> (G query heads of
// one KV head per wave share each K row; the wave's token indices loaded once; the next slot's K row requested before
// the current slot's products) with every product and addition operand-pinned to the kernel it replaces. IEEE results
// do not depend on operand order except for which NaN survives when two NaN operands meet, and the compiler commutes
// fmul / fadd freely, so the pinned order (src0 first, as measured in the replaced kernel's ISA) makes the bytes equal
// for NaN inputs too:
//   products                 v_mul_f32 q, k                                   (both orders)
//   Order::window            attention_score_rows_wave_kernel<Rows, 4>: lane-local p[j + 4] + p[j], p[2] + p[0],
//                            p[3] + p[1], then p[0] + p[1]; lanes 16..1: own (lower lane) + shuffled (upper lane)
//   Order::lds_tree          attention_score_kernel / attention_score_body: every level scratch[f + s] + scratch[f]
//                            (upper first), lanes 16..1 included
// The lane steps run as the reduce-scatter of hip_qsa_scatter.inc (the same pairs: each step adds the partials of the
// lower lane L and the upper lane L ^ D), with the lower / upper operand chosen by the lane bit rather than by which
// lane computes it. head_dim == 256 (checked by the caller).
enum class QsaScoreOrder : unsigned { window, lds_tree };
__device__ __forceinline__ float qsa_pinned_add(float src0, float src1) {
    float d;
    asm("v_add_f32 %0, %1, %2" : "=v"(d) : "v"(src0), "v"(src1));
    return d;
}
__device__ __forceinline__ float qsa_pinned_mul(float src0, float src1) {
    float d;
    asm("v_mul_f32 %0, %1, %2" : "=v"(d) : "v"(src0), "v"(src1));
    return d;
}
template <QsaScoreOrder Order> __device__ __forceinline__ float qsa_pinned_pair(float lower, float upper) {
    if constexpr (Order == QsaScoreOrder::lds_tree) return qsa_pinned_add(upper, lower);
    else return qsa_pinned_add(lower, upper);
}
template <unsigned P, QsaScoreOrder Order, unsigned Step = 0U>
__device__ __forceinline__ void qsa_pinned_scatter(float (&v)[P], unsigned lane) {
    if constexpr (Step < 5U) {
        constexpr unsigned D = 16U >> Step, Count = (P >> Step) > 0U ? (P >> Step) : 1U;
        const bool upper = (lane & D) != 0U;
        if constexpr (Count > 1U) {
            constexpr unsigned H = Count / 2U;
#pragma unroll
            for (unsigned j = 0; j < H; ++j) {
                // Lower lane: its own head-j partial and the upper lane's; upper lane: the lower lane's head-(j + H)
                // partial and its own. Selects move bits.
                const float low = qsa_lane_xor<D>(v[j]), high = qsa_lane_xor<D>(v[j + H]);
                v[j] = qsa_pinned_pair<Order>(upper ? high : v[j], upper ? v[j + H] : low);
            }
        } else {
            const float other = qsa_lane_xor<D>(v[0]);
            v[0] = qsa_pinned_pair<Order>(upper ? other : v[0], upper ? v[0] : other);
        }
        qsa_pinned_scatter<P, Order, Step + 1U>(v, lane);
    }
}
template <unsigned Rows, unsigned G, unsigned Slots, QsaScoreOrder Order>
__global__ __launch_bounds__(256) void attention_score_rows_pinned_kernel(QsaRowsAttentionViews<Rows> views,
    const std::uint16_t* kv_state, std::uint32_t maximum_selected, std::uint32_t attention_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t slots) {
#pragma clang fp contract(off)
    static_assert(Slots <= 32U);
    constexpr unsigned Dim = 256U; // head_dim == Dim (checked by the caller)
    const auto lane = threadIdx.x % 32U;
    const auto wave = blockIdx.x * 8U + threadIdx.x / 32U;
    const auto groups = (slots + Slots - 1U) / Slots, head_groups = attention_heads / G;
    const auto r = wave / (head_groups * groups);
    if (r >= Rows) return;
    const auto head0 = wave / groups % head_groups * G, first = wave % groups * Slots;
    if (first >= views.count[r]) return;
    const auto kv_head = head0 / (attention_heads / kv_heads);
    float qv[G][8];
#pragma unroll
    for (unsigned h = 0; h < G; ++h) {
        const auto* q = views.q_gate[r] + static_cast<std::uint64_t>(head0 + h) * Dim * 2ULL;
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) qv[h][j] = q[lane + 32U * j];
    }
    const float scale = sqrtf(static_cast<float>(head_dim));
    const auto last = min(first + Slots, views.count[r]);
    const auto tokens = lane < last - first ? views.selected[r][first + lane] : 0U;
    float* const scores = views.scores[r]; // loaded once: a store could alias the argument block
    const auto* kbase = kv_state + static_cast<std::uint64_t>(kv_head) * Dim + lane;
    const auto row_elements = static_cast<std::uint64_t>(kv_heads) * Dim * 2ULL;
    std::uint16_t raw[8];
    const auto load = [&](std::uint32_t index) {
        const auto token = static_cast<std::uint32_t>(__builtin_amdgcn_readlane(static_cast<int>(tokens), static_cast<int>(index)));
        const auto* row = kbase + static_cast<std::uint64_t>(token) * row_elements;
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) raw[j] = row[32U * j];
    };
    load(0U);
    constexpr unsigned P = G <= 2U ? 2U : G <= 4U ? 4U : G <= 8U ? 8U : 16U;
    for (auto slot = first; slot < last; ++slot) {
        float kf[8];
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) kf[j] = bf16_to_float(raw[j]);
        if (slot + 1U < last) load(slot + 1U - first);
        float partial[P];
#pragma unroll
        for (unsigned h = 0; h < P; ++h) {
            if (h >= G) { partial[h] = 0.0F; continue; }
            float p[8];
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) p[j] = qsa_pinned_mul(qv[h][j], kf[j]);
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) p[j] = qsa_pinned_add(p[j + 4U], p[j]);
            p[0] = qsa_pinned_add(p[2], p[0]); p[1] = qsa_pinned_add(p[3], p[1]);
            partial[h] = qsa_pinned_pair<Order>(p[0], p[1]);
        }
        qsa_pinned_scatter<P, Order>(partial, lane);
        const auto h = qsa_scatter_head<P>(lane);
        if (h < G && (lane & ((32U / P) - 1U)) == 0U)
            scores[static_cast<std::uint64_t>(head0 + h) * maximum_selected + slot] = partial[0] / scale;
    }
}

// Draft attention variants (host). Score: the production instantiation (w4) or wider forms that are
// existing exact score kernels: wP = attention_score_rows_wave_kernel<1, P> (P slots per wave),
// gGsS = attention_score_rows_group256_kernel<1, G, S> (G query heads of one KV head per wave share
// each K row, S slots per wave). Tail: incumbent = attention_tail_kernel, gD = no staging with
// depth D, sD = staged with depth D, p32 = s32 with probabilities published to LDS by the softmax,
// v2 = attention_softmax_kernel + attention_value_draft_kernel<1, 2, 32> (48 workgroups),
// v4 = attention_softmax_kernel + attention_value_draft_kernel<4, 8, 16> (4 heads per value load).
enum class DraftScoreVariant : std::uint32_t { w4, w8, w16, g2s8, g4s8, g4s16 };
enum class DraftTailVariant : std::uint32_t { incumbent, g32, s8, s16, s32, s64, p32, v2, v4 };
constexpr std::array<std::string_view, 6> kDraftScoreNames{"w4", "w8", "w16", "g2s8", "g4s8", "g4s16"};
constexpr std::array<std::string_view, 9> kDraftTailNames{"incumbent", "g32", "s8", "s16", "s32", "s64", "p32", "v2", "v4"};
template <class Variant, std::size_t N>
Variant parse_draft_variant(const char* text, const std::array<std::string_view, N>& names, const char* what) {
    for (std::size_t i = 0; i < N; ++i) if (text != nullptr && names[i] == text) return static_cast<Variant>(i);
    throw std::invalid_argument(std::string("unknown draft attention ") + what + " variant: " + (text ? text : "(null)"));
}
DraftScoreVariant draft_score_variant(const char* text) { return parse_draft_variant<DraftScoreVariant>(text, kDraftScoreNames, "score"); }
DraftTailVariant draft_tail_variant(const char* text) { return parse_draft_variant<DraftTailVariant>(text, kDraftTailNames, "tail"); }
std::uint32_t draft_score_group(DraftScoreVariant v) {
    return v == DraftScoreVariant::g2s8 ? 2U : v == DraftScoreVariant::g4s8 || v == DraftScoreVariant::g4s16 ? 4U : 1U;
}
std::uint32_t draft_tail_group(DraftTailVariant v) { return v == DraftTailVariant::v4 ? 4U : 1U; }
bool draft_attention_admitted(DraftScoreVariant score, DraftTailVariant tail, std::uint32_t heads, std::uint32_t kv_heads,
    std::uint32_t dim, std::uint32_t count) {
    return dim == kThreads && kv_heads && heads % kv_heads == 0U && (heads / kv_heads) % draft_score_group(score) == 0U &&
        (heads / kv_heads) % draft_tail_group(tail) == 0U && count && count <= kDraftAttentionSlots;
}
void launch_draft_attention_scores(DraftScoreVariant variant, const QsaRowsAttentionViews<1U>& views,
    const std::uint16_t* kv, std::uint32_t maximum, std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t dim,
    std::uint32_t count, hipStream_t stream) {
    const auto blocks = [&](std::uint32_t head_groups, std::uint32_t slots) {
        const auto waves = static_cast<std::uint64_t>(head_groups) * ((count + slots - 1U) / slots);
        return dim3(checked_u32((waves + 7U) / 8U, "MTP draft score grid"));
    };
    switch (variant) {
    case DraftScoreVariant::w4:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_score_rows_wave_kernel<1U, 4U>), blocks(heads, 4U), dim3(kThreads), 0,
            stream, views, kv, maximum, heads, kv_heads, dim, count);
        break;
    case DraftScoreVariant::w8:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_score_rows_wave_kernel<1U, 8U>), blocks(heads, 8U), dim3(kThreads), 0,
            stream, views, kv, maximum, heads, kv_heads, dim, count);
        break;
    case DraftScoreVariant::w16:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_score_rows_wave_kernel<1U, 16U>), blocks(heads, 16U), dim3(kThreads), 0,
            stream, views, kv, maximum, heads, kv_heads, dim, count);
        break;
    case DraftScoreVariant::g2s8:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_score_rows_group256_kernel<1U, 2U, 8U>), blocks(heads / 2U, 8U),
            dim3(kThreads), 0, stream, views, kv, maximum, heads, kv_heads, dim, count);
        break;
    case DraftScoreVariant::g4s8:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_score_rows_group256_kernel<1U, 4U, 8U>), blocks(heads / 4U, 8U),
            dim3(kThreads), 0, stream, views, kv, maximum, heads, kv_heads, dim, count);
        break;
    case DraftScoreVariant::g4s16:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_score_rows_group256_kernel<1U, 4U, 16U>), blocks(heads / 4U, 16U),
            dim3(kThreads), 0, stream, views, kv, maximum, heads, kv_heads, dim, count);
        break;
    }
    check_hip(hipGetLastError(), "MTP draft attention scores enqueue");
}
// STRIX_QPERSIST_DRAFT_ATTN (attn) and its sub-knobs: unset or empty = on by default, "1" = on
// (explicit), "0" = incumbent; any other value, or an unknown variant, throws (fail closed).
// Default pair g4s16 + s32.
struct DraftAttentionChoice {
    bool on = false, defaulted = false;
    DraftScoreVariant score = DraftScoreVariant::g4s16;
    DraftTailVariant tail = DraftTailVariant::s32;
};
DraftAttentionChoice resolve_draft_attention(const char* attn, const char* score, const char* tail) {
    const std::string_view mode = attn ? attn : "";
    if (mode == "0") return {};
    if (!mode.empty() && mode != "1")
        throw std::invalid_argument(std::string("STRIX_QPERSIST_DRAFT_ATTN must be unset, 0 or 1: ") + attn);
    return {true, mode.empty(), draft_score_variant(score ? score : "g4s16"), draft_tail_variant(tail ? tail : "s32")};
}
// Two-launch tails: the incumbent softmax kernel, then the value kernel on the published probabilities.
void launch_draft_attention_softmax(float* scores, std::uint32_t count, std::uint32_t maximum, std::uint32_t heads,
    GraphSelectionArguments dynamic, hipStream_t stream) {
    hipLaunchKernelGGL(attention_softmax_kernel, dim3(heads), dim3(kThreads), 0, stream, scores, count, maximum, dynamic);
    check_hip(hipGetLastError(), "MTP draft attention softmax enqueue");
}
void launch_draft_attention_value(DraftTailVariant variant, const float* q_gate, const std::uint16_t* kv,
    const std::uint32_t* selected, const float* scores, std::uint32_t count, std::uint32_t maximum, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t dim, float* output, GraphSelectionArguments dynamic, hipStream_t stream) {
    if (variant == DraftTailVariant::v2)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_value_draft_kernel<1U, 2U, 32U>), dim3(heads * 2U), dim3(kThreads / 2U),
            0, stream, q_gate, kv, selected, scores, count, maximum, heads, kv_heads, dim, output, dynamic);
    else if (variant == DraftTailVariant::v4)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_value_draft_kernel<4U, 8U, 16U>), dim3(heads / 4U * 8U), dim3(kThreads / 8U),
            0, stream, q_gate, kv, selected, scores, count, maximum, heads, kv_heads, dim, output, dynamic);
    else throw std::invalid_argument("draft attention value launch needs a two-launch tail variant");
    check_hip(hipGetLastError(), "MTP draft attention value enqueue");
}
bool qsa_decode_value_v2_enabled() {
    static const bool on = qpf2k::long_context_item_enabled("STRIX_QSA_DECODE_VALUE_V2", false);
    return on && !qpf2k::baseline && !qpf2k::gsq_model;
}
bool launch_decode_r1_value_v2(const float* q_gate, const std::uint16_t* kv, const std::uint32_t* selected, const float* scores,
    std::uint32_t count, std::uint32_t maximum, std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t dim, float* output,
    const GraphSelectionArguments& selection, bool device_graph, hipStream_t stream) {
    if (!qsa_decode_value_v2_enabled() || device_graph || selection.position.position || selection.position.status || !count ||
        maximum > kDraftAttentionSlots || !draft_attention_admitted(DraftScoreVariant::w4, DraftTailVariant::v4, heads, kv_heads, dim, count))
        return false;
    launch_draft_attention_value(DraftTailVariant::v4, q_gate, kv, selected, scores, count, maximum, heads, kv_heads, dim, output,
        selection, stream);
    return true;
}
void launch_draft_attention_tail(DraftTailVariant variant, const float* q_gate, const std::uint16_t* kv,
    const std::uint32_t* selected, float* scores, std::uint32_t count, std::uint32_t maximum, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t dim, float* output, GraphSelectionArguments dynamic, hipStream_t stream) {
    const auto launch = [&](auto kernel) {
        hipLaunchKernelGGL(kernel, dim3(heads), dim3(kThreads), 0, stream,
            q_gate, kv, selected, scores, count, maximum, heads, kv_heads, dim, output, dynamic);
    };
    switch (variant) {
    case DraftTailVariant::incumbent: launch(attention_tail_kernel); break;
    case DraftTailVariant::g32: launch(attention_tail_draft_kernel<false, 32U>); break;
    case DraftTailVariant::s8: launch(attention_tail_draft_kernel<true, 8U>); break;
    case DraftTailVariant::s16: launch(attention_tail_draft_kernel<true, 16U>); break;
    case DraftTailVariant::s32: launch(attention_tail_draft_kernel<true, 32U>); break;
    case DraftTailVariant::s64: launch(attention_tail_draft_kernel<true, 64U>); break;
    case DraftTailVariant::p32: launch(attention_tail_draft_kernel<true, 32U, true>); break;
    case DraftTailVariant::v2:
    case DraftTailVariant::v4:
        launch_draft_attention_softmax(scores, count, maximum, heads, dynamic, stream);
        launch_draft_attention_value(variant, q_gate, kv, selected, scores, count, maximum, heads, kv_heads, dim, output,
            dynamic, stream);
        break;
    }
    check_hip(hipGetLastError(), "MTP draft attention tail enqueue");
}

// STRIX_QSA_WINDOW_ATTN_V2 (exact, on by default): single-session MTP verify windows of 2/4/8 rows (BF16 KV,
// head_dim 256, a multiple of 12 query heads per KV head) score with attention_score_rows_pinned_kernel<Rows, 12, S,
// window> (8 slots per wave at 2 rows, 16 at 4-8 rows) instead of attention_score_rows_wave_kernel<Rows, 4>; softmax
// and value/gate keep their kernels (fused tail included). The MTP draft step, where it runs attention_score_kernel
// (no STRIX_QPERSIST_DRAFT_SCORE), scores with attention_score_rows_pinned_kernel<1, 4, 16, lds_tree> and keeps its
// tail. Prefill tiles, multi-session cohort windows, int8 KV and other shapes keep their kernels, and so do windows
// with an explicit STRIX_QSA_WINDOW_GROUPED=1 or STRIX_QSA_SCORE_SLOTS other than 4 (the explicit selection wins).
// Off with STRIX_QSA_WINDOW_ATTN_V2=0|off|false, under the baseline and under STRIX_QPF2K_DEFAULTS=0.
bool window_attn_v2_switch(const char* setting, const char* defaults, bool baseline) {
    const auto off = [](const char* v) {
        return v && (std::string_view(v) == "0" || std::string_view(v) == "off" || std::string_view(v) == "false");
    };
    return !baseline && !off(setting) && !(defaults && defaults[0] == '0');
}
bool qsa_window_attn_v2_enabled() {
    static const bool on = window_attn_v2_switch(std::getenv("STRIX_QSA_WINDOW_ATTN_V2"), std::getenv("STRIX_QPF2K_DEFAULTS"), false);
    return on && !qpf2k::baseline;
}
// STRIX_QSA_WINDOW_GROUPED=1 (opt-in KV-head-grouped window score) and STRIX_QSA_SCORE_SLOTS=1|2|4|8 (default 4).
bool qsa_window_grouped_requested() {
    static const bool on = [] { const char* v = std::getenv("STRIX_QSA_WINDOW_GROUPED"); return v && v[0] == '1'; }();
    return on;
}
unsigned qsa_score_slots_per_wave() {
    static const unsigned per_wave = [] {
        const char* v = std::getenv("STRIX_QSA_SCORE_SLOTS");
        const int n = v ? std::atoi(v) : 4;
        return n == 2 || n == 4 || n == 8 ? static_cast<unsigned>(n) : 1U;
    }();
    return per_wave;
}
constexpr std::uint32_t kWindowV2Group = 12U, kDraftV2Group = 4U;
bool window_attn_v2_admitted(std::uint32_t group, std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t dim) {
    const auto per_kv = kv_heads && heads % kv_heads == 0U ? heads / kv_heads : 0U;
    return per_kv && per_kv % group == 0U && dim == 256U;
}
// Whether a known-rows score launch takes the v2 window score: only the single-session verify-window caller sets
// verify_window, and only where the replaced kernel is attention_score_rows_wave_kernel<Rows, 4>.
bool window_attn_v2_route(bool enabled, bool verify_window, unsigned rows, bool int8, bool fast, bool window_grouped,
    unsigned score_slots, std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t dim) {
    return enabled && verify_window && (rows == 2U || rows == 4U || rows == 8U) && !int8 && !fast && !window_grouped &&
        score_slots == 4U && window_attn_v2_admitted(kWindowV2Group, heads, kv_heads, dim);
}
// Whether the MTP draft step's score takes the v2 draft score (in place of attention_score_kernel).
bool draft_attn_v2_route(bool enabled, bool draft, bool qpersist_draft_score, std::uint32_t heads, std::uint32_t kv_heads,
    std::uint32_t dim) {
    return enabled && draft && !qpersist_draft_score && window_attn_v2_admitted(kDraftV2Group, heads, kv_heads, dim);
}
template <unsigned Rows>
void launch_window_attn_v2_scores(const QsaRowsAttentionViews<Rows>& views, const std::uint16_t* kv, std::uint32_t maximum,
    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t dim, std::uint32_t count, hipStream_t stream) {
    constexpr unsigned G = kWindowV2Group, S = Rows <= 2U ? 8U : 16U;
    const auto waves = static_cast<std::uint64_t>(Rows) * (heads / G) * ((count + S - 1U) / S);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_score_rows_pinned_kernel<Rows, G, S, QsaScoreOrder::window>),
        dim3(checked_u32((waves + 7U) / 8U, "QSA window v2 score grid")), dim3(kThreads), 0, stream, views, kv, maximum,
        heads, kv_heads, dim, count);
    check_hip(hipGetLastError(), "QSA window v2 score launch");
}
// The MTP draft step's score where it does not run the wave kernel: attention_score_kernel, or with v2 (the caller
// routes it with draft_attn_v2_route) attention_score_rows_pinned_kernel<1, 4, 16, lds_tree> on the same buffers.
void launch_draft_one_block_scores(bool v2, const float* q_gate, const std::uint16_t* kv, const std::uint32_t* selected,
    float* scores, std::uint32_t count, std::uint32_t maximum, std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t dim,
    GraphSelectionArguments selection, hipStream_t stream) {
    if (v2) {
        QsaRowsAttentionViews<1U> views{};
        views.q_gate[0] = q_gate; views.selected[0] = selected; views.scores[0] = scores; views.count[0] = count;
        const auto waves = static_cast<std::uint64_t>(heads / kDraftV2Group) * ((count + 15U) / 16U);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_score_rows_pinned_kernel<1U, kDraftV2Group, 16U, QsaScoreOrder::lds_tree>),
            dim3(checked_u32((waves + 7U) / 8U, "MTP draft v2 score grid")), dim3(kThreads), 0, stream, views, kv, maximum,
            heads, kv_heads, dim, count);
    } else hipLaunchKernelGGL(attention_score_kernel, dim3(checked_u32(static_cast<std::uint64_t>(heads) * count, "MTP attention score grid")),
        dim3(kThreads), 0, stream, q_gate, kv, selected, count, maximum, heads, kv_heads, dim, scores, selection);
    check_hip(hipGetLastError(), "MTP attention scores enqueue");
}
// STRIX_QSA_DECODE_ATTN_V2 (default on): the plain one-row QSA step (launch_indexed_attention, eager) scores with the
// draft step's attention_score_rows_pinned_kernel<1, 4, 16, lds_tree> in place of attention_score_kernel: the four query
// heads of one KV head share each K row, every product and addition operand-pinned to attention_score_kernel's tree, so the
// scores are byte-identical (NaN payloads included). Device graphs, graph-position selection and a non-finite status
// word keep attention_score_kernel (the pinned kernel reads neither). Off with STRIX_QSA_DECODE_ATTN_V2=0, under the
// baseline, STRIX_QPF2K_DEFAULTS=0 and GSQ checkpoints.
bool qsa_decode_attn_v2_enabled() {
    // Environment policy once; the GSQ exclusion per call (the model is a thread-local scope).
    static const bool on = qpf2k::long_context_item_enabled("STRIX_QSA_DECODE_ATTN_V2", false);
    return on && !qpf2k::baseline && !qpf2k::gsq_model;
}
bool decode_r1_attn_v2_route(bool enabled, bool device_graph, const GraphSelectionArguments& selection, std::uint32_t count,
    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t dim) {
    return enabled && !device_graph && !selection.position.position && !selection.position.status && count &&
        window_attn_v2_admitted(kDraftV2Group, heads, kv_heads, dim);
}
// STRIX_QSA_DECODE_VALUE_V2 (default on): after attention_softmax_kernel the plain one-row step forms value/gate with the
// draft step's attention_value_draft_kernel<4, 8, 16> (four query heads of one KV head per value load, eight feature slices,
// 16 slots per register block) in place of attention_output_kernel: per (head, feature) the same ascending-slot fma chain
// over the published probabilities and attention_output_body's BF16 epilogue, so the bytes are equal. Same admission as the
// v2 score (eager, no graph position or status) plus the draft value bounds. Off with STRIX_QSA_DECODE_VALUE_V2=0, under
// the baseline, STRIX_QPF2K_DEFAULTS=0 and GSQ checkpoints.
bool qsa_decode_value_v2_enabled();
bool launch_decode_r1_scores_v2(const float* q_gate, const std::uint16_t* kv, const std::uint32_t* selected, float* scores,
    std::uint32_t count, std::uint32_t maximum, std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t dim,
    const GraphSelectionArguments& selection, bool device_graph, hipStream_t stream) {
    if (!decode_r1_attn_v2_route(qsa_decode_attn_v2_enabled(), device_graph, selection, count, heads, kv_heads, dim)) return false;
    launch_draft_one_block_scores(true, q_gate, kv, selected, scores, count, maximum, heads, kv_heads, dim, selection, stream);
    return true;
}

#include "hip_qsa_pooled_score.inc"

// Journal: per absolute layer, (ratio-1) partial-block tail cells then 8 row keys.
std::uint64_t qsa_journal_layer_elements(const ModelContract& model) {
    return (model.indexer_compress_ratio - 1U + 8U) * model.indexer_head_dim;
}

std::string kernel_id(const ExecutionBindingRequest&, const HipIndexedAttentionLauncherConfig&);

template <unsigned Rows> struct QsaKnownRows {
    std::array<PreparedIndexedLaunch, Rows> row;
    std::array<PreparedIndexedSelection, Rows> selection;
    std::array<std::uint64_t, Rows> position{};
    std::array<bool, 5> prefetch{}; // index_q, index_k, q, k, v: serial prefetch8 choice
    std::uint16_t* journal = nullptr;
};

// head: a prefill row of the same step already admitted in full (its context and launch);
// the rows here then share its admission (prepare_indexed_attention_launch_like). share alone:
// rows after the first share the first row's admission (prefill tile 0).
struct QsaHeadRow { const ExecutionDispatchContext* context = nullptr; const PreparedIndexedLaunch* launch = nullptr; bool share = false; };
template <unsigned Rows>
QsaKnownRows<Rows> prepare_qsa_known_rows(std::span<const ExecutionDispatchContext> rows,
    const HipIndexedAttentionLauncherConfig& config, HipQsaIndexJournal journal, QsaHeadRow head = {}, bool ring_entry = false) {
    static_assert(Rows == 2U || Rows == 4U || Rows == 8U);
    if (rows.size() != Rows) throw std::invalid_argument("QSA known-window row count differs");
    if (config.enable_device_graph_inputs || config.qsa_hierarchical_selection || config.qsa_gqa_heads_per_block != 1)
        throw std::invalid_argument("QSA known-window rows support the eager legacy-selection launcher only");
    const auto* scope = current_hip_launch_context();
    if (scope && scope->graph_capture) throw std::invalid_argument("QSA known-window rows are eager only");
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    check_hip(hipStreamIsCapturing(hip_execution_stream(), &capture), "QSA known-window capture query");
    if (capture != hipStreamCaptureStatusNone) throw std::invalid_argument("QSA known-window rows are eager only");
    const auto expected_id = kernel_id(ExecutionBindingRequest{}, config);
    QsaKnownRows<Rows> result;
    const auto& first = rows[0];
    if (first.position > std::numeric_limits<std::uint64_t>::max() - (Rows - 1U))
        throw std::invalid_argument("QSA known-window position overflows");
    struct Range { std::uintptr_t address; std::uint64_t bytes; };
    const auto overlap = [](Range x, Range y) { return x.address < y.address + y.bytes && y.address < x.address + x.bytes; };
    std::array<Range, Rows> inputs{};
    std::array<Range, 2U * Rows + 2U> writable{};
    for (unsigned r = 0; r < Rows; ++r) {
        const auto& context = rows[r];
        if (context.kernel_id && *context.kernel_id != expected_id)
            throw std::invalid_argument("QSA known-window configuration differs from the row's serial kernel");
        if (head.context) result.row[r] = prepare_indexed_attention_launch_like(*head.launch, *head.context, context, config);
        else if (r && head.share) result.row[r] = prepare_indexed_attention_launch_like(result.row[0], rows[0], context, config);
        else result.row[r] = prepare_indexed_attention_launch(context, &config);
        const auto& p = result.row[r];
        require_qsa_score_source(!p.workspace_layout.block_scores_bytes,ring_entry && config.exact_score_owner!=nullptr);
        if(!p.workspace_layout.block_scores_bytes) {
            if(!context.memory) throw std::invalid_argument("QSA scratch requires an arena generation");
            config.exact_score_owner->validate_generation(context.memory->allocation_generation);
        }
        const auto& a = result.row[0];
        if (p.device_graph) throw std::invalid_argument("QSA known-window rows are eager only");
        if (context.execution_identity != first.execution_identity || context.step_id != first.step_id ||
            context.op != first.op || context.layer != first.layer || context.position != first.position + r ||
            context.reads[0] != first.reads[0] || context.writes[0] != first.writes[0] ||
            context.temporaries[0] != first.temporaries[0] || p.session_layout != a.session_layout ||
            p.kv_state.address != a.kv_state.address || p.kv_state.bytes != a.kv_state.bytes ||
            p.raw_state.address != a.raw_state.address || p.raw_state.bytes != a.raw_state.bytes)
            throw std::invalid_argument("QSA known-window rows do not retain one ordered session binding");
        for (const auto member : {&IndexedTensors::q, &IndexedTensors::k, &IndexedTensors::v, &IndexedTensors::out,
            &IndexedTensors::q_norm, &IndexedTensors::k_norm, &IndexedTensors::index_q, &IndexedTensors::index_k,
            &IndexedTensors::index_q_norm, &IndexedTensors::index_k_norm})
            if ((p.tensors.*member).runtime != (a.tensors.*member).runtime || (p.tensors.*member).manifest != (a.tensors.*member).manifest)
                throw std::invalid_argument("QSA known-window projection owners differ between rows");
        result.selection[r] = prepare_indexed_attention_selection(context, p, config.model);
        static_cast<void>(prepare_indexed_attention_score_grid(p.attention_heads, result.selection[r].selected_count));
        result.position[r] = context.position;
        inputs[r] = {p.input.address, p.input.bytes};
        writable[2U * r] = {p.output.address, p.output.bytes};
        writable[2U * r + 1U] = {p.temporary.address, p.temporary.bytes};
    }
    const auto& a = result.row[0];
    static_cast<void>(checked_u32(a.workspace_layout.maximum_selected_tokens, "maximum_selected_tokens"));
    static_cast<void>(checked_u32(config.model.indexer_n_heads, "indexer_n_heads"));
    static_cast<void>(checked_u32(config.model.rotary_dim(), "rotary_dim"));
    writable[2U * Rows] = {a.kv_state.address, a.kv_state.bytes};
    writable[2U * Rows + 1U] = {a.raw_state.address, a.raw_state.bytes};
    for (unsigned i = 0; i < writable.size(); ++i) {
        for (const auto input : inputs) if (overlap(writable[i], input)) throw std::invalid_argument("QSA known-window writable/input overlap");
        for (const auto* tensor : {&a.tensors.q, &a.tensors.k, &a.tensors.v, &a.tensors.out, &a.tensors.q_norm, &a.tensors.k_norm,
            &a.tensors.index_q, &a.tensors.index_k, &a.tensors.index_q_norm, &a.tensors.index_k_norm})
            if (overlap(writable[i], {tensor->runtime->address, tensor->runtime->bytes}))
                throw std::invalid_argument("QSA known-window writable/weight overlap");
        for (unsigned j = 0; j < i; ++j) if (overlap(writable[i], writable[j])) throw std::invalid_argument("QSA known-window writable owners overlap");
    }
    // Each projection keeps the serial launcher's own per-tensor organization.
    const std::array<const TensorPair*, 5> projections{&a.tensors.index_q, &a.tensors.index_k, &a.tensors.q, &a.tensors.k, &a.tensors.v};
    for (unsigned m = 0; m < projections.size(); ++m) {
        const auto* tensor = projections[m];
        static_cast<void>(weight_kind(tensor->manifest->ggml_type));
        result.prefetch[m] = config.prefetch_q8_input && qsa_input_prefetch::eligible(*tensor->manifest);
        if (result.prefetch[m] && (tensor->runtime->address % 16U || tensor->manifest->absolute_offset % 16U ||
            tensor->runtime->bytes < qsa_input_prefetch::required_weight_bytes(a.hidden, tensor->manifest->dimensions[1])))
            throw std::invalid_argument("QSA known-window prefetch weight alignment/extent invalid");
    }
    static_cast<void>(weight_kind(a.tensors.out.manifest->ggml_type));
    for (const auto* norm : {&a.tensors.q_norm, &a.tensors.k_norm, &a.tensors.index_q_norm, &a.tensors.index_k_norm})
        static_cast<void>(small_weight_kind(norm->manifest->ggml_type));
    if (config.wave8_q8_output && !(config.dense_a8_projections && a.tensors.out.manifest->ggml_type != 8U &&
            q8_rows::dense_a8_type(a.tensors.out.manifest->ggml_type))) {
        if (a.tensors.out.manifest->ggml_type != 8U || a.attention_width % 32U)
            throw std::invalid_argument("wave8 QSA output requires aligned Q8_0 weights");
        q8_projection_organization::validate_geometry(a.attention_width, a.hidden);
    }
    if (config.dense_a8_projections && (!q8_rows::dense_a8_width(a.hidden) || !q8_rows::dense_a8_width(a.attention_width)))
        throw std::invalid_argument("QSA known-window dense A8 width is not admitted");
    if (journal.address || journal.bytes) {
        const auto& model = config.model;
        if (!journal.address || journal.address % 2U || journal.bytes != qsa_index_journal_bytes(model) ||
            journal.bytes > UINTPTR_MAX - journal.address || *first.layer >= model.hidden_layers)
            throw std::invalid_argument("QSA known-window index journal view is invalid");
        for (const auto view : writable) if (overlap({journal.address, journal.bytes}, view))
            throw std::invalid_argument("QSA known-window index journal aliases a writable owner");
        result.journal = reinterpret_cast<std::uint16_t*>(journal.address) +
            *first.layer * qsa_journal_layer_elements(model) + (model.indexer_compress_ratio - 1U) * model.indexer_head_dim;
    }
    return result;
}

#include "hip_qsa_prefill_index.inc"
#include "hip_qsa_prefill_flash.inc"
// STRIX_QPF2K_QSA=1 (qpf2k/attn, default off): prefill groups score the indexer blocks with qsa_flash::score_kernel
// (one lane per block, the parked kernel's products, tree, relu sum and division: the same block scores).
// STRIX_QPF2K_QSA_MASK (diagnostic, with STRIX_QPF2K_QSA=1) keeps only some parts: bit 0 the scorer, bit 1 (with bit 0)
// the selection from the scorer's group maxima (qsa_flash::select_kernel, the same selections), bit 2 the wave form of the
// grouped norm / RoPE (with STRIX_QSA_NORM_GROUP=1), bit 3 16-row workgroups for the indexer q/k projections (with
// STRIX_QSA_INDEX_ROWS=1), bit 4 (with bits 0 and 1 and STRIX_QSA_ATTEND_BATCH=1) one selection launch per prefill call, bit 5 the FP16 KV
// shadow attend (bench NO-GAIN, kept off), bit 6 (with STRIX_QSA_ATTEND_BATCH=1, the default 4-row tile, prefill mode 0: the
// batched attend) the group-major order of the batched attend.
// Bit 9 (0x200, default OFF, requires bits 0 and 1): exact per-64-row-group stage batching; see
// hip_qsa_prefill_batch.inc for the all-or-nothing admission and descriptor lifetime rules.
// Non-null only inside one external score-ring group, never retained after selection.
thread_local float* qsa_exact_maxima = nullptr;
thread_local std::uint32_t qsa_exact_maxima_stride = 0;
// Bit 13 (0x2000, default ON on eligible routes, off with mask 0x257, with bit 6): the group-major attend under an
// eight-waves-per-SIMD register budget (three resident workgroups per WGP instead of two); same statements, bitwise
// the same outputs.
constexpr std::uint32_t qpf2k_maxima_stride = 65536U / qsa_flash::kGroup; // groups of max_position / 4 blocks
// Per pooled block magnitude ranges (qsa_flash::key_range_kernel), one buffer like the pool's (grown with a sync).
// STRIX_QPF2K_QSA bit 4 (with bit 1, prefill calls whose union / attend is batched): the groups' selections are deferred
// and run in one select_batch_kernel launch per call (or whenever the maxima arena fills). Maxima of the deferred groups
// live in an arena (rows x ceil(blocks / 4) floats per group); the row list lives in a pinned host ring slot (read by the
// kernel in place; a slot is reused after its kernel completed).
struct Qpf2kSelectGroup {
    std::array<std::uint64_t, 64> position{};
    std::array<const std::uint32_t*, 64> selected{};
    std::array<std::uint32_t, 64> count{};
    std::array<float*, 64> scores{};
    std::array<std::uint32_t, 64> blocks{};
    unsigned rows = 0;
};
struct Qpf2kSelectBatch {
    bool active = false;
    float* arena = nullptr;
    std::uint64_t capacity = 0, used = 0; // floats
    std::vector<qsa_flash::SelectRow> rows;
    std::vector<Qpf2kSelectGroup> traced;
    std::uint32_t ratio = 0, topk = 0;
};
Qpf2kSelectBatch& qpf2k_select_batch() { static Qpf2kSelectBatch b; return b; }
void qpf2k_select_flush(hipStream_t stream);
uint2* qpf2k_key_ranges(std::uint32_t capacity, bool allow_allocation_failure = false) {
    static uint2* ranges = nullptr;
    static std::uint32_t size = 0;
    if (capacity > size) {
        if (ranges) {
            check_hip(hipStreamSynchronize(hip_execution_stream()), "QSA prefill key-range resize sync");
            check_hip(hipFree(ranges), "QSA prefill key-range free");
            ranges = nullptr; size = 0;
        }
        void* p = nullptr;
        const auto status = hipMalloc(&p, std::size_t{capacity} * sizeof(uint2));
        if (allow_allocation_failure && status == hipErrorOutOfMemory) return nullptr;
        check_hip(status, "QSA prefill key-range allocation");
        ranges = static_cast<uint2*>(p);
        size = capacity;
    }
    return ranges;
}
// STRIX_QPF2K_QSA_TRACE=1 (diagnostic, independent of STRIX_QPF2K_QSA; synchronizes once per prefill call): hashes of every
// prefill group row's block scores before / after selection, its selected tokens and its attention output
// (qsa_flash::trace_kernel), cumulative per process, printed after every prefill call ("qpf2k qsa trace ...").
bool qpf2k_trace_enabled() {
    static const bool on = [] { const char* v = std::getenv("STRIX_QPF2K_QSA_TRACE"); return v && v[0] == '1'; }();
    return on;
}
struct Qpf2kTrace {
    unsigned long long* acc = nullptr; // [4]: scores, selected, post-selection scores, attention outputs
    std::uint32_t call = 0;
    std::uint64_t rows = 0;
};
// Execution witness (printed with the trace line): qpf2k / production scorer and selection launches of prefill groups.
std::atomic<std::uint64_t> qpf2k_launch_score{0}, qpf2k_launch_select{0}, qpf2k_launch_prod_score{0}, qpf2k_launch_prod_select{0};
std::atomic<std::uint64_t> qpf2k_launch_index16{0}, qpf2k_launch_norm_wave{0}, qpf2k_launch_select_batch{0}, qpf2k_select_batch_rows{0},
    qpf2k_select_early_flush{0};
std::array<std::atomic<std::uint64_t>, 4> qpf2k_groups_by_rows{};
std::atomic<std::uint64_t> qpf2k_launch_shadow{0}, qpf2k_shadow_skipped{0}, qpf2k_shadow_peak_bytes{0};
std::atomic<std::uint64_t> qpf2k_launch_order{0};
std::atomic<std::uint64_t> qpf2k_launch_index_batch{0}, qpf2k_index_batch_groups{0};
// STRIX_QPF2K_QSA bit 7 (0x80): the BF16 indexer projections that part 1 of one prefill call launches per 64-row group
// (qsa_index::rows_kernel<64, 4, 2, 16>) are collected and run as one qsa_index::rows_batch_kernel launch at the end of
// part 1 (before part 2, their only consumer). Each output is the same rows_body of the same group views; groups with
// other matrices (e.g. STRIX_QSA_SKIP_INDEX_Q) flush the batch first, so every launch has one matrix set.
// Exact decoder default; does not require STRIX_QSA_INDEX_ROWS or a QSA mask.
// Startup-cached like the other prefill switches; restart between engine A/B arms.
bool qpf2k_index_batch_enabled() {
    return qpf2k::index_batch_enabled();
}
// Bit-changing indexer arithmetic is governed by the shared public prefill controls.
bool qpf2k_index_wmma_enabled() {
    if (!prefill_option_enabled(PrefillOption::index_proj_wmma) ||
        qpf2k::suppressed(qpf2k::Item::index_rows)) return false;
    int device = 0;
    hipDeviceProp_t properties{};
    check_hip(hipGetDevice(&device), "QSA index WMMA device");
    check_hip(hipGetDeviceProperties(&properties, device), "QSA index WMMA architecture");
    const std::string_view arch(properties.gcnArchName);
    return (arch == "gfx1151" || arch.starts_with("gfx1151:")) && properties.warpSize == 32;
}
std::atomic<std::uint64_t> qpf2k_launch_index_wmma{0};
bool qpf2k_index_wmma_layout(const QsaRowsInputViews<64U>& v, std::uint32_t hidden) {
    if (hidden != 2560U || v.first[0] != 0U) return false;
    // Only the indexer q/k contract (including its skipped-query prefix).
    // Other BF16 projection groups keep exact arithmetic.
    if (!((v.count == 1U && v.first[1] == 128U) ||
          (v.count == 2U && v.first[1] == 512U && v.first[2] == 640U))) return false;
    for (unsigned m = 0U; m < v.count; ++m)
        if (reinterpret_cast<std::uintptr_t>(v.weights[m]) % 2U != 0U) return false;
    return true;
}
struct Qpf2kIndexBatch {
    bool active = false;
    bool wmma = false; // policy captured for this part-1 scope; false = exact batch
    std::uint32_t hidden = 0;
    std::uint64_t deferred = 0; // groups ever deferred (a caller compares it around a launch call)
    std::vector<QsaRowsInputViews<64U>> groups;
};
Qpf2kIndexBatch& qpf2k_index_batch() { static Qpf2kIndexBatch b; return b; }
unsigned qpf2k_qsa_mask();
// Own the production part-1 policy and clear pending views on normal or exceptional exit.
struct Qpf2kIndexScope {
    explicit Qpf2kIndexScope(unsigned mode, bool fast_qsa = true) {
        auto& ib = qpf2k_index_batch();
        ib.groups.clear();
        ib.wmma = mode == 0U && fast_qsa && qpf2k_index_wmma_enabled();
        ib.active = mode == 0U && (ib.wmma || qpf2k_index_batch_enabled() || (qpf2k_qsa_mask() & 0x80U));
    }
    ~Qpf2kIndexScope() {
        auto& ib = qpf2k_index_batch();
        ib.active = false; ib.wmma = false; ib.groups.clear();
    }
    Qpf2kIndexScope(const Qpf2kIndexScope&) = delete;
    Qpf2kIndexScope& operator=(const Qpf2kIndexScope&) = delete;
};
bool qpf2k_same_matrices(const QsaRowsInputViews<64U>& x, const QsaRowsInputViews<64U>& y) {
    if (x.count != y.count) return false;
    for (unsigned m = 0; m < x.count; ++m) if (x.weights[m] != y.weights[m]) return false;
    for (unsigned m = 0; m <= x.count; ++m) if (x.first[m] != y.first[m]) return false;
    return true;
}
void qpf2k_index_flush(hipStream_t stream) {
    auto& ib = qpf2k_index_batch();
    if (ib.groups.empty()) return;
    // Pinned host table slots (the GPU reads the views from them), reused after their launch completed.
    struct Slot { QsaRowsInputViews<64U>* views = nullptr; std::size_t capacity = 0; hipEvent_t done = nullptr; bool pending = false; };
    static std::array<Slot, 8> ring;
    static unsigned next = 0;
    auto& slot = ring[next];
    next = (next + 1U) % ring.size();
    if (slot.pending) check_hip(hipEventSynchronize(slot.done), "QSA index batch slot wait");
    if (!slot.done) check_hip(hipEventCreateWithFlags(&slot.done, hipEventDisableTiming), "QSA index batch event");
    if (ib.groups.size() > slot.capacity) {
        if (slot.views) check_hip(hipHostFree(slot.views), "QSA index batch slot free");
        slot.views = nullptr;
        slot.capacity = std::max<std::size_t>(ib.groups.size(), 128U);
        check_hip(hipHostMalloc(reinterpret_cast<void**>(&slot.views), slot.capacity * sizeof(QsaRowsInputViews<64U>), hipHostMallocDefault),
            "QSA index batch slot allocation");
    }
    std::copy(ib.groups.begin(), ib.groups.end(), slot.views);
    // STRIX_QPF2K_QSA_INDEX_SHAPE (launch shape of the batch; every output keeps its chains and tree, only which wave and
    // workgroup computes it changes): 0 = OW 2 W 16 (rows_kernel's), 1 = OW 4 W 16, 2 = OW 4 W 8, 3 = OW 2 W 8.
    static const unsigned shape = [] { const char* v = std::getenv("STRIX_QPF2K_QSA_INDEX_SHAPE"); return v ? static_cast<unsigned>(std::atoi(v)) & 3U : 0U; }();
    constexpr unsigned RB = 4U;
    const auto& v0 = ib.groups[0];
    const auto lds = ib.hidden * RB * static_cast<std::uint32_t>(sizeof(float));
    const auto launch = [&]<unsigned OW, unsigned W>() {
        const dim3 grid((v0.first[v0.count] + OW * W - 1U) / (OW * W), static_cast<std::uint32_t>(ib.groups.size()) * (64U / RB));
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::rows_batch_kernel<64U, RB, OW, W>), grid, dim3(W * 32U), lds, stream,
            static_cast<const QsaRowsInputViews<64U>*>(slot.views), ib.hidden);
    };
    if (ib.wmma && qpf2k_index_wmma_layout(v0, ib.hidden)) {
        hipLaunchKernelGGL(qsa_index::rows_wmma_kernel,
            dim3(v0.first[v0.count] / 64U, static_cast<unsigned>(ib.groups.size())), dim3(128), 0, stream,
            static_cast<const QsaRowsInputViews<64U>*>(slot.views), ib.hidden);
        qpf2k_launch_index_wmma.fetch_add(1U, std::memory_order_relaxed);
    }
    else if (qpf2k_index_batch_enabled() || ib.wmma) launch.template operator()<2U, 16U>();
    else if (shape == 1U) launch.template operator()<4U, 16U>();
    else if (shape == 2U) launch.template operator()<4U, 8U>();
    else if (shape == 3U) launch.template operator()<2U, 8U>();
    else launch.template operator()<2U, 16U>();
    check_hip(hipGetLastError(), "QSA index batch launch");
    check_hip(hipEventRecord(slot.done, stream), "QSA index batch slot record");
    slot.pending = true;
    qpf2k_launch_index_batch.fetch_add(1U, std::memory_order_relaxed);
    qpf2k_index_batch_groups.fetch_add(ib.groups.size(), std::memory_order_relaxed);
    ib.groups.clear();
}
// STRIX_QPF2K_QSA bit 8 (0x100, with bit 7): the batch launch runs on a side stream that forks from the execution
// stream where part 1 starts (fork, recorded before any part-1 launch: the row inputs are complete there) and joins it
// before part 2 (join), so the indexer projections overlap the dense q/k/v projections. Same launch, same views; the
// two branches share no buffers (indexer: index_query / current_index_key workspaces; dense: q_gate / current_k /
// current_v and their own slots; both only read the row inputs).
struct Qpf2kIndexSide {
    hipStream_t stream = nullptr;
    hipEvent_t fork = nullptr, join = nullptr;
};
Qpf2kIndexSide& qpf2k_index_side() {
    static Qpf2kIndexSide x = [] {
        Qpf2kIndexSide y;
        check_hip(hipStreamCreateWithFlags(&y.stream, hipStreamNonBlocking), "QSA index side stream");
        check_hip(hipEventCreateWithFlags(&y.fork, hipEventDisableTiming), "QSA index side fork event");
        check_hip(hipEventCreateWithFlags(&y.join, hipEventDisableTiming), "QSA index side join event");
        return y;
    }();
    return x;
}
std::atomic<std::uint64_t> qpf2k_launch_index_side{0};
Qpf2kTrace& qpf2k_trace() {
    static Qpf2kTrace t = [] {
        Qpf2kTrace x;
        void* p = nullptr;
        check_hip(hipMalloc(&p, 4U * sizeof(unsigned long long)), "QSA trace allocation");
        check_hip(hipMemset(p, 0, 4U * sizeof(unsigned long long)), "QSA trace clear");
        x.acc = static_cast<unsigned long long*>(p);
        return x;
    }();
    return t;
}
template <unsigned Rows, class Data, class Count>
void qpf2k_trace_rows(unsigned kind, const std::array<std::uint64_t, Rows>& position, Data data, Count count, hipStream_t stream, bool compact = false) {
    qsa_flash::TraceViews<Rows> v{};
    for (unsigned r = 0; r < Rows; ++r) { v.data[r] = reinterpret_cast<const std::uint32_t*>(data(r)); v.count[r] = count(r); v.position[r] = position[r]; v.compact[r] = compact; }
    auto& t = qpf2k_trace();
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::trace_kernel<Rows>), dim3(Rows), dim3(256), 0, stream, v, kind, t.call, t.acc);
    check_hip(hipGetLastError(), "QSA trace launch");
}
unsigned qpf2k_qsa_mask() {
    static const unsigned mask = [] {
        const char* v = std::getenv("STRIX_QPF2K_QSA");
        if (!v || v[0] != '1') return 0U;
        const char* m = std::getenv("STRIX_QPF2K_QSA_MASK");
        unsigned bits = m ? static_cast<unsigned>(std::strtoul(m, nullptr, 0)) : 0x3U; // default: the qualified parts (scorer, selection)
        // Bit 5 (global FP16 KV shadow attend) is declined: at 32K its logits differed from the default attend and
        // between runs, so it is not an exact variant. The default attend dispatch runs instead.
        if (bits & 32U) {
            std::fprintf(stderr, "qpf2k qsa mask bit 5 (FP16 KV shadow attend) declined: not byte-exact; default attend runs\n");
            bits &= ~32U;
        }
        return bits;
    }();
    return qpf2k::suppressed(qpf2k::Item::qsa) ? 0U : mask;
}
// STRIX_QSA_INDEX_ROWS=1: BF16 input matrices of prefill groups (the indexer q/k projections) run
// qsa_index::rows_kernel (the same chains and tree as linear_to_float_rows_kernel, all rows of the group per launch).
bool qsa_index_rows_enabled() {
    static const bool enabled = [] { const char* v = std::getenv("STRIX_QSA_INDEX_ROWS"); return v && v[0] == '1'; }();
    return !qpf2k::suppressed(qpf2k::Item::index_rows) && enabled;
}
// STRIX_QSA_NORM_GROUP=1: the three norm/RoPE launches of a prefill group as rope_table_kernel +
// norm_rope_group_kernel (same values).
bool qsa_norm_group_enabled() {
    static const bool enabled = [] { const char* v = std::getenv("STRIX_QSA_NORM_GROUP"); return v && v[0] == '1'; }();
    return !qpf2k::suppressed(qpf2k::Item::norm_group) && enabled;
}
// ggml type of a weight kind (GSQ dense A8 typed segments).
std::uint32_t kind_ggml_type(WeightKind kind) {
    return kind == WeightKind::q8_0 ? 8U : kind == WeightKind::iq4_xs ? 23U : ggml_type_of(kind);
}
// typed_rows8: rows > 8 of a GSQ typed kind under dense A8 run as 8-row typed A8 sites (cohort windows: every row
// gets its 1..8-row bits); otherwise (prefill groups) they keep the Slice-row FP32 kernels below.
template <WeightKind Kind, unsigned Rows>
void launch_qsa_rows_input(bool prefetch, bool a8, const QsaRowsInputViews<Rows>& views, std::uint32_t hidden, hipStream_t stream,
    bool any_order = false, bool typed_rows8 = false) {
    // GSQ-RCO dense types under dense A8: the typed q8_rows site (the R1 dense A8 bits).
    if constexpr (Kind != WeightKind::q8_0 && Rows <= 8U) if (a8 && q8_rows::dense_a8_type(kind_ggml_type(Kind))) {
        QsaInputRowsSite<Rows> site{};
        for (unsigned m = 0; m < views.count; ++m) {
            site.segments.weights[m] = static_cast<const unsigned char*>(views.weights[m]);
            site.segments.first[m + 1U] = views.first[m + 1U];
            site.segments.types[m] = kind_ggml_type(Kind);
            for (unsigned r = 0; r < Rows; ++r) site.output[m][r] = views.output[m][r];
        }
        site.segments.count = views.count;
        for (unsigned r = 0; r < Rows; ++r) site.input[r] = views.input[r];
        site.width = hidden; site.outputs = views.first[views.count];
        q8_rows::launch(site, true, stream, any_order);
        return;
    }
    if constexpr (Rows > 8U) if (Kind != WeightKind::q8_0 || !(prefetch || a8)) {
        if constexpr (Kind == WeightKind::bf16 && Rows >= 64U && Rows % 16U == 0U) if ((qsa_index_rows_enabled() || (qpf2k_index_batch().active && (qpf2k_index_batch_enabled() || qpf2k_index_batch().wmma))) && hidden % 256U == 0U && hidden <= 4096U) {
            bool aligned = true;
            for (unsigned r = 0; r < Rows; ++r) aligned = aligned && reinterpret_cast<std::uintptr_t>(views.input[r]) % 16U == 0U;
            // STRIX_QPF2K_QSA bit 3: 16 rows per workgroup (qsa_flash::index_rows_kernel, the same chains and tree).
            if (aligned && !(qpf2k_index_batch().active && (qpf2k_index_batch_enabled() || qpf2k_index_batch().wmma)) && (qpf2k_qsa_mask() & 8U) && hidden % 512U == 0U) {
                constexpr unsigned RB = 16U, W = 16U;
                const dim3 grid((views.first[views.count] + W - 1U) / W, Rows / RB);
                if (any_order)
                    hipExtLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::index_rows_kernel<Rows, RB, W>), grid, dim3(W * 32U), 0, stream,
                        nullptr, nullptr, hipExtAnyOrderLaunch, views, hidden);
                else hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::index_rows_kernel<Rows, RB, W>), grid, dim3(W * 32U), 0, stream, views, hidden);
                qpf2k_launch_index16.fetch_add(1U, std::memory_order_relaxed);
                check_hip(hipGetLastError(), "QSA prefill indexer rows16 launch");
                return;
            }
            if constexpr (Rows == 64U) if (auto& ib = qpf2k_index_batch(); aligned && ib.active) {
                // STRIX_QPF2K_QSA bit 7: deferred to qpf2k_index_flush (the same rows_kernel<64, 4, 2, 16> body).
                if (!ib.groups.empty() && (ib.hidden != hidden || !qpf2k_same_matrices(ib.groups[0], views))) qpf2k_index_flush(stream);
                ib.hidden = hidden;
                ib.groups.push_back(views);
                ++ib.deferred;
                return;
            }
            if (aligned) {
                constexpr unsigned RB = 4U, OW = 2U, W = 16U;
                const dim3 grid((views.first[views.count] + OW * W - 1U) / (OW * W), Rows / RB);
                const auto lds = hidden * RB * static_cast<std::uint32_t>(sizeof(float));
                if (any_order)
                    hipExtLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::rows_kernel<Rows, RB, OW, W>), grid, dim3(W * 32U), lds, stream,
                        nullptr, nullptr, hipExtAnyOrderLaunch, views, hidden);
                else hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::rows_kernel<Rows, RB, OW, W>), grid, dim3(W * 32U), lds, stream, views, hidden);
                check_hip(hipGetLastError(), "QSA prefill indexer rows launch");
                return;
            }
        }
        // Prefill blocks: only the prefetch8 Q8 site has a wide form; other kinds run Slice-row
        // launches of linear_to_float_rows_kernel (per row the same feature order and tree as
        // the 8-row window kernel). STRIX_QSA_INPUT_SLICE=8|16|32 (default 16).
        static const unsigned slice_rows = [] {
            const char* v = std::getenv("STRIX_QSA_INPUT_SLICE");
            const int n = v ? std::atoi(v) : 16;
            return n == 8 || n == 32 ? static_cast<unsigned>(n) : 16U;
        }();
        const auto run = [&]<unsigned Slice>() {
            for (unsigned first = 0; first < Rows; first += Slice) {
                QsaRowsInputViews<Slice> slice{};
                for (unsigned m = 0; m < 5U; ++m) slice.weights[m] = views.weights[m];
                for (unsigned m = 0; m < 6U; ++m) slice.first[m] = views.first[m];
                slice.count = views.count;
                for (unsigned r = 0; r < Slice; ++r) {
                    slice.input[r] = views.input[first + r];
                    for (unsigned m = 0; m < 5U; ++m) slice.output[m][r] = views.output[m][first + r];
                }
                if constexpr (Slice == 8U) launch_qsa_rows_input<Kind, 8U>(prefetch, a8, slice, hidden, stream, any_order);
                else if (any_order)
                    hipExtLaunchKernelGGL(HIP_KERNEL_NAME(linear_to_float_rows_kernel<Kind, Slice>), dim3(slice.first[slice.count]),
                        dim3(kThreads), 0, stream, nullptr, nullptr, hipExtAnyOrderLaunch, slice, hidden);
                else hipLaunchKernelGGL(HIP_KERNEL_NAME(linear_to_float_rows_kernel<Kind, Slice>), dim3(slice.first[slice.count]),
                    dim3(kThreads), 0, stream, slice, hidden);
            }
        };
        if (slice_rows == 8U || Rows % slice_rows || (typed_rows8 && a8 && q8_rows::dense_a8_type(kind_ggml_type(Kind))))
            run.template operator()<8U>();
        else if (slice_rows == 16U) run.template operator()<16U>();
        else if constexpr (Rows % 32U == 0U) run.template operator()<32U>();
        else run.template operator()<16U>();
        return;
    }
    if constexpr (Kind == WeightKind::q8_0) if (prefetch || a8) {
        QsaInputRowsSite<Rows> site{};
        for (unsigned m = 0; m < views.count; ++m) {
            site.segments.weights[m] = static_cast<const unsigned char*>(views.weights[m]);
            site.segments.first[m + 1U] = views.first[m + 1U];
            for (unsigned r = 0; r < Rows; ++r) site.output[m][r] = views.output[m][r];
        }
        site.segments.count = views.count;
        for (unsigned r = 0; r < Rows; ++r) site.input[r] = views.input[r];
        site.width = hidden; site.outputs = views.first[views.count];
        q8_rows::launch(site, a8, stream, any_order);
        return;
    }
    if (any_order)
        hipExtLaunchKernelGGL(HIP_KERNEL_NAME(linear_to_float_rows_kernel<Kind, Rows>), dim3(views.first[views.count]), dim3(kThreads), 0,
            stream, nullptr, nullptr, hipExtAnyOrderLaunch, views, hidden);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(linear_to_float_rows_kernel<Kind, Rows>), dim3(views.first[views.count]), dim3(kThreads), 0,
        stream, views, hidden);
}
template <WeightKind Kind, unsigned Rows, class InputType = float>
void launch_qsa_rows_output_storage(bool wave8, bool a8, const QsaRowsOutputViews<Rows>& views, const void* weights,
    std::uint32_t width, std::uint32_t hidden, hipStream_t stream, bool f16 = false, bool g64 = false) {
    if constexpr (Kind != WeightKind::q8_0 && Rows <= 8U) if (a8 && q8_rows::dense_a8_type(kind_ggml_type(Kind))) {
        QsaOutputRowsSite<Rows, InputType> site{};
        site.segments = q8_rows::segments({{weights, hidden}});
        site.segments.types[0] = kind_ggml_type(Kind);
        for (unsigned r = 0; r < Rows; ++r) { site.input[r] = views.input[r]; site.output[r] = views.output[r]; }
        site.width = width; site.outputs = hidden;
        q8_rows::launch(site, true, stream);
        return;
    }
    if constexpr (Rows > 8U) if (Kind != WeightKind::q8_0 || !(wave8 || a8)) {
        for (unsigned first = 0; first < Rows; first += 8U) {
            QsaRowsOutputViews<8U> slice{};
            slice.compact = views.compact;
            for (unsigned r = 0; r < 8U; ++r) { slice.input[r] = views.input[first + r]; slice.output[r] = views.output[first + r]; }
            launch_qsa_rows_output_storage<Kind, 8U, InputType>(wave8, a8, slice, weights, width, hidden, stream);
        }
        return;
    }
    if constexpr (Kind == WeightKind::q8_0) if (wave8 || a8) {
        QsaOutputRowsSite<Rows, InputType> site{};
        site.segments = q8_rows::segments({{weights, hidden}});
        for (unsigned r = 0; r < Rows; ++r) { site.input[r] = views.input[r]; site.output[r] = views.output[r]; }
        site.width = width; site.outputs = hidden;
        q8_rows::launch_prefill(site, a8, f16, stream, false, g64);
        return;
    }
    hipLaunchKernelGGL(HIP_KERNEL_NAME(linear_float_to_bf16_rows_kernel<Kind, Rows>), dim3(hidden), dim3(kThreads), 0,
        stream, views, weights, width, hidden);
}
template <WeightKind Kind, unsigned Rows>
void launch_qsa_rows_output(bool wave8, bool a8, const QsaRowsOutputViews<Rows>& views, const void* weights,
    std::uint32_t width, std::uint32_t hidden, hipStream_t stream, bool f16 = false, bool g64 = false) {
    if (views.compact) launch_qsa_rows_output_storage<Kind, Rows, std::uint16_t>(wave8,a8,views,weights,width,hidden,stream,f16,g64);
    else launch_qsa_rows_output_storage<Kind, Rows, float>(wave8,a8,views,weights,width,hidden,stream,f16,g64);
}

template <class Launch>
void dispatch_weight_kind(std::uint32_t type, Launch&& launch) {
    switch (weight_kind(type)) {
    case WeightKind::f32: launch(std::integral_constant<WeightKind, WeightKind::f32>{}); return;
    case WeightKind::f16: launch(std::integral_constant<WeightKind, WeightKind::f16>{}); return;
    case WeightKind::bf16: launch(std::integral_constant<WeightKind, WeightKind::bf16>{}); return;
    case WeightKind::q8_0: launch(std::integral_constant<WeightKind, WeightKind::q8_0>{}); return;
    case WeightKind::iq4_xs: launch(std::integral_constant<WeightKind, WeightKind::iq4_xs>{}); return;
#define STRIX_GGML_KIND_CASE(KIND, type) case WeightKind::KIND: launch(std::integral_constant<WeightKind, WeightKind::KIND>{}); return;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_CASE) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_CASE) STRIX_GGML_KIND_CASE(iq4_nl, 20)
#undef STRIX_GGML_KIND_CASE
    }
}
template <class Launch>
void dispatch_small_kind(std::uint32_t type, Launch&& launch) {
    switch (small_weight_kind(type)) {
    case SmallWeightKind::f32: launch(std::integral_constant<SmallWeightKind, SmallWeightKind::f32>{}); return;
    case SmallWeightKind::f16: launch(std::integral_constant<SmallWeightKind, SmallWeightKind::f16>{}); return;
    case SmallWeightKind::bf16: launch(std::integral_constant<SmallWeightKind, SmallWeightKind::bf16>{}); return;
    }
}

#include "hip_qsa_fast_prefill.inc"
#include "hip_qsa_fast_prefill_shadow.inc"
// Fast prefill profile, qsa component (hip_qsa_fast_prefill.inc): union, attend and combine kernels for
// one prefill group. STRIX_QSA_FAST_TILE=4|8|16 rows per tile (default 4); STRIX_QSA_FAST_SPLITS=1..8
// union slices per (tile, KV head, head group) (default: enough workgroups for 32; one slice needs no
// combine kernel). Returns false
// (the caller keeps the exact kernels) for a shape or scratch the fast kernels do not cover.
// STRIX_QSA_ATTEND_BATCH=1: the 64-row groups of one prefill call queue their union/attend views (default
// 4-row tiles, one slice) while active, and qsa_fast_batch_flush launches them in one union and one attend
// launch (block x = group * 16 + tile), the same per-tile work as the per-group launches. Views go through a
// ring of pinned host slots read in place; a slot is reused only after its kernels completed (event).
struct QsaFastBatch {
    std::vector<qsa_fast::Views<64U>> groups;
    const std::uint16_t* kv = nullptr;
    std::uint32_t kv_heads = 0, heads = 0, union_at = 0;
    bool active = false;
};
QsaFastBatch& qsa_fast_batch() { static QsaFastBatch batch; return batch; }
bool qsa_attend_batch_enabled() {
    static const bool enabled = [] { const char* v = std::getenv("STRIX_QSA_ATTEND_BATCH"); return v && v[0] == '1'; }();
    return !qpf2k::suppressed(qpf2k::Item::attend_batch) && enabled;
}
// STRIX_QSA_UNION_MERGE (default on; =0 or the baseline keeps the range form): the attend union of a row tile
// is built by merging the rows' ascending selections (qsa_fast::union_merge), so its cost no longer grows with
// the visible block range. Same (block, mask) pairs, order and token counts.
bool qsa_union_merge_enabled() { return qpf2k::long_context_item_enabled("STRIX_QSA_UNION_MERGE"); }
// STRIX_QSA_FAST_STATS=1 (diagnostic): counts of the fast prefill / fused tail launch kinds, printed on stderr at exit
// ("qsa launch stats ..."), so gates can show which path ran (e.g. kv-int8 fast attend, no int8 group in the attend batch).
struct QsaLaunchStats {
    std::atomic<std::uint64_t> fast_f16{0}, fast_int8{0}, batch_groups{0}, batch_launches{0}, tail_bf16{0}, tail_int8{0};
    ~QsaLaunchStats() {
        std::fprintf(stderr, "qsa launch stats fast_f16 %llu fast_int8 %llu batch_groups %llu batch_launches %llu tail_bf16 %llu tail_int8 %llu\n",
            static_cast<unsigned long long>(fast_f16.load()), static_cast<unsigned long long>(fast_int8.load()),
            static_cast<unsigned long long>(batch_groups.load()), static_cast<unsigned long long>(batch_launches.load()),
            static_cast<unsigned long long>(tail_bf16.load()), static_cast<unsigned long long>(tail_int8.load()));
    }
};
QsaLaunchStats* qsa_launch_stats() {
    static const bool enabled = [] { const char* v = std::getenv("STRIX_QSA_FAST_STATS"); return v && v[0] == '1'; }();
    if (!enabled) return nullptr;
    static QsaLaunchStats stats;   // printed by its destructor at exit
    return &stats;
}
void count_qsa_launch(std::atomic<std::uint64_t> QsaLaunchStats::* kind) {
    if (auto* stats = qsa_launch_stats()) (stats->*kind).fetch_add(1U, std::memory_order_relaxed);
}
void qsa_fast_batch_flush(hipStream_t stream) {
    auto& b = qsa_fast_batch();
    if (b.groups.empty()) return;
    struct Slot { qsa_fast::Views<64U>* views = nullptr; std::size_t capacity = 0; hipEvent_t done = nullptr; bool pending = false; };
    constexpr unsigned kSlots = 16U;
    static std::array<Slot, kSlots> ring;
    static unsigned next = 0;
    auto& slot = ring[next];
    next = (next + 1U) % kSlots;
    if (slot.pending) check_hip(hipEventSynchronize(slot.done), "QSA attend batch slot wait");
    if (!slot.done) check_hip(hipEventCreateWithFlags(&slot.done, hipEventDisableTiming), "QSA attend batch event");
    if (b.groups.size() > slot.capacity) {
        if (slot.views) check_hip(hipHostFree(slot.views), "QSA attend batch slot free");
        slot.views = nullptr;
        const auto capacity = std::max<std::size_t>(b.groups.size(), 128U);
        check_hip(hipHostMalloc(reinterpret_cast<void**>(&slot.views), capacity * sizeof(qsa_fast::Views<64U>), hipHostMallocDefault),
            "QSA attend batch slot allocation");
        slot.capacity = capacity;
    }
    std::copy(b.groups.begin(), b.groups.end(), slot.views);
    count_qsa_launch(&QsaLaunchStats::batch_launches);
    if (auto* stats = qsa_launch_stats()) stats->batch_groups.fetch_add(b.groups.size(), std::memory_order_relaxed);
    const auto n = static_cast<std::uint32_t>(b.groups.size());
    constexpr unsigned TR = 4U, G = 3U, HG = G * (16U / TR);
    // The merge-capable kernel only where some tile can reach the merge threshold: below it that kernel runs the range
    // form with the merge form's larger LDS footprint (lower occupancy).
    const bool merge_reachable = [&] {
        for (const auto& g : b.groups)
            for (const auto visible : g.visible) if ((visible + 3U) / 4U > qsa_fast::kMergeMinBlocks) return true;
        return false;
    }();
    if (merge_reachable && qsa_union_merge_enabled())
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::union_multi_kernel<64U, TR, true>), dim3(n * (64U / TR)), dim3(256), 0, stream,
            slot.views, b.union_at);
    else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::union_multi_kernel<64U, TR, false>), dim3(n * (64U / TR)), dim3(256), 0, stream,
            slot.views, b.union_at);
    // STRIX_QPF2K_QSA bit 5 (declined by qpf2k_qsa_mask, so not reached; kept for investigation): the batch's KV prefix (tokens [0, max visible)) converted once to FP16 with the loaders'
    // bf16x2_to_f16x2 (same bits), then the attend without per-chunk conversion (qsa_fast::attend_shadow_multi_kernel).
    // Memory: one layer's prefix, 4 x kv_heads x 256 bytes per token (2 KiB with 2 KV heads: 64 MiB at 32K, 256 MiB at
    // 128K), stream-ordered (hipMallocAsync before the conversion, hipFreeAsync after the attend): it lives only while
    // that attend runs, never into decode, and the BF16 cache itself is unchanged (decode never reads the shadow).
    // Length-gated so it never costs context length: only for batches of at most STRIX_QPF2K_SHADOW_MAX_TOKENS tokens
    // (default 131072) and only if at least 2 GiB of device memory stay free; otherwise the BF16 attend runs.
    void* shadow = nullptr;
    if (qpf2k_qsa_mask() & 32U) {
        static const std::uint64_t cap = [] { const char* v = std::getenv("STRIX_QPF2K_SHADOW_MAX_TOKENS");
            return v ? std::strtoull(v, nullptr, 0) : std::uint64_t{131072}; }();
        std::uint64_t tokens = 0;
        for (const auto& g : b.groups) for (unsigned r = 0; r < 64U; ++r) tokens = std::max<std::uint64_t>(tokens, g.visible[r]);
        const std::uint64_t words = tokens * b.kv_heads * qsa_fast::kDim; // u32 words: [K | V] heads x 256 BF16 per token
        // Earlier launches are checked first, so that only this block's own recoverable probe / allocation failure is
        // cleared below (hipGetLastError resets the sticky error).
        check_hip(hipGetLastError(), "QSA attend batch union launch");
        bool fits = false;
        if (tokens && tokens <= cap) {
            std::size_t free_bytes = 0, total_bytes = 0;
            if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess)
                fits = static_cast<std::uint64_t>(free_bytes) >= words * sizeof(unsigned) + (std::uint64_t{2} << 30U);
            else (void)hipGetLastError(); // the memory query failed: no shadow
        }
        if (fits) {
            if (hipMallocAsync(&shadow, words * sizeof(unsigned), stream) != hipSuccess) { (void)hipGetLastError(); shadow = nullptr; }
        }
        if (shadow) {
            hipLaunchKernelGGL(qsa_fast::kv_to_f16_kernel, dim3(static_cast<std::uint32_t>((words / 4U + 255U) / 256U + 1U)), dim3(256), 0, stream,
                reinterpret_cast<const unsigned*>(b.kv), static_cast<unsigned*>(shadow), 0ULL, words);
            qpf2k_shadow_peak_bytes.store(std::max<std::uint64_t>(qpf2k_shadow_peak_bytes.load(), words * sizeof(unsigned)));
        } else {
            qpf2k_shadow_skipped.fetch_add(1U, std::memory_order_relaxed);
        }
    }
    if (shadow) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_shadow_multi_kernel<64U, TR, G>), dim3(n * (64U / TR), b.kv_heads * (b.heads / b.kv_heads / HG), 1U),
            dim3((2U * G + 4U) * 32U), 0, stream, slot.views, static_cast<const std::uint16_t*>(shadow), b.kv_heads, b.heads, b.union_at);
        check_hip(hipFreeAsync(shadow, stream), "QSA attend shadow free");
        qpf2k_launch_shadow.fetch_add(1U, std::memory_order_relaxed);
    } else if (qpf2k_qsa_mask() & 64U) {
        // STRIX_QPF2K_QSA bit 6 (candidate I): the same attend body under a one-dimensional group-major grid
        // (qsa_fast::attend_order_kernel): the resident workgroups cover few row groups, whose unions overlap, so the
        // gathered KV blocks stay in the last-level cache. Scheduling only: every workgroup runs the same (group, tile, y) body.
        const std::uint32_t ys = b.kv_heads * (b.heads / b.kv_heads / HG);
        if (qpf2k_qsa_mask() & 0x2000U) {
            // Bit 13: the same grid and body with an eight-waves-per-SIMD register budget (qsa_fast::attend_order_occ_kernel).
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_order_occ_kernel<64U, TR, G>), dim3(n * (64U / TR) * ys),
                dim3((2U * G + 4U) * 32U), 0, stream, slot.views, b.kv, b.kv_heads, b.heads, b.union_at, ys);
            static const bool witnessed = [] {
                std::fprintf(stderr, "qpf2k qsa attend occ active mask 0x2000 waves_per_eu 8\n");
                return true;
            }();
            (void)witnessed;
        } else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_order_kernel<64U, TR, G>), dim3(n * (64U / TR) * ys), dim3((2U * G + 4U) * 32U), 0,
            stream, slot.views, b.kv, b.kv_heads, b.heads, b.union_at, ys);
        qpf2k_launch_order.fetch_add(1U, std::memory_order_relaxed);
    } else
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_multi_kernel<64U, TR, G>), dim3(n * (64U / TR), b.kv_heads * (b.heads / b.kv_heads / HG), 1U),
        dim3((2U * G + 4U) * 32U), 0, stream, slot.views, b.kv, b.kv_heads, b.heads, b.union_at);
    check_hip(hipGetLastError(), "QSA attend batch launch");
    check_hip(hipEventRecord(slot.done, stream), "QSA attend batch slot record");
    slot.pending = true;
    b.groups.clear();
}

template <unsigned Rows, bool Int8 = false>
bool launch_qsa_fast_prefill(const QsaKnownRows<Rows>& prepared, const PreparedIndexedLaunch& a, std::uint32_t maximum_selected,
    hipStream_t stream) {
    static const unsigned tile = [] { const char* v = std::getenv("STRIX_QSA_FAST_TILE"); const int n = v ? std::atoi(v) : 4;
        return n == 8 || n == 16 ? static_cast<unsigned>(n) : 4U; }();
    static const unsigned forced_splits = [] { const char* v = std::getenv("STRIX_QSA_FAST_SPLITS"); const int n = v ? std::atoi(v) : 0;
        return n >= 1 && n <= 8 ? static_cast<unsigned>(n) : 0U; }();
    // STRIX_QPF2K_QSA bit 4: any path below except the attend-batch push reads the selections now (or lets the exact
    // kernels read them), so deferred selections are launched first.
    const auto need_selections = [&] { qpf2k_select_flush(stream); };
    if (a.head_dim != qsa_fast::kDim || !a.kv_heads || a.attention_heads % a.kv_heads) { need_selections(); return false; }
    const auto heads = a.attention_heads, per_kv = heads / a.kv_heads;
    qsa_fast::Views<Rows> views{};
    for (unsigned r = 0; r < Rows; ++r) {
        const auto& w = prepared.row[r].workspace;
        views.q_gate[r] = w.q_gate; views.selected[r] = w.selected_indices; views.scratch[r] = w.attention_scores;
        views.output[r] = w.attention_output; views.compact_output = w.compact_attention_output;
        views.count[r] = prepared.selection[r].selected_count; views.visible[r] = prepared.selection[r].visible;
        if (!views.count[r] || views.visible[r] > (1U << 30U)) { need_selections(); return false; }
    }
    const auto capacity = static_cast<std::uint64_t>(heads) * maximum_selected;
    const auto run = [&]<unsigned TR, unsigned G>() {
        constexpr unsigned HG = G * (16U / TR);
        if (per_kv % HG) { need_selections(); return false; }
        const unsigned groups = Rows / TR * a.kv_heads * (per_kv / HG);
        unsigned splits = forced_splits ? forced_splits : std::clamp((32U + groups - 1U) / groups, 1U, 8U);
        while (splits > 1U && qsa_fast::scratch_floats(TR, splits, heads, maximum_selected) > capacity) --splits;
        if (qsa_fast::scratch_floats(TR, splits, heads, maximum_selected) > capacity) { need_selections(); return false; }
        const auto union_at = splits * heads * qsa_fast::kPartial;
        if constexpr (Rows == 64U && !Int8) {
            auto& b = qsa_fast_batch();
            if (b.active && splits == 1U && TR == 4U && G == 3U) {
                const auto* kv = reinterpret_cast<const std::uint16_t*>(a.kv_state.address);
                if (!b.groups.empty() && (b.kv != kv || b.kv_heads != a.kv_heads || b.heads != heads || b.union_at != union_at))
                {
                    need_selections(); // the flushed groups' selections precede their union / attend
                    qsa_fast_batch_flush(stream);
                }
                b.kv = kv; b.kv_heads = a.kv_heads; b.heads = heads; b.union_at = union_at;
                b.groups.push_back(views);
                return true;
            }
        }
        need_selections();
        const bool merge_reachable = [&] {
            for (unsigned r = 0; r < Rows; ++r) if ((views.visible[r] + 3U) / 4U > qsa_fast::kMergeMinBlocks) return true;
            return false;
        }();
        if (merge_reachable && qsa_union_merge_enabled())
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::union_kernel<Rows, TR, true>), dim3(Rows / TR), dim3(256), 0, stream, views, union_at);
        else
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::union_kernel<Rows, TR, false>), dim3(Rows / TR), dim3(256), 0, stream, views, union_at);
        count_qsa_launch(Int8 ? &QsaLaunchStats::fast_int8 : &QsaLaunchStats::fast_f16);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_kernel<Rows, TR, G, 3U, Int8>), dim3(Rows / TR, a.kv_heads * (per_kv / HG), splits),
            dim3((2U * G + 4U) * 32U), 0, stream, views, reinterpret_cast<const std::uint16_t*>(a.kv_state.address), a.kv_heads, heads, union_at, splits, 0U);
        if (splits > 1U)
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::combine_kernel<Rows>), dim3(heads, Rows), dim3(qsa_fast::kDim), 0, stream, views, heads, splits);
        check_hip(hipGetLastError(), "QSA fast prefill launch");
        return true;
    };
    if (tile == 16U) return run.template operator()<16U, 12U>();
    if (tile == 8U) return run.template operator()<8U, 6U>();
    return run.template operator()<4U, 3U>();
}

// Pooled block keys of one prefill call (block_pool_kernel): blocks [0, valid) are formed from the
// raw keys this call's groups and the earlier context left; keys has room for capacity blocks.
// The serial decode owner instead uses packed alone when its admitted BF16 storage policy is enabled.

void bind_qwen_pool(QsaPrefillPool* pool, const ModelContract& model) {
    if (!pool) return;
    if (pool->valid && pool->rotary_table != model.qwen_rotary_table)
        throw std::invalid_argument("QSA pool rotary source mismatch");
    pool->rotary_table = model.qwen_rotary_table;
}
bool qsa_pool_packed_enabled() {
    static const bool enabled = [] { const char* v = std::getenv("STRIX_QSA_POOL_PACKED"); return v && v[0] == '1'; }();
    return !qpf2k::baseline && enabled;
}

// RoPE table of the grouped norm (up to 1024 rows x 64 pairs); each launch writes it before its norm reads it in stream
// order. enqueue_qsa_prefill_rows allocates it on its first call: the cache-only step is often the first grouped-norm
// user, and a hipMalloc there would stall the deep launch queue of a long prefill.
float2* qsa_group_rope_table() {
    static float2* const table = [] {
        void* p = nullptr;
        check_hip(hipMalloc(&p, std::size_t{1024U} * 64U * sizeof(float2)), "QSA prefill RoPE table allocation");
        return static_cast<float2*>(p);
    }();
    return table;
}

// parts: 1 input projections, 2 norm/store/selection/attention, 4 output projection. Rows > 8
// (prefill groups of 8-row tiles) store every row's index key and K/V before any row selects
// or attends; each row reads only positions at or before its own, so this equals the serial
// tile order (see enqueue_qsa_prefill_rows).
// cohort_slot: a cohort window's dense A8 output slot. Part 2 of one session's rows writes the fused
// output codes at rows [cohort_row, cohort_row + Rows) of it and part 4 of all rows reads them, as
// one window's part 2 + 4 do within one call (the slot layout does not depend on the row count).
// subparts (part 2): 1 norms/rope, 2 KV/raw-index append, 4 block scores, 8 selection, 16 attention scores,
// 32 softmax, 64 value/gate output. A cohort runs the state-free ones (1, 8, 32) once for the rows of every
// session and the others per session, in this order.
template <unsigned Rows>
void enqueue_qsa_known_rows(const QsaKnownRows<Rows>& prepared, const HipIndexedAttentionLauncherConfig& config,
    unsigned parts = 7U, QsaPrefillPool* pool = nullptr, unsigned skip_inputs = 0U,
    const q8_rows::DenseA8Slot* cohort_slot = nullptr, unsigned cohort_row = 0U, unsigned subparts = 127U,
    const qsa_flash::ScoreGroup<Rows>* staged = nullptr, bool verify_window = false, bool cache_only = false) {
    for(const auto& row:prepared.row)
        require_qsa_score_source(!row.workspace_layout.block_scores_bytes,!(parts & 2U) || cache_only || row.workspace.block_scores!=nullptr);
    if (staged) subparts &= ~7U; // norms, append and scores already completed in stream order
    const auto& model = config.model;
    bind_qwen_pool(pool, model);
    const auto& a = prepared.row[0];
    const auto& tensors = a.tensors;
    // STRIX_QPF2K_QSA_TRACE witness: prefill groups by size (attention part; 64 / 32 / 16 / 8 rows).
    if constexpr (Rows == 64U || Rows == 32U || Rows == 16U || Rows == 8U)
        if (!cohort_slot && (parts & 2U) && qpf2k_trace_enabled())
            qpf2k_groups_by_rows[Rows == 64U ? 0 : Rows == 32U ? 1 : Rows == 16U ? 2 : 3].fetch_add(1U, std::memory_order_relaxed);
    const auto stream = hip_execution_stream();
    const auto rotary = checked_u32(model.rotary_dim(), "rotary_dim");
    const auto index_heads = checked_u32(model.indexer_n_heads, "indexer_n_heads");

    const std::array<const TensorPair*, 5> matrices{&tensors.index_q, &tensors.index_k, &tensors.q, &tensors.k, &tensors.v};
    const std::array<std::uint32_t, 5> outputs{a.index_q_width, a.index_dim, a.q_gate_width, a.kv_width, a.kv_width};
    const std::array<float* WorkspacePointers::*, 5> targets{&WorkspacePointers::index_query, &WorkspacePointers::current_index_key,
        &WorkspacePointers::q_gate, &WorkspacePointers::current_k, &WorkspacePointers::current_v};
    // skip_inputs: input matrices (bit m = matrices[m]) already projected by the caller.
    std::array<bool, 5> done{};
    for (unsigned m = 0; m < 5U; ++m) done[m] = (skip_inputs >> m) & 1U;
    // STRIX_QSA_SKIP_INDEX_Q=1: the index query feeds only the block scores, which no row computes
    // when every row's complete blocks fit block_topk (scores.blocks below are all 0).
    bool skip_index_q = qsa_skip_index_q_enabled();
    for (unsigned r = 0; skip_index_q && r < Rows; ++r)
        skip_index_q = prepared.selection[r].complete_blocks <= prepared.selection[r].block_topk;
    if (skip_index_q) done[0] = true;
    q8_rows::DenseA8Slot output_slot{};
    // Any-order (STRIX_ANYORDER=0 disables): input groups after the first read the same rows and write other
    // targets; the q and k norms follow a barrier-ordered index norm and touch only their own.
    static const bool any_order = strix_any_order_enabled();
    bool launched_group = false;
    if (parts & 1U) for (unsigned lead = 0; lead < 5U; ++lead) {
        if (done[lead]) continue;
        // Group every matrix sharing the lead's serial organization (format + prefetch8).
        QsaRowsInputViews<Rows> input{};
        for (unsigned r = 0; r < Rows; ++r) input.input[r] = reinterpret_cast<const std::uint16_t*>(prepared.row[r].input.address);
        for (unsigned m = lead; m < 5U; ++m) {
            const bool dense_q8 = config.dense_a8_projections && matrices[lead]->manifest->ggml_type == 8U;
            if (done[m] || (!dense_q8 && prepared.prefetch[m] != prepared.prefetch[lead]) ||
                matrices[m]->manifest->ggml_type != matrices[lead]->manifest->ggml_type) continue;
            done[m] = true;
            const auto g = input.count++;
            input.weights[g] = reinterpret_cast<const void*>(matrices[m]->runtime->address);
            input.first[g + 1U] = input.first[g] + outputs[m];
            for (unsigned r = 0; r < Rows; ++r) input.output[g][r] = prepared.row[r].workspace.*targets[m];
        }
        const auto deferred_before = qpf2k_index_batch().deferred;
        dispatch_weight_kind(matrices[lead]->manifest->ggml_type, [&](auto kind) {
            launch_qsa_rows_input<decltype(kind)::value, Rows>(prepared.prefetch[lead], config.dense_a8_projections,
                input, a.hidden, stream, any_order && launched_group, cohort_slot != nullptr);
        });
        // A group deferred by STRIX_QPF2K_QSA bit 7 launched nothing: the next group must not start any-order behind it.
        if (qpf2k_index_batch().deferred == deferred_before) launched_group = true;
        check_hip(hipGetLastError(), "QSA known-window input projection launch");
        if (prepared.prefetch[lead] && config.prefetch_stats)
            config.prefetch_stats->dispatched_projections.fetch_add(input.count, std::memory_order_relaxed);
    }

    if (parts & 2U) {
    const auto norm = [&](const TensorPair& weights, float* WorkspacePointers::* target, std::uint32_t heads,
                          std::uint32_t head_dim, std::uint32_t stride, bool unordered) {
        QsaRowsNormViews<Rows> views{};
        for (unsigned r = 0; r < Rows; ++r) { views.values[r] = prepared.row[r].workspace.*target; views.position[r] = prepared.position[r]; }
        dispatch_small_kind(weights.manifest->ggml_type, [&](auto kind) {
            if (unordered)
                with_qwen_rotary(model, stream, [&](auto... rotary_source) {
                    hipExtLaunchKernelGGL(HIP_KERNEL_NAME(norm_rope_rows_kernel<decltype(kind)::value, Rows, decltype(rotary_source)...>), dim3(heads, Rows), dim3(head_dim),
                        2ULL * head_dim * sizeof(float), stream, nullptr, nullptr, hipExtAnyOrderLaunch, views, heads, head_dim, stride,
                        reinterpret_cast<const void*>(weights.runtime->address), model.rms_norm_epsilon, rotary, model.rope_theta, rotary_source...);
                });
            else with_qwen_rotary(model, stream, [&](auto... rotary_source) {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(norm_rope_rows_kernel<decltype(kind)::value, Rows, decltype(rotary_source)...>), dim3(heads, Rows), dim3(head_dim),
                    2ULL * head_dim * sizeof(float), stream, views, heads, head_dim, stride,
                    reinterpret_cast<const void*>(weights.runtime->address), model.rms_norm_epsilon, rotary, model.rope_theta, rotary_source...);
            });
        });
        check_hip(hipGetLastError(), "QSA known-window norm/rope launch");
    };
    bool norm_grouped = false;
    if constexpr (Rows >= 64U) if ((subparts & 1U) && qsa_norm_group_enabled()) {
        const std::array<const TensorPair*, 3> norm_tensors{&tensors.index_q_norm, &tensors.q_norm, &tensors.k_norm};
        const std::array<float* WorkspacePointers::*, 3> norm_targets{&WorkspacePointers::index_query, &WorkspacePointers::q_gate,
            &WorkspacePointers::current_k};
        // STRIX_QSA_SKIP_INDEX_Q: no index query was projected, so the group has no index heads (the kernel's set 0 is empty).
        // STRIX_QSA_CACHE_ONLY: only the k heads (the stored keys).
        const std::array<std::uint32_t, 3> norm_heads{skip_index_q || cache_only ? 0U : index_heads, cache_only ? 0U : a.attention_heads, a.kv_heads},
            norm_dims{a.index_dim, a.head_dim, a.head_dim},
            norm_strides{a.index_dim, a.head_dim * 2U, a.head_dim};
        const auto type = norm_tensors[0]->manifest->ggml_type;
        bool fits = rotary / 2U <= 64U && rotary <= a.index_dim && rotary <= a.head_dim;
        for (unsigned s = 0; s < 3U; ++s) fits = fits && norm_tensors[s]->manifest->ggml_type == type && norm_dims[s] <= 256U && norm_dims[s] >= 2U &&
            (norm_dims[s] & (norm_dims[s] - 1U)) == 0U;
        if (fits) {
            float2* const rope_table = qsa_group_rope_table();
            static_assert(Rows <= 1024U);
            qsa_index::NormGroupViews<Rows> norm_views{};
            for (unsigned s = 0; s < 3U; ++s) {
                for (unsigned r = 0; r < Rows; ++r) norm_views.values[s][r] = prepared.row[r].workspace.*norm_targets[s];
                norm_views.weights[s] = reinterpret_cast<const void*>(norm_tensors[s]->runtime->address);
                norm_views.heads[s] = norm_heads[s]; norm_views.dim[s] = norm_dims[s]; norm_views.stride[s] = norm_strides[s];
            }
            for (unsigned r = 0; r < Rows; ++r) norm_views.position[r] = prepared.position[r];
            norm_views.table = rope_table;
            with_qwen_rotary(model, stream, [&](auto... rotary_source) {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::rope_table_kernel<Rows, false, decltype(rotary_source)...>), dim3(Rows), dim3(64), 0, stream, norm_views, rotary,
                    model.rope_theta, static_cast<const qsa_index::NormGroupViews<Rows>*>(nullptr), rotary_source...);
            });
            // STRIX_QPF2K_QSA bit 2: one wave per (row, head) (qsa_flash::norm_rope_wave_kernel, the same values); needs every
            // head dim >= 32 and the rotation partner in the same lane (rotary_dim / 2 a multiple of 32).
            bool wave_norm = !cohort_slot && (qpf2k_qsa_mask() & 4U) && (rotary / 2U) % 32U == 0U;
            for (unsigned s = 0; s < 3U; ++s) wave_norm = wave_norm && (norm_heads[s] == 0U || norm_dims[s] >= 32U);
            dispatch_small_kind(type, [&](auto kind) {
                if (wave_norm) {
                    const unsigned waves = Rows * (norm_heads[0] + norm_heads[1] + norm_heads[2]);
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::norm_rope_wave_kernel<decltype(kind)::value, Rows>), dim3((waves + 7U) / 8U), dim3(256), 0,
                        stream, norm_views, model.rms_norm_epsilon, rotary);
                    qpf2k_launch_norm_wave.fetch_add(1U, std::memory_order_relaxed);
                } else
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::norm_rope_group_kernel<decltype(kind)::value, Rows>),
                    dim3(norm_heads[0] + norm_heads[1] + norm_heads[2], Rows), dim3(256), 0, stream, norm_views, model.rms_norm_epsilon, rotary);
            });
            check_hip(hipGetLastError(), "QSA prefill grouped norm/rope launch");
            norm_grouped = true;
        }
    }
    if ((subparts & 1U) && !norm_grouped) {
    if (!skip_index_q && !cache_only) norm(tensors.index_q_norm, &WorkspacePointers::index_query, index_heads, a.index_dim, a.index_dim, false);
    // Without the index norm the q norm is the barrier-ordered one (it follows the input projections).
    if (!cache_only) norm(tensors.q_norm, &WorkspacePointers::q_gate, a.attention_heads, a.head_dim, a.head_dim * 2U, any_order && !skip_index_q);
    // Cache-only: the k norm is the first after the projections, so it is barrier-ordered.
    norm(tensors.k_norm, &WorkspacePointers::current_k, a.kv_heads, a.head_dim, a.head_dim, any_order && !cache_only);
    }

    QsaRowsStoreViews<Rows> store{};
    for (unsigned r = 0; r < Rows; ++r) {
        const auto& w = prepared.row[r].workspace;
        store.index_key[r] = w.current_index_key; store.k[r] = w.current_k; store.v[r] = w.current_v;
        store.position[r] = prepared.position[r];
    }
    store.journal = prepared.journal;
    const bool kv_int8 = model.attention_kv_storage == AttentionKvStorage::int8_token;
    const auto bf16_kv_width = kv_int8 ? 0U : a.kv_width; // 0: store_rows_kernel writes only the raw index keys
    if (subparts & 2U) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(store_rows_kernel<Rows>), dim3((std::max(bf16_kv_width, a.index_dim) + kThreads - 1U) / kThreads, Rows),
        dim3(kThreads), 0, stream, store, reinterpret_cast<std::uint16_t*>(a.raw_state.address),
        reinterpret_cast<std::uint16_t*>(a.kv_state.address), a.index_dim, bf16_kv_width);
    check_hip(hipGetLastError(), "QSA known-window raw-index/KV append");
    if (kv_int8) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(store_rows_kv_int8_kernel<Rows>), dim3(a.kv_heads, 2U, Rows), dim3(kThreads), 0, stream,
            store, reinterpret_cast<void*>(a.kv_state.address), a.kv_width, a.kv_heads, a.head_dim);
        check_hip(hipGetLastError(), "QSA known-window int8 KV append");
    }
    }
    // STRIX_QSA_CACHE_ONLY: the layer's persistent writes are done; nothing reads this step's rows.
    if (cache_only) return;

    QsaRowsScoreViews<Rows> scores{};
    QsaRowsSelectViews<Rows> select{};
    std::uint32_t scored_blocks = 0;
    const auto& s0 = prepared.selection[0];
    for (unsigned r = 0; r < Rows; ++r) {
        const auto& s = prepared.selection[r];
        const auto& w = prepared.row[r].workspace;
        scores.index_query[r] = w.index_query; scores.block_scores[r] = w.block_scores;
        scores.blocks[r] = s.complete_blocks > s.block_topk ? s.complete_blocks : 0U;
        scored_blocks = std::max(scored_blocks, scores.blocks[r]);
        select.block_scores[r] = w.block_scores; select.selected[r] = w.selected_indices; select.visible[r] = s.visible;
    }
    // STRIX_QPF2K_QSA bit 1: the scorer below left the group maxima here; the selection then runs qsa_flash::select_kernel.
    float* qpf2k_maxima = staged ? staged->maxima : nullptr;
    std::uint32_t qpf2k_maxima_row_stride = staged ? staged->maxima_stride : qpf2k_maxima_stride;
    // Deferred (batched) selection only where nothing reads this group's selections before the call's flush: 64-row
    // groups of the fast prefill profile with BF16 KV, whose union / attend go to the call's attend batch.
    const bool qpf2k_defer = Rows == 64U && !cohort_slot && qpf2k_select_batch().active && config.qsa_fast_prefill &&
        model.attention_kv_storage == AttentionKvStorage::bf16;
    // Prefill groups (Rows > 8) and, with STRIX_QSA_WINDOW_POOLED=1, verify windows of 2/4/8 rows
    // (indexed_attention_known_window_rows_launch passes a pool then).
    const bool pooled = pool && (pool->keys || pool->packed) && scored_blocks <= pool->capacity && a.index_dim == 128U && index_heads == 4U;
    if ((subparts & 4U) && scored_blocks && pooled) {
        {
            if (scored_blocks > pool->valid) {
                dispatch_small_kind(tensors.index_k_norm.manifest->ggml_type, [&](auto kind) {
                    with_qwen_rotary(model, stream, [&](auto... rotary_source) {
                        const auto build=[&]<bool PackedOnly>() {
                            hipLaunchKernelGGL(HIP_KERNEL_NAME(block_pool_kernel<decltype(kind)::value, false, PackedOnly, decltype(rotary_source)...>), dim3(scored_blocks - pool->valid),
                                dim3(a.index_dim), 2ULL * a.index_dim * sizeof(float), stream,
                                reinterpret_cast<const std::uint16_t*>(a.raw_state.address), pool->valid, a.index_dim, s0.ratio,
                                reinterpret_cast<const void*>(tensors.index_k_norm.runtime->address), model.rms_norm_epsilon, rotary,
                                model.rope_theta, pool->keys, static_cast<const uint2*>(nullptr), 0U, pool->packed, rotary_source...);
                        };
                        if (!pool->keys && pool->packed) build.template operator()<true>();
                        else build.template operator()<false>();
                    });
                });
                check_hip(hipGetLastError(), "QSA prefill block-pool launch");
                // STRIX_QPF2K_QSA bit 0: the new blocks' magnitude ranges for the scorer's guard.
                if constexpr (Rows >= 16U) if (!cohort_slot && (qpf2k_qsa_mask() & 1U)) {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::key_range_kernel<>), dim3((scored_blocks - pool->valid + 255U) / 256U), dim3(256), 0, stream,
                        static_cast<const float*>(pool->keys), pool->valid, scored_blocks - pool->valid, qpf2k_key_ranges(pool->capacity));
                    check_hip(hipGetLastError(), "QSA prefill key-range launch");
                }
                pool->valid = scored_blocks;
            }
            // STRIX_QSA_POOLED_ROWS=1|2|4|8: rows per scoring wave (default 4).
            static const unsigned rows_per_wave = [] {
                const char* v = std::getenv("STRIX_QSA_POOLED_ROWS");
                const int n = v ? std::atoi(v) : 4;
                return n == 1 || n == 2 || n == 8 ? static_cast<unsigned>(n) : 4U;
            }();
            // STRIX_QSA_SCORE_PARK=1 (prefill groups): qsa_index::block_score_parked_kernel, the same scores.
            static const bool park = [] { const char* v = std::getenv("STRIX_QSA_SCORE_PARK"); return v && v[0] == '1'; }();
            const auto score = [&]<unsigned Wanted>() {
                constexpr unsigned R = Wanted < Rows ? Wanted : Rows; // a 2-row window scores both rows per wave
                const auto waves = static_cast<std::uint64_t>(Rows / R) * ((scored_blocks + 31U) / 32U);
                // Prefill groups only (cohort windows pass their output slot); the scorer's prologue reads the queries
                // as float4 (16-byte aligned rows required, else the kernels below run).
                if constexpr (Rows >= 16U) if (!cohort_slot && (qpf2k_qsa_mask() & 1U) && [&] {
                        for (unsigned r = 0; r < Rows; ++r) if (reinterpret_cast<std::uintptr_t>(scores.index_query[r]) % 16U) return false;
                        return true; }()) {
                    constexpr unsigned W = qsa_flash::kScoreWaves;
                    const unsigned tiles = (scored_blocks + 31U) / 32U;
                    // Bit 1: the group maxima for qsa_flash::select_kernel (one group's rows at a time: stream order).
                    float* maxima = nullptr;
                    std::uint32_t maxima_stride = qpf2k_maxima_stride;
                    auto& sb = qpf2k_select_batch();
                    if (auto* external=qsa_selector_maxima(qpf2k_qsa_mask(),qsa_exact_maxima)) {
                        maxima=external;maxima_stride=qsa_exact_maxima_stride;qpf2k_maxima=maxima;
                    } else if (qpf2k_defer) {
                        // Deferred selection: this group's maxima in the arena, ceil(blocks / 4) floats per row.
                        maxima_stride = (scored_blocks + qsa_flash::kGroup - 1U) / qsa_flash::kGroup;
                        const std::uint64_t need = static_cast<std::uint64_t>(Rows) * maxima_stride;
                        if (sb.used + need > sb.capacity) {
                            // early_flush witnesses arena reuse: only a flush with queued selections counts
                            if (!sb.rows.empty()) qpf2k_select_early_flush.fetch_add(1U, std::memory_order_relaxed);
                            qpf2k_select_flush(stream);
                        }
                        if (need > sb.capacity) {
                            if (sb.arena) {
                                check_hip(hipStreamSynchronize(stream), "QSA prefill maxima arena resize sync");
                                check_hip(hipFree(sb.arena), "QSA prefill maxima arena free");
                            }
                            void* q = nullptr;
                            sb.capacity = std::max<std::uint64_t>(need, std::uint64_t{16U} << 20U);
                            check_hip(hipMalloc(&q, sb.capacity * sizeof(float)), "QSA prefill maxima arena allocation");
                            sb.arena = static_cast<float*>(q);
                        }
                        maxima = sb.arena + sb.used;
                        sb.used += need;
                        qpf2k_maxima = maxima;
                    } else if (qsa_maxima_selector_enabled(qpf2k_qsa_mask()) && scored_blocks <= qpf2k_maxima_stride * qsa_flash::kGroup) {
                        static float* buffer = [] {
                            void* p = nullptr;
                            check_hip(hipMalloc(&p, std::size_t{64U} * qpf2k_maxima_stride * sizeof(float)), "QSA prefill group-maxima allocation");
                            return static_cast<float*>(p);
                        }();
                        static_assert(Rows <= 64U);
                        maxima = buffer;
                        qpf2k_maxima = buffer;
                    }
                    qpf2k_launch_score.fetch_add(1U, std::memory_order_relaxed);
                    const auto launch = [&]<unsigned RowsPerCta>() {
                        if (pool->packed)
                            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::score_kernel<Rows, RowsPerCta, W, false, false, true, 1U, false, true>),
                                dim3((tiles + W - 1U) / W, Rows / RowsPerCta), dim3(W * 32U), 0, stream, scores, static_cast<const void*>(pool->packed),
                                scored_blocks, a.index_dim, maxima, maxima_stride, static_cast<const uint2*>(qpf2k_key_ranges(pool->capacity)));
                        else
                            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::score_kernel<Rows, RowsPerCta, W>), dim3((tiles + W - 1U) / W, Rows / RowsPerCta),
                                dim3(W * 32U), 0, stream, scores, static_cast<const float*>(pool->keys), scored_blocks, a.index_dim, maxima, maxima_stride,
                                static_cast<const uint2*>(qpf2k_key_ranges(pool->capacity)));
                    };
                    // 64-row groups use the stage batch's rows-per-CTA policy (qsa_flash::score_rows): this path scores every group above
                    // 65536 blocks, where the stage batch declines. Same bits for every RowsPerCta (tests/qsa_score_rows.hip).
                    constexpr unsigned BaseRows = Rows < qsa_flash::kScoreRows ? Rows : qsa_flash::kScoreRows;
                    if constexpr (Rows == 64U) {
                        const auto rows = qsa_flash::score_rows(scored_blocks);
                        if (rows == 64U) launch.template operator()<64U>();
                        else if (rows == 32U) launch.template operator()<32U>();
                        else launch.template operator()<BaseRows>();
                    } else launch.template operator()<BaseRows>();
                    qpf2k_maxima_row_stride = maxima_stride;
                    return;
                }
                if constexpr (Rows >= 16U) if (!cohort_slot && qpf2k_trace_enabled()) qpf2k_launch_prod_score.fetch_add(1U, std::memory_order_relaxed);
                if constexpr (Rows >= 64U) if (!qpf2k::suppressed(qpf2k::Item::score_park) && park) {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::block_score_parked_kernel<Rows, R>), dim3(static_cast<std::uint32_t>((waves + 7U) / 8U)),
                        dim3(kThreads), 0, stream, scores, static_cast<const float*>(pool->keys), scored_blocks, a.index_dim);
                    return;
                }
                if constexpr (Rows<=8U) {
                    if (!pool->keys && pool->packed) {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(block_score_pooled_rows_kernel<Rows, R, std::uint16_t>), dim3(static_cast<std::uint32_t>((waves + 7U) / 8U)),
                        dim3(kThreads), 0, stream, scores, static_cast<const std::uint16_t*>(pool->packed), scored_blocks, a.index_dim);
                } else {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(block_score_pooled_rows_kernel<Rows, R>), dim3(static_cast<std::uint32_t>((waves + 7U) / 8U)),
                    dim3(kThreads), 0, stream, scores, static_cast<const float*>(pool->keys), scored_blocks, a.index_dim);
                    }
                } else {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(block_score_pooled_rows_kernel<Rows, R>), dim3(static_cast<std::uint32_t>((waves + 7U) / 8U)),
                        dim3(kThreads), 0, stream, scores, static_cast<const float*>(pool->keys), scored_blocks, a.index_dim);
                }
            };
            if (rows_per_wave == 1U) score.template operator()<1U>();
            else if (rows_per_wave == 2U) score.template operator()<2U>();
            else if (rows_per_wave == 8U) score.template operator()<8U>();
            else score.template operator()<4U>();
            check_hip(hipGetLastError(), "QSA prefill pooled block-score launch");
            // STRIX_QSA_POOLED_CHECK=1: compare every row's block scores with block_score_rows_kernel's.
            static const bool check = [] { const char* v = std::getenv("STRIX_QSA_POOLED_CHECK"); return v && v[0] == '1'; }();
            const bool mtp_check=config.committed_pool && std::getenv("STRIX_QSA_MTP_POOL_CHECK") && std::getenv("STRIX_QSA_MTP_POOL_CHECK")[0]=='1';
            if (check || mtp_check) {
                const auto copy = [&] {
                    check_hip(hipStreamSynchronize(stream), "QSA pooled check sync");
                    std::vector<float> all;
                    for (unsigned r = 0; r < Rows; ++r) {
                        const auto at = all.size();
                        all.resize(at + scores.blocks[r]);
                        if (scores.blocks[r]) check_hip(hipMemcpy(all.data() + at, scores.block_scores[r], scores.blocks[r] * sizeof(float),
                            hipMemcpyDeviceToHost), "QSA pooled check copy");
                    }
                    return all;
                };
                const auto fast = copy();
                dispatch_small_kind(tensors.index_k_norm.manifest->ggml_type, [&](auto kind) {
                    with_qwen_rotary(model, stream, [&](auto... rotary_source) {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(block_score_rows_kernel<decltype(kind)::value, Rows, decltype(rotary_source)...>), dim3(scored_blocks, Rows <= 8U ? 1U : Rows / 8U),
                            dim3(a.index_dim), 2ULL * a.index_dim * sizeof(float), stream, scores,
                            reinterpret_cast<const std::uint16_t*>(a.raw_state.address), index_heads, a.index_dim, s0.ratio,
                            reinterpret_cast<const void*>(tensors.index_k_norm.runtime->address), model.rms_norm_epsilon, rotary, model.rope_theta, rotary_source...);
                    });
                });
                const auto reference = copy();
                std::size_t differing = 0;
                for (std::size_t i = 0; i < fast.size(); ++i) differing += std::memcmp(&fast[i], &reference[i], sizeof(float)) != 0;
                if (mtp_check) {
                    std::size_t at=0;
                    for (unsigned r=0;r<Rows;++r) {
                        if (scores.blocks[r]) check_hip(hipMemcpyAsync(scores.block_scores[r],fast.data()+at,scores.blocks[r]*sizeof(float),
                            hipMemcpyHostToDevice,stream),"MTP window check restore scores");
                        at+=scores.blocks[r];
                    }
                    check_hip(hipStreamSynchronize(stream),"MTP window check restore completion");
                    config.committed_pool->checked(fast.size(),differing,pool->mtp_role);
                }
                static std::atomic<std::uint64_t> groups{0}, compared{0}, total_differing{0};
                const auto g = groups.fetch_add(1U) + 1U;
                const auto c = compared.fetch_add(fast.size()) + fast.size();
                const auto d = total_differing.fetch_add(differing) + differing;
                if (differing || g % 64U == 0U)
                    std::fprintf(stderr, "QSA pooled check: group at position %llu, %zu scores, %zu differ (total %llu groups, %llu scores, %llu differ)\n",
                        static_cast<unsigned long long>(prepared.position[0]), fast.size(), differing, static_cast<unsigned long long>(g),
                        static_cast<unsigned long long>(c), static_cast<unsigned long long>(d));
            }
        }
    } else if ((subparts & 4U) && scored_blocks) {
        dispatch_small_kind(tensors.index_k_norm.manifest->ggml_type, [&](auto kind) {
            with_qwen_rotary(model, stream, [&](auto... rotary_source) {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(block_score_rows_kernel<decltype(kind)::value, Rows, decltype(rotary_source)...>),
                    dim3(scored_blocks, Rows <= 8U ? 1U : Rows / 8U), dim3(a.index_dim),
                    2ULL * a.index_dim * sizeof(float), stream, scores, reinterpret_cast<const std::uint16_t*>(a.raw_state.address),
                    index_heads, a.index_dim, s0.ratio, reinterpret_cast<const void*>(tensors.index_k_norm.runtime->address),
                    model.rms_norm_epsilon, rotary, model.rope_theta, rotary_source...);
            });
        });
        check_hip(hipGetLastError(), "QSA known-window block-score launch");
    }
    // STRIX_QPF2K_QSA_TRACE: prefill groups' scores before the selection (both scorers), selections and scores after it.
    const bool qpf2k_trace_group = Rows >= 16U && !cohort_slot && qpf2k_trace_enabled() && (subparts & 8U);
    if (qpf2k_trace_group && scored_blocks)
        qpf2k_trace_rows<Rows>(0U, prepared.position, [&](unsigned r) { return scores.block_scores[r]; }, [&](unsigned r) { return scores.blocks[r]; }, stream);
    std::vector<std::unique_ptr<MtpPoolSelectionCheck>> mtp_selection_checks;
    if(config.committed_pool && (subparts & 8U) && std::getenv("STRIX_QSA_MTP_POOL_CHECK") &&
        std::getenv("STRIX_QSA_MTP_POOL_CHECK")[0]=='1') for(unsigned r=0;r<Rows;++r) {
        const auto& w=prepared.row[r].workspace;const auto& sr=prepared.selection[r];
        mtp_selection_checks.push_back(std::make_unique<MtpPoolSelectionCheck>(config.committed_pool,
            HipQsaPoolRowDescriptor{w.index_query,w.block_scores,w.selected_indices,sr.visible,sr.complete_blocks,sr.block_topk,0U,nullptr},stream));
    }
    bool qpf2k_deferred = false;
    if ((subparts & 8U) && qpf2k_maxima && qpf2k_defer) {
        auto& sb = qpf2k_select_batch();
        sb.ratio = s0.ratio; sb.topk = s0.block_topk;
        for (unsigned r = 0; r < Rows; ++r)
            sb.rows.push_back(qsa_flash::SelectRow{select.block_scores[r], select.selected[r], qpf2k_maxima + static_cast<std::uint64_t>(r) * qpf2k_maxima_row_stride,
                select.visible[r], 0U});
        qpf2k_launch_select.fetch_add(1U, std::memory_order_relaxed);
        qpf2k_deferred = true;
    } else if ((subparts & 8U) && qpf2k_maxima) {
        qpf2k_launch_select.fetch_add(1U, std::memory_order_relaxed);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::select_kernel<Rows>), dim3(Rows), dim3(kSelectThreads), 0, stream, select,
            static_cast<const float*>(qpf2k_maxima), qpf2k_maxima_row_stride, s0.ratio, s0.block_topk);
    } else if (subparts & 8U) {
        if (Rows >= 16U && !cohort_slot && qpf2k_trace_enabled()) qpf2k_launch_prod_select.fetch_add(1U, std::memory_order_relaxed);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(select_rows_kernel<Rows>), dim3(Rows), dim3(kSelectThreads), 0, stream, select, s0.ratio, s0.block_topk);
    }
    check_hip(hipGetLastError(), "QSA known-window selection launch");
    for(auto& check:mtp_selection_checks) check->finish();
    if (qpf2k_trace_group && qpf2k_deferred) {
        Qpf2kSelectGroup g{};
        g.rows = Rows;
        for (unsigned r = 0; r < Rows; ++r) {
            g.position[r] = prepared.position[r]; g.selected[r] = select.selected[r]; g.count[r] = prepared.selection[r].selected_count;
            g.scores[r] = scores.block_scores[r]; g.blocks[r] = scored_blocks ? scores.blocks[r] : 0U;
        }
        qpf2k_select_batch().traced.push_back(g);
        qpf2k_trace().rows += Rows;
    } else if (qpf2k_trace_group) {
        qpf2k_trace_rows<Rows>(1U, prepared.position, [&](unsigned r) { return select.selected[r]; },
            [&](unsigned r) { return prepared.selection[r].selected_count; }, stream);
        if (scored_blocks)
            qpf2k_trace_rows<Rows>(2U, prepared.position, [&](unsigned r) { return scores.block_scores[r]; }, [&](unsigned r) { return scores.blocks[r]; }, stream);
        qpf2k_trace().rows += Rows;
    }
    // STRIX_QSA_UNION_DUMP=1 (diagnostic, synchronizes; prefill groups only): how many distinct tokens the
    // selections of 8/16/32/64 consecutive rows cover, against the per-row count (stderr, every 16th group).
    if constexpr (Rows > 8U) {
        static const bool union_dump = [] { const char* v = std::getenv("STRIX_QSA_UNION_DUMP"); return v && v[0] == '1'; }();
        static std::atomic<std::uint64_t> union_groups{0};
        if (union_dump && union_groups.fetch_add(1U) % 16U == 0U) {
            qpf2k_select_flush(stream);
            check_hip(hipStreamSynchronize(stream), "QSA union dump sync");
            std::vector<std::vector<std::uint32_t>> sel(Rows);
            for (unsigned r = 0; r < Rows; ++r) {
                sel[r].resize(prepared.selection[r].selected_count);
                if (!sel[r].empty()) check_hip(hipMemcpy(sel[r].data(), prepared.row[r].workspace.selected_indices,
                    sel[r].size() * sizeof(std::uint32_t), hipMemcpyDeviceToHost), "QSA union dump copy");
            }
            std::string line = "QSA union: position " + std::to_string(prepared.position[0]) + " count " + std::to_string(sel[0].size());
            for (const unsigned tile : {8U, 16U, 32U, 64U}) {
                if (tile > Rows) continue;
                double ratio = 0.0; unsigned tiles = 0;
                for (unsigned t0 = 0; t0 + tile <= Rows; t0 += tile) {
                    std::vector<std::uint32_t> all;
                    std::size_t mean = 0;
                    for (unsigned r = t0; r < t0 + tile; ++r) { all.insert(all.end(), sel[r].begin(), sel[r].end()); mean += sel[r].size(); }
                    std::sort(all.begin(), all.end()); all.erase(std::unique(all.begin(), all.end()), all.end());
                    ratio += mean ? static_cast<double>(all.size()) * tile / static_cast<double>(mean) : 0.0; ++tiles;
                }
                line += " | tile " + std::to_string(tile) + " union/count " + std::to_string(tiles ? ratio / tiles : 0.0);
            }
            bool sorted = true;
            for (unsigned r = 0; r < Rows; ++r) sorted = sorted && std::is_sorted(sel[r].begin(), sel[r].end());
            std::fprintf(stderr, "%s | rows sorted %d\n", line.c_str(), sorted ? 1 : 0);
        }
    }

    QsaRowsAttentionViews<Rows> attention{};
    std::uint32_t maximum_count = 0;
    for (unsigned r = 0; r < Rows; ++r) {
        const auto& w = prepared.row[r].workspace;
        attention.q_gate[r] = w.q_gate; attention.selected[r] = w.selected_indices;
        attention.scores[r] = w.attention_scores; attention.output[r] = w.attention_output; attention.compact_output = w.compact_attention_output;
        attention.count[r] = prepared.selection[r].selected_count;
        maximum_count = std::max(maximum_count, attention.count[r]);
    }
    const auto maximum_selected = checked_u32(a.workspace_layout.maximum_selected_tokens, "maximum_selected_tokens");
    // kv-int8-tok-v1 reads the same kernels' int8 forms (same arithmetic order, dequantized values); the fast
    // prefill QSA reads BF16 keys and values and does not run with it.
    const auto attend = [&]<bool Int8>() {
        const auto* kv = reinterpret_cast<const std::uint16_t*>(a.kv_state.address);
        // Prefill groups: KV-head-grouped score / output kernels. STRIX_QSA_PREFILL_GROUP=0 keeps the
        // window kernels; =2|3|4|6|12 query heads per score wave (default 12) with
        // STRIX_QSA_PREFILL_SLOTS=4|8|16 slots per wave (default 16); STRIX_QSA_PREFILL_OUT_GROUP=
        // 2|3|4|6|12 heads per output block (default 12). Measured at T = 256 (1032-token prompt).
        const auto env_choice = [](const char* name, int fallback) {
            const char* v = std::getenv(name);
            const int n = v ? std::atoi(v) : fallback;
            return n == 0 || n == 2 || n == 3 || n == 4 || n == 6 || n == 12 ? static_cast<unsigned>(n) : static_cast<unsigned>(fallback);
        };
        static const unsigned head_group = env_choice("STRIX_QSA_PREFILL_GROUP", 12);
        static const unsigned out_group = env_choice("STRIX_QSA_PREFILL_OUT_GROUP", 12);
        static const unsigned group_slots = [] {
            const char* v = std::getenv("STRIX_QSA_PREFILL_SLOTS");
            const int n = v ? std::atoi(v) : 16;
            return n == 4 || n == 8 ? static_cast<unsigned>(n) : 16U;
        }();
        // Fast prefill profile, qsa component: the fused union attention replaces score, softmax and value/gate.
        bool fast = false;
        // STRIX_QSA_FAST_INT8=1 (opt-in): kv-int8-tok-v1 prefill groups also run the fast kernels (int8 loaders).
        if constexpr (Rows > 8U) if ((!Int8 || qsa_fast_int8_enabled()) && config.qsa_fast_prefill)
            fast = launch_qsa_fast_prefill<Rows, Int8>(prepared, a, maximum_selected, stream);
        bool grouped = false, grouped_score = false;
        const auto per_kv = a.kv_heads && a.attention_heads % a.kv_heads == 0U ? a.attention_heads / a.kv_heads : 0U;
        if constexpr (Rows > 8U) {
            grouped = head_group != 0U && out_group != 0U && a.head_dim <= kThreads && per_kv &&
                per_kv % head_group == 0U && per_kv % out_group == 0U;
            grouped_score = grouped;
        } else {
            // STRIX_QSA_WINDOW_GROUPED=1: verify windows of 2/4/8 rows score with the prefill groups'
            // KV-head-grouped kernel (per (row, query head, slot) the same products and reduction as
            // attention_score_rows_wave_kernel; the key row is read once per head group instead of
            // once per head). STRIX_QSA_WINDOW_SLOTS=4|8|16 slots per wave (default 8). The value/gate
            // output keeps the window kernel.
            grouped_score = qsa_window_grouped_requested() && head_group != 0U && a.head_dim <= kThreads && per_kv && per_kv % head_group == 0U;
        }
        static const unsigned window_slots = [] {
            const char* v = std::getenv("STRIX_QSA_WINDOW_SLOTS");
            const int n = v ? std::atoi(v) : 8;
            return n == 4 || n == 16 ? static_cast<unsigned>(n) : 8U;
        }();
        const auto score_slots = Rows > 8U ? group_slots : window_slots;
        // STRIX_QSA_WINDOW_OUT_GROUP (exact, default 4): 8-row verify windows (verify_window) form value/gate with
        // attention_output_rows_window_group_kernel (G = 2|3|4|6|12 query heads per block, so each value row is
        // read once per G heads); 0 keeps the per-head kernel. Kernel bench (BF16, 2052 / 1031 slots): R8 186 -> 91 /
        // 99 -> 47 us at G4; 2- and 4-row windows keep the per-head kernel (G4 +9 % / +0 % there). Off under the baseline
        // and STRIX_QPF2K_DEFAULTS=0.
        static const unsigned window_out_group_choice = env_choice("STRIX_QSA_WINDOW_OUT_GROUP", 4);
        static const bool window_out_group_on = qpf2k::long_context_item_enabled("STRIX_QSA_WINDOW_OUT_GROUP", false);
        const unsigned window_out_group = Rows == 8U && verify_window && window_out_group_on ? window_out_group_choice : 0U;
        // STRIX_QSA_WINDOW_ATTN_V2 (default on): the operand-pinned grouped score for single-session verify windows
        // (see window_attn_v2_route).
        const bool window_v2 = window_attn_v2_route(qsa_window_attn_v2_enabled(), verify_window, Rows, Int8, fast,
            qsa_window_grouped_requested(), qsa_score_slots_per_wave(), a.attention_heads, a.kv_heads, a.head_dim);
        if (!(subparts & 16U) || fast) {
        } else if (window_v2) {
            static_cast<void>(prepare_indexed_attention_score_grid(a.attention_heads, maximum_count));
            if constexpr (Rows <= 8U && !Int8) launch_window_attn_v2_scores<Rows>(attention, kv, maximum_selected, a.attention_heads,
                a.kv_heads, a.head_dim, maximum_count, stream);
        } else if (grouped_score) {
            {
                static_cast<void>(prepare_indexed_attention_score_grid(a.attention_heads, maximum_count));
                static const bool scatter = [] { const char* v = std::getenv("STRIX_QSA_PREFILL_SCATTER"); return !v || v[0] != '0'; }();
                // STRIX_QSA_SCORE_PREFETCH=0 keeps the generic kernel for head_dim 256.
                static const bool prefetch = [] { const char* v = std::getenv("STRIX_QSA_SCORE_PREFETCH"); return !v || v[0] != '0'; }();
                const auto score = [&]<unsigned G, unsigned S>() {
                    const auto waves = static_cast<std::uint64_t>(Rows) * (a.attention_heads / G) * ((maximum_count + S - 1U) / S);
                    const auto blocks = dim3(static_cast<std::uint32_t>((waves + 7U) / 8U));
                    if (scatter && prefetch && a.head_dim == 256U) hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_score_rows_group256_kernel<Rows, G, S, Int8>),
                        blocks, dim3(kThreads), 0, stream, attention, kv, maximum_selected, a.attention_heads, a.kv_heads, a.head_dim, maximum_count);
                    else if (scatter) hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_score_rows_group_kernel<Rows, G, S, true, Int8>), blocks, dim3(kThreads),
                        0, stream, attention, kv, maximum_selected, a.attention_heads, a.kv_heads, a.head_dim, maximum_count);
                    else hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_score_rows_group_kernel<Rows, G, S, false, Int8>), blocks, dim3(kThreads),
                        0, stream, attention, kv, maximum_selected, a.attention_heads, a.kv_heads, a.head_dim, maximum_count);
                };
                const auto by_group = [&]<unsigned G>() {
                    if (score_slots == 4U) score.template operator()<G, 4U>();
                    else if (score_slots == 16U) score.template operator()<G, 16U>();
                    else score.template operator()<G, 8U>();
                };
                switch (head_group) {
                case 2U: by_group.template operator()<2U>(); break;
                case 3U: by_group.template operator()<3U>(); break;
                case 4U: by_group.template operator()<4U>(); break;
                case 12U: by_group.template operator()<12U>(); break;
                default: by_group.template operator()<6U>(); break;
                }
            }
        } else if (a.head_dim <= kThreads) {
            static_cast<void>(prepare_indexed_attention_score_grid(a.attention_heads, maximum_count));
            // STRIX_QSA_SCORE_SLOTS=1|2|4|8: selected slots per wave (same per-slot arithmetic).
            const unsigned per_wave = qsa_score_slots_per_wave();
            const auto launch_score = [&]<unsigned PerWave>() {
                const auto groups = (maximum_count + PerWave - 1U) / PerWave;
                const auto waves = static_cast<std::uint64_t>(Rows) * a.attention_heads * groups;
                hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_score_rows_wave_kernel<Rows, PerWave, Int8>),
                    dim3(static_cast<std::uint32_t>((waves + 7U) / 8U)), dim3(kThreads), 0, stream, attention, kv, maximum_selected,
                    a.attention_heads, a.kv_heads, a.head_dim, maximum_count);
            };
            if (per_wave == 2U) launch_score.template operator()<2U>();
            else if (per_wave == 4U) launch_score.template operator()<4U>();
            else if (per_wave == 8U) launch_score.template operator()<8U>();
            else launch_score.template operator()<1U>();
        } else hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_score_rows_kernel<Rows, Int8>),
            dim3(static_cast<std::uint32_t>(prepare_indexed_attention_score_grid(a.attention_heads, maximum_count)), Rows),
            dim3(kThreads), 0, stream, attention, kv, maximum_selected, a.attention_heads, a.kv_heads, a.head_dim);
        check_hip(hipGetLastError(), "QSA known-window score launch");
        // Complete BF16 window tails only; prefill schedules stay separate. STRIX_QSA_COHORT_TAIL=1: a cohort
        // session's part 2 (softmax and value/gate in the same call) takes the tail too; its dense A8 codes go
        // to the session's rows of the cohort slot (output_slot below), as the value/gate kernel writes them.
        // STRIX_QSA_TAIL_INT8=1 (opt-in): the fused tail for kv-int8 windows too (the same softmax and value bodies).
        const bool fused_tail = Rows <= 8U && qpersist_qsa_tail_enabled() && (!Int8 || qsa_tail_int8_enabled()) && !fast && !grouped &&
            (!cohort_slot || qsa_cohort_tail_enabled()) && a.head_dim <= kThreads && (subparts & 96U) == 96U;
        if (fused_tail) count_qsa_launch(Int8 ? &QsaLaunchStats::tail_int8 : &QsaLaunchStats::tail_bf16);
        if (!fused_tail && !fast && (subparts & 32U))
            hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_softmax_rows_kernel<Rows>), dim3(a.attention_heads, Rows), dim3(kThreads), 0,
                stream, attention, maximum_selected);
        check_hip(hipGetLastError(), "QSA known-window softmax launch");
        // Dense A8 window with the output projection in this call: the value/gate kernel writes its codes.
        if constexpr (Rows <= 8U) if (config.dense_a8_projections && ((parts & 4U) || cohort_slot) && tensors.out.manifest->ggml_type == 8U) {
            if (a.head_dim % 32U || a.head_dim > 1024U || a.attention_width != a.attention_heads * a.head_dim)
                throw std::invalid_argument("QSA rows dense A8 value/gate quantizer requires whole 32-feature head blocks");
            output_slot = cohort_slot ? q8_rows::DenseA8Slot{cohort_slot->codes + static_cast<std::uint64_t>(cohort_row) * a.attention_width,
                cohort_slot->deltas + cohort_row} : q8_rows::dense_a8_slot(0U);
        }
        if (fused_tail) {
            if (output_slot.codes)
                hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_tail_rows_kernel<Rows, Int8, true>),
                    dim3(a.attention_heads, Rows), dim3(kThreads), 0, stream, attention, kv, maximum_selected,
                    a.attention_heads, a.kv_heads, a.head_dim, output_slot.codes, output_slot.deltas);
            else hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_tail_rows_kernel<Rows, Int8>),
                dim3(a.attention_heads, Rows), dim3(kThreads), 0, stream, attention, kv, maximum_selected,
                a.attention_heads, a.kv_heads, a.head_dim, nullptr, nullptr);
            check_hip(hipGetLastError(), "QSA known-window fused softmax/value launch");
        }
        if (!(subparts & 64U) || fast || fused_tail) {
        } else if (grouped) {
            if constexpr (Rows > 8U) {
                // STRIX_QSA_OUTPUT_PREFETCH=0 keeps the 4-slot kernel without value prefetch.
                static const bool output_prefetch = [] { const char* v = std::getenv("STRIX_QSA_OUTPUT_PREFETCH"); return !v || v[0] != '0'; }();
                const auto output = [&]<unsigned G>() {
                    if (output_prefetch) hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_output_rows_group_prefetch_kernel<Rows, G, Int8>),
                        dim3(a.attention_heads / G, Rows), dim3(a.head_dim), 0, stream, attention, kv, maximum_selected, a.attention_heads,
                        a.kv_heads, a.head_dim);
                    else hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_output_rows_group_kernel<Rows, G, Int8>), dim3(a.attention_heads / G, Rows),
                        dim3(a.head_dim), 0, stream, attention, kv, maximum_selected, a.attention_heads, a.kv_heads, a.head_dim);
                };
                switch (out_group) {
                case 2U: output.template operator()<2U>(); break;
                case 3U: output.template operator()<3U>(); break;
                case 4U: output.template operator()<4U>(); break;
                case 12U: output.template operator()<12U>(); break;
                default: output.template operator()<6U>(); break;
                }
            }
        } else if (window_out_group && !Int8 && per_kv % (window_out_group ? window_out_group : 1U) == 0U && a.head_dim <= kThreads) {
            if constexpr (Rows <= 8U && !Int8) {
            const auto output = [&]<unsigned G>() {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_output_rows_window_group_kernel<Rows, G, Int8>), dim3(a.attention_heads / G, Rows),
                    dim3(a.head_dim), 0, stream, attention, kv, maximum_selected, a.attention_heads, a.kv_heads, a.head_dim,
                    output_slot.codes, output_slot.deltas);
            };
            switch (window_out_group) {
            case 2U: output.template operator()<2U>(); break;
            case 3U: output.template operator()<3U>(); break;
            case 4U: output.template operator()<4U>(); break;
            case 6U: output.template operator()<6U>(); break;
            default: output.template operator()<12U>(); break;
            }
            }
        } else if (output_slot.codes)
            hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_output_rows_dense_a8_kernel<Rows, Int8>), dim3(a.attention_heads, Rows),
                dim3(a.head_dim), 0, stream, attention, kv, maximum_selected, a.attention_heads, a.kv_heads, a.head_dim,
                output_slot.codes, output_slot.deltas);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_output_rows_kernel<Rows, Int8>), dim3(a.attention_heads, Rows), dim3(a.head_dim), 0,
            stream, attention, kv, maximum_selected, a.attention_heads, a.kv_heads, a.head_dim);
        check_hip(hipGetLastError(), "QSA known-window value/gate launch");
    };
    if (model.attention_kv_storage == AttentionKvStorage::int8_token) attend.template operator()<true>();
    else attend.template operator()<false>();
    }
    if (!(parts & 4U)) return;
    // Cohorts of 16 / 32 rows (several sessions' segments): the WMMA projection from the same slot.
    constexpr bool slot_rows = Rows <= 8U || (Rows % 16U == 0U && Rows <= q8_rows::kDenseA8Rows);
    if constexpr (slot_rows) if (cohort_slot && !(parts & 2U) && config.dense_a8_projections && tensors.out.manifest->ggml_type == 8U)
        output_slot = *cohort_slot;

    QsaRowsOutputViews<Rows> out{};
    out.compact = a.workspace.compact_attention_output;
    for (unsigned r = 0; r < Rows; ++r) {
        out.input[r] = prepared.row[r].workspace.attention_output;
        out.output[r] = reinterpret_cast<std::uint16_t*>(prepared.row[r].output.address);
    }
    if constexpr (slot_rows) if (output_slot.codes) {
        QsaOutputRowsSite<Rows> site{};
        site.segments = q8_rows::segments({{reinterpret_cast<const void*>(tensors.out.runtime->address), a.hidden}});
        site.segments.types[0] = q8_rows::segment_type(tensors.out.manifest->ggml_type);  // GSQ-RCO typed (0 = Q8_0)
        for (unsigned r = 0; r < Rows; ++r) { site.input[r] = out.input[r]; site.output[r] = out.output[r]; }
        site.width = a.attention_width; site.outputs = a.hidden;
        q8_rows::launch_dense_a8_quantized(site, output_slot, stream);
        check_hip(hipGetLastError(), "QSA known-window dense A8 output projection launch");
        return;
    }
    dispatch_weight_kind(tensors.out.manifest->ggml_type, [&](auto kind) {
        launch_qsa_rows_output<decltype(kind)::value, Rows>(config.wave8_q8_output, config.dense_a8_projections, out,
            reinterpret_cast<const void*>(tensors.out.runtime->address), a.attention_width, a.hidden, stream, config.dense_f16_prefill, config.dense_g64_prefill);
    });
    check_hip(hipGetLastError(), "QSA known-window output projection launch");
}

constexpr std::uint32_t kQsaTailLayers = 64;
struct QsaTailLayers {
    std::uint16_t* raw[kQsaTailLayers];
    std::uint32_t layer[kQsaTailLayers];
};
__global__ void qsa_index_tail_snapshot_kernel(QsaTailLayers layers, std::uint16_t* journal, const std::uint64_t* position,
    std::uint64_t context_tokens, std::uint32_t index_dim, std::uint32_t ratio, std::uint64_t layer_elements) {
    const auto p = *position;
    if (p >= context_tokens) return;
    const auto start = p - p % ratio;
    const auto count = static_cast<std::uint32_t>(p % ratio) * index_dim;
    const auto* raw = layers.raw[blockIdx.x] + start * index_dim;
    auto* tail = journal + layers.layer[blockIdx.x] * layer_elements;
    for (std::uint32_t e = threadIdx.x; e < count; e += blockDim.x) tail[e] = raw[e];
}
__global__ void qsa_index_tail_commit_kernel(QsaTailLayers layers, const std::uint16_t* journal, const std::uint64_t* position,
    const std::uint32_t* n_keep, std::uint64_t context_tokens, std::uint32_t index_dim, std::uint32_t ratio, std::uint64_t layer_elements) {
    const auto p = *position;
    const auto keep = *n_keep;
    if (p >= context_tokens || keep > 8U || keep > context_tokens - p) return;
    const auto start = p - p % ratio;
    const auto tail_count = static_cast<std::uint32_t>(p % ratio) * index_dim;
    auto* raw = layers.raw[blockIdx.x];
    const auto* tail = journal + layers.layer[blockIdx.x] * layer_elements;
    const auto* keys = tail + static_cast<std::uint64_t>(ratio - 1U) * index_dim;
    for (std::uint32_t e = threadIdx.x; e < tail_count; e += blockDim.x) raw[start * index_dim + e] = tail[e];
    for (std::uint32_t e = threadIdx.x; e < keep * index_dim; e += blockDim.x) raw[p * index_dim + e] = keys[e];
}

std::string kernel_id(
    const ExecutionBindingRequest& request,
    const HipIndexedAttentionLauncherConfig& config) {
    return
        "hip.indexed_attention.bf16.h" +
        std::to_string(
            config.model.hidden_size) +
        ".q" +
        std::to_string(
            config.model.attention_heads) +
        "x" +
        std::to_string(
            config.model.head_dim) +
        ".kv" +
        std::to_string(
            config.model.kv_heads) +
        ".idx" +
        std::to_string(
            config.model.indexer_n_heads) +
        "x" +
        std::to_string(
            config.model.indexer_head_dim) +
        ".b" +
        std::to_string(
            config.model.indexer_budget) +
        ".r" +
        std::to_string(
            request.logical_rows) +
        ".c" +
        std::to_string(
            request.request_concurrency) + (request.device_graph_inputs ? ".device" : ".scalar") +
        std::string(qsa_gqa_organization_marker(config.qsa_gqa_heads_per_block)) +
        (config.qsa_hierarchical_selection ? std::string(qsa_hierarchical_selection_marker) : "") +
        (config.wave8_q8_output ? std::string(q8_projection_organization::qsa_output_wave8_marker) : "") +
        (config.dense_a8_projections ? std::string(q8_projection_organization::dense_a8_marker) : "") +
        (config.dense_trellis_projections ? ".trellis-g1-fp32-s4-v1" : "") +
        (config.prefetch_q8_input ? std::string(qsa_input_prefetch::kernel_suffix) : "");
}

std::uint64_t implementation_identity(
    const HipIndexedAttentionLauncherConfig& config) noexcept {
    std::uint64_t hash =
        fnv_offset;
    hash = append_rotary_identity(hash, config.model.rotary_identity());
    hash_u64(hash, kLauncherTag);
    hash_u64(
        hash,
        config.model.hidden_size);
    hash_u64(
        hash,
        config.model.attention_heads);
    hash_u64(
        hash,
        config.model.kv_heads);
    hash_u64(
        hash,
        config.model.head_dim);
    hash_u64(
        hash,
        config.model.indexer_n_heads);
    hash_u64(
        hash,
        config.model.indexer_kv_heads);
    hash_u64(
        hash,
        config.model.indexer_head_dim);
    hash_u64(
        hash,
        config.model.indexer_budget);
    hash_u64(
        hash,
        config.model.indexer_compress_ratio);
    hash_u64(
        hash,
        config.model.rotary_dim());
    hash_float(
        hash,
        config.model.rope_theta);
    hash_float(
        hash,
        config.model.rms_norm_epsilon);
    hash_u64(hash, config.enable_device_graph_inputs);
    hash_u64(hash, config.model.max_position_embeddings);
    if(config.prefetch_q8_input){hash_u64(hash,0x5153415052465631ULL);for(auto layer:config.prefetch_eligible_layers)hash_u64(hash,layer);}
    if(config.wave8_q8_output) hash_u64(hash, 0x5153414f57385631ULL);
    if(config.dense_a8_projections) hash_u64(hash, 0x5153414441385631ULL); // QSADA8V1
    if(config.dense_trellis_projections) hash_u64(hash, 0x544c473146503453ULL); // TLG1FP4S
    if(config.qsa_hierarchical_selection) hash_u64(hash, 0x515341544f503531ULL);
    if(config.qsa_gqa_heads_per_block != 1) {
        hash_u64(hash, 0x5153414751417631ULL); hash_u64(hash, config.qsa_gqa_heads_per_block);
    }
    if(config.model.attention_kv_storage != AttentionKvStorage::bf16) {
        hash_u64(hash, 0x4b564938544f4b31ULL); hash_u64(hash, static_cast<std::uint32_t>(config.model.attention_kv_storage));
    }
    return hash;
}

}  // namespace

bool qsa_prefill_bundle_closed() noexcept {
    const auto& select=qpf2k_select_batch();const auto& index=qpf2k_index_batch();const auto& attend=qsa_fast_batch();
    return !select.active && select.rows.empty() && select.traced.empty() && !index.active && index.groups.empty() &&
        !attend.active && attend.groups.empty();
}


namespace {
// Prefill: every 8-row tile is admitted like a window call. Input and output projections run
// 64 rows per launch (R1-exact wide q8_rows); the attention part runs tile by tile in increasing
// order, so each tile reads the KV the earlier tiles appended.
std::vector<QsaKnownRows<8U>> prepare_qsa_prefill_rows(std::span<const ExecutionDispatchContext> rows,
    const HipIndexedAttentionLauncherConfig& config, HipQsaIndexJournal journal) {
    if (rows.size() < 16U || rows.size() % 8U) throw std::invalid_argument("QSA prefill rows require 8-row tiles");
    std::vector<QsaKnownRows<8U>> tiles;
    tiles.reserve(rows.size() / 8U);
    // STRIX_QSA_PREFILL_SHARED_ADMISSION=0: every row runs the full serial admission.
    static const bool shared = [] { const char* v = std::getenv("STRIX_QSA_PREFILL_SHARED_ADMISSION"); return !v || v[0] != '0'; }();
    for (std::size_t t = 0; t < rows.size() / 8U; ++t) {
        const QsaHeadRow head = !shared ? QsaHeadRow{} : t ? QsaHeadRow{&rows[0], &tiles.front().row[0], true} : QsaHeadRow{nullptr, nullptr, true};
        tiles.push_back(prepare_qsa_known_rows<8U>(rows.subspan(t * 8U, 8U), config, journal, head, true));
        if(t==0 && !tiles.back().row[0].workspace_layout.block_scores_bytes) {
            if(config.model.indexer_compress_ratio!=4) throw std::invalid_argument("QSA scratch requires the admitted ratio-four geometry");
            config.exact_score_owner->validate(config.model.effective_max_positions(),rows[t*8U].memory->allocation_generation);
        }
        if (tiles.back().prefetch != tiles.front().prefetch || tiles.back().journal != tiles.front().journal)
            throw std::invalid_argument("QSA prefill tiles select different organizations");
    }
    // Batched attention runs all rows of a group at once: their temporaries and outputs must be disjoint.
    std::vector<std::pair<std::uintptr_t, std::uint64_t>> owned;
    for (const auto& tile : tiles)
        for (const auto& row : tile.row) {
            owned.emplace_back(row.temporary.address, row.temporary.bytes);
            owned.emplace_back(row.output.address, row.output.bytes);
        }
    std::sort(owned.begin(), owned.end());
    for (std::size_t i = 1; i < owned.size(); ++i)
        if (owned[i - 1].first + owned[i - 1].second > owned[i].first)
            throw std::invalid_argument("QSA prefill rows share temporary or output storage");
    return tiles;
}
// Dense A8 input projections of W prefill rows (W / 8 tiles from `first`) in one WMMA launch, when
// all five input matrices are Q8_0: the same QsaInputRowsSite rows contract as the 64-row group
// (each row's outputs depend only on its own codes and deltas), so the rows get the same bits.
template <unsigned W>
unsigned enqueue_qsa_prefill_dense_inputs(const std::vector<QsaKnownRows<8U>>& tiles, std::size_t first,
    const HipIndexedAttentionLauncherConfig& config) {
    const auto& a = tiles[first].row[0];
    const auto& tensors = a.tensors;
    const std::array<const TensorPair*, 5> matrices{&tensors.index_q, &tensors.index_k, &tensors.q, &tensors.k, &tensors.v};
    if (!config.dense_a8_projections) return 0U;
    const std::array<std::uint32_t, 5> outputs{a.index_q_width, a.index_dim, a.q_gate_width, a.kv_width, a.kv_width};
    const std::array<float* WorkspacePointers::*, 5> targets{&WorkspacePointers::index_query, &WorkspacePointers::current_index_key,
        &WorkspacePointers::q_gate, &WorkspacePointers::current_k, &WorkspacePointers::current_v};
    auto site = std::make_unique<QsaInputRowsSite<W>>();
    unsigned mask = 0U;
    for (unsigned m = 0; m < 5U; ++m) {
        if (matrices[m]->manifest->ggml_type != 8U) continue;
        const auto g = site->segments.count++;
        site->segments.weights[g] = reinterpret_cast<const unsigned char*>(matrices[m]->runtime->address);
        site->segments.first[g + 1U] = site->segments.first[g] + outputs[m];
        for (unsigned r = 0; r < W; ++r) site->output[g][r] = tiles[first + r / 8U].row[r % 8U].workspace.*targets[m];
        mask |= 1U << m;
    }
    if (!mask) return 0U;
    for (unsigned r = 0; r < W; ++r) site->input[r] = reinterpret_cast<const void*>(tiles[first + r / 8U].row[r % 8U].input.address);
    site->width = a.hidden; site->outputs = site->segments.first[site->segments.count];
    q8_rows::launch_prefill(*site, true, config.dense_f16_prefill, hip_execution_stream(), false, config.dense_g64_prefill);
    check_hip(hipGetLastError(), "QSA prefill dense A8 input projection launch");
    return mask;
}
template <unsigned W>
void enqueue_qsa_prefill_projection(const std::vector<QsaKnownRows<8U>>& tiles, std::size_t first,
    const HipIndexedAttentionLauncherConfig& config, unsigned part, QsaPrefillPool* pool = nullptr, unsigned skip_inputs = 0U,
    const qsa_flash::ScoreGroup<W>* staged = nullptr, bool cache_only = false) {
    auto combined = std::make_unique<QsaKnownRows<W>>();
    combined->prefetch = tiles[first].prefetch; combined->journal = tiles[first].journal;
    for (unsigned r = 0; r < W; ++r) {
        const auto& tile = tiles[first + r / 8U];
        combined->row[r] = tile.row[r % 8U]; combined->selection[r] = tile.selection[r % 8U];
        combined->position[r] = tile.position[r % 8U];
    }
    enqueue_qsa_known_rows<W>(*combined, config, part, pool, skip_inputs, nullptr, 0U, cache_only ? 3U : 127U, staged, false, cache_only);
}
// Rows r of [0, rows) at view(row 0) + r * stride, or false.
template <class View> bool qsa_prefill_uniform(const std::vector<QsaKnownRows<8U>>& tiles, std::uint32_t rows, View&& view, std::uint64_t& stride) {
    const auto at = [&](std::uint32_t r) { return reinterpret_cast<std::uintptr_t>(view(tiles[r / 8U].row[r % 8U])); };
    const auto a0 = at(0U), a1 = at(1U);
    if (a1 <= a0) return false;
    stride = a1 - a0;
    for (std::uint32_t r = 2U; r < rows; ++r) if (at(r) != a0 + r * stride) return false;
    return true;
}
// STRIX_QPF2K_DENSE=1, part 1: the Q8_0 input matrices of the rows the 256/128-row launches of projections(1) would take
// (FP16 when dense_f16_prefill and not g64) in one tiled launch; then the 64-row groups of those rows run the other
// (BF16 indexer) matrices with the Q8_0 ones skipped, as after the 256-row launches. Returns the rows done (0: not taken).
std::uint32_t enqueue_qsa_prefill_tiled_inputs(const std::vector<QsaKnownRows<8U>>& tiles, const HipIndexedAttentionLauncherConfig& config) {
    if (!q8_rows::tiled::site_enabled(4U) || !config.dense_a8_projections || !config.dense_f16_prefill || config.dense_g64_prefill) return 0U;
    const auto n = static_cast<std::uint32_t>(tiles.size() * 8U);
    const std::uint32_t rows = n / 256U * 256U + (n % 256U >= 128U ? 128U : 0U);
    if (rows < q8_rows::tiled::min_rows() || rows < 128U) return 0U;
    const auto& a = tiles[0].row[0];
    const auto& tensors = a.tensors;
    const std::array<const TensorPair*, 5> matrices{&tensors.index_q, &tensors.index_k, &tensors.q, &tensors.k, &tensors.v};
    const std::array<std::uint32_t, 5> outputs{a.index_q_width, a.index_dim, a.q_gate_width, a.kv_width, a.kv_width};
    const std::array<float* WorkspacePointers::*, 5> targets{&WorkspacePointers::index_query, &WorkspacePointers::current_index_key,
        &WorkspacePointers::q_gate, &WorkspacePointers::current_k, &WorkspacePointers::current_v};
    QsaInputTiledSite site{};
    std::uint64_t si = 0, so = 0;
    if (!qsa_prefill_uniform(tiles, rows, [](const auto& row) { return reinterpret_cast<const void*>(row.input.address); }, si)) return 0U;
    unsigned mask = 0U;
    for (unsigned m = 0; m < 5U; ++m) {
        if (matrices[m]->manifest->ggml_type != 8U) continue;
        std::uint64_t s = 0;
        const auto target = targets[m];
        if (!qsa_prefill_uniform(tiles, rows, [target](const auto& row) { return static_cast<const void*>(row.workspace.*target); }, s) ||
            (so && s != so))
            return 0U;
        so = s;
        const auto g = site.segments.count++;
        site.segments.weights[g] = reinterpret_cast<const unsigned char*>(matrices[m]->runtime->address);
        site.segments.first[g + 1U] = site.segments.first[g] + outputs[m];
        site.output0[g] = reinterpret_cast<unsigned char*>(a.workspace.*target);
        mask |= 1U << m;
    }
    if (!mask) return 0U;
    site.input0 = reinterpret_cast<const unsigned char*>(a.input.address); site.input_stride = si; site.output_stride = so;
    site.width = a.hidden; site.outputs = site.segments.first[site.segments.count];
    if (!q8_rows::tiled::supported(site, rows)) return 0U;
    q8_rows::tiled::launch(site, rows, hip_execution_stream());
    check_hip(hipGetLastError(), "QSA prefill tiled input projection launch");
    if (mask != 0x1fU)
        for (std::uint32_t r = 0; r < rows; r += 64U) enqueue_qsa_prefill_projection<64U>(tiles, r / 8U, config, 1U, nullptr, mask);
    return rows;
}
// STRIX_QPF2K_DENSE=1, part 4: the output projection of the rows projections(4) gives to its >= 32-row groups (FP16 when
// the out matrix is Q8_0, wave8 or dense A8, dense_f16_prefill and not g64) in one tiled launch. Returns the rows done.
template <class InputType>
std::uint32_t enqueue_qsa_prefill_tiled_output_storage(const std::vector<QsaKnownRows<8U>>& tiles, const HipIndexedAttentionLauncherConfig& config) {
    if (!q8_rows::tiled::site_enabled(8U) || !(config.wave8_q8_output || config.dense_a8_projections) || !config.dense_f16_prefill ||
        config.dense_g64_prefill)
        return 0U;
    const auto& a = tiles[0].row[0];
    if (a.tensors.out.manifest->ggml_type != 8U) return 0U;
    const std::uint32_t rows = static_cast<std::uint32_t>(tiles.size() * 8U) / 32U * 32U;
    if (rows < q8_rows::tiled::min_rows() || rows < 32U) return 0U;
    std::uint64_t si = 0, so = 0;
    if (!qsa_prefill_uniform(tiles, rows, [](const auto& row) { return static_cast<const void*>(row.workspace.attention_output); }, si) ||
        !qsa_prefill_uniform(tiles, rows, [](const auto& row) { return reinterpret_cast<const void*>(row.output.address); }, so))
        return 0U;
    QsaOutputTiledSite<InputType> site{};
    site.segments = q8_rows::segments({{reinterpret_cast<const void*>(a.tensors.out.runtime->address), a.hidden}});
    site.width = a.attention_width; site.outputs = a.hidden;
    site.input_stride = si; site.output_stride = so;
    // The attention outputs sit in per-row temporaries (stride = a row's whole QSA workspace, which grows with the
    // context): the tiled kernel addresses its A rows with 32-bit offsets, so long prompts go in chunks of rows whose
    // input span stays below 4 GiB (t9: at T 8192 the single launch was unsupported and every row fell back to the
    // 64-row launches). A row's value does not depend on the chunk it lands in.
    const auto* const in0 = reinterpret_cast<const unsigned char*>(a.workspace.attention_output);
    auto* const out0 = reinterpret_cast<unsigned char*>(a.output.address);
    const std::uint64_t row_bytes = static_cast<std::uint64_t>(a.attention_width) * sizeof(InputType);
    if (row_bytes >= (1ULL << 32U)) return 0U;
    std::uint64_t chunk = ((1ULL << 32U) - 1U - row_bytes) / si + 1U;  // rows n with (n - 1) * si + row_bytes < 2^32
    if (chunk < rows) chunk = chunk / 256U * 256U;
    chunk = std::min<std::uint64_t>(chunk, rows);
    static const bool trace = [] { const char* v = std::getenv("STRIX_QPF2K_DENSE_TRACE"); return v && v[0] == '1'; }();
    static std::atomic<unsigned> traced{0};
    if (trace && traced.fetch_add(1U) == 0U)
        std::fprintf(stderr, "qpf2k dense QSA out rows %u input stride %llu output stride %llu chunk %llu\n", rows,
            static_cast<unsigned long long>(si), static_cast<unsigned long long>(so), static_cast<unsigned long long>(chunk));
    if (chunk < 256U && chunk < rows) return 0U;
    const auto last = static_cast<std::uint32_t>(rows - (rows - 1U) / chunk * chunk);
    site.input0 = in0; site.output0 = out0;
    if (!q8_rows::tiled::supported(site, static_cast<std::uint32_t>(chunk)) || !q8_rows::tiled::supported(site, last)) return 0U;
    for (std::uint32_t r0 = 0; r0 < rows;) {
        const auto n = static_cast<std::uint32_t>(std::min<std::uint64_t>(chunk, rows - r0));
        site.input0 = in0 + static_cast<std::uint64_t>(r0) * si; site.output0 = out0 + static_cast<std::uint64_t>(r0) * so;
        q8_rows::tiled::launch(site, n, hip_execution_stream());
        check_hip(hipGetLastError(), "QSA prefill tiled output projection launch");
        r0 += n;
    }
    return rows;
}
std::uint32_t enqueue_qsa_prefill_tiled_output(const std::vector<QsaKnownRows<8U>>& tiles, const HipIndexedAttentionLauncherConfig& config) {
    return tiles[0].row[0].workspace.compact_attention_output ?
        enqueue_qsa_prefill_tiled_output_storage<std::uint16_t>(tiles,config) :
        enqueue_qsa_prefill_tiled_output_storage<float>(tiles,config);
}
// Attention outputs of every prefill row (self-check).
std::vector<std::uint8_t> qsa_prefill_attention_outputs(const std::vector<QsaKnownRows<8U>>& tiles) {
    check_hip(hipStreamSynchronize(hip_execution_stream()), "QSA prefill self-check sync");
    std::vector<std::uint8_t> bytes;
    for (const auto& tile : tiles)
        for (const auto& row : tile.row) {
            const auto size = static_cast<std::size_t>(row.attention_width) * sizeof(float);
            const auto at = bytes.size();
            bytes.resize(at + size);
            if (row.workspace.compact_attention_output) {
                std::vector<std::uint16_t> compact(row.attention_width);
                check_hip(hipMemcpy(compact.data(), row.workspace.attention_output, compact.size()*2U, hipMemcpyDeviceToHost), "QSA compact self-check copy");
                for (std::size_t i=0;i<compact.size();++i) {
                    const auto bits=widen_compact_bits(compact[i]);
                    std::memcpy(bytes.data()+at+i*4U,&bits,4U);
                }
            } else check_hip(hipMemcpy(bytes.data() + at, row.workspace.attention_output, size, hipMemcpyDeviceToHost), "QSA prefill self-check copy");
        }
    return bytes;
}
void qpf2k_select_flush(hipStream_t stream) {
    auto& sb = qpf2k_select_batch();
    if (!sb.rows.empty()) {
        struct Slot { qsa_flash::SelectRow* rows = nullptr; std::size_t capacity = 0; hipEvent_t done = nullptr; bool pending = false; };
        static std::array<Slot, 8> ring;
        static unsigned next = 0;
        auto& slot = ring[next];
        next = (next + 1U) % ring.size();
        if (slot.pending) check_hip(hipEventSynchronize(slot.done), "QSA select batch slot wait");
        if (!slot.done) check_hip(hipEventCreateWithFlags(&slot.done, hipEventDisableTiming), "QSA select batch event");
        if (sb.rows.size() > slot.capacity) {
            if (slot.rows) check_hip(hipHostFree(slot.rows), "QSA select batch slot free");
            slot.rows = nullptr;
            slot.capacity = std::max<std::size_t>(sb.rows.size(), 8192U);
            check_hip(hipHostMalloc(reinterpret_cast<void**>(&slot.rows), slot.capacity * sizeof(qsa_flash::SelectRow), hipHostMallocDefault),
                "QSA select batch slot allocation");
        }
        std::copy(sb.rows.begin(), sb.rows.end(), slot.rows);
        hipLaunchKernelGGL(qsa_flash::select_batch_kernel, dim3(static_cast<std::uint32_t>(sb.rows.size())), dim3(kSelectThreads), 0, stream,
            slot.rows, sb.ratio, sb.topk);
        qpf2k_launch_select_batch.fetch_add(1U, std::memory_order_relaxed);
        qpf2k_select_batch_rows.fetch_add(sb.rows.size(), std::memory_order_relaxed);
        check_hip(hipGetLastError(), "QSA select batch launch");
        check_hip(hipEventRecord(slot.done, stream), "QSA select batch slot record");
        slot.pending = true;
        sb.rows.clear();
    }
    for (const auto& g : sb.traced) {
        std::array<std::uint64_t, 64> position = g.position;
        qpf2k_trace_rows<64U>(1U, position, [&](unsigned r) { return r < g.rows ? g.selected[r] : g.selected[0]; },
            [&](unsigned r) { return r < g.rows ? g.count[r] : 0U; }, stream);
        qpf2k_trace_rows<64U>(2U, position, [&](unsigned r) { return r < g.rows ? g.scores[r] : g.scores[0]; },
            [&](unsigned r) { return r < g.rows ? g.blocks[r] : 0U; }, stream);
    }
    sb.traced.clear();
    sb.used = 0;
}
struct QsaExactScoreLease {
    QsaScoreScratch<HipQsaScratchBackend>& ring;std::uint32_t slot;
    explicit QsaExactScoreLease(QsaScoreScratch<HipQsaScratchBackend>& r):ring(r),slot(r.acquire()){}
    ~QsaExactScoreLease() {
        qsa_exact_maxima=nullptr;qsa_exact_maxima_stride=0;
        ring.consumed(slot);
    }
};
#include "hip_qsa_prefill_batch.inc"

// Prefill attention: groups of up to 8 tiles (64 rows) in increasing order. A group stores the
// raw index keys and K/V of all its rows before any of them selects or attends. A row's
// selection is fixed by its own position (prepare_qsa_known_rows: visible prefix and
// complete blocks), and every kernel after the store reads raw keys / KV only at positions
// inside that prefix: the same values the serial tile order shows it (positions of later
// rows are written but never read). Every per-row computation is the window kernel's, with
// the row's own count and blocks, so the outputs are bitwise those of the serial tiles.
// The journal's 8 row keys end as the last tile's, as serially. STRIX_QSA_PREFILL_TILES=1
// runs the serial tiles; STRIX_QSA_PREFILL_SELFCHECK=1 runs both (re-running the input
// projections in between) and compares every row's attention output bitwise, throwing on a
// difference; =2 compares the serial tiles with themselves (checks the harness).
void enqueue_qsa_prefill_rows(const std::vector<QsaKnownRows<8U>>& tiles, const HipIndexedAttentionLauncherConfig& config,
    bool cache_only_step = false) {
    const auto submission=qsa_scratch_gate().enter(HipQsaScratchBackend{}.device());
    const auto submission_stream=hip_execution_stream();
    auto& shared=qsa_scratch_gate();
    shared.handoff.before(reinterpret_cast<std::uintptr_t>(submission_stream),[&] {
        check_hip(hipEventSynchronize(shared.done),"QSA prefill cross-stream scratch wait");
    });
    const auto exceptions=std::uncaught_exceptions();
    QsaPendingGuard pending([&] {
        // An exceptional flush may have launched a consumer before recording its
        // descriptor event. Drain before dropping descriptors or unlocking.
        if(std::uncaught_exceptions()>exceptions) {
            check_hip(hipDeviceSynchronize(),"QSA prefill exceptional shared scratch drain");
            shared.handoff.drained();
        } else {
            if(hipEventRecord(shared.done,submission_stream)!=hipSuccess) {
                check_hip(hipDeviceSynchronize(),"QSA prefill failed handoff drain");
                shared.handoff.drained();
            } else shared.handoff.recorded(reinterpret_cast<std::uintptr_t>(submission_stream));
        }
    }, [] {
        clear_qsa_pending(qsa_fast_batch(),qpf2k_select_batch(),qpf2k_index_batch(),qsa_exact_maxima,qsa_exact_maxima_stride);
    });
    static const unsigned mode = [] {
        if (const char* v = std::getenv("STRIX_QSA_PREFILL_SELFCHECK"); v && v[0] == '1') return 2U;
        if (const char* v = std::getenv("STRIX_QSA_PREFILL_SELFCHECK"); v && v[0] == '2') return 3U;
        if (const char* v = std::getenv("STRIX_QSA_PREFILL_TILES"); v && v[0] == '1') return 1U;
        return 0U;
    }();
    // STRIX_QSA_CACHE_ONLY (see strix/qsa_cache_only.hpp): the input projections exactly as below (their launch routes
    // depend on the full matrix set), then per group only the k norm/rope and the raw index-key / KV append.
    const bool cache_only = cache_only_step && mode == 0U;
    static_cast<void>(qsa_group_rope_table());
    // Pooled block keys (STRIX_QSA_PREFILL_POOLED=0 keeps block_score_rows_kernel): one device
    // buffer for the context's blocks, reused by every layer's call on the execution stream.
    static const bool pooled = [] { const char* v = std::getenv("STRIX_QSA_PREFILL_POOLED"); return !v || v[0] != '0'; }();
    auto& pool_mutex=qsa_prefill_pool_mutex();
    auto& pool_storage=qsa_prefill_pool_storage();
    std::unique_lock<std::mutex> pool_lock(pool_mutex, std::defer_lock);
    QsaPrefillPool* pool = nullptr;
    if (pooled) {
        pool_lock.lock();
        const auto& a = tiles[0].row[0];
        const auto ratio = tiles[0].selection[0].ratio;
        const auto tokens = a.index_dim ? a.raw_state.bytes / (static_cast<std::uint64_t>(a.index_dim) * sizeof(std::uint16_t)) : 0U;
        const auto blocks = static_cast<std::uint32_t>(std::min<std::uint64_t>(ratio ? tokens / ratio : 0U, UINT32_MAX));
        if (blocks > pool_storage.capacity && a.index_dim == 128U) {
            if (pool_storage.keys) {
                check_hip(hipStreamSynchronize(hip_execution_stream()), "QSA prefill block-pool resize sync");
                check_hip(hipFree(pool_storage.keys), "QSA prefill block-pool free");
                if (pool_storage.packed) check_hip(hipFree(pool_storage.packed), "QSA prefill packed block-pool free");
                pool_storage = {};
            }
            void* keys = nullptr;
            check_hip(hipMalloc(&keys, static_cast<std::uint64_t>(blocks) * a.index_dim * sizeof(float)), "QSA prefill block-pool allocation");
            pool_storage.keys = static_cast<float*>(keys);
            if (qsa_pool_packed_enabled()) {
                void* packed = nullptr;
                check_hip(hipMalloc(&packed, static_cast<std::uint64_t>(blocks) * a.index_dim * sizeof(std::uint16_t)),
                    "QSA prefill packed block-pool allocation");
                pool_storage.packed = static_cast<std::uint16_t*>(packed);
            }
            pool_storage.capacity = blocks;
        }
        pool_storage.valid = 0;
        pool = &pool_storage;
    }
    const auto projections = [&](unsigned part, const Qpf2kStageBatch* stages = nullptr) {
        for (std::size_t t = 0; t < tiles.size();) {
            const auto left = tiles.size() - t;
            // STRIX_PREFILL_DENSE_A8_ROWS=64 keeps the dense A8 input projections at 64 rows per launch.
            static const bool dense_wide = [] { const char* v = std::getenv("STRIX_PREFILL_DENSE_A8_ROWS"); return !v || std::atoi(v) != 64; }();
            // The Q8_0 input matrices of 256 (128) rows in one launch; the others keep their 64-row groups.
            // Part 1 only (the pool is part 2's).
            const auto wide_inputs = [&]<unsigned W>() {
                const auto mask = enqueue_qsa_prefill_dense_inputs<W>(tiles, t, config);
                if (!mask) return false;
                if (mask != 0x1fU)
                    for (unsigned g = 0; g < W / 64U; ++g) enqueue_qsa_prefill_projection<64U>(tiles, t + 8U * g, config, part, nullptr, mask);
                t += W / 8U;
                return true;
            };
            // STRIX_QPF2K_DENSE=1: one tiled launch for the rows of the 256/128-row input launches / >= 32-row output groups.
            if (t == 0U && part == 1U && dense_wide)
                if (const auto done = enqueue_qsa_prefill_tiled_inputs(tiles, config)) { t = done / 8U; continue; }
            if (t == 0U && part == 4U)
                if (const auto done = enqueue_qsa_prefill_tiled_output(tiles, config)) { t = done / 8U; continue; }
            if (part == 1U && dense_wide && left >= 32U && wide_inputs.template operator()<256U>()) continue;
            if (part == 1U && dense_wide && left >= 16U && wide_inputs.template operator()<128U>()) continue;
            // Cache-only part 2: no pool, staged scores or score source (the groups stop after their appends).
            const bool stores = cache_only && part == 2U;
            QsaPrefillPool* const group_pool = part == 2U && !stores ? pool : nullptr;
            if (left >= 8U) {
                enqueue_qsa_prefill_projection<64U>(tiles, t, config, part, group_pool, 0U,
                    stages && !stores ? stages->slot->scores + t / 8U : nullptr, stores);
                t += 8U;
            }
            else if (left >= 4U) { enqueue_qsa_prefill_projection<32U>(tiles, t, config, part, group_pool, 0U, nullptr, stores); t += 4U; }
            else if (left >= 2U) { enqueue_qsa_prefill_projection<16U>(tiles, t, config, part, group_pool, 0U, nullptr, stores); t += 2U; }
            else {
                enqueue_qsa_known_rows<8U>(tiles[t], config, part, nullptr, 0U, nullptr, 0U, stores ? 3U : 127U, nullptr, false, stores);
                t += 1U;
            }
        }
    };
    {
        // The batch is active only inside this block; the guard resets it on any exit (an exception drops unlaunched views).
        Qpf2kIndexScope guard(mode, config.qsa_fast_prefill);
        auto& ib = qpf2k_index_batch();
        const bool side = ib.active && !qpf2k_index_batch_enabled() && !ib.wmma && (qpf2k_qsa_mask() & 0x100U);
        const auto main_stream = hip_execution_stream();
        if (side) check_hip(hipEventRecord(qpf2k_index_side().fork, main_stream), "QSA index side fork");
        projections(1U);
        if (side && !ib.groups.empty()) {
            auto& x = qpf2k_index_side();
            check_hip(hipStreamWaitEvent(x.stream, x.fork, 0), "QSA index side wait fork");
            qpf2k_index_flush(x.stream);
            check_hip(hipEventRecord(x.join, x.stream), "QSA index side join");
            check_hip(hipStreamWaitEvent(main_stream, x.join, 0), "QSA index side wait join");
            qpf2k_launch_index_side.fetch_add(1U, std::memory_order_relaxed);
        }
        qpf2k_index_flush(main_stream);
    }
    if (cache_only) {
        projections(2U);
        static std::atomic<unsigned> reported{0};
        if (reported.fetch_add(1U) == 0U)
            std::fprintf(stderr, "QSA prefill cache-only: headless last-layer step, %zu rows\n", tiles.size() * 8U);
        return;
    }
    // STRIX_QSA_ATTEND_BATCH=1: every group's union/attend of part 2 in one launch pair (see QsaFastBatch).
    const auto part2 = [&] {
        auto& b = qsa_fast_batch();
        b.active = mode == 0U && qsa_attend_batch_enabled();
        auto& sb = qpf2k_select_batch();
        sb.active = b.active && (qpf2k_qsa_mask() & 16U) && (qpf2k_qsa_mask() & 3U) == 3U;
        const bool compact=!tiles.front().row[0].workspace_layout.block_scores_bytes;
        if(compact) {
            auto& slabs=config.exact_score_owner->scratch;
            const auto& geometry=slabs.geometry();
            const auto stage_rows = checked_u32(tiles.size() * 8U, "QSA stage rows");
            Qpf2kStageBatch stage;
            const bool split = mode == 0U && qpf2k_stage_batch_launch(tiles, config, pool, stage, &geometry);
            const QsaPrefillStagePlan plan(stage_rows,
                qsa_flash::score_rows(split ? stage.max_blocks : 0U), split ? geometry.rows : 64U);
            if (split && qpf2k_trace_enabled())
                std::fprintf(stderr, "qpf2k qsa stage_split slices %u rows %u score_rows %u ring_rows %u ring_bytes %llu\n",
                    plan.slices(), plan.rows, plan.score_rows, geometry.rows,
                    static_cast<unsigned long long>(geometry.total_bytes()));
            // Selection descriptors are flushed within each slice, including its
            // post-selection traces. They never survive a lease's release.
            // Non-stage/int8/tail fallbacks retain immediate per-group selection.
            if (!split) sb.active = false;
            for (std::uint32_t slice = 0; slice < plan.slices(); ++slice) {
                QsaExactScoreLease lease(slabs);
                const auto t = plan.first(slice) / 8U;
                const auto count = plan.count(slice) / 8U;
                std::vector<QsaKnownRows<8U>> group(tiles.begin()+t, tiles.begin()+t+count);
                auto* scores=static_cast<float*>(slabs.data(lease.slot));
                auto* maxima=scores+geometry.score_slot_bytes()/sizeof(float);
                for(std::size_t k=0;k<count;++k) for(unsigned r=0;r<8U;++r)
                    group[k].row[r].workspace.block_scores=scores+(k*8U+r)*geometry.blocks;
                qsa_exact_maxima=maxima;qsa_exact_maxima_stride=static_cast<std::uint32_t>(geometry.maxima_stride);
                if (split) {
                    const auto first_group = plan.first(slice) / 64U;
                    const auto groups = plan.count(slice) / 64U;
                    if (!groups || first_group >= stage.groups || groups > stage.groups - first_group)
                        throw std::invalid_argument("QSA ring slice outside prepared descriptors");
                    for (std::uint32_t g=0;g<groups;++g) {
                        auto& descriptor = stage.slot->scores[first_group+g];
                        if (descriptor.blocks > geometry.blocks || descriptor.maxima_stride > geometry.maxima_stride)
                            throw std::invalid_argument("QSA ring descriptor exceeds slot geometry");
                        for (unsigned r=0;r<64U;++r)
                            descriptor.views.block_scores[r] = group[g*8U+r/8U].row[r%8U].workspace.block_scores;
                        // Retain the group's local stride within its disjoint reservation.
                        descriptor.maxima = descriptor.blocks ? maxima + static_cast<std::uint64_t>(g)*64U*geometry.maxima_stride : nullptr;
                    }
                    qpf2k_stage_score_slice(stage, pool, group[0].row[0].index_dim, first_group, groups);
                    for (std::uint32_t g=0;g<groups;++g)
                        enqueue_qsa_prefill_projection<64U>(group,g*8U,config,2U,pool,0U,&stage.slot->scores[first_group+g]);
                } else if(count==8U) enqueue_qsa_prefill_projection<64U>(group,0U,config,2U,pool);
                else {
                    std::size_t first=0;
                    if(count>=4U) {enqueue_qsa_prefill_projection<32U>(group,first,config,2U,pool);first+=4U;}
                    if(count-first>=2U) {enqueue_qsa_prefill_projection<16U>(group,first,config,2U,pool);first+=2U;}
                    if(first<count) enqueue_qsa_known_rows<8U>(group[first],config,2U);
                }
                qpf2k_select_flush(hip_execution_stream());
                // The lease event follows selection and post-selection hashes.
                // Whole-call attend descriptors retain only selected lists/KV.
            }
            sb.active = false;
            qsa_fast_batch_flush(hip_execution_stream());b.active=false;
            return;
        }
        Qpf2kStageBatch stages;
        const bool stage_batch = mode == 0U && qpf2k_stage_batch_launch(tiles, config, pool, stages);
        if (!stage_batch && (qpf2k_qsa_mask() & 0x200U) && qpf2k_trace_enabled())
            std::fprintf(stderr, "qpf2k qsa stage_batch groups 0 pool 0 range 0 score 0 rope 0 norm 0 store 0 fallback 1\n");
        projections(2U, stage_batch ? &stages : nullptr);
        qpf2k_select_flush(hip_execution_stream());
        sb.active = false;
        qsa_fast_batch_flush(hip_execution_stream());
        b.active = false;
    };
    if (mode == 1U) {
        for (const auto& tile : tiles) enqueue_qsa_known_rows<8U>(tile, config, 2U);
    } else {
        if (mode == 3U) for (const auto& tile : tiles) enqueue_qsa_known_rows<8U>(tile, config, 2U);
        else part2();
        if (mode >= 2U) {
            const auto batched = qsa_prefill_attention_outputs(tiles);
            // The attention part normalises and rotates q / k / index query in place: rerun the
            // input projections so the serial tiles start from the same workspace values.
            projections(1U);
            for (const auto& tile : tiles) enqueue_qsa_known_rows<8U>(tile, config, 2U);
            const auto serial = qsa_prefill_attention_outputs(tiles);
            if (config.qsa_fast_prefill) {
                // The fast (qsa) attention differs from the exact one by design: report its error against
                // the serial tiles (BF16 values) instead of requiring equality.
                const auto count = serial.size() / sizeof(float);
                std::vector<float> x(count), y(count);
                std::memcpy(x.data(), serial.data(), serial.size()); std::memcpy(y.data(), batched.data(), batched.size());
                double sq = 0.0, ref = 0.0, worst = 0.0;
                std::size_t differ = 0, nonfinite = 0;
                for (std::size_t i = 0; i < count; ++i) {
                    if (!std::isfinite(y[i])) { ++nonfinite; continue; }
                    const double d = static_cast<double>(y[i]) - x[i];
                    sq += d * d; ref += static_cast<double>(x[i]) * x[i]; worst = std::max(worst, std::fabs(d));
                    differ += x[i] != y[i];
                }
                static std::atomic<unsigned> calls{0};
                std::fprintf(stderr, "QSA fast self-check %u: %zu rows from position %llu, %zu of %zu values differ, rel rms %.3e, "
                    "max |diff| %.3e (rms %.3e), nonfinite %zu\n", calls.fetch_add(1U), count / tiles[0].row[0].attention_width,
                    static_cast<unsigned long long>(tiles[0].position[0]), differ, count, ref > 0.0 ? std::sqrt(sq / ref) : 0.0, worst,
                    std::sqrt(ref / static_cast<double>(count)), nonfinite);
                if (nonfinite) throw std::runtime_error("QSA fast self-check: non-finite attention output");
            } else if (serial != batched) {
                const auto row_bytes = static_cast<std::size_t>(tiles[0].row[0].attention_width) * sizeof(float);
                std::size_t rows_differing = 0, first = SIZE_MAX;
                for (std::size_t r = 0; r * row_bytes < serial.size(); ++r)
                    if (std::memcmp(serial.data() + r * row_bytes, batched.data() + r * row_bytes, row_bytes)) {
                        ++rows_differing; if (first == SIZE_MAX) first = r;
                    }
                std::fprintf(stderr, "QSA prefill self-check: %zu of %zu rows differ, first row %zu (position %llu), layer %u\n",
                    rows_differing, serial.size() / row_bytes, first,
                    static_cast<unsigned long long>(tiles[first / 8U].position[first % 8U]), 0U);
                throw std::runtime_error("QSA prefill self-check: batched attention differs from the serial tiles");
            }
            static std::atomic<unsigned> reported{0};
            if (reported.fetch_add(1U) == 0U)
                std::fprintf(stderr, "QSA prefill self-check: batched attention equals the serial tiles bitwise\n");
        }
    }
    if (qpf2k_trace_enabled() && mode == 0U) {
        const auto stream = hip_execution_stream();
        for (std::size_t t = 0; t + 8U <= tiles.size(); t += 8U) {
            std::array<std::uint64_t, 64U> position{};
            for (unsigned r = 0; r < 64U; ++r) position[r] = tiles[t + r / 8U].position[r % 8U];
            qpf2k_trace_rows<64U>(3U, position, [&](unsigned r) { return tiles[t + r / 8U].row[r % 8U].workspace.attention_output; },
                [&](unsigned r) { return tiles[t + r / 8U].row[r % 8U].attention_width; }, stream, tiles[0].row[0].workspace.compact_attention_output);
        }
        for (std::size_t t = tiles.size() / 8U * 8U; t < tiles.size(); ++t) {
            std::array<std::uint64_t, 8U> position{};
            for (unsigned r = 0; r < 8U; ++r) position[r] = tiles[t].position[r];
            qpf2k_trace_rows<8U>(3U, position, [&](unsigned r) { return tiles[t].row[r].workspace.attention_output; },
                [&](unsigned r) { return tiles[t].row[r].attention_width; }, stream, tiles[0].row[0].workspace.compact_attention_output);
        }
    }
    projections(4U);
    if (qpf2k_trace_enabled() && mode == 0U) {
        auto& t = qpf2k_trace();
        std::array<unsigned long long, 4> acc{};
        check_hip(hipStreamSynchronize(hip_execution_stream()), "QSA trace sync");
        check_hip(hipMemcpy(acc.data(), t.acc, sizeof(acc), hipMemcpyDeviceToHost), "QSA trace copy");
        std::fprintf(stderr, "qpf2k qsa trace call %u rows %llu scores %016llx selected %016llx post %016llx attn %016llx\n", t.call,
            static_cast<unsigned long long>(t.rows), acc[0], acc[1], acc[2], acc[3]);
        std::fprintf(stderr, "qpf2k qsa launches score %llu select %llu production score %llu select %llu index16 %llu norm_wave %llu select_batch %llu batch_rows %llu early_flush %llu groups %llu %llu %llu %llu shadow %llu shadow_skipped %llu shadow_peak_mib %llu order %llu index_batch %llu index_batch_groups %llu index_side %llu index_wmma %llu\n",
            static_cast<unsigned long long>(qpf2k_launch_score.load()), static_cast<unsigned long long>(qpf2k_launch_select.load()),
            static_cast<unsigned long long>(qpf2k_launch_prod_score.load()), static_cast<unsigned long long>(qpf2k_launch_prod_select.load()),
            static_cast<unsigned long long>(qpf2k_launch_index16.load()), static_cast<unsigned long long>(qpf2k_launch_norm_wave.load()),
            static_cast<unsigned long long>(qpf2k_launch_select_batch.load()), static_cast<unsigned long long>(qpf2k_select_batch_rows.load()),
            static_cast<unsigned long long>(qpf2k_select_early_flush.load()),
            static_cast<unsigned long long>(qpf2k_groups_by_rows[0].load()), static_cast<unsigned long long>(qpf2k_groups_by_rows[1].load()),
            static_cast<unsigned long long>(qpf2k_groups_by_rows[2].load()), static_cast<unsigned long long>(qpf2k_groups_by_rows[3].load()),
            static_cast<unsigned long long>(qpf2k_launch_shadow.load()), static_cast<unsigned long long>(qpf2k_shadow_skipped.load()),
            static_cast<unsigned long long>(qpf2k_shadow_peak_bytes.load() >> 20U),
            static_cast<unsigned long long>(qpf2k_launch_order.load()),
            static_cast<unsigned long long>(qpf2k_launch_index_batch.load()), static_cast<unsigned long long>(qpf2k_index_batch_groups.load()),
            static_cast<unsigned long long>(qpf2k_launch_index_side.load()),
            static_cast<unsigned long long>(qpf2k_launch_index_wmma.load()));
        ++t.call;
    }
}
}  // namespace

void reserve_indexed_attention_prefill_pools(std::uint32_t rows, std::uint64_t context_tokens,
    std::uint32_t ratio, bool compact_scores) {
    qpf2k_stage_setup(rows, context_tokens, ratio, compact_scores);
}

void indexed_attention_known_prefill_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipIndexedAttentionLauncherConfig& config, HipQsaIndexJournal journal) {
    static_cast<void>(prepare_qsa_prefill_rows(rows, config, journal));
}
void indexed_attention_known_prefill_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipIndexedAttentionLauncherConfig& config, HipQsaIndexJournal journal) {
#if defined(STRIX_HAS_EXPERIMENTAL_DENSE_TRELLIS)
    HipDenseTrellisScope arithmetic(config.dense_trellis_projections);
#endif
    enqueue_qsa_prefill_rows(prepare_qsa_prefill_rows(rows, config, journal), config, qsa_cache_only::active);
}

void indexed_attention_known_window_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipIndexedAttentionLauncherConfig& config, HipQsaIndexJournal journal) {
    if (rows.size() == 2U) static_cast<void>(prepare_qsa_known_rows<2U>(rows, config, journal));
    else if (rows.size() == 4U) static_cast<void>(prepare_qsa_known_rows<4U>(rows, config, journal));
    else if (rows.size() == 8U) static_cast<void>(prepare_qsa_known_rows<8U>(rows, config, journal));
    else throw std::invalid_argument("QSA known-window requires exactly two, four or eight rows");
}
// This cache belongs to one eager owner, never to an address-keyed process singleton.
// The committed prefix is immutable until the owner's explicit discard/reset.
class HipQsaWindowPool {
    // false: per-entry allocations whose failure returns no pool (the serial decode pool).
    bool slab_allowed_ = true;
    bool bf16_keys_ = false; // serial decode only; prefill allocation and slabs stay FP32
    struct Entry {
        std::uintptr_t raw = 0, norm = 0;
        std::uint64_t raw_generation = 0, norm_generation = 0;
        QsaPrefillPool pool;
    };
    std::vector<Entry> entries_;
    // STRIX_QSA_POOL_SLAB (ON by default, =0 off): entries are carved from slabs of kSlabEntries same-size entries (one device
    // allocation per owner for every QSA layer instead of one per layer, made at the owner's first window).
    // Bits do not change (same keys, same kernels); only the allocation pattern during decode does.
    static constexpr std::uint32_t kSlabEntries = 16U;
    struct Slab { void* base = nullptr; std::uint64_t entry_bytes = 0; std::uint32_t used = 0; };
    std::vector<Slab> slabs_;
    // Serial decode pool only: one row of 4-block group maxima (qsa_decode_pool_select), reused by every layer and step on
    // the owner's stream; grown with a stream drain, freed at teardown.
    float* maxima_ = nullptr;
    std::uint32_t maxima_capacity_ = 0;
    hipStream_t stream_ = nullptr;
    int device_ = -1;
    static bool slab_enabled() { static const bool on = qsa_env_default_on("STRIX_QSA_POOL_SLAB"); return on; }
public:
    HipQsaWindowPool() = default;
    explicit HipQsaWindowPool(bool slab_allowed, bool bf16_keys) : slab_allowed_(slab_allowed), bf16_keys_(bf16_keys) {}
    ~HipQsaWindowPool() {
        if (maxima_) {
            if (hipStreamSynchronize(stream_) != hipSuccess || hipFree(maxima_) != hipSuccess) std::terminate();
            maxima_ = nullptr;
        }
        if (entries_.empty()) return;
        // Every entry's last consumer is ordered on this one owner stream.
        if (hipStreamSynchronize(stream_) != hipSuccess) std::terminate();
        if (!slabs_.empty()) { for (auto& slab : slabs_) if (slab.base && hipFree(slab.base) != hipSuccess) std::terminate(); return; }
        for (auto& entry : entries_) {
            void* keys = bf16_keys_ ? static_cast<void*>(entry.pool.packed) : static_cast<void*>(entry.pool.keys);
            if (keys && hipFree(keys) != hipSuccess) std::terminate();
        }
    }
    void invalidate() noexcept { for (auto& entry : entries_) entry.pool.valid = 0U; }
    // The decode selection's maxima row (at least groups floats) on the owner's device/stream; null when it cannot be
    // allocated (the caller then keeps select_blocks_kernel).
    float* maxima(std::uint32_t groups) {
        int device = -1;
        check_hip(hipGetDevice(&device), "QSA decode maxima device");
        const auto stream = hip_execution_stream();
        if (device_ < 0) { device_ = device; stream_ = stream; }
        if (device != device_ || stream != stream_) return nullptr;
        if (groups <= maxima_capacity_) return maxima_;
        if (maxima_) {
            check_hip(hipStreamSynchronize(stream_), "QSA decode maxima resize sync");
            check_hip(hipFree(maxima_), "QSA decode maxima free");
            maxima_ = nullptr; maxima_capacity_ = 0;
        }
        const auto grown = std::max<std::uint32_t>(groups, 16384U);
        void* p = nullptr;
        if (hipMalloc(&p, std::size_t{grown} * sizeof(float)) != hipSuccess) { static_cast<void>(hipGetLastError()); return nullptr; }
        maxima_ = static_cast<float*>(p); maxima_capacity_ = grown;
        return maxima_;
    }
    QsaPrefillPool* get(const PreparedIndexedLaunch& a, std::uint32_t ratio, std::uint64_t position) {
        int device = -1;
        check_hip(hipGetDevice(&device), "QSA persistent pool device");
        const auto stream = hip_execution_stream();
        if (device_ < 0) { device_ = device; stream_ = stream; }
        if (device != device_ || stream != stream_) throw std::invalid_argument("QSA persistent pool requires its owner's device/stream");
        const auto& norm = *a.tensors.index_k_norm.runtime;
        for (auto& entry : entries_) if (entry.raw == a.raw_state.address) {
            if (entry.raw_generation != a.raw_state.allocation_generation || entry.norm != norm.address ||
                entry.norm_generation != norm.allocation_generation)
                throw std::invalid_argument("QSA persistent pool owner bindings changed");
            // The previous window may have pooled rejected/padded rows. Never reuse
            // a block containing a position this window rewrites, including partial tails.
            entry.pool.valid = std::min(entry.pool.valid, static_cast<std::uint32_t>(position / ratio));
            return &entry.pool;
        }
        const auto tokens = a.raw_state.bytes / (static_cast<std::uint64_t>(a.index_dim) * sizeof(std::uint16_t));
        const auto blocks = static_cast<std::uint32_t>(std::min<std::uint64_t>(tokens / ratio, UINT32_MAX));
        if (!blocks || entries_.size() >= 64U) return nullptr;
        Entry entry;
        entry.raw = a.raw_state.address; entry.raw_generation = a.raw_state.allocation_generation;
        entry.norm = norm.address; entry.norm_generation = norm.allocation_generation;
        entry.pool.capacity = blocks;
        if (!slab_allowed_) {
            void* keys = nullptr;
            const auto bytes = static_cast<std::uint64_t>(blocks) * a.index_dim * (bf16_keys_ ? sizeof(std::uint16_t) : sizeof(float));
            if (hipMalloc(&keys, bytes) != hipSuccess) {
                static_cast<void>(hipGetLastError()); // an allocation failure must not poison the next launch check
                // Keep the keyless entry so later steps do not retry the allocation; that layer stays on block_score_kernel.
                entries_.push_back(entry);
                static std::atomic<bool> reported{false};
                if (!reported.exchange(true))
                    std::fprintf(stderr, "QSA decode pool: %llu-byte key allocation failed; that layer decodes with block_score_kernel\n",
                        static_cast<unsigned long long>(bytes));
                return nullptr;
            }
            if (bf16_keys_) entry.pool.packed = static_cast<std::uint16_t*>(keys);
            else entry.pool.keys = static_cast<float*>(keys);
            entries_.push_back(entry);
            return &entries_.back().pool;
        }
        // Allocate the host slot before taking ownership of a device allocation.
        entries_.push_back(entry);
        const auto bytes = (static_cast<std::uint64_t>(blocks) * a.index_dim * sizeof(float) + 255U) / 256U * 256U;
        // A slab pool never mixes with per-entry allocations (the destructor frees one kind).
        if (slab_allowed_ && slab_enabled() && (!slabs_.empty() || entries_.size() == 1U)) {
            Slab* slab = nullptr;
            for (auto& s : slabs_) if (s.entry_bytes == bytes && s.used < kSlabEntries) { slab = &s; break; }
            if (!slab) {
                slabs_.push_back(Slab{nullptr, bytes, 0U});
                check_hip(hipMalloc(&slabs_.back().base, bytes * kSlabEntries), "QSA persistent pool slab allocation");
                slab = &slabs_.back();
            }
            entries_.back().pool.keys = reinterpret_cast<float*>(static_cast<unsigned char*>(slab->base) + bytes * slab->used++);
            return &entries_.back().pool;
        }
        void* keys = nullptr;
        check_hip(hipMalloc(&keys, static_cast<std::uint64_t>(blocks) * a.index_dim * sizeof(float)), "QSA persistent pool allocation");
        entries_.back().pool.keys = static_cast<float*>(keys);
        return &entries_.back().pool;
    }
};
std::shared_ptr<HipQsaWindowPool> make_hip_qsa_window_pool() { return std::make_shared<HipQsaWindowPool>(); }
std::shared_ptr<HipQsaWindowPool> make_hip_qsa_decode_pool(bool bf16_keys) { return std::make_shared<HipQsaWindowPool>(false, bf16_keys); }

// Serial decode from the persistent pool: the new complete blocks (at most one per step after the first) are formed by
// block_pool_kernel (the statements of block_score_kernel's pooled part), then block_score_pooled_rows_kernel<1, 1> scores
// every complete block with block_score_kernel's products, 128-feature tree, relu head sum and division: bitwise the same
// block scores (STRIX_QSA_DECODE_POOL_CHECK=1 recomputes them with block_score_kernel and compares, synchronizing per
// step). The selection that follows is unchanged. get() drops every block at or after floor(position / ratio).
namespace {
// STRIX_QSA_DECODE_FUSED_MAXIMA (default on): where the maxima selection below will run (the serial decode pool's own
// scores, ratio 4, no graph position), the scorer also writes the 4-block group maxima (block_score_pooled_decode_maxima_kernel,
// row_maxima_kernel's expression over the same stored scores) and *fused_maxima returns them; the selection then skips
// row_maxima_kernel. Off with STRIX_QSA_DECODE_FUSED_MAXIMA=0, under the baseline, STRIX_QPF2K_DEFAULTS=0 and GSQ checkpoints.
bool qsa_decode_fused_maxima_enabled() {
    // Environment policy once; the GSQ exclusion per call. STRIX_QSA_DECODE_POOL_CHECK=1 rewrites the scores after
    // scoring, so the maxima are then formed from the rewritten scores by row_maxima_kernel as before.
    static const bool on = qpf2k::long_context_item_enabled("STRIX_QSA_DECODE_FUSED_MAXIMA", false) &&
        qpf2k::long_context_item_enabled("STRIX_QSA_DECODE_SELECT", false) &&
        !(std::getenv("STRIX_QSA_DECODE_POOL_CHECK") && std::getenv("STRIX_QSA_DECODE_POOL_CHECK")[0] == '1');
    return on && !qpf2k::baseline && !qpf2k::gsq_model;
}
bool qsa_decode_pool_scores(const PreparedIndexedLaunch& a, const HipIndexedAttentionLauncherConfig& config, std::uint64_t position,
    std::uint32_t complete_blocks, std::uint32_t ratio, QsaPrefillPool* borrowed,
    const std::shared_ptr<HipQsaCommittedPool>& committed_owner, bool dynamic_position, const float** fused_maxima) {
    const auto& model = config.model;
    if ((!config.decode_pool && !borrowed) || a.index_dim != 128U || model.indexer_n_heads != 4U || !ratio || !complete_blocks) return false;
    auto* pool = borrowed ? borrowed : config.decode_pool->get(a, ratio, position);
    if (!pool || (!pool->keys && !pool->packed) || complete_blocks > pool->capacity) return false;
    // Explicit YaRN forms the pooled keys from the model's rotary table, as the prefill pools do (bind_qwen_pool keeps
    // one table per pool; with_qwen_rotary passes the same source block_score_kernel reads).
    bind_qwen_pool(pool, model);
    const auto stream = hip_execution_stream();
    const auto rotary = checked_u32(model.rotary_dim(), "rotary_dim");
    const auto& norm = a.tensors.index_k_norm;
    if (complete_blocks > pool->valid) {
        dispatch_small_kind(norm.manifest->ggml_type, [&](auto kind) {
            with_qwen_rotary(model, stream, [&](auto... rotary_source) {
                const auto launch = [&]<bool PackedOnly>() {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(block_pool_kernel<decltype(kind)::value, false, PackedOnly, decltype(rotary_source)...>),
                        dim3(complete_blocks - pool->valid), dim3(a.index_dim), 2ULL * a.index_dim * sizeof(float), stream,
                        reinterpret_cast<const std::uint16_t*>(a.raw_state.address), pool->valid, a.index_dim, ratio,
                        reinterpret_cast<const void*>(norm.runtime->address), model.rms_norm_epsilon, rotary, model.rope_theta,
                        pool->keys, static_cast<const uint2*>(nullptr), 0U, pool->packed, rotary_source...);
                };
                if (pool->packed) launch.template operator()<true>();
                else launch.template operator()<false>();
            });
        });
        check_hip(hipGetLastError(), "QSA decode block-pool launch");
        pool->valid = complete_blocks;
    }
    QsaRowsScoreViews<1U> views{};
    views.index_query[0] = a.workspace.index_query;
    views.block_scores[0] = a.workspace.block_scores;
    views.blocks[0] = complete_blocks;
    const auto waves = (complete_blocks + 31U) / 32U;
    float* const maxima = fused_maxima && !borrowed && !dynamic_position && ratio == qsa_flash::kGroup && qsa_decode_fused_maxima_enabled()
        ? config.decode_pool->maxima((complete_blocks + qsa_flash::kGroup - 1U) / qsa_flash::kGroup) : nullptr;
    if (maxima) {
        if (pool->packed) hipLaunchKernelGGL(HIP_KERNEL_NAME(block_score_pooled_decode_maxima_kernel<std::uint16_t>), dim3((waves + 7U) / 8U),
            dim3(kThreads), 0, stream, views, static_cast<const std::uint16_t*>(pool->packed), complete_blocks, a.index_dim, maxima);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(block_score_pooled_decode_maxima_kernel<float>), dim3((waves + 7U) / 8U), dim3(kThreads), 0,
            stream, views, static_cast<const float*>(pool->keys), complete_blocks, a.index_dim, maxima);
        *fused_maxima = maxima;
    } else if (pool->packed) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(block_score_pooled_rows_kernel<1U, 1U, std::uint16_t>), dim3((waves + 7U) / 8U), dim3(kThreads), 0, stream,
            views, static_cast<const std::uint16_t*>(pool->packed), complete_blocks, a.index_dim);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(block_score_pooled_rows_kernel<1U, 1U>), dim3((waves + 7U) / 8U), dim3(kThreads), 0, stream,
            views, static_cast<const float*>(pool->keys), complete_blocks, a.index_dim);
    }
    check_hip(hipGetLastError(), "QSA decode pooled block-score launch");
    static const bool check = [] { const char* v = std::getenv("STRIX_QSA_DECODE_POOL_CHECK"); return v && v[0] == '1'; }();
    const bool mtp_check=committed_owner && std::getenv("STRIX_QSA_MTP_POOL_CHECK") && std::getenv("STRIX_QSA_MTP_POOL_CHECK")[0]=='1';
    if (check || mtp_check) {
        std::vector<float> pooled(complete_blocks), reference(complete_blocks);
        check_hip(hipMemcpyAsync(pooled.data(), a.workspace.block_scores, complete_blocks * sizeof(float), hipMemcpyDeviceToHost, stream),
            "QSA decode pool check copy");
        check_hip(hipStreamSynchronize(stream), "QSA decode pool check sync");
        launch_block_scores(a.workspace, a.raw_state, norm, model, complete_blocks, GraphSelectionArguments{});
        check_hip(hipMemcpyAsync(reference.data(), a.workspace.block_scores, complete_blocks * sizeof(float), hipMemcpyDeviceToHost, stream),
            "QSA decode pool check copy");
        check_hip(hipStreamSynchronize(stream), "QSA decode pool check sync");
        std::size_t differing = 0;
        for (std::uint32_t i = 0; i < complete_blocks; ++i) differing += std::memcmp(&pooled[i], &reference[i], sizeof(float)) != 0;
        if (mtp_check) {
            check_hip(hipMemcpyAsync(a.workspace.block_scores,pooled.data(),complete_blocks*sizeof(float),hipMemcpyHostToDevice,stream),"MTP pool check restore scores");
            check_hip(hipStreamSynchronize(stream),"MTP pool check restore completion");
            committed_owner->checked(complete_blocks,differing,pool->mtp_role);
        }
        static std::atomic<std::uint64_t> steps{0}, compared{0}, total_differing{0};
        const auto s = steps.fetch_add(1U) + 1U;
        const auto c = compared.fetch_add(complete_blocks) + complete_blocks;
        const auto d = total_differing.fetch_add(differing) + differing;
        if (differing || s % 48U == 0U)
            std::fprintf(stderr, "QSA decode pool check: position %llu, %u scores, %zu differ (total %llu layer-steps, %llu scores, %llu differ)\n",
                static_cast<unsigned long long>(position), complete_blocks, differing, static_cast<unsigned long long>(s),
                static_cast<unsigned long long>(c), static_cast<unsigned long long>(d));
    }
    return true;
}
bool qsa_decode_pool_select(const PreparedIndexedLaunch& a, const HipIndexedAttentionLauncherConfig& config,
    const PreparedIndexedSelection& selection, const float* fused_maxima) {
    static const bool enabled = qpf2k::long_context_item_enabled("STRIX_QSA_DECODE_SELECT");
    if (!enabled || !config.decode_pool || selection.ratio != qsa_flash::kGroup || selection.dynamic_selection.position.position) return false;
    const auto groups = (selection.complete_blocks + qsa_flash::kGroup - 1U) / qsa_flash::kGroup;
    const auto stream = hip_execution_stream();
    float* const maxima = config.decode_pool->maxima(groups);
    if (!maxima || (fused_maxima && fused_maxima != maxima)) return false;
    // STRIX_QSA_DECODE_POOL_CHECK=1 also runs select_blocks_kernel on a copy of the scores and compares the selected list
    // and the post-selection scores bitwise (synchronizes per layer-step).
    static const bool check = [] { const char* v = std::getenv("STRIX_QSA_DECODE_POOL_CHECK"); return v && v[0] == '1'; }();
    const auto blocks = selection.complete_blocks, count = selection.selected_count;
    std::vector<float> before;
    if (check) {
        before.resize(blocks);
        check_hip(hipMemcpyAsync(before.data(), a.workspace.block_scores, blocks * sizeof(float), hipMemcpyDeviceToHost, stream), "QSA decode select check copy");
        check_hip(hipStreamSynchronize(stream), "QSA decode select check sync");
    }
    if (!fused_maxima) hipLaunchKernelGGL(qsa_flash::row_maxima_kernel, dim3((groups + 255U) / 256U), dim3(256), 0, stream,
        static_cast<const float*>(a.workspace.block_scores), selection.complete_blocks, maxima);
    hipLaunchKernelGGL(qsa_flash::select_one_kernel, dim3(1), dim3(kSelectThreads), 0, stream,
        a.workspace.block_scores, a.workspace.selected_indices, selection.visible, static_cast<const float*>(maxima),
        selection.ratio, selection.block_topk);
    check_hip(hipGetLastError(), "QSA decode maxima selection launch");
    if (check) {
        std::vector<float> scores(blocks), reference_scores(blocks);
        std::vector<std::uint32_t> selected(count), reference(count);
        const auto pull = [&](std::vector<float>& s, std::vector<std::uint32_t>& sel) {
            check_hip(hipMemcpyAsync(s.data(), a.workspace.block_scores, blocks * sizeof(float), hipMemcpyDeviceToHost, stream), "QSA decode select check copy");
            check_hip(hipMemcpyAsync(sel.data(), a.workspace.selected_indices, count * sizeof(std::uint32_t), hipMemcpyDeviceToHost, stream),
                "QSA decode select check copy");
            check_hip(hipStreamSynchronize(stream), "QSA decode select check sync");
        };
        pull(scores, selected);
        check_hip(hipMemcpyAsync(a.workspace.block_scores, before.data(), blocks * sizeof(float), hipMemcpyHostToDevice, stream), "QSA decode select check restore");
        hipLaunchKernelGGL(select_blocks_kernel, dim3(1), dim3(kSelectThreads), 0, stream, a.workspace.block_scores, blocks,
            selection.block_topk, selection.ratio, selection.visible, a.workspace.selected_indices, GraphSelectionArguments{});
        check_hip(hipGetLastError(), "QSA decode select check launch");
        pull(reference_scores, reference);
        const bool same = std::memcmp(scores.data(), reference_scores.data(), blocks * sizeof(float)) == 0 && selected == reference;
        static std::atomic<std::uint64_t> steps{0}, differing{0};
        const auto s = steps.fetch_add(1U) + 1U;
        const auto d = differing.fetch_add(same ? 0U : 1U) + (same ? 0U : 1U);
        if (!same || s % 48U == 0U)
            std::fprintf(stderr, "QSA decode select check: %u blocks, %u selected, %s (total %llu layer-steps, %llu differ)\n", blocks, count,
                same ? "same" : "DIFFER", static_cast<unsigned long long>(s), static_cast<unsigned long long>(d));
    }
    return true;
}
} // namespace
void invalidate_hip_qsa_window_pool(const std::shared_ptr<HipQsaWindowPool>& pool) { if (pool) pool->invalidate(); }

// STRIX_QSA_WINDOW_POOLED=1: verify windows score the indexer blocks as prefill groups do: one
// block_pool_kernel pass forms every complete block's pooled/normalised/rotated key (the statements
// of block_score_rows_kernel's pooled part), then block_score_pooled_rows_kernel scores them with the
// same per-(row, block, head) products and reduction order. Bitwise the same block scores
// (STRIX_QSA_POOLED_CHECK=1 compares them). The pool is formed again by every call.
namespace {
QsaPrefillPool* qsa_window_pool(const PreparedIndexedLaunch& a, std::uint32_t ratio,
    const HipIndexedAttentionLauncherConfig& config, std::uint64_t position) {
    static const bool enabled = [] { const char* v = std::getenv("STRIX_QSA_WINDOW_POOLED"); return v && v[0] == '1'; }();
    if (!enabled || a.index_dim != 128U || !ratio) return nullptr;
    if (config.window_pool) return config.window_pool->get(a, ratio, position);
    static QsaPrefillPool pool;
    const auto tokens = a.raw_state.bytes / (static_cast<std::uint64_t>(a.index_dim) * sizeof(std::uint16_t));
    const auto blocks = static_cast<std::uint32_t>(std::min<std::uint64_t>(tokens / ratio, UINT32_MAX));
    if (blocks > pool.capacity) {
        if (pool.keys) {
            check_hip(hipStreamSynchronize(hip_execution_stream()), "QSA window block-pool resize sync");
            check_hip(hipFree(pool.keys), "QSA window block-pool free");
            pool = {};
        }
        void* keys = nullptr;
        check_hip(hipMalloc(&keys, static_cast<std::uint64_t>(blocks) * a.index_dim * sizeof(float)), "QSA window block-pool allocation");
        pool.keys = static_cast<float*>(keys);
        pool.capacity = blocks;
    }
    pool.valid = 0;
    return &pool;
}
template <unsigned Rows>
void enqueue_qsa_window_rows(std::span<const ExecutionDispatchContext> rows, const HipIndexedAttentionLauncherConfig& config,
    HipQsaIndexJournal journal) {
    const auto prepared = prepare_qsa_known_rows<Rows>(rows, config, journal);
    std::unique_ptr<HipQsaCommittedAccess> access;
    if (config.committed_pool) {
        const auto& a=prepared.row[0];
        access=config.committed_pool->before_rewrite(false,a.raw_state.address,a.raw_state.bytes,a.raw_state.allocation_generation,
            *a.tensors.index_k_norm.runtime,a.tensors.index_k_norm.manifest->ggml_type,config.model,
            prepared.position[0],Rows,hip_execution_stream(),true);
    }
    auto* pool=access?static_cast<QsaPrefillPool*>(access->storage()):
        qsa_window_pool(prepared.row[0], prepared.selection[0].ratio, config, prepared.position[0]);
    enqueue_qsa_known_rows<Rows>(prepared, config, 7U, pool, 0U, nullptr, 0U, 127U, nullptr, true);
    // This owner may be idle while serial steps or other owners overwrite its
    // rejected rows. A later frontier could already be past the stale blocks,
    // so clamping only on the next call cannot detect them. Retain only blocks
    // known committed BEFORE this window; rebuild its newly formed blocks
    // (at most three for the canonical R8 / compress-ratio-four path).
    if (pool && config.window_pool && !access)
        pool->valid = std::min(pool->valid, static_cast<std::uint32_t>(prepared.position[0] / prepared.selection[0].ratio));
}
} // namespace
void indexed_attention_known_window_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipIndexedAttentionLauncherConfig& config, HipQsaIndexJournal journal) {
#if defined(STRIX_HAS_EXPERIMENTAL_DENSE_TRELLIS)
    HipDenseTrellisScope arithmetic(config.dense_trellis_projections);
#endif
    if (rows.size() == 2U) enqueue_qsa_window_rows<2U>(rows, config, journal);
    else if (rows.size() == 4U) enqueue_qsa_window_rows<4U>(rows, config, journal);
    else if (rows.size() == 8U) enqueue_qsa_window_rows<8U>(rows, config, journal);
    else throw std::invalid_argument("QSA known-window requires exactly two, four or eight rows");
}

namespace {
template <class Body> void qsa_rows_dispatch(std::size_t rows, Body&& body) {
    if (rows == 2U) body(std::integral_constant<unsigned, 2U>{});
    else if (rows == 4U) body(std::integral_constant<unsigned, 4U>{});
    else if (rows == 8U) body(std::integral_constant<unsigned, 8U>{});
    else throw std::invalid_argument("QSA known-window requires exactly two, four or eight rows");
}
using QsaSegmentRows = std::variant<QsaKnownRows<2U>, QsaKnownRows<4U>, QsaKnownRows<8U>>;
struct QsaCohortPrepared {
    std::vector<QsaSegmentRows> segments;
    std::vector<unsigned> first;
    std::size_t count = 0;
};
QsaCohortPrepared prepare_qsa_cohort(std::span<const HipQsaCohortSegment> segments) {
    if (segments.size() < 2U) throw std::invalid_argument("QSA cohort window requires two or more session segments");
    // Reject mixed numerics before preparing any row or querying device state.
    for (const auto& segment : segments)
        if (!segment.config || !segments[0].config ||
            !same_rotary_model(segment.config->model, segments[0].config->model))
            throw std::invalid_argument("QSA cohort rotary identity differs");
    QsaCohortPrepared out;
    for (const auto& segment : segments) {
        if (!segment.config || out.count + segment.rows.size() > 32U)
            throw std::invalid_argument("QSA cohort window segments exceed 32 rows");
        // Every session segment passes the complete single-session admission.
        qsa_rows_dispatch(segment.rows.size(), [&](auto rows) {
            out.segments.emplace_back(prepare_qsa_known_rows<decltype(rows)::value>(segment.rows, *segment.config, {}));
        });
        out.first.push_back(static_cast<unsigned>(out.count));
        out.count += segment.rows.size();
    }
    if (out.count != 4U && out.count != 8U && (out.count < 9U || out.count > 32U))
        throw std::invalid_argument("QSA cohort window requires 4, 8 or 9..32 rows in total");
    // More than 8 rows: dense A8 with a Q8_0 output projection only (the pre-quantized slot path; per row the
    // arithmetic of the 1..8-row forms), padded to 16 / 32 rows for the projections.
    if (out.count > 8U) {
        const auto& p0 = std::visit([](const auto& p) -> const PreparedIndexedLaunch& { return p.row[0]; }, out.segments[0]);
        if (!segments[0].config->dense_a8_projections || p0.tensors.out.manifest->ggml_type != 8U)
            throw std::invalid_argument("QSA cohort window of 16 or 32 rows requires dense A8 and a Q8_0 output projection");
    }
    const auto& c0 = *segments[0].config;
    const auto& a = std::visit([](const auto& p) -> const PreparedIndexedLaunch& { return p.row[0]; }, out.segments[0]);
    const auto prefetch0 = std::visit([](const auto& p) { return p.prefetch; }, out.segments[0]);
    for (std::size_t s = 0; s < segments.size(); ++s) {
        const auto& c = *segments[s].config;
        const auto& b = std::visit([](const auto& p) -> const PreparedIndexedLaunch& { return p.row[0]; }, out.segments[s]);
        if (!same_rotary_model(c.model, c0.model) || c.dense_a8_projections != c0.dense_a8_projections || c.dense_trellis_projections != c0.dense_trellis_projections || c.wave8_q8_output != c0.wave8_q8_output ||
            c.prefetch_q8_input != c0.prefetch_q8_input || std::visit([](const auto& p) { return p.prefetch; }, out.segments[s]) != prefetch0 ||
            *segments[s].rows[0].layer != *segments[0].rows[0].layer || b.hidden != a.hidden || b.attention_width != a.attention_width)
            throw std::invalid_argument("QSA cohort window segments differ in layer or arithmetic");
        for (const auto member : {&IndexedTensors::q, &IndexedTensors::k, &IndexedTensors::v, &IndexedTensors::out,
            &IndexedTensors::q_norm, &IndexedTensors::k_norm, &IndexedTensors::index_q, &IndexedTensors::index_k,
            &IndexedTensors::index_q_norm, &IndexedTensors::index_k_norm})
            if ((b.tensors.*member).runtime != (a.tensors.*member).runtime || (b.tensors.*member).manifest != (a.tensors.*member).manifest)
                throw std::invalid_argument("QSA cohort window projection owners differ between sessions");
        for (std::size_t o = 0; o < s; ++o) {
            const auto& other = std::visit([](const auto& p) -> const PreparedIndexedLaunch& { return p.row[0]; }, out.segments[o]);
            if (other.kv_state.address == b.kv_state.address || other.raw_state.address == b.raw_state.address)
                throw std::invalid_argument("QSA cohort window segments share session state");
        }
    }
    return out;
}
// Shared sink of the padding rows of cohort projections (workspace and output; grown on demand).
std::uintptr_t qsa_cohort_pad_sink(std::uint64_t bytes) {
    static void* memory = nullptr;
    static std::uint64_t held = 0;
    static std::mutex mutex;
    const std::lock_guard lock(mutex);
    if (held < bytes) {
        if (memory) { check_hip(hipDeviceSynchronize(), "drain QSA cohort pad sink"); check_hip(hipFree(memory), "free QSA cohort pad sink"); }
        check_hip(hipMalloc(&memory, bytes), "allocate QSA cohort pad sink");
        held = bytes;
    }
    return reinterpret_cast<std::uintptr_t>(memory);
}
// The rows of every segment as one T-row view for the state-free parts (input and output projections).
// Rows past the segments' (padding up to 16 / 32) read row 0's input and write only the pad sink.
template <unsigned Rows>
QsaKnownRows<Rows> qsa_cohort_rows(const QsaCohortPrepared& prepared) {
    QsaKnownRows<Rows> all{};
    if (prepared.count < Rows) {
        const auto& head = std::visit([](const auto& p) -> const PreparedIndexedLaunch& { return p.row[0]; }, prepared.segments[0]);
        const auto workspace = (head.workspace_layout.total_bytes + 255U) / 256U * 256U;
        const auto sink = qsa_cohort_pad_sink(workspace + static_cast<std::uint64_t>(head.hidden) * 2U);
        auto pad = head;
        pad.temporary = {pad.temporary.use, sink, head.workspace_layout.total_bytes, pad.temporary.allocation_generation};
        pad.workspace = workspace_pointers(pad.temporary, head.workspace_layout);
        pad.output = {pad.output.use, sink + workspace, static_cast<std::uint64_t>(head.hidden) * 2U, pad.output.allocation_generation};
        const auto& selection = std::visit([](const auto& p) -> const PreparedIndexedSelection& { return p.selection[0]; }, prepared.segments[0]);
        for (auto r = static_cast<unsigned>(prepared.count); r < Rows; ++r) {
            all.row[r] = pad; all.selection[r] = selection; all.position[r] = std::visit([](const auto& p) { return p.position[0]; }, prepared.segments[0]);
        }
    }
    for (std::size_t s = 0; s < prepared.segments.size(); ++s)
        std::visit([&](const auto& p) {
            constexpr auto count = std::tuple_size_v<decltype(p.row)>;
            if (s == 0U) all.prefetch = p.prefetch;
            for (std::size_t r = 0; r < count; ++r) {
                all.row[prepared.first[s] + r] = p.row[r];
                all.selection[prepared.first[s] + r] = p.selection[r];
                all.position[prepared.first[s] + r] = p.position[r];
            }
        }, prepared.segments[s]);
    return all;
}
} // namespace

void indexed_attention_known_window_cohort_preflight(std::span<const HipQsaCohortSegment> segments) {
    static_cast<void>(prepare_qsa_cohort(segments));
}
void indexed_attention_known_window_cohort_launch(std::span<const HipQsaCohortSegment> segments) {
    const auto prepared = prepare_qsa_cohort(segments);
    const auto& config = *segments[0].config;
#if defined(STRIX_HAS_EXPERIMENTAL_DENSE_TRELLIS)
    HipDenseTrellisScope arithmetic(config.dense_trellis_projections);
#endif
    const auto slot = q8_rows::dense_a8_slot(0U);
    const auto merged = [&](unsigned part) {
        if (prepared.count == 4U) enqueue_qsa_known_rows<4U>(qsa_cohort_rows<4U>(prepared), config, part, nullptr, 0U, &slot);
        else if (prepared.count == 8U) enqueue_qsa_known_rows<8U>(qsa_cohort_rows<8U>(prepared), config, part, nullptr, 0U, &slot);
        else if (prepared.count <= 16U) enqueue_qsa_known_rows<16U>(qsa_cohort_rows<16U>(prepared), config, part, nullptr, 0U, &slot);
        else enqueue_qsa_known_rows<32U>(qsa_cohort_rows<32U>(prepared), config, part, nullptr, 0U, &slot);
    };
    merged(1U);
    // Part 2: STRIX_COHORT_QSA_SEGMENTS=1 runs the state-free sub-parts (norms, selection, softmax) once for every
    // row (measured neutral at eight sessions: 148.0 against 147.0 ms per cycle); default: the part per session.
    static const bool merge_free = [] { const char* v = std::getenv("STRIX_COHORT_QSA_SEGMENTS"); return v && v[0] == '1'; }();
    const bool merge = merge_free && !known_window_cohort_feature_off("qsaseg");
    // STRIX_QSA_COHORT_POOL=1: each session scores its blocks from its own owner's persistent pool, as
    // its single-session window does (enqueue_qsa_window_rows): the same pooled keys and scoring kernels
    // (bitwise the per-call block_score_rows_kernel scores). Only owners with a persistent pool
    // (STRIX_QKERN_QSA_PERSISTENT=1) take part: the process-wide fallback pool is formed per call and
    // would be shared between the sessions of this window.
    std::vector<QsaPrefillPool*> pools(prepared.segments.size(), nullptr);
    std::vector<std::unique_ptr<HipQsaCommittedAccess>> accesses;
    if(std::any_of(segments.begin(),segments.end(),[](const auto& segment){return static_cast<bool>(segment.config->committed_pool);}))
        accesses.resize(prepared.segments.size());
    for(std::size_t s=0;s<prepared.segments.size();++s) {
        const auto& c=*segments[s].config;
        if(c.committed_pool) std::visit([&](const auto& p) {
            const auto& a=p.row[0];
            accesses[s]=c.committed_pool->before_rewrite(false,a.raw_state.address,a.raw_state.bytes,a.raw_state.allocation_generation,
                *a.tensors.index_k_norm.runtime,a.tensors.index_k_norm.manifest->ggml_type,c.model,p.position[0],
                static_cast<std::uint32_t>(segments[s].rows.size()),hip_execution_stream(),true);
            pools[s]=static_cast<QsaPrefillPool*>(accesses[s]->storage());
        },prepared.segments[s]);
        else if(qsa_cohort_pool_enabled() && c.window_pool) pools[s]=std::visit([&](const auto& p) {
            return qsa_window_pool(p.row[0],p.selection[0].ratio,c,p.position[0]);
        },prepared.segments[s]);
    }
    const auto per_session = [&](unsigned subparts) {
        for (std::size_t s = 0; s < prepared.segments.size(); ++s)
            std::visit([&](const auto& p) {
                constexpr auto count = static_cast<unsigned>(std::tuple_size_v<decltype(p.row)>);
                enqueue_qsa_known_rows<count>(p, *segments[s].config, 2U, pools[s], 0U, &slot, prepared.first[s], subparts);
            }, prepared.segments[s]);
    };
    const auto all_rows = [&](unsigned subparts) {
        if (prepared.count == 4U) enqueue_qsa_known_rows<4U>(qsa_cohort_rows<4U>(prepared), config, 2U, nullptr, 0U, &slot, 0U, subparts);
        else if (prepared.count == 8U) enqueue_qsa_known_rows<8U>(qsa_cohort_rows<8U>(prepared), config, 2U, nullptr, 0U, &slot, 0U, subparts);
        else if (prepared.count <= 16U) enqueue_qsa_known_rows<16U>(qsa_cohort_rows<16U>(prepared), config, 2U, nullptr, 0U, &slot, 0U, subparts);
        else enqueue_qsa_known_rows<32U>(qsa_cohort_rows<32U>(prepared), config, 2U, nullptr, 0U, &slot, 0U, subparts);
    };
    if (merge) {
        all_rows(1U); per_session(2U | 4U); all_rows(8U); per_session(16U); all_rows(32U); per_session(64U);
    } else per_session(127U);
    // As enqueue_qsa_window_rows: keep only the blocks committed before this window (its rows may be rejected).
    for (std::size_t s = 0; s < prepared.segments.size(); ++s)
        if (pools[s] && (accesses.empty() || !accesses[s]))
            std::visit([&](const auto& p) {
                pools[s]->valid = std::min(pools[s]->valid, static_cast<std::uint32_t>(p.position[0] / p.selection[0].ratio));
            }, prepared.segments[s]);
    merged(4U);
}


#ifdef STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC
namespace {
template<unsigned Rows> void qsa_committed_pool_probe(bool pooled,bool predictor,const std::shared_ptr<HipQsaCommittedPool>& owner,
    const ModelContract& model,std::uintptr_t raw,std::uint64_t capacity,const RuntimeTensorBinding& norm,
    std::uint64_t first,std::uint32_t writer_rows,std::span<const HipQsaPoolRowDescriptor> rows,hipStream_t stream) {
    std::unique_ptr<HipQsaCommittedAccess> access;
    if(pooled) access=owner->before_rewrite(predictor,raw,capacity*256U,1,norm,0,model,first,writer_rows,stream,true);
    QsaRowsScoreViews<Rows> score{};QsaRowsSelectViews<Rows> select{};std::uint32_t blocks=0;
    for(unsigned r=0;r<Rows;++r) {
        score.index_query[r]=rows[r].query;score.block_scores[r]=rows[r].scores;score.blocks[r]=rows[r].complete_blocks;
        blocks=std::max(blocks,score.blocks[r]);select.block_scores[r]=rows[r].scores;
        select.selected[r]=rows[r].selected;select.visible[r]=rows[r].visible;
    }
    if(access && blocks) {
        auto& pool=*static_cast<QsaPrefillPool*>(access->storage());bind_qwen_pool(&pool,model);
        if(blocks>pool.valid) with_qwen_rotary(model,stream,[&](auto... source) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(block_pool_kernel<SmallWeightKind::f32,false,true,decltype(source)...>),
                dim3(blocks-pool.valid),dim3(128),1024,stream,reinterpret_cast<const std::uint16_t*>(raw),pool.valid,128U,4U,
                reinterpret_cast<const void*>(norm.address),model.rms_norm_epsilon,checked_u32(model.rotary_dim(),"probe rotary"),
                model.rope_theta,nullptr,nullptr,0U,pool.packed,source...);
        });
        pool.valid=blocks;
        constexpr unsigned Reuse=Rows<4?Rows:4;
        const auto waves=(Rows/Reuse)*((blocks+31U)/32U);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(block_score_pooled_rows_kernel<Rows,Reuse,std::uint16_t>),dim3((waves+7U)/8U),dim3(256),0,stream,
            score,static_cast<const std::uint16_t*>(pool.packed),blocks,128U);
    } else if(blocks) with_qwen_rotary(model,stream,[&](auto... source) {
        if constexpr (Rows==1) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(block_score_kernel<SmallWeightKind::f32,decltype(source)...>),dim3(blocks),dim3(128),1024,stream,
                rows[0].query,reinterpret_cast<const std::uint16_t*>(raw),4U,128U,4U,reinterpret_cast<const void*>(norm.address),
                model.rms_norm_epsilon,checked_u32(model.rotary_dim(),"probe rotary"),model.rope_theta,rows[0].scores,GraphSelectionArguments{},source...);
        } else {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(block_score_rows_kernel<SmallWeightKind::f32,Rows,decltype(source)...>),dim3(blocks),dim3(128),1024,stream,
                score,reinterpret_cast<const std::uint16_t*>(raw),4U,128U,4U,reinterpret_cast<const void*>(norm.address),
                model.rms_norm_epsilon,checked_u32(model.rotary_dim(),"probe rotary"),model.rope_theta,source...);
        }
    });
    // topk==UINT32_MAX requests preselection scores only.
    if(rows[0].topk!=UINT32_MAX) {
        if constexpr (Rows==1) {
            hipLaunchKernelGGL(select_blocks_kernel,dim3(1),dim3(kSelectThreads),0,stream,rows[0].scores,rows[0].complete_blocks,
                rows[0].topk,4U,rows[0].visible,rows[0].selected,GraphSelectionArguments{});
        } else {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(select_rows_kernel<Rows>),dim3(Rows),dim3(kSelectThreads),0,stream,select,4U,rows[0].topk);
        }
    }
    check_hip(hipGetLastError(),"MTP QSA production fixture launch");
}
}
void launch_hip_qsa_committed_pool_probe(bool pooled,bool predictor,const std::shared_ptr<HipQsaCommittedPool>& owner,
    const ModelContract& model,std::uintptr_t raw,std::uint64_t capacity,const RuntimeTensorBinding& norm,
    std::uint64_t first,std::uint32_t writer_rows,std::span<const HipQsaPoolRowDescriptor> rows,void* stream) {
    if(!owner || !stream || rows.empty()) throw std::invalid_argument("MTP QSA fixture requires retained owner/views/stream");
    const auto run=[&]<unsigned Rows>() {qsa_committed_pool_probe<Rows>(pooled,predictor,owner,model,raw,capacity,norm,first,writer_rows,rows,static_cast<hipStream_t>(stream));};
    if(rows.size()==1)run.template operator()<1>();else if(rows.size()==2)run.template operator()<2>();
    else if(rows.size()==4)run.template operator()<4>();else if(rows.size()==8)run.template operator()<8>();
    else throw std::invalid_argument("MTP QSA fixture row count");
}
#endif

std::uint64_t qsa_index_journal_bytes(const ModelContract& model) {
    if (const auto error = model.validate(); !error.empty()) throw std::invalid_argument(error);
    if (model.hidden_layers > kQsaTailLayers || model.indexer_compress_ratio > 64U || model.indexer_head_dim > kThreads)
        throw std::invalid_argument("QSA index journal geometry exceeds its closed bounds");
    return model.hidden_layers * qsa_journal_layer_elements(model) * sizeof(std::uint16_t);
}

HipQsaIndexTailPlan prepare_qsa_index_tail(const ModelContract& model, const SerialSessionStateLayout& layout,
    const SessionStateMemoryBinding& memory, HipQsaIndexJournal journal) {
    const auto bytes = qsa_index_journal_bytes(model);
    if (!journal.address || journal.address % 2U || journal.bytes != bytes || journal.bytes > UINTPTR_MAX - journal.address)
        throw std::invalid_argument("QSA index tail journal view is invalid");
    HipQsaIndexTailPlan plan;
    plan.journal = journal; plan.context_tokens = layout.context_tokens;
    plan.index_dim = checked_u32(model.indexer_head_dim, "indexer_head_dim");
    plan.ratio = checked_u32(model.indexer_compress_ratio, "indexer_compress_ratio");
    const auto row_bytes = build_indexed_attention_state_layout(model).raw_index_row_bytes;
    for (const auto& step : build_serial_decode_execution_plan(model).steps) {
        if (step.op != SerialOp::indexed_attention) continue;
        const auto state = resolve_indexed_attention_layer_state(layout, memory, model, *step.layer, SessionStateComponent::sparse_index);
        if (state.storage != SessionStateStorage::device || !state.address || state.address % 2U ||
            state.bytes < layout.context_tokens * row_bytes ||
            (state.address < journal.address + journal.bytes && journal.address < state.address + state.bytes))
            throw std::invalid_argument("QSA index tail requires disjoint device raw-index state");
        plan.raw_index.push_back(state.address); plan.layers.push_back(*step.layer);
    }
    if (plan.layers.empty()) throw std::invalid_argument("QSA index tail model has no indexed-attention layer");
    return plan;
}

namespace {
QsaTailLayers tail_layers(const HipQsaIndexTailPlan& plan) {
    if (plan.layers.empty() || plan.layers.size() != plan.raw_index.size() || plan.layers.size() > kQsaTailLayers)
        throw std::invalid_argument("QSA index tail plan is not prepared");
    QsaTailLayers layers{};
    for (std::size_t i = 0; i < plan.layers.size(); ++i) {
        layers.raw[i] = reinterpret_cast<std::uint16_t*>(plan.raw_index[i]); layers.layer[i] = plan.layers[i];
    }
    return layers;
}
std::uint64_t tail_layer_elements(const HipQsaIndexTailPlan& plan) {
    return (static_cast<std::uint64_t>(plan.ratio) - 1U + 8U) * plan.index_dim;
}
} // namespace

void enqueue_qsa_index_tail_snapshot(const HipQsaIndexTailPlan& plan, const std::uint64_t* device_position, void* stream) {
    if (!device_position) throw std::invalid_argument("QSA index tail snapshot requires a device position");
    hipLaunchKernelGGL(qsa_index_tail_snapshot_kernel, dim3(static_cast<std::uint32_t>(plan.layers.size())), dim3(kThreads), 0,
        static_cast<hipStream_t>(stream), tail_layers(plan), reinterpret_cast<std::uint16_t*>(plan.journal.address),
        device_position, plan.context_tokens, plan.index_dim, plan.ratio, tail_layer_elements(plan));
    check_hip(hipGetLastError(), "QSA index tail snapshot launch");
}
void enqueue_qsa_index_tail_commit(const HipQsaIndexTailPlan& plan, const std::uint64_t* device_position,
    const std::uint32_t* device_n_keep, void* stream) {
    if (!device_position || !device_n_keep) throw std::invalid_argument("QSA index tail commit requires device position and row count");
    hipLaunchKernelGGL(qsa_index_tail_commit_kernel, dim3(static_cast<std::uint32_t>(plan.layers.size())), dim3(kThreads), 0,
        static_cast<hipStream_t>(stream), tail_layers(plan), reinterpret_cast<const std::uint16_t*>(plan.journal.address),
        device_position, device_n_keep, plan.context_tokens, plan.index_dim, plan.ratio, tail_layer_elements(plan));
    check_hip(hipGetLastError(), "QSA index tail commit launch");
}

void launch_hip_qsa_input_projection_probe(HipQsaInputProjectionMode mode,HipQsaInputProjectionView input,HipQsaInputProjectionView weights,HipQsaInputProjectionView output,std::uint32_t width,std::uint32_t outputs,void* stream){
    const auto required=qsa_input_prefetch::required_weight_bytes(width,outputs);
    const auto valid=[](HipQsaInputProjectionView v,std::uint64_t n,unsigned align){return v.address!=0 && v.address%align==0 && v.bytes==n && v.bytes<=UINTPTR_MAX-v.address;};
    if((mode!=HipQsaInputProjectionMode::legacy256 && mode!=HipQsaInputProjectionMode::virtual256 && mode!=HipQsaInputProjectionMode::prefetch8) || !valid(input,static_cast<std::uint64_t>(width)*2U,16) || !valid(weights,required,16) || !valid(output,static_cast<std::uint64_t>(outputs)*4U,16))throw std::invalid_argument("QSA input projection probe exact views invalid");
    const auto overlap=[](HipQsaInputProjectionView x,HipQsaInputProjectionView y){return x.address<y.address+y.bytes && y.address<x.address+x.bytes;};
    if(overlap(output,input)||overlap(output,weights))throw std::invalid_argument("QSA input projection probe writable alias");
    check_hip(hipPeekAtLastError(),"QSA projection probe stale error");
    auto hip_stream=static_cast<hipStream_t>(stream);auto* in=reinterpret_cast<const std::uint16_t*>(input.address);auto* w=reinterpret_cast<const void*>(weights.address);auto* out=reinterpret_cast<float*>(output.address);
    if(mode==HipQsaInputProjectionMode::legacy256)hipLaunchKernelGGL(HIP_KERNEL_NAME(linear_to_float_kernel<WeightKind::q8_0>),dim3(outputs),dim3(256),0,hip_stream,in,w,width,outputs,out);
    else if(mode==HipQsaInputProjectionMode::virtual256)hipLaunchKernelGGL(linear_virtual256_q8_qsa_input_kernel,dim3(outputs),dim3(32),0,hip_stream,in,w,width,outputs,out);
    else hipLaunchKernelGGL(linear_prefetch8_q8_qsa_input_kernel,dim3(outputs),dim3(32),0,hip_stream,in,w,width,outputs,out);
    check_hip(hipGetLastError(),"QSA input projection probe launch");
}

#ifdef STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC
void launch_hip_qsa_block_select_probe(bool reference, std::span<const HipQsaBlockSelectRow> rows, std::uint32_t ratio,
    std::uint32_t block_topk, const std::uint64_t* device_position, std::uint64_t context_tokens, std::uint32_t* device_status,
    void* stream) {
    const auto owned = static_cast<hipStream_t>(stream);
    if (!owned || owned == hipStreamPerThread || owned == hipStreamLegacy || rows.empty() || rows.size() > 8U || !ratio ||
        (device_position && (rows.size() != 1U || !device_status)))
        throw std::invalid_argument("QSA block-select probe requires an owned stream, 1..8 rows and a guarded device position");
    check_hip(hipPeekAtLastError(), "QSA block-select probe stale error");
    const GraphSelectionArguments dynamic{{device_position, context_tokens, device_status}, ratio, block_topk};
    if (reference || rows.size() == 1U) {
        for (const auto& row : rows) {
            const auto complete = row.visible / ratio;
            if (!device_position && complete <= block_topk) {
                if (row.visible) hipLaunchKernelGGL(fill_visible_indices_kernel, dim3((row.visible + kThreads - 1U) / kThreads),
                    dim3(kThreads), 0, owned, row.selected, row.visible);
            } else if (reference) hipLaunchKernelGGL(select_blocks_serial_kernel, dim3(1), dim3(1), 0, owned,
                row.scores, complete, block_topk, ratio, row.visible, row.selected, dynamic);
            else hipLaunchKernelGGL(select_blocks_kernel, dim3(1), dim3(kSelectThreads), 0, owned,
                row.scores, complete, block_topk, ratio, row.visible, row.selected, dynamic);
        }
    } else {
        const auto launch = [&](auto count) {
            constexpr unsigned Rows = decltype(count)::value;
            QsaRowsSelectViews<Rows> views{};
            for (unsigned r = 0; r < Rows; ++r) {
                views.block_scores[r] = rows[r].scores; views.selected[r] = rows[r].selected; views.visible[r] = rows[r].visible;
            }
            hipLaunchKernelGGL(HIP_KERNEL_NAME(select_rows_kernel<Rows>), dim3(Rows), dim3(kSelectThreads), 0, owned, views, ratio, block_topk);
        };
        if (rows.size() == 2U) launch(std::integral_constant<unsigned, 2>{});
        else if (rows.size() == 4U) launch(std::integral_constant<unsigned, 4>{});
        else if (rows.size() == 8U) launch(std::integral_constant<unsigned, 8>{});
        else throw std::invalid_argument("QSA block-select probe rows kernel takes 2, 4 or 8 rows");
    }
    check_hip(hipGetLastError(), "QSA block-select probe launch");
}
#endif

#ifdef STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC
__global__ void compact_qsa_status_probe(QsaRowsAttentionViews<8U> v, const std::uint16_t* kv,
    unsigned maximum_selected, std::uint32_t* status) {
    GraphSelectionArguments dynamic{};
    dynamic.position.status = status;
    attention_output_body(v.q_gate[blockIdx.y], kv, v.selected[blockIdx.y], v.scores[blockIdx.y],
        v.count[blockIdx.y], maximum_selected, 24U, 2U, 256U, v.image(blockIdx.y), dynamic);
}
__global__ void compact_qsa_quantizer_probe(q8_rows::DenseA8Slot slot, unsigned char* out) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<8U*6144U) out[i]=reinterpret_cast<const unsigned char*>(slot.codes)[i];
    if(i<192U*8U) reinterpret_cast<float*>(out+8U*6144U)[i]=slot.deltas[(i/8U)*q8_rows::kDenseA8Rows+i%8U];
}
void launch_hip_qsa_window_out_group_probe(const HipQsaWindowOutGroupProbe& p, void* opaque_stream) {
    auto stream=static_cast<hipStream_t>(opaque_stream);
    constexpr unsigned heads=24U,kv_heads=2U,dim=256U;
    const auto run=[&]<unsigned Rows>() {
        QsaRowsAttentionViews<Rows> v{};
        for(unsigned r=0;r<Rows;++r) {
            v.q_gate[r]=p.query+static_cast<std::uint64_t>(r)*heads*dim*2U;
            v.selected[r]=p.selected+static_cast<std::uint64_t>(r)*p.maximum_selected;
            v.scores[r]=p.scores+static_cast<std::uint64_t>(r)*heads*p.maximum_selected;
            v.output[r]=p.image+static_cast<std::uint64_t>(r)*heads*dim;v.count[r]=p.count[r];
        }
        const auto group=[&]<unsigned G>() {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_output_rows_window_group_kernel<Rows,G>),dim3(heads/G,Rows),dim3(dim),0,stream,
                v,p.kv,p.maximum_selected,heads,kv_heads,dim,p.codes,p.deltas);
        };
        switch(p.group) {
        case 0U:
            if(p.codes) hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_output_rows_dense_a8_kernel<Rows>),dim3(heads,Rows),dim3(dim),0,stream,
                v,p.kv,p.maximum_selected,heads,kv_heads,dim,p.codes,p.deltas);
            else hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_output_rows_kernel<Rows>),dim3(heads,Rows),dim3(dim),0,stream,
                v,p.kv,p.maximum_selected,heads,kv_heads,dim);
            break;
        case 2U: group.template operator()<2U>(); break;
        case 3U: group.template operator()<3U>(); break;
        case 4U: group.template operator()<4U>(); break;
        case 6U: group.template operator()<6U>(); break;
        case 12U: group.template operator()<12U>(); break;
        default: throw std::invalid_argument("window out-group probe: group must be 0|2|3|4|6|12");
        }
    };
    switch(p.rows) {
    case 2U: run.template operator()<2U>(); break;
    case 4U: run.template operator()<4U>(); break;
    case 8U: run.template operator()<8U>(); break;
    default: throw std::invalid_argument("window out-group probe: rows must be 2|4|8");
    }
    check_hip(hipGetLastError(),"window out-group probe launch");
}
void launch_hip_compact_qsa_probe(const HipCompactQsaProbe& p, void* opaque_stream) {
    const auto stream = static_cast<hipStream_t>(opaque_stream);
    if (!stream || !p.rows || p.rows % 8U || p.mode > 12U || !p.query || !p.kv || !p.selected ||
        !p.scores || !p.image || !p.weights || !p.projected || !p.status || !p.quantized || p.maximum_selected != 2052U || p.count > 32U)
        throw std::invalid_argument("compact QSA fixture views or geometry invalid");
    constexpr unsigned heads=24U, kv_heads=2U, dim=256U, width=heads*dim, hidden=2560U;
    const auto padded=(p.rows+63U)/64U*64U;
    const auto output_at=[&](unsigned r) { return reinterpret_cast<float*>(static_cast<unsigned char*>(p.image)+
        static_cast<std::uint64_t>(r)*width*(p.compact?2U:4U)); };
    if (p.mode <= 4U || p.mode==10U) {
        for (unsigned first=0;first<padded;first+=64U) {
            qsa_fast::Views<64U> v{};
            v.compact_output=p.compact;
            for(unsigned r=0;r<64U;++r) {
                const auto row=first+r;
                v.q_gate[r]=p.query+static_cast<std::uint64_t>(row)*width*2U;
                v.selected[r]=p.selected+static_cast<std::uint64_t>(row)*p.maximum_selected;
                v.scratch[r]=p.scores+static_cast<std::uint64_t>(row)*heads*p.maximum_selected;
                v.output[r]=output_at(row);v.count[r]=p.count;v.visible[r]=32U;
            }
            const unsigned splits=p.mode==4U?2U:1U, union_at=splits*heads*qsa_fast::kPartial;
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::union_kernel<64U,4U>),dim3(16),dim3(256),0,stream,v,union_at);
            if (p.mode==1U || p.mode==2U || p.mode==3U) {
                qsa_fast::Views<64U>* device=nullptr;
                check_hip(hipMallocAsync(reinterpret_cast<void**>(&device),sizeof(v),stream),"compact fixture views");
                // Host descriptor lives until its asynchronous copy has consumed it.
                check_hip(hipMemcpyAsync(device,&v,sizeof(v),hipMemcpyHostToDevice,stream),"compact fixture views copy");
                if(p.mode==1U) hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_order_kernel<64U,4U,3U>),
                    dim3(32),dim3(320),0,stream,device,p.kv,kv_heads,heads,union_at,2U);
                else if(p.mode==2U) hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_order_occ_kernel<64U,4U,3U>),
                    dim3(32),dim3(320),0,stream,device,p.kv,kv_heads,heads,union_at,2U);
                else {
                    std::uint16_t* shadow=nullptr;
                    check_hip(hipMallocAsync(reinterpret_cast<void**>(&shadow),32U*kv_heads*dim*4U,stream),"compact fixture shadow");
                    const auto words=32ULL*kv_heads*dim;
                    hipLaunchKernelGGL(qsa_fast::kv_to_f16_kernel,dim3(17),dim3(256),0,stream,
                        reinterpret_cast<const unsigned*>(p.kv),reinterpret_cast<unsigned*>(shadow),0ULL,words);
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_shadow_multi_kernel<64U,4U,3U>),
                        dim3(16,2),dim3(320),0,stream,device,shadow,kv_heads,heads,union_at);
                    check_hip(hipFreeAsync(shadow,stream),"compact fixture shadow free");
                }
                check_hip(hipFreeAsync(device,stream),"compact fixture views free");
                check_hip(hipStreamSynchronize(stream),"compact fixture descriptor lifetime");
            } else {
                if(p.mode==10U) hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_kernel<64U,4U,3U,3U,true>),dim3(16,2,splits),dim3(320),0,stream,
                    v,p.kv,kv_heads,heads,union_at,splits,0U);
                else hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_kernel<64U,4U,3U>),dim3(16,2,splits),dim3(320),0,stream,
                    v,p.kv,kv_heads,heads,union_at,splits,0U);
                if(splits>1U) hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::combine_kernel<64U>),dim3(heads,64),dim3(dim),0,stream,v,heads,splits);
            }
        }
    } else {
        for(unsigned first=0;first<p.rows;first+=8U) {
            QsaRowsAttentionViews<8U> v{};v.compact_output=p.compact;
            for(unsigned r=0;r<8U;++r) {
                const auto row=first+r;
                v.q_gate[r]=p.query+static_cast<std::uint64_t>(row)*width*2U;
                v.selected[r]=p.selected+static_cast<std::uint64_t>(row)*p.maximum_selected;
                v.scores[r]=p.scores+static_cast<std::uint64_t>(row)*heads*p.maximum_selected;
                v.output[r]=output_at(row);v.count[r]=p.count;
            }
            if(p.mode==11U) hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_output_rows_kernel<8U,true>),dim3(heads,8),dim3(dim),0,stream,
                v,p.kv,p.maximum_selected,heads,kv_heads,dim);
            else if(p.mode==12U) hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_output_rows_group_prefetch_kernel<8U,12U,true>),dim3(2,8),dim3(dim),0,stream,
                v,p.kv,p.maximum_selected,heads,kv_heads,dim);
            else if(p.mode==5U) hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_output_rows_kernel<8U>),dim3(heads,8),dim3(dim),0,stream,
                v,p.kv,p.maximum_selected,heads,kv_heads,dim);
            else if(p.mode==6U) hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_output_rows_group_kernel<8U,12U>),dim3(2,8),dim3(dim),0,stream,
                v,p.kv,p.maximum_selected,heads,kv_heads,dim);
            else if(p.mode==7U) hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_output_rows_group_prefetch_kernel<8U,12U>),dim3(2,8),dim3(dim),0,stream,
                v,p.kv,p.maximum_selected,heads,kv_heads,dim);
            else if(p.mode==8U) {
                // Exercise the in-place read after the compact store, including quantizer diagnostics.
                auto slot=q8_rows::dense_a8_slot(0U);
                hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_tail_rows_kernel<8U,false,true>),dim3(heads,8),dim3(dim),0,stream,
                    v,p.kv,p.maximum_selected,heads,kv_heads,dim,slot.codes,slot.deltas);
                hipLaunchKernelGGL(compact_qsa_quantizer_probe,dim3(192),dim3(256),0,stream,slot,static_cast<unsigned char*>(p.quantized));
            } else hipLaunchKernelGGL(compact_qsa_status_probe,dim3(heads,8),dim3(dim),0,stream,v,p.kv,p.maximum_selected,p.status);
        }
    }
    check_hip(hipMemsetAsync(q8_rows::dense_f16_overflow_counter(),0,sizeof(unsigned),stream),"compact fixture range reset");
    if(p.tiled_projection) {
        const auto project=[&]<class Input>() {
            QsaOutputTiledSite<Input> site{};
            site.segments=q8_rows::segments({{p.weights,hidden}});site.width=width;site.outputs=hidden;
            site.input0=static_cast<const unsigned char*>(p.image);site.input_stride=width*sizeof(Input);
            site.output0=reinterpret_cast<unsigned char*>(p.projected);site.output_stride=hidden*2U;
            q8_rows::tiled::launch(site,p.rows,stream);
        };
        if(p.compact) project.template operator()<std::uint16_t>();else project.template operator()<float>();
    } else for(unsigned first=0;first<p.rows;first+=8U) {
        QsaRowsOutputViews<8U> v{};v.compact=p.compact;
        for(unsigned r=0;r<8U;++r) {v.input[r]=output_at(first+r);v.output[r]=p.projected+(first+r)*hidden;}
        if(p.scalar_projection) launch_qsa_rows_output<WeightKind::f32,8U>(false,false,v,p.weights,width,hidden,stream);
        else launch_qsa_rows_output<WeightKind::q8_0,8U>(true,p.dense_a8_projection,v,p.weights,width,hidden,stream,p.fast_f16_projection);
    }
    check_hip(hipMemcpyAsync(p.status+1U,q8_rows::dense_f16_overflow_counter(),sizeof(unsigned),hipMemcpyDeviceToDevice,stream),"compact fixture range receipt");
    check_hip(hipGetLastError(),"compact QSA fixture launch");
}
void launch_hip_attention_kv_int8_probe(const HipKvInt8Probe& p, void* stream) {
    const auto owned = static_cast<hipStream_t>(stream);
    if (!owned || owned == hipStreamPerThread || owned == hipStreamLegacy || p.head_dim != kThreads || !p.kv_heads ||
        !p.attention_heads || p.attention_heads % p.kv_heads || (p.attention_heads / p.kv_heads) % 12U || !p.count ||
        p.count > p.maximum_selected || !p.decode_state || !p.rows_state || !p.q_gate || !p.selected || !p.scores_decode ||
        !p.out_decode || !p.scores_rows || !p.out_rows || !p.scores_group || !p.out_group)
        throw std::invalid_argument("int8 KV probe requires an owned stream, head_dim 256, 12-head KV groups and all views");
    check_hip(hipPeekAtLastError(), "int8 KV probe stale error");
    const auto kv_width = p.kv_heads * p.head_dim, attention_width = p.attention_heads * p.head_dim;
    const GraphPositionKernelArguments none{};
    QsaRowsStoreViews<8U> store{};
    for (unsigned r = 0; r < 8U; ++r) {
        hipLaunchKernelGGL(store_kv_int8_kernel, dim3(p.kv_heads, 2U), dim3(kThreads), 0, owned, p.k[r], p.v[r], p.decode_state,
            kv_width, p.kv_heads, p.head_dim, p.position[r], none);
        store.k[r] = p.k[r]; store.v[r] = p.v[r]; store.position[r] = p.position[r];
    }
    hipLaunchKernelGGL(HIP_KERNEL_NAME(store_rows_kv_int8_kernel<8U>), dim3(p.kv_heads, 2U, 8U), dim3(kThreads), 0, owned, store,
        p.rows_state, kv_width, p.kv_heads, p.head_dim);
    const auto* kv = static_cast<const std::uint16_t*>(p.decode_state);
    const GraphSelectionArguments selection{};
    hipLaunchKernelGGL(attention_score_int8_kernel, dim3(p.attention_heads * p.count), dim3(kThreads), 0, owned, p.q_gate, kv,
        p.selected, p.count, p.maximum_selected, p.attention_heads, p.kv_heads, p.head_dim, p.scores_decode, selection);
    hipLaunchKernelGGL(attention_softmax_kernel, dim3(p.attention_heads), dim3(kThreads), 0, owned, p.scores_decode, p.count,
        p.maximum_selected, selection);
    hipLaunchKernelGGL(attention_output_int8_kernel, dim3(p.attention_heads), dim3(p.head_dim), 0, owned, p.q_gate, kv, p.selected,
        p.scores_decode, p.count, p.maximum_selected, p.attention_heads, p.kv_heads, p.head_dim, p.out_decode, selection);
    // Every row of the window (8, wave score with 4 slots per wave) and prefill group (16 rows,
    // 12 heads per score wave / output block, 16 slots per wave, prefetching forms) attends the
    // same query and selection.
    const auto views = [&]<unsigned Rows>(float* scores, float* output) {
        QsaRowsAttentionViews<Rows> v{};
        for (unsigned r = 0; r < Rows; ++r) {
            v.q_gate[r] = p.q_gate; v.selected[r] = p.selected; v.count[r] = p.count;
            v.scores[r] = scores + static_cast<std::uint64_t>(r) * p.attention_heads * p.maximum_selected;
            v.output[r] = output + static_cast<std::uint64_t>(r) * attention_width;
        }
        return v;
    };
    const auto window = views.template operator()<8U>(p.scores_rows, p.out_rows);
    const auto waves = 8ULL * p.attention_heads * ((p.count + 3U) / 4U);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_score_rows_wave_kernel<8U, 4U, true>), dim3(static_cast<std::uint32_t>((waves + 7U) / 8U)),
        dim3(kThreads), 0, owned, window, kv, p.maximum_selected, p.attention_heads, p.kv_heads, p.head_dim, p.count);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_softmax_rows_kernel<8U>), dim3(p.attention_heads, 8U), dim3(kThreads), 0, owned,
        window, p.maximum_selected);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_output_rows_kernel<8U, true>), dim3(p.attention_heads, 8U), dim3(p.head_dim), 0, owned,
        window, kv, p.maximum_selected, p.attention_heads, p.kv_heads, p.head_dim);
    const auto group = views.template operator()<16U>(p.scores_group, p.out_group);
    const auto group_waves = 16ULL * (p.attention_heads / 12U) * ((p.count + 15U) / 16U);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_score_rows_group256_kernel<16U, 12U, 16U, true>),
        dim3(static_cast<std::uint32_t>((group_waves + 7U) / 8U)), dim3(kThreads), 0, owned, group, kv, p.maximum_selected,
        p.attention_heads, p.kv_heads, p.head_dim, p.count);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_softmax_rows_kernel<16U>), dim3(p.attention_heads, 16U), dim3(kThreads), 0, owned,
        group, p.maximum_selected);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_output_rows_group_prefetch_kernel<16U, 12U, true>), dim3(p.attention_heads / 12U, 16U),
        dim3(p.head_dim), 0, owned, group, kv, p.maximum_selected, p.attention_heads, p.kv_heads, p.head_dim);
    check_hip(hipGetLastError(), "int8 KV probe launch");
}
#endif

#ifdef STRIX_BUILD_QSA_OUTPUT_PROJECTION_DIAGNOSTIC
void launch_hip_qsa_output_projection_probe(HipQsaOutputProjectionMode mode,
    HipQsaInputProjectionView input, HipQsaInputProjectionView weights,
    HipQsaInputProjectionView output, std::uint32_t width, std::uint32_t outputs, void* stream) {
    if ((mode != HipQsaOutputProjectionMode::legacy256 && mode != HipQsaOutputProjectionMode::wave8) ||
        !stream || stream == hipStreamPerThread || stream == hipStreamLegacy)
        throw std::invalid_argument("QSA output projection diagnostic requires mode and owned stream");
    q8_projection_organization::validate_fp32_bf16_views(width, outputs,
        {input.address, input.bytes}, {weights.address, weights.bytes}, {output.address, output.bytes});
    auto* in = reinterpret_cast<const float*>(input.address);
    auto* out = reinterpret_cast<std::uint16_t*>(output.address);
    const auto* weight = reinterpret_cast<const void*>(weights.address);
    const auto owned_stream = static_cast<hipStream_t>(stream);
    if (mode == HipQsaOutputProjectionMode::legacy256)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(linear_float_to_bf16_kernel<WeightKind::q8_0>),
            dim3(outputs), dim3(256), 0, owned_stream, in, weight, width, outputs, out);
    else
        hipLaunchKernelGGL(linear_prefetch8_wave8_q8_float_to_bf16_kernel,
            dim3((outputs + 7U) / 8U), dim3(256), 0, owned_stream, in, weight, width, outputs, out);
    check_hip(hipGetLastError(), "HIP QSA output projection diagnostic launch");
}
#endif
#ifdef STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC
namespace {
template <unsigned Rows>
void enqueue_qsa_rows_probe(bool input_side, bool a8, std::span<const HipQsaInputProjectionView> input,
    HipQsaInputProjectionView weights, std::span<const HipQsaInputProjectionView> output, std::uint32_t width,
    std::uint32_t outputs, hipStream_t stream) {
    const auto segments = q8_rows::segments({{reinterpret_cast<const void*>(weights.address), outputs}});
    if (input_side) {
        QsaInputRowsSite<Rows> site{};
        site.segments = segments; site.width = width; site.outputs = outputs;
        for (unsigned r = 0; r < Rows; ++r) {
            site.input[r] = reinterpret_cast<const void*>(input[r].address);
            site.output[0][r] = reinterpret_cast<float*>(output[r].address);
        }
        q8_rows::launch(site, a8, stream);
    } else {
        QsaOutputRowsSite<Rows> site{};
        site.segments = segments; site.width = width; site.outputs = outputs;
        for (unsigned r = 0; r < Rows; ++r) {
            site.input[r] = reinterpret_cast<const void*>(input[r].address);
            site.output[r] = reinterpret_cast<std::uint16_t*>(output[r].address);
        }
        q8_rows::launch(site, a8, stream);
    }
}
} // namespace
void launch_hip_qsa_rows_projection_probe(HipQsaRowsProjection projection, HipQsaRowsProbeMode mode,
    std::span<const HipQsaInputProjectionView> input, HipQsaInputProjectionView weights,
    std::span<const HipQsaInputProjectionView> output, std::uint32_t width, std::uint32_t outputs, void* stream) {
    const auto rows = static_cast<std::uint32_t>(input.size());
    // The fixture rewrites weights in place between calls (STRIX_Q8_ROWS_LANE_PROBE_CACHE=1 keeps the
    // lane kernel's repacked weights, for timing one unchanged matrix).
    if (!q8_rows::env_unsigned("STRIX_Q8_ROWS_LANE_PROBE_CACHE", 0U)) q8_rows::invalidate_lane_weights();
    const bool input_side = projection == HipQsaRowsProjection::input;
    if ((rows != 1U && rows != 2U && rows != 4U && rows != 8U && rows != 16U && rows != 32U && rows != 64U) || output.size() != rows ||
        (!input_side && projection != HipQsaRowsProjection::output) ||
        (mode != HipQsaRowsProbeMode::rows && mode != HipQsaRowsProbeMode::rows_a8 && mode != HipQsaRowsProbeMode::row_loop) ||
        !stream || static_cast<hipStream_t>(stream) == hipStreamLegacy || static_cast<hipStream_t>(stream) == hipStreamPerThread)
        throw std::invalid_argument("QSA rows projection diagnostic requires 1/2/4/8/16/32/64 rows, a mode and an owned stream");
    const auto overlap = [](HipQsaInputProjectionView x, HipQsaInputProjectionView y) {
        return x.address < y.address + y.bytes && y.address < x.address + x.bytes;
    };
    for (unsigned r = 0; r < rows; ++r) {
        const q8_projection_organization::View x{input[r].address, input[r].bytes}, w{weights.address, weights.bytes},
            y{output[r].address, output[r].bytes};
        if (input_side) q8_projection_organization::validate_views(width, outputs, x, w, y);
        else q8_projection_organization::validate_fp32_bf16_views(width, outputs, x, w, y);
        if (overlap(output[r], weights)) throw std::invalid_argument("QSA rows projection output aliases weights");
        for (unsigned other = 0; other < rows; ++other)
            if (overlap(output[r], input[other]) || (other < r && overlap(output[r], output[other])))
                throw std::invalid_argument("QSA rows projection output aliases a row");
    }
    if (mode == HipQsaRowsProbeMode::rows_a8 && !q8_rows::dense_a8_width(width))
        throw std::invalid_argument("QSA rows dense A8 width is not admitted");
    const auto native = static_cast<hipStream_t>(stream);
    check_hip(hipPeekAtLastError(), "QSA rows projection diagnostic pre-enqueue stale error");
    const auto* packed = reinterpret_cast<const void*>(weights.address);
    if (mode == HipQsaRowsProbeMode::row_loop) {
        for (unsigned r = 0; r < rows; ++r) {
            if (input_side) hipLaunchKernelGGL(linear_prefetch8_q8_qsa_input_kernel, dim3(outputs), dim3(32), 0, native,
                reinterpret_cast<const std::uint16_t*>(input[r].address), packed, width, outputs, reinterpret_cast<float*>(output[r].address));
            else hipLaunchKernelGGL(linear_prefetch8_wave8_q8_float_to_bf16_kernel, dim3((outputs + 7U) / 8U), dim3(256), 0, native,
                reinterpret_cast<const float*>(input[r].address), packed, width, outputs, reinterpret_cast<std::uint16_t*>(output[r].address));
            check_hip(hipGetLastError(), "QSA rows projection diagnostic row-loop launch");
        }
        return;
    }
    const bool a8 = mode == HipQsaRowsProbeMode::rows_a8;
    if (rows == 1U) enqueue_qsa_rows_probe<1U>(input_side, a8, input, weights, output, width, outputs, native);
    else if (rows == 2U) enqueue_qsa_rows_probe<2U>(input_side, a8, input, weights, output, width, outputs, native);
    else if (rows == 4U) enqueue_qsa_rows_probe<4U>(input_side, a8, input, weights, output, width, outputs, native);
    else if (rows == 8U) enqueue_qsa_rows_probe<8U>(input_side, a8, input, weights, output, width, outputs, native);
    else if (rows == 16U) enqueue_qsa_rows_probe<16U>(input_side, a8, input, weights, output, width, outputs, native);
    else if (rows == 32U) enqueue_qsa_rows_probe<32U>(input_side, a8, input, weights, output, width, outputs, native);
    else enqueue_qsa_rows_probe<64U>(input_side, a8, input, weights, output, width, outputs, native);
    check_hip(hipGetLastError(), "QSA rows projection diagnostic launch");
}
namespace {
template <unsigned Rows>
void enqueue_qsa_gsq_rows_probe(bool input_side, std::span<const std::uint32_t> types, std::span<const HipQsaInputProjectionView> weights,
    std::span<const std::uint32_t> counts, std::span<const HipQsaInputProjectionView> input,
    std::span<const HipQsaInputProjectionView> output, std::uint32_t width, hipStream_t stream) {
    q8_rows::Segments segments{};
    for (unsigned m = 0; m < types.size(); ++m) {
        segments.weights[m] = reinterpret_cast<const unsigned char*>(weights[m].address);
        segments.types[m] = q8_rows::segment_type(types[m]);
        segments.first[m + 1U] = segments.first[m] + counts[m];
    }
    segments.count = static_cast<unsigned>(types.size());
    if (input_side) {
        QsaInputRowsSite<Rows> site{};
        site.segments = segments; site.width = width; site.outputs = segments.first[segments.count];
        for (unsigned r = 0; r < Rows; ++r) {
            site.input[r] = reinterpret_cast<const void*>(input[r].address);
            for (unsigned m = 0; m < segments.count; ++m) site.output[m][r] = reinterpret_cast<float*>(output[m * Rows + r].address);
        }
        q8_rows::launch(site, true, stream);
    } else {
        QsaOutputRowsSite<Rows> site{};
        site.segments = segments; site.width = width; site.outputs = segments.first[1];
        for (unsigned r = 0; r < Rows; ++r) {
            site.input[r] = reinterpret_cast<const void*>(input[r].address);
            site.output[r] = reinterpret_cast<std::uint16_t*>(output[r].address);
        }
        q8_rows::launch(site, true, stream);
    }
}
// 16 / 32 rows of one typed kind: the cohort window dispatcher (launch_qsa_rows_input with typed_rows8).
template <unsigned Rows>
void enqueue_qsa_gsq_cohort_probe(std::span<const std::uint32_t> types, std::span<const HipQsaInputProjectionView> weights,
    std::span<const std::uint32_t> counts, std::span<const HipQsaInputProjectionView> input,
    std::span<const HipQsaInputProjectionView> output, std::uint32_t width, hipStream_t stream) {
    QsaRowsInputViews<Rows> views{};
    for (unsigned m = 0; m < types.size(); ++m) {
        views.weights[m] = reinterpret_cast<const void*>(weights[m].address);
        views.first[m + 1U] = views.first[m] + counts[m];
        for (unsigned r = 0; r < Rows; ++r) views.output[m][r] = reinterpret_cast<float*>(output[m * Rows + r].address);
    }
    views.count = static_cast<std::uint32_t>(types.size());
    for (unsigned r = 0; r < Rows; ++r) views.input[r] = reinterpret_cast<const std::uint16_t*>(input[r].address);
    dispatch_weight_kind(types[0], [&](auto kind) {
        launch_qsa_rows_input<decltype(kind)::value, Rows>(false, true, views, width, stream, false, true);
    });
}
} // namespace
void launch_hip_qsa_gsq_rows_probe(HipQsaRowsProjection projection, std::span<const std::uint32_t> types,
    std::span<const HipQsaInputProjectionView> weights, std::span<const std::uint32_t> counts,
    std::span<const HipQsaInputProjectionView> input, std::span<const HipQsaInputProjectionView> output,
    std::uint32_t width, void* stream) {
    const auto rows = static_cast<unsigned>(input.size());
    const bool input_side = projection == HipQsaRowsProjection::input;
    const bool cohort = rows == 16U || rows == 32U;
    if ((rows != 1U && rows != 2U && rows != 4U && rows != 8U && !(cohort && input_side)) || types.empty() || types.size() > (input_side ? 5U : 1U) ||
        weights.size() != types.size() || counts.size() != types.size() || output.size() != types.size() * rows ||
        (!input_side && projection != HipQsaRowsProjection::output) || !q8_rows::dense_a8_width(width) ||
        !stream || static_cast<hipStream_t>(stream) == hipStreamLegacy || static_cast<hipStream_t>(stream) == hipStreamPerThread)
        throw std::invalid_argument("QSA GSQ typed rows diagnostic requires 1/2/4/8 rows, 1..5 input (1 output) segments and an owned stream");
    for (unsigned m = 0; m < types.size(); ++m)
        if (!gsq_a8::dense_type(types[m]) || weights[m].bytes < gsq_a8::dense_row_bytes(types[m], width) * counts[m])
            throw std::invalid_argument("QSA GSQ typed rows diagnostic segment type or weight extent");
    for (unsigned m = 0; m < types.size(); ++m)
        if (width % gsq_a8::dense_block_elements(types[m]) || (cohort && (types[m] != types[0] || types[m] == 8U)))
            throw std::invalid_argument("QSA GSQ typed rows diagnostic: width is not whole blocks, or a 16/32-row cohort is not one typed kind");
    const auto native = static_cast<hipStream_t>(stream);
    check_hip(hipPeekAtLastError(), "QSA GSQ typed rows diagnostic pre-enqueue stale error");
    if (rows == 16U) enqueue_qsa_gsq_cohort_probe<16U>(types, weights, counts, input, output, width, native);
    else if (rows == 32U) enqueue_qsa_gsq_cohort_probe<32U>(types, weights, counts, input, output, width, native);
    else if (rows == 1U) enqueue_qsa_gsq_rows_probe<1U>(input_side, types, weights, counts, input, output, width, native);
    else if (rows == 2U) enqueue_qsa_gsq_rows_probe<2U>(input_side, types, weights, counts, input, output, width, native);
    else if (rows == 4U) enqueue_qsa_gsq_rows_probe<4U>(input_side, types, weights, counts, input, output, width, native);
    else enqueue_qsa_gsq_rows_probe<8U>(input_side, types, weights, counts, input, output, width, native);
    check_hip(hipGetLastError(), "QSA GSQ typed rows diagnostic launch");
}
#endif


void register_hip_indexed_attention_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipIndexedAttentionLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification) {
    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }
    validate_config(config);
    if (request.qsa_gqa_heads_per_block != config.qsa_gqa_heads_per_block)
        throw std::invalid_argument("QSA GQA request/config mismatch");
    if (config.qsa_gqa_heads_per_block != 1 && (!request.deferred_moe_status || request.device_graph_inputs || request.require_graph_safe))
        throw std::invalid_argument("QSA GQA organization registers eager D1 variants only");
    if (request.qsa_hierarchical_selection != config.qsa_hierarchical_selection)
        throw std::invalid_argument("hierarchical QSA request/config mismatch");
    if (config.qsa_hierarchical_selection && (!request.deferred_moe_status || request.device_graph_inputs || request.require_graph_safe))
        throw std::invalid_argument("hierarchical QSA registers eager D1 variants only");
    if (request.wave8_q8_qsa_output != config.wave8_q8_output)
        throw std::invalid_argument("wave8 QSA output request/config mismatch");
    if (config.wave8_q8_output && (request.device_graph_inputs || request.require_graph_safe))
        throw std::invalid_argument("wave8 QSA output registers eager variants only");
    if(config.prefetch_q8_input!=request.prefetch_q8_qsa_input || config.prefetch_eligible_layers!=request.qsa_input_prefetch_layers)throw std::invalid_argument("indexed QSA prefetch config differs from binding request");
    qsa_input_prefetch::validate_layers(model,request.prefetch_q8_qsa_input,request.qsa_input_prefetch_layers);

    if (config.model.hidden_size !=
            model.hidden_size ||
        config.model.attention_heads !=
            model.attention_heads ||
        config.model.kv_heads !=
            model.kv_heads ||
        config.model.head_dim !=
            model.head_dim ||
        config.model.indexer_n_heads !=
            model.indexer_n_heads ||
        config.model.indexer_head_dim !=
            model.indexer_head_dim ||
        config.model.indexer_budget !=
            model.indexer_budget ||
        config.model.indexer_compress_ratio !=
            model.indexer_compress_ratio) {
        throw std::invalid_argument(
            "HIP indexed-attention config does not match model contract");
    }
    if (request.logical_rows != 1 ||
        request.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP indexed-attention v1 registers only C1 one-row variants");
    }

    if (request.device_graph_inputs && (!request.require_graph_safe || !config.enable_device_graph_inputs)) {
        throw std::invalid_argument("indexed device graph inputs require explicit graph configuration");
    }

    // QSA top-k ties remain source-undefined. Keep v1 experimental until a
    // canonical tie contract or tie-free checkpoint fixtures qualify it.
    if (qualification !=
        kernels::QualificationStatus::
            experimental) {
        throw std::invalid_argument(
            "HIP indexed-attention v1 remains experimental while QSA tie semantics are unresolved");
    }

    kernels::KernelVariant variant;
    variant.id =
        kernel_id(
            request,
            config);
    variant.family =
        kernels::OperatorFamily::
            indexed_attention;
    variant.phase =
        kernels::ExecutionPhase::decode;
    variant.shape =
        execution_shape_for(
            model,
            SerialOp::indexed_attention,
            request);
    variant.wave =
        kernels::WaveSize::wave32;
    variant.resources.workgroup_threads =
        kThreads;
    variant.resources.lds_bytes =
        2U *
        kThreads *
        sizeof(float);
    variant.resources.scratch_bytes =
        build_indexed_attention_workspace_layout(
            model, 256, config.qsa_hierarchical_selection, config.qsa_gqa_heads_per_block).total_bytes;
    if (config.qsa_hierarchical_selection) variant.resources.lds_bytes = 8192;
    if (config.qsa_gqa_heads_per_block != 1) variant.resources.lds_bytes =
        std::max(variant.resources.lds_bytes, config.qsa_gqa_heads_per_block * 1024U);
    variant.numerics.input_precision =
        "bf16";
    variant.numerics.accumulator_precision =
        "fp32";
    variant.numerics.output_precision =
        "bf16";
    variant.numerics.row_invariant = true;
    variant.numerics.deterministic = true;
    variant.qualification =
        qualification;
    variant.graph_safe = request.device_graph_inputs && config.enable_device_graph_inputs;
    variant.input_layout =
        "row-major+layer-kv+raw-index";
    variant.output_layout =
        "row-major";
    variant.compiler_identity =
        config.prefetch_q8_input ? "hip-gfx1151-indexed-attention-v3-prefetch8-virtual256-q8-qsa-input" : "hip-gfx1151-indexed-attention-v2";
    if (config.wave8_q8_output) variant.compiler_identity += ":wave8-q8-qsa-output-v1";
    if (config.dense_a8_projections) {
        variant.numerics.accumulator_precision += ";" + std::string(q8_projection_organization::dense_a8_numerical_identity);
        variant.compiler_identity += ":dense-a8-q80-v1";
    }
    if (config.qsa_hierarchical_selection) variant.compiler_identity += ":hierarchical-top512-v1";
    if (config.qsa_gqa_heads_per_block != 1) variant.compiler_identity += ":gqa-heads" + std::to_string(config.qsa_gqa_heads_per_block) + "-v1";
    variant.measured_seconds = 0.0;

    registry.add(
        std::move(variant));
}

ExecutionLauncher
make_hip_indexed_attention_launcher(
    std::string kernel_id_value,
    HipIndexedAttentionLauncherConfig* config) {
    if (kernel_id_value.empty() ||
        config == nullptr) {
        throw std::invalid_argument(
            "HIP indexed-attention launcher requires id and config");
    }
    validate_config(*config);
    if (config->wave8_q8_output || config->qsa_hierarchical_selection || config->qsa_gqa_heads_per_block != 1 ||
        kernel_id_value.find(".gqa-heads2-v1") != std::string::npos || kernel_id_value.find(".gqa-heads4-v1") != std::string::npos ||
        kernel_id_value.find(qsa_hierarchical_selection_marker) != std::string::npos ||
        kernel_id_value.find(q8_projection_organization::qsa_output_wave8_marker) != std::string::npos) {
        ExecutionBindingRequest request;
        if (kernel_id_value != kernel_id(request, *config))
            throw std::invalid_argument("wave8 QSA output launcher requires its exact eager C1 kernel id");
    }
    return ExecutionLauncher{
        std::move(kernel_id_value),
        implementation_identity(
            *config),
        &launch_indexed_attention,
        config,
        &preflight_indexed_attention,
    };
}

void HipIndexedAttentionLauncherBundle::
register_for_compiled_execution(
    const CompiledSerialExecution& compiled,
    ExecutionLauncherRegistry& registry) {
    if (compiled.execution_identity == 0 ||
        compiled.execution.steps.size() !=
            compiled.bindings.bindings.size()) {
        throw std::invalid_argument(
            "HIP indexed-attention bundle requires valid compiled execution");
    }

    // GSQ-RCO (STRIX_GSQ_Q20_A8=1): a file without Q8_0 QSA inputs has no prefetch-bound step.
    const bool gsq_files=strix::moe_q4k_a8::q20_a8_enabled();
    if(config_.prefetch_q8_input && !gsq_files && std::none_of(compiled.bindings.bindings.begin(),
        compiled.bindings.bindings.end(),[](const auto& binding){return binding.op==SerialOp::indexed_attention &&
            binding.kernel_id && binding.kernel_id->ends_with(qsa_input_prefetch::kernel_suffix);}))
        throw std::invalid_argument("QSA prefetch bundle requires eligible bound step");
    ExecutionBindingRequest shape;shape.logical_rows=compiled.logical_rows;shape.request_concurrency=compiled.request_concurrency;shape.device_graph_inputs=compiled.device_graph_inputs;
    std::vector<std::string> registered;bool saw_prefetch=false;
    for(std::size_t i=0;i<compiled.execution.steps.size();++i){
        if(compiled.execution.steps[i].op!=SerialOp::indexed_attention)continue;
        const auto& binding=compiled.bindings.bindings[i];
        if(binding.host_or_runtime_step || !binding.kernel_id || binding.kernel_id->empty())throw std::invalid_argument("HIP indexed-attention step has no kernel binding");
        const bool prefetch=binding.kernel_id->ends_with(qsa_input_prefetch::kernel_suffix);
        if(prefetch!=qsa_input_prefetch::selected(config_.prefetch_q8_input,config_.prefetch_eligible_layers,compiled.execution.steps[i].layer))throw std::invalid_argument("HIP indexed QSA prefetch binding/config mismatch");
        auto* selected=prefetch?&config_:&fallback_config_;
        if(*binding.kernel_id!=kernel_id(shape,*selected))throw std::invalid_argument("HIP indexed-attention bundle permits only its base/prefetch pair");
        if(std::find(registered.begin(),registered.end(),*binding.kernel_id)==registered.end()){registry.add_kernel_launcher(make_hip_indexed_attention_launcher(*binding.kernel_id,selected));registered.push_back(*binding.kernel_id);}
        saw_prefetch|=prefetch;
    }
    if(registered.empty())throw std::invalid_argument("compiled execution has no indexed-attention step");
    if(config_.prefetch_q8_input && !saw_prefetch && !gsq_files)throw std::invalid_argument("QSA prefetch bundle requires eligible bound step");
}

std::vector<std::uint64_t>
HipIndexedAttentionLauncherBundle::known_window_position_readers(
    const CompiledSerialExecution& compiled,
    const ExecutionLauncherRegistry& registry) const {
    validate_config(config_);
    validate_config(fallback_config_);
    if (config_.enable_device_graph_inputs || fallback_config_.enable_device_graph_inputs ||
        compiled.device_graph_inputs || compiled.logical_rows != 1 ||
        compiled.request_concurrency != 1 || compiled.execution_identity == 0 ||
        compiled.execution.steps.size() != compiled.bindings.bindings.size()) {
        throw std::invalid_argument("QSA known-window readers require a valid eager C1 execution");
    }
    ExecutionBindingRequest shape;
    shape.logical_rows = compiled.logical_rows;
    shape.request_concurrency = compiled.request_concurrency;
    shape.device_graph_inputs = compiled.device_graph_inputs;
    std::vector<std::uint64_t> identities;
    bool saw_prefetch = false;
    for (std::size_t i = 0; i < compiled.execution.steps.size(); ++i) {
        const auto& step = compiled.execution.steps[i];
        const auto& binding = compiled.bindings.bindings[i];
        if (step.op != SerialOp::indexed_attention) {
            if (binding.op == SerialOp::indexed_attention)
                throw std::invalid_argument("QSA known-window binding operator differs from execution");
            continue;
        }
        if (step.id != i || binding.step_id != step.id || binding.op != step.op ||
            !step.layer || binding.host_or_runtime_step || !binding.kernel_id || binding.kernel_id->empty()) {
            throw std::invalid_argument("QSA known-window reader has invalid step binding");
        }
        const bool prefetch = qsa_input_prefetch::selected(
            config_.prefetch_q8_input, config_.prefetch_eligible_layers, step.layer);
        const auto* selected = prefetch ? &config_ : &fallback_config_;
        if (*binding.kernel_id != kernel_id(shape, *selected))
            throw std::invalid_argument("QSA known-window reader differs from bundle configuration");
        const auto* launcher = registry.find_kernel_launcher(*binding.kernel_id);
        const auto identity = implementation_identity(*selected);
        if (!launcher || launcher->key != *binding.kernel_id || identity == 0 ||
            launcher->implementation_identity != identity || launcher->function != &launch_indexed_attention ||
            launcher->preflight != &preflight_indexed_attention || launcher->user_data != selected) {
            throw std::invalid_argument("QSA known-window reader is not this bundle's exact launcher");
        }
        if (std::find(identities.begin(), identities.end(), identity) == identities.end())
            identities.push_back(identity);
        saw_prefetch |= prefetch;
    }
    if (identities.empty() || (config_.prefetch_q8_input && !saw_prefetch && !strix::moe_q4k_a8::q20_a8_enabled()))
        throw std::invalid_argument("QSA known-window bundle has no admitted eligible reader");
    return identities;
}

namespace {
template <SmallWeightKind Kind>
void launch_graph_attention_state_typed(const GraphAttentionStatePlan& p,
    std::uint64_t scalar_position, hipStream_t stream) {
    const auto& r = p.request;
    GraphPositionKernelArguments dynamic;
    dynamic.context_tokens = r.context_tokens;
    dynamic.status = reinterpret_cast<std::uint32_t*>(r.status.arena_base);
    if (r.position_mode == AttentionPositionMode::device_input) {
        dynamic.position = reinterpret_cast<const std::uint64_t*>(
            r.arguments.memory.arena_base + r.arguments.position_offset);
    }
    hipLaunchKernelGGL(initialize_graph_attention_status, dim3(1), dim3(1), 0, stream,
        dynamic.status, scalar_position, dynamic);
    check_hip(hipGetLastError(), "HIP attention state status initialization");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(validate_graph_attention_sources<Kind>), dim3(32),
        dim3(kThreads), 0, stream, reinterpret_cast<const float*>(r.query.address),
        reinterpret_cast<const float*>(r.key.address), reinterpret_cast<const float*>(r.value.address),
        reinterpret_cast<const void*>(r.query_norm.address), reinterpret_cast<const void*>(r.key_norm.address),
        p.query_elements, p.kv_elements, r.head_dim, r.query_head_stride, dynamic.status);
    check_hip(hipGetLastError(), "HIP attention state finite-source validation");
    ModelContract model;
    model.head_dim = r.head_dim;
    model.partial_rotary_factor = static_cast<float>(r.rotary_dim) / static_cast<float>(r.head_dim);
    model.rms_norm_epsilon = r.epsilon;
    model.rope_theta = r.theta;
    launch_norm_rope_typed<Kind>(reinterpret_cast<float*>(r.query.address), r.query_heads,
        r.head_dim, r.query_head_stride, r.query_norm, model, scalar_position, dynamic, stream);
    launch_norm_rope_typed<Kind>(reinterpret_cast<float*>(r.key.address), r.kv_heads,
        r.head_dim, r.head_dim, r.key_norm, model, scalar_position, dynamic, stream);
    const auto width = static_cast<std::uint32_t>(p.kv_elements);
    hipLaunchKernelGGL(store_kv_kernel, dim3((width - 1U) / kThreads + 1U), dim3(kThreads),
        0, stream, reinterpret_cast<const float*>(r.key.address), reinterpret_cast<const float*>(r.value.address),
        reinterpret_cast<std::uint16_t*>(r.kv_state.address), width, scalar_position, dynamic);
    check_hip(hipGetLastError(), "HIP attention state dynamic KV write");
}
} // namespace

void launch_hip_graph_attention_state(const GraphAttentionStatePlan& plan,
    std::uint64_t scalar_control_position, void* stream_pointer) {
    validate_graph_attention_state_plan(plan);
    if (stream_pointer == nullptr) throw std::invalid_argument("attention state requires an explicitly owned stream");
    const auto stream = static_cast<hipStream_t>(stream_pointer);
    if (plan.request.position_mode == AttentionPositionMode::scalar_control) {
        if (scalar_control_position >= plan.request.context_tokens) throw std::out_of_range("scalar attention position exceeds context");
        hipStreamCaptureStatus capturing;
        check_hip(hipStreamIsCapturing(stream, &capturing), "HIP attention control capture query");
        if (capturing != hipStreamCaptureStatusNone) throw std::invalid_argument("scalar attention control cannot be captured as a dynamic plan");
    } else if (scalar_control_position != 0) {
        throw std::invalid_argument("device-input attention mode does not accept a scalar position override");
    }
    switch (small_weight_kind(plan.request.norm_ggml_type)) {
    case SmallWeightKind::f32: launch_graph_attention_state_typed<SmallWeightKind::f32>(plan, scalar_control_position, stream); return;
    case SmallWeightKind::f16: launch_graph_attention_state_typed<SmallWeightKind::f16>(plan, scalar_control_position, stream); return;
    case SmallWeightKind::bf16: launch_graph_attention_state_typed<SmallWeightKind::bf16>(plan, scalar_control_position, stream); return;
    }
    throw std::logic_error("unreachable attention norm encoding");
}

namespace {
// Shared grouping of independent heads. Each head retains the incumbent
// 256-feature product/store and 128->1 tree. Only the BF16 K load is shared.
template <std::uint32_t Heads>
__global__ void qsa_gqa_score_diagnostic_kernel(
    const float* q_gate, const std::uint16_t* kv_state, const std::uint32_t* selected,
    std::uint32_t selected_count, std::uint32_t maximum_selected,
    std::uint32_t attention_heads, std::uint32_t kv_heads, std::uint32_t head_dim,
    float* scores, GraphSelectionArguments dynamic) {
    if (!graph_selected_count(dynamic, selected_count)) return;
    const auto flat = static_cast<std::uint32_t>(blockIdx.x);
    const auto selected_slot = flat % maximum_selected;
    if (selected_slot >= selected_count) return;
    const auto group = attention_heads / kv_heads;
    const auto blocks_per_group = (group + Heads - 1U) / Heads;
    const auto head_block = flat / maximum_selected;
    const auto kv_head = head_block / blocks_per_group;
    const auto first_head = kv_head * group + (head_block % blocks_per_group) * Heads;
    const auto feature = static_cast<std::uint32_t>(threadIdx.x);
    const auto kv_width = kv_heads * head_dim;
    const auto row_base = static_cast<std::uint64_t>(selected[selected_slot]) * kv_width * 2ULL;
    float key = 0.0F;
    if (feature < head_dim)
        key = bf16_to_float(kv_state[row_base + static_cast<std::uint64_t>(kv_head) * head_dim + feature]);
    __shared__ float scratch[Heads][kThreads];
#pragma unroll
    for (std::uint32_t local = 0; local < Heads; ++local) {
        const auto head = first_head + local;
        float product = 0.0F;
        if (head < (kv_head + 1U) * group && feature < head_dim)
            product = q_gate[static_cast<std::uint64_t>(head) * head_dim * 2ULL + feature] * key;
        scratch[local][feature] = product;
    }
    __syncthreads();
    for (std::uint32_t stride = kThreads / 2U; stride > 0; stride >>= 1U) {
        if (feature < stride) {
#pragma unroll
            for (std::uint32_t local = 0; local < Heads; ++local)
                scratch[local][feature] += scratch[local][feature + stride];
        }
        __syncthreads();
    }
    if (feature == 0) {
#pragma unroll
        for (std::uint32_t local = 0; local < Heads; ++local) {
            const auto head = first_head + local;
            if (head < (kv_head + 1U) * group) {
                if (dynamic.position.status != nullptr && !isfinite(scratch[local][0])) atomicOr(dynamic.position.status, 8U);
                scores[static_cast<std::uint64_t>(head) * maximum_selected + selected_slot] =
                    scratch[local][0] / sqrtf(static_cast<float>(head_dim));
            }
        }
    }
}

// Each independent accumulator visits slots in the incumbent chronological
// order and keeps the same score*value operand order and BF16/gate epilogue.
template <std::uint32_t Heads>
__global__ void qsa_gqa_value_diagnostic_kernel(
    const float* q_gate, const std::uint16_t* kv_state, const std::uint32_t* selected,
    const float* scores, std::uint32_t selected_count, std::uint32_t maximum_selected,
    std::uint32_t attention_heads, std::uint32_t kv_heads, std::uint32_t head_dim,
    float* output, GraphSelectionArguments dynamic) {
    if (!graph_selected_count(dynamic, selected_count)) return;
    const auto group = attention_heads / kv_heads;
    const auto blocks_per_group = (group + Heads - 1U) / Heads;
    const auto head_block = static_cast<std::uint32_t>(blockIdx.x);
    const auto kv_head = head_block / blocks_per_group;
    const auto first_head = kv_head * group + (head_block % blocks_per_group) * Heads;
    const auto feature = static_cast<std::uint32_t>(threadIdx.x);
    if (feature >= head_dim) return;
    const auto kv_width = kv_heads * head_dim;
    float sums[Heads]{};
    for (std::uint32_t slot = 0; slot < selected_count; ++slot) {
        const auto row_base = static_cast<std::uint64_t>(selected[slot]) * kv_width * 2ULL;
        const auto value = bf16_to_float(kv_state[row_base + kv_width +
            static_cast<std::uint64_t>(kv_head) * head_dim + feature]);
#pragma unroll
        for (std::uint32_t local = 0; local < Heads; ++local) {
            const auto head = first_head + local;
            if (head < (kv_head + 1U) * group)
                sums[local] += scores[static_cast<std::uint64_t>(head) * maximum_selected + slot] * value;
        }
    }
#pragma unroll
    for (std::uint32_t local = 0; local < Heads; ++local) {
        const auto head = first_head + local;
        if (head < (kv_head + 1U) * group) {
            const float attn = round_bf16(sums[local]);
            const float gate = round_bf16(1.0F / (1.0F + expf(-q_gate[
                static_cast<std::uint64_t>(head) * head_dim * 2ULL + head_dim + feature])));
            output[static_cast<std::uint64_t>(head) * head_dim + feature] = round_bf16(attn * gate);
        }
    }
}

__global__ void qsa_gqa_validate_diagnostic_kernel(const float* q_gate, const std::uint16_t* kv_state,
    const std::uint32_t* selected, std::uint32_t attention_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, GraphSelectionArguments dynamic) {
    __shared__ std::uint32_t admitted;
    __shared__ std::uint32_t count;
    __shared__ std::uint32_t visible;
    if (threadIdx.x == 0) {
        admitted = atomicAdd(dynamic.position.status, 0U) == 0U;
        if (admitted) {
            const auto position = *dynamic.position.position;
            if (position >= dynamic.position.context_tokens) {
                atomicOr(dynamic.position.status, qsa_gqa_reuse_diagnostic::invalid_position); admitted = 0U;
            } else {
                visible = static_cast<std::uint32_t>(position + 1ULL);
                count = min(visible / dynamic.ratio, dynamic.block_topk) * dynamic.ratio + visible % dynamic.ratio;
            }
        }
    }
    __syncthreads();
    if (!admitted) return;
    for (std::uint32_t index = threadIdx.x; index < attention_heads * head_dim * 2U; index += kThreads)
        if (!isfinite(q_gate[index])) atomicOr(dynamic.position.status, qsa_gqa_reuse_diagnostic::nonfinite_source);
    for (std::uint32_t slot = threadIdx.x; slot < count; slot += kThreads) {
        const auto token = selected[slot];
        if (token >= visible || (slot && selected[slot - 1U] >= token))
            atomicOr(dynamic.position.status, qsa_gqa_reuse_diagnostic::invalid_selection);
    }
    __syncthreads();
    if (threadIdx.x == 0) admitted = atomicAdd(dynamic.position.status, 0U) == 0U;
    __syncthreads();
    if (!admitted) return; // Never gather an unvalidated selected ID.
    const auto row_elements = kv_heads * head_dim * 2U;
    const auto elements = static_cast<std::uint64_t>(count) * row_elements;
    for (std::uint64_t index = threadIdx.x; index < elements; index += kThreads) {
        const auto token = selected[index / row_elements];
        const auto bits = kv_state[static_cast<std::uint64_t>(token) * row_elements + index % row_elements];
        if ((bits & 0x7f80U) == 0x7f80U)
            atomicOr(dynamic.position.status, qsa_gqa_reuse_diagnostic::nonfinite_source);
    }
}
__global__ void qsa_gqa_validate_output_diagnostic_kernel(const float* output, std::uint32_t count, std::uint32_t* status) {
    if (atomicAdd(status, 0U) != 0U) return;
    for (std::uint32_t index = threadIdx.x; index < count; index += kThreads)
        if (!isfinite(output[index])) atomicOr(status, qsa_gqa_reuse_diagnostic::invalid_arithmetic);
}
} // namespace

namespace qsa_gqa_reuse_diagnostic {
const void* score_kernel_symbol(Mode mode) {
    switch (mode) {
    case Mode::reference: return reinterpret_cast<const void*>(attention_score_kernel);
    case Mode::heads2: return reinterpret_cast<const void*>(qsa_gqa_score_diagnostic_kernel<2U>);
    case Mode::heads4: return reinterpret_cast<const void*>(qsa_gqa_score_diagnostic_kernel<4U>);
    }
    throw std::invalid_argument("invalid QSA GQA diagnostic mode");
}
const void* value_kernel_symbol(Mode mode) {
    switch (mode) {
    case Mode::reference: return reinterpret_cast<const void*>(attention_output_kernel);
    case Mode::heads2: return reinterpret_cast<const void*>(qsa_gqa_value_diagnostic_kernel<2U>);
    case Mode::heads4: return reinterpret_cast<const void*>(qsa_gqa_value_diagnostic_kernel<4U>);
    }
    throw std::invalid_argument("invalid QSA GQA diagnostic mode");
}
namespace {
void enqueue_impl(Shape shape, const Bindings& bindings, Mode mode, hipStream_t stream, bool scores_only) {
    const auto launch = validate(shape, bindings, mode);
    if (!stream || stream == hipStreamPerThread || stream == hipStreamLegacy)
        throw std::invalid_argument("QSA GQA diagnostic requires an explicitly owned stream");
    check_hip(hipPeekAtLastError(), "QSA GQA diagnostic stale error");
    const auto* query = reinterpret_cast<const float*>(bindings.q_gate.address);
    const auto* kv = reinterpret_cast<const std::uint16_t*>(bindings.kv_state.address);
    const auto* selected = reinterpret_cast<const std::uint32_t*>(bindings.selected.address);
    auto* scores = reinterpret_cast<float*>(bindings.scores.address);
    auto* output = reinterpret_cast<float*>(bindings.output.address);
    auto* status = reinterpret_cast<std::uint32_t*>(bindings.status.address);
    const GraphSelectionArguments dynamic{{reinterpret_cast<const std::uint64_t*>(bindings.position.address),
        shape.context_tokens, status}, compression, block_topk};
    hipLaunchKernelGGL(qsa_gqa_validate_diagnostic_kernel, dim3(1), dim3(kThreads), 0, stream,
        query, kv, selected, shape.query_heads, shape.kv_heads, shape.head_dim, dynamic);
    check_hip(hipGetLastError(), "QSA GQA diagnostic source validation");
#define STRIX_QSA_GQA_SCORE(kernel) \
    hipLaunchKernelGGL(kernel, dim3(launch.score_blocks), dim3(kThreads), 0, stream, \
        query, kv, selected, launch.maximum_selected, launch.maximum_selected, \
        shape.query_heads, shape.kv_heads, shape.head_dim, scores, dynamic)
    switch (mode) {
    case Mode::reference: STRIX_QSA_GQA_SCORE(attention_score_kernel); break;
    case Mode::heads2: STRIX_QSA_GQA_SCORE(HIP_KERNEL_NAME(qsa_gqa_score_diagnostic_kernel<2U>)); break;
    case Mode::heads4: STRIX_QSA_GQA_SCORE(HIP_KERNEL_NAME(qsa_gqa_score_diagnostic_kernel<4U>)); break;
    }
#undef STRIX_QSA_GQA_SCORE
    check_hip(hipGetLastError(), "QSA GQA diagnostic score");
    if (scores_only) return;
    hipLaunchKernelGGL(attention_softmax_kernel, dim3(shape.query_heads), dim3(kThreads), 0, stream,
        scores, launch.maximum_selected, launch.maximum_selected, dynamic);
    check_hip(hipGetLastError(), "QSA GQA diagnostic original softmax");
#define STRIX_QSA_GQA_VALUE(kernel) \
    hipLaunchKernelGGL(kernel, dim3(launch.value_blocks), dim3(shape.head_dim), 0, stream, \
        query, kv, selected, scores, launch.maximum_selected, launch.maximum_selected, \
        shape.query_heads, shape.kv_heads, shape.head_dim, output, dynamic)
    switch (mode) {
    case Mode::reference: STRIX_QSA_GQA_VALUE(attention_output_kernel); break;
    case Mode::heads2: STRIX_QSA_GQA_VALUE(HIP_KERNEL_NAME(qsa_gqa_value_diagnostic_kernel<2U>)); break;
    case Mode::heads4: STRIX_QSA_GQA_VALUE(HIP_KERNEL_NAME(qsa_gqa_value_diagnostic_kernel<4U>)); break;
    }
#undef STRIX_QSA_GQA_VALUE
    check_hip(hipGetLastError(), "QSA GQA diagnostic value/gate");
    hipLaunchKernelGGL(qsa_gqa_validate_output_diagnostic_kernel, dim3(1), dim3(kThreads), 0, stream,
        output, shape.query_heads * shape.head_dim, status);
    check_hip(hipGetLastError(), "QSA GQA diagnostic finite output");
}
} // namespace
void enqueue(Shape shape, const Bindings& bindings, Mode mode, hipStream_t stream) {
    enqueue_impl(shape, bindings, mode, stream, false);
}
void enqueue_scores(Shape shape, const Bindings& bindings, Mode mode, hipStream_t stream) {
    enqueue_impl(shape, bindings, mode, stream, true);
}
} // namespace qsa_gqa_reuse_diagnostic

// Closed single-request CLI handoff only. The caller has completed all prompt
// producers; the gate excludes submissions and its event covers the last score,
// selection and attend consumer before descriptor/pool destruction.
void release_hip_qsa_prefill_storage() {
    auto& shared=qsa_scratch_gate();
    const auto lock=shared.enter(HipQsaScratchBackend{}.device());
    if(shared.handoff.pending()) check_hip(hipEventSynchronize(shared.done),"QSA prefill retirement handoff wait");
    check_hip(hipDeviceSynchronize(),"QSA prefill retirement drain");
    shared.handoff.drained();
    release_qpf2k_stage_slots();
    std::lock_guard<std::mutex> pool_lock(qsa_prefill_pool_mutex());
    auto& pool=qsa_prefill_pool_storage();
    if(pool.keys) check_hip(hipFree(pool.keys),"QSA prefill retirement keys free");
    if(pool.packed) check_hip(hipFree(pool.packed),"QSA prefill retirement packed free");
    pool={};
}

}  // namespace strix

namespace strix {
namespace {
// Cache-only teacher rows must not publish a finite FP32 value that overflows
// its durable BF16 cast. This validator has no reduction or payload mutation.
__global__ void validate_mtp_cache_casts(const float* index,std::uint32_t index_dim,
    const float* key,const float* value,std::uint32_t kv_width,std::uint32_t* status) {
    if(atomicAdd(status,0U)!=0U)return;
    const auto count=index_dim>kv_width?index_dim:kv_width;
    for(std::uint32_t i=threadIdx.x;i<count;i+=blockDim.x) {
        bool invalid=false;
        if(i<index_dim)invalid=!isfinite(index[i]) || (float_to_bf16(index[i])&0x7f80U)==0x7f80U;
        if(i<kv_width)invalid=invalid || !isfinite(key[i]) || !isfinite(value[i]) ||
            (float_to_bf16(key[i])&0x7f80U)==0x7f80U || (float_to_bf16(value[i])&0x7f80U)==0x7f80U;
        if(invalid)atomicOr(status,8U);
    }
}
// One block validates the complete small cache row before publishing any byte.
// Shared admission keeps early exits uniform around the barriers. This
// folds only validators and casts/stores, with no arithmetic reduction.
__global__ __launch_bounds__(kThreads) void mtp_cache_publish_kernel(
    const float* index, std::uint32_t index_dim, const float* key, const float* value,
    std::uint32_t kv_width, std::uint16_t* raw_state, std::uint16_t* kv_state,
    std::uint64_t position, std::uint64_t capacity, std::uint32_t* status) {
    __shared__ bool admitted;
    if (threadIdx.x == 0U) admitted = atomicAdd(status, 0U) == 0U;
    __syncthreads();
    if (!admitted) return;
    const auto count = index_dim > kv_width ? index_dim : kv_width;
    for (std::uint32_t i = threadIdx.x; i < count; i += kThreads) {
        bool invalid = false;
        if (i < index_dim) invalid = !isfinite(index[i]) || (float_to_bf16(index[i]) & 0x7f80U) == 0x7f80U;
        if (i < kv_width) invalid = invalid || !isfinite(key[i]) || !isfinite(value[i]) ||
            (float_to_bf16(key[i]) & 0x7f80U) == 0x7f80U || (float_to_bf16(value[i]) & 0x7f80U) == 0x7f80U;
        if (invalid) atomicOr(status, 8U);
    }
    __syncthreads();
    if (threadIdx.x == 0U) admitted = atomicAdd(status, 0U) == 0U && position < capacity;
    __syncthreads();
    if (!admitted) return;
    for (std::uint32_t i = threadIdx.x; i < index_dim; i += kThreads)
        raw_state[position * index_dim + i] = float_to_bf16(index[i]);
    const auto row = position * static_cast<std::uint64_t>(kv_width) * 2ULL;
    for (std::uint32_t i = threadIdx.x; i < kv_width; i += kThreads) {
        kv_state[row + i] = float_to_bf16(key[i]);
        kv_state[row + kv_width + i] = float_to_bf16(value[i]);
    }
}
// mtp_cache_publish_kernel's body for the draft batch's rows kernel; the one-token kernel above stays as it was.
__device__ __forceinline__ void mtp_cache_publish_body(
    const float* index, std::uint32_t index_dim, const float* key, const float* value,
    std::uint32_t kv_width, std::uint16_t* raw_state, std::uint16_t* kv_state,
    std::uint64_t position, std::uint64_t capacity, std::uint32_t* status) {
    __shared__ bool admitted;
    if (threadIdx.x == 0U) admitted = atomicAdd(status, 0U) == 0U;
    __syncthreads();
    if (!admitted) return;
    const auto count = index_dim > kv_width ? index_dim : kv_width;
    for (std::uint32_t i = threadIdx.x; i < count; i += kThreads) {
        bool invalid = false;
        if (i < index_dim) invalid = !isfinite(index[i]) || (float_to_bf16(index[i]) & 0x7f80U) == 0x7f80U;
        if (i < kv_width) invalid = invalid || !isfinite(key[i]) || !isfinite(value[i]) ||
            (float_to_bf16(key[i]) & 0x7f80U) == 0x7f80U || (float_to_bf16(value[i]) & 0x7f80U) == 0x7f80U;
        if (invalid) atomicOr(status, 8U);
    }
    __syncthreads();
    if (threadIdx.x == 0U) admitted = atomicAdd(status, 0U) == 0U && position < capacity;
    __syncthreads();
    if (!admitted) return;
    for (std::uint32_t i = threadIdx.x; i < index_dim; i += kThreads)
        raw_state[position * index_dim + i] = float_to_bf16(index[i]);
    const auto row = position * static_cast<std::uint64_t>(kv_width) * 2ULL;
    for (std::uint32_t i = threadIdx.x; i < kv_width; i += kThreads) {
        kv_state[row + i] = float_to_bf16(key[i]);
        kv_state[row + kv_width + i] = float_to_bf16(value[i]);
    }
}
struct MtpAttentionPrepared {
    MtpDeviceView input,output,work,kv,index,status;
    WorkspacePointers workspace;
    IndexedTensors tensors;
    std::uint32_t hidden,heads,kv_heads,dim,attention_width,kv_width,q_width,index_width,index_dim;
    std::uint32_t visible,ratio,topk,blocks,selected,maximum_selected;
};
MtpAttentionPrepared prepare_mtp_attention(const MtpNativeContext& c) {
    mtp_native::context(c);const auto& m=*c.model;const auto& t=*c.tensors;
    check_qwen_rotary_position(m, c.position);
    HipIndexedAttentionLauncherConfig config;config.model=m;validate_config(config);
    const auto hidden=checked_u32(m.hidden_size,"MTP attention hidden");
    const auto heads=checked_u32(m.attention_heads,"MTP attention heads");
    const auto kv_heads=checked_u32(m.kv_heads,"MTP attention KV heads");
    const auto dim=checked_u32(m.head_dim,"MTP attention dimension");
    const auto attention_width=checked_u32(mtp_native::product(heads,dim),"MTP attention width");
    const auto kv_width=checked_u32(mtp_native::product(kv_heads,dim),"MTP KV width");
    const auto q_width=checked_u32(mtp_native::product(attention_width,2U),"MTP query/gate width");
    const auto index_width=checked_u32(mtp_native::product(m.indexer_n_heads,m.indexer_head_dim),"MTP index query width");
    const auto index_dim=checked_u32(m.indexer_head_dim,"MTP index dimension");
    const auto pair=[](MtpTensorView x) {return TensorPair{x.manifest,x.runtime};};
    IndexedTensors tensors{pair(t.query),pair(t.key),pair(t.value),pair(t.attention_output),pair(t.query_norm),pair(t.key_norm),
        pair(t.index_query),pair(t.index_key),pair(t.index_query_norm),pair(t.index_key_norm)};
    for(const auto* tensor:{&tensors.q,&tensors.k,&tensors.v,&tensors.out})
        if(!tensor->manifest || !tensor->runtime || tensor->manifest->ggml_type!=8U)
            throw std::invalid_argument("MTP attention profile requires original Q8 projections");
    for(const auto* tensor:{&tensors.index_q,&tensors.index_k})
        if(!tensor->manifest || !tensor->runtime || tensor->manifest->ggml_type!=30U)
            throw std::invalid_argument("MTP attention profile requires original BF16 index projections");
    for(const auto* tensor:{&tensors.q_norm,&tensors.k_norm,&tensors.index_q_norm,&tensors.index_k_norm})
        if(!tensor->manifest || !tensor->runtime || tensor->manifest->ggml_type!=0U)
            throw std::invalid_argument("MTP attention profile requires effective F32 norm gains");
    validate_matrix(tensors.q,hidden,q_width,"MTP query");validate_matrix(tensors.k,hidden,kv_width,"MTP key");
    validate_matrix(tensors.v,hidden,kv_width,"MTP value");validate_matrix(tensors.out,attention_width,hidden,"MTP output");
    validate_matrix(tensors.index_q,hidden,index_width,"MTP index query");validate_matrix(tensors.index_k,hidden,index_dim,"MTP index key");
    validate_norm(tensors.q_norm,dim,"MTP query norm");validate_norm(tensors.k_norm,dim,"MTP key norm");
    validate_norm(tensors.index_q_norm,index_dim,"MTP index query norm");validate_norm(tensors.index_k_norm,index_dim,"MTP index key norm");
    const auto& layout=c.resources->attention_workspace();
    const auto input=mtp_native::buffer(c,HipMtpBuffer::hidden0,mtp_native::product(hidden,2U));
    const auto output=mtp_native::buffer(c,HipMtpBuffer::hidden1,mtp_native::product(hidden,2U));
    const auto work=mtp_native::buffer(c,HipMtpBuffer::attention_workspace,layout.total_bytes,256U);
    const auto kv=mtp_native::buffer(c,HipMtpBuffer::kv,mtp_native::product(mtp_native::product(c.resources->context_capacity(),kv_width),4U));
    const auto index=mtp_native::buffer(c,HipMtpBuffer::raw_index,mtp_native::product(mtp_native::product(c.resources->context_capacity(),index_dim),2U));
    const auto status=mtp_native::buffer(c,HipMtpBuffer::status,8U);
    mtp_native::disjoint({input,output,work,kv,index,status});
    for(const auto* tensor:{&tensors.q,&tensors.k,&tensors.v,&tensors.out,&tensors.q_norm,&tensors.k_norm,
        &tensors.index_q,&tensors.index_k,&tensors.index_q_norm,&tensors.index_k_norm}) {
        const MtpDeviceView weights{tensor->runtime->address,tensor->runtime->bytes,tensor->runtime->allocation_generation};
        const auto alignment=tensor->manifest->ggml_type==0U?4U:2U;
        if(!weights.address || weights.address%alignment || !weights.allocation_generation || weights.bytes>UINTPTR_MAX-weights.address)
            throw std::invalid_argument("MTP attention weight range is invalid");
        mtp_native::disjoint({work,output,kv,index,status,weights});
    }
    const auto visible=checked_u32(c.position+1U,"MTP visible source rows");
    const auto ratio=checked_u32(m.indexer_compress_ratio,"MTP compression ratio");
    const auto topk=checked_u32(m.indexer_budget/m.indexer_compress_ratio,"MTP selected blocks");
    const auto blocks=visible/ratio;
    const auto selected=std::min(blocks,topk)*ratio+visible%ratio;
    if(!selected || selected>layout.maximum_selected_tokens)
        throw std::invalid_argument("MTP attention selection exceeds retained workspace");
    (void)prepare_indexed_attention_score_grid(heads,selected);
    const ExecutionBufferAddress temporary{{},work.address,work.bytes,work.allocation_generation};
    return {input,output,work,kv,index,status,workspace_pointers(temporary,layout),tensors,hidden,heads,kv_heads,dim,
        attention_width,kv_width,q_width,index_width,index_dim,visible,ratio,topk,blocks,selected,
        checked_u32(layout.maximum_selected_tokens,"MTP maximum selected rows")};
}
void require_mtp_attention_stream(const MtpNativeContext& c,void* stream) {
    mtp_native::stream(c,stream);
    if(current_hip_launch_context()) throw std::invalid_argument("MTP attention cannot nest an active native scope");
    hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
    check_hip(hipStreamIsCapturing(static_cast<hipStream_t>(stream),&capture),"MTP attention capture query");
    if(capture!=hipStreamCaptureStatusNone) throw std::invalid_argument("MTP attention supports eager execution only");
}
template<WeightKind Kind> void mtp_project(const MtpDeviceView& input,const TensorPair& tensor,
    std::uint32_t width,std::uint32_t outputs,float* output,hipStream_t stream) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(linear_to_float_kernel<Kind>),dim3(outputs),dim3(kThreads),0,stream,
        reinterpret_cast<const std::uint16_t*>(input.address),reinterpret_cast<const void*>(tensor.runtime->address),width,outputs,output);
    check_hip(hipGetLastError(),"MTP attention projection enqueue");
}
void mtp_attention_finite(const MtpAttentionPrepared& p,const float* values,std::uint32_t count,hipStream_t stream,
    bool any_order=false) {
    if(any_order) hipExtLaunchKernelGGL(validate_indexed_graph_values,dim3(1),dim3(kThreads),0,stream,nullptr,nullptr,
        hipExtAnyOrderLaunch,values,count,reinterpret_cast<std::uint32_t*>(p.status.address));
    else hipLaunchKernelGGL(validate_indexed_graph_values,dim3(1),dim3(kThreads),0,stream,values,count,
        reinterpret_cast<std::uint32_t*>(p.status.address));
    check_hip(hipGetLastError(),"MTP attention finite validation enqueue");
}

std::unique_ptr<HipQsaCommittedAccess> mtp_pool_access(const MtpNativeContext& c,const MtpAttentionPrepared& p,
    hipStream_t stream,bool committed) {
    const auto& owner=c.resources->qsa_pool();
    if (!owner) return {};
    return owner->before_rewrite(true,p.index.address,p.index.bytes,p.index.allocation_generation,
        *p.tensors.index_k_norm.runtime,p.tensors.index_k_norm.manifest->ggml_type,*c.model,c.position,1,stream,committed);
}
void mtp_pool_scores(const MtpNativeContext& c,const MtpAttentionPrepared& p,HipQsaCommittedAccess& access,hipStream_t stream) {
    auto& pool=*static_cast<QsaPrefillPool*>(access.storage());const auto& m=*c.model;
    bind_qwen_pool(&pool,m);
    if (p.blocks>pool.capacity) throw std::invalid_argument("MTP predictor pool score exceeds capacity");
    if (p.blocks>pool.valid) {
        with_qwen_rotary(m,stream,[&](auto... rotary_source) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(block_pool_kernel<SmallWeightKind::f32,false,true,decltype(rotary_source)...>),
                dim3(p.blocks-pool.valid),dim3(p.index_dim),2ULL*p.index_dim*sizeof(float),stream,
                reinterpret_cast<const std::uint16_t*>(p.index.address),pool.valid,p.index_dim,p.ratio,
                reinterpret_cast<const void*>(p.tensors.index_k_norm.runtime->address),m.rms_norm_epsilon,
                checked_u32(m.rotary_dim(),"MTP pool rotary"),m.rope_theta,nullptr,nullptr,0U,pool.packed,rotary_source...);
        });
        check_hip(hipGetLastError(),"MTP predictor packed pool build");pool.valid=p.blocks;
    }
    QsaRowsScoreViews<1> views{};views.index_query[0]=p.workspace.index_query;
    views.block_scores[0]=p.workspace.block_scores;views.blocks[0]=p.blocks;
    const auto waves=(p.blocks+31U)/32U;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(block_score_pooled_rows_kernel<1,1,std::uint16_t>),dim3((waves+7U)/8U),dim3(kThreads),0,stream,
        views,static_cast<const std::uint16_t*>(pool.packed),p.blocks,p.index_dim);
    check_hip(hipGetLastError(),"MTP predictor packed pool score");
    const char* check=std::getenv("STRIX_QSA_MTP_POOL_CHECK");
    if (check && check[0]=='1') {
        std::vector<float> pooled(p.blocks),reference(p.blocks);
        check_hip(hipMemcpyAsync(pooled.data(),p.workspace.block_scores,p.blocks*sizeof(float),hipMemcpyDeviceToHost,stream),"MTP pool check copy");
        check_hip(hipStreamSynchronize(stream),"MTP pool check copy completion");
        with_qwen_rotary(m,stream,[&](auto... rotary_source) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(block_score_kernel<SmallWeightKind::f32,decltype(rotary_source)...>),dim3(p.blocks),dim3(p.index_dim),
                2ULL*p.index_dim*sizeof(float),stream,p.workspace.index_query,reinterpret_cast<const std::uint16_t*>(p.index.address),
                checked_u32(m.indexer_n_heads,"MTP pool heads"),p.index_dim,p.ratio,reinterpret_cast<const void*>(p.tensors.index_k_norm.runtime->address),
                m.rms_norm_epsilon,checked_u32(m.rotary_dim(),"MTP pool rotary"),m.rope_theta,p.workspace.block_scores,GraphSelectionArguments{},rotary_source...);
        });
        check_hip(hipMemcpyAsync(reference.data(),p.workspace.block_scores,p.blocks*sizeof(float),hipMemcpyDeviceToHost,stream),"MTP pool reference copy");
        check_hip(hipStreamSynchronize(stream),"MTP pool reference completion");
        std::uint64_t diff=0;
        for (std::size_t i=0;i<pooled.size();++i) diff+=std::memcmp(&pooled[i],&reference[i],sizeof(float))!=0;
        check_hip(hipMemcpyAsync(p.workspace.block_scores,pooled.data(),p.blocks*sizeof(float),hipMemcpyHostToDevice,stream),"MTP pool restore pooled scores");
        check_hip(hipStreamSynchronize(stream),"MTP pool restore completion");
        c.resources->qsa_pool()->checked(p.blocks,diff,12U);
    }
}
}
void preflight_hip_mtp_attention(const MtpNativeContext& c) {(void)prepare_mtp_attention(c);}
void preflight_hip_mtp_cache_rows(const MtpNativeContext& c) {(void)prepare_mtp_attention(c);}
void launch_hip_mtp_cache_rows(const MtpNativeContext& c,void* raw_stream,bool fast_cache) {
    const auto p=prepare_mtp_attention(c);require_mtp_attention_stream(c,raw_stream);
    const auto stream=static_cast<hipStream_t>(raw_stream);const auto& m=*c.model;const auto& w=p.workspace;
    auto pool_access=mtp_pool_access(c,p,stream,true);
    // The original indexer stores the unnormalised BF16-rounded key. Pooled
    // index-key norm/RoPE is applied later by block_score_kernel, never here.
    mtp_project<WeightKind::bf16>(p.input,p.tensors.index_k,p.hidden,p.index_dim,w.current_index_key,stream);
    if(fast_cache) {
        launch_hip_mtp_draft_q8_gemv(reinterpret_cast<const void*>(p.input.address),true,
            reinterpret_cast<const void*>(p.tensors.k.runtime->address),p.hidden,p.kv_width,w.current_k,false,stream);
        launch_hip_mtp_draft_q8_gemv(reinterpret_cast<const void*>(p.input.address),true,
            reinterpret_cast<const void*>(p.tensors.v.runtime->address),p.hidden,p.kv_width,w.current_v,false,stream);
    } else {
        mtp_project<WeightKind::q8_0>(p.input,p.tensors.k,p.hidden,p.kv_width,w.current_k,stream);
        mtp_project<WeightKind::q8_0>(p.input,p.tensors.v,p.hidden,p.kv_width,w.current_v,stream);
    }
    launch_norm_rope_typed<SmallWeightKind::f32>(w.current_k,p.kv_heads,p.dim,p.dim,*p.tensors.k_norm.runtime,m,c.position,{},stream);
    auto* status=reinterpret_cast<std::uint32_t*>(p.status.address);
    if (qpersist_mtp_cache_enabled()) {
        hipLaunchKernelGGL(mtp_cache_publish_kernel, dim3(1), dim3(kThreads), 0, stream,
            w.current_index_key, p.index_dim, w.current_k, w.current_v, p.kv_width,
            reinterpret_cast<std::uint16_t*>(p.index.address), reinterpret_cast<std::uint16_t*>(p.kv.address),
            c.position, c.resources->context_capacity(), status);
        check_hip(hipGetLastError(), "MTP fused teacher cache publish enqueue");
        return;
    }
    hipLaunchKernelGGL(validate_mtp_cache_casts,dim3(1),dim3(kThreads),0,stream,
        w.current_index_key,p.index_dim,w.current_k,w.current_v,p.kv_width,status);
    check_hip(hipGetLastError(),"MTP teacher cache-cast validation enqueue");
    const GraphPositionKernelArguments guarded{nullptr,c.resources->context_capacity(),status};
    hipLaunchKernelGGL(store_raw_index_key_kernel,dim3((p.index_dim+kThreads-1U)/kThreads),dim3(kThreads),0,stream,
        w.current_index_key,reinterpret_cast<std::uint16_t*>(p.index.address),p.index_dim,c.position,guarded);
    check_hip(hipGetLastError(),"MTP teacher raw index store enqueue");
    hipLaunchKernelGGL(store_kv_kernel,dim3((p.kv_width+kThreads-1U)/kThreads),dim3(kThreads),0,stream,w.current_k,w.current_v,
        reinterpret_cast<std::uint16_t*>(p.kv.address),p.kv_width,c.position,guarded);
    check_hip(hipGetLastError(),"MTP teacher KV store enqueue");
}
void launch_hip_mtp_indexer(const MtpNativeContext& c,void* raw_stream) {
    const auto p=prepare_mtp_attention(c);require_mtp_attention_stream(c,raw_stream);
    const auto stream=static_cast<hipStream_t>(raw_stream);const auto& m=*c.model;const auto& w=p.workspace;
    auto pool_access=mtp_pool_access(c,p,stream,false);
    // STRIX_QSA_SKIP_INDEX_Q=1: with every visible token selected (blocks <= topk) the index query,
    // its norm and its finite check feed nothing; the index key is still projected and stored.
    const bool skip_index_q=qsa_skip_index_q_enabled() && p.blocks<=p.topk;
    if(!skip_index_q) mtp_project<WeightKind::bf16>(p.input,p.tensors.index_q,p.hidden,p.index_width,w.index_query,stream);
    mtp_project<WeightKind::bf16>(p.input,p.tensors.index_k,p.hidden,p.index_dim,w.current_index_key,stream);
    if(!skip_index_q) {
        launch_norm_rope_typed<SmallWeightKind::f32>(w.index_query,checked_u32(m.indexer_n_heads,"MTP index heads"),p.index_dim,p.index_dim,
            *p.tensors.index_q_norm.runtime,m,c.position,{},stream);
        mtp_attention_finite(p,w.index_query,p.index_width,stream);
    }
    auto* cache_status=reinterpret_cast<std::uint32_t*>(p.status.address);
    if (qpersist_mtp_cache_enabled()) {
        hipLaunchKernelGGL(mtp_cache_publish_kernel, dim3(1), dim3(kThreads), 0, stream,
            w.current_index_key, p.index_dim, nullptr, nullptr, 0U,
            reinterpret_cast<std::uint16_t*>(p.index.address), nullptr,
            c.position, c.resources->context_capacity(), cache_status);
        check_hip(hipGetLastError(), "MTP fused raw index publish enqueue");
    } else {
        mtp_attention_finite(p,w.current_index_key,p.index_dim,stream);
        hipLaunchKernelGGL(validate_mtp_cache_casts,dim3(1),dim3(kThreads),0,stream,
            w.current_index_key,p.index_dim,static_cast<const float*>(nullptr),static_cast<const float*>(nullptr),0U,cache_status);
        check_hip(hipGetLastError(),"MTP full raw-index cache-cast validation enqueue");
        const GraphPositionKernelArguments guarded_cache{nullptr,c.resources->context_capacity(),cache_status};
        hipLaunchKernelGGL(store_raw_index_key_kernel,dim3((p.index_dim+kThreads-1U)/kThreads),dim3(kThreads),0,stream,
            w.current_index_key,reinterpret_cast<std::uint16_t*>(p.index.address),p.index_dim,c.position,guarded_cache);
        check_hip(hipGetLastError(),"MTP raw index store enqueue");
    }
    const GraphSelectionArguments selection{{},p.ratio,p.topk};
    if(p.blocks<=p.topk) {
        hipLaunchKernelGGL(fill_visible_indices_kernel,dim3((p.visible+kThreads-1U)/kThreads),dim3(kThreads),0,stream,w.selected_indices,p.visible);
        check_hip(hipGetLastError(),"MTP full-visible index enqueue");
    } else {
        if (pool_access) mtp_pool_scores(c,p,*pool_access,stream);
        else with_qwen_rotary(m, stream, [&](auto... rotary_source) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(block_score_kernel<SmallWeightKind::f32, decltype(rotary_source)...>),dim3(p.blocks),dim3(p.index_dim),
                2ULL*p.index_dim*sizeof(float),stream,w.index_query,reinterpret_cast<const std::uint16_t*>(p.index.address),
                checked_u32(m.indexer_n_heads,"MTP index heads"),p.index_dim,p.ratio,reinterpret_cast<const void*>(p.tensors.index_k_norm.runtime->address),
                m.rms_norm_epsilon,checked_u32(m.rotary_dim(),"MTP rotary dimension"),m.rope_theta,w.block_scores,selection, rotary_source...);
        });
        check_hip(hipGetLastError(),"MTP compressed index score enqueue");
        mtp_attention_finite(p,w.block_scores,p.blocks,stream);
        MtpPoolSelectionCheck selection_check(c.resources->qsa_pool(),{w.index_query,w.block_scores,w.selected_indices,
            p.visible,p.blocks,p.topk,0U,nullptr},stream);
        hipLaunchKernelGGL(select_blocks_kernel,dim3(1),dim3(kSelectThreads),0,stream,w.block_scores,p.blocks,p.topk,p.ratio,p.visible,w.selected_indices,selection);
        check_hip(hipGetLastError(),"MTP compressed selection enqueue");
        selection_check.finish();
    }
}
void enqueue_hip_mtp_attention_q8_linear(const std::uint16_t* input,const void* q8,std::uint32_t width,
    std::uint32_t rows,float* output,void* stream) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(linear_to_float_kernel<WeightKind::q8_0>),dim3(rows),dim3(kThreads),0,static_cast<hipStream_t>(stream),
        input,q8,width,rows,output);
    check_hip(hipGetLastError(),"MTP attention projection enqueue");
}
void enqueue_hip_mtp_attention_bf16_linear(const std::uint16_t* input,const void* bf16,std::uint32_t width,
    std::uint32_t rows,float* output,void* stream) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(linear_to_float_kernel<WeightKind::bf16>),dim3(rows),dim3(kThreads),0,static_cast<hipStream_t>(stream),
        input,bf16,width,rows,output);
    check_hip(hipGetLastError(),"MTP attention BF16 projection enqueue");
}
namespace {
void mtp_sparse_attention(const MtpNativeContext& c,void* raw_stream,bool draft) {
    const auto p=prepare_mtp_attention(c);require_mtp_attention_stream(c,raw_stream);
    const auto stream=static_cast<hipStream_t>(raw_stream);const auto& m=*c.model;const auto& w=p.workspace;
    // Draft any-order: k and v read the same input as q and write their own buffers; the q
    // finite check is barrier-ordered after both norms, the k/v checks only OR the status word.
    const bool unordered=draft && strix_any_order_enabled();
    const auto project=[&](const TensorPair& tensor,std::uint32_t outputs,float* output,bool after_q) {
        if(draft) launch_hip_mtp_draft_q8_gemv(reinterpret_cast<const void*>(p.input.address),true,
            reinterpret_cast<const void*>(tensor.runtime->address),p.hidden,outputs,output,false,stream,unordered && after_q);
        else mtp_project<WeightKind::q8_0>(p.input,tensor,p.hidden,outputs,output,stream);
    };
    project(p.tensors.q,p.q_width,w.q_gate,false);project(p.tensors.k,p.kv_width,w.current_k,true);
    project(p.tensors.v,p.kv_width,w.current_v,true);
    launch_norm_rope_typed<SmallWeightKind::f32>(w.q_gate,p.heads,p.dim,p.dim*2U,*p.tensors.q_norm.runtime,m,c.position,{},stream);
    launch_norm_rope_typed<SmallWeightKind::f32>(w.current_k,p.kv_heads,p.dim,p.dim,*p.tensors.k_norm.runtime,m,c.position,{},stream);
    mtp_attention_finite(p,w.q_gate,p.q_width,stream);
    auto* cache_status=reinterpret_cast<std::uint32_t*>(p.status.address);
    if (qpersist_mtp_cache_enabled()) {
        hipLaunchKernelGGL(mtp_cache_publish_kernel, dim3(1), dim3(kThreads), 0, stream,
            nullptr, 0U, w.current_k, w.current_v, p.kv_width, nullptr,
            reinterpret_cast<std::uint16_t*>(p.kv.address), c.position, c.resources->context_capacity(), cache_status);
        check_hip(hipGetLastError(), "MTP fused KV publish enqueue");
    } else {
        mtp_attention_finite(p,w.current_k,p.kv_width,stream,unordered);
        mtp_attention_finite(p,w.current_v,p.kv_width,stream,unordered);
        if(unordered) hipExtLaunchKernelGGL(validate_mtp_cache_casts,dim3(1),dim3(kThreads),0,stream,nullptr,nullptr,hipExtAnyOrderLaunch,
            static_cast<const float*>(nullptr),0U,w.current_k,w.current_v,p.kv_width,cache_status);
        else hipLaunchKernelGGL(validate_mtp_cache_casts,dim3(1),dim3(kThreads),0,stream,
            static_cast<const float*>(nullptr),0U,w.current_k,w.current_v,p.kv_width,cache_status);
        check_hip(hipGetLastError(),"MTP full KV cache-cast validation enqueue");
        const GraphPositionKernelArguments guarded_cache{nullptr,c.resources->context_capacity(),cache_status};
        hipLaunchKernelGGL(store_kv_kernel,dim3((p.kv_width+kThreads-1U)/kThreads),dim3(kThreads),0,stream,w.current_k,w.current_v,
            reinterpret_cast<std::uint16_t*>(p.kv.address),p.kv_width,c.position,guarded_cache);
        check_hip(hipGetLastError(),"MTP KV store enqueue");
    }
    const GraphSelectionArguments selection{{},p.ratio,p.topk};
    // Q7 arm E (STRIX_QPERSIST_DRAFT_ATTN default on, =0 incumbent; on top of the D-059 score + tail
    // flags): draft only; draft_exact / teacher forward (draft == false) keep the incumbent kernels below.
    bool exact = false;
    static const auto choice = resolve_draft_attention(qpersist_draft_attention_setting(),
        std::getenv("STRIX_QPERSIST_DRAFT_ATTN_SCORE"), std::getenv("STRIX_QPERSIST_DRAFT_ATTN_TAIL"));
    if (draft && choice.on && qpersist_draft_score_enabled() && qpersist_qsa_tail_enabled()) {
        const auto score_variant = choice.score;
        const auto tail_variant = choice.tail;
        static std::atomic<std::uint64_t> engaged{0}, fallback{0};
        if (draft_attention_admitted(score_variant, tail_variant, p.heads, p.kv_heads, p.dim, p.selected)) {
            if (engaged.fetch_add(1U) == 0U)
                std::fprintf(stderr, "qpersist draft attention E engaged: score %s tail %s (selected %u)%s\n",
                    kDraftScoreNames[static_cast<std::size_t>(score_variant)].data(),
                    kDraftTailNames[static_cast<std::size_t>(tail_variant)].data(), p.selected,
                    choice.defaulted ? " (default)" : " (explicit)");
            QsaRowsAttentionViews<1U> attention{};
            attention.q_gate[0] = w.q_gate; attention.selected[0] = w.selected_indices;
            attention.scores[0] = w.attention_scores; attention.count[0] = p.selected;
            launch_draft_attention_scores(score_variant, attention, reinterpret_cast<const std::uint16_t*>(p.kv.address),
                p.maximum_selected, p.heads, p.kv_heads, p.dim, p.selected, stream);
            launch_draft_attention_tail(tail_variant, w.q_gate, reinterpret_cast<const std::uint16_t*>(p.kv.address),
                w.selected_indices, w.attention_scores, p.selected, p.maximum_selected, p.heads, p.kv_heads, p.dim,
                w.attention_output, selection, stream);
            exact = true;
        } else if (fallback.fetch_add(1U) == 0U)
            std::fprintf(stderr, "qpersist draft attention E not admitted (heads %u kv_heads %u dim %u selected %u): incumbent\n",
                p.heads, p.kv_heads, p.dim, p.selected);
    }
    if (!exact) { // incumbent score + tail
    // STRIX_QSA_WINDOW_ATTN_V2 (default on), draft only, in place of attention_score_kernel: the four query heads of one
    // KV head share each K row, in attention_score_kernel's operand order; the tail below is unchanged.
    const bool draft_v2 = draft_attn_v2_route(qsa_window_attn_v2_enabled(), draft, qpersist_draft_score_enabled(), p.heads,
        p.kv_heads, p.dim);
    if (!draft_v2 && qpersist_draft_score_enabled() && p.dim <= kThreads) {
        // The draft formerly used one 256-thread block for every head/slot.
        // Use the incumbent window tree: eight waves/block, four slots/wave,
        // with query features held across slots. No split or reassociated dot.
        QsaRowsAttentionViews<1U> attention{};
        attention.q_gate[0] = w.q_gate; attention.selected[0] = w.selected_indices;
        attention.scores[0] = w.attention_scores; attention.count[0] = p.selected;
        const auto groups = (p.selected + 3U) / 4U;
        const auto waves = static_cast<std::uint64_t>(p.heads) * groups;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_score_rows_wave_kernel<1U, 4U>),
            dim3(checked_u32((waves + 7U) / 8U, "MTP tiled score grid")), dim3(kThreads), 0, stream,
            attention, reinterpret_cast<const std::uint16_t*>(p.kv.address), p.maximum_selected,
            p.heads, p.kv_heads, p.dim, p.selected);
    } else launch_draft_one_block_scores(draft_v2, w.q_gate, reinterpret_cast<const std::uint16_t*>(p.kv.address),
        w.selected_indices, w.attention_scores, p.selected, p.maximum_selected, p.heads, p.kv_heads, p.dim, selection, stream);
    check_hip(hipGetLastError(),"MTP attention scores enqueue");
    if (qpersist_qsa_tail_enabled() && p.dim <= kThreads) {
        hipLaunchKernelGGL(attention_tail_kernel, dim3(p.heads), dim3(kThreads), 0, stream,
            w.q_gate, reinterpret_cast<const std::uint16_t*>(p.kv.address), w.selected_indices,
            w.attention_scores, p.selected, p.maximum_selected, p.heads, p.kv_heads, p.dim,
            w.attention_output, selection);
        check_hip(hipGetLastError(), "MTP fused attention tail enqueue");
    } else {
        hipLaunchKernelGGL(attention_softmax_kernel,dim3(p.heads),dim3(kThreads),0,stream,w.attention_scores,p.selected,p.maximum_selected,selection);
        check_hip(hipGetLastError(),"MTP attention softmax enqueue");
        hipLaunchKernelGGL(attention_output_kernel,dim3(p.heads),dim3(p.dim),0,stream,w.q_gate,reinterpret_cast<const std::uint16_t*>(p.kv.address),
            w.selected_indices,w.attention_scores,p.selected,p.maximum_selected,p.heads,p.kv_heads,p.dim,w.attention_output,selection);
        check_hip(hipGetLastError(),"MTP attention value enqueue");
    }
    } // incumbent score + tail
    mtp_attention_finite(p,w.attention_output,p.attention_width,stream);
    if(draft) {
        launch_hip_mtp_draft_q8_gemv(w.attention_output,false,reinterpret_cast<const void*>(p.tensors.out.runtime->address),
            p.attention_width,p.hidden,reinterpret_cast<void*>(p.output.address),true,stream);
        return;
    }
    hipLaunchKernelGGL(HIP_KERNEL_NAME(linear_float_to_bf16_kernel<WeightKind::q8_0>),dim3(p.hidden),dim3(kThreads),0,stream,
        w.attention_output,reinterpret_cast<const void*>(p.tensors.out.runtime->address),p.attention_width,p.hidden,
        reinterpret_cast<std::uint16_t*>(p.output.address));
    check_hip(hipGetLastError(),"MTP attention output projection enqueue");
}
}
void launch_hip_mtp_sparse_attention(const MtpNativeContext& c,void* stream) {mtp_sparse_attention(c,stream,false);}
void launch_hip_mtp_sparse_attention_draft(const MtpNativeContext& c,void* stream) {mtp_sparse_attention(c,stream,true);}
namespace {
// Multi-session draft batch kernels: row r of every table is one session's own one-token launch (its own
// buffers, position, counts, caches and status word); bodies are the one-token kernels' bodies.
constexpr unsigned kMtpBatchRows = 8U;
struct MtpBatchLinearRows { const std::uint16_t* input[kMtpBatchRows]; float* output[kMtpBatchRows]; };
// linear_to_float_kernel over Count tokens: each thread loads its weight once per feature and runs every
// token's unchanged chain (same feature stride, same product/accumulate expression, same 256-leaf LDS tree
// and BF16 epilogue).
template <WeightKind Kind, unsigned Count>
__global__ void linear_to_float_rows_kernel(MtpBatchLinearRows rows, const void* weights, std::uint32_t input_width,
    std::uint32_t output_width) {
    const auto out = static_cast<std::uint32_t>(blockIdx.x);
    if (out >= output_width) return;
    __shared__ float scratch[Count][kThreads];
    float sum[Count];
#pragma unroll
    for (unsigned t = 0; t < Count; ++t) sum[t] = 0.0F;
    for (std::uint32_t feature = static_cast<std::uint32_t>(threadIdx.x); feature < input_width; feature += kThreads) {
        const float weight = load_weight<Kind>(weights, input_width, out, feature);
#pragma unroll
        for (unsigned t = 0; t < Count; ++t) sum[t] += load_activation(&rows.input[t][feature]) * weight;
    }
#pragma unroll
    for (unsigned t = 0; t < Count; ++t) scratch[t][threadIdx.x] = sum[t];
    __syncthreads();
    for (std::uint32_t stride = kThreads / 2U; stride > 0; stride >>= 1U) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (unsigned t = 0; t < Count; ++t) scratch[t][threadIdx.x] += scratch[t][threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (unsigned t = 0; t < Count; ++t) rows.output[t][out] = round_bf16(scratch[t][0]);
    }
}
struct MtpBatchNormRopeRows { float* values[kMtpBatchRows]; std::uint64_t position[kMtpBatchRows]; };
template <SmallWeightKind Kind, class... Source>
__global__ void norm_rope_rows_kernel(MtpBatchNormRopeRows rows, std::uint32_t head_count, std::uint32_t head_dim,
    std::uint32_t head_stride, const void* norm_weights, float epsilon, std::uint32_t rotary_dim, float theta, Source... rotary_source) {
    norm_rope_body<Kind, Source...>(rows.values[blockIdx.y], head_count, head_dim, head_stride, norm_weights, epsilon, rotary_dim,
        theta, rows.position[blockIdx.y], GraphPositionKernelArguments{}, rotary_source...);
}
struct MtpBatchFiniteRows { const float* values[kMtpBatchRows]; std::uint32_t* status[kMtpBatchRows]; };
__global__ void validate_indexed_graph_values_rows(MtpBatchFiniteRows rows, std::uint32_t count) {
    auto* status = rows.status[blockIdx.x];
    const auto* values = rows.values[blockIdx.x];
    if (atomicAdd(status, 0U) != 0) return;
    for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        if (!isfinite(values[i])) atomicOr(status, 8U);
    }
}
struct MtpBatchPublishRows {
    const float* index[kMtpBatchRows]; const float* key[kMtpBatchRows]; const float* value[kMtpBatchRows];
    std::uint16_t* raw_state[kMtpBatchRows]; std::uint16_t* kv_state[kMtpBatchRows];
    std::uint64_t position[kMtpBatchRows], capacity[kMtpBatchRows]; std::uint32_t* status[kMtpBatchRows];
};
__global__ __launch_bounds__(kThreads) void mtp_cache_publish_rows_kernel(MtpBatchPublishRows rows, std::uint32_t index_dim,
    std::uint32_t kv_width) {
    const auto r = blockIdx.x;
    mtp_cache_publish_body(rows.index[r], index_dim, rows.key[r], rows.value[r], kv_width, rows.raw_state[r], rows.kv_state[r],
        rows.position[r], rows.capacity[r], rows.status[r]);
}
struct MtpBatchSelectRows {
    const float* index_query[kMtpBatchRows]; const std::uint16_t* raw_keys[kMtpBatchRows]; float* block_scores[kMtpBatchRows];
    std::uint32_t* selected[kMtpBatchRows]; std::uint32_t visible[kMtpBatchRows], blocks[kMtpBatchRows];
};
__global__ void fill_visible_indices_rows_kernel(MtpBatchSelectRows rows) {
    const auto r = blockIdx.y;
    const auto index = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < rows.visible[r]) rows.selected[r][index] = index;
}
template <SmallWeightKind Kind, class... Source>
__global__ void block_score_rows_kernel(MtpBatchSelectRows rows, std::uint32_t index_heads, std::uint32_t head_dim,
    std::uint32_t compress_ratio, const void* key_norm, float epsilon, std::uint32_t rotary_dim, float theta,
    GraphSelectionArguments dynamic, Source... rotary_source) {
    const auto r = blockIdx.y;
    if (blockIdx.x >= rows.blocks[r]) return;
    block_score_body<Kind, Source...>(rows.index_query[r], rows.raw_keys[r], index_heads, head_dim, compress_ratio, key_norm, epsilon,
        rotary_dim, theta, rows.block_scores[r], dynamic, rotary_source...);
}
__global__ __launch_bounds__(kSelectThreads) void select_blocks_rows_kernel(MtpBatchSelectRows rows, std::uint32_t select_blocks,
    std::uint32_t compress_ratio, GraphSelectionArguments dynamic) {
    const auto r = blockIdx.x;
    select_blocks_parallel_body(rows.block_scores[r], rows.blocks[r], select_blocks, compress_ratio, rows.visible[r],
        rows.selected[r], dynamic);
}
struct MtpBatchAttentionRows {
    const float* q_gate[kMtpBatchRows]; const std::uint32_t* selected[kMtpBatchRows]; float* scores[kMtpBatchRows];
    float* output[kMtpBatchRows]; const std::uint16_t* kv[kMtpBatchRows];
    std::uint32_t count[kMtpBatchRows], maximum_selected[kMtpBatchRows];
};
// attention_score_rows_wave_kernel with each row's own KV cache and score stride (the body is that kernel's:
// same products, pairings and order, contraction off).
template <unsigned PerWave>
__global__ __launch_bounds__(256) void attention_score_sessions_wave_kernel(MtpBatchAttentionRows views, std::uint32_t rows,
    std::uint32_t attention_heads, std::uint32_t kv_heads, std::uint32_t head_dim, std::uint32_t slots) {
#pragma clang fp contract(off)
    static_assert(kThreads == 256U);
    const auto lane = threadIdx.x % 32U;
    const auto wave = blockIdx.x * 8U + threadIdx.x / 32U;
    const auto groups = (slots + PerWave - 1U) / PerWave;
    const auto r = wave / (attention_heads * groups);
    if (r >= rows) return;
    const auto head = wave / groups % attention_heads, first = wave % groups * PerWave;
    if (first >= views.count[r]) return;
    const auto kv_head = head / (attention_heads / kv_heads);
    const auto* kv_state = views.kv[r];
    const auto* q = views.q_gate[r] + static_cast<std::uint64_t>(head) * head_dim * 2ULL;
    float qv[8];
#pragma unroll
    for (unsigned j = 0; j < 8U; ++j) { const auto feature = lane + 32U * j; qv[j] = feature < head_dim ? q[feature] : 0.0F; }
    float p[PerWave][8];
#pragma unroll
    for (unsigned i = 0; i < PerWave; ++i) {
        const auto slot = first + i < views.count[r] ? first + i : first;
        const auto token = views.selected[r][slot];
        const auto* k = kv_state + static_cast<std::uint64_t>(token) * kv_heads * head_dim * 2ULL +
            static_cast<std::uint64_t>(kv_head) * head_dim;
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) {
            const auto feature = lane + 32U * j;
            float product = 0.0F;
            if (feature < head_dim) product = qv[j] * bf16_to_float(k[feature]);
            p[i][j] = product;
        }
    }
#pragma unroll
    for (unsigned i = 0; i < PerWave; ++i) {
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) p[i][j] += p[i][j + 4U];
        p[i][0] += p[i][2]; p[i][1] += p[i][3];
        float sum = p[i][0] + p[i][1];
#pragma unroll
        for (unsigned stride = 16U; stride > 0U; stride >>= 1U) sum += __shfl_down(sum, stride, 32);
        if (lane == 0U && first + i < views.count[r])
            views.scores[r][static_cast<std::uint64_t>(head) * views.maximum_selected[r] + first + i] = sum / sqrtf(static_cast<float>(head_dim));
    }
}
__global__ void attention_tail_sessions_kernel(MtpBatchAttentionRows views, std::uint32_t attention_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, GraphSelectionArguments dynamic) {
    const auto r = blockIdx.y;
    attention_tail_body(views.q_gate[r], views.kv[r], views.selected[r], views.scores[r], views.count[r],
        views.maximum_selected[r], attention_heads, kv_heads, head_dim, views.output[r], dynamic);
}
void mtp_batch_finite(const std::vector<MtpAttentionPrepared>& rows, const std::vector<std::uint32_t>& members,
    float* WorkspacePointers::*field, std::uint32_t count, hipStream_t stream) {
    if (members.empty()) return;
    MtpBatchFiniteRows f{};
    for (std::size_t i = 0; i < members.size(); ++i) {
        const auto& p = rows[members[i]];
        f.values[i] = p.workspace.*field; f.status[i] = reinterpret_cast<std::uint32_t*>(p.status.address);
    }
    hipLaunchKernelGGL(validate_indexed_graph_values_rows, dim3(static_cast<unsigned>(members.size())), dim3(kThreads), 0, stream, f, count);
    check_hip(hipGetLastError(), "MTP batch finite validation enqueue");
}
void enqueue_bf16_linear_rows(const MtpBatchLinearRows& l, std::size_t count, const void* weights, std::uint32_t width,
    std::uint32_t outputs, hipStream_t stream) {
    const dim3 grid(outputs);
    switch (count) {
    case 1: hipLaunchKernelGGL(HIP_KERNEL_NAME(linear_to_float_rows_kernel<WeightKind::bf16, 1U>), grid, dim3(kThreads), 0, stream, l, weights, width, outputs); break;
    case 2: hipLaunchKernelGGL(HIP_KERNEL_NAME(linear_to_float_rows_kernel<WeightKind::bf16, 2U>), grid, dim3(kThreads), 0, stream, l, weights, width, outputs); break;
    case 3: hipLaunchKernelGGL(HIP_KERNEL_NAME(linear_to_float_rows_kernel<WeightKind::bf16, 3U>), grid, dim3(kThreads), 0, stream, l, weights, width, outputs); break;
    case 4: hipLaunchKernelGGL(HIP_KERNEL_NAME(linear_to_float_rows_kernel<WeightKind::bf16, 4U>), grid, dim3(kThreads), 0, stream, l, weights, width, outputs); break;
    case 5: hipLaunchKernelGGL(HIP_KERNEL_NAME(linear_to_float_rows_kernel<WeightKind::bf16, 5U>), grid, dim3(kThreads), 0, stream, l, weights, width, outputs); break;
    case 6: hipLaunchKernelGGL(HIP_KERNEL_NAME(linear_to_float_rows_kernel<WeightKind::bf16, 6U>), grid, dim3(kThreads), 0, stream, l, weights, width, outputs); break;
    case 7: hipLaunchKernelGGL(HIP_KERNEL_NAME(linear_to_float_rows_kernel<WeightKind::bf16, 7U>), grid, dim3(kThreads), 0, stream, l, weights, width, outputs); break;
    default: hipLaunchKernelGGL(HIP_KERNEL_NAME(linear_to_float_rows_kernel<WeightKind::bf16, 8U>), grid, dim3(kThreads), 0, stream, l, weights, width, outputs); break;
    }
    check_hip(hipGetLastError(), "MTP batch BF16 projection enqueue");
}
void mtp_batch_bf16_linear(const std::vector<MtpAttentionPrepared>& rows, const std::vector<std::uint32_t>& members,
    const void* weights, std::uint32_t width, std::uint32_t outputs, float* WorkspacePointers::*field, hipStream_t stream) {
    if (members.empty()) return;
    MtpBatchLinearRows l{};
    for (std::size_t i = 0; i < members.size(); ++i) {
        const auto& p = rows[members[i]];
        l.input[i] = reinterpret_cast<const std::uint16_t*>(p.input.address); l.output[i] = p.workspace.*field;
    }
    enqueue_bf16_linear_rows(l, members.size(), weights, width, outputs, stream);
}
void mtp_batch_norm_rope(const std::vector<MtpAttentionPrepared>& rows, const std::vector<std::uint32_t>& members,
    std::span<const MtpNativeContext> contexts, float* WorkspacePointers::*field, std::uint32_t heads, std::uint32_t head_dim,
    std::uint32_t head_stride, const RuntimeTensorBinding& norm, hipStream_t stream) {
    if (members.empty()) return;
    MtpBatchNormRopeRows n{};
    for (std::size_t i = 0; i < members.size(); ++i) {
        n.values[i] = rows[members[i]].workspace.*field; n.position[i] = contexts[members[i]].position;
    }
    const auto& m = *contexts.front().model;
    with_qwen_rotary(m, stream, [&](auto... rotary_source) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(norm_rope_rows_kernel<SmallWeightKind::f32, decltype(rotary_source)...>), dim3(heads, static_cast<unsigned>(members.size())),
            dim3(head_dim), 2ULL * head_dim * sizeof(float), stream, n, heads, head_dim, head_stride,
            reinterpret_cast<const void*>(norm.address), m.rms_norm_epsilon, checked_u32(m.rotary_dim(), "rotary_dim"), m.rope_theta, rotary_source...);
    });
    check_hip(hipGetLastError(), "MTP batch norm/RoPE enqueue");
}
std::vector<MtpAttentionPrepared> mtp_batch_prepare(std::span<const MtpNativeContext> contexts) {
    if (contexts.empty() || contexts.size() > kMtpBatchRows || !hip_mtp_attention_draft_rows_supported())
        throw std::invalid_argument("MTP attention batch size or configuration is invalid");
    std::vector<MtpAttentionPrepared> rows;rows.reserve(contexts.size());
    for (const auto& c : contexts) rows.push_back(prepare_mtp_attention(c));
    const auto& a = rows.front();
    for (const auto& p : rows) {
        // One shared model and weights; everything else is per session.
        if (p.tensors.q.runtime->address != a.tensors.q.runtime->address || p.tensors.k.runtime->address != a.tensors.k.runtime->address ||
            p.tensors.v.runtime->address != a.tensors.v.runtime->address || p.tensors.out.runtime->address != a.tensors.out.runtime->address ||
            p.tensors.index_q.runtime->address != a.tensors.index_q.runtime->address ||
            p.tensors.index_k.runtime->address != a.tensors.index_k.runtime->address ||
            p.tensors.q_norm.runtime->address != a.tensors.q_norm.runtime->address ||
            p.tensors.k_norm.runtime->address != a.tensors.k_norm.runtime->address ||
            p.tensors.index_q_norm.runtime->address != a.tensors.index_q_norm.runtime->address ||
            p.tensors.index_k_norm.runtime->address != a.tensors.index_k_norm.runtime->address ||
            p.hidden != a.hidden || p.heads != a.heads || p.kv_heads != a.kv_heads || p.dim != a.dim || p.q_width != a.q_width ||
            p.kv_width != a.kv_width || p.index_width != a.index_width || p.index_dim != a.index_dim || p.ratio != a.ratio ||
            p.topk != a.topk || p.attention_width != a.attention_width || p.dim > kThreads)
            throw std::invalid_argument("MTP attention batch members do not share one geometry and weights");
    }
    // Every owner holds its own copy of the model contract: compare the scalars the batched launches take from
    // the first context (each context's owner binding is validated by prepare_mtp_attention).
    const auto& m0 = *contexts.front().model;
    for (const auto& c : contexts)
        if (!same_rotary_model(*c.model, m0) || c.model->rms_norm_epsilon != m0.rms_norm_epsilon || c.model->rope_theta != m0.rope_theta ||
            c.model->rotary_dim() != m0.rotary_dim() || c.model->indexer_n_heads != m0.indexer_n_heads)
            throw std::invalid_argument("MTP attention batch members use different model contracts");
    return rows;
}
}
void enqueue_hip_mtp_attention_bf16_linear_rows(std::span<const std::uint16_t* const> inputs, const void* bf16, std::uint32_t width,
    std::uint32_t rows, std::span<float* const> outputs, void* stream) {
    if (inputs.empty() || inputs.size() > kMtpBatchRows || outputs.size() != inputs.size() || !bf16 || !stream || !rows || !width)
        throw std::invalid_argument("MTP batch BF16 projection views are invalid");
    MtpBatchLinearRows l{};
    for (std::size_t t = 0; t < inputs.size(); ++t) {
        if (!inputs[t] || !outputs[t]) throw std::invalid_argument("MTP batch BF16 projection row view is missing");
        l.input[t] = inputs[t]; l.output[t] = outputs[t];
    }
    enqueue_bf16_linear_rows(l, inputs.size(), bf16, width, rows, static_cast<hipStream_t>(stream));
}
bool hip_mtp_attention_draft_rows_supported() {
    return qpersist_mtp_cache_enabled() && qpersist_draft_score_enabled() && qpersist_qsa_tail_enabled();
}
void launch_hip_mtp_indexer_rows(std::span<const MtpNativeContext> contexts, void* raw_stream) {
    const auto rows = mtp_batch_prepare(contexts);
    if (!raw_stream) throw std::invalid_argument("MTP indexer batch needs a stream");
    const auto stream = static_cast<hipStream_t>(raw_stream);const auto& m = *contexts.front().model;const auto& a = rows.front();
    std::vector<std::unique_ptr<HipQsaCommittedAccess>> pool_access;
    const bool any_pool=std::any_of(contexts.begin(),contexts.end(),[](const auto& c){return static_cast<bool>(c.resources->qsa_pool());});
    if(any_pool) {
        pool_access.reserve(rows.size());
        for (std::size_t r=0;r<rows.size();++r) pool_access.push_back(mtp_pool_access(contexts[r],rows[r],stream,false));
    }
    std::vector<std::uint32_t> all, need_q, fill, select;
    for (std::uint32_t r = 0; r < rows.size(); ++r) {
        const auto& p = rows[r];
        all.push_back(r);
        if (!(qsa_skip_index_q_enabled() && p.blocks <= p.topk)) need_q.push_back(r);
        (p.blocks <= p.topk ? fill : select).push_back(r);
    }
    mtp_batch_bf16_linear(rows, need_q, reinterpret_cast<const void*>(a.tensors.index_q.runtime->address), a.hidden, a.index_width,
        &WorkspacePointers::index_query, stream);
    mtp_batch_bf16_linear(rows, all, reinterpret_cast<const void*>(a.tensors.index_k.runtime->address), a.hidden, a.index_dim,
        &WorkspacePointers::current_index_key, stream);
    mtp_batch_norm_rope(rows, need_q, contexts, &WorkspacePointers::index_query, checked_u32(m.indexer_n_heads, "MTP index heads"),
        a.index_dim, a.index_dim, *a.tensors.index_q_norm.runtime, stream);
    mtp_batch_finite(rows, need_q, &WorkspacePointers::index_query, a.index_width, stream);
    MtpBatchPublishRows publish{};
    for (std::uint32_t r = 0; r < rows.size(); ++r) {
        const auto& p = rows[r];
        publish.index[r] = p.workspace.current_index_key; publish.raw_state[r] = reinterpret_cast<std::uint16_t*>(p.index.address);
        publish.position[r] = contexts[r].position; publish.capacity[r] = contexts[r].resources->context_capacity();
        publish.status[r] = reinterpret_cast<std::uint32_t*>(p.status.address);
    }
    hipLaunchKernelGGL(mtp_cache_publish_rows_kernel, dim3(static_cast<unsigned>(rows.size())), dim3(kThreads), 0, stream, publish,
        a.index_dim, 0U);
    check_hip(hipGetLastError(), "MTP batch raw index publish enqueue");
    const GraphSelectionArguments selection{{}, a.ratio, a.topk};
    if (!fill.empty()) {
        MtpBatchSelectRows f{};std::uint32_t visible = 0;
        for (std::size_t i = 0; i < fill.size(); ++i) {
            const auto& p = rows[fill[i]];
            f.selected[i] = p.workspace.selected_indices; f.visible[i] = p.visible; visible = std::max(visible, p.visible);
        }
        hipLaunchKernelGGL(fill_visible_indices_rows_kernel, dim3((visible + kThreads - 1U) / kThreads, static_cast<unsigned>(fill.size())),
            dim3(kThreads), 0, stream, f);
        check_hip(hipGetLastError(), "MTP batch full-visible index enqueue");
    }
    if (!select.empty()) {
        MtpBatchSelectRows b{};std::uint32_t blocks = 0;
        for (std::size_t i = 0; i < select.size(); ++i) {
            const auto& p = rows[select[i]];
            b.index_query[i] = p.workspace.index_query; b.raw_keys[i] = reinterpret_cast<const std::uint16_t*>(p.index.address);
            b.block_scores[i] = p.workspace.block_scores; b.selected[i] = p.workspace.selected_indices;
            b.visible[i] = p.visible; b.blocks[i] = p.blocks; blocks = std::max(blocks, p.blocks);
        }
        if (any_pool) {
            for (const auto r:select) {
                const auto& p=rows[r];
                if (pool_access[r]) mtp_pool_scores(contexts[r],p,*pool_access[r],stream);
                else with_qwen_rotary(m,stream,[&](auto... rotary_source) {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(block_score_kernel<SmallWeightKind::f32,decltype(rotary_source)...>),dim3(p.blocks),dim3(p.index_dim),
                        2ULL*p.index_dim*sizeof(float),stream,p.workspace.index_query,reinterpret_cast<const std::uint16_t*>(p.index.address),
                        checked_u32(m.indexer_n_heads,"MTP index heads"),p.index_dim,p.ratio,reinterpret_cast<const void*>(p.tensors.index_k_norm.runtime->address),
                        m.rms_norm_epsilon,checked_u32(m.rotary_dim(),"MTP rotary dimension"),m.rope_theta,p.workspace.block_scores,selection,rotary_source...);
                });
            }
        } else with_qwen_rotary(m, stream, [&](auto... rotary_source) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(block_score_rows_kernel<SmallWeightKind::f32, decltype(rotary_source)...>), dim3(blocks, static_cast<unsigned>(select.size())),
                dim3(a.index_dim), 2ULL * a.index_dim * sizeof(float), stream, b, checked_u32(m.indexer_n_heads, "MTP index heads"), a.index_dim,
                a.ratio, reinterpret_cast<const void*>(a.tensors.index_k_norm.runtime->address), m.rms_norm_epsilon,
                checked_u32(m.rotary_dim(), "MTP rotary dimension"), m.rope_theta, selection, rotary_source...);
        });
        check_hip(hipGetLastError(), "MTP batch compressed index score enqueue");
        // Each row validates its own block count (the one-token check covers p.blocks scores).
        for (std::size_t i = 0; i < select.size(); ++i) {
            const auto& p = rows[select[i]];
            MtpBatchFiniteRows f{};f.values[0] = p.workspace.block_scores; f.status[0] = reinterpret_cast<std::uint32_t*>(p.status.address);
            hipLaunchKernelGGL(validate_indexed_graph_values_rows, dim3(1), dim3(kThreads), 0, stream, f, p.blocks);
            check_hip(hipGetLastError(), "MTP batch block score validation enqueue");
        }
        std::vector<std::unique_ptr<MtpPoolSelectionCheck>> selection_checks;
        if(any_pool && std::getenv("STRIX_QSA_MTP_POOL_CHECK") && std::getenv("STRIX_QSA_MTP_POOL_CHECK")[0]=='1') for(const auto r:select) {
            const auto& p=rows[r];const auto& w=p.workspace;
            selection_checks.push_back(std::make_unique<MtpPoolSelectionCheck>(contexts[r].resources->qsa_pool(),
                HipQsaPoolRowDescriptor{w.index_query,w.block_scores,w.selected_indices,p.visible,p.blocks,p.topk,0U,nullptr},stream));
        }
        hipLaunchKernelGGL(select_blocks_rows_kernel, dim3(static_cast<unsigned>(select.size())), dim3(kSelectThreads), 0, stream, b,
            a.topk, a.ratio, selection);
        check_hip(hipGetLastError(), "MTP batch compressed selection enqueue");
        for(auto& check:selection_checks) check->finish();
    }
}
void launch_hip_mtp_sparse_attention_draft_rows(std::span<const MtpNativeContext> contexts, void* raw_stream) {
    const auto rows = mtp_batch_prepare(contexts);
    if (!raw_stream) throw std::invalid_argument("MTP attention batch needs a stream");
    const auto stream = static_cast<hipStream_t>(raw_stream);const auto& a = rows.front();
    const auto n = static_cast<std::uint32_t>(rows.size());
    std::vector<std::uint32_t> all(n);for (std::uint32_t r = 0; r < n; ++r) all[r] = r;
    HipMtpDraftGemvRows qkv{};
    qkv.weights[0] = reinterpret_cast<const void*>(a.tensors.q.runtime->address);
    qkv.weights[1] = reinterpret_cast<const void*>(a.tensors.k.runtime->address);
    qkv.weights[2] = reinterpret_cast<const void*>(a.tensors.v.runtime->address);
    qkv.first[0] = 0; qkv.first[1] = a.q_width; qkv.first[2] = a.q_width + a.kv_width; qkv.first[3] = a.q_width + 2U * a.kv_width;
    qkv.segments = 3;
    for (std::uint32_t r = 0; r < n; ++r) {
        const auto& p = rows[r];
        qkv.input[r] = reinterpret_cast<const void*>(p.input.address);
        qkv.output[0][r] = p.workspace.q_gate; qkv.output[1][r] = p.workspace.current_k; qkv.output[2][r] = p.workspace.current_v;
    }
    launch_hip_mtp_draft_q8_gemv_rows(qkv, n, true, a.hidden, false, stream);
    mtp_batch_norm_rope(rows, all, contexts, &WorkspacePointers::q_gate, a.heads, a.dim, a.dim * 2U, *a.tensors.q_norm.runtime, stream);
    mtp_batch_norm_rope(rows, all, contexts, &WorkspacePointers::current_k, a.kv_heads, a.dim, a.dim, *a.tensors.k_norm.runtime, stream);
    mtp_batch_finite(rows, all, &WorkspacePointers::q_gate, a.q_width, stream);
    MtpBatchPublishRows publish{};
    for (std::uint32_t r = 0; r < n; ++r) {
        const auto& p = rows[r];
        publish.key[r] = p.workspace.current_k; publish.value[r] = p.workspace.current_v;
        publish.kv_state[r] = reinterpret_cast<std::uint16_t*>(p.kv.address);
        publish.position[r] = contexts[r].position; publish.capacity[r] = contexts[r].resources->context_capacity();
        publish.status[r] = reinterpret_cast<std::uint32_t*>(p.status.address);
    }
    hipLaunchKernelGGL(mtp_cache_publish_rows_kernel, dim3(n), dim3(kThreads), 0, stream, publish, 0U, a.kv_width);
    check_hip(hipGetLastError(), "MTP batch KV publish enqueue");
    MtpBatchAttentionRows views{};std::uint32_t slots = 0;
    for (std::uint32_t r = 0; r < n; ++r) {
        const auto& p = rows[r];
        views.q_gate[r] = p.workspace.q_gate; views.selected[r] = p.workspace.selected_indices; views.scores[r] = p.workspace.attention_scores;
        views.output[r] = p.workspace.attention_output; views.kv[r] = reinterpret_cast<const std::uint16_t*>(p.kv.address);
        views.count[r] = p.selected; views.maximum_selected[r] = p.maximum_selected; slots = std::max(slots, p.selected);
    }
    const auto groups = (slots + 3U) / 4U;
    const auto waves = static_cast<std::uint64_t>(n) * a.heads * groups;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_score_sessions_wave_kernel<4U>), dim3(checked_u32((waves + 7U) / 8U, "MTP batch score grid")),
        dim3(kThreads), 0, stream, views, n, a.heads, a.kv_heads, a.dim, slots);
    check_hip(hipGetLastError(), "MTP batch attention scores enqueue");
    const GraphSelectionArguments selection{{}, a.ratio, a.topk};
    hipLaunchKernelGGL(attention_tail_sessions_kernel, dim3(a.heads, n), dim3(kThreads), 0, stream, views, a.heads, a.kv_heads, a.dim, selection);
    check_hip(hipGetLastError(), "MTP batch fused attention tail enqueue");
    mtp_batch_finite(rows, all, &WorkspacePointers::attention_output, a.attention_width, stream);
    HipMtpDraftGemvRows out{};
    out.weights[0] = reinterpret_cast<const void*>(a.tensors.out.runtime->address);
    out.first[0] = 0; out.first[1] = a.hidden; out.segments = 1;
    for (std::uint32_t r = 0; r < n; ++r) {
        out.input[r] = rows[r].workspace.attention_output; out.output[0][r] = reinterpret_cast<void*>(rows[r].output.address);
    }
    launch_hip_mtp_draft_q8_gemv_rows(out, n, false, a.attention_width, true, stream);
}
#if defined(STRIX_BUILD_QWEN_ROTARY_DIAGNOSTIC)
#include "../tests/hip_qwen_rotary_fixtures.inc"
#endif
} // namespace strix

#if defined(STRIX_BUILD_QPERSIST_DIAGNOSTIC)
namespace strix::qpersist_diagnostic {
namespace {
void require_stream(hipStream_t stream) {
    if (!stream || stream == hipStreamPerThread || stream == hipStreamLegacy)
        throw std::invalid_argument("qpersist diagnostic requires an owned stream");
}
void validate(const Attention& a, hipStream_t stream) {
    require_stream(stream);
    if ((a.rows.size() != 1U && a.rows.size() != 2U && a.rows.size() != 4U && a.rows.size() != 8U) ||
        !a.kv || !a.heads || a.heads > 64U || !a.kv_heads || a.heads % a.kv_heads ||
        (a.dim != 128U && a.dim != 256U) || !a.maximum || a.maximum > 2051U ||
        (a.codes == nullptr) != (a.deltas == nullptr))
        throw std::invalid_argument("qpersist diagnostic attention shape is not admitted");
    for (const auto& row : a.rows)
        if (!row.query || !row.selected || !row.scores || !row.output || !row.count || row.count > a.maximum)
            throw std::invalid_argument("qpersist diagnostic attention row is not admitted");
}
template <unsigned Rows> QsaRowsAttentionViews<Rows> views(const Attention& a) {
    QsaRowsAttentionViews<Rows> v{};
    for (unsigned r = 0; r < Rows; ++r) {
        v.q_gate[r] = a.rows[r].query; v.selected[r] = a.rows[r].selected;
        v.scores[r] = a.rows[r].scores; v.output[r] = a.rows[r].output; v.count[r] = a.rows[r].count;
    }
    return v;
}
template <unsigned Rows> void scores(const Attention& a, bool tiled, hipStream_t stream) {
    const auto v = views<Rows>(a);
    std::uint32_t count = 0;
    for (const auto& row : a.rows) count = std::max(count, row.count);
    if (tiled) {
        const auto waves = static_cast<std::uint64_t>(Rows) * a.heads * ((count + 3U) / 4U);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_score_rows_wave_kernel<Rows, 4U>),
            dim3(static_cast<std::uint32_t>((waves + 7U) / 8U)), dim3(kThreads), 0, stream,
            v, a.kv, a.maximum, a.heads, a.kv_heads, a.dim, count);
    } else hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_score_rows_kernel<Rows>),
        dim3(a.heads * count, Rows), dim3(kThreads), 0, stream,
        v, a.kv, a.maximum, a.heads, a.kv_heads, a.dim);
}
template <unsigned Rows> void tail(const Attention& a, bool fused, hipStream_t stream) {
    const auto v = views<Rows>(a);
    if (fused) {
        if (a.codes) hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_tail_rows_kernel<Rows, false, true>),
            dim3(a.heads, Rows), dim3(kThreads), 0, stream, v, a.kv, a.maximum, a.heads, a.kv_heads, a.dim, a.codes, a.deltas);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_tail_rows_kernel<Rows>),
            dim3(a.heads, Rows), dim3(kThreads), 0, stream, v, a.kv, a.maximum, a.heads, a.kv_heads, a.dim, nullptr, nullptr);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_softmax_rows_kernel<Rows>), dim3(a.heads, Rows), dim3(kThreads), 0,
            stream, v, a.maximum);
        if (a.codes) hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_output_rows_dense_a8_kernel<Rows>),
            dim3(a.heads, Rows), dim3(a.dim), 0, stream, v, a.kv, a.maximum, a.heads, a.kv_heads, a.dim, a.codes, a.deltas);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(attention_output_rows_kernel<Rows>),
            dim3(a.heads, Rows), dim3(a.dim), 0, stream, v, a.kv, a.maximum, a.heads, a.kv_heads, a.dim);
    }
}
}
void enqueue_scores(const Attention& a, bool tiled, hipStream_t stream) {
    validate(a, stream);
    switch (a.rows.size()) {
    case 1U: scores<1U>(a, tiled, stream); break;
    case 2U: scores<2U>(a, tiled, stream); break;
    case 4U: scores<4U>(a, tiled, stream); break;
    default: scores<8U>(a, tiled, stream); break;
    }
    check_hip(hipGetLastError(), "qpersist diagnostic scores");
}
void enqueue_tail(const Attention& a, bool fused, hipStream_t stream) {
    validate(a, stream);
    switch (a.rows.size()) {
    case 1U: tail<1U>(a, fused, stream); break;
    case 2U: tail<2U>(a, fused, stream); break;
    case 4U: tail<4U>(a, fused, stream); break;
    default: tail<8U>(a, fused, stream); break;
    }
    check_hip(hipGetLastError(), "qpersist diagnostic tail");
}
bool window_v2_admitted(const Attention& a) {
    return window_attn_v2_admitted(kWindowV2Group, a.heads, a.kv_heads, a.dim);
}
void enqueue_window_v2_scores(const Attention& a, hipStream_t stream) {
    validate(a, stream);
    if (!window_v2_admitted(a)) throw std::invalid_argument("qpersist diagnostic window v2 score is not admitted");
    std::uint32_t count = 0;
    for (const auto& row : a.rows) count = std::max(count, row.count);
    switch (a.rows.size()) {
    case 1U: launch_window_attn_v2_scores<1U>(views<1U>(a), a.kv, a.maximum, a.heads, a.kv_heads, a.dim, count, stream); break;
    case 2U: launch_window_attn_v2_scores<2U>(views<2U>(a), a.kv, a.maximum, a.heads, a.kv_heads, a.dim, count, stream); break;
    case 4U: launch_window_attn_v2_scores<4U>(views<4U>(a), a.kv, a.maximum, a.heads, a.kv_heads, a.dim, count, stream); break;
    default: launch_window_attn_v2_scores<8U>(views<8U>(a), a.kv, a.maximum, a.heads, a.kv_heads, a.dim, count, stream); break;
    }
}
void enqueue_cache(const Cache& c, bool fused, hipStream_t stream) {
    require_stream(stream);
    if (!c.status || !c.capacity || (!c.index_dim && !c.kv_width) || c.index_dim > 256U || c.kv_width > 8192U ||
        (c.index_dim && (!c.index || !c.raw_state)) || (c.kv_width && (!c.key || !c.value || !c.kv_state)))
        throw std::invalid_argument("qpersist diagnostic cache shape is not admitted");
    if (fused) hipLaunchKernelGGL(mtp_cache_publish_kernel, dim3(1), dim3(kThreads), 0, stream,
        c.index, c.index_dim, c.key, c.value, c.kv_width, c.raw_state, c.kv_state, c.position, c.capacity, c.status);
    else {
        hipLaunchKernelGGL(validate_mtp_cache_casts, dim3(1), dim3(kThreads), 0, stream,
            c.index, c.index_dim, c.key, c.value, c.kv_width, c.status);
        const GraphPositionKernelArguments guarded{nullptr, c.capacity, c.status};
        if (c.index_dim) hipLaunchKernelGGL(store_raw_index_key_kernel, dim3((c.index_dim + kThreads - 1U) / kThreads),
            dim3(kThreads), 0, stream, c.index, c.raw_state, c.index_dim, c.position, guarded);
        if (c.kv_width) hipLaunchKernelGGL(store_kv_kernel, dim3((c.kv_width + kThreads - 1U) / kThreads),
            dim3(kThreads), 0, stream, c.key, c.value, c.kv_state, c.kv_width, c.position, guarded);
    }
    check_hip(hipGetLastError(), "qpersist diagnostic cache");
}
namespace {
void validate(const DraftAttention& a, hipStream_t stream) {
    require_stream(stream);
    if (!a.query || !a.selected || !a.scores || !a.output || !a.kv || !a.heads || a.heads > 64U || !a.kv_heads ||
        a.heads % a.kv_heads || !a.dim || a.dim > kThreads || !a.maximum || !a.count || a.count > a.maximum ||
        (a.position != nullptr && (!a.status || !a.ratio || !a.context)))
        throw std::invalid_argument("qpersist diagnostic draft attention shape is not admitted");
}
GraphSelectionArguments selection(const DraftAttention& a) {
    return {{a.position, a.context, a.status}, a.ratio, a.topk};
}
} // namespace
bool draft_attention_admitted(const DraftAttention& a, const char* score_variant, const char* tail_variant) {
    return strix::draft_attention_admitted(draft_score_variant(score_variant), draft_tail_variant(tail_variant), a.heads,
        a.kv_heads, a.dim, a.count);
}
// The incumbent variants issue the production launches of mtp_sparse_attention (the D-059 score
// and tail) with the same grids; every E variant must be admitted first.
void enqueue_draft_scores(const DraftAttention& a, const char* variant, hipStream_t stream) {
    validate(a, stream);
    const auto v = draft_score_variant(variant);
    if (v != DraftScoreVariant::w4 && !draft_attention_admitted(a, variant, "incumbent"))
        throw std::invalid_argument("qpersist diagnostic draft score variant is not admitted");
    QsaRowsAttentionViews<1U> views{};
    views.q_gate[0] = a.query; views.selected[0] = a.selected; views.scores[0] = a.scores; views.count[0] = a.count;
    launch_draft_attention_scores(v, views, a.kv, a.maximum, a.heads, a.kv_heads, a.dim, a.count, stream);
}
void enqueue_draft_tail(const DraftAttention& a, const char* variant, hipStream_t stream) {
    validate(a, stream);
    const auto v = draft_tail_variant(variant);
    if (v != DraftTailVariant::incumbent && !draft_attention_admitted(a, "w4", variant))
        throw std::invalid_argument("qpersist diagnostic draft tail variant is not admitted");
    launch_draft_attention_tail(v, a.query, a.kv, a.selected, a.scores, a.count, a.maximum, a.heads, a.kv_heads, a.dim,
        a.output, selection(a), stream);
}
void enqueue_draft_one_block(const DraftAttention& a, bool v2, bool tail, hipStream_t stream) {
    validate(a, stream);
    if (a.position != nullptr || (v2 && !window_attn_v2_admitted(kDraftV2Group, a.heads, a.kv_heads, a.dim)))
        throw std::invalid_argument("qpersist diagnostic draft one-block score is not admitted");
    const auto dynamic = selection(a);
    launch_draft_one_block_scores(v2, a.query, a.kv, a.selected, a.scores, a.count, a.maximum, a.heads, a.kv_heads, a.dim,
        dynamic, stream);
    if (!tail) return;
    // mtp_sparse_attention's unfused tail (no STRIX_QPERSIST_QSA_TAIL).
    hipLaunchKernelGGL(attention_softmax_kernel, dim3(a.heads), dim3(kThreads), 0, stream, a.scores, a.count, a.maximum, dynamic);
    hipLaunchKernelGGL(attention_output_kernel, dim3(a.heads), dim3(a.dim), 0, stream, a.query, a.kv, a.selected, a.scores,
        a.count, a.maximum, a.heads, a.kv_heads, a.dim, a.output, dynamic);
    check_hip(hipGetLastError(), "qpersist diagnostic draft one-block tail");
}
bool window_v2_switch(const char* setting, const char* defaults, bool baseline) {
    return window_attn_v2_switch(setting, defaults, baseline);
}
bool window_v2_routed(const WindowV2Route& r) {
    return window_attn_v2_route(r.enabled, r.verify_window, r.rows, r.int8, r.fast, r.window_grouped, r.score_slots, r.heads,
        r.kv_heads, r.dim);
}
bool draft_v2_routed(bool enabled, bool draft, bool qpersist_draft_score, unsigned heads, unsigned kv_heads, unsigned dim) {
    return draft_attn_v2_route(enabled, draft, qpersist_draft_score, heads, kv_heads, dim);
}
std::string draft_attention_selection(const char* attn, const char* score, const char* tail) {
    const auto c = resolve_draft_attention(attn, score, tail);
    if (!c.on) return "incumbent";
    return std::string(kDraftScoreNames[static_cast<std::size_t>(c.score)]) + "+" +
        std::string(kDraftTailNames[static_cast<std::size_t>(c.tail)]) + (c.defaulted ? " (default)" : " (explicit)");
}
void enqueue_draft_tail_stage(const DraftAttention& a, const char* variant, bool value, hipStream_t stream) {
    validate(a, stream);
    const auto v = draft_tail_variant(variant);
    if ((v != DraftTailVariant::v2 && v != DraftTailVariant::v4) || !draft_attention_admitted(a, "w4", variant))
        throw std::invalid_argument("qpersist diagnostic draft tail stage needs an admitted two-launch variant");
    if (value) launch_draft_attention_value(v, a.query, a.kv, a.selected, a.scores, a.count, a.maximum, a.heads, a.kv_heads,
        a.dim, a.output, selection(a), stream);
    else launch_draft_attention_softmax(a.scores, a.count, a.maximum, a.heads, selection(a), stream);
}
} // namespace strix::qpersist_diagnostic
#endif // STRIX_BUILD_QPERSIST_DIAGNOSTIC
