#include "strix/qpf2k_defaults.hpp"
#include "strix/gdn_recurrent_layout_diagnostic.hpp"
#include "strix/hip_ggml_weight_formats.hpp"
#ifdef STRIX_BUILD_GDN_LM_WAVE8_DIAGNOSTIC
#include "strix/hip_gdn_lm_wave8_diagnostic.hpp"
#endif
#include "strix/hip_gdn_launcher.hpp"
#include "strix/qpersist.hpp"
#include "strix/prefill_identity.hpp"
#include "strix/known_window_cohort.hpp"
#include "hip_q8_rows_gemv.hpp"
#include "hip_dense_f16_tiled.hpp"
#include "strix/hip_gdn_islands.hpp"
#include "strix/hip_model_residency.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/q8_projection_organization.hpp"
#include "strix/q8_projection_manifest.hpp"

#include "strix/dispatch_resources.hpp"
#include "strix/prepared_tensor_views.hpp"
#include "strix/execution_buffers.hpp"
#include "strix/memory_plan.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/graph_operator_control.hpp"

#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_fp16.h>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <initializer_list>
#include <bit>
#include <memory>
#include <tuple>
#include <vector>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace strix {
namespace {

constexpr std::uint32_t kProjectionThreads = 256;
constexpr std::uint32_t kRecurrentThreads = 128;
constexpr std::uint32_t kValueTile = 32;
constexpr std::uint64_t kLauncherTag =
    0x47444e4c41554e32ULL;
constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

struct Q8Block {
    __half scale;
    std::int8_t values[32];
};

struct IQ4NLBlock {
    __half d;
    std::uint8_t qs[16];
};

struct IQ4XSBlock {
    __half d;
    std::uint16_t scales_h;
    std::uint8_t scales_l[4];
    std::uint8_t qs[128];
};

static_assert(sizeof(Q8Block) == 34);
static_assert(sizeof(IQ4NLBlock) == 18);
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
    iq4_nl,
    iq4_xs,
#define STRIX_GGML_KIND_ENUM(kind, type) kind,
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_ENUM) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_ENUM)
#undef STRIX_GGML_KIND_ENUM
};

enum class ProjectionMode { legacy256, virtual_stream, prefetch8, prefetch8_wave8 };

enum class SmallWeightKind : std::uint32_t {
    f32 = 0,
    f16 = 1,
    bf16 = 2,
};

struct TensorPair {
    const ManifestTensor* manifest = nullptr;
    const RuntimeTensorBinding* runtime = nullptr;
};

struct GdnTensors {
    TensorPair qkv;
    TensorPair z;
    TensorPair beta;
    TensorPair alpha;
    TensorPair conv;
    TensorPair dt;
    TensorPair a;
    TensorPair norm;
    TensorPair out;
};

struct Workspace {
    float* qkv = nullptr;
    float* z = nullptr;
    float* beta = nullptr;
    float* gate = nullptr;
    float* core = nullptr;
    std::uint64_t elements = 0;
};

__global__ void validate_gdn_graph_position(const std::uint64_t* position,
    std::uint64_t context_tokens, std::uint32_t* status) {
    if (*position >= context_tokens) atomicOr(status, 2U);
}
__global__ void validate_gdn_graph_input(const std::uint16_t* input, std::uint32_t hidden,
    std::uint32_t* status, std::uint32_t failure) {
    if (atomicAdd(status, 0U) != 0) return;
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < hidden; index += gridDim.x * blockDim.x) {
        const auto bits = static_cast<std::uint32_t>(input[index]) << 16U;
        if ((bits & 0x7f800000U) == 0x7f800000U) atomicOr(status, failure);
    }
}
__global__ void validate_gdn_graph_values(const float* values, std::uint64_t count, std::uint32_t* status) {
    if (atomicAdd(status, 0U) != 0) return;
    for (std::uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += gridDim.x * blockDim.x) {
        if (!isfinite(values[i])) atomicOr(status, 8U);
    }
}

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
    static_assert(
        sizeof(bits) == sizeof(value));
    std::memcpy(
        &bits,
        &value,
        sizeof(bits));
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
            std::numeric_limits<
                std::uint32_t>::max()) {
        throw std::invalid_argument(
            std::string(name) +
            " does not fit uint32");
    }
    return static_cast<std::uint32_t>(
        value);
}

void validate_config(
    const HipGdnLauncherConfig& config) {
    if (config.dense_trellis_projections && (!config.dense_a8_projections || config.enable_device_graph_inputs))
        throw std::invalid_argument("dense trellis GDN requires its eager dense-A8 parent recipe");
#if !defined(STRIX_HAS_EXPERIMENTAL_DENSE_TRELLIS)
    if (config.dense_trellis_projections) throw std::invalid_argument("dense trellis runtime is disabled in this build");
#endif
    if (const auto error =
            config.model.validate();
        !error.empty()) {
        throw std::invalid_argument(
            error);
    }
    if (config.recurrent_layout != HipGdnRecurrentLayout::baseline &&
        config.recurrent_layout != HipGdnRecurrentLayout::stride32 &&
        config.recurrent_layout != HipGdnRecurrentLayout::stride33 &&
        config.recurrent_layout != HipGdnRecurrentLayout::transposed)
        throw std::invalid_argument("GDN recurrent LDS layout is invalid");
    if (config.recurrent_layout != HipGdnRecurrentLayout::baseline && config.enable_device_graph_inputs)
        throw std::invalid_argument("GDN recurrent LDS layouts support eager execution only");
    validate_gdn_recurrent_storage(config.recurrent_storage);
    if (config.recurrent_storage != GdnRecurrentStorage::key_value && config.enable_device_graph_inputs)
        throw std::invalid_argument("GDN recurrent storage supports eager execution only");
    if (config.prefetch_q8_input_projections && !config.virtual_q8_input_projections)
        throw std::invalid_argument("HIP GDN prefetch Q8 input requires virtual Q8 input");
    if (config.wave8_q8_input_projections) {
        q8_projection_organization::validate_modes(config.virtual_q8_input_projections,
            config.prefetch_q8_input_projections, true,
            config.activation == HipActivationStorage::bf16, config.model.hidden_size,
            config.model.gdn_value_heads * config.model.gdn_value_head_dim);
        if (config.enable_device_graph_inputs)
            throw std::invalid_argument("wave8 GDN input supports eager execution only");
    }
    if (config.wave8_q8_output_projection) {
        if (!config.virtual_q8_output_projection)
            throw std::invalid_argument("wave8 GDN output requires virtual Q8 output");
        if (config.enable_device_graph_inputs)
            throw std::invalid_argument("wave8 GDN output supports eager execution only");
        if (config.model.gdn_value_heads > 10240U / config.model.gdn_value_head_dim)
            throw std::invalid_argument("wave8 GDN output width outside admitted bounds");
        q8_projection_organization::validate_geometry(
            config.model.gdn_value_heads * config.model.gdn_value_head_dim, config.model.hidden_size);
    }
    if (config.activation !=
        HipActivationStorage::bf16) {
        throw std::invalid_argument(
            "HIP GDN v1 targets canonical BF16 activations only");
    }
    if (!config.model.gdn_qk_l2norm ||
        !config.model.gdn_conv_silu ||
        !config.model.gdn_output_gate_sigmoid) {
        throw std::invalid_argument(
            "HIP GDN v1 requires pinned Qwen4Exp GDN semantics");
    }
    if (config.model.gdn_key_head_dim !=
            kRecurrentThreads ||
        config.model.gdn_value_head_dim !=
            kRecurrentThreads ||
        config.model.gdn_key_head_dim !=
            config.model.gdn_value_head_dim ||
        config.model.gdn_value_heads %
                config.model.gdn_key_heads !=
            0 ||
        config.model.gdn_value_head_dim %
                kValueTile !=
            0) {
        throw std::invalid_argument(
            "HIP GDN v1 requires 128-wide equal/tileable key and value heads");
    }
}

WeightKind weight_kind(
    std::uint32_t ggml_type) {
    switch (ggml_type) {
    case 0: return WeightKind::f32;
    case 1: return WeightKind::f16;
    case 8: return WeightKind::q8_0;
    case 20: return WeightKind::iq4_nl;
    case 23: return WeightKind::iq4_xs;
#define STRIX_GGML_KIND_MAP(kind, type) case type: return WeightKind::kind;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_MAP) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_MAP)
#undef STRIX_GGML_KIND_MAP
    case 30: return WeightKind::bf16;
    default:
        throw std::invalid_argument(
            "HIP GDN projection supports F32, F16, BF16, Q8_0, IQ4_NL and IQ4_XS");
    }
}

SmallWeightKind small_weight_kind(
    std::uint32_t ggml_type) {
    switch (ggml_type) {
    case 0: return SmallWeightKind::f32;
    case 1: return SmallWeightKind::f16;
    case 30: return SmallWeightKind::bf16;
    default:
        throw std::invalid_argument(
            "HIP GDN small tensors support F32, F16 and BF16");
    }
}

__device__ float bf16_to_float(
    std::uint16_t bits) {
    return __uint_as_float(
        static_cast<std::uint32_t>(
            bits) << 16U);
}

__device__ std::uint16_t float_to_bf16(
    float value) {
    const auto bits =
        __float_as_uint(value);
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

__device__ std::uint8_t iq4_local_scale(
    const IQ4XSBlock& block,
    std::uint32_t subblock) {
    const auto low_byte =
        block.scales_l[subblock / 2U];
    const auto low =
        static_cast<std::uint8_t>(
            (low_byte >>
             (4U *
              (subblock % 2U))) &
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
    return static_cast<const float*>(
        weights)[
        row *
            row_width +
        feature];
}

template <>
__device__ float load_weight<WeightKind::f16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    return __half2float(
        static_cast<const __half*>(
            weights)[
            row *
                row_width +
            feature]);
}

template <>
__device__ float load_weight<WeightKind::bf16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    return bf16_to_float(
        static_cast<
            const std::uint16_t*>(
                weights)[
            row *
                row_width +
            feature]);
}

template <>
__device__ float load_weight<WeightKind::q8_0>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const Q8Block*>(
            weights);
    const auto blocks_per_row =
        row_width / 32U;
    const auto& block =
        blocks[
            row *
                blocks_per_row +
            feature / 32U];
    return __half2float(
               block.scale) *
           static_cast<float>(
               block.values[
                   feature % 32U]);
}

template <>
__device__ float load_weight<WeightKind::iq4_nl>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const IQ4NLBlock*>(
            weights);
    const auto blocks_per_row =
        row_width / 32U;
    const auto& block =
        blocks[
            row *
                blocks_per_row +
            feature / 32U];
    const auto within =
        feature % 32U;
    const auto packed =
        block.qs[
            within % 16U];
    const auto code =
        within < 16U
            ? static_cast<std::uint8_t>(
                  packed & 0x0fU)
            : static_cast<std::uint8_t>(
                  packed >> 4U);
    return __half2float(block.d) *
           static_cast<float>(
               iq4_codebook[code]);
}

template <>
__device__ float load_weight<WeightKind::iq4_xs>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const IQ4XSBlock*>(
            weights);
    const auto blocks_per_row =
        row_width / 256U;
    const auto& block =
        blocks[
            row *
                blocks_per_row +
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

__device__ float load_small(
    const void* values,
    SmallWeightKind kind,
    std::uint64_t index) {
    switch (kind) {
    case SmallWeightKind::f32:
        return static_cast<
            const float*>(values)[index];
    case SmallWeightKind::f16:
        return __half2float(
            static_cast<
                const __half*>(
                    values)[index]);
    case SmallWeightKind::bf16:
        return bf16_to_float(
            static_cast<
                const std::uint16_t*>(
                    values)[index]);
    }
    return 0.0F;
}

#define STRIX_GGML_KIND_LOAD(kind, type)                                                          \
    template <>                                                                                   \
    __device__ float load_weight<WeightKind::kind>(                                               \
        const void* weights, std::uint32_t row_width, std::uint64_t row, std::uint32_t feature) { \
        return hip_ggml::element<type>(weights, row_width, row, feature);                        \
    }
STRIX_GGML_KINDS_K(STRIX_GGML_KIND_LOAD) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_LOAD)
#undef STRIX_GGML_KIND_LOAD

std::uint32_t ggml_type_of(WeightKind kind) {
    switch (kind) {
#define STRIX_GGML_KIND_TYPE(k, type) case WeightKind::k: return type;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_TYPE) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_TYPE)
#undef STRIX_GGML_KIND_TYPE
    default: return 0;
    }
}

std::uint64_t required_weight_bytes(
    WeightKind kind,
    std::uint32_t input_width,
    std::uint64_t output_rows) {
    switch (kind) {
    case WeightKind::f32:
        return
            static_cast<std::uint64_t>(
                input_width) *
            output_rows * 4ULL;
    case WeightKind::f16:
    case WeightKind::bf16:
        return
            static_cast<std::uint64_t>(
                input_width) *
            output_rows * 2ULL;
    case WeightKind::q8_0:
    case WeightKind::iq4_nl:
        if (input_width % 32U != 0) {
            throw std::invalid_argument(
                "Q8_0/IQ4_NL GDN projection width must divide 32");
        }
        return
            static_cast<std::uint64_t>(
                input_width / 32U) *
            output_rows *
            (kind == WeightKind::q8_0
                 ? sizeof(Q8Block)
                 : sizeof(IQ4NLBlock));
    case WeightKind::iq4_xs:
        if (input_width % 256U != 0) {
            throw std::invalid_argument(
                "IQ4_XS GDN projection width must divide 256");
        }
        return
            static_cast<std::uint64_t>(
                input_width / 256U) *
            output_rows *
            sizeof(IQ4XSBlock);
    default: {
        const auto g = hip_ggml::geometry(ggml_type_of(kind));
        if (g.elements == 0 || input_width % g.elements != 0) {
            throw std::invalid_argument("weight width is not a multiple of its GGUF block");
        }
        return static_cast<std::uint64_t>(input_width / g.elements) * output_rows * g.bytes;
    }
    }
    throw std::logic_error(
        "unreachable GDN weight kind");
}

std::uint64_t small_weight_bytes(
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
        "unreachable GDN small weight kind");
}


template <WeightKind Kind>
__global__ void project_bf16_to_fp32_kernel(
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

    __shared__ float scratch[kProjectionThreads];
    float sum = 0.0F;
    for (std::uint32_t feature =
             static_cast<std::uint32_t>(
                 threadIdx.x);
         feature < input_width;
         feature += kProjectionThreads) {
        sum +=
            bf16_to_float(
                input[feature]) *
            load_weight<Kind>(
                weights,
                input_width,
                out,
                feature);
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();

    for (std::uint32_t stride =
             kProjectionThreads / 2U;
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

// Eight ascending virtual leaves per physical lane preserve the original
// 256-leaf tree, including its operand order and BF16-in-FP32 epilogue.
__global__ __launch_bounds__(32)
void project_virtual_q8_bf16_to_fp32_kernel(
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
            if (lane < distance) partial[local] += partner;
        }
    }
    if (lane == 0) {
        partial[0] += partial[4]; partial[1] += partial[5];
        partial[2] += partial[6]; partial[3] += partial[7];
        partial[0] += partial[2]; partial[1] += partial[3];
        partial[0] += partial[1];
        output[row] = round_bf16(partial[0]);
    }
}

// Distinct opt-in load organization; existing virtual and legacy kernels
// remain unchanged. Actual ISA, not these arrays, qualifies load batching.
__global__ __launch_bounds__(32)
void project_prefetch8_q8_bf16_to_fp32_kernel(
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
            if (lane < distance) partial[local] += partner;
        }
    }
    if (lane == 0) {
        partial[0] += partial[4]; partial[1] += partial[5];
        partial[2] += partial[6]; partial[3] += partial[7];
        partial[0] += partial[2]; partial[1] += partial[3];
        partial[0] += partial[1];
        output[row] = round_bf16(partial[0]);
    }
}
// Explicit geometry overlay of the incumbent prefetch8 body above.
__global__ __launch_bounds__(256)
void project_prefetch8_wave8_q8_bf16_to_fp32_kernel(
    const std::uint16_t* input, const void* weights,
    std::uint32_t input_width, std::uint32_t output_width, float* output) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x) * 8U + static_cast<std::uint32_t>(threadIdx.x) / 32U;
    if (row >= output_width) return;
    const auto lane = static_cast<std::uint32_t>(threadIdx.x) % 32U;
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
            if (lane < distance) partial[local] += partner;
        }
    }
    if (lane == 0) {
        partial[0] += partial[4]; partial[1] += partial[5];
        partial[2] += partial[6]; partial[3] += partial[7];
        partial[0] += partial[2]; partial[1] += partial[3];
        partial[0] += partial[1];
        output[row] = round_bf16(partial[0]);
    }
}


template <WeightKind Kind>
__global__ void project_fp32_to_bf16_kernel(
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

    __shared__ float scratch[kProjectionThreads];
    float sum = 0.0F;
    for (std::uint32_t feature =
             static_cast<std::uint32_t>(
                 threadIdx.x);
         feature < input_width;
         feature += kProjectionThreads) {
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
             kProjectionThreads / 2U;
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

// One wave simulates the same256 original leaves: virtual t=8*lane+j.
// Ascending per-leaf FMA chains and the128..1 tree are unchanged.
__global__ __launch_bounds__(32)
void project_virtual_q8_fp32_to_bf16_kernel(
    const float* input, const void* weights, std::uint32_t input_width,
    std::uint32_t output_width, std::uint16_t* output) {
    const auto row=static_cast<std::uint32_t>(blockIdx.x);
    if(row>=output_width) return;
    const auto lane=static_cast<std::uint32_t>(threadIdx.x);
    float partial[8]{};
    for(std::uint64_t chunk=0;chunk<input_width;chunk+=256U) {
#pragma unroll
        for(unsigned local=0;local<8;++local) {
            const auto feature=chunk+8U*lane+local;
            if(feature<input_width)
                partial[local]+=input[feature]*load_weight<WeightKind::q8_0>(
                    weights,input_width,row,static_cast<std::uint32_t>(feature));
        }
    }
#pragma unroll
    for(unsigned distance=16;distance>0;distance/=2) {
#pragma unroll
        for(unsigned local=0;local<8;++local) {
            const auto partner=__shfl_down(partial[local],distance,32);
            if(lane<distance) partial[local]+=partner;
        }
    }
    if(lane==0) {
        partial[0]+=partial[4];partial[1]+=partial[5];
        partial[2]+=partial[6];partial[3]+=partial[7];
        partial[0]+=partial[2];partial[1]+=partial[3];partial[0]+=partial[1];
        output[row]=float_to_bf16(partial[0]);
    }
}

// Actual incumbent virtual-output ISA uses lower/self as src0 at every tree
// site. Keep this new-only helper distinct from the input projection trees.
__device__ __forceinline__ float gdn_output_virtual_tree_add(float lower, float upper) {
    float result;
    asm volatile("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(lower), "v"(upper));
    return result;
}

// One complete wave owns one output. Eight ascending virtual leaves reproduce
// the incumbent256-leaf DAG and BF16 epilogue; waves share no mutable storage.
__global__ __launch_bounds__(256)
void project_prefetch8_wave8_q8_fp32_to_bf16_kernel(
    const float* input, const void* weights, std::uint32_t input_width,
    std::uint32_t output_width, std::uint16_t* output) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x) * 8U + static_cast<std::uint32_t>(threadIdx.x) / 32U;
    if (row >= output_width) return;
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
    // The shuffled value is read before this stage overwrites dead upper lanes.
    // Lane0's lineage therefore preserves the original128..8 stages exactly.
#pragma unroll
    for (unsigned distance = 16; distance > 0; distance /= 2) {
#pragma unroll
        for (unsigned local = 0; local < 8; ++local)
            partial[local] = gdn_output_virtual_tree_add(partial[local], __shfl_down(partial[local], distance, 32));
    }
    if (lane == 0) {
        partial[0] = gdn_output_virtual_tree_add(partial[0], partial[4]);
        partial[1] = gdn_output_virtual_tree_add(partial[1], partial[5]);
        partial[2] = gdn_output_virtual_tree_add(partial[2], partial[6]);
        partial[3] = gdn_output_virtual_tree_add(partial[3], partial[7]);
        partial[0] = gdn_output_virtual_tree_add(partial[0], partial[2]);
        partial[1] = gdn_output_virtual_tree_add(partial[1], partial[3]);
        partial[0] = gdn_output_virtual_tree_add(partial[0], partial[1]);
        output[row] = float_to_bf16(partial[0]);
    }
}


#ifdef STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC
// R2 reference kernels, kept for the R2 projection diagnostic.
// New-only rows2 arithmetic. The selected wave8 input ISA binds coefficient
// to src0 and BF16 activation to src1; its FP32 output counterpart reverses
// those two operands. Both selected trees bind lower/self to src0.
__device__ __forceinline__ float gdn_known_window_q8_coefficient(
    unsigned scale_bits, float code) {
    float result;
    asm volatile("v_fma_mix_f32 %0, %1, %2, neg(0) op_sel_hi:[1,0,0]"
                 : "=v"(result) : "v"(scale_bits), "v"(code));
    return result;
}
__device__ __forceinline__ float gdn_known_window_tree_add(float lower, float upper) {
    float result;
    asm volatile("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(lower), "v"(upper));
    return result;
}
template <bool Bf16Input>
__device__ __forceinline__ void gdn_known_window_q8_rows2_dot(
    const void* input0, const void* input1, const void* weights,
    std::uint32_t width, std::uint32_t out, float (&result)[2]) {
    const auto lane = static_cast<std::uint32_t>(threadIdx.x) % 32U;
    float partial[2][8]{};
    const auto* blocks = static_cast<const Q8Block*>(weights) +
        static_cast<std::uint64_t>(out) * (width / 32U);
    for (std::uint64_t chunk = 0; chunk < width; chunk += 256U) {
        const auto base = chunk + 8U * lane;
        if (base < width) {
            const auto& block = blocks[base / 32U];
            std::int8_t codes[8];
#pragma unroll
            for (unsigned local = 0; local < 8U; ++local)
                codes[local] = block.values[base % 32U + local];
            const auto scale_bits = static_cast<unsigned>(__half_as_ushort(block.scale));
            float values[2][8];
#pragma unroll
            for (unsigned local = 0; local < 8U; ++local) {
                if constexpr (Bf16Input) {
                    values[0][local] = bf16_to_float(static_cast<const std::uint16_t*>(input0)[base + local]);
                    values[1][local] = bf16_to_float(static_cast<const std::uint16_t*>(input1)[base + local]);
                } else {
                    values[0][local] = static_cast<const float*>(input0)[base + local];
                    values[1][local] = static_cast<const float*>(input1)[base + local];
                }
            }
#pragma unroll
            for (unsigned local = 0; local < 8U; ++local) {
                // One decoded coefficient is reused by two independent leaf chains.
                const float coefficient = gdn_known_window_q8_coefficient(scale_bits,
                    static_cast<float>(codes[local]));
                if constexpr (Bf16Input) {
                    asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(partial[0][local])
                                 : "v"(coefficient), "v"(values[0][local]));
                    asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(partial[1][local])
                                 : "v"(coefficient), "v"(values[1][local]));
                } else {
                    asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(partial[0][local])
                                 : "v"(values[0][local]), "v"(coefficient));
                    asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(partial[1][local])
                                 : "v"(values[1][local]), "v"(coefficient));
                }
            }
        }
    }
#pragma unroll
    for (unsigned distance = 16U; distance; distance /= 2U) {
#pragma unroll
        for (unsigned row = 0; row < 2U; ++row) {
#pragma unroll
            for (unsigned local = 0; local < 8U; ++local) {
                const float partner = __shfl_down(partial[row][local], distance, 32);
                if constexpr (Bf16Input) {
                    if (lane < distance)
                        partial[row][local] = gdn_known_window_tree_add(partial[row][local], partner);
                } else partial[row][local] = gdn_known_window_tree_add(partial[row][local], partner);
            }
        }
    }
    if (lane == 0U) {
#pragma unroll
        for (unsigned row = 0; row < 2U; ++row) {
            auto* values = partial[row];
            values[0] = gdn_known_window_tree_add(values[0], values[4]);
            values[1] = gdn_known_window_tree_add(values[1], values[5]);
            values[2] = gdn_known_window_tree_add(values[2], values[6]);
            values[3] = gdn_known_window_tree_add(values[3], values[7]);
            values[0] = gdn_known_window_tree_add(values[0], values[2]);
            values[1] = gdn_known_window_tree_add(values[1], values[3]);
            result[row] = gdn_known_window_tree_add(values[0], values[1]);
        }
    }
}
__global__ __launch_bounds__(256)
void project_wave8_q8_bf16_to_fp32_rows2_kernel(
    const std::uint16_t* input0, const std::uint16_t* input1, const void* weights,
    std::uint32_t width, std::uint32_t outputs, float* output0, float* output1) {
    const auto out = static_cast<std::uint32_t>(blockIdx.x) * 8U + static_cast<std::uint32_t>(threadIdx.x) / 32U;
    if (out >= outputs) return; // complete-wave guard, no block barriers
    float result[2]{};
    gdn_known_window_q8_rows2_dot<true>(input0, input1, weights, width, out, result);
    if ((threadIdx.x % 32U) == 0U) {
        output0[out] = round_bf16(result[0]); output1[out] = round_bf16(result[1]);
    }
}
__global__ __launch_bounds__(256)
void project_wave8_q8_fp32_to_bf16_rows2_kernel(
    const float* input0, const float* input1, const void* weights,
    std::uint32_t width, std::uint32_t outputs, std::uint16_t* output0, std::uint16_t* output1) {
    const auto out = static_cast<std::uint32_t>(blockIdx.x) * 8U + static_cast<std::uint32_t>(threadIdx.x) / 32U;
    if (out >= outputs) return;
    float result[2]{};
    gdn_known_window_q8_rows2_dot<false>(input0, input1, weights, width, out, result);
    if ((threadIdx.x % 32U) == 0U) {
        output0[out] = float_to_bf16(result[0]); output1[out] = float_to_bf16(result[1]);
    }
}
#endif

// Per-row activation/output pointers of the window kernels.
template <unsigned Rows> struct GdnWidePointers {
    const void* input[Rows];
    void* output[Rows];
};
// T-row sites (q8_rows engine) of the R1 wave8 projections above: same
// coefficient, FMA and tree operand bindings as the R2 kernels, per row.
__device__ __forceinline__ float gdn_rows_coefficient(unsigned scale, std::int8_t code) {
    float result;
    asm("v_fma_mix_f32 %0, %1, %2, neg(0) op_sel_hi:[1,0,0]" : "=v"(result) : "v"(scale), "v"(static_cast<float>(code)));
    return result;
}
__device__ __forceinline__ float gdn_rows_tree_add(float lower, float upper) {
    float result;
    asm("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(lower), "v"(upper));
    return result;
}
// qkv, z (and Q8 beta/alpha) segments from BF16 rows to BF16-rounded FP32 workspaces.
template <unsigned Rows> struct GdnInputRowsSite {
    static constexpr bool dense_trellis_inputs = true;
    using Input = std::uint16_t;
    static constexpr unsigned rows = Rows;
    static constexpr bool fused_fma = true;  // Site::fma compiles to v_fmac_f32 (wide VOPD rows)
    static constexpr bool tree_upper_src0 = false;  // add(lower, upper) reads lower as src0 (lane VOPD tree)
    const void* input[Rows];
    q8_rows::Segments segments;
    std::uint32_t width, outputs;
    float* output[4][Rows];
    __device__ static float coefficient(unsigned scale, std::int8_t code) { return gdn_rows_coefficient(scale, code); }
    // Fused single-rounding FMA, the v_fmac_f32 of the R1 kernel; as a builtin the compiler may dual-issue it (VOPD).
    __device__ static void fma(float& acc, float c, float x) { acc = __builtin_fmaf(c, x, acc); }
    __device__ static float add(float lower, float upper) { return gdn_rows_tree_add(lower, upper); }
    __device__ void store(std::uint32_t out, unsigned r, float value) const {
        unsigned m; std::uint32_t row;
        static_cast<void>(q8_rows::segment_row(segments, out, width, m, row));
        output[m][r][row] = round_bf16(value);
    }
};
template <unsigned Rows> struct GdnOutputRowsSite {
    using Input = float;
    static constexpr unsigned rows = Rows;
    static constexpr bool fused_fma = true;  // Site::fma compiles to v_fmac_f32 (wide VOPD rows)
    static constexpr bool tree_upper_src0 = false;  // add(lower, upper) reads lower as src0 (lane VOPD tree)
    const void* input[Rows];
    q8_rows::Segments segments;
    std::uint32_t width, outputs;
    std::uint16_t* output[Rows];
    __device__ static float coefficient(unsigned scale, std::int8_t code) { return gdn_rows_coefficient(scale, code); }
    __device__ static void fma(float& acc, float c, float x) { acc = __builtin_fmaf(x, c, acc); }
    __device__ static float add(float lower, float upper) { return gdn_rows_tree_add(lower, upper); }
    __device__ void store(std::uint32_t out, unsigned r, float value) const { output[r][out] = float_to_bf16(value); }
};
// STRIX_INTPREFILL_DENSE_ROWS (512 | 1024, fast prefill FP16 only): the input projection (qkv + z) of that many uniformly
// strided rows in one launch (row r at row 0's views + r * stride), so each weight byte streams once per that many rows
// instead of once per 256. Same kernels, operands and per-output arithmetic as the 256-row sites.
template <unsigned Rows> struct GdnInputStridedRowsSite {
    using Input = std::uint16_t;
    static constexpr unsigned rows = Rows;
    const unsigned char* input0;
    std::uint64_t input_stride;
    q8_rows::Segments segments;
    std::uint32_t width, outputs;
    unsigned char* output0[2];
    std::uint64_t output_stride;
    __device__ const Input* row_input(unsigned r) const { return reinterpret_cast<const Input*>(input0 + r * input_stride); }
    __device__ void store(std::uint32_t out, unsigned r, float value) const {
        unsigned m; std::uint32_t row;
        static_cast<void>(q8_rows::segment_row(segments, out, width, m, row));
        reinterpret_cast<float*>(output0[m] + r * output_stride)[row] = round_bf16(value);
    }
};
// STRIX_QPF2K_DENSE=1 (q8_rows::tiled): the input / output projections of all uniformly strided rows of a call in one
// launch; the same store arithmetic as GdnInputStridedRowsSite / GdnOutputRowsSite.
struct GdnInputTiledSite {
    using Input = std::uint16_t;
    static constexpr unsigned qpf2k_pre_form = 4U;  // FP16-scratch A (STRIX_QPF2K_DENSE_INLINE=0 only): form 4 (t8: -2.2 .. -4.1 %)
    const unsigned char* input0;
    std::uint64_t input_stride;
    q8_rows::Segments segments;
    std::uint32_t width, outputs;
    unsigned char* output0[2];
    std::uint64_t output_stride;
    __device__ void store(std::uint32_t out, std::uint32_t r, float value) const {
        unsigned m; std::uint32_t row;
        static_cast<void>(q8_rows::segment_row(segments, out, width, m, row));
        reinterpret_cast<float*>(output0[m] + r * output_stride)[row] = round_bf16(value);
    }
};
// FAST route qwen-gdn-fp16-chunk: GdnInputTiledSite's outputs as BF16 bits in one compact row (q | k | v, then z at
// z_offset); GdnInputTiledSite stores these same BF16 values as FP32.
struct GdnInputTiled16Site {
    using Input = std::uint16_t;
    static constexpr unsigned qpf2k_pre_form = GdnInputTiledSite::qpf2k_pre_form;
    const unsigned char* input0;
    std::uint64_t input_stride;
    q8_rows::Segments segments;
    std::uint32_t width, outputs;
    std::uint16_t* output;
    std::uint64_t output_stride;
    std::uint32_t z_offset;
    __device__ void store(std::uint32_t out, std::uint32_t r, float value) const {
        unsigned m; std::uint32_t row;
        static_cast<void>(q8_rows::segment_row(segments, out, width, m, row));
        output[r * output_stride + (m ? z_offset + row : row)] = float_to_bf16(value);
    }
};
struct GdnOutputTiledSite {
    using Input = float;
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

__global__ void conv_update_kernel(
    float* qkv,
    std::uint16_t* conv_state,
    const void* conv_weights,
    SmallWeightKind weight_kind_value,
    std::uint32_t channels,
    std::uint32_t kernel_size, std::uint32_t* graph_status) {
    if (graph_status != nullptr && *graph_status != 0) return;
    const auto channel =
        static_cast<std::uint32_t>(
            blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    if (channel >= channels) {
        return;
    }

    auto* state =
        conv_state +
        static_cast<std::uint64_t>(
            channel) *
            kernel_size;

    for (std::uint32_t index = 0;
         index + 1U < kernel_size;
         ++index) {
        state[index] =
            state[index + 1U];
    }
    state[kernel_size - 1U] =
        float_to_bf16(
            qkv[channel]);

    float sum = 0.0F;
    for (std::uint32_t index = 0;
         index < kernel_size;
         ++index) {
        sum +=
            bf16_to_float(
                state[index]) *
            load_small(
                conv_weights,
                weight_kind_value,
                static_cast<std::uint64_t>(
                    channel) *
                    kernel_size +
                index);
    }
    const float silu =
        sum /
        (1.0F +
         expf(-sum));
    qkv[channel] =
        round_bf16(
            silu);
}

__device__ float softplus(
    float value) {
    if (value > 20.0F) {
        return value;
    }
    if (value < -20.0F) {
        return expf(value);
    }
    return log1pf(
        expf(value));
}

__global__ void transform_beta_gate_kernel(
    float* beta,
    float* gate,
    const void* dt,
    SmallWeightKind dt_kind,
    const void* a,
    SmallWeightKind a_kind,
    std::uint32_t heads) {
    const auto head =
        static_cast<std::uint32_t>(
            blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    if (head >= heads) {
        return;
    }

    beta[head] =
        round_bf16(
            1.0F /
            (1.0F +
             expf(-beta[head])));

    const float biased =
        gate[head] +
        load_small(
            dt,
            dt_kind,
            head);
    gate[head] =
        load_small(
            a,
            a_kind,
            head) *
        softplus(
            biased);
}

__global__ void normalize_qk_kernel(
    float* qkv,
    std::uint32_t key_heads,
    std::uint32_t head_dim,
    float epsilon, std::uint32_t* graph_status) {
    __shared__ unsigned graph_reject;
    if (graph_status != nullptr) {
        if (threadIdx.x == 0) graph_reject = atomicAdd(graph_status, 0U) != 0;
        __syncthreads();
        if (graph_reject) return;
    }
    const auto block =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const bool is_key =
        block >= key_heads;
    const auto head =
        is_key
            ? block - key_heads
            : block;
    if (head >= key_heads ||
        threadIdx.x >= head_dim) {
        return;
    }

    __shared__ float scratch[kRecurrentThreads];

    const auto base =
        (is_key
             ? static_cast<std::uint64_t>(
                   key_heads) *
                   head_dim
             : 0ULL) +
        static_cast<std::uint64_t>(
            head) *
            head_dim;

    const float value =
        qkv[
            base +
            threadIdx.x];
    scratch[threadIdx.x] =
        value * value;
    __syncthreads();

    for (std::uint32_t stride =
             kRecurrentThreads / 2U;
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

    const float inverse =
        rsqrtf(
            scratch[0] +
            epsilon);
    if (graph_status != nullptr && (!isfinite(scratch[0]) || !isfinite(inverse))) {
        if (threadIdx.x == 0) atomicOr(graph_status, 8U);
        return;
    }
    const float q_scale =
        is_key
            ? 1.0F
            : rsqrtf(
                  static_cast<float>(
                      head_dim));
    qkv[
        base +
        threadIdx.x] =
        value *
        inverse *
        q_scale;
}

__global__ __launch_bounds__(kRecurrentThreads)
void recurrent_step_kernel(
    const float* qkv,
    const float* beta,
    const float* gate,
    float* state,
    float* core,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t head_dim, std::uint32_t* graph_status) {
    __shared__ unsigned graph_reject;
    if (graph_status != nullptr) {
        if (threadIdx.x == 0) graph_reject = atomicAdd(graph_status, 0U) != 0;
        __syncthreads();
        if (graph_reject) return;
    }
    const auto tile_count =
        head_dim /
        kValueTile;
    const auto block =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto value_head =
        block /
        tile_count;
    const auto tile =
        block %
        tile_count;
    const auto lane =
        static_cast<std::uint32_t>(
            threadIdx.x);

    if (value_head >= value_heads ||
        lane >= head_dim) {
        return;
    }

    const auto key_head =
        value_head %
        key_heads;
    const auto value_base =
        tile *
        kValueTile;

    const auto q_base =
        static_cast<std::uint64_t>(
            key_head) *
        head_dim;
    const auto k_base =
        static_cast<std::uint64_t>(
            key_heads) *
            head_dim +
        q_base;
    const auto v_base =
        static_cast<std::uint64_t>(
            key_heads) *
            head_dim *
            2ULL +
        static_cast<std::uint64_t>(
            value_head) *
            head_dim;

    __shared__ float reduction[
        kRecurrentThreads *
        kValueTile];
    // The first reduction's broadcast must remain immutable while every wave
    // reads it. Reusing reduction[column] for the next tree lets lane zero
    // overwrite a delta before another wave has consumed it.
    __shared__ float deltas[kValueTile];

    float retained[kValueTile];
#pragma unroll
    for (std::uint32_t column = 0;
         column < kValueTile;
         ++column) {
        const auto value_index =
            value_base +
            column;
        const auto state_index =
            (static_cast<std::uint64_t>(
                 value_head) *
                 head_dim +
             lane) *
                head_dim +
            value_index;
        retained[column] =
            state[state_index] *
            expf(
                gate[value_head]);
        reduction[
            static_cast<std::size_t>(
                lane) *
                kValueTile +
            column] =
            retained[column] *
            qkv[
                k_base +
                lane];
    }
    __syncthreads();

    for (std::uint32_t stride =
             kRecurrentThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
#pragma unroll
            for (std::uint32_t column = 0;
                 column < kValueTile;
                 ++column) {
                reduction[
                    static_cast<std::size_t>(
                        lane) *
                        kValueTile +
                    column] +=
                    reduction[
                        static_cast<std::size_t>(
                            lane + stride) *
                            kValueTile +
                        column];
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
#pragma unroll
        for (std::uint32_t column = 0;
             column < kValueTile;
             ++column) {
            const auto value_index =
                value_base +
                column;
            deltas[column] =
                (qkv[
                     v_base +
                     value_index] -
                 reduction[column]) *
                beta[value_head];
        }
    }
    __syncthreads();

#pragma unroll
    for (std::uint32_t column = 0;
         column < kValueTile;
         ++column) {
        const auto delta =
            deltas[column];
        retained[column] +=
            qkv[
                k_base +
                lane] *
            delta;
        reduction[
            static_cast<std::size_t>(
                lane) *
                kValueTile +
            column] =
            retained[column] *
            qkv[
                q_base +
                lane];
    }
    __syncthreads();

    for (std::uint32_t stride =
             kRecurrentThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
#pragma unroll
            for (std::uint32_t column = 0;
                 column < kValueTile;
                 ++column) {
                reduction[
                    static_cast<std::size_t>(
                        lane) *
                        kValueTile +
                    column] +=
                    reduction[
                        static_cast<std::size_t>(
                            lane + stride) *
                            kValueTile +
                        column];
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
#pragma unroll
        for (std::uint32_t column = 0;
             column < kValueTile;
             ++column) {
            core[
                static_cast<std::uint64_t>(
                    value_head) *
                    head_dim +
                value_base +
                column] =
                reduction[column];
        }
    }

#pragma unroll
    for (std::uint32_t column = 0;
         column < kValueTile;
         ++column) {
        const auto value_index =
            value_base +
            column;
        const auto state_index =
            (static_cast<std::uint64_t>(
                 value_head) *
                 head_dim +
             lane) *
                head_dim +
            value_index;
        state[state_index] =
            retained[column];
    }
}

#ifdef STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC
template <unsigned Rows> struct GdnRecurrentWindowPointers {
    const float* qkv[Rows];
    const float* beta[Rows];
    const float* gate[Rows];
    float* core[Rows];
};
template <unsigned Rows>
__global__ __launch_bounds__(kRecurrentThreads)
void recurrent_retained_window_kernel(GdnRecurrentWindowPointers<Rows> rows, float* state,
    std::uint32_t key_heads, std::uint32_t value_heads, std::uint32_t head_dim) {
    static_assert(Rows == 2U || Rows == 4U);
    const auto tile_count = head_dim / kValueTile;
    const auto value_head = static_cast<std::uint32_t>(blockIdx.x) / tile_count;
    const auto tile = static_cast<std::uint32_t>(blockIdx.x) % tile_count;
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    if (value_head >= value_heads || lane >= head_dim) return;
    const auto key_head = value_head % key_heads;
    const auto value_base = tile * kValueTile;
    const auto q_base = static_cast<std::uint64_t>(key_head) * head_dim;
    const auto k_base = static_cast<std::uint64_t>(key_heads) * head_dim + q_base;
    const auto v_base = static_cast<std::uint64_t>(key_heads) * head_dim * 2ULL +
        static_cast<std::uint64_t>(value_head) * head_dim;
    __shared__ float reduction[kRecurrentThreads * kValueTile];
    __shared__ float deltas[kValueTile];
    float retained[kValueTile];
#pragma unroll
    for (std::uint32_t column = 0; column < kValueTile; ++column) {
        const auto state_index = (static_cast<std::uint64_t>(value_head) * head_dim + lane) * head_dim + value_base + column;
        retained[column] = state[state_index];
    }
    for (unsigned row = 0; row < Rows; ++row) {
        const auto* qkv = rows.qkv[row];
        const auto* beta = rows.beta[row];
        const auto* gate = rows.gate[row];
#pragma unroll
        for (std::uint32_t column = 0; column < kValueTile; ++column) {
            retained[column] = retained[column] * expf(gate[value_head]);
            reduction[static_cast<std::size_t>(lane) * kValueTile + column] = retained[column] * qkv[k_base + lane];
        }
        __syncthreads();
        for (std::uint32_t stride = kRecurrentThreads / 2U; stride > 0; stride >>= 1U) {
            if (lane < stride) {
#pragma unroll
                for (std::uint32_t column = 0; column < kValueTile; ++column)
                    reduction[static_cast<std::size_t>(lane) * kValueTile + column] +=
                        reduction[static_cast<std::size_t>(lane + stride) * kValueTile + column];
            }
            __syncthreads();
        }
        if (lane == 0) {
#pragma unroll
            for (std::uint32_t column = 0; column < kValueTile; ++column)
                deltas[column] = (qkv[v_base + value_base + column] - reduction[column]) * beta[value_head];
        }
        __syncthreads();
#pragma unroll
        for (std::uint32_t column = 0; column < kValueTile; ++column) {
            const auto delta = deltas[column];
            retained[column] += qkv[k_base + lane] * delta;
            reduction[static_cast<std::size_t>(lane) * kValueTile + column] = retained[column] * qkv[q_base + lane];
        }
        __syncthreads();
        for (std::uint32_t stride = kRecurrentThreads / 2U; stride > 0; stride >>= 1U) {
            if (lane < stride) {
#pragma unroll
                for (std::uint32_t column = 0; column < kValueTile; ++column)
                    reduction[static_cast<std::size_t>(lane) * kValueTile + column] +=
                        reduction[static_cast<std::size_t>(lane + stride) * kValueTile + column];
            }
            __syncthreads();
        }
        if (lane == 0) {
#pragma unroll
            for (std::uint32_t column = 0; column < kValueTile; ++column)
                rows.core[row][static_cast<std::uint64_t>(value_head) * head_dim + value_base + column] = reduction[column];
        }
        // Replaces the incumbent kernel boundary: lane zero must finish reading
        // the output tree before another wave starts overwriting its LDS.
        __syncthreads();
    }
#pragma unroll
    for (std::uint32_t column = 0; column < kValueTile; ++column) {
        const auto state_index = (static_cast<std::uint64_t>(value_head) * head_dim + lane) * head_dim + value_base + column;
        state[state_index] = retained[column];
    }
}
#endif

// Only physical LDS addresses differ from the incumbent recurrence. Logical
// columns, arithmetic expressions, barriers and separate delta lifetime remain.
// These same instantiations serve explicit eager choices and diagnostic probes.
template <gdn_recurrent_layout_diagnostic::Layout Layout>
__device__ __forceinline__ constexpr std::size_t recurrent_layout_index(
    std::uint32_t lane, std::uint32_t column) {
    if constexpr (Layout == gdn_recurrent_layout_diagnostic::Layout::transposed)
        return static_cast<std::size_t>(column) * kRecurrentThreads + lane;
    else
        return static_cast<std::size_t>(lane) *
            (Layout == gdn_recurrent_layout_diagnostic::Layout::stride33 ? 33U : kValueTile) + column;
}

template <gdn_recurrent_layout_diagnostic::Layout Layout>
__global__ __launch_bounds__(kRecurrentThreads)
void recurrent_layout_diagnostic_kernel(
    const float* qkv,
    const float* beta,
    const float* gate,
    float* state,
    float* core,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t head_dim, std::uint32_t* graph_status) {
    __shared__ unsigned graph_reject;
    if (graph_status != nullptr) {
        if (threadIdx.x == 0) graph_reject = atomicAdd(graph_status, 0U) != 0;
        __syncthreads();
        if (graph_reject) return;
    }
    const auto tile_count =
        head_dim /
        kValueTile;
    const auto block =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto value_head =
        block /
        tile_count;
    const auto tile =
        block %
        tile_count;
    const auto lane =
        static_cast<std::uint32_t>(
            threadIdx.x);

    if (value_head >= value_heads ||
        lane >= head_dim) {
        return;
    }

    const auto key_head =
        value_head %
        key_heads;
    const auto value_base =
        tile *
        kValueTile;

    const auto q_base =
        static_cast<std::uint64_t>(
            key_head) *
        head_dim;
    const auto k_base =
        static_cast<std::uint64_t>(
            key_heads) *
            head_dim +
        q_base;
    const auto v_base =
        static_cast<std::uint64_t>(
            key_heads) *
            head_dim *
            2ULL +
        static_cast<std::uint64_t>(
            value_head) *
            head_dim;

    __shared__ float reduction[
        kRecurrentThreads *
        (Layout == gdn_recurrent_layout_diagnostic::Layout::stride33 ? 33U : kValueTile)];
    // The first reduction's broadcast must remain immutable while every wave
    // reads it. Reusing reduction[column] for the next tree lets lane zero
    // overwrite a delta before another wave has consumed it.
    __shared__ float deltas[kValueTile];

    float retained[kValueTile];
#pragma unroll
    for (std::uint32_t column = 0;
         column < kValueTile;
         ++column) {
        const auto value_index =
            value_base +
            column;
        const auto state_index =
            (static_cast<std::uint64_t>(
                 value_head) *
                 head_dim +
             lane) *
                head_dim +
            value_index;
        retained[column] =
            state[state_index] *
            expf(
                gate[value_head]);
        reduction[recurrent_layout_index<Layout>(lane, column)] =
            retained[column] *
            qkv[
                k_base +
                lane];
    }
    __syncthreads();

    for (std::uint32_t stride =
             kRecurrentThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
#pragma unroll
            for (std::uint32_t column = 0;
                 column < kValueTile;
                 ++column) {
                reduction[recurrent_layout_index<Layout>(lane, column)] +=
                    reduction[recurrent_layout_index<Layout>(lane + stride, column)];
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
#pragma unroll
        for (std::uint32_t column = 0;
             column < kValueTile;
             ++column) {
            const auto value_index =
                value_base +
                column;
            deltas[column] =
                (qkv[
                     v_base +
                     value_index] -
                 reduction[recurrent_layout_index<Layout>(0U, column)]) *
                beta[value_head];
        }
    }
    __syncthreads();

#pragma unroll
    for (std::uint32_t column = 0;
         column < kValueTile;
         ++column) {
        const auto delta =
            deltas[column];
        retained[column] +=
            qkv[
                k_base +
                lane] *
            delta;
        reduction[recurrent_layout_index<Layout>(lane, column)] =
            retained[column] *
            qkv[
                q_base +
                lane];
    }
    __syncthreads();

    for (std::uint32_t stride =
             kRecurrentThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
#pragma unroll
            for (std::uint32_t column = 0;
                 column < kValueTile;
                 ++column) {
                reduction[recurrent_layout_index<Layout>(lane, column)] +=
                    reduction[recurrent_layout_index<Layout>(lane + stride, column)];
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
#pragma unroll
        for (std::uint32_t column = 0;
             column < kValueTile;
             ++column) {
            core[
                static_cast<std::uint64_t>(
                    value_head) *
                    head_dim +
                value_base +
                column] =
                reduction[recurrent_layout_index<Layout>(0U, column)];
        }
    }

#pragma unroll
    for (std::uint32_t column = 0;
         column < kValueTile;
         ++column) {
        const auto value_index =
            value_base +
            column;
        const auto state_index =
            (static_cast<std::uint64_t>(
                 value_head) *
                 head_dim +
             lane) *
                head_dim +
            value_index;
        state[state_index] =
            retained[column];
    }
}

template <gdn_recurrent_layout_diagnostic::Layout Layout,
    GdnRecurrentStorage Storage = GdnRecurrentStorage::key_value>
__global__ __launch_bounds__(kRecurrentThreads)
void recurrent_storage_kernel(
    const float* qkv,
    const float* beta,
    const float* gate,
    float* state,
    float* core,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t head_dim, std::uint32_t* graph_status) {
    __shared__ unsigned graph_reject;
    if (graph_status != nullptr) {
        if (threadIdx.x == 0) graph_reject = atomicAdd(graph_status, 0U) != 0;
        __syncthreads();
        if (graph_reject) return;
    }
    const auto tile_count =
        head_dim /
        kValueTile;
    const auto block =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto value_head =
        block /
        tile_count;
    const auto tile =
        block %
        tile_count;
    const auto lane =
        static_cast<std::uint32_t>(
            threadIdx.x);

    if (value_head >= value_heads ||
        lane >= head_dim) {
        return;
    }

    const auto key_head =
        value_head %
        key_heads;
    const auto value_base =
        tile *
        kValueTile;

    const auto q_base =
        static_cast<std::uint64_t>(
            key_head) *
        head_dim;
    const auto k_base =
        static_cast<std::uint64_t>(
            key_heads) *
            head_dim +
        q_base;
    const auto v_base =
        static_cast<std::uint64_t>(
            key_heads) *
            head_dim *
            2ULL +
        static_cast<std::uint64_t>(
            value_head) *
            head_dim;

    __shared__ float reduction[
        kRecurrentThreads *
        (Layout == gdn_recurrent_layout_diagnostic::Layout::stride33 ? 33U : kValueTile)];
    // The first reduction's broadcast must remain immutable while every wave
    // reads it. Reusing reduction[column] for the next tree lets lane zero
    // overwrite a delta before another wave has consumed it.
    __shared__ float deltas[kValueTile];

    float retained[kValueTile];
#pragma unroll
    for (std::uint32_t column = 0;
         column < kValueTile;
         ++column) {
        const auto value_index =
            value_base +
            column;
        const auto state_index =
            (static_cast<std::uint64_t>(
                 value_head) *
                 head_dim +
             (Storage == GdnRecurrentStorage::value_key ? value_index : lane)) *
                head_dim +
            (Storage == GdnRecurrentStorage::value_key ? lane : value_index);
        retained[column] =
            state[state_index] *
            expf(
                gate[value_head]);
        reduction[recurrent_layout_index<Layout>(lane, column)] =
            retained[column] *
            qkv[
                k_base +
                lane];
    }
    __syncthreads();

    for (std::uint32_t stride =
             kRecurrentThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
#pragma unroll
            for (std::uint32_t column = 0;
                 column < kValueTile;
                 ++column) {
                reduction[recurrent_layout_index<Layout>(lane, column)] +=
                    reduction[recurrent_layout_index<Layout>(lane + stride, column)];
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
#pragma unroll
        for (std::uint32_t column = 0;
             column < kValueTile;
             ++column) {
            const auto value_index =
                value_base +
                column;
            deltas[column] =
                (qkv[
                     v_base +
                     value_index] -
                 reduction[recurrent_layout_index<Layout>(0U, column)]) *
                beta[value_head];
        }
    }
    __syncthreads();

#pragma unroll
    for (std::uint32_t column = 0;
         column < kValueTile;
         ++column) {
        const auto delta =
            deltas[column];
        retained[column] +=
            qkv[
                k_base +
                lane] *
            delta;
        reduction[recurrent_layout_index<Layout>(lane, column)] =
            retained[column] *
            qkv[
                q_base +
                lane];
    }
    __syncthreads();

    for (std::uint32_t stride =
             kRecurrentThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
#pragma unroll
            for (std::uint32_t column = 0;
                 column < kValueTile;
                 ++column) {
                reduction[recurrent_layout_index<Layout>(lane, column)] +=
                    reduction[recurrent_layout_index<Layout>(lane + stride, column)];
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
#pragma unroll
        for (std::uint32_t column = 0;
             column < kValueTile;
             ++column) {
            core[
                static_cast<std::uint64_t>(
                    value_head) *
                    head_dim +
                value_base +
                column] =
                reduction[recurrent_layout_index<Layout>(0U, column)];
        }
    }

#pragma unroll
    for (std::uint32_t column = 0;
         column < kValueTile;
         ++column) {
        const auto value_index =
            value_base +
            column;
        const auto state_index =
            (static_cast<std::uint64_t>(
                 value_head) *
                 head_dim +
             (Storage == GdnRecurrentStorage::value_key ? value_index : lane)) *
                head_dim +
            (Storage == GdnRecurrentStorage::value_key ? lane : value_index);
        state[state_index] =
            retained[column];
    }
}

__global__ void gated_rms_norm_kernel(
    float* core,
    const float* z,
    const void* norm_weights,
    SmallWeightKind norm_kind,
    std::uint32_t heads,
    std::uint32_t head_dim,
    float epsilon, std::uint32_t* graph_status) {
    __shared__ unsigned graph_reject;
    if (graph_status != nullptr) {
        if (threadIdx.x == 0) graph_reject = atomicAdd(graph_status, 0U) != 0;
        __syncthreads();
        if (graph_reject) return;
    }
    const auto head =
        static_cast<std::uint32_t>(
            blockIdx.x);
    if (head >= heads ||
        threadIdx.x >= head_dim) {
        return;
    }

    __shared__ float scratch[kRecurrentThreads];
    const auto index =
        static_cast<std::uint64_t>(
            head) *
            head_dim +
        threadIdx.x;
    const float value =
        core[index];
    scratch[threadIdx.x] =
        value * value;
    __syncthreads();

    for (std::uint32_t stride =
             kRecurrentThreads / 2U;
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

    const float inv =
        rsqrtf(
            scratch[0] /
                static_cast<float>(
                    head_dim) +
            epsilon);
    if (graph_status != nullptr && (!isfinite(scratch[0]) || !isfinite(inv))) {
        if (threadIdx.x == 0) atomicOr(graph_status, 8U);
        return;
    }
    const float gate_value =
        1.0F /
        (1.0F +
         expf(-z[index]));
    core[index] =
        value *
        inv *
        load_small(
            norm_weights,
            norm_kind,
            threadIdx.x) *
        gate_value;
}

// Deferred-commit window. Every arithmetic expression and reduction tree below
// repeats the incumbent R1 kernels; only work placement and LDS addresses differ.
// Slot `pending_slot` holds rows saved by the previous window, the other slot
// receives this window's rows (a window never commits its own rows).
constexpr std::uint32_t kGdnWindowRows = gdn_window_max_rows;
constexpr std::uint32_t kGdnWindowKernel = 4;
// Plain IEEE add hidden from contraction; not volatile, so independent trees interleave.
__device__ __forceinline__ float gdn_window_add(float lower, float upper) {
    float result;
    asm("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(lower), "v"(upper));
    return result;
}
template <unsigned Rows> struct GdnWindowRows {
    float* qkv[Rows]; float* z[Rows]; float* beta[Rows]; float* gate[Rows]; float* core[Rows];
};
struct GdnWindowSaved { std::uintptr_t base; std::uint64_t row_bytes, k, v, beta, gate; };
template <class T>
__device__ __forceinline__ T* gdn_saved(const GdnWindowSaved& saved, std::uint32_t slot, std::uint32_t row,
    std::uint64_t offset) {
    return reinterpret_cast<T*>(saved.base + (static_cast<std::uint64_t>(slot) * kGdnWindowRows + row) *
        saved.row_bytes + offset);
}
__device__ __forceinline__ std::uint32_t gdn_window_pending(const GdnWindowControl* control) {
    return min(control->pending_rows, kGdnWindowRows);
}
// The incumbent 128-leaf tree: strides 64 and 32 through LDS, 16..1 by shuffles.
__device__ __forceinline__ float gdn_window_sum128(float* scratch, std::uint32_t lane, float value) {
    scratch[lane] = value;
    __syncthreads();
    if (lane < 64U) scratch[lane] = gdn_window_add(scratch[lane], scratch[lane + 64U]);
    __syncthreads();
    if (lane < 32U) {
        float sum = gdn_window_add(scratch[lane], scratch[lane + 32U]);
#pragma unroll
        for (unsigned distance = 16U; distance; distance /= 2U)
            sum = gdn_window_add(sum, __shfl_down(sum, distance, 32));
        if (lane == 0U) scratch[0] = sum;
    }
    __syncthreads();
    const float total = scratch[0];
    __syncthreads();
    return total;
}
__device__ __forceinline__ void gdn_window_shift(std::uint16_t (&window)[kGdnWindowKernel], std::uint16_t value) {
#pragma unroll
    for (std::uint32_t index = 0; index + 1U < kGdnWindowKernel; ++index) window[index] = window[index + 1U];
    window[kGdnWindowKernel - 1U] = value;
}
// Conv history commit of the pending rows, then conv/SiLU, q/k L2 norm and
// beta/gate transforms of every window row. Blocks own one 128-channel head;
// the trailing block owns the per-head beta/gate scalars.
template <unsigned Rows, bool ParallelGates = false>
__device__ __forceinline__ void gdn_window_prepare_rows(const GdnWindowRows<Rows>& rows, std::uint16_t* conv_state, const void* conv_weights,
    SmallWeightKind conv_kind, const void* dt, SmallWeightKind dt_kind, const void* a, SmallWeightKind a_kind,
    const GdnWindowSaved& saved, const GdnWindowControl* control, std::uint32_t key_heads, std::uint32_t value_heads,
    std::uint32_t head_dim, float epsilon, float* scratch) {
    const auto lane = static_cast<std::uint32_t>(threadIdx.x), block = static_cast<std::uint32_t>(blockIdx.x);
    const auto pending = gdn_window_pending(control), slot = control->pending_slot & 1U, next = slot ^ 1U;
    const auto key_width = key_heads * head_dim;
    if (block >= key_heads * 2U + value_heads) {
        if constexpr (ParallelGates) {
            // One (head, row) pair per thread instead of a serial row loop per head; same per-element arithmetic.
            for (auto item = lane; item < value_heads * Rows; item += kRecurrentThreads) {
                const auto head = item % value_heads, row = item / value_heads;
                float* beta = rows.beta[row]; float* gate = rows.gate[row];
                beta[head] = round_bf16(1.0F / (1.0F + expf(-beta[head])));
                const float biased = gate[head] + load_small(dt, dt_kind, head);
                gate[head] = load_small(a, a_kind, head) * softplus(biased);
                gdn_saved<float>(saved, next, row, saved.beta)[head] = beta[head];
                gdn_saved<float>(saved, next, row, saved.gate)[head] = gate[head];
            }
            return;
        }
        for (auto head = lane; head < value_heads; head += kRecurrentThreads) {
            for (unsigned row = 0; row < Rows; ++row) {
                float* beta = rows.beta[row]; float* gate = rows.gate[row];
                beta[head] = round_bf16(1.0F / (1.0F + expf(-beta[head])));
                const float biased = gate[head] + load_small(dt, dt_kind, head);
                gate[head] = load_small(a, a_kind, head) * softplus(biased);
                gdn_saved<float>(saved, next, row, saved.beta)[head] = beta[head];
                gdn_saved<float>(saved, next, row, saved.gate)[head] = gate[head];
            }
        }
        return;
    }
    const auto channel = block * head_dim + lane;
    auto* history = conv_state + static_cast<std::uint64_t>(channel) * kGdnWindowKernel;
    std::uint16_t window[kGdnWindowKernel];
#pragma unroll
    for (std::uint32_t index = 0; index < kGdnWindowKernel; ++index) window[index] = history[index];
    for (std::uint32_t row = 0; row < pending; ++row)
        gdn_window_shift(window, gdn_saved<const std::uint16_t>(saved, slot, row, 0)[channel]);
    if (pending) {
#pragma unroll
        for (std::uint32_t index = 0; index < kGdnWindowKernel; ++index) history[index] = window[index];
    }
    const bool is_qk = block < key_heads * 2U, is_key = is_qk && block >= key_heads;
    for (unsigned row = 0; row < Rows; ++row) {
        float* qkv = rows.qkv[row];
        gdn_window_shift(window, float_to_bf16(qkv[channel]));
        gdn_saved<std::uint16_t>(saved, next, row, 0)[channel] = window[kGdnWindowKernel - 1U];
        float sum = 0.0F;
        for (std::uint32_t index = 0; index < kGdnWindowKernel; ++index)
            sum += bf16_to_float(window[index]) *
                load_small(conv_weights, conv_kind, static_cast<std::uint64_t>(channel) * kGdnWindowKernel + index);
        const float silu = sum / (1.0F + expf(-sum));
        float value = round_bf16(silu);
        if (is_qk) {
            const float inverse = rsqrtf(gdn_window_sum128(scratch, lane, value * value) + epsilon);
            const float q_scale = is_key ? 1.0F : rsqrtf(static_cast<float>(head_dim));
            value = value * inverse * q_scale;
            if (is_key) gdn_saved<float>(saved, next, row, saved.k)[channel - key_width] = value;
        } else gdn_saved<float>(saved, next, row, saved.v)[channel - key_width * 2U] = value;
        qkv[channel] = value;
    }
}
template <unsigned Rows, bool ParallelGates = false>
__global__ __launch_bounds__(kRecurrentThreads)
void gdn_window_prepare_kernel(GdnWindowRows<Rows> rows, std::uint16_t* conv_state, const void* conv_weights,
    SmallWeightKind conv_kind, const void* dt, SmallWeightKind dt_kind, const void* a, SmallWeightKind a_kind,
    GdnWindowSaved saved, const GdnWindowControl* control, std::uint32_t key_heads, std::uint32_t value_heads,
    std::uint32_t head_dim, float epsilon) {
    __shared__ float scratch[kRecurrentThreads];
    gdn_window_prepare_rows<Rows, ParallelGates>(rows, conv_state, conv_weights, conv_kind, dt, dt_kind, a, a_kind, saved,
        control, key_heads, value_heads, head_dim, epsilon, scratch);
}
// Cohorts: the prepare / recurrence of every session's segment (2, 4 or 8 rows; its own state, history,
// retained rows and control) in one launch, blockIdx.y = segment; each segment runs the per-session kernel's body.
struct GdnSegment {
    GdnWindowRows<8> rows;
    std::uint32_t count;
    std::uint16_t* conv_state;
    float* matrix;
    GdnWindowSaved saved;
    const GdnWindowControl* control;
};
struct GdnSegments { GdnSegment segment[8]; };
template <unsigned Rows> __device__ __forceinline__ GdnWindowRows<Rows> gdn_rows_prefix(const GdnWindowRows<8>& all) {
    GdnWindowRows<Rows> out;
#pragma unroll
    for (unsigned r = 0; r < Rows; ++r) {
        out.qkv[r] = all.qkv[r]; out.z[r] = all.z[r]; out.beta[r] = all.beta[r]; out.gate[r] = all.gate[r]; out.core[r] = all.core[r];
    }
    return out;
}
template <bool ParallelGates>
__global__ __launch_bounds__(kRecurrentThreads)
void gdn_window_prepare_segments_kernel(GdnSegments segments, const void* conv_weights, SmallWeightKind conv_kind,
    const void* dt, SmallWeightKind dt_kind, const void* a, SmallWeightKind a_kind, std::uint32_t key_heads,
    std::uint32_t value_heads, std::uint32_t head_dim, float epsilon) {
    __shared__ float scratch[kRecurrentThreads];
    const auto& s = segments.segment[blockIdx.y];
    switch (s.count) {
    case 2: gdn_window_prepare_rows<2, ParallelGates>(gdn_rows_prefix<2>(s.rows), s.conv_state, conv_weights, conv_kind, dt, dt_kind,
        a, a_kind, s.saved, s.control, key_heads, value_heads, head_dim, epsilon, scratch); break;
    case 4: gdn_window_prepare_rows<4, ParallelGates>(gdn_rows_prefix<4>(s.rows), s.conv_state, conv_weights, conv_kind, dt, dt_kind,
        a, a_kind, s.saved, s.control, key_heads, value_heads, head_dim, epsilon, scratch); break;
    default: gdn_window_prepare_rows<8, ParallelGates>(gdn_rows_prefix<8>(s.rows), s.conv_state, conv_weights, conv_kind, dt, dt_kind,
        a, a_kind, s.saved, s.control, key_heads, value_heads, head_dim, epsilon, scratch); break;
    }
}
template <GdnRecurrentStorage Storage>
__device__ __forceinline__ std::uint64_t gdn_window_state_index(std::uint32_t value_head, std::uint32_t lane,
    std::uint32_t value_index, std::uint32_t head_dim) {
    return (static_cast<std::uint64_t>(value_head) * head_dim +
        (Storage == GdnRecurrentStorage::value_key ? value_index : lane)) * head_dim +
        (Storage == GdnRecurrentStorage::value_key ? lane : value_index);
}
// One wave per (value head, 8-column tile). Lane l holds keys l, l+32, l+64,
// l+96, so the incumbent tree's stride-64 and stride-32 levels are in-lane,
// (p[l] + p[l+64]) + (p[l+32] + p[l+96]), and strides 16..1 are shuffles.
// Lane 0 holds the root; it is broadcast so every lane derives the same delta.
constexpr std::uint32_t kGdnWindowColumns = 8;
// Lane l < d receives lane l + d: permlanex16 across the two rows, then DPP row_shl.
template <unsigned Distance>
__device__ __forceinline__ float gdn_window_down(float value) {
    if constexpr (Distance == 16U) {
        const auto bits = __float_as_uint(value);
        return __uint_as_float(__builtin_amdgcn_permlanex16(bits, bits, 0x76543210U, 0xfedcba98U, false, false));
    } else {
        const auto bits = __float_as_int(value);
        return __int_as_float(__builtin_amdgcn_update_dpp(bits, bits, 0x100 + static_cast<int>(Distance), 0xf, 0xf, false));
    }
}
__device__ __forceinline__ float gdn_window_key_sum(const float (&p)[4]) {
    float sum = gdn_window_add(gdn_window_add(p[0], p[2]), gdn_window_add(p[1], p[3]));
    sum = gdn_window_add(sum, gdn_window_down<16U>(sum));
    sum = gdn_window_add(sum, gdn_window_down<8U>(sum));
    sum = gdn_window_add(sum, gdn_window_down<4U>(sum));
    sum = gdn_window_add(sum, gdn_window_down<2U>(sum));
    sum = gdn_window_add(sum, gdn_window_down<1U>(sum));
    return __int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(sum)));
}
template <std::uint32_t Columns> struct GdnWindowInputsT { float key[4], query[4], value[Columns], beta, gate; };
using GdnWindowInputs = GdnWindowInputsT<kGdnWindowColumns>;
// Columns: value columns per wave (8, or 4/2 with STRIX_GDN_WINDOW_COLUMNS: more waves, shorter per-wave chains;
// every column's arithmetic is the same).
template <unsigned Distance, unsigned Count>
__device__ __forceinline__ void gdn_window_scatter(float* sums, unsigned lane) {
#pragma unroll
    for (unsigned c = 0; c < Count / 2U; ++c) {
        const bool upper = (lane & Distance) != 0U;
        const float local = sums[2U * c + (upper ? 1U : 0U)];
        const float peer = __shfl_xor(sums[2U * c + (upper ? 0U : 1U)], Distance);
        // Always add the lower-key subtree first, including in upper lanes.
        sums[c] = upper ? gdn_window_add(peer, local) : gdn_window_add(local, peer);
    }
}
template <unsigned Columns>
__device__ __forceinline__ void gdn_window_sums(float (&sums)[Columns]) {
    const auto lane = static_cast<unsigned>(threadIdx.x) & 31U;
    static_assert(Columns == 4U || Columns == 8U);
    gdn_window_scatter<16U, Columns>(sums, lane);
    gdn_window_scatter<8U, Columns / 2U>(sums, lane);
    if constexpr (Columns == 8U) gdn_window_scatter<4U, 2U>(sums, lane);
    float sum = sums[0];
    if constexpr (Columns == 4U) sum = gdn_window_add(sum, gdn_window_down<4U>(sum));
    sum = gdn_window_add(sum, gdn_window_down<2U>(sum));
    sum = gdn_window_add(sum, gdn_window_down<1U>(sum));
#pragma unroll
    for (unsigned c = 0; c < Columns; ++c) {
        const auto root = (c & 1U) * 16U + (c & 2U) * 4U + (c & 4U);
        sums[c] = __shfl(sum, static_cast<int>(root));
    }
}

template <bool Output, std::uint32_t Columns = kGdnWindowColumns, bool Scatter = false>
__device__ __forceinline__ void gdn_window_step(float (&retained)[4][Columns], const GdnWindowInputsT<Columns>& in,
    float* core) {
    float deltas[Columns];
    const float decay = expf(in.gate);
#pragma unroll
    for (std::uint32_t column = 0; column < Columns; ++column) {
        float products[4];
#pragma unroll
        for (std::uint32_t part = 0; part < 4U; ++part) {
            retained[part][column] = retained[part][column] * decay;
            products[part] = retained[part][column] * in.key[part];
        }
        if constexpr (Scatter)
            deltas[column] = gdn_window_add(gdn_window_add(products[0], products[2]), gdn_window_add(products[1], products[3]));
        else deltas[column] = (in.value[column] - gdn_window_key_sum(products)) * in.beta;
    }
    if constexpr (Scatter) {
        gdn_window_sums(deltas);
#pragma unroll
        for (unsigned c = 0; c < Columns; ++c) deltas[c] = (in.value[c] - deltas[c]) * in.beta;
    }
    float output[Columns];
#pragma unroll
    for (std::uint32_t column = 0; column < Columns; ++column) {
        float products[4];
#pragma unroll
        for (std::uint32_t part = 0; part < 4U; ++part) {
            retained[part][column] += in.key[part] * deltas[column];
            products[part] = retained[part][column] * in.query[part];
        }
        if constexpr (Output) {
            if constexpr (Scatter)
                output[column] = gdn_window_add(gdn_window_add(products[0], products[2]), gdn_window_add(products[1], products[3]));
            else {
                const float sum = gdn_window_key_sum(products);
                if ((threadIdx.x & 31U) == 0U) core[column] = sum;
            }
        }
    }
    if constexpr (Output && Scatter) {
        gdn_window_sums(output);
        if ((threadIdx.x & 31U) == 0U)
#pragma unroll
            for (unsigned c = 0; c < Columns; ++c) core[c] = output[c];
    }
}
// Read the tile once, apply the pending rows and write it once (if any), then
// run the window rows in registers. Row inputs are loaded ahead of their step.
template <unsigned Rows, GdnRecurrentStorage Storage, std::uint32_t Columns = kGdnWindowColumns, bool Scatter = false,
    bool WholeHead = false>
__device__ __forceinline__ void gdn_window_recurrence(const GdnWindowRows<Rows ? Rows : 1U>& rows, float* state,
    const GdnWindowSaved& saved, const GdnWindowControl* control, std::uint32_t key_heads, std::uint32_t head_dim) {
    const auto lane = WholeHead ? static_cast<std::uint32_t>(threadIdx.x) & 31U : static_cast<std::uint32_t>(threadIdx.x);
    const auto tiles = head_dim / Columns;
    const auto raw_tile = WholeHead ? static_cast<std::uint32_t>(blockIdx.x) * tiles + static_cast<std::uint32_t>(threadIdx.x) / 32U :
        static_cast<std::uint32_t>(blockIdx.x);
    // Each complete wave owns one tile. Make that wave-uniform address explicit
    // so its row values, beta and gate retain the incumbent scalar placement.
    const auto tile = WholeHead ? static_cast<std::uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<int>(raw_tile))) : raw_tile;
    const auto value_head = tile / tiles;
    const auto value_base = tile % tiles * Columns;
    const auto key_offset = value_head % key_heads * head_dim, key_width = key_heads * head_dim;
    const auto value_offset = static_cast<std::uint64_t>(value_head) * head_dim + value_base;
    const auto load = [&](const float* key, const float* query, const float* value, float beta, float gate) {
        GdnWindowInputsT<Columns> in{};
#pragma unroll
        for (std::uint32_t part = 0; part < 4U; ++part) {
            in.key[part] = key[part * 32U + lane];
            if (query) in.query[part] = query[part * 32U + lane];
        }
#pragma unroll
        for (std::uint32_t column = 0; column < Columns; ++column) in.value[column] = value[column];
        in.beta = beta; in.gate = gate;
        return in;
    };
    const auto pending_inputs = [&](std::uint32_t slot, std::uint32_t row) {
        return load(gdn_saved<const float>(saved, slot, row, saved.k) + key_offset, nullptr,
            gdn_saved<const float>(saved, slot, row, saved.v) + value_offset,
            gdn_saved<const float>(saved, slot, row, saved.beta)[value_head],
            gdn_saved<const float>(saved, slot, row, saved.gate)[value_head]);
    };
    const auto window_inputs = [&](unsigned row) {
        const float* qkv = rows.qkv[row];
        return load(qkv + key_width + key_offset, qkv + key_offset, qkv + key_width * 2U + value_offset,
            rows.beta[row][value_head], rows.gate[row][value_head]);
    };
    GdnWindowInputsT<Columns> window[Rows ? Rows : 1U];
    if constexpr (Rows != 0U && !Scatter) {
#pragma unroll
        for (unsigned row = 0; row < Rows; ++row) window[row] = window_inputs(row);
    }
    float retained[4][Columns];
#pragma unroll
    for (std::uint32_t part = 0; part < 4U; ++part)
#pragma unroll
        for (std::uint32_t column = 0; column < Columns; ++column)
            retained[part][column] = state[gdn_window_state_index<Storage>(value_head, part * 32U + lane, value_base + column, head_dim)];
    const auto pending = gdn_window_pending(control), slot = control->pending_slot & 1U;
    if (pending) {
        auto next = pending_inputs(slot, 0U);
        for (std::uint32_t row = 0; row < pending; ++row) {
            const auto in = next;
            if (row + 1U < pending) next = pending_inputs(slot, row + 1U);
            gdn_window_step<false, Columns, Scatter>(retained, in, nullptr);
        }
#pragma unroll
        for (std::uint32_t part = 0; part < 4U; ++part)
#pragma unroll
            for (std::uint32_t column = 0; column < Columns; ++column)
                state[gdn_window_state_index<Storage>(value_head, part * 32U + lane, value_base + column, head_dim)] =
                    retained[part][column];
    }
    if constexpr (Rows != 0U) {
        if constexpr (Scatter) {
            // Keep only the current row's inputs live next to the scatter
            // accumulators. Preloading all eight rows raises VGPR pressure.
#pragma unroll 1
            for (unsigned row = 0; row < Rows; ++row)
                gdn_window_step<true, Columns, true>(retained, window_inputs(row), rows.core[row] + value_offset);
        } else {
#pragma unroll
        for (unsigned row = 0; row < Rows; ++row) gdn_window_step<true, Columns, Scatter>(retained, window[row], rows.core[row] + value_offset);
        }
    }
}
// STRIX_GDN_WINDOW_TILED=1 (key-value storage): one block of 32/Columns waves per (value head, 32 columns). The block
// reads and writes its 128 x 32 state slice through LDS with 128-byte coalesced rows; each wave then runs Columns
// columns with the same lane/key mapping and the same per-column arithmetic as gdn_window_recurrence.
constexpr std::uint32_t kGdnTiledColumns = 32;
template <unsigned Rows, std::uint32_t Columns>
__global__ __launch_bounds__(32U * (kGdnTiledColumns / Columns))
void gdn_window_recurrent_tiled_kernel(GdnWindowRows<Rows> rows, float* state, GdnWindowSaved saved,
    const GdnWindowControl* control, std::uint32_t key_heads, std::uint32_t head_dim) {
    constexpr std::uint32_t threads = 32U * (kGdnTiledColumns / Columns);
    __shared__ float tile[128][kGdnTiledColumns + 1U];
    const auto lane = static_cast<std::uint32_t>(threadIdx.x) & 31U, wave = static_cast<std::uint32_t>(threadIdx.x) / 32U;
    const auto blocks = head_dim / kGdnTiledColumns;
    const auto value_head = static_cast<std::uint32_t>(blockIdx.x) / blocks;
    const auto block_base = static_cast<std::uint32_t>(blockIdx.x) % blocks * kGdnTiledColumns;
    const auto local = wave * Columns, value_base = block_base + local;
    const auto key_offset = value_head % key_heads * head_dim, key_width = key_heads * head_dim;
    const auto value_offset = static_cast<std::uint64_t>(value_head) * head_dim + value_base;
    const auto load = [&](const float* key, const float* query, const float* value, float beta, float gate) {
        GdnWindowInputsT<Columns> in{};
#pragma unroll
        for (std::uint32_t part = 0; part < 4U; ++part) {
            in.key[part] = key[part * 32U + lane];
            if (query) in.query[part] = query[part * 32U + lane];
        }
#pragma unroll
        for (std::uint32_t column = 0; column < Columns; ++column) in.value[column] = value[column];
        in.beta = beta; in.gate = gate;
        return in;
    };
    const auto pending_inputs = [&](std::uint32_t slot, std::uint32_t row) {
        return load(gdn_saved<const float>(saved, slot, row, saved.k) + key_offset, nullptr,
            gdn_saved<const float>(saved, slot, row, saved.v) + value_offset,
            gdn_saved<const float>(saved, slot, row, saved.beta)[value_head],
            gdn_saved<const float>(saved, slot, row, saved.gate)[value_head]);
    };
    GdnWindowInputsT<Columns> window[Rows];
#pragma unroll
    for (unsigned row = 0; row < Rows; ++row) {
        const float* qkv = rows.qkv[row];
        window[row] = load(qkv + key_width + key_offset, qkv + key_offset, qkv + key_width * 2U + value_offset,
            rows.beta[row][value_head], rows.gate[row][value_head]);
    }
    // Coalesced slice read: key k, 16-byte chunk c of the 32 columns (key-value storage: row k is contiguous).
    float* const slice = state + static_cast<std::uint64_t>(value_head) * head_dim * head_dim + block_base;
    for (std::uint32_t i = threadIdx.x; i < 128U * (kGdnTiledColumns / 4U); i += threads) {
        const auto key = i / (kGdnTiledColumns / 4U), chunk = i % (kGdnTiledColumns / 4U);
        const float4 v = *reinterpret_cast<const float4*>(slice + static_cast<std::uint64_t>(key) * head_dim + chunk * 4U);
        tile[key][chunk * 4U] = v.x; tile[key][chunk * 4U + 1U] = v.y; tile[key][chunk * 4U + 2U] = v.z; tile[key][chunk * 4U + 3U] = v.w;
    }
    __syncthreads();
    float retained[4][Columns];
#pragma unroll
    for (std::uint32_t part = 0; part < 4U; ++part)
#pragma unroll
        for (std::uint32_t column = 0; column < Columns; ++column) retained[part][column] = tile[part * 32U + lane][local + column];
    const auto pending = gdn_window_pending(control), slot = control->pending_slot & 1U;
    if (pending) {
        auto next = pending_inputs(slot, 0U);
        for (std::uint32_t row = 0; row < pending; ++row) {
            const auto in = next;
            if (row + 1U < pending) next = pending_inputs(slot, row + 1U);
            gdn_window_step<false, Columns>(retained, in, nullptr);
        }
        __syncthreads();
#pragma unroll
        for (std::uint32_t part = 0; part < 4U; ++part)
#pragma unroll
            for (std::uint32_t column = 0; column < Columns; ++column) tile[part * 32U + lane][local + column] = retained[part][column];
        __syncthreads();
        for (std::uint32_t i = threadIdx.x; i < 128U * (kGdnTiledColumns / 4U); i += threads) {
            const auto key = i / (kGdnTiledColumns / 4U), chunk = i % (kGdnTiledColumns / 4U);
            *reinterpret_cast<float4*>(slice + static_cast<std::uint64_t>(key) * head_dim + chunk * 4U) =
                float4{tile[key][chunk * 4U], tile[key][chunk * 4U + 1U], tile[key][chunk * 4U + 2U], tile[key][chunk * 4U + 3U]};
        }
    }
#pragma unroll
    for (unsigned row = 0; row < Rows; ++row) gdn_window_step<true, Columns>(retained, window[row], rows.core[row] + value_offset);
}
template <unsigned Rows, GdnRecurrentStorage Storage, std::uint32_t Columns = kGdnWindowColumns, bool Scatter = false>
__global__ __launch_bounds__(32)
void gdn_window_recurrent_kernel(GdnWindowRows<Rows> rows, float* state, GdnWindowSaved saved,
    const GdnWindowControl* control, std::uint32_t key_heads, std::uint32_t head_dim) {
    gdn_window_recurrence<Rows, Storage, Columns, Scatter>(rows, state, saved, control, key_heads, head_dim);
}
template <GdnRecurrentStorage Storage, std::uint32_t Columns>
__global__ __launch_bounds__(32)
void gdn_window_recurrent_segments_kernel(GdnSegments segments, std::uint32_t key_heads, std::uint32_t head_dim) {
    const auto& s = segments.segment[blockIdx.y];
    switch (s.count) {
    case 2: gdn_window_recurrence<2, Storage, Columns>(gdn_rows_prefix<2>(s.rows), s.matrix, s.saved, s.control, key_heads, head_dim); break;
    case 4: gdn_window_recurrence<4, Storage, Columns>(gdn_rows_prefix<4>(s.rows), s.matrix, s.saved, s.control, key_heads, head_dim); break;
    default: gdn_window_recurrence<8, Storage, Columns>(gdn_rows_prefix<8>(s.rows), s.matrix, s.saved, s.control, key_heads, head_dim); break;
    }
}
// Standalone commit of the pending rows: recurrence tiles, then 32-channel conv groups.
template <GdnRecurrentStorage Storage>
__global__ __launch_bounds__(32)
void gdn_window_commit_kernel(float* state, std::uint16_t* conv_state, GdnWindowSaved saved,
    const GdnWindowControl* control, std::uint32_t key_heads, std::uint32_t value_heads, std::uint32_t head_dim) {
    const auto tiles = value_heads * (head_dim / kGdnWindowColumns);
    if (blockIdx.x < tiles) {
        gdn_window_recurrence<0U, Storage>({}, state, saved, control, key_heads, head_dim);
        return;
    }
    const auto pending = gdn_window_pending(control), slot = control->pending_slot & 1U;
    if (!pending) return;
    const auto channel = (static_cast<std::uint32_t>(blockIdx.x) - tiles) * 32U + static_cast<std::uint32_t>(threadIdx.x);
    auto* history = conv_state + static_cast<std::uint64_t>(channel) * kGdnWindowKernel;
    std::uint16_t window[kGdnWindowKernel];
#pragma unroll
    for (std::uint32_t index = 0; index < kGdnWindowKernel; ++index) window[index] = history[index];
    for (std::uint32_t row = 0; row < pending; ++row)
        gdn_window_shift(window, gdn_saved<const std::uint16_t>(saved, slot, row, 0)[channel]);
#pragma unroll
    for (std::uint32_t index = 0; index < kGdnWindowKernel; ++index) history[index] = window[index];
}
template <unsigned Rows>
__global__ __launch_bounds__(kRecurrentThreads)
void gdn_window_norm_kernel(GdnWindowRows<Rows> rows, const void* norm_weights, SmallWeightKind norm_kind,
    std::uint32_t head_dim, float epsilon) {
    __shared__ float scratch[kRecurrentThreads];
    const auto lane = static_cast<std::uint32_t>(threadIdx.x), row = static_cast<std::uint32_t>(blockIdx.y);
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * head_dim + lane;
    float* core = rows.core[row];
    const float value = core[index];
    const float inv = rsqrtf(gdn_window_sum128(scratch, lane, value * value) / static_cast<float>(head_dim) + epsilon);
    const float gate_value = 1.0F / (1.0F + expf(-rows.z[row][index]));
    core[index] = value * inv * load_small(norm_weights, norm_kind, lane) * gate_value;
}
// Dense A8: gdn_window_norm_kernel whose waves also quantize the 32-feature block of the out
// projection's input they just wrote (separate kernel so the default one compiles as before).
template <unsigned Rows>
__global__ __launch_bounds__(kRecurrentThreads)
void gdn_window_norm_dense_a8_kernel(GdnWindowRows<Rows> rows, const void* norm_weights, SmallWeightKind norm_kind,
    std::uint32_t head_dim, float epsilon, std::int8_t* a8_codes, float* a8_deltas, std::uint32_t a8_width) {
    __shared__ float scratch[kRecurrentThreads];
    const auto lane = static_cast<std::uint32_t>(threadIdx.x), row = static_cast<std::uint32_t>(blockIdx.y);
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * head_dim + lane;
    float* core = rows.core[row];
    const float value = core[index];
    const float inv = rsqrtf(gdn_window_sum128(scratch, lane, value * value) / static_cast<float>(head_dim) + epsilon);
    const float gate_value = 1.0F / (1.0F + expf(-rows.z[row][index]));
    const float result = value * inv * load_small(norm_weights, norm_kind, lane) * gate_value;
    core[index] = result;
    q8_rows::dense_a8_quantize_lane(result, a8_codes + static_cast<std::uint64_t>(row) * a8_width + index,
        a8_deltas + (index / 32U) * q8_rows::kDenseA8Rows + row);
}
// Sixteen independent recurrence waves own all 128 value columns of one head.
// They keep the incumbent tile/leaf arithmetic and physical stores. A local
// barrier then replaces the launch boundary before the same 128-leaf norm.
// The norm's surplus lanes write only unused LDS entries and join every barrier.
template <unsigned Rows, GdnRecurrentStorage Storage, bool Dense>
__global__ __launch_bounds__(512)
void gdn_window_recurrent_norm_kernel(GdnWindowRows<Rows> rows, float* state, GdnWindowSaved saved,
    const GdnWindowControl* control, std::uint32_t key_heads, const void* norm_weights, SmallWeightKind norm_kind,
    float epsilon, std::int8_t* a8_codes, float* a8_deltas, std::uint32_t a8_width) {
    constexpr std::uint32_t head_dim = 128U;
    gdn_window_recurrence<Rows, Storage, 8U, false, true>(rows, state, saved, control, key_heads, head_dim);
    __syncthreads();
    __shared__ float scratch[512];
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * head_dim + lane;
    for (unsigned row = 0; row < Rows; ++row) {
        const float value = lane < head_dim ? rows.core[row][index] : 0.0F;
        const float inv = rsqrtf(gdn_window_sum128(scratch, lane, value * value) / static_cast<float>(head_dim) + epsilon);
        if (lane < head_dim) {
            const float gate_value = 1.0F / (1.0F + expf(-rows.z[row][index]));
            const float result = value * inv * load_small(norm_weights, norm_kind, lane) * gate_value;
            rows.core[row][index] = result;
            if constexpr (Dense)
                q8_rows::dense_a8_quantize_lane(result, a8_codes + static_cast<std::uint64_t>(row) * a8_width + index,
                    a8_deltas + (index / 32U) * q8_rows::kDenseA8Rows + row);
        }
    }
}
// Batched F32 beta/alpha projections: per row the legacy 256-leaf chain and tree.
__device__ __forceinline__ float gdn_gate_weight(const float* w, std::uint64_t i) { return w[i]; }
__device__ __forceinline__ float gdn_gate_weight(const std::uint16_t* w, std::uint64_t i) { return bf16_to_float(w[i]); }
template <unsigned Rows, class Weight = float>
__global__ __launch_bounds__(kProjectionThreads)
void gdn_window_f32_gates_kernel(GdnWidePointers<Rows> input, const Weight* beta_weights, const Weight* alpha_weights,
    GdnWindowRows<Rows> rows, std::uint32_t width, std::uint32_t heads) {
    __shared__ float scratch[Rows][kProjectionThreads];
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    const bool alpha = blockIdx.x >= heads;
    const auto out = static_cast<std::uint32_t>(blockIdx.x) - (alpha ? heads : 0U);
    const Weight* weights = (alpha ? alpha_weights : beta_weights) + static_cast<std::uint64_t>(out) * width;
    float sum[Rows]{};
    for (auto feature = lane; feature < width; feature += kProjectionThreads) {
        const float weight = gdn_gate_weight(weights, feature);
#pragma unroll
        for (unsigned row = 0; row < Rows; ++row)
            sum[row] += bf16_to_float(static_cast<const std::uint16_t*>(input.input[row])[feature]) * weight;
    }
#pragma unroll
    for (unsigned row = 0; row < Rows; ++row) scratch[row][lane] = sum[row];
    __syncthreads();
    for (std::uint32_t stride = kProjectionThreads / 2U; stride > 32U; stride >>= 1U) {
        if (lane < stride) {
#pragma unroll
            for (unsigned row = 0; row < Rows; ++row)
                scratch[row][lane] = gdn_window_add(scratch[row][lane], scratch[row][lane + stride]);
        }
        __syncthreads();
    }
    if (lane >= 32U) return;
#pragma unroll
    for (unsigned row = 0; row < Rows; ++row) {
        float total = gdn_window_add(scratch[row][lane], scratch[row][lane + 32U]);
#pragma unroll
        for (unsigned distance = 16U; distance; distance /= 2U)
            total = gdn_window_add(total, __shfl_down(total, distance, 32));
        if (lane == 0U) (alpha ? rows.gate[row] : rows.beta[row])[out] = round_bf16(total);
    }
}

// Prefill rows: T consecutive known rows of one GDN layer, committed in order with no deferral. Row r's
// workspace is base + r * stride (the window arena stripes rows uniformly). Each kernel repeats the window
// kernels' per-row expressions and trees (gdn_window_shift / gdn_window_sum128 / gdn_window_step) with an
// empty pending set, and writes the conv history and recurrent state once, after the last row.
struct GdnPrefillRows {
    float* qkv; float* z; float* beta; float* gate; float* core; std::uint64_t stride;
};
__global__ __launch_bounds__(kRecurrentThreads)
void gdn_prefill_prepare_kernel(GdnPrefillRows rows, std::uint32_t count, std::uint16_t* conv_state,
    const void* conv_weights, SmallWeightKind conv_kind, const void* dt, SmallWeightKind dt_kind, const void* a,
    SmallWeightKind a_kind, std::uint32_t key_heads, std::uint32_t value_heads, std::uint32_t head_dim, float epsilon) {
    __shared__ float scratch[kRecurrentThreads];
    const auto lane = static_cast<std::uint32_t>(threadIdx.x), block = static_cast<std::uint32_t>(blockIdx.x);
    if (block >= key_heads * 2U + value_heads) {
        for (auto head = lane; head < value_heads; head += kRecurrentThreads) {
            for (std::uint32_t row = 0; row < count; ++row) {
                float* beta = rows.beta + row * rows.stride; float* gate = rows.gate + row * rows.stride;
                beta[head] = round_bf16(1.0F / (1.0F + expf(-beta[head])));
                const float biased = gate[head] + load_small(dt, dt_kind, head);
                gate[head] = load_small(a, a_kind, head) * softplus(biased);
            }
        }
        return;
    }
    const auto channel = block * head_dim + lane;
    auto* history = conv_state + static_cast<std::uint64_t>(channel) * kGdnWindowKernel;
    std::uint16_t window[kGdnWindowKernel];
#pragma unroll
    for (std::uint32_t index = 0; index < kGdnWindowKernel; ++index) window[index] = history[index];
    const bool is_qk = block < key_heads * 2U, is_key = is_qk && block >= key_heads;
    for (std::uint32_t row = 0; row < count; ++row) {
        float* qkv = rows.qkv + row * rows.stride;
        gdn_window_shift(window, float_to_bf16(qkv[channel]));
        float sum = 0.0F;
        for (std::uint32_t index = 0; index < kGdnWindowKernel; ++index)
            sum += bf16_to_float(window[index]) *
                load_small(conv_weights, conv_kind, static_cast<std::uint64_t>(channel) * kGdnWindowKernel + index);
        const float silu = sum / (1.0F + expf(-sum));
        float value = round_bf16(silu);
        if (is_qk) {
            const float inverse = rsqrtf(gdn_window_sum128(scratch, lane, value * value) + epsilon);
            const float q_scale = is_key ? 1.0F : rsqrtf(static_cast<float>(head_dim));
            value = value * inverse * q_scale;
        }
        qkv[channel] = value;
    }
#pragma unroll
    for (std::uint32_t index = 0; index < kGdnWindowKernel; ++index) history[index] = window[index];
}
template <GdnRecurrentStorage Storage>
__global__ __launch_bounds__(32)
void gdn_prefill_recurrent_kernel(GdnPrefillRows rows, std::uint32_t count, float* state, std::uint32_t key_heads,
    std::uint32_t head_dim) {
    const auto lane = static_cast<std::uint32_t>(threadIdx.x), tiles = head_dim / kGdnWindowColumns;
    const auto value_head = static_cast<std::uint32_t>(blockIdx.x) / tiles;
    const auto value_base = static_cast<std::uint32_t>(blockIdx.x) % tiles * kGdnWindowColumns;
    const auto key_offset = value_head % key_heads * head_dim, key_width = key_heads * head_dim;
    const auto value_offset = static_cast<std::uint64_t>(value_head) * head_dim + value_base;
    const auto row_inputs = [&](std::uint32_t row) {
        const float* qkv = rows.qkv + row * rows.stride;
        GdnWindowInputs in{};
#pragma unroll
        for (std::uint32_t part = 0; part < 4U; ++part) {
            in.key[part] = qkv[key_width + key_offset + part * 32U + lane];
            in.query[part] = qkv[key_offset + part * 32U + lane];
        }
#pragma unroll
        for (std::uint32_t column = 0; column < kGdnWindowColumns; ++column)
            in.value[column] = qkv[key_width * 2U + value_offset + column];
        in.beta = rows.beta[row * rows.stride + value_head]; in.gate = rows.gate[row * rows.stride + value_head];
        return in;
    };
    float retained[4][kGdnWindowColumns];
#pragma unroll
    for (std::uint32_t part = 0; part < 4U; ++part)
#pragma unroll
        for (std::uint32_t column = 0; column < kGdnWindowColumns; ++column)
            retained[part][column] = state[gdn_window_state_index<Storage>(value_head, part * 32U + lane, value_base + column, head_dim)];
    auto next = row_inputs(0U);
    for (std::uint32_t row = 0; row < count; ++row) {
        const auto in = next;
        if (row + 1U < count) next = row_inputs(row + 1U);
        gdn_window_step<true>(retained, in, rows.core + row * rows.stride + value_offset);
    }
#pragma unroll
    for (std::uint32_t part = 0; part < 4U; ++part)
#pragma unroll
        for (std::uint32_t column = 0; column < kGdnWindowColumns; ++column)
            state[gdn_window_state_index<Storage>(value_head, part * 32U + lane, value_base + column, head_dim)] =
                retained[part][column];
}
__global__ __launch_bounds__(kRecurrentThreads)
void gdn_prefill_norm_kernel(GdnPrefillRows rows, const void* norm_weights, SmallWeightKind norm_kind,
    std::uint32_t head_dim, float epsilon) {
    __shared__ float scratch[kRecurrentThreads];
    const auto lane = static_cast<std::uint32_t>(threadIdx.x), row = static_cast<std::uint32_t>(blockIdx.y);
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * head_dim + lane;
    float* core = rows.core + row * rows.stride;
    const float value = core[index];
    const float inv = rsqrtf(gdn_window_sum128(scratch, lane, value * value) / static_cast<float>(head_dim) + epsilon);
    const float gate_value = 1.0F / (1.0F + expf(-rows.z[row * rows.stride + index]));
    core[index] = value * inv * load_small(norm_weights, norm_kind, lane) * gate_value;
}

// Row-parallel prefill kernels (head_dim 128, conv kernel 4). Same per-row expressions and trees as the
// kernels above; only work placement differs.
//
// F32 beta/alpha projections of every 8-row tile in one launch (blockIdx.y = tile), each (head, row) the legacy
// 256-leaf chain and tree, then the prepare kernel's sigmoid / softplus transform of that row's value. A block
// computes Heads outputs of one kind for its Rows rows, so each input row is read once per Heads outputs; the
// 32-leaf shuffle stage of the (head, row) trees is spread over the block's waves. Input row r is
// input + r * input_stride.
template <unsigned Rows, unsigned Heads>
__global__ __launch_bounds__(kProjectionThreads)
void gdn_prefill_gates_kernel(const std::uint16_t* input, std::uint64_t input_stride, const float* beta_weights,
    const float* alpha_weights, GdnPrefillRows rows, const void* dt, SmallWeightKind dt_kind, const void* a,
    SmallWeightKind a_kind, std::uint32_t width, std::uint32_t heads) {
    constexpr unsigned Sums = Rows * Heads;
    __shared__ float scratch[Sums][kProjectionThreads];
    const auto lane = static_cast<std::uint32_t>(threadIdx.x), first = static_cast<std::uint32_t>(blockIdx.y) * Rows;
    const auto head0 = static_cast<std::uint32_t>(blockIdx.x) * Heads;
    const bool alpha = head0 >= heads;
    const auto out0 = head0 - (alpha ? heads : 0U);
    const float* weights = (alpha ? alpha_weights : beta_weights) + static_cast<std::uint64_t>(out0) * width;
    float sum[Heads][Rows]{};
    for (auto feature = lane; feature < width; feature += kProjectionThreads) {
        float x[Rows];
#pragma unroll
        for (unsigned row = 0; row < Rows; ++row) x[row] = bf16_to_float(input[(first + row) * input_stride + feature]);
#pragma unroll
        for (unsigned h = 0; h < Heads; ++h) {
            const float weight = weights[static_cast<std::uint64_t>(h) * width + feature];
#pragma unroll
            for (unsigned row = 0; row < Rows; ++row) sum[h][row] += x[row] * weight;
        }
    }
#pragma unroll
    for (unsigned h = 0; h < Heads; ++h)
#pragma unroll
        for (unsigned row = 0; row < Rows; ++row) scratch[h * Rows + row][lane] = sum[h][row];
    __syncthreads();
    for (std::uint32_t stride = kProjectionThreads / 2U; stride > 32U; stride >>= 1U) {
        if (lane < stride) {
#pragma unroll
            for (unsigned i = 0; i < Sums; ++i) scratch[i][lane] = gdn_window_add(scratch[i][lane], scratch[i][lane + stride]);
        }
        __syncthreads();
    }
    const auto wave = lane / 32U, l = lane % 32U;
    for (auto i = wave; i < Sums; i += kProjectionThreads / 32U) {
        float total = gdn_window_add(scratch[i][l], scratch[i][l + 32U]);
#pragma unroll
        for (unsigned distance = 16U; distance; distance /= 2U)
            total = gdn_window_add(total, __shfl_down(total, distance, 32));
        if (l == 0U) {
            const auto out = out0 + i / Rows, row = first + i % Rows;
            const float raw = round_bf16(total);
            const auto offset = row * rows.stride + out;
            if (alpha) {
                const float biased = raw + load_small(dt, dt_kind, out);
                rows.gate[offset] = load_small(a, a_kind, out) * softplus(biased);
            } else rows.beta[offset] = round_bf16(1.0F / (1.0F + expf(-raw)));
        }
    }
}
// Conv/SiLU and q/k L2 norm of every row. One block per 128-channel head; Chunk rows at a time, wave w takes
// rows w, w + Waves, ... of the chunk. Lane l holds channels l, l+32, l+64, l+96, so the 128-leaf norm tree is
// gdn_window_key_sum. The BF16 conv inputs of a chunk (plus the three rows before it) are staged in an LDS ring
// before any row of the chunk is overwritten in place.
template <unsigned Waves, unsigned Chunk>
__global__ __launch_bounds__(Waves * 32U)
void gdn_prefill_conv_kernel(GdnPrefillRows rows, std::uint32_t count, std::uint16_t* conv_state,
    const void* conv_weights, SmallWeightKind conv_kind, std::uint32_t key_heads, std::uint32_t head_dim, float epsilon) {
    constexpr unsigned Ring = Chunk * 2U;
    static_assert(Chunk >= 4U && Chunk % Waves == 0U);
    __shared__ std::uint16_t raw[Ring][128];
    const auto thread = static_cast<std::uint32_t>(threadIdx.x), lane = thread % 32U, wave = thread / 32U;
    const auto block = static_cast<std::uint32_t>(blockIdx.x), base = block * head_dim;
    const bool is_qk = block < key_heads * 2U, is_key = is_qk && block >= key_heads;
    const float q_scale = is_key ? 1.0F : rsqrtf(static_cast<float>(head_dim));
    auto* history = conv_state + static_cast<std::uint64_t>(base) * kGdnWindowKernel;
    // Rows -3..-1 are the history's last three inputs; ring slot of row r is (r + Ring) % Ring.
    for (auto i = thread; i < 3U * 128U; i += Waves * 32U)
        raw[Ring - 3U + i / 128U][i % 128U] = history[(i % 128U) * kGdnWindowKernel + 1U + i / 128U];
    float weight[4][kGdnWindowKernel];
#pragma unroll
    for (std::uint32_t part = 0; part < 4U; ++part)
#pragma unroll
        for (std::uint32_t index = 0; index < kGdnWindowKernel; ++index)
            weight[part][index] = load_small(conv_weights, conv_kind,
                static_cast<std::uint64_t>(base + part * 32U + lane) * kGdnWindowKernel + index);
    for (std::uint32_t first = 0; first < count; first += Chunk) {
        for (auto i = thread; i < Chunk * 128U; i += Waves * 32U) {
            const auto row = first + i / 128U;
            if (row < count) raw[row % Ring][i % 128U] = float_to_bf16(rows.qkv[row * rows.stride + base + i % 128U]);
        }
        __syncthreads();
        for (auto row = first + wave; row < first + Chunk && row < count; row += Waves) {
            float value[4], square[4];
#pragma unroll
            for (std::uint32_t part = 0; part < 4U; ++part) {
                const auto channel = part * 32U + lane;
                float sum = 0.0F;
                for (std::uint32_t index = 0; index < kGdnWindowKernel; ++index)
                    sum += bf16_to_float(raw[(row + Ring - 3U + index) % Ring][channel]) * weight[part][index];
                const float silu = sum / (1.0F + expf(-sum));
                value[part] = round_bf16(silu);
                square[part] = value[part] * value[part];
            }
            if (is_qk) {
                const float inverse = rsqrtf(gdn_window_key_sum(square) + epsilon);
#pragma unroll
                for (std::uint32_t part = 0; part < 4U; ++part) value[part] = value[part] * inverse * q_scale;
            }
#pragma unroll
            for (std::uint32_t part = 0; part < 4U; ++part)
                rows.qkv[row * rows.stride + base + part * 32U + lane] = value[part];
        }
        __syncthreads();
    }
    for (auto i = thread; i < kGdnWindowKernel * 128U; i += Waves * 32U)
        history[(i % 128U) * kGdnWindowKernel + i / 128U] = raw[(count - kGdnWindowKernel + i / 128U) % Ring][i % 128U];
}
// Recurrence over all rows with the 128 keys of a state column split across G lanes: lane `sub` of a column's
// group owns keys sub, sub+G, sub+2G, ... The incumbent tree pairs keys i and i+d for d = 64, 32, ..., 1, so
// levels 64..G are in-lane adds and levels G/2..1 are xor exchanges inside the group; every lane of the group
// ends with the same root (the operands of each add are the same two values). A block of Waves waves owns
// Waves*32/G columns of one value head. q/k (transposed by key residue), v, beta and exp(gate) of Chunk rows
// are staged in LDS, double-buffered: the next chunk's global loads are in flight during the current chunk.
template <unsigned G> struct GdnLaneShape {
    static constexpr unsigned keys = 128U / G, pitch = 128U / G + 4U, columns = 32U / G;
};
template <unsigned G, unsigned Columns, unsigned Chunk> struct GdnLaneStage {
    alignas(16) float key[Chunk][G][GdnLaneShape<G>::pitch];
    alignas(16) float query[Chunk][G][GdnLaneShape<G>::pitch];
    alignas(16) float value[Chunk][Columns];
    float beta[Chunk], decay[Chunk];
};
template <unsigned N>
__device__ __forceinline__ float gdn_lane_tree(float (&p)[N]) {
#pragma clang fp contract(off)
#pragma unroll
    for (unsigned half = N / 2U; half; half /= 2U)
#pragma unroll
        for (unsigned j = 0; j < half; ++j) p[j] = p[j] + p[j + half];
    return p[0];
}
template <unsigned G>
__device__ __forceinline__ float gdn_lane_group_sum(float sum) {
#pragma clang fp contract(off)
    if constexpr (G > 1U) {
        sum = sum + __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(sum), 0x160 + static_cast<int>(G / 2U),
            0xf, 0xf, false));  // DPP row_xmask:G/2
        return gdn_lane_group_sum<G / 2U>(sum);
    } else return sum;
}
// Workgroup barrier for LDS hand-offs only: unlike __syncthreads it does not wait for outstanding global stores.
__device__ __forceinline__ void gdn_lds_barrier() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }
template <GdnRecurrentStorage Storage, unsigned G, unsigned Waves, unsigned Chunk>
__global__ __launch_bounds__(Waves * 32U)
void gdn_prefill_lanes_kernel(GdnPrefillRows rows, std::uint32_t count, float* state, std::uint32_t key_heads) {
#pragma clang fp contract(off)
    using Shape = GdnLaneShape<G>;
    constexpr unsigned Threads = Waves * 32U, Columns = Waves * Shape::columns, K = Shape::keys;
    constexpr unsigned KeyQuads = Chunk * 64U / Threads, ValueQuads = (Chunk * Columns / 4U + Threads - 1U) / Threads;
    static_assert(Chunk * 64U % Threads == 0U && 2U * Chunk <= Threads);
    using Stage = GdnLaneStage<G, Columns, Chunk>;
    __shared__ Stage stage[2];
    const auto thread = static_cast<std::uint32_t>(threadIdx.x), lane = thread % 32U;
    const auto sub = lane % G, local = thread / 32U * Shape::columns + lane / G;
    const auto blocks = 128U / Columns;
    const auto value_head = static_cast<std::uint32_t>(blockIdx.x) / blocks;
    const auto column_base = static_cast<std::uint32_t>(blockIdx.x) % blocks * Columns, column = column_base + local;
    const auto key_width = key_heads * 128U, key_offset = value_head % key_heads * 128U;
    const auto value_offset = key_width * 2U + value_head * 128U + column_base;
    float4 key_query[KeyQuads], values[ValueQuads];
    float scalar = 0.0F;
    const auto fetch = [&](std::uint32_t first) {
#pragma unroll
        for (unsigned i = 0; i < KeyQuads; ++i) {
            const auto q = thread + i * Threads, row = q / 64U, within = q % 64U;
            const float* source = rows.qkv + (first + row) * rows.stride + key_offset +
                (within < 32U ? key_width + within * 4U : (within - 32U) * 4U);
            key_query[i] = *reinterpret_cast<const float4*>(source);
        }
#pragma unroll
        for (unsigned i = 0; i < ValueQuads; ++i) {
            const auto q = thread + i * Threads, row = q / (Columns / 4U);
            if (q < Chunk * Columns / 4U)
                values[i] = *reinterpret_cast<const float4*>(rows.qkv + (first + row) * rows.stride + value_offset +
                    q % (Columns / 4U) * 4U);
        }
        if (thread < 2U * Chunk)
            scalar = (thread < Chunk ? rows.beta : rows.gate)[(first + thread % Chunk) * rows.stride + value_head];
    };
    const auto commit = [&](Stage& s) {
#pragma unroll
        for (unsigned i = 0; i < KeyQuads; ++i) {
            const auto q = thread + i * Threads, row = q / 64U, within = q % 64U;
            const auto key = (within % 32U) * 4U;
            auto& target = within < 32U ? s.key[row] : s.query[row];
            const float parts[4] = {key_query[i].x, key_query[i].y, key_query[i].z, key_query[i].w};
#pragma unroll
            for (unsigned c = 0; c < 4U; ++c) target[(key + c) % G][(key + c) / G] = parts[c];
        }
#pragma unroll
        for (unsigned i = 0; i < ValueQuads; ++i) {
            const auto q = thread + i * Threads;
            if (q < Chunk * Columns / 4U)
                *reinterpret_cast<float4*>(&s.value[q / (Columns / 4U)][q % (Columns / 4U) * 4U]) = values[i];
        }
        if (thread < Chunk) s.beta[thread] = scalar;
        else if (thread < 2U * Chunk) s.decay[thread - Chunk] = expf(scalar);
    };
    float retained[K];
#pragma unroll
    for (unsigned j = 0; j < K; ++j)
        retained[j] = state[gdn_window_state_index<Storage>(value_head, sub + j * G, column, 128U)];
    fetch(0U);
    commit(stage[0]);
    __syncthreads();
    for (std::uint32_t first = 0, buffer = 0; first < count; first += Chunk, buffer ^= 1U) {
        const bool more = first + Chunk < count;
        if (more) fetch(first + Chunk);
        // Row pointers advance by constant strides (a multiplied LDS row offset compiles to a 64-bit mad whose
        // undefined high half can alias an in-flight load's register and force a wait for the prefetch).
        const Stage& s = stage[buffer];
        const float* key = s.key[0][sub];
        const float* query = s.query[0][sub];
        const float* value = &s.value[0][local];
        const float* beta = s.beta;
        const float* decay = s.decay;
        float* core = rows.core + first * rows.stride + value_head * 128U + column;
#pragma unroll 1
        for (unsigned r = 0; r < Chunk; ++r) {
            float products[K];
#pragma unroll
            for (unsigned j = 0; j < K; ++j) {
                retained[j] = retained[j] * *decay;
                products[j] = retained[j] * key[j];
            }
            const float delta = (*value - gdn_lane_group_sum<G>(gdn_lane_tree(products))) * *beta;
#pragma unroll
            for (unsigned j = 0; j < K; ++j) {
                retained[j] = __builtin_fmaf(key[j], delta, retained[j]);
                products[j] = retained[j] * query[j];
            }
            const float out = gdn_lane_group_sum<G>(gdn_lane_tree(products));
            if (sub == 0U) *core = out;
            key += G * Shape::pitch; query += G * Shape::pitch; value += Columns; ++beta; ++decay; core += rows.stride;
        }
        if (more) commit(stage[buffer ^ 1U]);
        gdn_lds_barrier();
    }
#pragma unroll
    for (unsigned j = 0; j < K; ++j)
        state[gdn_window_state_index<Storage>(value_head, sub + j * G, column, 128U)] = retained[j];
}
// Gated RMSNorm, one wave per (row, head): lane l holds channels l, l+32, l+64, l+96 (gdn_window_key_sum tree).
// A8: each 32-channel part is also one dense A8 block of the out projection's input; the wave quantizes it
// (dense_a8_quantize_lane, the quantizer kernel's values) into a8_codes [row][feature] and a8_deltas laid
// out as the projection's 256-row launches read them ([row / 256][block][row % 256]).
template <unsigned Waves, bool A8 = false>
__global__ __launch_bounds__(Waves * 32U)
void gdn_prefill_norm_rows_kernel(GdnPrefillRows rows, std::uint32_t count, std::uint32_t value_heads,
    const void* norm_weights, SmallWeightKind norm_kind, std::uint32_t head_dim, float epsilon, std::int8_t* a8_codes = nullptr,
    float* a8_deltas = nullptr) {
    const auto lane = static_cast<std::uint32_t>(threadIdx.x) % 32U;
    const auto item = static_cast<std::uint32_t>(blockIdx.x) * Waves + static_cast<std::uint32_t>(threadIdx.x) / 32U;
    if (item >= count * value_heads) return;
    const std::uint64_t row = item / value_heads, base = row * rows.stride + item % value_heads * head_dim;
    float value[4], square[4];
#pragma unroll
    for (std::uint32_t part = 0; part < 4U; ++part) {
        value[part] = rows.core[base + part * 32U + lane];
        square[part] = value[part] * value[part];
    }
    const float inv = rsqrtf(gdn_window_key_sum(square) / static_cast<float>(head_dim) + epsilon);
#pragma unroll
    for (std::uint32_t part = 0; part < 4U; ++part) {
        const auto channel = part * 32U + lane;
        const float gate_value = 1.0F / (1.0F + expf(-rows.z[base + channel]));
        const float result = value[part] * inv * load_small(norm_weights, norm_kind, channel) * gate_value;
        rows.core[base + channel] = result;
        if constexpr (A8) {
            constexpr std::uint64_t pitch = q8_rows::kDenseA8WideRows;
            const std::uint64_t width = static_cast<std::uint64_t>(value_heads) * head_dim, feature = item % value_heads * head_dim + channel;
            q8_rows::dense_a8_quantize_lane(result, a8_codes + row * width + feature,
                a8_deltas + row / pitch * (width / 32U * pitch) + feature / 32U * pitch + row % pitch);
        }
    }
}
#include "hip_gdn_prefill_scan.inc"
#include "hip_gdn_qpf2k_prefill.inc"
#include "hip_gdn_fp16_chunk.inc"
// qwen-gdn-fp16-chunk row epilogue: gdn_prefill_norm_rows_kernel's gated RMSNorm expression (and dense A8) for one row
// and head of the chunked core, or with core16 the fused core's CORE16 staging (RNE FP16 of the same normed value, no
// FP32 core row, one overflow count per aligned 8-lane group). z arrives prefetched; weights once per CTA.
struct GdnFp16NormEpilogue {
    static constexpr bool kRows = true;
    GdnPrefillRows rows;
    const void* norm_weights;
    SmallWeightKind norm_kind;
    std::uint32_t head_dim, value_heads;
    float epsilon;
    std::int8_t* a8_codes;
    float* a8_deltas;
    _Float16* core16;
    std::uint32_t core16_pitch;
    unsigned* core16_overflow;
    __device__ void weights(unsigned lane, float (&w)[4]) const {
#pragma unroll
        for (std::uint32_t part = 0; part < 4U; ++part) w[part] = load_small(norm_weights, norm_kind, part * 32U + lane);
    }
    __device__ void operator()(std::uint64_t row, unsigned head, const float (&value)[4], const float (&z)[4],
        const float (&w)[4]) const {
        const auto lane = static_cast<std::uint32_t>(threadIdx.x) % 32U;
        const std::uint64_t base = row * rows.stride + head * head_dim;
        float square[4];
#pragma unroll
        for (std::uint32_t part = 0; part < 4U; ++part) square[part] = value[part] * value[part];
        const float inv = rsqrtf(gdn_window_key_sum(square) / static_cast<float>(head_dim) + epsilon);
#pragma unroll
        for (std::uint32_t part = 0; part < 4U; ++part) {
            const auto channel = part * 32U + lane;
            const float gate_value = 1.0F / (1.0F + expf(-z[part]));
            const float result = value[part] * inv * w[part] * gate_value;
            if (core16) {
                core16[row * core16_pitch + head * head_dim + channel] = static_cast<_Float16>(result);
                const auto bad = __ballot(isfinite(result) && !(fabsf(result) <= 65504.0F));
                if (bad != 0ULL && lane % 8U == 0U && ((bad >> lane) & 0xffULL) != 0ULL) atomicAdd(core16_overflow, 1U);
                continue;
            }
            rows.core[base + channel] = result;
            if (a8_codes) {
                constexpr std::uint64_t pitch = q8_rows::kDenseA8WideRows;
                const std::uint64_t width = static_cast<std::uint64_t>(value_heads) * head_dim, feature = head * head_dim + channel;
                q8_rows::dense_a8_quantize_lane(result, a8_codes + row * width + feature,
                    a8_deltas + row / pitch * (width / 32U * pitch) + feature / 32U * pitch + row % pitch);
            }
        }
    }
};
// Quantized out-projection input of whole 256-row launches (the norm kernel writes it; prefill steps of up to
// kGdnFusedA8Rows rows at most kDenseA8MaxWidth wide).
constexpr std::uint32_t kGdnFusedA8Rows = 2048U;
q8_rows::DenseA8Slot gdn_fused_a8_scratch() {
    static unsigned char* const pool = [] {
        void* p = nullptr;
        const std::uint64_t bytes = std::uint64_t{kGdnFusedA8Rows} * q8_rows::kDenseA8MaxWidth +
            std::uint64_t{q8_rows::kDenseA8MaxWidth / 32U} * kGdnFusedA8Rows * 4U;
        if (hipMalloc(&p, bytes) != hipSuccess) throw std::runtime_error("GDN prefill: fused dense A8 scratch allocation failed");
        return static_cast<unsigned char*>(p);
    }();
    return {reinterpret_cast<std::int8_t*>(pool),
        reinterpret_cast<float*>(pool + std::uint64_t{kGdnFusedA8Rows} * q8_rows::kDenseA8MaxWidth)};
}

struct GdnKernelNode {
    const void* function;
    dim3 grid, block;
    std::size_t shared_bytes;
};
using GdnKernelLedger = std::vector<GdnKernelNode>;
struct GdnEmitTarget {
    hipStream_t stream;
    std::uint32_t* graph_status;
    const std::uint64_t* graph_position;
    std::uint64_t context_tokens;
    GdnKernelLedger* ledger;
};
void record_gdn_kernel(GdnKernelLedger* ledger, const void* function, dim3 grid, dim3 block) {
    if (ledger) ledger->push_back({function, grid, block, 0});
}
bool gdn_q6_r1_registered(const RuntimeTensorBinding& weights, std::uint32_t width, std::uint32_t outputs) {
    const auto* q6 = q6p::find(reinterpret_cast<const void*>(weights.address));
    return q6 && q6->kind == q6p::Kind::q6p32 && q6->rows == outputs && q6->cols == width;
}

template <WeightKind Kind>
void launch_projection(
    const std::uint16_t* input,
    const RuntimeTensorBinding& weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    float* output, hipStream_t stream, GdnKernelLedger* ledger) {
    record_gdn_kernel(ledger, reinterpret_cast<const void*>(project_bf16_to_fp32_kernel<Kind>),
        dim3(output_width), dim3(kProjectionThreads));
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            project_bf16_to_fp32_kernel<
                Kind>),
        dim3(output_width),
        dim3(kProjectionThreads),
        0,
        stream,
        input,
        reinterpret_cast<const void*>(
            weights.address),
        input_width,
        output_width,
        output);
    check_hip(
        hipGetLastError(),
        "HIP GDN input projection launch");
}

void launch_projection_by_kind(
    WeightKind kind,
    const std::uint16_t* input,
    const RuntimeTensorBinding& weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    float* output, ProjectionMode mode, hipStream_t stream, GdnKernelLedger* ledger) {
    switch (kind) {
    case WeightKind::f32:
        launch_projection<WeightKind::f32>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
    case WeightKind::f16:
        launch_projection<WeightKind::f16>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
    case WeightKind::bf16:
        launch_projection<WeightKind::bf16>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
    case WeightKind::q8_0:
        if (mode == ProjectionMode::prefetch8_wave8 && !ledger && gdn_q6_r1_registered(weights, input_width, output_width)) {
            // Registered Q6P32 copy: the one-row q8_rows site is bitwise the prefetch8 wave8 kernel.
            // Island emitters (ledger) keep the recorded Q8_0 kernel; their numerics are the same.
            GdnInputRowsSite<1> site{};
            site.segments = q8_rows::segments({{reinterpret_cast<const void*>(weights.address), output_width}});
            site.width = input_width; site.outputs = output_width; site.input[0] = input; site.output[0][0] = output;
            q8_rows::launch(site, false, stream);
            check_hip(hipGetLastError(), "HIP GDN Q6P32 input projection launch");
            return;
        }
        if (mode == ProjectionMode::prefetch8_wave8) {
            record_gdn_kernel(ledger, reinterpret_cast<const void*>(project_prefetch8_wave8_q8_bf16_to_fp32_kernel), dim3((output_width + 7U) / 8U), dim3(256));
            hipLaunchKernelGGL(project_prefetch8_wave8_q8_bf16_to_fp32_kernel,
                dim3((output_width + 7U) / 8U), dim3(256), 0, stream,
                input, reinterpret_cast<const void*>(weights.address),
                input_width, output_width, output);
            check_hip(hipGetLastError(), "HIP GDN prefetch8 wave8 Q8 input projection launch");
            return;
        }
        if (mode == ProjectionMode::prefetch8) {
            record_gdn_kernel(ledger, reinterpret_cast<const void*>(project_prefetch8_q8_bf16_to_fp32_kernel), dim3(output_width), dim3(32));
            hipLaunchKernelGGL(project_prefetch8_q8_bf16_to_fp32_kernel,
                dim3(output_width), dim3(32), 0, stream,
                input, reinterpret_cast<const void*>(weights.address),
                input_width, output_width, output);
            check_hip(hipGetLastError(), "HIP GDN prefetch8 Q8 input projection launch");
            return;
        }
        if (mode == ProjectionMode::virtual_stream) {
            record_gdn_kernel(ledger, reinterpret_cast<const void*>(project_virtual_q8_bf16_to_fp32_kernel), dim3(output_width), dim3(32));
            hipLaunchKernelGGL(project_virtual_q8_bf16_to_fp32_kernel,
                dim3(output_width), dim3(32), 0, stream,
                input, reinterpret_cast<const void*>(weights.address),
                input_width, output_width, output);
            check_hip(hipGetLastError(), "HIP GDN virtual Q8 input projection launch");
            return;
        }
        launch_projection<WeightKind::q8_0>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
    case WeightKind::iq4_nl:
        launch_projection<WeightKind::iq4_nl>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
    case WeightKind::iq4_xs:
        launch_projection<WeightKind::iq4_xs>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
#define STRIX_GGML_KIND_CASE(KIND, type) \
    case WeightKind::KIND: \
        launch_projection<WeightKind::KIND>( \
            input, weights, input_width, output_width, output, stream, ledger); \
        return;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_CASE) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_CASE)
#undef STRIX_GGML_KIND_CASE
    }
}

template <WeightKind Kind>
void launch_output_projection(
    const float* input,
    const RuntimeTensorBinding& weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    std::uint16_t* output, hipStream_t stream, GdnKernelLedger* ledger) {
    record_gdn_kernel(ledger, reinterpret_cast<const void*>(project_fp32_to_bf16_kernel<Kind>),
        dim3(output_width), dim3(kProjectionThreads));
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            project_fp32_to_bf16_kernel<
                Kind>),
        dim3(output_width),
        dim3(kProjectionThreads),
        0,
        stream,
        input,
        reinterpret_cast<const void*>(
            weights.address),
        input_width,
        output_width,
        output);
    check_hip(
        hipGetLastError(),
        "HIP GDN output projection launch");
}

void launch_output_by_kind(
    WeightKind kind,
    const float* input,
    const RuntimeTensorBinding& weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    std::uint16_t* output, bool virtual_q8, bool wave8, hipStream_t stream, GdnKernelLedger* ledger) {
    if (wave8) {
        if (!virtual_q8 || kind != WeightKind::q8_0)
            throw std::invalid_argument("wave8 GDN output requires virtual Q8_0 weights");
        q8_projection_organization::validate_fp32_bf16_views(input_width, output_width,
            {reinterpret_cast<std::uintptr_t>(input), static_cast<std::uint64_t>(input_width) * sizeof(float)},
            {weights.address, weights.bytes},
            {reinterpret_cast<std::uintptr_t>(output), static_cast<std::uint64_t>(output_width) * sizeof(std::uint16_t)});
        if (!ledger && gdn_q6_r1_registered(weights, input_width, output_width)) {
            // Registered Q6P32 copy: the one-row q8_rows site is bitwise the prefetch8 wave8 kernel.
            GdnOutputRowsSite<1> site{};
            site.segments = q8_rows::segments({{reinterpret_cast<const void*>(weights.address), output_width}});
            site.width = input_width; site.outputs = output_width; site.input[0] = input; site.output[0] = output;
            q8_rows::launch(site, false, stream);
            check_hip(hipGetLastError(), "HIP GDN Q6P32 output projection launch");
            return;
        }
        record_gdn_kernel(ledger, reinterpret_cast<const void*>(project_prefetch8_wave8_q8_fp32_to_bf16_kernel), dim3((output_width + 7U) / 8U), dim3(256));
        hipLaunchKernelGGL(project_prefetch8_wave8_q8_fp32_to_bf16_kernel, dim3((output_width + 7U) / 8U), dim3(256), 0,
            stream, input, reinterpret_cast<const void*>(weights.address), input_width, output_width, output);
        check_hip(hipGetLastError(), "HIP GDN prefetch8 wave8 Q8 output projection launch");
        return;
    }
    if(virtual_q8) {
        if(kind!=WeightKind::q8_0 || input_width%32U!=0U)
            throw std::invalid_argument("virtual Q8 GDN output requires aligned Q8_0 weights");
        record_gdn_kernel(ledger, reinterpret_cast<const void*>(project_virtual_q8_fp32_to_bf16_kernel), dim3(output_width), dim3(32));
        hipLaunchKernelGGL(project_virtual_q8_fp32_to_bf16_kernel,dim3(output_width),dim3(32),0,
            stream,input,reinterpret_cast<const void*>(weights.address),input_width,output_width,output);
        check_hip(hipGetLastError(),"HIP GDN virtual Q8 output projection launch");return;
    }
    switch (kind) {
    case WeightKind::f32:
        launch_output_projection<WeightKind::f32>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
    case WeightKind::f16:
        launch_output_projection<WeightKind::f16>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
    case WeightKind::bf16:
        launch_output_projection<WeightKind::bf16>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
    case WeightKind::q8_0:
        launch_output_projection<WeightKind::q8_0>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
    case WeightKind::iq4_nl:
        launch_output_projection<WeightKind::iq4_nl>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
    case WeightKind::iq4_xs:
        launch_output_projection<WeightKind::iq4_xs>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
#define STRIX_GGML_KIND_CASE(KIND, type) \
    case WeightKind::KIND: \
        launch_output_projection<WeightKind::KIND>( \
            input, weights, input_width, output_width, output, stream, ledger); \
        return;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_CASE) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_CASE)
#undef STRIX_GGML_KIND_CASE
    }
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
            "HIP GDN requires model manifest resources");
    }
    if (index >=
        context.resources->
            model_manifest->
            tensors.size()) {
        throw std::out_of_range(
            "HIP GDN tensor index exceeds manifest");
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
    TensorPair candidate) {
    destination =
        candidate;
    ++count;
}

GdnTensors identify_tensors(
    const ExecutionDispatchContext& context,
    const HipGdnLauncherConfig& config) {
    GdnTensors result;
    if(const auto* cached=context.prepared_tensor_views) {
        if(!context.resources) throw std::invalid_argument("prepared GDN roles require retained resources");
        cached->validate_stage_scope(context.resources->distributed_stage,context.step_id);
        cached->validate_source_handles(context.resources->model_manifest,context.resources->model_weights,context.resources->tensor_access);
        const auto& roles=cached->gdn_roles(context.execution_identity,context.step_id);
        result.qkv=tensor_pair(context,roles.qkv);result.z=tensor_pair(context,roles.z);
        result.beta=tensor_pair(context,roles.beta);result.alpha=tensor_pair(context,roles.alpha);
        result.conv=tensor_pair(context,roles.conv);result.dt=tensor_pair(context,roles.dt);
        result.a=tensor_pair(context,roles.a);result.norm=tensor_pair(context,roles.norm);result.out=tensor_pair(context,roles.out);
    } else {
    const auto& recurrent =
        resolve_dispatch_tensor_access(
            context, 0);
    const auto& attention =
        resolve_dispatch_tensor_access(
            context, 1);

    std::uint32_t qkv_count = 0;
    std::uint32_t z_count = 0;
    std::uint32_t beta_count = 0;
    std::uint32_t alpha_count = 0;
    std::uint32_t conv_count = 0;
    std::uint32_t dt_count = 0;
    std::uint32_t a_count = 0;
    std::uint32_t norm_count = 0;
    std::uint32_t out_count = 0;

    for (const auto index :
         attention.tensor_indices) {
        const auto pair =
            tensor_pair(
                context,
                index);
        const auto& tensor =
            *pair.manifest;
        if (tensor.role !=
            TensorRole::attention_projection) {
            continue;
        }
        if (contains(
                tensor.name,
                "attn_qkv")) {
            assign_unique(
                result.qkv,
                qkv_count,
                pair);
        } else if (contains(
                       tensor.name,
                       "attn_gate")) {
            assign_unique(
                result.z,
                z_count,
                pair);
        }
    }

    for (const auto index :
         recurrent.tensor_indices) {
        const auto pair =
            tensor_pair(
                context,
                index);
        const auto& tensor =
            *pair.manifest;
        if (tensor.role !=
            TensorRole::recurrent_state) {
            continue;
        }
        const auto name =
            std::string_view(
                tensor.name);
        if (contains(
                name,
                "ssm_conv1d")) {
            assign_unique(
                result.conv,
                conv_count,
                pair);
        } else if (contains(
                       name,
                       "ssm_beta")) {
            assign_unique(
                result.beta,
                beta_count,
                pair);
        } else if (contains(
                       name,
                       "ssm_alpha")) {
            assign_unique(
                result.alpha,
                alpha_count,
                pair);
        } else if (contains(
                       name,
                       "ssm_dt")) {
            assign_unique(
                result.dt,
                dt_count,
                pair);
        } else if (contains(
                       name,
                       "ssm_norm")) {
            assign_unique(
                result.norm,
                norm_count,
                pair);
        } else if (contains(
                       name,
                       "ssm_out")) {
            assign_unique(
                result.out,
                out_count,
                pair);
        } else if (contains(
                       name,
                       "ssm_a") &&
                   !contains(
                       name,
                       "ssm_alpha")) {
            assign_unique(
                result.a,
                a_count,
                pair);
        } else if (contains(
                       name,
                       "ssm_in")) {
            throw std::invalid_argument(
                "HIP GDN v1 does not support legacy fused ssm_in export");
        }
    }

    if (qkv_count != 1 ||
        z_count != 1 ||
        beta_count != 1 ||
        alpha_count != 1 ||
        conv_count != 1 ||
        dt_count != 1 ||
        a_count != 1 ||
        norm_count != 1 ||
        out_count != 1) {
        throw std::invalid_argument(
            "HIP GDN tensor family is missing or ambiguous");
    }

    }

    const auto hidden =
        config.model.hidden_size;
    const auto key_width =
        config.model.gdn_key_heads *
        config.model.gdn_key_head_dim;
    const auto value_width =
        config.model.gdn_value_heads *
        config.model.gdn_value_head_dim;
    const auto conv_width =
        key_width * 2ULL +
        value_width;

    const auto expect_matrix =
        [&](const TensorPair& pair,
            std::uint64_t input,
            std::uint64_t output,
            const char* name) {
            if (pair.manifest->
                    dimensions.size() < 2 ||
                pair.manifest->
                    dimensions[0] !=
                    input ||
                pair.manifest->
                    dimensions[1] !=
                    output) {
                throw std::invalid_argument(
                    std::string(name) +
                    " GDN tensor geometry mismatch");
            }
        };
    const auto expect_vector =
        [&](const TensorPair& pair,
            std::uint64_t elements,
            const char* name) {
            if (pair.manifest->
                    dimensions.empty() ||
                pair.manifest->
                    dimensions[0] !=
                    elements) {
                throw std::invalid_argument(
                    std::string(name) +
                    " GDN vector geometry mismatch");
            }
        };

    expect_matrix(
        result.qkv,
        hidden,
        conv_width,
        "qkv");
    expect_matrix(
        result.z,
        hidden,
        value_width,
        "z");
    expect_matrix(
        result.beta,
        hidden,
        config.model.gdn_value_heads,
        "beta");
    expect_matrix(
        result.alpha,
        hidden,
        config.model.gdn_value_heads,
        "alpha");
    expect_matrix(
        result.conv,
        config.model.gdn_conv_kernel,
        conv_width,
        "conv");
    expect_vector(
        result.dt,
        config.model.gdn_value_heads,
        "dt");
    expect_vector(
        result.a,
        config.model.gdn_value_heads,
        "A");
    expect_vector(
        result.norm,
        config.model.gdn_value_head_dim,
        "norm");
    expect_matrix(
        result.out,
        value_width,
        hidden,
        "out");

    return result;
}

void validate_projection_tensor(
    const TensorPair& pair,
    std::uint32_t input_width,
    std::uint64_t output_rows,
    const char* name) {
    if (pair.runtime == nullptr ||
        pair.manifest == nullptr ||
        !runtime_tensor_gpu_visible(pair.runtime->storage) ||
        pair.runtime->address == 0) {
        throw std::invalid_argument(
            std::string(name) +
            " GDN projection is not GPU-visible");
    }
    const auto required =
        required_weight_bytes(
            weight_kind(
                pair.manifest->
                    ggml_type),
            input_width,
            output_rows);
    if (pair.runtime->bytes <
        required) {
        throw std::invalid_argument(
            std::string(name) +
            " GDN projection binding is undersized");
    }
}

void validate_small_tensor(
    const TensorPair& pair,
    std::uint64_t elements,
    const char* name) {
    if (pair.runtime == nullptr ||
        pair.manifest == nullptr ||
        !runtime_tensor_gpu_visible(pair.runtime->storage) ||
        pair.runtime->address == 0) {
        throw std::invalid_argument(
            std::string(name) +
            " GDN tensor is not GPU-visible");
    }
    const auto required =
        small_weight_bytes(
            small_weight_kind(
                pair.manifest->
                    ggml_type),
            elements);
    if (pair.runtime->bytes <
        required) {
        throw std::invalid_argument(
            std::string(name) +
            " GDN tensor binding is undersized");
    }
}

Workspace workspace_for(
    const ExecutionBufferAddress& temporary,
    const ModelContract& model) {
    const auto key_width =
        model.gdn_key_heads *
        model.gdn_key_head_dim;
    const auto value_width =
        model.gdn_value_heads *
        model.gdn_value_head_dim;
    const auto conv_width =
        key_width * 2ULL +
        value_width;
    const auto elements =
        conv_width +
        value_width +
        model.gdn_value_heads * 2ULL +
        value_width;

    if (temporary.address == 0 ||
        temporary.bytes <
            elements *
                sizeof(float)) {
        throw std::invalid_argument(
            "HIP GDN temporary workspace is undersized");
    }

    auto* base =
        reinterpret_cast<float*>(
            temporary.address);
    Workspace result;
    result.qkv = base;
    result.z =
        result.qkv +
        conv_width;
    result.beta =
        result.z +
        value_width;
    result.gate =
        result.beta +
        model.gdn_value_heads;
    result.core =
        result.gate +
        model.gdn_value_heads;
    result.elements =
        elements;
    return result;
}

struct PreparedGdnLaunch {
    ExecutionBufferAddress input, output, temporary;
    Workspace workspace;
    GdnTensors tensors;
    SessionStateAddress matrix_state, conv_state;
    std::uint32_t hidden, key_heads, value_heads, head_dim, value_width, conv_width, kernel;
    ProjectionMode projection_mode;
    bool graph_capture;
};

PreparedGdnLaunch prepare_gdn_launch(
    const ExecutionDispatchContext& context, const HipGdnLauncherConfig* config_ptr) {
    if (context.op !=
        SerialOp::gated_delta_net) {
        throw std::invalid_argument(
            "HIP GDN launcher attached to wrong operator");
    }
    if (!context.layer.has_value() ||
        context.reads.size() != 1 ||
        context.writes.size() != 1 ||
        context.temporaries.size() != 1) {
        throw std::invalid_argument(
            "HIP GDN expects layer id, one read/write and one temporary");
    }
    if (context.logical_rows != 1 ||
        context.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP GDN v1 supports only C1 one-row decode");
    }
    if (config_ptr == nullptr) {
        throw std::invalid_argument(
            "HIP GDN requires configuration");
    }

    const auto& config = *config_ptr;
    validate_config(config);

    if (config.recurrent_storage != GdnRecurrentStorage::key_value) {
        if (const auto* scope = current_hip_launch_context(); scope && scope->graph_capture)
            throw std::invalid_argument("GDN recurrent storage supports eager execution only");
        hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
        check_hip(hipStreamIsCapturing(hip_execution_stream(), &capture), "query GDN recurrent storage capture");
        if (capture != hipStreamCaptureStatusNone)
            throw std::invalid_argument("GDN recurrent storage cannot be captured");
    }
    if (config.recurrent_layout != HipGdnRecurrentLayout::baseline) {
        if (const auto* scope = current_hip_launch_context(); scope && scope->graph_capture)
            throw std::invalid_argument("GDN recurrent LDS layouts support eager execution only");
        hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
        check_hip(hipStreamIsCapturing(hip_execution_stream(), &capture), "query GDN recurrent LDS capture");
        if (capture != hipStreamCaptureStatusNone)
            throw std::invalid_argument("GDN recurrent LDS layouts cannot be captured");
    }
    if (config.wave8_q8_input_projections || config.wave8_q8_output_projection) {
        if (const auto* scope = current_hip_launch_context(); scope && scope->graph_capture)
            throw std::invalid_argument("wave8 GDN projections support eager execution only");
        hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
        check_hip(hipStreamIsCapturing(hip_execution_stream(), &capture), "query wave8 GDN capture");
        if (capture != hipStreamCaptureStatusNone)
            throw std::invalid_argument("wave8 GDN projections cannot be captured");
    }

    const auto input =
        resolve_dispatch_read(
            context, 0);
    const auto output =
        resolve_dispatch_write(
            context, 0);
    const auto temporary =
        resolve_dispatch_temporary(
            context, 0);
    const auto workspace =
        workspace_for(
            temporary,
            config.model);
    const auto tensors =
        identify_tensors(
            context,
            config);

    const auto hidden =
        checked_u32(
            config.model.hidden_size,
            "hidden_size");
    const auto key_heads =
        checked_u32(
            config.model.gdn_key_heads,
            "gdn_key_heads");
    const auto value_heads =
        checked_u32(
            config.model.gdn_value_heads,
            "gdn_value_heads");
    const auto head_dim =
        checked_u32(
            config.model.gdn_key_head_dim,
            "gdn_head_dim");
    const auto key_width =
        checked_u32(
            config.model.gdn_key_heads *
                config.model.gdn_key_head_dim,
            "gdn_key_width");
    const auto value_width =
        checked_u32(
            config.model.gdn_value_heads *
                config.model.gdn_value_head_dim,
            "gdn_value_width");
    const auto conv_width =
        checked_u32(
            static_cast<std::uint64_t>(
                key_width) *
                2ULL +
            value_width,
            "gdn_conv_width");
    const auto kernel =
        checked_u32(
            config.model.gdn_conv_kernel,
            "gdn_conv_kernel");

    const auto hidden_bytes =
        static_cast<std::uint64_t>(
            hidden) *
        sizeof(std::uint16_t);
    if (input.bytes <
            hidden_bytes ||
        output.bytes <
            hidden_bytes) {
        throw std::invalid_argument(
            "HIP GDN hidden buffers are undersized");
    }

    validate_projection_tensor(
        tensors.qkv,
        hidden,
        conv_width,
        "qkv");
    validate_projection_tensor(
        tensors.z,
        hidden,
        value_width,
        "z");
    validate_projection_tensor(
        tensors.beta,
        hidden,
        value_heads,
        "beta");
    validate_projection_tensor(
        tensors.alpha,
        hidden,
        value_heads,
        "alpha");
    // Keep the eight-feature dereference proof local to the opt-in path,
    // independently of required_weight_bytes' existing alignment check.
    if (config.prefetch_q8_input_projections && hidden % 32U != 0U &&
        (tensors.qkv.manifest->ggml_type == 8U || tensors.z.manifest->ggml_type == 8U ||
         tensors.beta.manifest->ggml_type == 8U || tensors.alpha.manifest->ggml_type == 8U))
        throw std::invalid_argument("HIP GDN prefetch8 Q8 input width must align to32");
    if (config.wave8_q8_input_projections) {
        const bool dense_covers = config.dense_a8_projections && q8_rows::dense_a8_type(tensors.qkv.manifest->ggml_type) &&
            q8_rows::dense_a8_type(tensors.z.manifest->ggml_type);
        if (!dense_covers && (tensors.qkv.manifest->ggml_type != 8U || tensors.z.manifest->ggml_type != 8U))
            throw std::invalid_argument("wave8 GDN input requires Q8_0 qkv and z weights");
        const auto validate = [&](const TensorPair& tensor, std::uint32_t rows, float* target) {
            if (tensor.manifest->ggml_type != 8U) return;
            q8_projection_organization::validate_views(hidden, rows,
                {input.address, input.bytes}, {tensor.runtime->address, tensor.runtime->bytes},
                {reinterpret_cast<std::uintptr_t>(target), static_cast<std::uint64_t>(rows) * sizeof(float)});
        };
        validate(tensors.qkv, conv_width, workspace.qkv);
        validate(tensors.z, value_width, workspace.z);
        validate(tensors.beta, value_heads, workspace.beta);
        validate(tensors.alpha, value_heads, workspace.gate);
    }
    const auto projection_mode = config.wave8_q8_input_projections ? ProjectionMode::prefetch8_wave8
        : config.prefetch_q8_input_projections ? ProjectionMode::prefetch8
        : config.virtual_q8_input_projections ? ProjectionMode::virtual_stream : ProjectionMode::legacy256;

    validate_projection_tensor(
        tensors.out,
        value_width,
        hidden,
        "out");
    // GSQ-RCO (STRIX_GSQ_Q20_A8=1): a typed ssm_out runs the dense A8 segments, not the virtual/wave8 Q8 forms.
    const bool gsq_out = config.dense_a8_projections && tensors.out.manifest->ggml_type != 8U &&
        q8_rows::dense_a8_type(tensors.out.manifest->ggml_type);
    if(config.virtual_q8_output_projection && !gsq_out &&
        (tensors.out.manifest->ggml_type!=8U || value_width%32U!=0U))
        throw std::invalid_argument("virtual Q8 GDN output requires aligned Q8_0 weights");
    if (config.wave8_q8_output_projection && !gsq_out) {
        const auto core_offset = q8_projection_organization::gdn_output_core_offset_bytes(config.model);
        const auto core_address = reinterpret_cast<std::uintptr_t>(workspace.core);
        const auto core_bytes = static_cast<std::uint64_t>(value_width) * sizeof(float);
        if (core_address < temporary.address || core_address - temporary.address != core_offset ||
            core_offset > temporary.bytes || core_bytes > temporary.bytes - core_offset)
            throw std::invalid_argument("wave8 GDN output core workspace mapping differs");
        q8_projection_organization::validate_fp32_bf16_views(value_width, hidden,
            {core_address, core_bytes}, {tensors.out.runtime->address, tensors.out.runtime->bytes},
            {output.address, output.bytes});
    }
    validate_small_tensor(
        tensors.conv,
        static_cast<std::uint64_t>(
            conv_width) *
            kernel,
        "conv");
    validate_small_tensor(
        tensors.dt,
        value_heads,
        "dt");
    validate_small_tensor(
        tensors.a,
        value_heads,
        "A");
    validate_small_tensor(
        tensors.norm,
        head_dim,
        "norm");

    const auto matrix_state =
        resolve_dispatch_gdn_layer_state(
            context,
            config.model,
            SessionStateComponent::
                recurrent_matrix);
    const auto conv_state =
        resolve_dispatch_gdn_layer_state(
            context,
            config.model,
            SessionStateComponent::
                gdn_convolution);
    if (matrix_state.storage !=
            SessionStateStorage::device ||
        conv_state.storage !=
            SessionStateStorage::device ||
        matrix_state.bytes <
            gdn_matrix_state_bytes_per_layer(
                config.model) ||
        conv_state.bytes <
            gdn_conv_state_bytes_per_layer(
                config.model)) {
        throw std::invalid_argument(
            "HIP GDN persistent state bindings are invalid");
    }

    const auto* storage_layout = context.resources && context.resources->session_state_layout
        ? context.resources->session_state_layout : context.session_state_layout;
    if (!storage_layout || storage_layout->recurrent_storage != config.recurrent_storage)
        throw std::invalid_argument("GDN recurrent storage differs from session layout");

    if (const auto* graph = current_hip_launch_context(); graph && graph->graph_capture) {
        if (!config.enable_device_graph_inputs) {
            throw std::invalid_argument("GDN device graph inputs are not explicitly enabled");
        }
        const auto* layout = context.resources && context.resources->session_state_layout
            ? context.resources->session_state_layout : context.session_state_layout;
        if (!layout || graph->context_tokens != layout->context_tokens ||
            graph->context_tokens > config.model.effective_max_positions()) {
            throw std::invalid_argument("GDN graph context exceeds the model position limit");
        }
        GraphOperatorControlRequest request;
        request.execution_identity = context.execution_identity;
        request.context_tokens = graph->context_tokens;
        request.arguments = graph->arguments;
        request.status = graph->status;
        for (const auto* view : {&input, &output}) {
            request.resources.push_back({view->address, view->bytes, view->allocation_generation, 2});
        }
        request.resources.push_back({temporary.address, temporary.bytes, temporary.allocation_generation, 4});
        request.resources.push_back({matrix_state.address, matrix_state.bytes, matrix_state.allocation_generation, 4});
        request.resources.push_back({conv_state.address, conv_state.bytes, conv_state.allocation_generation, 2});
        for (const auto* tensor : {&tensors.qkv, &tensors.z, &tensors.beta, &tensors.alpha,
                                  &tensors.conv, &tensors.dt, &tensors.a, &tensors.norm, &tensors.out}) {
            request.resources.push_back({tensor->runtime->address, tensor->runtime->bytes,
                tensor->runtime->allocation_generation, tensor->manifest->ggml_type == 0 ? 4U : 2U});
        }
        (void)prepare_graph_operator_control(request);
    }
    return {input, output, temporary, workspace, tensors, matrix_state, conv_state,
            hidden, key_heads, value_heads, head_dim, value_width, conv_width, kernel,
            projection_mode, current_hip_launch_context() && current_hip_launch_context()->graph_capture};
}

void preflight_gdn(const ExecutionDispatchContext& context, void* user_data) {
    static_cast<void>(prepare_gdn_launch(context, static_cast<const HipGdnLauncherConfig*>(user_data)));
}

template <GdnRecurrentStorage Storage = GdnRecurrentStorage::key_value>
void emit_gdn_recurrent_layout(const PreparedGdnLaunch& row, HipGdnRecurrentLayout layout,
    const GdnEmitTarget& target) {
    const auto grid = dim3(row.value_heads * (row.head_dim / kValueTile));
    const auto block = dim3(kRecurrentThreads);
#define STRIX_GDN_RECURRENT_LAYOUT(kernel) \
    record_gdn_kernel(target.ledger, reinterpret_cast<const void*>(kernel), grid, block); \
    hipLaunchKernelGGL((kernel), grid, block, 0, target.stream, \
        row.workspace.qkv, row.workspace.beta, row.workspace.gate, \
        reinterpret_cast<float*>(row.matrix_state.address), row.workspace.core, \
        row.key_heads, row.value_heads, row.head_dim, target.graph_status)
    switch (layout) {
    case HipGdnRecurrentLayout::baseline:
    case HipGdnRecurrentLayout::stride32:
        if constexpr (Storage == GdnRecurrentStorage::key_value) {
            STRIX_GDN_RECURRENT_LAYOUT(HIP_KERNEL_NAME(recurrent_layout_diagnostic_kernel<gdn_recurrent_layout_diagnostic::Layout::stride32>));
        } else {
            STRIX_GDN_RECURRENT_LAYOUT(HIP_KERNEL_NAME(recurrent_storage_kernel<gdn_recurrent_layout_diagnostic::Layout::stride32, Storage>));
        }
        break;
    case HipGdnRecurrentLayout::stride33:
        if constexpr (Storage == GdnRecurrentStorage::key_value) {
            STRIX_GDN_RECURRENT_LAYOUT(HIP_KERNEL_NAME(recurrent_layout_diagnostic_kernel<gdn_recurrent_layout_diagnostic::Layout::stride33>));
        } else {
            STRIX_GDN_RECURRENT_LAYOUT(HIP_KERNEL_NAME(recurrent_storage_kernel<gdn_recurrent_layout_diagnostic::Layout::stride33, Storage>));
        }
        break;
    case HipGdnRecurrentLayout::transposed:
        if constexpr (Storage == GdnRecurrentStorage::key_value) {
            STRIX_GDN_RECURRENT_LAYOUT(HIP_KERNEL_NAME(recurrent_layout_diagnostic_kernel<gdn_recurrent_layout_diagnostic::Layout::transposed>));
        } else {
            STRIX_GDN_RECURRENT_LAYOUT(HIP_KERNEL_NAME(recurrent_storage_kernel<gdn_recurrent_layout_diagnostic::Layout::transposed, Storage>));
        }
        break;
    default:
        throw std::invalid_argument("GDN recurrent LDS candidate is invalid");
    }
#undef STRIX_GDN_RECURRENT_LAYOUT
    check_hip(hipGetLastError(), "HIP GDN recurrent LDS layout launch");
}

// Dense A8 R1 input projections: every Q8_0 matrix of qkv, z, beta, alpha in one q8_rows
// launch (the rows contract of the window and prompt sites, so R1 rows match them).
void launch_gdn_dense_a8_inputs(const PreparedGdnLaunch& prepared, hipStream_t stream) {
    GdnInputRowsSite<1U> in{};
    const auto add = [&](const TensorPair& tensor, std::uint32_t outputs, float* target) {
        if (!q8_rows::dense_a8_type(tensor.manifest->ggml_type)) return;
        const auto m = in.segments.count++;
        in.segments.types[m] = q8_rows::segment_type(tensor.manifest->ggml_type);
        in.segments.weights[m] = reinterpret_cast<const unsigned char*>(tensor.runtime->address);
        in.segments.first[m + 1U] = in.segments.first[m] + outputs;
        in.output[m][0] = target;
    };
    const auto& tensors = prepared.tensors;
    add(tensors.qkv, prepared.conv_width, prepared.workspace.qkv);
    add(tensors.z, prepared.value_width, prepared.workspace.z);
    add(tensors.beta, prepared.value_heads, prepared.workspace.beta);
    add(tensors.alpha, prepared.value_heads, prepared.workspace.gate);
    if (!in.segments.count) return;
    in.input[0] = reinterpret_cast<const void*>(prepared.input.address);
    in.width = prepared.hidden; in.outputs = in.segments.first[in.segments.count];
    q8_rows::launch(in, true, stream);
    check_hip(hipGetLastError(), "HIP GDN dense A8 input projection launch");
}

void emit_gdn_sequence(const PreparedGdnLaunch& prepared, const HipGdnLauncherConfig& config,
    const GdnEmitTarget& target) {
    const auto& input = prepared.input; const auto& output = prepared.output;
    const auto& workspace = prepared.workspace; const auto& tensors = prepared.tensors;
    const auto& matrix_state = prepared.matrix_state; const auto& conv_state = prepared.conv_state;
    const auto hidden = prepared.hidden, key_heads = prepared.key_heads, value_heads = prepared.value_heads;
    const auto head_dim = prepared.head_dim, value_width = prepared.value_width, conv_width = prepared.conv_width;
    const auto kernel = prepared.kernel;
    const auto projection_mode = prepared.projection_mode;
    const auto* hidden_input = reinterpret_cast<const std::uint16_t*>(input.address);
    if (prepared.graph_capture) {
        hipLaunchKernelGGL(validate_gdn_graph_position, dim3(1), dim3(1), 0, target.stream,
            target.graph_position, target.context_tokens, target.graph_status);
        check_hip(hipGetLastError(), "HIP GDN device position validation");
        hipLaunchKernelGGL(validate_gdn_graph_input, dim3(32), dim3(kProjectionThreads), 0, target.stream,
            hidden_input, hidden, target.graph_status, 4U);
        check_hip(hipGetLastError(), "HIP GDN finite input validation");
    }

    const bool dense = config.dense_a8_projections;
    if (dense) {
        if (prepared.graph_capture || target.ledger || target.graph_status)
            throw std::invalid_argument("HIP GDN dense A8 projections run eager outside islands and graphs");
        launch_gdn_dense_a8_inputs(prepared, target.stream);
    }
    // Under dense A8 the Q8_0 (and GSQ dense) matrices ran above; other formats keep their R1 kernels.
    const auto keep = [&](const TensorPair& tensor) { return !dense || !q8_rows::dense_a8_type(tensor.manifest->ggml_type); };
    if (keep(tensors.qkv)) launch_projection_by_kind(
        weight_kind(
            tensors.qkv.manifest->
                ggml_type),
        hidden_input,
        *tensors.qkv.runtime,
        hidden,
        conv_width,
        workspace.qkv, projection_mode, target.stream, target.ledger);
    if (keep(tensors.z)) launch_projection_by_kind(
        weight_kind(
            tensors.z.manifest->
                ggml_type),
        hidden_input,
        *tensors.z.runtime,
        hidden,
        value_width,
        workspace.z, projection_mode, target.stream, target.ledger);
    if (keep(tensors.beta)) launch_projection_by_kind(
        weight_kind(
            tensors.beta.manifest->
                ggml_type),
        hidden_input,
        *tensors.beta.runtime,
        hidden,
        value_heads,
        workspace.beta, projection_mode, target.stream, target.ledger);
    if (keep(tensors.alpha)) launch_projection_by_kind(
        weight_kind(
            tensors.alpha.manifest->
                ggml_type),
        hidden_input,
        *tensors.alpha.runtime,
        hidden,
        value_heads,
        workspace.gate, projection_mode, target.stream, target.ledger);

    if (target.graph_status != nullptr) {
        hipLaunchKernelGGL(validate_gdn_graph_values, dim3(32), dim3(kProjectionThreads), 0,
            target.stream, workspace.qkv,
            static_cast<std::uint64_t>(conv_width) + value_width + value_heads * 2ULL, target.graph_status);
        check_hip(hipGetLastError(), "HIP GDN finite projection validation");
    }
    const auto conv_blocks =
        (conv_width +
         kProjectionThreads - 1U) /
        kProjectionThreads;
    record_gdn_kernel(target.ledger, reinterpret_cast<const void*>(conv_update_kernel),
        dim3(conv_blocks), dim3(kProjectionThreads));
    hipLaunchKernelGGL(
        conv_update_kernel,
        dim3(conv_blocks),
        dim3(kProjectionThreads),
        0,
        target.stream,
        workspace.qkv,
        reinterpret_cast<
            std::uint16_t*>(
                conv_state.address),
        reinterpret_cast<const void*>(
            tensors.conv.runtime->
                address),
        small_weight_kind(
            tensors.conv.manifest->
                ggml_type),
        conv_width,
        kernel, target.graph_status);
    check_hip(
        hipGetLastError(),
        "HIP GDN causal convolution launch");

    const auto parameter_blocks =
        (value_heads +
         kProjectionThreads - 1U) /
        kProjectionThreads;
    record_gdn_kernel(target.ledger, reinterpret_cast<const void*>(transform_beta_gate_kernel),
        dim3(parameter_blocks), dim3(kProjectionThreads));
    hipLaunchKernelGGL(
        transform_beta_gate_kernel,
        dim3(parameter_blocks),
        dim3(kProjectionThreads),
        0,
        target.stream,
        workspace.beta,
        workspace.gate,
        reinterpret_cast<const void*>(
            tensors.dt.runtime->
                address),
        small_weight_kind(
            tensors.dt.manifest->
                ggml_type),
        reinterpret_cast<const void*>(
            tensors.a.runtime->
                address),
        small_weight_kind(
            tensors.a.manifest->
                ggml_type),
        value_heads);
    check_hip(
        hipGetLastError(),
        "HIP GDN beta/gate transform launch");

    record_gdn_kernel(target.ledger, reinterpret_cast<const void*>(normalize_qk_kernel),
        dim3(key_heads * 2U), dim3(kRecurrentThreads));
    hipLaunchKernelGGL(
        normalize_qk_kernel,
        dim3(key_heads * 2U),
        dim3(kRecurrentThreads),
        0,
        target.stream,
        workspace.qkv,
        key_heads,
        head_dim,
        config.model.
            gdn_qk_l2_epsilon, target.graph_status);
    check_hip(
        hipGetLastError(),
        "HIP GDN q/k normalization launch");

    if (target.graph_status != nullptr) {
        hipLaunchKernelGGL(validate_gdn_graph_values, dim3(32), dim3(kProjectionThreads), 0,
            target.stream, workspace.qkv,
            static_cast<std::uint64_t>(conv_width) + value_width + value_heads * 2ULL, target.graph_status);
        check_hip(hipGetLastError(), "HIP GDN finite recurrent input validation");
    }
    const auto tiles =
        head_dim /
        kValueTile;
    if (config.recurrent_layout == HipGdnRecurrentLayout::baseline &&
        config.recurrent_storage == GdnRecurrentStorage::key_value) {
        record_gdn_kernel(target.ledger, reinterpret_cast<const void*>(recurrent_step_kernel),
            dim3(value_heads * tiles), dim3(kRecurrentThreads));
        hipLaunchKernelGGL(
            recurrent_step_kernel,
            dim3(value_heads * tiles),
            dim3(kRecurrentThreads),
            0,
            target.stream,
            workspace.qkv,
            workspace.beta,
            workspace.gate,
            reinterpret_cast<float*>(
                matrix_state.address),
            workspace.core,
            key_heads,
            value_heads,
            head_dim, target.graph_status);
        check_hip(
            hipGetLastError(),
            "HIP GDN recurrent update launch");
    } else {
        if (config.recurrent_storage == GdnRecurrentStorage::value_key)
            emit_gdn_recurrent_layout<GdnRecurrentStorage::value_key>(prepared, config.recurrent_layout, target);
        else
            emit_gdn_recurrent_layout(prepared, config.recurrent_layout, target);
    }
    if (target.graph_status != nullptr) {
        hipLaunchKernelGGL(validate_gdn_graph_values, dim3(32), dim3(kProjectionThreads), 0,
            target.stream, workspace.core, value_width, target.graph_status);
        check_hip(hipGetLastError(), "HIP GDN finite recurrent output validation");
    }

    record_gdn_kernel(target.ledger, reinterpret_cast<const void*>(gated_rms_norm_kernel),
        dim3(value_heads), dim3(kRecurrentThreads));
    hipLaunchKernelGGL(
        gated_rms_norm_kernel,
        dim3(value_heads),
        dim3(kRecurrentThreads),
        0,
        target.stream,
        workspace.core,
        workspace.z,
        reinterpret_cast<const void*>(
            tensors.norm.runtime->
                address),
        small_weight_kind(
            tensors.norm.manifest->
                ggml_type),
        value_heads,
        head_dim,
        config.model.
            rms_norm_epsilon, target.graph_status);
    check_hip(
        hipGetLastError(),
        "HIP GDN gated RMSNorm launch");

    if (!keep(tensors.out)) {
        GdnOutputRowsSite<1U> out{};
        out.segments = q8_rows::segments({{reinterpret_cast<const void*>(tensors.out.runtime->address), hidden}});
        out.segments.types[0] = q8_rows::segment_type(tensors.out.manifest->ggml_type);
        out.width = value_width; out.outputs = hidden;
        out.input[0] = workspace.core;
        out.output[0] = reinterpret_cast<std::uint16_t*>(output.address);
        q8_rows::launch(out, true, target.stream);
        check_hip(hipGetLastError(), "HIP GDN dense A8 output projection launch");
    } else launch_output_by_kind(
        weight_kind(
            tensors.out.manifest->
                ggml_type),
        workspace.core,
        *tensors.out.runtime,
        value_width,
        hidden,
        reinterpret_cast<
            std::uint16_t*>(
                output.address),config.virtual_q8_output_projection,config.wave8_q8_output_projection, target.stream, target.ledger);
    if (target.graph_status != nullptr) {
        hipLaunchKernelGGL(validate_gdn_graph_input, dim3(32), dim3(kProjectionThreads), 0,
            target.stream, reinterpret_cast<const std::uint16_t*>(output.address), hidden,
            target.graph_status, 8U);
        check_hip(hipGetLastError(), "HIP GDN finite final output validation");
    }
}

void launch_gdn(const ExecutionDispatchContext& context, void* user_data) {
    const auto prepared = prepare_gdn_launch(context, static_cast<const HipGdnLauncherConfig*>(user_data));
#if defined(STRIX_HAS_EXPERIMENTAL_DENSE_TRELLIS)
    HipDenseTrellisScope arithmetic(static_cast<const HipGdnLauncherConfig*>(user_data)->dense_trellis_projections);
#endif
    const auto* graph = current_hip_launch_context();
    emit_gdn_sequence(prepared, *static_cast<const HipGdnLauncherConfig*>(user_data),
        {hip_execution_stream(), hip_graph_status(), hip_graph_position(),
         graph ? graph->context_tokens : 0, nullptr});
}

bool has_gdn_recurrent_layout_marker(std::string_view id) noexcept {
    return id.find(".lds32-v1") != std::string_view::npos ||
        id.find(".lds33-v1") != std::string_view::npos ||
        id.find(".lds-transposed-v1") != std::string_view::npos;
}

std::string kernel_id(
    const ExecutionBindingRequest& request,
    const HipGdnLauncherConfig& config) {
    return
        "hip.gdn.bf16.h" +
        std::to_string(
            config.model.hidden_size) +
        ".kh" +
        std::to_string(
            config.model.gdn_key_heads) +
        ".vh" +
        std::to_string(
            config.model.gdn_value_heads) +
        ".d" +
        std::to_string(
            config.model.gdn_key_head_dim) +
        ".r" +
        std::to_string(
            request.logical_rows) +
        ".c" +
        std::to_string(
            request.request_concurrency) + (request.device_graph_inputs ? ".device" : ".scalar") +
        std::string(gdn_recurrent_layout_marker(config.recurrent_layout)) +
        std::string(gdn_recurrent_storage_marker(config.recurrent_storage)) +
        (config.wave8_q8_input_projections ? std::string(q8_projection_organization::gdn_wave8_marker) : "") +
        (config.wave8_q8_output_projection ? std::string(q8_projection_organization::gdn_output_wave8_marker) : "") +
        (config.dense_a8_projections ? std::string(q8_projection_organization::dense_a8_marker) : "") +
        (config.dense_trellis_projections ? ".trellis-g1-fp32-s4-v1" : "") +
        (config.prefetch_q8_input_projections ? ".prefetch8" : "") +
        (config.virtual_q8_input_projections ? ".virtual256.q8-input" : "") +
        (config.virtual_q8_output_projection ? ".virtual256.q8-output" : "");
}

std::uint64_t implementation_identity(
    const HipGdnLauncherConfig& config) noexcept {
    std::uint64_t hash =
        fnv_offset;
    hash_u64(hash, kLauncherTag);
    hash_u64(hash, config.enable_device_graph_inputs);
    hash_u64(hash, config.model.max_position_embeddings);
    hash_u64(
        hash,
        config.model.hidden_size);
    hash_u64(
        hash,
        config.model.gdn_key_heads);
    hash_u64(
        hash,
        config.model.gdn_key_head_dim);
    hash_u64(
        hash,
        config.model.gdn_value_heads);
    hash_u64(
        hash,
        config.model.gdn_value_head_dim);
    hash_u64(
        hash,
        config.model.gdn_conv_kernel);
    hash_float(
        hash,
        config.model.rms_norm_epsilon);
    hash_float(
        hash,
        config.model.gdn_qk_l2_epsilon);
    hash_u64(
        hash,
        config.model.gdn_output_gate_sigmoid
            ? 1ULL
            : 0ULL);
    hash_u64(
        hash,
        config.model.gdn_qk_l2norm
            ? 1ULL
            : 0ULL);
    hash_u64(
        hash,
        config.model.gdn_conv_silu
            ? 1ULL
            : 0ULL);
    if (config.virtual_q8_input_projections) hash_u64(hash, 0x47444e5138563235ULL);
    if (config.virtual_q8_output_projection) hash_u64(hash, 0x47444e514f555431ULL);
    if (config.prefetch_q8_input_projections) hash_u64(hash, 0x47444e5138504631ULL);
    if (config.wave8_q8_input_projections) hash_u64(hash, 0x47444e5138574156ULL);
    if (config.wave8_q8_output_projection) hash_u64(hash, 0x47444e514f573831ULL);
    if (config.dense_a8_projections) hash_u64(hash, 0x47444e4441385631ULL); // GDNDA8V1
    if (config.dense_trellis_projections) hash_u64(hash, 0x544c473146503453ULL); // TLG1FP4S
    if (config.recurrent_layout != HipGdnRecurrentLayout::baseline) {
        hash_u64(hash, 0x47444e4c44533031ULL); // GDN LDS organization version 1.
        hash_u64(hash, static_cast<std::uint32_t>(config.recurrent_layout));
    }
    if (config.recurrent_storage != GdnRecurrentStorage::key_value) {
        hash_u64(hash, 0x47444e53544f5231ULL);
        hash_u64(hash, static_cast<std::uint32_t>(config.recurrent_storage));
    }
    return hash;
}


// R2 consumes the same complete pure host admission as the R1 factory.
using GdnKnownWindowRow = PreparedGdnLaunch;
enum class Format { all_q8, q8_large_f32_gates };

void launch_gdn_known_window_core(const GdnKnownWindowRow& row,
    const HipGdnLauncherConfig& config) {
    const auto& workspace = row.workspace; const auto& tensors = row.tensors;
    const auto& matrix_state = row.matrix_state; const auto& conv_state = row.conv_state;
    const auto key_heads = row.key_heads, value_heads = row.value_heads, head_dim = row.head_dim;
    const auto value_width = row.value_width, conv_width = row.conv_width, kernel = row.kernel;
    const auto conv_blocks =
        (conv_width +
         kProjectionThreads - 1U) /
        kProjectionThreads;
    hipLaunchKernelGGL(
        conv_update_kernel,
        dim3(conv_blocks),
        dim3(kProjectionThreads),
        0,
        hip_execution_stream(),
        workspace.qkv,
        reinterpret_cast<
            std::uint16_t*>(
                conv_state.address),
        reinterpret_cast<const void*>(
            tensors.conv.runtime->
                address),
        small_weight_kind(
            tensors.conv.manifest->
                ggml_type),
        conv_width,
        kernel, hip_graph_status());
    check_hip(
        hipGetLastError(),
        "HIP GDN causal convolution launch");

    const auto parameter_blocks =
        (value_heads +
         kProjectionThreads - 1U) /
        kProjectionThreads;
    hipLaunchKernelGGL(
        transform_beta_gate_kernel,
        dim3(parameter_blocks),
        dim3(kProjectionThreads),
        0,
        hip_execution_stream(),
        workspace.beta,
        workspace.gate,
        reinterpret_cast<const void*>(
            tensors.dt.runtime->
                address),
        small_weight_kind(
            tensors.dt.manifest->
                ggml_type),
        reinterpret_cast<const void*>(
            tensors.a.runtime->
                address),
        small_weight_kind(
            tensors.a.manifest->
                ggml_type),
        value_heads);
    check_hip(
        hipGetLastError(),
        "HIP GDN beta/gate transform launch");

    hipLaunchKernelGGL(
        normalize_qk_kernel,
        dim3(key_heads * 2U),
        dim3(kRecurrentThreads),
        0,
        hip_execution_stream(),
        workspace.qkv,
        key_heads,
        head_dim,
        config.model.
            gdn_qk_l2_epsilon, hip_graph_status());
    check_hip(
        hipGetLastError(),
        "HIP GDN q/k normalization launch");

    if (hip_graph_status() != nullptr) {
        hipLaunchKernelGGL(validate_gdn_graph_values, dim3(32), dim3(kProjectionThreads), 0,
            hip_execution_stream(), workspace.qkv,
            static_cast<std::uint64_t>(conv_width) + value_width + value_heads * 2ULL, hip_graph_status());
        check_hip(hipGetLastError(), "HIP GDN finite recurrent input validation");
    }
    const auto tiles =
        head_dim /
        kValueTile;
    hipLaunchKernelGGL(
        recurrent_step_kernel,
        dim3(value_heads * tiles),
        dim3(kRecurrentThreads),
        0,
        hip_execution_stream(),
        workspace.qkv,
        workspace.beta,
        workspace.gate,
        reinterpret_cast<float*>(
            matrix_state.address),
        workspace.core,
        key_heads,
        value_heads,
        head_dim, hip_graph_status());
    check_hip(
        hipGetLastError(),
        "HIP GDN recurrent update launch");
    if (hip_graph_status() != nullptr) {
        hipLaunchKernelGGL(validate_gdn_graph_values, dim3(32), dim3(kProjectionThreads), 0,
            hip_execution_stream(), workspace.core, value_width, hip_graph_status());
        check_hip(hipGetLastError(), "HIP GDN finite recurrent output validation");
    }

    hipLaunchKernelGGL(
        gated_rms_norm_kernel,
        dim3(value_heads),
        dim3(kRecurrentThreads),
        0,
        hip_execution_stream(),
        workspace.core,
        workspace.z,
        reinterpret_cast<const void*>(
            tensors.norm.runtime->
                address),
        small_weight_kind(
            tensors.norm.manifest->
                ggml_type),
        value_heads,
        head_dim,
        config.model.
            rms_norm_epsilon, hip_graph_status());
    check_hip(
        hipGetLastError(),
        "HIP GDN gated RMSNorm launch");

}


struct GdnKnownWindowRange { std::uintptr_t address; std::uint64_t bytes; };
void validate_gdn_known_window_range(GdnKnownWindowRange view, std::uint64_t alignment) {
    if (!view.address || view.address % alignment || !view.bytes ||
        view.bytes > std::numeric_limits<std::uintptr_t>::max() - view.address)
        throw std::invalid_argument("GDN known-window owner alignment or extent is invalid");
}
void require_gdn_known_window_disjoint(GdnKnownWindowRange a, GdnKnownWindowRange b) {
    if (a.address < b.address + b.bytes && b.address < a.address + a.bytes)
        throw std::invalid_argument("GDN known-window writable or row views alias");
}
std::array<GdnKnownWindowRow, 2> prepare_gdn_known_window_rows(
    std::span<const ExecutionDispatchContext, 2> rows, const HipGdnLauncherConfig& config,
    Format format) {
    validate_config(config);
    if (config.recurrent_layout != HipGdnRecurrentLayout::baseline)
        throw std::invalid_argument("GDN known-window R2 requires the baseline recurrent LDS layout");
    if (config.recurrent_storage != GdnRecurrentStorage::key_value)
        throw std::invalid_argument("GDN known-window R2 requires key-value recurrent storage");
    const auto& model = config.model;
    if (!config.virtual_q8_input_projections || !config.prefetch_q8_input_projections ||
        !config.wave8_q8_input_projections || !config.virtual_q8_output_projection ||
        !config.wave8_q8_output_projection || config.enable_device_graph_inputs ||
        model.hidden_size != 2560U || model.hidden_layers != 48U || model.full_attention_interval != 4U ||
        model.gdn_key_heads != 16U || model.gdn_value_heads != 48U ||
        model.gdn_key_head_dim != 128U || model.gdn_value_head_dim != 128U || model.gdn_conv_kernel != 4U)
        throw std::invalid_argument("GDN known-window R2 requires canonical wave8 Q8 input and output");
    const auto* scope = current_hip_launch_context();
    if (!scope || scope->graph_capture || !hip_execution_stream() || hip_graph_status() || hip_graph_position())
        throw std::invalid_argument("GDN known-window R2 requires one owned eager stream");
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    check_hip(hipStreamIsCapturing(hip_execution_stream(), &capture), "query GDN known-window R2 capture");
    if (capture != hipStreamCaptureStatusNone)
        throw std::invalid_argument("GDN known-window R2 cannot be captured");
    ExecutionBindingRequest selected;
    const auto expected_kernel = kernel_id(selected, config);
    const auto states = PersistentState::recurrent_matrix | PersistentState::convolution_history;
    for (const auto& context : rows) {
        if (!context.execution_identity || !context.arena || !context.resources || !context.resources->scratch ||
            (context.memory && context.memory != context.resources->scratch) ||
            context.prepared_tensor_views || context.prepared_arena_views ||
            !context.resources->session_state_layout || !context.resources->session_state_memory ||
            (context.session_state_layout && context.session_state_layout != context.resources->session_state_layout) ||
            (context.session_state_memory && context.session_state_memory != context.resources->session_state_memory))
            throw std::invalid_argument("GDN known-window R2 requires its current window resource bindings");
        validate_execution_memory_binding(context.execution_identity, *context.arena, *context.resources->scratch);
        if (!context.kernel_id || *context.kernel_id != expected_kernel ||
            context.reads_state != states || context.writes_state != states)
            throw std::invalid_argument("GDN known-window R2 selected serial variant or state flags differ");
        if (context.position >= context.resources->session_state_layout->context_tokens ||
            context.position >= model.effective_max_positions())
            throw std::invalid_argument("GDN known-window R2 position exceeds capacity");
        // Validate whole supplied scratch views before workspace pointer arithmetic.
        if (context.reads.size() != 1U || context.writes.size() != 1U || context.temporaries.size() != 1U)
            throw std::invalid_argument("GDN known-window R2 requires exact row IO");
        for (const auto view : {resolve_dispatch_read(context, 0), resolve_dispatch_write(context, 0),
                               resolve_dispatch_temporary(context, 0)}) {
            validate_gdn_known_window_range({view.address, view.bytes}, 16U);
            const auto& owner = *context.resources->scratch;
            if (view.address < owner.arena_base || view.address - owner.arena_base > owner.arena_bytes ||
                view.bytes > owner.arena_bytes - (view.address - owner.arena_base))
                throw std::invalid_argument("GDN known-window row view exceeds its physical owner");
        }
    }
    const auto& a = rows[0]; const auto& b = rows[1];
    if (a.execution_identity != b.execution_identity || a.step_id != b.step_id || a.op != b.op || a.layer != b.layer ||
        a.reads[0] != b.reads[0] || a.writes[0] != b.writes[0] || a.temporaries[0] != b.temporaries[0] ||
        a.resources->scratch != b.resources->scratch ||
        a.resources->model_manifest != b.resources->model_manifest ||
        a.resources->model_weights != b.resources->model_weights || a.resources->tensor_access != b.resources->tensor_access ||
        a.resources->session_state_layout != b.resources->session_state_layout ||
        a.resources->session_state_memory != b.resources->session_state_memory ||
        a.position == std::numeric_limits<std::uint64_t>::max() || b.position != a.position + 1U)
        throw std::invalid_argument("GDN known-window R2 rows do not share one ordered execution");
    // Every incumbent row-level deep check completes before any kernel submission.
    std::array<GdnKnownWindowRow, 2> result{
        prepare_gdn_launch(a, &config), prepare_gdn_launch(b, &config)};
    const auto same_state = [](const SessionStateAddress& x, const SessionStateAddress& y) {
        return x.component == y.component && x.storage == y.storage && x.address == y.address &&
            x.bytes == y.bytes && x.allocation_generation == y.allocation_generation && x.provisional == y.provisional;
    };
    if (!same_state(result[0].matrix_state, result[1].matrix_state) ||
        !same_state(result[0].conv_state, result[1].conv_state))
        throw std::invalid_argument("GDN known-window R2 causal states differ between rows");
    const std::array<TensorPair GdnTensors::*, 9> members{
        &GdnTensors::qkv, &GdnTensors::z, &GdnTensors::beta, &GdnTensors::alpha, &GdnTensors::conv,
        &GdnTensors::dt, &GdnTensors::a, &GdnTensors::norm, &GdnTensors::out};
    std::array<GdnKnownWindowRange, 11> readonly{};
    readonly[0] = {result[0].input.address, result[0].input.bytes};
    readonly[1] = {result[1].input.address, result[1].input.bytes};
    // q8_rows reads BF16 rows and FP32 core rows as 16-byte vectors.
    for (const auto& row : result)
        if (row.input.address % 16U || reinterpret_cast<std::uintptr_t>(row.workspace.core) % 16U)
            throw std::invalid_argument("GDN known-window rows require 16-byte aligned input and core rows");
    require_gdn_known_window_disjoint(readonly[0], readonly[1]);
    for (std::size_t i = 0; i < members.size(); ++i) {
        const auto& x = result[0].tensors.*members[i]; const auto& y = result[1].tensors.*members[i];
        if (x.manifest != y.manifest || x.runtime != y.runtime)
            throw std::invalid_argument("GDN known-window R2 weights differ between rows");
        readonly[i + 2U] = {x.runtime->address, x.runtime->bytes};
        validate_gdn_known_window_range(readonly[i + 2U], x.manifest->ggml_type == 0U ? 4U : 2U);
    }
    if (format == Format::all_q8) {
        for (const auto member : {&GdnTensors::qkv, &GdnTensors::z, &GdnTensors::beta, &GdnTensors::alpha, &GdnTensors::out})
            if ((result[0].tensors.*member).manifest->ggml_type != 8U)
                throw std::invalid_argument("GDN known-window R2 requires all five Q8_0 projections");
    } else {
        const auto& tensors = result[0].tensors;
        const auto is_f32_gate = [](const TensorPair& tensor) {
            const auto& manifest = *tensor.manifest;
            return manifest.ggml_type == 0U && manifest.dimensions.size() == 2U &&
                manifest.dimensions[0] == 2560U && manifest.dimensions[1] == 48U &&
                manifest.byte_size == 491520ULL;
        };
        // GSQ-RCO (STRIX_GSQ_Q20_A8=1): typed qkv/z/out run as typed dense A8 segments of the same rows sites.
        const auto matrix_ok = [&](const TensorPair& tensor) {
            const auto type = tensor.manifest->ggml_type;
            return type == 8U || (config.dense_a8_projections && q8_rows::dense_a8_type(type));
        };
        if (!matrix_ok(tensors.qkv) || !matrix_ok(tensors.z) ||
            !matrix_ok(tensors.out) || !is_f32_gate(tensors.beta) || !is_f32_gate(tensors.alpha))
            throw std::invalid_argument("GDN known-window R2 F32-gate variant requires Q8_0 qkv/z/out and F32 beta/alpha");
    }
    const std::array<GdnKnownWindowRange, 6> writable{{
        {result[0].output.address, result[0].output.bytes}, {result[1].output.address, result[1].output.bytes},
        {result[0].temporary.address, result[0].temporary.bytes}, {result[1].temporary.address, result[1].temporary.bytes},
        {result[0].matrix_state.address, result[0].matrix_state.bytes},
        {result[0].conv_state.address, result[0].conv_state.bytes}}};
    for (std::size_t i = 0; i < writable.size(); ++i) {
        validate_gdn_known_window_range(writable[i], i == 5U ? 2U : 4U);
        for (const auto input : readonly) require_gdn_known_window_disjoint(writable[i], input);
        for (std::size_t j = 0; j < i; ++j) require_gdn_known_window_disjoint(writable[i], writable[j]);
    }
    check_hip(hipPeekAtLastError(), "GDN known-window R2 pre-enqueue stale error");
    return result;
}

void enqueue_gdn_known_window_inputs(const std::array<GdnKnownWindowRow, 2>& prepared, Format format,
    const HipGdnLauncherConfig& config) {
    const auto& a = prepared[0];
    const bool q8_gates = format == Format::all_q8;
    GdnInputRowsSite<2U> in{};
    const auto weights = [](const TensorPair& tensor) { return reinterpret_cast<const void*>(tensor.runtime->address); };
    in.segments = q8_gates ? q8_rows::segments({{weights(a.tensors.qkv), a.conv_width}, {weights(a.tensors.z), a.value_width},
        {weights(a.tensors.beta), a.value_heads}, {weights(a.tensors.alpha), a.value_heads}}) :
        q8_rows::segments({{weights(a.tensors.qkv), a.conv_width}, {weights(a.tensors.z), a.value_width}});
    if (!q8_gates) {
        in.segments.types[0] = q8_rows::segment_type(a.tensors.qkv.manifest->ggml_type);
        in.segments.types[1] = q8_rows::segment_type(a.tensors.z.manifest->ggml_type);
    }
    in.width = a.hidden; in.outputs = in.segments.first[in.segments.count];
    for (unsigned r = 0; r < 2U; ++r) {
        const auto& w = prepared[r].workspace;
        in.input[r] = reinterpret_cast<const void*>(prepared[r].input.address);
        in.output[0][r] = w.qkv; in.output[1][r] = w.z; in.output[2][r] = w.beta; in.output[3][r] = w.gate;
    }
    q8_rows::launch(in, config.dense_a8_projections, hip_execution_stream());
    check_hip(hipGetLastError(), "GDN known-window R2 input projection launch");
    if (!q8_gates) {
        for (const auto& row : prepared) {
            const auto* input = reinterpret_cast<const std::uint16_t*>(row.input.address);
            launch_projection_by_kind(weight_kind(row.tensors.beta.manifest->ggml_type), input,
                *row.tensors.beta.runtime, row.hidden, row.value_heads, row.workspace.beta,
                row.projection_mode, hip_execution_stream(), nullptr);
            launch_projection_by_kind(weight_kind(row.tensors.alpha.manifest->ggml_type), input,
                *row.tensors.alpha.runtime, row.hidden, row.value_heads, row.workspace.gate,
                row.projection_mode, hip_execution_stream(), nullptr);
        }
    }
}

void enqueue_gdn_known_window_completion(const std::array<GdnKnownWindowRow, 2>& prepared,
    const HipGdnLauncherConfig& config) {
    const auto& a = prepared[0]; const auto& b = prepared[1];
    // Causality is unchanged: the same stream finishes row0's state update
    // before row1 reads that matrix/convolution state. Workspace rows are disjoint.
    launch_gdn_known_window_core(a, config);
    launch_gdn_known_window_core(b, config);
    GdnOutputRowsSite<2U> out{};
    out.segments = q8_rows::segments({{reinterpret_cast<const void*>(a.tensors.out.runtime->address), a.hidden}});
    out.segments.types[0] = q8_rows::segment_type(a.tensors.out.manifest->ggml_type);
    out.width = a.value_width; out.outputs = a.hidden;
    for (unsigned r = 0; r < 2U; ++r) {
        out.input[r] = prepared[r].workspace.core;
        out.output[r] = reinterpret_cast<std::uint16_t*>(prepared[r].output.address);
    }
    q8_rows::launch(out, config.dense_a8_projections, hip_execution_stream());
    check_hip(hipGetLastError(), "GDN known-window R2 output projection launch");
}

template <unsigned Rows>
std::array<GdnKnownWindowRow, Rows> prepare_gdn_wide_rows(
    std::span<const ExecutionDispatchContext> rows, const HipGdnLauncherConfig& config) {
    static_assert(Rows == 4U || Rows == 8U);
    if (rows.size() != Rows) throw std::invalid_argument("GDN wide-window row count differs");
    std::array<GdnKnownWindowRow, Rows> result{};
    // Reuse every incumbent deep row/format/state check, without enqueueing.
    for (unsigned r = 0; r < Rows; r += 2U) {
        const auto pair = prepare_gdn_known_window_rows(
            std::span<const ExecutionDispatchContext, 2>(rows.data() + r, 2), config, Format::q8_large_f32_gates);
        result[r] = pair[0]; result[r + 1U] = pair[1];
    }
    const auto& first = rows[0];
    if (first.position > std::numeric_limits<std::uint64_t>::max() - (Rows - 1U))
        throw std::invalid_argument("GDN wide-window position overflows");
    if (config.dense_a8_projections)
        for (const auto width : {result[0].hidden, result[0].value_width})
            if (!q8_rows::dense_a8_width(width)) throw std::invalid_argument("GDN wide-window dense A8 width is not admitted");
    const auto same_state = [](const SessionStateAddress& a, const SessionStateAddress& b) {
        return a.component == b.component && a.storage == b.storage && a.address == b.address &&
            a.bytes == b.bytes && a.allocation_generation == b.allocation_generation && a.provisional == b.provisional;
    };
    const std::array<TensorPair GdnTensors::*, 9> members{
        &GdnTensors::qkv, &GdnTensors::z, &GdnTensors::beta, &GdnTensors::alpha, &GdnTensors::conv,
        &GdnTensors::dt, &GdnTensors::a, &GdnTensors::norm, &GdnTensors::out};
    std::array<GdnKnownWindowRange, Rows + 9U> readonly{};
    std::array<GdnKnownWindowRange, 2U * Rows + 2U> writable{};
    for (unsigned r = 0; r < Rows; ++r) {
        const auto& context = rows[r];
        if (context.execution_identity != first.execution_identity || context.step_id != first.step_id ||
            context.op != first.op || context.layer != first.layer || context.position != first.position + r ||
            context.reads[0] != first.reads[0] || context.writes[0] != first.writes[0] ||
            context.temporaries[0] != first.temporaries[0] ||
            context.resources->scratch != first.resources->scratch ||
            context.resources->model_manifest != first.resources->model_manifest ||
            context.resources->model_weights != first.resources->model_weights ||
            context.resources->tensor_access != first.resources->tensor_access ||
            context.resources->session_state_layout != first.resources->session_state_layout ||
            context.resources->session_state_memory != first.resources->session_state_memory ||
            !same_state(result[r].matrix_state, result[0].matrix_state) ||
            !same_state(result[r].conv_state, result[0].conv_state))
            throw std::invalid_argument("GDN wide-window rows do not retain one ordered session binding");
        readonly[r] = {result[r].input.address, result[r].input.bytes};
        for (unsigned prior = 0; prior < r; ++prior)
            require_gdn_known_window_disjoint(readonly[r], readonly[prior]);
        for (const auto member : members) {
            const auto& a = result[0].tensors.*member; const auto& b = result[r].tensors.*member;
            if (a.manifest != b.manifest || a.runtime != b.runtime)
                throw std::invalid_argument("GDN wide-window projection owners differ between rows");
        }
        writable[2U * r] = {result[r].output.address, result[r].output.bytes};
        writable[2U * r + 1U] = {result[r].temporary.address, result[r].temporary.bytes};
    }
    for (unsigned i = 0; i < members.size(); ++i) {
        const auto& tensor = result[0].tensors.*members[i];
        readonly[Rows + i] = {tensor.runtime->address, tensor.runtime->bytes};
    }
    writable[2U * Rows] = {result[0].matrix_state.address, result[0].matrix_state.bytes};
    writable[2U * Rows + 1U] = {result[0].conv_state.address, result[0].conv_state.bytes};
    // Pair admission alone cannot establish disjointness across pair boundaries.
    for (unsigned i = 0; i < writable.size(); ++i) {
        for (const auto read : readonly) require_gdn_known_window_disjoint(writable[i], read);
        for (unsigned j = 0; j < i; ++j) require_gdn_known_window_disjoint(writable[i], writable[j]);
    }
    return result;
}

template <unsigned Rows>
void enqueue_gdn_wide_rows(const std::array<GdnKnownWindowRow, Rows>& prepared,
    const HipGdnLauncherConfig& config) {
    const auto& first = prepared[0];
    // qkv and z share one launch; the output projection reads every row's core.
    GdnInputRowsSite<Rows> in{};
    in.segments = q8_rows::segments({{reinterpret_cast<const void*>(first.tensors.qkv.runtime->address), first.conv_width},
        {reinterpret_cast<const void*>(first.tensors.z.runtime->address), first.value_width}});
    in.segments.types[0] = q8_rows::segment_type(first.tensors.qkv.manifest->ggml_type);  // GSQ-RCO typed (0 = Q8_0)
    in.segments.types[1] = q8_rows::segment_type(first.tensors.z.manifest->ggml_type);
    in.width = first.hidden; in.outputs = first.conv_width + first.value_width;
    for (unsigned r = 0; r < Rows; ++r) {
        in.input[r] = reinterpret_cast<const void*>(prepared[r].input.address);
        in.output[0][r] = prepared[r].workspace.qkv; in.output[1][r] = prepared[r].workspace.z;
    }
    q8_rows::launch(in, config.dense_a8_projections, hip_execution_stream());
    check_hip(hipGetLastError(), "GDN wide-window Q8 input projection launch");
    for (const auto& row : prepared) {
        const auto* input = reinterpret_cast<const std::uint16_t*>(row.input.address);
        launch_projection_by_kind(weight_kind(row.tensors.beta.manifest->ggml_type), input,
            *row.tensors.beta.runtime, row.hidden, row.value_heads, row.workspace.beta,
            row.projection_mode, hip_execution_stream(), nullptr);
        launch_projection_by_kind(weight_kind(row.tensors.alpha.manifest->ggml_type), input,
            *row.tensors.alpha.runtime, row.hidden, row.value_heads, row.workspace.gate,
            row.projection_mode, hip_execution_stream(), nullptr);
    }
    // Preserve every causal state transition and rounding point in row order.
    for (const auto& row : prepared) launch_gdn_known_window_core(row, config);
    GdnOutputRowsSite<Rows> out{};
    out.segments = q8_rows::segments({{reinterpret_cast<const void*>(first.tensors.out.runtime->address), first.hidden}});
    out.segments.types[0] = q8_rows::segment_type(first.tensors.out.manifest->ggml_type);
    out.width = first.value_width; out.outputs = first.hidden;
    for (unsigned r = 0; r < Rows; ++r) {
        out.input[r] = prepared[r].workspace.core;
        out.output[r] = reinterpret_cast<std::uint16_t*>(prepared[r].output.address);
    }
    q8_rows::launch(out, config.dense_a8_projections, hip_execution_stream());
    check_hip(hipGetLastError(), "GDN wide-window Q8 output projection launch");
}

}  // namespace

void gdn_known_window_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipGdnLauncherConfig& config) {
    if (rows.size() == 4U) static_cast<void>(prepare_gdn_wide_rows<4U>(rows, config));
    else if (rows.size() == 8U) static_cast<void>(prepare_gdn_wide_rows<8U>(rows, config));
    else throw std::invalid_argument("GDN wide-window requires exactly four or eight rows");
}
void gdn_known_window_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipGdnLauncherConfig& config) {
#if defined(STRIX_HAS_EXPERIMENTAL_DENSE_TRELLIS)
    HipDenseTrellisScope arithmetic(config.dense_trellis_projections);
#endif
    if (rows.size() == 4U) enqueue_gdn_wide_rows<4U>(prepare_gdn_wide_rows<4U>(rows, config), config);
    else if (rows.size() == 8U) enqueue_gdn_wide_rows<8U>(prepare_gdn_wide_rows<8U>(rows, config), config);
    else throw std::invalid_argument("GDN wide-window requires exactly four or eight rows");
}

namespace {
// T rows (a multiple of eight) of one GDN layer: the 8-row tiles keep the admitted wide projections and
// F32 gates, and the causal part (conv/norm prepare, recurrence, gated norm) runs once over all T rows.
std::vector<std::array<GdnKnownWindowRow, 8U>> prepare_gdn_prefill_rows(std::span<const ExecutionDispatchContext> rows,
    const HipGdnLauncherConfig& config, GdnPrefillRows& strided) {
    if (rows.size() < 16U || rows.size() % 8U)
        throw std::invalid_argument("GDN prefill rows require a multiple of eight, at least sixteen");
    std::vector<std::array<GdnKnownWindowRow, 8U>> tiles;
    tiles.reserve(rows.size() / 8U);
    for (std::size_t t = 0; t < rows.size(); t += 8U) tiles.push_back(prepare_gdn_wide_rows<8U>(rows.subspan(t, 8U), config));
    const auto& first = tiles[0][0];
    if (first.kernel != kGdnWindowKernel || first.tensors.beta.manifest->ggml_type != 0U ||
        first.tensors.alpha.manifest->ggml_type != 0U || !config.wave8_q8_input_projections || !config.wave8_q8_output_projection)
        throw std::invalid_argument("GDN prefill rows require conv kernel 4, F32 beta/alpha and wave8 Q8 projections");
    const auto stride = static_cast<std::uint64_t>(tiles[0][1].workspace.qkv - first.workspace.qkv);
    if (!stride) throw std::invalid_argument("GDN prefill rows share one workspace");
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const auto& row = tiles[r / 8U][r % 8U];
        const auto offset = static_cast<std::ptrdiff_t>(r * stride);
        if (rows[r].position != rows[0].position + r || rows[r].layer != rows[0].layer ||
            row.matrix_state.address != first.matrix_state.address || row.conv_state.address != first.conv_state.address ||
            row.tensors.qkv.runtime != first.tensors.qkv.runtime || row.tensors.out.runtime != first.tensors.out.runtime ||
            row.workspace.qkv != first.workspace.qkv + offset || row.workspace.z != first.workspace.z + offset ||
            row.workspace.beta != first.workspace.beta + offset || row.workspace.gate != first.workspace.gate + offset ||
            row.workspace.core != first.workspace.core + offset)
            throw std::invalid_argument("GDN prefill rows are not one layer at consecutive positions with uniformly strided workspaces");
    }
    strided = {first.workspace.qkv, first.workspace.z, first.workspace.beta, first.workspace.gate, first.workspace.core, stride};
    return tiles;
}
// BF16 element stride of the hidden input rows when row r is at row 0 + r * stride (else 0).
std::uint64_t gdn_prefill_input_stride(const std::vector<std::array<GdnKnownWindowRow, 8U>>& tiles) {
    const auto at = [&](std::size_t r) { return tiles[r / 8U][r % 8U].input.address; };
    const auto bytes = static_cast<std::int64_t>(at(1)) - static_cast<std::int64_t>(at(0));
    if (bytes <= 0 || bytes % 2) return 0;
    for (std::size_t r = 0; r < tiles.size() * 8U; ++r)
        if (at(r) != at(0) + r * static_cast<std::uint64_t>(bytes)) return 0;
    return static_cast<std::uint64_t>(bytes) / 2U;
}
// The row kernels are written for 128-wide heads and 16-byte aligned q/k/v rows.
bool gdn_prefill_rows_shape(const GdnKnownWindowRow& first, const GdnPrefillRows& strided, std::uint32_t count) {
    return first.head_dim == 128U && first.kernel == kGdnWindowKernel && first.value_heads % first.key_heads == 0U &&
        first.conv_width == (first.key_heads * 2U + first.value_heads) * 128U && first.value_width == first.value_heads * 128U &&
        strided.stride % 4U == 0U && reinterpret_cast<std::uintptr_t>(strided.qkv) % 16U == 0U && count % 8U == 0U;
}
void enqueue_gdn_prefill_core_tiles(const std::vector<std::array<GdnKnownWindowRow, 8U>>& tiles,
    const GdnPrefillRows& strided, const HipGdnLauncherConfig& config) {
    const auto stream = hip_execution_stream();
    const auto& first = tiles[0][0];
    const auto count = static_cast<std::uint32_t>(tiles.size() * 8U);
    for (const auto& tile : tiles) {
        GdnWidePointers<8U> x{};
        GdnWindowRows<8U> gates{};
        for (unsigned r = 0; r < 8U; ++r) {
            x.input[r] = reinterpret_cast<const void*>(tile[r].input.address);
            gates.beta[r] = tile[r].workspace.beta; gates.gate[r] = tile[r].workspace.gate;
        }
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_window_f32_gates_kernel<8U>), dim3(first.value_heads * 2U),
            dim3(kProjectionThreads), 0, stream, x, reinterpret_cast<const float*>(first.tensors.beta.runtime->address),
            reinterpret_cast<const float*>(first.tensors.alpha.runtime->address), gates, first.hidden, first.value_heads);
        check_hip(hipGetLastError(), "GDN prefill F32 gate projection launch");
    }
    hipLaunchKernelGGL(gdn_prefill_prepare_kernel, dim3(first.key_heads * 2U + first.value_heads + 1U), dim3(kRecurrentThreads),
        0, stream, strided, count, reinterpret_cast<std::uint16_t*>(first.conv_state.address),
        reinterpret_cast<const void*>(first.tensors.conv.runtime->address), small_weight_kind(first.tensors.conv.manifest->ggml_type),
        reinterpret_cast<const void*>(first.tensors.dt.runtime->address), small_weight_kind(first.tensors.dt.manifest->ggml_type),
        reinterpret_cast<const void*>(first.tensors.a.runtime->address), small_weight_kind(first.tensors.a.manifest->ggml_type),
        first.key_heads, first.value_heads, first.head_dim, config.model.gdn_qk_l2_epsilon);
    check_hip(hipGetLastError(), "GDN prefill prepare launch");
    const auto tiles_per_layer = dim3(first.value_heads * (first.head_dim / kGdnWindowColumns));
    auto* matrix = reinterpret_cast<float*>(first.matrix_state.address);
    if (config.recurrent_storage == GdnRecurrentStorage::value_key)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_prefill_recurrent_kernel<GdnRecurrentStorage::value_key>), tiles_per_layer,
            dim3(32), 0, stream, strided, count, matrix, first.key_heads, first.head_dim);
    else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_prefill_recurrent_kernel<GdnRecurrentStorage::key_value>), tiles_per_layer,
            dim3(32), 0, stream, strided, count, matrix, first.key_heads, first.head_dim);
    check_hip(hipGetLastError(), "GDN prefill recurrence launch");
    hipLaunchKernelGGL(gdn_prefill_norm_kernel, dim3(first.value_heads, count), dim3(kRecurrentThreads), 0, stream, strided,
        reinterpret_cast<const void*>(first.tensors.norm.runtime->address), small_weight_kind(first.tensors.norm.manifest->ggml_type),
        first.head_dim, config.model.rms_norm_epsilon);
    check_hip(hipGetLastError(), "GDN prefill gated RMSNorm launch");
}
// Shapes: STRIX_GDN_LANES (keys of a column split over 2|4|8 lanes, default 4), STRIX_GDN_LANE_WAVES
// (2|4|8 waves per recurrence block, default 8), STRIX_GDN_CHUNK (8|16 staged rows, default 8).
std::uint32_t gdn_prefill_env(const char* name, std::uint32_t fallback, std::initializer_list<std::uint32_t> allowed) {
    const char* v = std::getenv(name);
    const auto n = v ? static_cast<std::uint32_t>(std::atoi(v)) : fallback;
    return std::find(allowed.begin(), allowed.end(), n) != allowed.end() ? n : fallback;
}
// STRIX_QPF2K_GDN=1 (default OFF): the qpf2k core kernels (gdn_qpf2k_prefill.inc) replace the production gates and lane
// recurrence where their shape conditions hold; STRIX_QPF2K_GDN_PARTS (bit 1 recurrence, bit 2 gates, bit 4 fused core =
// q/k conv per row segment + recurrence with v conv and gated norm, bit 8 = that core with the norm exchange pipelined one
// chunk behind, bit 16 = that core with the lean epilogue (own norm sums in registers, z/norm weights loaded before
// the pair exchange, next chunk's v conv overlapped with the pair wait); default 55 = 1|2|4|16|32 since 4560c3b8, was 23) limits them; bit 32 =
// lean fused core with the decay-NaN dispatch (gdn_decay_nan_kernel flag; Nan 2 pinned launch + Nan 1 VOPD launch, exactly
// one does the work: exact like the pinned kernel, fast on calls without a NaN decay; in the default 55); bit 64 = lean fused core with 4
// columns per lane group (2 waves per block, 4-row chunks, post-order tree; with bit 32 also dispatched); STRIX_QPF2K_GDN_CHECK=1 runs
// the production kernels and the qpf2k kernels on the same inputs and compares bytes (with the fused core the raw v
// part of the qkv rows is left unconverted, so the check covers q/k of qkv and every other written byte).
// One stderr line per process the first time each qpf2k kernel is launched (launch witness for gates/receipts).
void qpf2k_gdn_witness(const char* what, std::uint32_t rows) {
    static std::once_flag gates_once, cols_once;
    std::call_once(what[0] == 'g' ? gates_once : cols_once,
        [&] { std::fprintf(stderr, "QPF2K GDN launch: %s (first call: %u rows)\n", what, rows); });
}
unsigned qpf2k_gdn_parts() {
    static const unsigned parts = [] {
        const char* v = std::getenv("STRIX_QPF2K_GDN");
        if (!v || v[0] != '1') return 0U;
        const char* p = std::getenv("STRIX_QPF2K_GDN_PARTS");
        // default 55 = 23 | 32 (decay-NaN dispatch ON: exact, t5 in-engine 32K core 1434.8 -> 1287.2 + 3.7 ms flag; 32K ABA never slower)
        return p ? static_cast<unsigned>(std::atoi(p)) & 255U : 55U;
    }();
    return qpf2k::suppressed(qpf2k::Item::gdn) ? 0U : parts;
}
// STRIX_QPF2K_GDN_FP16_CHUNK=1 (default OFF; FAST, bit-changing; prefill option gdn_fp16_chunk, identity tag gdnchunk16):
// numerical revision qwen-gdn-fp16-chunk rev 2 (hip_gdn_fp16_chunk.inc) replaces conv + recurrence + gated norm of the
// prefill row route with the chunked WMMA route. STRIX_PREFILL_EXACT_FP16=1 / --prefill-disable gdn_fp16_chunk turn it
// off; ignored under the baseline, for GSQ checkpoints and in the GDN check modes (they compare exact routes); shapes it
// cannot take keep the exact route.
bool gdn_fp16_chunk_enabled(const GdnKnownWindowRow& first, const GdnPrefillRows& rows) {
    static const bool requested = [] {
        const auto on = [](const char* name) { const char* v = std::getenv(name); return v && v[0] == '1'; };
        return strix::prefill_option_enabled(strix::PrefillOption::gdn_fp16_chunk) && !on("STRIX_QPF2K_GDN_CHECK") &&
            !on("STRIX_GDN_PREFILL_CHECK");
    }();
    if (!requested || qpf2k::baseline || qpf2k::gsq_model) return false;
    const bool fits = gdn_fp16_chunk_supported(first.head_dim, first.key_heads, first.value_heads, rows);
    static std::once_flag once;
    std::call_once(once, [&] {
        std::fprintf(stderr, fits ? "GDN FAST lever qwen-gdn-fp16-chunk rev 2: ON (bit-changing; STRIX_QPF2K_GDN_FP16_CHUNK=0 is exact)\n"
                                  : "GDN FAST lever qwen-gdn-fp16-chunk rev 2: requested, shape not supported -> exact recurrence\n");
    });
    return fits;
}
// Fused core resources: the norm exchange (per stream) and the q/k segment-boundary side buffer.
constexpr unsigned kQpf2kSegments = 8U, kQpf2kCoreWaves = 4U;
// Chunk-16 core default (STRIX_QPF2K_GDN_CHUNK16 unset): calls of at most this many rows. Exact either way; receipts
// (tight padded Chunk-16 vs 8-row dispatch core, gdn_core, ABBA): T 1024 -4.6 %, T 2048 -1.1 %, T 4096 -0.8 %,
// T 8192 +0.4 %, T 16384 +5 %: 4096 is the last measured winning point. STRIX_QPF2K_GDN_CHUNK16_MAX_ROWS overrides it
// for crossover measurements.
constexpr std::uint32_t kQpf2kChunk16MaxRows = 4096U;
struct Qpf2kCoreScratch {
    qpf2k_gdn::NormExchange exchange{}; std::uint16_t* side = nullptr; std::uint64_t side_bytes = 0;
    unsigned ring_rows = 8U;   // rows per norm ring slot (16 only once a Chunk-16 core call needed it)
};
// Returned by value: the pool may grow under a concurrent caller.
Qpf2kCoreScratch qpf2k_core_scratch(hipStream_t stream, std::uint32_t value_heads, std::uint64_t side_bytes, unsigned ring_rows = 8U) {
    static std::mutex mutex;
    static std::vector<std::pair<hipStream_t, Qpf2kCoreScratch>> pool;
    std::lock_guard<std::mutex> lock(mutex);
    for (auto& [s, scratch] : pool)
        if (s == stream && scratch.side_bytes >= side_bytes && scratch.ring_rows >= ring_rows) return scratch;
    for (auto& [s, scratch] : pool)
        if (s == stream) ring_rows = std::max(ring_rows, scratch.ring_rows);
    Qpf2kCoreScratch x;
    const std::uint64_t partial = std::uint64_t{value_heads} * 2U * qpf2k_gdn::kNormRing * ring_rows * 32U * 4U;
    void* p = nullptr;
    if (hipMalloc(&p, partial + std::uint64_t{value_heads} * 2U * 8U + 16U + side_bytes) != hipSuccess)
        throw std::runtime_error("qpf2k GDN core scratch allocation failed");
    x.exchange.partial = static_cast<float*>(p);
    x.exchange.published = reinterpret_cast<unsigned*>(static_cast<unsigned char*>(p) + partial);
    x.exchange.consumed = x.exchange.published + value_heads * 2U;
    x.exchange.base = 0;
    x.exchange.decay_nan = x.exchange.consumed + value_heads * 2U;   // one word (16 bytes reserved), zeroed with the counters
    x.exchange.witness = x.exchange.consumed + value_heads * 2U + 1U;  // PARTS bit 128: the NaN witness (next word, zeroed too)
    x.exchange.arrival = x.exchange.consumed + value_heads * 2U + 2U;  // two ticket words: first core, then paired / fallback
    x.side = reinterpret_cast<std::uint16_t*>(x.exchange.consumed + value_heads * 2U + 4U);
    x.side_bytes = side_bytes;
    x.ring_rows = ring_rows;
    for (auto& [s, scratch] : pool)
        if (s == stream) { scratch = x; return scratch; }  // grown (the old block stays allocated: in-flight launches)
    pool.emplace_back(stream, x);
    return pool.back().second;
}
// PARTS bit 128: per-stream scratch for the witness kernel's final state and v conv history.
struct Qpf2kWitnessScratch { float* state = nullptr; std::uint16_t* conv = nullptr; std::uint64_t state_bytes = 0, conv_bytes = 0; };
Qpf2kWitnessScratch qpf2k_witness_scratch(hipStream_t stream, std::uint64_t state_bytes, std::uint64_t conv_bytes) {
    static std::mutex mutex;
    static std::vector<std::pair<hipStream_t, Qpf2kWitnessScratch>> pool;
    std::lock_guard<std::mutex> lock(mutex);
    for (auto& [s, scratch] : pool)
        if (s == stream && scratch.state_bytes >= state_bytes && scratch.conv_bytes >= conv_bytes) return scratch;
    for (auto& [s, scratch] : pool)
        if (s == stream) { state_bytes = std::max(state_bytes, scratch.state_bytes); conv_bytes = std::max(conv_bytes, scratch.conv_bytes); }
    Qpf2kWitnessScratch x;
    void* p = nullptr;
    const std::uint64_t state_aligned = (state_bytes + 255U) / 256U * 256U;
    if (hipMalloc(&p, state_aligned + conv_bytes) != hipSuccess) throw std::runtime_error("qpf2k GDN witness scratch allocation failed");
    x.state = static_cast<float*>(p);
    x.conv = reinterpret_cast<std::uint16_t*>(static_cast<unsigned char*>(p) + state_aligned);
    x.state_bytes = state_bytes; x.conv_bytes = conv_bytes;
    for (auto& [s, scratch] : pool)
        if (s == stream) { scratch = x; return scratch; }  // grown (the old block stays allocated: in-flight launches)
    pool.emplace_back(stream, x);
    return pool.back().second;
}
// Bit 128 layout (STRIX_QPF2K_GDN_KR_G = 4 (default) | 8 | 16): G-lane groups, 4 waves, 64 columns per block, Cc = G / 2.
// G 4 runs Opt 118 (pipelined rows, paired query/key trees, fused DPP group sums: bench -0.25 ms per 8K call vs the
// decay-NaN dispatch) + bit 256 by default (Opt 374, qpf2k_kr_opt); G 8 / 16 run Opt 6 (bench: slower).
unsigned qpf2k_kr_g() {
    static const unsigned g = [] {
        const char* v = std::getenv("STRIX_QPF2K_GDN_KR_G");
        const int x = v ? std::atoi(v) : 4;
        return x == 16 ? 16U : x == 8 ? 8U : 4U;
    }();
    return g;
}
template <unsigned G> constexpr unsigned kQpf2kKrOpt = G == 4U ? 118U : 6U;
// G 4 default Opt 374 (+ bit 256: norm completion deferred one chunk; engine 8K: -0.30 ms per call vs Opt 118 in 5 / 5
// alternating pairs, receipts == BASE); STRIX_QPF2K_GDN_KR_OPT=118 restores Opt 118; 886 / 1398 are NO-GAIN variants.
unsigned qpf2k_kr_opt() {
    static const unsigned opt = [] {
        const char* v = std::getenv("STRIX_QPF2K_GDN_KR_OPT");
        return v && std::strcmp(v, "1398") == 0 ? 1398U : v && std::strcmp(v, "886") == 0 ? 886U : v && std::strcmp(v, "118") == 0 ? 118U : 374U;
    }();
    return opt;
}
bool qpf2k_gdn_coop_launch() {
    // Read once per process; launch-policy changes require a fresh process.
    static const bool enabled = [] { const char* v = std::getenv("STRIX_QPF2K_GDN_COOP_LAUNCH"); return v && v[0] == '1'; }();
    return enabled;
}
// Linux 7.4 cooperative-queue eviction/restore left the grid partly non-resident and hung prefill.
// Plain launches leave that queue; arrival pairing still has a pair-local norm wait, not arbitrary restore safety.
void qpf2k_core_launch(const void* kernel, std::uint32_t blocks, unsigned threads, void** args, hipStream_t stream, const char* what) {
    check_hip(qpf2k_gdn_coop_launch()
        ? hipLaunchCooperativeKernel(kernel, dim3(blocks), dim3(threads), args, 0, stream)
        : hipLaunchKernel(kernel, dim3(blocks), dim3(threads), args, 0, stream), what);
}
// Keep the historical admission in both modes so only the launch call changes. Capacity (co-resident blocks)
// is computed once per process (one device per engine process) and compared with every call's block count.
template <GdnRecurrentStorage S, unsigned G, unsigned Opt = kQpf2kKrOpt<G>, bool F16 = false>
bool qpf2k_kr_fits(std::uint32_t blocks) {
    static const std::uint64_t capacity = [] {
        int device = 0, cooperative = 0, units = 0, per_unit = 0;
        if (hipGetDevice(&device) != hipSuccess) return std::uint64_t{0};
        if (hipDeviceGetAttribute(&cooperative, hipDeviceAttributeCooperativeLaunch, device) != hipSuccess || !cooperative) return std::uint64_t{0};
        if (hipDeviceGetAttribute(&units, hipDeviceAttributeMultiprocessorCount, device) != hipSuccess) return std::uint64_t{0};
        if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_unit,
                reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<S, G, G / 2U, 4U, 8U, Opt, false, 1U, 4U, F16>), 128, 0) != hipSuccess)
            return std::uint64_t{0};
        if (units < 1 || per_unit < 1) return std::uint64_t{0};
        const auto c = static_cast<std::uint64_t>(units) * static_cast<std::uint64_t>(per_unit);
        std::fprintf(stderr, "QPF2K GDN core residency (witness G %u Opt %u F16 %u): units %d blocks/unit %d capacity %llu\n", G, Opt, static_cast<unsigned>(F16), units, per_unit,
            static_cast<unsigned long long>(c));
        return c;
    }();
    return capacity >= blocks;
}
// Both launch modes retain whole-grid capacity admission, decided before any q/k row is rewritten.
// Rows: chunk rows (0 = the column count's default: 4 for 4 columns, else 8); OptBits: added to Opt.
template <GdnRecurrentStorage S, unsigned Lean = 0U, unsigned Cc = 2U, unsigned Nan = 3U, bool F16 = false, unsigned Rows = 0U,
    unsigned OptBits = 0U>
bool qpf2k_core_fits(std::uint32_t blocks) {
    // Opt 6 (post-order tree) for the 4-column core and for the dispatched kernels (Nan 1 / 2); Opt 2 for the default Nan 3
    constexpr unsigned Waves = Cc == 4U ? 2U : kQpf2kCoreWaves, Chunk = Rows != 0U ? Rows : Cc == 4U ? 4U : 8U,
        Opt = (Cc == 4U || Nan == 1U || Nan == 2U ? 6U : 2U) | OptBits;
    static const bool fits = [&] {
        int device = 0, cooperative = 0, units = 0, per_unit = 0;
        if (hipGetDevice(&device) != hipSuccess) return false;
        if (hipDeviceGetAttribute(&cooperative, hipDeviceAttributeCooperativeLaunch, device) != hipSuccess || !cooperative) return false;
        if (hipDeviceGetAttribute(&units, hipDeviceAttributeMultiprocessorCount, device) != hipSuccess) return false;
        if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_unit,
                reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<S, 4U, Cc, Waves, Chunk, Opt, false, Lean, Nan, F16>),
                static_cast<int>(Waves * 32U), 0) != hipSuccess) return false;
        if (units < 1 || per_unit < 1) return false;
        const bool ok = static_cast<std::uint64_t>(units) * static_cast<std::uint64_t>(per_unit) >= blocks;
        std::fprintf(stderr, "QPF2K GDN core residency (%u columns per lane group, nan %u%s): units %d blocks/unit %d need %u -> %s\n", Cc,
            Nan, Rows != 16U ? "" : (OptBits & 2048U) != 0U ? ", chunk 16 single stage unpadded" : ", chunk 16 single stage",
            units, per_unit, blocks, ok ? "fused core" : Rows == 16U ? "not this variant" : "unfused kernels");
        return ok;
    }();
    return fits;
}
// The projection must not borrow tiled::scratch: another projection can grow and release it.
// Calls on a stream reuse these owned rows only after that stream's preceding projection.
struct Qpf2kCore16Route {
    GdnOutputTiledSite site{};
    _Float16* a16 = nullptr;                 // non-null only after a supported lean core selects this route
    unsigned form = 4U, raster = 2U;
};
_Float16* qpf2k_core16_scratch(hipStream_t stream, std::uint32_t rows) {
    struct Slot { void* p = nullptr; std::uint64_t bytes = 0; };
    static std::mutex mutex;
    static std::vector<std::pair<hipStream_t, Slot>> pool;
    const std::lock_guard lock(mutex);
    auto it = std::find_if(pool.begin(), pool.end(), [&](const auto& entry) { return entry.first == stream; });
    if (it == pool.end()) it = pool.emplace(pool.end(), stream, Slot{});
    auto& s = it->second;
    const auto bytes = (static_cast<std::uint64_t>(rows) + q8_rows::tiled::kBM - 1U) / q8_rows::tiled::kBM *
        q8_rows::tiled::kBM * q8_rows::tiled::scratch_pitch(6144U) * 2U;
    if (s.bytes < bytes) {
        if (s.p) {
            check_hip(hipStreamSynchronize(stream), "qpf2k GDN core16 scratch drain");
            check_hip(hipFree(s.p), "qpf2k GDN core16 scratch release");
            s = {};
        }
        check_hip(hipMalloc(&s.p, bytes), "qpf2k GDN core16 scratch allocation");
        s.bytes = bytes;
    }
    return static_cast<_Float16*>(s.p);
}
void enqueue_gdn_prefill_core_rows(const std::vector<std::array<GdnKnownWindowRow, 8U>>& tiles,
    const GdnPrefillRows& strided, std::uint64_t input_stride, const HipGdnLauncherConfig& config, const q8_rows::DenseA8Slot* fused_a8,
    unsigned qpf2k = qpf2k_gdn_parts(), Qpf2kCore16Route* core16 = nullptr, const std::uint16_t* x16 = nullptr) {
    const auto stream = hip_execution_stream();
    const auto& first = tiles[0][0];
    const auto count = static_cast<std::uint32_t>(tiles.size() * 8U);
    // The FAST chunked route replaces conv + recurrence + norm after the gates: no fused qpf2k core.
    const bool fp16_chunk = gdn_fp16_chunk_enabled(first, strided);
    if (fp16_chunk) qpf2k &= ~4U;
    // STRIX_GDN_GATE_HEADS: beta/alpha outputs per gates block (1|2|4, default 2; must divide the head count).
    static const auto gate_heads_env = gdn_prefill_env("STRIX_GDN_GATE_HEADS", 2U, {1U, 2U, 4U});
    const auto gate_heads = first.value_heads % gate_heads_env ? 1U : gate_heads_env;
    const auto gates = [&]<unsigned H>() {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_prefill_gates_kernel<8U, H>), dim3(first.value_heads * 2U / H, count / 8U),
            dim3(kProjectionThreads), 0, stream, reinterpret_cast<const std::uint16_t*>(first.input.address), input_stride,
            reinterpret_cast<const float*>(first.tensors.beta.runtime->address),
            reinterpret_cast<const float*>(first.tensors.alpha.runtime->address), strided,
            reinterpret_cast<const void*>(first.tensors.dt.runtime->address), small_weight_kind(first.tensors.dt.manifest->ggml_type),
            reinterpret_cast<const void*>(first.tensors.a.runtime->address), small_weight_kind(first.tensors.a.manifest->ggml_type),
            first.hidden, first.value_heads);
    };
    // STRIX_GDN_GATES_ROWS=1: gdn_gates::gdn_gates_rows_kernel (the same chains, tree and transform; all rows of 16-row
    // tiles share their staged inputs and the weights of 16 outputs).
    static const bool gates_rows = [] { const char* v = std::getenv("STRIX_GDN_GATES_ROWS"); return v && v[0] == '1'; }();
    constexpr unsigned GatesRB = 16U, GatesO = 4U, GatesS = 4U, GatesC = 8U;
    const bool rows_fit = gates_rows && count >= 256U && count % GatesRB == 0U && first.hidden % gdn_gates::kChains == 0U &&
        (first.value_heads * 2U) % (GatesO * GatesS) == 0U && input_stride % 8U == 0U &&
        reinterpret_cast<std::uintptr_t>(first.input.address) % 16U == 0U;
    constexpr unsigned QGatesO = 1U, QGatesW = 8U, QGatesRC = 4U, QGatesRB = 64U;
    const bool qpf2k_gates = (qpf2k & 2U) && first.hidden == 2560U && count % QGatesRB == 0U &&
        (first.value_heads * 2U) % (QGatesO * QGatesW) == 0U && input_stride % 8U == 0U &&
        reinterpret_cast<std::uintptr_t>(first.input.address) % 16U == 0U;
    // STRIX_QPF2K_GDN_GATES_BATCH=1: the same kernel with Sched bit 4 (dt/a preloaded, transforms across lanes).
    static const bool gates_batch_env = [] { const char* v = std::getenv("STRIX_QPF2K_GDN_GATES_BATCH"); return v && v[0] == '1'; }();
    const bool gates_batch = gates_batch_env && !qpf2k::suppressed(qpf2k::Item::gdn_gates_batch);
    if (qpf2k_gates && gates_batch)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_gdn::gdn_qpf2k_gates_kernel<QGatesO, QGatesW, QGatesRC, QGatesRB, 2560U, 4U>),
            dim3(first.value_heads * 2U / (QGatesO * QGatesW), count / QGatesRB), dim3(QGatesW * 32U), 0, stream,
            reinterpret_cast<const std::uint16_t*>(first.input.address), input_stride,
            reinterpret_cast<const float*>(first.tensors.beta.runtime->address),
            reinterpret_cast<const float*>(first.tensors.alpha.runtime->address), strided,
            reinterpret_cast<const void*>(first.tensors.dt.runtime->address), small_weight_kind(first.tensors.dt.manifest->ggml_type),
            reinterpret_cast<const void*>(first.tensors.a.runtime->address), small_weight_kind(first.tensors.a.manifest->ggml_type),
            first.value_heads);
    else if (qpf2k_gates)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_gdn::gdn_qpf2k_gates_kernel<QGatesO, QGatesW, QGatesRC, QGatesRB>),
            dim3(first.value_heads * 2U / (QGatesO * QGatesW), count / QGatesRB), dim3(QGatesW * 32U), 0, stream,
            reinterpret_cast<const std::uint16_t*>(first.input.address), input_stride,
            reinterpret_cast<const float*>(first.tensors.beta.runtime->address),
            reinterpret_cast<const float*>(first.tensors.alpha.runtime->address), strided,
            reinterpret_cast<const void*>(first.tensors.dt.runtime->address), small_weight_kind(first.tensors.dt.manifest->ggml_type),
            reinterpret_cast<const void*>(first.tensors.a.runtime->address), small_weight_kind(first.tensors.a.manifest->ggml_type),
            first.value_heads);
    if (qpf2k_gates) qpf2k_gdn_witness(gates_batch ? "gates gdn_qpf2k_gates_kernel<1,8,4,64,batch>" : "gates gdn_qpf2k_gates_kernel<1,8,4,64>", count);
    else if (rows_fit)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_gates::gdn_gates_rows_kernel<GatesRB, GatesO, GatesS, GatesC>),
            dim3(first.value_heads * 2U / (GatesO * GatesS), count / GatesRB), dim3(GatesS * 256U), 0, stream,
            reinterpret_cast<const std::uint16_t*>(first.input.address), input_stride,
            reinterpret_cast<const float*>(first.tensors.beta.runtime->address),
            reinterpret_cast<const float*>(first.tensors.alpha.runtime->address), strided,
            reinterpret_cast<const void*>(first.tensors.dt.runtime->address), small_weight_kind(first.tensors.dt.manifest->ggml_type),
            reinterpret_cast<const void*>(first.tensors.a.runtime->address), small_weight_kind(first.tensors.a.manifest->ggml_type),
            first.hidden, first.value_heads);
    else if (gate_heads == 2U) gates.template operator()<2U>();
    else if (gate_heads == 4U) gates.template operator()<4U>();
    else gates.template operator()<1U>();
    check_hip(hipGetLastError(), "GDN prefill F32 gate rows launch");
    // qpf2k fused core (bit 4): replaces the conv, recurrence and gated-norm launches below.
    {
        auto* matrix_state = reinterpret_cast<float*>(first.matrix_state.address);
        const bool vk = config.recurrent_storage == GdnRecurrentStorage::value_key;
        const auto blocks = first.value_heads * 2U;
        const bool pipe = qpf2k & 8U, lean = (qpf2k & 16U) && !pipe;
        // bits 32 / 64 (gdn-dense): decay-NaN dispatch / 4 columns per lane group, lean epilogue only. Both launch modes
        // check capacity for every kernel (Nan 1 and 2 with dispatch, else Nan 3); a 4-column core that does not
        // fit falls back to the 2-column core.
        const auto fits_of = [&]<unsigned Cc, unsigned Nan>() {
            return vk ? qpf2k_core_fits<GdnRecurrentStorage::value_key, 1U, Cc, Nan>(blocks) : qpf2k_core_fits<GdnRecurrentStorage::key_value, 1U, Cc, Nan>(blocks);
        };
        const bool dispatch_bit = lean && (qpf2k & 32U);
        // bit 128: witness core (Nan 4, G-lane layout) + pinned Nan 2 fallback guarded by the witness;
        // both launch modes check the 2-column Nan 2 kernel's capacity too.
        const unsigned kr_g = qpf2k_kr_g(), kr_opt = qpf2k_kr_opt();
        const bool witness = lean && (qpf2k & 128U) && !(qpf2k & 64U) &&
            (vk ? (kr_g == 16U ? qpf2k_kr_fits<GdnRecurrentStorage::value_key, 16U>(blocks) : kr_g == 8U ? qpf2k_kr_fits<GdnRecurrentStorage::value_key, 8U>(blocks)
                                 : (kr_opt == 1398U ? qpf2k_kr_fits<GdnRecurrentStorage::value_key, 4U, 1398U>(blocks)
                                 : kr_opt == 886U ? qpf2k_kr_fits<GdnRecurrentStorage::value_key, 4U, 886U>(blocks)
                                 : kr_opt == 374U ? qpf2k_kr_fits<GdnRecurrentStorage::value_key, 4U, 374U>(blocks) : qpf2k_kr_fits<GdnRecurrentStorage::value_key, 4U>(blocks)))
                : (kr_g == 16U ? qpf2k_kr_fits<GdnRecurrentStorage::key_value, 16U>(blocks) : kr_g == 8U ? qpf2k_kr_fits<GdnRecurrentStorage::key_value, 8U>(blocks)
                                 : (kr_opt == 1398U ? qpf2k_kr_fits<GdnRecurrentStorage::key_value, 4U, 1398U>(blocks)
                                 : kr_opt == 886U ? qpf2k_kr_fits<GdnRecurrentStorage::key_value, 4U, 886U>(blocks)
                                 : kr_opt == 374U ? qpf2k_kr_fits<GdnRecurrentStorage::key_value, 4U, 374U>(blocks) : qpf2k_kr_fits<GdnRecurrentStorage::key_value, 4U>(blocks)))) &&
            fits_of.template operator()<2U, 2U>();
        Qpf2kWitnessScratch witness_scratch{};
        const std::uint64_t witness_state_elems = std::uint64_t{first.value_heads} * 128U * 128U;
        const std::uint64_t witness_conv_elems = (std::uint64_t{first.key_heads} * 2U * 128U + std::uint64_t{first.value_heads} * 128U) * kGdnWindowKernel;
        if (witness) {
            try { witness_scratch = qpf2k_witness_scratch(stream, witness_state_elems * 4U, witness_conv_elems * 2U); }
            catch (const std::exception&) { witness_scratch = {}; (void)hipGetLastError(); }   // clear the failed allocation
        }
        const bool witness_ok = witness && witness_scratch.state != nullptr;
        const bool four = lean && (qpf2k & 64U) &&
            (dispatch_bit ? fits_of.template operator()<4U, 1U>() && fits_of.template operator()<4U, 2U>() : fits_of.template operator()<4U, 3U>());
        const bool dispatch = !witness_ok && dispatch_bit &&
            (four || (fits_of.template operator()<2U, 1U>() && fits_of.template operator()<2U, 2U>()));
        const bool fits = witness_ok || four || dispatch
                        || (lean ? (vk ? qpf2k_core_fits<GdnRecurrentStorage::value_key, 1U>(blocks) : qpf2k_core_fits<GdnRecurrentStorage::key_value, 1U>(blocks))
                                 : (vk ? qpf2k_core_fits<GdnRecurrentStorage::value_key>(blocks) : qpf2k_core_fits<GdnRecurrentStorage::key_value>(blocks)));
        if ((qpf2k & 4U) && !fused_a8 && count % 8U == 0U && count >= 8U && first.head_dim == 128U && fits) {
            // Check the actual FP16 variants in both launch modes before replacing any FP32 stores.
            const auto fits16 = [&]<unsigned Cc, unsigned Nan>() {
                return vk ? qpf2k_core_fits<GdnRecurrentStorage::value_key, 1U, Cc, Nan, true>(blocks)
                          : qpf2k_core_fits<GdnRecurrentStorage::key_value, 1U, Cc, Nan, true>(blocks);
            };
            const auto kr_fits16 = [&]<GdnRecurrentStorage S>() {
                if (kr_g == 16U) return qpf2k_kr_fits<S, 16U, kQpf2kKrOpt<16U>, true>(blocks);
                if (kr_g == 8U) return qpf2k_kr_fits<S, 8U, kQpf2kKrOpt<8U>, true>(blocks);
                if (kr_opt == 374U) return qpf2k_kr_fits<S, 4U, 374U, true>(blocks);
                if (kr_opt == 118U) return qpf2k_kr_fits<S, 4U, 118U, true>(blocks);
                return false;   // Opt 886 / 1398 remain FP32-only.
            };
            const bool f16 = core16 && lean && (witness_ok
                ? (vk ? kr_fits16.template operator()<GdnRecurrentStorage::value_key>()
                      : kr_fits16.template operator()<GdnRecurrentStorage::key_value>()) && fits16.template operator()<2U, 2U>()
                : (four
                ? (dispatch ? fits16.template operator()<4U, 1U>() && fits16.template operator()<4U, 2U>() : fits16.template operator()<4U, 3U>())
                : (dispatch ? fits16.template operator()<2U, 1U>() && fits16.template operator()<2U, 2U>() : fits16.template operator()<2U, 3U>())));
            if (f16) core16->a16 = qpf2k_core16_scratch(stream, count);
            // Chunk-16 core: the dispatched 2-column block pairs (Nan 2 / Nan 1) with 16-row chunks and one stage (half the
            // barriers and norm exchanges per row; the same per-row work, bit-identical outputs, so the choice is per call)
            // when the call's rows are a multiple of 16 (both modes: same capacity admission): padded rows (Opt 6) if they fit,
            // else (forced only) unpadded (Opt 2054); else the 8-row kernels. The norm ring is sized for 16 rows only for such a call.
            // STRIX_QPF2K_GDN_CHUNK16: unset = auto (calls of at most kQpf2kChunk16MaxRows rows, or
            // STRIX_QPF2K_GDN_CHUNK16_MAX_ROWS), 1 = every eligible call, 0 = off (8-row kernels). Off under the baseline,
            // STRIX_QPF2K_DEFAULTS=0 and GSQ models (code-default exact item, --print-effective-config long_context_items).
            static const bool chunk16_on = qpf2k::long_context_item_enabled("STRIX_QPF2K_GDN_CHUNK16", false);
            static const bool chunk16_forced = [] { const char* v = std::getenv("STRIX_QPF2K_GDN_CHUNK16"); return v && v[0] == '1'; }();
            static const std::uint32_t chunk16_max_rows = [] {
                const char* v = std::getenv("STRIX_QPF2K_GDN_CHUNK16_MAX_ROWS");
                const long long n = v ? std::atoll(v) : static_cast<long long>(kQpf2kChunk16MaxRows);
                return n >= 0 && n <= 0xffffffffLL ? static_cast<std::uint32_t>(n) : kQpf2kChunk16MaxRows;
            }();
            const auto chunk16_fits = [&]<GdnRecurrentStorage S, bool F16, unsigned Bits>() {
                return qpf2k_core_fits<S, 1U, 2U, 1U, F16, 16U, Bits>(blocks) && qpf2k_core_fits<S, 1U, 2U, 2U, F16, 16U, Bits>(blocks);
            };
            const auto chunk16_fits_of = [&]<unsigned Bits>() {
                return vk ? (f16 ? chunk16_fits.template operator()<GdnRecurrentStorage::value_key, true, Bits>()
                                 : chunk16_fits.template operator()<GdnRecurrentStorage::value_key, false, Bits>())
                          : (f16 ? chunk16_fits.template operator()<GdnRecurrentStorage::key_value, true, Bits>()
                                 : chunk16_fits.template operator()<GdnRecurrentStorage::key_value, false, Bits>());
            };
            const bool chunk16_route = chunk16_on && !qpf2k::gsq_model && !qpf2k::suppressed(qpf2k::Item::gdn) && dispatch && !four &&
                !witness_ok && lean && count % 16U == 0U && (chunk16_forced || count <= chunk16_max_rows);
            const bool chunk16_padded = chunk16_route && chunk16_fits_of.template operator()<0U>();
            // the unpadded rows (bank conflicts) measured slower than the 8-row kernels: only when forced (STRIX_QPF2K_GDN_CHUNK16=1)
            const bool chunk16_unpadded = chunk16_route && chunk16_forced && !chunk16_padded && chunk16_fits_of.template operator()<2048U>();
            const bool chunk16 = chunk16_padded || chunk16_unpadded;
            const auto qk_channels = first.key_heads * 2U * 128U;
            const unsigned segs = count >= 64U * kQpf2kSegments ? kQpf2kSegments : 1U;
            const unsigned seg_rows = (count / segs + 31U) / 32U * 32U, nseg = (count + seg_rows - 1U) / seg_rows;
            const auto scratch = chunk16
                ? qpf2k_core_scratch(stream, first.value_heads, std::uint64_t{kQpf2kSegments} * 3U * qk_channels * 2U, 16U)
                : qpf2k_core_scratch(stream, first.value_heads, std::uint64_t{kQpf2kSegments} * 3U * qk_channels * 2U);
            auto* conv_state = reinterpret_cast<std::uint16_t*>(first.conv_state.address);
            const void* conv_w = reinterpret_cast<const void*>(first.tensors.conv.runtime->address);
            const auto conv_kind = small_weight_kind(first.tensors.conv.manifest->ggml_type);
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_gdn::gdn_qk_boundary_kernel<>), dim3(16, nseg), dim3(256), 0, stream, strided, seg_rows,
                qk_channels, static_cast<const std::uint16_t*>(conv_state), scratch.side);
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_gdn::gdn_qk_conv_kernel<8U, 32U>), dim3(first.key_heads * 2U, nseg), dim3(256), 0, stream,
                strided, count, seg_rows, conv_state, static_cast<const std::uint16_t*>(scratch.side), conv_w, conv_kind, first.key_heads,
                first.head_dim, config.model.gdn_qk_l2_epsilon);
            check_hip(hipGetLastError(), "qpf2k GDN q/k conv launch");
            check_hip(hipMemsetAsync(scratch.exchange.published, 0, std::uint64_t{first.value_heads} * 2U * 2U * 4U + 16U, stream),
                "qpf2k GDN exchange reset");   // counters + decay-NaN flag + NaN witness + two per-launch ticket words
            if (dispatch) {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_gdn::gdn_decay_nan_kernel<>), dim3(64), dim3(256), 0, stream, strided, count,
                    first.value_heads, const_cast<unsigned*>(scratch.exchange.decay_nan));
                check_hip(hipGetLastError(), "qpf2k GDN decay-NaN flag launch");
            }
            auto rows_arg = strided; auto count_arg = count; auto* state_arg = matrix_state; auto key_heads_arg = first.key_heads;
            auto* conv_arg = conv_state; auto conv_w_arg = conv_w; auto conv_kind_arg = conv_kind;
            const void* norm_w = reinterpret_cast<const void*>(first.tensors.norm.runtime->address);
            auto norm_kind = small_weight_kind(first.tensors.norm.manifest->ggml_type);
            float eps = config.model.rms_norm_epsilon; auto xch = scratch.exchange;
            if (f16) {
                xch.core16 = core16->a16;
                xch.core16_pitch = q8_rows::tiled::scratch_pitch(6144U);
                xch.core16_overflow = q8_rows::dense_f16_overflow_counter();
            }
            void* args[] = {&rows_arg, &count_arg, &state_arg, &key_heads_arg, &conv_arg, &conv_w_arg, &conv_kind_arg, &norm_w, &norm_kind,
                &eps, &xch};
            using KV = std::integral_constant<GdnRecurrentStorage, GdnRecurrentStorage::key_value>;
            const void* kernel = lean ? (vk ? reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<GdnRecurrentStorage::value_key, 4U, 2U, kQpf2kCoreWaves, 8U, 2U, false, 1U>)
                                            : reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<KV::value, 4U, 2U, kQpf2kCoreWaves, 8U, 2U, false, 1U>))
                               : vk ? (pipe ? reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<GdnRecurrentStorage::value_key, 4U, 2U, kQpf2kCoreWaves, 8U, 2U, true>)
                                            : reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<GdnRecurrentStorage::value_key, 4U, 2U, kQpf2kCoreWaves, 8U, 2U>))
                                    : (pipe ? reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<KV::value, 4U, 2U, kQpf2kCoreWaves, 8U, 2U, true>)
                                            : reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<KV::value, 4U, 2U, kQpf2kCoreWaves, 8U, 2U>));
            if (f16) kernel = vk
                ? reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<GdnRecurrentStorage::value_key, 4U, 2U, kQpf2kCoreWaves, 8U, 2U, false, 1U, 3U, true>)
                : reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<KV::value, 4U, 2U, kQpf2kCoreWaves, 8U, 2U, false, 1U, 3U, true>);
            // gdn-dense variants (lean only): Nan 2 (pinned, runs iff the flag is set) then Nan 1 (VOPD, runs iff clear).
            const auto variant = [&]<GdnRecurrentStorage S, unsigned Nan>() -> const void* {
                if (f16) {
                    if (four) return reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<S, 4U, 4U, 2U, 4U, 6U, false, 1U, Nan, true>);
                    return reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<S, 4U, 2U, kQpf2kCoreWaves, 8U, 6U, false, 1U, Nan, true>);
                }
                if (four) return reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<S, 4U, 4U, 2U, 4U, 6U, false, 1U, Nan>);
                // t1 (bench): the post-order tree (Opt 6, 201 VGPRs) is faster than Opt 2 at 2 columns too (9.92 vs 10.10 ms per 8K call)
                return reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<S, 4U, 2U, kQpf2kCoreWaves, 8U, 6U, false, 1U, Nan>);
            };
            const auto chunk16_variant = [&]<GdnRecurrentStorage S, unsigned Nan>() -> const void* {
                if (chunk16_unpadded) {
                    if (f16) return reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<S, 4U, 2U, kQpf2kCoreWaves, 16U, 2054U, false, 1U, Nan, true>);
                    return reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<S, 4U, 2U, kQpf2kCoreWaves, 16U, 2054U, false, 1U, Nan>);
                }
                if (f16) return reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<S, 4U, 2U, kQpf2kCoreWaves, 16U, 6U, false, 1U, Nan, true>);
                return reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<S, 4U, 2U, kQpf2kCoreWaves, 16U, 6U, false, 1U, Nan>);
            };
            const unsigned threads = four ? 64U : kQpf2kCoreWaves * 32U;
            if (witness_ok) {
                const std::uint64_t state_elems = witness_state_elems;
                const auto& ws = witness_scratch;
                xch.state_out = ws.state; xch.conv_out = ws.conv;
                const auto kr_of = [&]<GdnRecurrentStorage S, bool F16 = false>() -> const void* {
                    if (kr_g == 16U) return reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<S, 16U, 8U, 4U, 8U, kQpf2kKrOpt<16U>, false, 1U, 4U, F16>);
                    if (kr_g == 8U) return reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<S, 8U, 4U, 4U, 8U, kQpf2kKrOpt<8U>, false, 1U, 4U, F16>);
                    if constexpr (!F16) {
                        if (kr_opt == 1398U) return reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<S, 4U, 2U, 4U, 8U, 1398U, false, 1U, 4U, F16>);
                        if (kr_opt == 886U) return reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<S, 4U, 2U, 4U, 8U, 886U, false, 1U, 4U, F16>);
                    }
                    if (kr_opt == 374U) return reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<S, 4U, 2U, 4U, 8U, 374U, false, 1U, 4U, F16>);
                    return reinterpret_cast<const void*>(&qpf2k_gdn::gdn_core_kernel<S, 4U, 2U, 4U, 8U, kQpf2kKrOpt<4U>, false, 1U, 4U, F16>);
                };
                const void* kr = f16
                    ? (vk ? kr_of.template operator()<GdnRecurrentStorage::value_key, true>() : kr_of.template operator()<KV::value, true>())
                    : (vk ? kr_of.template operator()<GdnRecurrentStorage::value_key>() : kr_of.template operator()<KV::value>());
                qpf2k_core_launch(kr, blocks, 128U, args, stream, "qpf2k GDN witness core launch");
                // fallback: pinned Nan 2 (exact for every input) from the untouched live state, iff the witness is set; its
                // exchange counters continue from the witness kernel's (count / 8 chunks published and consumed per block).
                // Witness excludes bit 64, so variant selects 2 columns and inherits f16 for a full core16 replay.
                xch.base = count / 8U; xch.decay_nan = xch.witness;
                ++xch.arrival;
                const void* pinned = vk ? variant.template operator()<GdnRecurrentStorage::value_key, 2U>() : variant.template operator()<KV::value, 2U>();
                qpf2k_core_launch(pinned, blocks, threads, args, stream, "qpf2k GDN witness fallback launch");
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_gdn::gdn_witness_commit_kernel<>), dim3(256), dim3(256), 0, stream,
                    static_cast<const unsigned*>(xch.witness), static_cast<const float*>(ws.state), matrix_state, state_elems / 4U,
                    static_cast<const std::uint16_t*>(ws.conv), conv_state, std::uint64_t{first.key_heads} * 2U * 128U * kGdnWindowKernel,
                    std::uint64_t{first.value_heads} * 128U * kGdnWindowKernel);
                check_hip(hipGetLastError(), "qpf2k GDN witness commit launch");
                qpf2k_gdn_witness(kr_g == 16U ? "recurrence gdn_core_kernel<16,8,4,8,6,lean,witness> + pinned fallback (fused conv + norm)"
                                  : kr_g == 8U ? "recurrence gdn_core_kernel<8,4,4,8,6,lean,witness> + pinned fallback (fused conv + norm)"
                                                : kr_opt == 1398U ? "recurrence gdn_core_kernel<4,2,4,8,1398,lean,witness> + pinned fallback (pending sigmoid + deferred norm)"
                                                : kr_opt == 886U ? "recurrence gdn_core_kernel<4,2,4,8,886,lean,witness> + pinned fallback (packed conv + deferred norm)"
                                               : kr_opt == 374U ? "recurrence gdn_core_kernel<4,2,4,8,374,lean,witness> + pinned fallback (fused conv + norm)"
                                               : "recurrence gdn_core_kernel<4,2,4,8,118,lean,witness> + pinned fallback (fused conv + norm)", count);
                return;
            }
            if (chunk16) {
                const void* pinned = vk ? chunk16_variant.template operator()<GdnRecurrentStorage::value_key, 2U>()
                                        : chunk16_variant.template operator()<KV::value, 2U>();
                const void* paired = vk ? chunk16_variant.template operator()<GdnRecurrentStorage::value_key, 1U>()
                                        : chunk16_variant.template operator()<KV::value, 1U>();
                qpf2k_core_launch(pinned, blocks, threads, args, stream, "qpf2k GDN fused core launch (chunk 16 pinned)");
                ++xch.arrival;
                qpf2k_core_launch(paired, blocks, threads, args, stream, "qpf2k GDN fused core launch (chunk 16 paired)");
            } else if (dispatch) {
                const void* pinned = vk ? variant.template operator()<GdnRecurrentStorage::value_key, 2U>() : variant.template operator()<KV::value, 2U>();
                const void* paired = vk ? variant.template operator()<GdnRecurrentStorage::value_key, 1U>() : variant.template operator()<KV::value, 1U>();
                qpf2k_core_launch(pinned, blocks, threads, args, stream, "qpf2k GDN fused core launch (pinned)");
                ++xch.arrival;
                qpf2k_core_launch(paired, blocks, threads, args, stream, "qpf2k GDN fused core launch (paired)");
            } else if (four) {
                const void* k4 = vk ? variant.template operator()<GdnRecurrentStorage::value_key, 3U>() : variant.template operator()<KV::value, 3U>();
                qpf2k_core_launch(k4, blocks, threads, args, stream, "qpf2k GDN fused core launch (4 columns)");
            } else
            qpf2k_core_launch(kernel, blocks, kQpf2kCoreWaves * 32U, args, stream, "qpf2k GDN fused core launch");
            if (chunk16)
                qpf2k_gdn_witness(chunk16_unpadded ? "recurrence gdn_core_kernel<4,2,4,16,2054,lean> x decay-NaN dispatch (chunk 16, single stage, unpadded)"
                                                   : "recurrence gdn_core_kernel<4,2,4,16,6,lean> x decay-NaN dispatch (chunk 16, single stage)", count);
            else if (four || dispatch)
                qpf2k_gdn_witness(four && dispatch ? "recurrence gdn_core_kernel<4,4,2,4,6,lean> x decay-NaN dispatch (fused conv + norm)"
                                  : four ? "recurrence gdn_core_kernel<4,4,2,4,6,lean> (fused conv + norm, 4 columns per lane group)"
                                         : "recurrence gdn_core_kernel<4,2,4,8,6,lean> x decay-NaN dispatch (fused conv + norm)", count);
            else
            qpf2k_gdn_witness(lean ? "recurrence gdn_core_kernel<4,2,4,8,2,lean> (fused conv + norm, lean epilogue)"
                              : pipe ? "recurrence gdn_core_kernel<4,2,4,8,2,pipe> (fused conv + pipelined norm)"
                                     : "recurrence gdn_core_kernel<4,2,4,8,2> (fused conv + norm)", count);
            return;
        }
    }
    // FAST lever qwen-gdn-fp16-chunk rev 2: conv fused into the chunk prep (q/k) and the scan (v), the gated norm
    // (+ A8 / CORE16) into the scan.
    if (fp16_chunk) {
        auto* matrix = reinterpret_cast<float*>(first.matrix_state.address);
        auto* conv_state = reinterpret_cast<std::uint16_t*>(first.conv_state.address);
        const std::uint64_t x16_width = static_cast<std::uint64_t>(first.conv_width) + first.value_width;
        const void* conv_w = reinterpret_cast<const void*>(first.tensors.conv.runtime->address);
        const auto conv_kind = small_weight_kind(first.tensors.conv.manifest->ggml_type);
        GdnFp16NormEpilogue epilogue{strided, reinterpret_cast<const void*>(first.tensors.norm.runtime->address),
            small_weight_kind(first.tensors.norm.manifest->ggml_type), first.head_dim, first.value_heads,
            config.model.rms_norm_epsilon, fused_a8 ? fused_a8->codes : nullptr, fused_a8 ? fused_a8->deltas : nullptr,
            nullptr, 0U, nullptr};
        if (core16 && !fused_a8) {
            core16->a16 = qpf2k_core16_scratch(stream, count);
            epilogue.core16 = core16->a16;
            epilogue.core16_pitch = q8_rows::tiled::scratch_pitch(6144U);
            epilogue.core16_overflow = q8_rows::dense_f16_overflow_counter();
        }
        if (config.recurrent_storage == GdnRecurrentStorage::value_key)
            launch_gdn_fp16_chunk_conv_fast<GdnRecurrentStorage::value_key>(strided, count, matrix, conv_state, conv_w, conv_kind,
                first.key_heads, first.value_heads, config.model.gdn_qk_l2_epsilon, stream, epilogue, x16, x16_width, first.conv_width);
        else launch_gdn_fp16_chunk_conv_fast<GdnRecurrentStorage::key_value>(strided, count, matrix, conv_state, conv_w, conv_kind,
            first.key_heads, first.value_heads, config.model.gdn_qk_l2_epsilon, stream, epilogue, x16, x16_width, first.conv_width);
        qpf2k_gdn_witness(x16 ? (epilogue.core16 ? "recurrence qwen-gdn-fp16-chunk rev 2 (BF16 rows, conv prep + staged scan + norm/CORE16)"
                                                 : "recurrence qwen-gdn-fp16-chunk rev 2 (BF16 rows, conv prep + staged scan + norm)")
                              : (epilogue.core16 ? "recurrence qwen-gdn-fp16-chunk rev 2 (conv prep + staged scan + norm/CORE16)"
                                                 : "recurrence qwen-gdn-fp16-chunk rev 2 (conv prep + staged scan + norm)"), count);
        check_hip(hipGetLastError(), "GDN fp16 chunk conv/recurrence/norm launch");
        return;
    }
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_prefill_conv_kernel<8U, 32U>), dim3(first.key_heads * 2U + first.value_heads),
        dim3(256), 0, stream, strided, count, reinterpret_cast<std::uint16_t*>(first.conv_state.address),
        reinterpret_cast<const void*>(first.tensors.conv.runtime->address), small_weight_kind(first.tensors.conv.manifest->ggml_type),
        first.key_heads, first.head_dim, config.model.gdn_qk_l2_epsilon);
    check_hip(hipGetLastError(), "GDN prefill conv rows launch");
    static const auto lanes = gdn_prefill_env("STRIX_GDN_LANES", 4U, {2U, 4U, 8U});
    static const auto waves = gdn_prefill_env("STRIX_GDN_LANE_WAVES", 8U, {2U, 4U, 8U});
    static const auto chunk_env = gdn_prefill_env("STRIX_GDN_CHUNK", 8U, {8U, 16U});
    const auto chunk = count % chunk_env ? 8U : chunk_env;
    auto* matrix = reinterpret_cast<float*>(first.matrix_state.address);
    const auto launch = [&]<GdnRecurrentStorage S, unsigned G, unsigned W, unsigned C>() {
        constexpr unsigned columns = W * 32U / G;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_prefill_lanes_kernel<S, G, W, C>), dim3(first.value_heads * (128U / columns)),
            dim3(W * 32U), 0, stream, strided, count, matrix, first.key_heads);
    };
    const auto by_chunk = [&]<GdnRecurrentStorage S, unsigned G, unsigned W>() {
        if (chunk == 16U) launch.template operator()<S, G, W, 16U>(); else launch.template operator()<S, G, W, 8U>();
    };
    const auto by_waves = [&]<GdnRecurrentStorage S, unsigned G>() {
        if (waves == 2U) by_chunk.template operator()<S, G, 2U>();
        else if (waves == 8U) by_chunk.template operator()<S, G, 8U>();
        else by_chunk.template operator()<S, G, 4U>();
    };
    const auto by_lanes = [&]<GdnRecurrentStorage S>() {
        if (lanes == 2U) by_waves.template operator()<S, 2U>();
        else if (lanes == 8U) by_waves.template operator()<S, 8U>();
        else by_waves.template operator()<S, 4U>();
    };
    // STRIX_GDN_FUSED_NORM=1 (prefill steps of >= 256 rows): gdn_gates::gdn_prefill_lanes_norm_kernel (recurrence + gated
    // RMSNorm, one block per value head; the same values as the lane recurrence followed by the norm rows kernel).
    static const bool fused_norm = [] { const char* v = std::getenv("STRIX_GDN_FUSED_NORM"); return v && v[0] == '1'; }();
    if (fused_norm && first.head_dim == 128U && count >= 256U && count % 8U == 0U) {
        const auto fused = [&]<GdnRecurrentStorage S, bool A8>() {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_gates::gdn_prefill_lanes_norm_kernel<S, 4U, 16U, 8U, A8>), dim3(first.value_heads), dim3(512),
                0, stream, strided, count, matrix, first.key_heads, first.value_heads,
                reinterpret_cast<const void*>(first.tensors.norm.runtime->address), small_weight_kind(first.tensors.norm.manifest->ggml_type),
                config.model.rms_norm_epsilon, fused_a8 ? fused_a8->codes : nullptr, fused_a8 ? fused_a8->deltas : nullptr);
        };
        const bool vk = config.recurrent_storage == GdnRecurrentStorage::value_key;
        if (vk && fused_a8) fused.template operator()<GdnRecurrentStorage::value_key, true>();
        else if (vk) fused.template operator()<GdnRecurrentStorage::value_key, false>();
        else if (fused_a8) fused.template operator()<GdnRecurrentStorage::key_value, true>();
        else fused.template operator()<GdnRecurrentStorage::key_value, false>();
        check_hip(hipGetLastError(), "GDN prefill fused recurrence/norm launch");
        return;
    }
    // qpf2k recurrence: QColsG lanes per column group, QColsC columns per group, QColsW waves per block (4 lanes x 32
    // keys, 2 columns per lane, 4 waves = 64 columns per block, 96 blocks; SGPR decay/beta).
    constexpr unsigned QColsG = 4U, QColsC = 2U, QColsW = 4U, QColsChunk = 8U, QColsOpt = 2U;
    using QCols = qpf2k_gdn::ColsShape<QColsG, QColsC, QColsW>;
    if ((qpf2k & 1U) && count % QColsChunk == 0U) {
        const auto cols = [&]<GdnRecurrentStorage S>() {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_gdn::gdn_qpf2k_cols_kernel<S, QColsG, QColsC, QColsW, QColsChunk, QColsOpt>),
                dim3(first.value_heads * (128U / QCols::columns)), dim3(QColsW * 32U), 0, stream, strided, count, matrix, first.key_heads,
                static_cast<const unsigned*>(nullptr));   // Nan 3 (pinned): the flag is not read
        };
        if (config.recurrent_storage == GdnRecurrentStorage::value_key) cols.template operator()<GdnRecurrentStorage::value_key>();
        else cols.template operator()<GdnRecurrentStorage::key_value>();
        qpf2k_gdn_witness("recurrence gdn_qpf2k_cols_kernel<4,2,4,8,2>", count);
    } else if (config.recurrent_storage == GdnRecurrentStorage::value_key) by_lanes.template operator()<GdnRecurrentStorage::value_key>();
    else by_lanes.template operator()<GdnRecurrentStorage::key_value>();
    check_hip(hipGetLastError(), "GDN prefill lane recurrence launch");
    if (fused_a8)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_prefill_norm_rows_kernel<8U, true>), dim3((count * first.value_heads + 7U) / 8U), dim3(256),
            0, stream, strided, count, first.value_heads,
            reinterpret_cast<const void*>(first.tensors.norm.runtime->address), small_weight_kind(first.tensors.norm.manifest->ggml_type),
            first.head_dim, config.model.rms_norm_epsilon, fused_a8->codes, fused_a8->deltas);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_prefill_norm_rows_kernel<8U>), dim3((count * first.value_heads + 7U) / 8U), dim3(256),
        0, stream, strided, count, first.value_heads,
        reinterpret_cast<const void*>(first.tensors.norm.runtime->address), small_weight_kind(first.tensors.norm.manifest->ggml_type),
        first.head_dim, config.model.rms_norm_epsilon);
    check_hip(hipGetLastError(), "GDN prefill gated RMSNorm rows launch");
}
// Debug snapshot of what the causal core reads and writes: q/k/v (and beta, gate, core when `outputs`) of every
// row, the conv history and the recurrent state.
struct GdnPrefillSnapshot {
    struct Region { std::uintptr_t device; std::uint64_t pitch, width, rows, offset; };
    std::vector<Region> regions;
    std::vector<std::uint8_t> bytes;
    GdnPrefillSnapshot(const GdnKnownWindowRow& first, const GdnPrefillRows& s, std::uint32_t count, bool outputs,
        std::uint64_t qkv_width = 0) {
        const auto pitch = s.stride * 4U;
        const auto add = [&](const void* device, std::uint64_t width, std::uint64_t rows, std::uint64_t row_pitch) {
            regions.push_back({reinterpret_cast<std::uintptr_t>(device), row_pitch, width, rows, bytes.size()});
            bytes.resize(bytes.size() + width * rows);
        };
        add(s.qkv, (qkv_width ? qkv_width : first.conv_width) * 4ULL, count, pitch);
        if (outputs) {
            add(s.beta, first.value_heads * 4ULL, count, pitch);
            add(s.gate, first.value_heads * 4ULL, count, pitch);
            add(s.core, first.value_width * 4ULL, count, pitch);
        }
        add(reinterpret_cast<const void*>(first.conv_state.address), first.conv_state.bytes, 1, first.conv_state.bytes);
        add(reinterpret_cast<const void*>(first.matrix_state.address), first.matrix_state.bytes, 1, first.matrix_state.bytes);
        check_hip(hipStreamSynchronize(hip_execution_stream()), "GDN prefill check synchronize");
        for (const auto& r : regions)
            check_hip(hipMemcpy2D(bytes.data() + r.offset, r.width, reinterpret_cast<const void*>(r.device), r.pitch, r.width,
                r.rows, hipMemcpyDeviceToHost), "GDN prefill check snapshot");
    }
    void restore() const {
        for (const auto& r : regions)
            check_hip(hipMemcpy2D(reinterpret_cast<void*>(r.device), r.pitch, bytes.data() + r.offset, r.width, r.width,
                r.rows, hipMemcpyHostToDevice), "GDN prefill check restore");
    }
};
void enqueue_gdn_prefill_rows(const std::vector<std::array<GdnKnownWindowRow, 8U>>& tiles, const GdnPrefillRows& strided,
    const HipGdnLauncherConfig& config) {
    const auto stream = hip_execution_stream();
    const auto& first = tiles[0][0];
    const auto count = static_cast<std::uint32_t>(tiles.size() * 8U);
    // Q8 projections run W rows per launch (64/32/16, R1-exact wide q8_rows; each weight byte is
    // read from DRAM about once per launch). STRIX_PREFILL_GDN_WIDE=8 keeps 8-row projections.
    static const std::uint32_t wide_env = [] {
        const char* v = std::getenv("STRIX_PREFILL_GDN_WIDE");
        const int n = v ? std::atoi(v) : 64;
        return n == 8 || n == 16 || n == 32 ? static_cast<std::uint32_t>(n) : 64U;
    }();
    const auto wide = wide_env;
    // STRIX_PREFILL_DENSE_A8_ROWS = 64 / 128 / 256 (default): rows per dense A8 projection launch.
    static const std::uint32_t dense_env = [] {
        const char* v = std::getenv("STRIX_PREFILL_DENSE_A8_ROWS");
        const int n = v ? std::atoi(v) : 256;
        return n == 64 || n == 128 ? static_cast<std::uint32_t>(n) : 256U;
    }();
    const std::uint32_t dense_wide = config.dense_a8_projections && wide >= 64U ? dense_env : 0U;
    const auto project_in = [&]<std::uint32_t W>(std::size_t first_tile) {
        GdnInputRowsSite<W> in{};
        in.segments = q8_rows::segments({{reinterpret_cast<const void*>(first.tensors.qkv.runtime->address), first.conv_width},
            {reinterpret_cast<const void*>(first.tensors.z.runtime->address), first.value_width}});
        in.segments.types[0] = q8_rows::segment_type(first.tensors.qkv.manifest->ggml_type);  // GSQ-RCO typed (0 = Q8_0)
        in.segments.types[1] = q8_rows::segment_type(first.tensors.z.manifest->ggml_type);
        in.width = first.hidden; in.outputs = first.conv_width + first.value_width;
        for (unsigned r = 0; r < W; ++r) {
            const auto& row = tiles[first_tile + r / 8U][r % 8U];
            in.input[r] = reinterpret_cast<const void*>(row.input.address);
            in.output[0][r] = row.workspace.qkv; in.output[1][r] = row.workspace.z;
        }
        q8_rows::launch_prefill(in, config.dense_a8_projections, config.dense_f16_prefill, stream, false, config.dense_g64_prefill);
        check_hip(hipGetLastError(), "GDN prefill Q8 input projection launch");
    };
    // Dense A8 with 256-row output launches over whole 256-row steps: the gated RMSNorm quantizes the out
    // projection's input (no quantizer launch). STRIX_GDN_FUSED_A8=0 keeps the quantizer launches.
    static const bool fused_env = [] { const char* v = std::getenv("STRIX_GDN_FUSED_A8"); return !v || v[0] != '0'; }();
    const q8_rows::DenseA8Slot fused_slot = fused_env && !config.dense_f16_prefill && !config.dense_g64_prefill && dense_wide >= q8_rows::kDenseA8WideRows && count % q8_rows::kDenseA8WideRows == 0U &&
            count <= kGdnFusedA8Rows && q8_rows::dense_a8_width(first.value_width) && first.tensors.out.manifest->ggml_type == 8U &&
            reinterpret_cast<std::uintptr_t>(first.tensors.out.runtime->address) % 4U == 0U ? gdn_fused_a8_scratch() : q8_rows::DenseA8Slot{};
    const q8_rows::DenseA8Slot* fused_a8 = nullptr;
    const auto project_out = [&]<std::uint32_t W>(std::size_t first_tile) {
        GdnOutputRowsSite<W> out{};
        out.segments = q8_rows::segments({{reinterpret_cast<const void*>(first.tensors.out.runtime->address), first.hidden}});
        out.segments.types[0] = q8_rows::segment_type(first.tensors.out.manifest->ggml_type);
        out.width = first.value_width; out.outputs = first.hidden;
        for (unsigned r = 0; r < W; ++r) {
            const auto& row = tiles[first_tile + r / 8U][r % 8U];
            out.input[r] = row.workspace.core;
            out.output[r] = reinterpret_cast<std::uint16_t*>(row.output.address);
        }
        if constexpr (W == q8_rows::kDenseA8WideRows) {
            // Rows quantized by the gated RMSNorm (fused_a8): the projection alone.
            if (fused_a8) {
                const std::uint64_t row0 = first_tile * 8U, width = first.value_width;
                const q8_rows::DenseA8Slot slot{fused_a8->codes + row0 * width, fused_a8->deltas + row0 / W * (width / 32U * W)};
                // STRIX_GDN_FUSED_A8_CHECK=1: the quantizer kernel on the same rows, compared byte for byte.
                static const bool fused_check = [] { const char* v = std::getenv("STRIX_GDN_FUSED_A8_CHECK"); return v && v[0] == '1'; }();
                if (fused_check) {
                    const auto reference = q8_rows::dense_a8_wide_slot(q8_rows::a8_family_of<decltype(out)>());
                    const auto items = static_cast<std::uint32_t>(W * (width / 8U));
                    hipLaunchKernelGGL(q8_rows::dense_a8_quantize_kernel<decltype(out)>, dim3((items + 255U) / 256U), dim3(256U), 0, stream,
                        out, reference.codes, reference.deltas);
                    check_hip(hipStreamSynchronize(stream), "GDN fused A8 check");
                    const std::uint64_t code_bytes = W * width, delta_bytes = width / 32U * W * 4U;
                    std::vector<unsigned char> a(code_bytes + delta_bytes), b(code_bytes + delta_bytes);
                    check_hip(hipMemcpy(a.data(), reference.codes, code_bytes, hipMemcpyDeviceToHost), "GDN fused A8 check copy");
                    check_hip(hipMemcpy(a.data() + code_bytes, reference.deltas, delta_bytes, hipMemcpyDeviceToHost), "GDN fused A8 check copy");
                    check_hip(hipMemcpy(b.data(), slot.codes, code_bytes, hipMemcpyDeviceToHost), "GDN fused A8 check copy");
                    check_hip(hipMemcpy(b.data() + code_bytes, slot.deltas, delta_bytes, hipMemcpyDeviceToHost), "GDN fused A8 check copy");
                    std::uint64_t differ = 0;
                    for (std::uint64_t i = 0; i < a.size(); ++i) differ += a[i] != b[i];
                    std::fprintf(stderr, "GDN fused A8 check: rows %u bytes %llu differ %llu\n", W, static_cast<unsigned long long>(a.size()),
                        static_cast<unsigned long long>(differ));
                    if (differ) throw std::runtime_error("GDN fused A8 codes differ from the quantizer kernel");
                }
                q8_rows::launch_dense_a8_projection(out, slot, stream);
                check_hip(hipGetLastError(), "GDN prefill Q8 output projection launch");
                return;
            }
        }
        q8_rows::launch_prefill(out, config.dense_a8_projections, config.dense_f16_prefill, stream, false, config.dense_g64_prefill);
        check_hip(hipGetLastError(), "GDN prefill Q8 output projection launch");
    };
    static const std::uint32_t dense_rows_env = [] {
        const char* v = std::getenv("STRIX_INTPREFILL_DENSE_ROWS");
        const int n = v ? std::atoi(v) : 0;
        return n == 512 || n == 1024 ? static_cast<std::uint32_t>(n) : 0U;
    }();
    // Row r of tiles [t, t + rows / 8) at view(0) + r * stride for every view of the step, or false.
    const auto uniform = [&](std::size_t t, std::uint32_t rows, const auto& view, std::uint64_t& stride) {
        const auto at = [&](std::uint32_t r) { return reinterpret_cast<std::uintptr_t>(view(tiles[t + r / 8U][r % 8U])); };
        const auto a0 = at(0U), a1 = at(1U);
        if (a1 <= a0) return false;
        stride = a1 - a0;
        for (std::uint32_t r = 2U; r < rows; ++r) if (at(r) != a0 + r * stride) return false;
        return true;
    };
    const auto strided_in = [&]<std::uint32_t W>(std::size_t t) {
        GdnInputStridedRowsSite<W> in{};
        std::uint64_t si = 0, sq = 0, sz = 0;
        if (!uniform(t, W, [](const auto& row) { return reinterpret_cast<const void*>(row.input.address); }, si) ||
            !uniform(t, W, [](const auto& row) { return static_cast<const void*>(row.workspace.qkv); }, sq) ||
            !uniform(t, W, [](const auto& row) { return static_cast<const void*>(row.workspace.z); }, sz) || sq != sz)
            return false;
        in.segments = q8_rows::segments({{reinterpret_cast<const void*>(first.tensors.qkv.runtime->address), first.conv_width},
            {reinterpret_cast<const void*>(first.tensors.z.runtime->address), first.value_width}});
        in.width = first.hidden; in.outputs = first.conv_width + first.value_width;
        in.input0 = reinterpret_cast<const unsigned char*>(tiles[t][0].input.address); in.input_stride = si;
        in.output0[0] = reinterpret_cast<unsigned char*>(tiles[t][0].workspace.qkv);
        in.output0[1] = reinterpret_cast<unsigned char*>(tiles[t][0].workspace.z); in.output_stride = sq;
        q8_rows::launch_dense_f16(in, stream, false);
        check_hip(hipGetLastError(), "GDN prefill strided input projection launch");
        return true;
    };
    // Largest width that divides what is left, so every row runs exactly once. The output projection keeps 256-row
    // launches (fixture p4: 6144->2560 at 512/1024 rows was slower with cold weights, equal warm).
    const auto none = [&]<std::uint32_t W>(std::size_t) { return false; };
    // STRIX_QPF2K_DENSE=1: every row the launches below would give to launch_dense_f16 (launches of >= 32 rows: all rows
    // when the count is a multiple of 32 and the widths reach 32) runs in one q8_rows::tiled launch per projection, same
    // bits; anything else keeps the launches below.
    const bool tiled_rows = config.dense_f16_prefill && !config.dense_g64_prefill && wide >= 32U && count % 32U == 0U &&
        count >= q8_rows::tiled::min_rows();
    // Causal core. STRIX_GDN_PREFILL_ROWS=0 keeps the per-tile gates and the serial prepare/recurrence/norm
    // kernels; STRIX_GDN_PREFILL_CHECK=1 runs both on the same inputs and compares every written byte.
    static const bool rows_env = [] { const char* v = std::getenv("STRIX_GDN_PREFILL_ROWS"); return !v || v[0] != '0'; }();
    static const bool check = [] { const char* v = std::getenv("STRIX_GDN_PREFILL_CHECK"); return v && v[0] == '1'; }();
    const auto input_stride = gdn_prefill_input_stride(tiles);
    const bool rows = rows_env && input_stride && gdn_prefill_rows_shape(first, strided, count);
    static const bool qpf2k_check = [] { const char* v = std::getenv("STRIX_QPF2K_GDN_CHECK"); return v && v[0] == '1'; }();
    // FAST route qwen-gdn-fp16-chunk on the row path: the tiled input projection writes BF16 rows for it.
    const bool fast16 = rows && !check && !qpf2k_check && gdn_fp16_chunk_enabled(first, strided);
    const std::uint16_t* x16 = nullptr;
    const auto tiled_in = [&]() {
        if (!tiled_rows || !q8_rows::tiled::site_enabled(1U) || count < q8_rows::tiled::min_rows(1U)) return false;
        std::uint64_t si = 0, sq = 0, sz = 0;
        if (!uniform(0U, count, [](const auto& row) { return reinterpret_cast<const void*>(row.input.address); }, si) ||
            !uniform(0U, count, [](const auto& row) { return static_cast<const void*>(row.workspace.qkv); }, sq) ||
            !uniform(0U, count, [](const auto& row) { return static_cast<const void*>(row.workspace.z); }, sz) || sq != sz)
            return false;
        if (fast16) {
            // FAST route: the same values as BF16 rows only (its kernels are the only readers of qkv / z).
            GdnInputTiled16Site in16{};
            in16.segments = q8_rows::segments({{reinterpret_cast<const void*>(first.tensors.qkv.runtime->address), first.conv_width},
                {reinterpret_cast<const void*>(first.tensors.z.runtime->address), first.value_width}});
            in16.width = first.hidden; in16.outputs = first.conv_width + first.value_width;
            in16.input0 = reinterpret_cast<const unsigned char*>(tiles[0][0].input.address); in16.input_stride = si;
            in16.output_stride = in16.outputs; in16.z_offset = first.conv_width;
            if (q8_rows::tiled::supported(in16, count)) {
                in16.output = gdn_fp16_chunk_input16(stream, count, in16.outputs);
                q8_rows::tiled::launch(in16, count, stream);
                check_hip(hipGetLastError(), "GDN prefill tiled BF16 input projection launch");
                x16 = in16.output;
                return true;
            }
        }
        GdnInputTiledSite in{};
        in.segments = q8_rows::segments({{reinterpret_cast<const void*>(first.tensors.qkv.runtime->address), first.conv_width},
            {reinterpret_cast<const void*>(first.tensors.z.runtime->address), first.value_width}});
        in.width = first.hidden; in.outputs = first.conv_width + first.value_width;
        in.input0 = reinterpret_cast<const unsigned char*>(tiles[0][0].input.address); in.input_stride = si;
        in.output0[0] = reinterpret_cast<unsigned char*>(tiles[0][0].workspace.qkv);
        in.output0[1] = reinterpret_cast<unsigned char*>(tiles[0][0].workspace.z); in.output_stride = sq;
        if (!q8_rows::tiled::supported(in, count)) return false;
        q8_rows::tiled::launch(in, count, stream);
        check_hip(hipGetLastError(), "GDN prefill tiled input projection launch");
        return true;
    };
    Qpf2kCore16Route core16;
    const auto prepare_core16 = [&]() {
        static const bool enabled = [] { const char* v = std::getenv("STRIX_QPF2K_GDN_CORE16"); return v && v[0] == '1'; }();
        static const auto minimum = [] {
            const char* v = std::getenv("STRIX_QPF2K_GDN_CORE16_MIN_ROWS");
            const int n = v ? std::atoi(v) : 2048;
            return n >= 0 ? static_cast<std::uint32_t>(n) : 2048U;
        }();
        if (!enabled || qpf2k::suppressed(qpf2k::Item::gdn_core16) || !tiled_rows || !q8_rows::tiled::site_enabled(2U) || count < minimum ||
            first.value_width != 6144U || first.head_dim != 128U) return false;
        hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
        if (hipStreamIsCapturing(stream, &capture) != hipSuccess || capture != hipStreamCaptureStatusNone) return false;
        std::uint64_t si = 0, so = 0;
        if (!uniform(0U, count, [](const auto& row) { return static_cast<const void*>(row.workspace.core); }, si) ||
            !uniform(0U, count, [](const auto& row) { return reinterpret_cast<const void*>(row.output.address); }, so)) return false;
        auto& out = core16.site;
        out.segments = q8_rows::segments({{reinterpret_cast<const void*>(first.tensors.out.runtime->address), first.hidden}});
        out.width = first.value_width; out.outputs = first.hidden;
        out.input0 = reinterpret_cast<const unsigned char*>(first.workspace.core); out.input_stride = si;
        out.output0 = reinterpret_cast<unsigned char*>(first.output.address); out.output_stride = so;
        if (!q8_rows::tiled::supported(out, count, q8_rows::tiled::kPre)) return false;
        core16.form = gdn_prefill_env("STRIX_QPF2K_GDN_CORE16_FORM", 4U, {0U, 1U, 2U, 3U, 4U});
        static const auto raster = [] {
            const char* v = std::getenv("STRIX_QPF2K_GDN_CORE16_RASTER");
            const int n = v ? std::atoi(v) : 2;
            return n >= 1 && n <= 64 ? static_cast<unsigned>(n) : 2U;
        }();
        core16.raster = raster;
        return true;                        // the core still must select a resident F16 lean variant
    };
    const auto tiled_out = [&]() {
        if (core16.a16) {
            // Same FP16 bits and K16 order as inline FP32 staging. Forms 3/4 have the same per-output arithmetic;
            // the fixture gates forms 0..4 bitwise in kPre and inline modes. Bypass launch()'s conversion/form policy.
            q8_rows::tiled::launch_projection<GdnOutputTiledSite, q8_rows::tiled::kPre>(core16.site, count, stream, false,
                core16.form, core16.raster, core16.a16);
            check_hip(hipGetLastError(), "qpf2k GDN core16 output projection launch");
            static std::once_flag once;
            std::call_once(once, [&] {
                std::fprintf(stderr, "qpf2k GDN core16: fp16 core rows + output projection form %u raster %u rows %u\n",
                    core16.form, core16.raster, count);
            });
            return true;
        }
        if (!tiled_rows || !q8_rows::tiled::site_enabled(2U)) return false;
        std::uint64_t si = 0, so = 0;
        if (!uniform(0U, count, [](const auto& row) { return static_cast<const void*>(row.workspace.core); }, si) ||
            !uniform(0U, count, [](const auto& row) { return reinterpret_cast<const void*>(row.output.address); }, so))
            return false;
        GdnOutputTiledSite out{};
        out.segments = q8_rows::segments({{reinterpret_cast<const void*>(first.tensors.out.runtime->address), first.hidden}});
        out.width = first.value_width; out.outputs = first.hidden;
        out.input0 = reinterpret_cast<const unsigned char*>(tiles[0][0].workspace.core); out.input_stride = si;
        out.output0 = reinterpret_cast<unsigned char*>(tiles[0][0].output.address); out.output_stride = so;
        if (!q8_rows::tiled::supported(out, count)) return false;
        q8_rows::tiled::launch(out, count, stream);
        check_hip(hipGetLastError(), "GDN prefill tiled output projection launch");
        return true;
    };
    const auto each_width = [&](auto&& project, auto&& strided) {
        for (std::size_t t = 0; t < tiles.size();) {
            const auto left = (tiles.size() - t) * 8U;
            if (!qpf2k::suppressed(qpf2k::Item::dense_rows) && dense_rows_env && config.dense_f16_prefill && dense_wide >= 256U && left >= dense_rows_env) {
                const bool done = dense_rows_env == 1024U ? strided.template operator()<1024U>(t) : strided.template operator()<512U>(t);
                if (done) { t += dense_rows_env / 8U; continue; }
            }
            // Dense A8: one WMMA launch per 256 (128) rows, so each weight byte streams once per 256 rows.
            if (dense_wide >= 256U && left >= 256U) { project.template operator()<256U>(t); t += 32U; }
            else if (dense_wide >= 128U && left >= 128U) { project.template operator()<128U>(t); t += 16U; }
            else if (wide >= 64U && left >= 64U) { project.template operator()<64U>(t); t += 8U; }
            else if (wide >= 32U && left >= 32U) { project.template operator()<32U>(t); t += 4U; }
            else if (wide >= 16U && left >= 16U) { project.template operator()<16U>(t); t += 2U; }
            else { project.template operator()<8U>(t); t += 1U; }
        }
    };
    if (!tiled_in()) each_width(project_in, strided_in);
    // Decide before the producer: check modes and every fallback must retain FP32 core rows.
    const bool core16_candidate = rows && !check && !qpf2k_check && prepare_core16();
    if (rows && qpf2k_check && qpf2k_gdn_parts() && !check) {
        // The production row kernels on a snapshot of the inputs, then the qpf2k kernels on the same inputs.
        const GdnPrefillSnapshot inputs(first, strided, count, false);
        enqueue_gdn_prefill_core_rows(tiles, strided, input_stride, config, nullptr, 0U);
        // With the fused core the v part of qkv keeps the raw projection values: compare q/k and every other output.
        const std::uint64_t qkv_width = (qpf2k_gdn_parts() & 4U) ? std::uint64_t{first.key_heads} * 2U * first.head_dim : 0U;
        const GdnPrefillSnapshot reference(first, strided, count, true, qkv_width);
        inputs.restore();
        enqueue_gdn_prefill_core_rows(tiles, strided, input_stride, config, nullptr);
        const GdnPrefillSnapshot candidate(first, strided, count, true, qkv_width);
        std::uint64_t differ = 0, mismatch = UINT64_MAX;
        for (std::uint64_t i = 0; i < reference.bytes.size(); ++i)
            if (reference.bytes[i] != candidate.bytes[i]) { ++differ; if (mismatch == UINT64_MAX) mismatch = i; }
        std::fprintf(stderr, "QPF2K GDN check: layer-rows %u parts %u bytes %llu differ %llu first %lld\n", count, qpf2k_gdn_parts(),
            static_cast<unsigned long long>(reference.bytes.size()), static_cast<unsigned long long>(differ),
            mismatch == UINT64_MAX ? -1LL : static_cast<long long>(mismatch));
        if (differ) throw std::runtime_error("qpf2k GDN prefill kernels differ from the production row kernels");
        each_width(project_out, none);
        return;
    }
    if (!rows || !check) {
        if (rows) {
            if (fused_slot.codes) fused_a8 = &fused_slot;
            enqueue_gdn_prefill_core_rows(tiles, strided, input_stride, config, fused_a8, qpf2k_gdn_parts(),
                core16_candidate ? &core16 : nullptr, x16);
        } else enqueue_gdn_prefill_core_tiles(tiles, strided, config);
        if (!tiled_out()) each_width(project_out, none);
        return;
    }
    const GdnPrefillSnapshot inputs(first, strided, count, false);
    enqueue_gdn_prefill_core_tiles(tiles, strided, config);
    const GdnPrefillSnapshot reference(first, strided, count, true);
    inputs.restore();
    enqueue_gdn_prefill_core_rows(tiles, strided, input_stride, config, nullptr);
    const GdnPrefillSnapshot candidate(first, strided, count, true);
    std::uint64_t differ = 0, mismatch = UINT64_MAX;
    for (std::uint64_t i = 0; i < reference.bytes.size(); ++i)
        if (reference.bytes[i] != candidate.bytes[i]) { ++differ; if (mismatch == UINT64_MAX) mismatch = i; }
    std::fprintf(stderr, "GDN prefill check: rows %u bytes %llu differ %llu first %lld\n", count,
        static_cast<unsigned long long>(reference.bytes.size()), static_cast<unsigned long long>(differ),
        mismatch == UINT64_MAX ? -1LL : static_cast<long long>(mismatch));
    if (differ) throw std::runtime_error("GDN prefill row kernels differ from the tile/serial kernels");
    if (!tiled_out()) each_width(project_out, none);
}
}  // namespace

void gdn_known_prefill_rows_preflight(std::span<const ExecutionDispatchContext> rows, const HipGdnLauncherConfig& config) {
    GdnPrefillRows strided{};
    static_cast<void>(prepare_gdn_prefill_rows(rows, config, strided));
}
void gdn_known_prefill_rows_launch(std::span<const ExecutionDispatchContext> rows, const HipGdnLauncherConfig& config) {
#if defined(STRIX_HAS_EXPERIMENTAL_DENSE_TRELLIS)
    HipDenseTrellisScope arithmetic(config.dense_trellis_projections);
#endif
    GdnPrefillRows strided{};
    const auto tiles = prepare_gdn_prefill_rows(rows, config, strided);
    enqueue_gdn_prefill_rows(tiles, strided, config);
}

void gdn_known_window_r2_preflight(std::span<const ExecutionDispatchContext, 2> rows,
    const HipGdnLauncherConfig& config) {
    static_cast<void>(prepare_gdn_known_window_rows(rows, config, Format::all_q8));
}
void gdn_known_window_r2_launch(std::span<const ExecutionDispatchContext, 2> rows,
    const HipGdnLauncherConfig& config) {
#if defined(STRIX_HAS_EXPERIMENTAL_DENSE_TRELLIS)
    HipDenseTrellisScope arithmetic(config.dense_trellis_projections);
#endif
    // Repeat current-view validation at this public entry; preflight isn't a permit.
    const auto prepared = prepare_gdn_known_window_rows(rows, config, Format::all_q8);
    enqueue_gdn_known_window_inputs(prepared, Format::all_q8, config);
    enqueue_gdn_known_window_completion(prepared, config);
}
void gdn_known_window_r2_f32_gates_preflight(std::span<const ExecutionDispatchContext, 2> rows,
    const HipGdnLauncherConfig& config) {
    static_cast<void>(prepare_gdn_known_window_rows(rows, config, Format::q8_large_f32_gates));
}
void gdn_known_window_r2_f32_gates_launch(std::span<const ExecutionDispatchContext, 2> rows,
    const HipGdnLauncherConfig& config) {
#if defined(STRIX_HAS_EXPERIMENTAL_DENSE_TRELLIS)
    HipDenseTrellisScope arithmetic(config.dense_trellis_projections);
#endif
    const auto prepared = prepare_gdn_known_window_rows(rows, config, Format::q8_large_f32_gates);
    enqueue_gdn_known_window_inputs(prepared, Format::q8_large_f32_gates, config);
    enqueue_gdn_known_window_completion(prepared, config);
}

namespace {
struct GdnWindowShape {
    std::uint32_t hidden, key_heads, value_heads, head_dim, key_width, value_width, conv_width;
};
GdnWindowShape gdn_window_shape(const ModelContract& model) {
    const auto head_dim = static_cast<std::uint32_t>(model.gdn_key_head_dim);
    const auto key_width = static_cast<std::uint32_t>(model.gdn_key_heads) * head_dim;
    const auto value_width = static_cast<std::uint32_t>(model.gdn_value_heads) * head_dim;
    return {static_cast<std::uint32_t>(model.hidden_size), static_cast<std::uint32_t>(model.gdn_key_heads),
        static_cast<std::uint32_t>(model.gdn_value_heads), head_dim, key_width, value_width, key_width * 2U + value_width};
}
GdnWindowSaved gdn_window_saved(const ModelContract& model, const void* retained) {
    const auto shape = gdn_window_shape(model);
    const auto align = [](std::uint64_t bytes) { return (bytes + 255U) / 256U * 256U; };
    GdnWindowSaved saved{reinterpret_cast<std::uintptr_t>(retained), 0, align(shape.conv_width * 2ULL), 0, 0, 0};
    saved.v = saved.k + shape.key_width * 4ULL;
    saved.beta = saved.v + shape.value_width * 4ULL;
    saved.gate = saved.beta + shape.value_heads * 4ULL;
    saved.row_bytes = align(saved.gate + shape.value_heads * 4ULL);
    return saved;
}
template <unsigned Rows>
GdnWindowRows<Rows> gdn_window_rows(std::span<const GdnWindowRow> input, const GdnWindowShape& shape) {
    GdnWindowRows<Rows> rows{};
    for (unsigned r = 0; r < Rows; ++r) {
        rows.qkv[r] = input[r].workspace; rows.z[r] = rows.qkv[r] + shape.conv_width;
        rows.beta[r] = rows.z[r] + shape.value_width; rows.gate[r] = rows.beta[r] + shape.value_heads;
        rows.core[r] = rows.gate[r] + shape.value_heads;
    }
    return rows;
}
void validate_gdn_window(const GdnWindowLayer& layer, std::span<const GdnWindowRow> rows, bool window,
    const GdnWindowControl* control, const HipGdnLauncherConfig& config, void* stream) {
    validate_config(config);
    if (!config.wave8_q8_input_projections || !config.wave8_q8_output_projection || config.model.gdn_conv_kernel != kGdnWindowKernel)
        throw std::invalid_argument("GDN window requires wave8 Q8 input/output projections and conv kernel 4");
    if (!control || !stream || !layer.matrix || !layer.conv_state || !layer.retained ||
        reinterpret_cast<std::uintptr_t>(layer.retained) % 256U)
        throw std::invalid_argument("GDN window state, retained rows, control or stream is missing");
    if (!window) return;
    if (rows.empty() || rows.size() > gdn_window_max_rows)
        throw std::invalid_argument("GDN window requires 1..8 rows");
    if (!layer.qkv || !layer.z || !layer.beta || !layer.alpha || !layer.out || !layer.conv || !layer.dt || !layer.a || !layer.norm)
        throw std::invalid_argument("GDN window weights are missing");
    if (layer.gate_type != 0U && layer.gate_type != 8U && !(layer.gate_type == 30U && q8_rows::gsq_dense_enabled()))
        throw std::invalid_argument("GDN window beta/alpha must be F32 or Q8_0 (BF16 for GSQ-RCO files)");
    for (const auto type : {layer.qkv_type, layer.z_type, layer.out_type})
        if (type != 8U && !(config.dense_a8_projections && q8_rows::dense_a8_type(type)))
            throw std::invalid_argument("GDN window qkv/z/out must be Q8_0 (or GSQ dense types under dense A8)");
    (void)small_weight_kind(layer.conv_type); (void)small_weight_kind(layer.dt_type);
    (void)small_weight_kind(layer.a_type); (void)small_weight_kind(layer.norm_type);
    for (const auto& row : rows)
        if (!row.input || !row.output || !row.workspace || reinterpret_cast<std::uintptr_t>(row.workspace) % 16U ||
            reinterpret_cast<std::uintptr_t>(row.input) % 16U)
            throw std::invalid_argument("GDN window row views are missing or misaligned");
    if (config.dense_a8_projections)
        for (const auto width : {static_cast<std::uint64_t>(config.model.hidden_size), static_cast<std::uint64_t>(gdn_window_shape(config.model).value_width)})
            if (!q8_rows::dense_a8_width(width)) throw std::invalid_argument("GDN window dense A8 width is not admitted");
}
// parts: 1 input projections and gates, 2 prepare and recurrence (the only parts that read or write
// the layer's recurrent/conv state, retained rows and control), 4 gated norm and output projection.
// Parts 1 and 4 are per-row arithmetic over weights only, so a cohort runs them once for the rows of
// several sessions and part 2 per session.
template <unsigned Rows>
void enqueue_gdn_window(const GdnWindowLayer& layer, std::span<const GdnWindowRow> input,
    const GdnWindowControl* control, const HipGdnLauncherConfig& config, hipStream_t stream, unsigned parts = 7U) {
    const auto shape = gdn_window_shape(config.model);
    const auto rows = gdn_window_rows<Rows>(input, shape);
    bool fused_norm = false;
    q8_rows::DenseA8Slot fused_slot{};
    if (parts & 1U) {
    GdnWidePointers<Rows> x{};
    for (unsigned r = 0; r < Rows; ++r) x.input[r] = input[r].input;
    // qkv, z and Q8 beta/alpha share one q8_rows launch.
    GdnInputRowsSite<Rows> in{};
    const bool q8_gates = layer.gate_type == 8U;
    in.segments = q8_gates ? q8_rows::segments({{layer.qkv, shape.conv_width}, {layer.z, shape.value_width},
        {layer.beta, shape.value_heads}, {layer.alpha, shape.value_heads}}) :
        q8_rows::segments({{layer.qkv, shape.conv_width}, {layer.z, shape.value_width}});
    in.segments.types[0] = q8_rows::segment_type(layer.qkv_type); in.segments.types[1] = q8_rows::segment_type(layer.z_type);
    in.width = shape.hidden; in.outputs = in.segments.first[in.segments.count];
    for (unsigned r = 0; r < Rows; ++r) {
        in.input[r] = input[r].input;
        in.output[0][r] = rows.qkv[r]; in.output[1][r] = rows.z[r]; in.output[2][r] = rows.beta[r]; in.output[3][r] = rows.gate[r];
    }
    q8_rows::launch(in, config.dense_a8_projections, stream);
    check_hip(hipGetLastError(), "GDN window Q8 input projection launch");
    if (!q8_gates) {
        // Any-order (STRIX_ANYORDER=0 disables): the F32 gates read the same inputs as the Q8 projection queued
        // just before and write only beta/gate, so they may run alongside it. STRIX_GDN_GATES_ANYORDER=0|1
        // (measurement) sets this launch alone.
        static const bool any_order = [] {
            const char* v = std::getenv("STRIX_GDN_GATES_ANYORDER");
            return v && (v[0] == '0' || v[0] == '1') ? v[0] == '1' : strix_any_order_enabled();
        }();
        const auto gates = [&]<class Weight>() {
            if (any_order)
                hipExtLaunchKernelGGL(HIP_KERNEL_NAME(gdn_window_f32_gates_kernel<Rows, Weight>), dim3(shape.value_heads * 2U),
                    dim3(kProjectionThreads), 0, stream, nullptr, nullptr, hipExtAnyOrderLaunch, x,
                    static_cast<const Weight*>(layer.beta), static_cast<const Weight*>(layer.alpha), rows, shape.hidden,
                    shape.value_heads);
            else hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_window_f32_gates_kernel<Rows, Weight>), dim3(shape.value_heads * 2U),
                dim3(kProjectionThreads), 0, stream, x, static_cast<const Weight*>(layer.beta),
                static_cast<const Weight*>(layer.alpha), rows, shape.hidden, shape.value_heads);
        };
        if (layer.gate_type == 30U) gates.template operator()<std::uint16_t>(); else gates.template operator()<float>();
        check_hip(hipGetLastError(), "GDN window F32 gate projection launch");
    }
    }
    if constexpr (Rows <= kGdnWindowRows) if (parts & 2U) {
    const auto saved = gdn_window_saved(config.model, layer.retained);
    // STRIX_GDN_WINDOW_PREPARE_GATES=1: the beta/gate scalars of the prepare spread over (head, row) pairs.
    static const bool parallel_gates = [] {
        const char* v = std::getenv("STRIX_GDN_WINDOW_PREPARE_GATES");
        return v && v[0] == '1';
    }();
    const auto prepare = parallel_gates ? gdn_window_prepare_kernel<Rows, true> : gdn_window_prepare_kernel<Rows, false>;
    hipLaunchKernelGGL(prepare, dim3(shape.key_heads * 2U + shape.value_heads + 1U), dim3(kRecurrentThreads), 0,
        stream, rows, layer.conv_state, layer.conv, small_weight_kind(layer.conv_type), layer.dt,
        small_weight_kind(layer.dt_type), layer.a, small_weight_kind(layer.a_type), saved, control, shape.key_heads,
        shape.value_heads, shape.head_dim, config.model.gdn_qk_l2_epsilon);
    check_hip(hipGetLastError(), "GDN window prepare launch");
    // STRIX_GDN_WINDOW_COLUMNS=4|2: fewer value columns per wave (more waves, shorter serial chains; same sums).
    static const std::uint32_t window_columns = [] {
        const char* v = std::getenv("STRIX_GDN_WINDOW_COLUMNS");
        const int n = v ? std::atoi(v) : 8;
        return static_cast<std::uint32_t>(n == 4 || n == 2 ? n : 8);
    }();
    if constexpr (Rows == 2U || Rows == 4U || Rows == 8U) {
        if (qpersist_gdn_tail_enabled() && (parts & 4U) && shape.head_dim == 128U && window_columns == 8U) {
            // Alternate recurrence organizations retain their own mapping.
            static const bool alternate = [] {
                const char* tiled = std::getenv("STRIX_GDN_WINDOW_TILED");
                const char* scatter = std::getenv("STRIX_QKERN_GDN_SCATTER");
                return (tiled && tiled[0] == '1') || (scatter && scatter[0] == '1');
            }();
            if (!alternate) {
                const bool dense = config.dense_a8_projections;
                if (dense) fused_slot = q8_rows::dense_a8_slot(0U);
                const auto launch = [&]<GdnRecurrentStorage Storage, bool Dense>() {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_window_recurrent_norm_kernel<Rows, Storage, Dense>),
                        dim3(shape.value_heads), dim3(512), 0, stream, rows, layer.matrix, saved, control,
                        shape.key_heads, layer.norm, small_weight_kind(layer.norm_type), config.model.rms_norm_epsilon,
                        fused_slot.codes, fused_slot.deltas, shape.value_width);
                };
                if (config.recurrent_storage == GdnRecurrentStorage::value_key) {
                    if (dense) launch.template operator()<GdnRecurrentStorage::value_key, true>();
                    else launch.template operator()<GdnRecurrentStorage::value_key, false>();
                } else {
                    if (dense) launch.template operator()<GdnRecurrentStorage::key_value, true>();
                    else launch.template operator()<GdnRecurrentStorage::key_value, false>();
                }
                fused_norm = true;
            }
        }
    }
    const auto recurrence = [&]<std::uint32_t Columns>() {
        const auto tiles = dim3(shape.value_heads * (shape.head_dim / Columns));
        static const bool scatter = [] { const char* v = std::getenv("STRIX_QKERN_GDN_SCATTER"); return v && v[0] == '1'; }();
        if constexpr (Columns == 4U || Columns == 8U) if (scatter) {
            if (config.recurrent_storage == GdnRecurrentStorage::value_key)
                hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_window_recurrent_kernel<Rows, GdnRecurrentStorage::value_key, Columns, true>), tiles,
                    dim3(32), 0, stream, rows, layer.matrix, saved, control, shape.key_heads, shape.head_dim);
            else
                hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_window_recurrent_kernel<Rows, GdnRecurrentStorage::key_value, Columns, true>), tiles,
                    dim3(32), 0, stream, rows, layer.matrix, saved, control, shape.key_heads, shape.head_dim);
            return;
        }
        if (config.recurrent_storage == GdnRecurrentStorage::value_key)
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_window_recurrent_kernel<Rows, GdnRecurrentStorage::value_key, Columns>), tiles,
                dim3(32), 0, stream, rows, layer.matrix, saved, control, shape.key_heads, shape.head_dim);
        else
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_window_recurrent_kernel<Rows, GdnRecurrentStorage::key_value, Columns>), tiles,
                dim3(32), 0, stream, rows, layer.matrix, saved, control, shape.key_heads, shape.head_dim);
    };
    // STRIX_GDN_WINDOW_TILED=1: key-value storage, coalesced 32-column slices through LDS (columns per wave as above).
    static const bool tiled = [] { const char* v = std::getenv("STRIX_GDN_WINDOW_TILED"); return v && v[0] == '1'; }();
    const auto tiled_recurrence = [&]<std::uint32_t Columns>() {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_window_recurrent_tiled_kernel<Rows, Columns>),
            dim3(shape.value_heads * (shape.head_dim / kGdnTiledColumns)), dim3(32U * (kGdnTiledColumns / Columns)), 0, stream,
            rows, layer.matrix, saved, control, shape.key_heads, shape.head_dim);
    };
    if (fused_norm) {
    } else if (tiled && config.recurrent_storage == GdnRecurrentStorage::key_value && shape.head_dim == 128U) {
        if (window_columns == 8U) tiled_recurrence.template operator()<8U>();
        else if (window_columns == 4U) tiled_recurrence.template operator()<4U>();
        else tiled_recurrence.template operator()<2U>();
    }
    else if (window_columns == 4U) recurrence.template operator()<4U>();
    else if (window_columns == 2U) recurrence.template operator()<2U>();
    else recurrence.template operator()<8U>();
    check_hip(hipGetLastError(), "GDN window recurrence launch");
    }
    if (!(parts & 4U)) return;
    // Dense A8: the gated norm (one wave per 32 features) also quantizes the out projection's
    // rows into a scratch slot, so the projection needs no quantizer launch.
    const bool dense = config.dense_a8_projections;
    if (dense && (shape.head_dim != kRecurrentThreads || shape.head_dim % 32U))
        throw std::invalid_argument("GDN window dense A8 norm quantizer requires one 128-feature head per block");
    const auto slot = fused_norm ? fused_slot : dense ? q8_rows::dense_a8_slot(0U) : q8_rows::DenseA8Slot{};
    if (!fused_norm && dense)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_window_norm_dense_a8_kernel<Rows>), dim3(shape.value_heads, Rows),
            dim3(kRecurrentThreads), 0, stream, rows, layer.norm, small_weight_kind(layer.norm_type), shape.head_dim,
            config.model.rms_norm_epsilon, slot.codes, slot.deltas, shape.value_width);
    else if (!fused_norm) hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_window_norm_kernel<Rows>), dim3(shape.value_heads, Rows),
        dim3(kRecurrentThreads), 0, stream, rows, layer.norm, small_weight_kind(layer.norm_type), shape.head_dim,
        config.model.rms_norm_epsilon);
    check_hip(hipGetLastError(), "GDN window gated RMSNorm launch");
    GdnOutputRowsSite<Rows> out{};
    out.segments = q8_rows::segments({{layer.out, shape.hidden}});
    out.segments.types[0] = q8_rows::segment_type(layer.out_type);
    out.width = shape.value_width; out.outputs = shape.hidden;
    for (unsigned r = 0; r < Rows; ++r) { out.input[r] = rows.core[r]; out.output[r] = static_cast<std::uint16_t*>(input[r].output); }
    if (dense) q8_rows::launch_dense_a8_quantized(out, slot, stream);
    else q8_rows::launch(out, false, stream);
    check_hip(hipGetLastError(), "GDN window Q8 output projection launch");
}
} // namespace

std::uint64_t gdn_window_retained_row_bytes(const ModelContract& model) {
    return gdn_window_saved(model, nullptr).row_bytes;
}
std::uint64_t gdn_window_retained_bytes(const ModelContract& model) {
    return 2ULL * gdn_window_max_rows * gdn_window_retained_row_bytes(model);
}
void gdn_window_enqueue(const GdnWindowLayer& layer, std::span<const GdnWindowRow> rows,
    const GdnWindowControl* control, const HipGdnLauncherConfig& config, void* stream) {
#if defined(STRIX_HAS_EXPERIMENTAL_DENSE_TRELLIS)
    HipDenseTrellisScope arithmetic(config.dense_trellis_projections);
#endif
    validate_gdn_window(layer, rows, true, control, config, stream);
    const auto native = static_cast<hipStream_t>(stream);
    switch (rows.size()) {
    case 1: enqueue_gdn_window<1U>(layer, rows, control, config, native); break;
    case 2: enqueue_gdn_window<2U>(layer, rows, control, config, native); break;
    case 3: enqueue_gdn_window<3U>(layer, rows, control, config, native); break;
    case 4: enqueue_gdn_window<4U>(layer, rows, control, config, native); break;
    case 5: enqueue_gdn_window<5U>(layer, rows, control, config, native); break;
    case 6: enqueue_gdn_window<6U>(layer, rows, control, config, native); break;
    case 7: enqueue_gdn_window<7U>(layer, rows, control, config, native); break;
    default: enqueue_gdn_window<8U>(layer, rows, control, config, native); break;
    }
}
void gdn_window_commit_enqueue(const GdnWindowLayer& layer, const GdnWindowControl* control,
    const HipGdnLauncherConfig& config, void* stream) {
    validate_gdn_window(layer, {}, false, control, config, stream);
    const auto shape = gdn_window_shape(config.model);
    const auto saved = gdn_window_saved(config.model, layer.retained);
    const auto grid = dim3(shape.value_heads * (shape.head_dim / kGdnWindowColumns) + shape.conv_width / 32U);
    const auto native = static_cast<hipStream_t>(stream);
    if (config.recurrent_storage == GdnRecurrentStorage::value_key)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_window_commit_kernel<GdnRecurrentStorage::value_key>), grid,
            dim3(32), 0, native, layer.matrix, layer.conv_state, saved, control,
            shape.key_heads, shape.value_heads, shape.head_dim);
    else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_window_commit_kernel<GdnRecurrentStorage::key_value>), grid,
            dim3(32), 0, native, layer.matrix, layer.conv_state, saved, control,
            shape.key_heads, shape.value_heads, shape.head_dim);
    check_hip(hipGetLastError(), "GDN window commit launch");
}

namespace {
struct GdnWindowBinding {
    GdnWindowLayer layer;
    std::array<GdnWindowRow, gdn_window_max_rows> rows{};
    std::size_t count = 0;
};
GdnWindowBinding bind_gdn_window(std::span<const ExecutionDispatchContext> rows, const HipGdnWindowConfig& config,
    bool window) {
    if (rows.empty() || rows.size() > gdn_window_max_rows || (!window && rows.size() != 1U))
        throw std::invalid_argument("GDN window requires 1..8 row contexts");
    GdnWindowBinding result;
    std::array<PreparedGdnLaunch, gdn_window_max_rows> prepared{};
    for (std::size_t r = 0; r < rows.size(); ++r) {
        prepared[r] = prepare_gdn_launch(rows[r], &config.gdn);
        const auto& first = prepared[0];
        if (rows[r].layer != rows[0].layer || rows[r].position != rows[0].position + r ||
            prepared[r].matrix_state.address != first.matrix_state.address ||
            prepared[r].conv_state.address != first.conv_state.address ||
            prepared[r].tensors.qkv.runtime != first.tensors.qkv.runtime ||
            prepared[r].tensors.out.runtime != first.tensors.out.runtime)
            throw std::invalid_argument("GDN window rows do not share one layer at consecutive positions");
        result.rows[r] = {reinterpret_cast<const std::uint16_t*>(prepared[r].input.address),
            reinterpret_cast<std::uint16_t*>(prepared[r].output.address), prepared[r].workspace.qkv};
    }
    const auto& row = prepared[0]; const auto& tensors = row.tensors;
    const bool gsq_dense = config.gdn.dense_a8_projections && q8_rows::dense_a8_type(tensors.qkv.manifest->ggml_type) &&
        q8_rows::dense_a8_type(tensors.z.manifest->ggml_type) && q8_rows::dense_a8_type(tensors.out.manifest->ggml_type);
    if ((!gsq_dense && (tensors.qkv.manifest->ggml_type != 8U || tensors.z.manifest->ggml_type != 8U ||
         tensors.out.manifest->ggml_type != 8U)) || tensors.beta.manifest->ggml_type != tensors.alpha.manifest->ggml_type)
        throw std::invalid_argument("GDN window requires Q8_0 qkv/z/out and one beta/alpha type");
    const auto& model = config.gdn.model;
    const auto layer = static_cast<std::uint64_t>(*rows[0].layer);
    const auto ordinal = layer - (layer + 1U) / model.full_attention_interval;
    const auto per_layer = gdn_window_retained_bytes(model);
    if (!config.retained || ordinal >= model.gdn_layers || config.retained_bytes < (ordinal + 1U) * per_layer)
        throw std::invalid_argument("GDN window retained owner does not cover this layer");
    const auto address = [](const TensorPair& tensor) { return reinterpret_cast<const void*>(tensor.runtime->address); };
    auto& out = result.layer;
    out.qkv = address(tensors.qkv); out.z = address(tensors.z); out.beta = address(tensors.beta);
    out.alpha = address(tensors.alpha); out.out = address(tensors.out); out.conv = address(tensors.conv);
    out.dt = address(tensors.dt); out.a = address(tensors.a); out.norm = address(tensors.norm);
    out.gate_type = tensors.beta.manifest->ggml_type; out.conv_type = tensors.conv.manifest->ggml_type;
    out.qkv_type = tensors.qkv.manifest->ggml_type; out.z_type = tensors.z.manifest->ggml_type;
    out.out_type = tensors.out.manifest->ggml_type;
    out.dt_type = tensors.dt.manifest->ggml_type; out.a_type = tensors.a.manifest->ggml_type;
    out.norm_type = tensors.norm.manifest->ggml_type;
    out.matrix = reinterpret_cast<float*>(row.matrix_state.address);
    out.conv_state = reinterpret_cast<std::uint16_t*>(row.conv_state.address);
    out.retained = static_cast<std::byte*>(config.retained) + ordinal * per_layer;
    result.count = rows.size();
    validate_gdn_window(out, {result.rows.data(), window ? result.count : 0U}, window, config.control, config.gdn,
        hip_execution_stream());
    return result;
}
} // namespace

void gdn_window_rows_preflight(std::span<const ExecutionDispatchContext> rows, const HipGdnWindowConfig& config) {
    static_cast<void>(bind_gdn_window(rows, config, true));
}
void gdn_window_rows_launch(std::span<const ExecutionDispatchContext> rows, const HipGdnWindowConfig& config) {
    const auto bound = bind_gdn_window(rows, config, true);
    gdn_window_enqueue(bound.layer, {bound.rows.data(), bound.count}, config.control, config.gdn, hip_execution_stream());
}
void gdn_window_commit_launch(const ExecutionDispatchContext& row, const HipGdnWindowConfig& config) {
    const auto bound = bind_gdn_window({&row, 1U}, config, false);
    gdn_window_commit_enqueue(bound.layer, config.control, config.gdn, hip_execution_stream());
}

namespace {
template <class Body> void gdn_window_rows_dispatch(std::size_t rows, Body&& body) {
    switch (rows) {
    case 1: body(std::integral_constant<unsigned, 1U>{}); break;
    case 2: body(std::integral_constant<unsigned, 2U>{}); break;
    case 3: body(std::integral_constant<unsigned, 3U>{}); break;
    case 4: body(std::integral_constant<unsigned, 4U>{}); break;
    case 5: body(std::integral_constant<unsigned, 5U>{}); break;
    case 6: body(std::integral_constant<unsigned, 6U>{}); break;
    case 7: body(std::integral_constant<unsigned, 7U>{}); break;
    case 8: body(std::integral_constant<unsigned, 8U>{}); break;
    default: throw std::invalid_argument("GDN window requires 1..8 rows");
    }
}
struct GdnCohortBinding {
    std::array<GdnWindowBinding, gdn_window_max_rows> segments;
    std::size_t count = 0;
};
GdnCohortBinding bind_gdn_window_cohort(std::span<const HipGdnWindowCohortSegment> segments) {
    if (segments.empty() || segments.size() > gdn_window_max_rows)
        throw std::invalid_argument("GDN cohort window requires 1..8 session segments");
    GdnCohortBinding out;
    for (std::size_t s = 0; s < segments.size(); ++s) {
        const auto& segment = segments[s];
        if (!segment.config || segment.rows.empty() || *segment.rows[0].layer != *segments[0].rows[0].layer ||
            segment.config->gdn.dense_a8_projections != segments[0].config->gdn.dense_a8_projections ||
            segment.config->gdn.dense_trellis_projections != segments[0].config->gdn.dense_trellis_projections ||
            segment.config->gdn.recurrent_storage != segments[0].config->gdn.recurrent_storage)
            throw std::invalid_argument("GDN cohort window segments differ in layer or arithmetic");
        // Every session segment passes the complete single-session window binding.
        out.segments[s] = bind_gdn_window(segment.rows, *segment.config, true);
    }
    out.count = segments.size();
    return out;
}
} // namespace

// Shared sink of the padding rows of cohort projections (allocated on first use, grown on demand; the
// stream is drained before a larger one replaces it).
static std::byte* gdn_cohort_pad_sink(std::uint64_t bytes) {
    static void* memory = nullptr;
    static std::uint64_t held = 0;
    static std::mutex mutex;
    const std::lock_guard lock(mutex);
    if (held < bytes) {
        if (memory) { check_hip(hipDeviceSynchronize(), "drain GDN cohort pad sink"); check_hip(hipFree(memory), "free GDN cohort pad sink"); }
        check_hip(hipMalloc(&memory, bytes), "allocate GDN cohort pad sink");
        held = bytes;
    }
    return static_cast<std::byte*>(memory);
}
// Parts 1 and 4 of a cohort: 1..8 rows, or 16 / 32 rows (cohorts of several 8-row chunks; the 16-row-tile
// projection forms, per-row arithmetic of the 1..8-row forms).
template <class Body> void gdn_cohort_rows_dispatch(std::size_t rows, Body&& body) {
    switch (rows) {
    case 16: body(std::integral_constant<unsigned, 16U>{}); break;
    case 32: body(std::integral_constant<unsigned, 32U>{}); break;
    default:
        if (rows > gdn_window_max_rows) throw std::invalid_argument("GDN cohort window requires 1..8, 16 or 32 rows");
        gdn_window_rows_dispatch(rows, std::forward<Body>(body));
    }
}
// Part 2 (prepare and recurrence) of every cohort segment in one launch each (segments of 2, 4 or 8 rows).
static void enqueue_gdn_window_segments(std::span<const GdnWindowCohortSegment> segments, const HipGdnLauncherConfig& config,
    hipStream_t stream) {
    if (segments.size() > gdn_window_max_rows) throw std::invalid_argument("GDN cohort segments exceed eight sessions");
    const auto shape = gdn_window_shape(config.model);
    GdnSegments all{};
    for (std::size_t s = 0; s < segments.size(); ++s) {
        const auto& segment = segments[s];
        auto& g = all.segment[s];
        g.count = static_cast<std::uint32_t>(segment.rows.size());
        for (std::uint32_t r = 0; r < g.count; ++r) {
            g.rows.qkv[r] = segment.rows[r].workspace; g.rows.z[r] = g.rows.qkv[r] + shape.conv_width;
            g.rows.beta[r] = g.rows.z[r] + shape.value_width; g.rows.gate[r] = g.rows.beta[r] + shape.value_heads;
            g.rows.core[r] = g.rows.gate[r] + shape.value_heads;
        }
        g.conv_state = segment.layer.conv_state; g.matrix = segment.layer.matrix;
        g.saved = gdn_window_saved(config.model, segment.layer.retained); g.control = segment.control;
    }
    const auto& layer = segments[0].layer;
    const auto count = static_cast<std::uint32_t>(segments.size());
    static const bool parallel_gates = [] { const char* v = std::getenv("STRIX_GDN_WINDOW_PREPARE_GATES"); return v && v[0] == '1'; }();
    const auto prepare = parallel_gates ? gdn_window_prepare_segments_kernel<true> : gdn_window_prepare_segments_kernel<false>;
    hipLaunchKernelGGL(prepare, dim3(shape.key_heads * 2U + shape.value_heads + 1U, count), dim3(kRecurrentThreads), 0, stream, all,
        layer.conv, small_weight_kind(layer.conv_type), layer.dt, small_weight_kind(layer.dt_type), layer.a,
        small_weight_kind(layer.a_type), shape.key_heads, shape.value_heads, shape.head_dim, config.model.gdn_qk_l2_epsilon);
    check_hip(hipGetLastError(), "GDN cohort segments prepare launch");
    static const std::uint32_t window_columns = [] {
        const char* v = std::getenv("STRIX_GDN_WINDOW_COLUMNS");
        const int n = v ? std::atoi(v) : 8;
        return static_cast<std::uint32_t>(n == 4 || n == 2 ? n : 8);
    }();
    const auto recurrence = [&]<std::uint32_t Columns>() {
        const auto tiles = dim3(shape.value_heads * (shape.head_dim / Columns), count);
        if (config.recurrent_storage == GdnRecurrentStorage::value_key)
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_window_recurrent_segments_kernel<GdnRecurrentStorage::value_key, Columns>), tiles,
                dim3(32), 0, stream, all, shape.key_heads, shape.head_dim);
        else
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gdn_window_recurrent_segments_kernel<GdnRecurrentStorage::key_value, Columns>), tiles,
                dim3(32), 0, stream, all, shape.key_heads, shape.head_dim);
    };
    if (window_columns == 4U) recurrence.template operator()<4U>();
    else if (window_columns == 2U) recurrence.template operator()<2U>();
    else recurrence.template operator()<8U>();
    check_hip(hipGetLastError(), "GDN cohort segments recurrence launch");
}
void gdn_window_cohort_enqueue(std::span<const GdnWindowCohortSegment> segments, const HipGdnLauncherConfig& config, void* stream) {
#if defined(STRIX_HAS_EXPERIMENTAL_DENSE_TRELLIS)
    HipDenseTrellisScope arithmetic(config.dense_trellis_projections);
#endif
    if (segments.empty() || segments.size() > gdn_window_max_rows)
        throw std::invalid_argument("GDN cohort window requires 1..8 session segments");
    std::array<GdnWindowRow, gdn_window_cohort_max_rows> rows{};
    std::size_t count = 0;
    const auto& a = segments[0].layer;
    for (std::size_t s = 0; s < segments.size(); ++s) {
        const auto& segment = segments[s];
        validate_gdn_window(segment.layer, segment.rows, true, segment.control, config, stream);
        const auto& b = segment.layer;
        if (count + segment.rows.size() > gdn_window_cohort_max_rows)
            throw std::invalid_argument("GDN cohort window segments exceed 32 rows");
        if (a.qkv != b.qkv || a.z != b.z || a.beta != b.beta || a.alpha != b.alpha || a.out != b.out || a.conv != b.conv ||
            a.dt != b.dt || a.a != b.a || a.norm != b.norm || a.gate_type != b.gate_type || a.conv_type != b.conv_type ||
            a.dt_type != b.dt_type || a.a_type != b.a_type || a.norm_type != b.norm_type)
            throw std::invalid_argument("GDN cohort window segments bind different weights");
        for (std::size_t o = 0; o < s; ++o)
            if (segments[o].layer.matrix == b.matrix || segments[o].layer.conv_state == b.conv_state ||
                segments[o].layer.retained == b.retained || segments[o].control == segment.control)
                throw std::invalid_argument("GDN cohort window segments share session state");
        for (const auto& row : segment.rows) rows[count++] = row;
    }
    // More than 8 rows: dense A8 projections only (the 16-row-tile WMMA forms; the FP32 Q8 wide forms do not
    // keep the per-row arithmetic of the 1..8-row forms). Totals other than 16 / 32 are padded for the
    // projections with rows that read row 0's input and write only a sink no row reads.
    if (count > gdn_window_max_rows && !config.dense_a8_projections)
        throw std::invalid_argument("GDN cohort window of more than 8 rows requires dense A8 projections");
    const auto native = static_cast<hipStream_t>(stream);
    const auto padded = count <= gdn_window_max_rows || count == 16U || count == 32U ? count : count < 16U ? 16U : 32U;
    if (padded > count) {
        const auto shape = gdn_window_shape(config.model);
        const auto workspace_bytes = ((static_cast<std::uint64_t>(shape.conv_width) + 2U * shape.value_width + 2U * shape.value_heads) *
            sizeof(float) + 255U) / 256U * 256U;
        const auto sink = gdn_cohort_pad_sink(workspace_bytes + static_cast<std::uint64_t>(config.model.hidden_size) * 2U);
        for (auto r = count; r < padded; ++r)
            rows[r] = {rows[0].input, reinterpret_cast<std::uint16_t*>(sink + workspace_bytes), reinterpret_cast<float*>(sink)};
    }
    const std::span<const GdnWindowRow> all(rows.data(), padded);
    gdn_cohort_rows_dispatch(padded, [&](auto width) {
        enqueue_gdn_window<decltype(width)::value>(a, all, segments[0].control, config, native, 1U);
    });
    // Part 2 of every segment in one prepare and one recurrence launch (STRIX_COHORT_GDN_SEGMENTS=0: one pair per
    // segment). The tiled recurrence (STRIX_GDN_WINDOW_TILED=1) keeps one pair per segment.
    static const bool segmented = [] { const char* v = std::getenv("STRIX_COHORT_GDN_SEGMENTS"); return !v || v[0] != '0'; }();
    static const bool tiled = [] { const char* v = std::getenv("STRIX_GDN_WINDOW_TILED"); return v && v[0] == '1'; }();
    bool fused = segmented && !tiled && segments.size() >= 2U && !known_window_cohort_feature_off("gdnseg");
    for (const auto& segment : segments)
        fused = fused && (segment.rows.size() == 2U || segment.rows.size() == 4U || segment.rows.size() == 8U);
    if (fused) enqueue_gdn_window_segments(segments, config, native);
    else for (const auto& segment : segments)
        gdn_window_rows_dispatch(segment.rows.size(), [&](auto width) {
            enqueue_gdn_window<decltype(width)::value>(segment.layer, segment.rows, segment.control, config, native, 2U);
        });
    gdn_cohort_rows_dispatch(padded, [&](auto width) {
        enqueue_gdn_window<decltype(width)::value>(a, all, segments[0].control, config, native, 4U);
    });
}

void gdn_window_cohort_rows_preflight(std::span<const HipGdnWindowCohortSegment> segments) {
    static_cast<void>(bind_gdn_window_cohort(segments));
}
void gdn_window_cohort_rows_launch(std::span<const HipGdnWindowCohortSegment> segments) {
    const auto bound = bind_gdn_window_cohort(segments);
    std::array<GdnWindowCohortSegment, gdn_window_max_rows> layers{};
    for (std::size_t s = 0; s < bound.count; ++s)
        layers[s] = {bound.segments[s].layer, {bound.segments[s].rows.data(), bound.segments[s].count}, segments[s].config->control};
    gdn_window_cohort_enqueue({layers.data(), bound.count}, segments[0].config->gdn, hip_execution_stream());
}

#ifdef STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC
void gdn_window_serial_row_probe(const GdnWindowLayer& layer, const GdnWindowRow& row,
    const HipGdnLauncherConfig& config, void* stream_ptr) {
    const GdnWindowControl unused{};
    validate_gdn_window(layer, {&row, 1U}, true, &unused, config, stream_ptr);
    const auto shape = gdn_window_shape(config.model);
    const auto stream = static_cast<hipStream_t>(stream_ptr);
    const auto workspace = gdn_window_rows<1U>({&row, 1U}, shape);
    const auto binding = [](const void* weights, std::uint64_t bytes) {
        RuntimeTensorBinding result; result.address = reinterpret_cast<std::uintptr_t>(weights); result.bytes = bytes;
        return result;
    };
    const auto q8 = [](std::uint32_t width, std::uint32_t outputs) { return width / 32ULL * outputs * sizeof(Q8Block); };
    const auto gate_bytes = layer.gate_type ? q8(shape.hidden, shape.value_heads) : 4ULL * shape.hidden * shape.value_heads;
    const auto mode = ProjectionMode::prefetch8_wave8;
    launch_projection_by_kind(WeightKind::q8_0, row.input, binding(layer.qkv, q8(shape.hidden, shape.conv_width)),
        shape.hidden, shape.conv_width, workspace.qkv[0], mode, stream, nullptr);
    launch_projection_by_kind(WeightKind::q8_0, row.input, binding(layer.z, q8(shape.hidden, shape.value_width)),
        shape.hidden, shape.value_width, workspace.z[0], mode, stream, nullptr);
    launch_projection_by_kind(weight_kind(layer.gate_type), row.input, binding(layer.beta, gate_bytes),
        shape.hidden, shape.value_heads, workspace.beta[0], mode, stream, nullptr);
    launch_projection_by_kind(weight_kind(layer.gate_type), row.input, binding(layer.alpha, gate_bytes),
        shape.hidden, shape.value_heads, workspace.gate[0], mode, stream, nullptr);
    hipLaunchKernelGGL(conv_update_kernel, dim3((shape.conv_width + kProjectionThreads - 1U) / kProjectionThreads),
        dim3(kProjectionThreads), 0, stream, workspace.qkv[0], layer.conv_state, layer.conv,
        small_weight_kind(layer.conv_type), shape.conv_width, kGdnWindowKernel, nullptr);
    hipLaunchKernelGGL(transform_beta_gate_kernel, dim3((shape.value_heads + kProjectionThreads - 1U) / kProjectionThreads),
        dim3(kProjectionThreads), 0, stream, workspace.beta[0], workspace.gate[0], layer.dt,
        small_weight_kind(layer.dt_type), layer.a, small_weight_kind(layer.a_type), shape.value_heads);
    hipLaunchKernelGGL(normalize_qk_kernel, dim3(shape.key_heads * 2U), dim3(kRecurrentThreads), 0, stream,
        workspace.qkv[0], shape.key_heads, shape.head_dim, config.model.gdn_qk_l2_epsilon, nullptr);
    if (config.recurrent_layout == HipGdnRecurrentLayout::baseline && config.recurrent_storage == GdnRecurrentStorage::key_value) {
        hipLaunchKernelGGL(recurrent_step_kernel, dim3(shape.value_heads * (shape.head_dim / kValueTile)),
            dim3(kRecurrentThreads), 0, stream, workspace.qkv[0], workspace.beta[0], workspace.gate[0], layer.matrix,
            workspace.core[0], shape.key_heads, shape.value_heads, shape.head_dim, nullptr);
    } else {
        PreparedGdnLaunch prepared{};
        prepared.workspace = {workspace.qkv[0], workspace.z[0], workspace.beta[0], workspace.gate[0], workspace.core[0], 0};
        prepared.matrix_state.address = reinterpret_cast<std::uintptr_t>(layer.matrix);
        prepared.key_heads = shape.key_heads; prepared.value_heads = shape.value_heads; prepared.head_dim = shape.head_dim;
        const GdnEmitTarget target{stream, nullptr, nullptr, 0, nullptr};
        if (config.recurrent_storage == GdnRecurrentStorage::value_key)
            emit_gdn_recurrent_layout<GdnRecurrentStorage::value_key>(prepared, config.recurrent_layout, target);
        else emit_gdn_recurrent_layout(prepared, config.recurrent_layout, target);
    }
    hipLaunchKernelGGL(gated_rms_norm_kernel, dim3(shape.value_heads), dim3(kRecurrentThreads), 0, stream,
        workspace.core[0], workspace.z[0], layer.norm, small_weight_kind(layer.norm_type), shape.value_heads,
        shape.head_dim, config.model.rms_norm_epsilon, nullptr);
    check_hip(hipGetLastError(), "GDN window serial reference core launch");
    launch_output_by_kind(WeightKind::q8_0, workspace.core[0], binding(layer.out, q8(shape.value_width, shape.hidden)),
        shape.value_width, shape.hidden, row.output, true, true, stream, nullptr);
}
#endif

#ifdef STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC
template <unsigned Rows>
void enqueue_retained_recurrent_probe(std::span<const GdnRecurrentWindowRowView> input,
    GdnKnownWindowProjectionView state, std::uint32_t key_heads,
    std::uint32_t value_heads, std::uint32_t head_dim, hipStream_t stream) {
    GdnRecurrentWindowPointers<Rows> views{};
    for (unsigned r = 0; r < Rows; ++r) {
        views.qkv[r] = reinterpret_cast<const float*>(input[r].qkv.address);
        views.beta[r] = reinterpret_cast<const float*>(input[r].beta.address);
        views.gate[r] = reinterpret_cast<const float*>(input[r].gate.address);
        views.core[r] = reinterpret_cast<float*>(input[r].core.address);
    }
    hipLaunchKernelGGL(HIP_KERNEL_NAME(recurrent_retained_window_kernel<Rows>),
        dim3(value_heads * (head_dim / kValueTile)), dim3(kRecurrentThreads), 0, stream,
        views, reinterpret_cast<float*>(state.address), key_heads, value_heads, head_dim);
    check_hip(hipGetLastError(), "GDN retained-window diagnostic recurrence launch");
}
void gdn_recurrent_window_probe(std::span<const GdnRecurrentWindowRowView> rows,
    GdnKnownWindowProjectionView state, std::uint32_t key_heads, std::uint32_t value_heads,
    std::uint32_t head_dim, GdnRetainedWindowMode mode, void* stream) {
    if ((rows.size() != 2U && rows.size() != 4U) || key_heads != 16U || value_heads != 48U || head_dim != 128U ||
        (mode != GdnRetainedWindowMode::serial_rows && mode != GdnRetainedWindowMode::retained_rows) ||
        !stream || static_cast<hipStream_t>(stream) == hipStreamLegacy || static_cast<hipStream_t>(stream) == hipStreamPerThread ||
        current_hip_launch_context())
        throw std::invalid_argument("GDN retained-window diagnostic requires canonical R2/R4 and a standalone eager stream");
    std::array<GdnKnownWindowRange, 12> reads{};
    std::array<GdnKnownWindowRange, 5> writes{};
    const auto check_view = [](GdnKnownWindowProjectionView view, std::uint64_t required) {
        validate_gdn_known_window_range({view.address, view.bytes}, 16U);
        if (view.bytes < required) throw std::invalid_argument("GDN retained-window diagnostic view is undersized");
    };
    for (unsigned r = 0; r < rows.size(); ++r) {
        check_view(rows[r].qkv, static_cast<std::uint64_t>(key_heads * 2U + value_heads) * head_dim * sizeof(float));
        check_view(rows[r].beta, static_cast<std::uint64_t>(value_heads) * sizeof(float));
        check_view(rows[r].gate, static_cast<std::uint64_t>(value_heads) * sizeof(float));
        check_view(rows[r].core, static_cast<std::uint64_t>(value_heads) * head_dim * sizeof(float));
        reads[3U*r] = {rows[r].qkv.address, rows[r].qkv.bytes};
        reads[3U*r+1U] = {rows[r].beta.address, rows[r].beta.bytes};
        reads[3U*r+2U] = {rows[r].gate.address, rows[r].gate.bytes};
        writes[r] = {rows[r].core.address, rows[r].core.bytes};
    }
    check_view(state, static_cast<std::uint64_t>(value_heads) * head_dim * head_dim * sizeof(float));
    writes[rows.size()] = {state.address, state.bytes};
    for (unsigned i = 0; i <= rows.size(); ++i) {
        for (unsigned r = 0; r < 3U*rows.size(); ++r) require_gdn_known_window_disjoint(writes[i], reads[r]);
        for (unsigned j = 0; j < i; ++j) require_gdn_known_window_disjoint(writes[i], writes[j]);
    }
    const auto native_stream = static_cast<hipStream_t>(stream);
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    check_hip(hipStreamIsCapturing(native_stream, &capture), "query GDN retained-window diagnostic capture");
    if (capture != hipStreamCaptureStatusNone) throw std::invalid_argument("GDN retained-window diagnostic cannot be captured");
    check_hip(hipPeekAtLastError(), "GDN retained-window diagnostic pre-enqueue stale error");
    if (mode == GdnRetainedWindowMode::serial_rows) {
        for (const auto& row : rows) {
            hipLaunchKernelGGL(recurrent_step_kernel, dim3(value_heads * (head_dim / kValueTile)), dim3(kRecurrentThreads), 0,
                native_stream, reinterpret_cast<const float*>(row.qkv.address), reinterpret_cast<const float*>(row.beta.address),
                reinterpret_cast<const float*>(row.gate.address), reinterpret_cast<float*>(state.address),
                reinterpret_cast<float*>(row.core.address), key_heads, value_heads, head_dim, nullptr);
            check_hip(hipGetLastError(), "GDN retained-window diagnostic incumbent recurrence launch");
        }
    } else if (rows.size() == 2U) enqueue_retained_recurrent_probe<2U>(rows, state, key_heads, value_heads, head_dim, native_stream);
    else enqueue_retained_recurrent_probe<4U>(rows, state, key_heads, value_heads, head_dim, native_stream);
}

void gdn_known_window_r2_input_stage_probe(std::span<const ExecutionDispatchContext, 2> rows,
    const HipGdnLauncherConfig& config, GdnKnownWindowFormat format,
    GdnKnownWindowProjectionMode mode) {
    if ((format != GdnKnownWindowFormat::q8_all5 && format != GdnKnownWindowFormat::q8_rows2_f32_gates) ||
        (mode != GdnKnownWindowProjectionMode::incumbent_row_loop && mode != GdnKnownWindowProjectionMode::rows2))
        throw std::invalid_argument("GDN known-window input-stage diagnostic format or mode is invalid");
    const auto internal_format = format == GdnKnownWindowFormat::q8_all5
        ? Format::all_q8 : Format::q8_large_f32_gates;
    const auto prepared = prepare_gdn_known_window_rows(rows, config, internal_format);
    if (mode == GdnKnownWindowProjectionMode::rows2) {
        enqueue_gdn_known_window_inputs(prepared, internal_format, config);
    } else {
        for (const auto& row : prepared) {
            const auto* input = reinterpret_cast<const std::uint16_t*>(row.input.address);
            const auto project = [&](const TensorPair& tensor, std::uint32_t outputs, float* output) {
                launch_projection_by_kind(weight_kind(tensor.manifest->ggml_type), input,
                    *tensor.runtime, row.hidden, outputs, output, row.projection_mode,
                    hip_execution_stream(), nullptr);
            };
            project(row.tensors.qkv, row.conv_width, row.workspace.qkv);
            project(row.tensors.z, row.value_width, row.workspace.z);
            project(row.tensors.beta, row.value_heads, row.workspace.beta);
            project(row.tensors.alpha, row.value_heads, row.workspace.gate);
        }
    }
}

void gdn_known_window_r2_projection_probe(GdnKnownWindowProjectionDirection direction,
    GdnKnownWindowProjectionMode mode, std::span<const GdnKnownWindowProjectionView, 2> input,
    GdnKnownWindowProjectionView weights, std::span<const GdnKnownWindowProjectionView, 2> output,
    std::uint32_t width, std::uint32_t outputs, void* stream) {
    if ((direction != GdnKnownWindowProjectionDirection::bf16_to_fp32 &&
         direction != GdnKnownWindowProjectionDirection::fp32_to_bf16) ||
        (mode != GdnKnownWindowProjectionMode::incumbent_row_loop && mode != GdnKnownWindowProjectionMode::rows2) ||
        !stream || static_cast<hipStream_t>(stream) == hipStreamPerThread || static_cast<hipStream_t>(stream) == hipStreamLegacy)
        throw std::invalid_argument("GDN known-window projection diagnostic mode or stream is invalid");
    const bool bf16 = direction == GdnKnownWindowProjectionDirection::bf16_to_fp32;
    for (unsigned row = 0; row < 2U; ++row) {
        const q8_projection_organization::View x{input[row].address, input[row].bytes};
        const q8_projection_organization::View w{weights.address, weights.bytes};
        const q8_projection_organization::View y{output[row].address, output[row].bytes};
        if (bf16) q8_projection_organization::validate_views(width, outputs, x, w, y);
        else q8_projection_organization::validate_fp32_bf16_views(width, outputs, x, w, y);
    }
    for (unsigned row = 0; row < 2U; ++row) {
        const GdnKnownWindowRange write{output[row].address, output[row].bytes};
        for (const auto read : input) require_gdn_known_window_disjoint(write, {read.address, read.bytes});
        require_gdn_known_window_disjoint(write, {weights.address, weights.bytes});
    }
    require_gdn_known_window_disjoint({output[0].address, output[0].bytes}, {output[1].address, output[1].bytes});
    const auto native_stream = static_cast<hipStream_t>(stream);
    const auto* packed = reinterpret_cast<const void*>(weights.address);
    check_hip(hipPeekAtLastError(), "GDN known-window diagnostic pre-enqueue stale error");
    if (mode == GdnKnownWindowProjectionMode::incumbent_row_loop) {
        for (unsigned row = 0; row < 2U; ++row) {
            if (bf16) {
                hipLaunchKernelGGL(project_prefetch8_wave8_q8_bf16_to_fp32_kernel,
                    dim3((outputs + 7U) / 8U), dim3(kProjectionThreads), 0, native_stream,
                    reinterpret_cast<const std::uint16_t*>(input[row].address), packed, width, outputs,
                    reinterpret_cast<float*>(output[row].address));
            } else {
                hipLaunchKernelGGL(project_prefetch8_wave8_q8_fp32_to_bf16_kernel,
                    dim3((outputs + 7U) / 8U), dim3(kProjectionThreads), 0, native_stream,
                    reinterpret_cast<const float*>(input[row].address), packed, width, outputs,
                    reinterpret_cast<std::uint16_t*>(output[row].address));
            }
            check_hip(hipGetLastError(), "GDN known-window diagnostic incumbent projection");
        }
    } else if (bf16) {
        hipLaunchKernelGGL(project_wave8_q8_bf16_to_fp32_rows2_kernel,
            dim3((outputs + 7U) / 8U), dim3(kProjectionThreads), 0, native_stream,
            reinterpret_cast<const std::uint16_t*>(input[0].address), reinterpret_cast<const std::uint16_t*>(input[1].address),
            packed, width, outputs, reinterpret_cast<float*>(output[0].address), reinterpret_cast<float*>(output[1].address));
        check_hip(hipGetLastError(), "GDN known-window diagnostic R2 input projection");
    } else {
        hipLaunchKernelGGL(project_wave8_q8_fp32_to_bf16_rows2_kernel,
            dim3((outputs + 7U) / 8U), dim3(kProjectionThreads), 0, native_stream,
            reinterpret_cast<const float*>(input[0].address), reinterpret_cast<const float*>(input[1].address),
            packed, width, outputs, reinterpret_cast<std::uint16_t*>(output[0].address), reinterpret_cast<std::uint16_t*>(output[1].address));
        check_hip(hipGetLastError(), "GDN known-window diagnostic R2 output projection");
    }
}
template <unsigned Rows>
void enqueue_gdn_wide_projection_probe(bool bf16, bool a8, std::span<const GdnKnownWindowProjectionView> input,
    GdnKnownWindowProjectionView weights, std::span<const GdnKnownWindowProjectionView> output,
    std::uint32_t width, std::uint32_t outputs, hipStream_t stream) {
    const auto segments = q8_rows::segments({{reinterpret_cast<const void*>(weights.address), outputs}});
    if (bf16) {
        GdnInputRowsSite<Rows> site{};
        site.segments = segments; site.width = width; site.outputs = outputs;
        for (unsigned r = 0; r < Rows; ++r) {
            site.input[r] = reinterpret_cast<const void*>(input[r].address);
            site.output[0][r] = reinterpret_cast<float*>(output[r].address);
        }
        q8_rows::launch(site, a8, stream);
    } else {
        GdnOutputRowsSite<Rows> site{};
        site.segments = segments; site.width = width; site.outputs = outputs;
        for (unsigned r = 0; r < Rows; ++r) {
            site.input[r] = reinterpret_cast<const void*>(input[r].address);
            site.output[r] = reinterpret_cast<std::uint16_t*>(output[r].address);
        }
        q8_rows::launch(site, a8, stream);
    }
    check_hip(hipGetLastError(), "GDN wide-window diagnostic projection launch");
}
void gdn_known_window_rows_projection_probe(GdnKnownWindowProjectionDirection direction,
    std::span<const GdnKnownWindowProjectionView> input, GdnKnownWindowProjectionView weights,
    std::span<const GdnKnownWindowProjectionView> output, std::uint32_t width, std::uint32_t outputs, void* stream,
    GdnKnownWindowRowsProbeMode mode) {
    const auto rows = static_cast<std::uint32_t>(input.size());
    // The fixture rewrites weights in place between calls (STRIX_Q8_ROWS_LANE_PROBE_CACHE=1 keeps the
    // lane kernel's repacked weights, for timing one unchanged matrix).
    if (!q8_rows::env_unsigned("STRIX_Q8_ROWS_LANE_PROBE_CACHE", 0U)) q8_rows::invalidate_lane_weights();
    if (rows < 1U || (rows > 8U && rows != 16U && rows != 32U && rows != 64U) || output.size() != rows ||
        (direction != GdnKnownWindowProjectionDirection::bf16_to_fp32 &&
         direction != GdnKnownWindowProjectionDirection::fp32_to_bf16) ||
        (mode != GdnKnownWindowRowsProbeMode::rows && mode != GdnKnownWindowRowsProbeMode::rows_a8 &&
         mode != GdnKnownWindowRowsProbeMode::row_loop) ||
        !stream || static_cast<hipStream_t>(stream) == hipStreamLegacy || static_cast<hipStream_t>(stream) == hipStreamPerThread)
        throw std::invalid_argument("GDN wide-window diagnostic requires 1..8, 16, 32 or 64 rows, a mode and an owned stream");
    const bool bf16 = direction == GdnKnownWindowProjectionDirection::bf16_to_fp32;
    for (unsigned r = 0; r < rows; ++r) {
        const q8_projection_organization::View x{input[r].address, input[r].bytes}, w{weights.address, weights.bytes},
            y{output[r].address, output[r].bytes};
        if (bf16) q8_projection_organization::validate_views(width, outputs, x, w, y);
        else q8_projection_organization::validate_fp32_bf16_views(width, outputs, x, w, y);
        for (unsigned prior = 0; prior < r; ++prior) {
            require_gdn_known_window_disjoint({input[r].address, input[r].bytes}, {input[prior].address, input[prior].bytes});
            require_gdn_known_window_disjoint({output[r].address, output[r].bytes}, {output[prior].address, output[prior].bytes});
        }
        const GdnKnownWindowRange write{output[r].address, output[r].bytes};
        for (const auto read : input) require_gdn_known_window_disjoint(write, {read.address, read.bytes});
        require_gdn_known_window_disjoint(write, {weights.address, weights.bytes});
    }
    if (mode == GdnKnownWindowRowsProbeMode::rows_a8 && !q8_rows::dense_a8_width(width))
        throw std::invalid_argument("GDN wide-window dense A8 width is not admitted");
    const auto native_stream = static_cast<hipStream_t>(stream);
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    check_hip(hipStreamIsCapturing(native_stream, &capture), "query GDN wide diagnostic capture");
    if (capture != hipStreamCaptureStatusNone) throw std::invalid_argument("GDN wide-window diagnostic cannot be captured");
    check_hip(hipPeekAtLastError(), "GDN wide-window diagnostic pre-enqueue stale error");
    const auto* packed = reinterpret_cast<const void*>(weights.address);
    if (mode == GdnKnownWindowRowsProbeMode::row_loop) {
        for (unsigned r = 0; r < rows; ++r) {
            if (bf16) hipLaunchKernelGGL(project_prefetch8_wave8_q8_bf16_to_fp32_kernel, dim3((outputs + 7U) / 8U),
                dim3(kProjectionThreads), 0, native_stream, reinterpret_cast<const std::uint16_t*>(input[r].address),
                packed, width, outputs, reinterpret_cast<float*>(output[r].address));
            else hipLaunchKernelGGL(project_prefetch8_wave8_q8_fp32_to_bf16_kernel, dim3((outputs + 7U) / 8U),
                dim3(kProjectionThreads), 0, native_stream, reinterpret_cast<const float*>(input[r].address),
                packed, width, outputs, reinterpret_cast<std::uint16_t*>(output[r].address));
            check_hip(hipGetLastError(), "GDN wide-window diagnostic row-loop projection");
        }
        return;
    }
    const bool a8 = mode == GdnKnownWindowRowsProbeMode::rows_a8;
    switch (rows) {
    case 1: enqueue_gdn_wide_projection_probe<1U>(bf16, a8, input, weights, output, width, outputs, native_stream); break;
    case 2: enqueue_gdn_wide_projection_probe<2U>(bf16, a8, input, weights, output, width, outputs, native_stream); break;
    case 3: enqueue_gdn_wide_projection_probe<3U>(bf16, a8, input, weights, output, width, outputs, native_stream); break;
    case 4: enqueue_gdn_wide_projection_probe<4U>(bf16, a8, input, weights, output, width, outputs, native_stream); break;
    case 5: enqueue_gdn_wide_projection_probe<5U>(bf16, a8, input, weights, output, width, outputs, native_stream); break;
    case 6: enqueue_gdn_wide_projection_probe<6U>(bf16, a8, input, weights, output, width, outputs, native_stream); break;
    case 7: enqueue_gdn_wide_projection_probe<7U>(bf16, a8, input, weights, output, width, outputs, native_stream); break;
    case 16: enqueue_gdn_wide_projection_probe<16U>(bf16, a8, input, weights, output, width, outputs, native_stream); break;
    case 32: enqueue_gdn_wide_projection_probe<32U>(bf16, a8, input, weights, output, width, outputs, native_stream); break;
    case 64: enqueue_gdn_wide_projection_probe<64U>(bf16, a8, input, weights, output, width, outputs, native_stream); break;
    default: enqueue_gdn_wide_projection_probe<8U>(bf16, a8, input, weights, output, width, outputs, native_stream); break;
    }
}
template <unsigned Rows>
void enqueue_gdn_gsq_rows_probe(bool bf16, std::span<const std::uint32_t> types,
    std::span<const GdnKnownWindowProjectionView> weights, std::span<const std::uint32_t> counts,
    std::span<const GdnKnownWindowProjectionView> input, std::span<const GdnKnownWindowProjectionView> output,
    std::uint32_t width, hipStream_t stream) {
    q8_rows::Segments segments{};
    for (unsigned m = 0; m < types.size(); ++m) {
        segments.weights[m] = reinterpret_cast<const unsigned char*>(weights[m].address);
        segments.types[m] = q8_rows::segment_type(types[m]);
        segments.first[m + 1U] = segments.first[m] + counts[m];
    }
    segments.count = static_cast<unsigned>(types.size());
    if (bf16) {
        GdnInputRowsSite<Rows> site{};
        site.segments = segments; site.width = width; site.outputs = segments.first[segments.count];
        for (unsigned r = 0; r < Rows; ++r) {
            site.input[r] = reinterpret_cast<const void*>(input[r].address);
            for (unsigned m = 0; m < segments.count; ++m) site.output[m][r] = reinterpret_cast<float*>(output[m * Rows + r].address);
        }
        q8_rows::launch(site, true, stream);
    } else {
        GdnOutputRowsSite<Rows> site{};
        site.segments = segments; site.width = width; site.outputs = segments.first[1];
        for (unsigned r = 0; r < Rows; ++r) {
            site.input[r] = reinterpret_cast<const void*>(input[r].address);
            site.output[r] = reinterpret_cast<std::uint16_t*>(output[r].address);
        }
        q8_rows::launch(site, true, stream);
    }
    check_hip(hipGetLastError(), "GDN GSQ typed rows diagnostic launch");
}
void gdn_gsq_rows_projection_probe(GdnKnownWindowProjectionDirection direction, std::span<const std::uint32_t> types,
    std::span<const GdnKnownWindowProjectionView> weights, std::span<const std::uint32_t> counts,
    std::span<const GdnKnownWindowProjectionView> input, std::span<const GdnKnownWindowProjectionView> output,
    std::uint32_t width, void* stream) {
    const auto rows = static_cast<std::uint32_t>(input.size());
    const bool bf16 = direction == GdnKnownWindowProjectionDirection::bf16_to_fp32;
    if (rows < 1U || rows > 8U || types.empty() || types.size() > (bf16 ? 4U : 1U) || weights.size() != types.size() ||
        counts.size() != types.size() || output.size() != types.size() * rows || !q8_rows::dense_a8_width(width) ||
        !stream || static_cast<hipStream_t>(stream) == hipStreamLegacy || static_cast<hipStream_t>(stream) == hipStreamPerThread)
        throw std::invalid_argument("GDN GSQ typed rows diagnostic requires 1..8 rows, 1..4 input (1 output) segments and an owned stream");
    for (unsigned m = 0; m < types.size(); ++m)
        if (!gsq_a8::dense_type(types[m]) || weights[m].bytes < gsq_a8::dense_row_bytes(types[m], width) * counts[m])
            throw std::invalid_argument("GDN GSQ typed rows diagnostic segment type or weight extent");
    for (unsigned m = 0; m < types.size(); ++m)
        if (width % gsq_a8::dense_block_elements(types[m]))
            throw std::invalid_argument("GDN GSQ typed rows diagnostic: width is not whole blocks of a segment type");
    const auto native_stream = static_cast<hipStream_t>(stream);
    check_hip(hipPeekAtLastError(), "GDN GSQ typed rows diagnostic pre-enqueue stale error");
    switch (rows) {
    case 1: enqueue_gdn_gsq_rows_probe<1U>(bf16, types, weights, counts, input, output, width, native_stream); break;
    case 2: enqueue_gdn_gsq_rows_probe<2U>(bf16, types, weights, counts, input, output, width, native_stream); break;
    case 3: enqueue_gdn_gsq_rows_probe<3U>(bf16, types, weights, counts, input, output, width, native_stream); break;
    case 4: enqueue_gdn_gsq_rows_probe<4U>(bf16, types, weights, counts, input, output, width, native_stream); break;
    case 5: enqueue_gdn_gsq_rows_probe<5U>(bf16, types, weights, counts, input, output, width, native_stream); break;
    case 6: enqueue_gdn_gsq_rows_probe<6U>(bf16, types, weights, counts, input, output, width, native_stream); break;
    case 7: enqueue_gdn_gsq_rows_probe<7U>(bf16, types, weights, counts, input, output, width, native_stream); break;
    default: enqueue_gdn_gsq_rows_probe<8U>(bf16, types, weights, counts, input, output, width, native_stream); break;
    }
}
#endif

void register_hip_gdn_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipGdnLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification) {
    validate_config(config);
    if (request.gdn_recurrent_layout != config.recurrent_layout)
        throw std::invalid_argument("GDN recurrent LDS request/config mismatch");
    if (request.gdn_recurrent_storage != config.recurrent_storage)
        throw std::invalid_argument("GDN recurrent storage request/config mismatch");
    if (config.recurrent_storage != GdnRecurrentStorage::key_value &&
        (request.device_graph_inputs || request.require_graph_safe))
        throw std::invalid_argument("GDN recurrent storage registers eager variants only");
    if (config.recurrent_layout != HipGdnRecurrentLayout::baseline &&
        (request.device_graph_inputs || request.require_graph_safe))
        throw std::invalid_argument("GDN recurrent LDS layouts register eager variants only");
    if (request.wave8_q8_gdn_input != config.wave8_q8_input_projections)
        throw std::invalid_argument("GDN wave8 request/config mismatch");
    if (request.wave8_q8_gdn_output != config.wave8_q8_output_projection)
        throw std::invalid_argument("GDN wave8 output request/config mismatch");
    if ((config.wave8_q8_input_projections || config.wave8_q8_output_projection) && (request.device_graph_inputs || request.require_graph_safe))
        throw std::invalid_argument("wave8 GDN input registers eager variants only");
    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::invalid_argument(
            error);
    }
    if (config.model.hidden_size !=
            model.hidden_size ||
        config.model.gdn_key_heads !=
            model.gdn_key_heads ||
        config.model.gdn_key_head_dim !=
            model.gdn_key_head_dim ||
        config.model.gdn_value_heads !=
            model.gdn_value_heads ||
        config.model.gdn_value_head_dim !=
            model.gdn_value_head_dim ||
        config.model.gdn_conv_kernel !=
            model.gdn_conv_kernel ||
        config.model.rms_norm_epsilon !=
            model.rms_norm_epsilon ||
        config.model.gdn_qk_l2_epsilon !=
            model.gdn_qk_l2_epsilon ||
        config.model.gdn_output_gate_sigmoid !=
            model.gdn_output_gate_sigmoid ||
        config.model.gdn_qk_l2norm !=
            model.gdn_qk_l2norm ||
        config.model.gdn_conv_silu !=
            model.gdn_conv_silu) {
        throw std::invalid_argument(
            "HIP GDN config does not match model contract");
    }
    if (request.logical_rows != 1 ||
        request.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP GDN v1 registers only C1 one-row variants");
    }
    if (request.device_graph_inputs && (!request.require_graph_safe || !config.enable_device_graph_inputs)) {
        throw std::invalid_argument("GDN device graph variant requires explicit graph input and graph-safe mode");
    }
    if (qualification ==
            kernels::QualificationStatus::
                target_qualified ||
        qualification ==
            kernels::QualificationStatus::
                production ||
        qualification ==
            kernels::QualificationStatus::
                rejected) {
        throw std::invalid_argument(
            "unmeasured HIP GDN may only be experimental or oracle-matched");
    }

    kernels::KernelVariant variant;
    variant.id =
        kernel_id(
            request,
            config);
    variant.family =
        kernels::OperatorFamily::
            recurrent;
    variant.phase =
        kernels::ExecutionPhase::
            decode;
    variant.shape =
        execution_shape_for(
            model,
            SerialOp::gated_delta_net,
            request);
    variant.wave =
        kernels::WaveSize::wave32;
    variant.resources.workgroup_threads =
        kProjectionThreads;
    variant.resources.lds_bytes =
        // Conservative 256-byte-aligned bound for both FP32 arrays and the
        // graph rejection broadcast, including possible compiler padding.
        ((kRecurrentThreads *
          (config.recurrent_layout == HipGdnRecurrentLayout::stride33 ? 33U : kValueTile) * sizeof(float) +
          kValueTile * sizeof(float) + sizeof(std::uint32_t) + 255U) / 256U) * 256U;
    variant.resources.scratch_bytes =
        build_serial_execution_buffer_plan(
            model,
            ExecutionBufferRequest{
                1,
                2,
                4,
                4,
                4,
                4,
            }).gdn_workspace_bytes;
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
        "row-major";
    variant.output_layout =
        "row-major";
    variant.compiler_identity = config.prefetch_q8_input_projections ? "hip-gfx1151-gdn-v4-prefetch8-virtual256-q8-input"
        : config.virtual_q8_input_projections ? "hip-gfx1151-gdn-v3-virtual256-q8-input" : "hip-gfx1151-gdn-v2";
    if(config.virtual_q8_output_projection) variant.compiler_identity += "-virtual256-q8-output";
    if(config.wave8_q8_input_projections) variant.compiler_identity += ":wave8-input-v1";
    if(config.wave8_q8_output_projection) variant.compiler_identity += ":wave8-q8-output-v1";
    if(config.dense_a8_projections) {
        variant.numerics.accumulator_precision += ";" + std::string(q8_projection_organization::dense_a8_numerical_identity);
        variant.compiler_identity += ":dense-a8-q80-v1";
    }
    if(config.recurrent_layout != HipGdnRecurrentLayout::baseline)
        variant.compiler_identity += ":gdn-recurrent" + std::string(gdn_recurrent_layout_marker(config.recurrent_layout));
    if(config.recurrent_storage != GdnRecurrentStorage::key_value)
        variant.compiler_identity += ":gdn-storage" + std::string(gdn_recurrent_storage_marker(config.recurrent_storage));
    variant.measured_seconds = 0.0;

    registry.add(
        std::move(variant));
}

namespace {
// The owner is immutable; these are full descriptor values, never hash keys.
struct GdnIslandView {
    std::uintptr_t address = 0;
    std::uint64_t bytes = 0, generation = 0;
    bool operator==(const GdnIslandView&) const = default;
};
struct GdnIslandTensor {
    const RuntimeTensorBinding* binding = nullptr;
    std::size_t slot = 0;
    std::uintptr_t address = 0;
    std::uint64_t bytes = 0, generation = 0;
    RuntimeTensorStorage storage = RuntimeTensorStorage::resident;
    std::uint32_t ggml_type = 0;
    std::uint32_t tensor_index = 0, owner_node = 0;
    bool operator==(const GdnIslandTensor&) const = default;
};
struct GdnIslandDescriptor {
    std::uint32_t step = 0, layer = 0;
    std::array<GdnIslandView, 3> buffers;
    std::array<std::uintptr_t, 5> workspace;
    std::array<GdnIslandTensor, 9> tensors;
    std::array<GdnIslandView, 2> state;
    std::array<SessionStateStorage, 2> state_storage;
    std::array<std::uint32_t, 7> dimensions;
    ProjectionMode projection_mode = ProjectionMode::legacy256;
    bool graph_capture = false;
    bool operator==(const GdnIslandDescriptor&) const = default;
};
std::array<TensorPair, 9> island_tensors(const GdnTensors& tensors) {
    return {tensors.qkv, tensors.z, tensors.beta, tensors.alpha, tensors.conv,
        tensors.dt, tensors.a, tensors.norm, tensors.out};
}
GdnIslandDescriptor island_descriptor(const ExecutionDispatchContext& context, const PreparedGdnLaunch& p,
    const ModelRuntimeBinding& actual, const GdnIslandDescriptor* expected = nullptr) {
    GdnIslandDescriptor d;
    d.step = context.step_id; d.layer = context.layer.value();
    d.buffers = {{{p.input.address, p.input.bytes, p.input.allocation_generation},
        {p.output.address, p.output.bytes, p.output.allocation_generation},
        {p.temporary.address, p.temporary.bytes, p.temporary.allocation_generation}}};
    d.workspace = {reinterpret_cast<std::uintptr_t>(p.workspace.qkv), reinterpret_cast<std::uintptr_t>(p.workspace.z),
        reinterpret_cast<std::uintptr_t>(p.workspace.beta), reinterpret_cast<std::uintptr_t>(p.workspace.gate),
        reinterpret_cast<std::uintptr_t>(p.workspace.core)};
    const auto tensors = island_tensors(p.tensors);
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto& t = tensors[i];
        std::size_t slot = 0;
        if (expected) {
            slot = expected->tensors[i].slot;
            // Check the current slot address without dereferencing the recorded
            // pointer: vector replacement invalidates admission, not lifetime.
            if (slot >= actual.tensors.size() || &actual.tensors[slot] != expected->tensors[i].binding)
                throw std::invalid_argument("GDN island canonical weight owner slot changed");
        } else {
            // Setup only: the unprepared context must resolve the real owner.
            const auto found = std::find_if(actual.tensors.begin(), actual.tensors.end(),
                [&](const RuntimeTensorBinding& tensor) { return &tensor == t.runtime; });
            if (found == actual.tensors.end())
                throw std::invalid_argument("GDN island setup tensor lacks an actual owner slot");
            slot = static_cast<std::size_t>(found - actual.tensors.begin());
        }
        const auto& canonical = actual.tensors[slot];
        const auto& resolved = *t.runtime;
        // PreparedTensorViews owns a deep metadata copy. Normalize that pointer
        // only after every resolved runtime field matches the retained owner.
        if (resolved.tensor_index != canonical.tensor_index || resolved.storage != canonical.storage ||
            resolved.address != canonical.address || resolved.bytes != canonical.bytes ||
            resolved.allocation_generation != canonical.allocation_generation || resolved.owner_node != canonical.owner_node)
            throw std::invalid_argument("GDN island resolved weight differs from its actual canonical owner slot");
        d.tensors[i] = {&canonical, slot, resolved.address, resolved.bytes,
            resolved.allocation_generation, resolved.storage, t.manifest->ggml_type,
            resolved.tensor_index, resolved.owner_node};
    }
    d.state = {{{p.matrix_state.address, p.matrix_state.bytes, p.matrix_state.allocation_generation},
        {p.conv_state.address, p.conv_state.bytes, p.conv_state.allocation_generation}}};
    d.state_storage = {p.matrix_state.storage, p.conv_state.storage};
    d.dimensions = {p.hidden, p.key_heads, p.value_heads, p.head_dim, p.value_width, p.conv_width, p.kernel};
    d.projection_mode = p.projection_mode; d.graph_capture = p.graph_capture;
    return d;
}

bool same_island_config(const HipGdnLauncherConfig& a, const HipGdnLauncherConfig& b) noexcept {
    return a.model.vocab_size == b.model.vocab_size &&
        a.model.hidden_size == b.model.hidden_size &&
        a.model.max_position_embeddings == b.model.max_position_embeddings &&
        a.model.hidden_layers == b.model.hidden_layers &&
        a.model.full_attention_interval == b.model.full_attention_interval &&
        a.model.ple_layer_1based == b.model.ple_layer_1based &&
        a.model.ple_embed_dim == b.model.ple_embed_dim &&
        a.model.ple_conv_kernel_size == b.model.ple_conv_kernel_size &&
        a.model.ngram_size == b.model.ngram_size &&
        a.model.heads_per_ngram == b.model.heads_per_ngram &&
        a.model.ngram_vocab_size_base == b.model.ngram_vocab_size_base &&
        a.model.ngram_vocab_divisor == b.model.ngram_vocab_divisor &&
        a.model.ngram_hash_seed == b.model.ngram_hash_seed &&
        a.model.ple_eos_token_id == b.model.ple_eos_token_id &&
        a.model.hc_count == b.model.hc_count &&
        a.model.hc_lowrank == b.model.hc_lowrank &&
        a.model.full_attention_layers == b.model.full_attention_layers &&
        a.model.gdn_layers == b.model.gdn_layers &&
        a.model.attention_heads == b.model.attention_heads &&
        a.model.kv_heads == b.model.kv_heads &&
        a.model.attention_kv_storage == b.model.attention_kv_storage &&
        a.model.head_dim == b.model.head_dim &&
        a.model.mtp_full_attention_layers == b.model.mtp_full_attention_layers &&
        a.model.mtp_use_dedicated_embeddings == b.model.mtp_use_dedicated_embeddings &&
        a.model.mtp_hidden_state_source_layer_explicit == b.model.mtp_hidden_state_source_layer_explicit &&
        a.model.gdn_key_heads == b.model.gdn_key_heads &&
        a.model.gdn_key_head_dim == b.model.gdn_key_head_dim &&
        a.model.gdn_value_heads == b.model.gdn_value_heads &&
        a.model.gdn_value_head_dim == b.model.gdn_value_head_dim &&
        a.model.gdn_conv_kernel == b.model.gdn_conv_kernel &&
        a.model.indexer_n_heads == b.model.indexer_n_heads &&
        a.model.indexer_kv_heads == b.model.indexer_kv_heads &&
        a.model.indexer_head_dim == b.model.indexer_head_dim &&
        a.model.indexer_budget == b.model.indexer_budget &&
        same_rotary_model(a.model, b.model) &&
        a.model.indexer_compress_ratio == b.model.indexer_compress_ratio &&
        std::bit_cast<std::uint32_t>(a.model.partial_rotary_factor) == std::bit_cast<std::uint32_t>(b.model.partial_rotary_factor) &&
        std::bit_cast<std::uint32_t>(a.model.rope_theta) == std::bit_cast<std::uint32_t>(b.model.rope_theta) &&
        a.model.moe_intermediate_size == b.model.moe_intermediate_size &&
        a.model.shared_expert_intermediate_size == b.model.shared_expert_intermediate_size &&
        a.model.num_experts == b.model.num_experts &&
        a.model.num_experts_per_token == b.model.num_experts_per_token &&
        std::bit_cast<std::uint32_t>(a.model.rms_norm_epsilon) == std::bit_cast<std::uint32_t>(b.model.rms_norm_epsilon) &&
        a.model.norm_topk_probability == b.model.norm_topk_probability &&
        a.model.gdn_output_gate_sigmoid == b.model.gdn_output_gate_sigmoid &&
        a.model.gdn_qk_l2norm == b.model.gdn_qk_l2norm &&
        std::bit_cast<std::uint32_t>(a.model.gdn_qk_l2_epsilon) == std::bit_cast<std::uint32_t>(b.model.gdn_qk_l2_epsilon) &&
        a.model.gdn_conv_silu == b.model.gdn_conv_silu &&
        a.model.bytes_bf16 == b.model.bytes_bf16 &&
        a.model.bytes_fp32 == b.model.bytes_fp32 &&
        a.activation == b.activation &&
        a.enable_device_graph_inputs == b.enable_device_graph_inputs &&
        a.virtual_q8_input_projections == b.virtual_q8_input_projections &&
        a.prefetch_q8_input_projections == b.prefetch_q8_input_projections &&
        a.wave8_q8_input_projections == b.wave8_q8_input_projections &&
        a.virtual_q8_output_projection == b.virtual_q8_output_projection &&
        a.wave8_q8_output_projection == b.wave8_q8_output_projection &&
        a.recurrent_layout == b.recurrent_layout && a.recurrent_storage == b.recurrent_storage;
}

std::uint64_t island_organization_identity(const HipGdnLauncherConfig& config) noexcept {
    auto identity = implementation_identity(config);
    hash_u64(identity, 0x47444e49534c3031ULL); // GDNISL01: organization only.
    return identity;
}
bool same_island_memory(const ExecutionMemoryBinding& a, const ExecutionMemoryBinding& b) noexcept {
    return a.execution_identity == b.execution_identity && a.arena_base == b.arena_base &&
        a.arena_bytes == b.arena_bytes && a.allocation_generation == b.allocation_generation;
}
void require_island_d1_scope(const HipLaunchContext& scope, std::uint64_t identity) {
    const auto& record = scope.deferred_moe_status;
    if (scope.graph_capture || !scope.stream || scope.stream == hipStreamLegacy || scope.stream == hipStreamPerThread ||
        record.execution_identity != identity || !record.allocation_generation || !record.arena_base ||
        record.arena_base % alignof(std::uint32_t) || record.arena_bytes != 8 ||
        record.arena_base > std::numeric_limits<std::uintptr_t>::max() - 8U)
        throw std::invalid_argument("GDN islands require an owned eager D1 stream and bounded eight-byte record");
}
const void* wave8_island_input_symbol(std::uint32_t ggml_type) {
    switch (weight_kind(ggml_type)) {
    case WeightKind::f32: return reinterpret_cast<const void*>(project_bf16_to_fp32_kernel<WeightKind::f32>);
    case WeightKind::f16: return reinterpret_cast<const void*>(project_bf16_to_fp32_kernel<WeightKind::f16>);
    case WeightKind::bf16: return reinterpret_cast<const void*>(project_bf16_to_fp32_kernel<WeightKind::bf16>);
    case WeightKind::q8_0: return reinterpret_cast<const void*>(project_prefetch8_wave8_q8_bf16_to_fp32_kernel);
    case WeightKind::iq4_nl: return reinterpret_cast<const void*>(project_bf16_to_fp32_kernel<WeightKind::iq4_nl>);
    case WeightKind::iq4_xs: return reinterpret_cast<const void*>(project_bf16_to_fp32_kernel<WeightKind::iq4_xs>);
#define STRIX_GGML_KIND_CASE(KIND, type) case WeightKind::KIND: return reinterpret_cast<const void*>(project_bf16_to_fp32_kernel<WeightKind::KIND>);
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_CASE) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_CASE)
#undef STRIX_GGML_KIND_CASE
    }
    throw std::invalid_argument("GDN island input projection type is unsupported");
}
void validate_island_graph(hipGraph_t graph, const GdnKernelLedger& ledger,
    const PreparedGdnLaunch& prepared, const HipGdnLauncherConfig& config) {
    if (ledger.size() != 10) throw std::runtime_error("GDN island emitter must record ten kernel nodes");
    std::size_t count = 0;
    check_hip(hipGraphGetNodes(graph, nullptr, &count), "GDN island node count");
    if (count != ledger.size()) throw std::runtime_error("GDN island must capture exactly ten nodes");
    std::vector<hipGraphNode_t> nodes(count);
    check_hip(hipGraphGetNodes(graph, nodes.data(), &count), "GDN island node inventory");
    if (count != ledger.size()) throw std::runtime_error("GDN island node inventory changed");
    std::vector<bool> used(count, false);
    hipGraphNode_t previous = nullptr;
    std::size_t ordinal = 0;
    // Derive symbols from admitted types/config independently of the emitter's
    // ledger: numerical equality alone cannot distinguish projection geometries.
    std::array<const void*, 5> projection_symbols{};
    if (config.wave8_q8_input_projections && config.wave8_q8_output_projection) {
        projection_symbols = {wave8_island_input_symbol(prepared.tensors.qkv.manifest->ggml_type),
            wave8_island_input_symbol(prepared.tensors.z.manifest->ggml_type),
            wave8_island_input_symbol(prepared.tensors.beta.manifest->ggml_type),
            wave8_island_input_symbol(prepared.tensors.alpha.manifest->ggml_type),
            reinterpret_cast<const void*>(project_prefetch8_wave8_q8_fp32_to_bf16_kernel)};
    }
    if (!gdn_island_recurrent_layout_admitted(config.recurrent_layout) ||
        config.recurrent_storage != GdnRecurrentStorage::key_value)
        throw std::runtime_error("GDN island recurrent configuration is not admitted");
    const auto recurrent_symbol = config.recurrent_layout == HipGdnRecurrentLayout::baseline
        ? reinterpret_cast<const void*>(recurrent_step_kernel)
        : reinterpret_cast<const void*>(recurrent_layout_diagnostic_kernel<gdn_recurrent_layout_diagnostic::Layout::transposed>);
    for (const auto& expected : ledger) {
        std::size_t selected = count;
        for (std::size_t i = 0; i < count; ++i) {
            if (used[i]) continue;
            std::size_t dependencies = 0;
            check_hip(hipGraphNodeGetDependencies(nodes[i], nullptr, &dependencies), "GDN island dependencies");
            if (dependencies != (previous ? 1U : 0U)) continue;
            if (previous) {
                hipGraphNode_t dependency = nullptr;
                check_hip(hipGraphNodeGetDependencies(nodes[i], &dependency, &dependencies), "GDN island dependency edge");
                if (dependencies != 1 || dependency != previous) continue;
            }
            if (selected != count) throw std::runtime_error("GDN island graph is not one linear chain");
            selected = i;
        }
        if (selected == count) throw std::runtime_error("GDN island graph has a missing linear dependency");
        hipGraphNodeType type{};
        check_hip(hipGraphNodeGetType(nodes[selected], &type), "GDN island node type");
        if (type != hipGraphNodeTypeKernel) throw std::runtime_error("GDN island captured a non-kernel node");
        hipKernelNodeParams parameters{};
        check_hip(hipGraphKernelNodeGetParams(nodes[selected], &parameters), "GDN island kernel parameters");
        if (parameters.func != expected.function || parameters.gridDim.x != expected.grid.x ||
            parameters.gridDim.y != expected.grid.y || parameters.gridDim.z != expected.grid.z ||
            parameters.blockDim.x != expected.block.x || parameters.blockDim.y != expected.block.y ||
            parameters.blockDim.z != expected.block.z || parameters.sharedMemBytes != expected.shared_bytes)
            throw std::runtime_error("GDN island kernel node differs from the emitter ledger");
        if (config.wave8_q8_input_projections && config.wave8_q8_output_projection &&
            ((ordinal < 4U && parameters.func != projection_symbols[ordinal]) ||
             (ordinal == 9U && parameters.func != projection_symbols[4])))
            throw std::runtime_error("GDN island captured projection symbol differs from its admitted wave8 organization");
        if (ordinal == 7U && parameters.func != recurrent_symbol)
            throw std::runtime_error("GDN island recurrent symbol differs from its admitted layout");
        used[selected] = true; previous = nodes[selected];
        ++ordinal;
    }
}
} // namespace

struct HipGdnIslands::Impl {
    struct Island {
        GdnIslandDescriptor descriptor;
        hipGraph_t graph = nullptr;
        hipGraphExec_t executable = nullptr;
    };
    // Retain actual owners until every graph is destroyed after quiescence.
    std::shared_ptr<const void> resources_owner;
    std::shared_ptr<const HipModelResidency> weights;
    std::shared_ptr<const HipSetupQuiescenceGuard> guard;
    const CompiledSerialExecution compiled;
    const HipGdnLauncherConfig config;
    const ExecutionRuntimeResources* resources;
    const ExecutionRuntimeResources resource_bindings;
    const HipLaunchContext scope;
    const HipGdnIslandBudget budget;
    std::vector<std::unique_ptr<Island>> steps;
    HipGdnIslandReceipt observed;
    hipStream_t setup_stream = nullptr;
    bool capturing = false, failed = false;
    hipGraph_t abandoned_graph = nullptr;
    ModelContentIdentity model_content;
    std::uint64_t manifest_identity = 0;
    std::uint64_t weight_generation = 0;

    Impl(const CompiledSerialExecution& c, const HipGdnLauncherConfig& cfg,
        const ExecutionRuntimeResources& r, const HipLaunchContext& s,
        std::shared_ptr<const HipModelResidency> w, std::shared_ptr<const void> owner,
        std::shared_ptr<const HipSetupQuiescenceGuard> g, HipGdnIslandBudget b)
        : resources_owner(std::move(owner)), weights(std::move(w)), guard(std::move(g)),
          compiled(c), config(cfg), resources(&r), resource_bindings(r), scope(s), budget(b) {}

    void finish_capture() noexcept {
        if (!capturing) return;
        hipGraph_t graph = nullptr;
        const auto error = hipStreamEndCapture(setup_stream, &graph);
        if (error == hipSuccess || error == hipErrorStreamCaptureInvalidated) capturing = false;
        if (graph && hipGraphDestroy(graph) != hipSuccess) abandoned_graph = graph;
    }
    bool destroy() noexcept {
        if (capturing) return false;
        bool okay = true;
        for (auto& island : steps) if (island) {
            if (island->executable) {
                if (hipGraphExecDestroy(island->executable) == hipSuccess) island->executable = nullptr;
                else okay = false;
            }
            if (island->graph) {
                if (hipGraphDestroy(island->graph) == hipSuccess) island->graph = nullptr;
                else okay = false;
            }
        }
        if (abandoned_graph) {
            if (hipGraphDestroy(abandoned_graph) == hipSuccess) abandoned_graph = nullptr;
            else okay = false;
        }
        if (setup_stream) {
            if (hipStreamDestroy(setup_stream) == hipSuccess) setup_stream = nullptr;
            else okay = false;
        }
        return okay;
    }
    void device_range(GdnIslandView range) const {
        if (!range.address || !range.bytes || !range.generation ||
            range.bytes > std::numeric_limits<std::uintptr_t>::max() - range.address)
            throw std::invalid_argument("GDN island has an invalid native resource range");
        hipPointerAttribute_t attributes{};
        check_hip(hipPointerGetAttributes(&attributes, reinterpret_cast<const void*>(range.address)), "GDN island device identity");
        if (attributes.type != hipMemoryTypeDevice || attributes.isManaged || attributes.device != guard->device_index())
            throw std::invalid_argument("GDN island scratch/state requires original-device allocations");
        hipDeviceptr_t allocation{}; std::size_t bytes = 0;
        check_hip(hipMemGetAddressRange(&allocation, &bytes, reinterpret_cast<hipDeviceptr_t>(range.address)),
            "GDN island native extent");
        const auto base = reinterpret_cast<std::uintptr_t>(allocation);
        if (range.address < base || range.address - base > bytes || range.bytes > bytes - (range.address - base))
            throw std::invalid_argument("GDN island resource exceeds its native allocation");
    }
    void build() {
        if (current_hip_launch_context()) throw std::invalid_argument("GDN island setup requires no active launch scope");
        validate_config(config);
        if (!gdn_island_recurrent_layout_admitted(config.recurrent_layout))
            throw std::invalid_argument("GDN islands require baseline or transposed recurrent LDS layout");
        if (config.recurrent_storage != GdnRecurrentStorage::key_value)
            throw std::invalid_argument("GDN islands require key-value recurrent storage");
        const bool legacy = !config.virtual_q8_input_projections && !config.prefetch_q8_input_projections &&
            !config.virtual_q8_output_projection;
        const bool profile = config.virtual_q8_input_projections && config.prefetch_q8_input_projections &&
            config.virtual_q8_output_projection;
        if (config.enable_device_graph_inputs ||
            config.wave8_q8_input_projections != config.wave8_q8_output_projection ||
            (config.wave8_q8_input_projections && !profile) ||
            (!legacy && !profile) ||
            compiled.device_graph_inputs || compiled.logical_rows != 1 || compiled.request_concurrency != 1 ||
            !compiled.execution_identity || !compiled.kernel_plan_identity ||
            compiled.execution.steps.size() != compiled.io.steps.size() ||
            compiled.execution.steps.size() != compiled.bindings.bindings.size())
            throw std::invalid_argument("GDN islands require a complete eager R1/C1 legacy, prefetch8 or both-wave8 source");
        if (!resources_owner || !weights || !guard || &weights->binding() != resources->model_weights || !weights->generation())
            throw std::invalid_argument("GDN islands require the actual retained D1 and weight owners");
        require_island_d1_scope(scope, compiled.execution_identity);
        validate_execution_runtime_resources(compiled.execution_identity, compiled.arena, *resources, 0);
        model_content = weights->binding().model_content_identity;
        manifest_identity = weights->binding().manifest_layout_identity;
        weight_generation = weights->generation();
        std::uint32_t count = 0;
        std::vector<std::uint32_t> layers;
        ExecutionBindingRequest request;
        for (std::size_t i = 0; i < compiled.execution.steps.size(); ++i) {
            const auto& step = compiled.execution.steps[i];
            const auto& binding = compiled.bindings.bindings[i];
            const auto& io = compiled.io.steps[i];
            if (step.id != i || binding.step_id != step.id || binding.op != step.op || io.step_id != step.id || io.op != step.op)
                throw std::invalid_argument("GDN island source step/binding/IO must be dense and consistent");
            if (step.op != SerialOp::gated_delta_net) continue;
            if (!step.layer || binding.host_or_runtime_step || !binding.kernel_id ||
                *binding.kernel_id != kernel_id(request, config) ||
                std::find(layers.begin(), layers.end(), *step.layer) != layers.end())
                throw std::invalid_argument("GDN island source requires one exact selected kernel per GDN layer");
            layers.push_back(*step.layer);
            if (count == std::numeric_limits<std::uint32_t>::max()) throw std::overflow_error("GDN island count overflow");
            ++count;
        }
        if (!count || count > budget.max_islands || count > budget.max_kernel_nodes / 10U)
            throw std::invalid_argument("GDN island count exceeds explicit setup admission");
        steps.resize(compiled.execution.steps.size());
        observed.organization_identity = island_organization_identity(config);
        observed.islands = count; observed.kernel_nodes = count * 10U;
        const auto& scratch = *resources->scratch;
        const auto& state = *resources->session_state_memory;
        device_range({scratch.arena_base, scratch.arena_bytes, scratch.allocation_generation});
        device_range({state.device_base, state.device_bytes, state.device_allocation_generation});
        device_range({scope.deferred_moe_status.arena_base, 8, scope.deferred_moe_status.allocation_generation});
        const std::array<GdnIslandView, 3> writable_owners{{
            {scratch.arena_base, scratch.arena_bytes, scratch.allocation_generation},
            {state.device_base, state.device_bytes, state.device_allocation_generation},
            {scope.deferred_moe_status.arena_base, 8, scope.deferred_moe_status.allocation_generation}}};
        const auto overlaps = [](GdnIslandView a, GdnIslandView b) {
            return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
        };
        for (std::size_t i = 0; i < writable_owners.size(); ++i)
            for (std::size_t j = i + 1; j < writable_owners.size(); ++j)
                if (overlaps(writable_owners[i], writable_owners[j]))
                    throw std::invalid_argument("GDN island scratch, state and D1 record owners must be disjoint");
        for (const auto& tensor : weights->binding().tensors) {
            if (!runtime_tensor_gpu_visible(tensor.storage)) continue;
            if (!tensor.address || !tensor.bytes || tensor.bytes > std::numeric_limits<std::uintptr_t>::max() - tensor.address)
                throw std::invalid_argument("GDN island weight extent is invalid");
            for (const auto& owner : writable_owners)
                if (overlaps(owner, {tensor.address, tensor.bytes, tensor.allocation_generation}))
                    throw std::invalid_argument("GDN island writable owner aliases weights");
        }
        check_hip(hipStreamCreateWithFlags(&setup_stream, hipStreamNonBlocking), "GDN island setup stream create");
        std::size_t before = 0, after = 0, total = 0;
        const bool before_observed = hipMemGetInfo(&before, &total) == hipSuccess;
        try {
            for (std::size_t i = 0; i < compiled.execution.steps.size(); ++i) {
                const auto& step = compiled.execution.steps[i];
                if (step.op != SerialOp::gated_delta_net) continue;
                const auto& io = compiled.io.steps[i];
                ExecutionDispatchContext context;
                context.execution_identity = compiled.execution_identity;
                context.step_id = step.id; context.op = step.op; context.layer = step.layer;
                context.logical_rows = 1; context.request_concurrency = 1;
                context.reads = io.reads; context.writes = io.writes; context.temporaries = io.temporaries;
                context.arena = &compiled.arena; context.memory = resources->scratch; context.resources = resources;
                context.session_state_layout = resources->session_state_layout;
                context.session_state_memory = resources->session_state_memory;
                context.reads_state = step.reads_state; context.writes_state = step.writes_state;
                context.kernel_id = *compiled.bindings.bindings[i].kernel_id;
                const auto prepared = [&] {
                    const HipLaunchScope prepare_scope(scope);
                    return prepare_gdn_launch(context, &config);
                }();
                auto island = std::make_unique<Island>();
                island->descriptor = island_descriptor(context, prepared, weights->binding());
                for (const auto& buffer : island->descriptor.buffers) device_range(buffer);
                for (const auto& view : island->descriptor.state) device_range(view);
                for (const auto& tensor : island->descriptor.tensors) {
                    if ((tensor.storage != RuntimeTensorStorage::resident && tensor.storage != RuntimeTensorStorage::repacked_cache &&
                         tensor.storage != RuntimeTensorStorage::registered_file_mapped) ||
                        tensor.generation != weights->generation() || !weights->covers_device_range(tensor.address, tensor.bytes))
                        throw std::invalid_argument("GDN island tensor is outside its actual retained residency");
                }
                steps[i] = std::move(island);
                GdnKernelLedger ledger; ledger.reserve(10);
                if (current_hip_launch_context())
                    throw std::runtime_error("GDN island capture requires no active launch scope before capture");
                check_hip(hipStreamBeginCapture(setup_stream, hipStreamCaptureModeThreadLocal), "GDN island capture begin");
                capturing = true;
                hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
                check_hip(hipStreamIsCapturing(setup_stream, &capture), "GDN island capture state");
                if (capture != hipStreamCaptureStatusActive || current_hip_launch_context())
                    throw std::runtime_error("GDN island capture requires the explicit setup target without a launch scope");
                emit_gdn_sequence(prepared, config, {setup_stream, nullptr, nullptr, 0, &ledger});
                const auto ended = hipStreamEndCapture(setup_stream, &steps[i]->graph);
                if (ended == hipSuccess || ended == hipErrorStreamCaptureInvalidated) capturing = false;
                check_hip(ended, "GDN island capture end");
                validate_island_graph(steps[i]->graph, ledger, prepared, config);
                check_hip(hipGraphInstantiate(&steps[i]->executable, steps[i]->graph, nullptr, nullptr, 0), "GDN island instantiate");
            }
        } catch (...) { finish_capture(); throw; }
        check_hip(hipStreamDestroy(setup_stream), "GDN island setup stream destroy"); setup_stream = nullptr;
        const bool after_observed = hipMemGetInfo(&after, &total) == hipSuccess;
        if (before_observed && after_observed && before <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) &&
            after <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
            observed.memory_observed = true;
            observed.observed_setup_free_delta_bytes = static_cast<std::int64_t>(before) - static_cast<std::int64_t>(after);
            if (budget.observed_setup_delta_reject_bytes && observed.observed_setup_free_delta_bytes > 0 &&
                static_cast<std::uint64_t>(observed.observed_setup_free_delta_bytes) > budget.observed_setup_delta_reject_bytes)
                throw std::invalid_argument("GDN island observed setup memory delta exceeds explicit rejection threshold");
        }
    }
    Island& admit(const ExecutionDispatchContext& context, const PreparedGdnLaunch& prepared) {
        if (failed) throw std::logic_error("GDN island owner is failed and cannot be reused");
        const auto* active = current_hip_launch_context();
        if (!active || active->graph_capture || active->stream != scope.stream ||
            !same_island_memory(active->deferred_moe_status, scope.deferred_moe_status) ||
            context.execution_identity != compiled.execution_identity || context.resources != resources ||
            resources->scratch != resource_bindings.scratch || resources->model_manifest != resource_bindings.model_manifest ||
            resources->tensor_access != resource_bindings.tensor_access ||
            resources->session_state_layout != resource_bindings.session_state_layout ||
            resources->session_state_memory != resource_bindings.session_state_memory ||
            resources->model_weights != &weights->binding() || weights->binding().model_content_identity != model_content ||
            weights->binding().manifest_layout_identity != manifest_identity || weights->generation() != weight_generation ||
            context.step_id >= steps.size() || !steps[context.step_id] ||
            !context.kernel_id || !compiled.bindings.bindings[context.step_id].kernel_id ||
            *context.kernel_id != *compiled.bindings.bindings[context.step_id].kernel_id ||
            !(island_descriptor(context, prepared, weights->binding(), &steps[context.step_id]->descriptor) ==
                steps[context.step_id]->descriptor))
            throw std::invalid_argument("GDN island replay scope, owner or complete descriptor changed");
        return *steps[context.step_id];
    }
    void registration(const CompiledSerialExecution& source, const HipGdnLauncherConfig& cfg) const {
        if (failed || source.execution_identity != compiled.execution_identity ||
            source.kernel_plan_identity != compiled.kernel_plan_identity || source.device_graph_inputs ||
            source.logical_rows != 1 || source.request_concurrency != 1 || !same_island_config(config, cfg) ||
            source.execution.steps.size() != compiled.execution.steps.size() ||
            source.bindings.bindings.size() != compiled.bindings.bindings.size())
            throw std::invalid_argument("GDN island registration source/config differs from its fixed owner");
        std::size_t count = 0;
        for (std::size_t i = 0; i < source.execution.steps.size(); ++i) {
            const auto& step = source.execution.steps[i];
            if (step.op != SerialOp::gated_delta_net) {
                if (steps[i]) throw std::invalid_argument("GDN island registration lost a GDN step");
                continue;
            }
            if (!steps[i] || step.id != i || !step.layer || *step.layer != steps[i]->descriptor.layer)
                throw std::invalid_argument("GDN island registration step/layer coverage changed");
            const auto& binding = source.bindings.bindings[i];
            const auto& original = compiled.bindings.bindings[i];
            if (binding.step_id != step.id || binding.op != step.op || binding.host_or_runtime_step ||
                binding.kernel_id != original.kernel_id)
                throw std::invalid_argument("GDN island registration selected numerical kernel changed");
            ++count;
        }
        if (count != observed.islands) throw std::invalid_argument("GDN island registration has incomplete coverage");
    }
};

HipGdnIslands::HipGdnIslands(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
std::shared_ptr<HipGdnIslands> HipGdnIslands::create(const CompiledSerialExecution& compiled,
    const HipGdnLauncherConfig& config, const ExecutionRuntimeResources& resources, const HipLaunchContext& scope,
    std::shared_ptr<const HipModelResidency> weights, std::shared_ptr<const void> owner,
    std::shared_ptr<const HipSetupQuiescenceGuard> guard, HipGdnIslandBudget budget) {
    if (!weights || !owner || !guard) throw std::invalid_argument("GDN island setup requires strong actual owners");
    if (config.dense_a8_projections) throw std::invalid_argument("GDN islands capture the incumbent projections, not dense A8");
    auto impl = std::make_unique<Impl>(compiled, config, resources, scope, std::move(weights), std::move(owner), guard, budget);
    try { guard->run([&] { impl->build(); }); }
    catch (...) {
        impl->finish_capture(); guard->recover_or_fail_stop();
        if (!impl->destroy()) (void)impl.release();
        throw;
    }
    return std::shared_ptr<HipGdnIslands>(new HipGdnIslands(std::move(impl)));
}
HipGdnIslands::~HipGdnIslands() {
    if (impl_) {
        impl_->guard->recover_or_fail_stop();
        if (!impl_->destroy()) (void)impl_.release();
    }
}
HipGdnIslandReceipt HipGdnIslands::receipt() const noexcept { return impl_->observed; }
void HipGdnIslands::preflight_gdn_island(const ExecutionDispatchContext& context, void* opaque) {
    if (!opaque) throw std::invalid_argument("GDN island launcher lacks its owner");
    auto& owner = *static_cast<HipGdnIslands*>(opaque);
    try {
        const auto prepared = prepare_gdn_launch(context, &owner.impl_->config);
        (void)owner.impl_->admit(context, prepared);
    } catch (...) { owner.impl_->failed = true; throw; }
}
void HipGdnIslands::launch_gdn_island(const ExecutionDispatchContext& context, void* opaque) {
    if (!opaque) throw std::invalid_argument("GDN island launcher lacks its owner");
    auto& owner = *static_cast<HipGdnIslands*>(opaque);
    try {
        const auto prepared = prepare_gdn_launch(context, &owner.impl_->config);
        auto& island = owner.impl_->admit(context, prepared);
        if (owner.impl_->observed.replays == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("GDN island replay counter overflow");
        check_hip(hipGraphLaunch(island.executable, owner.impl_->scope.stream), "GDN island replay");
        ++owner.impl_->observed.replays;
    } catch (...) { owner.impl_->failed = true; throw; }
}


ExecutionLauncher
make_hip_gdn_launcher(
    std::string kernel_id_value,
    HipGdnLauncherConfig* config, bool reserve_device_scratch) {
    if (kernel_id_value.empty() ||
        config == nullptr) {
        throw std::invalid_argument(
            "HIP GDN launcher requires id and config");
    }
    validate_config(*config);
    if ((kernel_id_value.find(q8_projection_organization::dense_a8_marker) != std::string::npos) != config->dense_a8_projections)
        throw std::invalid_argument("HIP GDN launcher dense A8 identity differs from config");
    // The dense A8 scratch is allocated at setup, never on the first decode step.
    if (reserve_device_scratch && config->dense_a8_projections) q8_rows::dense_a8_reserve();
    if (config->recurrent_storage != GdnRecurrentStorage::key_value || kernel_id_value.find(".state-vk-v1") != std::string::npos) {
        ExecutionBindingRequest request;
        if (kernel_id_value != kernel_id(request, *config))
            throw std::invalid_argument("GDN recurrent storage launcher requires its exact eager C1 kernel id");
    }
    if (config->recurrent_layout != HipGdnRecurrentLayout::baseline || has_gdn_recurrent_layout_marker(kernel_id_value)) {
        ExecutionBindingRequest request;
        if (kernel_id_value != kernel_id(request, *config))
            throw std::invalid_argument("GDN recurrent LDS launcher requires its exact eager C1 kernel id");
    }
    if (config->wave8_q8_input_projections || config->wave8_q8_output_projection ||
        kernel_id_value.find(q8_projection_organization::gdn_wave8_marker) != std::string::npos ||
        kernel_id_value.find(q8_projection_organization::gdn_output_wave8_marker) != std::string::npos) {
        ExecutionBindingRequest request;
        if (kernel_id_value != kernel_id(request, *config))
            throw std::invalid_argument("wave8 GDN launcher requires its exact eager C1 kernel id");
    }
    return ExecutionLauncher{
        std::move(kernel_id_value),
        implementation_identity(
            *config),
        &launch_gdn,
        config,
        &preflight_gdn,
    };
}

void HipGdnLauncherBundle::
register_for_compiled_execution(
    const CompiledSerialExecution& compiled,
    ExecutionLauncherRegistry& registry, HipGdnIslands* islands, bool reserve_device_scratch) {
    if (compiled.execution_identity == 0 ||
        compiled.execution.steps.size() !=
            compiled.bindings.bindings.size()) {
        throw std::invalid_argument(
            "HIP GDN bundle requires valid compiled execution");
    }

    if (islands) islands->impl_->registration(compiled, config_);
    std::string registered_id;
    for (std::size_t index = 0;
         index <
            compiled.execution.steps.size();
         ++index) {
        if (compiled.execution.steps[index].op !=
            SerialOp::gated_delta_net) {
            continue;
        }

        const auto& binding =
            compiled.bindings.bindings[index];
        if (binding.host_or_runtime_step ||
            !binding.kernel_id.has_value() ||
            binding.kernel_id->empty()) {
            throw std::invalid_argument(
                "HIP GDN step has no kernel binding");
        }

        std::string_view selected_id=*binding.kernel_id;
        if (config_.recurrent_storage != GdnRecurrentStorage::key_value || selected_id.find(".state-vk-v1") != std::string_view::npos) {
            ExecutionBindingRequest request;
            request.logical_rows = compiled.logical_rows;
            request.request_concurrency = compiled.request_concurrency;
            if (*binding.kernel_id != kernel_id(request, config_))
                throw std::invalid_argument("HIP GDN selected kernel disagrees with recurrent storage configuration");
        }
        if (config_.recurrent_layout != HipGdnRecurrentLayout::baseline || has_gdn_recurrent_layout_marker(selected_id)) {
            ExecutionBindingRequest request;
            request.logical_rows = compiled.logical_rows;
            request.request_concurrency = compiled.request_concurrency;
            if (*binding.kernel_id != kernel_id(request, config_))
                throw std::invalid_argument("HIP GDN selected kernel disagrees with recurrent LDS configuration");
        }
        constexpr std::string_view output_suffix=".virtual256.q8-output";
        if(selected_id.ends_with(output_suffix)!=config_.virtual_q8_output_projection)
            throw std::invalid_argument("HIP GDN selected kernel disagrees with virtual Q8 output configuration");
        if(config_.virtual_q8_output_projection) selected_id.remove_suffix(output_suffix.size());
        if (selected_id.ends_with(".virtual256.q8-input") !=
            config_.virtual_q8_input_projections)
            throw std::invalid_argument("HIP GDN selected kernel disagrees with virtual Q8 configuration");
        if (selected_id.ends_with(".prefetch8.virtual256.q8-input") != config_.prefetch_q8_input_projections)
            throw std::invalid_argument("HIP GDN selected kernel disagrees with prefetch Q8 configuration");
        if (config_.wave8_q8_input_projections || config_.wave8_q8_output_projection ||
            selected_id.find(q8_projection_organization::gdn_wave8_marker) != std::string::npos ||
            selected_id.find(q8_projection_organization::gdn_output_wave8_marker) != std::string::npos) {
            ExecutionBindingRequest request;
            request.logical_rows = compiled.logical_rows;
            request.request_concurrency = compiled.request_concurrency;
            if (*binding.kernel_id != kernel_id(request, config_))
                throw std::invalid_argument("HIP GDN wave8 selected kernel disagrees with exact configuration");
        }

        if (registered_id.empty()) {
            registered_id =
                *binding.kernel_id;
            if (islands) {
                registry.add_kernel_launcher({registered_id, island_organization_identity(config_),
                    &HipGdnIslands::launch_gdn_island, islands, &HipGdnIslands::preflight_gdn_island});
            } else {
                registry.add_kernel_launcher(
                    make_hip_gdn_launcher(registered_id, &config_, reserve_device_scratch));
            }
        } else if (registered_id !=
                   *binding.kernel_id) {
            throw std::invalid_argument(
                "HIP GDN bundle requires one selected kernel id");
        }
    }
}


#ifdef STRIX_BUILD_GDN_LM_WAVE8_DIAGNOSTIC
namespace gdn_lm_wave8_diagnostic::detail {
const void* gdn_symbol(Arm arm) {
    switch (arm) {
    case Arm::reference: case Arm::prefetch1:
        return reinterpret_cast<const void*>(project_prefetch8_q8_bf16_to_fp32_kernel);
    case Arm::wave8:
        return reinterpret_cast<const void*>(project_prefetch8_wave8_q8_bf16_to_fp32_kernel);
    }
    throw std::invalid_argument("invalid GDN diagnostic arm");
}
void enqueue_gdn(Shape shape, Span input, Span weights, Span output, Arm arm, hipStream_t stream) {
    const auto launch = validate(shape, input, weights, output, Kind::gdn_bf16, arm);
    check_hip(hipPeekAtLastError(), "GDN wave8 diagnostic stale error");
    if (arm == Arm::wave8) {
        hipLaunchKernelGGL(project_prefetch8_wave8_q8_bf16_to_fp32_kernel,
            dim3(launch.grid_x), dim3(launch.block_x), 0, stream,
            reinterpret_cast<const std::uint16_t*>(input.address),
            reinterpret_cast<const void*>(weights.address), shape.width, shape.outputs,
            reinterpret_cast<float*>(output.address));
    } else {
        hipLaunchKernelGGL(project_prefetch8_q8_bf16_to_fp32_kernel,
            dim3(launch.grid_x), dim3(launch.block_x), 0, stream,
            reinterpret_cast<const std::uint16_t*>(input.address),
            reinterpret_cast<const void*>(weights.address), shape.width, shape.outputs,
            reinterpret_cast<float*>(output.address));
    }
    check_hip(hipGetLastError(), "GDN wave8 diagnostic launch");
}
} // namespace gdn_lm_wave8_diagnostic::detail
#endif

#ifdef STRIX_BUILD_GDN_RECURRENT_LAYOUT_DIAGNOSTIC
namespace gdn_recurrent_layout_diagnostic {
const void* kernel_symbol(Layout layout) {
    switch (layout) {
    case Layout::reference:
        return reinterpret_cast<const void*>(recurrent_step_kernel);
    case Layout::stride32:
        return reinterpret_cast<const void*>(recurrent_layout_diagnostic_kernel<Layout::stride32>);
    case Layout::stride33:
        return reinterpret_cast<const void*>(recurrent_layout_diagnostic_kernel<Layout::stride33>);
    case Layout::transposed:
        return reinterpret_cast<const void*>(recurrent_layout_diagnostic_kernel<Layout::transposed>);
    }
    throw std::invalid_argument("invalid GDN recurrent diagnostic layout");
}

void enqueue(Shape shape, const Bindings& bindings, Layout layout, hipStream_t stream) {
    const auto launch = validate(shape, bindings, layout);
    if (!stream || stream == hipStreamPerThread || stream == hipStreamLegacy)
        throw std::invalid_argument("GDN recurrent diagnostic requires an explicitly owned stream");
    check_hip(hipPeekAtLastError(), "GDN recurrent diagnostic stale error");
    const auto* qkv = reinterpret_cast<const float*>(bindings.qkv.address);
    const auto* beta = reinterpret_cast<const float*>(bindings.beta.address);
    const auto* gate = reinterpret_cast<const float*>(bindings.gate.address);
    auto* state = reinterpret_cast<float*>(bindings.state.address);
    auto* core = reinterpret_cast<float*>(bindings.core.address);
    auto* status = reinterpret_cast<std::uint32_t*>(bindings.graph_status.address);
#define STRIX_RECURRENT_LAYOUT_LAUNCH(kernel) \
    hipLaunchKernelGGL(kernel, dim3(launch.grid_x), dim3(launch.block_x), 0, stream, \
        qkv, beta, gate, state, core, shape.key_heads, shape.value_heads, shape.head_dim, status)
    switch (layout) {
    case Layout::reference: STRIX_RECURRENT_LAYOUT_LAUNCH(recurrent_step_kernel); break;
    case Layout::stride32:
        STRIX_RECURRENT_LAYOUT_LAUNCH(HIP_KERNEL_NAME(recurrent_layout_diagnostic_kernel<Layout::stride32>)); break;
    case Layout::stride33:
        STRIX_RECURRENT_LAYOUT_LAUNCH(HIP_KERNEL_NAME(recurrent_layout_diagnostic_kernel<Layout::stride33>)); break;
    case Layout::transposed:
        STRIX_RECURRENT_LAYOUT_LAUNCH(HIP_KERNEL_NAME(recurrent_layout_diagnostic_kernel<Layout::transposed>)); break;
    }
#undef STRIX_RECURRENT_LAYOUT_LAUNCH
    check_hip(hipGetLastError(), "GDN recurrent diagnostic launch");
}
} // namespace gdn_recurrent_layout_diagnostic
#endif

}  // namespace strix
