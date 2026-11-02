#include "strix/hip_mtp_fast_draft.hpp"
#include "strix/prefill_identity.hpp"
#include "hip_prefill_pool.hpp"
#include "strix/dispatch_fuse.hpp"
#include "strix/hip_lane_chain_hyper_output_tests.hpp"
#include "strix/hip_hc_wide_rows_tests.hpp"
#include "strix/hip_lane_chain_hyper_projection_tests.hpp"
#include "strix/hip_hc_deferred_norm_tests.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_hyper_mix_launcher.hpp"
#include "strix/hip_mtp_projection.hpp"
#include "strix/hc_projection_prefetch.hpp"
#include "strix/distributed_stage_resources.hpp"
#include "strix/lane_chain_hyper_output.hpp"
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <map>
#include <mutex>
#include <set>

#include "strix/dispatch_resources.hpp"
#include "strix/q6p_registry.hpp"
#include "hip_q6p_weights.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <type_traits>
#include <vector>

namespace strix {
namespace {

constexpr std::uint32_t kThreads = 256;
constexpr std::uint32_t kMaxHc = 8;
constexpr std::uint32_t kMaxLowrank = 512;
constexpr std::uint32_t kStagedThreads = 128;
constexpr std::uint32_t kStagedOutputs = 8;
constexpr std::uint32_t kStagedPitch = kStagedThreads + 1;
constexpr std::uint64_t kLauncherTag =
    0x48595045524d4959ULL;
constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

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

void hash_u64(
    std::uint64_t& hash,
    std::uint64_t value) noexcept {
    for (unsigned shift = 0;
         shift < 64;
         shift += 8) {
        hash ^=
            (value >> shift) &
            0xffULL;
        hash *= fnv_prime;
    }
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

const char* storage_name(
    HipActivationStorage storage) noexcept {
    switch (storage) {
    case HipActivationStorage::bf16:
        return "bf16";
    case HipActivationStorage::fp32:
        return "fp32";
    }
    return "unknown";
}

void validate_config(
    const HipHyperMixLauncherConfig& config) {
    if (config.splitk_deferred_norm && (!config.splitk_projection || config.splitk_projection_fp64 ||
        !config.lane_chain_output || !config.fused_injection || !config.typed_projection ||
        !config.staged_projections || !config.parallel || config.prefetch_hc_projection_next_tile ||
        config.hidden_size != 2560U || config.hc_count != 4U || config.lowrank != 320U ||
        config.activation != HipActivationStorage::bf16))
        throw std::invalid_argument("deferred HC RMS requires canonical eager FP32 split-K lane-chain execution");
    if (config.prefetch_hc_projection_next_tile &&
        (!config.parallel || !config.staged_projections || !config.fused_injection ||
         !config.typed_projection || !config.prefetch_rows || config.splitk_projection ||
         config.splitk_projection_fp64 || config.hidden_size != 2560U || config.hc_count != 4U ||
         config.lowrank != 320U || config.activation != HipActivationStorage::bf16))
        throw std::invalid_argument("next-tile HC requires canonical BF16 typed prefetched eager projection");
    if (config.splitk_projection_fp64 && !config.splitk_projection)
        throw std::invalid_argument("FP64 HC diagnostic requires split-K projection");
    if (config.splitk_projection && (!config.parallel || !config.staged_projections ||
        !config.fused_injection || !config.typed_projection || !config.prefetch_rows ||
        static_cast<std::uint64_t>(config.hidden_size) * config.hc_count != 10240U))
        throw std::invalid_argument("split-K HC projection requires the admitted typed width10240 path");
    if (config.staged_projections && !config.parallel)
        throw std::invalid_argument("staged hyper projections require parallel hyper-mix");
    if (config.fused_injection && !config.staged_projections)
        throw std::invalid_argument("fused injection requires staged hyper projections");
    if(config.lane_chain_output && (!config.prefetch_q8_output || !lane_chain_hyper_output_lds(config.hc_count,config.lowrank) || config.hidden_size%32U))
        throw std::invalid_argument("lane-chain HC output requires eligible geometry and Q8 output prefetch");
    if (config.prefetch_q8_output && !config.typed_projection)
        throw std::invalid_argument("Q8 hyper output prefetch requires typed projection mode");
    if (config.prefetch_q8_output && config.lowrank % 32U != 0U)
        throw std::invalid_argument("Q8 hyper output prefetch requires lowrank divisible by 32");
    if (config.prefetch_rows && !config.typed_projection)
        throw std::invalid_argument("hyper row prefetch requires typed projection mode");
    if (config.typed_projection && !config.fused_injection)
        throw std::invalid_argument("typed hyper projection requires fused staged execution");
    if (config.hidden_size == 0 ||
        config.hc_count <= 1 ||
        config.hc_count > kMaxHc ||
        config.lowrank == 0 ||
        config.lowrank > kMaxLowrank ||
        !std::isfinite(config.rms_epsilon) ||
        config.rms_epsilon <= 0.0F) {
        throw std::invalid_argument(
            "HIP hyper-mix geometry is invalid");
    }
    if (config.activation !=
            HipActivationStorage::bf16 &&
        config.activation !=
            HipActivationStorage::fp32) {
        throw std::invalid_argument(
            "HIP hyper-mix activation storage is unsupported");
    }
}

__device__ float bf16_to_float(
    std::uint16_t bits) {
    return __uint_as_float(
        static_cast<std::uint32_t>(
            bits) << 16U);
}

template <typename Storage>
__device__ float load_activation(
    const Storage* values,
    std::uint64_t index);

template <>
__device__ float load_activation<float>(
    const float* values,
    std::uint64_t index) {
    return values[index];
}

template <>
__device__ float load_activation<std::uint16_t>(
    const std::uint16_t* values,
    std::uint64_t index) {
    return bf16_to_float(
        values[index]);
}

__device__ void store_activation(
    float* destination,
    float value) {
    *destination = value;
}

__device__ void store_activation(
    std::uint16_t* destination,
    float value) {
    const auto bits =
        __float_as_uint(value);
    const auto rounded =
        bits +
        0x7fffU +
        ((bits >> 16U) & 1U);
    *destination =
        static_cast<std::uint16_t>(
            rounded >> 16U);
}

#include "hip_qpf2k_pinned.inc"

__device__ std::uint8_t iq4_local_scale(
    const IQ4XSBlock& block,
    std::uint32_t subblock) {
    const auto packed =
        block.scales_l[
            subblock / 2U];
    const auto low =
        static_cast<std::uint8_t>(
            (packed >>
             (4U *
              (subblock % 2U))) &
            0x0fU);
    const auto high =
        static_cast<std::uint8_t>(
            (block.scales_h >>
             (2U * subblock)) &
            0x03U);
    return
        static_cast<std::uint8_t>(
            low |
            static_cast<std::uint8_t>(
                high << 4U));
}

__device__ float load_linear_weight(
    const void* weights,
    std::uint32_t ggml_type,
    std::uint32_t row_width,
    std::uint32_t output_index,
    std::uint32_t input_index) {
    const auto flat =
        static_cast<std::uint64_t>(
            output_index) *
            row_width +
        input_index;

    switch (ggml_type) {
    case 0:
        return static_cast<
            const float*>(
                weights)[flat];

    case 1:
        return __half2float(
            static_cast<
                const __half*>(
                    weights)[flat]);

    case 30:
        return bf16_to_float(
            static_cast<
                const std::uint16_t*>(
                    weights)[flat]);

    case 8: {
        const auto blocks_per_row =
            row_width / 32U;
        const auto& block =
            static_cast<
                const Q8Block*>(
                    weights)[
                static_cast<
                    std::uint64_t>(
                    output_index) *
                    blocks_per_row +
                input_index / 32U];
        return
            __half2float(
                block.scale) *
            static_cast<float>(
                block.values[
                    input_index % 32U]);
    }

    case 23: {
        const auto blocks_per_row =
            row_width / 256U;
        const auto& block =
            static_cast<
                const IQ4XSBlock*>(
                    weights)[
                static_cast<
                    std::uint64_t>(
                    output_index) *
                    blocks_per_row +
                input_index / 256U];
        const auto within =
            input_index % 256U;
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
                ? static_cast<
                      std::uint8_t>(
                      packed & 0x0fU)
                : static_cast<
                      std::uint8_t>(
                      packed >> 4U);
        const int local =
            static_cast<int>(
                iq4_local_scale(
                    block,
                    subblock)) -
            32;
        return
            __half2float(block.d) *
            static_cast<float>(local) *
            static_cast<float>(
                iq4_codebook[code]);
    }

    default:
        return NAN;
    }
}

__device__ float load_norm_weight(
    const void* weights,
    std::uint32_t ggml_type,
    std::uint32_t index) {
    switch (ggml_type) {
    case 0:
        return static_cast<
            const float*>(
                weights)[index];
    case 1:
        return __half2float(
            static_cast<
                const __half*>(
                    weights)[index]);
    case 30:
        return bf16_to_float(
            static_cast<
                const std::uint16_t*>(
                    weights)[index]);
    default:
        return NAN;
    }
}


template <typename Storage>
__device__ float normalized_input_value(
    const Storage* expanded_input,
    const void* norm_weight,
    std::uint32_t norm_type,
    const float* inverse_rms,
    std::uint32_t row,
    std::uint32_t hidden_size,
    std::uint32_t hc_count,
    std::uint32_t stream,
    std::uint32_t feature) {
    const auto expanded_width =
        hidden_size * hc_count;
    const auto local_index =
        stream * hidden_size +
        feature;
    const auto source_index =
        static_cast<std::uint64_t>(
            row) *
            expanded_width +
        local_index;
    const float source =
        load_activation(
            expanded_input,
            source_index);
    const float weight =
        load_norm_weight(
            norm_weight,
            norm_type,
            local_index);
    return
        source *
        inverse_rms[stream] *
        weight;
}

__device__ float sigmoid(
    float value) {
    return
        1.0F /
        (1.0F +
         expf(-value));
}

template <typename Storage>
__global__ __launch_bounds__(kThreads)
void hyper_mix_kernel(
    const Storage* expanded_input,
    const void* norm_weight,
    std::uint32_t norm_type,
    const void* down_weight,
    std::uint32_t down_type,
    const void* up_weight,
    std::uint32_t up_type,
    const void* inject_weight,
    std::uint32_t inject_type,
    bool produce_injection,
    Storage* mixed_output,
    float* injection_output,
    std::uint32_t hidden_size,
    std::uint32_t hc_count,
    std::uint32_t lowrank,
    float epsilon) {
    const auto row =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto lane =
        static_cast<std::uint32_t>(
            threadIdx.x);
    const auto expanded_width =
        hidden_size * hc_count;

    __shared__ float reduction[
        kMaxHc * kThreads];
    __shared__ float inverse_rms[
        kMaxHc];
    __shared__ float lowrank_values[
        kMaxLowrank];

    float local_square[kMaxHc] = {};
    for (std::uint32_t feature = lane;
         feature < hidden_size;
         feature += blockDim.x) {
        for (std::uint32_t stream = 0;
             stream < hc_count;
             ++stream) {
            const auto index =
                static_cast<std::uint64_t>(
                    row) *
                    expanded_width +
                static_cast<std::uint64_t>(
                    stream) *
                    hidden_size +
                feature;
            const float value =
                load_activation(
                    expanded_input,
                    index);
            local_square[stream] +=
                value * value;
        }
    }

    for (std::uint32_t stream = 0;
         stream < hc_count;
         ++stream) {
        reduction[
            stream * kThreads +
            lane] =
            local_square[stream];
    }
    __syncthreads();

    for (std::uint32_t stride =
             kThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            for (std::uint32_t stream = 0;
                 stream < hc_count;
                 ++stream) {
                reduction[
                    stream * kThreads +
                    lane] +=
                    reduction[
                        stream * kThreads +
                        lane +
                        stride];
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
        for (std::uint32_t stream = 0;
             stream < hc_count;
             ++stream) {
            inverse_rms[stream] =
                rsqrtf(
                    reduction[
                        stream * kThreads] /
                        static_cast<float>(
                            hidden_size) +
                    epsilon);
        }
    }
    __syncthreads();

    for (std::uint32_t output = lane;
         output < lowrank;
         output += blockDim.x) {
        float sum = 0.0F;
        for (std::uint32_t stream = 0;
             stream < hc_count;
             ++stream) {
            for (std::uint32_t feature = 0;
                 feature < hidden_size;
                 ++feature) {
                const auto input_index =
                    stream *
                        hidden_size +
                    feature;
                sum +=
                    normalized_input_value(
                        expanded_input,
                        norm_weight,
                        norm_type,
                        inverse_rms,
                        row,
                        hidden_size,
                        hc_count,
                        stream,
                        feature) *
                    load_linear_weight(
                        down_weight,
                        down_type,
                        expanded_width,
                        output,
                        input_index);
            }
        }
        const float scaled =
            sum /
            static_cast<float>(
                hc_count);
        lowrank_values[output] =
            scaled *
            sigmoid(scaled);
    }
    __syncthreads();

    if (produce_injection &&
        lane < hc_count) {
        float sum = 0.0F;
        for (std::uint32_t stream = 0;
             stream < hc_count;
             ++stream) {
            for (std::uint32_t feature = 0;
                 feature < hidden_size;
                 ++feature) {
                const auto input_index =
                    stream *
                        hidden_size +
                    feature;
                sum +=
                    normalized_input_value(
                        expanded_input,
                        norm_weight,
                        norm_type,
                        inverse_rms,
                        row,
                        hidden_size,
                        hc_count,
                        stream,
                        feature) *
                    load_linear_weight(
                        inject_weight,
                        inject_type,
                        expanded_width,
                        lane,
                        input_index);
            }
        }
        injection_output[
            static_cast<std::uint64_t>(
                row) *
                hc_count +
            lane] =
            2.0F *
            sigmoid(
                sum /
                static_cast<float>(
                    hc_count));
    }

    for (std::uint32_t feature = lane;
         feature < hidden_size;
         feature += blockDim.x) {
        float mixed = 0.0F;
        for (std::uint32_t stream = 0;
             stream < hc_count;
             ++stream) {
            const auto output_index =
                stream *
                    hidden_size +
                feature;
            float up = 0.0F;
            for (std::uint32_t inner = 0;
                 inner < lowrank;
                 ++inner) {
                up +=
                    lowrank_values[inner] *
                    load_linear_weight(
                        up_weight,
                        up_type,
                        lowrank,
                        output_index,
                        inner);
            }
            mixed +=
                sigmoid(up) *
                normalized_input_value(
                    expanded_input,
                    norm_weight,
                    norm_type,
                    inverse_rms,
                    row,
                    hidden_size,
                    hc_count,
                    stream,
                    feature);
        }

        store_activation(
            &mixed_output[
                static_cast<
                    std::uint64_t>(
                    row) *
                    hidden_size +
                feature],
            mixed /
                static_cast<float>(
                    hc_count));
    }
}

bool supported_linear_type(
    std::uint32_t ggml_type) noexcept {
    return ggml_type == 0 ||
           ggml_type == 1 ||
           ggml_type == 8 ||
           ggml_type == 23 ||
           ggml_type == 30;
}

// Preserve the legacy per-output accumulation order while distributing the
// independent projection outputs across workgroups. Normalized input is
// materialized once in the admitted, generation-owned execution scratch.
template <typename Storage>
__global__ void prepare_hyper_norm_kernel(
    const Storage* input, const void* norm, std::uint32_t norm_type,
    std::uint32_t hidden, std::uint32_t streams, float epsilon, float* normalized) {
    const auto stream = blockIdx.x;
    const auto row = blockIdx.y;
    const auto lane = threadIdx.x;
    const auto width = hidden * streams;
    const auto base = static_cast<std::uint64_t>(row) * width + stream * hidden;
    __shared__ float reduction[kThreads];
    __shared__ float inverse;
    float sum = 0.0F;
    for (std::uint32_t feature = lane; feature < hidden; feature += kThreads) {
        const float value = load_activation(input, base + feature);
        sum = qpf2k_fmac(sum, value, value);
    }
    reduction[lane] = sum;
    __syncthreads();
    for (std::uint32_t stride = kThreads / 2U; stride > 0; stride >>= 1U) {
        if (lane < stride) reduction[lane] = qpf2k_add(reduction[lane + stride], reduction[lane]);
        __syncthreads();
    }
    if (lane == 0) inverse = rsqrtf(reduction[0] / static_cast<float>(hidden) + epsilon);
    __syncthreads();
    for (std::uint32_t feature = lane; feature < hidden; feature += kThreads)
        normalized[base + feature] = qpf2k_mul(qpf2k_scale<Storage>(inverse, load_activation(input, base + feature)),
            load_norm_weight(norm, norm_type, stream * hidden + feature));
}

__global__ void parallel_hyper_lowrank_kernel(
    const float* normalized, const void* down, std::uint32_t kind,
    std::uint32_t width, std::uint32_t streams, std::uint32_t lowrank, float* low) {
    const auto output = blockIdx.x * blockDim.x + threadIdx.x;
    if (output >= lowrank) return;
    const auto row = blockIdx.y;
    float sum = 0.0F;
    for (std::uint32_t feature = 0; feature < width; ++feature)
        sum += normalized[static_cast<std::uint64_t>(row) * width + feature] *
            load_linear_weight(down, kind, width, output, feature);
    const float scaled = sum / static_cast<float>(streams);
    low[static_cast<std::uint64_t>(row) * lowrank + output] = scaled * sigmoid(scaled);
}

__global__ void parallel_hyper_injection_kernel(
    const float* normalized, const void* inject, std::uint32_t kind,
    std::uint32_t width, std::uint32_t streams, float* output) {
    const auto stream = threadIdx.x;
    if (stream >= streams) return;
    const auto row = blockIdx.x;
    float sum = 0.0F;
    for (std::uint32_t feature = 0; feature < width; ++feature)
        sum += normalized[static_cast<std::uint64_t>(row) * width + feature] *
            load_linear_weight(inject, kind, width, stream, feature);
    output[static_cast<std::uint64_t>(row) * streams + stream] =
        2.0F * sigmoid(sum / static_cast<float>(streams));
}

template <typename Storage>
__global__ void parallel_hyper_output_kernel(
    const float* normalized, const float* low, const void* up, std::uint32_t kind,
    std::uint32_t hidden, std::uint32_t streams, std::uint32_t lowrank, Storage* output) {
    const auto feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= hidden) return;
    const auto row = blockIdx.y;
    const auto width = hidden * streams;
    float mixed = 0.0F;
    for (std::uint32_t stream = 0; stream < streams; ++stream) {
        const auto index = stream * hidden + feature;
        float gate = 0.0F;
        for (std::uint32_t inner = 0; inner < lowrank; ++inner)
            gate += low[static_cast<std::uint64_t>(row) * lowrank + inner] *
                load_linear_weight(up, kind, lowrank, index, inner);
        mixed += normalized[static_cast<std::uint64_t>(row) * width + index] * sigmoid(gate);
    }
    store_activation(&output[static_cast<std::uint64_t>(row) * hidden + feature],
                     mixed / static_cast<float>(streams));
}

// Loader lanes fetch contiguous features of each coefficient row. Only the
// first eight lanes own accumulation chains; neither the feature order nor
// contraction expression changes. Padding avoids adjacent row bank aliasing.
// Every lane participates in both barriers, including partial output tiles.
template <bool Injection>
__global__ void staged_hyper_projection_kernel(
    const float* normalized, const void* weights, std::uint32_t kind,
    std::uint32_t width, std::uint32_t streams, std::uint32_t outputs, float* output) {
    __shared__ float values[kStagedThreads];
    __shared__ float coefficients[kStagedOutputs][kStagedPitch];
    const auto lane = threadIdx.x;
    const auto first_output = blockIdx.x * kStagedOutputs;
    const auto row = blockIdx.y;
    const auto chain = first_output + lane;
    const bool active = lane < kStagedOutputs && chain < outputs;
    float sum = 0.0F;
    for (std::uint32_t tile = 0; tile < width; tile += kStagedThreads) {
        const auto remaining = width - tile;
        const auto count = remaining < kStagedThreads ? remaining : kStagedThreads;
        if (lane < count) {
            values[lane] = normalized[static_cast<std::uint64_t>(row) * width + tile + lane];
            for (std::uint32_t local = 0; local < kStagedOutputs; ++local) {
                if (first_output + local < outputs)
                    coefficients[local][lane] = load_linear_weight(
                        weights, kind, width, first_output + local, tile + lane);
            }
        }
        __syncthreads();
        if (active)
            for (std::uint32_t inner = 0; inner < count; ++inner)
                sum += values[inner] * coefficients[lane][inner];
        // Consumers in other waves must finish before any loader refills LDS.
        __syncthreads();
    }
    if (active) {
        const float scaled = sum / static_cast<float>(streams);
        output[static_cast<std::uint64_t>(row) * outputs + chain] =
            Injection ? 2.0F * sigmoid(scaled) : scaled * sigmoid(scaled);
    }
}

// The injection role is uniform for an entire workgroup. Its tile and chain
// arithmetic match v4; the extra workgroups share the lowrank launch/stream.
__global__ void fused_staged_hyper_projection_kernel(
    const float* normalized, const void* down, std::uint32_t down_kind,
    std::uint32_t width, std::uint32_t streams, std::uint32_t lowrank, float* low,
    const void* inject, std::uint32_t inject_kind, float* injection_output,
    std::uint32_t lowrank_blocks) {
    __shared__ float values[kStagedThreads];
    __shared__ float coefficients[kStagedOutputs][kStagedPitch];
    const auto lane = threadIdx.x;
    const bool injection = blockIdx.x >= lowrank_blocks;
    const auto role_block = injection ? blockIdx.x - lowrank_blocks : blockIdx.x;
    const auto first_output = role_block * kStagedOutputs;
    const auto row = blockIdx.y;
    const auto chain = first_output + lane;
    const auto outputs = injection ? streams : lowrank;
    const auto* weights = injection ? inject : down;
    const auto kind = injection ? inject_kind : down_kind;
    auto* output = injection ? injection_output : low;
    const bool active = lane < kStagedOutputs && chain < outputs;
    float sum = 0.0F;
    for (std::uint32_t tile = 0; tile < width; tile += kStagedThreads) {
        const auto remaining = width - tile;
        const auto count = remaining < kStagedThreads ? remaining : kStagedThreads;
        if (lane < count) {
            values[lane] = normalized[static_cast<std::uint64_t>(row) * width + tile + lane];
            for (std::uint32_t local = 0; local < kStagedOutputs; ++local) {
                if (first_output + local < outputs)
                    coefficients[local][lane] = load_linear_weight(
                        weights, kind, width, first_output + local, tile + lane);
            }
        }
        __syncthreads();
        if (active)
            for (std::uint32_t inner = 0; inner < count; ++inner)
                sum += values[inner] * coefficients[lane][inner];
        __syncthreads();
    }
    if (active) {
        const float scaled = sum / static_cast<float>(streams);
        output[static_cast<std::uint64_t>(row) * outputs + chain] =
            injection ? 2.0F * sigmoid(scaled) : scaled * sigmoid(scaled);
    }
}

template <std::uint32_t Kind>
__device__ float typed_projection_weight(const void* weights, std::uint32_t width,
    std::uint32_t output, std::uint32_t feature) {
    static_assert(Kind == 8U || Kind == 0U);
    if constexpr (Kind == 8U) {
        const auto& block = static_cast<const Q8Block*>(weights)[
            static_cast<std::uint64_t>(output) * (width / 32U) + feature / 32U];
        return __half2float(block.scale) * static_cast<float>(block.values[feature % 32U]);
    } else {
        return static_cast<const float*>(weights)[static_cast<std::uint64_t>(output) * width + feature];
    }
}

template <std::uint32_t Kind, bool Injection>
__device__ void typed_projection_role(const float* normalized, const void* weights,
    std::uint32_t width, std::uint32_t streams, std::uint32_t outputs,
    std::uint32_t role_block, float* output, float* values,
    float (*coefficients)[kStagedPitch]) {
    const auto lane = threadIdx.x;
    const auto first_output = role_block * kStagedOutputs;
    const auto row = blockIdx.y;
    const auto chain = first_output + lane;
    const bool active = lane < kStagedOutputs && chain < outputs;
    float sum = 0.0F;
    for (std::uint32_t tile = 0; tile < width; tile += kStagedThreads) {
        const auto remaining = width - tile;
        const auto count = remaining < kStagedThreads ? remaining : kStagedThreads;
        if (lane < count) {
            values[lane] = normalized[static_cast<std::uint64_t>(row) * width + tile + lane];
            for (std::uint32_t local = 0; local < kStagedOutputs; ++local) {
                if (first_output + local < outputs)
                    coefficients[local][lane] = typed_projection_weight<Kind>(
                        weights, width, first_output + local, tile + lane);
            }
        }
        __syncthreads();
        if (active)
            for (std::uint32_t inner = 0; inner < count; ++inner)
                sum += values[inner] * coefficients[lane][inner];
        __syncthreads();
    }
    if (active) {
        const float scaled = sum / static_cast<float>(streams);
        output[static_cast<std::uint64_t>(row) * outputs + chain] =
            Injection ? 2.0F * sigmoid(scaled) : scaled * sigmoid(scaled);
    }
}

__global__ void typed_fused_hyper_projection_kernel(
    const float* normalized, const void* down, std::uint32_t width,
    std::uint32_t streams, std::uint32_t lowrank, float* low,
    const void* inject, float* injection_output, std::uint32_t lowrank_blocks) {
    __shared__ float values[kStagedThreads];
    __shared__ float coefficients[kStagedOutputs][kStagedPitch];
    // Role is block-uniform: every lane enters one complete producer/consumer loop.
    if (blockIdx.x < lowrank_blocks)
        typed_projection_role<8U, false>(normalized, down, width, streams, lowrank,
            blockIdx.x, low, values, coefficients);
    else
        typed_projection_role<0U, true>(normalized, inject, width, streams, streams,
            blockIdx.x - lowrank_blocks, injection_output, values, coefficients);
}

// Compile/ISA probe: all producer loads precede this producer's LDS stores.
// Full rows have no per-row control flow; partial rows reread an in-bounds row.
template <std::uint32_t Kind, bool Full>
__device__ void typed_batched_produce(const float* normalized, const void* weights,
    std::uint32_t width, std::uint32_t first_output, std::uint32_t valid,
    std::uint32_t tile, float* values, float (*coefficients)[kStagedPitch]) {
    const auto lane = threadIdx.x;
    const float value = normalized[static_cast<std::uint64_t>(blockIdx.y) * width + tile + lane];
    float prefetched[kStagedOutputs];
#pragma unroll
    for (std::uint32_t local = 0; local < kStagedOutputs; ++local) {
        const auto source = (Full || local < valid) ? local : valid - 1U;
        prefetched[local] = typed_projection_weight<Kind>(
            weights, width, first_output + source, tile + lane);
    }
    values[lane] = value;
#pragma unroll
    for (std::uint32_t local = 0; local < kStagedOutputs; ++local) {
        // Every destination is a valid row. Repeated last-row stores carry
        // the identical clamped coefficient and finish before the barrier.
        const auto destination = (Full || local < valid) ? local : valid - 1U;
        coefficients[destination][lane] = prefetched[local];
    }
}

template <std::uint32_t Kind, bool Injection, bool Full>
__device__ void typed_batched_projection_role(const float* normalized, const void* weights,
    std::uint32_t width, std::uint32_t streams, std::uint32_t outputs,
    std::uint32_t role_block, float* output, float* values,
    float (*coefficients)[kStagedPitch]) {
    const auto lane = threadIdx.x;
    const auto first_output = role_block * kStagedOutputs;
    const auto row = blockIdx.y;
    // The ceil(outputs/8) grid guarantees first_output < outputs and valid >= 1.
    const auto rows_left = outputs - first_output;
    const auto valid = rows_left < kStagedOutputs ? rows_left : kStagedOutputs;
    const auto chain = first_output + lane;
    const bool active = lane < kStagedOutputs && chain < outputs;
    float sum = 0.0F;
    for (std::uint32_t tile = 0; tile < width; tile += kStagedThreads) {
        const auto remaining = width - tile;
        const auto count = remaining < kStagedThreads ? remaining : kStagedThreads;
        if (lane < count) {
            typed_batched_produce<Kind, Full>(normalized, weights, width, first_output, valid,
                tile, values, coefficients);
        }
        __syncthreads();
        if (active)
            for (std::uint32_t inner = 0; inner < count; ++inner)
                sum += values[inner] * coefficients[lane][inner];
        __syncthreads();
    }
    if (active) {
        const float scaled = sum / static_cast<float>(streams);
        output[static_cast<std::uint64_t>(row) * outputs + chain] =
            Injection ? 2.0F * sigmoid(scaled) : scaled * sigmoid(scaled);
    }
}

__global__ void batched_typed_hyper_projection_kernel(
    const float* normalized, const void* down, std::uint32_t width,
    std::uint32_t streams, std::uint32_t lowrank, float* low,
    const void* inject, float* injection_output, std::uint32_t lowrank_blocks) {
    __shared__ float values[kStagedThreads];
    __shared__ float coefficients[kStagedOutputs][kStagedPitch];
    // Both choices are block-uniform and precede each complete tile loop.
    if (blockIdx.x < lowrank_blocks) {
        const auto rows_left = lowrank - blockIdx.x * kStagedOutputs;
        if (rows_left >= kStagedOutputs)
            typed_batched_projection_role<8U, false, true>(normalized, down, width, streams, lowrank,
                blockIdx.x, low, values, coefficients);
        else
            typed_batched_projection_role<8U, false, false>(normalized, down, width, streams, lowrank,
                blockIdx.x, low, values, coefficients);
    } else {
        const auto role_block = blockIdx.x - lowrank_blocks;
        const auto rows_left = streams - role_block * kStagedOutputs;
        if (rows_left >= kStagedOutputs)
            typed_batched_projection_role<0U, true, true>(normalized, inject, width, streams, streams,
                role_block, injection_output, values, coefficients);
        else
            typed_batched_projection_role<0U, true, false>(normalized, inject, width, streams, streams,
                role_block, injection_output, values, coefficients);
    }
}

// Diagnostic parallel projection: independent fixed reduction identity. Each
// wave owns a disjoint 640-feature tile, and registers reuse its input across R
// output rows. Existing ordered kernels and their production dispatch stay intact.
template <unsigned R, typename Acc, bool Injection, bool Q6 = false>
__device__ __forceinline__ void splitk_hyper_lane_dots(
    const float* normalized, const void* weights, unsigned first, unsigned valid,
    unsigned tile, unsigned lane, Acc (&sums)[R]) {
    float4 x[5];
#pragma unroll
    for (unsigned s=0;s<5U;++s)
        x[s]=*reinterpret_cast<const float4*>(normalized+tile*640U+s*128U+lane*4U);
    if constexpr (Injection) {
        float4 w[R][5];
#pragma unroll
        for (unsigned r=0;r<R;++r) if(r<valid) {
#pragma unroll
            for(unsigned s=0;s<5U;++s)
                w[r][s]=*reinterpret_cast<const float4*>(
                    static_cast<const float*>(weights)+
                    static_cast<std::uint64_t>(first+r)*10240U+tile*640U+s*128U+lane*4U);
        }
#pragma unroll
        for(unsigned r=0;r<R;++r) {
            sums[r]=Acc{0};
            if(r<valid) {
#pragma unroll
                for(unsigned s=0;s<5U;++s) {
                    if constexpr(sizeof(Acc)==sizeof(float)) {
                        sums[r]=fmaf(x[s].x,w[r][s].x,sums[r]);
                        sums[r]=fmaf(x[s].y,w[r][s].y,sums[r]);
                        sums[r]=fmaf(x[s].z,w[r][s].z,sums[r]);
                        sums[r]=fmaf(x[s].w,w[r][s].w,sums[r]);
                    } else {
                        sums[r]=fma(static_cast<double>(x[s].x),static_cast<double>(w[r][s].x),sums[r]);
                        sums[r]=fma(static_cast<double>(x[s].y),static_cast<double>(w[r][s].y),sums[r]);
                        sums[r]=fma(static_cast<double>(x[s].z),static_cast<double>(w[r][s].z),sums[r]);
                        sums[r]=fma(static_cast<double>(x[s].w),static_cast<double>(w[r][s].w),sums[r]);
                    }
                }
            }
        }
    } else {
        unsigned packed[R][5];
        float scales[R][5];
        if constexpr (Q6) {
            // Q6P32 copy of the down tensor: the same codes and scales, decoded once all loads are issued.
            unsigned qs[R][5], qh[R][5];
            unsigned short bits[R][5];
            const unsigned m=lane%8U;
#pragma unroll
            for(unsigned r=0;r<R;++r) if(r<valid) {
#pragma unroll
                for(unsigned s=0;s<5U;++s) {
                    const unsigned b=tile*20U+4U*s+lane/8U;
                    const auto o=q6p::chunk_offsets(10240U,b/8U);
                    const auto* row=static_cast<const unsigned char*>(weights)+static_cast<std::uint64_t>(first+r)*q6p::row_bytes(10240U);
                    qs[r][s]=*reinterpret_cast<const unsigned*>(row+o.qs+16U*(b%8U)+4U*(m/2U));
                    qh[r][s]=*reinterpret_cast<const unsigned*>(row+o.qh+8U*(b%8U)+4U*(m/4U));
                    bits[r][s]=*reinterpret_cast<const unsigned short*>(row+o.d+2U*(b%8U));
                }
            }
#pragma unroll
            for(unsigned r=0;r<R;++r) if(r<valid) {
#pragma unroll
                for(unsigned s=0;s<5U;++s) {
                    scales[r][s]=__half2float(__ushort_as_half(bits[r][s]));
                    packed[r][s]=q6p::codes4(qs[r][s],qh[r][s],m);
                }
            }
        } else
#pragma unroll
        for(unsigned r=0;r<R;++r) if(r<valid) {
#pragma unroll
            for(unsigned s=0;s<5U;++s) {
                const auto* block=static_cast<const unsigned char*>(weights)+
                    static_cast<std::uint64_t>(first+r)*320U*34U+
                    static_cast<std::uint64_t>(tile*20U+4U*s+lane/8U)*34U;
                // Original 34-byte blocks are only 2-aligned. Never issue an
                // incorrectly aligned 32-bit load for this packed code quartet.
                const auto bits=*reinterpret_cast<const unsigned short*>(block);
                const auto* codes=reinterpret_cast<const unsigned short*>(block+2U+4U*(lane%8U));
                scales[r][s]=__half2float(__ushort_as_half(bits));
                packed[r][s]=static_cast<unsigned>(codes[0])|
                    (static_cast<unsigned>(codes[1])<<16U);
            }
        }
#pragma unroll
        for(unsigned r=0;r<R;++r) {
            sums[r]=Acc{0};
            if(r<valid) {
#pragma unroll
                for(unsigned s=0;s<5U;++s) {
                    const float inputs[4]={x[s].x,x[s].y,x[s].z,x[s].w};
#pragma unroll
                    for(unsigned j=0;j<4U;++j) {
                        const unsigned byte=(packed[r][s]>>(j*8U))&255U;
                        const int code=byte<128U?static_cast<int>(byte):static_cast<int>(byte)-256;
                        const float coefficient=scales[r][s]*static_cast<float>(code);
                        if constexpr(sizeof(Acc)==sizeof(float))
                            sums[r]=fmaf(inputs[j],coefficient,sums[r]);
                        else
                            sums[r]=fma(static_cast<double>(inputs[j]),static_cast<double>(coefficient),sums[r]);
                    }
                }
            }
        }
    }
#pragma unroll
    for(unsigned r=0;r<R;++r) if(r<valid) {
#pragma unroll
        for(unsigned offset=16U;offset!=0U;offset/=2U)
            sums[r]=sums[r]+__shfl_xor(sums[r],static_cast<int>(offset),32);
    }
}

template <unsigned R, typename Acc, bool Injection>
__device__ __forceinline__ void deferred_hyper_lane_dots(
    const std::uint16_t* raw, const float* norm, const void* weights, unsigned first, unsigned valid,
    unsigned tile, unsigned lane, Acc (&sums)[R]) {
    float4 x[5];
#pragma unroll
    for (unsigned s=0;s<5U;++s) {
        const unsigned feature=tile*640U+s*128U+lane*4U;
        x[s]=make_float4(__fmul_rn(load_activation(raw,feature),norm[feature]),
            __fmul_rn(load_activation(raw,feature+1U),norm[feature+1U]),
            __fmul_rn(load_activation(raw,feature+2U),norm[feature+2U]),
            __fmul_rn(load_activation(raw,feature+3U),norm[feature+3U]));
    }
    if constexpr (Injection) {
        float4 w[R][5];
#pragma unroll
        for (unsigned r=0;r<R;++r) if(r<valid) {
#pragma unroll
            for(unsigned s=0;s<5U;++s)
                w[r][s]=*reinterpret_cast<const float4*>(
                    static_cast<const float*>(weights)+
                    static_cast<std::uint64_t>(first+r)*10240U+tile*640U+s*128U+lane*4U);
        }
#pragma unroll
        for(unsigned r=0;r<R;++r) {
            sums[r]=Acc{0};
            if(r<valid) {
#pragma unroll
                for(unsigned s=0;s<5U;++s) {
                    if constexpr(sizeof(Acc)==sizeof(float)) {
                        sums[r]=fmaf(x[s].x,w[r][s].x,sums[r]);
                        sums[r]=fmaf(x[s].y,w[r][s].y,sums[r]);
                        sums[r]=fmaf(x[s].z,w[r][s].z,sums[r]);
                        sums[r]=fmaf(x[s].w,w[r][s].w,sums[r]);
                    } else {
                        sums[r]=fma(static_cast<double>(x[s].x),static_cast<double>(w[r][s].x),sums[r]);
                        sums[r]=fma(static_cast<double>(x[s].y),static_cast<double>(w[r][s].y),sums[r]);
                        sums[r]=fma(static_cast<double>(x[s].z),static_cast<double>(w[r][s].z),sums[r]);
                        sums[r]=fma(static_cast<double>(x[s].w),static_cast<double>(w[r][s].w),sums[r]);
                    }
                }
            }
        }
    } else {
        unsigned packed[R][5];
        float scales[R][5];
#pragma unroll
        for(unsigned r=0;r<R;++r) if(r<valid) {
#pragma unroll
            for(unsigned s=0;s<5U;++s) {
                const auto* block=static_cast<const unsigned char*>(weights)+
                    static_cast<std::uint64_t>(first+r)*320U*34U+
                    static_cast<std::uint64_t>(tile*20U+4U*s+lane/8U)*34U;
                // Original 34-byte blocks are only 2-aligned. Never issue an
                // incorrectly aligned 32-bit load for this packed code quartet.
                const auto bits=*reinterpret_cast<const unsigned short*>(block);
                const auto* codes=reinterpret_cast<const unsigned short*>(block+2U+4U*(lane%8U));
                scales[r][s]=__half2float(__ushort_as_half(bits));
                packed[r][s]=static_cast<unsigned>(codes[0])|
                    (static_cast<unsigned>(codes[1])<<16U);
            }
        }
#pragma unroll
        for(unsigned r=0;r<R;++r) {
            sums[r]=Acc{0};
            if(r<valid) {
#pragma unroll
                for(unsigned s=0;s<5U;++s) {
                    const float inputs[4]={x[s].x,x[s].y,x[s].z,x[s].w};
#pragma unroll
                    for(unsigned j=0;j<4U;++j) {
                        const unsigned byte=(packed[r][s]>>(j*8U))&255U;
                        const int code=byte<128U?static_cast<int>(byte):static_cast<int>(byte)-256;
                        const float coefficient=scales[r][s]*static_cast<float>(code);
                        if constexpr(sizeof(Acc)==sizeof(float))
                            sums[r]=fmaf(inputs[j],coefficient,sums[r]);
                        else
                            sums[r]=fma(static_cast<double>(inputs[j]),static_cast<double>(coefficient),sums[r]);
                    }
                }
            }
        }
    }
#pragma unroll
    for(unsigned r=0;r<R;++r) if(r<valid) {
#pragma unroll
        for(unsigned offset=16U;offset!=0U;offset/=2U)
            sums[r]=sums[r]+__shfl_xor(sums[r],static_cast<int>(offset),32);
    }
}

// Deferred RMS: each wave owns 640 raw features. The four stream inverses
// are reproduced in every projection block; only block zero publishes them.
template <unsigned R>
__global__ __launch_bounds__(512) void deferred_hyper_projection_kernel(
    const std::uint16_t* raw,const float* norm,const void* down,unsigned rank,
    float epsilon,float* inverse_out,float* low,const void* inject,float* injection,unsigned low_blocks,
    float* inverse_audit,float* scaled_audit) {
    __shared__ float red[16][R];
    __shared__ float square[16];
    __shared__ float inverse[4];
    const unsigned lane=threadIdx.x%32U,tile=threadIdx.x/32U;
    const auto* input=raw+static_cast<std::uint64_t>(blockIdx.y)*10240U;
    float sum=0.0F;
#pragma unroll
    for(unsigned part=0;part<5U;++part) {
#pragma unroll
        for(unsigned j=0;j<4U;++j) {
            const float value=load_activation(input,tile*640U+part*128U+lane*4U+j);
            sum=fmaf(value,value,sum);
        }
    }
#pragma unroll
    for(unsigned offset=16U;offset!=0U;offset/=2U)
        sum=sum+__shfl_xor(sum,static_cast<int>(offset),32);
    if(lane==0U)square[tile]=sum;
    __syncthreads();
    if(threadIdx.x<4U) {
        float total=0.0F;
#pragma unroll
        for(unsigned k=0;k<4U;++k)total=total+square[threadIdx.x*4U+k];
        inverse[threadIdx.x]=rsqrtf(total/2560.0F+epsilon);
        if(blockIdx.x==0U)inverse_out[static_cast<std::uint64_t>(blockIdx.y)*4U+threadIdx.x]=inverse[threadIdx.x];
        if(inverse_audit)inverse_audit[static_cast<std::uint64_t>(blockIdx.x)*4U+threadIdx.x]=inverse[threadIdx.x];
    }
    __syncthreads();
    const bool is_inject=blockIdx.x>=low_blocks;
    const unsigned first=(is_inject?blockIdx.x-low_blocks:blockIdx.x)*R;
    const unsigned remaining=(is_inject?4U:rank)-first;
    const unsigned valid=remaining<R?remaining:R;
    float sums[R];
    if(is_inject)deferred_hyper_lane_dots<R,float,true>(input,norm,inject,first,valid,tile,lane,sums);
    else deferred_hyper_lane_dots<R,float,false>(input,norm,down,first,valid,tile,lane,sums);
    if(lane==0U) {
#pragma unroll
        for(unsigned r=0;r<R;++r)if(r<valid)red[tile][r]=sums[r];
    }
    __syncthreads();
    if(threadIdx.x<valid) {
        float total=0.0F;
#pragma unroll
        for(unsigned k=0;k<16U;++k)total=fmaf(red[k][threadIdx.x],inverse[k/4U],total);
        const float scaled=total/4.0F;
        if(scaled_audit)scaled_audit[(is_inject?rank:0U)+first+threadIdx.x]=scaled;
        if(is_inject)injection[static_cast<std::uint64_t>(blockIdx.y)*4U+first+threadIdx.x]=2.0F*sigmoid(scaled);
        else low[static_cast<std::uint64_t>(blockIdx.y)*rank+first+threadIdx.x]=scaled*sigmoid(scaled);
    }
}

template <unsigned R, typename Acc, bool Q6 = false>
__global__ __launch_bounds__(512) void splitk_hyper_projection_kernel(
    const float* __restrict__ normalized,const void* __restrict__ down,unsigned width,unsigned streams,
    unsigned rank,float* __restrict__ low,const void* __restrict__ inject,float* __restrict__ injection,unsigned low_blocks) {
    __shared__ Acc red[16][R];
    const unsigned lane=threadIdx.x%32U,tile=threadIdx.x/32U;
    const bool is_inject=blockIdx.x>=low_blocks;
    const unsigned first=(is_inject?blockIdx.x-low_blocks:blockIdx.x)*R;
    const unsigned remaining=(is_inject?streams:rank)-first;
    const unsigned valid=remaining<R?remaining:R;
    const auto* input=normalized+static_cast<std::uint64_t>(blockIdx.y)*width;
    Acc sums[R];
    if(is_inject)splitk_hyper_lane_dots<R,Acc,true>(input,inject,first,valid,tile,lane,sums);
    else splitk_hyper_lane_dots<R,Acc,false,Q6>(input,down,first,valid,tile,lane,sums);
    if(lane==0U) {
#pragma unroll
        for(unsigned r=0;r<R;++r)if(r<valid)red[tile][r]=sums[r];
    }
    __syncthreads();
    if(threadIdx.x<valid) {
        Acc total=Acc{0};
#pragma unroll
        for(unsigned k=0;k<16U;++k)total=total+red[k][threadIdx.x];
        const float scaled=static_cast<float>(total)/static_cast<float>(streams);
        if(is_inject)injection[static_cast<std::uint64_t>(blockIdx.y)*streams+first+threadIdx.x]=2.0F*sigmoid(scaled);
        else low[static_cast<std::uint64_t>(blockIdx.y)*rank+first+threadIdx.x]=scaled*sigmoid(scaled);
    }
}

// Same arithmetic as P1, with a separate global-partial/finalize comparator.
__global__ __launch_bounds__(256) void splitk_hyper_partial_kernel(
    const float* normalized,const void* down,const float* inject,unsigned streams,
    unsigned rank,unsigned low_blocks,float* partial) {
    const unsigned lane=threadIdx.x%32U,warp=threadIdx.x/32U,tile=blockIdx.y;
    const bool is_inject=blockIdx.x>=low_blocks;
    const unsigned first=is_inject?warp:(blockIdx.x*8U+warp)*4U;
    const unsigned limit=is_inject?streams:rank;
    if(first>=limit)return; // Whole wave; this kernel has no block barrier.
    const unsigned valid=is_inject?1U:((limit-first)<4U?(limit-first):4U);
    const auto* input=normalized+static_cast<std::uint64_t>(blockIdx.z)*10240U;
    const unsigned pitch=rank+(inject?streams:0U);
    const auto base=(static_cast<std::uint64_t>(blockIdx.z)*16U+tile)*pitch;
    if(is_inject) {
        float sums[1];splitk_hyper_lane_dots<1,float,true>(input,inject,first,1U,tile,lane,sums);
        if(lane==0U)partial[base+rank+first]=sums[0];
    } else {
        float sums[4];splitk_hyper_lane_dots<4,float,false>(input,down,first,valid,tile,lane,sums);
        if(lane==0U) {
#pragma unroll
            for(unsigned r=0;r<4U;++r)if(r<valid)partial[base+first+r]=sums[r];
        }
    }
}

__global__ void splitk_hyper_finalize_kernel(
    const float* partial,unsigned streams,unsigned rank,unsigned pitch,float* low,float* injection) {
    const unsigned output=blockIdx.x*128U+threadIdx.x;
    if(output>=pitch)return;
    float total=0.0F;
#pragma unroll
    for(unsigned tile=0;tile<16U;++tile)
        total=total+partial[(static_cast<std::uint64_t>(blockIdx.y)*16U+tile)*pitch+output];
    const float scaled=total/static_cast<float>(streams);
    if(output<rank)low[static_cast<std::uint64_t>(blockIdx.y)*rank+output]=scaled*sigmoid(scaled);
    else injection[static_cast<std::uint64_t>(blockIdx.y)*streams+output-rank]=2.0F*sigmoid(scaled);
}

// Exact C3 V2 current-weight pipeline; all incumbent kernels stay separate.
template <std::uint32_t Kind> struct NextTileRaw;
// Keep scale/code in independent 32-bit carriers before the current chain.
// Actual overlap still requires a fresh G7 ISA audit, never this source comment.
template <> struct NextTileRaw<8U> { std::uint32_t scale_bits; std::int32_t code; };
template <> struct NextTileRaw<0U> { float value; };

template <std::uint32_t Kind>
__device__ __forceinline__ NextTileRaw<Kind> next_tile_load_raw(const void* weights,std::uint32_t width,
    std::uint32_t output,std::uint32_t feature) {
    if constexpr(Kind==8U){
        const auto& block=static_cast<const Q8Block*>(weights)[
            static_cast<std::uint64_t>(output)*(width/32U)+feature/32U];
        return {static_cast<std::uint32_t>(__half_as_ushort(block.scale)),
                static_cast<std::int32_t>(block.values[feature%32U])};
    }else return {static_cast<const float*>(weights)[
        static_cast<std::uint64_t>(output)*width+feature]};
}
template <std::uint32_t Kind>
__device__ __forceinline__ float next_tile_decode(NextTileRaw<Kind> raw) {
    if constexpr(Kind==8U) return __half2float(__ushort_as_half(
        static_cast<unsigned short>(raw.scale_bits)))*static_cast<float>(raw.code);
    else return raw.value;
}

template <std::uint32_t Kind,bool Full,std::uint32_t Threads,std::uint32_t Tile,
          std::uint32_t Outputs>
__device__ __forceinline__ void next_tile_prefetch(const float* normalized,const void* weights,
    std::uint32_t width,std::uint32_t first,std::uint32_t valid,std::uint32_t tile,
    float (&values)[Tile/Threads],NextTileRaw<Kind> (&raw)[Tile/Threads][Outputs]) {
    const auto lane=threadIdx.x;
#pragma unroll
    for(std::uint32_t column=0;column<Tile/Threads;++column){
        const auto feature=tile+lane+column*Threads;
        if(feature<width){
            values[column]=normalized[static_cast<std::uint64_t>(blockIdx.y)*width+feature];
#pragma unroll
            for(std::uint32_t local=0;local<Outputs;++local)
                raw[column][local]=next_tile_load_raw<Kind>(weights,width,
                    first+((Full || local<valid)?local:valid-1U),feature);
        }
    }
}

template <std::uint32_t Kind,bool Injection,bool Full,std::uint32_t Threads,
          std::uint32_t Tile,std::uint32_t Outputs,bool Pipeline>
__device__ __forceinline__ void next_tile_projection_role(const float* normalized,const void* weights,
    std::uint32_t width,std::uint32_t streams,std::uint32_t outputs,
    std::uint32_t role_block,float* output,float* values,float (*coefficients)[Tile+1U]) {
    static_assert(Threads%32U==0U && Tile%Threads==0U && Tile/Threads<=2U);
    static_assert(Outputs>=8U && Outputs<=32U && Outputs%4U==0U);
    const auto lane=threadIdx.x;
    const auto first=role_block*Outputs;
    const auto left=outputs-first;
    const auto valid=left<Outputs?left:Outputs;
    const bool active=lane<Outputs && (Full || lane<valid);
    float sum=0.0F;
    [[maybe_unused]] float pending_values[Tile/Threads];
    [[maybe_unused]] NextTileRaw<Kind> pending[Tile/Threads][Outputs];
    if constexpr(Pipeline)
        next_tile_prefetch<Kind,Full,Threads,Tile,Outputs>(normalized,weights,width,first,valid,0,
            pending_values,pending);
    for(std::uint32_t tile=0;tile<width;tile+=Tile){
        const auto left_features=width-tile;
        const auto count=left_features<Tile?left_features:Tile;
        if constexpr(Pipeline){
#pragma unroll
            for(std::uint32_t column=0;column<Tile/Threads;++column){
                const auto feature=lane+column*Threads;
                if(feature<count){
                    values[feature]=pending_values[column];
#pragma unroll
                    for(std::uint32_t local=0;local<Outputs;++local)
                        coefficients[local][feature]=next_tile_decode<Kind>(pending[column][local]);
                }
            }
        }else{
#pragma unroll
            for(std::uint32_t column=0;column<Tile/Threads;++column){
                const auto feature=lane+column*Threads;
                if(feature<count){
                    const float value=normalized[static_cast<std::uint64_t>(blockIdx.y)*width+tile+feature];
                    float prefetched[Outputs];
#pragma unroll
                    for(std::uint32_t local=0;local<Outputs;++local)
                        prefetched[local]=typed_projection_weight<Kind>(weights,width,
                            first+((Full || local<valid)?local:valid-1U),tile+feature);
                    values[feature]=value;
#pragma unroll
                    for(std::uint32_t local=0;local<Outputs;++local)
                        coefficients[local][feature]=prefetched[local];
                }
            }
        }
        __syncthreads(); // A: all staged values/coefficients are visible.
        // Raw next-tile operands are not decoded until the next store phase.
        // C++ placement is not proof of overlap: G7 requires actual ISA showing
        // outstanding VMEM across the current chain, without an early vmcnt.
        if constexpr(Pipeline)
            if(tile+Tile<width)
                next_tile_prefetch<Kind,Full,Threads,Tile,Outputs>(normalized,weights,width,first,valid,
                    tile+Tile,pending_values,pending);
        if(active)
            for(std::uint32_t inner=0;inner<count;++inner)
                sum+=values[inner]*coefficients[lane][inner];
        __syncthreads(); // B: no producer overwrites LDS before all reads end.
    }
    if(active){
        const float scaled=sum/static_cast<float>(streams);
        output[static_cast<std::uint64_t>(blockIdx.y)*outputs+first+lane]=
            Injection?2.0F*sigmoid(scaled):scaled*sigmoid(scaled);
    }
}

__global__ __launch_bounds__(128) void next_tile_hyper_projection_kernel(const float* normalized,const void* down,
    std::uint32_t width,std::uint32_t streams,std::uint32_t rank,float* low,
    const void* inject,float* injection,std::uint32_t low_blocks) {
    constexpr std::uint32_t Threads=128U,Tile=128U,Outputs=8U;
    constexpr bool Pipeline=true;
    __shared__ float values[Tile];
    __shared__ float coefficients[Outputs][Tile+1U];
    // Role selection is block-uniform. Injection retains eight output lanes.
    if(blockIdx.x<low_blocks){
        const auto left=rank-blockIdx.x*Outputs;
        if(left>=Outputs)
            next_tile_projection_role<8U,false,true,Threads,Tile,Outputs,Pipeline>(
                normalized,down,width,streams,rank,blockIdx.x,low,values,coefficients);
        else
            next_tile_projection_role<8U,false,false,Threads,Tile,Outputs,Pipeline>(
                normalized,down,width,streams,rank,blockIdx.x,low,values,coefficients);
    }else{
        const auto role_block=blockIdx.x-low_blocks;
        const auto left=streams-role_block*8U;
        if(left>=8U)
            next_tile_projection_role<0U,true,true,Threads,Tile,8U,Pipeline>(
                normalized,inject,width,streams,streams,role_block,injection,values,coefficients);
        else
            next_tile_projection_role<0U,true,false,Threads,Tile,8U,Pipeline>(
                normalized,inject,width,streams,streams,role_block,injection,values,coefficients);
    }
}


// Diagnostic exact projection prototype. No production launcher selects it.
// Each active lane owns one entire ascending dot chain; K is never partitioned.
__device__ __forceinline__ unsigned lane_chain_q8_byte(const uint4* words,unsigned byte){
    const auto v=words[byte/16U];const unsigned slot=(byte/4U)%4U;
    const unsigned word=slot==0U?v.x:slot==1U?v.y:slot==2U?v.z:v.w;
    return (word>>(8U*(byte%4U)))&255U;
}
template <unsigned OutputsPerWave,unsigned Prefetch>
__global__ __launch_bounds__(32) void lane_chain_hyper_projection_kernel(const float* normalized,const void* down,const float* inject,
    unsigned width,unsigned streams,unsigned rank,float* low,float* injection,unsigned low_blocks){
    static_assert(OutputsPerWave==8 || OutputsPerWave==16 || OutputsPerWave==32);
    static_assert(Prefetch==1 || Prefetch==2); // P4 needs 272 packed dwords alone.
    const unsigned lane=threadIdx.x,row=blockIdx.y;
    const bool is_inject=blockIdx.x>=low_blocks;
    const unsigned chain=(is_inject?blockIdx.x-low_blocks:blockIdx.x)*OutputsPerWave+lane;
    if(lane>=OutputsPerWave || chain>=(is_inject?streams:rank))return;
    float sum=0.0F;
    const auto input_row=static_cast<std::uint64_t>(row)*width;
    if(is_inject){
        const auto base=static_cast<std::uint64_t>(chain)*width;
        for(unsigned tile=0;tile<width;tile+=4U){
            const float4 w=*reinterpret_cast<const float4*>(inject+base+tile);
            sum+=normalized[input_row+tile]*w.x;
            sum+=normalized[input_row+tile+1U]*w.y;
            sum+=normalized[input_row+tile+2U]*w.z;
            sum+=normalized[input_row+tile+3U]*w.w;
        }
        const float scaled=sum/static_cast<float>(streams);
        injection[static_cast<std::uint64_t>(row)*streams+chain]=2.0F*sigmoid(scaled);
    }else{
        const auto* weight_row=static_cast<const unsigned char*>(down)+static_cast<std::uint64_t>(chain)*(width/32U)*34U;
        const unsigned groups=width/256U;
        for(unsigned first=0;first<groups;first+=Prefetch){
            uint4 packed[Prefetch][17];
#pragma unroll
            for(unsigned p=0;p<Prefetch;++p)if(first+p<groups){
#pragma unroll
                for(unsigned v=0;v<17U;++v)
                    packed[p][v]=*reinterpret_cast<const uint4*>(weight_row+static_cast<std::uint64_t>(first+p)*272U+v*16U);
            }
#pragma unroll
            for(unsigned p=0;p<Prefetch;++p)if(first+p<groups){
#pragma unroll
                for(unsigned b=0;b<8U;++b){
                    const auto bits=static_cast<unsigned short>(lane_chain_q8_byte(packed[p],b*34U)|(lane_chain_q8_byte(packed[p],b*34U+1U)<<8U));
                    const float scale=__half2float(__ushort_as_half(bits));
#pragma unroll
                    for(unsigned k=0;k<32U;++k){
                        const auto code=static_cast<signed char>(lane_chain_q8_byte(packed[p],b*34U+2U+k));
                        const float coefficient=scale*static_cast<float>(code);
                        sum+=normalized[input_row+static_cast<std::uint64_t>(first+p)*256U+b*32U+k]*coefficient;
                    }
                }
            }
        }
        const float scaled=sum/static_cast<float>(streams);
        low[static_cast<std::uint64_t>(row)*rank+chain]=scaled*sigmoid(scaled);
    }
}

template <typename Storage>
__global__ void staged_hyper_output_kernel(
    const float* normalized, const float* low, const void* up, std::uint32_t kind,
    std::uint32_t hidden, std::uint32_t streams, std::uint32_t lowrank, Storage* output) {
    __shared__ float values[kStagedThreads];
    __shared__ float coefficients[kStagedOutputs][kStagedPitch];
    const auto lane = threadIdx.x;
    const auto first_feature = blockIdx.x * kStagedOutputs;
    const auto feature = first_feature + lane;
    const auto row = blockIdx.y;
    const bool active = lane < kStagedOutputs && feature < hidden;
    const auto width = hidden * streams;
    float mixed = 0.0F;
    for (std::uint32_t stream = 0; stream < streams; ++stream) {
        float gate = 0.0F;
        for (std::uint32_t tile = 0; tile < lowrank; tile += kStagedThreads) {
            const auto remaining = lowrank - tile;
            const auto count = remaining < kStagedThreads ? remaining : kStagedThreads;
            if (lane < count) {
                values[lane] = low[static_cast<std::uint64_t>(row) * lowrank + tile + lane];
                for (std::uint32_t local = 0; local < kStagedOutputs; ++local) {
                    if (first_feature + local < hidden)
                        coefficients[local][lane] = load_linear_weight(
                            up, kind, lowrank, stream * hidden + first_feature + local, tile + lane);
                }
            }
            __syncthreads();
            if (active)
                for (std::uint32_t inner = 0; inner < count; ++inner)
                    gate += values[inner] * coefficients[lane][inner];
            __syncthreads();
        }
        if (active)
            mixed = qpf2k_fmac(mixed, sigmoid(gate), normalized[static_cast<std::uint64_t>(row) * width + stream * hidden + feature]);
    }
    if (active)
        store_activation(&output[static_cast<std::uint64_t>(row) * hidden + feature],
                         mixed / static_cast<float>(streams));
}


// Experimental output producer: typed Q8 coefficients, independent rows loaded
// before LDS stores. Arithmetic and stream order match the generic output path.
template <typename Storage, bool Full>
__device__ void q8_batched_hyper_output_role(
    const float* normalized, const float* low, const void* up,
    std::uint32_t hidden, std::uint32_t streams, std::uint32_t lowrank,
    Storage* output, float* values, float (*coefficients)[kStagedPitch]) {
    const auto lane = threadIdx.x;
    const auto first_feature = blockIdx.x * kStagedOutputs;
    const auto feature = first_feature + lane;
    const auto row = blockIdx.y;
    const bool active = lane < kStagedOutputs && feature < hidden;
    const auto rows_left = hidden - first_feature;
    const auto valid = rows_left < kStagedOutputs ? rows_left : kStagedOutputs;
    const auto width = hidden * streams;
    float mixed = 0.0F;
    for (std::uint32_t stream = 0; stream < streams; ++stream) {
        float gate = 0.0F;
        for (std::uint32_t tile = 0; tile < lowrank; tile += kStagedThreads) {
            const auto remaining = lowrank - tile;
            const auto count = remaining < kStagedThreads ? remaining : kStagedThreads;
            if (lane < count) {
                typed_batched_produce<8U, Full>(low, up, lowrank,
                    stream * hidden + first_feature, valid, tile, values, coefficients);
            }
            __syncthreads();
            if (active)
                for (std::uint32_t inner = 0; inner < count; ++inner)
                    gate += values[inner] * coefficients[lane][inner];
            __syncthreads();
        }
        if (active)
            mixed = qpf2k_fmac(mixed, sigmoid(gate), normalized[static_cast<std::uint64_t>(row) * width + stream * hidden + feature]);
    }
    if (active)
        store_activation(&output[static_cast<std::uint64_t>(row) * hidden + feature],
                         mixed / static_cast<float>(streams));
}

template <typename Storage>
__global__ void q8_batched_hyper_output_kernel(
    const float* normalized, const float* low, const void* up,
    std::uint32_t hidden, std::uint32_t streams, std::uint32_t lowrank, Storage* output) {
    __shared__ float values[kStagedThreads];
    __shared__ float coefficients[kStagedOutputs][kStagedPitch];
    // The ceil(hidden/8) grid guarantees valid >= 1. This branch is uniform.
    if (hidden - blockIdx.x * kStagedOutputs >= kStagedOutputs)
        q8_batched_hyper_output_role<Storage, true>(normalized, low, up,
            hidden, streams, lowrank, output, values, coefficients);
    else
        q8_batched_hyper_output_role<Storage, false>(normalized, low, up,
            hidden, streams, lowrank, output, values, coefficients);
}

// Wave-strided 16-byte copy of a weight tile (bytes % 16 == 0) into LDS. A load-then-store loop keeps one
// load in flight per lane, which leaves these few-block kernels latency bound. Here Depth loads are issued
// before their stores, all unconditionally: an offset past the tile is clamped to its last 16 bytes, so it
// re-reads and re-writes the same data (a guarded store would let the compiler sink its load behind the branch).
// Same bytes, same LDS image.
template<unsigned Depth=8U>
__device__ __forceinline__ void stage_tile16(unsigned char* __restrict__ stage,const unsigned char* __restrict__ source,std::uint64_t bytes,unsigned lane){
    for(std::uint64_t base=static_cast<std::uint64_t>(lane)*16U;base<bytes;base+=512ULL*Depth){
        uint4 v[Depth];
#pragma unroll
        for(unsigned d=0;d<Depth;++d)
            v[d]=*reinterpret_cast<const uint4*>(source+std::min<std::uint64_t>(base+512ULL*d,bytes-16U));
        // Keep the scheduler from interleaving each store right behind its load (it reuses one register quad).
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for(unsigned d=0;d<Depth;++d) *reinterpret_cast<uint4*>(stage+std::min<std::uint64_t>(base+512ULL*d,bytes-16U))=v[d];
    }
}

// Exact output chains: one wave per stream and one lane per output feature.
// Row tiles occupy 34*rank bytes; rank%32 makes every wave tile 16-aligned.
// Q6: up is the Q6P32 copy (26*rank-byte tiles); same codes, scales and chain.
template <bool Q6>
__device__ __forceinline__ float lane_chain_scale(const unsigned char* row, std::uint32_t rank, std::uint32_t block);
template <bool Q6>
__device__ __forceinline__ std::uint64_t lane_chain_codes(const unsigned char* row, std::uint32_t rank, std::uint32_t block, std::uint32_t g);
template <typename Storage, bool Q6 = false>
__global__ void lane_chain_hyper_output_kernel(const float* normalized,const float* low,const void* up,
    std::uint32_t hidden,std::uint32_t streams,std::uint32_t rank,Storage* output){
    extern __shared__ __align__(16) unsigned char staging[];
    const auto lane=threadIdx.x&31U,stream=threadIdx.x/32U;
    const std::uint64_t row=blockIdx.y,first=static_cast<std::uint64_t>(blockIdx.x)*32U;
    const std::uint64_t block_bytes=Q6?26U:34U;
    const std::uint64_t tile_bytes=block_bytes*rank,row_bytes=static_cast<std::uint64_t>(rank/32U)*block_bytes;
    auto* stage=staging+stream*tile_bytes;
    const auto* source=static_cast<const unsigned char*>(up)+(static_cast<std::uint64_t>(stream)*hidden+first)*row_bytes;
    stage_tile16(stage,source,tile_bytes,lane);
    __syncthreads();
    float gate=0.0F;
    for(std::uint32_t block=0;block<rank/32U;++block){
        if constexpr (Q6) {
            const auto* weight=stage+static_cast<std::uint64_t>(lane)*row_bytes;
            const float scale=lane_chain_scale<true>(weight,rank,block);
            for(std::uint32_t g=0;g<4U;++g){
                const auto codes=lane_chain_codes<true>(weight,rank,block,g);
                for(std::uint32_t l=0;l<8U;++l){
                    const float coefficient=scale*static_cast<float>(static_cast<signed char>(codes>>(8U*l)));
                    gate+=low[row*rank+static_cast<std::uint64_t>(block)*32U+8U*g+l]*coefficient;
                }
            }
            continue;
        }
        const auto* weight=stage+static_cast<std::uint64_t>(lane)*row_bytes+static_cast<std::uint64_t>(block)*34U;
        const auto scale_bits=static_cast<unsigned short>(static_cast<unsigned>(weight[0])|(static_cast<unsigned>(weight[1])<<8U));
        const float scale=__half2float(__ushort_as_half(scale_bits));
        for(std::uint32_t k=0;k<32U;++k){
            const float coefficient=scale*static_cast<float>(static_cast<signed char>(weight[2U+k]));
            gate+=low[row*rank+static_cast<std::uint64_t>(block)*32U+k]*coefficient;
        }
    }
    auto* sig=reinterpret_cast<float*>(staging+static_cast<std::uint64_t>(streams)*tile_bytes);
    sig[stream*32U+lane]=sigmoid(gate);
    __syncthreads();
    if(stream==0U){
        float mixed=0.0F;const auto feature=first+lane;
        for(std::uint32_t t=0;t<streams;++t)
            mixed+=normalized[row*static_cast<std::uint64_t>(hidden)*streams+static_cast<std::uint64_t>(t)*hidden+feature]*sig[t*32U+lane];
        store_activation(&output[row*hidden+feature],mixed/static_cast<float>(streams));
    }
}

template <typename Storage>
__global__ void deferred_lane_chain_hyper_output_kernel(const Storage* raw,const float* inverse,const float* norm,const float* low,const void* up,
    std::uint32_t hidden,std::uint32_t streams,std::uint32_t rank,Storage* output){
    extern __shared__ __align__(16) unsigned char staging[];
    const auto lane=threadIdx.x&31U,stream=threadIdx.x/32U;
    const std::uint64_t row=blockIdx.y,first=static_cast<std::uint64_t>(blockIdx.x)*32U;
    const std::uint64_t tile_bytes=34ULL*rank,row_bytes=static_cast<std::uint64_t>(rank/32U)*34U;
    auto* stage=staging+stream*tile_bytes;
    const auto* source=static_cast<const unsigned char*>(up)+(static_cast<std::uint64_t>(stream)*hidden+first)*row_bytes;
    stage_tile16(stage,source,tile_bytes,lane);
    __syncthreads();
    float gate=0.0F;
    for(std::uint32_t block=0;block<rank/32U;++block){
        const auto* weight=stage+static_cast<std::uint64_t>(lane)*row_bytes+static_cast<std::uint64_t>(block)*34U;
        const auto scale_bits=static_cast<unsigned short>(static_cast<unsigned>(weight[0])|(static_cast<unsigned>(weight[1])<<8U));
        const float scale=__half2float(__ushort_as_half(scale_bits));
        for(std::uint32_t k=0;k<32U;++k){
            const float coefficient=scale*static_cast<float>(static_cast<signed char>(weight[2U+k]));
            gate+=low[row*rank+static_cast<std::uint64_t>(block)*32U+k]*coefficient;
        }
    }
    auto* sig=reinterpret_cast<float*>(staging+static_cast<std::uint64_t>(streams)*tile_bytes);
    sig[stream*32U+lane]=sigmoid(gate);
    __syncthreads();
    if(stream==0U){
        float mixed=0.0F;const auto feature=first+lane;
        for(std::uint32_t t=0;t<streams;++t) {
            const auto index=static_cast<std::uint64_t>(t)*hidden+feature;
            const float n=__fmul_rn(__fmul_rn(load_activation(raw,row*static_cast<std::uint64_t>(hidden)*streams+index),inverse[row*streams+t]),norm[index]);
            mixed+=n*sig[t*32U+lane];
        }
        store_activation(&output[row*hidden+feature],mixed/static_cast<float>(streams));
    }
}

// Known-window rows: one launch per HC step for 2..8 rows. Every row repeats the
// exact R1 expression order; only the weight tile is loaded once for all rows.
constexpr std::uint32_t kWindowRows = 8;
struct HyperRows {
    const void* input[kWindowRows];
    void* output[kWindowRows];
    float* normalized[kWindowRows];
    float* low[kWindowRows];
    float* injection[kWindowRows];
    // Prefill: blockIdx.z selects an 8-row tile, each view blockIdx.z * its stride bytes on (0 in windows).
    std::uint64_t tile_input, tile_output, tile_normalized, tile_injection;
};
struct HyperCohortRows {
    const void* input[32];void* output[32];
    float* normalized[32];float* low[32];float* injection[32];
    std::uint64_t tile_input=0,tile_output=0,tile_normalized=0,tile_injection=0;
};
template<class Rows> constexpr unsigned hyper_row_capacity=std::is_same_v<Rows,HyperCohortRows>?32U:kWindowRows;
template<typename Pointer> __device__ __forceinline__ Pointer tiled(Pointer pointer,std::uint64_t,const HyperCohortRows&,
    std::uint32_t=0U) {return pointer;}
// Prefill tiles: the same views, blockIdx.z selects the 8-row tile. Window kernels take
// HyperRows and compile without any tile arithmetic; HyperTileRows kernels shift each view
// by blockIdx.z times its stride (the kernel argument itself is never written: a written
// copy leaves the kernarg segment and dynamic row indexing goes to scratch).
struct HyperTileRows : HyperRows {
    std::uint32_t tiles; // tiles per block of the kernels that loop over tiles (grid z = all tiles / tiles)
};
template <typename Pointer> __device__ __forceinline__ Pointer tiled(Pointer pointer, std::uint64_t, const HyperRows&,
    std::uint32_t = 0U) {
    return pointer;
}
template <typename Pointer> __device__ __forceinline__ Pointer tiled(Pointer pointer, std::uint64_t stride, const HyperTileRows&,
    std::uint32_t tile = blockIdx.z) {
    const auto offset = static_cast<std::uint64_t>(tile) * stride;
    return pointer ? reinterpret_cast<Pointer>(reinterpret_cast<std::uintptr_t>(pointer) + offset) : pointer;
}
template <class Rows> __device__ __forceinline__ std::uint32_t looped_tiles(const Rows& rows) {
    if constexpr (std::is_same_v<Rows, HyperTileRows>) return rows.tiles; else return 1U;
}

template <typename Storage, class Rows = HyperRows>
__global__ void rows_prepare_hyper_norm_kernel(Rows rows, const void* norm, std::uint32_t norm_type,
    std::uint32_t hidden, float epsilon) {
    const auto stream = blockIdx.x;
    const auto* input = static_cast<const Storage*>(tiled(rows.input[blockIdx.y], rows.tile_input, rows));
    auto* normalized = tiled(rows.normalized[blockIdx.y], rows.tile_normalized, rows);
    const auto lane = threadIdx.x;
    const auto base = static_cast<std::uint64_t>(stream) * hidden;
    __shared__ float reduction[kThreads];
    __shared__ float inverse;
    float sum = 0.0F;
    for (std::uint32_t feature = lane; feature < hidden; feature += kThreads) {
        const float value = load_activation(input, base + feature);
        sum = qpf2k_fmac(sum, value, value);
    }
    reduction[lane] = sum;
    __syncthreads();
    for (std::uint32_t stride = kThreads / 2U; stride > 0; stride >>= 1U) {
        if (lane < stride) reduction[lane] = qpf2k_add(reduction[lane + stride], reduction[lane]);
        __syncthreads();
    }
    if (lane == 0) inverse = rsqrtf(reduction[0] / static_cast<float>(hidden) + epsilon);
    __syncthreads();
    for (std::uint32_t feature = lane; feature < hidden; feature += kThreads)
        normalized[base + feature] = qpf2k_mul(qpf2k_scale<Storage>(inverse, load_activation(input, base + feature)),
            load_norm_weight(norm, norm_type, stream * hidden + feature));
}

// Window rows: the inject step (and, with Combine, the MoE combine before it) folded into the HC
// normalisation that reads its streams. With the combine one block per row, 256 threads per stream: stream 0's threads first
// compute the combined hidden row (stored as the combine launch would; the arena may place it in place of
// its shared input, which only these threads read), then every stream computes its injected slice (stored
// in the activation format, as the inject launch would: it is this step's input) and normalises the stored
// values with rows_prepare_hyper_norm_kernel's leaves, tree and product. Every two-operand FP32 step is
// operand-pinned to the separate launch's instruction (qpf2k_add / qpf2k_fmac / qpf2k_mul), and the mean divides
// by the runtime width as the norm kernel does, so the bytes equal the combine, inject and norm launches for any
// input, NaN payloads included.
struct InjectNormRows {
    const void* hyper[kWindowRows];   // inject: current streams
    const void* hidden[kWindowRows];  // inject: hidden row (combine: shared expert output)
    const void* routed[kWindowRows];  // combine: routed experts output
    const float* gates[kWindowRows];  // inject: per-stream gates
    void* combined[kWindowRows];      // combine: output row
};
constexpr std::uint32_t kInjectNormHidden = 2560U, kInjectNormStreams = 4U;
// Without the combine, block (stream, row) does one stream (the normalisation kernel's grid).
template <typename Storage, bool Combine>
__global__ __launch_bounds__(kThreads * kInjectNormStreams) void rows_inject_hyper_norm_kernel(HyperRows rows, InjectNormRows in,
    const void* norm, std::uint32_t norm_type, std::uint32_t hidden_size, float epsilon) {
    constexpr std::uint32_t hidden = kInjectNormHidden;
    const auto row = Combine ? blockIdx.x : blockIdx.y, stream = Combine ? threadIdx.x / kThreads : blockIdx.x;
    const auto lane = threadIdx.x % kThreads, slot = Combine ? stream : 0U;
    auto* streams = static_cast<Storage*>(const_cast<void*>(rows.input[row]));
    auto* normalized = rows.normalized[row];
    const auto* hyper = static_cast<const Storage*>(in.hyper[row]);
    const float gate = in.gates[row][stream];
    const auto base = static_cast<std::uint64_t>(stream) * hidden;
    __shared__ float reduction[Combine ? kInjectNormStreams : 1U][kThreads];
    __shared__ float inverse[Combine ? kInjectNormStreams : 1U];
    __shared__ float combined[Combine ? hidden : 1U];
    if constexpr (Combine) {
        if (stream == 0U)
            for (std::uint32_t feature = lane; feature < hidden; feature += kThreads) {
                Storage value;
                store_activation(&value, qpf2k_add(load_activation(static_cast<const Storage*>(in.hidden[row]), feature),
                    load_activation(static_cast<const Storage*>(in.routed[row]), feature)));
                combined[feature] = load_activation(&value, 0U);
                static_cast<Storage*>(in.combined[row])[feature] = value;
            }
        __syncthreads();
    }
    float sum = 0.0F;
    for (std::uint32_t feature = lane; feature < hidden; feature += kThreads) {
        const float h = Combine ? combined[Combine ? feature : 0U] : load_activation(static_cast<const Storage*>(in.hidden[row]), feature);
        Storage stored;
        store_activation(&stored, qpf2k_fmac(load_activation(hyper, base + feature), gate, h));
        streams[base + feature] = stored;
        const float value = load_activation(&stored, 0U);
        sum = qpf2k_fmac(sum, value, value);
    }
    reduction[slot][lane] = sum;
    __syncthreads();
    for (std::uint32_t stride = kThreads / 2U; stride > 0; stride >>= 1U) {
        if (lane < stride) reduction[slot][lane] = qpf2k_add(reduction[slot][lane + stride], reduction[slot][lane]);
        __syncthreads();
    }
    if (lane == 0) inverse[slot] = rsqrtf(reduction[slot][0] / static_cast<float>(hidden_size) + epsilon);
    __syncthreads();
    for (std::uint32_t feature = lane; feature < hidden; feature += kThreads) {
        const float value = load_activation(streams, base + feature);
        normalized[base + feature] = qpf2k_mul(qpf2k_scale<Storage>(inverse[slot], value),
            load_norm_weight(norm, norm_type, stream * hidden + feature));
    }
}

// Wave w lanes 0..7 own row w and lanes 8..15 row w+4 (output lane%8): each row's
// ascending v8 chain runs on its own SIMD, reading one broadcast value row.
__device__ __forceinline__ std::uint32_t rows_chain_row(std::uint32_t lane) {
    return lane / 32U + ((lane % 32U) < kStagedOutputs ? 0U : (lane % 32U) < 2U * kStagedOutputs ? 4U : kWindowRows);
}
template <std::uint32_t Kind, bool Injection, bool Full, class Rows>
__device__ void rows_typed_batched_projection_role(const Rows& rows, std::uint32_t count,
    const void* weights, std::uint32_t width, std::uint32_t streams, std::uint32_t outputs,
    std::uint32_t role_block, float (*values)[kStagedPitch], float (*coefficients)[kStagedPitch]) {
    const auto lane = threadIdx.x;
    const auto first_output = role_block * kStagedOutputs;
    const auto rows_left = outputs - first_output;
    const auto valid = rows_left < kStagedOutputs ? rows_left : kStagedOutputs;
    const auto row = rows_chain_row(lane), local_output = lane % kStagedOutputs;
    const auto chain = first_output + local_output;
    const bool active = row < count && chain < outputs;
    float sum = 0.0F;
    for (std::uint32_t tile = 0; tile < width; tile += kStagedThreads) {
        const auto remaining = width - tile;
        const auto tile_count = remaining < kStagedThreads ? remaining : kStagedThreads;
        if (lane < tile_count) {
            float prefetched[kStagedOutputs];
#pragma unroll
            for (std::uint32_t local = 0; local < kStagedOutputs; ++local) {
                const auto source = (Full || local < valid) ? local : valid - 1U;
                prefetched[local] = typed_projection_weight<Kind>(weights, width, first_output + source, tile + lane);
            }
            float staged[kWindowRows];
#pragma unroll
            for (std::uint32_t r = 0; r < kWindowRows; ++r) if (r < count) staged[r] = tiled(rows.normalized[r], rows.tile_normalized, rows)[tile + lane];
#pragma unroll
            for (std::uint32_t r = 0; r < kWindowRows; ++r) if (r < count) values[r][lane] = staged[r];
#pragma unroll
            for (std::uint32_t local = 0; local < kStagedOutputs; ++local)
                coefficients[(Full || local < valid) ? local : valid - 1U][lane] = prefetched[local];
        }
        __syncthreads();
        if (active)
#pragma unroll 8
            for (std::uint32_t inner = 0; inner < tile_count; ++inner)
                sum += values[row][inner] * coefficients[local_output][inner];
        __syncthreads();
    }
    if (active) {
        const float scaled = sum / static_cast<float>(streams);
        (Injection ? tiled(rows.injection[row], rows.tile_injection, rows) : tiled(rows.low[row], rows.tile_normalized, rows))[chain] =
            Injection ? 2.0F * sigmoid(scaled) : scaled * sigmoid(scaled);
    }
}

template <class Rows = HyperRows>
__global__ void rows_batched_typed_hyper_projection_kernel(Rows rows, std::uint32_t count,
    const void* down, std::uint32_t width, std::uint32_t streams, std::uint32_t lowrank,
    const void* inject, std::uint32_t lowrank_blocks) {
    __shared__ float values[kWindowRows][kStagedPitch];
    __shared__ float coefficients[kStagedOutputs][kStagedPitch];
    if (blockIdx.x < lowrank_blocks) {
        if (lowrank - blockIdx.x * kStagedOutputs >= kStagedOutputs)
            rows_typed_batched_projection_role<8U, false, true>(rows, count, down, width, streams, lowrank,
                blockIdx.x, values, coefficients);
        else
            rows_typed_batched_projection_role<8U, false, false>(rows, count, down, width, streams, lowrank,
                blockIdx.x, values, coefficients);
    } else {
        const auto role_block = blockIdx.x - lowrank_blocks;
        if (streams - role_block * kStagedOutputs >= kStagedOutputs)
            rows_typed_batched_projection_role<0U, true, true>(rows, count, inject, width, streams, streams,
                role_block, values, coefficients);
        else
            rows_typed_batched_projection_role<0U, true, false>(rows, count, inject, width, streams, streams,
                role_block, values, coefficients);
    }
}

// Split-K P1: weights stay in registers while every row replays the R1 lane
// chain, the xor reduction and the ordered 16-tile block sum.
// Q6: weights is the Q6P32 copy of the Q8_0 down tensor; the same codes and
// scales are decoded once all loads are in flight.
template <unsigned R, bool Injection, class LoadX, class Rows, bool Q6 = false>
__device__ __forceinline__ void rows_splitk_hyper_lane_dots(const Rows& rows, unsigned count,
    const void* weights, unsigned first, unsigned valid, unsigned tile, unsigned lane, float (*red)[16][R], LoadX&& load_x) {
    (void)rows;
    float4 w[Injection ? R : 1U][5];
    unsigned packed[Injection ? 1U : R][5];
    float scales[Injection ? 1U : R][5];
    if constexpr (Q6 && !Injection) {
        // Lane: block tile*20+4s+lane/8 of the row, its weights 4(lane%8)..+3.
        unsigned qs[R][5], qh[R][5];
        unsigned short bits[R][5];
        const unsigned m=lane%8U;
#pragma unroll
        for (unsigned r=0;r<R;++r) if(r<valid) {
#pragma unroll
            for(unsigned s=0;s<5U;++s) {
                const unsigned b=tile*20U+4U*s+lane/8U;
                const auto o=q6p::chunk_offsets(10240U,b/8U);
                const auto* row=static_cast<const unsigned char*>(weights)+static_cast<std::uint64_t>(first+r)*q6p::row_bytes(10240U);
                qs[r][s]=*reinterpret_cast<const unsigned*>(row+o.qs+16U*(b%8U)+4U*(m/2U));
                qh[r][s]=*reinterpret_cast<const unsigned*>(row+o.qh+8U*(b%8U)+4U*(m/4U));
                bits[r][s]=*reinterpret_cast<const unsigned short*>(row+o.d+2U*(b%8U));
            }
        }
#pragma unroll
        for (unsigned r=0;r<R;++r) if(r<valid) {
#pragma unroll
            for(unsigned s=0;s<5U;++s) {
                scales[r][s]=__half2float(__ushort_as_half(bits[r][s]));
                packed[r][s]=q6p::codes4(qs[r][s],qh[r][s],m);
            }
        }
    } else
#pragma unroll
    for (unsigned r=0;r<R;++r) if(r<valid) {
#pragma unroll
        for(unsigned s=0;s<5U;++s) {
            if constexpr (Injection) {
                w[r][s]=*reinterpret_cast<const float4*>(static_cast<const float*>(weights)+
                    static_cast<std::uint64_t>(first+r)*10240U+tile*640U+s*128U+lane*4U);
            } else {
                const auto* block=static_cast<const unsigned char*>(weights)+
                    static_cast<std::uint64_t>(first+r)*320U*34U+static_cast<std::uint64_t>(tile*20U+4U*s+lane/8U)*34U;
                const auto bits=*reinterpret_cast<const unsigned short*>(block);
                const auto* codes=reinterpret_cast<const unsigned short*>(block+2U+4U*(lane%8U));
                scales[r][s]=__half2float(__ushort_as_half(bits));
                packed[r][s]=static_cast<unsigned>(codes[0])|(static_cast<unsigned>(codes[1])<<16U);
            }
        }
    }
    for (unsigned t=0;t<count;++t) {
        float4 x[5];
#pragma unroll
        for (unsigned s=0;s<5U;++s) x[s]=load_x(t,tile*640U+s*128U+lane*4U);
        float sums[R];
#pragma unroll
        for(unsigned r=0;r<R;++r) {
            sums[r]=0.0F;
            if(r<valid) {
#pragma unroll
                for(unsigned s=0;s<5U;++s) {
                    if constexpr (Injection) {
                        sums[r]=fmaf(x[s].x,w[r][s].x,sums[r]);
                        sums[r]=fmaf(x[s].y,w[r][s].y,sums[r]);
                        sums[r]=fmaf(x[s].z,w[r][s].z,sums[r]);
                        sums[r]=fmaf(x[s].w,w[r][s].w,sums[r]);
                    } else {
                        const float inputs[4]={x[s].x,x[s].y,x[s].z,x[s].w};
#pragma unroll
                        for(unsigned j=0;j<4U;++j) {
                            const unsigned byte=(packed[r][s]>>(j*8U))&255U;
                            const int code=byte<128U?static_cast<int>(byte):static_cast<int>(byte)-256;
                            const float coefficient=scales[r][s]*static_cast<float>(code);
                            sums[r]=fmaf(inputs[j],coefficient,sums[r]);
                        }
                    }
                }
            }
        }
#pragma unroll
        for(unsigned r=0;r<R;++r) if(r<valid) {
#pragma unroll
            for(unsigned offset=16U;offset!=0U;offset/=2U)
                sums[r]=qpf2k_add(sums[r],__shfl_xor(sums[r],static_cast<int>(offset),32));
        }
        if(lane==0U) {
#pragma unroll
            for(unsigned r=0;r<R;++r)if(r<valid)red[t][tile][r]=sums[r];
        }
    }
}

template <unsigned R, class Rows = HyperRows, bool Q6 = false>
__global__ __launch_bounds__(512) void rows_splitk_hyper_projection_kernel(Rows rows, unsigned count,
    const void* __restrict__ down, unsigned streams, unsigned rank, const void* __restrict__ inject, unsigned low_blocks) {
    __shared__ float red[hyper_row_capacity<Rows>][16][R];
    const unsigned lane=threadIdx.x%32U,tile=threadIdx.x/32U;
    const bool is_inject=blockIdx.x>=low_blocks;
    const unsigned first=(is_inject?blockIdx.x-low_blocks:blockIdx.x)*R;
    const unsigned remaining=(is_inject?streams:rank)-first;
    const unsigned valid=remaining<R?remaining:R;
    const auto stored=[&](unsigned t,unsigned at){return *reinterpret_cast<const float4*>(tiled(rows.normalized[t], rows.tile_normalized, rows)+at);};
    if(is_inject)rows_splitk_hyper_lane_dots<R,true>(rows,count,inject,first,valid,tile,lane,red,stored);
    else rows_splitk_hyper_lane_dots<R,false,decltype(stored)&,Rows,Q6>(rows,count,down,first,valid,tile,lane,red,stored);
    __syncthreads();
    const unsigned t=threadIdx.x/R,r=threadIdx.x%R;
    if(t<count && r<valid) {
        float total=0.0F;
#pragma unroll
        for(unsigned k=0;k<16U;++k)total=qpf2k_add(total,red[t][k][r]);
        const float scaled=total/static_cast<float>(streams);
        const float value=qpf2k_mul(is_inject?2.0F:scaled,sigmoid(scaled));
        if(is_inject)tiled(rows.injection[t], rows.tile_injection, rows)[first+r]=value;
        else tiled(rows.low[t], rows.tile_normalized, rows)[first+r]=value;
    }
}

// rows_prepare_hyper_norm_kernel folded into the split-K projection. Each wave rebuilds
// one (row, stream) inverse with that kernel's 256-leaf order (leaf l = lane + 32 j sums
// features l, l + 256, ...; strides 128..32 in registers, 16..1 across lanes), every block
// normalises its inputs on the fly with the same product, and the blocks together store
// the normalised rows the output kernel reads.
template <unsigned R, typename Storage, class Rows = HyperRows>
__global__ __launch_bounds__(512) void rows_splitk_norm_hyper_projection_kernel(Rows rows, unsigned count,
    const void* __restrict__ down, unsigned streams, unsigned rank, const void* __restrict__ inject, unsigned low_blocks,
    const void* __restrict__ norm, std::uint32_t norm_type, unsigned hidden, float epsilon) {
    __shared__ float red[hyper_row_capacity<Rows>][16][R];
    __shared__ float inverse[hyper_row_capacity<Rows>][8];
    const unsigned lane=threadIdx.x%32U,tile=threadIdx.x/32U;
    for(unsigned pair=tile;pair<count*streams;pair+=16U) {
        const unsigned t=pair/streams,stream=pair%streams;
        const auto* input=static_cast<const Storage*>(tiled(rows.input[t], rows.tile_input, rows));
        const auto base=static_cast<std::uint64_t>(stream)*hidden;
        float leaf[8];
#pragma unroll
        for(unsigned j=0;j<8U;++j) {
            float sum = 0.0F;
            for (std::uint32_t feature = lane + 32U * j; feature < hidden; feature += kThreads) {
                const float value = load_activation(input, base + feature);
                sum += value * value;
            }
            leaf[j]=sum;
        }
#pragma unroll
        for(unsigned j=0;j<4U;++j) leaf[j]+=leaf[j+4U];
#pragma unroll
        for(unsigned j=0;j<2U;++j) leaf[j]+=leaf[j+2U];
        leaf[0]+=leaf[1];
        float v=leaf[0];
#pragma unroll
        for(unsigned d=16U;d!=0U;d/=2U) {
            const float partner=__shfl_down(v,d,32);
            if(lane<d) v+=partner;
        }
        if(lane==0U) inverse[t][stream]=rsqrtf(v / static_cast<float>(hidden) + epsilon);
    }
    __syncthreads();
    const auto width=static_cast<std::uint64_t>(streams)*hidden,total=width*count;
    const auto share=(total+gridDim.x-1U)/gridDim.x,begin=share*blockIdx.x,end=begin+share<total?begin+share:total;
    for(auto e=begin+threadIdx.x;e<end;e+=512U) {
        const auto t=static_cast<unsigned>(e/width);const std::uint64_t at=e%width;
        tiled(rows.normalized[t], rows.tile_normalized, rows)[at] = load_activation(static_cast<const Storage*>(tiled(rows.input[t], rows.tile_input, rows)), at) *
            inverse[t][static_cast<unsigned>(at/hidden)] * load_norm_weight(norm, norm_type, static_cast<std::uint32_t>(at));
    }
    const auto normalized=[&](unsigned t,unsigned at) {
        const auto* input=static_cast<const Storage*>(tiled(rows.input[t], rows.tile_input, rows));
        const float scale=inverse[t][at/hidden];
        float4 x;
        x.x = load_activation(input, at) * scale * load_norm_weight(norm, norm_type, at);
        x.y = load_activation(input, at + 1U) * scale * load_norm_weight(norm, norm_type, at + 1U);
        x.z = load_activation(input, at + 2U) * scale * load_norm_weight(norm, norm_type, at + 2U);
        x.w = load_activation(input, at + 3U) * scale * load_norm_weight(norm, norm_type, at + 3U);
        return x;
    };
    const bool is_inject=blockIdx.x>=low_blocks;
    const unsigned first=(is_inject?blockIdx.x-low_blocks:blockIdx.x)*R;
    const unsigned remaining=(is_inject?streams:rank)-first;
    const unsigned valid=remaining<R?remaining:R;
    if(is_inject)rows_splitk_hyper_lane_dots<R,true>(rows,count,inject,first,valid,tile,lane,red,normalized);
    else rows_splitk_hyper_lane_dots<R,false>(rows,count,down,first,valid,tile,lane,red,normalized);
    __syncthreads();
    const unsigned t=threadIdx.x/R,r=threadIdx.x%R;
    if(t<count && r<valid) {
        float total_dot=0.0F;
#pragma unroll
        for(unsigned k=0;k<16U;++k)total_dot=total_dot+red[t][k][r];
        const float scaled=total_dot/static_cast<float>(streams);
        const float value=qpf2k_mul(is_inject?2.0F:scaled,sigmoid(scaled));
        if(is_inject)tiled(rows.injection[t], rows.tile_injection, rows)[first+r]=value;
        else tiled(rows.low[t], rows.tile_normalized, rows)[first+r]=value;
    }
}

template <typename Storage, bool Full, class Rows>
__device__ void rows_q8_batched_hyper_output_role(const Rows& rows, std::uint32_t count, const void* up,
    std::uint32_t hidden, std::uint32_t streams, std::uint32_t lowrank,
    float (*values)[kStagedPitch], float (*coefficients)[kStagedPitch]) {
    const auto lane = threadIdx.x;
    const auto first_feature = blockIdx.x * kStagedOutputs;
    const auto row = rows_chain_row(lane), local_feature = lane % kStagedOutputs;
    const auto feature = first_feature + local_feature;
    const bool active = row < count && feature < hidden;
    const auto rows_left = hidden - first_feature;
    const auto valid = rows_left < kStagedOutputs ? rows_left : kStagedOutputs;
    float mixed = 0.0F;
    for (std::uint32_t stream = 0; stream < streams; ++stream) {
        float gate = 0.0F;
        for (std::uint32_t tile = 0; tile < lowrank; tile += kStagedThreads) {
            const auto remaining = lowrank - tile;
            const auto tile_count = remaining < kStagedThreads ? remaining : kStagedThreads;
            if (lane < tile_count) {
                float prefetched[kStagedOutputs];
#pragma unroll
                for (std::uint32_t local = 0; local < kStagedOutputs; ++local) {
                    const auto source = (Full || local < valid) ? local : valid - 1U;
                    prefetched[local] = typed_projection_weight<8U>(up, lowrank,
                        stream * hidden + first_feature + source, tile + lane);
                }
                float staged[kWindowRows];
#pragma unroll
                for (std::uint32_t r = 0; r < kWindowRows; ++r) if (r < count) staged[r] = tiled(rows.low[r], rows.tile_normalized, rows)[tile + lane];
#pragma unroll
                for (std::uint32_t r = 0; r < kWindowRows; ++r) if (r < count) values[r][lane] = staged[r];
#pragma unroll
                for (std::uint32_t local = 0; local < kStagedOutputs; ++local)
                    coefficients[(Full || local < valid) ? local : valid - 1U][lane] = prefetched[local];
            }
            __syncthreads();
            if (active)
#pragma unroll 8
                for (std::uint32_t inner = 0; inner < tile_count; ++inner)
                    gate += values[row][inner] * coefficients[local_feature][inner];
            __syncthreads();
        }
        if (active)
            mixed = qpf2k_fmac(mixed, sigmoid(gate), tiled(rows.normalized[row], rows.tile_normalized, rows)[stream * hidden + feature]);
    }
    if (active)
        store_activation(&static_cast<Storage*>(tiled(rows.output[row], rows.tile_output, rows))[feature], mixed / static_cast<float>(streams));
}

template <typename Storage, class Rows = HyperRows>
__global__ void rows_q8_batched_hyper_output_kernel(Rows rows, std::uint32_t count, const void* up,
    std::uint32_t hidden, std::uint32_t streams, std::uint32_t lowrank) {
    __shared__ float values[kWindowRows][kStagedPitch];
    __shared__ float coefficients[kStagedOutputs][kStagedPitch];
    if (hidden - blockIdx.x * kStagedOutputs >= kStagedOutputs)
        rows_q8_batched_hyper_output_role<Storage, true>(rows, count, up, hidden, streams, lowrank, values, coefficients);
    else
        rows_q8_batched_hyper_output_role<Storage, false>(rows, count, up, hidden, streams, lowrank, values, coefficients);
}

// One staged up tile per block; each stream wave runs every row's R1 lane chain.
// Count > 0 fixes the row count at compile time: rows advance together, each
// coefficient decoded once for all of them, every row keeping its own serial chain.
// Q6: up is the Q6P32 copy (Q6P32 rows in the staged tile); same codes and scales.
template <bool Q6>
__device__ __forceinline__ float lane_chain_scale(const unsigned char* row, std::uint32_t rank, std::uint32_t block) {
    static_assert(Q6);
    const auto o=q6p::chunk_offsets(rank,block/8U);
    return __half2float(__ushort_as_half(*reinterpret_cast<const unsigned short*>(row+o.d+2U*(block%8U))));
}
// Codes of weights 8g..8g+7 of a block (byte l = weight 8g+l).
template <bool Q6>
__device__ __forceinline__ std::uint64_t lane_chain_codes(const unsigned char* row, std::uint32_t rank, std::uint32_t block, std::uint32_t g) {
    static_assert(Q6);
    const auto o=q6p::chunk_offsets(rank,block/8U);
    return q6p::codes8(*reinterpret_cast<const unsigned*>(row+o.qs+16U*(block%8U)+4U*g),
        *reinterpret_cast<const unsigned*>(row+o.qh+8U*(block%8U)+4U*(g/2U)),4U*(g&1U));
}
template <typename Storage, std::uint32_t Count = 0U, class Rows = HyperRows, bool Q6 = false>
__global__ void rows_lane_chain_hyper_output_kernel(Rows rows, std::uint32_t count, const void* up,
    std::uint32_t hidden, std::uint32_t streams, std::uint32_t rank) {
    extern __shared__ __align__(16) unsigned char staging[];
    const auto lane=threadIdx.x&31U,stream=threadIdx.x/32U;
    const std::uint64_t first=static_cast<std::uint64_t>(blockIdx.x)*32U;
    const std::uint64_t block_bytes=Q6?26U:34U;
    const std::uint64_t tile_bytes=block_bytes*rank,row_bytes=static_cast<std::uint64_t>(rank/32U)*block_bytes;
    auto* stage=staging+stream*tile_bytes;
    const auto* source=static_cast<const unsigned char*>(up)+(static_cast<std::uint64_t>(stream)*hidden+first)*row_bytes;
    stage_tile16(stage,source,tile_bytes,lane);
    __syncthreads();
    auto* sig=reinterpret_cast<float*>(staging+static_cast<std::uint64_t>(streams)*tile_bytes);
    // Tile rows reuse the staged weight tile: one block runs the chains of every tile.
    for(std::uint32_t z=0;z<looped_tiles(rows);++z){
    if(z)__syncthreads();
    if constexpr (Count != 0U) {
    const float* low[Count];
    float gate[Count];
#pragma unroll
    for(std::uint32_t t=0;t<Count;++t){ low[t]=tiled(rows.low[t], rows.tile_normalized, rows, blockIdx.z * looped_tiles(rows) + z); gate[t]=0.0F; }
    for(std::uint32_t block=0;block<rank/32U;++block){
        if constexpr (Q6) {
            const auto* weight=stage+static_cast<std::uint64_t>(lane)*row_bytes;
            const float scale=lane_chain_scale<true>(weight,rank,block);
#pragma unroll
            for(std::uint32_t g=0;g<4U;++g){
                const auto codes=lane_chain_codes<true>(weight,rank,block,g);
#pragma unroll
                for(std::uint32_t l=0;l<8U;++l){
                    const float coefficient=scale*static_cast<float>(static_cast<signed char>(codes>>(8U*l)));
#pragma unroll
                    for(std::uint32_t t=0;t<Count;++t)
                        gate[t]+=low[t][static_cast<std::uint64_t>(block)*32U+8U*g+l]*coefficient;
                }
            }
            continue;
        }
        const auto* weight=stage+static_cast<std::uint64_t>(lane)*row_bytes+static_cast<std::uint64_t>(block)*34U;
        const auto scale_bits=static_cast<unsigned short>(static_cast<unsigned>(weight[0])|(static_cast<unsigned>(weight[1])<<8U));
        const float scale=__half2float(__ushort_as_half(scale_bits));
#pragma unroll
        for(std::uint32_t k=0;k<32U;++k){
            const float coefficient=scale*static_cast<float>(static_cast<signed char>(weight[2U+k]));
#pragma unroll
            for(std::uint32_t t=0;t<Count;++t)
                gate[t]+=low[t][static_cast<std::uint64_t>(block)*32U+k]*coefficient;
        }
    }
#pragma unroll
    for(std::uint32_t t=0;t<Count;++t) sig[(t*streams+stream)*32U+lane]=sigmoid(gate[t]);
    } else
    for(std::uint32_t t=0;t<count;++t){
        const auto* low=tiled(rows.low[t], rows.tile_normalized, rows, blockIdx.z * looped_tiles(rows) + z);
        float gate=0.0F;
        for(std::uint32_t block=0;block<rank/32U;++block){
            if constexpr (Q6) {
                const auto* weight=stage+static_cast<std::uint64_t>(lane)*row_bytes;
                const float scale=lane_chain_scale<true>(weight,rank,block);
                for(std::uint32_t g=0;g<4U;++g){
                    const auto codes=lane_chain_codes<true>(weight,rank,block,g);
                    for(std::uint32_t l=0;l<8U;++l){
                        const float coefficient=scale*static_cast<float>(static_cast<signed char>(codes>>(8U*l)));
                        gate+=low[static_cast<std::uint64_t>(block)*32U+8U*g+l]*coefficient;
                    }
                }
                continue;
            }
            const auto* weight=stage+static_cast<std::uint64_t>(lane)*row_bytes+static_cast<std::uint64_t>(block)*34U;
            const auto scale_bits=static_cast<unsigned short>(static_cast<unsigned>(weight[0])|(static_cast<unsigned>(weight[1])<<8U));
            const float scale=__half2float(__ushort_as_half(scale_bits));
            for(std::uint32_t k=0;k<32U;++k){
                const float coefficient=scale*static_cast<float>(static_cast<signed char>(weight[2U+k]));
                gate+=low[static_cast<std::uint64_t>(block)*32U+k]*coefficient;
            }
        }
        sig[(t*streams+stream)*32U+lane]=sigmoid(gate);
    }
    __syncthreads();
    for(std::uint32_t t=stream;t<count;t+=streams){
        float mixed=0.0F;const auto feature=first+lane;
        for(std::uint32_t s=0;s<streams;++s)
            mixed=qpf2k_fmac(mixed,tiled(rows.normalized[t], rows.tile_normalized, rows, blockIdx.z * looped_tiles(rows) + z)[static_cast<std::uint64_t>(s)*hidden+feature],sig[(t*streams+s)*32U+lane]);
        store_activation(&static_cast<Storage*>(tiled(rows.output[t], rows.tile_output, rows, blockIdx.z * looped_tiles(rows) + z))[feature],mixed/static_cast<float>(streams));
    }
    }
}

// Row-split form (Q8_0, Count rows known at compile time): G groups of `streams` waves per block share the
// staged up tile; group g runs the chains of rows g, g + G, ... (Count / G rows per lane instead of Count),
// then all groups mix. Every (row, stream, feature) chain is the one of rows_lane_chain_hyper_output_kernel
// (block, then code order; same coefficient and accumulation expressions), so the bits are the same; only
// how many lanes share the rows changes (80 blocks of 128 threads ran 8 waves per CU).
template <typename Storage, std::uint32_t Count, std::uint32_t G, class Rows = HyperRows>
__global__ void rows_lane_chain_split_hyper_output_kernel(Rows rows, const void* up, std::uint32_t hidden,
    std::uint32_t streams, std::uint32_t rank) {
    static_assert(Count % G == 0U);
    extern __shared__ __align__(16) unsigned char staging[];
    const auto lane=threadIdx.x&31U,wave=threadIdx.x/32U,stream=wave%streams,group=wave/streams;
    const std::uint64_t first=static_cast<std::uint64_t>(blockIdx.x)*32U;
    const std::uint64_t tile_bytes=34U*static_cast<std::uint64_t>(rank),row_bytes=static_cast<std::uint64_t>(rank/32U)*34U;
    auto* stage=staging+stream*tile_bytes;
    const auto* source=static_cast<const unsigned char*>(up)+(static_cast<std::uint64_t>(stream)*hidden+first)*row_bytes;
    if(group==0U) stage_tile16(stage,source,tile_bytes,lane);
    __syncthreads();
    auto* sig=reinterpret_cast<float*>(staging+static_cast<std::uint64_t>(streams)*tile_bytes);
    for(std::uint32_t z=0;z<looped_tiles(rows);++z){
    if(z)__syncthreads();
    constexpr std::uint32_t Own=Count/G;
    const float* low[Own];
    float gate[Own];
#pragma unroll
    for(std::uint32_t i=0;i<Own;++i){ low[i]=tiled(rows.low[group+G*i], rows.tile_normalized, rows, blockIdx.z * looped_tiles(rows) + z); gate[i]=0.0F; }
    for(std::uint32_t block=0;block<rank/32U;++block){
        const auto* weight=stage+static_cast<std::uint64_t>(lane)*row_bytes+static_cast<std::uint64_t>(block)*34U;
        const auto scale_bits=static_cast<unsigned short>(static_cast<unsigned>(weight[0])|(static_cast<unsigned>(weight[1])<<8U));
        const float scale=__half2float(__ushort_as_half(scale_bits));
#pragma unroll
        for(std::uint32_t k=0;k<32U;++k){
            const float coefficient=scale*static_cast<float>(static_cast<signed char>(weight[2U+k]));
#pragma unroll
            for(std::uint32_t i=0;i<Own;++i)
                gate[i]+=low[i][static_cast<std::uint64_t>(block)*32U+k]*coefficient;
        }
    }
#pragma unroll
    for(std::uint32_t i=0;i<Own;++i) sig[((group+G*i)*streams+stream)*32U+lane]=sigmoid(gate[i]);
    __syncthreads();
    for(std::uint32_t t=wave;t<Count;t+=G*streams){
        float mixed=0.0F;const auto feature=first+lane;
        for(std::uint32_t s=0;s<streams;++s)
            mixed=qpf2k_fmac(mixed,tiled(rows.normalized[t], rows.tile_normalized, rows, blockIdx.z * looped_tiles(rows) + z)[static_cast<std::uint64_t>(s)*hidden+feature],sig[(t*streams+s)*32U+lane]);
        store_activation(&static_cast<Storage*>(tiled(rows.output[t], rows.tile_output, rows, blockIdx.z * looped_tiles(rows) + z))[feature],mixed/static_cast<float>(streams));
    }
    }
}

// Rank-320 form of the rows lane chain: each lane loads its own 340-byte Q8 row straight into registers
// (21 x 16 B + 4 B, all issued up front) and starts the chain as soon as the first block has arrived, so
// the chain overlaps the remaining loads instead of following a whole-tile LDS stage. Same per-row chain
// (block, then code order), same coefficient and accumulation expressions, same stream mix.
// Q6: up is the Q6P32 copy (260-byte rows); same codes, scales and chain.
template <typename Storage, std::uint32_t Count, class Rows = HyperRows, bool Q6 = false>
__global__ void rows_lane_chain_hyper_output_regs_kernel(Rows rows, const void* up,
    std::uint32_t hidden, std::uint32_t streams) {
    constexpr std::uint32_t kBlocks = 10U, kRowBytes = kBlocks * (Q6 ? 26U : 34U), kWords = kRowBytes / 4U;
    extern __shared__ __align__(16) unsigned char staging[];
    const auto lane = threadIdx.x & 31U, stream = threadIdx.x / 32U;
    const std::uint64_t first = static_cast<std::uint64_t>(blockIdx.x) * 32U;
    const auto* row = static_cast<const unsigned char*>(up) +
        (static_cast<std::uint64_t>(stream) * hidden + first + lane) * kRowBytes;
    unsigned words[kWords];
#pragma unroll
    for (std::uint32_t q = 0; q < kWords / 4U; ++q) {
        const uint4 v = *reinterpret_cast<const uint4*>(row + 16U * q);
        words[4U * q] = v.x; words[4U * q + 1U] = v.y; words[4U * q + 2U] = v.z; words[4U * q + 3U] = v.w;
    }
#pragma unroll
    for (std::uint32_t w = (kWords / 4U) * 4U; w < kWords; ++w) words[w] = *reinterpret_cast<const unsigned*>(row + 4U * w);
    const auto byte_at = [&](std::uint32_t i) -> unsigned { return (words[i / 4U] >> (8U * (i % 4U))) & 255U; };
    auto* sig = reinterpret_cast<float*>(staging);
    const float* low[Count];
    float gate[Count];
#pragma unroll
    for (std::uint32_t t = 0; t < Count; ++t) { low[t] = tiled(rows.low[t], rows.tile_normalized, rows); gate[t] = 0.0F; }
#pragma unroll
    for (std::uint32_t block = 0; block < kBlocks; ++block) {
        float scale;
        if constexpr (Q6) {
            const auto o = q6p::chunk_offsets(320U, block / 8U);
            scale = __half2float(__ushort_as_half(static_cast<unsigned short>(byte_at(o.d + 2U * (block % 8U)) |
                (byte_at(o.d + 2U * (block % 8U) + 1U) << 8U))));
        } else {
            const auto scale_bits = static_cast<unsigned short>(byte_at(block * 34U) | (byte_at(block * 34U + 1U) << 8U));
            scale = __half2float(__ushort_as_half(scale_bits));
        }
#pragma unroll
        for (std::uint32_t k = 0; k < 32U; ++k) {
            signed char code;
            if constexpr (Q6) {
                const auto o = q6p::chunk_offsets(320U, block / 8U);
                const std::uint32_t g = k / 8U;
                const auto codes = q6p::codes8(words[(o.qs + 16U * (block % 8U) + 4U * g) / 4U],
                    words[(o.qh + 8U * (block % 8U) + 4U * (g / 2U)) / 4U], 4U * (g & 1U));
                code = static_cast<signed char>(codes >> (8U * (k % 8U)));
            } else code = static_cast<signed char>(byte_at(block * 34U + 2U + k));
            const float coefficient = scale * static_cast<float>(code);
#pragma unroll
            for (std::uint32_t t = 0; t < Count; ++t)
                gate[t] += low[t][static_cast<std::uint64_t>(block) * 32U + k] * coefficient;
        }
    }
#pragma unroll
    for (std::uint32_t t = 0; t < Count; ++t) sig[(t * streams + stream) * 32U + lane] = sigmoid(gate[t]);
    __syncthreads();
    for (std::uint32_t t = stream; t < Count; t += streams) {
        float mixed = 0.0F; const auto feature = first + lane;
        for (std::uint32_t s2 = 0; s2 < streams; ++s2)
            mixed = qpf2k_fmac(mixed, tiled(rows.normalized[t], rows.tile_normalized, rows)[static_cast<std::uint64_t>(s2) * hidden + feature],
                sig[(t * streams + s2) * 32U + lane]);
        store_activation(&static_cast<Storage*>(tiled(rows.output[t], rows.tile_output, rows))[feature], mixed / static_cast<float>(streams));
    }
}

// Persistent window HC mix (STRIX_HC_PERSISTENT=1; rank 320, 4 streams of 2560, Q8_0 down/up, F32 inject):
// the normalisation, the split-K projection and the rank-320 lane-chain output of 2/4/8 rows in one launch
// of kPersistBlocks co-resident blocks of 16 waves, the phases separated by two grid barriers instead of
// two dependent dispatches. Every wave works in every phase: A) block g normalises (row, stream) pairs
// 2g and 2g + 1, eight waves each; B) the blocks take split-K units (two low outputs or one inject output,
// wave w = tile w) from a queue; C) waves 0..7 of block g run the up chains of features 64g..64g+63, their
// up rows requested at the end of B. Every value keeps the expression of the three-kernel form: the
// 256-leaf norm tree, the per-tile lane chain with its xor reduction and ascending 16-tile sum, the
// per-lane up chain and the ascending stream mix. The folded inject, the norm leaves and tree (upper operand
// first at every level, as rows_prepare_hyper_norm_kernel), the normalised product and the stream mix are
// operand-pinned to the separate kernels' instructions (qpf2k_add / qpf2k_fmac / qpf2k_mul), so NaN payloads
// match them too.
constexpr std::uint32_t kPersistBlocks = 40U, kPersistHidden = 2560U, kPersistStreams = 4U, kPersistRank = 320U;
constexpr std::uint32_t kPersistWidth = kPersistHidden * kPersistStreams;
constexpr std::uint32_t kPersistUnitOutputs = 2U; // low outputs per split-K work unit (4 spills at 8 rows)
static_assert(kPersistHidden == kPersistBlocks * 64U && kPersistRank % kPersistUnitOutputs == 0U);

// Orders LDS only: the waves keep their weight loads in flight across it.
__device__ __forceinline__ void persist_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
// Scalar, L0/GL1-bypassing read of the arrival counter: a vector load would make its wave wait for every
// vector load it still has in flight (vmcnt retires in order).
__device__ __forceinline__ unsigned persist_poll(const unsigned* counter) {
    unsigned value;
    asm volatile("s_load_b32 %0, %1, 0x0 glc dlc\n\ts_waitcnt lgkmcnt(0)" : "=s"(value) : "s"(counter) : "memory");
    return value;
}
// Every wave has waited for its own global stores (vscnt) before. Wave `poller` arrives and polls; the counter
// only grows and the host hands each launch its base. The L0/GL1 invalidation makes the other blocks'
// stores visible to the plain loads that follow.
__device__ __forceinline__ void persist_grid_barrier(unsigned* counter, unsigned target, unsigned poller) {
    persist_lds_barrier();
    if (threadIdx.x / 32U == poller) {
        if ((threadIdx.x & 31U) == 0U) __hip_atomic_fetch_add(counter, 1U, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        unsigned spins = 0;
        while (static_cast<int>(persist_poll(counter) - target) < 0) {
            __builtin_amdgcn_s_sleep(1);
            if (++spins > (1U << 26U)) __builtin_trap(); // a block never arrived: fault instead of hanging
        }
    }
    persist_lds_barrier();
    asm volatile("buffer_gl0_inv\n\tbuffer_gl1_inv" ::: "memory");
}
__device__ __forceinline__ void persist_wait_stores() { asm volatile("s_waitcnt_vscnt null, 0x0" ::: "memory"); }

// The preceding inject step folded into the normalisation (STRIX_HC_PERSISTENT_INJECT=1): the waves of
// (row, stream) compute that slice of the injected streams with the inject kernel's expression and store
// it as the inject would (it is this mix's input), then normalise the stored values.
struct PersistInjectRows {
    const void* hyper[kWindowRows];
    const void* hidden[kWindowRows];
    const float* weights[kWindowRows];
};
template <typename Storage, std::uint32_t Count, bool Inject = false>
__global__ __launch_bounds__(512) __attribute__((amdgpu_waves_per_eu(8))) void rows_persistent_hyper_mix_kernel(HyperRows rows,
    const void* __restrict__ norm, std::uint32_t norm_type, float epsilon, std::uint32_t hidden_size, const unsigned char* __restrict__ down,
    const float* __restrict__ inject, const unsigned char* __restrict__ up, unsigned* counter, unsigned base,
    unsigned* queue, unsigned queue_base, PersistInjectRows injected, unsigned long long* trace, bool up_late) {
    constexpr std::uint32_t kBlocks = 10U, kRowBytes = kBlocks * 34U, kWords = kRowBytes / 4U;
    constexpr std::uint32_t kLeaves = kPersistHidden / kThreads; // features per leaf
    __shared__ float leaf[2][kThreads];
    __shared__ float inverse_lds[2];
    __shared__ float red[Count][16][kPersistUnitOutputs];
    __shared__ __align__(16) float low_lds[kPersistRank][Count]; // transposed: a coefficient's rows side by side
    __shared__ float sig[Count][kPersistStreams][64];
    __shared__ unsigned unit_lds;
    const unsigned g = blockIdx.x, wave = threadIdx.x / 32U, lane = threadIdx.x & 31U;
    // STRIX_HC_PERSISTENT_TRACE: per-block phase timestamps (wall_clock64, 100 MHz).
    const auto mark = [&](unsigned k) { if (trace) trace[g * 8U + k] = static_cast<unsigned long long>(wall_clock64()); };
    if (threadIdx.x == 0U) mark(0U);
    // A: each leaf's ten inputs and norm weights are requested before any of them is used.
    const unsigned half = wave / 8U, pair = 2U * g + half, at = pair / kPersistStreams, stream = pair % kPersistStreams;
    const unsigned leaf_index = (wave % 8U) * 32U + lane;
    const bool norm_wave = pair < Count * kPersistStreams;
    const auto base_feature = static_cast<std::uint64_t>(stream) * kPersistHidden;
    float value[kLeaves], norm_w[kLeaves];
    if (norm_wave) {
        const auto* input = static_cast<const Storage*>(rows.input[at]);
#pragma unroll
        for (std::uint32_t k = 0; k < kLeaves; ++k) {
            const auto feature = leaf_index + k * kThreads;
            if constexpr (Inject) {
                const float result = qpf2k_fmac(load_activation(static_cast<const Storage*>(injected.hyper[at]), base_feature + feature),
                    injected.weights[at][stream], load_activation(static_cast<const Storage*>(injected.hidden[at]), feature));
                Storage stored;
                store_activation(&stored, result);
                const_cast<Storage*>(input)[base_feature + feature] = stored;
                value[k] = load_activation(&stored, 0U);
            } else value[k] = load_activation(input, base_feature + feature);
            norm_w[k] = load_norm_weight(norm, norm_type, stream * kPersistHidden + feature);
        }
    }
    // A: leaf l of a pair sums features l, l + 256, ... (the 256-thread kernel's leaves), the first wave
    // of the pair runs the tree (strides 128..32 in registers, 16..1 across lanes).
    if (norm_wave) {
        float sum = 0.0F;
#pragma unroll
        for (std::uint32_t k = 0; k < kLeaves; ++k) sum = qpf2k_fmac(sum, value[k], value[k]);
        leaf[half][leaf_index] = sum;
    }
    persist_lds_barrier();
    if (norm_wave && wave % 8U == 0U) {
        float l[8];
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) l[j] = leaf[half][lane + 32U * j];
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) l[j] = qpf2k_add(l[j + 4U], l[j]);
#pragma unroll
        for (unsigned j = 0; j < 2U; ++j) l[j] = qpf2k_add(l[j + 2U], l[j]);
        l[0] = qpf2k_add(l[1], l[0]);
        float v = l[0];
#pragma unroll
        for (unsigned d = 16U; d != 0U; d /= 2U) {
            const float partner = __shfl_down(v, d, 32);
            if (lane < d) v = qpf2k_add(partner, v);
        }
        if (lane == 0U) inverse_lds[half] = rsqrtf(v / static_cast<float>(hidden_size) + epsilon);
    }
    persist_lds_barrier();
    if (norm_wave) {
        const float inverse = inverse_lds[half];
#pragma unroll
        for (std::uint32_t k = 0; k < kLeaves; ++k)
            rows.normalized[at][base_feature + leaf_index + k * kThreads] = qpf2k_mul(qpf2k_scale<Storage>(inverse, value[k]), norm_w[k]);
    }
    persist_wait_stores();
    if (threadIdx.x == 0U) mark(1U);
    persist_grid_barrier(counter, base + kPersistBlocks, 0U);
    if (threadIdx.x == 0U) mark(2U);
    // B: 160 units of two low outputs and, with an inject matrix, 4 units of one inject output, each over
    // the 16 tiles (wave w = tile w), taken by the blocks from a launch-relative queue so that blocks whose
    // weights arrive late take fewer units. Every (row, output) keeps the tile's split-K lane chain
    // (s, then j ascending), its xor reduction and the ascending 16-tile sum; rows in groups of up to four
    // share each decoded coefficient.
    constexpr unsigned kUnitOutputs = kPersistUnitOutputs, kLowUnits = kPersistRank / kUnitOutputs;
    constexpr unsigned kGroup = Count < 4U ? Count : 4U;
    const unsigned units = kLowUnits + (inject != nullptr ? kPersistStreams : 0U);
    const unsigned tile = wave;
    for (;;) {
        if (threadIdx.x == 0U)
            unit_lds = __hip_atomic_fetch_add(queue, 1U, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) - queue_base;
        persist_lds_barrier();
        const unsigned unit = unit_lds;
        persist_lds_barrier(); // every wave has read the unit (and the previous unit's reductions)
        if (unit >= units) break;
        if (unit < kLowUnits) {
            const unsigned first = unit * kUnitOutputs;
            unsigned codes[kUnitOutputs][5];
            float scales[kUnitOutputs][5];
#pragma unroll
            for (unsigned r = 0; r < kUnitOutputs; ++r)
#pragma unroll
                for (unsigned s = 0; s < 5U; ++s) {
                    const auto* block = down + static_cast<std::uint64_t>(first + r) * kPersistRank * 34U +
                        static_cast<std::uint64_t>(tile * 20U + 4U * s + lane / 8U) * 34U;
                    const auto bits = *reinterpret_cast<const unsigned short*>(block);
                    const auto* words = reinterpret_cast<const unsigned short*>(block + 2U + 4U * (lane % 8U));
                    scales[r][s] = __half2float(__ushort_as_half(bits));
                    codes[r][s] = static_cast<unsigned>(words[0]) | (static_cast<unsigned>(words[1]) << 16U);
                }
#pragma unroll
            for (unsigned t0 = 0; t0 < Count; t0 += kGroup) {
                float sums[kGroup][kUnitOutputs];
#pragma unroll
                for (unsigned t = 0; t < kGroup; ++t)
#pragma unroll
                    for (unsigned r = 0; r < kUnitOutputs; ++r) sums[t][r] = 0.0F;
#pragma unroll
                for (unsigned s = 0; s < 5U; ++s) {
                    float4 x[kGroup];
#pragma unroll
                    for (unsigned t = 0; t < kGroup; ++t)
                        x[t] = *reinterpret_cast<const float4*>(rows.normalized[t0 + t] + tile * 640U + s * 128U + lane * 4U);
#pragma unroll
                    for (unsigned r = 0; r < kUnitOutputs; ++r)
#pragma unroll
                        for (unsigned j = 0; j < 4U; ++j) {
                            const unsigned byte = (codes[r][s] >> (j * 8U)) & 255U;
                            const int code = byte < 128U ? static_cast<int>(byte) : static_cast<int>(byte) - 256;
                            const float coefficient = scales[r][s] * static_cast<float>(code);
#pragma unroll
                            for (unsigned t = 0; t < kGroup; ++t) {
                                const float input = j == 0U ? x[t].x : j == 1U ? x[t].y : j == 2U ? x[t].z : x[t].w;
                                sums[t][r] = fmaf(input, coefficient, sums[t][r]);
                            }
                        }
                }
#pragma unroll
                for (unsigned t = 0; t < kGroup; ++t)
#pragma unroll
                    for (unsigned r = 0; r < kUnitOutputs; ++r) {
#pragma unroll
                        for (unsigned offset = 16U; offset != 0U; offset /= 2U)
                            sums[t][r] = qpf2k_add(sums[t][r], __shfl_xor(sums[t][r], static_cast<int>(offset), 32));
                        if (lane == 0U) red[t0 + t][tile][r] = sums[t][r];
                    }
            }
        } else {
            const unsigned output = unit - kLowUnits;
            float4 w[5];
#pragma unroll
            for (unsigned s = 0; s < 5U; ++s)
                w[s] = *reinterpret_cast<const float4*>(inject + static_cast<std::uint64_t>(output) * kPersistWidth +
                    tile * 640U + s * 128U + lane * 4U);
            for (unsigned t = 0; t < Count; ++t) {
                float4 x[5];
#pragma unroll
                for (unsigned s = 0; s < 5U; ++s)
                    x[s] = *reinterpret_cast<const float4*>(rows.normalized[t] + tile * 640U + s * 128U + lane * 4U);
                float sum = 0.0F;
#pragma unroll
                for (unsigned s = 0; s < 5U; ++s) {
                    sum = fmaf(x[s].x, w[s].x, sum);
                    sum = fmaf(x[s].y, w[s].y, sum);
                    sum = fmaf(x[s].z, w[s].z, sum);
                    sum = fmaf(x[s].w, w[s].w, sum);
                }
#pragma unroll
                for (unsigned offset = 16U; offset != 0U; offset /= 2U) sum = qpf2k_add(sum, __shfl_xor(sum, static_cast<int>(offset), 32));
                if (lane == 0U) red[t][tile][0] = sum;
            }
        }
        persist_lds_barrier();
        if (threadIdx.x < Count * kUnitOutputs) {
            const unsigned t = threadIdx.x / kUnitOutputs, r = threadIdx.x % kUnitOutputs;
            if (unit < kLowUnits || r == 0U) {
                float total = 0.0F;
#pragma unroll
                for (unsigned k = 0; k < 16U; ++k) total = qpf2k_add(total, red[t][k][r]);
                const float scaled = total / static_cast<float>(kPersistStreams);
                if (unit < kLowUnits) rows.low[t][unit * kUnitOutputs + r] = qpf2k_mul(scaled, sigmoid(scaled));
                else rows.injection[t][unit - kLowUnits] = 2.0F * sigmoid(scaled);
            }
        }
    }
    // C weights: lane's 340-byte up row (stream w % 4, feature 64 g + 32 (w / 4) + lane), requested now
    // so it streams through the reductions and the second barrier.
    const bool c_wave = wave < 8U;
    const unsigned c_stream = wave % 4U, c_group = (wave / 4U) % 2U, feature = 64U * g + 32U * c_group + lane;
    unsigned held[kWords];
    const auto load_up = [&] {
        const auto* row = up + (static_cast<std::uint64_t>(c_stream) * kPersistHidden + feature) * kRowBytes;
#pragma unroll
        for (std::uint32_t q = 0; q < kWords / 4U; ++q) {
            const uint4 v = *reinterpret_cast<const uint4*>(row + 16U * q);
            held[4U * q] = v.x; held[4U * q + 1U] = v.y; held[4U * q + 2U] = v.z; held[4U * q + 3U] = v.w;
        }
#pragma unroll
        for (std::uint32_t w = (kWords / 4U) * 4U; w < kWords; ++w) held[w] = *reinterpret_cast<const unsigned*>(row + 4U * w);
    };
    // up_late (STRIX_HC_PERSISTENT_UP_LATE=1): request the up rows after the second barrier instead, so they
    // do not compete with the split-K weights of slower blocks.
    if (c_wave && !up_late) load_up();
    if (threadIdx.x == 0U) mark(3U);
    persist_lds_barrier();
    persist_wait_stores();
    persist_grid_barrier(counter, base + 2U * kPersistBlocks, 8U); // wave 8 has no up rows in flight
    if (threadIdx.x == 0U) mark(4U);
    if (c_wave && up_late) load_up();
    // C: waves 8..15 (no loads in flight) stage the low rows; waves 0..7 run the up chains from LDS.
    if (!c_wave) {
        for (unsigned e = threadIdx.x - 256U; e < Count * kPersistRank; e += 256U)
            low_lds[e % kPersistRank][e / kPersistRank] = rows.low[e / kPersistRank][e % kPersistRank];
    }
    persist_lds_barrier();
    if (c_wave) {
        const auto byte_at = [&](std::uint32_t i) -> unsigned { return (held[i / 4U] >> (8U * (i % 4U))) & 255U; };
        float gate[Count];
#pragma unroll
        for (std::uint32_t t = 0; t < Count; ++t) gate[t] = 0.0F;
#pragma unroll
        for (std::uint32_t block = 0; block < kBlocks; ++block) {
            const auto bits = static_cast<unsigned short>(byte_at(block * 34U) | (byte_at(block * 34U + 1U) << 8U));
            const float scale = __half2float(__ushort_as_half(bits));
#pragma unroll
            for (std::uint32_t k = 0; k < 32U; ++k) {
                const float coefficient = scale * static_cast<float>(static_cast<signed char>(byte_at(block * 34U + 2U + k)));
#pragma unroll
                for (std::uint32_t t = 0; t < Count; ++t) gate[t] += low_lds[block * 32U + k][t] * coefficient;
            }
        }
#pragma unroll
        for (std::uint32_t t = 0; t < Count; ++t) sig[t][c_stream][32U * c_group + lane] = sigmoid(gate[t]);
    }
    if (threadIdx.x == 0U) mark(5U);
    persist_lds_barrier();
    if (c_wave)
        for (std::uint32_t t = c_stream; t < Count; t += kPersistStreams) {
            float mixed = 0.0F;
            for (std::uint32_t s2 = 0; s2 < kPersistStreams; ++s2)
                mixed = qpf2k_fmac(mixed, rows.normalized[t][static_cast<std::uint64_t>(s2) * kPersistHidden + feature],
                    sig[t][s2][32U * c_group + lane]);
            store_activation(&static_cast<Storage*>(rows.output[t])[feature], mixed / static_cast<float>(kPersistStreams));
        }
    if (threadIdx.x == 0U) mark(6U);
    // Block 0 also accumulates its phase spans per row count: [A + barrier, B + barrier, C, launches].
    if (trace && g == 0U && threadIdx.x == 0U) {
        auto* sums = trace + kPersistBlocks * 8U + (Count == 2U ? 0U : Count == 4U ? 4U : 8U);
        sums[0] += trace[2] - trace[0]; sums[1] += trace[4] - trace[2]; sums[2] += trace[6] - trace[4]; sums[3] += 1U;
    }
}

// Persistent HC mix with a static split-K phase (STRIX_HC_PERSISTENT_B=<variant>, default off): phase B without
// the work queue. Block g owns the low outputs 8g..8g+7 (four units of two) and, with an inject matrix and g < 4,
// inject output g; wave w = tile w. The queue form pays, per unit, an atomic, two barriers, the full weight latency
// and a re-read of every row's tile from L2; here each wave reads its tile of the normalised rows once (registers,
// up to four rows), keeps the down weights of the next Prefetch units in flight while it reduces the current one,
// and all tile partials meet in LDS behind a single barrier. Early requests the first units' weights during A,
// UpEarly the up rows before the last unit. Coalesced reads each (output, tile) run of 680 bytes with contiguous
// 8-byte loads and hands every lane its scale and code bytes through LDS: on the fine-grained (MTYPE UC) weight
// pages nothing merges the three 2-byte loads per lane of the lane map, so each of them fetches the lines again.
// UpCoalesced likewise reads each C wave's 32 up rows (10880 contiguous bytes) as uint4 and transposes them through
// LDS in four 8-row quarters into the same per-lane rows. Every value's expression is the queue form's.
// The split-K tile reduction sum + xor(sum, 16), ..., + xor(sum, 1) with the lane exchange in DPP row_xmask and
// permlanex16 (as the wide kernels' wide_xor) instead of __shfl_xor's ds_bpermute: same operands, same order, no
// LDS round trip per level (five dependent ones per reduction, measured as the bulk of the static phase B).
template <unsigned D> __device__ __forceinline__ float static_xor(float value) {
    const int bits = __float_as_int(value);
    if constexpr (D == 16U) {
        const auto u = static_cast<unsigned>(bits);
        return __uint_as_float(__builtin_amdgcn_permlanex16(u, u, 0x76543210U, 0xfedcba98U, false, false));
    } else return __int_as_float(__builtin_amdgcn_mov_dpp(bits, 0x160 + static_cast<int>(D), 0xf, 0xf, false));
}
// Pinned like the queue form's reduction (own value first), so the exchange cannot fold into a DPP source operand.
__device__ __forceinline__ float static_xor_sum(float sum) {
    sum = qpf2k_add(sum, static_xor<16U>(sum));
    sum = qpf2k_add(sum, static_xor<8U>(sum));
    sum = qpf2k_add(sum, static_xor<4U>(sum));
    sum = qpf2k_add(sum, static_xor<2U>(sum));
    return qpf2k_add(sum, static_xor<1U>(sum));
}
constexpr std::uint32_t kStaticOutputs = kPersistRank / kPersistBlocks, kStaticUnits = kStaticOutputs / kPersistUnitOutputs;
static_assert(kStaticOutputs == 8U && kStaticUnits == 4U && kPersistStreams <= kPersistBlocks);
template <typename Storage, std::uint32_t Count, bool Inject, unsigned Prefetch, bool Early, bool UpEarly, bool Coalesced = false, bool UpCoalesced = false>
__global__ __launch_bounds__(512) __attribute__((amdgpu_waves_per_eu(8))) void rows_static_hyper_mix_kernel(HyperRows rows,
    const void* __restrict__ norm, std::uint32_t norm_type, float epsilon, std::uint32_t hidden_size, const unsigned char* __restrict__ down,
    const float* __restrict__ inject, const unsigned char* __restrict__ up, unsigned* counter, unsigned base,
    PersistInjectRows injected, unsigned long long* trace) {
    static_assert(Prefetch >= 1U && Prefetch <= kStaticUnits);
    constexpr std::uint32_t kBlocks = 10U, kRowBytes = kBlocks * 34U, kWords = kRowBytes / 4U;
    constexpr std::uint32_t kLeaves = kPersistHidden / kThreads;
    constexpr unsigned kUnitOutputs = kPersistUnitOutputs, kGroup = Count < 4U ? Count : 4U;
    __shared__ float leaf[2][kThreads];
    __shared__ float inverse_lds[2];
    __shared__ float red[Count][16][kStaticOutputs];
    __shared__ float red_inject[Count][16];
    __shared__ __align__(16) float low_lds[kPersistRank][Count];
    __shared__ float sig[Count][kPersistStreams][64];
    constexpr unsigned kRun = 20U * 34U / 8U; // one (output, tile) run: 85 uint2
    __shared__ uint2 stage[Coalesced || UpCoalesced ? 16U : 1U][kPersistUnitOutputs][kRun];
    static_assert(!UpCoalesced || !UpEarly);
    const unsigned g = blockIdx.x, wave = threadIdx.x / 32U, lane = threadIdx.x & 31U, tile = wave;
    const bool inject_block = inject != nullptr && g < kPersistStreams;
    const auto mark = [&](unsigned k) { if (trace) trace[g * 8U + k] = static_cast<unsigned long long>(wall_clock64()); };
    if (threadIdx.x == 0U) mark(0U);
    // B weights: unit u of block g = low outputs 8g + 2u, 8g + 2u + 1 on tile `tile` (the queue form's lane map).
    unsigned codes[Coalesced ? 1U : Prefetch][kUnitOutputs][5];
    unsigned bits[Coalesced ? 1U : Prefetch][5]; // the two outputs' fp16 scale bits, output r in bits 16r..16r+15
    uint2 raw[Coalesced ? Prefetch : 1U][kUnitOutputs][3];
    float4 inject_w[5];
    const auto load_unit = [&](unsigned u, unsigned slot) {
        const unsigned first = g * kStaticOutputs + u * kUnitOutputs;
        if constexpr (Coalesced) {
#pragma unroll
            for (unsigned r = 0; r < kUnitOutputs; ++r) {
                const auto* run = reinterpret_cast<const uint2*>(down + (static_cast<std::uint64_t>(first + r) * kPersistRank + tile * 20U) * 34U);
#pragma unroll
                for (unsigned k = 0; k < 3U; ++k) if (lane + 32U * k < kRun) raw[slot][r][k] = run[lane + 32U * k];
            }
            return;
        }
#pragma unroll
        for (unsigned r = 0; r < kUnitOutputs; ++r)
#pragma unroll
            for (unsigned s = 0; s < 5U; ++s) {
                const auto* block = down + static_cast<std::uint64_t>(first + r) * kPersistRank * 34U +
                    static_cast<std::uint64_t>(tile * 20U + 4U * s + lane / 8U) * 34U;
                const unsigned scale_bits = *reinterpret_cast<const unsigned short*>(block);
                bits[slot][s] = r == 0U ? scale_bits : bits[slot][s] | (scale_bits << 16U);
                const auto* words = reinterpret_cast<const unsigned short*>(block + 2U + 4U * (lane % 8U));
                codes[slot][r][s] = static_cast<unsigned>(words[0]) | (static_cast<unsigned>(words[1]) << 16U);
            }
    };
    // The lane map's scale and code words of a unit (Coalesced: from the staged runs; the wave's own LDS rows).
    unsigned ubits[5], ucodes[kUnitOutputs][5];
    const auto unpack = [&](unsigned slot) {
        if constexpr (Coalesced) {
#pragma unroll
            for (unsigned r = 0; r < kUnitOutputs; ++r)
#pragma unroll
                for (unsigned k = 0; k < 3U; ++k) if (lane + 32U * k < kRun) stage[wave][r][lane + 32U * k] = raw[slot][r][k];
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
#pragma unroll
            for (unsigned r = 0; r < kUnitOutputs; ++r) {
                const auto* bytes = reinterpret_cast<const unsigned char*>(&stage[wave][r][0]);
#pragma unroll
                for (unsigned s = 0; s < 5U; ++s) {
                    const unsigned at = (4U * s + lane / 8U) * 34U;
                    const unsigned scale_bits = *reinterpret_cast<const unsigned short*>(bytes + at);
                    ubits[s] = r == 0U ? scale_bits : ubits[s] | (scale_bits << 16U);
                    const auto* words = reinterpret_cast<const unsigned short*>(bytes + at + 2U + 4U * (lane % 8U));
                    ucodes[r][s] = static_cast<unsigned>(words[0]) | (static_cast<unsigned>(words[1]) << 16U);
                }
            }
            // The next unit's stores to this wave's rows come after every lane's reads.
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
        } else {
#pragma unroll
            for (unsigned s = 0; s < 5U; ++s) {
                ubits[s] = bits[slot][s];
#pragma unroll
                for (unsigned r = 0; r < kUnitOutputs; ++r) ucodes[r][s] = codes[slot][r][s];
            }
        }
    };
    const auto load_first = [&] {
#pragma unroll
        for (unsigned u = 0; u < Prefetch; ++u) load_unit(u, u);
    };
    // The inject weights take the first weight slot that no later unit refills.
    const auto load_inject = [&] {
        if (inject_block) {
#pragma unroll
            for (unsigned s = 0; s < 5U; ++s)
                inject_w[s] = *reinterpret_cast<const float4*>(inject + static_cast<std::uint64_t>(g) * kPersistWidth +
                    tile * 640U + s * 128U + lane * 4U);
        }
    };
    // A: as the queue form (block g normalises (row, stream) pairs 2g and 2g + 1).
    const unsigned half = wave / 8U, pair = 2U * g + half, at = pair / kPersistStreams, stream = pair % kPersistStreams;
    const unsigned leaf_index = (wave % 8U) * 32U + lane;
    const bool norm_wave = pair < Count * kPersistStreams;
    const auto base_feature = static_cast<std::uint64_t>(stream) * kPersistHidden;
    float value[kLeaves], norm_w[kLeaves];
    if (norm_wave) {
        const auto* input = static_cast<const Storage*>(rows.input[at]);
#pragma unroll
        for (std::uint32_t k = 0; k < kLeaves; ++k) {
            const auto feature = leaf_index + k * kThreads;
            if constexpr (Inject) {
                const float result = qpf2k_fmac(load_activation(static_cast<const Storage*>(injected.hyper[at]), base_feature + feature),
                    injected.weights[at][stream], load_activation(static_cast<const Storage*>(injected.hidden[at]), feature));
                Storage stored;
                store_activation(&stored, result);
                const_cast<Storage*>(input)[base_feature + feature] = stored;
                value[k] = load_activation(&stored, 0U);
            } else value[k] = load_activation(input, base_feature + feature);
            norm_w[k] = load_norm_weight(norm, norm_type, stream * kPersistHidden + feature);
        }
    }
    if constexpr (Early) load_first();
    if (norm_wave) {
        float sum = 0.0F;
#pragma unroll
        for (std::uint32_t k = 0; k < kLeaves; ++k) sum = qpf2k_fmac(sum, value[k], value[k]);
        leaf[half][leaf_index] = sum;
    }
    persist_lds_barrier();
    if (norm_wave && wave % 8U == 0U) {
        float l[8];
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) l[j] = leaf[half][lane + 32U * j];
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) l[j] = qpf2k_add(l[j + 4U], l[j]);
#pragma unroll
        for (unsigned j = 0; j < 2U; ++j) l[j] = qpf2k_add(l[j + 2U], l[j]);
        l[0] = qpf2k_add(l[1], l[0]);
        float v = l[0];
#pragma unroll
        for (unsigned d = 16U; d != 0U; d /= 2U) {
            const float partner = __shfl_down(v, d, 32);
            if (lane < d) v = qpf2k_add(partner, v);
        }
        if (lane == 0U) inverse_lds[half] = rsqrtf(v / static_cast<float>(hidden_size) + epsilon);
    }
    persist_lds_barrier();
    if (norm_wave) {
        const float inverse = inverse_lds[half];
#pragma unroll
        for (std::uint32_t k = 0; k < kLeaves; ++k)
            rows.normalized[at][base_feature + leaf_index + k * kThreads] = qpf2k_mul(qpf2k_scale<Storage>(inverse, value[k]), norm_w[k]);
    }
    persist_wait_stores();
    if (threadIdx.x == 0U) mark(1U);
    persist_grid_barrier(counter, base + kPersistBlocks, 0U);
    if (threadIdx.x == 0U) mark(2U);
    // C weights (as the queue form): lane's 340-byte up row.
    const bool c_wave = wave < 8U;
    const unsigned c_stream = wave % 4U, c_group = (wave / 4U) % 2U, feature = 64U * g + 32U * c_group + lane;
    unsigned held[kWords];
    constexpr unsigned kQuarter = 8U * kRowBytes / 16U; // 170 uint4: eight rows
    static_assert(kQuarter == 2U * kRun && 8U * kRowBytes % 16U == 0U);
    uint4 uq0[6], uq1[6], uq2[6], uq3[6]; // the four quarters (separate arrays: a 2-D one stays in scratch)
    const auto quarter = [&]<unsigned Q>() -> uint4(&)[6] {
        if constexpr (Q == 0U) return uq0; else if constexpr (Q == 1U) return uq1; else if constexpr (Q == 2U) return uq2; else return uq3;
    };
    const auto load_up_coalesced = [&] {
        const auto* rows0 = reinterpret_cast<const uint4*>(up + (static_cast<std::uint64_t>(c_stream) * kPersistHidden +
            64U * g + 32U * c_group) * kRowBytes);
        const auto one = [&]<unsigned Q>() {
            auto& dst = quarter.template operator()<Q>();
#pragma unroll
            for (unsigned k = 0; k < 6U; ++k) if (lane + 32U * k < kQuarter) dst[k] = rows0[kQuarter * Q + lane + 32U * k];
        };
        one.template operator()<0U>(); one.template operator()<1U>(); one.template operator()<2U>(); one.template operator()<3U>();
    };
    // Quarter q: the wave stores its 170 uint4 in its LDS rows, lanes 8q..8q+7 take their 340-byte row.
    const auto transpose_up = [&] {
        auto* region = reinterpret_cast<uint4*>(&stage[wave][0][0]);
        const auto one = [&]<unsigned Q>() {
            const auto& src = quarter.template operator()<Q>();
#pragma unroll
            for (unsigned k = 0; k < 6U; ++k) if (lane + 32U * k < kQuarter) region[lane + 32U * k] = src[k];
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
            {   // every lane reads, the lanes of quarter Q keep it (no divergent writes of the row registers)
                const auto* row = reinterpret_cast<const unsigned*>(reinterpret_cast<const unsigned char*>(region) + (lane % 8U) * kRowBytes);
                const bool mine = lane / 8U == Q;
#pragma unroll
                for (unsigned w = 0; w < kWords; ++w) { const unsigned v = row[w]; held[w] = mine ? v : held[w]; }
            }
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
        };
        one.template operator()<0U>(); one.template operator()<1U>(); one.template operator()<2U>(); one.template operator()<3U>();
    };
    const auto load_up = [&] {
        if constexpr (UpCoalesced) { load_up_coalesced(); return; }
        const auto* row = up + (static_cast<std::uint64_t>(c_stream) * kPersistHidden + feature) * kRowBytes;
#pragma unroll
        for (std::uint32_t q = 0; q < kWords / 4U; ++q) {
            const uint4 v = *reinterpret_cast<const uint4*>(row + 16U * q);
            held[4U * q] = v.x; held[4U * q + 1U] = v.y; held[4U * q + 2U] = v.z; held[4U * q + 3U] = v.w;
        }
#pragma unroll
        for (std::uint32_t w = (kWords / 4U) * 4U; w < kWords; ++w) held[w] = *reinterpret_cast<const unsigned*>(row + 4U * w);
    };
    // B: the tile of the group's rows, read once per group (once per launch up to four rows).
    float4 x[kGroup][5];
    const auto load_x = [&](unsigned t0) {
#pragma unroll
        for (unsigned t = 0; t < kGroup; ++t)
#pragma unroll
            for (unsigned s = 0; s < 5U; ++s)
                x[t][s] = *reinterpret_cast<const float4*>(rows.normalized[t0 + t] + tile * 640U + s * 128U + lane * 4U);
    };
    // The tile reads (L2) go ahead of the first weight unit: a load queued behind the weight stream waits for it.
    if constexpr (Count == kGroup) load_x(0U);
    if constexpr (!Early) load_first();
    // More than four rows with every unit's weights held: groups outside, units inside, so that only one
    // group's tile is live (unit-outer, the compiler keeps both groups' loads and spills).
    constexpr bool kGroupOuter = Count != kGroup && Prefetch == kStaticUnits;
#pragma unroll
    for (unsigned it = 0; it < kStaticUnits * (Count / kGroup); ++it) {
        const unsigned u = kGroupOuter ? it % kStaticUnits : it / (Count / kGroup);
        const unsigned t0 = kGroup * (kGroupOuter ? it / kStaticUnits : it % (Count / kGroup));
        const bool unit_end = kGroupOuter ? it + kStaticUnits >= kStaticUnits * (Count / kGroup) : t0 + kGroup == Count;
        const unsigned slot = u % Prefetch;
        if constexpr (UpEarly) if (u + 1U == kStaticUnits && t0 + kGroup == Count && c_wave) load_up();
        {
            if constexpr (Count != kGroup) if (kGroupOuter ? u == 0U : true) {
                // Reload, never merge with an earlier load of the same rows (that keeps both groups live).
                asm volatile("" ::: "memory");
                load_x(t0);
            }
            if (kGroupOuter ? true : t0 == 0U) unpack(slot);
            float sums[kGroup][kUnitOutputs];
#pragma unroll
            for (unsigned t = 0; t < kGroup; ++t)
#pragma unroll
                for (unsigned r = 0; r < kUnitOutputs; ++r) sums[t][r] = 0.0F;
#pragma unroll
            for (unsigned s = 0; s < 5U; ++s)
#pragma unroll
                for (unsigned r = 0; r < kUnitOutputs; ++r) {
                    const float scale = __half2float(__ushort_as_half(static_cast<unsigned short>(ubits[s] >> (16U * r))));
#pragma unroll
                    for (unsigned j = 0; j < 4U; ++j) {
                        const unsigned byte = (ucodes[r][s] >> (j * 8U)) & 255U;
                        const int code = byte < 128U ? static_cast<int>(byte) : static_cast<int>(byte) - 256;
                        const float coefficient = scale * static_cast<float>(code);
#pragma unroll
                        for (unsigned t = 0; t < kGroup; ++t) {
                            const float input = j == 0U ? x[t][s].x : j == 1U ? x[t][s].y : j == 2U ? x[t][s].z : x[t][s].w;
                            sums[t][r] = fmaf(input, coefficient, sums[t][r]);
                        }
                    }
                }
#pragma unroll
            for (unsigned t = 0; t < kGroup; ++t)
#pragma unroll
                for (unsigned r = 0; r < kUnitOutputs; ++r) {
                    sums[t][r] = static_xor_sum(sums[t][r]);
                    if (lane == 0U) red[t0 + t][tile][u * kUnitOutputs + r] = sums[t][r];
                }
        }
        if (it == 0U && threadIdx.x == 0U) mark(7U); // the first unit reduced (tile and weights arrived)
        if (unit_end && u + Prefetch < kStaticUnits) load_unit(u + Prefetch, slot);
        if (unit_end && u + Prefetch == kStaticUnits) load_inject();
    }
    if (inject_block) {
#pragma unroll
        for (unsigned t0 = 0; t0 < Count; t0 += kGroup) {
            if constexpr (Count != kGroup) load_x(t0);
#pragma unroll
            for (unsigned t = 0; t < kGroup; ++t) {
                float sum = 0.0F;
#pragma unroll
                for (unsigned s = 0; s < 5U; ++s) {
                    sum = fmaf(x[t][s].x, inject_w[s].x, sum);
                    sum = fmaf(x[t][s].y, inject_w[s].y, sum);
                    sum = fmaf(x[t][s].z, inject_w[s].z, sum);
                    sum = fmaf(x[t][s].w, inject_w[s].w, sum);
                }
                sum = static_xor_sum(sum);
                if (lane == 0U) red_inject[t0 + t][tile] = sum;
            }
        }
    }
    // STRIX_HC_PERSISTENT_TRACE: when each wave finished its units (after the block marks and the row sums).
    if (trace && lane == 0U) trace[kPersistBlocks * 8U + 16U + g * 16U + wave] = static_cast<unsigned long long>(wall_clock64());
    if constexpr (!UpEarly) if (c_wave) load_up();
    persist_lds_barrier();
    if (threadIdx.x < Count * kStaticOutputs) {
        const unsigned t = threadIdx.x / kStaticOutputs, r = threadIdx.x % kStaticOutputs;
        float total = 0.0F;
#pragma unroll
        for (unsigned k = 0; k < 16U; ++k) total = qpf2k_add(total, red[t][k][r]);
        const float scaled = total / static_cast<float>(kPersistStreams);
        rows.low[t][g * kStaticOutputs + r] = qpf2k_mul(scaled, sigmoid(scaled));
    } else if (inject_block && threadIdx.x - Count * kStaticOutputs < Count) {
        const unsigned t = threadIdx.x - Count * kStaticOutputs;
        float total = 0.0F;
#pragma unroll
        for (unsigned k = 0; k < 16U; ++k) total = qpf2k_add(total, red_inject[t][k]);
        const float scaled = total / static_cast<float>(kPersistStreams);
        rows.injection[t][g] = 2.0F * sigmoid(scaled);
    }
    if (threadIdx.x == 0U) mark(3U);
    persist_wait_stores();
    persist_grid_barrier(counter, base + 2U * kPersistBlocks, 8U); // wave 8 has no up rows in flight
    if (threadIdx.x == 0U) mark(4U);
    if constexpr (UpCoalesced) if (c_wave) transpose_up();
    // C: as the queue form.
    if (!c_wave) {
        for (unsigned e = threadIdx.x - 256U; e < Count * kPersistRank; e += 256U)
            low_lds[e % kPersistRank][e / kPersistRank] = rows.low[e / kPersistRank][e % kPersistRank];
    }
    persist_lds_barrier();
    if (c_wave) {
        const auto byte_at = [&](std::uint32_t i) -> unsigned { return (held[i / 4U] >> (8U * (i % 4U))) & 255U; };
        float gate[Count];
#pragma unroll
        for (std::uint32_t t = 0; t < Count; ++t) gate[t] = 0.0F;
#pragma unroll
        for (std::uint32_t block = 0; block < kBlocks; ++block) {
            const auto scale_bits = static_cast<unsigned short>(byte_at(block * 34U) | (byte_at(block * 34U + 1U) << 8U));
            const float scale = __half2float(__ushort_as_half(scale_bits));
#pragma unroll
            for (std::uint32_t k = 0; k < 32U; ++k) {
                const float coefficient = scale * static_cast<float>(static_cast<signed char>(byte_at(block * 34U + 2U + k)));
#pragma unroll
                for (std::uint32_t t = 0; t < Count; ++t) gate[t] += low_lds[block * 32U + k][t] * coefficient;
            }
        }
#pragma unroll
        for (std::uint32_t t = 0; t < Count; ++t) sig[t][c_stream][32U * c_group + lane] = sigmoid(gate[t]);
    }
    if (threadIdx.x == 0U) mark(5U);
    persist_lds_barrier();
    if (c_wave)
        for (std::uint32_t t = c_stream; t < Count; t += kPersistStreams) {
            float mixed = 0.0F;
            for (std::uint32_t s2 = 0; s2 < kPersistStreams; ++s2)
                mixed = qpf2k_fmac(mixed, rows.normalized[t][static_cast<std::uint64_t>(s2) * kPersistHidden + feature],
                    sig[t][s2][32U * c_group + lane]);
            store_activation(&static_cast<Storage*>(rows.output[t])[feature], mixed / static_cast<float>(kPersistStreams));
        }
    if (threadIdx.x == 0U) mark(6U);
    if (trace && g == 0U && threadIdx.x == 0U) {
        auto* sums = trace + kPersistBlocks * 8U + (Count == 2U ? 0U : Count == 4U ? 4U : 8U);
        sums[0] += trace[2] - trace[0]; sums[1] += trace[4] - trace[2]; sums[2] += trace[6] - trace[4]; sums[3] += 1U;
    }
}

// Prefill wide rows: every row of a prefill step in one launch, row r of each view at
// base + r * stride bytes. Blocks cover many rows per weight read; every (row, output)
// keeps the window kernels' chains, lane tree and block order.
struct HyperWideRows {
    float* normalized; // row r: normalized[0 .. width), then low[0 .. rank)
    float* injection;
    void* output;
    std::uint64_t normalized_stride, injection_stride, output_stride;
    std::uint32_t count;
    // Optional transposed copy of the low values (low_t[k * low_t_pitch + r]) for the wide2 output kernel;
    // written by the wide2 projection when set.
    float* low_t = nullptr;
    std::uint32_t low_t_pitch = 0;
    // STRIX_QPF2K_HC_LOWBAND bit 0: low_t in 512-row bands, low_t[(r / 512) * 320 * 512 + k * 512 + r % 512] (low_t_pitch
    // 512); only the qpf2k projection and the output v3 band kernel use this layout.
    bool low_t_band = false;
    // Optional FP16 copy of the normalized rows (row r at normalized16 + r * normalized16_pitch halves), written by the
    // fast-profile HC norm and read by the FP16-input WMMA projection.
    const _Float16* normalized16 = nullptr;
    std::uint32_t normalized16_pitch = 0;
    bool normalized16_only = false;  // the FP32 normalized values were not written (only low values live there)
    // Fast profile, STRIX_HCW_NORMFREE: the norm writes only the stream inverses (inverse[4 r + s]); the WMMA projection
    // and the lane-chain output recompute the normalized values from the input streams (row r at input + r * input_stride)
    // with the norm's expression, input * inverse * weight (FP32 weights [stream][hidden]).
    const void* input = nullptr;
    std::uint64_t input_stride = 0;
    const float* inverse = nullptr;
    const float* norm_weight = nullptr;
};
template <typename T> __device__ __forceinline__ T* wide_row(T* base, std::uint64_t stride, std::uint32_t row) {
    return reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(base) + static_cast<std::uint64_t>(row) * stride);
}

// Split-K lane tree over N partials at once: level d pairs lanes l and l ^ d exactly like the
// window xor reduction, each lane keeping half of the values. Afterwards lane l holds the
// N / 32 sums of partials wide_tree_first(l) + i.
// Lane l ^ D's value (wave32): DPP row_xmask inside 16-lane rows, permlanex16 across them.
template <unsigned D> __device__ __forceinline__ float wide_xor(float value) {
    const int bits = __float_as_int(value);
    if constexpr (D == 16U) {
        const auto u = static_cast<unsigned>(bits);
        return __uint_as_float(__builtin_amdgcn_permlanex16(u, u, 0x76543210U, 0xfedcba98U, false, false));
    }
    else return __int_as_float(__builtin_amdgcn_mov_dpp(bits, 0x160 + static_cast<int>(D), 0xf, 0xf, false));
}
template <unsigned N, unsigned Level = 0U>
__device__ __forceinline__ void wide_lane_tree(float (&v)[N], unsigned lane) {
    static_assert(N >= 32U && (N & (N - 1U)) == 0U);
    constexpr unsigned d = 16U >> Level, half = (N >> Level) / 2U;
    const bool high = (lane & d) != 0U;
#pragma unroll
    for (unsigned i = 0; i < half; ++i) {
        const float keep = high ? v[i + half] : v[i];
        const float send = high ? v[i] : v[i + half];
        v[i] = keep + wide_xor<d>(send);
    }
    if constexpr (Level < 4U) wide_lane_tree<N, Level + 1U>(v, lane);
}
template <unsigned N> __device__ __forceinline__ unsigned wide_tree_first(unsigned lane) {
    unsigned first = 0;
#pragma unroll
    for (unsigned level = 0; level < 5U; ++level) if (lane & (16U >> level)) first += (N >> level) / 2U;
    return first;
}

// Split-K projection over Rows x 32 outputs per block: wave w owns 4 outputs of every block
// row, lane l the window lane-l chain of each 640-feature tile (s, then j order), tile k's
// partials go through the lane tree and are added to the running total in k order. The
// 128-feature input chunks of the block rows are staged once in LDS for all waves.
constexpr unsigned kWideProjectionOutputs = 4U;
template <unsigned Rows, unsigned Waves, bool Injection>
__device__ __forceinline__ void wide_splitk_projection_body(const HyperWideRows& rows,
    const void* __restrict__ weights, unsigned streams, unsigned first, unsigned valid, float4 (*xs)[Rows][32]) {
    constexpr unsigned O = kWideProjectionOutputs, N = Rows * O, Chunks = 80U;
    constexpr unsigned Staged = Rows * 32U / (32U * Waves);
    const unsigned lane = threadIdx.x % 32U, row0 = blockIdx.y * Rows;
    const auto x_at = [&](unsigned chunk, unsigned index) {
        const unsigned t = index / 32U, l = index % 32U;
        if (row0 + t >= rows.count) return make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        return *reinterpret_cast<const float4*>(wide_row(rows.normalized, rows.normalized_stride, row0 + t) +
            (chunk / 5U) * 640U + (chunk % 5U) * 128U + l * 4U);
    };
    float4 w[Injection ? O : 1U];
    unsigned packed[Injection ? 1U : O];
    float scales[Injection ? 1U : O];
    const auto load_weights = [&](unsigned chunk) {
        const unsigned tile = chunk / 5U, s = chunk % 5U;
#pragma unroll
        for (unsigned o = 0; o < O; ++o) if (o < valid) {
            if constexpr (Injection) {
                w[o] = *reinterpret_cast<const float4*>(static_cast<const float*>(weights) +
                    static_cast<std::uint64_t>(first + o) * 10240U + tile * 640U + s * 128U + lane * 4U);
            } else {
                const auto* block = static_cast<const unsigned char*>(weights) +
                    static_cast<std::uint64_t>(first + o) * 320U * 34U + static_cast<std::uint64_t>(tile * 20U + 4U * s + lane / 8U) * 34U;
                const auto bits = *reinterpret_cast<const unsigned short*>(block);
                const auto* codes = reinterpret_cast<const unsigned short*>(block + 2U + 4U * (lane % 8U));
                scales[o] = __half2float(__ushort_as_half(bits));
                packed[o] = static_cast<unsigned>(codes[0]) | (static_cast<unsigned>(codes[1]) << 16U);
            }
        }
    };
    float acc[N], total[N / 32U];
#pragma unroll
    for (unsigned i = 0; i < N; ++i) acc[i] = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < N / 32U; ++i) total[i] = 0.0F;
#pragma unroll
    for (unsigned m = 0; m < Staged; ++m) {
        const unsigned index = threadIdx.x + m * 32U * Waves;
        xs[0][index / 32U][index % 32U] = x_at(0U, index);
    }
    if (valid) load_weights(0U);
    __syncthreads();
    for (unsigned chunk = 0; chunk < Chunks; ++chunk) {
        float4 next[Staged];
        if (chunk + 1U < Chunks) {
#pragma unroll
            for (unsigned m = 0; m < Staged; ++m) next[m] = x_at(chunk + 1U, threadIdx.x + m * 32U * Waves);
        }
        if (valid) {
            float coefficient[O][4];
#pragma unroll
            for (unsigned o = 0; o < O; ++o) {
                if constexpr (Injection) {
                    coefficient[o][0] = w[o].x; coefficient[o][1] = w[o].y; coefficient[o][2] = w[o].z; coefficient[o][3] = w[o].w;
                } else {
#pragma unroll
                    for (unsigned j = 0; j < 4U; ++j) {
                        const unsigned byte = (packed[o] >> (j * 8U)) & 255U;
                        const int code = byte < 128U ? static_cast<int>(byte) : static_cast<int>(byte) - 256;
                        coefficient[o][j] = scales[o] * static_cast<float>(code);
                    }
                }
            }
            if (chunk + 1U < Chunks) load_weights(chunk + 1U);
#pragma unroll
            for (unsigned t = 0; t < Rows; ++t) {
                const float4 x = xs[chunk & 1U][t][lane];
                const float inputs[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j)
#pragma unroll
                    for (unsigned o = 0; o < O; ++o)
                        acc[t * O + o] = fmaf(inputs[j], coefficient[o][j], acc[t * O + o]);
            }
            if (chunk % 5U == 4U) {
                wide_lane_tree(acc, lane);
#pragma unroll
                for (unsigned i = 0; i < N / 32U; ++i) total[i] = total[i] + acc[i];
#pragma unroll
                for (unsigned i = 0; i < N; ++i) acc[i] = 0.0F;
            }
        }
        if (chunk + 1U < Chunks) {
#pragma unroll
            for (unsigned m = 0; m < Staged; ++m) {
                const unsigned index = threadIdx.x + m * 32U * Waves;
                xs[(chunk + 1U) & 1U][index / 32U][index % 32U] = next[m];
            }
        }
        __syncthreads();
    }
    if (!valid) return;
    const unsigned at = wide_tree_first<N>(lane);
#pragma unroll
    for (unsigned i = 0; i < N / 32U; ++i) {
        const unsigned t = (at + i) / O, o = (at + i) % O;
        if (row0 + t >= rows.count || o >= valid) continue;
        const float scaled = total[i] / static_cast<float>(streams);
        if constexpr (Injection) wide_row(rows.injection, rows.injection_stride, row0 + t)[first + o] = 2.0F * sigmoid(scaled);
        else wide_row(rows.normalized, rows.normalized_stride, row0 + t)[10240U + first + o] = scaled * sigmoid(scaled);
    }
}

// Split-K projection, register-permuted form (same chains, same tree, same sums):
// hardware lane L runs window lane l = bitrev5(L), so tree level d = 16, 8, 4, 2, 1 pairs L with
// L ^ 1, 2, 4, 8, 16. A wave owns 8 outputs of 16 rows (128 accumulators per lane). Register
// k = 8 q + p of lane L holds (row q ^ (l >> 1), output p ^ 4 (l & 1)): lanes whose bit is set at
// a tree level hold the two halves of that level swapped, so the level is one partner add per
// surviving pair, v[i] = v[i] + partner(v[i + half]) (the window's keep + xor(send): each lane
// adds its own kept value and its partner's value of the same (row, output)). After the five
// levels lane L holds (row l >> 1, outputs 4 (l & 1) + i) in v[0..3]. The 128-feature chunk of
// the 16 rows is staged in LDS at [q][L] in the lane's permuted row order.
__device__ __forceinline__ unsigned bitrev5(unsigned v) { return __builtin_bitreverse32(v) >> 27U; }
template <unsigned D> __device__ __forceinline__ float wide2_partner(float value) {
    if constexpr (D == 16U) {
        const auto u = __float_as_uint(value);
        return __uint_as_float(__builtin_amdgcn_permlanex16(u, u, 0x76543210U, 0xfedcba98U, false, false));
    } else return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(value), 0x160 + static_cast<int>(D), 0xf, 0xf, false));
}
template <unsigned N, unsigned Level = 0U> __device__ __forceinline__ void wide2_lane_tree(float (&v)[N]) {
    constexpr unsigned half = (N >> Level) / 2U;
#pragma unroll
    for (unsigned i = 0; i < half; ++i) v[i] = v[i] + wide2_partner<1U << Level>(v[i + half]);
    if constexpr (Level < 4U) wide2_lane_tree<N, Level + 1U>(v);
}
#include "hip_hc_wide2_vopd.inc"
constexpr unsigned kWide2Rows = 16U, kWide2Outputs = 8U;
// Lane-ordered copy of the Q8_0 down weights (rank 320 x 10240) the wide2 projection reads with four
// coalesced 16-byte loads per chunk instead of 24 two-byte gathers. Per (chunk c, output group g of 8,
// lane L of window lane l = bitrev5(L)): the four code bytes (features 128 c + 4 l .. + 3) of outputs
// 8 g + (p ^ 4 (l & 1)), p = 0..7, as 8 dwords; per (c, g, Q8 block j of the chunk): the 8 outputs'
// scales as FP32 (exact widening). Same bytes and values as the Q8 rows; rebuilt by every prefill call.
constexpr unsigned kWide2Groups = 40U, kWide2Chunks = 80U;
constexpr std::uint64_t kWide2PackCodeBytes = std::uint64_t{kWide2Chunks} * kWide2Groups * 32U * 32U;
constexpr std::uint64_t kWide2PackScaleBytes = std::uint64_t{kWide2Chunks} * kWide2Groups * 4U * 8U * 4U;
__global__ __launch_bounds__(32) void wide2_pack_down_kernel(const unsigned char* __restrict__ down, uint4* __restrict__ codes,
    float* __restrict__ scales) {
    const unsigned c = blockIdx.x / kWide2Groups, g = blockIdx.x % kWide2Groups, lane = threadIdx.x, l = bitrev5(lane);
    const unsigned flip = (l & 1U) * 4U, first_block = (c / 5U) * 20U + 4U * (c % 5U);
    unsigned d[8];
#pragma unroll
    for (unsigned p = 0; p < 8U; ++p) {
        const auto* block = down + static_cast<std::uint64_t>(g * 8U + (p ^ flip)) * 320U * 34U + (first_block + l / 8U) * 34U;
        const auto* q = reinterpret_cast<const unsigned short*>(block + 2U + 4U * (l % 8U));
        d[p] = static_cast<unsigned>(q[0]) | (static_cast<unsigned>(q[1]) << 16U);
    }
    auto* out = codes + (static_cast<std::uint64_t>(c) * kWide2Groups + g) * 64U + lane * 2U;
    out[0] = make_uint4(d[0], d[1], d[2], d[3]);
    out[1] = make_uint4(d[4], d[5], d[6], d[7]);
    const unsigned j = lane / 8U, p = lane % 8U;
    const auto* block = down + static_cast<std::uint64_t>(g * 8U + p) * 320U * 34U + (first_block + j) * 34U;
    scales[((static_cast<std::uint64_t>(c) * kWide2Groups + g) * 4U + j) * 8U + p] =
        __half2float(__ushort_as_half(*reinterpret_cast<const unsigned short*>(block)));
}
template <unsigned Waves, bool Injection, bool Vopd>
__device__ __forceinline__ void wide2_splitk_projection_body(const HyperWideRows& rows, const void* __restrict__ weights,
    const float* __restrict__ pack_scales, unsigned streams, unsigned first, unsigned total_outputs, float4 (*xs)[kWide2Rows][32],
    unsigned row_block) {
    constexpr unsigned R = kWide2Rows, O = kWide2Outputs, N = R * O, Chunks = 80U, Staged = R * 32U / (32U * Waves);
    const unsigned lane = threadIdx.x % 32U, l = bitrev5(lane), flip = (l & 1U) * 4U, row0 = row_block * R;
    const bool active = first < total_outputs;
    // Staging item e: row e / 32, window lane e % 32 -> LDS [row ^ (window lane >> 1)][bitrev5(window lane)].
    const auto x_at = [&](unsigned chunk, unsigned e) {
        const unsigned t = e / 32U, k = e % 32U;
        if (row0 + t >= rows.count) return make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        return *reinterpret_cast<const float4*>(wide_row(rows.normalized, rows.normalized_stride, row0 + t) +
            (chunk / 5U) * 640U + (chunk % 5U) * 128U + k * 4U);
    };
    const auto x_store = [&](unsigned buffer, unsigned e, float4 value) {
        const unsigned t = e / 32U, k = e % 32U;
        xs[buffer][t ^ (k >> 1U)][bitrev5(k)] = value;
    };
    float4 w[Injection ? O : 1U];
    unsigned packed[Injection ? 1U : O];
    float scales[Injection ? 1U : O];
    // Injection: output p of every lane loads row first + p (wave-uniform row addresses) and the decode
    // takes slot p ^ flip. Q8: the packed copy is already in slot order (codes) or is read in slot
    // order (scales: the lane's block, halves swapped when flip is set).
    const auto load_weights = [&](unsigned chunk) {
        if constexpr (Injection) {
            const unsigned tile = chunk / 5U, s = chunk % 5U;
#pragma unroll
            for (unsigned p = 0; p < O; ++p) {
                const unsigned out = first + p;
                w[p] = out < total_outputs ? *reinterpret_cast<const float4*>(static_cast<const float*>(weights) +
                    static_cast<std::uint64_t>(out) * 10240U + tile * 640U + s * 128U + l * 4U) : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
            }
        } else {
            const std::uint64_t at = static_cast<std::uint64_t>(chunk) * kWide2Groups + first / O;
            const auto* c4 = static_cast<const uint4*>(weights) + at * 64U + lane * 2U;
            const uint4 lo = c4[0], hi = c4[1];
            packed[0] = lo.x; packed[1] = lo.y; packed[2] = lo.z; packed[3] = lo.w;
            packed[4] = hi.x; packed[5] = hi.y; packed[6] = hi.z; packed[7] = hi.w;
            const auto* s4 = reinterpret_cast<const float4*>(pack_scales + (at * 4U + l / 8U) * 8U);
            const float4 a = s4[flip ? 1U : 0U], b = s4[flip ? 0U : 1U];
            scales[0] = a.x; scales[1] = a.y; scales[2] = a.z; scales[3] = a.w;
            scales[4] = b.x; scales[5] = b.y; scales[6] = b.z; scales[7] = b.w;
        }
    };
    float acc[N], total[N / 32U];
#pragma unroll
    for (unsigned i = 0; i < N; ++i) acc[i] = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < N / 32U; ++i) total[i] = 0.0F;
#pragma unroll
    for (unsigned m = 0; m < Staged; ++m) {
        const unsigned e = threadIdx.x + m * 32U * Waves;
        x_store(0U, e, x_at(0U, e));
    }
    if (active) load_weights(0U);
    __syncthreads();
    for (unsigned chunk = 0; chunk < Chunks; ++chunk) {
        float4 next[Staged];
        if (chunk + 1U < Chunks) {
#pragma unroll
            for (unsigned m = 0; m < Staged; ++m) next[m] = x_at(chunk + 1U, threadIdx.x + m * 32U * Waves);
        }
        if (active) {
            // coefficient = scale * float(code); float(code) = cvt_ubyte(code ^ 0x80) - 128, exact.
            float coefficient[O][4];
            const bool swap = flip != 0U;
#pragma unroll
            for (unsigned p = 0; p < O; ++p) {
                if constexpr (Injection) {
                    const float4 v = swap ? w[p ^ 4U] : w[p];
                    coefficient[p][0] = v.x; coefficient[p][1] = v.y; coefficient[p][2] = v.z; coefficient[p][3] = v.w;
                } else {
                    const unsigned biased = packed[p] ^ 0x80808080U;
                    const float scale = scales[p];
#pragma unroll
                    for (unsigned j = 0; j < 4U; ++j)
                        coefficient[p][j] = scale * (static_cast<float>((biased >> (8U * j)) & 255U) - 128.0F);
                }
            }
            if (chunk + 1U < Chunks) load_weights(chunk + 1U);
            if constexpr (Vopd) {
                // The same FMAs (each accumulator's j order) as hand-paired VOPD (hip_hc_wide2_vopd.inc).
                float flat[O * 4U];
#pragma unroll
                for (unsigned p = 0; p < O; ++p)
#pragma unroll
                    for (unsigned j = 0; j < 4U; ++j) flat[p * 4U + j] = coefficient[p][j];
                hc_wide2_chunk(acc, flat, static_cast<unsigned>(reinterpret_cast<std::uintptr_t>(&xs[chunk & 1U][0][lane])));
            } else {
#pragma unroll
                for (unsigned q = 0; q < R; ++q) {
                    const float4 x = xs[chunk & 1U][q][lane];
                    const float inputs[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
                    for (unsigned j = 0; j < 4U; ++j)
#pragma unroll
                        for (unsigned p = 0; p < O; ++p)
                            acc[q * O + p] = fmaf(inputs[j], coefficient[p][j], acc[q * O + p]);
                }
            }
            if (chunk % 5U == 4U) {
                wide2_lane_tree(acc);
#pragma unroll
                for (unsigned i = 0; i < N / 32U; ++i) total[i] = total[i] + acc[i];
#pragma unroll
                for (unsigned i = 0; i < N; ++i) acc[i] = 0.0F;
            }
        }
        if (chunk + 1U < Chunks) {
#pragma unroll
            for (unsigned m = 0; m < Staged; ++m) x_store((chunk + 1U) & 1U, threadIdx.x + m * 32U * Waves, next[m]);
        }
        __syncthreads();
    }
    if (!active) return;
    const unsigned t = l >> 1U;
    if (row0 + t >= rows.count) return;
#pragma unroll
    for (unsigned i = 0; i < N / 32U; ++i) {
        const unsigned out = first + flip + i;
        if (out >= total_outputs) continue;
        const float scaled = total[i] / static_cast<float>(streams);
        if constexpr (Injection) wide_row(rows.injection, rows.injection_stride, row0 + t)[out] = 2.0F * sigmoid(scaled);
        else {
            const float low = scaled * sigmoid(scaled);
            wide_row(rows.normalized, rows.normalized_stride, row0 + t)[10240U + out] = low;
            if (rows.low_t) rows.low_t[static_cast<std::uint64_t>(out) * rows.low_t_pitch + row0 + t] = low;
        }
    }
}
// RowsFirst: blockIdx.x walks the row blocks of one output block (its packed weights stay in L2 across them).
template <unsigned Waves, bool Vopd, bool RowsFirst = false>
__global__ __launch_bounds__(32U * Waves) void wide2_splitk_hyper_projection_kernel(HyperWideRows rows,
    const uint4* __restrict__ pack_codes, const float* __restrict__ pack_scales, unsigned streams, unsigned rank,
    const void* __restrict__ inject, unsigned low_blocks) {
    __shared__ float4 xs[2][kWide2Rows][32];
    const unsigned wave = threadIdx.x / 32U;
    const unsigned out_block = RowsFirst ? blockIdx.y : blockIdx.x, row_block = RowsFirst ? blockIdx.x : blockIdx.y;
    if (out_block >= low_blocks)
        wide2_splitk_projection_body<Waves, true, Vopd>(rows, inject, nullptr, streams, wave * kWide2Outputs, wave == 0U ? streams : 0U, xs,
            row_block);
    else wide2_splitk_projection_body<Waves, false, Vopd>(rows, pack_codes, pack_scales, streams,
        (out_block * Waves + wave) * kWide2Outputs, rank, xs, row_block);
}
// The packed copy's scratch (one per process: prefill runs on the execution stream).
unsigned char* wide2_pack_scratch() {
    static unsigned char* const pool = [] {
        void* p = nullptr;
        if (hipMalloc(&p, kWide2PackCodeBytes + kWide2PackScaleBytes) != hipSuccess)
            throw std::runtime_error("HC wide rows: packed down-projection scratch allocation failed");
        return static_cast<unsigned char*>(p);
    }();
    return pool;
}

template <unsigned Rows, unsigned Waves>
__global__ __launch_bounds__(32U * Waves) void wide_splitk_hyper_projection_kernel(HyperWideRows rows,
    const void* __restrict__ down, unsigned streams, unsigned rank, const void* __restrict__ inject, unsigned low_blocks) {
    __shared__ float4 xs[2][Rows][32];
    const unsigned wave = threadIdx.x / 32U;
    const bool is_inject = blockIdx.x >= low_blocks;
    const unsigned first = (is_inject ? 0U : blockIdx.x * Waves * kWideProjectionOutputs) + wave * kWideProjectionOutputs;
    const unsigned total = is_inject ? streams : rank;
    const unsigned valid = first < total ? (total - first < kWideProjectionOutputs ? total - first : kWideProjectionOutputs) : 0U;
    if (is_inject) wide_splitk_projection_body<Rows, Waves, true>(rows, inject, streams, first, valid, xs);
    else wide_splitk_projection_body<Rows, Waves, false>(rows, down, streams, first, valid, xs);
}

// Rank-320 lane chain over Rows rows per block: lane = one output feature with the chains of all
// its streams, accumulators = streams x rows. Each 32-code Q8 block is decoded once for all rows;
// the rows' low values of that block sit in LDS (index-major, one broadcast read serves four rows
// of every stream). Same chain (block, then code order), coefficient expression and stream mix
// (in the lane, stream order) as the window kernel.
constexpr unsigned kWideOutputWaves = 4U, kWideStreams = 4U;
template <typename Storage, unsigned Rows>
__global__ __launch_bounds__(128) void wide_lane_chain_hyper_output_kernel(HyperWideRows rows, const void* __restrict__ up,
    std::uint32_t hidden) {
    constexpr unsigned kBlocks = 10U, kRowBytes = kBlocks * 34U, Pitch = Rows + 4U, S = kWideStreams;
    constexpr unsigned Staged = Rows * 32U / (32U * kWideOutputWaves);
    __shared__ __align__(16) float lows[2][32][Pitch];
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U, row0 = blockIdx.y * Rows;
    const std::uint64_t feature = (static_cast<std::uint64_t>(blockIdx.x) * kWideOutputWaves + wave) * 32U + lane;
    const auto* up_rows = static_cast<const unsigned char*>(up) + feature * kRowBytes;
    const auto low_at = [&](unsigned block, unsigned index) {
        const unsigned t = index / 32U, k = index % 32U;
        return row0 + t < rows.count ? wide_row(rows.normalized, rows.normalized_stride, row0 + t)[10240U + block * 32U + k] : 0.0F;
    };
    float gate[S][Rows];
#pragma unroll
    for (unsigned s = 0; s < S; ++s)
#pragma unroll
        for (unsigned t = 0; t < Rows; ++t) gate[s][t] = 0.0F;
#pragma unroll
    for (unsigned m = 0; m < Staged; ++m) {
        const unsigned index = threadIdx.x + m * 32U * kWideOutputWaves;
        lows[0][index % 32U][index / 32U] = low_at(0U, index);
    }
    __syncthreads();
    for (unsigned block = 0; block < kBlocks; ++block) {
        // Block b's 34 bytes from the dword-aligned 36 around them (b * 34 % 4 is 0 or 2, the same for every lane):
        // bytes 0-1 hold the scale, bytes 2-33 the codes; codes[s][m] = codes 4m .. 4m + 3 of stream s.
        const bool odd = block % 2U;
        float scale[S];
        unsigned codes[S][8];
#pragma unroll
        for (unsigned s = 0; s < S; ++s) {
            const auto* aligned = reinterpret_cast<const unsigned*>(up_rows + static_cast<std::uint64_t>(s) * hidden * kRowBytes +
                block * 34U - (odd ? 2U : 0U));
            unsigned words[9];
#pragma unroll
            for (unsigned n = 0; n < 9U; ++n) words[n] = aligned[n];
            scale[s] = __half2float(__ushort_as_half(static_cast<unsigned short>(odd ? words[0] >> 16U : words[0] & 0xffffU)));
#pragma unroll
            for (unsigned m = 0; m < 8U; ++m) codes[s][m] = odd ? words[m + 1U] : __builtin_amdgcn_alignbyte(words[m + 1U], words[m], 2U);
        }
        float next[Staged];
        if (block + 1U < kBlocks) {
#pragma unroll
            for (unsigned m = 0; m < Staged; ++m) next[m] = low_at(block + 1U, threadIdx.x + m * 32U * kWideOutputWaves);
        }
        const auto* values = lows[block & 1U];
        // Four codes per iteration, the codes rotating through registers: a fully unrolled
        // block front-loads all of its LDS reads and spills.
#pragma unroll 1
        for (unsigned m = 0; m < 8U; ++m) {
            unsigned four[S];
#pragma unroll
            for (unsigned s = 0; s < S; ++s) {
                four[s] = codes[s][0];
#pragma unroll
                for (unsigned n = 0; n < 7U; ++n) codes[s][n] = codes[s][n + 1U];
            }
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) {
                float coefficient[S];
#pragma unroll
                for (unsigned s = 0; s < S; ++s)
                    coefficient[s] = scale[s] * static_cast<float>(static_cast<signed char>((four[s] >> (8U * j)) & 255U));
#pragma unroll
                for (unsigned q = 0; q < Rows / 4U; ++q) {
                    const float4 low = *reinterpret_cast<const float4*>(&values[4U * m + j][4U * q]);
#pragma unroll
                    for (unsigned s = 0; s < S; ++s) {
                        gate[s][4U * q] += low.x * coefficient[s];
                        gate[s][4U * q + 1U] += low.y * coefficient[s];
                        gate[s][4U * q + 2U] += low.z * coefficient[s];
                        gate[s][4U * q + 3U] += low.w * coefficient[s];
                    }
                }
            }
        }
        if (block + 1U < kBlocks) {
#pragma unroll
            for (unsigned m = 0; m < Staged; ++m) {
                const unsigned index = threadIdx.x + m * 32U * kWideOutputWaves;
                lows[(block + 1U) & 1U][index % 32U][index / 32U] = next[m];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned t = 0; t < Rows; ++t) {
        if (row0 + t >= rows.count) break;
        const auto* normalized = wide_row(rows.normalized, rows.normalized_stride, row0 + t);
        float mixed = 0.0F;
#pragma unroll
        for (unsigned s = 0; s < S; ++s)
            mixed += normalized[static_cast<std::uint64_t>(s) * hidden + feature] * sigmoid(gate[s][t]);
        store_activation(&wide_row(static_cast<Storage*>(rows.output), rows.output_stride, row0 + t)[feature],
            mixed / static_cast<float>(S));
    }
}


// Rank-320 lane chains over 32 rows per wave with the rows' low values as scalar operands: lane = one
// output feature with the gate chains of its 4 streams x 32 rows (hip_hc_wide2_output_vopd.inc runs one
// rank step as dual-issued FMAs whose shared operand is the row's low value in an SGPR, read from the
// transposed copy low_t). Same chains (k ascending, gate = fma(low, coefficient, gate)), coefficient
// value (scale * float(code): float(code) = cvt_ubyte(code ^ 0x80) - 128, exact) and stream mix as the
// lane-chain kernels; no LDS, no barriers.
#include "hip_hc_wide2_output_vopd.inc"
// Rows per wave: 32 (two rank steps per scalar-load wait) or 16 (fewer VGPRs, four steps per wait).
constexpr unsigned kWide2OutputWaves = 4U, kWide2OutputRows = 32U;
// RowsFirst: blockIdx.x walks the row groups of one feature block (its up weights stay in L2 across them).
// Half (fast profile, STRIX_HCW_NORM16=2): the stream mix reads the FP16 normalized rows (rows.normalized16).
// Recompute (STRIX_HCW_NORMFREE): the stream mix recomputes the normalized values (rows.input, rows.inverse).
template <typename Storage, unsigned R, bool RowsFirst, bool Half = false, bool Recompute = false>
__global__ __launch_bounds__(32U * kWide2OutputWaves) void wide2_hyper_output_kernel(HyperWideRows rows, const void* __restrict__ up,
    std::uint32_t hidden) {
    constexpr unsigned kBlocks = 10U, kRowBytes = kBlocks * 34U, S = kWideStreams, Steps = 64U / R;
    static_assert(R == 16U || R == 32U);
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U, row0 = (RowsFirst ? blockIdx.x : blockIdx.y) * R;
    const std::uint64_t feature = (static_cast<std::uint64_t>(RowsFirst ? blockIdx.y : blockIdx.x) * kWide2OutputWaves + wave) * 32U + lane;
    const auto* up_rows = static_cast<const unsigned char*>(up) + feature * kRowBytes;
    // Constant address space: uniform loads become scalar loads (s_load), so the values are SGPRs.
    using ConstantFloat = const __attribute__((address_space(4))) float;
    auto* low_t = reinterpret_cast<ConstantFloat*>(reinterpret_cast<std::uintptr_t>(rows.low_t + row0));
    const std::uint32_t pitch = rows.low_t_pitch;
    float g[S * R];
#pragma unroll
    for (unsigned i = 0; i < S * R; ++i) g[i] = 0.0F;
    // Block b's 34 bytes of stream s from the dword-aligned 36 around them (as the lane-chain kernel).
    unsigned words[S][9];
    const auto fetch = [&](unsigned block) {
        const bool odd = block % 2U;
#pragma unroll
        for (unsigned s = 0; s < S; ++s) {
            const auto* aligned = reinterpret_cast<const unsigned*>(up_rows + static_cast<std::uint64_t>(s) * hidden * kRowBytes +
                block * 34U - (odd ? 2U : 0U));
#pragma unroll
            for (unsigned n = 0; n < 9U; ++n) words[s][n] = aligned[n];
        }
    };
    fetch(0U);
    for (unsigned block = 0; block < kBlocks; ++block) {
        const bool odd = block % 2U;
        float scale[S];
        unsigned codes[S][8];
#pragma unroll
        for (unsigned s = 0; s < S; ++s) {
            scale[s] = __half2float(__ushort_as_half(static_cast<unsigned short>(odd ? words[s][0] >> 16U : words[s][0] & 0xffffU)));
#pragma unroll
            for (unsigned m = 0; m < 8U; ++m)
                codes[s][m] = (odd ? words[s][m + 1U] : __builtin_amdgcn_alignbyte(words[s][m + 1U], words[s][m], 2U)) ^ 0x80808080U;
        }
        if (block + 1U < kBlocks) fetch(block + 1U);
#pragma unroll 1
        for (unsigned m = 0; m < 8U; ++m) {
            unsigned four[S];
#pragma unroll
            for (unsigned s = 0; s < S; ++s) {
                four[s] = codes[s][0];
#pragma unroll
                for (unsigned n = 0; n < 7U; ++n) codes[s][n] = codes[s][n + 1U];
            }
            // Steps rank steps per scalar-load wait (scalar loads return out of order, so each wait drains
            // every load in flight): their 64 low values are requested together.
#pragma unroll
            for (unsigned j = 0; j < 4U; j += Steps) {
                const unsigned k = block * 32U + 4U * m + j;
                float low[Steps][R];
#pragma unroll
                for (unsigned h = 0; h < Steps; ++h)
#pragma unroll
                    for (unsigned t = 0; t < R; ++t) low[h][t] = low_t[static_cast<std::uint64_t>(k + h) * pitch + t];
#pragma unroll
                for (unsigned h = 0; h < Steps; ++h) {
                    float c[S];
#pragma unroll
                    for (unsigned s = 0; s < S; ++s) c[s] = scale[s] * (static_cast<float>((four[s] >> (8U * (j + h))) & 255U) - 128.0F);
                    hc_wide2_output_step(g, c[0], c[1], c[2], c[3], low[h]);
                }
            }
        }
    }
#pragma unroll
    for (unsigned t = 0; t < R; ++t) {
        if (row0 + t >= rows.count) break;
        float mixed = 0.0F;
        if constexpr (Half) {
            const auto* normalized = rows.normalized16 + static_cast<std::uint64_t>(row0 + t) * rows.normalized16_pitch;
#pragma unroll
            for (unsigned s = 0; s < S; ++s)
                mixed += static_cast<float>(normalized[static_cast<std::uint64_t>(s) * hidden + feature]) * sigmoid(g[4U * t + s]);
        } else if constexpr (Recompute) {
            const auto* input = static_cast<const Storage*>(wide_row(rows.input, rows.input_stride, row0 + t));
            const float* inverse = rows.inverse + 4ULL * (row0 + t);
#pragma unroll
            for (unsigned s = 0; s < S; ++s) {
                const std::uint64_t at = static_cast<std::uint64_t>(s) * hidden + feature;
                mixed += load_activation(input, at) * inverse[s] * rows.norm_weight[at] * sigmoid(g[4U * t + s]);
            }
        } else {
            const auto* normalized = wide_row(rows.normalized, rows.normalized_stride, row0 + t);
#pragma unroll
            for (unsigned s = 0; s < S; ++s)
                mixed += normalized[static_cast<std::uint64_t>(s) * hidden + feature] * sigmoid(g[4U * t + s]);
        }
        store_activation(&wide_row(static_cast<Storage*>(rows.output), rows.output_stride, row0 + t)[feature],
            mixed / static_cast<float>(S));
    }
}


bool supported_norm_type(
    std::uint32_t ggml_type) noexcept {
    return ggml_type == 0 ||
           ggml_type == 1 ||
           ggml_type == 30;
}

std::uint64_t linear_bytes(
    std::uint32_t ggml_type,
    std::uint32_t row_width,
    std::uint64_t output_rows) {
    switch (ggml_type) {
    case 0:
        return
            static_cast<std::uint64_t>(
                row_width) *
            output_rows *
            4ULL;
    case 1:
    case 30:
        return
            static_cast<std::uint64_t>(
                row_width) *
            output_rows *
            2ULL;
    case 8:
        if (row_width % 32U != 0) {
            throw std::invalid_argument(
                "Q8_0 hyper-mix row width must be divisible by 32");
        }
        return
            static_cast<std::uint64_t>(
                row_width / 32U) *
            output_rows *
            sizeof(Q8Block);
    case 23:
        if (row_width % 256U != 0) {
            throw std::invalid_argument(
                "IQ4_XS hyper-mix row width must be divisible by 256");
        }
        return
            static_cast<std::uint64_t>(
                row_width / 256U) *
            output_rows *
            sizeof(IQ4XSBlock);
    default:
        throw std::invalid_argument(
            "unsupported hyper-mix linear weight type");
    }
}

std::uint64_t norm_bytes(
    std::uint32_t ggml_type,
    std::uint32_t elements) {
    switch (ggml_type) {
    case 0:
        return
            static_cast<std::uint64_t>(
                elements) *
            4ULL;
    case 1:
    case 30:
        return
            static_cast<std::uint64_t>(
                elements) *
            2ULL;
    default:
        throw std::invalid_argument(
            "hyper-mix norm supports only F32, F16 and BF16");
    }
}

const ManifestTensor& manifest_tensor(
    const ExecutionDispatchContext& context,
    const ExecutionTensorAccess& access) {
    if (access.tensor_indices.size() != 1U ||
        context.resources == nullptr ||
        context.resources->model_manifest ==
            nullptr) {
        throw std::invalid_argument(
            "HIP hyper-mix requires one tensor per binding slot");
    }
    const auto index =
        access.tensor_indices.front();
    if (index >=
        context.resources->
            model_manifest->
            tensors.size()) {
        throw std::out_of_range(
            "HIP hyper-mix tensor index exceeds manifest");
    }
    return
        context.resources->
            model_manifest->
            tensors[index];
}

const RuntimeTensorBinding& runtime_tensor(
    const ExecutionDispatchContext& context,
    const ExecutionTensorAccess& access) {
    if (access.tensor_indices.size() != 1U) {
        throw std::invalid_argument(
            "HIP hyper-mix runtime tensor slot is not unique");
    }
    const auto& tensor =
        resolve_dispatch_tensor(
            context,
            access.tensor_indices.front());
    if (!runtime_tensor_gpu_visible(tensor.storage) ||
        tensor.address == 0) {
        throw std::invalid_argument(
            "HIP hyper-mix requires resident or repacked GPU-visible weights");
    }
    return tensor;
}

ExecutionBufferAddress prepare_parallel_hyper_workspace(
    const ExecutionDispatchContext& context, const HipHyperMixLauncherConfig& config,
    const RuntimeTensorBinding& down_runtime, const RuntimeTensorBinding* inject_runtime) {
    if (context.temporaries.size() != 1)
        throw std::invalid_argument("parallel hyper-mix requires one admitted workspace");
    const auto temporary = resolve_dispatch_temporary(context, 0);
    if (temporary.use.buffer.buffer_class != ExecutionBufferClass::hyper_workspace)
        throw std::invalid_argument("parallel hyper-mix requires its typed workspace binding");
    const auto width = static_cast<std::uint64_t>(config.hidden_size) * config.hc_count;
    const auto required = static_cast<std::uint64_t>(context.logical_rows) * (width + config.lowrank) * sizeof(float);
    if (temporary.bytes < required || temporary.address % alignof(float) != 0)
        throw std::invalid_argument("parallel hyper-mix workspace is undersized or unaligned");
    if (config.splitk_projection && (context.logical_rows > 65535U || temporary.address % 16U ||
        down_runtime.address % 16U || (inject_runtime && inject_runtime->address % 16U)))
        throw std::invalid_argument("split-K HC launch owner alignment or row count differs");
    return temporary;
}

template <typename Storage>
void launch_typed(
    const ExecutionDispatchContext& context,
    const HipHyperMixLauncherConfig& config,
    const ExecutionBufferAddress& input,
    const ExecutionBufferAddress& output,
    const ExecutionBufferAddress* injection_output,
    const ManifestTensor& norm_manifest,
    const RuntimeTensorBinding& norm_runtime,
    const ManifestTensor& down_manifest,
    const RuntimeTensorBinding& down_runtime,
    const ManifestTensor& up_manifest,
    const RuntimeTensorBinding& up_runtime,
    const ManifestTensor* inject_manifest,
    const RuntimeTensorBinding* inject_runtime,
    const ExecutionBufferAddress* prepared_temporary) {
    if (config.splitk_projection && (context.logical_rows != 1U || hip_graph_status() ||
        (current_hip_launch_context() && current_hip_launch_context()->graph_capture)))
        throw std::invalid_argument("split-K HC projection rejects captured or multirow execution");
    if (config.parallel) {
        const auto& temporary = *prepared_temporary;
        const auto width = static_cast<std::uint64_t>(config.hidden_size) * config.hc_count;
        auto* normalized = reinterpret_cast<float*>(temporary.address);
        auto* low = normalized + static_cast<std::uint64_t>(context.logical_rows) * width;
        constexpr std::uint32_t projection_threads = 32;
        if (!config.splitk_deferred_norm) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(prepare_hyper_norm_kernel<Storage>),
            dim3(config.hc_count, context.logical_rows), dim3(kThreads), 0, hip_execution_stream(),
            reinterpret_cast<const Storage*>(input.address), reinterpret_cast<const void*>(norm_runtime.address),
            norm_manifest.ggml_type, config.hidden_size, config.hc_count, config.rms_epsilon, normalized);
        check_hip(hipGetLastError(), "parallel hyper-mix normalization launch");
        if(config.arithmetic_stats)
            config.arithmetic_stats->materialized_norm_submissions.fetch_add(1,std::memory_order_relaxed);
        }
        if (config.staged_projections) {
            if (config.fused_injection) {
                const auto lowrank_blocks = (config.lowrank + kStagedOutputs - 1U) / kStagedOutputs;
                const auto injection_blocks = injection_output ? (config.hc_count + kStagedOutputs - 1U) / kStagedOutputs : 0U;
                if (config.typed_projection) {
                    if (config.splitk_projection) {
                        constexpr unsigned outputs = 4U;
                        const auto blocks = (config.lowrank + outputs - 1U) / outputs;
                        const dim3 grid(blocks + (injection_output ? (config.hc_count + outputs - 1U) / outputs : 0U), context.logical_rows);
                        // The registered Q6P32 copy of the down tensor replaces it (same bits).
                        const auto* q6_down = q6p::find(reinterpret_cast<const void*>(down_runtime.address));
                        const bool q6 = q6_down && q6_down->kind == q6p::Kind::q6p32 && q6_down->rows == config.lowrank &&
                            q6_down->cols == 10240U && width == 10240U;
                        const auto launch = [&]<typename Acc>() {
                            if (q6 && std::is_same_v<Acc, float>)
                                hipLaunchKernelGGL(HIP_KERNEL_NAME(splitk_hyper_projection_kernel<outputs,float,true>),
                                    grid, dim3(512U), 0, hip_execution_stream(), normalized,
                                    reinterpret_cast<const void*>(q6_down->base), static_cast<std::uint32_t>(width),
                                    config.hc_count, config.lowrank, low,
                                    inject_runtime ? reinterpret_cast<const void*>(inject_runtime->address) : nullptr,
                                    injection_output ? reinterpret_cast<float*>(injection_output->address) : nullptr, blocks);
                            else
                            hipLaunchKernelGGL(HIP_KERNEL_NAME(splitk_hyper_projection_kernel<outputs,Acc>),
                                grid, dim3(512U), 0, hip_execution_stream(), normalized,
                                reinterpret_cast<const void*>(down_runtime.address), static_cast<std::uint32_t>(width),
                                config.hc_count, config.lowrank, low,
                                inject_runtime ? reinterpret_cast<const void*>(inject_runtime->address) : nullptr,
                                injection_output ? reinterpret_cast<float*>(injection_output->address) : nullptr, blocks);
                        };
                        if (config.splitk_deferred_norm) {
                            hipLaunchKernelGGL(HIP_KERNEL_NAME(deferred_hyper_projection_kernel<outputs>),
                                grid,dim3(512U),0,hip_execution_stream(),
                                reinterpret_cast<const std::uint16_t*>(input.address),
                                reinterpret_cast<const float*>(norm_runtime.address),
                                reinterpret_cast<const void*>(down_runtime.address),config.lowrank,
                                config.rms_epsilon,normalized,low,
                                inject_runtime ? reinterpret_cast<const void*>(inject_runtime->address) : nullptr,
                                injection_output ? reinterpret_cast<float*>(injection_output->address) : nullptr,blocks,nullptr,nullptr);
                            check_hip(hipGetLastError(), "deferred RMS hyper projection enqueue");
                            if(config.arithmetic_stats) {
                                config.arithmetic_stats->splitk_f32_submissions.fetch_add(1,std::memory_order_relaxed);
                                config.arithmetic_stats->deferred_norm_submissions.fetch_add(1,std::memory_order_relaxed);
                            }
                        } else if (config.splitk_projection_fp64) launch.template operator()<double>();
                        else {
                            launch.template operator()<float>();
                            check_hip(hipGetLastError(), "FP32 split-K hyper projection enqueue");
                            if (config.arithmetic_stats)
                                config.arithmetic_stats->splitk_f32_submissions.fetch_add(1, std::memory_order_relaxed);
                        }
                    } else if (config.prefetch_hc_projection_next_tile) {
                        hipLaunchKernelGGL(next_tile_hyper_projection_kernel,
                            dim3(lowrank_blocks + injection_blocks, context.logical_rows), dim3(128U),
                            0, hip_execution_stream(), normalized, reinterpret_cast<const void*>(down_runtime.address),
                            static_cast<std::uint32_t>(width), config.hc_count, config.lowrank, low,
                            inject_runtime ? reinterpret_cast<const void*>(inject_runtime->address) : nullptr,
                            injection_output ? reinterpret_cast<float*>(injection_output->address) : nullptr, lowrank_blocks);
                    } else if (config.prefetch_rows) {
                        hipLaunchKernelGGL(batched_typed_hyper_projection_kernel,
                            dim3(lowrank_blocks + injection_blocks, context.logical_rows), dim3(kStagedThreads),
                            0, hip_execution_stream(), normalized, reinterpret_cast<const void*>(down_runtime.address),
                            static_cast<std::uint32_t>(width), config.hc_count, config.lowrank, low,
                            inject_runtime ? reinterpret_cast<const void*>(inject_runtime->address) : nullptr,
                            injection_output ? reinterpret_cast<float*>(injection_output->address) : nullptr, lowrank_blocks);
                    } else {
                    hipLaunchKernelGGL(typed_fused_hyper_projection_kernel,
                        dim3(lowrank_blocks + injection_blocks, context.logical_rows), dim3(kStagedThreads),
                        0, hip_execution_stream(), normalized, reinterpret_cast<const void*>(down_runtime.address),
                        static_cast<std::uint32_t>(width), config.hc_count, config.lowrank, low,
                        inject_runtime ? reinterpret_cast<const void*>(inject_runtime->address) : nullptr,
                        injection_output ? reinterpret_cast<float*>(injection_output->address) : nullptr, lowrank_blocks);
                    }
                } else {
                hipLaunchKernelGGL(fused_staged_hyper_projection_kernel,
                    dim3(lowrank_blocks + injection_blocks, context.logical_rows), dim3(kStagedThreads),
                    0, hip_execution_stream(), normalized,
                    reinterpret_cast<const void*>(down_runtime.address), down_manifest.ggml_type,
                    static_cast<std::uint32_t>(width), config.hc_count, config.lowrank, low,
                    inject_runtime ? reinterpret_cast<const void*>(inject_runtime->address) : nullptr,
                    inject_manifest ? inject_manifest->ggml_type : 0U,
                    injection_output ? reinterpret_cast<float*>(injection_output->address) : nullptr, lowrank_blocks);
                }
                check_hip(hipGetLastError(), "fused staged hyper-mix projection launch");
                if(config.splitk_deferred_norm){
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(deferred_lane_chain_hyper_output_kernel<Storage>),
                        dim3(config.hidden_size/32U,context.logical_rows),dim3(32U*config.hc_count),
                        static_cast<std::size_t>(lane_chain_hyper_output_lds(config.hc_count,config.lowrank)),hip_execution_stream(),
                        reinterpret_cast<const Storage*>(input.address),normalized,
                        reinterpret_cast<const float*>(norm_runtime.address),low,
                        reinterpret_cast<const void*>(up_runtime.address),config.hidden_size,config.hc_count,config.lowrank,
                        reinterpret_cast<Storage*>(output.address));
                }else if(config.lane_chain_output){
                    // The registered Q6P32 copy of the up tensor replaces it (same bits).
                    const auto* q6_up=q6p::find(reinterpret_cast<const void*>(up_runtime.address));
                    if(q6_up && q6_up->kind==q6p::Kind::q6p32 && q6_up->cols==config.lowrank &&
                       q6_up->rows==config.hidden_size*config.hc_count)
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(lane_chain_hyper_output_kernel<Storage,true>),
                        dim3(config.hidden_size/32U,context.logical_rows),dim3(32U*config.hc_count),
                        static_cast<std::size_t>(lane_chain_hyper_output_lds(config.hc_count,config.lowrank)),hip_execution_stream(),
                        normalized,low,reinterpret_cast<const void*>(q6_up->base),config.hidden_size,config.hc_count,config.lowrank,reinterpret_cast<Storage*>(output.address));
                    else
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(lane_chain_hyper_output_kernel<Storage>),
                        dim3(config.hidden_size/32U,context.logical_rows),dim3(32U*config.hc_count),
                        static_cast<std::size_t>(lane_chain_hyper_output_lds(config.hc_count,config.lowrank)),hip_execution_stream(),
                        normalized,low,reinterpret_cast<const void*>(up_runtime.address),config.hidden_size,config.hc_count,config.lowrank,reinterpret_cast<Storage*>(output.address));
                }else if (config.prefetch_q8_output) {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_batched_hyper_output_kernel<Storage>),
                        dim3((config.hidden_size + kStagedOutputs - 1U) / kStagedOutputs, context.logical_rows),
                        dim3(kStagedThreads), 0, hip_execution_stream(), normalized, low,
                        reinterpret_cast<const void*>(up_runtime.address),
                        config.hidden_size, config.hc_count, config.lowrank, reinterpret_cast<Storage*>(output.address));
                } else {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_hyper_output_kernel<Storage>),
                        dim3((config.hidden_size + kStagedOutputs - 1U) / kStagedOutputs, context.logical_rows),
                        dim3(kStagedThreads), 0, hip_execution_stream(), normalized, low,
                        reinterpret_cast<const void*>(up_runtime.address), up_manifest.ggml_type,
                        config.hidden_size, config.hc_count, config.lowrank, reinterpret_cast<Storage*>(output.address));
                }
                check_hip(hipGetLastError(), "fused staged hyper-mix output launch");
                return;
            }
            hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_hyper_projection_kernel<false>),
                dim3((config.lowrank + kStagedOutputs - 1U) / kStagedOutputs, context.logical_rows),
                dim3(kStagedThreads), 0, hip_execution_stream(), normalized,
                reinterpret_cast<const void*>(down_runtime.address), down_manifest.ggml_type,
                static_cast<std::uint32_t>(width), config.hc_count, config.lowrank, low);
            check_hip(hipGetLastError(), "staged hyper-mix lowrank launch");
            if (injection_output) {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_hyper_projection_kernel<true>),
                    dim3((config.hc_count + kStagedOutputs - 1U) / kStagedOutputs, context.logical_rows),
                    dim3(kStagedThreads), 0, hip_execution_stream(), normalized,
                    reinterpret_cast<const void*>(inject_runtime->address), inject_manifest->ggml_type,
                    static_cast<std::uint32_t>(width), config.hc_count, config.hc_count,
                    reinterpret_cast<float*>(injection_output->address));
                check_hip(hipGetLastError(), "staged hyper-mix injection launch");
            }
            hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_hyper_output_kernel<Storage>),
                dim3((config.hidden_size + kStagedOutputs - 1U) / kStagedOutputs, context.logical_rows),
                dim3(kStagedThreads), 0, hip_execution_stream(), normalized, low,
                reinterpret_cast<const void*>(up_runtime.address), up_manifest.ggml_type,
                config.hidden_size, config.hc_count, config.lowrank, reinterpret_cast<Storage*>(output.address));
            check_hip(hipGetLastError(), "staged hyper-mix output launch");
            return;
        }
        hipLaunchKernelGGL(parallel_hyper_lowrank_kernel,
            dim3((config.lowrank + projection_threads - 1U) / projection_threads, context.logical_rows),
            dim3(projection_threads), 0, hip_execution_stream(), normalized,
            reinterpret_cast<const void*>(down_runtime.address), down_manifest.ggml_type,
            static_cast<std::uint32_t>(width), config.hc_count, config.lowrank, low);
        check_hip(hipGetLastError(), "parallel hyper-mix lowrank launch");
        if (injection_output) {
            hipLaunchKernelGGL(parallel_hyper_injection_kernel, dim3(context.logical_rows),
                dim3(projection_threads), 0, hip_execution_stream(), normalized,
                reinterpret_cast<const void*>(inject_runtime->address), inject_manifest->ggml_type,
                static_cast<std::uint32_t>(width), config.hc_count, reinterpret_cast<float*>(injection_output->address));
            check_hip(hipGetLastError(), "parallel hyper-mix injection launch");
        }
        hipLaunchKernelGGL(HIP_KERNEL_NAME(parallel_hyper_output_kernel<Storage>),
            dim3((config.hidden_size + projection_threads - 1U) / projection_threads, context.logical_rows),
            dim3(projection_threads), 0, hip_execution_stream(), normalized, low,
            reinterpret_cast<const void*>(up_runtime.address), up_manifest.ggml_type,
            config.hidden_size, config.hc_count, config.lowrank, reinterpret_cast<Storage*>(output.address));
        check_hip(hipGetLastError(), "parallel hyper-mix output launch");
        return;
    }
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            hyper_mix_kernel<Storage>),
        dim3(context.logical_rows),
        dim3(kThreads),
        0,
        hip_execution_stream(),
        reinterpret_cast<
            const Storage*>(
                input.address),
        reinterpret_cast<const void*>(
            norm_runtime.address),
        norm_manifest.ggml_type,
        reinterpret_cast<const void*>(
            down_runtime.address),
        down_manifest.ggml_type,
        reinterpret_cast<const void*>(
            up_runtime.address),
        up_manifest.ggml_type,
        inject_runtime != nullptr
            ? reinterpret_cast<
                  const void*>(
                  inject_runtime->address)
            : nullptr,
        inject_manifest != nullptr
            ? inject_manifest->ggml_type
            : 0U,
        injection_output != nullptr,
        reinterpret_cast<Storage*>(
            output.address),
        injection_output != nullptr
            ? reinterpret_cast<float*>(
                  injection_output->address)
            : nullptr,
        config.hidden_size,
        config.hc_count,
        config.lowrank,
        config.rms_epsilon);
    check_hip(
        hipGetLastError(),
        "HIP hyper-mix kernel launch");
}

struct PreparedHyperMix {
    const HipHyperMixLauncherConfig* config;
    ExecutionBufferAddress input, output;
    std::optional<ExecutionBufferAddress> injection_output, temporary;
    const ManifestTensor *norm_manifest, *down_manifest, *up_manifest, *inject_manifest;
    const RuntimeTensorBinding *norm_runtime, *down_runtime, *up_runtime, *inject_runtime;
};

PreparedHyperMix prepare_hyper_mix(
    const ExecutionDispatchContext& context,
    void* user_data) {
    const bool final_mix =
        context.op ==
        SerialOp::final_hyper_mix;
    if (context.op !=
            SerialOp::attn_hyper_mix &&
        context.op !=
            SerialOp::mlp_hyper_mix &&
        !final_mix) {
        throw std::invalid_argument(
            "HIP hyper-mix launcher attached to wrong operator");
    }
    if (context.reads.size() != 1 ||
        context.writes.size() !=
            (final_mix ? 1U : 2U)) {
        throw std::invalid_argument(
            "HIP hyper-mix scratch IO does not match operator form");
    }
    if (user_data == nullptr) {
        throw std::invalid_argument(
            "HIP hyper-mix requires configuration");
    }

    const auto& config =
        *static_cast<
            HipHyperMixLauncherConfig*>(
                user_data);
    validate_config(config);

    if(config.splitk_deferred_norm) {
        if(context.logical_rows!=1U || context.request_concurrency!=1U || hip_graph_status() ||
            (current_hip_launch_context() && current_hip_launch_context()->graph_capture))
            throw std::invalid_argument("deferred HC RMS supports eager R1 C1 only");
        hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
        check_hip(hipStreamIsCapturing(hip_execution_stream(),&capture),"query deferred HC RMS capture");
        if(capture!=hipStreamCaptureStatusNone)
            throw std::invalid_argument("deferred HC RMS cannot be captured");
    }
    if(config.prefetch_hc_projection_next_tile) {
        if(context.logical_rows!=1U || context.request_concurrency!=1U ||
           (current_hip_launch_context() && current_hip_launch_context()->graph_capture))
            throw std::invalid_argument("next-tile HC supports eager R1 C1 only");
        hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
        check_hip(hipStreamIsCapturing(hip_execution_stream(),&capture),"query next-tile HC capture");
        if(capture!=hipStreamCaptureStatusNone)
            throw std::invalid_argument("next-tile HC cannot be captured");
    }
    const auto input =
        resolve_dispatch_read(
            context, 0);
    const auto output =
        resolve_dispatch_write(
            context, 0);
    std::optional<
        ExecutionBufferAddress>
        injection_output;
    if (!final_mix) {
        injection_output =
            resolve_dispatch_write(
                context, 1);
    }

    const auto& norm_access =
        resolve_dispatch_tensor_access(
            context, 0);
    const auto& down_access =
        resolve_dispatch_tensor_access(
            context, 1);
    const auto& up_access =
        resolve_dispatch_tensor_access(
            context, 2);

    const auto& norm_manifest =
        manifest_tensor(
            context,
            norm_access);
    const auto& down_manifest =
        manifest_tensor(
            context,
            down_access);
    const auto& up_manifest =
        manifest_tensor(
            context,
            up_access);

    const auto& norm_runtime =
        runtime_tensor(
            context,
            norm_access);
    const auto& down_runtime =
        runtime_tensor(
            context,
            down_access);
    const auto& up_runtime =
        runtime_tensor(
            context,
            up_access);

    const ManifestTensor*
        inject_manifest = nullptr;
    const RuntimeTensorBinding*
        inject_runtime = nullptr;
    if (!final_mix) {
        const auto& access =
            resolve_dispatch_tensor_access(
                context, 3);
        inject_manifest =
            &manifest_tensor(
                context,
                access);
        inject_runtime =
            &runtime_tensor(
                context,
                access);
    }

    if (config.typed_projection && (down_manifest.ggml_type != 8U ||
        (inject_manifest != nullptr && inject_manifest->ggml_type != 0U)))
        throw std::invalid_argument("typed hyper projection requires Q8_0 down and F32 injection");

    if (config.prefetch_q8_output && up_manifest.ggml_type != 8U)
        throw std::invalid_argument("Q8 hyper output prefetch requires Q8_0 up projection");

    const auto expanded_width =
        static_cast<std::uint64_t>(
            config.hidden_size) *
        config.hc_count;
    const auto activation_bytes =
        static_cast<std::uint32_t>(
            config.activation);

    if (input.bytes <
            static_cast<std::uint64_t>(
                context.logical_rows) *
                expanded_width *
                activation_bytes ||
        output.bytes <
            static_cast<std::uint64_t>(
                context.logical_rows) *
                config.hidden_size *
                activation_bytes) {
        throw std::invalid_argument(
            "HIP hyper-mix activation buffers are undersized");
    }
    if (injection_output.has_value() &&
        injection_output->bytes <
            static_cast<std::uint64_t>(
                context.logical_rows) *
                config.hc_count *
                sizeof(float)) {
        throw std::invalid_argument(
            "HIP hyper-mix injection buffer is undersized");
    }

    if (!supported_norm_type(
            norm_manifest.ggml_type) ||
        !supported_linear_type(
            down_manifest.ggml_type) ||
        !supported_linear_type(
            up_manifest.ggml_type) ||
        (inject_manifest != nullptr &&
         !supported_linear_type(
             inject_manifest->
                 ggml_type))) {
        throw std::invalid_argument(
            "HIP hyper-mix contains unsupported weight type");
    }

    if (norm_manifest.dimensions.empty() ||
        norm_manifest.dimensions[0] !=
            expanded_width ||
        down_manifest.dimensions.size() < 2 ||
        down_manifest.dimensions[0] !=
            expanded_width ||
        down_manifest.dimensions[1] !=
            config.lowrank ||
        up_manifest.dimensions.size() < 2 ||
        up_manifest.dimensions[0] !=
            config.lowrank ||
        up_manifest.dimensions[1] !=
            expanded_width) {
        throw std::invalid_argument(
            "HIP hyper-mix tensor geometry does not match contract");
    }
    if (inject_manifest != nullptr &&
        (inject_manifest->dimensions.size() < 2 ||
         inject_manifest->dimensions[0] !=
             expanded_width ||
         inject_manifest->dimensions[1] !=
             config.hc_count)) {
        throw std::invalid_argument(
            "HIP hyper-mix inject tensor geometry does not match contract");
    }

    const auto norm_required =
        norm_bytes(
            norm_manifest.ggml_type,
            static_cast<
                std::uint32_t>(
                expanded_width));
    const auto down_required =
        linear_bytes(
            down_manifest.ggml_type,
            static_cast<
                std::uint32_t>(
                expanded_width),
            config.lowrank);
    const auto up_required =
        linear_bytes(
            up_manifest.ggml_type,
            config.lowrank,
            expanded_width);

    if (norm_runtime.bytes <
            norm_required ||
        down_runtime.bytes <
            down_required ||
        up_runtime.bytes <
            up_required) {
        throw std::invalid_argument(
            "HIP hyper-mix runtime weight binding is undersized");
    }
    if (inject_manifest != nullptr) {
        const auto inject_required =
            linear_bytes(
                inject_manifest->ggml_type,
                static_cast<
                    std::uint32_t>(
                    expanded_width),
                config.hc_count);
        if (inject_runtime->bytes <
            inject_required) {
            throw std::invalid_argument(
                "HIP hyper-mix inject runtime binding is undersized");
        }
    }

    if(config.splitk_deferred_norm && (norm_manifest.ggml_type!=0U ||
        norm_manifest.dimensions!=std::vector<std::uint64_t>{10240U} || norm_manifest.byte_size!=40960U))
        throw std::invalid_argument("deferred HC RMS requires its canonical F32 norm owner");
    if(config.lane_chain_output){
        if(!lane_chain_hyper_output_eligible(up_manifest,config.hidden_size,config.hc_count,config.lowrank,up_runtime.address%16U) || up_runtime.bytes<up_manifest.byte_size ||
           up_manifest.byte_size>std::numeric_limits<std::uintptr_t>::max()-up_runtime.address)
            throw std::invalid_argument("lane-chain HC output up owner admission differs");
        if(input.address%activation_bytes || output.address%activation_bytes || norm_runtime.address%(norm_manifest.ggml_type==0U?4U:2U) || down_runtime.address%2U ||
           (inject_runtime && inject_runtime->address%4U) || (injection_output && injection_output->address%4U))
            throw std::invalid_argument("lane-chain HC output typed owner alignment differs");
        const auto temp=resolve_dispatch_temporary(context,0);
        std::vector<std::pair<std::uintptr_t,std::uint64_t>> writes={{temp.address,temp.bytes},{output.address,output.bytes}};
        if(injection_output)writes.emplace_back(injection_output->address,injection_output->bytes);
        std::vector<std::pair<std::uintptr_t,std::uint64_t>> reads={{input.address,input.bytes},{norm_runtime.address,norm_runtime.bytes},{down_runtime.address,down_runtime.bytes},{up_runtime.address,up_runtime.bytes}};
        if(inject_runtime)reads.emplace_back(inject_runtime->address,inject_runtime->bytes);
        for(const auto& owners:{writes,reads})for(const auto& owner:owners)if(!owner.first || !owner.second || owner.second>std::numeric_limits<std::uintptr_t>::max()-owner.first)
            throw std::invalid_argument("lane-chain HC output invalid owner range");
        const auto overlap=[](const auto& x,const auto& y){return x.first<y.first+y.second && y.first<x.first+x.second;};
        for(std::size_t i=0;i<writes.size();++i){
            for(std::size_t j=i+1;j<writes.size();++j)if(overlap(writes[i],writes[j]))throw std::invalid_argument("lane-chain HC output mutable owners overlap");
            for(const auto& read:reads)if(overlap(writes[i],read))throw std::invalid_argument("lane-chain HC output read/write owners overlap");
        }
    }
    if(config.prefetch_hc_projection_next_tile) {
        const auto temporary=resolve_dispatch_temporary(context,0);
        if(context.temporaries.size()!=1U || temporary.use.buffer.buffer_class!=ExecutionBufferClass::hyper_workspace ||
           norm_manifest.dimensions.size()!=1U || norm_manifest.dimensions[0]!=10240U ||
           down_manifest.dimensions.size()!=2U || down_manifest.dimensions[0]!=10240U || down_manifest.dimensions[1]!=320U || down_manifest.byte_size!=3481600U ||
           up_manifest.dimensions.size()!=2U || up_manifest.dimensions[0]!=320U || up_manifest.dimensions[1]!=10240U ||
           (inject_manifest && (inject_manifest->dimensions.size()!=2U || inject_manifest->dimensions[0]!=10240U ||
                               inject_manifest->dimensions[1]!=4U || inject_manifest->byte_size!=163840U)))
            throw std::invalid_argument("next-tile HC actual tensor/workspace geometry differs");
        if(!temporary.address || temporary.address%16U || temporary.bytes<42240U ||
           temporary.bytes>std::numeric_limits<std::uintptr_t>::max()-temporary.address)
            throw std::invalid_argument("next-tile HC workspace alignment or extent differs");
        if(!norm_runtime.allocation_generation || !down_runtime.allocation_generation || !up_runtime.allocation_generation ||
           (inject_runtime && !inject_runtime->allocation_generation))
            throw std::invalid_argument("next-tile HC weight generation is missing");
        using hc_projection_prefetch::View;
        const View norm_owner{norm_runtime.address,norm_runtime.bytes},up_owner{up_runtime.address,up_runtime.bytes};
        hc_projection_prefetch::validate_view(norm_owner,norm_required);
        hc_projection_prefetch::validate_view(up_owner,up_required);
        const std::array<View,3> mutable_views{{{temporary.address,40960U},
            {temporary.address+40960U,temporary.bytes-40960U},
            {injection_output?injection_output->address:0U,injection_output?injection_output->bytes:0U}}};
        // Validate every active mutable interval before doing overlap arithmetic.
        for(std::size_t i=0;i<(injection_output?3U:2U);++i)
            hc_projection_prefetch::validate_view(mutable_views[i],i==0U?40960U:i==1U?1280U:16U);
        for(std::size_t i=0;i<(injection_output?3U:2U);++i)
            if(hc_projection_prefetch::overlaps(mutable_views[i],norm_owner) || hc_projection_prefetch::overlaps(mutable_views[i],up_owner))
                throw std::invalid_argument("next-tile HC projection workspace aliases norm/up weights");
        hc_projection_prefetch::validate_projection_views(
            {temporary.address,40960U},{temporary.address+40960U,temporary.bytes-40960U},
            {down_runtime.address,down_runtime.bytes},
            inject_runtime?std::optional<View>{{inject_runtime->address,inject_runtime->bytes}}:std::nullopt,
            injection_output?std::optional<View>{{injection_output->address,injection_output->bytes}}:std::nullopt);
    }
    std::optional<ExecutionBufferAddress> temporary;
    if (config.parallel)
        temporary = prepare_parallel_hyper_workspace(context, config, down_runtime, inject_runtime);
    return {&config, input, output, injection_output, temporary,
        &norm_manifest, &down_manifest, &up_manifest, inject_manifest,
        &norm_runtime, &down_runtime, &up_runtime, inject_runtime};
}

void preflight_hyper_mix(const ExecutionDispatchContext& context, void* user_data) {
    (void)prepare_hyper_mix(context, user_data);
}

void launch_hyper_mix(const ExecutionDispatchContext& context, void* user_data) {
    const auto prepared = prepare_hyper_mix(context, user_data);
    const auto& config = *prepared.config;
    const auto invoke = [&]<typename Storage>() {
        launch_typed<Storage>(context, config, prepared.input, prepared.output,
            prepared.injection_output ? &*prepared.injection_output : nullptr,
            *prepared.norm_manifest, *prepared.norm_runtime,
            *prepared.down_manifest, *prepared.down_runtime,
            *prepared.up_manifest, *prepared.up_runtime,
            prepared.inject_manifest, prepared.inject_runtime,
            prepared.temporary ? &*prepared.temporary : nullptr);
    };
    switch (config.activation) {
    case HipActivationStorage::bf16: invoke.template operator()<std::uint16_t>(); return;
    case HipActivationStorage::fp32: invoke.template operator()<float>(); return;
    }
    throw std::logic_error("unreachable hyper-mix activation storage");
}

std::string kernel_id(
    const ExecutionBindingRequest& request,
    const HipHyperMixLauncherConfig& config) {
    return
        "hip.hyper_mix." +
        std::string(
            storage_name(
                config.activation)) +
        ".h" +
        std::to_string(
            config.hidden_size) +
        ".hc" +
        std::to_string(
            config.hc_count) +
        ".lr" +
        std::to_string(
            config.lowrank) +
        ".r" +
        std::to_string(
            request.logical_rows) +
        ".c" +
        std::to_string(
            request.request_concurrency) + (config.parallel ? ".parallel" : "") +
        (config.staged_projections ? ".staged" : "") + (config.fused_injection ? ".fused-injection-v5" : "") + (config.typed_projection ? ".typed-q8-f32-v1" : "") + (config.prefetch_rows ? ".prefetch-rows8-v2" : "") + (config.prefetch_q8_output ? ".prefetch-q8-output-v1" : "") +
        (config.splitk_projection ? (config.splitk_projection_fp64 ? ".splitk-hc-proj-fp64-oracle-v1" : ".splitk-hc-proj-fp32-v1") : "") +
        (config.splitk_deferred_norm ? ".hc-deferred-norm-v1" : "") +
        (config.prefetch_hc_projection_next_tile ? std::string(hc_projection_next_tile_suffix) : "") +
        (config.lane_chain_output ? lane_chain_hyper_output_suffix : "");
}

std::uint64_t implementation_identity(
    const HipHyperMixLauncherConfig& config) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, kLauncherTag);
    hash_u64(hash, config.hidden_size);
    hash_u64(hash, config.hc_count);
    hash_u64(hash, config.lowrank);
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            config.activation));
    std::uint32_t epsilon_bits = 0;
    static_assert(
        sizeof(epsilon_bits) ==
        sizeof(config.rms_epsilon));
    std::memcpy(
        &epsilon_bits,
        &config.rms_epsilon,
        sizeof(epsilon_bits));
    hash_u64(hash, epsilon_bits);
    hash_u64(hash, config.parallel ? 1U : 0U);
    if (config.staged_projections) hash_u64(hash, 0x535441474544ULL);
    if (config.fused_injection) hash_u64(hash, 0x4655534544484305ULL);
    if (config.typed_projection) hash_u64(hash, 0x5459504544484301ULL);
    if (config.prefetch_rows) hash_u64(hash, 0x5052464852430802ULL);
    if (config.prefetch_q8_output) hash_u64(hash, 0x5052464855500801ULL);
    if (config.splitk_projection) hash_u64(hash, config.splitk_projection_fp64 ? 0x534b484346363401ULL : 0x534b484346333201ULL);
    if (config.splitk_deferred_norm) hash_u64(hash, 0x48434446524d5301ULL);
    if(config.lane_chain_output)hash_u64(hash,0x4c414e4548430001ULL);
    if(config.prefetch_hc_projection_next_tile)hash_u64(hash,0x48434e4558540001ULL);
    return hash;
}

struct PreparedHyperRows {
    PreparedHyperMix first;
    HyperRows rows{};
    std::uint32_t count = 0;
};

PreparedHyperRows prepare_hyper_rows(std::span<const ExecutionDispatchContext> rows,
    const HipHyperMixLauncherConfig& config) {
    validate_config(config);
    if (!config.parallel || !config.staged_projections || !config.fused_injection || !config.typed_projection ||
        !config.prefetch_rows || !config.prefetch_q8_output || config.splitk_projection_fp64 ||
        config.splitk_deferred_norm || config.prefetch_hc_projection_next_tile)
        throw std::invalid_argument("known-window HC rows support the typed prefetched Q8 path only");
    if (rows.size() < 2U || rows.size() > kWindowRows)
        throw std::invalid_argument("known-window HC rows require two to eight rows");
    const auto selected = kernel_id(ExecutionBindingRequest{}, config);
    PreparedHyperRows out;
    out.count = static_cast<std::uint32_t>(rows.size());
    std::vector<std::pair<std::uintptr_t, std::uint64_t>> writes, reads;
    const auto width = static_cast<std::uint64_t>(config.hidden_size) * config.hc_count;
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const auto& context = rows[r];
        if (!context.kernel_id || *context.kernel_id != selected || context.logical_rows != 1U ||
            context.request_concurrency != 1U || context.op != rows[0].op || context.step_id != rows[0].step_id)
            throw std::invalid_argument("known-window HC row differs from its serial R1 selection");
        auto prepared = prepare_hyper_mix(context, const_cast<HipHyperMixLauncherConfig*>(&config));
        if (r == 0U) out.first = prepared;
        else if (prepared.norm_runtime->address != out.first.norm_runtime->address ||
            prepared.down_runtime->address != out.first.down_runtime->address ||
            prepared.up_runtime->address != out.first.up_runtime->address ||
            (prepared.inject_runtime ? prepared.inject_runtime->address : 0U) !=
                (out.first.inject_runtime ? out.first.inject_runtime->address : 0U))
            throw std::invalid_argument("known-window HC rows bind different weights");
        const auto& temporary = *prepared.temporary;
        if (temporary.address % 16U) throw std::invalid_argument("known-window HC row workspace is unaligned");
        out.rows.input[r] = reinterpret_cast<const void*>(prepared.input.address);
        out.rows.output[r] = reinterpret_cast<void*>(prepared.output.address);
        out.rows.normalized[r] = reinterpret_cast<float*>(temporary.address);
        out.rows.low[r] = out.rows.normalized[r] + width;
        out.rows.injection[r] = prepared.injection_output ? reinterpret_cast<float*>(prepared.injection_output->address) : nullptr;
        writes.emplace_back(temporary.address, (width + config.lowrank) * sizeof(float));
        writes.emplace_back(prepared.output.address, prepared.output.bytes);
        if (prepared.injection_output) writes.emplace_back(prepared.injection_output->address, prepared.injection_output->bytes);
        reads.emplace_back(prepared.input.address, prepared.input.bytes);
    }
    for (const auto* runtime : {out.first.norm_runtime, out.first.down_runtime, out.first.up_runtime, out.first.inject_runtime})
        if (runtime) reads.emplace_back(runtime->address, runtime->bytes);
    const auto overlap = [](const auto& x, const auto& y) { return x.first < y.first + y.second && y.first < x.first + x.second; };
    for (std::size_t i = 0; i < writes.size(); ++i) {
        for (std::size_t j = i + 1; j < writes.size(); ++j)
            if (overlap(writes[i], writes[j])) throw std::invalid_argument("known-window HC row outputs overlap");
        for (const auto& read : reads)
            if (overlap(writes[i], read)) throw std::invalid_argument("known-window HC row output aliases an input");
    }
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    check_hip(hipStreamIsCapturing(hip_execution_stream(), &capture), "query known-window HC rows capture");
    if (capture != hipStreamCaptureStatusNone || hip_graph_status() ||
        (current_hip_launch_context() && current_hip_launch_context()->graph_capture))
        throw std::invalid_argument("known-window HC rows are eager only");
    return out;
}

}  // namespace

void hyper_mix_known_window_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipHyperMixLauncherConfig& config) {
    (void)prepare_hyper_rows(rows, config);
}

namespace {
template <class Rows> std::uint32_t looped(const Rows& rows) {
    if constexpr (std::is_same_v<Rows, HyperTileRows>) return rows.tiles; else return 1U;
}
// Grid barrier state of the persistent HC mix, one per stream: the device arrival counter only grows, and
// launches on one stream run in order, so the host knows each launch's base. Allocated on first use.
std::atomic<std::uint64_t> persistent_hyper_launches{0};
std::atomic<unsigned long long*> persistent_hyper_last_trace{nullptr};
struct PersistentHyperBarrier {
    unsigned* counter = nullptr;
    float* partial = nullptr; // [8 rows][4 inject outputs][16 tiles]
    unsigned long long* trace = nullptr; // [blocks][8] phase timestamps
    float* workspace = nullptr; // folded inject: [8 rows][width + rank] when the arena workspace aliases its inputs
    unsigned issued = 0;
    unsigned queued = 0; // B work units handed out so far (queue at counter + 16)
};
PersistentHyperBarrier& persistent_hyper_barrier(hipStream_t stream) {
    static std::mutex mutex;
    static std::map<hipStream_t, PersistentHyperBarrier> barriers;
    const std::lock_guard lock(mutex);
    auto& b = barriers[stream];
    if (!b.counter) {
        void* memory = nullptr;
        check_hip(hipMalloc(&memory, 256U + kWindowRows * kPersistStreams * 16U * sizeof(float) + kPersistBlocks * 24U * 8U + 16U * 8U +
            kWindowRows * (kPersistWidth + kPersistRank) * sizeof(float)),
            "persistent HC barrier allocation");
        check_hip(hipMemsetAsync(memory, 0, 256U, stream), "persistent HC barrier reset");
        b.counter = static_cast<unsigned*>(memory);
        b.partial = reinterpret_cast<float*>(static_cast<unsigned char*>(memory) + 256U);
        b.trace = reinterpret_cast<unsigned long long*>(b.partial + kWindowRows * kPersistStreams * 16U);
        b.workspace = reinterpret_cast<float*>(b.trace + kPersistBlocks * 24U + 16U); // + per-wave B ends (static form)
    }
    return b;
}
// STRIX_HC_PERSISTENT=1: window rows 2/4/8 of the split-K + rank-320 lane-chain form run as one persistent
// launch when all kPersistBlocks blocks fit on the device at once; otherwise the three-kernel form runs.
bool persistent_up_late() {
    static const bool late = [] { const char* v = std::getenv("STRIX_HC_PERSISTENT_UP_LATE"); return v && v[0] == '1'; }();
    return late;
}
// All kPersistBlocks blocks of one instantiation fit on the device at once (checked once per instantiation).
template <auto Kernel, std::uint32_t Count, bool Inject>
bool persistent_kernel_resident() {
    const auto kernel = Kernel;
    static const bool resident = [&] {
        int device = 0, per_cu = 0;
        hipDeviceProp_t properties{};
        if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&properties, device) != hipSuccess ||
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, kernel, 512, 0) != hipSuccess) return false;
        hipFuncAttributes attributes{};
        if (hipFuncGetAttributes(&attributes, reinterpret_cast<const void*>(kernel)) != hipSuccess) return false;
        // The occupancy query counts one 16-wave block per WGP on gfx1151 (it reports WGPs as CUs); a WGP has
        // 4 SIMDs of 1536 VGPRs and 128 KiB of LDS, so two blocks fit when each wave needs <= 192 VGPRs and no
        // scratch. The fixtures and the decode runs have all 40 blocks resident at once (the barrier traps if not).
        const bool wgp_pair = std::string_view(properties.gcnArchName).starts_with("gfx1151") && attributes.numRegs <= 192 &&
            attributes.localSizeBytes == 0U && 2U * attributes.sharedSizeBytes <= 128U * 1024U;
        const bool fits = static_cast<long>(per_cu) * properties.multiProcessorCount >= static_cast<long>(kPersistBlocks) ||
            (wgp_pair && 2L * properties.multiProcessorCount >= static_cast<long>(kPersistBlocks));
        if (std::getenv("STRIX_HC_PERSISTENT_VERBOSE"))
            std::fprintf(stderr, "persistent HC rows=%u inject=%d: %d blocks/CU x %d CUs, %d VGPRs, %zu B LDS, %zu B scratch -> %s\n",
                Count, Inject ? 1 : 0, per_cu, properties.multiProcessorCount, attributes.numRegs, attributes.sharedSizeBytes,
                attributes.localSizeBytes, fits ? "persistent" : "three-kernel form");
        return fits;
    }();
    return resident;
}
template <typename Storage, std::uint32_t Count, bool Inject>
bool persistent_hyper_resident() {
    return persistent_kernel_resident<rows_persistent_hyper_mix_kernel<Storage, Count, Inject>, Count, Inject>();
}
void register_persistent_trace_report() {
    if (std::getenv("STRIX_HC_PERSISTENT_TRACE")) {
        static const bool registered = [] {
            std::atexit([] {
                const auto* trace = persistent_hyper_last_trace.load();
                unsigned long long sums[12] = {};
                if (!trace || hipMemcpy(sums, trace + kPersistBlocks * 8U, sizeof(sums), hipMemcpyDeviceToHost) != hipSuccess) return;
                for (unsigned c = 0; c < 3U; ++c) if (sums[4U * c + 3U])
                    std::fprintf(stderr, "persistent HC rows=%u: %llu launches, block 0 mean us: A+barrier %.2f, B+barrier %.2f, C %.2f\n",
                        2U << c, sums[4U * c + 3U], static_cast<double>(sums[4U * c]) / 100.0 / static_cast<double>(sums[4U * c + 3U]),
                        static_cast<double>(sums[4U * c + 1U]) / 100.0 / static_cast<double>(sums[4U * c + 3U]),
                        static_cast<double>(sums[4U * c + 2U]) / 100.0 / static_cast<double>(sums[4U * c + 3U]));
            });
            return true;
        }();
        (void)registered;
    }
}
// STRIX_HC_PERSISTENT_B selects the static split-K phase B of the persistent launch: the units of down weights
// kept in flight (1, 2 or 4), then 'e' to request the first ones during A and/or 'u' to request the up rows
// before the last unit, 'c' for the coalesced down reads, 'w' for the coalesced (wave-contiguous) up rows
// (variants 1, 2, 4, 2e, 4e, 4u, 4eu, 1c, 2c, 2ec, 1cw, 2ecw, 1w). Unset or 0 keeps the work-queue form.
constexpr std::array<std::string_view, 13> kStaticVariants{"1", "2", "4", "2e", "4e", "4u", "4eu", "1c", "2c", "2ec", "1cw", "2ecw", "1w"};
unsigned persistent_static_variant() {
    static const unsigned variant = [] {
        const char* v = std::getenv("STRIX_HC_PERSISTENT_B");
        if (!v || !*v || std::string_view(v) == "0") return 0U;
        for (unsigned i = 0; i < kStaticVariants.size(); ++i) if (kStaticVariants[i] == v) return i + 1U;
        std::fprintf(stderr, "STRIX_HC_PERSISTENT_B=%s is not a variant: work-queue form\n", v);
        return 0U;
    }();
    return variant;
}
template <typename Storage, std::uint32_t Count, bool Inject, unsigned Prefetch, bool Early, bool UpEarly, bool Coalesced = false,
    bool UpCoalesced = false>
bool launch_static_hyper_rows(const PreparedHyperMix& p, const HyperRows& rows, const HipHyperMixLauncherConfig& config,
    const PersistInjectRows& injected) {
    constexpr auto kernel = rows_static_hyper_mix_kernel<Storage, Count, Inject, Prefetch, Early, UpEarly, Coalesced, UpCoalesced>;
    if (!persistent_kernel_resident<kernel, Count, Inject>()) return false;
    const auto stream = hip_execution_stream();
    auto& barrier = persistent_hyper_barrier(stream);
    const auto* inject = p.inject_runtime ? reinterpret_cast<const float*>(p.inject_runtime->address) : nullptr;
    hipLaunchKernelGGL(kernel, dim3(kPersistBlocks), dim3(512U), 0,
        stream, rows, reinterpret_cast<const void*>(p.norm_runtime->address), p.norm_manifest->ggml_type, config.rms_epsilon,
        config.hidden_size, reinterpret_cast<const unsigned char*>(p.down_runtime->address), inject,
        reinterpret_cast<const unsigned char*>(p.up_runtime->address), barrier.counter, barrier.issued, injected,
        std::getenv("STRIX_HC_PERSISTENT_TRACE") ? barrier.trace : nullptr);
    persistent_hyper_last_trace.store(barrier.trace);
    register_persistent_trace_report();
    check_hip(hipGetLastError(), "known-window static persistent HC rows launch");
    barrier.issued += 2U * kPersistBlocks; // no queue units
    persistent_hyper_launches.fetch_add(1U, std::memory_order_relaxed);
    return true;
}
template <typename Storage, std::uint32_t Count, bool Inject = false>
bool launch_persistent_hyper_rows(const PreparedHyperMix& p, const HyperRows& rows, const HipHyperMixLauncherConfig& config,
    const PersistInjectRows& injected = {}) {
    switch (persistent_static_variant()) {
    case 1U: return launch_static_hyper_rows<Storage, Count, Inject, 1U, false, false>(p, rows, config, injected);
    case 2U: return launch_static_hyper_rows<Storage, Count, Inject, 2U, false, false>(p, rows, config, injected);
    case 3U: return launch_static_hyper_rows<Storage, Count, Inject, 4U, false, false>(p, rows, config, injected);
    case 4U: return launch_static_hyper_rows<Storage, Count, Inject, 2U, true, false>(p, rows, config, injected);
    case 5U: return launch_static_hyper_rows<Storage, Count, Inject, 4U, true, false>(p, rows, config, injected);
    case 6U: return launch_static_hyper_rows<Storage, Count, Inject, 4U, false, true>(p, rows, config, injected);
    case 7U: return launch_static_hyper_rows<Storage, Count, Inject, 4U, true, true>(p, rows, config, injected);
    case 8U: return launch_static_hyper_rows<Storage, Count, Inject, 1U, false, false, true>(p, rows, config, injected);
    case 9U: return launch_static_hyper_rows<Storage, Count, Inject, 2U, false, false, true>(p, rows, config, injected);
    case 10U: return launch_static_hyper_rows<Storage, Count, Inject, 2U, true, false, true>(p, rows, config, injected);
    case 11U: return launch_static_hyper_rows<Storage, Count, Inject, 1U, false, false, true, true>(p, rows, config, injected);
    case 12U: return launch_static_hyper_rows<Storage, Count, Inject, 2U, true, false, true, true>(p, rows, config, injected);
    case 13U: return launch_static_hyper_rows<Storage, Count, Inject, 1U, false, false, false, true>(p, rows, config, injected);
    default: break;
    }
    if (!persistent_hyper_resident<Storage, Count, Inject>()) return false;
    const auto stream = hip_execution_stream();
    auto& barrier = persistent_hyper_barrier(stream);
    const auto* inject = p.inject_runtime ? reinterpret_cast<const float*>(p.inject_runtime->address) : nullptr;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_persistent_hyper_mix_kernel<Storage, Count, Inject>), dim3(kPersistBlocks), dim3(512U), 0,
        stream, rows, reinterpret_cast<const void*>(p.norm_runtime->address), p.norm_manifest->ggml_type, config.rms_epsilon,
        config.hidden_size, reinterpret_cast<const unsigned char*>(p.down_runtime->address), inject,
        reinterpret_cast<const unsigned char*>(p.up_runtime->address), barrier.counter, barrier.issued, barrier.counter + 16, barrier.queued, injected,
        std::getenv("STRIX_HC_PERSISTENT_TRACE") ? barrier.trace : nullptr, persistent_up_late());
    persistent_hyper_last_trace.store(barrier.trace);
    register_persistent_trace_report();
    check_hip(hipGetLastError(), "known-window persistent HC rows launch");
    barrier.issued += 2U * kPersistBlocks;
    // Every unit is taken once and every block takes one past the end before it leaves the loop.
    barrier.queued += kPersistRank / kPersistUnitOutputs + (inject ? kPersistStreams : 0U) + kPersistBlocks;
    persistent_hyper_launches.fetch_add(1U, std::memory_order_relaxed);
    return true;
}
// Fixtures: the separate launches as the reference of the persistent form within one process.
std::atomic<bool> persistent_hyper_testing_off{false};
bool persistent_hyper_rows_enabled() {
    return !persistent_hyper_testing_off.load(std::memory_order_relaxed) && strix_hc_persistent_enabled();
}
// STRIX_MTP_HC_PERSISTENT=1 (STRIX_DISPATCH_FUSE=1): one HC mix of the draft chain (one row, BF16 streams,
// F32 norm, Q8_0 down/up, F32 [4][10240] injection or none) as the persistent launch: one dispatch instead of
// the normalisation, the one-row split-K projection and the lane-chain output. The persistent kernel keeps
// their expressions and order (the HC rows fixtures check it against the serial R1 launches bit for bit).
bool launch_persistent_hyper_draft(const std::uint16_t* input, const float* norm, const void* down, const float* inject,
    const void* up, float* normalized, float* low, float* injection, std::uint16_t* output, float epsilon, hipStream_t stream) {
    static const bool enabled = [] { const char* v = std::getenv("STRIX_MTP_HC_PERSISTENT"); return v && v[0] == '1'; }();
    if (!enabled || !persistent_hyper_resident<std::uint16_t, 1U, false>()) return false;
    HyperRows rows{};
    rows.input[0] = input; rows.output[0] = output; rows.normalized[0] = normalized; rows.low[0] = low; rows.injection[0] = injection;
    auto& barrier = persistent_hyper_barrier(stream);
    // STRIX_HC_PERSISTENT_B (any variant): the draft row takes the static split-K phase with one unit in flight.
    if (persistent_static_variant() != 0U) {
        constexpr auto kernel = rows_static_hyper_mix_kernel<std::uint16_t, 1U, false, 1U, false, false>;
        if (persistent_kernel_resident<kernel, 1U, false>()) {
            hipLaunchKernelGGL(kernel, dim3(kPersistBlocks), dim3(512U), 0, stream, rows, static_cast<const void*>(norm), 0U, epsilon,
                kPersistHidden, static_cast<const unsigned char*>(down), inject, static_cast<const unsigned char*>(up), barrier.counter, barrier.issued,
                PersistInjectRows{}, static_cast<unsigned long long*>(nullptr));
            check_hip(hipGetLastError(), "MTP draft static persistent HC launch");
            barrier.issued += 2U * kPersistBlocks;
            persistent_hyper_launches.fetch_add(1U, std::memory_order_relaxed);
            return true;
        }
    }
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_persistent_hyper_mix_kernel<std::uint16_t, 1U, false>), dim3(kPersistBlocks), dim3(512U), 0,
        stream, rows, static_cast<const void*>(norm), 0U, epsilon, kPersistHidden, static_cast<const unsigned char*>(down), inject,
        static_cast<const unsigned char*>(up), barrier.counter, barrier.issued, barrier.counter + 16, barrier.queued, PersistInjectRows{},
        static_cast<unsigned long long*>(nullptr), persistent_up_late());
    check_hip(hipGetLastError(), "MTP draft persistent HC launch");
    barrier.issued += 2U * kPersistBlocks;
    barrier.queued += kPersistRank / kPersistUnitOutputs + (inject ? kPersistStreams : 0U) + kPersistBlocks;
    persistent_hyper_launches.fetch_add(1U, std::memory_order_relaxed);
    return true;
}
// Window sizes that take the persistent HC form. STRIX_HC_PERSISTENT_ROWS=2,4,8 (comma list) selects them; unset,
// an explicit STRIX_HC_PERSISTENT=1 keeps every size and STRIX_DISPATCH_FUSE alone takes 2 and 4: at 8 rows the
// persistent split-K phase costs more than the two dispatches it removes (kernel trace, 2026-10-21).
bool persistent_hyper_rows_size(std::uint32_t count) {
    static const unsigned mask = [] {
        const char* v = std::getenv("STRIX_HC_PERSISTENT_ROWS");
        if (!v || !*v) {
            const char* p = std::getenv("STRIX_HC_PERSISTENT");
            return p && *p ? ~0U : (1U << 2U) | (1U << 4U);
        }
        unsigned m = 0;
        for (const char* c = v; *c;) {
            char* end = nullptr;
            const auto n = std::strtoul(c, &end, 10);
            if (end == c) { ++c; continue; }
            if (n < 32U) m |= 1U << n;
            c = end;
        }
        return m;
    }();
    return count < 32U && ((mask >> count) & 1U) != 0U;
}
bool persistent_decline(const char* why) {
    if (std::getenv("STRIX_HC_PERSISTENT_VERBOSE")) std::fprintf(stderr, "persistent HC inject fold declined: %s\n", why);
    return false;
}
bool persistent_hyper_rows_eligible(const PreparedHyperRows& prepared, const HipHyperMixLauncherConfig& config) {
    const auto& p = prepared.first;
    const auto count = prepared.count;
    return persistent_hyper_rows_enabled() && !std::getenv("STRIX_HC_ROWS_FUSED_NORM") && config.splitk_projection &&
        config.lane_chain_output && config.hidden_size == kPersistHidden && config.hc_count == kPersistStreams &&
        config.lowrank == kPersistRank && p.down_manifest->ggml_type == 8U && p.up_manifest->ggml_type == 8U &&
        (!p.inject_runtime || p.inject_manifest->ggml_type == 0U) && (count == 2U || count == 4U || count == 8U) &&
        persistent_hyper_rows_size(count) &&
        // The persistent kernel has no Q6P32 form: weights with a registered copy keep the launches that read it.
        !q6p::registered(reinterpret_cast<const void*>(p.down_runtime->address)) &&
        !q6p::registered(reinterpret_cast<const void*>(p.up_runtime->address));
}

template <class Rows>
void launch_hyper_rows(const PreparedHyperRows& prepared, const Rows& rows, const HipHyperMixLauncherConfig& config,
    std::uint32_t tiles, bool normalized_ready = false) {
    const auto& p = prepared.first;
    const auto count = prepared.count;
    if constexpr (std::is_same_v<Rows, HyperRows>) {
        if (persistent_hyper_rows_eligible(prepared, config)) {
            const auto run = [&]<typename Storage>() {
                return count == 2U ? launch_persistent_hyper_rows<Storage, 2U>(p, rows, config) :
                    count == 4U ? launch_persistent_hyper_rows<Storage, 4U>(p, rows, config) :
                    launch_persistent_hyper_rows<Storage, 8U>(p, rows, config);
            };
            const bool launched = config.activation == HipActivationStorage::bf16 ? run.template operator()<std::uint16_t>() :
                run.template operator()<float>();
            if (launched) {
                if (config.arithmetic_stats) {
                    config.arithmetic_stats->materialized_norm_submissions.fetch_add(tiles, std::memory_order_relaxed);
                    config.arithmetic_stats->splitk_f32_submissions.fetch_add(tiles, std::memory_order_relaxed);
                }
                return;
            }
        }
    }
    const auto width = config.hidden_size * config.hc_count;
    const auto* inject = p.inject_runtime ? reinterpret_cast<const void*>(p.inject_runtime->address) : nullptr;
    const auto invoke = [&]<typename Storage>() {
        // STRIX_HC_ROWS_FUSED_NORM=1: the split-K projection also builds the normalisation.
        static const bool fused_norm = std::getenv("STRIX_HC_ROWS_FUSED_NORM") != nullptr;
        const bool fused = !normalized_ready && fused_norm && config.splitk_projection && config.hc_count <= 8U &&
            config.hidden_size % 4U == 0U && count * config.hc_count <= 64U;
        if (!fused && !normalized_ready) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_prepare_hyper_norm_kernel<Storage, Rows>), dim3(config.hc_count, count, tiles),
                dim3(kThreads), 0, hip_execution_stream(), rows, reinterpret_cast<const void*>(p.norm_runtime->address),
                p.norm_manifest->ggml_type, config.hidden_size, config.rms_epsilon);
            check_hip(hipGetLastError(), "known-window HC rows normalization launch");
        }
        if (fused) {
            static const unsigned fused_per_block = [] {
                const char* v = std::getenv("STRIX_HC_ROWS_SPLITK_OUTPUTS");
                const unsigned n = v ? static_cast<unsigned>(std::atoi(v)) : 2U;
                return n == 4U ? 4U : 2U;
            }();
            const auto launch = [&]<unsigned outputs>() {
                const auto blocks = (config.lowrank + outputs - 1U) / outputs;
                hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_splitk_norm_hyper_projection_kernel<outputs, Storage, Rows>),
                    dim3(blocks + (inject ? (config.hc_count + outputs - 1U) / outputs : 0U), 1U, tiles), dim3(512U), 0,
                    hip_execution_stream(), rows, count, reinterpret_cast<const void*>(p.down_runtime->address),
                    config.hc_count, config.lowrank, inject, blocks, reinterpret_cast<const void*>(p.norm_runtime->address),
                    p.norm_manifest->ggml_type, config.hidden_size, config.rms_epsilon);
            };
            if (fused_per_block == 2U) launch.template operator()<2U>(); else launch.template operator()<4U>();
        } else if (config.splitk_projection) {
            // Outputs per block only regroups independent output chains (bit-identical);
            // STRIX_HC_ROWS_SPLITK_OUTPUTS=1|2|4|8 trades blocks in flight against input re-reads (2 measured best).
            static const unsigned per_block = [] {
                const char* v = std::getenv("STRIX_HC_ROWS_SPLITK_OUTPUTS");
                const unsigned n = v ? static_cast<unsigned>(std::atoi(v)) : 2U;
                return n == 1U || n == 4U || n == 8U ? n : 2U;
            }();
            // The registered Q6P32 copy of the down tensor replaces it (same bits).
            const auto* q6_down = q6p::find(reinterpret_cast<const void*>(p.down_runtime->address));
            const bool q6 = q6_down && q6_down->kind == q6p::Kind::q6p32 && q6_down->rows == config.lowrank &&
                q6_down->cols == 10240U && width == 10240U;
            const auto launch = [&]<unsigned outputs>() {
                const auto blocks = (config.lowrank + outputs - 1U) / outputs;
                if (q6)
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_splitk_hyper_projection_kernel<outputs, Rows, true>),
                        dim3(blocks + (inject ? (config.hc_count + outputs - 1U) / outputs : 0U), 1U, tiles), dim3(512U), 0,
                        hip_execution_stream(), rows, count, reinterpret_cast<const void*>(q6_down->base),
                        config.hc_count, config.lowrank, inject, blocks);
                else
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_splitk_hyper_projection_kernel<outputs, Rows>),
                        dim3(blocks + (inject ? (config.hc_count + outputs - 1U) / outputs : 0U), 1U, tiles), dim3(512U), 0,
                        hip_execution_stream(), rows, count, reinterpret_cast<const void*>(p.down_runtime->address),
                        config.hc_count, config.lowrank, inject, blocks);
            };
            if (per_block == 1U) launch.template operator()<1U>();
            else if (per_block == 2U) launch.template operator()<2U>();
            else if (per_block == 8U) launch.template operator()<8U>();
            else launch.template operator()<4U>();
        } else {
            const auto blocks = (config.lowrank + kStagedOutputs - 1U) / kStagedOutputs;
            hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_batched_typed_hyper_projection_kernel<Rows>),
                dim3(blocks + (inject ? (config.hc_count + kStagedOutputs - 1U) / kStagedOutputs : 0U), 1U, tiles),
                dim3(kStagedThreads), 0, hip_execution_stream(), rows, count,
                reinterpret_cast<const void*>(p.down_runtime->address), width, config.hc_count, config.lowrank, inject, blocks);
        }
        check_hip(hipGetLastError(), "known-window HC rows projection launch");
        if (config.lane_chain_output) {
            const auto lds = lane_chain_hyper_output_lds(config.hc_count, config.lowrank) +
                static_cast<std::uint64_t>(count - 1U) * config.hc_count * 32U * sizeof(float);
            // STRIX_HC_LANE_CHAIN_REGS=0: keep the LDS-staged form for rank 320 as well.
            static const bool regs = [] { const char* v = std::getenv("STRIX_HC_LANE_CHAIN_REGS"); return !v || v[0] != '0'; }();
            // The registered Q6P32 copy of the up tensor replaces it (same bits).
            const auto* q6_up = q6p::find(reinterpret_cast<const void*>(p.up_runtime->address));
            const bool q6 = q6_up && q6_up->kind == q6p::Kind::q6p32 && q6_up->cols == config.lowrank &&
                q6_up->rows == config.hidden_size * config.hc_count;
            const auto* up = q6 ? reinterpret_cast<const void*>(q6_up->base) : reinterpret_cast<const void*>(p.up_runtime->address);
            // Q8_0 windows of 8 rows run the row-split output form with 2 row groups per block (in-run R8 window
            // 3.22 -> 2.98 ms per window; 4 and 8 groups were not faster, and at 4 rows the register form wins).
            // STRIX_HC_LANE_SPLIT=0 keeps the LDS form; =2|4|8 forces that many groups (also for 4-row windows).
            static const int split_env = [] { const char* v = std::getenv("STRIX_HC_LANE_SPLIT"); return v ? std::atoi(v) : -1; }();
            const auto output = [&]<std::uint32_t Count, bool Q6>() {
                const unsigned split = split_env >= 0 ? static_cast<unsigned>(split_env) : (Count == 8U ? 2U : 0U);
                if constexpr (!Q6 && (Count == 4U || Count == 8U)) if (split >= 2U && split <= Count && config.hc_count * 32U * split <= 1024U) {
                    const auto launch_split = [&]<std::uint32_t G>() {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_lane_chain_split_hyper_output_kernel<Storage, Count, G, Rows>),
                            dim3(config.hidden_size / 32U, 1U, tiles / looped(rows)), dim3(32U * config.hc_count * G),
                            static_cast<std::size_t>(lds), hip_execution_stream(), rows, up, config.hidden_size, config.hc_count,
                            config.lowrank);
                    };
                    if (split == 2U) launch_split.template operator()<2U>();
                    else if (split == 4U) launch_split.template operator()<4U>();
                    else if constexpr (Count == 8U) launch_split.template operator()<8U>();
                    else launch_split.template operator()<4U>();
                    return;
                }
                // Register form only up to 4 rows: at 8 rows it runs out of registers (32.1 -> 37.3 us).
                if constexpr (Count != 0U && Count <= 4U) if (regs && config.lowrank == 320U) {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_lane_chain_hyper_output_regs_kernel<Storage, Count, Rows, Q6>),
                        dim3(config.hidden_size / 32U, 1U, tiles), dim3(32U * config.hc_count),
                        static_cast<std::size_t>(Count) * config.hc_count * 32U * sizeof(float), hip_execution_stream(),
                        rows, up, config.hidden_size, config.hc_count);
                    return;
                }
                hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_lane_chain_hyper_output_kernel<Storage, Count, Rows, Q6>),
                    dim3(config.hidden_size / 32U, 1U, tiles / looped(rows)),
                    dim3(32U * config.hc_count), static_cast<std::size_t>(lds), hip_execution_stream(), rows, count,
                    up, config.hidden_size, config.hc_count, config.lowrank);
            };
            const auto by_count = [&]<bool Q6>() {
                if (count == 2U) output.template operator()<2U, Q6>();
                else if (count == 4U) output.template operator()<4U, Q6>();
                else if (count == 8U) output.template operator()<8U, Q6>();
                else output.template operator()<0U, Q6>();
            };
            if (q6) by_count.template operator()<true>(); else by_count.template operator()<false>();
        } else {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_q8_batched_hyper_output_kernel<Storage, Rows>),
                dim3((config.hidden_size + kStagedOutputs - 1U) / kStagedOutputs, 1U, tiles), dim3(kStagedThreads), 0,
                hip_execution_stream(), rows, count, reinterpret_cast<const void*>(p.up_runtime->address),
                config.hidden_size, config.hc_count, config.lowrank);
        }
        check_hip(hipGetLastError(), "known-window HC rows output launch");
    };
    switch (config.activation) {
    case HipActivationStorage::bf16: invoke.template operator()<std::uint16_t>(); break;
    case HipActivationStorage::fp32: invoke.template operator()<float>(); break;
    }
    if (config.arithmetic_stats) {
        config.arithmetic_stats->materialized_norm_submissions.fetch_add(tiles, std::memory_order_relaxed);
        if (config.splitk_projection) config.arithmetic_stats->splitk_f32_submissions.fetch_add(tiles, std::memory_order_relaxed);
    }
}
}  // namespace

std::uint64_t hip_hyper_mix_testing_persistent_launches() { return persistent_hyper_launches.load(); }
void hip_hyper_mix_testing_separate_launches(bool separate) { persistent_hyper_testing_off.store(separate); }
std::vector<std::uint64_t> hip_hyper_mix_testing_persistent_trace() {
    std::vector<std::uint64_t> out(kPersistBlocks * 24U + 16U);
    if (const auto* trace = persistent_hyper_last_trace.load())
        check_hip(hipMemcpy(out.data(), trace, out.size() * 8U, hipMemcpyDeviceToHost), "read persistent HC trace");
    return out;
}

bool hyper_mix_known_window_rows_launch_injected(std::span<const ExecutionDispatchContext> inject_rows,
    const HipElementwiseLauncherConfig& inject_config, std::span<const ExecutionDispatchContext> rows,
    const HipHyperMixLauncherConfig& config) {
    const bool enabled = strix_hc_persistent_inject_enabled();
    if (!enabled || inject_rows.size() != rows.size()) return false;
    const auto prepared = prepare_hyper_rows(rows, config);
    if (!persistent_hyper_rows_eligible(prepared, config) || inject_config.activation != config.activation ||
        inject_config.hidden_size != config.hidden_size || inject_config.hc_count != config.hc_count)
        return persistent_decline("HC step or activation not eligible");
    const auto views = inject_known_window_rows_views(inject_rows, inject_config);
    const auto count = prepared.count;
    const auto& r = prepared.rows;
    const std::uint64_t value = config.activation == HipActivationStorage::bf16 ? 2U : 4U;
    const auto at = [](const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer); };
    struct Extent { std::uintptr_t low, high; };
    const auto overlap = [](Extent a, Extent b) { return a.low < b.high && b.low < a.high; };
    PersistInjectRows injected{};
    for (std::uint32_t t = 0; t < count; ++t) {
        // The mix reads exactly the streams the inject writes.
        if (views.output[t] != r.input[t]) return persistent_decline("the HC step does not read the injected streams");
        injected.hyper[t] = views.first[t]; injected.hidden[t] = views.second[t]; injected.weights[t] = views.weights[t];
    }
    // Before the first grid barrier, block (row, stream) writes its slice of the injected streams and of the
    // normalised row while other blocks still read their inject inputs: no written stream may reach any
    // row's inject inputs (a row's own streams updated in place excepted). The window arena may hand the HC
    // workspace (normalised row and low values, dead after this step) the inject's input memory; the folded
    // launch then works in an owner-held workspace and leaves the arena workspace unwritten.
    bool private_workspace = false;
    for (std::uint32_t t = 0; t < count; ++t) {
        const Extent written[] = {{at(r.input[t]), at(r.input[t]) + kPersistWidth * value},
            {at(r.normalized[t]), at(r.normalized[t]) + kPersistWidth * sizeof(float)}};
        for (std::uint32_t o = 0; o < count; ++o) {
            const Extent hyper{at(views.first[o]), at(views.first[o]) + kPersistWidth * value};
            const Extent read[] = {{at(views.second[o]), at(views.second[o]) + kPersistHidden * value},
                {at(views.weights[o]), at(views.weights[o]) + kPersistStreams * sizeof(float)}};
            for (std::size_t w = 0; w < 2U; ++w) {
                bool alias = false;
                for (const auto& x : read) alias = alias || overlap(written[w], x);
                alias = alias || (overlap(written[w], hyper) && !(w == 0U && o == t && views.first[t] == r.input[t]));
                if (alias && w == 0U) return persistent_decline("streams alias an inject input");
                private_workspace = private_workspace || alias;
            }
        }
    }
    auto used = r;
    if (private_workspace) {
        auto* workspace = persistent_hyper_barrier(hip_execution_stream()).workspace;
        for (std::uint32_t t = 0; t < count; ++t) {
            used.normalized[t] = workspace + static_cast<std::uint64_t>(t) * (kPersistWidth + kPersistRank);
            used.low[t] = used.normalized[t] + kPersistWidth;
        }
    }
    const auto run = [&]<typename Storage>() {
        return count == 2U ? launch_persistent_hyper_rows<Storage, 2U, true>(prepared.first, used, config, injected) :
            count == 4U ? launch_persistent_hyper_rows<Storage, 4U, true>(prepared.first, used, config, injected) :
            launch_persistent_hyper_rows<Storage, 8U, true>(prepared.first, used, config, injected);
    };
    const bool launched = config.activation == HipActivationStorage::bf16 ? run.template operator()<std::uint16_t>() :
        run.template operator()<float>();
    if (launched && config.arithmetic_stats) {
        config.arithmetic_stats->materialized_norm_submissions.fetch_add(1U, std::memory_order_relaxed);
        config.arithmetic_stats->splitk_f32_submissions.fetch_add(1U, std::memory_order_relaxed);
    }
    return launched;
}

void hyper_mix_known_window_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipHyperMixLauncherConfig& config) {
    const auto prepared = prepare_hyper_rows(rows, config);
    launch_hyper_rows(prepared, prepared.rows, config, 1U);
}
void hyper_mix_known_window_cohort_launch(std::span<const std::span<const ExecutionDispatchContext>> segments,
    const HipHyperMixLauncherConfig& config) {
    if(segments.size()<2U || segments.size()>8U || !config.splitk_projection || !config.lane_chain_output)
        throw std::invalid_argument("wide HC cohort requires split-K and lane-chain output");
    auto lead=prepare_hyper_rows(segments[0],config);
    HyperCohortRows rows{};
    std::uint32_t count=0;
    for(const auto segment:segments) {
        const auto p=prepare_hyper_rows(segment,config);
        if(count+p.count>32U || p.first.down_runtime->address!=lead.first.down_runtime->address ||
            p.first.up_runtime->address!=lead.first.up_runtime->address ||
            p.first.norm_runtime->address!=lead.first.norm_runtime->address ||
            (p.first.inject_runtime?p.first.inject_runtime->address:0U)!=
                (lead.first.inject_runtime?lead.first.inject_runtime->address:0U))
            throw std::invalid_argument("wide HC cohort rows or weights differ");
        for(std::uint32_t r=0;r<p.count;++r,++count) {
            rows.input[count]=p.rows.input[r];rows.output[count]=p.rows.output[r];
            rows.normalized[count]=p.rows.normalized[r];rows.low[count]=p.rows.low[r];rows.injection[count]=p.rows.injection[r];
        }
    }
    lead.count=count;
    launch_hyper_rows(lead,rows,config,1U);
}

namespace {
std::atomic<std::uint64_t> folded_launches{0};
std::atomic<const float*> folded_workspace{nullptr};
// Owner-held HC workspace per stream for folded launches whose arena workspace aliases a folded input.
float* folded_hc_workspace(hipStream_t stream, std::uint64_t bytes) {
    struct Buffer { void* memory = nullptr; std::uint64_t bytes = 0; };
    static std::mutex mutex;
    static std::map<hipStream_t, Buffer> buffers;
    const std::lock_guard lock(mutex);
    auto& b = buffers[stream];
    if (b.bytes < bytes) {
        // Launches on this stream are ordered: the previous buffer is only freed after the stream drains it.
        if (b.memory) { check_hip(hipStreamSynchronize(stream), "drain folded HC workspace"); check_hip(hipFree(b.memory), "free folded HC workspace"); }
        check_hip(hipMalloc(&b.memory, bytes), "folded HC workspace allocation");
        b.bytes = bytes;
    }
    return static_cast<float*>(b.memory);
}
bool folded_decline(const char* why) {
    if (std::getenv("STRIX_WINDOW_FOLD_VERBOSE")) std::fprintf(stderr, "window HC fold declined: %s\n", why);
    return false;
}
} // namespace

bool hyper_mix_known_window_rows_launch_folded(const ElementwiseRowsViews* combine, const ElementwiseRowsViews& inject,
    std::span<const ExecutionDispatchContext> rows, const HipHyperMixLauncherConfig& config,
    const HipElementwiseLauncherConfig& elementwise) {
    const auto prepared = prepare_hyper_rows(rows, config);
    const auto count = prepared.count;
    const auto& r = prepared.rows;
    if (inject.count != count || (combine && combine->count != count)) return folded_decline("row counts differ");
    if (elementwise.activation != config.activation || elementwise.hidden_size != config.hidden_size ||
        elementwise.hc_count != config.hc_count || config.hidden_size != kInjectNormHidden || config.hc_count != kInjectNormStreams)
        return folded_decline("elementwise and HC configurations differ from the folded shape");
    const std::uint64_t value = config.activation == HipActivationStorage::bf16 ? 2U : 4U;
    const std::uint64_t hidden = config.hidden_size, width = hidden * config.hc_count;
    const auto at = [](const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer); };
    struct Extent { std::uintptr_t low, high; };
    const auto extent = [&](const void* pointer, std::uint64_t bytes) { return Extent{at(pointer), at(pointer) + bytes}; };
    const auto overlap = [](Extent a, Extent b) { return a.low < b.high && b.low < a.high; };
    InjectNormRows in{};
    for (std::uint32_t t = 0; t < count; ++t) {
        // The HC step reads exactly the streams the inject writes; the inject reads the combine's output.
        if (inject.output[t] != r.input[t]) return folded_decline("the HC step does not read the injected streams");
        if (combine && combine->output[t] != inject.second[t]) return folded_decline("the inject does not read the combined row");
        in.hyper[t] = inject.first[t];
        in.hidden[t] = combine ? combine->first[t] : inject.second[t];
        in.routed[t] = combine ? combine->second[t] : nullptr;
        in.gates[t] = inject.weights[t];
        in.combined[t] = combine ? combine->output[t] : nullptr;
    }
    // Blocks of every row run at once: no written view may reach a view another block still reads (a row's
    // own streams updated in place excepted). An aliased HC workspace (dead after the step) moves to an
    // owner-held buffer; any other alias declines the fold.
    bool private_workspace = false;
    for (std::uint32_t t = 0; t < count; ++t) {
        const Extent streams = extent(r.input[t], width * value), workspace = extent(r.normalized[t], (width + config.lowrank) * 4U);
        const Extent combined = combine ? extent(in.combined[t], hidden * value) : Extent{0U, 0U};
        for (std::uint32_t o = 0; o < count; ++o) {
            const Extent reads[] = {extent(in.hidden[o], hidden * value), in.routed[o] ? extent(in.routed[o], hidden * value) : Extent{0U, 0U},
                extent(in.gates[o], config.hc_count * 4ULL)};
            const Extent hyper = extent(in.hyper[o], width * value);
            for (std::size_t k = 0; k < 3U; ++k) {
                const auto& x = reads[k];
                // With the combine, a row's shared and routed inputs are read (by stream 0) before the block
                // barrier that precedes every stream store of that row.
                if (overlap(streams, x) && !(combine && o == t && k < 2U)) {
                    if (std::getenv("STRIX_WINDOW_FOLD_VERBOSE"))
                        std::fprintf(stderr, "window HC fold: streams of row %u [%#lx, %#lx) overlap input %zu of row %u [%#lx, %#lx)\n",
                            t, static_cast<unsigned long>(streams.low), static_cast<unsigned long>(streams.high), k, o,
                            static_cast<unsigned long>(x.low), static_cast<unsigned long>(x.high));
                    return folded_decline("streams alias a folded input");
                }
                // The combined row may replace its own row's shared or routed input exactly: only that row's
                // stream-0 threads read them, each element before it is written.
                if (overlap(combined, x) && !(o == t && k < 2U && x.low == combined.low && x.high == combined.high))
                    return folded_decline("the combined row aliases a folded input");
                private_workspace = private_workspace || overlap(workspace, x);
            }
            if (overlap(streams, hyper) && !(o == t && in.hyper[t] == r.input[t])) return folded_decline("streams alias another row's streams");
            if (overlap(combined, hyper)) return folded_decline("the combined row aliases the streams");
            private_workspace = private_workspace || overlap(workspace, hyper);
            if (o != t && combine && overlap(combined, extent(in.combined[o], hidden * value)))
                return folded_decline("combined rows overlap");
            // A row's combined row is stored before its streams, as the combine launch precedes the inject.
            if (combine && o != t && overlap(combined, extent(r.input[o], width * value)))
                return folded_decline("combined row aliases another row's streams");
        }
    }
    auto used = r;
    if (private_workspace) {
        auto* workspace = folded_hc_workspace(hip_execution_stream(), kWindowRows * (width + config.lowrank) * sizeof(float));
        for (std::uint32_t t = 0; t < count; ++t) {
            used.normalized[t] = workspace + t * (width + config.lowrank);
            used.low[t] = used.normalized[t] + width;
        }
    }
    const auto& p = prepared.first;
    const auto launch = [&]<typename Storage>() {
        if (combine)
            hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_inject_hyper_norm_kernel<Storage, true>), dim3(count), dim3(kThreads * kInjectNormStreams), 0,
                hip_execution_stream(), used, in, reinterpret_cast<const void*>(p.norm_runtime->address), p.norm_manifest->ggml_type,
                config.hidden_size, config.rms_epsilon);
        else
            hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_inject_hyper_norm_kernel<Storage, false>), dim3(kInjectNormStreams, count), dim3(kThreads), 0,
                hip_execution_stream(), used, in, reinterpret_cast<const void*>(p.norm_runtime->address), p.norm_manifest->ggml_type,
                config.hidden_size, config.rms_epsilon);
        check_hip(hipGetLastError(), "known-window folded inject + HC normalisation launch");
    };
    if (config.activation == HipActivationStorage::bf16) launch.template operator()<std::uint16_t>();
    else launch.template operator()<float>();
    launch_hyper_rows(prepared, used, config, 1U, true);
    folded_launches.fetch_add(1U, std::memory_order_relaxed);
    folded_workspace.store(private_workspace ? used.normalized[0] : nullptr, std::memory_order_relaxed);
    return true;
}
std::uint64_t hip_hyper_mix_testing_folded_launches() { return folded_launches.load(); }
const float* hip_hyper_mix_testing_folded_workspace() { return folded_workspace.load(); }

namespace {
// All T rows of a prefill step as T/8 tiles of one launch. Each tile is admitted like an
// 8-row window call; the tiles must be one uniform stride apart in every row view.
PreparedHyperRows prepare_hyper_prefill_rows(std::span<const ExecutionDispatchContext> rows,
    const HipHyperMixLauncherConfig& config, std::uint32_t& tiles) {
    if (rows.size() < 16U || rows.size() % kWindowRows) throw std::invalid_argument("HC prefill rows require 8-row tiles");
    tiles = static_cast<std::uint32_t>(rows.size() / kWindowRows);
    if (tiles > 65535U) throw std::invalid_argument("HC prefill rows exceed the grid");
    auto first = prepare_hyper_rows(rows.first(kWindowRows), config);
    const auto at = [](const auto* pointer) { return reinterpret_cast<std::uintptr_t>(pointer); };
    const auto second = prepare_hyper_rows(rows.subspan(kWindowRows, kWindowRows), config);
    const auto stride = [&](const auto* a, const auto* b) -> std::uint64_t {
        if (!a != !b) throw std::invalid_argument("HC prefill tiles bind different optional views");
        if (!a) return 0U;
        if (at(b) <= at(a) || (at(b) - at(a)) % 16U) throw std::invalid_argument("HC prefill tiles are not strided");
        return at(b) - at(a);
    };
    first.rows.tile_input = stride(first.rows.input[0], second.rows.input[0]);
    first.rows.tile_output = stride(first.rows.output[0], second.rows.output[0]);
    first.rows.tile_normalized = stride(first.rows.normalized[0], second.rows.normalized[0]);
    first.rows.tile_injection = stride(first.rows.injection[0], second.rows.injection[0]);
    for (std::uint32_t t = 1; t < tiles; ++t) {
        const auto tile = prepare_hyper_rows(rows.subspan(t * kWindowRows, kWindowRows), config);
        const auto same = [&](const auto* a, const auto* b, std::uint64_t step) {
            return a ? at(b) == at(a) + static_cast<std::uint64_t>(t) * step : b == nullptr;
        };
        for (std::uint32_t r = 0; r < kWindowRows; ++r)
            if (!same(first.rows.input[r], tile.rows.input[r], first.rows.tile_input) ||
                !same(first.rows.output[r], tile.rows.output[r], first.rows.tile_output) ||
                !same(first.rows.normalized[r], tile.rows.normalized[r], first.rows.tile_normalized) ||
                !same(first.rows.low[r], tile.rows.low[r], first.rows.tile_normalized) ||
                !same(first.rows.injection[r], tile.rows.injection[r], first.rows.tile_injection))
                throw std::invalid_argument("HC prefill tile views are not strided");
    }
    // Tiles run concurrently: no written view of any row may reach any row's input.
    const auto width = static_cast<std::uint64_t>(config.hidden_size) * config.hc_count;
    const auto input_bytes = width * (config.activation == HipActivationStorage::bf16 ? 2U : 4U);
    std::uintptr_t in_low = UINTPTR_MAX, in_high = 0;
    for (std::uint32_t r = 0; r < kWindowRows; ++r) {
        in_low = std::min(in_low, at(first.rows.input[r]));
        in_high = std::max(in_high, at(first.rows.input[r]) + (tiles - 1U) * first.rows.tile_input + input_bytes);
    }
    const auto written = [&](const auto* const* views, std::uint64_t stride, std::uint64_t bytes) {
        for (std::uint32_t r = 0; r < kWindowRows; ++r) {
            if (!views[r]) continue;
            const auto low = at(views[r]), high = low + (tiles - 1U) * stride + bytes;
            if (low < in_high && in_low < high) throw std::invalid_argument("HC prefill outputs alias the input rows");
        }
    };
    written(first.rows.output, first.rows.tile_output, input_bytes / config.hc_count);
    written(first.rows.normalized, first.rows.tile_normalized, (width + config.lowrank) * sizeof(float));
    written(first.rows.injection, first.rows.tile_injection, config.hc_count * sizeof(float));
    return first;
}

// The wide kernels run when every view is one uniform row stride apart (row r of tile z at
// view[0] + (8 z + r) * stride) and the shapes are the split-K / rank-320 lane-chain ones.
// STRIX_HC_WIDE=0 keeps the 8-row tile kernels.
bool wide_hyper_rows(const PreparedHyperRows& prepared, const HipHyperMixLauncherConfig& config, std::uint32_t tiles,
    HyperWideRows& wide) {
    static const bool enabled = [] { const char* v = std::getenv("STRIX_HC_WIDE"); return !v || v[0] != '0'; }();
    static const bool fused_norm = std::getenv("STRIX_HC_ROWS_FUSED_NORM") != nullptr;
    if (!enabled || fused_norm || !config.splitk_projection || !config.lane_chain_output ||
        config.hidden_size * config.hc_count != 10240U || config.lowrank != 320U ||
        config.hc_count != kWideStreams || config.hidden_size % (32U * kWideOutputWaves) ||
        config.hc_count > kWideProjectionOutputs || prepared.count != kWindowRows)
        return false;
    const auto& rows = prepared.rows;
    const auto at = [](const auto* pointer) { return reinterpret_cast<std::uintptr_t>(pointer); };
    const auto uniform = [&](const auto* const* views, std::uint64_t tile, std::uint64_t& stride) {
        if (!views[0]) { stride = 0; return true; }
        if (tile % kWindowRows) return false;
        stride = tile / kWindowRows;
        for (std::uint32_t r = 0; r < kWindowRows; ++r)
            if (!views[r] || at(views[r]) != at(views[0]) + r * stride) return false;
        return true;
    };
    std::uint64_t low_stride = 0;
    if (!uniform(rows.normalized, rows.tile_normalized, wide.normalized_stride) ||
        !uniform(rows.low, rows.tile_normalized, low_stride) || low_stride != wide.normalized_stride ||
        !uniform(rows.injection, rows.tile_injection, wide.injection_stride) ||
        !uniform(rows.output, rows.tile_output, wide.output_stride) ||
        wide.normalized_stride % 16U || at(rows.normalized[0]) % 16U || at(rows.low[0]) != at(rows.normalized[0]) + 10240U * sizeof(float) ||
        (prepared.first.inject_runtime && !rows.injection[0]) || prepared.first.up_runtime->address % 4U)
        return false;
    wide.normalized = rows.normalized[0];
    wide.injection = rows.injection[0];
    wide.output = rows.output[0];
    wide.count = tiles * kWindowRows;
    return true;
}

#include "hip_hc_fast_wmma.inc"
#include "hip_qpf2k_hc.inc"
#include "hip_qpf2k_hc_out16.inc"

constexpr unsigned kSkipKernel = 255U;  // test seam: launch only the other kernel
// Transposed low values for the wide2 output kernel: 320 x up to kWide2LowRows rows (prefill steps).
constexpr std::uint32_t kWide2LowRows = known_prefill_max_rows;
float* wide2_low_scratch(std::uint32_t rows = 8192U, bool setup = false) {
    static HipPrefillPool pool;
    const auto capacity = hc_prefill_pool_owner().rows((rows + kWide2OutputRows - 1U) / kWide2OutputRows * kWide2OutputRows);
    return static_cast<float*>(pool.get(std::uint64_t{320U} * capacity * sizeof(float),
        "HC wide rows: transposed low scratch",setup));
}
// Projection and output kernels of the wide prefill rows. projection_kernel 0: the 4-output lane-tree
// kernel (STRIX_HC_WIDE_PROJECTION_ROWS 8|16 / _WAVES 4|8 block shape, default by row count), 1: the
// register-permuted 8-output kernel over the packed down weights. output_kernel 0: the lane-chain kernel
// (STRIX_HC_WIDE_OUTPUT_ROWS 16|32, default 32 from 128 rows on), 1: the scalar-low kernel (it needs the
// transposed low values of projection kernel 1; with another projection kernel 0 runs).
void launch_wide_hyper_kernels(const HyperWideRows& wide_in, const void* down, const void* inject, const void* up,
    const HipHyperMixLauncherConfig& config, unsigned projection_kernel, unsigned output_kernel, hipStream_t stream,
    unsigned output3_band = UINT32_MAX, bool output3_zero_low_rank_stride = false, unsigned low_band = UINT32_MAX) {
    // The wide2 output kernel reads the transposed low values the wide2 projection writes (rows padded
    // to whole 32-row groups).
    HyperWideRows wide = wide_in;
    // Test seam: output kernels 3..6 = the STRIX_QPF2K_HC output (packed coefficients) on the lane-chain output's inputs:
    // 3 v1 (ring 4, conditional look-ahead), 4 v2 (blocks of 8 steps; the engine's form), 5 blocks of 4, 6 blocks of 10.
    // Projection kernels 3 / 4 = the STRIX_QPF2K_HC projection, register prefetch depth 2 (v1) / 1 (v2, the engine's form).
    // Form 14 is the output-only / paired fixture seam; the env selects only the
    // normal fused FP32-normalized route and leaves fast_output_kernel() at 1.
    const bool out16 = output_kernel == 14U || (output_kernel == 1U && qpf2k_hc_out16() &&
        config.fast_prefill_wmma && config.hidden_size == 2560U && config.hc_count == 4U &&
        projection_kernel == 2U && !wide.normalized16 && !wide.inverse);
    if (output_kernel == 14U) output_kernel = 1U;
    if (out16 && (config.hidden_size != 2560U || config.hc_count != 4U || wide.normalized16 || wide.inverse ||
        (projection_kernel != 2U && projection_kernel != 4U && projection_kernel != kSkipKernel)))
        throw std::invalid_argument("HC out16 requires FP32 normalized rows and the WMMA low image");
    const unsigned qpf2k_output_form = output_kernel >= 3U && output_kernel <= 6U ? output_kernel : 0U;
    const bool qpf2k_output = qpf2k_output_form != 0U;
    if (qpf2k_output) output_kernel = 1U;
    // Output kernels 7..11 = the STRIX_QPF2K_HC output v3 (STRIX_QPF2K_HC_OUTPUT3): 7 the engine's form (16-step loop, low
    // values prefetched one 32-step block ahead, epilogue rows loaded 8 ahead), 8 no low prefetch, 9 base epilogue, 10 low
    // prefetch two blocks ahead, 11 32-step loop.
    // 12 / 13: form 7 + bulk scalar-cache prefetch of the next 8 / 16 steps' low values (wave 0 of each block).
    const unsigned qpf2k_output3_form = output_kernel >= 7U && output_kernel <= 13U ? output_kernel : 0U;
    const bool qpf2k_output3 = qpf2k_output3_form != 0U;
    if (qpf2k_output3) output_kernel = 1U;
    const unsigned qpf2k_projection_form = projection_kernel == 3U || projection_kernel == 4U ? projection_kernel : 0U;
    const bool qpf2k_projection = qpf2k_projection_form != 0U;
    if (qpf2k_projection) projection_kernel = 2U;
    if (!out16 && output_kernel == 1U && (projection_kernel == 1U || projection_kernel == 2U || projection_kernel == kSkipKernel) &&
        (wide.count + kWide2OutputRows - 1U) / kWide2OutputRows * kWide2OutputRows <= kWide2LowRows) {
        wide.low_t = wide2_low_scratch(wide.count);
        wide.low_t_pitch = (wide.count + kWide2OutputRows - 1U) / kWide2OutputRows * kWide2OutputRows;
    } else if (!out16 && output_kernel == 1U) {
        if (qpf2k_output || qpf2k_output3) throw std::invalid_argument("qpf2k HC output test needs the transposed low values");
        output_kernel = 0U;
    }
    if (wide.inverse && !wide.low_t) throw std::invalid_argument("HC norm-free rows require the lane-chain output");
    // STRIX_HC_WIDE2_OUTPUT_ROWS / _ORDER of the lane-chain output below (read here: the low layout depends on its route).
    static const unsigned output_rows16 = [] { const char* v = std::getenv("STRIX_HC_WIDE2_OUTPUT_ROWS"); return v && std::atoi(v) == 16; }();
    static const bool output_rows_first = [] { const char* v = std::getenv("STRIX_HC_WIDE2_OUTPUT_ORDER"); return !v || v[0] != 'f'; }();
    const unsigned output3_band_used = qpf2k_output3_form == 0U || qpf2k_output3_form == 7U ?
        (output3_band == UINT32_MAX ? qpf2k_hc_output3_band() : output3_band) : 0U;
    // STRIX_QPF2K_HC_LOWBAND (test seam: low_band): only on the engine's qpf2k projection -> output v3 band 16 route.
    unsigned low_mode = 0U;
    {
        const bool output3_route = output_kernel == 1U && wide.low_t && !output_rows16 && !wide.normalized16 && !wide.inverse &&
            config.hidden_size == 2560U && (qpf2k_output3 || (!qpf2k_output && output_rows_first && config.fast_prefill_wmma && qpf2k_hc_output3()));
        const bool qpf2k_projection_route = projection_kernel == 2U && !wide.normalized16 && !wide.inverse && !qpf2k_hc_check() &&
            (qpf2k_projection ? qpf2k_projection_form == 4U : (qpf2k_hc_mask() & 8U) != 0U);
        if (!out16 && output3_route && qpf2k_projection_route && output3_band_used == 16U && !output3_zero_low_rank_stride)
            low_mode = (low_band == UINT32_MAX ? qpf2k_hc_lowband() : low_band) & 3U;
        if (wide.low_t_pitch <= kQpf2kLowBandRows) low_mode &= ~1U;
        if (low_mode & 1U) { wide.low_t_band = true; wide.low_t_pitch = kQpf2kLowBandRows; }
        // Output-only test seams (projection skipped) read the low values an earlier call left in the shared scratch: refused when
        // that call wrote another layout (bit 0: low_t banded) or skipped low16 (bit 1). Every projection records what it wrote.
        static std::atomic<unsigned> low_state{0U};
        if (projection_kernel == kSkipKernel) {
            if (output_kernel != kSkipKernel && (((low_state.load() & 1U) && output_kernel == 1U && !out16 && wide.low_t) ||
                                                 ((low_state.load() & 2U) && (output_kernel == 2U || out16))))
                throw std::invalid_argument("HC wide rows test: the last projection left the low values in another layout");
        } else if (projection_kernel == 2U) {
            low_state.store(((wide.low_t ? low_mode : low_state.load()) & 1U) | (low_mode & 2U));
        } else if (wide.low_t) {
            low_state.fetch_and(~1U);
        }
        // Path witness (once per process and mode set): the engine receipt shows the candidate route ran.
        static std::atomic<unsigned> witnessed{0U};
        if (low_mode && !(witnessed.fetch_or(1U << low_mode) & (1U << low_mode)))
            std::fprintf(stderr, "qpf2k HC lowband mode %u active (rows %u, low pitch %u)\n", low_mode, wide.count, wide.low_t_pitch);
    }
    static const unsigned projection_rows = [] {
        const char* v = std::getenv("STRIX_HC_WIDE_PROJECTION_ROWS"); return v ? static_cast<unsigned>(std::atoi(v)) : 0U; }();
    static const unsigned projection_waves = [] {
        const char* v = std::getenv("STRIX_HC_WIDE_PROJECTION_WAVES"); return v ? static_cast<unsigned>(std::atoi(v)) : 0U; }();
    static const unsigned output_rows = [] {
        const char* v = std::getenv("STRIX_HC_WIDE_OUTPUT_ROWS"); return v ? static_cast<unsigned>(std::atoi(v)) : 0U; }();
    const bool large = wide.count >= 128U;
    // Kernels 2 (fast-prefill-f16wmma-v1): FP16 WMMA projection and output over per-call FP16 weight packs;
    // output kernel 2 reads the FP16 low values projection kernel 2 writes; output kernel 1 (the FP32 lane-chain
    // output) reads the transposed FP32 low values it also writes.
    if (projection_kernel == 2U || output_kernel == 2U || out16) {
        if (wide.count > kHcwLowRows || config.hc_count != 4U)
            throw std::invalid_argument("HC WMMA rows: row count or stream count unsupported");
        const auto scratch = hcw_scratch(wide.count);
        if (projection_kernel == 2U) {
            hipLaunchKernelGGL(hcw_pack_down_kernel, dim3((kHcwRank * kHcwStages + 255U) / 256U), dim3(256U), 0, stream,
                static_cast<const unsigned char*>(down), scratch.down, scratch.overflow);
            const dim3 grid(kHcwProjColBlocks, (wide.count + kHcwProjRows - 1U) / kHcwProjRows);
            if (wide.normalized16)
                hipLaunchKernelGGL(HIP_KERNEL_NAME(hcw_projection_kernel<true>), grid, dim3(256U), 0, stream, wide, scratch.down,
                    static_cast<const float*>(inject), config.hc_count, scratch.low16);
            else if (wide.inverse)
                hipLaunchKernelGGL(HIP_KERNEL_NAME(hcw_projection_kernel<false, true>), grid, dim3(256U), 0, stream, wide, scratch.down,
                    static_cast<const float*>(inject), config.hc_count, scratch.low16);
            else if ((qpf2k_hc_mask() & 8U) || qpf2k_projection) {
                // STRIX_QPF2K_HC bit 3 (test seam: projection kernels 3 / 4): the prefetching form, same bytes.
                const auto qpf2k = [&] {
                    if (qpf2k_projection_form == 3U)
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_hc_projection_kernel<2U>), grid, dim3(256U), 0, stream, wide, scratch.down,
                            static_cast<const float*>(inject), config.hc_count, scratch.low16);
                    else {
                        const auto launch = [&]<unsigned LowMode>() {
                            hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_hc_projection_kernel<1U, LowMode>), grid, dim3(256U), 0, stream, wide,
                                scratch.down, static_cast<const float*>(inject), config.hc_count, scratch.low16);
                        };
                        switch (low_mode) {
                        case 1U: launch.template operator()<1U>(); break;
                        case 2U: launch.template operator()<2U>(); break;
                        case 3U: launch.template operator()<3U>(); break;
                        default: launch.template operator()<0U>();
                        }
                    }
                };
                if (!qpf2k_hc_check() || qpf2k_projection) qpf2k();
                else {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(hcw_projection_kernel<false>), grid, dim3(256U), 0, stream, wide, scratch.down,
                        static_cast<const float*>(inject), config.hc_count, scratch.low16);
                    const auto snap = [&] {
                        std::vector<std::vector<unsigned char>> v;
                        v.push_back(qpf2k_snapshot_rows(wide.normalized + 10240U, wide.normalized_stride, 320U * 4U, wide.count, stream));
                        if (wide.injection) v.push_back(qpf2k_snapshot_rows(wide.injection, wide.injection_stride, config.hc_count * 4U, wide.count, stream));
                        if (wide.low_t) v.push_back(qpf2k_snapshot_rows(wide.low_t, std::uint64_t{wide.low_t_pitch} * 4U, wide.count * 4U, 320U, stream));
                        v.push_back(qpf2k_snapshot_rows(scratch.low16, 320U * 2U, 320U * 2U, wide.count, stream));
                        return v;
                    };
                    const auto reference = snap();
                    // Every destination poisoned, so a missing store cannot inherit the reference's bytes.
                    check_hip(hipMemset2DAsync(wide.normalized + 10240U, wide.normalized_stride, 0x5a, 320U * 4U, wide.count, stream), "qpf2k HC check poison");
                    if (wide.injection)
                        check_hip(hipMemset2DAsync(wide.injection, wide.injection_stride, 0x5a, config.hc_count * 4U, wide.count, stream), "qpf2k HC check poison");
                    if (wide.low_t)
                        check_hip(hipMemset2DAsync(wide.low_t, std::uint64_t{wide.low_t_pitch} * 4U, 0x5a, wide.count * 4U, 320U, stream), "qpf2k HC check poison");
                    check_hip(hipMemsetAsync(scratch.low16, 0x5a, std::uint64_t{wide.count} * 320U * 2U, stream), "qpf2k HC check poison");
                    qpf2k();
                    const auto candidate = snap();
                    const char* names[] = {"projection low", "projection injection", "projection low_t", "projection low16"};
                    std::size_t n = 0;
                    qpf2k_compare(names[0], reference[n], candidate[n], wide.count); ++n;
                    if (wide.injection) { qpf2k_compare(names[1], reference[n], candidate[n], wide.count); ++n; }
                    if (wide.low_t) { qpf2k_compare(names[2], reference[n], candidate[n], wide.count); ++n; }
                    qpf2k_compare(names[3], reference[n], candidate[n], wide.count);
                    std::fprintf(stderr, "qpf2k HC check call projection: rows %u all destinations equal\n", wide.count);
                }
            } else hipLaunchKernelGGL(HIP_KERNEL_NAME(hcw_projection_kernel<false>), grid, dim3(256U), 0, stream, wide, scratch.down,
                    static_cast<const float*>(inject), config.hc_count, scratch.low16);
            check_hip(hipGetLastError(), "prefill HC WMMA projection launch");
        } else if (projection_kernel != kSkipKernel) throw std::invalid_argument("HC WMMA output requires the WMMA projection");
        if (output_kernel == kSkipKernel) return;
        if (out16) {
            static std::atomic<bool> witnessed{false};
            if (qpf2k_hc_out16() && !witnessed.exchange(true))
                std::fprintf(stderr, "qpf2k HC out16 active (rows %u, FP16 low image retained)\n", wide.count);
            // As for output3: cached byte-identical Q8 image, including copy cost.
            const void* up16 = qpf2k_hc_upcopy() ? qpf2k_hc_up_copy(up,
                std::uint64_t{4U} * config.hidden_size * 340U, wide.count, stream) : up;
            const dim3 grid((wide.count + 31U) / 32U, config.hidden_size / 128U);
            if (config.activation == HipActivationStorage::bf16)
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_hc_out16_kernel<std::uint16_t>), grid, dim3(128U), 0, stream,
                    wide, static_cast<const unsigned char*>(up16), scratch.low16, config.hidden_size);
            else hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_hc_out16_kernel<float>), grid, dim3(128U), 0, stream,
                    wide, static_cast<const unsigned char*>(up16), scratch.low16, config.hidden_size);
            check_hip(hipGetLastError(), "qpf2k HC out16 launch");
            return;
        }
        if (output_kernel == 1U && wide.low_t) {
            projection_kernel = kSkipKernel;  // fall through to the lane-chain output below
        } else {
        if (wide.inverse) throw std::invalid_argument("HC norm-free rows require the lane-chain output");
        if (output_kernel != 2U) throw std::invalid_argument("HC WMMA projection requires the WMMA or lane-chain output");
        hipLaunchKernelGGL(hcw_pack_up_kernel, dim3((4U * config.hidden_size * (kHcwRank / 32U) + 255U) / 256U), dim3(256U), 0, stream,
            static_cast<const unsigned char*>(up), scratch.up, config.hidden_size, scratch.overflow);
        const dim3 grid((wide.count + kHcwOutRows - 1U) / kHcwOutRows, config.hidden_size / kHcwOutFeatures);
        if (config.activation == HipActivationStorage::bf16)
            hipLaunchKernelGGL(HIP_KERNEL_NAME(hcw_output_kernel<std::uint16_t>), grid, dim3(256U), 0, stream, wide, scratch.up,
                scratch.low16, config.hidden_size);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(hcw_output_kernel<float>), grid, dim3(256U), 0, stream, wide, scratch.up, scratch.low16,
                config.hidden_size);
        check_hip(hipGetLastError(), "prefill HC WMMA output launch");
        return;
        }
    }
    if (projection_kernel == kSkipKernel) {
    } else if (projection_kernel == 1U) {
        // STRIX_HC_WIDE2_WAVES (8 default, 4): waves per block. STRIX_HC_WIDE2_VOPD=0: the C++ FMA loop
        // instead of the hand-paired VOPD chunk.
        static const unsigned waves = [] { const char* v = std::getenv("STRIX_HC_WIDE2_WAVES"); return v && std::atoi(v) == 4 ? 4U : 8U; }();
        static const bool vopd = [] { const char* v = std::getenv("STRIX_HC_WIDE2_VOPD"); return !v || v[0] != '0'; }();
        auto* scratch = wide2_pack_scratch();
        const auto* pack_codes = reinterpret_cast<const uint4*>(scratch);
        const auto* pack_scales = reinterpret_cast<const float*>(scratch + kWide2PackCodeBytes);
        hipLaunchKernelGGL(wide2_pack_down_kernel, dim3(kWide2Chunks * kWide2Groups), dim3(32U), 0, stream,
            static_cast<const unsigned char*>(down), reinterpret_cast<uint4*>(scratch), reinterpret_cast<float*>(scratch + kWide2PackCodeBytes));
        // STRIX_HC_WIDE2_PROJECTION_ORDER=rows: row blocks fastest in the grid (output blocks fastest by default).
        static const bool rows_first = [] { const char* v = std::getenv("STRIX_HC_WIDE2_PROJECTION_ORDER"); return v && v[0] == 'r'; }();
        const auto run = [&]<unsigned Waves, bool Vopd>() {
            constexpr unsigned per_block = Waves * kWide2Outputs;
            const unsigned low_blocks = (config.lowrank + per_block - 1U) / per_block;
            const unsigned outs = low_blocks + (inject ? 1U : 0U), row_blocks = (wide.count + kWide2Rows - 1U) / kWide2Rows;
            if (rows_first)
                hipLaunchKernelGGL(HIP_KERNEL_NAME(wide2_splitk_hyper_projection_kernel<Waves, Vopd, true>), dim3(row_blocks, outs),
                    dim3(32U * Waves), 0, stream, wide, pack_codes, pack_scales, config.hc_count, config.lowrank, inject, low_blocks);
            else hipLaunchKernelGGL(HIP_KERNEL_NAME(wide2_splitk_hyper_projection_kernel<Waves, Vopd>), dim3(outs, row_blocks),
                    dim3(32U * Waves), 0, stream, wide, pack_codes, pack_scales, config.hc_count, config.lowrank, inject, low_blocks);
        };
        if (!vopd) run.template operator()<8U, false>();
        else if (waves == 4U) run.template operator()<4U, true>();
        else run.template operator()<8U, true>();
    } else {
        const auto projection = [&]<unsigned Rows, unsigned Waves>() {
            const unsigned per_block = Waves * kWideProjectionOutputs;
            const unsigned low_blocks = (config.lowrank + per_block - 1U) / per_block;
            hipLaunchKernelGGL(HIP_KERNEL_NAME(wide_splitk_hyper_projection_kernel<Rows, Waves>),
                dim3(low_blocks + (inject ? 1U : 0U), (wide.count + Rows - 1U) / Rows), dim3(32U * Waves), 0, stream, wide, down,
                config.hc_count, config.lowrank, inject, low_blocks);
        };
        // Defaults (measured): 16 rows x 8 waves from 128 rows on, 8 x 4 below (more blocks in flight).
        const unsigned shape_rows = projection_rows == 8U || projection_rows == 16U ? projection_rows : large ? 16U : 8U;
        const unsigned shape_waves = projection_waves == 4U || projection_waves == 8U ? projection_waves : large ? 8U : 4U;
        if (shape_rows == 8U) {
            if (shape_waves == 4U) projection.template operator()<8U, 4U>(); else projection.template operator()<8U, 8U>();
        } else {
            if (shape_waves == 4U) projection.template operator()<16U, 4U>(); else projection.template operator()<16U, 8U>();
        }
    }
    check_hip(hipGetLastError(), "prefill HC wide rows projection launch");
    if (output_kernel == kSkipKernel) return;
    const auto invoke = [&]<typename Storage>() {
        if (output_kernel == 1U) {
            // STRIX_HC_WIDE2_OUTPUT_ROWS = 32 (default) | 16: rows per wave.
            // Row groups fastest in the grid: the 32-row groups of one feature block run together, so its up
            // weights (174 KB) stay in L2 (in prefill 1898 -> 1556 us per call at 4K). STRIX_HC_WIDE2_OUTPUT_ORDER=features:
            // feature blocks fastest.
            const auto run = [&]<unsigned R>() {
                const unsigned features = config.hidden_size / (32U * kWide2OutputWaves), groups = (wide.count + R - 1U) / R;
                if (wide.normalized16_only) {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(wide2_hyper_output_kernel<Storage, R, true, true>), dim3(groups, features),
                        dim3(32U * kWide2OutputWaves), 0, stream, wide, up, config.hidden_size);
                    return;
                }
                if (wide.inverse) {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(wide2_hyper_output_kernel<Storage, R, true, false, true>), dim3(groups, features),
                        dim3(32U * kWide2OutputWaves), 0, stream, wide, up, config.hidden_size);
                    return;
                }
                if constexpr (R == 32U) {
                    // STRIX_QPF2K_HC_OUTPUT3 (fast profile): output v3, the same arithmetic as the kernel below; same bytes.
                    if (qpf2k_output3 || (!qpf2k_output && output_rows_first && config.fast_prefill_wmma && qpf2k_hc_output3() && config.hidden_size == 2560U)) {
                        const dim3 grid(groups, features);
                        // Engine and fast-profile fixtures include the cached per-call copy.
                        // Ordinary seam fixtures keep their supplied buffer.
                        const void* up3 = config.fast_prefill_wmma && qpf2k_hc_upcopy() ?
                            qpf2k_hc_up_copy(up, std::uint64_t{kWideStreams} * config.hidden_size * 340U, wide.count, stream) : up;
                        const unsigned band = output3_band_used;
                        if (wide.low_t_band && band != 16U) throw std::logic_error("qpf2k HC banded low values without output v3 band 16");
                        const auto qpf2k3 = [&] {
                            if (output3_zero_low_rank_stride) wide.low_t_pitch = 0U;  // test seam diagnostic only; outputs discarded
                            if (band) {
                                const auto launch = [&]<unsigned Band>() {
                                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_hc_output3_band_kernel<Storage, Band>), grid, dim3(128U), 0, stream, wide, up3, config.hidden_size);
                                };
                                // Low-table reuse by CTA permutation only; rank chains, operand forms and epilogue unchanged; unmeasured.
                                switch (band) {
                                case 1U: launch.template operator()<1U>(); break;
                                case 2U: launch.template operator()<2U>(); break;
                                case 3U: launch.template operator()<3U>(); break;
                                case 4U: launch.template operator()<4U>(); break;
                                case 5U: launch.template operator()<5U>(); break;
                                case 6U: launch.template operator()<6U>(); break;
                                case 7U: launch.template operator()<7U>(); break;
                                case 8U: launch.template operator()<8U>(); break;
                                case 12U: launch.template operator()<12U>(); break;
                                case 16U:
                                    if (wide.low_t_band)
                                        hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_hc_output3_band_kernel<Storage, 16U, true>), grid, dim3(128U), 0,
                                            stream, wide, up3, config.hidden_size);
                                    else launch.template operator()<16U>();
                                    break;
                                case 24U: launch.template operator()<24U>(); break;
                                case 32U: launch.template operator()<32U>(); break;
                                case 48U: launch.template operator()<48U>(); break;
                                case 64U: launch.template operator()<64U>(); break;
                                }
                                return;
                            }
                            switch (qpf2k_output3_form) {
                            case 8U: hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_hc_output3_kernel<Storage, 16U, 0U, 8U>), grid, dim3(128U), 0, stream, wide, up3, config.hidden_size); break;
                            case 9U: hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_hc_output3_kernel<Storage, 16U, 1U, 0U>), grid, dim3(128U), 0, stream, wide, up3, config.hidden_size); break;
                            case 10U: hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_hc_output3_kernel<Storage, 16U, 2U, 8U>), grid, dim3(128U), 0, stream, wide, up3, config.hidden_size); break;
                            case 11U: hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_hc_output3_kernel<Storage, 32U, 1U, 8U>), grid, dim3(128U), 0, stream, wide, up3, config.hidden_size); break;
                            case 12U: hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_hc_output3_kernel<Storage, 16U, 1U, 8U, 8U>), grid, dim3(128U), 0, stream, wide, up3, config.hidden_size); break;
                            case 13U: hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_hc_output3_kernel<Storage, 16U, 1U, 8U, 16U>), grid, dim3(128U), 0, stream, wide, up3, config.hidden_size); break;
                            default: hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_hc_output3_kernel<Storage, 16U, 1U, 8U>), grid, dim3(128U), 0, stream, wide, up3, config.hidden_size);
                            }
                        };
                        if (!qpf2k_hc_check() || qpf2k_output3) { qpf2k3(); return; }
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(wide2_hyper_output_kernel<Storage, R, true>), grid, dim3(32U * kWide2OutputWaves), 0, stream,
                            wide, up, config.hidden_size);
                        const auto reference = qpf2k_snapshot_rows(wide.output, wide.output_stride, config.hidden_size * sizeof(Storage), wide.count, stream);
                        check_hip(hipMemset2DAsync(wide.output, wide.output_stride, 0x5a, config.hidden_size * sizeof(Storage), wide.count, stream),
                            "qpf2k HC check output poison");
                        qpf2k3();
                        const auto candidate = qpf2k_snapshot_rows(wide.output, wide.output_stride, config.hidden_size * sizeof(Storage), wide.count, stream);
                        qpf2k_compare("output v3", reference, candidate, wide.count);
                        std::fprintf(stderr, "qpf2k HC check call output: rows %u equal (v3)\n", wide.count);
                        return;
                    }
                    // STRIX_QPF2K_HC bit 0 (fast profile): packed coefficients, double-buffered low values; same bytes.
                    if (qpf2k_output || (output_rows_first && config.fast_prefill_wmma && (qpf2k_hc_mask() & 1U) && config.hidden_size == 2560U)) {
                        const auto qpf2k = [&] {
                            auto* coef = qpf2k_hc_coef_scratch(stream);
                            hipLaunchKernelGGL(qpf2k_hc_pack_coef_kernel, dim3((config.hidden_size * 10U + 255U) / 256U), dim3(256U), 0, stream,
                                static_cast<const unsigned char*>(up), coef, config.hidden_size);
                            const dim3 grid(groups, features);
                            switch (qpf2k_output_form) {
                            case 3U: hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_hc_output_kernel<Storage, 4U, false>), grid, dim3(128U), 0, stream, wide, coef, config.hidden_size); break;
                            case 5U: hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_hc_output_kernel<Storage, 4U, true>), grid, dim3(128U), 0, stream, wide, coef, config.hidden_size); break;
                            case 6U: hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_hc_output_kernel<Storage, 10U, true>), grid, dim3(128U), 0, stream, wide, coef, config.hidden_size); break;
                            default: hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_hc_output_kernel<Storage>), grid, dim3(128U), 0, stream, wide, coef, config.hidden_size);
                            }
                        };
                        if (!qpf2k_hc_check() || qpf2k_output) { qpf2k(); return; }
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(wide2_hyper_output_kernel<Storage, R, true>), dim3(groups, features),
                            dim3(32U * kWide2OutputWaves), 0, stream, wide, up, config.hidden_size);
                        const auto reference = qpf2k_snapshot_rows(wide.output, wide.output_stride, config.hidden_size * sizeof(Storage), wide.count, stream);
                        check_hip(hipMemset2DAsync(wide.output, wide.output_stride, 0x5a, config.hidden_size * sizeof(Storage), wide.count, stream),
                            "qpf2k HC check output poison");
                        qpf2k();
                        const auto candidate = qpf2k_snapshot_rows(wide.output, wide.output_stride, config.hidden_size * sizeof(Storage), wide.count, stream);
                        qpf2k_compare("output", reference, candidate, wide.count);
                        std::fprintf(stderr, "qpf2k HC check call output: rows %u equal\n", wide.count);
                        return;
                    }
                }
                if (output_rows_first) {
                    // STRIX_QPF2K_HC_UPCOPY_BASE (attribution): the base kernel on the cached up weight copy.
                    const void* upb = config.fast_prefill_wmma && qpf2k_hc_upcopy_base() ?
                        qpf2k_hc_up_copy(up, std::uint64_t{kWideStreams} * config.hidden_size * 340U, wide.count, stream) : up;
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(wide2_hyper_output_kernel<Storage, R, true>), dim3(groups, features),
                        dim3(32U * kWide2OutputWaves), 0, stream, wide, upb, config.hidden_size);
                }
                else hipLaunchKernelGGL(HIP_KERNEL_NAME(wide2_hyper_output_kernel<Storage, R, false>), dim3(features, groups),
                        dim3(32U * kWide2OutputWaves), 0, stream, wide, up, config.hidden_size);
            };
            if (output_rows16) run.template operator()<16U>(); else run.template operator()<32U>();
            return;
        }
        const auto output = [&]<unsigned Rows>() {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(wide_lane_chain_hyper_output_kernel<Storage, Rows>),
                dim3(config.hidden_size / (32U * kWideOutputWaves), (wide.count + Rows - 1U) / Rows), dim3(32U * kWideOutputWaves), 0,
                stream, wide, up, config.hidden_size);
        };
        const unsigned rows_per_block = output_rows == 16U || output_rows == 32U ? output_rows : large ? 32U : 16U;
        if (rows_per_block == 16U) output.template operator()<16U>(); else output.template operator()<32U>();
    };
    switch (config.activation) {
    case HipActivationStorage::bf16: invoke.template operator()<std::uint16_t>(); break;
    case HipActivationStorage::fp32: invoke.template operator()<float>(); break;
    }
    check_hip(hipGetLastError(), "prefill HC wide rows output launch");
}
// STRIX_HC_WIDE_PROJECTION_KERNEL / STRIX_HC_WIDE_OUTPUT_KERNEL (default 1 each): the kernels above.
unsigned wide_projection_kernel() {
    static const unsigned k = [] { const char* v = std::getenv("STRIX_HC_WIDE_PROJECTION_KERNEL"); return v ? static_cast<unsigned>(std::atoi(v)) : 1U; }();
    return k;
}
// Fast prefill profile HC output: 1 (default, the FP32 lane-chain output on the WMMA projection's FP32 low
// values) or STRIX_HCW_OUTPUT=2 (the FP16 WMMA output).
unsigned fast_output_kernel() {
    static const unsigned k = [] { const char* v = std::getenv("STRIX_HCW_OUTPUT"); return v && std::atoi(v) == 2 ? 2U : 1U; }();
    return strix::governed_hc_output_kernel(k, strix::prefill_options_active());
}
unsigned wide_output_kernel() {
    static const unsigned k = [] { const char* v = std::getenv("STRIX_HC_WIDE_OUTPUT_KERNEL"); return v ? static_cast<unsigned>(std::atoi(v)) : 1U; }();
    return k;
}

// STRIX_HCW_NORMFREE (fast prefill profile, BF16 streams, FP32 norm weights, lane-chain output): the norm writes only the
// stream inverses and the projection and output recompute the normalized values from the streams (the 335 MB of FP32
// rows of an 8192-row call are neither written nor read). The streams must be uniform rows not overlapping the output.
bool hcw_normfree_rows(const PreparedHyperRows& prepared, const HyperTileRows& tile_rows, const HyperWideRows& wide,
    const HipHyperMixLauncherConfig& config, HyperWideRows& rows) {
    const auto& p = prepared.first;
    if (!config.fast_prefill_wmma || config.activation != HipActivationStorage::bf16 || fast_output_kernel() != 1U ||
        wide.count > kHcwLowRows || p.norm_manifest->ggml_type != 0U || tile_rows.tile_input % kWindowRows)
        return false;
    const auto at = [](const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer); };
    const std::uint64_t stride = tile_rows.tile_input / kWindowRows;
    for (std::uint32_t r = 0; r < kWindowRows; ++r)
        if (!tile_rows.input[r] || at(tile_rows.input[r]) != at(tile_rows.input[0]) + r * stride) return false;
    const std::uintptr_t in0 = at(tile_rows.input[0]), in1 = in0 + wide.count * stride, out0 = at(wide.output),
        out1 = out0 + wide.count * wide.output_stride;
    if (in0 % 16U || stride % 16U || stride < 10240U * 2U || p.norm_runtime->address % 16U || (in0 < out1 && out0 < in1)) return false;
    rows.input = tile_rows.input[0];
    rows.input_stride = stride;
    rows.inverse = hcw_inverse_scratch(wide.count);
    rows.norm_weight = reinterpret_cast<const float*>(p.norm_runtime->address);
    return true;
}

// norm_launched (STRIX_QPF2K_HC folds): the normalized rows were already written by the folded inject + norm kernel;
// everything else (projection, output, statistics) runs as with norm.
void launch_wide_hyper_rows(const PreparedHyperRows& prepared, const HyperTileRows& tile_rows, const HyperWideRows& wide,
    const HipHyperMixLauncherConfig& config, std::uint32_t tiles, bool norm, bool allow_normfree = true, bool norm_launched = false) {
    const auto& p = prepared.first;
    const auto* inject = p.inject_runtime ? reinterpret_cast<const void*>(p.inject_runtime->address) : nullptr;
    // Fast prefill profile, STRIX_HCW_NORM16: 1 = the norm also writes the FP16 rows the WMMA projection reads (FP32 rows
    // kept for the output's mix); 2 = the norm writes only the FP16 rows and the lane-chain output's mix reads them too.
    // Mode 1 measured no net gain in the engine (16K, T=8192 trace: projection -2 %, norm +25 %).
    static const unsigned norm16_env = [] { const char* v = std::getenv("STRIX_HCW_NORM16"); return v ? static_cast<unsigned>(std::atoi(v)) : 0U; }();
    HyperWideRows rows = wide;
    const bool norm16 = norm && config.fast_prefill_wmma && (norm16_env == 1U || norm16_env == 2U) && wide.count <= kHcwLowRows;
    const bool only16 = norm16 && norm16_env == 2U && fast_output_kernel() == 1U;
    if (norm16) { rows.normalized16 = hcw_normalized16_scratch(wide.count); rows.normalized16_pitch = kHcwNorm16Pitch; rows.normalized16_only = only16; }
    static const bool normfree_env = [] { const char* v = std::getenv("STRIX_HCW_NORMFREE"); return v && v[0] == '1'; }();
    const bool normfree = norm && !norm_launched && !norm16 && normfree_env && allow_normfree && hcw_normfree_rows(prepared, tile_rows, wide, config, rows);
    if (norm_launched && norm16) throw std::logic_error("qpf2k HC fold with STRIX_HCW_NORM16");
    const auto invoke = [&]<typename Storage>() {
        if (norm_launched) {
        } else if (normfree) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_prepare_hyper_inverse_kernel<Storage>), dim3(config.hc_count, kWindowRows, tiles),
                dim3(kThreads), 0, hip_execution_stream(), tile_rows, config.hidden_size, config.rms_epsilon, hcw_inverse_scratch(wide.count));
            check_hip(hipGetLastError(), "prefill HC wide rows inverse launch");
        } else if (norm16) {
            const auto kernel = only16 ? rows_prepare_hyper_norm16_kernel<Storage, false> : rows_prepare_hyper_norm16_kernel<Storage, true>;
            hipLaunchKernelGGL(kernel, dim3(config.hc_count, kWindowRows, tiles),
                dim3(kThreads), 0, hip_execution_stream(), tile_rows, reinterpret_cast<const void*>(p.norm_runtime->address),
                p.norm_manifest->ggml_type, config.hidden_size, config.rms_epsilon, hcw_normalized16_scratch(wide.count));
            check_hip(hipGetLastError(), "prefill HC wide rows FP16 normalization launch");
        } else if (norm) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_prepare_hyper_norm_kernel<Storage, HyperTileRows>), dim3(config.hc_count, kWindowRows, tiles),
                dim3(kThreads), 0, hip_execution_stream(), tile_rows, reinterpret_cast<const void*>(p.norm_runtime->address),
                p.norm_manifest->ggml_type, config.hidden_size, config.rms_epsilon);
            check_hip(hipGetLastError(), "prefill HC wide rows normalization launch");
        }
    };
    switch (config.activation) {
    case HipActivationStorage::bf16: invoke.template operator()<std::uint16_t>(); break;
    case HipActivationStorage::fp32: invoke.template operator()<float>(); break;
    }
    // Fast prefill profile: the FP16 WMMA kernels (2, 2).
    launch_wide_hyper_kernels(rows, reinterpret_cast<const void*>(p.down_runtime->address), inject,
        reinterpret_cast<const void*>(p.up_runtime->address), config, config.fast_prefill_wmma ? 2U : wide_projection_kernel(),
        config.fast_prefill_wmma ? fast_output_kernel() : wide_output_kernel(), hip_execution_stream());
    if (norm && config.arithmetic_stats) {
        config.arithmetic_stats->materialized_norm_submissions.fetch_add(tiles, std::memory_order_relaxed);
        config.arithmetic_stats->splitk_f32_submissions.fetch_add(tiles, std::memory_order_relaxed);
    }
}

// Check mode: every value the HC step writes (low, injection, output) of every row, as bytes.
std::vector<unsigned char> snapshot_wide_hyper_rows(const HyperWideRows& wide, const HipHyperMixLauncherConfig& config) {
    check_hip(hipStreamSynchronize(hip_execution_stream()), "HC wide check synchronize");
    const std::uint64_t storage = config.activation == HipActivationStorage::bf16 ? 2U : 4U;
    const std::uint64_t low_bytes = config.lowrank * 4ULL, injection_bytes = wide.injection ? config.hc_count * 4ULL : 0U,
        output_bytes = config.hidden_size * storage;
    std::vector<unsigned char> out;
    const auto copy = [&](const void* base, std::uint64_t stride, std::uint64_t bytes) {
        const auto at = out.size();
        out.resize(at + bytes * wide.count);
        check_hip(hipMemcpy2D(out.data() + at, bytes, base, stride, bytes, wide.count, hipMemcpyDeviceToHost), "HC wide check copy");
    };
    copy(wide.normalized + 10240U, wide.normalized_stride, low_bytes);
    if (injection_bytes) copy(wide.injection, wide.injection_stride, injection_bytes);
    copy(wide.output, wide.output_stride, output_bytes);
    return out;
}
}  // namespace

void reserve_hyper_mix_prefill_pools(std::uint32_t rows) {
    hc_prefill_pool_owner().setup(rows, [=] {
        (void)wide2_low_scratch(rows, true); (void)hcw_scratch(rows, true);
        (void)hcw_inverse_scratch(rows, true); (void)hcw_normalized16_scratch(rows, true);
    });
}

void hyper_mix_known_prefill_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipHyperMixLauncherConfig& config) {
    if (rows.size() > hc_prefill_pool_owner().rows())
        throw std::runtime_error("HC prefill call exceeds setup capacity");
    std::uint32_t tiles = 0;
    (void)prepare_hyper_prefill_rows(rows, config, tiles);
}

void hyper_mix_known_prefill_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipHyperMixLauncherConfig& config) {
    if (rows.size() > hc_prefill_pool_owner().rows())
        throw std::runtime_error("HC prefill call exceeds setup capacity");
    std::uint32_t tiles = 0;
    const auto prepared = prepare_hyper_prefill_rows(rows, config, tiles);
    HyperTileRows tile_rows;
    static_cast<HyperRows&>(tile_rows) = prepared.rows;
    // STRIX_HC_TILES_PER_BLOCK: 8-row tiles one lane-chain block runs on its staged weights.
    static const std::uint32_t per_block = [] {
        const char* v = std::getenv("STRIX_HC_TILES_PER_BLOCK");
        const int n = v ? std::atoi(v) : 4;
        return n == 1 || n == 2 || n == 4 || n == 8 ? static_cast<std::uint32_t>(n) : 4U;
    }();
    tile_rows.tiles = tiles % per_block ? 1U : per_block;
    HyperWideRows wide{};
    if (!wide_hyper_rows(prepared, config, tiles, wide)) {
        launch_hyper_rows(prepared, tile_rows, config, tiles);
        return;
    }
    // STRIX_HC_WIDE_CHECK=1: also run the 8-row tile kernels and compare every written value bitwise.
    static const bool check = [] { const char* v = std::getenv("STRIX_HC_WIDE_CHECK"); return v && v[0] == '1'; }();
    // STRIX_HCW_NORMFREE_CHECK=1: run the step with and without STRIX_HCW_NORMFREE and compare every written value bitwise.
    static const bool normfree_check = [] { const char* v = std::getenv("STRIX_HCW_NORMFREE_CHECK"); return v && v[0] == '1'; }();
    if (normfree_check && config.fast_prefill_wmma) {
        launch_wide_hyper_rows(prepared, tile_rows, wide, config, tiles, true, false);
        const auto reference = snapshot_wide_hyper_rows(wide, config);
        launch_wide_hyper_rows(prepared, tile_rows, wide, config, tiles, true);
        const auto candidate = snapshot_wide_hyper_rows(wide, config);
        std::uint64_t differ = 0;
        for (std::uint64_t i = 0; i < reference.size(); ++i) differ += reference[i] != candidate[i];
        std::fprintf(stderr, "HC norm-free check: rows %u bytes %llu differ %llu\n", wide.count,
            static_cast<unsigned long long>(reference.size()), static_cast<unsigned long long>(differ));
        if (differ) throw std::runtime_error("HC norm-free rows differ");
        return;
    }
    if (!check || config.fast_prefill_wmma) {
        launch_wide_hyper_rows(prepared, tile_rows, wide, config, tiles, true);
        return;
    }
    launch_hyper_rows(prepared, tile_rows, config, tiles);
    const auto reference = snapshot_wide_hyper_rows(wide, config);
    launch_wide_hyper_rows(prepared, tile_rows, wide, config, tiles, false);
    const auto candidate = snapshot_wide_hyper_rows(wide, config);
    std::uint64_t differ = 0, first = UINT64_MAX;
    for (std::uint64_t i = 0; i < reference.size(); ++i)
        if (reference[i] != candidate[i]) { ++differ; if (first == UINT64_MAX) first = i; }
    std::fprintf(stderr, "HC wide check: rows %u bytes %llu differ %llu first %lld\n", wide.count,
        static_cast<unsigned long long>(reference.size()), static_cast<unsigned long long>(differ),
        first == UINT64_MAX ? -1LL : static_cast<long long>(first));
    if (differ) throw std::runtime_error("HC wide rows differ from the 8-row tile kernels");
}

bool hyper_mix_prefill_fold_enabled(bool combine) { return (qpf2k_hc_mask() & (combine ? 4U : 2U)) != 0U; }

namespace {
std::atomic<std::uint64_t> qpf2k_folds{0}, qpf2k_private_workspaces{0};  // private: folds that moved their workspace
// Owner-held HC workspace (normalized + low rows) for a fold whose arena workspace aliases a folded input: the workspace
// is dead after the HC step, so the fold may write it elsewhere (as the window folds do). One buffer per stream, grown
// on demand (the stream is drained before a smaller buffer is freed).
float* qpf2k_hc_workspace(hipStream_t stream, std::uint64_t bytes) {
    struct Buffer { void* memory = nullptr; std::uint64_t bytes = 0; };
    static std::mutex mutex;
    static std::map<hipStream_t, Buffer> buffers;
    const std::lock_guard lock(mutex);
    auto& b = buffers[stream];
    if (b.bytes < bytes) {
        if (b.memory) { check_hip(hipStreamSynchronize(stream), "drain qpf2k HC workspace"); check_hip(hipFree(b.memory), "free qpf2k HC workspace"); }
        b.memory = nullptr; b.bytes = 0;
        check_hip(hipMalloc(&b.memory, bytes), "qpf2k HC workspace allocation");
        b.bytes = bytes;
    }
    return static_cast<float*>(b.memory);
}
bool qpf2k_decline(const char* why) {
    if (std::getenv("STRIX_QPF2K_HC_VERBOSE")) std::fprintf(stderr, "qpf2k HC fold declined: %s\n", why);
    return false;
}
// Check mode: device copy of `rows` rows of `bytes` bytes one `stride` apart (inputs the reference launches overwrite).
struct Qpf2kSaved {
    void* base = nullptr;
    std::uint64_t stride = 0, bytes = 0;
    std::uint32_t rows = 0;
    void* copy = nullptr;
    Qpf2kSaved(const void* b, std::uint64_t s, std::uint64_t n, std::uint32_t r, hipStream_t stream)
        : base(const_cast<void*>(b)), stride(s), bytes(n), rows(r) {
        check_hip(hipMalloc(&copy, bytes * rows), "qpf2k HC check save allocation");
        check_hip(hipMemcpy2DAsync(copy, bytes, base, stride, bytes, rows, hipMemcpyDeviceToDevice, stream), "qpf2k HC check save");
    }
    void restore(hipStream_t stream) const {
        check_hip(hipMemcpy2DAsync(base, stride, copy, bytes, bytes, rows, hipMemcpyDeviceToDevice, stream), "qpf2k HC check restore");
    }
    ~Qpf2kSaved() {
        if (copy) { (void)hipStreamSynchronize(hip_execution_stream()); (void)hipFree(copy); }
    }
    Qpf2kSaved(const Qpf2kSaved&) = delete;
    Qpf2kSaved& operator=(const Qpf2kSaved&) = delete;
};
}  // namespace

bool hyper_mix_known_prefill_rows_launch_folded(std::span<const ExecutionDispatchContext> combine_rows,
    std::span<const ExecutionDispatchContext> inject_rows, std::span<const ExecutionDispatchContext> rows,
    const HipHyperMixLauncherConfig& config, const HipElementwiseLauncherConfig& elementwise) {
    const bool combine = !combine_rows.empty();
    // Partial blocks (STRIX_QPF2K_SCHED bit 4: the first tokens.size() rows of the T arena, a multiple of 8) fold like
    // full ones; the prefill elementwise admission needs 16 rows or more.
    if (rows.size() < 16U || rows.size() % 8U || !config.fast_prefill_wmma || config.activation != HipActivationStorage::bf16 || elementwise.activation != config.activation ||
        elementwise.hidden_size != config.hidden_size || elementwise.hc_count != config.hc_count || config.hidden_size != kInjectNormHidden ||
        config.hc_count != kInjectNormStreams || inject_rows.size() != rows.size() || (combine && combine_rows.size() != rows.size()) ||
        fast_output_kernel() != 1U)
        return qpf2k_decline("configuration");
    // The fold replaces the plain FP32 norm only (no other norm form or self-check of the separate launches).
    static const bool other_forms = [] {
        const auto on = [](const char* n) { const char* v = std::getenv(n); return v && v[0] == '1'; };
        const char* n16 = std::getenv("STRIX_HCW_NORM16");
        return on("STRIX_HCW_NORMFREE") || on("STRIX_HCW_NORMFREE_CHECK") || on("STRIX_HC_WIDE_CHECK") || (n16 && std::atoi(n16) != 0);
    }();
    if (other_forms) return qpf2k_decline("other HC norm forms or checks requested");
    ElementwisePrefillViews in{}, cb{};
    if (!inject_known_prefill_rows_views(inject_rows, elementwise, in)) return qpf2k_decline("inject views not uniform");
    if (combine && !moe_combine_known_prefill_rows_views(combine_rows, elementwise, cb)) return qpf2k_decline("combine views not uniform");
    std::uint32_t tiles = 0;
    const auto prepared = prepare_hyper_prefill_rows(rows, config, tiles);
    HyperWideRows wide{};
    if (!wide_hyper_rows(prepared, config, tiles, wide)) return qpf2k_decline("HC rows not wide");
    const auto& p = prepared.first;
    const auto& r = prepared.rows;
    const auto at = [](const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer); };
    if (r.tile_input % kWindowRows) return qpf2k_decline("HC input not uniform");
    const std::uint64_t input_stride = r.tile_input / kWindowRows;
    for (std::uint32_t k = 0; k < kWindowRows; ++k)
        if (at(r.input[k]) != at(r.input[0]) + k * input_stride) return qpf2k_decline("HC input not uniform");
    const std::uint32_t count = wide.count;
    if (in.count != count || (combine && cb.count != count)) return qpf2k_decline("row counts differ");
    // The HC step reads exactly the streams the inject writes; the inject reads the combine's output.
    if (at(in.output) != at(r.input[0]) || in.output_stride != input_stride) return qpf2k_decline("the HC step does not read the injected streams");
    if (combine && (at(cb.output) != at(in.second) || cb.output_stride != in.second_stride))
        return qpf2k_decline("the inject does not read the combined rows");
    constexpr std::uint64_t width = 10240U * 2U, hidden = 2560U * 2U, workspace = (10240U + 320U) * 4U;
    const auto aligned = [&](const void* x, std::uint64_t stride) { return at(x) % 16U == 0U && stride % 16U == 0U; };
    if (!aligned(in.first, in.first_stride) || !aligned(in.second, in.second_stride) || !aligned(in.output, in.output_stride) ||
        !aligned(wide.normalized, wide.normalized_stride) || at(in.weights) % 4U || in.weights_stride % 4U ||
        (combine && (!aligned(cb.first, cb.first_stride) || !aligned(cb.second, cb.second_stride))))
        return qpf2k_decline("alignment");
    // Every row runs at once: no written extent may reach an extent another row reads (a row's own streams updated in
    // place, and the combined row replacing its own row's shared or routed input exactly, excepted).
    struct Extent { std::uintptr_t low, high; };
    const auto extent = [&](const void* base, std::uint64_t stride, std::uint64_t bytes) {
        return Extent{at(base), at(base) + static_cast<std::uint64_t>(count - 1U) * stride + bytes};
    };
    const auto overlap = [](Extent x, Extent y) { return x.low < y.high && y.low < x.high; };
    const Extent streams = extent(in.output, in.output_stride, width), hyper = extent(in.first, in.first_stride, width),
        gates = extent(in.weights, in.weights_stride, 16U), normalized = extent(wide.normalized, wide.normalized_stride, workspace);
    const Extent shared = combine ? extent(cb.first, cb.first_stride, hidden) : extent(in.second, in.second_stride, hidden);
    const Extent routed = combine ? extent(cb.second, cb.second_stride, hidden) : Extent{0U, 0U};
    const Extent combined = combine ? extent(cb.output, cb.output_stride, hidden) : Extent{0U, 0U};
    const bool in_place = at(in.first) == at(in.output) && in.first_stride == in.output_stride;
    if (overlap(streams, hyper) && !in_place) return qpf2k_decline("streams alias the input streams");
    if (overlap(streams, shared) || overlap(streams, routed) || overlap(streams, gates)) {
        if (std::getenv("STRIX_QPF2K_HC_VERBOSE"))
            std::fprintf(stderr, "qpf2k HC fold streams [%#lx,%#lx) shared %d routed %d gates %d (combine %d)\n", static_cast<unsigned long>(streams.low),
                static_cast<unsigned long>(streams.high), overlap(streams, shared) ? 1 : 0, overlap(streams, routed) ? 1 : 0, overlap(streams, gates) ? 1 : 0,
                combine ? 1 : 0);
        return qpf2k_decline("streams alias an input");
    }
    if (combine) {
        const auto exact = [&](const void* base, std::uint64_t stride) { return at(base) == at(cb.output) && stride == cb.output_stride; };
        if ((overlap(combined, shared) && !exact(cb.first, cb.first_stride)) || (overlap(combined, routed) && !exact(cb.second, cb.second_stride)))
            return qpf2k_decline("combined rows alias an input");
        if (overlap(combined, hyper) || overlap(combined, streams) || overlap(combined, gates))
            return qpf2k_decline("combined rows alias the streams");
    }
    // An arena workspace that aliases a folded input (dead MoE / attention temporaries placed over it) moves to an
    // owner-held buffer: the fold writes rows while other rows still read their inputs.
    const Extent hidden_rows = combine ? Extent{0U, 0U} : shared;
    bool private_workspace = false;
    for (const Extent x : {hyper, streams, shared, routed, gates, combined, hidden_rows})
        private_workspace = private_workspace || overlap(normalized, x);
    const HyperWideRows arena_wide = wide;
    if (private_workspace) {
        constexpr std::uint64_t pitch = workspace;  // 42,240 bytes, a multiple of 16
        wide.normalized = qpf2k_hc_workspace(hip_execution_stream(), pitch * count);
        wide.normalized_stride = pitch;
        qpf2k_private_workspaces.fetch_add(1U, std::memory_order_relaxed);
        if (std::getenv("STRIX_QPF2K_HC_VERBOSE")) std::fprintf(stderr, "qpf2k HC fold private workspace rows %u\n", count);
    }
    Qpf2kInjectNormRows a{};
    a.hyper = static_cast<const std::uint16_t*>(in.first); a.hyper_stride = in.first_stride;
    a.hidden = static_cast<const std::uint16_t*>(combine ? cb.first : in.second); a.hidden_stride = combine ? cb.first_stride : in.second_stride;
    a.routed = combine ? static_cast<const std::uint16_t*>(cb.second) : nullptr; a.routed_stride = combine ? cb.second_stride : 0U;
    a.gates = in.weights; a.gates_stride = in.weights_stride;
    a.combined = combine ? static_cast<std::uint16_t*>(cb.output) : nullptr; a.combined_stride = combine ? cb.output_stride : 0U;
    a.streams = static_cast<std::uint16_t*>(in.output); a.streams_stride = in.output_stride;
    a.normalized = wide.normalized; a.normalized_stride = wide.normalized_stride;
    if (p.norm_manifest->ggml_type != 0U || p.norm_runtime->address % 16U) return qpf2k_decline("HC norm weights not 16-byte aligned FP32");
    a.norm = reinterpret_cast<const float*>(p.norm_runtime->address);
    a.count = count; a.hidden_size = config.hidden_size; a.epsilon = config.rms_epsilon;
    HyperTileRows tile_rows;
    static_cast<HyperRows&>(tile_rows) = prepared.rows;
    tile_rows.tiles = 1U;
    const auto stream = hip_execution_stream();
    const auto run = [&] {
        if (combine) hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_inject_norm_kernel<true>), dim3(count), dim3(128U), 0, stream, a);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_inject_norm_kernel<false>), dim3(count), dim3(128U), 0, stream, a);
        check_hip(hipGetLastError(), "qpf2k HC folded inject + norm launch");
        launch_wide_hyper_rows(prepared, tile_rows, wide, config, tiles, true, true, true);
    };
    if (!qpf2k_hc_check()) {
        run();
        qpf2k_folds.fetch_add(1U, std::memory_order_relaxed);
        if (std::getenv("STRIX_QPF2K_HC_VERBOSE")) std::fprintf(stderr, "qpf2k HC fold %s rows %u\n", combine ? "combine+inject+norm" : "inject+norm", count);
        return true;
    }
    // Check: the separate launches first (their overwritten inputs saved, then restored), then the fold; every written
    // byte (streams, normalized + low, injection, output, combined) compared.
    const auto snapshot = [&](const HyperWideRows& ws) {
        std::vector<std::vector<unsigned char>> s;
        s.push_back(qpf2k_snapshot_rows(in.output, in.output_stride, width, count, stream));
        s.push_back(qpf2k_snapshot_rows(ws.normalized, ws.normalized_stride, workspace, count, stream));
        if (wide.injection) s.push_back(qpf2k_snapshot_rows(wide.injection, wide.injection_stride, 16U, count, stream));
        s.push_back(qpf2k_snapshot_rows(wide.output, wide.output_stride, hidden, count, stream));
        if (combine) s.push_back(qpf2k_snapshot_rows(cb.output, cb.output_stride, hidden, count, stream));
        return s;
    };
    {
        const Qpf2kSaved saved_hyper(in.first, in.first_stride, width, count, stream);
        const Qpf2kSaved saved_shared(a.hidden, a.hidden_stride, hidden, count, stream);
        const Qpf2kSaved saved_gates(in.weights, in.weights_stride, 16U, count, stream);
        std::unique_ptr<Qpf2kSaved> saved_routed;
        if (combine) saved_routed = std::make_unique<Qpf2kSaved>(cb.second, cb.second_stride, hidden, count, stream);
        if (combine) moe_combine_known_prefill_rows_launch(combine_rows, elementwise);
        inject_known_prefill_rows_launch(inject_rows, elementwise);
        launch_wide_hyper_rows(prepared, tile_rows, arena_wide, config, tiles, true);
        const auto reference = snapshot(arena_wide);
        saved_shared.restore(stream);
        saved_gates.restore(stream);
        if (saved_routed) saved_routed->restore(stream);
        check_hip(hipMemset2DAsync(in.output, in.output_stride, 0x5a, width, count, stream), "qpf2k HC check poison");
        check_hip(hipMemset2DAsync(wide.normalized, wide.normalized_stride, 0x5a, workspace, count, stream), "qpf2k HC check poison");
        saved_hyper.restore(stream);
        run();
        const auto candidate = snapshot(wide);
        std::vector<const char*> names{"fold streams", "fold normalized+low"};
        if (wide.injection) names.push_back("fold injection");
        names.push_back("fold output");
        if (combine) names.push_back("fold combined");
        for (std::size_t i = 0; i < reference.size(); ++i) qpf2k_compare(names[i], reference[i], candidate[i], count);
        std::fprintf(stderr, "qpf2k HC check call fold: rows %u all destinations equal\n", count);
    }
    qpf2k_folds.fetch_add(1U, std::memory_order_relaxed);
    if (std::getenv("STRIX_QPF2K_HC_VERBOSE")) std::fprintf(stderr, "qpf2k HC fold %s rows %u (checked)\n", combine ? "combine+inject+norm" : "inject+norm", count);
    return true;
}
std::uint64_t hip_hyper_mix_testing_prefill_folds() { return qpf2k_folds.load(); }

void testing::enqueue_wide_hyper_rows_test(const HipHyperMixLauncherConfig& config, const WideHyperRowsTestViews& views,
    unsigned projection_kernel, unsigned output_kernel, hipStream_t stream) {
    const auto aligned = [](const void* p, std::uintptr_t a) { return reinterpret_cast<std::uintptr_t>(p) % a == 0U; };
    if (config.hidden_size * config.hc_count != 10240U || config.lowrank != 320U || config.hc_count != kWideStreams ||
        config.hidden_size % (32U * kWideOutputWaves) || !views.rows || !views.normalized || !views.output || !views.down ||
        !views.up || (views.inject && !views.injection) || views.normalized_stride % 16U || !aligned(views.normalized, 16U) ||
        !aligned(views.up, 4U) || (projection_kernel > 4U && projection_kernel != kSkipKernel) ||
        (output_kernel > 14U && output_kernel != kSkipKernel) ||
        (views.output3_band != UINT32_MAX && (!qpf2k_hc_output3_band_valid(views.output3_band) || output_kernel != 7U)) ||
        (views.output3_zero_low_rank_stride && (output_kernel != 7U || projection_kernel != kSkipKernel || views.output3_band != 0U)) ||
        (views.low_band != UINT32_MAX && views.low_band > 3U))
        throw std::invalid_argument("wide HC rows test views invalid");
    HyperWideRows wide{};
    wide.normalized = views.normalized; wide.normalized_stride = views.normalized_stride;
    wide.injection = views.injection; wide.injection_stride = views.injection_stride;
    wide.output = views.output; wide.output_stride = views.output_stride;
    wide.count = views.rows;
    wide.normalized16 = static_cast<const _Float16*>(views.normalized16);
    wide.normalized16_pitch = views.normalized16_pitch ? views.normalized16_pitch : 10240U;
    wide.normalized16_only = views.normalized16 && views.normalized16_only;
    launch_wide_hyper_kernels(wide, views.down, views.inject, views.up, config, projection_kernel, output_kernel, stream,
        views.output3_band, views.output3_zero_low_rank_stride, views.low_band);
}

const float* testing::wide_hyper_rows_test_low_t() { return wide2_low_scratch(); }
const void* testing::wide_hyper_rows_test_low16() { return hcw_scratch().low16; }

void testing::enqueue_qpf2k_inject_norm_test(const HipHyperMixLauncherConfig& config, const Qpf2kInjectNormTestViews& v, bool reference,
    hipStream_t stream) {
    if (config.hidden_size != kInjectNormHidden || config.hc_count != kInjectNormStreams || !v.rows || v.rows % kWindowRows ||
        !v.hyper || !v.streams || !v.hidden || !v.gates || !v.normalized || !v.norm || (v.routed && !v.combined))
        throw std::invalid_argument("qpf2k inject norm test views invalid");
    const bool combine = v.routed != nullptr;
    const auto row = [](const void* base, std::uint64_t stride, std::uint32_t r) {
        return reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(base) + static_cast<std::uint64_t>(r) * stride);
    };
    if (!reference) {
        Qpf2kInjectNormRows a{};
        a.hyper = static_cast<const std::uint16_t*>(v.hyper); a.hyper_stride = v.hyper_stride;
        a.hidden = static_cast<const std::uint16_t*>(v.hidden); a.hidden_stride = v.hidden_stride;
        a.routed = static_cast<const std::uint16_t*>(v.routed); a.routed_stride = v.routed_stride;
        a.gates = v.gates; a.gates_stride = v.gates_stride;
        a.combined = static_cast<std::uint16_t*>(v.combined); a.combined_stride = v.combined_stride;
        a.streams = static_cast<std::uint16_t*>(v.streams); a.streams_stride = v.streams_stride;
        a.normalized = v.normalized; a.normalized_stride = v.normalized_stride;
        a.norm = v.norm; a.count = v.rows; a.hidden_size = config.hidden_size; a.epsilon = config.rms_epsilon;
        if (combine) hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_inject_norm_kernel<true>), dim3(v.rows), dim3(128U), 0, stream, a);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(qpf2k_inject_norm_kernel<false>), dim3(v.rows), dim3(128U), 0, stream, a);
        check_hip(hipGetLastError(), "qpf2k inject norm test launch");
        return;
    }
    // The production prefill norm launch: rows_prepare_hyper_norm_kernel<bf16, HyperTileRows>, grid (streams, 8, tiles).
    HyperTileRows rows{};
    for (std::uint32_t r = 0; r < kWindowRows; ++r) {
        rows.input[r] = row(v.streams, v.streams_stride, r);
        rows.normalized[r] = static_cast<float*>(row(v.normalized, v.normalized_stride, r));
    }
    rows.tile_input = kWindowRows * v.streams_stride;
    rows.tile_normalized = kWindowRows * v.normalized_stride;
    rows.tiles = 1U;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_prepare_hyper_norm_kernel<std::uint16_t, HyperTileRows>), dim3(config.hc_count, kWindowRows, v.rows / kWindowRows),
        dim3(kThreads), 0, stream, rows, static_cast<const void*>(v.norm), 0U, config.hidden_size, config.rms_epsilon);
    check_hip(hipGetLastError(), "qpf2k inject norm reference launch");
}

namespace {
__global__ __launch_bounds__(256) void qpf2k_coef_reference_kernel(const unsigned char* __restrict__ up, float4* __restrict__ coef, std::uint32_t hidden) {
    const unsigned index = blockIdx.x * 256U + threadIdx.x;
    if (index >= hidden * 10U) return;
    const unsigned f = index % hidden, b = index / hidden;
    for (unsigned k = 0; k < 32U; ++k) {
        float c[4];
        for (unsigned s = 0; s < 4U; ++s) {
            const auto* block = up + (static_cast<std::uint64_t>(s) * hidden + f) * 340U + b * 34U;
            const float scale = __half2float(__ushort_as_half(static_cast<unsigned short>(block[0] | (block[1] << 8U))));
            const unsigned u = static_cast<unsigned>(block[2U + k]) ^ 0x80U;
            c[s] = scale * (static_cast<float>(u) - 128.0F);
        }
        coef[static_cast<std::uint64_t>(b * 32U + k) * hidden + f] = make_float4(c[0], c[1], c[2], c[3]);
    }
}
}  // namespace
void testing::enqueue_qpf2k_coef_test(const void* up, std::uint32_t hidden, void* packed, void* reference, hipStream_t stream) {
    hipLaunchKernelGGL(qpf2k_hc_pack_coef_kernel, dim3((hidden * 10U + 255U) / 256U), dim3(256U), 0, stream, static_cast<const unsigned char*>(up),
        static_cast<float4*>(packed), hidden);
    hipLaunchKernelGGL(qpf2k_coef_reference_kernel, dim3((hidden * 10U + 255U) / 256U), dim3(256U), 0, stream, static_cast<const unsigned char*>(up),
        static_cast<float4*>(reference), hidden);
    check_hip(hipGetLastError(), "qpf2k coefficient test launch");
}

void testing::enqueue_lane_chain_hyper_output_test(const HipHyperMixLauncherConfig& config,const float* normalized,const float* low,const void* up,void* output,std::uint32_t rows,bool lane_chain,hipStream_t stream){
    auto selected=config;selected.lane_chain_output=lane_chain;validate_config(selected);
    if(!rows || rows>65535 || !normalized || !low || !up || !output || (lane_chain && reinterpret_cast<std::uintptr_t>(up)%16U))throw std::invalid_argument("lane-chain HC output test owner/rows invalid");
    const auto launch=[&]<typename Storage>(){
        if(lane_chain){hipLaunchKernelGGL(HIP_KERNEL_NAME(lane_chain_hyper_output_kernel<Storage>),dim3(config.hidden_size/32U,rows),dim3(32U*config.hc_count),static_cast<std::size_t>(lane_chain_hyper_output_lds(config.hc_count,config.lowrank)),stream,normalized,low,up,config.hidden_size,config.hc_count,config.lowrank,reinterpret_cast<Storage*>(output));}
        else{hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_batched_hyper_output_kernel<Storage>),dim3((config.hidden_size+kStagedOutputs-1U)/kStagedOutputs,rows),dim3(kStagedThreads),0,stream,normalized,low,up,config.hidden_size,config.hc_count,config.lowrank,reinterpret_cast<Storage*>(output));}
    };
    if(config.activation==HipActivationStorage::bf16)launch.template operator()<std::uint16_t>();else launch.template operator()<float>();
    check_hip(hipGetLastError(),"lane-chain HC same-input test launch");
}

void hc_projection_diagnostic::enqueue(Shape shape,Span normalized,Span down,Span inject,Span low,Span injection,
    std::uint32_t opw,std::uint32_t prefetch,bool reference_rows8,hipStream_t stream){
    if(!shape.width || shape.width>10240U || shape.width%256U || shape.streams<2U || shape.streams>8U || shape.width%shape.streams || !shape.rank || shape.rank>512U || !shape.rows || shape.rows>65535U ||
       (opw!=0U && opw!=8U && opw!=16U && opw!=32U) || (prefetch!=1U && prefetch!=2U))throw std::invalid_argument("HC projection diagnostic unsupported shape/variant");
    const bool with_inject=injection.address!=0;
    if(with_inject!=(inject.address!=0) || (!with_inject && (inject.bytes || injection.bytes)))throw std::invalid_argument("HC projection diagnostic injection owner pair differs");
    const auto check_span=[](Span owner,std::uint64_t required,std::uint64_t alignment){if(!owner.address || owner.address%alignment || owner.bytes<required || owner.bytes>std::numeric_limits<std::uintptr_t>::max()-owner.address)throw std::invalid_argument("HC projection diagnostic owner extent/alignment differs");};
    check_span(normalized,static_cast<std::uint64_t>(shape.rows)*shape.width*4U,4);
    check_span(down,static_cast<std::uint64_t>(shape.rank)*(shape.width/32U)*34U,16);
    check_span(low,static_cast<std::uint64_t>(shape.rows)*shape.rank*4U,4);
    if(with_inject){check_span(inject,static_cast<std::uint64_t>(shape.streams)*shape.width*4U,16);check_span(injection,static_cast<std::uint64_t>(shape.rows)*shape.streams*4U,4);}
    const auto overlap=[](Span x,Span y){return x.address<y.address+y.bytes && y.address<x.address+x.bytes;};
    if(overlap(low,normalized)||overlap(low,down)||(with_inject&&(overlap(low,inject)||overlap(low,injection)||overlap(injection,normalized)||overlap(injection,down)||overlap(injection,inject))))throw std::invalid_argument("HC projection diagnostic owners overlap");
    check_hip(hipPeekAtLastError(),"HC projection diagnostic stale error");
    const auto base_blocks=(shape.rank+kStagedOutputs-1U)/kStagedOutputs;
    if(!opw){
        const dim3 grid(base_blocks+(with_inject?(shape.streams+kStagedOutputs-1U)/kStagedOutputs:0U),shape.rows);
        if(reference_rows8){hipLaunchKernelGGL(batched_typed_hyper_projection_kernel,grid,dim3(kStagedThreads),0,stream,reinterpret_cast<const float*>(normalized.address),reinterpret_cast<const void*>(down.address),shape.width,shape.streams,shape.rank,reinterpret_cast<float*>(low.address),reinterpret_cast<const void*>(inject.address),reinterpret_cast<float*>(injection.address),base_blocks);}
        else{hipLaunchKernelGGL(typed_fused_hyper_projection_kernel,grid,dim3(kStagedThreads),0,stream,reinterpret_cast<const float*>(normalized.address),reinterpret_cast<const void*>(down.address),shape.width,shape.streams,shape.rank,reinterpret_cast<float*>(low.address),reinterpret_cast<const void*>(inject.address),reinterpret_cast<float*>(injection.address),base_blocks);}
    }else{
        const auto launch=[&]<unsigned Outputs,unsigned P>(){
            const unsigned blocks=(shape.rank+Outputs-1U)/Outputs;
            hipLaunchKernelGGL(HIP_KERNEL_NAME(lane_chain_hyper_projection_kernel<Outputs,P>),dim3(blocks+(with_inject?(shape.streams+Outputs-1U)/Outputs:0U),shape.rows),dim3(32),0,stream,reinterpret_cast<const float*>(normalized.address),reinterpret_cast<const void*>(down.address),reinterpret_cast<const float*>(inject.address),shape.width,shape.streams,shape.rank,reinterpret_cast<float*>(low.address),reinterpret_cast<float*>(injection.address),blocks);
        };
        if(opw==8U){if(prefetch==1U)launch.template operator()<8,1>();else launch.template operator()<8,2>();}
        else if(opw==16U){if(prefetch==1U)launch.template operator()<16,1>();else launch.template operator()<16,2>();}
        else{if(prefetch==1U)launch.template operator()<32,1>();else launch.template operator()<32,2>();}
    }
    check_hip(hipGetLastError(),"HC projection diagnostic enqueue");
}

void hc_projection_diagnostic::enqueue_splitk(
    Shape shape,Span normalized,Span down,Span inject,Span low,Span injection,
    Span partial,SplitKVariant variant,hipStream_t stream) {
    const bool is_p2=variant==SplitKVariant::p2_r4;
    if(shape.width!=10240U || shape.streams<2U || shape.streams>8U ||
       !shape.rank || shape.rank>512U || !shape.rows || shape.rows>65535U ||
       (variant!=SplitKVariant::p1_r4 && variant!=SplitKVariant::p1_r2 &&
        variant!=SplitKVariant::p2_r4 && variant!=SplitKVariant::p1_r4_fp64))
        throw std::invalid_argument("HC split-K projection unsupported shape/variant");
    const bool with_inject=inject.address!=0U;
    if(with_inject!=(injection.address!=0U) ||
       (!with_inject && (inject.bytes!=0U || injection.bytes!=0U)))
        throw std::invalid_argument("HC split-K projection injection owner pair differs");
    if(!is_p2 && (partial.address!=0U || partial.bytes!=0U))
        throw std::invalid_argument("HC split-K projection unexpected partial owner");
    const auto check_span=[](Span owner,std::uint64_t required,std::uint64_t alignment) {
        if(!owner.address || owner.address%alignment || owner.bytes<required ||
           owner.bytes>std::numeric_limits<std::uintptr_t>::max()-owner.address)
            throw std::invalid_argument("HC split-K projection owner extent/alignment differs");
    };
    check_span(normalized,static_cast<std::uint64_t>(shape.rows)*10240U*4U,16U);
    check_span(down,static_cast<std::uint64_t>(shape.rank)*320U*34U,16U);
    check_span(low,static_cast<std::uint64_t>(shape.rows)*shape.rank*4U,4U);
    if(with_inject) {
        check_span(inject,static_cast<std::uint64_t>(shape.streams)*10240U*4U,16U);
        check_span(injection,static_cast<std::uint64_t>(shape.rows)*shape.streams*4U,4U);
    }
    const unsigned pitch=shape.rank+(with_inject?shape.streams:0U);
    if(is_p2)check_span(partial,static_cast<std::uint64_t>(shape.rows)*16U*pitch*4U,16U);
    const auto overlaps=[](Span a,Span b) {
        return a.address<b.address+b.bytes && b.address<a.address+a.bytes;
    };
    const Span inputs[3]={normalized,down,inject};
    const Span outputs[3]={low,injection,partial};
    for(unsigned i=0;i<3U;++i)if(outputs[i].address) {
        for(const auto input:inputs)if(input.address && overlaps(outputs[i],input))
            throw std::invalid_argument("HC split-K projection owners overlap");
        for(unsigned j=i+1U;j<3U;++j)if(outputs[j].address && overlaps(outputs[i],outputs[j]))
            throw std::invalid_argument("HC split-K projection owners overlap");
    }
    check_hip(hipPeekAtLastError(),"HC split-K projection stale error");
    int device=0;hipDeviceProp_t properties{};
    check_hip(hipGetDevice(&device),"HC split-K projection current device");
    check_hip(hipGetDeviceProperties(&properties,device),"HC split-K projection device properties");
    if(properties.warpSize!=32)throw std::invalid_argument("HC split-K projection requires wave32");
    const auto* n=reinterpret_cast<const float*>(normalized.address);
    const auto* d=reinterpret_cast<const void*>(down.address);
    const auto* in=reinterpret_cast<const void*>(inject.address);
    auto* l=reinterpret_cast<float*>(low.address);
    auto* out=reinterpret_cast<float*>(injection.address);
    if(is_p2) {
        const unsigned low_blocks=(shape.rank+31U)/32U;
        auto* p=reinterpret_cast<float*>(partial.address);
        hipLaunchKernelGGL(splitk_hyper_partial_kernel,
            dim3(low_blocks+(with_inject?1U:0U),16U,shape.rows),dim3(256U),0,stream,
            n,d,reinterpret_cast<const float*>(in),shape.streams,shape.rank,low_blocks,p);
        check_hip(hipGetLastError(),"HC split-K projection partial enqueue");
        hipLaunchKernelGGL(splitk_hyper_finalize_kernel,
            dim3((pitch+127U)/128U,shape.rows),dim3(128U),0,stream,
            p,shape.streams,shape.rank,pitch,l,out);
    } else {
        const auto launch=[&]<unsigned R,typename Acc>() {
            const unsigned low_blocks=(shape.rank+R-1U)/R;
            hipLaunchKernelGGL(HIP_KERNEL_NAME(splitk_hyper_projection_kernel<R,Acc>),
                dim3(low_blocks+(with_inject?(shape.streams+R-1U)/R:0U),shape.rows),
                dim3(512U),0,stream,n,d,shape.width,shape.streams,shape.rank,l,in,out,low_blocks);
        };
        if(variant==SplitKVariant::p1_r2)launch.template operator()<2U,float>();
        else if(variant==SplitKVariant::p1_r4_fp64)launch.template operator()<4U,double>();
        else launch.template operator()<4U,float>();
    }
    check_hip(hipGetLastError(),"HC split-K projection final enqueue");
}

void hc_deferred_norm_diagnostic::enqueue(Span raw,Span norm,Span down,Span inject,Span up,
    Span inverse,Span low,Span injection,Span output,Span inverse_audit,Span scaled_audit,float epsilon,hipStream_t stream) {
    const bool with_inject=inject.address!=0U;
    if(!std::isfinite(epsilon) || epsilon<=0.0F || with_inject!=(injection.address!=0U) ||
        (!with_inject && (inject.bytes || injection.bytes)))
        throw std::invalid_argument("deferred HC diagnostic epsilon/injection differs");
    const auto check=[](Span span,std::uint64_t required,std::uint64_t alignment) {
        if(!span.address || span.address%alignment || span.bytes<required ||
            span.bytes>std::numeric_limits<std::uintptr_t>::max()-span.address)
            throw std::invalid_argument("deferred HC diagnostic owner extent/alignment differs");
    };
    check(raw,20480U,2U);check(norm,40960U,4U);check(down,3481600U,16U);
    check(up,3481600U,16U);check(inverse,16U,4U);check(low,1280U,4U);check(output,5120U,2U);
    if(with_inject){check(inject,163840U,16U);check(injection,16U,4U);}
    if(inverse_audit.address)check(inverse_audit,(80U+(with_inject?1U:0U))*16U,4U);
    else if(inverse_audit.bytes)throw std::invalid_argument("deferred HC diagnostic empty audit extent differs");
    if(scaled_audit.address)check(scaled_audit,(320U+(with_inject?4U:0U))*4U,4U);
    else if(scaled_audit.bytes)throw std::invalid_argument("deferred HC diagnostic empty scaled audit extent differs");
    const auto overlaps=[](Span a,Span b) {
        return a.address && b.address && a.address<b.address+b.bytes && b.address<a.address+a.bytes;
    };
    const std::array<Span,5> reads{raw,norm,down,inject,up};
    const std::array<Span,6> writes{inverse,low,injection,output,inverse_audit,scaled_audit};
    for(std::size_t i=0;i<writes.size();++i) {
        for(const auto read:reads)if(overlaps(writes[i],read))
            throw std::invalid_argument("deferred HC diagnostic read/write alias");
        for(std::size_t j=i+1;j<writes.size();++j)if(overlaps(writes[i],writes[j]))
            throw std::invalid_argument("deferred HC diagnostic mutable alias");
    }
    hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
    check_hip(hipStreamIsCapturing(stream,&capture),"deferred HC diagnostic capture query");
    if(capture!=hipStreamCaptureStatusNone || hip_graph_status() ||
        (current_hip_launch_context() && current_hip_launch_context()->graph_capture))
        throw std::invalid_argument("deferred HC diagnostic rejects capture");
    int device=0;hipDeviceProp_t properties{};
    check_hip(hipGetDevice(&device),"deferred HC diagnostic device");
    check_hip(hipGetDeviceProperties(&properties,device),"deferred HC diagnostic device properties");
    if(properties.warpSize!=32 || std::string_view(properties.gcnArchName,7U)!="gfx1151")
        throw std::invalid_argument("deferred HC diagnostic requires gfx1151 wave32");
    check_hip(hipPeekAtLastError(),"deferred HC diagnostic stale error");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(deferred_hyper_projection_kernel<4U>),
        dim3(80U+(with_inject?1U:0U)),dim3(512U),0,stream,
        reinterpret_cast<const std::uint16_t*>(raw.address),reinterpret_cast<const float*>(norm.address),
        reinterpret_cast<const void*>(down.address),320U,epsilon,reinterpret_cast<float*>(inverse.address),
        reinterpret_cast<float*>(low.address),reinterpret_cast<const void*>(inject.address),
        reinterpret_cast<float*>(injection.address),80U,reinterpret_cast<float*>(inverse_audit.address),
        reinterpret_cast<float*>(scaled_audit.address));
    check_hip(hipGetLastError(),"deferred HC diagnostic projection enqueue");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(deferred_lane_chain_hyper_output_kernel<std::uint16_t>),
        dim3(80U),dim3(128U),static_cast<std::size_t>(lane_chain_hyper_output_lds(4U,320U)),stream,
        reinterpret_cast<const std::uint16_t*>(raw.address),reinterpret_cast<const float*>(inverse.address),
        reinterpret_cast<const float*>(norm.address),reinterpret_cast<const float*>(low.address),
        reinterpret_cast<const void*>(up.address),2560U,4U,320U,reinterpret_cast<std::uint16_t*>(output.address));
    check_hip(hipGetLastError(),"deferred HC diagnostic lane-chain enqueue");
}

void register_hip_hyper_mix_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipHyperMixLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification) {
    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::invalid_argument(
            error);
    }
    validate_config(config);
    if (config.hidden_size !=
            model.hidden_size ||
        config.hc_count !=
            model.hc_count ||
        config.lowrank !=
            model.hc_lowrank) {
        throw std::invalid_argument(
            "HIP hyper-mix config does not match model contract");
    }
    if (request.logical_rows == 0 ||
        request.request_concurrency == 0) {
        throw std::invalid_argument(
            "HIP hyper-mix variant requires non-zero rows and concurrency");
    }
    if (config.splitk_projection && (request.logical_rows != 1U || request.request_concurrency != 1U ||
        request.require_graph_safe || request.device_graph_inputs))
        throw std::invalid_argument("split-K HC variant requires eager one-row serial execution");
    if (qualification ==
            kernels::QualificationStatus::target_qualified ||
        qualification ==
            kernels::QualificationStatus::production ||
        qualification ==
            kernels::QualificationStatus::rejected) {
        throw std::invalid_argument(
            "unmeasured HIP hyper-mix may only be experimental or oracle-matched");
    }

    if(request.prefetch_hc_projection_next_tile!=config.prefetch_hc_projection_next_tile)
        throw std::invalid_argument("next-tile HC request/config mismatch");
    if(config.prefetch_hc_projection_next_tile && (model.hidden_layers!=48U || request.logical_rows!=1U || request.request_concurrency!=1U ||
        request.device_graph_inputs || request.require_graph_safe))
        throw std::invalid_argument("next-tile HC registers canonical48-layer eager R1 C1 only");
    if(config.lane_chain_output && !request.lane_chain_hyper_output)
        throw std::invalid_argument("lane-chain HC output requires matching binding opt-in");
    kernels::KernelVariant variant;
    variant.id =
        kernel_id(
            request,
            config);
    variant.family =
        kernels::OperatorFamily::
            hyper_mix;
    variant.phase =
        kernels::ExecutionPhase::decode;
    variant.shape =
        execution_shape_for(
            model,
            SerialOp::attn_hyper_mix,
            request);
    variant.wave =
        kernels::WaveSize::wave32;
    variant.resources.workgroup_threads =
        config.splitk_projection ? 512U : kThreads;
    variant.resources.lds_bytes =
        (kMaxHc * kThreads +
         kMaxHc +
         kMaxLowrank) *
        sizeof(float);
    if (config.parallel) {
        // Conservative 256-byte-rounded declarations include the shared
        // normalization inverse as well as the reduction array.
        variant.resources.lds_bytes = ((kThreads + 1U) * sizeof(float) + 255U) & ~255ULL;
        if (config.staged_projections)
            variant.resources.lds_bytes = ((kStagedThreads + kStagedOutputs * kStagedPitch) * sizeof(float) + 255U) & ~255ULL;
        variant.resources.scratch_bytes = static_cast<std::uint64_t>(request.logical_rows) *
            (static_cast<std::uint64_t>(config.hidden_size) * config.hc_count + config.lowrank) * sizeof(float);
    }
    if(config.lane_chain_output)variant.resources.lds_bytes=std::max(variant.resources.lds_bytes,
        static_cast<std::uint32_t>(lane_chain_hyper_output_lds(config.hc_count,config.lowrank)));
    variant.numerics.input_precision =
        storage_name(
            config.activation);
    variant.numerics.accumulator_precision =
        config.splitk_deferred_norm ? std::string(deferred_hc_numerical_identity) :
        config.splitk_projection_fp64 ? "fp64-projection/fp32-epilogue" : "fp32";
    variant.numerics.output_precision =
        storage_name(
            config.activation);
    variant.numerics.row_invariant = true;
    variant.numerics.deterministic = true;
    variant.qualification =
        qualification;
    variant.graph_safe = !config.prefetch_hc_projection_next_tile && !config.splitk_projection;
    variant.input_layout =
        "hc-stream-major";
    variant.output_layout =
        "row-major+fp32-injection";
    variant.compiler_identity =
        config.splitk_deferred_norm ? "hip-gfx1151-hyper-mix-splitk-fp32-deferred-rms-v1" :
        config.splitk_projection ? (config.splitk_projection_fp64 ? "hip-gfx1151-hyper-mix-splitk-fp64-oracle-v1" : "hip-gfx1151-hyper-mix-splitk-fp32-v1") :
        config.lane_chain_output ? "hip-gfx1151-hyper-mix-v10-lane-chain-output" :
        config.prefetch_q8_output && config.prefetch_rows ? "hip-gfx1151-hyper-mix-v9-q8-output-prefetch8-with-projection-prefetch" :
        config.prefetch_q8_output ? "hip-gfx1151-hyper-mix-v9-q8-output-prefetch8" :
        config.prefetch_rows ? "hip-gfx1151-hyper-mix-v8-typed-prefetch-rows8" :
        config.typed_projection ? "hip-gfx1151-hyper-mix-v6-typed-projection" :
        config.fused_injection ? "hip-gfx1151-hyper-mix-v5-fused-injection" :
        config.staged_projections ? "hip-gfx1151-hyper-mix-v4-staged" :
        config.parallel ? "hip-gfx1151-hyper-mix-v3-parallel" : "hip-gfx1151-hyper-mix-v2";
    if(config.prefetch_hc_projection_next_tile)variant.compiler_identity += ":prefetch-next-tile-v1";
    variant.measured_seconds = 0.0;

    registry.add(
        std::move(variant));
}

ExecutionLauncher
make_hip_hyper_mix_launcher(
    std::string kernel_id_value,
    HipHyperMixLauncherConfig* config) {
    if (kernel_id_value.empty() ||
        config == nullptr) {
        throw std::invalid_argument(
            "HIP hyper-mix launcher requires id and config");
    }
    validate_config(*config);
    constexpr std::string_view deferred_marker=".hc-deferred-norm-v1";
    const auto deferred_position=kernel_id_value.find(deferred_marker);
    if((deferred_position!=std::string::npos)!=config->splitk_deferred_norm ||
        (deferred_position!=std::string::npos &&
         kernel_id_value.find(deferred_marker,deferred_position+deferred_marker.size())!=std::string::npos))
        throw std::invalid_argument("deferred HC RMS launcher identity differs");
    if(config->splitk_deferred_norm || config->prefetch_hc_projection_next_tile || kernel_id_value.find(hc_projection_next_tile_suffix)!=std::string::npos) {
        ExecutionBindingRequest expected;
        if(kernel_id_value!=kernel_id(expected,*config))
            throw std::invalid_argument("HC launcher requires exact eager identity");
    }
    if(kernel_id_value.ends_with(lane_chain_hyper_output_suffix)!=config->lane_chain_output)
        throw std::invalid_argument("lane-chain HC output launcher identity differs");
    return ExecutionLauncher{
        std::move(kernel_id_value),
        implementation_identity(
            *config),
        &launch_hyper_mix,
        config,
        &preflight_hyper_mix,
    };
}

void preflight_typed_hyper_projection_model(const CompiledSerialExecution& compiled,
    const ExecutionRuntimeResources& resources) {
    if (compiled.execution_identity == 0 || resources.model_manifest == nullptr || resources.tensor_access == nullptr)
        throw std::invalid_argument("typed hyper preflight requires a complete compiled model binding");
    std::uint32_t count = 0;
    for (const auto& step : compiled.execution.steps) {
        if (step.op != SerialOp::attn_hyper_mix && step.op != SerialOp::mlp_hyper_mix && step.op != SerialOp::final_hyper_mix) continue;
        ExecutionDispatchContext context;
        context.execution_identity = compiled.execution_identity;
        context.step_id = step.id;
        context.op = step.op;
        context.resources = &resources;
        const auto& down = manifest_tensor(context, resolve_dispatch_tensor_access(context, 1));
        if (down.ggml_type != 8U) throw std::invalid_argument("typed hyper model contains a non-Q8_0 down projection");
        if (step.op != SerialOp::final_hyper_mix) {
            const auto& inject = manifest_tensor(context, resolve_dispatch_tensor_access(context, 3));
            if (inject.ggml_type != 0U) throw std::invalid_argument("typed hyper model contains a non-F32 injection projection");
        }
        ++count;
    }
    if (count == 0U) throw std::invalid_argument("typed hyper preflight found no mixer bindings");
}

void preflight_q8_hyper_output_model(const CompiledSerialExecution& compiled,
    const ExecutionRuntimeResources& resources, const HipHyperMixLauncherConfig& config) {
    if (compiled.execution_identity == 0 || resources.model_manifest == nullptr || resources.tensor_access == nullptr)
        throw std::invalid_argument("Q8 hyper output preflight requires complete compiled model resources");
    std::uint32_t count = 0;
    for (const auto& step : compiled.execution.steps) {
        if (step.op != SerialOp::attn_hyper_mix && step.op != SerialOp::mlp_hyper_mix && step.op != SerialOp::final_hyper_mix) continue;
        ExecutionDispatchContext context;
        context.execution_identity = compiled.execution_identity;
        context.step_id = step.id;
        context.op = step.op;
        context.resources = &resources;
        const auto& up = manifest_tensor(context, resolve_dispatch_tensor_access(context, 2));
        if (up.ggml_type != 8U || up.dimensions.size() < 2U || up.dimensions[0] == 0U || up.dimensions[0] != config.lowrank || up.dimensions[0] % 32U != 0U ||
            up.dimensions[1] != static_cast<std::uint64_t>(config.hidden_size) * config.hc_count)
            throw std::invalid_argument("Q8 hyper output model contains a non-Q8_0 or invalid-width up projection");
        ++count;
    }
    if (count == 0U) throw std::invalid_argument("Q8 hyper output preflight found no mixer bindings");
}

void preflight_splitk_hyper_projection_model(const CompiledSerialExecution& compiled,
    const ExecutionRuntimeResources& resources, const HipHyperMixLauncherConfig& config, bool validate_device) {
    validate_config(config);
    if (!config.splitk_projection || !compiled.logical_rows || compiled.logical_rows > 65535U ||
        compiled.request_concurrency != 1U || compiled.io.steps.size() != compiled.execution.steps.size())
        throw std::invalid_argument("split-K HC preflight requires an admitted serial compiled plan");
    validate_execution_runtime_resources(compiled.execution_identity, compiled.arena, resources, 0);
    using hc_projection_diagnostic::Span;
    const auto checked = [](Span span, std::uint64_t required, std::uint64_t alignment) {
        if (!span.address || span.address % alignment || span.bytes < required ||
            span.bytes > std::numeric_limits<std::uintptr_t>::max() - span.address)
            throw std::invalid_argument("split-K HC preflight owner extent/alignment differs");
    };
    const auto overlaps = [](Span a, Span b) {
        return a.address && b.address && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
    };
    const std::uint64_t normalized_bytes = static_cast<std::uint64_t>(compiled.logical_rows) * 10240U * sizeof(float);
    const std::uint64_t low_bytes = static_cast<std::uint64_t>(compiled.logical_rows) * config.lowrank * sizeof(float);
    const std::uint64_t down_bytes = static_cast<std::uint64_t>(config.lowrank) * 320U * 34U;
    const std::uint64_t inject_bytes = static_cast<std::uint64_t>(config.hc_count) * 10240U * sizeof(float);
    std::uint32_t count = 0;
    for (const auto& step : compiled.execution.steps) {
        if (step.op != SerialOp::attn_hyper_mix && step.op != SerialOp::mlp_hyper_mix && step.op != SerialOp::final_hyper_mix) continue;
        if (step.id >= compiled.io.steps.size()) throw std::invalid_argument("split-K HC step has no IO binding");
        const auto& io = compiled.io.steps[step.id];
        const bool final = step.op == SerialOp::final_hyper_mix;
        if (io.step_id != step.id || io.op != step.op || io.reads.size() != 1U ||
            io.temporaries.size() != 1U || io.writes.size() != (final ? 1U : 2U))
            throw std::invalid_argument("split-K HC preflight IO topology differs");
        ExecutionDispatchContext context;
        context.execution_identity = compiled.execution_identity;context.step_id = step.id;context.op = step.op;context.layer = step.layer;
        context.logical_rows = compiled.logical_rows;context.request_concurrency = compiled.request_concurrency;
        context.reads = io.reads;context.writes = io.writes;context.temporaries = io.temporaries;
        context.arena = &compiled.arena;context.memory = resources.scratch;context.resources = &resources;
        const auto temporary = resolve_dispatch_temporary(context, 0);
        if (temporary.use.buffer.buffer_class != ExecutionBufferClass::hyper_workspace)
            throw std::invalid_argument("split-K HC preflight requires its hyper workspace");
        checked({temporary.address,temporary.bytes}, normalized_bytes + low_bytes, 16U);
        const Span normalized{temporary.address,normalized_bytes},low{temporary.address + normalized_bytes,low_bytes};
        const auto& down_access = resolve_dispatch_tensor_access(context, 1);
        const auto& down_manifest = manifest_tensor(context, down_access);
        const auto& down_owner = runtime_tensor(context, down_access);
        if (down_manifest.ggml_type != 8U || down_manifest.dimensions != std::vector<std::uint64_t>{10240U,config.lowrank} || down_manifest.byte_size != down_bytes)
            throw std::invalid_argument("split-K HC preflight down tensor differs");
        checked({down_owner.address,down_owner.bytes},down_bytes,16U);
        const Span down{down_owner.address,down_bytes};Span inject{},injection{};
        if (!final) {
            const auto& access = resolve_dispatch_tensor_access(context, 3);
            const auto& manifest = manifest_tensor(context, access);const auto& owner = runtime_tensor(context, access);
            if (manifest.ggml_type != 0U || manifest.dimensions != std::vector<std::uint64_t>{10240U,config.hc_count} || manifest.byte_size != inject_bytes)
                throw std::invalid_argument("split-K HC preflight injection tensor differs");
            checked({owner.address,owner.bytes},inject_bytes,16U);inject = {owner.address,inject_bytes};
            const auto output = resolve_dispatch_write(context, 1);const auto bytes = static_cast<std::uint64_t>(compiled.logical_rows) * config.hc_count * sizeof(float);
            checked({output.address,output.bytes},bytes,4U);injection = {output.address,bytes};
        }
        // Normalization writes the first slice before projection reads it.
        if (overlaps(normalized,down) || overlaps(normalized,inject))
            throw std::invalid_argument("split-K HC normalized workspace aliases weights");
        const std::array<Span,3> inputs{normalized,down,inject};
        for (const auto output : {low,injection}) {
            for (const auto input : inputs) if (overlaps(output,input))
                throw std::invalid_argument("split-K HC projection output aliases input");
        }
        if (overlaps(low,injection)) throw std::invalid_argument("split-K HC projection outputs overlap");
        if (config.splitk_deferred_norm) {
            const auto& norm_access=resolve_dispatch_tensor_access(context,0);
            const auto& norm_manifest=manifest_tensor(context,norm_access);
            const auto& norm_owner=runtime_tensor(context,norm_access);
            const auto& up_access=resolve_dispatch_tensor_access(context,2);
            const auto& up_manifest=manifest_tensor(context,up_access);
            const auto& up_owner=runtime_tensor(context,up_access);
            if(norm_manifest.ggml_type!=0U || norm_manifest.dimensions!=std::vector<std::uint64_t>{10240U} ||
                norm_manifest.byte_size!=40960U || !lane_chain_hyper_output_eligible(up_manifest,
                config.hidden_size,config.hc_count,config.lowrank,up_owner.address%16U))
                throw std::invalid_argument("deferred HC RMS whole-model norm/up contract differs");
            checked({norm_owner.address,norm_owner.bytes},40960U,4U);
            checked({up_owner.address,up_owner.bytes},up_manifest.byte_size,16U);
            const auto raw=resolve_dispatch_read(context,0);
            const auto result=resolve_dispatch_write(context,0);
            checked({raw.address,raw.bytes},20480U,2U);
            checked({result.address,result.bytes},5120U,2U);
            const std::array<Span,5> reads{{{raw.address,raw.bytes},{norm_owner.address,norm_owner.bytes},
                down,{up_owner.address,up_owner.bytes},inject}};
            const std::array<Span,3> writes{{{temporary.address,temporary.bytes},
                {result.address,result.bytes},injection}};
            for(std::size_t i=0;i<writes.size();++i) {
                for(const auto read:reads)if(overlaps(writes[i],read))
                    throw std::invalid_argument("deferred HC RMS whole-model read/write alias");
                for(std::size_t j=i+1;j<writes.size();++j)if(overlaps(writes[i],writes[j]))
                    throw std::invalid_argument("deferred HC RMS whole-model mutable alias");
            }
            if(step.id>=compiled.bindings.bindings.size())
                throw std::invalid_argument("deferred HC RMS missing whole-model binding");
            const auto& binding=compiled.bindings.bindings[step.id];
            if(!binding.kernel_id || binding.kernel_id->find(".hc-deferred-norm-v1")==std::string::npos ||
                !binding.kernel_id->ends_with(lane_chain_hyper_output_suffix))
                throw std::invalid_argument("deferred HC RMS whole-model identity cannot fall back");
        }
        ++count;
    }
    if (!count) throw std::invalid_argument("split-K HC preflight found no mixer bindings");
    if(config.splitk_deferred_norm && (count!=97U || compiled.logical_rows!=1U))
        throw std::invalid_argument("deferred HC RMS requires all 97 canonical mixers");
    // Device queries happen once, after every host owner has been admitted, and
    // before the bundle registers its first launcher or captures any graph.
    if (!validate_device) return;
    int device = 0;hipDeviceProp_t properties{};
    check_hip(hipGetDevice(&device),"split-K HC preflight device");
    check_hip(hipGetDeviceProperties(&properties,device),"split-K HC preflight properties");
    if (properties.warpSize != 32 || std::string_view(properties.gcnArchName,7U) != "gfx1151")
        throw std::invalid_argument("split-K HC integration requires gfx1151 wave32");
}

void HipHyperMixLauncherBundle::
register_for_compiled_execution(
    const CompiledSerialExecution& compiled,
    ExecutionLauncherRegistry& registry,
    const ExecutionRuntimeResources* resources, bool validate_device) {
    if(config_.prefetch_hc_projection_next_tile) {
        validate_config(config_);
        if(!resources || compiled.logical_rows!=1U || compiled.request_concurrency!=1U || compiled.device_graph_inputs)
            throw std::invalid_argument("next-tile HC bundle requires eager R1 C1 resources");
    }
    if (config_.splitk_projection) {
        if(resources && resources->distributed_stage)
            throw std::invalid_argument("capacity stages do not admit experimental split-K HC");
        if (!resources) throw std::invalid_argument("split-K HC bundle requires whole-model preflight resources");
        preflight_splitk_hyper_projection_model(compiled,*resources,config_,validate_device);
    }
    if(config_.lane_chain_output){
        validate_config(config_);
        if(config_.typed_projection){if(!resources)throw std::invalid_argument("typed hyper bundle requires complete model preflight resources");preflight_typed_hyper_projection_model(compiled,*resources);}
        if(config_.prefetch_q8_output){if(!resources)throw std::invalid_argument("Q8 hyper output bundle requires complete model preflight resources");preflight_q8_hyper_output_model(compiled,*resources,config_);}
        if(!compiled.execution_identity || compiled.execution.steps.size()!=compiled.bindings.bindings.size())throw std::invalid_argument("HIP hyper-mix bundle requires valid compiled execution");
        ExecutionBindingRequest expected;expected.logical_rows=compiled.logical_rows;expected.request_concurrency=compiled.request_concurrency;
        const auto candidate_id=kernel_id(expected,config_),base_id=kernel_id(expected,fallback_config_);
        std::set<std::string> selected_ids;
        for(std::size_t index=0;index<compiled.execution.steps.size();++index){
            const auto& step=compiled.execution.steps[index];
            if(step.op!=SerialOp::attn_hyper_mix && step.op!=SerialOp::mlp_hyper_mix && step.op!=SerialOp::final_hyper_mix)continue;
            const auto& binding=compiled.bindings.bindings[index];
            if(binding.host_or_runtime_step || !binding.kernel_id)throw std::invalid_argument("HIP hyper-mix step has no kernel binding");
            if(resources && resources->distributed_stage && !resources->distributed_stage->contains_step(step.id)) {
                // Keep real global IDs registered for the dense source plan,
                // but never resolve a nonlocal absent owner. The peer owning
                // this step still performs the identical physical admission.
                if(*binding.kernel_id!=candidate_id && *binding.kernel_id!=base_id)
                    throw std::invalid_argument("nonlocal HC identity is outside the closed source family");
                selected_ids.insert(*binding.kernel_id);continue;
            }
            bool eligible=false;
            {
                ExecutionDispatchContext context;context.execution_identity=compiled.execution_identity;context.step_id=step.id;context.op=step.op;context.resources=resources;
                const auto& access=resolve_dispatch_tensor_access(context,2);const auto& up=manifest_tensor(context,access);const auto& owner=runtime_tensor(context,access);
                eligible=lane_chain_hyper_output_eligible(up,config_.hidden_size,config_.hc_count,config_.lowrank,owner.address%16U);
                if(owner.bytes<up.byte_size || up.byte_size>std::numeric_limits<std::uintptr_t>::max()-owner.address)throw std::invalid_argument("lane-chain HC output up owner extent differs");
                if(eligible && (owner.address%16U || owner.bytes<up.byte_size || up.byte_size>std::numeric_limits<std::uintptr_t>::max()-owner.address))throw std::invalid_argument("lane-chain HC output up owner admission differs");
            }
            if(config_.splitk_deferred_norm && !eligible)
                throw std::invalid_argument("deferred HC RMS forbids lane-chain fallback");
            const auto& id=!eligible?base_id:candidate_id;
            if(*binding.kernel_id!=id)throw std::invalid_argument("typed hyper bundle differs from selected kernel identity");
            selected_ids.insert(id);
        }
        if(config_.lane_chain_output && !selected_ids.contains(candidate_id))throw std::invalid_argument("lane-chain HC output has no eligible steps");
        for(const auto& id:selected_ids)registry.add_kernel_launcher(make_hip_hyper_mix_launcher(id,id==base_id?&fallback_config_:&config_));

        return;
    }

    if (config_.typed_projection) {
        if (resources == nullptr) throw std::invalid_argument("typed hyper bundle requires complete model preflight resources");
        preflight_typed_hyper_projection_model(compiled, *resources);
    }
    if (config_.prefetch_q8_output) {
        if (resources == nullptr) throw std::invalid_argument("Q8 hyper output bundle requires complete model preflight resources");
        preflight_q8_hyper_output_model(compiled, *resources, config_);
    }
    if (compiled.execution_identity == 0 ||
        compiled.execution.steps.size() !=
            compiled.bindings.bindings.size()) {
        throw std::invalid_argument(
            "HIP hyper-mix bundle requires valid compiled execution");
    }

    // Fail atomically for the typed selection before registering its first id.
    if (config_.typed_projection) {
        ExecutionBindingRequest expected;
        expected.logical_rows = compiled.logical_rows;
        expected.request_concurrency = compiled.request_concurrency;
        const auto selected_id = kernel_id(expected, config_);
        for (std::size_t index = 0; index < compiled.execution.steps.size(); ++index) {
            const auto op = compiled.execution.steps[index].op;
            if (op != SerialOp::attn_hyper_mix && op != SerialOp::mlp_hyper_mix &&
                op != SerialOp::final_hyper_mix) continue;
            const auto& binding = compiled.bindings.bindings[index];
            if (binding.host_or_runtime_step || !binding.kernel_id.has_value() ||
                *binding.kernel_id != selected_id)
                throw std::invalid_argument("typed hyper bundle differs from selected kernel identity");
        }
    }

    std::string registered_id;
    for (std::size_t index = 0;
         index <
            compiled.execution.steps.size();
         ++index) {
        const auto op =
            compiled.execution.steps[index].op;
        if (op !=
                SerialOp::attn_hyper_mix &&
            op !=
                SerialOp::mlp_hyper_mix &&
            op !=
                SerialOp::final_hyper_mix) {
            continue;
        }

        const auto& binding =
            compiled.bindings.bindings[index];
        if (binding.host_or_runtime_step ||
            !binding.kernel_id.has_value() ||
            binding.kernel_id->empty()) {
            throw std::invalid_argument(
                "HIP hyper-mix step has no kernel binding");
        }

        if (registered_id.empty()) {
            registered_id =
                *binding.kernel_id;
            registry.add_kernel_launcher(
                make_hip_hyper_mix_launcher(
                    registered_id,
                    &config_));
        } else if (registered_id !=
                   *binding.kernel_id) {
            throw std::invalid_argument(
                "HIP hyper-mix bundle requires one selected kernel id for all mixer steps");
        }
    }
}

}  // namespace strix

namespace strix {
namespace {
struct MtpHyperPrepared {
    MtpDeviceView input,output,injection,workspace;
    MtpTensorView norm,down,up,inject;
    unsigned h,c,rank;
};
void mtp_hyper_range(MtpDeviceView v,std::uint64_t required,unsigned align) {
    if(!v.address || !v.allocation_generation || v.bytes<required || v.address%align ||
       v.bytes>std::numeric_limits<std::uintptr_t>::max()-v.address)
        throw std::invalid_argument("MTP HC device view is invalid");
}
bool mtp_hyper_overlap(MtpDeviceView a,MtpDeviceView b){return a.address<b.address+b.bytes && b.address<a.address+a.bytes;}
MtpDeviceView mtp_hyper_tensor(MtpTensorView t,unsigned type,std::vector<std::uint64_t> dims,std::uint64_t bytes) {
    if(!t.manifest || !t.runtime || t.manifest->ggml_type!=type || t.manifest->dimensions!=dims ||
       t.manifest->byte_size!=bytes || !runtime_tensor_gpu_visible(t.runtime->storage))
        throw std::invalid_argument("MTP HC tensor format or geometry is invalid");
    MtpDeviceView v{t.runtime->address,t.runtime->bytes,t.runtime->allocation_generation};mtp_hyper_range(v,bytes,16U);return v;
}
MtpHyperPrepared mtp_hyper_prepare(const MtpNativeContext& c,MtpHyperRole role) {
    if(!c.model || !c.plan || !c.tensors || !c.resources ||
       (role!=MtpHyperRole::attention && role!=MtpHyperRole::mlp && role!=MtpHyperRole::head))
        throw std::invalid_argument("MTP HC context or role is invalid");
    c.resources->validate_stage();
    if(c.model!=&c.resources->model() || c.plan!=&c.resources->plan() || c.tensors!=&c.resources->tensors() ||
       c.position>=c.resources->context_capacity() || c.model->hidden_size!=2560U || c.model->hc_count!=4U || c.model->hc_lowrank!=320U)
        throw std::invalid_argument("MTP HC context is not bound to its actual canonical owner");
    const auto& t=*c.tensors;const auto& owner=*c.resources;
    MtpHyperPrepared p{};p.h=2560U;p.c=4U;p.rank=320U;
    if(role==MtpHyperRole::attention){p.input=owner.view(HipMtpBuffer::fused_wide);p.norm=t.attention_hc_norm;p.down=t.attention_hc_down;p.up=t.attention_hc_up;p.inject=t.attention_hc_inject;}
    else if(role==MtpHyperRole::mlp){p.input=owner.view(HipMtpBuffer::wide_work);p.norm=t.mlp_hc_norm;p.down=t.mlp_hc_down;p.up=t.mlp_hc_up;p.inject=t.mlp_hc_inject;}
    else {p.input=owner.view(HipMtpBuffer::wide_current);p.norm=t.head_norm;p.down=t.head_down;p.up=t.head_up;}
    p.output=owner.view(HipMtpBuffer::hidden0);p.workspace=owner.view(HipMtpBuffer::hyper_workspace);
    if(role!=MtpHyperRole::head)p.injection=owner.view(HipMtpBuffer::injection);
    const std::uint64_t width=p.h*p.c;
    const auto norm=mtp_hyper_tensor(p.norm,0U,{width},width*4U);
    const auto down=mtp_hyper_tensor(p.down,8U,{width,p.rank},width/32U*34U*p.rank);
    const auto up=mtp_hyper_tensor(p.up,8U,{p.rank,width},p.rank/32U*34U*width);
    mtp_hyper_range(p.input,width*2U,2U);mtp_hyper_range(p.output,p.h*2U,2U);mtp_hyper_range(p.workspace,(width+p.rank)*4U,4U);
    std::array<MtpDeviceView,4> weights{norm,down,up,{}};std::size_t count=3;
    if(role!=MtpHyperRole::head){weights[3]=mtp_hyper_tensor(p.inject,8U,{width,p.c},width/32U*34U*p.c);count=4;mtp_hyper_range(p.injection,p.c*4U,4U);}
    const std::array writes{p.output,p.workspace,p.injection};
    for(std::size_t i=0;i<writes.size();++i)if(writes[i].bytes){
        if(mtp_hyper_overlap(writes[i],p.input))throw std::invalid_argument("MTP HC output aliases input");
        for(std::size_t j=0;j<count;++j)if(mtp_hyper_overlap(writes[i],weights[j]))throw std::invalid_argument("MTP HC writable view aliases weights");
        for(std::size_t j=0;j<i;++j)if(writes[j].bytes && mtp_hyper_overlap(writes[i],writes[j]))throw std::invalid_argument("MTP HC writable owners overlap");
    }
    return p;
}
}
void preflight_hip_mtp_hyper_mix(const MtpNativeContext& c,MtpHyperRole role){(void)mtp_hyper_prepare(c,role);}
void launch_hip_mtp_hyper_mix(const MtpNativeContext& c,MtpHyperRole role,void* raw_stream) {
    const auto p=mtp_hyper_prepare(c,role);
    if(!raw_stream || raw_stream!=c.resources->stream())throw std::invalid_argument("MTP HC stream differs from retained owner");
    const auto stream=static_cast<hipStream_t>(raw_stream);hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
    check_hip(hipStreamIsCapturing(stream,&capture),"MTP HC capture query");
    if(capture!=hipStreamCaptureStatusNone)throw std::invalid_argument("MTP HC is eager only");
    check_hip(hipPeekAtLastError(),"MTP HC prior HIP error");
    const bool inject=role!=MtpHyperRole::head;
    enqueue_hip_mtp_hyper_mix_q8(reinterpret_cast<const std::uint16_t*>(p.input.address),reinterpret_cast<const float*>(p.norm.runtime->address),
        reinterpret_cast<const void*>(p.down.runtime->address),inject?reinterpret_cast<const void*>(p.inject.runtime->address):nullptr,
        reinterpret_cast<const void*>(p.up.runtime->address),reinterpret_cast<float*>(p.workspace.address),
        inject?reinterpret_cast<float*>(p.injection.address):nullptr,reinterpret_cast<std::uint16_t*>(p.output.address),
        c.model->rms_norm_epsilon,false,stream);
}
void launch_hip_mtp_hyper_mix_draft(const MtpNativeContext& c,MtpHyperRole role,const float* inject_f32,void* raw_stream) {
    const auto p=mtp_hyper_prepare(c,role);
    if(!raw_stream || raw_stream!=c.resources->stream())throw std::invalid_argument("MTP HC stream differs from retained owner");
    const bool inject=role!=MtpHyperRole::head;
    if(inject && !inject_f32)throw std::invalid_argument("MTP draft HC injection needs its F32 [4][10240] copy");
    check_hip(hipPeekAtLastError(),"MTP draft HC prior HIP error");
    enqueue_hip_mtp_hyper_mix_q8(reinterpret_cast<const std::uint16_t*>(p.input.address),reinterpret_cast<const float*>(p.norm.runtime->address),
        reinterpret_cast<const void*>(p.down.runtime->address),inject?inject_f32:nullptr,reinterpret_cast<const void*>(p.up.runtime->address),
        reinterpret_cast<float*>(p.workspace.address),inject?reinterpret_cast<float*>(p.injection.address):nullptr,
        reinterpret_cast<std::uint16_t*>(p.output.address),c.model->rms_norm_epsilon,true,raw_stream);
}
// Multi-session draft batch: launch_hip_mtp_hyper_mix_draft of every context as one HC rows mix on the batch
// stream. The one-token draft runs the normalisation, the <2,HyperRows> split-K projection with count 1 and the
// serial lane-chain output; here the rows forms run them with count T: per row the same leaves, tree, rsqrtf
// and product (rows_prepare_hyper_norm_kernel), the same split-K chains and ordered tile sum, and the same
// lane-chain coefficients, gate expression, stream mix and BF16 store (rows_lane_chain_hyper_output_kernel,
// Count 0). Only the production branch is batched (rows kernels on, persistent HC off, 16-aligned workspaces).
bool hip_mtp_hyper_mix_draft_rows_supported() {
    static const bool rows_kernels=[]{const char* v=std::getenv("STRIX_MTP_HC_ROWS");return !v || v[0]!='0';}();
    static const bool persistent=[]{const char* v=std::getenv("STRIX_MTP_HC_PERSISTENT");return v && v[0]=='1';}();
    return rows_kernels && !persistent;
}
void enqueue_hip_mtp_hyper_mix_q8_rows(const HipMtpHyperDraftRows& views,std::uint32_t count,const float* norm,const void* down,
    const float* inject,const void* up,float epsilon,void* raw_stream) {
    constexpr unsigned h=2560U,streams=4U,rank=320U;
    if(!count || count>kWindowRows || !raw_stream || !norm || !down || !up || !hip_mtp_hyper_mix_draft_rows_supported() ||
       (inject && reinterpret_cast<std::uintptr_t>(inject)%16U))
        throw std::invalid_argument("MTP draft HC batch views, size, stream or configuration are invalid");
    HyperRows rows{};
    for(std::uint32_t t=0;t<count;++t) {
        if(!views.input[t] || !views.output[t] || !views.workspace[t] || reinterpret_cast<std::uintptr_t>(views.workspace[t])%16U ||
           (inject==nullptr)!=(views.injection[t]==nullptr))
            throw std::invalid_argument("MTP draft HC batch row views are incomplete or misaligned");
        rows.input[t]=views.input[t];rows.output[t]=views.output[t];
        rows.normalized[t]=views.workspace[t];rows.low[t]=views.workspace[t]+h*streams;rows.injection[t]=views.injection[t];
    }
    const auto stream=static_cast<hipStream_t>(raw_stream);
    check_hip(hipPeekAtLastError(),"MTP draft HC batch prior HIP error");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_prepare_hyper_norm_kernel<std::uint16_t,HyperRows>),dim3(streams,count),dim3(kThreads),0,stream,
        rows,static_cast<const void*>(norm),0U,h,epsilon);
    check_hip(hipGetLastError(),"MTP draft HC batch normalization");
    constexpr unsigned outputs=2U,blocks=(rank+outputs-1U)/outputs;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_splitk_hyper_projection_kernel<outputs,HyperRows>),
        dim3(blocks+(inject?(streams+outputs-1U)/outputs:0U)),dim3(512U),0,stream,rows,count,down,streams,rank,
        static_cast<const void*>(inject),blocks);
    check_hip(hipGetLastError(),"MTP draft HC batch split-K projection");
    const auto lds=lane_chain_hyper_output_lds(streams,rank)+static_cast<std::uint64_t>(count-1U)*streams*32U*sizeof(float);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_lane_chain_hyper_output_kernel<std::uint16_t,0U,HyperRows>),dim3(h/32U),dim3(32U*streams),
        static_cast<std::size_t>(lds),stream,rows,count,up,h,streams,rank);
    check_hip(hipGetLastError(),"MTP draft HC batch lane-chain output");
}
void launch_hip_mtp_hyper_mix_draft_rows(std::span<const MtpNativeContext> contexts,MtpHyperRole role,const float* inject_f32,
    void* raw_stream) {
    if(contexts.empty() || contexts.size()>kWindowRows) throw std::invalid_argument("MTP draft HC batch size is invalid");
    const bool inject=role!=MtpHyperRole::head;
    if(inject && !inject_f32) throw std::invalid_argument("MTP draft HC batch injection needs its F32 [4][10240] copy");
    HipMtpHyperDraftRows views{};const void* norm=nullptr;const void* down=nullptr;const void* up=nullptr;float epsilon=0.0F;
    for(std::size_t t=0;t<contexts.size();++t) {
        const auto& c=contexts[t];const auto p=mtp_hyper_prepare(c,role);
        const void* n=reinterpret_cast<const void*>(p.norm.runtime->address);
        const void* d=reinterpret_cast<const void*>(p.down.runtime->address);const void* u=reinterpret_cast<const void*>(p.up.runtime->address);
        if(!t){norm=n;down=d;up=u;epsilon=c.model->rms_norm_epsilon;}
        else if(n!=norm || d!=down || u!=up || c.model->rms_norm_epsilon!=epsilon)
            throw std::invalid_argument("MTP draft HC batch members do not share their weights");
        views.input[t]=reinterpret_cast<const std::uint16_t*>(p.input.address);views.output[t]=reinterpret_cast<std::uint16_t*>(p.output.address);
        views.workspace[t]=reinterpret_cast<float*>(p.workspace.address);
        views.injection[t]=inject?reinterpret_cast<float*>(p.injection.address):nullptr;
    }
    enqueue_hip_mtp_hyper_mix_q8_rows(views,static_cast<std::uint32_t>(contexts.size()),static_cast<const float*>(norm),down,
        inject?inject_f32:nullptr,up,epsilon,raw_stream);
}
void enqueue_hip_mtp_hyper_mix_q8(const std::uint16_t* input,const float* norm,const void* down,const void* inject,const void* up,
    float* workspace,float* injection,std::uint16_t* output,float epsilon,bool draft,void* raw_stream) {
    constexpr unsigned h=2560U,streams=4U,rank=320U;
    if(!input || !norm || !down || !up || !workspace || !output || !raw_stream || (inject==nullptr)!=(injection==nullptr) ||
       (draft && inject && reinterpret_cast<std::uintptr_t>(inject)%16U))
        throw std::invalid_argument("MTP HC raw views are incomplete");
    const auto stream=static_cast<hipStream_t>(raw_stream);
    auto* normalized=workspace;auto* low=normalized+h*streams;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(prepare_hyper_norm_kernel<std::uint16_t>),dim3(streams),dim3(kThreads),0,stream,
        input,static_cast<const void*>(norm),0U,h,streams,epsilon,normalized);check_hip(hipGetLastError(),"MTP HC normalization");
    // STRIX_MTP_HC_ROWS=0: the serial split-K kernel instead of its one-row window form (two
    // outputs per block, twice the blocks). The window kernel reproduces the serial kernel per
    // row bit for bit (the known-window fixtures check this); outputs per block only regroup
    // independent chains. The serial lane-chain output stays: its one-row register form is slower.
    static const bool rows_kernels=[]{const char* v=std::getenv("STRIX_MTP_HC_ROWS");return !v || v[0]!='0';}();
    if(draft && rows_kernels && reinterpret_cast<std::uintptr_t>(normalized)%16U==0U) {
        if(launch_persistent_hyper_draft(input,norm,down,static_cast<const float*>(inject),up,normalized,low,injection,output,epsilon,stream))
            return;
        HyperRows rows{};
        rows.input[0]=input;rows.output[0]=output;rows.normalized[0]=normalized;rows.low[0]=low;rows.injection[0]=injection;
        constexpr unsigned outputs=2U,blocks=(rank+outputs-1U)/outputs;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_splitk_hyper_projection_kernel<outputs,HyperRows>),
            dim3(blocks+(inject?(streams+outputs-1U)/outputs:0U)),dim3(512U),0,stream,rows,1U,down,streams,rank,inject,blocks);
        check_hip(hipGetLastError(),"MTP draft HC split-K projection rows");
        hipLaunchKernelGGL(HIP_KERNEL_NAME(lane_chain_hyper_output_kernel<std::uint16_t>),dim3(h/32U),dim3(32U*streams),
            static_cast<std::size_t>(lane_chain_hyper_output_lds(streams,rank)),stream,normalized,low,up,h,streams,rank,output);
        check_hip(hipGetLastError(),"MTP draft HC lane-chain output");
        return;
    }
    if(draft) {
        // Target decode's FP32 split-K projection and lane-chain output (Q8 down/up, F32 injection).
        constexpr unsigned outputs=4U,blocks=rank/outputs;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(splitk_hyper_projection_kernel<outputs,float>),dim3(blocks+(inject?1U:0U)),dim3(512U),0,stream,
            normalized,down,h*streams,streams,rank,low,inject,injection,blocks);
        check_hip(hipGetLastError(),"MTP draft HC split-K projection");
        hipLaunchKernelGGL(HIP_KERNEL_NAME(lane_chain_hyper_output_kernel<std::uint16_t>),dim3(h/32U),dim3(32U*streams),
            static_cast<std::size_t>(lane_chain_hyper_output_lds(streams,rank)),stream,normalized,low,up,h,streams,rank,output);
        check_hip(hipGetLastError(),"MTP draft HC lane-chain output");
        return;
    }
    const unsigned low_blocks=(rank+kStagedOutputs-1U)/kStagedOutputs;
    hipLaunchKernelGGL(fused_staged_hyper_projection_kernel,dim3(low_blocks+(inject?1U:0U)),dim3(kStagedThreads),0,stream,
        normalized,down,8U,h*streams,streams,rank,low,inject,8U,injection,low_blocks);check_hip(hipGetLastError(),"MTP HC Q8 down/injection");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(parallel_hyper_output_kernel<std::uint16_t>),dim3((h+255U)/256U),dim3(256U),0,stream,
        normalized,low,up,8U,h,streams,rank,output);check_hip(hipGetLastError(),"MTP HC Q8 output");
}
} // namespace strix
