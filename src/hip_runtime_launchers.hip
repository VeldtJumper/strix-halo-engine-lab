#include "strix/hip_hc_wide_rows_tests.hpp"
#include "strix/hip_mtp_native_validation.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_runtime_launchers.hpp"

#include "strix/dispatch_resources.hpp"
#include "strix/rows_launch_registry.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <span>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace strix {
namespace {

constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

constexpr std::uint64_t hyper_expand_tag =
    0x485950455850414eULL;
constexpr std::uint64_t inject_tag =
    0x494e4a4543540001ULL;
constexpr std::uint64_t moe_combine_tag =
    0x4d4f45434f4d4201ULL;

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

std::uint64_t checked_mul(
    std::uint64_t left,
    std::uint64_t right) {
    if (left != 0 &&
        right >
            std::numeric_limits<std::uint64_t>::max() /
                left) {
        throw std::overflow_error(
            "HIP elementwise launcher size overflow");
    }
    return left * right;
}

std::uint64_t activation_bytes(
    const HipElementwiseLauncherConfig& config) {
    switch (config.activation) {
    case HipActivationStorage::bf16:
        return 2;
    case HipActivationStorage::fp32:
        return 4;
    }
    throw std::invalid_argument(
        "unsupported HIP activation storage");
}

const HipElementwiseLauncherConfig&
require_config(void* user_data) {
    if (user_data == nullptr) {
        throw std::invalid_argument(
            "HIP elementwise launcher requires configuration");
    }
    const auto& config =
        *static_cast<HipElementwiseLauncherConfig*>(
            user_data);
    if (config.hidden_size == 0 ||
        config.hc_count == 0) {
        throw std::invalid_argument(
            "HIP elementwise launcher geometry must be non-zero");
    }
    static_cast<void>(
        activation_bytes(config));
    return config;
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

__device__ float load_activation(
    const float* value) {
    return *value;
}

__device__ float load_activation(
    const std::uint16_t* value) {
    return __uint_as_float(
        static_cast<std::uint32_t>(*value) << 16U);
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
// Serial and window-row kernels share these per-element bodies, operand-pinned (hip_qpf2k_pinned.inc) so the
// fused forms that store the same values reproduce them for NaN inputs too.
template <typename Storage>
__device__ __forceinline__ void inject_element(const Storage* hyper, const Storage* hidden, float weight, Storage* output) {
    store_activation(output, qpf2k_fmac(load_activation(hyper), weight, load_activation(hidden)));
}
template <typename Storage>
__device__ __forceinline__ void combine_element(const Storage* shared, const Storage* routed, Storage* output) {
    store_activation(output, qpf2k_add(load_activation(shared), load_activation(routed)));
}
// Windows of 2/4/8 rows; cohorts of several sessions' windows up to 32 rows (inject cohort launches).
constexpr std::uint32_t kElementwiseMaxRows = 32;
struct ElementwiseRowsArgs {
    const void* first[kElementwiseMaxRows];
    const void* second[kElementwiseMaxRows];
    const float* weights[kElementwiseMaxRows];
    void* output[kElementwiseMaxRows];
    // Prefill tile strides (read only by ElementwiseTileRowsArgs kernels).
    std::uint64_t tile_first, tile_second, tile_weights, tile_output;
};
// Prefill: blockIdx.z selects the 8-row tile; window kernels compile without tile arithmetic.
struct ElementwiseTileRowsArgs : ElementwiseRowsArgs {};
template <typename Pointer> __device__ __forceinline__ Pointer elementwise_tiled(Pointer p, std::uint64_t, const ElementwiseRowsArgs&) {
    return p;
}
template <typename Pointer>
__device__ __forceinline__ Pointer elementwise_tiled(Pointer p, std::uint64_t stride, const ElementwiseTileRowsArgs&) {
    return reinterpret_cast<Pointer>(reinterpret_cast<std::uintptr_t>(p) + static_cast<std::uint64_t>(blockIdx.z) * stride);
}
// blockIdx.y selects the window row; each row uses its own resolved buffers.
template <typename Storage, class Args = ElementwiseRowsArgs>
__global__ void inject_rows_kernel(Args rows, std::uint32_t hidden_size, std::uint32_t hc_count) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<std::uint64_t>(hidden_size) * hc_count) return;
    const auto r = blockIdx.y;
    inject_element(static_cast<const Storage*>(elementwise_tiled(rows.first[r], rows.tile_first, rows)) + index,
        static_cast<const Storage*>(elementwise_tiled(rows.second[r], rows.tile_second, rows)) + index % hidden_size,
        elementwise_tiled(rows.weights[r], rows.tile_weights, rows)[index / hidden_size],
        static_cast<Storage*>(elementwise_tiled(rows.output[r], rows.tile_output, rows)) + index);
}
template <typename Storage, class Args = ElementwiseRowsArgs>
__global__ void combine_rows_kernel(Args rows, std::uint32_t hidden_size) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= hidden_size) return;
    const auto r = blockIdx.y;
    combine_element(static_cast<const Storage*>(elementwise_tiled(rows.first[r], rows.tile_first, rows)) + index,
        static_cast<const Storage*>(elementwise_tiled(rows.second[r], rows.tile_second, rows)) + index,
        static_cast<Storage*>(elementwise_tiled(rows.output[r], rows.tile_output, rows)) + index);
}

template <typename Storage>
__global__ void hyper_expand_kernel(
    const Storage* input,
    Storage* output,
    std::uint32_t hidden_size,
    std::uint32_t hc_count,
    std::uint32_t rows) {
    const auto index =
        static_cast<std::uint64_t>(blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    const auto expanded_width =
        static_cast<std::uint64_t>(
            hidden_size) *
        hc_count;
    const auto total =
        expanded_width * rows;
    if (index >= total) {
        return;
    }

    const auto row =
        index / expanded_width;
    const auto within =
        index % expanded_width;
    const auto feature =
        within %
        static_cast<std::uint64_t>(
            hidden_size);

    const auto source =
        row *
            static_cast<std::uint64_t>(
                hidden_size) +
        feature;

    store_activation(
        &output[index],
        load_activation(
            &input[source]));
}

// Rows of a tiled prefill block in one launch (STRIX_PREFILL_PIPE bit 16): blockIdx.y selects the row, whose input
// and output start one pitch after the previous row's; each element is hyper_expand_kernel's for one row.
template <typename Storage>
__global__ void hyper_expand_pitched_rows_kernel(const Storage* input, std::uint64_t input_pitch, Storage* output,
    std::uint64_t output_pitch, std::uint32_t hidden_size, std::uint32_t hc_count) {
    const auto within = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (within >= static_cast<std::uint64_t>(hidden_size) * hc_count) return;
    const auto row = static_cast<std::uint64_t>(blockIdx.y);
    store_activation(&output[row * output_pitch + within],
        load_activation(&input[row * input_pitch + within % static_cast<std::uint64_t>(hidden_size)]));
}

template <typename Storage>
__global__ void inject_kernel(
    const Storage* hyper_input,
    const Storage* hidden,
    const float* injection_weights,
    Storage* output,
    std::uint32_t hidden_size,
    std::uint32_t hc_count,
    std::uint32_t rows) {
    const auto index =
        static_cast<std::uint64_t>(blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    const auto expanded_width =
        static_cast<std::uint64_t>(
            hidden_size) *
        hc_count;
    const auto total =
        expanded_width * rows;
    if (index >= total) {
        return;
    }

    const auto row =
        index / expanded_width;
    const auto within =
        index % expanded_width;
    const auto stream =
        within /
        static_cast<std::uint64_t>(
            hidden_size);
    const auto feature =
        within %
        static_cast<std::uint64_t>(
            hidden_size);

    const auto hidden_index =
        row *
            static_cast<std::uint64_t>(
                hidden_size) +
        feature;
    const auto weight_index =
        row *
            static_cast<std::uint64_t>(
                hc_count) +
        stream;

    inject_element(&hyper_input[index], &hidden[hidden_index], injection_weights[weight_index], &output[index]);
}

// Pitched rows form of inject_kernel (STRIX_PREFILL_PIPE bit 16): blockIdx.y selects the row; each buffer of a row
// starts one pitch (in elements of that buffer) after the previous row's.
template <typename Storage>
__global__ void inject_pitched_rows_kernel(const Storage* hyper_input, std::uint64_t hyper_pitch, const Storage* hidden,
    std::uint64_t hidden_pitch, const float* injection_weights, std::uint64_t weight_pitch, Storage* output,
    std::uint64_t output_pitch, std::uint32_t hidden_size, std::uint32_t hc_count) {
    const auto within = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (within >= static_cast<std::uint64_t>(hidden_size) * hc_count) return;
    const auto row = static_cast<std::uint64_t>(blockIdx.y);
    const auto stream = within / static_cast<std::uint64_t>(hidden_size);
    const auto feature = within % static_cast<std::uint64_t>(hidden_size);
    inject_element(&hyper_input[row * hyper_pitch + within], &hidden[row * hidden_pitch + feature],
        injection_weights[row * weight_pitch + stream], &output[row * output_pitch + within]);
}

// Multi-session draft batch: row blockIdx.y is one session's one-row inject (same element function).
struct MtpInjectRows {
    const std::uint16_t* hyper[8]; const std::uint16_t* hidden[8]; const float* weights[8]; std::uint16_t* output[8];
};
__global__ void mtp_inject_rows_kernel(MtpInjectRows rows, std::uint32_t hidden_size, std::uint32_t hc_count) {
    const auto r = blockIdx.y;
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto expanded_width = static_cast<std::uint64_t>(hidden_size) * hc_count;
    if (index >= expanded_width) return;
    const auto stream = index / static_cast<std::uint64_t>(hidden_size);
    const auto feature = index % static_cast<std::uint64_t>(hidden_size);
    inject_element(&rows.hyper[r][index], &rows.hidden[r][feature], rows.weights[r][stream], &rows.output[r][index]);
}
// Multi-session draft batch: the publish_next_state copy (wide_next -> wide_current) of every row, 16 bytes per lane.
struct MtpCopyRows { const uint4* source[8]; uint4* destination[8]; };
__global__ void mtp_copy_rows_kernel(MtpCopyRows rows, std::uint64_t vectors) {
    const auto r = blockIdx.y;
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < vectors;
         i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
        rows.destination[r][i] = rows.source[r][i];
}

template <typename Storage>
__global__ void moe_combine_kernel(
    const Storage* shared,
    const Storage* routed,
    Storage* output,
    std::uint64_t elements) {
    const auto index =
        static_cast<std::uint64_t>(blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    if (index >= elements) {
        return;
    }
    combine_element(&shared[index], &routed[index], &output[index]);
}

std::uint32_t prepare_1d_blocks(std::uint64_t elements) {
    constexpr std::uint32_t threads = 256;
    if (elements == 0) {
        throw std::invalid_argument(
            "HIP elementwise launch requires non-zero elements");
    }
    const auto blocks64 =
        (elements + threads - 1ULL) /
        threads;
    if (blocks64 >
        std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
            "HIP elementwise grid exceeds uint32 block count");
    }
    return static_cast<std::uint32_t>(blocks64);
}

template <typename Kernel>
void launch_1d(
    std::uint64_t elements,
    Kernel&& launch) {
    launch(prepare_1d_blocks(elements), 256U);
    check_hip(
        hipGetLastError(),
        "HIP elementwise kernel launch");
}

void validate_buffer(
    const ExecutionBufferAddress& buffer,
    std::uint64_t required,
    const char* name) {
    if (buffer.address == 0 ||
        buffer.bytes < required) {
        throw std::invalid_argument(
            std::string(name) +
            " buffer is missing or undersized");
    }
}

template <typename Storage>
void launch_hyper_expand_typed(
    const ExecutionDispatchContext& context,
    const HipElementwiseLauncherConfig& config,
    const ExecutionBufferAddress& input,
    const ExecutionBufferAddress& output) {
    const auto elements =
        checked_mul(
            checked_mul(
                context.logical_rows,
                config.hidden_size),
            config.hc_count);
    launch_1d(
        elements,
        [&](std::uint32_t blocks,
            std::uint32_t threads) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    hyper_expand_kernel<Storage>),
                dim3(blocks),
                dim3(threads),
                0,
                hip_execution_stream(),
                reinterpret_cast<const Storage*>(
                    input.address),
                reinterpret_cast<Storage*>(
                    output.address),
                config.hidden_size,
                config.hc_count,
                context.logical_rows);
        });
}

struct PreparedHyperExpand {
    const HipElementwiseLauncherConfig& config;
    ExecutionBufferAddress input, output;
};
PreparedHyperExpand prepare_hyper_expand(
    const ExecutionDispatchContext& context,
    void* user_data) {
    if (context.op !=
        SerialOp::hyper_expand) {
        throw std::invalid_argument(
            "HIP hyper-expand launcher attached to wrong operator");
    }
    if (context.reads.size() != 1 ||
        context.writes.size() != 1) {
        throw std::invalid_argument(
            "HIP hyper-expand expects one read and one write buffer");
    }

    const auto& config =
        require_config(user_data);
    const auto input =
        resolve_dispatch_read(
            context, 0);
    const auto output =
        resolve_dispatch_write(
            context, 0);
    const auto element_bytes =
        activation_bytes(config);

    const auto input_required =
        checked_mul(
            checked_mul(
                context.logical_rows,
                config.hidden_size),
            element_bytes);
    const auto output_required =
        checked_mul(
            checked_mul(
                checked_mul(
                    context.logical_rows,
                    config.hidden_size),
                config.hc_count),
            element_bytes);
    validate_buffer(
        input,
        input_required,
        "hyper-expand input");
    validate_buffer(
        output,
        output_required,
        "hyper-expand output");

    static_cast<void>(prepare_1d_blocks(checked_mul(
        checked_mul(context.logical_rows, config.hidden_size), config.hc_count)));
    return {config, input, output};
}

void preflight_hyper_expand(const ExecutionDispatchContext& context, void* user_data) {
    if (const auto* scope = current_hip_launch_context(); scope && scope->graph_capture)
        throw std::invalid_argument("known-window launcher preflight requires eager scope");
    static_cast<void>(prepare_hyper_expand(context, user_data));
}

void launch_hyper_expand(
    const ExecutionDispatchContext& context,
    void* user_data) {
    const auto prepared = prepare_hyper_expand(context, user_data);
    const auto& config = prepared.config;
    const auto& input = prepared.input;
    const auto& output = prepared.output;

    switch (config.activation) {
    case HipActivationStorage::bf16:
        launch_hyper_expand_typed<
            std::uint16_t>(
                context,
                config,
                input,
                output);
        return;
    case HipActivationStorage::fp32:
        launch_hyper_expand_typed<float>(
            context,
            config,
            input,
            output);
        return;
    }
    throw std::logic_error(
        "unreachable activation storage");
}

// Every row is admitted as launch_hyper_expand admits it; false (nothing enqueued) unless every row is one row
// and the inputs and outputs each sit at one pitch.
bool launch_hyper_expand_rows(std::span<const ExecutionDispatchContext> rows, void* user_data) {
    if (rows.size() < 2U || rows.size() > 65535U) return false;
    if (const auto* scope = current_hip_launch_context(); scope && scope->graph_capture) return false;
    const auto first = prepare_hyper_expand(rows[0], user_data);
    const auto second = prepare_hyper_expand(rows[1], user_data);
    const auto element = activation_bytes(first.config);
    const auto expanded = static_cast<std::uint64_t>(first.config.hidden_size) * first.config.hc_count;
    if (rows[0].logical_rows != 1U || second.input.address <= first.input.address || second.output.address <= first.output.address)
        return false;
    const auto input_pitch = static_cast<std::uint64_t>(second.input.address - first.input.address);
    const auto output_pitch = static_cast<std::uint64_t>(second.output.address - first.output.address);
    if (input_pitch % element || output_pitch % element || input_pitch < first.config.hidden_size * element ||
        output_pitch < expanded * element)
        return false;
    for (std::size_t r = 1; r < rows.size(); ++r) {
        const auto row = r == 1U ? second : prepare_hyper_expand(rows[r], user_data);
        if (rows[r].logical_rows != 1U || &row.config != &first.config ||
            row.input.address != first.input.address + r * input_pitch || row.output.address != first.output.address + r * output_pitch)
            return false;
    }
    const auto blocks = prepare_1d_blocks(expanded);
    const auto count = static_cast<std::uint32_t>(rows.size());
    switch (first.config.activation) {
    case HipActivationStorage::bf16:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(hyper_expand_pitched_rows_kernel<std::uint16_t>), dim3(blocks, count), dim3(256U), 0,
            hip_execution_stream(), reinterpret_cast<const std::uint16_t*>(first.input.address), input_pitch / element,
            reinterpret_cast<std::uint16_t*>(first.output.address), output_pitch / element, first.config.hidden_size, first.config.hc_count);
        break;
    case HipActivationStorage::fp32:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(hyper_expand_pitched_rows_kernel<float>), dim3(blocks, count), dim3(256U), 0,
            hip_execution_stream(), reinterpret_cast<const float*>(first.input.address), input_pitch / element,
            reinterpret_cast<float*>(first.output.address), output_pitch / element, first.config.hidden_size, first.config.hc_count);
        break;
    default:
        throw std::logic_error("unreachable activation storage");
    }
    check_hip(hipGetLastError(), "HIP hyper-expand rows kernel launch");
    return true;
}
[[maybe_unused]] const bool hyper_expand_rows_registered = register_rows_launch(&launch_hyper_expand, &launch_hyper_expand_rows);

template <typename Storage>
void launch_inject_typed(
    const ExecutionDispatchContext& context,
    const HipElementwiseLauncherConfig& config,
    const ExecutionBufferAddress& hyper,
    const ExecutionBufferAddress& hidden,
    const ExecutionBufferAddress& weights,
    const ExecutionBufferAddress& output) {
    const auto elements =
        checked_mul(
            checked_mul(
                context.logical_rows,
                config.hidden_size),
            config.hc_count);
    launch_1d(
        elements,
        [&](std::uint32_t blocks,
            std::uint32_t threads) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    inject_kernel<Storage>),
                dim3(blocks),
                dim3(threads),
                0,
                hip_execution_stream(),
                reinterpret_cast<const Storage*>(
                    hyper.address),
                reinterpret_cast<const Storage*>(
                    hidden.address),
                reinterpret_cast<const float*>(
                    weights.address),
                reinterpret_cast<Storage*>(
                    output.address),
                config.hidden_size,
                config.hc_count,
                context.logical_rows);
        });
}

struct PreparedInject {
    const HipElementwiseLauncherConfig& config;
    ExecutionBufferAddress hyper, hidden, weights, output;
};
PreparedInject prepare_inject(
    const ExecutionDispatchContext& context,
    void* user_data) {
    if (context.op != SerialOp::attn_inject &&
        context.op != SerialOp::mlp_inject) {
        throw std::invalid_argument(
            "HIP inject launcher attached to wrong operator");
    }
    if (context.reads.size() != 3 ||
        context.writes.size() != 1) {
        throw std::invalid_argument(
            "HIP inject expects three reads and one write");
    }

    const auto& config =
        require_config(user_data);
    const auto hyper =
        resolve_dispatch_read(
            context, 0);
    const auto hidden =
        resolve_dispatch_read(
            context, 1);
    const auto weights =
        resolve_dispatch_read(
            context, 2);
    const auto output =
        resolve_dispatch_write(
            context, 0);
    const auto element_bytes =
        activation_bytes(config);

    const auto expanded_elements =
        checked_mul(
            checked_mul(
                context.logical_rows,
                config.hidden_size),
            config.hc_count);
    const auto expanded_bytes =
        checked_mul(
            expanded_elements,
            element_bytes);
    const auto hidden_bytes =
        checked_mul(
            checked_mul(
                context.logical_rows,
                config.hidden_size),
            element_bytes);
    const auto weight_bytes =
        checked_mul(
            checked_mul(
                context.logical_rows,
                config.hc_count),
            sizeof(float));

    validate_buffer(
        hyper,
        expanded_bytes,
        "inject hyper input");
    validate_buffer(
        hidden,
        hidden_bytes,
        "inject hidden input");
    validate_buffer(
        weights,
        weight_bytes,
        "inject weight input");
    validate_buffer(
        output,
        expanded_bytes,
        "inject output");

    static_cast<void>(prepare_1d_blocks(expanded_elements));
    return {config, hyper, hidden, weights, output};
}

void preflight_inject(const ExecutionDispatchContext& context, void* user_data) {
    if (const auto* scope = current_hip_launch_context(); scope && scope->graph_capture)
        throw std::invalid_argument("known-window launcher preflight requires eager scope");
    static_cast<void>(prepare_inject(context, user_data));
}

void launch_inject(
    const ExecutionDispatchContext& context,
    void* user_data) {
    const auto prepared = prepare_inject(context, user_data);
    const auto& config = prepared.config;
    const auto& hyper = prepared.hyper;
    const auto& hidden = prepared.hidden;
    const auto& weights = prepared.weights;
    const auto& output = prepared.output;

    switch (config.activation) {
    case HipActivationStorage::bf16:
        launch_inject_typed<
            std::uint16_t>(
                context,
                config,
                hyper,
                hidden,
                weights,
                output);
        return;
    case HipActivationStorage::fp32:
        launch_inject_typed<float>(
            context,
            config,
            hyper,
            hidden,
            weights,
            output);
        return;
    }
    throw std::logic_error(
        "unreachable activation storage");
}
// Every row is admitted as launch_inject admits it; false (nothing enqueued) unless every row is one row and each
// of its four buffers sits at one pitch.
bool launch_inject_rows(std::span<const ExecutionDispatchContext> rows, void* user_data) {
    if (rows.size() < 2U || rows.size() > 65535U) return false;
    if (const auto* scope = current_hip_launch_context(); scope && scope->graph_capture) return false;
    const auto first = prepare_inject(rows[0], user_data);
    const auto second = prepare_inject(rows[1], user_data);
    const auto element = activation_bytes(first.config);
    const auto expanded = static_cast<std::uint64_t>(first.config.hidden_size) * first.config.hc_count;
    const auto pitch = [](const ExecutionBufferAddress& a, const ExecutionBufferAddress& b, std::uint64_t unit,
                           std::uint64_t minimum) -> std::uint64_t {
        if (b.address <= a.address) return 0U;
        const auto bytes = static_cast<std::uint64_t>(b.address - a.address);
        return bytes % unit || bytes < minimum * unit ? 0U : bytes / unit;
    };
    const auto hyper = pitch(first.hyper, second.hyper, element, expanded);
    const auto hidden = pitch(first.hidden, second.hidden, element, first.config.hidden_size);
    const auto weights = pitch(first.weights, second.weights, sizeof(float), first.config.hc_count);
    const auto output = pitch(first.output, second.output, element, expanded);
    if (rows[0].logical_rows != 1U || !hyper || !hidden || !weights || !output) return false;
    for (std::size_t r = 1; r < rows.size(); ++r) {
        const auto row = r == 1U ? second : prepare_inject(rows[r], user_data);
        if (rows[r].logical_rows != 1U || &row.config != &first.config ||
            row.hyper.address != first.hyper.address + r * hyper * element ||
            row.hidden.address != first.hidden.address + r * hidden * element ||
            row.weights.address != first.weights.address + r * weights * sizeof(float) ||
            row.output.address != first.output.address + r * output * element)
            return false;
    }
    const auto blocks = prepare_1d_blocks(expanded);
    const auto count = static_cast<std::uint32_t>(rows.size());
    const auto* weight_base = reinterpret_cast<const float*>(first.weights.address);
    switch (first.config.activation) {
    case HipActivationStorage::bf16:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(inject_pitched_rows_kernel<std::uint16_t>), dim3(blocks, count), dim3(256U), 0,
            hip_execution_stream(), reinterpret_cast<const std::uint16_t*>(first.hyper.address), hyper,
            reinterpret_cast<const std::uint16_t*>(first.hidden.address), hidden, weight_base, weights,
            reinterpret_cast<std::uint16_t*>(first.output.address), output, first.config.hidden_size, first.config.hc_count);
        break;
    case HipActivationStorage::fp32:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(inject_pitched_rows_kernel<float>), dim3(blocks, count), dim3(256U), 0,
            hip_execution_stream(), reinterpret_cast<const float*>(first.hyper.address), hyper,
            reinterpret_cast<const float*>(first.hidden.address), hidden, weight_base, weights,
            reinterpret_cast<float*>(first.output.address), output, first.config.hidden_size, first.config.hc_count);
        break;
    default:
        throw std::logic_error("unreachable activation storage");
    }
    check_hip(hipGetLastError(), "HIP inject rows kernel launch");
    return true;
}
[[maybe_unused]] const bool inject_rows_registered = register_rows_launch(&launch_inject, &launch_inject_rows);

template <typename Storage>
void launch_moe_combine_typed(
    const ExecutionDispatchContext& context,
    const HipElementwiseLauncherConfig& config,
    const ExecutionBufferAddress& shared,
    const ExecutionBufferAddress& routed,
    const ExecutionBufferAddress& output) {
    const auto elements =
        checked_mul(
            context.logical_rows,
            config.hidden_size);
    launch_1d(
        elements,
        [&](std::uint32_t blocks,
            std::uint32_t threads) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    moe_combine_kernel<Storage>),
                dim3(blocks),
                dim3(threads),
                0,
                hip_execution_stream(),
                reinterpret_cast<const Storage*>(
                    shared.address),
                reinterpret_cast<const Storage*>(
                    routed.address),
                reinterpret_cast<Storage*>(
                    output.address),
                elements);
        });
}

struct PreparedMoeCombine {
    const HipElementwiseLauncherConfig& config;
    ExecutionBufferAddress shared, routed, output;
};
PreparedMoeCombine prepare_moe_combine(
    const ExecutionDispatchContext& context,
    void* user_data) {
    if (context.op !=
        SerialOp::moe_combine) {
        throw std::invalid_argument(
            "HIP MoE-combine launcher attached to wrong operator");
    }
    if (context.reads.size() != 2 ||
        context.writes.size() != 1) {
        throw std::invalid_argument(
            "HIP MoE-combine expects two reads and one write");
    }

    const auto& config =
        require_config(user_data);
    const auto shared =
        resolve_dispatch_read(
            context, 0);
    const auto routed =
        resolve_dispatch_read(
            context, 1);
    const auto output =
        resolve_dispatch_write(
            context, 0);
    const auto required =
        checked_mul(
            checked_mul(
                context.logical_rows,
                config.hidden_size),
            activation_bytes(config));

    validate_buffer(
        shared,
        required,
        "MoE-combine shared input");
    validate_buffer(
        routed,
        required,
        "MoE-combine routed input");
    validate_buffer(
        output,
        required,
        "MoE-combine output");

    static_cast<void>(prepare_1d_blocks(checked_mul(context.logical_rows, config.hidden_size)));
    return {config, shared, routed, output};
}

void preflight_moe_combine(const ExecutionDispatchContext& context, void* user_data) {
    if (const auto* scope = current_hip_launch_context(); scope && scope->graph_capture)
        throw std::invalid_argument("known-window launcher preflight requires eager scope");
    static_cast<void>(prepare_moe_combine(context, user_data));
}

void launch_moe_combine(
    const ExecutionDispatchContext& context,
    void* user_data) {
    const auto prepared = prepare_moe_combine(context, user_data);
    const auto& config = prepared.config;
    const auto& shared = prepared.shared;
    const auto& routed = prepared.routed;
    const auto& output = prepared.output;

    switch (config.activation) {
    case HipActivationStorage::bf16:
        launch_moe_combine_typed<
            std::uint16_t>(
                context,
                config,
                shared,
                routed,
                output);
        return;
    case HipActivationStorage::fp32:
        launch_moe_combine_typed<float>(
            context,
            config,
            shared,
            routed,
            output);
        return;
    }
    throw std::logic_error(
        "unreachable activation storage");
}

ExecutionLauncher make_launcher(
    std::string kernel_id,
    std::uint64_t tag,
    ExecutionLaunchFunction function,
    HipElementwiseLauncherConfig* config,
    ExecutionLaunchFunction preflight) {
    if (kernel_id.empty()) {
        throw std::invalid_argument(
            "HIP launcher kernel id must not be empty");
    }
    if (config == nullptr) {
        throw std::invalid_argument(
            "HIP launcher configuration must not be null");
    }
    const auto identity =
        hip_elementwise_launcher_identity(
            tag, *config);
    return ExecutionLauncher{
        std::move(kernel_id),
        identity,
        function,
        config,
        preflight,
    };
}

}  // namespace


namespace {

const char* activation_name(
    HipActivationStorage storage) noexcept {
    switch (storage) {
    case HipActivationStorage::bf16:
        return "bf16";
    case HipActivationStorage::fp32:
        return "fp32";
    }
    return "unknown";
}

kernels::KernelVariant make_elementwise_variant(
    std::string id,
    kernels::OperatorFamily family,
    const kernels::ShapeClass& shape,
    const HipElementwiseLauncherConfig& config,
    kernels::QualificationStatus qualification) {
    kernels::KernelVariant variant;
    variant.id = std::move(id);
    variant.family = family;
    variant.phase =
        kernels::ExecutionPhase::decode;
    variant.shape = shape;
    variant.wave =
        kernels::WaveSize::wave32;
    variant.resources.workgroup_threads = 256;
    variant.numerics.input_precision =
        activation_name(config.activation);
    variant.numerics.accumulator_precision = "fp32";
    variant.numerics.output_precision =
        activation_name(config.activation);
    variant.numerics.row_invariant = true;
    variant.numerics.deterministic = true;
    variant.qualification = qualification;
    variant.graph_safe = true;
    variant.input_layout = "row-major";
    variant.output_layout = "row-major";
    variant.compiler_identity =
        "hip-gfx1151-elementwise-v1";
    variant.measured_seconds = 0.0;
    return variant;
}

std::string elementwise_kernel_id(
    std::string_view operation,
    const ExecutionBindingRequest& request,
    const HipElementwiseLauncherConfig& config) {
    return
        "hip." +
        std::string(operation) +
        "." +
        activation_name(config.activation) +
        ".h" +
        std::to_string(config.hidden_size) +
        ".hc" +
        std::to_string(config.hc_count) +
        ".r" +
        std::to_string(request.logical_rows) +
        ".c" +
        std::to_string(request.request_concurrency);
}

}  // namespace

void register_hip_elementwise_kernel_variants(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipElementwiseLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification) {
    if (const auto error = model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (request.logical_rows == 0 ||
        request.request_concurrency == 0) {
        throw std::invalid_argument(
            "HIP elementwise variants require non-zero rows and concurrency");
    }
    if (config.hidden_size != model.hidden_size ||
        config.hc_count != model.hc_count) {
        throw std::invalid_argument(
            "HIP elementwise variant geometry does not match model contract");
    }
    if (qualification ==
            kernels::QualificationStatus::target_qualified ||
        qualification ==
            kernels::QualificationStatus::production ||
        qualification ==
            kernels::QualificationStatus::rejected) {
        throw std::invalid_argument(
            "unmeasured HIP elementwise catalog may only register experimental or oracle-matched variants");
    }
    static_cast<void>(
        activation_bytes(config));

    const auto hyper_shape =
        execution_shape_for(
            model,
            SerialOp::hyper_expand,
            request);
    const auto inject_shape =
        execution_shape_for(
            model,
            SerialOp::attn_inject,
            request);
    const auto combine_shape =
        execution_shape_for(
            model,
            SerialOp::moe_combine,
            request);

    registry.add(
        make_elementwise_variant(
            elementwise_kernel_id(
                "hyper_expand",
                request,
                config),
            kernels::OperatorFamily::hyper_expand,
            hyper_shape,
            config,
            qualification));

    registry.add(
        make_elementwise_variant(
            elementwise_kernel_id(
                "inject",
                request,
                config),
            kernels::OperatorFamily::elementwise_inject,
            inject_shape,
            config,
            qualification));

    registry.add(
        make_elementwise_variant(
            elementwise_kernel_id(
                "moe_combine",
                request,
                config),
            kernels::OperatorFamily::moe_combine,
            combine_shape,
            config,
            qualification));
}

std::uint64_t
hip_elementwise_launcher_identity(
    std::uint64_t operation_tag,
    const HipElementwiseLauncherConfig& config) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, operation_tag);
    hash_u64(hash, config.hidden_size);
    hash_u64(hash, config.hc_count);
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            config.activation));
    return hash;
}

ExecutionLauncher
make_hip_hyper_expand_launcher(
    std::string kernel_id,
    HipElementwiseLauncherConfig* config) {
    return make_launcher(
        std::move(kernel_id),
        hyper_expand_tag,
        &launch_hyper_expand,
        config,
        &preflight_hyper_expand);
}

ExecutionLauncher
make_hip_inject_launcher(
    std::string kernel_id,
    HipElementwiseLauncherConfig* config) {
    return make_launcher(
        std::move(kernel_id),
        inject_tag,
        &launch_inject,
        config,
        &preflight_inject);
}

ExecutionLauncher
make_hip_moe_combine_launcher(
    std::string kernel_id,
    HipElementwiseLauncherConfig* config) {
    return make_launcher(
        std::move(kernel_id),
        moe_combine_tag,
        &launch_moe_combine,
        config,
        &preflight_moe_combine);
}

}  // namespace strix


namespace strix {
namespace {

enum class ElementwiseLauncherKind {
    hyper_expand,
    inject,
    moe_combine,
};

std::optional<ElementwiseLauncherKind>
elementwise_launcher_kind(
    SerialOp op) noexcept {
    switch (op) {
    case SerialOp::hyper_expand:
        return ElementwiseLauncherKind::hyper_expand;
    case SerialOp::attn_inject:
    case SerialOp::mlp_inject:
        return ElementwiseLauncherKind::inject;
    case SerialOp::moe_combine:
        return ElementwiseLauncherKind::moe_combine;
    default:
        return std::nullopt;
    }
}

}  // namespace

void HipElementwiseLauncherBundle::
register_for_compiled_execution(
    const CompiledSerialExecution& compiled,
    ExecutionLauncherRegistry& registry) {
    if (compiled.execution_identity == 0 ||
        compiled.execution.steps.size() !=
            compiled.bindings.bindings.size()) {
        throw std::invalid_argument(
            "HIP launcher bundle requires a valid compiled execution");
    }

    struct Registered {
        std::string kernel_id;
        ElementwiseLauncherKind kind;
    };
    std::vector<Registered> registered;

    for (std::size_t index = 0;
         index < compiled.execution.steps.size();
         ++index) {
        const auto& step =
            compiled.execution.steps[index];
        const auto kind =
            elementwise_launcher_kind(
                step.op);
        if (!kind.has_value()) {
            continue;
        }

        const auto& binding =
            compiled.bindings.bindings[index];
        if (binding.host_or_runtime_step ||
            !binding.kernel_id.has_value() ||
            binding.kernel_id->empty()) {
            throw std::invalid_argument(
                "HIP elementwise step is not bound to a kernel id");
        }

        const auto existing =
            std::find_if(
                registered.begin(),
                registered.end(),
                [&](const Registered& value) {
                    return value.kernel_id ==
                           *binding.kernel_id;
                });

        if (existing != registered.end()) {
            if (existing->kind != *kind) {
                throw std::invalid_argument(
                    "one kernel id is bound to incompatible HIP elementwise semantics");
            }
            continue;
        }

        ExecutionLauncher launcher;
        switch (*kind) {
        case ElementwiseLauncherKind::hyper_expand:
            launcher =
                make_hip_hyper_expand_launcher(
                    *binding.kernel_id,
                    &config_);
            break;
        case ElementwiseLauncherKind::inject:
            launcher =
                make_hip_inject_launcher(
                    *binding.kernel_id,
                    &config_);
            break;
        case ElementwiseLauncherKind::moe_combine:
            launcher =
                make_hip_moe_combine_launcher(
                    *binding.kernel_id,
                    &config_);
            break;
        }

        registry.add_kernel_launcher(
            std::move(launcher));
        registered.push_back(
            Registered{
                *binding.kernel_id,
                *kind,
            });
    }
}

void preflight_hip_mtp_moe_combine(const MtpNativeContext& c) {
    mtp_native::context(c);
    const auto bytes=mtp_native::product(c.model->hidden_size,2);
    auto a=mtp_native::buffer(c,HipMtpBuffer::hidden1,bytes);
    auto b=mtp_native::buffer(c,HipMtpBuffer::hidden2,bytes);
    auto out=mtp_native::buffer(c,HipMtpBuffer::hidden0,bytes);
    mtp_native::disjoint({a,b,out});
    (void)prepare_1d_blocks(c.model->hidden_size);
}
void launch_hip_mtp_moe_combine(const MtpNativeContext& c,void* stream) {
    preflight_hip_mtp_moe_combine(c); mtp_native::stream(c,stream);
    check_hip(hipGetLastError(),"MTP MoE combine pending HIP error");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(moe_combine_kernel<std::uint16_t>),
        dim3(prepare_1d_blocks(c.model->hidden_size)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),
        reinterpret_cast<const std::uint16_t*>(c.resources->view(HipMtpBuffer::hidden1).address),
        reinterpret_cast<const std::uint16_t*>(c.resources->view(HipMtpBuffer::hidden2).address),
        reinterpret_cast<std::uint16_t*>(c.resources->view(HipMtpBuffer::hidden0).address),
        c.model->hidden_size);
    check_hip(hipGetLastError(),"MTP native MoE combine launch");
}

void preflight_hip_mtp_inject(const MtpNativeContext& c,MtpInjectRole role) {
    mtp_native::context(c);
    if(role!=MtpInjectRole::attention && role!=MtpInjectRole::mlp)
        throw std::invalid_argument("MTP injection role unsupported");
    const auto h=c.model->hidden_size,n=c.model->hc_count;
    if(!h || h>std::numeric_limits<std::uint32_t>::max() || !n || n>std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("MTP injection geometry exceeds native ABI");
    const auto wide=mtp_native::product(mtp_native::product(h,n),2);
    const auto attn=role==MtpInjectRole::attention;
    auto input=mtp_native::buffer(c,attn?HipMtpBuffer::fused_wide:HipMtpBuffer::wide_work,wide);
    auto hidden=mtp_native::buffer(c,attn?HipMtpBuffer::hidden1:HipMtpBuffer::hidden0,mtp_native::product(h,2));
    auto injection=mtp_native::buffer(c,HipMtpBuffer::injection,mtp_native::product(n,4));
    auto output=mtp_native::buffer(c,attn?HipMtpBuffer::wide_work:HipMtpBuffer::wide_next,wide);
    mtp_native::disjoint({input,hidden,injection,output});
    (void)prepare_1d_blocks(wide/2);
}
void launch_hip_mtp_inject(const MtpNativeContext& c,MtpInjectRole role,void* stream) {
    preflight_hip_mtp_inject(c,role);mtp_native::stream(c,stream);
    check_hip(hipGetLastError(),"MTP injection pending HIP error");
    const auto attn=role==MtpInjectRole::attention;
    const auto wide=mtp_native::product(c.model->hidden_size,c.model->hc_count);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(inject_kernel<std::uint16_t>),
        dim3(prepare_1d_blocks(wide)),dim3(256),0,reinterpret_cast<hipStream_t>(stream),
        reinterpret_cast<const std::uint16_t*>(c.resources->view(attn?HipMtpBuffer::fused_wide:HipMtpBuffer::wide_work).address),
        reinterpret_cast<const std::uint16_t*>(c.resources->view(attn?HipMtpBuffer::hidden1:HipMtpBuffer::hidden0).address),
        reinterpret_cast<const float*>(c.resources->view(HipMtpBuffer::injection).address),
        reinterpret_cast<std::uint16_t*>(c.resources->view(attn?HipMtpBuffer::wide_work:HipMtpBuffer::wide_next).address),
        static_cast<std::uint32_t>(c.model->hidden_size),static_cast<std::uint32_t>(c.model->hc_count),1U);
    check_hip(hipGetLastError(),"MTP native injection launch");
}
void launch_hip_mtp_inject_rows(std::span<const MtpNativeContext> contexts,MtpInjectRole role,void* stream) {
    if(contexts.empty() || contexts.size()>8U || !stream) throw std::invalid_argument("MTP injection batch size or stream is invalid");
    const auto attn=role==MtpInjectRole::attention;
    MtpInjectRows rows{};
    const auto h=static_cast<std::uint32_t>(contexts.front().model->hidden_size);
    const auto hc=static_cast<std::uint32_t>(contexts.front().model->hc_count);
    for(std::size_t t=0;t<contexts.size();++t) {
        const auto& c=contexts[t];
        preflight_hip_mtp_inject(c,role);
        if(c.model->hidden_size!=h || c.model->hc_count!=hc) throw std::invalid_argument("MTP injection batch geometry differs");
        rows.hyper[t]=reinterpret_cast<const std::uint16_t*>(c.resources->view(attn?HipMtpBuffer::fused_wide:HipMtpBuffer::wide_work).address);
        rows.hidden[t]=reinterpret_cast<const std::uint16_t*>(c.resources->view(attn?HipMtpBuffer::hidden1:HipMtpBuffer::hidden0).address);
        rows.weights[t]=reinterpret_cast<const float*>(c.resources->view(HipMtpBuffer::injection).address);
        rows.output[t]=reinterpret_cast<std::uint16_t*>(c.resources->view(attn?HipMtpBuffer::wide_work:HipMtpBuffer::wide_next).address);
    }
    check_hip(hipGetLastError(),"MTP injection batch pending HIP error");
    const auto wide=mtp_native::product(h,hc);
    hipLaunchKernelGGL(mtp_inject_rows_kernel,dim3(prepare_1d_blocks(wide),static_cast<unsigned>(contexts.size())),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),rows,h,hc);
    check_hip(hipGetLastError(),"MTP injection batch launch");
}
void launch_hip_mtp_publish_next_rows(std::span<const MtpNativeContext> contexts,void* stream) {
    if(contexts.empty() || contexts.size()>8U || !stream) throw std::invalid_argument("MTP state publish batch size or stream is invalid");
    MtpCopyRows rows{};std::uint64_t bytes=0;
    for(std::size_t t=0;t<contexts.size();++t) {
        const auto source=contexts[t].resources->view(HipMtpBuffer::wide_next);
        const auto destination=contexts[t].resources->view(HipMtpBuffer::wide_current);
        if(source.bytes!=destination.bytes || !source.bytes || source.bytes%16U || source.address%16U || destination.address%16U ||
           (t && source.bytes!=bytes))
            throw std::invalid_argument("MTP state publish batch views are invalid");
        bytes=source.bytes;
        rows.source[t]=reinterpret_cast<const uint4*>(source.address);rows.destination[t]=reinterpret_cast<uint4*>(destination.address);
    }
    const auto vectors=bytes/16U;
    hipLaunchKernelGGL(mtp_copy_rows_kernel,dim3(static_cast<unsigned>(std::min<std::uint64_t>((vectors+255U)/256U,64U)),
        static_cast<unsigned>(contexts.size())),dim3(256),0,reinterpret_cast<hipStream_t>(stream),rows,vectors);
    check_hip(hipGetLastError(),"MTP state publish batch launch");
}

}  // namespace strix

namespace strix {
namespace {
struct ElementwiseRowView {
    std::array<std::pair<std::uintptr_t, std::uint64_t>, 3> reads{};
    std::pair<std::uintptr_t, std::uint64_t> write{};
};
// Each row passes the complete serial admission; rows add the window shape
// (one step, consecutive positions) and cross-row aliasing. In-place rows
// remain legal exactly as in the serial launcher.
template <typename Prepare>
ElementwiseRowsArgs prepare_elementwise_rows(std::span<const ExecutionDispatchContext> rows, Prepare&& prepare) {
    if (rows.size() != 2U && rows.size() != 4U && rows.size() != 8U)
        throw std::invalid_argument("known-window elementwise rows count must be 2, 4 or 8");
    std::array<ElementwiseRowView, kElementwiseMaxRows> views{};
    ElementwiseRowsArgs args{};
    const auto& first = rows.front();
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const auto& row = rows[r];
        if (row.logical_rows != 1U || row.request_concurrency != 1U ||
            row.execution_identity != first.execution_identity || row.step_id != first.step_id ||
            row.op != first.op || row.layer != first.layer || row.kernel_id != first.kernel_id ||
            row.position != first.position + r)
            throw std::invalid_argument("known-window elementwise rows identity/position differs");
        views[r] = prepare(row, args, r);
    }
    const auto overlap = [](auto a, auto b) { return a.first < b.first + b.second && b.first < a.first + a.second; };
    for (std::size_t r = 0; r < rows.size(); ++r)
        for (std::size_t o = 0; o < rows.size(); ++o) {
            if (o == r) continue;
            if (o < r && overlap(views[r].write, views[o].write))
                throw std::invalid_argument("known-window elementwise rows outputs alias");
            for (const auto read : views[o].reads) if (read.second && overlap(views[r].write, read))
                throw std::invalid_argument("known-window elementwise rows output aliases another row's input");
        }
    return args;
}
ElementwiseRowsArgs prepare_inject_rows(std::span<const ExecutionDispatchContext> rows,
    const HipElementwiseLauncherConfig& config) {
    auto* user = const_cast<HipElementwiseLauncherConfig*>(&config);
    const auto hidden = checked_mul(config.hidden_size, activation_bytes(config));
    const auto expanded = checked_mul(hidden, config.hc_count);
    return prepare_elementwise_rows(rows, [&](const ExecutionDispatchContext& row, ElementwiseRowsArgs& args, std::size_t r) {
        const auto p = prepare_inject(row, user);
        args.first[r] = reinterpret_cast<const void*>(p.hyper.address);
        args.second[r] = reinterpret_cast<const void*>(p.hidden.address);
        args.weights[r] = reinterpret_cast<const float*>(p.weights.address);
        args.output[r] = reinterpret_cast<void*>(p.output.address);
        return ElementwiseRowView{{{{p.hyper.address, expanded}, {p.hidden.address, hidden},
            {p.weights.address, config.hc_count * 4ULL}}}, {p.output.address, expanded}};
    });
}
ElementwiseRowsArgs prepare_combine_rows(std::span<const ExecutionDispatchContext> rows,
    const HipElementwiseLauncherConfig& config) {
    auto* user = const_cast<HipElementwiseLauncherConfig*>(&config);
    const auto hidden = checked_mul(config.hidden_size, activation_bytes(config));
    return prepare_elementwise_rows(rows, [&](const ExecutionDispatchContext& row, ElementwiseRowsArgs& args, std::size_t r) {
        const auto p = prepare_moe_combine(row, user);
        args.first[r] = reinterpret_cast<const void*>(p.shared.address);
        args.second[r] = reinterpret_cast<const void*>(p.routed.address);
        args.output[r] = reinterpret_cast<void*>(p.output.address);
        return ElementwiseRowView{{{{p.shared.address, hidden}, {p.routed.address, hidden}, {0U, 0U}}},
            {p.output.address, hidden}};
    });
}
hipStream_t elementwise_rows_stream() { return hip_execution_stream(); }
// Prefill: every 8-row tile is admitted like a window call; tiles must be one stride apart in each view and,
// since they run concurrently, no row may write what another row (of any tile) reads or writes. A row may
// still update its own input in place, as in the serial launcher.
template <typename Prepare>
ElementwiseTileRowsArgs prepare_elementwise_prefill(std::span<const ExecutionDispatchContext> rows, Prepare&& prepare,
    std::uint64_t first_bytes, std::uint64_t second_bytes, std::uint64_t weight_bytes, std::uint64_t output_bytes,
    std::uint32_t& tiles) {
    if (rows.size() < 16U || rows.size() % 8U || rows.size() / 8U > 65535U)
        throw std::invalid_argument("elementwise prefill rows require 8-row tiles");
    tiles = static_cast<std::uint32_t>(rows.size() / 8U);
    const auto a = prepare(rows.first(8U)), b = prepare(rows.subspan(8U, 8U));
    const auto at = [](const auto* p) { return reinterpret_cast<std::uintptr_t>(p); };
    const auto stride = [&](const auto* x, const auto* y) -> std::uint64_t {
        if (!x) return 0U;
        if (at(y) <= at(x)) throw std::invalid_argument("elementwise prefill tiles are not strided");
        return at(y) - at(x);
    };
    ElementwiseTileRowsArgs args;
    static_cast<ElementwiseRowsArgs&>(args) = a;
    args.tile_first = stride(a.first[0], b.first[0]); args.tile_second = stride(a.second[0], b.second[0]);
    args.tile_weights = stride(a.weights[0], b.weights[0]); args.tile_output = stride(a.output[0], b.output[0]);
    for (std::uint32_t t = 2; t < tiles; ++t) {
        const auto c = prepare(rows.subspan(t * 8U, 8U));
        const auto same = [&](const auto* x, const auto* y, std::uint64_t step) {
            return x ? at(y) == at(x) + static_cast<std::uint64_t>(t) * step : y == nullptr;
        };
        for (std::uint32_t r = 0; r < 8U; ++r)
            if (!same(a.first[r], c.first[r], args.tile_first) || !same(a.second[r], c.second[r], args.tile_second) ||
                !same(a.weights[r], c.weights[r], args.tile_weights) || !same(a.output[r], c.output[r], args.tile_output))
                throw std::invalid_argument("elementwise prefill tile views are not strided");
    }
    // Each tile's views must fit within one tile stride (tiles never overlap each other), and the output field
    // must stay clear of every input field over all T rows, except an exact in-place first input (same address
    // per row and same stride), which the window admission already allows within a tile.
    const auto extent = [&](const auto* const* v, std::uint64_t bytes) {
        std::uintptr_t low = UINTPTR_MAX, high = 0;
        for (std::uint32_t r = 0; r < 8U; ++r) { low = std::min(low, at(v[r])); high = std::max(high, at(v[r]) + bytes); }
        return std::pair<std::uintptr_t, std::uintptr_t>{low, high};
    };
    const auto whole = [&](std::pair<std::uintptr_t, std::uintptr_t> e, std::uint64_t step) {
        if (e.second - e.first > step) throw std::invalid_argument("elementwise prefill tile views exceed their stride");
        return std::pair<std::uintptr_t, std::uintptr_t>{e.first, e.second + static_cast<std::uint64_t>(tiles - 1U) * step};
    };
    const auto out = whole(extent(a.output, output_bytes), args.tile_output);
    const auto clear = [&](std::pair<std::uintptr_t, std::uintptr_t> x) {
        if (x.first < out.second && out.first < x.second) throw std::invalid_argument("elementwise prefill rows alias across tiles");
    };
    bool in_place = args.tile_first == args.tile_output;
    for (std::uint32_t r = 0; r < 8U; ++r) in_place = in_place && at(a.first[r]) == at(a.output[r]);
    if (!in_place) clear(whole(extent(a.first, first_bytes), args.tile_first));
    clear(whole(extent(a.second, second_bytes), args.tile_second));
    if (a.weights[0]) clear(whole(extent(a.weights, weight_bytes), args.tile_weights));
    return args;
}
} // namespace

void inject_known_window_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipElementwiseLauncherConfig& config) {
    static_cast<void>(prepare_inject_rows(rows, config));
}
namespace {
ElementwiseRowsViews elementwise_views(const ElementwiseRowsArgs& args, std::size_t count) {
    ElementwiseRowsViews views;
    views.count = static_cast<std::uint32_t>(count);
    for (std::uint32_t r = 0; r < views.count; ++r) {
        views.first[r] = args.first[r]; views.second[r] = args.second[r];
        views.weights[r] = args.weights[r]; views.output[r] = args.output[r];
    }
    return views;
}
} // namespace
ElementwiseRowsViews inject_known_window_rows_views(std::span<const ExecutionDispatchContext> rows,
    const HipElementwiseLauncherConfig& config) {
    return elementwise_views(prepare_inject_rows(rows, config), rows.size());
}
ElementwiseRowsViews moe_combine_known_window_rows_views(std::span<const ExecutionDispatchContext> rows,
    const HipElementwiseLauncherConfig& config) {
    auto views = elementwise_views(prepare_combine_rows(rows, config), rows.size());
    for (auto& weights : views.weights) weights = nullptr;
    return views;
}
void inject_known_window_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipElementwiseLauncherConfig& config) {
    const auto args = prepare_inject_rows(rows, config);
    const dim3 grid(prepare_1d_blocks(checked_mul(config.hidden_size, config.hc_count)), static_cast<std::uint32_t>(rows.size()));
    if (config.activation == HipActivationStorage::bf16)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(inject_rows_kernel<std::uint16_t>), grid, dim3(256U), 0,
            elementwise_rows_stream(), args, config.hidden_size, config.hc_count);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(inject_rows_kernel<float>), grid, dim3(256U), 0,
            elementwise_rows_stream(), args, config.hidden_size, config.hc_count);
    check_hip(hipGetLastError(), "known-window inject rows launch");
}
void inject_known_prefill_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipElementwiseLauncherConfig& config) {
    std::uint32_t tiles = 0;
    const auto hidden = checked_mul(config.hidden_size, activation_bytes(config));
    static_cast<void>(prepare_elementwise_prefill(rows, [&](auto tile) { return prepare_inject_rows(tile, config); },
        checked_mul(hidden, config.hc_count), hidden, config.hc_count * 4ULL, checked_mul(hidden, config.hc_count), tiles));
}
void inject_known_prefill_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipElementwiseLauncherConfig& config) {
    std::uint32_t tiles = 0;
    const auto hidden = checked_mul(config.hidden_size, activation_bytes(config));
    const auto args = prepare_elementwise_prefill(rows, [&](auto tile) { return prepare_inject_rows(tile, config); },
        checked_mul(hidden, config.hc_count), hidden, config.hc_count * 4ULL, checked_mul(hidden, config.hc_count), tiles);
    const dim3 grid(prepare_1d_blocks(checked_mul(config.hidden_size, config.hc_count)), 8U, tiles);
    if (config.activation == HipActivationStorage::bf16)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(inject_rows_kernel<std::uint16_t, ElementwiseTileRowsArgs>), grid, dim3(256U), 0,
            elementwise_rows_stream(), args, config.hidden_size, config.hc_count);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(inject_rows_kernel<float, ElementwiseTileRowsArgs>), grid, dim3(256U), 0,
            elementwise_rows_stream(), args, config.hidden_size, config.hc_count);
    check_hip(hipGetLastError(), "known prefill inject rows launch");
}
namespace {
bool uniform_prefill_views(const ElementwiseTileRowsArgs& args, std::uint32_t tiles, bool weights, ElementwisePrefillViews& views) {
    const auto at = [](const auto* p) { return reinterpret_cast<std::uintptr_t>(p); };
    const auto uniform = [&](const auto* const* v, std::uint64_t tile, std::uint64_t& stride) {
        if (tile % 8U) return false;
        stride = tile / 8U;
        for (std::uint32_t r = 0; r < 8U; ++r)
            if (!v[r] || at(v[r]) != at(v[0]) + r * stride) return false;
        return true;
    };
    ElementwisePrefillViews out;
    if (!uniform(args.first, args.tile_first, out.first_stride) || !uniform(args.second, args.tile_second, out.second_stride) ||
        !uniform(args.output, args.tile_output, out.output_stride) ||
        (weights && !uniform(args.weights, args.tile_weights, out.weights_stride)))
        return false;
    out.first = args.first[0]; out.second = args.second[0]; out.output = args.output[0];
    out.weights = weights ? args.weights[0] : nullptr;
    out.count = tiles * 8U;
    views = out;
    return true;
}
} // namespace
void testing::enqueue_elementwise_prefill_rows_test(bool combine, const void* first, std::uint64_t first_stride, const void* second,
    std::uint64_t second_stride, const float* weights, std::uint64_t weights_stride, void* output, std::uint64_t output_stride,
    std::uint32_t rows, std::uint32_t hidden_size, std::uint32_t hc_count, hipStream_t stream) {
    if (!rows || rows % 8U || rows / 8U > 65535U) throw std::invalid_argument("elementwise prefill test rows");
    ElementwiseTileRowsArgs args{};
    const auto at = [](const void* p, std::uint64_t stride, std::uint32_t r) {
        return reinterpret_cast<std::uintptr_t>(p) + static_cast<std::uint64_t>(r) * stride;
    };
    for (std::uint32_t r = 0; r < 8U; ++r) {
        args.first[r] = reinterpret_cast<const void*>(at(first, first_stride, r));
        args.second[r] = reinterpret_cast<const void*>(at(second, second_stride, r));
        args.weights[r] = combine ? nullptr : reinterpret_cast<const float*>(at(weights, weights_stride, r));
        args.output[r] = reinterpret_cast<void*>(at(output, output_stride, r));
    }
    args.tile_first = 8U * first_stride; args.tile_second = 8U * second_stride;
    args.tile_weights = combine ? 0U : 8U * weights_stride; args.tile_output = 8U * output_stride;
    const auto tiles = rows / 8U;
    if (combine)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(combine_rows_kernel<std::uint16_t, ElementwiseTileRowsArgs>), dim3(prepare_1d_blocks(hidden_size), 8U, tiles),
            dim3(256U), 0, stream, args, hidden_size);
    else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(inject_rows_kernel<std::uint16_t, ElementwiseTileRowsArgs>),
            dim3(prepare_1d_blocks(checked_mul(hidden_size, hc_count)), 8U, tiles), dim3(256U), 0, stream, args, hidden_size, hc_count);
    check_hip(hipGetLastError(), "elementwise prefill rows test launch");
}
bool inject_known_prefill_rows_views(std::span<const ExecutionDispatchContext> rows, const HipElementwiseLauncherConfig& config,
    ElementwisePrefillViews& views) {
    std::uint32_t tiles = 0;
    const auto hidden = checked_mul(config.hidden_size, activation_bytes(config));
    const auto args = prepare_elementwise_prefill(rows, [&](auto tile) { return prepare_inject_rows(tile, config); },
        checked_mul(hidden, config.hc_count), hidden, config.hc_count * 4ULL, checked_mul(hidden, config.hc_count), tiles);
    return uniform_prefill_views(args, tiles, true, views);
}
bool moe_combine_known_prefill_rows_views(std::span<const ExecutionDispatchContext> rows, const HipElementwiseLauncherConfig& config,
    ElementwisePrefillViews& views) {
    std::uint32_t tiles = 0;
    const auto hidden = checked_mul(config.hidden_size, activation_bytes(config));
    const auto args = prepare_elementwise_prefill(rows, [&](auto tile) { return prepare_combine_rows(tile, config); },
        hidden, hidden, 0U, hidden, tiles);
    return uniform_prefill_views(args, tiles, false, views);
}
void moe_combine_known_prefill_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipElementwiseLauncherConfig& config) {
    std::uint32_t tiles = 0;
    const auto hidden = checked_mul(config.hidden_size, activation_bytes(config));
    static_cast<void>(prepare_elementwise_prefill(rows, [&](auto tile) { return prepare_combine_rows(tile, config); },
        hidden, hidden, 0U, hidden, tiles));
}
void moe_combine_known_prefill_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipElementwiseLauncherConfig& config) {
    std::uint32_t tiles = 0;
    const auto hidden = checked_mul(config.hidden_size, activation_bytes(config));
    const auto args = prepare_elementwise_prefill(rows, [&](auto tile) { return prepare_combine_rows(tile, config); },
        hidden, hidden, 0U, hidden, tiles);
    const dim3 grid(prepare_1d_blocks(config.hidden_size), 8U, tiles);
    if (config.activation == HipActivationStorage::bf16)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(combine_rows_kernel<std::uint16_t, ElementwiseTileRowsArgs>), grid, dim3(256U), 0,
            elementwise_rows_stream(), args, config.hidden_size);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(combine_rows_kernel<float, ElementwiseTileRowsArgs>), grid, dim3(256U), 0,
            elementwise_rows_stream(), args, config.hidden_size);
    check_hip(hipGetLastError(), "known prefill MoE-combine rows launch");
}
void moe_combine_known_window_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipElementwiseLauncherConfig& config) {
    static_cast<void>(prepare_combine_rows(rows, config));
}
namespace {
// The inject (or MoE-combine) rows of several sessions' windows (each segment admitted as its own window) as
// one argument block; no row may write what a row of another segment reads or writes.
ElementwiseRowsArgs prepare_elementwise_cohort_rows(std::span<const std::span<const ExecutionDispatchContext>> segments,
    const HipElementwiseLauncherConfig& config, std::uint32_t& count, bool inject) {
    ElementwiseRowsArgs all{};
    count = 0;
    struct Range { std::uintptr_t low, high; };
    const auto hidden = checked_mul(config.hidden_size, activation_bytes(config));
    const auto expanded = inject ? checked_mul(hidden, config.hc_count) : hidden;
    for (const auto& segment : segments) {
        if (count + segment.size() > kElementwiseMaxRows) throw std::invalid_argument("known-window elementwise cohort exceeds 32 rows");
        const auto args = inject ? prepare_inject_rows(segment, config) : prepare_combine_rows(segment, config);
        for (std::size_t r = 0; r < segment.size(); ++r, ++count) {
            all.first[count] = args.first[r]; all.second[count] = args.second[r];
            all.weights[count] = args.weights[r]; all.output[count] = args.output[r];
        }
    }
    const auto at = [](const void* p) { return reinterpret_cast<std::uintptr_t>(p); };
    for (std::uint32_t r = 0; r < count; ++r)
        for (std::uint32_t o = 0; o < count; ++o) {
            if (o == r) continue;
            const Range out{at(all.output[r]), at(all.output[r]) + expanded};
            for (const Range other : {Range{at(all.output[o]), at(all.output[o]) + expanded}, Range{at(all.first[o]), at(all.first[o]) + expanded},
                     Range{at(all.second[o]), at(all.second[o]) + hidden}})
                if (out.low < other.high && other.low < out.high)
                    throw std::invalid_argument("known-window elementwise cohort rows alias across rows");
        }
    return all;
}
} // namespace
void inject_known_window_cohort_rows_launch(std::span<const std::span<const ExecutionDispatchContext>> segments,
    const HipElementwiseLauncherConfig& config) {
    std::uint32_t count = 0;
    const auto args = prepare_elementwise_cohort_rows(segments, config, count, true);
    const dim3 grid(prepare_1d_blocks(checked_mul(config.hidden_size, config.hc_count)), count);
    if (config.activation == HipActivationStorage::bf16)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(inject_rows_kernel<std::uint16_t>), grid, dim3(256U), 0,
            elementwise_rows_stream(), args, config.hidden_size, config.hc_count);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(inject_rows_kernel<float>), grid, dim3(256U), 0,
            elementwise_rows_stream(), args, config.hidden_size, config.hc_count);
    check_hip(hipGetLastError(), "known-window inject cohort rows launch");
}
void moe_combine_known_window_cohort_rows_launch(std::span<const std::span<const ExecutionDispatchContext>> segments,
    const HipElementwiseLauncherConfig& config) {
    std::uint32_t count = 0;
    const auto args = prepare_elementwise_cohort_rows(segments, config, count, false);
    const dim3 grid(prepare_1d_blocks(config.hidden_size), count);
    if (config.activation == HipActivationStorage::bf16)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(combine_rows_kernel<std::uint16_t>), grid, dim3(256U), 0,
            elementwise_rows_stream(), args, config.hidden_size);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(combine_rows_kernel<float>), grid, dim3(256U), 0,
            elementwise_rows_stream(), args, config.hidden_size);
    check_hip(hipGetLastError(), "known-window MoE-combine cohort rows launch");
}
void moe_combine_known_window_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipElementwiseLauncherConfig& config) {
    const auto args = prepare_combine_rows(rows, config);
    const dim3 grid(prepare_1d_blocks(config.hidden_size), static_cast<std::uint32_t>(rows.size()));
    if (config.activation == HipActivationStorage::bf16)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(combine_rows_kernel<std::uint16_t>), grid, dim3(256U), 0,
            elementwise_rows_stream(), args, config.hidden_size);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(combine_rows_kernel<float>), grid, dim3(256U), 0,
            elementwise_rows_stream(), args, config.hidden_size);
    check_hip(hipGetLastError(), "known-window MoE-combine rows launch");
}
void moe_combine_known_window_testing_launch(const ElementwiseRowsViews& views, std::uint32_t hidden_size) {
    if (views.count == 0U || views.count > 8U || hidden_size == 0U)
        throw std::invalid_argument("known-window MoE-combine testing rows are invalid");
    ElementwiseRowsArgs args{};
    for (std::uint32_t r = 0; r < views.count; ++r) {
        args.first[r] = views.first[r]; args.second[r] = views.second[r]; args.output[r] = views.output[r];
    }
    hipLaunchKernelGGL(HIP_KERNEL_NAME(combine_rows_kernel<std::uint16_t>), dim3(prepare_1d_blocks(hidden_size), views.count), dim3(256U), 0,
        elementwise_rows_stream(), args, hidden_size);
    check_hip(hipGetLastError(), "known-window MoE-combine testing rows launch");
}

}  // namespace strix
