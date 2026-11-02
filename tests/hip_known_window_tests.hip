#include "strix/dispatch_fuse.hpp"
#include "strix/dispatch_resources.hpp"
#include "strix/execution_arena.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_plan.hpp"
#include "strix/hip_gdn_launcher.hpp"
#include "strix/hip_generation_output.hpp"
#include "strix/hip_lm_head_launcher.hpp"
#include "strix/hip_hyper_mix_launcher.hpp"
#include "strix/hip_moe_experts_launcher.hpp"
#include "strix/hip_indexed_attention_launcher.hpp"
#include "strix/indexed_attention_state.hpp"
#include "strix/indexed_attention_workspace.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/execution_buffers.hpp"
#include "strix/hip_moe_route_launcher.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/wave_router_contract.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/hip_shared_expert_launcher.hpp"
#include "strix/q6p_registry.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/session_state_memory.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <tuple>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using namespace strix;
void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
}
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
struct Span { std::uintptr_t address = 0; std::uint64_t bytes = 0; };
struct Owner {
    HipSetupQuiescenceGuard quiescence;
    hipStream_t stream{};
    std::vector<void*> allocations;
    std::vector<hipGraph_t> graphs;
    Owner() { quiescence.run([&] { check(hipStreamCreate(&stream), "create known-window stream"); }); }
    ~Owner() {
        quiescence.recover_or_fail_stop();
        const auto cleanup = [](hipError_t error) { if (error != hipSuccess) std::_Exit(3); };
        for (const auto graph : graphs) cleanup(hipGraphDestroy(graph));
        for (const auto allocation : allocations) cleanup(hipFree(allocation));
        cleanup(hipStreamDestroy(stream));
    }
    Span allocate(std::uint64_t bytes) {
        require(bytes && bytes < (1ULL << 30U), "fixture allocation exceeds its one-GiB bound");
        void* pointer = nullptr;
        check(hipMalloc(&pointer, static_cast<std::size_t>(bytes + 512U)), "allocate guarded known-window fixture");
        try { allocations.push_back(pointer); }
        catch (...) { quiescence.recover_or_fail_stop(); (void)hipFree(pointer); throw; }
        check(hipMemset(pointer, 0xa5, static_cast<std::size_t>(bytes + 512U)), "poison fixture guards");
        return {reinterpret_cast<std::uintptr_t>(pointer) + 256U, bytes};
    }
    void finish() { check(hipStreamSynchronize(stream), "complete known-window fixture"); quiescence.require_original(); }
};
std::vector<unsigned char> read(Span span) {
    std::vector<unsigned char> bytes(static_cast<std::size_t>(span.bytes));
    check(hipMemcpy(bytes.data(), reinterpret_cast<const void*>(span.address), bytes.size(), hipMemcpyDeviceToHost), "read fixture bytes");
    return bytes;
}
void write(Span span, const std::vector<unsigned char>& bytes) {
    require(bytes.size() == span.bytes, "fixture upload extent differs");
    check(hipMemcpy(reinterpret_cast<void*>(span.address), bytes.data(), bytes.size(), hipMemcpyHostToDevice), "upload fixture bytes");
}
void guards(Span span) {
    for (const auto address : {span.address - 256U, span.address + span.bytes}) {
        const auto bytes = read({address, 256U});
        require(std::all_of(bytes.begin(), bytes.end(), [](auto value) { return value == 0xa5; }), "fixture exterior canary changed");
    }
}
template<class T> std::vector<unsigned char> bytes_of(const std::vector<T>& values) {
    std::vector<unsigned char> bytes(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}
std::uint16_t bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}
struct Q8 { __half scale; std::int8_t codes[32]; };
static_assert(sizeof(Q8) == 34);
std::vector<unsigned char> q8(std::uint32_t width, std::uint32_t outputs, unsigned salt) {
    std::vector<Q8> blocks(static_cast<std::size_t>(width / 32U) * outputs);
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        blocks[index].scale = __float2half(0.0005F * static_cast<float>(1U + (index + salt) % 5U));
        for (unsigned code = 0; code < 32U; ++code)
            blocks[index].codes[code] = static_cast<std::int8_t>(static_cast<int>((index * 7U + code * 3U + salt) % 13U) - 6);
    }
    return bytes_of(blocks);
}
std::vector<unsigned char> f32(std::size_t size, float base, float variation = 0.0F) {
    std::vector<float> values(size);
    for (std::size_t index = 0; index < size; ++index)
        values[index] = base + variation * static_cast<float>(static_cast<int>(index % 7U) - 3);
    return bytes_of(values);
}
struct Weight { Span device; std::vector<unsigned char> original; };
struct Result {
    std::array<std::vector<unsigned char>, 2> output, temporary;
    std::vector<unsigned char> matrix, convolution;
};

// One real selected operator, two C1 row views, original canonical model
// geometry. No decoder/session execution is inferred from this component test.
struct Fixture {
    Owner owner; // retained until every stream operation and borrowed owner is drained
    const bool gdn;
    const bool f32_gates;
    const std::uint32_t context_capacity;
    ModelContract model = ModelContract::qwen38_flash_next();
    SerialExecutionPlan execution = build_serial_decode_execution_plan(model);
    SerialExecutionIoPlan io = build_serial_execution_io_plan(execution);
    ExecutionStep step;
    ExecutionArenaPlan serial;
    KnownWindowArena window;
    Span scratch, state;
    ExecutionMemoryBinding memory;
    ModelManifest manifest;
    ModelRuntimeBinding runtime;
    ExecutionTensorAccessPlan access;
    std::vector<Weight> weights;
    SerialSessionStateLayout state_layout;
    SessionStateMemoryBinding state_memory;
    std::vector<unsigned char> host_state;
    std::vector<unsigned char> initial_state, seed;
    std::array<ExecutionRuntimeResources, 2> resources;
    std::array<ExecutionDispatchContext, 2> rows;
    std::array<ExecutionBufferAddress, 2> input{}, output{}, temporary{};
    HipSharedExpertLauncherConfig shared_config;
    HipGdnLauncherConfig gdn_config;
    std::string selected_id;
    ExecutionLauncher selected;
    Span matrix, convolution;

    explicit Fixture(bool is_gdn, bool shared_prefetch = false, bool mixed_gates = false,std::uint32_t context_tokens=4U)
        : gdn(is_gdn), f32_gates(mixed_gates),context_capacity(context_tokens) {
        const auto wanted = gdn ? SerialOp::gated_delta_net : SerialOp::shared_expert;
        const auto found = std::find_if(execution.steps.begin(), execution.steps.end(),
            [&](const auto& value) { return value.op == wanted && value.layer == 0U; });
        require(found != execution.steps.end(), "canonical layer-zero operator missing");
        step = *found;
        const auto& step_io = io.steps[step.id];
        require(step_io.reads.size() == 1U && step_io.writes.size() == 1U && step_io.temporaries.size() == 1U,
            "fixture must represent one selected row IO contract");
        const std::uint64_t workspace_bytes = gdn ? (10240ULL + 6144ULL + 96ULL + 6144ULL) * 4U : 640ULL * 8U;
        serial.alignment = 256U;
        const auto allocation = [&](ExecutionBufferUse use, std::uint64_t bytes) {
            const auto offset = (serial.total_bytes + 255U) / 256U * 256U;
            serial.allocations.push_back({use, step.id, step.id, offset, bytes});
            serial.total_bytes = offset + bytes;
        };
        allocation(step_io.reads[0], 2560U * 2U);
        allocation(step_io.writes[0], 2560U * 2U);
        allocation(step_io.temporaries[0], workspace_bytes);
        serial.total_bytes = (serial.total_bytes + 255U) / 256U * 256U;
        serial.naive_nonaliased_bytes = serial.total_bytes;
        window = build_known_window_arena(serial, 256U);
        scratch = owner.allocate(window.physical.total_bytes);
        memory = {0x4b4e4f574e523201ULL, scratch.address, scratch.bytes, 19U};
        manifest.architecture = "qwen4_exp";
        if (gdn) setup_gdn(); else setup_shared(shared_prefetch);
        manifest.layout_identity = model_manifest_layout_identity(manifest);
        runtime.manifest_layout_identity = manifest.layout_identity;
        runtime.model_content_identity.bytes[0] = 1U;
        access.manifest_layout_identity = manifest.layout_identity;
        access.plan_identity = 0x4b4e4f574e523202ULL;
        for (unsigned row = 0; row < 2U; ++row) {
            resources[row] = {&memory, &manifest, &runtime, &access};
            if (gdn) {
                resources[row].session_state_layout = &state_layout;
                resources[row].session_state_memory = &state_memory;
            }
            auto& context = rows[row];
            context.execution_identity = memory.execution_identity;
            context.step_id = step.id; context.op = step.op; context.layer = step.layer;
            context.kernel_id = selected_id;
            context.reads = step_io.reads; context.writes = step_io.writes; context.temporaries = step_io.temporaries;
            context.arena = &window.row_view[row]; context.memory = &memory; context.resources = &resources[row];
            context.session_state_layout = gdn ? &state_layout : nullptr;
            context.session_state_memory = gdn ? &state_memory : nullptr;
            context.reads_state = step.reads_state; context.writes_state = step.writes_state;
            context.position = row; context.input_token_id = 17U + row;
            input[row] = resolve_dispatch_read(context, 0);
            output[row] = resolve_dispatch_write(context, 0);
            temporary[row] = resolve_dispatch_temporary(context, 0);
        }
    }
    void add(std::string name, TensorRole role, std::uint32_t type,
             std::vector<std::uint64_t> dimensions, std::vector<unsigned char> bytes) {
        ManifestTensor tensor;
        tensor.name = std::move(name); tensor.role = role; tensor.layer = step.layer;
        tensor.ggml_type = type; tensor.dimensions = std::move(dimensions); tensor.byte_size = bytes.size();
        tensor.absolute_offset = (manifest.total_tensor_bytes + 15U) / 16U * 16U;
        manifest.total_tensor_bytes += tensor.byte_size;
        const auto device = owner.allocate(bytes.size());
        write(device, bytes);
        const auto index = static_cast<std::uint32_t>(weights.size());
        weights.push_back({device, std::move(bytes)});
        manifest.tensors.push_back(std::move(tensor));
        runtime.tensors.push_back({index, RuntimeTensorStorage::resident, device.address, device.bytes, 7U, 0U});
    }
    void setup_shared(bool prefetch) {
        add("blk.0.shared_expert.gate_proj.weight", TensorRole::dense_projection, 8U, {2560, 640}, q8(2560, 640, 1));
        add("blk.0.shared_expert.up_proj.weight", TensorRole::dense_projection, 8U, {2560, 640}, q8(2560, 640, 2));
        add("blk.0.shared_expert.down_proj.weight", TensorRole::dense_projection, 8U, {640, 2560}, q8(640, 2560, 3));
        add("blk.0.shared_expert_gate.weight", TensorRole::dense_projection, 0U, {2560}, f32(2560, 0.0002F, 0.00003F));
        access.accesses.push_back({step.id, step.op, 0U, {0, 1, 2, 3}});
        shared_config.hidden_size = 2560U; shared_config.intermediate_size = 640U;
        shared_config.prefetch_q8 = prefetch;
        if (prefetch) {
            shared_config.prefetch_eligible_layers = {0U};
            shared_config.prefetch_stats = std::make_shared<shared_expert_prefetch::RuntimeDispatchStats>();
        }
        kernels::KernelRegistry registry;
        ExecutionBindingRequest binding;
        binding.prefetch_q8_shared_expert = prefetch;
        binding.shared_expert_prefetch_layers = shared_config.prefetch_eligible_layers;
        register_hip_shared_expert_kernel_variant(model, binding, shared_config, registry);
        require(registry.variants().size() == 1U, "shared serial selection is ambiguous");
        selected_id = registry.variants().front().id;
        selected = make_hip_shared_expert_launcher(selected_id, &shared_config);
    }
    void setup_gdn() {
        add("blk.0.attn_qkv.weight", TensorRole::attention_projection, 8U, {2560, 10240}, q8(2560, 10240, 4));
        add("blk.0.attn_gate.weight", TensorRole::attention_projection, 8U, {2560, 6144}, q8(2560, 6144, 5));
        add("blk.0.ssm_conv1d.weight", TensorRole::recurrent_state, 0U, {4, 10240}, f32(4U * 10240U, 0.2F, 0.015F));
        add("blk.0.ssm_dt.bias", TensorRole::recurrent_state, 0U, {48}, f32(48U, -0.1F, 0.01F));
        add("blk.0.ssm_a", TensorRole::recurrent_state, 0U, {48}, f32(48U, -0.25F, 0.01F));
        add("blk.0.ssm_beta.weight", TensorRole::recurrent_state, f32_gates ? 0U : 8U, {2560, 48},
            f32_gates ? f32(2560U * 48U, 0.0005F, 0.0001F) : q8(2560, 48, 6));
        add("blk.0.ssm_alpha.weight", TensorRole::recurrent_state, f32_gates ? 0U : 8U, {2560, 48},
            f32_gates ? f32(2560U * 48U, -0.0004F, 0.0001F) : q8(2560, 48, 7));
        add("blk.0.ssm_norm.weight", TensorRole::recurrent_state, 0U, {128}, f32(128U, 0.75F, 0.01F));
        add("blk.0.ssm_out.weight", TensorRole::recurrent_state, 8U, {6144, 2560}, q8(6144, 2560, 8));
        access.accesses.push_back({step.id, step.op, 0U, {2, 3, 4, 5, 6, 7, 8}});
        access.accesses.push_back({step.id, step.op, 1U, {0, 1}});
        SerialSessionStateLayoutRequest request; request.context_tokens = context_capacity;
        state_layout = build_serial_session_state_layout(model, request);
        state = owner.allocate(state_layout.device_total_bytes);
        host_state.resize(static_cast<std::size_t>(state_layout.host_total_bytes));
        state_memory = {state_layout.layout_identity, state.address, state.bytes, 23U,
            reinterpret_cast<std::uintptr_t>(host_state.data()), state_layout.host_total_bytes, 29U};
        initial_state.resize(static_cast<std::size_t>(state.bytes), 0U);
        const auto m = resolve_gdn_layer_state(state_layout, state_memory, model, 0U, SessionStateComponent::recurrent_matrix);
        const auto c = resolve_gdn_layer_state(state_layout, state_memory, model, 0U, SessionStateComponent::gdn_convolution);
        matrix = {m.address, m.bytes}; convolution = {c.address, c.bytes};
        const auto initial_matrix = f32(static_cast<std::size_t>(matrix.bytes / 4U), 0.002F, 0.0001F);
        std::memcpy(initial_state.data() + (matrix.address - state.address), initial_matrix.data(), initial_matrix.size());
        std::vector<std::uint16_t> initial_conv(static_cast<std::size_t>(convolution.bytes / 2U));
        for (std::size_t i = 0; i < initial_conv.size(); ++i) initial_conv[i] = bf16(0.01F * static_cast<float>(1U + i % 9U));
        std::memcpy(initial_state.data() + (convolution.address - state.address), initial_conv.data(), initial_conv.size() * 2U);
        gdn_config.model = model; gdn_config.virtual_q8_input_projections = true;
        gdn_config.prefetch_q8_input_projections = true; gdn_config.wave8_q8_input_projections = true;
        gdn_config.virtual_q8_output_projection = true; gdn_config.wave8_q8_output_projection = true;
        ExecutionBindingRequest binding;
        binding.wave8_q8_gdn_input = true; binding.wave8_q8_gdn_output = true;
        kernels::KernelRegistry registry;
        register_hip_gdn_kernel_variant(model, binding, gdn_config, registry);
        require(registry.variants().size() == 1U, "GDN serial selection is ambiguous");
        selected_id = registry.variants().front().id;
        selected = make_hip_gdn_launcher(selected_id, &gdn_config);
    }
    HipLaunchContext scope() const { HipLaunchContext result; result.stream = owner.stream; return result; }
    void preflight(std::span<const ExecutionDispatchContext, 2> contexts) const {
        if (gdn && f32_gates) gdn_known_window_r2_f32_gates_preflight(contexts, gdn_config);
        else if (gdn) gdn_known_window_r2_preflight(contexts, gdn_config);
        else shared_expert_known_window_r2_preflight(contexts, shared_config);
    }
    void launch(std::span<const ExecutionDispatchContext, 2> contexts) const {
        if (gdn && f32_gates) gdn_known_window_r2_f32_gates_launch(contexts, gdn_config);
        else if (gdn) gdn_known_window_r2_launch(contexts, gdn_config);
        else shared_expert_known_window_r2_launch(contexts, shared_config);
    }
    void prepare(unsigned flavor, bool swapped = false) {
        seed.assign(static_cast<std::size_t>(scratch.bytes), 0xa5);
        for (unsigned row = 0; row < 2U; ++row) {
            const auto source_row = swapped ? 1U - row : row;
            std::vector<std::uint16_t> values(2560U);
            constexpr std::array<std::uint16_t, 12> edge{0, 0x8000, 1, 0x8001, 0x007f, 0x807f,
                0x7f80, 0xff80, 0x7fc1, 0xffc2, 0x7f81, 0xff83};
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (flavor == 4U) values[i] = i % 32U == 3U ? static_cast<std::uint16_t>(source_row ? 0xff83U : 0x7f81U) :
                    i % 32U == 7U ? static_cast<std::uint16_t>(source_row ? 0xffc3U : 0x7fc1U) :
                    bf16((i == 255U || i == 256U || i == 2559U) ? 2.0F :
                        (i % 2U ? -1.0F : 1.0F) * (source_row ? 0.125F : 0.25F));
                else if (flavor == 1U) values[i] = edge[(i + source_row * 3U) % 6U];
                else if (flavor == 3U) values[i] = edge[(i + source_row * 5U) % edge.size()];
                else if (flavor == 2U) values[i] = bf16((i % 2U ? -1.0F : 1.0F) * (source_row ? 0.125F : 0.25F));
                else values[i] = bf16(static_cast<float>(static_cast<int>((i * 7U + source_row * 11U) % 29U) - 14) * (source_row ? -0.043F : 0.031F));
            }
            std::memcpy(seed.data() + (input[row].address - scratch.address), values.data(), values.size() * 2U);
        }
    }
    void reset() { write(scratch, seed); if (gdn) write(state, initial_state); }
    void readonly() const {
        const auto actual = read(scratch);
        auto expected = seed;
        for (const auto& writes : {output, temporary})
            for (const auto view : writes)
                std::copy_n(actual.begin() + static_cast<std::ptrdiff_t>(view.address - scratch.address),
                    static_cast<std::size_t>(view.bytes), expected.begin() + static_cast<std::ptrdiff_t>(view.address - scratch.address));
        require(actual == expected, "readonly row input or interior arena canary changed");
        guards(scratch);
        for (const auto& weight : weights) {
            require(read(weight.device) == weight.original, "known-window weights changed");
            guards(weight.device);
        }
        if (gdn) {
            auto expected_state = initial_state;
            const auto actual_state = read(state);
            for (const auto view : {matrix, convolution})
                std::copy_n(actual_state.begin() + static_cast<std::ptrdiff_t>(view.address - state.address),
                    static_cast<std::size_t>(view.bytes), expected_state.begin() + static_cast<std::ptrdiff_t>(view.address - state.address));
            require(actual_state == expected_state, "GDN modified another layer or state component");
            guards(state);
        }
    }
    Result run(bool candidate) {
        reset();
        const auto before = shared_config.prefetch_stats ?
            shared_config.prefetch_stats->dispatched_projections.load(std::memory_order_relaxed) : 0U;
        {
            const HipLaunchScope owned(scope());
            if (candidate) { preflight(rows); launch(rows); }
            else for (const auto& row : rows) selected.function(row, selected.user_data);
        }
        owner.finish();
        if (shared_config.prefetch_stats)
            require(shared_config.prefetch_stats->dispatched_projections.load(std::memory_order_relaxed) - before == (candidate ? 3U : 6U),
                "shared R2 actual projection submission count differs");
        Result result;
        for (unsigned row = 0; row < 2U; ++row) {
            result.output[row] = read({output[row].address, output[row].bytes});
            result.temporary[row] = read({temporary[row].address, temporary[row].bytes});
        }
        if (gdn) { result.matrix = read(matrix); result.convolution = read(convolution); }
        readonly();
        return result;
    }
    // Shared rows on a second stream with a private workspace (the window owner's detached form): the
    // outputs and the workspace rows equal the arena run's; the arena workspace is not touched.
    Result run_detached() {
        reset();
        const auto stride = temporary[1].address - temporary[0].address;
        const auto workspace = owner.allocate(stride + temporary[1].bytes);
        hipStream_t side = nullptr;
        check(hipStreamCreateWithFlags(&side, hipStreamNonBlocking), "create detached shared stream");
        try {
            shared_expert_known_window_rows_launch_detached(rows, shared_config, side, workspace.address, workspace.bytes);
            check(hipStreamSynchronize(side), "complete detached shared rows");
        } catch (...) { (void)hipStreamSynchronize(side); (void)hipStreamDestroy(side); throw; }
        check(hipStreamDestroy(side), "destroy detached shared stream");
        Result result;
        for (unsigned row = 0; row < 2U; ++row) {
            result.output[row] = read({output[row].address, output[row].bytes});
            result.temporary[row] = read({workspace.address + row * stride, temporary[row].bytes});
            require(read({temporary[row].address, temporary[row].bytes}) ==
                std::vector<unsigned char>(seed.begin() + static_cast<std::ptrdiff_t>(temporary[row].address - scratch.address),
                    seed.begin() + static_cast<std::ptrdiff_t>(temporary[row].address - scratch.address + temporary[row].bytes)),
                "detached shared rows wrote the arena workspace");
        }
        readonly();
        guards(workspace);
        return result;
    }
};
void same(const Result& left, const Result& right) {
    require(left.output == right.output, "R2 output rows differ bitwise from two selected serial calls");
    require(left.temporary == right.temporary, "R2 complete row temporaries differ bitwise from selected serial calls");
    require(left.matrix == right.matrix, "R2 complete recurrent matrix differs from serial after two rows");
    require(left.convolution == right.convolution, "R2 complete convolution history differs from serial after two rows");
}
void finite(const Result& result) {
    const auto bf16_values = [](const std::vector<unsigned char>& bytes) {
        require(bytes.size() % 2U == 0U, "BF16 result has partial element");
        for (std::size_t offset = 0; offset < bytes.size(); offset += 2U) {
            std::uint16_t bits{};
            std::memcpy(&bits, bytes.data() + offset, sizeof(bits));
            require((bits & 0x7f80U) != 0x7f80U, "finite fixture produced a nonfinite BF16 value");
        }
    };
    for (const auto& output : result.output) bf16_values(output);
    bf16_values(result.convolution);
    require(result.matrix.size() % 4U == 0U, "FP32 matrix has partial element");
    for (std::size_t offset = 0; offset < result.matrix.size(); offset += 4U) {
        float value{};
        std::memcpy(&value, result.matrix.data() + offset, sizeof(value));
        require(std::isfinite(value), "finite fixture produced a nonfinite recurrent state");
    }
}
template<class Work> void invalid(Work&& work) {
    bool rejected = false;
    try { work(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid known-window admission was accepted");
}
void capture_rejection(Fixture& fixture, std::span<const ExecutionDispatchContext, 2> rows) {
    check(hipStreamBeginCapture(fixture.owner.stream, hipStreamCaptureModeThreadLocal), "begin rejection capture");
    std::exception_ptr error;
    try { const HipLaunchScope owned(fixture.scope()); invalid([&] { fixture.launch(rows); }); }
    catch (...) { error = std::current_exception(); }
    hipGraph_t graph{};
    check(hipStreamEndCapture(fixture.owner.stream, &graph), "end rejection capture");
    require(graph != nullptr, "rejection capture did not return an empty graph");
    try { fixture.owner.graphs.push_back(graph); }
    catch (...) { (void)hipGraphDestroy(graph); throw; }
    std::size_t nodes = 0;
    check(hipGraphGetNodes(graph, nullptr, &nodes), "count rejection capture nodes");
    require(nodes == 0U, "rejected known-window call enqueued work");
    if (error) std::rethrow_exception(error);
}
void rejections(Fixture& fixture) {
    fixture.prepare(0U); fixture.reset();
    const auto original_arena = read(fixture.scratch);
    const auto original_state = fixture.gdn ? read(fixture.state) : std::vector<unsigned char>{};
    const auto original_count = fixture.shared_config.prefetch_stats ?
        fixture.shared_config.prefetch_stats->dispatched_projections.load(std::memory_order_relaxed) : 0U;
    const auto bad = [&](auto mutate) {
        auto rows = fixture.rows;
        mutate(rows);
        { const HipLaunchScope owned(fixture.scope()); invalid([&] { fixture.preflight(rows); }); }
        capture_rejection(fixture, rows);
        fixture.owner.finish();
        require(read(fixture.scratch) == original_arena, "rejected R2 call changed scratch bytes");
        if (fixture.gdn) require(read(fixture.state) == original_state, "rejected R2 call changed state bytes");
    };
    bad([](auto& rows) { rows[1].execution_identity ^= 1U; });
    bad([](auto& rows) { rows[1].position += 1U; });
    bad([](auto& rows) { rows[1].logical_rows = 2U; });
    bad([](auto& rows) { rows[1].temporaries = {}; });
    const auto same_values_other_object = fixture.memory;
    bad([&](auto& rows) { rows[1].memory = &same_values_other_object; });
    auto stale = fixture.memory; stale.allocation_generation = 0U;
    auto stale_resources = fixture.resources[1]; stale_resources.scratch = &stale;
    bad([&](auto& rows) { rows[1].memory = &stale; rows[1].resources = &stale_resources; });
    auto alias = fixture.window.row_view[1];
    for (auto& allocation : alias.allocations)
        if (allocation.use == fixture.rows[1].writes[0]) allocation.offset = fixture.input[0].address - fixture.scratch.address;
    bad([&](auto& rows) { rows[1].arena = &alias; });
    auto stride = fixture.window.row_view[1];
    for (auto& allocation : stride.allocations)
        if (allocation.use == fixture.rows[1].writes[0]) allocation.offset += 16U;
    bad([&](auto& rows) { rows[1].arena = &stride; });
    const auto weight_generation = fixture.runtime.tensors[0].allocation_generation;
    fixture.runtime.tensors[0].allocation_generation = 0U;
    bad([](auto&) {});
    fixture.runtime.tensors[0].allocation_generation = weight_generation;
    if (fixture.gdn) {
        const auto same_state_other_object = fixture.state_memory;
        auto changed_resources = fixture.resources[1];
        changed_resources.session_state_memory = &same_state_other_object;
        bad([&](auto& rows) { rows[1].resources = &changed_resources; });
    }
    // Valid metadata still cannot enter the eager-only wrapper during capture.
    capture_rejection(fixture, fixture.rows);
    fixture.owner.finish();
    require(read(fixture.scratch) == original_arena, "capture rejection changed scratch");
    if (fixture.gdn) require(read(fixture.state) == original_state, "capture rejection changed state");
    if (fixture.shared_config.prefetch_stats)
        require(fixture.shared_config.prefetch_stats->dispatched_projections.load(std::memory_order_relaxed) == original_count,
            "rejected shared R2 call incremented submission count");
    fixture.readonly();
}

// Mutate only the selected original-format Q8 owner. Raw HALF bits are copied,
// never converted on the CPU: signaling/quiet payloads and signed subnormals
// reach the actual device decoder. Keep the readonly oracle in step with the
// intentional mutation, then restore both host oracle and device bytes even
// when a bitwise comparison throws. Drain before any restoration can overwrite
// a weight still borrowed by an unsuccessful launch.
template<class Work> void adversarial_q8_weight(Fixture& fixture, unsigned index,
    std::uint32_t width, std::uint32_t outputs, Work&& work) {
    auto& weight = fixture.weights[index];
    auto saved = weight.original;
    auto mutated = saved;
    const auto blocks = width / 32U;
    require(width % 32U == 0U && blocks && outputs >= 24U &&
        mutated.size() == static_cast<std::size_t>(blocks) * outputs * sizeof(Q8),
        "adversarial Q8 fixture shape differs from original owner");
    constexpr std::array<std::uint16_t, 12> scales{0x0000, 0x8000, 0x0001, 0x8001,
        0x03ff, 0x83ff, 0x7c00, 0xfc00, 0x7e11, 0xfe35, 0x7c03, 0xfc07};
    constexpr std::array<std::int8_t, 8> codes{0, 1, -1, 127, -128, 7, -7, 0};
    for (std::size_t pattern = 0; pattern < scales.size(); ++pattern) {
        for (unsigned end = 0; end < 2U; ++end) {
            const auto feature = end ? static_cast<std::size_t>(outputs) - 1U - pattern : pattern;
            const auto block = (pattern * 7U + end * 13U) % blocks;
            const auto offset = (feature * blocks + block) * sizeof(Q8);
            std::memcpy(mutated.data() + offset, &scales[pattern], sizeof(scales[pattern]));
            for (unsigned code = 0; code < 32U; ++code) {
                const auto value = codes[(code + pattern + end) % codes.size()];
                std::memcpy(mutated.data() + offset + 2U + code, &value, sizeof(value));
            }
        }
    }
    weight.original.swap(mutated);
    const auto restore = [&] {
        weight.original.swap(saved);
        fixture.owner.quiescence.run([&] { write(weight.device, weight.original); });
        require(read(weight.device) == weight.original, "original Q8 owner was not restored exactly");
        guards(weight.device);
    };
    try {
        write(weight.device, weight.original);
        work();
        fixture.owner.finish();
    } catch (...) {
        const auto failure = std::current_exception();
        fixture.owner.quiescence.recover_or_fail_stop();
        restore();
        std::rethrow_exception(failure);
    }
    restore();
}

// Observe complete projection intermediates too: comparing only the final BF16
// operator output could hide a changed input projection behind later rounding.
void gdn_projection_rows(Fixture& fixture) {
    using View = GdnKnownWindowProjectionView;
    using Direction = GdnKnownWindowProjectionDirection;
    using Mode = GdnKnownWindowProjectionMode;
    struct Case { unsigned weight; std::uint32_t width, outputs; Direction direction; };
    const std::array<Case, 5> cases{{{0U, 2560U, 10240U, Direction::bf16_to_fp32},
        {1U, 2560U, 6144U, Direction::bf16_to_fp32}, {5U, 2560U, 48U, Direction::bf16_to_fp32},
        {6U, 2560U, 48U, Direction::bf16_to_fp32}, {8U, 6144U, 2560U, Direction::fp32_to_bf16}}};
    for (const auto item : cases) {
        const bool input_bf16 = item.direction == Direction::bf16_to_fp32;
        const auto in_bytes = static_cast<std::uint64_t>(item.width) * (input_bf16 ? 2U : 4U);
        const auto out_bytes = static_cast<std::uint64_t>(item.outputs) * (input_bf16 ? 4U : 2U);
        const std::array<Span, 2> inputs{fixture.owner.allocate(in_bytes), fixture.owner.allocate(in_bytes)};
        const std::array<Span, 2> outputs{fixture.owner.allocate(out_bytes), fixture.owner.allocate(out_bytes)};
        const std::array<View, 2> input_views{{{inputs[0].address, inputs[0].bytes}, {inputs[1].address, inputs[1].bytes}}};
        const std::array<View, 2> output_views{{{outputs[0].address, outputs[0].bytes}, {outputs[1].address, outputs[1].bytes}}};
        const auto weight = fixture.weights[item.weight].device;
        const auto compare = [&](unsigned flavor) {
            std::array<std::vector<unsigned char>, 2> original;
            for (unsigned row = 0; row < 2U; ++row) {
                std::vector<std::uint16_t> half(item.width);
                std::vector<float> full(item.width);
                constexpr std::array<std::uint16_t, 12> edge{0, 0x8000, 1, 0x8001, 0x7f80, 0xff80,
                    0x7fc1, 0xffc3, 0x7f81, 0xff83, 0x7fc5, 0xffc7};
                for (std::uint32_t i = 0; i < item.width; ++i) {
                    const auto bits = flavor == 4U ?
                        (i % 32U == 3U ? static_cast<std::uint16_t>(row ? 0xff83U : 0x7f81U) :
                         i % 32U == 7U ? static_cast<std::uint16_t>(row ? 0xffc3U : 0x7fc1U) :
                         bf16((i % 2U ? -1.0F : 1.0F) * (row ? 0.125F : 0.25F))) :
                        flavor == 3U ? edge[(i + row * 3U) % edge.size()] :
                        flavor == 1U ? edge[(i + row) % 4U] :
                        bf16((flavor == 2U ? (i % 2U ? -0.25F : 0.25F) :
                            static_cast<float>(static_cast<int>((i * 3U + row * 7U) % 23U) - 11) * 0.031F));
                    half[i] = bits;
                    full[i] = std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
                }
                original[row] = input_bf16 ? bytes_of(half) : bytes_of(full);
                write(inputs[row], original[row]);
            }
            const auto run = [&](Mode mode) {
                for (const auto output : outputs)
                    check(hipMemset(reinterpret_cast<void*>(output.address), 0xa5, static_cast<std::size_t>(output.bytes)), "poison raw projection output");
                gdn_known_window_r2_projection_probe(item.direction, mode, input_views,
                    {weight.address, weight.bytes}, output_views, item.width, item.outputs, fixture.owner.stream);
                fixture.owner.finish();
                std::array<std::vector<unsigned char>, 2> result;
                for (unsigned row = 0; row < 2U; ++row) {
                    result[row] = read(outputs[row]);
                    require(read(inputs[row]) == original[row], "raw projection mutated its independent row input");
                    guards(inputs[row]); guards(outputs[row]);
                }
                return result;
            };
            const auto reference = run(Mode::incumbent_row_loop);
            require(run(Mode::rows2) == reference, "GDN R2 complete projection intermediates differ bitwise");
            require(run(Mode::rows2) == reference, "GDN R2 projection is not bitwise repeatable");
            require(read(weight) == fixture.weights[item.weight].original, "raw projection changed packed weights");
            guards(weight);
        };
        for (unsigned flavor = 0; flavor < 4U; ++flavor) compare(flavor);
        // Exactly one additional decoder-header pattern per actual projection,
        // with independent cancellation/NaN-bearing row inputs. Full F32/BF16
        // payload equality remains mandatory, including every NaN payload bit.
        adversarial_q8_weight(fixture, item.weight, item.width, item.outputs, [&] { compare(4U); });
    }
}

// Typed stage observation keeps NaN payloads visible before dt/sigmoid overwrite
// beta/gate. Synthetic canonical geometry only, not real-weight/model evidence.
void gdn_typed_input_stage(Fixture& fixture) {
    constexpr std::uint64_t prefix_bytes = 16528ULL * sizeof(float);
    const auto run = [&](GdnKnownWindowProjectionMode mode) {
        fixture.reset();
        {
            const HipLaunchScope scope(fixture.scope());
            gdn_known_window_r2_input_stage_probe(fixture.rows, fixture.gdn_config,
                fixture.f32_gates ? GdnKnownWindowFormat::q8_rows2_f32_gates : GdnKnownWindowFormat::q8_all5, mode);
        }
        fixture.owner.finish();
        auto expected = fixture.seed;
        std::array<std::vector<unsigned char>,2> result;
        for (unsigned row = 0; row < 2U; ++row) {
            require(fixture.temporary[row].bytes >= prefix_bytes, "typed input prefix exceeds temporary");
            result[row] = read({fixture.temporary[row].address, prefix_bytes});
            std::copy(result[row].begin(), result[row].end(), expected.begin() +
                static_cast<std::ptrdiff_t>(fixture.temporary[row].address - fixture.scratch.address));
        }
        require(read(fixture.scratch) == expected,
            "input-stage probe touched core, row outputs, input or non-prefix scratch bytes");
        require(read(fixture.state) == fixture.initial_state, "input-stage probe touched persistent state");
        fixture.readonly();
        return result;
    };
    for (unsigned flavor = 0; flavor < 5U; ++flavor) {
        fixture.prepare(flavor);
        const auto reference = run(GdnKnownWindowProjectionMode::incumbent_row_loop);
        require(run(GdnKnownWindowProjectionMode::rows2) == reference,
            "typed R2 complete 16528-F32 input prefix differs bitwise from both incumbent rows");
        require(run(GdnKnownWindowProjectionMode::rows2) == reference,
            "typed R2 input prefix is not bitwise repeatable");
    }
}

template<class Work> void adversarial_f32_gates(Fixture& fixture, Work&& work) {
    require(fixture.f32_gates, "F32 adversary requires actual F32 gate owners");
    const std::array<std::vector<unsigned char>,2> saved{fixture.weights[5].original, fixture.weights[6].original};
    constexpr std::array<std::uint32_t,16> patterns{0x00000000U,0x80000000U,0x00000001U,0x80000001U,
        0x007fffffU,0x807fffffU,0x00800000U,0x80800000U,0x7f800000U,0xff800000U,
        0x7fc00001U,0xffc12345U,0x7f800001U,0xff812345U,0x7f7fffffU,0xff7fffffU};
    constexpr std::array<unsigned,4> outputs{0,1,46,47}, features{0,255,256,2559};
    for (unsigned gate = 0; gate < 2U; ++gate) {
        auto& target = fixture.weights[5U + gate];
        require(target.original.size() == 491520U, "F32 gate fixture extent differs");
        for (unsigned o = 0; o < outputs.size(); ++o)
            for (unsigned f = 0; f < features.size(); ++f) {
                const auto bits = patterns[(o * 4U + f + gate * 5U) % patterns.size()];
                const auto offset = (static_cast<std::size_t>(outputs[o]) * 2560U + features[f]) * 4U;
                std::memcpy(target.original.data() + offset, &bits, sizeof(bits));
            }
        for (const auto output : outputs) {
            constexpr std::array<std::uint32_t,2> cancellation{0x3f800000U,0xbf800000U};
            std::memcpy(target.original.data() + (static_cast<std::size_t>(output) * 2560U + 64U) * 4U,
                cancellation.data(), sizeof(cancellation));
        }
    }
    const auto restore = [&] {
        for (unsigned gate = 0; gate < 2U; ++gate) {
            fixture.weights[5U + gate].original = saved[gate];
            fixture.owner.quiescence.run([&] { write(fixture.weights[5U + gate].device, saved[gate]); });
        }
    };
    try {
        for (const auto index : {5U,6U}) write(fixture.weights[index].device, fixture.weights[index].original);
        work(); fixture.owner.finish();
    } catch (...) {
        const auto error = std::current_exception();
        fixture.owner.quiescence.recover_or_fail_stop(); restore(); std::rethrow_exception(error);
    }
    restore();
}

void wrong_gdn_entry(Fixture& fixture) {
    fixture.prepare(0U); fixture.reset();
    const auto scratch = read(fixture.scratch), state = read(fixture.state);
    const auto call = [&](bool launch) {
        if (fixture.f32_gates) {
            if (launch) gdn_known_window_r2_launch(fixture.rows, fixture.gdn_config);
            else gdn_known_window_r2_preflight(fixture.rows, fixture.gdn_config);
        } else {
            if (launch) gdn_known_window_r2_f32_gates_launch(fixture.rows, fixture.gdn_config);
            else gdn_known_window_r2_f32_gates_preflight(fixture.rows, fixture.gdn_config);
        }
    };
    {
        const HipLaunchScope scope(fixture.scope());
        std::string message;
        try { call(false); } catch (const std::invalid_argument& error) { message = error.what(); }
        require(message == (fixture.f32_gates ? "GDN known-window R2 requires all five Q8_0 projections" :
            "GDN known-window R2 F32-gate variant requires Q8_0 qkv/z/out and F32 beta/alpha"),
            "wrong closed GDN entry did not preserve its exact format rejection");
    }
    check(hipStreamBeginCapture(fixture.owner.stream, hipStreamCaptureModeThreadLocal), "capture wrong typed GDN entry");
    std::exception_ptr error;
    try { const HipLaunchScope scope(fixture.scope()); invalid([&] { call(true); }); }
    catch (...) { error = std::current_exception(); }
    hipGraph_t graph{};
    check(hipStreamEndCapture(fixture.owner.stream, &graph), "end wrong typed GDN entry capture");
    require(graph != nullptr, "wrong typed GDN entry did not return graph");
    try { fixture.owner.graphs.push_back(graph); } catch (...) { (void)hipGraphDestroy(graph); throw; }
    std::size_t nodes = 0; check(hipGraphGetNodes(graph, nullptr, &nodes), "wrong typed GDN entry nodes");
    require(nodes == 0U, "wrong typed GDN entry enqueued capture nodes");
    if (error) std::rethrow_exception(error);
    fixture.owner.finish();
    require(read(fixture.scratch) == scratch && read(fixture.state) == state,
        "wrong typed GDN entry changed scratch or state");
}

void mixed_gdn_rejections(Fixture& fixture) {
    fixture.prepare(0U); fixture.reset();
    const auto before_scratch = read(fixture.scratch), before_state = read(fixture.state);
    const auto original_manifest = fixture.manifest;
    const auto original_runtime = fixture.runtime;
    const auto restore = [&] { fixture.manifest = original_manifest; fixture.runtime = original_runtime; };
    for (unsigned fault = 0; fault < 14U; ++fault) {
        restore();
        auto contexts = fixture.rows;
        auto other_runtime = fixture.runtime;
        auto other_resources = fixture.resources[1];
        if (fault == 0U || fault == 1U) {
            const auto index = fault == 0U ? 5U : 6U;
            fixture.manifest.tensors[index].ggml_type = 8U;
            fixture.manifest.tensors[index].byte_size = 2560ULL / 32U * 34U * 48U;
        }
        if (fault == 2U || fault == 3U) fixture.manifest.tensors[5].ggml_type = fault == 2U ? 1U : 30U;
        if (fault == 4U) {
            fixture.manifest.tensors[0].ggml_type = 0U;
            fixture.manifest.tensors[0].byte_size = 2560ULL * 10240U * 4U;
        }
        if (fault == 5U) fixture.runtime.tensors[5].bytes = 491520U - 4U;
        if (fault == 6U) fixture.runtime.tensors[5].address += 2U;
        if (fault == 7U) fixture.runtime.tensors[5].address = fixture.output[0].address;
        if (fault == 8U) fixture.runtime.tensors[5].address = fixture.temporary[1].address;
        if (fault == 9U) {
            other_resources.model_weights = &other_runtime;
            contexts[1].resources = &other_resources;
        }
        if (fault == 10U) fixture.runtime.tensors[5].allocation_generation = 0U;
        if (fault == 11U) fixture.manifest.tensors[5].byte_size -= 4U;
        if (fault == 12U) fixture.manifest.tensors[5].dimensions[1] = 47U;
        if (fault == 13U) fixture.runtime.tensors[6].address += 2U;
        try {
            { const HipLaunchScope scope(fixture.scope()); invalid([&] { fixture.preflight(contexts); }); }
            capture_rejection(fixture, contexts);
            fixture.owner.finish();
            require(read(fixture.scratch) == before_scratch && read(fixture.state) == before_state,
                "mixed typed metadata rejection changed scratch or persistent state");
        } catch (...) { restore(); throw; }
    }
    restore(); fixture.readonly();
}

void family_gdn_f32_gates() {
    Fixture fixture(true, false, true);
    Result finite_reference;
    for (unsigned flavor = 0; flavor < 5U; ++flavor) {
        fixture.prepare(flavor);
        const auto reference = fixture.run(false);
        same(reference, fixture.run(true)); same(reference, fixture.run(true));
        if (flavor < 3U) finite(reference);
        if (flavor == 0U) finite_reference = reference;
    }
    fixture.prepare(0U, true);
    const auto swapped = fixture.run(true);
    same(fixture.run(false), swapped); finite(swapped);
    require(swapped.convolution != finite_reference.convolution && swapped.matrix != finite_reference.matrix,
        "mixed F32 GDN causal-order witness is insensitive to swapped rows");
    rejections(fixture); mixed_gdn_rejections(fixture); wrong_gdn_entry(fixture);
    gdn_typed_input_stage(fixture);
    adversarial_f32_gates(fixture, [&] {
        gdn_typed_input_stage(fixture);
        for (const auto flavor : {0U,3U,4U}) {
            fixture.prepare(flavor);
            const auto reference = fixture.run(false);
            same(reference, fixture.run(true)); same(reference, fixture.run(true));
        }
    });
    for (const auto index : {0U,1U,8U}) {
        const auto& dims = fixture.manifest.tensors[index].dimensions;
        adversarial_q8_weight(fixture, index, static_cast<std::uint32_t>(dims[0]),
            static_cast<std::uint32_t>(dims[1]), [&] {
                fixture.prepare(4U); same(fixture.run(false), fixture.run(true));
                if (index != 8U) gdn_typed_input_stage(fixture);
            });
    }
}

// Native component evidence for the exact closed projection tiles used by the
// T runtime. This does not replace full-model T producer/continuation gates.
std::vector<unsigned char> bf16_matrix(std::uint32_t width, std::uint32_t outputs, unsigned salt) {
    std::vector<std::uint16_t> values(static_cast<std::size_t>(width) * outputs);
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = bf16(static_cast<float>(static_cast<int>((i * 7U + (i >> 8U) * 5U + salt) % 29U) - 14) * 0.00041F);
    return bytes_of(values);
}
// One real QSA step (canonical layer-3 geometry, Q8 projections, budget 2048,
// 4-cell blocks), eight row views of one physical window arena.
struct QsaFixture {
    Owner owner;
    ModelContract model = ModelContract::qwen38_flash_next();
    SerialExecutionPlan execution = build_serial_decode_execution_plan(model);
    SerialExecutionIoPlan io = build_serial_execution_io_plan(execution);
    ExecutionStep step;
    ExecutionArenaPlan serial;
    KnownWindowArena window;
    Span scratch, state, kv, raw;
    ExecutionMemoryBinding memory;
    ModelManifest manifest;
    ModelRuntimeBinding runtime;
    ExecutionTensorAccessPlan access;
    std::vector<Weight> weights;
    SerialSessionStateLayout state_layout;
    SessionStateMemoryBinding state_memory;
    std::vector<unsigned char> host_state, initial_state, seed;
    ExecutionRuntimeResources resources;
    std::array<ExecutionDispatchContext, 8> rows;
    HipIndexedAttentionLauncherConfig config;
    std::string selected_id;
    ExecutionLauncher selected;

    const bool bf16_indexer;
    QsaFixture(bool prefetch, bool wave8, bool bf16_index, std::uint64_t context_tokens, bool dense = false) : bf16_indexer(bf16_index) {
        const auto found = std::find_if(execution.steps.begin(), execution.steps.end(),
            [](const auto& value) { return value.op == SerialOp::indexed_attention; });
        require(found != execution.steps.end(), "canonical QSA operator missing");
        step = *found;
        const auto& step_io = io.steps[step.id];
        serial.alignment = 256U;
        const auto allocation = [&](ExecutionBufferUse use, std::uint64_t bytes) {
            const auto offset = (serial.total_bytes + 255U) / 256U * 256U;
            serial.allocations.push_back({use, step.id, step.id, offset, bytes});
            serial.total_bytes = offset + bytes;
        };
        allocation(step_io.reads[0], 2560U * 2U);
        allocation(step_io.writes[0], 2560U * 2U);
        allocation(step_io.temporaries[0], build_indexed_attention_workspace_layout(model).total_bytes);
        serial.total_bytes = (serial.total_bytes + 255U) / 256U * 256U;
        serial.naive_nonaliased_bytes = serial.total_bytes;
        window = build_known_window_arena(serial, 256U, 8U);
        scratch = owner.allocate(window.physical.total_bytes);
        memory = {0x4b4e4f574e515341ULL, scratch.address, scratch.bytes, 43U};
        manifest.architecture = "qwen4_exp";
        const auto layer = std::to_string(*step.layer);
        add("blk." + layer + ".attn_q.weight", TensorRole::attention_projection, {2560, 12288}, q8(2560, 12288, 11));
        add("blk." + layer + ".attn_k.weight", TensorRole::attention_projection, {2560, 512}, q8(2560, 512, 12));
        add("blk." + layer + ".attn_v.weight", TensorRole::attention_projection, {2560, 512}, q8(2560, 512, 13));
        add("blk." + layer + ".attn_output.weight", TensorRole::attention_projection, {6144, 2560}, q8(6144, 2560, 14));
        add("blk." + layer + ".attn_q_norm.weight", TensorRole::normalization, {256}, f32(256U, 1.0F, 0.03F), 0U);
        add("blk." + layer + ".attn_k_norm.weight", TensorRole::normalization, {256}, f32(256U, 0.9F, 0.02F), 0U);
        // The real checkpoint keeps both indexer projections in BF16.
        if (bf16_indexer) {
            add("blk." + layer + ".indexer.q_proj.weight", TensorRole::sparse_index, {2560, 512}, bf16_matrix(2560, 512, 15), 30U);
            add("blk." + layer + ".indexer.k_proj.weight", TensorRole::sparse_index, {2560, 128}, bf16_matrix(2560, 128, 16), 30U);
        } else {
            add("blk." + layer + ".indexer.q_proj.weight", TensorRole::sparse_index, {2560, 512}, q8(2560, 512, 15));
            add("blk." + layer + ".indexer.k_proj.weight", TensorRole::sparse_index, {2560, 128}, q8(2560, 128, 16));
        }
        add("blk." + layer + ".indexer.q_norm.weight", TensorRole::sparse_index, {128}, f32(128U, 1.1F, 0.02F), 0U);
        add("blk." + layer + ".indexer.k_norm.weight", TensorRole::sparse_index, {128}, f32(128U, 0.95F, 0.03F), 0U);
        manifest.layout_identity = model_manifest_layout_identity(manifest);
        runtime.manifest_layout_identity = manifest.layout_identity;
        runtime.model_content_identity.bytes[0] = 1U;
        access.manifest_layout_identity = manifest.layout_identity;
        access.plan_identity = 0x4b4e4f574e515342ULL;
        access.accesses.push_back({step.id, step.op, 0U, {0, 1, 2, 3}});
        access.accesses.push_back({step.id, step.op, 1U, {6, 7, 8, 9}});
        access.accesses.push_back({step.id, step.op, 2U, {4, 5}});
        SerialSessionStateLayoutRequest request; request.context_tokens = context_tokens;
        state_layout = build_serial_session_state_layout(model, request);
        state = owner.allocate(state_layout.device_total_bytes);
        host_state.resize(static_cast<std::size_t>(state_layout.host_total_bytes));
        state_memory = {state_layout.layout_identity, state.address, state.bytes, 47U,
            reinterpret_cast<std::uintptr_t>(host_state.data()), state_layout.host_total_bytes, 53U};
        const auto k = resolve_indexed_attention_layer_state(state_layout, state_memory, model, *step.layer, SessionStateComponent::attention_kv);
        const auto r = resolve_indexed_attention_layer_state(state_layout, state_memory, model, *step.layer, SessionStateComponent::sparse_index);
        kv = {k.address, k.bytes}; raw = {r.address, r.bytes};
        // Finite BF16 history in every QSA cell of the tested layer.
        initial_state.assign(static_cast<std::size_t>(state.bytes), 0U);
        for (const auto view : {kv, raw}) {
            std::vector<std::uint16_t> values(static_cast<std::size_t>(view.bytes / 2U));
            for (std::size_t i = 0; i < values.size(); ++i) {
                const auto mixed = (i * 2654435761ULL + (view.address == kv.address ? 17U : 91U)) >> 7U;
                values[i] = bf16(static_cast<float>(static_cast<int>(mixed % 2001U) - 1000) * 0.001F);
            }
            std::memcpy(initial_state.data() + (view.address - state.address), values.data(), values.size() * 2U);
        }
        resources = {&memory, &manifest, &runtime, &access};
        resources.session_state_layout = &state_layout; resources.session_state_memory = &state_memory;
        config.model = model;
        config.prefetch_q8_input = prefetch;
        config.wave8_q8_output = wave8;
        config.dense_a8_projections = dense;
        if (prefetch) {
            config.prefetch_eligible_layers = {*step.layer};
            config.prefetch_stats = std::make_shared<qsa_input_prefetch::RuntimeDispatchStats>();
        }
        ExecutionBindingRequest binding;
        binding.prefetch_q8_qsa_input = prefetch; binding.qsa_input_prefetch_layers = config.prefetch_eligible_layers;
        binding.wave8_q8_qsa_output = wave8;
        binding.dense_a8 = dense;
        kernels::KernelRegistry registry;
        register_hip_indexed_attention_kernel_variant(model, binding, config, registry);
        require(registry.variants().size() == 1U, "QSA serial selection is ambiguous");
        selected_id = registry.variants().front().id;
        selected = make_hip_indexed_attention_launcher(selected_id, &config);
        seed.assign(static_cast<std::size_t>(scratch.bytes), 0xa5U);
        for (unsigned row = 0; row < 8U; ++row) {
            auto& context = rows[row];
            context.execution_identity = memory.execution_identity;
            context.step_id = step.id; context.op = step.op; context.layer = step.layer;
            context.kernel_id = selected_id;
            context.reads = step_io.reads; context.writes = step_io.writes; context.temporaries = step_io.temporaries;
            context.arena = &window.row_view[row]; context.memory = &memory; context.resources = &resources;
            context.session_state_layout = &state_layout; context.session_state_memory = &state_memory;
            context.reads_state = step.reads_state; context.writes_state = step.writes_state;
            context.input_token_id = 101U + row;
            const auto input = resolve_dispatch_read(context, 0);
            std::vector<std::uint16_t> values(2560U);
            for (std::size_t i = 0; i < values.size(); ++i)
                values[i] = bf16(static_cast<float>(static_cast<int>((i * 7U + row * 13U) % 41U) - 20) * (0.011F + 0.003F * static_cast<float>(row)));
            std::memcpy(seed.data() + (input.address - scratch.address), values.data(), values.size() * 2U);
        }
    }
    void add(std::string name, TensorRole role, std::vector<std::uint64_t> dimensions, std::vector<unsigned char> bytes,
             std::uint32_t type = 8U) {
        ManifestTensor tensor;
        tensor.name = std::move(name); tensor.role = role; tensor.layer = step.layer;
        tensor.ggml_type = type; tensor.dimensions = std::move(dimensions); tensor.byte_size = bytes.size();
        tensor.absolute_offset = (manifest.total_tensor_bytes + 15U) / 16U * 16U;
        manifest.total_tensor_bytes = tensor.absolute_offset + tensor.byte_size;
        const auto device = owner.allocate(bytes.size());
        write(device, bytes);
        const auto index = static_cast<std::uint32_t>(weights.size());
        weights.push_back({device, std::move(bytes)});
        manifest.tensors.push_back(std::move(tensor));
        runtime.tensors.push_back({index, RuntimeTensorStorage::resident, device.address, device.bytes, 7U, 0U});
    }
    HipLaunchContext scope() const { HipLaunchContext result; result.stream = owner.stream; return result; }
    std::span<const ExecutionDispatchContext> group(std::uint32_t count) const { return {rows.data(), count}; }
    void place(std::uint64_t position) { for (unsigned r = 0; r < 8U; ++r) rows[r].position = position + r; }
    void reset() { invalidate_hip_qsa_window_pool(config.window_pool); write(scratch, seed); write(state, initial_state); }
    void serial_rows(std::uint32_t count) { for (std::uint32_t r = 0; r < count; ++r) selected.function(rows[r], selected.user_data); }
    struct Payload { std::vector<unsigned char> arena, state; };
    Payload run(std::uint32_t count, std::uint64_t position, bool candidate) {
        reset(); place(position);
        {
            const HipLaunchScope active(scope());
            if (candidate) { indexed_attention_known_window_rows_preflight(group(count), config);
                indexed_attention_known_window_rows_launch(group(count), config); }
            else serial_rows(count);
        }
        owner.finish();
        return {read(scratch), read(state)};
    }
    // Cells [0, cells) of the tested layer's K/V and raw index state.
    std::vector<unsigned char> cells(const std::vector<unsigned char>& bytes, Span view, std::uint64_t row_bytes, std::uint64_t count) const {
        const auto begin = bytes.begin() + static_cast<std::ptrdiff_t>(view.address - state.address);
        return {begin, begin + static_cast<std::ptrdiff_t>(row_bytes * count)};
    }
};

void qsa_rows_family(bool prefetch, bool wave8, bool bf16_indexer) {
    QsaFixture f(prefetch, wave8, bf16_indexer, 2304U);
    f.config.window_pool = make_hip_qsa_window_pool();
    const std::uint64_t prefetched = bf16_indexer ? 3U : 5U; // BF16 indexer stays on the legacy256 tree
    const char* skip_env = std::getenv("STRIX_QSA_SKIP_INDEX_Q");
    const bool skip_index_q = !skip_env || skip_env[0] != '0';
    const auto qsa_layout = build_indexed_attention_workspace_layout(f.model);
    // STRIX_QSA_SKIP_INDEX_Q (default ON): a step whose complete blocks fit block_topk never projects the index
    // query: a serial row when its own selection takes every visible token, the rows launch only when every row's
    // does. In a window straddling the top-k boundary the rows launch therefore writes the index-query slot of a
    // row whose serial call leaves it untouched. That row computes no block scores, so the slot is dead scratch:
    // only those slots are excluded from the serial/rows arena comparison, every other byte stays bit equal.
    const auto row_skips_of = [&](std::uint32_t count, std::uint64_t position) {
        std::array<bool, 8> skips{};
        for (std::uint32_t r = 0; r < count; ++r) {
            const auto visible = position + r + 1U;
            skips[r] = skip_index_q &&
                visible / f.model.indexer_compress_ratio <= f.model.indexer_budget / f.model.indexer_compress_ratio;
        }
        return skips;
    };
    const auto live_arena = [&](std::vector<unsigned char> arena, std::uint32_t count, std::uint64_t position) {
        const auto skips = row_skips_of(count, position);
        const auto skipped = static_cast<std::uint32_t>(std::count(skips.begin(), skips.begin() + count, true));
        if (skipped != 0U && skipped != count)
            for (std::uint32_t r = 0; r < count; ++r) {
                if (!skips[r]) continue;
                const auto slot = resolve_dispatch_temporary(f.rows[r], 0).address - f.scratch.address + qsa_layout.index_query_offset;
                std::fill_n(arena.begin() + static_cast<std::ptrdiff_t>(slot), qsa_layout.index_query_bytes, static_cast<unsigned char>(0U));
            }
        return arena;
    };
    const auto report_arena = [&](const char* what, const std::vector<unsigned char>& a, const std::vector<unsigned char>& b,
                                  std::uint32_t count, std::uint64_t position) {
        if (a == b) return;
        std::size_t first = a.size(), last = 0, diff = 0;
        for (std::size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) { first = std::min(first, i); last = i; ++diff; }
        std::cerr << "  " << what << " arena mismatch: prefetch=" << prefetch << " bf16_indexer=" << bf16_indexer << " rows=" << count
            << " position=" << position << " diff_bytes=" << diff << " first=" << first << " last=" << last << "\n";
    };
    unsigned compared = 0;
    // Positions cover p=0, partial-block tails, the full-visible/top-k boundary
    // (visible 2048..2053 straddles 512 complete blocks) and a pure top-k window.
    for (const std::uint32_t count : {2U, 4U, 8U})
        for (const std::uint64_t position : {0ULL, 1ULL, 3ULL, 6ULL, 2042ULL, 2045ULL, 2047ULL, 2049ULL, 2201ULL}) {
            const auto before = prefetch ? f.config.prefetch_stats->dispatched_projections.load() : 0U;
            const auto reference = f.run(count, position, false);
            const auto middle = prefetch ? f.config.prefetch_stats->dispatched_projections.load() : 0U;
            const auto candidate = f.run(count, position, true);
            // The prefetch counters see the index-query skip only for the prefetched Q8_0 indexer.
            const auto row_skips = row_skips_of(count, position);
            const auto skipped_rows = static_cast<std::uint32_t>(std::count(row_skips.begin(), row_skips.begin() + count, true));
            const auto prefetch_skipped_rows = bf16_indexer ? 0U : skipped_rows;
            const auto serial_expected = prefetched * count - prefetch_skipped_rows;
            const auto rows_expected = prefetched - (!bf16_indexer && skipped_rows == count ? 1U : 0U);
            if (prefetch && !(middle - before == serial_expected && f.config.prefetch_stats->dispatched_projections.load() - middle == rows_expected))
                std::cerr << "  QSA rows prefetch count: rows=" << count << " position=" << position << " serial " << middle - before
                    << " (expected " << serial_expected << ") rows " << f.config.prefetch_stats->dispatched_projections.load() - middle
                    << " (expected " << rows_expected << ")\n";
            if (prefetch) require(middle - before == serial_expected && f.config.prefetch_stats->dispatched_projections.load() - middle == rows_expected,
                "QSA rows prefetch submission count differs");
            const auto candidate_arena = live_arena(candidate.arena, count, position), reference_arena = live_arena(reference.arena, count, position);
            report_arena("QSA rows", candidate_arena, reference_arena, count, position);
            require(candidate_arena == reference_arena, "QSA rows output/workspace bytes differ from ascending serial calls");
            require(candidate.state == reference.state, "QSA rows KV/raw-index state differs from ascending serial calls");
            std::vector<std::vector<unsigned char>> outputs;
            for (std::uint32_t r = 0; r < count; ++r) {
                const auto view = resolve_dispatch_write(f.rows[r], 0);
                const auto begin = candidate.arena.begin() + static_cast<std::ptrdiff_t>(view.address - f.scratch.address);
                outputs.emplace_back(begin, begin + 5120);
                for (std::size_t i = 0; i < 5120U; i += 2U)
                    require((outputs.back()[i + 1U] & 0x7fU) != 0x7fU || (outputs.back()[i] & 0x80U) == 0U, "QSA rows output is nonfinite");
                require(r == 0U || outputs[r] != outputs[r - 1U], "QSA fixture rows are insensitive to their inputs");
            }
            ++compared;
        }
    guards(f.scratch); guards(f.state);
    for (const auto& weight : f.weights) { require(read(weight.device) == weight.original, "QSA weights changed"); guards(weight.device); }

    // Tail snapshot/commit for every n_keep, the commit replayed from one captured graph.
    const auto journal = f.owner.allocate(qsa_index_journal_bytes(f.model));
    const auto plan = prepare_qsa_index_tail(f.model, f.state_layout, f.state_memory, {journal.address, journal.bytes});
    const auto words = f.owner.allocate(16U);
    auto* device_position = reinterpret_cast<std::uint64_t*>(words.address);
    auto* device_keep = reinterpret_cast<std::uint32_t*>(words.address + 8U);
    hipGraph_t graph{}; hipGraphExec_t commit{};
    check(hipStreamBeginCapture(f.owner.stream, hipStreamCaptureModeThreadLocal), "capture QSA tail commit");
    try { enqueue_qsa_index_tail_commit(plan, device_position, device_keep, f.owner.stream); }
    catch (...) { (void)hipStreamEndCapture(f.owner.stream, &graph); throw; }
    check(hipStreamEndCapture(f.owner.stream, &graph), "finish QSA tail commit capture");
    f.owner.graphs.push_back(graph);
    check(hipGraphInstantiate(&commit, graph, nullptr, nullptr, 0), "instantiate QSA tail commit");
    const auto state_contract = build_indexed_attention_state_layout(f.model);
    unsigned commits = 0;
    for (const std::uint32_t count : {2U, 4U, 8U})
        for (const std::uint64_t position : {5ULL, 2046ULL, 2203ULL}) {
            for (std::uint32_t keep = 0; keep <= count; ++keep) {
                const auto golden = keep ? f.run(keep, position, false).state : f.initial_state;
                const auto expected_cells = position + keep;
                f.reset(); f.place(position);
                check(hipMemcpy(device_position, &position, 8U, hipMemcpyHostToDevice), "QSA tail position");
                check(hipMemcpy(device_keep, &keep, 4U, hipMemcpyHostToDevice), "QSA tail n_keep");
                {
                    const HipLaunchScope active(f.scope());
                    enqueue_qsa_index_tail_snapshot(plan, device_position, f.owner.stream);
                    indexed_attention_known_window_rows_launch(f.group(count), f.config, {journal.address, journal.bytes});
                }
                // Clobber the tail block and every window cell before committing.
                const auto start = position - position % 4U;
                check(hipMemsetAsync(reinterpret_cast<void*>(f.raw.address + start * state_contract.raw_index_row_bytes), 0xff,
                    static_cast<std::size_t>((position + count - start) * state_contract.raw_index_row_bytes), f.owner.stream), "clobber QSA tail");
                check(hipGraphLaunch(commit, f.owner.stream), "replay QSA tail commit");
                f.owner.finish();
                const auto actual = read(f.state);
                require(f.cells(actual, f.raw, state_contract.raw_index_row_bytes, expected_cells) ==
                        f.cells(golden, f.raw, state_contract.raw_index_row_bytes, expected_cells),
                    "QSA tail commit raw index differs from serial accepted prefix");
                require(f.cells(actual, f.kv, state_contract.kv_row_bytes, expected_cells) ==
                        f.cells(golden, f.kv, state_contract.kv_row_bytes, expected_cells),
                    "QSA accepted KV prefix differs from serial");
                // Reuse the same pool after a rejection at every depth, including
                // complete speculative blocks and a partial block at the frontier.
                // Compare the next window's entire workspace, logits input and state.
                f.place(position + keep); write(f.scratch, f.seed);
                {
                    const HipLaunchScope active(f.scope());
                    indexed_attention_known_window_rows_launch(f.group(8U), f.config);
                }
                f.owner.finish();
                const auto next_arena = read(f.scratch), next_state = read(f.state);
                write(f.scratch, f.seed); write(f.state, actual);
                { const HipLaunchScope active(f.scope()); f.serial_rows(8U); }
                f.owner.finish();
                const auto serial_arena = read(f.scratch), serial_state = read(f.state);
                const auto live_next = live_arena(next_arena, 8U, position + keep), live_serial = live_arena(serial_arena, 8U, position + keep);
                report_arena("QSA pool reuse", live_next, live_serial, 8U, position + keep);
                if (serial_state != next_state || live_next != live_serial)
                    std::cerr << "  QSA pool reuse: rows=" << count << " keep=" << keep << " next position=" << position + keep
                        << " state_equal=" << (serial_state == next_state) << " raw_arena_equal=" << (serial_arena == next_arena) << "\n";
                require(live_serial == live_next && serial_state == next_state,
                    "QSA persistent pool reused a rejected/tail block");
                ++commits;
            }
        }
    check(hipGraphExecDestroy(commit), "destroy QSA tail commit");

    // Pure admission rejects before any enqueue.
    f.reset(); f.place(64U);
    const auto rejected = [&](auto&& mutate, const char* message) {
        auto bad = f.rows; mutate(bad);
        bool thrown = false;
        { const HipLaunchScope active(f.scope());
          try { indexed_attention_known_window_rows_launch(std::span<const ExecutionDispatchContext>(bad.data(), 4U), f.config); }
          catch (const std::invalid_argument&) { thrown = true; } }
        f.owner.finish();
        require(thrown && read(f.scratch) == f.seed && read(f.state) == f.initial_state, message);
    };
    rejected([](auto& rows) { rows[2].position += 1U; }, "QSA rows require consecutive positions");
    rejected([](auto& rows) { rows[1].arena = rows[0].arena; }, "QSA rows require disjoint row views");
    rejected([](auto& rows) { rows[3].layer = 7U; }, "QSA rows require one layer");
    const std::string foreign = f.selected_id + ".foreign";
    rejected([&](auto& rows) { rows[0].kernel_id = foreign; }, "QSA rows require the row's serial kernel configuration");
    {
        bool thrown = false;
        try { indexed_attention_known_window_rows_preflight(f.group(3U), f.config); } catch (const std::invalid_argument&) { thrown = true; }
        require(thrown, "QSA rows admit only two, four or eight rows");
    }
    std::cout << "QSA rows prefetch=" << prefetch << " wave8=" << wave8 << " bf16_indexer=" << bf16_indexer << ": " << compared
              << " bit-exact windows, " << commits << " tail commits\n";

    if (!std::getenv("STRIX_QSA_ROWS_BENCH")) return;
    hipEvent_t begin{}, end{};
    check(hipEventCreate(&begin), "bench event"); check(hipEventCreate(&end), "bench event");
    const auto time = [&](auto&& work) {
        std::vector<float> samples;
        for (unsigned i = 0; i < 28U; ++i) {
            check(hipEventRecord(begin, f.owner.stream), "bench begin");
            { const HipLaunchScope active(f.scope()); work(); }
            check(hipEventRecord(end, f.owner.stream), "bench end");
            check(hipEventSynchronize(end), "bench wait");
            float ms = 0.0F; check(hipEventElapsedTime(&ms, begin, end), "bench elapsed");
            if (i >= 3U) samples.push_back(ms);
        }
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2U];
    };
    for (const std::uint64_t position : {256ULL, 2201ULL})
        for (const std::uint32_t count : {2U, 4U, 8U}) {
            f.reset(); f.place(position);
            const auto serial_ms = time([&] { f.serial_rows(count); });
            const auto rows_ms = time([&] { indexed_attention_known_window_rows_launch(f.group(count), f.config); });
            const auto one_ms = time([&] { f.serial_rows(1U); });
            std::cout << "QSA bench p=" << position << " T=" << count << " serial(Tx1)=" << serial_ms << "ms rows=" << rows_ms
                      << "ms 1-row=" << one_ms << "ms speedup=" << serial_ms / rows_ms << "x rows/1-row=" << rows_ms / one_ms << "\n";
        }
    check(hipEventDestroy(begin), "bench event"); check(hipEventDestroy(end), "bench event");
}

// A cached owner can skip several cycles while its rejected raw keys are
// overwritten by serial steps or another owner. Its next frontier is then past
// the old speculative blocks: clamping only at that frontier misses stale keys.
void qsa_pool_idle_owner() {
    QsaFixture f(true, true, true, 2304U);
    const auto journal = f.owner.allocate(qsa_index_journal_bytes(f.model));
    const auto plan = prepare_qsa_index_tail(f.model, f.state_layout, f.state_memory, {journal.address, journal.bytes});
    const auto words = f.owner.allocate(16U);
    auto* position_word = reinterpret_cast<std::uint64_t*>(words.address);
    auto* keep_word = reinterpret_cast<std::uint32_t*>(words.address + 8U);
    constexpr std::uint64_t position = 2203U, keep = 1U;
    const std::vector<std::uint64_t> control{position, keep};
    for (const bool other_owner : {false, true}) for (const bool negate_query : {false, true}) {
        f.config.window_pool = make_hip_qsa_window_pool();
        f.reset(); f.place(position); write(words, bytes_of(control));
        {
            const HipLaunchScope active(f.scope());
            enqueue_qsa_index_tail_snapshot(plan, position_word, f.owner.stream);
            indexed_attention_known_window_rows_launch(f.group(8U), f.config, {journal.address, journal.bytes});
            enqueue_qsa_index_tail_commit(plan, position_word, keep_word, f.owner.stream);
        }
        f.owner.finish();
        const auto speculative = read(f.state);
        // Skip beyond every block the first owner formed speculatively.
        f.place(position + keep); write(f.scratch, f.seed);
        {
            const HipLaunchScope active(f.scope());
            if (other_owner) {
                auto other = f.config; other.window_pool = make_hip_qsa_window_pool();
                indexed_attention_known_window_rows_launch(f.group(8U), other);
            } else f.serial_rows(8U);
        }
        f.owner.finish();
        const auto advanced = read(f.state);
        const auto raw_begin = static_cast<std::ptrdiff_t>(f.raw.address - f.state.address +
            (position + keep) * f.model.indexer_head_dim * sizeof(std::uint16_t));
        const auto raw_bytes = static_cast<std::ptrdiff_t>(4U * f.model.indexer_head_dim * sizeof(std::uint16_t));
        require(!std::equal(speculative.begin() + raw_begin, speculative.begin() + raw_begin + raw_bytes,
            advanced.begin() + raw_begin), "QSA catch-up fixture did not overwrite the speculative block");
        // Scores apply ReLU per head. Exercise both query signs so an all-negative
        // dot product cannot hide a stale pooled key behind a zero score.
        auto next_seed = f.seed;
        if (negate_query) for (const auto& row : f.rows) {
            const auto input = resolve_dispatch_read(row, 0);
            for (std::uint32_t i = 0; i < 2560U; ++i) {
                const auto offset = static_cast<std::size_t>(input.address - f.scratch.address) + i * sizeof(std::uint16_t);
                std::uint16_t bits = 0; std::memcpy(&bits, next_seed.data() + offset, sizeof(bits));
                bits ^= 0x8000U; std::memcpy(next_seed.data() + offset, &bits, sizeof(bits));
            }
        }
        f.place(position + keep + 8U); write(f.scratch, next_seed);
        { const HipLaunchScope active(f.scope()); indexed_attention_known_window_rows_launch(f.group(8U), f.config); }
        f.owner.finish();
        const auto candidate_arena = read(f.scratch), candidate_state = read(f.state);
        write(f.state, advanced); write(f.scratch, next_seed);
        { const HipLaunchScope active(f.scope()); f.serial_rows(8U); }
        f.owner.finish();
        require(read(f.scratch) == candidate_arena && read(f.state) == candidate_state,
            "QSA idle owner reused speculative blocks overwritten during catch-up");
    }
    guards(f.scratch); guards(f.state);
    std::cout << "QSA persistent pool: serial and other-owner catch-up bit-exact\n";
}

// Cohort: the rows of two or three sessions (own KV and raw-index state, own positions) in one
// cohort launch against each session's own rows launch from the same state and inputs.
void qsa_cohort_family(bool dense, bool bf16_indexer) {
    QsaFixture f(!dense, true, bf16_indexer, 2304U, dense);
    const char* skip_env = std::getenv("STRIX_QSA_SKIP_INDEX_Q");
    const bool skip_index_q = !skip_env || skip_env[0] != '0';
    const auto qsa_layout = build_indexed_attention_workspace_layout(f.model);
    // Each session: its own window arena (the fixture's layout and inputs), state and resources.
    struct Session {
        Span state, scratch; std::vector<unsigned char> host, initial; SessionStateMemoryBinding memory;
        ExecutionMemoryBinding arena; ExecutionRuntimeResources resources; std::array<ExecutionDispatchContext, 8> rows;
    };
    std::array<Session, 8> sessions;
    for (std::size_t k = 0; k < sessions.size(); ++k) {
        auto& session = sessions[k];
        session.scratch = f.owner.allocate(f.window.physical.total_bytes);
        session.arena = {f.memory.execution_identity, session.scratch.address, session.scratch.bytes, 43U + k};
        session.state = f.owner.allocate(f.state_layout.device_total_bytes);
        session.host.resize(static_cast<std::size_t>(f.state_layout.host_total_bytes));
        session.memory = {f.state_layout.layout_identity, session.state.address, session.state.bytes, 61U + k,
            reinterpret_cast<std::uintptr_t>(session.host.data()), f.state_layout.host_total_bytes, 67U + k};
        session.resources = f.resources; session.resources.session_state_memory = &session.memory;
        session.resources.scratch = &session.arena;
        session.rows = f.rows;
        for (auto& row : session.rows) {
            row.memory = &session.arena; row.resources = &session.resources; row.session_state_memory = &session.memory;
        }
        // Each session's own finite BF16 history in every QSA cell of the tested layer.
        session.initial.assign(static_cast<std::size_t>(session.state.bytes), 0U);
        for (const auto component : {SessionStateComponent::attention_kv, SessionStateComponent::sparse_index}) {
            const auto view = resolve_indexed_attention_layer_state(f.state_layout, session.memory, f.model, *f.step.layer, component);
            std::vector<std::uint16_t> values(static_cast<std::size_t>(view.bytes / 2U));
            for (std::size_t i = 0; i < values.size(); ++i) {
                const auto mixed = (i * 2654435761ULL + 37U * (k + 1U) + (component == SessionStateComponent::attention_kv ? 17U : 91U)) >> 7U;
                values[i] = bf16(static_cast<float>(static_cast<int>(mixed % 2001U) - 1000) * 0.001F);
            }
            std::memcpy(session.initial.data() + (view.address - session.state.address), values.data(), values.size() * 2U);
        }
    }
    const auto run = [&](const std::vector<std::uint32_t>& sizes, const std::vector<std::uint64_t>& positions, bool cohort) {
        std::vector<HipQsaCohortSegment> segments;
        for (std::size_t k = 0; k < sizes.size(); ++k) {
            auto& session = sessions[k];
            write(session.scratch, f.seed); write(session.state, session.initial);
            for (std::uint32_t r = 0; r < sizes[k]; ++r) session.rows[r].position = positions[k] + r;
            segments.push_back({std::span<const ExecutionDispatchContext>(session.rows.data(), sizes[k]), &f.config});
        }
        {
            const HipLaunchScope active(f.scope());
            if (cohort) { indexed_attention_known_window_cohort_preflight(segments); indexed_attention_known_window_cohort_launch(segments); }
            else for (const auto& segment : segments) indexed_attention_known_window_rows_launch(segment.rows, f.config);
        }
        f.owner.finish();
        std::vector<std::vector<unsigned char>> out;
        for (std::size_t k = 0; k < sizes.size(); ++k) { out.push_back(read(sessions[k].scratch)); out.push_back(read(sessions[k].state)); }
        return out;
    };
    unsigned compared = 0;
    // 4 / 8 rows; 16 / 32 rows (cohorts of several 8-row chunks) with dense A8 only.
    std::vector<std::vector<std::uint32_t>> shapes{{2U, 2U}, {4U, 4U}, {2U, 2U, 4U}, {4U, 2U, 2U}};
    if (dense)
        for (const auto& wide : std::vector<std::vector<std::uint32_t>>{{8U, 8U}, {4U, 4U, 4U, 4U}, {8U, 4U, 2U, 2U},
                 {2U, 2U, 2U, 2U, 2U, 2U, 2U, 2U}, {8U, 8U, 8U, 8U}, {8U, 4U, 4U, 8U, 2U, 2U, 2U, 2U},
                 {4U, 4U, 2U}, {8U, 4U, 2U}, {8U, 8U, 4U, 2U, 2U}})
            shapes.push_back(wide);
    const std::vector<std::vector<std::uint64_t>> placements{{0U, 6U, 2045U, 2047U, 1U, 2201U, 3U, 1000U},
        {2047U, 1U, 2201U, 3U, 2042U, 0U, 6U, 1500U}, {2201U, 2042U, 3U, 0U, 1000U, 6U, 2047U, 1U}};
    for (const auto& sizes : shapes)
        for (const auto& positions : placements) {
            const auto reference_bytes = run(sizes, positions, false);
            const auto candidate_bytes = run(sizes, positions, true);
            // STRIX_QSA_SKIP_INDEX_Q (default ON): a rows launch skips the index-query projection only when every
            // one of its rows takes every visible token. The cohort projects the union of its segments, so a
            // session whose own window skips can get its index-query slots written by the cohort. Those rows
            // compute no block scores, so the slots are dead scratch: only they are excluded from the comparison.
            const auto skips = [&](std::uint64_t position, std::uint32_t count) {
                for (std::uint32_t r = 0; r < count; ++r)
                    if (!(skip_index_q && (position + r + 1U) / f.model.indexer_compress_ratio <=
                            f.model.indexer_budget / f.model.indexer_compress_ratio)) return false;
                return true;
            };
            bool union_skips = true;
            for (std::size_t k = 0; k < sizes.size(); ++k) union_skips = union_skips && skips(positions[k], sizes[k]);
            auto reference = reference_bytes, candidate = candidate_bytes;
            for (std::size_t k = 0; k < sizes.size() && !union_skips; ++k) {
                if (!skips(positions[k], sizes[k])) continue;
                for (std::uint32_t r = 0; r < sizes[k]; ++r) {
                    const auto slot = resolve_dispatch_temporary(sessions[k].rows[r], 0).address - sessions[k].scratch.address +
                        qsa_layout.index_query_offset;
                    for (auto* bytes : {&reference[2U * k], &candidate[2U * k]})
                        std::fill_n(bytes->begin() + static_cast<std::ptrdiff_t>(slot), qsa_layout.index_query_bytes, static_cast<unsigned char>(0U));
                }
            }
            for (std::size_t k = 0; k < candidate.size(); ++k)
                if (candidate[k] != reference[k]) {
                    std::size_t first = candidate[k].size(), last = 0, diff = 0;
                    for (std::size_t i = 0; i < candidate[k].size(); ++i)
                        if (candidate[k][i] != reference[k][i]) { first = std::min(first, i); last = i; ++diff; }
                    std::cerr << "  QSA cohort mismatch: dense=" << dense << " bf16_indexer=" << bf16_indexer << " session=" << k / 2U
                        << (k % 2U ? " state" : " arena") << " size=" << sizes[k / 2U] << " position=" << positions[k / 2U]
                        << " union_skips=" << union_skips << " diff_bytes=" << diff << " first=" << first << " last=" << last << "\n";
                }
            for (std::size_t k = 0; k < candidate.size(); ++k)
                require(candidate[k] == reference[k], k % 2U ? "QSA cohort KV/raw-index state differs from the session's own window" :
                    "QSA cohort output/workspace bytes differ from each session's own window");
            ++compared;
        }
    // Every segment keeps the single-session admission; totals other than 4 or 8 are rejected.
    const auto rejected = [&](const std::vector<std::uint32_t>& sizes, bool same_state, const char* message) {
        std::vector<std::array<ExecutionDispatchContext, 8>> rows;
        for (std::size_t k = 0; k < sizes.size(); ++k) rows.push_back(sessions[same_state ? 0U : k].rows);
        std::vector<HipQsaCohortSegment> segments;
        for (std::size_t k = 0; k < sizes.size(); ++k) {
            for (std::uint32_t r = 0; r < sizes[k]; ++r) rows[k][r].position = 100U * (k + 1U) + r;
            segments.push_back({std::span<const ExecutionDispatchContext>(rows[k].data(), sizes[k]), &f.config});
        }
        bool thrown = false;
        try { indexed_attention_known_window_cohort_preflight(segments); } catch (const std::invalid_argument&) { thrown = true; }
        require(thrown, message);
    };
    rejected({2U, 4U}, false, "QSA cohort admits only 4, 8 or 9..32 rows in total");
    if (!dense) rejected({8U, 4U}, false, "QSA cohort of more than 8 rows requires dense A8");
    rejected({2U, 2U}, true, "QSA cohort rejects two segments of one session state");
    std::cout << "QSA cohort dense=" << dense << " bf16_indexer=" << bf16_indexer << ": " << compared
              << " cohort windows bit-exact against per-session windows\n";
}

// Whole QSA layer (synthetic weights, model shapes) at long context: the
// 1-row launcher at visible = context and T-row windows ending there.
void bench_qsa_long_context() {
    hipEvent_t begin{}, end{};
    check(hipEventCreate(&begin), "bench event"); check(hipEventCreate(&end), "bench event");
    for (const std::uint64_t context : {2049ULL, 4096ULL, 32768ULL, 131072ULL, 262144ULL}) {
        QsaFixture f(true, true, true, context);
        f.reset();
        const auto time = [&](auto&& work) {
            std::vector<float> samples;
            for (unsigned i = 0; i < 23U; ++i) {
                check(hipEventRecord(begin, f.owner.stream), "bench begin");
                { const HipLaunchScope active(f.scope()); work(); }
                check(hipEventRecord(end, f.owner.stream), "bench end");
                check(hipEventSynchronize(end), "bench wait");
                float ms = 0.0F; check(hipEventElapsedTime(&ms, begin, end), "bench elapsed");
                if (i >= 3U) samples.push_back(ms);
            }
            std::sort(samples.begin(), samples.end());
            return samples[samples.size() / 2U];
        };
        f.place(context - 1U);
        const auto one_ms = time([&] { f.serial_rows(1U); });
        std::cout << "QSA layer context=" << context << " 1-row=" << one_ms << "ms";
        for (const std::uint32_t count : {2U, 4U, 8U}) {
            f.place(context - count);
            const auto rows_ms = time([&] { indexed_attention_known_window_rows_launch(f.group(count), f.config); });
            std::cout << " T" << count << "=" << rows_ms << "ms";
        }
        std::cout << '\n' << std::flush;
    }
    check(hipEventDestroy(begin), "bench event"); check(hipEventDestroy(end), "bench event");
}

void family_prefill_tiles(bool gdn,std::uint32_t count) {
    Fixture f(gdn,false,gdn,64U);
    const auto arena=build_known_prefill_arena(f.serial,256U,count);
    const auto storage=f.owner.allocate(arena.physical.total_bytes);
    const ExecutionMemoryBinding memory{0x5450524546494c31ULL,storage.address,storage.bytes,41U};
    auto resources=f.resources[0];resources.scratch=&memory;
    std::vector<ExecutionDispatchContext> rows(count);
    std::vector<ExecutionBufferAddress> input(count),output(count),temporary(count);
    std::vector<unsigned char> seed(static_cast<std::size_t>(storage.bytes),0xa5U);
    for(std::uint32_t r=0;r<count;++r) {
        rows[r]=f.rows[0];rows[r].execution_identity=memory.execution_identity;rows[r].arena=&arena.row_view[r];
        rows[r].memory=&memory;rows[r].resources=&resources;rows[r].position=r;rows[r].input_token_id=31U+r;
        input[r]=resolve_dispatch_read(rows[r],0);output[r]=resolve_dispatch_write(rows[r],0);temporary[r]=resolve_dispatch_temporary(rows[r],0);
        std::vector<std::uint16_t> values(2560U);
        for(std::size_t col=0;col<values.size();++col)
            values[col]=bf16(static_cast<float>(static_cast<int>((col*7U+r*13U)%37U)-18)*(.003F+static_cast<float>(r%5U)*.002F));
        std::memcpy(seed.data()+(input[r].address-storage.address),values.data(),values.size()*2U);
    }
    struct Payload {std::vector<unsigned char> arena,state;};
    const auto run=[&](bool candidate) {
        write(storage,seed);if(gdn) write(f.state,f.initial_state);
        {
            const HipLaunchScope active(f.scope());
            // Whole-T pure admission is complete before any row can mutate.
            for(std::uint32_t first=0;first<count;) {
                const auto width=candidate&&count-first>=8U?8U:1U;
                if(width==8U) {
                    const auto group=std::span<const ExecutionDispatchContext>(rows.data()+first,8U);
                    if(gdn) gdn_known_window_rows_preflight(group,f.gdn_config);
                    else shared_expert_known_window_rows_preflight(group,f.shared_config);
                } else {require(f.selected.preflight!=nullptr,"serial component lacks pure preflight");f.selected.preflight(rows[first],f.selected.user_data);}
                first+=width;
            }
            for(std::uint32_t first=0;first<count;) {
                const auto width=candidate&&count-first>=8U?8U:1U;
                if(width==8U) {
                    const auto group=std::span<const ExecutionDispatchContext>(rows.data()+first,8U);
                    if(gdn) gdn_known_window_rows_launch(group,f.gdn_config);
                    else shared_expert_known_window_rows_launch(group,f.shared_config);
                } else f.selected.function(rows[first],f.selected.user_data);
                first+=width;
            }
        }
        f.owner.finish();Payload result{read(storage),gdn?read(f.state):std::vector<unsigned char>{}};
        auto unchanged=seed;
        for(std::uint32_t r=0;r<count;++r) for(const auto view:{output[r],temporary[r]})
            std::copy_n(result.arena.begin()+static_cast<std::ptrdiff_t>(view.address-storage.address),
                static_cast<std::size_t>(view.bytes),unchanged.begin()+static_cast<std::ptrdiff_t>(view.address-storage.address));
        require(result.arena==unchanged,"T projection tile modified read-only input or internal arena canary");guards(storage);
        for(const auto& weight:f.weights) {require(read(weight.device)==weight.original,"T projection tile modified weights");guards(weight.device);}
        if(gdn) {
            auto expected=f.initial_state;
            for(const auto view:{f.matrix,f.convolution}) std::copy_n(result.state.begin()+static_cast<std::ptrdiff_t>(view.address-f.state.address),
                static_cast<std::size_t>(view.bytes),expected.begin()+static_cast<std::ptrdiff_t>(view.address-f.state.address));
            require(result.state==expected,"T causal tile modified another layer/state component");guards(f.state);
        }
        return result;
    };
    const auto reference=run(false);
    for(unsigned repeat=0;repeat<2U;++repeat) {
        const auto candidate=run(true);
        require(candidate.arena==reference.arena&&candidate.state==reference.state,
            "T tiled exact projections/full workspace/causal state differ from ascending original R1 calls");
    }
    // Late tile admission faults must be rejected before the first tile runs.
    write(storage,seed);if(gdn) write(f.state,f.initial_state);
    const auto state_before=gdn?read(f.state):std::vector<unsigned char>{};
    auto bad=rows;bad.back().resources=nullptr;bool rejected=false;
    {
        const HipLaunchScope active(f.scope());
        try {for(std::uint32_t first=0;first<count;) {
            const auto width=count-first>=8U?8U:1U;
            if(width==8U) {
                const auto group=std::span<const ExecutionDispatchContext>(bad.data()+first,8U);
                if(gdn) gdn_known_window_rows_preflight(group,f.gdn_config);else shared_expert_known_window_rows_preflight(group,f.shared_config);
            } else f.selected.preflight(bad[first],f.selected.user_data);
            first+=width;
        }}
        catch(const std::exception&) {rejected=true;}
        require(rejected,"late T row missing its resource owner reaches original admission before any tile enqueue");
    }
    f.owner.finish();require(read(storage)==seed&&(!gdn||read(f.state)==state_before),"late T admission failure changed arena or state");
}

void family(bool gdn, bool prefetch = false) {
    Fixture fixture(gdn, prefetch);
    Result finite_reference;
    for (unsigned flavor = 0; flavor < 4U; ++flavor) {
        fixture.prepare(flavor);
        const auto serial = fixture.run(false);
        const auto candidate = fixture.run(true);
        same(serial, candidate);
        same(candidate, fixture.run(true)); // reset, repeat, full-bit determinism
        if (!gdn) same(serial, fixture.run_detached());
        if (flavor < 3U) finite(candidate);
        if (flavor == 0U) finite_reference = candidate;
    }
    if (gdn) {
        fixture.prepare(0U, true);
        const auto swapped = fixture.run(true);
        same(fixture.run(false), swapped);
        finite(swapped);
        require(swapped.convolution != finite_reference.convolution && swapped.matrix != finite_reference.matrix,
            "finite GDN fixture is insensitive to causal input order");
    }
    rejections(fixture);
    if (gdn) { gdn_projection_rows(fixture); gdn_typed_input_stage(fixture); wrong_gdn_entry(fixture); }
    else for (unsigned weight = 0; weight < 3U; ++weight) {
        const auto& dimensions = fixture.manifest.tensors[weight].dimensions;
        adversarial_q8_weight(fixture, weight, static_cast<std::uint32_t>(dimensions[0]),
            static_cast<std::uint32_t>(dimensions[1]), [&] {
                fixture.prepare(3U);
                const auto reference = fixture.run(false);
                same(reference, fixture.run(true));
            });
    }
}
// HC rows: one canonical attn/mlp/final mixer, T row views of the window arena,
// real HC shapes and formats (F32 norm, Q8 down/up, F32 injection).
struct HyperFixture {
    Owner owner;
    ModelContract model = ModelContract::qwen38_flash_next();
    SerialExecutionPlan execution = build_serial_decode_execution_plan(model);
    SerialExecutionIoPlan io = build_serial_execution_io_plan(execution);
    ExecutionStep step;
    ExecutionArenaPlan serial;
    KnownWindowArena window;
    Span scratch;
    ExecutionMemoryBinding memory;
    ModelManifest manifest;
    ModelRuntimeBinding runtime;
    ExecutionTensorAccessPlan access;
    std::vector<Weight> weights;
    ExecutionRuntimeResources resources;
    std::vector<ExecutionDispatchContext> rows;
    std::vector<unsigned char> seed;
    HipHyperMixLauncherConfig config;
    std::string selected_id;
    ExecutionLauncher selected;
    bool final_mix;

    HyperFixture(SerialOp op, bool splitk, bool lane, std::uint32_t count) : final_mix(op == SerialOp::final_hyper_mix) {
        const auto found = std::find_if(execution.steps.begin(), execution.steps.end(),
            [&](const auto& value) { return value.op == op && (final_mix || value.layer == 0U); });
        require(found != execution.steps.end(), "canonical HC mixer missing");
        step = *found;
        const auto& step_io = io.steps[step.id];
        require(step_io.reads.size() == 1U && step_io.writes.size() == (final_mix ? 1U : 2U) &&
            step_io.temporaries.size() == 1U, "HC fixture IO contract differs");
        serial.alignment = 256U;
        const auto allocation = [&](ExecutionBufferUse use, std::uint64_t bytes) {
            const auto offset = (serial.total_bytes + 255U) / 256U * 256U;
            serial.allocations.push_back({use, step.id, step.id, offset, bytes});
            serial.total_bytes = offset + bytes;
        };
        allocation(step_io.reads[0], 10240U * 2U);
        allocation(step_io.writes[0], 2560U * 2U);
        if (!final_mix) allocation(step_io.writes[1], 4U * 4U);
        allocation(step_io.temporaries[0], (10240U + 320U) * 4U);
        serial.total_bytes = (serial.total_bytes + 255U) / 256U * 256U;
        serial.naive_nonaliased_bytes = serial.total_bytes;
        window = build_known_window_arena(serial, 256U, count);
        scratch = owner.allocate(window.physical.total_bytes);
        memory = {0x4b4e4f574e484301ULL, scratch.address, scratch.bytes, 19U};
        manifest.architecture = "qwen4_exp";
        const auto salt = static_cast<unsigned>(op) * 7U;
        add("hc_norm.weight", 0U, {10240}, f32(10240U, 1.0F, 0.04F));
        add("hc_down.weight", 8U, {10240, 320}, q8(10240, 320, 11U + salt));
        add("hc_up.weight", 8U, {320, 10240}, q8(320, 10240, 5U + salt));
        if (!final_mix) add("hc_inject.weight", 0U, {10240, 4}, f32(40960U, 0.0004F, 0.0002F));
        for (std::uint32_t slot = 0; slot < weights.size(); ++slot) access.accesses.push_back({step.id, step.op, slot, {slot}});
        manifest.layout_identity = model_manifest_layout_identity(manifest);
        runtime.manifest_layout_identity = manifest.layout_identity;
        runtime.model_content_identity.bytes[0] = 1U;
        access.manifest_layout_identity = manifest.layout_identity;
        access.plan_identity = 0x4b4e4f574e484302ULL;
        resources = {&memory, &manifest, &runtime, &access};
        config.hidden_size = 2560U; config.hc_count = 4U; config.lowrank = 320U;
        config.rms_epsilon = model.rms_norm_epsilon;
        config.parallel = config.staged_projections = config.fused_injection = config.typed_projection = true;
        config.prefetch_rows = config.prefetch_q8_output = true;
        config.lane_chain_output = lane; config.splitk_projection = splitk;
        ExecutionBindingRequest binding; binding.lane_chain_hyper_output = lane;
        kernels::KernelRegistry registry;
        register_hip_hyper_mix_kernel_variant(model, binding, config, registry);
        require(registry.variants().size() == 1U, "HC serial selection is ambiguous");
        selected_id = registry.variants().front().id;
        selected = make_hip_hyper_mix_launcher(selected_id, &config);
        rows.resize(count);
        for (std::uint32_t r = 0; r < count; ++r) {
            auto& context = rows[r];
            context.execution_identity = memory.execution_identity;
            context.step_id = step.id; context.op = step.op; context.layer = step.layer;
            context.kernel_id = selected_id;
            context.reads = step_io.reads; context.writes = step_io.writes; context.temporaries = step_io.temporaries;
            context.arena = &window.row_view[r]; context.memory = &memory; context.resources = &resources;
            context.position = r; context.input_token_id = 17U + r;
        }
    }
    void add(std::string name, std::uint32_t type, std::vector<std::uint64_t> dimensions, std::vector<unsigned char> bytes) {
        ManifestTensor tensor;
        tensor.name = final_mix ? std::move(name) : "blk.0." + name; tensor.role = TensorRole::dense_projection;
        tensor.layer = step.layer; tensor.ggml_type = type; tensor.dimensions = std::move(dimensions);
        tensor.byte_size = bytes.size();
        tensor.absolute_offset = (manifest.total_tensor_bytes + 15U) / 16U * 16U;
        manifest.total_tensor_bytes = tensor.absolute_offset + tensor.byte_size;
        const auto device = owner.allocate(bytes.size());
        write(device, bytes);
        const auto index = static_cast<std::uint32_t>(weights.size());
        weights.push_back({device, std::move(bytes)});
        manifest.tensors.push_back(std::move(tensor));
        runtime.tensors.push_back({index, RuntimeTensorStorage::resident, device.address, device.bytes, 7U, 0U});
    }
    HipLaunchContext scope() const { HipLaunchContext result; result.stream = owner.stream; return result; }
    void prepare(unsigned flavor) {
        seed.assign(static_cast<std::size_t>(scratch.bytes), 0xa5);
        constexpr std::array<std::uint16_t, 10> edge{0, 0x8000, 1, 0x8001, 0x007f, 0x807f, 0x0080, 0x4f00, 0xcf00, 0x3f80};
        for (std::uint32_t r = 0; r < rows.size(); ++r) {
            const auto source = flavor == 3U ? 0U : r;
            std::vector<std::uint16_t> values(10240U);
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (flavor == 1U) values[i] = edge[(i * 3U + source) % edge.size()];
                else if (flavor == 2U) values[i] = bf16((i % 2U ? -1.0F : 1.0F) * std::ldexp(1.0F, static_cast<int>((i + source * 5U) % 41U) - 20));
                else values[i] = bf16(static_cast<float>(static_cast<int>((i * 7U + source * 13U) % 37U) - 18) *
                    (0.003F + static_cast<float>(source % 5U) * 0.002F));
            }
            const auto input = resolve_dispatch_read(rows[r], 0);
            std::memcpy(seed.data() + (input.address - scratch.address), values.data(), values.size() * 2U);
        }
    }
    void serial_rows() const { for (const auto& row : rows) selected.function(row, selected.user_data); }
    void candidate_rows() const {
        hyper_mix_known_window_rows_preflight(rows, config);
        hyper_mix_known_window_rows_launch(rows, config);
    }
    std::vector<unsigned char> run(bool candidate) {
        write(scratch, seed);
        { const HipLaunchScope owned(scope()); if (candidate) candidate_rows(); else serial_rows(); }
        owner.finish();
        const auto result = read(scratch);
        guards(scratch);
        for (const auto& weight : weights) {
            require(read(weight.device) == weight.original, "HC rows changed weights");
            guards(weight.device);
        }
        auto unchanged = seed;
        for (const auto& row : rows) {
            std::vector<ExecutionBufferAddress> views{resolve_dispatch_write(row, 0), resolve_dispatch_temporary(row, 0)};
            if (!final_mix) views.push_back(resolve_dispatch_write(row, 1));
            for (const auto& view : views)
                std::copy_n(result.begin() + static_cast<std::ptrdiff_t>(view.address - scratch.address),
                    static_cast<std::size_t>(view.bytes), unchanged.begin() + static_cast<std::ptrdiff_t>(view.address - scratch.address));
        }
        require(result == unchanged, "HC rows modified a read-only input or interior canary");
        return result;
    }
    std::vector<unsigned char> output(const std::vector<unsigned char>& arena, std::uint32_t r) const {
        const auto view = resolve_dispatch_write(rows[r], 0);
        const auto begin = arena.begin() + static_cast<std::ptrdiff_t>(view.address - scratch.address);
        return {begin, begin + static_cast<std::ptrdiff_t>(view.bytes)};
    }
};

void family_hyper_rows(SerialOp op, bool splitk, bool lane, std::uint32_t count) {
    HyperFixture f(op, splitk, lane, count);
    // Q6P32 copies of the (six-bit) down and up tensors.
    std::vector<unsigned char> q6_down(q6p::q6p32_bytes(320U, 10240U)), q6_up(q6p::q6p32_bytes(10240U, 320U));
    require(q6p::to_q6p32(f.weights[1].original.data(), 320U, 10240U, q6_down.data()) &&
        q6p::to_q6p32(f.weights[2].original.data(), 10240U, 320U, q6_up.data()), "HC fixture weights are not six-bit");
    const auto q6_down_device = f.owner.allocate(q6_down.size()), q6_up_device = f.owner.allocate(q6_up.size());
    write(q6_down_device, q6_down); write(q6_up_device, q6_up);
    for (unsigned flavor = 0; flavor < 4U; ++flavor) {
        f.prepare(flavor);
        const auto serial = f.run(false);
        const auto candidate = f.run(true);
        require(candidate == serial, "HC rows output/injection/workspace differ bitwise from ascending R1 calls");
        require(f.run(true) == candidate, "HC rows are not deterministic across repeats");
        q6p::register_tensor(reinterpret_cast<const void*>(f.weights[1].device.address),
            {reinterpret_cast<const unsigned char*>(q6_down_device.address), 320U, 10240U, q6p::Kind::q6p32});
        q6p::register_tensor(reinterpret_cast<const void*>(f.weights[2].device.address),
            {reinterpret_cast<const unsigned char*>(q6_up_device.address), 10240U, 320U, q6p::Kind::q6p32});
        const auto hits = q6p::hits();
        const auto q6 = f.run(true);
        const auto used = q6p::hits() - hits;
        q6p::clear();
        require(q6 == candidate, "Q6P32 HC rows differ from the Q8_0 rows");
        require(used == (splitk ? 1U : 0U) + (lane ? 1U : 0U), "Q6P32 HC rows did not read the registered copies");
        if (flavor == 0U) {
            for (std::uint32_t r = 0; r < count; ++r) {
                const auto bytes = f.output(candidate, r);
                for (std::size_t offset = 0; offset < bytes.size(); offset += 2U) {
                    std::uint16_t bits{}; std::memcpy(&bits, bytes.data() + offset, 2U);
                    require((bits & 0x7f80U) != 0x7f80U, "finite HC fixture produced a nonfinite output");
                }
                if (r) require(bytes != f.output(candidate, 0U), "HC rows fixture is insensitive to the row input");
            }
        }
        if (flavor == 3U)
            for (std::uint32_t r = 1; r < count; ++r)
                require(f.output(candidate, r) == f.output(candidate, 0U), "identical HC rows produce different bytes");
    }
    // Pure admission: every rejection precedes any enqueue.
    f.prepare(0U); write(f.scratch, f.seed);
    const auto reject = [&](auto&& work, const char* message) {
        bool rejected = false;
        { const HipLaunchScope owned(f.scope());
          try { work(); } catch (const std::exception&) { rejected = true; } }
        f.owner.finish();
        require(rejected, message);
        require(read(f.scratch) == f.seed, "rejected HC rows admission changed the arena");
    };
    reject([&] { hyper_mix_known_window_rows_preflight(std::span<const ExecutionDispatchContext>(f.rows.data(), 1U), f.config); },
        "one-row HC rows entry was admitted");
    auto other = f.config; other.lane_chain_output = !lane; other.prefetch_q8_output = true;
    reject([&] { hyper_mix_known_window_rows_launch(f.rows, other); }, "HC rows accepted a different serial selection");
    auto bad = f.rows; bad.back().arena = bad.front().arena;
    reject([&] { hyper_mix_known_window_rows_launch(bad, f.config); }, "aliased HC row outputs were admitted");
    bad = f.rows; bad.back().kernel_id = std::nullopt;
    reject([&] { hyper_mix_known_window_rows_launch(bad, f.config); }, "HC row without serial id was admitted");
    bad = f.rows; bad.back().resources = nullptr;
    reject([&] { hyper_mix_known_window_rows_launch(bad, f.config); }, "late HC row without resources was admitted");
    auto fp64 = f.config; fp64.splitk_projection = true; fp64.splitk_projection_fp64 = true;
    reject([&] { hyper_mix_known_window_rows_preflight(f.rows, fp64); }, "FP64 HC oracle was admitted as a window entry");
}

// Fold corpus, BF16 bits (`x` random): 1 one value in eight special (Inf, -0, denormal, the largest finite,
// NaN of any payload, sign and quiet bit), 2 every value a NaN with its own payload, 3 finite (single NaN / Inf
// elements are planted per stream), 4 denormals and -0. Flavor 0 keeps `finite`.
std::uint16_t fold_corpus_bf16(unsigned flavor, std::uint32_t x, std::uint16_t finite) {
    const auto nan = static_cast<std::uint16_t>((x & 0x8000U) | 0x7f80U | (((x >> 4U) % 0x7fU) + 1U));
    const auto denormal = static_cast<std::uint16_t>((x & 0x8000U) | (((x >> 8U) % 0x7fU) + 1U));
    if (flavor == 1U) {
        switch (x % 16U) {
        case 0: return 0x7f80U;
        case 1: return 0xff80U;
        case 2: return 0x8000U;
        case 3: return denormal;
        case 4: return 0x7f7fU;
        case 5: case 6: return nan;
        default: return finite;
        }
    }
    if (flavor == 2U) return nan;
    if (flavor == 4U) return x % 4U == 0U ? denormal : x % 4U == 1U ? static_cast<std::uint16_t>(x & 0x8000U) : finite;
    return finite;
}
std::uint32_t fold_mix(std::uint32_t x) {
    x ^= x >> 16U; x *= 0x7feb352dU; x ^= x >> 15U; x *= 0x846ca68bU; x ^= x >> 16U;
    return x;
}

// The window arena image of a fold / persistent HC fixture: BF16 values everywhere from the LCG `state` (flavor 0
// finite, else fold_corpus_bf16), then each inject row's four FP32 gates. Flavors 1 and 2 give the gates NaN
// payloads (quiet / signalling, both signs), Inf, -0 and denormals; 4 denormal gates. Flavors 3 and 5 plant, per
// row, NaNs of different payloads in many different leaves of one stream (every tree level meets NaN with NaN),
// two in one leaf of another, a single NaN and an Inf in others, and a NaN in the hidden row.
// Activation storage of the fold and persistent HC fixtures: 2 (BF16 streams) or 4 (FP32 streams, --folds-fp32).
std::uint32_t fold_activation_bytes = 2U;
// FP32 streams take each BF16 corpus value as their upper half and, outside flavor 0, random lower payload bits.
std::vector<unsigned char> fold_corpus_seed(std::uint64_t base, std::uint64_t bytes, std::span<const ExecutionDispatchContext> inject_rows,
    unsigned flavor, std::uint32_t state) {
    std::vector<unsigned char> seed(static_cast<std::size_t>(bytes));
    const std::uint32_t e = fold_activation_bytes;
    for (std::size_t i = 0; i + e <= seed.size(); i += e) {
        state = state * 1664525U + 1013904223U;
        auto value = bf16(static_cast<float>(static_cast<int>(state >> 20U) - 2048) * 0.0007F);
        if (flavor && flavor != 5U) value = fold_corpus_bf16(flavor, fold_mix(state ^ (flavor << 28U)), value);
        if (e == 2U) std::memcpy(seed.data() + i, &value, sizeof(value));
        else {
            const std::uint32_t low = flavor ? fold_mix(state + 0x9e3779b9U) & 0xffffU : 0U;
            const std::uint32_t wide = (static_cast<std::uint32_t>(value) << 16U) | low;
            std::memcpy(seed.data() + i, &wide, sizeof(wide));
        }
    }
    // One activation element (BF16 bits; FP32 streams store them as the upper half).
    const auto put16 = [&](std::uint64_t address, std::uint16_t bits) {
        if (e == 2U) std::memcpy(seed.data() + (address - base), &bits, 2U);
        else { const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16U; std::memcpy(seed.data() + (address - base), &wide, 4U); }
    };
    const auto as_float = [](std::uint32_t u) { return std::bit_cast<float>(u); };
    for (std::uint32_t r = 0; r < inject_rows.size(); ++r) {
        const auto gates = resolve_dispatch_read(inject_rows[r], 2);
        float values[4] = {0.75F + 0.01F * static_cast<float>(r), 1.25F, -0.5F, 1.0F / 3.0F};
        if (flavor == 1U && r % 2U) { values[1] = as_float(0x7fc12345U | (r << 24U)); values[2] = as_float(0xff800000U); values[3] = as_float(1U); }
        if (flavor == 1U && !(r % 2U)) { values[1] = as_float(0x80000000U); values[2] = as_float(0x7f800001U + (r << 16U)); }
        if (flavor == 2U)
            for (std::uint32_t s = 0; s < 4U; ++s) values[s] = as_float((s % 2U ? 0xff800000U : 0x7f800000U) | (0x12345U * (s + 1U) + r));
        if (flavor == 4U) { values[0] = as_float(3U); values[1] = as_float(0x80000005U); values[2] = as_float(0x007fffffU); }
        std::memcpy(seed.data() + (gates.address - base), values, sizeof(values));
        if (flavor == 3U || flavor == 5U) {
            const auto hyper = resolve_dispatch_read(inject_rows[r], 0), hidden = resolve_dispatch_read(inject_rows[r], 1);
            const std::uint64_t f = (static_cast<std::uint64_t>(r) * 397U + 11U) % 2560U;
            const auto at = [&](std::uint32_t stream, std::uint64_t feature) { return hyper.address + e * ((stream % 4U) * 2560U + feature % 2560U); };
            put16(at(r, f), static_cast<std::uint16_t>(0x7fa0U + r));
            put16(at(r + 1U, f), 0xff80U);
            put16(at(r + 1U, f + 512U), 0xffd7U);
            // Leaves f, f + 1, f + 2, ..., f + 128 of one stream: distinct payloads, both signs, quiet and signalling.
            for (std::uint32_t d = 0; d < 9U; ++d)
                put16(at(r + 2U, f + (d ? 1U << (d - 1U) : 0U)), static_cast<std::uint16_t>((d % 2U ? 0xff80U : 0x7f80U) | (0x11U * d + r + 1U)));
            put16(at(r + 3U, f), 0x7fe1U);
            put16(at(r + 3U, f + 256U), 0xff83U);
            put16(hidden.address + e * ((f + 512U) % 2560U), 0x7fb3U);
        }
    }
    return seed;
}
// HC norm weights (FP32): flavor 5 sets NaNs of different payloads (and an Inf) in every stream.
std::vector<unsigned char> fold_norm_weights(unsigned flavor) {
    auto bytes = f32(10240U, 1.0F, 0.04F);
    if (flavor == 5U)
        for (std::uint32_t i = 0; i < 10240U; i += 97U) {
            const std::uint32_t bits = i % 5U == 0U ? 0x7f800000U : ((i % 2U ? 0xff800000U : 0x7f800000U) | (0x10001U + i));
            std::memcpy(bytes.data() + 4U * i, &bits, 4U);
        }
    return bytes;
}

// Folded window HC steps in the real window arena: the inject rows step (and, with `combine`, the MoE-combine
// rows step before it) folded into the HC rows step equals the separate rows launches over the whole arena;
// only the HC workspace (dead after its step) may differ, as the fold moves it when the arena aliases it.
// `flavor` selects the input corpus (0: finite; see fold_corpus_bf16) and the gates; `strict` also requires
// the HC workspace (normalised rows, low values) to equal the separate launches', wherever the fold put it.
// Returns false when the fold declined (nothing enqueued).
bool family_window_fold(SerialOp inject_op, std::uint32_t inject_layer, bool combine, std::uint32_t count,
    unsigned flavor = 0U, bool strict = false) {
    Owner owner;
    const auto model = ModelContract::qwen38_flash_next();
    const auto execution = build_serial_decode_execution_plan(model);
    const auto io = build_serial_execution_io_plan(execution, true);
    const auto inject_step = std::find_if(execution.steps.begin(), execution.steps.end(),
        [&](const auto& s) { return s.op == inject_op && s.layer == inject_layer; });
    require(inject_step != execution.steps.begin() && inject_step + 1 != execution.steps.end(), "inject step missing");
    const auto mix_step = *(inject_step + 1);
    const auto combine_step = *(inject_step - 1);
    require(!combine || combine_step.op == SerialOp::moe_combine, "MLP inject is not preceded by its MoE combine");
    const bool final_mix = mix_step.op == SerialOp::final_hyper_mix;
    require(mix_step.op == SerialOp::attn_hyper_mix || mix_step.op == SerialOp::mlp_hyper_mix || final_mix,
        "inject step is not followed by an HC step");
    ExecutionBufferRequest request; request.parallel_moe_router = true; request.activation_element_bytes = fold_activation_bytes;
    const auto serial = build_serial_execution_arena_plan(model, request, execution, io, 256U);
    const auto window = build_known_window_arena(serial, 256U, count);
    const auto scratch = owner.allocate(window.physical.total_bytes);
    const ExecutionMemoryBinding memory{0x4b4e4f574e464c01ULL + count, scratch.address, scratch.bytes, 41U};
    ModelManifest manifest; manifest.architecture = "qwen4_exp";
    ModelRuntimeBinding runtime;
    ExecutionTensorAccessPlan access;
    std::vector<Weight> weights;
    const auto add = [&](std::string name, std::uint32_t type, std::vector<std::uint64_t> dimensions, std::vector<unsigned char> bytes) {
        ManifestTensor tensor;
        tensor.name = final_mix ? std::move(name) : "blk." + std::to_string(*mix_step.layer) + "." + name;
        tensor.role = TensorRole::dense_projection; tensor.layer = mix_step.layer; tensor.ggml_type = type;
        tensor.dimensions = std::move(dimensions); tensor.byte_size = bytes.size();
        tensor.absolute_offset = (manifest.total_tensor_bytes + 15U) / 16U * 16U;
        manifest.total_tensor_bytes = tensor.absolute_offset + tensor.byte_size;
        const auto device = owner.allocate(bytes.size());
        write(device, bytes);
        const auto index = static_cast<std::uint32_t>(weights.size());
        weights.push_back({device, std::move(bytes)});
        manifest.tensors.push_back(std::move(tensor));
        runtime.tensors.push_back({index, RuntimeTensorStorage::resident, device.address, device.bytes, 7U, 0U});
    };
    add("hc_norm.weight", 0U, {10240}, fold_norm_weights(flavor));
    add("hc_down.weight", 8U, {10240, 320}, q8(10240, 320, 23U));
    add("hc_up.weight", 8U, {320, 10240}, q8(320, 10240, 19U));
    if (!final_mix) add("hc_inject.weight", 0U, {10240, 4}, f32(40960U, 0.0004F, 0.0002F));
    for (std::uint32_t slot = 0; slot < weights.size(); ++slot) access.accesses.push_back({mix_step.id, mix_step.op, slot, {slot}});
    manifest.layout_identity = model_manifest_layout_identity(manifest);
    runtime.manifest_layout_identity = manifest.layout_identity;
    runtime.model_content_identity.bytes[0] = 1U;
    access.manifest_layout_identity = manifest.layout_identity;
    access.plan_identity = 0x4b4e4f574e464c02ULL;
    const ExecutionRuntimeResources resources{&memory, &manifest, &runtime, &access};
    HipHyperMixLauncherConfig config;
    config.hidden_size = 2560U; config.hc_count = 4U; config.lowrank = 320U; config.rms_epsilon = model.rms_norm_epsilon;
    config.parallel = config.staged_projections = config.fused_injection = config.typed_projection = true;
    config.prefetch_rows = config.prefetch_q8_output = true;
    config.lane_chain_output = true; config.splitk_projection = true;
    const auto activation = fold_activation_bytes == 4U ? HipActivationStorage::fp32 : HipActivationStorage::bf16;
    config.activation = activation;
    ExecutionBindingRequest binding; binding.lane_chain_hyper_output = true;
    kernels::KernelRegistry registry;
    register_hip_hyper_mix_kernel_variant(model, binding, config, registry);
    require(registry.variants().size() == 1U, "HC serial selection is ambiguous");
    const auto mix_id = registry.variants().front().id;
    HipElementwiseLauncherConfig elementwise{2560U, 4U, activation};
    const auto inject_launcher = make_hip_inject_launcher("hip.inject.test", &elementwise);
    const auto combine_launcher = make_hip_moe_combine_launcher("hip.moe_combine.test", &elementwise);
    const auto contexts = [&](const ExecutionStep& s, std::string_view id) {
        std::vector<ExecutionDispatchContext> out(count);
        for (std::uint32_t r = 0; r < count; ++r) {
            auto& c = out[r];
            c.execution_identity = memory.execution_identity; c.step_id = s.id; c.op = s.op; c.layer = s.layer;
            c.kernel_id = id; c.reads = io.steps[s.id].reads; c.writes = io.steps[s.id].writes;
            c.temporaries = io.steps[s.id].temporaries; c.arena = &window.row_view[r]; c.memory = &memory;
            c.resources = &resources; c.reads_state = s.reads_state; c.writes_state = s.writes_state;
            c.position = 5U + r; c.input_token_id = 40U + r;
        }
        return out;
    };
    const auto combine_rows = contexts(combine_step, combine_launcher.key);
    const auto inject_rows = contexts(*inject_step, inject_launcher.key);
    const auto mix_rows = contexts(mix_step, mix_id);
    // Finite BF16 everywhere (or the flavor's corpus), then FP32 injection gates per row.
    const auto seed = fold_corpus_seed(scratch.address, scratch.bytes, inject_rows, flavor,
        0x2545f491U ^ count ^ (static_cast<std::uint32_t>(inject_op) << 8U) ^ (combine ? 0x10000U : 0U));
    HipLaunchContext scope; scope.stream = owner.stream;
    const auto run = [&](auto&& work) {
        write(scratch, seed);
        { const HipLaunchScope owned(scope); work(); }
        owner.finish();
        for (const auto& weight : weights) require(read(weight.device) == weight.original, "folded HC rows changed weights");
        return read(scratch);
    };
    // The separate launches, also where STRIX_DISPATCH_FUSE or STRIX_HC_PERSISTENT give the fold's HC step the
    // persistent form.
    hip_hyper_mix_testing_separate_launches(true);
    const auto reference = run([&] {
        if (combine) moe_combine_known_window_rows_launch(combine_rows, elementwise);
        inject_known_window_rows_launch(inject_rows, elementwise);
        hyper_mix_known_window_rows_launch(mix_rows, config);
    });
    hip_hyper_mix_testing_separate_launches(false);
    require(reference != seed, "separate rows launches wrote nothing");
    const auto folded = [&] {
        const auto combine_views = combine ? moe_combine_known_window_rows_views(combine_rows, elementwise) : ElementwiseRowsViews{};
        return hyper_mix_known_window_rows_launch_folded(combine ? &combine_views : nullptr,
            inject_known_window_rows_views(inject_rows, elementwise), mix_rows, config, elementwise);
    };
    bool launched = false;
    auto candidate = run([&] { launched = folded(); });
    if (!launched) require(candidate == seed, "declined folded HC rows enqueued work");
    if (!launched && !combine) return false;
    // A declined combine fold falls back, as in the window executor, to the combine launch and the inject fold.
    if (!launched && combine) {
        candidate = run([&] {
            moe_combine_known_window_rows_launch(combine_rows, elementwise);
            launched = hyper_mix_known_window_rows_launch_folded(nullptr, inject_known_window_rows_views(inject_rows, elementwise),
                mix_rows, config, elementwise);
        });
        require(launched, "the inject fold after a declined combine fold was declined");
        launched = false;
    }
    const auto masked = [&](std::vector<unsigned char> arena) {
        for (const auto& row : mix_rows) {
            const auto view = resolve_dispatch_temporary(row, 0);
            const auto offset = static_cast<std::ptrdiff_t>(view.address - scratch.address);
            std::copy_n(seed.begin() + offset, static_cast<std::size_t>(view.bytes), arena.begin() + offset);
        }
        return arena;
    };
    require(masked(candidate) == masked(reference), "folded HC rows differ from the separate rows launches");
    if (strict) {
        // The HC workspace too: in the arena, or moved to the owner-held buffer (the arena's then holds whatever
        // the folded input there became, which the masked comparison above covers).
        const auto* moved = hip_hyper_mix_testing_folded_workspace();
        if (!moved) require(candidate == reference, "folded HC workspace (normalised rows, low values) differs");
        else {
            constexpr std::uint64_t bytes = (10240U + 320U) * sizeof(float);
            for (std::uint32_t t = 0; t < count; ++t) {
                const auto view = resolve_dispatch_temporary(mix_rows[t], 0);
                const auto offset = static_cast<std::ptrdiff_t>(view.address - scratch.address);
                require(view.bytes >= bytes, "HC workspace view is smaller than the normalised row and low values");
                std::vector<unsigned char> workspace(static_cast<std::size_t>(bytes));
                check(hipMemcpy(workspace.data(), moved + static_cast<std::size_t>(t) * (10240U + 320U), workspace.size(),
                    hipMemcpyDeviceToHost), "read moved HC workspace");
                require(std::equal(workspace.begin(), workspace.end(), reference.begin() + offset),
                    "moved folded HC workspace (normalised rows, low values) differs");
            }
        }
    }
    if (launched) require(run([&] { (void)folded(); }) == candidate, "folded HC rows are not deterministic");
    return launched;
}
// An inject rows step and the HC rows step that reads its streams, in the real window arena: the
// persistent launch with the inject folded in equals the two launches byte for byte over the arena.
// `flavor` selects the input corpus and gates (fold_corpus_seed) and the norm weights (fold_norm_weights).
// With `separate` the reference is the separate launches (inject, norm, split-K, output; persistent HC off) and
// the persistent HC step without the fold must also equal it over the whole arena, HC workspace included.
// Returns false when the folded launch declined the pair (nothing enqueued).
bool family_hyper_inject_rows(SerialOp inject_op, std::uint32_t inject_layer, std::uint32_t count, unsigned flavor = 0U,
    bool separate = false) {
    Owner owner;
    const auto model = ModelContract::qwen38_flash_next();
    const auto execution = build_serial_decode_execution_plan(model);
    const auto io = build_serial_execution_io_plan(execution, true);
    const auto inject_step = std::find_if(execution.steps.begin(), execution.steps.end(),
        [&](const auto& s) { return s.op == inject_op && s.layer == inject_layer; });
    require(inject_step != execution.steps.end() && inject_step + 1 != execution.steps.end(), "inject step missing");
    const auto mix_step = *(inject_step + 1);
    const bool final_mix = mix_step.op == SerialOp::final_hyper_mix;
    require(mix_step.op == SerialOp::attn_hyper_mix || mix_step.op == SerialOp::mlp_hyper_mix || final_mix,
        "inject step is not followed by an HC step");
    ExecutionBufferRequest request; request.parallel_moe_router = true; request.activation_element_bytes = fold_activation_bytes;
    const auto serial = build_serial_execution_arena_plan(model, request, execution, io, 256U);
    const auto window = build_known_window_arena(serial, 256U, count);
    const auto scratch = owner.allocate(window.physical.total_bytes);
    const ExecutionMemoryBinding memory{0x4b4e4f574e494a01ULL + count, scratch.address, scratch.bytes, 37U};
    ModelManifest manifest; manifest.architecture = "qwen4_exp";
    ModelRuntimeBinding runtime;
    ExecutionTensorAccessPlan access;
    std::vector<Weight> weights;
    const auto add = [&](std::string name, std::uint32_t type, std::vector<std::uint64_t> dimensions, std::vector<unsigned char> bytes) {
        ManifestTensor tensor;
        tensor.name = final_mix ? std::move(name) : "blk." + std::to_string(*mix_step.layer) + "." + name;
        tensor.role = TensorRole::dense_projection; tensor.layer = mix_step.layer; tensor.ggml_type = type;
        tensor.dimensions = std::move(dimensions); tensor.byte_size = bytes.size();
        tensor.absolute_offset = (manifest.total_tensor_bytes + 15U) / 16U * 16U;
        manifest.total_tensor_bytes = tensor.absolute_offset + tensor.byte_size;
        const auto device = owner.allocate(bytes.size());
        write(device, bytes);
        const auto index = static_cast<std::uint32_t>(weights.size());
        weights.push_back({device, std::move(bytes)});
        manifest.tensors.push_back(std::move(tensor));
        runtime.tensors.push_back({index, RuntimeTensorStorage::resident, device.address, device.bytes, 7U, 0U});
    };
    add("hc_norm.weight", 0U, {10240}, fold_norm_weights(flavor));
    add("hc_down.weight", 8U, {10240, 320}, q8(10240, 320, 23U));
    add("hc_up.weight", 8U, {320, 10240}, q8(320, 10240, 19U));
    if (!final_mix) add("hc_inject.weight", 0U, {10240, 4}, f32(40960U, 0.0004F, 0.0002F));
    for (std::uint32_t slot = 0; slot < weights.size(); ++slot) access.accesses.push_back({mix_step.id, mix_step.op, slot, {slot}});
    manifest.layout_identity = model_manifest_layout_identity(manifest);
    runtime.manifest_layout_identity = manifest.layout_identity;
    runtime.model_content_identity.bytes[0] = 1U;
    access.manifest_layout_identity = manifest.layout_identity;
    access.plan_identity = 0x4b4e4f574e494a02ULL;
    const ExecutionRuntimeResources resources{&memory, &manifest, &runtime, &access};
    HipHyperMixLauncherConfig config;
    config.hidden_size = 2560U; config.hc_count = 4U; config.lowrank = 320U; config.rms_epsilon = model.rms_norm_epsilon;
    config.parallel = config.staged_projections = config.fused_injection = config.typed_projection = true;
    config.prefetch_rows = config.prefetch_q8_output = true;
    config.lane_chain_output = true; config.splitk_projection = true;
    const auto activation = fold_activation_bytes == 4U ? HipActivationStorage::fp32 : HipActivationStorage::bf16;
    config.activation = activation;
    ExecutionBindingRequest binding; binding.lane_chain_hyper_output = true;
    kernels::KernelRegistry registry;
    register_hip_hyper_mix_kernel_variant(model, binding, config, registry);
    require(registry.variants().size() == 1U, "HC serial selection is ambiguous");
    const auto mix_id = registry.variants().front().id;
    HipElementwiseLauncherConfig elementwise{2560U, 4U, activation};
    const auto inject_launcher = make_hip_inject_launcher("hip.inject.test", &elementwise);
    const auto contexts = [&](const ExecutionStep& s, std::string_view id) {
        std::vector<ExecutionDispatchContext> out(count);
        for (std::uint32_t r = 0; r < count; ++r) {
            auto& c = out[r];
            c.execution_identity = memory.execution_identity; c.step_id = s.id; c.op = s.op; c.layer = s.layer;
            c.kernel_id = id; c.reads = io.steps[s.id].reads; c.writes = io.steps[s.id].writes;
            c.temporaries = io.steps[s.id].temporaries; c.arena = &window.row_view[r]; c.memory = &memory;
            c.resources = &resources; c.reads_state = s.reads_state; c.writes_state = s.writes_state;
            c.position = 5U + r; c.input_token_id = 40U + r;
        }
        return out;
    };
    const auto inject_rows = contexts(*inject_step, inject_launcher.key);
    const auto mix_rows = contexts(mix_step, mix_id);
    // Finite BF16 everywhere (or the flavor's corpus), then FP32 injection gates per row.
    const auto seed = fold_corpus_seed(scratch.address, scratch.bytes, inject_rows, flavor,
        0x2545f491U ^ count ^ (static_cast<std::uint32_t>(inject_op) << 8U));
    HipLaunchContext scope; scope.stream = owner.stream;
    const auto run = [&](auto&& work) {
        write(scratch, seed);
        { const HipLaunchScope owned(scope); work(); }
        owner.finish();
        for (const auto& weight : weights) require(read(weight.device) == weight.original, "fused HC rows changed weights");
        return read(scratch);
    };
    const auto separate_launches = [&] {
        inject_known_window_rows_launch(inject_rows, elementwise);
        hyper_mix_known_window_rows_launch(mix_rows, config);
    };
    hip_hyper_mix_testing_separate_launches(separate);
    const auto reference = run(separate_launches);
    hip_hyper_mix_testing_separate_launches(false);
    require(reference != seed, "inject + HC rows wrote nothing");
    if (separate) {
        // The persistent HC step alone (the inject launched separately): every byte, normalised rows included.
        const auto before = hip_hyper_mix_testing_persistent_launches();
        const auto persistent = run(separate_launches);
        if (hip_hyper_mix_testing_persistent_launches() != before)
            require(persistent == reference, "persistent HC rows differ from the separate norm, split-K and output launches");
    }
    bool fused = false;
    const auto candidate = run([&] { fused = hyper_mix_known_window_rows_launch_injected(inject_rows, elementwise, mix_rows, config); });
    if (!fused) { require(candidate == seed, "declined fused HC rows enqueued work"); return false; }
    // The HC workspace is dead after its step; the folded launch may keep it in an owner-held buffer when
    // the arena gave it the inject's input memory. Every other byte of the arena must match.
    const auto masked = [&](std::vector<unsigned char> arena) {
        for (const auto& row : mix_rows) {
            const auto view = resolve_dispatch_temporary(row, 0);
            const auto offset = static_cast<std::ptrdiff_t>(view.address - scratch.address);
            std::copy_n(seed.begin() + offset, static_cast<std::size_t>(view.bytes), arena.begin() + offset);
        }
        return arena;
    };
    require(masked(candidate) == masked(reference), "fused inject + HC rows differ from the inject and HC rows launches");
    require(run([&] { (void)hyper_mix_known_window_rows_launch_injected(inject_rows, elementwise, mix_rows, config); }) == candidate,
        "fused inject + HC rows are not deterministic");
    return true;
}

// Median of warm hipEvent timings; MALL flushed before each sample when cold.
void bench_hyper_rows(bool splitk, bool lane, std::uint32_t count, bool cold) {
    HyperFixture f(SerialOp::attn_hyper_mix, splitk, lane, count);
    f.prepare(0U); write(f.scratch, f.seed);
    const auto flush = cold ? f.owner.allocate(96ULL << 20U) : Span{};
    hipEvent_t start{}, stop{};
    check(hipEventCreate(&start), "create HC bench event"); check(hipEventCreate(&stop), "create HC bench event");
    const auto measure = [&](bool candidate) {
        std::vector<float> samples;
        const HipLaunchScope owned(f.scope());
        for (unsigned i = 0; i < 28U; ++i) {
            if (cold) check(hipMemsetAsync(reinterpret_cast<void*>(flush.address), static_cast<int>(i), flush.bytes, f.owner.stream), "flush MALL");
            check(hipEventRecord(start, f.owner.stream), "record HC bench start");
            if (candidate) hyper_mix_known_window_rows_launch(f.rows, f.config); else f.serial_rows();
            check(hipEventRecord(stop, f.owner.stream), "record HC bench stop");
            check(hipEventSynchronize(stop), "complete HC bench sample");
            float ms = 0.0F; check(hipEventElapsedTime(&ms, start, stop), "HC bench elapsed");
            if (i >= 3U) samples.push_back(ms * 1000.0F);
        }
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2U];
    };
    const auto serial = measure(false), rows = measure(true);
    // Same rows with Q6P32 copies of the (six-bit) down and up tensors registered.
    std::vector<unsigned char> q6_down(q6p::q6p32_bytes(320U, 10240U)), q6_up(q6p::q6p32_bytes(10240U, 320U));
    require(q6p::to_q6p32(f.weights[1].original.data(), 320U, 10240U, q6_down.data()) &&
        q6p::to_q6p32(f.weights[2].original.data(), 10240U, 320U, q6_up.data()), "HC bench weights are not six-bit");
    const auto q6_down_device = f.owner.allocate(q6_down.size()), q6_up_device = f.owner.allocate(q6_up.size());
    write(q6_down_device, q6_down); write(q6_up_device, q6_up);
    q6p::register_tensor(reinterpret_cast<const void*>(f.weights[1].device.address),
        {reinterpret_cast<const unsigned char*>(q6_down_device.address), 320U, 10240U, q6p::Kind::q6p32});
    q6p::register_tensor(reinterpret_cast<const void*>(f.weights[2].device.address),
        {reinterpret_cast<const unsigned char*>(q6_up_device.address), 10240U, 320U, q6p::Kind::q6p32});
    const auto q6_rows = measure(true);
    q6p::clear();
    f.owner.finish();
    (void)hipEventDestroy(start); (void)hipEventDestroy(stop);
    std::cout << "hc-rows-bench projection=" << (splitk ? "splitk-f32" : "batched-v8") << " output=" << (lane ? "lane-chain" : "q8-batched")
        << " T=" << count << (cold ? " cold" : " warm") << " serial_us=" << serial << " rows_us=" << rows
        << " rows_q6p32_us=" << q6_rows << " speedup=" << serial / rows << '\n';
}
// The decode's weight memory: `sets` copies of the HC tensors in host pages registered fine-grained (MTYPE UC, as
// the model residency registers the GGUF), each window mix reading its own copy as the 97 mixes of a window do.
// Per-mix time = one event pair around `sets` back-to-back rows launches; median over 15 samples. The launch path
// (three kernels / persistent / static B) follows the process environment.
void bench_hyper_rows_uc(std::uint32_t count, std::uint32_t sets) {
    HyperFixture f(SerialOp::attn_hyper_mix, true, true, count);
    f.prepare(0U); write(f.scratch, f.seed);
    std::vector<std::uint64_t> offsets;
    std::uint64_t set_bytes = 0;
    for (const auto& weight : f.weights) { offsets.push_back(set_bytes); set_bytes += (weight.original.size() + 4095U) / 4096U * 4096U; }
    const auto total = set_bytes * sets;
    void* host = std::aligned_alloc(4096U, static_cast<std::size_t>(total));
    require(host != nullptr, "UC bench host allocation");
    for (std::uint32_t s = 0; s < sets; ++s)
        for (std::size_t i = 0; i < f.weights.size(); ++i)
            std::memcpy(static_cast<unsigned char*>(host) + s * set_bytes + offsets[i], f.weights[i].original.data(), f.weights[i].original.size());
    check(hipHostRegister(host, static_cast<std::size_t>(total), hipHostRegisterMapped), "register UC bench weights");
    void* device = nullptr;
    check(hipHostGetDevicePointer(&device, host, 0), "UC bench device alias");
    const auto original = f.runtime.tensors;
    const auto select = [&](std::uint32_t s) {
        for (std::size_t i = 0; i < f.weights.size(); ++i)
            f.runtime.tensors[i].address = reinterpret_cast<std::uintptr_t>(device) + s * set_bytes + offsets[i];
    };
    hipEvent_t start{}, stop{};
    check(hipEventCreate(&start), "create UC bench event"); check(hipEventCreate(&stop), "create UC bench event");
    std::vector<float> samples;
    {
        const HipLaunchScope owned(f.scope());
        for (unsigned i = 0; i < 18U; ++i) {
            check(hipEventRecord(start, f.owner.stream), "record UC bench start");
            for (std::uint32_t s = 0; s < sets; ++s) { select(s); hyper_mix_known_window_rows_launch(f.rows, f.config); }
            check(hipEventRecord(stop, f.owner.stream), "record UC bench stop");
            check(hipEventSynchronize(stop), "complete UC bench sample");
            float ms = 0.0F; check(hipEventElapsedTime(&ms, start, stop), "UC bench elapsed");
            if (i >= 3U) samples.push_back(ms * 1000.0F / static_cast<float>(sets));
        }
    }
    f.runtime.tensors = original;
    f.owner.finish();
    std::string phases;
    if (std::getenv("STRIX_HC_PERSISTENT_TRACE")) {
        // Last launch: per phase, min / median / max over blocks, us after the earliest block start.
        const auto trace = hip_hyper_mix_testing_persistent_trace();
        std::uint64_t first = UINT64_MAX;
        for (std::size_t b = 0; b < 40U; ++b) first = std::min(first, trace[b * 8U]);
        // Per block: the last wave's unit end minus the first wave's (static form), and the block's B end.
        if (std::getenv("STRIX_HC_UC_WAVES")) {
            std::string blocks = " waves(block: first-last-Bdone us)";
            for (std::size_t b = 0; b < 40U; ++b) {
                std::uint64_t lo = UINT64_MAX, hi = 0;
                for (std::size_t w = 0; w < 16U; ++w) { const auto v = trace[40U * 8U + 16U + b * 16U + w]; lo = std::min(lo, v); hi = std::max(hi, v); }
                char cell[64];
                std::snprintf(cell, sizeof(cell), " %zu:%.1f-%.1f-%.1f", b, static_cast<double>(lo - first) / 100.0,
                    static_cast<double>(hi - first) / 100.0, static_cast<double>(trace[b * 8U + 3U] - first) / 100.0);
                blocks += cell;
            }
            phases += blocks;
        }
        phases += " phases(start,A,bar1,B,bar2,Cchain,end,unit0)";
        for (unsigned k = 0; k < 8U; ++k) {
            std::vector<double> v;
            for (std::size_t b = 0; b < 40U; ++b) v.push_back(static_cast<double>(trace[b * 8U + k] - first) / 100.0);
            std::sort(v.begin(), v.end());
            char cell[48];
            std::snprintf(cell, sizeof(cell), " %.1f/%.1f/%.1f", v.front(), v[v.size() / 2U], v.back());
            phases += cell;
        }
    }
    (void)hipEventDestroy(start); (void)hipEventDestroy(stop);
    check(hipHostUnregister(host), "unregister UC bench weights");
    std::free(host);
    std::sort(samples.begin(), samples.end());
    const char* b = std::getenv("STRIX_HC_PERSISTENT_B");
    const char* p = std::getenv("STRIX_HC_PERSISTENT");
    std::cout << "hc-uc-bench T=" << count << " sets=" << sets << " persistent=" << (p ? p : "-") << " B=" << (b ? b : "queue")
        << " us/mix min/med/max=" << samples.front() << '/' << samples[samples.size() / 2U] << '/' << samples.back()
        << " floor_us=" << static_cast<double>(set_bytes) / 236e3 << phases << '\n';
}
// Terminal head over real window row views: the configured prefetch R1 head
// per row is the oracle; device greedy rows land in host-mapped memory.
// Cohort heads of 9..32 rows: the rows of several 8-row windows (own arenas; the last one partly used), in
// launches of at most 16 rows, against every row's R1 head.
void family_lm_head_cohort_rows(std::uint32_t rows) {
    Owner owner;
    const auto model = ModelContract::qwen38_flash_next();
    const auto plan = build_serial_decode_execution_plan(model);
    const auto io = build_serial_execution_io_plan(plan);
    const auto step = plan.steps.back();
    constexpr std::uint32_t hidden = 2560U, vocab = 4099U;
    const auto& step_io = io.steps[step.id];
    ExecutionArenaPlan serial; serial.alignment = 256U;
    serial.allocations.push_back({step_io.reads[0], step.id, step.id, 0, hidden * 2U});
    serial.allocations.push_back({step_io.writes[0], step.id, step.id, 5120U, vocab * 4U + 64U});
    serial.total_bytes = (5120U + vocab * 4U + 64U + 255U) / 256U * 256U;
    serial.naive_nonaliased_bytes = serial.total_bytes;
    const auto window = build_known_window_arena(serial, 256U, 8U);
    const auto weights = q8(hidden, vocab, 5U);
    const auto device_weights = owner.allocate(weights.size());
    write(device_weights, weights);
    ModelManifest manifest; manifest.architecture = "qwen4_exp";
    ManifestTensor tensor; tensor.name = "output.weight"; tensor.role = TensorRole::output_head;
    tensor.dimensions = {hidden, vocab}; tensor.ggml_type = 8U; tensor.byte_size = weights.size();
    manifest.tensors.push_back(tensor); manifest.total_tensor_bytes = tensor.byte_size;
    manifest.layout_identity = model_manifest_layout_identity(manifest);
    ModelRuntimeBinding runtime; runtime.manifest_layout_identity = manifest.layout_identity;
    runtime.model_content_identity.bytes[0] = 1U;
    runtime.tensors.push_back({0, RuntimeTensorStorage::resident, device_weights.address, device_weights.bytes, 1, 0});
    ExecutionTensorAccessPlan access; access.manifest_layout_identity = manifest.layout_identity;
    access.plan_identity = 0x4b4e4f574e4c4d02ULL; access.accesses.push_back({step.id, SerialOp::lm_head, 0, {0}});
    const auto arenas = (rows + 7U) / 8U;
    std::vector<ExecutionMemoryBinding> memory(arenas);
    std::vector<ExecutionRuntimeResources> resources(arenas);
    for (std::uint32_t a = 0; a < arenas; ++a) {
        const auto scratch = owner.allocate(window.physical.total_bytes);
        memory[a] = {0x4b4e4f574e4c4d01ULL, scratch.address, scratch.bytes, 23U + a};
    }
    for (std::uint32_t a = 0; a < arenas; ++a) resources[a] = {&memory[a], &manifest, &runtime, &access};
    HipLmHeadLauncherConfig config{hidden, vocab, HipActivationStorage::bf16, true, true, true};
    const std::string id = "hip.lm_head.bf16.h2560.v4099.r1.c1.prefetch8-wave8-v1.virtual256.q8";
    std::vector<ExecutionDispatchContext> contexts(rows);
    std::vector<ExecutionBufferAddress> logits(rows);
    for (std::uint32_t r = 0; r < rows; ++r) {
        auto& c = contexts[r];
        const auto a = r / 8U;
        c.execution_identity = memory[a].execution_identity; c.step_id = step.id; c.op = step.op; c.kernel_id = id;
        c.reads = step_io.reads; c.writes = step_io.writes; c.arena = &window.row_view[r % 8U]; c.memory = &memory[a];
        c.resources = &resources[a]; c.position = r % 8U;
        const auto in = resolve_dispatch_read(c, 0);
        std::vector<std::uint16_t> bits(hidden);
        for (std::uint32_t i = 0; i < hidden; ++i)
            bits[i] = bf16(static_cast<float>(static_cast<int>((i * (2U * r + 5U)) % 23U) - 11) / 17.0F);
        write({in.address, hidden * 2U}, bytes_of(bits));
        logits[r] = resolve_dispatch_write(c, 0);
    }
    HipLaunchContext scope; scope.stream = owner.stream;
    const auto launcher = make_hip_lm_head_launcher(id, &config);
    const auto run = [&](bool candidate) {
        for (const auto& l : logits) check(hipMemset(reinterpret_cast<void*>(l.address), 0xff, l.bytes), "poison head rows");
        {
            const HipLaunchScope owned(scope);
            if (candidate) { lm_head_known_window_rows_preflight(contexts, config); lm_head_known_window_rows_launch(contexts, config); }
            else for (const auto& c : contexts) launcher.function(c, launcher.user_data);
        }
        owner.finish();
        std::vector<std::vector<unsigned char>> out;
        for (const auto& l : logits) out.push_back(read({l.address, l.bytes}));
        return out;
    };
    const auto reference = run(false);
    require(run(true) == reference && run(true) == reference, "cohort head rows differ from R1 head bytes or tails");
    std::cout << "LM head cohort rows " << rows << ": bit-exact against R1 heads\n";
}

void family_lm_head_rows(std::uint32_t rows) {
    Owner owner;
    const auto model = ModelContract::qwen38_flash_next();
    const auto plan = build_serial_decode_execution_plan(model);
    const auto io = build_serial_execution_io_plan(plan);
    const auto step = plan.steps.back();
    require(step.op == SerialOp::lm_head, "canonical plan ends in its LM head");
    constexpr std::uint32_t hidden = 2560U, vocab = 4099U;
    const auto& step_io = io.steps[step.id];
    ExecutionArenaPlan serial; serial.alignment = 256U;
    serial.allocations.push_back({step_io.reads[0], step.id, step.id, 0, hidden * 2U});
    serial.allocations.push_back({step_io.writes[0], step.id, step.id, 5120U, vocab * 4U + 64U});
    serial.total_bytes = (5120U + vocab * 4U + 64U + 255U) / 256U * 256U;
    serial.naive_nonaliased_bytes = serial.total_bytes;
    const auto window = build_known_window_arena(serial, 256U, rows);
    const auto scratch = owner.allocate(window.physical.total_bytes);
    const ExecutionMemoryBinding memory{0x4b4e4f574e4c4d01ULL, scratch.address, scratch.bytes, 23U};
    const auto weights = q8(hidden, vocab, 5U);
    const auto device_weights = owner.allocate(weights.size());
    write(device_weights, weights);
    ModelManifest manifest; manifest.architecture = "qwen4_exp";
    ManifestTensor tensor; tensor.name = "output.weight"; tensor.role = TensorRole::output_head;
    tensor.dimensions = {hidden, vocab}; tensor.ggml_type = 8U; tensor.byte_size = weights.size();
    manifest.tensors.push_back(tensor); manifest.total_tensor_bytes = tensor.byte_size;
    manifest.layout_identity = model_manifest_layout_identity(manifest);
    ModelRuntimeBinding runtime; runtime.manifest_layout_identity = manifest.layout_identity;
    runtime.model_content_identity.bytes[0] = 1U;
    runtime.tensors.push_back({0, RuntimeTensorStorage::resident, device_weights.address, device_weights.bytes, 1, 0});
    ExecutionTensorAccessPlan access; access.manifest_layout_identity = manifest.layout_identity;
    access.plan_identity = 0x4b4e4f574e4c4d02ULL; access.accesses.push_back({step.id, SerialOp::lm_head, 0, {0}});
    const ExecutionRuntimeResources resources{&memory, &manifest, &runtime, &access};
    HipLmHeadLauncherConfig config{hidden, vocab, HipActivationStorage::bf16, true, true, true};
    const std::string id = "hip.lm_head.bf16.h2560.v4099.r1.c1.prefetch8-wave8-v1.virtual256.q8";
    std::vector<ExecutionDispatchContext> contexts(rows);
    std::vector<ExecutionBufferAddress> logits(rows);
    for (std::uint32_t r = 0; r < rows; ++r) {
        auto& c = contexts[r];
        c.execution_identity = memory.execution_identity; c.step_id = step.id; c.op = step.op; c.kernel_id = id;
        c.reads = step_io.reads; c.writes = step_io.writes; c.arena = &window.row_view[r]; c.memory = &memory;
        c.resources = &resources; c.position = r;
        const auto in = resolve_dispatch_read(c, 0);
        std::vector<std::uint16_t> bits(hidden);
        for (std::uint32_t i = 0; i < hidden; ++i)
            bits[i] = bf16(static_cast<float>(static_cast<int>((i * (2U * r + 5U)) % 23U) - 11) / 17.0F);
        write({in.address, hidden * 2U}, bytes_of(bits));
        logits[r] = resolve_dispatch_write(c, 0);
    }
    HipLaunchContext scope; scope.stream = owner.stream;
    const auto launcher = make_hip_lm_head_launcher(id, &config);
    const auto run = [&](bool candidate) {
        for (const auto& l : logits) check(hipMemset(reinterpret_cast<void*>(l.address), 0xff, l.bytes), "poison head rows");
        {
            const HipLaunchScope owned(scope);
            if (candidate) { lm_head_known_window_rows_preflight(contexts, config); lm_head_known_window_rows_launch(contexts, config); }
            else for (const auto& c : contexts) launcher.function(c, launcher.user_data);
        }
        owner.finish();
        std::vector<std::vector<unsigned char>> out;
        for (const auto& l : logits) out.push_back(read({l.address, l.bytes}));
        return out;
    };
    const auto reference = run(false);
    require(run(true) == reference && run(true) == reference, "window head rows differ from R1 head bytes or tails");
    {
        // Q6P32 copy of the (six-bit) head: the same bytes. Windows of up to 4 rows and the R1 head read
        // the copy (the Q8_0 head is poisoned for them); wider windows keep the Q8_0 head.
        std::vector<unsigned char> packed(q6p::q6p32_bytes(vocab, hidden));
        require(q6p::to_q6p32(weights.data(), vocab, hidden, packed.data()), "head fixture is not six-bit");
        const auto device_q6 = owner.allocate(packed.size());
        write(device_q6, packed);
        if (rows <= 4U) check(hipMemset(reinterpret_cast<void*>(device_weights.address), 0xa5, device_weights.bytes), "poison Q8 head");
        q6p::register_tensor(reinterpret_cast<const void*>(device_weights.address),
            {reinterpret_cast<const unsigned char*>(device_q6.address), vocab, hidden, q6p::Kind::q6p32});
        auto hits = q6p::hits();
        const auto q6_rows = run(true);
        const auto window_hits = q6p::hits() - hits;
        check(hipMemset(reinterpret_cast<void*>(device_weights.address), 0xa5, device_weights.bytes), "poison Q8 head");
        hits = q6p::hits();
        const auto q6_serial = run(false);
        const auto serial_hits = q6p::hits() - hits;
        q6p::clear();
        write(device_weights, weights);
        require(q6_rows == reference && window_hits == (rows <= 4U ? 1U : 0U), "Q6P32 window head rows differ from the Q8_0 head");
        require(q6_serial == reference && serial_hits == rows, "Q6P32 R1 head differs from the Q8_0 head or skipped the copy");
    }
    require(window.row_stride[1] != 0U, "window head rows use striped row views");
    check(hipStreamBeginCapture(owner.stream, hipStreamCaptureModeThreadLocal), "begin head rows capture");
    bool rejected = false;
    { const HipLaunchScope owned(scope);
      try { lm_head_known_window_rows_launch(contexts, config); } catch (const std::invalid_argument&) { rejected = true; } }
    hipGraph_t graph{};
    check(hipStreamEndCapture(owner.stream, &graph), "end head rows capture");
    owner.graphs.push_back(graph);
    std::size_t nodes = 0;
    check(hipGraphGetNodes(graph, nullptr, &nodes), "count head rows capture nodes");
    require(rejected && nodes == 0U, "eager-only head rows reject stream capture before any enqueue");
    const auto partials = owner.allocate(hip_greedy_rows_scratch_bytes(rows, vocab));
    HipGreedyRowsReply* replies = nullptr;
    check(hipHostMalloc(reinterpret_cast<void**>(&replies), rows * sizeof(*replies), hipHostMallocMapped | hipHostMallocCoherent),
        "allocate mapped head replies");
    std::fill(replies, replies + rows, HipGreedyRowsReply{~0U, ~0U});
    {
        const HipLaunchScope owned(scope);
        lm_head_known_window_rows_launch(contexts, config);
        ExecutionBufferAddress p; p.address = partials.address; p.bytes = partials.bytes; p.allocation_generation = 1;
        enqueue_hip_greedy_rows(logits, vocab, p, replies, owner.stream);
    }
    owner.finish();
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::vector<float> values(vocab);
        std::memcpy(values.data(), reference[r].data(), vocab * 4U);
        std::uint32_t best = 0;
        for (std::uint32_t i = 1; i < vocab; ++i) if (values[i] > values[best]) best = i;
        require(replies[r].token == best && replies[r].status == 0U, "mapped greedy row differs from first-max R1 logits");
    }
    check(hipHostFree(replies), "free mapped head replies");
}
// Router/combine/inject rows on the real parallel-router serial arena: every
// row of one rows launch must equal T serial launches byte for byte across the
// complete window arena (outputs, router logits workspace, untouched bytes).
struct RowsStepFixture {
    Owner owner;
    ModelContract model = ModelContract::qwen38_flash_next();
    SerialExecutionPlan execution = build_serial_decode_execution_plan(model);
    SerialExecutionIoPlan io = build_serial_execution_io_plan(execution, true);
    ExecutionArenaPlan serial;
    KnownWindowArena window;
    Span scratch, router;
    ExecutionMemoryBinding memory;
    ModelManifest manifest;
    ModelRuntimeBinding runtime;
    ExecutionTensorAccessPlan access;
    std::shared_ptr<const WaveRouterManifestAdmission> admission;
    ExecutionRuntimeResources resources;
    HipElementwiseLauncherConfig elementwise{2560U, 4U, HipActivationStorage::bf16};
    std::vector<unsigned char> seed;
    const std::uint32_t rows;
    explicit RowsStepFixture(std::uint32_t row_count, std::uint32_t router_type = 0U) : rows(row_count) {
        ExecutionBufferRequest request; request.parallel_moe_router = true;
        serial = build_serial_execution_arena_plan(model, request, execution, io, 256U);
        window = build_known_window_arena(serial, 256U, rows);
        scratch = owner.allocate(window.physical.total_bytes);
        memory = {0x4b4e4f574e523301ULL + rows, scratch.address, scratch.bytes, 31U};
        manifest.architecture = "qwen4_exp";
        const auto route = step(SerialOp::moe_route, 0U);
        ManifestTensor tensor;
        tensor.name = "blk.0.ffn_gate_inp.weight"; tensor.role = TensorRole::moe_router; tensor.layer = 0U;
        tensor.ggml_type = router_type; tensor.dimensions = {2560U, 512U};
        std::vector<unsigned char> bytes;
        if (router_type == 8U) bytes = q8(2560U, 512U, 9U);
        else {
            // Experts 7/300 and 11/12 are identical: exact logit ties exercise index order.
            std::vector<float> values(2560U * 512U);
            for (std::size_t i = 0; i < values.size(); ++i) {
                const auto expert = i / 2560U, feature = i % 2560U;
                const auto source = expert == 300U ? 7U : expert == 12U ? 11U : expert;
                values[i] = 0.0007F * static_cast<float>(static_cast<int>((feature * 13U + source * 29U) % 61U) - 30) *
                    (1.0F + 0.01F * static_cast<float>(source % 17U));
            }
            if (router_type == 30U) {
                std::vector<std::uint16_t> half(values.size());
                for (std::size_t i = 0; i < values.size(); ++i) half[i] = bf16(values[i]);
                bytes = bytes_of(half);
            } else bytes = bytes_of(values);
        }
        tensor.byte_size = bytes.size();
        router = owner.allocate(bytes.size()); write(router, bytes);
        manifest.total_tensor_bytes = tensor.byte_size;
        manifest.tensors.push_back(tensor);
        manifest.layout_identity = model_manifest_layout_identity(manifest);
        runtime.manifest_layout_identity = manifest.layout_identity;
        runtime.model_content_identity.bytes[0] = 1U;
        runtime.tensors.push_back({0U, RuntimeTensorStorage::resident, router.address, router.bytes, 7U, 0U});
        access.manifest_layout_identity = manifest.layout_identity;
        access.plan_identity = 0x4b4e4f574e523302ULL;
        access.accesses.push_back({route.id, route.op, 0U, {0U}});
        admission = WaveRouterManifestAdmission::create(manifest);
        resources = {&memory, &admission->manifest(), &runtime, &access};
        // Finite moderate BF16 everywhere (so every FP32 view is finite too),
        // plus signed zeros, subnormals, infinities and NaNs at fixed strides.
        seed.resize(static_cast<std::size_t>(scratch.bytes));
        constexpr std::array<std::uint16_t, 8> edge{0x0000, 0x8000, 0x0001, 0x8001, 0x7f80, 0xff80, 0x7fc1, 0xffc3};
        std::uint32_t state = 0x9e3779b9U ^ rows;
        for (std::size_t i = 0; i + 1U < seed.size(); i += 2U) {
            state = state * 1664525U + 1013904223U;
            auto value = bf16(static_cast<float>(static_cast<int>(state >> 20U) - 2048) * 0.00061F);
            if (i % 4093U == 0U) value = edge[(i / 4093U) % edge.size()];
            std::memcpy(seed.data() + i, &value, sizeof(value));
        }
    }
    const ExecutionStep& step(SerialOp op, std::uint32_t layer) const {
        const auto found = std::find_if(execution.steps.begin(), execution.steps.end(),
            [&](const auto& value) { return value.op == op && value.layer == layer; });
        require(found != execution.steps.end(), "rows fixture operator missing");
        return *found;
    }
    std::vector<ExecutionDispatchContext> contexts(const ExecutionStep& s, std::string_view id) const {
        std::vector<ExecutionDispatchContext> out(rows);
        for (std::uint32_t r = 0; r < rows; ++r) {
            auto& c = out[r];
            c.execution_identity = memory.execution_identity; c.step_id = s.id; c.op = s.op; c.layer = s.layer;
            c.kernel_id = id; c.reads = io.steps[s.id].reads; c.writes = io.steps[s.id].writes;
            c.temporaries = io.steps[s.id].temporaries; c.arena = &window.row_view[r]; c.memory = &memory;
            c.resources = &resources; c.reads_state = s.reads_state; c.writes_state = s.writes_state;
            c.position = 5U + r; c.input_token_id = 40U + r;
        }
        return out;
    }
    HipLaunchContext scope() const { HipLaunchContext result; result.stream = owner.stream; return result; }
    template <class Work> std::vector<unsigned char> run(Work&& work, const std::vector<unsigned char>& input) {
        write(scratch, input);
        { const HipLaunchScope owned(scope()); work(); }
        owner.finish();
        guards(scratch); guards(router);
        return read(scratch);
    }
};
HipMoeRouteLauncherConfig router_config(const RowsStepFixture& f, bool staged, bool ranked, bool wave) {
    HipMoeRouteLauncherConfig config;
    config.hidden_size = 2560U; config.expert_count = 512U; config.top_k = 10U;
    config.normalize_selected = f.model.norm_topk_probability; config.parallel_projection = true;
    config.staged_logits = staged; config.ranked_topk = ranked; config.wave_topk = wave;
    if (wave) config.wave_manifest_admission = f.admission;
    return config;
}
template <class Serial, class Rows>
void rows_equal_serial(RowsStepFixture& f, std::span<const ExecutionDispatchContext> contexts,
    Serial&& serial, Rows&& rows, const char* what) {
    // Two different inputs: the seed and the seed rotated by one row stride.
    for (unsigned variant = 0; variant < 2U; ++variant) {
        auto input = f.seed;
        if (variant) std::rotate(input.begin(), input.begin() + static_cast<std::ptrdiff_t>(f.window.row_stride.front()), input.end());
        const auto reference = f.run([&] { for (const auto& c : contexts) serial(c); }, input);
        require(reference != input, "rows fixture serial launches wrote nothing");
        const auto candidate = f.run([&] { rows(contexts); }, input);
        if (candidate != reference) throw std::runtime_error(std::string(what) + ": rows launch differs from serial row launches");
        require(f.run([&] { rows(contexts); }, input) == candidate, "rows launch is not deterministic");
    }
}
template <class Preflight> void rows_rejections(RowsStepFixture& f, const std::vector<ExecutionDispatchContext>& contexts,
    Preflight&& preflight) {
    const auto bad = [&](auto mutate) {
        auto copy = contexts; mutate(copy);
        invalid([&] { preflight(std::span<const ExecutionDispatchContext>(copy)); });
    };
    invalid([&] { preflight(std::span<const ExecutionDispatchContext>(contexts.data(), 3U)); });
    bad([](auto& c) { c.back().position += 1U; });
    bad([](auto& c) { c.back().step_id += 1U; });
    bad([](auto& c) { c.back().execution_identity ^= 1U; });
    bad([](auto& c) { c.back().logical_rows = 2U; });
    bad([](auto& c) { c.back().arena = c.front().arena; }); // cross-row output alias
    bad([](auto& c) { c.back().kernel_id = "hip.other"; });
    static_cast<void>(f);
}
void router_rows_family() {
    for (const auto type : {0U, 30U, 8U}) for (const auto rows : {2U, 4U, 8U}) {
        RowsStepFixture f(rows, type);
        const auto& s = f.step(SerialOp::moe_route, 0U);
        struct Mode { bool staged, ranked, wave; const char* name; };
        for (const auto mode : {Mode{true, false, true, "staged+wave"}, Mode{false, false, true, "wave"},
                                Mode{false, true, false, "ranked"}, Mode{false, false, false, "sorted"},
                                Mode{true, false, false, "staged+sorted"}}) {
            auto config = router_config(f, mode.staged, mode.ranked, mode.wave);
            const auto launcher = make_hip_moe_route_launcher("hip.moe_route.test.parallel", &config);
            const auto contexts = f.contexts(s, launcher.key);
            rows_equal_serial(f, contexts, [&](const ExecutionDispatchContext& c) { launcher.function(c, launcher.user_data); },
                [&](std::span<const ExecutionDispatchContext> c) {
                    moe_route_known_window_rows_preflight(c, config); moe_route_known_window_rows_launch(c, config);
                }, mode.name);
            if (rows == 4U) rows_rejections(f, contexts, [&](std::span<const ExecutionDispatchContext> c) {
                moe_route_known_window_rows_preflight(c, config); });
        }
        auto serial_only = router_config(f, false, false, false); serial_only.parallel_projection = false;
        const auto contexts = f.contexts(s, "hip.moe_route.test");
        invalid([&] { moe_route_known_window_rows_preflight(contexts, serial_only); });
        if (type == 0U) {
            // A registered BF16 copy of BF16-exact F32 weights (as a dense Q6P32 pack provides it): same bytes.
            const auto original = read(f.router);
            std::vector<float> exact(original.size() / 4U);
            std::vector<std::uint16_t> half(exact.size());
            for (std::size_t i = 0; i < exact.size(); ++i) {
                float value; std::memcpy(&value, original.data() + 4U * i, 4U);
                half[i] = bf16(value);
                exact[i] = std::bit_cast<float>(static_cast<std::uint32_t>(half[i]) << 16U);
            }
            write(f.router, bytes_of(exact));
            const auto copy = f.owner.allocate(half.size() * 2U);
            write(copy, bytes_of(half));
            auto config = router_config(f, false, false, true);
            const auto launcher = make_hip_moe_route_launcher("hip.moe_route.test.parallel", &config);
            const auto rows_contexts = f.contexts(s, launcher.key);
            const auto launch = [&] { moe_route_known_window_rows_preflight(rows_contexts, config);
                moe_route_known_window_rows_launch(rows_contexts, config); };
            const auto reference = f.run(launch, f.seed);
            q6p::register_tensor(reinterpret_cast<const void*>(f.router.address),
                {reinterpret_cast<const unsigned char*>(copy.address), 512U, 2560U, q6p::Kind::bf16});
            const auto hits = q6p::hits();
            const auto candidate = f.run(launch, f.seed);
            const auto used = q6p::hits() - hits;
            q6p::clear();
            write(f.router, original);
            require(candidate == reference && used == 1U, "BF16 router rows differ from the F32 rows or skipped the copy");
        }
    }
    std::cout << "router rows 2/4/8 x F32/BF16/Q8_0 x staged/wave/ranked/sorted equal serial bytes; BF16 copies equal F32\n";
}
void elementwise_rows_family() {
    for (const auto rows : {2U, 4U, 8U}) {
        RowsStepFixture f(rows);
        for (const auto op : {SerialOp::attn_inject, SerialOp::mlp_inject, SerialOp::moe_combine}) for (const auto layer : {0U, 47U}) {
            const auto& s = f.step(op, layer);
            const bool inject = op != SerialOp::moe_combine;
            const auto launcher = inject ? make_hip_inject_launcher("hip.inject.test", &f.elementwise) :
                make_hip_moe_combine_launcher("hip.moe_combine.test", &f.elementwise);
            const auto contexts = f.contexts(s, launcher.key);
            const auto preflight = [&](std::span<const ExecutionDispatchContext> c) {
                if (inject) inject_known_window_rows_preflight(c, f.elementwise); else moe_combine_known_window_rows_preflight(c, f.elementwise);
            };
            rows_equal_serial(f, contexts, [&](const ExecutionDispatchContext& c) { launcher.function(c, launcher.user_data); },
                [&](std::span<const ExecutionDispatchContext> c) {
                    preflight(c);
                    if (inject) inject_known_window_rows_launch(c, f.elementwise); else moe_combine_known_window_rows_launch(c, f.elementwise);
                }, inject ? "inject" : "combine");
            if (rows == 4U && layer == 0U) rows_rejections(f, contexts, preflight);
            // Cohort inject / combine launch: the window's rows as session segments of 2/4 rows, one launch.
            if (rows == 8U)
                for (const auto& sizes : std::vector<std::vector<std::size_t>>{{4U, 4U}, {2U, 2U, 4U}, {2U, 2U, 2U, 2U}}) {
                    rows_equal_serial(f, contexts, [&](const ExecutionDispatchContext& c) { launcher.function(c, launcher.user_data); },
                        [&](std::span<const ExecutionDispatchContext> c) {
                            std::vector<std::span<const ExecutionDispatchContext>> segments;
                            std::size_t first = 0;
                            for (const auto size : sizes) { segments.push_back(c.subspan(first, size)); first += size; }
                            if (inject) inject_known_window_cohort_rows_launch(segments, f.elementwise);
                            else moe_combine_known_window_cohort_rows_launch(segments, f.elementwise);
                        }, inject ? "inject cohort" : "combine cohort");
                }
        }
    }
    std::cout << "attn/mlp inject and MoE combine rows 2/4/8 (and cohorts of 2/4-row segments) equal serial bytes\n";
}

// hipEvent timing: median over 25 samples (after 5 warmups) of 20 back-to-back
// repetitions, so the GPU stays clocked up as in decode. "cold" sweeps 96 MB
// (read only: no dirty write-back inside the timed work) through the 32 MB
// MALL before every repetition, as decode evicts each layer's router between
// tokens, and subtracts the sweep-only median.
__global__ void bench_sweep_kernel(const uint4* data, std::uint64_t count, unsigned* sink) {
    unsigned accumulator = 0;
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * 256U + threadIdx.x; i < count; i += 256ULL * gridDim.x) {
        const auto v = data[i]; accumulator ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (accumulator == 0x12345679U) *sink = accumulator;
}
template <class Work> float median_us(RowsStepFixture& f, Span flush, bool cold, Work&& work) {
    hipEvent_t begin{}, end{};
    check(hipEventCreate(&begin), "create bench event"); check(hipEventCreate(&end), "create bench event");
    const auto measure = [&](bool with_work) {
        std::vector<float> samples;
        const HipLaunchScope owned(f.scope());
        for (int i = 0; i < 30; ++i) {
            check(hipEventRecord(begin, f.owner.stream), "record bench begin");
            for (int repeat = 0; repeat < 20; ++repeat) {
                if (cold) {
                    hipLaunchKernelGGL(bench_sweep_kernel, dim3(640), dim3(256), 0, f.owner.stream,
                        reinterpret_cast<const uint4*>(flush.address), flush.bytes / 16U, reinterpret_cast<unsigned*>(flush.address));
                    check(hipGetLastError(), "sweep MALL");
                }
                if (with_work) work();
            }
            check(hipEventRecord(end, f.owner.stream), "record bench end");
            check(hipEventSynchronize(end), "complete bench sample");
            float ms = 0.0F; check(hipEventElapsedTime(&ms, begin, end), "bench elapsed");
            if (i >= 5) samples.push_back(ms * 1000.0F / 20.0F);
        }
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2U];
    };
    const auto result = measure(true) - (cold ? measure(false) : 0.0F);
    check(hipEventDestroy(begin), "destroy bench event"); check(hipEventDestroy(end), "destroy bench event");
    return result;
}
void rows_bench() {
    for (const auto rows : {2U, 4U, 8U}) {
        RowsStepFixture f(rows);
        const auto flush = f.owner.allocate(96ULL << 20U);
        write(f.scratch, f.seed);
        auto config = router_config(f, true, false, true);
        const auto launcher = make_hip_moe_route_launcher("hip.moe_route.test.parallel", &config);
        const auto route = f.contexts(f.step(SerialOp::moe_route, 0U), launcher.key);
        for (const bool cold : {false, true}) {
            const auto serial = median_us(f, flush, cold, [&] { for (const auto& c : route) launcher.function(c, launcher.user_data); });
            const auto multi = median_us(f, flush, cold, [&] { moe_route_known_window_rows_launch(route, config); });
            std::cout << "bench router staged+wave T=" << rows << (cold ? " cold" : " warm") << ": " << rows << "x serial "
                      << serial << " us, rows " << multi << " us\n";
        }
        for (const auto op : {SerialOp::attn_inject, SerialOp::moe_combine}) {
            const bool inject = op == SerialOp::attn_inject;
            const auto el = inject ? make_hip_inject_launcher("hip.inject.test", &f.elementwise) :
                make_hip_moe_combine_launcher("hip.moe_combine.test", &f.elementwise);
            const auto contexts = f.contexts(f.step(op, 0U), el.key);
            const auto serial = median_us(f, flush, false, [&] { for (const auto& c : contexts) el.function(c, el.user_data); });
            const auto multi = median_us(f, flush, false, [&] {
                if (inject) inject_known_window_rows_launch(contexts, f.elementwise); else moe_combine_known_window_rows_launch(contexts, f.elementwise); });
            std::cout << "bench " << (inject ? "inject" : "combine") << " T=" << rows << " warm: " << rows << "x serial "
                      << serial << " us, rows " << multi << " us\n";
        }
    }
}
} // namespace

int main(int argc, char** argv) try {
    if (argc > 1 && std::string_view(argv[1]) == "--qsa-pool-lifecycle") { qsa_pool_idle_owner(); return 0; }
    if (argc > 1 && std::string_view(argv[1]) == "--qsa-pool-tests") {
        for (bool prefetch : {false, true}) for (bool bf16_indexer : {false, true})
            qsa_rows_family(prefetch, prefetch, bf16_indexer);
        qsa_pool_idle_owner();
        return 0;
    }
    if (std::getenv("STRIX_QSA_LONG_BENCH")) { bench_qsa_long_context(); return 0; }
    if (argc > 1 && std::string_view(argv[1]) == "--rows-bench") { rows_bench(); return 0; }
    if (argc > 1 && std::string_view(argv[1]) == "--window-folds") {
        // attn inject -> MLP mix, MLP inject -> next attention mix (layer 2: PLE follows layer 0's MLP inject),
        // last MLP inject -> final mix; with combine, the MoE combine before each MLP inject as well.
        const auto last = static_cast<std::uint32_t>(ModelContract::qwen38_flash_next().hidden_layers - 1U);
        std::uint32_t folded = 0, cases = 0;
        for (const auto& [op, layer, combine] : {std::tuple{SerialOp::attn_inject, 0U, false},
                 std::tuple{SerialOp::mlp_inject, 2U, false}, std::tuple{SerialOp::mlp_inject, last, false},
                 std::tuple{SerialOp::mlp_inject, 2U, true}, std::tuple{SerialOp::mlp_inject, last, true}})
            for (std::uint32_t rows : {2U, 4U, 8U}) {
                const bool f = family_window_fold(op, layer, combine, rows);
                require(f || combine, "a window inject fold fixture case was declined");
                folded += f ? 1U : 0U; ++cases;
            }
        std::cout << "window HC folds: " << folded << " of " << cases << " cases folded (declined combine folds checked through their fallback), all equal the separate launches\n";
        // Inject folds always apply here; a combine fold declines when another row's streams overlap a row's
        // shared-expert output (the fallback is checked above).
        require(folded >= 9U, "a window inject fold fixture case was declined");
        // NaN-payload corpora (fold_corpus_seed flavors 1..5: specials, all-NaN, NaNs of different payloads in different
        // leaves of one stream, NaN norm weights; NaN gates, shared and routed values): every byte, HC workspace included.
        std::uint32_t nan_folded = 0, nan_cases = 0;
        for (unsigned flavor = 1; flavor < 6U; ++flavor)
            for (const auto& [op, layer, combine] : {std::tuple{SerialOp::attn_inject, 0U, false},
                     std::tuple{SerialOp::mlp_inject, 2U, false}, std::tuple{SerialOp::mlp_inject, last, false},
                     std::tuple{SerialOp::mlp_inject, 2U, true}, std::tuple{SerialOp::mlp_inject, last, true}})
                for (std::uint32_t rows : {2U, 4U, 8U}) {
                    const bool f = family_window_fold(op, layer, combine, rows, flavor, true);
                    require(f || combine, "a NaN-corpus window inject fold fixture case was declined");
                    nan_folded += f ? 1U : 0U; ++nan_cases;
                }
        std::cout << "window HC folds, NaN corpora: " << nan_folded << " of " << nan_cases
                  << " cases folded, every byte (HC workspace included) equals the separate launches\n";
        // The routed commit + MoE combine fold against the commit and combine kernels on rows of any bits.
        std::uint32_t routed = 0;
        for (std::uint32_t rows : {2U, 4U, 8U})
            for (std::uint32_t flavor = 0; flavor < 9U; ++flavor) {
                const auto r = known_moe_commit_combine_selftest(rows, flavor, 7U + rows * 13U + flavor);
                if (r[1] != 0.0)
                    std::cout << "FAIL commit + combine rows " << rows << " flavor " << flavor << ": " << r[1] << " of " << r[0] << " words differ\n";
                require(r[1] == 0.0, "commit + combine fold differs from the commit and combine kernels");
                ++routed;
            }
        std::cout << "commit + combine fold: " << routed << " cases, every byte equals the commit and combine kernels\n";
        return 0;
    }
    // FP32 streams (HipActivationStorage::fp32): the window folds (qpf2k_scale's FP32 order, inject_rows_kernel<float>,
    // combine_rows_kernel<float>) and, with the persistent HC enabled, its FP32 instantiations, each against the
    // separate launches over the finite and NaN-payload corpora (FP32 values with random low payload bits).
    if (argc > 1 && std::string_view(argv[1]) == "--folds-fp32") {
        fold_activation_bytes = 4U;
        const auto last = static_cast<std::uint32_t>(ModelContract::qwen38_flash_next().hidden_layers - 1U);
        std::uint32_t folded = 0, cases = 0;
        for (unsigned flavor = 0; flavor < 6U; ++flavor)
            for (const auto& [op, layer, combine] : {std::tuple{SerialOp::attn_inject, 0U, false},
                     std::tuple{SerialOp::mlp_inject, 2U, false}, std::tuple{SerialOp::mlp_inject, last, false},
                     std::tuple{SerialOp::mlp_inject, 2U, true}, std::tuple{SerialOp::mlp_inject, last, true}})
                for (std::uint32_t rows : {2U, 4U, 8U}) {
                    const bool f = family_window_fold(op, layer, combine, rows, flavor, true);
                    require(f || combine, "an FP32 window inject fold fixture case was declined");
                    folded += f ? 1U : 0U; ++cases;
                }
        std::cout << "FP32 window HC folds: " << folded << " of " << cases << " cases folded, every byte equals the separate launches\n";
        if (strix_hc_persistent_enabled()) {
            std::uint32_t fused = 0, pairs = 0;
            const auto launches = hip_hyper_mix_testing_persistent_launches();
            for (unsigned flavor = 0; flavor < 6U; ++flavor)
                for (const auto& [op, layer] : {std::pair{SerialOp::attn_inject, 0U}, std::pair{SerialOp::mlp_inject, 1U},
                         std::pair{SerialOp::mlp_inject, last}})
                    for (std::uint32_t rows : {2U, 4U, 8U}) { fused += family_hyper_inject_rows(op, layer, rows, flavor, true) ? 1U : 0U; ++pairs; }
            require(hip_hyper_mix_testing_persistent_launches() > launches, "FP32 persistent cases ran no persistent launch");
            std::cout << "FP32 persistent HC: " << pairs << " pairs equal the separate launches (" << fused << " with the inject folded)\n";
        }
        return 0;
    }
    // HC rows only (the persistent ctest runs this with STRIX_HC_PERSISTENT=1 and requires its launches).
    if (argc > 1 && std::string_view(argv[1]) == "--hc-uc-bench") {
        const auto sets = argc > 2 ? static_cast<std::uint32_t>(std::atoi(argv[2])) : 24U;
        for (std::uint32_t rows : {2U, 4U, 8U}) bench_hyper_rows_uc(rows, sets);
        return 0;
    }
    if (argc > 1 && std::string_view(argv[1]) == "--hc-rows") {
        std::uint32_t cases = 0;
        for (auto op : {SerialOp::attn_hyper_mix, SerialOp::mlp_hyper_mix, SerialOp::final_hyper_mix})
            for (bool splitk : {false, true}) for (bool lane : {false, true}) for (std::uint32_t rows : {2U, 4U, 8U}) {
                family_hyper_rows(op, splitk, lane, rows); ++cases;
            }
        const auto persistent = hip_hyper_mix_testing_persistent_launches();
        std::cout << "known-window HC rows bit-exact cases passed: " << cases << ", persistent launches " << persistent << '\n';
        const auto on = [](const char* name) { const char* v = std::getenv(name); return v && std::string_view(v) == "1"; };
        if (on("STRIX_HC_PERSISTENT")) require(persistent > 0U, "STRIX_HC_PERSISTENT=1 ran no persistent HC launch");
        std::uint32_t fused = 0, pairs = 0;
        const auto last = ModelContract::qwen38_flash_next().hidden_layers - 1U;
        for (const auto& [op, layer] : {std::pair{SerialOp::attn_inject, 0U}, std::pair{SerialOp::mlp_inject, 1U},
                 std::pair{SerialOp::mlp_inject, static_cast<std::uint32_t>(last)}})
            for (std::uint32_t rows : {2U, 4U, 8U}) { fused += family_hyper_inject_rows(op, layer, rows) ? 1U : 0U; ++pairs; }
        std::cout << "fused inject + HC rows: " << fused << " of " << pairs << " pairs folded, all equal the two launches\n";
        if (on("STRIX_HC_PERSISTENT") && on("STRIX_HC_PERSISTENT_INJECT"))
            require(fused == pairs, "STRIX_HC_PERSISTENT_INJECT=1 declined a fixture pair");
        // NaN-payload corpora (fold_corpus_seed / fold_norm_weights flavors 1..5) against the separate launches: the
        // persistent HC step alone over the whole arena, and the persistent inject fold over every live byte.
        // STRIX_HC_PERSISTENT=1, or STRIX_DISPATCH_FUSE=1 alone (persistent at 2 and 4 rows, 8 rows separate).
        if (strix_hc_persistent_enabled()) {
            std::uint32_t nan_fused = 0, nan_pairs = 0;
            const auto launches = hip_hyper_mix_testing_persistent_launches();
            for (unsigned flavor = 1; flavor < 6U; ++flavor)
                for (const auto& [op, layer] : {std::pair{SerialOp::attn_inject, 0U}, std::pair{SerialOp::mlp_inject, 1U},
                         std::pair{SerialOp::mlp_inject, static_cast<std::uint32_t>(last)}})
                    for (std::uint32_t rows : {2U, 4U, 8U}) {
                        nan_fused += family_hyper_inject_rows(op, layer, rows, flavor, true) ? 1U : 0U; ++nan_pairs;
                    }
            require(hip_hyper_mix_testing_persistent_launches() > launches, "NaN-corpus persistent cases ran no persistent launch");
            if (on("STRIX_HC_PERSISTENT_INJECT")) require(nan_fused == nan_pairs, "STRIX_HC_PERSISTENT_INJECT=1 declined a NaN-corpus pair");
            std::cout << "persistent HC, NaN corpora: " << nan_pairs << " pairs equal the separate launches (" << nan_fused
                      << " with the inject folded)\n";
        }
        if (std::getenv("STRIX_HC_ROWS_BENCH"))
            for (bool cold : {false, true}) for (std::uint32_t rows : {2U, 4U, 8U}) {
                bench_hyper_rows(true, true, rows, cold);
                if (!std::getenv("STRIX_HC_PERSISTENT_TRACE")) continue;
                // Last launch: per phase, min / median / max over blocks, us after the earliest block start.
                const auto trace = hip_hyper_mix_testing_persistent_trace();
                std::uint64_t start = UINT64_MAX;
                for (std::size_t b = 0; b < 40U; ++b) start = std::min(start, trace[b * 8U]);
                std::cout << "  trace T=" << rows << (cold ? " cold" : " warm") << " (start, A done, barrier1, B done, barrier2, C chain, end):";
                for (unsigned k = 0; k < 7U; ++k) {
                    std::vector<double> v;
                    for (std::size_t b = 0; b < 40U; ++b) v.push_back(static_cast<double>(trace[b * 8U + k] - start) / 100.0);
                    std::sort(v.begin(), v.end());
                    std::cout << ' ' << v.front() << '/' << v[v.size() / 2U] << '/' << v.back();
                }
                std::cout << '\n';
            }
        return 0;
    }
    router_rows_family();
    elementwise_rows_family();
    family(false, false);
    family(false, true);
    family(true);
    family_gdn_f32_gates();
    for(auto rows:{16U,17U}) {family_prefill_tiles(false,rows);family_prefill_tiles(true,rows);}
    std::uint32_t hc_cases = 0;
    for (auto op : {SerialOp::attn_hyper_mix, SerialOp::mlp_hyper_mix, SerialOp::final_hyper_mix})
        for (bool splitk : {false, true}) for (bool lane : {false, true}) for (std::uint32_t rows : {2U, 4U, 8U}) {
            family_hyper_rows(op, splitk, lane, rows); ++hc_cases;
        }
    std::cout << "known-window HC rows bit-exact cases passed: " << hc_cases << '\n';
    if (std::getenv("STRIX_HC_ROWS_BENCH"))
        for (bool cold : {false, true}) for (bool splitk : {false, true}) for (bool lane : {false, true})
            for (std::uint32_t rows : {2U, 4U, 8U}) bench_hyper_rows(splitk, lane, rows, cold);
    qsa_rows_family(true, true, true); // real checkpoint formats
    qsa_rows_family(true, true, false);
    qsa_rows_family(false, false, true);
    qsa_rows_family(false, false, false);
    qsa_pool_idle_owner();
    for (bool dense : {false, true}) for (bool bf16_indexer : {true, false}) qsa_cohort_family(dense, bf16_indexer);
    for(auto rows:{2U,4U,8U}) family_lm_head_rows(rows);
    for (auto rows : {9U, 12U, 16U, 17U, 24U, 32U}) family_lm_head_cohort_rows(rows);
    std::cout << "known-window shared/GDN/QSA/LM-head component fixtures passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "known-window component fixture failed: " << error.what() << '\n';
    return 1;
}
