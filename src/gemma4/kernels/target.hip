#include <hip/hip_runtime.h>
#include "device.hpp"
#include "internal.hpp"
#include "a8_ops.hpp"
#include "weight_policy.hpp"
#include "gemma4/exl3/device.hpp"
#include <algorithm>
#include <cfenv>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <set>
#include <string_view>

namespace strix::gemma4::gpu {
namespace {
bool enabled(const char* name) {
    const auto* p = std::getenv(name);
    return !p || std::string_view(p) != "0";
}
bool output_tiling() {
    const auto* p = std::getenv("STRIX_G4_OUTPUT_TILE");
    return p && std::string_view(p) == "1";
}
bool prefetch_matrix(LinearView v) {
    return v.data && v.inputs >= 1024U && v.inputs % 16U == 0 && v.format == WeightFormat::bf16;
}
bool input_cache(const HostWeights& weights, bool fusion) {
    const auto* p = std::getenv("STRIX_G4_INPUT_CACHE");
    if (!p || std::string_view(p) != "1")
        return false;
    const auto eligible = [](LinearView v) {
        return v.data && v.inputs >= 256U && v.inputs <= 4096U && v.inputs % 16U == 0 && v.outputs <= 4096U;
    };
    if (eligible(weights.embedding) || eligible(weights.ple_model))
        return true;
    for (const auto& y : weights.layer) {
        for (const auto v : {y.q, y.k, y.v, y.o, y.down, y.ple_gate, y.ple_projection})
            if (eligible(v))
                return true;
        if (!fusion && (eligible(y.gate) || eligible(y.up)))
            return true;
    }
    return false;
}
bool chunk_prefetch(const HostWeights& weights) {
    const auto* p = std::getenv("STRIX_G4_CHUNK_PREFETCH");
    if (p)
        return std::string_view(p) == "1";
    if (prefetch_matrix(weights.embedding) || prefetch_matrix(weights.ple_model))
        return true;
    for (const auto& y : weights.layer)
        for (const auto v : {y.q, y.k, y.v, y.o, y.gate, y.up, y.down, y.ple_gate, y.ple_projection})
            if (prefetch_matrix(v))
                return true;
    return false;
}
bool packed_headers(const HostWeights& weights) {
    const auto* p = std::getenv("STRIX_G4_PACKED_HEADER");
    if (p && std::string_view(p) == "0")
        return false;
    for (const auto v : {weights.embedding, weights.ple_model})
        if (v.data && (v.format == WeightFormat::q4_k || v.format == WeightFormat::q6_k))
            return true;
    for (const auto& y : weights.layer)
        for (const auto v : {y.q, y.k, y.v, y.o, y.gate, y.up, y.down, y.ple_gate, y.ple_projection})
            if (v.data && (v.format == WeightFormat::q4_k || v.format == WeightFormat::q6_k))
                return true;
    return false;
}
bool streaming_matrices(const HostWeights& weights, unsigned rows) {
    const auto eligible = [rows](LinearView v) {
        return v.data && stream_weights(v.format, rows);
    };
    if (eligible(weights.embedding) || eligible(weights.ple_model))
        return true;
    for (const auto& y : weights.layer)
        for (const auto v : {y.q, y.k, y.v, y.o, y.gate, y.up, y.down, y.ple_gate, y.ple_projection})
            if (eligible(v))
                return true;
    return false;
}
void hip_checked(hipError_t e) {
    if (e != hipSuccess)
        throw std::runtime_error(std::string("device target: ") + hipGetErrorString(e));
}
class PinnedBuffer {
  public:
    PinnedBuffer() = default;
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    ~PinnedBuffer() {
        if (data_)
            (void)hipHostFree(data_);
    }
    void allocate(std::size_t bytes) {
        if (data_)
            throw std::runtime_error("pinned staging already allocated");
        hip_checked(hipHostMalloc(&data_, bytes, hipHostMallocDefault));
    }
    [[nodiscard]] void* data() const { return data_; }
    template <class T> [[nodiscard]] T* as() const { return static_cast<T*>(data_); }

  private:
    void* data_ = nullptr;
};
void expect(LinearView v, unsigned outputs, unsigned inputs) {
    if (!v.data || v.outputs != outputs || v.inputs != inputs ||
        (v.format == WeightFormat::exl3_mul1 ? !exl3::geometry(v) : v.row_bytes < packed_row_bytes(v.format, inputs)))
        throw std::runtime_error("incomplete device target tensor geometry");
}
void validate_weights(const HostWeights& w) {
    const auto& c = w.config;
    validate_config(c);
    if (w.layer.size() != c.layers)
        throw std::runtime_error("incomplete device layers");
    expect(w.embedding, c.vocab, c.hidden);
    if (w.head.data) expect(w.head, c.vocab, c.hidden);
    expect(w.ple_table, c.vocab, c.layers * c.ple_dim);
    expect(w.ple_model, c.layers * c.ple_dim, c.hidden);
    expect(w.ple_model_norm, 1, c.ple_dim);
    expect(w.final_norm, 1, c.hidden);
    for (unsigned l = 0; l < c.layers; ++l) {
        const auto& y = w.layer[l];
        const auto d = c.layer[l].head_dim;
        for (auto v : {y.input_norm, y.post_attention_norm, y.pre_ffn_norm, y.post_ffn_norm, y.post_ple_norm})
            expect(v, 1, c.hidden);
        expect(y.q, c.heads * d, c.hidden);
        expect(y.o, c.hidden, c.heads * d);
        expect(y.q_norm, 1, d);
        if (c.layer[l].producer == l) {
            expect(y.k, c.kv_heads * d, c.hidden);
            expect(y.v, c.kv_heads * d, c.hidden);
            expect(y.k_norm, 1, d);
        }
        expect(y.gate, c.inter, c.hidden);
        expect(y.up, c.inter, c.hidden);
        expect(y.down, c.hidden, c.inter);
        expect(y.ple_gate, c.ple_dim, c.hidden);
        expect(y.ple_projection, c.hidden, c.ple_dim);
        expect(y.scalar, 1, 1);
    }
}
} // namespace
struct DeviceTarget::Impl {
    HostWeights w;
    DeviceOptions options;
    DeviceStats stats;
    std::vector<Buffer> weights;
    std::map<const std::uint8_t*, LinearView> uploaded;
    Buffer sliding, global;
    unsigned table_positions = 0;
    Buffer residual, x, q, k, v, att, out, gate, up, ple, logits, taps, format_scratch;
    // g4-a8-v1 only: two slots of quantized rows. a8_source[s] names the FP32 buffer whose current
    // contents slot s holds; any other enqueued work may rewrite buffers, so run() forgets both.
    Buffer a8_codes[2], a8_meta[2];
    const void* a8_source[2] = {nullptr, nullptr};
    Buffer tokens, positions, reads, writes, scores, maxima, partials, failure, candidates, winners, attention_counters;
    Buffer tickets;
    PinnedBuffer host_tokens, host_positions, host_reads, host_writes, host_failure, host_logits, host_taps, host_winners,
        host_hidden;
    std::vector<Candidate> selected;
    hipEvent_t start = nullptr, stop = nullptr;
    hipStream_t queue = nullptr;
    struct Graph {
        hipGraph_t graph = nullptr;
        hipGraphExec_t executable = nullptr;
        std::size_t traffic = 0;
    };
    std::map<std::pair<unsigned, unsigned>, Graph> graphs;
    void clear_graphs() {
        for (auto& [key, g] : graphs) {
            (void)key;
            (void)hipGraphExecDestroy(g.executable);
            (void)hipGraphDestroy(g.graph);
        }
        graphs.clear();
    }

    ~Impl() {
        clear_graphs();
        if (queue)
            (void)hipStreamDestroy(queue);
        if (start)
            (void)hipEventDestroy(start);
        if (stop)
            (void)hipEventDestroy(stop);
    }
    LinearView upload(LinearView view) {
        if (!view.data)
            return {};
        const auto it = uploaded.find(view.data);
        if (it != uploaded.end()) {
            const auto d = it->second;
            if (d.inputs != view.inputs || d.outputs != view.outputs || d.row_bytes != view.row_bytes ||
                d.format != view.format)
                throw std::runtime_error("ambiguous aliased device weight");
            return d;
        }
        validate_finite(view);
        weights.emplace_back(weight_storage_bytes(view));
        weights.back().upload(view.data, weights.back().size());
        const auto original = view.data;
        view.data = weights.back().as<std::uint8_t>();
        stats.resident_weight_bytes += weights.back().size();
        uploaded.emplace(original, view);
        return view;
    }
    void run(const char* name, const std::function<void()>& f, bool keeps_a8 = false) {
        if (!keeps_a8)
            a8_source[0] = a8_source[1] = nullptr;
        if (!options.profile) {
            f();
            return;
        }
        hip_checked(hipEventRecord(start, queue));
        f();
        hip_checked(hipEventRecord(stop, queue));
        hip_checked(hipEventSynchronize(stop));
        float ms = 0;
        hip_checked(hipEventElapsedTime(&ms, start, stop));
        stats.kernel_ms[name] += ms;
    }
    void finite(const Buffer& b, std::size_t count) { check_finite(b.as<float>(), count, failure.as<unsigned>(), queue); }
    void input(Buffer& device, PinnedBuffer& staging, const void* source, std::size_t bytes) {
        if (stats.pinned_io) {
            std::memcpy(staging.data(), source, bytes);
            hip_checked(hipMemcpyAsync(device.data(), staging.data(), bytes, hipMemcpyHostToDevice, queue));
        } else {
            device.upload(source, bytes);
        }
    }
    A8Rows a8_rows(unsigned width, int slot) const {
        return {a8_codes[slot].as<std::int8_t>(), a8_meta[slot].as<a8::Meta>(), width};
    }
    // Requested quantized copy of a producer's output (slot 0), or none outside g4-a8-v1.
    A8Rows a8_output(unsigned width) const { return stats.a8 ? a8_rows(width, 0) : A8Rows{}; }
    void a8_produced(int slot, const Buffer& output) {
        for (auto& source : a8_source)
            if (source == output.data())
                source = nullptr;
        a8_source[slot] = output.data();
    }
    // The slot holding input's quantized rows; quantizes into slot 0 unless a producer already did.
    int a8_input(const Buffer& input, unsigned rows, unsigned width) {
        for (int slot = 0; slot < 2; ++slot)
            if (a8_source[slot] == input.data())
                return slot;
        run("a8_quantize", [&] { a8_quantize(input.as<float>(), rows, width, a8_rows(width, 0), queue); }, true);
        a8_produced(0, input);
        return 0;
    }
    // quantize_output: the stored outputs are also quantized into the other slot (outputs % 32 == 0).
    bool a8_dot(const char* name, LinearView matrix, const Buffer& input, Buffer& output, unsigned rows,
                const LinearEpilogue& epilogue, bool quantize_output = false) {
        if (!stats.a8 || !a8::quantized(matrix.format))
            return false;
        const int slot = a8_input(input, rows, matrix.inputs);
        quantize_output &= !epilogue.extra_out[0] && matrix.outputs % 32U == 0;
        bool launched = false;
        run(name, [&] {
            launched = a8_linear(matrix, a8_rows(matrix.inputs, slot), rows, output.as<float>(), epilogue, queue,
                                 failure.as<unsigned>(), quantize_output ? a8_rows(matrix.outputs, 1 - slot) : A8Rows{});
        }, true);
        for (auto& source : a8_source)
            if (source == output.data() || (epilogue.extra_out[0] && (source == epilogue.extra_out[0] ||
                                                                     source == epilogue.extra_out[1])))
                source = nullptr;
        if (launched) {
            stats.pass_weight_bytes += weight_storage_bytes(matrix);
            if (quantize_output)
                a8_source[1 - slot] = output.data();
        }
        return launched;
    }
    void dot(const char* name, LinearView matrix, const Buffer& input, Buffer& output, unsigned rows) {
        if (a8_dot(name, matrix, input, output, rows, {}))
            return;
        stats.pass_weight_bytes += weight_storage_bytes(matrix) * (options.shared_weights ? 1 : rows);
        run(name, [&] {
            linear(matrix, input.as<float>(), rows, output.as<float>(), options.shared_weights, queue,
                   stats.linear_finite || matrix.format == WeightFormat::exl3_mul1 ? failure.as<unsigned>() : nullptr,
                   format_scratch.data());
            if (!stats.linear_finite)
                finite(output, std::size_t(rows) * matrix.outputs);
        });
    }
    // EXL3 linears over one input share transform launches; bytes equal separate dots.
    bool dot_group(const char* name, std::initializer_list<std::pair<LinearView, Buffer*>> members, const Buffer& input,
                   unsigned rows) {
        LinearView views[exl3::max_group];
        float* outputs[exl3::max_group];
        unsigned count = 0;
        for (const auto& [view, output] : members) {
            if (view.format != WeightFormat::exl3_mul1 || count == exl3::max_group)
                return false;
            views[count] = view;
            outputs[count++] = output->as<float>();
        }
        for (const auto& member : members)
            stats.pass_weight_bytes += weight_storage_bytes(member.first) * (options.shared_weights ? 1 : rows);
        run(name, [&] {
            exl3::device_linear_group(views, count, input.as<float>(), rows, outputs, format_scratch.data(),
                                      failure.as<unsigned>(), options.shared_weights, queue);
            if (!stats.linear_finite)
                for (const auto& [view, output] : members)
                    finite(*output, std::size_t(rows) * view.outputs);
        });
        return true;
    }
    // A linear with its store/epilogue fused; false (nothing enqueued) selects the separate schedule.
    bool fused_dot(const char* name, LinearView matrix, const Buffer& input, Buffer& output, unsigned rows,
                   LinearEpilogue epilogue, bool quantize_output = false) {
        if (stats.a8 && a8::quantized(matrix.format))
            return a8_dot(name, matrix, input, output, rows, epilogue, quantize_output);
        if (!options.shared_weights)
            return false;
        bool launched = false;
        run(name, [&] {
            launched = linear_fused(matrix, input.as<float>(), rows, output.as<float>(), epilogue, queue,
                                    failure.as<unsigned>());
        });
        if (launched)
            stats.pass_weight_bytes += std::size_t(matrix.outputs) * matrix.row_bytes;
        return launched;
    }
    // quantize: g4-a8-v1 also writes the quantized rows of output when the schedule allows it.
    void normalize(const char* name, const Buffer& input, LinearView weight, Buffer& output, unsigned vectors,
                   unsigned width, bool quantize = false) {
        bool quantized = false;
        run(name, [&] {
            quantized = norm(input.as<float>(), weight, output.as<float>(), vectors, width, queue,
                             stats.norm_finite ? failure.as<unsigned>() : nullptr,
                             quantize ? a8_output(width) : A8Rows{});
            if (!stats.norm_finite)
                finite(output, std::size_t(vectors) * width);
        });
        if (quantized)
            a8_produced(0, output);
    }
    // Branch norm/residual epilogue; with g4-a8-v1 the following linear input (next, else residual)
    // is quantized in the same launch when quantize is set.
    void epilogue(const char* name, LinearView weight, unsigned rows, unsigned width, LinearView scalar, Buffer* next,
                  LinearView next_weight, bool quantize) {
        bool quantized = false;
        run(name, [&] {
            quantized = norm_residual(out.as<float>(), weight, residual.as<float>(), rows, width, scalar,
                                      next ? next->as<float>() : nullptr, next_weight, failure.as<unsigned>(), queue,
                                      quantize ? a8_output(width) : A8Rows{});
        });
        if (quantized)
            a8_produced(0, next ? *next : residual);
    }
};
DeviceTarget::DeviceTarget(const HostWeights& host, RopeTables tables, DeviceOptions options)
    : impl_(std::make_unique<Impl>()) {
    validate_weights(host);
    if (options.profile && options.graph_timing)
        throw std::runtime_error("graph timing and per-family profiling are mutually exclusive");
    require_device();
    if (std::fegetround() != FE_TONEAREST)
        throw std::runtime_error("g4-v1 requires RNE host environment");
    if (!tables.positions || tables.positions > max_positions ||
        tables.sliding.size() != std::size_t(tables.positions) * 256 ||
        tables.global.size() != std::size_t(tables.positions) * 512)
        throw std::runtime_error("invalid device rotary tables");
    for (const auto t : {tables.sliding, tables.global})
        for (float a : t)
            if (!std::isfinite(a))
                throw std::runtime_error("nonfinite immutable rotary table");
    for (unsigned p = 0; p < tables.positions; ++p)
        for (unsigned i = 64; i < 256; ++i)
            if (bit_cast<unsigned>(tables.global[std::size_t(p) * 512 + 2 * i]) != 0x3f800000U ||
                bit_cast<unsigned>(tables.global[std::size_t(p) * 512 + 2 * i + 1]) != 0)
                throw std::runtime_error("noncanonical inactive rotary coordinates");
    auto& b = *impl_;
    b.options = options;
    b.options.shared_weights &= enabled("STRIX_G4_SHARED_WEIGHTS");
    b.options.fusion &= enabled("STRIX_G4_FUSION");
    b.options.graphs &= enabled("STRIX_G4_GRAPHS") && !options.profile;
    // Named identity: the option, else STRIX_G4_IDENTITY, else g4-v1. Unknown names are refused.
    const auto* identity = options.identity ? options.identity : std::getenv("STRIX_G4_IDENTITY");
    if (identity && std::string_view(identity) != arithmetic_id && std::string_view(identity) != a8::arithmetic_id)
        throw std::runtime_error(std::string("unknown gemma4 numerical identity: ") + identity);
    b.stats.a8 = identity && std::string_view(identity) == a8::arithmetic_id;
    b.stats.identity = b.stats.a8 ? a8::arithmetic_id : arithmetic_id;
    hip_checked(hipStreamCreateWithFlags(&b.queue, hipStreamNonBlocking));
    b.stats.shared_weights = b.options.shared_weights;
    b.stats.fusion = b.options.fusion;
    b.stats.norm_group = enabled("STRIX_G4_NORM_GROUP");
    // Default on: the PLE epilogue also writes the next input/final norm.
    b.stats.layer_norm_fusion = b.options.fusion && b.stats.norm_group && enabled("STRIX_G4_LAYER_NORM_FUSION");
    b.stats.chunk_decode = enabled("STRIX_G4_CHUNK_DECODE");
    b.stats.norm_search = enabled("STRIX_G4_NORM_SEARCH");
    b.stats.kv_pool = enabled("STRIX_G4_KV_POOL");
    b.stats.output_tile = b.stats.chunk_decode && output_tiling();
    b.stats.input_cache = b.stats.chunk_decode && !output_tiling() && input_cache(host, b.options.fusion);
    b.stats.attention_prefetch = enabled("STRIX_G4_ATTN_PREFETCH");
    b.stats.fused_attention = enabled("STRIX_G4_FUSED_ATTN");
    b.stats.store_fusion = store_fusion_enabled();
    b.stats.multi_linear = enabled("STRIX_G4_MULTI_LINEAR");
    b.stats.kv_publish = enabled("STRIX_G4_KV_PUBLISH");
    b.stats.linear_finite = enabled("STRIX_G4_LINEAR_FINITE");
    b.stats.norm_prefetch = enabled("STRIX_G4_NORM_PREFETCH");
    b.stats.norm_finite = enabled("STRIX_G4_NORM_FINITE");
    b.stats.pinned_io = enabled("STRIX_G4_PINNED_IO");
    b.stats.norm_bounds = b.stats.norm_search && enabled("STRIX_G4_NORM_BOUNDS");
    const auto* norm_intervals = std::getenv("STRIX_G4_NORM_INTERVALS");
    b.stats.norm_intervals = b.stats.norm_bounds && norm_intervals && std::string_view(norm_intervals) == "1";
    b.stats.norm_stage = enabled("STRIX_G4_NORM_STAGE") && host.config.hidden <= 4096U &&
                         (b.stats.norm_group || b.options.fusion);
    b.stats.head_registers = enabled("STRIX_G4_HEAD_REGISTERS") &&
                            std::any_of(host.config.layer.begin(), host.config.layer.end(), [](const auto& layer) {
                                return (layer.attention == AttentionKind::global && layer.head_dim == 512U) ||
                                       (layer.attention == AttentionKind::sliding && layer.head_dim == 256U);
                            });
    b.stats.chunk_prefetch = b.stats.chunk_decode && chunk_prefetch(host) && !output_tiling();
    b.stats.chunk_prefetch_bf16_only = b.stats.chunk_prefetch && !std::getenv("STRIX_G4_CHUNK_PREFETCH");
    b.stats.packed_header = b.stats.chunk_decode && packed_headers(host);
    b.stats.streaming_weights = b.stats.chunk_decode && streaming_matrices(host, 1U);
    b.stats.streaming_weights_auto = b.stats.chunk_decode && streaming_override() < 0;
    b.w.config = host.config;
    b.w.embedding = b.upload(host.embedding);
    b.w.head = b.upload(host.head);
    b.w.ple_table = b.upload(host.ple_table);
    b.w.ple_model = b.upload(host.ple_model);
    b.w.ple_model_norm = b.upload(host.ple_model_norm);
    b.w.final_norm = b.upload(host.final_norm);
    for (const auto& h : host.layer) {
        LayerWeights y;
        y.input_norm = b.upload(h.input_norm);
        y.post_attention_norm = b.upload(h.post_attention_norm);
        y.pre_ffn_norm = b.upload(h.pre_ffn_norm);
        y.post_ffn_norm = b.upload(h.post_ffn_norm);
        y.post_ple_norm = b.upload(h.post_ple_norm);
        y.q = b.upload(h.q);
        y.k = b.upload(h.k);
        y.v = b.upload(h.v);
        y.o = b.upload(h.o);
        y.q_norm = b.upload(h.q_norm);
        y.k_norm = b.upload(h.k_norm);
        y.gate = b.upload(h.gate);
        y.up = b.upload(h.up);
        y.down = b.upload(h.down);
        y.ple_gate = b.upload(h.ple_gate);
        y.ple_projection = b.upload(h.ple_projection);
        y.scalar = b.upload(h.scalar);
        b.w.layer.push_back(y);
    }
    b.sliding.resize(tables.sliding.size_bytes());
    b.sliding.upload(tables.sliding.data(), b.sliding.size());
    b.global.resize(tables.global.size_bytes());
    b.global.upload(tables.global.data(), b.global.size());
    b.table_positions = tables.positions;
    const auto& c = host.config;
    const std::size_t hbytes = 8ULL * c.hidden * sizeof(float), abytes = 8ULL * c.heads * 512 * sizeof(float),
                      kbytes = 8ULL * c.kv_heads * 512 * sizeof(float);
    b.residual.resize(hbytes);
    b.x.resize(hbytes);
    b.out.resize(hbytes);
    b.q.resize(abytes);
    b.att.resize(abytes);
    b.k.resize(kbytes);
    b.v.resize(kbytes);
    b.gate.resize(8ULL * std::max(c.inter, c.ple_dim) * sizeof(float));
    b.up.resize(b.gate.size());
    b.format_scratch.resize(std::max<std::size_t>(exl3::device_scratch_bytes,
        8ULL * std::max({c.hidden, c.inter, c.heads * 512U, c.ple_dim}) * sizeof(std::uint16_t)));
    b.ple.resize(8ULL * c.layers * c.ple_dim * sizeof(float));
    b.logits.resize(8ULL * c.vocab * sizeof(float));
    b.taps.resize(5 * hbytes);
    if (b.stats.a8) {
        const std::size_t widest = std::max({c.hidden, c.inter, c.heads * 512U});
        for (int slot = 0; slot < 2; ++slot) {
            b.a8_codes[slot].resize(8ULL * widest);
            b.a8_meta[slot].resize(8ULL * (widest / 32U) * sizeof(a8::Meta));
        }
    }
    b.tokens.resize(8 * sizeof(unsigned));
    b.positions.resize(8 * sizeof(std::uint64_t));
    b.reads.resize(8ULL * c.layers * sizeof(KvRead));
    b.writes.resize(8ULL * c.layers * sizeof(KvWrite));
    b.failure.resize(sizeof(unsigned));
    b.attention_counters.resize(8ULL * c.heads * sizeof(unsigned));
    b.attention_counters.zero();
    b.tickets.resize(sizeof(unsigned));
    b.tickets.zero();
    b.candidates.resize(8ULL * ((c.vocab + 255U) / 256U) * sizeof(Candidate));
    b.winners.resize(8 * sizeof(Candidate));
    if (b.stats.pinned_io) {
        b.host_tokens.allocate(b.tokens.size());
        b.host_positions.allocate(b.positions.size());
        b.host_reads.allocate(b.reads.size());
        b.host_writes.allocate(b.writes.size());
        b.host_failure.allocate(b.failure.size());
        b.host_logits.allocate(b.logits.size());
        b.host_taps.allocate(b.taps.size());
        b.host_winners.allocate(b.winners.size());
        b.host_hidden.allocate(b.x.size());
    }
    if (options.profile || options.graph_timing) {
        hip_checked(hipEventCreate(&b.start));
        hip_checked(hipEventCreate(&b.stop));
    }
}
DeviceTarget::~DeviceTarget() = default;
LinearView DeviceTarget::resident_embedding() const { return impl_->w.embedding; }
LinearView DeviceTarget::resident_head() const { return impl_->w.head.data ? impl_->w.head : impl_->w.embedding; }
const HostWeights& DeviceTarget::borrow_device_weights() const { return impl_->w; }
const DeviceStats& DeviceTarget::stats() const {
    return impl_->stats;
}
namespace {
// STRIX_G4_DECODE_EXPORT (read per pass; unset or 1 = on, 0 = off): logits and
// taps reach the host only when a row of the pass exports them; every pass
// returns the device greedy winners and reports its device finite fence.
bool decode_export_trim() {
    const auto* p = std::getenv("STRIX_G4_DECODE_EXPORT");
    return !p || std::string_view(p) != "0";
}
} // namespace
const std::vector<Candidate>& DeviceTarget::selected() const {
    return impl_->selected;
}
ForwardResult DeviceTarget::forward_rows(std::span<const RowPlan> plans) {
    auto& b = *impl_;
    const auto& w = b.w;
    const auto& c = w.config;
    std::vector<const RowPlan*> rows;
    std::map<DeviceStream::Impl::Txn*, std::vector<std::uint64_t>> transactions;
    std::map<DeviceStream::Impl::Txn*, std::pair<std::uint64_t, std::uint64_t>> owners;
    std::set<std::size_t> outputs;
    ForwardResult result;
    result.vocab = c.vocab;
    result.tap_width = 5 * c.hidden;
    for (const auto& row : plans)
        if (row.active) {
            auto* tx = dynamic_cast<DeviceStream::Impl::Txn*>(row.transaction);
            if (!tx || tx->published || tx->failed || (tx->rows && (!tx->reserved || !tx->evaluated)) || tx->stream->poisoned ||
                !same_config(tx->stream->config, c))
                throw std::runtime_error("invalid device row transaction");
            if (row.token >= c.vocab || row.position >= tx->stream->admitted || row.position >= b.table_positions ||
                row.output_row >= plans.size() || !outputs.insert(row.output_row).second)
                throw std::runtime_error("invalid device RowPlan");
            const auto owner = std::make_pair(row.stream, row.generation);
            if (owners.contains(tx) && owners[tx] != owner)
                throw std::runtime_error("transaction stream/generation changed");
            owners[tx] = owner;
            transactions[tx].push_back(row.position);
            rows.push_back(&row);
            result.rows = std::max(result.rows, row.output_row + 1);
        }
    if (rows.size() > 8)
        throw std::runtime_error("g4-v1 device admits at most eight rows per pass");
    for (auto& [tx, p] : transactions) {
        std::sort(p.begin(), p.end());
        for (std::size_t i = 0; i < p.size(); ++i)
            if (p[i] != tx->start + tx->rows + i)
                throw std::runtime_error("device transaction needs contiguous positions");
        if (tx->reserved && p.size() > tx->reserved - tx->rows)
            throw std::runtime_error("device overlay reservation exceeded");
    }
    const auto n = unsigned(rows.size());
    b.stats.streaming_weights = n != 0 && b.stats.chunk_decode &&
                                streaming_matrices(b.w, b.options.shared_weights ? n : 1U);
    b.stats.streaming_weights_bf16_only = b.stats.streaming_weights && b.stats.streaming_weights_auto &&
                                         (!b.options.shared_weights || n < 4U);
    b.stats.output_tile = n != 0 && n <= 4U && b.stats.chunk_decode && output_tiling();
    b.stats.chunk_prefetch = n == 1U && b.stats.chunk_decode && chunk_prefetch(b.w) && !output_tiling();
    b.stats.input_cache = (n == 1U || !b.options.shared_weights) && b.stats.chunk_decode && !output_tiling() &&
                          input_cache(b.w, b.options.fusion);
    b.stats.chunk_prefetch_bf16_only = b.stats.chunk_prefetch && !std::getenv("STRIX_G4_CHUNK_PREFETCH");
    b.stats.kernel_ms.clear();
    b.stats.gpu_pass_ms = 0.0;
    b.stats.pass_weight_bytes = 0;
    b.selected.clear();
    const bool trim = decode_export_trim();
    const bool want_logits = !trim || std::any_of(rows.begin(), rows.end(), [](const RowPlan* r) { return r->export_logits; });
    const bool want_taps = !trim || std::any_of(rows.begin(), rows.end(), [](const RowPlan* r) { return r->export_taps; });
    const bool want_hidden = std::any_of(rows.begin(), rows.end(), [](const RowPlan* r) { return r->export_hidden; });
    // With the trim on, exported hidden rows leave on the pass queue behind the head (no synchronous copy
    // after the pass, so the decode step never waits on another queue's work).
    const bool queued_hidden = want_hidden && trim && b.stats.pinned_io;
    if (!b.stats.pinned_io) {
        if (want_logits) result.logits.resize(result.rows * c.vocab);
        if (want_taps) result.taps.resize(result.rows * result.tap_width);
    }
    if (!n)
        return result;
    unsigned stride = 0;
    std::vector<unsigned> tokens;
    std::vector<std::uint64_t> positions;
    for (const auto* r : rows) {
        tokens.push_back(r->token);
        positions.push_back(r->position);
        stride = std::max(stride, unsigned(r->position + 1));
    }
    // Stable 64-position buckets alter storage/launch bounds only. Visible-key
    // iteration in every reduction still comes from the absolute row descriptor.
    stride = ((stride + attention_chunk - 1U) / attention_chunk) * attention_chunk;
    const auto* one_chunk = std::getenv("STRIX_G4_ONE_CHUNK_ATTN");
    b.stats.one_chunk_attention = b.options.fusion && stride <= attention_chunk && one_chunk &&
                                  std::string_view(one_chunk) == "1";
    const auto chunks = stride / attention_chunk;
    const auto score_bytes = std::size_t(n) * c.heads * stride * sizeof(float);
    const auto maximum_bytes = std::size_t(n) * c.heads * (chunks + 1U) * sizeof(float);
    const auto partial_bytes = std::size_t(n) * c.heads * chunks * 513 * sizeof(float);
    if (score_bytes > b.scores.size() || maximum_bytes > b.maxima.size() || partial_bytes > b.partials.size()) {
        b.clear_graphs();
        b.scores.resize(std::max(score_bytes, b.scores.size()));
        b.maxima.resize(std::max(maximum_bytes, b.maxima.size()));
        b.partials.resize(std::max(partial_bytes, b.partials.size()));
    }
    bool capturing = false;
    Impl::Graph pending;
    b.a8_source[0] = b.a8_source[1] = nullptr;
    try {
        for (auto& [tx, p] : transactions) {
            tx->rows += p.size();
            tx->evaluated = false;
            for (std::size_t l = 0; l < tx->overlay.size(); ++l) {
                const auto bytes = (tx->stream->pooled ? 8ULL : tx->rows) * tx->stream->cache[l].width * sizeof(float);
                if (!tx->reserved && (!tx->stream->pooled || bytes > tx->overlay[l].k.size())) {
                    tx->overlay[l].k.resize(bytes);
                    tx->overlay[l].v.resize(bytes);
                }
            }
        }
        b.input(b.tokens, b.host_tokens, tokens.data(), tokens.size() * sizeof(unsigned));
        b.input(b.positions, b.host_positions, positions.data(), positions.size() * sizeof(std::uint64_t));
        hip_checked(hipMemsetAsync(b.failure.data(), 0, b.failure.size(), b.queue));
        hip_checked(hipMemsetAsync(b.taps.data(), 0, b.taps.size(), b.queue));
        std::vector<KvRead> descriptors(8ULL * c.layers);
        std::vector<KvWrite> destinations(8ULL * c.layers);
        for (unsigned l = 0; l < c.layers; ++l)
            for (unsigned r = 0; r < n; ++r) {
                auto* tx = static_cast<DeviceStream::Impl::Txn*>(rows[r]->transaction);
                descriptors[8ULL * l + r] = tx->read(c.layer[l].producer, rows[r]->position);
                if (c.layer[l].producer == l)
                    destinations[8ULL * l + r] = {tx->overlay[l].k.as<float>(), tx->overlay[l].v.as<float>(),
                                                   unsigned(rows[r]->position - tx->start)};
            }
        b.input(b.reads, b.host_reads, descriptors.data(), descriptors.size() * sizeof(KvRead));
        b.input(b.writes, b.host_writes, destinations.data(), destinations.size() * sizeof(KvWrite));
        // K/V epilogue folded into attention only when every row is its
        // transaction's only new position; the schedule is part of the graph key.
        const bool single_rows = std::all_of(transactions.begin(), transactions.end(),
                                             [](const auto& t) { return t.second.size() == 1U; });
        const auto key = std::make_pair(n, stride * 2U + (single_rows ? 1U : 0U));
        const auto cached = b.graphs.find(key);
        b.stats.graphs = b.options.graphs;
        if (b.options.graph_timing)
            hip_checked(hipEventRecord(b.start, b.queue));
        if (b.options.graphs && cached != b.graphs.end()) {
            b.stats.pass_weight_bytes = cached->second.traffic;
            hip_checked(hipGraphLaunch(cached->second.executable, b.queue));
        } else {
            if (b.graphs.size() >= 8)
                b.clear_graphs();
            if (b.options.graphs) {
                hip_checked(hipStreamBeginCapture(b.queue, hipStreamCaptureModeThreadLocal));
                capturing = true;
            }
            b.run("embedding", [&] {
                embedding(w.embedding, b.tokens.as<unsigned>(), n, b.residual.as<float>(), embedding_scale, b.queue);
                b.finite(b.residual, std::size_t(n) * c.hidden);
            });
            b.dot("ple_model", w.ple_model, b.residual, b.ple, n);
            b.run("ple_combine", [&] {
                ple_combine(b.ple.as<float>(), w.ple_table, w.ple_model_norm, b.tokens.as<unsigned>(), n, c.layers,
                            c.ple_dim, b.queue, b.stats.norm_finite ? b.failure.as<unsigned>() : nullptr);
                if (!b.stats.norm_finite)
                    b.finite(b.ple, std::size_t(n) * c.layers * c.ple_dim);
            });
            b.stats.pass_weight_bytes += std::size_t(n) * (w.embedding.row_bytes + w.ple_table.row_bytes);
            for (unsigned l = 0; l < c.layers; ++l) {
                const auto& y = w.layer[l];
                const auto& a = c.layer[l];
                const auto dim = a.head_dim;
                const bool global = a.attention == AttentionKind::global;
                const auto* table = global ? b.global.as<float>() : b.sliding.as<float>();
                if (!b.stats.layer_norm_fusion || l == 0U)
                    b.normalize("input_norm", b.residual, y.input_norm, b.x, n, c.hidden, true);
                // Producer Q/K/V projections of one input share a launch when their formats match.
                bool k_done = false, v_done = false;
                LinearEpilogue projections;
                // EXL3 producer projections share one group of transform-fused launches.
                if (a.producer == l && b.dot_group("qkv", {{y.q, &b.q}, {y.k, &b.k}, {y.v, &b.v}}, b.x, n)) {
                    k_done = v_done = true;
                } else if (b.stats.multi_linear && a.producer == l && y.k.format == y.q.format) {
                    projections.extra[0] = y.k;
                    projections.extra_out[0] = b.k.as<float>();
                    if (y.v.format == y.q.format) {
                        projections.extra[1] = y.v;
                        projections.extra_out[1] = b.v.as<float>();
                    }
                }
                if (projections.extra_out[0] && b.fused_dot("qkv", y.q, b.x, b.q, n, projections)) {
                    k_done = true;
                    v_done = projections.extra_out[1] != nullptr;
                    b.stats.pass_weight_bytes += std::size_t(y.k.outputs) * y.k.row_bytes;
                    if (v_done)
                        b.stats.pass_weight_bytes += std::size_t(y.v.outputs) * y.v.row_bytes;
                } else {
                    b.dot("q", y.q, b.x, b.q, n);
                }
                if (a.producer == l) {
                    if (!k_done)
                        b.dot("k", y.k, b.x, b.k, n);
                    if (!v_done)
                        b.dot("v", y.v, b.x, b.v, n);
                }
                bool attended = false;
                const bool attention_a8 = b.stats.a8 && a8::quantized(y.o.format) && (c.heads * dim) % 256U == 0;
                if (a.producer == l && single_rows)
                    b.run("attention_kv_store", [&] {
                        attended = fused_attention_store(
                            b.q.as<float>(), y.q_norm, b.k.as<float>(), b.v.as<float>(), y.k_norm,
                            b.writes.as<KvWrite>() + 8ULL * l, c.kv_heads, b.positions.as<std::uint64_t>(), table,
                            b.reads.as<KvRead>() + 8ULL * l, n, c.heads, dim, stride, global, b.att.as<float>(),
                            b.failure.as<unsigned>(), b.queue, b.stats.norm_search, b.stats.norm_bounds,
                            b.stats.norm_intervals, attention_a8 ? b.a8_output(c.heads * dim) : A8Rows{});
                    });
                if (a.producer == l && !attended) {
                    bool stored = false;
                    b.run("kv_store", [&] {
                        stored = kv_store(b.k.as<float>(), b.v.as<float>(), y.k_norm, b.positions.as<std::uint64_t>(),
                                          table, b.writes.as<KvWrite>() + 8ULL * l, n, c.kv_heads, dim, global,
                                          b.failure.as<unsigned>(), b.queue, b.stats.norm_search,
                                          b.stats.norm_bounds, b.stats.norm_intervals);
                    });
                    if (!stored) {
                        b.run("kv_norm_rope", [&] {
                            head_norm_rope(b.k.as<float>(), y.k_norm, n, c.kv_heads, dim,
                                           b.positions.as<std::uint64_t>(), table, global, b.queue,
                                           b.stats.norm_finite ? b.failure.as<unsigned>() : nullptr);
                            norm(b.v.as<float>(), {}, b.v.as<float>(), n * c.kv_heads, dim, b.queue,
                                 b.stats.norm_finite ? b.failure.as<unsigned>() : nullptr);
                            if (!b.stats.norm_finite) {
                                b.finite(b.k, std::size_t(n) * c.kv_heads * dim);
                                b.finite(b.v, std::size_t(n) * c.kv_heads * dim);
                            }
                        });
                        b.run("overlay", [&] {
                            store_overlay(b.k.as<float>(), b.v.as<float>(), b.writes.as<KvWrite>() + 8ULL * l, n,
                                          c.kv_heads * dim, b.queue);
                        });
                    }
                }
                if (!attended)
                    b.run("fused_attention", [&] {
                        attended = fused_attention(b.q.as<float>(), y.q_norm, b.positions.as<std::uint64_t>(), table,
                                                   b.reads.as<KvRead>() + 8ULL * l, n, c.heads, dim, stride, global,
                                                   b.att.as<float>(), b.failure.as<unsigned>(), b.queue,
                                                   b.stats.norm_search, b.stats.norm_bounds, b.stats.norm_intervals,
                                                   b.scores.as<float>(), b.maxima.as<float>(),
                                                   b.partials.as<float>(), b.attention_counters.as<unsigned>(),
                                                   attention_a8 ? b.a8_output(c.heads * dim) : A8Rows{});
                    });
                if (attended && attention_a8)
                    b.a8_produced(0, b.att);
                if (!attended) {
                    b.run("q_norm_rope", [&] {
                        head_norm_rope(b.q.as<float>(), y.q_norm, n, c.heads, dim, b.positions.as<std::uint64_t>(),
                                       table, global, b.queue,
                                       b.stats.norm_finite ? b.failure.as<unsigned>() : nullptr);
                        if (!b.stats.norm_finite)
                            b.finite(b.q, std::size_t(n) * c.heads * dim);
                    });
                    b.run("attention", [&] {
                        attention(b.q.as<float>(), b.reads.as<KvRead>() + 8ULL * l, n, c.heads, dim, stride,
                                  b.scores.as<float>(), b.maxima.as<float>(), b.partials.as<float>(),
                                  b.att.as<float>(), b.failure.as<unsigned>(), b.queue, b.options.fusion);
                    });
                }
                b.dot("o", y.o, b.att, b.out, n);
                if (b.options.fusion && b.stats.norm_group) {
                    b.epilogue("attention_epilogue", y.post_attention_norm, n, c.hidden, {}, &b.x, y.pre_ffn_norm, true);
                } else {
                    b.normalize("post_attention_norm", b.out, y.post_attention_norm, b.x, n, c.hidden);
                    b.run("attention_residual", [&] {
                        residual(b.residual.as<float>(), b.x.as<float>(), n * c.hidden, {}, b.queue);
                        b.finite(b.residual, std::size_t(n) * c.hidden);
                    });
                    b.normalize("pre_ffn_norm", b.residual, y.pre_ffn_norm, b.x, n, c.hidden, true);
                }
                if (b.options.fusion && y.gate.format != WeightFormat::exl3_mul1 && y.up.format != WeightFormat::exl3_mul1) {
                    b.stats.pass_weight_bytes +=
                        std::size_t(y.gate.outputs) * y.gate.row_bytes * (b.options.shared_weights ? 1 : n);
                    b.stats.pass_weight_bytes +=
                        std::size_t(y.up.outputs) * y.up.row_bytes * (b.options.shared_weights ? 1 : n);
                    bool quantized = false;
                    if (b.stats.a8 && a8::quantized(y.gate.format) && y.gate.format == y.up.format) {
                        // The GELU product is quantized for the down projection in the same launch.
                        const int slot = b.a8_input(b.x, n, c.hidden);
                        const bool product = c.inter % 32U == 0;
                        b.run("gate_up", [&] {
                            quantized = a8_gate_up(y.gate, y.up, b.a8_rows(c.hidden, slot), n, b.gate.as<float>(),
                                                   b.queue, b.failure.as<unsigned>(),
                                                   product ? b.a8_rows(c.inter, 1 - slot) : A8Rows{});
                        }, true);
                        for (auto& source : b.a8_source)
                            if (source == b.gate.data())
                                source = nullptr;
                        if (quantized && product)
                            b.a8_source[1 - slot] = b.gate.data();
                    }
                    if (!quantized)
                    b.run("gate_up", [&] {
                        gate_up(y.gate, y.up, b.x.as<float>(), n, b.gate.as<float>(), b.options.shared_weights, b.queue,
                                b.stats.linear_finite ? b.failure.as<unsigned>() : nullptr);
                        if (!b.stats.linear_finite)
                            b.finite(b.gate, std::size_t(n) * c.inter);
                    });
                } else {
                    if (!b.dot_group("gate_up", {{y.gate, &b.gate}, {y.up, &b.up}}, b.x, n)) {
                        b.dot("gate", y.gate, b.x, b.gate, n);
                        b.dot("up", y.up, b.x, b.up, n);
                    }
                    b.run("gelu_product", [&] {
                        gelu_product(b.gate.as<float>(), b.up.as<float>(), n * c.inter, b.queue);
                        b.finite(b.gate, std::size_t(n) * c.inter);
                    });
                }
                b.dot("down", y.down, b.gate, b.out, n);
                // FFN epilogue and PLE gate/activation in one launch when admitted (g4-v1 gate only).
                bool gated = false;
                if (b.options.fusion && b.stats.norm_group && b.stats.store_fusion && b.options.shared_weights &&
                    y.ple_gate.format != WeightFormat::exl3_mul1 && !(b.stats.a8 && a8::quantized(y.ple_gate.format)))
                    b.run("ffn_epilogue_ple_gate", [&] {
                        gated = ffn_ple_gate(b.out.as<float>(), y.post_ffn_norm, b.residual.as<float>(), y.ple_gate,
                                             b.ple.as<float>() + std::size_t(l) * c.ple_dim, c.layers * c.ple_dim, n,
                                             c.hidden, b.gate.as<float>(), b.failure.as<unsigned>(),
                                             b.tickets.as<unsigned>(), b.queue);
                    });
                if (gated)
                    b.stats.pass_weight_bytes += std::size_t(y.ple_gate.outputs) * y.ple_gate.row_bytes;
                else if (b.options.fusion && b.stats.norm_group) {
                    b.epilogue("ffn_epilogue", y.post_ffn_norm, n, c.hidden, {}, nullptr, {}, true);
                } else {
                    b.normalize("post_ffn_norm", b.out, y.post_ffn_norm, b.x, n, c.hidden);
                    b.run("ffn_residual", [&] {
                        residual(b.residual.as<float>(), b.x.as<float>(), n * c.hidden, {}, b.queue);
                        b.finite(b.residual, std::size_t(n) * c.hidden);
                    });
                }
                LinearEpilogue activation;
                activation.product = b.ple.as<float>() + std::size_t(l) * c.ple_dim;
                activation.product_stride = c.layers * c.ple_dim;
                if (!gated && (!b.stats.store_fusion || !b.fused_dot("ple_gate_activation", y.ple_gate, b.residual,
                                                                     b.gate, n, activation, true))) {
                    b.dot("ple_gate", y.ple_gate, b.residual, b.gate, n);
                    b.run("ple_activation", [&] {
                        ple_gate(b.gate.as<float>(), b.ple.as<float>(), n, c.layers, l, c.ple_dim, b.queue,
                                 b.stats.norm_finite ? b.failure.as<unsigned>() : nullptr);
                        if (!b.stats.norm_finite)
                            b.finite(b.gate, std::size_t(n) * c.ple_dim);
                    });
                }
                b.dot("ple_projection", y.ple_projection, b.gate, b.out, n);
                if (b.options.fusion && b.stats.norm_group) {
                    const auto next_weight = l + 1U < c.layers ? w.layer[l + 1U].input_norm : w.final_norm;
                    b.epilogue("ple_epilogue", y.post_ple_norm, n, c.hidden, y.scalar,
                               b.stats.layer_norm_fusion ? &b.x : nullptr, next_weight, b.stats.layer_norm_fusion);
                } else {
                    b.normalize("post_ple_norm", b.out, y.post_ple_norm, b.x, n, c.hidden);
                    b.run("ple_residual_scalar", [&] {
                        residual(b.residual.as<float>(), b.x.as<float>(), n * c.hidden, y.scalar, b.queue);
                        b.finite(b.residual, std::size_t(n) * c.hidden);
                    });
                }
                for (unsigned t = 0; t < 5; ++t)
                    if (tap_layers[t] == l) {
                        hip_checked(hipMemcpy2DAsync(b.taps.as<float>() + t * c.hidden, 5ULL * c.hidden * sizeof(float),
                                                b.residual.data(), std::size_t(c.hidden) * sizeof(float),
                                                std::size_t(c.hidden) * sizeof(float), n, hipMemcpyDeviceToDevice, b.queue));
                    }
            }
            if (!b.stats.layer_norm_fusion)
                b.normalize("final_norm", b.residual, w.final_norm, b.x, n, c.hidden, true);
            const auto head = w.head.data ? w.head : w.embedding;
            LinearEpilogue capped;
            capped.cap = true;
            if (b.stats.store_fusion && b.fused_dot("head_softcap", head, b.x, b.logits, n, capped)) {
                b.run("argmax", [&] {
                    argmax(b.logits.as<float>(), n, c.vocab, b.candidates.as<Candidate>(), b.winners.as<Candidate>(),
                           b.queue);
                });
            } else {
                b.dot("head", head, b.x, b.logits, n);
                b.run("softcap_argmax", [&] {
                    cap_logits(b.logits.as<float>(), n * c.vocab, b.queue);
                    argmax(b.logits.as<float>(), n, c.vocab, b.candidates.as<Candidate>(), b.winners.as<Candidate>(),
                           b.queue);
                    b.finite(b.logits, std::size_t(n) * c.vocab);
                });
            }
            if (capturing) {
                const auto status = hipStreamEndCapture(b.queue, &pending.graph);
                capturing = false;
                hip_checked(status);
                hip_checked(hipGraphInstantiate(&pending.executable, pending.graph, nullptr, nullptr, 0));
                pending.traffic = b.stats.pass_weight_bytes;
                b.graphs.emplace(key, pending);
                pending = {};
                hip_checked(hipGraphLaunch(b.graphs.at(key).executable, b.queue));
            }
        }
        if (b.options.graph_timing)
            hip_checked(hipEventRecord(b.stop, b.queue));
        if (b.stats.pinned_io) {
            hip_checked(hipMemcpyAsync(b.host_failure.data(), b.failure.data(), sizeof(unsigned), hipMemcpyDeviceToHost,
                                       b.queue));
            if (want_logits)
                hip_checked(hipMemcpyAsync(b.host_logits.data(), b.logits.data(), std::size_t(n) * c.vocab * sizeof(float),
                                           hipMemcpyDeviceToHost, b.queue));
            if (want_taps)
                hip_checked(hipMemcpyAsync(b.host_taps.data(), b.taps.data(), std::size_t(n) * result.tap_width * sizeof(float),
                                           hipMemcpyDeviceToHost, b.queue));
            hip_checked(hipMemcpyAsync(b.host_winners.data(), b.winners.data(), n * sizeof(Candidate),
                                       hipMemcpyDeviceToHost, b.queue));
            if (queued_hidden)
                hip_checked(hipMemcpyAsync(b.host_hidden.data(), b.x.data(), std::size_t(n) * c.hidden * sizeof(float),
                                           hipMemcpyDeviceToHost, b.queue));
            // Allocate/clear the returned vectors while the queued device work runs.
            if (want_logits) result.logits.resize(result.rows * c.vocab);
            if (want_taps) result.taps.resize(result.rows * result.tap_width);
        }
        synchronize(b.queue);
        if (b.options.graph_timing) {
            float ms = 0.0F;
            hip_checked(hipEventElapsedTime(&ms, b.start, b.stop));
            b.stats.gpu_pass_ms = ms;
        }
        unsigned failure = 0;
        if (b.stats.pinned_io)
            failure = *b.host_failure.as<unsigned>();
        else
            b.failure.download(&failure, sizeof(failure));
        if (failure)
            throw std::runtime_error("nonfinite execution or missing device KV tag");
        if (want_hidden) {
            std::vector<float> normalized;
            const float* hidden_data = b.host_hidden.as<float>();
            if (!queued_hidden) {
                normalized.resize(std::size_t(n) * c.hidden);
                b.x.download(normalized.data(),normalized.size() * sizeof(float));
                hidden_data = normalized.data();
            }
            result.hidden.resize(result.rows * c.hidden);
            for (unsigned r = 0; r < n; ++r) if (rows[r]->export_hidden)
                std::copy_n(hidden_data + std::size_t(r) * c.hidden,c.hidden,
                            result.hidden.data() + rows[r]->output_row * c.hidden);
        }
        std::vector<float> logits, taps;
        std::vector<Candidate> selection;
        const float* logit_data = b.host_logits.as<float>();
        const float* tap_data = b.host_taps.as<float>();
        const Candidate* selected_data = b.host_winners.as<Candidate>();
        if (!b.stats.pinned_io) {
            if (want_logits) logits.resize(std::size_t(n) * c.vocab);
            if (want_taps) taps.resize(std::size_t(n) * result.tap_width);
            selection.resize(n);
            if (want_logits) b.logits.download(logits.data(), logits.size() * sizeof(float));
            if (want_taps) b.taps.download(taps.data(), taps.size() * sizeof(float));
            b.winners.download(selection.data(), n * sizeof(Candidate));
            logit_data = logits.data();
            tap_data = taps.data();
            selected_data = selection.data();
        }
        b.selected.resize(result.rows);
        if (trim) { result.selected.assign(result.rows, 0U); result.device_checked = true; }
        for (unsigned r = 0; r < n; ++r) {
            const auto o = rows[r]->output_row;
            if (want_logits) std::copy_n(logit_data + std::size_t(r) * c.vocab, c.vocab, result.logits.data() + o * c.vocab);
            if (want_taps)
                std::copy_n(tap_data + std::size_t(r) * result.tap_width, result.tap_width,
                            result.taps.data() + o * result.tap_width);
            b.selected[o] = selected_data[r];
            if (trim) result.selected[o] = selected_data[r].id;
        }
        for (auto& [tx, p] : transactions) {
            (void)p;
            tx->evaluated = true;
        }
    } catch (...) {
        if (capturing)
            (void)hipStreamEndCapture(b.queue, &pending.graph);
        if (pending.executable)
            (void)hipGraphExecDestroy(pending.executable);
        if (pending.graph)
            (void)hipGraphDestroy(pending.graph);
        (void)hipStreamSynchronize(b.queue);
        for (auto& [tx, p] : transactions) {
            (void)p;
            tx->failed = true;
        }
        throw;
    }
    return result;
}
} // namespace strix::gemma4::gpu
