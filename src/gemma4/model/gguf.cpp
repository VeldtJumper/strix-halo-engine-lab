#include "model.hpp"
#include "strix/gguf_reader.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace strix::gemma4 {
namespace {
double numeric(const strix::gguf::MetadataEntry* e) {
    if (e) {
        if (e->unsigned_value) return double(*e->unsigned_value);
        if (e->signed_value) return double(*e->signed_value);
        if (e->floating_value) return *e->floating_value;
    }
    throw std::runtime_error("missing/nonnumeric gemma4 GGUF metadata");
}
std::string canonical_name(const std::string& name) {
    const std::string p = "model.language_model.";
    const std::map<std::string, std::string> top = {{"token_embd.weight","embed_tokens.weight"},
        {"output_norm.weight","norm.weight"},{"per_layer_model_proj.weight","per_layer_model_projection.weight"},
        {"per_layer_proj_norm.weight","per_layer_projection_norm.weight"},{"per_layer_token_embd.weight","embed_tokens_per_layer.weight"}};
    if (const auto it = top.find(name); it != top.end()) return p + it->second;
    if (!name.starts_with("blk.")) throw std::runtime_error("unknown gemma4 GGUF tensor: " + name);
    const auto dot = name.find('.', 4);
    const auto layer = name.substr(4, dot - 4);
    if (layer.empty() || layer.find_first_not_of("0123456789") != std::string::npos || dot == std::string::npos) throw std::runtime_error("invalid GGUF layer name");
    const std::map<std::string, std::string> tails = {{"attn_q","self_attn.q_proj"},{"attn_k","self_attn.k_proj"},
        {"attn_v","self_attn.v_proj"},{"attn_output","self_attn.o_proj"},{"attn_q_norm","self_attn.q_norm"},
        {"attn_k_norm","self_attn.k_norm"},{"attn_norm","input_layernorm"},{"post_attention_norm","post_attention_layernorm"},
        {"ffn_norm","pre_feedforward_layernorm"},{"post_ffw_norm","post_feedforward_layernorm"},
        {"post_norm","post_per_layer_input_norm"},{"ffn_gate","mlp.gate_proj"},{"ffn_up","mlp.up_proj"},
        {"ffn_down","mlp.down_proj"},{"inp_gate","per_layer_input_gate"},{"proj","per_layer_projection"}};
    const auto tail = name.substr(dot + 1);
    if (tail == "layer_output_scale.weight") return p + "layers." + layer + ".layer_scalar";
    if (!tail.ends_with(".weight")) throw std::runtime_error("unsupported GGUF tensor suffix");
    const auto it = tails.find(tail.substr(0, tail.size() - 7));
    if (it == tails.end()) throw std::runtime_error("unknown GGUF layer tensor: " + name);
    return p + "layers." + layer + "." + it->second + ".weight";
}
}
std::string gguf_tensor_name(const std::string& name) { return canonical_name(name); }
ModelManifest load_gguf_model(const std::filesystem::path& path) {
    strix::gguf::Limits limits;
    limits.max_retained_array_elements = 64;
    const auto info = strix::gguf::read_file(path, limits);
    const auto* arch = info.find_metadata("general.architecture");
    if (!arch || arch->string_value != "gemma4") throw std::runtime_error("GGUF architecture is not gemma4");
    const std::map<std::string, double> fields = {{"block_count",42},{"context_length",65536},{"embedding_length",2560},
        {"feed_forward_length",10240},{"attention.head_count",8},{"attention.head_count_kv",2},{"rope.freq_base",1000000},
        {"rope.freq_base_swa",10000},{"attention.layer_norm_rms_epsilon",double(norm_eps)},{"attention.key_length",512},
        {"attention.value_length",512},{"final_logit_softcapping",30},{"attention.sliding_window",512},
        {"attention.shared_kv_layers",18},{"embedding_length_per_layer_input",256},{"attention.key_length_swa",256},
        {"attention.value_length_swa",256},{"rope.dimension_count",512},{"rope.dimension_count_swa",256}};
    for (const auto& [key, value] : fields)
        if (numeric(info.find_metadata("gemma4." + key)) != value) throw std::runtime_error("unsupported GGUF metadata: " + key);
    for (const auto& e : info.metadata)
        if (e.key.starts_with("gemma4.") && e.key != "gemma4.attention.sliding_window_pattern" && !fields.contains(e.key.substr(7)))
            throw std::runtime_error("unsupported gemma4 GGUF override: " + e.key);
    const auto* pattern = info.find_metadata("gemma4.attention.sliding_window_pattern");
    if (!pattern || pattern->unsigned_array.size() != layers) throw std::runtime_error("missing GGUF layer schedule");
    ModelManifest m;
    for (std::uint32_t l = 0; l < layers; ++l) {
        const bool g = l % 6 == 5;
        if (pattern->unsigned_array[l] != (g ? 0U : 1U)) throw std::runtime_error("unsupported GGUF attention schedule");
        m.config.layer.push_back({g ? AttentionKind::global : AttentionKind::sliding, g ? 512U : 256U, l < 24 ? l : (g ? 23U : 22U)});
    }
    const auto storage = std::make_shared<MappedFile>(path);
    m.storage.push_back(storage);
    m.model_hash = hash_bytes(storage->data(), storage->size());
    if (info.find_metadata("general.file_type")) m.gguf_file_type = std::int32_t(numeric(info.find_metadata("general.file_type")));
    std::size_t frozen = 0, rope_markers = 0;
    for (const auto& t : info.tensors) {
        WeightFormat format;
        switch (t.ggml_type) {
        case 0: format = WeightFormat::f32; break;
        case 8: format = WeightFormat::q8_0; break;
        case 12: format = WeightFormat::q4_k; break;
        case 14: format = WeightFormat::q6_k; break;
        case 30: format = WeightFormat::bf16; break;
        default: throw std::runtime_error("unsupported gemma4 per-tensor GGUF format");
        }
        auto shape = t.dimensions;
        std::reverse(shape.begin(), shape.end()); // GGUF ne[0] is the contiguous input width.
        const auto view = model_detail::make_view(storage->data() + t.absolute_offset, format, shape);
        if (t.name == "rope_freqs.weight") {
            // The admitted proportional marker denotes 64 active pairs, then zero frequencies.
            // Its large divisors are metadata, not approximate rotations of the inactive coordinates.
            if (format != WeightFormat::f32 || shape != std::vector<std::uint64_t>{256}) throw std::runtime_error("invalid GGUF proportional rotary marker");
            for (std::uint32_t i = 0; i < 256; ++i)
                if (weight_at(view, 0, i) != (i < 64 ? 1.0F : 1e30F)) throw std::runtime_error("unsupported GGUF rotary override");
            ++rope_markers; m.ignored_tensors.push_back(t.name + " (validated proportional rotary metadata)"); continue;
        }
        const auto name = canonical_name(t.name);
        if (frozen_compatibility_tensor(name)) {
            const auto start = std::string("model.language_model.layers.").size();
            const auto l = std::stoul(name.substr(start, name.find('.', start) - start));
            const auto d = m.config.layer[l].head_dim;
            const auto expected = name.ends_with("k_norm.weight") ? std::vector<std::uint64_t>{d} : std::vector<std::uint64_t>{std::uint64_t(m.config.kv_heads)*d,m.config.hidden};
            if (shape != expected) throw std::runtime_error("invalid GGUF compatibility shape");
            validate_finite(view);
            ++frozen; m.ignored_tensors.push_back(t.name + " (frozen shared-KV compatibility)"); continue;
        }
        validate_finite(view);
        m.tensors.push_back({name, shape, view, m.model_hash});
    }
    if (frozen != 54 || rope_markers != 1) throw std::runtime_error("incomplete GGUF compatibility/rotary set");
    model_detail::validate_manifest(m);
    return m;
}
} // namespace strix::gemma4
