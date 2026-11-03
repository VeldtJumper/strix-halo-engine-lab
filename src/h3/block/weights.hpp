#pragma once
// MiniMax H3 DiT weights (curve-form int8 convrot checkpoints, e.g. the fl2va / ref2va pruned files): host loader.
//
// Value policy (pinned by the reference capture): every module parameter is held at the compute dtype
// (bf16) by the reference loader, also the modules whose compute is fp32 (adaLN linears, patch projections, output
// heads): their fp32 / fp16 file values are rounded once to bf16 and used as fp32. Buffers keep the file precision
// (adaln_t_table fp32, rope.inv_freq fp32). The int8 GEMM weights keep int8 codes + fp32 per-channel scales.
// LoadPolicy::round_params_bf16 = false keeps the file precision instead (not the reference).
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "h3/block/safetensors.hpp"
#include "h3/gemm/ref.hpp"

namespace strix::h3::block {

inline constexpr std::uint32_t kHidden = 5376;
inline constexpr std::uint32_t kHeads = 56;
inline constexpr std::uint32_t kHeadDim = 128;
inline constexpr std::uint32_t kInner = kHeads * kHeadDim;  // 7168
inline constexpr std::uint32_t kFfn = 14336;
inline constexpr std::uint32_t kTextDim = 5120;
inline constexpr std::uint32_t kVideoPatch = 96;  // 24 x 1 x 2 x 2
inline constexpr std::uint32_t kAudioDim = 32;
inline constexpr std::uint32_t kRopeFreqs = 16;
inline constexpr std::uint32_t kRotDim = 96;  // 3 axes x 16 freqs x 2
inline constexpr std::uint32_t kNumBlocks = 50;
inline constexpr float kNormEps = 1e-5F;

struct LoadPolicy {
    bool round_params_bf16 = true;
};

// int8 tensorwise + convrot (group 256) linear: w[n][k] int8 codes, ws[n] fp32 per-channel scales, no bias.
struct QLinear {
    std::uint32_t n = 0, k = 0;
    std::vector<std::int8_t> w;
    std::vector<float> ws;
};

// bf16 linear (refiner, condition_proj): w[n][k] bf16 bits, optional bias bf16 bits.
struct Bf16Linear {
    std::uint32_t n = 0, k = 0;
    std::vector<std::uint16_t> w, b;
};

// fp32-compute linear (adaLN, patch projections, heads): w[n][k], b[n] as fp32 (values per LoadPolicy).
struct F32Linear {
    std::uint32_t n = 0, k = 0;
    std::vector<float> w, b;
};

struct RefinerBlockWeights {
    std::vector<std::uint16_t> norm1, norm2, q_norm, k_norm;  // bf16
    Bf16Linear qkv, out, fc1, fc2;
};

struct BlockWeights {
    std::uint32_t index = 0;
    std::vector<std::uint16_t> norm1, norm2, q_norm, k_norm;  // bf16 [5376] / [128]
    F32Linear adaln;  // [6 * 5376 * 3][basis]
    QLinear qkv, out, fc1, fc2;
};

struct GlobalWeights {
    std::uint32_t basis = 0;       // curve basis width (time_embed_dim of the curve form)
    std::uint32_t curve_grid = 0;  // adaln_t_table rows (1025)
    std::uint32_t pdd_steps = 1;   // stacked output heads
    std::vector<float> adaln_t_table;  // [curve_grid][basis] fp32 (buffer)
    std::vector<float> inv_freq;       // [16] fp32 (buffer)
    F32Linear video_patch_proj, audio_patch_proj;  // [5376][96], [5376][32]
    Bf16Linear condition_proj;                     // [5376][5120] + bias
    std::vector<RefinerBlockWeights> refiner;      // 2 blocks
    std::vector<std::uint16_t> refiner_final_norm;
    std::vector<std::uint16_t> final_norm;  // bf16 [5376]
    F32Linear final_adaln;                  // [2 * 5376][basis]
    F32Linear video_out, audio_out;         // [96 * pdd][5376], [32 * pdd][5376]
};

namespace detail {
// Norm-vector shape checks (shared by the loaders; host unit tests call them on malformed weights).
template <class B>
inline void check_norms(const B& b, const std::string& what) {
    if (b.norm1.size() != kHidden || b.norm2.size() != kHidden || b.q_norm.size() != kHeadDim || b.k_norm.size() != kHeadDim)
        throw std::runtime_error("weights: norm shapes of " + what);
}
inline void check_final_norms(const GlobalWeights& g) {
    if (g.refiner_final_norm.size() != kHidden || g.final_norm.size() != kHidden) throw std::runtime_error("weights: final norm shapes");
}

inline std::vector<float> read_as_f32(const SafeTensors& st, const std::string& name, bool round_bf16) {
    const StTensor& t = st.at(name);
    std::vector<float> o(static_cast<std::size_t>(t.numel()));
    if (t.dtype == StDtype::f32) {
        st.read_raw(t, o.data());
    } else if (t.dtype == StDtype::f16) {
        std::vector<std::uint16_t> h(o.size());
        st.read_raw(t, h.data());
        for (std::size_t i = 0; i < o.size(); ++i) o[i] = fixture::f16_to_f32(h[i]);
    } else if (t.dtype == StDtype::bf16) {
        std::vector<std::uint16_t> h(o.size());
        st.read_raw(t, h.data());
        for (std::size_t i = 0; i < o.size(); ++i) o[i] = gemm::bf16_to_f32(h[i]);
    } else {
        throw std::runtime_error("weights: " + name + " is not a float tensor");
    }
    if (round_bf16)
        for (auto& v : o) v = gemm::bf16_to_f32(gemm::f32_to_bf16(v));
    return o;
}
inline std::vector<std::uint16_t> read_bf16(const SafeTensors& st, const std::string& name) {
    const StTensor& t = st.at(name);
    if (t.dtype == StDtype::bf16) return st.read_vec<std::uint16_t>(name, StDtype::bf16);
    std::vector<float> f = read_as_f32(st, name, false);
    std::vector<std::uint16_t> o(f.size());
    for (std::size_t i = 0; i < f.size(); ++i) o[i] = gemm::f32_to_bf16(f[i]);
    return o;
}
inline void expect_shape(const SafeTensors& st, const std::string& name, std::vector<std::int64_t> shape) {
    if (st.at(name).shape != shape) throw std::runtime_error("weights: unexpected shape of " + name);
}
inline F32Linear read_f32_linear(const SafeTensors& st, const std::string& prefix, bool round_bf16, bool bias = true) {
    F32Linear l;
    const StTensor& w = st.at(prefix + ".weight");
    if (w.shape.size() != 2) throw std::runtime_error("weights: " + prefix + ".weight is not 2-D");
    l.n = static_cast<std::uint32_t>(w.shape[0]);
    l.k = static_cast<std::uint32_t>(w.shape[1]);
    l.w = read_as_f32(st, prefix + ".weight", round_bf16);
    if (bias) {
        l.b = read_as_f32(st, prefix + ".bias", round_bf16);
        if (l.b.size() != l.n) throw std::runtime_error("weights: bias size of " + prefix);
    } else {
        l.b.assign(l.n, 0.0F);
    }
    return l;
}
inline Bf16Linear read_bf16_linear(const SafeTensors& st, const std::string& prefix, bool bias) {
    Bf16Linear l;
    const StTensor& w = st.at(prefix + ".weight");
    l.n = static_cast<std::uint32_t>(w.shape.at(0));
    l.k = static_cast<std::uint32_t>(w.shape.at(1));
    l.w = read_bf16(st, prefix + ".weight");
    if (bias) l.b = read_bf16(st, prefix + ".bias");
    return l;
}
// The per-layer quant config (a JSON document stored as u8 bytes) must be int8_tensorwise + convrot group 256.
inline void check_quant_config(const SafeTensors& st, const std::string& prefix) {
    // The config is the single uint8 tensor directly under the layer besides its data tensors (found by elimination).
    const StTensor* cq = nullptr;
    const std::string dot = prefix + ".";
    for (auto it = st.tensors().lower_bound(dot); it != st.tensors().end() && it->first.compare(0, dot.size(), dot) == 0; ++it) {
        const std::string suffix = it->first.substr(dot.size());
        if (suffix.find('.') != std::string::npos || suffix == "weight" || suffix == "weight_scale" || suffix == "bias") continue;
        if (it->second.dtype_name != "U8") continue;
        if (cq) throw std::runtime_error("weights: " + prefix + " has more than one quant config tensor");
        cq = &it->second;
    }
    if (!cq) throw std::runtime_error("weights: " + prefix + " has no quant config");
    std::string s(static_cast<std::size_t>(cq->nbytes()), '\0');
    st.read_raw(*cq, s.data());
    fixture::Json j = fixture::JsonParser(s).parse();
    const fixture::Json* fmt = j.find("format");
    const fixture::Json* rot = j.find("convrot");
    const fixture::Json* gs = j.find("convrot_groupsize");
    const bool ok = fmt && fmt->is_string() && fmt->str() == "int8_tensorwise" && rot &&
                    std::holds_alternative<bool>(rot->v) && std::get<bool>(rot->v) && gs && gs->is_number() &&
                    gs->num() == 256.0;
    if (!ok) throw std::runtime_error("weights: unsupported quant config for " + prefix + ": " + s);
}
inline QLinear read_qlinear(const SafeTensors& st, const std::string& prefix, std::uint32_t n, std::uint32_t k) {
    check_quant_config(st, prefix);
    expect_shape(st, prefix + ".weight", {n, k});
    expect_shape(st, prefix + ".weight_scale", {n, 1});
    QLinear q;
    q.n = n;
    q.k = k;
    q.w = st.read_vec<std::int8_t>(prefix + ".weight", StDtype::i8);
    q.ws = st.read_vec<float>(prefix + ".weight_scale", StDtype::f32);
    return q;
}
}  // namespace detail

inline GlobalWeights load_global(const SafeTensors& st, const LoadPolicy& pol = {}) {
    using namespace detail;
    GlobalWeights g;
    const StTensor* tab = st.find("adaln_t_table");
    if (!tab) throw std::runtime_error("weights: not a curve-form checkpoint (no adaln_t_table)");
    if (tab->shape.size() != 2 || tab->dtype != StDtype::f32) throw std::runtime_error("weights: bad adaln_t_table");
    g.curve_grid = static_cast<std::uint32_t>(tab->shape[0]);
    g.basis = static_cast<std::uint32_t>(tab->shape[1]);
    g.adaln_t_table = st.read_vec<float>("adaln_t_table", StDtype::f32);
    expect_shape(st, "rope.inv_freq", {kRopeFreqs});
    g.inv_freq = st.read_vec<float>("rope.inv_freq", StDtype::f32);
    const bool r = pol.round_params_bf16;
    g.video_patch_proj = read_f32_linear(st, "video_patch_proj", r);
    g.audio_patch_proj = read_f32_linear(st, "audio_patch_proj", r);
    if (g.video_patch_proj.n != kHidden || g.video_patch_proj.k != kVideoPatch || g.audio_patch_proj.n != kHidden ||
        g.audio_patch_proj.k != kAudioDim)
        throw std::runtime_error("weights: bad patch projection shapes");
    g.condition_proj = read_bf16_linear(st, "condition_proj", true);
    if (g.condition_proj.n != kHidden || g.condition_proj.k != kTextDim) throw std::runtime_error("weights: condition_proj");
    for (unsigned i = 0; st.find("token_refiner.blocks." + std::to_string(i) + ".norm1.weight"); ++i) {
        const std::string p = "token_refiner.blocks." + std::to_string(i);
        RefinerBlockWeights b;
        b.norm1 = read_bf16(st, p + ".norm1.weight");
        b.norm2 = read_bf16(st, p + ".norm2.weight");
        b.q_norm = read_bf16(st, p + ".attn.q_norm.weight");
        b.k_norm = read_bf16(st, p + ".attn.k_norm.weight");
        b.qkv = read_bf16_linear(st, p + ".attn.qkv_proj", false);
        b.out = read_bf16_linear(st, p + ".attn.out_proj", false);
        b.fc1 = read_bf16_linear(st, p + ".mlp.fc1", false);
        b.fc2 = read_bf16_linear(st, p + ".mlp.fc2", false);
        check_norms(b, p);
        if (b.qkv.n != 3 * kInner || b.qkv.k != kHidden || b.out.n != kHidden || b.out.k != kInner ||
            b.fc1.n != 2 * kFfn || b.fc1.k != kHidden || b.fc2.n != kHidden || b.fc2.k != kFfn)
            throw std::runtime_error("weights: refiner block shapes");
        g.refiner.push_back(std::move(b));
    }
    g.refiner_final_norm = read_bf16(st, "token_refiner.final_norm.weight");
    g.final_norm = read_bf16(st, "final_layer.norm.weight");
    check_final_norms(g);
    if (g.refiner.empty()) throw std::runtime_error("weights: no token refiner blocks");
    g.final_adaln = read_f32_linear(st, "final_layer.adaln_proj.linear", r);
    if (g.final_adaln.n != 2 * kHidden || g.final_adaln.k != g.basis) throw std::runtime_error("weights: final adaln");
    g.video_out = read_f32_linear(st, "final_layer.video_out", r);
    g.audio_out = read_f32_linear(st, "final_layer.audio_out", r);
    if (g.video_out.k != kHidden || g.video_out.n % kVideoPatch || g.audio_out.k != kHidden || g.audio_out.n % kAudioDim)
        throw std::runtime_error("weights: head shapes");
    g.pdd_steps = g.video_out.n / kVideoPatch;
    if (g.audio_out.n / kAudioDim != g.pdd_steps) throw std::runtime_error("weights: video/audio head banks differ");
    return g;
}

inline std::uint32_t count_blocks(const SafeTensors& st) {
    std::uint32_t n = 0;
    while (st.find("blocks." + std::to_string(n) + ".norm1.weight")) ++n;
    return n;
}

inline BlockWeights load_block(const SafeTensors& st, std::uint32_t i, std::uint32_t basis, const LoadPolicy& pol = {}) {
    using namespace detail;
    const std::string p = "blocks." + std::to_string(i);
    BlockWeights b;
    b.index = i;
    b.norm1 = read_bf16(st, p + ".norm1.weight");
    b.norm2 = read_bf16(st, p + ".norm2.weight");
    b.q_norm = read_bf16(st, p + ".attn.q_norm.weight");
    b.k_norm = read_bf16(st, p + ".attn.k_norm.weight");
    check_norms(b, p);
    b.adaln = read_f32_linear(st, p + ".adaln_proj.linear", pol.round_params_bf16);
    if (b.adaln.n != 6U * kHidden * 3U || b.adaln.k != basis) throw std::runtime_error("weights: adaln shape of " + p);
    if (st.find(p + ".attn.to_gate_compress.weight")) {
        // VSA gate: unused by the dense forward (ignored)
    }
    b.qkv = read_qlinear(st, p + ".attn.qkv_proj", 3 * kInner, kHidden);
    b.out = read_qlinear(st, p + ".attn.out_proj", kHidden, kInner);
    b.fc1 = read_qlinear(st, p + ".mlp.fc1", 2 * kFfn, kHidden);
    b.fc2 = read_qlinear(st, p + ".mlp.fc2", kHidden, kFfn);
    return b;
}

}  // namespace strix::h3::block
