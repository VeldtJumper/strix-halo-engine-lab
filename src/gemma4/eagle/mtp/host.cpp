// MTP assistant drafter: safetensors loader, synthetic weights, host g4-v1 reference.
#include "mtp.hpp"
#include "gemma4/kernels/ops.hpp"
#include "gemma4/model/model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>

namespace strix::gemma4::eagle::mtp {
namespace {
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error("mtp: " + message);
}
bool finite_bits(float x) { return (bit_cast<std::uint32_t>(x) & 0x7f800000U) != 0x7f800000U; }
void finite(std::span<const float> v, const char* what) {
    for (const float x : v) require(finite_bits(x), std::string("nonfinite ") + what);
}
void linear(LinearView w, const float* x, float* out) {
    std::vector<float> row(w.inputs);
    for (std::uint32_t o = 0; o < w.outputs; ++o) {
        for (std::uint32_t i = 0; i < w.inputs; ++i) row[i] = gpu::coefficient(w, o, i);
        out[o] = chunk16_dot([&](std::uint32_t i) { return row[i]; }, x, w.inputs);
    }
    finite({out, w.outputs}, "linear output");
}
void norm(const float* x, const float* w, float* out, std::uint32_t width) {
    require(finite_bits(lane_dot(x, x, width)), "nonfinite norm reduction");
    rms_norm(x, w, out, width, norm_eps);
    finite({out, width}, "norm output");
}
const float* kv_row(const KvLayer& kv, std::uint64_t p, bool value) {
    if (p >= kv.frontier) {
        require(kv.overlay_k && p - kv.frontier < kv.candidate_rows, "position beyond committed KV and transaction overlay");
        return (value ? kv.overlay_v : kv.overlay_k) + (p - kv.frontier) * kv.width;
    }
    const auto slot = kv.global ? p : p % kv.slots;
    require(slot < kv.slots, "KV position outside the view");
    require(!kv.tags || kv.tags[slot] == p, "target KV tag missing (rolled back or overwritten)");
    return (value ? kv.v : kv.k) + slot * kv.width;
}
std::uint32_t mtp_format_check(WeightFormat f) {
    require(f == WeightFormat::bf16 || f == WeightFormat::q8_0 || f == WeightFormat::f32, "matrix format");
    return 0;
}
} // namespace

void Config::validate() const {
    require(hidden > 0 && hidden % 32 == 0 && backbone == gemma4::hidden && heads == 2 * kv_heads && kv_heads == 2, "assistant shape");
    require(sliding_dim == 256 && full_dim == 512, "head dims must match the target sliding/global layers");
    require(vocab == gemma4::vocab && centroids > 0 && vocab % centroids == 0 && top_k > 0 && top_k <= 64 && top_k <= centroids,
            "clustered head (top_k 1..64)");
    require(sliding_keys >= 1 && sliding_keys <= 513 && ffn % 32 == 0, "sliding extent/ffn");
    require(embedding_scale == gemma4::embedding_scale, "embedding scale");
}
std::uint32_t top_k_from_environment(std::uint32_t fallback) {
    const char* v = std::getenv("STRIX_G4_MTP_TOPK");
    if (!v || !*v) return fallback;
    const auto k = std::strtoul(v, nullptr, 10);
    require(k >= 1 && k <= 64, "STRIX_G4_MTP_TOPK must be 1..64");
    return std::uint32_t(k);
}
void Tables::validate() const {
    require(positions > 0 && sliding.size() == std::size_t(positions) * 256 && global.size() == std::size_t(positions) * 512,
            "RoPE tables must use the RopeTables layout");
}

void Weights::validate() const {
    const auto& c = config;
    c.validate();
    auto mat = [&](const Matrix& m, std::uint32_t outputs, std::uint32_t inputs, const char* name) {
        require(m.outputs == outputs && m.inputs == inputs && m.bytes.size() == std::size_t(outputs) * m.row_bytes(),
                std::string("matrix shape ") + name);
        (void)mtp_format_check(m.format);
    };
    auto vec = [&](const std::vector<float>& v, std::uint32_t width, const char* name) {
        require(v.size() == width, std::string("norm shape ") + name);
        finite(v, name);
    };
    mat(pre_projection, c.hidden, 2 * c.backbone, "pre_projection");
    mat(post_projection, c.backbone, c.hidden, "post_projection");
    mat(centroids, c.centroids, c.hidden, "centroids");
    mat(table, c.vocab, c.hidden, "embed_tokens");
    for (std::uint32_t l = 0; l < layers; ++l) {
        const auto& y = layer[l];
        const auto d = c.head_dim(l);
        mat(y.q, c.heads * d, c.hidden, "q_proj"); mat(y.o, c.hidden, c.heads * d, "o_proj");
        mat(y.gate, c.ffn, c.hidden, "gate"); mat(y.up, c.ffn, c.hidden, "up"); mat(y.down, c.hidden, c.ffn, "down");
        vec(y.input_norm, c.hidden, "input_norm"); vec(y.q_norm, d, "q_norm"); vec(y.post_attn_norm, c.hidden, "post_attn");
        vec(y.pre_ffn_norm, c.hidden, "pre_ffn"); vec(y.post_ffn_norm, c.hidden, "post_ffn");
        require(finite_bits(y.scalar), "layer_scalar");
    }
    vec(final_norm, c.hidden, "final_norm");
    require(ordering.size() == c.vocab, "token_ordering size");
    std::vector<bool> seen(c.vocab, false);
    for (const auto id : ordering) {
        require(id < c.vocab && !seen[id], "token_ordering must be a permutation");
        seen[id] = true;
    }
}

std::filesystem::path resolve_drafter(const std::filesystem::path& path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const auto has = [&](const fs::path& d) { return fs::is_regular_file(d / "model.safetensors", ec); };
    if (fs::is_directory(path, ec)) {
        if (has(path)) return path;
        if (has(path / drafter_subdirectory)) return path / drafter_subdirectory;
        return {};
    }
    if (fs::is_regular_file(path, ec) && has(path.parent_path() / drafter_subdirectory)) return path.parent_path() / drafter_subdirectory;
    return {};
}

// Measured thresholds (gate n2/n3, see the comment below); ConfidenceConfig::builtin_tau covers unknown weights.
constexpr float TAU_UNTUNED = 0.6F, TAU_FT1 = 0.4F;

ConfidenceKeys confidence_keys(const strix::serve::Json& cfg) {
    ConfidenceKeys k;
    if (const auto* t = cfg.find("confidence_tau")) {
        require(t->is_number() && t->as_double() > 0 && t->as_double() <= 1, "config.json confidence_tau must be in (0,1]");
        k.tau = float(t->as_double());
    }
    for (auto [key, out] : {std::pair{"confidence_max_k", &k.max_k}, std::pair{"confidence_first", &k.first}})
        if (const auto* v = cfg.find(key)) {
            const double x = v->is_number() ? v->as_double() : 0;
            require(x >= 1 && x <= 7 && x == std::floor(x), std::string("config.json ") + key + " must be an integer 1..7");
            *out = std::uint32_t(x);
        }
    return k;
}
ConfidenceKeys confidence_keys(std::string_view config_json) { return confidence_keys(strix::serve::parse_json(config_json)); }

float builtin_confidence_tau(std::string_view sha) {
    // Measured C1 gate (EN/ES/JA/code, 256 tokens, ids == OFF): the untuned Google assistant is over-confident (top-1
    // prob 0.80 vs 0.65 accepted), the Nemotron fine-tune ft1 is calibrated, hence different thresholds.
    struct Known { std::string_view prefix; float tau; };
    static constexpr Known known[] = {
        {"12875062fc25c51e8fa9b62abd2de7ad48b7d63f8559d5d604fbd5a3d6bcff16", TAU_UNTUNED}, // google gemma-4-E4B-it-assistant
        {"19594909271ef877", TAU_FT1},                                                      // e4bdraft ft1 (release drafter)
    };
    for (const auto& k : known) if (sha.substr(0, k.prefix.size()) == k.prefix) return k.tau;
    return 0.0F;
}

Weights Weights::load(const std::filesystem::path& path, WeightFormat matrices) {
    (void)mtp_format_check(matrices);
    const auto directory = resolve_drafter(path);
    require(!directory.empty(), "no MTP drafter at " + path.string() + " (model.safetensors or " + drafter_subdirectory + "/)");
    ConfidenceKeys confidence;
    if (std::filesystem::exists(directory / "config.json")) {
        const auto cfg = read_json(directory / "config.json");
        const auto* type = cfg.find("model_type");
        const auto* backbone = cfg.find("backbone_hidden_size");
        require(type && type->is_string() && type->as_string() == "gemma4_assistant", "config.json model_type must be gemma4_assistant");
        require(!backbone || backbone->as_int() == gemma4::hidden, "config.json backbone_hidden_size must be 2560");
        confidence = confidence_keys(cfg);
    }
    const auto file = read_safetensors(directory / "model.safetensors");
    std::map<std::string, const SafeTensor*> by;
    for (const auto& t : file.tensors) by[t.name] = &t;
    std::size_t used = 0;
    auto get = [&](const std::string& name, const char* dtype, std::vector<std::uint64_t> shape) -> const SafeTensor& {
        const auto it = by.find(name);
        require(it != by.end(), "missing tensor " + name);
        require(it->second->dtype == dtype && it->second->shape == shape, "dtype/shape mismatch " + name);
        ++used;
        return *it->second;
    };
    auto bf16_floats = [&](const SafeTensor& t) {
        std::vector<float> out(t.bytes / 2);
        const auto* p = file.storage->data() + t.offset;
        for (std::size_t i = 0; i < out.size(); ++i) out[i] = bf16_value(load16(p + 2 * i));
        return out;
    };
    auto mat = [&](const std::string& name, std::uint32_t outputs, std::uint32_t inputs) {
        const auto& t = get(name, "BF16", {outputs, inputs});
        if (matrices == WeightFormat::bf16) {
            const auto* p = file.storage->data() + t.offset;
            return Matrix{std::vector<std::uint8_t>(p, p + t.bytes), outputs, inputs, WeightFormat::bf16};
        }
        return pack(bf16_floats(t), outputs, inputs, matrices);
    };
    auto vec = [&](const std::string& name, std::uint32_t width) { return bf16_floats(get(name, "BF16", {width})); };
    Weights w;
    auto& c = w.config;
    {
        const auto it = by.find("model.layers.0.self_attn.q_proj.weight");
        require(it != by.end() && it->second->shape.size() == 2, "missing model.layers.0 q_proj");
        c.hidden = std::uint32_t(it->second->shape[1]);
    }
    {
        const auto it = by.find("masked_embedding.centroids.weight");
        require(it != by.end() && it->second->shape.size() == 2, "missing centroids");
        c.centroids = std::uint32_t(it->second->shape[0]);
    }
    {
        const auto it = by.find("model.layers.0.mlp.gate_proj.weight");
        require(it != by.end() && it->second->shape.size() == 2, "missing gate_proj");
        c.ffn = std::uint32_t(it->second->shape[0]);
    }
    c.top_k = top_k_from_environment(c.top_k); // runtime parameter (default 64), not the checkpoint's 32
    for (std::uint32_t l = 0; l < layers; ++l) {
        const auto it = by.find("model.layers." + std::to_string(l) + ".self_attn.q_norm.weight");
        require(it != by.end(), "missing q_norm");
        c.full[l] = it->second->shape.at(0) == c.full_dim;
    }
    c.validate();
    w.pre_projection = mat("pre_projection.weight", c.hidden, 2 * c.backbone);
    w.post_projection = mat("post_projection.weight", c.backbone, c.hidden);
    w.centroids = mat("masked_embedding.centroids.weight", c.centroids, c.hidden);
    w.table = mat("model.embed_tokens.weight", c.vocab, c.hidden);
    for (std::uint32_t l = 0; l < layers; ++l) {
        const auto s = "model.layers." + std::to_string(l) + ".";
        const auto d = c.head_dim(l);
        auto& y = w.layer[l];
        y.q = mat(s + "self_attn.q_proj.weight", c.heads * d, c.hidden);
        y.o = mat(s + "self_attn.o_proj.weight", c.hidden, c.heads * d);
        y.gate = mat(s + "mlp.gate_proj.weight", c.ffn, c.hidden);
        y.up = mat(s + "mlp.up_proj.weight", c.ffn, c.hidden);
        y.down = mat(s + "mlp.down_proj.weight", c.hidden, c.ffn);
        y.input_norm = vec(s + "input_layernorm.weight", c.hidden);
        y.q_norm = vec(s + "self_attn.q_norm.weight", d);
        y.post_attn_norm = vec(s + "post_attention_layernorm.weight", c.hidden);
        y.pre_ffn_norm = vec(s + "pre_feedforward_layernorm.weight", c.hidden);
        y.post_ffn_norm = vec(s + "post_feedforward_layernorm.weight", c.hidden);
        y.scalar = vec(s + "layer_scalar", 1)[0];
    }
    w.final_norm = vec("model.norm.weight", c.hidden);
    {
        const auto& t = get("masked_embedding.token_ordering", "I64", {c.vocab});
        w.ordering.resize(c.vocab);
        const auto* p = file.storage->data() + t.offset;
        for (std::size_t i = 0; i < c.vocab; ++i) {
            std::int64_t x = 0;
            std::memcpy(&x, p + 8 * i, 8);
            require(x >= 0 && x < std::int64_t(c.vocab), "token_ordering id");
            w.ordering[i] = std::uint32_t(x);
        }
    }
    require(used == file.tensors.size(), "unexpected tensor in assistant checkpoint (lm_head must stay tied)");
    w.source = hash_bytes(file.storage->data(), file.storage->size());
    w.config.confidence_tau = confidence.tau;
    w.config.confidence_max_k = confidence.max_k;
    w.config.confidence_first = confidence.first;
    w.config.confidence_builtin = builtin_confidence_tau(w.source);
    w.directory = directory;
    w.has_license = std::filesystem::exists(directory / "LICENSE");
    w.has_notice = std::filesystem::exists(directory / "NOTICE");
    w.has_lineage = std::filesystem::exists(directory / "LINEAGE.md");
    w.validate();
    return w;
}

Weights Weights::random(const Config& c, std::uint64_t seed, WeightFormat matrices) {
    c.validate();
    Weights w;
    w.config = c;
    std::uint64_t state = seed;
    auto next = [&] {
        std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31U);
    };
    auto uniform = [&] { return float(double(next() >> 11U) * 0x1p-53 * 2.0 - 1.0); };
    auto mat = [&](std::uint32_t outputs, std::uint32_t inputs, float scale) {
        std::vector<float> x(std::size_t(outputs) * inputs);
        for (auto& a : x) a = bf16_value(bf16_bits(uniform() * scale)); // checkpoint-like BF16 values
        return pack(x, outputs, inputs, matrices);
    };
    auto vec = [&](std::uint32_t width) {
        std::vector<float> x(width);
        for (auto& a : x) a = bf16_value(bf16_bits(1.0F + 0.25F * uniform()));
        return x;
    };
    auto s = [](std::uint32_t fan_in) { return 1.7320508F / std::sqrt(float(fan_in)); };
    w.pre_projection = mat(c.hidden, 2 * c.backbone, s(2 * c.backbone));
    w.post_projection = mat(c.backbone, c.hidden, s(c.hidden));
    w.centroids = mat(c.centroids, c.hidden, 4 * s(c.hidden));
    w.table = mat(c.vocab, c.hidden, 4 * s(c.hidden));
    for (std::uint32_t l = 0; l < layers; ++l) {
        auto& y = w.layer[l];
        const auto d = c.head_dim(l);
        y.q = mat(c.heads * d, c.hidden, s(c.hidden)); y.o = mat(c.hidden, c.heads * d, s(c.heads * d));
        y.gate = mat(c.ffn, c.hidden, s(c.hidden)); y.up = mat(c.ffn, c.hidden, s(c.hidden)); y.down = mat(c.hidden, c.ffn, s(c.ffn));
        y.input_norm = vec(c.hidden); y.q_norm = vec(d); y.post_attn_norm = vec(c.hidden);
        y.pre_ffn_norm = vec(c.hidden); y.post_ffn_norm = vec(c.hidden);
        y.scalar = bf16_value(bf16_bits(0.5F + 0.25F * uniform()));
    }
    w.final_norm = vec(c.hidden);
    w.ordering.resize(c.vocab);
    std::iota(w.ordering.begin(), w.ordering.end(), 0U);
    for (std::uint32_t i = c.vocab - 1; i > 0; --i) std::swap(w.ordering[i], w.ordering[next() % (i + 1)]);
    w.source = "synthetic";
    w.validate();
    return w;
}

void attend_head(const float* q, std::uint32_t dim, const KvLayer& kv, std::uint32_t kv_head, std::uint64_t first,
                 std::uint64_t last, float* out) {
    require(last >= first, "empty attention extent");
    const auto n = last - first + 1, len = (n + attention_splits - 1) / attention_splits;
    std::array<float, attention_splits> m{}, l{};
    std::vector<float> acc(std::size_t(attention_splits) * dim, 0.0F);
    std::array<bool, attention_splits> used{};
    for (std::uint32_t s = 0; s < attention_splits; ++s) {
        const auto b = first + s * len, e = std::min(last + 1, first + (s + 1) * len);
        if (b >= e) continue;
        used[s] = true;
        std::vector<float> score(std::size_t(e - b));
        float maximum = -std::numeric_limits<float>::infinity();
        for (auto j = b; j < e; ++j) {
            const float x = lane_dot(q, kv_row(kv, j, false) + std::size_t(kv_head) * dim, dim);
            require(finite_bits(x), "nonfinite attention score");
            score[std::size_t(j - b)] = x;
            maximum = std::max(maximum, x);
        }
        float sum = 0.0F;
        for (auto& x : score) { x = declared_exp(x - maximum); sum = sum + x; }
        for (std::uint32_t d = 0; d < dim; ++d) {
            float a = 0.0F;
            for (auto j = b; j < e; ++j) a = a + score[std::size_t(j - b)] * kv_row(kv, j, true)[std::size_t(kv_head) * dim + d];
            acc[std::size_t(s) * dim + d] = a;
        }
        m[s] = maximum; l[s] = sum;
    }
    float M = -std::numeric_limits<float>::infinity();
    for (std::uint32_t s = 0; s < attention_splits; ++s) if (used[s]) M = std::max(M, m[s]);
    std::array<float, attention_splits> f{};
    float L = 0.0F;
    for (std::uint32_t s = 0; s < attention_splits; ++s) if (used[s]) { f[s] = declared_exp(m[s] - M); L = L + l[s] * f[s]; }
    for (std::uint32_t d = 0; d < dim; ++d) {
        float num = 0.0F;
        for (std::uint32_t s = 0; s < attention_splits; ++s) if (used[s]) num = num + acc[std::size_t(s) * dim + d] * f[s];
        out[d] = num / L;
    }
    finite({out, dim}, "attention output");
}

std::vector<std::uint32_t> top_clusters(std::span<const float> logits, std::uint32_t k) {
    std::vector<bool> taken(logits.size(), false);
    std::vector<std::uint32_t> out;
    for (std::uint32_t r = 0; r < k; ++r) {
        float best = -std::numeric_limits<float>::infinity();
        std::uint32_t id = UINT32_MAX;
        for (std::uint32_t i = 0; i < logits.size(); ++i)
            if (!taken[i] && better(logits[i], i, best, id)) { best = logits[i]; id = i; }
        taken[id] = true;
        out.push_back(id);
    }
    return out;
}

void Drafter::remember(const Inputs& in, std::uint32_t k) {
    if (in.hidden.data() != saved_hidden_.data()) saved_hidden_.assign(in.hidden.begin(), in.hidden.end());
    saved_ = in;
    saved_.hidden = saved_hidden_;
    saved_depth_ = k;
}
Draft Drafter::extend(std::uint32_t k) {
    require(saved_depth_ && k > saved_depth_ && k <= max_depth, "extend needs a previous shallower draft");
    const auto in = saved_;
    return draft(in, k); // draft() calls remember(): saved_depth_ becomes k
}

struct HostDrafter::Impl {
    const Weights& w;
    const Config& c;
    Tables tables;
    LinearView embedding;
    Timings times;
    Impl(const Weights& weights, Tables t, LinearView e) : w(weights), c(weights.config), tables(t), embedding(e) {
        w.validate(); tables.validate();
        require(embedding.data && embedding.inputs == c.backbone && embedding.outputs >= c.vocab, "target embedding geometry");
    }
};
HostDrafter::HostDrafter(const Weights& weights, Tables tables, LinearView target_embedding)
    : impl_(std::make_unique<Impl>(weights, tables, target_embedding)) {}
HostDrafter::~HostDrafter() = default;
const Config& HostDrafter::config() const { return impl_->c; }
const Timings& HostDrafter::timings() const { return impl_->times; }

Draft HostDrafter::draft(const Inputs& in, std::uint32_t k) {
    const auto start = std::chrono::steady_clock::now();
    const auto& d = *impl_;
    const auto& c = d.c;
    const auto& w = d.w;
    const auto p = in.position;
    require(k >= 1 && k <= max_depth && p >= 1 && p < d.tables.positions && in.hidden.size() == c.backbone &&
            in.anchor < c.vocab, "draft inputs");
    finite(in.hidden, "hidden row");
    const auto H = c.hidden, B = c.backbone;
    std::vector<float> cat(2ULL * B), x(H), a(H), b(H), y(H), g(c.ffn), u(c.ffn), q, att, o(H), cl(c.centroids);
    std::copy(in.hidden.begin(), in.hidden.end(), cat.begin() + B);
    std::uint32_t token = in.anchor;
    Draft out;
    out.depth = k;
    for (std::uint32_t depth = 0; depth < k; ++depth) {
        for (std::uint32_t i = 0; i < B; ++i) cat[i] = embedding_value(gpu::coefficient(d.embedding, token, i), c.embedding_scale);
        linear(w.pre_projection.view(), cat.data(), x.data());
        norm(x.data(), w.layer[0].input_norm.data(), a.data(), H);
        for (std::uint32_t l = 0; l < layers; ++l) {
            const auto& L = w.layer[l];
            const bool full = c.full[l];
            const auto dim = c.head_dim(l);
            q.assign(std::size_t(c.heads) * dim, 0.0F);
            att.assign(q.size(), 0.0F);
            linear(L.q.view(), a.data(), q.data());
            const float* rope = full ? d.tables.global.data() + p * 512 : d.tables.sliding.data() + p * 256;
            const auto active = full ? 64U : 128U;
            const auto& kv = full ? in.full : in.sliding;
            const auto first = full ? 0 : (p > c.sliding_keys ? p - c.sliding_keys : 0);
            for (std::uint32_t h = 0; h < c.heads; ++h) {
                float* v = q.data() + std::size_t(h) * dim;
                norm(v, L.q_norm.data(), v, dim);
                for (std::uint32_t i = 0; i < active; ++i) {
                    const float e0 = v[i], e1 = v[i + dim / 2U], cs = rope[2 * i], sn = rope[2 * i + 1];
                    v[i] = e0 * cs - e1 * sn;
                    v[i + dim / 2U] = e1 * cs + e0 * sn;
                }
                attend_head(v, dim, kv, h / (c.heads / c.kv_heads), first, p - 1, att.data() + std::size_t(h) * dim);
            }
            linear(L.o.view(), att.data(), o.data());
            norm(o.data(), L.post_attn_norm.data(), b.data(), H);
            for (std::uint32_t i = 0; i < H; ++i) x[i] = x[i] + b[i];
            norm(x.data(), L.pre_ffn_norm.data(), y.data(), H);
            linear(L.gate.view(), y.data(), g.data());
            linear(L.up.view(), y.data(), u.data());
            for (std::uint32_t i = 0; i < c.ffn; ++i) g[i] = gelu_tanh(g[i]) * u[i];
            linear(L.down.view(), g.data(), o.data());
            norm(o.data(), L.post_ffn_norm.data(), b.data(), H);
            for (std::uint32_t i = 0; i < H; ++i) x[i] = (x[i] + b[i]) * L.scalar;
            finite(x, "residual");
            norm(x.data(), l + 1 < layers ? w.layer[l + 1].input_norm.data() : w.final_norm.data(), a.data(), H);
        }
        // a = final-normed state: next depth's backbone half and the clustered head.
        linear(w.post_projection.view(), a.data(), cat.data() + B);
        linear(w.centroids.view(), a.data(), cl.data());
        const auto top = top_clusters(cl, c.top_k);
        const auto per = c.per_cluster();
        std::vector<float> z(std::size_t(c.top_k) * per);
        std::vector<std::uint32_t> ids(z.size());
        std::vector<float> row(H);
        const auto table = w.table.view();
        for (std::uint32_t j = 0; j < c.top_k; ++j)
            for (std::uint32_t m = 0; m < per; ++m) {
                const auto id = w.ordering[std::size_t(top[j]) * per + m];
                for (std::uint32_t i = 0; i < H; ++i) row[i] = gpu::coefficient(table, id, i);
                ids[std::size_t(j) * per + m] = id;
                z[std::size_t(j) * per + m] = lane_dot(a.data(), row.data(), H);
            }
        finite(z, "candidate logits");
        // Selection recipe of the device kernel: 1024 strided partial argmaxes + tree, then strided exp sums + tree.
        constexpr std::uint32_t T = select_threads;
        std::vector<float> bv(T, -std::numeric_limits<float>::infinity()), sum(T, 0.0F);
        std::vector<std::uint32_t> bi(T, UINT32_MAX);
        const auto n = std::uint32_t(z.size());
        for (std::uint32_t t = 0; t < T; ++t)
            for (std::uint32_t i = t; i < n; i += T)
                if (better(z[i], ids[i], bv[t], bi[t])) { bv[t] = z[i]; bi[t] = ids[i]; }
        for (std::uint32_t s = T / 2; s; s /= 2)
            for (std::uint32_t t = 0; t < s; ++t)
                if (better(bv[t + s], bi[t + s], bv[t], bi[t])) { bv[t] = bv[t + s]; bi[t] = bi[t + s]; }
        token = bi[0];
        out.tokens[depth] = token;
        // Requested statistics only (Drafter::set_statistics); the device head skips them the same way.
        if (statistics() & statistic_margin) {
            float second = -std::numeric_limits<float>::infinity();
            for (std::uint32_t i = 0; i < n; ++i) if (ids[i] != bi[0]) second = std::max(second, z[i]);
            out.margin[depth] = bv[0] - second;
        }
        if (statistics() & statistic_probability) {
            for (std::uint32_t t = 0; t < T; ++t)
                for (std::uint32_t i = t; i < n; i += T) sum[t] = sum[t] + declared_exp(z[i] - bv[0]);
            for (std::uint32_t s = T / 2; s; s /= 2)
                for (std::uint32_t t = 0; t < s; ++t) sum[t] = sum[t] + sum[t + s];
            out.prob[depth] = 1.0F / sum[0];
        }
    }
    impl_->times.draft_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    remember(in, k);
    return out;
}

} // namespace strix::gemma4::eagle::mtp
