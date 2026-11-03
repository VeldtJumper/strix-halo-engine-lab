#pragma once
// LoRA descriptor loader (host, no HIP) for the step-distilled LoRAs (FAST-class identities, OFF by default):
//   LoraFile f(path);                                   // safetensors header + data (read on demand)
//   f.prefix()                                          // "" or "diffusion_model." (stripped from every name below)
//   f.has_branch(module) / f.branch(module, strength)   // module e.g. "blocks.3.attn.qkv_proj" (lora_module_name)
//       -> {A bf16 [r][k], B bf16 [n][r], rank, k, n, scale = strength * (alpha / rank if an .alpha tensor exists else 1)}
//          (F32 / F16 tensors are rounded RNE to bf16: the reference casts A and B to the bf16 compute dtype)
//   f.diff_names() / f.diff(name)                       // full-weight deltas "<module>.diff" (weight) / ".diff_b" (bias)
//       -> fp32 values + shape; apply with apply_diff_bf16 (W = bf16(W + strength * diff), as the reference's
//          "diff" patch on a bf16 weight: fp32 intermediate, one rounding to the weight dtype)
//   lora_identity(id)                                   // the identity table (file name, preset strength / steps / sampler)
// Supported tensor dtypes: BF16, F16, F32. Every key outside {lora_A, lora_B, alpha, diff, diff_b} throws (no silent
// partial loads).
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "h3/gemm/ref.hpp"

namespace strix::h3::levers::lora {

struct LoraIdentity {
    const char* id;
    const char* file;      // file name (in the models directory)
    float strength;        // preset strength
    std::uint32_t steps;   // preset steps
    const char* sampler;   // flow Euler on the "simple" schedule for all of them
    float shift;           // video shift (audio 3, as the base)
    bool fc2_bypass;       // true: run-time fc2 branch (step LoRAs); false: merged fc2 (Turbo v4)
    const char* notes;
};
inline const std::vector<LoraIdentity>& lora_identities() {
    static const std::vector<LoraIdentity> t = {
        {"h3-turbo-v4-bypass", "minimax_h3_turbo_v4_step600_ema_comfyui.safetensors", 0.70F, 8, "euler-simple", 12.0F, false,
         "r64 no alpha; adaLN E-grid injection; merged fc2 (merged_fc2_s070)"},
        {"h3-lora-fasth3-6step-v1", "fasth3_dense_comfy_6step.safetensors", 1.0F, 6, "euler-simple", 12.0F, true,
         "qkv r192 / others r64, alpha = rank; .diff deltas on 10 norm1, 8 norm2, final norm, condition_proj, patch projections, "
         "heads; adaLN dropped; sampler not documented locally (Euler simple shift 12 assumed)"},
        {"h3-lora-turbo-v12-v1", "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors", 1.0F, 4, "euler-simple", 12.0F, true,
         "qkv r384 (alpha 24) / others r128 (alpha 8): scale 0.0625; presets 4 steps (Fast) and 6 steps (Balanced)"},
    };
    return t;
}
inline const LoraIdentity& lora_identity(const std::string& id) {
    for (const auto& i : lora_identities())
        if (id == i.id) return i;
    throw std::invalid_argument("h3 lora: unknown identity " + id);
}

namespace detail {
// Minimal strict JSON reader for safetensors headers (objects, arrays, strings, numbers, true / false / null).
struct Json {
    enum Kind { kNull, kBool, kNum, kStr, kArr, kObj } kind = kNull;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;
    const Json* get(const std::string& k) const {
        for (const auto& kv : obj)
            if (kv.first == k) return &kv.second;
        return nullptr;
    }
};
class JsonParser {
public:
    explicit JsonParser(const std::string& s) : s_(s) {}
    Json parse() {
        Json j = value();
        ws();
        if (i_ != s_.size()) fail("trailing data");
        return j;
    }

private:
    [[noreturn]] void fail(const char* w) const { throw std::runtime_error(std::string("h3 lora: safetensors header JSON: ") + w + " at " + std::to_string(i_)); }
    void ws() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == '\t')) ++i_;
    }
    char peek() {
        ws();
        if (i_ >= s_.size()) fail("unexpected end");
        return s_[i_];
    }
    void expect(char c) {
        if (peek() != c) fail("unexpected character");
        ++i_;
    }
    std::string string() {
        expect('"');
        std::string o;
        while (true) {
            if (i_ >= s_.size()) fail("unterminated string");
            const char c = s_[i_++];
            if (c == '"') break;
            if (c == '\\') {
                if (i_ >= s_.size()) fail("bad escape");
                const char e = s_[i_++];
                switch (e) {
                    case '"': o += '"'; break;
                    case '\\': o += '\\'; break;
                    case '/': o += '/'; break;
                    case 'b': o += '\b'; break;
                    case 'f': o += '\f'; break;
                    case 'n': o += '\n'; break;
                    case 'r': o += '\r'; break;
                    case 't': o += '\t'; break;
                    case 'u':
                        if (i_ + 4 > s_.size()) fail("bad \\u escape");
                        i_ += 4;
                        o += '?';  // metadata text only; tensor names are ASCII
                        break;
                    default: fail("bad escape");
                }
            } else {
                o += c;
            }
        }
        return o;
    }
    Json value() {
        Json j;
        const char c = peek();
        if (c == '{') {
            j.kind = Json::kObj;
            ++i_;
            if (peek() == '}') { ++i_; return j; }
            while (true) {
                std::string k = string();
                expect(':');
                j.obj.emplace_back(std::move(k), value());
                if (peek() == ',') { ++i_; continue; }
                expect('}');
                return j;
            }
        }
        if (c == '[') {
            j.kind = Json::kArr;
            ++i_;
            if (peek() == ']') { ++i_; return j; }
            while (true) {
                j.arr.push_back(value());
                if (peek() == ',') { ++i_; continue; }
                expect(']');
                return j;
            }
        }
        if (c == '"') {
            j.kind = Json::kStr;
            j.str = string();
            return j;
        }
        if (s_.compare(i_, 4, "true") == 0) { i_ += 4; j.kind = Json::kBool; j.num = 1; return j; }
        if (s_.compare(i_, 5, "false") == 0) { i_ += 5; j.kind = Json::kBool; return j; }
        if (s_.compare(i_, 4, "null") == 0) { i_ += 4; return j; }
        std::size_t used = 0;
        j.num = std::stod(s_.substr(i_, 32), &used);
        if (!used) fail("bad number");
        i_ += used;
        j.kind = Json::kNum;
        return j;
    }
    const std::string& s_;
    std::size_t i_ = 0;
};
inline float f16_to_f32(std::uint16_t h) {
    const std::uint32_t s = (h >> 15U) & 1U, e = (h >> 10U) & 0x1fU, m = h & 0x3ffU;
    float v;
    if (e == 0) v = std::ldexp(static_cast<float>(m), -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp(static_cast<float>(m | 0x400U), static_cast<int>(e) - 25);
    return s ? -v : v;
}
}  // namespace detail

struct LoraBranch {
    std::vector<std::uint16_t> a, b;  // bf16 [rank][k], [n][rank]
    std::uint32_t rank = 0, k = 0, n = 0;
    float alpha = 0.0F;  // 0: no alpha tensor
    float scale = 0.0F;  // strength * (alpha / rank or 1)
};
struct LoraTensorInfo {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t begin = 0, end = 0;
};

class LoraFile {
public:
    explicit LoraFile(const std::string& path) : path_(path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("h3 lora: cannot open " + path);
        std::uint64_t n = 0;
        f.read(reinterpret_cast<char*>(&n), 8);
        if (!f || n == 0 || n > (1ULL << 30U)) throw std::runtime_error("h3 lora: bad safetensors header length in " + path);
        std::string h(n, '\0');
        f.read(h.data(), static_cast<std::streamsize>(n));
        if (!f) throw std::runtime_error("h3 lora: truncated header in " + path);
        data0_ = 8U + n;
        const detail::Json j = detail::JsonParser(h).parse();
        if (j.kind != detail::Json::kObj) throw std::runtime_error("h3 lora: header is not an object");
        bool any_prefixed = false, any_plain = false;
        for (const auto& kv : j.obj) {
            if (kv.first == "__metadata__") {
                for (const auto& m : kv.second.obj) meta_[m.first] = m.second.str;
                continue;
            }
            LoraTensorInfo t;
            const detail::Json* dt = kv.second.get("dtype");
            const detail::Json* sh = kv.second.get("shape");
            const detail::Json* of = kv.second.get("data_offsets");
            if (!dt || !sh || !of || of->arr.size() != 2) throw std::runtime_error("h3 lora: bad tensor entry " + kv.first);
            t.dtype = dt->str;
            for (const auto& d : sh->arr) t.shape.push_back(static_cast<std::uint64_t>(d.num));
            t.begin = static_cast<std::uint64_t>(of->arr[0].num);
            t.end = static_cast<std::uint64_t>(of->arr[1].num);
            (kv.first.rfind(kPrefix, 0) == 0 ? any_prefixed : any_plain) = true;
            tensors_[kv.first] = t;
        }
        if (any_prefixed && any_plain) throw std::runtime_error("h3 lora: mixed key prefixes in " + path);
        prefix_ = any_prefixed ? kPrefix : "";
        for (const auto& kv : tensors_) {
            const std::string k = kv.first.substr(prefix_.size());
            const auto ends = [&](const char* suf) {
                const std::size_t n2 = std::strlen(suf);
                return k.size() > n2 && k.compare(k.size() - n2, n2, suf) == 0;
            };
            if (ends(".lora_A.weight")) modules_[k.substr(0, k.size() - 14)] |= 1U;
            else if (ends(".lora_B.weight")) modules_[k.substr(0, k.size() - 14)] |= 2U;
            else if (ends(".alpha")) modules_[k.substr(0, k.size() - 6)] |= 4U;
            else if (ends(".diff") || ends(".diff_b")) diffs_.push_back(k);
            else throw std::runtime_error("h3 lora: unsupported tensor key " + kv.first);
        }
        for (const auto& m : modules_)
            if ((m.second & 3U) != 3U) throw std::runtime_error("h3 lora: incomplete LoRA pair for " + m.first);
    }
    const std::string& path() const { return path_; }
    const std::string& prefix() const { return prefix_; }
    const std::map<std::string, std::string>& metadata() const { return meta_; }
    std::vector<std::string> modules() const {
        std::vector<std::string> v;
        for (const auto& m : modules_) v.push_back(m.first);
        return v;
    }
    bool has_branch(const std::string& module) const { return modules_.count(module) != 0U; }
    const std::vector<std::string>& diff_names() const { return diffs_; }
    const LoraTensorInfo& info(const std::string& name) const {
        const auto it = tensors_.find(prefix_ + name);
        if (it == tensors_.end()) throw std::out_of_range("h3 lora: no tensor " + name);
        return it->second;
    }

    LoraBranch branch(const std::string& module, float strength) const {
        if (!has_branch(module)) throw std::out_of_range("h3 lora: no LoRA branch for " + module);
        const LoraTensorInfo& ia = info(module + ".lora_A.weight");
        const LoraTensorInfo& ib = info(module + ".lora_B.weight");
        if (ia.shape.size() != 2 || ib.shape.size() != 2 || ia.shape[0] != ib.shape[1]) throw std::runtime_error("h3 lora: bad A / B shapes for " + module);
        LoraBranch br;
        br.rank = static_cast<std::uint32_t>(ia.shape[0]);
        br.k = static_cast<std::uint32_t>(ia.shape[1]);
        br.n = static_cast<std::uint32_t>(ib.shape[0]);
        br.a = read_bf16(module + ".lora_A.weight");
        br.b = read_bf16(module + ".lora_B.weight");
        float ratio = 1.0F;
        if (tensors_.count(prefix_ + module + ".alpha")) {
            const std::vector<float> al = read_f32(module + ".alpha");
            if (al.size() != 1) throw std::runtime_error("h3 lora: alpha of " + module + " is not a scalar");
            br.alpha = al[0];
            ratio = br.alpha / static_cast<float>(br.rank);
        }
        br.scale = strength * ratio;
        return br;
    }
    std::vector<float> read_f32(const std::string& name) const { return read(name, false).f; }
    std::vector<std::uint16_t> read_bf16(const std::string& name) const { return read(name, true).h; }

private:
    static constexpr const char* kPrefix = "diffusion_model.";
    struct Data {
        std::vector<float> f;
        std::vector<std::uint16_t> h;
    };
    Data read(const std::string& name, bool to_bf16) const {
        const LoraTensorInfo& t = info(name);
        std::uint64_t count = 1;
        for (const auto d : t.shape) count *= d;
        const std::uint64_t es = t.dtype == "F32" ? 4U : (t.dtype == "BF16" || t.dtype == "F16") ? 2U : 0U;
        if (!es) throw std::runtime_error("h3 lora: unsupported dtype " + t.dtype + " of " + name);
        if (t.end - t.begin != count * es) throw std::runtime_error("h3 lora: size mismatch of " + name);
        std::ifstream f(path_, std::ios::binary);
        f.seekg(static_cast<std::streamoff>(data0_ + t.begin));
        std::vector<char> raw(t.end - t.begin);
        f.read(raw.data(), static_cast<std::streamsize>(raw.size()));
        if (!f) throw std::runtime_error("h3 lora: cannot read " + name);
        Data d;
        if (to_bf16) d.h.resize(count);
        else d.f.resize(count);
        for (std::uint64_t i = 0; i < count; ++i) {
            float v;
            if (t.dtype == "F32") {
                std::memcpy(&v, raw.data() + 4 * i, 4);
            } else {
                std::uint16_t u;
                std::memcpy(&u, raw.data() + 2 * i, 2);
                if (t.dtype == "BF16") {
                    if (to_bf16) { d.h[i] = u; continue; }
                    v = gemm::bf16_to_f32(u);
                } else {
                    v = detail::f16_to_f32(u);
                }
            }
            if (to_bf16) d.h[i] = gemm::f32_to_bf16(v);
            else d.f[i] = v;
        }
        return d;
    }
    std::string path_, prefix_;
    std::uint64_t data0_ = 0;
    std::map<std::string, LoraTensorInfo> tensors_;
    std::map<std::string, unsigned> modules_;  // bit 0 A, 1 B, 2 alpha
    std::vector<std::string> diffs_;
    std::map<std::string, std::string> meta_;
};

// W (bf16 values, as fp32 or bf16 bits) += strength * diff, one rounding to bf16 (the reference's "diff" patch on a
// bf16 weight with an fp32 intermediate): w = bf16(fp32(w) + fp32(fp32(strength) * diff)).
inline void apply_diff_bf16(std::uint16_t* w, std::size_t n, const std::vector<float>& diff, float strength) {
    if (diff.size() != n) throw std::invalid_argument("h3 lora: diff size mismatch");
    for (std::size_t i = 0; i < n; ++i) {
        const float d = strength * diff[i];
        w[i] = gemm::f32_to_bf16(gemm::bf16_to_f32(w[i]) + d);
    }
}
inline void apply_diff_bf16(float* w, std::size_t n, const std::vector<float>& diff, float strength) {
    if (diff.size() != n) throw std::invalid_argument("h3 lora: diff size mismatch");
    for (std::size_t i = 0; i < n; ++i) {
        const float d = strength * diff[i];
        w[i] = gemm::bf16_to_f32(gemm::f32_to_bf16(w[i] + d));
    }
}

}  // namespace strix::h3::levers::lora
