// Host side of the step-LoRA identities (see turbo_accel.hpp): the LoRA file reader (levers lora_file.hpp, host-only
// headers) for the branches and the full-weight deltas; the device holder lives in turbo_accel.hip.
#include <cmath>
#include <stdexcept>
#include <string>

#include "h3/block/turbo_accel.hpp"
#include "h3/levers/lora/lora_file.hpp"
#include "h3/levers/lora/turbo_host.hpp"

namespace strix::h3::block {

namespace lv = strix::h3::levers::lora;

void for_each_step_lora_branch(const std::string& id, const std::string& file, float strength, std::uint32_t dit_blocks,
    std::uint32_t refiner_blocks, const std::function<void(const HostLoraBranch&)>& fn) {
    const lv::LoraIdentity& li = lv::lora_identity(id);
    if (!li.fc2_bypass) throw std::invalid_argument("step LoRA: " + id + " is not a run-time-fc2 identity (use --turbo)");
    if (!(std::isfinite(strength) && strength > 0.0F && strength <= 4.0F)) throw std::invalid_argument("step LoRA: strength (0, 4]");
    const lv::LoraFile f(file);
    auto one = [&](lv::Site site, std::uint32_t blk, bool required) {
        const std::string m = lv::lora_module_name(site, blk);
        if (!f.has_branch(m)) {
            if (required) throw std::runtime_error("step LoRA " + id + ": no branch for " + m);
            return;
        }
        lv::LoraBranch br = f.branch(m, strength);
        HostLoraBranch h;
        h.site = static_cast<std::uint8_t>(site);
        h.block = blk;
        h.k = br.k;
        h.n = br.n;
        h.rank = br.rank;
        h.scale = br.scale;
        h.a = std::move(br.a);
        h.b = std::move(br.b);
        fn(h);
    };
    for (std::uint32_t b = 0; b < dit_blocks; ++b)
        for (lv::Site s : {lv::Site::kQkv, lv::Site::kOut, lv::Site::kFc1, lv::Site::kFc2}) one(s, b, true);
    for (std::uint32_t b = 0; b < refiner_blocks; ++b)
        for (lv::Site s : {lv::Site::kRefQkv, lv::Site::kRefOut, lv::Site::kRefFc1, lv::Site::kRefFc2}) one(s, b, false);
}

// ---------------------------------------------------------------- step-LoRA full-weight deltas
namespace {
bool block_norm_target(const std::string& t, std::uint32_t* blk, int* which) {  // "blocks.N.norm1.weight" / norm2
    if (t.rfind("blocks.", 0) != 0) return false;
    const std::size_t dot = t.find('.', 7);
    if (dot == std::string::npos || dot == 7) return false;
    for (std::size_t i = 7; i < dot; ++i)
        if (t[i] < '0' || t[i] > '9') return false;
    const std::string rest = t.substr(dot + 1);
    if (rest != "norm1.weight" && rest != "norm2.weight") return false;
    if (blk) *blk = static_cast<std::uint32_t>(std::stoul(t.substr(7, dot - 7)));
    if (which) *which = rest == "norm1.weight" ? 1 : 2;
    return true;
}
const char* const kGlobalTargets[] = {"final_layer.norm.weight", "condition_proj.weight", "condition_proj.bias", "video_patch_proj.weight",
    "video_patch_proj.bias", "audio_patch_proj.weight", "audio_patch_proj.bias", "final_layer.video_out.weight", "final_layer.video_out.bias",
    "final_layer.audio_out.weight", "final_layer.audio_out.bias"};
}  // namespace

LoraDiffs::LoraDiffs(const std::string& file, float strength) : s_(strength) {
    const lv::LoraFile f(file);
    for (const std::string& n : f.diff_names()) {
        std::string t;
        if (n.size() > 7 && n.compare(n.size() - 7, 7, ".diff_b") == 0) t = n.substr(0, n.size() - 7) + ".bias";
        else if (n.size() > 5 && n.compare(n.size() - 5, 5, ".diff") == 0) t = n.substr(0, n.size() - 5) + ".weight";
        else throw std::runtime_error("step LoRA: unexpected delta name " + n);
        bool known = block_norm_target(t, nullptr, nullptr);
        for (const char* g : kGlobalTargets) known = known || t == g;
        if (!known) throw std::runtime_error("step LoRA: delta " + n + " targets " + t + ", which this runner does not patch");
        if (!d_.emplace(t, f.read_f32(n)).second) throw std::runtime_error("step LoRA: two deltas for " + t);
    }
}

std::size_t LoraDiffs::values() const {
    std::size_t v = 0;
    for (const auto& kv : d_) v += kv.second.size();
    return v;
}

void LoraDiffs::apply_global(GlobalWeights& g) const {
    auto one = [&](const char* name, auto& vec) {
        const auto it = d_.find(name);
        if (it == d_.end()) return;
        if (it->second.size() != vec.size()) throw std::runtime_error(std::string("step LoRA: delta size of ") + name);
        lv::apply_diff_bf16(vec.data(), vec.size(), it->second, s_);
    };
    one("final_layer.norm.weight", g.final_norm);
    one("condition_proj.weight", g.condition_proj.w);
    one("condition_proj.bias", g.condition_proj.b);
    one("video_patch_proj.weight", g.video_patch_proj.w);
    one("video_patch_proj.bias", g.video_patch_proj.b);
    one("audio_patch_proj.weight", g.audio_patch_proj.w);
    one("audio_patch_proj.bias", g.audio_patch_proj.b);
    one("final_layer.video_out.weight", g.video_out.w);
    one("final_layer.video_out.bias", g.video_out.b);
    one("final_layer.audio_out.weight", g.audio_out.w);
    one("final_layer.audio_out.bias", g.audio_out.b);
}

void LoraDiffs::apply_block(BlockWeights& b) const {
    for (const auto& kv : d_) {
        std::uint32_t blk = 0;
        int which = 0;
        if (!block_norm_target(kv.first, &blk, &which) || blk != b.index) continue;
        std::vector<std::uint16_t>& v = which == 1 ? b.norm1 : b.norm2;
        if (kv.second.size() != v.size()) throw std::runtime_error("step LoRA: delta size of " + kv.first);
        lv::apply_diff_bf16(v.data(), v.size(), kv.second, s_);
    }
}

std::shared_ptr<const BlockWeights> DiffBlockSource::block(std::uint32_t i) {
    auto b = std::make_shared<BlockWeights>(*base_.block(i));
    if (b->index != i) throw std::logic_error("step LoRA: block index");
    diffs_.apply_block(*b);
    return b;
}

}  // namespace strix::h3::block
