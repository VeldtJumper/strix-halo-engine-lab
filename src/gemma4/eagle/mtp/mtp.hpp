#pragma once

// Google Gemma-4-E4B-it MTP assistant (Gemma4AssistantForCausalLM, Apache-2.0; fine-tuned on Nemotron
// data, same tensor layout) as the Nemotron 3.5 Bolt E4B drafter. Semantics: the assistant drafter notes and
// assist_core.py ref_chain_hf (HF 5.21 SinglePositionMultiTokenCandidateGenerator), clean KV extent.
//
// Cycle start p = position of t_p, the last committed token (the target has NOT run it yet; target rows < p exist).
//   depth 1:  x = [ E_t(t_p) ; h_{p-1} ]        E_t = bf16(bf16(target token_embd[t]) * 50.5), h = target
//                                               post-final-norm hidden of row p-1 (RowPlan::export_hidden)
//   depth d:  x = [ E_t(draft_{d-1}) ; s_{d-1} ] s = post_projection(final_norm(assistant)) of depth d-1 (2560)
//   x = pre_projection(x) (5120 -> 256); 4 layers (3 sliding, 1 full): RMSNorm -> q_proj -> q_norm -> RoPE at the
//   CONSTANT position p -> attention (scale 1, 4 q heads over 2 KV heads) into the TARGET's KV: sliding layers read
//   target layer 22 (positions max(0,p-sliding_keys)..p-1), the full layer reads layer 23 (positions 0..p-1) ->
//   o_proj -> post-attn norm + residual -> pre-FFN norm -> GeGLU 2048 -> post-FFN norm + residual -> * layer_scalar.
//   final norm -> head: centroids (256->2048), top-32 clusters, 32 x 128 candidate ids via token_ordering, dots with
//   the assistant's own 256-d embed_tokens, argmax with the lowest canonical id on ties -> draft (target id space).
// No drafter KV, no priming, no rollback: the target's committed KV (positions < p) is the clean extent.
// Arithmetic: g4-v1 scalar recipe; HostDrafter and DeviceDrafter are bit-identical by construction.

#include "gemma4/eagle/eagle.hpp"

#include <array>
#include <filesystem>
#include <memory>
#include <string_view>
#include <vector>

namespace strix::gemma4::eagle::mtp {

inline constexpr const char* arithmetic_id = "g4-mtp-v1";
inline constexpr std::uint32_t max_depth = 7, layers = 4, attention_splits = 32;

struct Config {
    std::uint32_t hidden = 256, backbone = 2560, heads = 4, kv_heads = 2, ffn = 2048, vocab = 262144;
    // top_k: clusters selected at inference. HF/config uses 32; drafter default 64 (+1.3 % tokens/cycle offline,
    // +1.1 MB/step). Runtime override STRIX_G4_MTP_TOPK (1..64); see top_k_from_environment().
    std::uint32_t sliding_dim = 256, full_dim = 512, centroids = 2048, top_k = 64;
    // Sliding layers attend the last sliding_keys target positions. HF = window + 1 = 513; the engine's layer-22 ring
    // holds 512 slots, so the engine default is 512 (ablation: assist zeroshot with window 511).
    std::uint32_t sliding_keys = 512;
    std::array<bool, layers> full{false, false, false, true};
    float embedding_scale = 50.5F;
    // Confidence-window defaults of this drafter (engine::ConfidenceConfig, used when STRIX_G4_MTP_CONF is unset):
    // optional config.json keys confidence_tau (0,1] / confidence_max_k / confidence_first (1..7); 0 = key absent.
    // confidence_builtin: the engine's builtin tau for this drafter (known weight hashes, see builtin_confidence_tau).
    float confidence_tau = 0.0F, confidence_builtin = 0.0F;
    std::uint32_t confidence_max_k = 0, confidence_first = 0;
    [[nodiscard]] std::uint32_t per_cluster() const { return vocab / centroids; }
    [[nodiscard]] std::uint32_t candidates() const { return top_k * per_cluster(); }
    [[nodiscard]] std::uint32_t head_dim(std::uint32_t l) const { return full[l] ? full_dim : sliding_dim; }
    void validate() const;
};

[[nodiscard]] std::uint32_t top_k_from_environment(std::uint32_t fallback = 64); // STRIX_G4_MTP_TOPK
// Builtin confidence-window tau for known drafter weights (sha256 of model.safetensors), 0 when unknown (the engine
// then uses ConfidenceConfig::builtin_tau). Measured on the C1 gate prompts (EN/ES/JA/code), ids == OFF.
[[nodiscard]] float builtin_confidence_tau(std::string_view weights_sha256);
// Optional confidence-window keys of drafter-mtp/config.json (validated; 0 = absent). Weights::load applies them.
struct ConfidenceKeys { float tau = 0.0F; std::uint32_t max_k = 0, first = 0; };
[[nodiscard]] ConfidenceKeys confidence_keys(std::string_view config_json);

// Packaging convention (release): the drafter ships as <model dir>/drafter-mtp/ with config.json, model.safetensors,
// LICENSE, NOTICE and LINEAGE.md (src/gemma4/eagle/mtp/export_drafter.py builds it). Accepts the drafter directory
// itself, a model directory containing drafter-mtp/, or a model file whose directory contains it. Returns the
// directory holding model.safetensors, or an empty path when there is no drafter (callers then run exact without one).
[[nodiscard]] std::filesystem::path resolve_drafter(const std::filesystem::path& path);
inline constexpr const char* drafter_subdirectory = "drafter-mtp";

struct Layer {
    Matrix q, o, gate, up, down;
    std::vector<float> input_norm, q_norm, post_attn_norm, pre_ffn_norm, post_ffn_norm;
    float scalar = 1.0F;
};
struct Weights {
    Config config;
    Matrix pre_projection, post_projection, centroids, table; // table = assistant embed_tokens / tied lm_head [V][256]
    std::array<Layer, layers> layer;
    std::vector<float> final_norm;
    std::vector<std::uint32_t> ordering; // token_ordering (permutation of [0, vocab))
    std::string source;                 // content hash of model.safetensors
    std::filesystem::path directory;    // resolved drafter directory
    bool has_license = false, has_notice = false, has_lineage = false;
    void validate() const;
    // Original HF layout (model.safetensors, BF16 + I64 token_ordering), as the assistant directory itself or as
    // <model dir>/drafter-mtp (see resolve_drafter). matrices: bf16 keeps the checkpoint bytes, q8_0 repacks every
    // linear (incl. the 256-d head table) for a ~2x smaller step. config.json, when present, must name
    // gemma4_assistant with backbone_hidden_size 2560.
    static Weights load(const std::filesystem::path& path, WeightFormat matrices = WeightFormat::bf16);
    static Weights random(const Config& config, std::uint64_t seed, WeightFormat matrices = WeightFormat::bf16);
};

// Read-only view of one target KV producer, addressed by ABSOLUTE position (same rule as the target's KvRead):
//   position >= frontier: transaction overlay row (position - frontier) < candidate_rows, else missing;
//   otherwise committed slot = global ? position : position % slots, tags[slot] must equal the position.
// Row = [kv_heads][head_dim] FP32 (post k_norm/RoPE K, post v_norm V). Missing/mismatched tags are a hard failure.
// Absolute positions read for cycle start p (anchor t_p not yet run, clean extent):
//   full layer (producer 23):    0 .. p-1
//   sliding layers (producer 22): max(0, p - sliding_keys) .. p-1   (engine: sliding_keys = 512 = the committed ring)
// HF's clean extent for the sliding layers is the last W+1 = 513 positions p-513 .. p-1; the 512-slot ring has already
// overwritten p-513 (its slot holds p-1), so for p >= 513 exactly position p-513 is omitted. t_p's own K/V at p is not
// part of the clean extent (the target has not run t_p), so no overlay row is read at draft time (candidate_rows 0).
struct KvLayer {
    const float* k = nullptr;
    const float* v = nullptr;
    const std::uint64_t* tags = nullptr;
    std::uint32_t width = 0, slots = 0;
    bool global = false;
    const float* overlay_k = nullptr;
    const float* overlay_v = nullptr;
    std::uint64_t frontier = UINT64_MAX;
    std::uint32_t candidate_rows = 0;
};
struct Inputs {
    std::uint32_t anchor = 0;        // t_p
    std::uint64_t position = 0;      // p >= 1
    std::span<const float> hidden;   // h_{p-1}, backbone floats (host memory)
    KvLayer sliding, full;           // target layers 22 and 23 (host pointers for HostDrafter, device for DeviceDrafter)
};
struct Tables {                       // target RoPE tables (RopeTables layout)
    std::span<const float> sliding;  // [positions][128][cos,sin]
    std::span<const float> global;   // [positions][256][cos,sin], 64 active pairs
    std::uint32_t positions = 0;
    void validate() const;
};
// Optional head statistics of a draft (Drafter::set_statistics). The draft ids never depend on them.
inline constexpr std::uint32_t statistic_probability = 1, statistic_margin = 2, statistic_all = 3;
struct Draft {
    std::uint32_t depth = 0;
    std::array<std::uint32_t, max_depth> tokens{};
    std::array<float, max_depth> prob{};   // top-1 softmax probability over the 4096 candidates (0 when not requested)
    std::array<float, max_depth> margin{}; // top-1 minus top-2 candidate logit (diagnostics; 0 when not requested)
};
struct Timings {
    double draft_ms = 0;
    std::uint64_t launches = 0;
    bool graph = false;
};

class Drafter {
public:
    virtual ~Drafter() = default;
    [[nodiscard]] virtual const Config& config() const = 0;
    [[nodiscard]] virtual const Timings& timings() const = 0;
    virtual Draft draft(const Inputs& inputs, std::uint32_t k) = 0;
    // Continue the most recent draft() call to total depth k (its depth < k <= max_depth). Result (tokens, prob,
    // margin of every depth) is bit-identical to draft(inputs, k) from scratch: the chain is deterministic. The target
    // KV views passed to draft() must be unchanged. Default: re-runs draft() on the inputs saved by remember().
    virtual Draft extend(std::uint32_t k);
    // Which head statistics draft()/extend() compute (statistic_* bits, default all). Without a consumer (plain
    // propose_window) the session requests none, so the head skips the exp-sum and runner-up reductions; the
    // confidence window requests the probability only. Tokens are identical for every setting.
    void set_statistics(std::uint32_t bits) { statistics_ = bits & statistic_all; }
    [[nodiscard]] std::uint32_t statistics() const { return statistics_; }
protected:
    // Saves the inputs (hidden row copied; KV views by pointer) and depth of a draft() call for the default extend().
    void remember(const Inputs& inputs, std::uint32_t k);
private:
    Inputs saved_;
    std::vector<float> saved_hidden_;
    std::uint32_t saved_depth_ = 0;
    std::uint32_t statistics_ = statistic_all;
};

class HostDrafter final : public Drafter {
public:
    // target_embedding: target token_embd (host view); weights/tables outlive the drafter.
    HostDrafter(const Weights& weights, Tables tables, LinearView target_embedding);
    ~HostDrafter() override;
    [[nodiscard]] const Config& config() const override;
    [[nodiscard]] const Timings& timings() const override;
    Draft draft(const Inputs& inputs, std::uint32_t k) override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Host recipes shared with the device kernels (exposed for tests).
// Split-K attention of one query head over keys [first, last] (scale 1): fixed attention_splits contiguous splits,
// per split: lane-ordered q.k, max, declared exp, ordered sums; merge in split order with exp(m_s - M) rescale.
void attend_head(const float* q, std::uint32_t dim, const KvLayer& kv, std::uint32_t kv_head, std::uint64_t first,
                 std::uint64_t last, float* out);
// Top-k cluster ids: repeated argmax (higher value, lower index on ties), in selection order.
std::vector<std::uint32_t> top_clusters(std::span<const float> logits, std::uint32_t k);
// One compare-exchange of the device top-k bitonic network (power-of-two n, stage `size`, `stride`, pair p < n/2):
// sorts (value, id) best first under better(). Running every pair of every stage (size = 2..n, stride = size/2..1)
// sorts the array; its first k ids equal top_clusters (distinct ids make better() a total order).
STRIX_G4_HD inline void bitonic_pair(float* value, std::uint32_t* id, std::uint32_t size, std::uint32_t stride,
                                     std::uint32_t p) {
    const std::uint32_t lo = 2U * stride * (p / stride) + p % stride, hi = lo + stride;
    const bool forward = (lo & size) == 0U; // forward runs put the better element first
    if (forward ? better(value[hi], id[hi], value[lo], id[lo]) : better(value[lo], id[lo], value[hi], id[hi])) {
        const float v = value[lo]; value[lo] = value[hi]; value[hi] = v;
        const std::uint32_t x = id[lo]; id[lo] = id[hi]; id[hi] = x;
    }
}

} // namespace strix::gemma4::eagle::mtp
