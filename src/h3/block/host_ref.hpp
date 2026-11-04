#pragma once
// MiniMax H3 DiT: host reference forward (plain C++) with pluggable linear / attention ops, plus the embedding,
// refiner, final layer and the model-level velocity conversion (audio carry). The device driver uses the same
// interfaces with its own ops.
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "h3/block/layout.hpp"
#include "h3/block/ops_ref.hpp"
#include "h3/block/weights.hpp"

namespace strix::h3::block {

// ---------------------------------------------------------------------------------------- pluggable ops
// W8A8 convrot linear: x [m][k] bf16 bits -> y [m][n] bf16 bits (the full activation-quantiser chain).
struct LinearOp {
    virtual ~LinearOp() = default;
    virtual void run(const QLinear& w, const std::uint16_t* x, std::uint32_t m, std::uint16_t* y) = 0;
};
// Dense non-causal attention: Q, K token-major [S][56][128], V^T [56][128][keys] -> out [S][7168] bf16.
struct AttentionOp {
    virtual ~AttentionOp() = default;
    virtual void run(const std::uint16_t* q, const std::uint16_t* k, const std::uint16_t* vt, std::uint32_t S,
        std::uint32_t keys, std::uint16_t* out) = 0;
};

// Host reference of the W8A8 chain (the gemm package's arithmetic; threaded, exact integer dot products).
// Groups whose magnitudes span more than the fixed-point window (tiny values flushed before the rotation; the
// rounded result can then differ from the exact rotation only at a rounding tie) are counted, not rejected.
struct HostLinear final : LinearOp {
    void run(const QLinear& w, const std::uint16_t* x, std::uint32_t m, std::uint16_t* y) override;
    std::uint64_t inexact_groups() const { return inexact_groups_.load(); }

private:
    std::atomic<std::uint64_t> inexact_groups_{0};
};
// fp64 exact-softmax attention, one bf16 rounding of the output.
struct HostAttention final : AttentionOp {
    void run(const std::uint16_t* q, const std::uint16_t* k, const std::uint16_t* vt, std::uint32_t S, std::uint32_t keys,
        std::uint16_t* out) override;
};

// Emulation of the capture's CPU attention (no kv chunking): scores rounded to bf16 (bf16(fp32(q.k) * 128^-0.5)),
// softmax in fp32 rounded to bf16, out = bf16(fp32 sum p_j v_j). Test-only: it tracks the fixtures closely so the other
// ops can be gated tightly; the engine reference is HostAttention / the flash kernel.
struct CaptureAttention final : AttentionOp {
    void run(const std::uint16_t* q, const std::uint16_t* k, const std::uint16_t* vt, std::uint32_t S, std::uint32_t keys,
        std::uint16_t* out) override;
};

// bf16 linear (refiner / condition_proj): y = bf16(s + bias) with s the fp32 k-ascending sum of x_k w_k (every product
// of two bf16 values is exact in fp32, so mul + add == fma). The device kernel uses the same order (bitwise).
void bf16_linear_host(const Bf16Linear& l, const std::uint16_t* x, std::uint32_t m, std::uint16_t* y);
// SwiGLU on fc1's [gate | up] halves: [m][2 * ffn] -> [m][ffn].
void swiglu_host(const std::uint16_t* f1, std::uint32_t m, std::uint32_t ffn, std::uint16_t* out);

// ---------------------------------------------------------------------------------------- block
// Optional sub-op captures of one block (any pointer may be null).
struct BlockTaps {
    std::vector<std::uint16_t>* norm1_out = nullptr;  // [S][5376]
    std::vector<std::uint16_t>* attn_in = nullptr;    // modulated norm1 output
    std::vector<std::uint16_t>* qkv = nullptr;        // [S][21504]
    std::vector<std::uint16_t>* q = nullptr;          // [S][56][128]
    std::vector<std::uint16_t>* k = nullptr;
    std::vector<std::uint16_t>* vt = nullptr;         // [56][128][S]
    std::vector<std::uint16_t>* attn = nullptr;       // [S][7168] attention output
    std::vector<std::uint16_t>* out_proj = nullptr;   // [S][5376]
    std::vector<std::uint16_t>* norm2_in = nullptr;   // residual after the attention half
    std::vector<std::uint16_t>* norm2_out = nullptr;
    std::vector<std::uint16_t>* mlp_in = nullptr;
    std::vector<std::uint16_t>* fc1 = nullptr;        // [S][28672]
    std::vector<std::uint16_t>* mlp_out = nullptr;    // [S][5376]
};

// Row -> mod row map of a segment table (S entries).
std::vector<std::uint32_t> mod_row_of(const std::vector<ModSegment>& segs, std::uint32_t S);

// One DiT block in place on x [S][5376] (bf16 bits).
void block_forward_host(const BlockWeights& b, const ModVectors& mv, const std::vector<std::uint32_t>& mod_row,
    const RopeTable& rope, std::vector<std::uint16_t>& x, std::uint32_t S, LinearOp& lin, AttentionOp& att,
    BlockTaps* taps = nullptr);

// ---------------------------------------------------------------------------------------- text path
// condition_proj + token refiner on [L][5120] text states (bf16) -> [L][5376] bf16. Optional per-stage captures.
std::vector<std::uint16_t> text_refine_host(const GlobalWeights& g, const std::uint16_t* text, std::uint32_t L,
    AttentionOp& att, std::vector<std::vector<std::uint16_t>>* stages = nullptr);

// ---------------------------------------------------------------------------------------- embedding
// Visual condition rows of one keyframe latent ([24][vt][H][W] fp32): patchify, then aug * r + (1 - aug) * noise
// (fp32; noise [rows][96], the reference draws it from a CPU generator seeded with the payload seed) when aug < 1.
std::vector<float> cond_video_rows(const float* latent, std::uint32_t vt, std::uint32_t H, std::uint32_t W, double aug,
    const float* noise);

struct StreamInputs {
    std::uint32_t T = 0, H = 0, W = 0;   // video latent [24][T][H][W] (H, W even)
    std::vector<std::uint16_t> video_x;  // bf16 bits (as the model receives it)
    std::uint32_t audio_t = 0;
    std::vector<std::uint16_t> audio_x;  // bf16 bits [32][2][audio_t] (the stream's own latent, un-carried)
    std::vector<std::uint16_t> text;     // [L][5376] refined text (bf16)
    std::vector<float> cond_video;       // concatenated cond rows [n][96] fp32 (keyframes then refs), may be empty
    std::vector<float> cond_audio;       // [n][32] fp32, may be empty
};

// All video-like rows [n_img][96] and audio-like rows [n_audio][32] (fp32) in packed order (cond rows in the
// non-update slots).
void embed_rows(const PackedLayout& L, const StreamInputs& in, std::vector<float>& all_v, std::vector<float>& all_a);
// Packed residual stream h [S][5376] bf16 before block 0.
std::vector<std::uint16_t> embed_and_pack(const GlobalWeights& g, const PackedLayout& L, const StreamInputs& in);

// ---------------------------------------------------------------------------------------- final layer
struct PddContext {
    std::vector<float> sample_sigmas;  // the sampler's schedule (needed when pdd_steps > 1)
    float sigma = 0.0F;                // the step's sigma (sampler value)
    float shift_video = 12.0F, shift_audio = 3.0F;
};
struct FinalRows {
    std::vector<float> video;  // [n_video][96] fp32
    std::vector<float> audio;  // [n_audio][32] fp32
    std::vector<float> shift, scale;  // final adaLN [n_t][5376] each (fp32)
};
// PDD head bank (pdd_steps > 1): the dt-weighted heads of this step (returns false for plain single heads).
bool select_heads(const GlobalWeights& g, const ForwardPlan& p, const PddContext* pdd, F32Linear& video_head, F32Linear& audio_head);
FinalRows final_layer_host(const GlobalWeights& g, const std::vector<std::uint16_t>& x, const PackedLayout& L,
    const ForwardPlan& p, const std::vector<float>& t_emb, const PddContext* pdd = nullptr);

// ---------------------------------------------------------------------------------------- whole model
// Supplies block weights (loaded lazily or cached).
struct BlockSource {
    virtual ~BlockSource() = default;
    virtual std::shared_ptr<const BlockWeights> block(std::uint32_t i) = 0;
    virtual std::uint32_t num_blocks() const = 0;
};
struct FileBlockSource final : BlockSource {
    FileBlockSource(const SafeTensors& st, std::uint32_t basis, bool cache, LoadPolicy pol = {});
    std::shared_ptr<const BlockWeights> block(std::uint32_t i) override;
    std::uint32_t num_blocks() const override { return n_; }

private:
    const SafeTensors& st_;
    std::uint32_t basis_, n_;
    bool cache_;
    LoadPolicy pol_;
    std::vector<std::shared_ptr<const BlockWeights>> blocks_;
};

struct ForwardTaps {
    std::vector<std::uint16_t>* h_in = nullptr;                // residual before block 0
    std::vector<std::vector<std::uint16_t>>* blocks = nullptr;  // residual after every block
    // called after block i with the residual (teacher forcing: may overwrite x)
    std::function<void(std::uint32_t, std::vector<std::uint16_t>&)> after_block;
    BlockTaps* block0 = nullptr;
    FinalRows* final_rows = nullptr;
    std::vector<float>* t_emb = nullptr;
};

// _forward of the reference: stream inputs -> -velocity (video [24][T][H][W], audio [32][2][Ta]) as bf16 bits.
struct StreamOutputs {
    std::vector<std::uint16_t> video_neg_v, audio_neg_v;
};
StreamOutputs stream_forward_host(const GlobalWeights& g, BlockSource& blocks, const PackedLayout& L, const ForwardPlan& p,
    const StreamInputs& in, LinearOp& lin, AttentionOp& att, ForwardTaps* taps = nullptr, const PddContext* pdd = nullptr);

// Model-level audio carry (ModelSamplingAV): the sampler carries audio * audio_scale.
// carry = bf16(sigma_a / sigma_v); audio_x = bf16(audio_carried * carry) (both bf16).
std::vector<std::uint16_t> uncarry_audio(const std::vector<std::uint16_t>& audio_carried, const ForwardPlan& p);
// out_carried = bf16( bf16((1 - scale) * bf16(audio_carried * carry)) + bf16(bf16(1 + (scale - 1) * sigma_a) * out) ).
std::vector<std::uint16_t> carry_audio_velocity(const std::vector<std::uint16_t>& audio_carried,
    const std::vector<std::uint16_t>& out_a, const ForwardPlan& p, float audio_scale);

}  // namespace strix::h3::block
