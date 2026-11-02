#pragma once

// MiniMax H3 text encoder, GPU text path (gfx1151).
//
// Residency: TextEncoder::load uploads the 4-bit pack (codes + block scales, ~15.7 GB); encode()
// expands each layer's weights to FP16 panels (exact: e2m1 * e4m3 fits FP16) just before the layer
// runs, so at most one layer of FP16 weights (~1 GB) is resident. The object can be destroyed after
// the clip or kept if the headroom beside the DiT allows.
//
// Numerics (dense class, identity h3-te-lm-dense-split-v1): FP32 residual stream. Every FP16 operand
// is a hi + lo pair (lo = f16((v - hi) * 2^11)): GEMM inputs, attention Q/K/V and P. WMMA FP16 x FP16
// with FP32 accumulation; lo terms run in their own chains and are added x 2^-11 in FP32; the
// per-tensor weight scale is applied in FP32 in the epilogue. RMSNorm, q/k norm, RoPE, SwiGLU,
// softmax in FP32. The causal diagonal 16-key slice of P.V is summed in FP32 per row (masked keys are
// never fed to a WMMA). Fast class (STRIX_H3_TE_SPLIT=0, identity h3-te-lm-fast-f16-v1): hi only.
// Deterministic and row-count invariant: no split-K, no atomics on values, every reduction in a
// fixed order independent of the sequence length; row i depends only on tokens 0..i.
//
// Output contract: FP32 [S][5120] = residual after layer 50 (no final norm); the DiT side casts to
// its inference dtype before its condition projection. Token tags from the presentation layer.

#include "h3/te/te_tokenizer.hpp"
#include "h3/te/te_vision_host.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strix::h3::te {

struct EncodeTimings {
    double upload_ms = 0;     // embedding gather + upload
    double forward_ms = 0;    // 50 layers incl. per-layer weight expansion
    double download_ms = 0;
};

// Input of a generic encode: layer-0 input embeddings (text rows already looked up, vision rows
// = merged vision tokens, spliced prompt embeddings), optional 3-axis MRoPE positions [3][S]
// (T, H, W; empty: 0..S-1), optional DeepStack additions after layers 0..2 at visual_rows.
struct EncodeInput {
    std::vector<float> embeds;                   // [S][5120]
    std::vector<float> positions;                // [3][S] or empty
    std::vector<std::vector<float>> deepstack;   // each [visual_rows.size()][5120]
    std::vector<std::uint32_t> visual_rows;
};

class TextEncoder {
public:
    // pack_dir: te-pack.json, te-embed.safetensors, te-blk.NN.safetensors.
    static std::unique_ptr<TextEncoder> load(const std::string& pack_dir, int n_layers = 50);
    ~TextEncoder();

    // Hidden states after the last layer, row-major [ids.size(), 5120] FP32. An id of -2
    // (kEmbeddingRow) takes the next 5120 values of `spliced` as its input embedding (prompt
    // embeddings of the front end, rows in order of appearance), as the reference does.
    [[nodiscard]] std::vector<float> encode(const std::vector<std::int32_t>& ids, EncodeTimings* t = nullptr,
                                            const std::vector<float>& spliced = {});
    [[nodiscard]] std::vector<float> encode(const EncodeInput& in, EncodeTimings* t = nullptr);
    // Debug: encode() returns the intermediate of decoder layer `layer` (0-based) at `stage`
    // (1 attn-norm out, 2 qkv, 3 q after norm+rope, 4 attention out, 5 after o_proj, 0 layer out);
    // layer < 0 disables. Row-major [S][width] FP32.
    void set_debug_stop(int layer, int stage);
    // Debug (shared-input sublayer replay): the next encode() calls treat their input as the residual
    // entering layer first_layer and start there; ffn_only skips that layer's attention block.
    // Combine with set_debug_stop(layer, 5) to stop after an attention block.
    void set_replay(int first_layer, bool ffn_only);
    // Debug: keep a copy of residual row `row` after every layer of the next encode() calls
    // (row < 0 disables); row_taps() = [n_layers][5120]. With sub = true, per layer
    // [attn norm 5120 | o_proj input 8192 | residual after attention 5120 | FFN norm 5120 |
    //  down_proj input 25600 | layer output 5120] (GEMM inputs as hi + lo / 2^11).
    void set_row_tap(int row, bool sub = false);
    [[nodiscard]] const std::vector<float>& row_taps() const;
    // Reference embedding lookup (bf16-rounded rows) of token ids into out [ids.size()][5120].
    // kEmbeddingRow (-2) ids take the next 5120 values of *spliced (rows in order of appearance).
    void embed_rows(const std::vector<std::int32_t>& ids, std::vector<float>& out,
                    const std::vector<float>* spliced = nullptr) const;
    // Time of expanding all layers once (diagnostic).
    [[nodiscard]] double expand_all_ms();
    [[nodiscard]] double load_seconds() const noexcept;
    // SHA-256 of te-pack.json (which lists the SHA-256 of every pack file, verified at load unless
    // STRIX_H3_TE_VERIFY=0): the content identity of the weights, to key caches next to identity().
    [[nodiscard]] std::string pack_digest() const;
    // Numeric identity: "h3-te-lm-dense-split-v1" (hi + lo FP16 activations, the dense class) or
    // "h3-te-lm-fast-f16-v1" (STRIX_H3_TE_SPLIT=0, plain FP16 activations; fast class, gated vs dense).
    [[nodiscard]] std::string identity() const;
    [[nodiscard]] std::uint64_t device_bytes() const noexcept;

private:
    TextEncoder();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

namespace probe {
std::vector<float> wmma_rows(const std::vector<float>& a16x16, const std::vector<float>& b16x16, const float* c16x16 = nullptr);
std::vector<float> attention_rows(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
                                  unsigned rows, bool split, int variant = 0);
std::vector<float> attention_taps(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
                                  unsigned rows, bool split, unsigned row);
std::vector<float> gemm_rows(const std::vector<float>& x, const std::vector<float>& w, unsigned rows, unsigned n, unsigned k,
                             bool split);
}  // namespace probe

struct VisionOutput {
    std::vector<float> merged;    // [merged tokens][5120]
    std::vector<float> deep[3];   // DeepStack features of vision layers 8, 16, 24
};

// Vision tower (27 x 1152 ViT, 2D RoPE, 2x2 merger, 3 DeepStack mergers). bf16 weights expanded to
// FP16 panels with a power-of-two per-tensor scale (exact for values in the f16 normal range),
// FP32 residual, LayerNorm / GELU / softmax in FP32; activations hi + lo split like the LM
// (STRIX_H3_TE_VSPLIT=0: hi only). One call per vision block (image or 2-frame video block).
class VisionTower {
public:
    static std::unique_ptr<VisionTower> load(const std::string& pack_dir);
    ~VisionTower();
    [[nodiscard]] VisionOutput run(const Patches& p, double* ms = nullptr);
    [[nodiscard]] std::uint64_t device_bytes() const noexcept;

private:
    VisionTower();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Whole presented prompt: expands the vision slots, runs the tower per block, splices the merged
// tokens, builds MRoPE positions and DeepStack rows, runs the text encoder.
// vt may be null for prompts without vision blocks. spliced: rows of the named prompt embeddings
// (Presented::embeddings, concatenated in order), one 5120 row per kEmbeddingRow id.
[[nodiscard]] std::vector<float> encode_presented(TextEncoder& te, VisionTower* vt, const Presented& pres,
                                                  const std::vector<Frames>& blocks,
                                                  const std::vector<float>& spliced = {},
                                                  EncodeTimings* t = nullptr, double* vision_ms = nullptr);

// A prompt embedding file (safetensors, tensor `key` of [rows, 5120], bf16 or f32) as FP32 rows.
struct PromptEmbedding {
    int rows = 0;
    std::vector<float> data;
};
[[nodiscard]] PromptEmbedding read_prompt_embedding(const std::string& path, const std::string& key = "qwen3vl_32b");

}  // namespace strix::h3::te
