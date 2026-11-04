#pragma once
// GPU ViT3D video decoder of the H3 VAE (36-layer stock decoder, default tiled + temporal-chunked path).
// decode_call() runs one decoder call (one tile x temporal chunk) and is a pure function of its latent box;
// decode() runs a plan's calls in canonical order (or a contiguous range of them, the two-node split) and
// blends on the host in canonical order (vae_plan.hpp), so 1-node and 2-node frames are bit-identical.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "h3/vae/vae_plan.hpp"
#include "h3/vae/vae_ref.hpp"
#include "h3/vae/vae_shard.hpp"

namespace strix::h3::vae {

struct CallTiming { double embed_ms = 0, layers_ms = 0, head_ms = 0, total_ms = 0; };

struct ChunkRange { int first = 0, last = 0; };  // temporal chunks [first, last)

class GpuDecoder {
public:
    // weights: the video VAE FP16 safetensors (decoder.* + post_quant_conv + latents stats)
    // fused_swiglu: w1 + gated SiLU in one GEMM epilogue (same bits as the two-step form; needs the tiled GEMM)
    explicit GpuDecoder(const std::string& weights, int layers = kLayers36, bool fused_swiglu = true);
    ~GpuDecoder();
    GpuDecoder(const GpuDecoder&) = delete;
    GpuDecoder& operator=(const GpuDecoder&) = delete;

    const HeadH& head() const { return head_; }
    // dec_in: post_quant_conv output FP16 [24][nt][ny][nx]; returns FP16 [3][4 nt][16 ny][16 nx].
    // h_out (optional): the residual stream after `stop_after` layers (all layers when < 0), [L][2048].
    // rows_out (optional): receives the projection rows [P][3072] instead (the return value is then empty).
    H16 decode_call(const H16& dec_in, int nt, int ny, int nx, CallTiming* timing = nullptr, H16* h_out = nullptr,
                    int stop_after = -1, const H16* h_in_override = nullptr, H16* rows_out = nullptr);
    // Whole decode: normalized FP16 latents [24][T][H][W] -> FP32 frames [3][F][16H][16W] in [0, 1].
    std::vector<float> decode(const H16& z_norm, int T, int H, int W, double* seconds = nullptr);
    // Decoder outputs of calls [r.first, r.last) of the canonical order (one rank's share).
    std::vector<Pix16> decode_calls(const DecodePlan& p, const H16& z_denorm_padded, CallRange r);
    // Chunk canvases (spatially assembled, FP16 [3][28][16 H][16 W]) of chunks [r.first, r.last), GPU and host work
    // pipelined per chunk; finish_frames() turns a full canonical set into frames (same bits as blend_calls).
    std::vector<Pix16> decode_chunks(const DecodePlan& p, const H16& z_denorm_padded, ChunkRange r);
    // Two-node entry (gather mode, vae_shard.hpp): decode calls [r.first, r.last) of the canonical order, GPU and host
    // work pipelined per chunk segment, and hand every piece (whole chunk -> canvas, cut chunk -> its tiles) to `sink`.
    // sink runs on worker threads, one call at a time (serialized), pieces in any order; an exception thrown by sink
    // stops the decode and is rethrown here. Returns after the last sink call.
    void decode_pieces(const DecodePlan& p, const H16& z_denorm_padded, CallRange r,
                       const std::function<void(DecodePiece&&)>& sink);

private:
    struct Dev;
    Dev* d_ = nullptr;
    HeadH head_;
    int layers_ = 0;
    bool fused_ = true;
};

// Latent box of a call from the denormalized, padded latent [24][Tp][H][W], then post_quant_conv (host).
H16 call_input(const HeadH& head, const DecodePlan& p, const H16& z_denorm_padded, const DecoderCall& c);
// Temporal blending + finalize of all chunk canvases (canonical chunk order) into frames.
std::vector<float> finish_frames(const DecodePlan& p, std::vector<Pix16> canvases);
// Blend a full set of call outputs (canonical order) into frames.
std::vector<float> blend_calls(const DecodePlan& p, const std::vector<Pix16>& outputs);

}  // namespace strix::h3::vae
