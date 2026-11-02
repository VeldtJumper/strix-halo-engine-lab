#pragma once
// MiniMax H3 video VAE decode plan (36-layer ViT3D decoder, default tiled path) and the canonical
// blending that turns per-call decoder outputs into frames. Everything here mirrors the reference
// module's integer plan and its FP16 elementwise blending bit for bit (host code; cheap next to the decoder).
//
// A decoder call decodes one spatial tile (<= 16 x 16 latents) of one temporal chunk (7 latents):
// output FP16 [3][4 * 7][16 h][16 w]. Calls are independent; only the blending is ordered.

#include <cstdint>
#include <functional>
#include <vector>

#include "h3/vae/vae_half.hpp"

namespace strix::h3::vae {

struct Tiles1D {
    std::vector<int> start, len, overlap;  // pixel units; overlap.size() == start.size() - 1
};

struct ChunkSpan { int t0 = 0, t1 = 0; };  // latent frames [t0, t1) of the padded latent

struct DecodePlan {
    int T = 0, H = 0, W = 0;          // latent shape (normalized latents [24][T][H][W])
    int pad_tokens = 0, num_chunks = 0;
    std::vector<ChunkSpan> chunks;
    Tiles1D ty, tx;                   // pixel tiles along H*16 and W*16
    int out_frames = 0, out_h = 0, out_w = 0;
    static constexpr int ratio = 16, ratio_t = 4, clip_length = 17, token_drop = 3, tile_size = 256, tile_overlap_min = 64;
    static constexpr int frame_pre_padding = 3, tokens_chunk_size = 5, token_overlap = 2, frame_overlap = 5;
    int calls() const { return num_chunks * int(ty.start.size() * tx.start.size()); }
};

Tiles1D split_tiles(int input_len);
DecodePlan make_plan(int T, int H, int W);  // T >= 2 (a still latent T == 1 decodes as 5 copies: not covered here)

// One decoder call: chunk c, tile row i, tile column j; latent box of the call.
struct DecoderCall {
    int index = 0, chunk = 0, ti = 0, tj = 0;
    int t0 = 0, nt = 0, y0 = 0, ny = 0, x0 = 0, nx = 0;  // latent units
};
std::vector<DecoderCall> decoder_calls(const DecodePlan& p);  // canonical order: chunk, tile row, tile column
// Two-node partition: rank r of `world` owns the contiguous call range [first, last) of the canonical order.
// Decoder outputs do not depend on where a call runs (each call is a pure function of its latent box and
// the kernels are batch invariant), and the blending below runs once in canonical order on the gathered
// outputs, so any partition gives the same frames bit for bit.
struct CallRange { int first = 0, last = 0; };
CallRange partition(const DecodePlan& p, int rank, int world);

// FP16 tensor stored as bits, layout [C][F][Hp][Wp] (C = 3).
struct Pix16 {
    int C = 3, F = 0, Hh = 0, Ww = 0;
    std::vector<half_bits> v;
    half_bits& at(int c, int f, int y, int x) { return v[((std::size_t(c) * F + f) * Hh + y) * Ww + x]; }
    half_bits at(int c, int f, int y, int x) const { return v[((std::size_t(c) * F + f) * Hh + y) * Ww + x]; }
};

// Reference blend along one axis (0 = frames, 1 = rows, 2 = columns): a's last `extent` slices against b's
// first ones, weights fp16(pos / e) and fp16(1 - fp16(pos / e)), fp16(fp16(a * wa) + fp16(b * wb)); b's rest kept.
Pix16 blend(const Pix16& a, const Pix16& b, int extent, int axis);

// Spatial tiling of one chunk: tile(i, j) returns the decoder output of that tile; returns the canvas.
Pix16 assemble_chunk(const DecodePlan& p, const std::function<Pix16(int i, int j)>& tile);

// Temporal chunks -> float32 frames [3][out_frames][out_h][out_w] in [0, 1] (finalize: fp32(x) * std + mean,
// clamp; std/mean are the ImageNet constants rounded to FP16 as the reference's cast buffers).
std::vector<float> assemble_video(const DecodePlan& p, const std::function<Pix16(int chunk)>& chunk_canvas);

// Denormalize: fp16(fp16(z * std) + mean) with the checkpoint's FP16 latents_std / latents_mean, then pad the
// time axis with copies of the last latent. z: FP16 bits [24][T][H][W] -> [24][T + pad][H][W].
std::vector<half_bits> denormalize_and_pad(const DecodePlan& p, const std::vector<half_bits>& z,
                                           const half_bits* latents_mean, const half_bits* latents_std);

}  // namespace strix::h3::vae
