#pragma once
// H3 video VAE encoder (3D causal CNN), single-image path (T == 1: keyframes / still conditioning).
// With one frame the causal temporal taps reduce to the last tap (kt = 2) of every 3x3x3 conv, so every conv is 2-D.
//   x [3][H][W] in [-1, 1] -> fp16 chain ((x + 1) * 0.5 - mean) / std (ImageNet, FP16 constants)
//   -> conv_in (3 -> 128, 3x3, reflect pad 1)
//   -> 6 levels (channels 128 256 256 512 512 1024) x 2 resnet blocks
//        block: h = conv1(silu(gn1(x))); s = nin_shortcut(x) (1x1, when channels change) or x; out = fp16(conv2(silu(gn2(h))) + s)
//      levels 0-3 then downsample: 3x3 stride 2, reflect pad right/bottom 1 (the time stride is moot for T == 1)
//   -> conv_out(silu(gn_out(x))) (1024 -> 48) -> quant_conv (1x1, 48 -> 48) = moments; mean = first 24 channels;
//   latent = (fp32(mean) - mean_c) / std_c (FP16 checkpoint statistics, FP32 math).
// GroupNorm: 32 groups, eps 1e-6, statistics per frame; SiLU rounds separately from the norm.
// Tiled (default): 256 px tiles (same split as the decoder), moments blended in latent units (16 px), above then
// left, overlaps cut.  Fixture-free description; see enc_gpu for the device path.

#include <string>
#include <vector>

#include "h3/vae/vae_io.hpp"
#include "h3/vae/vae_plan.hpp"
#include "h3/vae/vae_ref.hpp"

namespace strix::h3::vae::enc {

struct Conv2 {  // causal conv: w = 2-D view [cout][cin][kh][kw] (last temporal tap, T == 1 path); w3 = all kt taps
    int cin = 0, cout = 0, k = 0, kt = 1;
    H16 w, b, w3;  // w3 [cout][cin][kt][kh][kw]
};
struct Norm { H16 w, b; };
struct Res { Norm n1, n2; Conv2 c1, c2, nin; bool has_nin = false; };
struct Weights {
    Conv2 conv_in, conv_out, quant;
    Res blocks[6][2];
    Conv2 down[4];
    Norm norm_out;
    H16 latents_mean, latents_std;
};
Weights load(const SafeTensors& st);

// Host ops on FP16 [C][H][W] maps.
struct Map { int C = 0, H = 0, W = 0; H16 v; };

// Supported shapes: H, W multiples of 16 and >= 32. Every 3x3 conv reflect-pads by 1, which needs >= 2 pixels at every
// level; four stride-2 downsamples take 32 px to 2 px (16 px would reach 1 px, where the reflection is undefined and the
// reference rejects the padding as well). Tiles are <= 256 px (the split keeps every tile >= 32 px when the image is).
// Throw std::invalid_argument otherwise; the buffer length must match the shape.
void check_image(int H, int W, std::size_t values);  // x [3][H][W]
void check_tile(const Map& pixels);                    // one tile: C 3, H/W in [32, 256] multiples of 16, |v| = 3 H W
// conv: stride 1 + reflect pad (k / 2) on all sides, or stride 2 + reflect pad 1 on the right/bottom only (downsample)
Map conv(const Conv2& c, const Map& x, int stride, const Map* residual = nullptr);
Map gn_silu(const Norm& n, const Map& x);
Map resblock(const Res& r, const Map& x);
Map normalize_pixels(const std::vector<float>& x, int H, int W);  // [3][H][W] in [-1, 1]
Map quant(const Weights& w, const Map& enc_out);                  // quant_conv 1x1 (48 -> 48)
Map encode_tile(const Weights& w, const Map& pixels);             // host: normalized tile -> moments (slow: tests only)
// Tiled moments: raw tiles in; each tile blends above then left against the RAW neighbouring tiles (latent units),
// bottom/right overlaps cut, rows concatenated.
Map blend_tiles(const std::vector<std::vector<Map>>& tiles, const Tiles1D& ty, const Tiles1D& tx);
std::vector<float> latent_of(const Weights& w, const Map& moments);  // (fp32(mean half) - mean) / std

// ---- multi-frame path (references / control videos): causal 3-D convs over 17-frame clips ----
// conv3: out[t] = b + sum_kt conv2d_kt(x[t * st + kt - front]) (frames < 0 are zeros), front = 2 for the 3-tap convs (k 3
// and the downsamples), 0 for 1x1x1; the spatial padding is the single-image path's. GroupNorm per frame.
struct Vid { int C = 0, T = 0, H = 0, W = 0; H16 v; };  // [C][T][H][W]
Vid conv3(const Conv2& c, const Vid& x, int t_stride, int s_stride, const Vid* residual = nullptr);
Vid gn_silu3(const Norm& n, const Vid& x);
Vid resblock3(const Res& r, const Vid& x);
Vid encode_clip_tile(const Weights& w, const Vid& pixels);  // normalized [3][T][h][w] -> moments [48][T'][h/16][w/16]
// Clip plan of encode(): 17-frame clips (the last frame repeated to fill), 5 latents each, the last 3 dropped.
inline int video_latents(int frames) { return 5 * ((frames + 16) / 17) - 3; }
void check_video(int T, int H, int W, std::size_t values);  // T >= 2, sides as check_image, |x| = 3 T H W
// one clip of 17 frames (frames [17 c, 17 c + 17) of x, the last frame repeated past the end), normalized as the images
Vid clip_pixels(const std::vector<float>& x, int T, int H, int W, int clip);
// tiled moments of one clip: raw tile moments in (Vid per tile), blended per frame above then left, overlaps cut
Vid blend_tiles3(const std::vector<std::vector<Vid>>& tiles, const Tiles1D& ty, const Tiles1D& tx);
// clips' moments [48][5][h][w] -> latent FP32 [24][5 n - 3][h][w]
std::vector<float> video_latent(const Weights& w, const std::vector<Vid>& clip_moments);

}  // namespace strix::h3::vae::enc
