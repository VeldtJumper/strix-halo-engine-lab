#pragma once
// H3 audio VAE decoder (GAN vocoder, 32 kHz, FP32): weights, host reference, shapes.
//   z [32][2][T] normalized -> per stereo row: z * std + mean -> dec_in_proj (1x1, 32 -> 2048) -> conv_pre (k7, -> 1024)
//   -> 7 levels { ConvTranspose1d (k 9 9 4 4 4 4 4, stride 5 5 2 2 2 2 2, pad (k - s) / 2, C -> C / 2);
//                 x = (AMP_k3(x) + AMP_k7(x) + AMP_k11(x)) / 3 }
//   -> Activation1d -> conv_post (k7, 8 -> 1, no bias) -> clamp [-1, 1].  T latents -> 800 T samples.
// AMP_k(x): for d in (1, 3, 5): x = conv2_k,1(act(conv1_k,d(act(x)))) + x.
// Activation1d(x): replicate-pad 5, x2 transposed conv with the 12-tap filter (depthwise) * 2, crop 15 | 15,
//   SnakeBeta v = u + sin(exp(a) u)^2 * (1 / (exp(b) + 1e-9)), replicate-pad 5 | 6, 12-tap low-pass stride 2.

#include <string>
#include <vector>

#include "h3/vae/vae_io.hpp"

namespace strix::h3::vae::audio {

struct Conv {  // Conv1d [cout][cin][k] or ConvTranspose1d [cin][cout][k]
    int cin = 0, cout = 0, k = 0, stride = 1, pad = 0, dil = 1;
    bool transposed = false;
    std::vector<float> w, b;  // b empty = no bias
};
struct Act {
    std::vector<float> alpha, beta;  // stored (log) parameters
    float up[12] = {}, down[12] = {};
};
struct Amp {
    int k = 0;
    Conv c1[3], c2[3];
    Act a[6];  // a[2 i] before c1[i], a[2 i + 1] before c2[i]
};
struct Weights {
    std::vector<float> latents_mean, latents_std;
    Conv dec_in, conv_pre, ups[7], conv_post;
    Amp amp[21];  // level i: amp[3 i + j], kernels 3 / 7 / 11
    Act act_post;
};
Weights load(const SafeTensors& st);

using Sig = std::vector<float>;  // [C][N] one mono row
Sig conv(const Conv& c, const Sig& x, int n_in);  // returns [cout][n_out]
int conv_out_len(const Conv& c, int n_in);
Sig activation(const Act& a, const Sig& x, int C, int N);
Sig amp(const Amp& b, const Sig& x, int C, int N);

// Host reference: z_norm [32][2][T] (the latent of one clip) -> wave [2][800 T]. taps (optional): per row r,
// level outputs [r][level] ([C][N]) for diagnostics.
// Input validation shared by the host and GPU decoders: T >= 1 and |z| == 32 x 2 x T (std::invalid_argument).
void check_latent(const std::vector<float>& z, int T);
std::vector<float> decode(const Weights& w, const std::vector<float>& z_norm, int T, std::vector<std::vector<Sig>>* level_taps = nullptr);

}  // namespace strix::h3::vae::audio
