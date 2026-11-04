#pragma once
// GPU encoder of the H3 video VAE, single-image path (keyframes): convs as im2col + the f16 WMMA GEMM
// (channels-last activations), GroupNorm+SiLU, residual epilogue; tiling and latent-space blending on the host.

#include <string>
#include <vector>

#include "h3/vae/enc_ref.hpp"

namespace strix::h3::vae::enc {

// multi-frame convs: 1 implicit GEMM per conv (default), 0 per-output-frame im2col + GEMM, -1 environment. Same bits.
void set_implicit_conv(int on);

class GpuEncoder {
public:
    explicit GpuEncoder(const std::string& weights);
    ~GpuEncoder();
    GpuEncoder(const GpuEncoder&) = delete;
    GpuEncoder& operator=(const GpuEncoder&) = delete;
    // one tile: normalized pixels FP16 [3][h][w] (h, w <= 256, multiples of 16) -> moments FP16 [48][h/16][w/16]
    Map moments(const Map& pixels, double* gpu_ms = nullptr);
    // image x [3][H][W] in [-1, 1] (FP32; rounded to FP16 as the wrapper's cast) -> latent FP32 [24][H/16][W/16]
    std::vector<float> encode(const std::vector<float>& x, int H, int W, double* seconds = nullptr);
    // multi-frame path (references / control videos): one normalized clip tile [3][T][h][w] -> moments [48][T'][h/16][w/16]
    Vid moments_video(const Vid& pixels, double* gpu_ms = nullptr);
    // video x [3][T][H][W] in [-1, 1], T >= 2 -> latent FP32 [24][5 ceil(T / 17) - 3][H/16][W/16]
    std::vector<float> encode_video(const std::vector<float>& x, int T, int H, int W, double* seconds = nullptr);
    const Weights& weights() const { return w_; }

private:
    struct Dev;
    Dev* d_ = nullptr;
    Weights w_;
};

}  // namespace strix::h3::vae::enc
