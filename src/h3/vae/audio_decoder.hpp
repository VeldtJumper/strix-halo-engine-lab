#pragma once
// GPU GAN-vocoder decoder of the H3 audio VAE (FP32 VALU kernels, deterministic: every output's sum runs over input
// channels ascending, then taps ascending, independent of the tiling). Stereo rows decode in one launch each.

#include <string>
#include <vector>

#include "h3/vae/audio_ref.hpp"

namespace strix::h3::vae::audio {

class GpuDecoder {
public:
    explicit GpuDecoder(const std::string& weights);
    ~GpuDecoder();
    GpuDecoder(const GpuDecoder&) = delete;
    GpuDecoder& operator=(const GpuDecoder&) = delete;
    // z_norm [32][2][T] -> wave [2][800 T]; level_taps (optional): level outputs [level] = [2][C][N]
    std::vector<float> decode(const std::vector<float>& z_norm, int T, double* gpu_ms = nullptr,
                              std::vector<std::vector<float>>* level_taps = nullptr);

private:
    struct Dev;
    Dev* d_ = nullptr;
    Weights w_;
};

}  // namespace strix::h3::vae::audio
