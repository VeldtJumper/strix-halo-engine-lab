#pragma once
// FAST-class lever (identity "h3-vae-fast24-v1", OFF by default): the distilled 24-layer non-causal ViT decoder that
// decodes a whole clip at once with local attention. Not the reference decoder; quality is gated as a delta against the
// 36-layer decoder (PSNR reported) and it is only used when explicitly requested.
//
// Clip rows are block-major: per 5-latent block b, its tokens in (t, y, x) order, then 5 global rows (4 register
// tokens + 1 zero row). Every layer is the 36-layer block math on all rows, with these differences:
//   - attention: queries of a window = its tokens (16 x 16 spatial x the latents of one 5-latent temporal window,
//     clipped); keys = those tokens + the global rows of the one or two blocks the temporal window touches;
//     odd layers shift the windows by (2 latents, 8, 8); the global rows of block b attend to all rows of block b;
//   - rope ids are window-local: per axis ((i + shift) % window + 0.5) / window * 2 - 1 (FP32, then FP16), global rows 0;
//   - x_embedder runs on a contiguous row view: one rounding (fp16(x W^T + b));
//   - output: per block, norm_out + proj_out + unpatchify of its tokens, the first 3 frames of every block dropped
//     (4 T - 3 ceil(T / 5) frames); no tiling, no temporal blending.

#include <string>
#include <vector>

#include "h3/vae/vae_ref.hpp"

namespace strix::h3::vae::fast {

inline constexpr const char* kIdentity = "h3-vae-fast24-v1";

// One attention problem: queries rows q[0..nq), keys rows k[0..nk) (row indices into the clip rows).
struct Problem { unsigned qoff = 0, nq = 0, koff = 0, nk = 0; };
struct Geometry {
    int T = 0, H = 0, W = 0, nb = 0;
    std::vector<int> block_row0;     // first row of block b (nb + 1 entries: last = total rows)
    std::vector<int> token_row;      // clip row of latent token (t, y, x)
    struct Shift {
        std::vector<unsigned> qidx, kidx;
        std::vector<Problem> problems;
        std::vector<float> ids;      // [rows][3] FP16 values (window-local rope ids)
    } shift[2];
    int rows() const { return block_row0.back(); }
};
Geometry build_geometry(int T, int H, int W);
// rope table rows from ids: c, s FP16 values [rows][24]
void rope_from_ids(const std::vector<float>& ids, int rows, std::vector<float>& c, std::vector<float>& s);

class GpuDecoder {
public:
    explicit GpuDecoder(const std::string& weights);  // the fast decoder safetensors (FP32, unprefixed keys)
    ~GpuDecoder();
    GpuDecoder(const GpuDecoder&) = delete;
    GpuDecoder& operator=(const GpuDecoder&) = delete;
    // normalized FP16 latents [24][T][H][W] -> FP32 frames [3][4 T - 3 nb][16 H][16 W] in [0, 1]
    std::vector<float> decode(const H16& z_norm, int T, int H, int W, double* seconds = nullptr);
    int frames(int T) const { return 4 * T - 3 * ((T + 4) / 5); }

private:
    struct Dev;
    Dev* d_ = nullptr;
    HeadH head_;
};

}  // namespace strix::h3::vae::fast
