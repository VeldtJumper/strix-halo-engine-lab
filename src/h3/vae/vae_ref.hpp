#pragma once
// Plain C++ host reference of the H3 ViT3D video decoder (one decoder call = one tile x temporal chunk).
// Numerics follow the reference module run in FP16: every tensor op computes in FP32 and rounds its FP16
// output once; linears accumulate in FP32 (order differs from the reference's GEMM: not bitwise), norms are
// single-round FP32 (bitwise for RMSNorm), RoPE rounds after each product/add like the eager split-half path.
//
// Per call (tokens L = nt * ny * nx + 5: patches in (t, y, x) order, 4 register tokens, 1 zero token):
//   z  [24][nt][ny][nx] (denormalized FP16) -> post_quant_conv (1x1x1, 24 -> 24) -> x_embedder (24 -> 2048)
//   36 x { a = rms(x) * norm1; qkv = a Wqkv + b; per head (32): q,k,v = qkv[h*192 + 0/64/128 ..];
//          q,k = rms(.)(unit, eps 1e-5) then split-half rope on dims [0, 48) (pair i: i, i + 24);
//          o = softmax(q k^T / 8) v; x = x + (o Wout + b) * scale1;
//          a = rms(x) * norm2; h = a W1 + b1 [16384]; g = silu(h[:8192]) * h[8192:]; x = x + (g W2 + b2) * scale2 }
//   y = LayerNorm(x) Wproj + b [3072] -> unpatchify (c, pt 4, py 16, px 16) of the first nt*ny*nx rows.

#include <cstdint>
#include <string>
#include <vector>

#include "h3/vae/vae_half.hpp"
#include "h3/vae/vae_io.hpp"

namespace strix::h3::vae {

constexpr int kDim = 2048, kHeads = 32, kHeadDim = 64, kQkv = 3 * kDim, kFf = 8192, kFf2 = 2 * kFf, kProj = 3072;
constexpr int kRotPairs = 24, kRotDim = 48, kRegisters = 4, kSuffix = 5, kLatentC = 24, kLayers36 = 36;

using H16 = std::vector<half_bits>;

struct LinearH {  // FP16 weights [out][in], bias [out]
    int in = 0, out = 0;
    H16 w, b;
};
struct LayerH {
    H16 norm1, norm2, scale1, scale2;
    LinearH qkv, out, w1, w2;
};
struct HeadH {  // everything outside the 36 blocks
    LinearH post_quant, x_embed, proj;  // post_quant: [24][24] (1x1x1 conv), x_embed [2048][24], proj [3072][2048]
    H16 registers, norm_out_w, norm_out_b, latents_mean, latents_std;
};

LinearH load_linear(const SafeTensors& st, const std::string& prefix);  // prefix.weight / prefix.bias (F16)
LayerH load_layer(const SafeTensors& st, int layer, const std::string& block_prefix = "decoder.transformer_blocks.");
HeadH load_head(const SafeTensors& st, const std::string& dec_prefix = "decoder.");

// RoPE table of one call: per token, per pair, fp16 cos and fp16 sin (as the reference's cast table:
// entries [c, -s, s, c]). Suffix tokens have id 0 (c = 1, s = 0).
struct RopeTable {
    int L = 0;
    std::vector<float> c, s;  // [L][24], FP16 values
};
RopeTable rope_table(int nt, int ny, int nx);

// Host ops (FP16 tensors as bits, row-major). Multithreaded over rows.
H16 linear(const H16& x, int rows, const LinearH& w, bool round_before_bias = false);  // fp16(x W^T + b) or fp16(fp16(x W^T) + b)
void linear_residual(H16& x, const H16& in, int rows, const LinearH& w, const H16& scale);  // x = fp16(x + fp16(in W^T + b) * scale)
H16 rms_norm(const H16& x, int rows, int width, const H16* weight, float eps);
H16 layer_norm(const H16& x, int rows, int width, const H16& w, const H16& b, float eps);
// Attention of one call: q,k,v FP16 [L][32][64] views inside qkv rows (row stride 6144), rope applied here.
// Returns o FP16 [L][2048] (head-major columns h * 64 + d). Softmax and P V in FP32 (exact-ish, the reference SDPA).
half_bits nan_to_num16(half_bits h);
H16 attention(const H16& qkv, int L, const RopeTable& rope, H16* q_out = nullptr, H16* k_out = nullptr);
void layer_forward(const LayerH& w, H16& x, int L, const RopeTable& rope);

// Embedding of one call: z FP16 [24][nt][ny][nx] (decoder input before post_quant_conv) -> x FP16 [L][2048].
H16 post_quant(const HeadH& h, const H16& z, int voxels);  // [24][voxels] -> [24][voxels]
H16 embed(const HeadH& h, const H16& dec_in, int nt, int ny, int nx);  // dec_in = post_quant output
// Final norm + projection + unpatchify -> FP16 [3][4 nt][16 ny][16 nx].
H16 head_out(const HeadH& h, const H16& x, int nt, int ny, int nx, H16* proj_rows = nullptr);
H16 unpatchify(const H16& proj_rows, int nt, int ny, int nx);

// Comparison helpers.
struct Diff {
    std::size_t n = 0, mismatches = 0, nonfinite = 0;  // nonfinite: positions where either side is NaN / Inf
    bool size_mismatch = false;
    double max_abs = 0, rms_err = 0, rms_ref = 0;
    bool valid() const { return !size_mismatch && n > 0 && nonfinite == 0; }
    bool bitwise() const { return valid() && mismatches == 0; }
    double psnr(double peak) const;  // 20 log10(peak / rms_err); -inf when !valid()
    double rel_l2() const;           // +inf when !valid()
    double snr_db() const;           // 20 log10(rms_ref / rms_err); -inf when !valid()
};
Diff diff_f16(const H16& got, const H16& ref);
Diff diff_f32(const std::vector<float>& got, const std::vector<float>& ref);

}  // namespace strix::h3::vae
