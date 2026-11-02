#pragma once
// Launchers of the H3 VAE decoder kernels (vae_kernels.hip). Buffers are device FP16 unless noted.

#include <hip/hip_runtime.h>

namespace strix::h3::vae::gpu {

// GEMM form: 1 tiled (default), 0 simple, -1 environment (STRIX_H3_VAE_GEMM=0 -> simple). Same bits either way.
void set_gemm_tiled(int on);
// tiled GEMM block tile (0: 128x256, 1: 256x256, 2: 256x128, 3: 128x128; -1: default / STRIX_H3_VAE_TILE). Same bits.
void set_gemm_tile(int tile);
// Implicit-GEMM causal conv geometry (channels-last frames x [T][H][W][C], C % 32 == 0): output row m = (t, oy, ox),
// column k = ((kt * k + ky) * k + kx) * C + ci (the im2col order), input frame t * ts + kt - front (zero outside), spatial
// reflect pad k / 2 (ss 1) or 1 on the right / bottom (ss 2). Kreal = kt k k C real columns (Kp padded with zeros).
struct ConvGeom { int T = 0, H = 0, W = 0, C = 0, Ho = 1, Wo = 1, k = 1, ts = 1, ss = 1, front = 0, Kreal = 0; };
// C[M][N] = fp16(conv(x) + b) (+ R: fp16(fp16(...) + R[m][n])) over all output frames in one launch; M = To Ho Wo.
// Same operands in the same K16 order as im2col + gemm / gemm_add: the same bits.
void conv_gemm(const _Float16* x, const ConvGeom& geo, const _Float16* W, const _Float16* bias, const _Float16* R, _Float16* C,
               unsigned ldc, unsigned M, unsigned N, unsigned Kp, hipStream_t st);
void rms_norm(const _Float16* x, const _Float16* w, _Float16* y, unsigned rows, hipStream_t st);              // width 2048
void layer_norm(const _Float16* x, const _Float16* w, const _Float16* b, _Float16* y, unsigned rows, hipStream_t st);
// C[M][N] = fp16(A W^T + b); N % 128 == 0, K % 32 == 0
void gemm(const _Float16* A, unsigned lda, const _Float16* W, const _Float16* bias, _Float16* C, unsigned ldc,
          unsigned M, unsigned N, unsigned K, hipStream_t st);
// X[M][N] = fp16(X + fp16(A W^T + b) * scale)
void gemm_residual(const _Float16* A, unsigned lda, const _Float16* W, const _Float16* bias, const _Float16* scale,
                   _Float16* X, unsigned ldx, unsigned M, unsigned N, unsigned K, hipStream_t st);
// C[M][N] = fp16(fp16(A W^T + b) + R[M][N]) (R pitch ldc)
void gemm_add(const _Float16* A, unsigned lda, const _Float16* W, const _Float16* bias, const _Float16* R, _Float16* C, unsigned ldc,
              unsigned M, unsigned N, unsigned K, hipStream_t st);
// Fused w1 + gated SiLU: Wp / bp rows in swiglu_source_row order (N = 2 x gated width); G [M][N / 2] (pitch ldg).
// Same bits as gemm (tiled) + swiglu on the unpermuted weights.
void gemm_swiglu(const _Float16* A, unsigned lda, const _Float16* Wp, const _Float16* bp, _Float16* G, unsigned ldg,
                 unsigned M, unsigned N, unsigned K, hipStream_t st);
// permuted row p of the fused w1 holds source row swiglu_source_row(p, N)
inline unsigned swiglu_source_row(unsigned p, unsigned n) {
    const unsigned tile = p / 256U, within = p % 256U, wn = within / 64U, u = (within % 64U) / 16U, c = within % 16U;
    return (u % 2U ? n / 2U : 0U) + tile * 128U + wn * 32U + (u / 2U) * 16U + c;
}
void swiglu(const _Float16* h, _Float16* g, unsigned rows, hipStream_t st);  // [rows][16384] -> [rows][8192]
void rms_rope(const _Float16* qkv, const _Float16* rope_c, const _Float16* rope_s, _Float16* Q, _Float16* K, _Float16* V,
              unsigned L, unsigned Lp, hipStream_t st);
void attention(const _Float16* Q, const _Float16* K, const _Float16* V, _Float16* O, unsigned L, unsigned Lp, hipStream_t st);
void embed(const _Float16* zin, const _Float16* w, const _Float16* b, const _Float16* registers, _Float16* x, unsigned P,
           hipStream_t st);

}  // namespace strix::h3::vae::gpu
