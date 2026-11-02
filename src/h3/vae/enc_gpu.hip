#include "h3/vae/enc_gpu.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "h3/vae/vae_kernels.hpp"

namespace strix::h3::vae::enc {

namespace {
void ck(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("h3 vae encoder: ") + what + ": " + hipGetErrorString(e));
}
__device__ __forceinline__ float hf(_Float16 x) { return static_cast<float>(x); }
__device__ __forceinline__ _Float16 r16(float x) { return static_cast<_Float16>(x); }
__device__ __forceinline__ int reflect1(int i, int n) { return i < 0 ? -i : (i >= n ? 2 * n - 2 - i : i); }

// im2col of a channels-last map x [H][W][C] for a k x k conv: A [P][Kp], column (ky * k + kx) * C + ci,
// columns K..Kp zero. stride 1: reflect pad k / 2 all sides; stride 2: reflect pad 1 right/bottom.
__global__ __launch_bounds__(256) void im2col_kernel(const _Float16* __restrict__ x, _Float16* __restrict__ a, int H, int W, int C,
                                                     int k, int stride, int Ho, int Wo, int Kp) {
    const std::size_t i = std::size_t(blockIdx.x) * 256U + threadIdx.x;
    const std::size_t total = std::size_t(Ho) * Wo * Kp;
    if (i >= total) return;
    const int col = int(i % Kp);
    const std::size_t p = i / Kp;
    const int oy = int(p / Wo), ox = int(p % Wo);
    const int K = k * k * C;
    if (col >= K) { a[i] = r16(0.0F); return; }
    const int tap = col / C, ci = col % C, ky = tap / k, kx = tap % k;
    const int pad = stride == 1 ? k / 2 : 0;
    const int iy = reflect1(oy * stride + ky - pad, H), ix = reflect1(ox * stride + kx - pad, W);
    a[i] = x[(std::size_t(iy) * W + ix) * C + ci];
}

// GroupNorm (32 groups, eps 1e-6) + SiLU on x [P][C]; one block per group, fixed reduction tree (two passes).
__global__ __launch_bounds__(256) void gn_silu_kernel(const _Float16* __restrict__ x0, const _Float16* __restrict__ w,
                                                      const _Float16* __restrict__ b, _Float16* __restrict__ y0, int P, int C) {
    // frame = blockIdx.y (statistics per frame; frames independent, same per-frame arithmetic as one launch per frame)
    const _Float16* x = x0 + std::size_t(blockIdx.y) * P * C;
    _Float16* y = y0 + std::size_t(blockIdx.y) * P * C;
    __shared__ float red[256];
    const int g = int(blockIdx.x), cg = C / 32, c0 = g * cg;
    const std::size_t n = std::size_t(P) * cg;
    float s = 0.0F;
    for (std::size_t i = threadIdx.x; i < n; i += 256U) s += hf(x[(i / cg) * C + c0 + int(i % cg)]);
    red[threadIdx.x] = s;
    __syncthreads();
    for (unsigned d = 128; d; d >>= 1U) {
        if (threadIdx.x < d) red[threadIdx.x] += red[threadIdx.x + d];
        __syncthreads();
    }
    const float mu = red[0] / float(n);
    __syncthreads();
    float q = 0.0F;
    for (std::size_t i = threadIdx.x; i < n; i += 256U) { const float d = hf(x[(i / cg) * C + c0 + int(i % cg)]) - mu; q += d * d; }
    red[threadIdx.x] = q;
    __syncthreads();
    for (unsigned d = 128; d; d >>= 1U) {
        if (threadIdx.x < d) red[threadIdx.x] += red[threadIdx.x + d];
        __syncthreads();
    }
    const float inv = 1.0F / sqrtf(red[0] / float(n) + 1e-6F);
    for (std::size_t i = threadIdx.x; i < n; i += 256U) {
        const std::size_t idx = (i / cg) * C + c0 + int(i % cg);
        const int c = c0 + int(i % cg);
        const float nv = hf(r16((hf(x[idx]) - mu) * inv * hf(w[c]) + hf(b[c])));
        y[idx] = r16(nv / (1.0F + expf(-nv)));
    }
}

// im2col of one output frame of a causal 3-D conv: x channels-last frames [T][H][W][C]; A [P][Kp], column
// ((kt * k + ky) * k + kx) * C + ci; input frame t * ts + kt - front, frames < 0 zero; spatial as im2col_kernel.
__global__ __launch_bounds__(256) void im2col3_kernel(const _Float16* __restrict__ x, _Float16* __restrict__ a, int T, int H, int W, int C,
                                                      int kt, int k, int ts, int ss, int front, int t_out, int Ho, int Wo, int Kp) {
    const std::size_t i = std::size_t(blockIdx.x) * 256U + threadIdx.x;
    const std::size_t total = std::size_t(Ho) * Wo * Kp;
    if (i >= total) return;
    const int col = int(i % Kp);
    const std::size_t p = i / Kp;
    const int oy = int(p / Wo), ox = int(p % Wo);
    const int K = kt * k * k * C;
    if (col >= K) { a[i] = r16(0.0F); return; }
    const int tap = col / C, ci = col % C, tt = tap / (k * k), ky = (tap / k) % k, kx = tap % k;
    const int it = t_out * ts + tt - front;
    if (it < 0 || it >= T) { a[i] = r16(0.0F); return; }
    const int pad = ss == 1 ? k / 2 : 0;
    const int iy = reflect1(oy * ss + ky - pad, H), ix = reflect1(ox * ss + kx - pad, W);
    a[i] = x[((std::size_t(it) * H + iy) * W + ix) * C + ci];
}

struct DevConv { _Float16 *w = nullptr, *b = nullptr; int cin = 0, cout = 0, coutp = 0, k = 0, kp = 0, kt = 1; };

}  // namespace

struct GpuEncoder::Dev {
    std::vector<void*> owned;
    DevConv conv_in, conv_out, down[4], c1[6][2], c2[6][2], nin[6][2];
    DevConv v_in, v_out, v_down[4], v_c1[6][2], v_c2[6][2], v_nin[6][2];  // all temporal taps (multi-frame path)
    _Float16 *n1w[6][2], *n1b[6][2], *n2w[6][2], *n2b[6][2], *now = nullptr, *nob = nullptr;
    _Float16 *x = nullptr, *t = nullptr, *h = nullptr, *s = nullptr, *col = nullptr, *pix = nullptr;
    hipStream_t st = nullptr;
    hipEvent_t e0 = nullptr, e1 = nullptr;
    ~Dev() {
        (void)hipDeviceSynchronize();
        for (void* p : owned) (void)hipFree(p);
        if (e0) (void)hipEventDestroy(e0);
        if (e1) (void)hipEventDestroy(e1);
        if (st) (void)hipStreamDestroy(st);
    }
    _Float16* alloc(std::size_t halves) {
        void* p = nullptr;
        ck(hipMalloc(&p, halves * 2), "hipMalloc");
        owned.push_back(p);
        return static_cast<_Float16*>(p);
    }
    _Float16* upload(const H16& v) {
        _Float16* p = alloc(v.size());
        ck(hipMemcpy(p, v.data(), v.size() * 2, hipMemcpyHostToDevice), "upload");
        return p;
    }
    // weights reordered to [coutp][Kp], column (ky * k + kx) * cin + ci (im2col order), rows >= cout and columns >= K zero
    DevConv upload(const Conv2& c) {
        DevConv d;
        d.cin = c.cin; d.cout = c.cout; d.k = c.k;
        d.coutp = (c.cout + 127) / 128 * 128;
        const int K = c.k * c.k * c.cin;
        d.kp = (K + 63) / 64 * 64;
        H16 w(std::size_t(d.coutp) * d.kp, 0), b(std::size_t(d.coutp), 0);
        for (int co = 0; co < c.cout; ++co) {
            b[std::size_t(co)] = c.b[std::size_t(co)];
            for (int ci = 0; ci < c.cin; ++ci)
                for (int t = 0; t < c.k * c.k; ++t)
                    w[std::size_t(co) * d.kp + std::size_t(t) * c.cin + ci] = c.w[(std::size_t(co) * c.cin + ci) * c.k * c.k + t];
        }
        d.w = upload(w);
        d.b = upload(b);
        return d;
    }
    // all kt temporal taps: [coutp][Kp], column ((kt * k + ky) * k + kx) * cin + ci
    DevConv upload3(const Conv2& c) {
        DevConv d;
        d.cin = c.cin; d.cout = c.cout; d.k = c.k; d.kt = c.kt;
        d.coutp = (c.cout + 127) / 128 * 128;
        const int taps = c.kt * c.k * c.k, K = taps * c.cin;
        d.kp = (K + 63) / 64 * 64;
        H16 w(std::size_t(d.coutp) * d.kp, 0), b(std::size_t(d.coutp), 0);
        for (int co = 0; co < c.cout; ++co) {
            b[std::size_t(co)] = c.b[std::size_t(co)];
            for (int ci = 0; ci < c.cin; ++ci)
                for (int t = 0; t < taps; ++t)
                    w[std::size_t(co) * d.kp + std::size_t(t) * c.cin + ci] = c.w3[(std::size_t(co) * c.cin + ci) * taps + t];
        }
        d.w = upload(w);
        d.b = upload(b);
        return d;
    }
};


GpuEncoder::GpuEncoder(const std::string& weights) {
    SafeTensors st(weights);
    w_ = load(st);
    auto dev = std::make_unique<Dev>();
    Dev& d = *dev;
    ck(hipStreamCreate(&d.st), "stream");
    ck(hipEventCreate(&d.e0), "event");
    ck(hipEventCreate(&d.e1), "event");
    d.conv_in = d.upload(w_.conv_in);
    d.conv_out = d.upload(w_.conv_out);
    for (int l = 0; l < 4; ++l) d.down[l] = d.upload(w_.down[l]);
    for (int l = 0; l < 6; ++l)
        for (int b = 0; b < 2; ++b) {
            const Res& r = w_.blocks[l][b];
            d.c1[l][b] = d.upload(r.c1);
            d.c2[l][b] = d.upload(r.c2);
            if (r.has_nin) d.nin[l][b] = d.upload(r.nin);
            d.n1w[l][b] = d.upload(r.n1.w); d.n1b[l][b] = d.upload(r.n1.b);
            d.n2w[l][b] = d.upload(r.n2.w); d.n2b[l][b] = d.upload(r.n2.b);
        }
    d.v_in = d.upload3(w_.conv_in);
    d.v_out = d.upload3(w_.conv_out);
    for (int l = 0; l < 4; ++l) d.v_down[l] = d.upload3(w_.down[l]);
    for (int l = 0; l < 6; ++l)
        for (int b = 0; b < 2; ++b) {
            d.v_c1[l][b] = d.upload3(w_.blocks[l][b].c1);
            d.v_c2[l][b] = d.upload3(w_.blocks[l][b].c2);
            if (w_.blocks[l][b].has_nin) d.v_nin[l][b] = d.upload3(w_.blocks[l][b].nin);
        }
    d.now = d.upload(w_.norm_out.w);
    d.nob = d.upload(w_.norm_out.b);
    // scratch for a 256 x 256 tile: activations <= 65536 x 128 (and 16384 x 256 ...), im2col <= 65536 x 1152
    constexpr std::size_t act = std::size_t(65536) * 128, col = std::size_t(65536) * 1152;
    d.x = d.alloc(act); d.t = d.alloc(act); d.h = d.alloc(act); d.s = d.alloc(act); d.col = d.alloc(col);
    d.pix = d.alloc(std::size_t(65536) * 3);
    d_ = dev.release();
}

GpuEncoder::~GpuEncoder() { delete d_; }

namespace {
// y [Po][coutp] = conv(x [H][W][cin]); residual (optional) [Po][cout] added after the bias rounding.
void run_conv(const DevConv& c, const _Float16* x, int H, int W, int stride, _Float16* col, _Float16* y, const _Float16* res,
              int& Ho, int& Wo, hipStream_t st) {
    const int pad = stride == 1 ? c.k / 2 : 0;
    const int Hp = stride == 1 ? H + 2 * pad : H + 1, Wp = stride == 1 ? W + 2 * pad : W + 1;
    Ho = (Hp - c.k) / stride + 1;
    Wo = (Wp - c.k) / stride + 1;
    const unsigned P = unsigned(Ho * Wo);
    const _Float16* a = x;
    unsigned lda = unsigned(c.cin);
    if (c.k != 1 || stride != 1 || c.kp != c.cin) {
        const std::size_t total = std::size_t(P) * c.kp;
        im2col_kernel<<<unsigned((total + 255U) / 256U), 256, 0, st>>>(x, col, H, W, c.cin, c.k, stride, Ho, Wo, c.kp);
        ck(hipGetLastError(), "im2col");
        a = col;
        lda = unsigned(c.kp);
    }
    if (res) gpu::gemm_add(a, lda, c.w, c.b, res, y, unsigned(c.coutp), P, unsigned(c.coutp), unsigned(c.kp), st);
    else gpu::gemm(a, lda, c.w, c.b, y, unsigned(c.coutp), P, unsigned(c.coutp), unsigned(c.kp), st);
}
}  // namespace

Map GpuEncoder::moments(const Map& pixels, double* gpu_ms) {
    Dev& d = *d_;
    hipStream_t st = d.st;
    check_tile(pixels);
    H16 cl(std::size_t(pixels.H) * pixels.W * 3);
    for (int c = 0; c < 3; ++c)
        for (int p = 0; p < pixels.H * pixels.W; ++p) cl[std::size_t(p) * 3 + c] = pixels.v[std::size_t(c) * pixels.H * pixels.W + p];
    ck(hipMemcpyAsync(d.pix, cl.data(), cl.size() * 2, hipMemcpyHostToDevice, st), "pixels");
    ck(hipEventRecord(d.e0, st), "event");
    int H = pixels.H, W = pixels.W, Ho = 0, Wo = 0;
    run_conv(d.conv_in, d.pix, H, W, 1, d.col, d.x, nullptr, Ho, Wo, st);
    int C = 128;
    for (int l = 0; l < 6; ++l) {
        for (int b = 0; b < 2; ++b) {
            const DevConv& c1 = d.c1[l][b];
            const int P = H * W;
            gn_silu_kernel<<<32, 256, 0, st>>>(d.x, d.n1w[l][b], d.n1b[l][b], d.t, P, C);
            run_conv(c1, d.t, H, W, 1, d.col, d.h, nullptr, Ho, Wo, st);
            const int Cn = c1.cout;
            const _Float16* sc = d.x;
            if (w_.blocks[l][b].has_nin) {
                run_conv(d.nin[l][b], d.x, H, W, 1, d.col, d.s, nullptr, Ho, Wo, st);
                sc = d.s;
            }
            gn_silu_kernel<<<32, 256, 0, st>>>(d.h, d.n2w[l][b], d.n2b[l][b], d.t, P, Cn);
            // conv2 writes a fresh buffer (the shortcut may be x itself)
            run_conv(d.c2[l][b], d.t, H, W, 1, d.col, d.h, sc, Ho, Wo, st);
            std::swap(d.x, d.h);
            C = Cn;
        }
        if (l < 4) {
            run_conv(d.down[l], d.x, H, W, 2, d.col, d.h, nullptr, Ho, Wo, st);
            std::swap(d.x, d.h);
            H = Ho; W = Wo;
        }
    }
    gn_silu_kernel<<<32, 256, 0, st>>>(d.x, d.now, d.nob, d.t, H * W, C);
    run_conv(d.conv_out, d.t, H, W, 1, d.col, d.h, nullptr, Ho, Wo, st);  // [P][128] (48 real)
    ck(hipEventRecord(d.e1, st), "event");
    H16 out(std::size_t(H) * W * d.conv_out.coutp);
    ck(hipMemcpyAsync(out.data(), d.h, out.size() * 2, hipMemcpyDeviceToHost, st), "out");
    ck(hipStreamSynchronize(st), "sync");
    if (gpu_ms) {
        float ms = 0;
        ck(hipEventElapsedTime(&ms, d.e0, d.e1), "elapsed");
        *gpu_ms = ms;
    }
    Map eo;
    eo.C = 48; eo.H = H; eo.W = W;
    eo.v.resize(std::size_t(48) * H * W);
    for (int p = 0; p < H * W; ++p)
        for (int c = 0; c < 48; ++c) eo.v[std::size_t(c) * H * W + p] = out[std::size_t(p) * d.conv_out.coutp + c];
    return quant(w_, eo);
}

std::vector<float> GpuEncoder::encode(const std::vector<float>& x, int H, int W, double* seconds) {
    check_image(H, W, x.size());
    const auto t0 = std::chrono::steady_clock::now();
    const Map px = normalize_pixels(x, H, W);
    const Tiles1D ty = split_tiles(H), tx = split_tiles(W);
    std::vector<std::vector<Map>> tiles(ty.start.size());
    for (std::size_t i = 0; i < ty.start.size(); ++i)
        for (std::size_t j = 0; j < tx.start.size(); ++j) {
            Map t;
            t.C = 3; t.H = ty.len[i]; t.W = tx.len[j];
            t.v.resize(std::size_t(3) * t.H * t.W);
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < t.H; ++y)
                    std::memcpy(t.v.data() + (std::size_t(c) * t.H + y) * t.W,
                                px.v.data() + (std::size_t(c) * H + ty.start[i] + y) * W + tx.start[j], std::size_t(t.W) * 2);
            tiles[i].push_back(moments(t));
        }
    const Map m = blend_tiles(tiles, ty, tx);
    auto z = latent_of(w_, m);
    if (seconds) *seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return z;
}

namespace {
// STRIX_H3_VAE_ENC_IMPLICIT=0: the per-output-frame im2col + GEMM form (same bits); set_implicit_conv() for tests
int implicit_override = -1;
bool implicit_conv() {
    static const bool on = [] { const char* v = std::getenv("STRIX_H3_VAE_ENC_IMPLICIT"); return !v || v[0] != '0'; }();
    return implicit_override < 0 ? on : implicit_override != 0;
}
struct Scratch3 {
    std::vector<void*> p;
    Scratch3() = default;
    Scratch3(const Scratch3&) = delete;
    Scratch3& operator=(const Scratch3&) = delete;
    ~Scratch3() { for (void* q : p) (void)hipFree(q); }
    _Float16* get(std::size_t halves) {
        void* q = nullptr;
        ck(hipMalloc(&q, std::max<std::size_t>(halves * 2, 16)), "hipMalloc");
        p.push_back(q);
        return static_cast<_Float16*>(q);
    }
};
// causal 3-D conv on channels-last frames: one im2col + GEMM per output frame (same K order for every frame);
// res (optional, [To][Po][coutp]) added after the bias rounding
void run_conv3(const DevConv& c, const _Float16* x, int T, int H, int W, int ts, int ss, _Float16* col, _Float16* y, const _Float16* res,
               int& To, int& Ho, int& Wo, hipStream_t st) {
    const int front = c.kt > 1 ? 2 : 0;
    const int pad = ss == 1 ? c.k / 2 : 0;
    const int Hp = ss == 1 ? H + 2 * pad : H + 1, Wp = ss == 1 ? W + 2 * pad : W + 1;
    To = (T + front - c.kt) / ts + 1;
    Ho = (Hp - c.k) / ss + 1;
    Wo = (Wp - c.k) / ss + 1;
    const unsigned P = unsigned(Ho * Wo);
    if (implicit_conv() && c.cin % 32 == 0) {  // one launch over all output frames, same bits as the per-frame form
        gpu::ConvGeom g;
        g.T = T; g.H = H; g.W = W; g.C = c.cin; g.Ho = Ho; g.Wo = Wo; g.k = c.k; g.ts = ts; g.ss = ss; g.front = front;
        g.Kreal = c.kt * c.k * c.k * c.cin;
        gpu::conv_gemm(x, g, c.w, c.b, res, y, unsigned(c.coutp), unsigned(To) * P, unsigned(c.coutp), unsigned(c.kp), st);
        return;
    }
    for (int t = 0; t < To; ++t) {
        const std::size_t total = std::size_t(P) * c.kp;
        im2col3_kernel<<<unsigned((total + 255U) / 256U), 256, 0, st>>>(x, col, T, H, W, c.cin, c.kt, c.k, ts, ss, front, t, Ho, Wo, c.kp);
        ck(hipGetLastError(), "im2col3");
        _Float16* yo = y + std::size_t(t) * P * c.coutp;
        if (res) gpu::gemm_add(col, unsigned(c.kp), c.w, c.b, res + std::size_t(t) * P * c.coutp, yo, unsigned(c.coutp), P, unsigned(c.coutp), unsigned(c.kp), st);
        else gpu::gemm(col, unsigned(c.kp), c.w, c.b, yo, unsigned(c.coutp), P, unsigned(c.coutp), unsigned(c.kp), st);
    }
}
}  // namespace

void set_implicit_conv(int on) { implicit_override = on; }

Vid GpuEncoder::moments_video(const Vid& px, double* gpu_ms) {
    Dev& d = *d_;
    hipStream_t st = d.st;
    if (px.C != 3 || px.T < 1 || px.H < 32 || px.W < 32 || px.H > 256 || px.W > 256 || px.H % 16 || px.W % 16 ||
        px.v.size() != std::size_t(3) * px.T * px.H * px.W)
        throw std::invalid_argument("h3 vae encoder: video tile must be 3 x T x [32, 256] x [32, 256], multiples of 16");
    const std::size_t plane = std::size_t(px.H) * px.W;
    // largest activation: T x H x W x 128 (level 0; deeper levels are smaller); im2col: one output frame
    const std::size_t act = std::size_t(px.T) * plane * 128U;
    std::size_t colmax = 0;
    {
        int hh = px.H, ww = px.W;
        const std::size_t cin_lvl[6] = {128U, 256U, 256U, 512U, 512U, 1024U};
        for (int l = 0; l < 6; ++l) {
            // conv1 of the level's first block reads the previous level's width, conv2 its own: take the max of both
            const std::size_t cmax = std::max(cin_lvl[l], l > 0 ? cin_lvl[l - 1] : 128U);
            colmax = std::max(colmax, std::size_t(hh) * ww * ((27U * cmax + 63U) / 64U * 64U));
            if (l < 4) { hh /= 2; ww /= 2; }
        }
    }
    Scratch3 sc;
    _Float16 *x = sc.get(act), *t = sc.get(act), *h = sc.get(act), *s = sc.get(act), *col = sc.get(colmax);
    _Float16* pix = sc.get(std::size_t(3) * px.T * plane);
    H16 cl(std::size_t(px.T) * plane * 3);
    for (int c = 0; c < 3; ++c)
        for (int f = 0; f < px.T; ++f)
            for (std::size_t q = 0; q < plane; ++q) cl[(std::size_t(f) * plane + q) * 3 + c] = px.v[(std::size_t(c) * px.T + f) * plane + q];
    ck(hipMemcpyAsync(pix, cl.data(), cl.size() * 2, hipMemcpyHostToDevice, st), "pixels");
    ck(hipEventRecord(d.e0, st), "event");
    const auto gn = [&](const _Float16* in, const _Float16* gw, const _Float16* gb, _Float16* out, int T, int P, int C) {
        if (implicit_conv()) gn_silu_kernel<<<dim3(32, unsigned(T)), 256, 0, st>>>(in, gw, gb, out, P, C);
        else
            for (int f = 0; f < T; ++f)
                gn_silu_kernel<<<32, 256, 0, st>>>(in + std::size_t(f) * P * C, gw, gb, out + std::size_t(f) * P * C, P, C);
    };
    int T = px.T, H = px.H, W = px.W, To = 0, Ho = 0, Wo = 0;
    run_conv3(d.v_in, pix, T, H, W, 1, 1, col, x, nullptr, To, Ho, Wo, st);
    int C = 128;
    const int tdown[4] = {1, 2, 2, 1};
    for (int l = 0; l < 6; ++l) {
        for (int b = 0; b < 2; ++b) {
            const int P = H * W;
            gn(x, d.n1w[l][b], d.n1b[l][b], t, T, P, C);
            run_conv3(d.v_c1[l][b], t, T, H, W, 1, 1, col, h, nullptr, To, Ho, Wo, st);
            const int Cn = d.v_c1[l][b].cout;
            const _Float16* scut = x;
            if (w_.blocks[l][b].has_nin) {
                run_conv3(d.v_nin[l][b], x, T, H, W, 1, 1, col, s, nullptr, To, Ho, Wo, st);
                scut = s;
            }
            gn(h, d.n2w[l][b], d.n2b[l][b], t, T, P, Cn);
            run_conv3(d.v_c2[l][b], t, T, H, W, 1, 1, col, h, scut, To, Ho, Wo, st);
            std::swap(x, h);
            C = Cn;
        }
        if (l < 4) {
            run_conv3(d.v_down[l], x, T, H, W, tdown[l], 2, col, h, nullptr, To, Ho, Wo, st);
            std::swap(x, h);
            T = To; H = Ho; W = Wo;
        }
    }
    gn(x, d.now, d.nob, t, T, H * W, C);
    run_conv3(d.v_out, t, T, H, W, 1, 1, col, h, nullptr, To, Ho, Wo, st);  // [T][P][128] (48 real)
    ck(hipEventRecord(d.e1, st), "event");
    H16 out(std::size_t(T) * H * W * d.v_out.coutp);
    ck(hipMemcpyAsync(out.data(), h, out.size() * 2, hipMemcpyDeviceToHost, st), "out");
    ck(hipStreamSynchronize(st), "sync");
    if (gpu_ms) {
        float ms = 0;
        ck(hipEventElapsedTime(&ms, d.e0, d.e1), "elapsed");
        *gpu_ms = ms;
    }
    // quant_conv (1x1x1) per frame on the host, as the single-image path
    Vid mo;
    mo.C = 48; mo.T = T; mo.H = H; mo.W = W;
    mo.v.resize(std::size_t(48) * T * H * W);
    const std::size_t pl = std::size_t(H) * W;
    for (int f = 0; f < T; ++f) {
        Map eo;
        eo.C = 48; eo.H = H; eo.W = W;
        eo.v.resize(std::size_t(48) * pl);
        for (std::size_t q = 0; q < pl; ++q)
            for (int c = 0; c < 48; ++c) eo.v[std::size_t(c) * pl + q] = out[(std::size_t(f) * pl + q) * d.v_out.coutp + c];
        const Map m = quant(w_, eo);
        for (int c = 0; c < 48; ++c) std::memcpy(mo.v.data() + (std::size_t(c) * T + f) * pl, m.v.data() + std::size_t(c) * pl, pl * 2);
    }
    return mo;
}

std::vector<float> GpuEncoder::encode_video(const std::vector<float>& x, int T, int H, int W, double* seconds) {
    check_video(T, H, W, x.size());
    const auto t0 = std::chrono::steady_clock::now();
    const Tiles1D ty = split_tiles(H), tx = split_tiles(W);
    const int clips = (T + 16) / 17;
    std::vector<Vid> cm;
    const std::size_t plane = std::size_t(H) * W;
    for (int c = 0; c < clips; ++c) {
        const Vid px = clip_pixels(x, T, H, W, c);
        std::vector<std::vector<Vid>> tiles(ty.start.size());
        for (std::size_t i = 0; i < ty.start.size(); ++i)
            for (std::size_t j = 0; j < tx.start.size(); ++j) {
                Vid t;
                t.C = 3; t.T = 17; t.H = ty.len[i]; t.W = tx.len[j];
                t.v.resize(std::size_t(3) * 17 * t.H * t.W);
                for (int ch = 0; ch < 3; ++ch)
                    for (int f = 0; f < 17; ++f)
                        for (int y = 0; y < t.H; ++y)
                            std::memcpy(t.v.data() + ((std::size_t(ch) * 17 + f) * t.H + y) * t.W,
                                        px.v.data() + (std::size_t(ch) * 17 + f) * plane + std::size_t(ty.start[i] + y) * W + tx.start[j],
                                        std::size_t(t.W) * 2);
                tiles[i].push_back(moments_video(t));
            }
        cm.push_back(blend_tiles3(tiles, ty, tx));
    }
    auto z = video_latent(w_, cm);
    if (seconds) *seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return z;
}

}  // namespace strix::h3::vae::enc

