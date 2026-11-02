#include "h3/vae/audio_decoder.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

namespace strix::h3::vae::audio {

namespace {
void ck(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("h3 audio decoder: ") + what + ": " + hipGetErrorString(e));
}
// Device allocations owned by a list: released together (decoder destruction, failed construction, scratch).
struct Owned {
    std::vector<void*> ptrs;
    Owned() = default;
    Owned(const Owned&) = delete;
    Owned& operator=(const Owned&) = delete;
    ~Owned() { for (void* p : ptrs) (void)hipFree(p); }
    float* alloc(std::size_t n) {
        void* p = nullptr;
        ck(hipMalloc(&p, n * 4), "hipMalloc");
        ptrs.push_back(p);
        return static_cast<float*>(p);
    }
    float* upload(const std::vector<float>& v) {
        if (v.empty()) return nullptr;
        float* p = alloc(v.size());
        ck(hipMemcpy(p, v.data(), v.size() * 4, hipMemcpyHostToDevice), "upload");
        return p;
    }
};

// Conv1d, stride 1, 'same'-style length (n_out = n + 2 pad - dil (k - 1)), rows on grid z.
// Block: 64 output channels x 64 positions, 256 threads, each 4 channels x 4 positions (positions strided by 16).
// Input channels in chunks of 8 staged in LDS with the k taps; sum order: ci ascending, tap ascending.
// res (optional): y += res (same shape as y), added after the bias.
constexpr unsigned kCo = 64U, kT = 64U, kCi = 8U, kMaxK = 11U, kMaxSpan = kT + (kMaxK - 1U) * 5U;
__global__ __launch_bounds__(256) void conv1d_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                                                     const float* __restrict__ res, float* __restrict__ y, int cin, int cout, int k,
                                                     int dil, int pad, int n, int m) {
    __shared__ float xs[kCi][kMaxSpan];
    __shared__ float ws[kCi * kMaxK][kCo];
    const int tx = int(threadIdx.x % 16U), ty = int(threadIdx.x / 16U);
    const int t0 = int(blockIdx.x * kT), co0 = int(blockIdx.y * kCo), row = int(blockIdx.z);
    const float* xr = x + std::size_t(row) * cin * n;
    const int span = int(kT) + (k - 1) * dil;
    float acc[4][4] = {};
    for (int c0 = 0; c0 < cin; c0 += int(kCi)) {
        __syncthreads();
        for (int i = int(threadIdx.x); i < int(kCi) * span; i += 256) {
            const int ci = i / span, j = i % span, s = t0 - pad + j;
            xs[ci][j] = (c0 + ci < cin && s >= 0 && s < n) ? xr[std::size_t(c0 + ci) * n + s] : 0.0F;
        }
        for (int i = int(threadIdx.x); i < int(kCi) * k * int(kCo); i += 256) {
            const int co = i % int(kCo), q = i / int(kCo), ci = q / k, kk = q % k;
            ws[q][co] = (co0 + co < cout && c0 + ci < cin) ? w[(std::size_t(co0 + co) * cin + c0 + ci) * k + kk] : 0.0F;
        }
        __syncthreads();
        const int nci = min(int(kCi), cin - c0);
        for (int ci = 0; ci < nci; ++ci)
            for (int kk = 0; kk < k; ++kk) {
                float wv[4], xv[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) wv[i] = ws[ci * k + kk][ty * 4 + i];
#pragma unroll
                for (int j = 0; j < 4; ++j) xv[j] = xs[ci][tx + 16 * j + kk * dil];
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[i][j] += wv[i] * xv[j];
            }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int co = co0 + ty * 4 + i;
        if (co >= cout) continue;
        const float bv = b ? b[co] : 0.0F;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int t = t0 + tx + 16 * j;
            if (t >= m) continue;
            const std::size_t o = (std::size_t(row) * cout + co) * m + t;
            float v = acc[i][j] + bv;
            if (res) v = v + res[o];
            y[o] = v;
        }
    }
}

// ConvTranspose1d, weight [cin][cout][k]: y[co][t] = b + sum_ci sum_{k = (t + pad) % s, step s} x[ci][(t + pad - k) / s] w[ci][co][k].
__global__ __launch_bounds__(256) void convt_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                                                    float* __restrict__ y, int cin, int cout, int k, int stride, int pad, int n, int m) {
    const std::size_t i = std::size_t(blockIdx.x) * 256U + threadIdx.x;
    const int row = int(blockIdx.y);
    if (i >= std::size_t(cout) * m) return;
    const int co = int(i / m), t = int(i % m);
    const float* xr = x + std::size_t(row) * cin * n;
    float acc = 0.0F;
    const int k0 = (t + pad) % stride;
    for (int ci = 0; ci < cin; ++ci)
        for (int kk = k0; kk < k; kk += stride) {
            const int s = (t + pad - kk) / stride;
            if (s >= 0 && s < n) acc += xr[std::size_t(ci) * n + s] * w[(std::size_t(ci) * cout + co) * k + kk];
        }
    y[(std::size_t(row) * cout + co) * m + t] = acc + b[co];
}

// Activation1d (x2 up, SnakeBeta, x2 down), one output per thread. Same index math as the host reference.
__global__ __launch_bounds__(256) void act_kernel(const float* __restrict__ x, const float* __restrict__ alpha, const float* __restrict__ beta,
                                                  const float* __restrict__ up, const float* __restrict__ down, float* __restrict__ y,
                                                  int C, int N) {
    const std::size_t i = std::size_t(blockIdx.x) * 256U + threadIdx.x;
    const int row = int(blockIdx.y);
    if (i >= std::size_t(C) * N) return;
    const int c = int(i / N), t = int(i % N);
    const float* xr = x + (std::size_t(row) * C + c) * N;
    const float al = expf(alpha[c]), inv = 1.0F / (expf(beta[c]) + 1e-9F);
    float s = 0.0F;
    for (int kd = 0; kd < 12; ++kd) {
        const int vi = min(max(2 * t + kd - 5, 0), 2 * N - 1);
        // v[vi]: u = 2 * sum_{k = m % 2, step 2} xp[(m - k) / 2] up[k], m = vi + 15
        const int mm = vi + 15;
        float su = 0.0F;
        for (int ku = mm % 2; ku < 12; ku += 2) {
            const int j = (mm - ku) / 2;
            if (j < 0 || j >= N + 10) continue;
            su += xr[min(max(j - 5, 0), N - 1)] * up[ku];
        }
        const float u = su * 2.0F;
        float q = sinf(al * u);
        q = q * q;
        q = q * inv;
        s += (q + u) * down[kd];
    }
    y[(std::size_t(row) * C + c) * N + t] = s;
}

__global__ __launch_bounds__(256) void mean3_kernel(const float* a, const float* b, const float* c, float* y, std::size_t n) {
    const std::size_t i = std::size_t(blockIdx.x) * 256U + threadIdx.x;
    if (i < n) y[i] = ((a[i] + b[i]) + c[i]) / 3.0F;
}

struct DevConv { const Conv* h; float *w, *b; };
struct DevAct { float *alpha, *beta, *up, *down; };
}  // namespace

struct GpuDecoder::Dev {
    Owned mem;
    DevConv dec_in, conv_pre, ups[7], conv_post;
    DevConv c1[21][3], c2[21][3];
    DevAct a[21][6], post;
    hipStream_t st = nullptr;
    hipEvent_t e0 = nullptr, e1 = nullptr;
    ~Dev() {
        (void)hipDeviceSynchronize();
        if (e0) (void)hipEventDestroy(e0);
        if (e1) (void)hipEventDestroy(e1);
        if (st) (void)hipStreamDestroy(st);
    }
};

namespace {
DevConv up_conv(Owned& m, const Conv& c) { return {&c, m.upload(c.w), m.upload(c.b)}; }
DevAct up_act(Owned& m, const Act& a) {
    return {m.upload(a.alpha), m.upload(a.beta), m.upload(std::vector<float>(a.up, a.up + 12)), m.upload(std::vector<float>(a.down, a.down + 12))};
}
void run_conv(const DevConv& c, const float* x, const float* res, float* y, int n, int rows, hipStream_t st) {
    const Conv& h = *c.h;
    const int m = conv_out_len(h, n);
    if (h.transposed) {
        convt_kernel<<<dim3(unsigned((std::size_t(h.cout) * m + 255U) / 256U), unsigned(rows)), 256, 0, st>>>(x, c.w, c.b, y, h.cin, h.cout, h.k,
                                                                                                          h.stride, h.pad, n, m);
    } else {
        if (h.stride != 1 || h.k > int(kMaxK) || h.dil > 5) throw std::invalid_argument("h3 audio conv1d: unsupported shape");
        conv1d_kernel<<<dim3(unsigned((m + int(kT) - 1) / int(kT)), unsigned((h.cout + int(kCo) - 1) / int(kCo)), unsigned(rows)), 256, 0, st>>>(
            x, c.w, c.b, res, y, h.cin, h.cout, h.k, h.dil, h.pad, n, m);
    }
    ck(hipGetLastError(), "conv launch");
}
void run_act(const DevAct& a, const float* x, float* y, int C, int N, int rows, hipStream_t st) {
    act_kernel<<<dim3(unsigned((std::size_t(C) * N + 255U) / 256U), unsigned(rows)), 256, 0, st>>>(x, a.alpha, a.beta, a.up, a.down, y, C, N);
    ck(hipGetLastError(), "act launch");
}
}  // namespace

GpuDecoder::GpuDecoder(const std::string& weights) {
    SafeTensors st(weights);
    w_ = load(st);
    auto dev = std::make_unique<Dev>();  // everything below is released if a step throws
    Dev& d = *dev;
    ck(hipStreamCreate(&d.st), "stream");
    ck(hipEventCreate(&d.e0), "event");
    ck(hipEventCreate(&d.e1), "event");
    d.dec_in = up_conv(d.mem, w_.dec_in);
    d.conv_pre = up_conv(d.mem, w_.conv_pre);
    for (int i = 0; i < 7; ++i) d.ups[i] = up_conv(d.mem, w_.ups[i]);
    d.conv_post = up_conv(d.mem, w_.conv_post);
    for (int r = 0; r < 21; ++r) {
        for (int i = 0; i < 3; ++i) { d.c1[r][i] = up_conv(d.mem, w_.amp[r].c1[i]); d.c2[r][i] = up_conv(d.mem, w_.amp[r].c2[i]); }
        for (int i = 0; i < 6; ++i) d.a[r][i] = up_act(d.mem, w_.amp[r].a[i]);
    }
    d.post = up_act(d.mem, w_.act_post);
    d_ = dev.release();
}

GpuDecoder::~GpuDecoder() { delete d_; }

std::vector<float> GpuDecoder::decode(const std::vector<float>& z, int T, double* gpu_ms, std::vector<std::vector<float>>* taps) {
    check_latent(z, T);
    const int rows = 2, L = 800 * T;
    const std::size_t cap = std::size_t(rows) * std::max<std::size_t>(std::size_t(6400) * T, std::size_t(2048) * T);
    Owned scratch;  // per-call buffers, released on return or exception
    std::vector<float*> buf;
    for (int i = 0; i < 7; ++i) buf.push_back(scratch.alloc(cap));
    float *x = buf[0], *t1 = buf[1], *t2 = buf[2], *r[3] = {buf[3], buf[4], buf[5]}, *cur = buf[6];
    // denormalize on the host: [32][2][T] -> rows [2][32][T]
    std::vector<float> zin(std::size_t(rows) * 32 * T);
    for (int row = 0; row < rows; ++row)
        for (int c = 0; c < 32; ++c)
            for (int t = 0; t < T; ++t)
                zin[(std::size_t(row) * 32 + c) * T + t] = z[(std::size_t(c) * 2 + row) * T + t] * w_.latents_std[std::size_t(c)] + w_.latents_mean[std::size_t(c)];
    hipStream_t st = d_->st;
    ck(hipMemcpyAsync(t1, zin.data(), zin.size() * 4, hipMemcpyHostToDevice, st), "z");
    ck(hipEventRecord(d_->e0, st), "event");
    int N = T, C = 2048;
    run_conv(d_->dec_in, t1, nullptr, t2, N, rows, st);
    run_conv(d_->conv_pre, t2, nullptr, x, N, rows, st);
    C = w_.conv_pre.cout;
    if (taps) taps->clear();
    for (int lvl = 0; lvl < 7; ++lvl) {
        run_conv(d_->ups[lvl], x, nullptr, cur, N, rows, st);
        N = conv_out_len(w_.ups[lvl], N);
        C = w_.ups[lvl].cout;
        for (int j = 0; j < 3; ++j) {
            const int ri = 3 * lvl + j;
            const float* xin = cur;
            float* xo = r[j];
            for (int i = 0; i < 3; ++i) {
                run_act(d_->a[ri][2 * i], xin, t1, C, N, rows, st);
                run_conv(d_->c1[ri][i], t1, nullptr, t2, N, rows, st);
                run_act(d_->a[ri][2 * i + 1], t2, t1, C, N, rows, st);
                run_conv(d_->c2[ri][i], t1, xin, xo, N, rows, st);  // xo = conv + xin (xo != xin: ping-pong below)
                if (i < 2) {
                    // next iteration reads xo; write the following result into x (scratch) and swap
                    xin = xo;
                    xo = xo == r[j] ? x : r[j];
                }
            }
            if (xo != r[j]) ck(hipMemcpyAsync(r[j], xo, std::size_t(rows) * C * N * 4, hipMemcpyDeviceToDevice, st), "copy");
        }
        const std::size_t n = std::size_t(rows) * C * N;
        mean3_kernel<<<unsigned((n + 255U) / 256U), 256, 0, st>>>(r[0], r[1], r[2], x, n);
        ck(hipGetLastError(), "mean3");
        if (taps) {
            std::vector<float> h(n);
            ck(hipMemcpyAsync(h.data(), x, n * 4, hipMemcpyDeviceToHost, st), "tap");
            ck(hipStreamSynchronize(st), "sync");
            taps->push_back(std::move(h));
        }
    }
    run_act(d_->post, x, t1, C, N, rows, st);
    run_conv(d_->conv_post, t1, nullptr, t2, N, rows, st);
    ck(hipEventRecord(d_->e1, st), "event");
    std::vector<float> wave(std::size_t(rows) * L);
    ck(hipMemcpyAsync(wave.data(), t2, wave.size() * 4, hipMemcpyDeviceToHost, st), "wave");
    ck(hipStreamSynchronize(st), "sync");
    for (auto& v : wave) v = std::clamp(v, -1.0F, 1.0F);
    if (gpu_ms) {
        float ms = 0;
        ck(hipEventElapsedTime(&ms, d_->e0, d_->e1), "elapsed");
        *gpu_ms = ms;
    }
    return wave;
}

}  // namespace strix::h3::vae::audio
