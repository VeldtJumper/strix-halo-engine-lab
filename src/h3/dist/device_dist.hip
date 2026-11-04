// MiniMax H3 two-node schedule on the device (see device_dist.hpp).
// The HIP runtime first: gemm/ref.hpp (via the block headers) uses device intrinsics in its host/device helpers.
#include <hip/hip_runtime.h>

#include "h3/dist/device_dist.hpp"

#include <algorithm>
#include <map>
#include <thread>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>

#include "h3/attn/attn.hpp"
#include "h3/block/glue.hpp"
#include "h3/gemm/api.hpp"
#include "h3/dist/accel.hpp"
#include "h3/levers/fbc/fbc.hpp"
#include "h3/levers/lora/turbo.hpp"
#include "h3/levers/sparse/attn_sparse.hpp"
#include "h3/levers/sparse/router.hpp"
#include "strix/dist/communicator.hpp"

namespace strix::h3::dist {

using block::kFfn;
using block::kHidden;
using block::kNormEps;

namespace {

void ck(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("h3 dist hip: ") + what + ": " + hipGetErrorString(e));
}

class Buf {
public:
    Buf() = default;
    explicit Buf(std::size_t bytes) { ck(hipMalloc(&p_, bytes ? bytes : 16), "hipMalloc"); n_ = bytes; }
    ~Buf() { if (p_) static_cast<void>(hipFree(p_)); }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    Buf(Buf&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
    Buf& operator=(Buf&& o) noexcept {
        if (this != &o) { if (p_) static_cast<void>(hipFree(p_)); p_ = o.p_; n_ = o.n_; o.p_ = nullptr; o.n_ = 0; }
        return *this;
    }
    template <class T> T* as() const { return static_cast<T*>(p_); }
    std::size_t bytes() const { return n_; }

private:
    void* p_ = nullptr;
    std::size_t n_ = 0;
};

template <class T> Buf upload(const T* data, std::size_t n) {
    Buf b(n * sizeof(T));
    if (n) ck(hipMemcpy(b.as<void>(), data, n * sizeof(T), hipMemcpyHostToDevice), "upload");
    return b;
}
template <class T> Buf upload(const std::vector<T>& v) { return upload(v.data(), v.size()); }

// Owning set of HIP events (released on every exit path, including a throw after the block loop).
struct Events {
    std::vector<hipEvent_t> v;
    explicit Events(std::size_t n) : v(n, nullptr) {
        for (auto& e : v) ck(hipEventCreate(&e), "event");
    }
    Events() = default;
    Events(const Events&) = delete;
    Events& operator=(const Events&) = delete;
    ~Events() {
        for (auto e : v)
            if (e) static_cast<void>(hipEventDestroy(e));
    }
};

double ms_between(hipEvent_t a, hipEvent_t b) {
    float ms = 0;
    ck(hipEventElapsedTime(&ms, a, b), "event elapsed");
    return ms;
}

// Deterministic pseudo-random bf16 in [-2, 2) * scale (bench inputs only).
__global__ void fill_bf16_kernel(std::uint16_t* p, std::size_t n, std::uint64_t seed, float scale) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    std::uint64_t z = seed + i * 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    const float u = (static_cast<float>(z >> 40) / 16777216.0F) * 4.0F - 2.0F;
    p[i] = static_cast<std::uint16_t>(__float_as_uint(u * scale) >> 16);
}
void fill_bf16(std::uint16_t* p, std::size_t n, std::uint64_t seed, float scale, hipStream_t s) {
    hipLaunchKernelGGL(fill_bf16_kernel, dim3(static_cast<unsigned>((n + 255) / 256)), dim3(256), 0, s, p, n, seed, scale);
    ck(hipGetLastError(), "fill launch");
}

// Order-free residual digest: sum over elements of splitmix64(global index * 2^16 + bits) mod 2^64 (integer adds are
// associative, so the atomic order never changes the value). Same value for the same rows on any rank / launch shape.
__device__ __forceinline__ std::uint64_t mix64(std::uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
__global__ void digest_kernel(const std::uint16_t* x, std::uint64_t n, std::uint64_t index0, unsigned long long* out) {
    std::uint64_t acc = 0;
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += std::uint64_t{gridDim.x} * blockDim.x)
        acc += mix64(((index0 + i) << 16) | x[i]);
    atomicAdd(out, static_cast<unsigned long long>(acc));
}
// Max over (row, head) of ||v||_2 of a token-major [rows][56][128] bf16 buffer (row pitch kInner), fp64 sum rounded
// up to fp32, folded with an unsigned atomic max (non-negative floats order as their bits).
__global__ void max_norm_kernel(const std::uint16_t* v, std::uint64_t vectors, unsigned* out) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= vectors) return;
    const std::uint16_t* p = v + i * kHeadDim;
    double ss = 0;
    for (unsigned d = 0; d < kHeadDim; ++d) {
        const double x = static_cast<double>(__uint_as_float(static_cast<unsigned>(p[d]) << 16));
        ss += x * x;
    }
    const float nrm = __double2float_ru(sqrt(ss));
    atomicMax(out, __float_as_uint(nrm));
}
// Max bf16 magnitude bits (& 0x7fff) of the V part of the fused QKV rows [rows][3 * 7168] (V at column 2 * 7168):
// as unsigned, NaN > Inf > every finite magnitude, so one atomic max keeps a non-finite value visible.
__global__ void max_abs_v_kernel(const std::uint16_t* qkv, std::uint64_t rows, unsigned* out) {
    unsigned m = 0;
    const std::uint64_t n = rows * kInner;
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += std::uint64_t{gridDim.x} * blockDim.x) {
        const std::uint64_t r = i / kInner, c = i % kInner;
        const unsigned b = qkv[r * (3U * kInner) + 2U * kInner + c] & 0x7fffU;
        m = b > m ? b : m;
    }
    atomicMax(out, m);
}
// x[i] = x[i] + d[i] (fp32, the adaLN E-grid injection's plain add)
__global__ void add_f32_kernel(float* x, const float* d, std::uint64_t n) {
#pragma clang fp contract(off)
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += std::uint64_t{gridDim.x} * blockDim.x)
        x[i] = x[i] + d[i];
}
void launch_add_f32(float* x, const float* d, std::uint64_t n, hipStream_t s) {
    if (!n) return;
    hipLaunchKernelGGL(add_f32_kernel, dim3(256), dim3(256), 0, s, x, d, n);
    ck(hipGetLastError(), "add launch");
}
void launch_max_abs_v(const std::uint16_t* qkv, std::uint64_t rows, unsigned* out, hipStream_t s) {
    if (!rows) return;
    hipLaunchKernelGGL(max_abs_v_kernel, dim3(512), dim3(256), 0, s, qkv, rows, out);
    ck(hipGetLastError(), "v scan launch");
}
void launch_max_norm(const std::uint16_t* v, std::uint64_t vectors, unsigned* out, hipStream_t s) {
    if (!vectors) return;
    hipLaunchKernelGGL(max_norm_kernel, dim3(static_cast<unsigned>((vectors + 255) / 256)), dim3(256), 0, s, v, vectors, out);
    ck(hipGetLastError(), "norm launch");
}
// Order-free digest of LUT lists: sum over lists l (global list id = list0 + l, x 2 + key block) and positions e of
// mix64(mix64(id) + e * golden + entry): a rank's lists add to the other rank's into the 1-node value.
__global__ void lut_digest_kernel(const std::int32_t* entries, const std::uint32_t* offsets, std::uint32_t lists, std::uint64_t list0, unsigned kb,
    unsigned long long* out) {
    std::uint64_t acc = 0;
    for (std::uint32_t l = blockIdx.x * blockDim.x + threadIdx.x; l < lists; l += gridDim.x * blockDim.x) {
        const std::uint64_t id = mix64((list0 + l) * 2U + kb);
        for (std::uint32_t e = offsets[l]; e < offsets[l + 1U]; ++e)
            acc += mix64(id + std::uint64_t{e - offsets[l]} * 0x9e3779b97f4a7c15ULL + static_cast<std::uint32_t>(entries[e]));
    }
    atomicAdd(out, static_cast<unsigned long long>(acc));
}
void launch_lut_digest(const std::int32_t* entries, const std::uint32_t* offsets, std::uint32_t lists, std::uint64_t list0, unsigned kb,
    unsigned long long* out, hipStream_t s) {
    if (!lists) return;
    hipLaunchKernelGGL(lut_digest_kernel, dim3(64), dim3(256), 0, s, entries, offsets, lists, list0, kb, out);
    ck(hipGetLastError(), "lut digest launch");
}
void launch_digest(const std::uint16_t* x, std::uint64_t n, std::uint64_t index0, unsigned long long* out, hipStream_t s) {
    if (!n) return;
    hipLaunchKernelGGL(digest_kernel, dim3(512), dim3(256), 0, s, x, n, index0, out);
    ck(hipGetLastError(), "digest launch");
}

}  // namespace

// Host twin of the device digest (tests).
bool device_synchronize_ok() noexcept { return hipDeviceSynchronize() == hipSuccess; }

std::uint64_t residual_digest(const std::uint16_t* x, std::uint64_t n, std::uint64_t index0) {
    auto mix = [](std::uint64_t z) {
        z += 0x9e3779b97f4a7c15ULL;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    };
    std::uint64_t acc = 0;
    for (std::uint64_t i = 0; i < n; ++i) acc += mix(((index0 + i) << 16) | x[i]);
    return acc;
}

// --------------------------------------------------------------------------------------------- resident weights
struct DeviceModel::Impl {
    struct Block {
        Buf qkv_c, qkv_s, out_c, out_s, fc1_c, fc1_s, fc2_c, fc2_s, norm1, norm2, qn, kn, ada_w, ada_b;
        std::uint32_t ada_n = 0;
        gemm::Weights qkv, out, fc1, fc2;
    };
    std::vector<Block> blocks;
    std::uint32_t basis = 0;
    std::uint64_t bytes = 0;
    const block::GlobalWeights* g = nullptr;
    double weight_bound = 0;  // max over blocks of the weight-only operand bound (loose; info)
    bool fc1_plain = false;   // fc1 in checkpoint order (unfused fc1 + K5 SwiGLU; needed by Turbo)
    Buf final_norm, final_ada_w, final_ada_b, vout_w, vout_b, aout_w, aout_b, vproj_w, vproj_b, aproj_w, aproj_b;
};

namespace {
// Attention admission (operand contract of the attention package): after the per-head RMSNorm with weight w,
// ||y|| <= sqrt(128) * max|w|; the split-half RoPE with bf16 (c, s) scales a pair by at most sqrt(c^2 + s^2) <= 1 + 2^-7
// (bf16 rounding of a unit vector), plus one bf16 rounding of each output (1 + 2^-8). Bound per block:
// ||q|| ||k|| log2(e) / sqrt(128) <= 2^14.
double admission_bound(const block::BlockWeights& b) {
    auto mx = [](const std::vector<std::uint16_t>& w) {
        double m = 0;
        for (auto v : w) m = std::max(m, static_cast<double>(std::fabs(block::bf(v))));
        return m;
    };
    const double grow = (1.0 + 1.0 / 128.0) * (1.0 + 1.0 / 256.0);
    const double nq = std::sqrt(128.0) * mx(b.q_norm) * grow, nk = std::sqrt(128.0) * mx(b.k_norm) * grow;
    return nq * nk * 1.4426950408889634 / std::sqrt(128.0);
}
}  // namespace

DeviceModel::DeviceModel(const block::GlobalWeights& g, block::BlockSource& src, std::uint32_t nblocks, bool fc1_plain)
    : impl_(std::make_unique<Impl>()) {
    const std::uint32_t basis = g.basis;
    impl_->fc1_plain = fc1_plain;
    impl_->basis = basis;
    impl_->g = &g;
    impl_->final_norm = upload(g.final_norm);
    impl_->final_ada_w = upload(g.final_adaln.w);
    impl_->final_ada_b = upload(g.final_adaln.b);
    impl_->vout_w = upload(g.video_out.w);
    impl_->vout_b = upload(g.video_out.b);
    impl_->aout_w = upload(g.audio_out.w);
    impl_->aout_b = upload(g.audio_out.b);
    impl_->vproj_w = upload(g.video_patch_proj.w);
    impl_->vproj_b = upload(g.video_patch_proj.b);
    impl_->aproj_w = upload(g.audio_patch_proj.w);
    impl_->aproj_b = upload(g.audio_patch_proj.b);
    impl_->blocks.resize(nblocks);
    for (std::uint32_t i = 0; i < nblocks; ++i) {
        const auto b = src.block(i);
        auto& d = impl_->blocks[i];
        auto q = [&](const block::QLinear& l, Buf& c, Buf& sc, gemm::Weights& w) {
            c = upload(l.w);
            sc = upload(l.ws);
            w = gemm::Weights{c.as<std::int8_t>(), sc.as<float>(), l.n, l.k};
            impl_->bytes += c.bytes() + sc.bytes();
        };
        q(b->qkv, d.qkv_c, d.qkv_s, d.qkv);
        q(b->out, d.out_c, d.out_s, d.out);
        if (fc1_plain) {
            q(b->fc1, d.fc1_c, d.fc1_s, d.fc1);
        } else {
            const gemm::SwiGLUWeights p = gemm::permute_fc1_for_swiglu(b->fc1.w.data(), b->fc1.ws.data(), b->fc1.n, b->fc1.k);
            d.fc1_c = upload(p.codes);
            d.fc1_s = upload(p.scales);
            d.fc1 = gemm::Weights{d.fc1_c.as<std::int8_t>(), d.fc1_s.as<float>(), b->fc1.n, b->fc1.k};
            impl_->bytes += d.fc1_c.bytes() + d.fc1_s.bytes();
        }
        q(b->fc2, d.fc2_c, d.fc2_s, d.fc2);
        d.norm1 = upload(b->norm1);
        d.norm2 = upload(b->norm2);
        d.qn = upload(b->q_norm);
        d.kn = upload(b->k_norm);
        d.ada_w = upload(b->adaln.w);
        d.ada_b = upload(b->adaln.b);
        d.ada_n = b->adaln.n;
        if (b->adaln.k != basis) throw std::runtime_error("h3 dist: adaLN basis mismatch");
        impl_->weight_bound = std::max(impl_->weight_bound, admission_bound(*b));  // worst case from the weights (info)
        impl_->bytes += d.ada_w.bytes();
    }
}
DeviceModel::~DeviceModel() = default;
std::uint32_t DeviceModel::num_blocks() const { return static_cast<std::uint32_t>(impl_->blocks.size()); }
std::uint64_t DeviceModel::resident_bytes() const { return impl_->bytes; }
double DeviceModel::weight_bound() const { return impl_->weight_bound; }
bool DeviceModel::fc1_plain() const { return impl_->fc1_plain; }

// --------------------------------------------------------------------------------------------- accelerators
namespace lora = strix::h3::levers::lora;
namespace fbcl = strix::h3::levers::fbc;
namespace spz = strix::h3::levers::sparse;

struct DistAccel::Impl {
    const block::TurboV4* T = nullptr;
    bool fbc_enabled = false, observe = false;
    double thr = 0.08;
    FbcForce force = FbcForce::none;
    StepInfo last;
    std::uint32_t rows = 0;
    std::uint32_t run_id = 0, call_index = 0;  // coordinated across the ranks (asserted at every decision exchange)
    bool dead = false, inject_fbc_fail = false;
    std::unique_ptr<fbcl::FirstBlockCacheDevice> fbc;  // bound to the stream of its rank (first begin)
    block::TurboV4::AdalnDeltas deltas;                // of the current forward
    // sparse attention
    bool sparse_on = false;
    spz::Params spp;
    std::uint32_t sp_call = 0, sp_P = 0;
    std::size_t sp_bytes_kc = 0, sp_bytes_q = 0, sp_blocks = 0;
    Buf kc, qbar, thrb, st, mv, lut_err, dummy, lutd;
    void bind_sparse(std::uint32_t S, std::uint32_t r, std::uint32_t nb) {
        const std::size_t KT = (S + 63U) / 64U, QB = (r + 63U) / 64U + 1U;
        if (KT * kInner * 2U > sp_bytes_kc) {
            kc = Buf(KT * kInner * 2U);
            sp_bytes_kc = KT * kInner * 2U;
        }
        if (QB * kInner * 4U > sp_bytes_q) {
            qbar = Buf(QB * kInner * 4U);
            thrb = Buf(QB * kHeads * 4U);
            sp_bytes_q = QB * kInner * 4U;
        }
        if (!st.bytes()) {
            st = Buf(4U * kInner * 4U);
            mv = Buf(2U * kInner * 4U);
            lut_err = Buf(4U);
            dummy = Buf(16U);
        }
        if (nb > sp_blocks) {
            lutd = Buf(std::size_t{nb} * 2U * 8U);
            sp_blocks = nb;
        }
    }
    bool turbo_on() const { return T != nullptr; }
    bool fbc_on() const { return fbc_enabled; }
    void bind(std::uint32_t r) {
        if (rows && r != rows) throw std::invalid_argument("h3 dist accel: one accelerator per rank object (row count differs)");
        rows = r;
        if (fbc_enabled && !fbc) fbc = std::make_unique<fbcl::FirstBlockCacheDevice>(r, kHidden);
    }
};

DistAccel::DistAccel(const block::TurboV4* turbo, bool fbc, double fbc_threshold, bool fbc_observe) : impl_(std::make_unique<Impl>()) {
    impl_->T = turbo;
    impl_->fbc_enabled = fbc;
    impl_->thr = fbc_threshold;
    impl_->observe = fbc_observe;
    if (fbc) fbcl::check_threshold(fbc_threshold);
}
DistAccel::~DistAccel() = default;
void DistAccel::set_fbc_force(FbcForce f) { impl_->force = f; }
void DistAccel::reset_fbc() {
    if (impl_->dead) throw std::logic_error("h3 dist accel: failed earlier (pair-wide abort): rebuild the pair and the accelerators");
    if (impl_->fbc) impl_->fbc->reset();
    ++impl_->run_id;
    impl_->call_index = 0;
    impl_->sp_call = 0;
}
void DistAccel::set_sparse(const levers::sparse::Params& p) {
    if (impl_->rows) throw std::logic_error("h3 dist accel: set_sparse before the first forward");
    if (!(p.tau == p.tau)) throw std::invalid_argument("h3 dist sparse: tau NaN");
    impl_->sparse_on = true;
    impl_->spp = p;
}
void DistAccel::inject_fbc_failure_for_test() { impl_->inject_fbc_fail = true; }
bool DistAccel::failed() const { return impl_->dead; }
const DistAccel::StepInfo& DistAccel::last() const { return impl_->last; }
DistAccel::Impl& DistAccel::impl() { return *impl_; }

// --------------------------------------------------------------------------------------------- one rank
struct DeviceDistRank::Impl {
    const DeviceModel::Impl& M;
    Split sp;
    Role role;
    strix::dist::Communicator* comm;
    hipStream_t s = nullptr;
    std::uint32_t r0 = 0, n = 0;
    std::uint64_t pitch = 0;  // V^T pitch of the K/V storage (message pitch, or the 1-node buffer's)
    Buf x, h, qkv, q, ao, o, sw, po, pml, tx[2], rx[2], kloc, vtloc, bound{16};
    gemm::Workspace ws;
    hipEvent_t source_free[2] = {}, last_use[2] = {};
    bool src_pending[2] = {false, false};
    bool in_flight = false;  // a started exchange not yet finished: teardown must cancel the transport first
    bool injected = false;   // a test hook fired (DeviceRunOptions::inject)
    DistAccel* acc = nullptr;  // FAST-class accelerators of the current forward (nullptr = dense)
    std::int64_t last_started = -1;
    std::uint64_t request = 0;
    Buf digests;             // [nblocks][2] unsigned long long (device digests of the owned key blocks)
    Buf norms;               // [nblocks][3] unsigned: max ||q||, max ||k|| (fp32 bits), max |V| (bf16 bits) of the rank's rows

    Impl(const DeviceModel::Impl& m, const Split& split, const Role& r, strix::dist::Communicator* c)
        : M(m), sp(split), role(r), comm(c), ws(r.rows(split), kFfn) {
        if (role.world == 2U && !comm) throw std::invalid_argument("h3 dist: two ranks need a communicator");
        ck(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), "stream");
        r0 = role.row0(sp);
        n = role.rows(sp);
        const std::size_t N = n;
        x = Buf(N * kHidden * 2);
        h = Buf(N * kHidden * 2);
        qkv = Buf(N * 3 * kInner * 2);
        q = Buf(N * kInner * 2);
        ao = Buf(N * kInner * 2);
        o = Buf(N * kHidden * 2);
        sw = Buf(N * kFfn * 2);
        po = Buf(2 * N * kHeads * kHeadDim * 4);
        pml = Buf(2 * N * kHeads * 2 * 4);
        if (role.world == 2U) {
            pitch = sp.vt_pitch();
            for (int j = 0; j < 2; ++j) {
                tx[j] = Buf(sp.message_bytes());
                rx[j] = Buf(sp.message_bytes());
                ck(hipMemsetAsync(tx[j].as<void>(), 0, sp.message_bytes(), s), "tx clear");  // pads are sent: define them
                ck(hipEventCreateWithFlags(&source_free[j], hipEventDisableTiming), "event");
                ck(hipEventCreateWithFlags(&last_use[j], hipEventDisableTiming), "event");
                ck(hipEventRecord(last_use[j], s), "initial last use");  // nothing reads RX yet
            }
        } else {
            pitch = (std::uint64_t{sp.S} + kCutAlign - 1U) / kCutAlign * kCutAlign;
            kloc = Buf(std::size_t{sp.S} * kInner * 2);
            vtloc = Buf(std::size_t{kInner} * pitch * 2);
        }
        ck(hipStreamSynchronize(s), "init");
    }
    ~Impl() {
        // Never release TX / RX / events while transport workers may still submit copies that use them.
        if (in_flight && comm) comm->cancel_pending("h3 dist: rank torn down with an exchange in flight");
        if (s) static_cast<void>(hipStreamSynchronize(s));
        for (int j = 0; j < 2; ++j) {
            if (source_free[j]) static_cast<void>(hipEventDestroy(source_free[j]));
            if (last_use[j]) static_cast<void>(hipEventDestroy(last_use[j]));
        }
        if (s) static_cast<void>(hipStreamDestroy(s));
    }
    attn::Queries queries() const { return attn::Queries{q.as<std::uint16_t>(), kHeadDim, kInner, n, kHeads, kHeadDim}; }
    attn::Partials partials() const { return attn::Partials{po.as<float>(), pml.as<float>(), n, kHeads, kHeadDim, bound.as<unsigned>()}; }
    attn::KeyBlock key_block(const std::uint16_t* k, const std::uint16_t* vt, std::uint32_t keys) const {
        return attn::KeyBlock{k, vt, kHeadDim, kInner, kHeadDim * pitch, pitch, keys};
    }
    const std::uint16_t* msg_k(const Buf& m) const { return m.as<std::uint16_t>(); }
    const std::uint16_t* msg_vt(const Buf& m) const { return m.as<std::uint16_t>() + sp.k_bytes() / 2U; }
    strix::dist::Tag tag(std::uint32_t layer) { return strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::raw_slots), layer, request++}; }
    // TX[j] may be overwritten only after the transport's D2H of its previous message.
    void wait_tx(int j) {
        if (src_pending[j]) {
            ck(hipStreamWaitEvent(s, source_free[j], 0), "tx source lifetime");
            src_pending[j] = false;
        }
    }
    void start(int j, std::uint32_t layer) {
        in_flight = true;
        last_started = layer;
        src_pending[j] = comm->start_async(s, tx[j].as<void>(), rx[j].as<void>(), sp.message_bytes(), tag(layer), last_use[j], source_free[j]);
    }
    void finish(int j) {
        comm->finish(s, rx[j].as<void>());
        in_flight = false;
    }
    // Any failure with an exchange possibly in flight: stop the transport's submission, then drain the GPU.
    void fail(const char* why) noexcept {
        if (comm) comm->cancel_pending(why);
        in_flight = false;
        if (s) static_cast<void>(hipStreamSynchronize(s));
    }
};

DeviceDistRank::DeviceDistRank(const DeviceModel& model, const Split& sp, const Role& role, strix::dist::Communicator* comm)
    : impl_(std::make_unique<Impl>(model.impl(), sp, role, comm)) {
    if (comm) comm->enable_async_start();
}
DeviceDistRank::~DeviceDistRank() = default;
bool DeviceDistRank::injection_fired() const { return impl_->injected; }
std::int64_t DeviceDistRank::last_started_block() const { return impl_->last_started; }

DeviceRunResult DeviceDistRank::run(const ForwardTables& t, std::uint32_t basis, const std::vector<std::uint16_t>& x_in, std::uint32_t nblocks,
    const DeviceRunOptions& opt) {
    if (x_in.size() != std::size_t{impl_->n} * kHidden) throw std::invalid_argument("h3 dist: residual rows");
    try {
        ck(hipMemcpyAsync(impl_->x.as<void>(), x_in.data(), x_in.size() * 2, hipMemcpyHostToDevice, impl_->s), "x upload");
        return run_blocks(t, basis, nblocks, opt, true);
    } catch (const std::exception& e) {
        impl_->fail(e.what());  // e.g. a contract violation after the loop: the transport stops too
        throw;
    }
}

// The block loop on the residual already in I.x (the rank's rows).
DeviceRunResult DeviceDistRank::run_blocks(const ForwardTables& t, std::uint32_t basis, std::uint32_t nblocks, const DeviceRunOptions& opt,
    bool download_x) {
    Impl& I = *impl_;
    const DeviceModel::Impl& M = I.M;
    const Split& sp = I.sp;
    const Role& role = I.role;
    hipStream_t s = I.s;
    void* sv = static_cast<void*>(s);
    const std::uint32_t n = I.n, r0 = I.r0;
    if (nblocks > M.blocks.size()) throw std::invalid_argument("h3 dist: more blocks than resident");
    if (basis != M.basis || t.t_emb.size() != std::size_t{t.n_t} * basis) throw std::invalid_argument("h3 dist: t_emb");
    using Inject = DeviceRunOptions::Inject;
    if (opt.inject != Inject::none) {
        if (opt.inject_block < 0 || static_cast<std::uint32_t>(opt.inject_block) >= nblocks)
            throw std::invalid_argument("h3 dist: inject_block outside [0, nblocks)");
        if (opt.inject == Inject::throw_in_flight && role.world != 2U) throw std::invalid_argument("h3 dist: throw_in_flight needs two ranks");
    }
    I.injected = false;
    I.last_started = -1;
    DistAccel::Impl* A = I.acc ? &I.acc->impl() : nullptr;
    const bool turbo = A && A->turbo_on(), fbc_on = A && A->fbc_on(), unfused = M.fc1_plain;
    if (turbo && !unfused) throw std::invalid_argument("h3 dist: Turbo needs a DeviceModel built with fc1_plain (unfused fc1 in checkpoint order)");
    if (A) {
        if (fbc_on && opt.timing) throw std::invalid_argument("h3 dist: per-block timing with FirstBlockCache is not supported");
        A->bind(n);
        A->last.fbc_action = -1;
        A->last.fbc_cold = false;
        A->last.fbc_score = 0.0F;
        A->last.fbc_would_reuse = false;
        A->last.blocks_run = 0;
        A->last.sparse = false;
        A->last.sparse_layers = 0;
        A->last.lut_digest.clear();
        A->last.lut_entries.clear();
    }
    // sparse attention geometry of this forward (identity h3-attn-sparse-thr-v1)
    constexpr std::uint64_t kSW = kInner;  // row pitch of Q / K (56 x 128)
    spz::Geometry geo{};
    const bool sparse_fw = A && A->sparse_on && !(geo = spz::geometry(sp.S, A->sp_P)).dense;
    std::uint32_t qi0 = 0, qi1 = 0, qt0 = 0, qt1 = 0;
    std::vector<std::uint64_t> lut_entries;
    std::vector<spz::DeviceLut> lut_keep;  // alive until the forward's final sync
    if (sparse_fw) {
        if (geo.cut != sp.cut) throw std::invalid_argument("h3 dist sparse: the rank split must be Split::sparse(S) (cut 128 floor(S / 256))");
        A->bind_sparse(sp.S, n, nblocks);
        ck(hipMemsetAsync(A->lut_err.as<void>(), 0, 4, s), "lut errors reset");
        ck(hipMemsetAsync(A->lutd.as<void>(), 0, std::size_t{nblocks} * 2U * 8U, s), "lut digest reset");
        qi0 = r0 / 64U;
        qi1 = (r0 + n + 63U) / 64U;
        qt0 = r0 / 128U;
        qt1 = (r0 + n + 127U) / 128U;
        lut_entries.assign(nblocks, 0);
        A->last.sparse = true;
        A->last.sparse_call = A->sp_call;
    }
    auto ktiles = [&](unsigned b, std::uint32_t& j0, std::uint32_t& j1) {
        j0 = sp.first(b) / 64U;
        j1 = (sp.first(b) + sp.rows(b) + 63U) / 64U;
    };
    DeviceRunResult res;
    // per-forward tables (all S rows on both ranks; the rank reads its own rows)
    Buf d_temb = upload(t.t_emb), d_rc = upload(t.rope.c), d_rs = upload(t.rope.s), d_mrow = upload(t.mod_row);
    const std::uint32_t nt = t.n_t, mrows = nt * 3U;
    Buf d_ada(std::size_t{nt} * 6U * kHidden * 3U * 4U);
    Buf d_mod[6];
    std::uint16_t* mod[6];
    for (int j = 0; j < 6; ++j) {
        d_mod[j] = Buf(std::size_t{mrows} * kHidden * 2);
        mod[j] = d_mod[j].as<std::uint16_t>();
    }
    Buf d_delta, d_f1;
    const bool turbo_ada = turbo && A->T->has_adaln(), turbo_fc2 = turbo && A->T->has_fc2();  // Turbo v4 / step LoRAs
    if (turbo_ada) {
        if (A->deltas.nt != nt || A->deltas.blocks.size() < std::size_t{nblocks} * nt * A->deltas.block_out) throw std::logic_error("h3 dist turbo: adaLN deltas");
        d_delta = upload(A->deltas.blocks);
    }
    if (unfused) d_f1 = Buf(std::size_t{n} * 2U * kFfn * 2U);
    I.ws.reset_flush_count(s);
    ck(hipMemsetAsync(I.bound.as<void>(), 0, 4, s), "bound counter reset");
    ck(hipStreamSynchronize(s), "setup");
    std::uint16_t* x = I.x.as<std::uint16_t>();
    std::uint16_t* h = I.h.as<std::uint16_t>();
    std::uint16_t* o = I.o.as<std::uint16_t>();
    std::uint16_t* qkv = I.qkv.as<std::uint16_t>();
    std::uint16_t* ao = I.ao.as<std::uint16_t>();
    std::uint16_t* sw = I.sw.as<std::uint16_t>();
    const std::uint32_t* mr = d_mrow.as<std::uint32_t>() + r0;
    const std::uint16_t* rc = d_rc.as<std::uint16_t>() + std::size_t{r0} * 48U;
    const std::uint16_t* rs = d_rs.as<std::uint16_t>() + std::size_t{r0} * 48U;
    const attn::Queries Q = I.queries();
    const attn::Partials P = I.partials();
    // events: 0 block start, 1 K4 done, 2 local partial done, 3 RX ready, 4 attention done
    constexpr int kEv = 5;
    Events events(opt.timing ? std::size_t{nblocks} * kEv + 1 : 0U);
    std::vector<hipEvent_t>& ev = events.v;
    auto rec = [&](std::uint32_t i, int k) {
        if (opt.timing) ck(hipEventRecord(ev[std::size_t{i} * kEv + k], s), "event record");
    };
    res.hash.assign(nblocks, {0, 0});
    res.digest.assign(nblocks, {0, 0});
    res.timing.assign(nblocks, {});
    I.digests = Buf(std::size_t{nblocks} * 2U * 8U);
    ck(hipMemsetAsync(I.digests.as<void>(), 0, std::size_t{nblocks} * 2U * 8U, s), "digest reset");
    I.norms = Buf(std::size_t{nblocks} * kAdmitPerBlock * 4U);
    ck(hipMemsetAsync(I.norms.as<void>(), 0, std::size_t{nblocks} * kAdmitPerBlock * 4U, s), "norm reset");
    std::vector<std::uint16_t> host_x;
    const auto w0 = std::chrono::steady_clock::now();
    // per-kernel-group profile: consecutive events on the stream; the time from one mark to the next goes to the group
    enum Pk { kpAdaln, kpNorm, kpQkv, kpTurbo, kpRope, kpTranspose, kpSparse, kpScans, kpAttnLocal, kpExchWait, kpAttnRemote, kpOut, kpGlue,
        kpFc1, kpSwiglu, kpFc2, kpDigest, kpFbc, kpCount };
    static const char* const kPkNames[kpCount] = {"adaLN (linear + mod vectors + Turbo delta)", "rmsnorm + modulation (K3)", "int8 GEMM qkv (+quantize)",
        "Turbo LoRA branches", "q/k norm + RoPE (K4)", "transpose V", "sparse router (kc, stats, message, thresholds)", "admission scans",
        "attention local partial (+ local LUT)", "K/V exchange wait (exposed)", "attention remote partial + merge (+ remote kc / LUT)",
        "int8 GEMM out (+quantize)", "gates / residual", "int8 GEMM fc1 (+quantize)", "SwiGLU (K5, unfused)", "int8 GEMM fc2 (+quantize)",
        "digests / hashes", "FirstBlockCache"};
    Events pev(opt.profile ? std::size_t{nblocks} * 28U + 2U : 0U);
    std::vector<int> pcat;
    std::size_t pn = 0;
    auto pm = [&](int cat) {
        if (!opt.profile) return;
        if (pn >= pev.v.size()) throw std::logic_error("h3 dist profile: event budget");
        ck(hipEventRecord(pev.v[pn++], s), "profile event");
        pcat.push_back(cat);
    };
    bool reused = false, commit = false;
    try {
        if (fbc_on) A->fbc->begin(x, n, s);
        pm(-1);
        for (std::uint32_t i = 0; i < nblocks; ++i) {
            const auto& W = M.blocks[i];
            const int par = static_cast<int>(i & 1U);
            rec(i, 0);
            block::gpu::launch_linear_f64(d_temb.as<float>(), basis, nt, W.ada_w.as<float>(), W.ada_b.as<float>(), W.ada_n, basis,
                d_ada.as<float>(), sv);
            if (turbo_ada) {
                if (W.ada_n != A->deltas.block_out) throw std::runtime_error("h3 dist turbo: adaLN width");
                launch_add_f32(d_ada.as<float>(), d_delta.as<float>() + std::size_t{i} * nt * W.ada_n, std::uint64_t{nt} * W.ada_n, s);
            }
            block::gpu::launch_mod_vectors(d_ada.as<float>(), mrows, mod, sv);
            pm(kpAdaln);
            // attention half: K3, QKV (local rows)
            block::gpu::launch_rmsnorm_mod(x, W.norm1.as<std::uint16_t>(), mr, mod[1], mod[0], n, kHidden, kNormEps, h, sv);
            pm(kpNorm);
            gemm::linear(h, kHidden, n, W.qkv, I.ws, qkv, 3U * kInner, s);
            pm(kpQkv);
            if (turbo) A->T->device().apply(lora::Site::kQkv, i, h, kHidden, n, qkv, 3U * kInner, s);
            if (turbo) pm(kpTurbo);
            // K4: Q, K, V^T (K and V^T straight into the TX message / the 1-node buffer)
            std::uint16_t* kdst;
            std::uint16_t* vtdst;
            if (role.world == 2U) {
                I.wait_tx(par);
                kdst = I.tx[par].as<std::uint16_t>();
                vtdst = kdst + sp.k_bytes() / 2U;
            } else {
                kdst = I.kloc.as<std::uint16_t>();
                vtdst = I.vtloc.as<std::uint16_t>();
            }
            block::gpu::launch_qk_norm_rope(qkv, n, W.qn.as<std::uint16_t>(), W.kn.as<std::uint16_t>(), rc, rs, kNormEps,
                I.q.as<std::uint16_t>(), kdst, sv);
            pm(kpRope);
            attn::launch_transpose_v(qkv + 2U * kInner, kHeadDim, 3U * kInner, n, kHeads, vtdst, kHeadDim * I.pitch, I.pitch, sv);
            pm(kpTranspose);
            rec(i, 1);
            const bool hook = opt.inject != Inject::none && static_cast<std::int64_t>(i) == opt.inject_block;
            const bool sl = sparse_fw && spz::sparse_call(A->spp, geo, A->sp_call, i);
            std::uint16_t* kcb = sl ? A->kc.as<std::uint16_t>() : nullptr;
            float* qbar = sl ? A->qbar.as<float>() : nullptr;
            float* thr = sl ? A->thrb.as<float>() : nullptr;
            if (sl) {
                // own key block(s): kc + (S1, S2); local query centroids; the stats message BEFORE the K/V transfer
                float* st = A->st.as<float>();
                for (unsigned b = 0; b < 2; ++b) {
                    if (role.world == 2U && b != role.block()) continue;
                    std::uint32_t j0, j1;
                    ktiles(b, j0, j1);
                    const std::uint16_t* kb = role.world == 2U ? kdst : kdst + std::size_t{sp.first(b)} * kSW;
                    spz::launch_key_centroids(kb, kSW, geo, j0, j1, kcb + std::size_t{j0} * kSW, s);
                    spz::launch_stats_partial(kcb + std::size_t{j0} * kSW, j1 - j0, st + std::size_t{b} * 2U * kSW, st + std::size_t{b} * 2U * kSW + kSW, s);
                }
                spz::launch_query_centroids(I.q.as<std::uint16_t>(), kSW, geo, qi0, qi1, qbar, s);
                if (role.world == 2U) {
                    const unsigned own = role.block(), peer = role.peer_block();
                    std::vector<std::uint32_t> mine(2U * kSW + 2U), theirs(mine.size(), 0U);
                    ck(hipMemcpyAsync(mine.data(), st + std::size_t{own} * 2U * kSW, 2U * kSW * 4U, hipMemcpyDeviceToHost, s), "stats d2h");
                    ck(hipStreamSynchronize(s), "stats");
                    mine[2U * kSW] = A->sp_call;
                    mine[2U * kSW + 1U] = i;
                    I.comm->exchange_host(std::as_bytes(std::span<const std::uint32_t>(mine)), std::as_writable_bytes(std::span<std::uint32_t>(theirs)),
                        strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 0xAAU, I.request++});
                    if (theirs[2U * kSW] != mine[2U * kSW] || theirs[2U * kSW + 1U] != mine[2U * kSW + 1U])
                        throw std::runtime_error("h3 dist sparse: the ranks are at different model calls / layers (dense window)");
                    ck(hipMemcpyAsync(st + std::size_t{peer} * 2U * kSW, theirs.data(), 2U * kSW * 4U, hipMemcpyHostToDevice, s), "stats h2d");
                    ck(hipStreamSynchronize(s), "stats");  // `theirs` goes out of scope
                }
                pm(kpSparse);
            }
            if (role.world == 2U) {
                I.start(par, i);  // the transfer progresses from here (pipeline workers), finish only joins
                if (hook && opt.inject == Inject::throw_in_flight) {
                    I.injected = true;
                    throw InjectedFailure("injected failure in block " + std::to_string(i) + " with the exchange in flight");
                }
            } else {
                I.last_started = i;
            }
            if (hook && opt.inject == Inject::nan_q) ck(hipMemsetD16Async(I.q.as<void>(), 0x7fc0U, 1, s), "inject nan q");
            if (hook && opt.inject == Inject::v_big) ck(hipMemsetD16Async(qkv + 2U * kInner, 0x7200U, 1, s), "inject |V| 2^101");
            if (hook && opt.inject == Inject::bound_rows) ck(hipMemsetD32Async(I.bound.as<void>(), 1U, 1, s), "inject bound rows");
            if (hook) I.injected = true;
            // admission measurement (attention operand contract) of the rows this rank produced: max ||q||, max ||k||,
            // max |V|; read-only scans issued after the exchange started (they never delay the transfer)
            unsigned* nb_i = I.norms.as<unsigned>() + std::size_t{i} * kAdmitPerBlock;
            launch_max_norm(I.q.as<std::uint16_t>(), std::uint64_t{n} * kHeads, nb_i, s);
            launch_max_norm(kdst, std::uint64_t{n} * kHeads, nb_i + 1, s);
            launch_max_abs_v(qkv, n, nb_i + 2, s);
            pm(kpScans);
            if (sl) {
                float* st = A->st.as<float>();
                spz::launch_stats_finalize(st, st + kSW, st + 2U * kSW, st + 3U * kSW, geo.KT, A->mv.as<float>(), A->mv.as<float>() + kSW, s);
                spz::launch_thresholds(qbar, qi1 - qi0, A->mv.as<float>(), A->mv.as<float>() + kSW, A->spp.tau, thr, s);
                pm(kpSparse);
            }
            // sparse partial of key block b (its kc must be in place): LUT part of this rank's query tiles, K1 sparse
            auto sparse_partial = [&](unsigned b, const attn::KeyBlock& blk) {
                std::uint32_t j0, j1;
                ktiles(b, j0, j1);
                spz::DeviceLut lut = spz::build_lut(geo, A->spp, qt0, qt1, qbar, thr, j0, j1, kcb + std::size_t{j0} * kSW, s);
                const spz::BlockLut bl{lut.entries_n ? lut.entries.as<std::int32_t>() : A->dummy.as<std::int32_t>(), lut.offsets.as<std::uint32_t>(), j0};
                spz::launch_sparse_partials(Q, &blk, &bl, 1, P, b, A->lut_err.as<unsigned>(), sv);
                if (lut.entries_n)
                    launch_lut_digest(lut.entries.as<std::int32_t>(), lut.offsets.as<std::uint32_t>(), (qt1 - qt0) * spz::kHeads,
                        std::uint64_t{qt0} * spz::kHeads, b, A->lutd.as<unsigned long long>() + std::size_t{i} * 2U + b, s);
                lut_entries[i] += lut.entries_n;
                lut_keep.push_back(std::move(lut));
            };
            if (role.world == 2U) {
                const unsigned own = role.block(), peer = role.peer_block();
                const attn::KeyBlock kb_own = I.key_block(kdst, vtdst, sp.rows(own));
                if (sl) sparse_partial(own, kb_own);
                else attn::launch_partials(Q, &kb_own, 1, P, own, sv);
                rec(i, 2);
                pm(kpAttnLocal);
                const auto f0 = std::chrono::steady_clock::now();
                I.finish(par);
                res.timing[i].finish_host_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - f0).count();
                rec(i, 3);
                pm(kpExchWait);
                const attn::KeyBlock kb_peer = I.key_block(I.msg_k(I.rx[par]), I.msg_vt(I.rx[par]), sp.rows(peer));
                if (sl) {  // the remote block's kc from RX (same kernel, same bits as on its owner), then its LUT part
                    std::uint32_t j0, j1;
                    ktiles(peer, j0, j1);
                    spz::launch_key_centroids(I.msg_k(I.rx[par]), kSW, geo, j0, j1, kcb + std::size_t{j0} * kSW, s);
                    sparse_partial(peer, kb_peer);
                } else {
                    attn::launch_partials(Q, &kb_peer, 1, P, peer, sv);
                }
                ck(hipEventRecord(I.last_use[par], s), "rx last use");  // the H2D of block i + 2 into RX[par] waits on it
            } else {
                for (unsigned kb = 0; kb < 2; ++kb) {
                    const attn::KeyBlock blk = I.key_block(kdst + std::size_t{sp.first(kb)} * kInner, vtdst + sp.first(kb), sp.rows(kb));
                    if (sl) sparse_partial(kb, blk);
                    else attn::launch_partials(Q, &blk, 1, P, kb, sv);
                    if (kb == 0) { rec(i, 2); rec(i, 3); }
                }
                pm(kpAttnLocal);
            }
            if (sl) ++A->last.sparse_layers;
            attn::launch_merge(P, 2, attn::Output{ao, nullptr, kInner}, sv);
            rec(i, 4);
            pm(kpAttnRemote);
            gemm::linear(ao, kInner, n, W.out, I.ws, o, kHidden, s);
            pm(kpOut);
            if (turbo) A->T->device().apply(lora::Site::kOut, i, ao, kInner, n, o, kHidden, s);
            if (turbo) pm(kpTurbo);
            block::gpu::launch_gate_residual(x, o, mr, mod[2], n, kHidden, sv);
            pm(kpGlue);
            // MLP half
            block::gpu::launch_rmsnorm_mod(x, W.norm2.as<std::uint16_t>(), mr, mod[4], mod[3], n, kHidden, kNormEps, h, sv);
            pm(kpNorm);
            if (unfused) {  // fc1 in checkpoint order [rows][28672], Turbo branch, K5 SwiGLU (== the fused epilogue's bits)
                std::uint16_t* f1 = d_f1.as<std::uint16_t>();
                gemm::linear(h, kHidden, n, W.fc1, I.ws, f1, 2U * kFfn, s);
                pm(kpFc1);
                if (turbo) A->T->device().apply(lora::Site::kFc1, i, h, kHidden, n, f1, 2U * kFfn, s);
                if (turbo) pm(kpTurbo);
                block::gpu::launch_swiglu(f1, n, kFfn, gemm::device_silu_table(), sw, sv);
                pm(kpSwiglu);
            } else {
                gemm::linear_fc1_swiglu(h, kHidden, n, W.fc1, I.ws, sw, kFfn, s);
                pm(kpFc1);
            }
            gemm::linear(sw, kFfn, n, W.fc2, I.ws, o, kHidden, s);
            pm(kpFc2);
            if (turbo_fc2) {  // step LoRAs: run-time fc2 branch right after the int8 fc2 GEMM (local rows), before the gate
                A->T->device().apply(lora::Site::kFc2, i, sw, kFfn, n, o, kHidden, s);
                pm(kpTurbo);
            }
            block::gpu::launch_gate_residual(x, o, mr, mod[5], n, kHidden, sv);
            pm(kpGlue);
            for (unsigned kb = 0; kb < 2; ++kb) {  // cheap device digest of each owned key block (no sync)
                if (role.world == 2U && kb != role.block()) continue;
                const std::size_t off = role.world == 2U ? 0U : std::size_t{sp.first(kb)} * kHidden;
                launch_digest(x + off, std::size_t{sp.rows(kb)} * kHidden, std::uint64_t{sp.first(kb)} * kHidden,
                    I.digests.as<unsigned long long>() + std::size_t{i} * 2U + kb, s);
            }
            if (opt.hashes) {
                host_x.resize(std::size_t{n} * kHidden);
                ck(hipMemcpyAsync(host_x.data(), x, host_x.size() * 2, hipMemcpyDeviceToHost, s), "x download");
                ck(hipStreamSynchronize(s), "block sync");
                for (unsigned kb = 0; kb < 2; ++kb) {
                    if (role.world == 2U && kb != role.block()) continue;
                    const std::size_t off = role.world == 2U ? 0U : std::size_t{sp.first(kb)} * kHidden;
                    res.hash[i][kb] = fnv1a64(host_x.data() + off, std::size_t{sp.rows(kb)} * kHidden * 2U);
                }
            }
            pm(kpDigest);
            if (A) A->last.blocks_run = i + 1U;
            if (fbc_on && i == 0U) {
                // FirstBlockCache after block 0: chunk sums of the local rows (r = h1 - h0), the ranks' (N_b, D_b) pairs
                // in key-block order, one decision on identical inputs everywhere, asserted equal across the ranks
                if (A->inject_fbc_fail) {
                    A->inject_fbc_fail = false;
                    A->fbc->inject_failure_for_test();
                }
                const std::vector<fbcl::Sums> ch = A->fbc->after_block0(x, n, s);
                const bool ready = !ch.empty();
                const fbcl::Sums mine = ready ? fbcl::block_sums(ch, 0, ch.size()) : fbcl::Sums{};
                std::vector<fbcl::Sums> blocks;
                if (role.world == 2U) {
                    struct Msg {
                        std::uint32_t ready, chunks, run_id, call_index;
                        double n, d;
                    } m{ready ? 1U : 0U, static_cast<std::uint32_t>(ch.size()), A->run_id, A->call_index, mine.n, mine.d}, pm{};
                    I.comm->exchange_host(std::as_bytes(std::span(&m, 1)), std::as_writable_bytes(std::span(&pm, 1)),
                        strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 0xA6U, I.request++});
                    if (pm.run_id != m.run_id || pm.call_index != m.call_index)
                        throw std::runtime_error("h3 dist fbc: the ranks are in different sampling runs / calls (run " + std::to_string(m.run_id) + " call " +
                            std::to_string(m.call_index) + " vs peer run " + std::to_string(pm.run_id) + " call " + std::to_string(pm.call_index) + ")");
                    if (pm.ready != m.ready) throw std::runtime_error("h3 dist fbc: the cache state differs between the ranks");
                    blocks.resize(2);
                    blocks[role.block()] = mine;
                    blocks[role.peer_block()] = fbcl::Sums{pm.n, pm.d};
                } else if (ready) {
                    blocks = sp.cut ? fbcl::split_block_sums(ch, n, sp.cut) : std::vector<fbcl::Sums>{mine};
                }
                fbcl::Decision dec;
                if (ready) dec = fbcl::decide(blocks, std::uint64_t{sp.S} * kHidden, A->thr);
                fbcl::Decision d = dec;
                if (A->observe) d.reuse = false;
                if (A->force == DistAccel::FbcForce::reuse) { d.finite = true; d.reuse = true; }
                if (A->force == DistAccel::FbcForce::refresh) { d.finite = true; d.reuse = false; }
                if (role.world == 2U) {
                    std::uint32_t sb;
                    std::memcpy(&sb, &d.score, 4);
                    const std::array<std::uint32_t, 4> dm{ready ? 1U : 0U, d.reuse ? 1U : 0U, d.finite ? 1U : 0U, sb};
                    std::array<std::uint32_t, 4> pd{};
                    I.comm->exchange_host(std::as_bytes(std::span(dm)), std::as_writable_bytes(std::span(pd)),
                        strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 0xA7U, I.request++});
                    if (pd != dm) throw std::runtime_error("h3 dist fbc: the ranks decided differently (decision / score bits)");
                }
                const fbcl::Action act = A->fbc->resolve(d, x, s);
                pm(kpFbc);
                A->last.fbc_cold = !ready;
                A->last.fbc_score = dec.score;
                A->last.fbc_would_reuse = ready && dec.reuse;
                A->last.fbc_action = static_cast<int>(act);
                A->last.run_id = A->run_id;
                A->last.call_index = A->call_index;
                commit = act == fbcl::Action::kFullCommit;
                if (act == fbcl::Action::kReuse) {
                    reused = true;
                    if (nblocks > 1U)  // the final state's digest in the last block's slot (blocks 1.. skipped)
                        for (unsigned kb = 0; kb < 2; ++kb) {
                            if (role.world == 2U && kb != role.block()) continue;
                            const std::size_t off = role.world == 2U ? 0U : std::size_t{sp.first(kb)} * kHidden;
                            launch_digest(x + off, std::size_t{sp.rows(kb)} * kHidden, std::uint64_t{sp.first(kb)} * kHidden,
                                I.digests.as<unsigned long long>() + std::size_t{nblocks - 1U} * 2U + kb, s);
                        }
                    break;
                }
            }
        }
        if (commit) A->fbc->commit(x, n, s);
        if (fbc_on) ++A->call_index;
        if (A && A->sparse_on && !reused) ++A->sp_call;  // an FBC-reused call does not advance the dense window
        if (opt.timing) ck(hipEventRecord(ev.back(), s), "event record");
        ck(hipStreamSynchronize(s), "forward");
    } catch (const std::exception& e) {
        // stop the transport's submission before any buffer or event can be released, then drain the GPU
        I.fail(e.what());
        if (A && A->fbc) A->fbc->reset();  // an abandoned call leaves no cache behind
        throw;
    }
    res.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
    res.flushed_groups = I.ws.flush_count(s);
    if (opt.profile) {
        res.kernel_names.assign(kPkNames, kPkNames + kpCount);
        res.kernel_ms.assign(kpCount, 0.0);
        for (std::size_t k = 1; k < pn; ++k)
            if (pcat[k] >= 0) res.kernel_ms[static_cast<std::size_t>(pcat[k])] += ms_between(pev.v[k - 1], pev.v[k]);
        // standalone activation quantizer (the first kernel of every int8 linear) per K on this rank's rows
        const std::uint16_t* src[3] = {h, ao, sw};
        const std::uint32_t kk[3] = {kHidden, kInner, kFfn};
        Events qe(2);
        for (int k = 0; k < 3; ++k) {
            std::vector<double> t;
            for (int r = 0; r < 6; ++r) {
                ck(hipEventRecord(qe.v[0], s), "q ev");
                gemm::launch_quantize(src[k], kk[k], n, kk[k], I.ws.codes(), I.ws.row_scale(), I.ws.flush(), s);
                ck(hipEventRecord(qe.v[1], s), "q ev");
                ck(hipStreamSynchronize(s), "q sync");
                if (r) t.push_back(ms_between(qe.v[0], qe.v[1]));
            }
            std::sort(t.begin(), t.end());
            res.quantize_ms[k] = t[t.size() / 2];
        }
    }
    ck(hipMemcpy(res.digest.data(), I.digests.as<void>(), std::size_t{nblocks} * 2U * 8U, hipMemcpyDeviceToHost), "digests");
    {
        // Admission (attention operand contract), the same verdict on both ranks: each rank's per-block measurements
        // and bound-rows counter are exchanged (one small message per forward) and every value is checked on its own.
        unsigned brows = 0;
        ck(hipMemcpy(&brows, I.bound.as<void>(), 4, hipMemcpyDeviceToHost), "bound counter");
        res.attn_bound_rows = brows;
        if (sparse_fw) {  // sparse K1 LUT contract violations are attention contract failures (same pair-wide verdict)
            unsigned le = 0;
            ck(hipMemcpy(&le, A->lut_err.as<void>(), 4, hipMemcpyDeviceToHost), "lut errors");
            brows += le;
            A->last.lut_entries = lut_entries;
            std::vector<unsigned long long> d(std::size_t{nblocks} * 2U);
            ck(hipMemcpy(d.data(), A->lutd.as<void>(), d.size() * 8U, hipMemcpyDeviceToHost), "lut digests");
            A->last.lut_digest.assign(nblocks, 0);
            for (std::uint32_t k = 0; k < nblocks; ++k) A->last.lut_digest[k] = d[2U * k] + d[2U * k + 1U];
        }
        std::vector<unsigned> nb(std::size_t{nblocks} * kAdmitPerBlock + 1U), peer(nb.size(), 0U);
        ck(hipMemcpy(nb.data(), I.norms.as<void>(), (nb.size() - 1U) * 4U, hipMemcpyDeviceToHost), "norms");
        nb.back() = brows;
        std::vector<const std::vector<unsigned>*> ops;
        std::vector<std::uint64_t> rows{brows};
        std::vector<unsigned> mine(nb.begin(), nb.end() - 1), theirs;
        ops.push_back(&mine);
        if (role.world == 2U) {
            I.comm->exchange_host(std::as_bytes(std::span<const unsigned>(nb)), std::as_writable_bytes(std::span<unsigned>(peer)),
                strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 0xA5U, I.request++});
            theirs.assign(peer.begin(), peer.end() - 1);
            // rank order in the verdict: rank 0 first on both ranks (same first violation reported everywhere)
            if (role.rank == 0U) ops.push_back(&theirs);
            else ops.insert(ops.begin(), &theirs);
            rows.push_back(peer.back());
        }
        const AdmissionResult ad = admit(ops, rows, nblocks);  // throws ContractViolation (run / forward cancel the transport)
        res.admission_bound = ad.worst;
        res.admission_block = ad.worst_block;
        res.v_max = ad.v_max;
    }
    if (download_x) {
        res.x.resize(std::size_t{n} * kHidden);
        ck(hipMemcpy(res.x.data(), x, res.x.size() * 2, hipMemcpyDeviceToHost), "x final");
    }
    if (opt.timing) {
        for (std::uint32_t i = 0; i < nblocks; ++i) {
            hipEvent_t* e = ev.data() + std::size_t{i} * kEv;
            auto& tm = res.timing[i];
            tm.block_ms = ms_between(e[0], i + 1 < nblocks ? e[kEv] : ev.back());
            tm.local_ms = ms_between(e[1], e[2]);
            tm.exposed_ms = ms_between(e[2], e[3]);
            tm.attn_ms = tm.local_ms + ms_between(e[3], e[4]);
        }
    }
    if (I.comm) {
        const auto& st = I.comm->stats();
        char b[400];
        std::snprintf(b, sizeof b,
            "exchanges %llu bytes %.1f MB stage %.3f s wire %.3f s check %.3f s overlap %.3f s finish_wait %.3f s pipeline_reuse_wait %.3f s",
            static_cast<unsigned long long>(st.exchanges), st.bytes_sent / 1e6, st.stage, st.wire, st.check, st.overlap, st.finish_wait,
            st.pipeline_reuse_wait);
        res.comm_stats = b;
    }
    return res;
}

// --------------------------------------------------------------------------------------------- whole forward
namespace {
// The rows [a, b) of segment `kind` that fall into [r0, r0 + n): returns false when empty.
bool seg_overlap(const block::PackedLayout& L, block::SegKind kind, std::uint32_t r0, std::uint32_t n, std::uint32_t& a, std::uint32_t& b) {
    const block::Segment& seg = L.find(kind);
    a = std::max(seg.start, r0);
    b = std::min(seg.stop, r0 + n);
    return a < b;
}
}  // namespace

DeviceDistRank::ForwardResult DeviceDistRank::forward(const block::PackedLayout& L, const block::ForwardPlan& p, const block::StreamInputs& in,
    const ForwardTables& t, std::uint32_t nblocks, const DeviceRunOptions& opt, DistAccel* accel) {
    if (accel && accel->impl().dead) throw std::logic_error("h3 dist accel: failed earlier (pair-wide abort): rebuild the pair and the accelerators");
    impl_->acc = accel;
    try {
        if (accel && accel->impl().sparse_on) accel->impl().sp_P = L.find(block::SegKind::video).start;
        if (accel && accel->impl().turbo_on() && accel->impl().T->has_adaln()) {
            DistAccel::Impl& A = accel->impl();
            A.deltas = A.T->adaln_deltas(*impl_->M.g, t.t_emb, t.n_t);
            A.last.adaln_idx = A.deltas.idx;
        }
        ForwardResult r = forward_impl(L, p, in, t, nblocks, opt);
        impl_->acc = nullptr;
        return r;
    } catch (const std::exception& e) {
        impl_->fail(e.what());  // cancels the communicator: the peer fails at its next exchange (pair-wide abort)
        if (accel) {
            if (accel->impl().fbc) accel->impl().fbc->reset();
            accel->impl().dead = true;
        }
        impl_->acc = nullptr;
        throw;
    }
}

DeviceDistRank::ForwardResult DeviceDistRank::forward_impl(const block::PackedLayout& L, const block::ForwardPlan& p, const block::StreamInputs& in,
    const ForwardTables& t, std::uint32_t nblocks, const DeviceRunOptions& opt) {
    Impl& I = *impl_;
    const DeviceModel::Impl& M = I.M;
    const block::GlobalWeights& g = *M.g;
    hipStream_t s = I.s;
    void* sv = static_cast<void*>(s);
    const std::uint32_t r0 = I.r0, n = I.n;
    if (L.seq_len != I.sp.S) throw std::invalid_argument("h3 dist: layout does not match the split");
    ForwardResult fr;
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto secs = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };
    const auto t0 = now();
    // ---- embedding of the rank's rows: host patchify (all rows, cheap), device projections of the local rows, text
    // rows copied (row-local: the same bits as the single-node embedding of those rows)
    {
        std::vector<float> all_v, all_a;
        block::embed_rows(L, in, all_v, all_a);
        std::vector<float> lv, la;
        std::vector<std::int32_t> dv, da;
        for (std::size_t i = 0; i < L.img_pos.size(); ++i) {
            const auto r = static_cast<std::uint32_t>(L.img_pos[i]);
            if (r < r0 || r >= r0 + n) continue;
            lv.insert(lv.end(), all_v.begin() + static_cast<std::ptrdiff_t>(i * block::kVideoPatch),
                all_v.begin() + static_cast<std::ptrdiff_t>((i + 1) * block::kVideoPatch));
            dv.push_back(static_cast<std::int32_t>(r - r0));
        }
        for (std::size_t i = 0; i < L.audio_pos.size(); ++i) {
            const auto r = static_cast<std::uint32_t>(L.audio_pos[i]);
            if (r < r0 || r >= r0 + n) continue;
            la.insert(la.end(), all_a.begin() + static_cast<std::ptrdiff_t>(i * block::kAudioDim),
                all_a.begin() + static_cast<std::ptrdiff_t>((i + 1) * block::kAudioDim));
            da.push_back(static_cast<std::int32_t>(r - r0));
        }
        Buf d_lv = upload(lv), d_la = upload(la), d_dv = upload(dv), d_da = upload(da);
        std::uint16_t* x = I.x.as<std::uint16_t>();
        if (!dv.empty())
            block::gpu::launch_patch_proj(d_lv.as<float>(), static_cast<std::uint32_t>(dv.size()), block::kVideoPatch, M.vproj_w.as<float>(),
                M.vproj_b.as<float>(), d_dv.as<std::int32_t>(), x, sv);
        if (!da.empty())
            block::gpu::launch_patch_proj(d_la.as<float>(), static_cast<std::uint32_t>(da.size()), block::kAudioDim, M.aproj_w.as<float>(),
                M.aproj_b.as<float>(), d_da.as<std::int32_t>(), x, sv);
        std::uint32_t a = 0, b = 0;
        const block::Segment& ts = L.find(block::SegKind::text);
        if (in.text.size() != std::size_t{ts.stop - ts.start} * kHidden) throw std::runtime_error("h3 dist: text size");
        if (seg_overlap(L, block::SegKind::text, r0, n, a, b))
            ck(hipMemcpyAsync(x + std::size_t{a - r0} * kHidden, in.text.data() + std::size_t{a - ts.start} * kHidden,
                   std::size_t{b - a} * kHidden * 2, hipMemcpyHostToDevice, s), "text rows");
        ck(hipStreamSynchronize(s), "embed");  // the host vectors above go out of scope
    }
    const auto t1 = now();
    fr.run = run_blocks(t, g.basis, nblocks, opt, false);
    const auto t2 = now();
    // ---- final layer of the rank's video / audio rows (row-local), fp32 rows
    block::F32Linear vbank, abank;
    const bool bank = block::select_heads(g, p, nullptr, vbank, abank);
    Buf d_vbw, d_vbb, d_abw, d_abb;
    if (bank) {
        d_vbw = upload(vbank.w);
        d_vbb = upload(vbank.b);
        d_abw = upload(abank.w);
        d_abb = upload(abank.b);
    }
    Buf d_temb = upload(t.t_emb);
    Buf d_fada(std::size_t{t.n_t} * 2U * kHidden * 4U);
    block::gpu::launch_linear_f64(d_temb.as<float>(), g.basis, t.n_t, M.final_ada_w.as<float>(), M.final_ada_b.as<float>(), 2U * kHidden, g.basis,
        d_fada.as<float>(), sv);
    Buf d_fdelta;
    if (I.acc && I.acc->impl().turbo_on() && I.acc->impl().T->has_adaln()) {
        const DistAccel::Impl& A = I.acc->impl();
        if (A.deltas.final_out != 2U * kHidden || A.deltas.final.size() != std::size_t{t.n_t} * A.deltas.final_out)
            throw std::runtime_error("h3 dist turbo: final adaLN deltas");
        d_fdelta = upload(A.deltas.final);
        launch_add_f32(d_fada.as<float>(), d_fdelta.as<float>(), std::uint64_t{t.n_t} * A.deltas.final_out, s);
    }
    struct Part {
        block::SegKind kind;
        std::uint32_t trow, nout;
        const Buf* w;
        const Buf* b;
        std::vector<float> rows;
        std::uint32_t a = 0, m = 0;  // first row within the segment, row count
    };
    Part parts[2] = {{block::SegKind::video, p.video_t_row, block::kVideoPatch, bank ? &d_vbw : &M.vout_w, bank ? &d_vbb : &M.vout_b, {}},
        {block::SegKind::audio, p.audio_t_row, block::kAudioDim, bank ? &d_abw : &M.aout_w, bank ? &d_abb : &M.aout_b, {}}};
    for (auto& pt : parts) {
        std::uint32_t a = 0, b = 0;
        if (!seg_overlap(L, pt.kind, r0, n, a, b)) continue;
        pt.a = a - L.find(pt.kind).start;
        pt.m = b - a;
        Buf d_f(std::size_t{pt.m} * kHidden * 4U), d_rows(std::size_t{pt.m} * pt.nout * 4U);
        const float* sh = d_fada.as<float>() + std::size_t{pt.trow} * 2U * kHidden;
        block::gpu::launch_final_mod(I.x.as<std::uint16_t>(), a - r0, pt.m, M.final_norm.as<std::uint16_t>(), sh, sh + kHidden, kNormEps,
            d_f.as<float>(), sv);
        block::gpu::launch_linear_f64(d_f.as<float>(), kHidden, pt.m, pt.w->as<float>(), pt.b->as<float>(), pt.nout, kHidden, d_rows.as<float>(), sv);
        pt.rows.resize(std::size_t{pt.m} * pt.nout);
        ck(hipMemcpyAsync(pt.rows.data(), d_rows.as<void>(), pt.rows.size() * 4, hipMemcpyDeviceToHost, s), "final rows");
        ck(hipStreamSynchronize(s), "final layer");
    }
    const auto t3 = now();
    // ---- gather: one symmetric exchange of each rank's final rows (padded to the larger count), so both ranks
    // hold the full velocity (rank 0 feeds the sampler)
    block::FinalRows full;
    for (auto& pt : parts) {
        const block::Segment& seg = L.find(pt.kind);
        std::vector<float>& dst = pt.kind == block::SegKind::video ? full.video : full.audio;
        dst.assign(std::size_t{seg.stop - seg.start} * pt.nout, 0.0F);
        if (pt.m) std::copy(pt.rows.begin(), pt.rows.end(), dst.begin() + static_cast<std::ptrdiff_t>(std::size_t{pt.a} * pt.nout));
    }
    if (I.role.world == 2U) {
        const Role peer{1U - I.role.rank, 2U, I.role.swap};  // the peer's rows, from the shared split
        std::size_t mine_n = 0, peer_n = 0;
        std::uint32_t pa[2] = {0, 0}, pm[2] = {0, 0};
        for (int k = 0; k < 2; ++k) {
            std::uint32_t a = 0, b = 0;
            const block::Segment& seg = L.find(parts[k].kind);
            if (seg_overlap(L, parts[k].kind, peer.row0(I.sp), peer.rows(I.sp), a, b)) {
                pa[k] = a - seg.start;
                pm[k] = b - a;
            }
            mine_n += std::size_t{parts[k].m} * parts[k].nout;
            peer_n += std::size_t{pm[k]} * parts[k].nout;
        }
        const std::size_t max_n = std::max<std::size_t>({mine_n, peer_n, 1});
        std::vector<float> out(max_n, 0.0F), got(max_n, 0.0F);
        std::size_t at = 0;
        for (auto& pt : parts) {
            std::copy(pt.rows.begin(), pt.rows.end(), out.begin() + static_cast<std::ptrdiff_t>(at));
            at += pt.rows.size();
        }
        I.comm->exchange_host(std::as_bytes(std::span<const float>(out)), std::as_writable_bytes(std::span<float>(got)),
            strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::raw_slots), 0xF1U, I.request++});
        at = 0;
        for (int k = 0; k < 2; ++k) {
            std::vector<float>& dst = parts[k].kind == block::SegKind::video ? full.video : full.audio;
            const std::size_t cnt = std::size_t{pm[k]} * parts[k].nout;
            std::copy(got.begin() + static_cast<std::ptrdiff_t>(at), got.begin() + static_cast<std::ptrdiff_t>(at + cnt),
                dst.begin() + static_cast<std::ptrdiff_t>(std::size_t{pa[k]} * parts[k].nout));
            at += cnt;
        }
        fr.gather_bytes = max_n * 4U;
    }
    const auto t4 = now();
    const std::vector<float> v = block::unpatchify_video(full.video.data(), L.latent_t, L.latent_h, L.latent_w);
    const std::vector<float> au = block::unpack_audio(full.audio.data(), L.audio_t);
    fr.out.video_neg_v.resize(v.size());
    fr.out.audio_neg_v.resize(au.size());
    for (std::size_t i = 0; i < v.size(); ++i) fr.out.video_neg_v[i] = block::tob(-v[i]);
    for (std::size_t i = 0; i < au.size(); ++i) fr.out.audio_neg_v[i] = block::tob(-au[i]);
    fr.embed_s = secs(t0, t1);
    fr.blocks_s = secs(t1, t2);
    fr.final_s = secs(t2, t3);
    fr.gather_s = secs(t3, t4);
    fr.wall_s = secs(t0, now());
    return fr;
}

DeviceDistRank::AttnBench DeviceDistRank::bench_attention(unsigned reps) {
    try {
        return bench_attention_impl(reps);
    } catch (const std::exception& e) {
        impl_->fail(e.what());
        throw;
    }
}

DeviceDistRank::AttnBench DeviceDistRank::bench_attention_impl(unsigned reps) {
    Impl& I = *impl_;
    if (I.role.world != 2U) throw std::invalid_argument("h3 dist: the attention bench needs two ranks");
    hipStream_t s = I.s;
    void* sv = static_cast<void*>(s);
    const Split& sp = I.sp;
    const unsigned own = I.role.block(), peer = I.role.peer_block();
    fill_bf16(I.q.as<std::uint16_t>(), std::size_t{I.n} * kInner, 11 + I.role.rank, 0.5F, s);
    for (int j = 0; j < 2; ++j) {
        I.wait_tx(j);
        fill_bf16(I.tx[j].as<std::uint16_t>(), sp.message_bytes() / 2U, 101 + 7 * j + I.role.rank, 1.0F, s);
        fill_bf16(I.rx[j].as<std::uint16_t>(), sp.message_bytes() / 2U, 301 + 7 * j + I.role.rank, 1.0F, s);
    }
    ck(hipStreamSynchronize(s), "bench fill");
    const attn::Queries Q = I.queries();
    const attn::Partials P = I.partials();
    Events evs(3);
    hipEvent_t a = evs.v[0], b = evs.v[1], c = evs.v[2];
    Buf scan(16);
    // the block loop's read-only operand scans (between the exchange start and the local partial), in both variants
    auto scans = [&] {
        launch_max_norm(I.q.as<std::uint16_t>(), std::uint64_t{I.n} * kHeads, scan.as<unsigned>(), s);
        launch_max_norm(I.msg_k(I.tx[0]), std::uint64_t{I.n} * kHeads, scan.as<unsigned>() + 1, s);
        launch_max_abs_v(I.qkv.as<std::uint16_t>(), I.n, scan.as<unsigned>() + 2, s);
    };
    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    std::vector<double> attn_ms, local_ms, xch_ms, local_x_ms, wall_ms, fin_ms;
    const attn::KeyBlock kb_own = I.key_block(I.msg_k(I.tx[0]), I.msg_vt(I.tx[0]), sp.rows(own));
    const attn::KeyBlock kb_peer = I.key_block(I.msg_k(I.rx[0]), I.msg_vt(I.rx[0]), sp.rows(peer));
    attn::launch_partials(Q, &kb_own, 1, P, own, sv);  // warm-up
    ck(hipStreamSynchronize(s), "warm");
    // one warm-up exchange (lines the ranks up), then the three measurements INTERLEAVED per repetition, so slow
    // drifts (clocks, other host load) hit all three alike: attention alone, exchange alone, local partial with the
    // exchange in flight (start_async, as the block loop does)
    I.comm->exchange(s, I.tx[1].as<void>(), I.rx[1].as<void>(), sp.message_bytes(), I.tag(0xBE00U));
    ck(hipStreamSynchronize(s), "sync");
    std::vector<double> slow_ms;
    for (unsigned r = 0; r < reps; ++r) {
        ck(hipEventRecord(a, s), "ev");
        scans();
        attn::launch_partials(Q, &kb_own, 1, P, own, sv);
        ck(hipEventRecord(b, s), "ev");
        attn::launch_partials(Q, &kb_peer, 1, P, peer, sv);
        attn::launch_merge(P, 2, attn::Output{I.ao.as<std::uint16_t>(), nullptr, kInner}, sv);
        ck(hipEventRecord(c, s), "ev");
        ck(hipStreamSynchronize(s), "sync");
        local_ms.push_back(ms_between(a, b));
        attn_ms.push_back(ms_between(a, c));

        const auto x0 = std::chrono::steady_clock::now();
        I.comm->exchange(s, I.tx[1].as<void>(), I.rx[1].as<void>(), sp.message_bytes(), I.tag(0xBE01U + r));
        ck(hipStreamSynchronize(s), "sync");
        xch_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - x0).count());

        const auto t0 = std::chrono::steady_clock::now();
        I.wait_tx(0);
        ck(hipEventRecord(a, s), "ev");  // as the block loop's event 1: before the start
        I.start(0, 0xBF00U + r);
        scans();
        attn::launch_partials(Q, &kb_own, 1, P, own, sv);
        ck(hipEventRecord(b, s), "ev");
        const auto f0 = std::chrono::steady_clock::now();
        I.finish(0);
        const auto f1 = std::chrono::steady_clock::now();
        ck(hipEventRecord(I.last_use[0], s), "last use");
        ck(hipStreamSynchronize(s), "sync");
        wall_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        fin_ms.push_back(std::chrono::duration<double, std::milli>(f1 - f0).count());
        local_x_ms.push_back(ms_between(a, b));
        slow_ms.push_back(local_x_ms.back() - local_ms.back());  // paired: same repetition
    }
    AttnBench r;
    r.local_slowdown_ms = median(slow_ms);
    r.attn_ms = median(attn_ms);
    r.local_ms = median(local_ms);
    r.exchange_ms = median(xch_ms);
    r.local_with_exchange_ms = median(local_x_ms);
    r.overlapped_wall_ms = median(wall_ms);
    r.finish_wait_ms = median(fin_ms);
    return r;
}

}  // namespace strix::h3::dist
