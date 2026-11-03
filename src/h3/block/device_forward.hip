// MiniMax H3 single-node device forward driver (see device_forward.hpp).
#include <hip/hip_runtime.h>  // first: the device builtins used by the gemm reference header

#include "h3/block/device_forward.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

#include "h3/attn/attn.hpp"
#include "h3/block/glue.hpp"
#include "h3/gemm/api.hpp"
#include "h3/gemm/ref.hpp"
#include "h3/levers/fbc/fbc.hpp"
#include "h3/levers/lora/turbo.hpp"

namespace strix::h3::block {

namespace lv = strix::h3::levers::lora;
namespace fbcv = strix::h3::levers::fbc;

namespace {

// x[i] += d[i] (fp32, one rounding): the Turbo adaLN injection on the device adaLN output
__global__ void add_f32_kernel(float* x, const float* d, std::uint64_t n) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] + d[i];
}
void launch_add_f32(float* x, const float* d, std::uint64_t n, hipStream_t s) {
    if (!n) return;
    hipLaunchKernelGGL(add_f32_kernel, dim3(static_cast<unsigned>((n + 255) / 256)), dim3(256), 0, s, x, d, n);
    if (hipGetLastError() != hipSuccess) throw std::runtime_error("hip: add_f32 launch");
}

void hip_check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("hip: ") + what + ": " + hipGetErrorString(e));
}

// Device buffer (bytes), RAII.
class DevBuf {
public:
    DevBuf() = default;
    explicit DevBuf(std::size_t bytes) { alloc(bytes); }
    ~DevBuf() {
        if (p_) static_cast<void>(hipFree(p_));
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    DevBuf(DevBuf&& o) noexcept : p_(o.p_), n_(o.n_) {
        o.p_ = nullptr;
        o.n_ = 0;
    }
    DevBuf& operator=(DevBuf&& o) noexcept {
        if (this != &o) {
            if (p_) static_cast<void>(hipFree(p_));
            p_ = o.p_;
            n_ = o.n_;
            o.p_ = nullptr;
            o.n_ = 0;
        }
        return *this;
    }
    void alloc(std::size_t bytes) {
        if (p_ && bytes <= n_) return;
        if (p_) static_cast<void>(hipFree(p_));
        p_ = nullptr;
        hip_check(hipMalloc(&p_, bytes ? bytes : 16), "hipMalloc");
        n_ = bytes;
    }
    template <class T> T* as() const { return static_cast<T*>(p_); }

private:
    void* p_ = nullptr;
    std::size_t n_ = 0;
};

// Uploads complete before returning (the host vectors may be temporaries).
template <class T> DevBuf upload(const std::vector<T>& v, hipStream_t s) {
    DevBuf b(v.size() * sizeof(T));
    if (!v.empty()) hip_check(hipMemcpyAsync(b.as<void>(), v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice, s), "upload");
    hip_check(hipStreamSynchronize(s), "sync");
    return b;
}
template <class T> std::vector<T> download(const void* d, std::size_t n, hipStream_t s) {
    std::vector<T> v(n);
    if (n) hip_check(hipMemcpyAsync(v.data(), d, n * sizeof(T), hipMemcpyDeviceToHost, s), "download");
    hip_check(hipStreamSynchronize(s), "sync");
    return v;
}

struct BlockDev {
    DevBuf norm1, norm2, q_norm, k_norm, ada_w, ada_b;
    std::uint32_t ada_n = 0;
};
struct RefinerDev {
    DevBuf norm1, norm2, q_norm, k_norm, qkv, out, fc1, fc2;
};

}  // namespace

// ------------------------------------------------------------------------------------------------- fallbacks
void HostFallbackLinear::run(std::uint32_t block, LinearKind kind, const std::uint16_t* d_x, std::uint32_t m, std::uint16_t* d_y, void* stream) {
    const BlockWeights& b = *blocks.at(block);
    const QLinear* w = kind == LinearKind::qkv ? &b.qkv : kind == LinearKind::out ? &b.out : kind == LinearKind::fc1 ? &b.fc1 : &b.fc2;
    hipStream_t s = static_cast<hipStream_t>(stream);
    const auto x = download<std::uint16_t>(d_x, static_cast<std::size_t>(m) * w->k, s);
    std::vector<std::uint16_t> y(static_cast<std::size_t>(m) * w->n);
    host.run(*w, x.data(), m, y.data());
    hip_check(hipMemcpyAsync(d_y, y.data(), y.size() * 2, hipMemcpyHostToDevice, s), "upload y");
    hip_check(hipStreamSynchronize(s), "sync");
}

void HostFallbackAttention::run(const std::uint16_t* d_q, const std::uint16_t* d_k, const std::uint16_t* d_vt, std::uint64_t vt_pitch,
    std::uint32_t rows, std::uint32_t keys, bool /*split*/, std::uint16_t* d_out, void* stream) {
    hipStream_t s = static_cast<hipStream_t>(stream);
    const auto q = download<std::uint16_t>(d_q, static_cast<std::size_t>(rows) * kInner, s);
    const auto k = download<std::uint16_t>(d_k, static_cast<std::size_t>(keys) * kInner, s);
    const auto vp = download<std::uint16_t>(d_vt, static_cast<std::size_t>(kInner) * vt_pitch, s);
    std::vector<std::uint16_t> vt(static_cast<std::size_t>(kInner) * keys);
    for (std::size_t r = 0; r < kInner; ++r) std::memcpy(vt.data() + r * keys, vp.data() + r * vt_pitch, keys * 2U);
    std::vector<std::uint16_t> o(static_cast<std::size_t>(rows) * kInner);
    host->run(q.data(), k.data(), vt.data(), rows, keys, o.data());
    hip_check(hipMemcpyAsync(d_out, o.data(), o.size() * 2, hipMemcpyHostToDevice, s), "upload attn");
    hip_check(hipStreamSynchronize(s), "sync");
}

// ------------------------------------------------------------------------------------------------- gemm linear
struct GemmDeviceLinear::Impl {
    struct DevW {
        DevBuf codes, scales;
        gemm::Weights w;
    };
    struct BlockW {
        DevW qkv, out, fc1, fc2;
    };
    gemm::Workspace ws;
    std::map<std::uint32_t, BlockW> blocks;
    std::uint64_t bytes = 0;
    explicit Impl(std::uint32_t max_rows) : ws(max_rows, kFfn) {}
    void up(DevW& d, const std::int8_t* codes, const float* scales, std::uint32_t n, std::uint32_t k) {
        d.codes.alloc(static_cast<std::size_t>(n) * k);
        d.scales.alloc(static_cast<std::size_t>(n) * 4U);
        hip_check(hipMemcpy(d.codes.as<void>(), codes, static_cast<std::size_t>(n) * k, hipMemcpyHostToDevice), "upload codes");
        hip_check(hipMemcpy(d.scales.as<void>(), scales, static_cast<std::size_t>(n) * 4U, hipMemcpyHostToDevice), "upload scales");
        d.w = gemm::Weights{d.codes.as<std::int8_t>(), d.scales.as<float>(), n, k};
        bytes += static_cast<std::uint64_t>(n) * k + static_cast<std::uint64_t>(n) * 4U;
    }
};
GemmDeviceLinear::GemmDeviceLinear(std::uint32_t max_rows, bool fused) : fused_(fused), impl_(std::make_unique<Impl>(max_rows)) {}
GemmDeviceLinear::~GemmDeviceLinear() = default;
void GemmDeviceLinear::prepare(const std::shared_ptr<const BlockWeights>& b) {
    if (prepared(b->index)) return;
    Impl::BlockW& d = impl_->blocks[b->index];
    impl_->up(d.qkv, b->qkv.w.data(), b->qkv.ws.data(), b->qkv.n, b->qkv.k);
    impl_->up(d.out, b->out.w.data(), b->out.ws.data(), b->out.n, b->out.k);
    if (fused_) {
        const gemm::SwiGLUWeights p = gemm::permute_fc1_for_swiglu(b->fc1.w.data(), b->fc1.ws.data(), b->fc1.n, b->fc1.k);
        impl_->up(d.fc1, p.codes.data(), p.scales.data(), b->fc1.n, b->fc1.k);
    } else {
        impl_->up(d.fc1, b->fc1.w.data(), b->fc1.ws.data(), b->fc1.n, b->fc1.k);
    }
    impl_->up(d.fc2, b->fc2.w.data(), b->fc2.ws.data(), b->fc2.n, b->fc2.k);
}
bool GemmDeviceLinear::prepared(std::uint32_t block) const { return impl_->blocks.count(block) != 0; }
void GemmDeviceLinear::run(std::uint32_t block, LinearKind kind, const std::uint16_t* d_x, std::uint32_t m, std::uint16_t* d_y, void* stream) {
    const Impl::BlockW& d = impl_->blocks.at(block);
    hipStream_t s = static_cast<hipStream_t>(stream);
    switch (kind) {
        case LinearKind::qkv: gemm::linear(d_x, kHidden, m, d.qkv.w, impl_->ws, d_y, d.qkv.w.n, s); break;
        case LinearKind::out: gemm::linear(d_x, kInner, m, d.out.w, impl_->ws, d_y, d.out.w.n, s); break;
        case LinearKind::fc1:
            if (fused_) gemm::linear_fc1_swiglu(d_x, kHidden, m, d.fc1.w, impl_->ws, d_y, kFfn, s);
            else gemm::linear(d_x, kHidden, m, d.fc1.w, impl_->ws, d_y, d.fc1.w.n, s);
            break;
        case LinearKind::fc2: gemm::linear(d_x, kFfn, m, d.fc2.w, impl_->ws, d_y, d.fc2.w.n, s); break;
    }
}
std::uint64_t GemmDeviceLinear::flush_count(void* stream) { return impl_->ws.flush_count(static_cast<hipStream_t>(stream)); }
void GemmDeviceLinear::reset_flush_count(void* stream) { impl_->ws.reset_flush_count(static_cast<hipStream_t>(stream)); }
std::uint64_t GemmDeviceLinear::resident_bytes() const { return impl_->bytes; }

// ------------------------------------------------------------------------------------------------- flash attention
struct FlashDeviceAttention::Impl {
    DevBuf scratch;
    DevBuf counter;  // one unsigned, the attention's bound_rows
};
FlashDeviceAttention::FlashDeviceAttention() : impl_(std::make_unique<Impl>()) {}
FlashDeviceAttention::~FlashDeviceAttention() = default;
void FlashDeviceAttention::run(const std::uint16_t* d_q, const std::uint16_t* d_k, const std::uint16_t* d_vt, std::uint64_t vt_pitch,
    std::uint32_t rows, std::uint32_t keys, bool split, std::uint16_t* d_out, void* stream) {
    const std::vector<std::uint32_t> cuts = key_cuts(keys, split);
    const std::uint32_t nb = static_cast<std::uint32_t>(cuts.size() - 1);
    impl_->scratch.alloc(attn::attention_scratch_floats(rows, kHeads, kHeadDim, nb) * 4U);
    attn::AttentionCall c;
    c.q = d_q;
    c.k = d_k;
    c.vt = d_vt;
    c.rows = rows;
    c.heads = kHeads;
    c.dim = kHeadDim;
    c.keys = keys;
    c.vt_pitch = vt_pitch;
    c.cuts = cuts.data();
    c.nblocks = nb;
    c.scratch = impl_->scratch.as<float>();
    c.out = d_out;
    c.out_rs = kInner;
    if (!impl_->counter.as<void>()) {
        impl_->counter.alloc(16);
        hip_check(hipMemsetAsync(impl_->counter.as<void>(), 0, 16, static_cast<hipStream_t>(stream)), "bound_rows");
    }
    c.bound_rows = impl_->counter.as<unsigned>();
    attn::attention(c, stream);
}
std::uint64_t FlashDeviceAttention::bound_rows(void* stream) {
    if (!impl_->counter.as<void>()) return 0;
    unsigned v = 0;
    hip_check(hipMemcpyAsync(&v, impl_->counter.as<void>(), 4, hipMemcpyDeviceToHost, static_cast<hipStream_t>(stream)), "bound_rows");
    hip_check(hipStreamSynchronize(static_cast<hipStream_t>(stream)), "sync");
    return v;
}
void FlashDeviceAttention::reset_bound_rows(void* stream) {
    if (impl_->counter.as<void>())
        hip_check(hipMemsetAsync(impl_->counter.as<void>(), 0, 16, static_cast<hipStream_t>(stream)), "bound_rows");
}

// ------------------------------------------------------------------------------------------------- driver
struct DeviceForward::Impl {
    const GlobalWeights& g;
    hipStream_t stream = nullptr;
    DevBuf final_norm, final_ada_w, final_ada_b, vout_w, vout_b, aout_w, aout_b, vproj_w, vproj_b, aproj_w, aproj_b, silu;
    std::map<std::uint32_t, BlockDev> blocks;
    bool refiner_ready = false;
    bool strict_flush = false;
    // per-forward work buffers (grown on demand, reused across forwards)
    DevBuf w_x, w_h, w_o, w_qkv, w_q, w_k, w_vt, w_ao, w_sw, w_f1, w_ada, w_mod[6], w_vrows, w_arows, w_vdst, w_adst, w_temb, w_mrow;
    std::uint64_t vt_bytes_zeroed = 0;
    DevBuf w_nmax;  // [blocks][112] per-head max ||q||^2 / ||k||^2 (float bits), the attention contract check
    // rope table per layout (positions are fixed for a sampling run)
    std::vector<double> rope_pos;
    DevBuf rope_c, rope_s;
    DevBuf cond_w, cond_b, ref_final_norm;
    std::vector<RefinerDev> refiner;
    // accelerated forms
    AccelOptions acc;
    std::unique_ptr<fbcv::FirstBlockCacheDevice> fbc;  // bound to `stream` (the only stream of this driver)
    DevBuf w_adelta, w_fdelta;
    explicit Impl(const GlobalWeights& gw) : g(gw) {
        hip_check(hipStreamCreate(&stream), "stream");
        final_norm = upload(g.final_norm, stream);
        final_ada_w = upload(g.final_adaln.w, stream);
        final_ada_b = upload(g.final_adaln.b, stream);
        vout_w = upload(g.video_out.w, stream);
        vout_b = upload(g.video_out.b, stream);
        aout_w = upload(g.audio_out.w, stream);
        aout_b = upload(g.audio_out.b, stream);
        vproj_w = upload(g.video_patch_proj.w, stream);
        vproj_b = upload(g.video_patch_proj.b, stream);
        aproj_w = upload(g.audio_patch_proj.w, stream);
        aproj_b = upload(g.audio_patch_proj.b, stream);
        const auto& st = gemm::silu_table();
        silu = upload(std::vector<std::uint16_t>(st.begin(), st.end()), stream);
    }
    ~Impl() {
        if (stream) static_cast<void>(hipStreamDestroy(stream));
    }
    void prepare_refiner() {
        if (refiner_ready) return;
        cond_w = upload(g.condition_proj.w, stream);
        cond_b = upload(g.condition_proj.b, stream);
        ref_final_norm = upload(g.refiner_final_norm, stream);
        for (const auto& b : g.refiner) {
            RefinerDev d;
            d.norm1 = upload(b.norm1, stream);
            d.norm2 = upload(b.norm2, stream);
            d.q_norm = upload(b.q_norm, stream);
            d.k_norm = upload(b.k_norm, stream);
            d.qkv = upload(b.qkv.w, stream);
            d.out = upload(b.out.w, stream);
            d.fc1 = upload(b.fc1.w, stream);
            d.fc2 = upload(b.fc2.w, stream);
            refiner.push_back(std::move(d));
        }
        refiner_ready = true;
    }
    BlockDev& block_dev(const BlockWeights& b) {
        auto it = blocks.find(b.index);
        if (it != blocks.end()) return it->second;
        BlockDev d;
        d.ada_n = b.adaln.n;
        d.norm1 = upload(b.norm1, stream);
        d.norm2 = upload(b.norm2, stream);
        d.q_norm = upload(b.q_norm, stream);
        d.k_norm = upload(b.k_norm, stream);
        d.ada_w = upload(b.adaln.w, stream);
        d.ada_b = upload(b.adaln.b, stream);
        return blocks.emplace(b.index, std::move(d)).first->second;
    }
};

DeviceForward::DeviceForward(const GlobalWeights& g) : impl_(std::make_unique<Impl>(g)) {}
DeviceForward::~DeviceForward() = default;
void DeviceForward::set_strict_flush(bool strict) { impl_->strict_flush = strict; }
void DeviceForward::set_accel(const AccelOptions& o) {
    if (o.fbc && !(o.fbc_threshold >= 0.0)) throw std::invalid_argument("device forward: FBC threshold");
    impl_->acc = o;
    if (!o.fbc) impl_->fbc.reset();
}
void DeviceForward::reset_accel_run() {
    if (impl_->fbc) impl_->fbc->reset();
}

StreamOutputs DeviceForward::run(BlockSource& blocks, const PackedLayout& L, const ForwardPlan& p, const StreamInputs& in,
    DeviceLinearOp& lin, DeviceAttentionOp& att, DeviceForwardTaps* taps, const PddContext* pdd) {
    Impl& I = *impl_;
    const GlobalWeights& g = I.g;
    hipStream_t s = I.stream;
    void* sv = static_cast<void*>(s);
    const std::uint32_t S = L.seq_len;
    const std::size_t SH = static_cast<std::size_t>(S) * kHidden;
    // ---- embedding (patchify on the host, projections on the device)
    const auto tprep = std::chrono::steady_clock::now();
    std::vector<float> all_v, all_a;
    embed_rows(L, in, all_v, all_a);
    std::vector<std::int32_t> vdst, adst;
    for (auto r : L.img_pos) vdst.push_back(static_cast<std::int32_t>(r));
    for (auto r : L.audio_pos) adst.push_back(static_cast<std::int32_t>(r));
    I.w_x.alloc(SH * 2);
    I.w_h.alloc(SH * 2);
    I.w_o.alloc(SH * 2);
    DevBuf& d_x = I.w_x;
    auto up_into = [&](DevBuf& b, const void* src, std::size_t bytes) {
        b.alloc(bytes);
        if (bytes) hip_check(hipMemcpyAsync(b.as<void>(), src, bytes, hipMemcpyHostToDevice, s), "upload");
    };
    up_into(I.w_vrows, all_v.data(), all_v.size() * 4);
    up_into(I.w_arows, all_a.data(), all_a.size() * 4);
    up_into(I.w_vdst, vdst.data(), vdst.size() * 4);
    up_into(I.w_adst, adst.data(), adst.size() * 4);
    DevBuf &d_vrows = I.w_vrows, &d_arows = I.w_arows, &d_vdst = I.w_vdst, &d_adst = I.w_adst;
    const double host_embed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tprep).count();
    hip_check(hipStreamSynchronize(s), "sync");
    const auto tproj = std::chrono::steady_clock::now();
    gpu::launch_patch_proj(d_vrows.as<float>(), static_cast<std::uint32_t>(vdst.size()), kVideoPatch, I.vproj_w.as<float>(),
        I.vproj_b.as<float>(), d_vdst.as<std::int32_t>(), d_x.as<std::uint16_t>(), sv);
    gpu::launch_patch_proj(d_arows.as<float>(), static_cast<std::uint32_t>(adst.size()), kAudioDim, I.aproj_w.as<float>(),
        I.aproj_b.as<float>(), d_adst.as<std::int32_t>(), d_x.as<std::uint16_t>(), sv);
    for (const auto& seg : L.segments)
        if (seg.kind == SegKind::text) {
            const std::size_t n = static_cast<std::size_t>(seg.stop - seg.start) * kHidden;
            if (in.text.size() != n) throw std::runtime_error("device forward: text size");
            hip_check(hipMemcpyAsync(d_x.as<std::uint16_t>() + static_cast<std::size_t>(seg.start) * kHidden, in.text.data(), n * 2,
                          hipMemcpyHostToDevice, s), "text");
        }
    hip_check(hipStreamSynchronize(s), "sync");
    const double proj_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tproj).count();
    const auto ttab = std::chrono::steady_clock::now();
    if (taps && taps->h_in) *taps->h_in = download<std::uint16_t>(d_x.as<void>(), SH, s);
    // ---- per-forward tables (host: curve lerp, rope table, mod rows)
    const std::vector<float> t_emb = curve_t_emb(g, p.t_vals);
    if (I.rope_pos != L.position_ids) {  // once per layout
        const RopeTable rope = rope_table(rope_angles(L.position_ids, g.inv_freq));
        I.rope_c = upload(rope.c, s);
        I.rope_s = upload(rope.s, s);
        I.rope_pos = L.position_ids;
    }
    DevBuf &d_rc = I.rope_c, &d_rs = I.rope_s;
    const std::vector<std::uint32_t> mrow = mod_row_of(p.mod_segments, S);
    up_into(I.w_temb, t_emb.data(), t_emb.size() * 4);
    up_into(I.w_mrow, mrow.data(), mrow.size() * 4);
    DevBuf &d_temb = I.w_temb, &d_mrow = I.w_mrow;
    const std::uint32_t nt = p.n_t(), rows = p.n_mod_rows();
    const TurboV4* turbo = I.acc.turbo;
    std::uint32_t ada_out = 0;
    if (turbo) {
        if (lin.fused_swiglu()) throw std::invalid_argument("device forward: Turbo needs the non-fused fc1 linear");
    }
    if (turbo && turbo->has_adaln()) {
        TurboV4::AdalnDeltas dl = turbo->adaln_deltas(g, t_emb, nt);
        ada_out = dl.block_out;
        up_into(I.w_adelta, dl.blocks.data(), dl.blocks.size() * 4);
        up_into(I.w_fdelta, dl.final.data(), dl.final.size() * 4);
        hip_check(hipStreamSynchronize(s), "sync deltas");  // dl goes out of scope
        if (taps && taps->accel) taps->accel->adaln_idx = dl.idx;
    }
    I.w_ada.alloc(static_cast<std::size_t>(nt) * 6U * kHidden * 3U * 4U);
    DevBuf& d_ada = I.w_ada;
    std::uint16_t* mod_ptr[6];
    for (int j = 0; j < 6; ++j) {
        I.w_mod[j].alloc(static_cast<std::size_t>(rows) * kHidden * 2);
        mod_ptr[j] = I.w_mod[j].as<std::uint16_t>();
    }
    const std::uint64_t vtp = vt_pitch_for(S);
    I.w_qkv.alloc(static_cast<std::size_t>(S) * 3U * kInner * 2);
    I.w_q.alloc(static_cast<std::size_t>(S) * kInner * 2);
    I.w_k.alloc(static_cast<std::size_t>(S) * kInner * 2);
    I.w_ao.alloc(static_cast<std::size_t>(S) * kInner * 2);
    I.w_sw.alloc(static_cast<std::size_t>(S) * kFfn * 2);
    if (!lin.fused_swiglu()) I.w_f1.alloc(static_cast<std::size_t>(S) * 2U * kFfn * 2);
    const std::uint64_t vt_bytes = static_cast<std::uint64_t>(kInner) * vtp * 2U;
    I.w_vt.alloc(vt_bytes);
    // the pad columns [S, vtp) must read as zero: zero the buffer when its use changes (the transpose rewrites [0, S))
    if (I.vt_bytes_zeroed != vt_bytes + S) {
        hip_check(hipMemsetAsync(I.w_vt.as<void>(), 0, vt_bytes, s), "vt pad");
        I.vt_bytes_zeroed = vt_bytes + S;
    }
    DevBuf &d_qkv = I.w_qkv, &d_q = I.w_q, &d_k = I.w_k, &d_vt = I.w_vt, &d_ao = I.w_ao, &d_sw = I.w_sw, &d_f1 = I.w_f1;
    std::uint16_t* x = d_x.as<std::uint16_t>();
    std::uint16_t* h = I.w_h.as<std::uint16_t>();
    std::uint16_t* o = I.w_o.as<std::uint16_t>();
    const std::uint32_t* mr = d_mrow.as<std::uint32_t>();
    hip_check(hipStreamSynchronize(s), "sync");
    if (taps) {
        taps->embed_proj_ms = proj_ms;
        taps->host_prep_ms = host_embed_ms + std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ttab).count();
    }
    // ---- blocks (DiT attention: the sparse lever's op when it is on; the refiner never goes through it)
    SparseAccel* sparse = I.acc.sparse;
    DeviceAttentionOp& datt = sparse ? sparse->op() : att;
    lin.reset_flush_count(sv);
    datt.reset_bound_rows(sv);
    const std::size_t nmax_n = static_cast<std::size_t>(blocks.num_blocks()) * 2U * kHeads;
    I.w_nmax.alloc(nmax_n * 4U);
    hip_check(hipMemsetAsync(I.w_nmax.as<void>(), 0, nmax_n * 4U, s), "nmax");
    const std::uint32_t nall = blocks.num_blocks();
    const std::uint32_t nblk = (taps && taps->stop_after_block < nall) ? taps->stop_after_block + 1U : nall;
    fbcv::Action fbc_action = fbcv::Action::kFullNoCommit;
    if (I.acc.fbc) {
        if (nblk != nall) throw std::invalid_argument("device forward: FBC with a partial forward");
        if (!I.fbc) I.fbc = std::make_unique<fbcv::FirstBlockCacheDevice>(S, kHidden);
        I.fbc->begin(x, S, s);  // h0 (throws on a row-count change without reset_accel_run)
    }
    if (sparse) sparse->begin_forward(L.find(SegKind::video).start);
    for (std::uint32_t i = 0; i < nblk; ++i) {
        if (!lin.prepared(i) || I.blocks.count(i) == 0) {
            const auto b = blocks.block(i);
            lin.prepare(b);
            I.block_dev(*b);
        }
        BlockDev& bd = I.blocks.at(i);
        const auto tb0 = std::chrono::steady_clock::now();
        gpu::launch_linear_f64(d_temb.as<float>(), g.basis, nt, bd.ada_w.as<float>(), bd.ada_b.as<float>(), bd.ada_n, g.basis,
            d_ada.as<float>(), sv);
        if (turbo && turbo->has_adaln()) {
            if (ada_out != bd.ada_n) throw std::runtime_error("device forward: Turbo adaLN width");
            launch_add_f32(d_ada.as<float>(), I.w_adelta.as<float>() + static_cast<std::size_t>(i) * nt * ada_out, static_cast<std::uint64_t>(nt) * ada_out, s);
        }
        gpu::launch_mod_vectors(d_ada.as<float>(), rows, mod_ptr, sv);
        // attention half
        gpu::launch_rmsnorm_mod(x, bd.norm1.as<std::uint16_t>(), mr, mod_ptr[1], mod_ptr[0], S, kHidden, kNormEps, h, sv);
        lin.run(i, LinearKind::qkv, h, S, d_qkv.as<std::uint16_t>(), sv);
        if (turbo) turbo->device().apply(lv::Site::kQkv, i, h, kHidden, S, d_qkv.as<std::uint16_t>(), 3U * kInner, s);
        gpu::launch_qk_norm_rope(d_qkv.as<std::uint16_t>(), S, bd.q_norm.as<std::uint16_t>(), bd.k_norm.as<std::uint16_t>(),
            d_rc.as<std::uint16_t>(), d_rs.as<std::uint16_t>(), kNormEps, d_q.as<std::uint16_t>(), d_k.as<std::uint16_t>(), sv,
            I.w_nmax.as<std::uint32_t>() + static_cast<std::size_t>(i) * 2U * kHeads);
        gpu::launch_transpose_v(d_qkv.as<std::uint16_t>(), S, d_vt.as<std::uint16_t>(), vtp, sv);
        datt.run(d_q.as<std::uint16_t>(), d_k.as<std::uint16_t>(), d_vt.as<std::uint16_t>(), vtp, S, S, true, d_ao.as<std::uint16_t>(), sv);
        lin.run(i, LinearKind::out, d_ao.as<std::uint16_t>(), S, o, sv);
        if (turbo) turbo->device().apply(lv::Site::kOut, i, d_ao.as<std::uint16_t>(), kInner, S, o, kHidden, s);
        gpu::launch_gate_residual(x, o, mr, mod_ptr[2], S, kHidden, sv);
        // MLP half
        gpu::launch_rmsnorm_mod(x, bd.norm2.as<std::uint16_t>(), mr, mod_ptr[4], mod_ptr[3], S, kHidden, kNormEps, h, sv);
        if (lin.fused_swiglu()) {
            lin.run(i, LinearKind::fc1, h, S, d_sw.as<std::uint16_t>(), sv);
        } else {
            lin.run(i, LinearKind::fc1, h, S, d_f1.as<std::uint16_t>(), sv);
            if (turbo) turbo->device().apply(lv::Site::kFc1, i, h, kHidden, S, d_f1.as<std::uint16_t>(), 2U * kFfn, s);
            gpu::launch_swiglu(d_f1.as<std::uint16_t>(), S, kFfn, I.silu.as<std::uint16_t>(), d_sw.as<std::uint16_t>(), sv);
        }
        lin.run(i, LinearKind::fc2, d_sw.as<std::uint16_t>(), S, o, sv);
        // step LoRAs: the run-time fc2 branch right after the int8 fc2 GEMM, before the gated residual
        if (turbo && turbo->has_fc2()) turbo->device().apply(lv::Site::kFc2, i, d_sw.as<std::uint16_t>(), kFfn, S, o, kHidden, s);
        gpu::launch_gate_residual(x, o, mr, mod_ptr[5], S, kHidden, sv);
        if (i == 0 && I.acc.fbc) {
            // FirstBlockCache: decide on the block-0 delta; the canonical key-block cut makes the fp64 sums (and so
            // the decision) the same as on two nodes
            const std::vector<fbcv::Sums> ch = I.fbc->after_block0(x, S, s);
            fbcv::Decision dec;
            if (!ch.empty()) {
                const std::vector<std::uint32_t> cuts = key_cuts(S, true);
                const std::uint32_t cut = cuts.size() > 2 ? cuts[1] : S;
                dec = fbcv::decide(fbcv::split_block_sums(ch, S, cut), static_cast<std::uint64_t>(S) * kHidden, I.acc.fbc_threshold);
            }
            fbcv::Decision use = dec;
            if (I.acc.fbc_observe) use.reuse = false;
            fbc_action = I.fbc->resolve(use, x, s);
            if (taps && taps->accel) {
                taps->accel->fbc_cold = ch.empty();
                taps->accel->fbc_score = dec.score;
                taps->accel->fbc_would_reuse = !ch.empty() && dec.reuse;
                taps->accel->fbc_action = static_cast<int>(fbc_action);
            }
            if (fbc_action == fbcv::Action::kReuse) {
                if (sparse) sparse->abandon_forward();  // a reused call is not a counted model call
                break;  // x = h1 + tail = h_last: on to the final layer
            }
        }
        if (taps && taps->block_ms) {
            hip_check(hipStreamSynchronize(s), "sync");
            taps->block_ms->push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tb0).count());
        }
        if (taps && taps->blocks) taps->blocks->push_back(download<std::uint16_t>(x, SH, s));
        if (taps && taps->teacher && i < taps->teacher->size() && (*taps->teacher)[i].size() == SH) {
            hip_check(hipMemcpyAsync(x, (*taps->teacher)[i].data(), SH * 2, hipMemcpyHostToDevice, s), "teacher");
            hip_check(hipStreamSynchronize(s), "sync");
        }
    }
    if (I.acc.fbc && fbc_action == fbcv::Action::kFullCommit) I.fbc->commit(x, S, s);
    if (sparse && !(I.acc.fbc && fbc_action == fbcv::Action::kReuse)) sparse->end_forward(sv);  // throws on LUT contract violations
    hip_check(hipStreamSynchronize(s), "sync");
    const std::uint64_t fl = lin.flush_count(sv);
    if (taps) taps->flushed_groups = fl;
    if (fl && I.strict_flush) throw std::runtime_error("device forward: " + std::to_string(fl) + " flushed rotation groups");
    {
        const std::vector<float> nm = download<float>(I.w_nmax.as<float>(), static_cast<std::size_t>(nblk) * 2U * kHeads, s);
        double worst = 0.0;
        std::uint32_t wb = 0;
        for (std::uint32_t i = 0; i < nblk; ++i)
            for (std::uint32_t hd = 0; hd < kHeads; ++hd) {
                const double v = attn_score_from_sumsq(nm[(static_cast<std::size_t>(i) * 2U) * kHeads + hd],
                    nm[(static_cast<std::size_t>(i) * 2U + 1U) * kHeads + hd]);
                if (!(v <= worst)) {
                    worst = v;
                    wb = i;
                }
            }
        if (taps) {
            taps->attn_score_max = worst;
            taps->attn_score_block = wb;
        }
        if (!(worst <= kAttnScoreLimit))
            throw std::runtime_error("device forward: attention input contract violated in block " + std::to_string(wb) +
                                     ": max||q|| max||k|| log2(e)/sqrt(128) = " + std::to_string(worst) + " > 2^14");
        const std::uint64_t br = datt.bound_rows(sv);
        if (taps) taps->attn_bound_rows = br;
        if (br != 0) throw std::runtime_error("device forward: attention bound_rows telemetry " + std::to_string(br) + " (must be 0)");
    }
    if (nblk < nall) return {};
    // ---- final layer
    const auto tfin = std::chrono::steady_clock::now();
    F32Linear vbank, abank;
    const bool bank = select_heads(g, p, pdd, vbank, abank);
    DevBuf d_vbw, d_vbb, d_abw, d_abb;
    if (bank) {
        d_vbw = upload(vbank.w, s);
        d_vbb = upload(vbank.b, s);
        d_abw = upload(abank.w, s);
        d_abb = upload(abank.b, s);
    }
    DevBuf d_fada(static_cast<std::size_t>(nt) * 2U * kHidden * 4U);
    gpu::launch_linear_f64(d_temb.as<float>(), g.basis, nt, I.final_ada_w.as<float>(), I.final_ada_b.as<float>(), 2U * kHidden, g.basis,
        d_fada.as<float>(), sv);
    if (turbo && turbo->has_adaln()) launch_add_f32(d_fada.as<float>(), I.w_fdelta.as<float>(), static_cast<std::uint64_t>(nt) * 2U * kHidden, s);
    FinalRows fr;
    auto head = [&](SegKind kind, std::uint32_t trow, const DevBuf& w, const DevBuf& bb, std::uint32_t nout, std::vector<float>& out) {
        const Segment& seg = L.find(kind);
        const std::uint32_t m = seg.stop - seg.start;
        DevBuf d_f(static_cast<std::size_t>(m) * kHidden * 4U), d_rows(static_cast<std::size_t>(m) * nout * 4U);
        const float* sh = d_fada.as<float>() + static_cast<std::size_t>(trow) * 2U * kHidden;
        gpu::launch_final_mod(x, seg.start, m, I.final_norm.as<std::uint16_t>(), sh, sh + kHidden, kNormEps, d_f.as<float>(), sv);
        gpu::launch_linear_f64(d_f.as<float>(), kHidden, m, w.as<float>(), bb.as<float>(), nout, kHidden, d_rows.as<float>(), sv);
        out = download<float>(d_rows.as<void>(), static_cast<std::size_t>(m) * nout, s);
    };
    head(SegKind::video, p.video_t_row, bank ? d_vbw : I.vout_w, bank ? d_vbb : I.vout_b, kVideoPatch, fr.video);
    head(SegKind::audio, p.audio_t_row, bank ? d_abw : I.aout_w, bank ? d_abb : I.aout_b, kAudioDim, fr.audio);
    StreamOutputs so;
    const std::vector<float> v = unpatchify_video(fr.video.data(), L.latent_t, L.latent_h, L.latent_w);
    const std::vector<float> a = unpack_audio(fr.audio.data(), L.audio_t);
    so.video_neg_v.resize(v.size());
    so.audio_neg_v.resize(a.size());
    for (std::size_t i = 0; i < v.size(); ++i) so.video_neg_v[i] = tob(-v[i]);
    for (std::size_t i = 0; i < a.size(); ++i) so.audio_neg_v[i] = tob(-a[i]);
    if (taps) taps->final_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tfin).count();
    if (taps && taps->final_rows) *taps->final_rows = std::move(fr);
    return so;
}

std::vector<std::uint16_t> DeviceForward::refine_text(const std::uint16_t* text, std::uint32_t L, DeviceAttentionOp& att) {
    Impl& I = *impl_;
    I.prepare_refiner();
    hipStream_t s = I.stream;
    void* sv = static_cast<void*>(s);
    const std::size_t LH = static_cast<std::size_t>(L) * kHidden;
    DevBuf d_t(static_cast<std::size_t>(L) * kTextDim * 2U), d_x(LH * 2U), d_h(LH * 2U), d_o(LH * 2U),
        d_qkv(static_cast<std::size_t>(L) * 3U * kInner * 2U), d_q(static_cast<std::size_t>(L) * kInner * 2U),
        d_k(static_cast<std::size_t>(L) * kInner * 2U), d_ao(static_cast<std::size_t>(L) * kInner * 2U),
        d_f1(static_cast<std::size_t>(L) * 2U * kFfn * 2U), d_sw(static_cast<std::size_t>(L) * kFfn * 2U);
    const std::size_t rn = I.refiner.size() * 2U * kHeads;
    DevBuf d_nmax(rn * 4U);
    hip_check(hipMemsetAsync(d_nmax.as<void>(), 0, rn * 4U, s), "nmax");
    std::size_t rb = 0;
    att.reset_bound_rows(sv);
    const std::uint64_t vtp = vt_pitch_for(L);
    DevBuf d_vt(static_cast<std::size_t>(kInner) * vtp * 2U);
    hip_check(hipMemsetAsync(d_vt.as<void>(), 0, static_cast<std::size_t>(kInner) * vtp * 2U, s), "vt pad");
    hip_check(hipMemcpyAsync(d_t.as<void>(), text, static_cast<std::size_t>(L) * kTextDim * 2U, hipMemcpyHostToDevice, s), "text");
    std::uint16_t* x = d_x.as<std::uint16_t>();
    std::uint16_t* h = d_h.as<std::uint16_t>();
    std::uint16_t* o = d_o.as<std::uint16_t>();
    gpu::launch_bf16_linear(d_t.as<std::uint16_t>(), L, I.cond_w.as<std::uint16_t>(), I.cond_b.as<std::uint16_t>(), kHidden, kTextDim, x, sv);
    const TurboV4* turbo = I.acc.turbo;
    std::uint32_t ri = 0;
    auto branch = [&](lv::Site site, const std::uint16_t* xin, std::uint64_t x_ld, std::uint16_t* y, std::uint64_t y_ld) {
        if (turbo) turbo->device().apply(site, ri, xin, x_ld, L, y, y_ld, s);
    };
    for (const RefinerDev& b : I.refiner) {
        gpu::launch_rmsnorm_mod(x, b.norm1.as<std::uint16_t>(), nullptr, nullptr, nullptr, L, kHidden, kNormEps, h, sv);
        gpu::launch_bf16_linear(h, L, b.qkv.as<std::uint16_t>(), nullptr, 3U * kInner, kHidden, d_qkv.as<std::uint16_t>(), sv);
        branch(lv::Site::kRefQkv, h, kHidden, d_qkv.as<std::uint16_t>(), 3U * kInner);
        gpu::launch_qk_norm_rope(d_qkv.as<std::uint16_t>(), L, b.q_norm.as<std::uint16_t>(), b.k_norm.as<std::uint16_t>(), nullptr, nullptr,
            kNormEps, d_q.as<std::uint16_t>(), d_k.as<std::uint16_t>(), sv, d_nmax.as<std::uint32_t>() + 2U * kHeads * rb++);
        gpu::launch_transpose_v(d_qkv.as<std::uint16_t>(), L, d_vt.as<std::uint16_t>(), vtp, sv);
        att.run(d_q.as<std::uint16_t>(), d_k.as<std::uint16_t>(), d_vt.as<std::uint16_t>(), vtp, L, L, false, d_ao.as<std::uint16_t>(), sv);
        gpu::launch_bf16_linear(d_ao.as<std::uint16_t>(), L, b.out.as<std::uint16_t>(), nullptr, kHidden, kInner, o, sv);
        branch(lv::Site::kRefOut, d_ao.as<std::uint16_t>(), kInner, o, kHidden);
        gpu::launch_add_residual(x, o, LH, sv);
        gpu::launch_rmsnorm_mod(x, b.norm2.as<std::uint16_t>(), nullptr, nullptr, nullptr, L, kHidden, kNormEps, h, sv);
        gpu::launch_bf16_linear(h, L, b.fc1.as<std::uint16_t>(), nullptr, 2U * kFfn, kHidden, d_f1.as<std::uint16_t>(), sv);
        branch(lv::Site::kRefFc1, h, kHidden, d_f1.as<std::uint16_t>(), 2U * kFfn);
        gpu::launch_swiglu(d_f1.as<std::uint16_t>(), L, kFfn, I.silu.as<std::uint16_t>(), d_sw.as<std::uint16_t>(), sv);
        gpu::launch_bf16_linear(d_sw.as<std::uint16_t>(), L, b.fc2.as<std::uint16_t>(), nullptr, kHidden, kFfn, o, sv);
        branch(lv::Site::kRefFc2, d_sw.as<std::uint16_t>(), kFfn, o, kHidden);
        gpu::launch_add_residual(x, o, LH, sv);
        ++ri;
    }
    gpu::launch_rmsnorm_mod(x, I.ref_final_norm.as<std::uint16_t>(), nullptr, nullptr, nullptr, L, kHidden, kNormEps, h, sv);
    std::vector<std::uint16_t> out = download<std::uint16_t>(h, LH, s);
    const std::vector<float> nm = download<float>(d_nmax.as<float>(), rn, s);
    for (std::size_t i = 0; i < I.refiner.size(); ++i)
        for (std::uint32_t hd = 0; hd < kHeads; ++hd) {
            const double v = attn_score_from_sumsq(nm[i * 2U * kHeads + hd], nm[i * 2U * kHeads + kHeads + hd]);
            last_refiner_score_ = std::max(v, i == 0 && hd == 0 ? 0.0 : last_refiner_score_);
            if (!(v <= kAttnScoreLimit))
                throw std::runtime_error("token refiner: attention input contract violated in block " + std::to_string(i) +
                                         ": " + std::to_string(v) + " > 2^14");
        }
    const std::uint64_t br = att.bound_rows(sv);
    if (br != 0) throw std::runtime_error("token refiner: attention bound_rows telemetry " + std::to_string(br) + " (must be 0)");
    return out;
}

}  // namespace strix::h3::block
