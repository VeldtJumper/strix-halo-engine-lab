// GLM-5.3-Flash engine (see include/strix/glm/engine.hpp, design notes).
//
// Skeleton v1: correctness first. Per layer: mHC pre (Tier A) -> RMSNorm -> attention seam -> TP reduction -> mHC
// expand -> mHC pre -> RMSNorm -> dense MLP (strix_gemv) or router (Tier A) + MoE seam -> reduction -> mHC expand. Head:
// final collapse -> RMSNorm -> lm_head vocab slice (strix_gemv) -> local argmax -> argmax exchange. The default seams
// process one row at a time with the Tier A kernels (bitwise to src/glm/host) and strix_gemv (Tier B reference); the
// Optimized implementations replace them with fused / batched paths that must reproduce these bits.
// Weights: O_DIRECT reads through one pinned staging buffer (MiniMax DirectLoader pattern), rank slices per
// tools/glm/split_plan.py (H128-aligned EXL3 slices, plain tensors by rows).
#include "strix/glm/engine.hpp"

#include "glm/engine/glm_fast_attn.hpp"
#include "glm/engine/glm_fast_moe.hpp"
#include "glm/engine/glm_lib.hpp"
#include "glm/engine/glm_pack.hpp"
#include "glm/engine/glm_r1b.hpp"
#include "glm/kernels/glm_dsa.hpp"
#include "glm/kernels/glm_tier_a.hpp"
#include "glm/prefill/glm_prefill.hpp"
#include "glm/spec/glm_spec_dev.hpp"
#include "glm/xfer/glm_post_fast.hpp"
#include "glm/xfer/glm_xfer_exchange.hpp"
#include "strix/dist/communicator.hpp"

#include <strix_trellis.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>

namespace strix::glm::engine {

namespace {

using namespace detail;
namespace D = strix::glm::dev;
using Clock = std::chrono::steady_clock;

const std::string kP = "model.language_model.";

// ---------------------------------------------------------------- small engine-only kernels
__global__ void embed4_kernel(const std::uint16_t* table, const std::uint32_t* tokens, float* X) {
    const unsigned r = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
    if (d >= kHidden) return;
    const float v = bf16_to_float(table[static_cast<std::size_t>(tokens[r]) * kHidden + d]);
    for (unsigned c = 0; c < kHc; ++c) X[(static_cast<std::size_t>(r) * kHc + c) * kHidden + d] = v;  // 4 identical streams
}
__global__ void embed1_kernel(const std::uint16_t* table, const std::uint32_t* tokens, float* e) {
    const unsigned r = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
    if (d < kHidden) e[static_cast<std::size_t>(r) * kHidden + d] = bf16_to_float(table[static_cast<std::size_t>(tokens[r]) * kHidden + d]);
}
__global__ void cat2_kernel(const float* a, const float* b, float* y) {  // y[r] = [a[r] | b[r]], 2 x 4096
    const unsigned r = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
    if (d >= kHidden) return;
    y[static_cast<std::size_t>(r) * 2 * kHidden + d] = a[static_cast<std::size_t>(r) * kHidden + d];
    y[static_cast<std::size_t>(r) * 2 * kHidden + kHidden + d] = b[static_cast<std::size_t>(r) * kHidden + d];
}
__global__ void add_kernel(const float* a, const float* b, float* y, std::size_t n) {  // residual + branch (MTP, no mHC)
    const std::size_t i = blockIdx.x * std::size_t{256} + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}
// Drafter confidence: max M and Z = sum exp(l - M) of one row's rank slice (fixed tree order: identical per rank).
__global__ __launch_bounds__(256) void lse_kernel(const float* l, unsigned n, float* mz) {
    __shared__ float s[256];
    const unsigned t = threadIdx.x;
    float m = -__builtin_huge_valf();
    for (unsigned i = t; i < n; i += 256) m = fmaxf(m, l[i]);
    s[t] = m;
    __syncthreads();
    for (unsigned h = 128; h > 0; h >>= 1U) {
        if (t < h) s[t] = fmaxf(s[t], s[t + h]);
        __syncthreads();
    }
    const float M = s[0];
    __syncthreads();
    float z = 0.0F;
    for (unsigned i = t; i < n; i += 256) z += expf(l[i] - M);
    s[t] = z;
    __syncthreads();
    for (unsigned h = 128; h > 0; h >>= 1U) {
        if (t < h) s[t] = s[t] + s[t + h];
        __syncthreads();
    }
    if (t == 0) {
        mz[0] = M;
        mz[1] = s[0];
    }
}
__global__ void sum2_kernel(const float* p0, const float* p1, float* y, std::size_t n) {  // R17: p0 + p1, rank order
    const std::size_t i = blockIdx.x * std::size_t{256} + threadIdx.x;
    if (i < n) y[i] = p0[i] + p1[i];
}
// Staged prefill exchange: the partial goes into the communicator's coherent TX view on the compute stream.
__global__ void stage_copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst, std::size_t n4) {
    const std::size_t i = blockIdx.x * std::size_t{256} + threadIdx.x;
    if (i < n4) dst[i] = src[i];
}
// STRIX_GLM_PF_STAGED (default on, =0 off): pipelined prefill exchanges through the communicator's coherent staging
// views (one copy kernel into TX, sum2 reads RX in place) instead of the async chunked D2H / H2D copies, which a side
// stream only gets between the long prefill kernels. Same p0 + p1 adds in rank order, same bits.
bool pf_staged_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRIX_GLM_PF_STAGED");
        if (v == nullptr || *v == '\0') return true;
        const std::string s(v);
        if (s != "0" && s != "1") throw std::invalid_argument("STRIX_GLM_PF_STAGED must be 0 or 1");
        return s == "1";
    }();
    return on;
}

// ---------------------------------------------------------------- default seams (bitwise references)
struct Scratch {  // per-row scratch shared by the default seams (one row at a time)
    float *qkv, *f, *g, *b, *t128a, *t128b, *kda_scr, *beta, *o, *y1;
    float *qa, *qr, *q, *kva, *lat, *qI, *wh, *scores, *qt, *key, *gate;
    std::uint32_t *work, *tokens, *ntok;
    D::DsaSlot* slot_d;
    unsigned* t_d;
    float *eo, *ge, *ue, *ae, *sh;
    std::uint64_t cap;
};

class DefaultAttn final : public AttnPath {
public:
    explicit DefaultAttn(std::shared_ptr<Scratch> s) : s_(std::move(s)) {}
    void kda(const KdaLayerWeights& w, KdaState& st, const CallCtx& c, const float* x, float* y) override {
        if (c.spec) throw std::logic_error("speculative verify runs on the FastAttn seam (STRIX_GLM_ATTN=ref is plain only)");
        const unsigned H = w.heads, C = H * kKdaDim;
        const D::KdaDev kd{H, w.conv, w.A_log, w.dt_bias, w.o_norm};
        for (std::size_t r = 0; r < c.rows.size(); ++r) {
            const float* xr = x + r * kHidden;
            const std::uint32_t slot = c.rows[r].slot;
            for (unsigned p = 0; p < 3; ++p) c.lin->run(w.qkv[p], xr, s_->qkv + p * C, 1, c.stream);
            if (strix::glm::r1b::active()) {  // R1b generated identity (default; prefill scopes R1): the K = 4096 chains blocked; f_b / g_b (K 128) unchanged
                const int selected = strix::glm::r1b::dense_variant();
                const bool packed = (selected == 5 || selected == 6 || selected == 7) && w.f_a_packed != nullptr &&
                                    w.g_a_packed != nullptr && w.b_packed != nullptr;
                const int variant = selected == 5 || selected == 6 || selected == 7 ? (packed ? selected : 2) : selected;
                strix::glm::r1b::dense_f16_k4096(xr, kHidden, packed ? w.f_a_packed : w.f_a,
                                               kKdaDim, s_->t128a, kKdaDim, 1, c.stream, variant);
                strix::glm::r1b::dense_f16_k4096(xr, kHidden, packed ? w.g_a_packed : w.g_a,
                                               kKdaDim, s_->t128b, kKdaDim, 1, c.stream, variant);
                strix::glm::r1b::dense_f16_k4096(xr, kHidden, packed ? w.b_packed : w.b,
                                               H, s_->b, H, 1, c.stream, variant);
            } else {
                D::dense(xr, w.f_a, D::WType::F16, kHidden, 0, kKdaDim, s_->t128a, 1, c.stream);
                D::dense(xr, w.g_a, D::WType::F16, kHidden, 0, kKdaDim, s_->t128b, 1, c.stream);
                D::dense(xr, w.b, D::WType::F16, kHidden, 0, H, s_->b, 1, c.stream);
            }
            D::dense(s_->t128a, w.f_b, D::WType::F16, kKdaDim, 0, C, s_->f, 1, c.stream);
            D::dense(s_->t128b, w.g_b, D::WType::F16, kKdaDim, 0, C, s_->g, 1, c.stream);
            float* S = st.S + static_cast<std::size_t>(slot) * H * kKdaDim * kKdaDim;
            float* cv = st.conv + static_cast<std::size_t>(slot) * 3 * C * 3;
            D::kda_step(kd, s_->qkv, s_->f, s_->g, s_->b, S, cv, s_->kda_scr, s_->beta, s_->o, s_->y1, c.flag, 1, c.stream);
            c.lin->run(w.o, s_->y1, y + r * kHidden, 1, c.stream);
        }
    }
    void dsa(const DsaLayerWeights& w, DsaState& st, const CallCtx& c, const float* x, float* y) override {
        if (c.spec) throw std::logic_error("speculative verify runs on the FastAttn seam (STRIX_GLM_ATTN=ref is plain only)");
        const unsigned H = w.heads;
        const D::IndexerDev iw{w.wk, w.gate, w.weights_proj, w.k_norm_w, w.k_norm_b, w.ape};
        for (std::size_t r = 0; r < c.rows.size(); ++r) {
            const float* xr = x + r * kHidden;
            const std::uint32_t slot = c.rows[r].slot;
            const auto t = static_cast<unsigned>(c.rows[r].pos);
            if (t >= st.cap) throw std::out_of_range("DSA position beyond the slot capacity");
            c.lin->run(w.q_a, xr, s_->qa, 1, c.stream);
            D::rmsnorm(s_->qa, w.q_a_norm, D::WType::BF16, kQLora, kEps, s_->qr, 1, c.stream);
            c.lin->run(w.q_b, s_->qr, s_->q, 1, c.stream);
            c.lin->run(w.kv_a, xr, s_->kva, 1, c.stream);
            D::rmsnorm(s_->kva, w.kv_a_norm, D::WType::BF16, kKvLora, kEps, s_->lat, 1, c.stream);
            c.lin->run(w.wq_b, s_->qr, s_->qI, 1, c.stream);
            const D::DsaSlot sl{st.latent + static_cast<std::size_t>(slot) * st.cap * kKvLora,
                                st.ikey + static_cast<std::size_t>(slot) * st.cap * kIdxDim,
                                st.igate + static_cast<std::size_t>(slot) * st.cap * kIdxDim,
                                st.pooled + static_cast<std::size_t>(slot) * (st.cap / kKpool) * kIdxDim, static_cast<unsigned>(st.cap)};
            hip_ok(hipMemcpyAsync(s_->slot_d, &sl, sizeof(sl), hipMemcpyHostToDevice, c.stream), "slot");
            hip_ok(hipMemcpyAsync(s_->t_d, &t, sizeof(t), hipMemcpyHostToDevice, c.stream), "t");
            D::dsa_append(s_->slot_d, s_->t_d, s_->lat, xr, iw, s_->key, s_->gate, c.flag, 1, c.stream);
            D::indexer_select(sl, t, s_->qI, xr, iw, s_->wh, s_->scores, s_->work, s_->tokens, s_->ntok, c.stream);
            D::mla_attention(sl, s_->tokens, s_->ntok, s_->q, w.kv_b, H, s_->qt, s_->o, c.stream);
            c.lin->run(w.o, s_->o, y + r * kHidden, 1, c.stream);
            hip_ok(hipStreamSynchronize(c.stream), "dsa row");  // host-to-device argument copies above are reused per row
        }
    }

private:
    std::shared_ptr<Scratch> s_;
};

class DefaultMoe final : public MoePath {
public:
    explicit DefaultMoe(std::shared_ptr<Scratch> s) : s_(std::move(s)) {}
    void forward(const MoeLayerWeights& w, const CallCtx& c, const float* x, const std::uint32_t* ids, const float* wts,
                 float* y) override {
        const std::size_t R = c.rows.size();
        std::vector<std::uint32_t> hid(R * kTopE);
        hip_ok(hipMemcpyAsync(hid.data(), ids, hid.size() * 4, hipMemcpyDeviceToHost, c.stream), "ids");
        hip_ok(hipStreamSynchronize(c.stream), "ids sync");
        const unsigned I = w.shared.down.raw.input;  // I_local
        for (std::size_t r = 0; r < R; ++r) {
            const float* xr = x + r * kHidden;
            for (unsigned k = 0; k < kTopE; ++k) mlp(w.experts.at(hid[r * kTopE + k]), xr, I, s_->eo + static_cast<std::size_t>(k) * kHidden, c);
            mlp(w.shared, xr, I, s_->sh, c);
            D::moe_combine(wts + r * kTopE, s_->eo, s_->sh, y + r * kHidden, 1, c.stream);
        }
    }
    void mlp(const Mlp& m, const float* x, unsigned I, float* out, const CallCtx& c) {
        if (m.gate.prepared == nullptr) throw std::logic_error("DefaultMoe needs prepared experts (loaded raw-only for the fast MoE seam)");
        c.lin->run(m.gate, x, s_->ge, 1, c.stream);
        c.lin->run(m.up, x, s_->ue, 1, c.stream);
        D::swiglu(s_->ge, s_->ue, s_->ae, I, c.stream);
        c.lin->run(m.down, s_->ae, out, 1, c.stream);
    }

private:
    std::shared_ptr<Scratch> s_;
};

class ReferencePrefill final : public PrefillPath {  // never called: Engine::prefill runs decode steps row by row
public:
    void kda(const KdaLayerWeights&, KdaState&, const CallCtx&, const float*, float*) override { throw std::logic_error("reference prefill = decode"); }
    void dsa(const DsaLayerWeights&, DsaState&, const CallCtx&, const float*, float*) override { throw std::logic_error("reference prefill = decode"); }
    void moe(const MoeLayerWeights&, const CallCtx&, const float*, const std::uint32_t*, const float*, float*) override {
        throw std::logic_error("reference prefill = decode");
    }
    void dense(const Mlp&, const CallCtx&, const float*, unsigned, float*) override { throw std::logic_error("reference prefill = decode"); }
};

// Placeholder for Options::external_exchange until the caller's set_exchange: any use is an error.
class MissingExchange final : public Exchange {
public:
    void reduce(float*, unsigned, hipStream_t) override { throw std::logic_error("external exchange not installed (set_exchange)"); }
    void argmax(float*, std::uint32_t*, unsigned, hipStream_t) override { throw std::logic_error("external exchange not installed (set_exchange)"); }
};

class LocalExchange final : public Exchange {
public:
    void reduce(float*, unsigned, hipStream_t) override {}
    void argmax(float*, std::uint32_t*, unsigned, hipStream_t) override {}
    void merge(MergeRow*, unsigned, MergeCtl&, hipStream_t) override {}  // one rank: already merged
};

// TP2 over the communicator: send this rank's partial, receive the peer's, y = p0 + p1 in rank order on both ranks.
class CommExchange final : public Exchange {
public:
    CommExchange(dist::Communicator& comm, std::uint32_t rank, unsigned max_rows)
        : comm_(comm), rank_(rank), capacity_(max_rows), remote_(dalloc<float>(static_cast<std::size_t>(max_rows) * kHidden)),
          am_l_(dalloc<std::uint64_t>(max_rows)), am_r_(dalloc<std::uint64_t>(max_rows)),
          merge_l_(dalloc<std::uint8_t>(kMergeBytes)), merge_r_(dalloc<std::uint8_t>(kMergeBytes)) {}
    ~CommExchange() override {
        for (auto e : source_free_) if (e) (void)hipEventDestroy(e);
        for (auto e : remote_last_use_) if (e) (void)hipEventDestroy(e);
        (void)hipFree(remote_);
        (void)hipFree(am_l_);
        (void)hipFree(am_r_);
        (void)hipFree(merge_l_);
        (void)hipFree(merge_r_);
    }
    void resize_rows(unsigned rows) {
        if (rows == capacity_) return;
        hip_ok(hipDeviceSynchronize(), "exchange resize sync");
        (void)hipFree(remote_);
        (void)hipFree(am_l_);
        (void)hipFree(am_r_);
        remote_ = nullptr; am_l_ = am_r_ = nullptr;
        remote_ = dalloc<float>(static_cast<std::size_t>(rows) * kHidden);
        am_l_ = dalloc<std::uint64_t>(rows);
        am_r_ = dalloc<std::uint64_t>(rows);
        capacity_ = rows;
    }
    void reduce(float* p, unsigned rows, hipStream_t s) override {
        dist::Tag tag;
        tag.kind = static_cast<std::uint32_t>(dist::TagKind::half_sum);
        tag.request = seq_++;
        const std::size_t n = static_cast<std::size_t>(rows) * kHidden;
        comm_.exchange(s, p, remote_, n * sizeof(float), tag);
        const float* p0 = rank_ == 0 ? p : remote_;
        const float* p1 = rank_ == 0 ? remote_ : p;
        sum2_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(p0, p1, p, n);
        hip_ok(hipGetLastError(), "sum2");
    }
    void start_reduce(float* p, unsigned rows, hipStream_t s) override {
        start_reduce_region(p, rows, 0, -1, s);
    }
    // STRIX_GLM_PF_STAGED: staged and chunked framings do not interoperate, so both ranks vote once (whatever their
    // environment) before the first pipelined layer and use the staging views only if both offer them.
    void agree_staged(hipStream_t s, bool async_start) {
        if (staged_vote_ >= 0) return;
        const std::size_t probe = static_cast<std::size_t>(capacity_ / 2U + 1U) * kHidden * sizeof(float);
        const std::uint64_t mine = async_start && pf_staged_enabled() && comm_.send_staging(std::min(probe, comm_.max_bytes())) != nullptr ? 1U : 0U;
        std::uint64_t peer = 0;
        dist::Tag tag;
        tag.kind = static_cast<std::uint32_t>(dist::TagKind::control);
        tag.request = seq_++;
        hip_ok(hipMemcpyAsync(am_l_, &mine, sizeof(mine), hipMemcpyHostToDevice, s), "staged vote");
        comm_.exchange(s, am_l_, am_r_, sizeof(mine), tag);
        hip_ok(hipMemcpyAsync(&peer, am_r_, sizeof(peer), hipMemcpyDeviceToHost, s), "staged vote peer");
        hip_ok(hipStreamSynchronize(s), "staged vote sync");
        staged_vote_ = mine == 1U && peer == 1U ? 1 : 0;
        std::fprintf(stderr, "rank %u prefill exchange: %s (local %u, peer %u)\n", rank_,
                     staged_vote_ == 1 ? "staging views" : "chunked copies", static_cast<unsigned>(mine), static_cast<unsigned>(peer));
    }
    // Only the N=2 opt-in schedule supplies a half and a disjoint row region.
    void wait_source(unsigned half, hipStream_t s) {
        if (source_pending_[half]) {
            hip_ok(hipStreamWaitEvent(s, source_free_[half], 0), "prefill source lifetime");
            source_pending_[half] = false;
        }
    }
    void start_reduce_region(float* p, unsigned rows, unsigned offset, int half, hipStream_t s) {
        if (pending_part_ != nullptr) throw std::logic_error("prefill reduction already in flight");
        dist::Tag tag;
        tag.kind = static_cast<std::uint32_t>(dist::TagKind::half_sum);
        tag.request = seq_++;
        pending_remote_ = remote_ + static_cast<std::size_t>(offset) * kHidden;
        pending_half_ = half;
        const std::size_t bytes = static_cast<std::size_t>(rows) * kHidden * sizeof(float);
        if (half >= 0 && staged_vote_ == 1) {
            // Both ranks agreed on the staging views; a per-size refusal here (above max_bytes, a direct route) is
            // rank-symmetric because the old chunked path depends on the same sizes and routes.
            if (auto* tx = static_cast<float4*>(comm_.send_staging(bytes))) {
                const std::size_t n4 = bytes / sizeof(float4);
                stage_copy_kernel<<<static_cast<unsigned>((n4 + 255) / 256), 256, 0, s>>>(reinterpret_cast<const float4*>(p), tx, n4);
                hip_ok(hipGetLastError(), "prefill stage copy");
                comm_.start_staged(s, bytes, tag);
                staged_ = true;
                pending_part_ = p;
                pending_rows_ = rows;
                return;
            }
        }
        if (half >= 0) {
            if (!source_free_[half]) hip_ok(hipEventCreateWithFlags(&source_free_[half], hipEventDisableTiming), "prefill source event");
            if (!remote_last_use_[half]) {
                hip_ok(hipEventCreateWithFlags(&remote_last_use_[half], hipEventDisableTiming), "prefill remote event");
                hip_ok(hipEventRecord(remote_last_use_[half], s), "prefill initial remote lifetime");
            }
            source_pending_[half] = comm_.start_async(s, p, pending_remote_, bytes, tag, remote_last_use_[half], source_free_[half]);
        } else comm_.start(s, p, bytes, tag);
        pending_part_ = p;
        pending_rows_ = rows;
    }
    void finish_reduce(hipStream_t s) override {
        if (pending_part_ == nullptr) throw std::logic_error("prefill reduction not in flight");
        float* p = pending_part_;
        const std::size_t n = static_cast<std::size_t>(pending_rows_) * kHidden;
        pending_part_ = nullptr;  // a failed finish stops the communicator; do not retry it during unwinding
        if (staged_) {
            staged_ = false;
            // RX is read in place; release_staged fences its reuse after this sum2.
            const auto* rx = static_cast<const float*>(comm_.finish_staged(s));
            const float* p0 = rank_ == 0 ? p : rx;
            const float* p1 = rank_ == 0 ? rx : p;
            sum2_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(p0, p1, p, n);
            hip_ok(hipGetLastError(), "sum2 prefill staged");
            comm_.release_staged(s);
            return;
        }
        // finish waits for checked peer bytes. Each half's last-use event protects
        // its remote region from the next off-stream H2D; pinned staging is owned by comm_.
        comm_.finish(s, pending_remote_);
        const float* p0 = rank_ == 0 ? p : pending_remote_;
        const float* p1 = rank_ == 0 ? pending_remote_ : p;
        sum2_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(p0, p1, p, n);
        hip_ok(hipGetLastError(), "sum2 prefill pipeline");
        if (pending_half_ >= 0) hip_ok(hipEventRecord(remote_last_use_[pending_half_], s), "prefill remote last use");
    }
    void complete(hipStream_t s) override {
        if (pending_part_ != nullptr) finish_reduce(s);
    }
    void argmax(float* value, std::uint32_t* id, unsigned rows, hipStream_t s) override {
        std::vector<float> v(rows);
        std::vector<std::uint32_t> i(rows);
        std::vector<std::uint64_t> packed(rows), peer(rows);
        hip_ok(hipMemcpyAsync(v.data(), value, rows * 4, hipMemcpyDeviceToHost, s), "am v");
        hip_ok(hipMemcpyAsync(i.data(), id, rows * 4, hipMemcpyDeviceToHost, s), "am i");
        hip_ok(hipStreamSynchronize(s), "am sync");
        for (unsigned r = 0; r < rows; ++r) {
            std::uint32_t vb;
            std::memcpy(&vb, &v[r], 4);
            packed[r] = (static_cast<std::uint64_t>(i[r]) << 32U) | vb;
        }
        hip_ok(hipMemcpy(am_l_, packed.data(), rows * 8, hipMemcpyHostToDevice), "am up");
        dist::Tag tag;
        tag.kind = static_cast<std::uint32_t>(dist::TagKind::argmax);
        tag.request = seq_++;
        comm_.exchange(s, am_l_, am_r_, rows * 8, tag);
        hip_ok(hipMemcpyAsync(peer.data(), am_r_, rows * 8, hipMemcpyDeviceToHost, s), "am down");
        hip_ok(hipStreamSynchronize(s), "am sync2");
        for (unsigned r = 0; r < rows; ++r) {
            float pv;
            const auto pb = static_cast<std::uint32_t>(peer[r]);
            std::memcpy(&pv, &pb, 4);
            const auto pi = static_cast<std::uint32_t>(peer[r] >> 32U);
            if (pv > v[r] || (pv == v[r] && pi < i[r])) {  // R18: larger value, ties -> lower id
                v[r] = pv;
                i[r] = pi;
            }
        }
        hip_ok(hipMemcpy(value, v.data(), rows * 4, hipMemcpyHostToDevice), "am v up");
        hip_ok(hipMemcpy(id, i.data(), rows * 4, hipMemcpyHostToDevice), "am i up");
    }
    // design notes section 2: one exchange of [ctl][rows] per call; both ranks merge the same two halves in rank order.
    void merge(MergeRow* rows, unsigned n, MergeCtl& ctl, hipStream_t s) override {
        if (n > 8) throw std::invalid_argument("merge: at most eight rows");
        const std::size_t bytes = sizeof(MergeCtl) + static_cast<std::size_t>(n) * sizeof(MergeRow);
        std::vector<std::uint8_t> mine(bytes), peer(bytes);
        std::memcpy(mine.data(), &ctl, sizeof(MergeCtl));
        std::memcpy(mine.data() + sizeof(MergeCtl), rows, n * sizeof(MergeRow));
        hip_ok(hipMemcpy(merge_l_, mine.data(), bytes, hipMemcpyHostToDevice), "merge up");
        dist::Tag tag;
        tag.kind = static_cast<std::uint32_t>(dist::TagKind::argmax);
        tag.request = seq_++;
        comm_.exchange(s, merge_l_, merge_r_, bytes, tag);
        hip_ok(hipMemcpyAsync(peer.data(), merge_r_, bytes, hipMemcpyDeviceToHost, s), "merge down");
        hip_ok(hipStreamSynchronize(s), "merge sync");
        MergeCtl pc;
        std::memcpy(&pc, peer.data(), sizeof(MergeCtl));
        std::array<MergeRow, 8> peer_rows{};
        std::memcpy(peer_rows.data(), peer.data() + sizeof(MergeCtl), n * sizeof(MergeRow));
        spec::merge_rank_order({rows, n}, ctl, {peer_rows.data(), n}, pc, rank_);
    }

private:
    dist::Communicator& comm_;
    std::uint32_t rank_;
    unsigned capacity_;
    float* remote_;
    static_assert(sizeof(MergeRow) == 12);
    static_assert(sizeof(MergeCtl) == 72);
    static constexpr std::size_t kMergeBytes = sizeof(MergeCtl) + 8 * sizeof(MergeRow);
    std::uint64_t *am_l_, *am_r_;
    std::uint8_t *merge_l_, *merge_r_;
    std::uint64_t seq_ = 0;
    float* pending_part_ = nullptr;
    unsigned pending_rows_ = 0;
    float* pending_remote_ = nullptr;
    int pending_half_ = -1;
    hipEvent_t source_free_[2] = {}, remote_last_use_[2] = {};
    bool source_pending_[2] = {};
    bool staged_ = false;  // the in-flight reduction uses the staging views (STRIX_GLM_PF_STAGED)
    int staged_vote_ = -1;  // -1 not yet agreed, 0 chunked copies, 1 staging views (both ranks)
};

// ---------------------------------------------------------------- layer weights
struct HcSite {
    float *fn = nullptr, *base = nullptr, *scale = nullptr;
};
struct LayerW {
    unsigned index = 0;
    bool kda = false, moe = false, mtp = false;
    HcSite hc_attn, hc_ffn;
    const std::uint16_t *in_norm = nullptr, *post_norm = nullptr;
    KdaLayerWeights kdaw;
    DsaLayerWeights dsaw;
    MoeLayerWeights moew;
    Mlp dense;
    KdaState kst;
    DsaState dst;
    // MTP only
    Linear eh_proj;
    const std::uint16_t *enorm = nullptr, *hnorm = nullptr, *head_norm = nullptr;
};

}  // namespace

void Exchange::merge(MergeRow*, unsigned, MergeCtl&, hipStream_t) {
    throw std::logic_error("this Exchange does not implement the speculative merge (design notes section 2)");
}

std::shared_ptr<PrefillPath> reference_prefill() { return std::make_shared<ReferencePrefill>(); }
std::shared_ptr<Exchange> local_exchange() { return std::make_shared<LocalExchange>(); }

struct Engine::Impl {
    // Declared first = destroyed last: the memory watchdog keeps running through the HIP teardown of a failed load.
    mem::Probe probe;
    std::unique_ptr<mem::Monitor> monitor;
    Options opt;
    dist::Communicator* comm;
    PackReader pack;
    Lib lib;
    std::shared_ptr<Scratch> scr = std::make_shared<Scratch>();
    std::shared_ptr<AttnPath> attn;
    std::shared_ptr<MoePath> moe;
    std::shared_ptr<Exchange> xch;          // active for the entire forward, including its head / merge
    std::shared_ptr<Exchange> fallback_xch; // installed seam (CommExchange unless explicitly overridden)
    std::shared_ptr<Exchange> decode_xch;   // borrowed transport; only automatic plain decode uses this
    std::shared_ptr<PrefillPath> pf;
    bool loaded = false;        // load() finished
    bool pf_installed = false;  // pf came from Engine::set_prefill (not the default reference path)
    std::vector<LayerW> layers;
    std::uint16_t* embed = nullptr;
    const std::uint16_t* final_norm = nullptr;
    Linear head;
    std::uint32_t vocab0 = 0, vocab_n = 0;
    unsigned kda_heads = 0, mla_heads = 0, expert_i = 0, dense_i = 0;
    // activations
    float* hz = nullptr;                // [rows][24] mHC chains of a prefill chunk (pf::hc_pre_rows)
    float* hrr = nullptr;               // [rows] next-site norm factors of a prefill chunk (STRIX_GLM_PF_HC_FUSE)
    std::vector<unsigned char> no_mtp;  // per slot: chunk-prefilled, MTP cache not filled
    int null_layer = -1;  // set_null_flip (test hook)
    std::uint64_t null_pos = 0;
    unsigned null_elem = 0;
    float *X = nullptr, *X2 = nullptr, *F = nullptr, *mix = nullptr, *xc = nullptr, *xn = nullptr, *part = nullptr;
    float *logits = nullptr, *rlog = nullptr, *wts = nullptr, *hfin = nullptr, *hidden = nullptr, *amv = nullptr;
    float *dg = nullptr, *du = nullptr, *da = nullptr;
    std::uint32_t *ids = nullptr, *tok_d = nullptr, *ami = nullptr, *slot_d = nullptr;
    std::uint64_t* pos_d = nullptr;
    unsigned* flag = nullptr;
    hipStream_t stream = nullptr;
    Profile prof;
    bool profiling = false;
    // profiling: events at segment boundaries; seg_cat[i] = category of the segment ending at event i
    // Profile segment categories. NOT k-prefixed: an enumerator named kHc here shadowed strix::glm::kHc (= 4) inside Impl
    // as 1, so X / X2 / F were sized for 2 rows (R >= 3 memory fault (review finding)) and gate_residual copied stream 0 only.
    enum Cat : int { segStart = -1, segEmbed, segHc, segKda, segDsa, segXch, segMoe, segDense, segHead };
    static_assert(kHc == 4 && kHidden == 4096, "a member name shadows glm::kHc / kHidden inside Impl (R >= 3 fault 7a5a9381)");
    std::vector<hipEvent_t> seg_ev;
    std::vector<int> seg_cat;
    std::size_t seg_n = 0;
    void mark(int cat) {
        if (!profiling) return;
        if (seg_n == seg_ev.size()) {
            hipEvent_t e{};
            hip_ok(hipEventCreate(&e), "profile event");
            seg_ev.push_back(e);
            seg_cat.push_back(cat);
        }
        seg_cat[seg_n] = cat;
        hip_ok(hipEventRecord(seg_ev[seg_n++], stream), "profile record");
    }
    void collect() {  // after the call's stream sync
        if (!profiling) return;
        hip_ok(hipEventSynchronize(seg_ev[seg_n - 1]), "profile sync");
        for (std::size_t i = 1; i < seg_n; ++i) {
            float ms = 0;
            hip_ok(hipEventElapsedTime(&ms, seg_ev[i - 1], seg_ev[i]), "profile elapsed");
            double* f[] = {&prof.embed, &prof.hc, &prof.kda, &prof.dsa, &prof.exchange, &prof.moe, &prof.dense, &prof.head};
            if (seg_cat[i] >= 0) *f[seg_cat[i]] += ms;
        }
        prof.attn = prof.kda + prof.dsa;
        prof.ffn = prof.moe + prof.dense;
        seg_n = 0;
    }
    bool smoke_head = false;
    // STRIX_GLM_STEP_FLAG=0: a plain step syncs for the layers' checkpoints before the head (old path).
    const bool defer_layer_flag = !(std::getenv("STRIX_GLM_STEP_FLAG") != nullptr && std::string(std::getenv("STRIX_GLM_STEP_FLAG")) == "0");
    const bool boundary_fused = !(std::getenv("STRIX_GLM_BOUNDARY") != nullptr && std::string(std::getenv("STRIX_GLM_BOUNDARY")) == "0");
    // MoE kernels (TP2 rank slices, I = 1024): default for world 2; STRIX_GLM_MOE=ref = DefaultMoe.
    const bool moe_fast = opt.world == 2 && !(std::getenv("STRIX_GLM_MOE") != nullptr && std::string(std::getenv("STRIX_GLM_MOE")) == "ref");
    // attention fused Tier B (src/glm/attn/glm_attn_tierb, bits of strix_gemv): the FastAttn seam reads raw attention EXL3
    // linears (file layout, no strix_weights). STRIX_GLM_ATTN=ref or STRIX_GLM_TIERB=lib load them prepared and keep the
    // library calls (same bits). The MTP layer keeps prepared linears (its drafter callbacks use the library).
    // attention fused groups are built for the TP2 rank shapes (32 KDA heads / rank): world 1 keeps strix_gemv (same bits).
    const bool attn_tierb = opt.world == 2 && !(std::getenv("STRIX_GLM_ATTN") != nullptr && std::string(std::getenv("STRIX_GLM_ATTN")) == "ref") &&
                            !(std::getenv("STRIX_GLM_TIERB") != nullptr && std::string(std::getenv("STRIX_GLM_TIERB")) == "lib");
    const bool post_ref = std::getenv("STRIX_GLM_POST") != nullptr && std::string(std::getenv("STRIX_GLM_POST")) == "ref";
    // hc_pre_v3 variant: default 7 (qualified revision a7c9d6db, bitwise rows 1-8); STRIX_GLM_HC_PRE_V=4 keeps the previous one.
    const int hc_pre_variant = std::getenv("STRIX_GLM_HC_PRE_V") != nullptr ? std::atoi(std::getenv("STRIX_GLM_HC_PRE_V")) : 7;
    IdentityMode identity = opt.identity;
    // Frozen when load starts for identical load-plan and actual-load allocations.
    IdentityMode load_identity = opt.identity;
    bool r1b = false;
    // route log (identity comparisons): ids [kLayers][8] of row 0 of the last forward, device; off by default.
    bool route_log = false;
    unsigned cur_layer = 0;
    std::uint32_t* route_log_d = nullptr;
    bool load_test = false;
    // speculative verify (Options::spec): per-row / replay flags, per-slot pending rows + current replay set
    unsigned *row_flag = nullptr, *replay_flag = nullptr;
    std::vector<unsigned> pending, cur;
    // MTP drafter buffers (Options::mtp): own flag (drafter non-finite never touches the trunk flag), h' rows
    unsigned* mtp_flag = nullptr;
    float* trunk_logits = nullptr;
    std::vector<Row> mtp_rows;
    std::vector<std::uint32_t> mtp_slots;
    std::vector<std::uint64_t> mtp_positions;
    float *m_e = nullptr, *m_en = nullptr, *m_hn = nullptr, *m_cat = nullptr, *m_z = nullptr, *m_r = nullptr, *m_y = nullptr, *m_h = nullptr, *m_hs = nullptr,
          *m_mz = nullptr;
    // load memory accounting (design notes): plan (dry run) -> calibration -> admission -> checked load
    mem::LoadPlan plan;
    mem::AllocModel amodel;
    mem::Margins margins;
    mem::Snapshot base;

    Impl(Options o, dist::Communicator* c) : opt(std::move(o)), comm(c), pack(opt.model_dir), lib(pack, opt.raw_linears) {
        if (const char* v = std::getenv("STRIX_GLM_PF_PIPELINE"); v != nullptr) {
            const std::string value(v);
            if (value != "1" && value != "2") throw std::invalid_argument("STRIX_GLM_PF_PIPELINE must be 1 or 2");
            opt.pf_pipeline = value == "2" ? 2U : 1U;
        }
        if (opt.pf_pipeline != 1 && opt.pf_pipeline != 2) throw std::invalid_argument("pf_pipeline must be 1 or 2");
        if (opt.world != 1 && opt.world != 2) throw std::invalid_argument("world must be 1 or 2");
        // STRIX_GLM_LOAD_TEST=1: one rank's slice loaded without a peer (load timing only; any forward throws).
        load_test = opt.world == 2 && comm == nullptr && std::getenv("STRIX_GLM_LOAD_TEST") != nullptr &&
                    std::string(std::getenv("STRIX_GLM_LOAD_TEST")) == "1";
        if (opt.world == 2 && comm == nullptr && !load_test && !opt.external_exchange)
            throw std::invalid_argument("world 2 needs a communicator (or Options::external_exchange + set_exchange)");
        if (opt.capacity % kKpool != 0) throw std::invalid_argument("capacity must be a multiple of 4");
        if (const char* env = std::getenv("STRIX_GLM_IDENTITY")) {
            const std::string mode(env);
            if (mode == "r1") identity = IdentityMode::kR1;
            else if (mode == "r1b") identity = IdentityMode::kR1b;
            else if (mode == "r1b-all") identity = IdentityMode::kR1bAll;
            else throw std::invalid_argument("STRIX_GLM_IDENTITY must be r1|r1b|r1b-all");
        }
        (void)identity_mode_name(identity);  // Reject invalid Options enum values as well.
        r1b = identity != IdentityMode::kR1;
        strix::glm::r1b::set_active(r1b);  // KDA seams follow generated forwards; prefill scopes R1.
        kda_heads = kKdaHeads / opt.world;
        mla_heads = kMlaHeads / opt.world;
        expert_i = kExpertI / opt.world;
        dense_i = kDenseI / opt.world;
        vocab_n = kVocab / opt.world;
        vocab0 = opt.rank * vocab_n;
        const char* attn_env = std::getenv("STRIX_GLM_ATTN");
        if (attn_env != nullptr && std::string(attn_env) == "ref") attn = std::make_shared<DefaultAttn>(scr);
        else attn = fast_attention(kda_heads, mla_heads, opt.capacity);
        moe = moe_fast ? fast_moe() : std::make_shared<DefaultMoe>(scr);
        xch = opt.world == 1 || load_test ? local_exchange()
              : opt.external_exchange   ? std::shared_ptr<Exchange>(std::make_shared<MissingExchange>())
                                        : std::shared_ptr<Exchange>(std::make_shared<CommExchange>(*comm, opt.rank, std::max(8U, opt.exchange_rows ? opt.exchange_rows : opt.max_rows)));  // decode / verify calls carry up to 8 rows
        fallback_xch = xch;
        if (const char* v = std::getenv("STRIX_GLM_DECODE_XCH"); v != nullptr) {
            const std::string kind(v);
            if (kind != "comm" && kind != "xfer") throw std::invalid_argument("STRIX_GLM_DECODE_XCH must be comm or xfer");
            opt.decode_xfer = kind == "xfer";
        }
        if (opt.world == 2 && comm != nullptr && !opt.external_exchange && opt.decode_xfer)
            decode_xch = std::make_shared<xfer::XferExchange>(comm->transport(), opt.rank, 8U);
        pf = reference_prefill();
    }

    // Never share transport progress between implementations. complete() joins Xfer jobs and seals any deferred
    // CRC; the stream sync also retires device consumers of either implementation's receive buffers / flags.
    void switch_exchange(const std::shared_ptr<Exchange>& next) {
        if (xch == next) return;
        xch->complete(stream);
        hip_ok(hipStreamSynchronize(stream), "exchange switch sync");
        xch = next;
    }

    enum class ForwardKind { Plain, Verify, Mtp, Gate, Prefill };
    void select_exchange(ForwardKind kind, unsigned rows) {
        const char* env = std::getenv("STRIX_GLM_SPEC_XCH");
        const std::string mode = env == nullptr || *env == '\0' ? "xfer" : env;
        if (mode != "xfer" && mode != "comm") throw std::invalid_argument("STRIX_GLM_SPEC_XCH must be xfer or comm");
        const bool fast = kind == ForwardKind::Plain ||
                          ((kind == ForwardKind::Verify || kind == ForwardKind::Mtp) && mode == "xfer");
        switch_exchange(decode_xch && fast && rows > 0 && rows <= 8 ? decode_xch : fallback_xch);
    }

    std::string L(unsigned l) const { return kP + "layers." + std::to_string(l) + "."; }

    void load_hc(LayerW& w, unsigned l) {
        for (int s = 0; s < 2; ++s) {
            HcSite& h = s == 0 ? w.hc_attn : w.hc_ffn;
            const std::string n = L(l) + (s == 0 ? "hc_attn_" : "hc_ffn_");
            if (lib.dtype(n + "fn") != "F32") throw std::runtime_error("hc fn must be F32 (identity): " + n + "fn");
            h.fn = static_cast<float*>(lib.upload_rows(n + "fn"));
            h.base = static_cast<float*>(lib.upload_rows(n + "base"));
            h.scale = static_cast<float*>(lib.upload_rows(n + "scale"));
        }
    }
    void load_dsa(LayerW& w, const std::string& a) {
        const std::uint64_t r = opt.rank, H = mla_heads;
        auto& d = w.dsaw;
        d.heads = mla_heads;
        const bool raw = attn_tierb && !w.mtp;
        d.q_a = raw ? lib.whole_raw(a + "q_a_proj") : lib.whole(a + "q_a_proj");
        d.kv_a = raw ? lib.whole_raw(a + "kv_a_proj_with_mqa") : lib.whole(a + "kv_a_proj_with_mqa");
        d.wq_b = raw ? lib.whole_raw(a + "indexer.wq_b") : lib.whole(a + "indexer.wq_b");
        d.q_b = raw ? lib.out_slice_raw(a + "q_b_proj", r * H * kMlaDim, H * kMlaDim) : lib.out_slice(a + "q_b_proj", r * H * kMlaDim, H * kMlaDim);
        d.o = raw ? lib.in_slice_raw(a + "o_proj", r * H * kMlaDim, H * kMlaDim) : lib.in_slice(a + "o_proj", r * H * kMlaDim, H * kMlaDim);
        d.q_a_norm = static_cast<const std::uint16_t*>(lib.upload_rows(a + "q_a_layernorm.weight"));
        d.kv_a_norm = static_cast<const std::uint16_t*>(lib.upload_rows(a + "kv_a_layernorm.weight"));
        d.kv_b = static_cast<const std::uint16_t*>(lib.upload_rows(a + "kv_b_proj.weight", r * H * 2 * kMlaDim, H * 2 * kMlaDim));
        d.wk = static_cast<const std::uint16_t*>(lib.upload_rows(a + "indexer.wk.weight"));
        d.gate = static_cast<const std::uint16_t*>(lib.upload_rows(a + "indexer.index_kpool_compress_gate"));
        d.weights_proj = static_cast<const std::uint16_t*>(lib.upload_rows(a + "indexer.weights_proj.weight"));
        d.k_norm_w = static_cast<const std::uint16_t*>(lib.upload_rows(a + "indexer.k_norm.weight"));
        d.k_norm_b = static_cast<const std::uint16_t*>(lib.upload_rows(a + "indexer.k_norm.bias"));
        d.ape = static_cast<const float*>(lib.upload_rows(a + "indexer.index_kpool_compress_ape"));
        for (const char* n : {"indexer.wk.weight", "indexer.index_kpool_compress_gate", "indexer.weights_proj.weight", "indexer.k_norm.weight",
                              "indexer.k_norm.bias", "kv_b_proj.weight"})
            if (lib.dtype(a + n) != "F16") throw std::runtime_error(std::string("expected F16: ") + a + n);
        if (lib.dtype(a + "indexer.index_kpool_compress_ape") != "F32") throw std::runtime_error("ape must be F32");
        const std::size_t S = opt.slots, cap = opt.capacity;
        w.dst.cap = cap;
        w.dst.latent = lib.state<std::uint16_t>(S * cap * kKvLora);
        w.dst.ikey = lib.state<std::uint16_t>(S * cap * kIdxDim);
        w.dst.igate = lib.state<std::uint16_t>(S * cap * kIdxDim);
        w.dst.pooled = lib.state<float>(S * (cap / kKpool) * kIdxDim);
    }
    const std::uint16_t* pack_r1b(const std::uint16_t* w, unsigned n) {
        const int variant = strix::glm::r1b::dense_variant();
        if (load_identity == IdentityMode::kR1 || (variant != 5 && variant != 6 && variant != 7)) return nullptr;
        // Identical load-time condition and lib.buffer allocation in both planning/loading passes.
        auto* packed = lib.buffer<std::uint16_t>(static_cast<std::size_t>(n) * kHidden);
        if (!lib.planning()) {
            strix::glm::r1b::pack_k4096(w, n, packed, stream);
            hip_ok(hipStreamSynchronize(stream), "r1b load pack");
        }
        return packed;
    }
    void load_moe(LayerW& w, const std::string& m) {
        const std::uint64_t r = opt.rank, I = expert_i;
        w.moew.router = static_cast<const std::uint16_t*>(lib.upload_rows(m + "gate.weight"));
        if (lib.dtype(m + "gate.weight") != "F16") throw std::runtime_error("router must be F16: " + m);
        w.moew.router_packed = pack_r1b(w.moew.router, kExperts);
        w.moew.bias = static_cast<const float*>(lib.upload_rows(m + "gate.e_score_correction_bias"));
        auto mlp = [&](const std::string& p) {
            if (moe_fast)  // MoE kernels read the raw file layout: no strix_weights per expert (no 2 MiB granules)
                return Mlp{lib.out_slice_raw(p + "gate_proj", r * I, I), lib.out_slice_raw(p + "up_proj", r * I, I),
                           lib.in_slice_raw(p + "down_proj", r * I, I)};
            return Mlp{lib.out_slice(p + "gate_proj", r * I, I), lib.out_slice(p + "up_proj", r * I, I), lib.in_slice(p + "down_proj", r * I, I)};
        };
        w.moew.experts.reserve(kExperts);
        for (unsigned e = 0; e < kExperts; ++e) w.moew.experts.push_back(mlp(m + "experts." + std::to_string(e) + "."));
        w.moew.shared = mlp(m + "shared_experts.");
        auto* ed = static_cast<Mlp*>(lib.arena(sizeof(Mlp) * kExperts));
        if (!lib.planning()) hip_ok(hipMemcpy(ed, w.moew.experts.data(), sizeof(Mlp) * kExperts, hipMemcpyHostToDevice), "experts_d");
        w.moew.experts_d = ed;
    }
    void load_layer(unsigned l) {
        LayerW w;
        w.index = l;
        w.mtp = l == kMtpLayer;
        w.kda = !w.mtp && is_kda_layer(l);
        w.moe = is_moe_layer(l);
        const std::string p = L(l), a = p + "self_attn.";
        if (!w.mtp) load_hc(w, l);
        w.in_norm = static_cast<const std::uint16_t*>(lib.upload_rows(p + "input_layernorm.weight"));
        w.post_norm = static_cast<const std::uint16_t*>(lib.upload_rows(p + "post_attention_layernorm.weight"));
        if (w.kda) {
            const std::uint64_t r = opt.rank, H = kda_heads, C = H * kKdaDim;
            auto& k = w.kdaw;
            k.heads = kda_heads;
            for (unsigned c = 0; c < 3; ++c)
                k.qkv[c] = attn_tierb ? lib.out_slice_raw(a + "qkv_proj", c * kKdaQkv + r * C, C) : lib.out_slice(a + "qkv_proj", c * kKdaQkv + r * C, C);
            k.o = attn_tierb ? lib.in_slice_raw(a + "o_proj", r * C, C) : lib.in_slice(a + "o_proj", r * C, C);
            k.f_a = static_cast<const std::uint16_t*>(lib.upload_rows(a + "f_a_proj.weight"));
            k.g_a = static_cast<const std::uint16_t*>(lib.upload_rows(a + "g_a_proj.weight"));
            k.f_b = static_cast<const std::uint16_t*>(lib.upload_rows(a + "f_b_proj.weight", r * C, C));
            k.g_b = static_cast<const std::uint16_t*>(lib.upload_rows(a + "g_b_proj.weight", r * C, C));
            k.b = static_cast<const std::uint16_t*>(lib.upload_rows(a + "b_proj.weight", r * H, H));
            k.f_a_packed = pack_r1b(k.f_a, kKdaDim);
            k.g_a_packed = pack_r1b(k.g_a, kKdaDim);
            k.b_packed = pack_r1b(k.b, kda_heads);
            k.conv = static_cast<const std::uint16_t*>(
                lib.upload_row_ranges(a + "conv1d.weight", {{r * C, C}, {kKdaQkv + r * C, C}, {2 * kKdaQkv + r * C, C}}));
            k.A_log = static_cast<const float*>(lib.upload_rows(a + "A_log", r * H, H));
            k.dt_bias = static_cast<const float*>(lib.upload_rows(a + "dt_bias", r * C, C));
            k.o_norm = static_cast<const std::uint16_t*>(lib.upload_rows(a + "o_norm.weight"));
            for (const char* n : {"f_a_proj.weight", "g_a_proj.weight", "f_b_proj.weight", "g_b_proj.weight", "b_proj.weight"})
                if (lib.dtype(a + n) != "F16") throw std::runtime_error(std::string("expected F16: ") + a + n);
            if (lib.dtype(a + "conv1d.weight") != "BF16" || lib.dtype(a + "o_norm.weight") != "BF16")
                throw std::runtime_error("expected BF16 conv1d / o_norm: " + a);
            w.kst.S = lib.state<float>(static_cast<std::size_t>(opt.slots) * H * kKdaDim * kKdaDim);
            w.kst.conv = lib.state<float>(static_cast<std::size_t>(opt.slots) * 3 * C * 3);
            if (opt.spec) {  // deferred-replay sets (design notes section 1): [slots][2][8] rows
                const std::size_t sets = static_cast<std::size_t>(opt.slots) * 2 * 8;
                w.kst.rep_qkv = lib.state<float>(sets * 3 * C);
                w.kst.rep_scr = lib.state<float>(sets * H * 4 * kKdaDim);
                w.kst.rep_beta = lib.state<float>(sets * H);
            }
        } else {
            load_dsa(w, a);
        }
        if (w.moe) {
            load_moe(w, p + "mlp.");
        } else {
            const std::uint64_t r = opt.rank, I = dense_i;
            const std::string m = p + "mlp.";
            w.dense = Mlp{lib.out_slice(m + "gate_proj", r * I, I), lib.out_slice(m + "up_proj", r * I, I), lib.in_slice(m + "down_proj", r * I, I)};
        }
        if (w.mtp) {
            w.eh_proj = lib.whole(p + "eh_proj");
            w.enorm = static_cast<const std::uint16_t*>(lib.upload_rows(p + "enorm.weight"));
            w.hnorm = static_cast<const std::uint16_t*>(lib.upload_rows(p + "hnorm.weight"));
            w.head_norm = static_cast<const std::uint16_t*>(lib.upload_rows(p + "shared_head.norm.weight"));
        }
        layers.push_back(std::move(w));
    }

    // The load sequence, run twice: once as a dry run (lib.planning(): records every allocation, nothing read or
    // allocated) and once for real with a memory check after every step. `step` is called after each step.
    template <class Step> void load_steps(bool want_head, Step&& step) {
        embed = static_cast<std::uint16_t*>(lib.upload_rows(kP + "embed_tokens.weight"));
        if (lib.dtype(kP + "embed_tokens.weight") != "BF16") throw std::runtime_error("embedding must be BF16");
        step("embed");
        for (unsigned l = 0; l < opt.layers; ++l) {
            load_layer(l);
            step("layer " + std::to_string(l));
        }
        if (opt.mtp) {
            load_layer(kMtpLayer);
            step("mtp");
        }
        if (want_head) {
            final_norm = static_cast<const std::uint16_t*>(lib.upload_rows(kP + "norm.weight"));
            head = lib.out_slice("lm_head", vocab0, vocab_n);
            step("head");
        }
    }

    // STRIX_GLM_MEMTRACE=FILE ('-' = stderr): one memory line per second for the engine's lifetime (load + forwards).
    // Watchdog (default ON, STRIX_GLM_WATCHDOG=0 off): aborts the process (exit 86) near the OOM line (glm_mem.hpp).
    void start_monitor() {
        const char* tp = std::getenv("STRIX_GLM_MEMTRACE");
        monitor = std::make_unique<mem::Monitor>(probe, tp ? std::string(tp) : std::string(), mem::Watchdog::from_env());
        monitor->phase("plan");
    }

    // Physical bytes of one hipMalloc of the plan's dominant size (the expert linear): K touched allocations, delta of
    // the driver's device counters (amdgpu GTT + VRAM used; else /proc/meminfo GPUActive). STRIX_GLM_ALLOC_GRANULE=BYTES
    // skips the measurement; no counters -> 2 MiB assumed.
    void calibrate() {
        const std::uint64_t size = plan.dominant_size();
        if (const char* e = std::getenv("STRIX_GLM_ALLOC_GRANULE"); e != nullptr && *e) {
            amodel = mem::AllocModel{};
            amodel.granule = std::strtoull(e, nullptr, 10);
            if (amodel.granule == 0) throw std::invalid_argument("STRIX_GLM_ALLOC_GRANULE must be > 0 bytes");
            return;
        }
        const mem::Snapshot a = probe.read();
        if (!a.dev_counters() || size == 0) {
            amodel = mem::infer_alloc_model(size, 0.0);
            std::fprintf(stderr, "mem: no device memory counters: allocation cost assumed %s\n", amodel.str().c_str());
            return;
        }
        // Retained batches (a repeatable slope, not one global delta): kBatches x K touched allocations, all kept until
        // the end; per batch the larger of (GTT + VRAM used) and GPUActive deltas. Two attempts, then the conservative model.
        constexpr int K = 16, kBatches = 4;
        mem::admit("allocation calibration",
                   {{"calibration buffers (2x worst case)", 2 * K * kBatches * mem::round_up(size, 2 * mem::kMiB), 0}}, a, margins);
        struct Bufs {  // freed on every exit path (a HIP error included)
            std::vector<void*> p;
            ~Bufs() {
                for (void* q : p) (void)hipFree(q);
            }
        };
        bool consistent = false;
        for (int attempt = 0; attempt < 2 && !consistent; ++attempt) {
            Bufs bufs;
            bufs.p.reserve(static_cast<std::size_t>(K) * kBatches);  // push_back after hipMalloc cannot throw
            std::vector<double> per;
            std::string detail;
            mem::Snapshot prev = probe.read();
            for (int b = 0; b < kBatches; ++b) {
                for (int i = 0; i < K; ++i) {
                    void* p = nullptr;
                    hip_ok(hipMalloc(&p, size), "calibration malloc");
                    bufs.p.push_back(p);
                    hip_ok(hipMemset(p, 0, size), "calibration memset");
                }
                hip_ok(hipDeviceSynchronize(), "calibration sync");
                const mem::Snapshot now_s = probe.read();
                const double dev = now_s.dev_used() > prev.dev_used() ? static_cast<double>(now_s.dev_used() - prev.dev_used()) / K : 0.0;
                const double act = now_s.gpu_active > prev.gpu_active ? static_cast<double>(now_s.gpu_active - prev.gpu_active) / K : 0.0;
                per.push_back(std::max(dev, act));
                detail += " " + std::to_string(static_cast<long long>(dev)) + "/" + std::to_string(static_cast<long long>(act));
                prev = now_s;
            }
            amodel = mem::model_from_batches(size, per, consistent);
            std::fprintf(stderr, "mem: calibration attempt %d: %d x %d x hipMalloc(%llu B), B each per batch (GTT+VRAM / GPUActive):%s -> %s%s\n",
                         attempt + 1, kBatches, K, static_cast<unsigned long long>(size), detail.c_str(), amodel.str().c_str(),
                         consistent ? "" : " (batches disagree)");
        }
    }

    LoadReport load(const std::function<unsigned(const mem::LoadAdmission&)>& select_max_rows) {
        try {
            return load_checked(select_max_rows);
        } catch (const std::exception& e) {  // printed before the unwinding (HIP teardown can stall)
            std::fprintf(stderr, "load: rank %u FAILED: %s\n", opt.rank, e.what());
            if (monitor) monitor->phase("failed");
            throw;
        }
    }

    LoadReport load_checked(const std::function<unsigned(const mem::LoadAdmission&)>& select_max_rows) {
        load_identity = identity;
        const auto t0 = Clock::now();
        start_monitor();
        hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
        // STRIX_GLM_SMOKE_HEAD=1 (test only): the head on a truncated stack, allowed only for a synthetic model
        // (tools/glm/make_synth_model.py writes "synthetic": true into the index metadata); refused for any other model.
        if (const char* e = std::getenv("STRIX_GLM_SMOKE_HEAD"); e != nullptr && std::string(e) == "1" && opt.layers != kLayers) {
            std::ifstream f(opt.model_dir + "/model.safetensors.index.json");
            const std::string idx((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            if (idx.find("\"synthetic\": true") == std::string::npos)
                throw std::invalid_argument("STRIX_GLM_SMOKE_HEAD=1 is only allowed on a synthetic model");
            smoke_head = true;
        }
        const bool want_head = opt.layers == kLayers || smoke_head;

        // 1. Plan: the exact list of device allocations, staging and file bytes, before the first byte is allocated.
        margins = mem::Margins::from_env();
        std::uint64_t prefill_install = 0;
        auto make_plan = [&]() {
            plan = {};
            lib.set_plan(&plan);
            load_steps(want_head, [&](const std::string& n) { plan.step(n); });
            alloc();
            plan.step("buffers");
            lib.set_plan(nullptr);
            // Conservative install bound: planned handles are null even for already-prepared linears.
            prefill_install = 0;
            std::uint64_t staging = 0;
            for (const auto& w : layers) {
                if (!w.moe) continue;
                prefill_install += moe->warm_bytes(w.moew);
                for (const Linear* l : {&w.moew.shared.gate, &w.moew.shared.up, &w.moew.shared.down}) {
                    const auto parts = Lib::raw_trellis_bytes(l->raw) + std::uint64_t{l->raw.input} * 2 + std::uint64_t{l->raw.output} * 2;
                    prefill_install += mem::round_up(parts + 3 * 256, 2 * mem::kMiB);
                    staging = std::max(staging, parts);
                }
            }
            prefill_install += staging;
            layers.clear();
        };
        make_plan();
        // 2. Physical cost of the dominant allocation, 3. admission: fail closed before loading anything.
        monitor->phase("calibrate");
        calibrate();
        auto load_terms = [&]() -> std::vector<mem::Term> {
            return {{"device plan (physical)", plan.physical(amodel), 0},
                    {"first-forward reserve (strix workspace, runtime)", fwd_reserve(), 0},
                    {"pinned staging", mem::round_up(std::max<std::uint64_t>(plan.stage_max, 1), 2 * mem::kMiB), 0},
                    {"pageable whole-tensor reads", 0, plan.whole_max}, {"O_DIRECT bounce", 0, plan.bounce}};
        };
        auto select_rows = [&](bool recheck) {
            std::size_t dev_free = 0, dev_total = 0;
            hip_ok(hipMemGetInfo(&dev_free, &dev_total), "row admission mem info");
            const unsigned rows = select_max_rows({load_terms(), probe.read(), margins, prefill_install, dev_free, recheck, amodel});
            if (rows == 0 || rows > opt.max_rows) throw std::invalid_argument("load row selector must reduce max_rows within 1..max_rows");
            if (rows != opt.max_rows) {
                opt.max_rows = rows;
                make_plan();
            }
            if (auto* cx = dynamic_cast<CommExchange*>(fallback_xch.get())) cx->resize_rows(std::max(8U, rows));
        };
        if (select_max_rows) {
            select_rows(false);
            // Only negotiated auto peers enter this second round. It sees actual communication allocations,
            // so an admission refusal at the larger size becomes a shared fallback before the first weight.
            if (opt.auto_rows) select_rows(true);
        }
        const std::uint64_t stage = mem::round_up(std::max<std::uint64_t>(plan.stage_max, 1), 2 * mem::kMiB);
        lib.set_stage_bytes(stage);
        const std::uint64_t phys = plan.physical(amodel);
        const mem::Snapshot s0 = probe.read();
        std::fprintf(stderr, "mem: rank %u %s\nmem: node %s\n", opt.rank, plan.summary(amodel).c_str(), mem::describe(s0).c_str());
        if (const char* e = std::getenv("STRIX_GLM_MEM_ADMIT"); e != nullptr && std::string(e) == "0") {
            std::fprintf(stderr, "mem: WARNING admission check disabled (STRIX_GLM_MEM_ADMIT=0); the watchdog stays on\n");
        } else {
            mem::admit("rank " + std::to_string(opt.rank) + " load (" + std::to_string(opt.layers) + " layers)",
                       load_terms(),
                       s0, margins);
        }
        if (const char* e = std::getenv("STRIX_GLM_MEM_PLAN_ONLY"); e != nullptr && std::string(e) == "1") {  // --mem-plan
            monitor->phase("plan-only");
            LoadReport rep;
            rep.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
            return rep;
        }
        lib.reserve_staging();
        base = probe.read();
        const bool load_limit = base.dev_counters() && mem::Watchdog::from_env().max_gtt_used == 0;
        if (load_limit) monitor->set_max_gtt_used(base.dev_used() + phys + 2 * mem::kGiB);

        // 4. The load, checked after every step against the plan (measured gtt_used growth) and the node's headroom.
        monitor->phase("load");
        std::size_t idx = 0;
        load_steps(want_head, [&](const std::string& n) { check_step(n, idx++, t0); });
        lib.release_staging();
        alloc();
        check_step("buffers", idx++, t0);
        // parity: the real pass made exactly the planned allocations, in order (covers raw linears, MTP, spec, options)
        const auto& real = lib.recorded();
        if (real != plan.device) {
            std::size_t i = 0;
            while (i < real.size() && i < plan.device.size() && real[i] == plan.device[i]) ++i;
            throw std::logic_error("memory plan diverged from the load at allocation " + std::to_string(i) + " (plan " +
                                   std::to_string(plan.device.size()) + " allocations, real " + std::to_string(real.size()) + ")");
        }
        // the plan-derived device limit covers the load only: later owners (prefill path, strix workspaces) admit their own
        // allocations; the RAM headroom / swap / PSI / RSS thresholds stay on
        if (load_limit) monitor->set_max_gtt_used(0);
        monitor->phase("ready");
        loaded = true;
        if (pf_installed && pf) install_prefill(*pf);  // a path set before load() (admits its own allocations)
        LoadReport rep;
        rep.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
        rep.gib_read = pack.gib_read();
        rep.device_gib = lib.device_gib();
        return rep;
    }

    // Headroom kept for allocations made after the load by their owners' first use (strix matmul workspace, HIP runtime);
    // STRIX_GLM_MEM_FWD_RESERVE_GIB (default 0.5). A prefill path admits its own buffers (make_prefill_path).
    static std::uint64_t fwd_reserve() {
        const char* e = std::getenv("STRIX_GLM_MEM_FWD_RESERVE_GIB");
        return e != nullptr && *e ? static_cast<std::uint64_t>(std::strtod(e, nullptr) * static_cast<double>(mem::kGiB)) : mem::kGiB / 2;
    }

    // After load step `i`: measured gtt_used growth must follow the plan (a model error fails closed long before the
    // OOM line) and the rest of the plan must still fit the node's current headroom (other tenants can grow).
    void check_step(const std::string& name, std::size_t i, Clock::time_point t0) {
        if (i >= plan.mark.size() || plan.step_name[i] != name) throw std::logic_error("load diverged from its plan at " + name);
        monitor->counters(lib.allocated_bytes(), pack.bytes_read());
        const mem::Snapshot s = probe.read();
        const std::uint64_t pred = plan.physical(amodel, plan.mark[i]), total = plan.physical(amodel);
        const std::uint64_t measured = s.dev_counters() && base.dev_counters() && s.dev_used() > base.dev_used() ? s.dev_used() - base.dev_used() : 0;
        const bool report = name.rfind("layer ", 0) != 0 || (std::stoul(name.substr(6)) % 5 == 4) || i + 2 >= plan.mark.size();
        if (report)
            std::fprintf(stderr, "load: rank %u %s %.1f s %.1f GiB read %.2f GiB device (plan %.2f, measured +%.2f GiB) headroom ram %.2f gtt %.2f GiB\n",
                         opt.rank, name.c_str(), std::chrono::duration<double>(Clock::now() - t0).count(), pack.gib_read(),
                         mem::gib(lib.allocated_bytes()), mem::gib(pred), mem::gib(measured), mem::gib(s.ram_headroom()),
                         mem::gib(s.gtt_headroom()));
        if (s.dev_counters() && base.dev_counters() && measured > pred + pred / 32 + 512 * mem::kMiB)
            throw mem::BudgetError("memory budget: STOPPED at " + name + ": device memory used grew " + std::to_string(measured >> 20) +
                                   " MiB but the plan predicts " + std::to_string(pred >> 20) + " MiB (" + amodel.str() +
                                   "): the allocation model is wrong; nothing more is allocated. node: " + mem::describe(s));
        if (std::getenv("STRIX_GLM_MEM_ADMIT") == nullptr || std::string(std::getenv("STRIX_GLM_MEM_ADMIT")) != "0")
            mem::admit("rest of the load after " + name, {{"device plan still to allocate", total > pred ? total - pred : 0, 0},
                                                          {"first-forward reserve", fwd_reserve(), 0},
                                                          {"pageable whole-tensor reads", 0, plan.whole_max}},
                       s, margins);
    }

    // Everything a chunked prefill through `p` needs, allocated now so a prefill allocates no device memory: prepared
    // copies of raw-only shared experts (needs_prepared_shared), the MoE seam's per-layer tables (used per row when
    // moe_by_row, and by decode), and the LinearRunner workspace for the path's calls and every GEMV call. Each step
    // is admitted against hipMemGetInfo free and MemAvailable with the path's margin before it allocates.
    void install_prefill(PrefillPath& p) {
        constexpr std::uint64_t kGranule = std::uint64_t{2} << 20;
        auto granules = [&](std::uint64_t b) { return (b + kGranule - 1) / kGranule * kGranule; };
        std::uint64_t need = 0, staging = 0;
        for (auto& w : layers) {
            if (!w.moe) continue;
            need += moe->warm_bytes(w.moew);
            if (!p.needs_prepared_shared()) continue;
            for (const Linear* l : {&w.moew.shared.gate, &w.moew.shared.up, &w.moew.shared.down}) {
                if (l->prepared != nullptr) continue;
                const std::uint64_t parts = Lib::raw_trellis_bytes(l->raw) + std::uint64_t{l->raw.input} * 2 + std::uint64_t{l->raw.output} * 2;
                need += granules(parts + 3 * 256);  // one allocation per prepared matrix, regions 256-byte aligned
                staging = std::max(staging, parts);  // prepare_from_raw's host copy, one matrix at a time
            }
        }
        if (need != 0) {
            hip_ok(hipDeviceSynchronize(), "sync");
            Lib::admit(need + staging, p.admission_margin(), "prefill install (prepared shared experts, MoE tables)");
            for (std::size_t i = 0; i < layers.size(); ++i) {
                auto& w = layers[i];
                if (!w.moe) continue;
                moe->warm(w.moew);
                if (!p.needs_prepared_shared()) continue;
                const std::string s = "layer " + std::to_string(i) + " shared ";
                lib.prepare_from_raw(w.moew.shared.gate, s + "gate");
                lib.prepare_from_raw(w.moew.shared.up, s + "up");
                lib.prepare_from_raw(w.moew.shared.down, s + "down");
            }
        }
        // No workspace growth during a prefill: the path's own calls and every GEMV call (gemv-mode Tier B, the head)
        // fit. The reservation fails closed (device free and MemAvailable must cover it + the margin while the old
        // workspace still exists).
        lib.reserve(std::max(p.linear_workspace_bytes(), lib.gemv_workspace_bound(8, p.admission_margin())), p.admission_margin());
    }

    void warm_moe(MoePath& m, std::uint64_t margin) {
        std::uint64_t need = 0;
        for (auto& w : layers)
            if (w.moe) need += m.warm_bytes(w.moew);
        if (need == 0) return;
        hip_ok(hipDeviceSynchronize(), "sync");
        Lib::admit(need, margin, "MoE seam tables");
        for (auto& w : layers)
            if (w.moe) m.warm(w.moew);
    }

    void alloc() {
        const std::size_t R = 8;  // decode rows per call
        const std::size_t RC = std::max<std::size_t>(R, opt.max_rows);  // chunked prefill rows (PrefillPath)
        X = lib.buffer<float>(RC * kHc * kHidden);
        X2 = lib.buffer<float>(RC * kHc * kHidden);
        F = lib.buffer<float>(RC * kHc * kHidden);
        hz = lib.buffer<float>(RC * kHcMix);
        hrr = lib.buffer<float>(RC);
        mix = lib.buffer<float>(RC * kHcMix);
        xc = lib.buffer<float>(RC * kHidden);
        xn = lib.buffer<float>(RC * kHidden);
        part = lib.buffer<float>(RC * kHidden);
        hfin = lib.buffer<float>(RC * kHidden);
        hidden = lib.buffer<float>(RC * kHidden);
        logits = lib.buffer<float>(R * std::max<std::size_t>(vocab_n, kExperts));
        rlog = lib.buffer<float>(RC * kExperts);
        wts = lib.buffer<float>(RC * kTopE);
        ids = lib.buffer<std::uint32_t>(RC * kTopE);
        tok_d = lib.buffer<std::uint32_t>(RC);
        slot_d = lib.buffer<std::uint32_t>(RC);
        pos_d = lib.buffer<std::uint64_t>(RC);
        amv = lib.buffer<float>(R);
        ami = lib.buffer<std::uint32_t>(R);
        flag = lib.state<unsigned>(1);
        if (opt.spec) {
            row_flag = lib.state<unsigned>(R);
            replay_flag = lib.state<unsigned>(1);
            pending.assign(opt.slots, 0);
            cur.assign(opt.slots, 0);
        }
        if (opt.mtp) {
            mtp_flag = lib.state<unsigned>(1);
            trunk_logits = lib.buffer<float>(vocab_n);
            mtp_rows.reserve(RC);
            mtp_slots.reserve(RC);
            mtp_positions.reserve(RC);
            m_e = lib.buffer<float>(RC * kHidden);
            m_en = lib.buffer<float>(RC * kHidden);
            m_hn = lib.buffer<float>(RC * kHidden);
            m_cat = lib.buffer<float>(RC * 2 * kHidden);
            m_z = lib.buffer<float>(RC * kHidden);
            m_r = lib.buffer<float>(RC * kHidden);
            m_y = lib.buffer<float>(RC * kHidden);
            m_h = lib.buffer<float>(RC * kHidden);
            m_hs = lib.buffer<float>(static_cast<std::size_t>(opt.slots) * 8 * kHidden);
            m_mz = lib.buffer<float>(2);
        }
        dg = lib.buffer<float>(R * dense_i);
        du = lib.buffer<float>(R * dense_i);
        da = lib.buffer<float>(R * dense_i);
        no_mtp.assign(opt.slots, 0);
        Scratch& s = *scr;
        const unsigned C = kda_heads * kKdaDim;
        s.qkv = lib.buffer<float>(3 * C);
        s.f = lib.buffer<float>(C);
        s.g = lib.buffer<float>(C);
        s.b = lib.buffer<float>(kda_heads);
        s.t128a = lib.buffer<float>(kKdaDim);
        s.t128b = lib.buffer<float>(kKdaDim);
        s.kda_scr = lib.buffer<float>(static_cast<std::size_t>(kda_heads) * 4 * kKdaDim);
        s.beta = lib.buffer<float>(kda_heads);
        s.o = lib.buffer<float>(std::max<std::size_t>(C, static_cast<std::size_t>(mla_heads) * kMlaDim));
        s.y1 = lib.buffer<float>(C);
        s.qa = lib.buffer<float>(kQLora);
        s.qr = lib.buffer<float>(kQLora);
        s.q = lib.buffer<float>(static_cast<std::size_t>(mla_heads) * kMlaDim);
        s.kva = lib.buffer<float>(kKvLora);
        s.lat = lib.buffer<float>(kKvLora);
        s.qI = lib.buffer<float>(kIdxHeads * kIdxDim);
        s.wh = lib.buffer<float>(kIdxHeads);
        s.cap = opt.capacity;
        s.scores = lib.buffer<float>(opt.capacity / kKpool);
        s.work = lib.buffer<std::uint32_t>(opt.capacity / kKpool);
        s.tokens = lib.buffer<std::uint32_t>(kIdxTopk + kKpool);
        s.ntok = lib.buffer<std::uint32_t>(1);
        s.qt = lib.buffer<float>(static_cast<std::size_t>(mla_heads) * kKvLora);
        s.key = lib.buffer<float>(kIdxDim);
        s.gate = lib.buffer<float>(kIdxDim);
        s.slot_d = lib.buffer<D::DsaSlot>(1);
        s.t_d = lib.buffer<unsigned>(1);
        s.eo = lib.buffer<float>(static_cast<std::size_t>(kTopE) * kHidden);
        s.ge = lib.buffer<float>(expert_i);
        s.ue = lib.buffer<float>(expert_i);
        s.ae = lib.buffer<float>(expert_i);
        s.sh = lib.buffer<float>(kHidden);
        if (!lib.planning()) {  // STRIX_GLM_ALLOC_LOG=1: names of the per-row buffers (the dalloc lines carry only sizes)
            const std::pair<const char*, const void*> named[] = {{"X", X}, {"X2", X2}, {"F", F}, {"mix", mix}, {"xc", xc}, {"xn", xn},
                                                                 {"part", part}, {"logits", logits}, {"tok_d", tok_d}, {"slot_d", slot_d},
                                                                 {"pos_d", pos_d}, {"flag", flag}, {"dg", dg}, {"da", da}};
            for (const auto& [n, q] : named) alloc_log(q, 0, n);
        }
    }

    void check_flag(const char* where) {
        unsigned f = 0;
        hip_ok(hipMemcpyAsync(&f, flag, 4, hipMemcpyDeviceToHost, stream), "flag");
        hip_ok(hipStreamSynchronize(stream), "flag sync");
        if (f) throw std::runtime_error(std::string("non-finite value (checkpoints C1-C5) at ") + where);
    }

    // One forward of R <= 8 rows through layers [0, n_layers); X holds the 4-stream residual afterwards.
    // Post-exchange consumers: the exact fast forms (glm_post_fast, gated bitwise vs Tier A) by
    // default; STRIX_GLM_POST=ref keeps the Tier A kernels (dev::hc_pre / dense / route).
    void hc_pre(const float* x, const float* fn, const float* hc_base, const float* scale, float* f, float* m, float* c, unsigned R) {
        if (r1b) strix::glm::r1b::hc_pre(x, fn, hc_base, scale, f, m, c, R, stream);  // default generated identity (blocked fn chains; prompts use R1)
        else if (post_ref) D::hc_pre(x, fn, hc_base, scale, f, m, c, R, stream);
        else xfer::hc_pre_v3(x, fn, hc_base, scale, f, m, c, R, stream, hc_pre_variant);
    }
    void router_route(const float* x, const MoeLayerWeights& w, unsigned R) {
        router_route_impl(x, w, R);
        if (route_log && route_log_d != nullptr && cur_layer < kLayers + 1)
            hip_ok(hipMemcpyAsync(route_log_d + static_cast<std::size_t>(cur_layer) * kTopE, ids, kTopE * 4, hipMemcpyDeviceToDevice, stream), "route log");
    }
    void router_route_impl(const float* x, const MoeLayerWeights& w, unsigned R) {
        if (r1b) {  // default generated identity: blocked router logits; selection / weights (R5 / R14) unchanged
            const int variant = strix::glm::r1b::dense_variant();
            const bool packed = (variant == 7 || variant == 6 || (variant == 5 && R == 1)) && w.router_packed != nullptr;
            strix::glm::r1b::router_dense(x, packed ? w.router_packed : w.router, rlog, R, stream,
                                         variant == 5 || variant == 6 || variant == 7 ? (packed ? variant : 2) : variant);
            xfer::route_v2(rlog, w.bias, ids, wts, R, stream);
        } else if (post_ref) {
            D::dense(x, w.router, D::WType::F16, kHidden, 0, kExperts, rlog, R, stream);
            D::route(rlog, w.bias, ids, wts, R, stream);
        } else {
            xfer::router_dense(x, w.router, rlog, R, stream, -1);
            xfer::route_v2(rlog, w.bias, ids, wts, R, stream);
        }
    }

    // Sublayer boundary: TP reduce of `part` + R7 expand into X. The exchange's fused form when it has one and the call
    // supports per-row flags when the exchange provides them (STRIX_GLM_BOUNDARY=0 forces the split form).
    // pfc (chunked prefill rows, up to max_rows): always the split form (the fused kernels take decode rows only).
    void boundary(unsigned* c1, unsigned c1_stride, unsigned R, bool pfc = false) {
        if (!pfc && boundary_fused && xch->fused_boundary_rows(part, X, mix, X2, c1, R, c1_stride, stream)) {
            mark(segXch);  // the fused kernel: exchange wait + R17 + R7
        } else {
            xch->reduce(part, R, stream);
            mark(segXch);
            D::hc_expand(X, mix, part, X2, c1, R, stream, c1_stride);
            mark(segHc);
        }
        std::swap(X, X2);
    }

    // N=2 whole-layer schedule. Only transport progresses off-stream; every seam and boundary is
    // enqueued on stream. Each half swaps its own residual pointers twice per layer, ending in X.
    void layers_forward_pipeline(std::span<const Row> rows, unsigned n_layers) {
        const unsigned R = static_cast<unsigned>(rows.size()), A = (R + 1) / 2;
        struct Half {
            unsigned offset, count;
            float *x, *out;
            bool normed = false;  // the half's F already holds its next site's norm (STRIX_GLM_PF_HC_FUSE)
        };
        Half a{0, A, X, X2};
        Half b{A, R - A, X + static_cast<std::size_t>(A) * kHc * kHidden,
               X2 + static_cast<std::size_t>(A) * kHc * kHidden};
        const bool hc_fuse = glm::pf::hc_expand_norm_enabled();
        CommExchange* async_xch = nullptr;
        if (const char* v = std::getenv("STRIX_DIST_ASYNC_START"); v && std::string(v) == "1")
            async_xch = dynamic_cast<CommExchange*>(xch.get());
        // Every Comm rank votes (the async start is part of the vote: only its half regions use the staging views).
        if (auto* cx = dynamic_cast<CommExchange*>(xch.get())) cx->agree_staged(stream, async_xch != nullptr);
        auto wait_source = [&](const Half& h) {
            if (async_xch) async_xch->wait_source(h.offset == 0 ? 0U : 1U, stream);
        };
        auto context = [&](const Half& h) {
            return CallCtx{&lib, rows.subspan(h.offset, h.count), slot_d + h.offset, pos_d + h.offset,
                           flag, stream, false, nullptr};
        };
        auto partial = [&](const Half& h) { return part + static_cast<std::size_t>(h.offset) * kHidden; };
        auto pre = [&](Half& h, const HcSite& site, const std::uint16_t* norm) {
            const std::size_t q = h.offset;
            if (h.normed) glm::pf::hc_pre_rows_normed(h.x, site.fn, site.base, site.scale, hrr + q,
                                                      hz + q * kHcMix, mix + q * kHcMix, xc + q * kHidden, h.count, stream);
            else glm::pf::hc_pre_rows(h.x, site.fn, site.base, site.scale, F + q * kHc * kHidden,
                                      hz + q * kHcMix, mix + q * kHcMix, xc + q * kHidden, h.count, stream);
            h.normed = false;
            D::rmsnorm(xc + q * kHidden, norm, D::WType::BF16, kHidden, kEps, xn + q * kHidden, h.count, stream);
            mark(segHc);
        };
        auto attention = [&](Half& h, LayerW& w) {
            wait_source(h);  // this half's seam may overwrite part; the other half remains independent
            pre(h, w.hc_attn, w.in_norm);
            const CallCtx c = context(h);
            const float* x = xn + static_cast<std::size_t>(h.offset) * kHidden;
            if (w.kda) pf->kda(w.kdaw, w.kst, c, x, partial(h));
            else pf->dsa(w.dsaw, w.dst, c, x, partial(h));
            mark(w.kda ? segKda : segDsa);
        };
        auto ffn = [&](Half& h, LayerW& w) {
            wait_source(h);
            pre(h, w.hc_ffn, w.post_norm);
            const CallCtx c = context(h);
            const std::size_t q = h.offset;
            const float* x = xn + q * kHidden;
            if (w.moe) {
                float* log = rlog + q * kExperts;
                std::uint32_t* id = ids + q * kTopE;
                float* wt = wts + q * kTopE;
                glm::pf::dense_rows(x, w.moew.router, D::WType::F16, kHidden, 0, kExperts, log, h.count, stream);
                D::route(log, w.moew.bias, id, wt, h.count, stream);
                if (pf->moe_by_row()) {
                    const unsigned B = std::clamp(pf->moe_row_block(), 1U, 8U);
                    for (unsigned r = 0; r < h.count; r += B) {
                        const CallCtx cr{&lib, c.rows.subspan(r, std::min(B, h.count - r)), slot_d + q + r,
                                         pos_d + q + r, flag, stream, false, nullptr};
                        moe->forward(w.moew, cr, x + static_cast<std::size_t>(r) * kHidden,
                                     id + static_cast<std::size_t>(r) * kTopE, wt + static_cast<std::size_t>(r) * kTopE,
                                     partial(h) + static_cast<std::size_t>(r) * kHidden);
                    }
                } else pf->moe(w.moew, c, x, id, wt, partial(h));
            } else pf->dense(w.dense, c, x, dense_i, partial(h));
            mark(w.moe ? segMoe : segDense);
        };
        auto start = [&](const Half& h) {
            if (async_xch) async_xch->start_reduce_region(partial(h), h.count, h.offset, h.offset == 0 ? 0 : 1, stream);
            else xch->start_reduce(partial(h), h.count, stream);
        };
        auto finish = [&] {
            xch->finish_reduce(stream);
            mark(segXch);
        };
        auto expand = [&](Half& h) {
            if (hc_fuse && glm::pf::hc_expand_norm_split()) {  // split control: unfused expand, then rr alone
                D::hc_expand(h.x, mix + static_cast<std::size_t>(h.offset) * kHcMix, partial(h), h.out, flag, h.count, stream);
                glm::pf::hc_norm_factor_rows(h.out, hrr + h.offset, h.count, stream);
                h.normed = true;
            } else if (hc_fuse) {  // also this half's next-site norm factors (its next pre skips hc_norm_rows and F)
                glm::pf::hc_expand_norm_rows(h.x, mix + static_cast<std::size_t>(h.offset) * kHcMix, partial(h), h.out,
                                             hrr + h.offset, flag, h.count, stream);
                h.normed = true;
            } else {
                D::hc_expand(h.x, mix + static_cast<std::size_t>(h.offset) * kHcMix, partial(h), h.out, flag, h.count, stream);
            }
            mark(segHc);
            std::swap(h.x, h.out);
        };
        try {
            for (unsigned li = 0; li < n_layers; ++li) {
                LayerW& w = layers[li];
                cur_layer = li;
                attention(a, w);
                // B's previous FFN may overlap next-layer attention A. Retire before the next start.
                if (li != 0 && b.count != 0) {
                    finish();
                    expand(b);
                }
                start(a);
                if (b.count != 0) attention(b, w);  // A's recurrence / cache commits precede B on stream
                finish();
                if (b.count != 0) start(b);
                expand(a);
                ffn(a, w);
                if (b.count != 0) finish();
                start(a);
                if (b.count != 0) {
                    expand(b);
                    ffn(b, w);
                }
                finish();
                if (b.count != 0) start(b);
                expand(a);
            }
            if (n_layers != 0 && b.count != 0) {
                finish();
                expand(b);
            }
        } catch (...) {
            // Do not leave a producer / host transfer borrowing row buffers after a seam throws.
            try { xch->complete(stream); } catch (...) {}
            throw;
        }
        check_flag("layers");
    }

    // sp != nullptr: speculative verify rows (one slot): checkpoints go to sp->row_flag[r], nothing throws here.
    // pfc: a chunk of a chunked prefill (rows of one slot, any count <= max_rows): attention, MoE and the dense FFN go
    // through the PrefillPath seam; the row-batched Tier A sites use pf::hc_pre_rows / pf::dense_rows (== decode bits).
    void layers_forward(std::span<const Row> rows, unsigned n_layers, ForwardKind kind, const SpecRows* sp = nullptr) {
        const bool pfc = kind == ForwardKind::Prefill;
        const auto R = static_cast<unsigned>(rows.size());
        if (pfc && opt.pf_pipeline == 2) {
            if (sp != nullptr || layer_dump != nullptr || null_layer >= 0)
                throw std::logic_error("prefill pipeline 2 does not support spec / layer_dump / null_layer hooks");
            if (!pf->supports_pipeline()) throw std::logic_error("prefill pipeline 2 needs an exact half-row PrefillPath");
            if (rows.empty()) throw std::invalid_argument("prefill pipeline 2 needs nonempty rows");
            for (unsigned r = 1; r < R; ++r)
                if (rows[r].slot != rows[0].slot || rows[r].pos != rows[0].pos + r)
                    throw std::invalid_argument("prefill pipeline 2 needs consecutive positions of one slot");
        }
        std::vector<std::uint32_t> tk(R), sl(R);
        std::vector<std::uint64_t> ps(R);
        for (unsigned r = 0; r < R; ++r) {
            tk[r] = rows[r].token;
            sl[r] = rows[r].slot;
            ps[r] = rows[r].pos;
            if (rows[r].slot >= opt.slots || rows[r].pos >= opt.capacity) throw std::out_of_range("row slot / position");
            if (rows[r].token >= kVocab) throw std::out_of_range("token id");
            // Rows of one slot: consecutive ascending positions in row order (the seams append before they read).
            for (unsigned q = r; q-- > 0;)
                if (rows[q].slot == rows[r].slot) {
                    if (rows[r].pos != rows[q].pos + 1) throw std::invalid_argument("rows of one slot need consecutive ascending positions");
                    break;
                }
        }
        hip_ok(hipMemcpyAsync(tok_d, tk.data(), R * 4, hipMemcpyHostToDevice, stream), "tokens");
        hip_ok(hipMemcpyAsync(slot_d, sl.data(), R * 4, hipMemcpyHostToDevice, stream), "slots");
        hip_ok(hipMemcpyAsync(pos_d, ps.data(), R * 8, hipMemcpyHostToDevice, stream), "pos");
        seg_n = 0;  // only step() collects; a verify / gate forward's marks are dropped here
        mark(segStart);
        embed4_kernel<<<dim3(kHidden / 256, R), 256, 0, stream>>>(embed, tok_d, X);
        hip_ok(hipGetLastError(), "embed");
        mark(segEmbed);
        if (!pfc && R > 8) throw std::invalid_argument("layers_forward: decode calls take 1..8 rows");
        if (R > std::max<unsigned>(8U, opt.max_rows)) throw std::invalid_argument("layers_forward: rows > max_rows");
        if (pfc && r1b) throw std::logic_error("r1b-all needs the reference prefill; the chunked prefill implements R1 prompts");
        select_exchange(kind, R);
        if (pfc && opt.pf_pipeline == 2) {
            layers_forward_pipeline(rows, n_layers);
            return;
        }
        const CallCtx c{&lib, rows, slot_d, pos_d, flag, stream, !pfc, sp};
        unsigned* const c1 = sp ? sp->row_flag : flag;  // C1 per row in spec mode
        const unsigned c1_stride = sp ? 1U : 0U;
        for (unsigned li = 0; li < n_layers; ++li) {
            LayerW& w = layers[li];
            cur_layer = li;
            if (!pfc && r1b && strix::glm::r1b::hc_norm_enabled()) {
                strix::glm::r1b::hc_pre_norm(X, w.hc_attn.fn, w.hc_attn.base, w.hc_attn.scale, F, mix, xc,
                                          w.in_norm, xn, R, stream);
            } else {
                if (pfc) glm::pf::hc_pre_rows(X, w.hc_attn.fn, w.hc_attn.base, w.hc_attn.scale, F, hz, mix, xc, R, stream);
                else hc_pre(X, w.hc_attn.fn, w.hc_attn.base, w.hc_attn.scale, F, mix, xc, R);
                D::rmsnorm(xc, w.in_norm, D::WType::BF16, kHidden, kEps, xn, R, stream);
            }
            mark(segHc);
            if (pfc) {
                if (w.kda) pf->kda(w.kdaw, w.kst, c, xn, part);
                else pf->dsa(w.dsaw, w.dst, c, xn, part);
            } else if (w.kda) {
                attn->kda(w.kdaw, w.kst, c, xn, part);
            } else {
                attn->dsa(w.dsaw, w.dst, c, xn, part);
            }
            mark(w.kda ? segKda : segDsa);
            boundary(c1, c1_stride, R, pfc);
            if (!pfc && r1b && strix::glm::r1b::hc_norm_enabled()) {
                strix::glm::r1b::hc_pre_norm(X, w.hc_ffn.fn, w.hc_ffn.base, w.hc_ffn.scale, F, mix, xc,
                                          w.post_norm, xn, R, stream);
            } else {
                if (pfc) glm::pf::hc_pre_rows(X, w.hc_ffn.fn, w.hc_ffn.base, w.hc_ffn.scale, F, hz, mix, xc, R, stream);
                else hc_pre(X, w.hc_ffn.fn, w.hc_ffn.base, w.hc_ffn.scale, F, mix, xc, R);
                D::rmsnorm(xc, w.post_norm, D::WType::BF16, kHidden, kEps, xn, R, stream);
            }
            mark(segHc);
            if (w.moe) {
                if (pfc) {  // chunk rows: the row-tiled router chain (== decode bits) + R14 route
                    glm::pf::dense_rows(xn, w.moew.router, D::WType::F16, kHidden, 0, kExperts, rlog, R, stream);
                    D::route(rlog, w.moew.bias, ids, wts, R, stream);
                    if (pf->moe_by_row()) {  // the engine's MoE seam, blocks of <= 8 rows: the decode identity (batch invariant)
                        const unsigned B = std::clamp(pf->moe_row_block(), 1U, 8U);
                        for (unsigned q = 0; q < R; q += B) {
                            const CallCtx cq{&lib, c.rows.subspan(q, std::min(B, R - q)), slot_d + q, pos_d + q, flag, stream, false, nullptr};
                            moe->forward(w.moew, cq, xn + static_cast<std::size_t>(q) * kHidden, ids + static_cast<std::size_t>(q) * kTopE,
                                         wts + static_cast<std::size_t>(q) * kTopE, part + static_cast<std::size_t>(q) * kHidden);
                        }
                    } else {
                        pf->moe(w.moew, c, xn, ids, wts, part);
                    }
                } else {
                    router_route(xn, w.moew, R);
                    moe->forward(w.moew, c, xn, ids, wts, part);
                }
            } else if (pfc) {
                pf->dense(w.dense, c, xn, dense_i, part);
            } else {
                lib.run(w.dense.gate, xn, dg, R, stream);
                lib.run(w.dense.up, xn, du, R, stream);
                D::swiglu(dg, du, da, static_cast<std::size_t>(R) * dense_i, stream);
                lib.run(w.dense.down, da, part, R, stream);
            }
            mark(w.moe ? segMoe : segDense);
            boundary(c1, c1_stride, R, pfc);
            if (layer_dump != nullptr)
                hip_ok(hipMemcpyAsync(layer_dump->data() + static_cast<std::size_t>(li) * kHc * kHidden, X, kHc * kHidden * 4,
                                      hipMemcpyDeviceToHost, stream), "layer dump");
            if (null_layer == static_cast<int>(li))  // test hook: +1 ulp on stream 0 of the row at null_pos
                for (unsigned q = 0; q < R; ++q)
                    if (rows[q].pos == null_pos) {
                        float* at = X + (static_cast<std::size_t>(q) * kHc) * kHidden + null_elem;
                        std::uint32_t bits = 0;
                        hip_ok(hipMemcpyAsync(&bits, at, 4, hipMemcpyDeviceToHost, stream), "null flip read");
                        hip_ok(hipStreamSynchronize(stream), "null flip sync");
                        bits += 1;
                        hip_ok(hipMemcpyAsync(at, &bits, 4, hipMemcpyHostToDevice, stream), "null flip write");
                        hip_ok(hipStreamSynchronize(stream), "null flip sync");
                    }
        }
        // Plain steps check the layers' checkpoints together with the head's (step), without a host round trip mid-step:
        // the flag is sticky, so a layer checkpoint still throws before the argmax exchange and before any token is
        // returned. Cross-rank symmetry is as before (each rank throws on its own flag).
        if (sp == nullptr && !(kind == ForwardKind::Plain && defer_layer_flag)) check_flag("layers");
    }

    std::vector<std::uint32_t> step(std::span<const Row> rows) {
        if (load_test) throw std::logic_error("STRIX_GLM_LOAD_TEST: load only");
        if (rows.empty() || rows.size() > 8) throw std::invalid_argument("step: 1..8 rows");
        if (opt.layers != kLayers && !smoke_head) throw std::logic_error("step needs all 45 layers (G3 uses gate_residual)");
        const auto R = static_cast<unsigned>(rows.size());
        if (opt.spec)  // mode change spec -> plain: apply the slots' pending KDA rows first (design notes section 1)
            for (const Row& r : rows)
                if (pending.at(r.slot) > 0) flush(r.slot);
        layers_forward(rows, opt.layers, ForwardKind::Plain);
        D::hc_final(X, hfin, R, stream);
        D::rmsnorm(hfin, final_norm, D::WType::BF16, kHidden, kEps, hidden, R, stream);
        lib.run(head, hidden, logits, R, stream);
        D::argmax(logits, vocab_n, vocab0, amv, ami, flag, R, stream);
        check_flag(defer_layer_flag ? "layers / head" : "head");
        xch->argmax(amv, ami, R, stream);
        mark(segHead);
        std::vector<std::uint32_t> out(R);
        hip_ok(hipMemcpyAsync(out.data(), ami, R * 4, hipMemcpyDeviceToHost, stream), "tokens out");
        hip_ok(hipStreamSynchronize(stream), "step sync");
        collect();
        ++prof.calls;
        return out;
    }

    // Per-layer residuals of row 0 (identity comparisons, one load): out[t][l][4][4096] after every layer l < n_layers.
    std::vector<float>* layer_dump = nullptr;
    std::vector<float> gate_residual_layers(std::span<const std::uint32_t> tokens, unsigned n_layers) {
        if (n_layers > opt.layers) throw std::invalid_argument("gate: more layers than loaded");
        const std::size_t per = static_cast<std::size_t>(n_layers) * kHc * kHidden;
        std::vector<float> out(tokens.size() * per), one(per);
        layer_dump = &one;
        for (std::size_t t = 0; t < tokens.size(); ++t) {
            const Row r{0, tokens[t], t};
            layers_forward({&r, 1}, n_layers, ForwardKind::Gate);
            hip_ok(hipStreamSynchronize(stream), "gate sync");
            xch->complete(stream);
            std::copy(one.begin(), one.end(), out.begin() + static_cast<std::ptrdiff_t>(t * per));
        }
        layer_dump = nullptr;
        return out;
    }
    std::vector<float> gate_residual(std::span<const std::uint32_t> tokens, unsigned n_layers) {
        if (n_layers > opt.layers) throw std::invalid_argument("gate: more layers than loaded");
        std::vector<float> out(tokens.size() * kHc * kHidden);
        for (std::size_t t = 0; t < tokens.size(); ++t) {
            const Row r{0, tokens[t], t};
            layers_forward({&r, 1}, n_layers, ForwardKind::Gate);
            hip_ok(hipMemcpyAsync(out.data() + t * kHc * kHidden, X, kHc * kHidden * 4, hipMemcpyDeviceToHost, stream), "residual");
            hip_ok(hipStreamSynchronize(stream), "gate sync");
            xch->complete(stream);  // every exchange of this forward verified before its residual is trusted
        }
        return out;
    }

    bool reference_prefill_installed() const { return dynamic_cast<const ReferencePrefill*>(pf.get()) != nullptr; }
    // Chunked prefill through the PrefillPath (prefill): chunks of max_rows rows; head on the last row only.
    std::uint32_t prefill_chunked(std::uint32_t slot, std::span<const std::uint32_t> tokens, std::uint64_t pos0) {
        if (tokens.empty()) throw std::invalid_argument("prefill: no tokens");
        if (opt.layers != kLayers && !smoke_head) throw std::logic_error("prefill needs all 45 layers (G3 uses gate_residual)");
        if (slot >= opt.slots) throw std::out_of_range("prefill slot");
        if (opt.spec && pending.at(slot) > 0) flush(slot);
        const std::size_t C = std::max<std::size_t>(1, opt.max_rows);
        no_mtp.at(slot) = 1;  // before the first chunk mutates the slot (an exception mid-prefill keeps MTP refused)
        std::vector<Row> rows;
        for (std::size_t c0 = 0; c0 < tokens.size(); c0 += C) {
            const std::size_t n = std::min(C, tokens.size() - c0);
            rows.resize(n);
            for (std::size_t i = 0; i < n; ++i) rows[i] = Row{slot, tokens[c0 + i], pos0 + c0 + i};
            layers_forward(rows, opt.layers, ForwardKind::Prefill);
        }
        const unsigned last = static_cast<unsigned>(rows.size() - 1);
        D::hc_final(X + static_cast<std::size_t>(last) * kHc * kHidden, hfin, 1, stream);
        D::rmsnorm(hfin, final_norm, D::WType::BF16, kHidden, kEps, hidden, 1, stream);
        lib.run(head, hidden, logits, 1, stream);
        D::argmax(logits, vocab_n, vocab0, amv, ami, flag, 1, stream);
        check_flag("head");
        xch->argmax(amv, ami, 1, stream);
        std::uint32_t out = 0;
        hip_ok(hipMemcpyAsync(&out, ami, 4, hipMemcpyDeviceToHost, stream), "token out");
        hip_ok(hipStreamSynchronize(stream), "prefill sync");
        ++prof.calls;
        return out;
    }

    Engine::MtpPrefillOut prefill_mtp(std::uint32_t slot, std::span<const std::uint32_t> tokens,
                                     std::uint64_t pos0, Exchange::MergeCtl& ctl) {
        if (load_test) throw std::logic_error("STRIX_GLM_LOAD_TEST: load only");
        if (!opt.mtp || layers.empty() || !layers.back().mtp || opt.layers != kLayers || final_norm == nullptr)
            throw std::logic_error("prefill_mtp needs all 45 trunk layers and MTP");
        if (!pf->exact_linears() || reference_prefill_installed())
            throw std::logic_error("prefill_mtp requires an exact chunked PrefillPath (gemv)");
        if (tokens.empty()) throw std::invalid_argument("prefill_mtp: no tokens");
        if (slot >= opt.slots || pos0 > opt.capacity || tokens.size() > opt.capacity - pos0)
            throw std::out_of_range("prefill_mtp: slot / capacity");
        for (auto t : tokens)
            if (t >= kVocab) throw std::out_of_range("token id");
        if (opt.spec && pending.at(slot) > 0) flush(slot);
        no_mtp.at(slot) = 1;  // stays poisoned until the final single-row seed also succeeds
        const std::size_t C = std::max<std::size_t>(1, opt.max_rows);
        std::vector<Row> rows;
        unsigned last = 0, interior_flag = 0;
        for (std::size_t c0 = 0; c0 < tokens.size(); c0 += C) {
            const auto R = static_cast<unsigned>(std::min(C, tokens.size() - c0));
            rows.resize(R);
            for (unsigned j = 0; j < R; ++j) rows[j] = Row{slot, tokens[c0 + j], pos0 + c0 + j};
            layers_forward(rows, opt.layers, ForwardKind::Prefill);
            D::hc_final(X, hfin, R, stream);
            D::rmsnorm(hfin, final_norm, D::WType::BF16, kHidden, kEps, hidden, R, stream);
            // Like the plain chunked prefill, interior rows run no vocabulary head (trunk or MTP): their ids are
            // discarded and drafts are always verified, so only the final row's head (y0) and the seed drafter
            // below need logits. Per-row vocabulary heads cost ~15 % of the prefill at 8K/32K.
            // Look ahead in the full prompt: a chunk's last hidden pairs with the next chunk's first token.
            // Only the final prompt row waits for the trunk head's y0.
            const auto pairs = static_cast<unsigned>(std::min<std::size_t>(R, tokens.size() - c0 - 1));
            if (pairs != 0) {
                mtp_cache(slot, hidden, tokens.subspan(c0 + 1, pairs), pos0 + c0, true);
                unsigned f = 0;
                hip_ok(hipGetLastError(), "MTP prefill kernels");
                hip_ok(hipMemcpyAsync(&f, mtp_flag, 4, hipMemcpyDeviceToHost, stream), "MTP prefill flag");
                hip_ok(hipStreamSynchronize(stream), "MTP prefill sync");
                xch->complete(stream);
                interior_flag |= f;  // propagated through the final seed merge on both ranks
            }
            last = R - 1;
        }
        // Preserve last_hidden()'s single final-row contract before the existing head and MTP seed paths.
        if (last != 0)
            hip_ok(hipMemcpyAsync(hidden, hidden + static_cast<std::size_t>(last) * kHidden, kHidden * 4,
                                  hipMemcpyDeviceToDevice, stream), "MTP prefill last hidden");
        lib.run(head, hidden, logits, 1, stream);
        D::argmax(logits, vocab_n, vocab0, amv, ami, flag, 1, stream);
        check_flag("head");
        xch->argmax(amv, ami, 1, stream);
        Engine::MtpPrefillOut out;
        hip_ok(hipMemcpyAsync(&out.y0, ami, 4, hipMemcpyDeviceToHost, stream), "MTP prefill y0");
        hip_ok(hipStreamSynchronize(stream), "MTP prefill head sync");
        hip_ok(hipMemcpyAsync(trunk_logits, logits, std::size_t{vocab_n} * sizeof(float),
                              hipMemcpyDeviceToDevice, stream), "save trunk logits");
        out.mtp = mtp(slot, hidden, {&out.y0, 1}, pos0 + tokens.size() - 1, ctl, true, interior_flag);
        hip_ok(hipMemcpyAsync(logits, trunk_logits, std::size_t{vocab_n} * sizeof(float),
                              hipMemcpyDeviceToDevice, stream), "restore trunk logits");
        hip_ok(hipStreamSynchronize(stream), "trunk logits sync");
        out.hp = m_h;
        no_mtp.at(slot) = 0;
        ++prof.calls;
        return out;
    }

    void reset(std::uint32_t slot) {
        for (LayerW& w : layers) {
            if (w.kda) {
                const std::size_t s = static_cast<std::size_t>(kda_heads) * kKdaDim * kKdaDim, c = 3ULL * kda_heads * kKdaDim * 3;
                hip_ok(hipMemsetAsync(w.kst.S + slot * s, 0, s * 4, stream), "reset S");
                hip_ok(hipMemsetAsync(w.kst.conv + slot * c, 0, c * 4, stream), "reset conv");
            } else {
                const std::size_t cap = w.dst.cap;
                hip_ok(hipMemsetAsync(w.dst.latent + slot * cap * kKvLora, 0, cap * kKvLora * 2, stream), "reset latent");
                hip_ok(hipMemsetAsync(w.dst.ikey + slot * cap * kIdxDim, 0, cap * kIdxDim * 2, stream), "reset key");
                hip_ok(hipMemsetAsync(w.dst.igate + slot * cap * kIdxDim, 0, cap * kIdxDim * 2, stream), "reset gate");
                hip_ok(hipMemsetAsync(w.dst.pooled + slot * (cap / kKpool) * kIdxDim, 0, (cap / kKpool) * kIdxDim * 4, stream), "reset pooled");
            }
        }
        hip_ok(hipStreamSynchronize(stream), "reset");
        if (opt.spec) pending.at(slot) = 0;  // the zeroed state needs no replay
        if (slot < no_mtp.size()) no_mtp[slot] = 0;
    }

    // Shared layer-45 body: chunked teacher forcing and the single-row drafter use the same weights/state.
    // Only the caller runs the LM head; no chunk-sized vocabulary scratch is needed.
    void mtp_cache(std::uint32_t slot, const float* h, std::span<const std::uint32_t> tokens,
                   std::uint64_t pos0, bool pfc) {
        const auto R = static_cast<unsigned>(tokens.size());
        select_exchange(pfc ? ForwardKind::Prefill : ForwardKind::Mtp, R);
        LayerW& w = layers.back();
        // Keep asynchronous metadata uploads alive through the caller's head/flag synchronization.
        mtp_slots.assign(R, slot);
        mtp_positions.resize(R);
        mtp_rows.resize(R);
        auto& sl = mtp_slots;
        auto& ps = mtp_positions;
        auto& rows = mtp_rows;
        for (unsigned j = 0; j < R; ++j) {
            if (tokens[j] >= kVocab) throw std::out_of_range("token id");
            ps[j] = pos0 + j;
            rows[j] = Row{slot, tokens[j], pos0 + j};
        }
        hip_ok(hipMemsetAsync(mtp_flag, 0, 4, stream), "mtp flag");
        hip_ok(hipMemcpyAsync(tok_d, tokens.data(), R * 4, hipMemcpyHostToDevice, stream), "mtp tokens");
        hip_ok(hipMemcpyAsync(slot_d, sl.data(), R * 4, hipMemcpyHostToDevice, stream), "mtp slots");
        hip_ok(hipMemcpyAsync(pos_d, ps.data(), R * 8, hipMemcpyHostToDevice, stream), "mtp pos");
        const CallCtx c{&lib, rows, slot_d, pos_d, mtp_flag, stream, !pfc, nullptr};
        embed1_kernel<<<dim3(kHidden / 256, R), 256, 0, stream>>>(embed, tok_d, m_e);
        D::rmsnorm(m_e, w.enorm, D::WType::BF16, kHidden, kEps, m_en, R, stream);
        D::rmsnorm(h, w.hnorm, D::WType::BF16, kHidden, kEps, m_hn, R, stream);
        cat2_kernel<<<dim3(kHidden / 256, R), 256, 0, stream>>>(m_en, m_hn, m_cat);
        if (pfc) pf->linear_exact(w.eh_proj, c, m_cat, m_z, 2 * kHidden, kHidden);
        else lib.run(w.eh_proj, m_cat, m_z, R, stream);  // [enorm(e) | hnorm(h)] -> z
        D::rmsnorm(m_z, w.in_norm, D::WType::BF16, kHidden, kEps, xn, R, stream);
        if (pfc) pf->dsa(w.dsaw, w.dst, c, xn, part);
        else attn->dsa(w.dsaw, w.dst, c, xn, part);
        xch->reduce(part, R, stream);
        const std::size_t n = static_cast<std::size_t>(R) * kHidden;
        add_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(m_z, part, m_r, n);  // r = z + attn
        D::rmsnorm(m_r, w.post_norm, D::WType::BF16, kHidden, kEps, xn, R, stream);
        if (pfc) {
            glm::pf::dense_rows(xn, w.moew.router, D::WType::F16, kHidden, 0, kExperts, rlog, R, stream);
            D::route(rlog, w.moew.bias, ids, wts, R, stream);
            if (pf->moe_by_row()) {
                const unsigned B = std::clamp(pf->moe_row_block(), 1U, 8U);
                for (unsigned q = 0; q < R; q += B) {
                    const CallCtx cq{&lib, c.rows.subspan(q, std::min(B, R - q)), slot_d + q, pos_d + q,
                                     mtp_flag, stream, false, nullptr};
                    moe->forward(w.moew, cq, xn + static_cast<std::size_t>(q) * kHidden,
                                 ids + static_cast<std::size_t>(q) * kTopE, wts + static_cast<std::size_t>(q) * kTopE,
                                 part + static_cast<std::size_t>(q) * kHidden);
                }
            } else {
                pf->moe(w.moew, c, xn, ids, wts, part);
            }
        } else {
            router_route(xn, w.moew, R);
            moe->forward(w.moew, c, xn, ids, wts, part);
        }
        xch->reduce(part, R, stream);
        add_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(m_r, part, m_y, n);  // y = r + MoE
        D::rmsnorm(m_y, w.head_norm, D::WType::BF16, kHidden, kEps, m_h, R, stream);  // h' = shared_head.norm(y)
    }

    // ---- MTP drafter (MTP arithmetic consensus; design notes section 3)
    Engine::MtpOut mtp(std::uint32_t slot, const float* h, std::span<const std::uint32_t> tokens, std::uint64_t pos0,
                       Exchange::MergeCtl& ctl, bool bootstrap = false, unsigned inherited_flag = 0) {
        if (!opt.mtp || layers.empty() || !layers.back().mtp) throw std::logic_error("mtp needs Options::mtp (layer 45 loaded)");
        if (final_norm == nullptr) throw std::logic_error("mtp needs the head");
        const auto R = static_cast<unsigned>(tokens.size());
        if (R == 0 || R > 8) throw std::invalid_argument("mtp: 1..8 rows");
        if (slot >= opt.slots || pos0 + R > opt.capacity) throw std::out_of_range("mtp: slot / MTP capacity (clamp first)");
        if (no_mtp.at(slot) && !bootstrap) throw std::logic_error("mtp: the slot was chunk-prefilled (MTP cache not filled; reset first)");
        mtp_cache(slot, h, tokens, pos0, false);
        hip_ok(hipMemcpyAsync(m_hs + static_cast<std::size_t>(slot) * 8 * kHidden, m_h, std::size_t{R} * kHidden * sizeof(float),
                              hipMemcpyDeviceToDevice, stream), "mtp hidden per slot");
        lib.run(head, m_h, logits, R, stream);
        D::argmax(logits, vocab_n, vocab0, amv, ami, mtp_flag, R, stream);
        lse_kernel<<<1, 256, 0, stream>>>(logits + static_cast<std::size_t>(R - 1) * vocab_n, vocab_n, m_mz);
        hip_ok(hipGetLastError(), "mtp kernels");
        std::vector<float> v(R), mz(2);
        std::vector<std::uint32_t> id(R);
        unsigned f = 0;
        hip_ok(hipMemcpyAsync(v.data(), amv, R * 4, hipMemcpyDeviceToHost, stream), "mtp v");
        hip_ok(hipMemcpyAsync(id.data(), ami, R * 4, hipMemcpyDeviceToHost, stream), "mtp id");
        hip_ok(hipMemcpyAsync(mz.data(), m_mz, 8, hipMemcpyDeviceToHost, stream), "mtp mz");
        hip_ok(hipMemcpyAsync(&f, mtp_flag, 4, hipMemcpyDeviceToHost, stream), "mtp flag");
        hip_ok(hipStreamSynchronize(stream), "mtp sync");
        std::vector<Exchange::MergeRow> m(R);
        for (unsigned j = 0; j < R; ++j) m[j] = Exchange::MergeRow{v[j], id[j], f | inherited_flag};
        ctl.replay_flag = 0;
        ctl.mtp_max = mz[0];
        ctl.mtp_sum = mz[1];
        xch->merge(m.data(), R, ctl, stream);
        Engine::MtpOut out;
        for (unsigned j = 0; j < R; ++j) {
            out.id.push_back(m[j].id);
            out.flagged = out.flagged || m[j].flag != 0;
        }
        // the argmax logit equals the merged max M, so its probability is exp(M - M) / Z = 1 / Z
        out.confidence = ctl.mtp_sum > 0.0F ? 1.0F / ctl.mtp_sum : 0.0F;
        return out;
    }

    // ---- speculative verify (design notes v3, approved arithmetic contract)
    Engine::VerifyOut verify(std::uint32_t slot, std::span<const std::uint32_t> tokens, std::uint64_t p, Exchange::MergeCtl& ctl) {
        if (!opt.spec) throw std::logic_error("verify needs Options::spec");
        if (opt.layers != kLayers && !smoke_head) throw std::logic_error("verify needs all 45 layers");
        const auto R = static_cast<unsigned>(tokens.size());
        if (R == 0 || R > 8) throw std::invalid_argument("verify: 1..8 rows");
        if (slot >= opt.slots || p + R > opt.capacity) throw std::out_of_range("verify: slot / capacity (clamp K first)");
        hip_ok(hipMemsetAsync(row_flag, 0, R * 4, stream), "row flags");
        hip_ok(hipMemsetAsync(replay_flag, 0, 4, stream), "replay flag");
        const SpecRows sp{slot, pending[slot], cur[slot], row_flag, replay_flag};
        pending[slot] = 0;  // consumed by this call whatever the outcome (stored by the replay); commit sets it again
        std::vector<Row> rows(R);
        for (unsigned j = 0; j < R; ++j) rows[j] = Row{slot, tokens[j], p + j};
        return verify_rows(rows, sp, ctl);
    }
    bool verify_multi_supported() const { return attn && attn->supports_multi_slot_verify(); }
    Engine::VerifyOut verify_multi(std::span<const Row> rows, Exchange::MergeCtl& ctl) {
        if (!opt.spec) throw std::logic_error("verify needs Options::spec");
        if (opt.layers != kLayers && !smoke_head) throw std::logic_error("verify needs all 45 layers");
        const auto R = static_cast<unsigned>(rows.size());
        if (R == 0 || R > 8) throw std::invalid_argument("verify_multi: 1..8 rows");
        std::vector<SpecSeg> seg;
        for (unsigned r = 0; r < R; ++r) {
            if (rows[r].slot >= opt.slots || rows[r].pos >= opt.capacity) throw std::out_of_range("verify_multi: slot / capacity");
            if (r > 0 && rows[r].slot == rows[r - 1].slot) {
                if (rows[r].pos != rows[r - 1].pos + 1) throw std::invalid_argument("verify_multi: a slot's rows need consecutive positions");
                ++seg.back().n;
                continue;
            }
            if (r > 0 && rows[r].slot < rows[r - 1].slot) throw std::invalid_argument("verify_multi: rows not sorted by slot");
            seg.push_back(SpecSeg{rows[r].slot, 0, 0, r, 1});
        }
        if (seg.size() == 1) {
            std::vector<std::uint32_t> t(R);
            for (unsigned r = 0; r < R; ++r) t[r] = rows[r].token;
            return verify(seg[0].slot, t, rows[0].pos, ctl);
        }
        if (!verify_multi_supported()) throw std::logic_error("verify_multi: the attention seam verifies one slot per call");
        hip_ok(hipMemsetAsync(row_flag, 0, R * 4, stream), "row flags");
        hip_ok(hipMemsetAsync(replay_flag, 0, 4, stream), "replay flag");
        for (auto& g : seg) {
            g.n_rep = pending[g.slot];
            g.cur = cur[g.slot];
            pending[g.slot] = 0;  // consumed whatever the outcome; each slot's commit sets it again
        }
        SpecRows sp{seg[0].slot, seg[0].n_rep, seg[0].cur, row_flag, replay_flag};
        sp.seg = seg.data();
        sp.nseg = static_cast<unsigned>(seg.size());
        return verify_rows(rows, sp, ctl);
    }
    // The verify forward + head + merge (rows prepared by verify / verify_multi, flags cleared).
    Engine::VerifyOut verify_rows(std::span<const Row> rows, const SpecRows& sp, Exchange::MergeCtl& ctl) {
        const auto R = static_cast<unsigned>(rows.size());
        layers_forward(rows, opt.layers, ForwardKind::Verify, &sp);
        D::hc_final(X, hfin, R, stream);
        D::rmsnorm(hfin, final_norm, D::WType::BF16, kHidden, kEps, hidden, R, stream);
        lib.run(head, hidden, logits, R, stream);
        D::argmax(logits, vocab_n, vocab0, amv, ami, row_flag, R, stream, 1);
        std::vector<float> v(R);
        std::vector<std::uint32_t> id(R), fl(R);
        unsigned rf = 0;
        hip_ok(hipMemcpyAsync(v.data(), amv, R * 4, hipMemcpyDeviceToHost, stream), "verify v");
        hip_ok(hipMemcpyAsync(id.data(), ami, R * 4, hipMemcpyDeviceToHost, stream), "verify id");
        hip_ok(hipMemcpyAsync(fl.data(), row_flag, R * 4, hipMemcpyDeviceToHost, stream), "verify flags");
        hip_ok(hipMemcpyAsync(&rf, replay_flag, 4, hipMemcpyDeviceToHost, stream), "verify replay flag");
        hip_ok(hipStreamSynchronize(stream), "verify sync");
        std::vector<Exchange::MergeRow> m(R);
        for (unsigned j = 0; j < R; ++j) m[j] = Exchange::MergeRow{v[j], id[j], fl[j]};
        ctl.replay_flag = rf;
        xch->merge(m.data(), R, ctl, stream);
        Engine::VerifyOut out;
        out.replay_flag = ctl.replay_flag != 0;
        for (unsigned j = 0; j < R; ++j) {
            out.id.push_back(m[j].id);
            out.flag.push_back(m[j].flag);
        }
        ++prof.calls;
        return out;
    }
    void commit(std::uint32_t slot, unsigned rows, std::uint64_t) {
        if (!opt.spec) throw std::logic_error("commit needs Options::spec");
        if (rows == 0 || rows > 8) throw std::invalid_argument("commit: 1..8 rows");
        cur.at(slot) = 1 - cur.at(slot);  // the verify's new rows (set 1 - cur) become the replay set
        pending.at(slot) = rows;
        // DSA: frontier only (positions >= p + rows are overwritten by later appends); MTP truncation lands with mtp.
    }
    void flush(std::uint32_t slot) {
        if (!opt.spec || pending.at(slot) == 0) return;
        const unsigned C = kda_heads * kKdaDim, n = pending[slot], cs = cur[slot];
        const std::size_t SZ = static_cast<std::size_t>(kda_heads) * kKdaDim * kKdaDim, CZ = 3ULL * C * 3;
        hip_ok(hipMemsetAsync(replay_flag, 0, 4, stream), "flush flag");
        for (LayerW& w : layers) {
            if (!w.kda) continue;
            const D::KdaDev kd{kda_heads, w.kdaw.conv, w.kdaw.A_log, w.kdaw.dt_bias, w.kdaw.o_norm};
            const std::size_t set = (static_cast<std::size_t>(slot) * 2 + cs) * 8;
            const std::size_t nxt = (static_cast<std::size_t>(slot) * 2 + 1 - cs) * 8;
            D::kda_verify(kd, w.kst.rep_qkv + set * 3 * C, w.kst.rep_scr + set * kda_heads * 4 * kKdaDim,
                          w.kst.rep_beta + set * kda_heads, n, nullptr, nullptr, nullptr, nullptr, 0, w.kst.S + slot * SZ,
                          w.kst.conv + slot * CZ, w.kst.rep_scr + nxt * kda_heads * 4 * kKdaDim, w.kst.rep_beta + nxt * kda_heads,
                          part, part, replay_flag, row_flag, stream);
        }
        pending[slot] = 0;
        unsigned rf = 0;
        hip_ok(hipMemcpyAsync(&rf, replay_flag, 4, hipMemcpyDeviceToHost, stream), "flush flag");
        hip_ok(hipStreamSynchronize(stream), "flush sync");
        if (rf) throw std::runtime_error("non-finite value in a committed KDA row (C2) at flush");
    }

    std::vector<std::uint8_t> state_bytes(std::uint32_t slot, std::uint64_t frontier = ~std::uint64_t{0}) {
        std::vector<std::uint8_t> out;
        auto grab = [&](const void* p, std::size_t bytes) {
            const std::size_t o = out.size();
            out.resize(o + bytes);
            hip_ok(hipMemcpy(out.data() + o, p, bytes, hipMemcpyDeviceToHost), "state");
        };
        for (const LayerW& w : layers) {
            if (w.kda) {
                const std::size_t s = static_cast<std::size_t>(kda_heads) * kKdaDim * kKdaDim, c = 3ULL * kda_heads * kKdaDim * 3;
                grab(w.kst.S + slot * s, s * 4);
                grab(w.kst.conv + slot * c, c * 4);
            } else {
                const std::size_t cap = w.dst.cap, n = std::min<std::uint64_t>(cap, frontier);
                grab(w.dst.latent + slot * cap * kKvLora, n * kKvLora * 2);
                grab(w.dst.ikey + slot * cap * kIdxDim, n * kIdxDim * 2);
                grab(w.dst.igate + slot * cap * kIdxDim, n * kIdxDim * 2);
                grab(w.dst.pooled + slot * (cap / kKpool) * kIdxDim, (n / kKpool) * kIdxDim * 4);
            }
        }
        return out;
    }

    void restore_state(std::uint32_t slot, std::span<const std::uint8_t> in, std::uint64_t frontier) {
        if (slot >= opt.slots) throw std::out_of_range("restore_state: slot");
        reset(slot);
        std::size_t o = 0;
        auto put = [&](void* p, std::size_t bytes) {
            if (o + bytes > in.size()) throw std::invalid_argument("restore_state: state shorter than this engine's layout");
            hip_ok(hipMemcpy(p, in.data() + o, bytes, hipMemcpyHostToDevice), "restore");
            o += bytes;
        };
        for (LayerW& w : layers) {
            if (w.kda) {
                const std::size_t s = static_cast<std::size_t>(kda_heads) * kKdaDim * kKdaDim, c = 3ULL * kda_heads * kKdaDim * 3;
                put(w.kst.S + slot * s, s * 4);
                put(w.kst.conv + slot * c, c * 4);
            } else {
                const std::size_t cap = w.dst.cap, n = std::min<std::uint64_t>(cap, frontier);
                put(w.dst.latent + slot * cap * kKvLora, n * kKvLora * 2);
                put(w.dst.ikey + slot * cap * kIdxDim, n * kIdxDim * 2);
                put(w.dst.igate + slot * cap * kIdxDim, n * kIdxDim * 2);
                put(w.dst.pooled + slot * (cap / kKpool) * kIdxDim, (n / kKpool) * kIdxDim * 4);
            }
        }
        if (o != in.size()) throw std::invalid_argument("restore_state: state longer than this engine's layout");
    }
};

Engine::Engine(Options options, dist::Communicator* comm) : impl_(std::make_unique<Impl>(std::move(options), comm)) {}
Engine::~Engine() = default;
LoadReport Engine::load(const std::function<unsigned(const mem::LoadAdmission&)>& select_max_rows) { return impl_->load(select_max_rows); }
std::uint32_t Engine::prefill(std::uint32_t slot, std::span<const std::uint32_t> tokens, std::uint64_t pos0) {
    // Cover every forward issued by either path, including prompt-time MTP work.
    struct PromptIdentityScope {
        bool& current;
        bool saved, saved_active;
        PromptIdentityScope(bool& value, bool prompt_r1)
            : current(value), saved(value), saved_active(strix::glm::r1b::active()) {
            if (prompt_r1) {
                current = false;
                strix::glm::r1b::set_active(false);
            }
        }
        ~PromptIdentityScope() {
            current = saved;
            strix::glm::r1b::set_active(saved_active);
        }
    } scope(impl_->r1b, impl_->identity == IdentityMode::kR1b);
    if (!impl_->reference_prefill_installed()) return impl_->prefill_chunked(slot, tokens, pos0);
    // Reference identity (arithmetic contract 6.2): prefill = decode, one row at a time; returns the greedy token after the last.
    std::uint32_t next = 0;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const Row r{slot, tokens[i], pos0 + i};
        next = impl_->step({&r, 1})[0];
    }
    return next;
}
Engine::MtpPrefillOut Engine::prefill_mtp(std::uint32_t slot, std::span<const std::uint32_t> tokens,
                                        std::uint64_t pos0, Exchange::MergeCtl& ctl) {
    if (impl_->identity == IdentityMode::kR1bAll)
        throw std::logic_error("prefill_mtp requires R1 prompts (r1 or r1b)");
    struct Scope {
        Impl& impl;
        bool saved, active;
        explicit Scope(Impl& i) : impl(i), saved(i.r1b), active(strix::glm::r1b::active()) {
            impl.r1b = false;
            strix::glm::r1b::set_active(false);
        }
        ~Scope() {
            impl.r1b = saved;
            strix::glm::r1b::set_active(active);
        }
    } scope(*impl_);
    return impl_->prefill_mtp(slot, tokens, pos0, ctl);
}
std::vector<std::uint32_t> Engine::step(std::span<const Row> rows) { return impl_->step(rows); }
void Engine::reset(std::uint32_t slot) { impl_->reset(slot); }
void Engine::set_attn(std::shared_ptr<AttnPath> p) { impl_->attn = std::move(p); }
void Engine::set_null_flip(int layer, std::uint64_t pos, unsigned elem) {
    if (layer >= 0 && (layer >= static_cast<int>(impl_->layers.size()) || elem >= kHidden)) throw std::invalid_argument("null flip: layer / elem");
    impl_->null_layer = layer;
    impl_->null_pos = pos;
    impl_->null_elem = elem;
}
std::uint64_t Engine::moe_cold_bytes() const {
    std::uint64_t n = 0;
    for (const auto& w : impl_->layers)
        if (w.moe) n += impl_->moe->warm_bytes(w.moew);
    return n;
}
void Engine::set_moe(std::shared_ptr<MoePath> p) {
    // With a prefill path installed, the new seam's tables are admitted and built first (a failure keeps the old seam).
    if (p && impl_->loaded && impl_->pf_installed && impl_->pf) impl_->warm_moe(*p, impl_->pf->admission_margin());
    impl_->moe = std::move(p);
}
void Engine::set_identity_mode(IdentityMode mode) {
    (void)identity_mode_name(mode);
    impl_->identity = mode;
    impl_->r1b = mode != IdentityMode::kR1;
    strix::glm::r1b::set_active(impl_->r1b);
}
IdentityMode Engine::identity_mode() const { return impl_->identity; }
void Engine::set_identity(bool r1b) { set_identity_mode(r1b ? IdentityMode::kR1bAll : IdentityMode::kR1); }
bool Engine::identity_r1b() const { return impl_->identity != IdentityMode::kR1; }
void Engine::set_route_log(bool on) {
    if (on && impl_->route_log_d == nullptr) {
        impl_->route_log_d = dalloc<std::uint32_t>(static_cast<std::size_t>(kLayers + 1) * kTopE);
        hip_ok(hipMemset(impl_->route_log_d, 0xFF, static_cast<std::size_t>(kLayers + 1) * kTopE * 4), "route log init");
    }
    impl_->route_log = on;
}
std::vector<float> Engine::gate_residual_layers(std::span<const std::uint32_t> tokens, unsigned n_layers) {
    return impl_->gate_residual_layers(tokens, n_layers);
}
std::vector<std::uint32_t> Engine::route_log() const {
    std::vector<std::uint32_t> v(static_cast<std::size_t>(kLayers + 1) * kTopE, 0xFFFFFFFFU);
    if (impl_->route_log_d == nullptr) return v;
    hip_ok(hipStreamSynchronize(impl_->stream), "route log sync");
    hip_ok(hipMemcpy(v.data(), impl_->route_log_d, v.size() * 4, hipMemcpyDeviceToHost), "route log copy");
    return v;
}
void Engine::set_exchange(std::shared_ptr<Exchange> p) {
    if (!p) throw std::invalid_argument("set_exchange: null exchange");
    impl_->switch_exchange(p);
    impl_->fallback_xch = std::move(p);
    impl_->decode_xch.reset();  // explicit seam overrides retain their original all-forward meaning
}
std::shared_ptr<Exchange> Engine::exchange() const { return impl_->fallback_xch; }
bool Engine::decode_xfer_enabled() const { return impl_->decode_xch != nullptr; }
unsigned Engine::prefill_pipeline() const { return impl_->opt.pf_pipeline; }
void Engine::set_prefill(std::shared_ptr<PrefillPath> p) {
    // Before load() the path is only stored; load() installs it once the weights exist (Impl::install_prefill).
    if (p && impl_->loaded) impl_->install_prefill(*p);
    impl_->pf = std::move(p);
    impl_->pf_installed = impl_->pf != nullptr;
}
Engine::VerifyOut Engine::verify(std::uint32_t slot, std::span<const std::uint32_t> tokens, std::uint64_t p, Exchange::MergeCtl& ctl) {
    return impl_->verify(slot, tokens, p, ctl);
}
void Engine::commit(std::uint32_t slot, unsigned rows, std::uint64_t p) { impl_->commit(slot, rows, p); }
void Engine::flush(std::uint32_t slot) { impl_->flush(slot); }
const float* Engine::last_hidden() const { return impl_->hidden; }
const float* Engine::last_logits() const { return impl_->logits; }
std::uint32_t Engine::vocab_slice() const { return impl_->vocab_n; }
void Engine::save_trunk_logits() {
    if (!impl_->trunk_logits) throw std::logic_error("save_trunk_logits requires MTP");
    hip_ok(hipMemcpyAsync(impl_->trunk_logits, impl_->logits, std::size_t{impl_->vocab_n} * sizeof(float),
                          hipMemcpyDeviceToDevice, impl_->stream), "save trunk logits");
}
void Engine::restore_trunk_logits() {
    if (!impl_->trunk_logits) throw std::logic_error("restore_trunk_logits requires MTP");
    hip_ok(hipMemcpyAsync(impl_->logits, impl_->trunk_logits, std::size_t{impl_->vocab_n} * sizeof(float),
                          hipMemcpyDeviceToDevice, impl_->stream), "restore trunk logits");
    hip_ok(hipStreamSynchronize(impl_->stream), "trunk logits sync");
}
std::vector<std::uint8_t> Engine::state_bytes_upto(std::uint32_t slot, std::uint64_t frontier) const {
    return impl_->state_bytes(slot, frontier);
}
void Engine::restore_state(std::uint32_t slot, std::span<const std::uint8_t> bytes, std::uint64_t frontier) {
    impl_->restore_state(slot, bytes, frontier);
}
Engine::MtpOut Engine::mtp(std::uint32_t slot, const float* h, std::span<const std::uint32_t> tokens, std::uint64_t mtp_pos0,
                           Exchange::MergeCtl& ctl) {
    return impl_->mtp(slot, h, tokens, mtp_pos0, ctl);
}
const float* Engine::mtp_hidden() const { return impl_->m_h; }
const float* Engine::mtp_hidden(std::uint32_t slot) const {
    if (slot >= impl_->opt.slots || impl_->m_hs == nullptr) throw std::out_of_range("mtp_hidden: slot / MTP not loaded");
    return impl_->m_hs + static_cast<std::size_t>(slot) * 8 * kHidden;
}
Engine::VerifyOut Engine::verify_multi(std::span<const Row> rows, Exchange::MergeCtl& ctl) { return impl_->verify_multi(rows, ctl); }
bool Engine::verify_multi_supported() const { return impl_->verify_multi_supported(); }
std::vector<float> Engine::gate_residual(std::span<const std::uint32_t> tokens, unsigned layers, bool) {
    return impl_->gate_residual(tokens, layers);
}
std::vector<std::uint8_t> Engine::state_bytes(std::uint32_t slot) const { return impl_->state_bytes(slot); }
void Engine::set_profiling(bool on) { impl_->profiling = on; }
const Profile& Engine::profile() const { return impl_->prof; }
std::uint32_t Engine::rank() const { return impl_->opt.rank; }

}  // namespace strix::glm::engine
