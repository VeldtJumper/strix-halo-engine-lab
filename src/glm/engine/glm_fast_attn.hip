// GLM engine: the attention seam on the fast decode kernels (src/glm/attn, review finding, gate
// 40/40 bitwise vs the Tier A host emulation and the P1 kernels). Same op sequence per row as DefaultAttn (engine.hip):
// Tier B linears through the engine's LinearRunner, Tier A through glm::fast; rows go in chunks of <= 8 sharing every
// weight read. KDA: one launch when the chunk's slots are consecutive and distinct, kda_steps_seq for consecutive
// positions of one slot, otherwise one launch per row in row order (the default's sequential semantics). DSA row
// descriptors are built on the device from CallCtx::slot_d / pos_d (no host copies). STRIX_GLM_ATTN=ref keeps DefaultAttn.
#include "glm/engine/glm_fast_attn.hpp"
#include "glm/engine/glm_r1b.hpp"

#include "glm/attn/glm_attn_fast.hpp"
#include "glm/attn/glm_attn_tierb.hpp"
#include "glm/engine/glm_lib.hpp"
#include "glm/spec/glm_spec_dev.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace strix::glm::engine {

namespace {

using namespace detail;
namespace D = strix::glm::dev;
namespace F = strix::glm::fast;
namespace TB = strix::glm::attn_tierb;

// Tier B dispatch: raw-only linears (engine loader, attn_tierb) go through the fused attention groups, prepared ones
// through the engine's LinearRunner (strix_gemv); both give the strix_gemv bits.
bool raw(const Linear& l) { return l.prepared == nullptr && l.raw.trellis != nullptr; }
TB::Mat mat(const Linear& l) { return TB::Mat{l.raw.trellis, l.raw.suh, l.raw.svh, l.raw.input, l.raw.output, l.raw.bits}; }

__global__ void dsa_rows_kernel(const std::uint32_t* slot, const std::uint64_t* pos, std::uint16_t* latent, std::uint16_t* ikey,
                                std::uint16_t* igate, float* pooled, unsigned cap, D::DsaSlot* out, unsigned* tpos, unsigned n) {
    const unsigned r = threadIdx.x;
    if (r >= n) return;
    const std::size_t s = slot[r];
    out[r] = D::DsaSlot{latent + s * cap * kKvLora, ikey + s * cap * kIdxDim, igate + s * cap * kIdxDim,
                        pooled + s * (cap / kKpool) * kIdxDim, cap};
    tpos[r] = static_cast<unsigned>(pos[r]);
}

class FastAttn final : public AttnPath {
public:
    FastAttn(unsigned kda_heads, unsigned mla_heads, std::uint64_t cap)
        : Hk_(kda_heads), Hm_(mla_heads), C_(kda_heads * kKdaDim), cap_(cap), kstride_(static_cast<unsigned>(cap / kKpool)) {
        constexpr std::size_t R = F::kMaxRows;
        qkv3_ = dalloc<float>(3 * R * C_);
        qkv_ = dalloc<float>(R * 3 * C_);
        fa_ = dalloc<float>(R * kKdaDim);
        ga_ = dalloc<float>(R * kKdaDim);
        bb_ = dalloc<float>(R * Hk_);
        f_ = dalloc<float>(R * C_);
        g_ = dalloc<float>(R * C_);
        ky_ = dalloc<float>(R * C_);
        o_kda_ = dalloc<float>(R * C_);  // kda_verify o scratch [n][H][128]
        qa_ = dalloc<float>(R * kQLora);
        qr_ = dalloc<float>(R * kQLora);
        q_ = dalloc<float>(R * Hm_ * kMlaDim);
        kva_ = dalloc<float>(R * kKvLora);
        lat_ = dalloc<float>(R * kKvLora);
        qI_ = dalloc<float>(R * kIdxHeads * kIdxDim);
        key_ = dalloc<float>(R * kIdxDim);
        gt_ = dalloc<float>(R * kIdxDim);
        wraw_ = dalloc<float>(R * kIdxHeads);
        keys_ = dalloc<std::uint32_t>(R * std::max(kstride_, 1U));
        hist_ = dalloc<std::uint32_t>(R * 256);
        tok_ = dalloc<std::uint32_t>(R * F::kTokStride);
        ntok_ = dalloc<std::uint32_t>(R);
        mw_ = F::MlaWork{dalloc<float>(R * Hm_ * kKvLora), dalloc<std::uint32_t>(R * Hm_), dalloc<float>(R * Hm_ * F::kTokStride),
                         dalloc<float>(R * Hm_ * F::kTokStride), dalloc<float>(R * Hm_ * kKvLora)};
        o_ = dalloc<float>(R * Hm_ * kMlaDim);
        slots_ = dalloc<D::DsaSlot>(R);
        tpos_ = dalloc<unsigned>(R);
        tb_ws_ = dalloc<unsigned char>(TB::workspace_bytes());
        hip_ok(hipMemset(tb_ws_, 0, TB::workspace_bytes()), "tierb workspace");
    }
    ~FastAttn() override {
        for (void* p : {static_cast<void*>(qkv3_), static_cast<void*>(qkv_), static_cast<void*>(fa_), static_cast<void*>(ga_),
                        static_cast<void*>(bb_), static_cast<void*>(f_), static_cast<void*>(g_), static_cast<void*>(ky_), static_cast<void*>(o_kda_),
                        static_cast<void*>(qa_), static_cast<void*>(qr_), static_cast<void*>(q_), static_cast<void*>(kva_),
                        static_cast<void*>(lat_), static_cast<void*>(qI_), static_cast<void*>(key_), static_cast<void*>(gt_),
                        static_cast<void*>(wraw_), static_cast<void*>(keys_), static_cast<void*>(hist_), static_cast<void*>(tok_),
                        static_cast<void*>(ntok_), static_cast<void*>(mw_.qt), static_cast<void*>(mw_.mkey),
                        static_cast<void*>(mw_.sc), static_cast<void*>(mw_.p), static_cast<void*>(mw_.ob), static_cast<void*>(o_),
                        static_cast<void*>(slots_), static_cast<void*>(tpos_), static_cast<void*>(tb_ws_)})
            (void)hipFree(p);
    }

    void kda(const KdaLayerWeights& w, KdaState& st, const CallCtx& c, const float* x, float* y) override {
        if (w.heads != Hk_) throw std::invalid_argument("fast attention: KDA heads");
        const std::size_t SZ = static_cast<std::size_t>(Hk_) * kKdaDim * kKdaDim, CZ = static_cast<std::size_t>(3) * C_ * 3;
        const D::KdaDev kd{Hk_, w.conv, w.A_log, w.dt_bias, w.o_norm};
        if (c.spec) {
            kda_verify(w, st, c, x, y, kd, SZ, CZ);
            return;
        }
        for (std::size_t r0 = 0; r0 < c.rows.size(); r0 += F::kMaxRows) {
            const auto n = static_cast<unsigned>(std::min<std::size_t>(F::kMaxRows, c.rows.size() - r0));
            const float* xr = x + r0 * kHidden;
            qkv(w, xr, qkv_, n, c);
            small_a(w, xr, n, c.stream);
            const F::DenseJob j2[2] = {{w.f_b, fa_, f_, C_, kKdaDim, kKdaDim, C_}, {w.g_b, ga_, g_, C_, kKdaDim, kKdaDim, C_}};
            F::dense_f16(j2, 2, n, c.stream);
            const auto rows = c.rows.subspan(r0, n);
            bool distinct_consecutive = true, one_slot_seq = n > 1;
            for (unsigned r = 1; r < n; ++r) {
                distinct_consecutive = distinct_consecutive && rows[r].slot == rows[0].slot + r;
                one_slot_seq = one_slot_seq && rows[r].slot == rows[0].slot && rows[r].pos == rows[0].pos + r;
            }
            if (distinct_consecutive) {
                F::kda_step(kd, qkv_, f_, g_, bb_, st.S + rows[0].slot * SZ, st.conv + rows[0].slot * CZ, ky_, c.flag, n, c.stream);
            } else if (one_slot_seq) {
                F::kda_steps_seq(kd, qkv_, f_, g_, bb_, st.S + rows[0].slot * SZ, st.conv + rows[0].slot * CZ, ky_, nullptr, nullptr,
                                 c.flag, n, c.stream);
            } else {
                for (unsigned r = 0; r < n; ++r)
                    F::kda_step(kd, qkv_ + static_cast<std::size_t>(r) * 3 * C_, f_ + static_cast<std::size_t>(r) * C_,
                                g_ + static_cast<std::size_t>(r) * C_, bb_ + static_cast<std::size_t>(r) * Hk_, st.S + rows[r].slot * SZ,
                                st.conv + rows[r].slot * CZ, ky_ + static_cast<std::size_t>(r) * C_, c.flag, 1, c.stream);
            }
            kda_o(w, y + r0 * kHidden, n, c);
        }
    }

    bool supports_multi_slot_verify() const override { return true; }

    void dsa(const DsaLayerWeights& w, DsaState& st, const CallCtx& c, const float* x, float* y) override {
        if (w.heads != Hm_ || st.cap != cap_) throw std::invalid_argument("fast attention: DSA heads / capacity");
        if (w.kv_a.raw.output != kKvLora || w.q_a.raw.output != kQLora) throw std::invalid_argument("fast attention: q_a / kv_a width");
        const D::IndexerDev iw{w.wk, w.gate, w.weights_proj, w.k_norm_w, w.k_norm_b, w.ape};
        for (std::size_t r0 = 0; r0 < c.rows.size(); r0 += F::kMaxRows) {
            const auto n = static_cast<unsigned>(std::min<std::size_t>(F::kMaxRows, c.rows.size() - r0));
            const float* xr = x + r0 * kHidden;
            unsigned p_bound = 1;
            std::uint64_t t_vis = 0;  // max visible tokens over the rows (scores schedule hint)
            for (unsigned r = 0; r < n; ++r) {
                const std::uint64_t t = c.rows[r0 + r].pos;
                if (t >= st.cap) throw std::out_of_range("DSA position beyond the slot capacity");
                p_bound = std::max(p_bound, static_cast<unsigned>((t + 1) / kKpool));
                t_vis = std::max<std::uint64_t>(t_vis, t + 1);
            }
            if (raw(w.q_a)) {  // [q_a + kv_a], both norms, [q_b + wq_b]: the same ops on the same inputs, fused launches
                TB::dsa_a(mat(w.q_a), mat(w.kv_a), xr, kHidden, n, qa_, kQLora, kva_, kKvLora, tb_ws_, c.stream);
                D::rmsnorm(qa_, w.q_a_norm, D::WType::BF16, kQLora, kEps, qr_, n, c.stream);
                D::rmsnorm(kva_, w.kv_a_norm, D::WType::BF16, kKvLora, kEps, lat_, n, c.stream);
                TB::dsa_b(mat(w.q_b), mat(w.wq_b), qr_, kQLora, n, q_, Hm_ * kMlaDim, qI_, kIdxHeads * kIdxDim, tb_ws_, c.stream);
            } else {
                c.lin->run(w.q_a, xr, qa_, n, c.stream);
                D::rmsnorm(qa_, w.q_a_norm, D::WType::BF16, kQLora, kEps, qr_, n, c.stream);
                c.lin->run(w.q_b, qr_, q_, n, c.stream);
                c.lin->run(w.kv_a, xr, kva_, n, c.stream);
                D::rmsnorm(kva_, w.kv_a_norm, D::WType::BF16, kKvLora, kEps, lat_, n, c.stream);
                c.lin->run(w.wq_b, qr_, qI_, n, c.stream);
            }
            const F::DenseJob j3[3] = {{w.wk, xr, key_, kIdxDim, kHidden, kHidden, kIdxDim},
                                       {w.gate, xr, gt_, kIdxDim, kHidden, kHidden, kIdxDim},
                                       {w.weights_proj, xr, wraw_, kIdxHeads, kHidden, kHidden, kIdxHeads}};
            F::dense_f16(j3, 3, n, c.stream);
            dsa_rows_kernel<<<1, F::kMaxRows, 0, c.stream>>>(c.slot_d + r0, c.pos_d + r0, st.latent, st.ikey, st.igate, st.pooled,
                                                              static_cast<unsigned>(st.cap), slots_, tpos_, n);
            hip_ok(hipGetLastError(), "dsa rows");
            const F::DsaRows d{slots_, tpos_, n};
            if (c.spec) F::dsa_store(d, lat_, key_, gt_, iw, hist_, c.spec->row_flag + r0, c.stream, 1);  // C3/C4 per row
            else F::dsa_store(d, lat_, key_, gt_, iw, hist_, c.flag, c.stream);
            // At most kIdxPools visible pools in every row: indexer_select takes them all and never reads the keys or the
            // histogram, so their computation is skipped (no cache writes happen there).
            if (p_bound > kIdxPools) F::indexer_score(d, qI_, wraw_, keys_, kstride_, hist_, p_bound, c.stream, c.spec != nullptr);
            F::indexer_select(d, keys_, kstride_, hist_, tok_, ntok_, c.stream);
            F::mla_attention(d, tok_, ntok_, q_, w.kv_b, Hm_, mw_, o_, c.stream, static_cast<unsigned>(std::min<std::uint64_t>(t_vis, 1U << 30)));
            if (raw(w.o)) TB::dsa_o(mat(w.o), o_, Hm_ * kMlaDim, n, y + r0 * kHidden, kHidden, tb_ws_, c.stream);
            else c.lin->run(w.o, o_, y + r0 * kHidden, n, c.stream);
        }
    }

private:
    // Speculative verify (design notes section 1): R consecutive rows of ONE slot through dev::kda_verify (deferred
    // replay of the slot's pending rows from set cur, new rows' raw qkv / prep outputs into set 1 - cur). With
    // SpecRows::seg (Engine::verify_multi) the windows of several slots: the projections run batched over all rows (row
    // invariant), then dev::kda_verify once per window on its rows and its slot's state and replay sets.
    void kda_verify(const KdaLayerWeights& w, KdaState& st, const CallCtx& c, const float* x, float* y, const D::KdaDev& kd,
                    std::size_t SZ, std::size_t CZ) {
        const auto n = static_cast<unsigned>(c.rows.size());
        const SpecRows& sp = *c.spec;
        if (n == 0 || n > F::kMaxRows || st.rep_qkv == nullptr) throw std::logic_error("spec verify: rows / Options::spec");
        const std::size_t scr = static_cast<std::size_t>(Hk_) * 4 * kKdaDim;
        if (sp.seg != nullptr && sp.nseg > 1) {
            qkv(w, x, qkv_, n, c);
            small_a(w, x, n, c.stream);
            const F::DenseJob j2[2] = {{w.f_b, fa_, f_, C_, kKdaDim, kKdaDim, C_}, {w.g_b, ga_, g_, C_, kKdaDim, kKdaDim, C_}};
            F::dense_f16(j2, 2, n, c.stream);
            for (unsigned g = 0; g < sp.nseg; ++g) {
                const SpecSeg& sg = sp.seg[g];
                if (sg.n == 0 || sg.row0 + sg.n > n) throw std::logic_error("spec verify: window rows");
                const std::size_t base = static_cast<std::size_t>(sg.slot) * 2;
                const std::size_t rep = (base + sg.cur) * F::kMaxRows, nxt = (base + 1 - sg.cur) * F::kMaxRows;
                float* qkv_new = st.rep_qkv + nxt * 3 * C_;  // the window's raw pre-conv rows: its slot's next replay input
                hip_ok(hipMemcpyAsync(qkv_new, qkv_ + static_cast<std::size_t>(sg.row0) * 3 * C_,
                                      static_cast<std::size_t>(sg.n) * 3 * C_ * sizeof(float), hipMemcpyDeviceToDevice, c.stream),
                       "spec verify window qkv");
                D::kda_verify(kd, st.rep_qkv + rep * 3 * C_, st.rep_scr + rep * scr, st.rep_beta + rep * Hk_, sg.n_rep, qkv_new,
                              f_ + static_cast<std::size_t>(sg.row0) * C_, g_ + static_cast<std::size_t>(sg.row0) * C_,
                              bb_ + static_cast<std::size_t>(sg.row0) * Hk_, sg.n, st.S + sg.slot * SZ, st.conv + sg.slot * CZ,
                              st.rep_scr + nxt * scr, st.rep_beta + nxt * Hk_, o_kda_ + static_cast<std::size_t>(sg.row0) * C_,
                              ky_ + static_cast<std::size_t>(sg.row0) * C_, sp.replay_flag, sp.row_flag + sg.row0, c.stream);
            }
            kda_o(w, y, n, c);
            return;
        }
        const std::size_t base = static_cast<std::size_t>(sp.slot) * 2;
        const std::size_t rep = (base + sp.cur) * F::kMaxRows, nxt = (base + 1 - sp.cur) * F::kMaxRows;
        float* qkv_new = st.rep_qkv + nxt * 3 * C_;  // [n][3C] raw pre-conv: the next call's replay input
        qkv(w, x, qkv_new, n, c);
        small_a(w, x, n, c.stream);
        const F::DenseJob j2[2] = {{w.f_b, fa_, f_, C_, kKdaDim, kKdaDim, C_}, {w.g_b, ga_, g_, C_, kKdaDim, kKdaDim, C_}};
        F::dense_f16(j2, 2, n, c.stream);
        D::kda_verify(kd, st.rep_qkv + rep * 3 * C_, st.rep_scr + rep * scr, st.rep_beta + rep * Hk_, sp.n_rep, qkv_new, f_, g_, bb_, n,
                      st.S + sp.slot * SZ, st.conv + sp.slot * CZ, st.rep_scr + nxt * scr, st.rep_beta + nxt * Hk_, o_kda_, ky_,
                      sp.replay_flag, sp.row_flag, c.stream);
        kda_o(w, y, n, c);
    }

    // KDA f_a, g_a (128 x 4096) and b_proj (rank rows x 4096): R1 chains (attention dense_f16) or, in the R1b identity
    // default for generated forwards (glm_r1b.hpp), the blocked chains; default prompts use R1. f_b / g_b (K 128) stay R1 in both identities.
    void small_a(const KdaLayerWeights& w, const float* x, unsigned n, hipStream_t s) {
        if (strix::glm::r1b::active()) {
            const int selected = strix::glm::r1b::dense_variant();
            const bool packed = (selected == 7 || selected == 6 || (selected == 5 && n == 1)) && w.f_a_packed != nullptr &&
                                w.g_a_packed != nullptr && w.b_packed != nullptr;
            const std::uint16_t* const W[3] = {packed ? w.f_a_packed : w.f_a, packed ? w.g_a_packed : w.g_a,
                                             packed ? w.b_packed : w.b};
            const unsigned N[3] = {kKdaDim, kKdaDim, Hk_};
            float* const Y[3] = {fa_, ga_, bb_};
            strix::glm::r1b::dense_f16_k4096_x3(x, kHidden, W, N, Y, N, n, s,
                                               selected == 5 || selected == 6 || selected == 7 ? (packed ? selected : 2) : selected);
            return;
        }
        const F::DenseJob j3[3] = {{w.f_a, x, fa_, kKdaDim, kHidden, kHidden, kKdaDim},
                                   {w.g_a, x, ga_, kKdaDim, kHidden, kHidden, kKdaDim},
                                   {w.b, x, bb_, Hk_, kHidden, kHidden, Hk_}};
        F::dense_f16(j3, 3, n, s);
    }

    // KDA q | k | v into [n][3C] (y): one fused launch for raw linears; else three library calls + packs.
    void qkv(const KdaLayerWeights& w, const float* x, float* y, unsigned n, const CallCtx& c) {
        if (raw(w.qkv[0])) {
            const TB::Mat m[3] = {mat(w.qkv[0]), mat(w.qkv[1]), mat(w.qkv[2])};
            TB::kda_qkv(m, x, kHidden, n, y, 3 * C_, tb_ws_, c.stream);
            return;
        }
        for (unsigned p = 0; p < 3; ++p) {
            c.lin->run(w.qkv[p], x, qkv3_ + static_cast<std::size_t>(p) * n * C_, n, c.stream);
            hip_ok(hipMemcpy2DAsync(y + static_cast<std::size_t>(p) * C_, 3 * C_ * sizeof(float), qkv3_ + static_cast<std::size_t>(p) * n * C_,
                                    C_ * sizeof(float), C_ * sizeof(float), n, hipMemcpyDeviceToDevice, c.stream),
                   "qkv pack");
        }
    }
    void kda_o(const KdaLayerWeights& w, float* y, unsigned n, const CallCtx& c) {
        if (raw(w.o)) TB::kda_o(mat(w.o), ky_, C_, n, y, kHidden, tb_ws_, c.stream);
        else c.lin->run(w.o, ky_, y, n, c.stream);
    }

    unsigned Hk_, Hm_, C_;
    std::uint64_t cap_;
    unsigned kstride_;
    float *qkv3_, *qkv_, *fa_, *ga_, *bb_, *f_, *g_, *ky_, *o_kda_;
    float *qa_, *qr_, *q_, *kva_, *lat_, *qI_, *key_, *gt_, *wraw_, *o_;
    std::uint32_t *keys_, *hist_, *tok_, *ntok_;
    F::MlaWork mw_{};
    D::DsaSlot* slots_;
    unsigned* tpos_;
    unsigned char* tb_ws_;  // attention Tier B workspace (stream-ordered calls only)
};

}  // namespace

std::shared_ptr<AttnPath> fast_attention(unsigned kda_heads, unsigned mla_heads, std::uint64_t capacity) {
    return std::make_shared<FastAttn>(kda_heads, mla_heads, capacity);
}

}  // namespace strix::glm::engine
