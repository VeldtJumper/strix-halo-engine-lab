// GLM-5.3-Flash chunked prefill seam (see glm_pf_path.hpp). Build with -ffp-contract=off -fno-fast-math
// -fno-gpu-flush-denormals-to-zero (the Tier A kernels it calls are bitwise to src/glm/host).
#include "glm/prefill/glm_pf_path.hpp"

#include "glm/kernels/glm_dsa.hpp"
#include "glm/kernels/glm_tier_a.hpp"
#include "glm/prefill/glm_pf_moe.hpp"
#include "glm/prefill/glm_pf_exact.hpp"
#include "glm/prefill/glm_pf_exact_moe.hpp"
#include "glm/prefill/glm_prefill.hpp"

#include <strix_trellis.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::glm::pf {

namespace {

namespace D = strix::glm::dev;
using namespace strix::glm::engine;

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("glm pf path ") + what + ": " + hipGetErrorString(e));
}
template <class T> T* dalloc(std::size_t n) {
    T* p = nullptr;
    ok(hipMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "alloc");
    return p;
}
// Owns device allocations: freed by its destructor, which also runs when a constructor that owns it throws.
struct DeviceOwner {
    std::vector<void*> ptrs;
    std::uint64_t bytes = 0;
    DeviceOwner() = default;
    DeviceOwner(const DeviceOwner&) = delete;
    DeviceOwner& operator=(const DeviceOwner&) = delete;
    ~DeviceOwner() {
        for (void* p : ptrs) (void)hipFree(p);
    }
    template <class T> T* alloc(std::size_t n) {
        if (n == 0) throw std::invalid_argument("glm pf path: zero-size buffer");
        ptrs.reserve(ptrs.size() + 1);
        T* p = dalloc<T>(n);
        ptrs.push_back(p);
        bytes += n * sizeof(T);
        return p;
    }
};
// Path geometry limits (validated before any budgeting or allocation).
constexpr unsigned kMaxChunkRows = 65536;
void check_geometry(unsigned kda_heads, unsigned mla_heads, std::uint64_t cap, unsigned max_rows) {
    if (kda_heads == 0 || kKdaHeads % kda_heads || mla_heads * kKdaHeads != kda_heads * kMlaHeads)
        throw std::invalid_argument("glm pf path: heads must be the full or the TP2 split");
    if (cap < kKpool || cap % kKpool || cap > (std::uint64_t{1} << 31)) throw std::invalid_argument("glm pf path: capacity");
    if (max_rows == 0 || max_rows > kMaxChunkRows) throw std::invalid_argument("glm pf path: max_rows must be 1..65536");
}
std::uint64_t align256(std::uint64_t v) { return (v + 255) & ~std::uint64_t{255}; }

// Frozen split table of glm-pf-wmma-v1: the library's own split at M = 1024 for every GLM linear shape, measured on the
// real tensors (design notes, PF-B): s = 1 everywhere. A change here is a new identity.
constexpr unsigned kPinnedSplits = 1;
bool exact_moe_enabled() {
    static const bool enabled = [] {
        const char* s = std::getenv("STRIX_GLM_PF_EXACT_MOE");
        // default ON (measurement: real TP2 pair, gemv-chunked with it == reference prefill bitwise, 3000 tokens)
        if (!s || std::string(s) == "1") return true;
        if (std::string(s) == "0") return false;
        throw std::invalid_argument("STRIX_GLM_PF_EXACT_MOE must be 0 or 1");
    }();
    return enabled;
}
constexpr unsigned kExpertBits = 4;  // identity-4.05bpw.json: routed experts K4

// [R][3C] row-major (q | k | v per row) from three [R][C] blocks.
__global__ void pack_qkv_kernel(const float* q, const float* k, const float* v, float* out, unsigned C, unsigned R) {
    const unsigned r = blockIdx.y, i = blockIdx.x * 256 + threadIdx.x;
    if (r >= R || i >= C) return;
    const std::size_t s = static_cast<std::size_t>(r) * C + i, d = static_cast<std::size_t>(r) * 3 * C + i;
    out[d] = q[s];
    out[d + C] = k[s];
    out[d + 2 * C] = v[s];
}

__global__ void gather_rows_kernel(const float* x, const std::uint32_t* rows, float* xg) {
    const unsigned r = blockIdx.y, i = blockIdx.x * 256 + threadIdx.x;
    xg[static_cast<std::size_t>(r) * kHidden + i] = x[static_cast<std::size_t>(rows[r]) * kHidden + i];
}
__global__ void scatter_rows_kernel(const float* y, const std::uint32_t* dst, float* eo) {  // eo [R][8][4096], dst = row * 8 + slot
    const unsigned r = blockIdx.y, i = blockIdx.x * 256 + threadIdx.x;
    eo[static_cast<std::size_t>(dst[r]) * kHidden + i] = y[static_cast<std::size_t>(r) * kHidden + i];
}

class GlmPfPath final : public PrefillPath {
public:
    GlmPfPath(unsigned kda_heads, unsigned mla_heads, std::uint64_t cap, unsigned max_rows, TierB tb)
        : Hk_(kda_heads), Hm_(mla_heads), I_(kExpertI * kda_heads / kKdaHeads), cap_(cap), Rmax_(max_rows), tb_(tb), xmoe_plan_(max_rows) {
        check_geometry(kda_heads, mla_heads, cap, max_rows);
        const std::size_t R = Rmax_, C = static_cast<std::size_t>(Hk_) * kKdaDim;
        tq_ = own_.alloc<float>(R * C);
        tk_ = own_.alloc<float>(R * C);
        tv_ = own_.alloc<float>(R * C);
        qkv_ = own_.alloc<float>(R * 3 * C);
        f_ = own_.alloc<float>(R * C);
        g_ = own_.alloc<float>(R * C);
        b_ = own_.alloc<float>(R * Hk_);
        ta_ = own_.alloc<float>(R * kKdaDim);
        tb128_ = own_.alloc<float>(R * kKdaDim);
        scr_ = own_.alloc<float>(R * Hk_ * 4 * kKdaDim);
        beta_ = own_.alloc<float>(R * Hk_);
        o_ = own_.alloc<float>(R * std::max<std::size_t>(C, static_cast<std::size_t>(Hm_) * kMlaDim));
        y1_ = own_.alloc<float>(R * C);
        qa_ = own_.alloc<float>(R * kQLora);
        qr_ = own_.alloc<float>(R * kQLora);
        q_ = own_.alloc<float>(R * Hm_ * kMlaDim);
        kva_ = own_.alloc<float>(R * kKvLora);
        lat_ = own_.alloc<float>(R * kKvLora);
        qI_ = own_.alloc<float>(R * kIdxHeads * kIdxDim);
        key_ = own_.alloc<float>(R * kIdxDim);
        gate_ = own_.alloc<float>(R * kIdxDim);
        wh_ = own_.alloc<float>(R * kIdxHeads);
        keys_ = own_.alloc<std::uint32_t>(R * (cap_ / kKpool));
        tok_ = own_.alloc<std::uint32_t>(R * kMaxTokens);
        ntok_ = own_.alloc<std::uint32_t>(R);
        qt_ = own_.alloc<float>(R * Hm_ * kKvLora);
        e_ = own_.alloc<float>(R * Hm_ * kEStride);
        ob_ = own_.alloc<float>(R * Hm_ * kKvLora);
        sh_ = own_.alloc<float>(R * kHidden);
        moe_wsb_ = prefill_moe_workspace_bound(Rmax_);
        moe_ws_ = own_.alloc<unsigned char>(moe_wsb_);
        // Reserved up front and included in device_bytes / pf_mem_budget even when
        // disabled: no allocation if the exact linear switch is enabled.
        exact_wsb_ = exact_workspace_bytes(Hk_, Hm_, Rmax_);
        exact_ws_ = own_.alloc<unsigned char>(exact_wsb_);
        const char* ex = std::getenv("STRIX_GLM_PF_EXACT_LIN");
        if (ex && std::string(ex) != "0" && std::string(ex) != "1")
            throw std::invalid_argument("STRIX_GLM_PF_EXACT_LIN must be 0 or 1");
        exact_on_ = !(ex && std::string(ex) == "0");  // default ON (exact, gated 6322/0 + synthetic 6-layer engine gate; 0 = strix_gemv blocks)
        (void)exact::tile_rows();
        route_ids_.resize(R * kTopE);
        route_weights_.resize(R * kTopE);
        if (exact_moe_enabled() && I_ == kMoeInter && tb_ == TierB::gemv) {  // the wmma identity never uses it
            xmoe_wsb_ = exact_moe::workspace_bytes(Rmax_);
            xmoe_ws_ = own_.alloc<unsigned char>(xmoe_wsb_);
        }
        // gemv-mode per-row MoE scratch (DefaultMoe's)
        eo_ = own_.alloc<float>(static_cast<std::size_t>(kTopE) * kHidden);
        ge_ = own_.alloc<float>(kExpertI);
        ue_ = own_.alloc<float>(kExpertI);
        ae_ = own_.alloc<float>(kExpertI);
        sh1_ = own_.alloc<float>(kHidden);
        // FFN intermediates (3 views of Rmax x the largest rank I) and, when the grouped core does not apply (world 1,
        // I != 1024), the per-expert chain buffers: allocated now, so no prefill call allocates.
        big_n_ = R * ffn_i();
        big_ = own_.alloc<float>(3 * big_n_);
        if (I_ != kMoeInter) {
            pe_cap_ = R * kTopE;
            pe_x_ = own_.alloc<float>(pe_cap_ * kHidden);
            pe_y_ = own_.alloc<float>(pe_cap_ * kHidden);
            pe_eo_ = own_.alloc<float>(pe_cap_ * kHidden);
            pe_rows_ = own_.alloc<std::uint32_t>(pe_cap_);
            pe_dst_ = own_.alloc<std::uint32_t>(pe_cap_);
        }
        if (own_.bytes != device_bytes(Hk_, Hm_, cap_, Rmax_, tb_) || own_.ptrs.size() > kPrefillAllocationBound)
            throw std::logic_error("glm pf path: allocation tally differs from prefill_budget (update both)");
    }
    // Exact bytes the constructor allocates (checked there against its own tally). The admission budget passes the
    // default (gemv, the larger: it adds the exact MoE workspace the wmma identity never allocates).
    static std::uint64_t device_bytes(unsigned Hk, unsigned Hm, std::uint64_t cap, unsigned Rmax, TierB tb = TierB::gemv) {
        const std::uint64_t R = Rmax, C = std::uint64_t{Hk} * kKdaDim, I = std::uint64_t{kExpertI} * Hk / kKdaHeads;
        const std::uint64_t f = 3 * C + 3 * C + 2 * C + Hk + 2 * kKdaDim + std::uint64_t{Hk} * 4 * kKdaDim + Hk +
                                std::max<std::uint64_t>(C, std::uint64_t{Hm} * kMlaDim) + C + 2 * kQLora + std::uint64_t{Hm} * kMlaDim +
                                2 * kKvLora + std::uint64_t{kIdxHeads} * kIdxDim + 2 * kIdxDim + kIdxHeads + std::uint64_t{Hm} * kKvLora +
                                std::uint64_t{Hm} * kEStride + std::uint64_t{Hm} * kKvLora + kHidden;
        std::uint64_t b = R * f * 4 + R * (cap / kKpool) * 4 + R * kMaxTokens * 4 + R * 4;
        b += prefill_moe_workspace_bound(Rmax);
        b += exact_workspace_bytes(Hk, Hm, Rmax);
        // pf_mem_budget mirror: optional TP2 exact MoE adds workspace_bytes(Rmax),
        // including 9R pairs of GU FP32, down-input FP16, output FP32,
        // two matrix-specific H-wide FP16 input transforms and plan/table.
        // 792849408 bytes at R=2048; 1585660416 bytes at R=4096 (64-bit ABI).
        if (exact_moe_enabled() && I == kMoeInter && tb == TierB::gemv) b += exact_moe::workspace_bytes(Rmax);
        b += (std::uint64_t{kTopE} * kHidden + 3 * kExpertI + kHidden) * 4;
        b += 3 * R * std::max<std::uint64_t>(std::uint64_t{kDenseI} * Hk / kKdaHeads, I) * 4;
        if (I != kMoeInter) b += R * kTopE * (3 * kHidden * 4 + 2 * 4);
        return b;
    }
    // Include MTP eh_proj (2H -> H), also in the pf_mem_budget exact-linear scratch mirror.
    static std::size_t exact_workspace_bytes(unsigned Hk, unsigned Hm, unsigned Rmax) {
        const unsigned K = std::max({2 * kHidden, Hk * kKdaDim, Hm * kMlaDim, kDenseI * Hk / kKdaHeads, kExpertI * Hk / kKdaHeads});
        return exact::workspace_bytes(Rmax, K);
    }
    // Largest strix_gemm_pinned (s = 1) workspace of the path's linears at Rmax rows (strix-trellis backend.hip
    // linear_workspace_bytes: align256(rows * in * 2) + rows * out * 4 * s).
    static std::uint64_t workspace_bytes(unsigned Hk, unsigned Hm, unsigned Rmax) {
        const std::uint64_t R = Rmax, C = std::uint64_t{Hk} * kKdaDim, I = std::uint64_t{kExpertI} * Hk / kKdaHeads,
                            Id = std::uint64_t{kDenseI} * Hk / kKdaHeads;
        const std::uint64_t shapes[][2] = {{kHidden, C},       {C, kHidden},  {kHidden, kQLora}, {kHidden, kKvLora},
                                           {kQLora, std::uint64_t{Hm} * kMlaDim}, {kQLora, std::uint64_t{kIdxHeads} * kIdxDim},
                                           {std::uint64_t{Hm} * kMlaDim, kHidden}, {kHidden, Id}, {Id, kHidden}, {kHidden, I}, {I, kHidden}};
        std::uint64_t m = 0;
        for (const auto& s : shapes) m = std::max(m, align256(R * s[0] * 2) + R * s[1] * 4 * kPinnedSplits);
        return m;
    }
    std::uint64_t linear_workspace_bytes() const override { return tb_ == TierB::wmma ? workspace_bytes(Hk_, Hm_, Rmax_) : 0; }
    // gemv: opt-in grouped exact TP2, otherwise the engine decode seam per row; wmma: the shared expert
    // runs pinned over the chunk, so it needs a prepared handle (raw-only experts are read by the grouped core directly).
    bool moe_by_row() const override { return tb_ == TierB::gemv && xmoe_ws_ == nullptr; }
    unsigned moe_row_block() const override { return gemv_block(); }
    bool supports_pipeline() const override { return tb_ == TierB::gemv; }
    bool needs_prepared_shared() const override { return tb_ == TierB::wmma; }
    std::uint64_t admission_margin() const override { return margin_; }
    std::uint64_t margin_ = std::uint64_t{8} << 30;  // set by make_prefill_path (STRIX_GLM_PF_MARGIN_GIB)

    ~GlmPfPath() override {  // own_ frees every device buffer
        for (auto& t : pinned_tables_) (void)hipHostFree(t.second);
    }

    void kda(const KdaLayerWeights& w, KdaState& st, const CallCtx& c, const float* x, float* y) override {
        const unsigned R = rows(c), H = w.heads, C = H * kKdaDim;
        if (H != Hk_) throw std::invalid_argument("glm pf path: KDA heads");
        const std::uint32_t slot = c.rows[0].slot;
        linear(c, w.qkv[0], x, tq_, R, kHidden, C);
        linear(c, w.qkv[1], x, tk_, R, kHidden, C);
        linear(c, w.qkv[2], x, tv_, R, kHidden, C);
        pack_qkv_kernel<<<dim3((C + 255) / 256, R), 256, 0, c.stream>>>(tq_, tk_, tv_, qkv_, C, R);
        ok(hipGetLastError(), "pack_qkv");
        dense_rows(x, w.f_a, D::WType::F16, kHidden, 0, kKdaDim, ta_, R, c.stream);
        dense_rows(ta_, w.f_b, D::WType::F16, kKdaDim, 0, C, f_, R, c.stream);
        dense_rows(x, w.g_a, D::WType::F16, kHidden, 0, kKdaDim, tb128_, R, c.stream);
        dense_rows(tb128_, w.g_b, D::WType::F16, kKdaDim, 0, C, g_, R, c.stream);
        dense_rows(x, w.b, D::WType::F16, kHidden, 0, H, b_, R, c.stream);
        const D::KdaDev kd{H, w.conv, w.A_log, w.dt_bias, w.o_norm};
        float* S = st.S + static_cast<std::size_t>(slot) * H * kKdaDim * kKdaDim;
        float* cv = st.conv + static_cast<std::size_t>(slot) * 3 * C * 3;
        kda_prefill(kd, qkv_, f_, g_, b_, S, cv, scr_, beta_, o_, y1_, c.flag, R, c.stream);
        linear(c, w.o, y1_, y, R, C, kHidden);
    }

    void dsa(const DsaLayerWeights& w, DsaState& st, const CallCtx& c, const float* x, float* y) override {
        const unsigned R = rows(c), H = w.heads;
        if (H != Hm_) throw std::invalid_argument("glm pf path: MLA heads");
        if (st.cap != cap_) throw std::invalid_argument("glm pf path: DSA capacity differs from the path's");
        const std::uint32_t slot = c.rows[0].slot;
        const auto t0 = static_cast<unsigned>(c.rows[0].pos);
        if (static_cast<std::uint64_t>(t0) + R > st.cap) throw std::out_of_range("glm pf path: DSA positions beyond the slot capacity");
        linear(c, w.q_a, x, qa_, R, kHidden, kQLora);
        D::rmsnorm(qa_, w.q_a_norm, D::WType::BF16, kQLora, kEps, qr_, R, c.stream);
        linear(c, w.q_b, qr_, q_, R, kQLora, H * kMlaDim);
        linear(c, w.kv_a, x, kva_, R, kHidden, kKvLora);
        D::rmsnorm(kva_, w.kv_a_norm, D::WType::BF16, kKvLora, kEps, lat_, R, c.stream);
        linear(c, w.wq_b, qr_, qI_, R, kQLora, kIdxHeads * kIdxDim);
        const D::IndexerDev iw{w.wk, w.gate, w.weights_proj, w.k_norm_w, w.k_norm_b, w.ape};
        const D::DsaSlot sl{st.latent + static_cast<std::size_t>(slot) * st.cap * kKvLora, st.ikey + static_cast<std::size_t>(slot) * st.cap * kIdxDim,
                            st.igate + static_cast<std::size_t>(slot) * st.cap * kIdxDim,
                            st.pooled + static_cast<std::size_t>(slot) * (st.cap / kKpool) * kIdxDim, static_cast<unsigned>(st.cap)};
        dense_rows(x, w.wk, D::WType::F16, kHidden, 0, kIdxDim, key_, R, c.stream);
        dense_rows(x, w.gate, D::WType::F16, kHidden, 0, kIdxDim, gate_, R, c.stream);
        dsa_append_rows(sl, t0, lat_, key_, gate_, iw, c.flag, R, c.stream);
        dense_rows(x, w.weights_proj, D::WType::F16, kHidden, 0, kIdxHeads, wh_, R, c.stream);
        indexer_select_rows(sl, t0, qI_, wh_, keys_, static_cast<unsigned>(cap_ / kKpool), tok_, ntok_, R, c.stream);
        mla_attention_rows(sl, tok_, ntok_, q_, w.kv_b, H, qt_, e_, ob_, o_, R, c.stream);
        linear(c, w.o, o_, y, R, H * kMlaDim, kHidden);
    }

    void moe(const MoeLayerWeights& w, const CallCtx& c, const float* x, const std::uint32_t* ids, const float* wts, float* y) override {
        const unsigned R = rows(c), I = I_;
        if (tb_ == TierB::gemv && xmoe_ws_ != nullptr && c.flag != nullptr && exact_moe::device_plan(R)) {
            // Device plan: no route download, host grouping or plan upload. The layer's expert table is kept in
            // pinned host memory and copied in stream order into the workspace table slot.
            if (w.experts.size() != moe::kExperts) throw std::invalid_argument("glm pf exact moe: expected 288 experts");
            for (unsigned e = 0; e < moe::kExperts; ++e) exact_table_[e] = exact_slices(w.experts[e], false);
            const moe::ExpertSlices shared = exact_slices(w.shared, true);
            constexpr std::size_t kTableBytes = (moe::kExperts + 1) * sizeof(moe::ExpertSlices);
            auto& pinned = pinned_tables_[&w];
            const bool fresh = pinned == nullptr;
            if (fresh) ok(hipHostMalloc(reinterpret_cast<void**>(&pinned), kTableBytes, hipHostMallocDefault), "pinned expert table");
            if (fresh || std::memcmp(pinned, exact_table_.data(), moe::kExperts * sizeof(moe::ExpertSlices)) != 0 ||
                std::memcmp(pinned + moe::kExperts, &shared, sizeof(shared)) != 0) {
                // A previous copy of this table may still be queued: rewrite it only after the stream drains.
                if (!fresh) ok(hipStreamSynchronize(c.stream), "expert table change");
                std::memcpy(pinned, exact_table_.data(), moe::kExperts * sizeof(moe::ExpertSlices));
                std::memcpy(pinned + moe::kExperts, &shared, sizeof(shared));
            }
            auto* table = exact_moe::workspace_table(xmoe_ws_);
            ok(hipMemcpyAsync(table, pinned, kTableBytes, hipMemcpyHostToDevice, c.stream), "expert table");
            exact_moe::DeviceView v;
            v.rows = R;
            v.table = table;
            v.shared = shared;
            v.x = x;
            v.ids = ids;
            v.weights = wts;
            v.out = y;
            exact_moe::enqueue_device(v, xmoe_plan_, xmoe_ws_, xmoe_wsb_, c.stream, c.flag);
            return;
        }
        auto& hid = route_ids_;
        auto& hw = route_weights_;
        hid.resize(static_cast<std::size_t>(R) * kTopE);
        hw.resize(hid.size());
        ok(hipMemcpyAsync(hid.data(), ids, hid.size() * 4, hipMemcpyDeviceToHost, c.stream), "ids");
        ok(hipMemcpyAsync(hw.data(), wts, hw.size() * 4, hipMemcpyDeviceToHost, c.stream), "wts");
        ok(hipStreamSynchronize(c.stream), "routes");
        if (tb_ == TierB::gemv && xmoe_ws_ != nullptr) {
            if (w.experts.size() != moe::kExperts) throw std::invalid_argument("glm pf exact moe: expected 288 experts");
            for (unsigned e = 0; e < moe::kExperts; ++e) exact_table_[e] = exact_slices(w.experts[e], false);
            exact_moe::View v;
            v.rows = R;
            v.experts = exact_table_.data();
            v.shared = exact_slices(w.shared, true);
            v.x = x;
            v.ids = hid.data();
            v.host_weights = hw.data();
            v.weights = wts;
            v.out = y;
            exact_moe::enqueue(v, xmoe_plan_, xmoe_ws_, xmoe_wsb_, c.stream);
            return;
        }
        if (tb_ == TierB::gemv) {  // DefaultMoe, row by row: the decode identity (the engine uses moe_by_row instead)
            if (w.shared.gate.prepared == nullptr) throw std::logic_error("glm pf path: gemv moe() needs prepared experts");
            for (unsigned r = 0; r < R; ++r) {
                const float* xr = x + static_cast<std::size_t>(r) * kHidden;
                for (unsigned k = 0; k < kTopE; ++k)
                    mlp_row(c, w.experts.at(hid[static_cast<std::size_t>(r) * kTopE + k]), xr, I, eo_ + static_cast<std::size_t>(k) * kHidden);
                mlp_row(c, w.shared, xr, I, sh1_);
                D::moe_combine(wts + static_cast<std::size_t>(r) * kTopE, eo_, sh1_, y + static_cast<std::size_t>(r) * kHidden, 1, c.stream);
            }
            return;
        }
        // shared expert over the chunk (pinned), then every routed expert in the grouped core + R16 combine
        if (w.shared.gate.prepared == nullptr || w.shared.up.prepared == nullptr || w.shared.down.prepared == nullptr)
            throw std::logic_error("glm pf path: shared expert not prepared (Engine prefill install)");
        linear(c, w.shared.gate, x, ta_big(R, I), R, kHidden, I);
        linear(c, w.shared.up, x, tb_big(R, I), R, kHidden, I);
        D::swiglu(ta_big(R, I), tb_big(R, I), tc_big(R, I), static_cast<std::size_t>(R) * I, c.stream);
        linear(c, w.shared.down, tc_big(R, I), sh_, R, I, kHidden);
        if (I != kMoeInter) {  // the grouped core is the TP2 geometry (I 1024); otherwise its own reference chain
            if (w.experts.empty() || w.experts[0].gate.prepared == nullptr) throw std::logic_error("glm pf path: per-expert chain needs prepared experts");
            per_expert_pinned(w, c, x, hid, wts, R, I, y);
            return;
        }
        const auto& ex = experts_of(w, I);
        PrefillMoeView v;
        v.rows = R;
        v.x = x;
        v.ids = hid.data();
        v.weights = hw.data();
        v.experts = ex.data();
        v.bits_gu = kExpertBits;
        v.bits_dn = kExpertBits;
        v.shared = sh_;
        v.out = y;
        enqueue_prefill_moe_glm(v, moe_ws_, moe_wsb_, c.stream);
    }

    void dense(const Mlp& m, const CallCtx& c, const float* x, unsigned I, float* y) override {
        const unsigned R = rows(c);
        linear(c, m.gate, x, ta_big(R, I), R, kHidden, I);
        linear(c, m.up, x, tb_big(R, I), R, kHidden, I);
        D::swiglu(ta_big(R, I), tb_big(R, I), tc_big(R, I), static_cast<std::size_t>(R) * I, c.stream);
        linear(c, m.down, tc_big(R, I), y, R, I, kHidden);
    }

    bool exact_linears() const override { return tb_ == TierB::gemv; }
    void linear_exact(const Linear& l, const CallCtx& c, const float* x, float* y,
                      unsigned K, unsigned N) const override {
        if (!exact_linears()) throw std::logic_error("MTP prefill requires the exact gemv identity");
        linear(c, l, x, y, rows(c), K, N);
    }

private:
    unsigned rows(const CallCtx& c) const {
        const auto R = static_cast<unsigned>(c.rows.size());
        if (R == 0 || R > Rmax_) throw std::invalid_argument("glm pf path: rows per chunk");
        if (c.spec) throw std::logic_error("glm pf path: prefill chunks are plain rows");
        for (unsigned r = 1; r < R; ++r)
            if (c.rows[r].slot != c.rows[0].slot || c.rows[r].pos != c.rows[0].pos + r)
                throw std::invalid_argument("glm pf path: a chunk is consecutive positions of one slot");
        return R;
    }
    // Tier B by identity: pinned WMMA over the chunk, or strix_gemv R = 1 per row.
    void linear(const CallCtx& c, const Linear& l, const float* in, float* out, unsigned R, unsigned K, unsigned N) const {
        if (tb_ == TierB::wmma) {
            c.lin->run_pinned(l, in, out, R, kPinnedSplits, c.stream);
            return;
        }
        if (exact_on_ && exact::supported(l)) {
            exact::linear_rows(l, in, K, out, N, R, exact_ws_, exact_wsb_, c.stream);
            return;
        }
        // strix_gemv blocks of <= kGemvBlock rows: a row's bits do not depend on the call's R (P1 batch invariance
        // R = 1..8, the library split policy is row-count independent for every GLM shape; gated against R = 1 rows by
        // pf-tierb-tests and the rw gate). STRIX_GLM_PF_GEMV_BLOCK=1 restores one row per call.
        const unsigned B = gemv_block();
        for (unsigned r = 0; r < R; r += B)
            c.lin->run(l, in + static_cast<std::size_t>(r) * K, out + static_cast<std::size_t>(r) * N, std::min(B, R - r), c.stream);
    }
    // Routed experts as the per-expert strix_gemm_pinned chain (the bitwise reference of the grouped core, gated in
    // hip_glm_pf_moe_tests): rows of each expert in ascending token order, then R16 over the route slots.
    void per_expert_pinned(const MoeLayerWeights& w, const CallCtx& c, const float* x, const std::vector<std::uint32_t>& hid, const float* wts,
                           unsigned R, unsigned I, float* y) {
        const std::size_t R8 = static_cast<std::size_t>(R) * kTopE;
        if (R8 > pe_cap_) throw std::logic_error("glm pf path: per-expert buffers smaller than the chunk");
        std::vector<std::vector<std::uint32_t>> rows(w.experts.size()), dst(w.experts.size());
        for (unsigned r = 0; r < R; ++r)
            for (unsigned k = 0; k < kTopE; ++k) {
                const std::uint32_t e = hid[static_cast<std::size_t>(r) * kTopE + k];
                if (e >= w.experts.size()) throw std::out_of_range("glm pf path: route id");
                rows[e].push_back(r);
                dst[e].push_back(r * kTopE + k);
            }
        std::vector<std::uint32_t> fr, fd;
        std::vector<std::size_t> off(w.experts.size() + 1, 0);
        for (std::size_t e = 0; e < w.experts.size(); ++e) {
            off[e + 1] = off[e] + rows[e].size();
            fr.insert(fr.end(), rows[e].begin(), rows[e].end());
            fd.insert(fd.end(), dst[e].begin(), dst[e].end());
        }
        ok(hipMemcpyAsync(pe_rows_, fr.data(), fr.size() * 4, hipMemcpyHostToDevice, c.stream), "rows");
        ok(hipMemcpyAsync(pe_dst_, fd.data(), fd.size() * 4, hipMemcpyHostToDevice, c.stream), "dst");
        for (std::size_t e = 0; e < w.experts.size(); ++e) {
            const auto M = static_cast<unsigned>(off[e + 1] - off[e]);
            if (M == 0) continue;
            float* xe = pe_x_ + off[e] * kHidden;
            float* ye = pe_y_ + off[e] * kHidden;
            gather_rows_kernel<<<dim3(kHidden / 256, M), 256, 0, c.stream>>>(x, pe_rows_ + off[e], xe);
            ok(hipGetLastError(), "gather");
            linear(c, w.experts[e].gate, xe, ta_big(R8_cap(), I), M, kHidden, I);
            linear(c, w.experts[e].up, xe, tb_big(R8_cap(), I), M, kHidden, I);
            D::swiglu(ta_big(R8_cap(), I), tb_big(R8_cap(), I), tc_big(R8_cap(), I), static_cast<std::size_t>(M) * I, c.stream);
            linear(c, w.experts[e].down, tc_big(R8_cap(), I), ye, M, I, kHidden);
            scatter_rows_kernel<<<dim3(kHidden / 256, M), 256, 0, c.stream>>>(ye, pe_dst_ + off[e], pe_eo_);
            ok(hipGetLastError(), "scatter");
        }
        D::moe_combine(wts, pe_eo_, sh_, y, R, c.stream);
        ok(hipStreamSynchronize(c.stream), "per-expert tables");  // host vectors fr / fd die here
    }
    unsigned R8_cap() const { return Rmax_; }
    std::size_t ffn_i() const { return std::max<std::size_t>(kDenseI * Hk_ / kKdaHeads, I_); }  // big() buffers hold Rmax_ rows; expert groups never exceed R <= Rmax_

    void mlp_row(const CallCtx& c, const Mlp& m, const float* x, unsigned I, float* out) {
        c.lin->run(m.gate, x, ge_, 1, c.stream);
        c.lin->run(m.up, x, ue_, 1, c.stream);
        D::swiglu(ge_, ue_, ae_, I, c.stream);
        c.lin->run(m.down, ae_, out, 1, c.stream);
    }
    // FFN intermediates [R][I] (I <= dense rank I = 6144 on TP2, 12288 single): one buffer sized at construction, 3 views.
    float* ta_big(unsigned R, unsigned I) { return big(R, I, 0); }
    float* tb_big(unsigned R, unsigned I) { return big(R, I, 1); }
    float* tc_big(unsigned R, unsigned I) { return big(R, I, 2); }
    float* big(unsigned R, unsigned I, unsigned k) {
        if (static_cast<std::size_t>(Rmax_) * I > big_n_) throw std::logic_error("glm pf path: FFN buffer smaller than Rmax x I");
        (void)R;
        return big_ + k * big_n_;
    }
    const std::vector<MoeExpert>& experts_of(const MoeLayerWeights& w, unsigned I) {
        auto it = experts_.find(&w);
        if (it != experts_.end()) return it->second;
        std::vector<MoeExpert> v(w.experts.size());
        for (std::size_t e = 0; e < w.experts.size(); ++e) {
            const Mlp& m = w.experts[e];
            // prepared handles hold the parts verbatim (hip_glm_pf_moe_tests checks it); raw-only experts (fast MoE seam)
            // are the same file-layout parts in the engine arena
            auto part = [&](const Linear& l, unsigned N, unsigned K) {
                if (l.prepared != nullptr) return moe_exl3_from_weights(static_cast<strix_weights>(l.prepared), N, K, kExpertBits);
                if (l.raw.trellis == nullptr || l.raw.suh == nullptr || l.raw.svh == nullptr || l.raw.output != N || l.raw.input != K ||
                    l.raw.bits != kExpertBits)
                    throw std::invalid_argument("glm pf path: raw expert parts missing or not " + std::to_string(N) + " x " + std::to_string(K));
                return MoeExl3{l.raw.trellis, l.raw.suh, l.raw.svh};
            };
            v[e] = {part(m.gate, I, kHidden), part(m.up, I, kHidden), part(m.down, kHidden, I)};
        }
        return experts_.emplace(&w, std::move(v)).first->second;
    }

    static moe::Exl3 exact_part(const Linear& l, unsigned N, unsigned K, bool shared) {
        const unsigned bits = l.raw.bits;
        if (l.raw.trellis != nullptr) {
            if (!l.raw.suh || !l.raw.svh || l.raw.output != N || l.raw.input != K ||
                (bits != 4 && !(shared && bits == 6)))
                throw std::invalid_argument("glm pf exact moe: invalid raw expert");
            return {l.raw.trellis, l.raw.suh, l.raw.svh, K, N, bits};
        }
        // Prepared loads still retain geometry/bit metadata in Linear::raw.
        if (!l.prepared || (bits != 4 && !(shared && bits == 6)))
            throw std::invalid_argument("glm pf exact moe: expert parts/bit metadata missing");
        const auto p = moe_exl3_from_weights(static_cast<strix_weights>(l.prepared), N, K, bits);
        return {p.trellis, p.suh, p.svh, K, N, bits};
    }
    static moe::ExpertSlices exact_slices(const Mlp& m, bool shared) {
        return {exact_part(m.gate, kMoeInter, kHidden, shared), exact_part(m.up, kMoeInter, kHidden, shared),
                exact_part(m.down, kHidden, kMoeInter, shared)};
    }

    DeviceOwner own_;       // first member: owns every buffer below
    unsigned Hk_, Hm_, I_;  // rank heads; rank expert I (2048 / world)
    std::uint64_t cap_;
    unsigned Rmax_;
    TierB tb_;
    exact_moe::Plan xmoe_plan_;
    void* xmoe_ws_ = nullptr;
    std::size_t xmoe_wsb_ = 0;
    std::array<moe::ExpertSlices, moe::kExperts> exact_table_{};
    std::map<const MoeLayerWeights*, moe::ExpertSlices*> pinned_tables_;  // host-pinned [289] per MoE layer
    std::vector<std::uint32_t> route_ids_;
    std::vector<float> route_weights_;
    float *tq_, *tk_, *tv_, *qkv_, *f_, *g_, *b_, *ta_, *tb128_, *scr_, *beta_, *o_, *y1_;
    float *qa_, *qr_, *q_, *kva_, *lat_, *qI_, *key_, *gate_, *wh_, *qt_, *e_, *ob_, *sh_;
    std::uint32_t *keys_, *tok_, *ntok_;
    void* exact_ws_ = nullptr;
    std::size_t exact_wsb_ = 0;
    bool exact_on_ = false;
    void* moe_ws_;
    std::size_t moe_wsb_;
    float *eo_, *ge_, *ue_, *ae_, *sh1_;
    float* big_ = nullptr;
    std::size_t big_n_ = 0;
    std::map<const MoeLayerWeights*, std::vector<MoeExpert>> experts_;
    float *pe_x_ = nullptr, *pe_y_ = nullptr, *pe_eo_ = nullptr;
    std::uint32_t *pe_rows_ = nullptr, *pe_dst_ = nullptr;
    std::size_t pe_cap_ = 0;
};

}  // namespace

TierB tierb_from_env() {
    const char* e = std::getenv("STRIX_GLM_PF_TIERB");
    if (e == nullptr || std::string(e) == "gemv") return TierB::gemv;
    if (std::string(e) == "wmma") return TierB::wmma;
    throw std::invalid_argument("STRIX_GLM_PF_TIERB must be wmma or gemv");
}

unsigned gemv_block() {
    static const unsigned b = [] {
        const char* e = std::getenv("STRIX_GLM_PF_GEMV_BLOCK");
        if (e == nullptr || *e == '\0') return 8U;
        const std::string v(e);
        if (v.size() != 1 || v[0] < '1' || v[0] > '8') throw std::invalid_argument("STRIX_GLM_PF_GEMV_BLOCK must be 1..8");
        return static_cast<unsigned>(v[0] - '0');
    }();
    return b;
}

PrefillBudget prefill_budget(unsigned kda_heads, unsigned mla_heads, std::uint64_t capacity, unsigned max_rows, bool mtp) {
    check_geometry(kda_heads, mla_heads, capacity, max_rows);
    PrefillBudget b;
    b.path = GlmPfPath::device_bytes(kda_heads, mla_heads, capacity, max_rows);
    b.workspace = GlmPfPath::workspace_bytes(kda_heads, mla_heads, max_rows);
    const std::uint64_t RC = std::max(8U, max_rows);
    b.engine_rows = RC * ((3ULL * kHc * kHidden + 2ULL * kHcMix + (5ULL + (mtp ? 9ULL : 0ULL)) * kHidden + kExperts + kTopE) * 4 + (kTopE + 2) * 4 + 8);
    return b;
}

namespace {
std::uint64_t mem_available_bytes() {  // /proc/meminfo MemAvailable; refuses when it cannot be read
    std::ifstream f("/proc/meminfo");
    std::string key;
    std::uint64_t kb = 0;
    std::string unit;
    while (f >> key >> kb >> unit)
        if (key == "MemAvailable:" && unit == "kB") return kb * 1024;
    throw std::runtime_error("glm pf path: cannot read MemAvailable from /proc/meminfo (refusing to allocate unchecked)");
}
}  // namespace
std::uint64_t admission_margin_bytes() {  // STRIX_GLM_PF_MARGIN_GIB: finite, 0 .. 1024 GiB; default 8
    const char* m = std::getenv("STRIX_GLM_PF_MARGIN_GIB");
    if (m == nullptr) return std::uint64_t{8} << 30;
    char* end = nullptr;
    const double g = std::strtod(m, &end);
    if (end == m || *end != '\0' || !std::isfinite(g) || g < 0 || g > 1024)
        throw std::invalid_argument("STRIX_GLM_PF_MARGIN_GIB must be a number of GiB in [0, 1024]");
    return static_cast<std::uint64_t>(g * static_cast<double>(std::uint64_t{1} << 30));
}

std::shared_ptr<engine::PrefillPath> make_prefill_path(unsigned kda_heads, unsigned mla_heads, std::uint64_t capacity, unsigned max_rows,
                                                       TierB tierb) {
    const PrefillBudget b = prefill_budget(kda_heads, mla_heads, capacity, max_rows);  // validates the geometry
    const std::uint64_t need = b.path + (tierb == TierB::wmma ? b.workspace : 0);
    const std::uint64_t margin = admission_margin_bytes();
    std::size_t dev_free = 0, dev_total = 0;
    ok(hipMemGetInfo(&dev_free, &dev_total), "mem info");
    const std::uint64_t host_avail = mem_available_bytes();
    if (dev_free < need + margin || host_avail < need + margin) {
        const auto gib = [](std::uint64_t v) { return static_cast<double>(v) / static_cast<double>(std::uint64_t{1} << 30); };
        std::ostringstream os;
        os << "glm pf path: refusing to allocate " << gib(need) << " GiB (path " << gib(b.path) << " + workspace " << gib(b.workspace)
           << ") + margin " << gib(margin) << " GiB: device free " << gib(dev_free) << " GiB, host MemAvailable " << gib(host_avail)
           << " GiB (lower max_rows / capacity)";
        throw std::runtime_error(os.str());
    }
    auto path = std::make_shared<GlmPfPath>(kda_heads, mla_heads, capacity, max_rows, tierb);
    path->margin_ = margin;
    return path;
}

}  // namespace strix::glm::pf
