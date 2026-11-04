#include "minimax/engine/route_identity.hpp"
// MiniMax M3.1 Flash engine, TP2 (see include/strix/minimax/engine.hpp and qualification notes).
//
// Layer flow (per call of n rows), identical to the G4 harness (tests/hip_minimax_g4.hip) so the teacher-forced gate is
// bitwise: q/k/v slices (EXL3) and the index projections (FP32 dense_t) on hn -> K1 prep (norms, RoPE, cache write)
// -> K2 scores / K3 select / K4 attention in groups of <= 8 rows -> o_proj input slice -> exchange X1 + fused
// post-exchange (h += p0 + p1, post-attention norm, router top-4) -> dense MLP or shared + routed experts (ascending
// expert id per token, batch invariant) -> exchange X2 + fused post-exchange (next input norm / final norm).
// Linears: strix_gemm for prefill / gate calls, strix_gemv for decode steps (<= 8 rows).
#include "strix/minimax/engine.hpp"
#include "strix/dist/communicator.hpp"

#include "mm_kernels.hpp"
#include "strix/hip_minimax_moe.hpp"
#include "strix/hip_minimax_msa.hpp"
#include "minimax/attn/index_proj.hpp"
#include "minimax/attn/msa_fused.hpp"
#include "minimax/dspark/hip_minimax_dspark.hpp"
#include "draft_lin.hpp"
#include "prefill_timeline.hpp"
#include "minimax/xfer/minimax_xfer.hpp"
#include "minimax/pfx/pf_exchange.hpp"
#include "minimax/moe/mmx_moe.hpp"
#include "minimax/moe/mmx_prefill.hpp"
#include "minimax/rtxpf/fp32_identity.hpp"
#include "minimax/engine/mm_defaults.hpp"

#include <strix_trellis.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <set>

extern "C" void strix_trellis_lab_set_tile_override(int cfg) __attribute__((weak));
extern "C" int strix_trellis_lab_last_tile() __attribute__((weak));

namespace strix::minimax::engine {

namespace {

using namespace detail;
namespace msa = strix::minimax::msa;
namespace mh = strix::minimax::msa::hip;
namespace mmoe = strix::minimax::moe::opt;
namespace mfused = strix::minimax::msa::fused;
namespace mx = strix::minimax::xfer;
namespace rtxpf = strix::minimax::rtxpf;
using Clock = std::chrono::steady_clock;

constexpr unsigned kWorld = 2;
constexpr unsigned kQLocal = kQ / kWorld, kKVLocal = kKV / kWorld, kIdxLocal = kIdxQ / kWorld;
constexpr unsigned kDenseLocal = kDense / kWorld, kExpertLocal = kExpertI / kWorld;
constexpr std::size_t kGatherRows = 256;  // G4's head gather chunk
constexpr std::uint32_t kMaxStepRows = 8;
constexpr std::size_t kControlBytes = std::size_t{64} << 10;
constexpr std::uint32_t kF32JobRing = 4096;  // FP32 prefill identity: job-table entries between two stream syncs

double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

void lib_ok(strix_status s, const std::string& what) {
    if (s != STRIX_SUCCESS) throw std::runtime_error("trellis " + what + ": " + strix_status_string(s));
}

template <class T> T* dalloc(std::size_t n) {
    T* p = nullptr;
    hip_ok(hipMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "malloc");
    return p;
}
template <class T> T* dupload(const std::vector<T>& h) {
    T* p = dalloc<T>(h.size());
    hip_ok(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "h2d");
    return p;
}
std::vector<float> transpose_f(const std::vector<float>& w, std::size_t n, std::size_t d) {  // [n][d] -> [d][n]
    std::vector<float> t(n * d);
    for (std::size_t o = 0; o < n; ++o)
        for (std::size_t i = 0; i < d; ++i) t[i * n + o] = w[o * d + i];
    return t;
}

bool env_flag(const char* name, bool def) {
    const char* v = std::getenv(name);
    return v == nullptr ? def : std::atoi(v) != 0;
}

int kv_tile_mode(const char* value) {
    if (!value) return -2;  // default auto (measurement: exact, 8K B/A 0.960/0.949/1.017); needs the Trellis tile hook
    if (std::strcmp(value, "off") == 0) return -1;
    if (std::strcmp(value, "auto") == 0) return -2;
    int cfg = -1;
    const char* end = value + std::strlen(value);
    const auto result = std::from_chars(value, end, cfg);
    if (result.ec != std::errc{} || result.ptr != end || cfg < 0)
        throw std::invalid_argument("STRIX_MM_PF_KV_TILE must be off, auto or a nonnegative config index");
    return cfg;
}

bool kv_tile_available() {
    if (strix_trellis_lab_set_tile_override) return true;
    static const bool reported = [] {
        std::printf("STRIX_MM_PF_KV_TILE: trellis lab tile override absent; switch is a no-op\n");
        return true;
    }();
    (void)reported;
    return false;
}

// strix-trellis EXL3 linears, resident. Slices are built in ONE pinned staging buffer from O_DIRECT reads (never from
// file pages), then prepared; the layout is the G4 harness's (H128-aligned output / input slices).
class Lib {
public:
    explicit Lib(PackReader& pack) : pack_(pack) {
        strix_context_desc d{};
        d.struct_size = sizeof(d);
        d.abi_version = STRIX_TRELLIS_ABI_VERSION;
        d.device_ordinal = 0;
        lib_ok(strix_context_create(&d, &ctx_), "context");
        st_.struct_size = sizeof(st_);
        st_.abi_version = STRIX_TRELLIS_ABI_VERSION;
        st_.hip_stream = nullptr;
    }
    ~Lib() {
        (void)hipDeviceSynchronize();
        for (auto w : all_) (void)strix_weights_destroy(w);
        for (void* p : arenas_) (void)hipFree(p);
        if (ws_) (void)hipFree(ws_);
        release_staging();
        (void)strix_context_destroy(ctx_);
    }
    struct Lin {
        strix_weights w = nullptr;
        std::uint64_t rows = 0, cols = 0;  // N outputs, I inputs
        std::uint64_t bits = 0;
    };
    // One H128-aligned rank slice of an EXL3 linear assembled in the pinned staging buffer (G4 layout): an output
    // slice [o0, o0 + n) keeps those outputs' tiles of every input row and the matching svh; an input slice
    // [i0, i0 + n) keeps the contiguous trellis rows and the matching suh (a partial sum to reduce across ranks).
    struct Staged {
        const std::uint8_t *tr = nullptr, *suh = nullptr, *svh = nullptr;
        std::uint64_t tr_bytes = 0, suh_bytes = 0, svh_bytes = 0, rows = 0, cols = 0, bits = 0;
    };
    Staged stage_out(const std::string& prefix, std::uint64_t o0, std::uint64_t n) { return stage_out(pack_, prefix, o0, n); }
    // The same from another pack (a draft-only requant of a target tensor, Engine::attach_draft_head).
    Staged stage_out(PackReader& pk, const std::string& prefix, std::uint64_t o0, std::uint64_t n) {
        const TensorInfo &tr = pk.info(prefix + ".trellis"), &suh = pk.info(prefix + ".suh"), &svh = pk.info(prefix + ".svh");
        if (o0 % 128 != 0 || n % 128 != 0 || o0 + n > svh.shape[0]) throw std::invalid_argument("bad output slice " + prefix);
        const std::uint64_t in_tiles = tr.shape[0], out_tiles = tr.shape[1], w = tr.shape[2] * 2;  // bytes per tile
        const std::uint64_t tr_bytes = in_tiles * (n / 16) * w;
        if (whole_.size() < tr.bytes) whole_.resize(tr.bytes);
        pk.read(tr, 0, tr.bytes, whole_.data());
        std::uint8_t* t = stage(tr_bytes + suh.bytes + n * 2);
        for (std::uint64_t i = 0; i < in_tiles; ++i)
            std::memcpy(t + i * (n / 16) * w, whole_.data() + (i * out_tiles + o0 / 16) * w, (n / 16) * w);
        pk.read(suh, 0, suh.bytes, t + tr_bytes);
        pk.read(svh, o0 * 2, n * 2, t + tr_bytes + suh.bytes);
        return {t, t + tr_bytes, t + tr_bytes + suh.bytes, tr_bytes, suh.bytes, n * 2, n, suh.shape[0], tr.shape[2] / 16};
    }
    Staged stage_in(const std::string& prefix, std::uint64_t i0, std::uint64_t n) {
        const TensorInfo &tr = pack_.info(prefix + ".trellis"), &suh = pack_.info(prefix + ".suh"), &svh = pack_.info(prefix + ".svh");
        if (i0 % 128 != 0 || n % 128 != 0 || i0 + n > suh.shape[0]) throw std::invalid_argument("bad input slice " + prefix);
        const std::uint64_t row_bytes = tr.shape[1] * tr.shape[2] * 2, tr_bytes = (n / 16) * row_bytes;
        std::uint8_t* t = stage(tr_bytes + n * 2 + svh.bytes);
        pack_.read(tr, (i0 / 16) * row_bytes, tr_bytes, t);
        pack_.read(suh, i0 * 2, n * 2, t + tr_bytes);
        pack_.read(svh, 0, svh.bytes, t + tr_bytes + n * 2);
        return {t, t + tr_bytes, t + tr_bytes + n * 2, tr_bytes, n * 2, svh.bytes, svh.shape[0], n, tr.shape[2] / 16};
    }
    Lin out_slice(const std::string& prefix, std::uint64_t o0, std::uint64_t n) { return prepared(prefix, stage_out(prefix, o0, n)); }
    Lin out_slice(PackReader& pk, const std::string& prefix, std::uint64_t o0, std::uint64_t n) {
        return prepared(prefix, stage_out(pk, prefix, o0, n));
    }
    Lin in_slice(const std::string& prefix, std::uint64_t i0, std::uint64_t n) { return prepared(prefix, stage_in(prefix, i0, n)); }
    // The same slices as raw device buffers in the strix-trellis device layout (= file layout), for the MiniMax MoE
    // kernels (src/minimax/moe): trellis words, suh / svh FP16.
    mmoe::Exl3 raw_out(const std::string& prefix, std::uint64_t o0, std::uint64_t n) { return raw(stage_out(prefix, o0, n)); }
    // From another pack into its own exact-size device allocation (not the 1 GiB arena chunks: a draft-only copy).
    mmoe::Exl3 raw_out(PackReader& pk, const std::string& prefix, std::uint64_t o0, std::uint64_t n) {
        return raw(stage_out(pk, prefix, o0, n), true);
    }
    mmoe::Exl3 raw_in(const std::string& prefix, std::uint64_t i0, std::uint64_t n) { return raw(stage_in(prefix, i0, n)); }
    // Raw bytes of a plain tensor to device memory through the staging buffer.
    void upload_raw(const std::string& name, void* dst) {
        const TensorInfo& t = pack_.info(name);
        constexpr std::uint64_t chunk = std::uint64_t{256} << 20;
        for (std::uint64_t off = 0; off < t.bytes; off += chunk) {
            const std::uint64_t n = std::min(chunk, t.bytes - off);
            std::uint8_t* s = stage(n);
            pack_.read(t, off, n, s);
            hip_ok(hipMemcpy(static_cast<std::uint8_t*>(dst) + off, s, n, hipMemcpyHostToDevice), "upload");
        }
    }
    void release_staging() {
        if (stage_) (void)hipHostFree(stage_);
        stage_ = nullptr;
        std::vector<std::uint8_t>().swap(whole_);
    }
    // out[t][N] = in[t][I] @ W; gemv (the decode entry point) when `gemv`, else gemm.
    void mm(const Lin& l, const float* in, float* out, std::uint64_t t, bool gemv) {
        if (t == 0) return;
        strix_matmul_desc d{};
        strix_matmul_desc_init(&d);
        d.weights = l.w;
        set(d.input, const_cast<float*>(in), t, l.cols);
        set(d.output, out, t, l.rows);
        std::uint64_t bytes = 0;
        lib_ok(strix_matmul_workspace_bytes(ctx_, &d, &bytes), "workspace");
        if (bytes > ws_bytes_) {
            if (reserved_ && grows_++ == 0)  // a shape reserve() did not see: the stall it removes is back for this one
                std::fprintf(stderr, "strix-minimax: matmul workspace grows past the start-up reserve (%llu -> %llu bytes, %llu x %llu, %llu rows)\n",
                             static_cast<unsigned long long>(ws_bytes_), static_cast<unsigned long long>(bytes),
                             static_cast<unsigned long long>(l.rows), static_cast<unsigned long long>(l.cols),
                             static_cast<unsigned long long>(t));
            hip_ok(hipDeviceSynchronize(), "sync");
            if (ws_) hip_ok(hipFree(ws_), "free");
            hip_ok(hipMalloc(&ws_, bytes), "workspace");
            ws_bytes_ = bytes;
        }
        d.workspace = bytes ? ws_ : nullptr;
        d.workspace_bytes = bytes ? ws_bytes_ : 0;
        lib_ok(gemv ? strix_gemv(ctx_, &d, &st_) : strix_gemm(ctx_, &d, &st_), gemv ? "gemv" : "gemm");
    }

    // The largest workspace mm(l, in, out, t, .) needs for t = 1..t_max (host-only queries on the given buffers, which
    // must hold t_max rows of l.cols / l.rows floats).
    std::uint64_t workspace_bound(const Lin& l, float* in, float* out, std::uint64_t t_max) const {
        std::uint64_t most = 0;
        for (std::uint64_t t = 1; t <= t_max; ++t) {
            strix_matmul_desc d{};
            strix_matmul_desc_init(&d);
            d.weights = l.w;
            set(d.input, in, t, l.cols);
            set(d.output, out, t, l.rows);
            std::uint64_t bytes = 0;
            lib_ok(strix_matmul_workspace_bytes(ctx_, &d, &bytes), "workspace");
            most = std::max(most, bytes);
        }
        return most;
    }
    // Start-up allocation of the workspace (gap audit 14): mm() then never synchronises / frees / allocates for a shape
    // the reserve covered (the grow path stays for any other and prints once).
    void reserve_workspace(std::uint64_t bytes) {
        if (bytes > ws_bytes_) {
            hip_ok(hipDeviceSynchronize(), "sync");
            if (ws_) hip_ok(hipFree(ws_), "free");
            hip_ok(hipMalloc(&ws_, bytes), "workspace");
            ws_bytes_ = bytes;
        }
        reserved_ = true;
    }
    [[nodiscard]] std::uint64_t workspace_bytes() const { return ws_bytes_; }
    [[nodiscard]] std::uint64_t workspace_grows() const { return grows_; }

private:
    bool reserved_ = false;
    std::uint64_t grows_ = 0;
    static constexpr std::size_t kStage = std::size_t{768} << 20;  // > the largest slice (lm_head half, ~616 MB)
    std::uint8_t* stage(std::uint64_t bytes) {
        if (bytes > kStage) throw std::runtime_error("staging buffer too small");
        if (!stage_) hip_ok(hipHostMalloc(reinterpret_cast<void**>(&stage_), kStage, hipHostMallocDefault), "staging");
        return stage_;  // prepare_parts / hipMemcpy finish their upload before returning, so the buffer is free again
    }
    Lin prepared(const std::string& what, const Staged& g) {
        strix_exl3_parts_desc p{};
        p.struct_size = sizeof(p);
        p.abi_version = STRIX_TRELLIS_ABI_VERSION;
        p.transport_version = STRIX_TRELLIS_TRANSPORT_VERSION;
        p.bits_x2 = static_cast<std::uint32_t>(2 * g.bits);
        p.rows = g.rows;
        p.columns = g.cols;
        p.codebook = STRIX_CODEBOOK_MUL1;
        p.marker = STRIX_EXL3_MARKER_MUL1;
        p.trellis = {g.tr, g.tr_bytes};
        p.suh = {g.suh, g.suh_bytes};
        p.svh = {g.svh, g.svh_bytes};
        Lin l;
        lib_ok(strix_weights_prepare_parts(ctx_, &p, &st_, &l.w), "prepare " + what);
        l.rows = g.rows;
        l.cols = g.cols;
        l.bits = g.bits;
        all_.push_back(l.w);
        return l;
    }
    // Bump allocation in 1 GiB device chunks (tens of thousands of expert slices).
    std::uint8_t* arena(std::uint64_t bytes) {
        bytes = (bytes + 255) & ~std::uint64_t{255};
        if (arena_left_ < bytes) {
            const std::uint64_t chunk = std::max<std::uint64_t>(bytes, std::uint64_t{1} << 30);
            void* p = nullptr;
            hip_ok(hipMalloc(&p, chunk), "arena");
            arenas_.push_back(p);
            arena_next_ = static_cast<std::uint8_t*>(p);
            arena_left_ = chunk;
        }
        std::uint8_t* r = arena_next_;
        arena_next_ += bytes;
        arena_left_ -= bytes;
        return r;
    }
    mmoe::Exl3 raw(const Staged& g, bool own = false) {
        const std::uint64_t bytes = g.tr_bytes + 256 + g.suh_bytes + 256 + g.svh_bytes;
        std::uint8_t* d = nullptr;
        if (own) {
            void* p = nullptr;
            hip_ok(hipMalloc(&p, bytes), "raw");
            arenas_.push_back(p);
            d = static_cast<std::uint8_t*>(p);
        } else {
            d = arena(bytes);
        }
        std::uint8_t* d_suh = d + ((g.tr_bytes + 255) & ~std::uint64_t{255});
        std::uint8_t* d_svh = d_suh + ((g.suh_bytes + 255) & ~std::uint64_t{255});
        hip_ok(hipMemcpy(d, g.tr, g.tr_bytes, hipMemcpyHostToDevice), "raw trellis");
        hip_ok(hipMemcpy(d_suh, g.suh, g.suh_bytes, hipMemcpyHostToDevice), "raw suh");
        hip_ok(hipMemcpy(d_svh, g.svh, g.svh_bytes, hipMemcpyHostToDevice), "raw svh");
        mmoe::Exl3 e;
        e.trellis = reinterpret_cast<const std::uint32_t*>(d);
        e.suh = reinterpret_cast<const std::uint16_t*>(d_suh);
        e.svh = reinterpret_cast<const std::uint16_t*>(d_svh);
        e.input = static_cast<std::uint32_t>(g.cols);
        e.output = static_cast<std::uint32_t>(g.rows);
        e.bits = static_cast<std::uint32_t>(g.bits);
        return e;
    }
    static void set(strix_tensor_desc& t, float* p, std::uint64_t rows, std::uint64_t width) {
        t.data = p;
        t.buffer_bytes = rows * width * sizeof(float);
        t.dtype = STRIX_DTYPE_F32;
        t.rank = 2;
        t.shape[0] = rows;
        t.shape[1] = width;
        t.strides[0] = width;
        t.strides[1] = 1;
    }
    PackReader& pack_;
    strix_context ctx_ = nullptr;
    strix_stream st_{};
    void* ws_ = nullptr;
    std::uint64_t ws_bytes_ = 0;
    std::uint8_t* stage_ = nullptr;
    std::vector<std::uint8_t> whole_;
    std::vector<strix_weights> all_;
    std::vector<void*> arenas_;
    std::uint8_t* arena_next_ = nullptr;
    std::uint64_t arena_left_ = 0;
};
using Lin = Lib::Lin;

struct Mlp {
    Lin g, u, d;
};
struct Layer {
    Lin q, k, v, o;
    // STRIX_MM_LINEARS: raw copies of the decode linears for src/minimax/moe enqueue_linears (q/k/v/o, dense MLP)
    mmoe::Exl3 rq{}, rk{}, rv{}, ro{};
    mmoe::ExpertSlices rdense{};
    _Float16 *wiq16 = nullptr, *wik16 = nullptr;
    _Float16 *wiq16b = nullptr, *wik16b = nullptr;
    _Float16* wip = nullptr;  // packed index projection (pack_index_proj layout) // block-major [ob][i][32] copies for the decode kernel // the same index projections, F16 (STRIX_MM_INDEX_PROJ=tiled)
    float *wiq = nullptr, *wik = nullptr, *norm[4] = {}, *w_post = nullptr, *w_next = nullptr, *wr = nullptr, *wb = nullptr;
    bool moe = false;
    Mlp dense, shared;
    std::vector<Mlp> experts;
    // STRIX_MM_MOE=opt: raw rank slices for src/minimax/moe (host array for prefill, device array for decode)
    std::vector<mmoe::ExpertSlices> experts_raw;
    mmoe::ExpertSlices* experts_raw_d = nullptr;
    mmoe::ExpertSlices shared_raw{};
};

// decode STRIX_MM_XB_WARM: streams up to 4 byte ranges through the memory-side cache with plain cached loads (the
// following phase's weights, read during an exchange wait; mb2: L3 re-reads 580-860 GB/s vs 210-235 GB/s from DRAM).
// Cache state only: no output except a sink store that a value test makes practically unreachable.
struct WarmRanges {
    const uint4* p[4] = {nullptr, nullptr, nullptr, nullptr};
    std::size_t n16[4] = {0, 0, 0, 0};
    unsigned* sink = nullptr;
};
__global__ __launch_bounds__(256) void xb_warm_kernel(WarmRanges w) {
    __shared__ unsigned wave_sum[8];
    unsigned acc = 0;
    for (int r = 0; r < 4; ++r)
        for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < w.n16[r]; i += std::size_t{gridDim.x} * blockDim.x) {
            const uint4 v = w.p[r][i];
            acc += v.x ^ v.y ^ v.z ^ v.w;
        }
    // every lane's loads feed one value per CTA, stored by one thread at its own address (no race)
    for (int d = 16; d >= 1; d >>= 1) acc += static_cast<unsigned>(__shfl_xor(static_cast<int>(acc), d, 32));
    if ((threadIdx.x & 31U) == 0) wave_sum[threadIdx.x >> 5U] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned s = 0;
        for (unsigned q = 0; q < blockDim.x / 32U; ++q) s += wave_sum[q];
        w.sink[blockIdx.x] = s;
    }
}

}  // namespace

struct Engine::Impl {
    Options opt;
    dist::Communicator& comm;
    PackReader pack;
    Lib lib;
    std::vector<Layer> layers;
    float* w_in0 = nullptr;
    std::uint16_t* embed = nullptr;  // [vocab][6144] BF16, replicated
    Lin head;
    std::uint64_t vn = 0;
    // activations, max_rows rows
    float *h = nullptr, *hn = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *iq = nullptr, *ik = nullptr, *attn = nullptr,
          *big1 = nullptr, *big2 = nullptr, *xe = nullptr, *ye = nullptr, *part = nullptr, *remote = nullptr, *logits = nullptr,
          *scores = nullptr, *partial = nullptr, *wts4_d = nullptr;
    // routes of the current call: [ids n x 4][weights n x 4] (post-exchange output), and the per-expert packing
    // [rows (n x 4)][weights (n x 4)], each moved with ONE pinned copy per MoE layer
    std::uint32_t *route_d = nullptr, *route_h = nullptr, *pack_d = nullptr, *pack_h = nullptr;
    std::uint8_t *ctl_local = nullptr, *ctl_remote = nullptr;
    std::uint32_t *ids_d = nullptr, *idx = nullptr, *cnt = nullptr, *rows_d = nullptr, *tok_d = nullptr, *am_local = nullptr,
                  *am_remote = nullptr;
    std::uint16_t *kc = nullptr, *vc = nullptr, *ikc = nullptr;  // [slot][layer][capacity][...]
    msa::RopeRow* d_rope = nullptr;
    std::uint64_t* d_pos = nullptr;
    std::uint64_t calls = 0, controls = 0;
    bool profiling = false;
    bool pf_table_stats = false;
    std::uint64_t pf_table_calls = 0;
    double pf_table_host_ms = 0;
    Profile prof;
    std::unique_ptr<PrefillTimeline> timeline;
    PrefillTimeline* tl = nullptr;
    bool gemv = false;  // linears of the current call
    bool fused = false;  // STRIX_MM_STAGING=fused
    bool shared_first = false;  // STRIX_MM_SHARED_FIRST=1: shared expert enqueued before the route copy
    hipStream_t route_stream = nullptr;
    hipStream_t idx_side = nullptr;  // concurrent index projection (decode): side stream + per-layer events
    std::vector<hipEvent_t> idx_fork, idx_join;
    std::vector<hipEvent_t> pf_idx_fork, pf_idx_join;
    std::vector<hipEvent_t> pf_idx_fork_device, pf_idx_join_device;
    int pf_idx_conc = 0;  // 0 off, 1 original events, 2 device-only event fences
    int pf_kv_tile = -2;  // STRIX_MM_PF_KV_TILE: -1 off, -2 auto (default), otherwise config index
    std::set<std::pair<std::size_t, int>> pf_kv_receipts;
    std::size_t pf_idx_member = 0;
    hipEvent_t routes_ready = nullptr;
    float* xs = nullptr;  // shared expert output
    bool attn_fused = true;  // STRIX_MM_ATTN (runtime, default fused; =off: per-row K1-K4): decode MSA in fused launches
    std::uint16_t** cache_tab_h = nullptr;  // per step: [layer][3][row] cache pointers (k, v, index key)
    std::uint16_t** cache_tab_d = nullptr;
    std::uint32_t* ticket = nullptr;
    msa::RopeRow* step_rope = nullptr;  // per step: the rows' RoPE rows and positions (contiguous)
    std::uint64_t* step_pos = nullptr;
    // fused MSA grouping of the current step: rows in distinct slots (seqs = n, 1 row each) or one slot's verify
    // window (1 sequence, n consecutive rows); 0 = mixed (per-row K1-K4 path)
    std::uint32_t step_seqs = 0, step_rows_per_seq = 0;
    mx::Exchanger* xf = nullptr;  // exchange module (decode steps), set by the app
    bool xfer_on = false;         // STRIX_MM_XFER (runtime switch once an exchanger is set)
    std::uint32_t* out_ids_d = nullptr;  // argmax_exchange output (one id per row)
    bool use_xfer() const { return xf != nullptr && xfer_on && gemv; }
    bool index_conc = false;  // STRIX_MM_INDEX_CONC=1 (runtime): decode index projection on a side stream, concurrent with q/k/v
    bool attend_diet = false;  // STRIX_MM_ATTEND_DIET (default ON via mm_defaults): v3 at its 32 KB LDS on route::attend_n2_sparse grids
    bool attend4_decode = false;  // STRIX_MM_ATTEND4_DECODE (default ON via mm_defaults): v4, one head per wave, on the same grids
    std::uint64_t select_once_calls = 0;  // decode attention calls that took the selection-once route (diagnostic)
    bool select_once = false;  // STRIX_MM_SELECT_ONCE (default ON via mm_defaults): one selection launch per (row, KV head) on route::select_once calls
    // decode fused exchange boundary (decode, module path; qualification notes), same bits:
    bool xb_wait = false;     // STRIX_MM_XB_WAIT=1: post_exchange waits for the peer's partial in-kernel (recv_view) instead of a stream wait packet
    bool xb_inplace = false;  // STRIX_MM_XB_INPLACE=1: o_proj / FFN producers write the module's TX slot, a publish kernel replaces the copy
    float* xb_tx = nullptr;   // the TX slot send_buffer() handed to the producers of the next exchange (XB_INPLACE), else null
    std::uint64_t xb_count[6] = {0, 0, 0, 0, 0, 0};  // in-kernel wait / in-place producer / prepare fold / self-publish / warms / ilp posts
    bool xb_postilp = false;  // STRIX_MM_XB_POSTILP=1|2: post_exchange loads-first (1) or partials staged through LDS (2), same bits
    unsigned xb_postilp_mode = 2;
    bool xb_pub = false;       // STRIX_MM_XB_PUBLISH=1 (with XB_INPLACE): the in-place producer publishes itself (lean), no publish launch
    mmoe::PublishTarget xb_pt{};  // the target handed to the producers of the next exchange (flag null: none)
    bool xb_published = false;    // the producer enqueued for xb_tx carries xb_pt
    unsigned xb_warm_mb = 8;      // STRIX_MM_XB_WARM=MB (0 = off; default 8, exchange boundary set): per exchange, warm up to MB of the next phase's weights
    hipStream_t xb_warm_stream = nullptr;  // side stream: waits for each message's TX flag, then the warm kernel
    unsigned* xb_warm_sink = nullptr;
    unsigned xb_warm_o_mb = 8;    // STRIX_MM_XB_WARM_O=MB (0 = off; default 8): during each layer's MSA, warm up to MB of its o_proj on the warm stream
    std::vector<hipEvent_t> xb_attn_ev;  // per layer: q/k/v and index projections done (the warm's gate), re-recorded each step
    std::uint64_t xb_warm_o_count = 0;
    bool xb_prep = false;      // STRIX_MM_XB_PREP=1: the FFN post_exchange writes the next layer's q/k/v input transform (no prepare launch)
    bool xb_prepped = false;   // the next q/k/v call's inputs were transformed by the post_exchange just enqueued
    unsigned xb_layers = 0;    // layers of the current layers_forward call (the fold needs a next layer)
    bool index_attn = true;  // STRIX_MM_INDEX_PROJ (runtime; default attn = index_proj, decode and prefill)
    bool index_decode = true;  // STRIX_MM_INDEX_DECODE (runtime, default 1; =0: tiled dense_t form): decode rows on index_proj_decode_kernel
    bool index_tiled = true;  // STRIX_MM_INDEX_PROJ (runtime; default tiled, =off: dense_t): index q/k projections, token-tiled, F16 weights
    bool attn_prefill_fused = true;  // STRIX_MM_ATTN_PREFILL (runtime; default fused, =off: K1-K4): prefill MSA in fused 16-row windows
    bool pf_idx_tile = false;
    unsigned pf_msa_scratch_mb = 1152;  // STRIX_MM_PF_MSA_SCRATCH_MB: prefill MSA window scratch cap (former 512)
    bool pf_dense_o = true;  // STRIX_MM_PF_DENSE_O (default ON): prefill o_proj on the v4 core for full 4096-row pieces; =0 library
    bool pf_dense_qkv = true;  // STRIX_MM_PF_DENSE_QKV (default ON): prefill q/k/v in one v4-core call from 1024 rows (same bits as strix_gemm); =0 library
    unsigned pf_idx_rows = 32;
    unsigned pf_msa_rework = 6;  // STRIX_MM_PF_MSA_REWORK: prefill only, default 6 (= 5 with the tile corrections in the packed exp lanes, exact); =5 / =3 previous defaults, =0 original
    bool pf_msa_batch = true;  // STRIX_MM_PF_MSA_BATCH (default ON, exact measurement): independent scratch per prefill window
    unsigned pf_msa_windows = 0;
    float *pf_scores = nullptr, *pf_partial = nullptr;
    std::uint32_t *pf_idx = nullptr, *pf_cnt = nullptr, *pf_ticket = nullptr;
    void set_prefill_msa_rework(unsigned mode) {
        if (mode > 6) throw std::invalid_argument("prefill MSA rework must be 0, 1, 2, 3, 4, 5 or 6");
        pf_msa_rework = mode;
    }
    bool pf_attend4 = true;  // STRIX_MM_PF_ATTEND4 (default ON, exact measurement): prefill windows use v4, two heads per wave
    std::uint16_t** pf_tab_h = nullptr;  // prefill: [layer][3] cache pointers of the call's slot
    std::uint16_t** pf_tab_d = nullptr;
    std::uint32_t pf_tab_slot = ~0U;  // immutable cache allocation/layout; decode has its own table
    bool linears_loaded = true;  // STRIX_MM_LINEARS (load time; default on; =off: no raw copies, =load: loaded but off)
    bool linears_opt = true;     // decode linears on enqueue_linears (runtime switch once loaded)
    void* lin_ws = nullptr;
    bool moe_opt = true;  // STRIX_MM_MOE (load time; default opt, =off: G4 chain): MoE layers on the src/minimax/moe kernels
    mmoe::MoeWorkspace moe_ws{};
    // Only enqueue_prefill_moe uses this workspace, always on the default stream, including interleaved chunks.
    void* pf_ws = nullptr;
    std::size_t pf_ws_bytes = 0;
    //: prefill exchange module (pf_active only inside prefill()) and the chunk-interleaved schedule
    strix::minimax::pfx::Exchanger* pf = nullptr;
    bool pf_on = false, pf_active = false, pf_interleave = true;  // interleave ON by default (measurement: exact, 8K -11.5 %)
    bool pf_force_comm = false;                // STRIX_MM_PF_INTERLEAVE=comm: the communicator backend even with the module on
    bool pf_router_batch = true;               // STRIX_MM_PF_ROUTER_BATCH: prefill only, default ON (prefill measurement exact); =0 off
    unsigned pf_router_rows = 32;               // STRIX_MM_PF_ROUTER_ROWS: 32 (default: tiled scores, weights read once per 16 rows), 4 or 8; same bits
    float* router_scratch = nullptr;           // tiled router scores [2][128][rows] (STRIX_MM_PF_ROUTER_ROWS=32)
    bool pf_tx_uncached = false;              // STRIX_MM_PF_TX_UNCACHED: prefill producers write uncached TX
    bool pf_inplace = false;                   // STRIX_MM_PFX_INPLACE: o_proj / FFN outputs written straight into the TX slot
    bool pf_async_start = true, pf_late_src_fence = true;
    std::size_t pf_async_min_rows = 2048;
    // Persistent per-member/per-activation-bank fences. Children currently fit
    // one chunk, but must never alias the parked parent's event generations.
    struct PfFences {
        std::array<hipEvent_t, 4> last_r_use{}, p_free{};
        std::array<bool, 4> destination_untracked{};
        std::array<bool, 4> source_pending{};
    };
    std::array<PfFences, 2> pf_fences{};
    bool pf_comm_inflight = false;
    unsigned pf_group = 2;                     // STRIX_MM_PF_GROUP: chunks interleaved together (2..4)
    std::vector<float*> grp_h, grp_hn;         // residual / normed rows of group members 1.. (member 0 uses h / hn)
    std::vector<float*> grp_part, grp_remote;  // communicator backend: each member's local / peer partials (member 0: part / remote)
    bool pf_comm_interleave = true;            // the communicator backend is available (start/finish; used when the module is off)
    // DSpark taps: residual after layers 3, 17, 31, 45, 59 of the current call's rows, [rows][5][6144]
    bool taps_on = false;
    float* taps_d = nullptr;
    Engine::PrefillTapSink tap_sink;
    // StepTrace: events [start][layer][6][head end], routed ids [layer][8][4] (device + pinned host)
    bool trace_on = false;
    std::vector<hipEvent_t> tev;
    std::uint32_t *trace_ids_d = nullptr, *trace_ids_h = nullptr;
    StepTrace trace;
    bool tracing() const { return trace_on && gemv; }
    void tmark(std::size_t i) {
        if (tracing()) hip_ok(hipEventRecord(tev[i], nullptr), "trace event");
    }
    static std::size_t tmark_at(unsigned L, unsigned phase) { return 1 + std::size_t{L} * StepTrace::kPhases + phase; }

    // The events and streams created once per Engine (decode 13:52: ~154 runtime signals left at exit otherwise; one
    // Engine per process made it harmless, re-created Engines would leak). Device buffers stay with the process.
    ~Impl() {
        if (pf_comm_inflight) comm.cancel_pending("MiniMax destroyed during grouped prefill");
        (void)hipDeviceSynchronize();
        for (auto& bank : pf_fences) {
            for (auto ev : bank.last_r_use) if (ev) (void)hipEventDestroy(ev);
            for (auto ev : bank.p_free) if (ev) (void)hipEventDestroy(ev);
        }
        moe::opt::cleanup_prefill_table_ring();
        moe::opt::cleanup_prefill_dense();
        moe::opt::cleanup_prefill_tail_stream();
        for (auto* ptr : {nest_bank.h, nest_bank.hn, nest_bank.taps}) if (ptr) (void)hipFree(ptr);
        for (auto* events : {&idx_fork, &idx_join, &xb_attn_ev, &pf_idx_fork, &pf_idx_join, &pf_idx_fork_device, &pf_idx_join_device, &tev})
            for (hipEvent_t ev : *events)
                if (ev != nullptr) (void)hipEventDestroy(ev);
        if (routes_ready != nullptr) (void)hipEventDestroy(routes_ready);
        if (idx_side != nullptr) (void)hipStreamDestroy(idx_side);
        if (route_stream != nullptr) (void)hipStreamDestroy(route_stream);
        if (xb_warm_stream != nullptr) (void)hipStreamDestroy(xb_warm_stream);  // decode: warm side stream and its sink
        if (xb_warm_sink != nullptr) (void)hipFree(xb_warm_sink);
        for (void* ptr : {static_cast<void*>(pf_scores), static_cast<void*>(pf_partial), static_cast<void*>(pf_idx),
                          static_cast<void*>(pf_cnt), static_cast<void*>(pf_ticket)})
            if (ptr != nullptr) (void)hipFree(ptr);
    }
    // STRIX_MM_PREFILL_IDENTITY=fp32 (runtime, default off): prefill linears (q/k/v/o, dense MLP,
    // experts, shared) in the FP32 identity of src/minimax/rtxpf (FP32 activations, one fmaf chain per output), which the
    // RTX PRO prefill service reproduces bit for bit. Decode steps are unchanged.
    bool fp32_prefill = false;
    float *f32_xt = nullptr, *f32_raw = nullptr, *f32_g = nullptr, *f32_u = nullptr, *f32_ye = nullptr;
    rtxpf::F32Job *f32_jobs_h = nullptr, *f32_jobs_d = nullptr;
    std::uint32_t f32_jobs_used = 0;
    std::uint32_t *f32_map_h = nullptr, *f32_map_d = nullptr;  // [pair source rows][row slots n x 4][row weights n x 4]

    Impl(Options o, dist::Communicator& c) : opt(std::move(o)), comm(c), pack(opt.model_dir), lib(pack) {
        // Defaults-on resolver first (mm_defaults.hpp): every default-ON item's value (or its off value under
        // STRIX_MM_BASELINE=1) is in the environment before any switch below is read. The apps call it at start-up.
        defaults::apply();
        // Per-site linear schedule (v8/v9/v10) stays OFF until an engine-level ABA shows it never
        // slower (measurement inconclusive): linear v4. Wide expert kernels at N2-4 only (form 2) ON: measured processes
        // W0/W2 alternated with pinned cable weights, rows == alone, N2 -3.0..-3.5 %, N4 -4..-6 %, N1/N8 the same
        // kernels. An explicit value wins (STRIX_MMX_MOE_WIDE=0 = 4-column CTAs at every N).
        (void)::setenv("STRIX_MMX_LINEAR_V", "4", 0);
        (void)::setenv("STRIX_MMX_MOE_WIDE", "2", 0);
        if (opt.rank >= kWorld) throw std::invalid_argument("rank must be 0 or 1");
        if (opt.max_rows < kMaxStepRows) throw std::invalid_argument("max_rows < 8");
        moe::opt::set_prefill_mtfit(-1);
        moe::opt::set_prefill_core8(-1);
        moe::opt::set_prefill_pipe(-1);
        moe::opt::set_prefill_pipe_pair(-1);
        moe::opt::set_prefill_tail_stream(-1);
        moe::opt::set_prefill_table_ring(-1);
        moe::opt::set_prefill_gufuse(-1);
        profiling = env_flag("STRIX_MM_PROFILE", false);
        pf_table_stats = profiling || env_flag("STRIX_MM_PF_TABLE_STATS", false);
        const char* staging = std::getenv("STRIX_MM_STAGING");
        fused = staging != nullptr && std::string(staging) == "fused";
        shared_first = env_flag("STRIX_MM_SHARED_FIRST", false);
        const char* at = std::getenv("STRIX_MM_ATTN");
        attn_fused = at == nullptr || std::string(at) != "off";  // ON by default (measurement: bitwise, ABA x3 never slower); =off
        const char* ip = std::getenv("STRIX_MM_INDEX_PROJ");
        index_attn = ip == nullptr || std::string(ip) == "attn";  // default (measurement: same tokens, decode -7.9 %, TTFT -2 %); =tiled|off
        index_tiled = ip == nullptr || std::string(ip) != "off";  // ON by default (measurement: bitwise, TTFT -6 %, decode -9.6 %); =off
        index_conc = env_flag("STRIX_MM_INDEX_CONC", true);  // ON (measurement: ids identical, 17/18 in-process block pairs faster, ~-0.2 ms/token); =0
        attend_diet = env_flag("STRIX_MM_ATTEND_DIET", true);  // ON on the N2/W1 sparse >= 8K grids only (t9: exact, 8K 83.9 -> 76.2 us); =0
        attend4_decode = env_flag("STRIX_MM_ATTEND4_DECODE", true);  // ON on the same grids (t12: exact, 8K 78.6 -> 75.2 us); =0
        select_once = env_flag("STRIX_MM_SELECT_ONCE", true);  // ON on sparse calls of >= route::kSelectOnceRows rows (exact, same IDs); =0
        xb_wait = env_flag("STRIX_MM_XB_WAIT", true);         // ON (exchange boundary 0f1d0818: exact, 1K 48.56 -> 44.40 ms/token with the set); =0 disables
        xb_inplace = env_flag("STRIX_MM_XB_INPLACE", true);   // ON (exchange boundary 0f1d0818: exact, 1K 48.56 -> 44.40 ms/token with the set); =0 disables
        xb_prep = env_flag("STRIX_MM_XB_PREP", false);        // OFF (decode: not part of the measured default set)
        xb_pub = env_flag("STRIX_MM_XB_PUBLISH", true);       // ON (exchange boundary 0f1d0818: exact, 1K 48.56 -> 44.40 ms/token with the set); =0 disables; needs XB_INPLACE
        xb_postilp = true;  // mode 2 (LDS-staged partials) ON by default (exchange boundary set); STRIX_MM_XB_POSTILP=0 disables
        if (const char* pi = std::getenv("STRIX_MM_XB_POSTILP")) {  // value = mode
            const unsigned m = static_cast<unsigned>(std::strtoul(pi, nullptr, 10));
            xb_postilp = m != 0;
            if (m == 1 || m == 2) xb_postilp_mode = m;
        }
        if (const char* pm = std::getenv("STRIX_MM_XB_POSTILP_MODE")) xb_postilp_mode = std::strtoul(pm, nullptr, 10) == 1 ? 1U : 2U;
        if (xb_pub && !mmoe::producer_publish_supported()) {  // default ON falls back with other cores; explicit =1 fails
            if (!defaults::from_defaults("STRIX_MM_XB_PUBLISH"))
                throw std::invalid_argument("STRIX_MM_XB_PUBLISH=1 needs the v4 linear core and the decode MoE overlap form (STRIX_MMX_LINEAR_V=4, MOE defaults)");
            std::printf("STRIX_MM_XB_PUBLISH: default ON needs the v4 linear core and the decode MoE overlap form; off\n");
            xb_pub = false;
        }
        if (const char* wm = std::getenv("STRIX_MM_XB_WARM")) xb_warm_mb = static_cast<unsigned>(std::strtoul(wm, nullptr, 10));
        if (const char* wo = std::getenv("STRIX_MM_XB_WARM_O")) xb_warm_o_mb = static_cast<unsigned>(std::strtoul(wo, nullptr, 10));
        index_decode = env_flag("STRIX_MM_INDEX_DECODE", true);  // ON by default (measurement: same tokens, decode 129.4 -> 55.4 ms/token); =0
        const char* apf = std::getenv("STRIX_MM_ATTN_PREFILL");
        attn_prefill_fused = apf == nullptr || std::string(apf) != "off";  // ON by default (measurement: bitwise, TTFT -5.5 %); =off
        if (const char* rework = std::getenv("STRIX_MM_PF_MSA_REWORK")) {
            const std::string mode = rework;
            if (mode != "0" && mode != "1" && mode != "2" && mode != "3" && mode != "4" && mode != "5" && mode != "6")
                throw std::invalid_argument("STRIX_MM_PF_MSA_REWORK must be 0, 1, 2, 3, 4, 5 or 6");
            set_prefill_msa_rework(mode == "6" ? 6U : mode == "5" ? 5U : mode == "4" ? 4U : mode == "3" ? 3U : mode == "2" ? 2U : mode == "1" ? 1U : 0U);
        }
        pf_attend4 = env_flag("STRIX_MM_PF_ATTEND4", true);
        pf_msa_batch = env_flag("STRIX_MM_PF_MSA_BATCH", true);
        if (const char* value = std::getenv("STRIX_MM_PF_IDX_CONC")) {
            const std::string mode(value);
            if (mode != "0" && mode != "1" && mode != "2")
                throw std::invalid_argument("STRIX_MM_PF_IDX_CONC must be 0, 1 or 2");
            pf_idx_conc = mode == "2" ? 2 : mode == "1" ? 1 : 0;
        }
        pf_kv_tile = kv_tile_mode(std::getenv("STRIX_MM_PF_KV_TILE"));
        if (pf_kv_tile != -1) (void)kv_tile_available();
        pf_idx_tile = env_flag("STRIX_MM_PF_IDX_TILE", false);
        pf_dense_qkv = env_flag("STRIX_MM_PF_DENSE_QKV", true);
        pf_dense_o = env_flag("STRIX_MM_PF_DENSE_O", true);
        if (const char* sm = std::getenv("STRIX_MM_PF_MSA_SCRATCH_MB"); sm != nullptr && *sm != '\0') {
            const unsigned long mb = std::strtoul(sm, nullptr, 10);
            if (mb < 64 || mb > 8192) throw std::invalid_argument("STRIX_MM_PF_MSA_SCRATCH_MB must be 64..8192");
            pf_msa_scratch_mb = static_cast<unsigned>(mb);
        }
        if (const char* ir = std::getenv("STRIX_MM_PF_IDX_ROWS")) {
            const std::string value(ir);
            if (value != "16" && value != "32" && value != "64")
                throw std::invalid_argument("STRIX_MM_PF_IDX_ROWS must be 16, 32 or 64");
            pf_idx_rows = static_cast<unsigned>(std::strtoul(ir, nullptr, 10));
        }
        const char* lin = std::getenv("STRIX_MM_LINEARS");
        linears_loaded = lin == nullptr || std::string(lin) != "off";  // default ON (measurement: bitwise, ABA x3 never slower); =off
        linears_opt = lin == nullptr || std::string(lin) == "opt" || std::string(lin) == "on";  // =load: loaded, switched off
        const char* moe = std::getenv("STRIX_MM_MOE");
        moe_opt = moe == nullptr || std::string(moe) != "off";  // default ON (measurement: bitwise, decode -15.5 %, TTFT -10.5 %); =off
        if (const char* async_mode = std::getenv("STRIX_MM_PF_ASYNC_START")) pf_async_start = std::string(async_mode) != "0";
        if (const char* late_fence = std::getenv("STRIX_MM_PF_LATE_SRC_FENCE")) pf_late_src_fence = std::string(late_fence) != "0";
        pf_async_min_rows = defaults::pf_async_min_rows(std::getenv("STRIX_MM_PF_ASYNC_MIN_ROWS"));
        if (pf_async_start) comm.enable_async_start();
        // STRIX_MM_PF_INTERLEAVE: unset / 1 / on = chunk interleave over the module when it is on, else over the
        // communicator (default; measurement: tokens + K/V/IK state + final rows identical, 8K TTFT 40.0 -> 35.4 s, 1K inert);
        // comm = the communicator even with the module loaded; 0 / off = chunk-serial.
        if (const char* il = std::getenv("STRIX_MM_PF_INTERLEAVE"); il != nullptr && *il != '\0') {
            const std::string mode(il);
            if (mode == "0" || mode == "off") pf_interleave = false;
            else if (mode != "1" && mode != "on" && mode != "comm") throw std::invalid_argument("STRIX_MM_PF_INTERLEAVE must be 0|off|1|on|comm");
            pf_force_comm = mode == "comm";
        }
        pf_router_batch = env_flag("STRIX_MM_PF_ROUTER_BATCH", true);
        if (const char* rr = std::getenv("STRIX_MM_PF_ROUTER_ROWS"); rr != nullptr && *rr != '\0') {
            const std::string mode(rr);
            if (mode != "4" && mode != "8" && mode != "32") throw std::invalid_argument("STRIX_MM_PF_ROUTER_ROWS must be 4, 8 or 32");
            pf_router_rows = mode == "4" ? 4U : mode == "8" ? 8U : 32U;
        }
        pf_inplace = env_flag("STRIX_MM_PFX_INPLACE", false);
        pf_tx_uncached = env_flag("STRIX_MM_PF_TX_UNCACHED", true);  // ON (measurement: exact, send 0.23 -> 0.002 s); =0
        if (const char* g = std::getenv("STRIX_MM_PF_GROUP"); g != nullptr && *g != '\0') {
            pf_group = static_cast<unsigned>(std::stoul(g));
            if (pf_group < 2 || pf_group > 4) throw std::invalid_argument("STRIX_MM_PF_GROUP must be 2..4");
        }
        const char* pid = std::getenv("STRIX_MM_PREFILL_IDENTITY");
        fp32_prefill = route::fp32_profile(pid);
        if (fp32_prefill && (!linears_loaded || !moe_opt)) throw std::invalid_argument("STRIX_MM_PREFILL_IDENTITY=fp32 needs STRIX_MM_LINEARS and STRIX_MM_MOE on");
    }

    std::size_t cache_off(std::uint32_t slot, unsigned layer, std::size_t width) const {
        return ((static_cast<std::size_t>(slot) * kLayers + layer) * opt.capacity) * width;
    }

    // Synchronised phase timer (profiling only).
    Clock::time_point mark_t{};
    void mark() {
        if (!profiling) return;
        hip_ok(hipDeviceSynchronize(), "profile");
        mark_t = Clock::now();
    }
    void lap(double& acc) {
        if (!profiling) return;
        hip_ok(hipDeviceSynchronize(), "profile");
        const auto t = Clock::now();
        acc += secs(mark_t, t);
        mark_t = t;
    }

    LoadReport load() {
        const auto t0 = Clock::now();
        static const char* const kMlp[3] = {"gate_proj", "up_proj", "down_proj"};
        static const char* const kExp[3] = {"w1", "w3", "w2"};
        const std::uint32_t rank = opt.rank;
        auto mlp = [&](const std::string& pre, const char* const* nm, unsigned inter_local) {
            return Mlp{lib.out_slice(pre + nm[0], rank * inter_local, inter_local),
                       lib.out_slice(pre + nm[1], rank * inter_local, inter_local),
                       lib.in_slice(pre + nm[2], rank * inter_local, inter_local)};
        };
        w_in0 = dupload(pack.floats(kP + "layers.0.input_layernorm.weight"));
        {
            const TensorInfo& e = pack.info(kP + "embed_tokens.weight");
            if (e.dtype != "BF16" || e.bytes != kVocab * kHidden * 2) throw std::runtime_error("unexpected embedding tensor");
            embed = dalloc<std::uint16_t>(kVocab * kHidden);
            lib.upload_raw(kP + "embed_tokens.weight", embed);
        }
        layers.resize(kLayers);
        for (unsigned L = 0; L < kLayers; ++L) {
            Layer& y = layers[L];
            const std::string lp = kP + "layers." + std::to_string(L) + ".", a = lp + "self_attn.", b = lp + "block_sparse_moe.";
            y.q = lib.out_slice(a + "q_proj", rank * kQLocal, kQLocal);
            y.k = lib.out_slice(a + "k_proj", rank * kKVLocal, kKVLocal);
            y.v = lib.out_slice(a + "v_proj", rank * kKVLocal, kKVLocal);
            y.o = lib.in_slice(a + "o_proj", rank * kQLocal, kQLocal);
            if (linears_loaded) {
                y.rq = lib.raw_out(a + "q_proj", rank * kQLocal, kQLocal);
                y.rk = lib.raw_out(a + "k_proj", rank * kKVLocal, kKVLocal);
                y.rv = lib.raw_out(a + "v_proj", rank * kKVLocal, kKVLocal);
                y.ro = lib.raw_in(a + "o_proj", rank * kQLocal, kQLocal);
            }
            const auto wiq_all = pack.floats(a + "index_q_proj.weight");  // [512][6144]
            const std::vector<float> wiq_mine(wiq_all.begin() + rank * kIdxLocal * kHidden,
                                              wiq_all.begin() + (rank + 1) * kIdxLocal * kHidden);
            const auto wiq_t = transpose_f(wiq_mine, kIdxLocal, kHidden);
            const auto wik_t = transpose_f(pack.floats(a + "index_k_proj.weight"), 128, kHidden);
            y.wiq = dupload(wiq_t);
            y.wik = dupload(wik_t);
            auto to16 = [](const std::vector<float>& f) {  // exact: the values are F16 in the file
                std::vector<_Float16> h16(f.size());
                for (std::size_t i = 0; i < f.size(); ++i) h16[i] = static_cast<_Float16>(f[i]);
                return h16;
            };
            y.wiq16 = dupload(to16(wiq_t));
            y.wik16 = dupload(to16(wik_t));
            auto block_major = [](const std::vector<float>& t, std::size_t d, std::size_t n) {  // [d][n] -> [n/32][d][32]
                std::vector<_Float16> b(t.size());
                for (std::size_t ob = 0; ob < n / kDecOut; ++ob)
                    for (std::size_t i = 0; i < d; ++i)
                        for (std::size_t j = 0; j < kDecOut; ++j)
                            b[(ob * d + i) * kDecOut + j] = static_cast<_Float16>(t[i * n + ob * kDecOut + j]);
                return b;
            };
            {
                const auto wik_rows = pack.floats(a + "index_k_proj.weight");  // [128][6144]
                y.wip = dupload(strix::minimax::attn::pack_index_proj(wiq_mine.data(), wik_rows.data()));
            }
            y.wiq16b = dupload(block_major(wiq_t, kHidden, kIdxLocal));
            y.wik16b = dupload(block_major(wik_t, kHidden, 128));
            const char* names[4] = {"q_norm", "k_norm", "index_q_norm", "index_k_norm"};
            for (int i = 0; i < 4; ++i) y.norm[i] = dupload(pack.floats(a + names[i] + ".weight"));
            y.w_post = dupload(pack.floats(lp + "post_attention_layernorm.weight"));
            y.w_next = dupload(pack.floats(L + 1 < kLayers ? kP + "layers." + std::to_string(L + 1) + ".input_layernorm.weight"
                                                           : kP + "norm.weight"));
            y.moe = L >= kFirstMoe;
            if (!y.moe) {
                y.dense = mlp(lp + "mlp.", kMlp, kDenseLocal);
                if (linears_loaded)
                    y.rdense = mmoe::ExpertSlices{lib.raw_out(lp + "mlp." + kMlp[0], rank * kDenseLocal, kDenseLocal),
                                                  lib.raw_out(lp + "mlp." + kMlp[1], rank * kDenseLocal, kDenseLocal),
                                                  lib.raw_in(lp + "mlp." + kMlp[2], rank * kDenseLocal, kDenseLocal)};
            } else {
                y.wr = dupload(pack.floats(b + "gate.weight"));
                y.wb = dupload(pack.floats(b + "e_score_correction_bias"));
                if (moe_opt) {
                    auto raw = [&](const std::string& pre, const char* const* nm) {
                        return mmoe::ExpertSlices{lib.raw_out(pre + nm[0], rank * kExpertLocal, kExpertLocal),
                                                  lib.raw_out(pre + nm[1], rank * kExpertLocal, kExpertLocal),
                                                  lib.raw_in(pre + nm[2], rank * kExpertLocal, kExpertLocal)};
                    };
                    y.experts_raw.reserve(kExperts);
                    for (unsigned e = 0; e < kExperts; ++e) y.experts_raw.push_back(raw(b + "experts." + std::to_string(e) + ".", kExp));
                    y.experts_raw_d = dupload(y.experts_raw);
                    y.shared_raw = raw(b + "shared_experts.", kMlp);
                } else {
                    y.experts.reserve(kExperts);
                    for (unsigned e = 0; e < kExperts; ++e) y.experts.push_back(mlp(b + "experts." + std::to_string(e) + ".", kExp, kExpertLocal));
                    y.shared = mlp(b + "shared_experts.", kMlp, kExpertLocal);
                }
            }
            hip_ok(hipDeviceSynchronize(), "load layer");
            if (L % 10 == 9) {
                std::size_t fr = 0, tot = 0;
                hip_ok(hipMemGetInfo(&fr, &tot), "meminfo");
                std::printf("mm rank %u: loaded layers 0-%u, %.1f s, %.1f GiB read, device used %.1f GiB\n", rank, L,
                            secs(t0, Clock::now()), pack.gib_read(), static_cast<double>(tot - fr) / 1073741824.0);
                std::fflush(stdout);
            }
        }
        const std::uint64_t v0 = rank == 0 ? 0 : kVocabPad;
        vn = rank == 0 ? kVocabPad : kVocab - kVocabPad;
        head = lib.out_slice("language_model.lm_head", v0, vn);
        hip_ok(hipDeviceSynchronize(), "load head");
        lib.release_staging();
        alloc();
        LoadReport r;
        r.seconds = secs(t0, Clock::now());
        r.gib_read = pack.gib_read();
        std::size_t fr = 0, tot = 0;
        hip_ok(hipMemGetInfo(&fr, &tot), "meminfo");
        r.device_gib = static_cast<double>(tot - fr) / 1073741824.0;
        r.direct_io = env_flag("STRIX_MM_LOAD_DIRECT", true);
        return r;
    }

    void alloc() {
        const std::size_t T = opt.max_rows, C = opt.capacity, S = opt.slots;
        h = dalloc<float>(T * kHidden);
        hn = dalloc<float>(T * kHidden);
        q = dalloc<float>(T * kQLocal);
        k = dalloc<float>(T * kKVLocal);
        v = dalloc<float>(T * kKVLocal);
        iq = dalloc<float>(T * kIdxLocal);
        ik = dalloc<float>(T * 128);
        attn = dalloc<float>(T * kQLocal);
        big1 = dalloc<float>(T * kDenseLocal);
        big2 = dalloc<float>(T * kDenseLocal);
        xe = dalloc<float>(T * kHidden);
        xs = dalloc<float>(T * kHidden);
        hip_ok(hipStreamCreateWithFlags(&route_stream, hipStreamNonBlocking), "route stream");
        hip_ok(hipStreamCreateWithFlags(&idx_side, hipStreamNonBlocking), "index stream");
        hip_ok(hipStreamCreateWithFlags(&xb_warm_stream, hipStreamNonBlocking), "warm stream");  // STRIX_MM_XB_WARM
        hip_ok(hipMalloc(&xb_warm_sink, 4096 * sizeof(unsigned)), "warm sink");
        // one fork/join pair per layer: re-recorded only in the next step, after its host synchronization
        idx_fork.resize(kLayers);
        idx_join.resize(kLayers);
        for (unsigned l = 0; l < kLayers; ++l) {
            hip_ok(hipEventCreateWithFlags(&idx_fork[l], hipEventDisableTiming), "index fork event");
            hip_ok(hipEventCreateWithFlags(&idx_join[l], hipEventDisableTiming), "index join event");
        }
        xb_attn_ev.resize(kLayers);  // STRIX_MM_XB_WARM_O: same lifetime rule as the index fork/join pairs
        for (unsigned l = 0; l < kLayers; ++l) hip_ok(hipEventCreateWithFlags(&xb_attn_ev[l], hipEventDisableTiming), "warm o event");
        // Distinct events for every pending chunk/layer. A later group can reuse
        // these slots: each call joins before any shared iq/ik scratch is reused.
        pf_idx_fork.resize(std::size_t{pf_group} * kLayers);
        pf_idx_join.resize(pf_idx_fork.size());
        for (std::size_t i = 0; i < pf_idx_fork.size(); ++i) {
            hip_ok(hipEventCreateWithFlags(&pf_idx_fork[i], hipEventDisableTiming), "prefill index fork");
            hip_ok(hipEventCreateWithFlags(&pf_idx_join[i], hipEventDisableTiming), "prefill index join");
        }
        pf_idx_fork_device.resize(pf_idx_fork.size());
        pf_idx_join_device.resize(pf_idx_fork.size());
        for (std::size_t i = 0; i < pf_idx_fork_device.size(); ++i) {
            const unsigned flags = hipEventDisableTiming | hipEventDisableSystemFence;
            hip_ok(hipEventCreateWithFlags(&pf_idx_fork_device[i], flags), "prefill index device fork");
            hip_ok(hipEventCreateWithFlags(&pf_idx_join_device[i], flags), "prefill index device join");
        }
        {
            const std::size_t tab = std::size_t{kLayers} * 3 * kMaxStepRows;
            hip_ok(hipHostMalloc(reinterpret_cast<void**>(&cache_tab_h), tab * sizeof(void*), hipHostMallocDefault), "cache table");
            cache_tab_d = dalloc<std::uint16_t*>(tab);
            ticket = dalloc<std::uint32_t>(16 * 2 * 4);
            hip_ok(hipMemset(ticket, 0, 16 * 2 * 4 * 4), "ticket");
            hip_ok(hipHostMalloc(reinterpret_cast<void**>(&pf_tab_h), std::size_t{kLayers} * 3 * sizeof(void*), hipHostMallocDefault), "prefill table");
            pf_tab_d = dalloc<std::uint16_t*>(std::size_t{kLayers} * 3);
            step_rope = dalloc<msa::RopeRow>(kMaxStepRows);
            out_ids_d = dalloc<std::uint32_t>(kMaxStepRows);
            step_pos = dalloc<std::uint64_t>(kMaxStepRows);
            trace_ids_d = dalloc<std::uint32_t>(std::size_t{kLayers} * kMaxStepRows * kTopE);
            hip_ok(hipMemset(trace_ids_d, 0, std::size_t{kLayers} * kMaxStepRows * kTopE * 4), "trace ids");
            hip_ok(hipHostMalloc(reinterpret_cast<void**>(&trace_ids_h), std::size_t{kLayers} * kMaxStepRows * kTopE * 4,
                                 hipHostMallocDefault), "trace ids host");
        }
        if (linears_loaded) {  // the largest call shape of each site (q+k+v, o, dense gate+up, dense down)
            const Layer& y0 = layers[0];
            const mmoe::LinearJob qkv[3] = {{y0.rq, hn, kHidden, q, kQLocal}, {y0.rk, hn, kHidden, k, kKVLocal},
                                            {y0.rv, hn, kHidden, v, kKVLocal}};
            const mmoe::LinearJob o1{y0.ro, attn, kQLocal, part, kHidden};
            const mmoe::LinearJob gu[2] = {{y0.rdense.gate, hn, kHidden, big1, kDenseLocal}, {y0.rdense.up, hn, kHidden, big2, kDenseLocal}};
            const mmoe::LinearJob dn{y0.rdense.down, big1, kDenseLocal, part, kHidden};
            const std::size_t wb = std::max({mmoe::linear_workspace_bytes(qkv, 3), mmoe::linear_workspace_bytes(&o1, 1),
                                             mmoe::linear_workspace_bytes(gu, 2), mmoe::linear_workspace_bytes(&dn, 1)});
            hip_ok(hipMalloc(&lin_ws, wb), "linear workspace");
            hip_ok(hipMemset(lin_ws, 0, wb), "linear workspace");
        }
        if (moe_opt) {
            const std::size_t wb = mmoe::moe_workspace_bytes();
            hip_ok(hipMalloc(&moe_ws.base, wb), "moe workspace");
            hip_ok(hipMemset(moe_ws.base, 0, wb), "moe workspace");
            pf_ws_bytes = mmoe::prefill_moe_workspace_bound(opt.max_rows);
            hip_ok(hipMalloc(&pf_ws, pf_ws_bytes), "prefill moe workspace");
        }
        hip_ok(hipEventCreateWithFlags(&routes_ready, hipEventDisableTiming), "route event");
        if (fp32_prefill) {  // FP32 identity workspaces: expert pairs (T x 4 rows) at the widest site
            const std::size_t pairs = T * kTopE;
            f32_xt = dalloc<float>(pairs * kHidden);
            f32_raw = dalloc<float>(pairs * kHidden);
            f32_g = dalloc<float>(pairs * kExpertLocal);
            f32_u = dalloc<float>(pairs * kExpertLocal);
            f32_ye = dalloc<float>(pairs * kHidden);
            hip_ok(hipHostMalloc(reinterpret_cast<void**>(&f32_jobs_h), kF32JobRing * sizeof(rtxpf::F32Job), hipHostMallocDefault), "fp32 jobs");
            f32_jobs_d = dalloc<rtxpf::F32Job>(kF32JobRing);
            hip_ok(hipHostMalloc(reinterpret_cast<void**>(&f32_map_h), 3 * pairs * 4, hipHostMallocDefault), "fp32 maps");
            f32_map_d = dalloc<std::uint32_t>(3 * pairs);
        }
        ye = dalloc<float>(T * kHidden);
        sav_h = dalloc<float>(kMaxStepRows * kHidden);
        sav_hn = dalloc<float>(kMaxStepRows * kHidden);
        sav_taps = dalloc<float>(kMaxStepRows * 5 * kHidden);
        route_d = dalloc<std::uint32_t>(2 * T * kTopE);
        pack_d = dalloc<std::uint32_t>(2 * T * kTopE);
        hip_ok(hipHostMalloc(reinterpret_cast<void**>(&route_h), 2 * T * kTopE * 4, hipHostMallocDefault), "route host");
        hip_ok(hipHostMalloc(reinterpret_cast<void**>(&pack_h), 2 * T * kTopE * 4, hipHostMallocDefault), "pack host");
        ids_d = route_d;
        wts4_d = reinterpret_cast<float*>(route_d + T * kTopE);
        router_scratch = dalloc<float>(moe::hip::router_tiled_scratch_floats(static_cast<std::uint32_t>(T)));  // any router form can be selected later
        part = dalloc<float>(T * kHidden);
        remote = dalloc<float>(T * kHidden);
        if (pf != nullptr && pf->options().depth < pf_group)
            throw std::invalid_argument("STRIX_MM_PFX_DEPTH must be >= STRIX_MM_PF_GROUP (slot reuse of the interleaved schedule)");
        if (pf != nullptr || pf_comm_interleave) {  // at start-up: no multi-GB allocation inside a prefill (KFD queue eviction)
            for (unsigned g = 1; g < pf_group; ++g) {
                grp_h.push_back(dalloc<float>(T * kHidden));
                grp_hn.push_back(dalloc<float>(T * kHidden));
                if (pf_comm_interleave) {
                    grp_part.push_back(dalloc<float>(T * kHidden));
                    grp_remote.push_back(dalloc<float>(T * kHidden));
                }
            }
        }
        for (auto& bank : pf_fences) {
            for (unsigned c = 0; c < pf_group; ++c) {
                hip_ok(hipEventCreateWithFlags(&bank.last_r_use[c], hipEventDisableTiming), "prefill destination event");
                hip_ok(hipEventCreateWithFlags(&bank.p_free[c], hipEventDisableTiming), "prefill source event");
                hip_ok(hipEventRecord(bank.last_r_use[c], nullptr), "prefill initial destination lifetime");
            }
        }
        kc = dalloc<std::uint16_t>(S * kLayers * C * kKVLocal);
        vc = dalloc<std::uint16_t>(S * kLayers * C * kKVLocal);
        ikc = dalloc<std::uint16_t>(S * kLayers * C * 128);
        const auto max_blocks = msa::block_count(C);
        // MSA scratch for up to 16 rows per call (fused prefill windows; decode uses <= 8)
        scores = dalloc<float>(16ULL * 2 * max_blocks);
        idx = dalloc<std::uint32_t>(16ULL * 2 * msa::kTopK);
        cnt = dalloc<std::uint32_t>(16ULL * 2);
        partial = dalloc<float>(16ULL * 32 * msa::kMaxChunks * 130);
        {
            const std::size_t per_window = mfused::prefill_window_bytes(static_cast<std::uint32_t>(max_blocks));
            // STRIX_MM_PF_MSA_SCRATCH_MB (default 1152: every window of a 4096-row piece in one batch; =512 the former cap).
            // Windows are independent, so the batch size changes only launches, never bits. Below 4 GiB of headroom
            // after the request the former 512 MiB cap is kept.
            constexpr std::size_t former = 512ULL * 1024 * 1024;
            std::size_t limit = std::size_t{pf_msa_scratch_mb} * 1024 * 1024;
            if (limit > former) {
                std::size_t free_b = 0, total_b = 0;
                if (hipMemGetInfo(&free_b, &total_b) != hipSuccess || free_b < limit + (4ULL << 30)) {
                    std::fprintf(stderr, "prefill MSA scratch: %u MiB requested, keeping 512 MiB (device headroom)\n", pf_msa_scratch_mb);
                    limit = former;
                }
            }
            const std::size_t wanted = (std::size_t{opt.max_rows} + 15) / 16;
            if (per_window > limit) throw std::invalid_argument("prefill MSA scratch window exceeds the scratch cap");
            pf_msa_windows = static_cast<unsigned>(std::min(wanted, limit / per_window));
            const std::size_t scratch_rows = std::size_t{pf_msa_windows} * 16;
            pf_scores = dalloc<float>(scratch_rows * 2 * max_blocks);
            pf_idx = dalloc<std::uint32_t>(scratch_rows * 2 * msa::kTopK);
            pf_cnt = dalloc<std::uint32_t>(scratch_rows * 2);
            pf_partial = dalloc<float>(scratch_rows * 32 * msa::kMaxChunks * 130);
            pf_ticket = dalloc<std::uint32_t>(scratch_rows * 2 * 4);
            hip_ok(hipMemset(pf_ticket, 0, scratch_rows * 2 * 4 * sizeof(std::uint32_t)), "prefill MSA tickets");
            std::fprintf(stderr, "prefill MSA scratch: %zu bytes, %u windows (%zu requested)\n",
                         per_window * pf_msa_windows, pf_msa_windows, wanted);
        }
        if (env_flag("STRIX_MM_PF_TIMELINE", false))
            timeline = std::make_unique<PrefillTimeline>(opt.capacity, opt.max_rows, kLayers, pf_msa_windows);
        logits = dalloc<float>(std::max<std::size_t>(kGatherRows, kMaxStepRows) * kVocabPad);
        rows_d = dalloc<std::uint32_t>(T);
        tok_d = dalloc<std::uint32_t>(T);
        am_local = dalloc<std::uint32_t>(2 * kMaxStepRows);
        am_remote = dalloc<std::uint32_t>(2 * kMaxStepRows);
        ctl_local = dalloc<std::uint8_t>(kControlBytes);
        ctl_remote = dalloc<std::uint8_t>(kControlBytes);
        std::vector<msa::RopeRow> rope(C);
        std::vector<std::uint64_t> pos(C);
        for (std::size_t t = 0; t < C; ++t) {
            pos[t] = t;
            rope[t] = msa::rope_row(t, 1e7);
        }
        d_rope = dupload(rope);
        d_pos = dupload(pos);
        hip_ok(hipDeviceSynchronize(), "alloc");
        reserve_workspace();
    }

    // Lib::mm's workspace for every distinct linear shape (outputs, inputs, bits) at every row count a call can have
    // (1..max_rows; the head up to G4's gather chunk), queried on the real activation buffers (review finding: the first
    // long prompt after start-up no longer synchronises the device to free / allocate a larger workspace).
    void reserve_workspace(unsigned bound_rows = 0) {
        const auto t0 = Clock::now();
        const std::uint64_t T = bound_rows == 0 ? opt.max_rows : bound_rows;
        const auto in_of = [&](std::uint64_t cols) -> float* {
            return cols == kHidden ? hn : cols == kQLocal ? attn : cols <= kDenseLocal ? big1 : nullptr;
        };
        const auto out_of = [&](std::uint64_t rows) -> float* {
            return rows == kHidden ? part : rows == kQLocal ? q : rows == kKVLocal ? k : rows <= kDenseLocal ? big2 : nullptr;
        };
        std::vector<std::array<std::uint64_t, 3>> seen;
        std::uint64_t most = 0;
        std::size_t shapes = 0;
        const auto add = [&](const Lin& l, float* in, float* out, std::uint64_t t_max) {
            if (l.w == nullptr || in == nullptr || out == nullptr || t_max == 0) return;
            const std::array<std::uint64_t, 3> key{l.rows, l.cols, l.bits};
            if (std::find(seen.begin(), seen.end(), key) != seen.end()) return;
            seen.push_back(key);
            ++shapes;
            most = std::max(most, lib.workspace_bound(l, in, out, t_max));
        };
        const auto add_mlp = [&](const Mlp& m) {
            add(m.g, in_of(m.g.cols), out_of(m.g.rows), T);
            add(m.u, in_of(m.u.cols), out_of(m.u.rows), T);
            add(m.d, in_of(m.d.cols), out_of(m.d.rows), T);
        };
        for (const Layer& y : layers) {
            for (const Lin* l : {&y.q, &y.k, &y.v, &y.o}) add(*l, in_of(l->cols), out_of(l->rows), T);
            add_mlp(y.dense);
            add_mlp(y.shared);
            for (const Mlp& e : y.experts) add_mlp(e);
        }
        add(head, hn, logits, std::min<std::uint64_t>(T, std::max<std::size_t>(kGatherRows, kMaxStepRows)));
        if (bound_rows != 0 && most > lib.workspace_bytes())
            throw std::runtime_error("nested bank needs a larger matmul workspace");
        lib.reserve_workspace(std::max(lib.workspace_bytes(), most));
        std::printf("rank %u: matmul workspace reserved %.1f MiB (%zu linear shapes x 1..%llu rows, %.0f ms)\n", opt.rank,
                    static_cast<double>(most) / 1048576.0, shapes, static_cast<unsigned long long>(T),
                    std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }

    // Where the producer of an exchanged partial writes: the communicator's pinned send buffer with fused staging
    // (STRIX_MM_STAGING=fused, host-path routes only: no device-to-host copy; the post-exchange kernel then reads the
    // peer's partial in the pinned receive buffer), else the device buffer `part`.
    float* send_buffer(std::size_t n) {
        if (pf_active && (pf_inplace || (pf_tx_uncached && n > kMaxStepRows))) return static_cast<float*>(pf->tx());  // direct prefill TX
        if (xb_inplace_now()) {  // STRIX_MM_XB_INPLACE: the decode producer writes the module's next TX slot
            xb_tx = static_cast<float*>(xf->tx());
            xb_pt = {};
            xb_published = false;
            if (xb_pub) {  // STRIX_MM_XB_PUBLISH: ... and publishes it itself (supported forms checked when the flag is set)
                const auto t = xf->producer_target();
                xb_pt = mmoe::PublishTarget{t.flag, t.counter, t.sequence};
            }
            return xb_tx;
        }
        if (!fused || use_xfer()) return part;
        void* s = comm.send_staging(n * kHidden * sizeof(float));
        return s != nullptr ? static_cast<float*>(s) : part;
    }

    // STRIX_MM_XB_INPLACE applies to the decode module path with the optimized producers only (enqueue_linears o_proj /
    // dense down, enqueue_moe); no decode reader of the partial exists after its exchange (taps read h, tracing records
    // events and routes), so taps / tracing keep the in-place path.
    bool xb_inplace_now() const { return xb_inplace && use_xfer() && !pf_active && linears_opt && moe_opt; }

    // STRIX_MM_XB_WARM: once the message `seq` is published (its TX flag), the side stream reads up to xb_warm_mb MB of
    // the weights the phase after this exchange streams first: after the attention exchange of layer L the router and
    // the shared gate / up of L (MoE layers) or the dense gate / up (L0-2); after the FFN exchange the next layer's
    // q / k / v. Cache state only (no data dependency, the main stream never waits for it).
    void xb_warm(unsigned layer, unsigned which, const std::uint64_t* flag, std::uint64_t seq) {
        WarmRanges w;
        w.sink = xb_warm_sink;
        std::size_t left = std::size_t{xb_warm_mb} << 20;
        int nr = 0;
        auto add = [&](const void* p, std::size_t bytes) {
            if (p == nullptr || left == 0 || nr >= 4) return;
            const std::size_t b = std::min(bytes, left) & ~std::size_t{15};
            w.p[nr] = static_cast<const uint4*>(p);
            w.n16[nr] = b / 16;
            left -= b;
            ++nr;
        };
        auto tr = [](const mmoe::Exl3& e) { return std::size_t{e.input} * e.output * e.bits / 8; };
        if (which == 0) {
            const Layer& y = layers[layer];
            if (y.moe) {
                add(y.wr, std::size_t{kExperts} * kHidden * sizeof(float));
                add(y.shared_raw.gate.trellis, tr(y.shared_raw.gate));
                add(y.shared_raw.up.trellis, tr(y.shared_raw.up));
            } else {
                add(y.rdense.gate.trellis, tr(y.rdense.gate));
                add(y.rdense.up.trellis, tr(y.rdense.up));
            }
        } else if (layer + 1 < xb_layers) {
            const Layer& y = layers[layer + 1];
            add(y.rq.trellis, tr(y.rq));
            add(y.rk.trellis, tr(y.rk));
            add(y.rv.trellis, tr(y.rv));
        }
        if (nr == 0) return;
        hip_ok(hipStreamWaitValue64(xb_warm_stream, const_cast<std::uint64_t*>(flag), seq, hipStreamWaitValueGte, ~std::uint64_t{0}),
               "warm gate");
        hipLaunchKernelGGL(xb_warm_kernel, dim3(160), dim3(256), 0, xb_warm_stream, w);
        hip_ok(hipGetLastError(), "warm kernel");
        ++xb_count[4];
    }

    // STRIX_MM_XB_WARM_O: the MSA kernels of a decode layer leave memory nearly idle (latency-bound, ~2-4 MB of cache
    // reads); the warm stream reads the first MB of the layer's o_proj trellis meanwhile, gated on the end of the q/k/v
    // and index projections (so it does not compete with them). Cache state only, as xb_warm.
    void xb_warm_o(unsigned layer) {
        const mmoe::Exl3& o = layers[layer].ro;
        const std::size_t bytes = std::min(std::size_t{o.input} * o.output * o.bits / 8, std::size_t{xb_warm_o_mb} << 20) & ~std::size_t{15};
        if (o.trellis == nullptr || bytes == 0) return;
        WarmRanges w;
        w.sink = xb_warm_sink;
        w.p[0] = reinterpret_cast<const uint4*>(o.trellis);
        w.n16[0] = bytes / 16;
        hip_ok(hipEventRecord(xb_attn_ev[layer], nullptr), "warm o record");
        hip_ok(hipStreamWaitEvent(xb_warm_stream, xb_attn_ev[layer], 0), "warm o gate");
        hipLaunchKernelGGL(xb_warm_kernel, dim3(160), dim3(256), 0, xb_warm_stream, w);
        hip_ok(hipGetLastError(), "warm o kernel");
        ++xb_warm_o_count;
    }

    // Every post-exchange launch of the engine (STRIX_MM_XB_POSTILP selects the loads-first form, same bits).
    void enqueue_post(moe::hip::PostExchangeView view, hipStream_t stream) {
        view.ilp = xb_postilp ? xb_postilp_mode : 0U;
        if (xb_postilp) ++xb_count[5];
        if (pf_router_batch && view.rows > 8 && view.router_w != nullptr) {
            const float* router_w = view.router_w;
            view.router_w = nullptr;
            moe::hip::enqueue_post_exchange(view, stream);
            // Tiled scores only on the measured piece sizes (1024..4096 rows); other piece sizes keep the 8-row form.
            if (pf_router_rows == 32 && router_scratch != nullptr && view.rows >= 1024 && view.rows <= 4096)
                moe::hip::enqueue_router_tiled(view.rows, view.y, router_w, view.bias, view.ids, view.weights, router_scratch, stream);
            else
                moe::hip::enqueue_router_batch(view.rows, view.y, router_w, view.bias, view.ids, view.weights, pf_router_rows == 4 ? 4U : 8U, stream);
        } else {
            moe::hip::enqueue_post_exchange(view, stream);
        }
    }

    // The post-exchange kernel on a prefill module view: rank 0's / rank 1's partials read in place from the TX / RX slots.
    void post_view(std::size_t n, const strix::minimax::pfx::Exchanger::View& x, const float* norm_w, const float* router_w,
                   const float* bias) {
        const auto* local = static_cast<const float*>(x.local);
        const auto* peer = static_cast<const float*>(x.peer);
        wts4_d = reinterpret_cast<float*>(route_d + n * kTopE);
        enqueue_post({static_cast<std::uint32_t>(n), h, opt.rank == 0 ? local : peer, opt.rank == 0 ? peer : local,
                                         norm_w, hn, router_w, bias, ids_d, wts4_d}, nullptr);
        if (pf_active) pf->retire();  // the only consumer of both slots is enqueued
    }

    void exchange_post(std::size_t n, std::uint32_t layer, std::uint32_t which, const float* local, const float* norm_w,
                       const float* router_w, const float* bias) {
        mark();
        const std::size_t bytes = n * kHidden * sizeof(float);
        if (pf_active) {  // prefill module: the copy kernel (or a publish kernel after an in-place producer), then the RX wait
            if (local == pf->tx()) pf->send(bytes, strix::minimax::pfx::Tag{which + 1, layer, calls});
            else pf->send_copy(local, bytes, strix::minimax::pfx::Tag{which + 1, layer, calls});
            const auto x = pf->recv();
            lap(prof.exchange);
            post_view(n, x, norm_w, router_w, bias);
            lap(prof.post);
            return;
        }
        if (use_xfer()) {  // the module: device partial copied into its TX slot (publishing), the stream waits for RX
            const mx::Tag tag{which == 0 ? mx::Kind::attn_sum : mx::Kind::ffn_sum, layer, calls};
            const auto warm_pt = xf->producer_target();  // this message's TX flag and sequence (STRIX_MM_XB_WARM gate)
            const bool inplace = xb_tx != nullptr && local == xb_tx;  // produced in the TX slot: publish only
            const bool self_pub = inplace && xb_published;          // ... by the producer itself (nothing to enqueue)
            xb_tx = nullptr;
            xb_pt = {};
            xb_published = false;
            if (inplace) {
                xf->send(bytes, tag, self_pub ? mx::Publish::producer : mx::Publish::kernel);
                ++xb_count[1];
                if (self_pub) ++xb_count[3];
            } else {
                xf->send_copy(local, bytes, tag);
            }
            if (xb_warm_mb != 0) xb_warm(layer, which, warm_pt.flag, warm_pt.sequence);
            moe::hip::PostExchangeView pv{static_cast<std::uint32_t>(n), h, nullptr, nullptr, norm_w, hn, router_w, bias, ids_d, nullptr};
            const float* peer = nullptr;
            if (xb_wait) {  // gated in-kernel wait (module recv_view): no stream wait packet before post_exchange
                const auto rv = xf->recv_view();
                peer = static_cast<const float*>(rv.payload);
                pv.rx_flag = rv.flag;
                pv.rx_seq = rv.sequence;
                ++xb_count[0];
            } else {
                peer = static_cast<const float*>(xf->recv());
            }
            wts4_d = reinterpret_cast<float*>(route_d + n * kTopE);
            pv.part0 = opt.rank == 0 ? local : peer;
            pv.part1 = opt.rank == 0 ? peer : local;
            pv.weights = wts4_d;
            xb_prepped = false;
            if (xb_prep && which == 1 && layer + 1 < xb_layers && linears_opt) {  // fold the next layer's q/k/v prepare
                const Layer& nx = layers[layer + 1];
                const mmoe::LinearJob j[3] = {{nx.rq, hn, kHidden, q, kQLocal}, {nx.rk, hn, kHidden, k, kKVLocal},
                                              {nx.rv, hn, kHidden, v, kKVLocal}};
                std::size_t offs[3];
                mmoe::linear_xt_offsets(j, 3, offs);
                pv.prep_jobs = 3;
                for (unsigned i = 0; i < 3; ++i) {
                    pv.prep_suh[i] = j[i].w.suh;
                    pv.prep_xt[i] = static_cast<std::uint16_t*>(lin_ws) + offs[i];
                }
                xb_prepped = true;
                ++xb_count[2];
            }
            enqueue_post(pv, nullptr);
            lap(prof.exchange);
            return;
        }
        dist::Tag tag;
        tag.kind = static_cast<std::uint32_t>(dist::TagKind::half_sum);
        tag.layer = layer;
        tag.request = calls * 2 + which;
        const bool staged = local != part;
        const float* peer_part = remote;
        if (staged) {
            comm.start_staged(nullptr, bytes, tag);
            peer_part = static_cast<const float*>(comm.finish_staged(nullptr));
        } else {
            comm.exchange(nullptr, part, remote, bytes, tag);
        }
        const float* p0 = opt.rank == 0 ? local : peer_part;
        const float* p1 = opt.rank == 0 ? peer_part : local;
        // ids then weights, contiguous for this call's n rows (one copy to the host)
        wts4_d = reinterpret_cast<float*>(route_d + n * kTopE);
        lap(prof.exchange);
        enqueue_post({static_cast<std::uint32_t>(n), h, p0, p1, norm_w, hn, router_w, bias, ids_d, wts4_d}, nullptr);
        if (staged) comm.release_staged(nullptr);
        lap(prof.post);
    }

    // FP32 identity: one grouped call (matrices of one size), job table through the pinned ring (reset after every stream
    // sync: layers_forward start, MoE route copies).
    void f32_grouped(std::vector<rtxpf::F32Job>& jobs, const float* x, std::uint32_t x_stride, const std::uint32_t* xmap, float* y,
                     std::uint32_t y_stride) {
        std::uint64_t total = 0;
        for (const auto& j : jobs) total += j.rows;
        const std::uint32_t tr = rtxpf::tile_rows_for(total, jobs.size());
        const auto ext = rtxpf::layout_jobs(jobs, tr);
        if (ext.entries == 0) return;
        if (f32_jobs_used + jobs.size() > kF32JobRing) throw std::runtime_error("fp32 identity: job ring overflow");
        rtxpf::F32Job* jh = f32_jobs_h + f32_jobs_used;
        rtxpf::F32Job* jd = f32_jobs_d + f32_jobs_used;
        std::copy(jobs.begin(), jobs.end(), jh);
        hip_ok(hipMemcpyAsync(jd, jh, jobs.size() * sizeof(rtxpf::F32Job), hipMemcpyHostToDevice, nullptr), "fp32 jobs");
        f32_jobs_used += static_cast<std::uint32_t>(jobs.size());
        rtxpf::GroupedCall c;
        c.jobs_d = jd;
        c.jobs = static_cast<std::uint32_t>(jobs.size());
        c.tiles = ext.tiles;
        c.tile_rows = tr;
        c.input = jobs[0].input;
        c.output = jobs[0].output;
        c.entries = ext.entries;
        c.x = x;
        c.x_stride = x_stride;
        c.xmap = xmap;
        c.y = y;
        c.y_stride = y_stride;
        rtxpf::enqueue_grouped(c, f32_xt, f32_raw, nullptr);
    }
    void f32_linear(const mmoe::Exl3& w, const float* x, std::uint32_t x_stride, float* y, std::uint32_t y_stride, std::size_t n) {
        std::vector<rtxpf::F32Job> j{rtxpf::job_of(w, static_cast<std::uint32_t>(n))};
        f32_grouped(j, x, x_stride, nullptr, y, y_stride);
    }
    void f32_mlp(const mmoe::ExpertSlices& m, const float* xin, std::size_t n, unsigned inter_local, float* out) {
        f32_linear(m.gate, xin, kHidden, big1, inter_local, n);
        f32_linear(m.up, xin, kHidden, big2, inter_local, n);
        rtxpf::enqueue_swiglu(big1, big2, big1, n * inter_local, nullptr);
        f32_linear(m.down, big1, inter_local, out, kHidden, n);
    }
    // FP32 identity MoE of one rank (routes already in route_h): routed pairs grouped per expert (ascending id, rows
    // ascending), SwiGLU, down, the shared expert on all rows, then the fixed-order combine into `out`.
    void f32_moe(const Layer& y, std::size_t n, float* out) {
        const std::uint32_t* ids = route_h;
        const float* w4 = reinterpret_cast<const float*>(route_h + n * kTopE);
        const std::size_t pairs = n * kTopE;
        std::array<std::uint32_t, kExperts + 1> first{};
        for (std::size_t i = 0; i < pairs; ++i) ++first[ids[i] + 1];
        for (unsigned e = 0; e < kExperts; ++e) first[e + 1] += first[e];
        std::array<std::uint32_t, kExperts> fill{};
        std::uint32_t* xmap = f32_map_h;
        std::uint32_t* slot = f32_map_h + pairs;
        float* wt = reinterpret_cast<float*>(f32_map_h + 2 * pairs);
        for (std::size_t t = 0; t < n; ++t) {
            std::array<std::pair<std::uint32_t, float>, kTopE> r;
            for (unsigned i = 0; i < kTopE; ++i) r[i] = {ids[t * kTopE + i], w4[t * kTopE + i]};
            std::sort(r.begin(), r.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            for (unsigned i = 0; i < kTopE; ++i) {
                const std::uint32_t at = first[r[i].first] + fill[r[i].first]++;
                xmap[at] = static_cast<std::uint32_t>(t);
                slot[t * kTopE + i] = at;
                wt[t * kTopE + i] = r[i].second;
            }
        }
        hip_ok(hipMemcpyAsync(f32_map_d, f32_map_h, 3 * pairs * 4, hipMemcpyHostToDevice, nullptr), "fp32 maps");
        std::vector<rtxpf::F32Job> jg, ju, jd;
        for (unsigned e = 0; e < kExperts; ++e) {
            const std::uint32_t m = first[e + 1] - first[e];
            if (m == 0) continue;
            jg.push_back(rtxpf::job_of(y.experts_raw[e].gate, m));
            ju.push_back(rtxpf::job_of(y.experts_raw[e].up, m));
            jd.push_back(rtxpf::job_of(y.experts_raw[e].down, m));
        }
        f32_grouped(jg, hn, kHidden, f32_map_d, f32_g, kExpertLocal);
        f32_grouped(ju, hn, kHidden, f32_map_d, f32_u, kExpertLocal);
        rtxpf::enqueue_swiglu(f32_g, f32_u, f32_g, pairs * kExpertLocal, nullptr);
        f32_grouped(jd, f32_g, kExpertLocal, nullptr, f32_ye, kHidden);
        f32_mlp(y.shared_raw, hn, n, kExpertLocal, xs);
        rtxpf::enqueue_combine(f32_ye, xs, f32_map_d + pairs, reinterpret_cast<const float*>(f32_map_d + 2 * pairs), static_cast<std::uint32_t>(n),
                               out, nullptr);
    }

    void run_mlp(const Mlp& m, const float* xin, std::size_t n, unsigned inter_local, float* out) {
        lib.mm(m.g, xin, big1, n, gemv);
        lib.mm(m.u, xin, big2, n, gemv);
        hipLaunchKernelGGL(swiglu_kernel, dim3(grid(n * inter_local)), dim3(256), 0, nullptr, big1, big2, big1, n * inter_local);
        lib.mm(m.d, big1, out, n, gemv);
    }

    // h (n rows) already holds the embeddings. `rows` describes each row's slot / position (decode), or is empty for a
    // contiguous chunk of `slot` at positions [pos0, pos0 + n).
    // Cache the slot's pointers across resumes. Positions/rows do not affect this table.
    void prefill_table(std::uint32_t slot) {
        if (pf_tab_slot == slot) return;
        for (unsigned L = 0; L < kLayers; ++L) {
            pf_tab_h[L * 3 + 0] = kc + cache_off(slot, L, kKVLocal);
            pf_tab_h[L * 3 + 1] = vc + cache_off(slot, L, kKVLocal);
            pf_tab_h[L * 3 + 2] = ikc + cache_off(slot, L, 128);
        }
        hip_ok(hipMemcpyAsync(pf_tab_d, pf_tab_h, std::size_t{kLayers} * 3 * sizeof(void*), hipMemcpyHostToDevice, nullptr), "prefill table");
        hip_ok(hipStreamSynchronize(nullptr), "prefill table");
        pf_tab_slot = slot;
    }

    void layers_forward(std::size_t n, std::span<const Row> rows, std::uint32_t slot, std::uint64_t pos0, unsigned n_layers) {
        layers_range(n, rows, slot, pos0, 0, n_layers);
    }
    // Layers [L0, L1) of the call; L0 > 0 resumes a prefill chunk paused after layer L0 - 1 (h / hn hold its rows).
    void layers_range(std::size_t n, std::span<const Row> rows, std::uint32_t slot, std::uint64_t pos0, unsigned L0, unsigned L1) {
        mark();
        if (rows.empty() && attn_prefill_fused) prefill_table(slot);
        if (L0 == 0) {
            tmark(0);
            hipLaunchKernelGGL(rmsnorm_gemma_kernel, dim3(static_cast<unsigned>(n)), dim3(256), 0, nullptr, h, w_in0, hn, kHidden);
            f32_jobs_used = 0;
        }
        xb_layers = L1;
        xb_prepped = false;
        for (unsigned L = L0; L < L1; ++L) {
            const Layer& y = layers[L];
            float* out = send_buffer(n);
            stage_attn(L, n, rows, slot, pos0, out);
            const bool own_router = moe_opt && gemv;  // enqueue_moe routes itself (same route_row bits)
            exchange_post(n, L, 0, out, y.w_post, own_router ? nullptr : y.wr, y.wb);
            tmark(tmark_at(L, 3));
            if (tracing() && y.moe && !own_router)  // the post-exchange router's ids (G4 chain)
                hip_ok(hipMemcpyAsync(trace_ids_d + std::size_t{L} * kMaxStepRows * kTopE, ids_d, n * kTopE * 4,
                                      hipMemcpyDeviceToDevice, nullptr), "trace ids");
            out = send_buffer(n);
            stage_ffn(L, n, out);
            exchange_post(n, L, 1, out, y.w_next, nullptr, nullptr);
            tmark(tmark_at(L, 5));
            if (taps_on) capture_tap(L, n);
        }
    }

    // Prefill q/k/v on the v4 core: measured faster than three strix_gemm calls from 1024 rows (bench pfdv4 q+k+v
    // T1024 0.78, T4096 0.70; o alone is slower at 1024 rows and stays on the library).
    static constexpr std::size_t kDenseQkvMinRows = 1024;
    bool dense_qkv(std::size_t n, const mmoe::PrefillDenseJob (&jobs)[3]) const {
        if (!pf_dense_qkv || n < kDenseQkvMinRows || pf_ws == nullptr) return false;
        for (const auto& j : jobs)  // raw weights loaded, same input width, implemented bit width; else the library
            if (j.w.trellis == nullptr || j.w.input != jobs[0].w.input || !mmoe::prefill_dense_supported(j.w)) return false;
        return mmoe::prefill_dense_workspace_bytes(jobs, 3, static_cast<unsigned>(n)) <= pf_ws_bytes;
    }

    // Prefill o_proj on the v4 core only for exactly 4096-row pieces (bench pfdv4 o: 0.96 at 4096 rows, 1.10 at 1024 rows;
    // larger --max-rows pieces are not qualified).
    static constexpr std::size_t kDenseORows = 4096;
    bool dense_o(std::size_t n, const mmoe::PrefillDenseJob (&job)[1]) const {
        return pf_dense_o && n == kDenseORows && pf_ws != nullptr && job[0].w.trellis != nullptr &&
               mmoe::prefill_dense_supported(job[0].w) &&
               mmoe::prefill_dense_workspace_bytes(job, 1, static_cast<unsigned>(n)) <= pf_ws_bytes;
    }

    // Layer L up to the attention partial: q/k/v + index projections, MSA, o_proj -> out (n rows; h / hn: the rows').
    void stage_attn(unsigned L, std::size_t n, std::span<const Row> rows, std::uint32_t slot, std::uint64_t pos0, float* out) {
        if (tl) tl->context(task.c + pf_idx_member, L);
        {
            const bool f32 = fp32_prefill && !gemv;  // FP32 prefill identity
            const Layer& y = layers[L];
            const bool lin_opt = gemv && linears_opt;  // decode linears on enqueue_linears (strix_gemv bits)
            const bool conc = index_attn && (gemv ? index_conc && !f32 : pf_idx_conc != 0);  // same kernels and bits, side stream
            const strix::minimax::attn::IndexProjView ipv{hn, y.wip, iq, ik, static_cast<std::uint32_t>(n)};
            const std::size_t ie = pf_idx_member * kLayers + L;
            // hn belongs to this chunk; q/k/v only read it. Join below precedes
            // prep (the first iq/ik reader), every hn write and the next chunk.
            if (conc) strix::minimax::attn::index_proj_fork(ipv, nullptr, idx_side,
                gemv ? idx_fork[L] : pf_idx_conc == 2 ? pf_idx_fork_device[ie] : pf_idx_fork[ie]);
            if (tl && lin_opt) tl->mark("qkv-projections", true);
            if (f32) {
                if (tl) tl->mark("q-projection", true);
                f32_linear(y.rq, hn, kHidden, q, kQLocal, n);
                if (tl) tl->mark("q-projection", false);
                if (tl) tl->mark("k-projection", true);
                f32_linear(y.rk, hn, kHidden, k, kKVLocal, n);
                if (tl) tl->mark("k-projection", false);
                if (tl) tl->mark("v-projection", true);
                f32_linear(y.rv, hn, kHidden, v, kKVLocal, n);
                if (tl) tl->mark("v-projection", false);
            } else if (lin_opt) {
                const mmoe::LinearJob j[3] = {{y.rq, hn, kHidden, q, kQLocal}, {y.rk, hn, kHidden, k, kKVLocal},
                                              {y.rv, hn, kHidden, v, kKVLocal}};
                if (xb_prepped) mmoe::enqueue_linears_prepared(j, 3, static_cast<unsigned>(n), lin_ws, nullptr);  // STRIX_MM_XB_PREP
                else mmoe::enqueue_linears(j, 3, static_cast<unsigned>(n), lin_ws, nullptr);
                xb_prepped = false;
            } else if (const mmoe::PrefillDenseJob qkv[3] = {{y.rq, q}, {y.rk, k}, {y.rv, v}}; dense_qkv(n, qkv)) {
                // STRIX_MM_PF_DENSE_QKV: one v4-core call, one timeline interval (compare with q + k + v of the library).
                if (tl) tl->mark("qkv-projections", true);
                mmoe::enqueue_prefill_dense(qkv, 3, hn, static_cast<unsigned>(n), pf_ws, pf_ws_bytes, nullptr);
                if (tl) tl->mark("qkv-projections", false);
            } else {
                if (tl) tl->mark("q-projection", true);
                lib.mm(y.q, hn, q, n, gemv);
                if (tl) tl->mark("q-projection", false);
                if (tl) tl->mark("k-projection", true);
                const bool tile_override = !gemv && n > 8 && pf_kv_tile != -1 && kv_tile_available();
                const int cfg = route::kv_tile_config(n, pf_kv_tile);
                const bool receipt = tile_override && pf_kv_receipts.emplace(n, cfg).second;
                int kt = -1, vt = -1;
                // Each process-wide override covers only one K/V submission, including
                // exception paths. All other projections retain library policy.
                auto kv_mm = [&](const Lin& weights, float* dst, int& used) {
                    struct ResetTile {
                        bool active;
                        ~ResetTile() { if (active) strix_trellis_lab_set_tile_override(-1); }
                    } reset{tile_override};
                    if (tile_override) strix_trellis_lab_set_tile_override(cfg);
                    lib.mm(weights, hn, dst, n, gemv);
                    if (receipt && strix_trellis_lab_last_tile) used = strix_trellis_lab_last_tile();
                };
                kv_mm(y.k, k, kt);
                if (tl) tl->mark("k-projection", false);
                if (tl) tl->mark("v-projection", true);
                kv_mm(y.v, v, vt);
                if (tl) tl->mark("v-projection", false);
                if (receipt) std::printf("rank %u pfkvtile rows=%zu requested=%d used K=%d V=%d%s\n",
                    opt.rank, n, cfg, kt, vt, strix_trellis_lab_last_tile ? "" : " (receipt symbol absent)");
            }
            if (tl && lin_opt) tl->mark("qkv-projections", false);
            if (tl) tl->mark(conc ? "index-join" : "index-projection", true);
            if (conc) {
                strix::minimax::attn::index_proj_join(nullptr, idx_side,
                    gemv ? idx_join[L] : pf_idx_conc == 2 ? pf_idx_join_device[ie] : pf_idx_join[ie]);
            } else if (index_attn) {  // The one-launch projection (decode register ring / prefill tiles), same bits
                strix::minimax::attn::enqueue_index_proj(ipv, nullptr);
            } else if (index_tiled && n <= 8 && index_decode) {
                hipLaunchKernelGGL(index_proj_decode_kernel, dim3(kIdxLocal / kDecOut + 128 / kDecOut), dim3(256), 0, nullptr, hn, y.wiq16b, iq,
                                   kIdxLocal, y.wik16b, ik, 128U, kHidden, static_cast<unsigned>(n));
            } else if (index_tiled && n <= 8) {
                const auto nn = static_cast<unsigned>(n);
                hipLaunchKernelGGL(dense_t16_kernel<8>, dim3(1, kIdxLocal / 128), dim3(128), 0, nullptr, hn, y.wiq16, iq, kHidden, kIdxLocal, nn);
                hipLaunchKernelGGL(dense_t16_kernel<8>, dim3(1, 1), dim3(128), 0, nullptr, hn, y.wik16, ik, kHidden, 128U, nn);
            } else if (index_tiled) {
                const auto nn = static_cast<unsigned>(n), gx = (nn + 15) / 16;
                hipLaunchKernelGGL(dense_t16_kernel<16>, dim3(gx, kIdxLocal / 128), dim3(128), 0, nullptr, hn, y.wiq16, iq, kHidden, kIdxLocal, nn);
                hipLaunchKernelGGL(dense_t16_kernel<16>, dim3(gx, 1), dim3(128), 0, nullptr, hn, y.wik16, ik, kHidden, 128U, nn);
            } else {
                hipLaunchKernelGGL(dense_t_kernel, dim3(static_cast<unsigned>(n), kIdxLocal / 128), dim3(128), 0, nullptr, hn, y.wiq, iq, kHidden, kIdxLocal);
                hipLaunchKernelGGL(dense_t_kernel, dim3(static_cast<unsigned>(n), 1), dim3(128), 0, nullptr, hn, y.wik, ik, kHidden, 128U);
            }
            if (tl) tl->mark(conc ? "index-join" : "index-projection", false);
            lap(prof.attn_proj);
            tmark(tmark_at(L, 0));
            if (lin_opt && xb_warm_o_mb != 0 && use_xfer() && !pf_active) xb_warm_o(L);  // STRIX_MM_XB_WARM_O
            const bool traced_batch = rows.empty() && attn_prefill_fused &&
                (pf_msa_batch || (pf_idx_tile && n > 8) || (pf_msa_rework != 0 && n > 8));
            if (tl && !traced_batch) tl->mark("msa", true);
            if (rows.empty() && attn_prefill_fused) {  // one sequence, windows of <= 16 consecutive rows (causal)
                std::uint16_t* const* tab = pf_tab_d + std::size_t{L} * 3;
                for (std::size_t r0 = 0; r0 < n; r0 += 16) {
                    const auto m = static_cast<std::uint32_t>(std::min<std::size_t>(16, n - r0));
                    mfused::DecodeView dv{};
                    dv.seqs = 1;
                    dv.rows_per_seq = m;
                    dv.q_heads = 32;
                    dv.kv_heads = 2;
                    dv.idx_heads = 2;
                    dv.max_blocks = static_cast<std::uint32_t>(msa::block_count(opt.capacity));
                    dv.q = q + r0 * kQLocal;
                    dv.k = k + r0 * kKVLocal;
                    dv.v = v + r0 * kKVLocal;
                    dv.iq = iq + r0 * kIdxLocal;
                    dv.ik = ik + r0 * 128;
                    dv.q_w = y.norm[0];
                    dv.k_w = y.norm[1];
                    dv.iq_w = y.norm[2];
                    dv.ik_w = y.norm[3];
                    dv.rope = d_rope + pos0 + r0;
                    dv.pos = d_pos + pos0 + r0;
                    dv.k_cache = tab;
                    dv.v_cache = tab + 1;
                    dv.ik_cache = tab + 2;
                    dv.scores = scores;
                    dv.idx = idx;
                    dv.idx_count = cnt;
                    dv.partial = partial;
                    dv.ticket = ticket;
                    dv.o = attn + r0 * kQLocal;
                    dv.dense = pos0 + r0 + m - 1 < msa::kTopK * msa::kBlock;
                    if (pf_attend4) {
                        dv.attend_version = 4;
                        dv.heads_per_wave = 2;
                    }
                    if (pf_msa_batch || (pf_idx_tile && n > 8) || (pf_msa_rework != 0 && n > 8)) {
                        dv.scores = pf_scores;
                        dv.idx = pf_idx;
                        dv.idx_count = pf_cnt;
                        dv.partial = pf_partial;
                        dv.ticket = pf_ticket;
                        mfused::enqueue_prefill({dv, static_cast<std::uint32_t>(n), pos0, pf_msa_windows, 0, pf_idx_tile && n > 8, pf_idx_rows, n > 8 ? pf_msa_rework : 0U}, nullptr, tl ? PrefillTimeline::msa_mark : nullptr, tl);
                        break;
                    }
                    mfused::enqueue_decode(dv, nullptr);
                }
            } else if (rows.empty()) {
                std::uint16_t *kcl = kc + cache_off(slot, L, kKVLocal), *vcl = vc + cache_off(slot, L, kKVLocal),
                              *ikl = ikc + cache_off(slot, L, 128);
                const auto max_blocks = static_cast<std::uint32_t>(msa::block_count(pos0 + n));
                mh::PrepView pv{static_cast<std::uint32_t>(n), 32, 2, 2, q, k, v, iq, ik, y.norm[0], y.norm[1], y.norm[2], y.norm[3],
                                d_rope + pos0, d_pos + pos0, kcl, vcl, ikl};
                mh::enqueue_prep(pv, nullptr);
                for (std::size_t r0 = 0; r0 < n; r0 += 8) {
                    const auto m = static_cast<std::uint32_t>(std::min<std::size_t>(8, n - r0));
                    const std::uint64_t* pp = d_pos + pos0 + r0;
                    mh::enqueue_scores({m, 2, max_blocks, iq + r0 * kIdxLocal, ikl, pp, scores}, nullptr);
                    mh::enqueue_select({m, 2, max_blocks, pp, scores, idx, cnt}, nullptr);
                    mh::enqueue_attention({m, 2, q + r0 * kQLocal, kcl, vcl, pp, idx, cnt, partial, attn + r0 * kQLocal}, nullptr);
                }
            } else if (attn_fused && step_seqs != 0) {  // all rows in two launches (one dense launch below 2048 positions)
                std::uint16_t* const* tab = cache_tab_d + std::size_t{L} * 3 * kMaxStepRows;
                const bool dense = route::step_dense(rows, msa::kTopK * msa::kBlock);
                mfused::DecodeView dv{};
                dv.seqs = step_seqs;
                dv.rows_per_seq = step_rows_per_seq;
                dv.q_heads = 32;
                dv.kv_heads = 2;
                dv.idx_heads = 2;
                dv.max_blocks = static_cast<std::uint32_t>(msa::block_count(opt.capacity));
                dv.q = q;
                dv.k = k;
                dv.v = v;
                dv.iq = iq;
                dv.ik = ik;
                dv.q_w = y.norm[0];
                dv.k_w = y.norm[1];
                dv.iq_w = y.norm[2];
                dv.ik_w = y.norm[3];
                dv.rope = step_rope;
                dv.pos = step_pos;
                dv.k_cache = tab;
                dv.v_cache = tab + kMaxStepRows;
                dv.ik_cache = tab + 2 * kMaxStepRows;
                dv.scores = scores;
                dv.idx = idx;
                dv.idx_count = cnt;
                dv.partial = partial;
                dv.ticket = ticket;
                dv.o = attn;
                dv.dense = dense;
                std::uint64_t min_pos = ~std::uint64_t{0};
                for (const Row& w : rows) min_pos = std::min<std::uint64_t>(min_pos, w.pos);
                // Measured winning grids only (two single-row sparse sequences at >= 8K): same bits, scheduling/LDS.
                // The v3 32 KB LDS diet first (same-binary bench, 8K/32K/128K us v3 74.4/133.2/343.7, diet 69.9/130.1/340.3,
                // v4 one head per wave 71.0/130.9/341.6); v4 one head per wave when the diet is off.
                if (route::attend_n2_sparse(step_seqs, step_rows_per_seq, dense, min_pos)) {
                    if (attend_diet) {
                        dv.lds_diet = true;
                    } else if (attend4_decode) {
                        dv.attend_version = 4;
                        dv.heads_per_wave = 1;
                    }
                }
                dv.select_once = route::select_once(select_once, static_cast<unsigned>(rows.size()), dense, min_pos);
                if (dv.select_once) ++select_once_calls;
                mfused::enqueue_decode(dv, nullptr);
            } else {
                for (std::size_t r = 0; r < n; ++r) {  // each row in its own slot: one-row K1-K4 on that slot's caches
                    const Row& w = rows[r];
                    std::uint16_t *kcl = kc + cache_off(w.slot, L, kKVLocal), *vcl = vc + cache_off(w.slot, L, kKVLocal),
                                  *ikl = ikc + cache_off(w.slot, L, 128);
                    const auto max_blocks = static_cast<std::uint32_t>(msa::block_count(w.pos + 1));
                    const std::uint64_t* pp = d_pos + w.pos;
                    mh::PrepView pv{1, 32, 2, 2, q + r * kQLocal, k + r * kKVLocal, v + r * kKVLocal, iq + r * kIdxLocal, ik + r * 128,
                                    y.norm[0], y.norm[1], y.norm[2], y.norm[3], d_rope + w.pos, pp, kcl, vcl, ikl};
                    mh::enqueue_prep(pv, nullptr);
                    mh::enqueue_scores({1, 2, max_blocks, iq + r * kIdxLocal, ikl, pp, scores}, nullptr);
                    mh::enqueue_select({1, 2, max_blocks, pp, scores, idx, cnt}, nullptr);
                    mh::enqueue_attention({1, 2, q + r * kQLocal, kcl, vcl, pp, idx, cnt, partial, attn + r * kQLocal}, nullptr);
                }
            }
            if (tl && !traced_batch) tl->mark("msa", false);
            lap(prof.msa);
            tmark(tmark_at(L, 1));
            if (tl) tl->mark("o-projection", true);
            if (f32) {
                f32_linear(y.ro, attn, kQLocal, out, kHidden, n);
            } else if (lin_opt) {
                const mmoe::LinearJob j{y.ro, attn, kQLocal, out, kHidden};
                if (xb_pt.flag != nullptr && out == xb_tx) {  // STRIX_MM_XB_PUBLISH: o_proj publishes its TX message
                    mmoe::enqueue_linears_publish(&j, 1, static_cast<unsigned>(n), lin_ws, nullptr, xb_pt, false);
                    xb_published = true;
                } else {
                    mmoe::enqueue_linears(&j, 1, static_cast<unsigned>(n), lin_ws, nullptr);
                }
            } else if (const mmoe::PrefillDenseJob o1[1] = {{y.ro, out}}; dense_o(n, o1)) {
                mmoe::enqueue_prefill_dense(o1, 1, attn, static_cast<unsigned>(n), pf_ws, pf_ws_bytes, nullptr);  // STRIX_MM_PF_DENSE_O
            } else {
                lib.mm(y.o, attn, out, n, gemv);
            }
            if (tl) tl->mark("o-projection", false);
            lap(prof.o_proj);
            tmark(tmark_at(L, 2));
        }
    }

    // Layer L's FFN on hn (after the attention exchange): dense MLP or router-routed experts + shared -> out.
    void stage_ffn(unsigned L, std::size_t n, float* out) {
        {
            const bool f32 = fp32_prefill && !gemv;  // FP32 prefill identity
            const Layer& y = layers[L];
            const bool lin_opt = gemv && linears_opt;
            const bool moe_marks = y.moe && moe_opt && !gemv && !f32;
            if (tl && !moe_marks) tl->mark(y.moe ? "moe" : "dense-mlp", true);
            if (!y.moe) {
                if (f32) {
                    f32_mlp(y.rdense, hn, n, kDenseLocal, out);
                } else if (lin_opt) {
                    const mmoe::LinearJob gu[2] = {{y.rdense.gate, hn, kHidden, big1, kDenseLocal},
                                                   {y.rdense.up, hn, kHidden, big2, kDenseLocal}};
                    mmoe::enqueue_linears(gu, 2, static_cast<unsigned>(n), lin_ws, nullptr);
                    hipLaunchKernelGGL(swiglu_kernel, dim3(grid(n * kDenseLocal)), dim3(256), 0, nullptr, big1, big2, big1, n * kDenseLocal);
                    const mmoe::LinearJob dn{y.rdense.down, big1, kDenseLocal, out, kHidden};
                    if (xb_pt.flag != nullptr && out == xb_tx) {  // STRIX_MM_XB_PUBLISH
                        mmoe::enqueue_linears_publish(&dn, 1, static_cast<unsigned>(n), lin_ws, nullptr, xb_pt, false);
                        xb_published = true;
                    } else {
                        mmoe::enqueue_linears(&dn, 1, static_cast<unsigned>(n), lin_ws, nullptr);
                    }
                } else {
                    run_mlp(y.dense, hn, n, kDenseLocal, out);
                }
            } else if (moe_opt && gemv) {  // decode: router + experts + shared + combine, 4 launches, no host sync
                mmoe::MoeView mv;
                mv.rows = static_cast<std::uint32_t>(n);
                mv.y = hn;
                mv.router_w = y.wr;
                mv.bias = y.wb;
                mv.experts = y.experts_raw_d;
                mv.shared = y.shared_raw;
                mv.out = out;
                mv.ws = moe_ws;
                if (tracing()) mv.ids = trace_ids_d + std::size_t{L} * kMaxStepRows * kTopE;  // optional output only
                if (xb_pt.flag != nullptr && out == xb_tx) {  // STRIX_MM_XB_PUBLISH: the down/combine launch publishes
                    mv.pub = xb_pt;
                    xb_published = true;
                }
                mmoe::enqueue_moe(mv, nullptr);
            } else if (f32) {  // prefill in the FP32 identity: host routes, grouped experts + shared, fixed-order combine
                hip_ok(hipMemcpyAsync(route_h, route_d, n * kTopE * 8, hipMemcpyDeviceToHost, nullptr), "routes");
                hip_ok(hipStreamSynchronize(nullptr), "routes");
                f32_jobs_used = 0;
                lap(prof.route_host);
                f32_moe(y, n, out);
            } else if (moe_opt) {  // prefill / gate: grouped launches with the G4 strix_gemm identity
                hip_ok(hipMemcpyAsync(route_h, route_d, n * kTopE * 8, hipMemcpyDeviceToHost, nullptr), "routes");
                hip_ok(hipStreamSynchronize(nullptr), "routes");
                const auto table_host_start = pf_table_stats ? Clock::now() : Clock::time_point{};
                if (pf_active) pf->check();  // a failed exchange released the stream with garbage: never plan from it
                lap(prof.route_host);
                if (profiling) {
                    std::array<std::uint32_t, kExperts> cnt_e{};
                    for (std::size_t i = 0; i < n * kTopE; ++i) ++cnt_e[route_h[i] % kExperts];
                    for (const auto c : cnt_e)
                        ++prof.rows_per_expert[c == 0 ? 0U : c <= 8 ? 1U : c <= 16 ? 2U : c <= 32 ? 3U : c <= 64 ? 4U : c <= 128 ? 5U : c <= 256 ? 6U : 7U];
                }
                mmoe::PrefillMoeView pv;
                pv.rows = static_cast<std::uint32_t>(n);
                pv.y = hn;
                pv.ids = route_h;
                pv.weights = reinterpret_cast<const float*>(route_h + n * kTopE);
                pv.experts = y.experts_raw.data();
                pv.shared = y.shared_raw;
                pv.out = out;
                // The module markers are host callbacks: record one interval
                // from its first launch to combine, excluding host planning.
                struct TimelineMoe {
                    bool on;
                    ~TimelineMoe() { if (on) mmoe::set_prefill_marker(nullptr, nullptr); }
                } timeline_moe{tl != nullptr};
                if (tl) mmoe::set_prefill_marker(PrefillTimeline::moe_mark, tl);
                mmoe::enqueue_prefill_moe(pv, pf_ws, pf_ws_bytes, nullptr);
                if (pf_table_stats) {
                    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - table_host_start).count();
                    ++pf_table_calls;
                    pf_table_host_ms += ms;
                    std::printf("rank %u pf-table-host: call %llu rows %zu route-sync-to-last-launch %.3f ms total %.3f ms\n",
                                opt.rank, static_cast<unsigned long long>(pf_table_calls), n, ms, pf_table_host_ms);
                }
            } else {
                if (shared_first) {  // the shared expert runs while the host waits for the routes and packs them
                    hip_ok(hipEventRecord(routes_ready, nullptr), "routes event");
                    run_mlp(y.shared, hn, n, kExpertLocal, xs);
                    hip_ok(hipStreamWaitEvent(route_stream, routes_ready, 0), "routes wait");
                    hip_ok(hipMemcpyAsync(route_h, route_d, n * kTopE * 8, hipMemcpyDeviceToHost, route_stream), "routes");
                    hip_ok(hipStreamSynchronize(route_stream), "routes");
                } else {
                    hip_ok(hipMemcpyAsync(route_h, route_d, n * kTopE * 8, hipMemcpyDeviceToHost, nullptr), "routes");
                    hip_ok(hipStreamSynchronize(nullptr), "routes");
                }
                if (pf_active) pf->check();  // never plan from routes of a failed exchange
                const std::uint32_t* ids = route_h;
                const std::uint32_t* w4 = route_h + n * kTopE;  // float bits
                std::array<std::uint32_t, kExperts + 1> first{};
                for (std::size_t i = 0; i < n * kTopE; ++i) ++first[ids[i] + 1];
                for (unsigned e = 0; e < kExperts; ++e) first[e + 1] += first[e];
                std::array<std::uint32_t, kExperts> fill = {};
                for (std::size_t t = 0; t < n; ++t)  // rows of each expert in ascending row order, as G4
                    for (unsigned i = 0; i < kTopE; ++i) {
                        const std::uint32_t e = ids[t * kTopE + i], at = first[e] + fill[e]++;
                        pack_h[at] = static_cast<std::uint32_t>(t);
                        pack_h[n * kTopE + at] = w4[t * kTopE + i];
                    }
                hip_ok(hipMemcpyAsync(pack_d, pack_h, n * kTopE * 8, hipMemcpyHostToDevice, nullptr), "route pack");
                lap(prof.route_host);
                hip_ok(hipMemsetAsync(out, 0, n * kHidden * 4, nullptr), "zero part");
                for (unsigned e = 0; e < kExperts; ++e) {  // ascending expert id per token (batch invariance)
                    const std::size_t m = first[e + 1] - first[e];
                    if (m == 0) continue;
                    const std::uint32_t* rows_e = pack_d + first[e];
                    const float* wts_e = reinterpret_cast<const float*>(pack_d + n * kTopE + first[e]);
                    hipLaunchKernelGGL(gather_kernel, dim3(static_cast<unsigned>(m)), dim3(256), 0, nullptr, hn, rows_e, xe, kHidden);
                    run_mlp(y.experts[e], xe, m, kExpertLocal, ye);
                    hipLaunchKernelGGL(scatter_add_kernel, dim3(static_cast<unsigned>(m)), dim3(256), 0, nullptr, ye, rows_e, wts_e, out, kHidden);
                }
                if (!shared_first) run_mlp(y.shared, hn, n, kExpertLocal, xs);
                hipLaunchKernelGGL(add_kernel, dim3(grid(n * kHidden)), dim3(256), 0, nullptr, out, xs, n * kHidden);
            }
            if (tl && !moe_marks) tl->mark(y.moe ? "moe" : "dense-mlp", false);
            lap(y.moe ? prof.ffn : prof.dense);
            tmark(tmark_at(L, 4));
        }
    }

    // h holds the residual after layer L (updated in place by the post-exchange kernel).
    void capture_tap(unsigned L, std::size_t n) {
        static constexpr unsigned kTapLayers[5] = {3, 17, 31, 45, 59};
        for (unsigned t = 0; t < 5; ++t)
            if (kTapLayers[t] == L)
                hip_ok(hipMemcpy2DAsync(taps_d + static_cast<std::size_t>(t) * kHidden, 5 * kHidden * sizeof(float), h, kHidden * sizeof(float),
                                        kHidden * sizeof(float), n, hipMemcpyDeviceToDevice, nullptr), "tap");
    }

    void embed_rows(const std::uint32_t* toks, std::size_t n) {
        mark();
        hip_ok(hipMemcpy(tok_d, toks, n * 4, hipMemcpyHostToDevice), "tokens");
        hipLaunchKernelGGL(embed_kernel, dim3(static_cast<unsigned>(n)), dim3(256), 0, nullptr, embed, tok_d, h, kHidden);
        lap(prof.embed);
    }

    // Head on `n` rows of hn starting at `first` (vocabulary half), then the global argmax of each row: one exchange of
    // (max, id) per row; ties go to the lower id (rank 0's half).
    std::vector<std::uint32_t> head_argmax(const float* rows_in, std::size_t n) {
        mark();
        lib.mm(head, rows_in, logits, n, true);
        std::vector<float> lg(n * vn);
        hip_ok(hipMemcpy(lg.data(), logits, lg.size() * 4, hipMemcpyDeviceToHost), "logits");
        std::vector<std::uint32_t> mine(2 * n), theirs(2 * n);
        for (std::size_t r = 0; r < n; ++r) {
            const float* l = lg.data() + r * vn;
            std::uint32_t best = 0;
            bool finite = true;
            for (std::uint32_t i = 0; i < vn; ++i) {
                finite &= std::isfinite(l[i]);
                if (l[i] > l[best]) best = i;
            }
            if (!finite) throw std::runtime_error("non-finite logits");
            std::memcpy(&mine[2 * r], &l[best], 4);
            mine[2 * r + 1] = best + (opt.rank == 0 ? 0U : static_cast<std::uint32_t>(kVocabPad));
        }
        hip_ok(hipMemcpy(am_local, mine.data(), n * 8, hipMemcpyHostToDevice), "am");
        dist::Tag tag;
        tag.kind = static_cast<std::uint32_t>(dist::TagKind::argmax);
        tag.layer = kLayers;
        tag.request = calls;
        comm.exchange(nullptr, am_local, am_remote, n * 8, tag);
        hip_ok(hipMemcpy(theirs.data(), am_remote, n * 8, hipMemcpyDeviceToHost), "am r");
        std::vector<std::uint32_t> out(n);
        for (std::size_t r = 0; r < n; ++r) {
            const std::uint32_t* r0 = opt.rank == 0 ? &mine[2 * r] : &theirs[2 * r];
            const std::uint32_t* r1 = opt.rank == 0 ? &theirs[2 * r] : &mine[2 * r];
            float v0, v1;
            std::memcpy(&v0, &r0[0], 4);
            std::memcpy(&v1, &r1[0], 4);
            out[r] = v1 > v0 ? r1[1] : r0[1];
        }
        lap(prof.head);
        return out;
    }

    // Resumable prefill (serve scheduler, qualification notes): prefill()'s walk over the chunks / groups as a
    // state machine that can pause between two layers. The resumed call makes the same kernel calls with the same rows
    // and arguments as the uninterrupted one (prefill() = begin + one resume of every layer). A pause happens with no
    // exchange in flight; step() calls of OTHER slots may run in between, and they write rows [0, 8) of h / hn (and of
    // the taps), so those rows of the open chunk are saved at the pause and restored at the resume.
    struct PfTask {
        bool open = false, grouped = false, saved = false;
        std::uint32_t slot = 0, next = 0;
        std::vector<std::uint32_t> toks;
        std::uint64_t pos0 = 0;
        std::size_t c0 = 0, c = 0, chunks = 0, m = 1, rows = 0;  // next chunk's first row / index; the open unit's members, rows
        unsigned layer = 0;                                      // next layer of the open chunk / group (0: not started)
        std::uint64_t output_fnv = 14695981039346656037ULL;
        Engine::PfTaskId id = 0, parent = 0;
        std::uint64_t generation = 0;
    } task, parked;
    struct ActivationBank { float *h = nullptr, *hn = nullptr, *taps = nullptr; } base_bank, nest_bank;
    bool child_bank = false;
    unsigned nest_rows = 0;
    std::string nest_reason = "not_reserved";
    Engine::PrefillTapSink parked_sink;
    std::unique_ptr<PrefillTimeline> child_timeline;

    bool check_prompt_outputs = false;
    float *sav_h = nullptr, *sav_hn = nullptr, *sav_taps = nullptr;  // rows [0, kMaxStepRows) of h, hn, taps

    void no_task(const char* what) {
        if (task.open || parked.open) throw std::logic_error(std::string(what) + ": a resumable prefill is open");
        // A completed orphan child keeps its result bank until the next operation consumes that lifetime.
        if (child_bank) { retire_prefill(); bind_bank(false); }
    }

    void prefill_begin(std::uint32_t slot, std::span<const std::uint32_t> toks, std::uint64_t pos0) {
        if (task.open) throw std::logic_error("prefill: active task");
        if (slot >= opt.slots || toks.empty() || pos0 + toks.size() > opt.capacity) throw std::invalid_argument("prefill: bad slot / range");
        task = PfTask{};
        last_out = {};
        last_result = {};
        task.slot = slot;
        task.toks.assign(toks.begin(), toks.end());
        task.pos0 = pos0;
        task.chunks = (toks.size() + opt.max_rows - 1) / opt.max_rows;
        const bool module = pf != nullptr && pf_on && !pf_force_comm;
        task.grouped = pf_interleave && (module || pf_comm_interleave) && !profiling && !(taps_on && tap_sink) && task.chunks >= 2;
        task.m = task.grouped ? std::min<std::size_t>(pf_group, task.chunks) : 1;  // the first unit (as prefill_resume sets it)
        task.rows = route::unit_rows(task.m, opt.max_rows, toks.size());
        task.open = true;
        if (auto* t = child_bank ? child_timeline.get() : timeline.get()) t->begin();
    }

    // Runs at most `layers` more layers of the open prefill (a chunk's tap sink and the head run in the unit of its last
    // layer). Returns true with *next = the greedy token after the last row once the prefill is complete.
    bool prefill_resume(unsigned budget, std::uint32_t* next) {
        if (!task.open) throw std::logic_error("prefill_resume: no open prefill");
        if (budget == 0) throw std::invalid_argument("prefill_resume: 0 layers");
        if (profiling || fp32_prefill) budget = ~0U;  // synchronised laps / the FP32 job ring: one resume runs it all
        gemv = false;
        tl = child_bank ? child_timeline.get() : timeline.get();
        // The prefill module carries this call's layer exchanges; it is drained before the head's argmax exchange
        // (communicator, same transport), at a pause and on every exit.
        struct Active {
            Impl& m;
            ~Active() {
                if (m.pf_active) {
                    m.pf_active = false;
                    try {
                        m.pf->drain();
                    } catch (...) {
                    }
                }
            }
        } active{*this};
        pf_active = pf != nullptr && pf_on;
        if (pf_force_comm) pf_active = false;
        try {
            const std::span<const std::uint32_t> toks(task.toks);
            if (task.saved) restore_rows();
            unsigned left = budget;
            while (task.c0 < toks.size()) {
                if (task.layer == 0) {
                    task.m = task.grouped ? std::min<std::size_t>(pf_group, task.chunks - task.c) : 1;
                    task.rows = route::unit_rows(task.m, opt.max_rows, toks.size() - task.c0);
                }
                if (left == 0) break;
                const unsigned L0 = task.layer, L1 = route::layer_end(L0, left, kLayers);
                left -= L1 - L0;
                const std::size_t c0 = task.c0, rows = task.rows, m = task.m;
                const bool last = c0 + rows == toks.size();
                if (m >= 2) {
                    const float* last_hn = prefill_group(task.slot, toks.subspan(c0, rows), task.pos0 + c0, L0, L1);
                    if (L1 < kLayers) {
                        task.layer = L1;
                        continue;
                    }
                    if (check_prompt_outputs) {
                        for (std::size_t member = 0; member < m; ++member)
                            check_output(member == 0 ? hn : grp_hn[member - 1],
                                std::min<std::size_t>(opt.max_rows, rows - member * opt.max_rows));
                    }
                    calls += m - 1;  // the head's tag is the last chunk's, as in the chunk-serial order
                    if (last) {
                        if (pf_active) pf->drain();
                        const std::size_t n_last = rows - (m - 1) * opt.max_rows;
                        last_out = {last_hn, n_last};
                        task.next = head_argmax(last_hn + (n_last - 1) * kHidden, 1)[0];
                    }
                    ++calls;
                    task.c += m;
                } else {
                    if (L0 == 0) embed_rows(toks.data() + c0, rows);
                    layers_range(rows, {}, task.slot, task.pos0 + c0, L0, L1);
                    if (L1 < kLayers) {
                        task.layer = L1;
                        continue;
                    }
                    if (check_prompt_outputs) check_output(hn, rows);
                    if (taps_on && tap_sink) tap_sink(taps_d, rows, task.slot, task.pos0 + c0);
                    if (last) {
                        if (pf_active) pf->drain();
                        last_out = {hn, rows};
                        task.next = head_argmax(hn + (rows - 1) * kHidden, 1)[0];
                    }
                    ++calls;
                    ++task.c;
                }
                task.c0 += rows;
                task.layer = 0;
            }
            if (task.c0 == toks.size()) {
                if (pf_active) pf->drain();
                pf_active = false;
                hip_ok(hipDeviceSynchronize(), "prefill");
                if (tl) tl->report(opt.rank);
                tl = nullptr;
                ++prof.calls;
                task.open = false;
                last_result = {task.id, task.c0 - last_out.second, last_out.second,
                               child_bank ? 1U : 0U, static_cast<std::uint32_t>(task.m - 1), task.next};
                if (next != nullptr) *next = task.next;
                return true;
            }
            if (task.layer > 0) save_rows();
            if (pf_active) pf->drain();
            pf_active = false;
            tl = nullptr;
            return false;
        } catch (...) {
            comm.cancel_pending("prefill failure");
            pf_comm_inflight = false;
            (void)hipDeviceSynchronize(); // joins above close host submission before device retirement
            tl = nullptr;
            task.open = false;
            task.saved = false;
            throw;
        }
    }

    void prefill_cancel() {
        retire_prefill();
        tl = nullptr;
        task.open = false;
        task.saved = false;
    }

    // No scratch, route reader, side stream or final post may survive a switch.
    // Transport counters and table-ring cursors deliberately remain global.
    void retire_prefill() {
        try {
            if (pf_active) { pf->drain(); pf_active = false; }
            if (pf != nullptr && pf_on) pf->drain();
            hip_ok(hipDeviceSynchronize(), "prefill retire");
            comm.check_exchanges();
        } catch (...) {
            comm.transport().abort("prefill retirement failure");
            throw;
        }
    }
    void bind_bank(bool child) {
        const auto& b = child ? nest_bank : base_bank;
        h = b.h; hn = b.hn; taps_d = b.taps;
        child_bank = child;
        pf_tab_slot = ~0U;
        last_out = {};
        last_result = {};
        if (task.open && attn_prefill_fused) prefill_table(task.slot);
    }
    void park_task(Engine::PfTaskId id) {
        if (!task.open || task.id != id || parked.open || child_bank)
            throw std::logic_error("prefill park: task identity / nesting level");
        retire_prefill();
        if (task.saved) restore_rows();
        hip_ok(hipDeviceSynchronize(), "prefill freeze");
        base_bank = {h, hn, taps_d};
        parked = std::move(task);
        task = PfTask{};
        parked_sink = std::move(tap_sink);
    }
    void resume_parked(Engine::PfTaskId id) {
        if (task.open || !parked.open || parked.id != id)
            throw std::logic_error("prefill unpark: task identity");
        retire_prefill();
        task = std::move(parked);
        parked = PfTask{};
        tap_sink = std::move(parked_sink);
        bind_bank(false);
        if (task.layer > 0) save_rows();
    }
    void cancel_task(Engine::PfTaskId id) {
        if (task.open && task.id == id) {
            prefill_cancel();
            tap_sink = {};
        } else if (parked.open && parked.id == id) {
            retire_prefill();
            parked = PfTask{};
            parked_sink = {};
        } else throw std::logic_error("prefill cancel: stale task ID");
    }

    // Rows [0, min(rows of the open chunk, kMaxStepRows)) of h and hn (a group's member 0 uses the base buffers) and taps.
    std::size_t saved_rows() const { return std::min<std::size_t>({kMaxStepRows, static_cast<std::size_t>(opt.max_rows), task.rows}); }
    void save_rows() {
        const std::size_t r = saved_rows();
        hip_ok(hipMemcpyAsync(sav_h, h, r * kHidden * 4, hipMemcpyDeviceToDevice, nullptr), "prefill pause");
        hip_ok(hipMemcpyAsync(sav_hn, hn, r * kHidden * 4, hipMemcpyDeviceToDevice, nullptr), "prefill pause");
        if (taps_on) hip_ok(hipMemcpyAsync(sav_taps, taps_d, r * 5 * kHidden * 4, hipMemcpyDeviceToDevice, nullptr), "prefill pause");
        task.saved = true;
    }
    void restore_rows() {
        const std::size_t r = saved_rows();
        hip_ok(hipMemcpyAsync(h, sav_h, r * kHidden * 4, hipMemcpyDeviceToDevice, nullptr), "prefill resume");
        hip_ok(hipMemcpyAsync(hn, sav_hn, r * kHidden * 4, hipMemcpyDeviceToDevice, nullptr), "prefill resume");
        if (taps_on) hip_ok(hipMemcpyAsync(taps_d, sav_taps, r * 5 * kHidden * 4, hipMemcpyDeviceToDevice, nullptr), "prefill resume");
        task.saved = false;
    }

    std::uint32_t prefill(std::uint32_t slot, std::span<const std::uint32_t> toks, std::uint64_t pos0) {
        no_task("prefill");
        prefill_begin(slot, toks, pos0);
        std::uint32_t next = 0;
        (void)prefill_resume(~0U, &next);
        return next;
    }

    // m = ceil(rows / max_rows) consecutive chunks of one prompt, layer-interleaved on the one stream (design notes):
    // per layer S1 (post of layer L-1, attention, o_proj, send) for every chunk in order, then S2 (post, FFN, send) for
    // every chunk in order. Each chunk's kernels get exactly the arguments of the chunk-serial order; the exchanges of
    // one chunk run while the GPU computes the next one. Returns the last chunk's normed rows (final norm applied).
    // Layers [L0, L1) (resumable prefill): a unit ending before the last layer completes every member's receive and post
    // of layer L1 - 1 before it returns (nullptr; nothing in flight), which the uninterrupted order runs at the start of
    // layer L1 (independent kernels on per-member buffers, the same arguments); the resumed unit skips them.
    const float* prefill_group(std::uint32_t slot, std::span<const std::uint32_t> toks, std::uint64_t pos0, unsigned L0 = 0,
                               unsigned L1 = kLayers) {
        namespace px = strix::minimax::pfx;
        const std::size_t m = (toks.size() + opt.max_rows - 1) / opt.max_rows;
        float* const h0 = h;
        float* const hn0 = hn;
        std::array<float*, 4> H{}, HN{}, P{}, R{};
        std::array<std::size_t, 4> N{};
        auto& fences = pf_fences[child_bank ? 1U : 0U];
        const double pipeline_reuse_before = comm.stats().pipeline_reuse_wait;
        const double host_reuse_before = comm.stats().host_reuse_wait;
        const bool via_comm = !pf_active;  // STRIX_MM_PF_INTERLEAVE=comm (pf_active is off then)
        for (std::size_t c = 0; c < m; ++c) {
            H[c] = c == 0 ? h0 : grp_h[c - 1];
            HN[c] = c == 0 ? hn0 : grp_hn[c - 1];
            P[c] = !via_comm ? part : c == 0 ? part : grp_part[c - 1];  // the module copies `part` into its TX slot at once
            R[c] = !via_comm ? nullptr : c == 0 ? remote : grp_remote[c - 1];
            N[c] = std::min<std::size_t>(opt.max_rows, toks.size() - c * opt.max_rows);
        }
        // One deterministic decision for the whole unit, identical on both ranks.
        // Rejected groups take exactly the ordinary off-switch scheduling path.
        const bool async_group = defaults::pf_async_group(pf_async_start, {N.data(), m}, pf_async_min_rows);
        // A rejected unit records no events. If this bank later returns to async,
        // capture its intervening ordinary posts before publishing destinations.
        for (std::size_t c = 0; via_comm && async_group && c < m; ++c) {
            if (fences.destination_untracked[c]) {
                hip_ok(hipEventRecord(fences.last_r_use[c], nullptr), "prefill destination route transition");
                fences.destination_untracked[c] = false;
            }
        }
        const auto use = [&](std::size_t c) {
            h = H[c];
            hn = HN[c];
            pf_idx_member = c;
            if (tl) tl->context(task.c + c, tl->layer);
        };
        struct Restore {
            Impl& e;
            float *saved_h, *saved_hn;
            ~Restore() {
                e.h = saved_h;
                e.hn = saved_hn;
                e.pf_idx_member = 0;
            }
        } restore{*this, h0, hn0};
        // Communicator backend: one exchange in flight (its start() / finish() contract). Sends are finished in FIFO
        // order; a receive finishes its exchange if it is the one in flight (the CPU waits while the GPU computes the
        // other chunk's stage, enqueued before).
        struct Pending {
            std::size_t c;
            bool done;
        };
        std::array<Pending, 8> fifo{};
        std::size_t f_head = 0, f_tail = 0;
        bool inflight = false;
        const auto finish_inflight = [&] {
            if (!inflight) return;
            Pending& e = fifo[(f_tail - 1) % fifo.size()];
            const auto ht = tl ? tl->host_begin("recv", task.c + e.c) : 0;
            comm.finish(nullptr, R[e.c]);
            pf_comm_inflight = false;
            if (tl) tl->host_end(ht, true);
            e.done = true;
            inflight = false;
        };
        // Where chunk c's next partial is produced: its own buffer (communicator), `part` (module, copied at once), or the
        // module's next TX slot (in place).
        const auto wait_source = [&](std::size_t c) {
            if (via_comm && fences.source_pending[c]) {
                hip_ok(hipStreamWaitEvent(nullptr, fences.p_free[c], 0), "prefill source lifetime");
                fences.source_pending[c] = false;
            }
        };
        const auto last_read = [&](std::size_t c) {
            if (via_comm && async_group)
                hip_ok(hipEventRecord(fences.last_r_use[c], nullptr), "prefill destination last reader");
            else if (via_comm) fences.destination_untracked[c] = true;
        };
        // All compute and shared scratch remain on main. Only copies overlap.
        const auto out_of = [&](std::size_t c) { wait_source(c); return !via_comm && (pf_inplace || (pf_tx_uncached && N[c] > kMaxStepRows)) ? static_cast<float*>(pf->tx()) : P[c]; };
        const auto send = [&](std::size_t c, std::uint32_t which, unsigned L, const float* produced) {
            const std::size_t bytes = N[c] * kHidden * sizeof(float);
            if (!via_comm) {
                if (tl) tl->mark("exchange-send", true);
                const auto ht = tl ? tl->host_begin("send") : 0;
                if (produced == pf->tx()) pf->send(bytes, px::Tag{which + 1, L, calls + c});
                else pf->send_copy(produced, bytes, px::Tag{which + 1, L, calls + c});
                if (tl) tl->host_end(ht, false);
                if (tl) tl->mark("exchange-send", false);
                return;
            }
            finish_inflight();
            if (tl) tl->mark("exchange-send", true);
            dist::Tag tag;
            tag.kind = static_cast<std::uint32_t>(dist::TagKind::half_sum);
            tag.layer = L;
            tag.request = (calls + c) * 2 + which;
            const auto ht = tl ? tl->host_begin("send") : 0;
            pf_comm_inflight = true; // also cover a partially submitted start throwing
            if (async_group) {
                fences.source_pending[c] = comm.start_async(nullptr, P[c], R[c], bytes, tag,
                                                            fences.last_r_use[c], fences.p_free[c]);
                if (!pf_late_src_fence) wait_source(c); // Fix A alone retains the all-D2H main fence
            } else comm.start(nullptr, P[c], bytes, tag);
            if (tl) tl->host_end(ht, false);
            fifo[f_tail++ % fifo.size()] = Pending{c, false};
            inflight = true;
            if (tl) tl->mark("exchange-send", false);
        };
        const auto recv = [&](std::size_t c) {
            if (!via_comm) {
                const auto ht = tl ? tl->host_begin("recv") : 0;
                const auto view = pf->recv();
                if (tl) tl->host_end(ht, true);
                return view;
            }
            Pending& e = fifo[f_head % fifo.size()];
            if (e.c != c) throw std::logic_error("prefill_group: receive order");
            if (!e.done) finish_inflight();
            ++f_head;
            return px::Exchanger::View{P[c], R[c]};
        };
        if (attn_prefill_fused) prefill_table(slot);
        for (std::size_t c = 0; L0 == 0 && c < m; ++c) {
            use(c);
            embed_rows(toks.data() + c * opt.max_rows, N[c]);
            hipLaunchKernelGGL(rmsnorm_gemma_kernel, dim3(static_cast<unsigned>(N[c])), dim3(256), 0, nullptr, h, w_in0, hn, kHidden);
        }
        for (unsigned L = L0; L < L1; ++L) {
            const Layer& y = layers[L];
            if (tl) tl->context(task.c, L);
            for (std::size_t c = 0; c < m; ++c) {
                use(c);
                if (L > L0) {
                    const auto view = recv(c);
                    if (tl) tl->mark("ffn-post", true);
                    post_view(N[c], view, layers[L - 1].w_next, nullptr, nullptr);
                    last_read(c);
                    if (tl) tl->mark("ffn-post", false);
                }  // L0 > 0: done at the pause
                float* o = out_of(c);
                stage_attn(L, N[c], {}, slot, pos0 + c * opt.max_rows, o);
                send(c, 0, L, o);
            }
            for (std::size_t c = 0; c < m; ++c) {
                use(c);
                const auto view = recv(c);
                if (tl) tl->mark("attention-post-router", true);
                post_view(N[c], view, y.w_post, y.wr, y.wb);
                last_read(c);
                if (tl) tl->mark("attention-post-router", false);
                float* o = out_of(c);
                stage_ffn(L, N[c], o);
                send(c, 1, L, o);
            }
        }
        for (std::size_t c = 0; c < m; ++c) {
            use(c);
            const auto view = recv(c);
            if (tl) tl->mark("ffn-post", true);
            post_view(N[c], view, layers[L1 - 1].w_next, nullptr, nullptr);
            last_read(c);
            if (tl) tl->mark("ffn-post", false);
        }
        // A unit may pause, run decode on part/remote, park, or start an
        // ungrouped tail next. Close every outstanding source read on main.
        for (std::size_t c = 0; c < m; ++c) wait_source(c);
        if (tl) {
            tl->pipeline_reuse_wait += comm.stats().pipeline_reuse_wait - pipeline_reuse_before;
            tl->host_reuse_wait += comm.stats().host_reuse_wait - host_reuse_before;
        }
        return L1 == kLayers ? HN[m - 1] : nullptr;
    }

    // FNV-1a of the last prefill's final normed rows (the head's input; exactness checks).
    std::pair<const float*, std::size_t> last_out{nullptr, 0};
    Engine::PfResult last_result;
    std::uint64_t prefill_out_fnv() {
        if (last_out.first == nullptr) return 0;
        hip_ok(hipDeviceSynchronize(), "out fnv");
        std::vector<float> rows(last_out.second * kHidden);
        hip_ok(hipMemcpy(rows.data(), last_out.first, rows.size() * 4, hipMemcpyDeviceToHost), "out fnv copy");
        std::uint64_t f = 14695981039346656037ULL;
        const auto* b = reinterpret_cast<const std::uint8_t*>(rows.data());
        for (std::size_t i = 0; i < rows.size() * 4; ++i) f = (f ^ b[i]) * 1099511628211ULL;
        return f;
    }

    void check_output(const float* src, std::size_t rows) {
        // Qualification only: retain a digest of ALL final normalized prompt rows, including earlier groups.
        hip_ok(hipDeviceSynchronize(), "check prompt output");
        std::vector<float> host(rows * kHidden);
        hip_ok(hipMemcpy(host.data(), src, host.size() * sizeof(float), hipMemcpyDeviceToHost), "check prompt output copy");
        const auto* b = reinterpret_cast<const unsigned char*>(host.data());
        for (std::size_t i = 0; i < host.size() * sizeof(float); ++i)
            task.output_fnv = (task.output_fnv ^ b[i]) * 1099511628211ULL;
    }

    std::uint64_t activation_fnv(Engine::PfTaskId id) {
        const bool is_parked = parked.open && parked.id == id;
        if (!is_parked && (!task.open || task.id != id)) throw std::logic_error("activation fnv: task ID");
        retire_prefill();
        const bool saved = !is_parked && task.saved;
        if (saved) restore_rows();
        hip_ok(hipDeviceSynchronize(), "activation fnv");
        const auto& t = is_parked ? parked : task;
        const float* h0 = is_parked ? base_bank.h : h;
        const float* hn0 = is_parked ? base_bank.hn : hn;
        const float* taps = is_parked ? base_bank.taps : taps_d;
        std::uint64_t f = 14695981039346656037ULL;
        const auto mix = [&](const float* src, std::size_t count) {
            std::vector<float> host_values(count);
            hip_ok(hipMemcpy(host_values.data(), src, count * sizeof(float), hipMemcpyDeviceToHost), "activation fnv copy");
            const auto* b = reinterpret_cast<const unsigned char*>(host_values.data());
            for (std::size_t i = 0; i < count * sizeof(float); ++i) f = (f ^ b[i]) * 1099511628211ULL;
        };
        if (t.layer > 0) {
            for (std::size_t member = 0; member < t.m; ++member) {
                const auto rows = std::min<std::size_t>(opt.max_rows, t.rows - member * opt.max_rows);
                mix(member == 0 ? h0 : grp_h[member - 1], rows * kHidden);
                mix(member == 0 ? hn0 : grp_hn[member - 1], rows * kHidden);
            }
            if (taps_on && !t.grouped) {
                constexpr unsigned tap_layers[] = {3, 17, 31, 45, 59};
                for (std::size_t row = 0; row < t.rows; ++row)
                    for (unsigned tap = 0; tap < 5; ++tap)
                        if (tap_layers[tap] < t.layer) mix(taps + (row * 5 + tap) * kHidden, kHidden);
            }
        }
        if (saved) save_rows();
        return f;
    }

    std::uint64_t state_fnv(std::uint32_t slot, std::uint64_t positions) {
        if (slot >= opt.slots || positions > opt.capacity) throw std::invalid_argument("state_fnv: bad slot / range");
        hip_ok(hipDeviceSynchronize(), "state fnv");
        std::uint64_t f = 14695981039346656037ULL;
        std::vector<std::uint16_t> buf(positions * kKVLocal);
        const auto mix = [&](const std::uint16_t* src, std::size_t width) {
            hip_ok(hipMemcpy(buf.data(), src, positions * width * 2, hipMemcpyDeviceToHost), "state fnv copy");
            const auto* b = reinterpret_cast<const std::uint8_t*>(buf.data());
            for (std::size_t i = 0; i < positions * width * 2; ++i) f = (f ^ b[i]) * 1099511628211ULL;
        };
        for (unsigned L = 0; L < kLayers; ++L) {
            mix(kc + cache_off(slot, L, kKVLocal), kKVLocal);
            mix(vc + cache_off(slot, L, kKVLocal), kKVLocal);
            mix(ikc + cache_off(slot, L, 128), 128);
        }
        return f;
    }

    std::vector<std::uint32_t> step(std::span<const Row> rows) {
        if (parked.open && !child_bank) throw std::logic_error("step: parked bank must remain frozen");
        const std::size_t n = rows.size();
        if (n == 0 || n > kMaxStepRows) throw std::invalid_argument("step: 1..8 rows");
        std::vector<std::uint32_t> toks(n);
        for (std::size_t r = 0; r < n; ++r) {
            if (rows[r].slot >= opt.slots || rows[r].pos >= opt.capacity) throw std::invalid_argument("step: bad row");
            if ((task.open && rows[r].slot == task.slot) || (parked.open && rows[r].slot == parked.slot)) throw std::logic_error("step: the slot of the open prefill");
            // Rows of one slot: consecutive ascending positions (verify window). The per-row K1-K4 loop below writes row
            // r's KV before its attention and after the earlier rows', so each row sees exactly its causal prefix.
            for (std::size_t s = r; s-- > 0;)
                if (rows[s].slot == rows[r].slot) {
                    if (rows[r].pos != rows[s].pos + 1) throw std::invalid_argument("step: rows of one slot must be consecutive");
                    break;
                }
            toks[r] = rows[r].token;
        }
        gemv = true;
        const auto layout = route::step_layout(rows);
        step_seqs = layout.seqs;
        step_rows_per_seq = layout.rows_per_seq;
        if (attn_fused && step_seqs != 0) {  // per-step tables of the fused MSA view: cache pointers, RoPE rows, positions
            for (unsigned L = 0; L < kLayers; ++L)
                for (std::size_t r = 0; r < step_seqs; ++r) {
                    std::uint16_t** t = cache_tab_h + std::size_t{L} * 3 * kMaxStepRows;
                    t[r] = kc + cache_off(rows[r].slot, L, kKVLocal);
                    t[kMaxStepRows + r] = vc + cache_off(rows[r].slot, L, kKVLocal);
                    t[2 * kMaxStepRows + r] = ikc + cache_off(rows[r].slot, L, 128);
                }
            hip_ok(hipMemcpyAsync(cache_tab_d, cache_tab_h, std::size_t{kLayers} * 3 * kMaxStepRows * sizeof(void*),
                                  hipMemcpyHostToDevice, nullptr), "cache table");
            for (std::size_t r = 0; r < n; ++r) {
                hip_ok(hipMemcpyAsync(step_rope + r, d_rope + rows[r].pos, sizeof(msa::RopeRow), hipMemcpyDeviceToDevice, nullptr), "rope");
                hip_ok(hipMemcpyAsync(step_pos + r, d_pos + rows[r].pos, 8, hipMemcpyDeviceToDevice, nullptr), "pos");
            }
            hip_ok(hipStreamSynchronize(nullptr), "step tables");  // the host table is rewritten by the next step
        }
        embed_rows(toks.data(), n);
        layers_forward(n, rows, 0, 0, kLayers);
        std::vector<std::uint32_t> out;
        if (use_xfer()) {  // head + GPU argmax candidates + pick through the module; the CPU waits once, here
            mark();
            lib.mm(head, hn, logits, n, true);
            mx::argmax_exchange(*xf, logits, static_cast<unsigned>(n), static_cast<unsigned>(vn), static_cast<unsigned>(vn),
                                opt.rank == 0 ? 0U : static_cast<std::uint32_t>(kVocabPad), calls, out_ids_d);
            out.resize(n);
            tmark(tmark_at(kLayers, 0));
            hip_ok(hipMemcpy(out.data(), out_ids_d, n * 4, hipMemcpyDeviceToHost), "ids");
            xf->drain();
            for (auto t : out)
                if (t == mx::kNonFinite) throw std::runtime_error("non-finite logits");
            lap(prof.head);
        } else {
            out = head_argmax(hn, n);
            tmark(tmark_at(kLayers, 0));
        }
        hip_ok(hipDeviceSynchronize(), "step");
        if (tracing()) read_trace(n);
        ++calls;
        ++prof.calls;
        return out;
    }

    void set_trace(bool on) {
        if (on && tev.empty()) {
            tev.resize(tmark_at(kLayers, 0) + 1);
            for (auto& e : tev) hip_ok(hipEventCreate(&e), "trace event");
        }
        trace_on = on;
    }

    // After the step's synchronisation: phase times from consecutive events, routed ids of the MoE layers.
    void read_trace(std::size_t n) {
        auto ms = [&](std::size_t a, std::size_t b) {
            float t = 0;
            hip_ok(hipEventElapsedTime(&t, tev[a], tev[b]), "trace time");
            return t;
        };
        trace.rows = static_cast<std::uint32_t>(n);
        trace.layer_ms.assign(std::size_t{kLayers} * StepTrace::kPhases, 0.0F);
        for (unsigned L = 0; L < kLayers; ++L)
            for (unsigned p = 0; p < StepTrace::kPhases; ++p) trace.layer_ms[L * StepTrace::kPhases + p] = ms(tmark_at(L, p) - 1, tmark_at(L, p));
        trace.head_ms = ms(tmark_at(kLayers, 0) - 1, tmark_at(kLayers, 0));
        trace.total_ms = ms(0, tmark_at(kLayers, 0));
        hip_ok(hipMemcpy(trace_ids_h, trace_ids_d, std::size_t{kLayers} * kMaxStepRows * kTopE * 4, hipMemcpyDeviceToHost), "trace ids");
        trace.routes.assign(std::size_t{kLayers} * n * kTopE, 0U);
        for (unsigned L = kFirstMoe; L < kLayers; ++L)
            std::memcpy(trace.routes.data() + std::size_t{L} * n * kTopE, trace_ids_h + std::size_t{L} * kMaxStepRows * kTopE, n * kTopE * 4);
    }

    std::vector<float> gate_residual(std::span<const std::uint32_t> toks, unsigned n_layers) {
        no_task("gate");
        const std::size_t T = toks.size();
        if (T > opt.max_rows || T > opt.capacity || n_layers == 0 || n_layers > kLayers) throw std::invalid_argument("gate: bad size");
        gemv = false;
        embed_rows(toks.data(), T);
        layers_forward(T, {}, 0, 0, n_layers);
        ++calls;
        std::vector<float> out(T * kHidden);
        hip_ok(hipMemcpy(out.data(), h, out.size() * 4, hipMemcpyDeviceToHost), "gate residual");
        return out;
    }

    std::vector<float> gate_logits(std::span<const std::uint32_t> toks, std::span<const std::uint32_t> rows) {
        no_task("gate");
        const std::size_t T = toks.size(), S = rows.size();
        if (T > opt.max_rows || T > opt.capacity) throw std::invalid_argument("gate: bad size");
        gemv = false;
        embed_rows(toks.data(), T);
        layers_forward(T, {}, 0, 0, kLayers);
        // As G4: gather the scored rows of hn (final norm), gemm the vocabulary half in 256-row chunks, exchange.
        hip_ok(hipMemcpy(rows_d, rows.data(), S * 4, hipMemcpyHostToDevice), "scored rows");
        hipLaunchKernelGGL(gather_kernel, dim3(static_cast<unsigned>(S)), dim3(256), 0, nullptr, hn, rows_d, ye, kHidden);
        std::vector<float> full(opt.rank == 0 ? S * kVocab : 0);
        float* lg_local = dalloc<float>(kGatherRows * kVocabPad);
        float* lg_remote = dalloc<float>(kGatherRows * kVocabPad);
        for (std::size_t r0 = 0; r0 < S; r0 += kGatherRows) {
            const std::size_t n = std::min(kGatherRows, S - r0);
            hip_ok(hipMemset(lg_local, 0, kGatherRows * kVocabPad * 4), "zero");
            lib.mm(head, ye + r0 * kHidden, logits, n, false);
            hip_ok(hipMemcpy2D(lg_local, kVocabPad * 4, logits, vn * 4, vn * 4, n, hipMemcpyDeviceToDevice), "pad");
            dist::Tag tag;
            tag.kind = static_cast<std::uint32_t>(dist::TagKind::raw_slots);
            tag.layer = kLayers;
            tag.request = r0;
            comm.exchange(nullptr, lg_local, lg_remote, kGatherRows * kVocabPad * 4, tag);
            if (opt.rank == 0) {
                std::vector<float> a(n * kVocabPad), b(n * kVocabPad);
                hip_ok(hipMemcpy(a.data(), lg_local, a.size() * 4, hipMemcpyDeviceToHost), "a");
                hip_ok(hipMemcpy(b.data(), lg_remote, b.size() * 4, hipMemcpyDeviceToHost), "b");
                for (std::size_t i = 0; i < n; ++i) {
                    std::memcpy(&full[(r0 + i) * kVocab], &a[i * kVocabPad], kVocabPad * 4);
                    std::memcpy(&full[(r0 + i) * kVocab + kVocabPad], &b[i * kVocabPad], (kVocab - kVocabPad) * 4);
                }
            }
        }
        hip_ok(hipFree(lg_local), "free");
        hip_ok(hipFree(lg_remote), "free");
        ++calls;
        return full;
    }

    std::uint32_t agree(std::uint32_t val) {  // a collective: both ranks exchange, rank 0's value wins
        const std::uint32_t other = peer(val);
        return opt.rank == 0 ? val : other;
    }

    void copy_slot(std::uint32_t dst, std::uint32_t src, std::uint64_t positions) {
        no_task("copy_slot");
        if (dst >= opt.slots || src >= opt.slots || positions > opt.capacity) throw std::invalid_argument("copy_slot");
        for (unsigned L = 0; L < kLayers; ++L) {
            hip_ok(hipMemcpy(kc + cache_off(dst, L, kKVLocal), kc + cache_off(src, L, kKVLocal), positions * kKVLocal * 2,
                             hipMemcpyDeviceToDevice), "copy k");
            hip_ok(hipMemcpy(vc + cache_off(dst, L, kKVLocal), vc + cache_off(src, L, kKVLocal), positions * kKVLocal * 2,
                             hipMemcpyDeviceToDevice), "copy v");
            hip_ok(hipMemcpy(ikc + cache_off(dst, L, 128), ikc + cache_off(src, L, 128), positions * 128 * 2,
                             hipMemcpyDeviceToDevice), "copy ik");
        }
    }

    // Prompt-state hand-off blob (engine.hpp export_rows / import_rows): per layer K, V, index key rows.
    void state_rows(std::uint32_t slot, std::uint64_t pos0, std::uint64_t n, std::uint8_t* host, bool to_host, unsigned l0 = 0,
                    unsigned l1 = kLayers) {
        no_task("state rows");
        if (slot >= opt.slots || pos0 > opt.capacity || n > opt.capacity - pos0 || l1 > kLayers) throw std::invalid_argument("state rows: bad slot / range");
        const hipMemcpyKind kind = to_host ? hipMemcpyDeviceToHost : hipMemcpyHostToDevice;
        for (unsigned L = l0; L < l1; ++L) {
            std::uint8_t* b = host + static_cast<std::size_t>(L - l0) * n * 1280;
            std::uint16_t* dev[3] = {kc + cache_off(slot, L, kKVLocal) + pos0 * kKVLocal, vc + cache_off(slot, L, kKVLocal) + pos0 * kKVLocal,
                                     ikc + cache_off(slot, L, 128) + pos0 * 128};
            const std::size_t bytes[3] = {n * kKVLocal * 2, n * kKVLocal * 2, n * 128 * 2};
            std::size_t off = 0;
            for (int i = 0; i < 3; ++i) {
                if (to_host) hip_ok(hipMemcpy(b + off, dev[i], bytes[i], kind), "export rows");
                else hip_ok(hipMemcpy(dev[i], b + off, bytes[i], kind), "import rows");
                off += bytes[i];
            }
        }
    }

    void set_taps(bool on) {
        no_task("set_taps");
        if (child_bank) bind_bank(false);
        if (on && taps_d == nullptr) taps_d = dalloc<float>(static_cast<std::size_t>(opt.max_rows) * 5 * kHidden);
        taps_on = on;
    }
    const float* head_logits(const float* rows_in, std::size_t n) {
        if (n == 0 || n > kMaxStepRows) throw std::invalid_argument("head_logits: 1..8 rows");
        const DraftHead* d = dhead_sel >= 0 ? &dheads[static_cast<std::size_t>(dhead_sel)] : nullptr;
        if (d != nullptr && d->ws) {
            const mmoe::LinearJob j{d->raw, rows_in, static_cast<std::uint32_t>(kHidden), logits, static_cast<std::uint32_t>(vn)};
            if (d->version != 0) mmoe::enqueue_linears_version(&j, 1, static_cast<unsigned>(n), d->ws.get(), nullptr, d->version);
            else mmoe::enqueue_linears_scheduled(&j, 1, static_cast<unsigned>(n), d->ws.get(), nullptr);
        } else {
            lib.mm(d != nullptr ? d->lin : head, rows_in, logits, n, true);
        }
        return logits;
    }
    // Draft-only head (attach_draft_head): this rank's slice of a requant of the lm_head, read by head_logits() only.
    // STRIX_MM_DRAFT_HEAD=lib (default: strix_gemv) | sched (enqueue_linears_scheduled on a raw copy, strix_gemv bits) |
    // sched:V (that enqueue_linears form, V in 4 8 9 10; measured at R=7: K4 v9 1.77 ms vs K8 strix_gemv 2.96 ms).
    struct HipFree {
        void operator()(void* p) const { (void)hipFree(p); }
    };
    // Several may be attached (in-process comparisons); head_logits() reads dheads[dhead_sel], the last attached by
    // default, the target's head when dhead_sel < 0 (set_draft_head).
    struct DraftHead {
        Lin lin;                            // lib
        mmoe::Exl3 raw{};                   // sched
        std::unique_ptr<void, HipFree> ws;  // sched: enqueue_linears workspace (block counters zeroed)
        int version = 0;                    // sched:V
    };
    std::vector<DraftHead> dheads;
    int dhead_sel = -1;
    void attach_draft_head(const std::string& dir) {
        no_task("attach_draft_head");
        const auto t0 = Clock::now();
        const char* e = std::getenv("STRIX_MM_DRAFT_HEAD");
        const std::string spec = e != nullptr ? e : "lib";
        const std::string mode = spec.substr(0, spec.find(':'));
        int version = 0;
        if (mode == "sched" && spec.size() > 5) {
            const std::string vs = spec.substr(6);
            version = vs == "4" ? 4 : vs == "8" ? 8 : vs == "9" ? 9 : vs == "10" ? 10 : -1;
        }
        if ((mode != "lib" && mode != "sched") || version < 0 || (mode == "lib" && spec != "lib"))
            throw std::invalid_argument("STRIX_MM_DRAFT_HEAD must be lib, sched or sched:4|8|9|10");
        PackReader pk(dir);
        const std::string nm = "language_model.lm_head";
        {  // the same shape as the target's head, MUL1 trellis (the library and the kernels assume it) of 3..8 bits
            const TensorInfo &tr = pk.info(nm + ".trellis"), &suh = pk.info(nm + ".suh"), &svh = pk.info(nm + ".svh");
            const bool ok = tr.shape.size() == 3 && suh.shape.size() == 1 && svh.shape.size() == 1 && suh.shape[0] == kHidden &&
                            svh.shape[0] == kVocab && tr.shape[0] == kHidden / 16 && tr.shape[1] == kVocab / 16 &&
                            tr.shape[2] % 16 == 0 && tr.shape[2] / 16 >= 3 && tr.shape[2] / 16 <= 8 &&
                            (mode == "lib" || tr.shape[2] / 16 != 7);
            if (!ok) throw std::invalid_argument("attach_draft_head: " + dir + " lm_head is not a 3..8-bit " + std::to_string(kVocab) + " x " +
                                                 std::to_string(kHidden) + " EXL3 tensor" + (mode == "sched" ? " (sched: not 7 bits)" : ""));
            const TensorInfo& mul1 = pk.info(nm + ".mul1");
            const bool types = tr.dtype == "I16" && tr.bytes == tr.shape[0] * tr.shape[1] * tr.shape[2] * 2 && suh.dtype == "F16" &&
                               suh.bytes == kHidden * 2 && svh.dtype == "F16" && svh.bytes == kVocab * 2 && mul1.dtype == "I32" &&
                               mul1.bytes == 4;
            if (!types) throw std::invalid_argument("attach_draft_head: " + dir + " lm_head dtypes / sizes differ from the target pack's");
            if (version == 10 && (kVocabPad % 256 != 0 || (kVocab - kVocabPad) % 256 != 0))  // both ranks refuse together
                throw std::invalid_argument("attach_draft_head: v10 needs a multiple of 256 outputs on both ranks (slices " +
                                            std::to_string(kVocabPad) + ", " + std::to_string(kVocab - kVocabPad) + ")");
        }
        struct Release {  // the pinned staging buffer and the whole-trellis host copy go on every exit
            Lib& l;
            ~Release() { l.release_staging(); }
        } release{lib};
        const std::uint64_t v0 = opt.rank == 0 ? 0 : kVocabPad;
        std::uint64_t bits = 0;
        if (mode == "sched") {
            const mmoe::Exl3 r = lib.raw_out(pk, nm, v0, vn);
            const mmoe::LinearJob j{r, nullptr, 0, nullptr, 0};
            const std::size_t wb = mmoe::linear_workspace_bytes(&j, 1);
            std::unique_ptr<void, HipFree> ws(dalloc<std::uint8_t>(wb));
            hip_ok(hipMemset(ws.get(), 0, wb), "draft head workspace");  // block counters start at 0 (plan_linears)
            hip_ok(hipDeviceSynchronize(), "load draft head");
            DraftHead dh;
            dh.raw = r;
            dh.ws = std::move(ws);
            dh.version = version;
            dheads.push_back(std::move(dh));
            bits = r.bits;
        } else {
            const Lin l = lib.out_slice(pk, nm, v0, vn);
            lib.reserve_workspace(std::max(lib.workspace_bytes(), lib.workspace_bound(l, hn, logits, kMaxStepRows)));
            hip_ok(hipDeviceSynchronize(), "load draft head");
            DraftHead dh;
            dh.lin = l;
            dheads.push_back(std::move(dh));
            bits = l.bits;
        }
        dhead_sel = static_cast<int>(dheads.size()) - 1;
        std::printf("rank %u: DSpark draft-only head #%d %s (K%llu, %s, vocabulary [%llu, %llu)), %.1f ms\n", opt.rank, dhead_sel, dir.c_str(),
                    static_cast<unsigned long long>(bits), spec.c_str(), static_cast<unsigned long long>(v0),
                    static_cast<unsigned long long>(v0 + vn), std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    // DSpark draft linears (attach_draft_exl3): DraftLinears, STRIX_MM_DRAFT_LIN=opt (default: q+k+v and gate+up grouped on
    // enqueue_linears, strix_gemv bits) | lib (strix_gemv per linear, 9bc47241) | load (both, Engine::set_draft_linears_opt).
    std::unique_ptr<DraftLinears> dlin;
    void attach_draft_exl3(const std::string& dir, dspark::DeviceDrafter& dd) {
        const auto w0 = std::chrono::steady_clock::now();
        dlin = std::make_unique<DraftLinears>(dir, DraftLinears::mode_from_env());
        const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
        // Warm-up at load, one call per shape the drafter uses (7-row block linears, 32-row fc / k / v appends): the
        // draft's library context paid ~2.5 s at its first gemm inside the first spec TTFT (measurement).
        const auto w1 = std::chrono::steady_clock::now();
        dlin->warm_up();
        static const char* const kModes[3] = {"lib", "opt", "load"};
        std::printf("rank %u: DSpark EXL3 drafter %s, %u linears%s, linears %s (%.2f GB device), load %.1f ms, warm-up %.1f ms\n", opt.rank,
                    dir.c_str(), DraftLinears::kBlock + (dlin->has_fc() ? 1U : 0U), dlin->has_fc() ? " (fc included)" : "",
                    kModes[static_cast<int>(dlin->mode())], static_cast<double>(dlin->device_bytes()) / 1e9, load_ms,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w1).count());
        const bool has_fc = dlin->has_fc();
        dd.set_linear_hook([this, has_fc](std::uint32_t L, std::uint32_t kind, const float* x, std::size_t rows, float* out, hipStream_t st) {
            if (kind == dspark::DeviceDrafter::lin_fc && !has_fc) return false;
            dlin->run(L, kind, x, rows, out, st);
            return true;
        });
        dd.set_linear_group_hook([this](std::uint32_t L, const std::uint32_t* kinds, std::uint32_t count, const float* x, std::size_t rows,
                                        float* const* outs, hipStream_t st) {
            dlin->run_group(L, kinds, count, x, rows, outs, st);
            return true;
        });
    }

    std::array<std::uint32_t, 2> peer2(std::uint32_t a, std::uint32_t b) {
        std::uint32_t mine[2] = {a, b};
        std::array<std::uint32_t, 2> theirs{};
        hip_ok(hipMemcpy(am_local, mine, 8, hipMemcpyHostToDevice), "ctl");
        dist::Tag tag;
        tag.kind = static_cast<std::uint32_t>(dist::TagKind::control);
        tag.request = controls++;
        comm.exchange(nullptr, am_local, am_remote, 8, tag);
        hip_ok(hipMemcpy(theirs.data(), am_remote, 8, hipMemcpyDeviceToHost), "ctl r");
        return theirs;
    }

    // Control channel (serve): `bytes` <= kControlBytes of host memory each way, same size on both ranks.
    void exchange_bytes(const void* out, void* in, std::size_t bytes) {
        if (bytes == 0 || bytes > kControlBytes) throw std::invalid_argument("exchange_bytes: 1..65536 bytes");
        hip_ok(hipMemcpy(ctl_local, out, bytes, hipMemcpyHostToDevice), "ctl");
        dist::Tag tag;
        tag.kind = static_cast<std::uint32_t>(dist::TagKind::control);
        tag.request = controls++;
        comm.exchange(nullptr, ctl_local, ctl_remote, bytes, tag);
        hip_ok(hipMemcpy(in, ctl_remote, bytes, hipMemcpyDeviceToHost), "ctl r");
    }

    void exchange_device(const void* src_dev, void* dst_dev, std::size_t bytes, hipStream_t on) {
        if (bytes == 0) throw std::invalid_argument("exchange_device: bytes must be > 0");
        dist::Tag tag;
        tag.kind = static_cast<std::uint32_t>(dist::TagKind::control);
        tag.request = controls++;
        comm.exchange(on, src_dev, dst_dev, bytes, tag);
    }

    std::uint32_t peer(std::uint32_t val) {
        std::uint32_t mine[2] = {val, 0}, theirs[2] = {0, 0};
        hip_ok(hipMemcpy(am_local, mine, 8, hipMemcpyHostToDevice), "ctl");
        dist::Tag tag;
        tag.kind = static_cast<std::uint32_t>(dist::TagKind::control);
        tag.request = controls++;
        comm.exchange(nullptr, am_local, am_remote, 8, tag);
        hip_ok(hipMemcpy(theirs, am_remote, 8, hipMemcpyDeviceToHost), "ctl r");
        return theirs[0];
    }
};

Engine::Engine(Options options, dist::Communicator& comm) : impl_(std::make_unique<Impl>(std::move(options), comm)) {}
Engine::~Engine() = default;
LoadReport Engine::load() { return impl_->load(); }
std::uint32_t Engine::prefill(std::uint32_t slot, std::span<const std::uint32_t> tokens, std::uint64_t pos0) {
    return impl_->prefill(slot, tokens, pos0);
}
std::vector<std::uint32_t> Engine::step(std::span<const Row> rows) { return impl_->step(rows); }
void Engine::set_taps(bool on) { impl_->set_taps(on); }
const float* Engine::last_taps() const { return impl_->taps_d; }
void Engine::set_prefill_tap_sink(PrefillTapSink sink) {
    impl_->no_task("set_prefill_tap_sink");
    impl_->tap_sink = std::move(sink);
}
bool Engine::reserve_nested(bool taps) {
    impl_->no_task("reserve_nested");
    auto& m = *impl_;
    if (m.nest_rows != 0) throw std::logic_error("reserve_nested: already reserved");
    const unsigned rows = std::min(512U, m.opt.max_rows);
    const std::size_t physical = std::max(rows, kMaxStepRows);
    const std::size_t bytes = physical * kHidden * sizeof(float) * (taps ? 7 : 2);
    std::size_t free = 0, total = 0;
    hip_ok(hipMemGetInfo(&free, &total), "nested memory");
    std::ifstream mem("/proc/meminfo");
    std::string key, line;
    std::uint64_t available = 0;
    while (mem >> key) {
        if (key == "MemAvailable:") { mem >> available; available *= 1024; break; }
        std::getline(mem, line);
    }
    bool ready = m.opt.slots > 1 && !m.fp32_prefill && !m.profiling && !m.pf_on &&
                 free >= bytes + (4ULL << 30) && available >= bytes + (4ULL << 30);
    if (ready) {
        try {
            m.nest_bank.h = dalloc<float>(physical * kHidden);
            m.nest_bank.hn = dalloc<float>(physical * kHidden);
            if (taps) m.nest_bank.taps = dalloc<float>(physical * 5 * kHidden);
            if (m.timeline) m.child_timeline = std::make_unique<PrefillTimeline>(rows, rows, kLayers, m.pf_msa_windows);
            // Query the actual aligned child inputs before READY; scratch/output shapes stay unchanged.
            float* original = m.hn;
            m.hn = m.nest_bank.hn;
            try { m.reserve_workspace(rows); }
            catch (...) { m.hn = original; throw; }
            m.hn = original;
            // The resource margin also covers timeline events and any additional workspace.
            hip_ok(hipMemGetInfo(&free, &total), "nested reserved memory");
            std::ifstream after("/proc/meminfo");
            available = 0;
            while (after >> key) {
                if (key == "MemAvailable:") { after >> available; available *= 1024; break; }
                std::getline(after, line);
            }
            ready = free >= (4ULL << 30) && available >= (4ULL << 30);
        } catch (...) { ready = false; }
    }
    ready = peer(ready ? 1U : 0U) != 0 && ready;
    if (!ready) {
        for (auto* ptr : {m.nest_bank.h, m.nest_bank.hn, m.nest_bank.taps}) if (ptr) (void)hipFree(ptr);
        m.nest_bank = {};
        m.child_timeline.reset();
        m.nest_reason = "resources_or_mode";
        return false;
    }
    m.base_bank = {m.h, m.hn, m.taps_d};
    m.nest_rows = rows;
    m.nest_reason.clear();
    return true;
}
void Engine::set_prefill_check_outputs(bool on) { impl_->no_task("check outputs"); impl_->check_prompt_outputs = on; }
std::uint64_t Engine::prefill_check_output_fnv() const { return impl_->task.output_fnv; }
std::uint64_t Engine::prefill_activation_fnv(PfTaskId id) { return impl_->activation_fnv(id); }
unsigned Engine::nested_rows() const { return impl_->nest_rows; }
const char* Engine::nested_reason() const { return impl_->nest_reason.c_str(); }
Engine::PfResult Engine::prefill_result() const { return impl_->last_result; }
Engine::PfProgress Engine::prefill_progress(bool parked) const {
    const auto& t = parked ? impl_->parked : impl_->task;
    return {t.id, t.parent, t.generation, t.c0, t.c, t.rows, t.slot, parked ? 0U : impl_->child_bank ? 1U : 0U, t.layer, t.next, t.open};
}
void Engine::prefill_begin_task(PfTaskId id, PfTaskId parent, std::uint64_t generation,
                               std::uint32_t slot, std::span<const std::uint32_t> tokens, PrefillTapSink sink) {
    auto& m = *impl_;
    if (id == 0 || m.task.open) throw std::logic_error("prefill begin: task identity");
    if (parent != 0) {
        if (!m.parked.open || m.parked.id != parent || id == parent || slot == m.parked.slot ||
            tokens.empty() || tokens.size() > m.nest_rows || (m.taps_on && m.nest_bank.taps == nullptr))
            throw std::logic_error("prefill nest: parent / shape / slot");
        m.bind_bank(true);
    } else {
        m.no_task("prefill begin");
        if (m.child_bank) m.bind_bank(false);
    }
    m.tap_sink = std::move(sink);
    m.prefill_begin(slot, tokens, 0);
    m.task.id = id; m.task.parent = parent; m.task.generation = generation;
}
void Engine::prefill_park(PfTaskId id) { impl_->park_task(id); }
void Engine::prefill_resume_parked(PfTaskId id) { impl_->resume_parked(id); }
void Engine::prefill_cancel(PfTaskId id) { impl_->cancel_task(id); }
bool Engine::prefill_active() const { return impl_->task.open; }
void Engine::prefill_begin(std::uint32_t slot, std::span<const std::uint32_t> tokens, std::uint64_t pos0) {
    impl_->no_task("prefill");
    impl_->prefill_begin(slot, tokens, pos0);
}
bool Engine::prefill_resume(unsigned layers, std::uint32_t* next) { return impl_->prefill_resume(layers, next); }
void Engine::prefill_cancel() { impl_->prefill_cancel(); }
const char* Engine::prefill_unlimited_reason() const {
    return impl_->profiling ? "profiling" : impl_->fp32_prefill ? "fp32_prefill" : "";
}
bool Engine::prefill_open() const { return impl_->task.open || impl_->parked.open; }
LoaderSnapshot Engine::loader_snapshot() const { return impl_->pack.loader_snapshot(); }
LoaderSnapshot Engine::draft_loader_snapshot() const {
    return impl_->dlin ? impl_->dlin->loader_snapshot() : LoaderSnapshot{};
}
unsigned Engine::prefill_layer() const { return impl_->task.layer; }
std::uint64_t Engine::prefill_unit_rows() const { return impl_->task.open ? impl_->task.rows : 0; }
std::uint64_t Engine::prefill_rows_left() const { return impl_->task.open ? impl_->task.toks.size() - impl_->task.c0 : 0; }
const std::uint16_t* Engine::embed_row(std::uint32_t token) const {
    if (token >= kVocab) throw std::out_of_range("embed_row");
    return impl_->embed + static_cast<std::size_t>(token) * kHidden;
}
std::pair<std::uint32_t, std::uint32_t> Engine::vocab_range() const {
    return {impl_->opt.rank == 0 ? 0U : static_cast<std::uint32_t>(kVocabPad), static_cast<std::uint32_t>(impl_->vn)};
}
const float* Engine::head_logits(const float* rows, std::size_t n) { return impl_->head_logits(rows, n); }
std::array<std::uint32_t, 2> Engine::peer2(std::uint32_t a, std::uint32_t b) { return impl_->peer2(a, b); }
void Engine::attach_draft_exl3(const std::string& dir, dspark::DeviceDrafter& drafter) { impl_->attach_draft_exl3(dir, drafter); }
void Engine::attach_draft_head(const std::string& dir) { impl_->attach_draft_head(dir); }
int Engine::draft_heads() const { return static_cast<int>(impl_->dheads.size()); }
void Engine::set_draft_head(int index) {
    if (index < -1 || index >= static_cast<int>(impl_->dheads.size())) throw std::out_of_range("set_draft_head: no such draft head");
    impl_->dhead_sel = index;
}
void Engine::set_draft_linears_opt(bool on) {
    if (!impl_->dlin) throw std::logic_error("set_draft_linears_opt: no EXL3 drafter attached");
    impl_->dlin->set_opt(on);
}
std::vector<float> Engine::gate_residual(std::span<const std::uint32_t> tokens, unsigned layers) {
    return impl_->gate_residual(tokens, layers);
}
std::vector<float> Engine::gate_logits(std::span<const std::uint32_t> tokens, std::span<const std::uint32_t> rows) {
    return impl_->gate_logits(tokens, rows);
}
std::uint32_t Engine::agree(std::uint32_t value) { return impl_->agree(value); }
std::uint32_t Engine::peer(std::uint32_t value) { return impl_->peer(value); }
std::vector<float> Engine::last_logits(std::size_t rows) const {
    if (rows == 0 || rows > kMaxStepRows) throw std::invalid_argument("last_logits: 1..8 rows");
    std::vector<float> out(rows * impl_->vn);
    hip_ok(hipMemcpy(out.data(), impl_->logits, out.size() * 4, hipMemcpyDeviceToHost), "last logits");
    return out;
}
void Engine::abort(const std::string& reason) { impl_->comm.transport().abort(reason); }
void Engine::exchange_bytes(const void* out, void* in, std::size_t bytes) { impl_->exchange_bytes(out, in, bytes); }
void Engine::exchange_device(const void* local, void* remote, std::size_t bytes, void* stream) {
    impl_->exchange_device(local, remote, bytes, static_cast<hipStream_t>(stream));
}
std::uint64_t Engine::capacity() const { return impl_->opt.capacity; }
std::uint32_t Engine::slots() const { return impl_->opt.slots; }
void Engine::copy_slot(std::uint32_t dst, std::uint32_t src, std::uint64_t positions) { impl_->copy_slot(dst, src, positions); }
void Engine::export_rows(std::uint32_t slot, std::uint64_t pos0, std::uint64_t n, void* host) const {
    impl_->state_rows(slot, pos0, n, static_cast<std::uint8_t*>(host), true);
}
void Engine::import_rows(std::uint32_t slot, std::uint64_t pos0, std::uint64_t n, const void* host) {
    impl_->state_rows(slot, pos0, n, static_cast<std::uint8_t*>(const_cast<void*>(host)), false);
}
void Engine::import_layer(std::uint32_t slot, unsigned layer, std::uint64_t pos0, std::uint64_t n, const void* host) {
    impl_->state_rows(slot, pos0, n, static_cast<std::uint8_t*>(const_cast<void*>(host)), false, layer, layer + 1);
}
const dist::CommunicatorStats& Engine::comm_stats() const { return impl_->comm.stats(); }
const Profile& Engine::profile() const { return impl_->prof; }
std::uint32_t Engine::rank() const { return impl_->opt.rank; }
void Engine::set_fused_staging(bool on) {
    impl_->no_task("set_fused_staging");
    impl_->fused = on;
}
bool Engine::fused_staging() const { return impl_->fused; }
void Engine::set_fused_attention(bool on) { impl_->attn_fused = on; }
void Engine::set_fused_prefill_attention(bool on) {
    impl_->no_task("set_fused_prefill_attention");
    impl_->attn_prefill_fused = on;
}
void Engine::set_prefill_index_tile(bool on) {
    impl_->no_task("set_prefill_index_tile");
    impl_->pf_idx_tile = on;
}
bool Engine::prefill_index_tile() const { return impl_->pf_idx_tile; }
void Engine::set_prefill_msa_batch(bool on) {
    impl_->no_task("set_prefill_msa_batch");
    impl_->pf_msa_batch = on;
}
bool Engine::prefill_msa_batch() const { return impl_->pf_msa_batch; }
void Engine::set_prefill_msa_rework(unsigned mode) {
    impl_->no_task("set_prefill_msa_rework");
    impl_->set_prefill_msa_rework(mode);
}
unsigned Engine::prefill_msa_rework() const { return impl_->pf_msa_rework; }
void Engine::set_prefill_attend4(bool on) {
    impl_->no_task("set_prefill_attend4");
    impl_->pf_attend4 = on;
}
bool Engine::prefill_attend4() const { return impl_->pf_attend4; }
void Engine::set_index_tiled(bool on) {
    impl_->no_task("set_index_tiled");
    impl_->index_tiled = on;
}
void Engine::set_index_decode(bool on) { impl_->index_decode = on; }
void Engine::set_index_attn(bool on) {
    impl_->no_task("set_index_attn");
    impl_->index_attn = on;
}
void Engine::set_prefill_index_concurrent(int mode) {
    impl_->no_task("set_prefill_index_concurrent");
    if (mode < 0 || mode > 2) throw std::invalid_argument("prefill index concurrency must be 0, 1 or 2");
    impl_->pf_idx_conc = mode;
}
int Engine::prefill_index_concurrent() const { return impl_->pf_idx_conc; }
void Engine::set_prefill_kv_tile(int cfg) {
    impl_->no_task("set_prefill_kv_tile");
    if (cfg < -2) throw std::invalid_argument("prefill KV tile must be -2 (auto), -1 (off) or a config index");
    impl_->pf_kv_tile = cfg;
    if (cfg != -1) (void)kv_tile_available();
}
int Engine::prefill_kv_tile() const { return impl_->pf_kv_tile; }
void Engine::set_index_concurrent(bool on) { impl_->index_conc = on; }  // decode only: no prefill task state
void Engine::set_attend_diet(bool on) { impl_->attend_diet = on; }  // decode only
bool Engine::attend_diet() const { return impl_->attend_diet; }
void Engine::set_attend4_decode(bool on) { impl_->attend4_decode = on; }  // decode only
bool Engine::attend4_decode() const { return impl_->attend4_decode; }
void Engine::set_select_once(bool on) { impl_->select_once = on; }  // decode only
bool Engine::select_once() const { return impl_->select_once; }
std::uint64_t Engine::select_once_count() const { return impl_->select_once_calls; }
void Engine::set_xb_wait(bool on) { impl_->xb_wait = on; }
bool Engine::xb_wait() const { return impl_->xb_wait; }
void Engine::set_xb_inplace(bool on) { impl_->xb_inplace = on; }
bool Engine::xb_inplace() const { return impl_->xb_inplace; }
std::pair<std::uint64_t, std::uint64_t> Engine::xb_counts() const { return {impl_->xb_count[0], impl_->xb_count[1]}; }
void Engine::set_xb_prep(bool on) { impl_->xb_prep = on; }
bool Engine::xb_prep() const { return impl_->xb_prep; }
std::uint64_t Engine::xb_prep_count() const { return impl_->xb_count[2]; }
void Engine::set_xb_publish(bool on) {
    if (on && !mmoe::producer_publish_supported())
        throw std::invalid_argument("set_xb_publish: needs the v4 linear core and the decode MoE overlap form");
    impl_->xb_pub = on;
}
bool Engine::xb_publish() const { return impl_->xb_pub; }
std::uint64_t Engine::xb_publish_count() const { return impl_->xb_count[3]; }
void Engine::set_xb_warm(unsigned mb) { impl_->xb_warm_mb = mb; }
unsigned Engine::xb_warm() const { return impl_->xb_warm_mb; }
std::uint64_t Engine::xb_warm_count() const { return impl_->xb_count[4]; }
void Engine::set_xb_warm_o(unsigned mb) { impl_->xb_warm_o_mb = mb; }
unsigned Engine::xb_warm_o() const { return impl_->xb_warm_o_mb; }
std::uint64_t Engine::xb_warm_o_count() const { return impl_->xb_warm_o_count; }
void Engine::set_xb_postilp(bool on) { impl_->xb_postilp = on; }
bool Engine::xb_postilp() const { return impl_->xb_postilp; }
std::uint64_t Engine::xb_postilp_count() const { return impl_->xb_count[5]; }
bool Engine::index_concurrent() const { return impl_->index_conc; }
bool Engine::index_attn() const { return impl_->index_attn; }
bool Engine::index_decode() const { return impl_->index_decode; }
bool Engine::index_tiled() const { return impl_->index_tiled; }
bool Engine::fused_prefill_attention() const { return impl_->attn_prefill_fused; }
void Engine::set_exchanger(xfer::Exchanger* x, bool on) {
    impl_->xf = x;
    impl_->xfer_on = x != nullptr && on;
}
void Engine::set_xfer(bool on) {
    if (on && impl_->xf == nullptr) throw std::logic_error("set_xfer: no exchanger (STRIX_MM_XFER)");
    impl_->xfer_on = on;
}
bool Engine::xfer() const { return impl_->xfer_on; }
void Engine::set_linears_opt(bool on) {
    if (on && !impl_->linears_loaded) throw std::logic_error("set_linears_opt: load with STRIX_MM_LINEARS=load|opt");
    impl_->linears_opt = on;
}
bool Engine::linears_opt() const { return impl_->linears_opt; }
bool Engine::fused_attention() const { return impl_->attn_fused; }
void Engine::set_shared_first(bool on) {
    impl_->no_task("set_shared_first");
    impl_->shared_first = on;
}
bool Engine::shared_first() const { return impl_->shared_first; }
void Engine::set_profiling(bool on) {
    impl_->no_task("set_profiling");
    impl_->profiling = on;
}
void Engine::set_trace(bool on) { impl_->set_trace(on); }
const StepTrace& Engine::last_trace() const { return impl_->trace; }
void Engine::set_prefill_exchanger(pfx::Exchanger* x, bool on) {
    if (!impl_->layers.empty()) throw std::logic_error("set_prefill_exchanger: before load()");
    if (x != nullptr && !x->set_tx_uncached(impl_->pf_tx_uncached)) impl_->pf_tx_uncached = false;  // no uncached memory: notice
    impl_->pf = x;
    impl_->pf_on = on && x != nullptr;
}
void Engine::set_prefill_xchg(bool on) {
    impl_->no_task("set_prefill_xchg");
    if (on && impl_->pf == nullptr) throw std::logic_error("set_prefill_xchg: no prefill exchange module (STRIX_MM_PFX)");
    impl_->pf_on = on;
}
bool Engine::prefill_xchg() const { return impl_->pf_on; }
void Engine::set_prefill_mtfit(int on) {
    impl_->no_task("set_prefill_mtfit");
    moe::opt::set_prefill_mtfit(on);
}
void Engine::set_prefill_table_ring(int on) {
    impl_->no_task("set_prefill_table_ring");
    moe::opt::set_prefill_table_ring(on);
}
void Engine::set_prefill_gufuse(int on) {
    impl_->no_task("set_prefill_gufuse");
    moe::opt::set_prefill_gufuse(on);
}
void Engine::set_prefill_pipe(int value) {
    impl_->no_task("set_prefill_pipe");
    moe::opt::set_prefill_pipe(value);
}
void Engine::set_prefill_core8(int value) {
    impl_->no_task("set_prefill_core8");
    moe::opt::set_prefill_core8(value);
}

void Engine::set_prefill_router_batch(bool on) {
    impl_->no_task("set_prefill_router_batch");
    impl_->pf_router_batch = on;
}
bool Engine::prefill_router_batch() const { return impl_->pf_router_batch; }
void Engine::set_prefill_router_rows(unsigned rows) {
    impl_->no_task("set_prefill_router_rows");
    if (rows != 4 && rows != 8 && rows != 32) throw std::invalid_argument("set_prefill_router_rows: 4, 8 or 32");
    impl_->pf_router_rows = rows;
}
unsigned Engine::prefill_router_rows() const { return impl_->pf_router_rows; }
void Engine::set_prefill_interleave(bool on) {
    impl_->no_task("set_prefill_interleave");
    impl_->pf_interleave = on;
}
void Engine::set_prefill_inplace(bool on) {
    impl_->no_task("set_prefill_inplace");
    impl_->pf_inplace = on;
}
bool Engine::prefill_inplace() const { return impl_->pf_inplace; }
void Engine::set_prefill_tx_uncached(bool on) {
    impl_->no_task("set_prefill_tx_uncached");
    if (impl_->pf != nullptr && !impl_->pf->set_tx_uncached(on)) on = false;  // no uncached memory: notice, stays coherent
    impl_->pf_tx_uncached = on;
}
bool Engine::prefill_tx_uncached() const { return impl_->pf_tx_uncached; }
bool Engine::prefill_interleave() const { return impl_->pf_interleave; }
std::uint64_t Engine::state_fnv(std::uint32_t slot, std::uint64_t positions) { return impl_->state_fnv(slot, positions); }
std::uint64_t Engine::prefill_out_fnv() { return impl_->prefill_out_fnv(); }

}  // namespace strix::minimax::engine
