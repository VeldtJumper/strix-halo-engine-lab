#pragma once
// GLM-5.3-Flash engine (strix-glm5-rank), TP2 over the pair or one node (world = 1). Plan: design notes; arithmetic
// contract: design notes (approved arithmetic contract): Tier A ops bitwise to src/glm/host, EXL3 linears bitwise to strix_gemv.
//
// Rank r of 2 holds (tools/glm/split_plan.py): KDA heads 32r..32r+31 (q/k/v out-slices, conv/f_b/g_b/b/A_log/dt rows,
// o_proj in-slice), MLA heads 32r.. (q_b out-slice, kv_b rows, o_proj in-slice), I-half of every MLP (dense 6144,
// shared and all 288 experts 1024), vocab half of the head (77,440 ids); replicated: embedding, mHC, norms, q_a,
// kv_a, the indexer, f_a/g_a, the router, eh_proj. Per layer two reductions of FP32 partial rows (after o_proj, after
// the FFN), y = p0 + p1 in rank order on both ranks, so every replicated value is bit-identical on both ranks.
//
// PLUG-IN SEAMS (design notes): AttnPath, MoePath, Exchange, PrefillPath,
// and the speculative hooks on Engine (speculative decoding). Each seam has a default implementation (Tier A kernels + strix_gemv)
// that is the bitwise reference; an alternative must reproduce its outputs AND state bit for bit (arithmetic contract section 1)
// unless a declared arithmetic deviation is approved. Implementations are installed with Engine::set_*() before load() or between
// calls; the engine owns weights and state, seams own only their scratch.
#include "glm/spec/glm_spec_merge.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

namespace strix::dist {
class Communicator;
}

namespace strix::glm::engine {
namespace mem { struct LoadAdmission; }

// ---------------------------------------------------------------- weights handed to the seams (device memory)
// EXL3 linear in two forms: a prepared strix-trellis handle (for strix_gemv / strix_gemm, the Tier B reference) and
// the raw rank slice in the strix-trellis device layout (= file layout: trellis words, suh / svh FP16) for fused kernels.
struct Exl3Raw {
    const std::uint32_t* trellis = nullptr;
    const std::uint16_t* suh = nullptr;  // [input] FP16
    const std::uint16_t* svh = nullptr;  // [output] FP16
    std::uint32_t input = 0, output = 0, bits = 0;
    std::uint32_t codebook = 2;  // mul1; loader creates only mul1, also for prepared-only linears
};
struct Linear {
    void* prepared = nullptr;  // strix_weights (opaque here)
    Exl3Raw raw;               // filled when the loader keeps raw copies (STRIX_GLM_RAW_LINEARS / a seam asks for it)
};
struct Mlp {  // one expert / shared / dense MLP, rank slice: gate, up = output slices; down = input slice
    Linear gate, up, down;
};
struct KdaLayerWeights {
    Linear qkv[3];                      // q, k, v rank out-slices (heads 32r..), 4096 outputs each on TP2
    Linear o;                           // in-slice [4096 rank channels] -> 4096 partial
    const std::uint16_t* f_a = nullptr;  // F16 [128][4096] replicated
    const std::uint16_t* g_a = nullptr;  // F16 [128][4096] replicated
    const std::uint16_t* f_b = nullptr;  // F16 [rank channels][128]
    const std::uint16_t* g_b = nullptr;  // F16 [rank channels][128]
    const std::uint16_t* b = nullptr;    // F16 [rank heads][4096]
    const std::uint16_t* conv = nullptr; // BF16 [3 * rank channels][4] (q | k | v)
    const float* A_log = nullptr;        // [rank heads]
    const float* dt_bias = nullptr;      // [rank channels]
    const std::uint16_t* o_norm = nullptr;  // BF16 [128]
    std::uint32_t heads = 0;             // rank heads (32 on TP2, 64 on one node)
    // Optional R1b v5 copies; originals retained for identity switching / multi-row calls.
    const std::uint16_t* f_a_packed = nullptr;
    const std::uint16_t* g_a_packed = nullptr;
    const std::uint16_t* b_packed = nullptr;
};
struct DsaLayerWeights {
    Linear q_a, kv_a, q_b, o, wq_b;      // q_a, kv_a, wq_b replicated; q_b out-slice; o in-slice
    const std::uint16_t* q_a_norm = nullptr;   // BF16 [1536]
    const std::uint16_t* kv_a_norm = nullptr;  // BF16 [512]
    const std::uint16_t* kv_b = nullptr;       // F16 rank rows [heads * 512][512] (K rows h*512+d, V rows h*512+256+d)
    const std::uint16_t* wk = nullptr;         // F16 [128][4096]
    const std::uint16_t* gate = nullptr;       // F16 [128][4096] (index_kpool_compress_gate)
    const std::uint16_t* weights_proj = nullptr;  // F16 [32][4096]
    const std::uint16_t* k_norm_w = nullptr;   // F16 [128]
    const std::uint16_t* k_norm_b = nullptr;   // F16 [128]
    const float* ape = nullptr;                // [4][128]
    std::uint32_t heads = 0;                   // rank MLA heads
};
struct MoeLayerWeights {
    const std::uint16_t* router = nullptr;  // F16 [288][4096] replicated
    const float* bias = nullptr;            // [288] e_score_correction_bias
    std::vector<Mlp> experts;               // 288 rank slices (I_local 1024)
    const Mlp* experts_d = nullptr;         // the same array in device memory (for one-launch kernels)
    Mlp shared;                             // rank slice
    const std::uint16_t* router_packed = nullptr;  // optional R1b v5 copy
};

// ---------------------------------------------------------------- per-slot state (device memory, owned by the engine)
struct KdaState {  // one KDA layer, all slots
    float* S = nullptr;     // [slots][rank heads][128][128] fp32
    float* conv = nullptr;  // [slots][3 * rank channels][3] fp32, oldest first
    // Speculative verify (Options::spec, design notes §1): the deferred-replay inputs of each slot's pending rows,
    // two disjoint sets (ping-pong) per slot: qkv [slots][2][8][3C] (raw pre-conv), scr [slots][2][8][H][4][128],
    // beta [slots][2][8][H]. Engine-owned; a verify reads set SpecRows::cur and writes set 1 - cur.
    float* rep_qkv = nullptr;
    float* rep_scr = nullptr;
    float* rep_beta = nullptr;
};
struct DsaState {  // one DSA layer, all slots: capacity positions per slot (arithmetic contract 1.4)
    std::uint16_t* latent = nullptr;  // [slots][cap][512] bf16
    std::uint16_t* ikey = nullptr;    // [slots][cap][128] bf16
    std::uint16_t* igate = nullptr;   // [slots][cap][128] bf16
    float* pooled = nullptr;          // [slots][cap / 4][128] fp32
    std::uint64_t cap = 0;
};

// One call row: a sequence slot, the token at `pos`, and its position. Rows of one slot in one call must have
// consecutive ascending positions (a verify window); rows of different slots are independent.
struct Row {
    std::uint32_t slot = 0, token = 0;
    std::uint64_t pos = 0;
};
// Runs a prepared EXL3 linear through the engine's strix-trellis context: out[rows][N] = in[rows][I] @ W
// (strix_gemv for rows <= 8, strix_gemm above) - the Tier B reference every fused kernel must reproduce bit for bit.
class LinearRunner {
public:
    virtual ~LinearRunner() = default;
    virtual void run(const Linear& l, const float* in, float* out, unsigned rows, hipStream_t s) const = 0;
    // Prefill identity glm-pf-wmma-v1 (design notes section 1.1, deviation D10): strix_gemm_pinned with
    // a caller-pinned K split, WMMA for every rows >= 1, so a row's bits do not depend on the rows of the call.
    virtual void run_pinned(const Linear& l, const float* in, float* out, unsigned rows, unsigned splits, hipStream_t s) const {
        (void)l, (void)in, (void)out, (void)rows, (void)splits, (void)s;
        throw std::logic_error("LinearRunner::run_pinned not available");
    }
    // Grow the shared matmul workspace to >= bytes now (so no call allocates later); default: no workspace to reserve.
    virtual void reserve(std::uint64_t workspace_bytes) const { (void)workspace_bytes; }
};
// One slot's window inside a multi-slot verify call: rows row0 .. row0 + n - 1 of the call (consecutive positions).
struct SpecSeg {
    std::uint32_t slot = 0;
    unsigned n_rep = 0, cur = 0;      // as SpecRows, for this slot
    unsigned row0 = 0, n = 0;
};
// Speculative verify call (design notes §1): the call's rows are R <= 8 consecutive positions of ONE slot, or with
// `seg` (Engine::verify_multi, nseg >= 2) the windows of several slots, sorted by slot; slot / n_rep / cur then
// describe seg[0].
struct SpecRows {
    std::uint32_t slot = 0;
    unsigned n_rep = 0;               // pending rows of the slot replayed into the committed KDA state by this call
    unsigned cur = 0;                 // replay set read by this call; its new rows go to set 1 - cur
    unsigned* row_flag = nullptr;     // device [R]: per-row C1-C5 (new rows' C2 included)
    unsigned* replay_flag = nullptr;  // device [1]: C2 of the replayed (committed) rows
    const SpecSeg* seg = nullptr;     // host [nseg]: multi-slot verify windows (nullptr: one slot)
    unsigned nseg = 0;
};
// What a seam sees of the current call (host + device copies, valid during the call).
struct CallCtx {
    const LinearRunner* lin = nullptr;
    std::span<const Row> rows;          // host
    const std::uint32_t* slot_d = nullptr;  // [rows] device
    const std::uint64_t* pos_d = nullptr;   // [rows] device
    unsigned* flag = nullptr;           // device non-finite flag (C1-C5)
    hipStream_t stream = nullptr;
    bool decode = true;                 // false: prefill chunk
    const SpecRows* spec = nullptr;     // non-null: speculative verify rows; checkpoints go to spec->row_flag[r]
};

// ---------------------------------------------------------------- seams
// Attention of one layer: x = normed layer input [R][4096] (replicated, identical on both ranks) -> y = this rank's
// partial [R][4096] (after o_proj, before the reduction). Updates the layer's state for every row.
class AttnPath {
public:
    virtual ~AttnPath() = default;
    virtual void kda(const KdaLayerWeights& w, KdaState& st, const CallCtx& c, const float* x, float* y) = 0;
    virtual void dsa(const DsaLayerWeights& w, DsaState& st, const CallCtx& c, const float* x, float* y) = 0;
    // Speculative verify calls with the windows of several slots (SpecRows::seg); false: one slot per verify call.
    virtual bool supports_multi_slot_verify() const { return false; }
};
// FFN of one MoE layer: x = normed [R][4096]; route ids [R][8] ascending + weights [R][8] (computed by the engine,
// R5/R14) -> y = sum over the row's experts in ascending id of fmaf(w, expert(x)) + shared(x), this rank's partial.
class MoePath {
public:
    virtual ~MoePath() = default;
    virtual void forward(const MoeLayerWeights& w, const CallCtx& c, const float* x, const std::uint32_t* ids,
                         const float* wts, float* y) = 0;
    // Per-layer tables a seam builds lazily on its first forward of `w`: bytes to admit before building them (upper
    // bound of the device allocation plus the host staging held meanwhile; 0 once built), and building them now (Engine
    // prefill install and set_moe with a prefill installed: a prefill allocates no device memory).
    virtual std::uint64_t warm_bytes(const MoeLayerWeights& w) const {
        (void)w;
        return 0;
    }
    virtual void warm(const MoeLayerWeights& w) { (void)w; }
};
// TP reduction: p [R][4096] in place -> p0 + p1 (rank order) on both ranks; head argmax merge. world = 1: identity.
class Exchange {
public:
    virtual ~Exchange() = default;
    virtual void reduce(float* p, unsigned rows, hipStream_t s) = 0;
    // One reduction in flight, on one stream: start, enqueue independent work, finish.
    // The default is synchronous; implementations may overlap transport, preserving reduce() arithmetic.
    virtual void start_reduce(float* p, unsigned rows, hipStream_t s) { reduce(p, rows, s); }
    virtual void finish_reduce(hipStream_t /*s*/) {}
    // (value, id) of each row's local argmax -> the global one on both ranks (larger value, ties -> lower id).
    virtual void argmax(float* value, std::uint32_t* id, unsigned rows, hipStream_t s) = 0;
    // Completion point of a call that ends without argmax (prefill chunk, gate_residual): returns once every exchange
    // posted on `s` is complete and verified, throws otherwise. Asynchronous exchanges (XferExchange) seal
    // their last deferred CRC here; synchronous ones have nothing to do.
    virtual void complete(hipStream_t /*s*/) {}
    // Optional fused boundary in place of reduce(part) + dev::hc_expand(X, mix, part, Xout, flag) with ONE flag for the call
    // (XferExchange: send, then R17 + R7 in one kernel with the in-kernel wait; same bits).
    // Returns false when not provided: the engine then runs reduce + hc_expand.
    virtual bool fused_boundary(const float* /*part*/, const float* /*X*/, const float* /*mix*/, float* /*Xout*/,
                                unsigned* /*flag*/, unsigned /*rows*/, hipStream_t /*s*/) {
        return false;
    }
    // Row-aware hook: existing seams retain the single-flag contract.
    virtual bool fused_boundary_rows(const float* part, const float* X, const float* mix, float* Xout,
                                     unsigned* flags, unsigned rows, unsigned flag_stride, hipStream_t s) {
        return flag_stride == 0 && fused_boundary(part, X, mix, Xout, flags, rows, s);
    }
    // Speculative control merge (design notes §2), host side, one call per verify / MTP step: per row R18 on
    // (value, id) and OR on flag; OR on ctl.replay_flag; rank 0's cost table; MTP confidence (max M, sum Z) merged as
    // M = max(M0, M1), Z = Z0 e^(M0-M) + Z1 e^(M1-M); a (cycle, step, kind) mismatch between ranks throws on both.
    using MergeRow = spec::MergeRow;
    using MergeCtl = spec::MergeCtl;
    virtual void merge(MergeRow* rows, unsigned n, MergeCtl& ctl, hipStream_t s);
};
// Prefill of one layer over a chunk of rows of ONE slot at consecutive positions. Default: row-by-row decode order
// (the --reference identity). A chunked implementation must reproduce it bit for bit or carry a declared deviation.
class PrefillPath {
public:
    virtual ~PrefillPath() = default;
    virtual void kda(const KdaLayerWeights& w, KdaState& st, const CallCtx& c, const float* x, float* y) = 0;
    virtual void dsa(const DsaLayerWeights& w, DsaState& st, const CallCtx& c, const float* x, float* y) = 0;
    virtual void moe(const MoeLayerWeights& w, const CallCtx& c, const float* x, const std::uint32_t* ids,
                     const float* wts, float* y) = 0;
    // Dense FFN of layers 0-2 over the chunk (rank I-slice): y = down(swiglu(gate(x), up(x))), this rank's partial.
    virtual void dense(const Mlp& m, const CallCtx& c, const float* x, unsigned I, float* y) {
        (void)m, (void)c, (void)x, (void)I, (void)y;
        throw std::logic_error("PrefillPath::dense not implemented by this seam");
    }
    // Exact under consecutive half-row calls on one stream (including recurrence / cache commits).
    virtual bool supports_pipeline() const { return false; }
    // Exact row-invariant linear for bootstrap projections. Unsupported paths fail closed.
    virtual bool exact_linears() const { return false; }
    virtual void linear_exact(const Linear& l, const CallCtx& c, const float* x, float* y,
                              unsigned K, unsigned N) const {
        (void)l, (void)c, (void)x, (void)y, (void)K, (void)N;
        throw std::logic_error("PrefillPath::linear_exact not implemented by this seam");
    }
    // Largest LinearRunner workspace any call of this path needs; Engine::set_prefill reserves it up front.
    virtual std::uint64_t linear_workspace_bytes() const { return 0; }
    // Free-memory margin (device and host) the reservation must leave; the path's own admission uses the same one.
    virtual std::uint64_t admission_margin() const { return std::uint64_t{1} << 30; }
    // true: the engine runs the chunk's routed + shared experts through its own MoePath one row at a time (the decode
    // identity; needed when experts are loaded raw-only for the fast MoE seam) instead of calling moe().
    virtual bool moe_by_row() const { return false; }
    // Rows per MoePath call when moe_by_row (1..8): a row's bits do not depend on it (P1 batch invariance R = 1..8).
    virtual unsigned moe_row_block() const { return 1; }
    // true: moe() runs the shared expert through LinearRunner, so Engine::set_prefill gives every raw-only shared
    // expert a prepared handle (from its raw parts, admitted against free memory) before installing the path.
    virtual bool needs_prepared_shared() const { return false; }
};

// ---------------------------------------------------------------- engine
enum class IdentityMode { kR1, kR1b, kR1bAll };
inline const char* identity_mode_name(IdentityMode mode) {
    switch (mode) {
        case IdentityMode::kR1: return "r1";
        case IdentityMode::kR1b: return "r1b";
        case IdentityMode::kR1bAll: return "r1b-all";
    }
    throw std::invalid_argument("invalid GLM identity mode");
}
struct Options {
    std::string model_dir;
    std::uint32_t rank = 0, world = 2;   // world 1 = one node (all heads / full I / full vocab)
    std::uint32_t slots = 1;
    std::uint64_t capacity = 8192;       // positions per slot (multiple of 4)
    bool auto_rows = false;             // recheck admission after communication growth, before weights
    unsigned exchange_rows = 0;         // 0 uses max_rows; startup may grow Comm buffers after selection
    std::uint32_t max_rows = 1024;       // rows per forward call (prefill chunk)
    std::uint32_t layers = 45;           // first `layers` main layers only (G3 uses 6); the head needs all 45
    bool mtp = true;                     // load layer 45 (MTP) + its DSA state
    bool raw_linears = false;            // keep raw EXL3 copies of every linear (fused seams need them)
    bool spec = false;                   // allocate the speculative-verify buffers (KDA replay sets, row flags)
    IdentityMode identity = IdentityMode::kR1b;  // Default: R1 prompts, R1b generated forwards.
    // STRIX_GLM_IDENTITY=r1|r1b|r1b-all overrides this option at construction.
    bool external_exchange = false;      // world 2 without a Communicator: the caller installs its Exchange
                                         // (set_exchange) before the first forward (e.g. XferExchange)
    bool decode_xfer = true;             // TP2: Plain/Verify/MTP with 1..8 rows can borrow its transport.
                                         // STRIX_GLM_DECODE_XCH=comm|xfer overrides this option on both ranks.
                                         // STRIX_GLM_SPEC_XCH=comm keeps Verify/MTP on Comm (default xfer).
    unsigned pf_pipeline = 1;           // 1: current schedule; 2: half-row whole-layer overlap
                                         // STRIX_GLM_PF_PIPELINE=1|2 overrides this option on both ranks.
};
struct LoadReport {
    double seconds = 0, gib_read = 0, device_gib = 0;
    bool direct_io = false;
};
// Device time per component, summed over profiled calls (ms), from HIP events at every segment boundary of the
// engine stream (set_profiling(true)); exchange includes the host-side wait of a synchronous Exchange.
struct Profile {
    double embed = 0, hc = 0, attn = 0, exchange = 0, ffn = 0, head = 0;  // attn = kda + dsa, ffn = moe + dense
    double kda = 0, dsa = 0, moe = 0, dense = 0;
    std::uint64_t calls = 0;
};

class Engine {
public:
    Engine(Options options, dist::Communicator* comm);  // comm may be null when world == 1
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Optional startup selector runs after calibration/planning, before load admission or weight/row allocation.
    // Receives the calibrated load terms and headroom; may reduce max_rows, causing a fresh plan.
    // The caller must coordinate the result across ranks. Comm row buffers grow only after selection.
    LoadReport load(const std::function<unsigned(const mem::LoadAdmission&)>& select_max_rows = {});
    // Positions [pos0, pos0 + n) of `slot`; returns the greedy next token after the last row (both ranks).
    // Default kR1b runs all prompt forwards in R1, restoring R1b on return or exception.
    // kR1bAll requires the reference path and runs prompts in R1b; kR1 uses R1 throughout.
    // With the reference PrefillPath (default) the prompt runs as decode steps row by row. With another PrefillPath
    // (prefill: set_prefill(pf::make_prefill_path(...))) it runs in chunks of Options::max_rows rows through that seam;
    // the head runs on the last row only, and the slot's MTP cache is not filled (mtp() refuses the slot until reset).
    std::uint32_t prefill(std::uint32_t slot, std::span<const std::uint32_t> tokens, std::uint64_t pos0);
    // One decode step of up to 8 rows; returns each row's greedy token (identical on both ranks).
    std::vector<std::uint32_t> step(std::span<const Row> rows);
    // Reset a slot (sequence start): token caches emptied, KDA / conv state zeroed.
    void reset(std::uint32_t slot);

    // ---- seams
    // Default R1b uses R1 inside prefill; R1bAll also uses R1b for reference prompts.
    void set_identity_mode(IdentityMode mode);
    [[nodiscard]] IdentityMode identity_mode() const;
    // Compatibility: true -> R1bAll, false -> R1; does not flush state.
    void set_identity(bool r1b);
    [[nodiscard]] bool identity_r1b() const;  // Generated forwards use R1b (both kR1b and kR1bAll).
    // Route log: router top-8 ids of row 0 per layer of the last forward ([kLayers + 1][8], 0xFFFFFFFF = no MoE).
    void set_route_log(bool on);
    // G3-style teacher forcing (R = 1, slot 0, positions 0..T-1, caller resets) returning the residual after EVERY layer
    // < n_layers: [T][n_layers][4][4096] (one load for per-layer comparisons).
    [[nodiscard]] std::vector<float> gate_residual_layers(std::span<const std::uint32_t> tokens, unsigned n_layers);
    [[nodiscard]] std::vector<std::uint32_t> route_log() const;
    void set_attn(std::shared_ptr<AttnPath> p);
    void set_moe(std::shared_ptr<MoePath> p);
    // Explicit seam override disables automatic decode routing (including when restoring exchange() for an A/B).
    // Both ranks must install the same policy between forwards; the outgoing exchange is completed before switching.
    void set_exchange(std::shared_ptr<Exchange> p);
    [[nodiscard]] std::shared_ptr<Exchange> exchange() const;  // installed fallback, independent of the last forward
    [[nodiscard]] bool decode_xfer_enabled() const;
    [[nodiscard]] unsigned prefill_pipeline() const;
    void set_prefill(std::shared_ptr<PrefillPath> p);
    // Bytes the installed MoE seam would still allocate lazily (sum of warm_bytes over the loaded MoE layers); 0 after
    // a prefill install or a set_moe with a prefill installed (test hook: a prefill must not allocate).
    std::uint64_t moe_cold_bytes() const;
    // Test hook (D10 noise reference, measurement test 1 nulls): after layer `layer` of the forward that holds position
    // `pos`, add 1 ulp (+1 to the bit pattern) to residual stream 0, element `elem`, of that row. layer < 0 disables.
    void set_null_flip(int layer, std::uint64_t pos, unsigned elem);

    // ---- speculative decoding (deferred replay; design notes v3, approved arithmetic contract; needs Options::spec)
    // Verify rows tokens[0..R) of `slot` at positions p..p+R-1 (R <= 8, consecutive). Returns each row's merged argmax
    // and merged flag word; checkpoints never throw here (accept_greedy decides). h rows: last_hidden().
    struct VerifyOut {
        std::vector<std::uint32_t> id, flag;
        bool replay_flag = false;
    };
    VerifyOut verify(std::uint32_t slot, std::span<const std::uint32_t> tokens, std::uint64_t p, Exchange::MergeCtl& ctl);
    // The verify windows of several slots in ONE call (served batched self-speculation): rows sorted by slot, each slot's
    // rows at consecutive positions (its anchor, then its drafts), R <= 8 in total. Per-row merged ids / flags as
    // verify(); each slot then commits its accepted rows with commit(slot, ...). One slot == verify(). Several slots
    // need an AttnPath with supports_multi_slot_verify() (verify_multi_supported()); otherwise this throws.
    VerifyOut verify_multi(std::span<const Row> rows, Exchange::MergeCtl& ctl);
    [[nodiscard]] bool verify_multi_supported() const;
    // Commit the first `rows` rows of the last verify of `slot` (verify at p): DSA frontier p + rows, KDA pending =
    // those rows (ping-pong swap), MTP cache truncated to p. Not called on abort.
    void commit(std::uint32_t slot, unsigned rows, std::uint64_t p);
    // Apply the slot's pending KDA rows to its stored state (n_new = 0 replay); before mode changes, reset, state_bytes.
    void flush(std::uint32_t slot);
    // The final-normed hidden h_p (mean of the 4 streams, then norm) of the last step's / verify's rows, device
    // [rows][4096]: the MTP input (design notes); valid until the next call on the engine.
    [[nodiscard]] const float* last_hidden() const;
    // The rank's logits slice of the last step's / verify's rows, device [rows][vocab_slice()] (gates: per-row logits
    // of spec vs plain); valid until the next call.
    [[nodiscard]] const float* last_logits() const;
    [[nodiscard]] std::uint32_t vocab_slice() const;
    // One vocabulary slice, allocated/planned with MTP: preserve bootstrap trunk logits across the seed drafter.
    void save_trunk_logits();
    void restore_trunk_logits();
    // The MTP layer (45, design notes) on n <= 8 pairs (h_j, x_{j+1}) of `slot` at MTP cache indices mtp_pos0..:
    // h = device [n][4096] (trunk last_hidden rows or the previous step's mtp_hidden), tokens = x_{j+1}. Writes the
    // pairs into the MTP DSA cache (its frontier is implicit: later passes overwrite) and returns each row's merged
    // greedy draft, the last row's merged confidence (softmax probability of its argmax) and a drafter non-finite flag
    // (drafts are then discarded; output exactness never depends on the drafter). h' rows: mtp_hidden().
    // v1 selects the indexer pools afresh for every row (llama.cpp variant; vLLM/SGLang index sharing to follow).
    struct MtpOut {
        std::vector<std::uint32_t> id;
        float confidence = 0;
        bool flagged = false;
    };
    MtpOut mtp(std::uint32_t slot, const float* h, std::span<const std::uint32_t> tokens, std::uint64_t mtp_pos0,
               Exchange::MergeCtl& ctl);
    [[nodiscard]] const float* mtp_hidden() const;
    // The same h' rows of the last mtp() call of `slot`, kept per slot ([<= 8][4096] device) until that slot's next
    // mtp() call: drafter chains of several slots interleave (served batched self-speculation).
    [[nodiscard]] const float* mtp_hidden(std::uint32_t slot) const;
    // Exact chunked R1 prompt bootstrap, including layer 45. hp is device [4096], valid until the next call.
    // Requires an exact PrefillPath, all trunk layers and MTP; restores the generated identity on every exit.
    struct MtpPrefillOut {
        std::uint32_t y0 = 0;
        MtpOut mtp;
        const float* hp = nullptr;
    };
    MtpPrefillOut prefill_mtp(std::uint32_t slot, std::span<const std::uint32_t> tokens, std::uint64_t pos0,
                             Exchange::MergeCtl& ctl);

    // ---- gates (G3 / G4 / replay)
    // Teacher-forced run of the first `layers` layers over `tokens` (slot 0, positions 0..): returns the 4-stream
    // residual [T][4][4096] after layer layers-1; with `dump` set, every op's output is kept for the op-level checks.
    std::vector<float> gate_residual(std::span<const std::uint32_t> tokens, unsigned layers, bool dump = false);
    // Copies of every persistent state buffer of `slot` (KDA S/conv, latents, indexer key/gate, pooled), for replay.
    std::vector<std::uint8_t> state_bytes(std::uint32_t slot) const;
    // The same, frontier-bounded (spec gates): DSA token caches of positions < frontier and pooled keys of the pools
    // closed below it (positions written by rejected verify rows beyond the frontier are excluded); KDA S / conv in full
    // (call flush first).
    std::vector<std::uint8_t> state_bytes_upto(std::uint32_t slot, std::uint64_t frontier) const;
    // Inverse of state_bytes_upto (same capacity / layers / mtp): reset(slot), then every segment written back, so the
    // slot continues at `frontier` as if it had run there (G4 runs across turns; gate SR). Throws on a size mismatch.
    void restore_state(std::uint32_t slot, std::span<const std::uint8_t> bytes, std::uint64_t frontier);

    void set_profiling(bool on);
    [[nodiscard]] const Profile& profile() const;
    [[nodiscard]] std::uint32_t rank() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The engine installs its default AttnPath / MoePath (Tier A kernels + strix_gemv, one row at a time: the bitwise
// references) and the exchange for its world size; these two are available standalone.
std::shared_ptr<PrefillPath> reference_prefill();  // row-by-row decode order
std::shared_ptr<Exchange> local_exchange();        // world = 1

}  // namespace strix::glm::engine
