#pragma once
#include "strix/file_snapshot.hpp"
// MiniMax M3.1 Flash engine, TP2 over the pair (fork of the DeepSeek V4.1 TP2 engine; see qualification notes).
// Rank r holds q heads 32r..32r+31, KV / index-q heads {2r, 2r+1}, half of every MLP intermediate (dense, shared, all 128
// experts), and its vocabulary half of the head; router, index key and embedding are replicated. Per layer two exchanges
// of FP32 partial rows (after o_proj and after the FFN), each followed by the fused post-exchange kernel (fixed rank
// order sum, Gemma norm, router top-4), so the residual is bit-identical on both ranks. Weights are loaded once and stay
// resident (O_DIRECT reads through one pinned staging buffer).
#include "strix/dist/communicator.hpp"
#include "strix/minimax/prefill_types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace strix::minimax::xfer {
class Exchanger;
}
namespace strix::minimax::pfx {
class Exchanger;
}

namespace strix::minimax::dspark {
class DeviceDrafter;
}

namespace strix::minimax::engine {

struct Options {
    std::string model_dir;
    std::uint32_t rank = 0;
    std::uint32_t slots = 1;        // independent sequences (decode rows), each with its own KV / index-key cache
    std::uint64_t capacity = 8256;  // positions per slot
    std::uint32_t max_rows = 4096;  // rows per forward call (prefill chunk / gate); sizes the activations
};

struct LoadReport {
    double seconds = 0, gib_read = 0, device_gib = 0;
    bool direct_io = false;
};

// Per-phase times (seconds, synchronised) when STRIX_MM_PROFILE=1; zero otherwise.
struct Profile {
    double embed = 0, attn_proj = 0, msa = 0, o_proj = 0, exchange = 0, ffn = 0, route_host = 0, head = 0;
    double post = 0;    // post-exchange kernel (sum, norm, router), split out of `exchange`
    double dense = 0;   // FFN of the dense layers (L0-2), split out of `ffn` (which is then the MoE layers only)
    std::uint64_t calls = 0;
    // prefill MoE: (layer, expert) pairs by routed rows: 0, 1-8, 9-16, 17-32, 33-64, 65-128, 129-256, > 256
    std::array<std::uint64_t, 8> rows_per_expert{};
};

// Step trace (set_trace, decode steps only): GPU-timeline events at the phase boundaries of every layer, read after the
// step's own final synchronisation (nothing is synchronised inside the step), and the routed expert ids of every MoE
// layer (the router's optional id output). The kernels, their order and the bits are the same as with the trace off.
struct StepTrace {
    static constexpr unsigned kPhases = 6;  // attn_proj (+ taps), msa, o_proj, x_attn, ffn, x_ffn (x_*: wait + post-exchange)
    std::uint32_t rows = 0;
    std::vector<float> layer_ms;        // [layer][kPhases]
    float head_ms = 0, total_ms = 0;    // head + argmax exchange; first layer start -> head end
    std::vector<std::uint32_t> routes;  // [layer][rows][4] expert ids (zeros for the dense layers)
};

class Engine {
public:
    using PrefillTapSink = std::function<void(const float* taps, std::size_t n, std::uint32_t slot, std::uint64_t pos0)>;
    Engine(Options options, dist::Communicator& comm);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    LoadReport load();
    // Positions [pos0, pos0 + n) of `slot` (prefill, chunked by max_rows); returns the greedy next token after the last
    // row (identical on both ranks).
    std::uint32_t prefill(std::uint32_t slot, std::span<const std::uint32_t> tokens, std::uint64_t pos0);
    // One decode step of up to 8 rows; rows of one slot must have consecutive ascending positions (a verify window:
    // each row sees the rows before it, exactly as if they had been decoded one by one). Returns each row's greedy token.
    std::vector<std::uint32_t> step(std::span<const Row> rows);
    // Resumable prefill (serve scheduler, qualification notes): prefill()'s chunks / groups paused between two layers.
    // begin opens it (one active plus one explicitly parked parent); resume runs at most `layers` (>= 1) more layers and returns true with *next = the
    // greedy token once complete. The slot's caches and the token are bit-identical to prefill() (the same kernel calls,
    // rows and arguments); step() of OTHER slots may run between two resumes, on both ranks in the same order. cancel
    // abandons it (the slot's caches are then partial). While open: no prefill / gate / copy_slot / state rows / set_taps.
    void prefill_begin(std::uint32_t slot, std::span<const std::uint32_t> tokens, std::uint64_t pos0);
    bool prefill_resume(unsigned layers, std::uint32_t* next);
    void prefill_cancel();
    using PfTaskId = std::uint64_t;
    using PfProgress = strix::minimax::engine::PfProgress;
    struct PfResult {
        PfTaskId id = 0;
        std::uint64_t offset = 0, rows = 0;
        std::uint32_t bank = 0, member = 0, next = 0;
    };
    [[nodiscard]] PfResult prefill_result() const;  // transient; invalidated by a bank switch
    // Startup-only reservation, after the optional drafter. Returns the both-rank capacity.
    bool reserve_nested(bool taps);
    // Qualification-only read-only boundary digest of every live group activation and valid tap row.
    std::uint64_t prefill_activation_fnv(PfTaskId id);
    void set_prefill_check_outputs(bool on);
    [[nodiscard]] std::uint64_t prefill_check_output_fnv() const;
    [[nodiscard]] unsigned nested_rows() const;
    [[nodiscard]] const char* nested_reason() const;
    [[nodiscard]] PfProgress prefill_progress(bool parked = false) const;
    void prefill_begin_task(PfTaskId id, PfTaskId parent, std::uint64_t generation,
                            std::uint32_t slot, std::span<const std::uint32_t> tokens, PrefillTapSink sink);
    void prefill_park(PfTaskId id);
    void prefill_resume_parked(PfTaskId id);
    void prefill_cancel(PfTaskId id);
    [[nodiscard]] bool prefill_active() const;

    [[nodiscard]] bool prefill_open() const;
    [[nodiscard]] const char* prefill_unlimited_reason() const;  // empty, profiling or fp32_prefill
    [[nodiscard]] unsigned prefill_layer() const;           // next layer of the open chunk / group (0: at a chunk boundary)
    [[nodiscard]] std::uint64_t prefill_unit_rows() const;  // rows of the open chunk / group (0: none open)
    [[nodiscard]] std::uint64_t prefill_rows_left() const;  // rows from the open chunk / group to the end (0: none open)
    // Teacher-forced gate: the first `layers` layers over `tokens` (positions 0.., slot 0) in ONE call with the G4
    // harness's kernels and call order; returns the residual [T x 6144] after layer `layers - 1`.
    std::vector<float> gate_residual(std::span<const std::uint32_t> tokens, unsigned layers);
    // Full-model teacher-forced logits of the scored rows, gathered on rank 0 as the G4 harness does ([S x vocab] on
    // rank 0, empty on rank 1).
    std::vector<float> gate_logits(std::span<const std::uint32_t> tokens, std::span<const std::uint32_t> rows);
    // A value from rank 0, delivered to both ranks (control channel).
    std::uint32_t agree(std::uint32_t value);
    // Exchange one value with the other rank; returns the peer's value.
    std::uint32_t peer(std::uint32_t value);
    // This rank's vocabulary half of the logits of the last step's rows ([rows][vn], vocab_range()), for checks.
    [[nodiscard]] std::vector<float> last_logits(std::size_t rows) const;
    // Control channel: `bytes` (1..65536) of host memory sent, the peer's `bytes` received (both ranks, same size).
    void exchange_bytes(const void* out, void* in, std::size_t bytes);
    void abort(const std::string& reason);
    // Data channel for device buffers of any size (both ranks, same size): `local` (ordered after its producer on
    // `stream`) to the peer, the peer's bytes into `remote` (enqueued on `stream`). One control-tagged comm.exchange.
    void exchange_device(const void* local, void* remote, std::size_t bytes, void* stream);
    [[nodiscard]] std::uint64_t capacity() const;
    [[nodiscard]] std::uint32_t slots() const;
    // Copy positions [0, positions) of every layer's caches from slot `src` to slot `dst` (no exchange).
    void copy_slot(std::uint32_t dst, std::uint32_t src, std::uint64_t positions);
    // Prompt-state hand-off (RTX PRO prefill): this rank's cache rows of positions [pos0, pos0 + n)
    // of `slot`, every layer, as one host blob: for L = 0..59: K [n][256] BF16, V [n][256] BF16, index key [n][128] BF16
    // (rank r = KV heads 2r, 2r + 1; the index key is replicated). n * 60 * 1280 bytes. No exchange.
    static constexpr std::size_t kStateRowBytes = 60 * 1280;
    void export_rows(std::uint32_t slot, std::uint64_t pos0, std::uint64_t n, void* host) const;
    void import_rows(std::uint32_t slot, std::uint64_t pos0, std::uint64_t n, const void* host);
    // One layer's block of that blob (K [n][256], V [n][256], index key [n][128]): n * 1280 bytes.
    void import_layer(std::uint32_t slot, unsigned layer, std::uint64_t pos0, std::uint64_t n, const void* host);

    // ---- speculative decoding hooks (DSpark, src/minimax/dspark) ----
    // Draft taps: with taps on, every forward call keeps the residual after target layers 3, 17, 31, 45, 59 of its rows
    // in a device buffer [rows][5][6144] f32 (no exchange: the residual is replicated).
    void set_taps(bool on);
    // The taps of the last step() (device, valid until the next call).
    [[nodiscard]] const float* last_taps() const;
    // Called after every prefill chunk with its taps (device), slot and first position; empty = none.
    void set_prefill_tap_sink(PrefillTapSink sink);
    // Device pointer of the replicated BF16 embedding row of `token`.
    [[nodiscard]] const std::uint16_t* embed_row(std::uint32_t token) const;
    // This rank's vocabulary range: first id and count.
    [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> vocab_range() const;
    // This rank's head on n <= 8 device rows -> device logits [n][count] (valid until the next call).
    const float* head_logits(const float* rows, std::size_t n);
    // Exchange two words with the other rank (control channel); returns the peer's.
    std::array<std::uint32_t, 2> peer2(std::uint32_t a, std::uint32_t b);
    // Run the drafter's 35 linears through this engine's strix-trellis path on an EXL3 requant of the draft (dir with
    // model.safetensors.index.json and <name>.{trellis,suh,svh,mul1}; ref/quant_draft.py). Whole linears on both ranks
    // (the draft is replicated); gemv for <= 8 rows, gemm above. Draft numerics change acceptance only.
    void attach_draft_exl3(const std::string& dir, dspark::DeviceDrafter& drafter);
    // Draft-only head (DSpark): this rank's vocabulary slice of an EXL3 requant of the lm_head (dir with
    // model.safetensors.index.json and language_model.lm_head.{trellis,suh,svh}; ref/quant_head.py) replaces the target's
    // head in head_logits() (the drafter's call). Decode and verify keep the target's head, so outputs stay exact; the
    // draft's acceptance can change. STRIX_MM_DRAFT_HEAD=lib (strix_gemv, default) | sched (enqueue_linears_scheduled).
    void attach_draft_head(const std::string& dir);
    // attach_draft_head may be called several times (in-process comparisons): set_draft_head(i) selects the i-th attached
    // head (default: the last one), -1 the target's head.
    [[nodiscard]] int draft_heads() const;
    void set_draft_head(int index);
    // With STRIX_MM_DRAFT_LIN=load: switch the drafter's block linears between enqueue_linears (true) and strix_gemv (same bits).
    void set_draft_linears_opt(bool on);

    // Runtime switches (both ranks, between calls), for ABA timing within one load.
    void set_fused_staging(bool on);
    [[nodiscard]] bool fused_staging() const;
    void set_fused_attention(bool on);  // STRIX_MM_ATTN=fused: decode MSA in the fused batched launches (same bits)
    [[nodiscard]] bool fused_attention() const;
    // STRIX_MM_LINEARS=opt|load (load time: raw copies of the decode linears, ~3 GB): q+k+v, o and the dense MLP of
    // decode steps on src/minimax/moe enqueue_linears (strix_gemv bits); switchable at runtime once loaded.
    void set_linears_opt(bool on);
    [[nodiscard]] bool linears_opt() const;
    // exchange module for decode steps (layer exchanges + the head's argmax); it shares the
    // transport with the communicator (prefill and control stay there; each step drains the module).
    void set_exchanger(xfer::Exchanger* x, bool on);
    void set_xfer(bool on);
    [[nodiscard]] bool xfer() const;
    // STRIX_MM_ATTN_PREFILL=fused: prefill / gate MSA in the fused launches, windows of 16 rows (same bits as K1-K4).
    void set_fused_prefill_attention(bool on);
    void set_prefill_index_tile(bool on);  // STRIX_MM_PF_IDX_TILE, default off; implies batched prefill
    [[nodiscard]] bool prefill_index_tile() const;
    void set_prefill_msa_batch(bool on);  // STRIX_MM_PF_MSA_BATCH, default off; prefill only
    [[nodiscard]] bool prefill_msa_batch() const;
    void set_prefill_msa_rework(unsigned mode);  // STRIX_MM_PF_MSA_REWORK=0..7; prefill only, default 7
    [[nodiscard]] unsigned prefill_msa_rework() const;
    void set_prefill_attend4(bool on);  // STRIX_MM_PF_ATTEND4, default off; prefill only
    [[nodiscard]] bool prefill_attend4() const;
    [[nodiscard]] bool fused_prefill_attention() const;
    // STRIX_MM_INDEX_PROJ=tiled: index q/k projections token-tiled with F16 weights (same per-output chain, same bits).
    void set_index_tiled(bool on);
    // STRIX_MM_INDEX_DECODE=1: decode rows' index projections in one wide launch (block-major F16, same chain/bits).
    void set_index_decode(bool on);
    // STRIX_MM_INDEX_PROJ=attn: one-launch index projection (decode and prefill, same bits).
    void set_index_attn(bool on);
    [[nodiscard]] bool index_attn() const;
    // STRIX_MM_PF_IDX_CONC=0 (default), 1 (original events), 2 (disable system fences).
    void set_prefill_index_concurrent(int mode);
    [[nodiscard]] int prefill_index_concurrent() const;
    // STRIX_MM_PF_KV_TILE=off (default), auto, or config index; K/V prefill only.
    // -1 = off/library policy, -2 = auto (rows >= 2048: tile 1; otherwise tile 7).
    void set_prefill_kv_tile(int cfg);
    [[nodiscard]] int prefill_kv_tile() const;
    // STRIX_MM_INDEX_CONC=1: decode index projection concurrent with q/k/v (same kernels, same bits).
    void set_index_concurrent(bool on);
    // STRIX_MM_ATTEND_DIET (default ON): decode attend v3 at its 32 KB LDS on two single-row sparse sequences >= 8K.
    void set_attend_diet(bool on);
    [[nodiscard]] bool attend_diet() const;
    // STRIX_MM_ATTEND4_DECODE (default ON): decode attend v4, one head per wave, on the same grids (before the diet).
    void set_attend4_decode(bool on);
    [[nodiscard]] bool attend4_decode() const;
    // STRIX_MM_SELECT_ONCE (default ON): decode top-k selection once per (row, KV head) on sparse calls of >= 4 rows.
    void set_select_once(bool on);
    [[nodiscard]] bool select_once() const;
    [[nodiscard]] std::uint64_t select_once_count() const;  // decode attention calls that took it (diagnostic)
    [[nodiscard]] bool index_concurrent() const;
    // decode fused exchange boundary (decode module path, same bits; qualification notes):
    // STRIX_MM_XB_WAIT=1: post_exchange waits for the peer's partial in-kernel; STRIX_MM_XB_INPLACE=1: the producers
    // write the module's TX slot (publish kernel instead of the copy kernel).
    void set_xb_wait(bool on);
    [[nodiscard]] bool xb_wait() const;
    void set_xb_inplace(bool on);
    [[nodiscard]] bool xb_inplace() const;
    // Exchanges so far that took the in-kernel wait / the in-place producer (path activation receipts).
    [[nodiscard]] std::pair<std::uint64_t, std::uint64_t> xb_counts() const;
    // STRIX_MM_XB_PREP=1: the FFN post_exchange (module path) writes the next layer's q/k/v input transform, the q/k/v
    // linears skip their prepare launch (same bits); xb_prep_count = layers that took it.
    void set_xb_prep(bool on);
    [[nodiscard]] bool xb_prep() const;
    [[nodiscard]] std::uint64_t xb_prep_count() const;
    // STRIX_MM_XB_PUBLISH=1 (with XB_INPLACE): the in-place producers (o_proj v4 core, MoE down4, dense down) publish
    // their exchange message themselves (lean form), no publish launch; xb_publish_count = exchanges that took it.
    void set_xb_publish(bool on);
    [[nodiscard]] bool xb_publish() const;
    [[nodiscard]] std::uint64_t xb_publish_count() const;
    // STRIX_MM_XB_WARM=MB (0 = off): per decode exchange, a side stream gated on the message's TX flag reads up to MB
    // of the next phase's weights into the memory-side cache (cache state only, same bits); xb_warm_count = warms.
    void set_xb_warm(unsigned mb);
    [[nodiscard]] unsigned xb_warm() const;
    [[nodiscard]] std::uint64_t xb_warm_count() const;
    // STRIX_MM_XB_WARM_O=MB (0 = off): per decode layer, the warm stream reads up to MB of the layer's o_proj into the
    // memory-side cache while its MSA kernels run (gated on the end of the q/k/v and index projections; cache state
    // only, same bits); xb_warm_o_count = warms.
    void set_xb_warm_o(unsigned mb);
    [[nodiscard]] unsigned xb_warm_o() const;
    [[nodiscard]] std::uint64_t xb_warm_o_count() const;
    // STRIX_MM_XB_POSTILP=1: every post-exchange kernel issues all its loads before the first store (same bits);
    // xb_postilp_count = post-exchange launches that took it.
    void set_xb_postilp(bool on);
    [[nodiscard]] bool xb_postilp() const;
    [[nodiscard]] std::uint64_t xb_postilp_count() const;
    [[nodiscard]] bool index_decode() const;
    [[nodiscard]] bool index_tiled() const;
    void set_shared_first(bool on);  // STRIX_MM_SHARED_FIRST: shared expert overlaps the route copy (same sums)
    [[nodiscard]] bool shared_first() const;
    void set_profiling(bool on);  // STRIX_MM_PROFILE: synchronised per-phase times (slows the run)
    void set_trace(bool on);      // StepTrace of every decode step (events only, same bits)
    [[nodiscard]] const StepTrace& last_trace() const;
    // prefill exchange module (src/minimax/pfx): set before load() (it sizes the per-chunk buffers);
    // it shares the transport with the communicator, prefill() drains it before any communicator exchange.
    void set_prefill_exchanger(pfx::Exchanger* x, bool on);
    void set_prefill_xchg(bool on);  // STRIX_MM_PFX (runtime switch once set): prefill layer exchanges on the module
    [[nodiscard]] bool prefill_xchg() const;
    // STRIX_MM_PF_ROUTER_BATCH=1: split routing for n > 8; STRIX_MM_PF_ROUTER_ROWS=4|8 (default 8).
    void set_prefill_router_batch(bool on);
    [[nodiscard]] bool prefill_router_batch() const;
    void set_prefill_router_rows(unsigned rows);  // batch router form: 4, 8 or 32 (tiled scores); same bits
    [[nodiscard]] unsigned prefill_router_rows() const;
    // STRIX_MM_PF_INTERLEAVE=1 (runtime; needs the module on): the max_rows chunks of a prompt run layer-interleaved in
    // groups of STRIX_MM_PF_GROUP (default 2): each chunk's exchange overlaps the other chunk's compute (same kernel
    // calls). =comm: the same schedule over the communicator's start()/finish() (module off; attribution arm).
    void set_prefill_interleave(bool on);
    void set_prefill_mtfit(int on);  // -1 = environment default, 0 = off, 1 = on (v4 only)
    void set_prefill_table_ring(int on);  // -1 = STRIX_MM_PF_TABLE_RING (default off), 0 = off, 1 = on
    void set_prefill_gufuse(int on);  // -1 = STRIX_MMX_PREFILL_GUFUSE (default off), 0 = off, 1 = on
    void set_prefill_pipe(int value);  // -1 = environment, 0 = production, 1 = GLOBAL pipeline, 2 = simplified decode
    void set_prefill_core8(int value);  // -1 = environment, 0 = off (default), 1 = A LDS, 2 = shared B
    [[nodiscard]] bool prefill_interleave() const;
    // STRIX_MM_PFX_INPLACE=1 (runtime; module on): o_proj and the FFN write their partial straight into the module's TX
    // slot (no copy kernel); the same values at a different address.
    void set_prefill_inplace(bool on);
    [[nodiscard]] bool prefill_inplace() const;
    // STRIX_MM_PF_TX_UNCACHED=1: prefill rows > 8 write uncached TX directly (default off).
    void set_prefill_tx_uncached(bool on);
    [[nodiscard]] bool prefill_tx_uncached() const;
    // FNV-1a of `slot`'s K, V and index-key caches over positions [0, positions) of every layer (exactness checks).
    [[nodiscard]] std::uint64_t state_fnv(std::uint32_t slot, std::uint64_t positions);
    // FNV-1a of the last prefill's final normed rows of its last chunk (the head's input), 0 before any prefill.
    [[nodiscard]] std::uint64_t prefill_out_fnv();

    [[nodiscard]] const dist::CommunicatorStats& comm_stats() const;
    [[nodiscard]] const Profile& profile() const;
    [[nodiscard]] LoaderSnapshot loader_snapshot() const;
    [[nodiscard]] LoaderSnapshot draft_loader_snapshot() const;
    [[nodiscard]] std::uint32_t rank() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::minimax::engine
