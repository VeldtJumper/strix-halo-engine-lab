#pragma once
// deepseek41 CUDA prefill (sm_120): T prompt rows of layers [first, end) at once on one device, the plan of the HIP
// DevicePrefill (deepseek41_prefill.hpp) on the CUDA layer kernels (cuda_ds41_core / _attention / _moe): every row gets
// the bits its decode step would give, and the attention state left behind is the one T decode steps leave.
//
//  - Cone (exact suffix replay, ds41_prefill.hpp): over the full 40 layers the decoder layers (20..39) compute only the
//    trailing rows the final state and the last row's logits depend on (~2.2-2.4K rows whatever T).
//  - Routed experts by layer from a tiered provider (cuda_ds41_expert_provider.hpp): resident on the device, in pinned
//    host memory, or read from the model file; non-resident layers stream through device slots on a copy stream while
//    the earlier layers run.
//  - Engram rows of every prompt token read from the model file (local NVMe) on a thread while layer 0 runs.
//
// One sequence, one process, world 1. Row-local work runs in chunks of chunk_rows rows, attention in windows of
// window_rows rows, the routed experts in chunks of moe_chunk_rows rows (every expert of a resident or streamed layer
// is read once per chunk).
#include "strix/cuda_ds41_attention.hpp"
#include "strix/cuda_ds41_common.hpp"
#include "strix/cuda_ds41_core.hpp"
#include "strix/deepseek41_attention_state.hpp"
#include "strix/ds41_host_pin_plan.hpp"
#include "strix/ds41_host_pin_admission.hpp"
#include "strix/ds41_weight_read_options.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strix::ds41::cuda {

inline constexpr double kDefaultLongVramCapGiB = 38.0;

class CudaPrefillInterrupted : public std::runtime_error {
public:
    explicit CudaPrefillInterrupted(const char* reason) : std::runtime_error(reason) {}
};

struct CudaPrefillProgress {
    std::uint32_t position = 0, total = 0;
    double elapsed_seconds = 0;
    bool cancel_pending = false;
};

struct CudaPrefillStateSection {
    std::string name;
    std::uint64_t bytes = 0;
};

// Optional bounded-pass state observer. Chunks are immutable compressed entries.
// Their span remains valid until the returned future completes. Implementations
// must complete every returned future, including when transport fails.
class CudaPrefillStateStream {
public:
    virtual ~CudaPrefillStateStream() = default;
    virtual void begin(std::uint32_t position, std::span<const CudaPrefillStateSection> sections) = 0;
    virtual std::future<void> chunk(std::string_view section, std::uint64_t offset,
                                   std::span<const std::uint8_t> bytes) = 0;
};

class CudaPrefillTransferError : public std::runtime_error {
public:
    explicit CudaPrefillTransferError(const std::string& reason) : std::runtime_error(reason) {}
};

// Where a layer's routed experts live during a prefill.
enum class ExpertTier : std::uint8_t { device = 0, host = 1, file = 2 };

struct CudaPrefillOptions {
    std::uint32_t first_layer = 0, end_layer = 40;  // layers [first, end); embedding when first == 0, head when end == 40
    std::uint32_t max_positions = 8192;             // context capacity (global positions)
    // Bounded prefill: row buffers hold min(pass_rows, max_positions); keys and RoPE hold max_positions.
    // Zero preserves the original single-pass allocation and execution path.
    std::uint32_t pass_rows = 0;  // F = 8192 bounds row memory; exact but slower (more expert streaming per pass)
    std::uint32_t chunk_rows = 1024;                // row-local chunk (q_b, attention windows, o_a / o_b, Engram)
    std::uint32_t moe_chunk_rows = 4096;            // routed-expert chunk
    std::uint32_t window_rows = 64;                 // attention call rows (<= attn::kMaxCallRows)
    // An object that only imports a state and decodes (step): none of run()'s prefill buffers (sized by max_positions,
    // chunk_rows, moe_chunk_rows) are allocated, the routed scratch is sized for decode_rows, and run() is refused.
    bool decode_only = false;
    bool lean_state = true;                        // decode_only: Dmax rows and a 136 + Dmax ring; false restores original state
    bool cone = true;                               // exact suffix replay (only with the full 40 layers)
    std::uint32_t draft_rows = 0;                   // drafter seed rows the cone also covers
    // Approximate bounded decoder replay: 0 (default) keeps the exact cone; 256 caps
    // output rows at 256 and window K/V rows at 383 (rounded up to call boundaries).
    // Older rows read the newest stream available, possibly encoder output. The
    // KV source still covers the whole prompt and every ring retains 128 rows.
    // Requires full-cone fast dense/MoE, head/draft rows <=256. Quality is gated
    // separately for each execution identity; this flag does not imply a pass.
    std::uint32_t replay_rows = 0;
    // Expert tiers per layer (size end_layer, entries outside the range ignored); empty: from the budgets below.
    std::vector<ExpertTier> tiers;
    double device_expert_gib = 0.0;                 // budget of resident expert layers on the device
    bool host_tier_implicit = false;
    bool service_host_buffers = false;
    bool reserve_export_capture = false;
    double host_expert_gib = 0.0;                   // budget of pinned host memory for expert layers
    HostPinOrder host_pin_order = HostPinOrder::layer;  // deadline: pin streamed suffix; early: ascending fill; both exclude idle-preload slots
    std::uint32_t slots = 2;                        // device slots for non-resident layers (0: none; decode-only objects)
    // TIMING ONLY (not the model's bits): host-tier layers beyond this many share the pinned copies of the first ones
    // (real transfers, wrong weights), to measure a RAM tier larger than the free memory (0: off).
    std::uint32_t host_alias_layers = 0;
    // Optional host admission: commit >= 2n + floor, physical >= n + floor + 4 GiB.
    // Implicit service tiers always use immediate admission, even with floor 0:
    // no commit-growth probe or wait; failed/unknown telemetry demotes to FILE.
    // Explicit budgets with a positive floor may grow commit and wait up to the
    // shared deadline. Explicit floor 0 retains the legacy allocation path.
    double host_commit_floor_gib = 0.0;
    double host_pin_wait_s = 360.0;
    // Lazy host promotion (streamer transport; strix/ds41_host_promote.hpp): a file-tier layer read for a run is kept in
    // a new pinned buffer and streams from host memory in the next runs (the same file bytes), while the host keeps
    // max(host_commit_floor_gib, 20) GiB free (and the promoted bytes stay under host_promote_max_gib when > 0); below
    // that floor at a run start, the least recently used promoted layers go back to the file tier. Off here; the
    // prefill service turns it on (--no-host-promote / STRIX_DS41_HOST_PROMOTE=0 / --baseline). Not with pass_rows,
    // host_alias_layers or the whole-layer provider.
    bool host_promote = false;
    double host_promote_max_gib = 0.0;
    // Engram rows: "file" (the model file's tables), "zero" (no rows read: timing only, NOT the model's bits).
    std::string engram = "file";
    bool engram_direct = true;                      // unbuffered reads of the Engram rows / file-tier experts
    WeightReadOptions weight_read;
    unsigned io_threads = 8;                        // expert file reads (8 MB pieces)
    unsigned engram_threads = 64;                   // Engram row reads (queue depth of the random 4 KiB reads)
    std::uint32_t engram_cache_mb = 0;              // decode steps: Engram records kept in RAM by file offset (0 = off)
    std::uint32_t head_rows = 1;                     // logits of the last head_rows rows (quality gate: KL over positions)
    bool stream_hashes = false;                     // FNV of the last row's stream after each layer
    // Timing (events, no host synchronisation). timing: per-layer device times (begin, attention end, MoE begin, end)
    // and the device time each layer's stream waits for its expert slot; timing_ops adds an event before every op
    // group (the op profile; its extra events measured within the run-to-run noise). Without timing: no events.
    bool timing = false;
    bool timing_ops = true;
    // CUDA graphs: after graph_warmup_runs eager runs, each layer runs as two captured segments replayed on later runs
    // of the same T (A: Engram .. router + shared expert; B: routed experts + hc_post). The host waits stay outside
    // (the Engram rows before A, the expert slot between A and B, its release after B). A changed T recaptures; a
    // changed expert slot address recaptures that layer's B. Per-op events are off in graph runs.
    bool graphs = false;
    std::uint32_t graph_warmup_runs = 1;
    // Prefill-shape lane-dot kernels (cuda_ds41_prefill_kernels.hpp: Engram kv, router logits, compressor and
    // indexer F16 projections) for calls of at least prefill_kernels_min_rows rows; same bits as the decode kernels.
    bool prefill_kernels = true;
    std::uint32_t prefill_kernels_min_rows = 16;
    // Many-row Q8_0 x A8 sites (cuda_ds41_prefill_q8.hpp: q_a + kv, q_b, o_a, o_b, shared expert) for calls of at least
    // prefill_q8_min_rows rows; same bits as the tiled kernel.
    bool prefill_q8 = true;
    std::uint32_t prefill_q8_min_rows = 64;
    // Routed experts on the fast arm (cuda_ds41_moe_fast.hpp: INT8 MMAs, free FP32 accumulation order; NOT the
    // reference's bits, a quality arm) for MoE chunks of at least 64 rows; off: the exact path.
    bool fast_moe = false;
    bool fast_moe_signed_lut = false;               // experimental exact IQ2 sign decoding table; default off
    bool fast_moe_wide_tiles = false;               // experimental INT8 256-entry tile; default off
    // Diagnostic only: dump the actual selected routes after each layer's router (p8 route-file format).
    // One file per batch member, suffix .N for N > 0; overwritten per run. Synchronizes the compute stream.
    std::string moe_routes_out;
    // Fast dense sites (cuda_ds41_dense_fast.hpp): NOT the exact bits (FP32 sums in any order with FMA). The Q8_0 x
    // A8 sites on the relaxed int8 GEMM; hc_post + the next collapse / RMSNorm / A8 + the next hc mixes in one pass
    // over all output rows. Quality gate: KL of the logits against the exact prefill.
    bool fast_dense = false;
    // Fast attention (cuda_ds41_fast_attention.hpp): tensor-core attention and FP8 index scores per chunk of rows, the
    // A8 rows of o from the attention's epilogue. NOT the exact bits (quality gate: KL against the exact logits).
    bool fast_attention = false;
    bool fast_index = true;  // with fast_attention: FP8 index scores (else the exact indexer per window)
    int device = 0;

    // ---- incremental decode (step(): R <= decode_rows rows after a run(), an import_state() or a previous step) ----
    std::uint32_t decode_rows = 8;                  // decode_max_rows Dmax (the decode windows: 1..8)
    // CUDA graphs of the step: after decode_graph_warmup eager steps of a shape, the step runs as captured graph
    // segments (one launch per segment: [first, first Engram layer) and the rest + head), the positions, token ids
    // and Engram rows read by the graph's own copies from pinned host buffers. Captured per (rows, position bucket,
    // stop layer, io); the bucket bounds the indexer launches (blocks past a row's count exit at once: same bits).
    // Exact launch fusions of decode sites only (each default off).
    bool decode_fuse_hc = false;
    bool decode_fuse_shared = false;  // two-launch shared at R1; wider rows retain the legacy plan
    // router of decode steps (and the predictor's route_other) as one launch with the weight rows streamed at full
    // width and the selection in the last block (cuda_ds41_router_v2.hpp, same bits); rows 1..8
    bool decode_router_v2 = false;
    // dense Q8_0 GEMVs of decode steps launched with programmatic dependent launch, weight slab prefetched into L2
    // before the dependency wait (sm_90+; same bits)
    bool decode_pdl = false;
    // o_b's two K-halves as one launch (Q8A8View::k_halves = 2; same bits as the write + accumulate pair)
    bool decode_ob_one = false;
    // with a moe_site hook, the shared expert on a side branch forked before the router and joined before the
    // site's combine (CudaMoeSite::shared_join); same bits
    bool decode_shared_fork = false;
    // a layer's attention preparation (window write, compressor, indexer, selection) on a side branch forked
    // after q_norm, concurrent with q_b; joined before the attention kernels; same bits
    bool decode_attn_fork = false;
    // the decode qa_kv GEMV (112 tiles) with the class split S = 1 instead of the automatic 2 (no
    // scratch + reduce kernel; the split is documented bit-preserving); same bits
    bool decode_qakv_s1 = false;
    // with decode_ob_one, the attention site's hc_expand in o_b's epilogue (one launch fewer); same bits
    bool decode_ob_expand = false;
    // with a moe_site hook, the FFN site's hc_expand in the site's combine (CudaMoeSite::hc_*); same bits
    bool decode_moe_expand = false;
    bool decode_graphs = true;
    std::uint32_t decode_graph_warmup = 1;
    std::uint32_t decode_bucket = 512;              // positions per graph bucket
    // Non-resident expert layers in a step (exactness checks, not a speed path): the step synchronises after the
    // router, reads only the routed experts' bytes of that layer from the model file into one device slot and runs the
    // same kernels on them (eager; graphs need every layer of the step resident).
    bool decode_sparse_experts = true;
    bool decode_spin = true;                        // the host polls the step's completion event (else blocks on it)
    // Engram rows of the next token read as soon as a head step knows its greedy token (R1): the next step whose
    // token is that one finds its rows in (or in flight), any other token drops them and reads its own. Same bytes.
    bool engram_prefetch = true;
    // The pair's P15 target identity on this device (docs/models/deepseek-v4.1/expert-tp2-i1152-v1.md): decode steps compute the routed
    // experts' down products and combine in the order of ds41-expert-tp2-i1152-v1 (two K-half parts per output, each
    // folded over the row's experts, then (shared + S0) + S1). run() (prefill) keeps the EP order, as the pair's prefill.
    bool expert_tp = false;
    // Requantization overlay GGUF (the dense tensors it holds replace the model's; same types and sizes).
    std::string weights_overlay;
    // Trellis shared expert in decode steps (the pair's trellis site class shexp, e.g. ds41-trellis-mixB-v1): a pack
    // directory (dense.L<l>.safetensors, layers.<l>.ffn.shared_experts.w1 / w3 / w2) whose gate / up / down replace the
    // Q8_0 shared expert of step(): F32 normed x in (no A8), the clamped SwiGLU in F32, F32 out. Each linear runs the
    // pair library's GEMV order (docs/models/deepseek-v4.1/cuda-trellis-shexp.md): splits of the K16 tiles as the library's vector policy,
    // per lane class two pair products per tile into one accumulator, lanes folded (0 + 2) + (1 + 3), splits folded in
    // order. run() (prefill) keeps the Q8_0 shared expert, as the pair's prefill. Empty: off. Needs the build option
    // STRIX_ENABLE_CUDA_TRELLIS.
    std::string trellis_shexp;
    // ---- Transport and residency (the expert streamer, ds41_expert_streamer.hpp). All off: the whole-layer provider.
    // Residency per EXPERT: device_expert_gib (slots included) keeps the hottest experts of every layer resident and
    // streams the rest of each layer; a layer streams in proportion to its rows at max_positions (more rows, more
    // compute to hide its copy behind). Same bits as whole layers (the MoE runs by ranges, the combine is the layer's).
    bool expert_residency = false;
    std::string expert_rank;            // ranking: a routes file (lines "layer:id,id,.. ...") or a placement JSON; empty: by id
    std::uint32_t xfer_group_experts = 0;  // streamed experts copied in groups of this many (an event each), the MoE of a
                                           // layer runs group by group as they land (0: one group)
    std::uint32_t xfer_piece_mib = 0;      // largest single host -> device copy (0: a group's tensor piece at once)
    // Sizes from the device at hand: device_expert_gib from the free memory after the dense weights and buffers (and
    // vram_cap_gib), the device profile (VRAM, PCIe, measured H2D, SMs) in describe().
    bool auto_budget = false;
    double vram_cap_gib = 0.0;             // everything this object allocates on the device stays under this (0: none)
    double vram_reserve_gib = 1.0;         // kept free for the runtime (auto_budget)
    bool gpu_probe = true;                 // measure the H2D bandwidth when the per-GPU cache has none
    // Engram rows: "mapped" (kernels read the pinned rows over PCIe) or "stream" (a high-priority stream copies each
    // Engram layer's rows to the device as soon as they are read, then the kernel reads device memory).
    std::string engram_upload = "mapped";
    std::size_t engram_cache_mib = 0;      // LRU of Engram file pages in RAM, kept across runs (0: none)
    // The streamer's idle preload (the next run's first slots, GBs of H2D) starts only after capture_state() (or
    // release_idle_preload()), not at the end of run(): on WDDM every copy queues behind it, so a state capture issued
    // while it runs waits ~70-170 ms. A run() that starts with the preload still held releases it first.
    bool defer_idle_preload = false;

    // ---- Closed batches (run_batch): one extra per-sequence context per entry, sized for that many prompt rows
    // (row buffers + attention state; allocated before the expert budget, so they come out of the resident experts).
    // Context 0 is the run() one (max_positions). Empty: no batches.
    std::vector<std::uint32_t> batch_positions;
};

// One prompt of a closed batch (run_batch).
struct CudaBatchItem {
    std::span<const std::uint32_t> tokens;
    bool fast_attention = false;  // this prompt's fast-attention policy (needs fast_attention at construction)
};

struct CudaPrefillLayerTiming {
    float total_ms = 0, attention_ms = 0, moe_ms = 0;
    float expert_wait_ms = 0;         // host: acquire() blocked until the slot's copy was enqueued
    float expert_wait_device_ms = 0;  // device: the stream's wait for the slot copy (timing)
    std::uint32_t rows = 0;  // output rows
    // accounting (timing, streamer): the stream idle inside the MoE waiting for this layer's streamed groups to land
    // (events around each range's wait), the device time since the previous layer's end (or the run's first event),
    // and this layer's slot copy on the copy stream (span and bytes; 0 when the slot was preloaded between runs)
    float range_wait_ms = 0, gap_before_ms = 0, copy_ms = 0;
    std::uint64_t copy_bytes = 0;
};

struct CudaPrefillResult {
    std::vector<float> logits;                   // last row (with the head)
    std::vector<float> head_logits;              // head_rows > 1: [row][vocab] of the last rows
    std::uint32_t next_token = 0;
    std::vector<std::uint64_t> stream_hashes;    // last row after each layer (stream_hashes)
    double seconds = 0.0;                        // wall time of run()
    double enqueue_seconds = 0.0;                // host time until every layer was enqueued (launch + host waits)
    double engram_seconds = 0.0;                 // Engram row reads (overlapped with layer 0)
    double expert_io_seconds = 0.0;              // file-tier reads during the run (overlapped)
    std::uint64_t row_layers = 0;                // rows x layers computed
    std::uint64_t expert_bytes_streamed = 0;     // bytes copied into slots during the run
    double expert_copy_busy_s = 0.0;             // streamer + timing: sum of the streamed layers' copy spans
    double expert_stall_ms = 0.0;                // streamer + timing: sum over layers of max(0, copy end - MoE start)
    std::uint64_t engram_records = 0, engram_pages = 0, engram_page_hits = 0;  // distinct rows, pages, cached pages
    std::vector<CudaPrefillLayerTiming> layers;  // timing
    std::map<std::string, double> ops;           // timing_ops: device milliseconds per op group over every layer
    // accounting (timing): the device span of the run from its first enqueued op to the end of the head (events),
    // split into embed (before the first layer), the layers (CudaPrefillLayerTiming) and head (after the last layer);
    // host time before the first op and after the final synchronization
    double device_ms = 0.0, embed_ms = 0.0, head_ms = 0.0;
    double host_setup_s = 0.0, host_tail_s = 0.0;
    double engram_join_s = 0.0;                  // part of host_tail_s: waiting for the Engram reader thread after the device finished
    // the between-runs preload of the first slots this run consumed (streamer; not in expert_bytes_streamed): bytes,
    // its wall time, and the part of it that ran after this run began (it then delays the first streamed layer)
    std::uint64_t preload_bytes = 0;
    double preload_s = 0.0, preload_after_begin_s = 0.0;
    // lazy host promotion (host_promote): layers promoted / given back since the previous run (its between-runs preload
    // included), promoted layers and bytes held now; and the file-tier bytes this run read (part of expert_bytes_streamed)
    std::uint32_t host_promoted_layers = 0, host_demoted_layers = 0, host_promoted_total = 0;
    std::uint64_t host_promoted_bytes = 0, file_bytes_streamed = 0;
    bool graphs = false;                         // this run replayed (or captured) the layer segments as graphs
    std::uint32_t graph_captures = 0;            // segments captured during this run
    // run_batch: seconds = until this prompt's logits were on the host; batch_seconds = the whole batch's wall
    double batch_seconds = 0.0;
    std::uint32_t batch_size = 1;
};

// One decode step's optional stream input / output (pipeline stages). Host pointers (any memory; copied through the
// object's pinned buffers).
// One routed-expert site of a decode step, handed to CudaStepIo::moe_site (a parts-host computes the layer's MoE).
struct CudaMoeSite {
    std::uint32_t layer = 0, rows = 0;
    cudaStream_t stream = nullptr;     // the step's stream: the router has been enqueued on it
    A8Rows x;                          // A8 of the MoE input [rows][hidden]
    const std::uint32_t* ids = nullptr;  // [rows][top_k] router order
    const float* weights = nullptr;      // [rows][top_k]
    float* shared = nullptr;           // [rows][hidden]: written by shared_expert(stream)
    float* raw = nullptr;              // [rows * top_k][hidden] scratch for the raw expert vectors
    float* y = nullptr;                // [rows][hidden]: the site's output, y = (shared + low) + high
    std::function<void(cudaStream_t)> shared_expert;  // enqueues the shared expert into `shared`
    // Non-null: the shared expert is already enqueued on a side branch (shared_expert is empty); the callee makes
    // `stream` wait on this event before it reads `shared` (the combine), and the branch joins there.
    cudaEvent_t shared_join = nullptr;
    // Non-null hc_out (decode_moe_expand): the callee runs hc_expand (sub = y, residual / mixes / out / pre_carry as
    // enqueue_hc_expand) in its combine, and the step skips its own; same bits.
    const float* hc_residual = nullptr;
    const float* hc_mixes = nullptr;
    float* hc_out = nullptr;
    float* hc_pre_carry = nullptr;
    // The router's F32 input [rows][hidden] (the layer's normed MoE input) and a launcher of ANOTHER layer's router on
    // it (that layer's weights / bias / bias_image, top_k <= 32, scores [rows][experts], ids / weights [rows][top_k]):
    // routed-expert prefetchers predict the next layer's experts with it. Capturable; never touches ids / weights.
    const float* x_f32 = nullptr;
    std::function<void(std::uint32_t layer, std::uint32_t top_k, float* scores, std::uint32_t* ids, float* weights, cudaStream_t)> route_other;
};

struct CudaStepIo {
    // Routed-expert sites (optional): called for every layer of the step after the router instead of the shared
    // expert + routed experts + combine. The callee must enqueue the shared expert (site.shared_expert) and leave y
    // on site.stream with the one-device combine's bits. Steps with a hook run eagerly (unless moe_site_graphs) and never read
    // expert weights of their own (the expert tiers of the object are not used).
    std::function<void(const CudaMoeSite&)> moe_site;
    // moe_site only enqueues capturable device work that stays valid for later replays (it is called while the
    // step's segments are captured, not at replays): the step may then run as graphs like a resident step.
    bool moe_site_graphs = false;
    // Captured graphs are keyed by this value too: a caller that changes what its moe_site enqueues (another callee,
    // other options, other buffers) sets a new epoch so the step captures again instead of replaying the old graphs.
    std::uint32_t moe_site_epoch = 0;
    // Input of layer first (required when first_layer > 0, else ignored): the residual stream rows x 4 x hidden FP32
    // after layer first - 1 and that layer's FFN hc mix rows x 24 (pre 4, post 4, comb 16; its pre collapses the next
    // attention input).
    const float* stream_in = nullptr;
    const float* mix_in = nullptr;
    // Output after layer stop - 1 (optional): the stream rows x 4 x hidden and the FFN hc mix rows x 24.
    float* stream_out = nullptr;
    float* mix_out = nullptr;
    // Output without a host copy: the stream and the mix stay in the object's mapped output buffer (written there by the
    // step's last kernel), CudaStepResult::stream_view / mix_view point at them until the next step (a FRONT sends them
    // from there). Independent of stream_out / mix_out (both may be set).
    bool stream_view = false;
    std::uint32_t stop_layer = 0;  // run layers [first, stop) (0 = end_layer); the head only when stop is the model's end
    bool logits = true;            // download the logits of every row (with the head); else only the greedy tokens
    // The boundary sections of a pipeline cut after layer sections_cut - 1 (0: none), copied by the step itself (inside
    // its graph, into pinned memory, no extra synchronisation): the compressed entries the step completed on every KV
    // source < cut that a layer >= cut reads, the step rows' top-k of every index source < cut used >= cut, and the
    // step rows' candidate blocks when the candidate source is < cut and a user is >= cut. In CudaStepResult::sections.
    std::uint32_t sections_cut = 0;
    // DSpark drafter input (optional, device memory [rows][3 * hidden]): the step writes, for every row, the mean over
    // the four hyper-connection copies of the input stream of layers draft_hidden_first .. + 2 (~0 = the last three
    // layers), ((s0 + s1) + s2) + s3 then * 0.25, concatenated by layer, inside its own launches (graphs keyed by the
    // address). Extra writes only: the step's other outputs are unchanged. draft_hidden_raw (optional, tests, device
    // memory [rows][3][4 * hidden]): the same layers' input streams themselves.
    float* draft_hidden = nullptr;
    std::uint32_t draft_hidden_first = ~0U;
    float* draft_hidden_raw = nullptr;
    // Called by step() once its device work is enqueued, before it waits for it: caller host work that may overlap the
    // step (e.g. the previous step's records). Not called by a step that fails before that point.
    std::function<void()> after_launch;
};

struct CudaStepSections {
    struct Kv {
        std::uint32_t layer = 0, ratio = 0, first = 0, count = 0;  // entries [first, first + count)
        std::vector<std::uint8_t> kv, idx_keys;                     // count x 288 / count x 68 bytes
    };
    struct Topk {
        std::uint32_t layer = 0;
        std::vector<std::uint32_t> counts, ids;  // rows / rows x 512 (the first counts[r] of row r)
    };
    std::vector<Kv> kv;
    std::vector<Topk> topk;
    std::int32_t candidate_source = -1;
    std::uint32_t cand_stride = 0;
    std::vector<std::uint8_t> candidates;  // rows x cand_stride
};

struct CudaStepResult {
    std::uint32_t position = 0, rows = 0;        // the step's first position and rows
    const float* stream_view = nullptr;          // io.stream_view: rows x 4 x hidden in mapped host memory (until the next step)
    const float* mix_view = nullptr;             // io.stream_view: rows x 24
    std::vector<std::uint32_t> tokens;           // greedy token of every row (head)
    std::vector<float> logits;                   // [row][vocab] (head, io.logits)
    std::vector<std::uint64_t> logits_fnv;       // per row (head, io.logits)
    std::vector<std::uint64_t> stream_hashes;    // last row after each layer (options.stream_hashes; eager steps)
    double seconds = 0.0;                        // wall time of step() (enqueue + Engram reads + device + downloads)
    double enqueue_seconds = 0.0;                // host time until the last launch returned
    double engram_seconds = 0.0;                 // Engram row reads of the step's rows (I/O thread; prefetched: issued by the previous step)
    double engram_wait_seconds = 0.0;            // host waits for them before the segments that start at an Engram layer
    double engram_wait_first_seconds = 0.0;      // of engram_wait_seconds: the wait before the first Engram layer (GPU idle)
    double engram_issue_seconds = 0.0;           // of engram_seconds: issuing the reads (the rest: waiting for them)
    bool engram_prefetched = false;              // the rows were issued by the previous step (its greedy token)
    std::uint32_t engram_checked = 0, engram_check_mismatches = 0;  // STRIX_DS41_CUDA_ENGRAM_CHECK=1: records re-read and compared
    double expert_io_seconds = 0.0;              // sparse expert reads (non-resident layers)
    float device_ms = 0.0F;                      // device time of the step (first launch to the last op; timing)
    bool graphs = false;
    std::uint32_t graph_captures = 0;
    std::vector<CudaPrefillLayerTiming> layers;  // timing: per layer total / attention / moe (attention = up to hc_post/attn)
    std::map<std::string, double> ops;           // timing, eager steps (timing_ops): device ms per op group
    CudaStepSections sections;                   // io.sections_cut
};

class CudaPrefill {
public:
    CudaPrefill(const std::filesystem::path& gguf, const CudaPrefillOptions& options);
    ~CudaPrefill();
    CudaPrefill(const CudaPrefill&) = delete;
    CudaPrefill& operator=(const CudaPrefill&) = delete;

    // A fresh sequence: tokens at positions [0, T).
    CudaPrefillResult run(std::span<const std::uint32_t> tokens);
    // Continue the previous/imported sequence (requires pass_rows). Fresh run() resets the sequence.
    CudaPrefillResult extend(std::span<const std::uint32_t> tokens);
    // (import_state() below: with pass_rows the state extend() continues)
    [[nodiscard]] std::uint32_t positions() const noexcept;
    // Thread-safe control/status for bounded passes. Cancellation is observed before the next pass.
    void request_cancel() noexcept;
    // Only while no run is active; a service uses this before publishing its next job.
    void clear_cancel() noexcept;
    [[nodiscard]] CudaPrefillProgress progress() const noexcept;
    // Set only while idle. Non-null requires pass_rows; allocates two 4 MiB pinned copy buffers per run.
    void set_state_stream(std::shared_ptr<CudaPrefillStateStream> stream);
    // A closed batch of fresh sequences, layer-major: every layer runs each prompt's rows (its own context: stream,
    // attention state, Engram rows, cone, fast-attention policy) with the layer's experts acquired ONCE and released
    // after the last prompt, so the batch shares one pass of the streamed weights. Each prompt's kernels and their
    // order are those of its solo run (same bits). Prompts go to the smallest free context that holds them
    // (batch_positions); at most batch_contexts() prompts. Results in item order.
    std::vector<CudaPrefillResult> run_batch(std::span<const CudaBatchItem> items);
    [[nodiscard]] std::size_t batch_contexts() const noexcept;                 // 1 + batch_positions.size()
    [[nodiscard]] std::uint32_t context_positions(std::size_t k) const noexcept;  // rows context k holds
    // FNV of every layer's attention state (labels and order of DeviceForward::state_hashes).
    [[nodiscard]] std::vector<std::pair<std::string, std::uint64_t>> state_hashes() const;
    // The decode-format attention state after the last run (layers of the range present).
    [[nodiscard]] deepseek41::AttentionState export_state() const;
    // export_state()'s state captured into `out`'s pinned buffer already in the serialized format (attn::ExportCapture,
    // cuda_ds41_attention.hpp): a handful of copies, synchronized on return; then the next run() may start and
    // out.seal() (any thread) completes serialize_attention_state(export_state())'s bytes. With defer_idle_preload it
    // then lets the streamer's idle preload start.
    void capture_state(attn::ExportCapture& out);
    // Serialized bytes of a capture at `positions` positions (max_positions when 0): out.reserve() up front.
    [[nodiscard]] std::size_t capture_bound(std::uint32_t positions = 0) const;
    // defer_idle_preload: start the held idle preload now (no-op otherwise).
    void release_idle_preload();

    // ---- incremental decode ----
    // R = tokens.size() rows (1..decode_rows) at positions [position(), position() + R): layers [first, stop) on the
    // attention state left by the last run() / import_state() / step(), continued exactly as R decode rows of the
    // reference (every row the bits of its decode step); the embedding when first == 0, the head when stop is the
    // model's end. The Engram history follows the tokens. After a head step the greedy token of the last row is the
    // state's next token.
    CudaStepResult step(std::span<const std::uint32_t> tokens, const CudaStepIo& io = {});
    // Continue a sequence from a decode-format state (from this object, a CUDA prefill or the pair): the layers of the
    // range must be present (and the KV sources the range reads); position, tokens and next token are taken over.
    // With pass_rows (a long-context object) the state is the one extend() continues (its producers' selections must
    // be known); step() / decode_residency() refuse such objects.
    void import_state(const deepseek41::AttentionState& state);
    void clear_decode_graphs() noexcept;  // drain before freeing external capture destinations
    void reserve_decode_buffers();  // fixed ABI and boundary staging, before external arena planning
    // A fresh sequence at position 0 without a prefill: every cache, carry and selection zeroed, no tokens; the next
    // step's rows start at position 0 (a pipeline FRONT that feeds the prompt as decode rows, as the pair's decode does).
    void reset_sequence();
    // Sequence slots (a FRONT of several requests): `count` attention states that share the weights and buffers of this
    // object. select_sequence(i) makes slot i the sequence that step(), import_state(), reset_sequence(), commit_window(),
    // export_state(), state_hashes(), position(), next_token() and the boundary section copies act on; each slot keeps its
    // own rings, caches, carries, selections, token history and Engram n-gram state, so a slot's steps are bitwise the
    // steps of an object of its own (every slot has the geometry of slot 0; its graphs are captured per slot). Slot 0 is
    // the object's own state (run() needs it selected). New slots start empty (reset_sequence() or import_state()).
    // Selecting another slot synchronizes the stream; outputs of the last step (stream_view) stay valid until the next step.
    void set_sequences(std::uint32_t count);
    void select_sequence(std::uint32_t slot);
    [[nodiscard]] std::uint32_t sequences() const noexcept;
    [[nodiscard]] std::uint32_t sequence() const noexcept;
    // Partial commit of the last step (a verify window of R rows): keep its first `kept` rows (1..R), as if the step had
    // been those rows only; `next_token` becomes the state's next token (the target's token after the kept rows). Local
    // and cheap (no state export): position, tokens, Engram history, the ratio-2 compressor carries and the last row
    // of the index selections; everything the rejected rows wrote is unreachable from the kept prefix and rewritten
    // before any read (AttentionState::commit_rows). Valid once per step, before the next step.
    void commit_window(std::uint32_t kept, std::uint32_t next_token);
    // One process that prefills and then decodes a FRONT range (a pipeline FRONT with its own prefill): after a run(),
    // make the expert layers [first, stop) resident for step() -- a layer the streaming slots hold after the run's idle
    // preload is used in place, any other non-resident layer is uploaded into its own device buffer (from its pinned
    // copy, else the file) on the copy stream (the next step waits for it; the call only enqueues). With
    // release_run_buffers the run-only row / chunk buffers (max_positions rows) are freed first to make room. The next
    // run() undoes it (frees the uploads, re-allocates the buffers, back to the run's tiers). Steps of a range that
    // reaches past `stop` stay non-resident. Returns the host seconds; decode_residency_report() is a JSON line.
    double decode_residency(std::uint32_t stop, bool release_run_buffers = true);
    [[nodiscard]] std::string decode_residency_report() const;
    // Waits for the decode's outstanding Engram reads (e.g. the next token's prefetch issued by the last head step); the
    // prefetch stays valid for the next step.
    void wait_io();
    // Issues the Engram rows of `token` at the next position (after commit_window: e.g. a verify window's token after
    // its kept rows) on the I/O thread while the caller drafts; the next step uses them for its row 0 (R = 1: the whole
    // job, as the head step's own prefetch; R > 1: rows 1..R-1 are read after it). No-op without file Engram, before
    // the first step, or when the head step already issued this token.
    void prefetch_engram(std::uint32_t token);
    // Diagnostics: every Engram job's records read again after use and compared (CudaStepResult::engram_checked /
    // engram_check_mismatches); the initial value comes from STRIX_DS41_CUDA_ENGRAM_CHECK=1.
    void set_engram_check(bool on) noexcept;
    [[nodiscard]] std::uint32_t position() const noexcept;
    [[nodiscard]] std::uint32_t next_token() const noexcept;
    // Boundary sections of the last step for a pipeline cut (strix/ds41-pipeline-v1): compressed entries [first,
    // first + count) of KV source `layer` (288 B KV + 68 B index key each), the last step's top-k rows of index source
    // `layer` (per row: count, then 512 ids), the last step's candidate rows (cand_stride() bytes per row).
    void copy_kv_entries(std::uint32_t layer, std::uint32_t first, std::uint32_t count, std::uint8_t* kv, std::uint8_t* idx_keys) const;
    void copy_topk(std::uint32_t layer, std::uint32_t rows, std::uint32_t* counts, std::uint32_t* ids) const;
    void copy_candidates(std::uint32_t rows, std::uint8_t* out) const;
    [[nodiscard]] std::uint32_t cand_stride() const noexcept;
    // The TAIL side: the same sections written into this object's mirrors of sources outside its range (before the
    // step that reads them; topk / candidates rows are the step's rows).
    void write_kv_entries(std::uint32_t layer, std::uint32_t first, std::uint32_t count, const std::uint8_t* kv, const std::uint8_t* idx_keys);
    void write_topk(std::uint32_t layer, std::uint32_t rows, const std::uint32_t* counts, const std::uint32_t* ids);
    void write_candidates(std::uint32_t rows, const std::uint8_t* in);
    // The state of item `item` of the last run_batch (0: run()'s).
    [[nodiscard]] deepseek41::AttentionState export_state(std::size_t item) const;

    // Turns the graph runs on or off for the next runs (A/B in one process; the captured graphs are kept).
    void set_graphs(bool on) noexcept;
    void set_decode_graphs(bool on) noexcept;
    void set_timing(bool on, bool ops) noexcept;
    // Turns the fast attention on or off for the next runs (a policy by prompt length in one process). On needs
    // fast_attention at construction (its buffers); returns whether it is on. A change drops the captured graphs.
    bool set_fast_attention(bool on);
    [[nodiscard]] const std::vector<ExpertTier>& tiers() const noexcept;
    [[nodiscard]] double load_seconds() const noexcept;
    // dense weights + resident experts + slots + buffers + attention state held by this object (describe() splits it)
    [[nodiscard]] std::size_t device_bytes() const noexcept;
    [[nodiscard]] std::shared_ptr<attn::ExportCapture> export_capture() const noexcept;
    [[nodiscard]] const HostPinStats& host_pin_stats() const noexcept;
    [[nodiscard]] std::size_t pinned_bytes() const noexcept;   // pinned host memory held
    [[nodiscard]] std::string weight_io_json() const;
    // Empty when no multi-location reader is active; otherwise a comma-prefixed JSON member.
    [[nodiscard]] std::string weight_io_fields() const;
    [[nodiscard]] std::string describe() const;                // tiers, budgets, sizes (one line of JSON)
    // Device copy of a model tensor's GGUF bytes (uploaded on first use and kept by this object; e.g. a drafter on the
    // same device reads token_embd.weight / output.weight from here instead of holding its own copies).
    [[nodiscard]] const std::uint8_t* device_tensor(const std::string& name);
    // The head (output.weight) in the tiled layout the decode reads (empty without the head): a drafter on the same device
    // reads it instead of a raw copy (device_tensor would upload the raw rows again, ~0.65 GiB).
    [[nodiscard]] Q8Tiled head_tiled() const noexcept;

    struct Impl;

private:
    CudaPrefillResult run_long(std::span<const std::uint32_t> tokens, bool continuation);
    std::vector<CudaPrefillResult> run_contexts(std::span<const CudaBatchItem> items, const std::vector<std::size_t>& ctx);
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::ds41::cuda
