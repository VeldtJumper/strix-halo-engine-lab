// CUDA Qwen prefill driver (see include/strix/cuda_qwen_prefill.hpp): model upload, per-block step loop in the HIP
// step order, node state export. The arithmetic lives in the family libraries (qwen_hc, qwen_dense, qwen_gdn, qwen_qsa,
// qwen_moe, qwen_ple, qwen_mtp), each bit-identical to its HIP kernels; this file only wires buffers and order.
//
// HIP step order per layer (src/execution_plan.cpp:100-210; fixture notes/d64-dump-index.txt):
//   [ple (layer 1)] attn_hyper_mix -> {gated_delta_net | indexed_attention (L % 4 == 3)} -> attn_inject -> mlp_hyper_mix
//   -> moe_route -> shared_expert -> moe_experts -> moe_combine -> mlp_inject.
// Block prologue: token_embedding, hyper_expand. The MTP tap is the wide row after layer 47's mlp_inject. The final hyper
// mix and the LM head are not run: the node computes row p itself.
//
// Row-independent steps (HC mixes, dense projections, router, shared and routed experts, inject/combine) may run on
// padded row counts (the MoE needs >= 16 rows, a multiple of 8); steps that carry state across rows (GDN, QSA, PLE)
// run on the exact rows of the block.
#include "strix/cuda_qwen_prefill.hpp"
#include <cstdlib>
#include "strix/moe_route_bf16.hpp"

#include "strix/cuda_qwen_dense.hpp"
#include "strix/cuda_qwen_gdn.hpp"
#include "strix/cuda_qwen_hc.hpp"
#include "strix/cuda_qwen_moe.hpp"
#include "strix/cuda_qwen_mtp_draft.hpp"
#include "strix/cuda_qwen_plemtp.hpp"
#include "strix/cuda_qwen_qsa.hpp"
#include "strix/cuda_qwen_transcode.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/cuda_vram_guard.hpp"

#include "qwen_amd_math_tables.hpp"

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace strix::cuda::qwen_prefill {
namespace {

namespace am = strix::cuda::amdmath;
namespace hc = strix::cuda::qwen_hc;
namespace dn = strix::cuda::qwen_dense;
namespace gdn = strix::cuda::qwen_gdn;
namespace qsa = strix::cuda::qwen_qsa;
namespace moe = strix::cuda::qwen_moe;
namespace pm = strix::cuda::qwen_plemtp;
namespace md = strix::cuda::qwen_mtp_draft;

using Clock = std::chrono::steady_clock;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

constexpr std::uint32_t kMaxStepRows = 8;    // rows of one session in a window (1 + up to 7 drafts); sessions per window
constexpr std::uint32_t kMaxWindowRows = 16; // rows of a whole cohort window (9..16: the row-independent families run on the
                                             // exact prefill-shaped kernels over 16 padded rows, as a prefill block)
constexpr std::uint32_t kLayers = 48, kHidden = 2560, kWide = 10240, kTopK = 10, kExperts = 512;
constexpr bool is_qsa(std::uint32_t layer) { return layer % 4U == 3U; }
constexpr std::uint32_t gdn_ordinal(std::uint32_t layer) { return layer - (layer + 1U) / 4U; }
constexpr std::uint32_t qsa_ordinal(std::uint32_t layer) { return layer / 4U; }

std::uint32_t round_up(std::uint32_t v, std::uint32_t m) { return (v + m - 1U) / m * m; }

// The parts of a split GGUF ("...-00001-of-00002.gguf"): every part in order; otherwise the file itself.
std::vector<std::string> model_parts(const std::string& path) {
    const auto at = path.rfind("-00001-of-");
    if (at == std::string::npos || path.size() < at + 20) return {path};
    const auto total = static_cast<unsigned>(std::stoul(path.substr(at + 10, 5)));
    std::vector<std::string> parts;
    for (unsigned i = 1; i <= total; ++i) {
        char index[8];
        std::snprintf(index, sizeof(index), "%05u", i);
        parts.push_back(path.substr(0, at + 1) + index + path.substr(at + 6));
    }
    return parts;
}
bool ends_with(const std::string& s, const char* tail) {
    const std::size_t n = std::strlen(tail);
    return s.size() >= n && s.compare(s.size() - n, n, tail) == 0;
}
// Storage the kernels take for a tensor of another type (the GSQ-RCO IQ build): Q8_0 for the A8 dense projections, the HC
// mixes, the PLE projections, the embedding and the head; FP32 for the small vectors and the router; else as stored
// (routed experts: native kernels; BF16 indexer projections; the IQ4_NL PLE table).
enum class Target { native, q8_0, f32 };
Target transcode_target(const gguf::TensorInfo& t) {
    const auto& n = t.name;
    if (t.ggml_type == 0U || n.find("_exps.") != std::string::npos || n.find(".indexer.") != std::string::npos ||
        n.rfind("per_layer_token_embd", 0) == 0)
        return Target::native;
    static const char* q8[] = {".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight", ".attn_q.weight", ".attn_k.weight",
                               ".attn_v.weight", ".attn_output.weight", ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight",
                               ".ffn_down_shexp.weight", "_down.weight", "_up.weight", ".ple_key.weight", ".ple_value.weight",
                               "token_embd.weight"};
    bool to_q8 = n == "output.weight";
    for (const char* tail : q8) to_q8 = to_q8 || ends_with(n, tail);
    if (to_q8) return t.ggml_type == 8U ? Target::native : Target::q8_0;
    return Target::f32;
}
std::uint64_t elements_of(const gguf::TensorInfo& t) {
    std::uint64_t e = 1;
    for (const auto d : t.dimensions) e *= d;
    return e;
}

// Per-layer tensors the step loop reads, resolved once after the upload (no name lookups per step).
enum LayerTensor : std::uint32_t { k_attn_gate_weight, k_attn_k_weight, k_attn_k_norm_weight, k_attn_output_weight, k_attn_q_weight, k_attn_q_norm_weight, k_attn_qkv_weight, k_attn_v_weight, k_ffn_down_exps_weight, k_ffn_down_shexp_weight, k_ffn_gate_exps_weight, k_ffn_gate_inp_weight, k_ffn_gate_inp_shexp_weight, k_ffn_gate_shexp_weight, k_ffn_up_exps_weight, k_ffn_up_shexp_weight, k_indexer_k_norm_weight, k_indexer_k_proj_weight, k_indexer_q_norm_weight, k_indexer_q_proj_weight, k_ssm_a, k_ssm_alpha_weight, k_ssm_beta_weight, k_ssm_conv1d_weight, k_ssm_dt_bias, k_ssm_norm_weight, k_ssm_out_weight, k_hc_attn_norm, k_hc_attn_inject, k_hc_ffn_norm, k_hc_ffn_inject, kLayerTensors };
constexpr const char* kLayerTensorSuffix[] = {".attn_gate.weight", ".attn_k.weight", ".attn_k_norm.weight", ".attn_output.weight", ".attn_q.weight", ".attn_q_norm.weight", ".attn_qkv.weight", ".attn_v.weight", ".ffn_down_exps.weight", ".ffn_down_shexp.weight", ".ffn_gate_exps.weight", ".ffn_gate_inp.weight", ".ffn_gate_inp_shexp.weight", ".ffn_gate_shexp.weight", ".ffn_up_exps.weight", ".ffn_up_shexp.weight", ".indexer.k_norm.weight", ".indexer.k_proj.weight", ".indexer.q_norm.weight", ".indexer.q_proj.weight", ".ssm_a", ".ssm_alpha.weight", ".ssm_beta.weight", ".ssm_conv1d.weight", ".ssm_dt.bias", ".ssm_norm.weight", ".ssm_out.weight", ".hc_attn_norm.weight", ".hc_attn_inject.weight", ".hc_ffn_norm.weight", ".hc_ffn_inject.weight"};

// Device arena of the uploaded model: one cudaMalloc per tensor (keeps alignment simple; ~1200 tensors).
struct Weights {
    std::map<std::string, void*> tensors;
    std::uint64_t bytes = 0;
    void* get(const std::string& name) const {
        const auto it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error("model tensor not uploaded: " + name);
        return it->second;
    }
};

template <class T> T* dev_alloc(std::uint64_t count, const char* what) {
    void* p = nullptr;
    strix::cuda::require_vram(std::max<std::uint64_t>(count, 1) * sizeof(T), what);
    ck(cudaMalloc(&p, std::max<std::uint64_t>(count, 1) * sizeof(T)), what);
    return static_cast<T*>(p);
}

// Activation and window scratch only (never weights): zeroed so rows a block never computes (padding rows of a 16-row
// window, window rows above an earlier window's) hold finite values and a padding row can never raise a non-finite
// failure word. Zeroed on the legacy stream and waited for here: the engine streams are cudaStreamNonBlocking, so an
// unwaited legacy-stream memset is unordered with their work (a plain cudaMemset in dev_alloc raced the async weight
// uploads and transcodes and zeroed parts of the weights).
template <class T> T* dev_zero(std::uint64_t count, const char* what) {
    T* p = dev_alloc<T>(count, what);
    ck(cudaMemsetAsync(p, 0, std::max<std::uint64_t>(count, 1) * sizeof(T), cudaStreamLegacy), what);
    ck(cudaStreamSynchronize(cudaStreamLegacy), what);
    return p;
}

// Per-session decode state (include/strix/cuda_qwen_prefill.hpp, SessionId). Everything a request owns across windows;
// the engine (Prefill::Impl) holds the weights and the shared activation/window scratch.
struct SessionState {
    SessionId id = 0;
    std::vector<std::uint16_t*> kv, raw_index;  // per QSA ordinal: [max_positions][1024] / [max_positions][128]
    std::vector<float*> pooled;                 // per QSA ordinal: pooled index keys [max_blocks][128] (cached across calls)
    std::uint32_t pooled_valid[12] = {};
    std::uint16_t *mtp_kv = nullptr, *mtp_index = nullptr;  // MTP predictor cache rows
    float* mtp_pooled = nullptr;                // the draft attention's pooled keys over the MTP cache
    std::uint32_t mtp_pooled_valid = 0;
    float* gdn_state = nullptr;                 // [36][48*128*128] (current; decode windows swap it with gdn_other)
    float* gdn_other = nullptr;                 // decode: the window's output state (S0 stays in gdn_state)
    float* gdn_base = nullptr;                  // the first GDN state buffer (graph key parity)
    std::uint16_t* gdn_conv = nullptr;          // [36][10240*4]
    std::uint16_t* ple_conv = nullptr;          // [10240*9]
    std::uint16_t* conv_before = nullptr;       // decode: [36][10240*4] window snapshot
    std::uint16_t* ple_before = nullptr;        // decode: [10240*9] window snapshot
    std::uint16_t* held = nullptr;              // h_{length-1}: wide row after layer 47 of the last committed row
    std::uint32_t ngram[2] = {0, 0};
    std::uint32_t ngram_before[2] = {0, 0};
    std::uint64_t length = 0;                   // session length (rows in the state)
    bool prefilling = false;                    // a prefill_chunk_async of this session is in flight
    // ---- open verify window
    bool window_open = false;
    std::uint32_t window_rows = 0;
    std::uint32_t window_offset = 0;            // first row of this session in the cohort window
    std::uint64_t window_position = 0;
    std::vector<std::uint32_t> window_tokens;
    std::vector<void*> allocations;
    std::uint64_t bytes = 0;
};

struct StateCopyStream {
    cudaStream_t stream = nullptr;
    ~StateCopyStream() { if (stream) cudaStreamDestroy(stream); }
};

// One session's rows of a block (prefill: one segment; cohort windows: one per session, concatenated).
struct Segment {
    SessionState* s;
    const std::uint32_t* tokens;   // host; tokens[0] is position p0
    std::uint64_t p0;
    std::uint32_t rows;
    std::uint32_t offset;          // first row in the block
};

// Decode windows: the routed MoE's failure words of every layer, accumulated (read only; the expert arithmetic is
// untouched). sticky[0] |= the CSR summary word 0 (a non-zero word makes the MoE skip every row of that layer, so the
// whole window is suspect); sticky[1 + r] |= row r's status bits (which rows raised it).
__global__ void moe_sticky_kernel(const std::uint32_t* summary, const std::uint32_t* status, std::uint32_t status_words,
                                  std::uint32_t rows, std::uint32_t* sticky) {
    const unsigned t = threadIdx.x;
    if (t == 0U) sticky[0] |= summary[0];
    if (t < rows) sticky[1U + t] |= status[std::uint64_t{t} * status_words];
}

// NVTX ranges per op family in run_block / step_tail (profiling: STRIX_QRTX_NVTX=1). Unset: one cached flag test per
// range, no NVTX call. Ranges cover the host enqueue, so profile with STRIX_QRTX_GRAPHS=0 (a replayed window has none).
bool nvtx_on() {
    static const bool on = [] { const char* v = std::getenv("STRIX_QRTX_NVTX"); return v && v[0] == '1'; }();
    return on;
}
inline void nvtx_push(const char* name) { if (nvtx_on()) nvtxRangePushA(name); }
inline void nvtx_pop() { if (nvtx_on()) nvtxRangePop(); }
// Ranges of one run_block: whatever is still open when it unwinds (an exception mid-enqueue) is popped.
struct NvtxScope {
    int depth = 0;
    void push(const char* name) { if (nvtx_on()) { nvtxRangePushA(name); ++depth; } }
    void pop() { if (depth > 0) { nvtxRangePop(); --depth; } }
    ~NvtxScope() { while (depth > 0) pop(); }
};

// Graph descriptor table (STRIX_QRTX_GRAPH_TABLE, default on): one entry per cohort segment, written from pinned memory
// at the start of every replayed window, so a captured window reads the session's buffers (and its GDN parity) from here
// instead of baking pointers: one graph per rows-per-entry layout serves any sessions.
struct SegTable {
    float* gdn_state;            // S0 of the window (current)
    float* gdn_other;            // the window's output state
    std::uint16_t* gdn_conv;
    std::uint16_t* ple_conv;
    std::uint16_t* conv_before;  // window snapshots
    std::uint16_t* ple_before;
};
constexpr std::uint32_t kTableKey = 0xffffffffU;  // first word of a table-mode graph key (never a session id); multiseg: kTableKey - 1
constexpr std::uint32_t kModeKey = 0xfffffff0U;   // session-keyed graphs: (kModeKey, multiseg) pair at the end (never a session id)

// dst = src (32-bit words), both read from table slots: the snapshot copies of a table-mode window.
__global__ void copy_slot_kernel(std::uint16_t* const* src_slot, std::uint16_t* const* dst_slot, std::uint64_t words) {
    const auto* src = reinterpret_cast<const std::uint32_t*>(*src_slot);
    auto* dst = reinterpret_cast<std::uint32_t*>(*dst_slot);
    for (std::uint64_t i = blockIdx.x * std::uint64_t{blockDim.x} + threadIdx.x; i < words; i += std::uint64_t{gridDim.x} * blockDim.x)
        dst[i] = src[i];
}

// Everything one block computation (run_block + its prefill tail) reads or writes besides the session state and the
// weights: the stream it is enqueued on, the row capacity and every activation / scratch / staging buffer. Impl derives
// from it; a second set (enable_prefill_overlap) is swapped in while a prefill chunk is enqueued on its own stream, so a
// chunk computing concurrently with a decode window shares no buffer with it.
struct BlockFields {
    cudaStream_t stream = nullptr;
    std::uint32_t cap = 0;                     // rows per block (padded capacity)
    std::uint16_t* final_wide = nullptr;       // wide rows after the last layer of the last run_block
    std::uint32_t* ids = nullptr;
    std::uint16_t *emb = nullptr, *wide_a = nullptr, *wide_b = nullptr, *hidden = nullptr, *step_out = nullptr;
    float *injection = nullptr, *normalized = nullptr;
    std::int8_t* codes = nullptr;
    float* deltas = nullptr;
    float* gdn_rows = nullptr;
    void* gdn_scratch = nullptr;
    float *q_gate = nullptr, *qk = nullptr, *qv = nullptr, *index_q = nullptr, *index_k = nullptr, *attn_out = nullptr;
    qsa::Workspace qws{};
    float* logits = nullptr;
    std::uint32_t* routes = nullptr;
    float* route_w = nullptr;
    std::uint16_t *shared_out = nullptr, *routed_out = nullptr, *combined = nullptr;
    float* shared_ws = nullptr;
    // routed MoE per-row buffers
    std::int32_t* moe_sums = nullptr;
    std::uint32_t* moe_status = nullptr;
    float *moe_acc = nullptr, *moe_ddeltas = nullptr, *moe_raw = nullptr;
    std::int8_t *moe_codes = nullptr, *moe_dcodes = nullptr;
    float* moe_deltas = nullptr;
    std::int32_t* moe_dsums = nullptr;
    float* moe_inter = nullptr;                // native path: FP32 h per (row, slot)
    moe::Csr csr{};
    std::uint16_t* tap = nullptr;              // wide rows after layer 47 (MTP)
    // PLE rows
    std::uint64_t* ple_ids_host = nullptr;     // pinned [cap][16] (host hashing; graph memcpy source in decode)
    std::uint64_t* ple_ids = nullptr;
    std::uint16_t* ple_rows = nullptr;
    float *ple_keys = nullptr, *ple_values = nullptr, *ple_gates = nullptr;
    std::uint32_t* ple_status = nullptr;
    // MTP predictor cache rows
    void* mtp_ws = nullptr;
    std::uint32_t* mtp_next = nullptr;
    std::uint32_t* mtp_status = nullptr;
    std::uint32_t* next_pinned = nullptr;      // pinned [cap]: next tokens of prefill_chunk's MTP pairs (no pageable copy)
    std::uint32_t* tok_pinned = nullptr;       // pinned [cap]: prefill_chunk's block tokens
    std::vector<std::uint32_t> next_host;
};

}  // namespace

struct Prefill::Impl : BlockFields {
    Config cfg;
    cudaStream_t copy_stream = nullptr;                    // release_by_layer: per-layer D2H
    cudaStream_t side_stream = nullptr;                    // decode: the shared expert beside the routed MoE
    cudaEvent_t side_fork = nullptr, side_join = nullptr;
    std::vector<cudaEvent_t> layer_done, copy_done;        // [kLayers]
    std::function<void(std::uint32_t)> after_layer;        // called (host) after layer l's launches of a block
    am::Tables tables{};
    qsa::RopeTable* rope = nullptr;
    gguf::FileInfo info;                       // tensors of every part, with the types as stored on the device
    std::string ple_part;                      // the part holding per_layer_token_embd
    bool native_experts = false;               // some layer's experts need the native MoE path
    Weights w;
    std::vector<hc::HcPacked> hc_attn, hc_ffn;  // per layer
    struct LayerPtrs { void* p[kLayerTensors]{}; std::uint32_t gate_type = 0, down_type = 0; };
    std::vector<LayerPtrs> lw;                 // [kLayers]
    hc::HcPacked hc_out{};                     // final hyper mix (decode)
    float* head_logits = nullptr;              // decode: [cap][vocab]
    std::uint32_t* argmax_dev = nullptr;       // decode: [8]
    // ---- decode window (rollback of a partial accept); the window scratch below holds the rows of every session of the
    // open cohort window (session i at its window_offset) until commit_cohort.
    bool window = false;                       // run_block in window mode (GDN rows kept per layer, split state)
    bool cohort_open = false;
    std::vector<SessionState*> cohort;         // the sessions of the open window, in cohort order
    float* win_rows = nullptr;                 // [36][8][gdn_stride]
    float* win_raw = nullptr;                  // [36][8][10240] conv inputs
    // ---- decode step graphs: 13 segments per cohort layout (session, rows, GDN buffer parity per entry); the 12
    // position-dependent QSA attention cores run between them as ordinary launches, per session (STRIX_QRTX_GRAPHS=0:
    // everything eager). LRU-capped.
    struct StepGraph { std::vector<cudaGraphExec_t> seg; std::uint64_t used = 0; };
    using GraphKey = std::vector<std::uint32_t>;
    std::map<GraphKey, StepGraph> graphs;
    std::map<GraphKey, std::uint32_t> warmed;      // eager runs per key before capture (lazy attributes, once flags)
    std::function<void(std::uint32_t)> qsa_cut;    // capture: called at each QSA core instead of launching it
    bool use_graphs = true;
    std::size_t graph_cap = 512;                   // layouts up to 16 rows (C8 with 1-draft windows: 256)
    std::uint64_t graph_clock = 0, graph_captures = 0, graph_replays = 0, graph_evictions = 0, eager_windows = 0;
    void drop_graph(std::map<GraphKey, StepGraph>::iterator it);
    std::uint32_t* ids_pinned = nullptr;           // [8] window tokens (graph memcpy source)
    std::uint32_t* argmax_pinned = nullptr;        // [8]
    std::uint32_t* moe_sticky = nullptr;           // decode: [1 + 8] MoE failure words of the window (moe_sticky_kernel)
    std::uint32_t* moe_sticky_pinned = nullptr;    // [1 + 8]
    SegTable* seg_table = nullptr;                 // decode: device [8] (graph descriptor table)
    SegTable* seg_table_pinned = nullptr;          // [8] host source of the table copy captured in the window prologue
    bool graph_table = true;                       // STRIX_QRTX_GRAPH_TABLE=0: graphs bake session pointers (keyed by session)
    bool table_mode = false;                       // run_block / window_prologue emit table-relative launches (capture)
    void qsa_core(std::uint32_t l, const Segment& seg);
    // every segment's QSA core of layer l: one launch per step across the segments (multiseg, >= 2 segments), else per segment
    void qsa_layer(std::uint32_t l, const Segment* seg, std::uint32_t count);
    bool multiseg = true;                          // STRIX_QRTX_MULTISEG=0: GDN/QSA cores launched per session
    void window_prologue(const Segment* seg, std::uint32_t count);
    void step_tail(std::uint32_t rows);
    void commit_one(SessionState& s, std::uint32_t keep);
    void undo_one(SessionState& s);
    void poison(SessionState& s) noexcept;
    // ---- sessions
    std::map<SessionId, std::unique_ptr<SessionState>> sessions;
    SessionId next_session = 0;
    SessionState& session(SessionId id) const {
        const auto it = sessions.find(id);
        if (it == sessions.end()) throw std::invalid_argument("unknown decode session " + std::to_string(id));
        return *it->second;
    }
    SessionState& new_session();
    // ---- MTP draft chain (decode + sidecar)
    hc::HcPacked mtp_hc_attn{}, mtp_hc_ffn{}, mtp_hc_head{};
    float *mtp_attn_inject = nullptr, *mtp_ffn_inject = nullptr;   // F32 [4][10240] (sidecar Q8_0 widened)
    void* draft_head = nullptr;                // Q8_0 rows of output.weight over the draft vocabulary
    std::uint32_t* draft_ids = nullptr;
    std::uint32_t draft_vocab = 0;
    float* draft_logits = nullptr;
    float* draft_scratch = nullptr;            // routed experts [10][640]
    std::uint16_t* draft_wide = nullptr;       // [3][10240]
    std::uint32_t* draft_token = nullptr;      // device word: the token the next draft step embeds
    std::uint32_t* draft_sel = nullptr;        // device [8][2] token, probability
    // batched draft chains (draft_cohort): row j = the j-th chain by descending depth
    std::uint16_t* bd_wide = nullptr;          // [3][8][10240] attention / mlp / source wide rows
    std::uint16_t* bd_fused = nullptr;         // [8][10240] input-fusion rows
    float* bd_logits = nullptr;                // [8][draft_vocab]
    std::uint32_t* bd_token = nullptr;         // [8] the token each chain's next step embeds
    std::uint32_t* bd_sel = nullptr;           // [8 chains][8 steps][2]
    std::uint16_t* teacher_hidden = nullptr;   // [9][10240]
    std::uint32_t* teacher_next = nullptr;     // [9]
    void upload_draft();
    pm::MtpWeights mtp_weights() const {
        pm::MtpWeights mwts{};
        mwts.token_embd = w.get("token_embd.weight");
        mwts.vocab = 248320;
        mwts.enorm = static_cast<const float*>(mw.get("blk.48.nextn.enorm.weight"));
        mwts.hnorm = static_cast<const float*>(mw.get("blk.48.nextn.hnorm.weight"));
        mwts.eh_q4 = mtp_eh_q4;
        mwts.hc_norm = static_cast<const float*>(mw.get("blk.48.hc_attn_norm.weight"));
        mwts.hc_down = mw.get("blk.48.hc_attn_down.weight");
        mwts.hc_up = mw.get("blk.48.hc_attn_up.weight");
        mwts.index_k = static_cast<const std::uint16_t*>(mw.get("blk.48.indexer.k_proj.weight"));
        mwts.k = mw.get("blk.48.attn_k.weight");
        mwts.v = mw.get("blk.48.attn_v.weight");
        mwts.k_norm = static_cast<const float*>(mw.get("blk.48.attn_k_norm.weight"));
        mwts.rope_cs = reinterpret_cast<const float*>(rope);
        mwts.rope_positions = cfg.max_positions;
        return mwts;
    }
    std::vector<void*> hc_storage;
    std::uint64_t gdn_stride = 0;

    // ---- PLE (one-based layer 2): IQ4_NL table in mapped pinned host memory, gathered on the device
    std::uint8_t* ple_table_host = nullptr;
    const std::uint8_t* ple_table_dev = nullptr;
    std::uint64_t ple_table_rows = 0;
    pm::PleHashContract ple_contract{};

    // ---- MTP predictor cache rows (sidecar)
    bool mtp = false;
    Weights mw;                    // sidecar tensors on the device
    void* mtp_eh_q4 = nullptr;

    // ---- host staging of the exported state (pinned)
    std::uint8_t* host = nullptr;
    std::uint64_t host_bytes = 0;

    std::string describe;
    std::uint32_t transcoded = 0;              // tensors transcoded at load
    StateCopyStream state_copy;
    cudaStream_t state_copy_stream(SessionState& s) {
        if (s.window_open || s.prefilling || cohort_open)
            throw std::logic_error("state copy with an open window or chunk in flight");
        if (!state_copy.stream)
            ck(cudaStreamCreateWithFlags(&state_copy.stream, cudaStreamNonBlocking), "state copy stream");
        ck(cudaStreamSynchronize(stream), "state copy producer");
        return state_copy.stream;
    }
    void copy_rows(SessionState& s, std::uint64_t first, std::uint64_t count, void* data, bool importing);
    void copy_fixed(SessionState& s, void* data, bool importing);

    void upload_model();
    void upload_ple();
    void upload_mtp();
    void allocate();
    void allocate_block(std::uint32_t rows, std::uint32_t qsa_chunk, bool engine_set);
    static void free_block(BlockFields& b) noexcept;
    int stream_priority_high = 0, stream_priority_low = 0;  // cudaDeviceGetStreamPriorityRange (greatest, least)
    // ---- prefill overlap: a second BlockFields set (own stream and scratch) for prefill chunks enqueued while decode
    // windows run on the main stream
    std::unique_ptr<BlockFields> pre;
    cudaEvent_t pre_start = nullptr, pre_end = nullptr;
    SessionState* pre_session = nullptr;       // the session whose chunk is in flight (prefill_wait)
    std::uint64_t pre_length = 0;
    std::uint32_t prefill_enqueue(SessionState& S, const std::vector<std::uint32_t>& tokens, std::uint64_t position,
                                  std::uint64_t from, std::uint32_t max_rows);
    // The block's segments (one per session, rows concatenated). pad: rows of the row-independent steps (>= total rows).
    void run_block(const Segment* seg, std::uint32_t count, std::uint32_t pad);
};

void Prefill::Impl::upload_model() {
    constexpr std::uint64_t kChunk = std::uint64_t{256} << 20;
    std::uint8_t* staging[2] = {nullptr, nullptr};
    ck(cudaMallocHost(&staging[0], kChunk), "pinned staging");
    ck(cudaMallocHost(&staging[1], kChunk), "pinned staging");
    cudaEvent_t done[2];
    ck(cudaEventCreate(&done[0]), "event");
    ck(cudaEventCreate(&done[1]), "event");
    ck(cudaEventRecord(done[0], stream), "event");
    ck(cudaEventRecord(done[1], stream), "event");
    int slot = 0;
    void* scratch = nullptr;  // raw bytes of a tensor that is transcoded on the device
    std::uint64_t scratch_bytes = 0;
    for (const auto& part : model_parts(cfg.model_gguf)) {
    const auto pinfo = gguf::read_file(part);
    std::FILE* f = std::fopen(part.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + part);
    for (auto t : pinfo.tensors) {
        if (t.name.rfind("per_layer_token_embd", 0) == 0) { ple_part = part; info.tensors.push_back(t); continue; }  // PLE (host)
        if (t.name == "output.weight" && !cfg.decode) { info.tensors.push_back(t); continue; }                  // head (node)
        const Target target = transcode_target(t);
        if (target != Target::native && !transcode::supported(t.ggml_type))
            throw std::runtime_error("no decoder for " + t.name + " type " + std::to_string(t.ggml_type));
        void* raw = nullptr;
        if (target == Target::native) raw = dev_alloc<std::uint8_t>(t.byte_size, t.name.c_str());
        else {
            if (scratch_bytes < t.byte_size) {
                ck(cudaStreamSynchronize(stream), "scratch reuse");
                if (scratch) cudaFree(scratch);
                strix::cuda::require_vram(t.byte_size, "transcode scratch");
                ck(cudaMalloc(&scratch, t.byte_size), "transcode scratch");
                scratch_bytes = t.byte_size;
            }
            raw = scratch;
        }
        _fseeki64(f, static_cast<long long>(t.absolute_offset), SEEK_SET);
        for (std::uint64_t off = 0; off < t.byte_size; off += kChunk) {
            const auto n = std::min(kChunk, t.byte_size - off);
            ck(cudaEventSynchronize(done[slot]), "staging reuse");
            if (std::fread(staging[slot], 1, static_cast<std::size_t>(n), f) != n) throw std::runtime_error("short read " + t.name);
            ck(cudaMemcpyAsync(static_cast<std::uint8_t*>(raw) + off, staging[slot], n, cudaMemcpyHostToDevice, stream), "upload");
            ck(cudaEventRecord(done[slot], stream), "event");
            slot ^= 1;
        }
        void* d = raw;
        if (target != Target::native) {
            const auto elements = elements_of(t);
            if (target == Target::q8_0) {
                const auto bytes = transcode::q8_0_bytes(elements);
                d = dev_alloc<std::uint8_t>(bytes, t.name.c_str());
                transcode::to_q8_0(raw, t.ggml_type, elements, d, stream);
                t.ggml_type = 8;
                t.byte_size = bytes;
            } else {
                d = dev_alloc<float>(elements, t.name.c_str());
                transcode::to_f32(raw, t.ggml_type, elements, static_cast<float*>(d), stream);
                t.ggml_type = 0;
                t.byte_size = elements * 4;
            }
            transcoded += 1;
        }
        w.tensors[t.name] = d;
        w.bytes += t.byte_size;
        info.tensors.push_back(t);
    }
    std::fclose(f);
    }
    ck(cudaStreamSynchronize(stream), "upload sync");
    if (scratch) cudaFree(scratch);
    cudaFreeHost(staging[0]);
    cudaFreeHost(staging[1]);
    cudaEventDestroy(done[0]);
    cudaEventDestroy(done[1]);
    // HC: pack every mixer's down/up once; the raw Q8_0 copies are dropped.
    hc_attn.resize(kLayers);
    hc_ffn.resize(kLayers);
    auto pack = [&](const std::string& prefix) {
        void* base = dev_alloc<std::uint8_t>(hc::hc_packed_bytes(), "hc packed");
        hc_storage.push_back(base);
        const auto packed = hc::hc_packed_view(base);
        hc::hc_pack(w.get(prefix + "_down.weight"), w.get(prefix + "_up.weight"), packed, stream);
        return packed;
    };
    for (std::uint32_t l = 0; l < kLayers; ++l) {
        hc_attn[l] = pack("blk." + std::to_string(l) + ".hc_attn");
        hc_ffn[l] = pack("blk." + std::to_string(l) + ".hc_ffn");
    }
    if (cfg.decode) hc_out = pack("output_hc");
    lw.resize(kLayers);
    for (std::uint32_t l = 0; l < kLayers; ++l) {
        const std::string blk = "blk." + std::to_string(l);
        for (std::uint32_t i = 0; i < kLayerTensors; ++i) {
            const auto it = w.tensors.find(blk + kLayerTensorSuffix[i]);
            if (it != w.tensors.end()) lw[l].p[i] = it->second;  // GDN-only / QSA-only tensors are absent elsewhere
        }
        for (const auto& t : info.tensors) {
            if (t.name == blk + ".ffn_gate_exps.weight") lw[l].gate_type = t.ggml_type;
            if (t.name == blk + ".ffn_down_exps.weight") lw[l].down_type = t.ggml_type;
        }
        moe::Weights probe;
        probe.gate_type = lw[l].gate_type;
        probe.down_type = lw[l].down_type;
        native_experts = native_experts || moe::native_types(probe);
    }
    ck(cudaStreamSynchronize(stream), "hc pack");
}

void Prefill::Impl::upload_ple() {
    const gguf::TensorInfo* t = nullptr;
    for (const auto& x : info.tensors) if (x.name == "per_layer_token_embd.weight") t = &x;
    if (!t) throw std::runtime_error("per_layer_token_embd.weight not in the model");
    ple_table_rows = t->byte_size / pm::kIq4RowBytes;
    ck(cudaHostAlloc(&ple_table_host, t->byte_size, cudaHostAllocMapped | cudaHostAllocPortable), "pinned PLE table");
    std::FILE* f = std::fopen(ple_part.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + ple_part);
    _fseeki64(f, static_cast<long long>(t->absolute_offset), SEEK_SET);
    constexpr std::uint64_t kChunk = std::uint64_t{1} << 30;
    for (std::uint64_t off = 0; off < t->byte_size; off += kChunk) {
        const auto n = std::min(kChunk, t->byte_size - off);
        if (std::fread(ple_table_host + off, 1, static_cast<std::size_t>(n), f) != n) throw std::runtime_error("short read PLE table");
    }
    std::fclose(f);
    void* d = nullptr;
    ck(cudaHostGetDevicePointer(&d, ple_table_host, 0), "PLE device view");
    ple_table_dev = static_cast<const std::uint8_t*>(d);
    ple_contract = pm::ple_hash_contract();
}

void Prefill::Impl::upload_mtp() {
    if (cfg.mtp_sidecar_gguf.empty()) return;
    const auto sinfo = gguf::read_file(cfg.mtp_sidecar_gguf);
    std::FILE* f = std::fopen(cfg.mtp_sidecar_gguf.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + cfg.mtp_sidecar_gguf);
    const char* wanted[] = {"blk.48.nextn.enorm.weight", "blk.48.nextn.hnorm.weight", "blk.48.nextn.eh_proj.weight",
                            "blk.48.hc_attn_norm.weight", "blk.48.hc_attn_down.weight", "blk.48.hc_attn_up.weight",
                            "blk.48.indexer.k_proj.weight", "blk.48.attn_k.weight", "blk.48.attn_v.weight",
                            "blk.48.attn_k_norm.weight"};
    std::vector<std::string> names(std::begin(wanted), std::end(wanted));
    if (cfg.decode) {  // the whole MTP layer for the draft chain
        names.clear();
        for (const auto& x : sinfo.tensors) names.push_back(x.name);
    }
    for (const auto& name : names) {
        const gguf::TensorInfo* t = nullptr;
        for (const auto& x : sinfo.tensors) if (x.name == name) t = &x;
        if (!t) throw std::runtime_error("MTP sidecar lacks " + name);
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(t->byte_size));
        _fseeki64(f, static_cast<long long>(t->absolute_offset), SEEK_SET);
        if (std::fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) throw std::runtime_error("short read " + name);
        void* d = dev_alloc<std::uint8_t>(bytes.size(), name.c_str());
        ck(cudaMemcpy(d, bytes.data(), bytes.size(), cudaMemcpyHostToDevice), "mtp upload");
        mw.tensors[name] = d;
        mw.bytes += bytes.size();
    }
    std::fclose(f);
    const std::uint64_t eh_blocks = std::uint64_t{kHidden} * 2 * kHidden / 32;  // [2560][5120]
    mtp_eh_q4 = dev_alloc<std::uint8_t>(eh_blocks * 18, "mtp eh q4");
    pm::launch_mtp_q8_to_q4(mw.get("blk.48.nextn.eh_proj.weight"), eh_blocks, mtp_eh_q4, stream);
    ck(cudaStreamSynchronize(stream), "mtp q4");
    mtp = true;
}

void Prefill::Impl::upload_draft() {
    if (!cfg.decode || !mtp) return;
    const auto pack = [&](const std::string& down, const std::string& up) {
        void* base = dev_alloc<std::uint8_t>(hc::hc_packed_bytes(), "mtp hc packed");
        hc_storage.push_back(base);
        const auto packed = hc::hc_packed_view(base);
        hc::hc_pack(mw.get(down), mw.get(up), packed, stream);
        return packed;
    };
    mtp_hc_attn = pack("blk.48.hc_attn_down.weight", "blk.48.hc_attn_up.weight");
    mtp_hc_ffn = pack("blk.48.hc_ffn_down.weight", "blk.48.hc_ffn_up.weight");
    mtp_hc_head = pack("blk.48.nextn.hc_head_down.weight", "blk.48.nextn.hc_head_up.weight");
    mtp_attn_inject = dev_alloc<float>(4 * kWide, "mtp attn inject");
    mtp_ffn_inject = dev_alloc<float>(4 * kWide, "mtp ffn inject");
    md::q8_to_f32(mw.get("blk.48.hc_attn_inject.weight"), 4 * kWide, mtp_attn_inject, stream);
    md::q8_to_f32(mw.get("blk.48.hc_ffn_inject.weight"), 4 * kWide, mtp_ffn_inject, stream);
    // draft vocabulary: the listed ids (little-endian uint32), else the whole vocabulary
    std::vector<std::uint32_t> ids;
    if (!cfg.draft_vocab.empty()) {
        std::FILE* f = std::fopen(cfg.draft_vocab.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open draft vocabulary " + cfg.draft_vocab);
        std::uint32_t v;
        while (std::fread(&v, 4, 1, f) == 1) if (v < hc::kVocab) ids.push_back(v);
        std::fclose(f);
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    }
    if (ids.empty()) for (std::uint32_t v = 0; v < hc::kVocab; ++v) ids.push_back(v);
    draft_vocab = static_cast<std::uint32_t>(ids.size());
    draft_ids = dev_alloc<std::uint32_t>(ids.size(), "draft ids");
    ck(cudaMemcpy(draft_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "draft ids");
    constexpr std::uint64_t kRowBytes = kHidden / 32 * 34;
    draft_head = dev_alloc<std::uint8_t>(ids.size() * kRowBytes, "draft head");
    md::gather_rows(w.get("output.weight"), kRowBytes, draft_ids, draft_vocab, draft_head, stream);
    draft_logits = dev_alloc<float>(ids.size(), "draft logits");
    draft_scratch = dev_alloc<float>(10 * 640, "draft moe scratch");
    draft_wide = dev_alloc<std::uint16_t>(3 * kWide, "draft wide");
    draft_token = dev_alloc<std::uint32_t>(1, "draft token");
    draft_sel = dev_alloc<std::uint32_t>(2 * kMaxStepRows, "draft selections");
    bd_wide = dev_alloc<std::uint16_t>(3ULL * kMaxStepRows * kWide, "batched draft wide");
    bd_fused = dev_alloc<std::uint16_t>(std::uint64_t{kMaxStepRows} * kWide, "batched draft fused");
    bd_logits = dev_alloc<float>(std::uint64_t{kMaxStepRows} * ids.size(), "batched draft logits");
    bd_token = dev_alloc<std::uint32_t>(kMaxStepRows, "batched draft tokens");
    bd_sel = dev_alloc<std::uint32_t>(2ULL * kMaxStepRows * kMaxStepRows, "batched draft selections");
    teacher_hidden = dev_alloc<std::uint16_t>((kMaxStepRows + 1) * kWide, "teacher hidden");
    teacher_next = dev_alloc<std::uint32_t>(kMaxStepRows + 1, "teacher next");
    ck(cudaStreamSynchronize(stream), "draft upload");
}

// The BlockFields buffers for blocks of up to `rows` rows (the current BlockFields set). qsa_chunk: rows per QSA chunk
// of the attention workspace (the same for every set, so the chunking never differs between sets).
void Prefill::Impl::allocate_block(std::uint32_t rows, std::uint32_t qsa_chunk, bool engine_set) {
    cap = rows;
    const std::uint64_t R = cap;
    ids = dev_zero<std::uint32_t>(R, "ids");
    emb = dev_zero<std::uint16_t>(R * kHidden, "emb");
    wide_a = dev_zero<std::uint16_t>(R * kWide, "wide a");
    wide_b = dev_zero<std::uint16_t>(R * kWide, "wide b");
    tap = dev_zero<std::uint16_t>(R * kWide, "tap");
    hidden = dev_zero<std::uint16_t>(R * kHidden, "hidden");
    step_out = dev_zero<std::uint16_t>(R * kHidden, "step out");
    injection = dev_zero<float>(R * 4, "injection");
    normalized = dev_zero<float>(R * hc::kNormalizedPitch, "hc normalized");
    codes = dev_zero<std::int8_t>(R * 6144, "a8 codes");
    deltas = dev_zero<float>(R * (6144 / 32), "a8 deltas");
    gdn_rows = dev_zero<float>(R * gdn_stride, "gdn rows");
    strix::cuda::require_vram(std::max(gdn::core_scratch_bytes(cap), gdn::multi_scratch_bytes()), "gdn scratch");
    ck(cudaMalloc(&gdn_scratch, std::max(gdn::core_scratch_bytes(cap), gdn::multi_scratch_bytes())), "gdn scratch");
    q_gate = dev_zero<float>(R * qsa::kQGateWidth, "q gate");
    qk = dev_zero<float>(R * qsa::kKvWidth, "k");
    qv = dev_zero<float>(R * qsa::kKvWidth, "v");
    index_q = dev_zero<float>(R * qsa::kIndexQWidth, "index q");
    index_k = dev_zero<float>(R * qsa::kIndexDim, "index k");
    attn_out = dev_zero<float>(R * qsa::kAttnWidth, "attention out");
    qws = qsa::allocate_workspace(qsa_chunk, cfg.max_positions);
    logits = dev_zero<float>(R * kExperts, "router logits");
    routes = dev_zero<std::uint32_t>(R * kTopK, "routes");
    route_w = dev_zero<float>(R * kTopK, "route weights");
    shared_out = dev_zero<std::uint16_t>(R * kHidden, "shared out");
    routed_out = dev_zero<std::uint16_t>(R * kHidden, "routed out");
    combined = dev_zero<std::uint16_t>(R * kHidden, "combined");
    shared_ws = dev_zero<float>(R * 1280, "shared workspace");
    moe_codes = dev_zero<std::int8_t>(R * kHidden, "moe codes");
    moe_deltas = dev_zero<float>(R * (kHidden / 32), "moe deltas");
    moe_sums = dev_zero<std::int32_t>(R * (kHidden / 32), "moe sums");
    moe_status = dev_zero<std::uint32_t>(R * 4, "moe status");
    moe_acc = dev_zero<float>(R * kHidden, "moe accumulation");
    moe_dcodes = dev_zero<std::int8_t>(R * kTopK * 640, "moe down codes");
    moe_ddeltas = dev_zero<float>(R * kTopK * 640 / 32, "moe down deltas");
    moe_dsums = dev_zero<std::int32_t>(R * kTopK * 640 / 32, "moe down sums");
    moe_raw = dev_zero<float>(R * kTopK * kHidden, "moe raw dots");
    if (native_experts) moe_inter = dev_zero<float>(R * kTopK * 2 * 640, "moe native h");
    const auto sz = moe::csr_sizes(cap);
    csr.summary = dev_zero<std::uint32_t>(32, "csr");
    csr.offsets = dev_zero<std::uint32_t>(kExperts + 1, "csr");
    csr.touched = dev_zero<std::uint32_t>(kExperts, "csr");
    csr.rows = dev_zero<std::uint32_t>(sz.assignments, "csr");
    csr.slots = dev_zero<std::uint32_t>(sz.assignments, "csr");
    csr.batch_experts = dev_zero<std::uint32_t>(sz.max_batches, "csr");
    csr.batch_begins = dev_zero<std::uint32_t>(sz.max_batches + 1, "csr");
    csr.tile_count = dev_zero<std::uint32_t>(1, "csr");
    csr.tile_experts = dev_zero<std::uint32_t>(sz.max_tiles, "csr");
    csr.tile_begins = dev_zero<std::uint32_t>(sz.max_tiles + 1, "csr");
    ck(cudaMallocHost(&ple_ids_host, R * pm::kPleHeads * 8), "pinned ple ids");
    ck(cudaMallocHost(&tok_pinned, R * 4), "pinned block tokens");
    ple_ids = dev_zero<std::uint64_t>(R * pm::kPleHeads, "ple ids");
    ple_rows = dev_zero<std::uint16_t>(R * pm::kPleWidth, "ple rows");
    ple_keys = dev_zero<float>(R * pm::kExpanded, "ple keys");
    ple_values = dev_zero<float>(R * pm::kHidden, "ple values");
    ple_gates = dev_zero<float>(R * 4, "ple gates");
    ple_status = dev_zero<std::uint32_t>(1, "ple status");
    if (mtp) {
        strix::cuda::require_vram(pm::mtp_workspace_bytes(cap), "mtp workspace");
        ck(cudaMalloc(&mtp_ws, pm::mtp_workspace_bytes(cap)), "mtp workspace");
        mtp_next = dev_zero<std::uint32_t>(R, "mtp next");
        mtp_status = dev_zero<std::uint32_t>(1, "mtp status");
        ck(cudaMallocHost(&next_pinned, R * 4), "pinned mtp next");
        if (engine_set) next_host.resize(R);  // run()'s pageable MTP staging (the engine set only)
    }
}

void Prefill::Impl::allocate() {
    gdn_stride = 10240 + 6144 + 48 + 48 + 6144;
    const auto rows = round_up(*std::max_element(cfg.block_schedule.begin(), cfg.block_schedule.end()), 16U);
    allocate_block(rows, std::min<std::uint32_t>(rows, 1024U), true);
    if (cfg.decode) {
        head_logits = dev_zero<float>(std::uint64_t{kMaxWindowRows} * hc::kVocab, "head logits");
        argmax_dev = dev_zero<std::uint32_t>(kMaxWindowRows, "argmax");
        ck(cudaMallocHost(&ids_pinned, kMaxWindowRows * 4), "pinned ids");
        ck(cudaMallocHost(&argmax_pinned, kMaxWindowRows * 4), "pinned argmax");
        moe_sticky = dev_zero<std::uint32_t>(1 + kMaxWindowRows, "moe sticky");
        seg_table = dev_zero<SegTable>(kMaxStepRows, "graph descriptor table");
        ck(cudaMallocHost(&seg_table_pinned, kMaxStepRows * sizeof(SegTable)), "pinned descriptor table");
        std::memset(seg_table_pinned, 0, kMaxStepRows * sizeof(SegTable));
        ck(cudaMallocHost(&moe_sticky_pinned, (1 + kMaxWindowRows) * 4), "pinned moe sticky");
        win_rows = dev_zero<float>(std::uint64_t{36} * kMaxWindowRows * gdn_stride, "window gdn rows");
        win_raw = dev_zero<float>(std::uint64_t{36} * kMaxWindowRows * gdn::kConvWidth, "window conv inputs");
    }
    // host staging: every row component's rows plus the fixed blob
    const std::uint64_t P = cfg.max_positions;
    host_bytes = P * (12ULL * 2048 + 12ULL * 256 + 2048 + 256) + 36 * gdn::kMatrixFloats * 4 + 36 * gdn::kConvStateHalfs * 2 +
                 10240ULL * 9 * 2 + 16 + kWide * 2;
    ck(cudaMallocHost(&host, host_bytes), "pinned export staging");
}

// The per-session device state (same buffers and sizes the single-session engine allocated).
SessionState& Prefill::Impl::new_session() {
    auto s = std::make_unique<SessionState>();
    s->id = next_session++;
    const std::uint64_t P = cfg.max_positions;
    const auto alloc = [&](std::uint64_t bytes, const char* what) {
        void* p = nullptr;
        strix::cuda::require_vram(std::max<std::uint64_t>(bytes, 1), what);
        ck(cudaMalloc(&p, std::max<std::uint64_t>(bytes, 1)), what);
        s->allocations.push_back(p);
        s->bytes += bytes;
        return p;
    };
    try {
        for (std::uint32_t o = 0; o < 12; ++o) {
            s->kv.push_back(static_cast<std::uint16_t*>(alloc(P * 1024 * 2, "kv")));
            s->raw_index.push_back(static_cast<std::uint16_t*>(alloc(P * 128 * 2, "raw index")));
            s->pooled.push_back(static_cast<float*>(alloc((P + 3) / 4 * 128 * 4, "pooled index keys")));
        }
        s->gdn_state = static_cast<float*>(alloc(36 * gdn::kMatrixFloats * 4, "gdn state"));
        s->gdn_base = s->gdn_state;  // graph key parity: which of the two buffers is current
        s->gdn_conv = static_cast<std::uint16_t*>(alloc(36 * gdn::kConvStateHalfs * 2, "gdn conv"));
        s->ple_conv = static_cast<std::uint16_t*>(alloc(10240 * 9 * 2, "ple conv"));
        if (cfg.decode) {
            s->gdn_other = static_cast<float*>(alloc(36 * gdn::kMatrixFloats * 4, "gdn state window"));
            s->conv_before = static_cast<std::uint16_t*>(alloc(36 * gdn::kConvStateHalfs * 2, "conv before"));
            s->ple_before = static_cast<std::uint16_t*>(alloc(10240 * 9 * 2, "ple conv before"));
        }
        if (mtp) {
            s->mtp_kv = static_cast<std::uint16_t*>(alloc(P * 1024 * 2, "mtp kv"));
            s->mtp_pooled = static_cast<float*>(alloc((P + 3) / 4 * 128 * 4, "mtp pooled keys"));
            s->mtp_index = static_cast<std::uint16_t*>(alloc(P * 128 * 2, "mtp index"));
            if (cfg.decode) s->held = static_cast<std::uint16_t*>(alloc(kWide * 2, "held row"));
        }
    } catch (...) {
        for (void* p : s->allocations) cudaFree(p);
        throw;
    }
    auto& ref = *s;
    sessions.emplace(ref.id, std::move(s));
    return ref;
}

void Prefill::Impl::drop_graph(std::map<GraphKey, StepGraph>::iterator it) {
    for (auto exec : it->second.seg) cudaGraphExecDestroy(exec);
    graphs.erase(it);
    ++graph_evictions;
}

Prefill::Prefill(const Config& config) : impl_(std::make_unique<Impl>()) {
    if (strix::moe_route_bf16_environment_requested(std::getenv("STRIX_QPF2K_MOE_ROUTE_BF16"),
        std::getenv("STRIX_BASELINE"), std::getenv("STRIX_QPF2K_DEFAULTS")))
        throw std::invalid_argument("MoE route BF16 candidate declined: CUDA writer/reader unsupported");
    auto& m = *impl_;
    m.cfg = config;
    validate_blocks(config.block_schedule);
    (void)plan_blocks(1, config.block_schedule, config.tail_large, config.final_block_rows);
    ck(cudaSetDevice(config.device), "set device");
    // Priorities change only which ready blocks the GPU schedules first, never what a kernel computes: the engine and
    // side streams (decode windows) get the highest, the prefill-overlap stream the lowest (decode first).
    if (cudaDeviceGetStreamPriorityRange(&m.stream_priority_low, &m.stream_priority_high) != cudaSuccess) {
        (void)cudaGetLastError();
        m.stream_priority_low = m.stream_priority_high = 0;
    }
    ck(cudaStreamCreateWithPriority(&m.stream, cudaStreamNonBlocking, m.stream_priority_high), "stream");
    if (config.release_by_layer) {
        ck(cudaStreamCreateWithFlags(&m.copy_stream, cudaStreamNonBlocking), "copy stream");
        m.layer_done.resize(kLayers);
        m.copy_done.resize(kLayers);
        for (std::uint32_t l = 0; l < kLayers; ++l) {
            ck(cudaEventCreateWithFlags(&m.layer_done[l], cudaEventDisableTiming), "layer event");
            ck(cudaEventCreateWithFlags(&m.copy_done[l], cudaEventDisableTiming), "copy event");
        }
    }
    if (config.decode) {
        const char* v = std::getenv("STRIX_QRTX_SIDE_SHARED");
        if (!(v && v[0] == '0')) {
            ck(cudaStreamCreateWithPriority(&m.side_stream, cudaStreamNonBlocking, m.stream_priority_high), "side stream");
            ck(cudaEventCreateWithFlags(&m.side_fork, cudaEventDisableTiming), "side event");
            ck(cudaEventCreateWithFlags(&m.side_join, cudaEventDisableTiming), "side event");
        }
    }
    m.tables = am::load_tables(config.tables_dir);
    {
        std::FILE* f = std::fopen(config.rope_table.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open rope table " + config.rope_table);
        std::vector<qsa::RopeTable> cs(static_cast<std::size_t>(config.max_positions) * qsa::kRotaryPairs);
        if (std::fread(cs.data(), sizeof(qsa::RopeTable), cs.size(), f) != cs.size())
            throw std::runtime_error("rope table shorter than max_positions");
        std::fclose(f);
        m.rope = dev_alloc<qsa::RopeTable>(cs.size(), "rope");
        ck(cudaMemcpy(m.rope, cs.data(), cs.size() * sizeof(qsa::RopeTable), cudaMemcpyHostToDevice), "rope upload");
    }
    const auto t0 = Clock::now();
    m.upload_model();
    m.upload_ple();
    m.upload_mtp();
    m.allocate();
    m.upload_draft();
    m.new_session();  // session 0: the single-session API
    {
        const char* v = std::getenv("STRIX_QRTX_GRAPHS");
        m.use_graphs = !(v && v[0] == '0');
        if (const char* c = std::getenv("STRIX_QRTX_GRAPH_CACHE")) m.graph_cap = std::max<std::size_t>(1, std::strtoul(c, nullptr, 10));
        const char* t = std::getenv("STRIX_QRTX_GRAPH_TABLE");
        m.graph_table = !(t && t[0] == '0');
        const char* ms = std::getenv("STRIX_QRTX_MULTISEG");
        m.multiseg = !(ms && ms[0] == '0');
    }
    char buf[384];
    std::snprintf(buf, sizeof(buf), "{\"weights_gb\":%.2f,\"upload_s\":%.1f,\"block_rows\":%u,\"max_positions\":%u,"
                  "\"transcoded_tensors\":%u,\"native_experts\":%s}",
                  static_cast<double>(m.w.bytes) / 1e9, std::chrono::duration<double>(Clock::now() - t0).count(), m.cap,
                  config.max_positions, m.transcoded, m.native_experts ? "true" : "false");
    m.describe = buf;
}

Prefill::~Prefill() = default;

std::string Prefill::describe() const { return impl_->describe; }

void Prefill::Impl::run_block(const Segment* seg, std::uint32_t count, std::uint32_t pad) {
    NvtxScope nv;
    std::uint32_t rows = 0;  // all rows of the block
    for (std::uint32_t i = 0; i < count; ++i) rows += seg[i].rows;
    if (pad > rows) ck(cudaMemsetAsync(ids, 0, std::uint64_t{pad} * 4, stream), "ids clear");
    for (std::uint32_t i = 0; i < count; ++i)
        ck(cudaMemcpyAsync(ids + seg[i].offset, seg[i].tokens, std::uint64_t{seg[i].rows} * 4, cudaMemcpyHostToDevice, stream), "ids");
    nv.push("embed");
    hc::token_embedding(w.get("token_embd.weight"), ids, pad, emb, stream);
    hc::hyper_expand(emb, pad, wide_a, stream);
    nv.pop();
    std::uint16_t* wide = wide_a;
    std::uint16_t* wide_next = wide_b;
    const auto mix = [&](const void* norm, const void* inject_w, const hc::HcPacked& packed) {
        hc::HcMixArgs a;
        a.input = wide;
        a.norm = static_cast<const float*>(norm);
        a.inject = static_cast<const float*>(inject_w);
        a.packed = packed;
        a.rows = pad;
        a.normalized = normalized;
        a.injection = injection;
        a.output = hidden;
        hc::hc_mix(a, tables, stream);
    };
    const auto inject = [&](const std::uint16_t* value) {
        hc::inject(wide, value, injection, pad, wide_next, stream);
        std::swap(wide, wide_next);
    };
    for (std::uint32_t l = 0; l < kLayers; ++l) {
        if (l == 1) {
            // PLE (one-based layer 2): host hashing, device gather from the mapped table, key/value projections, tail
            for (std::uint32_t i = 0; i < count; ++i)
                pm::ple_hash_rows(ple_contract, seg[i].tokens, seg[i].rows, seg[i].p0, seg[i].s->ngram,
                                  ple_ids_host + std::uint64_t{seg[i].offset} * pm::kPleHeads);
            ck(cudaMemcpyAsync(ple_ids, ple_ids_host, std::uint64_t{rows} * pm::kPleHeads * 8, cudaMemcpyHostToDevice, stream),
               "ple ids");
            pm::launch_ple_gather_iq4nl(ple_table_dev, ple_table_rows, ple_ids, rows, ple_rows, pm::kPleWidth, ple_status, stream);
            pm::launch_ple_project(ple_rows, pm::kPleWidth, rows, w.get("blk.1.ple_key.weight"), pm::kPleWidth, pm::kExpanded,
                                   ple_keys, pm::kExpanded, stream);
            pm::launch_ple_project(ple_rows, pm::kPleWidth, rows, w.get("blk.1.ple_value.weight"), pm::kPleWidth, pm::kHidden,
                                   ple_values, pm::kHidden, stream);
            // the tail carries the session's PLE conv history: per session, on its rows
            for (std::uint32_t i = 0; i < count; ++i) {
                const std::uint64_t off = seg[i].offset;
                pm::PleTailArgs ta{};
                ta.keys = ple_keys + off * pm::kExpanded; ta.key_stride = pm::kExpanded;
                ta.values = ple_values + off * pm::kHidden; ta.value_stride = pm::kHidden;
                ta.input = wide + off * kWide; ta.input_stride = kWide;
                ta.output = wide + off * kWide; ta.output_stride = kWide;
                ta.conv_state = seg[i].s->ple_conv;
                if (table_mode) { ta.conv_state = nullptr; ta.conv_state_slot = &seg_table[i].ple_conv; }
                ta.norm_key = static_cast<const float*>(w.get("blk.1.ple_norm_key.weight"));
                ta.norm_query = static_cast<const float*>(w.get("blk.1.ple_norm_query.weight"));
                ta.norm_conv = static_cast<const float*>(w.get("blk.1.ple_norm_conv.weight"));
                ta.conv = static_cast<const float*>(w.get("blk.1.ple_conv1d.weight"));
                ta.gates = ple_gates + off * 4;
                ta.rows = seg[i].rows;
                ta.epsilon = 1e-6F;
                pm::launch_ple_tail(ta, tables, stream);
            }
        }
        nv.push("hc");
        mix(lw[l].p[k_hc_attn_norm], lw[l].p[k_hc_attn_inject], hc_attn[l]);
        nv.pop();
        if (!is_qsa(l)) {
            // gated delta net: qkv||z (A8), core, ssm_out (A8)
            const auto o = gdn_ordinal(l);
            float* base = window ? win_rows + std::uint64_t{o} * kMaxWindowRows * gdn_stride : gdn_rows;
            const gdn::Rows gr{base, base + 10240, base + 16384, base + 16432, base + 16480, gdn_stride};
            dn::Site in;
            in.store = dn::Store::f32_round_bf16;
            in.width = kHidden;
            in.segments = 2;
            in.segment[0] = {lw[l].p[k_attn_qkv_weight], 10240, gr.qkv, gdn_stride};
            in.segment[1] = {lw[l].p[k_attn_gate_weight], 6144, gr.z, gdn_stride};
            nv.push("dense_qkv");
            ck(dn::quantize_project(stream, in, pad, dn::Input::bf16, hidden, kHidden, codes, deltas, cap), "gdn in");
            nv.pop();
            nv.push("gdn");
            gdn::Weights gw{static_cast<const float*>(lw[l].p[k_ssm_beta_weight]), static_cast<const float*>(lw[l].p[k_ssm_alpha_weight]),
                            static_cast<const float*>(lw[l].p[k_ssm_conv1d_weight]), static_cast<const float*>(lw[l].p[k_ssm_dt_bias]),
                            static_cast<const float*>(lw[l].p[k_ssm_a]), static_cast<const float*>(lw[l].p[k_ssm_norm_weight])};
            // the recurrence and conv history are the session's: one multi-segment launch per step (each grid slice one
            // session, row-wise steps over all rows), or per session on its rows
            if (window && multiseg && count > 1) {
                gdn::Segments sg;
                sg.n = count;
                for (std::uint32_t i = 0; i < count; ++i) {
                    SessionState& ss = *seg[i].s;
                    auto& x = sg.s[i];
                    x.row = seg[i].offset;
                    x.count = seg[i].rows;
                    if (table_mode) {
                        x.conv = gdn::DevRef<std::uint16_t>{nullptr, &seg_table[i].gdn_conv, std::uint64_t{o} * gdn::kConvStateHalfs};
                        x.state_in = gdn::DevRef<float>{nullptr, &seg_table[i].gdn_state, std::uint64_t{o} * gdn::kMatrixFloats};
                        x.state_out = gdn::DevRef<float>{nullptr, &seg_table[i].gdn_other, std::uint64_t{o} * gdn::kMatrixFloats};
                    } else {
                        x.conv = gdn::DevRef<std::uint16_t>{ss.gdn_conv + o * gdn::kConvStateHalfs};
                        x.state_in = gdn::DevRef<float>{ss.gdn_state + o * gdn::kMatrixFloats};
                        x.state_out = gdn::DevRef<float>{ss.gdn_other + o * gdn::kMatrixFloats};
                    }
                }
                gdn::launch_core_multi(hidden, kHidden, gr, sg, win_raw + std::uint64_t{o} * kMaxWindowRows * gdn::kConvWidth, gdn_scratch,
                    gw, tables, stream);
            } else
            for (std::uint32_t i = 0; i < count; ++i) {
                const std::uint64_t off = seg[i].offset;
                SessionState& ss = *seg[i].s;
                float* sb = base + off * gdn_stride;
                const gdn::Rows grs{sb, sb + 10240, sb + 16384, sb + 16432, sb + 16480, gdn_stride};
                if (window && table_mode)
                    gdn::launch_core_split_ref(hidden + off * kHidden, kHidden, grs, seg[i].rows,
                        gdn::DevRef<std::uint16_t>{nullptr, &seg_table[i].gdn_conv, std::uint64_t{o} * gdn::kConvStateHalfs},
                        gdn::DevRef<float>{nullptr, &seg_table[i].gdn_state, std::uint64_t{o} * gdn::kMatrixFloats},
                        gdn::DevRef<float>{nullptr, &seg_table[i].gdn_other, std::uint64_t{o} * gdn::kMatrixFloats},
                        win_raw + (std::uint64_t{o} * kMaxWindowRows + off) * gdn::kConvWidth, gdn_scratch, gw, tables, stream);
                else if (window)
                    gdn::launch_core_split(hidden + off * kHidden, kHidden, grs, seg[i].rows, ss.gdn_conv + o * gdn::kConvStateHalfs,
                        ss.gdn_state + o * gdn::kMatrixFloats, ss.gdn_other + o * gdn::kMatrixFloats,
                        win_raw + (std::uint64_t{o} * kMaxWindowRows + off) * gdn::kConvWidth, gdn_scratch, gw, tables, stream);
                else
                    gdn::launch_core(hidden + off * kHidden, kHidden, grs, seg[i].rows, ss.gdn_conv + o * gdn::kConvStateHalfs,
                                     ss.gdn_state + o * gdn::kMatrixFloats, gdn_scratch, gw, tables, stream);
            }
            nv.pop();
            nv.push("dense_out");
            dn::Site out;
            out.store = dn::Store::bf16;
            out.width = 6144;
            out.segments = 1;
            out.segment[0] = {lw[l].p[k_ssm_out_weight], kHidden, step_out, kHidden};
            ck(dn::quantize_project(stream, out, pad, dn::Input::f32, gr.core, gdn_stride, codes, deltas, cap), "gdn out");
            nv.pop();
        } else {
            dn::Site in;
            in.store = dn::Store::f32_round_bf16;
            in.width = kHidden;
            in.segments = 3;
            in.segment[0] = {lw[l].p[k_attn_q_weight], qsa::kQGateWidth, q_gate, qsa::kQGateWidth};
            in.segment[1] = {lw[l].p[k_attn_k_weight], qsa::kKvWidth, qk, qsa::kKvWidth};
            in.segment[2] = {lw[l].p[k_attn_v_weight], qsa::kKvWidth, qv, qsa::kKvWidth};
            nv.push("dense_qkv");
            ck(dn::quantize_project(stream, in, pad, dn::Input::bf16, hidden, kHidden, codes, deltas, cap), "qsa in");
            nv.pop();
            nv.push("qsa");
            if (qsa_cut) qsa_cut(l);
            else qsa_layer(l, seg, count);
            nv.pop();
            nv.push("dense_out");
            dn::Site out;
            out.store = dn::Store::bf16;
            out.width = qsa::kAttnWidth;
            out.segments = 1;
            out.segment[0] = {lw[l].p[k_attn_output_weight], kHidden, step_out, kHidden};
            ck(dn::quantize_project(stream, out, pad, dn::Input::f32, attn_out, qsa::kAttnWidth, codes, deltas, cap), "qsa out");
            nv.pop();
        }
        nv.push("hc");
        inject(step_out);
        mix(lw[l].p[k_hc_ffn_norm], lw[l].p[k_hc_ffn_inject], hc_ffn[l]);
        nv.pop();
        nv.push("moe_route");
        hc::route_logits(hidden, static_cast<const float*>(lw[l].p[k_ffn_gate_inp_weight]), pad, logits, stream);
        hc::route_select(logits, pad, kTopK, true, tables, routes, route_w, stream);
        nv.pop();
        dn::SharedExpert sw;
        sw.gate_q8 = lw[l].p[k_ffn_gate_shexp_weight];
        sw.up_q8 = lw[l].p[k_ffn_up_shexp_weight];
        sw.down_q8 = lw[l].p[k_ffn_down_shexp_weight];
        sw.scalar = static_cast<const float*>(lw[l].p[k_ffn_gate_inp_shexp_weight]);
        // Decode windows: the shared expert runs on the side stream while the routed MoE runs here (disjoint scratch: the
        // shared expert owns codes/deltas/shared_ws/shared_out until the join before the combine). Same kernels, same bits.
        const bool side = window && side_stream;
        if (side) {
            ck(cudaEventRecord(side_fork, stream), "side fork");
            ck(cudaStreamWaitEvent(side_stream, side_fork, 0), "side wait");
        }
        nv.push("moe_shared");
        ck(dn::shared_expert(side ? side_stream : stream, pad, sw, hidden, kHidden, shared_ws, 1280, shared_out, kHidden, codes, deltas,
                             cap, tables), "shared expert");
        nv.pop();
        if (side) ck(cudaEventRecord(side_join, side_stream), "side join");
        moe::Rows mr;
        mr.input = {hidden, kHidden * 2};
        mr.routes = {routes, kTopK * 4};
        mr.route_weights = {route_w, kTopK * 4};
        mr.codes = {moe_codes, kHidden};
        mr.deltas = {moe_deltas, kHidden / 32 * 4};
        mr.sums = {moe_sums, kHidden / 32 * 4};
        mr.status = {moe_status, 16};
        mr.intermediates = {moe_inter, kTopK * 2 * 640 * 4};
        mr.accumulation = {moe_acc, kHidden * 4};
        mr.down_codes = {moe_dcodes, kTopK * 640};
        mr.down_deltas = {moe_ddeltas, kTopK * 640 / 32 * 4};
        mr.down_sums = {moe_dsums, kTopK * 640 / 32 * 4};
        mr.rawdots = {moe_raw, kTopK * kHidden * 4};
        mr.output = {routed_out, kHidden * 2};
        mr.csr = csr;
        moe::Weights ew;
        ew.gate_type = lw[l].gate_type;
        ew.down_type = lw[l].down_type;
        ew.gate = static_cast<const std::uint8_t*>(lw[l].p[k_ffn_gate_exps_weight]);
        ew.up = static_cast<const std::uint8_t*>(lw[l].p[k_ffn_up_exps_weight]);
        ew.down = static_cast<const std::uint8_t*>(lw[l].p[k_ffn_down_exps_weight]);
        nv.push("moe_routed");
        moe::enqueue(mr, pad, kHidden, ew, tables, nullptr, stream);
        nv.pop();
        if (window) {
            moe_sticky_kernel<<<1, 32, 0, stream>>>(csr.summary, moe_status, 4, pad, moe_sticky);
            ck(cudaGetLastError(), "moe sticky");
        }
        if (side) ck(cudaStreamWaitEvent(stream, side_join, 0), "side join wait");
        nv.push("hc");
        hc::combine(shared_out, routed_out, pad, combined, stream);
        inject(combined);
        nv.pop();
        if (l == kLayers - 1U) ck(cudaMemcpyAsync(tap, wide, std::uint64_t{pad} * kWide * 2, cudaMemcpyDeviceToDevice, stream), "tap");
        if (after_layer) after_layer(l);
    }
    final_wide = wide;
    ck(cudaGetLastError(), "block launches");
}

void Prefill::Impl::qsa_core(std::uint32_t l, const Segment& seg) {
    const std::uint64_t r = seg.offset;
    const qsa::RowBuffers rb{hidden + r * kHidden, q_gate + r * qsa::kQGateWidth, qk + r * qsa::kKvWidth, qv + r * qsa::kKvWidth,
                             index_q + r * qsa::kIndexQWidth, index_k + r * qsa::kIndexDim, attn_out + r * qsa::kAttnWidth};
    const qsa::LayerWeights qw{static_cast<const std::uint16_t*>(lw[l].p[k_indexer_q_proj_weight]),
                               static_cast<const std::uint16_t*>(lw[l].p[k_indexer_k_proj_weight]),
                               static_cast<const float*>(lw[l].p[k_attn_q_norm_weight]),
                               static_cast<const float*>(lw[l].p[k_attn_k_norm_weight]),
                               static_cast<const float*>(lw[l].p[k_indexer_q_norm_weight]),
                               static_cast<const float*>(lw[l].p[k_indexer_k_norm_weight])};
    const auto o = qsa_ordinal(l);
    SessionState& ss = *seg.s;
    const qsa::LayerState st{ss.kv[o], ss.raw_index[o], ss.pooled[o], &ss.pooled_valid[o]};
    qsa::layer_core(rb, seg.rows, seg.p0, qw, st, rope, tables, qws, stream);
}

void Prefill::Impl::qsa_layer(std::uint32_t l, const Segment* seg, std::uint32_t count) {
    if (!(window && multiseg && count > 1)) {
        for (std::uint32_t i = 0; i < count; ++i) qsa_core(l, seg[i]);
        return;
    }
    const qsa::RowBuffers rb{hidden, q_gate, qk, qv, index_q, index_k, attn_out};
    const qsa::LayerWeights qw{static_cast<const std::uint16_t*>(lw[l].p[k_indexer_q_proj_weight]),
                               static_cast<const std::uint16_t*>(lw[l].p[k_indexer_k_proj_weight]),
                               static_cast<const float*>(lw[l].p[k_attn_q_norm_weight]),
                               static_cast<const float*>(lw[l].p[k_attn_k_norm_weight]),
                               static_cast<const float*>(lw[l].p[k_indexer_q_norm_weight]),
                               static_cast<const float*>(lw[l].p[k_indexer_k_norm_weight])};
    const auto o = qsa_ordinal(l);
    qsa::SegmentState st[kMaxStepRows];
    for (std::uint32_t i = 0; i < count; ++i) {
        if (i > 0 && seg[i].offset != seg[i - 1].offset + seg[i - 1].rows) throw std::logic_error("qsa_layer: segments not contiguous");
        SessionState& ss = *seg[i].s;
        st[i] = qsa::SegmentState{seg[i].rows, seg[i].p0, qsa::LayerState{ss.kv[o], ss.raw_index[o], ss.pooled[o], &ss.pooled_valid[o]}};
    }
    if (seg[0].offset != 0) throw std::logic_error("qsa_layer: segments must start at row 0");
    qsa::layer_core_multi(rb, st, count, qw, rope, tables, qws, stream);
}

Timing Prefill::run(const std::vector<std::uint32_t>& tokens, std::uint64_t position, const Sink& sink, const RunOptions& options) {
    return run(0, tokens, position, sink, options);
}

Timing Prefill::run(SessionId session, const std::vector<std::uint32_t>& tokens, std::uint64_t position, const Sink& sink,
                    const RunOptions& options) {
    auto& m = *impl_;
    auto& S = m.session(session);
    if (m.cohort_open) throw std::logic_error("run() with an open decode window");
    if (S.prefilling || m.pre_session) throw std::logic_error("run() while an overlapped prefill chunk is in flight");
    if (position > tokens.size() || position == 0) throw std::invalid_argument("prefill position out of range");
    if (position > m.cfg.max_positions) throw std::invalid_argument("prompt longer than max_positions");
    const auto plan = plan_blocks(position, m.cfg.block_schedule,
        options.tail_large.value_or(m.cfg.tail_large),
        options.final_block_rows.value_or(m.cfg.final_block_rows));
    auto* trace = options.trace;
    // Stream callbacks timestamp observed completion without host synchronization or CUDA API calls.
    // They exist only for traced requests and finish before run returns.
    const auto mark_stream = [&](cudaStream_t stream, const char* stage, std::uint64_t id,
                                 const std::string& name, std::uint64_t first, std::uint64_t rows, std::uint64_t bytes) {
        if (!trace) return;
        struct Mark { qwen_prefill_trace::Trace* trace; std::string stage, name; std::uint64_t id, first, rows, bytes; };
        auto* mark = new Mark{trace, stage, name, id, first, rows, bytes};
        const auto status = cudaLaunchHostFunc(stream, [](void* data) {
            std::unique_ptr<Mark> p(static_cast<Mark*>(data));
            try { p->trace->mark(p->stage, p->id, p->name, p->first, p->rows, p->bytes); }
            catch (...) { p->trace->drop(); }
        }, mark);
        if (status != cudaSuccess) { delete mark; ck(status, "trace callback"); }
    };
    struct TraceDrain {
        Impl& m; bool active;
        ~TraceDrain() { if (active) { cudaStreamSynchronize(m.stream); if (m.copy_stream) cudaStreamSynchronize(m.copy_stream); } }
    } drain{m, trace != nullptr};
    const auto emit = [&](const Piece& p) {
        if (!trace) { sink(p); return; }
        trace->mark("piece", static_cast<std::uint64_t>(p.data-m.host)+1, p.component, p.first, p.count, p.bytes);
        Piece copy = p;
        copy.trace_id = static_cast<std::uint64_t>(p.data-m.host)+1;
        sink(copy);
    };
    const auto t0 = Clock::now();
    Timing timing;
    // fresh state
    for (std::uint32_t o = 0; o < 12; ++o) {
        ck(cudaMemsetAsync(S.kv[o], 0, position * 2048, m.stream), "kv clear");
        ck(cudaMemsetAsync(S.raw_index[o], 0, position * 256, m.stream), "index clear");
    }
    ck(cudaMemsetAsync(S.gdn_state, 0, 36 * gdn::kMatrixFloats * 4, m.stream), "gdn clear");
    for (auto& v : S.pooled_valid) v = 0;
    S.mtp_pooled_valid = 0;
    ck(cudaMemsetAsync(S.gdn_conv, 0, 36 * gdn::kConvStateHalfs * 2, m.stream), "conv clear");
    ck(cudaMemsetAsync(S.ple_conv, 0, 10240 * 9 * 2, m.stream), "ple conv clear");
    // release per layer only when the state leaves the device at all
    const bool by_layer = m.cfg.release_by_layer && m.cfg.export_state;
    std::uint64_t p0 = 0;
    std::size_t k = 0;
    std::uint8_t* out = m.host;
    const std::uint8_t* held = nullptr;
    S.ngram[0] = S.ngram[1] = 0;
    while (p0 < position) {
        const auto rows = plan[k];
        mark_stream(m.stream, "block_start", k, {}, p0, rows, 0);
        const auto tb = Clock::now();
        const auto compute = [&] {
            mark_stream(m.stream, "compute_start", k, {}, p0, rows, 0);
            const Segment seg{&S, tokens.data() + p0, p0, rows, 0};
            m.run_block(&seg, 1, std::max<std::uint32_t>(16U, round_up(rows, 8U)));
            if (S.held && p0 + rows == position)  // decode: h_{position-1} seeds the first draft chain
                ck(cudaMemcpyAsync(S.held, m.tap + std::uint64_t{rows - 1} * kWide, kWide * 2, cudaMemcpyDeviceToDevice, m.stream), "held");
            mark_stream(m.stream, "compute_end", k, {}, p0, rows, 0);
            if (m.mtp && p0+rows == position) {
                const auto pairs = position-1-p0;
                const auto row_bytes = by_layer ? 0 : std::uint64_t{rows}*27648;
                const auto held_id = static_cast<std::uint64_t>(out-m.host)+row_bytes+pairs*2304+1;
                mark_stream(m.stream, "gpu_ready", held_id, "mtp.held", 0, 0, kWide*2);
            }
        };
        struct Pending { std::string name; std::uint64_t first, count; std::uint8_t* data; std::uint64_t bytes; bool fixed = false; const void* dev = nullptr; };
        std::vector<Pending> pend;
        const bool last_block = p0 + rows == position;
        if (by_layer) {
            // Per layer l of every block: the block's QSA rows of l; in the last block also GDN l's recurrent matrix +
            // conv slices (final after it) and the PLE conv after layer 1. D2H on copy_stream after layer_done[l]; an
            // emitter thread hands them over while the next layers compute.
            std::vector<std::vector<Pending>> per(kLayers);
            std::atomic<int> launched{-1};
            m.after_layer = [&](std::uint32_t l) {
                const auto copy = [&](const std::string& name, bool fixed_piece, std::uint64_t first, std::uint64_t count,
                                      const void* src, std::uint64_t bytes) {
                    per[l].push_back({name, first, count, out, bytes, fixed_piece, src});
                    out += bytes;
                };
                if (is_qsa(l)) {
                    const auto o = qsa_ordinal(l);
                    const auto layer = std::to_string(l);
                    copy("kv.L" + layer, false, p0, rows, S.kv[o] + p0 * 1024, std::uint64_t{rows} * 2048);
                    copy("index.L" + layer, false, p0, rows, S.raw_index[o] + p0 * 128, std::uint64_t{rows} * 256);
                } else if (last_block) {
                    const auto g = gdn_ordinal(l);
                    const std::uint64_t mb = std::uint64_t{gdn::kMatrixFloats} * 4, cb = std::uint64_t{gdn::kConvStateHalfs} * 2;
                    copy("recurrent_matrix", true, g * mb, 0, S.gdn_state + std::uint64_t{g} * gdn::kMatrixFloats, mb);
                    copy("gdn_convolution", true, g * cb, 0, S.gdn_conv + std::uint64_t{g} * gdn::kConvStateHalfs, cb);
                }
                if (l == 1 && last_block) copy("ple_convolution", true, 0, 0, S.ple_conv, 10240ULL * 9 * 2);
                for (const auto& p : per[l])
                    mark_stream(m.stream, "gpu_ready", static_cast<std::uint64_t>(p.data-m.host)+1,
                        p.name, p.first, p.count, p.bytes);
                ck(cudaEventRecord(m.layer_done[l], m.stream), "layer done");
                ck(cudaStreamWaitEvent(m.copy_stream, m.layer_done[l], 0), "copy wait");
                for (const auto& p : per[l]) {
                    ck(cudaMemcpyAsync(p.data, p.dev, p.bytes, cudaMemcpyDeviceToHost, m.copy_stream), "layer d2h");
                    mark_stream(m.copy_stream, "d2h_done", static_cast<std::uint64_t>(p.data-m.host)+1,
                        p.name, p.first, p.count, p.bytes);
                }
                ck(cudaEventRecord(m.copy_done[l], m.copy_stream), "copy done");
                launched.store(static_cast<int>(l), std::memory_order_release);
            };
            std::exception_ptr emit_error;
            std::thread emitter([&] {
                try {
                    for (std::uint32_t l = 0; l < kLayers; ++l) {
                        while (launched.load(std::memory_order_acquire) < static_cast<int>(l)) std::this_thread::yield();
                        if (launched.load(std::memory_order_acquire) >= static_cast<int>(kLayers)) return;  // aborted
                        ck(cudaEventSynchronize(m.copy_done[l]), "layer copy");
                        for (const auto& p : per[l]) emit(Piece{p.fixed, p.name, p.first, p.count, p.data, p.bytes});
                    }
                } catch (...) { emit_error = std::current_exception(); }
            });
            try {
                compute();
            } catch (...) {
                launched.store(static_cast<int>(kLayers), std::memory_order_release);
                emitter.join();
                m.after_layer = nullptr;
                throw;
            }
            emitter.join();
            m.after_layer = nullptr;
            if (emit_error) std::rethrow_exception(emit_error);
        } else {
        compute();
        // the block's rows of every row component, D2H into the staging, then to the sink
        if (m.cfg.export_state) {
        if (trace) {
            auto* dest = out;
            for (std::uint32_t o = 0; o < 12; ++o) {
                mark_stream(m.stream, "gpu_ready", static_cast<std::uint64_t>(dest-m.host)+1,
                    "kv.L"+std::to_string(4*o+3), p0, rows, std::uint64_t{rows}*2048);
                dest += std::uint64_t{rows}*2048;
            }
            for (std::uint32_t o = 0; o < 12; ++o) {
                mark_stream(m.stream, "gpu_ready", static_cast<std::uint64_t>(dest-m.host)+1,
                    "index.L"+std::to_string(4*o+3), p0, rows, std::uint64_t{rows}*256);
                dest += std::uint64_t{rows}*256;
            }
        }
        for (std::uint32_t o = 0; o < 12; ++o) {
            const auto layer = std::to_string(4 * o + 3);
            const auto kb = std::uint64_t{rows} * 2048;
            ck(cudaMemcpyAsync(out, S.kv[o] + p0 * 1024, kb, cudaMemcpyDeviceToHost, m.stream), "kv d2h");
            mark_stream(m.stream, "d2h_done", static_cast<std::uint64_t>(out-m.host)+1, {}, 0, 0, 0);
            pend.push_back({"kv.L" + layer, p0, rows, out, kb});
            out += kb;
        }
        for (std::uint32_t o = 0; o < 12; ++o) {
            const auto layer = std::to_string(4 * o + 3);
            const auto ib = std::uint64_t{rows} * 256;
            ck(cudaMemcpyAsync(out, S.raw_index[o] + p0 * 128, ib, cudaMemcpyDeviceToHost, m.stream), "index d2h");
            mark_stream(m.stream, "d2h_done", static_cast<std::uint64_t>(out-m.host)+1, {}, 0, 0, 0);
            pend.push_back({"index.L" + layer, p0, rows, out, ib});
            out += ib;
        }
        }
        }
        if (m.mtp) {
            const std::uint64_t last = std::min<std::uint64_t>(p0 + rows, position - 1);  // pairs i in [p0, last)
            if (last > p0) {
                const auto n = static_cast<std::uint32_t>(last - p0);
                for (std::uint32_t r = 0; r < n; ++r) m.next_host[r] = tokens[p0 + r + 1];
                ck(cudaMemcpyAsync(m.mtp_next, m.next_host.data(), std::uint64_t{n} * 4, cudaMemcpyHostToDevice, m.stream), "mtp next");
                const auto mwts = m.mtp_weights();
                mark_stream(m.stream, "mtp_compute_start", k, {}, p0, n, 0);
                pm::launch_mtp_cache_rows(mwts, pm::MtpRounding{}, m.tap, m.mtp_next, n, p0, S.mtp_kv, S.mtp_index, m.mtp_ws,
                                          m.mtp_status, m.tables, m.stream);
                mark_stream(m.stream, "mtp_compute_end", k, {}, p0, n, 0);
                if (m.cfg.export_state) {
                const auto kb = std::uint64_t{n} * 2048, ib = std::uint64_t{n} * 256;
                mark_stream(m.stream, "gpu_ready", static_cast<std::uint64_t>(out-m.host)+1, "mtp.kv", p0, n, kb);
                mark_stream(m.stream, "gpu_ready", static_cast<std::uint64_t>(out-m.host)+kb+1, "mtp.index", p0, n, ib);
                ck(cudaMemcpyAsync(out, S.mtp_kv + p0 * 1024, kb, cudaMemcpyDeviceToHost, m.stream), "mtp kv d2h");
                mark_stream(m.stream, "d2h_done", static_cast<std::uint64_t>(out-m.host)+1, {}, 0, 0, 0);
                pend.push_back({"mtp.kv", p0, n, out, kb});
                out += kb;
                ck(cudaMemcpyAsync(out, S.mtp_index + p0 * 128, ib, cudaMemcpyDeviceToHost, m.stream), "mtp index d2h");
                mark_stream(m.stream, "d2h_done", static_cast<std::uint64_t>(out-m.host)+1, {}, 0, 0, 0);
                pend.push_back({"mtp.index", p0, n, out, ib});
                out += ib;
                }
            }
            if (p0 + rows == position && m.cfg.export_state) {  // held row h_{p-1}: the tap of the last row
                ck(cudaMemcpyAsync(out, m.tap + std::uint64_t{rows - 1} * kWide, kWide * 2, cudaMemcpyDeviceToHost, m.stream), "held");
                mark_stream(m.stream, "d2h_done", static_cast<std::uint64_t>(out-m.host)+1, {}, 0, 0, 0);
                held = out;
                out += kWide * 2;
            }
        }
        mark_stream(m.stream, "block_end", k, {}, p0, rows, 0);
        ck(cudaStreamSynchronize(m.stream), "block");
        const double bs = std::chrono::duration<double>(Clock::now() - tb).count();
        timing.block_s.push_back(bs);
        if (k == 0) timing.first_block_s = bs;
        for (const auto& p : pend) emit(Piece{false, p.name, p.first, p.count, p.data, p.bytes});
        p0 += rows;
        ++k;
    }
    // fixed blob pieces (offsets inside each component)
    const auto fixed = [&](const char* name, const void* dev, std::uint64_t bytes, bool device = true) {
        if (device) {
            mark_stream(m.stream, "gpu_ready", static_cast<std::uint64_t>(out-m.host)+1, name, 0, 0, bytes);
            ck(cudaMemcpyAsync(out, dev, bytes, cudaMemcpyDeviceToHost, m.stream), "fixed d2h");
            mark_stream(m.stream, "d2h_done", static_cast<std::uint64_t>(out-m.host)+1, name, 0, 0, bytes);
        } else {
            if (trace) trace->mark("gpu_ready", static_cast<std::uint64_t>(out-m.host)+1, name, 0, 0, bytes);
            std::memcpy(out, dev, bytes);
            if (trace) trace->mark("d2h_done", static_cast<std::uint64_t>(out-m.host)+1, name, 0, 0, bytes);
        }
        ck(cudaStreamSynchronize(m.stream), "fixed");
        emit(Piece{true, name, 0, 0, out, bytes});
        out += bytes;
    };
    if (!m.cfg.export_state) {  // decode-only session: nothing leaves the device
        ck(cudaStreamSynchronize(m.stream), "prefill");
        timing.rows = position;
        timing.total_s = std::chrono::duration<double>(Clock::now() - t0).count();
        S.length = position;
        return timing;
    }
    if (!m.cfg.release_by_layer) {  // else sent per layer with the last block (release_by_layer)
        fixed("recurrent_matrix", S.gdn_state, 36 * gdn::kMatrixFloats * 4);
        fixed("gdn_convolution", S.gdn_conv, 36 * gdn::kConvStateHalfs * 2);
        fixed("ple_convolution", S.ple_conv, 10240 * 9 * 2);
    }
    fixed("ngram_history", S.ngram, 8, false);
    // the session's position word holds the last processed position (node exports: 63 after 64 rows, 1030 after 1031)
    const std::uint64_t last_position = position - 1;
    fixed("position_history", &last_position, 8, false);
    if (held) emit(Piece{true, "mtp.held", 0, 0, held, kWide * 2});
    timing.rows = position;
    S.length = position;
    timing.total_s = std::chrono::duration<double>(Clock::now() - t0).count();
    return timing;
}

StepResult Prefill::step(const std::uint32_t* tokens, std::uint32_t rows, std::uint64_t position, bool want_logits) {
    auto r = verify(tokens, rows, position, want_logits);
    commit(rows);
    return r;
}

void Prefill::commit(std::uint32_t keep) {
    if (keep == 0) throw std::invalid_argument("commit keep out of range");  // single-session API: 1 <= keep <= rows
    commit_cohort({keep});
}

// One session of the open cohort window keeps rows [window_position, window_position + keep). Its window rows sit at
// window_offset in the shared window scratch (win_rows, win_raw, ple_keys, tap).
void Prefill::Impl::commit_one(SessionState& s, std::uint32_t keep) {
    s.window_open = false;
    const std::uint64_t off = s.window_offset;
    if (keep < s.window_rows) {
        // GDN: replay the kept rows from S0 (same ascending-row arithmetic) into the other buffer; conv/PLE histories and
        // the n-gram word from their pre-window values plus the kept rows. QSA rows beyond keep are overwritten later.
        for (std::uint32_t o = 0; o < 36; ++o) {
            float* base = win_rows + (std::uint64_t{o} * kMaxWindowRows + off) * gdn_stride;
            const gdn::Rows gr{base, base + 10240, base + 16384, base + 16432, base + 16480, gdn_stride};
            gdn::launch_recurrence_split(gr, keep, s.gdn_state + o * gdn::kMatrixFloats, s.gdn_other + o * gdn::kMatrixFloats,
                gdn_scratch, tables, stream);
            gdn::launch_conv_rollback(win_raw + (std::uint64_t{o} * kMaxWindowRows + off) * gdn::kConvWidth, keep,
                s.conv_before + o * gdn::kConvStateHalfs, s.gdn_conv + o * gdn::kConvStateHalfs, stream);
        }
        pm::launch_ple_conv_rollback(s.ple_before, ple_keys + off * pm::kExpanded, pm::kExpanded, keep, s.ple_conv, stream);
        s.ngram[0] = s.ngram_before[0];
        s.ngram[1] = s.ngram_before[1];
        pm::ple_hash_rows(ple_contract, s.window_tokens.data(), keep, s.window_position, s.ngram, ple_ids_host);
    }
    std::swap(s.gdn_state, s.gdn_other);
    if (s.held) {
        // MTP teacher cache rows of the committed pairs: (h_{L-1}, w_0) at L-1 and (h_{L+j}, w_{j+1}) at L+j, j < keep-1;
        // they replace any draft rows written at those positions. Then h_{L+keep-1} seeds the next chain.
        const std::uint64_t wb = kWide * 2;
        ck(cudaMemcpyAsync(teacher_hidden, s.held, wb, cudaMemcpyDeviceToDevice, stream), "teacher h");
        if (keep > 1)
            ck(cudaMemcpyAsync(teacher_hidden + kWide, tap + off * kWide, (keep - 1) * wb, cudaMemcpyDeviceToDevice, stream), "teacher taps");
        ck(cudaMemcpyAsync(teacher_next, s.window_tokens.data(), std::uint64_t{keep} * 4, cudaMemcpyHostToDevice, stream), "teacher next");
        pm::launch_mtp_cache_rows(mtp_weights(), pm::MtpRounding{}, teacher_hidden, teacher_next, keep, s.window_position - 1,
            s.mtp_kv, s.mtp_index, mtp_ws, mtp_status, tables, stream);
        s.mtp_pooled_valid = std::min<std::uint32_t>(s.mtp_pooled_valid, static_cast<std::uint32_t>((s.window_position - 1) / 4));
        ck(cudaMemcpyAsync(s.held, tap + (off + keep - 1) * kWide, wb, cudaMemcpyDeviceToDevice, stream), "held");
    }
    s.length = s.window_position + keep;
}

// keep == 0: the window never happened for this session (exact undo): GDN S0 is still current (the window wrote the other
// buffer), the conv / PLE histories come back from the window snapshots, the n-gram word from before the window. QSA and
// MTP cache rows at positions >= the window's are rewritten before they are read again; the pooled-key caches are
// clamped as commit() does.
void Prefill::Impl::undo_one(SessionState& s) {
    s.window_open = false;
    ck(cudaMemcpyAsync(s.gdn_conv, s.conv_before, 36 * gdn::kConvStateHalfs * 2, cudaMemcpyDeviceToDevice, stream), "conv undo");
    ck(cudaMemcpyAsync(s.ple_conv, s.ple_before, 10240 * 9 * 2, cudaMemcpyDeviceToDevice, stream), "ple undo");
    s.ngram[0] = s.ngram_before[0];
    s.ngram[1] = s.ngram_before[1];
    if (s.window_position > 0)
        s.mtp_pooled_valid = std::min<std::uint32_t>(s.mtp_pooled_valid, static_cast<std::uint32_t>((s.window_position - 1) / 4));
    s.length = s.window_position;
}

// A session whose state can no longer be trusted (failure mid-window): it needs a new prefill (verify rejects length 0).
void Prefill::Impl::poison(SessionState& s) noexcept {
    s.ngram[0] = s.ngram_before[0];
    s.ngram[1] = s.ngram_before[1];
    s.window_open = false;
    s.length = 0;
}

void Prefill::commit_cohort(const std::vector<std::uint32_t>& keep) {
    auto& m = *impl_;
    if (!m.cohort_open) throw std::logic_error("commit() without an open window");
    if (keep.size() != m.cohort.size()) throw std::invalid_argument("commit_cohort: one keep count per cohort entry");
    for (std::size_t i = 0; i < keep.size(); ++i)
        if (keep[i] > m.cohort[i]->window_rows) throw std::invalid_argument("commit keep out of range");
    // the window closes whatever happens below; a failure part-way poisons the sessions not yet committed
    std::vector<SessionState*> members;
    members.swap(m.cohort);
    m.cohort_open = false;
    std::size_t done = 0;
    try {
        // the teacher-next H2D copies read each session's window_tokens (pageable): stream-ordered, staged at the call
        for (; done < keep.size(); ++done) {
            if (keep[done] == 0) m.undo_one(*members[done]);
            else m.commit_one(*members[done], keep[done]);
        }
        ck(cudaStreamSynchronize(m.stream), "commit");
    } catch (...) {
        cudaStreamSynchronize(m.stream);
        for (std::size_t i = done; i < members.size(); ++i) m.poison(*members[i]);
        if (done == members.size())  // the final sync failed: nothing is known to be committed
            for (auto* s : members) m.poison(*s);
        throw;
    }
}

void Prefill::abort_cohort() noexcept {
    auto& m = *impl_;
    cudaStreamSynchronize(m.stream);
    for (auto* s : m.cohort) m.poison(*s);  // conv/PLE histories may have advanced: the session needs a new prefill
    m.cohort.clear();
    m.cohort_open = false;
    m.window = false;
    m.qsa_cut = nullptr;
}

void Prefill::Impl::window_prologue(const Segment* seg, std::uint32_t count) {
    ck(cudaMemsetAsync(moe_sticky, 0, (1 + kMaxWindowRows) * 4, stream), "moe sticky clear");
    if (table_mode) {
        // the replay's descriptor table (filled on the host before the launch), then the snapshots through it
        ck(cudaMemcpyAsync(seg_table, seg_table_pinned, count * sizeof(SegTable), cudaMemcpyHostToDevice, stream), "descriptor table");
        for (std::uint32_t i = 0; i < count; ++i) {
            copy_slot_kernel<<<64, 256, 0, stream>>>(&seg_table[i].gdn_conv, &seg_table[i].conv_before, 36 * gdn::kConvStateHalfs / 2);
            copy_slot_kernel<<<8, 256, 0, stream>>>(&seg_table[i].ple_conv, &seg_table[i].ple_before, 10240 * 9 / 2);
        }
        ck(cudaGetLastError(), "snapshot copies");
        return;
    }
    for (std::uint32_t i = 0; i < count; ++i) {
        SessionState& s = *seg[i].s;
        ck(cudaMemcpyAsync(s.conv_before, s.gdn_conv, 36 * gdn::kConvStateHalfs * 2, cudaMemcpyDeviceToDevice, stream), "conv snapshot");
        ck(cudaMemcpyAsync(s.ple_before, s.ple_conv, 10240 * 9 * 2, cudaMemcpyDeviceToDevice, stream), "ple snapshot");
    }
}

void Prefill::Impl::step_tail(std::uint32_t rows) {
    nvtx_push("head");
    struct Pop { ~Pop() { nvtx_pop(); } } pop;
    hc::HcMixArgs a;
    a.input = final_wide;
    a.norm = static_cast<const float*>(w.get("output_hc_norm.weight"));
    a.inject = nullptr;
    a.packed = hc_out;
    a.rows = rows;
    a.normalized = normalized;
    a.injection = injection;
    a.output = hidden;
    hc::hc_mix(a, tables, stream);
    const void* head = w.get("output.weight");
    static const bool head1 = [] { const char* v = std::getenv("STRIX_QRTX_HEAD1"); return v && v[0] == '1'; }();
    if (head1)  // reference seam: the one-row kernel per row
        for (std::uint32_t r = 0; r < rows; ++r)
            hc::lm_head(hidden + std::uint64_t{r} * kHidden, head, hc::kVocab, head_logits + std::uint64_t{r} * hc::kVocab, stream);
    else
        for (std::uint32_t r0 = 0; r0 < rows; r0 += 8U)  // the head kernel takes 1..8 rows; same per-row DAG at every count
            hc::lm_head_rows(hidden + std::uint64_t{r0} * kHidden, kHidden, std::min(8U, rows - r0), head, hc::kVocab,
                             head_logits + std::uint64_t{r0} * hc::kVocab, hc::kVocab, stream);
    hc::first_max_rows(head_logits, hc::kVocab, rows, hc::kVocab, argmax_dev, stream);
    ck(cudaMemcpyAsync(argmax_pinned, argmax_dev, std::uint64_t{rows} * 4, cudaMemcpyDeviceToHost, stream), "argmax d2h");
    if (window) ck(cudaMemcpyAsync(moe_sticky_pinned, moe_sticky, (1 + kMaxWindowRows) * 4, cudaMemcpyDeviceToHost, stream), "moe sticky d2h");
}

StepResult Prefill::verify(const std::uint32_t* tokens, std::uint32_t rows, std::uint64_t position, bool want_logits) {
    auto r = verify_cohort({CohortRows{0, tokens, rows, position}}, want_logits);
    if (!r[0].ok) {  // the single-session contract: throw (the window is undone, the session stays usable)
        commit_cohort({0});
        throw std::runtime_error(r[0].error);
    }
    return std::move(r[0]);
}

std::vector<StepResult> Prefill::verify_cohort(const std::vector<CohortRows>& cohort, bool want_logits) {
    auto& m = *impl_;
    if (!m.cfg.decode) throw std::logic_error("step() needs Config::decode");
    if (m.cohort_open) throw std::logic_error("verify() with an open window: commit() first");
    if (cohort.empty()) throw std::invalid_argument("verify_cohort: empty cohort");
    std::uint32_t total = 0;
    std::vector<SessionState*> members;
    for (const auto& c : cohort) {
        auto& s = m.session(c.session);
        if (std::find(members.begin(), members.end(), &s) != members.end()) throw std::invalid_argument("verify_cohort: session twice");
        if (s.window_open) throw std::logic_error("verify() with an open window: commit() first");
        if (s.prefilling) throw std::logic_error("verify(): session " + std::to_string(s.id) + " has a prefill chunk in flight");
        if (s.length == 0) throw std::logic_error("verify(): session " + std::to_string(s.id) + " has no prefilled state");
        if (c.position != s.length)
            throw std::invalid_argument("verify(): position " + std::to_string(c.position) + " != session length " + std::to_string(s.length));
        if (c.rows == 0 || c.rows > kMaxStepRows) throw std::invalid_argument("step rows out of range");
        if (c.position + c.rows > m.cfg.max_positions) throw std::invalid_argument("step beyond max_positions");
        if (!s.gdn_base || (s.gdn_state != s.gdn_base && s.gdn_other != s.gdn_base))
            throw std::logic_error("decode graphs: GDN buffer parity lost");
        members.push_back(&s);
        total += c.rows;
    }
    if (cohort.size() > kMaxStepRows) throw std::invalid_argument("verify_cohort: more than 8 sessions in a window");
    if (total > kMaxWindowRows || total > m.cap) throw std::invalid_argument("step rows out of range");
    // rows of the row-independent families: the window's rows (decode-shaped kernels) up to 8; 9..16 run as a prefill block
    // of 16 padded rows (prefill-shaped kernels, the rows-invariant families of the prefill gate; padding rows are token 0
    // and never read back)
    const std::uint32_t pad = total <= 8U ? total : 16U;
    if (pad > m.cap) throw std::invalid_argument("step rows out of range");
    // The pinned window inputs (ids_pinned, ple_ids_host, seg_table_pinned) are rewritten below and read by the copies of
    // this window only when it executes: every earlier window synchronized the decode stream before returning, so no
    // reader is in flight. Enforced here (one query), so a later change that returns before the sync fails loudly
    // instead of cross-wiring sessions.
    if (cudaStreamQuery(m.stream) != cudaSuccess)
        throw std::logic_error("verify_cohort(): the decode stream is still busy (pinned window inputs would race)");
    std::vector<Segment> segs;
    Impl::GraphKey key;
    const bool table = m.use_graphs && m.graph_table;  // key = rows per entry; sessions and parity via the descriptor table
    const bool multi_key = m.multiseg && cohort.size() > 1;  // single-session windows launch the same in both modes
    if (table) key.push_back(multi_key ? kTableKey - 1U : kTableKey);
    std::uint32_t off = 0;
    for (std::size_t i = 0; i < cohort.size(); ++i) {
        const auto& c = cohort[i];
        auto& s = *members[i];
        std::memcpy(m.ids_pinned + off, c.tokens, std::uint64_t{c.rows} * 4);
        s.ngram_before[0] = s.ngram[0];
        s.ngram_before[1] = s.ngram[1];
        s.window_tokens.assign(c.tokens, c.tokens + c.rows);
        s.window_rows = c.rows;
        s.window_position = c.position;
        s.window_offset = off;
        segs.push_back(Segment{&s, m.ids_pinned + off, c.position, c.rows, off});
        if (table) key.push_back(c.rows);
        else {
            key.push_back(s.id);
            key.push_back(c.rows * 2U + (s.gdn_state == s.gdn_base ? 0U : 1U));
        }
        off += c.rows;
    }
    if (!table) { key.push_back(kModeKey); key.push_back(multi_key ? 1U : 0U); }
    const auto n = static_cast<std::uint32_t>(segs.size());
    // registered before the first launch: any failure below poisons exactly these sessions (n-gram restored, length 0)
    for (auto* s : members) s->window_open = true;
    m.cohort = members;
    m.cohort_open = true;
    std::vector<float> logits;
    m.window = true;
    try {
        auto it = m.use_graphs ? m.graphs.find(key) : m.graphs.end();
        std::uint32_t* seen = nullptr;
        if (m.use_graphs && it == m.graphs.end()) {
            if (m.warmed.size() > 16 * m.graph_cap && !m.warmed.count(key)) m.warmed.clear();
            seen = &m.warmed[key];
        }
        if (!m.use_graphs || (it == m.graphs.end() && *seen == 0)) {
            // eager (also the warm-up before the first capture of this key)
            m.window_prologue(segs.data(), n);
            m.run_block(segs.data(), n, pad);
            m.step_tail(total);
            if (seen) *seen += 1;
            ++m.eager_windows;
        } else {
            if (it == m.graphs.end()) {
                // capture: the same launches as the eager path, cut at every QSA core; host-side PLE hashing during the
                // capture is undone (the replay hashes the window itself)
                while (m.graphs.size() >= m.graph_cap) {
                    auto lru = m.graphs.begin();
                    for (auto g = m.graphs.begin(); g != m.graphs.end(); ++g) if (g->second.used < lru->second.used) lru = g;
                    m.drop_graph(lru);
                }
                Impl::StepGraph g;
                const auto close = [&] {
                    cudaGraph_t graph = nullptr;
                    ck(cudaStreamEndCapture(m.stream, &graph), "end capture");
                    cudaGraphExec_t exec = nullptr;
                    const auto st = cudaGraphInstantiate(&exec, graph, 0);
                    cudaGraphDestroy(graph);
                    ck(st, "graph instantiate");
                    g.seg.push_back(exec);
                };
                ck(cudaStreamBeginCapture(m.stream, cudaStreamCaptureModeRelaxed), "begin capture");
                m.qsa_cut = [&](std::uint32_t) {
                    close();
                    ck(cudaStreamBeginCapture(m.stream, cudaStreamCaptureModeRelaxed), "begin capture");
                };
                try {
                    m.table_mode = table;
                    m.window_prologue(segs.data(), n);
                    m.run_block(segs.data(), n, pad);
                    m.step_tail(total);
                    m.table_mode = false;
                    close();
                } catch (...) {
                    m.table_mode = false;
                    m.qsa_cut = nullptr;
                    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
                    if (cudaStreamIsCapturing(m.stream, &cs) == cudaSuccess && cs != cudaStreamCaptureStatusNone) {
                        cudaGraph_t graph = nullptr;
                        cudaStreamEndCapture(m.stream, &graph);
                        if (graph) cudaGraphDestroy(graph);
                    }
                    for (auto exec : g.seg) cudaGraphExecDestroy(exec);
                    throw;
                }
                m.qsa_cut = nullptr;
                for (auto* s : members) { s->ngram[0] = s->ngram_before[0]; s->ngram[1] = s->ngram_before[1]; }
                m.warmed.erase(key);
                ++m.graph_captures;
                it = m.graphs.emplace(key, std::move(g)).first;
            }
            it->second.used = ++m.graph_clock;
            if (table)  // this window's sessions and buffer parity (the replay copies it to the device first)
                for (std::uint32_t i = 0; i < n; ++i) {
                    const auto& ss = *segs[i].s;
                    m.seg_table_pinned[i] = SegTable{ss.gdn_state, ss.gdn_other, ss.gdn_conv, ss.ple_conv, ss.conv_before, ss.ple_before};
                }
            for (const auto& sg : segs)
                pm::ple_hash_rows(m.ple_contract, sg.tokens, sg.rows, sg.p0, sg.s->ngram,
                                  m.ple_ids_host + std::uint64_t{sg.offset} * pm::kPleHeads);
            const auto& seg = it->second.seg;
            ck(cudaGraphLaunch(seg[0], m.stream), "graph launch");
            for (std::uint32_t o = 0; o < 12; ++o) {
                m.qsa_layer(4 * o + 3, segs.data(), n);
                ck(cudaGraphLaunch(seg[o + 1], m.stream), "graph launch");
            }
            ++m.graph_replays;
        }
        m.window = false;
        if (want_logits) {
            logits.resize(std::uint64_t{total} * hc::kVocab);
            ck(cudaMemcpyAsync(logits.data(), m.head_logits, logits.size() * 4, cudaMemcpyDeviceToHost, m.stream), "logits d2h");
        }
        ck(cudaStreamSynchronize(m.stream), "step");
    } catch (...) {
        m.window = false;
        m.qsa_cut = nullptr;
        cudaStreamSynchronize(m.stream);
        for (auto* s : members) m.poison(*s);
        m.cohort.clear();
        m.cohort_open = false;
        throw;
    }
    // Per-entry outcome. A routed-MoE failure word (any layer) makes the MoE skip every row of that layer: every entry
    // of the window is then suspect. A non-finite row fails only its own entry. Failed entries are committed with keep 0
    // (undo) or dropped by the caller; the others are unaffected.
    const std::uint32_t moe_failure = m.moe_sticky_pinned[0];
    std::vector<StepResult> out(cohort.size());
    for (std::size_t i = 0; i < cohort.size(); ++i) {
        const auto o = segs[i].offset, rows = segs[i].rows;
        auto& r = out[i];
        r.argmax.assign(m.argmax_pinned + o, m.argmax_pinned + o + rows);
        if (want_logits)
            r.logits.assign(logits.begin() + static_cast<std::ptrdiff_t>(std::uint64_t{o} * hc::kVocab),
                            logits.begin() + static_cast<std::ptrdiff_t>(std::uint64_t{o + rows} * hc::kVocab));
        for (std::uint32_t k = 0; k < rows; ++k) r.moe_status |= m.moe_sticky_pinned[1 + o + k];
        if (moe_failure) {
            r.ok = false;
            r.error = "decode step: routed MoE failure in the window (summary bits " + std::to_string(moe_failure) +
                      ", this entry's row bits " + std::to_string(r.moe_status) + ")";
        }
        for (std::uint32_t k = 0; k < rows && r.ok; ++k) {
            if (r.argmax[k] == 0xffffffffU) { r.ok = false; r.error = "decode step: non-finite logits"; }
            else if (want_logits && r.argmax[k] != hc::first_max_token(r.logits.data() + std::uint64_t{k} * hc::kVocab, hc::kVocab)) {
                r.ok = false;
                r.error = "decode step: device argmax differs from the host selection";
            }
        }
    }
    if (moe_failure) {
        std::string rows_bits;
        for (std::uint32_t k = 0; k < total; ++k) rows_bits += (k ? "," : "") + std::to_string(m.moe_sticky_pinned[1 + k]);
        std::fprintf(stderr, "qwen-rtx: routed MoE failure in a %zu-session window (summary %u, row bits [%s])\n", cohort.size(),
                     moe_failure, rows_bits.c_str());
    }
    return out;
}

SessionId Prefill::create_session() {
    auto& m = *impl_;
    if (!m.cfg.decode) throw std::logic_error("create_session() needs Config::decode");
    return m.new_session().id;
}

void Prefill::destroy_session(SessionId session) {
    auto& m = *impl_;
    if (session == 0) throw std::invalid_argument("session 0 belongs to the engine");
    auto& s = m.session(session);
    if (s.window_open || s.prefilling) throw std::logic_error("destroy_session() with an open window or a chunk in flight");
    ck(cudaStreamSynchronize(m.stream), "session drop");
    // graphs and warm-up counts that captured this session's pointers (keys: session, layout pairs)
    const auto uses = [&](const Impl::GraphKey& k) {
        if (!k.empty() && k[0] >= kTableKey - 1U) return false;  // descriptor-table graphs bake no session pointer
        for (std::size_t i = 0; i < k.size(); i += 2) if (k[i] == session) return true;
        return false;
    };
    for (auto it = m.graphs.begin(); it != m.graphs.end();) {
        if (uses(it->first)) { auto next = std::next(it); m.drop_graph(it); it = next; } else ++it;
    }
    for (auto it = m.warmed.begin(); it != m.warmed.end();) it = uses(it->first) ? m.warmed.erase(it) : std::next(it);
    for (void* p : s.allocations) cudaFree(p);
    m.sessions.erase(session);
}

std::uint64_t Prefill::session_bytes() const { return impl_->session(0).bytes; }

std::uint64_t Prefill::length(SessionId session) const { return impl_->session(session).length; }

std::uint64_t Prefill::row_bytes() const { return (12ULL + (impl_->mtp ? 1ULL : 0ULL)) * (2048 + 256); }

std::uint64_t Prefill::fixed_bytes() const {
    return 36ULL * gdn::kMatrixFloats * sizeof(float) + 36ULL * gdn::kConvStateHalfs * 2 + 10240ULL * 9 * 2 + 8;
}

void Prefill::Impl::copy_rows(SessionState& s, std::uint64_t first, std::uint64_t count, void* data, bool importing) {
    auto* cursor = static_cast<std::uint8_t*>(data);
    const auto copy = [&](std::uint16_t* dev, std::uint64_t width) {
        const auto bytes = count * width;
        auto* row = reinterpret_cast<std::uint8_t*>(dev) + first * width;
        ck(cudaMemcpyAsync(importing ? row : cursor, importing ? cursor : row, bytes,
                           cudaMemcpyDefault, state_copy.stream), "prefix rows");
        cursor += bytes;
    };
    for (auto* kv : s.kv) copy(kv, 2048);
    for (auto* index : s.raw_index) copy(index, 256);
    if (mtp) { copy(s.mtp_kv, 2048); copy(s.mtp_index, 256); }
}

void Prefill::Impl::copy_fixed(SessionState& s, void* data, bool importing) {
    auto* cursor = static_cast<std::uint8_t*>(data);
    const auto copy = [&](void* state, std::uint64_t bytes) {
        ck(cudaMemcpyAsync(importing ? state : cursor, importing ? cursor : state, bytes,
                           cudaMemcpyDefault, state_copy.stream), "prefix fixed");
        cursor += bytes;
    };
    // Preserve the current-buffer parity: pointer graphs key it and table graphs read it at replay.
    copy(s.gdn_state, 36ULL * gdn::kMatrixFloats * sizeof(float));
    copy(s.gdn_conv, 36ULL * gdn::kConvStateHalfs * 2);
    copy(s.ple_conv, 10240ULL * 9 * 2);
    copy(s.ngram, sizeof(s.ngram));
}

void Prefill::export_rows(SessionId session, std::uint64_t first, std::uint64_t count, void* dst) {
    export_segments(session, {{first, count, dst}}, nullptr);
}

void Prefill::export_fixed(SessionId session, void* dst) {
    if (!dst) throw std::invalid_argument("prefix export fixed pointer");
    export_segments(session, {}, dst);
}

void Prefill::export_segments(SessionId session, const std::vector<RowSegment>& segments, void* fixed_dst) {
    auto& m = *impl_;
    auto& s = m.session(session);
    for (const auto& seg : segments)
        if (!seg.data || seg.first > s.length || seg.count > s.length - seg.first)
            throw std::invalid_argument("prefix export rows range");
    const auto stream = m.state_copy_stream(s);
    try {
        for (const auto& seg : segments) m.copy_rows(s, seg.first, seg.count, const_cast<void*>(seg.data), false);
        if (fixed_dst) m.copy_fixed(s, fixed_dst, false);
        ck(cudaStreamSynchronize(stream), "prefix export segments");
    }
    catch (...) { cudaStreamSynchronize(stream); throw; }
}

void Prefill::import_state(SessionId session, std::uint64_t length, const std::vector<RowSegment>& segments, const void* fixed) {
    auto& m = *impl_;
    auto& s = m.session(session);
    if (!fixed || length == 0 || length >= m.cfg.max_positions) throw std::invalid_argument("prefix import length/fixed");
    std::uint64_t end = 0;
    for (const auto& seg : segments) {
        if (!seg.data || seg.first != end || seg.count == 0 || seg.count > length - end)
            throw std::invalid_argument("prefix import segments must cover [0,length)");
        end += seg.count;
    }
    if (end != length) throw std::invalid_argument("prefix import incomplete rows");
    const auto stream = m.state_copy_stream(s);
    s.length = 0;  // a failed transfer must never expose a partially restored session
    try {
        for (const auto& seg : segments) m.copy_rows(s, seg.first, seg.count, const_cast<void*>(seg.data), true);
        m.copy_fixed(s, const_cast<void*>(fixed), true);
        ck(cudaStreamSynchronize(stream), "prefix import");
    } catch (...) { cudaStreamSynchronize(stream); throw; }
    for (auto& valid : s.pooled_valid) valid = 0;
    s.mtp_pooled_valid = 0;
    s.length = length;
}

void Prefill::set_graphs(bool on) { impl_->use_graphs = on; }

void Prefill::set_multiseg(bool on) {
    if (impl_->cohort_open) throw std::logic_error("set_multiseg() with an open window");
    impl_->multiseg = on;
}

void Prefill::set_graph_table(bool on) {
    if (impl_->cohort_open) throw std::logic_error("set_graph_table() with an open window");
    impl_->graph_table = on;
}

void Prefill::set_graph_cap(std::size_t cap) {
    auto& m = *impl_;
    if (m.cohort_open) throw std::logic_error("set_graph_cap() with an open window");
    m.graph_cap = std::max<std::size_t>(cap, 1);
    while (m.graphs.size() > m.graph_cap) {
        auto lru = m.graphs.begin();
        for (auto g = m.graphs.begin(); g != m.graphs.end(); ++g) if (g->second.used < lru->second.used) lru = g;
        m.drop_graph(lru);
    }
}

GraphCounters Prefill::graph_counters() const {
    const auto& m = *impl_;
    return GraphCounters{m.graphs.size(), m.graph_cap, m.graph_captures, m.graph_replays, m.graph_evictions, m.eager_windows};
}

std::uint32_t Prefill::max_positions() const { return impl_->cfg.max_positions; }

std::string Prefill::graph_stats() const {
    const auto& m = *impl_;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "{\"graphs\":%s,\"table\":%s,\"cached\":%zu,\"cap\":%zu,\"captures\":%llu,\"replays\":%llu,\"evictions\":%llu,"
                  "\"eager_windows\":%llu}", m.use_graphs ? "true" : "false", m.graph_table ? "true" : "false", m.graphs.size(), m.graph_cap,
                  static_cast<unsigned long long>(m.graph_captures), static_cast<unsigned long long>(m.graph_replays),
                  static_cast<unsigned long long>(m.graph_evictions), static_cast<unsigned long long>(m.eager_windows));
    return buf;
}

// Enqueues one prefill block of session S on the current BlockFields set (its stream and buffers); no host sync.
// Host inputs go through this set's pinned staging (no pageable copy, so the enqueue never waits for the stream).
// Returns the rows of the block.
std::uint32_t Prefill::Impl::prefill_enqueue(SessionState& S, const std::vector<std::uint32_t>& tokens, std::uint64_t position,
                                             std::uint64_t from, std::uint32_t max_rows) {
    if (from == 0) {  // fresh state, as run()
        for (std::uint32_t o = 0; o < 12; ++o) {
            ck(cudaMemsetAsync(S.kv[o], 0, position * 2048, stream), "kv clear");
            ck(cudaMemsetAsync(S.raw_index[o], 0, position * 256, stream), "index clear");
        }
        ck(cudaMemsetAsync(S.gdn_state, 0, 36 * gdn::kMatrixFloats * 4, stream), "gdn clear");
        for (auto& v : S.pooled_valid) v = 0;
        S.mtp_pooled_valid = 0;
        ck(cudaMemsetAsync(S.gdn_conv, 0, 36 * gdn::kConvStateHalfs * 2, stream), "conv clear");
        ck(cudaMemsetAsync(S.ple_conv, 0, 10240 * 9 * 2, stream), "ple conv clear");
        S.ngram[0] = S.ngram[1] = 0;
        S.length = 0;
    }
    const auto rows = static_cast<std::uint32_t>(
        std::min<std::uint64_t>({std::uint64_t{std::max<std::uint32_t>(max_rows, 1U)}, position - from, std::uint64_t{cap}}));
    const std::uint64_t p0 = from;
    std::memcpy(tok_pinned, tokens.data() + p0, std::uint64_t{rows} * 4);
    const Segment seg{&S, tok_pinned, p0, rows, 0};
    run_block(&seg, 1, std::max<std::uint32_t>(16U, round_up(rows, 8U)));
    if (S.held && p0 + rows == position)
        ck(cudaMemcpyAsync(S.held, tap + std::uint64_t{rows - 1} * kWide, kWide * 2, cudaMemcpyDeviceToDevice, stream), "held");
    if (mtp) {
        const std::uint64_t last = std::min<std::uint64_t>(p0 + rows, position - 1);  // pairs i in [p0, last)
        if (last > p0) {
            const auto cnt = static_cast<std::uint32_t>(last - p0);
            for (std::uint32_t r = 0; r < cnt; ++r) next_pinned[r] = tokens[p0 + r + 1];
            ck(cudaMemcpyAsync(mtp_next, next_pinned, std::uint64_t{cnt} * 4, cudaMemcpyHostToDevice, stream), "mtp next");
            pm::launch_mtp_cache_rows(mtp_weights(), pm::MtpRounding{}, tap, mtp_next, cnt, p0, S.mtp_kv, S.mtp_index, mtp_ws,
                                      mtp_status, tables, stream);
        }
    }
    return rows;
}

std::uint64_t Prefill::prefill_chunk(SessionId session, const std::vector<std::uint32_t>& tokens, std::uint64_t position,
                                     std::uint64_t from, std::uint32_t max_rows) {
    auto& m = *impl_;
    auto& S = m.session(session);
    if (m.cohort_open) throw std::logic_error("prefill_chunk() with an open decode window");
    if (S.prefilling || m.pre_session) throw std::logic_error("prefill_chunk() while an overlapped chunk is in flight");
    if (m.window || m.table_mode || m.qsa_cut || m.after_layer) throw std::logic_error("prefill_chunk() inside a decode window or capture");
    if (m.cfg.export_state) throw std::logic_error("prefill_chunk() needs Config::export_state = false");
    if (position > tokens.size() || position == 0 || from >= position) throw std::invalid_argument("prefill chunk out of range");
    if (position > m.cfg.max_positions) throw std::invalid_argument("prompt longer than max_positions");
    if (from != 0 && from != S.length) throw std::invalid_argument("prefill chunk does not continue the session");
    const auto rows = m.prefill_enqueue(S, tokens, position, from, max_rows);
    ck(cudaStreamSynchronize(m.stream), "prefill chunk");
    S.length = from + rows;
    return S.length;
}

// Frees every buffer, staging block and the stream of a BlockFields set (null members are skipped).
void Prefill::Impl::free_block(BlockFields& b) noexcept {
    void* dev[] = {b.ids, b.emb, b.wide_a, b.wide_b, b.hidden, b.step_out, b.injection, b.normalized, b.codes, b.deltas, b.gdn_rows,
                   b.gdn_scratch, b.q_gate, b.qk, b.qv, b.index_q, b.index_k, b.attn_out, b.logits, b.routes, b.route_w, b.shared_out,
                   b.routed_out, b.combined, b.shared_ws, b.moe_sums, b.moe_status, b.moe_acc, b.moe_ddeltas, b.moe_raw, b.moe_codes,
                   b.moe_dcodes, b.moe_deltas, b.moe_dsums, b.moe_inter, b.csr.summary, b.csr.offsets, b.csr.touched, b.csr.rows,
                   b.csr.slots, b.csr.batch_experts, b.csr.batch_begins, b.csr.tile_count, b.csr.tile_experts, b.csr.tile_begins, b.tap,
                   b.ple_ids, b.ple_rows, b.ple_keys, b.ple_values, b.ple_gates, b.ple_status, b.mtp_ws, b.mtp_next, b.mtp_status};
    for (void* d : dev) if (d) cudaFree(d);
    void* host[] = {b.ple_ids_host, b.tok_pinned, b.next_pinned};
    for (void* h : host) if (h) cudaFreeHost(h);
    if (b.qws.pooled_keys || b.qws.block_scores || b.qws.selected || b.qws.scores) qsa::free_workspace(b.qws);
    if (b.stream) cudaStreamDestroy(b.stream);
    b = BlockFields{};
}

void Prefill::enable_prefill_overlap(std::uint32_t max_rows) {
    auto& m = *impl_;
    if (!m.cfg.decode || m.cfg.export_state) throw std::logic_error("prefill overlap needs a decode engine with export_state = false");
    if (m.cohort_open || m.window || m.table_mode || m.qsa_cut || m.after_layer)
        throw std::logic_error("enable_prefill_overlap() inside a decode window");
    if (m.pre) return;
    const auto rows = round_up(std::clamp<std::uint32_t>(max_rows, 16U, m.cap), 16U);
    const auto qsa_chunk = m.qws.chunk_rows;  // the same QSA chunking as the engine set
    cudaEvent_t start = nullptr, end = nullptr;
    ck(cudaEventCreate(&start), "prefill event");
    if (const auto e = cudaEventCreate(&end); e != cudaSuccess) {
        cudaEventDestroy(start);
        ck(e, "prefill event");
    }
    auto set = std::make_unique<BlockFields>();
    std::swap(static_cast<BlockFields&>(m), *set);  // m's BlockFields empty, `set` holds the engine's set
    try {
        m.allocate_block(rows, qsa_chunk, false);
        ck(cudaStreamCreateWithPriority(&m.stream, cudaStreamNonBlocking, m.stream_priority_low), "prefill stream");
    } catch (...) {
        Impl::free_block(static_cast<BlockFields&>(m));  // what was allocated so far
        std::swap(static_cast<BlockFields&>(m), *set);   // the engine set back
        cudaEventDestroy(start);
        cudaEventDestroy(end);
        throw;
    }
    std::swap(static_cast<BlockFields&>(m), *set);  // the engine's set back; `set` now holds the prefill set
    m.pre_start = start;
    m.pre_end = end;
    m.pre = std::move(set);
}

bool Prefill::prefill_overlap() const { return impl_->pre != nullptr; }

std::uint32_t Prefill::prefill_overlap_rows() const { return impl_->pre ? impl_->pre->cap : 0U; }

void Prefill::prefill_chunk_async(SessionId session, const std::vector<std::uint32_t>& tokens, std::uint64_t position,
                                  std::uint64_t from, std::uint32_t max_rows) {
    auto& m = *impl_;
    auto& S = m.session(session);
    if (!m.pre) throw std::logic_error("prefill_chunk_async() needs enable_prefill_overlap()");
    // run_block reads window / table_mode / qsa_cut / after_layer from the engine: an enqueue inside a decode window or a
    // capture would write the window's shared buffers (win_rows, moe_sticky, seg_table) or record into the capture
    if (m.cohort_open || m.window || m.table_mode || m.qsa_cut || m.after_layer)
        throw std::logic_error("prefill_chunk_async() inside a decode window or capture");
    if (m.pre_session) throw std::logic_error("prefill_chunk_async(): a chunk is already in flight");
    if (S.window_open || S.prefilling) throw std::logic_error("prefill_chunk_async() on a session in a window");
    if (m.cfg.export_state) throw std::logic_error("prefill_chunk() needs Config::export_state = false");
    if (position > tokens.size() || position == 0 || from >= position) throw std::invalid_argument("prefill chunk out of range");
    if (position > m.cfg.max_positions) throw std::invalid_argument("prompt longer than max_positions");
    if (from != 0 && from != S.length) throw std::invalid_argument("prefill chunk does not continue the session");
    // enqueue with the prefill set swapped in: every launch reads the prefill stream and buffers; nothing is shared with
    // the decode windows (the session is in no window until prefill_wait()).
    std::swap(static_cast<BlockFields&>(m), *m.pre);
    std::uint32_t rows = 0;
    try {
        ck(cudaEventRecord(m.pre_start, m.stream), "prefill start");
        rows = m.prefill_enqueue(S, tokens, position, from, max_rows);
        ck(cudaEventRecord(m.pre_end, m.stream), "prefill end");
    } catch (...) {
        cudaStreamSynchronize(m.stream);
        std::swap(static_cast<BlockFields&>(m), *m.pre);
        S.length = 0;  // partial chunk: a new prefill is needed
        throw;
    }
    std::swap(static_cast<BlockFields&>(m), *m.pre);
    S.prefilling = true;
    m.pre_session = &S;
    m.pre_length = from + rows;
}

std::optional<PrefillChunk> Prefill::prefill_poll() {
    auto& m = *impl_;
    if (!m.pre_session) throw std::logic_error("prefill_poll() without a chunk in flight");
    const auto e = cudaEventQuery(m.pre_end);
    if (e == cudaErrorNotReady) return std::nullopt;  // still running: the session stays excluded, nothing changes
    if (e != cudaSuccess) (void)cudaGetLastError();   // prefill_wait reports the failure and poisons the session
    return prefill_wait();                            // complete: finalize exactly as the blocking path
}

bool Prefill::prefill_pending() const { return impl_->pre_session != nullptr; }

PrefillChunk Prefill::prefill_wait() {
    auto& m = *impl_;
    if (!m.pre_session) throw std::logic_error("prefill_wait() without a chunk in flight");
    auto& S = *m.pre_session;
    m.pre_session = nullptr;
    S.prefilling = false;
    const auto e = cudaEventSynchronize(m.pre_end);
    if (e != cudaSuccess) {
        S.length = 0;
        ck(e, "prefill chunk");
    }
    float ms = 0.0F;
    ck(cudaEventElapsedTime(&ms, m.pre_start, m.pre_end), "prefill chunk time");
    S.length = m.pre_length;
    return PrefillChunk{S.length, static_cast<double>(ms)};
}

Draft Prefill::draft(std::uint32_t pending, std::uint32_t depth, float min_probability) {
    return draft(0, pending, depth, min_probability);
}

Draft Prefill::draft(SessionId session, std::uint32_t pending, std::uint32_t depth, float min_probability) {
    auto& m = *impl_;
    auto& S = m.session(session);
    if (S.prefilling) throw std::logic_error("draft() on a session with a prefill chunk in flight");
    if (!S.held) throw std::logic_error("draft() needs Config::decode and an MTP sidecar");
    if (m.cohort_open) throw std::logic_error("draft() with an open window");
    if (S.length == 0) throw std::logic_error("draft() before run()");
    depth = std::min(depth, kMaxStepRows - 1U);
    // the window that verifies these drafts covers positions [length, length + depth]
    const std::uint64_t room = m.cfg.max_positions > S.length ? m.cfg.max_positions - S.length - 1U : 0U;
    depth = static_cast<std::uint32_t>(std::min<std::uint64_t>(depth, room));
    Draft out;
    if (depth == 0) return out;
    const auto mw = [&](const char* n) { return m.mw.get(std::string("blk.48.") + n); };
    const auto mwf = [&](const char* n) { return static_cast<const float*>(mw(n)); };
    const auto mwts = m.mtp_weights();
    const auto views = pm::mtp_workspace_views(m.mtp_ws, 1);
    ck(cudaMemcpyAsync(m.draft_token, &pending, 4, cudaMemcpyHostToDevice, m.stream), "draft pending");
    const std::uint16_t* source = S.held;  // h_{L-1}, then the chain's wide state
    std::uint16_t* wide_attn = m.draft_wide;
    std::uint16_t* wide_mlp = m.draft_wide + kWide;
    std::uint16_t* wide_src = m.draft_wide + 2 * kWide;
    std::uint32_t sel[2];
    static const bool per_step_sync = [] { const char* v = std::getenv("STRIX_QRTX_DRAFT_SYNC"); return v && v[0] == '1'; }();
    const auto mix = [&](const std::uint16_t* in, const float* norm, const float* inject, const hc::HcPacked& packed) {
        hc::HcMixArgs a;
        a.input = in; a.norm = norm; a.inject = inject; a.packed = packed; a.rows = 1;
        a.normalized = m.normalized; a.injection = m.injection; a.output = m.hidden;
        hc::hc_mix(a, m.tables, m.stream);
    };
    for (std::uint32_t k = 0; k < depth; ++k) {
        const std::uint64_t pos = S.length - 1 + k;
        // input fusion (enorm/hnorm/eh_proj) -> the MTP wide row; also stores a provisional cache row at pos
        pm::launch_mtp_cache_rows(mwts, pm::MtpRounding{}, source, m.draft_token, 1, pos, S.mtp_kv, S.mtp_index, m.mtp_ws,
            m.mtp_status, m.tables, m.stream);
        mix(views.fused, mwf("hc_attn_norm.weight"), m.mtp_attn_inject, m.mtp_hc_attn);
        dn::Site in;
        in.store = dn::Store::f32_round_bf16;
        in.width = kHidden;
        in.segments = 3;
        in.segment[0] = {mw("attn_q.weight"), qsa::kQGateWidth, m.q_gate, qsa::kQGateWidth};
        in.segment[1] = {mw("attn_k.weight"), qsa::kKvWidth, m.qk, qsa::kKvWidth};
        in.segment[2] = {mw("attn_v.weight"), qsa::kKvWidth, m.qv, qsa::kKvWidth};
        ck(dn::quantize_project(m.stream, in, 1, dn::Input::bf16, m.hidden, kHidden, m.codes, m.deltas, m.cap), "draft qkv");
        const qsa::RowBuffers rb{m.hidden, m.q_gate, m.qk, m.qv, m.index_q, m.index_k, m.attn_out};
        const qsa::LayerWeights lw{static_cast<const std::uint16_t*>(mw("indexer.q_proj.weight")),
                                   static_cast<const std::uint16_t*>(mw("indexer.k_proj.weight")), mwf("attn_q_norm.weight"),
                                   mwf("attn_k_norm.weight"), mwf("indexer.q_norm.weight"), mwf("indexer.k_norm.weight")};
        qsa::layer_core(rb, 1, pos, lw, qsa::LayerState{S.mtp_kv, S.mtp_index, S.mtp_pooled, &S.mtp_pooled_valid}, m.rope, m.tables,
            m.qws, m.stream);
        dn::Site o;
        o.store = dn::Store::bf16;
        o.width = qsa::kAttnWidth;
        o.segments = 1;
        o.segment[0] = {mw("attn_output.weight"), kHidden, m.step_out, kHidden};
        ck(dn::quantize_project(m.stream, o, 1, dn::Input::f32, m.attn_out, qsa::kAttnWidth, m.codes, m.deltas, m.cap), "draft o");
        hc::inject(views.fused, m.step_out, m.injection, 1, wide_attn, m.stream);
        mix(wide_attn, mwf("hc_ffn_norm.weight"), m.mtp_ffn_inject, m.mtp_hc_ffn);
        hc::route_logits(m.hidden, mwf("ffn_gate_inp.weight"), 1, m.logits, m.stream);
        hc::route_select(m.logits, 1, kTopK, true, m.tables, m.routes, m.route_w, m.stream);
        dn::SharedExpert sw;
        sw.gate_q8 = mw("ffn_gate_shexp.weight");
        sw.up_q8 = mw("ffn_up_shexp.weight");
        sw.down_q8 = mw("ffn_down_shexp.weight");
        sw.scalar = mwf("ffn_gate_inp_shexp.weight");
        ck(dn::shared_expert(m.stream, 1, sw, m.hidden, kHidden, m.shared_ws, 1280, m.shared_out, kHidden, m.codes, m.deltas, m.cap,
            m.tables), "draft shared expert");
        md::routed_q8_one_row(m.hidden, m.routes, m.route_w, mw("ffn_gate_exps.weight"), mw("ffn_up_exps.weight"),
            mw("ffn_down_exps.weight"), m.draft_scratch, m.routed_out, m.stream);
        hc::combine(m.shared_out, m.routed_out, 1, m.combined, m.stream);
        hc::inject(wide_attn, m.combined, m.injection, 1, wide_mlp, m.stream);
        mix(wide_mlp, mwf("nextn.hc_head_norm.weight"), nullptr, m.mtp_hc_head);
        hc::lm_head(m.hidden, m.draft_head, m.draft_vocab, m.draft_logits, m.stream);
        md::select(m.draft_logits, m.draft_vocab, m.draft_ids, m.draft_sel + 2 * k, m.draft_token, m.stream);
        ck(cudaMemcpyAsync(wide_src, wide_mlp, kWide * 2, cudaMemcpyDeviceToDevice, m.stream), "draft state");
        source = wide_src;
        if (!per_step_sync) continue;
        ck(cudaMemcpyAsync(sel, m.draft_sel + 2 * k, 8, cudaMemcpyDeviceToHost, m.stream), "draft read");
        ck(cudaStreamSynchronize(m.stream), "draft step");
        float probability;
        std::memcpy(&probability, &sel[1], 4);
        out.tokens.push_back(sel[0]);
        out.probability.push_back(probability);
        if (probability < min_probability) break;  // the node keeps the low draft and stops after it
    }
    if (!per_step_sync) {
        // every step was launched back to back; one read, then the same stop rule (later steps are discarded)
        std::uint32_t all[2 * kMaxStepRows];
        ck(cudaMemcpyAsync(all, m.draft_sel, std::uint64_t{depth} * 8, cudaMemcpyDeviceToHost, m.stream), "draft read");
        ck(cudaStreamSynchronize(m.stream), "draft chain");
        for (std::uint32_t k = 0; k < depth; ++k) {
            float probability;
            std::memcpy(&probability, &all[2 * k + 1], 4);
            out.tokens.push_back(all[2 * k]);
            out.probability.push_back(probability);
            if (probability < min_probability) break;
        }
    }
    return out;
}

std::vector<Draft> Prefill::draft_cohort(const std::vector<DraftRequest>& requests) {
    auto& m = *impl_;
    if (m.cohort_open) throw std::logic_error("draft() with an open window");
    std::vector<Draft> out(requests.size());
    // chains by descending depth: the chains still running at step k are a prefix (rows 0 .. active(k) - 1)
    std::vector<std::uint32_t> order, depth(requests.size(), 0);
    std::vector<SessionState*> ss(requests.size(), nullptr);
    for (std::uint32_t i = 0; i < requests.size(); ++i) {
        auto& S = m.session(requests[i].session);
        for (std::uint32_t j = 0; j < i; ++j) if (ss[j] == &S) throw std::invalid_argument("draft_cohort: session twice");
        ss[i] = &S;
        if (requests[i].depth == 0) continue;  // no chain: nothing to check
        if (!S.held) throw std::logic_error("draft() needs Config::decode and an MTP sidecar");
        if (S.prefilling) throw std::logic_error("draft() on a session with a prefill chunk in flight");
        if (S.length == 0) throw std::logic_error("draft() before run()");
        const std::uint64_t room = m.cfg.max_positions > S.length ? m.cfg.max_positions - S.length - 1U : 0U;
        depth[i] = static_cast<std::uint32_t>(std::min<std::uint64_t>(std::min(requests[i].depth, kMaxStepRows - 1U), room));
        if (depth[i] > 0) order.push_back(i);
    }
    if (order.size() > kMaxStepRows) throw std::invalid_argument("draft_cohort: more than 8 chains");
    // one chain, or more chains than the block's row buffers hold: the per-session path itself
    if (order.size() <= 1 || order.size() > m.cap) {
        for (const auto i : order) out[i] = draft(requests[i].session, requests[i].pending, depth[i], requests[i].min_probability);
        return out;
    }
    std::stable_sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) { return depth[a] > depth[b]; });
    const auto chains = static_cast<std::uint32_t>(order.size());
    const std::uint32_t steps = depth[order[0]];
    std::uint32_t pending[kMaxStepRows];
    for (std::uint32_t j = 0; j < chains; ++j) pending[j] = requests[order[j]].pending;
    ck(cudaMemcpyAsync(m.bd_token, pending, chains * 4, cudaMemcpyHostToDevice, m.stream), "draft pending");
    const auto mw = [&](const char* n) { return m.mw.get(std::string("blk.48.") + n); };
    const auto mwf = [&](const char* n) { return static_cast<const float*>(mw(n)); };
    const auto mwts = m.mtp_weights();
    const auto views = pm::mtp_workspace_views(m.mtp_ws, 1);
    std::uint16_t* wide_attn = m.bd_wide;
    std::uint16_t* wide_mlp = m.bd_wide + std::uint64_t{kMaxStepRows} * kWide;
    std::uint16_t* wide_src = m.bd_wide + 2ULL * kMaxStepRows * kWide;
    const auto mix = [&](const std::uint16_t* in, std::uint32_t rows, const float* norm, const float* inject, const hc::HcPacked& packed) {
        hc::HcMixArgs a;
        a.input = in; a.norm = norm; a.inject = inject; a.packed = packed; a.rows = rows;
        a.normalized = m.normalized; a.injection = m.injection; a.output = m.hidden;
        hc::hc_mix(a, m.tables, m.stream);
    };
    const qsa::LayerWeights lw{static_cast<const std::uint16_t*>(mw("indexer.q_proj.weight")),
                               static_cast<const std::uint16_t*>(mw("indexer.k_proj.weight")), mwf("attn_q_norm.weight"),
                               mwf("attn_k_norm.weight"), mwf("indexer.q_norm.weight"), mwf("indexer.k_norm.weight")};
    for (std::uint32_t k = 0; k < steps; ++k) {
        std::uint32_t a = 0;  // chains still running
        while (a < chains && depth[order[a]] > k) ++a;
        // per chain: input fusion (enorm/hnorm/eh_proj) at its position into its MTP cache, the fused row gathered
        for (std::uint32_t j = 0; j < a; ++j) {
            auto& S = *ss[order[j]];
            const std::uint16_t* source = k == 0 ? S.held : wide_src + std::uint64_t{j} * kWide;
            pm::launch_mtp_cache_rows(mwts, pm::MtpRounding{}, source, m.bd_token + j, 1, S.length - 1 + k, S.mtp_kv, S.mtp_index,
                m.mtp_ws, m.mtp_status, m.tables, m.stream);
            ck(cudaMemcpyAsync(m.bd_fused + std::uint64_t{j} * kWide, views.fused, kWide * 2, cudaMemcpyDeviceToDevice, m.stream), "draft fused");
        }
        mix(m.bd_fused, a, mwf("hc_attn_norm.weight"), m.mtp_attn_inject, m.mtp_hc_attn);
        dn::Site in;
        in.store = dn::Store::f32_round_bf16;
        in.width = kHidden;
        in.segments = 3;
        in.segment[0] = {mw("attn_q.weight"), qsa::kQGateWidth, m.q_gate, qsa::kQGateWidth};
        in.segment[1] = {mw("attn_k.weight"), qsa::kKvWidth, m.qk, qsa::kKvWidth};
        in.segment[2] = {mw("attn_v.weight"), qsa::kKvWidth, m.qv, qsa::kKvWidth};
        ck(dn::quantize_project(m.stream, in, a, dn::Input::bf16, m.hidden, kHidden, m.codes, m.deltas, m.cap), "draft qkv");
        for (std::uint32_t j = 0; j < a; ++j) {  // draft attention over the chain's own MTP cache
            auto& S = *ss[order[j]];
            const std::uint64_t r = j;
            const qsa::RowBuffers rb{m.hidden + r * kHidden, m.q_gate + r * qsa::kQGateWidth, m.qk + r * qsa::kKvWidth,
                                     m.qv + r * qsa::kKvWidth, m.index_q + r * qsa::kIndexQWidth, m.index_k + r * qsa::kIndexDim,
                                     m.attn_out + r * qsa::kAttnWidth};
            qsa::layer_core(rb, 1, S.length - 1 + k, lw, qsa::LayerState{S.mtp_kv, S.mtp_index, S.mtp_pooled, &S.mtp_pooled_valid},
                m.rope, m.tables, m.qws, m.stream);
        }
        dn::Site o;
        o.store = dn::Store::bf16;
        o.width = qsa::kAttnWidth;
        o.segments = 1;
        o.segment[0] = {mw("attn_output.weight"), kHidden, m.step_out, kHidden};
        ck(dn::quantize_project(m.stream, o, a, dn::Input::f32, m.attn_out, qsa::kAttnWidth, m.codes, m.deltas, m.cap), "draft o");
        hc::inject(m.bd_fused, m.step_out, m.injection, a, wide_attn, m.stream);
        mix(wide_attn, a, mwf("hc_ffn_norm.weight"), m.mtp_ffn_inject, m.mtp_hc_ffn);
        hc::route_logits(m.hidden, mwf("ffn_gate_inp.weight"), a, m.logits, m.stream);
        hc::route_select(m.logits, a, kTopK, true, m.tables, m.routes, m.route_w, m.stream);
        dn::SharedExpert sw;
        sw.gate_q8 = mw("ffn_gate_shexp.weight");
        sw.up_q8 = mw("ffn_up_shexp.weight");
        sw.down_q8 = mw("ffn_down_shexp.weight");
        sw.scalar = mwf("ffn_gate_inp_shexp.weight");
        ck(dn::shared_expert(m.stream, a, sw, m.hidden, kHidden, m.shared_ws, 1280, m.shared_out, kHidden, m.codes, m.deltas, m.cap,
            m.tables), "draft shared expert");
        for (std::uint32_t j = 0; j < a; ++j)
            md::routed_q8_one_row(m.hidden + std::uint64_t{j} * kHidden, m.routes + std::uint64_t{j} * kTopK,
                m.route_w + std::uint64_t{j} * kTopK, mw("ffn_gate_exps.weight"), mw("ffn_up_exps.weight"), mw("ffn_down_exps.weight"),
                m.draft_scratch, m.routed_out + std::uint64_t{j} * kHidden, m.stream);
        hc::combine(m.shared_out, m.routed_out, a, m.combined, m.stream);
        hc::inject(wide_attn, m.combined, m.injection, a, wide_mlp, m.stream);
        mix(wide_mlp, a, mwf("nextn.hc_head_norm.weight"), nullptr, m.mtp_hc_head);
        hc::lm_head_rows(m.hidden, kHidden, a, m.draft_head, m.draft_vocab, m.bd_logits, m.draft_vocab, m.stream);
        for (std::uint32_t j = 0; j < a; ++j)
            md::select(m.bd_logits + std::uint64_t{j} * m.draft_vocab, m.draft_vocab, m.draft_ids,
                m.bd_sel + (std::uint64_t{j} * kMaxStepRows + k) * 2, m.bd_token + j, m.stream);
        ck(cudaMemcpyAsync(wide_src, wide_mlp, std::uint64_t{a} * kWide * 2, cudaMemcpyDeviceToDevice, m.stream), "draft state");
    }
    std::vector<std::uint32_t> all(2ULL * kMaxStepRows * chains);
    ck(cudaMemcpyAsync(all.data(), m.bd_sel, all.size() * 4, cudaMemcpyDeviceToHost, m.stream), "draft read");
    ck(cudaStreamSynchronize(m.stream), "draft chains");
    for (std::uint32_t j = 0; j < chains; ++j) {
        const auto i = order[j];
        for (std::uint32_t k = 0; k < depth[i]; ++k) {
            float probability;
            std::memcpy(&probability, &all[(std::uint64_t{j} * kMaxStepRows + k) * 2 + 1], 4);
            out[i].tokens.push_back(all[(std::uint64_t{j} * kMaxStepRows + k) * 2]);
            out[i].probability.push_back(probability);
            if (probability < requests[i].min_probability) break;  // the same stop rule as draft()
        }
    }
    return out;
}

}  // namespace strix::cuda::qwen_prefill
