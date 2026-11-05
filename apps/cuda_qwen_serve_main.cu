// Qwen3.8-Flash-Next serving on an RTX card: the node's OpenAI/Anthropic API front end (strix::serve: chat template,
// GGUF BPE tokenizer, SSE, /tokenize, /health) over the CUDA engine (qwen_prefill::Prefill). Greedy decode; every
// committed token is bit-identical to the Strix Halo node's dense-A8 decode, with or without MTP drafts.
//
//   strix-cuda-qwen-serve --model M.gguf --tables DIR --rope-table rope_cs.bin [--mtp-sidecar S.gguf --draft-vocab V.bin]
//       [--max-positions 32768] [--bind 127.0.0.1] [--port 8090] [--api-key-file F] [--model-id ID]
//       [--mtp-depth 7] [--min-p 0.7] [--max-queue 64] [--http-slots 4] [--prefill-chunk 1024|auto] [--prefill-overlap-rows 1024]
//       [--admit-grace-ms 2] [--cohort-depth fair|max] [--slot-margin-mib 2048] [--block-rows N]
//       [--prefix-cache on|off] [--prefix-cache-block 256] [--prefix-cache-stride 16384]
//       [--prefix-cache-host-mb N] [--prefix-cache-vram-mb 0]
// Context cap: --max-positions P sizes every session (KV/index/pooled + MTP cache, 31,616 B per position + 233 MB).
// Sessions of slots 1.. are allocated when a slot is first needed (one ahead of demand), only while cudaMemGetInfo free
// VRAM >= session bytes + --slot-margin-mib (env STRIX_QRTX_SLOT_MARGIN_MIB, default 2048: the WDDM fallback engaged at ~1.4 GiB free); otherwise that slot is
// refused for the server's life and requests wait for a slot with a session (never a WDDM system-memory spill). Every
// engine allocation also refuses past the card (strix/cuda_vram_guard.hpp). --block-rows N: one engine block size of N
// rows instead of 1024/2048/4096/8192 (whole-prompt prefill when nothing decodes; less activation memory).
// Prefill while others decode (decode first): chunks of --prefill-chunk rows (default 1024) between the decode steps.
// STRIX_QRTX_PREFILL_OVERLAP=1 runs the chunks on a second stream beside the steps (opt-in); --prefill-chunk auto sizes
// them from a fitted chunk time model (STRIX_QRTX_PREFILL_AUTO_K: target chunk time in decode steps, default 1).
// --http-slots N (default 4; env STRIX_QRTX_HTTP_SLOTS): up to N requests decode together. Each scheduler step gathers
// every decoding request into cohort verify windows (<= 8 rows: one row per request first, the spare rows to the MTP
// draft chains, cuda_qwen_cohort_plan.hpp); a new request is prefilled in chunks of --prefill-chunk rows between the
// steps (whole prompt blocks when nothing decodes). Every request's tokens equal it running alone.
// --http-slots 1 or STRIX_QRTX_COHORT=0: the old serialized path (one request at a time, generate() to completion).
// Token-block prefix checkpoints restore exact prompt state before the remaining prefill rows.
#include "cuda_qwen_prefix_cache.hpp"
#include "strix/cuda_qwen_cohort.hpp"
#include "strix/cuda_qwen_prefill.hpp"
#include "strix/serve/api.hpp"
#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/engine.hpp"
#include "strix/serve/http_server.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "strix/preflight.hpp"

// The node's preflight (src/preflight.cpp) is POSIX-only; it is reached only through ApiConfig::health_record, which this
// server does not set.
std::string strix::render_preflight(const PreflightObservation&, const PreflightAssessment&) { return "{}"; }

namespace qp = strix::cuda::qwen_prefill;
namespace qc = strix::cuda::qwen_cohort;
namespace px = strix::cuda::qwen_prefix;
namespace serve = strix::serve;
using Clock = std::chrono::steady_clock;

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }

struct Options {
    qp::Config engine;
    px::Settings prefix;
    std::string bind = "127.0.0.1", api_key_file, model_id = "qwen3.8-flash-next";
    int port = 8090;
    std::uint32_t mtp_depth = 7, max_queue = 64;
    float min_p = 0.7F;
    std::uint32_t http_slots = 4, admit_grace_ms = 2;
    std::uint32_t prefill_chunk = 1024;         // rows per prefill chunk while others decode; 0 = auto (fitted, see chunk_done)
    std::uint32_t prefill_overlap_rows = 1024;  // largest chunk on the prefill stream (overlap); its scratch is sized to it
    bool prefill_overlap = false;               // STRIX_QRTX_PREFILL_OVERLAP=1: chunks on a second stream beside the steps
    bool prefill_poll = false;                  // STRIX_QRTX_PREFILL_POLL=1 (overlap): decode steps keep running until the chunk
                                                // completes (cudaEventQuery), then it joins at the next step boundary
    double prefill_auto_k = 1.0;                // auto: target chunk time = k decode steps
    std::string cohort_depth = "fair";
    std::uint32_t slot_margin_mib = 2048;       // free VRAM that must remain after a new slot's session
};

Options parse(int argc, char** argv) {
    Options o;
    if (const char* v = std::getenv("STRIX_QRTX_PREFIX_CACHE")) o.prefix.enabled = v[0] != '0';
    o.engine.decode = true;
    o.engine.export_state = false;
    o.engine.release_by_layer = false;
    if (const char* v = std::getenv("STRIX_QRTX_HTTP_SLOTS")) o.http_slots = static_cast<std::uint32_t>(std::stoul(v));
    if (const char* v = std::getenv("STRIX_QRTX_SLOT_MARGIN_MIB")) o.slot_margin_mib = static_cast<std::uint32_t>(std::stoul(v));
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return argv[++i];
        };
        const auto mib = [&]() -> std::uint64_t {
            const auto value = next();
            std::size_t end = 0;
            if (value.empty() || value[0] == '-') throw std::invalid_argument(a + " requires nonnegative MiB");
            const auto n = std::stoull(value, &end);
            if (end != value.size() || n > (std::numeric_limits<std::uint64_t>::max() >> 21U))
                throw std::invalid_argument(a + " budget out of range");
            return n << 20U;
        };
        const auto cache_count = [&]() -> std::uint32_t {
            const auto value = next();
            std::size_t end = 0;
            if (value.empty() || value[0] == '-') throw std::invalid_argument(a + " requires nonnegative tokens");
            const auto count = std::stoull(value, &end);
            if (end != value.size() || count > UINT32_MAX) throw std::invalid_argument(a + " token count out of range");
            return static_cast<std::uint32_t>(count);
        };
        if (a == "--prefix-cache") {
            const auto value = next();
            if (value != "on" && value != "off") throw std::invalid_argument("--prefix-cache is on or off");
            o.prefix.enabled = value == "on";
        }
        else if (a == "--prefix-cache-block") o.prefix.block = cache_count();
        else if (a == "--prefix-cache-stride") o.prefix.stride = cache_count();
        else if (a == "--prefix-cache-host-mb") o.prefix.host_bytes = mib();
        else if (a == "--prefix-cache-vram-mb") o.prefix.vram_bytes = mib();
        else if (a == "--model") o.engine.model_gguf = next();
        else if (a == "--mtp-sidecar") o.engine.mtp_sidecar_gguf = next();
        else if (a == "--draft-vocab") o.engine.draft_vocab = next();
        else if (a == "--tables") o.engine.tables_dir = next();
        else if (a == "--rope-table") o.engine.rope_table = next();
        else if (a == "--max-positions") o.engine.max_positions = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--bind") o.bind = next();
        else if (a == "--port") o.port = std::stoi(next());
        else if (a == "--api-key-file") o.api_key_file = next();
        else if (a == "--model-id") o.model_id = next();
        else if (a == "--mtp-depth") o.mtp_depth = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--min-p") o.min_p = std::stof(next());
        else if (a == "--max-queue") o.max_queue = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--http-slots") o.http_slots = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--prefill-chunk") { const auto v = next(); o.prefill_chunk = v == "auto" ? 0U : static_cast<std::uint32_t>(std::stoul(v)); }
        else if (a == "--prefill-overlap-rows") o.prefill_overlap_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--admit-grace-ms") o.admit_grace_ms = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--cohort-depth") o.cohort_depth = next();
        else if (a == "--slot-margin-mib") o.slot_margin_mib = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--block-rows") {
            const auto r = static_cast<std::uint32_t>(std::stoul(next()));
            if (r < 1024 || r % 16) throw std::invalid_argument("--block-rows must be a multiple of 16 and >= 1024");
            o.engine.block_schedule = {r};
        }
        else throw std::invalid_argument("unknown option " + a);
    }
    if (o.engine.model_gguf.empty() || o.engine.tables_dir.empty() || o.engine.rope_table.empty())
        throw std::invalid_argument("--model, --tables and --rope-table are required");
    if (const char* v = std::getenv("STRIX_QRTX_COHORT"); v && v[0] == '0') o.http_slots = 1;
    if (o.http_slots == 0) throw std::invalid_argument("--http-slots must be >= 1");
    if (const char* v = std::getenv("STRIX_QRTX_PREFILL_OVERLAP")) o.prefill_overlap = v[0] == '1';
    if (const char* v = std::getenv("STRIX_QRTX_PREFILL_POLL")) o.prefill_poll = v[0] == '1';
    if (const char* v = std::getenv("STRIX_QRTX_PREFILL_AUTO_K")) o.prefill_auto_k = std::stod(v);
    if (!(o.prefill_auto_k > 0)) throw std::invalid_argument("STRIX_QRTX_PREFILL_AUTO_K must be > 0");
    if (o.prefill_overlap_rows < 16) throw std::invalid_argument("--prefill-overlap-rows must be >= 16");
    if (o.cohort_depth != "fair" && o.cohort_depth != "max") throw std::invalid_argument("--cohort-depth is fair or max");
    px::validate_block(o.prefix.block);
    if (o.prefix.stride && o.prefix.stride % o.prefix.block)
        throw std::invalid_argument("--prefix-cache-stride must be 0 or a multiple of --prefix-cache-block");
    return o;
}

std::string read_key(const std::string& path) {
    if (path.empty()) return {};
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read --api-key-file");
    std::string key;
    std::getline(f, key);
    while (!key.empty() && (key.back() == '\r' || key.back() == ' ')) key.pop_back();
    return key;
}

class CudaQwenEngine final : public serve::GenerationEngine {
public:
    CudaQwenEngine(qp::Prefill& engine, const Options& o, px::Cache& cache) : e_(engine), o_(o), cache_(cache) {}

    serve::GenerationStats generate(serve::GenerationJob& job) override {
        serve::GenerationStats st;
        st.prompt_tokens = job.prompt.size();
        const auto t0 = Clock::now();
        const std::uint64_t n = job.prompt.size();
        const std::uint64_t cap = o_.engine.max_positions;
        if (n < 2) throw serve::GenerationRequestError(st, "prompt needs at least 2 tokens");
        if (n >= cap) { st.error_status = 400; st.error_code = "context_length_exceeded"; throw serve::GenerationRequestError(st, "prompt longer than the context"); }
        if (job.on_progress) job.on_progress({serve::GenerationPhase::prefill, 0, 0, n});
        double prefix_restore_ms = 0, prefix_capture_ms = 0;
        if (!cache_.enabled()) e_.run(job.prompt, n - 1, [](const qp::Piece&) {});
        else {
            auto prefilled = cache_.restore(e_, 0, job.prompt, prefix_restore_ms);
            st.cached_tokens = prefilled;
            const auto points = cache_.plan(n, prefilled);
            while (prefilled < n - 1) {
                const auto rows = px::clip_chunk(prefilled, UINT32_MAX, points);
                prefilled = e_.prefill_chunk(0, job.prompt, n - 1, prefilled, rows);
                if (std::binary_search(points.begin(), points.end(), prefilled)) prefix_capture_ms += cache_.capture(e_, 0, job.prompt, prefilled);
            }
        }
        const auto t1 = Clock::now();
        st.prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (job.on_progress) job.on_progress({serve::GenerationPhase::decode, n, 0, n});
        const bool mtp = !o_.engine.mtp_sidecar_gguf.empty() && o_.mtp_depth > 0;
        std::uint32_t depth = o_.mtp_depth;
        if (job.draft_depth_max) depth = std::min(depth, job.draft_depth_max);
        std::uint32_t pending = job.prompt[n - 1];
        std::uint64_t pos = n - 1;
        st.finish = "length";
        bool first = true;
        std::vector<std::uint32_t> emitted;
        while (st.generated_tokens < job.max_tokens) {
            if (pos + 1 >= cap) { st.finish = "context"; break; }
            std::vector<std::uint32_t> rows{pending};
            if (mtp) {
                const auto d = e_.draft(pending, depth, o_.min_p);
                rows.insert(rows.end(), d.tokens.begin(), d.tokens.end());
                st.proposed += d.tokens.size();
            }
            const auto room = static_cast<std::uint32_t>(std::min<std::uint64_t>(rows.size(), cap - pos));
            rows.resize(room);
            const auto r = e_.verify(rows.data(), static_cast<std::uint32_t>(rows.size()), pos);
            std::uint32_t keep = 1;
            while (keep < rows.size() && rows[keep] == r.argmax[keep - 1]) ++keep;
            e_.commit(keep);
            ++st.cycles;
            st.accepted += keep - 1;
            pos += keep;
            pending = r.argmax[keep - 1];
            // emit the verified tokens up to max_tokens and the first stop token (included)
            std::vector<std::uint32_t> out;
            bool stopped = false;
            for (std::uint32_t j = 0; j < keep && st.generated_tokens + out.size() < job.max_tokens; ++j) {
                out.push_back(r.argmax[j]);
                if (std::find(job.stop_tokens.begin(), job.stop_tokens.end(), r.argmax[j]) != job.stop_tokens.end()) { stopped = true; break; }
            }
            if (first && job.observed) st.first_token_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            first = false;
            st.generated_tokens += out.size();
            emitted.insert(emitted.end(), out.begin(), out.end());
            if (job.on_tokens && !job.on_tokens(out)) { st.finish = "cancelled"; break; }
            if (stopped) { st.finish = "stop"; break; }
        }
        st.decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - t1).count();
        char head[200];
        std::snprintf(head, sizeof(head), "{\"engine\":\"cuda-qwen-a8\",\"mtp_depth\":%u,\"tokens_per_cycle\":%.3f,\"prefill_ms\":%.2f,\"decode_ms\":%.2f,\"ids\":[",
                      mtp ? depth : 0U, st.cycles ? static_cast<double>(st.generated_tokens) / static_cast<double>(st.cycles) : 0.0,
                      st.prefill_ms, st.decode_ms);
        std::string extra = head;
        for (std::size_t i = 0; i < emitted.size() && i < 4096; ++i) extra += (i ? "," : "") + std::to_string(emitted[i]);
        extra += "],\"prefix_cached\":" + std::to_string(st.cached_tokens) + ",\"prefix_restore_ms\":" + std::to_string(prefix_restore_ms)
            + ",\"prefix_capture_ms\":" + std::to_string(prefix_capture_ms) + "}";
        st.engine_extra.emplace_back("rtx", extra);
        return st;
    }
    [[nodiscard]] std::uint64_t context_tokens() const override { return o_.engine.max_positions; }

private:
    qp::Prefill& e_;
    const Options& o_;
    px::Cache& cache_;
};

// Continuous batching over up to http_slots requests (one engine session each): between scheduler steps it admits
// queued requests into free slots; each step runs one cohort decode step over every decoding request (qc::step: drafts,
// packed windows, greedy acceptance), then one prefill chunk of the oldest request still prefilling (decode first).
// The tokens a request emits, its stop rule and its receipts follow CudaQwenEngine::generate() exactly.
class CohortServer {
public:
    CohortServer(qp::Prefill& engine, const Options& o, serve::GenerationQueue& queue, px::Cache& cache) : e_(engine), o_(o), queue_(queue), cache_(cache) {
        mtp_ = !o_.engine.mtp_sidecar_gguf.empty() && o_.mtp_depth > 0;
        if (o_.prefill_overlap) e_.enable_prefill_overlap(o_.prefill_overlap_rows);
        // slot 0 is the engine's session 0; the others get theirs when first needed (ensure_session)
        for (std::uint32_t i = 0; i < o.http_slots; ++i) {
            slots_.push_back(Slot{});
            slots_.back().has_session = i == 0;
        }
        const auto vr = vram();
        std::cerr << "strix-cuda-qwen-serve: VRAM after load " << (vr.first >> 20U) << " MiB free of " << (vr.second >> 20U)
                  << " MiB; session " << (e_.session_bytes() >> 20U) << " MiB, slot margin " << o_.slot_margin_mib
                  << " MiB -> " << std::min<std::uint64_t>(o_.http_slots - 1U, slot_fit(vr.first)) + 1U << " of " << o_.http_slots
                  << " slots fit now" << std::endl;
        max_chunk_ = o_.prefill_overlap ? e_.prefill_overlap_rows() : 1024U;
        if (o_.prefill_overlap && o_.prefill_chunk > max_chunk_)
            std::cerr << "strix-cuda-qwen-serve: --prefill-chunk " << o_.prefill_chunk << " capped at " << max_chunk_
                      << " rows (--prefill-overlap-rows) for overlapped chunks" << std::endl;
    }

    void run() {
        for (;;) {
            admit();
            if (queue_.stopping()) {
                if (inflight_) {  // the chunk in flight completes before its request is finished
                    try { (void)e_.prefill_wait(); } catch (...) {}
                    inflight_ = nullptr;
                }
                for (auto& s : slots_) if (s.job) finish(s, "server shutting down", "cancelled");
                return;
            }
            bool any = false;
            for (auto& s : slots_) any = any || s.job.has_value();
            if (!any) continue;  // take() woke for a control request
            // decode first. Overlap: the oldest prefilling request's chunk runs on the prefill stream (no shared buffer; the
            // request is in no window) while decode steps run. Wait mode: one decode step, then wait for the chunk. Poll
            // mode: decode steps keep running and the chunk is polled after each; it joins at the next step boundary once
            // complete; the loop blocks on it only when no session can decode.
            if (!inflight_ && e_.prefill_overlap() && decoding()) {
                Slot* slot = oldest_prefill();
                if (slot && slot->job->ticket->cancelled()) { finish(*slot, {}, "cancelled"); slot = nullptr; }
                if (slot) {
                    try {
                        const auto rows = px::clip_chunk(slot->prefilled, next_chunk_rows(true), slot->capture_points);
                        e_.prefill_chunk_async(slot->session, slot->job->job.prompt, slot->n - 1, slot->prefilled, rows);
                        inflight_ = slot;
                        inflight_steps_ = 0;
                    } catch (const std::exception& ex) {
                        finish(*slot, ex.what(), "length");
                    }
                }
            }
            const bool stepped = decoding();
            if (stepped) {
                decode_step();
                if (inflight_) ++inflight_steps_;
            }
            if (inflight_) {
                std::optional<qp::PrefillChunk> r;
                const bool block = !o_.prefill_poll || !decoding();
                const auto tw = Clock::now();
                try {
                    r = block ? std::optional<qp::PrefillChunk>(e_.prefill_wait()) : e_.prefill_poll();
                } catch (const std::exception& ex) {
                    Slot* slot = inflight_;
                    inflight_ = nullptr;
                    finish(*slot, ex.what(), "length");
                    continue;
                }
                if (block) inflight_->prefill_wait_ms += std::chrono::duration<double, std::milli>(Clock::now() - tw).count();
                if (r) {
                    Slot* slot = inflight_;
                    inflight_ = nullptr;
                    slot->overlap_steps += inflight_steps_;
                    chunk_done(*slot, r->length, r->gpu_ms, true);
                }
            } else if (!stepped || !e_.prefill_overlap()) {
                prefill_step();
            }
        }
    }

private:
    using Clock = std::chrono::steady_clock;
    enum class Phase { prefill, decode };
    struct Slot {
        qp::SessionId session = 0;
        bool has_session = false, refused = false;  // refused: its session did not fit (never retried)
        double admit_free_mib = 0;                  // cudaMemGetInfo free VRAM when the request was admitted
        std::optional<serve::TakenJob> job;
        Phase phase = Phase::prefill;
        std::uint64_t n = 0, prefilled = 0;
        std::vector<std::uint64_t> capture_points;
        double prefix_restore_ms = 0, prefix_capture_ms = 0;
        qc::Stream stream;
        serve::GenerationStats st;
        Clock::time_point t0, t_decode;
        double prefill_gpu_ms = 0;   // overlapped chunks: GPU time on the prefill stream (events)
        double prefill_sync_ms = 0;  // chunks between steps / whole blocks: host wall time
        std::uint32_t prefill_chunks = 0, overlapped_chunks = 0, overlap_steps = 0;
        double prefill_wait_ms = 0;  // overlap: host time blocked waiting for a chunk (wait mode, or nothing could decode)
        std::vector<std::uint32_t> emitted;
        std::uint64_t rows = 0, window_sessions = 0, max_window_sessions = 0, window_rows = 0;
        bool first = true;
    };

    static std::pair<std::uint64_t, std::uint64_t> vram() {
        std::size_t free = 0, total = 0;
        if (cudaMemGetInfo(&free, &total) != cudaSuccess) { (void)cudaGetLastError(); return {0, 0}; }
        return {free, total};
    }
    std::uint64_t slot_fit(std::uint64_t free) const {
        const std::uint64_t need = e_.session_bytes() + (std::uint64_t{o_.slot_margin_mib} << 20U);
        return free / need;
    }
    unsigned live_slots() const {
        unsigned n = 0;
        for (const auto& x : slots_) n += x.has_session ? 1U : 0U;
        return n;
    }
    // A slot's session, allocated when the slot is first needed and only while it fits with the margin; else the slot is
    // refused (it only refuses: requests wait for another slot, nothing an admitted request computes changes).
    bool ensure_session(Slot& s, std::size_t index) {
        if (s.has_session) return true;
        if (s.refused) return false;
        const auto before = vram();
        const std::uint64_t need = e_.session_bytes() + (std::uint64_t{o_.slot_margin_mib} << 20U);
        std::string why;
        if (before.first < need)
            why = "free " + std::to_string(before.first >> 20U) + " MiB < session " + std::to_string(e_.session_bytes() >> 20U) +
                  " + margin " + std::to_string(o_.slot_margin_mib) + " MiB";
        else {
            try {
                s.session = e_.create_session();
                s.has_session = true;
            } catch (const std::exception& ex) {
                why = ex.what();
            }
        }
        if (!s.has_session) {
            s.refused = true;
            std::cerr << "strix-cuda-qwen-serve: slot " << index << " refused (" << why << "); serving " << live_slots()
                      << " slots" << std::endl;
            return false;
        }
        const auto after = vram();
        std::cerr << "strix-cuda-qwen-serve: slot " << index << " session allocated (" << live_slots() << " live), VRAM free "
                  << (before.first >> 20U) << " -> " << (after.first >> 20U) << " MiB of " << (after.second >> 20U) << std::endl;
        return true;
    }

    void admit() {
        bool woke = false;
        for (std::size_t si = 0; si < slots_.size(); ++si) {
            auto& s = slots_[si];
            if (s.job) continue;
            if (!ensure_session(s, si)) continue;
            bool busy = false;
            for (const auto& x : slots_) busy = busy || x.job.has_value();
            auto taken = queue_.take(!busy);  // block only when nothing runs
            if (!taken && woke && o_.admit_grace_ms > 0) {
                // requests that arrive together (a client burst) join the same first step instead of waiting behind
                // the first one's whole-prompt prefill
                std::this_thread::sleep_for(std::chrono::milliseconds(o_.admit_grace_ms));
                woke = false;
                taken = queue_.take(false);
            }
            if (!taken) return;
            woke = woke || !busy;
            start(s, std::move(*taken));
        }
    }

    void start(Slot& s, serve::TakenJob taken) {
        s = Slot{s.session, s.has_session, s.refused};
        s.admit_free_mib = static_cast<double>(vram().first) / 1048576.0;
        s.job = std::move(taken);
        auto& job = s.job->job;
        s.t0 = Clock::now();
        s.st.prompt_tokens = job.prompt.size();
        s.n = job.prompt.size();
        const std::uint64_t cap = o_.engine.max_positions;
        std::string error;
        if (s.n < 2) error = "prompt needs at least 2 tokens";
        else if (s.n >= cap) {
            s.st.error_status = 400;
            s.st.error_code = "context_length_exceeded";
            error = "prompt longer than the context";
        }
        if (!error.empty()) {
            queue_.finish(*s.job, s.st, error);
            s.job.reset();
            return;
        }
        s.prefilled = cache_.restore(e_, s.session, job.prompt, s.prefix_restore_ms);
        s.st.cached_tokens = s.prefilled;
        s.capture_points = cache_.plan(s.n, s.prefilled);
        if (job.on_progress) job.on_progress({serve::GenerationPhase::prefill, s.prefilled, 0, s.n});
    }

    bool decoding() const {
        for (const auto& s : slots_) if (s.job && s.phase == Phase::decode) return true;
        return false;
    }
    Slot* oldest_prefill() {
        Slot* pick = nullptr;
        for (auto& s : slots_)
            if (s.job && s.phase == Phase::prefill && (!pick || s.job->started < pick->job->started)) pick = &s;
        return pick;
    }
    // Rows of the next chunk while others decode: --prefill-chunk N, or auto. Auto fits chunk_ms = a + b * rows (the fixed
    // per-chunk cost a matters: a per-row rate measured on small chunks would shrink the chunks without end) by least squares
    // with exponential forgetting over the chunks that ran beside decode steps, and solves a + b * rows = k * step_ms.
    // Sizes: first 256, second 512 (two sizes for the fit), then the solution, a multiple of 16 in
    // [max(256, the first size), max_chunk_].
    std::uint32_t next_chunk_rows(bool overlap) const {
        const std::uint32_t cap = overlap ? max_chunk_ : std::numeric_limits<std::uint32_t>::max();
        return std::min(cap, o_.prefill_chunk ? o_.prefill_chunk : auto_rows_);
    }
    void fit_chunk_time(double rows, double ms) {
        constexpr double keep = 0.8;  // forgetting factor per observation
        fit_[0] = keep * fit_[0] + 1.0;
        fit_[1] = keep * fit_[1] + rows;
        fit_[2] = keep * fit_[2] + rows * rows;
        fit_[3] = keep * fit_[3] + ms;
        fit_[4] = keep * fit_[4] + rows * ms;
        ++fit_n_;
        const std::uint32_t floor_rows = 256U, top = std::max(floor_rows, max_chunk_);
        if (fit_n_ == 1) { auto_rows_ = std::min(512U, top); return; }  // a second size for the fit
        const double det = fit_[0] * fit_[2] - fit_[1] * fit_[1];
        if (!(det > 1e-6 * fit_[0] * fit_[2]) || step_ms_ <= 0) return;
        const double b = (fit_[0] * fit_[4] - fit_[1] * fit_[3]) / det;
        const double a = (fit_[3] - b * fit_[1]) / fit_[0];
        fit_a_ = a;
        fit_b_ = b;
        if (!(b > 0)) return;
        const double want = (o_.prefill_auto_k * step_ms_ - a) / b;
        const double lo = std::max<double>(floor_rows, first_rows_), hi = std::max<double>(lo, top);
        auto_rows_ = static_cast<std::uint32_t>(std::clamp(want, lo, hi)) / 16U * 16U;
    }
    void chunk_done(Slot& s, std::uint64_t length, double ms, bool overlapped) {
        const auto rows = static_cast<double>(length - s.prefilled);
        if (o_.prefill_chunk == 0 && rows > 0 && ms > 0 && decoding()) {  // auto learns from chunks beside decode steps
            if (first_rows_ == 0) first_rows_ = static_cast<std::uint32_t>(rows);
            fit_chunk_time(rows, ms);
        }
        s.prefilled = length;
        if (std::binary_search(s.capture_points.begin(), s.capture_points.end(), length))
            s.prefix_capture_ms += cache_.capture(e_, s.session, s.job->job.prompt, length);
        (overlapped ? s.prefill_gpu_ms : s.prefill_sync_ms) += ms;
        ++s.prefill_chunks;
        s.overlapped_chunks += overlapped ? 1U : 0U;
        auto& job = s.job->job;
        if (s.prefilled < s.n - 1) {
            if (job.on_progress) job.on_progress({serve::GenerationPhase::prefill, s.prefilled, 0, s.n});
            return;
        }
        s.t_decode = Clock::now();
        s.st.prefill_ms = std::chrono::duration<double, std::milli>(s.t_decode - s.t0).count();
        if (job.on_progress) job.on_progress({serve::GenerationPhase::decode, s.n, 0, s.n});
        s.phase = Phase::decode;
        s.stream = qc::Stream{s.session, job.prompt[s.n - 1], s.n - 1, job.draft_depth_max};
        s.st.finish = "length";
        if (job.max_tokens == 0) finish(s, {}, "length");
    }

    // One chunk of the oldest prefilling request after the decode step: next_chunk_rows() while others decode, else
    // whole blocks.
    void prefill_step() {
        Slot* pick = oldest_prefill();
        if (!pick) return;
        Slot& s = *pick;
        if (s.job->ticket->cancelled()) { finish(s, {}, "cancelled"); return; }
        const std::uint32_t chunk = decoding() ? next_chunk_rows(false) : std::numeric_limits<std::uint32_t>::max();
        const auto tc = Clock::now();
        std::uint64_t length = 0;
        try {
            length = e_.prefill_chunk(s.session, s.job->job.prompt, s.n - 1, s.prefilled, px::clip_chunk(s.prefilled, chunk, s.capture_points));
        } catch (const std::exception& ex) {
            finish(s, ex.what(), "length");
            return;
        }
        chunk_done(s, length, std::chrono::duration<double, std::milli>(Clock::now() - tc).count(), false);
    }

    void decode_step() {
        std::vector<Slot*> live;
        std::vector<qc::Stream> streams;
        for (auto& s : slots_) {
            if (!s.job || s.phase != Phase::decode) continue;
            live.push_back(&s);
            streams.push_back(s.stream);
        }
        if (live.empty()) return;
        qc::StepOptions opt;
        opt.mtp = mtp_;
        opt.depth = std::min<std::uint32_t>(o_.mtp_depth, qc::kWindowRows - 1U);
        opt.min_p = o_.min_p;
        opt.policy = qc::depth_policy_of(o_.cohort_depth);
        opt.max_positions = o_.engine.max_positions;
        std::vector<qc::StreamStep> r;
        const auto ts = Clock::now();
        try {
            r = qc::step(e_, streams, opt);
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - ts).count();
            step_ms_ = step_ms_ > 0 ? 0.5 * step_ms_ + 0.5 * ms : ms;
        } catch (const std::exception& ex) {
            e_.abort_cohort();
            for (auto* s : live) finish(*s, std::string("engine failure: ") + ex.what(), "length");
            return;
        }
        const std::uint64_t cap = o_.engine.max_positions;
        for (std::size_t i = 0; i < live.size(); ++i) {
            Slot& s = *live[i];
            auto& job = s.job->job;
            const auto& step = r[i];
            s.stream = streams[i];
            if (step.failed) {  // only this request: its window was undone and the other sessions re-ran alone
                std::cerr << "strix-cuda-qwen-serve: request failed in decode: " << step.error << std::endl;
                finish(s, step.error, "length");
                continue;
            }
            ++s.st.cycles;
            s.st.proposed += step.drafted;
            s.st.accepted += step.tokens.size() - 1;
            s.rows += step.rows;
            s.window_sessions += step.window_sessions;
            s.max_window_sessions = std::max<std::uint64_t>(s.max_window_sessions, step.window_sessions);
            s.window_rows += step.window_rows;
            std::vector<std::uint32_t> out;
            bool stopped = false;
            for (const auto t : step.tokens) {
                if (s.st.generated_tokens + out.size() >= job.max_tokens) break;
                out.push_back(t);
                if (std::find(job.stop_tokens.begin(), job.stop_tokens.end(), t) != job.stop_tokens.end()) { stopped = true; break; }
            }
            if (s.first && job.observed)
                s.st.first_token_ms = std::chrono::duration<double, std::milli>(Clock::now() - s.job->submitted).count();
            s.first = false;
            s.st.generated_tokens += out.size();
            s.emitted.insert(s.emitted.end(), out.begin(), out.end());
            if (job.on_tokens && !job.on_tokens(out)) { finish(s, {}, "cancelled"); continue; }
            if (stopped) { finish(s, {}, "stop"); continue; }
            if (s.st.generated_tokens >= job.max_tokens) { finish(s, {}, "length"); continue; }
            if (s.stream.position + 1 >= cap) { finish(s, {}, "context"); continue; }
        }
    }

    void finish(Slot& s, const std::string& error, const char* why) {
        auto& st = s.st;
        st.finish = why;
        if (s.phase == Phase::decode)
            st.decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - s.t_decode).count();
        const double queue_ms = std::chrono::duration<double, std::milli>(s.job->started - s.job->submitted).count();
        const double steps = st.cycles ? static_cast<double>(st.cycles) : 1.0;
        char head[1152];
        std::snprintf(head, sizeof(head),
            "{\"engine\":\"cuda-qwen-a8\",\"mtp_depth\":%u,\"tokens_per_cycle\":%.3f,\"prefill_ms\":%.2f,\"decode_ms\":%.2f,"
            "\"queue_ms\":%.2f,\"http_slots\":%u,\"prefill_chunks\":%u,\"overlapped_chunks\":%u,\"overlap_steps\":%u,\"prefill_wait_ms\":%.2f,\"prefill_overlap_gpu_ms\":%.2f,\"prefill_sync_ms\":%.2f,\"steps\":%llu,"
            "\"rows_per_step\":%.3f,\"cohort_sessions\":%.3f,\"cohort_sessions_max\":%llu,\"window_rows\":%.3f,\"prefill_auto_rows\":%u,\"prefill_fit_a_ms\":%.3f,\"prefill_fit_b_ms_per_row\":%.5f,"
            "\"max_positions\":%u,\"session_mib\":%.1f,\"admit_free_vram_mib\":%.1f,\"live_slots\":%u,\"ids\":[",
            mtp_ ? o_.mtp_depth : 0U, st.cycles ? static_cast<double>(st.generated_tokens) / static_cast<double>(st.cycles) : 0.0,
            st.prefill_ms, st.decode_ms, queue_ms, o_.http_slots, s.prefill_chunks, s.overlapped_chunks, s.overlap_steps, s.prefill_wait_ms, s.prefill_gpu_ms, s.prefill_sync_ms,
            static_cast<unsigned long long>(st.cycles), static_cast<double>(s.rows) / steps, static_cast<double>(s.window_sessions) / steps,
            static_cast<unsigned long long>(s.max_window_sessions), static_cast<double>(s.window_rows) / steps, o_.prefill_chunk ? o_.prefill_chunk : auto_rows_, fit_a_, fit_b_,
            o_.engine.max_positions, static_cast<double>(e_.session_bytes()) / 1048576.0, s.admit_free_mib, live_slots());
        std::string extra = head;
        for (std::size_t i = 0; i < s.emitted.size() && i < 4096; ++i) extra += (i ? "," : "") + std::to_string(s.emitted[i]);
        extra += "],\"prefix_cached\":" + std::to_string(st.cached_tokens) + ",\"prefix_restore_ms\":" + std::to_string(s.prefix_restore_ms)
            + ",\"prefix_capture_ms\":" + std::to_string(s.prefix_capture_ms) + "}";
        st.engine_extra.emplace_back("rtx", extra);
        queue_.finish(*s.job, std::move(st), error);
        s.job.reset();
        s.phase = Phase::prefill;
    }

    qp::Prefill& e_;
    const Options& o_;
    serve::GenerationQueue& queue_;
    px::Cache& cache_;
    std::vector<Slot> slots_;
    bool mtp_ = false;
    std::uint32_t max_chunk_ = 1024, auto_rows_ = 256, first_rows_ = 0, fit_n_ = 0, inflight_steps_ = 0;
    Slot* inflight_ = nullptr;  // overlap: the slot whose chunk is on the prefill stream
    double step_ms_ = 0, fit_[5] = {0, 0, 0, 0, 0}, fit_a_ = 0, fit_b_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto o = parse(argc, argv);
        std::cerr << "strix-cuda-qwen-serve: loading " << o.engine.model_gguf << std::endl;
        qp::Prefill engine(o.engine);
        std::cerr << "strix-cuda-qwen-serve: loaded " << engine.describe() << std::endl;
        const auto tokenizer = serve::BpeTokenizer::load_gguf(o.engine.model_gguf);
        serve::ApiConfig config;
        config.model_id = o.model_id;
        config.http_slots = o.http_slots;
        config.api_key = read_key(o.api_key_file);
        config.engine_description = o.http_slots > 1
            ? "cuda-qwen-a8 (RTX, node-exact dense A8 decode, MTP verify windows, multi-session cohort windows)"
            : "cuda-qwen-a8 (RTX, node-exact dense A8 decode, MTP verify windows)";
        config.engine_progress = true;
        config.stop_tokens.push_back(tokenizer->eos_token());
        if (const auto end = tokenizer->find("<|endoftext|>"); end != UINT32_MAX) config.stop_tokens.push_back(end);
        std::sort(config.stop_tokens.begin(), config.stop_tokens.end());
        config.stop_tokens.erase(std::unique(config.stop_tokens.begin(), config.stop_tokens.end()), config.stop_tokens.end());
        serve::GenerationQueue queue(o.max_queue);
        px::Cache prefix_cache(o.engine, o.prefix);
        config.state_cache_stats = [&prefix_cache] { return prefix_cache.stats_json(); };
        config.status_extra = [&prefix_cache](serve::Json& status) { status.set("prefix_cache", serve::parse_json(prefix_cache.stats_json())); };
        CudaQwenEngine adapter(engine, o, prefix_cache);
        std::optional<CohortServer> cohort;  // --http-slots > 1: continuous batching; 1: the serialized queue.run path
        if (o.http_slots > 1) {
            cohort.emplace(engine, o, queue, prefix_cache);
            std::cerr << "strix-cuda-qwen-serve: " << o.http_slots << " slots (cohort decode), session state "
                      << engine.session_bytes() / 1000000 << " MB each, prefill chunk "
                      << (o.prefill_chunk ? std::to_string(o.prefill_chunk) + " rows" : std::string("auto"))
                      << (o.prefill_overlap ? ", overlapped on a prefill stream (max " + std::to_string(o.prefill_overlap_rows) + " rows, " +
                                                  (o.prefill_poll ? "polled: decode steps run until it completes)" : "one step then wait)")
                                            : std::string(", between steps"))
                      << ", draft depth "
                      << o.cohort_depth << std::endl;
        }
        serve::ApiService api(config, tokenizer, queue, o.engine.max_positions, false);
        serve::HttpServerConfig http;
        http.bind_address = o.bind;
        http.port = static_cast<std::uint16_t>(o.port);
        serve::HttpServer server(http, [&api](const serve::HttpRequest& request, serve::HttpResponseWriter& response) {
            api.handle(request, response);
        });
        server.listen();
        api.set_state("ready");
        std::cerr << "strix-cuda-qwen-serve: listening on " << o.bind << ':' << server.port() << " model " << config.model_id << std::endl;
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
        std::atomic<bool> finished{false};
        std::thread http_thread([&server] { server.serve(); });
        std::thread watcher([&] {
            while (!finished.load() && !g_stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
            server.stop();
            queue.shutdown();
        });
        if (cohort) cohort->run();
        else queue.run(adapter);
        finished.store(true);
        watcher.join();
        http_thread.join();
        std::cerr << "strix-cuda-qwen-serve: stopped" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "strix-cuda-qwen-serve: " << e.what() << std::endl;
        return 1;
    }
}
