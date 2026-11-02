#include "h3/vae/vae_decoder.hpp"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <map>
#include <thread>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>

#include "h3/vae/vae_kernels.hpp"
#include "h3/vae/vae_threads.hpp"

namespace strix::h3::vae {

namespace {
void ck(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("h3 vae decoder: ") + what + ": " + hipGetErrorString(e));
}
struct DevLinear { _Float16 *w = nullptr, *b = nullptr; int in = 0, out = 0; };
}  // namespace

// Owns every device allocation, the stream and the events; the destructor releases all of them, so a failed
// construction (exception after some allocations) and a destroyed decoder leave nothing behind.
struct GpuDecoder::Dev {
    struct Layer { _Float16 *norm1, *norm2, *scale1, *scale2; DevLinear qkv, out, w1, w2; };
    std::vector<Layer> layers;
    DevLinear x_embed, proj;
    _Float16 *registers = nullptr, *ln_w = nullptr, *ln_b = nullptr;
    // activations, sized for the largest call (7 x 16 x 16 latents)
    static constexpr unsigned kMaxP = 7U * 16U * 16U, kMaxL = kMaxP + kSuffix, kMaxLp = (kMaxL + 63U) / 64U * 64U;
    _Float16 *x = nullptr, *a = nullptr, *qkv = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr, *hb = nullptr,
             *g = nullptr, *proj_out = nullptr, *zin = nullptr, *rc = nullptr, *rs = nullptr;
    std::map<std::tuple<int, int, int>, std::pair<H16, H16>> rope_cache;
    hipStream_t st = nullptr;
    hipEvent_t ev[4] = {};
    std::vector<void*> owned;

    ~Dev() {
        (void)hipDeviceSynchronize();
        for (void* p : owned) (void)hipFree(p);
        for (auto& e : ev) if (e) (void)hipEventDestroy(e);
        if (st) (void)hipStreamDestroy(st);
    }
    _Float16* alloc(std::size_t halves) {
        void* p = nullptr;
        ck(hipMalloc(&p, halves * 2), "hipMalloc");
        owned.push_back(p);
        return static_cast<_Float16*>(p);
    }
    _Float16* upload(const H16& v) {
        _Float16* p = alloc(v.size());
        ck(hipMemcpy(p, v.data(), v.size() * 2, hipMemcpyHostToDevice), "upload");
        return p;
    }
    DevLinear upload(const LinearH& l) { return {upload(l.w), upload(l.b), l.in, l.out}; }
};

GpuDecoder::GpuDecoder(const std::string& weights, int layers, bool fused_swiglu) : layers_(layers), fused_(fused_swiglu) {
    SafeTensors stf(weights);
    head_ = load_head(stf);
    auto dev = std::make_unique<Dev>();  // released on any exception below
    Dev& d = *dev;
    ck(hipStreamCreate(&d.st), "stream");
    for (auto& e : d.ev) ck(hipEventCreate(&e), "event");
    for (int l = 0; l < layers; ++l) {
        const LayerH w = load_layer(stf, l);
        LinearH w1 = w.w1;
        if (fused_) {  // rows (and bias) in the fused epilogue's gate / up order
            LinearH p = w1;
            for (unsigned r = 0; r < unsigned(w1.out); ++r) {
                const unsigned src = gpu::swiglu_source_row(r, unsigned(w1.out));
                std::memcpy(p.w.data() + std::size_t(r) * w1.in, w1.w.data() + std::size_t(src) * w1.in, std::size_t(w1.in) * 2);
                p.b[r] = w1.b[src];
            }
            w1 = std::move(p);
        }
        Dev::Layer dl{d.upload(w.norm1), d.upload(w.norm2), d.upload(w.scale1), d.upload(w.scale2),
                      d.upload(w.qkv), d.upload(w.out), d.upload(w1), d.upload(w.w2)};
        d.layers.push_back(dl);
    }
    d.x_embed = d.upload(head_.x_embed);
    d.proj = d.upload(head_.proj);
    d.registers = d.upload(head_.registers);
    d.ln_w = d.upload(head_.norm_out_w);
    d.ln_b = d.upload(head_.norm_out_b);
    const std::size_t L = Dev::kMaxL, Lp = Dev::kMaxLp;
    d.x = d.alloc(L * kDim); d.a = d.alloc(L * kDim); d.qkv = d.alloc(L * kQkv);
    d.q = d.alloc(32 * Lp * 64); d.k = d.alloc(32 * Lp * 64); d.v = d.alloc(32 * Lp * 64);
    d.o = d.alloc(L * kDim); d.hb = d.alloc(L * kFf2); d.g = d.alloc(L * kFf);
    d.proj_out = d.alloc(std::size_t(Dev::kMaxP) * kProj); d.zin = d.alloc(std::size_t(kLatentC) * Dev::kMaxP);
    d.rc = d.alloc(L * kRotPairs); d.rs = d.alloc(L * kRotPairs);
    // Q/K/V padding rows [L, Lp) are rewritten with zeros by every call (rms_rope)
    d_ = dev.release();
}

GpuDecoder::~GpuDecoder() { delete d_; }

H16 GpuDecoder::decode_call(const H16& dec_in, int nt, int ny, int nx, CallTiming* timing, H16* h_out, int stop_after,
                            const H16* h_in_override, H16* rows_out) {
    const unsigned P = unsigned(nt * ny * nx), L = P + kSuffix, Lp = (L + 63U) / 64U * 64U;
    if (nt < 1 || ny < 1 || nx < 1 || P > Dev::kMaxP) throw std::invalid_argument("h3 vae decoder: call box must be 1..7 x 1..16 x 1..16 (P <= 1792)");
    if (h_in_override) {
        if (h_in_override->size() != std::size_t(L) * kDim) throw std::invalid_argument("h3 vae decoder: residual input length != L x 2048");
    } else if (dec_in.size() != std::size_t(kLatentC) * P) {
        throw std::invalid_argument("h3 vae decoder: call input length != 24 x nt x ny x nx");
    }
    if (stop_after > layers_) throw std::invalid_argument("h3 vae decoder: stop_after beyond the loaded layers");
    auto& rope = d_->rope_cache[{nt, ny, nx}];
    if (rope.first.empty()) {
        const RopeTable r = rope_table(nt, ny, nx);
        rope.first.resize(r.c.size()); rope.second.resize(r.s.size());
        for (std::size_t i = 0; i < r.c.size(); ++i) { rope.first[i] = f2h(r.c[i]); rope.second[i] = f2h(r.s[i]); }
    }
    hipStream_t st = d_->st;
    const auto t0 = std::chrono::steady_clock::now();
    ck(hipMemcpyAsync(d_->rc, rope.first.data(), rope.first.size() * 2, hipMemcpyHostToDevice, st), "rope c");
    ck(hipMemcpyAsync(d_->rs, rope.second.data(), rope.second.size() * 2, hipMemcpyHostToDevice, st), "rope s");
    ck(hipEventRecord(d_->ev[0], st), "ev");
    if (h_in_override) {
        ck(hipMemcpyAsync(d_->x, h_in_override->data(), std::size_t(L) * kDim * 2, hipMemcpyHostToDevice, st), "h_in");
    } else {
        ck(hipMemcpyAsync(d_->zin, dec_in.data(), std::size_t(kLatentC) * P * 2, hipMemcpyHostToDevice, st), "zin");
        gpu::embed(d_->zin, d_->x_embed.w, d_->x_embed.b, d_->registers, d_->x, P, st);
    }
    ck(hipEventRecord(d_->ev[1], st), "ev");
    const int nl = stop_after < 0 ? layers_ : stop_after;
    // STRIX_H3_VAE_PROFILE=1: events around every op of every layer (timing only; same launches)
    static const bool profile = [] { const char* v = std::getenv("STRIX_H3_VAE_PROFILE"); return v && v[0] == '1'; }();
    std::vector<hipEvent_t> pev;
    const auto mark = [&] {
        if (!profile) return;
        hipEvent_t e;
        ck(hipEventCreate(&e), "event");
        ck(hipEventRecord(e, st), "event");
        pev.push_back(e);
    };
    mark();
    for (int li = 0; li < nl; ++li) {
        const auto& w = d_->layers[std::size_t(li)];
        gpu::rms_norm(d_->x, w.norm1, d_->a, L, st); mark();
        gpu::gemm(d_->a, kDim, w.qkv.w, w.qkv.b, d_->qkv, kQkv, L, kQkv, kDim, st); mark();
        gpu::rms_rope(d_->qkv, d_->rc, d_->rs, d_->q, d_->k, d_->v, L, Lp, st); mark();
        gpu::attention(d_->q, d_->k, d_->v, d_->o, L, Lp, st); mark();
        gpu::gemm_residual(d_->o, kDim, w.out.w, w.out.b, w.scale1, d_->x, kDim, L, kDim, kDim, st); mark();
        gpu::rms_norm(d_->x, w.norm2, d_->a, L, st); mark();
        if (fused_) {
            gpu::gemm_swiglu(d_->a, kDim, w.w1.w, w.w1.b, d_->g, kFf, L, kFf2, kDim, st); mark(); mark();
        } else {
            gpu::gemm(d_->a, kDim, w.w1.w, w.w1.b, d_->hb, kFf2, L, kFf2, kDim, st); mark();
            gpu::swiglu(d_->hb, d_->g, L, st); mark();
        }
        gpu::gemm_residual(d_->g, kFf, w.w2.w, w.w2.b, w.scale2, d_->x, kDim, L, kDim, kFf, st); mark();
    }
    if (profile && !pev.empty()) {
        ck(hipStreamSynchronize(st), "sync");
        static const char* names[9] = {"rms1", "qkv", "rmsrope", "attn", "out", "rms2", "w1", "swiglu", "w2"};
        double t[9] = {};
        for (std::size_t i = 1; i < pev.size(); ++i) {
            float ms = 0;
            ck(hipEventElapsedTime(&ms, pev[i - 1], pev[i]), "elapsed");
            t[(i - 1) % 9] += ms;
        }
        std::printf("  profile L=%u:", L);
        for (int i = 0; i < 9; ++i) std::printf(" %s %.1f", names[i], t[i]);
        std::printf(" ms\n");
        for (auto e : pev) (void)hipEventDestroy(e);
    }
    ck(hipEventRecord(d_->ev[2], st), "ev");
    if (h_out) {
        h_out->resize(std::size_t(L) * kDim);
        ck(hipMemcpyAsync(h_out->data(), d_->x, h_out->size() * 2, hipMemcpyDeviceToHost, st), "h_out");
    }
    gpu::layer_norm(d_->x, d_->ln_w, d_->ln_b, d_->a, P, st);
    gpu::gemm(d_->a, kDim, d_->proj.w, d_->proj.b, d_->proj_out, kProj, P, kProj, kDim, st);
    ck(hipEventRecord(d_->ev[3], st), "ev");
    H16 rows(std::size_t(P) * kProj);
    ck(hipMemcpyAsync(rows.data(), d_->proj_out, rows.size() * 2, hipMemcpyDeviceToHost, st), "proj rows");
    ck(hipStreamSynchronize(st), "sync");
    H16 out;
    if (rows_out) *rows_out = std::move(rows);  // caller unpatchifies (off the GPU's critical path)
    else out = unpatchify(rows, nt, ny, nx);
    if (timing) {
        float a = 0, b = 0;
        ck(hipEventElapsedTime(&a, d_->ev[0], d_->ev[1]), "elapsed");
        ck(hipEventElapsedTime(&b, d_->ev[1], d_->ev[2]), "elapsed");
        timing->embed_ms = a;
        timing->layers_ms = b;
        ck(hipEventElapsedTime(&a, d_->ev[2], d_->ev[3]), "elapsed");
        timing->head_ms = a;
        timing->total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    return out;
}

// Host work overlaps the GPU: the call inputs (latent box + post_quant_conv) are built by worker threads ahead of the
// GPU loop, and each call's unpatchify runs on a worker while the GPU decodes the next calls. Same functions on the
// same data: the outputs are the bits of the serial loop.
std::vector<Pix16> GpuDecoder::decode_calls(const DecodePlan& p, const H16& zp, CallRange r) {
    const auto calls = decoder_calls(p);
    if (zp.size() != std::size_t(24) * std::size_t(p.T + p.pad_tokens) * std::size_t(p.H) * std::size_t(p.W))
        throw std::invalid_argument("h3 vae decoder: padded latent length != 24 x (T + pad) x H x W");
    if (r.first < 0 || r.last > int(calls.size()) || r.first > r.last) throw std::invalid_argument("h3 vae decoder: call range");
    const int n = r.last - r.first;
    std::vector<Pix16> out(static_cast<std::size_t>(std::max(0, n)));
    if (n <= 0) return out;
    const unsigned workers = std::max(2U, std::min(8U, std::thread::hardware_concurrency() / 2U));
    std::vector<H16> inputs(static_cast<std::size_t>(n)), rows(static_cast<std::size_t>(n));
    {
        std::atomic<int> next{0};
        Workers pool;  // declared after the buffers it writes: joined before they are destroyed
        for (unsigned w = 0; w < workers; ++w)
            pool.run([&] {
                for (int i = next++; i < n; i = next++) {
                    maybe_fail(FaultStage::call_input, i);
                    inputs[std::size_t(i)] = call_input(head_, p, zp, calls[std::size_t(r.first + i)]);
                }
            });
        pool.join();  // rethrows a worker's exception
    }
    Workers post;  // unpatchify workers, at most `workers` in flight; joined on every path (also if the GPU loop throws)
    for (int i = 0; i < n; ++i) {
        const DecoderCall& c = calls[std::size_t(r.first + i)];
        maybe_fail(FaultStage::gpu_loop, i);
        (void)decode_call(inputs[std::size_t(i)], c.nt, c.ny, c.nx, nullptr, nullptr, -1, nullptr, &rows[std::size_t(i)]);
        H16().swap(inputs[std::size_t(i)]);
        if (post.failed()) break;  // stop early; join() below rethrows
        if (post.size() >= workers) post.join_oldest();
        post.run([&, i, c] {
            maybe_fail(FaultStage::unpatchify, i);
            Pix16& t = out[std::size_t(i)];
            t.C = 3; t.F = 4 * c.nt; t.Hh = 16 * c.ny; t.Ww = 16 * c.nx;
            t.v = unpatchify(rows[std::size_t(i)], c.nt, c.ny, c.nx);
            H16().swap(rows[std::size_t(i)]);
        });
    }
    post.join();
    return out;
}

std::vector<float> GpuDecoder::decode(const H16& z_norm, int T, int H, int W, double* seconds) {
    const auto t0 = std::chrono::steady_clock::now();
    const DecodePlan p = make_plan(T, H, W);
    const H16 zp = denormalize_and_pad(p, z_norm, head_.latents_mean.data(), head_.latents_std.data());
    // one node = the two-node gather path with one rank: chunks are finalized while the GPU decodes the next ones
    FrameAssembler frames(p);
    decode_pieces(p, zp, {0, p.calls()}, [&](DecodePiece&& x) { frames.push(std::move(x)); });
    auto out = frames.take();
    if (seconds) *seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

// Whole chunks [r.first, r.last) as canvases (decode_pieces over the chunks' calls; every piece is whole).
std::vector<Pix16> GpuDecoder::decode_chunks(const DecodePlan& p, const H16& zp, ChunkRange r) {
    if (r.first < 0 || r.last > p.num_chunks || r.first > r.last) throw std::invalid_argument("h3 vae decoder: chunk range");
    const int per = int(p.ty.start.size() * p.tx.start.size());
    std::vector<Pix16> canvases(static_cast<std::size_t>(r.last - r.first));
    decode_pieces(p, zp, {r.first * per, r.last * per}, [&](DecodePiece&& x) { canvases[std::size_t(x.chunk - r.first)] = std::move(x.canvas); });
    return canvases;
}

// Segment-pipelined decode of a call range (a segment = the range's calls inside one chunk): the GPU runs the calls in
// canonical order; when a segment's last call is done, a worker unpatchifies its calls and, for a whole chunk,
// assembles the canvas while the GPU decodes the next segment. The next segment's call inputs are built ahead by a
// worker. Same functions on the same data in the same per-chunk order as decode_calls + blend_calls: the same bits.
void GpuDecoder::decode_pieces(const DecodePlan& p, const H16& zp, CallRange r, const std::function<void(DecodePiece&&)>& sink) {
    const auto calls = decoder_calls(p);
    if (zp.size() != std::size_t(24) * std::size_t(p.T + p.pad_tokens) * std::size_t(p.H) * std::size_t(p.W))
        throw std::invalid_argument("h3 vae decoder: padded latent length != 24 x (T + pad) x H x W");
    const auto spans = piece_spans(p, r);  // validates the range
    const int nseg = int(spans.size());
    if (nseg == 0) return;
    std::vector<std::vector<H16>> rows(static_cast<std::size_t>(nseg)), inputs(static_cast<std::size_t>(nseg));
    for (int k = 0; k < nseg; ++k) rows[std::size_t(k)].resize(std::size_t(spans[std::size_t(k)].call_last - spans[std::size_t(k)].call_first));
    std::mutex sink_mu;
    const auto build_inputs = [&](int k) {  // segment k: its calls' inputs
        const PieceSpan& sp = spans[std::size_t(k)];
        auto& in = inputs[std::size_t(k)];
        in.resize(std::size_t(sp.call_last - sp.call_first));
        for (int ci = sp.call_first; ci < sp.call_last; ++ci) {
            maybe_fail(FaultStage::call_input, ci);
            in[std::size_t(ci - sp.call_first)] = call_input(head_, p, zp, calls[std::size_t(ci)]);
        }
    };
    build_inputs(0);
    Workers ahead, post;  // declared after the buffers they use: joined before those are destroyed
    for (int k = 0; k < nseg; ++k) {
        const PieceSpan sp = spans[std::size_t(k)];
        if (k + 1 < nseg) ahead.run([&, k] { build_inputs(k + 1); });
        for (int ci = sp.call_first; ci < sp.call_last; ++ci) {
            const DecoderCall& c = calls[std::size_t(ci)];
            maybe_fail(FaultStage::gpu_loop, ci);
            (void)decode_call(inputs[std::size_t(k)][std::size_t(ci - sp.call_first)], c.nt, c.ny, c.nx, nullptr, nullptr, -1, nullptr,
                              &rows[std::size_t(k)][std::size_t(ci - sp.call_first)]);
        }
        std::vector<H16>().swap(inputs[std::size_t(k)]);
        post.run([&, k, sp] {
            DecodePiece x;
            x.chunk = sp.chunk; x.call_first = sp.call_first; x.call_last = sp.call_last; x.whole = sp.whole;
            std::vector<Pix16> tiles(static_cast<std::size_t>(sp.call_last - sp.call_first));
            for (int ci = sp.call_first; ci < sp.call_last; ++ci) {
                const DecoderCall& c = calls[std::size_t(ci)];
                maybe_fail(FaultStage::unpatchify, ci);
                Pix16& t = tiles[std::size_t(ci - sp.call_first)];
                t.C = 3; t.F = 4 * c.nt; t.Hh = 16 * c.ny; t.Ww = 16 * c.nx;
                t.v = unpatchify(rows[std::size_t(k)][std::size_t(ci - sp.call_first)], c.nt, c.ny, c.nx);
                H16().swap(rows[std::size_t(k)][std::size_t(ci - sp.call_first)]);
            }
            maybe_fail(FaultStage::chunk_assembly, sp.chunk);
            if (sp.whole) {
                const int nx = int(p.tx.start.size());
                x.canvas = assemble_chunk(p, [&](int i, int jj) { return tiles[std::size_t(i * nx + jj)]; });
            } else {
                x.tiles = std::move(tiles);
            }
            const std::lock_guard<std::mutex> lock(sink_mu);
            sink(std::move(x));
        });
        ahead.join();  // next segment's inputs (rethrows a worker failure)
        if (post.failed()) break;
    }
    post.join();
}

}  // namespace strix::h3::vae
