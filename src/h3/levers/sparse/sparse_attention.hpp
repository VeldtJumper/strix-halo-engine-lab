#pragma once
// Runner hook of the sparse-attention lever (identity h3-attn-sparse-thr-v1, FAST-class, OFF by default; 1 node v1).
// One SparseAttention per sampling run on one stream; the block runner wraps it in a DeviceAttentionOp adapter:
//
//   sparse::SparseAttention sa(params);                    // tau (0.75), dense_calls (3), dense_layers (2), band (1)
//   sa.begin_run();                                        // start of every sampling run: model-call counter = 0
//   per model call (one DiT forward):
//     sa.begin_forward(P);                                 // P = rows before the target video (layout video start)
//     DeviceForward::run(..., adapter, ...)                // adapter::run(q, k, vt, pitch, rows, keys, split, out, s):
//                                                          //   split == false (refiner): the dense op, unchanged;
//                                                          //   split == true: sa.run(q, k, vt, pitch, rows, keys, out, s)
//     sa.end_forward(s);                                   // throws if any LUT contract violation was counted
//
// Model calls: count ONLY calls that run the DiT blocks: with FirstBlockCache, a reused call (blocks 1..
// skipped) must NOT advance the dense-window counter - the runner calls begin_forward / end_forward only around
// forwards that run the blocks, or calls abandon_forward() when FBC decides to reuse after block 0 (block 0's call
// has already been made: layer 0 is a dense layer anyway). Dense layers: layers < dense_layers plus
// Params::dense_layer_mask (e.g. layer_mask("39-42")).
// Calls are counted in order: DiT layer = index of the split call within the forward. Every call of this identity uses
// the lever cut 128 floor(S / 256) (two key blocks, canonical merge): dense calls (model call < dense_calls, layer <
// dense_layers, or a degenerate shape) run dense K1 at that cut; the other calls run the router (router.hpp) + the sparse
// K1 variant (attn_sparse.hpp). Q / K token-major [S][56][128] (row pitch 7168), V^T [56][128][vt_pitch], out [S][7168]
// bf16. Requires rows == keys (one node). Telemetry: calls, mean density, union inflation, per-layer density.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "h3/attn/attn.hpp"
#include "h3/levers/sparse/attn_sparse.hpp"
#include "h3/levers/sparse/router.hpp"

namespace strix::h3::levers::sparse {

struct SparseTelemetry {
    std::uint64_t dense_calls = 0, sparse_calls = 0;
    double density_sum = 0.0;                    // over sparse calls
    unsigned long long union_sum = 0, parts_sum = 0;
    std::vector<double> layer_density;           // last forward, per layer (-1 = dense call)
    double mean_density() const { return sparse_calls ? density_sum / static_cast<double>(sparse_calls) : 0.0; }
};

class SparseAttention {
public:
    explicit SparseAttention(Params p = {}) : p_(p) {
        if (!(p.tau == p.tau)) throw std::invalid_argument("h3 sparse: tau NaN");
        dev::check(hipMalloc(&lut_errors_, 4), "malloc");
        dev::check(hipMemset(lut_errors_, 0, 4), "memset");
        dev::check(hipMalloc(&dummy_, 16), "malloc");
    }
    SparseAttention(const SparseAttention&) = delete;
    SparseAttention& operator=(const SparseAttention&) = delete;
    ~SparseAttention() {
        static_cast<void>(hipFree(lut_errors_));
        static_cast<void>(hipFree(dummy_));
    }
    const Params& params() const { return p_; }
    const SparseTelemetry& telemetry() const { return tele_; }
    void begin_run() {
        call_ = 0;
        in_forward_ = false;
        tele_ = SparseTelemetry{};
    }
    void begin_forward(std::uint32_t prefix_rows) {
        if (in_forward_) throw std::logic_error("h3 sparse: begin_forward twice");
        P_ = prefix_rows;
        layer_ = 0;
        in_forward_ = true;
        tele_.layer_density.clear();
    }
    void end_forward(hipStream_t s) {
        if (!in_forward_) throw std::logic_error("h3 sparse: end_forward without begin_forward");
        in_forward_ = false;
        ++call_;
        unsigned errs = 0;
        dev::check(hipMemcpyAsync(&errs, lut_errors_, 4, hipMemcpyDeviceToHost, s), "copy");
        dev::check(hipStreamSynchronize(s), "sync");
        if (errs) throw std::runtime_error("h3 sparse: " + std::to_string(errs) + " LUT contract violations");
    }
    // FBC reuse after block 0: close the forward WITHOUT advancing the model-call counter.
    void abandon_forward() {
        if (!in_forward_) throw std::logic_error("h3 sparse: abandon_forward without begin_forward");
        in_forward_ = false;
    }
    std::uint32_t model_call() const { return call_; }

    void run(const std::uint16_t* q, const std::uint16_t* k, const std::uint16_t* vt, std::uint64_t vt_pitch, std::uint32_t rows, std::uint32_t keys,
        std::uint16_t* out, hipStream_t s) {
        if (!in_forward_) throw std::logic_error("h3 sparse: run outside begin_forward / end_forward");
        if (rows != keys) throw std::invalid_argument("h3 sparse: one node only (rows == keys)");
        const Geometry g = geometry(keys, P_);
        const std::uint32_t layer = layer_++;
        constexpr std::uint64_t W = static_cast<std::uint64_t>(kHeads) * kDim;
        part_.need(attn::attention_scratch_floats(rows, kHeads, kDim, 2) * 4);
        if (!sparse_call(p_, g, call_, layer)) {
            const std::uint32_t cut = g.cut ? g.cut : 0U;
            std::uint32_t cuts[3] = {0U, cut, keys};
            if (!cut) cuts[1] = keys;
            attn::AttentionCall c;
            c.q = q;
            c.k = k;
            c.vt = vt;
            c.rows = rows;
            c.keys = keys;
            c.vt_pitch = vt_pitch;
            c.cuts = cuts;
            c.nblocks = cut ? 2U : 1U;
            c.scratch = part_.as<float>();
            c.out = out;
            attn::attention(c, s);
            ++tele_.dense_calls;
            tele_.layer_density.push_back(-1.0);
            return;
        }
        const std::uint32_t jc = g.cut / kTile;
        kc_.need(static_cast<std::size_t>(g.KT) * W * 2);
        qbar_.need(static_cast<std::size_t>(g.KT) * W * 4);
        st_.need(4U * W * 4);
        mv_.need(2U * W * 4);
        thr_.need(static_cast<std::size_t>(g.KT) * kHeads * 4);
        std::uint16_t* kc = kc_.as<std::uint16_t>();
        float* st = st_.as<float>();
        float* mean = mv_.as<float>();
        float* var = mean + W;
        launch_key_centroids(k, W, g, 0, jc, kc, s);
        launch_key_centroids(k + static_cast<std::size_t>(g.cut) * W, W, g, jc, g.KT, kc + static_cast<std::size_t>(jc) * W, s);
        launch_stats_partial(kc, jc, st, st + W, s);
        launch_stats_partial(kc + static_cast<std::size_t>(jc) * W, g.KT - jc, st + 2 * W, st + 3 * W, s);
        launch_stats_finalize(st, st + W, st + 2 * W, st + 3 * W, g.KT, mean, var, s);
        launch_query_centroids(q, W, g, 0, g.KT, qbar_.as<float>(), s);
        launch_thresholds(qbar_.as<float>(), g.KT, mean, var, p_.tau, thr_.as<float>(), s);
        const DeviceLut l0 = build_lut(g, p_, 0, g.QT, qbar_.as<float>(), thr_.as<float>(), 0, jc, kc, s);
        const DeviceLut l1 = build_lut(g, p_, 0, g.QT, qbar_.as<float>(), thr_.as<float>(), jc, g.KT, kc + static_cast<std::size_t>(jc) * W, s);
        const BlockLut luts[2] = {BlockLut{l0.entries_n ? l0.entries.as<std::int32_t>() : dummy_, l0.offsets.as<std::uint32_t>(), 0},
                                  BlockLut{l1.entries_n ? l1.entries.as<std::int32_t>() : dummy_, l1.offsets.as<std::uint32_t>(), jc}};
        float* o = part_.as<float>();
        const attn::Partials parts{o, o + static_cast<std::size_t>(2) * kHeads * rows * kDim, rows, kHeads, kDim, nullptr};
        const attn::Queries qq{q, kDim, W, rows, kHeads, kDim};
        const attn::KeyBlock blocks[2] = {attn::KeyBlock{k, vt, kDim, W, kDim * vt_pitch, vt_pitch, g.cut},
                                          attn::KeyBlock{k + static_cast<std::size_t>(g.cut) * W, vt + g.cut, kDim, W, kDim * vt_pitch, vt_pitch, keys - g.cut}};
        launch_sparse_partials(qq, blocks, luts, 2, parts, 0, lut_errors_, s);
        attn::launch_merge(parts, 2, attn::Output{out, nullptr, W, nullptr}, s);
        dev::check(hipStreamSynchronize(s), "sync");  // the LUT buffers (l0 / l1) are freed at scope exit
        const double dens = static_cast<double>(l0.entries_n + l1.entries_n) / (static_cast<double>(g.QT) * kHeads * g.KT);
        ++tele_.sparse_calls;
        tele_.density_sum += dens;
        tele_.union_sum += l0.union_sum + l1.union_sum;
        tele_.parts_sum += l0.parts_sum + l1.parts_sum;
        tele_.layer_density.push_back(dens);
    }

private:
    Params p_;
    std::uint32_t P_ = 0, layer_ = 0, call_ = 0;
    bool in_forward_ = false;
    unsigned* lut_errors_ = nullptr;
    std::int32_t* dummy_ = nullptr;
    SparseTelemetry tele_;
    struct Grow {
        dev::Buf b;
        std::size_t bytes = 0;
        void need(std::size_t n) {
            if (n <= bytes) return;
            b = dev::Buf(n);
            bytes = n;
        }
        template <class T> T* as() const { return b.as<T>(); }
    };
    Grow kc_, qbar_, st_, mv_, thr_, part_;
};

}  // namespace strix::h3::levers::sparse
