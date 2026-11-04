// h3-attn-sparse-thr-v1 adapter for the single-node driver (see device_forward.hpp, levers sparse_attention.hpp).
#include <hip/hip_runtime.h>

#include "h3/block/device_forward.hpp"

#include <stdexcept>

#include "h3/levers/sparse/sparse_attention.hpp"

namespace strix::h3::block {

namespace sp = strix::h3::levers::sparse;

namespace {
// DiT attention: split calls through the lever, others (none in the block loop) through the dense K1.
struct SparseOp final : DeviceAttentionOp {
    sp::SparseAttention* sa = nullptr;
    FlashDeviceAttention dense;
    void run(const std::uint16_t* d_q, const std::uint16_t* d_k, const std::uint16_t* d_vt, std::uint64_t vt_pitch, std::uint32_t rows,
        std::uint32_t keys, bool split, std::uint16_t* d_out, void* stream) override {
        if (split) sa->run(d_q, d_k, d_vt, vt_pitch, rows, keys, d_out, static_cast<hipStream_t>(stream));
        else dense.run(d_q, d_k, d_vt, vt_pitch, rows, keys, split, d_out, stream);
    }
    // the sparse path's contract counter is checked by end_forward (LUT violations); dense calls of this op report here
    std::uint64_t bound_rows(void* stream) override { return dense.bound_rows(stream); }
    void reset_bound_rows(void* stream) override { dense.reset_bound_rows(stream); }
};
}  // namespace

struct SparseAccel::Impl {
    sp::SparseAttention sa;
    SparseOp op;
    explicit Impl(const sp::Params& p) : sa(p) { op.sa = &sa; }
};

SparseAccel::SparseAccel(float tau, std::uint32_t dense_calls, std::uint32_t dense_layers, const std::string& extra_dense_layers) {
    sp::Params p;
    p.tau = tau;
    p.dense_calls = dense_calls;
    p.dense_layers = dense_layers;
    p.dense_layer_mask = extra_dense_layers.empty() ? 0 : sp::layer_mask(extra_dense_layers);
    impl_ = std::make_unique<Impl>(p);
}
SparseAccel::~SparseAccel() = default;
void SparseAccel::begin_run() { impl_->sa.begin_run(); }
DeviceAttentionOp& SparseAccel::op() { return impl_->op; }
void SparseAccel::begin_forward(std::uint32_t prefix_rows) { impl_->sa.begin_forward(prefix_rows); }
void SparseAccel::end_forward(void* stream) { impl_->sa.end_forward(static_cast<hipStream_t>(stream)); }
void SparseAccel::abandon_forward() { impl_->sa.abandon_forward(); }
SparseAccel::Info SparseAccel::info() const {
    const sp::SparseTelemetry& t = impl_->sa.telemetry();
    Info i;
    double sum = 0.0;
    for (double d : t.layer_density) {
        if (d < 0.0) ++i.dense_layers;
        else {
            ++i.sparse_layers;
            sum += d;
        }
    }
    i.mean_density = i.sparse_layers ? sum / i.sparse_layers : 0.0;
    i.run_dense_calls = t.dense_calls;
    i.run_sparse_calls = t.sparse_calls;
    i.model_call = impl_->sa.model_call();
    return i;
}

}  // namespace strix::h3::block
