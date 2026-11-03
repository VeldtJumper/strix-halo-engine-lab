#pragma once
// MiniMax H3 two-node schedule (D1) on the device: the DiT block stack on one rank's rows with the per-block K/V
// exchange on the engine's Communicator (chunked host pipeline, CRC-32C trailers, start_async / finish) overlapping
// the local-block attention partial. world 1 runs the same code with both key blocks local (the 1-node path).
// Buffers, events and hazards: see host_dist.hpp; the arithmetic is the gemm, attn and block (glue kernels)
// packages, unchanged.
#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "h3/block/host_ref.hpp"
#include "h3/dist/accel.hpp"
#include "h3/dist/admission.hpp"
#include "h3/dist/host_dist.hpp"
#include "h3/dist/schedule.hpp"

namespace strix::dist {
class Communicator;
}

namespace strix::h3::dist {

// Order-free digest of residual rows (index0 = global element index of x[0]); the device driver computes the same value
// per block and owned key block without a host sync.
std::uint64_t residual_digest(const std::uint16_t* x, std::uint64_t n, std::uint64_t index0);

// hipDeviceSynchronize() == hipSuccess (teardown check of the failure-injection test).
bool device_synchronize_ok() noexcept;

// Block weights resident on the device (int8 projections, fc1 permuted for the fused SwiGLU, norms, adaLN fp32).
// Shared by the ranks of one process (read only after construction).
class DeviceModel {
public:
    // fc1_plain: keep fc1 in checkpoint order (forward runs fc1 unfused + the K5 SwiGLU: the same bits as the fused
    // epilogue; required by Turbo). For Turbo the block source is block::MergedFc2Source. `g` is borrowed (keep it alive).
    DeviceModel(const block::GlobalWeights& g, block::BlockSource& blocks, std::uint32_t nblocks, bool fc1_plain = false);
    bool fc1_plain() const;
    // Worst-case operand bound from the q / k norm weights alone (loose: assumes every dim at max |w|; info only).
    // The admission gate is the measured one (DeviceRunResult::admission_bound).
    double weight_bound() const;
    ~DeviceModel();
    DeviceModel(const DeviceModel&) = delete;
    DeviceModel& operator=(const DeviceModel&) = delete;
    std::uint32_t num_blocks() const;
    std::uint64_t resident_bytes() const;
    struct Impl;
    Impl& impl() const { return *impl_; }

private:
    std::unique_ptr<Impl> impl_;
};

struct DeviceRunOptions {
    bool hashes = true;   // download the residual after every block (per-block FNV-1a of each owned key block)
    bool timing = false;  // per-block events (exposure etc.)
    bool profile = false; // per-kernel-group events in every block (time split, kernel_ms) + a standalone quantize bench
    // Test hooks (failure injection) on block `inject_block` (must be < nblocks, else the run is refused):
    //   throw_in_flight  world 2: throw InjectedFailure right after the block's exchange started (transfer in flight)
    //   nan_q            write a bf16 NaN into this rank's Q before the operand scan (admission must reject it)
    //   v_big            write 2^101 into this rank's V (scan input, after V^T was built) (|V| <= 2^100 violated)
    //   bound_rows       set this rank's attention bound-rows counter to 1
    enum class Inject { none, throw_in_flight, nan_q, v_big, bound_rows };
    Inject inject = Inject::none;
    std::int32_t inject_block = -1;
};

// Thrown only by the throw_in_flight test hook.
class InjectedFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct DeviceBlockTiming {
    double block_ms = 0;      // block start -> next block start (GPU timeline)
    double attn_ms = 0;       // local partial + remote partial + merge (GPU time of the kernels)
    double local_ms = 0;      // local-block partial
    double exposed_ms = 0;    // GPU gap between the local partial's end and the stream passing the RX wait
    double finish_host_ms = 0;  // CPU time inside finish()
};

struct DeviceRunResult {
    std::vector<std::array<std::uint64_t, 2>> hash;    // [block][key block] FNV-1a (opt.hashes), 0 where not owned
    std::vector<std::array<std::uint64_t, 2>> digest;  // [block][key block] device digest (always), 0 where not owned
    std::uint64_t attn_bound_rows = 0;               // attention contract telemetry of this rank (both ranks' sum != 0 throws)
    double admission_bound = 0;                      // measured max over blocks of ||q|| ||k|| log2(e)/sqrt(128) (<= 2^14, else throws)
    std::uint32_t admission_block = 0;
    double v_max = 0;                                // measured max |V| (finite, <= 2^100, else throws)
    std::vector<DeviceBlockTiming> timing;
    // opt.profile: GPU time per kernel group summed over the blocks (ms; group names in kernel_names), and the
    // standalone activation quantizer time per call at K = 5376 / 7168 / 14336 on this rank's rows (median of 5)
    std::vector<std::string> kernel_names;
    std::vector<double> kernel_ms;
    double quantize_ms[3] = {0, 0, 0};
    std::vector<std::uint16_t> x;                    // the rank's rows after the last block
    std::uint64_t flushed_groups = 0;                // gemm rotation groups outside the exact window (must be 0)
    double wall_s = 0;
    std::string comm_stats;
};

// One rank of the schedule. `comm` is required for world 2 (built without a device exchange: host path only).
class DeviceDistRank {
public:
    DeviceDistRank(const DeviceModel& model, const Split& sp, const Role& role, strix::dist::Communicator* comm);
    ~DeviceDistRank();
    DeviceDistRank(const DeviceDistRank&) = delete;
    DeviceDistRank& operator=(const DeviceDistRank&) = delete;
    // x = the rank's rows [rows][5376] (bf16 bits); blocks [0, nblocks).
    DeviceRunResult run(const ForwardTables& t, std::uint32_t basis, const std::vector<std::uint16_t>& x, std::uint32_t nblocks,
        const DeviceRunOptions& opt);

    // Whole forward from the stream inputs (text = the refined text, replicated on both ranks): embedding of the
    // rank's rows, blocks [0, nblocks), final layer of the rank's video / audio rows, then one symmetric exchange of
    // the final rows (an all-gather: both ranks end with the full velocity, rank 0 is the one the sampler uses).
    // Returns -velocity as bf16 bits (the stream_forward_host contract) in `out`.
    struct ForwardResult {
        DeviceRunResult run;
        block::StreamOutputs out;
        double embed_s = 0, blocks_s = 0, final_s = 0, gather_s = 0, wall_s = 0;
        std::size_t gather_bytes = 0;
    };
    // accel: FAST-class accelerators (accel.hpp; nullptr = the dense path, bit for bit unchanged). Both ranks must pass
    // equivalent accelerators (same assets, threshold, forced decisions); one DistAccel per rank object.
    ForwardResult forward(const block::PackedLayout& L, const block::ForwardPlan& p, const block::StreamInputs& in, const ForwardTables& t,
        std::uint32_t nblocks, const DeviceRunOptions& opt, DistAccel* accel = nullptr);

    // Real-shape attention bench on random Q/K/V (no weights): attention alone, local partial alone, local partial
    // with a concurrent exchange of the full message, exchange alone. Both ranks call it together (world 2).
    struct AttnBench {
        double attn_ms = 0, local_ms = 0, local_with_exchange_ms = 0, exchange_ms = 0, overlapped_wall_ms = 0, finish_wait_ms = 0;
        double local_slowdown_ms = 0;  // median over repetitions of (local with exchange - local alone) of the SAME repetition
    };
    AttnBench bench_attention(unsigned reps);

    // Test evidence for the failure injection: the hook fired (on the block asked for), and the last block whose
    // exchange this rank started (-1 = none; world 1: the last block entered).
    bool injection_fired() const;
    std::int64_t last_started_block() const;

    struct Impl;

private:
    DeviceRunResult run_blocks(const ForwardTables& t, std::uint32_t basis, std::uint32_t nblocks, const DeviceRunOptions& opt, bool download_x);
    AttnBench bench_attention_impl(unsigned reps);
    ForwardResult forward_impl(const block::PackedLayout& L, const block::ForwardPlan& p, const block::StreamInputs& in, const ForwardTables& t,
        std::uint32_t nblocks, const DeviceRunOptions& opt);
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::h3::dist
