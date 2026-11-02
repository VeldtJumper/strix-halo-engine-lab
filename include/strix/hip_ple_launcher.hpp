#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/hip_host_upload_backend.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/model_contract.hpp"
#include "strix/ple_lookup.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace strix {

struct HipPleLauncherConfig {
    ModelContract model =
        ModelContract::qwen38_flash_next();
    HipActivationStorage activation =
        HipActivationStorage::bf16;
    // Default keeps the existing synchronous copy and creates no upload pool.
    // Opt-in bundles require this fixed additional pinned-host admission budget.
    static constexpr std::uint64_t owned_event_upload_maximum_bytes =
        PleHashContract::maximum_heads * 512;
    bool enable_owned_event_upload = false;
    // Captured path uses a resident BF16 table and stable device n-gram IO.
    // File-backed lookup and event-upload paths remain eager only.
    bool enable_resident_graph = false;
    // An injected pool is an explicit alternative and outlives the launcher.
    HostUploadPool* upload_pool = nullptr;
};

void register_hip_ple_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipPleLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification =
        kernels::QualificationStatus::experimental);

[[nodiscard]] ExecutionLauncher
make_hip_ple_launcher(
    std::string kernel_id,
    HipPleLauncherConfig* config);

class HipPleLauncherBundle {
public:
    explicit HipPleLauncherBundle(
        HipPleLauncherConfig config);
    ~HipPleLauncherBundle();

    HipPleLauncherBundle(
        const HipPleLauncherBundle&) = delete;
    HipPleLauncherBundle& operator=(
        const HipPleLauncherBundle&) = delete;
    HipPleLauncherBundle(
        HipPleLauncherBundle&&) = delete;
    HipPleLauncherBundle& operator=(
        HipPleLauncherBundle&&) = delete;

    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry);

    [[nodiscard]] std::uint64_t owned_upload_admission_bytes() const noexcept {
        return uploads_ ? HipPleLauncherConfig::owned_event_upload_maximum_bytes : 0;
    }

private:
    struct OwnedUploads;
    std::unique_ptr<OwnedUploads> uploads_;
    HipPleLauncherConfig config_;
};

// Known-window PLE for T consecutive rows. The tokens of a verify window are known before it
// starts, so every row's n-gram lookup is resolved on the host into pinned staging that the GPU
// reads in place (unified memory: no copy and no synchronisation inside the window). The host
// n-gram history is not advanced by launch(); the PLE convolution state after each row is kept,
// and commit(n_keep) restores row n_keep-1's convolution state and publishes its history.
// STRIX_PLE_WINDOW_WARM (default on; =0, STRIX_QPF2K_DEFAULTS=0 or STRIX_BASELINE=1 off; effective only with a
// rolling-residency decode turn): the MTP draft loop reports the next verify window's
// input tokens as the host learns them (begin: row 0, the pending token; draft(k): row k + 1, draft k). Their PLE
// table pages are loaded into the decode read cache in the background; the window's lookups and bytes are unchanged.
namespace ple_window_warm {
[[nodiscard]] bool enabled() noexcept;
void begin(std::uint32_t token) noexcept;
void draft(std::uint32_t step, std::uint32_t token) noexcept;
}  // namespace ple_window_warm

class HipPleKnownWindow {
public:
    // prefill: up to known_prefill_max_rows (16384) rows that are always all kept; launch() publishes the last row's
    // history itself and keeps no convolution snapshots (commit() is not used).
    HipPleKnownWindow(const ModelContract& model, HipActivationStorage activation, std::uint32_t max_rows,
        bool prefill = false, const ExecutionRuntimeResources* prime_resources = nullptr);
    ~HipPleKnownWindow();
    HipPleKnownWindow(const HipPleKnownWindow&) = delete;
    HipPleKnownWindow& operator=(const HipPleKnownWindow&) = delete;
    void preflight(std::span<const ExecutionDispatchContext> rows) const;
    // Stage a successor before its predecessor launches; an occupied stage is
    // retained until its slot is selected. Never used for reclaim windows.
    void stage_serve(std::span<const ExecutionDispatchContext> rows);
    void launch(std::span<const ExecutionDispatchContext> rows);
    // Enqueues on `stream` (a hipStream_t); the host history is written immediately.
    void commit(std::uint32_t n_keep, void* stream);
    [[nodiscard]] std::uint32_t last_rows() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

namespace testing {
// Raw PLE Q8 projection (generic block-per-row or the wave8 kernel with the same
// 256-leaf summation tree); width % 32 == 0 for wave8. Bit-exactness fixture only.
void enqueue_ple_q8_projection(bool wave8,const std::uint16_t* input,const void* q8,std::uint32_t input_width,
    std::uint32_t output_width,float* output,void* stream);
// PLE tail (key/query gate, gated value, grouped norm, causal convolution and residual) of `rows`
// consecutive rows. keys: rows x hidden*hc FP32 (overwritten); values: rows x hidden FP32; input and
// output: rows of hidden*hc BF16 at their element strides; conv_state: hidden*hc x history BF16,
// advanced by all rows; norms and convolution F32, or BF16 when bf16_weights. rows_form selects
// the prefill rows kernels, otherwise the per-row kernels run row by row. scratch: rows*hc +
// 2*hidden*hc floats. Bit-exactness fixture only.
void enqueue_ple_tail(bool rows_form,float* keys,const float* values,const std::uint16_t* input,std::uint32_t input_stride,
    std::uint16_t* output,std::uint32_t output_stride,std::uint16_t* conv_state,const void* norm_key,const void* norm_query,
    const void* norm_conv,const void* conv,bool bf16_weights,std::uint32_t rows,std::uint32_t hidden_size,std::uint32_t hc_count,
    std::uint32_t kernel_size,std::uint32_t ngram_size,float epsilon,float* scratch,void* stream);
}
}  // namespace strix
