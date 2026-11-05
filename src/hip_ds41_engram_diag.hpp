// Test-only diagnostics for the Engram prefill projection's exact fused multiply-add route. Defined only when the layer
// library is built with BUILD_TESTING (STRIX_DS41_ENGRAM_DIAG); not part of the public headers.
#pragma once

#include "strix/hip_ds41_layer.hpp"

namespace strix::ds41 {
// Whether every x of a supported view is in the fused form's exact-product set (own flag, synchronous).
[[nodiscard]] bool engram_f16_pf_x_exact(const F16RowsView& view);
// The fused 48 x 64 form without the scan: not exact for x outside that set.
void enqueue_engram_f16_pf_fused_unchecked(const F16RowsView& view, void* stream);
// The production flag of this stream after its queued work (0 = fused form ran, 1 = separate), or -1 if the stream never
// took the scanned route (separate form only).
[[nodiscard]] int engram_f16_pf_stream_flag(void* stream);
}  // namespace strix::ds41
