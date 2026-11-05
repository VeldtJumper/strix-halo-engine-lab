#pragma once
// Refuse a device allocation that would not fit in dedicated VRAM. On Windows (WDDM) the driver's system-memory
// fallback lets cudaMalloc succeed past the card's memory and the work then runs from host RAM over PCIe (pro524:
// 524K prefill at 333 tok/s instead of ~3,500). The guard makes that a clean refusal instead.
//   STRIX_QRTX_VRAM_HEADROOM_MIB (default 256): free VRAM that must remain after the allocation
//   STRIX_QRTX_ALLOW_SYSMEM=1: no check (the old behaviour)
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strix::cuda {

inline bool vram_guard_enabled() {
    static const bool on = [] { const char* v = std::getenv("STRIX_QRTX_ALLOW_SYSMEM"); return !(v && v[0] == '1'); }();
    return on;
}

inline std::uint64_t vram_headroom_bytes() {
    static const std::uint64_t bytes = [] {
        const char* v = std::getenv("STRIX_QRTX_VRAM_HEADROOM_MIB");
        return (v && v[0] ? std::strtoull(v, nullptr, 10) : 256ULL) << 20U;
    }();
    return bytes;
}

// Throws std::runtime_error when `bytes` plus the headroom exceed the device's free memory.
inline void require_vram(std::uint64_t bytes, const char* what) {
    if (!vram_guard_enabled()) return;
    std::size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess) throw std::runtime_error("cudaMemGetInfo failed");
    if (bytes + vram_headroom_bytes() > free)
        throw std::runtime_error(std::string("VRAM full: ") + what + " needs " + std::to_string(bytes >> 20U) + " MiB, " +
                                 std::to_string(free >> 20U) + " MiB free of " + std::to_string(total >> 20U) +
                                 " MiB (headroom " + std::to_string(vram_headroom_bytes() >> 20U) +
                                 " MiB); refusing instead of spilling into system memory (STRIX_QRTX_ALLOW_SYSMEM=1 overrides)");
}

}  // namespace strix::cuda
