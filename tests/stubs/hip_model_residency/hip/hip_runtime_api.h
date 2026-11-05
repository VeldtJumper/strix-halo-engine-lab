#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

using hipError_t = int;
inline constexpr hipError_t hipSuccess = 0, hipErrorInvalidValue = 1, hipErrorOutOfMemory = 2;
inline constexpr unsigned hipHostRegisterMapped = 1, hipExtHostRegisterCoarseGrained = 2;
inline constexpr unsigned hipHostMallocDefault = 0, hipDeviceMallocFinegrained = 1, hipDeviceMallocUncached = 2;
inline constexpr int hipMemcpyHostToDevice = 1;
using hipStream_t = void*;
inline const hipStream_t hipStreamPerThread = reinterpret_cast<void*>(std::uintptr_t{2});
inline const hipStream_t hipStreamLegacy = reinterpret_cast<void*>(std::uintptr_t{1});
enum hipStreamCaptureStatus { hipStreamCaptureStatusNone };
struct hipUUID { unsigned char bytes[16]; };
struct hipDeviceProp_t { char gcnArchName[256]; };

namespace residency_mock {
inline bool fail_unregister = false;
inline void (*event)(char) = nullptr;
inline void record(char value) { if (event) event(value); }
}

inline const char* hipGetErrorString(hipError_t) { return "mock runtime error"; }
inline hipError_t hipGetDevice(int* device) { *device = 0; return hipSuccess; }
inline hipError_t hipSetDevice(int) { return hipSuccess; }
inline hipError_t hipRuntimeGetVersion(int* version) { *version = 1; return hipSuccess; }
inline hipError_t hipDriverGetVersion(int* version) { *version = 1; return hipSuccess; }
inline hipError_t hipDeviceGetUuid(hipUUID* uuid, int) { *uuid = {{1}}; return hipSuccess; }
inline hipError_t hipGetDeviceProperties(hipDeviceProp_t* properties, int) {
    *properties = {"mock"};
    return hipSuccess;
}
inline hipError_t hipDeviceSynchronize() { residency_mock::record('D'); return hipSuccess; }
inline hipError_t hipStreamIsCapturing(hipStream_t, hipStreamCaptureStatus* status) {
    *status = hipStreamCaptureStatusNone;
    return hipSuccess;
}
inline hipError_t hipHostRegister(void*, std::size_t, unsigned) {
    residency_mock::record('G');
    return hipSuccess;
}
inline hipError_t hipHostGetDevicePointer(void** device, void* host, unsigned) {
    *device = host;
    return hipSuccess;
}
inline hipError_t hipHostUnregister(void*) {
    residency_mock::record('U');
    return residency_mock::fail_unregister ? hipErrorInvalidValue : hipSuccess;
}
// Copy routes are outside this fixture; an unexpected call must fail the child.
inline hipError_t hipMalloc(void**, std::size_t) { std::abort(); }
inline hipError_t hipExtMallocWithFlags(void**, std::size_t, unsigned) { std::abort(); }
inline hipError_t hipHostMalloc(void**, std::size_t, unsigned) { std::abort(); }
inline hipError_t hipMemcpy(void*, const void*, std::size_t, int) { std::abort(); }
inline hipError_t hipFree(void*) { std::abort(); }
inline hipError_t hipHostFree(void*) { std::abort(); }
