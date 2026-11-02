// STRIX_QPF2K_HC=1 (fast prefill profile, default off): faster forms of the prefill HC row kernels, bitwise equal
// to the ones they replace (same FP32 operations on the same values in the same order).
//
// Parts (STRIX_QPF2K_HC_MASK, default all):
//  bit 0  lane-chain output: the coefficients scale * code of the up weights are formed once per call by a pack kernel
//         (the same FP32 product the output kernel forms in the loop) and read as one float4 per rank step and lane;
//         the rows' low values (SGPR operands) are double-buffered so a scalar-load wait covers the previous step only.
//         Same chains (k ascending, gate = fma(low, coefficient, gate) via hc_wide2_output_step) and the same epilogue.
//  bit 1  inject + HC norm: an inject rows step followed by the HC step that reads its streams runs as one kernel that
//         writes the streams and the normalized rows (no second read of the streams).
//  bit 2  MoE combine + inject + HC norm: the same with the combine before the inject (combined row also written).
//  bit 3  low projection (WMMA): the same kernel with the next-but-one stage's global loads in a second register set.
// STRIX_QPF2K_HC_OUTPUT3=1 (with STRIX_QPF2K_HC=1, any mask; wins over bit 0): lane-chain output v3, the base kernel's
//         arithmetic with the low values prefetched by vector loads, a 16-step rank loop and a pipelined epilogue.
// STRIX_QPF2K_HC_CHECK=1: every replaced launch also runs the original kernels and compares every written byte.

unsigned qpf2k_hc_mask() {
    static const unsigned mask = [] {
        const char* on = std::getenv("STRIX_QPF2K_HC");
        if (!on || on[0] != '1') return 0U;
        const char* m = std::getenv("STRIX_QPF2K_HC_MASK");
        return m ? static_cast<unsigned>(std::strtoul(m, nullptr, 0)) : 0xffU;
    }();
    return qpf2k::suppressed(qpf2k::Item::hc) ? 0U : mask;
}
bool qpf2k_hc_check() {
    static const bool on = [] { const char* v = std::getenv("STRIX_QPF2K_HC_CHECK"); return v && v[0] == '1'; }();
    return on;
}

// Check mode: `rows` rows of `bytes` bytes, one `stride` apart, copied to the host after the stream drains.
std::vector<unsigned char> qpf2k_snapshot_rows(const void* base, std::uint64_t stride, std::uint64_t bytes, std::uint32_t rows,
    hipStream_t stream) {
    std::vector<unsigned char> out(bytes * rows);
    check_hip(hipStreamSynchronize(stream), "qpf2k HC check synchronize");
    check_hip(hipMemcpy2D(out.data(), bytes, base, stride, bytes, rows, hipMemcpyDeviceToHost), "qpf2k HC check copy");
    return out;
}
void qpf2k_compare(const char* what, const std::vector<unsigned char>& reference, const std::vector<unsigned char>& candidate,
    std::uint32_t rows) {
    std::uint64_t differ = 0, first = UINT64_MAX;
    for (std::uint64_t i = 0; i < reference.size(); ++i)
        if (reference[i] != candidate[i]) { ++differ; if (first == UINT64_MAX) first = i; }
    std::fprintf(stderr, "qpf2k HC check %s: rows %u bytes %llu differ %llu first %lld\n", what, rows,
        static_cast<unsigned long long>(reference.size()), static_cast<unsigned long long>(differ),
        first == UINT64_MAX ? -1LL : static_cast<long long>(first));
    if (differ || reference.size() != candidate.size()) throw std::runtime_error(std::string("qpf2k HC check: ") + what + " differs");
}

// ---- bit 0: lane-chain output over packed coefficients ----
// coef[k * hidden + f] = (c_0, c_1, c_2, c_3), c_s = scale_s * float(code_s): stream s's up row s * hidden + f, Q8_0
// block k / 32, code k % 32. The output kernel forms scale[s] * (float(code ^ 0x80) - 128.0F), the same exact integer
// value times the same scale, rounded once: equal bits.
__global__ __launch_bounds__(256) void qpf2k_hc_pack_coef_kernel(const unsigned char* __restrict__ up, float4* __restrict__ coef,
    std::uint32_t hidden) {
    const unsigned index = blockIdx.x * 256U + threadIdx.x;
    if (index >= hidden * 10U) return;
    const unsigned f = index % hidden, b = index / hidden;
    const unsigned char* block[4];
    float scale[4];
#pragma unroll
    for (unsigned s = 0; s < 4U; ++s) {
        block[s] = up + (static_cast<std::uint64_t>(s) * hidden + f) * 340U + b * 34U;
        scale[s] = __half2float(__ushort_as_half(static_cast<unsigned short>(block[s][0] | (block[s][1] << 8U))));
    }
#pragma unroll 4
    for (unsigned k = 0; k < 32U; ++k) {
        float c[4];
#pragma unroll
        for (unsigned s = 0; s < 4U; ++s) c[s] = scale[s] * static_cast<float>(static_cast<signed char>(block[s][2U + k]));
        coef[static_cast<std::uint64_t>(b * 32U + k) * hidden + f] = make_float4(c[0], c[1], c[2], c[3]);
    }
}
// One coefficient buffer per stream (13.1 MB; launches on one stream are ordered).
float4* qpf2k_hc_coef_scratch(hipStream_t stream) {
    static std::mutex mutex;
    static std::map<hipStream_t, float4*> pools;
    const std::lock_guard lock(mutex);
    auto& p = pools[stream];
    if (!p) {
        void* q = nullptr;
        if (hipMalloc(&q, std::uint64_t{320U} * 2560U * sizeof(float4)) != hipSuccess)
            throw std::runtime_error("qpf2k HC: coefficient scratch allocation failed");
        p = static_cast<float4*>(q);
    }
    return p;
}

// 32 rows per wave, lane = feature; block = 4 waves (128 features) of one 32-row group (row groups fastest in the grid,
// as wide2_hyper_output_kernel<Storage, 32, true>). Each rank step: wait for its 32 low values (requested one step
// earlier), request the next step's, run the step on the coefficients loaded A steps earlier.
// Blocked (v2, the engine's form): the coefficients of a block of A rank steps are loaded together into one register
// buffer (ext_vector_type, not float4 structs) at the start of the previous block and copied to the working buffer at
// the block boundary: one vmcnt wait per A steps for loads issued A steps earlier (the base kernel's structure for its
// Q8 codes). v1 (Blocked false: a ring of A per-step look-ahead loads, skipped past the end) and the clamped ring made
// the compiler rotate the ring through copies behind a full vmcnt drain every iteration (t2c: v1 6.98 ms vs 4.60 ms;
// a clamped ring of 8 also spilled SGPRs). The chains, operands and epilogue are the same in every form.
constexpr unsigned kQpf2kCoefAhead = 8U;
template <typename Storage, unsigned A = kQpf2kCoefAhead, bool Blocked = true>
__global__ __launch_bounds__(128) void qpf2k_hc_output_kernel(HyperWideRows rows, const float4* __restrict__ coef, std::uint32_t hidden) {
    constexpr unsigned R = 32U, S = kWideStreams, K = 320U;
    static_assert(K % A == 0U && A % 2U == 0U);
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U, row0 = blockIdx.x * R;
    const std::uint64_t feature = (static_cast<std::uint64_t>(blockIdx.y) * 4U + wave) * 32U + lane;
    using ConstantFloat = const __attribute__((address_space(4))) float;
    auto* low_t = reinterpret_cast<ConstantFloat*>(reinterpret_cast<std::uintptr_t>(rows.low_t + row0));
    const std::uint32_t pitch = rows.low_t_pitch;
    const float4* cp = coef + feature;
    float g[S * R];
#pragma unroll
    for (unsigned i = 0; i < S * R; ++i) g[i] = 0.0F;
    typedef float qf4 __attribute__((ext_vector_type(4)));
    const auto gload = [&](unsigned k) {
        return *reinterpret_cast<const __attribute__((address_space(1))) qf4*>(
            reinterpret_cast<std::uintptr_t>(cp + static_cast<std::uint64_t>(k) * hidden));
    };
    if constexpr (Blocked) {
        float la[R], lb[R];
#pragma unroll
        for (unsigned t = 0; t < R; ++t) la[t] = low_t[t];
        qf4 nx[A];
#pragma unroll
        for (unsigned j = 0; j < A; ++j) nx[j] = gload(j);
        for (unsigned k0 = 0; k0 < K; k0 += A) {
            qf4 cu[A];
#pragma unroll
            for (unsigned j = 0; j < A; ++j) cu[j] = nx[j];
            // Unconditional: the last block reloads itself (never used), so the loop body has no branch.
            const unsigned kn = k0 + A < K ? k0 + A : K - A;
#pragma unroll
            for (unsigned j = 0; j < A; ++j) nx[j] = gload(kn + j);
#pragma unroll
            for (unsigned j = 0; j < A; j += 2U) {
#pragma unroll
                for (unsigned h = 0; h < 2U; ++h) {
                    const unsigned k = k0 + j + h;
                    float (&cur)[R] = h ? lb : la;
                    float (&next)[R] = h ? la : lb;
                    // lgkmcnt(0): the current step's low values (the only scalar loads in flight).
                    __builtin_amdgcn_s_waitcnt(0xfc07);
                    const unsigned kl = k + 1U < K ? k + 1U : K - 1U;
#pragma unroll
                    for (unsigned t = 0; t < R; ++t) next[t] = low_t[static_cast<std::uint64_t>(kl) * pitch + t];
                    const qf4 v = cu[j + h];
                    hc_wide2_output_step(g, v.x, v.y, v.z, v.w, cur);
                }
            }
        }
    } else {
    float4 c[A];
#pragma unroll
    for (unsigned a = 0; a < A; ++a) c[a] = cp[static_cast<std::uint64_t>(a) * hidden];
    float la[R], lb[R];
#pragma unroll
    for (unsigned t = 0; t < R; ++t) la[t] = low_t[t];
    const auto step = [&](unsigned k, float (&cur)[R], float (&next)[R], unsigned slot) {
        // lgkmcnt(0): the current step's low values (the only scalar loads in flight).
        __builtin_amdgcn_s_waitcnt(0xfc07);
        if (k + 1U < K) {
#pragma unroll
            for (unsigned t = 0; t < R; ++t) next[t] = low_t[static_cast<std::uint64_t>(k + 1U) * pitch + t];
        }
        const float4 v = c[slot];
        if (k + A < K) c[slot] = cp[static_cast<std::uint64_t>(k + A) * hidden];
        hc_wide2_output_step(g, v.x, v.y, v.z, v.w, cur);
    };
    for (unsigned k = 0; k < K; k += A) {
#pragma unroll
        for (unsigned j = 0; j < A; j += 2U) {
            step(k + j, la, lb, j);
            step(k + j + 1U, lb, la, j + 1U);
        }
    }
    }
#pragma unroll
    for (unsigned t = 0; t < R; ++t) {
        if (row0 + t >= rows.count) break;
        float mixed = 0.0F;
        const auto* normalized = wide_row(rows.normalized, rows.normalized_stride, row0 + t);
#pragma unroll
        for (unsigned s = 0; s < S; ++s)
            mixed += normalized[static_cast<std::uint64_t>(s) * hidden + feature] * sigmoid(g[4U * t + s]);
        store_activation(&wide_row(static_cast<Storage*>(rows.output), rows.output_stride, row0 + t)[feature],
            mixed / static_cast<float>(S));
    }
}

// ---- output v3 (STRIX_QPF2K_HC=1 + STRIX_QPF2K_HC_OUTPUT3=1, opt-in, independent of the mask) ----
// wide2_hyper_output_kernel<Storage, 32, true> with the same value for everything it computes: the Q8 code words (same
// 36-byte fetch one block ahead, same alignbyte / xor), the coefficient RN(scale * (float(byte) - 128)), the low values as
// SGPR operands (two rank steps per scalar-load wait, same addresses), hc_wide2_output_step (k ascending), and the
// epilogue's expression on the same normalized values. Only the schedule and the instruction selection change:
//  (1) low values prefetched A blocks (of 32 rank steps) ahead by vector loads: at the start of block b, two global loads
//      per wave touch the 32 x 128 bytes the scalar loads of block b + A will read (lane i: step 32 (b + A) + i / 2 (+16 for
//      the second load), byte 64 (i % 2) of the row group's 128-byte segment), so those scalar loads find the lines in
//      the shader array's cache / L2 instead of MALL/DRAM. t1 (base): 64 % of wave time in s_waitcnt, ~800-1,100 cycles
//      per two-step scalar round trip at 5 waves/SIMD. The prefetch registers are consumed (empty asm use) A blocks after
//      their load, so the compiler's in-order vmcnt waits for them are free and no load is unaccounted. (Rev 2's per-step
//      ring cost two VALU address adds per step and made the block-end vmcnt(0) wait for a one-step-old load.)
//  (2) the rank loop runs U (16 or 32) steps per iteration with static code-word indices (no per-4-step rotation), and the
//      byte goes to FP32 with v_cvt_f32_ubyteN (asm: the compiler otherwise forms int(byte) - 128 and v_cvt_f32_i32, four
//      issue slots per coefficient; float(byte) - 128 is the same exact integer, then the same single-rounding multiply).
//  (3) epilogue: the normalized values of rows t + P are loaded (global, clamped row index, unconditional) while row t is
//      mixed; the base issued each row's four loads after the previous row's store (32 serialized round trips per wave).
// A = 0: no prefetch, P = 0: the base epilogue (attribution forms).
template <unsigned B> __device__ __forceinline__ float qpf2k_cvt_ubyte(unsigned w) {
    float f;
    if constexpr (B == 0U) asm("v_cvt_f32_ubyte0 %0, %1" : "=v"(f) : "v"(w));
    else if constexpr (B == 1U) asm("v_cvt_f32_ubyte1 %0, %1" : "=v"(f) : "v"(w));
    else if constexpr (B == 2U) asm("v_cvt_f32_ubyte2 %0, %1" : "=v"(f) : "v"(w));
    else asm("v_cvt_f32_ubyte3 %0, %1" : "=v"(f) : "v"(w));
    return f;
}
__device__ __forceinline__ float qpf2k_cvt_ubyte(unsigned w, unsigned b) {
    switch (b) {  // b is a constant after unrolling
    case 0U: return qpf2k_cvt_ubyte<0U>(w);
    case 1U: return qpf2k_cvt_ubyte<1U>(w);
    case 2U: return qpf2k_cvt_ubyte<2U>(w);
    default: return qpf2k_cvt_ubyte<3U>(w);
    }
}
bool qpf2k_hc_output3() {
    static const bool on = [] { const char* v = std::getenv("STRIX_QPF2K_HC_OUTPUT3"); return v && v[0] == '1'; }();
    return on && !qpf2k::suppressed(qpf2k::Item::hc_output3) && qpf2k_hc_mask() != 0U;
}
// STRIX_QPF2K_HC_OUTPUT3_BAND (0 default, 1..8, 12, 16, 24, 32, 48, 64): CTA permutation only; identical chains/epilogue.
constexpr bool qpf2k_hc_output3_band_valid(unsigned band) {
    return band <= 8U || band == 12U || band == 16U || band == 24U || band == 32U || band == 48U || band == 64U;
}
unsigned qpf2k_hc_output3_band() {
    static const unsigned band = [] {
        const char* v = std::getenv("STRIX_QPF2K_HC_OUTPUT3_BAND");
        if (!v || !*v) return 0U;
        unsigned value = 0U;
        for (; *v; ++v) {
            if (*v < '0' || *v > '9') return 0U;
            value = value * 10U + static_cast<unsigned>(*v - '0');
            if (value > 64U) return 0U;  // Bound every step, including arbitrarily long inputs.
        }
        return qpf2k_hc_output3_band_valid(value) ? value : 0U;
    }();
    return qpf2k::suppressed(qpf2k::Item::hc_output3_band) ? 0U : band;
}
// STRIX_QPF2K_HC_LOWBAND (decoder default 3 through qpf2k defaults; decimal 0..3, anything else = 0), with the qpf2k projection (mask bit 3) and output v3
// band 16 (the defaults); same values, same chains, only where the low values are stored:
//  bit 0  the projection writes low_t in 512-row bands, low_t[(r / 512) * 320 * 512 + k * 512 + r % 512]: the output's rank
//         stride is 2 KiB instead of the padded row count times 4 bytes (32 KiB at 8192 rows). A band-16 CTA's 32 rows lie in
//         one 512-row band (row0 % 32 == 0), so each step's 32 values stay contiguous (one 128-byte segment) as before.
//  bit 1  the projection skips the FP16 low copy (low16), read only by the WMMA output (STRIX_HCW_OUTPUT=2) and check mode.
// Declines (old layout and stores) under STRIX_QPF2K_HC_CHECK, FP16 normalized rows, norm-free rows, other projection or
// output forms. Calls of at most 512 padded rows keep the old low_t layout (bit 0 only: one band, the layouts coincide up to
// the pitch); bit 1 still applies to them.
constexpr std::uint32_t kQpf2kLowBandRows = 512U;
unsigned qpf2k_hc_lowband() {
    static const unsigned mode = [] {
        const char* v = std::getenv("STRIX_QPF2K_HC_LOWBAND");
        if (!v || !v[0] || v[1] || v[0] < '0' || v[0] > '3') return 0U;
        return static_cast<unsigned>(v[0] - '0');
    }();
    return !qpf2k::suppressed(qpf2k::Item::hc_lowband) && qpf2k_hc_mask() != 0U ? mode : 0U;
}
__device__ __forceinline__ std::uint64_t qpf2k_low_band_index(std::uint32_t k, std::uint32_t r) {
    return static_cast<std::uint64_t>(r / kQpf2kLowBandRows) * (kHcwRank * kQpf2kLowBandRows) + k * kQpf2kLowBandRows +
        r % kQpf2kLowBandRows;
}
// Bulk scalar-cache prefetch (Q > 0): one s_load_b32 per 64-byte line of the low values of Q rank steps, into a dummy SGPR
// that is consumed only after the next compiler-inserted lgkmcnt(0) (which waits for every scalar load in flight); s[100:101]
// is a clobbered scratch pair for the running step address. The values are never used.
#define QPF2K_SQC_STEP "s_load_b32 %0, s[100:101], 0x0\n\t" "s_load_b32 %0, s[100:101], 0x40\n\t" "s_add_u32 s100, s100, %3\n\t" "s_addc_u32 s101, s101, 0\n\t"

// In the engine the HC up weights sit in the fine-grained registered model range (MTYPE UC, hip_model_residency.cpp D-028), so
// each of the 256 row groups of an 8192-row call re-reads its feature block's weights from DRAM (943 MB per call) instead of
// L2 (t2 trace: base 5,346 us vs v3 5,292 us per 8192-row call in-engine, while the cached-weight bench gives 4.4 vs 4.1 ms).
// STRIX_QPF2K_HC_UPCOPY (default on with STRIX_QPF2K_HC_OUTPUT3): one device byte copy of the site's 3.5 MB of up weights per
// call into a cached per-stream scratch, read by the output kernel; the bytes are identical.
// STRIX_QPF2K_HC_UPCOPY_BASE=1 (with STRIX_QPF2K_HC=1): the base lane-chain output reads the copy too (attribution arm).
bool qpf2k_hc_upcopy() {
    static const bool on = [] { const char* v = std::getenv("STRIX_QPF2K_HC_UPCOPY"); return !v || v[0] != '0'; }();
    return on;
}
bool qpf2k_hc_upcopy_base() {
    static const bool on = [] { const char* v = std::getenv("STRIX_QPF2K_HC_UPCOPY_BASE"); return v && v[0] == '1'; }();
    return on && qpf2k_hc_mask() != 0U;
}
__global__ __launch_bounds__(256) void qpf2k_hc_up_copy_kernel(const uint4* __restrict__ from, uint4* __restrict__ to, std::uint64_t n) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * 256U + threadIdx.x;
    if (i < n) to[i] = from[i];
}
// The copy is a kernel on the call's stream (no copy-engine path or host staging); the per-stream scratch is reused by the
// next call only after this call's output kernel (in-order stream). Declines (returns the original pointer): during stream
// capture (no allocation there), for unaligned sizes/pointers, and below kQpf2kUpCopyMinRows rows (t2 trace: at 32 rows v3
// alone already beats the base, 55 vs 79 us, and a 3.5 MB copy would cost about as much as the call).
constexpr std::uint32_t kQpf2kUpCopyMinRows = 256U;
const void* qpf2k_hc_up_copy(const void* up, std::uint64_t bytes, std::uint32_t rows, hipStream_t stream) {
    if (rows < kQpf2kUpCopyMinRows || bytes % 16U || reinterpret_cast<std::uintptr_t>(up) % 16U) return up;
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(stream, &capture) != hipSuccess || capture != hipStreamCaptureStatusNone) return up;
    static std::mutex mutex;
    static std::map<hipStream_t, std::pair<void*, std::uint64_t>> pools;
    void* copy = nullptr;
    {
        const std::lock_guard lock(mutex);
        auto& pool = pools[stream];
        // The copy is an optional optimization (default ON with v3): no scratch (allocation failed once, or a larger site) =
        // decline and read the original weights, never fail the call.
        if (!pool.first && !pool.second) {
            if (hipMalloc(&pool.first, bytes) != hipSuccess) { (void)hipGetLastError(); pool.first = nullptr; pool.second = 1U; return up; }
            pool.second = bytes;
        }
        if (!pool.first || pool.second < bytes) return up;
        copy = pool.first;
    }
    const std::uint64_t n = bytes / 16U;
    hipLaunchKernelGGL(qpf2k_hc_up_copy_kernel, dim3(static_cast<unsigned>((n + 255U) / 256U)), dim3(256U), 0, stream,
        static_cast<const uint4*>(up), static_cast<uint4*>(copy), n);
    check_hip(hipGetLastError(), "qpf2k HC up weight copy launch");
    return copy;
}
// Shared literal body preserves the default kernel signature and launch; only the banded instances remap CTAs.
template <typename Storage, unsigned U = 16U, unsigned A = 1U, unsigned P = 8U, unsigned Q = 0U>
__global__ __launch_bounds__(128) void qpf2k_hc_output3_kernel(HyperWideRows rows, const void* __restrict__ up, std::uint32_t hidden) {
    constexpr unsigned R = 32U, S = kWideStreams, kBlocks = 10U, kRowBytes = kBlocks * 34U, Steps = 2U;
    static_assert(U == 16U || U == 32U);
    static_assert(A <= 2U);
    static_assert(P <= R);
    static_assert(Q == 0U || Q == 8U || Q == 16U);
    static_assert(Q == 0U || U % Q == 0U);
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U, row0 = blockIdx.x * R;
    const std::uint64_t feature = (static_cast<std::uint64_t>(blockIdx.y) * kWide2OutputWaves + wave) * 32U + lane;
    const unsigned low0 = row0;
#include "hip_qpf2k_hc_output3_body.inc"
}

// LowBand (STRIX_QPF2K_HC_LOWBAND bit 0): the CTA's low values start at its row group's offset in the banded table (rank
// pitch rows.low_t_pitch = 512); the body's loads are otherwise unchanged.
template <typename Storage, unsigned Band, bool LowBand = false>
__global__ __launch_bounds__(128) void qpf2k_hc_output3_band_kernel(HyperWideRows rows, const void* __restrict__ up, std::uint32_t hidden) {
    constexpr unsigned U = 16U, A = 1U, P = 8U, Q = 0U;
    constexpr unsigned R = 32U, S = kWideStreams, kBlocks = 10U, kRowBytes = kBlocks * 34U, Steps = 2U;
    static_assert(Band != 0U && qpf2k_hc_output3_band_valid(Band));
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    const unsigned groups = gridDim.x, features = gridDim.y, cta = blockIdx.x + groups * blockIdx.y;
    const unsigned base = cta / (Band * features) * Band, count = min(Band, groups - base), local = cta - base * features;
    // Within each band, visit its row groups for one feature tile, then the next feature tile (tail uses count).
    const unsigned row0 = (base + local % count) * R;
    const std::uint64_t feature = (static_cast<std::uint64_t>(local / count) * kWide2OutputWaves + wave) * 32U + lane;
    const unsigned low0 = LowBand ? static_cast<unsigned>(qpf2k_low_band_index(0U, row0)) : row0;
#include "hip_qpf2k_hc_output3_body.inc"
}


// ---- bit 3: HC low projection with the injection weights in the register prefetch ----
// hcw_projection_kernel<false, false> (FP32 normalized rows) with the same block shape (256 rows x 64 low outputs,
// 8 waves of 32 rows x 64 outputs = 2 x 4 WMMA tiles), the same operands (FP16 RNE of the FP32 normalized values, the
// packed FP16 down weights), the same per-accumulator WMMA order (stage ascending, h = 0, 1, tiles c = 0..3), the same
// injection chains (per row, 8 partials over k = 32 stage + 4 sub .. + 3 ascending, then the xor tree) and the same
// epilogue. Only the schedule changes: the global loads of stage s + Depth (rows, weights, injection weights) are
// issued right after stage s's barrier, so the injection weights no longer stall each stage.
// Depth (v2 = 1): the stage's loads (rows, weights AND injection weights) are issued one stage ahead, after the
// barrier, as the base kernel does for rows and weights; the base kernel loaded the injection weights inside the stage
// and waited for them at once (global_load_b128 + s_waitcnt vmcnt(0) per stage). Depth 2 (v1) kept two stages in
// registers; its 80 VGPRs made the compiler serialize the LDS fragment loads (ds_load pair, lgkmcnt(0), 2 WMMA)
// (t2c: 3.67 ms vs 2.94 ms).
// LowMode (STRIX_QPF2K_HC_LOWBAND): bit 0 low_t in 512-row bands, bit 1 no low16 stores; the stored values are unchanged.
template <unsigned Depth, unsigned LowMode = 0U>
__global__ __launch_bounds__(256) void qpf2k_hc_projection_kernel(HyperWideRows rows, const _Float16* __restrict__ pack,
    const float* __restrict__ inject, unsigned streams, _Float16* __restrict__ low16) {
    __shared__ __align__(16) _Float16 bs[2][kHcwProjCols * kHcwPitch];
    __shared__ __align__(16) _Float16 as[8][32U * kHcwPitch];
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U, cb = blockIdx.x, sub = lane % 8U, rq = lane / 8U;
    const unsigned row0 = blockIdx.y * kHcwProjRows + wave * 32U;
    const float* xq[8];
#pragma unroll
    for (unsigned q = 0; q < 8U; ++q) xq[q] = wide_row(rows.normalized, rows.normalized_stride, min(row0 + 4U * q + rq, rows.count - 1U)) + 4U * sub;
    unsigned valid = 0U;
#pragma unroll
    for (unsigned q = 0; q < 8U; ++q) valid |= (row0 + 4U * q + rq < rows.count ? 1U : 0U) << q;
    const bool injecting = inject && cb < streams;
    const float* w = injecting ? inject + static_cast<std::uint64_t>(cb) * kHcwWidth + 4U * sub : inject;
    const auto* bsrc = reinterpret_cast<const uint4*>(pack + static_cast<std::uint64_t>(cb) * kHcwStages * kHcwProjCols * 32U) + threadIdx.x;
    const unsigned bdst = (threadIdx.x / 4U) * kHcwPitch + (threadIdx.x % 4U) * 8U;
    struct Stage { float4 xa[8]; uint4 bw; float4 c; };
    // Global (not flat) loads: flat loads also count in LGKM, so every LDS fragment wait would drain them.
    typedef float qf4 __attribute__((ext_vector_type(4)));
    typedef unsigned qu4 __attribute__((ext_vector_type(4)));
    const auto gf4 = [](const void* p) {
        const qf4 v = *reinterpret_cast<const __attribute__((address_space(1))) qf4*>(reinterpret_cast<std::uintptr_t>(p));
        return make_float4(v.x, v.y, v.z, v.w);
    };
    const auto load = [&](Stage& d, unsigned stage) {
        const qu4 b = *reinterpret_cast<const __attribute__((address_space(1))) qu4*>(
            reinterpret_cast<std::uintptr_t>(bsrc + stage * (kHcwProjCols * 32U / 8U)));
        d.bw = make_uint4(b.x, b.y, b.z, b.w);
        // Unconditional loads (a row past the end reads the last row; its WMMA rows and injection are never stored), so
        // the waits before the stores can count the newer stage's loads still in flight.
#pragma unroll
        for (unsigned q = 0; q < 8U; ++q) d.xa[q] = gf4(xq[q] + stage * 32U);
        if (injecting) d.c = gf4(w + stage * 32U);
    };
    hcw_v8f acc[2][4];
#pragma unroll
    for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
        for (unsigned c = 0; c < 4U; ++c) acc[t][c] = hcw_v8f{};
    float injection[8];
#pragma unroll
    for (unsigned q = 0; q < 8U; ++q) injection[q] = 0.0F;
    static_assert(Depth == 1U || Depth == 2U);
    Stage sa, sb;
    load(sa, 0U);
    if constexpr (Depth == 2U) load(sb, 1U);
    const auto body = [&](Stage& cur, unsigned stage) {
        const unsigned buffer = stage & 1U;
        *reinterpret_cast<uint4*>(&bs[buffer][bdst]) = cur.bw;
#pragma unroll
        for (unsigned q = 0; q < 8U; ++q) {
            _Float16 h[4] = {static_cast<_Float16>(cur.xa[q].x), static_cast<_Float16>(cur.xa[q].y), static_cast<_Float16>(cur.xa[q].z),
                static_cast<_Float16>(cur.xa[q].w)};
            uint2 word;
            __builtin_memcpy(&word, h, 8U);
            *reinterpret_cast<uint2*>(as[wave] + (4U * q + rq) * kHcwPitch + 4U * sub) = word;
        }
        if (injecting) {
#pragma unroll
            for (unsigned q = 0; q < 8U; ++q) {
                injection[q] = fmaf(cur.xa[q].x, cur.c.x, injection[q]);
                injection[q] = fmaf(cur.xa[q].y, cur.c.y, injection[q]);
                injection[q] = fmaf(cur.xa[q].z, cur.c.z, injection[q]);
                injection[q] = fmaf(cur.xa[q].w, cur.c.w, injection[q]);
            }
        }
        // LDS-scoped barrier (release / acquire on the local address space only): __syncthreads()'s full workgroup fence
        // would also wait for every global load in flight, i.e. for the prefetched stages.
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
        // Unconditional (the last Depth stages reload the final stage, never used): the waits stay countable.
        load(cur, stage + Depth < kHcwStages ? stage + Depth : kHcwStages - 1U);
#pragma unroll
        for (unsigned h = 0; h < 2U; ++h) {
            const hcw_v16h a0 = hcw_lds_fragment(as[wave] + (lane % 16U) * kHcwPitch + 16U * h);
            const hcw_v16h a1 = hcw_lds_fragment(as[wave] + (16U + lane % 16U) * kHcwPitch + 16U * h);
#pragma unroll
            for (unsigned c = 0; c < 4U; ++c) {
                const hcw_v16h b = hcw_lds_fragment(&bs[buffer][(16U * c + lane % 16U) * kHcwPitch + 16U * h]);
                acc[0][c] = hcw_wmma(a0, b, acc[0][c]);
                acc[1][c] = hcw_wmma(a1, b, acc[1][c]);
            }
        }
    };
    if constexpr (Depth == 2U) {
        for (unsigned stage = 0; stage < kHcwStages; stage += 2U) {
            body(sa, stage);
            body(sb, stage + 1U);
        }
    } else {
        for (unsigned stage = 0; stage < kHcwStages; ++stage) body(sa, stage);
    }
    const float divisor = static_cast<float>(streams);
#pragma unroll
    for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
        for (unsigned c = 0; c < 4U; ++c)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const unsigned r = row0 + 16U * t + 2U * e + lane / 16U, col = cb * kHcwProjCols + 16U * c + lane % 16U;
                if (r >= rows.count) continue;
                const float scaled = acc[t][c][e] / divisor, low = scaled * sigmoid(scaled);
                wide_row(rows.normalized, rows.normalized_stride, r)[kHcwWidth + col] = low;
                if constexpr (!(LowMode & 2U)) low16[static_cast<std::uint64_t>(r) * kHcwRank + col] = static_cast<_Float16>(low);
                if constexpr ((LowMode & 1U) != 0U) rows.low_t[qpf2k_low_band_index(col, r)] = low;
                else if (rows.low_t) rows.low_t[static_cast<std::uint64_t>(col) * rows.low_t_pitch + r] = low;
            }
    if (!injecting) return;
#pragma unroll
    for (unsigned q = 0; q < 8U; ++q) {
        float v = injection[q];
        v = v + wide2_partner<1U>(v);
        v = v + wide2_partner<2U>(v);
        v = v + wide2_partner<4U>(v);
        if (sub == 0U && (valid >> q & 1U))
            wide_row(rows.injection, rows.injection_stride, row0 + 4U * q + rq)[cb] = 2.0F * sigmoid(v / divisor);
    }
}

// ---- bits 1, 2: inject (+ MoE combine) + HC norm over uniformly strided prefill rows ----
// Row r of every view at base + r * stride. Block = one row, wave s = stream s. Lane t owns the norm leaves (chains)
// 8 t .. 8 t + 7 of rows_prepare_hyper_norm_kernel (its thread l sums features l, l + 256, ... in order): chunk c of
// 256 features is one 16-byte load per lane (features 256 c + 8 t .. + 7), so lane t's eight chains run in the
// original feature order. The leaf tree (r[l] += r[l + stride], stride 128 .. 1) pairs lanes t, t + 16 / 8 / 4 / 2 / 1
// at the same slot, then slots j, j + 4 / 2 / 1 inside lane 0. Inverse and normalized values as the norm kernel.
// Inject: stored = bf16(fma(hidden, gate, hyper)) (the inject kernel's v_fmac); the norm reads the stored value.
// Combine: hidden = bf16(shared + routed) (the combine kernel's value), stored to the combined row by wave 0.
struct Qpf2kInjectNormRows {
    const std::uint16_t* hyper; std::uint64_t hyper_stride;
    const std::uint16_t* hidden; std::uint64_t hidden_stride;   // inject hidden row, or the combine's shared input
    const std::uint16_t* routed; std::uint64_t routed_stride;   // combine only
    const float* gates; std::uint64_t gates_stride;
    std::uint16_t* combined; std::uint64_t combined_stride;     // combine only
    std::uint16_t* streams; std::uint64_t streams_stride;       // inject output = HC input
    float* normalized; std::uint64_t normalized_stride;
    const float* norm;  // FP32 norm weights [4][2560] (ggml type 0; other types decline the fold)
    std::uint32_t count, hidden_size;
    float epsilon;
};
template <typename T> __device__ __forceinline__ T* qpf2k_row(T* base, std::uint64_t stride, std::uint32_t row) {
    return reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(base) + static_cast<std::uint64_t>(row) * stride);
}
__device__ __forceinline__ float qpf2k_bf16(std::uint32_t word, unsigned half) {
    return __uint_as_float(half ? word & 0xffff0000U : word << 16U);
}
__device__ __forceinline__ std::uint32_t qpf2k_round_bf16(float value) {
    const auto bits = __float_as_uint(value);
    return (bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U;
}
// Operand-pinned FP32 operations: qpf2k_add / qpf2k_mul / qpf2k_fmac (hip_qpf2k_pinned.inc, with the measured
// operand order of each base kernel).
template <unsigned D> __device__ __forceinline__ float qpf2k_from_lane(float value) {
    return __shfl_down(value, D, 32);
}
template <bool Combine>
__global__ __launch_bounds__(128) void qpf2k_inject_norm_kernel(Qpf2kInjectNormRows a) {
    constexpr unsigned H = kInjectNormHidden, C = H / 256U;
    const unsigned row = blockIdx.x, s = threadIdx.x / 32U, t = threadIdx.x % 32U;
    const auto* hyper = qpf2k_row(a.hyper, a.hyper_stride, row) + s * H + 8U * t;
    const auto* hidden = qpf2k_row(a.hidden, a.hidden_stride, row) + 8U * t;
    auto* streams = qpf2k_row(a.streams, a.streams_stride, row) + s * H + 8U * t;
    auto* normalized = qpf2k_row(a.normalized, a.normalized_stride, row) + s * H + 8U * t;
    const float gate = qpf2k_row(a.gates, a.gates_stride, row)[s];
    uint4 h[C];
    if constexpr (Combine) {
        // Every wave forms the combined row from the shared and routed inputs (the combine kernel's FP32 sum, rounded
        // once); wave 0 stores it after the barrier, since the combined row may replace its own row's shared or
        // routed input in place.
        const auto* routed = qpf2k_row(a.routed, a.routed_stride, row) + 8U * t;
#pragma unroll
        for (unsigned c = 0; c < C; ++c) {
            const uint4 x = *reinterpret_cast<const uint4*>(hidden + 256U * c), q = *reinterpret_cast<const uint4*>(routed + 256U * c);
            const std::uint32_t xw[4] = {x.x, x.y, x.z, x.w}, qw[4] = {q.x, q.y, q.z, q.w};
            std::uint32_t o[4];
#pragma unroll
            for (unsigned w = 0; w < 4U; ++w)
                o[w] = qpf2k_round_bf16(qpf2k_add(qpf2k_bf16(xw[w], 0U), qpf2k_bf16(qw[w], 0U))) |
                    (qpf2k_round_bf16(qpf2k_add(qpf2k_bf16(xw[w], 1U), qpf2k_bf16(qw[w], 1U))) << 16U);
            h[c] = make_uint4(o[0], o[1], o[2], o[3]);
        }
        __syncthreads();
        if (s == 0U) {
            auto* combined = qpf2k_row(a.combined, a.combined_stride, row) + 8U * t;
#pragma unroll
            for (unsigned c = 0; c < C; ++c) *reinterpret_cast<uint4*>(combined + 256U * c) = h[c];
        }
    } else {
#pragma unroll
        for (unsigned c = 0; c < C; ++c) h[c] = *reinterpret_cast<const uint4*>(hidden + 256U * c);
    }
    uint4 stored[C];
    float sum[8];
#pragma unroll
    for (unsigned j = 0; j < 8U; ++j) sum[j] = 0.0F;
#pragma unroll
    for (unsigned c = 0; c < C; ++c) {
        const uint4 x = *reinterpret_cast<const uint4*>(hyper + 256U * c);
        const std::uint32_t xw[4] = {x.x, x.y, x.z, x.w}, hw[4] = {h[c].x, h[c].y, h[c].z, h[c].w};
        std::uint32_t o[4];
#pragma unroll
        for (unsigned w = 0; w < 4U; ++w) {
            const std::uint32_t lo = qpf2k_round_bf16(qpf2k_fmac(qpf2k_bf16(xw[w], 0U), gate, qpf2k_bf16(hw[w], 0U)));
            const std::uint32_t hi = qpf2k_round_bf16(qpf2k_fmac(qpf2k_bf16(xw[w], 1U), gate, qpf2k_bf16(hw[w], 1U)));
            o[w] = lo | (hi << 16U);
            const float vlo = qpf2k_bf16(o[w], 0U), vhi = qpf2k_bf16(o[w], 1U);
            sum[2U * w] = qpf2k_fmac(sum[2U * w], vlo, vlo);
            sum[2U * w + 1U] = qpf2k_fmac(sum[2U * w + 1U], vhi, vhi);
        }
        stored[c] = make_uint4(o[0], o[1], o[2], o[3]);
        *reinterpret_cast<uint4*>(streams + 256U * c) = stored[c];
    }
    // r[v] = r[v + stride] + r[v] (the norm kernel's operand order), v = 8 t + j.
#pragma unroll
    for (unsigned j = 0; j < 8U; ++j) {
        float v = sum[j];
        v = qpf2k_add(qpf2k_from_lane<16U>(v), v);
        v = qpf2k_add(qpf2k_from_lane<8U>(v), v);
        v = qpf2k_add(qpf2k_from_lane<4U>(v), v);
        v = qpf2k_add(qpf2k_from_lane<2U>(v), v);
        v = qpf2k_add(qpf2k_from_lane<1U>(v), v);
        sum[j] = v;
    }
#pragma unroll
    for (unsigned j = 0; j < 4U; ++j) sum[j] = qpf2k_add(sum[j + 4U], sum[j]);
#pragma unroll
    for (unsigned j = 0; j < 2U; ++j) sum[j] = qpf2k_add(sum[j + 2U], sum[j]);
    const float total = __shfl(qpf2k_add(sum[1], sum[0]), 0, 32);
    const float inverse = rsqrtf(total / static_cast<float>(a.hidden_size) + a.epsilon);
    const float* weight = a.norm + s * H + 8U * t;
#pragma unroll
    for (unsigned c = 0; c < C; ++c) {
        const std::uint32_t w[4] = {stored[c].x, stored[c].y, stored[c].z, stored[c].w};
        const float4 w0 = *reinterpret_cast<const float4*>(weight + 256U * c), w1 = *reinterpret_cast<const float4*>(weight + 256U * c + 4U);
        const float nw[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        float n[8];
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) n[j] = qpf2k_mul(qpf2k_mul(inverse, qpf2k_bf16(w[j / 2U], j % 2U)), nw[j]);
        *reinterpret_cast<float4*>(normalized + 256U * c) = make_float4(n[0], n[1], n[2], n[3]);
        *reinterpret_cast<float4*>(normalized + 256U * c + 4U) = make_float4(n[4], n[5], n[6], n[7]);
    }
}
