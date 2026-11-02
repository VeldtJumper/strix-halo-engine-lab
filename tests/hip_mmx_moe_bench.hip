// minimax-m3 src/minimax/moe/ kernels: bitwise tests against the current kernels and ABA microbenchmark.
//
// A (current decode path): strix_gemv of strix-trellis 0.3 for every EXL3 linear (input transform, split GEMV core, output
//   transform), the fused post-exchange kernel with the router, and for the MoE the G4 chain per expert in ascending id
//   (gather, gate, up, swiglu, down, scatter_add fmaf) + shared MLP + add, routes known on the host (no D2H sync).
// B (new): strix::minimax::moe::opt (enqueue_moe after the post-exchange kernel without router; enqueue_linears).
//
// usage: strix-hip-mmx-moe-bench [--test] [--bench] [--reps N] [--iters N] [--only SUBSTR] [--model DIR --layer L --rank R]
//   Without --model the weights are synthetic (random trellis words, random FP16 scales; same bits contract).
//   --test: bitwise A == B per row for N = 1..8 (MoE: random, shared-route and mixed batches; every dense site), and
//           batch invariance (a row's bits alone == inside N).
//   --bench: per cell (site x N) the order A B A B A B A (ABA x3), each a mean over --iters calls on rotating cold
//           weights; prints us per call, GB/s of weight bytes, and B/A per triplet.
//   --diverse: decode-only production enqueue_moe v4 cell; --test exhausts N=1..8 and feasible U/S,
//           IL/SINGLE=0/1 x DOWN_SPLIT=0/1/2 and every row vs standalone (DOWN 0/2);
//           --bench: B8 U=4,8,12,16,20,22,24,25,26,27,28,30,32; B4 U=4,8,12,16; B1 U=4.
//   --diverse-fallback --test: separate wide/K5/W8/Depth8+12 negative-reader gate (not singleton qualification).
//           --singletons min|max|COUNT controls singleton expert count. Stage probes follow total timing.
//           --ab-down adds, per cell, one-process DOWN_SPLIT=0 (A) vs 1 (B) brackets A B B A (x reps, iters each).
#include "minimax/engine/route_identity.hpp"
#include "minimax/moe/mmx_moe.hpp"
#include "minimax/moe/mmx_prefill.hpp"
#include "minimax/moe/mmx_prefill_core.hpp"
#include "minimax/moe/mmx_trellis_device.hpp"
#include "minimax_gate_common.hpp"
#include "strix/hip_minimax_moe.hpp"
#include "strix/minimax_moe.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace opt = strix::minimax::moe::opt;
namespace mref = strix::minimax::moe;

constexpr unsigned H = opt::kHidden, IR = opt::kInter;

// ---------------------------------------------------------------- weights (host bytes -> library + raw device)
struct HostExl3 {
    std::vector<std::uint32_t> trellis;
    std::vector<std::uint16_t> suh, svh;
    unsigned input = 0, output = 0, bits = 0;
    std::uint64_t bytes() const { return trellis.size() * 4ULL; }
};

std::uint64_t splitmix(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
std::uint16_t f16_bits(float f) {
    const _Float16 h = static_cast<_Float16>(f);
    std::uint16_t b;
    std::memcpy(&b, &h, 2);
    return b;
}
// Exhaustive GPU counterpart to the independent CPU codebook reference below.
__global__ void pipe_codebook_check_kernel(std::uint32_t* result) {
    const unsigned state = blockIdx.x * blockDim.x + threadIdx.x;
    if (state >= 65536U) return;
    const auto old_bits = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(opt::pfk::codebook_mul1(state)));
    const auto new_bits = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(opt::pfk::pipe_codebook(state)));
    result[state * 2U] = static_cast<std::uint32_t>(old_bits) | (static_cast<std::uint32_t>(new_bits) << 16U);
    result[state * 2U + 1U] = opt::pfk::pipe_product(state);
}

HostExl3 synthetic(unsigned input, unsigned output, unsigned bits, std::uint64_t seed) {
    HostExl3 h;
    h.input = input;
    h.output = output;
    h.bits = bits;
    h.trellis.resize(static_cast<std::size_t>(input / 16) * (output / 16) * bits * 8);
    std::uint64_t s = seed;
    for (std::size_t i = 0; i < h.trellis.size(); i += 2) {
        const std::uint64_t v = splitmix(s);
        h.trellis[i] = static_cast<std::uint32_t>(v);
        if (i + 1 < h.trellis.size()) h.trellis[i + 1] = static_cast<std::uint32_t>(v >> 32);
    }
    auto scale = [&](std::vector<std::uint16_t>& v, unsigned n, float mag) {
        v.resize(n);
        for (unsigned i = 0; i < n; ++i) {
            const std::uint64_t r = splitmix(s);
            const float m = mag * (0.5F + static_cast<float>(r & 0xffff) / 65536.0F);
            v[i] = f16_bits((r >> 16) & 1 ? -m : m);
        }
    };
    scale(h.suh, input, 0.06F);
    scale(h.svh, output, 0.03F);
    return h;
}
// Real slices (same slicing as tests/hip_minimax_g4.hip): output slice [o0, o0 + n) / input slice [i0, i0 + n).
HostExl3 real_slice(Pack& pack, const std::string& prefix, bool out_slice, std::uint64_t first, std::uint64_t n) {
    const Tensor tr = pack.get(prefix + ".trellis"), suh = pack.get(prefix + ".suh"), svh = pack.get(prefix + ".svh");
    HostExl3 h;
    h.bits = static_cast<unsigned>(tr.shape[2] / 16);
    const std::uint64_t in_tiles = tr.shape[0], out_tiles = tr.shape[1], w = tr.shape[2] * 2;  // bytes per tile
    auto copy16 = [](std::vector<std::uint16_t>& v, const std::uint8_t* p, std::uint64_t count) {
        v.resize(count);
        std::memcpy(v.data(), p, count * 2);
    };
    if (out_slice) {
        h.input = static_cast<unsigned>(suh.shape[0]);
        h.output = static_cast<unsigned>(n);
        h.trellis.resize(in_tiles * (n / 16) * w / 4);
        auto* dst = reinterpret_cast<std::uint8_t*>(h.trellis.data());
        for (std::uint64_t i = 0; i < in_tiles; ++i)
            std::memcpy(dst + i * (n / 16) * w, tr.data + (i * out_tiles + first / 16) * w, (n / 16) * w);
        copy16(h.suh, suh.data, suh.shape[0]);
        copy16(h.svh, svh.data + first * 2, n);
    } else {
        h.input = static_cast<unsigned>(n);
        h.output = static_cast<unsigned>(svh.shape[0]);
        const std::uint64_t row_bytes = out_tiles * w;
        h.trellis.resize((n / 16) * row_bytes / 4);
        std::memcpy(h.trellis.data(), tr.data + (first / 16) * row_bytes, (n / 16) * row_bytes);
        copy16(h.suh, suh.data + first * 2, n);
        copy16(h.svh, svh.data, svh.shape[0]);
    }
    return h;
}

class Lib {
public:
    Lib() {
        strix_context_desc d{};
        d.struct_size = sizeof(d);
        d.abi_version = STRIX_TRELLIS_ABI_VERSION;
        d.device_ordinal = 0;
        lib_ok(strix_context_create(&d, &ctx_), "context");
        st_.struct_size = sizeof(st_);
        st_.abi_version = STRIX_TRELLIS_ABI_VERSION;
        st_.hip_stream = nullptr;
    }
    ~Lib() {
        (void)hipDeviceSynchronize();
        for (auto w : owned_) (void)strix_weights_destroy(w);
        if (ws_) (void)hipFree(ws_);
        (void)strix_context_destroy(ctx_);
    }
    strix_weights prepare(const HostExl3& h) {
        strix_exl3_parts_desc p{};
        p.struct_size = sizeof(p);
        p.abi_version = STRIX_TRELLIS_ABI_VERSION;
        p.transport_version = STRIX_TRELLIS_TRANSPORT_VERSION;
        p.bits_x2 = 2 * h.bits;
        p.rows = h.output;
        p.columns = h.input;
        p.codebook = STRIX_CODEBOOK_MUL1;
        p.marker = STRIX_EXL3_MARKER_MUL1;
        p.trellis = {h.trellis.data(), h.trellis.size() * 4};
        p.suh = {h.suh.data(), h.suh.size() * 2};
        p.svh = {h.svh.data(), h.svh.size() * 2};
        strix_weights w = nullptr;
        lib_ok(strix_weights_prepare_parts(ctx_, &p, &st_, &w), "prepare");
        owned_.push_back(w);
        return w;
    }
    // The query validates the tensor descriptors too: pass real device buffers of at least rows x width floats.
    void reserve(strix_weights w, unsigned input, unsigned output, std::uint64_t rows, float* in, float* out) {
        strix_matmul_desc d = desc(w, input, output, in, out, rows);
        std::uint64_t bytes = 0;
        lib_ok(strix_matmul_workspace_bytes(ctx_, &d, &bytes), "workspace");
        if (bytes > ws_bytes_) {
            hip_ok(hipDeviceSynchronize(), "sync");
            if (ws_) hip_ok(hipFree(ws_), "free");
            hip_ok(hipMalloc(&ws_, bytes), "workspace");
            ws_bytes_ = bytes;
        }
    }
    // strix_gemm: the prefill identity (WMMA core at every row count); grows the workspace when needed.
    void gemm_wmma(strix_weights w, unsigned input, unsigned output, const float* in, float* out, std::uint64_t rows) {
        if (rows == 0) return;
        reserve(w, input, output, rows, const_cast<float*>(in), out);
        strix_matmul_desc d = desc(w, input, output, const_cast<float*>(in), out, rows);
        d.workspace = ws_;
        d.workspace_bytes = ws_bytes_;
        lib_ok(strix_gemm(ctx_, &d, &st_), "gemm");
    }
    void gemm(strix_weights w, unsigned input, unsigned output, const float* in, float* out, std::uint64_t rows) {
        if (rows == 0) return;
        strix_matmul_desc d = desc(w, input, output, const_cast<float*>(in), out, rows);
        d.workspace = ws_;
        d.workspace_bytes = ws_bytes_;
        lib_ok(strix_gemv(ctx_, &d, &st_), "gemv");  // the decode identity (strix_gemm would take the WMMA core)
    }

private:
    static strix_matmul_desc desc(strix_weights w, unsigned input, unsigned output, float* in, float* out, std::uint64_t rows) {
        strix_matmul_desc d{};
        strix_matmul_desc_init(&d);
        d.weights = w;
        set(d.input, in, rows, input);
        set(d.output, out, rows, output);
        return d;
    }
    static void set(strix_tensor_desc& t, float* p, std::uint64_t rows, std::uint64_t width) {
        t.data = p;
        t.buffer_bytes = rows * width * sizeof(float);
        t.dtype = STRIX_DTYPE_F32;
        t.rank = 2;
        t.shape[0] = rows;
        t.shape[1] = width;
        t.strides[0] = width;
        t.strides[1] = 1;
    }
    strix_context ctx_ = nullptr;
    strix_stream st_{};
    void* ws_ = nullptr;
    std::uint64_t ws_bytes_ = 0;
    std::vector<strix_weights> owned_;
};

// Streaming read of n uint4 words (4 independent loads in flight per thread per step); the result only feeds a guard
// store so the loads are kept. The size-matched read ceiling for a weight matrix of the same bytes.
// gfx1151 WMMA compute ceiling: NACC independent 16x16x16 chains per wave on register data; every
// chain has its own runtime operands and seed (no merging), every lane stores its sum (no dead code).
using CHalf16 = _Float16 __attribute__((ext_vector_type(16)));
using CFloat8 = float __attribute__((ext_vector_type(8)));
using CInt4 = int __attribute__((ext_vector_type(4)));
using CInt8 = int __attribute__((ext_vector_type(8)));
template <int NACC>
__global__ __launch_bounds__(128) void wmma_f16_ceiling_kernel(int iters, float seed, float* out) {
    const int lane = static_cast<int>(threadIdx.x & 31U);
    CHalf16 a[NACC], b;
#pragma unroll
    for (int k = 0; k < 16; ++k) b[k] = static_cast<_Float16>(seed * static_cast<float>((lane * 3 + k + static_cast<int>(blockIdx.x)) & 7));
    CFloat8 acc[NACC];
#pragma unroll
    for (int j = 0; j < NACC; ++j) {
#pragma unroll
        for (int k = 0; k < 16; ++k) a[j][k] = static_cast<_Float16>(seed * static_cast<float>((lane + k + 5 * j) & 15));
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[j][e] = seed * static_cast<float>(j * 8 + e);
    }
    for (int i = 0; i < iters; ++i)
#pragma unroll
        for (int j = 0; j < NACC; ++j) acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[j], b, acc[j]);
    float s = 0.0F;
#pragma unroll
    for (int j = 0; j < NACC; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) s += acc[j][e];
    out[blockIdx.x * 128 + threadIdx.x] = s;
}
template <int NACC>
__global__ __launch_bounds__(128) void wmma_iu8_ceiling_kernel(int iters, int seed, int* out) {
    const int lane = static_cast<int>(threadIdx.x & 31U);
    CInt4 a[NACC], b;
#pragma unroll
    for (int k = 0; k < 4; ++k) b[k] = seed * ((lane * 3 + k + static_cast<int>(blockIdx.x)) & 3);
    CInt8 acc[NACC];
#pragma unroll
    for (int j = 0; j < NACC; ++j) {
#pragma unroll
        for (int k = 0; k < 4; ++k) a[j][k] = seed * ((lane + k + 5 * j) & 7) + j;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[j][e] = j * 8 + e;
    }
    for (int i = 0; i < iters; ++i)
#pragma unroll
        for (int j = 0; j < NACC; ++j) acc[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a[j], true, b, acc[j], false);
    int s = 0;
#pragma unroll
    for (int j = 0; j < NACC; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) s += acc[j][e];
    out[blockIdx.x * 128 + threadIdx.x] = s;
}
// The v2 core's full B-fragment build (window load from an L1/L2-resident trellis block, state extraction, mul1
// codebook, permlanex16, selects) without WMMA: the dequant ceiling per weight. 512 weights per wave per k-tile.
template <int Bits>
__global__ __launch_bounds__(128) void decode_ceiling_kernel(const std::uint32_t* trellis, int tiles_k, int iters, std::uint32_t* out) {
    namespace pk = strix::minimax::moe::opt::pfk;
    using W = pk::Window<Bits>;
    const int lane = static_cast<int>(threadIdx.x & 31U), wave = static_cast<int>(threadIdx.x >> 5U);
    const int c = lane & 15;
    const bool upper = lane >= 16, h = c >= 8;
    const int my_tile = 2 * (static_cast<int>(blockIdx.x % 4U) * 4 + wave) + (upper ? 1 : 0);  // 32 tile columns
    const std::uint32_t* tp = trellis + static_cast<std::size_t>(my_tile) * W::kWords;
    const int w0 = W::first_word(c);
    std::uint32_t x = 0;
    for (int i = 0; i < iters; ++i)
        for (int k = 0; k < tiles_k; ++k) {
            std::uint32_t wv[W::kNW], mine[8];
            pk::load_window<Bits>(tp + static_cast<std::size_t>(k) * 32 * W::kWords, w0, wv);
            pk::decode_fragment<Bits>(wv, h, mine);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const std::uint32_t other = __builtin_amdgcn_permlanex16(mine[j], mine[j], 0x76543210U, 0xfedcba98U, false, false);
                x ^= (upper ? other : mine[j]) + 3U * (upper ? mine[j] : other);
            }
        }
    out[blockIdx.x * 128 + threadIdx.x] = x;
}
__global__ __launch_bounds__(256) void read_kernel(const uint4* p, std::size_t n, unsigned* out) {
    unsigned acc = 0;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * 256U;
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * 256U + threadIdx.x;
    for (; i + 3 * stride < n; i += 4 * stride) {
        const uint4 a = p[i], b = p[i + stride], c = p[i + 2 * stride], d = p[i + 3 * stride];
        acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
    }
    for (; i < n; i += stride) {
        const uint4 a = p[i];
        acc ^= a.x ^ a.y ^ a.z ^ a.w;
    }
    if (acc == 0x9e3779b9U) out[0] = acc;
}

// Load-only replica of the GEMV core's weight stream (no decode, no math): chain c = (split, tile column) with
// consecutive chains = adjacent columns of one split; wave g runs chains g, g + P, g + 2P, ... (P = waves in the grid)
// with a Depth-deep ring of the lane's three words that runs across chain boundaries. P = all chains reproduces the
// v4 grid (one chain per wave); smaller P is the persistent form.
// Layout 0: the file layout (tile (k, col) at (k * cols + col) * tile bytes); 1: chain-contiguous (the tiles of one
// (split, column) chain consecutive); 2: CTA-interleaved (per (split, 4-column group): k-major, the group's 4 tiles of
// one k consecutive, so a CTA streams one contiguous region). Same bytes, only the placement differs.
template <int Bits, int Depth, int Layout = 0>
__global__ __launch_bounds__(128) void pattern_kernel(const std::uint32_t* trellis, unsigned output, unsigned input,
                                                      unsigned splits, unsigned* out) {
    namespace d = strix::minimax::moe::opt::dev;
    const unsigned lane = threadIdx.x & 31U;
    const unsigned cols = output / 16U, kt = input / 16U, len = kt / splits, total = cols * splits;
    const unsigned P = gridDim.x * 4U;
    const unsigned g = blockIdx.x * 4U + (threadIdx.x >> 5U);
    const d::LaneGeom<Bits> geo(lane);
    const std::size_t kstride = static_cast<std::size_t>(cols) * Bits * 32U;
    const auto* base = reinterpret_cast<const unsigned char*>(trellis);
    const unsigned chains = g < total ? (total - g + P - 1) / P : 0U, steps = chains * len;
    auto addr = [&](unsigned t) {
        const unsigned c = g + (t / len) * P, split = c / cols, col = c % cols, kk = t % len;
        constexpr std::size_t tb = Bits * 32U;
        if constexpr (Layout == 1) return base + ((static_cast<std::size_t>(split) * cols + col) * len + kk) * tb;
        if constexpr (Layout == 2)
            return base + (((static_cast<std::size_t>(split) * (cols / 4) + col / 4) * len + kk) * 4 + col % 4) * tb;
        return base + (static_cast<std::size_t>(split * len + kk)) * kstride + static_cast<std::size_t>(col) * tb;
    };
    std::uint32_t ra[Depth], rb[Depth], rc[Depth], acc = 0;
#pragma unroll
    for (int j = 0; j < Depth; ++j)
        if (static_cast<unsigned>(j) < steps) {
            const unsigned char* t = addr(static_cast<unsigned>(j));
            ra[j] = d::load_u32(t, geo.oa);
            rb[j] = d::load_u32(t, geo.ob);
            rc[j] = d::load_u32(t, geo.oc);
        }
    for (unsigned t0 = 0; t0 < steps; t0 += Depth) {
#pragma unroll
        for (int j = 0; j < Depth; ++j) {
            const unsigned t = t0 + static_cast<unsigned>(j);
            if (t < steps) {
                acc ^= ra[j] ^ rb[j] ^ rc[j];
                if (t + Depth < steps) {
                    const unsigned char* q = addr(t + Depth);
                    ra[j] = d::load_u32(q, geo.oa);
                    rb[j] = d::load_u32(q, geo.ob);
                    rc[j] = d::load_u32(q, geo.oc);
                }
            }
        }
    }
    if (acc == 0x9e3779b9U) out[0] = acc;
}

// Wide-load variants of the same stream (request-rate test). Mode 3: file layout, one dwordx2 load per tile by
// lanes < Bits * 4 (the whole tile in one instruction instead of three dword loads per lane). Mode 4: chain-contiguous
// layout, the wave reads its chain with dwordx4 loads (512 B per instruction).
template <int Bits, int Depth, int Mode>
__global__ __launch_bounds__(128) void pattern_wide_kernel(const std::uint32_t* trellis, unsigned output, unsigned input,
                                                           unsigned splits, unsigned* out) {
    const unsigned lane = threadIdx.x & 31U;
    const unsigned cols = output / 16U, kt = input / 16U, len = kt / splits, total = cols * splits;
    const unsigned P = gridDim.x * 4U;
    const unsigned g = blockIdx.x * 4U + (threadIdx.x >> 5U);
    constexpr std::size_t tb = Bits * 32U;
    const auto* base = reinterpret_cast<const unsigned char*>(trellis);
    const unsigned chains = g < total ? (total - g + P - 1) / P : 0U;
    std::uint32_t acc = 0;
    if constexpr (Mode == 3) {
        const unsigned steps = chains * len;
        const bool on = lane < Bits * 4U;
        auto addr = [&](unsigned t) {
            const unsigned c = g + (t / len) * P, split = c / cols, col = c % cols, kk = t % len;
            return reinterpret_cast<const uint2*>(base + (static_cast<std::size_t>(split * len + kk) * cols + col) * tb) + lane;
        };
        uint2 r[Depth];
#pragma unroll
        for (int j = 0; j < Depth; ++j)
            if (on && static_cast<unsigned>(j) < steps) r[j] = *addr(static_cast<unsigned>(j));
        for (unsigned t0 = 0; t0 < steps; t0 += Depth) {
#pragma unroll
            for (int j = 0; j < Depth; ++j) {
                const unsigned t = t0 + static_cast<unsigned>(j);
                if (on && t < steps) {
                    acc ^= r[j].x ^ r[j].y;
                    if (t + Depth < steps) r[j] = *addr(t + Depth);
                }
            }
        }
    } else {
        // chain bytes = len * tb; uint4 per lane per step: 512 B per wave instruction
        const unsigned per_chain = static_cast<unsigned>((len * tb + 511) / 512), steps = chains * per_chain;
        auto load = [&](unsigned t, uint4& v) {
            const unsigned c = g + (t / per_chain) * P, split = c / cols, col = c % cols, piece = t % per_chain;
            const std::size_t off = (static_cast<std::size_t>(split) * cols + col) * len * tb + piece * 512U + lane * 16U;
            if (piece * 512U + lane * 16U < len * tb) v = *reinterpret_cast<const uint4*>(base + off);
        };
        uint4 r[Depth];
#pragma unroll
        for (int j = 0; j < Depth; ++j) {
            r[j] = make_uint4(0, 0, 0, 0);
            if (static_cast<unsigned>(j) < steps) load(static_cast<unsigned>(j), r[j]);
        }
        for (unsigned t0 = 0; t0 < steps; t0 += Depth) {
#pragma unroll
            for (int j = 0; j < Depth; ++j) {
                const unsigned t = t0 + static_cast<unsigned>(j);
                if (t < steps) {
                    acc ^= r[j].x ^ r[j].y ^ r[j].z ^ r[j].w;
                    if (t + Depth < steps) load(t + Depth, r[j]);
                }
            }
        }
    }
    if (acc == 0x9e3779b9U) out[0] = acc;
}

// Block stream in the file layout (L5): CTA = C adjacent tile columns x one split; per k-row the C tiles are one
// contiguous run of C * tile bytes, read by the CTA's 128 threads with 16-byte loads (a Depth-deep ring over the
// CTA's whole region). No repack: the question is how long the contiguous runs must be.
template <int Bits, int C, int Depth>
__global__ __launch_bounds__(128) void pattern_block_kernel(const std::uint32_t* trellis, unsigned output, unsigned input,
                                                            unsigned splits, unsigned* out) {
    constexpr unsigned run16 = C * Bits * 32 / 16;  // 16-byte units per k-row run
    const unsigned cols = output / 16U, kt = input / 16U, len = kt / splits, groups = cols / C;
    const unsigned split = blockIdx.x / groups, c0 = (blockIdx.x % groups) * C;
    const auto* base = reinterpret_cast<const uint4*>(trellis);
    const unsigned units = len * run16;
    auto at = [&](unsigned e) {
        const unsigned k = split * len + e / run16, off = e % run16;
        return base + (static_cast<std::size_t>(k) * cols + c0) * (Bits * 2U) + off;  // tile = Bits * 2 uint4
    };
    std::uint32_t acc = 0;
    uint4 r[Depth];
#pragma unroll
    for (int j = 0; j < Depth; ++j) {
        const unsigned e = threadIdx.x + 128U * static_cast<unsigned>(j);
        r[j] = e < units ? *at(e) : make_uint4(0, 0, 0, 0);
    }
    for (unsigned i0 = 0; i0 * 128U < units; i0 += Depth) {
#pragma unroll
        for (int j = 0; j < Depth; ++j) {
            const unsigned e = threadIdx.x + 128U * (i0 + static_cast<unsigned>(j));
            if (e < units) {
                acc ^= r[j].x ^ r[j].y ^ r[j].z ^ r[j].w;
                const unsigned en = e + 128U * Depth;
                if (en < units) r[j] = *at(en);
            }
        }
    }
    if (acc == 0x9e3779b9U) out[0] = acc;
}

struct Mat {  // one matrix held twice: by the library (A) and as raw device buffers (B)
    strix_weights lib = nullptr;
    opt::Exl3 dev{};
    std::uint64_t bytes = 0;
};
Mat upload(Lib& lib, const HostExl3& h) {
    Mat m;
    m.lib = lib.prepare(h);
    m.dev.trellis = dupload(h.trellis);
    m.dev.suh = dupload(h.suh);
    m.dev.svh = dupload(h.svh);
    m.dev.input = h.input;
    m.dev.output = h.output;
    m.dev.bits = h.bits;
    m.bytes = h.bytes();
    return m;
}

// ---------------------------------------------------------------- model pieces of one MoE layer + dense sites
struct Layer {
    std::vector<Mat> gate, up, down;  // 128 experts
    Mat sg, su, sd;                   // shared
    std::vector<float> router, bias;
    float* router_d = nullptr;
    float* bias_d = nullptr;
    opt::ExpertSlices* experts_d = nullptr;
    opt::ExpertSlices shared{};
};
struct DenseSite {
    std::string name;
    std::vector<Mat> copies;  // rotating cold copies
    unsigned input = 0, output = 0;
};

struct Options {
    bool test = false, bench = false, no_moe = false;
    int reps = 3, iters = 40;
    std::string only, model;
    unsigned layer = 3, rank = 0;
    //: --routes FILE (u8 x 4 expert ids per token, qualification notes<L>.u8),
    // --tests SUBSTR (run only the test sections whose name contains it), --no-load (no weights: ceiling only),
    // --syn-bits B (bits of the synthetic routed experts, default 3).
    std::string routes, tests;
    bool no_load = false;
    unsigned syn_bits = 3;
    bool diverse = false;
    bool diverse_fallback = false;  // explicit negative-reader gate, never singleton qualification
    std::string singletons = "min";  // --singletons min|max|COUNT; infeasible cells are skipped
    bool ab_down = false;  // --ab-down: in-process DOWN_SPLIT=0 (A) vs 1 (B) per diverse cell, ABBA brackets
};

// Per-call switches are restored after the gate, including combined --test --bench invocations.
struct EnvMode {
    const char* name;
    bool had_value;
    std::string saved;
    explicit EnvMode(const char* key) : name(key), had_value(std::getenv(key) != nullptr),
                                       saved(had_value ? std::getenv(key) : "") {}
    void set(const char* value) const {
#ifdef _WIN32
        if (_putenv_s(name, value) != 0) throw std::runtime_error(std::string("set ") + name);
#else
        if (setenv(name, value, 1) != 0) throw std::runtime_error(std::string("set ") + name);
#endif
    }
    ~EnvMode() {
#ifdef _WIN32
        (void)_putenv_s(name, had_value ? saved.c_str() : "");
#else
        if (had_value) (void)setenv(name, saved.c_str(), 1);
        else (void)unsetenv(name);
#endif
    }
};

// Workspace comparison, including inactive bytes and reset counters, up to the pipeline's sequencing words (the route
// epoch advances on every call; the ticket must be back at 0). Avoid multi-GB host snapshots during
// the exhaustive gate: the reference remains a separate device allocation and only the mismatch bit comes back.
__global__ void workspace_equal_kernel(const std::uint32_t* a, const std::uint32_t* b, std::size_t words, unsigned* bad) {
    unsigned differs = 0;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < words;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        differs |= a[i] != b[i] ? 1U : 0U;
    if (differs != 0) atomicOr(bad, 1U);
}

struct WorkspaceCheck {
    std::uint32_t* reference;
    unsigned* bad;
    const std::size_t bytes = opt::moe_workspace_bytes();
    const std::size_t compared = opt::moe_pipe_offset();
    WorkspaceCheck() : reference(dalloc<std::uint32_t>(opt::moe_workspace_bytes() / sizeof(std::uint32_t))),
                       bad(dalloc<unsigned>(1)) {
        if (bytes % sizeof(std::uint32_t) != 0 || compared % sizeof(std::uint32_t) != 0) throw std::runtime_error("workspace alignment");
    }
    void save(const void* ws) const {
        hip_ok(hipMemcpy(reference, ws, bytes, hipMemcpyDeviceToDevice), "workspace snapshot");
    }
    bool equal(const void* ws) const {
        hip_ok(hipMemset(bad, 0, sizeof(unsigned)), "workspace compare reset");
        hipLaunchKernelGGL(workspace_equal_kernel, dim3(256), dim3(256), 0, nullptr,
                           static_cast<const std::uint32_t*>(ws), reference, compared / sizeof(std::uint32_t), bad);
        hip_ok(hipGetLastError(), "workspace compare");
        unsigned mismatch = 0, ticket = 0;
        hip_ok(hipMemcpy(&mismatch, bad, sizeof(mismatch), hipMemcpyDeviceToHost), "workspace compare result");
        hip_ok(hipMemcpy(&ticket, static_cast<const unsigned char*>(ws) + compared + sizeof(std::uint32_t), sizeof(ticket),
                         hipMemcpyDeviceToHost), "pipeline ticket");
        return mismatch == 0 && ticket == 0;
    }
    ~WorkspaceCheck() {
        (void)hipFree(bad);
        (void)hipFree(reference);
    }
};

void require_diverse_path(bool fallback = false) {
    if (fallback) return;
    for (const auto& flag : {std::pair<const char*, const char*>{"STRIX_MMX_MOE_V", "4"}, {"STRIX_MMX_MOE_W", "4"}, {"STRIX_MMX_MOE_DEPTH3", "4"}}) {
        const char* value = std::getenv(flag.first);
        if (value != nullptr && std::string(value) != flag.second)
            throw std::invalid_argument(std::string("diverse cell requires ") + flag.first + "=" + flag.second);
    }
}

// Bipartite degree construction: four distinct experts per row, exactly U experts and S singleton experts.
// Descending expert degrees, then the rows with the most capacity (Havel-Hakimi); independent of GPU routing.
std::vector<std::uint32_t> diverse_ids(unsigned n, unsigned u, unsigned singles, unsigned step) {
    if (n < 1 || n > 8 || u < 4 || u > 4 * n || singles > u) return {};
    std::vector<unsigned> degree(u, 2U), capacity(n, 4U), used(n, 0U);
    std::fill_n(degree.begin(), singles, 1U);
    unsigned total = 2 * u - singles;
    if (total > 4 * n) return {};
    for (unsigned e = singles; e < u && total < 4 * n; ++e)
        while (degree[e] < n && total < 4 * n) { ++degree[e]; ++total; }
    if (total != 4 * n || *std::max_element(degree.begin(), degree.end()) > n) return {};
    std::vector<unsigned> experts(u), rows(n);
    std::iota(experts.begin(), experts.end(), 0U);
    std::stable_sort(experts.begin(), experts.end(), [&](unsigned a, unsigned b) { return degree[a] > degree[b]; });
    std::vector<std::uint32_t> ids(std::size_t{n} * 4);
    for (unsigned e : experts) {
        std::iota(rows.begin(), rows.end(), 0U);
        std::stable_sort(rows.begin(), rows.end(), [&](unsigned a, unsigned b) { return capacity[a] > capacity[b]; });
        for (unsigned j = 0; j < degree[e]; ++j) {
            const unsigned r = rows[j];
            if (capacity[r] == 0) return {};
            ids[r * 4 + used[r]++] = (e * 37U + step * 17U) % opt::kExperts;
            --capacity[r];
        }
    }
    return ids;
}

// ---------------------------------------------------------------- A: the current MoE chain (G4), routes on the host
struct Scratch {
    float *xe, *ye, *g, *u, *rows_w;
    std::uint32_t* rows_i;
};
struct Routes {  // per expert ascending: rows and weights (G4 grouping)
    std::vector<std::vector<std::uint32_t>> rows;
    std::vector<std::vector<float>> wts;
    std::uint32_t* rows_d = nullptr;  // flattened on the device, per expert at offset[e]
    float* wts_d = nullptr;
    std::vector<std::size_t> offset;
};
Routes group_routes(const std::vector<std::uint32_t>& ids, const std::vector<float>& w4, unsigned n) {
    Routes r;
    r.rows.assign(opt::kExperts, {});
    r.wts.assign(opt::kExperts, {});
    for (unsigned t = 0; t < n; ++t)
        for (unsigned k = 0; k < opt::kTopE; ++k) {
            r.rows[ids[t * 4 + k]].push_back(t);
            r.wts[ids[t * 4 + k]].push_back(w4[t * 4 + k]);
        }
    std::vector<std::uint32_t> fr;
    std::vector<float> fw;
    r.offset.assign(opt::kExperts, 0);
    for (unsigned e = 0; e < opt::kExperts; ++e) {
        r.offset[e] = fr.size();
        fr.insert(fr.end(), r.rows[e].begin(), r.rows[e].end());
        fw.insert(fw.end(), r.wts[e].begin(), r.wts[e].end());
    }
    fr.push_back(0);
    fw.push_back(0.0F);
    r.rows_d = dupload(fr);
    r.wts_d = dupload(fw);
    return r;
}
void free_routes(Routes& r) {
    hip_ok(hipFree(r.rows_d), "free");
    hip_ok(hipFree(r.wts_d), "free");
}
void moe_chain_A(Lib& lib, Layer& L, const Routes& rt, const float* y, unsigned n, float* out, Scratch& s) {
    hip_ok(hipMemsetAsync(out, 0, std::size_t{n} * H * 4, nullptr), "zero");
    for (unsigned e = 0; e < opt::kExperts; ++e) {
        const std::size_t k = rt.rows[e].size();
        if (k == 0) continue;
        hipLaunchKernelGGL(gather_kernel, dim3(static_cast<unsigned>(k)), dim3(256), 0, nullptr, y, rt.rows_d + rt.offset[e], s.xe, H);
        lib.gemm(L.gate[e].lib, H, IR, s.xe, s.g, k);
        lib.gemm(L.up[e].lib, H, IR, s.xe, s.u, k);
        hipLaunchKernelGGL(swiglu_kernel, dim3(grid(k * IR)), dim3(256), 0, nullptr, s.g, s.u, s.g, k * IR);
        lib.gemm(L.down[e].lib, IR, H, s.g, s.ye, k);
        hipLaunchKernelGGL(scatter_add_kernel, dim3(static_cast<unsigned>(k)), dim3(256), 0, nullptr, s.ye,
                           rt.rows_d + rt.offset[e], rt.wts_d + rt.offset[e], out, H);
    }
    lib.gemm(L.sg.lib, H, IR, y, s.g, n);
    lib.gemm(L.su.lib, H, IR, y, s.u, n);
    hipLaunchKernelGGL(swiglu_kernel, dim3(grid(std::size_t{n} * IR)), dim3(256), 0, nullptr, s.g, s.u, s.g, std::size_t{n} * IR);
    lib.gemm(L.sd.lib, IR, H, s.g, s.xe, n);
    hipLaunchKernelGGL(add_kernel, dim3(grid(std::size_t{n} * H)), dim3(256), 0, nullptr, out, s.xe, std::size_t{n} * H);
}

// One decode batch of the MoE: residual inputs and the post-exchange outputs.
struct Batch {
    unsigned n = 0;
    float *h0, *p0, *p1, *nw;  // device inputs (h0 copied into h before each call)
    std::vector<std::uint32_t> ids;
    std::vector<float> w4;
    Routes routes;
};

int g_fail = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        ++g_fail;
        std::printf("FAIL: %s\n", what.c_str());
    }
}
template <class T> std::vector<T> d2h(const T* d, std::size_t n) {
    std::vector<T> h(n);
    hip_ok(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost), "d2h");
    return h;
}
bool same_bits(const std::vector<float>& a, const std::vector<float>& b, std::size_t* first = nullptr) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        std::uint32_t x, y;
        std::memcpy(&x, &a[i], 4);
        std::memcpy(&y, &b[i], 4);
        if (x != y) {
            if (first) *first = i;
            return false;
        }
    }
    return true;
}

struct Bench {
    Options o;
    Lib lib;
    Layer L;
    std::vector<DenseSite> dense;
    Scratch s{};
    float *h = nullptr, *y = nullptr, *outA = nullptr, *outB = nullptr, *xin = nullptr, *yA = nullptr, *yB = nullptr;
    std::uint32_t* idsA = nullptr;
    float* wA = nullptr;
    std::uint32_t* idsB = nullptr;
    float* wB = nullptr;
    void* moe_ws = nullptr;
    void* lin_ws = nullptr;
    std::mt19937_64 rng{12345};

    std::vector<float> randn(std::size_t n, float scale) {
        std::normal_distribution<float> nd(0.0F, scale);
        std::vector<float> v(n);
        for (auto& x : v) x = nd(rng);
        return v;
    }

    void load() {
        const auto t0 = std::chrono::steady_clock::now();
        std::unique_ptr<Pack> pack;
        if (!o.model.empty()) pack = std::make_unique<Pack>(o.model);
        const std::string lp = kP + "layers." + std::to_string(o.layer) + ".", b = lp + "block_sparse_moe.";
        const std::uint64_t r = o.rank;
        auto mk = [&](const std::string& prefix, bool out_slice, std::uint64_t first, std::uint64_t n, unsigned in_s,
                      unsigned out_s, unsigned bits, std::uint64_t seed) {
            return upload(lib, pack ? real_slice(*pack, prefix, out_slice, first, n) : synthetic(in_s, out_s, bits, seed));
        };
        for (unsigned e = 0; e < opt::kExperts; ++e) {
            const std::string ep = b + "experts." + std::to_string(e) + ".";
            L.gate.push_back(mk(ep + "w1", true, r * IR, IR, H, IR, o.syn_bits, 1000 + e * 3));
            L.up.push_back(mk(ep + "w3", true, r * IR, IR, H, IR, o.syn_bits, 1001 + e * 3));
            L.down.push_back(mk(ep + "w2", false, r * IR, IR, IR, H, o.syn_bits, 1002 + e * 3));
        }
        L.sg = mk(b + "shared_experts.gate_proj", true, r * IR, IR, H, IR, 5, 7);
        L.su = mk(b + "shared_experts.up_proj", true, r * IR, IR, H, IR, 5, 8);
        L.sd = mk(b + "shared_experts.down_proj", false, r * IR, IR, IR, H, 5, 9);
        if (pack) {
            L.router = pack->floats(b + "gate.weight");
            L.bias = pack->floats(b + "e_score_correction_bias");
        } else {
            L.router = randn(std::size_t{opt::kExperts} * H, 0.02F);
            L.bias = randn(opt::kExperts, 0.05F);
        }
        L.router_d = dupload(L.router);
        L.bias_d = dupload(L.bias);
        std::vector<opt::ExpertSlices> ex(opt::kExperts);
        for (unsigned e = 0; e < opt::kExperts; ++e) ex[e] = {L.gate[e].dev, L.up[e].dev, L.down[e].dev};
        L.experts_d = dupload(ex);
        L.shared = {L.sg.dev, L.su.dev, L.sd.dev};
        // Dense sites (rank slices as in G4), with rotating copies so a timed pass reads cold weights.
        const std::string a = lp + "self_attn.";
        struct Spec {
            const char* name;
            std::string prefix;
            bool out_slice;
            std::uint64_t first, n;
            unsigned in, out, bits, copies;
        };
        const std::vector<Spec> specs = {
            {"q", a + "q_proj", true, r * 4096, 4096, H, 4096, 6, 16},
            {"k", a + "k_proj", true, r * 256, 256, H, 256, 6, 64},
            {"v", a + "v_proj", true, r * 256, 256, H, 256, 6, 64},
            {"o", a + "o_proj", false, r * 4096, 4096, 4096, H, 6, 16},
            {"head", "language_model.lm_head", true, r * 100096, r == 0 ? 100096U : 99968U, H, r == 0 ? 100096U : 99968U, 8, 1},
            {"dense_gate", kP + "layers.0.mlp.gate_proj", true, r * 6144, 6144, H, 6144, 5, 8},
            {"dense_down", kP + "layers.0.mlp.down_proj", false, r * 6144, 6144, 6144, H, 5, 8},
            {"xgate", b + "experts.0.w1", true, r * IR, IR, H, IR, 3, 32},
            {"xdown", b + "experts.0.w2", false, r * IR, IR, IR, H, 3, 32},
        };
        for (const auto& sp : specs) {
            if (o.diverse) break;  // the decode-only cell needs no dense/head copies
            DenseSite ds;
            ds.name = sp.name;
            ds.input = sp.in;
            ds.output = sp.out;
            for (unsigned c = 0; c < sp.copies; ++c)
                ds.copies.push_back(mk(sp.prefix, sp.out_slice, sp.first, sp.n, sp.in, sp.out, sp.bits, 5000 + dense.size() * 100 + c));
            dense.push_back(std::move(ds));
        }
        // Scratch and workspaces.
        xin = dalloc<float>(8ULL * 6144);
        yA = dalloc<float>(8ULL * 100096);
        yB = dalloc<float>(8ULL * 100096);
        s.xe = dalloc<float>(8ULL * H);
        s.ye = dalloc<float>(8ULL * H);
        s.g = dalloc<float>(8ULL * IR);
        s.u = dalloc<float>(8ULL * IR);
        h = dalloc<float>(8ULL * H);
        y = dalloc<float>(8ULL * H);
        outA = dalloc<float>(8ULL * H);
        outB = dalloc<float>(8ULL * H);
        idsA = dalloc<std::uint32_t>(32);
        wA = dalloc<float>(32);
        idsB = dalloc<std::uint32_t>(32);
        wB = dalloc<float>(32);
        for (auto& ds : dense) lib.reserve(ds.copies[0].lib, ds.input, ds.output, 8, xin, yA);
        lib.reserve(L.gate[0].lib, H, IR, 8, s.xe, s.g);
        lib.reserve(L.down[0].lib, IR, H, 8, s.g, s.ye);
        hip_ok(hipMalloc(&moe_ws, opt::moe_workspace_bytes()), "moe ws");
        hip_ok(hipMemset(moe_ws, 0, opt::moe_workspace_bytes()), "moe ws zero");
        std::size_t lin = 0;
        for (auto& ds : dense) {
            opt::LinearJob j{ds.copies[0].dev, xin, ds.input, yB, ds.output};
            lin = std::max(lin, opt::linear_workspace_bytes(&j, 1));
        }
        if (!o.diverse) {
            opt::LinearJob qkv[3] = {{dense[0].copies[0].dev, xin, H, yB, 4096}, {dense[1].copies[0].dev, xin, H, yB, 256},
                                     {dense[2].copies[0].dev, xin, H, yB, 256}};
            lin = std::max(lin, opt::linear_workspace_bytes(qkv, 3));
        }
        if (lin != 0) {
            hip_ok(hipMalloc(&lin_ws, lin), "lin ws");
            hip_ok(hipMemset(lin_ws, 0, lin), "lin ws zero");
        }
        hip_ok(hipDeviceSynchronize(), "load");
        std::printf("weights: %s, layer %u rank %u, %.1f s\n", pack ? "real pack" : "synthetic", o.layer, o.rank,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    // --- MoE batches -----------------------------------------------------------------------------------------------
    // kind 0: random rows; 1: all rows equal (one shared route, a slot with n rows); 2: rows pairwise equal (mixed).
    Batch make_batch(unsigned n, int kind) {
        Batch bt;
        bt.n = n;
        std::vector<float> h0 = randn(std::size_t{n} * H, 1.0F), p0 = randn(std::size_t{n} * H, 0.5F), p1 = randn(std::size_t{n} * H, 0.5F);
        for (unsigned t = 1; t < n; ++t) {
            const unsigned src = kind == 1 ? 0 : (kind == 2 ? t / 2 * 2 : t);
            if (src == t) continue;
            std::copy_n(h0.data() + std::size_t{src} * H, H, h0.data() + std::size_t{t} * H);
            std::copy_n(p0.data() + std::size_t{src} * H, H, p0.data() + std::size_t{t} * H);
            std::copy_n(p1.data() + std::size_t{src} * H, H, p1.data() + std::size_t{t} * H);
        }
        bt.h0 = dupload(h0);
        bt.p0 = dupload(p0);
        bt.p1 = dupload(p1);
        bt.nw = dupload(randn(H, 0.1F));
        // Routes of A from the current post-exchange kernel (router on).
        hip_ok(hipMemcpy(h, bt.h0, std::size_t{n} * H * 4, hipMemcpyDeviceToDevice), "h");
        mref::hip::enqueue_post_exchange({n, h, bt.p0, bt.p1, bt.nw, y, L.router_d, L.bias_d, idsA, wA}, nullptr);
        bt.ids = d2h(idsA, std::size_t{n} * 4);
        bt.w4 = d2h(wA, std::size_t{n} * 4);
        bt.routes = group_routes(bt.ids, bt.w4, n);
        return bt;
    }
    void free_batch(Batch& bt) {
        for (float* p : {bt.h0, bt.p0, bt.p1, bt.nw}) hip_ok(hipFree(p), "free");
        free_routes(bt.routes);
    }
    void run_A(const Batch& bt) {
        hip_ok(hipMemcpyAsync(h, bt.h0, std::size_t{bt.n} * H * 4, hipMemcpyDeviceToDevice, nullptr), "h");
        mref::hip::enqueue_post_exchange({bt.n, h, bt.p0, bt.p1, bt.nw, y, L.router_d, L.bias_d, idsA, wA}, nullptr);
        moe_chain_A(lib, L, bt.routes, y, bt.n, outA, s);
    }
    opt::MoeView view_B(unsigned n) {
        opt::MoeView v;
        v.rows = n;
        v.y = y;
        v.router_w = L.router_d;
        v.bias = L.bias_d;
        v.experts = L.experts_d;
        v.shared = L.shared;
        v.out = outB;
        v.ids = idsB;
        v.weights = wB;
        v.ws.base = moe_ws;
        return v;
    }
    // B for one batch of rows: post-exchange (or its fused form in the router) + enqueue_moe.
    void run_B_rows(const float* h0, const float* p0, const float* p1, const float* nw, unsigned n) {
        hip_ok(hipMemcpyAsync(h, h0, std::size_t{n} * H * 4, hipMemcpyDeviceToDevice, nullptr), "h");
        opt::MoeView v = view_B(n);
        if (fused_post()) {
            v.h = h;
            v.part0 = p0;
            v.part1 = p1;
            v.norm_w = nw;
        } else {
            mref::hip::enqueue_post_exchange({n, h, p0, p1, nw, y, nullptr, nullptr, nullptr, nullptr}, nullptr);
        }
        opt::enqueue_moe(v, nullptr);
    }
    static bool fused_post() {
        static const bool f = std::getenv("STRIX_MMX_POST_ROUTER") != nullptr && std::atoi(std::getenv("STRIX_MMX_POST_ROUTER")) == 1;
        return f;
    }
    void run_B(const Batch& bt) { run_B_rows(bt.h0, bt.p0, bt.p1, bt.nw, bt.n); }
    std::uint64_t moe_bytes(const Batch& bt) const {
        std::uint64_t b = L.sg.bytes + L.su.bytes + L.sd.bytes + std::uint64_t{opt::kExperts} * H * 4;
        for (unsigned e = 0; e < opt::kExperts; ++e)
            if (!bt.routes.rows[e].empty()) b += L.gate[e].bytes + L.up[e].bytes + L.down[e].bytes;
        return b;
    }

    // Routed pipeline epoch wrap (STRIX_MMX_MOE_PIPE): the next route bumps the epoch past UINT32_MAX while every ready word
    // holds the reused epoch 1 from long ago; the call must still wait for this call's gate/up (same bits as A).
    void seed_pipe_wrap() {
        const std::size_t off = opt::moe_pipe_offset(), words = (opt::moe_workspace_bytes() - off) / sizeof(std::uint32_t);
        std::vector<std::uint32_t> w(words, 1U);
        w[0] = 0xFFFFFFFFU;  // route epoch
        w[1] = 0;            // ticket
        hip_ok(hipMemcpy(static_cast<unsigned char*>(moe_ws) + off, w.data(), words * sizeof(std::uint32_t), hipMemcpyHostToDevice),
               "pipeline wrap seed");
    }
    void test_moe() {
        unsigned cases = 0;
        for (unsigned n = 1; n <= 8; ++n)
            for (int kind = 0; kind < 3; ++kind) {
                if (n == 1 && kind > 0) continue;
                for (int rep = 0; rep < 2; ++rep) {
                    Batch bt = make_batch(n, kind);
                    run_A(bt);
                    hip_ok(hipDeviceSynchronize(), "A");
                    const auto a = d2h(outA, std::size_t{n} * H);
                    if (rep == 1 && (n == 2 || n == 4)) seed_pipe_wrap();
                    run_B(bt);
                    hip_ok(hipDeviceSynchronize(), "B");
                    const auto bo = d2h(outB, std::size_t{n} * H);
                    if (fused_post()) {  // the fused post-exchange must leave the same h and y as the post-exchange kernel
                        const auto hb = d2h(h, std::size_t{n} * H), yb = d2h(y, std::size_t{n} * H);
                        run_A(bt);
                        hip_ok(hipDeviceSynchronize(), "A again");
                        check(same_bits(d2h(h, std::size_t{n} * H), hb) && same_bits(d2h(y, std::size_t{n} * H), yb),
                              "fused post-exchange h/y differ n=" + std::to_string(n));
                        run_B(bt);
                        hip_ok(hipDeviceSynchronize(), "B again");
                    }
                    const auto ib = d2h(idsB, std::size_t{n} * 4);
                    const auto wb = d2h(wB, std::size_t{n} * 4);
                    std::vector<float> wa = bt.w4;
                    std::size_t at = 0;
                    const bool ok = same_bits(a, bo, &at);
                    check(ok, "moe n=" + std::to_string(n) + " kind " + std::to_string(kind) + " out differs at " + std::to_string(at) +
                                  (ok ? "" : " (A " + std::to_string(a[at]) + " B " + std::to_string(bo[at]) + ")"));
                    check(ib == bt.ids && same_bits(wa, wb), "moe n=" + std::to_string(n) + " routes differ");
                    opt::Dispatch dh{};  // before the single-row invariance runs overwrite it
                    hip_ok(hipMemcpy(&dh, opt::moe_dispatch({moe_ws}), sizeof(dh), hipMemcpyDeviceToHost), "dispatch");
                    // Batch invariance: each row alone gives the same bits as inside the batch.
                    if (n > 1 && rep == 0)
                        for (unsigned t = 0; t < n; t += 3) {
                            run_B_rows(bt.h0 + std::size_t{t} * H, bt.p0 + std::size_t{t} * H, bt.p1 + std::size_t{t} * H, bt.nw, 1);
                            hip_ok(hipDeviceSynchronize(), "B1");
                            const auto one = d2h(outB, H);
                            check(std::equal(one.begin(), one.end(), bo.data() + std::size_t{t} * H,
                                             [](float x, float z) { return std::memcmp(&x, &z, 4) == 0; }),
                                  "moe batch invariance n=" + std::to_string(n) + " row " + std::to_string(t));
                        }
                    if (rep == 0) std::printf("moe test n=%u kind %d: %u distinct experts, %s\n", n, kind, dh.count, ok ? "bitwise" : "DIFF");
                    ++cases;
                    free_batch(bt);
                }
            }
        std::printf("moe tests: %u cases\n", cases);
    }

    opt::MoeView diverse_view(unsigned n, const std::uint32_t* route_ids, const float* route_weights) {
        auto v = view_B(n);
        v.controlled_ids = route_ids;
        v.controlled_weights = route_weights;
        return v;  // h == nullptr: production v4 route, no post-exchange in the timed cell
    }
    std::vector<unsigned> singleton_cases(unsigned n, unsigned u, bool exhaustive) const {
        std::vector<unsigned> feasible;
        for (unsigned s_count = 0; s_count <= u; ++s_count)
            if (!diverse_ids(n, u, s_count, 0).empty()) feasible.push_back(s_count);
        if (exhaustive || feasible.empty()) return feasible;
        if (o.singletons == "min") return {feasible.front()};
        if (o.singletons == "max") return {feasible.back()};
        const auto requested = static_cast<unsigned>(std::stoul(o.singletons));
        return std::find(feasible.begin(), feasible.end(), requested) != feasible.end() ? std::vector<unsigned>{requested}
                                                                                     : std::vector<unsigned>{};
    }
    void test_diverse() {
        require_diverse_path(o.diverse_fallback);
        const auto env_int = [](const char* key, int fallback) {
            const char* value = std::getenv(key);
            return value != nullptr ? std::atoi(value) : fallback;
        };
        const int wide_mode = env_int("STRIX_MMX_MOE_WIDE", 2);
        const unsigned waves = static_cast<unsigned>(env_int("STRIX_MMX_MOE_W", 4));
        const int overlap = env_int("STRIX_MMX_MOE_OVERLAP", 1);
        const unsigned depth = static_cast<unsigned>(env_int("STRIX_MMX_MOE_DEPTH3", 4));
        const unsigned bits = L.gate[0].dev.bits;
        for (unsigned e = 0; e < opt::kExperts; ++e)
            if (L.gate[e].dev.bits != bits || L.up[e].dev.bits != bits || L.down[e].dev.bits != bits)
                throw std::invalid_argument("diverse gate requires uniform routed bits");
        const bool b8_fallback = strix::minimax::route::wide_for(8, wide_mode) || waves != 4 || bits != 3 || depth != 4;
        if (b8_fallback != o.diverse_fallback)
            throw std::invalid_argument("singleton qualification requires narrow B8/K3/W4/Depth4; use --diverse-fallback for wide/K5/W8/Depth8+12");
        if (env_int("STRIX_MMX_MOE_V", 4) != 4)
            throw std::invalid_argument("diverse gate requires v4");
        opt::check_moe_single_resources();
        const auto reader_counts = dupload(std::vector<std::uint32_t>(2, 0));
        unsigned observed_cases = 0, split_arms = 0;
        EnvMode mode("STRIX_MMX_MOE_SINGLE"), interleave("STRIX_MMX_MOE_GU_INTERLEAVE"),
            down("STRIX_MMX_MOE_DOWN_SPLIT");
        WorkspaceCheck workspace;
        unsigned cases = 0;
        const auto input = randn(8ULL * H, 0.5F);
        hip_ok(hipMemcpy(y, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice), "diverse input");
        std::vector<float> host_weights(32);
        constexpr float weights[4] = {0.13F, 0.27F, 0.61F, 0.99F};
        for (unsigned k = 0; k < 32; ++k) host_weights[k] = weights[k % 4];
        const auto route_weights = dupload(host_weights);
        for (unsigned n = 1; n <= 8; ++n)
            for (unsigned u = 4; u <= 4 * n; ++u)
                for (unsigned s_count : singleton_cases(n, u, true)) {
                    const auto route_ids = dupload(diverse_ids(n, u, s_count, cases));
                    const std::string label = "N=" + std::to_string(n) + " U=" + std::to_string(u) + " S=" + std::to_string(s_count);
                    interleave.set("0");
                    mode.set("0");
                    down.set("0");
                    hip_ok(hipMemset(moe_ws, 0, opt::moe_workspace_bytes()), "diverse ws zero");
                    opt::enqueue_moe(diverse_view(n, route_ids, route_weights), nullptr);
                    const auto batch = d2h(outB, std::size_t{n} * H);
                    workspace.save(moe_ws);
                    const auto expected_ids = d2h(route_ids, std::size_t{n} * 4);
                    const auto expected_weights = d2h(route_weights, std::size_t{n} * 4);
                    if (d2h(idsB, std::size_t{n} * 4) != expected_ids || !same_bits(d2h(wB, std::size_t{n} * 4), expected_weights))
                        throw std::runtime_error("diverse controlled routes mismatch " + label);
                    const auto dispatch = d2h(opt::moe_dispatch({moe_ws}), 1).front();
                    if (dispatch.count != u || dispatch.single_count != s_count || dispatch.reuse_count != u - s_count)
                        throw std::runtime_error("diverse bucket counts mismatch " + label);
                    for (const char* grid : {"0", "1"}) {
                        interleave.set(grid);
                        for (const char* single : {"0", "1"}) {
                            mode.set(single);
                            // DOWN_SPLIT 0 = fused finish, 1 = measured regime only, 2 = every v4 down.
                            for (const char* split : {"0", "1", "2"}) {
                                down.set(split);
                                const std::string arm = label + " IL=" + grid + " SINGLE=" + single + " DOWN=" + split;
                                hip_ok(hipMemset(moe_ws, 0, opt::moe_workspace_bytes()), "diverse ws zero");
                                hip_ok(hipMemset(reader_counts, 0, 2 * sizeof(std::uint32_t)), "reader counters zero");
                                auto observed = diverse_view(n, route_ids, route_weights);
                                observed.single_reader_counts = reader_counts;
                                opt::enqueue_moe(observed, nullptr);
                                const auto counts = d2h(reader_counts, 2);
                                const bool eligible = strix::minimax::route::moe_single_reader(
                                    single[0] == '1', n, u, s_count, false, 1, bits,
                                    strix::minimax::route::wide_for(n, wide_mode), waves, depth);
                                // One counter arrival per CTA: two GU matrices x 24 groups x 16 splits;
                                // down has 96 groups x 4 splits. Shared/reused slots must contribute zero.
                                const unsigned expected_gu = eligible ? s_count * 2U * 24U * 16U : 0U;
                                const unsigned expected_dn = eligible ? s_count * 96U * 4U : 0U;
                                if (counts[0] != expected_gu || counts[1] != expected_dn)
                                    throw std::runtime_error("singleton reader execution mismatch " + arm);
                                if (eligible && split[0] == '0') ++observed_cases;
                                if (strix::minimax::route::moe_down_split(static_cast<unsigned>(split[0] - '0'), n,
                                                                          strix::minimax::route::wide_for(n, wide_mode),
                                                                          waves, depth))
                                    ++split_arms;
                                if (!same_bits(batch, d2h(outB, std::size_t{n} * H)) || !workspace.equal(moe_ws) ||
                                    d2h(idsB, std::size_t{n} * 4) != expected_ids ||
                                    !same_bits(d2h(wB, std::size_t{n} * 4), expected_weights))
                                    throw std::runtime_error("diverse output/workspace/routes mismatch " + arm);
                                if (split[0] == '0') continue;
                                // Repeat without a reset: counters must remain reusable across calls, too.
                                opt::enqueue_moe(diverse_view(n, route_ids, route_weights), nullptr);
                                if (!same_bits(batch, d2h(outB, std::size_t{n} * H)) || !workspace.equal(moe_ws))
                                    throw std::runtime_error("diverse repeated output/workspace mismatch " + arm);
                            }
                        }
                    }
                    // All eight batch arms equal `batch`; compare every row against all eight standalone arms.
                    for (unsigned r = 0; r < n; ++r) {
                        auto v = diverse_view(1, route_ids + r * 4, route_weights + r * 4);
                        v.y = y + std::size_t{r} * H;
                        for (const char* grid : {"0", "1"}) {
                            interleave.set(grid);
                            for (const char* single : {"0", "1"}) {
                                mode.set(single);
                                for (const char* split : {"0", "2"}) {
                                    down.set(split);
                                    const std::string arm = label + " row=" + std::to_string(r) +
                                        " IL=" + grid + " SINGLE=" + single + " DOWN=" + split;
                                    hip_ok(hipMemset(moe_ws, 0, opt::moe_workspace_bytes()), "alone ws zero");
                                    opt::enqueue_moe(v, nullptr);
                                    const auto one = d2h(outB, H);
                                    if (std::memcmp(one.data(), batch.data() + std::size_t{r} * H, H * sizeof(float)) != 0 ||
                                        d2h(idsB, 4) != std::vector<std::uint32_t>(expected_ids.begin() + r * 4,
                                                                                 expected_ids.begin() + (r + 1) * 4) ||
                                        !same_bits(d2h(wB, 4), std::vector<float>(expected_weights.begin() + r * 4,
                                                                                expected_weights.begin() + (r + 1) * 4)))
                                        throw std::runtime_error("diverse alone output/routes mismatch " + arm);
                                    if (grid[0] == '0' && single[0] == '0' && split[0] == '0') workspace.save(moe_ws);
                                    else if (!workspace.equal(moe_ws))
                                        throw std::runtime_error("diverse alone workspace mismatch " + arm);
                                }
                            }
                        }
                    }
                    hip_ok(hipFree(route_ids), "diverse ids free");
                    ++cases;
                }
        hip_ok(hipFree(route_weights), "diverse weights free");
        hip_ok(hipFree(reader_counts), "reader counters free");
        if (!o.diverse_fallback && observed_cases == 0) throw std::runtime_error("singleton reader never ran");
        if (split_arms == 0) throw std::runtime_error("down split never ran");
        std::printf("diverse fatal: mode=%s WIDE=%d bits=%u W=%u Depth=%u overlap=%d cases=%u reader_cases=%u split_arms=%u "
                    "N=1..8 every feasible U/S GU_INTERLEAVE=0/1 SINGLE=0/1 DOWN_SPLIT=0/1/2 output/workspace/routes rows==alone: PASS\n",
                    o.diverse_fallback ? "fallback" : "singleton", wide_mode, bits, waves, depth, overlap, cases, observed_cases,
                    split_arms);
    }
    void bench_diverse() {
        require_diverse_path();
        opt::check_moe_single_resources();
        for (unsigned e = 0; e < opt::kExperts; ++e)
            if (L.gate[e].dev.bits != 3 || L.up[e].dev.bits != 3 || L.down[e].dev.bits != 3)
                throw std::invalid_argument("diverse byte floor requires K3 routed weights");
        if (L.shared.gate.bits != 5 || L.shared.up.bits != 5 || L.shared.down.bits != 5)
            throw std::invalid_argument("diverse byte floor requires K5 shared weights");
        const char* single = std::getenv("STRIX_MMX_MOE_SINGLE");
        const char* interleave = std::getenv("STRIX_MMX_MOE_GU_INTERLEAVE");
        const char* down = std::getenv("STRIX_MMX_MOE_DOWN_SPLIT");
        const unsigned down_mode = down != nullptr ? static_cast<unsigned>(std::atoi(down)) : 0U;
        std::printf("diverse overlap SINGLE=%s GU_INTERLEAVE=%s DOWN=%u, routed=10.616832 MB/expert shared=17.694720 MB floor=220 GB/s\n",
                    single != nullptr ? single : "0", interleave != nullptr ? interleave : "0", down_mode);
        auto median = [](std::vector<double> values) {
            std::sort(values.begin(), values.end());
            const std::size_t mid = values.size() / 2;
            return values.size() % 2 != 0 ? values[mid] : (values[mid - 1] + values[mid]) * 0.5;
        };
        const char* wide_env = std::getenv("STRIX_MMX_MOE_WIDE");
        const int wide_mode = wide_env != nullptr ? std::atoi(wide_env) : 2;
        const auto input = randn(8ULL * H, 0.5F);
        hip_ok(hipMemcpy(y, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice), "diverse input");
        const auto route_weights = dupload(std::vector<float>(32, 0.5F));
        // Natural router histogram: real layer weights with explicitly synthetic activations, no route override.
        if (!o.model.empty()) {
            for (unsigned n : {1U, 4U, 8U}) {
                unsigned histogram[33] = {};
                std::printf("route steps B=%u (real weights, synthetic activations):", n);
                for (unsigned step = 0; step < 8; ++step) {
                    const auto x = randn(std::size_t{n} * H, 0.5F);
                    hip_ok(hipMemcpy(y, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice), "route input");
                    opt::enqueue_moe(view_B(n), nullptr);
                    const auto dispatch = d2h(opt::moe_dispatch({moe_ws}), 1).front();
                    ++histogram[dispatch.count];
                    std::printf(" %u", dispatch.count);
                }
                std::printf("; histogram");
                for (unsigned u = 4; u <= 4 * n; ++u) if (histogram[u] != 0) std::printf(" U%u:%u", u, histogram[u]);
                std::printf("\n");
            }
            hip_ok(hipMemcpy(y, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice), "restore input");
        }
        unsigned cells = 0;
        for (unsigned n : {1U, 4U, 8U})
            for (unsigned u : n == 8 ? std::vector<unsigned>{4, 8, 12, 16, 20, 22, 24, 25, 26, 27, 28, 30, 32} :
                              n == 4 ? std::vector<unsigned>{4, 8, 12, 16} : std::vector<unsigned>{4})
                for (unsigned s_count : singleton_cases(n, u, false)) {
                    std::vector<std::uint32_t> all_ids;
                    constexpr unsigned rotation = 16;
                    for (unsigned step = 0; step < rotation; ++step) {
                        const auto ids = diverse_ids(n, u, s_count, step);
                        all_ids.insert(all_ids.end(), ids.begin(), ids.end());
                    }
                    const auto route_ids = dupload(all_ids);
                    auto run = [&](unsigned step) {
                        opt::enqueue_moe(diverse_view(n, route_ids + (step % rotation) * n * 4, route_weights), nullptr);
                    };
                    for (unsigned step = 0; step < 4; ++step) run(step);
                    hipEvent_t begin, end;
                    hip_ok(hipEventCreate(&begin), "diverse event");
                    hip_ok(hipEventCreate(&end), "diverse event");
                    std::vector<double> samples;
                    for (int rep = 0; rep < o.reps; ++rep)
                        for (int it = 0; it < o.iters; ++it) {
                            hip_ok(hipEventRecord(begin, nullptr), "diverse begin");
                            run(static_cast<unsigned>(rep * o.iters + it));
                            hip_ok(hipEventRecord(end, nullptr), "diverse end");
                            hip_ok(hipEventSynchronize(end), "diverse sync");
                            float ms = 0;
                            hip_ok(hipEventElapsedTime(&ms, begin, end), "diverse elapsed");
                            samples.push_back(1000.0 * ms);
                        }
                    const double us = median(samples);
                    if (o.ab_down) {  // same process, same routes/rotation: A = fused finish, B = the resolver default
                        EnvMode down_env("STRIX_MMX_MOE_DOWN_SPLIT");
                        std::vector<double> arm_samples[2];
                        std::string brackets;
                        for (int rep = 0; rep < o.reps; ++rep) {
                            double bracket[2] = {0.0, 0.0};
                            for (unsigned k = 0; k < 4; ++k) {
                                const unsigned arm = (k == 0 || k == 3) ? 0U : 1U;  // A B B A
                                down_env.set(arm == 0 ? "0" : "1");
                                for (int it = 0; it < o.iters; ++it) {
                                    hip_ok(hipEventRecord(begin, nullptr), "ab begin");
                                    run(static_cast<unsigned>((rep * 4 + static_cast<int>(k)) * o.iters + it));
                                    hip_ok(hipEventRecord(end, nullptr), "ab end");
                                    hip_ok(hipEventSynchronize(end), "ab sync");
                                    float ms = 0;
                                    hip_ok(hipEventElapsedTime(&ms, begin, end), "ab elapsed");
                                    arm_samples[arm].push_back(1000.0 * ms);
                                    bracket[arm] += 1000.0 * ms;
                                }
                            }
                            char cell[32];
                            std::snprintf(cell, sizeof cell, " %.3f", bracket[1] / bracket[0]);
                            brackets += cell;
                        }
                        const double a_us = median(arm_samples[0]), b_us = median(arm_samples[1]);
                        std::printf("diverse-ab DOWN B=%u U=%u S=%u A_us=%.3f B_us=%.3f B/A=%.4f bracket_B/A:%s split_at_B=%u\n", n, u,
                                    s_count, a_us, b_us, b_us / a_us, brackets.c_str(),
                                    strix::minimax::route::moe_down_split(1U, n, strix::minimax::route::wide_for(n, wide_mode), 4, 4)
                                        ? 1U : 0U);
                    }
                    const bool down_split = strix::minimax::route::moe_down_split(
                        down_mode, n, strix::minimax::route::wide_for(n, wide_mode), 4, 4);
                    // Probe phases separately: event boundaries never enter the total median. DOWN=0 includes
                    // combine in down_us; combine_us=0 means fused, not a measured zero-cost combine. DOWN=1 splits B4 only.
                    hipEvent_t marks[3];
                    for (auto& mark : marks) hip_ok(hipEventCreate(&mark), "diverse phase event");
                    std::vector<double> router_times, gu_times, down_times, combine_times;
                    for (unsigned probe = 0; probe < 3; ++probe) {
                        const auto v = diverse_view(n, route_ids + ((probe + 8U) % rotation) * n * 4, route_weights);
                        hip_ok(hipEventRecord(begin, nullptr), "diverse probe begin");
                        opt::enqueue_moe_profiled(v, nullptr, marks, marks[2]);
                        hip_ok(hipEventRecord(end, nullptr), "diverse probe end");
                        hip_ok(hipEventSynchronize(end), "diverse probe sync");
                        float router_ms = 0, gu_ms = 0, down_ms = 0, combine_ms = 0;
                        hip_ok(hipEventElapsedTime(&router_ms, begin, marks[0]), "diverse router time");
                        hip_ok(hipEventElapsedTime(&gu_ms, marks[0], marks[1]), "diverse gate/up time");
                        hip_ok(hipEventElapsedTime(&down_ms, marks[1], marks[2]), "diverse down time");
                        if (down_split) hip_ok(hipEventElapsedTime(&combine_ms, marks[2], end), "diverse combine time");
                        router_times.push_back(1000.0 * router_ms);
                        gu_times.push_back(1000.0 * gu_ms);
                        down_times.push_back(1000.0 * down_ms);
                        combine_times.push_back(1000.0 * combine_ms);
                    }
                    const bool wide = strix::minimax::route::wide_for(n, wide_mode);
                    const unsigned gu_groups = IR / (wide ? 256U : 64U), dn_groups = H / (wide ? 256U : 64U);
                    // Trellis-only lower bound: K3 routed and K5 shared, decimal bytes and GB/s. Scales/router excluded.
                    const double bytes = (3.0 * H * IR * 3.0 / 8.0) * u + (3.0 * H * IR * 5.0 / 8.0);
                    std::printf("diverse SINGLE=%s B=%u U=%u S=%u reuse=%u median_us=%.3f bytes=%.0f bytes/us=%.1f GB/s=%.2f "
                                "floor220_us=%.3f floor_ratio=%.3f samples=%zu router_us=%.3f gu_us=%.3f down_us=%.3f combine_us=%.3f "
                                "gu_ctas=%u dn_ctas=%u gu_arrivals=%u dn_arrivals=%u combine_ctas=%u DOWN=%u down_includes_combine=%u\n",
                                single != nullptr ? single : "0", n, u, s_count, u - s_count, us, bytes,
                                bytes / us, bytes / us / 1000.0, bytes / 220000.0, us / (bytes / 220000.0), samples.size(),
                                median(router_times), median(gu_times), median(down_times), median(combine_times), gu_groups * 16U * 2U * u,
                                dn_groups * 4U * (u + 1U), wide ? 32U : 64U, down_split ? 0U : (u + 1U) * (wide ? 4U : 8U),
                                down_split ? H / 128U : 0U, down_split ? 1U : 0U, down_split ? 0U : 1U);
                    for (auto mark : marks) hip_ok(hipEventDestroy(mark), "diverse phase event free");
                    hip_ok(hipEventDestroy(begin), "diverse event free");
                    hip_ok(hipEventDestroy(end), "diverse event free");
                    hip_ok(hipFree(route_ids), "diverse ids free");
                    ++cells;
                    std::fflush(stdout);
                }
        hip_ok(hipFree(route_weights), "diverse weights free");
        if (cells == 0) throw std::invalid_argument("no feasible diverse cells for --singletons");
    }

    void run_dense_A(const std::vector<std::pair<DenseSite*, Mat*>>& jobs, unsigned n, const float* x) {
        std::size_t off = 0;
        for (auto& [ds, m] : jobs) {
            lib.gemm(m->lib, ds->input, ds->output, x, yA + off, n);
            off += std::size_t{n} * ds->output;
        }
    }
    void run_dense_B(const std::vector<std::pair<DenseSite*, Mat*>>& jobs, unsigned n, const float* x) {
        std::vector<opt::LinearJob> js;
        std::size_t off = 0;
        for (auto& [ds, m] : jobs) {
            js.push_back({m->dev, x, ds->input, yB + off, ds->output});
            off += std::size_t{n} * ds->output;
        }
        opt::enqueue_linears(js.data(), static_cast<unsigned>(js.size()), n, lin_ws, nullptr);
    }
    std::vector<std::vector<DenseSite*>> dense_groups() {
        std::vector<std::vector<DenseSite*>> g;
        for (auto& ds : dense) g.push_back({&ds});
        g.push_back({&dense[0], &dense[1], &dense[2]});  // q/k/v grouped
        return g;
    }
    static std::string group_name(const std::vector<DenseSite*>& g) {
        std::string s;
        for (auto* ds : g) s += (s.empty() ? "" : "+") + ds->name;
        return s;
    }
    void test_dense() {
        for (auto& g : dense_groups()) {
            for (unsigned n = 1; n <= 8; ++n) {
                std::vector<std::pair<DenseSite*, Mat*>> jobs;
                std::size_t total = 0;
                for (auto* ds : g) {
                    jobs.push_back({ds, &ds->copies[0]});
                    total += std::size_t{n} * ds->output;
                }
                const auto x = randn(std::size_t{n} * 6144, 1.0F);
                hip_ok(hipMemcpy(xin, x.data(), x.size() * 4, hipMemcpyHostToDevice), "x");
                run_dense_A(jobs, n, xin);
                run_dense_B(jobs, n, xin);
                hip_ok(hipDeviceSynchronize(), "dense");
                std::size_t at = 0;
                const bool ok = same_bits(d2h(yA, total), d2h(yB, total), &at);
                check(ok, "dense " + group_name(g) + " n=" + std::to_string(n) + " differs at " + std::to_string(at));
                if (n == 8) {  // batch invariance: row 5 alone
                    std::vector<std::pair<DenseSite*, Mat*>> one = jobs;
                    run_dense_B(one, 1, xin + 5ULL * g[0]->input);
                    hip_ok(hipDeviceSynchronize(), "dense1");
                    const auto all = d2h(yA, total);
                    const auto b1 = d2h(yB, g[0]->output);
                    std::vector<float> ref(all.data() + 5ULL * g[0]->output, all.data() + 6ULL * g[0]->output);
                    check(same_bits(ref, b1), "dense batch invariance " + group_name(g));
                }
            }
            std::printf("dense test %s: n=1..8 done\n", group_name(g).c_str());
        }
    }

    // Router for many rows (prefill): enqueue_router_rows == the fused post-exchange router per row, bitwise.
    struct RouterRows {
        unsigned T;
        float *h0, *p0, *p1, *nw, *hh, *yy, *wa, *wb, *scratch;
        std::uint32_t *ia, *ib;
    };
    RouterRows make_router_rows(unsigned T) {
        RouterRows q{};
        q.T = T;
        q.h0 = dupload(randn(std::size_t{T} * H, 1.0F));
        q.p0 = dupload(randn(std::size_t{T} * H, 0.5F));
        q.p1 = dupload(randn(std::size_t{T} * H, 0.5F));
        q.nw = dupload(randn(H, 0.1F));
        q.hh = dalloc<float>(std::size_t{T} * H);
        q.yy = dalloc<float>(std::size_t{T} * H);
        q.wa = dalloc<float>(std::size_t{T} * 4);
        q.wb = dalloc<float>(std::size_t{T} * 4);
        q.ia = dalloc<std::uint32_t>(std::size_t{T} * 4);
        q.ib = dalloc<std::uint32_t>(std::size_t{T} * 4);
        q.scratch = dalloc<float>(std::size_t{T} * opt::kExperts * 2);
        return q;
    }
    void free_router_rows(RouterRows& q) {
        for (void* p : {static_cast<void*>(q.h0), static_cast<void*>(q.p0), static_cast<void*>(q.p1), static_cast<void*>(q.nw),
                        static_cast<void*>(q.hh), static_cast<void*>(q.yy), static_cast<void*>(q.wa), static_cast<void*>(q.wb),
                        static_cast<void*>(q.ia), static_cast<void*>(q.ib), static_cast<void*>(q.scratch)})
            hip_ok(hipFree(p), "free");
    }
    void router_rows_A(const RouterRows& q) {
        hip_ok(hipMemcpyAsync(q.hh, q.h0, std::size_t{q.T} * H * 4, hipMemcpyDeviceToDevice, nullptr), "h");
        mref::hip::enqueue_post_exchange({q.T, q.hh, q.p0, q.p1, q.nw, q.yy, L.router_d, L.bias_d, q.ia, q.wa}, nullptr);
    }
    void router_rows_B(const RouterRows& q) {
        hip_ok(hipMemcpyAsync(q.hh, q.h0, std::size_t{q.T} * H * 4, hipMemcpyDeviceToDevice, nullptr), "h");
        mref::hip::enqueue_post_exchange({q.T, q.hh, q.p0, q.p1, q.nw, q.yy, nullptr, nullptr, nullptr, nullptr}, nullptr);
        opt::enqueue_router_rows(q.T, q.yy, L.router_d, L.bias_d, q.scratch, q.ib, q.wb, nullptr);
    }
    void test_router_rows() {
        for (unsigned T : {1U, 7U, 64U, 333U, 1024U}) {
            RouterRows q = make_router_rows(T);
            router_rows_A(q);
            router_rows_B(q);
            hip_ok(hipDeviceSynchronize(), "router rows");
            const bool ok = d2h(q.ia, std::size_t{T} * 4) == d2h(q.ib, std::size_t{T} * 4) &&
                            same_bits(d2h(q.wa, std::size_t{T} * 4), d2h(q.wb, std::size_t{T} * 4));
            check(ok, "router rows T=" + std::to_string(T));
            std::printf("router rows test T=%u: %s\n", T, ok ? "bitwise" : "DIFF");
            free_router_rows(q);
        }
    }
    void bench_router_rows() {
        for (unsigned T : {256U, 1024U}) {
            const std::string cell = "router_rows T" + std::to_string(T);
            if (!wanted(cell)) continue;
            RouterRows q = make_router_rows(T);
            aba(cell, std::uint64_t{opt::kExperts} * H * 4, [&](int) { router_rows_A(q); }, [&](int) { router_rows_B(q); });
            free_router_rows(q);
        }
    }

    // Prefill MoE: enqueue_prefill_moe == the G4 prefill chain (strix_gemm per expert, ascending fmaf combine, + shared).
    struct Prefill {
        unsigned T;
        float *yy, *xe, *ye, *g, *u, *outA, *outB;
        std::vector<std::uint32_t> ids;
        std::vector<float> w4;
        Routes routes;
        void* ws = nullptr;
        std::size_t ws_bytes = 0;
        std::vector<opt::ExpertSlices> ex;
    };
    opt::PrefillMoeView view_prefill(const Prefill& q) const {
        opt::PrefillMoeView v;
        v.rows = q.T;
        v.y = q.yy;
        v.ids = q.ids.data();
        v.weights = q.w4.data();
        v.experts = q.ex.data();
        v.shared = L.shared;
        v.out = q.outB;
        return v;
    }
    Prefill make_prefill(unsigned T) {
        Prefill q{};
        q.T = T;
        RouterRows rr = make_router_rows(T);
        router_rows_A(rr);
        hip_ok(hipDeviceSynchronize(), "routes");
        q.ids = d2h(rr.ia, std::size_t{T} * 4);
        q.w4 = d2h(rr.wa, std::size_t{T} * 4);
        q.yy = dalloc<float>(std::size_t{T} * H);
        hip_ok(hipMemcpy(q.yy, rr.yy, std::size_t{T} * H * 4, hipMemcpyDeviceToDevice), "y");
        free_router_rows(rr);
        q.xe = dalloc<float>(std::size_t{T} * H);
        q.ye = dalloc<float>(std::size_t{T} * H);
        q.g = dalloc<float>(std::size_t{T} * IR);
        q.u = dalloc<float>(std::size_t{T} * IR);
        q.outA = dalloc<float>(std::size_t{T} * H);
        q.outB = dalloc<float>(std::size_t{T} * H);
        q.routes = group_routes(q.ids, q.w4, T);
        q.ex.resize(opt::kExperts);
        for (unsigned e = 0; e < opt::kExperts; ++e) q.ex[e] = {L.gate[e].dev, L.up[e].dev, L.down[e].dev};
        q.ws_bytes = opt::prefill_moe_workspace_bytes(view_prefill(q));
        hip_ok(hipMalloc(&q.ws, q.ws_bytes), "prefill ws");
        return q;
    }
    void free_prefill(Prefill& q) {
        for (void* p : {static_cast<void*>(q.yy), static_cast<void*>(q.xe), static_cast<void*>(q.ye), static_cast<void*>(q.g),
                        static_cast<void*>(q.u), static_cast<void*>(q.outA), static_cast<void*>(q.outB), q.ws})
            hip_ok(hipFree(p), "free");
        free_routes(q.routes);
    }
    void prefill_A(Prefill& q) {  // tests/hip_minimax_g4.hip MoE block, routes on the host
        const unsigned T = q.T;
        hip_ok(hipMemsetAsync(q.outA, 0, std::size_t{T} * H * 4, nullptr), "zero");
        for (unsigned e = 0; e < opt::kExperts; ++e) {
            const std::size_t k = q.routes.rows[e].size();
            if (k == 0) continue;
            hipLaunchKernelGGL(gather_kernel, dim3(static_cast<unsigned>(k)), dim3(256), 0, nullptr, q.yy, q.routes.rows_d + q.routes.offset[e], q.xe, H);
            lib.gemm_wmma(L.gate[e].lib, H, IR, q.xe, q.g, k);
            lib.gemm_wmma(L.up[e].lib, H, IR, q.xe, q.u, k);
            hipLaunchKernelGGL(swiglu_kernel, dim3(grid(k * IR)), dim3(256), 0, nullptr, q.g, q.u, q.g, k * IR);
            lib.gemm_wmma(L.down[e].lib, IR, H, q.g, q.ye, k);
            hipLaunchKernelGGL(scatter_add_kernel, dim3(static_cast<unsigned>(k)), dim3(256), 0, nullptr, q.ye,
                               q.routes.rows_d + q.routes.offset[e], q.routes.wts_d + q.routes.offset[e], q.outA, H);
        }
        lib.gemm_wmma(L.sg.lib, H, IR, q.yy, q.g, T);
        lib.gemm_wmma(L.su.lib, H, IR, q.yy, q.u, T);
        hipLaunchKernelGGL(swiglu_kernel, dim3(grid(std::size_t{T} * IR)), dim3(256), 0, nullptr, q.g, q.u, q.g, std::size_t{T} * IR);
        lib.gemm_wmma(L.sd.lib, IR, H, q.g, q.xe, T);
        hipLaunchKernelGGL(add_kernel, dim3(grid(std::size_t{T} * H)), dim3(256), 0, nullptr, q.outA, q.xe, std::size_t{T} * H);
    }
    void prefill_B(Prefill& q) { opt::enqueue_prefill_moe(view_prefill(q), q.ws, q.ws_bytes, nullptr); }
    void test_prefill() {
        for (unsigned T : {16U, 100U, 333U}) {
            Prefill q = make_prefill(T);
            prefill_A(q);
            prefill_B(q);
            hip_ok(hipDeviceSynchronize(), "prefill");
            std::size_t at = 0;
            const auto a = d2h(q.outA, std::size_t{T} * H), b2 = d2h(q.outB, std::size_t{T} * H);
            const bool ok = same_bits(a, b2, &at);
            check(ok, "prefill moe T=" + std::to_string(T) + " differs at " + std::to_string(at) +
                          (ok ? "" : " (A " + std::to_string(a[at]) + " B " + std::to_string(b2[at]) + ")"));
            std::printf("prefill moe test T=%u: %s (ws %.1f MB)\n", T, ok ? "bitwise" : "DIFF", static_cast<double>(q.ws_bytes) / 1e6);
            free_prefill(q);
        }
    }
    void bench_prefill() {
        for (unsigned T : {256U, 1024U}) {
            const std::string cell = "prefill_moe T" + std::to_string(T);
            if (!wanted(cell)) continue;
            Prefill q = make_prefill(T);
            std::uint64_t bytes = L.sg.bytes + L.su.bytes + L.sd.bytes;
            for (unsigned e = 0; e < opt::kExperts; ++e)
                if (!q.routes.rows[e].empty()) bytes += L.gate[e].bytes + L.up[e].bytes + L.down[e].bytes;
            const int saved = o.iters;
            o.iters = std::max(2, o.iters / 8);
            aba(cell, bytes, [&](int) { prefill_A(q); }, [&](int) { prefill_B(q); });
            o.iters = saved;
            free_prefill(q);
        }
    }

    // --- prefill core v2 ------------------------------------------------------------------------
    // Prefill with the routes of the first T tokens of --routes (real layer routes, random positive weights); the
    // file must hold >= T tokens. Without --routes: the router-rows routes of make_prefill (random hidden rows).
    std::vector<std::uint8_t> route_file;
    void set_ids(Prefill& q, const std::vector<std::uint32_t>& ids) {
        std::uniform_real_distribution<float> u(0.05F, 1.0F);
        const unsigned T = q.T;
        for (unsigned i = 0; i < T * 4; ++i) {
            q.ids[i] = ids[i];
            q.w4[i] = u(rng);
        }
        for (unsigned t = 0; t < T; ++t)  // distinct experts per token, as the router guarantees
            for (unsigned k = 1; k < 4; ++k)
                for (unsigned j = 0; j < k; ++j)
                    if (q.ids[t * 4 + k] == q.ids[t * 4 + j] || q.ids[t * 4 + k] >= opt::kExperts)
                        throw std::runtime_error("routes: repeated or invalid expert in a token");
        free_routes(q.routes);
        q.routes = group_routes(q.ids, q.w4, T);
        hip_ok(hipFree(q.ws), "free");
        q.ws_bytes = opt::prefill_moe_workspace_bytes(view_prefill(q));
        hip_ok(hipMalloc(&q.ws, q.ws_bytes), "prefill ws");
    }
    Prefill make_prefill_routes(unsigned T) {
        Prefill q = make_prefill(T);
        if (o.routes.empty()) return q;
        if (route_file.empty()) {
            std::ifstream f(o.routes, std::ios::binary);
            if (!f) throw std::runtime_error("cannot open --routes " + o.routes);
            route_file.assign(std::istreambuf_iterator<char>(f), {});
            std::printf("routes %s: %zu tokens\n", o.routes.c_str(), route_file.size() / 4);
        }
        if (route_file.size() < std::size_t{T} * 4) throw std::runtime_error("--routes holds fewer than T tokens");
        set_ids(q, std::vector<std::uint32_t>(route_file.begin(), route_file.begin() + static_cast<std::ptrdiff_t>(std::size_t{T} * 4)));
        return q;
    }
    // Row counts at every row-tile class boundary (1, 15..17, 31..33, 63..65, 127..129, 255..257, 500 rows), the rest
    // of the experts evenly (5-6 rows). Token t, slot k = the (k T + t)-th entry of the expert-sorted list: distinct experts per
    // token because every count <= T.
    Prefill make_prefill_crafted(bool mtfit = false, bool core8 = false) {
        const std::vector<unsigned> counts = core8
            ? std::vector<unsigned>{1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 79, 80, 81,
                                    95, 96, 97, 111, 112, 113, 127, 128, 129, 143, 144, 145, 255, 256, 257,
                                    300, 383, 384, 385, 499, 500, 511, 512, 513}
            : mtfit
            ? std::vector<unsigned>{1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 79, 80, 81,
                                    95, 96, 97, 111, 112, 113, 127, 128, 129, 143, 144, 145, 255, 256, 257, 300, 500}
            : std::vector<unsigned>{1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 500};
        const unsigned T = core8 ? 2048 : mtfit ? 1024 : 640;
        std::vector<std::uint32_t> list;
        unsigned e = 0;
        for (unsigned c : counts) list.insert(list.end(), c, e++);
        while (list.size() < 4U * T) {  // the rest spread evenly over the remaining experts
            if (e >= opt::kExperts) throw std::runtime_error("crafted routes: too many experts");
            const unsigned rem = 4U * T - static_cast<unsigned>(list.size()), left = opt::kExperts - e;
            const unsigned c = std::min(rem, (rem + left - 1) / left);
            list.insert(list.end(), c, e++);
        }
        if (e > opt::kExperts) throw std::runtime_error("crafted routes: too many experts");
        std::vector<std::uint32_t> ids(4U * T);
        for (unsigned t = 0; t < T; ++t)
            for (unsigned k = 0; k < 4; ++k) ids[t * 4 + k] = list[k * T + t];
        Prefill q = make_prefill(T);
        set_ids(q, ids);
        return q;
    }
    static double prefill_flops(const Prefill& q) {  // 2 x (gate + up + down) MACs over the 4 routes + shared per token
        return 2.0 * 5.0 * q.T * 3.0 * H * IR;
    }
    std::string rows_summary(const Prefill& q) const {
        std::vector<std::size_t> c;
        for (const auto& r : q.routes.rows) c.push_back(r.size());
        std::sort(c.begin(), c.end());
        std::size_t hit = 0, cls[4] = {};
        for (auto x : c)
            if (x) {
                ++hit;
                ++cls[opt::pfk::mt_class(static_cast<unsigned>(x))];
            }
        char buf[200];
        std::snprintf(buf, sizeof buf, "experts hit %zu, rows p10/p50/p90/max %zu/%zu/%zu/%zu, MT1/2/4/8 %zu/%zu/%zu/%zu", hit,
                      c[12], c[64], c[115], c[127], cls[0], cls[1], cls[2], cls[3]);
        return buf;
    }
    // One call of the given core on its own freshly poisoned workspace; returns out + every intermediate region.
    struct PfDump {
        std::vector<float> out;
        std::vector<std::uint8_t> xt, at, part;
    };
    PfDump run_dump(Prefill& q, int core, int mtfit = -1, bool raw_tiled = false, int core8 = 0, int gufuse = 0, bool debug_partials = false, float* intermediates = nullptr, bool producer_only = false, int pipe = 0) {
        const int saved_pipe = opt::prefill_pipe_override();
        struct RestorePipe { int value; ~RestorePipe() { opt::set_prefill_pipe(value); } } restore_pipe{saved_pipe};
        opt::set_prefill_pipe(pipe);
        void* ws = nullptr;
        hip_ok(hipMalloc(&ws, q.ws_bytes), "dump ws");
        hip_ok(hipMemset(ws, 0xff, q.ws_bytes), "poison ws");
        hip_ok(hipMemset(q.outB, 0xff, std::size_t{q.T} * H * 4), "poison out");
        opt::set_prefill_core(core);
        opt::set_prefill_mtfit(mtfit);
        opt::set_prefill_core8(core8);
        opt::set_prefill_gufuse(gufuse);
        opt::set_prefill_gufuse_debug(debug_partials, intermediates, producer_only);
        opt::enqueue_prefill_moe(view_prefill(q), ws, q.ws_bytes, nullptr);
        opt::set_prefill_core(-1);
        opt::set_prefill_mtfit(-1);
        opt::set_prefill_core8(-1);
        opt::set_prefill_gufuse(-1);
        opt::set_prefill_gufuse_debug(false);
        hip_ok(hipDeviceSynchronize(), "dump");
        const opt::PrefillDebugRegions r = opt::prefill_moe_debug_regions(view_prefill(q));
        auto bytes = [&](std::size_t off, std::size_t n) {
            std::vector<std::uint8_t> v(n);
            hip_ok(hipMemcpy(v.data(), static_cast<std::uint8_t*>(ws) + off, n, hipMemcpyDeviceToHost), "dump d2h");
            return v;
        };
        PfDump d{d2h(q.outB, std::size_t{q.T} * H), {}, {}, bytes(r.part_offset, r.part_bytes)};
        if (core == 3) {  // v4: xt / at are tile-major; un-tile into the row-major layout of the other cores
            const std::size_t tx = 2 * r.tpairs * H * 2, ta = r.tpairs * IR * 2;
            const auto xt = bytes(r.xt_offset, tx), at = bytes(r.at_offset, ta);
            if (raw_tiled) {
                d.xt = xt;
                d.at = at;
                hip_ok(hipFree(ws), "free");
                return d;
            }
            d.xt.assign(r.xt_bytes, 0);
            d.at.assign(r.at_bytes, 0);
            auto untile = [](const std::vector<std::uint8_t>& src, std::size_t src_rows, std::vector<std::uint8_t>& dst, std::size_t dst_rows,
                             unsigned K, unsigned R, unsigned p, unsigned mat) {
                for (unsigned ch = 0; ch < K; ++ch) {
                    const std::size_t t = ((static_cast<std::size_t>(R >> 4) * (K >> 4) + (ch >> 4)) << 8) + ((R & 15U) << 4) + (ch & 15U);
                    std::memcpy(&dst[((mat * dst_rows + p) * K + ch) * 2], &src[((mat * src_rows * K) + t) * 2], 2);
                }
            };
            for (const auto& g : r.groups)
                for (unsigned i = 0; i < g[2]; ++i) {
                    for (unsigned mat = 0; mat < 2; ++mat) untile(xt, r.tpairs, d.xt, r.pairs, H, g[1] + i, g[0] + i, mat);
                    untile(at, r.tpairs, d.at, r.pairs, IR, g[1] + i, g[0] + i, 0);
                }
        } else {
            d.xt = bytes(r.xt_offset, r.xt_bytes);
            d.at = bytes(r.at_offset, r.at_bytes);
        }
        hip_ok(hipFree(ws), "free");
        return d;
    }
    void check_prefill_v2(Prefill& q, const std::string& name) {
        prefill_A(q);  // the G4 chain: one strix_gemm per expert, ascending fmaf combine, + shared
        hip_ok(hipDeviceSynchronize(), "prefill A");
        const auto a = d2h(q.outA, std::size_t{q.T} * H);
        const PfDump dl = run_dump(q, 0);
        const bool ok_lib_a = same_bits(a, dl.out);
        check(ok_lib_a, "prefill lib core " + name + " differs from the G4 chain");
        for (int core : {1, 2, 3}) {  // v2, v3, v4: every intermediate region and the output vs the lib core, output vs G4
            const PfDump dv = run_dump(q, core);
            std::size_t at = 0;
            const bool ok_part = dl.part == dv.part, ok_xt = dl.xt == dv.xt, ok_at = dl.at == dv.at;
            const bool ok_out = same_bits(dl.out, dv.out, &at), ok_a = same_bits(a, dv.out, &at);
            const bool ok = ok_part && ok_xt && ok_at && ok_out && ok_a;
            const char* cn = core == 1 ? "v2" : core == 2 ? "v3" : "v4";
            check(ok, std::string("prefill ") + cn + " " + name + " differs (part " + std::to_string(ok_part) + " xt " + std::to_string(ok_xt) +
                          " at " + std::to_string(ok_at) + " out " + std::to_string(ok_out) + " G4 " + std::to_string(ok_a) + ")");
            std::printf("prefill %s test %s: partials %s (%.1f MB), xt %s, at %s, out ==lib %s, ==G4 %s, lib==G4 %s; %s\n", cn, name.c_str(),
                        ok_part ? "bitwise" : "DIFF", static_cast<double>(dl.part.size()) / 1e6, ok_xt ? "bitwise" : "DIFF",
                        ok_at ? "bitwise" : "DIFF", ok_out ? "bitwise" : "DIFF", ok_a ? "bitwise" : "DIFF", ok_lib_a ? "bitwise" : "DIFF",
                        rows_summary(q).c_str());
            std::fflush(stdout);
        }
    }
    void test_prefill_v2() {
        for (unsigned T : {16U, 100U, 333U, 1024U, 4096U}) {
            Prefill q = make_prefill_routes(T);
            check_prefill_v2(q, "T=" + std::to_string(T) + (o.routes.empty() ? " (router routes)" : " (file routes)"));
            free_prefill(q);
        }
        Prefill q = make_prefill_crafted();
        check_prefill_v2(q, "crafted T=640 (class boundaries)");
        free_prefill(q);
    }
    // Per-launch GPU times of one call through the labelled markers.
    struct Marks {
        std::vector<hipEvent_t> pool;
        std::vector<std::string> labels;
        std::size_t used = 0;
    };
    static void marker_cb(const char* label, hipStream_t stream, void* user) {
        auto* m = static_cast<Marks*>(user);
        if (m->used == m->pool.size()) {
            hipEvent_t e;
            hip_ok(hipEventCreate(&e), "ev");
            m->pool.push_back(e);
        }
        hip_ok(hipEventRecord(m->pool[m->used], stream), "rec");
        if (m->labels.size() <= m->used) m->labels.resize(m->used + 1);
        m->labels[m->used++] = label;
    }
    void stage_split(Prefill& q, const std::string& cell, int mtfit = -1, int core8 = 0) {
        const double fl = prefill_flops(q);
        for (int core : (mtfit < 0 ? std::vector<int>{0, 1, 2, 3, 8, 9} : std::vector<int>{3})) {  // lib, v2, v3, v4, v3 without A loads, v3 without decode (8, 9: timing only)
            Marks mk;
            std::vector<std::pair<std::string, double>> acc;
            const int n = 3;
            for (int r = 0; r < n; ++r) {
                mk.used = 0;
                if (core >= 8) opt::set_prefill_diagnostic_core(core);
                else opt::set_prefill_core(core);
                opt::set_prefill_mtfit(mtfit);
                opt::set_prefill_core8(core8);
                opt::set_prefill_marker(&Bench::marker_cb, &mk);
                prefill_B(q);
                opt::set_prefill_marker(nullptr, nullptr);
                opt::set_prefill_core(-1);
                opt::set_prefill_mtfit(-1);
                opt::set_prefill_core8(-1);
                hip_ok(hipDeviceSynchronize(), "sync");
                for (std::size_t i = 1; i < mk.used; ++i) {
                    float ms = 0;
                    hip_ok(hipEventElapsedTime(&ms, mk.pool[i - 1], mk.pool[i]), "elapsed");
                    if (acc.size() < i) acc.emplace_back(mk.labels[i], 0.0);
                    acc[i - 1].second += ms / n;
                }
            }
            double total = 0, gemm = 0;
            const char* cn = core == 0 ? "lib" : core == 1 ? "v2 " : core == 2 ? "v3 " : core == 3 ? "v4 " : core == 8 ? "v3-noA" : "v3-nodec";
            std::printf("stages %s %s ms:", cell.c_str(), cn);
            for (const auto& [l, ms] : acc) {
                std::printf(" %s %.3f", l.c_str(), ms);
                total += ms;
                if (l.rfind("gate_up", 0) == 0 || l.rfind("down", 0) == 0) gemm += ms;
            }
            std::printf(" = %.3f; GEMM stages %.3f ms = %.2f TFLOPS, call (GPU) %.2f TFLOPS\n", total, gemm, fl / (gemm * 1e9), fl / (total * 1e9));
            std::fflush(stdout);
            for (auto e : mk.pool) (void)hipEventDestroy(e);
        }
    }
    void check_prefill_mtfit(Prefill& q, const std::string& name) {
        check(q.ws_bytes <= opt::prefill_moe_workspace_bound(q.T), "mtfit workspace bound " + name);
        const PfDump off = run_dump(q, 3, 0, true), on = run_dump(q, 3, 1, true);
        const bool part = off.part == on.part, xt = off.xt == on.xt, at = off.at == on.at;
        const bool out = same_bits(off.out, on.out);
        check(part && xt && at && out, "mtfit " + name + " differs (part " + std::to_string(part) + " xt " +
              std::to_string(xt) + " at " + std::to_string(at) + " out " + std::to_string(out) + ")");
        std::printf("mtfit %s: partials %s, tiled xt %s, tiled at %s, output %s\n", name.c_str(),
                    part ? "bitwise" : "DIFF", xt ? "bitwise" : "DIFF", at ? "bitwise" : "DIFF", out ? "bitwise" : "DIFF");
        std::fflush(stdout);
    }
    void check_prefill_core8(Prefill& q, const std::string& name) {
        check(q.ws_bytes <= opt::prefill_moe_workspace_bound(q.T), "core8 workspace bound " + name);
        for (int mtfit : {0, 1}) {
            const PfDump off = run_dump(q, 3, mtfit, true, 0);
            for (int pipe : {1, 2}) {
                const PfDump on = run_dump(q, 3, mtfit, true, 0, 0, false, nullptr, false, pipe);
                const bool exact = off.part == on.part && off.xt == on.xt && off.at == on.at && same_bits(off.out, on.out);
                const std::string cell = "PIPE=" + std::to_string(pipe) + " mtfit=" + std::to_string(mtfit) + " " + name;
                check(exact, cell + " raw partials/transformed inputs/down inputs/output");
                if (!exact) throw std::runtime_error(cell + " failed bitwise qualification");
                std::printf("%s: partials, transformed inputs, down inputs, output bitwise\n", cell.c_str());
            }
            for (int core8 : {1, 2}) {
                const PfDump on = run_dump(q, 3, mtfit, true, core8);
                const bool part = off.part == on.part, xt = off.xt == on.xt, at = off.at == on.at;
                const bool out = same_bits(off.out, on.out);
                const std::string cell = "core8=" + std::to_string(core8) + " mtfit=" + std::to_string(mtfit) + " " + name;
                check(part && xt && at && out, cell + " differs (part " + std::to_string(part) + " xt " +
                      std::to_string(xt) + " at " + std::to_string(at) + " out " + std::to_string(out) + ")");
                std::printf("%s: partials %s, tiled xt %s, down input %s, output %s\n", cell.c_str(),
                            part ? "bitwise" : "DIFF", xt ? "bitwise" : "DIFF", at ? "bitwise" : "DIFF", out ? "bitwise" : "DIFF");
                std::fflush(stdout);
                if (!(part && xt && at && out)) throw std::runtime_error(cell + " failed bitwise qualification");
            }
        }
    }
    void test_prefill_table_ring() {
        for (int bad : {-2, 2}) {
            bool rejected = false;
            try { opt::set_prefill_table_ring(bad); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("table ring accepted invalid setter value");
        }
        for (unsigned T : {9U, 16U, 100U, 0U, 4096U}) {
            Prefill q = T == 0 ? make_prefill_crafted(true, true) : make_prefill(T);
            struct Restore {
                Bench& bench;
                Prefill& prefill;
                hipStream_t stream = nullptr;
                float* results = nullptr;
                ~Restore() {
                    if (hipStreamSynchronize(stream) != hipSuccess) std::abort();
                    opt::cleanup_prefill_table_ring();
                    opt::set_prefill_table_ring(-1);
                    opt::set_prefill_core(-1);
                    opt::set_prefill_mtfit(-1);
                    opt::set_prefill_core8(-1);
                    if (results != nullptr) (void)hipFree(results);
                    if (stream != nullptr) (void)hipStreamDestroy(stream);
                    bench.free_prefill(prefill);
                }
            } restore{*this, q};
            // Each call changes group ordering and table offsets but reuses this device workspace.
            hip_ok(hipFree(q.ws), "table ring old ws");
            q.ws = nullptr;
            q.ws_bytes = opt::prefill_moe_workspace_bound(q.T);
            hip_ok(hipMalloc(&q.ws, q.ws_bytes), "table ring ws");
            const std::size_t stride = std::size_t{q.T} * H;
            restore.results = dalloc<float>(6 * stride);
            hip_ok(hipDeviceSynchronize(), "table ring setup");
            hip_ok(hipStreamCreateWithFlags(&restore.stream, hipStreamNonBlocking), "table ring stream");
            opt::set_prefill_marker(nullptr, nullptr);
            opt::set_prefill_core8(0);
            for (int core : {0, 1, 2, 3}) {
                opt::set_prefill_core(core);
                for (int mtfit : {0, 1}) {
                    if (mtfit != 0 && core != 3) continue;
                    opt::set_prefill_mtfit(mtfit);
                    for (unsigned count : {2U, 6U}) {
                        const std::size_t elements = count * stride;
                        auto run = [&](int ring) {
                            opt::cleanup_prefill_table_ring();  // start from slot zero, six calls wrap all four slots
                            opt::set_prefill_table_ring(ring);
                            hip_ok(hipMemsetAsync(q.ws, 0xff, q.ws_bytes, restore.stream), "table ring poison ws");
                            hip_ok(hipMemsetAsync(restore.results, 0xff, elements * sizeof(float), restore.stream), "table ring poison out");
                            for (unsigned call = 0; call < count; ++call) {
                                auto ids = q.ids;
                                auto weights = q.w4;
                                for (std::size_t i = 0; i < ids.size(); ++i) {
                                    ids[i] = (ids[i] + 17U * call) % opt::kExperts;
                                    weights[i] *= 1.0F + static_cast<float>(call) * 0.125F;
                                }
                                auto v = view_prefill(q);
                                v.rows -= call % 3;  // shared/expert tails and table offsets differ between calls
                                v.ids = ids.data();
                                v.weights = weights.data();
                                v.out = restore.results + call * stride;
                                opt::enqueue_prefill_moe(v, q.ws, q.ws_bytes, restore.stream);
                                // No stream sync between calls; destroy/overwrite each caller's host routes now.
                                std::fill(ids.begin(), ids.end(), 0xffffffffU);
                                std::fill(weights.begin(), weights.end(), -123.0F);
                            }
                            hip_ok(hipStreamSynchronize(restore.stream), "table ring sequence");
                            return d2h(restore.results, elements);
                        };
                        const auto reference = run(0), actual = run(1);
                        const bool exact = std::memcmp(reference.data(), actual.data(), elements * sizeof(float)) == 0;
                        const std::string cell = "prefill table ring T" + std::to_string(q.T) + " core" +
                            std::to_string(core) + " mtfit" + std::to_string(mtfit) + " calls" + std::to_string(count);
                        check(exact, cell);
                        std::printf("%s: %s\n", cell.c_str(), exact ? "bitwise" : "DIFF");
                        if (!exact) throw std::runtime_error(cell + " failed bitwise qualification");
                    }
                }
            }
        }
    }
    void check_prefill_gufuse(Prefill& q, const std::string& name) {
        const auto regions = opt::prefill_moe_debug_regions(view_prefill(q));
        const std::size_t elements = 4 * regions.pairs * IR;
        float* trace = dalloc<float>(elements);
        struct Restore {
            float* ptr;
            ~Restore() {
                (void)hipDeviceSynchronize();
                opt::set_prefill_gufuse_debug(false);
                opt::set_prefill_gufuse(-1);
                (void)hipFree(ptr);
            }
        } restore{trace};
        for (int mtfit : {0, 1}) {
            const PfDump off = run_dump(q, 3, mtfit, true, 0, 0, true, trace);
            const auto reference = d2h(trace, elements);
            for (int core8 : {0, 1, 2}) {
                hip_ok(hipMemset(trace, 0xff, elements * sizeof(float)), "gufuse poison trace");
                const PfDump on = run_dump(q, 3, mtfit, true, core8, 1, true, trace);
                const auto actual = d2h(trace, elements);
                const bool exact = off.part == on.part && off.xt == on.xt && off.at == on.at &&
                    same_bits(off.out, on.out) && std::memcmp(reference.data(), actual.data(), elements * sizeof(float)) == 0;
                const std::string cell = "gufuse " + name + " mtfit=" + std::to_string(mtfit) + " core8=" + std::to_string(core8);
                check(exact, cell + " partials/xt/at/output/epilogue");
                if (!exact) throw std::runtime_error(cell + " failed bitwise qualification");
            }
            const PfDump paired = run_dump(q, 3, mtfit, true, 0, 1, true, nullptr, true);
            check(off.part == paired.part && off.xt == paired.xt && off.at == paired.at && same_bits(off.out, paired.out),
                  "gufuse paired producer " + name);
            // Production mode omits fused partial writes; down partials and all outputs must still match.
            const PfDump production = run_dump(q, 3, mtfit, true, 0, 1);
            check(off.xt == production.xt && off.at == production.at && same_bits(off.out, production.out),
                  "gufuse production " + name);
            std::vector<opt::ExpertSlices> group_weights;
            for (unsigned e = 0; e < opt::kExperts; ++e)
                if (!q.routes.rows[e].empty()) group_weights.push_back(q.ex[e]);
            group_weights.push_back(L.shared);
            std::size_t offset = 0;
            for (std::size_t gi = 0; gi < regions.groups.size(); ++gi) {
                const unsigned rows = regions.groups[gi][2];
                const auto& w = group_weights[gi];
                const unsigned gu = opt::gemm_splits(H, IR, w.gate.bits, rows);
                const std::size_t gu_bytes = 2ULL * gu * rows * IR * sizeof(float);
                if (gu != 1)
                    check(std::memcmp(off.part.data() + offset, production.part.data() + offset, gu_bytes) == 0,
                          "gufuse multisplit partials " + name);
                else
                    check(std::all_of(production.part.begin() + static_cast<std::ptrdiff_t>(offset), production.part.begin() + static_cast<std::ptrdiff_t>(offset + gu_bytes),
                                      [](std::uint8_t byte) { return byte == 0xff; }), "gufuse omitted partials " + name);
                offset += gu_bytes;
                const std::size_t dn_bytes = static_cast<std::size_t>(opt::gemm_splits(IR, H, w.down.bits, rows)) * rows * H * sizeof(float);
                check(std::memcmp(off.part.data() + offset, production.part.data() + offset, dn_bytes) == 0,
                      "gufuse down partials " + name);
                offset += dn_bytes;
            }
            check(offset == off.part.size(), "gufuse partial extent " + name);
        }
        std::printf("gufuse %s: partials, epilogue, tiled at and output checks complete\n", name.c_str());
    }
    void test_prefill_gufuse() {
        for (int bad : {-2, 2}) {
            bool rejected = false;
            try { opt::set_prefill_gufuse(bad); }
            catch (const std::invalid_argument&) { rejected = true; }
            check(rejected, "gufuse invalid setter");
        }
        check(opt::gemm_splits(H, IR, 3, 96) != 1 && opt::gemm_splits(H, IR, 3, 97) == 1,
              "gufuse 96/97 eligibility");
        std::vector<void*> owned;
        struct Restore {
            Layer& layer;
            opt::ExpertSlices shared;
            std::vector<void*>& buffers;
            ~Restore() {
                (void)hipDeviceSynchronize();
                layer.shared = shared;
                opt::set_prefill_gufuse_debug(false);
                opt::set_prefill_gufuse(-1);
                opt::set_prefill_core(-1);
                opt::set_prefill_mtfit(-1);
                opt::set_prefill_core8(-1);
                for (void* ptr : buffers) (void)hipFree(ptr);
            }
        } restore{L, L.shared, owned};
        auto mk = [&](unsigned input, unsigned output, unsigned bits, std::uint64_t seed) {
            const HostExl3 host = synthetic(input, output, bits, seed);
            opt::Exl3 w{};
            auto* tr = dupload(host.trellis);
            auto* su = dupload(host.suh);
            auto* sv = dupload(host.svh);
            owned.push_back(tr);
            owned.push_back(su);
            owned.push_back(sv);
            w.trellis = tr;
            w.suh = su;
            w.svh = sv;
            w.input = input;
            w.output = output;
            w.bits = bits;
            return w;
        };
        std::vector<opt::ExpertSlices> syn;
        for (unsigned bits : {3U, 4U, 5U, 6U, 8U})
            syn.push_back({mk(H, IR, bits, 13000 + bits * 3), mk(H, IR, bits, 13001 + bits * 3),
                           mk(IR, H, bits, 13002 + bits * 3)});
        // Include every 16-row boundary, then every exact-fit tail class above a full MT8 tile.
        const std::vector<unsigned> counts = {1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 79, 80, 81,
            95, 96, 97, 111, 112, 113, 127, 128, 129, 143, 144, 145, 159, 160, 161, 175, 176, 177,
            191, 192, 193, 207, 208, 209, 223, 224, 225, 239, 240, 241, 255, 256, 257};
        for (unsigned rotation = 0; rotation < 5; ++rotation) {
            L.shared = syn[rotation];
            Prefill q = make_prefill(2048);
            struct Free { Bench& b; Prefill& q; ~Free() { b.free_prefill(q); } } release{*this, q};
            for (unsigned e = 0; e < opt::kExperts; ++e) {
                q.ex[e] = syn[(e + rotation) % syn.size()];
                // Mixed gate/up widths exercise independent wave-local decode dispatch.
                q.ex[e].up = syn[(e + rotation + 1) % syn.size()].up;
            }
            std::vector<std::uint32_t> list;
            unsigned expert = 0;
            for (unsigned rows : counts) list.insert(list.end(), rows, expert++);
            while (list.size() < 4U * q.T) {
                const unsigned rem = 4U * q.T - static_cast<unsigned>(list.size());
                const unsigned rows = (rem + opt::kExperts - expert - 1) / (opt::kExperts - expert);
                list.insert(list.end(), rows, expert++);
            }
            std::vector<std::uint32_t> ids(4U * q.T);
            for (unsigned t = 0; t < q.T; ++t)
                for (unsigned k = 0; k < 4; ++k) ids[t * 4 + k] = list[k * q.T + t];
            set_ids(q, ids);
            check_prefill_gufuse(q, "boundaries bits rotation " + std::to_string(rotation));
            // Per-row constants retain both zero signs and subnormals before transforms;
            // infinities, NaNs and overflow also pass through WMMA and the exact clamp/expf path.
            const std::uint32_t fixtures[] = {0, 0x80000000U, 1, 0x80000001U, 0x33800000U, 0xb3800000U,
                0x35800000U, 0xb5800000U, 0x38800000U, 0xb8800000U,
                0x3f800000U, 0xbf800000U, 0x477fe000U, 0xc77fe000U, 0x7f800000U, 0xff800000U, 0x7fc12345U};
            std::vector<float> x(static_cast<std::size_t>(q.T) * H);
            for (unsigned row = 0; row < q.T; ++row) {
                const std::uint32_t bits = fixtures[row % (sizeof(fixtures) / sizeof(fixtures[0]))];
                float value;
                std::memcpy(&value, &bits, sizeof(value));
                std::fill_n(x.data() + static_cast<std::size_t>(row) * H, H, value);
            }
            hip_ok(hipMemcpy(q.yy, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice), "gufuse fixtures");
            check_prefill_gufuse(q, "exceptional bits rotation " + std::to_string(rotation));
        }
        L.shared = restore.shared;
        for (unsigned rows : {8U, 9U, 96U, 97U}) {
            Prefill q = make_prefill(rows);
            check_prefill_gufuse(q, "loaded shared rows " + std::to_string(rows));
            if (rows == 97)
                for (int core : {0, 1, 2}) {
                    const PfDump a = run_dump(q, core, 0, false, 0, 0);
                    const PfDump b = run_dump(q, core, 0, false, 0, 1);
                    check(a.part == b.part && a.xt == b.xt && a.at == b.at && same_bits(a.out, b.out),
                          "gufuse older core fallback " + std::to_string(core));
                }
            free_prefill(q);
        }
    }
    void check_pipe_codebook() {
        std::vector<std::uint32_t> reference(65536U * 2U);
        for (std::uint32_t state = 0; state < 65536U; ++state) {
            const std::uint32_t product = state * 0x83dcd12dU;
            const std::uint32_t split = state * 0xdcd12dU + ((state * 0x83U) << 24U);
            const unsigned sum = (product & 255U) + ((product >> 8U) & 255U) +
                ((product >> 16U) & 255U) + (product >> 24U);
            const auto old_bits = f16_bits(std::fma(static_cast<float>(1024U + sum), 887.0f / 131072.0f, -1329.0f / 128.0f));
            const auto new_bits = f16_bits(std::fma(static_cast<float>(sum), 887.0f / 131072.0f, -3.453125f));
            if (product != split || old_bits != new_bits)
                throw std::runtime_error("PIPE CPU codebook mismatch at state " + std::to_string(state));
            reference[state * 2U] = static_cast<std::uint32_t>(old_bits) | (static_cast<std::uint32_t>(new_bits) << 16U);
            reference[state * 2U + 1U] = product;
        }
        auto* result = dalloc<std::uint32_t>(reference.size());
        hipLaunchKernelGGL(pipe_codebook_check_kernel, dim3(512), dim3(128), 0, nullptr, result);
        hip_ok(hipGetLastError(), "PIPE codebook launch");
        const auto actual = d2h(result, reference.size());
        hip_ok(hipFree(result), "PIPE codebook free");
        if (reference != actual) throw std::runtime_error("PIPE GPU codebook differs from CPU reference");
        std::printf("PIPE CPU/GPU codebook: all 65536 products and old/new f16 encodings bitwise\n");
        for (int pipe : {1, 2}) {
            hipFuncAttributes attrs{};
            const auto error = pipe == 1
                ? hipFuncGetAttributes(&attrs, reinterpret_cast<const void*>(opt::pfk::grouped_pipe_kernel<8, 4, false>))
                : hipFuncGetAttributes(&attrs, reinterpret_cast<const void*>(opt::pfk::grouped_pipe_kernel<8, 4, true>));
            if (error != hipSuccess) {
                std::printf("PIPE%d resource query unavailable: %s\n", pipe, hipGetErrorString(error));
                continue;
            }
            std::printf("PIPE%d numRegs=%d scratch=%zu LDS=%zu budget=%s (<=240, zero scratch/LDS)\n",
                        pipe, attrs.numRegs, attrs.localSizeBytes, attrs.sharedSizeBytes,
                        attrs.numRegs <= 240 && attrs.localSizeBytes == 0 && attrs.sharedSizeBytes == 0 ? "met" : "MISS");
            int blocks = 0;
            const auto occupancy_error = pipe == 1
                ? hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, opt::pfk::grouped_pipe_kernel<8, 4, false>, 128, 0)
                : hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, opt::pfk::grouped_pipe_kernel<8, 4, true>, 128, 0);
            if (occupancy_error == hipSuccess)
                std::printf("PIPE%d occupancy: %d active 128-thread blocks per multiprocessor\n", pipe, blocks);
            else std::printf("PIPE%d occupancy unavailable: %s\n", pipe, hipGetErrorString(occupancy_error));
        }
    }
    void time_prefill_pipe(Prefill& q, const std::string& name) {
        const int saved_pipe = opt::prefill_pipe_override();
        struct Restore {
            int pipe;
            ~Restore() {
                opt::set_prefill_marker(nullptr, nullptr);
                opt::set_prefill_pipe(pipe);
                opt::set_prefill_core(-1);
                opt::set_prefill_core8(-1);
                opt::set_prefill_mtfit(-1);
                opt::set_prefill_gufuse(-1);
            }
        } restore{saved_pipe};
        for (int mtfit : {0, 1}) {
            double times[3][5] = {};
            // One warmup per arm followed by five interleaved repetitions.
            for (int rep = -1; rep < 5; ++rep) {
                for (int arm = 0; arm < 3; ++arm) {
                    Marks mk;
                    opt::set_prefill_core(3);
                    opt::set_prefill_core8(0);
                    opt::set_prefill_gufuse(0);
                    opt::set_prefill_mtfit(mtfit);
                    opt::set_prefill_pipe(arm);
                    opt::set_prefill_marker(&Bench::marker_cb, &mk);
                    prefill_B(q);
                    opt::set_prefill_marker(nullptr, nullptr);
                    hip_ok(hipDeviceSynchronize(), "PIPE timing");
                    if (rep >= 0) {
                        for (std::size_t i = 1; i < mk.used; ++i) {
                            const auto& label = mk.labels[i];
                            const int stage = label == "gather" ? 0 : label.rfind("gate_up", 0) == 0 ? 1 :
                                label == "finish" ? 2 : label.rfind("down", 0) == 0 ? 3 : label == "combine" ? 4 : -1;
                            if (stage < 0) continue;
                            float ms = 0;
                            hip_ok(hipEventElapsedTime(&ms, mk.pool[i - 1], mk.pool[i]), "PIPE elapsed");
                            times[arm][stage] += static_cast<double>(ms) / 5.0;
                        }
                    }
                    for (auto event : mk.pool) (void)hipEventDestroy(event);
                }
            }
            for (int arm = 0; arm < 3; ++arm)
                std::printf("PIPE stages %s mtfit=%d PIPE=%d mean of 5 interleaved ms: gather %.3f gate/up %.3f finish %.3f down %.3f combine %.3f\n",
                            name.c_str(), mtfit, arm, times[arm][0], times[arm][1], times[arm][2], times[arm][3], times[arm][4]);
        }
    }
    void test_prefill_pipe() {
        for (int bad : {-2, 3}) {
            bool rejected = false;
            try { opt::set_prefill_pipe(bad); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("PIPE setter accepted invalid value");
        }
        check_pipe_codebook();
        test_prefill_mtfit(true);
    }
    void test_prefill_core8() { test_prefill_pipe(); }
    // Token-major v4 gather (STRIX_MMX_PREFILL_GATHER_TOK) == per-pair gather: raw tile-major xt / at (padded rows
    // included: both start from the 0xff-poisoned workspace), split partials and output bitwise, gufuse off and on.
    void test_prefill_gather() {
        struct Restore { ~Restore() { opt::set_prefill_gather_tok(-1); } } restore;
        auto run = [&](Prefill& q, const std::string& name) {
            for (int gufuse : {0, 1}) {
                opt::set_prefill_gather_tok(0);
                const PfDump off = run_dump(q, 3, -1, true, 0, gufuse);
                opt::set_prefill_gather_tok(1);
                const PfDump on = run_dump(q, 3, -1, true, 0, gufuse);
                const bool part = off.part == on.part, xt = off.xt == on.xt, at = off.at == on.at, out = same_bits(off.out, on.out);
                const std::string cell = "gather_tok " + name + " gufuse=" + std::to_string(gufuse);
                check(part && xt && at && out, cell + " differs (part " + std::to_string(part) + " xt " + std::to_string(xt) +
                      " at " + std::to_string(at) + " out " + std::to_string(out) + ")");
                std::printf("%s: tiled xt %s, tiled at %s, partials %s, output %s\n", cell.c_str(), xt ? "bitwise" : "DIFF",
                            at ? "bitwise" : "DIFF", part ? "bitwise" : "DIFF", out ? "bitwise" : "DIFF");
                std::fflush(stdout);
            }
        };
        for (unsigned T : {1U, 17U, 100U, 1024U, 4096U}) {
            Prefill q = make_prefill_routes(T);
            run(q, "T" + std::to_string(T));
            free_prefill(q);
        }
        Prefill c = make_prefill_crafted();
        run(c, "crafted640");
        free_prefill(c);
    }
    void bench_prefill_gather() {
        for (unsigned T : {1024U, 4096U}) {
            const std::string cell = "pfg T" + std::to_string(T);
            if (!wanted(cell)) continue;
            Prefill q = make_prefill_routes(T);
            const int saved = o.iters;
            o.iters = std::max(2, o.iters / 8);
            // Whole calls, A = per-pair gather, B = token-major gather (production v4 otherwise).
            aba(cell + " pair/tok", 0, [&](int) { opt::set_prefill_gather_tok(0); prefill_B(q); },
                [&](int) { opt::set_prefill_gather_tok(1); prefill_B(q); });
            o.iters = saved;
            for (int tok : {0, 1, 0, 1}) {
                opt::set_prefill_gather_tok(tok);
                stage_split(q, cell + (tok ? " tok" : " pair"), 1);  // production v4 (mtfit default on)
            }
            opt::set_prefill_gather_tok(-1);
            free_prefill(q);
        }
    }
    void test_prefill_mtfit(bool core8 = false) {
        // Five small weight sets cover all supported bit widths in the same grouped launches.
        std::vector<void*> owned;
        struct Restore {
            Layer& layer;
            opt::ExpertSlices shared;
            std::vector<void*>& buffers;
            ~Restore() {
                layer.shared = shared;
                opt::set_prefill_marker(nullptr, nullptr);
                opt::set_prefill_core(-1);
                opt::set_prefill_mtfit(-1);
                opt::set_prefill_core8(-1);
                for (void* ptr : buffers) (void)hipFree(ptr);
            }
        } restore{L, L.shared, owned};
        auto mk = [&](unsigned input, unsigned output, unsigned bits, std::uint64_t seed) {
            const HostExl3 hst = synthetic(input, output, bits, seed);
            opt::Exl3 w{};
            auto* tr = dupload(hst.trellis);
            auto* su = dupload(hst.suh);
            auto* sv = dupload(hst.svh);
            owned.push_back(tr);
            owned.push_back(su);
            owned.push_back(sv);
            w.trellis = tr;
            w.suh = su;
            w.svh = sv;
            w.input = input;
            w.output = output;
            w.bits = bits;
            return w;
        };
        std::vector<opt::ExpertSlices> syn;
        for (unsigned bits : {3U, 4U, 6U, 8U, 5U})
            syn.push_back({mk(H, IR, bits, 9000 + bits * 3), mk(H, IR, bits, 9001 + bits * 3),
                           mk(IR, H, bits, 9002 + bits * 3)});
        const auto saved_shared = L.shared;
        if (core8) {
            Prefill real = make_prefill_routes(4096);
            struct Free { Bench& b; Prefill& q; ~Free() { b.free_prefill(q); } } release{*this, real};
            const std::string name = o.routes.empty() ? "loaded weights/router routes T4096" : "loaded weights/file routes T4096";
            check_prefill_core8(real, name);
            time_prefill_pipe(real, name);
        }
        for (bool synthetic_weights : {false, true}) {
            L.shared = synthetic_weights ? syn.back() : saved_shared;
            for (unsigned T : {0U, 100U, 1024U, 4096U}) {
                Prefill q = T == 0 ? make_prefill_crafted(true, core8) : make_prefill(T);
                if (synthetic_weights)
                    for (unsigned e = 0; e < opt::kExperts; ++e) q.ex[e] = syn[e % (core8 ? 5 : 4)];
                if (T != 0) {
                    std::uniform_int_distribution<unsigned> pick(0, opt::kExperts - 1);
                    std::vector<std::uint32_t> ids(4U * T);
                    for (unsigned t = 0; t < T; ++t)
                        for (unsigned k = 0; k < 4; ++k) {
                            unsigned id;
                            bool duplicate;
                            do {
                                id = pick(rng);
                                duplicate = false;
                                for (unsigned j = 0; j < k; ++j) duplicate |= ids[t * 4 + j] == id;
                            } while (duplicate);
                            ids[t * 4 + k] = id;
                        }
                    set_ids(q, ids);
                } else {
                    set_ids(q, q.ids);  // rebuild the workspace after changing bit widths
                }
                const std::string name = std::string(synthetic_weights ? (core8 ? "synthetic 3/4/5/6/8 " : "synthetic 3/4/6/8 + shared5 ") : "loaded weights ") +
                    (T == 0 ? "crafted boundaries" : "random T" + std::to_string(T));
                if (core8) check_prefill_core8(q, name);
                else check_prefill_mtfit(q, name);
                if (T == 1024 || T == 4096) {
                    if (core8) {
                        time_prefill_pipe(q, name);
                    } else {
                        stage_split(q, name + " mtfit OFF", 0);
                        stage_split(q, name + " mtfit ON", 1);
                    }
                }
                free_prefill(q);
            }
        }
        L.shared = saved_shared;
    }
    void bench_prefill_gufuse() {
        const std::string cell = "pfgufuse paired GEMM";
        if (!wanted(cell)) return;
        Prefill q = make_prefill_routes(4096);
        struct Restore {
            Bench& bench;
            Prefill& prefill;
            ~Restore() {
                (void)hipDeviceSynchronize();
                opt::set_prefill_marker(nullptr, nullptr);
                opt::set_prefill_gufuse_debug(false);
                opt::set_prefill_gufuse(-1);
                opt::set_prefill_core(-1);
                opt::set_prefill_mtfit(-1);
                opt::set_prefill_core8(-1);
                bench.free_prefill(prefill);
            }
        } restore{*this, q};
        unsigned eligible = opt::gemm_splits(H, IR, L.shared.gate.bits, q.T) == 1 ? 1U : 0U;
        for (unsigned e = 0; e < opt::kExperts; ++e)
            if (!q.routes.rows[e].empty() && opt::gemm_splits(H, IR, q.ex[e].gate.bits,
                    static_cast<unsigned>(q.routes.rows[e].size())) == 1) ++eligible;
        if (eligible == 0) throw std::runtime_error("gufuse benchmark has no eligible group");
        for (bool producer_only : {true, false}) {
            std::vector<double> times[2];
            for (int arm = -2; arm < 7; ++arm) {
                const int on = arm < 0 ? arm + 2 : arm % 2;
                Marks mk;
                opt::set_prefill_core(3);
                opt::set_prefill_core8(0);
                opt::set_prefill_mtfit(1);
                opt::set_prefill_gufuse(on);
                opt::set_prefill_gufuse_debug(producer_only, nullptr, producer_only);
                opt::set_prefill_marker(&Bench::marker_cb, &mk);
                for (int i = 0; i < o.iters; ++i) prefill_B(q);
                opt::set_prefill_marker(nullptr, nullptr);
                hip_ok(hipDeviceSynchronize(), "gufuse timing");
                double ms = 0;
                for (std::size_t i = 1; i < mk.used; ++i) {
                    if (mk.labels[i].rfind("gate_up", 0) != 0 && (producer_only || mk.labels[i] != "finish")) continue;
                    float delta = 0;
                    hip_ok(hipEventElapsedTime(&delta, mk.pool[i - 1], mk.pool[i]), "gufuse elapsed");
                    ms += delta / static_cast<float>(o.iters);
                }
                if (arm >= 0) times[on].push_back(ms);
                for (hipEvent_t event : mk.pool) (void)hipEventDestroy(event);
            }
            auto median = [](std::vector<double> values) {
                std::sort(values.begin(), values.end());
                return (values[(values.size() - 1) / 2] + values[values.size() / 2]) * 0.5;
            };
            const double a = median(times[0]), b = median(times[1]);
            std::printf("pfgufuse %s eligible=%u: A %.3f ms B %.3f ms B/A %.4f\n",
                        producer_only ? "paired GEMM" : "producer+finish", eligible, a, b, b / a);
            check(b <= a, producer_only ? "gufuse paired GEMM throughput regressed" : "gufuse producer+finish regressed");
        }
    }
    void bench_prefill_v2() {
        for (unsigned T : {256U, 1024U, 4096U, 640U}) {
            const std::string cell = T == 640 ? std::string("pf2 crafted640") : "pf2 T" + std::to_string(T);
            if (!wanted(cell)) continue;
            Prefill q = T == 640 ? make_prefill_crafted() : make_prefill_routes(T);
            std::printf("%s: %s\n", cell.c_str(), rows_summary(q).c_str());
            std::uint64_t bytes = L.sg.bytes + L.su.bytes + L.sd.bytes;
            for (unsigned e = 0; e < opt::kExperts; ++e)
                if (!q.routes.rows[e].empty()) bytes += L.gate[e].bytes + L.up[e].bytes + L.down[e].bytes;
            const int saved = o.iters;
            o.iters = std::max(2, o.iters / 8);
            // Whole calls (host plan + table upload + launches): lib/v4, v2/v4, v3/v4 (A/B).
            aba(cell + " lib/v4", bytes, [&](int) { opt::set_prefill_core(0); prefill_B(q); }, [&](int) { opt::set_prefill_core(3); prefill_B(q); });
            aba(cell + " v2/v4", bytes, [&](int) { opt::set_prefill_core(1); prefill_B(q); }, [&](int) { opt::set_prefill_core(3); prefill_B(q); });
            aba(cell + " v3/v4", bytes, [&](int) { opt::set_prefill_core(2); prefill_B(q); }, [&](int) { opt::set_prefill_core(3); prefill_B(q); });
            o.iters = saved;
            opt::set_prefill_core(-1);
            stage_split(q, cell);
            free_prefill(q);
        }
    }
    // Dense prefill sites through strix_gemm (the engine's prefill identity) at T rows: us and TFLOPS.
    // Dense prefill linears on the v4 core (opt::enqueue_prefill_dense) == the library strix_gemm, bitwise, for the
    // attention sites alone and k+v in one call; outputs start poisoned.
    std::vector<std::vector<DenseSite*>> dense_v4_groups() {
        std::vector<std::vector<DenseSite*>> g;
        DenseSite *q = nullptr, *k = nullptr, *v = nullptr, *ov = nullptr;
        for (auto& ds : dense) {
            if (ds.name == "q") q = &ds;
            if (ds.name == "k") k = &ds;
            if (ds.name == "v") v = &ds;
            if (ds.name == "o") ov = &ds;
        }
        if (k != nullptr && v != nullptr) g.push_back({k, v});
        if (q != nullptr) g.push_back({q});
        if (ov != nullptr) g.push_back({ov});
        if (q != nullptr && k != nullptr && v != nullptr) g.push_back({q, k, v});
        return g;
    }
    void run_dense_v4(const std::vector<DenseSite*>& g, const float* in, std::vector<float*>& outs, unsigned T, void* ws, std::size_t ws_bytes, std::size_t copy) {
        std::vector<opt::PrefillDenseJob> jobs;
        for (std::size_t i = 0; i < g.size(); ++i) jobs.push_back({g[i]->copies[copy % g[i]->copies.size()].dev, outs[i]});
        opt::enqueue_prefill_dense(jobs.data(), static_cast<unsigned>(jobs.size()), in, T, ws, ws_bytes, nullptr);
    }
    std::size_t dense_v4_bytes(const std::vector<DenseSite*>& g, unsigned T) {
        std::vector<opt::PrefillDenseJob> jobs;
        for (auto* ds : g) jobs.push_back({ds->copies[0].dev, reinterpret_cast<float*>(1)});
        return opt::prefill_dense_workspace_bytes(jobs.data(), static_cast<unsigned>(jobs.size()), T);
    }
    void test_prefill_dense() {
        {  // unsupported bit widths are refused (the engine then keeps the library call)
            opt::Exl3 w = dense[0].copies[0].dev;
            w.bits = 7;
            bool refused = false;
            try {
                const opt::PrefillDenseJob j{w, reinterpret_cast<float*>(1)};
                (void)opt::prefill_dense_workspace_bytes(&j, 1, 1024);
            } catch (const std::invalid_argument&) {
                refused = true;
            }
            check(refused && !opt::prefill_dense_supported(w), "dense_v4 refuses 7-bit weights");
            std::printf("dense_v4 7-bit descriptor: %s\n", refused ? "refused" : "ACCEPTED");
        }
        for (unsigned T : {9U, 17U, 100U, 129U, 1024U, 4096U}) {
            for (const auto& g : dense_v4_groups()) {
                const unsigned K = g[0]->input;
                std::vector<float> hin(std::size_t{T} * K);
                std::normal_distribution<float> nd(0.0F, 1.0F);
                for (auto& x : hin) x = nd(rng);
                float* in = dupload(hin);
                std::vector<float*> libOut, v4Out;
                for (auto* ds : g) {
                    libOut.push_back(dalloc<float>(std::size_t{T} * ds->output));
                    v4Out.push_back(dalloc<float>(std::size_t{T} * ds->output));
                    hip_ok(hipMemset(libOut.back(), 0xff, std::size_t{T} * ds->output * 4), "poison");
                    hip_ok(hipMemset(v4Out.back(), 0xff, std::size_t{T} * ds->output * 4), "poison");
                }
                const std::size_t bytes = dense_v4_bytes(g, T);
                void* ws = nullptr;
                hip_ok(hipMalloc(&ws, bytes), "dense ws");
                hip_ok(hipMemset(ws, 0xff, bytes), "poison ws");
                for (std::size_t i = 0; i < g.size(); ++i) lib.gemm_wmma(g[i]->copies[0].lib, g[i]->input, g[i]->output, in, libOut[i], T);
                run_dense_v4(g, in, v4Out, T, ws, bytes, 0);
                hip_ok(hipDeviceSynchronize(), "dense sync");
                bool all = true;
                std::string detail;
                for (std::size_t i = 0; i < g.size(); ++i) {
                    const auto a = d2h(libOut[i], std::size_t{T} * g[i]->output), b = d2h(v4Out[i], std::size_t{T} * g[i]->output);
                    const bool same = same_bits(a, b);
                    std::size_t diff = 0;
                    for (std::size_t j = 0; j < a.size(); ++j)
                        if (std::memcmp(&a[j], &b[j], 4) != 0) ++diff;
                    all = all && same;
                    detail += " " + g[i]->name + (same ? " bitwise" : " DIFF " + std::to_string(diff) + "/" + std::to_string(a.size()));
                }
                const std::string cell = "dense_v4 " + group_name(g) + " T" + std::to_string(T);
                check(all, cell + " differs:" + detail);
                std::printf("%s:%s (splits", cell.c_str(), detail.c_str());
                for (auto* ds : g) std::printf(" %u", opt::gemm_splits(ds->input, ds->output, ds->copies[0].dev.bits, T));
                std::printf(")\n");
                std::fflush(stdout);
                hip_ok(hipFree(ws), "free");
                hip_ok(hipFree(in), "free");
                for (auto* x : libOut) hip_ok(hipFree(x), "free");
                for (auto* x : v4Out) hip_ok(hipFree(x), "free");
            }
        }
    }
    void bench_prefill_dense() {
        for (unsigned T : {1024U, 4096U}) {
            for (const auto& g : dense_v4_groups()) {
                const std::string cell = "pfdv4 " + group_name(g) + " T" + std::to_string(T);
                if (!wanted(cell)) continue;
                const unsigned K = g[0]->input;
                float* in = dalloc<float>(std::size_t{T} * K);
                hip_ok(hipMemset(in, 0, std::size_t{T} * K * 4), "zero");
                std::vector<float*> outs;
                double fl = 0;
                std::uint64_t wbytes = 0;
                for (auto* ds : g) {
                    outs.push_back(dalloc<float>(std::size_t{T} * ds->output));
                    fl += 2.0 * T * ds->input * ds->output;
                    wbytes += ds->copies[0].bytes;
                }
                const std::size_t bytes = dense_v4_bytes(g, T);
                void* ws = nullptr;
                hip_ok(hipMalloc(&ws, bytes), "dense ws");
                // Rotating cold copies on both sides; A = library strix_gemm per site, B = one v4 call for the group.
                aba(cell + " lib/v4", wbytes, [&](int i) {
                        for (std::size_t si = 0; si < g.size(); ++si) {
                            const Mat& m = g[si]->copies[static_cast<std::size_t>(i < 0 ? 0 : i) % g[si]->copies.size()];
                            lib.gemm_wmma(m.lib, g[si]->input, g[si]->output, in, outs[si], T);
                        }
                    }, [&](int i) { run_dense_v4(g, in, outs, T, ws, bytes, static_cast<std::size_t>(i < 0 ? 0 : i)); });
                const double us = time_us([&](int i) { run_dense_v4(g, in, outs, T, ws, bytes, static_cast<std::size_t>(i < 0 ? 0 : i)); }, o.iters);
                std::printf("cell %-26s v4 %9.1f us  %.2f TFLOPS\n", cell.c_str(), us, fl / us / 1e6);
                std::fflush(stdout);
                hip_ok(hipFree(ws), "free");
                hip_ok(hipFree(in), "free");
                for (auto* x : outs) hip_ok(hipFree(x), "free");
            }
        }
    }
    void bench_dense_prefill() {
        for (unsigned T : {1024U, 4096U}) {
            float* in = dalloc<float>(std::size_t{T} * H);
            float* out = dalloc<float>(std::size_t{T} * H);
            hip_ok(hipMemset(in, 0, std::size_t{T} * H * 4), "zero");
            for (auto& ds : dense) {
                if (ds.name == "head" || ds.name == "xgate" || ds.name == "xdown") continue;
                const std::string cell = "pfdense " + ds.name + " T" + std::to_string(T);
                if (!wanted(cell)) continue;
                const double us = time_us([&](int i) {
                    const Mat& m = ds.copies[static_cast<std::size_t>(i < 0 ? 0 : i) % ds.copies.size()];
                    lib.gemm_wmma(m.lib, ds.input, ds.output, in, out, T);
                }, std::max(4, o.iters / 4));
                const double fl = 2.0 * T * ds.input * ds.output;
                std::printf("cell %-26s %9.1f us  %.2f TFLOPS\n", cell.c_str(), us, fl / us / 1e6);
                std::fflush(stdout);
            }
            hip_ok(hipFree(in), "free");
            hip_ok(hipFree(out), "free");
        }
    }
    // WMMA ceiling: f16 (the prefill core's instruction) and iu8 at 1/4/8 independent chains per wave over a grid sweep,
    // each run calibrated to ~100 ms; then ~3 s back-to-back f16 (sustained clocks/power, sampled by the benchmark script);
    // then the v2 B-fragment build alone (dequant ceiling, G weights/s) for 3- and 6-bit tiles.
    double timed_ms(const std::function<void()>& launch) {
        hipEvent_t a, b;
        hip_ok(hipEventCreate(&a), "ev");
        hip_ok(hipEventCreate(&b), "ev");
        hip_ok(hipEventRecord(a, nullptr), "rec");
        launch();
        hip_ok(hipEventRecord(b, nullptr), "rec");
        hip_ok(hipEventSynchronize(b), "sync");
        float ms = 0;
        hip_ok(hipEventElapsedTime(&ms, a, b), "elapsed");
        (void)hipEventDestroy(a);
        (void)hipEventDestroy(b);
        return ms;
    }
    void bench_wmma_ceiling() {
        if (!wanted("ceiling")) return;
        float* sink = dalloc<float>(1280 * 128);
        int* isink = reinterpret_cast<int*>(sink);
        using Launch = std::function<void(unsigned, int)>;
        // ops per wave per iteration, launcher(grid, iters)
        auto sweep = [&](const char* name, double ops, const Launch& go) {
            for (unsigned grid : {20U, 40U, 80U, 160U, 320U, 640U, 1280U}) {
                go(grid, 64);  // warm
                hip_ok(hipDeviceSynchronize(), "warm");
                const double probe = timed_ms([&] { go(grid, 256); });
                const int iters = static_cast<int>(std::clamp(256.0 * 100.0 / std::max(probe, 1e-3), 256.0, 4.0e7));
                const double ms = timed_ms([&] { go(grid, iters); });
                std::printf("ceiling %-12s grid %5u x128 thr  iters %9d  %8.2f ms  %7.2f T/s\n", name, grid, iters, ms,
                            ops * iters * grid * 4.0 / ms / 1e9);
                std::fflush(stdout);
            }
        };
        const float fs = 1.0e-3F;
        sweep("f16 nacc1", 8192.0, [&](unsigned g, int it) { hipLaunchKernelGGL(wmma_f16_ceiling_kernel<1>, dim3(g), dim3(128), 0, nullptr, it, fs, sink); });
        sweep("f16 nacc4", 4 * 8192.0, [&](unsigned g, int it) { hipLaunchKernelGGL(wmma_f16_ceiling_kernel<4>, dim3(g), dim3(128), 0, nullptr, it, fs, sink); });
        sweep("f16 nacc8", 8 * 8192.0, [&](unsigned g, int it) { hipLaunchKernelGGL(wmma_f16_ceiling_kernel<8>, dim3(g), dim3(128), 0, nullptr, it, fs, sink); });
        sweep("iu8 nacc8", 8 * 8192.0, [&](unsigned g, int it) { hipLaunchKernelGGL(wmma_iu8_ceiling_kernel<8>, dim3(g), dim3(128), 0, nullptr, it, 1, isink); });
        // Sustained: 30 back-to-back launches of ~100 ms (f16 nacc8, grid 640), each timed.
        {
            const double probe = timed_ms([&] { hipLaunchKernelGGL(wmma_f16_ceiling_kernel<8>, dim3(640), dim3(128), 0, nullptr, 256, fs, sink); });
            const int iters = static_cast<int>(256.0 * 100.0 / std::max(probe, 1e-3));
            std::vector<hipEvent_t> ev(31);
            for (auto& e : ev) hip_ok(hipEventCreate(&e), "ev");
            hip_ok(hipEventRecord(ev[0], nullptr), "rec");
            for (std::size_t r = 0; r < 30; ++r) {
                hipLaunchKernelGGL(wmma_f16_ceiling_kernel<8>, dim3(640), dim3(128), 0, nullptr, iters, fs, sink);
                hip_ok(hipEventRecord(ev[r + 1], nullptr), "rec");
            }
            hip_ok(hipEventSynchronize(ev[30]), "sync");
            std::printf("ceiling sustained f16 nacc8 grid 640 iters %d, T/s per ~100 ms launch:", iters);
            float total = 0;
            for (std::size_t r = 0; r < 30; ++r) {
                float ms = 0;
                hip_ok(hipEventElapsedTime(&ms, ev[r], ev[r + 1]), "elapsed");
                total += ms;
                std::printf(" %.1f", 8 * 8192.0 * iters * 640 * 4.0 / ms / 1e9);
            }
            std::printf("  (%.2f s)\n", total / 1e3);
            std::fflush(stdout);
            for (auto e : ev) (void)hipEventDestroy(e);
        }
        // Dequant ceiling: B fragments of 32 tile columns x 64 k-tiles (<= 96 KB, cache resident), 512 weights per wave-tile.
        {
            std::vector<std::uint32_t> words(64 * 32 * 64);
            std::uint64_t sm = 99;
            for (auto& w : words) w = static_cast<std::uint32_t>(splitmix(sm));
            std::uint32_t* tr = dupload(words);
            auto dsweep = [&](const char* name, int bits, const Launch& go) {
                for (unsigned grid : {80U, 320U, 1280U}) {
                    go(grid, 2);
                    hip_ok(hipDeviceSynchronize(), "warm");
                    const double probe = timed_ms([&] { go(grid, 8); });
                    const int iters = static_cast<int>(std::clamp(8.0 * 100.0 / std::max(probe, 1e-3), 8.0, 1.0e6));
                    const double ms = timed_ms([&] { go(grid, iters); });
                    std::printf("ceiling %-12s grid %5u x128 thr  iters %9d  %8.2f ms  %7.2f G weights/s\n", name, grid, iters, ms,
                                512.0 * 64 * iters * grid * 4.0 / ms / 1e6);
                    std::fflush(stdout);
                }
                (void)bits;
            };
            auto* us = reinterpret_cast<std::uint32_t*>(sink);
            dsweep("decode 3bit", 3, [&](unsigned g, int it) { hipLaunchKernelGGL(decode_ceiling_kernel<3>, dim3(g), dim3(128), 0, nullptr, tr, 64, it, us); });
            dsweep("decode 6bit", 6, [&](unsigned g, int it) { hipLaunchKernelGGL(decode_ceiling_kernel<6>, dim3(g), dim3(128), 0, nullptr, tr, 64, it, us); });
            hip_ok(hipFree(tr), "free");
        }
        hip_ok(hipFree(sink), "free");
    }

    // --- timing ----------------------------------------------------------------------------------------------------
    double time_us(const std::function<void(int)>& fn, int iters) {
        hipEvent_t a, b;
        hip_ok(hipEventCreate(&a), "ev");
        hip_ok(hipEventCreate(&b), "ev");
        fn(-1);  // warm (code, workspace); weights stay cold through the rotation
        hip_ok(hipEventRecord(a, nullptr), "rec");
        for (int i = 0; i < iters; ++i) fn(i);
        hip_ok(hipEventRecord(b, nullptr), "rec");
        hip_ok(hipEventSynchronize(b), "sync");
        float ms = 0;
        hip_ok(hipEventElapsedTime(&ms, a, b), "elapsed");
        (void)hipEventDestroy(a);
        (void)hipEventDestroy(b);
        return 1000.0 * ms / iters;
    }
    void aba(const std::string& cell, std::uint64_t bytes, const std::function<void(int)>& fa, const std::function<void(int)>& fb) {
        std::vector<double> A, B;
        const auto reps = static_cast<std::size_t>(o.reps);
        for (std::size_t r = 0; r < reps; ++r) {
            if (r == 0) A.push_back(time_us(fa, o.iters));
            B.push_back(time_us(fb, o.iters));
            A.push_back(time_us(fa, o.iters));
        }
        std::string trip;
        for (std::size_t r = 0; r < reps; ++r) {
            char buf[32];
            std::snprintf(buf, sizeof buf, " %.3f", B[r] / (0.5 * (A[r] + A[r + 1])));
            trip += buf;
        }
        auto med = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        const double ma = med(A), mb = med(B), gb = static_cast<double>(bytes);
        std::printf("cell %-22s A %9.1f us (%6.1f GB/s)  B %9.1f us (%6.1f GB/s)  B/A %.3f  triplets%s  A:", cell.c_str(), ma,
                    gb / ma / 1e3, mb, gb / mb / 1e3, mb / ma, trip.c_str());
        for (double v : A) std::printf(" %.1f", v);
        std::printf(" B:");
        for (double v : B) std::printf(" %.1f", v);
        std::printf("\n");
        std::fflush(stdout);
    }
    bool wanted(const std::string& cell) const { return o.only.empty() || cell.find(o.only) != std::string::npos; }

    void bench_moe() {
        constexpr std::size_t kBatches = 24;
        for (unsigned n = 1; n <= 8; ++n) {
            const std::string cell = "moe n" + std::to_string(n);
            if (!wanted(cell)) continue;
            std::vector<Batch> bs;
            std::uint64_t bytes = 0;
            for (std::size_t i = 0; i < kBatches; ++i) {
                bs.push_back(make_batch(n, 0));
                bytes += moe_bytes(bs.back());
            }
            bytes /= kBatches;
            aba(cell, bytes, [&](int i) { run_A(bs[static_cast<std::size_t>(i < 0 ? 0 : i) % kBatches]); },
                [&](int i) { run_B(bs[static_cast<std::size_t>(i < 0 ? 0 : i) % kBatches]); });
            // Router alone: post-exchange with router (A) vs without + router kernel (B).
            if (n == 1 || n == 8)
                aba("router n" + std::to_string(n), std::uint64_t{opt::kExperts} * H * 4,
                    [&](int i) {
                        const Batch& bt = bs[static_cast<std::size_t>(i < 0 ? 0 : i) % kBatches];
                        mref::hip::enqueue_post_exchange({bt.n, h, bt.p0, bt.p1, bt.nw, y, L.router_d, L.bias_d, idsA, wA}, nullptr);
                    },
                    [&](int i) {
                        const Batch& bt = bs[static_cast<std::size_t>(i < 0 ? 0 : i) % kBatches];
                        if (fused_post()) {
                            opt::MoeView v = view_B(bt.n);
                            v.h = h;
                            v.part0 = bt.p0;
                            v.part1 = bt.p1;
                            v.norm_w = bt.nw;
                            opt::enqueue_router(v, nullptr);
                        } else {
                            mref::hip::enqueue_post_exchange({bt.n, h, bt.p0, bt.p1, bt.nw, y, nullptr, nullptr, nullptr, nullptr}, nullptr);
                            opt::enqueue_router(view_B(bt.n), nullptr);
                        }
                    });
            for (auto& bt : bs) free_batch(bt);
        }
    }
    // Per-kernel breakdown of B (event marks between the launches of each call; rotating cold batches).
    void profile_moe() {
        constexpr std::size_t kBatches = 24;
        for (unsigned n : {1U, 2U, 4U, 8U}) {
            if (!wanted("parts n" + std::to_string(n))) continue;
            std::vector<Batch> bs;
            for (std::size_t i = 0; i < kBatches; ++i) bs.push_back(make_batch(n, 0));
            constexpr int kEv = 6;
            std::vector<hipEvent_t> ev(static_cast<std::size_t>(o.iters) * kEv);
            for (auto& e : ev) hip_ok(hipEventCreate(&e), "ev");
            double sum[kEv - 1] = {};
            for (int rep = 0; rep < 2; ++rep) {
                for (int i = 0; i < o.iters; ++i) {
                    const Batch& bt = bs[static_cast<std::size_t>(i) % kBatches];
                    hipEvent_t* e = &ev[static_cast<std::size_t>(i) * kEv];
                    hip_ok(hipMemcpyAsync(h, bt.h0, std::size_t{bt.n} * H * 4, hipMemcpyDeviceToDevice, nullptr), "h");
                    hip_ok(hipEventRecord(e[0], nullptr), "rec");
                    mref::hip::enqueue_post_exchange({bt.n, h, bt.p0, bt.p1, bt.nw, y, nullptr, nullptr, nullptr, nullptr}, nullptr);
                    hip_ok(hipEventRecord(e[1], nullptr), "rec");
                    opt::enqueue_router(view_B(bt.n), nullptr);
                    hip_ok(hipEventRecord(e[2], nullptr), "rec");
                    hipEvent_t marks[2] = {e[3], e[4]};
                    opt::enqueue_experts(view_B(bt.n), nullptr, marks);
                    hip_ok(hipEventRecord(e[5], nullptr), "rec");
                }
                hip_ok(hipDeviceSynchronize(), "sync");
                if (rep == 0) continue;  // first pass warms code and workspaces
                for (int i = 0; i < o.iters; ++i)
                    for (int k = 0; k + 1 < kEv; ++k) {
                        float ms = 0;
                        hip_ok(hipEventElapsedTime(&ms, ev[static_cast<std::size_t>(i * kEv + k)], ev[static_cast<std::size_t>(i * kEv + k + 1)]), "el");
                        sum[k] += 1000.0 * ms / o.iters;
                    }
            }
            std::printf("parts n%u (us, B, event marks): post_exchange %.1f router %.1f prepare %.1f gate_up %.1f down %.1f\n", n,
                        sum[0], sum[1], sum[2], sum[3], sum[4]);
            for (auto& e : ev) (void)hipEventDestroy(e);
            for (auto& bt : bs) free_batch(bt);
        }
    }
    // Size-matched read ceiling per dense site: the same trellis buffers (rotating copies) read by read_kernel at a
    // few grid sizes; the best grid is the ceiling a GEMV of that matrix could reach in one launch.
    static bool baseline_linears() {  // ceiling / lparts only in the default (two-launch) arm
        const char* e = std::getenv("STRIX_MMX_LINEAR_V");
        return e == nullptr || std::atoi(e) == 4 || std::atoi(e) == 1;
    }
    void bench_ceiling() {
        if (!baseline_linears()) return;
        unsigned* guard = dalloc<unsigned>(1);
        for (auto& g : dense_groups()) {
            const std::string cell = "ceiling " + group_name(g);
            if (!wanted(cell)) continue;
            std::uint64_t bytes = 0;
            std::size_t copies = 1;
            for (auto* ds : g) {
                bytes += ds->copies[0].bytes;
                copies = std::max(copies, ds->copies.size());
            }
            std::printf("%s (%.2f MB):", cell.c_str(), static_cast<double>(bytes) / 1e6);
            for (unsigned grid : {160U, 320U, 640U, 1280U, 2560U}) {
                const double us = time_us([&](int i) {
                    for (auto* ds : g) {
                        const Mat& m = ds->copies[static_cast<std::size_t>(i < 0 ? 0 : i) % ds->copies.size()];
                        const std::size_t words = static_cast<std::size_t>(m.dev.input / 16) * (m.dev.output / 16) * m.dev.bits * 32 / 16;
                        hipLaunchKernelGGL(read_kernel, dim3(grid), dim3(256), 0, nullptr, reinterpret_cast<const uint4*>(m.dev.trellis), words, guard);
                    }
                }, o.iters);
                std::printf("  grid %u %.1f us %.1f GB/s", grid, us, static_cast<double>(bytes) / us / 1e3);
            }
            std::printf("\n");
            std::fflush(stdout);
        }
        hip_ok(hipFree(guard), "free");
    }
    // Per-launch split of B for the two-launch forms (event after prepare): prepare, core (+ finish), and the whole call.
    void profile_dense() {
        if (!baseline_linears()) return;
        for (auto& g : dense_groups())
            for (unsigned n : {1U, 8U}) {
                const std::string cell = "lparts " + group_name(g) + " n" + std::to_string(n);
                if (!wanted(cell)) continue;
                const auto x = randn(std::size_t{n} * 6144, 1.0F);
                hip_ok(hipMemcpy(xin, x.data(), x.size() * 4, hipMemcpyHostToDevice), "x");
                std::vector<hipEvent_t> ev(static_cast<std::size_t>(o.iters) * 3);
                for (auto& e : ev) hip_ok(hipEventCreate(&e), "ev");
                double pre = 0, core = 0;
                for (int rep = 0; rep < 2; ++rep) {
                    for (int i = 0; i < o.iters; ++i) {
                        std::vector<opt::LinearJob> js;
                        std::size_t off = 0;
                        for (auto* ds : g) {
                            const Mat& m = ds->copies[static_cast<std::size_t>(i) % ds->copies.size()];
                            js.push_back({m.dev, xin, ds->input, yB + off, ds->output});
                            off += std::size_t{n} * ds->output;
                        }
                        hipEvent_t* e = &ev[static_cast<std::size_t>(i) * 3];
                        hip_ok(hipEventRecord(e[0], nullptr), "rec");
                        opt::enqueue_linears(js.data(), static_cast<unsigned>(js.size()), n, lin_ws, nullptr, &e[1]);
                        hip_ok(hipEventRecord(e[2], nullptr), "rec");
                    }
                    hip_ok(hipDeviceSynchronize(), "sync");
                    if (rep == 0) continue;
                    for (int i = 0; i < o.iters; ++i) {
                        float a = 0, b = 0;
                        const auto k = static_cast<std::size_t>(i) * 3;
                        hip_ok(hipEventElapsedTime(&a, ev[k], ev[k + 1]), "el");
                        hip_ok(hipEventElapsedTime(&b, ev[k + 1], ev[k + 2]), "el");
                        pre += 1000.0 * a / o.iters;
                        core += 1000.0 * b / o.iters;
                    }
                }
                std::uint64_t bytes = 0;
                for (auto* ds : g) bytes += ds->copies[0].bytes;
                std::printf("%s (us, B, event marks, isolated calls): prepare %.1f core+finish %.1f (%.1f GB/s) total %.1f (%.1f GB/s)\n",
                            cell.c_str(), pre, core, static_cast<double>(bytes) / core / 1e3, pre + core,
                            static_cast<double>(bytes) / (pre + core) / 1e3);
                std::fflush(stdout);
                for (auto& e : ev) (void)hipEventDestroy(e);
            }
    }
    // Per-CTA timeline of one cold call of the v4/v7 core (set_linear_trace): span, phase means, residency. CSV per cell.
    void trace_dense() {
        int khz = 0;
        hip_ok(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0), "wallclock");
        const double tick_us = 1000.0 / khz;
        constexpr std::size_t kCtas = 8192;
        auto* tr = dalloc<std::uint64_t>(kCtas * 5);
        const char* ve = std::getenv("STRIX_MMX_LINEAR_V");
        const std::string ver = ve != nullptr ? ve : "4";
        for (auto& g : dense_groups()) {
            if (g.size() == 1 && g[0]->name == "head") continue;
            for (unsigned n : {1U, 8U}) {
                const std::string cell = "trace " + group_name(g) + " n" + std::to_string(n);
                if (!wanted(cell)) continue;
                const auto x = randn(std::size_t{n} * 6144, 1.0F);
                hip_ok(hipMemcpy(xin, x.data(), x.size() * 4, hipMemcpyHostToDevice), "x");
                std::vector<std::pair<DenseSite*, Mat*>> warm, cold;
                for (auto* ds : g) {
                    warm.push_back({ds, &ds->copies[0]});
                    cold.push_back({ds, &ds->copies[1 % ds->copies.size()]});
                }
                run_dense_B(warm, n, xin);
                hip_ok(hipMemset(tr, 0, kCtas * 5 * 8), "tr0");
                hip_ok(hipDeviceSynchronize(), "warm");
                // steady state: the 6th of 8 back-to-back calls on rotating cold copies is traced
                for (int i = 0; i < 8; ++i) {
                    std::vector<std::pair<DenseSite*, Mat*>> jobs;
                    for (auto* ds : g) jobs.push_back({ds, &ds->copies[static_cast<std::size_t>(i + 1) % ds->copies.size()]});
                    if (i == 5) opt::set_linear_trace(tr);
                    run_dense_B(jobs, n, xin);
                    if (i == 5) opt::set_linear_trace(nullptr);
                }
                hip_ok(hipDeviceSynchronize(), "traced");
                const auto t = d2h(tr, kCtas * 5);
                std::size_t ctas = 0;
                while (ctas < kCtas && t[ctas * 5] != 0) ++ctas;
                std::uint64_t t0 = UINT64_MAX, t3 = 0;
                double ph1 = 0, ph2 = 0, ph3 = 0;
                for (std::size_t c = 0; c < ctas; ++c) {
                    t0 = std::min(t0, t[c * 5]);
                    t3 = std::max(t3, t[c * 5 + 3]);
                    ph1 += static_cast<double>(t[c * 5 + 1] - t[c * 5]);
                    ph2 += static_cast<double>(t[c * 5 + 2] - t[c * 5 + 1]);
                    ph3 += static_cast<double>(t[c * 5 + 3] - t[c * 5 + 2]);
                }
                // residency: max CTAs alive at once (sweep over start/end events)
                std::vector<std::pair<std::uint64_t, int>> evs;
                for (std::size_t c = 0; c < ctas; ++c) {
                    evs.push_back({t[c * 5], 1});
                    evs.push_back({t[c * 5 + 3], -1});
                }
                std::sort(evs.begin(), evs.end());
                int live = 0, peak = 0;
                for (auto& e : evs) peak = std::max(peak, live += e.second);
                const double c = static_cast<double>(std::max<std::size_t>(ctas, 1));
                std::printf("%s v%s: %zu CTAs, span %.1f us, mean per CTA: inputs %.2f us, split %.2f us, tail %.2f us; peak resident %d\n",
                            cell.c_str(), ver.c_str(), ctas, static_cast<double>(t3 - t0) * tick_us, ph1 / c * tick_us,
                            ph2 / c * tick_us, ph3 / c * tick_us, peak);
                std::string fn = "trace-" + group_name(g) + "-n" + std::to_string(n) + "-v" + ver + ".csv";
                for (auto& ch : fn)
                    if (ch == '+') ch = '_';
                if (FILE* f = std::fopen(fn.c_str(), "w")) {
                    std::fprintf(f, "cta,t0,t1,t2,t3,hwid\n");
                    for (std::size_t k = 0; k < ctas; ++k)
                        std::fprintf(f, "%zu,%llu,%llu,%llu,%llu,%llu\n", k, static_cast<unsigned long long>(t[k * 5] - t0),
                                     static_cast<unsigned long long>(t[k * 5 + 1] - t0), static_cast<unsigned long long>(t[k * 5 + 2] - t0),
                                     static_cast<unsigned long long>(t[k * 5 + 3] - t0), static_cast<unsigned long long>(t[k * 5 + 4]));
                    std::fclose(f);
                }
                std::fflush(stdout);
            }
        }
        hip_ok(hipFree(tr), "free");
    }
    // Load-only weight streams (pattern_kernel) per site: the v4 grid (P = all chains) and persistent grids of P waves.
    void bench_pattern() {
        unsigned* guard = dalloc<unsigned>(1);
        for (auto& g : dense_groups()) {
            if (g.size() != 1) continue;
            DenseSite* ds = g[0];
            const std::string cell = "pattern " + ds->name;
            if (!wanted(cell)) continue;
            const Mat& m0 = ds->copies[0];
            const unsigned splits = std::min(strix::minimax::moe::opt::dev::vector_splits(m0.dev.input, m0.dev.output), m0.dev.input / 16);
            const unsigned total = (m0.dev.output / 16) * splits;
            std::printf("%s (%.2f MB, %u chains of %u tiles):", cell.c_str(), static_cast<double>(m0.bytes) / 1e6, total,
                        m0.dev.input / 16 / splits);
            for (unsigned P : {total, 768U, 1024U}) {
                if (P > total) continue;
                for (int depth : {4, 14, 24, 34, 44}) {  // layout 0 / 1 / 2, wide 3 / 4
                    const double us = time_us([&](int i) {
                        const Mat& m = ds->copies[static_cast<std::size_t>(i < 0 ? 0 : i) % ds->copies.size()];
                        const dim3 grid((P + 3) / 4);
                        auto launch = [&](auto kern) {
                            hipLaunchKernelGGL(kern, grid, dim3(128), 0, nullptr, m.dev.trellis, m.dev.output, m.dev.input, splits, guard);
                        };
                        switch (static_cast<int>(m.dev.bits) * 100 + depth) {
                            case 304: launch(pattern_kernel<3, 4>); break;
                            case 314: launch(pattern_kernel<3, 4, 1>); break;
                            case 324: launch(pattern_kernel<3, 4, 2>); break;
                            case 504: launch(pattern_kernel<5, 4>); break;
                            case 514: launch(pattern_kernel<5, 4, 1>); break;
                            case 524: launch(pattern_kernel<5, 4, 2>); break;
                            case 604: launch(pattern_kernel<6, 4>); break;
                            case 614: launch(pattern_kernel<6, 4, 1>); break;
                            case 624: launch(pattern_kernel<6, 4, 2>); break;
                            case 804: launch(pattern_kernel<8, 4>); break;
                            case 814: launch(pattern_kernel<8, 4, 1>); break;
                            case 824: launch(pattern_kernel<8, 4, 2>); break;
                            case 334: launch(pattern_wide_kernel<3, 4, 3>); break;
                            case 344: launch(pattern_wide_kernel<3, 4, 4>); break;
                            case 534: launch(pattern_wide_kernel<5, 4, 3>); break;
                            case 544: launch(pattern_wide_kernel<5, 4, 4>); break;
                            case 634: launch(pattern_wide_kernel<6, 4, 3>); break;
                            case 644: launch(pattern_wide_kernel<6, 4, 4>); break;
                            case 834: launch(pattern_wide_kernel<8, 4, 3>); break;
                            default: launch(pattern_wide_kernel<8, 4, 4>); break;
                        }
                    }, o.iters);
                    std::printf("  P%u L%d %.1f us %.1f GB/s", P, depth / 10, us, static_cast<double>(m0.bytes) / us / 1e3);
                }
            }
            for (unsigned C : {4U, 16U, 32U, 64U}) {
                if ((m0.dev.output / 16) % C) continue;
                for (int depth : {4, 8}) {
                    const double us = time_us([&](int i) {
                        const Mat& m = ds->copies[static_cast<std::size_t>(i < 0 ? 0 : i) % ds->copies.size()];
                        const dim3 grid((m.dev.output / 16 / C) * splits);
                        auto launch = [&](auto kern) {
                            hipLaunchKernelGGL(kern, grid, dim3(128), 0, nullptr, m.dev.trellis, m.dev.output, m.dev.input, splits, guard);
                        };
#define PB(B) \
    if (m.dev.bits == B) { \
        if (C == 4) depth == 4 ? launch(pattern_block_kernel<B, 4, 4>) : launch(pattern_block_kernel<B, 4, 8>); \
        else if (C == 16) depth == 4 ? launch(pattern_block_kernel<B, 16, 4>) : launch(pattern_block_kernel<B, 16, 8>); \
        else if (C == 32) depth == 4 ? launch(pattern_block_kernel<B, 32, 4>) : launch(pattern_block_kernel<B, 32, 8>); \
        else depth == 4 ? launch(pattern_block_kernel<B, 64, 4>) : launch(pattern_block_kernel<B, 64, 8>); \
    }
                        PB(3) PB(5) PB(6) PB(8)
#undef PB
                    }, o.iters);
                    std::printf("  C%u d%d %.1f us %.1f GB/s", C, depth, us, static_cast<double>(m0.bytes) / us / 1e3);
                }
            }
            std::printf("\n");
            std::fflush(stdout);
        }
        hip_ok(hipFree(guard), "free");
    }
    void bench_dense() {
        for (auto& g : dense_groups())
            for (unsigned n = 1; n <= 8; ++n) {
                const std::string cell = group_name(g) + " n" + std::to_string(n);
                if (!wanted(cell)) continue;
                std::uint64_t bytes = 0;
                std::size_t copies = 1;
                for (auto* ds : g) {
                    bytes += ds->copies[0].bytes;
                    copies = std::max(copies, ds->copies.size());
                }
                const auto x = randn(std::size_t{n} * 6144, 1.0F);
                hip_ok(hipMemcpy(xin, x.data(), x.size() * 4, hipMemcpyHostToDevice), "x");
                auto jobs_at = [&](int i) {
                    std::vector<std::pair<DenseSite*, Mat*>> jobs;
                    for (auto* ds : g) jobs.push_back({ds, &ds->copies[static_cast<std::size_t>(i < 0 ? 0 : i) % ds->copies.size()]});
                    return jobs;
                };
                aba(cell, bytes, [&](int i) { run_dense_A(jobs_at(i), n, xin); }, [&](int i) { run_dense_B(jobs_at(i), n, xin); });
            }
    }
};

}  // namespace

int main(int argc, char** argv) {
    try {
        Options o;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
                return argv[++i];
            };
            if (a == "--test") o.test = true;
            else if (a == "--bench") o.bench = true;
            else if (a == "--reps") o.reps = std::stoi(next());
            else if (a == "--iters") o.iters = std::stoi(next());
            else if (a == "--only") o.only = next();
            else if (a == "--no-moe") o.no_moe = true;
            else if (a == "--model") o.model = next();
            else if (a == "--layer") o.layer = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--rank") o.rank = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--routes") o.routes = next();
            else if (a == "--tests") { o.tests = next(); o.test = true; }
            else if (a == "--no-load") o.no_load = true;
            else if (a == "--syn-bits") o.syn_bits = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--diverse") o.diverse = true;
            else if (a == "--diverse-fallback") { o.diverse = true; o.diverse_fallback = true; }
            else if (a == "--singletons") o.singletons = next();
            else if (a == "--ab-down") { o.diverse = true; o.ab_down = true; }
            else throw std::invalid_argument("unknown option " + a);
        }
        if (o.reps < 1 || o.iters < 1 || o.reps > 1000 || o.iters > 10000)
            throw std::invalid_argument("reps/iters must be positive (maximum 1000/10000)");
        if (o.rank > 1) throw std::invalid_argument("rank must be 0 or 1");
        if (!o.test && !o.bench) o.test = o.bench = true;
        hipDeviceProp_t prop{};
        hip_ok(hipGetDeviceProperties(&prop, 0), "props");
        std::printf("device %s, %d CUs; moe workspace %zu B\n", prop.gcnArchName, prop.multiProcessorCount, opt::moe_workspace_bytes());
        Bench b;
        b.o = o;
        if (o.no_load) {  // ceiling only: no weights, no other section
            b.bench_wmma_ceiling();
            return 0;
        }
        b.load();
        if (o.diverse) {
            // Fail immediately: never print benchmark results after a failed exactness gate.
            if (o.test) b.test_diverse();
            if (o.bench) b.bench_diverse();
            return 0;
        }
        // Each section runs on its own: a library error in one is reported and the next still runs.
        auto section = [&](const char* name, const std::function<void()>& fn) {
            try {
                fn();
            } catch (const std::exception& e) {
                ++g_fail;
                std::printf("SECTION %s ERROR: %s\n", name, e.what());
            }
            std::fflush(stdout);
        };
        auto test_section = [&](const char* name, const std::function<void()>& fn) {
            if (o.tests.empty() || std::string(name).find(o.tests) != std::string::npos) section(name, fn);
        };
        if (o.test) {
            test_section("test_dense", [&] { b.test_dense(); });
            if (!o.no_moe) test_section("test_moe", [&] { b.test_moe(); });
            if (!o.no_moe) test_section("test_diverse", [&] { b.test_diverse(); });
            if (!o.no_moe) test_section("test_router_rows", [&] { b.test_router_rows(); });
            if (!o.no_moe) test_section("test_prefill", [&] { b.test_prefill(); });
            if (!o.no_moe) test_section("test_prefill_v2", [&] { b.test_prefill_v2(); });
            if (!o.no_moe) test_section("test_prefill_mtfit", [&] { b.test_prefill_mtfit(); });
            if (!o.no_moe) test_section("test_prefill_core8", [&] { b.test_prefill_core8(); });
            if (!o.no_moe && o.tests == "test_prefill_pipe") test_section("test_prefill_pipe", [&] { b.test_prefill_pipe(); });
            if (!o.no_moe) test_section("test_prefill_gufuse", [&] { b.test_prefill_gufuse(); });
            if (!o.no_moe) test_section("test_prefill_table_ring", [&] { b.test_prefill_table_ring(); });
            if (!o.no_moe) test_section("test_prefill_gather", [&] { b.test_prefill_gather(); });
            test_section("test_prefill_dense", [&] { b.test_prefill_dense(); });
            if (g_fail == 0) std::printf("TESTS PASS\n");
            else std::printf("TESTS FAIL (%d failures)\n", g_fail);
            std::fflush(stdout);
        }
        if (o.bench) {
            if (!o.no_moe) section("bench_moe", [&] { b.bench_moe(); });
            if (!o.no_moe) section("profile_moe", [&] { b.profile_moe(); });
            if (!o.no_moe) section("bench_router_rows", [&] { b.bench_router_rows(); });
            if (!o.no_moe) section("bench_prefill", [&] { b.bench_prefill(); });
            if (!o.no_moe) section("bench_prefill_gufuse", [&] { b.bench_prefill_gufuse(); });
            if (!o.no_moe) section("bench_prefill_v2", [&] { b.bench_prefill_v2(); });
            if (!o.no_moe) section("bench_prefill_gather", [&] { b.bench_prefill_gather(); });
            section("bench_prefill_dense", [&] { b.bench_prefill_dense(); });
            section("bench_dense_prefill", [&] { b.bench_dense_prefill(); });
            section("bench_dense", [&] { b.bench_dense(); });
            section("bench_ceiling", [&] { b.bench_ceiling(); });
            section("profile_dense", [&] { b.profile_dense(); });
            section("trace_dense", [&] { b.trace_dense(); });
            section("bench_pattern", [&] { b.bench_pattern(); });
        }
        return g_fail == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
