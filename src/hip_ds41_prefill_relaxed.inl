// FAST lever ds41-moe-relaxed-v1 (STRIX_DS41_FAST_MOE=1, default off; prefill routed experts only). NOT bit-exact: the
// same A8 activations, the same decoded weight integers and the same exact int32 group dots (WMMA iu8) as the split
// kernels, but each output's per-group terms are added in ascending group order with one FMA per term (Q2_K: the scaled
// and min terms in one FMA too) instead of the declared 32-class butterfly. That frees the class-wave tree, so a block
// takes up to 64 rows of one expert (4 row tiles share every decoded weight group) and runs two barriers per block of
// 256 weights. SwiGLU and the A8 of the activations are today's code on the new sums.
// Included by src/hip_ds41_prefill.hip after the split kernels (uses their decoders, staging layout and helpers).
constexpr unsigned relaxed_rows = 64U, relaxed_tiles = relaxed_rows / 16U, relaxed_threads = 256U, relaxed_chunk_groups = 8U;
constexpr unsigned relaxed_code_bytes = relaxed_chunk_groups * 2U * relaxed_rows * 16U;   // 16 KB
constexpr unsigned relaxed_delta_bytes = relaxed_chunk_groups * relaxed_rows * 4U;        // 2 KB
constexpr unsigned relaxed_stage_bytes = relaxed_code_bytes + 2U * relaxed_delta_bytes;   // + Q2_K half sums
constexpr unsigned relaxed_values_bytes = relaxed_rows * 128U * 4U;                       // 32 KB (gate/up epilogue)
constexpr unsigned relaxed_lds_bytes = relaxed_values_bytes > relaxed_stage_bytes ? relaxed_values_bytes : relaxed_stage_bytes;
// Q2_K (down): the block's 256 weight rows of the chunk (one 84-byte block each, dword aligned) are read with coalesced
// dword loads into an LDS tile one chunk ahead, so a single copy of the next chunk's weights is in registers (two
// per-lane register copies spilled ~235 VGPRs).
constexpr unsigned relaxed_tile_words = 256U * static_cast<unsigned>(q2_k_block_bytes) / 4U, relaxed_tile_loads = relaxed_tile_words / relaxed_threads;
constexpr unsigned relaxed_down_lds_bytes = relaxed_stage_bytes + relaxed_tile_words * 4U;

// The block's rows: route batches of one expert grouped greedily from the expert's first batch, as many consecutive
// batches as fit in 64 rows; block i (batch i) runs the group that starts at batch i, else it has no work. Any batch
// layout the views accept is covered exactly once (no assumption that batches are full or contiguous).
struct RelaxedRows { unsigned expert, first, batches, count; };
template <unsigned R = relaxed_rows>
__device__ __forceinline__ bool relaxed_rows_of(const PrefillRoutesView& routes, RelaxedRows& b, unsigned i) {
    const unsigned total = *routes.batch_total;
    if (i >= total) return false;
    const unsigned e = routes.batch_expert[i];
    unsigned start = i;
    while (start > 0U && routes.batch_expert[start - 1U] == e) --start;
    for (;;) {
        unsigned c = 0, j = start;
        while (j < total && routes.batch_expert[j] == e && c + routes.batch_count[j] <= R) c += routes.batch_count[j++];
        if (j == start) c = routes.batch_count[j++];   // a batch above 64 rows (not produced by the builders: <= 32)
        if (start == i) { b = RelaxedRows{e, i, j - i, c > R ? R : c}; return true; }
        if (j > i) return false;
        start = j;
    }
}
// Entry of the group's row r (r < count)
__device__ __forceinline__ unsigned relaxed_entry(const PrefillRoutesView& routes, const RelaxedRows& b, unsigned r) {
    for (unsigned k = b.first; k < b.first + b.batches; ++k) {
        if (r < routes.batch_count[k]) return routes.batch_begin[k] + r;
        r -= routes.batch_count[k];
    }
    return routes.batch_begin[b.first];   // unreachable for r < count
}
// Activation items of one chunk (8 groups x 2 halves x 64 rows = 1024; 4 per thread) in registers, then into LDS.
// R rows: 8 groups x 2 halves x R rows = 16 R items, R / 16 per thread
template <unsigned R = relaxed_rows> struct RelaxedItems { v4i v[R / 16U]; float d[R / 16U]; };
template <unsigned R = relaxed_rows>
__device__ __forceinline__ RelaxedItems<R> relaxed_load(unsigned chunk, unsigned width, const unsigned* row_of, unsigned count,
                                                        const signed char* codes, const float* deltas) {
    const unsigned groups = width / 32U;
    RelaxedItems<R> x{};
#pragma unroll
    for (unsigned i = 0; i < R / 16U; ++i) {
        const unsigned item = threadIdx.x + relaxed_threads * i;
        const unsigned r = item % R, kh = (item / R) % 2U, gg = item / (2U * R);
        const unsigned g = relaxed_chunk_groups * chunk + gg;
        x.v[i] = v4i{0, 0, 0, 0};
        if (r < count) x.v[i] = *reinterpret_cast<const v4i*>(codes + static_cast<unsigned long long>(row_of[r]) * width + 32U * g + 16U * kh);
        x.d[i] = kh == 0U && r < count ? deltas[static_cast<unsigned long long>(row_of[r]) * groups + g] : 0.0F;
    }
    return x;
}
template <bool Q2, unsigned R = relaxed_rows>
__device__ __forceinline__ void relaxed_store(unsigned char* lds, const RelaxedItems<R>& x) {
    auto* a = reinterpret_cast<v4i*>(lds);
    auto* d = reinterpret_cast<float*>(lds + relaxed_code_bytes);
    auto* sums = reinterpret_cast<short*>(lds + relaxed_code_bytes + relaxed_delta_bytes);
#pragma unroll
    for (unsigned i = 0; i < R / 16U; ++i) {
        const unsigned item = threadIdx.x + relaxed_threads * i;
        const unsigned r = item % R, kh = (item / R) % 2U, gg = item / (2U * R);
        const v4i v = x.v[i];
        a[(2U * gg + kh) * R + r] = v;
        const unsigned at = tile_position(r);
        if (kh == 0U) d[gg * R + at] = x.d[i];
        if constexpr (Q2) {
            const int ones = 0x01010101;
            int sum = 0;
            sum = __builtin_amdgcn_sudot4(false, ones, true, v.x, sum, false);
            sum = __builtin_amdgcn_sudot4(false, ones, true, v.y, sum, false);
            sum = __builtin_amdgcn_sudot4(false, ones, true, v.z, sum, false);
            sum = __builtin_amdgcn_sudot4(false, ones, true, v.w, sum, false);
            sums[2U * (gg * R + at) + kh] = static_cast<short>(sum);
        }
    }
}
// One wave: its 32 weight rows (lane l decodes row l; the halves trade operands) x the block's row tiles over the whole
// width; acc[mt][t][e] = (row 16 mt + 2 e + h, output 16 t + lane % 16).
template <WeightFormat Format, unsigned R = relaxed_rows>
__device__ __forceinline__ void relaxed_products(unsigned char* lds, const unsigned char* weights_row, unsigned width, const unsigned* row_of,
                                                 unsigned count, const signed char* codes, const float* deltas, const unsigned long long* grid,
                                                 float (&acc)[R / 16U][2][8], const unsigned char* tile_base = nullptr,
                                                 std::uint64_t row_bytes = 0) {
    constexpr bool q2 = Format == WeightFormat::q2_k;
    constexpr bool wl = q2;   // weight tile in LDS
    constexpr unsigned group_unroll = wl ? 1U : relaxed_chunk_groups;
    constexpr unsigned row_words = static_cast<unsigned>(q2_k_block_bytes) / 4U;
    auto* tile = reinterpret_cast<unsigned*>(lds + relaxed_stage_bytes);
    unsigned tile_next[wl ? relaxed_tile_loads : 1];
    const auto load_tile = [&](unsigned chunk) {
#pragma unroll
        for (unsigned i = 0; i < relaxed_tile_loads; ++i) {
            const unsigned word = threadIdx.x + relaxed_threads * i, row = word / row_words, off = word % row_words;
            tile_next[i] = reinterpret_cast<const unsigned*>(tile_base + static_cast<std::uint64_t>(row) * row_bytes + chunk * q2_k_block_bytes)[off];
        }
    };
    const unsigned lane = threadIdx.x % 32U, i16 = lane & 15U, h = lane >> 4U, chunks = width / 256U;
    const bool upper = h != 0U;
    const unsigned tiles = (count + 15U) / 16U;
#pragma unroll
    for (unsigned mt = 0; mt < R / 16U; ++mt)
#pragma unroll
        for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) acc[mt][t][e] = 0.0F;
    ChunkRaw<Format> w_lo{}, w_hi{};
    if constexpr (wl) {
        (void)weights_row;
        load_tile(0U);
    } else {
        w_lo = load_chunk<Format>(weights_row, 0U);
        w_hi = load_chunk<Format>(weights_row, 4U);
    }
    RelaxedItems<R> items = relaxed_load<R>(0U, width, row_of, count, codes, deltas);
    const auto* a = reinterpret_cast<const v4i*>(lds);
    const auto* d = reinterpret_cast<const float*>(lds + relaxed_code_bytes);
    const auto* sums = reinterpret_cast<const int*>(lds + relaxed_code_bytes + relaxed_delta_bytes);
#pragma unroll 1
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
        split_sync<6>();   // the previous chunk's readers are done
        relaxed_store<q2, R>(lds, items);
        if constexpr (wl)
#pragma unroll
            for (unsigned i = 0; i < relaxed_tile_loads; ++i) tile[threadIdx.x + relaxed_threads * i] = tile_next[i];
        split_sync<6>();
        const ChunkRaw<Format> c_lo = w_lo, c_hi = w_hi;
        if (chunk + 1U < chunks) {
            items = relaxed_load<R>(chunk + 1U, width, row_of, count, codes, deltas);
            if constexpr (wl) load_tile(chunk + 1U);
            else {
                w_lo = load_chunk<Format>(weights_row, 8U * (chunk + 1U));
                w_hi = load_chunk<Format>(weights_row, 8U * (chunk + 1U) + 4U);
            }
        }
        // register raws (gate/up) need the groups unrolled; the LDS tile (down) reads by address: rolled, fewer live values
#pragma unroll group_unroll
        for (unsigned gg = 0; gg < relaxed_chunk_groups; ++gg) {
            Raw<Format> raw;
            if constexpr (wl) raw = load_raw<Format>(reinterpret_cast<const unsigned char*>(tile) + threadIdx.x * q2_k_block_bytes, gg);
            else raw = chunk_group<Format>(gg < 4U ? c_lo : c_hi, gg % 4U);
            const Operands own = decode_form<1, Format>(raw, grid, nullptr);
            const v4i x0 = other_half(own.b0), x1 = other_half(own.b1);
            const unsigned x_dm = q2 ? static_cast<unsigned>(other_half(static_cast<int>(own.dm_bits))) : 0U;
            const unsigned x_sc = q2 ? static_cast<unsigned>(other_half(static_cast<int>(own.sc_bits))) : 0U;
            const float x_scale = q2 ? 0.0F : other_half(own.scale);
#pragma unroll
            for (unsigned mt = 0; mt < R / 16U; ++mt) {
                if (mt >= tiles) break;
                const v4i a0 = a[(2U * gg) * R + 16U * mt + i16];
                const v4i a1 = a[(2U * gg + 1U) * R + 16U * mt + i16];
                const float* dt = d + gg * R + 16U * mt + 8U * h;
#pragma unroll
                for (unsigned t = 0; t < 2U; ++t) {
                    const bool mine = (t == 1U) == upper;
                    const v8i r = group_dot(a0, pick(mine, own.b0, x0), a1, pick(mine, own.b1, x1));
                    if constexpr (q2) {
                        const unsigned dmb = mine ? own.dm_bits : x_dm, scb = mine ? own.sc_bits : x_sc;
                        const float dd = half_bits(static_cast<unsigned short>(dmb & 0xffffU));
                        const float dm = half_bits(static_cast<unsigned short>(dmb >> 16U));
                        const int m0 = static_cast<int>((scb & 0xffU) >> 4U), m1 = static_cast<int>((scb >> 8U) >> 4U);
                        const int* sp = sums + gg * R + 16U * mt + 8U * h;
#pragma unroll
                        for (unsigned e = 0; e < 8U; ++e) {
                            const int packed = sp[e];
                            const int b = __mul24(m0, static_cast<int>(static_cast<short>(packed & 0xffff))) + __mul24(m1, packed >> 16);
                            const float term = __builtin_fmaf(dd, magic_float(r[e]), -(dm * static_cast<float>(b)));
                            acc[mt][t][e] = __builtin_fmaf(term, dt[e], acc[mt][t][e]);
                        }
                    } else {
                        const float scale = mine ? own.scale : x_scale;
#pragma unroll
                        for (unsigned e = 0; e < 8U; ++e) acc[mt][t][e] = __builtin_fmaf(magic_float(r[e]), scale * dt[e], acc[mt][t][e]);
                    }
                }
            }
        }
    }
}

// Gate/up: wave w = 2q + m (matrix m: gate, up; outputs 32 q .. of the block's 128).
// Swap (STRIX_DS41_FAST_MOE_ORDER, default 1): the grid's x walks the output blocks and y the batches, so the output blocks of one
// row group run together and share its activations and its expert's weights in the cache; only the block order changes.
template <WeightFormat Format, bool Swap>
__global__ __launch_bounds__(relaxed_threads) void relaxed_gate_up_kernel(PrefillGateUpView view) {
    const unsigned bx = Swap ? blockIdx.y : blockIdx.x, by = Swap ? blockIdx.x : blockIdx.y;
    __shared__ __attribute__((aligned(16))) unsigned char lds[relaxed_lds_bytes];
    __shared__ unsigned long long grid[Format == WeightFormat::iq2_xxs ? 256 : 1];
    __shared__ unsigned row_of[relaxed_rows], slot_of[relaxed_rows];
    RelaxedRows b;
    if (!relaxed_rows_of(view.routes, b, bx)) return;
    if constexpr (Format == WeightFormat::iq2_xxs) load_grid(grid);
    if (threadIdx.x < b.count) {
        const unsigned entry = relaxed_entry(view.routes, b, threadIdx.x);
        row_of[threadIdx.x] = view.routes.entry_rows[entry];
        slot_of[threadIdx.x] = view.routes.entry_slots[entry];
    }
    __syncthreads();
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U, m = wave & 1U, q = wave >> 1U, h = lane >> 4U, i16 = lane & 15U;
    const unsigned out0 = by * 128U;
    const auto* weights = static_cast<const unsigned char*>(m == 0U ? view.gate : view.up) + b.expert * view.expert_stride_bytes +
        static_cast<unsigned long long>(out0 + 32U * q + lane) * view.row_bytes;
    float acc[relaxed_tiles][2][8];
    relaxed_products<Format>(lds, weights, view.hidden, row_of, b.count, view.codes, view.deltas, grid, acc);
    auto* values = reinterpret_cast<float(*)[128]>(lds);
    __syncthreads();
    if (m == 1U)
#pragma unroll
        for (unsigned mt = 0; mt < relaxed_tiles; ++mt)
#pragma unroll
            for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) {
                    const unsigned r = 16U * mt + 2U * e + h;
                    if (r < b.count) values[r][32U * q + 16U * t + i16] = acc[mt][t][e];
                }
    __syncthreads();
    if (m == 0U)
#pragma unroll
        for (unsigned mt = 0; mt < relaxed_tiles; ++mt)
#pragma unroll
            for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) {
                    const unsigned r = 16U * mt + 2U * e + h, o = 32U * q + 16U * t + i16;
                    if (r >= b.count) continue;
                    const float g = acc[mt][t][e], u = values[r][o];
                    if (view.gate_out) {
                        view.gate_out[static_cast<unsigned long long>(slot_of[r]) * view.inter + out0 + o] = g;
                        view.up_out[static_cast<unsigned long long>(slot_of[r]) * view.inter + out0 + o] = u;
                    }
                    values[r][o] = swiglu_clamped(g, u, view.limit);
                }
    __syncthreads();
    // A8 of each (row, 32-output group): today's code
    const unsigned groups = view.inter / 32U;
    for (unsigned item = wave; item < b.count * 4U; item += relaxed_threads / 32U) {
        const unsigned r = item / 4U, gq = item % 4U;
        const unsigned long long slot = slot_of[r];
        const float act = values[r][32U * gq + lane];
        float amax = fabsf(act);
        for (int stride = 16; stride; stride /= 2) amax = fmaxf(amax, __shfl_xor(amax, stride, 32));
        const bool tiny = amax > 0.0F && amax < 0x1p-119F;
        const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
        const float rounded = tiny ? 0.0F : rintf(act / delta);
        const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
        int sum = code;
        for (int stride = 16; stride; stride /= 2) sum += __shfl_xor(sum, stride, 32);
        view.act_codes[slot * view.inter + out0 + 32U * gq + lane] = static_cast<signed char>(code);
        if (lane == 0U) {
            view.act_deltas[slot * groups + by * 4U + gq] = delta;
            view.act_sums[slot * groups + by * 4U + gq] = sum;
        }
    }
}

// Down: wave w: outputs 32 w .. of the block's 256.
// R: row groups of up to R rows; 32 (the default, STRIX_DS41_FAST_MOE_DOWN_ROWS=64 keeps 64) holds half the accumulators
// and staged items of 64 and compiles without register spills. Each output's terms keep their order: the same bits.
template <WeightFormat Format, bool Swap, unsigned R = relaxed_rows>
__global__ __launch_bounds__(relaxed_threads) void relaxed_down_kernel(PrefillDownView view) {
    const unsigned bx = Swap ? blockIdx.y : blockIdx.x, by = Swap ? blockIdx.x : blockIdx.y;
    __shared__ __attribute__((aligned(16))) unsigned char lds[relaxed_down_lds_bytes];
    __shared__ unsigned slot_of[R];
    RelaxedRows b;
    if (!relaxed_rows_of<R>(view.routes, b, bx)) return;
    if (threadIdx.x < b.count) slot_of[threadIdx.x] = view.routes.entry_slots[relaxed_entry(view.routes, b, threadIdx.x)];
    __syncthreads();
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U, h = lane >> 4U, i16 = lane & 15U;
    const unsigned first = by * 256U + 32U * wave;
    const auto* weights = static_cast<const unsigned char*>(view.weights) + b.expert * view.expert_stride_bytes +
        static_cast<unsigned long long>(first + lane) * view.row_bytes;
    float acc[R / 16U][2][8];
    const auto* tile_base = static_cast<const unsigned char*>(view.weights) + b.expert * view.expert_stride_bytes +
        static_cast<unsigned long long>(by) * 256U * view.row_bytes;
    relaxed_products<Format, R>(lds, weights, view.width, slot_of, b.count, view.codes, view.deltas, nullptr, acc, tile_base, view.row_bytes);
#pragma unroll
    for (unsigned mt = 0; mt < R / 16U; ++mt)
#pragma unroll
        for (unsigned t = 0; t < 2U; ++t)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const unsigned r = 16U * mt + 2U * e + h;
                if (r < b.count) view.out[static_cast<unsigned long long>(slot_of[r]) * view.outputs + first + 16U * t + i16] = acc[mt][t][e];
            }
}
