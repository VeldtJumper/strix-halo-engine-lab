// qpf2k GDN prefill core kernels (STRIX_QPF2K_GDN=1, default OFF). Included by hip_gdn_launcher.hip after
// hip_gdn_prefill_scan.inc; every kernel here forms bitwise the values of the production row kernels it replaces
// (STRIX_QPF2K_GDN_CHECK=1 runs both on the same inputs and compares every written byte).
//
// gdn_qpf2k_cols_kernel<Storage, G, Cc, W, Chunk>: gdn_prefill_lanes_kernel with Cc adjacent value columns per G-lane
// group. Lane `sub` of a group owns keys sub, sub + G, ... (K = 128 / G) of each of its Cc columns, so every staged
// key/query value a lane reads from LDS (as ds_load_b128) feeds Cc columns. Each column runs exactly the production
// chain: decay multiply, products, in-lane tree levels 64..G (gdn_lane_tree), row_xmask levels G/2..1
// (gdn_lane_group_sum, unchanged), delta = (v - sum) * beta, fmaf(k, delta, r), products with q, the same tree.
namespace qpf2k_gdn {
// Opt bit 2: the row's decay and beta (the same staged LDS word in every lane) read into SGPRs with readfirstlane,
// so the decay multiplies can pair as VOPD (SGPR sources have no VGPR bank constraint). Bit 1 is reserved (a
// DPP-fused group-sum add was measured but changes the operand order of the add: not used).
// Decay operand order (NaN payloads): the production lanes kernel's ISA multiplies retained (src0) by decay (src1). An
// SGPR decay must be src0, so with two NaN operands the SGPR form keeps the decay's payload where production keeps the
// retained value's. Whenever the decay is not NaN both orders give the same bits (one NaN operand: that NaN, quieted;
// none: the same IEEE product). Template Nan of the recurrence kernels:
//   0 = SGPR decay (6991b17f; exact unless a decay of the call is NaN), no flag (bench reference only);
//   1 = as 0, but the kernel returns at entry when the call's decay-NaN flag is set;
//   2 = pinned order (per-lane VGPR decay, asm v_mul retained, decay; exact for every input, the multiplies lose
//       VOPD), returns at entry when the flag is clear;
//   3 = pinned order, no flag (default: c4bc77c1);
//   4 = NaN witness (PARTS bit 128; gdn_core_kernel lean only): SGPR decay, no pinned order, and the
//       kernel writes its final recurrent state / v conv history into scratch (xch.state_out / xch.conv_out) instead of
//       the live buffers and sets *xch.witness when any final state element or any row's first-level norm sum is NaN.
//       Without a NaN operand IEEE add / mul / fma give the same bits in either operand order and in every lane, so a
//       kernel with the production's reduction pairs (stride-halving tree over the 128 keys: any G) is bitwise equal to
//       production unless a NaN arises in a column chain; every such NaN reaches that column's final state (delta NaN ->
//       fma(k, NaN, r)) or that row's output (q phase), which the witness tests. The launcher then runs the pinned Nan 2
//       kernel guarded by the witness word (it reads the untouched live state / conv history) and otherwise copies the
//       scratch into the live buffers (gdn_witness_commit_kernel).
// STRIX_QPF2K_GDN_PARTS bit 32 (decay-NaN dispatch): gdn_decay_nan_kernel sets the flag, then the launcher runs Nan 2
// and Nan 1 on the same arguments: exactly one of them does the work, so every call is exact and calls without a NaN
// decay (all real prompts) run the VOPD-paired loop.
template <unsigned Opt>
__device__ __forceinline__ float uniform(float value) {
    if constexpr (Opt & 2U) return __int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(value)));
    else return value;
}
template <unsigned Nan> constexpr bool kPinDecay = Nan == 2U || Nan == 3U;
// gdn_lane_tree's in-lane tree (level h adds p[j] + p[j + h], h = K/2 .. 1) evaluated in post order: node (r, s) =
// node (r, 2s) + node (r + s, 2s), node (r, K) = leaf r. The same additions with the same operands in the same order;
// only the evaluation order changes, so at most log2(K) + 1 partial sums are live instead of K products (Opt bit 4).
template <unsigned K, unsigned R = 0U, unsigned S = 1U, class F>
__device__ __forceinline__ float gdn_post_tree(const F& leaf) {
#pragma clang fp contract(off)
    if constexpr (S == K) return leaf(std::integral_constant<unsigned, R>{});
    else {
        const float lower = gdn_post_tree<K, R, 2U * S>(leaf);
        const float upper = gdn_post_tree<K, R + S, 2U * S>(leaf);
        return lower + upper;
    }
}
// Two post-order trees evaluated node by node (leaf returns {a, b}): each tree has exactly gdn_post_tree's nodes and
// operands; the two independent trees are interleaved (witness Opt bit 64).
template <unsigned K, unsigned R = 0U, unsigned S = 1U, class F>
__device__ __forceinline__ float2 gdn_post_tree2(const F& leaf) {
#pragma clang fp contract(off)
    if constexpr (S == K) return leaf(std::integral_constant<unsigned, R>{});
    else {
        const float2 lower = gdn_post_tree2<K, R, 2U * S>(leaf);
        const float2 upper = gdn_post_tree2<K, R + S, 2U * S>(leaf);
        return make_float2(lower.x + upper.x, lower.y + upper.y);
    }
}
// gdn_lane_group_sum's pairs (row_xmask G/2 .. 1) with an identity-0, bound_ctrl DPP source so the DPP combiner can fuse
// the move into the add (witness Opt bit 32; the operand order is free under the witness; all lanes of a row are valid).
template <unsigned G>
__device__ __forceinline__ float gdn_lane_group_sum_fused(float sum) {
#pragma clang fp contract(off)
    if constexpr (G > 1U) {
        sum = sum + __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(sum), 0x160 + static_cast<int>(G / 2U), 0xf, 0xf, true));
        return gdn_lane_group_sum_fused<G / 2U>(sum);
    } else return sum;
}
// true when this Nan variant does no work for the call (flag read once, uniform: a scalar load and branch)
template <unsigned Nan>
__device__ __forceinline__ bool decay_nan_skip(const unsigned* flag) {
    if constexpr (Nan == 1U || Nan == 2U) {
        const auto set = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(*flag)));
        return Nan == 1U ? set != 0U : set == 0U;
    } else return false;
}
// Sets *flag (zeroed before the launch, stream-ordered) when any decay of the call is NaN: the recurrence's own
// expression expf(gate) per (row, value head) (the commit of the stage), tested by its bits (immune to finite-math
// folding of isnan).
template <unsigned Block = 256U>
__global__ __launch_bounds__(Block) void gdn_decay_nan_kernel(GdnPrefillRows rows, std::uint32_t count, std::uint32_t heads, unsigned* flag) {
    const std::uint64_t n = static_cast<std::uint64_t>(count) * heads;
    bool any = false;
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * Block + threadIdx.x; i < n; i += static_cast<std::uint64_t>(gridDim.x) * Block)
        any |= (__float_as_uint(expf(rows.gate[i / heads * rows.stride + i % heads])) & 0x7fffffffU) > 0x7f800000U;
    if (any) *flag = 1U;
}
// IEEE multiply with a fixed operand order (src0 = a). With two NaN operands the hardware returns src0's payload, and
// the compiler treats fmul as commutative, so multiplies whose two operands can both be NaN pin the order the
// production kernels' ISA uses (engine object: gdn_prefill_norm_rows_kernel inv * value, then * weight, then
// gate * product; gdn_prefill_lanes_kernel retained * decay).
__device__ __forceinline__ float mul_src0(float a, float b) {
    float result;
    asm("v_mul_f32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
}
template <unsigned G, unsigned Cc, unsigned W> struct ColsShape {
    static constexpr unsigned keys = 128U / G, pitch = 128U / G + 4U, threads = W * 32U, groups = threads / G,
        columns = groups * Cc;
};
// Pad: floats after each key / query row (4: Shape::pitch, every kernel but the unpadded Chunk-16 core, Opt bit 2048)
template <unsigned G, unsigned Cc, unsigned W, unsigned Chunk, unsigned Pad = 4U> struct ColsStage {
    using Shape = ColsShape<G, Cc, W>;
    alignas(16) float key[Chunk][G][Shape::keys + Pad];
    alignas(16) float query[Chunk][G][Shape::keys + Pad];
    alignas(16) float value[Chunk][Shape::columns];
    float beta[Chunk], decay[Chunk];
};
// Chunk-16 padded core (Opt 6): a key / query row keeps the sub bases 0 / 36 / 72 / 108 floats (bank-spread) but drops
// the last sub's pad - 512 B less, so the stage fits 5 blocks per unit (25,600 B kernel LDS).
template <unsigned G, unsigned Cc, unsigned W, unsigned Chunk> struct ColsStageTight {
    using Shape = ColsShape<G, Cc, W>;
    static constexpr unsigned row = G * Shape::pitch - 4U;
    alignas(16) float key[Chunk][row];
    alignas(16) float query[Chunk][row];
    alignas(16) float value[Chunk][Shape::columns];
    float beta[Chunk], decay[Chunk];
};
template <GdnRecurrentStorage Storage, unsigned G, unsigned Cc, unsigned W, unsigned Chunk, unsigned Opt = 2U, unsigned Nan = 3U>
__global__ __launch_bounds__(W * 32U)
void gdn_qpf2k_cols_kernel(GdnPrefillRows rows, std::uint32_t count, float* state, std::uint32_t key_heads, const unsigned* decay_nan) {
#pragma clang fp contract(off)
    if (decay_nan_skip<Nan>(decay_nan)) return;
    using Shape = ColsShape<G, Cc, W>;
    constexpr unsigned Threads = Shape::threads, Columns = Shape::columns, K = Shape::keys;
    static_assert(128U % Columns == 0U && K % 4U == 0U && 2U * Chunk <= Threads);
    constexpr unsigned KeyQuads = (Chunk * 64U + Threads - 1U) / Threads;
    constexpr unsigned ValueQuads = (Chunk * Columns / 4U + Threads - 1U) / Threads;
    using Stage = ColsStage<G, Cc, W, Chunk>;
    __shared__ Stage stage[2];
    const auto thread = static_cast<std::uint32_t>(threadIdx.x);
    const auto sub = thread % G, local = thread / G * Cc;
    constexpr unsigned blocks = 128U / Columns;
    const auto value_head = static_cast<std::uint32_t>(blockIdx.x) / blocks;
    const auto column_base = static_cast<std::uint32_t>(blockIdx.x) % blocks * Columns, column = column_base + local;
    const auto key_width = key_heads * 128U, key_offset = value_head % key_heads * 128U;
    const auto value_offset = key_width * 2U + value_head * 128U + column_base;
    float4 key_query[KeyQuads], values[ValueQuads];
    float scalar = 0.0F;
    const auto fetch = [&](std::uint32_t first) {
#pragma unroll
        for (unsigned i = 0; i < KeyQuads; ++i) {
            const auto q = thread + i * Threads, row = q / 64U, within = q % 64U;
            if (Chunk * 64U % Threads == 0U || q < Chunk * 64U) {
                const float* source = rows.qkv + (first + row) * rows.stride + key_offset +
                    (within < 32U ? key_width + within * 4U : (within - 32U) * 4U);
                key_query[i] = *reinterpret_cast<const float4*>(source);
            }
        }
#pragma unroll
        for (unsigned i = 0; i < ValueQuads; ++i) {
            const auto q = thread + i * Threads, row = q / (Columns / 4U);
            if (q < Chunk * Columns / 4U)
                values[i] = *reinterpret_cast<const float4*>(rows.qkv + (first + row) * rows.stride + value_offset +
                    q % (Columns / 4U) * 4U);
        }
        if (thread < 2U * Chunk)
            scalar = (thread < Chunk ? rows.beta : rows.gate)[(first + thread % Chunk) * rows.stride + value_head];
    };
    const auto commit = [&](Stage& s) {
#pragma unroll
        for (unsigned i = 0; i < KeyQuads; ++i) {
            const auto q = thread + i * Threads, row = q / 64U, within = q % 64U;
            if (Chunk * 64U % Threads == 0U || q < Chunk * 64U) {
                const auto key = (within % 32U) * 4U;
                auto& target = within < 32U ? s.key[row] : s.query[row];
                const float parts[4] = {key_query[i].x, key_query[i].y, key_query[i].z, key_query[i].w};
#pragma unroll
                for (unsigned c = 0; c < 4U; ++c) target[(key + c) % G][(key + c) / G] = parts[c];
            }
        }
#pragma unroll
        for (unsigned i = 0; i < ValueQuads; ++i) {
            const auto q = thread + i * Threads;
            if (q < Chunk * Columns / 4U)
                *reinterpret_cast<float4*>(&s.value[q / (Columns / 4U)][q % (Columns / 4U) * 4U]) = values[i];
        }
        if (thread < Chunk) s.beta[thread] = scalar;
        else if (thread < 2U * Chunk) s.decay[thread - Chunk] = expf(scalar);
    };
    float retained[Cc][K];
#pragma unroll
    for (unsigned c = 0; c < Cc; ++c)
#pragma unroll
        for (unsigned j = 0; j < K; ++j)
            retained[c][j] = state[gdn_window_state_index<Storage>(value_head, sub + j * G, column + c, 128U)];
    fetch(0U);
    commit(stage[0]);
    __syncthreads();
    for (std::uint32_t first = 0, buffer = 0; first < count; first += Chunk, buffer ^= 1U) {
        const bool more = first + Chunk < count;
        if (more) fetch(first + Chunk);
        // Row pointers advance by constant strides (see gdn_prefill_lanes_kernel).
        const Stage& s = stage[buffer];
        const float4* key = reinterpret_cast<const float4*>(s.key[0][sub]);
        const float4* query = reinterpret_cast<const float4*>(s.query[0][sub]);
        const float* value = &s.value[0][local];
        const float* beta = s.beta;
        const float* decay = s.decay;
        float* core = rows.core + first * rows.stride + value_head * 128U + column;
#pragma unroll 1
        for (unsigned r = 0; r < Chunk; ++r) {
            float k[K];
#pragma unroll
            for (unsigned j = 0; j < K / 4U; ++j) {
                const float4 v4 = key[j];
                k[4U * j] = v4.x; k[4U * j + 1U] = v4.y; k[4U * j + 2U] = v4.z; k[4U * j + 3U] = v4.w;
            }
            // Pinned (Nan >= 2): decay a per-lane VGPR multiplied as src1 (production lanes kernel ISA: retained src0,
            // decay src1; an SGPR decay would have to be src0, and the compiler also commutes VGPR operands to pair VOPD).
            const float d = kPinDecay<Nan> ? *decay : uniform<Opt>(*decay), b = uniform<Opt>(*beta);
            float delta[Cc];
#pragma unroll
            for (unsigned c = 0; c < Cc; ++c) {
                float products[K];
#pragma unroll
                for (unsigned j = 0; j < K; ++j) {
                    if constexpr (kPinDecay<Nan>) retained[c][j] = mul_src0(retained[c][j], d);
                    else retained[c][j] = retained[c][j] * d;
                    products[j] = retained[c][j] * k[j];
                }
                delta[c] = (value[c] - gdn_lane_group_sum<G>(gdn_lane_tree(products))) * b;
            }
            float qv[K];
#pragma unroll
            for (unsigned j = 0; j < K / 4U; ++j) {
                const float4 v4 = query[j];
                qv[4U * j] = v4.x; qv[4U * j + 1U] = v4.y; qv[4U * j + 2U] = v4.z; qv[4U * j + 3U] = v4.w;
            }
            float out[Cc];
#pragma unroll
            for (unsigned c = 0; c < Cc; ++c) {
                float products[K];
#pragma unroll
                for (unsigned j = 0; j < K; ++j) {
                    retained[c][j] = __builtin_fmaf(k[j], delta[c], retained[c][j]);
                    products[j] = retained[c][j] * qv[j];
                }
                out[c] = gdn_lane_group_sum<G>(gdn_lane_tree(products));
            }
            if (sub == 0U) {
#pragma unroll
                for (unsigned c = 0; c < Cc; ++c) core[c] = out[c];
            }
            key += G * Shape::pitch / 4U; query += G * Shape::pitch / 4U; value += Columns; ++beta; ++decay; core += rows.stride;
        }
        if (more) commit(stage[buffer ^ 1U]);
        gdn_lds_barrier();
    }
#pragma unroll
    for (unsigned c = 0; c < Cc; ++c)
#pragma unroll
        for (unsigned j = 0; j < K; ++j)
            state[gdn_window_state_index<Storage>(value_head, sub + j * G, column + c, 128U)] = retained[c][j];
}

// gdn_qpf2k_gates_kernel<O, W, RC, RB>: gdn_prefill_gates_kernel's values with every (row, output) owned by one
// wave. Lane l holds chains l + 32 m (m = 0..7; chain t: fmaf over features t, t + 256, ... from +0, the production
// kernel's contracted `sum += x * w`), so tree levels 128, 64, 32 are in-lane adds (m with m + 4, m + 2, m + 1) and
// levels 16..1 move lane l + d into lane l (permlanex16, then DPP), the operands of every addition the same as the
// production LDS/shuffle tree; lane 0 then applies the same sigmoid / softplus transform. A wave keeps the 80 weights
// of each of its O outputs in registers; the block's W waves share RC staged FP32 rows in LDS. width == Width.
// Sched bit 2: the next rows are not prefetched into registers during the current rows (fewer live registers).
// Sched bit 1 (O >= 2): a scheduling fence after each 256-feature step keeps the step's 8 LDS reads next to its FMAs
// (otherwise the compiler hoists the reads of all steps and the O x 80 weights no longer fit in registers). Placement
// only: every chain, tree and transform is unchanged.
// Sched bit 4 (STRIX_QPF2K_GDN_GATES_BATCH): dt / a are loaded once per wave (not on every row's serial
// tail), and the transforms run across lanes: each row's raw total (lane 0 after the same tree, rounded to BF16 as
// before) is parked in lane (row - r0) % 32 of a register (readfirstlane + per-lane select), and every 32 rows (or at
// the end of the block's rows) all lanes apply the unchanged sigmoid / softplus expression to their own row at once
// (the lane-0 tail was ~158 alpha / ~38 beta instructions per row with one active lane). The chains, the tree, the
// BF16 rounding and the transform expressions are the same per value; only the lane that evaluates them changes.
template <unsigned O, unsigned W, unsigned RC, unsigned RB, unsigned Width = 2560U, unsigned Sched = 0U>
__global__ __launch_bounds__(W * 32U)
void gdn_qpf2k_gates_kernel(const std::uint16_t* input, std::uint64_t input_stride, const float* beta_weights,
    const float* alpha_weights, GdnPrefillRows rows, const void* dt, SmallWeightKind dt_kind, const void* a,
    SmallWeightKind a_kind, std::uint32_t heads) {
    constexpr unsigned Steps = Width / 256U, Threads = W * 32U;
    constexpr unsigned Units = RC * Width / 8U, UnitsPerThread = (Units + Threads - 1U) / Threads;  // 8 BF16 per unit
    static_assert(Width % 256U == 0U && RB % RC == 0U);
    __shared__ __align__(16) float xs[RC][Width];
    const unsigned tid = threadIdx.x, lane = tid % 32U, wave = tid / 32U;
    const unsigned g0 = (blockIdx.x * W + wave) * O, r0 = blockIdx.y * RB;
    float w[O][Steps][8];
#pragma unroll
    for (unsigned o = 0; o < O; ++o) {
        const unsigned g = g0 + o, out = g >= heads ? g - heads : g;
        const float* weights = (g >= heads ? alpha_weights : beta_weights) + static_cast<std::uint64_t>(out) * Width;
#pragma unroll
        for (unsigned k = 0; k < Steps; ++k)
#pragma unroll
            for (unsigned m = 0; m < 8U; ++m) w[o][k][m] = weights[k * 256U + m * 32U + lane];
    }
    uint4 staged[UnitsPerThread];
    const auto fetch = [&](unsigned first) {
#pragma unroll
        for (unsigned i = 0; i < UnitsPerThread; ++i) {
            const unsigned u = tid + i * Threads;
            if (Units % Threads == 0U || u < Units)
                staged[i] = *reinterpret_cast<const uint4*>(input + (first + u / (Width / 8U)) * input_stride + u % (Width / 8U) * 8U);
        }
    };
    const auto commit = [&]() {
#pragma unroll
        for (unsigned i = 0; i < UnitsPerThread; ++i) {
            const unsigned u = tid + i * Threads;
            if (Units % Threads == 0U || u < Units) {
                const unsigned r = u / (Width / 8U), f = u % (Width / 8U) * 8U;
                const unsigned words[4] = {staged[i].x, staged[i].y, staged[i].z, staged[i].w};
                float4 lo, hi;  // bf16_to_float of the even (low) and odd (high) halves
                lo.x = __uint_as_float(words[0] << 16U); lo.y = __uint_as_float(words[0] & 0xffff0000U);
                lo.z = __uint_as_float(words[1] << 16U); lo.w = __uint_as_float(words[1] & 0xffff0000U);
                hi.x = __uint_as_float(words[2] << 16U); hi.y = __uint_as_float(words[2] & 0xffff0000U);
                hi.z = __uint_as_float(words[3] << 16U); hi.w = __uint_as_float(words[3] & 0xffff0000U);
                *reinterpret_cast<float4*>(&xs[r][f]) = lo;
                *reinterpret_cast<float4*>(&xs[r][f + 4U]) = hi;
            }
        }
    };
    constexpr bool Prefetch = (Sched & 2U) == 0U;   // Sched bit 2: no prefetch (rows staged right before use)
    constexpr bool Batch = (Sched & 4U) != 0U;      // Sched bit 4: parameters preloaded, transforms across lanes
    static_assert(!Batch || RB % RC == 0U);
    float param_dt[O], param_a[O], held[O];
#pragma unroll
    for (unsigned o = 0; o < O; ++o) {
        param_dt[o] = param_a[o] = held[o] = 0.0F;
        if constexpr (Batch) {
            const unsigned g = g0 + o;
            if (g >= heads) { param_dt[o] = load_small(dt, dt_kind, g - heads); param_a[o] = load_small(a, a_kind, g - heads); }
        }
    }
    // Batch: lanes 0 .. n - 1 hold the raw totals of rows base .. base + n - 1; all lanes transform and store them.
    const auto flush = [&](unsigned base, unsigned n) {
#pragma unroll
        for (unsigned o = 0; o < O; ++o) {
            if (lane < n) {
                const unsigned g = g0 + o, row = base + lane;
                const bool alpha = g >= heads;
                const unsigned out = alpha ? g - heads : g;
                const float raw = held[o];
                const auto offset = row * rows.stride + out;
                if (alpha) {
                    const float biased = raw + param_dt[o];
                    rows.gate[offset] = param_a[o] * softplus(biased);
                } else rows.beta[offset] = round_bf16(1.0F / (1.0F + expf(-raw)));
            }
        }
    };
    if constexpr (Prefetch) fetch(r0);
    for (unsigned first = r0; first < r0 + RB; first += RC) {
        __syncthreads();
        if constexpr (!Prefetch) fetch(first);
        commit();
        __syncthreads();
        if constexpr (Prefetch) { if (first + RC < r0 + RB) fetch(first + RC); }
#pragma unroll 1
        for (unsigned r = 0; r < RC; ++r) {
            float sum[O][8];
#pragma unroll
            for (unsigned o = 0; o < O; ++o)
#pragma unroll
                for (unsigned m = 0; m < 8U; ++m) sum[o][m] = 0.0F;
#pragma unroll
            for (unsigned k = 0; k < Steps; ++k) {
                float x[8];
#pragma unroll
                for (unsigned m = 0; m < 8U; ++m) x[m] = xs[r][k * 256U + m * 32U + lane];
#pragma unroll
                for (unsigned o = 0; o < O; ++o)
#pragma unroll
                    for (unsigned m = 0; m < 8U; ++m) sum[o][m] = __builtin_fmaf(x[m], w[o][k][m], sum[o][m]);
                if constexpr ((Sched & 1U) != 0U) __builtin_amdgcn_sched_barrier(0);
            }
#pragma unroll
            for (unsigned o = 0; o < O; ++o) {
                float* v = sum[o];
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j) v[j] = gdn_window_add(v[j], v[j + 4U]);
                v[0] = gdn_window_add(v[0], v[2]); v[1] = gdn_window_add(v[1], v[3]);
                float total = gdn_window_add(v[0], v[1]);
                total = gdn_window_add(total, gdn_window_down<16U>(total));
                total = gdn_window_add(total, gdn_window_down<8U>(total));
                total = gdn_window_add(total, gdn_window_down<4U>(total));
                total = gdn_window_add(total, gdn_window_down<2U>(total));
                total = gdn_window_add(total, gdn_window_down<1U>(total));
                if constexpr (Batch) {
                    const float raw = round_bf16(__int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(total))));
                    if (lane == (first + r - r0) % 32U) held[o] = raw;
                } else if (lane == 0U) {
                    const unsigned g = g0 + o, row = first + r;
                    const bool alpha = g >= heads;
                    const unsigned out = alpha ? g - heads : g;
                    const float raw = round_bf16(total);
                    const auto offset = row * rows.stride + out;
                    if (alpha) {
                        const float biased = raw + load_small(dt, dt_kind, out);
                        rows.gate[offset] = load_small(a, a_kind, out) * softplus(biased);
                    } else rows.beta[offset] = round_bf16(1.0F / (1.0F + expf(-raw)));
                }
            }
        }
        if constexpr (Batch) {
            const unsigned done = first + RC - r0;
            if (done % 32U == 0U || done == RB) flush(r0 + (done - 1U) / 32U * 32U, (done - 1U) % 32U + 1U);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Fused GDN core (STRIX_QPF2K_GDN_PARTS bit 4): (1) the q/k conv/SiLU/L2-norm of gdn_prefill_conv_kernel run once per
// q/k head and per row segment (gdn_qk_boundary_kernel saves the BF16 raw rows before every segment start first);
// (2) gdn_core_kernel: the cols recurrence whose blocks form the v conv/SiLU of their own 64 value columns while
// staging each chunk (gdn_prefill_conv_kernel's v expressions, the raw v rows are not rewritten) and run
// gdn_prefill_norm_rows_kernel's gated RMSNorm in the chunk epilogue. A value head is split over a block pair: block
// half h owns columns {32h..32h+31} and {64+32h..64+32h+31}, so the norm tree's first level (c with c + 64) is block
// local; the pair exchanges the 32 first-level sums of each row through a small ring with release/acquire counters
// (zeroed by a stream-ordered memset before each launch; ring slots reused only after the peer's consumed counter
// passes them) and both blocks finish the production tree ((p0+p2)+(p1+p3), shuffles 16..1, readfirstlane), rsqrtf,
// sigmoid and products. The kernel is launched cooperatively, so all 96 blocks are co-resident (partner progress).
// Norm exchange of a block pair (see above).
constexpr unsigned kNormRing = 4U;          // chunks in flight per block in the exchange ring
struct NormExchange {
    float* partial;                          // [heads][2][kNormRing][Chunk][32]
    unsigned* published;                     // [heads][2] chunks published (monotonic within a launch, from base)
    unsigned* consumed;                      // [heads][2] chunks of the partner consumed
    unsigned base;                           // flag value at launch start (flags are never reset; base advances)
    const unsigned* decay_nan;               // the call's decay-NaN flag (gdn_decay_nan_kernel); read only when Nan is 1 or 2
    _Float16* core16 = nullptr;              // owned compact output rows (F16 lean core only)
    std::uint32_t core16_pitch = 0;          // halves per row, tiled::scratch_pitch(value_width)
    unsigned* core16_overflow = nullptr;     // dense FP16 diagnostic: one count per aligned group of 8
    float* state_out;                        // Nan 4 (witness): final recurrent state written here (live layout)
    std::uint16_t* conv_out;                 // Nan 4: v conv history written here (conv_state layout)
    unsigned* witness;                       // Nan 4: set to 1 when a NaN arose (the pinned fallback then runs)
};
__device__ __forceinline__ bool is_nan_bits(float x) { return (__float_as_uint(x) & 0x7fffffffU) > 0x7f800000U; }
// Nan 4 commit: when the witness is clear, the scratch state and v conv history become the live ones.
template <unsigned Block = 256U>
__global__ __launch_bounds__(Block) void gdn_witness_commit_kernel(const unsigned* witness, const float* state_out, float* state,
    std::uint64_t state_quads, const std::uint16_t* conv_out, std::uint16_t* conv_state, std::uint64_t conv_begin, std::uint64_t conv_n) {
    if (__builtin_amdgcn_readfirstlane(static_cast<int>(*witness)) != 0) return;
    const std::uint64_t step = static_cast<std::uint64_t>(gridDim.x) * 256U;
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * 256U + threadIdx.x; i < state_quads; i += step)
        reinterpret_cast<float4*>(state)[i] = reinterpret_cast<const float4*>(state_out)[i];
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * 256U + threadIdx.x; i < conv_n; i += step)
        conv_state[conv_begin + i] = conv_out[conv_begin + i];
}
// production gated-norm product value * inv * weight * gate in the production ISA's operand order
__device__ __forceinline__ float gated_norm_product(float value, float inv, float weight, float gate) {
    return mul_src0(gate, mul_src0(mul_src0(inv, value), weight));
}
__device__ __forceinline__ unsigned load_acquire(const unsigned* p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void store_release(unsigned* p, unsigned v) {
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

// (1) BF16 raw inputs of the 3 rows before every segment start (segments 1..S-1) of the q/k channels, read before
// the segment kernel rewrites any row in place: side[s][i][channel] = bf16(qkv[s * L - 3 + i][channel]).
// Segment 0 rows come from the conv history (copied here, so the last segment may rewrite the history).
template <unsigned Taps = kGdnWindowKernel>
__global__ __launch_bounds__(256)
void gdn_qk_boundary_kernel(GdnPrefillRows rows, std::uint32_t seg_rows, std::uint32_t qk_channels, const std::uint16_t* conv_state,
    std::uint16_t* side) {
    const auto s = blockIdx.y;
    for (auto i = threadIdx.x + blockIdx.x * 256U; i < 3U * qk_channels; i += gridDim.x * 256U) {
        const auto r = i / qk_channels, c = i % qk_channels;
        side[(static_cast<std::uint64_t>(s) * 3U + r) * qk_channels + c] = s == 0U
            ? conv_state[static_cast<std::uint64_t>(c) * kGdnWindowKernel + 1U + r]
            : float_to_bf16(rows.qkv[(s * seg_rows - 3U + r) * rows.stride + c]);
    }
}
// (2) gdn_prefill_conv_kernel<Waves, Chunk> restricted to the q/k heads and to rows [s L, s L + L) of segment s =
// blockIdx.y: the same per-row expressions and tree; rows -3..-1 of the segment come from conv_state (s = 0) or from
// the boundary side buffer; the last segment writes the conv history of its head.
template <unsigned Waves, unsigned Chunk>
__global__ __launch_bounds__(Waves * 32U)
void gdn_qk_conv_kernel(GdnPrefillRows rows, std::uint32_t count, std::uint32_t seg_rows, std::uint16_t* conv_state,
    const std::uint16_t* side, const void* conv_weights, SmallWeightKind conv_kind, std::uint32_t key_heads, std::uint32_t head_dim,
    float epsilon) {
    constexpr unsigned Ring = Chunk * 2U;
    static_assert(Chunk >= 4U && Chunk % Waves == 0U);
    __shared__ std::uint16_t raw[Ring][128];
    const auto thread = static_cast<std::uint32_t>(threadIdx.x), lane = thread % 32U, wave = thread / 32U;
    const auto block = static_cast<std::uint32_t>(blockIdx.x), base = block * head_dim, seg = static_cast<std::uint32_t>(blockIdx.y);
    const auto qk_channels = key_heads * 2U * head_dim;
    const bool is_key = block >= key_heads;
    const float q_scale = is_key ? 1.0F : rsqrtf(static_cast<float>(head_dim));
    auto* history = conv_state + static_cast<std::uint64_t>(base) * kGdnWindowKernel;
    const auto begin = seg * seg_rows, end = min(count, begin + seg_rows);
    // Ring slot of row r is (r - begin + Ring) % Ring; rows begin-3..begin-1 in slots Ring-3..Ring-1.
    for (auto i = thread; i < 3U * 128U; i += Waves * 32U)
        raw[Ring - 3U + i / 128U][i % 128U] = side[(static_cast<std::uint64_t>(seg) * 3U + i / 128U) * qk_channels + base + i % 128U];
    float weight[4][kGdnWindowKernel];
#pragma unroll
    for (std::uint32_t part = 0; part < 4U; ++part)
#pragma unroll
        for (std::uint32_t index = 0; index < kGdnWindowKernel; ++index)
            weight[part][index] = load_small(conv_weights, conv_kind,
                static_cast<std::uint64_t>(base + part * 32U + lane) * kGdnWindowKernel + index);
    for (std::uint32_t first = begin; first < end; first += Chunk) {
        for (auto i = thread; i < Chunk * 128U; i += Waves * 32U) {
            const auto row = first + i / 128U;
            if (row < end) raw[(row - begin) % Ring][i % 128U] = float_to_bf16(rows.qkv[row * rows.stride + base + i % 128U]);
        }
        __syncthreads();
        for (auto row = first + wave; row < first + Chunk && row < end; row += Waves) {
            float value[4], square[4];
#pragma unroll
            for (std::uint32_t part = 0; part < 4U; ++part) {
                const auto channel = part * 32U + lane;
                float sum = 0.0F;
                for (std::uint32_t index = 0; index < kGdnWindowKernel; ++index)
                    sum += bf16_to_float(raw[(row - begin + Ring - 3U + index) % Ring][channel]) * weight[part][index];
                const float silu = sum / (1.0F + expf(-sum));
                value[part] = round_bf16(silu);
                square[part] = value[part] * value[part];
            }
            const float inverse = rsqrtf(gdn_window_key_sum(square) + epsilon);
#pragma unroll
            for (std::uint32_t part = 0; part < 4U; ++part) value[part] = value[part] * inverse * q_scale;
#pragma unroll
            for (std::uint32_t part = 0; part < 4U; ++part)
                rows.qkv[row * rows.stride + base + part * 32U + lane] = value[part];
        }
        __syncthreads();
    }
    if (end == count)
        for (auto i = thread; i < kGdnWindowKernel * 128U; i += Waves * 32U)
            history[(i % 128U) * kGdnWindowKernel + i / 128U] = raw[(count - begin - kGdnWindowKernel + i / 128U) % Ring][i % 128U];
}
// One v channel conv/SiLU output: gdn_prefill_conv_kernel expressions, outside any fp-contract pragma (the
// production conv kernel contracted 4-tap sum from 0).
template <unsigned Ring, unsigned Columns>
__device__ __forceinline__ float gdn_v_conv_value(const std::uint16_t (&vraw)[Ring][Columns], const float (&vweight)[Columns][kGdnWindowKernel],
    unsigned row, unsigned lc) {
    float sum = 0.0F;
    for (std::uint32_t index = 0; index < kGdnWindowKernel; ++index)
        sum += bf16_to_float(vraw[(row + Ring - 3U + index) % Ring][lc]) * vweight[lc][index];
    const float silu = sum / (1.0F + expf(-sum));
    return round_bf16(silu);
}
// Nan 4 (witness) compact stage: key i of a row at [i / G / 4][i % G][i / G % 4], so lane sub reads its keys sub + G m
// (m = 0..K-1) as K/4 float4 at a stride of G float4 (the padded ColsStage at G 8 needs 27.3 KB per block: 5 blocks per
// WGP would exceed 128 KiB and the cooperative launch would not be resident).
template <unsigned G, unsigned Chunk> struct KrStage {
    static constexpr unsigned K = 128U / G;
    alignas(16) float key[Chunk][K / 4U][G][4];
    alignas(16) float query[Chunk][K / 4U][G][4];
    alignas(16) float value[Chunk][64];
    float beta[Chunk], decay[Chunk];
};
// gdn_v_conv_value's expressions up to the SiLU (before round_bf16): the witness kernel tests it for NaN, because
// float_to_bf16 maps some NaN payloads to infinities or zeros (a NaN whose payload depends on operand order could vanish).
template <unsigned Ring, unsigned Columns>
__device__ __forceinline__ float gdn_v_conv_silu(const std::uint16_t (&vraw)[Ring][Columns], const float (&vweight)[Columns][kGdnWindowKernel],
    unsigned row, unsigned lc) {
    float sum = 0.0F;
    for (std::uint32_t index = 0; index < kGdnWindowKernel; ++index)
        sum += bf16_to_float(vraw[(row + Ring - 3U + index) % Ring][lc]) * vweight[lc][index];
    return sum / (1.0F + expf(-sum));
}
// (3) Recurrence (gdn_cols_norm_kernel's layout and norm exchange) whose block stages its 64 RAW v columns of each
// chunk in a BF16 ring and forms their conv/SiLU (gdn_prefill_conv_kernel's v expressions) into the stage; the block
// updates the conv history of its 64 v channels at the end (no other block owns them).
// Pipe: the norm of chunk n is finished after the recurrence of chunk n + 1 (its partial sums are published right
// after its own recurrence), so the pair exchange never waits on the peer; same values, same order.
// Lean (bit 1, not with Pipe): the chunk epilogue keeps each row's own first-level norm sums in registers (the peer
// reads them from the ring; this block no longer reads its own back), loads the chunk's z and norm weights before the
// exchange (their latency overlaps it), and forms the next chunk's v conv between publishing and waiting for the
// peer; same values, same expressions, same order. Bits 16/32 are timing-only diagnostics of the microbench (wrong
// outputs; compiled only with QPF2K_GDN_BENCH_DIAG): 16 = no pair wait (the peer sums are replaced by the own sums),
// 32 = no norm (the raw outputs are stored).
// Chunk 16 (STRIX_QPF2K_GDN_CHUNK16, block-pair decay-NaN dispatch kernels only): 16-row chunks with ONE stage (LDS for
// 5 blocks per WGP), so the epilogue commits the next chunk's q/k/beta/decay and v rows after a barrier that follows every
// wave's last token-loop read, and forms its v conv after every wave has read its outputs (kept in registers for the norm).
// Opt bit 2048: unpadded key / query rows (bank layout only). Same per-row arithmetic, norm order and stores.
// LDS (G 4, Cc 2, W 4): 25,600 B padded (ColsStageTight), 24,064 B unpadded; the 8-row kernels keep 25,216 B.
template <GdnRecurrentStorage Storage, unsigned G, unsigned Cc, unsigned W, unsigned Chunk, unsigned Opt, bool Pipe = false,
    unsigned Lean = 0U, unsigned Nan = 3U, bool F16 = false>
__global__ __launch_bounds__(W * 32U) __attribute__((amdgpu_waves_per_eu(5, 16)))
void gdn_core_kernel(GdnPrefillRows rows, std::uint32_t count, float* state, std::uint32_t key_heads, std::uint16_t* conv_state,
    const void* conv_weights, SmallWeightKind conv_kind, const void* norm_weights, SmallWeightKind norm_kind, float epsilon,
    NormExchange xch) {
#pragma clang fp contract(off)
    static_assert(!F16 || Lean == 1U, "FP16 output requires the lean gated-norm epilogue");
    // Nan 1/2: both blocks of a pair read the same flag, so a pair always runs (or skips) together.
    if (decay_nan_skip<Nan>(xch.decay_nan)) return;
    constexpr bool Witness = Nan == 4U;
    static_assert(!Witness || (Lean == 1U && !Pipe && Chunk == 8U && G >= 4U));
    constexpr bool DeferNorm = Witness && (Opt & 256U) != 0U;
    static_assert(!DeferNorm || (G == 4U && Cc == 2U && W == 4U));
    // Opt bit 512: packed two-row v conv is a witness-only extension of deferred norm.
    constexpr bool PackedValue = (Opt & 512U) != 0U;
    static_assert(!PackedValue || (Witness && DeferNorm && Opt == (374U | 512U)));
    static_assert((374U | 512U) == 886U);
    // Opt bit 1024: schedule the previous chunk's sigmoid gates inside the next recurrence.
    constexpr bool PendingSigmoid = (Opt & 1024U) != 0U;
    static_assert(!PendingSigmoid || (Witness && DeferNorm && !PackedValue && Opt == (374U | 1024U)));
    static_assert((374U | 1024U) == 1398U);
    [[maybe_unused]] bool nan_seen = false;   // Nan 4 only
    using Shape = ColsShape<G, Cc, W>;
    // Chunk 16: one stage; v ring of Chunk + 3 rows (the 3 conv history rows plus the chunk); Opt bit 2048 unpadded rows.
    constexpr bool Single = Chunk == 16U;
    static_assert(!Single || (G == 4U && Cc == 2U && W == 4U && (Opt == 6U || Opt == 2054U) && !Pipe && Lean == 1U &&
        (Nan == 1U || Nan == 2U)), "Chunk 16 is built for the block-pair decay-NaN dispatch kernels only");
    static_assert(Chunk <= 8U || Single);
    static_assert((Opt & 2048U) == 0U || Single);
    constexpr unsigned Pad = (Opt & 2048U) != 0U ? 0U : 4U, Pitch = 128U / G + Pad;
    constexpr bool Tight = Single && Pad != 0U;   // ColsStageTight rows: sub s at s * Pitch, row G * Pitch - 4 floats
    constexpr unsigned Threads = Shape::threads, Columns = Shape::columns, K = Shape::keys, Ring = Single ? Chunk + 3U : 12U;
    static_assert(Columns == 64U && Chunk + 3U <= Ring && Chunk % W == 0U);
    constexpr unsigned KeyQuads = (Chunk * 64U + Threads - 1U) / Threads;
    constexpr unsigned ValueQuads = (Chunk * Columns / 4U + Threads - 1U) / Threads;
    using Stage = std::conditional_t<Witness, KrStage<G, Chunk>,
        std::conditional_t<Tight, ColsStageTight<G, Cc, W, Chunk>, ColsStage<G, Cc, W, Chunk, Pad>>>;
    __shared__ Stage stage[Single ? 1U : 2U];
    __shared__ std::uint16_t vraw[Ring][Columns];
    __shared__ float vweight[Columns][kGdnWindowKernel];
    const auto thread = static_cast<std::uint32_t>(threadIdx.x), lane = thread % 32U, wave = thread / 32U;
    const auto sub = thread % G, local = thread / G * Cc;
    const auto value_head = static_cast<std::uint32_t>(blockIdx.x) / 2U, half = static_cast<std::uint32_t>(blockIdx.x) % 2U;
    const auto column_of = [&](unsigned lc) { return lc < 32U ? 32U * half + lc : 64U + 32U * half + (lc - 32U); };
    const auto column = column_of(local);
    const auto key_width = key_heads * 128U, key_offset = value_head % key_heads * 128U;
    const auto value_offset = key_width * 2U + value_head * 128U;   // conv channel of value column c: value_offset + c
    for (auto i = thread; i < Columns * kGdnWindowKernel; i += Threads) {
        const auto lc = i / kGdnWindowKernel, t = i % kGdnWindowKernel;
        vweight[lc][t] = load_small(conv_weights, conv_kind, static_cast<std::uint64_t>(value_offset + column_of(lc)) * kGdnWindowKernel + t);
    }
    for (auto i = thread; i < 3U * Columns; i += Threads)
        vraw[Ring - 3U + i / Columns][i % Columns] =
            conv_state[static_cast<std::uint64_t>(value_offset + column_of(i % Columns)) * kGdnWindowKernel + 1U + i / Columns];
    float4 key_query[KeyQuads], values[ValueQuads];
    float scalar = 0.0F;
    const auto fetch = [&](std::uint32_t first) {
#pragma unroll
        for (unsigned i = 0; i < KeyQuads; ++i) {
            const auto q = thread + i * Threads, row = q / 64U, within = q % 64U;
            if (Chunk * 64U % Threads == 0U || q < Chunk * 64U) {
                const float* source = rows.qkv + (first + row) * rows.stride + key_offset +
                    (within < 32U ? key_width + within * 4U : (within - 32U) * 4U);
                key_query[i] = *reinterpret_cast<const float4*>(source);
            }
        }
#pragma unroll
        for (unsigned i = 0; i < ValueQuads; ++i) {
            const auto q = thread + i * Threads, row = q / (Columns / 4U);
            if (q < Chunk * Columns / 4U)
                values[i] = *reinterpret_cast<const float4*>(rows.qkv + (first + row) * rows.stride + value_offset +
                    column_of(q % (Columns / 4U) * 4U));
        }
        if (thread < 2U * Chunk)
            scalar = (thread < Chunk ? rows.beta : rows.gate)[(first + thread % Chunk) * rows.stride + value_head];
    };
    // q/k (post-conv, in place) -> stage; raw v -> BF16 ring.
    const auto commit = [&](Stage& s, std::uint32_t first) {
#pragma unroll
        for (unsigned i = 0; i < KeyQuads; ++i) {
            const auto q = thread + i * Threads, row = q / 64U, within = q % 64U;
            if (Chunk * 64U % Threads == 0U || q < Chunk * 64U) {
                const auto key = (within % 32U) * 4U;
                auto& target = within < 32U ? s.key[row] : s.query[row];
                const float parts[4] = {key_query[i].x, key_query[i].y, key_query[i].z, key_query[i].w};
                if constexpr (Witness) {
#pragma unroll
                    for (unsigned c = 0; c < 4U; ++c) target[(key + c) / G / 4U][(key + c) % G][(key + c) / G % 4U] = parts[c];
                } else if constexpr (Tight) {
#pragma unroll
                    for (unsigned c = 0; c < 4U; ++c) target[(key + c) % G * Pitch + (key + c) / G] = parts[c];
                } else {
#pragma unroll
                for (unsigned c = 0; c < 4U; ++c) target[(key + c) % G][(key + c) / G] = parts[c];
                }
            }
        }
#pragma unroll
        for (unsigned i = 0; i < ValueQuads; ++i) {
            const auto q = thread + i * Threads;
            if (q < Chunk * Columns / 4U) {
                const auto row = q / (Columns / 4U), lc = q % (Columns / 4U) * 4U, slot = (first + row) % Ring;
                vraw[slot][lc] = float_to_bf16(values[i].x); vraw[slot][lc + 1U] = float_to_bf16(values[i].y);
                vraw[slot][lc + 2U] = float_to_bf16(values[i].z); vraw[slot][lc + 3U] = float_to_bf16(values[i].w);
            }
        }
        if (thread < Chunk) s.beta[thread] = scalar;
        else if (thread < 2U * Chunk) s.decay[thread - Chunk] = expf(scalar);
    };
    // v conv/SiLU of the chunk (rows first..first+Chunk-1) from the ring into s.value.
    const auto convolve = [&](Stage& s, std::uint32_t first) {
        for (unsigned i = thread; i < Chunk * Columns; i += Threads) {
            const unsigned r = i / Columns, lc = i % Columns, row = first + r;
            if constexpr (Witness) {
                const float silu = gdn_v_conv_silu<Ring, Columns>(vraw, vweight, row, lc);
                nan_seen |= is_nan_bits(silu);
                s.value[r][lc] = round_bf16(silu);
            } else
            s.value[r][lc] = gdn_v_conv_value<Ring, Columns>(vraw, vweight, row, lc);
        }
    };
    unsigned* const my_pub = xch.published + value_head * 2U + half;
    unsigned* const peer_pub = xch.published + value_head * 2U + (half ^ 1U);
    unsigned* const my_con = xch.consumed + value_head * 2U + half;
    unsigned* const peer_con = xch.consumed + value_head * 2U + (half ^ 1U);
    float* const my_ring = xch.partial + (static_cast<std::uint64_t>(value_head) * 2U + half) * kNormRing * Chunk * 32U;
    float* const peer_ring = xch.partial + (static_cast<std::uint64_t>(value_head) * 2U + (half ^ 1U)) * kNormRing * Chunk * 32U;
    const auto publish = [&](float (&outs)[Chunk][Columns], unsigned n) {
        if (n >= kNormRing && thread == 0U)
            while (load_acquire(peer_con) < xch.base + n - kNormRing + 1U) __builtin_amdgcn_s_sleep(1);
        __syncthreads();
        for (unsigned r = wave; r < Chunk; r += W) {
            const float lo = outs[r][lane], hi = outs[r][32U + lane];
            my_ring[((n % kNormRing) * Chunk + r) * 32U + lane] = gdn_window_add(lo * lo, hi * hi);
        }
        __syncthreads();
        if (thread == 0U) { __threadfence(); store_release(my_pub, xch.base + n + 1U); }
    };
    const auto finish = [&](float (&outs)[Chunk][Columns], std::uint32_t first, unsigned n) {
        if (thread == 0U) while (load_acquire(peer_pub) < xch.base + n + 1U) __builtin_amdgcn_s_sleep(1);
        __syncthreads();
        for (unsigned r = wave; r < Chunk; r += W) {
            const float mine = my_ring[((n % kNormRing) * Chunk + r) * 32U + lane];
            const float peer = __hip_atomic_load(&peer_ring[((n % kNormRing) * Chunk + r) * 32U + lane], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            float sum = half == 0U ? gdn_window_add(mine, peer) : gdn_window_add(peer, mine);
            sum = gdn_window_add(sum, gdn_window_down<16U>(sum));
            sum = gdn_window_add(sum, gdn_window_down<8U>(sum));
            sum = gdn_window_add(sum, gdn_window_down<4U>(sum));
            sum = gdn_window_add(sum, gdn_window_down<2U>(sum));
            sum = gdn_window_add(sum, gdn_window_down<1U>(sum));
            sum = __int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(sum)));
            // The production norm divides by its runtime head_dim (a full IEEE division); a literal 128 would fold to a
            // multiply by 2^-7, which differs from the division for NaN inputs. Keep the divisor opaque.
            float head_dim_f;
            asm volatile("v_mov_b32 %0, 0x43000000" : "=v"(head_dim_f));
            const float inv = rsqrtf(sum / head_dim_f + epsilon);
            const std::uint64_t base = static_cast<std::uint64_t>(first + r) * rows.stride + value_head * 128U;
#pragma unroll
            for (unsigned part = 0; part < 2U; ++part) {
                const auto channel = part * 64U + 32U * half + lane;
                const float gate_value = 1.0F / (1.0F + expf(-rows.z[base + channel]));
                rows.core[base + channel] = gated_norm_product(outs[r][part * 32U + lane], inv, load_small(norm_weights, norm_kind, channel), gate_value);
            }
        }
        __syncthreads();
        if (thread == 0U) store_release(my_con, xch.base + n + 1U);
    };
    const auto norm_chunk = [&](float (&outs)[Chunk][Columns], std::uint32_t first, unsigned n) {
        publish(outs, n);
        finish(outs, first, n);
    };
    constexpr unsigned PendingRows = Chunk / W;
    float pending_mine[PendingRows], pending_z[PendingRows][2], pending_nw[2];
    // Opt bit 256 (witness only): retain Lean sums and z across the next recurrence.
    const auto publish_pending = [&](float (&outs)[Chunk][Columns], std::uint32_t first, unsigned n) {
#pragma unroll
        for (unsigned part = 0; part < 2U; ++part) {
            const auto channel = part * 64U + 32U * half + lane;
            pending_nw[part] = load_small(norm_weights, norm_kind, channel);
#pragma unroll
            for (unsigned i = 0; i < PendingRows; ++i)
                pending_z[i][part] = rows.z[static_cast<std::uint64_t>(first + wave + i * W) * rows.stride + value_head * 128U + channel];
        }
        if (n >= kNormRing && thread == 0U)
            while (load_acquire(peer_con) < xch.base + n - kNormRing + 1U) __builtin_amdgcn_s_sleep(1);
        __syncthreads();
#pragma unroll
        for (unsigned i = 0; i < PendingRows; ++i) {
            const unsigned r = wave + i * W;
            const float lo = outs[r][lane], hi = outs[r][32U + lane];
            pending_mine[i] = gdn_window_add(lo * lo, hi * hi);
            if constexpr (Witness) nan_seen |= is_nan_bits(pending_mine[i]);   // NaN iff lo or hi is NaN
            my_ring[((n % kNormRing) * Chunk + r) * 32U + lane] = pending_mine[i];
        }
        __syncthreads();
#ifdef QPF2K_GDN_BENCH_DELAY_PARTNER
        if (half == 1U && thread == 0U)
            for (unsigned delay = 0; delay < QPF2K_GDN_BENCH_DELAY_PARTNER; ++delay) __builtin_amdgcn_s_sleep(1);
#endif
        if (thread == 0U) { __threadfence(); store_release(my_pub, xch.base + n + 1U); }
    };
    const auto finish_pending = [&](float (&outs)[Chunk][Columns], std::uint32_t first, unsigned n, bool gates_ready) {
        if (thread == 0U) while (load_acquire(peer_pub) < xch.base + n + 1U) __builtin_amdgcn_s_sleep(1);
        __syncthreads();
#pragma unroll
        for (unsigned i = 0; i < PendingRows; ++i) {
            const unsigned r = wave + i * W;
            const float peer = __hip_atomic_load(&peer_ring[((n % kNormRing) * Chunk + r) * 32U + lane], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            float sum = half == 0U ? gdn_window_add(pending_mine[i], peer) : gdn_window_add(peer, pending_mine[i]);
            sum = gdn_window_add(sum, gdn_window_down<16U>(sum));
            sum = gdn_window_add(sum, gdn_window_down<8U>(sum));
            sum = gdn_window_add(sum, gdn_window_down<4U>(sum));
            sum = gdn_window_add(sum, gdn_window_down<2U>(sum));
            sum = gdn_window_add(sum, gdn_window_down<1U>(sum));
            sum = __int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(sum)));
            float head_dim_f;   // opaque divisor: the production runtime IEEE division (see finish)
            asm volatile("v_mov_b32 %0, 0x43000000" : "=v"(head_dim_f));
            const float inv = rsqrtf(sum / head_dim_f + epsilon);
            const std::uint64_t base = static_cast<std::uint64_t>(first + r) * rows.stride + value_head * 128U;
#pragma unroll
            for (unsigned part = 0; part < 2U; ++part) {
                const auto channel = part * 64U + 32U * half + lane;
                // The final drain has no following recurrence: its z entries are still raw.
                const float gate_value = PendingSigmoid && gates_ready ? pending_z[i][part]
                    : 1.0F / (1.0F + expf(-pending_z[i][part]));
                const float normed = gated_norm_product(outs[r][part * 32U + lane], inv, pending_nw[part], gate_value);
                if constexpr (Witness) nan_seen |= is_nan_bits(normed);
                if constexpr (F16) {
                    // Same FP32 value and RNE conversion as tiled kF32 staging; no FP32 reader on this route.
                    xch.core16[static_cast<std::uint64_t>(first + r) * xch.core16_pitch + value_head * 128U + channel] =
                        static_cast<_Float16>(normed);
                    // Each aligned 8-lane group is one row's 8 consecutive channels (including across heads).
                    const auto bad = __ballot(isfinite(normed) && !(fabsf(normed) <= 65504.0F));
                    if (bad != 0ULL && lane % 8U == 0U && ((bad >> lane) & 0xffULL) != 0ULL)
                        atomicAdd(xch.core16_overflow, 1U);
                } else rows.core[base + channel] = normed;
            }
        }
        __syncthreads();
        if (thread == 0U) store_release(my_con, xch.base + n + 1U);
    };
    float retained[Cc][K];
#pragma unroll
    for (unsigned c = 0; c < Cc; ++c)
#pragma unroll
        for (unsigned j = 0; j < K; ++j)
            retained[c][j] = state[gdn_window_state_index<Storage>(value_head, sub + j * G, column + c, 128U)];
    fetch(0U);
    __syncthreads();
    commit(stage[0], 0U);
    __syncthreads();
    if constexpr (!PackedValue) convolve(stage[0], 0U);
    __syncthreads();
    unsigned n = 0;
    for (std::uint32_t first = 0, buffer = 0; first < count; first += Chunk, buffer ^= Single ? 0U : 1U, ++n) {
        const bool more = first + Chunk < count;
        if (more) fetch(first + Chunk);
        const Stage& s = stage[buffer];
        // Witness (compact stage): a lane's key quads are G float4 apart, rows 32 float4 apart.
        constexpr unsigned QuadStep = Witness ? G : 1U, RowStep = Witness ? 32U : (G * Pitch - (Tight ? 4U : 0U)) / 4U;
        const float4* key;
        const float4* query;
        if constexpr (Witness) {
            key = reinterpret_cast<const float4*>(&s.key[0][0][sub][0]);
            query = reinterpret_cast<const float4*>(&s.query[0][0][sub][0]);
        } else if constexpr (Tight) {
            key = reinterpret_cast<const float4*>(&s.key[0][sub * Pitch]);
            query = reinterpret_cast<const float4*>(&s.query[0][sub * Pitch]);
        } else {
            key = reinterpret_cast<const float4*>(s.key[0][sub]);
            query = reinterpret_cast<const float4*>(s.query[0][sub]);
        }
        const float* value = &s.value[0][local];
        const float* beta = s.beta;
        const float* decay = s.decay;
        // Raw outputs feed norm; the non-packed path overwrites staged conv inputs after their last read.
        float* out_row = &stage[buffer].value[0][local];
        // Witness + Opt bit 16: software-pipelined rows: the query phase of row r - 1 (reads the retained state after
        // row r - 1's update) is issued together with the key phase of row r (decay, products, tree, delta, update), two
        // independent reduction trees per column per iteration. Every value is formed by the same operations on the
        // same operands with the same pairs as the plain loop; only the issue order of independent work changes (operand
        // order inside an operation is free under the NaN witness).
        if constexpr (Witness && (Opt & 16U) != 0U) {
            static_assert((Opt & 4U) != 0U);
            const auto load_quads = [&](const float4* p, float (&dst)[K]) {
#pragma unroll
                for (unsigned j = 0; j < K / 4U; ++j) {
                    const float4 v4 = p[j * QuadStep];
                    dst[4U * j] = v4.x; dst[4U * j + 1U] = v4.y; dst[4U * j + 2U] = v4.z; dst[4U * j + 3U] = v4.w;
                }
            };
            const auto group_sum = [](float x) {
                if constexpr ((Opt & 32U) != 0U) return gdn_lane_group_sum_fused<G>(x);
                else return gdn_lane_group_sum<G>(x);
            };
            float packed = 0.0F;
            const auto pack_pair = [&](unsigned r) {
                // Match the standalone conv's contraction context, outside the recurrence arithmetic.
#pragma clang fp contract(on)
                const unsigned conv_column = local + (sub & 1U);
                const unsigned conv_row = first + r + (sub >> 1U);
                const float silu = gdn_v_conv_silu<Ring, Columns>(vraw, vweight, conv_row, conv_column);
                nan_seen |= is_nan_bits(silu);   // before BF16 rounding can erase a NaN payload
                packed = round_bf16(silu);
            };
            const auto recurrence_value = [&](unsigned r, unsigned c) {
                if constexpr (PackedValue) {
                    // Every lane participates with full EXEC; shuffles only transport the rounded bits.
                    return __shfl(packed, static_cast<int>((lane & ~3U) + 2U * (r & 1U) + c), 32);
                } else return value[r * Columns + c];
            };
            const auto key_phase = [&](unsigned r, const float (&k)[K]) {
                const float d = uniform<Opt>(decay[r]), b = uniform<Opt>(beta[r]);
#pragma unroll
                for (unsigned c = 0; c < Cc; ++c) {
#pragma unroll
                    for (unsigned j = 0; j < K; ++j) retained[c][j] = retained[c][j] * d;
                    const float sum = gdn_post_tree<K>([&](auto jj) {
#pragma clang fp contract(off)
                        return retained[c][decltype(jj)::value] * k[decltype(jj)::value];
                    });
                    const float delta = (recurrence_value(r, c) - group_sum(sum)) * b;
#pragma unroll
                    for (unsigned j = 0; j < K; ++j) retained[c][j] = __builtin_fmaf(k[j], delta, retained[c][j]);
                }
            };
            // Opt bit 64: row r - 1's query tree and row r's key tree in one paired post-order walk; each leaf reads the
            // updated state for the query product before the decay overwrites it.
            const auto paired_phase = [&](unsigned r, const float (&qv)[K], const float (&k)[K], float (&out)[Cc]) {
                const float d = uniform<Opt>(decay[r]), b = uniform<Opt>(beta[r]);
#pragma unroll
                for (unsigned c = 0; c < Cc; ++c) {
                    const float2 sums = gdn_post_tree2<K>([&](auto jj) {
#pragma clang fp contract(off)
                        constexpr unsigned j = decltype(jj)::value;
                        const float updated = retained[c][j];
                        const float qp = updated * qv[j];
                        retained[c][j] = updated * d;
                        return make_float2(qp, retained[c][j] * k[j]);
                    });
                    out[c] = group_sum(sums.x);
                    const float delta = (recurrence_value(r, c) - group_sum(sums.y)) * b;
#pragma unroll
                    for (unsigned j = 0; j < K; ++j) retained[c][j] = __builtin_fmaf(k[j], delta, retained[c][j]);
                }
            };
            const auto query_phase = [&](const float (&qv)[K], float (&out)[Cc]) {
#pragma unroll
                for (unsigned c = 0; c < Cc; ++c)
                    out[c] = group_sum(gdn_post_tree<K>([&](auto jj) {
#pragma clang fp contract(off)
                        return retained[c][decltype(jj)::value] * qv[decltype(jj)::value];
                    }));
            };
            float k[K], qv[K], out[Cc];
            load_quads(key, k);
            if constexpr (PackedValue) pack_pair(0U);
            key_phase(0U, k);
#pragma unroll 1
            for (unsigned r = 1; r < Chunk; ++r) {
                load_quads(query + (r - 1U) * RowStep, qv);
                load_quads(key + r * RowStep, k);
                if constexpr (PackedValue) { if ((r & 1U) == 0U) pack_pair(r); }
                if constexpr ((Opt & 64U) != 0U) paired_phase(r, qv, k, out);
                else {
                    query_phase(qv, out);          // row r - 1 (state after its update)
                    key_phase(r, k);               // row r
                }
                if (sub == 0U) {
#pragma unroll
                    for (unsigned c = 0; c < Cc; ++c) out_row[(r - 1U) * Columns + c] = out[c];
                }
                if constexpr (PendingSigmoid) {
                    // Full Chunk=8 rows only (launcher admission); chunk zero has no pending z.
                    // Literal subscripts keep all four entries in registers, with no indexed scratch.
                    if (n != 0U) {
                        if (r == 1U) pending_z[0][0] = 1.0F / (1.0F + expf(-pending_z[0][0]));
                        if (r == 3U) pending_z[0][1] = 1.0F / (1.0F + expf(-pending_z[0][1]));
                        if (r == 5U) pending_z[1][0] = 1.0F / (1.0F + expf(-pending_z[1][0]));
                        if (r == 7U) pending_z[1][1] = 1.0F / (1.0F + expf(-pending_z[1][1]));
                    }
                }
            }
            load_quads(query + (Chunk - 1U) * RowStep, qv);
            query_phase(qv, out);
            if (sub == 0U) {
#pragma unroll
                for (unsigned c = 0; c < Cc; ++c) out_row[(Chunk - 1U) * Columns + c] = out[c];
            }
        } else
#pragma unroll 1
        for (unsigned r = 0; r < Chunk; ++r) {
            float k[K];
#pragma unroll
            for (unsigned j = 0; j < K / 4U; ++j) {
                const float4 v4 = key[j * QuadStep];
                k[4U * j] = v4.x; k[4U * j + 1U] = v4.y; k[4U * j + 2U] = v4.z; k[4U * j + 3U] = v4.w;
            }
            // Pinned (Nan >= 2): decay a per-lane VGPR multiplied as src1 (production lanes kernel ISA: retained src0,
            // decay src1; an SGPR decay would have to be src0, and the compiler also commutes VGPR operands to pair VOPD).
            const float d = kPinDecay<Nan> ? *decay : uniform<Opt>(*decay), b = uniform<Opt>(*beta);
            float delta[Cc];
#pragma unroll
            for (unsigned c = 0; c < Cc; ++c) {
                if constexpr ((Opt & 4U) != 0U) {
#pragma unroll
                    for (unsigned j = 0; j < K; ++j) {
                        if constexpr (kPinDecay<Nan>) retained[c][j] = mul_src0(retained[c][j], d);
                        else retained[c][j] = retained[c][j] * d;
                    }
                    const float sum = gdn_post_tree<K>([&](auto jj) {
#pragma clang fp contract(off)
                        return retained[c][decltype(jj)::value] * k[decltype(jj)::value];
                    });
                    delta[c] = (value[c] - gdn_lane_group_sum<G>(sum)) * b;
                } else {
                float products[K];
#pragma unroll
                for (unsigned j = 0; j < K; ++j) {
                    if constexpr (kPinDecay<Nan>) retained[c][j] = mul_src0(retained[c][j], d);
                    else retained[c][j] = retained[c][j] * d;
                    products[j] = retained[c][j] * k[j];
                }
                delta[c] = (value[c] - gdn_lane_group_sum<G>(gdn_lane_tree(products))) * b;
                }
            }
            // Opt bit 8: keep the query loads below the first phase (register pressure; scheduling only)
            if constexpr ((Opt & 8U) != 0U) __builtin_amdgcn_sched_barrier(0);
            float qv[K];
#pragma unroll
            for (unsigned j = 0; j < K / 4U; ++j) {
                const float4 v4 = query[j * QuadStep];
                qv[4U * j] = v4.x; qv[4U * j + 1U] = v4.y; qv[4U * j + 2U] = v4.z; qv[4U * j + 3U] = v4.w;
            }
            float out[Cc];
#pragma unroll
            for (unsigned c = 0; c < Cc; ++c) {
                if constexpr ((Opt & 4U) != 0U) {
#pragma unroll
                    for (unsigned j = 0; j < K; ++j) retained[c][j] = __builtin_fmaf(k[j], delta[c], retained[c][j]);
                    out[c] = gdn_lane_group_sum<G>(gdn_post_tree<K>([&](auto jj) {
#pragma clang fp contract(off)
                        return retained[c][decltype(jj)::value] * qv[decltype(jj)::value];
                    }));
                } else {
                float products[K];
#pragma unroll
                for (unsigned j = 0; j < K; ++j) {
                    retained[c][j] = __builtin_fmaf(k[j], delta[c], retained[c][j]);
                    products[j] = retained[c][j] * qv[j];
                }
                out[c] = gdn_lane_group_sum<G>(gdn_lane_tree(products));
                }
            }
            if (sub == 0U) {
#pragma unroll
                for (unsigned c = 0; c < Cc; ++c) out_row[c] = out[c];
            }
            key += RowStep; query += RowStep; value += Columns; ++beta; ++decay; out_row += Columns;
        }
        if constexpr (DeferNorm) {
            // Packed conv now reads vraw during recurrence: no wave may commit over another wave's taps.
            // Later chunks already rendezvous in finish_pending; chunk zero has no pending norm to finish.
            if constexpr (PackedValue) { if (n == 0U && more) __syncthreads(); }
            if (n) finish_pending(stage[buffer ^ 1U].value, first - Chunk, n - 1U, PendingSigmoid);
            if (more) commit(stage[buffer ^ 1U], first + Chunk);
            publish_pending(stage[buffer].value, first, n);
            if constexpr (!PackedValue) { if (more) convolve(stage[buffer ^ 1U], first + Chunk); }
            __syncthreads();
            continue;
        }
        if constexpr (Lean != 0U) {
            static_assert(!Pipe && Chunk % W == 0U);
#ifndef QPF2K_GDN_BENCH_DIAG
            static_assert(Lean == 1U);
#endif
            constexpr unsigned PerWave = Chunk / W;
            float (&outs)[Chunk][Columns] = stage[buffer].value;
            // z of this chunk and the norm weights (independent of the exchange): loads in flight across it.
            float zv[PerWave][2], nw[2];
#pragma unroll
            for (unsigned part = 0; part < 2U; ++part) {
                const auto channel = part * 64U + 32U * half + lane;
                nw[part] = load_small(norm_weights, norm_kind, channel);
#pragma unroll
                for (unsigned i = 0; i < PerWave; ++i)
                    zv[i][part] = rows.z[static_cast<std::uint64_t>(first + wave + i * W) * rows.stride + value_head * 128U + channel];
            }
            if constexpr (Single) {
                // One stage (buffer 0): the lean sequence with the commit after a barrier and the conv after the outputs
                // are in registers. Ring slot reuse waits for the peer's consumed counter, as below.
                if (n >= kNormRing && thread == 0U)
                    while (load_acquire(peer_con) < xch.base + n - kNormRing + 1U) __builtin_amdgcn_s_sleep(1);
                __syncthreads();   // every token-loop read of the stage and every output write done; ring slot n free
                if (more) commit(stage[0], first + Chunk);   // next chunk's q/k/beta/decay and v rows (slots of rows first-3..)
                float mine[PerWave], own[PerWave][2];
#pragma unroll
                for (unsigned i = 0; i < PerWave; ++i) {
                    const unsigned r = wave + i * W;
                    own[i][0] = outs[r][lane]; own[i][1] = outs[r][32U + lane];
                    const float lo = own[i][0], hi = own[i][1];
                    mine[i] = gdn_window_add(lo * lo, hi * hi);
                    my_ring[((n % kNormRing) * Chunk + r) * 32U + lane] = mine[i];
                }
                __syncthreads();   // ring rows and the commit complete; every read of the outputs done
                if (thread == 0U) { __threadfence(); store_release(my_pub, xch.base + n + 1U); }
                // next chunk's v conv while the peer publishes (overwrites the value slots: the outputs are in own)
                if (more) convolve(stage[0], first + Chunk);
                if (thread == 0U) while (load_acquire(peer_pub) < xch.base + n + 1U) __builtin_amdgcn_s_sleep(1);
                __syncthreads();   // peer sums published; the conv complete before the next token loop
#pragma unroll
                for (unsigned i = 0; i < PerWave; ++i) {
                    const unsigned r = wave + i * W;
                    const float peer = __hip_atomic_load(&peer_ring[((n % kNormRing) * Chunk + r) * 32U + lane], __ATOMIC_RELAXED,
                        __HIP_MEMORY_SCOPE_AGENT);
                    float sum = half == 0U ? gdn_window_add(mine[i], peer) : gdn_window_add(peer, mine[i]);
                    sum = gdn_window_add(sum, gdn_window_down<16U>(sum));
                    sum = gdn_window_add(sum, gdn_window_down<8U>(sum));
                    sum = gdn_window_add(sum, gdn_window_down<4U>(sum));
                    sum = gdn_window_add(sum, gdn_window_down<2U>(sum));
                    sum = gdn_window_add(sum, gdn_window_down<1U>(sum));
                    sum = __int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(sum)));
                    float head_dim_f;   // opaque divisor: the production runtime IEEE division (see finish)
                    asm volatile("v_mov_b32 %0, 0x43000000" : "=v"(head_dim_f));
                    const float inv = rsqrtf(sum / head_dim_f + epsilon);
                    const std::uint64_t base = static_cast<std::uint64_t>(first + r) * rows.stride + value_head * 128U;
#pragma unroll
                    for (unsigned part = 0; part < 2U; ++part) {
                        const auto channel = part * 64U + 32U * half + lane;
                        const float gate_value = 1.0F / (1.0F + expf(-zv[i][part]));
                        const float normed = gated_norm_product(own[i][part], inv, nw[part], gate_value);   // outs[r][part * 32 + lane]
                        if constexpr (F16) {
                            // Same FP32 value and RNE conversion as tiled kF32 staging; no FP32 reader on this route.
                            xch.core16[static_cast<std::uint64_t>(first + r) * xch.core16_pitch + value_head * 128U + channel] =
                                static_cast<_Float16>(normed);
                            // Each aligned 8-lane group is one row's 8 consecutive channels (including across heads).
                            const auto bad = __ballot(isfinite(normed) && !(fabsf(normed) <= 65504.0F));
                            if (bad != 0ULL && lane % 8U == 0U && ((bad >> lane) & 0xffULL) != 0ULL)
                                atomicAdd(xch.core16_overflow, 1U);
                        } else rows.core[base + channel] = normed;
                    }
                }
                __syncthreads();   // every peer ring read done before the consumed release
                if (thread == 0U) store_release(my_con, xch.base + n + 1U);
                continue;
            }
            if (more) commit(stage[buffer ^ 1U], first + Chunk);
            if constexpr ((Lean & 32U) != 0U) {
                __syncthreads();
#pragma unroll
                for (unsigned i = 0; i < PerWave; ++i)
#pragma unroll
                    for (unsigned part = 0; part < 2U; ++part)
                        rows.core[static_cast<std::uint64_t>(first + wave + i * W) * rows.stride + value_head * 128U + part * 64U + 32U * half + lane] =
                            outs[wave + i * W][part * 32U + lane] + zv[i][part] * nw[part];
                if (more) convolve(stage[buffer ^ 1U], first + Chunk);
                __syncthreads();
            } else {
                // publish: ring slot reuse waits for the peer's consumed counter; the own sums stay in registers.
                if (n >= kNormRing && thread == 0U && (Lean & 16U) == 0U)
                    while (load_acquire(peer_con) < xch.base + n - kNormRing + 1U) __builtin_amdgcn_s_sleep(1);
                __syncthreads();   // also: the commit above (v ring) is complete before the conv below
                float mine[PerWave];
#pragma unroll
                for (unsigned i = 0; i < PerWave; ++i) {
                    const unsigned r = wave + i * W;
                    const float lo = outs[r][lane], hi = outs[r][32U + lane];
                    mine[i] = gdn_window_add(lo * lo, hi * hi);
                    if constexpr (Witness) nan_seen |= is_nan_bits(mine[i]);   // NaN iff lo or hi is NaN
                    my_ring[((n % kNormRing) * Chunk + r) * 32U + lane] = mine[i];
                }
                __syncthreads();
                if (thread == 0U && (Lean & 16U) == 0U) { __threadfence(); store_release(my_pub, xch.base + n + 1U); }
                // next chunk's v conv while the peer publishes (writes only the other stage's value slots)
                if (more) convolve(stage[buffer ^ 1U], first + Chunk);
                if (thread == 0U && (Lean & 16U) == 0U) while (load_acquire(peer_pub) < xch.base + n + 1U) __builtin_amdgcn_s_sleep(1);
                __syncthreads();
#pragma unroll
                for (unsigned i = 0; i < PerWave; ++i) {
                    const unsigned r = wave + i * W;
                    const float peer = (Lean & 16U) != 0U ? mine[i]
                        : __hip_atomic_load(&peer_ring[((n % kNormRing) * Chunk + r) * 32U + lane], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                    float sum = half == 0U ? gdn_window_add(mine[i], peer) : gdn_window_add(peer, mine[i]);
                    sum = gdn_window_add(sum, gdn_window_down<16U>(sum));
                    sum = gdn_window_add(sum, gdn_window_down<8U>(sum));
                    sum = gdn_window_add(sum, gdn_window_down<4U>(sum));
                    sum = gdn_window_add(sum, gdn_window_down<2U>(sum));
                    sum = gdn_window_add(sum, gdn_window_down<1U>(sum));
                    sum = __int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(sum)));
                    float head_dim_f;   // opaque divisor: the production runtime IEEE division (see finish)
                    asm volatile("v_mov_b32 %0, 0x43000000" : "=v"(head_dim_f));
                    const float inv = rsqrtf(sum / head_dim_f + epsilon);
                    const std::uint64_t base = static_cast<std::uint64_t>(first + r) * rows.stride + value_head * 128U;
#pragma unroll
                    for (unsigned part = 0; part < 2U; ++part) {
                        const auto channel = part * 64U + 32U * half + lane;
                        const float gate_value = 1.0F / (1.0F + expf(-zv[i][part]));
                        const float normed = gated_norm_product(outs[r][part * 32U + lane], inv, nw[part], gate_value);
                        if constexpr (Witness) nan_seen |= is_nan_bits(normed);
                        if constexpr (F16) {
                            // Same FP32 value and RNE conversion as tiled kF32 staging; no FP32 reader on this route.
                            xch.core16[static_cast<std::uint64_t>(first + r) * xch.core16_pitch + value_head * 128U + channel] =
                                static_cast<_Float16>(normed);
                            // Each aligned 8-lane group is one row's 8 consecutive channels (including across heads).
                            const auto bad = __ballot(isfinite(normed) && !(fabsf(normed) <= 65504.0F));
                            if (bad != 0ULL && lane % 8U == 0U && ((bad >> lane) & 0xffULL) != 0ULL)
                                atomicAdd(xch.core16_overflow, 1U);
                        } else rows.core[base + channel] = normed;
                    }
                }
                __syncthreads();
                if (thread == 0U && (Lean & 16U) == 0U) store_release(my_con, xch.base + n + 1U);
            }
            continue;
        }
        if constexpr (Pipe) {
            // chunk n - 1's outputs are still in the other stage (its next commit is below, after the barriers in publish)
            if (n) finish(stage[buffer ^ 1U].value, first - Chunk, n - 1U);
            publish(stage[buffer].value, n);
            if (more) commit(stage[buffer ^ 1U], first + Chunk);
            __syncthreads();
        } else {
            if (more) commit(stage[buffer ^ 1U], first + Chunk);
            norm_chunk(stage[buffer].value, first, n);     // barriers: the ring writes above are complete before the conv below
        }
        if (more) { convolve(stage[buffer ^ 1U], first + Chunk); __syncthreads(); }
    }
    if constexpr (DeferNorm) finish_pending(stage[(n - 1U) % 2U].value, (n - 1U) * Chunk, n - 1U, false);
    if constexpr (Pipe) finish(stage[(n - 1U) % 2U].value, (n - 1U) * Chunk, n - 1U);
    // conv history of this block's v channels: the last 4 raw rows (ring slots).
    std::uint16_t* const conv_target = Witness ? xch.conv_out : conv_state;
    float* const state_target = Witness ? xch.state_out : state;
    for (auto i = thread; i < Columns * kGdnWindowKernel; i += Threads) {
        const auto lc = i / kGdnWindowKernel, t = i % kGdnWindowKernel;
        conv_target[static_cast<std::uint64_t>(value_offset + column_of(lc)) * kGdnWindowKernel + t] = vraw[(count - kGdnWindowKernel + t) % Ring][lc];
    }
#pragma unroll
    for (unsigned c = 0; c < Cc; ++c)
#pragma unroll
        for (unsigned j = 0; j < K; ++j) {
            if constexpr (Witness) nan_seen |= is_nan_bits(retained[c][j]);
            state_target[gdn_window_state_index<Storage>(value_head, sub + j * G, column + c, 128U)] = retained[c][j];
        }
    if constexpr (Witness) {
        // one relaxed agent-scope atomic OR per wave; the replay launch follows on the same stream (kernel boundary)
        if (__builtin_amdgcn_ballot_w32(nan_seen) != 0U && lane == 0U)
            __hip_atomic_fetch_or(xch.witness, 1U, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
}
}  // namespace qpf2k_gdn
