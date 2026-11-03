# Deterministic matrix prefill

`g4-wmma-pf-v1` uses FP16 operands and FP32 matrix accumulators on gfx1151,
wave32. It is a separate numerical identity from decode `g4-v1`.
Automatic selection remains `exact` until native arithmetic, overflow, fidelity,
continuation and complete-workload gates pass. `resolve()` accepts `auto`,
`exact`, `wmma` and the explicit WMMA identity; unknown settings fail.
The coordinator owns reading `STRIX_G4_PREFILL`, default admission and receipts.
Its optional `resolve(setting, wmma_admitted)` argument may promote `auto` only
after complete qualification. Explicit `exact` always overrides that admission.

Every linear decodes the original BF16/Q4_K/Q6_K/Q8_0 (or synthetic FP32)
coefficient with the shared `g4-v1` decoder, then rounds to FP16, RNE. Activation
conversion also uses FP16 RNE, including subnormals. No activation quantization,
clipping, per-call scaling or split-K is used. FP16 rounding overflow and
nonfinite intermediates fail the chunk before publication. Input absolute
maxima are available per linear for range diagnostics.

One output accumulator starts at +0 and consumes 16-element K tiles in increasing
order. Final incomplete K tiles contain +0 in missing positions. A 64-by-64
block shares decoded weights through LDS, with four independent 16-by-16
accumulators per wave. `STRIX_G4_PREFILL_TILING=0` selects a one-wave,
16-by-16 implementation with the same operands and instruction sequence per
output. This setting is cached; restart to change it. Arithmetic qualification
must establish both paths equal to the instruction-order host emulator.

`DevicePrefill` also prepares immutable FP16 weight panels once, before prompt
execution, using those same coefficient decoders and conversion rules. This
uses an additional two bytes per padded matrix coefficient, reported as
`prepared_bytes`; the packed matrices remain available for decode and gathers.
`STRIX_G4_PREFILL_PREPARE=0` disables this memory/performance tradeoff. The
prepared LDS path stages four consecutive K16 panels per barrier. An alternate
register path assigns sixteen accumulators to each wave. Mode `2`
uses 256-by-128 workgroups, or 128-by-128 for output widths at most 512. Mode
`1` always uses 128-by-128; `STRIX_G4_PREFILL_DIRECT=0` uses LDS. All paths retain
exactly the original per-output
instruction sequence. Modes `0` through `2` use the packed route for small head calls. All
three switches are read once per process; restart between comparisons.
Mode `3` (default) uses 64-by-64 tiles for output widths at most
2048 (`STRIX_G4_PREFILL_NARROW_MAX` changes the limit, same bytes; 2048 instead
of 1024 measured +0.7 % at 2K, flat at 512/8K), 128-by-128 below 8192, and
256-by-128 otherwise. It also uses prepared
one-wave tiles for calls of at most eight rows, including the last-row head.
Activation conversion stages `[K16][row][K]` panels in the same scratch space,
so neighboring lanes read neighboring rows without a large input stride.
The accumulator and padded-operand order are unchanged. Mode `2` restores the
previous register policy and packed small-row calls.

`g4-wmma-pf-v2` (corrected linears, "hilo") keeps the same FP16 weight panels
and splits each activation into `high = RNE16(x)` and
`low = RNE16((x - high) * 2^12)`. Two independent increasing-K16 WMMA chains
run on high and low, and each output is `h + l * 2^-12` in FP32. In
`hilo_k128` both chains restart every 128 inputs and the block parts are
added in increasing K (`s = s + part`, separately rounded FP32 adds). `emulate_linear_v2`
(`emulator.cpp`) is the specification; every device kernel (`hilo_gemm`,
`hilo_direct`, `staged_hilo`, the fused gate/up with its low panel and the
narrow tiles) equals it byte for byte in `strix-gemma4-prefill-device-tests`.
The precision is chosen per linear by a `LinearPolicy`, a list of
`regex=mode` rules (`fp16`, `exact`, `hilo`, `hilo_k128`; first match wins),
for example `ple_model=fp16;.*=hilo`. `strix-gemma4-prefill-check` takes it as
`--linear-policy`; `STRIX_G4_PREFILL_LINEAR_POLICY` sets it for any
`DevicePrefill` that runs WMMA linears.

The [AMD ISA reference](https://docs.amd.com/v/u/en-US/rdna3-shader-instruction-set-architecture-feb-2023_0)
and [AMD WMMA introduction](https://gpuopen.com/learn/wmma_on_rdna3/)
describe the opcode and register layout. They do not establish a portable
IEEE multiply/add expansion for the internal accumulator. The standalone
`strix-gemma4-prefill-characterize OUTPUT.bin` captures project-generated
instruction cases. It is a GPU executable and requires the normal exclusive
device reservation. Ordinary host FMA is not an oracle.

The measured instruction recipe is eight increasing DOT2 steps, pairing K
indices (0,1), (2,3), ..., (14,15). `emulator.cpp` implements this recipe using
integer dyadics and explicit FP32 RNE. For each DOT2 step:

1. Flush subnormal FP32 C to signed zero. Its sign remains the alignment sign.
   A normal C has magnitude integer `c = 2^23 + fraction` and exponent
   `ec = exponent_field - 150`.
2. Decode each FP16 operand as signed `m * 2^e`: normal `m = 1024 + fraction`,
   `e = exponent_field - 25`; subnormal `m = fraction`, `e = -24`.
   A product has magnitude `ma * mb`, exponent `ea + eb`, and XOR sign,
   including signed zero. Do not normalize the product before alignment.
3. Choose the common quantum exponent as the maximum of `ec - 3` for nonzero
   C and `ep - 4` for each nonzero product. Zero magnitudes do not contribute an
   exponent. All-zero inputs return +0.
4. Truncate each aligned magnitude toward zero. Start with C's aligned
   magnitude. A product with C's sign contributes `q`; an opposite-sign product
   contributes `-q - 1`, **including signed zero**. Do not repair discarded bits
   with a sticky bit or a carry. Apply C's sign to the integer sum and round once
   to FP32 RNE. This becomes C for the next pair.

The complement term explains why a one-hot multiplication need not return the
decoded coefficient exactly. Even zero products can matter. FP16 subnormal
operands and FP32 accumulator underflow are separate boundaries. Scalar norms,
residuals and nonlinearities retain gradual FP32 underflow as declared by
`g4-v1`. The committed instruction fixture and native replay gate qualify this
matrix recipe separately from whole-model fidelity/default admission.

Norms, residuals, nonlinearities, PLE embedding boundaries, RoPE and canonical
attention retain `g4-v1` semantics. Bulk norms use the target library's public primitive; the
elementwise PLE and RoPE stages preserve its operation boundaries. All
producer rows of the prompt chunk stay in the unpublished overlay.

Canonical attention evaluates the whole chunk with the kernels in
`canonical.hip`, byte-identical to the target's `gpu::attention` called row by
row (`STRIX_G4_PREFILL_CANON_KERNEL=0` restores the per-eight-row loop over that
primitive). Every value keeps its decode operation order: a score is the 32
lane chains `sum = sum + q[L + 32j] * k[L + 32j]` combined by the wave tree's
additions (visited in post-order, lanes bit-reversed); maxima use an
order-preserving integer atomicMax (only +0/-0 may differ from the decode scan,
and both give the same `declared_exp`); per-64-key-chunk denominators and
numerators are ascending sums from 0.0F merged in chunk order. Rows of a block
that do not see a key stage `e = +0`, which adds exactly nothing because these
sums are never -0. `STRIX_G4_PREFILL_CANON_PACK=0` keeps strided query/key loads
instead of the post-order packed copies; `STRIX_G4_PREFILL_CANON_FUSE=0` restores
the separate max/exp/denominator launch. All four combinations are byte-gated
by `strix-gemma4-prefill-canonical-device-tests` (its `bench` argument prints
per-phase times). `strix-gemma4-prefill-check --canonical-kernel-ab` interleaves
both canonical paths on one prefill object and requires equal committed KV
digests, last logits and taps.

Block order and shape change only which block computes which outputs, never an
output's operations, so they are byte-gated by the same device tests. The score
grid runs row tiles fastest (`STRIX_G4_PREFILL_CANON_ROWS_MAJOR=0` restores key
tiles fastest; attention 19.26 -> 16.51 s at 32K, 8K unchanged) in groups of 8 row tiles (`STRIX_G4_PREFILL_CANON_GROUP=G`, 0 =
off). The D512 values run in 512-thread blocks whose two 128-column halves share
one staged probability tile (`STRIX_G4_PREFILL_CANON_WIDE=0` restores 256-thread
blocks). On Strix Halo (gfx1151, Nemotron 3.5 Bolt E4B Q4_K_M, fast-profile
linears, one chunk, `--kernel-profile`, fresh processes interleaved), the group
and wide values together take canonical attention from 1,333 to 1,265 ms at 8K
(-5.3 %) and from 16.83 to 15.91 s at 32K (-5.5 %); OFF is
`STRIX_G4_PREFILL_CANON_WIDE=0 STRIX_G4_PREFILL_CANON_GROUP=0`. Opt-in shapes:
`STRIX_G4_PREFILL_CANON_VGROUP=G` groups the value grid likewise, and
`STRIX_G4_PREFILL_CANON_TALL=1` lets the two halves of a D512 block take two
16-row tiles that share one staged value tile (level at 8K, -1 % at 32K in one
pair). The long-context value kernel waits on LDS loads: tighter dual-issue
scheduling, 8 x 8 outputs per thread and register prefetch of the next key were
measured slower or level and are not kept.

`DevicePrefill(..., AttentionMode::canonical, Linears::exact)` is the exact
prefill: every linear is `gpu::linear_rows`/`gate_up_rows` (the decode target's
FP32 arithmetic over the whole chunk, see the kernels README) with canonical
attention and the stages above, and no FP16 panels are prepared. Its committed
KV, logits and taps equal the decode target's eight-row passes over the same
prompt (`prefill-exact`); `strix-gemma4-prefill-check --linears exact
--exact-ab` checks exactly that. `STRIX_G4_PREFILL_EXACT_MANYROW=0` runs its
linears as eight-row slices instead (same bytes, slower).

`STRIX_G4_PREFILL_ATTN=wmma` (or `g4-wmma-attn-v1`) selects an additional,
opt-in attention identity. Empty, `auto`, and `canonical` retain canonical
attention; unknown settings fail. This setting is cached at startup. The
constructor also accepts an explicit `AttentionMode`, allowing a canonical
reference in the same process. Neither numerical-band helpers nor CLI success
admit this sub-profile as a default. Its frozen recipe per query/head is:

1. Keep FP32 Q/K norms, RoPE, V norms and the authoritative FP32 cache. Round
   Q, K and V operands to FP16 RNE, preserving subnormals; reject overflow and
   nonfinite operands. Preserve GQA four and the causal/sliding visibility rule.
2. Visit absolute key tiles `[16*b,16*b+15]` in increasing order. A row omits
   entirely absent tiles. QK starts at +0 for each score and consumes increasing
   D16 tiles with the characterized instruction, padding a partial D16 with +0.
   Scores/maxima stay FP32.
3. Start `m=-inf`, `l=+0`, `o[d]=+0`. For each present tile, update `m1` by
   visiting visible scores in ascending key order, seeded by `m`. Let `a=0`
   on the first present tile; otherwise `a=declared_exp(RN(m-m1))`.
   Set `l=RN(l*a)` then, for each visible key in order, compute
   `p[j]=declared_exp(RN(score[j]-m1))` and `l=RN(l+p[j])`.
4. Round p to FP16 RNE; masked p is +0. Split the already rounded V into
   nonnegative positive and negative magnitudes, using +0 in the other part.
   Compute two PV instructions per D16 output tile, each starting at +0.
   Set `part=RN(PVpositive-PVnegative)` and
   `o[d]=RN(RN(o[d]*a)+part)`, then `m=m1`. This split prevents masked signed
   zero products from perturbing the instruction accumulator when another row
   sees additional keys. Future V signs cannot affect an earlier query.
5. Return FP32 `RN(o[d]/l)`. No atomics, split-K, adaptive tile policy, online
   scaling of operands or scheduler-dependent merging is permitted.

`attention.cpp` emulates the matrix instructions and scalar recurrence on the
host. Device tests compare every output byte, both direct and packed KV reads,
individual versus grouped queries, GQA, ring tags, and boundary masks. The
optional FP16 KV scratch is a per-call conversion cache; it is not a new target
cache dtype. `STRIX_G4_PREFILL_ATTN_PACK=0` disables it with identical arithmetic.
Its allocation is `attention_packed_bytes(packed_stride, width)`.

The packed route runs the split kernel by default; it issues the same
instructions in the same order per row as the original kernel. The pack stores
FP16 K rows plus V transposed and already divided into its nonnegative ± halves,
and zero-fills the K16-tile margins around the visible range so every operand
is finite. Each wave owns a K16 score chain and the row softmax in registers.
Masks are selects, and keys outside a row's range still contribute +0 products.
The query heads of a KV group share a workgroup (four for D256, two for D512),
with two (D256) or four (D512) waves per head owning disjoint output columns.
`STRIX_G4_PREFILL_ATTN_SPLIT=0` restores the original kernel and
`STRIX_G4_PREFILL_ATTN_WIDE=0` one/two waves per head. `STRIX_G4_PREFILL_ATTN_GROUP`
(2 default, 1 or 4) is a diagnostic V-prefetch depth. All give identical bytes. Layers
that read a producer's KV (24 and above) reuse its pack within a chunk
(`STRIX_G4_PREFILL_PACK_REUSE=0` repacks). On the panel route, the k and v projections of a
KV-producing layer read the FP16 panels that q's conversion of the same
normalized input left in scratch, instead of converting it twice more (same
operands, same bytes; their recorded input maximum is q's).
`STRIX_G4_PREFILL_QKV_SHARE=0` converts per projection. `strix-gemma4-prefill-attention-device-tests
--bench` times the last 512-row chunk of an 8K prompt (diagnostic only).

Other exact speed switches of the matrix route:
- `STRIX_G4_PREFILL_FUSE_FFN=0` turns off the fused gate/up kernel. When on, it
  forms `gelu_tanh(gate)*up` in FP32 and writes down's FP16 input panels with
  convert's rounding, overflow flag and abs maximum.
- `STRIX_G4_PREFILL_STAGED=0` turns off the LDS-staged GEMM. It is used only for
  long-K projections (down).
- `STRIX_G4_PREFILL_DEFER_HEAD=0` computes the head for every chunk instead of
  only the prompt's last row. A prompt ending on a chunk boundary evaluates the
  same head on the saved final-norm row.

Descriptors for all layers are uploaded once per chunk.
`strix-gemma4-prefill-check --warmup N` adds untimed repetitions, and each length
prints the committed-KV digest and a last-logits FNV. The low-level attention API requires consecutive positions in one
stream, identical descriptors except position, an absolute stride beyond the
last position, and an optional key capacity bounding the visible key union.

`DevicePrefill::run(stream, tokens)` commits chunks ending at absolute multiples
of 512, plus the final partial chunk. `Prompt::append(fragment)` provides the
same policy for arbitrarily split scheduler input; only `finish()` ends the
prompt. Masked matrix padding never becomes KV. These objects are used
sequentially, with no concurrent mutation of their stream. A failure closes the
prompt; earlier committed chunks remain valid, the failed overlay is discarded.
An empty prompt returns the unchanged frontier and empty outputs.
An optional `DraftContext*` appends every newly committed chunk in groups of at
most eight rows. Use this for extending an existing stream by more than 1024
known rows. The draft context must correspond to the initial target frontier;
a prefix cache hit must restore its context or retained taps first. If a draft
append throws after target publication, the target prefix stays committed;
the caller must discard or rebuild that draft context before continuing.

The result contains capped last-row logits and the last 1024 committed rows of
the five target taps, with absolute positions. Append those taps to draft
context once, before target decode. `forward(transaction, tokens, all_logits)`
is an uncommitted diagnostic entry point, limited to 512 rows. It uses the
transaction descriptors exposed by the target library, which owns publication and cache layout.
`Residency::device` borrows immutable resident weight views; `Residency::host`
uploads them once. Host table and model-view geometry are checked in either
case. The borrowing caller retains and validates the weight allocation.
`borrow_device_weights()` exposes read-only device-address views for a second
prefill with `Residency::device`; its owner must outlive that borrower. These
views cannot be given to host readers or host-residency constructors. The
canonical-reference diagnostic shares packed weights this way; each instance
still owns its FP16 panels and request scratch.

The coordinator hook is:

```cpp
if (prefill::resolve(setting) == prefill::Mode::wmma) {
    prefill::Prompt prompt(matrix_prefill, target_stream, &draft_context);
    for (auto fragment : prompt_fragments)
        prompt.append(fragment);
    auto ready = prompt.finish();
    next_logits = std::move(ready.last_logits);
    prefill_identity = prefill::identity;
}
```

Alternatively, with a fresh draft context, omit the callback and call
`append_tail(draft_context, ready, model.hidden)` once after the complete initial
prefill. Do not append the tail again when the callback was supplied.

Keep decode, matrix-prefill and attention identities in cache keys and receipts.
The attention identity is `g4-v1` for canonical or `g4-wmma-attn-v1` for WMMA,
available from `DevicePrefill::stats().attention`. Subsequent
decode, row batching and DFlash must compare against this same prefilled state.
WMMA-prefilled output is not required to match exact-prefilled output.
`Reference` is the slow host graph, with a separate transactional overlay and
canonical logical KV digest. It uses the instruction emulator for matrix
prefill and declared `g4-v1` row dots for decode from that state. Its text graph
and embedding boundaries follow the supplied HF 5.21 Gemma 4 implementation
and the [official model assets](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16).
This reference is intended for small identity fixtures or explicit offline work.

Host tests need no Python, model assets or device. Device tests use small
synthetic tensors and check packed lookup/layout, reference/shared tiling,
scheduler splits, ring boundaries, shared KV producers, rejected overlays and
canonical decode after prefill. Full-model diagnostics are separate:

```sh
strix-gemma4-prefill-check --model MODEL --rope TABLE_DIR --fixture PROMPT.npz \
    --same-weight --split 13 --continuation 32
strix-gemma4-prefill-check --model MODEL --rope TABLE_DIR --fixture PROMPT.npz \
    --length 2048 --repetitions 3
```

The length override repeats fixture IDs for throughput diagnostics. It supplies
no long-context fidelity reference. CLI success means diagnostics completed;
it does not admit a default. Full-vocabulary FP64 `KL(reference || candidate)`,
top-1, nearest-rank p99 KL, logit RMSE and maximum error are reported separately.
The fidelity helpers check numerical bands only; corpus coverage, asset
provenance and native/speculation gates remain separate. Supplied top-K/lse
teacher records cannot replace full-vocabulary logits for KL. Missing reference
is never an exactness or fidelity pass. Ordinary FP32 embedding evaluation and
the required BF16 embedding boundary must be identified separately.
Repeat `--fixture` to reuse one resident model across a multilingual pilot.
Repeat `--length` to benchmark several lengths without reloading weights.
`--profile` records cumulative linear and canonical-attention event times; such
instrumented runs are distinct from the normal throughput measurements.
It also records each named linear's elapsed milliseconds, arithmetic TFLOPS,
and minimum algorithmic GB/s (not measured device-memory traffic). The explicit
`strix-gemma4-prefill-matrix-bench` checks full-matrix byte equality and measures
the principal linear shapes independently of attention and model loading.
Throughput runs also sample the active `pp_dpm_sclk` entry every 20 ms and
report the observed minimum, maximum and mean; zero samples means unavailable.
Sampling is read-only and does not set clocks or host parameters.
`--check-c4 --continuation 32` compares four unequal-context streams with solo
decode, including logits, taps, token selection and committed-state digests.
`--draft-model DRAFT.gguf --draft-rope DRAFT_TABLES --continuation 32` additionally
tests the public DFlash cycle at k=0/1/7 against plain decode from the identical
WMMA state. Recent taps append in groups of at most eight, after prefill commit.
These short diagnostics do not replace the complete speculative corpus gate.
Label HF fixtures explicitly with `--hf-embedding ordinary-fp32` or
`--hf-embedding bf16-boundary`; the default is `unspecified`.
`--canonical-reference` compares full-vocabulary teacher logits against WMMA
linears with canonical attention on the same weights. This isolates the
attention sub-profile from the separate `--same-weight` prefill-exact comparison.

For a bounded diagnostic turn, `--after-benchmark-checks` first measures the
explicit lengths, then runs the requested split/continuation/C4/draft checks on
the original fixture. It reuses one loaded model. `--compare-attention-profile`
requires `STRIX_G4_PREFILL_ATTN=wmma` and adds separate instrumented canonical
and WMMA attention runs at each explicit length of at least 2048; ordinary timing
repetitions remain uninstrumented.
The profile objects borrow packed weights but each owns another FP16 panel set.
Timing records include Unix start/end seconds for matching an external power
sampler; this does not replace the steady clock used for elapsed time.

Profiling separates activation conversion and matrix event times. `matrix_tflops`
uses only the matrix interval; ordinary `tflops` includes conversion. Optional
`--counter` on the matrix diagnostic runs one prepared gate/up shape without
warm-up or byte qualification, for an external device counter collector.

`strix-gemma4-prefill-teacher --model MODEL_DIR --rope TABLES --parent
BOUNDARY.npz ORDINARY.npz [--parent ...] [--continuation 64]` is a separate
BF16 full-model qualification tool. It compares exact prefill, matrix prefill
with canonical attention, and matrix prefill with the WMMA attention profile
on the same teacher-forced inputs. Output contains every counted row's top-1,
same-weight full-vocabulary KL, sparse head/tail KL lower bound, and sampled
HF full-vocabulary KL. The final beyond-document prediction is excluded.
FP16 teacher rows use their own logsumexp; sparse FP32 normalizers never
normalize rounded full rows. HF error magnitudes are diagnostic outputs.
Each parent also checks a scheduler split against committed state and reports
independent greedy continuation agreement relative to exact prefill. Bound
the number of parents per invocation to the external reservation budget.
No fixture corpus, comparative receipt, or admission decision is embedded in
the executable. Frozen corpus coverage and native instruction/state gates
must be assessed together with numerical bands before changing any default.
