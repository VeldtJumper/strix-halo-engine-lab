# Engine reference

This page holds the precise technical material for the engine: the measured decode configuration, every
build option, the complete command line, the draft vocabularies, the environment switches and the repository
layout. The [README](../../README.md) is the short, user-facing introduction; nothing here contradicts it, this
page only goes deeper.

- [Speculative decode configuration](#speculative-decode-configuration)
- [Decode by workload (earlier measurement)](#decode-by-workload-earlier-measurement)
- [Build options](#build-options)
- [Command line](#command-line)
- [Prompt prefill](#prompt-prefill)
- [Draft vocabulary](#draft-vocabulary)
- [Environment switches](#environment-switches)
- [Host-only tools](#host-only-tools)
- [Repository layout](#repository-layout)
- [Documentation map](#documentation-map)
- [Working rules](#working-rules)
- [Multi-node direction](#multi-node-direction)

## Speculative decode configuration

Decoding is speculative. The MTP head drafts a few tokens; the target model then verifies them in one pass over
a window of 2, 4 or 8 rows and commits the longest accepted prefix. Every multi-row kernel performs each row's
arithmetic exactly as the single-row kernel does, so the emitted tokens are bit-identical to serial greedy
decode (D-021).

The configuration behind the published decode numbers:

- **Adaptive verify windows `{2,4,8}`.** Each cycle the window size is 1 + the number of leading drafts with
  probability >= `--mtp-spec-min-p` (0.85), rounded up to an available window. An online cost model (expected
  tokens per ms for each window size, with acceptance calibrated per probability bin) can choose a smaller window
  or a serial step. Before each extra draft step the chain asks the same model whether a larger window could
  still pay off, and stops drafting when it cannot (D-022).
- **Longest-prefix acceptance with deferred GDN commit.** No state checkpoint is taken: the next window (or a
  flush before a serial step) applies the accepted recurrent rows (D-023).
- **Draft head over a vocabulary subset** (`--mtp-draft-vocab`), with Q4_0 copies of the draft projections.
  The draft only affects acceptance, never the output (D-026).
- **A8 numerical profiles** for the MoE (int8 activations, `dot4`), optionally also for the dense Q8_0
  projections (dense A8, D-032); routed-expert union per window (each expert read once for all rows, D-025);
  any-order launches for independent kernels (D-027); and the PLE table prefetched into the page cache after
  registration (D-030).

Per-cycle cost measured at `b0d1cd1`: R2 ~49 ms, R4 ~61 ms, R8 ~86 ms including the draft. The R8 cycle is
dominated by the routed-expert union, which is inherent to verifying eight exact rows. Decode is
memory-bandwidth bound: a 4-row window reads about 10.6 GB of weights.

Start-up to the first decode step takes about 11 s once the model file's SHA-256 is cached (the first start
hashes the 111 GB file, about 50 s).

## Decode by workload (earlier measurement)

Measured on one Strix Halo mini PC (Ryzen AI Max+ 395, 128 GB), UD-Q4_K_XL, A8 profile without dense A8, single stream,
256 generated tokens, at `b0d1cd1` (before D-032 and later decode work; the code figure has since risen to the
values in the README):

| Workload | tok/s | Tokens per cycle |
|---|---|---|
| Python code (raw completion prompt) | 48.5, stable across runs | 2.9 |
| English prose (500-word article, chat template, no thinking) | 37.2 | 1.9 |
| Spanish prose (same article) | 31-34 with an English-centred draft vocabulary, 37.4 with a Spanish pack | 1.7-1.9 |

<!-- BENCH: refresh this table per workload (code, English prose, Spanish prose, other languages) at the final campaign commit. -->

The generated tokens are identical in every run and identical to serial decode. The difference between
workloads is acceptance: code is far more predictable for the MTP head than prose. For languages other than
English, acceptance also depends on the draft vocabulary covering that language (see
[Draft vocabulary](#draft-vocabulary)).

## Build options

Requirements: Linux, CMake >= 3.24, Ninja, a C++20 compiler, ROCm (tested with ROCm core-10.1 under
`/opt/rocm/core-10.1`) and a `gfx1151` GPU for the HIP targets.

```sh
export PATH=/opt/rocm/core-10.1/bin:$PATH
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/bin/amdclang++ \
  -DSTRIX_ENABLE_HIP_RUNTIME=ON \
  -DSTRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME=ON \
  -DSTRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE=ON \
  -DSTRIX_USE_OPENSSL_SHA256=ON \
  -DBUILD_TESTING=ON
ninja -C build strix-hip-decode-token
```

| Option | Default | Purpose |
|---|---|---|
| `STRIX_ENABLE_HIP_RUNTIME` | OFF | Build the HIP runtime and the GPU executables, including `strix-hip-decode-token`. Without it only the host tools build. |
| `STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME` | OFF | Native Q4K/Q5K A8G32 MoE gate/up integration used by the A8 numerical profiles. |
| `STRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE` | OFF | Standalone Q4K A8G32 HIP probe and its tests. |
| `STRIX_USE_OPENSSL_SHA256` | OFF | Use OpenSSL for full-file SHA-256 verification. |
| `STRIX_WARNINGS_AS_ERRORS` | ON | Treat compiler warnings as errors. |
| `STRIX_EXPORT_OFFLINE_TUNING_PROVENANCE` | OFF | Export Ninja/FileAPI inputs for offline compile provenance (Ninja on POSIX only). |
| `STRIX_HC_REPACKED_ADMITTED_SCHEDULES` | empty | Externally admitted HC research schedules (subset of `31,32,33`). |

`ctest --test-dir build` runs the host and device tests. Many device tests need the GPU and model files.

## Command line

Qwen accepts `--profile default|fast|quality` for CLI decode, `--http-serve`,
MTP and the MTP qualification runner. Omission means `default`: it exports no
profile switches and preserves the existing exact bundle, output WMMA and
automatic prefill pair, including their existing admission guards.
`fast` applies ready rows and explicitly provisional runnable rows in the
[profile table](../../include/strix/qwen_profile_table.hpp). It currently exports
`STRIX_QPF2K_MOE_ROUTE_BF16=1` for BF16 MoE route storage without `--allow-pending`.
This row is selected ahead of its qualification receipt: it is bit-changing,
changes logits, and its qualification is in progress. The receipt reports status
`provisional` and stamps `qualification in progress: moe_route_bf16`; one stderr
line states the bit change and qualification status. Attention budget, FP16
recurrent chunks and output head remain pending because their implementations
are absent. Each keeps the builtin running, appears with its target in
`profile.pending`, and prints one stderr notice. Pending rows never cause a
refusal. `quality` currently runs default settings and prints
`quality: no quality rows for this model, using default`.

Explicit environment and lever options take precedence and are recorded as
explicit or off; explicit off prevents profile application. An explicit request
for unavailable code is refused. `--allow-pending` (or
`STRIX_QWEN_PROFILE_ALLOW_PENDING=1`) is for benchmarking: only runnable genuinely
pending rows execute, with `unqualified: <rows>` in the qualification stamp and
an `unqualified` list. If both types apply, the stamp lists qualification in
progress first, then `; unqualified: <rows>`. The three pending rows currently
remain pending even with this flag. The existing `--prefill-moe-route-bf16 on|off`
contract is unchanged. On YaRN, proposed exports are checked against rotary
admission first: a refused row stays pending with the admission reason, is not
exported, and is not stamped as applied. Explicit switches retain their existing
admission contracts.
CLI receipts, HTTP capabilities and generation stats, and `--print-effective-config`
include the same `profile` object. Selection happens immediately after parsing,
before switch readers; provenance markers preserve resolution across a future
re-exec. Cache identity changes only through the applied arithmetic switches.
Profile controls are server-level and require a restart.

The governed HC output option is independent of the product profile table.
`default`, `fast` and `quality` all retain default `hc_output_wmma`;
`--prefill-disable hc_output_wmma` or `--prefill-exact-fp16` restricts that
registry without changing the product profile or its BF16 MoE route storage
selection. To restore FP32 route storage with `fast`, also use
`--prefill-moe-route-bf16 off`. The declared `prefill_identity` names the
governed option set, while `prefill_storage_identity` describes MoE storage.
Compiled prefill arithmetic binds the option mask and hash to the complete
profile-pair identity, including the BF16 route component and GSQ arithmetic.
Prefix-cache execution signatures retain rotary identity and effective BF16
suppression state and also bind the governed option name, mask and hash.

`strix-hip-decode-token` takes a GGUF model and the prompt as token IDs; it has no tokenizer. It writes a JSON
receipt to stdout that includes `generated_tokens`, cycle counts and timings. `--help` lists the general options;
the MTP options are documented in [mtp-native-cli.md](../engine/speculative/mtp-native-cli.md).

The recommended single-request CLI command is:

```sh
strix-hip-decode-token MODEL.gguf $(cat prompt.ids) --generate 256
```

Add `--print-effective-config` to print deterministic resolution JSON and exit **before HIP initialization**.
It includes effective arguments, environment values and their provenance, T, context, budgets and profile pair.
`--baseline` / `STRIX_BASELINE=1` restores the previous minimal resolution of the same arguments.
Registered mmap residency (`--residency register`) is the default on UMA; it avoids a second full model copy.

For the pinned, content-verified Qwen checkpoint on a qualified Strix Halo target with the A8 runtime built,
at least 16 prompt tokens and positive generation, the resolver selects eager known-prefix full-row tiling,
routed expert union, router rows and terminal-row output. It chooses the largest prompt-fitting T from
8192, 4096, 2048, 1024, 256, 64 and 16, with T16384 tried first when the known prompt has at least
32,768 tokens: T8192 below that threshold and T4096 at 4K. Short tails use the admitted
tail ladder and then serial rows. Implicit context is at least 8192 and rounds prompt length plus
`max(generation, 512)` up to 512 tokens; beyond 8192 it adds `--unqualified-context` automatically.
An explicit context is preserved and retains its validation requirements.

`STRIX_QPF2K_T16K` unset uses the 32,768-token prompt threshold. Exact `STRIX_QPF2K_T16K=1`
tries T16384 first for any prompt of at least 16,384 tokens, subject to the same route and memory guards.
Any value starting with `0` disables T16384; other values except exact `1` also retain the T8192 ladder.
Ineffective values are omitted from the effective environment: the receipt shows the selected T and
historical policy, without an unused T16384 request or fallback. Unset prompts below 32,768 retain the
T8192-first ladder and arithmetic policy, but native-pool pricing at every T can
lower the admitted tile; previous budget and tile decisions are not guaranteed. When T16384 is tried, `--print-effective-config` adds `t16k_policy`
with `source` (`automatic` or `explicit`) and `threshold_tokens: 32768`, including after memory fallback.
The notice reports `automatic T16384` when selected, plus the policy source and threshold; its reason is
`automatic-t16k` or `opt-in-t16k`. Automatic selection does not inject a T16384 environment override.
Baseline, `STRIX_QPF2K_DEFAULTS=0*`, GSQ, MTP, HTTP and cohort exclusions still apply to both policies.

Admission needs unambiguous Linux UMA/GTT observations and sufficient simultaneous backing capacity.
For T8192 and smaller, the full-capacity budgets are 8 GiB workspace, 1 GiB metadata, 120 GiB UMA and 4 GiB host headroom;
smaller available memory scales them down (headroom starts at at least 1 GiB). Native pools
are priced at every T under UMA/GTT admission, raising required headroom when necessary
and potentially lowering the admitted tile. These are admission limits, not promised
allocations. T16384 prices its compiled main/tail workspace, both metadata copies and
native pools under the same limits before falling back to the smaller-tile ladder.
Explicit tiled routes must supply `--known-prefix-host-headroom-bytes` covering the
reported native host and device pools; zero or insufficient headroom is refused.
`--baseline` does not restore the old explicit-route acceptance: the pools remain reserved.
The receipt uses dense-A8 decode with
`fast-prefill-f16wmma-v1` prefill when FP16 operands and planning are admitted. FP16 prefill is its own
numerical identity, exact against its own reference, **not bit-identical to dense-A8 serial prefill**.
FP16 operand/plan refusal keeps dense-A8 tiled prefill with an arithmetic-fallback reason.
Digest equality at 32K/64K/128K/262K does not extend the 8192 model-fidelity qualification.

An admitted run prints `Qwen auto-prefill: automatic T... context=...`, including the longer-context
acknowledgement when applicable. Missing content/target/build qualification, memory observation or capacity,
ineligible prompts and controlled off values retain serial resolution with one line:
`Qwen auto-prefill declined: <reason>; using serial prefill`.

Explicit routes keep their own options, without that serial-decline notice. This includes explicit
`--mtp` or MTP cohorts, HTTP serving, serving prefix files, prompt lookup, explicit known-prefix or low-level
organization flags, and target-input export/diagnostic routes. **MTP decode and the HTTP server do not get
fast prefill automatically yet.** GSQ manifests decline the new auto route; default-injected settings are
filtered per GSQ owner, so GSQ runs without the former requirement to disable the whole default set.

Long prompts run with the default command up to the model maximum of 262,144 tokens (prompt plus generation):
the automatic context is clamped to that maximum, the admission estimate credits the PLE pages that reclaim will
free, and the tile falls back from T16384 (when eligible) to T8192, T4096 and then T2048 before the route declines,
reporting each failed tile and its reason. A prompt plus generation
above the maximum is refused (`prompt-exceeds-model-context`); a tile that does not fit the memory budget reports
`insufficient-memory`. Under the priced native pools, at 131,072 and 262,143 real-text
tokens the default admits automatic T16384, and with `STRIX_QPF2K_T16K=0` it admits T8192
as before, with no fallback.

Diagnostic and execution-control variables also decline the auto route (reason
`explicit-diagnostic-or-execution-control`, with the decline line): `STRIX_PREFILL_LOGITS_DIGEST`,
`STRIX_RTX_EXPORT_STATE`, `STRIX_QRTX_STEP_DUMP` and `STRIX_PROMPT_BATCH_ROWS` (any value other than empty or `0`).
A declined run uses the historical resolution, including its 8192-token default context: a prompt longer than
that is refused before any work ("Qwen auto-prefill declined: <reason>; prompt has N tokens, exceeding serial
prefill context C; cannot use serial prefill"); pass `--context` for a serial run, which is much slower than the
auto route. Leave these variables unset for normal runs; use
`--print-effective-config` to check which route a command resolves to.

Off switches:

- `--baseline` / `STRIX_BASELINE=1`: disable the new policy and all controlled QPF2K, PLE and wrapper defaults;
  resolve the same arguments historically (implicit context 8192, ordinary-generation exact-v5 organization,
  prepared execution, shared overlap and GDN islands). Explicit arithmetic/organization still has historical validators.
- `STRIX_PREFILL_F16=0`: decline automatic tiling and retain historical resolution.
- `--prefill-numerical-profile dense-a8`: retain automatic tiling with the dense-A8 prefill identity.
- `STRIX_QPF2K_DEFAULTS=0` (any value starting with `0`): stop default injection and decline automatic tiling; explicit environment values remain.
- Individual controlled feature values `=0` also decline automatic tiling, except the seven exact prefill
  alternatives below, whose `=0` disables only that item and keeps the route. To measure a single operator inside
  the same tiled identity, use an explicit tiled route with the same T, budgets and profile pair.
- `STRIX_PLE_RECLAIMABLE=0`: keep the original warm PLE pages; `auto` has its own route/setup admission.
- `STRIX_PLE_WINDOW_WARM=0`: stop loading the next MTP verify window's PLE pages during the draft phase
  (default on with rolling residency; never changes bytes, see [PLE verify-window warming](../optimizations.md#qwen-ple-warm)).
- `--known-prefix-window=0`: disable automatic tiling. The full-rows, expert-union, router-rows and prefill-rows
  route flags also accept `=0`.

The admitted environment includes `STRIX_DISPATCH_FUSE=1`, `STRIX_INTPREFILL_DENSE_ROWS=512`,
`STRIX_PREFILL_PIPE=255`, `STRIX_QSA_ATTEND_BATCH=1`, `STRIX_QSA_SCORE_PARK=1`, `STRIX_QSA_NORM_GROUP=1`,
`STRIX_QSA_INDEX_ROWS=1`, `STRIX_LM_HEAD_COHORT_LEAN=1`, `STRIX_LM_HEAD_ROWS_LEAN=2`, the 13 QPF2K settings
and `STRIX_PLE_RECLAIMABLE=auto`. These nine wrapper settings are injected only for the admitted decoder
route; declined MTP/serving/cohort/diagnostic/model/memory routes keep historical defaults and explicit overrides,
except that the single-session MTP route also takes `STRIX_DISPATCH_FUSE=1` for its verify windows: the explicit
MTP command below (groups 1-4, plus the known-prefix window options), with no other route, low-level or `--mtp-*`
option, cohort, HTTP or diagnostic control, on a qualified target and checkpoint. A `--mtp-golden-dump` run (which
writes the reference from an unfused run), `--mtp-metadata-only`, sampling, an explicit `--mtp-ngram-draft` mode
and session or state import keep the historical environment.
Pre-main initialization retains the historical tiled settings and route-gated PLE reclaim for other clients.
See the [full catalog](../optimizations.md#qwen-defaults) for every value and guard.

The same decoder admission also defaults seven exact prefill alternatives when unset (bitwise identical results;
deferred at startup, so MTP/HTTP serving, cohorts and other binaries keep their opt-ins; filtered per GSQ owner):

| Default | Operation and retained gate | Off switch |
|---|---|---|
| `STRIX_QPF2K_SCORE_ROWS=auto` | Stage-batch QSA scorer: 32 rows above 4096 blocks, 16 otherwise. | `STRIX_QPF2K_SCORE_ROWS=0` (16 rows) |
| `STRIX_PREFILL_PLE_BATCH=1` | PLE 64-row groups; wide lane recipe, layout and scratch admission still apply. | `STRIX_PREFILL_PLE_BATCH=0` |
| `STRIX_QPF2K_SHARED_PRE=1` | Shared-expert FP16 operand preimage; existing tiled down admission and fallback apply. | `STRIX_QPF2K_SHARED_PRE=0` |
| `STRIX_QPF2K_MOE_MC=1` | Ordered MoE merge + BF16 commit (mode 1) behind the existing DA16/ROUTE=14 certificate. | `STRIX_QPF2K_MOE_MC=0` |
| `STRIX_QPF2K_MOE_HDR=1` | Gate/up superblock header reuse within the DA16 tune 80/82 path. | `STRIX_QPF2K_MOE_HDR=0` |
| `STRIX_QPF2K_MOE_XOR64=1` | MoE 64-assignment DA16 tune (automatic tune only) with XOR-permuted LDS rows (three CTAs per WGP) from 3,072 rows; 128-assignment batches (from 4,915 rows for 512 experts, top-10) are unchanged. | `STRIX_QPF2K_MOE_XOR64=0` |
| `STRIX_QPF2K_HC_LOWBAND=3` | Banded low_t and unused low16-store omission; projection form 4 to FP32 output v3 band 16 only. Output WMMA and other routes keep mode 0. | `STRIX_QPF2K_HC_LOWBAND=0` |

MC mode 2 requires an explicit value.

The following explicit MTP configuration remains useful for speculative runs. Its low-level flags own that
route; do not append them to the recommended plain command to request defaults that are already automatic.

A speculative run has four groups of options.

**1. Target model and generation for the explicit MTP route**

```sh
strix-hip-decode-token MODEL.gguf $(cat prompt.ids) --generate 256 \
  --context 8192 --execution-mode eager --residency register \
  --optimization-profile auto \
  --numerical-profile hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1
```

The admitted production decode profile is dense A8: dense Q8_0 projections use int8 activations per group
of 32. This has a separate numerical/quality identity (D-032). Explicit arithmetic retains its profile validators.
`--prefill-numerical-profile NAME` records a prefill/decode profile pair; MTP verifies from the state that
prefill produced. See [profile identities](../engine/prefill/prefill-numerical-profile.md).

Teacher-forced scoring (`--teacher-forced-score`, `--teacher-logits-file`) also runs over the tiled prefill with the
all-rows output policy, producing byte-identical logits files to the serial route within a profile, several times
faster. See [teacher-forced-tiled-prefill.md](../engine/prefill/teacher-forced-tiled-prefill.md) (D-034).

**2. Kernel organization required by the windows** (A8 MoE, wave8 Q8 GDN, deferred MoE status, HC chain)

```
  --q51-slot-parallel-down --batched-moe-activation --q4k-a8g32-gate-up --grouped-exact-q51-down \
  --deferred-moe-status --parallel-moe-router --wave-moe-router-topk \
  --virtual-q8-gdn-input --prefetch-q8-gdn-input --virtual-q8-gdn-output \
  --wave8-q8-gdn-input --wave8-q8-gdn-output --wave8-q8-qsa-output \
  --prefetch-q8-lm-head --wave8-q8-lm-head --virtual-q8-lm-head \
  --prefetch-q8-qsa-input --prefetch-q8-shared-expert \
  --parallel-hyper-mix --staged-hyper-projections --fused-staged-hyper-mix \
  --typed-fused-hyper-projections --prefetch-typed-hyper-rows \
  --prefetch-q8-hyper-output --lane-chain-hyper-output
```

**3. MTP sidecar, provenance and budgets.** The sidecar file and the target are pinned by SHA-256, and every
memory reservation is explicit. See [mtp-native-cli.md](../engine/speculative/mtp-native-cli.md) for the meaning of each value.

```
  --mtp --mtp-sidecar MTP.gguf --mtp-sidecar-logical-name NAME --mtp-sidecar-sha256 HEX --mtp-sidecar-bytes N \
  --mtp-target-sha256 HEX --mtp-target-content HEX \
  --mtp-source-repository REPO --mtp-source-revision REV --mtp-source-path PATH \
  --mtp-converter-revision REV --mtp-converter-sha256 HEX --mtp-norm-converter-sha256 HEX \
  --mtp-conversion effective-norm-embedding-first-v1 \
  --mtp-normalized-rounding bf16-rne --mtp-projection-rounding bf16-rne --mtp-sidecar-residency register \
  --mtp-header-bytes 33554432 --mtp-sidecar-retained-bytes 3221225472 --mtp-sidecar-setup-bytes 4294967296 \
  --mtp-golden prompt.golden --mtp-golden-sha256 HEX --mtp-golden-max-bytes 67108864 \
  --mtp-replay-host-bytes 67108864 --mtp-runtime-bytes 134217728 --mtp-target-capture-bytes 16777216 \
  --mtp-policy-id 1 --mtp-candidate-depth 7 --mtp-max-hidden-error 0 --mtp-max-logit-error 0
```

MTP replay needs a reference forward for the prompt. For benchmarking, generate one with the engine itself: run
once with `--mtp-golden-dump prompt.golden` instead of `--mtp-golden`/`--mtp-golden-sha256`, then pass the file
and the printed SHA-256. A self-generated reference is reported as not independent; it does not change the
output, because every emitted token is verified by the target. Size `--mtp-golden-max-bytes` at about 1.03 MB
per prompt row (64 MiB covers a 43-token prompt); see [mtp-native-cli.md](../engine/speculative/mtp-native-cli.md).

**4. Speculative decode**

```
  --mtp-target-window-rows 8 --mtp-target-window-rows-set 2,4,8 \
  --mtp-window-arena-bytes 2147483648 --mtp-window-capture-bytes 67108864 \
  --mtp-window-checkpoint-bytes 8589934592 --mtp-window-total-bytes 17179869184 \
  --mtp-window-history-intervals 8192 \
  --mtp-window-expert-union --mtp-window-expert-union-bytes 65536 \
  --mtp-window-prefix --mtp-window-hyper-mix --mtp-window-qsa-rows \
  --mtp-draft-chain --mtp-spec-min-p 0.85 --mtp-draft-q4 \
  --mtp-draft-vocab auto:data/draft-vocab --mtp-draft-context-vocab 4096
```

| Option | Meaning |
|---|---|
| `--mtp-target-window-rows-set 2,4,8` | Window sizes available to the verifier (ascending subset of 2/4/8). Requires `--mtp-window-prefix`; `--mtp-target-window-rows` must equal the largest entry. |
| `--mtp-window-prefix` | Longest-prefix acceptance with deferred GDN commit and per-row PLE snapshots (no checkpoint restore). |
| `--mtp-window-expert-union` | Routed experts read once per window for all rows. |
| `--mtp-window-hyper-mix`, `--mtp-window-qsa-rows` | Multi-row HC mixers and QSA attention inside the window. |
| `--mtp-draft-chain` | Chained MTP drafting on the device without per-draft host syncs. |
| `--mtp-spec-min-p P` | Stop drafting after the first draft whose probability is below P; sets the window size. |
| `--mtp-draft-q4` | Q4_0 copy of the draft head over the vocabulary subset (needs a subset). |
| `--mtp-draft-vocab FILE[,FILE...]` or `auto:DIR` | Draft vocabulary: little-endian `uint32` token IDs; several files are merged without duplicates. `auto:DIR` loads `DIR/core32k.bin` plus the `DIR/pack-<lang>16k.bin` that holds the most prompt tokens (see [Draft vocabulary](#draft-vocabulary)). |
| `--mtp-draft-context-vocab N` | Also admit up to N token IDs seen in the prompt and output into the draft vocabulary. |
| `--mtp-candidate-depth D` | Maximum drafts per cycle; at least `rows - 1` of the largest window. |

## Prompt prefill

Long prompts are read by the step-major tiled prefill:

```
--known-prefix-window --known-prefix-full-rows --known-prefix-prefill-rows T \
--known-prefix-workspace-budget-bytes N --known-prefix-host-metadata-budget-bytes N \
--known-prefix-uma-budget-bytes N --known-prefix-host-headroom-bytes N
```

T is 16 to 2048 rows per block. Every complete block runs each source step for all T rows before moving on;
the unpadded suffix runs as serial rows. GDN recurrence still updates rows in position order, so the result is
bit-identical to serial prefill within one numerical profile. On this route the full organization flags of
group 2 are stated explicitly. See [tiled-prefill.md](../engine/prefill/tiled-prefill.md),
[known-prefix-prefill.md](../engine/prefill/known-prefix-prefill.md) for the budgets, and
[prefill-numerical-profile.md](../engine/prefill/prefill-numerical-profile.md) for a prefill profile that differs from decode.

The prefill numbers in the README are interleaved request throughput (prompt tokens per second of wall time
for the request) at `dff1dbe`, 2026-10-18, same binary for both profiles; the commit message records the
kernel-level changes and the base it was measured against.

## Draft vocabulary

The draft head scores only a subset of the 248,320-token vocabulary, because reading the whole Q4_0 head costs
about 0.65 ms per draft step at 106k rows and grows linearly with the row count. A token outside the subset can
never be drafted, so the subset sets acceptance but never changes the output (the target verifies over the full
vocabulary).

The repository ships the vocabularies in `data/draft-vocab/`: a 32k language-independent core (code, English,
Markdown, numbers, chat control tokens) and 16k packs for `de`, `es`, `fr`, `it`, `pt`, `ru`, `ja` and `zh`.
`--mtp-draft-vocab auto:data/draft-vocab` picks the core plus the pack holding the most prompt tokens (at least 2
tokens and 5% of the prompt; otherwise the core alone) and prints the choice; explicit lists also work:

```
--mtp-draft-vocab data/draft-vocab/core32k.bin,data/draft-vocab/pack-es16k.bin --mtp-draft-context-vocab 4096
```

Measured against a 106k general subset: on a code prompt the core needs 86 verify cycles instead of 89 at the same
speed within noise; on a Spanish chat prompt `auto` selects core + `es16k` and decodes at 38.9 tok/s instead of
32.9 (+18%). Below 16k rows acceptance drops. `tools/build_draft_vocab.py` documents the file format and builds
lists from the GGUF tokenizer and text corpora; [data/draft-vocab/README.md](../../data/draft-vocab/README.md)
describes how the shipped lists were built.

## Environment switches

Defaults are the measured configuration.

Measurement only, default off: `STRIX_BITCALC_LOGITS_OUT=PATH` writes each token
selection's FP32 logits to a fresh file, created exclusively as `PATH.partial` and
renamed to `PATH` on success. Each little-endian binary record contains six uint32
fields (`0x42434C47`, step, position, argmax, emitted token, vocab), followed by
`vocab` float32 values: `(24 + 4*vocab)` bytes per selection, bounded by the
requested generation count. Both `PATH` and `PATH.partial` must be absent.
`STRIX_BITCALC_FORCE_TOKENS=PATH` reads comma/whitespace-separated uint32 IDs,
at most one per generated token, and resumes greedy selection when exhausted.
These controls support single-request full-logits CLI generation only, excluding
serving, batching, MTP, prompt lookup, prefix cache, teacher scoring and all session
protocols. Compare completed captures with `tools/bitcalc_cmp.py`.

| Variable | Effect |
|---|---|
| `STRIX_HASH_CACHE=0` / `=DIR` | Disable, or relocate, the verified SHA-256 cache of model files (default `~/.cache/strix-engine/file-sha256`). A cached digest is reused only while the file keeps its device, inode, size, mtime and ctime, and it is still checked against the expected SHA-256. |
| `STRIX_MTP_WINDOW_COST=legacy` | Min-p window controller: EMA cost estimates without re-probes (the behaviour before D-042; default is median/minimum estimates with bounded re-probes). |
| `STRIX_MTP_REQUEST_RESET=0` | Retain learned controller costs, acceptance bins and exploration history across served requests for controlled comparisons. By default each request starts from its configured prior while reusing the slot's device owners. |
| `STRIX_MTP_DRAFT_VOCAB=2` | Draft vocabulary policy 2 with `--mtp-draft-vocab auto:DIR` (opt-in, D-043): the chosen language pack comes from `pack2-<lang>.bin` (no rows repeated from the core; 32k rows for Chinese) and a prompt without a pack gets `core64k.bin`. Tokens are unchanged. |
| `STRIX_MTP_DRAFT_CORE=<N>k` | With `auto:DIR`: use `core<N>k.bin` (32k, 48k, 64k ship) as the core. |
| `STRIX_MTP_DRAFT_WORTH=0` | Draft up to `--mtp-spec-min-p` only, without asking the window cost model before each extra step. |
| `STRIX_MTP_DRAFT_Q4_ALL=0` | Disable Q4_0 copies of the MTP draft projections. |
| `STRIX_ANYORDER=0` | Disable any-order launches of independent kernels. |
| `STRIX_PLE_WARM=0` | Skip prefetching the file-backed PLE table after registration. |
| `STRIX_PLE_WARM_PARALLEL=0` | Skip the parallel prefault of the PLE table that runs before the random-access warm (default: up to 16 threads read it in 64 MiB chunks; best effort, not with `STRIX_PLE_WARM=0`, `STRIX_MODEL_COPY_ALL` or an admitted PLE reclaim). Measured on a cold page cache: 137.8 s to listening without it, 87.5 s with it. |
| `STRIX_MTP_GREEDY_ROWS=0` | Use per-row selectors instead of on-stream argmax of all window rows. |
| `STRIX_CYCLE_TIMING=1` | Print host phase timings per MTP cycle. |
| `STRIX_CYCLE_LOG=1` | Print each MTP cycle's window rows and wall time (with the survival controller also the draft chain's wall time and steps). |
| `STRIX_MTP_TILED_START=0` | Tiled MTP prompt: run the last prompt row serially instead of starting decode from the last row of a prefill block that ends there. |
| `STRIX_MTP_CONTROLLER=survival` | Survival window controller (opt-in, D-041): the draft cut and the window choice come from a calibrated estimate of prefix survival against separated costs instead of the `--mtp-spec-min-p` cut. Tokens are unchanged. |
| `STRIX_MTP_ACTIVE_ROWS=1` | With the survival controller: the R4 owner may also verify 3 active rows and the R8 owner 6. The remaining rows are padding outside the routed expert union and acceptance. |
| `STRIX_MTP_EXPLORE=P` | With the survival controller: one more draft after a first draft whose calibrated acceptance is below P (default 0, off). |
| `STRIX_MTP_REPROBE=N` | With the survival controller: a window option unused for N cycles (default 64, 0 = never) is chosen once more and a lower cost sample replaces its estimate, so one slow sample cannot price it out for the rest of the run. |
| `STRIX_MTP_SURVIVAL_ADAPT=0` | With the survival controller: keep the offline calibration fixed (default adapts its bias online). |
| `STRIX_MTP_DRAFT_LOG=1` | Print every draft proposal (chain start, step, token, probability and runner-up probability bits) and each cycle's outcome (window rows, drafts, accepted, emitted ids). |
| `STRIX_MTP_DRAFT_TOP2=1` | Diagnostic: the draft head also reduces the runner-up probability (always on with the survival controller); with `STRIX_MTP_DRAFT_LOG=1` each step's logits are also copied to host and the runner-up token is printed. Selection is unchanged. |
| `STRIX_MTP_DIAG_FULL_CHAIN=1` | Diagnostic, never timed: every cycle drafts the full depth and verifies one serial row, so every emitted position starts a complete draft chain. |
| `STRIX_WINDOW_ROUTE_LOG=1` | Diagnostic, never timed: after every router step of a verify window, the routed expert ids of each row (`route pos= rows= row= layer= ids=` on stderr); synchronizes the device per layer. With `--mtp-spec-min-p 0` every cycle is an R8 window, which gives the routing of all eight rows. |
| `STRIX_GDN_WINDOW_TILED=1` | Window GDN recurrence with one block per (value head, 32 value columns): the state slice is read and written with coalesced loads through LDS and each wave runs `STRIX_GDN_WINDOW_COLUMNS` (2, 4 or 8; default 8) columns, so the per-row dependent chain is shorter. Key-value storage, head dim 128. Same arithmetic per column; bit-identical outputs and state. Off by default. |
| `STRIX_GDN_WINDOW_PREPARE_GATES=1` | Window GDN prepare with the beta/gate scalar transforms spread over (value head, row) pairs instead of a serial row loop per head (that block was the longest one of the prepare at R4/R8). Same per-element arithmetic; bit-identical. Off by default. |
| `STRIX_LM_HEAD_ROWS_LEAN=2` | Verify windows of 5-8 rows: lean rows LM head. Inputs staged in LDS and read as 16-byte words, two logits per wave sharing them, and the lane and local reduction trees evaluated two partial sums per register (each add keeps its operand pair). The greedy partials of every row are computed in the same kernel, so the separate pass over the logits disappears (`STRIX_LM_HEAD_ROWS_GREEDY=0` keeps it). `=1` is the one-logit-per-wave form (slower). Logits bit-identical to the R1 head; same greedy replies. Defaulted for admitted plain CLI auto-prefill; MTP/serving routes retain their previous setting. |
| `STRIX_QSA_WINDOW_POOLED=1` | Verify windows of 2/4/8 rows score the indexer blocks as prefill groups do: one pass forms every complete block's pooled, normalised and rotated key, then the pooled scoring kernel reads them (instead of pooling and reducing inside the per-block kernel with LDS trees). The block scores are bitwise the same (`STRIX_QSA_POOLED_CHECK=1` compares them). Off by default. |
| `STRIX_QSA_WINDOW_GROUPED=1` | Verify windows of 2/4/8 rows score attention with the prefill groups' KV-head-grouped kernel: each key row is read once per group of query heads. Same products and reduction order per (row, head, slot); bitwise the same scores. `STRIX_QSA_WINDOW_SLOTS=4\|8\|16` slots per wave (default 8). Off by default. |
| `STRIX_QSA_WINDOW_ATTN_V2=0` | Turns off the grouped QSA attention score of single-session MTP verify windows and the MTP draft step (on by default; `off` / `false` also accepted; off under `--baseline` and `STRIX_QPF2K_DEFAULTS=0`). When on, BF16 windows of 2/4/8 rows (head_dim 256, a multiple of 12 query heads per KV head) score with one wave per KV-head group, each key row read once per group; the draft step uses four heads per wave where it runs the one-block score kernel. Prefill, multi-session cohort windows, softmax and value/gate are unchanged; an explicit `STRIX_QSA_WINDOW_GROUPED=1` or `STRIX_QSA_SCORE_SLOTS` other than 4 keeps the selected kernel. Operand order pinned to the replaced kernels: byte-identical scores, probabilities and outputs, NaN results included (`strix-hip-qsa-window-attn-v2-tests`). |
| `STRIX_HC_LANE_SPLIT=0` | 8-row verify windows keep the LDS form of the lane-chain HC output instead of the row-split form (D-049); `=2`, `4` or `8` forces that many wave groups (also at 4 rows). Same bits. |
| `STRIX_Q6P=0` | With `--dense-q6p-pack`, every launcher keeps its Q8_0 kernel on the emulated checkpoint (A/B on one file, D-048). |
| `STRIX_SAMPLE_PATH=host\|device\|auto` | Sampled decisions on the host or the device (default auto = device, D-050); `STRIX_SAMPLE_DRAFT_HOST=1` keeps only the draft chain on the host. Same tokens. |
| `STRIX_WINDOW_TIMING=1` | Print host-side phase totals of every known window at exit. |
| `STRIX_PREFILL_TIMING=1` | Print download, scoring and spool times per block of teacher-forced tiled prefill. |

Opt-in experiments that were measured and not adopted: `STRIX_WEIGHTS_COARSE_GRAINED=1`,
`STRIX_HC_ROWS_FUSED_NORM=1`, `STRIX_LM_HEAD_ROWS_WIDE=w16|direct`, `STRIX_MOE_GATEUP_CHUNK=5`.
Kernel-level fallbacks for the newest prefill kernels (`STRIX_HC_WIDE_PROJECTION_KERNEL=0`,
`STRIX_HC_WIDE_OUTPUT_KERNEL=0`, `STRIX_DENSE_A8_LDS=0|2`, `STRIX_GDN_FUSED_A8=0`, `STRIX_PREFILL_PLE_WIDE=0`)
select the previous bit-identical kernels; they exist for A/B checks.

## Host-only tools

These need no GPU: `strix-memory-plan` (UMA admission planning), `strix-inspect-gguf` (GGUF structure and
storage), `strix-model-manifest` (tensor roles, bytes, residency and node placement), `strix-platform-probe`
(read-only discovery of RAM, KFD GPU, ROCm and USB4 topology).

## Repository layout

| Path | Contents |
|---|---|
| `src/`, `include/strix/` | Engine: GGUF reader, model contract, HIP kernels and launchers, MTP runtime and window verifier |
| `apps/` | Executables: `strix-hip-decode-token` and host tools |
| `tests/` | Host and device tests (CTest), oracles and fixtures |
| `tools/` | Draft vocabulary builder, oracles, qualification and metadata scripts, README chart generator |
| `lab/` | Standalone experiments: kernels, quantization, UMA residency, USB4/direct GPU transport |
| `data/draft-vocab/` | Shipped MTP draft vocabularies |
| `docs/` | Architecture guide, this reference and implementation notes |
| `artifacts/` | Pinned official model configuration |

## Documentation map

- [docs/architecture/README.md](../architecture/README.md): architecture guide (platform, memory, kernels, numerics, speculative decoding,
  multi-node USB4) and index of implementation notes.
- [docs/engine/speculative/verify-window-plan.md](../engine/speculative/verify-window-plan.md): multi-row verify window design and status.
- [docs/engine/speculative/mtp-native-cli.md](../engine/speculative/mtp-native-cli.md), [docs/engine/speculative/mtp-native-runtime.md](../engine/speculative/mtp-native-runtime.md): MTP
  command line and runtime contract.
- [docs/engine/decode/optimization-profile.md](../engine/decode/optimization-profile.md): automatic optimization profile and
  teacher-forced scoring.
- [docs/engine/prefill/prefill-numerical-profile.md](../engine/prefill/prefill-numerical-profile.md),
  [docs/engine/prefill/tiled-prefill.md](../engine/prefill/tiled-prefill.md),
  [docs/engine/prefill/teacher-forced-tiled-prefill.md](../engine/prefill/teacher-forced-tiled-prefill.md): prompt prefill.
- [docs/reference/qualification-ledger.md](../reference/qualification-ledger.md): what has been qualified, and how.
- [docs/decisions/decision-register.md](../decisions/decision-register.md): accepted, pending and rejected decisions with their reasons.
- [docs/reference/implementation-status.md](../reference/implementation-status.md): implemented components and how to reproduce them.
- [docs/research/history/spec-draft-v0.4.md](../research/history/spec-draft-v0.4.md), [docs/decisions/evidence-register.md](../decisions/evidence-register.md), [docs/research/history/2026-09-28-design-acceptance.md](../research/history/2026-09-28-design-acceptance.md):
  specification, evidence register and acceptance criteria.
- [docs/operations/memory-budget.md](../operations/memory-budget.md): UMA/GTT policy and admission guards.
- [docs/reference/gguf-reader.md](../reference/gguf-reader.md): GGUF reader contract and defenses.
- [docs/reference/sources-ledger.md](../reference/sources-ledger.md): canonical sources, licenses and provenance.

## Working rules

New observations go to `docs/decisions/evidence-register.md` first. Design changes are reflected in `docs/research/history/spec-draft-v0.4.md`, and the reason is
recorded in `docs/decisions/decision-register.md`. A hypothesis does not become a requirement until a measurement or an explicit
decision supports it. Native `gfx1151` qualification is required before a HIP or direct-transport path becomes a
default.

## Multi-node direction

The distributed design treats every physical USB4 link as an independent resource with its own epoch, lease,
TX/RX pools, completion accounting and health state. Striping or splitting traffic across two links is allowed
only after simultaneous measurements show the links are independent. See
[architecture/16-direct-gpu-transport.md](../architecture/16-direct-gpu-transport.md) and
[architecture/12-multi-node-and-usb4.md](../architecture/12-multi-node-and-usb4.md).

### Candidate MoE route storage

`--prefill-moe-route-bf16 on|off` (environment
`STRIX_QPF2K_MOE_ROUTE_BF16`, effective config key `prefill_moe_route_bf16`)
selects the [BF16 route storage candidate](../optimizations.md#qwen-moe-route-bf16).
It is bit-changing, opt-in, default OFF and outside READY. Effective-config inspection
reports its storage identity/hash, component mask and decline/suppression reason.
It requires the known-prefix DA16 certificate/MC=1 route and excludes serving and MTP.
Native qualification is pending; no speed claim is established.
