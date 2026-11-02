# Optimizations

## Qwen product profiles

`--profile default|fast|quality` selects the Qwen product policy for decode,
prefill, MTP and HTTP serving. `default` preserves current production behavior
without exporting profile switches. `fast` applies ready and explicitly provisional
runnable levers in the [single profile table](../include/strix/qwen_profile_table.hpp).
Where the route admits it, fast currently applies BF16 MoE route storage by exporting
`STRIX_QPF2K_MOE_ROUTE_BF16=1`, without `--allow-pending`. This bit-changing row
changes logits and is selected ahead of its qualification receipt; qualification
is in progress. Its receipt status is `provisional` and it is listed in
`profile.provisional`, with qualification stamp
`qualification in progress: moe_route_bf16`, and exactly one stderr line states
that it is bit-changing and qualification is in progress. On other routes, including
HTTP and MTP, the row stays `pending` with the same reason as an explicit request's
decline, keeps FP32 route storage and is not exported or stamped as applied. If no
provisional or unqualified row runs, the qualification stamp is `qualified`.
Attention budget, FP16 recurrent chunks and output head remain pending: builtin paths continue,
with one notice per row and declared targets in `profile.pending`. The FP16 recurrent chunk row
(`STRIX_QPF2K_GDN_FP16_CHUNK`, [chunked GDN prefill](#qwen-gdn-chunk)) is runnable but awaits fidelity qualification.
`quality` runs default settings with one notice because no quality pack exists.

Explicit environment variables and lever options override the profile, including
explicit off. `--allow-pending` / `STRIX_QWEN_PROFILE_ALLOW_PENDING=1` applies only
runnable genuinely pending rows for benchmarks and stamps `unqualified: <rows>`.
When combined with provisional rows, this follows the qualification-in-progress
stamp, separated by `; `. Of the three pending rows only the FP16 recurrent chunk row is runnable,
so with this flag fast also exports `STRIX_QPF2K_GDN_FP16_CHUNK=1` and stamps `unqualified: gdn_fp16_chunk`. The existing
`--prefill-moe-route-bf16` contract is preserved. On YaRN, profile rows refused by
rotary admission stay pending with their reason and are not exported or stamped
as applied; selection alone cannot cause that refusal. Required assets remain
mandatory, and explicit requests for unavailable code refuse. Selecting a profile
does not establish qualification. All receipts, HTTP capabilities/stats and
effective config report `profile`, `qualification`, `levers`, `pending`, `provisional`
and `unqualified` inside the `profile` object. Apply profiles at startup and restart
servers to change them.
See the [command-line contract](getting-started/engine-reference.md#command-line).

The fast path is the default in all five public engines. Use the plain recommended command and inspect its
resolution before adding overrides. `ON` means enabled; `auto` means enabled when admission guards pass;
`opt-in` requires an explicit selection. An unlisted isolated gain means unmeasured, not zero.
The [JSON catalog](optimizations.json) provides the same controls for tools; the [README](../README.md#optimizations)
is the compact overview. Run commands and agent instructions are in [AGENTS.md](../AGENTS.md).

Exactness requires fixed model content, prompt, state, profile pair and capacity. Qwen fast FP16 prefill,
dense A8, GSQ, DeepSeek trellis and GLM R1b are distinct numerical identities. Speculation verifies committed
tokens against its target from the prefill handoff state; draft changes can affect acceptance without changing
that sequence. Gains from separate campaigns must not be added. Measurements below are prior device
qualification; a host-only publication build cannot establish new device exactness or speed.

## Bounded agent prefix retention

**Engine:** `deepseek`. **Control:** `--cache-profile agent|off`. **Default:** ON,
512 MiB per rank, block 256, reserve 8192 MiB, in-run reach 0, sessions only without DSpark;
disk and before-tail OFF. Native qualification: prefix-tool resumes equal fresh prefills on both
ranks (6000 / 8192 rows, logits + state digest, 16 decoded tokens); serve, fresh process per arm,
off vs default ABBA x2 with identical text and token IDs: cold 2K miss -0.14 s mean (not slower),
an 8K prompt sharing a cached 2K prefix -3.63 s (-24 %). CPU policy, planner and
persistence-refusal contracts are tested.

Both ranks admit capture before planning/copying. Host leases cap retention at 1024 MiB per
service user/node. Explicit disk uses one shared service root: 4096 MiB aggregate directory
quota, 128 MiB outstanding payloads, 16384 MiB free-space floor; separate roots/users are outside
one disk quota. Target kernels and numerical identity are unchanged. Exactness means live
continuation state, fresh logits and complete IDs, with the documented digest normalization.

OFF: `--cache-profile off` supplies no fills; explicit options remain. Hard vetoes are
`--no-prefix-cache` / `--prefix-cache-mb 0`, `--no-prefix-disk`, `--no-prefix-before-tail`,
`--no-prefix-session-snapshots` and `--no-prefix-inrun-snapshots`, independent of argv order.
`STRIX_DS41_PREFIX_DIAGNOSTICS=0` disables diagnostic counters. Every qualifying exact gain
becomes default without a percentage threshold. See [controls](engine/serving/prefix-state-cache.md#deepseek-pair-agent-prefix-cache)
and [qualification](engine/serving/ds41-prefix-cache-qualification.md).

## Baselines, inspection and exactness

| Engine | Master baseline | Inspect before device work | Scope / precedence |
|---|---|---|---|
| Qwen | `--baseline` / `STRIX_BASELINE=1` | `--print-effective-config` | Historical organization/arithmetic; controlled defaults suppressed, implicit context 8192; explicit tiled routes still require priced native-pool headroom |
| DeepSeek pair | `STRIX_DS41_BASELINE=1` | `--print-effective-config` | Stop production pack/draft/argv injection and switch the nine recent code defaults off unless explicitly set; older kernel defaults remain |
| DeepSeek RTX | `--baseline` / `STRIX_DS41_RTX_BASELINE=1` | `--print-config` | CLI baseline applies at its argument position; later flags win; env baseline applies before argv |
| MiniMax | `STRIX_MM_BASELINE=1` | `--print-effective-config` | All 22 resolver items use their off values; explicit environment wins; older built-in routes remain |
| GLM | `--baseline` / `STRIX_GLM_BASELINE=1` | `--print-effective-config` | R1, reference prefill, MTP loaded, comm decode exchange and 21 env off values; explicit overrides win; fast MoE retained for capacity |

Use a fresh process for startup-cached settings, and the same settings on both ranks where the protocol requires it.
For A/B, change one off switch at a time, first compare complete token IDs and logits/state digests, then compare
medians in interleaved runs. A baseline that changes arithmetic is a separate identity/quality comparison.
Do not add legacy opt-in flags to the default command: on Qwen they can own the explicit route and decline auto selection.

Use each engine's built-in gates now: Qwen profile-pair qualification plus operator byte checks; DeepSeek
both-rank prefill logits, greedy/verify/batch rows and transport integrity checks; RTX replica pressure,
canonical-state and plain/speculative chat gates; MiniMax golden logits, decode/verify/exchange checks;
GLM G3/G4, identity, prefill-logit and speculative-oracle gates. References must match the selected identity.
Once the bench kit lands, run `bash scripts/bench/bench.sh list`, then
`bash scripts/bench/bench.sh check --engine E --only exactness --suite quick --json` and
`bash scripts/bench/bench.sh run --engine E --only exactness --suite quick --json`.
Verdicts are `EXACT`, `MISMATCH` and `NO-REFERENCE`; the last is not a pass. The kit is not assumed present today.

### Early tiled-prefill geometry setup

**Default ON.** OFF: `STRIX_QPF2K_EARLY_SETUP=0`, `--no-early-setup`, or
config `qpf2k_early_setup = false`. CLI, config and environment restrictions combine;
`true` cannot undo another OFF. These controls preserve automatic route selection.
Eligible fresh plain CLI generation prepares immutable full-block plan validation
and prompt-independent static launcher admission before the request timer, using
a session copy and a separate static memo. Live token, position, PLE and QSA admission
still happens in the live block. MTP, HTTP, restored sessions, prefix-cache requests,
RTX state sessions and partial blocks retain their existing admission order.
This moves work out of request TTFT into startup; it does not remove work or reduce
process wall time by itself. Native local CI found 32K last-row logits bitwise
equal with early setup OFF; three-round interleaved medians were 20.09 s default
and 20.48 s OFF, a measured request-time gain. Effective config reports
`early_setup`; eligibility still determines whether the policy executes.

### Eager code-object loading

**Default ON.** OFF: `STRIX_EAGER_CODE_LOAD=0`, `--no-eager-code-load`, or config
`eager_code_load = false`. Before the first HIP runtime call the engine sets
`HIP_ENABLE_DEFERRED_LOADING=0` only when that variable is absent. Any user-supplied
value, including an empty value, wins. OFF leaves HIP's own default or user value
untouched. Inspection predicts this policy without initializing HIP or setting it.
Effective config reports `eager_code_load` and `hip_enable_deferred_loading` (null
when left to HIP). This moves first-launch code-object loading stalls out of request
time; startup grows accordingly. Native local CI found 32K last-row logits
bitwise equal with loading OFF. Three-round interleaved medians were 20.09 s
default and 20.07 s OFF, neutral within noise; no speed gain claimed.

### Automatic T16384 prefill

**Default ON at >=32,768 prompt tokens**, on the existing qualified automatic route.
`STRIX_QPF2K_T16K=0` restores the T8192-first ladder. Exact value `1` tries T16384
from 16,384 tokens; other explicit values retain the T8192 ladder. T16384 charges
workspace, metadata, native pools and mapped aliases under UMA and GTT admission.
When it does not fit, try T8192, then T4096 and T2048, preserving the fallback reason.
The tile change keeps per-output arithmetic. Native local CI found 32K last-row
logits bitwise equal with T16384 OFF. Three-round interleaved medians were
20.09 s default and 20.34 s OFF, a measured gain. T16384 at 262K and its
memory-pressure fallback remain open, nonblocking gates. Compare
full tokens, logits and committed state locally before timing. No numerical identity
tag is added. See [scratch ownership](engine/prefill/tiled-prefill.md#process-scratch-ownership).

### Exact indexer projection batching

**Default ON** for eligible ordinary prefill, with `STRIX_QPF2K_INDEX_BATCH=0` as
the OFF switch. Baseline, GSQ and `STRIX_QPF2K_DEFAULTS=0` suppress the independent
default. Explicit index rows OFF, shape and side-stream selections suppress automatic batching. Group views
are collected until their first consumer and run with the same ordered FP32 chains
and BF16 rounding; unsupported shapes use the existing exact path. This is separate
from candidate `index_proj_wmma`; disabling that numerical option keeps exact
batching. Effective config reports `index_batch`. Native local CI passed the
index-batch fixture and found 32K last-row logits bitwise equal with batching OFF.

## Bit-changing prefill options

Qwen's registry is [prefill_identity.hpp](../include/strix/prefill_identity.hpp). Each option is
**on by default once READY**, with an individual off switch and a master exact-FP16 switch.
The READY mask contains `hc_output_wmma`. The implemented candidate `index_proj_wmma` stays outside READY and is default OFF pending qualification. The default prefill identity remains
`fp16-v1+hcout16` (`prefill_identity_hash` ee2cd64684149069); `--prefill-exact-fp16` gives the plain
`fp16-v1` identity (a0094634843e16ee).

| Option | What it does | Gain | Changes bits vs FP16 identity | Measured quality | Switch off (CLI / config / API / env) |
|---|---|---|---|---|---|
| `hc_output_wmma` (on) | HC output: FP16 low values and Q8 coefficients, WMMA accumulation in ascending K16 order, same sigmoid and stream mix | Passed a pre-declared confirmation held-out set | yes | `hc_output_wmma` passed a pre-declared confirmation held-out set; its NLL endpoint is measured against a rounding-equivalent control whose per-case movement is capped at +-0.02 nats/token (a bounded control allowance, not an unbiased estimate of the uncapped difference); the absolute NLL change vs the exact reference also passed (-0.37 % NLL/token, upper bound +0.38 %). The off switch restores the exact output path. | `--prefill-disable hc_output_wmma` / `prefill_disable = ["hc_output_wmma"]` / server-level only / `STRIX_PREFILL_HC_OUTPUT_WMMA=0` |
| `index_proj_wmma` (off; candidate, outside READY) | BF16 indexer q/k WMMA with independent K16 panels and ordered FP32 panel addition; tag `idxproj16` | Pending local qualification | yes | Compiler, native execution, model fidelity and complete-workload gates pending | Opt in: `STRIX_PREFILL_INDEX_PROJ_WMMA=1`; off: `STRIX_PREFILL_INDEX_PROJ_WMMA=0`, `--prefill-disable index_proj_wmma`, config `prefill_disable = ["index_proj_wmma"]`; master exact switch also disables |

The prefill identity hash is FNV-1a 64 of the identity name. The enabled-set tag is not
executed-path evidence: OUT16 requires the fast HC component, the qualified wide shape
(hidden 2560, four streams, rank 320), QPF2K HC admission, FP32 normalized rows and
compatible output selection. Serial suffixes, incompatible normalization forms and
non-fast profiles can retain `hcout16` without executing OUT16. The once-per-process
stderr witness does not establish execution for every request.

Use `--prefill-exact-fp16`, `prefill_exact_fp16 = true`, or `STRIX_PREFILL_EXACT_FP16=1`
to switch all arithmetic options in this registry off. `--prefill-disable` accepts comma-separated names and can be repeated;
unknown or empty names are rejected. CLI/config restrictions combine with environment restrictions;
a config `false` or empty array does not re-enable an environment-disabled option.
These arithmetic controls preserve automatic route selection.

The decode-token app and its `--http-serve` mode share these startup controls.
`--config FILE` reads the arithmetic keys above and the exact startup booleans `qpf2k_early_setup` and `eager_code_load` as `key = value` assignments,
one per line, with lowercase booleans, single-line arrays of double-quoted names and `#` comments.
Unknown/duplicate keys and malformed values are rejected. For example:

```text
prefill_exact_fp16 = false
prefill_disable = ["hc_output_wmma"]
```

Receipts, `--print-effective-config`, HTTP response `strix` receipts and `GET /v1/capabilities`
report `prefill_identity`, `prefill_identity_hash` (16 lowercase hex digits), and
`prefill_options_active` (names in registry order). This identifies the option set; the existing
numerical profile pair still identifies the surrounding arithmetic. Every changed active set
must be re-qualified against its own reference. The set is declared on every route, but
`hc_output_wmma` executes only when prefill runs a fast prefill profile (stderr witness
`qpf2k HC out16 active`); on other routes, including `--baseline` and explicit dense-A8 routes,
it is inert. Equivalence never uses the declared set alone: prefix-state cache keys include the
effective prefill command line (both numerical profiles and `--baseline`), and session state
carries the compiled prefill execution identity, so routes with different arithmetic never compare equal.

Product profiles remain independent of this registry. In particular, `--profile fast`
can select provisional BF16 MoE route storage even with `--prefill-exact-fp16`;
use `--prefill-moe-route-bf16 off` to restore FP32 route storage too. The storage
descriptor and component bit remain in the profile-pair identity, which is composed
with the governed option mask/hash for compiled prefill and cache compatibility.

**Server-level only:** restart the server to change these options. Compiled prefill plans and
prefix-state caches are shared across requests; the enabled set is bound into compiled execution,
lineage and persistent-cache keys. The process-wide override has no request-scoped synchronization.
A per-call getter alone cannot isolate concurrent or resumed requests. Request fields `prefill_exact_fp16` and `prefill_disable` are refused with HTTP 400.

## Qwen

```sh
strix-hip-decode-token MODEL.gguf $(cat prompt.ids) --generate 256
```

Add `--print-effective-config` to inspect effective argv, environment/provenance, budgets and profile pair,
then exit before HIP initialization. See [command-line admission](getting-started/engine-reference.md#command-line).
The historical baseline resolves ordinary generation with exact-v5, prepared execution, shared overlap,
GDN islands and implicit context 8192; it does not automatically select dense A8 or enlarge the context.

### Certified prefix restart and in-flight joining

**Engine:** `qwen`. **Default:** disk persistence/durability on when an explicit
prefix directory is configured; joining on wherever the cache is on, acting only on
concurrent strict L0 certified batch routes (inert on the single runner).
Automatic HTTP budgets and batch L1 qualification remain off.

- **Contract:** reuse the existing v1 store, execution signature, full-T certificates
  and cold continuation. Joining requires solo groups and compatible suffix units;
  it never changes the T/tail block plan. Invalid, cancelled or unavailable producers
  fall back to lookup/cold execution. Native gate on the strict L0 batch route:
  9/9 mixed-arrival joining runs and a disk restart bitwise identical to cold;
  joiner TTFT -40 %, aggregate throughput +65 %.
- **Off switches:** `--prefix-cache-inflight off` / `STRIX_PREFIX_CACHE_INFLIGHT=0`,
  `--prefix-cache-disk off` / `STRIX_PREFIX_CACHE_DISK=0`, and
  `--prefix-cache-durable off` / `STRIX_PREFIX_CACHE_DURABLE=0`; CLI wins. Master
  cache off disables all reuse; baseline preserves the legacy cache behavior.
- **Inspect/gate:** `prefix_cache` in effective config and HTTP capabilities reports
  guarded policy. See [guards and disk contract](engine/serving/prefix-state-cache.md#certified-restart-and-in-flight-joining)
  and [mixed-arrival/restart gate](engine/serving/serving.md#prefix-joining-and-restart-qualification).

<a id="qwen-tiled"></a>

### Automatic tiled prefill, expert union and router rows

**Engine:** `qwen`. **Control:** `--known-prefix-window; --known-prefix-full-rows; --known-prefix-prefill-rows; --known-prefix-expert-union; --known-window-router-rows; --prefill-numerical-profile`. **Default:** auto.

The plain generation command resolves to the measured production organization with dense-A8 decode and fast-prefill-f16wmma-v1 prefill.

- **Preconditions:** Pinned verified checkpoint, qualified gfx1151 target, A8 runtime, >=16 prompt tokens, positive generation, eligible eager registered mapping, unambiguous UMA/GTT observations and sufficient backing capacity.
- **Automatic selection:** Largest prompt-fitting T in 16384/8192/4096/2048/1024/256/64/16; automatic T16384 starts at 32,768 prompt tokens. Context rounds prompt + max(generation,512) up to 512, minimum 8192, clamped to the model maximum (262,144); T falls back from T16384 (when eligible) through 8192 -> 4096 -> 2048 before declining (insufficient-memory when no tile fits the budget; prompt-exceeds-model-context above the maximum); --unqualified-context is automatic above 8192. Explicit context is preserved. Full budgets: 8 GiB workspace / 1 GiB metadata / 120 GiB UMA / 4 GiB headroom, scaled down with available memory (headroom starts at at least 1 GiB). Native pools are priced at every T under UMA/GTT limits and can raise required headroom or lower the admitted tile. Under this pricing, 131,072 and 262,143 real-text tokens admit automatic T16384, and T8192 as before with `STRIX_QPF2K_T16K=0`, with no fallback. Routed expert union and router rows are on. Tail owners follow the admitted ladder.
- **Fallback notice:** One admitted line reports T/context. Refusal prints `Qwen auto-prefill declined: <reason>; using serial prefill`. Explicit MTP/cohorts, serving prefix files, lookup, known-prefix/low-level flags and target-input export retain their explicit route without this notice. The diagnostic/execution-control variables STRIX_PREFILL_LOGITS_DIGEST, STRIX_RTX_EXPORT_STATE, STRIX_QRTX_STEP_DUMP and STRIX_PROMPT_BATCH_ROWS (non-empty, not 0) decline with the line above and fall back to the historical resolution, including its 8192-token default context: a longer prompt is then refused before any work (pass an explicit --context for a slow serial run). GSQ declines the new auto route. MTP decode and the HTTP server do not get fast prefill automatically yet. FP16 operand/plan refusal can retain dense-A8 tiling with an arithmetic-fallback reason. HTTP serving resolves its own automatic T8192 ([HTTP serving prefill](#qwen-serve-prefill)).
- **Exactness evidence:** Tokens and terminal FP32 logits digests match the reference for the selected profile pair, including 32K/64K/128K/262K. Fast FP16 prefill is exact relative to its own reference, not bit-identical to dense-A8 serial prefill; longer-context fidelity remains outside the 8192 qualification.
- **Off switch:** --baseline / STRIX_BASELINE=1 restores historical resolution; STRIX_PREFILL_F16=0 or --known-prefix-window=0 declines automatic tiling. --prefill-numerical-profile dense-a8 retains tiling with dense-A8 arithmetic. Controlled feature =0 or STRIX_QPF2K_DEFAULTS=0 also declines auto selection; exact operator A/B needs an explicit route with the same profile pair.
- **Measured gain:** One Strix Halo node: 32K 22.03 s = 1,487 tok/s; 131,072 tokens 100.3 s; 262,143 tokens 202.8 s = 1,293 tok/s. Previous documented default externally reported about 119 tok/s versus about 822 with union + router rows; separate from local measurements.
- **Implementation / evidence:** [Resolver](../apps/qwen_effective_config.hpp), [profiles](engine/prefill/prefill-numerical-profile.md), [results](performance.md#qwen-prefill-and-defaults)

<a id="qwen-serve-prefill"></a>

### HTTP serving prefill: automatic T8192 and 1024-row cache checkpoints

**Engine:** `qwen`. **Control:** `--known-prefix-prefill-rows; STRIX_SERVE_PREFILL; --prefix-cache-block-rows`. **Default:** ON.

HTTP serving (`--http-serve`) without an explicit tiled configuration resolves the tiled known-prefix route at T8192, with the same arithmetic as an explicit T. A configured prefix cache uses 1024-row checkpoints when 1024 divides T. Tail-size prompt units (`STRIX_BATCH_PROMPT_SMALL=1`) stay opt-in.

- **Preconditions:** Qualified native Qwen checkpoint/target, MTP assets and calibration contract, row-count-independent arithmetic, native rotary, eager execution, registered (not copied) target residency. Startup admits T only if compiled arenas, metadata, committed model backing, native primary/tail prefill owners, staging reserve and one runnable MTP slot fit observed host/GTT memory and headroom; otherwise it tries T4096 down to T16 and prints `Qwen serve-prefill: ...` with each rejection reason. `--print-effective-config` runs the same admission.
- **Exactness evidence:** Batch service gate (service off / cancel-only / on, nested admission on / off) and HTTP slots gate (1/2/4/8 clients) EXACT against a sequential explicit-T2048 reference: complete token IDs and committed target state (service/nesting also prompt state). The gates do not compare logits.
- **Off switch:** `STRIX_SERVE_PREFILL=0` with an explicit tiled configuration; `--prefix-cache-block-rows 0` (T-sized checkpoints). Explicit `--known-prefix-prefill-rows` and budgets always win; `--baseline` / `STRIX_BASELINE=1` suppresses the automatic T.
- **Measured gain:** One Strix Halo node, same binary, interleaved arms, identical token IDs, versus explicit T2048: cold TTFT 21.95 vs 22.93 s at 17K tokens (-4.3 %), 11.08/11.11 vs 11.53/11.58 s for two 8.5K prompts (-3.9 %/-4.0 %); eight staggered 8.5K prompts queued: TTFT p50 65.3 vs 66.4-71.7 s, wave wall 103.7 vs 105.2-110.2 s. T8192 uses capacity: 29 vs 55 admitted slots, a queueing tradeoff only above 29 concurrent requests.
- **Implementation / evidence:** [Resolver](../apps/qwen_effective_config.hpp), [serving](engine/serving/serving.md#default-prompt-prefill)

<a id="qwen-residency"></a>

### UMA registered residency

**Engine:** `qwen`. **Control:** `--residency register`. **Default:** ON.

Register the mmap-backed GGUF without a second full model allocation.

- **Preconditions:** UMA GPU, mappable GGUF and capacity for model/state/workspaces.
- **Automatic selection:** CLI default on UMA.
- **Fallback notice:** Registration/setup failure is an error; there is no silent full-copy conversion.
- **Exactness evidence:** Weight bytes are unchanged; compare same-profile tokens and state/logit digests.
- **Off switch:** --residency copy (requires capacity for the copy)
- **Measured gain:** No isolated measurement
- **Implementation / evidence:** [Residency](operations/model-weight-residency.md)

### Registered page lock

**Engine:** `qwen`. **Control:** `STRIX_MODEL_REGISTER_MLOCK`. **Default:** ON when `RLIMIT_MEMLOCK` covers the registered bytes.

Keep registered model pages off the reclaim LRU so host memory pressure cannot evict the process's GPU queues.

- **Preconditions:** Linux, `--residency register`, `RLIMIT_MEMLOCK` >= registered bytes.
- **Automatic selection:** Every registered model mapping; HTTP slot sizing subtracts any registered bytes that stay unlocked from `MemAvailable`.
- **Fallback notice:** `model residency: RLIMIT_MEMLOCK below ...` or `mlock of ... failed`; the pages stay unlocked.
- **Exactness evidence:** Weight bytes, kernels and arithmetic are unchanged; only page residency and the slot-capacity estimate change.
- **Off switch:** STRIX_MODEL_REGISTER_MLOCK=0
- **Measured gain:** One Strix Halo node, HTTP server: cold starts locked 42/42 vs unlocked 5 hangs in 41 (GPU queues evicted, never restored); 8 x 8500-token clients 106.1 vs 106.8 s (1.55 vs 1.5 tok/s, ABBA); 16 clients complete (320/320 tokens, 187-199 s) where the unlocked server stalled during on-demand slot growth in both legs. Automatic slots reflect the real headroom (29 -> 8 on the measured node).
- **Implementation / evidence:** [Residency](operations/model-weight-residency.md#registered-page-lock)

<a id="qwen-l1"></a>

### PLE page reclaim and prefetch admission

**Engine:** `qwen`. **Control:** `STRIX_PLE_RECLAIMABLE=auto`. **Default:** auto.

Reclaim file-backed L1 PLE pages and gather immutable rows through bounded owned O_DIRECT buffers.

- **Preconditions:** Linux, nonempty plain CLI prompt, tiled known prefill, registered file mapping, no model copies or graph promotion, no cohort/serving/planned PLE IO. The MTP CLI extension requires one eager tiled request, R2/R4/R8 verifier owners (including adaptive row sets and min-p speculation), and no wide-copy verifier, depth/survival controller, golden dump, metadata-only diagnostic, recording or sampled timing schedule. Exactly one file-backed IQ4_NL/BF16 n-gram table, valid extents and unregistered interior pages, valid prompt hashes, and a verified page drop within the existing residual bound.
- **Automatic selection:** Auto tries verified page-drop admission. Rolling opens the actual mapped inode a second time with O_DIRECT and probes aligned reads before setup drop or credit. Raw Linux io_uring syscalls use a 512-request queue and a dedicated submission/completion thread, without liburing. Sorted, deduplicated 4 KiB pages coalesce into runs of at most 128 KiB. Unsupported setup/opcode detection (ENOSYS, EPERM or EINVAL) retains the checked O_DIRECT descriptor and selects the existing 32-thread pread pool; the activation witness names the backend and fallback errno. Short aligned completions continue, EINTR/EAGAIN retry within a fixed limit, and other errors invalidate source credit after draining I/O. The setup helper submits the next GPU block first and waits for its complete read wave before issuing later-block lookahead. Small caps split that wave in position order. Queued shared runs are promoted on critical demand, and lookahead dispatch pauses while a critical request remains pending, including beyond device queue depth. Rows, including page straddles, are copied exclusively from an owned aligned arena; the rolling reader has no model-mapping pointer. The primary plain-CLI prefill owner starts block-0 staging during runtime setup, overlapping the already active bounded direct-read lookahead and subsequent setup. A single claim prevents tail owners from duplicating that prime. The c1 worker starts the next GPU block's copy before the current block's PLE GPU launches, using the idle staging slot in H=1024 source chunks, while lookahead targets two GPU blocks ahead. Each completed source chunk supplies an immutable pinned page-to-buffer view to the joined copy workers; these helpers avoid per-row ledger/cache locks while using the same row conversion and destination offsets. Plain rolling hashes independent position ranges in parallel with the same preceding token history and EOS rules. Decode retains its existing source leases and lookup paths. Plain CLI reservations protect current, future and in-flight pages; consumed pages remain cached until capacity pressure requires replacement. Smaller caps split source windows and shrink or throttle lookahead. Both plain CLI and MTP decode use a fixed replaceable 256 MiB source LRU by default, independent of generation length. MTP retains explicit turn/epoch bindings, calibration draining and fresh repeated-sample turns. After calibration drains, its prompt phase starts the same critical-first capped lookahead pipeline; preflight primes block-0 staging and c1 fills the next block at the original offsets. Joined helpers validate immutable source views against the turn, phase, mapping and file offset. Tail-owner copies share one source permit, and phase changes/cancellation drain registered staging tasks before retiring prompt pages. The prompt pipeline retires before adaptive decode reuses the source arena. Adaptive R2/R4/R8 verify gathers lease their complete physical row horizon, including min-p padding, against the same fixed decode quota until CPU staging completes. Serial fallback and rejection replay retain or re-fetch identical owned rows; speculative absolute positions never advance the prompt cursor. Min-p may select owners using measured costs: enabling residency does not freeze that schedule or establish native token/logit/state equality. The predictor receives no PLE credit. Source failures invalidate credit, and cleanup drains readers and I/O. After the one-time setup drop, rolling performs no madvise, fadvise or mincore during source gathers or decode. One table-wide mincore observes the mapping at each drained request end (and final retirement); shared boundary pages and the setup residual may remain. HTTP, cohorts, graph, copied-model, cache restore and planned-I/O routes retain their exclusions. Mapper-protected PLE interiors retain the previous plain prompt-wide path. Defaulted auto remains suppressed for GSQ owners and STRIX_QPF2K_DEFAULTS=0.
- **MTP admission and status:** Inspection predicts the envelope but always reports verified credit as zero. Runtime credit is available only after setup drop/mincore verification and is subtracted once from model backing. Plain and MTP prefill share one reservation function: verified credit already includes the cap, decode quota and margin, which are never added again. Reader/turn metadata stays separately charged. Sidecar setup has retired before this check; only its retained cap remains. The MTP window charge is bounded by its shared arena cap plus four capture caps per configured owner and the checkpoint cap only when full rollback allocates it, also bounded by the native combined total cap. Golden/replay caps are setup payload. All native limits stay enforced, with conservative total/incremental/GTT checks and a 4 GiB physical margin. Credit never augments MemAvailable or remaining GTT. Automatic MTP setup failure refuses before warming because a zero-credit fallback envelope has not been admitted. CPU source-byte/lifecycle tests do not qualify native MTP tokens, terminal logits or canonical state; GPU ON/OFF qualification remains required. No extended-context or serving qualification follows from this memory policy.
- **Rolling controls:** `STRIX_PLE_ROLLING=0` or `STRIX_CTX_EXACT_MEMORY=0` restores the previous prompt-wide mmap planner, credit and staging path on plain CLI, and whole-table warm residency with zero credit on MTP. Both switches ignore rolling tuning. `STRIX_PLE_ROLLING_POSITIONS` accepts powers of two from 256 through 65536. `STRIX_PLE_ROLLING_CAP_BYTES` selects the owned prefill cap for both rolling routes; the default is 3221225472 bytes. It must be page aligned and hold two worst-case positions; MTP additionally requires space for its H-sized wave. The aligned arena reserves cap plus decode allowance as process memory. Admission and inspection use the same cap plus decode plus 64 MiB plus cap/64 formula; metadata has a separate conservative owner reservation (512 bytes per possible arena page plus 2 MiB for reader/ledger/queue metadata, in addition to prompt IDs and turn metadata). File-cache contribution of direct source reads is zero. `STRIX_PLE_ROLLING_STATS=1` reports reads issued, bytes read, cache hits, background/demand read-wait time, owned arena and used-page peaks, and end-of-request mapping resident pages. Plain rolling also emits one stderr line per consumed block with its block index, first position, rows, bytes-read, read-wait-ms, copy-ms, hash-ms and join-wait-ms, plus critical-bytes, critical-read-ms, critical-iops, critical-MB-s and lookahead-bytes. Critical bytes and operations count newly issued reads assigned to source windows overlapping the consumed block, including setup and independent lookahead reads that prepared it. Critical read time spans their first submission through last completion; IOPS includes retry/short-read completions and MB/s uses decimal megabytes. These are application I/O rates, not device counters. Shared completed cache hits contribute zero new bytes. Lookahead bytes count completed read waves assigned beyond this block at the observation. Cap-split waves retain their original source-window attribution; overlapping partial launches can therefore include the same wave and must not be summed as independent I/O. Bytes count new reads assigned to that block's staging/demand issuer (including its setup prime); shared pending reads are counted only by their issuer, and independent lookahead remains in the summary. Read wait is caller time joining owned-page completion, copy is joined conversion/copy wall time, hash includes launcher validation, and join wait is exposed launch-thread time. Copy/read times may be hidden by setup or GPU work and must not be added to join wait. All block fields are STATS-only and never enter stdout or numerical digests. Existing gather hidden/exposed time, horizon, throttle and split-window diagnostics remain. Background read wait includes lookahead and c1 demand; it measures CPU waiting, not proof of GPU overlap. Inspection reports zero verified credit because it exits before setup.
- **Session and prefix ownership:** Session load, live continuation and a follow-up session prompt retain the public warm route; fixed MTP sessions do not acquire rolling turn leases. A plain save of the initial request may finish its current lease after synchronous capture. Prefix-cache restore/import and follow-up prompt routes retain the public prompt-wide PLE planner: their restored numerical state does not carry an owned rolling source window, and a finished owner is never reused as a new rolling prompt. Native qualification of rolling restore ownership remains pending.
  - **Fallback notice:** Automatic setup errors, including unsupported O_DIRECT opens and direct probe failures, tear down rolling readers and drop all credit. A fresh complete resident admission must pass before warming or prompt mutation; failure clearly refuses and never disables MTP. Required reclaim refuses without substitution. Post-admission read errors fail closed with a nonzero exit: mutated target/predictor state has no pristine transactional replay checkpoint, even before token publication. A fresh invocation is required. No rolling route silently substitutes buffered reads. Unsupported-route witnesses name the first applicable exclusion (for example `mtp-copy-wide-rows`, `mtp-golden-dump`, `mtp-sample-timing-schedule`, `mtp-metadata-only` or `raw-recording`). The [shipped qualification table](engine/qsa-committed-pool.md) specifies defaults and explicit opt-ins.
- **Exactness evidence:** Hashing, stored rows, IQ4_NL decoding, BF16 conversion, destination offsets and GPU arithmetic are unchanged. CPU tests cover aligned/coalesced direct reads, EOF/EINTR/read failures, queue draining, in-flight deduplication, adversarial cap and decode-LRU pressure, and literal byte equality across 200,000-position plain and MTP prefill/decode streams with the model mapping protected against reads. These tests do not establish native GPU state/logit/token identity or throughput. Native capacity, fidelity and complete-workload qualification remain required.
- **Off switch:** STRIX_PLE_RECLAIMABLE=0
- **Measured gain:** Historical prompt-wide admission supports T8192 at 128K/262K; rolling native qualification is pending. No isolated speed attribution.
- **Implementation / evidence:** [Admission](../include/strix/ple_reclaim.hpp), [implementation](../src/ple_reclaim.cpp)
- **Decode warming:** with rolling residency, MTP decode loads each verify window's PLE pages during the draft phase; see [PLE verify-window warming](#qwen-ple-warm).

<a id="qwen-ple-warm"></a>

### PLE verify-window warming

**Engine:** `qwen`. **Control:** `STRIX_PLE_WINDOW_WARM`. **Default:** ON.

Under rolling PLE residency a verify window's PLE step reads its table rows (16 per window row) with O_DIRECT
reads, about 3.4 ms per R8 window, while the GPU waits after layer 0. The MTP draft loop now reports each window
row's input token as soon as the host knows it: the pending token when the draft chain starts, and each draft after
its draft step. The row's table row ids are hashed from the last published n-gram history exactly as the window
hashes them, and a background thread loads their pages into the rolling decode cache. The window itself is
unchanged; its leased gather finds the pages cached.

- **Preconditions:** MTP decode with a rolling-residency decode turn (`STRIX_PLE_RECLAIMABLE` auto/required with
  rolling active). Without a turn (plain mapping, rolling off) the item is inert.
- **Automatic selection:** On unless `STRIX_PLE_WINDOW_WARM=0`, `STRIX_QPF2K_DEFAULTS=0` or `STRIX_BASELINE=1`.
- **Fallback notice:** Rows guessed from a stale history, rejected or padded rows and min-p stops only leave warm
  pages that the fixed 256 MiB decode cache replaces; warming errors are counted and ignored.
- **Exactness evidence:** Exact by construction: warming never hands bytes to the window. It only admits pages
  into the decode cache that the window's own lease would read; hashing, stored rows, IQ4_NL decoding, BF16
  conversion, staging and GPU arithmetic are unchanged. The fixture `strix-hip-ple-window-warm-tests` compares
  windows with and without warming bitwise (R 1/2/4/8, vocabulary-edge, EOS and repeated tokens, wrong-token and
  stale-history warming) on the plain mapping and on a rolling decode turn; decode A/B token IDs were identical.
- **Off switch:** STRIX_PLE_WINDOW_WARM=0
- **Measured gain:** 1K-token decode, MTP G70 preset, 2 interleaved rounds: code +1.1 % (58.70 -> 59.36 tok/s),
  code at R8 +0.5 %, prose +1.0 %, prose at R8 +1.0 %. Traced R8: window PLE host gather 3.4 -> 0.87 ms; the
  idle gap before the PLE projection disappears from the top idle list.
- **Diagnostics:** `STRIX_CYCLE_TIMING=1` prints `window PLE warm:` (rows warmed, pages read, warm time, window
  rows matched, warm completion relative to the window start).
- **Implementation / evidence:** [warming](../src/hip_ple_launcher.hip), [decode cache](../src/ple_rolling.cpp),
  [fixture](../tests/hip_ple_window_warm_tests.hip)

<a id="qwen-defaults"></a>

### Exact compact QSA attention output

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_COMPACT_IMAGES`. **Default:** ON for eligible full-T score-ring owners.

The already BF16-rounded attention output uses half-size BF16 storage, with exact
FP32 widening at projection and diagnostic readers. Raw GDN/QSA images and shared
decode/MTP layouts remain FP32. `STRIX_QPF2K_COMPACT_IMAGES=0` independently restores
FP32 output containers after restart. Calculated memory saving: 96 MiB per T8192 owner.
Native local CI passed host/HIP fixtures and the 32K last-row bitwise logit check.
Interleaved A/B was neutral within noise (OFF 19.95 s, n=2, versus default 20.09 s);
no speed gain claimed. Numerical identity is unchanged; physical execution identity
and workspace pricing change. The seven-state-payload comparison remains open
and nonblocking. See the
[reader audit and native gate](engine/prefill/compact-stage-images.md). No measured gain is claimed.

### Combined QPF2K and wrapper defaults

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_DEFAULTS`. **Default:** ON.

Supply unset per-item values; retain explicit overrides. The 13 QPF2K items and nine admitted wrapper settings are listed individually below.

- **Preconditions:** Eligible tiled operators and selected numerical identity; the decoder resolver admits wrappers only for the single-session automatic route.
- **Automatic selection:** Historical tiled items and route-gated PLE auto may initialize before main. Nine wrappers are deferred until decoder admission. QSA/HC masks are supplied with a newly defaulted parent enable. Injected items are tracked and suppressed per GSQ owner after lazy caching, preserving explicit environment values.
- **Fallback notice:** No general initializer notice. Operators retain guards; GSQ no longer requires disabling defaults just to run. Explicit unqualified GSQ controls still fail validation.
- **Exactness evidence:** Tokens and terminal FP32 logits digests match the reference for the selected profile pair, including 32K/64K/128K/262K. Fast FP16 prefill is exact relative to its own reference, not bit-identical to dense-A8 serial prefill; longer-context fidelity remains outside the 8192 qualification.
- **Off switch:** STRIX_QPF2K_DEFAULTS=0 in a fresh process with individual overrides unset; --baseline / STRIX_BASELINE=1 additionally suppresses controlled explicit environment items and restores historical policy.
- **Measured gain:** Combined 32K 25.95 -> 23.49 s (-9.5%); later 22.03 s is a separate measurement
- **Implementation / evidence:** [Default table](../include/strix/qpf2k_defaults.hpp)

#### The 13 QPF2K defaults

Each item inherits its linked operator's preconditions, automatic selection, fallback and same-identity gates.
An explicit `=0` disables its feature and also declines the new automatic CLI route. QSA mask `0x57`
disables only stage batching; HC mask values select folds. Use explicit tiling to hold the profile pair fixed for A/B.

| Environment | Default | Off | Operator detail |
|---|---|---|---|
| `STRIX_QPF2K_SCHED` | `14` | `STRIX_QPF2K_SCHED=0` (bit 2 only: `STRIX_QPF2K_SCHED=12`) | [Merged tails, lazy tail owners and deferred block completion](#qwen-schedule) |
| `STRIX_QPF2K_DENSE` | `1` | `STRIX_QPF2K_DENSE=0` | [Register-blocked dense projections](#qwen-dense) |
| `STRIX_QPF2K_GDN` | `1` | `STRIX_QPF2K_GDN=0` | [GDN recurrence, gates batching and core16](#qwen-gdn) |
| `STRIX_QPF2K_GDN_GATES_BATCH` | `1` | `STRIX_QPF2K_GDN_GATES_BATCH=0` | [GDN recurrence, gates batching and core16](#qwen-gdn) |
| `STRIX_QPF2K_GDN_CORE16` | `1` | `STRIX_QPF2K_GDN_CORE16=0` | [GDN recurrence, gates batching and core16](#qwen-gdn) |
| `STRIX_QPF2K_QSA` | `1` | `STRIX_QPF2K_QSA=0` | [QSA stages and stage batching](#qwen-qsa) |
| `STRIX_QPF2K_QSA_MASK` | `0x2257` | `STRIX_QPF2K_QSA_MASK=0` | [QSA stages and stage batching](#qwen-qsa) |
| `STRIX_QPF2K_HC` | `1` | `STRIX_QPF2K_HC=0` | [HC folds, output v3 and band 16](#qwen-hc) |
| `STRIX_QPF2K_HC_MASK` | `14` | `STRIX_QPF2K_HC_MASK=0` | [HC folds, output v3 and band 16](#qwen-hc) |
| `STRIX_QPF2K_HC_OUTPUT3` | `1` | `STRIX_QPF2K_HC_OUTPUT3=0` | [HC folds, output v3 and band 16](#qwen-hc) |
| `STRIX_QPF2K_HC_OUTPUT3_BAND` | `16` | `STRIX_QPF2K_HC_OUTPUT3_BAND=0` | [HC folds, output v3 and band 16](#qwen-hc) |
| `STRIX_QPF2K_MOE_DA16` | `1` | `STRIX_QPF2K_MOE_DA16=0` | [Routed MoE operand image and route setup](#qwen-moe) |
| `STRIX_QPF2K_MOE_ROUTE` | `14` | `STRIX_QPF2K_MOE_ROUTE=0` | [Routed MoE operand image and route setup](#qwen-moe) |


<a id="qwen-schedule"></a>

### Merged tails, lazy tail owners and deferred block completion

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_SCHED=14`. **Default:** auto.

Bit 4 merges eligible tail work into the main arena; bit 8 allocates only the tail owners the prompt ladder uses.
Bit 2 (SCHED bit 2, see the row below) lets a headless tiled block that another block of the same owner follows end
without its host fence; that next block runs its completion checks before it publishes anything.

- **Preconditions:** Tiled full rows; merged tails require terminal-row output and exclude the dense8 fast component. Lazy owner pruning requires a single known CLI prompt without serving, import, prefix caches, cohorts, lookup or wide-copy work.
- **Automatic selection:** The defaults bundle supplies 14 (bits 2, 4 and 8). Ineligible pruning retains the full tail-owner ladder; ineligible merging retains the configured chunks.
- **Fallback notice:** No ordinary decline line. Requesting merged/balanced scheduling with dense8 throws STRIX_QPF2K_SCHED bits 4/16 (merged / balanced prefill blocks) exclude the dense8 prefill component.
- **Exactness evidence:** The qualification campaign compared 1K code and 8K prose receipts, terminal FP32 logits FNV and generated tokens against the unchanged reference. Repeat with the same prefill/decode identity; changing arithmetic profiles is a separate quality comparison.
- **Off switch:** STRIX_QPF2K_SCHED=0
- **Measured gain:** -
- **Public evidence:** [Qwen results](performance.md#qwen-prefill-and-defaults), [chunk cursor](../src/known_prompt_chunks.cpp)

#### SCHED bit 2: deferred block completion

| Item | Detail |
|---|---|
| Default | On (part of the default `14`) on the main CLI known-prompt path |
| Off switch | `STRIX_QPF2K_SCHED=12` (keeps bits 4 and 8) |
| Measured gain | Plain command, tokens identical: 128K TTFT 84,132 -> 83,255 ms (-1.04 %, seven deferred blocks, release build); 32K neutral on the current route (2 x T16384, one boundary; five paired rounds -0.1 / -0.2 / +6.6 outlier / -0.7 / +0.1 %). On the earlier T8192 route: 32K 20,933 -> 20,734 ms (-0.95 %, three paired rounds) and 21,809 -> 21,396 ms (-1.9 %, n=4/5); 128K -0.8 % |
| Exactness | Same kernels, rows and order; only the host fence moves. A failed completion check or any exception while a block is unchecked poisons the session before any output. The last block of a prompt always fences. |
| Inert | MTP bridge, HTTP serving and cohorts never defer a block. They plan, certify and key their caches with bit 2 masked (cache-family qualification, the batch plan identity and the prefix-state storage namespace), so the default and an explicit 14 compare as 12 there and prefix-cache entries written under the former default 12 still hit. A CLI prefix-state cache run with a defaulted schedule keys the same way; a CLI run that defers binds 14. |
| Dropped | YaRN: a defaulted schedule resolves to 12 (an explicit 14 is still refused). CLI prefix-state cache: with a defaulted schedule bit 2 is dropped for the run, so full-T boundaries stay save/resume points; an explicit 14 keeps bit 2 and those runs have no save points after a deferred block. |
| Excluded | Baseline, `STRIX_QPF2K_DEFAULTS=0` and GSQ checkpoints (the same exclusion as the other QPF2K items); teacher-forced scoring and all-row heads (no headless block). |
| Batch certification | Batch certification runs on the MTP bridge, where bit 2 is inert: the default and an explicit 14 certify as 12. |

<a id="qwen-dense"></a>

### Register-blocked dense projections

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_DENSE=1`. **Default:** auto.

Reuses dense projection tiles across prompt rows while preserving the operand image and K order.

- **Preconditions:** Fast FP16 prefill dense sites, supported Q8_0 widths/segments/alignment and uniform strides; default minimum 1024 rows at GDN input and shared gate/up, 512 at other sites. The site mask defaults to all six sites; STRIX_QPF2K_DENSE_MIN_ROWS can override the threshold.
- **Automatic selection:** Selects supported sites at or above their row threshold; otherwise keeps the preceding projection.
- **Fallback notice:** Silent eligibility fallback. STRIX_QPF2K_DENSE_TRACE=1 prints qpf2k dense tiled: first launch ... when selected, not a decline line.
- **Exactness evidence:** The tiled dense fixture suite checked 120/120 bitwise cases; also use the whole-engine receipt/logit/token comparison. The source target is strix-hip-dense-f16-tiled-tests.
- **Off switch:** STRIX_QPF2K_DENSE=0
- **Measured gain:** -
- **Public evidence:** [Qwen results](performance.md#qwen-prefill-and-defaults), [dense selector](../src/hip_dense_f16_tiled.hpp)
- **WMMA register banks:** form 4 issues its WMMAs with A and B pinned to VGPR banks other than the accumulators' (STRIX_QPF2K_WMMA_PIN). It is an independent switch read once per process, not one of the QPF2K resolver defaults: on unless set to `0`, and `=0` only restores the compiler's register assignment (route selection is unchanged). Register assignment only: the same instructions on the same values. The accumulator bank is chosen by the compiler; `tools/check_wmma_banks.py` checks each build (distinct banks, tied and disjoint D, zero scratch, at most 192 VGPRs).

<a id="qwen-gdn"></a>

### GDN recurrence, gates batching and core16

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_GDN=1; STRIX_QPF2K_GDN_PARTS=55; STRIX_QPF2K_GDN_GATES_BATCH=1; STRIX_QPF2K_GDN_CORE16=1`. **Default:** auto.

Uses the exact recurrence forms, batches gates and feeds eligible dense output directly from FP16 core rows.

- **Preconditions:** Eligible GDN prefill rows and supported lane/column shapes. Core16 additionally needs dense FP16 (not dense G64), tiled rows in multiples of 32, output site enabled, at least 2048 rows by default, value width 6144, head dimension 128, uniform core/output strides and a supported dense output site; graph capture declines it. The FP16 core consumer must be admitted before the producer drops FP32 rows.
- **Automatic selection:** The initializer enables GDN, gates batching and core16; the GDN reader defaults PARTS to 55. Shape, check-mode and consumer guards retain the older forms.
- **Fallback notice:** Ineligible forms fall back without a general decline line. Check modes retain FP32 core rows. Diagnostic output is QPF2K GDN check: ... differ ... or GDN prefill check: ... differ ....
- **Exactness evidence:** STRIX_QPF2K_GDN_CHECK=1 and STRIX_GDN_PREFILL_CHECK=1 compare written bytes; require zero differences. Whole-engine receipts and operand-identity checks cover the qualified special-value/NaN order.
- **Off switch:** STRIX_QPF2K_GDN=0; independently STRIX_QPF2K_GDN_GATES_BATCH=0 or STRIX_QPF2K_GDN_CORE16=0 (one at a time)
- **Measured gain:** -
- **Public evidence:** [Qwen results](performance.md#qwen-prefill-and-defaults)

<a id="qwen-gdn-chunk"></a>

### Chunked GDN prefill on FP16 WMMA (FAST)

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_GDN_FP16_CHUNK=1` or `--gdn-fp16-chunk 1`. **Default:** OFF (profile `fast`
row `gdn_fp16_chunk`, pending qualification).

Replaces conv, recurrence and gated norm of the GDN prefill row route with identity `qwen-gdn-fp16-chunk rev 2`
([contract](engine/kernels/gdn-fp16-chunk.md)): 32-row chunks in WY form, FP16 WMMA operands with FP32 accumulation,
FP32 state, decays applied in FP32. Under it the tiled input projection writes its (already BF16-rounded) q/k/v/z
values as BF16 rows for the route.

- **Preconditions:** Prefill row route, head dimension 128, value heads a multiple of key heads (at most 4 per key head).
  Ignored under the baseline and for GSQ checkpoints; other shapes keep the exact route.
- **Fallback notice:** stderr `GDN FAST lever qwen-gdn-fp16-chunk rev 2: ON (...)` or `... shape not supported -> exact recurrence`.
- **Exactness evidence:** Bit-changing by design. Fixture `strix-hip-gdn-fp16-chunk-tests` screens against an FP64 token
  recurrence (relL2 ~4e-4) and checks determinism and aligned-split invariance; model fidelity is qualified separately.
- **Identity:** prefill option `gdn_fp16_chunk` (candidate, tag `gdnchunk16` in the prefill identity name and hash);
  excluded on automatic YaRN MTP and bundle routes like the other candidates; the GDN check modes run the exact route.
- **Memory:** persistent per stream, grow-only: about 211 MiB of chunk scratch plus 256 MiB of BF16 projection rows at
  T8192 (16 key / 48 value heads).
- **Off switch:** unset or `STRIX_QPF2K_GDN_FP16_CHUNK=0`; also `STRIX_PREFILL_EXACT_FP16=1` or
  `--prefill-disable gdn_fp16_chunk` (exact production route).
- **Measured gain:** plain CLI, same build, FAST vs exact: 4K 2.42 → 2.36 s (−2.6 %), 32K 19.61 → 18.81 s
  (−4.1 %), 262K 177.9 → 170.0 s (−4.5 %, one pair). Pending fidelity qualification.

<a id="qwen-hc"></a>

### HC folds, output v3 and band 16

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_HC=1; STRIX_QPF2K_HC_MASK=14; STRIX_QPF2K_HC_OUTPUT3=1; STRIX_QPF2K_HC_OUTPUT3_BAND=16; STRIX_QPF2K_HC_UPCOPY`. **Default:** auto.

Folds admitted inject/norm work and changes the lane-chain output schedule; band 16 permutes CTAs without changing sums.

- **Preconditions:** Eligible full-row HC shapes, supported norm/weight types and non-overlapping buffers; v3 requires HC enabled. Optional up-weight copy requires scratch.
- **Automatic selection:** The bundle supplies HC mask 14, output v3 and band 16; the up-weight copy is on with v3 unless disabled.
- **Fallback notice:** Unsupported folds keep original launches silently. If optional copy allocation fails or a site exceeds scratch, read the original weights; do not fail the call merely for that copy.
- **Exactness evidence:** STRIX_QPF2K_HC_CHECK=1 compares every replaced launch byte for byte and prints qpf2k HC check ... differ .... Require zero differences; band fixtures passed 519/519 and engine logits/tokens matched.
- **Off switch:** STRIX_QPF2K_HC=0; independently STRIX_QPF2K_HC_OUTPUT3=0, STRIX_QPF2K_HC_OUTPUT3_BAND=0 or STRIX_QPF2K_HC_UPCOPY=0
- **Measured gain:** HC v3/band: 32K -0.52 s (-2.2%); 4K -54 ms (-1.8%)
- **Public evidence:** [Qwen results](performance.md#qwen-prefill-and-defaults)

<a id="qwen-qsa"></a>

### QSA stages and stage batching

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_QSA=1; STRIX_QPF2K_QSA_MASK=0x2257`. **Default:** auto.

Uses admitted exact stage implementations and batches independent stages with bit 0x200.

- **Preconditions:** Stage batching requires mask bits 0x3 as well as 0x200; normal eager prefill, 1..256 complete 64-row groups with no partial tiles, grouped norm enabled, BF16 KV and pooled-key storage. Index geometry is four heads of dimension 128, 16-byte aligned queries and prefixes bounded by pool capacity and 65536 blocks. The norm tensors share a weight format, have power-of-two dimensions 2..256, and admit the rotary geometry (half rotary dimension <=64). Rows need contiguous positions and common raw/KV owners, geometry, norm owners, journal, compression ratio and top-k. Private device tables/maxima and pinned descriptors must allocate before any launch/state write. Windows, cohorts, graphs, int8 KV, explicit tiled diagnostics and self-check modes keep their existing paths.
- **Automatic selection:** The bundle supplies QSA=1 and mask 0x2257 (bit 13 = attend occupancy, mask 0x257 turns only that off); ineligible stage shapes keep their prior implementation.
- **Fallback notice:** Without tracing, stage-batch declines are silent and keep the per-group path. With STRIX_QPF2K_QSA_TRACE=1 and bit 0x200 requested, a decline prints qpf2k qsa stage_batch groups 0 pool 0 range 0 score 0 rope 0 norm 0 store 0 fallback 1. HIP execution errors propagate without retry after partial execution.
- **Exactness evidence:** The qualification campaign compared 1K code and 8K prose receipts, terminal FP32 logits FNV and generated tokens against the unchanged reference. Repeat with the same prefill/decode identity; changing arithmetic profiles is a separate quality comparison. Stage-batching qualification additionally compared terminal logits and 1K code/8K prose receipts. Attend occupancy (bit 13): its kernel copy is byte-compared with the production attend kernel in 460 fixture cases (single and multiple 64-row groups, causal/visible tails, padded union chunks); terminal logits FNV and tokens equal at 4K, 17.7K, 32K and 131K.
- **Off switch:** STRIX_QPF2K_QSA=0 for the stages; STRIX_QPF2K_QSA_MASK=0x2057 for stage batching alone; 0x257 for attend occupancy alone (it requires bit 6 and attend batching)
- **Measured gain:** Stage batching: 32K mean -0.39 s (-1.7%); published-build ABBA -0.17 s; 64K -1.6%; attend occupancy: 8192-row attend -8..-17 %, TTFT 32K 22.17 -> 22.03 s, 131K 95.40 -> 93.89 s (-1.5 %)
- **Public evidence:** [Qwen results](performance.md#qwen-prefill-and-defaults), [stage implementation](../src/hip_indexed_attention_launcher.hip)

<a id="qwen-moe"></a>

### Routed MoE operand image and route setup

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_MOE_DA16=1; STRIX_QPF2K_MOE_ROUTE=14`. **Default:** auto.

Produces the FP16 down operand image once; reduces CSR/prologue work and moves the clear into the ordered merge.

- **Preconditions:** Production Q4_K/Q5_K gate/up, supported down type and fast FP16 prefill tunes 80/82; DA16 uses the admitted gate/up output image.
- **Automatic selection:** Injected defaults are filtered per GSQ owner for all controlled readers, including DA16/ROUTE; explicit environment values are preserved. Explicitly forcing those combinations is rejected until separately qualified.
- **Fallback notice:** Defaulted GSQ decline has no separate warning. Explicit GSQ overrides throw STRIX_QPF2K_MOE_ROUTE / STRIX_QPF2K_MOE_DA16 are not qualified with the GSQ prefill profile.
- **Exactness evidence:** The qualification campaign compared 1K code and 8K prose receipts, terminal FP32 logits FNV and generated tokens against the unchanged reference. Repeat with the same prefill/decode identity; changing arithmetic profiles is a separate quality comparison. Routed-MoE known-row fixtures check operator equality; do not use diagnostic skip-work tune bits for exactness.
- **Off switch:** STRIX_QPF2K_MOE_DA16=0 or STRIX_QPF2K_MOE_ROUTE=0, separately
- **Measured gain:** -
- **Public evidence:** [Qwen results](performance.md#qwen-prefill-and-defaults)

<a id="qwen-router"></a>

### Router selection and activation staging

**Engine:** `qwen`. **Control:** `STRIX_PREFILL_ROUTER_SELECT; STRIX_ROUTER_STAGE_ACT`. **Default:** ON.

Runs one wave per row for selection and stages reusable activations.

- **Preconditions:** Admitted router-row layouts; staged activation form must fit the supported shape/LDS limits.
- **Automatic selection:** Selection is on unless =0. Activation staging defaults on (mode 2 at the scalar/window reader); unsupported cases use the older forms.
- **Fallback notice:** No ordinary decline line. STRIX_PREFILL_ROUTER_CHECK=1 is a synchronizing diagnostic for router logits, not an automatic fallback notice.
- **Exactness evidence:** The qualification campaign compared 1K code and 8K prose receipts, terminal FP32 logits FNV and generated tokens against the unchanged reference. Repeat with the same prefill/decode identity; changing arithmetic profiles is a separate quality comparison. The router diagnostic compares against logits_kernel<2>; selection still needs route IDs/weights and engine token checks.
- **Off switch:** STRIX_PREFILL_ROUTER_SELECT=0 or STRIX_ROUTER_STAGE_ACT=0, separately; these are outside the QPF2K master initializer
- **Measured gain:** -
- **Public evidence:** [Qwen results](performance.md#qwen-prefill-and-defaults), [router readers](../src/hip_moe_route_launcher.hip)

### Admitted decoder wrapper settings

Each setting below is injected only after the plain CLI route is admitted. Ordinary unsupported shapes keep the prior launcher without a general decline line; execution failures propagate. Each uses whole-engine token/logit digest gates within the selected identity. No isolated gain is attributed to these nine settings.

<a id="qwen-intprefill-dense-rows"></a>

### Dense row batching

**Engine:** `qwen`. **Control:** `STRIX_INTPREFILL_DENSE_ROWS`. **Default:** 512.

Batch supported Q8-weight dense projections into 512-row fast-FP16 prefill work.

- **Preconditions:** Fast-FP16 prefill; supported Q8 projection widths, uniform input/output strides and a complete 512-row block; otherwise keep the previous projection.
- **Automatic selection:** Unspecified value becomes 512 only after decoder admission; explicit values win.
- **Fallback notice:** Unsupported operator shapes retain the prior launcher without a general decline notice. Declined routes do not receive this wrapper default.
- **Exactness evidence:** Qualified profile-pair token IDs and terminal FP32 logits digests match; hold T, capacity and arithmetic fixed for operator comparisons.
- **Off switch:** STRIX_INTPREFILL_DENSE_ROWS=0
- **Measured gain:** No isolated measurement

<a id="qwen-prefill-pipe"></a>

### Prefill pipeline

**Engine:** `qwen`. **Control:** `STRIX_PREFILL_PIPE`. **Default:** 255.

Apply the measured 255 bitmask for overlap, row loops, headless work and persistent row views.

- **Preconditions:** Admitted automatic CLI prefill only; MTP priming bits are not injected into declined MTP routes.
- **Automatic selection:** Unspecified value becomes 255 only after decoder admission; explicit values win.
- **Fallback notice:** Unsupported operator shapes retain the prior launcher without a general decline notice. Declined routes do not receive this wrapper default.
- **Exactness evidence:** Qualified profile-pair token IDs and terminal FP32 logits digests match; hold T, capacity and arithmetic fixed for operator comparisons.
- **Off switch:** STRIX_PREFILL_PIPE=0
- **Measured gain:** No isolated measurement

<a id="qwen-qsa-attend-batch"></a>

### QSA attend batching

**Engine:** `qwen`. **Control:** `STRIX_QSA_ATTEND_BATCH`. **Default:** 1.

Batch admitted group union/attend work.

- **Preconditions:** Complete 64-row pooled-key/BF16 QSA groups, compatible KV owners/geometry, default four-row tiles and supported scratch; pinned view slots are reused only after completion events.
- **Automatic selection:** Unspecified value becomes 1 only after decoder admission; explicit values win.
- **Fallback notice:** Unsupported operator shapes retain the prior launcher without a general decline notice. Declined routes do not receive this wrapper default.
- **Exactness evidence:** Qualified profile-pair token IDs and terminal FP32 logits digests match; hold T, capacity and arithmetic fixed for operator comparisons.
- **Off switch:** STRIX_QSA_ATTEND_BATCH=0
- **Measured gain:** No isolated measurement

<a id="qwen-qsa-score-park"></a>

### QSA score parking

**Engine:** `qwen`. **Control:** `STRIX_QSA_SCORE_PARK`. **Default:** 1.

Retain reusable score work for the admitted schedule.

- **Preconditions:** Prefill groups (at least 16 rows), supported pooled score buffers/group ownership and 16-byte aligned index query rows; windows/cohorts retain their route.
- **Automatic selection:** Unspecified value becomes 1 only after decoder admission; explicit values win.
- **Fallback notice:** Unsupported operator shapes retain the prior launcher without a general decline notice. Declined routes do not receive this wrapper default.
- **Exactness evidence:** Qualified profile-pair token IDs and terminal FP32 logits digests match; hold T, capacity and arithmetic fixed for operator comparisons.
- **Off switch:** STRIX_QSA_SCORE_PARK=0
- **Measured gain:** No isolated measurement

<a id="qwen-qsa-norm-group"></a>

### QSA grouped norm

**Engine:** `qwen`. **Control:** `STRIX_QSA_NORM_GROUP`. **Default:** 1.

Group compatible QSA normalization rows.

- **Preconditions:** Compatible grouped norm/RoPE geometry, weight types and owners; use a rotary table plus grouped normalization instead of three per-group launches.
- **Automatic selection:** Unspecified value becomes 1 only after decoder admission; explicit values win.
- **Fallback notice:** Unsupported operator shapes retain the prior launcher without a general decline notice. Declined routes do not receive this wrapper default.
- **Exactness evidence:** Qualified profile-pair token IDs and terminal FP32 logits digests match; hold T, capacity and arithmetic fixed for operator comparisons.
- **Off switch:** STRIX_QSA_NORM_GROUP=0
- **Measured gain:** No isolated measurement

<a id="qwen-qsa-index-rows"></a>

### QSA index rows

**Engine:** `qwen`. **Control:** `STRIX_QSA_INDEX_ROWS`. **Default:** 1.

Batch compatible index projection rows.

- **Preconditions:** BF16 input matrices in supported QSA prefill groups; batched index q/k projections retain the same chains and reduction trees.
- **Automatic selection:** Unspecified value becomes 1 only after decoder admission; explicit values win.
- **Fallback notice:** Unsupported operator shapes retain the prior launcher without a general decline notice. Declined routes do not receive this wrapper default.
- **Exactness evidence:** Qualified profile-pair token IDs and terminal FP32 logits digests match; hold T, capacity and arithmetic fixed for operator comparisons.
- **Off switch:** STRIX_QSA_INDEX_ROWS=0
- **Measured gain:** No isolated measurement

<a id="qwen-lm-head-cohort-lean"></a>

### Lean cohort head

**Engine:** `qwen`. **Control:** `STRIX_LM_HEAD_COHORT_LEAN`. **Default:** 1.

For an explicitly active multi-session head, split compatible 9-32-row cohorts into eight-row pieces using the lean head. The admitted plain CLI receipt includes this setting but has no active cohort.

- **Preconditions:** Admitted decoder environment; this injection does not enable active multi-session serving heads.
- **Automatic selection:** Unspecified value becomes 1 only after decoder admission; explicit values win.
- **Fallback notice:** Unsupported operator shapes retain the prior launcher without a general decline notice. Declined routes do not receive this wrapper default.
- **Exactness evidence:** Qualified profile-pair token IDs and terminal FP32 logits digests match; hold T, capacity and arithmetic fixed for operator comparisons.
- **Off switch:** STRIX_LM_HEAD_COHORT_LEAN=0
- **Measured gain:** No isolated measurement

<a id="qwen-lm-head-rows-lean"></a>

### Lean row head

**Engine:** `qwen`. **Control:** `STRIX_LM_HEAD_ROWS_LEAN`. **Default:** 2.

Use two logits per wave with fused greedy partials.

- **Preconditions:** Supported row-head shapes and unchanged reduction operand pairs.
- **Automatic selection:** Unspecified value becomes 2 only after decoder admission; explicit values win.
- **Fallback notice:** Unsupported operator shapes retain the prior launcher without a general decline notice. Declined routes do not receive this wrapper default.
- **Exactness evidence:** Qualified profile-pair token IDs and terminal FP32 logits digests match; hold T, capacity and arithmetic fixed for operator comparisons.
- **Off switch:** STRIX_LM_HEAD_ROWS_LEAN=0
- **Measured gain:** No isolated measurement

<a id="qwen-dispatch-fuse"></a>

### Dispatch fusion

**Engine:** `qwen`. **Control:** `STRIX_DISPATCH_FUSE`. **Default:** 1.

Combine eligible verification/row dispatch work.

- **Preconditions:** Admitted single-session tiled decoder, or the single-session MTP route (the documented MTP command: the group 3-4 --mtp* options, the window kernel organization and known-prefix window options, and no other route, low-level or --mtp* option, golden dump, metadata-only diagnostic, sampling, explicit n-gram mode, session or state import, recording, final-state validation or diagnostic control; qualified target and checkpoint), supported fused steps; HTTP, cohort and serving-prefix routes are excluded from injection.
- **Automatic selection:** Unspecified value becomes 1 after decoder admission or on the single-session MTP route (the startup notice then reports `Qwen MTP route: STRIX_DISPATCH_FUSE=1 (automatic; ...)`); explicit values win.
- **Fallback notice:** Unsupported operator shapes retain the prior launcher without a general decline notice. Declined routes do not receive this wrapper default.
- **Exactness evidence:** Qualified profile-pair token IDs and terminal FP32 logits digests match; hold T, capacity and arithmetic fixed for operator comparisons.
- **Off switch:** STRIX_DISPATCH_FUSE=0
- **Measured gain:** MTP decode on the documented MTP command, default vs STRIX_DISPATCH_FUSE=0, same binary: code 1K, R2, R8 and prose 1K, R2, R8 decode time -0.3 to -3.2 % (B/A 0.968 to 0.997); tokens identical

<a id="qwen-gsq"></a>

### GSQ compatibility and typed dense-A8 rounding

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_DEFAULTS`. **Default:** ON.

Allow GSQ models to run while prefill defaults are enabled, filtering injected settings per model owner; typed dense-A8 sites retain their declared activation rounding and per-format scale/minimum accumulation order across R1, window and prefill sites.

- **Preconditions:** GSQ manifest and its separately selected numerical identity. Non-GSQ FP16 and PLE qualification does not qualify GSQ.
- **Automatic selection:** Detect GSQ gate tensors and suppress only injected defaults after each reader cache; no global environment mutation when owners alternate.
- **Fallback notice:** GSQ declines the new content-qualified auto route; normal serial/explicit GSQ execution continues. Forced incompatible controls retain validation errors.
- **Exactness evidence:** Host GSQ-first/ordinary-first reader tests check isolation; typed-site device gates check rounding. Host tests alone do not establish device equality.
- **Off switch:** STRIX_QPF2K_DEFAULTS=0 suppresses injection globally; explicit feature overrides retain GSQ admission checks.
- **Measured gain:** No isolated measurement
- **Implementation / evidence:** [Typed dense sites](../include/strix/hip_gsq_a8_groups.hpp), [GSQ tests](../tests/hip_gsq_typed_sites_tests.hip)

<a id="qwen-gdn-witness"></a>

### Opt-in GDN witness core

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_GDN_PARTS`. **Default:** opt-in (bit 128 off; normal mask 55).

Use scratch state and a NaN witness to guard a faster core; replay the pinned exact kernel if the witness requires it.

- **Preconditions:** Eligible lean GDN core, bit 128 set, bit 64 excluded, supported cooperative residency and scratch capacity.
- **Automatic selection:** Never selected by the defaults mask 55. Explicit PARTS bit 128 requests witness admission.
- **Fallback notice:** Failed scratch/shape admission retains the existing core; a set witness runs the pinned fallback from untouched state, otherwise scratch state is committed.
- **Exactness evidence:** Requires GDN byte/state checks and same-profile engine digests, including NaN cases; no defaults-on performance claim.
- **Off switch:** Clear bit 128; STRIX_QPF2K_GDN_PARTS=55 restores the normal parts mask
- **Measured gain:** No isolated measurement
- **Implementation / evidence:** [Witness launcher](../src/hip_gdn_launcher.hip)

<a id="qwen-qsa-score-rows"></a>

### QSA scorer row batching

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_SCORE_ROWS`. **Default:** auto (32 rows above 4096 blocks, 16 otherwise).

Stage-batch the QSA scorer over more rows, reusing the loaded keys.

- **Preconditions:** Automatic decoder route (CLI), admitted tiled prefill; MTP/HTTP serving and cohorts keep it opt-in; not applied under GSQ.
- **Automatic selection:** The decoder injects the value only when unset; explicit values win; `STRIX_QPF2K_DEFAULTS=0` or `--baseline` disables all five.
- **Fallback notice:** `=0` disables only this item and keeps the tiled route; shapes outside the admission keep the previous kernel.
- **Exactness evidence:** Bitwise: 32K terminal FP32 logits and 4K digests identical with the item on and off; golden-code and prose gates equal the archived base.
- **Off switch:** STRIX_QPF2K_SCORE_ROWS=0 (16 rows)
- **Measured gain:** ~ -1.6 s at 262K together with PLE row batching (~-10 ms at 32K); combined five-item set -0.71 s (-3.2 %) at 32K vs all five off (interleaved, same build)
- **Implementation / evidence:** [defaults](../include/strix/qpf2k_defaults.hpp), [engine reference](getting-started/engine-reference.md)

<a id="qwen-ple-batch"></a>

### PLE row batching

**Engine:** `qwen`. **Control:** `STRIX_PREFILL_PLE_BATCH`. **Default:** 1.

Run the per-layer-embedding projection in 64-row groups; the wide lane recipe, layout and scratch admission still apply.

- **Preconditions:** Automatic decoder route (CLI), admitted tiled prefill; MTP/HTTP serving and cohorts keep it opt-in; not applied under GSQ.
- **Automatic selection:** The decoder injects the value only when unset; explicit values win; `STRIX_QPF2K_DEFAULTS=0` or `--baseline` disables all five.
- **Fallback notice:** `=0` disables only this item and keeps the tiled route; shapes outside the admission keep the previous kernel.
- **Exactness evidence:** Bitwise: 32K terminal FP32 logits and 4K digests identical with the item on and off; golden-code and prose gates equal the archived base.
- **Off switch:** STRIX_PREFILL_PLE_BATCH=0
- **Measured gain:** Included in the scorer figure above (-0.16 s raw kernel time at 262K); combined five-item set -0.71 s (-3.2 %) at 32K vs all five off (interleaved, same build)
- **Implementation / evidence:** [defaults](../include/strix/qpf2k_defaults.hpp), [engine reference](getting-started/engine-reference.md)

<a id="qwen-shared-pre"></a>

### Shared-expert operand preimage

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_SHARED_PRE`. **Default:** 1.

Build the shared expert's FP16 operand once for gate/up and down; the existing tiled down admission and fallback apply.

- **Preconditions:** Automatic decoder route (CLI), admitted tiled prefill; MTP/HTTP serving and cohorts keep it opt-in; not applied under GSQ.
- **Automatic selection:** The decoder injects the value only when unset; explicit values win; `STRIX_QPF2K_DEFAULTS=0` or `--baseline` disables all five.
- **Fallback notice:** `=0` disables only this item and keeps the tiled route; shapes outside the admission keep the previous kernel.
- **Exactness evidence:** Bitwise: 32K terminal FP32 logits and 4K digests identical with the item on and off; golden-code and prose gates equal the archived base.
- **Off switch:** STRIX_QPF2K_SHARED_PRE=0
- **Measured gain:** ~-1 % TTFT at 32K; combined five-item set -0.71 s (-3.2 %) at 32K vs all five off (interleaved, same build)
- **Implementation / evidence:** [defaults](../include/strix/qpf2k_defaults.hpp), [engine reference](getting-started/engine-reference.md)

<a id="qwen-moe-merge-commit"></a>

### MoE ordered merge + BF16 commit

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_MOE_MC`. **Default:** 1 (mode 1; mode 2 needs an explicit value).

Merge expert outputs in a fixed order and commit BF16 rows in the same pass, behind the existing DA16/ROUTE=14 certificate.

- **Preconditions:** Automatic decoder route (CLI), admitted tiled prefill; MTP/HTTP serving and cohorts keep it opt-in; not applied under GSQ.
- **Automatic selection:** The decoder injects the value only when unset; explicit values win; `STRIX_QPF2K_DEFAULTS=0` or `--baseline` disables all five.
- **Fallback notice:** `=0` disables only this item and keeps the tiled route; shapes outside the admission keep the previous kernel.
- **Exactness evidence:** Bitwise: 32K terminal FP32 logits and 4K digests identical with the item on and off; golden-code and prose gates equal the archived base.
- **Off switch:** STRIX_QPF2K_MOE_MC=0
- **Measured gain:** ~-0.10 s at 32K; combined five-item set -0.71 s (-3.2 %) at 32K vs all five off (interleaved, same build)
- **Implementation / evidence:** [defaults](../include/strix/qpf2k_defaults.hpp), [engine reference](getting-started/engine-reference.md)

<a id="qwen-moe-header-reuse"></a>

### MoE gate/up header reuse

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_MOE_HDR`. **Default:** 1.

Reuse loaded superblock headers between gate and up within the DA16 tune 80/82 path.

- **Preconditions:** Automatic decoder route (CLI), admitted tiled prefill; MTP/HTTP serving and cohorts keep it opt-in; not applied under GSQ.
- **Automatic selection:** The decoder injects the value only when unset; explicit values win; `STRIX_QPF2K_DEFAULTS=0` or `--baseline` disables all five.
- **Fallback notice:** `=0` disables only this item and keeps the tiled route; shapes outside the admission keep the previous kernel.
- **Exactness evidence:** Bitwise: 32K terminal FP32 logits and 4K digests identical with the item on and off; golden-code and prose gates equal the archived base.
- **Off switch:** STRIX_QPF2K_MOE_HDR=0
- **Measured gain:** ~-0.07 s at 32K; combined five-item set -0.71 s (-3.2 %) at 32K vs all five off (interleaved, same build)
- **Implementation / evidence:** [defaults](../include/strix/qpf2k_defaults.hpp), [engine reference](getting-started/engine-reference.md)

<a id="qwen-moe-xor-rows"></a>

### MoE XOR LDS rows (64-assignment tune)

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_MOE_XOR64`. **Default:** 1.

When a chunk averages fewer than 96 assignments per expert (fewer than 4,915 rows for the shipped 512-expert, top-10 checkpoint) the routed experts use 64-assignment batches (tune 80) whose padded LDS stage (46 KB) fits two CTAs per WGP. With the item on, such chunks of at least 3,072 rows stage the same operands in XOR-permuted rows without padding (40 KB, three CTAs per WGP); operands, WMMAs and the K order of every output are unchanged.

- **Preconditions:** Automatic decoder route (CLI), admitted tiled prefill, DA16 chain, automatic MoE tune (no `STRIX_INTPREFILL_MOE_TUNE`), 64-assignment batches with at least 3,072 rows (3,072 to 4,914 rows for 512 experts, top-10); MTP/HTTP serving and cohorts keep it opt-in; not applied under GSQ.
- **Automatic selection:** The decoder injects the value only when unset; explicit values win; `STRIX_QPF2K_DEFAULTS=0` or `--baseline` disables it.
- **Fallback notice:** `=0` disables only this item and keeps the tiled route; below 3,072 rows (where the third CTA does not pay) and with 128-assignment batches the previous kernels run.
- **Exactness evidence:** Bitwise: routed-expert fixture (every written view, the CSR words and the certificate word, HDR on and off, certificate off, armed and unproven, non-finite inputs) identical with the item on and off on synthetic and real layers; 4K/32K terminal FP32 digests and plain-command tokens identical.
- **Off switch:** STRIX_QPF2K_MOE_XOR64=0
- **Measured gain:** routed call −2.6 % to −8.2 % at 3,072 to 4,864 rows (512 experts, top-10, real layers, ABBA); +1 % to +5 % at 1,024 to 2,048 rows (hence the threshold); no change at 8,192-row chunks.
- **Implementation / evidence:** [defaults](../include/strix/qpf2k_defaults.hpp), [engine reference](getting-started/engine-reference.md)

<a id="qwen-qsa-union-merge"></a>

### QSA attend-union merge

**Engine:** `qwen`. **Control:** `STRIX_QSA_UNION_MERGE`. **Default:** 1 (from 49,152 blocks; range form below).

Build the attend union by merging per-row selections instead of the range form; same (block, mask) pairs, order and token counts.

- **Preconditions:** Qwen QSA prefill from 49,152 blocks (about 196K tokens); below that the range form runs.
- **Automatic selection:** On when unset; explicit values win; `--baseline`, `STRIX_QPF2K_DEFAULTS=0` and the GSQ profile turn these long-context items off; `--print-effective-config` reports the requested value under `environment` and the resolved state under `long_context_items` (`enabled`, plus `suppression`: `baseline`, `gsq`, `STRIX_QPF2K_DEFAULTS=0`, `explicit-off`, or the excluded decode route such as `mtp`).
- **Fallback notice:** Below its threshold or outside the plain route, the previous exact form runs.
- **Exactness evidence:** Exact: union, score and decode fixtures byte-identical (score rows R16/R32/R64 to 262,144 blocks); at 262,143 tokens the 456-call QSA trace with these defaults is byte-identical to all levers off and the token matches; tokens identical at 32K/131K/262,080.
- **Off switch:** STRIX_QSA_UNION_MERGE=0
- **Measured gain:** union kernel -46 % at 75K blocks, -73 % at 262K blocks
- **Implementation / evidence:** [QSA launcher](../src/hip_indexed_attention_launcher.hip)

<a id="qwen-qsa-decode-pool"></a>

### QSA persistent pooled decode keys

**Engine:** `qwen`. **Control:** `STRIX_QSA_DECODE_POOL`. **Default:** 1 (plain append-only serial decode).

Keep pooled keys across decode steps instead of rebuilding them. Serial decode stores only
BF16 upper halves (`uint16_t`) by default; the pool kernel already rounds each key to BF16.
The scorer widens with `uint32_t(key) << 16` before the same FP32 products, 128-feature
tree, ordered ReLU head sum and division. No FP32 key copy is retained. Prefill pools keep
their existing storage behavior. `STRIX_QSA_DECODE_POOL_CHECK=1` re-derives scores from raw
keys with `block_score_kernel` and compares every score bitwise in either format.

`STRIX_QSA_DECODE_POOL_BF16=0` restores FP32 pool storage. The BF16 policy is off under
`--baseline`, GSQ and `STRIX_QPF2K_DEFAULTS=0`; those policies also disable the serial pool
route. `STRIX_QSA_DECODE_POOL=0` disables that route independently. Requested format
settings appear in `--print-effective-config`.

The storage fixture `strix-hip-qsa-decode-pool-bf16-tests` shares the production scorer and
checks BF16 against FP32 at counts spanning 1..262144 blocks, including wave and grid
boundaries. Its TIME section reports interleaved median milliseconds for both formats at
65536 / 131072 / 262144 blocks (262144: 0.777 ms FP32, 0.447 ms BF16). On the full model at 131,072 tokens
`STRIX_QSA_DECODE_POOL_CHECK=1` compared 11,011,140 scores with 0 differences in both formats, with the same
selections and tokens; with explicit YaRN at 32,768 and 131,072 tokens also 0 differences. The 262K decode step
is unchanged (51.38 vs 51.34 ms/step, same tokens): BF16 halves the pool's key memory, not the step time. The
evidence below was measured with FP32 pools.

- **Preconditions:** Plain append-only serial decode on the CLI route (not MTP, HTTP, prompt lookup, prefix round trip, the prefix state cache, serving prefixes or graphs). The pool's device memory is charged in admission and reported as `qsa_decode_pool_bytes`, with `qsa_decode_pool_key_bytes` (2 or 4). For B = floor(C / 4), the bytes are 12 x B x 128 x key_bytes + max(ceil(B / 4), 16384) x 4. Default BF16 totals at C = 262144 / 524288 / 1048576 are 201392128 / 402784256 / 805568512 bytes, including the maxima row.
- **Automatic selection:** On when unset; explicit values win; `--baseline`, `STRIX_QPF2K_DEFAULTS=0` and the GSQ profile turn these long-context items off; `--print-effective-config` reports the requested value under `environment` and the resolved state under `long_context_items` (`enabled`, plus `suppression`: `baseline`, `gsq`, `STRIX_QPF2K_DEFAULTS=0`, `explicit-off`, or the excluded decode route such as `mtp`).
- **Fallback notice:** Below its threshold or outside the plain route, the previous exact form runs.
- **Exactness evidence:** Exact: union, score and decode fixtures byte-identical (score rows R16/R32/R64 to 262,144 blocks); at 262,143 tokens the 456-call QSA trace with these defaults is byte-identical to all levers off and the token matches; tokens identical at 32K/131K/262,080.
- **Off switch:** STRIX_QSA_DECODE_POOL=0
- **Measured gain:** plain serial CLI decode at 262K 75.7 to 53.0 ms/step (-30 %), at 131K -14.7 % (measured with decode selection also on; the separate MTP committed key service is measured [below](#qwen-qsa-mtp-pool))
- **Implementation / evidence:** [QSA launcher](../src/hip_indexed_attention_launcher.hip)

<a id="qwen-plain-decode-step"></a>

### Plain decode step: grouped QSA attention, fused selection maxima, two-launch MoE glue

**Engine:** `qwen`. **Controls:** `STRIX_QSA_DECODE_ATTN_V2`, `STRIX_QSA_DECODE_VALUE_V2`, `STRIX_QSA_DECODE_FUSED_MAXIMA`, `STRIX_MOE_R1_GLUE_FUSE`. **Default:** 1 each.

Four exact changes to the one-row (plain, non-MTP) decode step, each with its own switch:

- `STRIX_QSA_DECODE_ATTN_V2`: the QSA attention score runs the draft step's grouped form (four query heads of one KV head
  share each K row, one wave per 16 slots) instead of one 256-thread workgroup per (head, slot). Every product and
  addition is operand-pinned to `attention_score_kernel`'s tree.
- `STRIX_QSA_DECODE_VALUE_V2`: after the same softmax, value/gate runs the draft step's two-launch value kernel (four
  query heads per value load, eight feature slices) instead of one workgroup per head; the same ascending-slot fma
  chain per (head, feature) and the same BF16 epilogue.
- `STRIX_QSA_DECODE_FUSED_MAXIMA`: the pooled indexer scorer also writes the 4-block group maxima the selection reads
  (the same `fmaxf` expression over the stored scores), so the separate maxima launch is gone.
- `STRIX_MOE_R1_GLUE_FUSE`: the routed experts step with A8 down runs its glue in two launches instead of nine (one CTA
  clears the accumulation, checks routes and weights and quantizes; one CTA merges, scans for non-finite values,
  commits and publishes the first-error record), with the same per-element work and status semantics.

- **Preconditions:** Eager plain one-row decode (CLI serial route, either rope route); BF16 KV, head_dim 256, a multiple of four query heads per KV head; no device graph, graph position or status word. The fused maxima need the serial decode pool's own scores with the maxima selection (`STRIX_QSA_DECODE_SELECT`) and are declined under `STRIX_QSA_DECODE_POOL_CHECK=1`. The MoE glue needs the A8 down profile with deferred MoE status. MTP windows, prefill tiles and cohorts are untouched.
- **Automatic selection:** On when unset; `=0`, `--baseline`, `STRIX_BASELINE=1` and `STRIX_QPF2K_DEFAULTS=0` turn each off; GSQ checkpoints keep the three QSA items off. All four are allowed on the explicit YaRN route.
- **Fallback notice:** Unadmitted shapes and routes run the previous kernels without a notice.
- **Exactness evidence:** Byte-identical by construction and fixture: `strix-hip-qsa-window-attn-v2-tests` (score, probabilities and value/gate output vs `attention_score_kernel` + `attention_softmax_kernel` + `attention_output_kernel`, NaN keys and queries), `strix-hip-qsa-decode-pool-bf16-tests` (scores and group maxima vs the scorer plus `row_maxima_kernel`, 1..262,144 blocks, non-finite keys, canaries), `strix-hip-q4k-a8-probe-tests` (serial prologue vs clear + route checks + quantizer for every error class) and `strix-hip-moe-r1-glue-tail-tests` (tail vs merge, finite scan, commit and publication: sticky status, +-inf, NaN payloads, BF16 overflow, invalid route, preset record). Full model: a 2,304-token prompt (576 complete blocks, above the 512-block selection) and a 32,768-token prompt (64 steps) give byte-identical full FP32 logits for every decode step ON vs OFF vs ON, tokens identical at 1K and 32K.
- **Off switch:** STRIX_QSA_DECODE_ATTN_V2=0, STRIX_QSA_DECODE_VALUE_V2=0, STRIX_QSA_DECODE_FUSED_MAXIMA=0, STRIX_MOE_R1_GLUE_FUSE=0
- **Measured gain:** plain decode, prose prompt, 256 tokens, ON vs all four switches off, interleaved ABAB on one node: 1K 43.95 -> 41.59 ms/step (-5.4 %), 32K 46.75 -> 43.92 ms/step (-6.1 %), 262K (explicit YaRN factor 2, default prefill, 128 tokens) 49.46 -> 45.91 ms/step (-7.2 %, 20.2 -> 21.8 tok/s); tokens identical in every pair
- **Implementation / evidence:** [QSA launcher](../src/hip_indexed_attention_launcher.hip), [pooled scorer](../src/hip_qsa_pooled_score.inc), [MoE experts launcher](../src/hip_moe_experts_launcher.hip), [A8 probe](../src/hip_moe_q4k_a8_probe.hip)

<a id="qwen-qsa-mtp-pool"></a>

### QSA MTP committed BF16 key service

The separate [MTP committed key service](engine/qsa-committed-pool.md) uses
`STRIX_QSA_MTP_POOL` and one thirteen-entry owner per session. The
[route table](engine/qsa-committed-pool.md) enables it by default on the native MTP CLI,
on the single-request HTTP runner, and for YaRN MTP CLI factor 2 through 524288 and factor 4 through
1048576 (with rolling PLE); both `=0` off switches remain authoritative.

- **Preconditions:** Single HTTP runner: `--http-batch 0`, a configuration the batch runner cannot take, or an AUTO/capped batch whose measured capacity is one slot. In that last case the fallback keeps the service only when the slot still fits with it charged. Includes the in-memory prefix cache. Batch slots, nested scheduling, sessions, disk prefix restore, state import, serving prefixes and YaRN HTTP keep the opt-in `STRIX_QSA_MTP_POOL=1`. Product profiles do not change the decision.
- **Exactness evidence:** Bitwise per request, same binary, default vs `STRIX_QSA_MTP_POOL=0`: token IDs, prompt state digest, committed target state and the full-vocabulary logits of every committed step. `STRIX_QSA_MTP_POOL_CHECK=1` compared all 13 roles with 0 differences. The timing-based window controller may pick a different R2/R4/R8 owner on the faster cycle, which changes speculative rows and predictor KV but not a committed token. Details in [the route document](engine/qsa-committed-pool.md#single-request-http-runner).
- **Measured gain:** single HTTP runner, 127.6K-token prompt (131K context), dense-A8 T=8192, R2/R4/R8 prefix windows, in-memory prefix cache, one Strix Halo node, ABBA on one binary: decode 39.5 to 52.0 tok/s on the cold request (+32 %) and 41.6 to 55.8 tok/s on restored follow-up turns (+34 %); prefill unchanged (176-183 s cold, about 7 s restored); tokens identical.
- **Off switch:** `STRIX_QSA_MTP_POOL=0`

<a id="qwen-qsa-decode-select"></a>

### QSA decode selection from 4-block maxima

**Engine:** `qwen`. **Control:** `STRIX_QSA_DECODE_SELECT`. **Default:** 1.

Select decode blocks from precomputed 4-block maxima; same selected blocks.

- **Preconditions:** Plain append-only serial decode on the CLI route (not MTP, HTTP, prompt lookup, prefix round trip, the prefix state cache, serving prefixes or graphs).
- **Automatic selection:** On when unset; explicit values win; `--baseline`, `STRIX_QPF2K_DEFAULTS=0` and the GSQ profile turn these long-context items off; `--print-effective-config` reports the requested value under `environment` and the resolved state under `long_context_items` (`enabled`, plus `suppression`: `baseline`, `gsq`, `STRIX_QPF2K_DEFAULTS=0`, `explicit-off`, or the excluded decode route such as `mtp`).
- **Fallback notice:** Below its threshold or outside the plain route, the previous exact form runs.
- **Exactness evidence:** Exact: union, score and decode fixtures byte-identical (score rows R16/R32/R64 to 262,144 blocks); at 262,143 tokens the 456-call QSA trace with these defaults is byte-identical to all levers off and the token matches; tokens identical at 32K/131K/262,080.
- **Off switch:** STRIX_QSA_DECODE_SELECT=0
- **Measured gain:** Not separately measured (no change at 131K: 50.69 vs 50.87 ms/step); an exact reorganisation for depth
- **Implementation / evidence:** [QSA launcher](../src/hip_indexed_attention_launcher.hip)

<a id="qwen-prefill-moe-merge-vec"></a>

### Vector MoE merge+commit in tiled prefill

**Engine:** `qwen`. **Control:** `STRIX_PREFILL_MOE_MERGE_VEC`. **Default:** 1.

The certified routed-MoE merge+commit handles four adjacent outputs per thread (vector raw-dot loads, packed BF16 stores); each output keeps its own guards and its slot-ordered FP32 FMA chain.

- **Preconditions:** Tiled prefill with the certified merge+commit tail (STRIX_QPF2K_MOE_MC); hidden size divisible by 4 and 16-byte raw-dot/accumulation rows, 8-byte output rows; otherwise the scalar merge runs.
- **Automatic selection:** On when unset; explicit values win; `--baseline`, `STRIX_QPF2K_DEFAULTS=0` and the GSQ profile turn it off; `--print-effective-config` reports it under `long_context_items`.
- **Fallback notice:** Unaligned views or widths run the scalar merge+commit.
- **Exactness evidence:** Exact: merge+commit fixture (synthetic, crafted, real layers; NaN/Inf/large-dot, bad-route, weight-above-one and H-1 injections) 1712 comparisons byte-identical to the scalar merge incl. the accumulation view; plain default command 4K/32K: 32 generated tokens identical on/off.
- **Off switch:** STRIX_PREFILL_MOE_MERGE_VEC=0
- **Measured gain:** -0.19 ms per T8192 admitted merge call (3.81 vs 3.99 ms, real layers); about -0.02 s at 32K
- **Implementation / evidence:** [MoE merge+commit](../src/hip_moe_prefill_commit.inc)

<a id="qwen-qsa-cache-only"></a>

### Last-layer cache-only QSA step on headless prefill blocks

**Engine:** `qwen`. **Control:** `STRIX_QSA_CACHE_ONLY`. **Default:** 1.

A headless tiled block already stops after its last state-writing step, the last layer's QSA step. That step now writes only what the session keeps: its input projections run unchanged, then only the k norm/rotation and the raw index-key / KV append (BF16 or int8, with the journal); pooled keys, block scores, selection, union, attention and the output projection are skipped because nothing reads them.

- **Preconditions:** Tiled prefill blocks without a head and without a step observer whose last state-writing step is a QSA layer (STRIX_PREFILL_PIPE bit 64 cut active); the MTP route (step observer) and blocks with a head run the full step.
- **Automatic selection:** On when unset; `=0`, `--baseline`, `STRIX_QPF2K_DEFAULTS=0` and the GSQ profile run the full step; `--print-effective-config` reports it under `long_context_items`. Persistent state is unchanged, so the value stays out of the prefix/session cache namespace (as the other exact startup settings).
- **Fallback notice:** Any other block, or a QSA step outside the full-T rows entry, runs the full step.
- **Exactness evidence:** Exact: plain default command, same binary on/off: full logits of the prefill and of 32 decode steps byte-identical with tokens at 4K (not admitted, one block with a head) and 32K (T8192, three headless blocks); 262K YaRN serial route: 32 generated tokens identical in four alternating runs (no logits/state digest on that route).
- **Off switch:** STRIX_QSA_CACHE_ONLY=0
- **Measured gain:** 262K YaRN serial prefill, 2 runs per arm alternating: 191.04 / 193.21 s on vs 194.76 / 200.17 s off (means 192.1 vs 197.5 s; adjacent pairs -1.6 to -7.0 s); 32K: neutral on the current T16384 route (one headless block; +0.04 / +0.07 s mean in two runs per order), -0.41 s on the earlier T8192 route (three headless blocks)
- **Implementation / evidence:** [QSA launcher](../src/hip_indexed_attention_launcher.hip), [prefill runtime](../src/known_prefix_prefill_runtime.cpp)

<a id="qwen-qsa-score-rows64"></a>

### QSA scorer R64 above 65,536 blocks

**Engine:** `qwen`. **Control:** `STRIX_QSA_SCORE_ROWS64`. **Default:** 1 (from 65,536 blocks; per-group 64-row scoring follows the policy).

Use 64-row scorer tiles at very long contexts (previously pinned to 16 rows above 262K).

- **Preconditions:** Scorer from 65,536 blocks, i.e. beyond the native 262,144-token context: no effect on native runs today.
- **Automatic selection:** On when unset; explicit values win; `--baseline`, `STRIX_QPF2K_DEFAULTS=0` and the GSQ profile turn these long-context items off; `--print-effective-config` reports the requested value under `environment` and the resolved state under `long_context_items` (`enabled`, plus `suppression`: `baseline`, `gsq`, `STRIX_QPF2K_DEFAULTS=0`, `explicit-off`, or the excluded decode route such as `mtp`).
- **Fallback notice:** Below its threshold or outside the plain route, the previous exact form runs.
- **Exactness evidence:** Exact: union, score and decode fixtures byte-identical (score rows R16/R32/R64 to 262,144 blocks); at 262,143 tokens the 456-call QSA trace with these defaults is byte-identical to all levers off and the token matches; tokens identical at 32K/131K/262,080. R64 evidence is fixture-only: no native shape reaches 65,536 blocks.
- **Off switch:** STRIX_QSA_SCORE_ROWS64=0
- **Measured gain:** scorer at 1M tokens R16 -> R64 -42 %
- **Implementation / evidence:** [QSA launcher](../src/hip_indexed_attention_launcher.hip)

`STRIX_QSA_POOL_PACKED=1` (packed BF16 pooled keys) is included as an exact opt-in; it is not a default.

<a id="qwen-qsa-window-grouped-score"></a>

### QSA grouped window attention score

**Engine:** `qwen`. **Control:** `STRIX_QSA_WINDOW_ATTN_V2`. **Default:** 1.

Single-session MTP verify windows (2/4/8 rows) score QSA attention with one wave per group of 12 query heads that
share a KV head, so each key row is loaded once per group instead of once per head (8 slots per wave at 2 rows, 16
at 4-8 rows). The MTP draft step scores with four heads per wave in place of its one-block score kernel. Softmax and
value/gate keep their kernels; prefill tiles and multi-session cohort windows are untouched.

- **Preconditions:** BF16 KV cache, head_dim 256, a multiple of 12 query heads per KV head for windows (of 4 for the draft step); the draft step only where it runs the one-block score kernel (`STRIX_QPERSIST_DRAFT_SCORE` unset). Int8 KV, other shapes, prefill and cohort windows keep the previous score kernels.
- **Automatic selection:** On when unset. Off with `STRIX_QSA_WINDOW_ATTN_V2=0` (also `off` / `false`), `--baseline` / `STRIX_BASELINE=1` and `STRIX_QPF2K_DEFAULTS=0`. An explicit `STRIX_QSA_WINDOW_GROUPED=1` or `STRIX_QSA_SCORE_SLOTS` other than 4 keeps the explicitly selected window score kernel.
- **Fallback notice:** Unadmitted shapes run the previous kernels without a notice.
- **Exactness evidence:** Exact by construction: per (row, head, slot) the same products, the same pairs and the same reduction tree, and every product and addition is operand-pinned (inline `v_mul_f32` / `v_add_f32`) to the order of the kernel it replaces, so results are byte-identical including NaN payloads. `strix-hip-qsa-window-attn-v2-tests` compares raw scores, probabilities, outputs and A8 codes/deltas byte for byte against the replaced production kernels (counts 1..2051, contiguous and sparse selections, non-finite inputs including NaN keys and queries with distinct payloads) and checks the routing. MTP decode A/B, 2 rounds: complete token IDs identical, 34 exactness checks pass.
- **Off switch:** STRIX_QSA_WINDOW_ATTN_V2=0
- **Measured gain:** MTP decode with 1K-token prompts +3.0 to +5.7 % (code prompt 59.25 -> 61.98 tok/s, +4.6 %, windows of 2 rows +3.1 %, of 8 rows +4.7 %; prose prompt +4.2 %, 2 rows +3.0 %, 8 rows +5.7 %). Score kernel at 2,051 selected tokens: 170 / 327 / 640 us -> about 19 / 30 / 55 us per call at 2 / 4 / 8 rows; draft score 326 -> 17 us.
- **Implementation / evidence:** [QSA launcher](../src/hip_indexed_attention_launcher.hip), [fixture](../tests/hip_qsa_window_attn_v2_tests.hip)

### Exact options that remain opt-in

MoE PK/MERGE, QSA_OUT16, attend pairing and `STRIX_QPF2K_HC_LOWBAND` are exact experiments with neutral or unmeasured complete-workload
gains. They are not part of the default set. Do not infer an environment spelling from these experiment names;
use only controls exposed by the current source and require the corresponding device gates before promotion.

<a id="qwen-serve-api"></a>

### HTTP API surface: Responses route, capabilities, stream receipts, keepalive

**Engine:** `qwen` (also `strix-deepseek41-serve`, same shared server layer). **Control:** `--http-responses`,
`--http-capabilities`, `--http-stream-receipts`, `--http-sse-keepalive`. **Default:** on, on, on, 10 seconds.

Serving features that change no generated token: a stateless `POST /v1/responses` route translated onto the chat
prompt path, a versioned `GET /v1/capabilities` document, the applied policy (requested vs effective sampling and
output limit, each sampling filter with its source, the effective seed of a sampled request) at the start and end of
every stream, and SSE keepalive comments while a stream waits. Sampling filters resolve request -> server default
(`--http-default-temperature`, `--http-default-top-k`, `--http-default-top-p`, `--http-default-min-p`; a positive
default temperature is a token-changing operator choice and needs `--http-sampling sample`) -> built-in greedy; an
unseeded sampled request gets a seed, echoed in the receipt (stream receipts on), so the same request with that seed
reproduces it.

- **Preconditions:** `--http-serve` (Qwen) or the DeepSeek serve app.
- **Automatic selection:** Always on in the server; a request disables its keepalive with `"strix":{"keepalive":false}`.
- **Fallback notice:** Unsupported Responses fields are refused with a 400 naming the field; nothing is ignored silently.
- **Exactness evidence:** Responses prompt IDs equal the equivalent chat request (CPU tests); greedy token IDs equal with
  the features on and off, and the stream bytes with every switch off equal the previous build after normalizing ids
  and timings (GPU gate).
- **Off switch:** `--http-responses off`, `--http-capabilities off`, `--http-stream-receipts off`, `--http-sse-keepalive 0`

<a id="qwen-serve-constraints"></a>

### Constrained JSON output

**Engine:** `qwen`. **Control:** `--http-constraints`. **Default:** on (requests opt in with `response_format` /
`text.format`).

Token selection restricted to a declared JSON language (`json-object` or a `strix-json-schema-v1` schema), greedy, on a
resolved `mtp-host` / `serial-host` route for the constrained request only.

Host selection defaults to admitted serial-equivalent MTP windows (`mtp-host`), with
`serial-host` fallback and per-request route/reason receipts. Argmax-first selection
(`--http-selection-fast`, default on) skips the mask walk, and on `mtp-host` the logits readback, for rows whose
unmasked argmax the constraint admits (same token IDs). No speed claim;
[route admission and exactness](engine/serving/serving.md#policy-selection-routes).

- **Preconditions:** `--http-serve`; a request with `response_format` / `text.format` `json_object` or `json_schema`,
  greedy, no stop strings, no tools.
- **Automatic selection:** Only requests that ask for it; other requests and other batch slots are unchanged.
- **Fallback notice:** Unsupported schema keywords, sampling, stop strings or tools are refused with a 400; output is
  never produced unconstrained in their place.
- **Exactness evidence:** Constrained output is exact relative to its declared constrained target (masked greedy), not
  to unconstrained greedy. Unconstrained requests, alone and sharing the batch with constrained ones, keep their token
  IDs (GPU gate).
- **Off switch:** `--http-constraints off`; `--http-selection-mtp off` keeps policy selection serial; `--http-selection-fast off` walks the full mask for every row

<a id="qwen-serve-progress"></a>

### SSE prefill progress

**Engine:** `qwen` (switch also on `strix-deepseek41-serve`). **Control:** `--http-sse-progress-interval`. **Default:** 1 second.

Streamed requests receive `: progress {"phase","prefilled_tokens","cached_tokens","total_tokens"}` SSE comments while
they prefill, from a host snapshot the engine updates after each completed prefill unit (no added device synchronization).

- **Preconditions:** a streamed request on an engine that reports progress (Qwen).
- **Automatic selection:** On; a request opts out with `"strix":{"progress":false}`.
- **Fallback notice:** Engines without reports send no progress comments (keepalive comments still apply).
- **Exactness evidence:** Comments are outside the data events: full token IDs and data events are identical with
  progress on and off (GPU gate).
- **Off switch:** `--http-sse-progress-interval 0`

<a id="qwen-serve-thinking-budget"></a>

### Thinking budget policy

**Engine:** `qwen`. **Control:** `--http-thinking-budget`, `--http-anthropic-thinking-budget`. **Default:** available
(requests opt in); Messages `budget_tokens` noted, not enforced.

`thinking-budget-v1` closes reasoning no later than the request's budget by feeding the end-of-thinking token and the
template separator through target execution, then lets the answer run (greedy or JSON-constrained).

Host selection defaults to admitted serial-equivalent MTP windows (`mtp-host`), with
`serial-host` fallback and per-request route/reason receipts. No MTP speed claim;
[route admission and exactness](engine/serving/serving.md#policy-selection-routes).

- **Preconditions:** `--http-serve`; a request with `strix.thinking_policy` (or Messages `budget_tokens` with
  `--http-anthropic-thinking-budget enforce`), thinking on, greedy, no stop strings, no tools.
- **Automatic selection:** Only requests that ask for it; they use the resolved `mtp-host` / `serial-host` route, other slots unchanged.
- **Fallback notice:** Impossible budgets and unsupported combinations are refused with 400.
- **Exactness evidence:** Changes tokens by design (declared policy, reported as `closure: natural|forced`). A budget that
  is never reached yields the same token IDs as ordinary speculative greedy decoding (GPU gate).
- **Off switch:** `--http-selection-mtp off` keeps policy selection serial; `--http-thinking-budget off`; Messages budgets are enforced only with `--http-anthropic-thinking-budget enforce`

<a id="qwen-serve-logprobs"></a>

### Logprobs and logit transforms

**Engine:** `qwen`. **Control:** `--http-logprobs`, `--http-logit-transforms`. **Default:** available (requests opt in).

`logprobs` / `top_logprobs` report the log-softmax of the exact row each token was selected from. `logit-transform-v1`
applies repetition (on raw logits), presence and frequency penalties and `logit_bias` before greedy selection.

Host selection defaults to admitted serial-equivalent MTP windows (`mtp-host`), with
`serial-host` fallback and per-request route/reason receipts. No MTP speed claim;
[route admission and exactness](engine/serving/serving.md#policy-selection-routes).

- **Preconditions:** `--http-serve`; greedy requests (sampling with these is refused for now).
- **Automatic selection:** Only requests that ask for them (non-neutral values); they use the resolved `mtp-host` / `serial-host` route.
- **Fallback notice:** Neutral values (zero penalties, empty bias, `logprobs` false) keep the ordinary route.
- **Exactness evidence:** Logprobs never change the selected tokens: IDs equal ordinary speculative greedy decoding
  (GPU gate). Transforms change tokens by design and are labelled `greedy-logit-transform-v1`.
- **Off switch:** `--http-logprobs off`, `--http-logit-transforms off`; `--http-selection-mtp off` keeps policy selection serial

<a id="qwen-rtx-tail-large"></a>

### RTX prefill tail-large block schedule

**Engine:** `qwen`. **Control:** `--block-schedule default|tail-large`, `STRIX_QWEN_TAIL_LARGE=0`. **Default:** `tail-large`.

The CUDA prefill service moves a short final remainder block to the front so the last block, whose transfer
cannot overlap later compute, is a large one. Rows stay chronological; only block sizes move.

- **Preconditions:** CUDA Qwen prefill service on the RTX PRO with node import; prompts whose final block is smaller than its predecessor.
- **Exactness evidence:** Every exported state component is byte-identical between the original order and tail-large at 1100, 8191 and 32767 rows (with and without MTP, release-by-layer on and off, plus an explicit final-block reservation); greedy IDs equal node-only prefill.
- **Off switch:** `--block-schedule default` (or `"block_schedule":"default"` per request); `STRIX_QWEN_TAIL_LARGE=0` for the gate harness.
- **Measured gain:** TTFT -12.7 % at 1K, -7.9 % at 8K, -5.0 % at 32K against the original order.
- **Implementation / evidence:** [CUDA Qwen prefill scheduling](engine/serving/qwen-cuda-prefill.md)

<a id="qwen-yarn-mtp"></a>

### YaRN long-context MTP CLI

**Engine:** `qwen`. **Control:** `--rope-yarn 2|4 --context C` with a verified MTP bundle (or `--mtp`), `--mtp-bundle FILE`, `STRIX_QWEN_YARN_MTP=0`, `STRIX_QSA_MTP_POOL`, `STRIX_PLE_ROLLING`. **Default:** table-backed QSA MTP pool and rolling PLE for factor 2 through 524,288 and factor 4 through 1,048,576 positions when admitted.

Speculative MTP decode on the static YaRN CLI route, with the committed QSA MTP pool and rolling PLE reads so that a
1,048,576-position context fits one node. YaRN itself stays opt-in because it changes outputs against native rotary.

- **Preconditions:** CLI only (HTTP refuses YaRN); explicit `--context`; pinned sidecar and version-3 golden; memory admission with a 4 GiB physical margin. The [storage admission ladder](engine/speculative/mtp-native-cli.md#explicit-yarn-mtp-cli) tries defaults, H1024/cap268435456/ring256/PIPE-default, then H256/cap33554432/ring64/PIPE0. Only memory refusal advances it; explicit environment values pin each knob. `storage_ladder_step` and chosen sizes appear in inspection and run receipts.
- **Exactness evidence:** Against serial YaRN at 1,048,576 positions (factor 4, real prose prompt): all 256 generated tokens, every committed step's full-vocabulary logits, prompt and final target state and predictor state are identical between MTP with the pool, plain serial YaRN and MTP with the pool off. 262K (full scope) and 524K (target scope) are identical as well.
- **Automatic selection:** a bare `--rope-yarn` command selects MTP when `<model>.mtp-bundle.json` (or `--mtp-bundle FILE`) verifies against its pins; the calibration golden is read from the bundle directory or generated once into the user cache (`$XDG_CACHE_HOME/strix-engine/goldens`); any bundle failure before mutation falls back to serial YaRN with one notice. At 131,072 positions the bare command with an empty golden cache matched serial YaRN on all 256 tokens, the full logit ledger and target state.
- **Off switch:** `STRIX_QWEN_YARN_MTP=0` (or no bundle and no `--mtp`) keeps serial YaRN; `STRIX_QSA_MTP_POOL=0` and `STRIX_PLE_ROLLING=0` restore the previous pool and PLE paths.
- **Measured gain:** MTP speculative decode on YaRN contexts through 1,048,576 positions on one node; serial YaRN is the comparator.
- **Implementation / evidence:** [YaRN MTP CLI](engine/speculative/mtp-native-cli.md#explicit-yarn-mtp-cli)

<a id="qwen-tool-draft"></a>

### Tool-call frame drafts

**Engine:** `qwen`. **Control:** `STRIX_BATCH_TOOL_DRAFT`, request field `tool_draft`, `STRIX_TOOL_DRAFT_PREFIX_GATE`. **Default:** ON (batched decoding).

Inside a tool-call frame, the MTP draft chain proposes the continuation of an earlier closed frame once the open frame's first 16 committed tokens match that frame. The target verifies every proposed row.

- **Preconditions:** Batched MTP decoding with a chat template that defines tool-call frame tokens.
- **Exactness evidence:** Drafts only: the target verifies every row, so committed tokens are unchanged; 9 tool-call cases 54/54 exact, 24/24 on repeated waves, and 18/18 default against off in one process.
- **Off switch:** `STRIX_BATCH_TOOL_DRAFT=0` (or `"tool_draft":false` per request); `STRIX_TOOL_DRAFT_PREFIX_GATE=0` restores the ungated drafter
- **Measured gain:** Tool-call traces: one case +3.3 to +3.5 %, all others within -0.08 to +0.9 %.
- **Implementation / evidence:** [frame drafts](../include/strix/mtp_frame_draft.hpp)

<a id="qwen-ple-warm-parallel"></a>

### Parallel cold PLE table populate

**Engine:** `qwen`. **Control:** `STRIX_PLE_WARM_PARALLEL=0`. **Default:** ON.

When the file-backed PLE table is not in the page cache (another model evicted it), its non-resident 64 MiB chunks are first read from up to 16 threads; the existing random-access warm then only maps resident pages.

- **Preconditions:** Linux; file-backed PLE table; `STRIX_PLE_WARM` enabled; not with `STRIX_MODEL_COPY_ALL` or an admitted PLE reclaim. Best effort (prefault only).
- **Exactness evidence:** Load path only: advisory reads of the same file bytes; generated text, tokens and finish reasons identical with and without it (5 requests).
- **Off switch:** `STRIX_PLE_WARM_PARALLEL=0` keeps the single random-access pass
- **Measured gain:** Cold start: listening 137.8 to 83.8 s, first response 139.3 to 91.9 s, PLE warm 94.1 to 6.4 s; restart with a partly cached table: listening 23.3 to 7.9 s. Decode unchanged.
- **Implementation / evidence:** [implementation status](reference/implementation-status.md)

## DeepSeek two-node

```sh
strix-deepseek41-serve MODEL.gguf --rank R --transport usb4auto
```

Launch one rank on each node using the same assets and compatible transport configuration. Plain rank,
batch, prefill and serve invocations resolve the production configuration. The EP prefill transport spelling
is `usb4-auto`. Inspect with `--print-effective-config`. The resolver discovers packs under `STRIX_DS41_PACKS`,
then the model's `packs` directory, then the model directory. Missing prerequisites print one
`deepseek41 defaults:` notice per dropped item; explicit arguments/environment win. See
[production defaults](models/deepseek-v4.1/production-defaults.md).

`STRIX_DS41_BASELINE=1` stops pack/draft/argv injection and sets the nine recent defaults off unless explicitly
overridden: grouped shared work, shared down, MoE prefill, dense selector, three attention settings, serving
warm-up and calibration. Older kernel/transport defaults remain; disable them individually for an older reference.
The combined qualified build measured 1K/8K TTFT 2.098 -> 1.954 s / 11.090 -> 9.757 s, R1 -1.9%,
R6 -2.4%, N8 -2.2%, and warmed C1 43.58 -> 48.00 tok/s. These are build gains, not item attribution.

<a id="ds-prepared-load"></a>

### Prepared rank-local expert loading

**Engine:** `deepseek`. **Control:** `STRIX_DS41_PREPARED_LOAD; STRIX_DS41_PREPARED_ROOT`. **Default:** auto discovery.

- **Preconditions:** An offline-prepared local EP/TP pack whose manifest exactly matches the model, overlay, trellis/environment, rank/world, EP/TP and down layout. Serve with expert TP and device prefill uses TP dup; the measured packs were TP packed.
- **Automatic selection:** Immediate children of STRIX_DS41_PREPARED_ROOT, else XDG_CACHE_HOME/strix/ds41-prepared, else HOME/.cache/strix/ds41-prepared; lexical first valid match. Manifest validation and payload size checks only at discovery; chunks verified before upload. Runtime never writes packs.
- **Fallback notice:** One concise no-match line and original loader for automatic misses; explicit directory rejections retain their reason and original loader fallback.
- **Exactness evidence:** R1 tokens (63 position records) identical in every arm on two Strix Halo nodes, production identity, fresh process per arm, ABBA x2, for both the explicit-directory and the default-discovery A/B; TP packed packs accepted on both ranks (72 GiB/node).
- **Off switch:** STRIX_DS41_PREPARED_LOAD=0. An explicit directory bypasses discovery.
- **Measured gain:** Default discovery vs STRIX_DS41_PREPARED_LOAD=0, fresh process per arm, ABBA x2: whole-run wall -22.1 s mean (66.0 -> 43.8 s; discovery 43.78-43.86 s, original loader 59.6-81.8 s, its first visit with a cold page cache). Explicit-directory A/B: -20.6 s (63.7 -> 43.0 s).
- **Implementation / evidence:** [Prepared-load contract and one-time preparation](models/deepseek-v4.1/prepared-load.md), [loader](../src/hip_ds41_prepared_load.hpp).

<a id="ds-packs"></a>

### Production dense overlay

**Engine:** `deepseek`. **Control:** `STRIX_DS41_OVERLAY; STRIX_DS41_CB4`. **Default:** auto.

Discover c-u6.gguf and enable CB4.

- **Preconditions:** Complete production packs with valid model content.
- **Automatic selection:** Set overlay and CB4=1 only when packs are found; existing environment wins.
- **Fallback notice:** Missing resolver prerequisites print a `deepseek41 defaults:` notice; ordinary operator shape dispatch has no universal decline notice. Explicit incompatible configurations retain errors.
- **Exactness evidence:** Both-rank reference gates compare prefill logits/state digests, greedy token IDs, verify rows and batch rows with fixed Q2 experts, dense overlay and trellis packs; pack identity changes require their separate quality reference.
- **Off switch:** STRIX_DS41_OVERLAY=0
- **Measured gain:** No isolated measurement
- **Implementation / evidence:** [Resolver](../src/deepseek41_pair_defaults.cpp), [contract](models/deepseek-v4.1/production-defaults.md)

<a id="ds-packs-trellis"></a>

### Production trellis dense identity

**Engine:** `deepseek`. **Control:** `STRIX_DS41_TRELLIS; STRIX_DS41_TRELLIS_MIX; STRIX_DS41_TRELLIS_SITES; STRIX_DS41_TRELLIS_DENSE_FULL; STRIX_DS41_TRELLIS_HEADOUT`. **Default:** auto.

Select ours-K4 dense, ours-K5 head, all sites, full dense and head output.

- **Preconditions:** Both packs, dense_r1 trellis library and API; excluded from the EP prefill app.
- **Automatic selection:** All-or-nothing discovered identity; no partial automatic pack set.
- **Fallback notice:** Missing resolver prerequisites print a `deepseek41 defaults:` notice; ordinary operator shape dispatch has no universal decline notice. Explicit incompatible configurations retain errors.
- **Exactness evidence:** Both-rank reference gates compare prefill logits/state digests, greedy token IDs, verify rows and batch rows with fixed Q2 experts, dense overlay and trellis packs; pack identity changes require their separate quality reference.
- **Off switch:** STRIX_DS41_TRELLIS=0; STRIX_DS41_PACKS=0 disables discovery
- **Measured gain:** No isolated measurement
- **Implementation / evidence:** [Resolver](../src/deepseek41_pair_defaults.cpp), [contract](models/deepseek-v4.1/production-defaults.md)

<a id="ds-topology"></a>

### TP attention and direct Engram

**Engine:** `deepseek`. **Control:** `--tp-attention; --engram-direct`. **Default:** auto.

Use two-rank TP attention and direct Engram reads in rank/batch apps.

- **Preconditions:** Rank/batch invocation; two-rank transport for TP attention.
- **Automatic selection:** Enable unless the corresponding explicit opt-out is present.
- **Fallback notice:** Missing resolver prerequisites print a `deepseek41 defaults:` notice; ordinary operator shape dispatch has no universal decline notice. Explicit incompatible configurations retain errors.
- **Exactness evidence:** Both-rank reference gates compare prefill logits/state digests, greedy token IDs, verify rows and batch rows with fixed Q2 experts, dense overlay and trellis packs; pack identity changes require their separate quality reference.
- **Off switch:** --no-tp-attention; --engram-mapped
- **Measured gain:** No isolated measurement
- **Implementation / evidence:** [Resolver](../src/deepseek41_pair_defaults.cpp), [contract](models/deepseek-v4.1/production-defaults.md)

<a id="ds-serve-shape"></a>

### Serving production shape

**Engine:** `deepseek`. **Control:** `--peer-timeout; --prefill-path; --slots; --prefill-rows`. **Default:** auto.

Resolve timeout 120, device prefill, 8 slots and `--prefill-rows auto`: the largest prefill piece (16384, 8192 or 4096 rows, at most the context) whose scratch and the prefill's own weights both ranks admit after the weights load (else 2048). Each piece re-reads every routed expert and replays the decoder cone, so fewer, larger pieces are faster; pieces are bitwise-exact for any size.

- **Preconditions:** Serve app and capacity for the requested slots/rows.
- **Automatic selection:** Prepend defaults before caller arguments so explicit flags win.
- **Fallback notice:** Missing resolver prerequisites print a `deepseek41 defaults:` notice; ordinary operator shape dispatch has no universal decline notice. Explicit incompatible configurations retain errors.
- **Exactness evidence:** Both-rank reference gates compare prefill logits/state digests, greedy token IDs, verify rows and batch rows with fixed Q2 experts, dense overlay and trellis packs; pack identity changes require their separate quality reference.
- **Off switch:** Explicit values override each flag (`--prefill-rows N`); STRIX_DS41_BASELINE=1 suppresses injection
- **Measured gain:** Pair, context 49152, fresh prompts: 8,399 tokens 14.57 -> 10.34 s, 32,499 tokens 60.8 -> 39.1 s (16384 vs 2048 rows), same logits and tokens.
- **Implementation / evidence:** [Resolver](../src/deepseek41_pair_defaults.cpp), [contract](models/deepseek-v4.1/production-defaults.md)

<a id="ds-draft"></a>

### Automatic DSpark and vocabulary

**Engine:** `deepseek`. **Control:** `--dspark; --dspark-vocab`. **Default:** auto.

Discover a DSpark-MXFP4 GGUF and auto vocabulary beside the model/packs.

- **Preconditions:** Draft asset; core48k.bin and core32k.bin for vocabulary; not BATCH_ROWS16 or prefix-session snapshots.
- **Automatic selection:** Missing draft disables drafting; missing vocabulary keeps full vocabulary.
- **Fallback notice:** Missing resolver prerequisites print a `deepseek41 defaults:` notice; ordinary operator shape dispatch has no universal decline notice. Explicit incompatible configurations retain errors.
- **Exactness evidence:** Both-rank reference gates compare prefill logits/state digests, greedy token IDs, verify rows and batch rows with fixed Q2 experts, dense overlay and trellis packs; pack identity changes require their separate quality reference.
- **Off switch:** --no-dspark; --dspark-vocab full
- **Measured gain:** No isolated measurement
- **Implementation / evidence:** [Resolver](../src/deepseek41_pair_defaults.cpp), [contract](models/deepseek-v4.1/production-defaults.md)

<a id="ds-engram-tp"></a>

### Engram projection split across ranks

**Engine:** `deepseek`. **Control:** `STRIX_DS41_ENGRAM_TP`. **Default:** on for steps of up to 8 rows.

The two ranks each compute half of the Engram projection's output rows (same per-row arithmetic) and exchange the
halves, instead of both reading the whole ~160 MB weight.

- **Preconditions:** Two ranks with the F8 Engram KV (STRIX_DS41_F16_AS_F8); otherwise each rank runs the full projection.
- **Automatic selection:** On for every DeviceForward / BatchForward step of up to 8 rows (decode or prompt); `=r1` restores the previous single-row-only default.
- **Fallback notice:** Without two ranks or the F8 Engram KV, the full projection runs on each rank.
- **Exactness evidence:** Bitwise: R1-R8 verify windows with rejects equal the archived full-projection digests on both ranks; batch N1/N2/N4/N8 in-process A/B (fed, argmax, gathered logits per row and final attention states) identical, also on the communicator fallbacks.
- **Off switch:** STRIX_DS41_ENGRAM_TP=0 (off), =r1 (single-row steps only)
- **Measured gain:** R1 decode -0.61 ms/step (-2.1 %); verify windows (rank app, in-process ABA x4, every arm bitwise on both ranks, steps of up to 8 rows vs off): R1 -2.1 %, R2 -1.5 %, R4 -1.0 %, R6 -1.0 % (95 % intervals below zero), R8 -0.6 % (3/3 pairs faster)
- **Implementation / evidence:** [production defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-engram-f16-pf"></a>

### Engram prefill projection tile with prefetch

**Engine:** `deepseek`. **Control:** `STRIX_DS41_ENGRAM_F16_PF`. **Default:** on.

The prefill Engram projection runs a 48 x 64 LDS tile (6 x 16 accumulators per lane, 1024 threads) that loads the next
128-element slab into registers while the current slab is computed, instead of the 64 x 64 tile that stages after a barrier.

- **Preconditions:** F16 Engram KV, the full aligned view (width 6144, 25600 outputs, first 0, 16-byte aligned x and weights, valid strides), 32..8192 rows, row padding at most 9/8 of the 64-row tile (49..64 rows excluded). F8 / TP-split views and other F16 projections keep enqueue_f16_rows.
- **Automatic selection:** Every prefill Engram piece (hip_deepseek41_prefill.hip) that meets the preconditions.
- **Fallback notice:** Other views and row counts take enqueue_f16_rows silently (same bits).
- **Exactness evidence:** Bitwise: every output byte equals enqueue_f16_rows (rows 1..8192 incl. tile edges 47/48/49/63/64/65/95/96/97, padded strides with guard values, 3 independent F16 weight matrices, BF16-rounded inputs with signed zeros); same per-lane chains (k = lane, lane + 32, ... ascending, separate multiply and add rounding, no contraction), same butterfly and storing lane. Compiler resources: 164 VGPR, no spill or scratch.
- **Off switch:** STRIX_DS41_ENGRAM_F16_PF=0 (read on every call)
- **Measured gain:** Isolated Engram projection (6144 -> 25600, event medians, interleaved arms, 3 weight copies) vs enqueue_f16_rows: rows 32 0.54x, 128 0.87x, 256 0.94x, 512 0.86x, 1024 0.86x (88.1 -> 75.8 ms), 2048 0.86x, 4096 0.86x, 8192 0.87x; 49..64 rows keep enqueue_f16_rows (64 rows: 1.16x). End-to-end TTFT not separately measured.
- **Implementation / evidence:** [kernel](../src/hip_ds41_layer.hip), [gate and bench](../tests/hip_ds41_engram_f16_pf_tests.hip)

<a id="ds-shared-grouped"></a>

### Grouped shared expert

**Engine:** `deepseek`. **Control:** `STRIX_DS41_SHARED_GROUPED`. **Default:** ON.

Group shared-expert SwiGLU pairs.

- **Preconditions:** Supported shared-expert shapes and production dense identity.
- **Automatic selection:** Default on; shape dispatch keeps compatible kernels.
- **Fallback notice:** Missing resolver prerequisites print a `deepseek41 defaults:` notice; ordinary operator shape dispatch has no universal decline notice. Explicit incompatible configurations retain errors.
- **Exactness evidence:** Both-rank reference gates compare prefill logits/state digests, greedy token IDs, verify rows and batch rows with fixed Q2 experts, dense overlay and trellis packs; pack identity changes require their separate quality reference.
- **Off switch:** STRIX_DS41_SHARED_GROUPED=0
- **Measured gain:** No isolated measurement
- **Implementation / evidence:** [Resolver](../src/deepseek41_pair_defaults.cpp), [contract](models/deepseek-v4.1/production-defaults.md)

<a id="ds-core"></a>

### Expert TP and batched operators

**Engine:** `deepseek`. **Control:** `STRIX_DS41_EXPERT_TP; STRIX_DS41_EXPERT_SLOTS; STRIX_DS41_ROUTES_FAST; STRIX_DS41_F16_PAIR; STRIX_DS41_TP_DOWN; STRIX_DS41_BATCH_SEGMENTS`. **Default:** ON.

Splits expert work across the pair, retains expert slots, builds integer routes faster, pairs eligible F16 projections and batches attention segments.

- **Preconditions:** Two ranks and supported expert geometry; TP down width 1152; segment batching does not combine with BATCH_ROWS16=1.
- **Automatic selection:** Each is on unless disabled; BATCH_SEGMENTS declines itself when STRIX_DS41_BATCH_ROWS16=1. Shape guards keep supported older paths.
- **Fallback notice:** Routine operator eligibility has no general decline log. Invalid expert-TP geometry fails validation; do not treat a transport change as the same numerical identity.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms.
- **Off switch:** Set one of STRIX_DS41_EXPERT_TP, STRIX_DS41_EXPERT_SLOTS, STRIX_DS41_ROUTES_FAST, STRIX_DS41_F16_PAIR, STRIX_DS41_TP_DOWN or STRIX_DS41_BATCH_SEGMENTS to 0 per comparison.
- **Measured gain:** Batch segments N8 +9%; TP down R6 -3%, N8 +1.6%
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-scheduling"></a>

### Serving interleave and draft admission

**Engine:** `deepseek`. **Control:** `STRIX_DS41_SERVE_SCHED; --dspark-ranks; --dspark-max-inflight; --draft-ranks`. **Default:** auto.

Interleaves prompt work with decode; uses both drafters and switches to plain target steps above the in-flight threshold.

- **Preconditions:** Device prefill serving; two draft ranks require the drafter assets on both ranks.
- **Automatic selection:** Device serving defaults to interleave, --dspark-ranks 2 and --dspark-max-inflight 4; batch defaults --draft-ranks 2. Row prefill keeps the compatible scheduler.
- **Fallback notice:** Scheduler selection follows the prefill path; no universal decline line. Draft proposal changes are visible in acceptance/timing, while committed tokens remain target verified.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms.
- **Off switch:** STRIX_DS41_SERVE_SCHED=off; independently --dspark-ranks 1, --dspark-max-inflight 0 (always draft), or batch --draft-ranks 1
- **Measured gain:** 8K under load: decode stall <=1 s instead of ~13 s; prompt TTFT about doubles
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-transport"></a>

### USB4 eager sends and fused staging

**Engine:** `deepseek`. **Control:** `STRIX_DIST_USB4_EAGER=65536; STRIX_DIST_STAGING=fused; --transport usb4gpu`. **Default:** auto.

Sends small payloads eagerly and lets TP products write/read pinned transport buffers directly.

- **Preconditions:** Two-node USB4 routes; fused staging applies to host-path routes. GPU-direct requires streams exposing that capability.
- **Automatic selection:** Eager threshold is 65536 bytes; fused staging is the default. usb4gpu chooses the device exchange only with GPU-direct streams.
- **Fallback notice:** Without GPU-direct streams the apps print gpu-direct unavailable ... host path only and use the host path with the same bits.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms.
- **Off switch:** STRIX_DIST_USB4_EAGER=0 or STRIX_DIST_STAGING=copy, separately; transport CLI spelling for the prefill app is usb4-gpu
- **Measured gain:** Eager R1 +2.8%; fused staging R1 -0.2..-0.4 ms/token
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-ready"></a>

### Ready views, CRC overlap and deferred checks

**Engine:** `deepseek`. **Control:** `STRIX_DIST_CRC_PARALLEL; STRIX_DIST_HOST_CRC_OVERLAP; STRIX_DIST_HOST_CHUNK_BYTES=1048576; STRIX_DIST_STAGED_READY_BYTES=262144; STRIX_DS41_READY_LEAN; STRIX_DS41_MOE_READY; STRIX_DS41_OB_READY_EPILOGUE; STRIX_DS41_READY_READS_ONLY; STRIX_DS41_TRELLIS_OB_READY; STRIX_DIST_CRC_DEFER`. **Default:** auto.

Publishes readiness from producers, reduces staging copies and overlaps CRC work with transfer without removing corruption checks.

- **Preconditions:** Compatible ready-view routes and identical transport modes on both ranks; trellis ready applies at trellis o_b sites.
- **Automatic selection:** Ready lean/MoE and trellis ready are on; epilogue/reads-only follow READY_LEAN. Chunks default to 1 MiB, readiness views to 256 KiB; CRC defer checks TX/RX via the next trailer.
- **Fallback notice:** Routes without ready views retain the previous staging/checks; Q8 o_b epilogues do not run at trellis o_b sites. There is no universal decline notice. Corruption fails the step and mixed deferred modes fail immediately.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms. Transport integrity gates remain required; token equality alone does not exercise corruption handling.
- **Off switch:** Set one boolean above to 0 per comparison; STRIX_DIST_HOST_CHUNK_BYTES=0 selects one piece, STRIX_DIST_STAGED_READY_BYTES=0 removes readiness views.
- **Measured gain:** CRC overlap bundle R6 -1.37%, N8 -1.77%; trellis ready + deferred CRC R1/R6/N8 -1.6/-1.9/-0.7%
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-overlap"></a>

### Shared expert overlap and late HC

**Engine:** `deepseek`. **Control:** `STRIX_DS41_BATCH_SHARED=overlap; STRIX_DS41_LATE_HC_LAYERS=all`. **Default:** auto.

Hides shared-expert work and attention-site HC mixes inside exchange waits.

- **Preconditions:** Eligible batch half-sum / attention o_b exchanges; late HC does not combine with BATCH_PREFETCH.
- **Automatic selection:** Batch shared defaults to overlap; late HC defaults to all eligible layers. With batch prefetch the late-HC path stays out.
- **Fallback notice:** Ineligible overlap retains the preceding ordering without a general decline line; explicit split shared mode remains a separate choice.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms.
- **Off switch:** STRIX_DS41_BATCH_SHARED=off or STRIX_DS41_LATE_HC_LAYERS=none, separately
- **Measured gain:** Shared overlap N1/N2/N4/N8 -2.8/-3.2/-5.1/-5.2%; late HC R1 -0.2 ms, R6 -0.35 ms
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-prefill"></a>

### Final KV ownership, prefill tiles and Engram cache

**Engine:** `deepseek`. **Control:** `STRIX_DS41_PREFILL_FINAL_KV=all; STRIX_DS41_PREFILL_TILES=1 (opt-in); STRIX_DS41_PREFILL_TILES_MIN_ROWS=2048; STRIX_DS41_ENGRAM_PAYLOAD_CACHE`. **Default:** auto.

Transfers final state instead of intermediate FP32 K/V and caches repeated Engram payloads; unsplit prefill tiles are available as an opt-in.

- **Preconditions:** Row-split prefill with final ring/compressed-state owners; tile runs at least 2048 rows by default; a cache budget that fits memory.
- **Automatic selection:** Final KV defaults to all; prefill tiles are off (split-K kernels), and with STRIX_DS41_PREFILL_TILES=1 they apply above the row threshold; payload caching is on with STRIX_DS41_ENGRAM_PAYLOAD_CACHE_MIB controlling its cap.
- **Fallback notice:** Subthreshold tiling silently keeps split-K. Cache misses use normal reads. No general decline line is printed for these ordinary selections.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms. Final KV qualification used prefill --reference and identical decode continuation.
- **Off switch:** STRIX_DS41_PREFILL_FINAL_KV=none or STRIX_DS41_ENGRAM_PAYLOAD_CACHE=0, separately (tiles are opt-in: STRIX_DS41_PREFILL_TILES=1)
- **Measured gain:** Final KV 8K 714.9 -> 729.4 tok/s; cache 32K TTFT -1.63%; tiles on are slower on the current kernels (TTFT 2K/8K/32K +1.0/+3.4/+1.9 %), hence off
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-prefetch"></a>

### Q8 exchange read-ahead

**Engine:** `deepseek`. **Control:** `STRIX_DS41_BATCH_PREFETCH; STRIX_DS41_PREFETCH_SIDE; STRIX_DS41_PREFETCH_ROW_MB; STRIX_DS41_DECODE_PREFETCH`. **Default:** auto.

Reads future dense weights into MALL while the ranks wait for exchange.

- **Preconditions:** Q8 o_b exchange waits; trellis sites use their own packs and are skipped.
- **Automatic selection:** Batch defaults to 24 MB with side stream on and 3 MB per row; decode defaults to 2 MB per row in the o_b window. Only Q8 sites read ahead.
- **Fallback notice:** Trellis sites skip read-ahead silently. PREFETCH_SIDE=0 chooses the main-stream form rather than disabling all prefetch.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms.
- **Off switch:** STRIX_DS41_BATCH_PREFETCH=0 or STRIX_DS41_DECODE_PREFETCH=0, separately
- **Measured gain:** Batch N2/N4/N8 -2.1/-4.9/-3.3%; decode R6 -1.8% on Q8 sites
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-index"></a>

### Indexer pruning and shared segment geometry

**Engine:** `deepseek`. **Control:** `STRIX_DS41_IDX_DIRECT; STRIX_DS41_IDX_CAND_FILTER; STRIX_DS41_ATTN_SEG_HPW`. **Default:** auto.

Skips redundant indexing, scores only candidates and shares eligible segment launch geometry.

- **Preconditions:** Direct indexing count <=512; candidate filtering at pool layers above 16K entries; shared heads-per-wave for N8 segments.
- **Automatic selection:** Each switch is on; its count/context/segment predicate decides whether the optimized form applies.
- **Fallback notice:** Outside the predicate the previous indexer/segment form runs without a general decline line.
- **Exactness evidence:** In-process ABA x3 per cell compared tokens, logits slices and attention states on both ranks; prefill 32K/400 rows compared logits and state digest.
- **Off switch:** STRIX_DS41_IDX_DIRECT=0, STRIX_DS41_IDX_CAND_FILTER=0 or STRIX_DS41_ATTN_SEG_HPW=0, separately
- **Measured gain:** Short R1/R6/N8 -2.0/-2.7/-1.0%; 128K -0.8/-1.9/-3.3% (combined)
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-exact-decode-launches"></a>

### Exact decode launch and staging options

**Engine:** `deepseek`. **Control:** `STRIX_DS41_TOUCH_BATCH; STRIX_DS41_TRELLIS_ROW_VIEWS; STRIX_DS41_NORM_ILP; STRIX_DS41_R1FUSE_ROUTES; STRIX_DS41_COMPRESS_FAST`. **Default:** on.

Five changes to launches, loads and staging only, never to the arithmetic: a prefetch window's read-ahead touches in one launch (up to 16 spans), strided rows passed straight to the dense-r1 trellis plan instead of a staging copy, the norm kernel's loads gathered before its first add, the router's top-k writing the expert routes directly, and the compressor kernel in fewer instructions (hardware F16 conversion of finite weights, 16-byte latent reads, the next weight step requested ahead).

- **Preconditions:** Two-node decode; TRELLIS_ROW_VIEWS needs the dense_r1 trellis API and applies to multi-row steps.
- **Automatic selection:** All five are on; nothing to select.
- **Fallback notice:** None; `=0` restores the previous form.
- **Exactness evidence:** In-process ABA on both ranks: R1 tokens + logits slice, verify-window tokens and the attention state of every arm equal arm 0's; batch N8 every row's fed/argmax/logits hash and final attention states equal.
- **Off switch:** `=0` on each variable, separately
- **Measured gain:** GPU, two Strix Halo nodes over 2x USB4 Gen3, PerfOpt kernel, production packs; one binary, all five off vs all five on in the same process (rank app --aba 5, batch app --aba 3, first pair discarded): R1 26.94 -> 26.68 ms/token (-0.95 %, paired 95 % interval -0.28..-0.24 ms, 4/4 pairs faster); R6 verify window 51.43 -> 51.08 ms (-0.68 %, 4/4 pairs faster); batch N8 decode 71.19 -> 70.91 ms/step (-0.40 %), wall -1.16 % (2/2 pairs faster); Per option, alone vs off (in-process ABA x4, 3 warm pairs, every arm bitwise on both ranks): TOUCH_BATCH R1 -0.31 % (95 % interval -0.14..-0.03 ms), TRELLIS_ROW_VIEWS R1 -0.08 % (-0.034..-0.009 ms), NORM_ILP R1 -0.35 % (-0.18..-0.01 ms), COMPRESS_FAST R1 -0.34 % and R6 -0.59 % (both intervals below zero), R1FUSE_ROUTES R1 -0.11 % (3/3 pairs faster, interval -0.066..+0.005 ms); R6 not slower for any of them
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-trellis"></a>

### Grouped trellis o_a and q_a/KV pair

**Engine:** `deepseek`. **Control:** `STRIX_DS41_TRELLIS_OA_GROUP; STRIX_DS41_TRELLIS_QAKV_PAIR`. **Default:** auto.

Groups o_a projections and submits q_a with KV in one launch.

- **Preconditions:** Production K4 dense / K5 head packs and a dense_r1 build linked to a library exposing dense-r1 and grouped GEMV APIs.
- **Automatic selection:** On for the admitted trellis sites; the production environment selects the packs independently of these scheduling switches.
- **Fallback notice:** Unsupported library/shape admission keeps supported staged forms or rejects an explicit incompatible configuration; there is no single generic decline notice for both adapters.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms. Exactness is against unchanged packs, not the original Q8_0 weights; pack quality has a separate gate.
- **Off switch:** STRIX_DS41_TRELLIS_OA_GROUP=0 or STRIX_DS41_TRELLIS_QAKV_PAIR=0, separately
- **Measured gain:** Combined R1 34.7 -> 31.15 ms (-10.2%); R6 -4.1%; N8 -2.1%
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-shared-down"></a>

### Single-launch shared down

**Engine:** `deepseek`. **Control:** `STRIX_DS41_SHARED_DOWN_FUSED`. **Default:** auto.

Replaces staged shared-down submission with one declared fused launch.

- **Preconditions:** All shared-down sites must be K4 [5120,2304] in one context, no shared-expert TP, and the linked library must pass declared-workspace probes for the supported rows.
- **Automatic selection:** Unset means try once at startup; an implicit refusal is retained for the process. Explicit =1 demands admission and throws if it fails.
- **Fallback notice:** Prints ds41 trellis: shared down fused launch not admitted by default (...): old path on implicit refusal; prints shared down as one fused launch ... when admitted. Once used, a later incompatible admission is an error.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms.
- **Off switch:** STRIX_DS41_SHARED_DOWN_FUSED=0
- **Measured gain:** -
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md), [startup probe](../src/hip_ds41_trellis.hip)

<a id="ds-moe7"></a>

### Split gate/up prefill form 7

**Engine:** `deepseek`. **Control:** `STRIX_DS41_MOE_PREFILL`. **Default:** ON.

Uses the qualified split gate/up MoE prefill schedule.

- **Preconditions:** Supported device-prefill expert shapes and quantized formats.
- **Automatic selection:** Default form 7; supported row/format dispatch determines its kernel, with the previous form selected by =0.
- **Fallback notice:** No general decline line for normal kernel selection; invalid geometry still fails the operator validation.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms.
- **Off switch:** STRIX_DS41_MOE_PREFILL=0
- **Measured gain:** -
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-defer-decoder"></a>

### Deferred decoder cone of non-final prefill pieces

**Engine:** `deepseek`. **Control:** `STRIX_DS41_DEFER_DECODER`. **Default:** ON.

A prefill piece runs the decoder layers on the cone of rows that the last row's logits, the decode state and the drafter seed need. For a piece that is not the last one, nobody reads those: the next piece's own cone lies inside the next piece, and only the first decoder layer's compressed-KV and index-key entries of every row are read across pieces. The server now runs such a piece with the decoder layers reduced to those entries (no output rows, head or seed) when the next piece's planned cone starts at least one attention window after its first row, there is no prefix-cache insert at the piece's end and no snapshot in it or the next piece. Both ranks agree on the decision before the run, the next piece keeps the planned length, and it checks its own cone before it runs.

- **Preconditions:** Two-node server device prefill of more than one piece, full layer set with the cone.
- **Automatic selection:** On for every non-final piece that meets the conditions above; other pieces run as before.
- **Fallback notice:** None.
- **Exactness evidence:** Bitwise: two-node serve logits and 32 generated ids identical to the published references at 8,399 / 32,499 / 44,699 tokens with the lever on and off, and identical on and off for a 16,500-token prompt whose short last piece is not admitted.
- **Off switch:** STRIX_DS41_DEFER_DECODER=0
- **Measured gain:** Two-node serve TTFT, two GPU turns per arm, one load per arm: 8.30 / 29.90 / 43.07 s off vs 8.19 / 28.92 / 40.51 s on at 8,399 / 32,499 / 44,699 tokens (-3.3 % at 32,499, -5.9 % at 44,699; one piece at 8,399).
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-engram-split-read"></a>

### Split Engram read of a prefill piece

**Engine:** `deepseek`. **Control:** `STRIX_DS41_ENGRAM_SPLIT_READ`. **Default:** ON.

A prefill piece reads the Engram rows of all its Engram layers on a thread while the layers before the first Engram layer run. The thread now reads the first Engram layer's rows first and signals them; the first Engram layer waits only for those, and the later Engram layers' rows keep arriving while the layers in between run. Before, the first Engram layer waited for every Engram layer's rows (about 1.5 s for a cold 16,384-row piece, with layer 0 covering about 0.7 s of it).

- **Preconditions:** Device prefill with the overlapped Engram read (the default) and at least two Engram layers.
- **Automatic selection:** On for every prefill run that reads Engram rows.
- **Fallback notice:** None.
- **Exactness evidence:** Bitwise: the same rows land in the same buffer; two-node serve logits and generated ids identical to the published references at 8,399 / 32,499 / 44,699 tokens with the lever on and off.
- **Off switch:** STRIX_DS41_ENGRAM_SPLIT_READ=0
- **Measured gain:** Two-node serve TTFT, two GPU turns in A/B and B/A order, one load per arm: 9.01 / 31.87 / 45.04 s off vs 8.50 / 30.79 / 44.41 s on at 8,399 / 32,499 / 44,699 tokens (-5.6 / -3.4 / -1.4 %).
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-engram-lookahead"></a>

### Engram lookahead between prefill pieces

**Engine:** `deepseek`. **Control:** `STRIX_DS41_ENGRAM_LOOKAHEAD`. **Default:** ON.

The server's prefill task names the next piece of its plan before each piece. Once the current piece's own Engram rows are read, a background thread reads this rank's Engram rows of the next piece (same row map and Engram history) through the row reader, whose payload cache then serves the next piece's read from host memory. Before, the GPU waited for that read before the first Engram layer of every piece (about 0.9 s per 16,384-row piece).

- **Preconditions:** Two-node server device prefill of more than one piece; Engram payload cache on (the default).
- **Automatic selection:** On for every piece that has a next piece; a single-piece prompt is unchanged.
- **Fallback notice:** None; a lookahead that does not match the next piece only costs extra reads.
- **Exactness evidence:** Bitwise: the cache returns the same bytes; two-node serve logits and generated ids identical to the published references at 8,399 / 32,499 / 44,699 tokens with the lever on and off.
- **Off switch:** STRIX_DS41_ENGRAM_LOOKAHEAD=0 (STRIX_DS41_ENGRAM_LOOKAHEAD_LOG=1 prints each piece's Engram read and the lookahead read)
- **Measured gain:** Two-node serve TTFT, same GPU turn, one load per arm: 9.66 / 34.53 / 49.13 s with the lever off vs 9.51 / 33.39 / 47.77 s on at 8,399 / 32,499 / 44,699 tokens (-3.3 % at 32,499, -2.8 % at 44,699); payload-cache hits 123,230 -> 1,074,078 with the same disk reads.
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-moe-v2"></a>

### Split prefill MoE kernels v2

**Engine:** `deepseek`. **Control:** `STRIX_DS41_PREFILL_MOE_V2; STRIX_DS41_PREFILL_MOE_V2_Q8; STRIX_DS41_PREFILL_MOE_BAND`. **Default:** ON.

IQ2_XXS gate/up and Q2_K down split kernels with the same products, chains, tree and epilogue as forms 7 / 6: each class-wave owns classes q + 8k (tree levels 1-2 in registers, a quarter of the LDS tree exchange; a partial last chunk is balanced across waves and loads only its blocks), two LDS-only barriers per chunk, the Q2_K minimum term as an exact fma chain, permlanes without source copies, weight tiles from two uniform bases with shared 32-bit offsets, and blocks dispatched in bands of 4 batches. The Q8_0 shared expert's gate/up and down (and the Q8_0 x A8 projections on 16-row site routes) run on the same v2 kernels with direct per-lane weight reads (STRIX_DS41_PREFILL_MOE_V2_Q8).

- **Preconditions:** Default form 7, batches of 16 rows, IQ2_XXS gate/up with hidden a multiple of 1024, Q2_K down with width a multiple of 256; Q8_0 gate/up and down.
- **Automatic selection:** On for the eligible split launches; other formats and forms keep their kernels.
- **Fallback notice:** None; ineligible shapes run the form 7 / 6 kernels silently.
- **Exactness evidence:** Bitwise: device prefill expert tests (synthetic incl. empty, 1-row and tail experts, real layer 20, DSpark MTP) with v2 on and off; rank-shaped bench (192 of 384 experts, pieces of 2048/4096/8192 rows) A8 codes, deltas, sums and down vectors identical to the previous kernels; the same for the Q8_0 shared expert (STRIX_DS41_PREFILL_MOE_V2_Q8 on and off). With NaN weight scales or activation deltas (never in valid tensors) every non-NaN result and every NaN position is identical; the payload or sign of a NaN result can differ.
- **Off switch:** STRIX_DS41_PREFILL_MOE_V2=0 (STRIX_DS41_PREFILL_MOE_V2_Q8=0 keeps the Q8_0 split kernels of form 7 / 6) (STRIX_DS41_PREFILL_MOE_BAND=1 keeps batch-fastest order)
- **Measured gain:** Rank-shaped kernel bench (layer 20, interleaved): down -27 / -30 / -30 %, gate/up -4.5 / -6.1 / -7.8 % at 2048 / 4096 / 8192-row pieces. Two-node serve TTFT at 8,399 / 32,499 / 44,699 tokens (one binary, cache off, one load per arm, separate GPU turns): STRIX_DS41_PREFILL_MOE_V2=0 10.42 / 37.29 / 53.35 s vs default 9.80 / 34.70 / 49.09 s (-5.9 / -6.9 / -8.0 %). Q8_0 shared expert (one expert, all rows), v2 vs STRIX_DS41_PREFILL_MOE_V2_Q8=0: gate/up -21 / -20 / -32 %, down -26 / -33 / -40 % at 2048 / 4096 / 8192 rows.
- **Public evidence:** [Prefill MoE kernels](../src/hip_ds41_prefill.hip)

<a id="ds-router-v2"></a>

### Prefill router scores v2

**Engine:** `deepseek`. **Control:** `STRIX_DS41_PREFILL_ROUTER_V2`. **Default:** ON.

The prefill router's scores kernel takes 32 rows x 32 experts per block (a wave 8 rows x 16 experts) and stages x and the router weights 128 elements at a time, issuing the next chunk's 16-byte loads before the current chunk's products; blocks walk the experts first so the blocks of one row tile share its activations in the cache. Each lane still adds the elements l, l + 32, ... of every (row, expert) in ascending order from 0, then the same butterfly and router score: the decode router's chain.

- **Preconditions:** 16-byte aligned activations and router weights and hidden a multiple of 4 (the 16-byte staging, STRIX_DS41_PREFILL_ROUTER_WIDE); otherwise the v1 kernel. Hidden a multiple of 32 is required by every prefill router call.
- **Automatic selection:** On for every prefill router call that admits 16-byte staging.
- **Fallback notice:** None; ineligible views run the v1 kernel silently.
- **Exactness evidence:** Bitwise: prefill router scores, ids and weights of the v2 and v1 kernels equal to the decode router and to the host reference (synthetic 64 experts / 77 rows, 40 experts / hidden 1056 / 33 and 1 rows, real layer 20 at 300 and 8192 rows).
- **Off switch:** STRIX_DS41_PREFILL_ROUTER_V2=0
- **Measured gain:** Router kernel (layer 20 weights, 384 experts, scores + selection, interleaved): 20.71 -> 5.87 ms at 8192 rows (-72 %), 5.26 -> 1.52 ms at 2048, 0.58 -> 0.40 ms at 512. Two-node serve TTFT at 8,399 / 32,499 tokens (one binary, cache off, one load per arm, two GPU turns in opposite arm order): 8.876 / 31.852 s with STRIX_DS41_PREFILL_ROUTER_V2=0 vs 8.707 / 31.373 s default (-1.9 / -1.5 %); logits and generated ids identical.
- **Public evidence:** [Prefill router](../src/hip_ds41_prefill.hip)

<a id="ds-o-a8-fused"></a>

### Attention output A8 from the combine

**Engine:** `deepseek`. **Control:** `STRIX_DS41_PREFILL_O_A8_FUSED`. **Default:** ON.

The exact attention combine already forms the A8 operands of o (int8 codes, deltas and sums per 32 values, enqueue_quantize_a8's arithmetic) from the values it stores; the prefill passes its o operand buffers to the attention calls and skips the separate quantization pass before o_a.

- **Preconditions:** Exact attention (not the BF16 or FP16 WMMA attention) and A8 o_a operands (not the FP16 fast o_a).
- **Automatic selection:** On for every prefill chunk that meets the preconditions.
- **Fallback notice:** None; other chunks keep the separate pass.
- **Exactness evidence:** Bitwise: device attention tests compare the combine's A8 codes, deltas and sums with enqueue_quantize_a8 of the stored o (and a host model), also with an all-NaN head (deltas NaN in both); two-node serve logits and generated ids identical with the switch on and off.
- **Off switch:** STRIX_DS41_PREFILL_O_A8_FUSED=0
- **Measured gain:** Two-node serve TTFT at 8,399 / 32,499 tokens (one binary, cache off, one load per arm, default first): 8.883 / 31.251 s with STRIX_DS41_PREFILL_O_A8_FUSED=0 vs 8.767 / 30.886 s default (-1.3 / -1.2 %); logits and generated ids identical.
- **Public evidence:** [Prefill attention site](../src/hip_deepseek41_prefill.hip)

<a id="ds-fast-moe-down-rows"></a>

### FAST MoE down row groups

**Engine:** `deepseek`. **Control:** `STRIX_DS41_FAST_MOE_DOWN_ROWS`. **Default:** 32 (with `STRIX_DS41_FAST_MOE=1`).

The FAST relaxed MoE down kernel takes row groups of up to 32 rows of one expert instead of 64: half the accumulators and staged activations per thread, so it compiles without register spills. Each output's terms keep their order; only the grouping of rows into blocks changes.

- **Preconditions:** STRIX_DS41_FAST_MOE=1 (--profile fast) and route batches of at most 32 rows.
- **Automatic selection:** On whenever the FAST relaxed down kernel runs on batches of at most 32 rows.
- **Fallback notice:** None.
- **Exactness evidence:** The FAST relaxed MoE down outputs are bitwise identical for 32- and 64-row groups: rank-shaped bench (layer 20, 2048 / 4096 / 8192 rows) A8 codes, deltas, sums and raw down vectors; two-node serve logits and generated ids under --profile fast. The FAST identity itself is unchanged.
- **Off switch:** STRIX_DS41_FAST_MOE_DOWN_ROWS=64
- **Measured gain:** FAST relaxed down kernel -23 / -16 / -14 % at 2048 / 4096 / 8192 rows (32-row groups compile without the ~69 spilled VGPRs of 64-row groups). Two-node serve TTFT with STRIX_DS41_FAST_MOE=1 at 8,399 / 32,499 tokens (one binary, cache off, one load per arm, four GPU turns in alternating arm order): 7.863 / 27.509 s with STRIX_DS41_FAST_MOE_DOWN_ROWS=64 vs 7.533 / 27.077 s default (-4.2 / -1.6 %).
- **Public evidence:** [FAST relaxed MoE kernels](../src/hip_ds41_prefill_relaxed.inl)

<a id="ds-prefill-hc-fuse"></a>

### Prefill hyper-connection fusion

**Engine:** `deepseek`. **Control:** `STRIX_DS41_PREFILL_HC_FUSE`. **Default:** on.

Prefill moved the 4-copy FP32 hyper-connection stream through separate passes: the expansion after each sublayer, the next site's mixes, the collapse into a row buffer and the RMSNorm of that buffer. Now each expansion also computes the next site's mixes from the values it stores (the hyper-connection site kernel), and each site's RMSNorm reads the collapse straight from the stream. Every value is formed by the same operations in the same order.

- **Preconditions:** Device prefill path.
- **Automatic selection:** On for every prefill call.
- **Fallback notice:** None.
- **Exactness evidence:** Same kernels' arithmetic: the expanded stream, the next site's mixes, the collapsed and normalised rows and their A8 / FP16 copies are bitwise identical to the separate expansion, mix, collapse and RMSNorm passes. Two-node serve logits and generated ids identical at 1,999 / 8,399 / 32,499 tokens.
- **Off switch:** STRIX_DS41_PREFILL_HC_FUSE=0
- **Measured gain:** Removes a read of the 4-copy FP32 stream at each sublayer boundary (not across an Engram layer's inject or a change of the output rows) and the collapsed-row write + read at both sites. Two-node serve TTFT at 8,399 / 32,499 tokens (one binary, cache off, one load per arm, two GPU turns in alternating arm order): 8.238 / 29.481 s with STRIX_DS41_PREFILL_HC_FUSE=0 vs 8.009 / 28.000 s default (-2.8 / -5.0 %).
- **Public evidence:** [Prefill hyper-connection sites](../src/hip_deepseek41_prefill.hip)

<a id="ds-prefill-hc-collapse"></a>

### Prefill hyper-connection collapse with the expansion

**Engine:** `deepseek`. **Control:** `STRIX_DS41_PREFILL_HC_COLLAPSE`. **Default:** on (with `STRIX_DS41_PREFILL_HC_FUSE`).

The collapse that feeds a site's RMSNorm uses the mixes of the expansion just before it. The fused expansion therefore takes the same hidden chunk of all four copies in one block, keeps the collapse of the values it stores, and writes the collapsed rows; the site's RMSNorm then reads those rows instead of the 4-copy stream. Every value is formed by the same operations in the same order.

- **Preconditions:** Device prefill path with STRIX_DS41_PREFILL_HC_FUSE on; boundaries whose expansion is fused with the next mixes and whose next site starts at the same row.
- **Automatic selection:** On at every such boundary.
- **Fallback notice:** None.
- **Exactness evidence:** The collapsed rows are formed with the same operations in the same order as the separate collapse: device tests against the separate expansion, mix and collapse kernels (row tiles 1-4, incomplete tiles, signed zeros, subnormals, infinities and NaNs) are bitwise equal; two-node serve logits and generated ids identical at 1,999 / 8,399 / 32,499 tokens in every arm.
- **Off switch:** STRIX_DS41_PREFILL_HC_COLLAPSE=0
- **Measured gain:** The next site's RMSNorm reads the 20 KB collapsed row instead of the 80 KB 4-copy stream. Two-node serve TTFT at 8,399 / 32,499 tokens (one binary, cache off, one load per arm, four GPU turns in alternating arm order): 8.008 / 28.017 s with STRIX_DS41_PREFILL_HC_COLLAPSE=0 vs 7.996 / 27.809 s default (-0.2 / -0.7 %, 32,499 tokens faster in 3 of 4 turns).
- **Public evidence:** [Hyper-connection site kernels](../src/hip_ds41_layer.hip)

<a id="ds-fast-moe-order"></a>

### FAST MoE block order

**Engine:** `deepseek`. **Control:** `STRIX_DS41_FAST_MOE_ORDER`. **Default:** ON (with `STRIX_DS41_FAST_MOE=1`).

The FAST relaxed MoE kernels (gate/up and down, `--profile fast`) are launched with the output blocks fastest, so the output blocks of one row group run together and reuse its activations and its expert's weight slabs from the cache instead of re-reading them for every output block. Only the block order changes.

- **Preconditions:** STRIX_DS41_FAST_MOE=1 (--profile fast) and at most 65,535 route batches per call (else the batch-fastest order).
- **Automatic selection:** On whenever the FAST relaxed MoE kernels run.
- **Fallback notice:** None.
- **Exactness evidence:** The FAST relaxed MoE kernels' outputs are bitwise identical in both block orders: rank-shaped bench (layer 20, 2048 / 4096 / 8192 rows) A8 codes, deltas, sums and raw down vectors, and two-node serve logits and generated ids under --profile fast. The FAST identity itself is unchanged.
- **Off switch:** STRIX_DS41_FAST_MOE_ORDER=0
- **Measured gain:** Two-node serve TTFT with STRIX_DS41_FAST_MOE=1 at 8,399 / 32,499 tokens (one binary, cache off, one load per arm, two GPU turns in opposite arm order): 8.279 / 28.34 s with STRIX_DS41_FAST_MOE_ORDER=0 vs 8.107 / 27.65 s default (-2.1 / -2.4 %). Rank-shaped kernel bench: -2..-4 %.
- **Public evidence:** [FAST relaxed MoE kernels](../src/hip_ds41_prefill_relaxed.inl)

<a id="ds-dense"></a>

### Exact dense prefill selector

**Engine:** `deepseek`. **Control:** `STRIX_DS41_DENSE_SELECT`. **Default:** auto.

Enables q_b/o_a dense sites and selects an exact form per call without enabling approximate dense arithmetic.

- **Preconditions:** Exact dense prefill, supported site geometry; approximate FP16 dense and BF16 attention remain separate opt-ins.
- **Automatic selection:** On unless the value is exactly 0. At exact calls with v.tall=0 it requests the auto form; explicit nonzero form settings and approximate FP16 calls keep their configured paths.
- **Fallback notice:** No general decline log; unsupported exact shapes keep the compatible form. Optional FP16 requests without the build backend fail with FP16 dense prefill requires STRIX_DS41_PREFILL_FP16_LT.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms.
- **Off switch:** STRIX_DS41_DENSE_SELECT=0 restores loaded STRIX_DS41_DENSE_TALL / STRIX_DS41_PREFILL_DENSE_SITES configuration
- **Measured gain:** -
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md), [selector](../src/hip_deepseek41_prefill.hip)

<a id="ds-attention"></a>

### Attention fold, live chunks and F16 tiles

**Engine:** `deepseek`. **Control:** `STRIX_DS41_ATTN_WINDOW_FOLD; STRIX_DS41_ATTN_LIVE_CHUNKS; STRIX_DS41_ATTN_TILE_F16`. **Default:** auto.

Folds window state work, schedules live chunks and stores eligible attention tiles in half precision while preserving the qualified arithmetic.

- **Preconditions:** Valid attention ring/KV views and supported window rows; half tiles apply only at one or two heads per wave.
- **Automatic selection:** All three default on. Four-heads-per-wave kernels keep the non-half tile because the half form exceeds the register file.
- **Fallback notice:** Unsupported half geometry uses the non-half kernel silently. Invalid folded KV/ring geometry throws ds41 attention: window fold needs kv, kv_norm and ring_out == ring together, at most kMaxRows rows.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms.
- **Off switch:** STRIX_DS41_ATTN_WINDOW_FOLD=0, STRIX_DS41_ATTN_LIVE_CHUNKS=0 or STRIX_DS41_ATTN_TILE_F16=0, separately
- **Measured gain:** -
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-attn-swizzle"></a>

### Attention LDS swizzle

**Engine:** `deepseek`. **Control:** `STRIX_DS41_ATTN_LDS_SWIZZLE`. **Default:** on.

The f32 attention tiles (the 4-heads-per-wave tile and the f16 kernels' f32 fallback tiles) permute their 16-byte LDS vectors with an involutive map applied to stores and reads; the f16 tile choice is unchanged.

- **Preconditions:** Packed-attention HIP kernels without the window fold (the fold keeps the unpermuted kernels).
- **Automatic selection:** On for every eligible launch, single and segmented.
- **Fallback notice:** Folded launches use the unpermuted kernels silently.
- **Exactness evidence:** Bitwise: same values, dot products, softmax, reductions and chunk combination order; HIP attention device tests with the swizzle and the f16 tile off and on; two-node in-process ABA identical on both ranks (R1 tokens + logits slice, window tokens, attention state; batch N8 fed / argmax / logits per row, final attention states).
- **Off switch:** STRIX_DS41_ATTN_LDS_SWIZZLE=0
- **Measured gain:** Two-node in-process ABA, swizzle with the f16 tile vs the f16 tile alone: R6 -0.20 % (3/4 pairs faster), R1 neutral, batch N8 decode -0.04 % / -0.40 % in two turns.
- **Public evidence:** [Exact memory options](ds41-exact-memory-options.md), [two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-attn-lean"></a>

### Attention lean chunk kernel

**Engine:** `deepseek`. **Control:** `STRIX_DS41_ATTN_LEAN`. **Default:** 3.

The packed-attention chunk kernels run the same operations in the same order with fewer instructions. Level 1: the
softmax exponentials use selects instead of branches over the same polynomial, the running sum is taken in the PV loop
in the same key order, and the PV loop is split at key 8. Level 2 also dequantises the key tile through f16
bit patterns and one fused multiply per value, which rounds the same real product once; a key part with a NaN code or
a scale outside the exact range uses the previous code. Level 3 (default) also reads each lane's QK keys in a
lane-permuted order so the transposed butterfly needs no selects (every addition keeps its own + partner operand
order), runs the 4-heads-per-wave QK half tile as one fixed-register asm block in which every FMA pair dual-issues
(the same per-lane chains, generated by tools/gen_ds41_qk_asm.py), addresses each PV key once, skips the rescale when
the running max holds (its factor is exactly 1), and evaluates each (head, key) probability once from the quad
lane-0 floats the previous form used.

- **Preconditions:** Packed-attention HIP chunk kernels (prefill windows, decode, verify and batch segments).
- **Automatic selection:** On for every launch.
- **Fallback notice:** None; parts outside the exact range use the previous dequantisation silently.
- **Exactness evidence:** Bitwise: HIP attention device tests against the host reference with the default level, and levels 1, 2 and 3 byte-equal to level 0 on prefill-shaped calls (four streams, every E4M3/E2M1 code and scale byte, early positions, window-only and compressed layers, A8 combine, every tile choice).
- **Off switch:** STRIX_DS41_ATTN_LEAN=0 (STRIX_DS41_ATTN_LEAN=1 keeps the previous dequantisation, =2 the previous QK, softmax and PV)
- **Measured gain:** Level 2 (previous default): Two-node serve TTFT at 8,399 / 32,499 / 44,699 tokens (one binary, cache off, one load per arm, separate GPU turns): STRIX_DS41_ATTN_LEAN=0 9.99 / 35.20 / 50.29 s vs default 9.80 / 34.70 / 49.09 s (-1.9 / -1.4 / -2.4 %); prefill window bench -11.8 % (32 heads) / -13.5 % (64 heads) per 8-row window. Level 3 vs level 2, prefill window bench (8-row windows, 640 keys): -7.0 % (32 heads) / -9.2 % (64 heads), window-only -24 % (32 heads), 64-row windows -7 / -9 %.
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-warm"></a>

### Server warm-up and DSpark calibration

**Engine:** `deepseek`. **Control:** `STRIX_DS41_SERVE_WARM; STRIX_DS41_DSPARK_CALIB`. **Default:** auto.

Warms supported serving paths before READY and prices the speculative controller on actual device costs.

- **Preconditions:** Non-split serving with dummy KV slots available; DSpark calibration requires the drafter.
- **Automatic selection:** Warm-up is on; calibration follows warm-up outside split decode, with explicit =0/=1 overrides. Split decode skips local decode warm-up.
- **Fallback notice:** If implicit warm-up cannot reserve dummy slots it prints rank ...: warm-up skipped: the KV pool has no room for ... . Explicit SERVE_WARM=1 instead fails when they do not fit.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms. The warmed-server campaign compared all 64 request token sequences in each run.
- **Off switch:** STRIX_DS41_SERVE_WARM=0 or STRIX_DS41_DSPARK_CALIB=0, separately
- **Measured gain:** Combined two-node build: warmed C1 43.58 -> 48.00 tok/s (+9.9%); not isolated attribution
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-observation"></a>

### Bounded serving observation

**Engine:** `deepseek`. **Control:** `STRIX_SERVE_OBSERVE_EXPIRY_S`. **Default:** 60 seconds.

Expire an armed detailed serving observation after a bounded interval.

- **Preconditions:** Serve app with an observation arming/renewal request.
- **Automatic selection:** An arming request starts the 60-second lease; metrics and detailed status requests renew it.
- **Fallback notice:** Expiry stops the observation; it does not change inference dispatch or model state.
- **Exactness evidence:** Observation lifetime changes no arithmetic or committed tokens; inference reference gates remain required.
- **Off switch:** STRIX_SERVE_OBSERVE_EXPIRY_S=0 keeps observation armed until exit
- **Measured gain:** No isolated measurement

### Product profiles (pair)

`--profile default|fast|quality` selects the product policy of the two-node DeepSeek apps
(`strix-deepseek41-rank`, `-batch`, `-prefill`, `-serve`); both ranks must run the same profile (checked at startup: a mismatch stops both ranks with one message naming both profiles).
It resolves before the [production defaults](models/deepseek-v4.1/production-defaults.md), one row per option in the
[profile table](../include/strix/deepseek41_pair_profile_table.hpp). `default` exports nothing: the resolution, the
arguments and every exact default are those of the plain invocation. `fast` applies every ready row; its two
bit-changing options (FP16 sparse attention, relaxed MoE epilogue) are pending, so the builtin keeps running, each
appears with its target in `profile.pending` and prints one stderr line. `quality` has no quality pack: it runs
default settings and says so. Pending rows never refuse.

Explicit environment variables override the profile and are recorded as explicit or off. `--allow-pending` /
`STRIX_DS41_PROFILE_ALLOW_PENDING=1` applies pending rows whose code is in the build (none today), stamped
`unqualified: <rows>`. `--print-effective-config` prints `profile ...` lines right after the `app` line; every app
prints a `profile {...}` receipt line on stderr; the serve also reports it in its capabilities (`profile`).

## DeepSeek RTX PRO standalone

Use the [standalone run command and required fixtures](models/deepseek-v4.1/rtx-standalone.md#run-and-persistent-worker).
`strix-cuda-ds41-replica MODEL.gguf --print-config` inspects resolved JSON before CUDA, model or fixture reads;
small prior/pruning/residency files may still be read. Defaults target the standalone Q2 GGUF/Q8_0 identity.
`--baseline` restores reference forms at that argument; later explicit flags win. `STRIX_DS41_RTX_BASELINE=1`
applies before all argv. Baseline also restores 96 host spares, 4 pressure spares, inferred row capacity,
run controller scope, full draft vocabulary, serial load unless explicitly set, and the old arena rule.
Normal defaults use 48 host spares, 8 pressure spares, 6 rows; arena is 74.8 GiB with keep-64, 76.5 with keep-32,
71.5 with full drafter, 80 without. Explicit arena size wins; host allocation is capped at 68 GiB with a 6 GiB margin.

Every RTX item uses the replica's exact0/pressure/exact8/wforce0/spec0/exact0 and plain/speculative/plain chat gates:
at least 64 compared steps, matched == compared, gate pass, canonical position/state digest equal, complete chat
vectors equal; pressure must exercise bank/resurrection/mid-step collection. Pruning/controller/vocabulary can
change acceptance, not committed target-greedy tokens. Compare within one load where supported (load spread ~10%).

<a id="rtx-kernels"></a>

### Decode kernel bundle

**Engine:** `deepseek-rtx`. **Control:** `--dsrtxk`. **Default:** ON.

Router, projection/fork, MoE, attention and norm decode kernels.

- **Preconditions:** Admitted Q2/Q8 standalone model and supported CUDA shapes.
- **Automatic selection:** Default bundle on; explicit member flags can override.
- **Fallback notice:** Supported shapes use optimized forms; incompatible explicit requests retain validation.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --dsrtxk-off
- **Measured gain:** All-hit 12.62 -> 10.50 ms/token; full loop 5.8-10.8%
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-fusions"></a>

### HC and shared-expert fusions

**Engine:** `deepseek-rtx`. **Control:** `--fuse`. **Default:** ON.

Fuse compatible HC/shared-expert work.

- **Preconditions:** Supported standalone sites and shapes.
- **Automatic selection:** On independently of decode bundle.
- **Fallback notice:** Unsupported shapes retain supported exact dispatch; invalid explicit combinations fail.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --fuse-off
- **Measured gain:** Included in decode qualification; not isolated
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-records"></a>

### Deferred step records

**Engine:** `deepseek-rtx`. **Control:** `--records-deferred`. **Default:** ON.

Move step records off the critical path.

- **Preconditions:** Replica step/qualification recording.
- **Automatic selection:** Deferred unless explicitly inline.
- **Fallback notice:** No refusal notice; inline is an explicit order choice.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --records-inline
- **Measured gain:** Qualified; no isolated measurement
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-router"></a>

### Router form 2

**Engine:** `deepseek-rtx`. **Control:** `--router-pf`. **Default:** ON.

Use form 2 router.

- **Preconditions:** Supported router rows 1-8.
- **Automatic selection:** Default form 2 in decode configuration.
- **Fallback notice:** Shape dispatch keeps a supported exact form; no general decline notice.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --router-pf0
- **Measured gain:** R1 17.3 -> 15.4 us; eight-layer device time -0.5%
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-qakv"></a>

### QA/KV split 1

**Engine:** `deepseek-rtx`. **Control:** `--qakv-s1`. **Default:** ON.

Use split-1 query/KV scheduling.

- **Preconditions:** Supported query/KV geometry.
- **Automatic selection:** Default split 1.
- **Fallback notice:** Supported exact dispatch retained outside specialized geometry.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --qakv-auto
- **Measured gain:** Eight-layer device time -1.54 to -1.94%
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-q8"></a>

### Compact Q8 R1

**Engine:** `deepseek-rtx`. **Control:** `--q8-r1`. **Default:** ON.

Use compact single-row kernel without padding.

- **Preconditions:** Supported Q8 single-row shapes.
- **Automatic selection:** Use compact form when admitted.
- **Fallback notice:** One-time notice for shapes requiring the generic exact Q8 kernel.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --q8-r1-off; --q8-r1-pad selects padded form
- **Measured gain:** Eight-layer device time -0.80 / -0.53%
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-pruning"></a>

### Keep-64 drafter pruning

**Engine:** `deepseek-rtx`. **Control:** `--draft-keep 64`. **Default:** ON.

Keep most-used 64 experts per stage.

- **Preconditions:** DSpark drafter with built-in expert ranking.
- **Automatic selection:** Default keep count 64.
- **Fallback notice:** Absent drafter makes pruning irrelevant; explicit unsupported configurations fail validation.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --draft-keep 0
- **Measured gain:** Code +16% with a four-row window
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-controller"></a>

### Warm controller prior, costs and scope

**Engine:** `deepseek-rtx`. **Control:** `--controller-prior; --controller-costs; --controller-scope`. **Default:** warm, chat scope.

Seed built-in survival prior/costs; disable exploration and cost learning.

- **Preconditions:** Speculative phase with compatible drafter.
- **Automatic selection:** Built-in keep-64 prior and seeded costs, reset at chat scope.
- **Fallback notice:** Explicit none removes prior/costs; no universal decline line.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --controller-prior none removes prior and implicit costs; --controller-costs none removes costs alone; --controller-scope run restores run scope
- **Measured gain:** Qualified; no isolated measurement
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-writeback"></a>

### Deferred write-backs

**Engine:** `deepseek-rtx`. **Control:** `--wb-defer`. **Default:** ON.

Schedule bounded write-backs away from demand work.

- **Preconditions:** Exclusive VRAM/pinned-host tierstream cache and spare capacity.
- **Automatic selection:** On in normal replica defaults.
- **Fallback notice:** Cache pressure follows demand/collection protocol; setup errors fail.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --no-wb-defer
- **Measured gain:** Qualified; no isolated measurement
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-engram"></a>

### Host Engram record cache

**Engine:** `deepseek-rtx`. **Control:** `--engram-cache-mb 512`. **Default:** ON.

Retain identical record bytes in bounded host RAM.

- **Preconditions:** 512 MiB host cache capacity.
- **Automatic selection:** Cache hits reuse bytes; misses read backing storage.
- **Fallback notice:** Misses read normally without a decline notice.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --engram-cache-mb 0
- **Measured gain:** Avoids repeated storage stalls; no isolated throughput measurement
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-load"></a>

### Pipelined cold load

**Engine:** `deepseek-rtx`. **Control:** `DG_LOAD_SERIAL`. **Default:** ON (unset).

Use tierstream cold-load pipeline.

- **Preconditions:** Supported tierstream loader and model storage.
- **Automatic selection:** Pipeline unless serial environment override.
- **Fallback notice:** Loader errors propagate; no generic retry promise.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** DG_LOAD_SERIAL=1
- **Measured gain:** Full loading 108.7 -> 61.8 s in the published workload
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-warmup"></a>

### Graph warm-up

**Engine:** `deepseek-rtx`. **Control:** `--warmup-graphs`. **Default:** auto (0,8 in serve).

Capture common graphs before readiness.

- **Preconditions:** Persistent serve worker and supported graph sizes.
- **Automatic selection:** Serve warms 0,8; non-serve does not unless explicit.
- **Fallback notice:** Unsupported graph/config requests retain validation; no general refusal notice.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --warmup-graphs none
- **Measured gain:** First 16 steps 56.05 -> 33.99 ms/step
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-residency"></a>

### Saved expert residency

**Engine:** `deepseek-rtx`. **Control:** `--residency-seed; --residency-save`. **Default:** auto.

Reuse valid placement and save on worker exit.

- **Preconditions:** Serve mode, compatible seed identity and writable destination for saves.
- **Automatic selection:** MODEL.gguf.residency unless STRIX_DS41_RESIDENCY overrides it; non-serve needs explicit save/warm-up requests.
- **Fallback notice:** Missing auto seed quietly uses route order; invalid auto seed gives one notice and falls back. Explicit seed errors are strict. Save uses temporary file + rename.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --residency-seed none; --residency-save none
- **Measured gain:** Step 0 332.6 -> 156.0 ms; first 16 steps -20%
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-vocab"></a>

### Automatic draft vocabulary

**Engine:** `deepseek-rtx`. **Control:** `--draft-vocab`. **Default:** auto.

Select vocabulary directory for the drafter.

- **Preconditions:** STRIX_DS41_DRAFT_VOCAB directory or ds41-draft-vocab beside drafter.
- **Automatic selection:** auto:DIR if a directory is available; explicit flag wins.
- **Fallback notice:** Absent directory uses full vocabulary with one line.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --draft-vocab full
- **Measured gain:** No isolated measurement
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-evict-auto"></a>

### Automatic scored eviction

**Engine:** `deepseek-rtx`. **Control:** `--evict lru|scored|auto`. **Default:** auto.

After a DSpark verify window, evict experts by the tierstream's learned score; after one-row plain steps keep the LRU walk.

- **Preconditions:** Standalone RTX replica with a tierstream that provides the auto policy (SS_DG_AUTO_EVICT); an older tierstream keeps lru with a notice.
- **Automatic selection:** auto when the linked tierstream has the policy; explicit `--evict` wins; `--baseline` = lru.
- **Fallback notice:** Older tierstream: one notice, lru.
- **Exactness evidence:** Only residency changes: chat token vectors identical across all 8 arms (lru/auto, forced-window and pressure gates).
- **Off switch:** --evict lru
- **Measured gain:** DSpark misses/token 10.55 -> 9.08 (-14 %), median +6.4 % tok/s (7 chats); plain decode neutral
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-wb-stream"></a>

### Write-back stream

**Engine:** `deepseek-rtx`. **Control:** `--wb-stream` / `--no-wb-stream`, `STRIX_DS41_WB_STREAM=0|1`. **Default:** ON.

Deferred expert write-backs go out on tierstream's dedicated D2H stream (`dg_wb_stream`) instead of the demand stream, so they overlap the next layer's demand H2D copies. Same copies, same order, same destinations.

- **Preconditions:** Standalone RTX replica with a tierstream that has `dg_wb_stream`; load-time option.
- **Exactness evidence:** Transport only: chat token vectors identical across 4 alternating loads (off/on/on/off).
- **Off switch:** `--no-wb-stream` (or `STRIX_DS41_WB_STREAM=0`; `--baseline` turns it off)
- **Measured gain:** DSpark per commit: plain 14.37 to 14.07 ms (-2.0 %), controller 12.24 to 11.45 ms (-6.4 %); copy bench duplex 43.3 to 44.8 GB/s.
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-ce-pacing"></a>

### Copy-engine write-back pacing

**Engine:** `deepseek-rtx`. **Control:** `--ce-pacing` / `--no-ce-pacing`, `STRIX_DS41_CE_PACING=0|1`. **Default:** ON.

Bound the background write-back bytes queued on the single WDDM copy engine to about one expert, so a demand fetch for the next layer never waits behind a long write-back queue. Demand and speculative fetches keep priority; unissued victims can still be cancelled.

- **Preconditions:** Standalone RTX replica; load-time option (no per-request switch).
- **Exactness evidence:** Only copy timing changes: chat token vectors identical across 4 alternating loads (base/on/base/on) and the exact, forced-window and pressure gates.
- **Off switch:** --no-ce-pacing (or STRIX_DS41_CE_PACING=0; --baseline turns it off)
- **Measured gain:** Neutral to small: adjacent loads DSpark +2.7 %, plain -0.5 %, full request wall -1.1 %; bracketed comparison within noise.
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-prefill-host-tier"></a>

### Prefill pinned host expert tier (opt-in)

**Engine:** `deepseek-rtx`. **Control:** `--host-gib-auto G; --overlay OVERLAY.gguf; --host-pin-order layer|deadline|early`. **Default:** off (file-only streaming).

Keep up to G GiB of prefill experts in pinned host memory when it is available, instead of streaming every layer from file.

- **Preconditions:** DeepSeek CUDA prefill service on the RTX PRO with free pinned host memory; mandatory allocations are made first, each layer gets one non-waiting admission, demoted layers stream from file with the same bits.
- **Automatic selection:** None yet: opt-in until coexistence with the other services on the same GPU is qualified.
- **Fallback notice:** Layers that do not get a pinned slot stream from file; `/health` reports host-tier accounting.
- **Exactness evidence:** Exact: prefill state sha256, logits FNV and next token identical to file-only streaming on every exercised path (implicit and explicit tier, export modes, pipeline, whole-layer provider, forced demotion).
- **Off switch:** omit `--host-gib-auto`
- **Measured gain:** TTFT 1K -24..-30 % (16.2 -> 12.4 s interleaved), 8K -28 %, 30K -12 %
- **Implementation / evidence:** [host tier](models/deepseek-v4.1/rtx-prefill-host-tier.md)

### Lean decode state

**Engine:** `deepseek-rtx`. **Control:** `--lean-state`. **Default:** ON.

Keep only the decode-time attention state: an allocation ledger and a split at the maximum decode position replace the full prefill-sized buffers, so the freed VRAM holds more experts.

- **Preconditions:** Standalone RTX replica or CUDA decode tool.
- **Automatic selection:** Always on unless switched off; `--baseline` turns it off.
- **Fallback notice:** None; the state layout is fixed at load.
- **Exactness evidence:** Bitwise against the full state at 2K, 8K, 32K and 128K: prefill states and stream hashes, 64-step greedy logits and tokens, DSpark commit windows, replica gates, chat tokens and persistent-worker requests.
- **Off switch:** --no-lean-state
- **Measured gain:** 8K state 673 to 25 MiB; 32K 2,486 to 59 MiB (+264 expert slots, plain decode +3.0 %); 128K loads with a 195 MiB state (without it the replica does not fit in host memory).
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

## MiniMax

```sh
strix-minimax-rank MODEL_DIR R usb4-auto OUTPUT_DIR
```

Launch both ranks with the deployment's transport/assets and requested workload, or use
`strix-minimax-serve MODEL_DIR --rank R --transport usb4-auto`. Inspect either command with
`--print-effective-config` before model/GPU work. The plain resolver supplies 8 slots, capacity 8256 and
max rows 4096 unless explicitly overridden. `STRIX_MM_BASELINE=1` puts all 27 resolver items below at their off values;
an explicit per-item environment value wins. It does not undo older built-in attention/index/MoE routes.
The resolver applies once before cached readers. Build-time scalar gu4 addressing is also on by default;
`STRIX_MINIMAX_GU4_SCALAR=OFF` selects its previous reader at build time.

The default exchange routes are controlled separately by `STRIX_MM_XFER`, `STRIX_MM_PFX` and
`STRIX_MM_PF_INTERLEAVE`; the resolver prints their effective selection. Preserve topology and memory budget
when comparing items. Built-in `--gate`/`--gold`, `--decode-check`, `--verify-trace` and `--xb-check` compare
golden logits, sequential/verify rows and exchange state. Speculative sessions require identical committed
token IDs to plain sessions. Use transport corruption checks as well as model gates for CRC changes.

<a id="mm-moe-wide"></a>

### Wide expert decode

**Engine:** `minimax`. **Control:** `STRIX_MMX_MOE_WIDE`. **Default:** 2.

Use wider expert kernels for N2-4.

- **Preconditions:** Supported MoE widths, quantization and N2-4 rows.
- **Automatic selection:** Resolver supplies 2 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Measurement: rows == alone, wide expert kernels at N2-4, N2 -3.0..-3.5 %, N4 -4..-6 %; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MMX_MOE_WIDE=0
- **Measured gain:** rows == alone, wide expert kernels at N2-4, N2 -3.0..-3.5 %, N4 -4..-6 %
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-index-conc"></a>

### Concurrent indexer

**Engine:** `minimax`. **Control:** `STRIX_MM_INDEX_CONC`. **Default:** 1.

Overlap independent index work.

- **Preconditions:** Decode indexer with valid stream/buffer ownership.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Measurement: ids identical, 17/18 in-process block pairs faster, ~-0.2 ms/token; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_INDEX_CONC=0
- **Measured gain:** ids identical, 17/18 in-process block pairs faster, ~-0.2 ms/token
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-xfer-crc-deferred"></a>

### Deferred exchange CRC

**Engine:** `minimax`. **Control:** `STRIX_MMX_CRC`. **Default:** deferred.

Check transport CRC through deferred exchange completion.

- **Preconditions:** Both ranks on compatible deferred protocol; corruption checks remain enabled.
- **Automatic selection:** Resolver supplies deferred only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Protocol errors/corruption still fail; the off value 1 means inline CRC, not disabled CRC.
- **Exactness evidence:** Published same-identity qualification: Measurement: tokens identical in 7 processes, each deferred arm below its inline neighbours by 0.04-0.12 ms/token; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MMX_CRC=1
- **Measured gain:** tokens identical in 7 processes, each deferred arm below its inline neighbours by 0.04-0.12 ms/token
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-xb-wait"></a>

### Exchange boundary wait

**Engine:** `minimax`. **Control:** `STRIX_MM_XB_WAIT`. **Default:** 1.

Use fused boundary wait scheduling.

- **Preconditions:** Compatible decode Xfer exchange boundary.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_XB_WAIT=0
- **Measured gain:** Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-xb-inplace"></a>

### In-place exchange boundary

**Engine:** `minimax`. **Control:** `STRIX_MM_XB_INPLACE`. **Default:** 1.

Consume exchange results in place.

- **Preconditions:** Compatible decode Xfer buffers and ownership.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_XB_INPLACE=0
- **Measured gain:** Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-xb-publish"></a>

### Producer exchange publish

**Engine:** `minimax`. **Control:** `STRIX_MM_XB_PUBLISH`. **Default:** 1.

Publish the exchange message from the producer.

- **Preconditions:** In-place boundary plus v4 linear core and decode MoE overlap form.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Defaulted publish logs that the v4 linear/overlap core is required and switches off; explicit =1 fails when unavailable.
- **Exactness evidence:** Published same-identity qualification: Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token (falls back without the v4/overlap cores); repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_XB_PUBLISH=0
- **Measured gain:** Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token (falls back without the v4/overlap cores)
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-xb-postilp"></a>

### Boundary post ILP

**Engine:** `minimax`. **Control:** `STRIX_MM_XB_POSTILP`. **Default:** 2.

Use LDS-staged mode-2 post-exchange partials.

- **Preconditions:** Supported decode boundary shapes/LDS capacity.
- **Automatic selection:** Resolver supplies 2 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_XB_POSTILP=0
- **Measured gain:** Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-xb-warm"></a>

### Boundary warm reads

**Engine:** `minimax`. **Control:** `STRIX_MM_XB_WARM`. **Default:** 8.

Warm 8 MiB of boundary data.

- **Preconditions:** Decode exchange boundary and supported memory region.
- **Automatic selection:** Resolver supplies 8 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_XB_WARM=0
- **Measured gain:** Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-xb-warm-o"></a>

### o_proj warm during decode attention

**Engine:** `minimax`. **Control:** `STRIX_MM_XB_WARM_O`. **Default:** 8.

The decode attention kernels of a layer leave memory nearly idle; a side stream reads the first 8 MiB of the layer's
o_proj weights into the memory-side cache meanwhile (gated on the end of the q/k/v and index projections).

- **Preconditions:** Decode exchange module path with the optimized linears (two-rank decode steps).
- **Automatic selection:** Resolver supplies 8 only when unset; explicit values win; 0 disables.
- **Exactness evidence:** Cache state only; decode logits bitwise on both ranks vs off at N1/N4/N8 and a 4-row verify window; in-process matched replay identical.
- **Off switch:** STRIX_MM_XB_WARM_O=0
- **Measured gain:** C1 8K two-node decode 44.64-44.66 -> 44.01-44.07 ms/token (-1.3 %); 14 MiB is slower.
- **Implementation / evidence:** [resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp) (`--ab xbwarmo`, `--xb-check` cfg `o`)

<a id="mm-moe-tight-grid"></a>

### Tight decode MoE grids

**Engine:** `minimax`. **Control:** `STRIX_MMX_MOE_TIGHT_GRID`. **Default:** 1.

The routed gate/up and down launches size their slot dimension to min(32, 4N) + 1 rows instead of 33; the dropped
rows returned before any work, so at N = 1 about 10,000 empty CTAs per launch are no longer dispatched.

- **Preconditions:** v4 decode MoE launches (N = 1..8).
- **Automatic selection:** Resolver supplies 1 only when unset; read per call.
- **Exactness evidence:** Decode logits bitwise on both ranks vs off at N1/N4/N8 and a 4-row verify window; in-process matched replay identical.
- **Off switch:** STRIX_MMX_MOE_TIGHT_GRID=0
- **Measured gain:** C1 8K two-node decode 45.12-45.18 -> 44.60-44.68 ms/token (-1.2 %).
- **Implementation / evidence:** [MoE launches](../src/minimax/moe/mmx_moe.hip), [rank gates](../apps/minimax_rank_main.cpp) (`--ab moegrid`, `--xb-check` cfg `g`)

<a id="mm-moe-pipe"></a>

### Decode MoE routed gate/up and down in one launch

**Engine:** `minimax`. **Control:** `STRIX_MMX_MOE_PIPE`. **Default:** 2 via the engine resolver; raw-library unset is 0.

After the router + shared-expert launch, the routed gate/up and down run as one launch. CTAs take arrival tickets, so
each waits only on lower tickets that are already running; a down CTA starts as soon as the three gate/up blocks of its
input split are finished (per-block ready word carrying the route's epoch) instead of after the whole gate/up launch.
Form 2 runs the shared expert's down first (its inputs come from the previous launch); 1 runs it last.

- **Preconditions:** v4 decode MoE overlap form, calls of 2-4 rows (DSpark verify windows, C2-C4 steps).
- **Automatic selection:** Resolver supplies 2 only when unset; read per call. It replaces the four-row down split. N1 keeps two launches (measured 0.1 ms/token faster); 5-8 rows keep them too (the merged 8-row kernel needs ~150 VGPRs vs 85 apart and loses 15-19 %).
- **Exactness evidence:** Decode logits bitwise on both ranks vs off at N1/N2/N4 and a 4-row verify window; in-process matched replay identical; MoE bench --test (incl. a seeded epoch wrap with stale ready words) and --diverse --test bitwise.
- **Off switch:** STRIX_MMX_MOE_PIPE=0
- **Measured gain:** Two-node decode, in-process blocks: N2 58.9-59.3 -> 57.3-57.7 ms/step (-2.7 %), N4 81.8-82.6 -> 80.6-80.8 ms/step (-1.6 %).
- **Implementation / evidence:** [MoE launches](../src/minimax/moe/mmx_moe.hip), [rank gates](../apps/minimax_rank_main.cpp) (`--ab moepipe`, `--xb-check` cfg `f`)

<a id="mm-serve-lookup"></a>

### Serve prompt lookup

**Engine:** `minimax`. **Control:** `STRIX_MM_LOOKUP` (startup), `/admin/sched {"lookup": bool}` (runtime). **Default:** on.

Rank 0 keeps a prompt-lookup index (`include/strix/lookup/index.hpp`) per request over its prompt and committed tokens.
Each cycle's longest earlier match is scored against what the request then commits (per match length and depth). A
request drafts its window from the match where the expected tokens per second of the whole cycle beat its alternative
(its measured DSpark windows while DSpark may draft, else a plain row); DSpark plans the other requests in the rows left.

- **Exactness evidence:** draft source only, greedy verification decides every token; generated token IDs identical with lookup on and off in every measured arm.
- **Off switch:** STRIX_MM_LOOKUP=0; /admin/sched lookup false; STRIX_MM_BASELINE=1
- **Measured gain:** Two-node serve, one load, lookup switched per request batch: copy-heavy rewrite (C1, DSpark off) 23.65 / 23.44 -> 26.01 / 25.69 tok/s (+9.8 %); agent edit / agent JSON / prose at C1 and four concurrent requests equal within noise (DSpark keeps drafting where it is admitted; no lookup window at C4 on these prompts).
- **Counters:** `/admin/sched` counters lookup_windows, lookup_drafted, lookup_accepted, lookup_s.

<a id="mm-serve-dspark"></a>

### DSpark automatic admission

**Engine:** `minimax`. **Control:** `STRIX_MM_SERVE_DSPARK`; `--dspark-mode`. **Default:** auto.

The frozen default prior `structural-v1` promotes eligible C1 JSON/code markers and prompts
of at least 6551 tokens into the legacy automatic controller. Other prose stays plain without
neural probes. `STRIX_MM_DSPARK_PRIOR=unqualified-v1` or `--dspark-prior unqualified-v1`
restores conservative plain admission. Thresholds and route reasons are documented in the
qualification contract. Serve node qualification under the pre-registered not-slower rule
(same load, mode switched per request): JSON -34.6 %, code -17.5 % wall time, Spanish prose and
English story plain and not slower, identical generated tokens in every class; long prompts on
serve stay plain (above max_rows).

- **Preconditions:** Local compatible assets, C1, no queued competitor or competing prefill, room for the block. Grouped trailing taps remain unqualified; grouped serve prompts stay plain. No request-local net-saving exit exists yet; promoted requests retain legacy control.
- **Compatibility:** `1|on` selects the legacy serve controller; rank `auto:legacy`, `auto:nogate` and `fixed:K` remain explicit. Serve still uses full Markov; candidate resolver items apply to rank neural policies.
- **Off switch:** `STRIX_MM_SERVE_DSPARK=0`, `--dspark-mode off`, `/admin/sched` `dspark:false` or `dspark_mode:"off"`. Baseline supplies 0 unless overridden. `STRIX_MM_SERVE_DSPARK_INFLIGHT=0` prevents legacy context admission; the cap otherwise defaults to 1.
- **Qualification:** [Contracts and paired protocol](engine/serving/dspark-auto-qualification.md). A paired 95% interval wholly slower rejects a selected regime; a straddling interval is not proof of equivalence. No minimum percentage gain veto.
- **Implementation:** [Resolver](../src/minimax/engine/mm_defaults.cpp), [host admission](../src/minimax/dspark/minimax_auto.hpp), [CPU tests](../tests/minimax_auto_tests.cpp).

<a id="mm-dspark-attn"></a>

### DSpark attention

**Engine:** `minimax`. **Control:** `STRIX_MM_DSPARK_ATTN`. **Default:** 2.

Use mode-2 drafter attention.

- **Preconditions:** Attached compatible DSpark drafter and attention geometry.
- **Automatic selection:** Resolver supplies 2 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Plain fallback uses the unchanged target schedule; promoted structural-v1 cells use legacy greedy verification; serve A/B: generated token IDs (ids_fnv) and text identical with auto and off for JSON, code, Spanish prose and English story.
- **Off switch:** STRIX_MM_DSPARK_ATTN=0
- **Measured gain:** Drafter measurement: bitwise, drafter attention B/A 0.964-0.975 / 0.918-0.924 / 0.815-0.819 at ctx 300/1000/4095
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-draft-lin"></a>

### Grouped draft linears

**Engine:** `minimax`. **Control:** `STRIX_MM_DRAFT_LIN`. **Default:** opt.

Group q/k/v and gate/up draft projections.

- **Preconditions:** Attached EXL3 draft linears with supported shapes.
- **Automatic selection:** Resolver supplies opt only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Optimized draft linears (measurement evidence); repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_DRAFT_LIN=lib
- **Measured gain:** Optimized draft linears (measurement evidence)
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-linear-v11"></a>

### Draft-block linear core v11

**Engine:** `minimax`. **Control:** `STRIX_MMX_LINEAR_V11`. **Default:** 1.

A linear core scheduled for the 7-row DSpark draft block (q+k+v, o and gate+up groups).

- **Preconditions:** DSpark draft block at 7 rows through the scheduled linear path; other callers keep their forms.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win.
- **Fallback notice:** Shapes outside the schedule keep the previous core.
- **Exactness evidence:** Same output bits as v10 (bitwise over every scheduled form and layer); speculative ids == greedy 36/36.
- **Off switch:** STRIX_MMX_LINEAR_V11=0
- **Measured gain:** DSpark draft block -11.9 %
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-markov-cand"></a>

### DSpark Markov candidate form

**Engine:** `minimax`. **Control:** `STRIX_MM_MARKOV; STRIX_MM_MARKOV_C; STRIX_MM_MARKOV_H; STRIX_MM_MARKOV_TOPC`. **Default:** cand, C 4096, H 8192, top-C v2.

The drafter's Markov chain runs over a candidate set (exact top-C per row) with one device-to-device payload exchange
per draft instead of a full-vocabulary scan and host exchange per step.

- **Preconditions:** Attached DSpark drafter on the pair.
- **Automatic selection:** Resolver supplies the four values only when unset; explicit values win.
- **Fallback notice:** A draft whose candidate set cannot cover the chain falls back to the full chain for that draft.
- **Exactness evidence:** Target verify is unchanged, so committed tokens equal plain greedy (ids == greedy on every gated session); top-C v2 payloads == v1 == CPU, including ties, +/-0 and NaN.
- **Off switch:** STRIX_MM_MARKOV=full (C/H/TOPC: 1024/2048/v1)
- **Measured gain:** ~1.7 ms saved per draft; DSpark median tok/s +1.1 to +1.6 % across json/code/English/Spanish prompts, acceptance unchanged
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-pf-core-v4"></a>

### v4 MoE prefill core

**Engine:** `minimax`. **Control:** `STRIX_MMX_PREFILL_CORE`. **Default:** v4.

Use v4 register/layout scheduling.

- **Preconditions:** Supported prefill expert shapes and quantized weights.
- **Automatic selection:** Resolver supplies v4 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Prefill core measurement: bitwise everywhere, whole call 0.34-0.58x lib; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MMX_PREFILL_CORE=lib
- **Measured gain:** Prefill core measurement: bitwise everywhere, whole call 0.34-0.58x lib
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-pf-msa-batch"></a>

### MSA prefill batching

**Engine:** `minimax`. **Control:** `STRIX_MM_PF_MSA_BATCH`. **Default:** 1.

Batch compatible MSA work across prompt rows.

- **Preconditions:** Prefill rows with supported MSA geometry.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Prefill measurement: exact, combined prefill TTFT 1K -17 %, 8K -20 %; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_PF_MSA_BATCH=0
- **Measured gain:** Prefill measurement: exact, combined prefill TTFT 1K -17 %, 8K -20 %
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-pf-attend4"></a>

### Four-way prefill attend

**Engine:** `minimax`. **Control:** `STRIX_MM_PF_ATTEND4`. **Default:** 1.

Use the admitted four-way attend schedule.

- **Preconditions:** Supported prefill attention groups and row layout.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Prefill measurement: exact, combined prefill TTFT 1K -17 %, 8K -20 %; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_PF_ATTEND4=0
- **Measured gain:** Prefill measurement: exact, combined prefill TTFT 1K -17 %, 8K -20 %
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-pf-router-batch"></a>

### Batched prefill router

**Engine:** `minimax`. **Control:** `STRIX_MM_PF_ROUTER_BATCH`. **Default:** 1.

Compute compatible router rows together.

- **Preconditions:** Supported router row blocks.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Prefill measurement: exact, combined prefill TTFT 1K -17 %, 8K -20 %; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_PF_ROUTER_BATCH=0
- **Measured gain:** Prefill measurement: exact, combined prefill TTFT 1K -17 %, 8K -20 %
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-pf-mtfit"></a>

### MoE tile fit

**Engine:** `minimax`. **Control:** `STRIX_MMX_PREFILL_MTFIT`. **Default:** 1.

Choose fitting prefill MoE tiles.

- **Preconditions:** Supported prefill core and expert row geometry.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Prefill measurement: exact, combined prefill TTFT 1K -17 %, 8K -20 %; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MMX_PREFILL_MTFIT=0
- **Measured gain:** Prefill measurement: exact, combined prefill TTFT 1K -17 %, 8K -20 %
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-pf-msa-rework"></a>

### MSA prefill mode 5

**Engine:** `minimax`. **Control:** `STRIX_MM_PF_MSA_REWORK`. **Default:** 6 (mode 5 below is the previous default).

Rework attention scheduling without changing the reference result.

- **Preconditions:** Supported MSA prefill shapes and row buffers.
- **Automatic selection:** Resolver supplies 5 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Mode 5 vs 3: prefill tokens, attention state fnv and final-row fnv identical across all arms on both ranks; kernel tests bitwise for every rework mode. Mode 3: published same-identity qualification (8K B/A 0.963 / 0.957 / 0.972).
- **Off switch:** STRIX_MM_PF_MSA_REWORK=3 (previous default), =0 (original)
- **Measured gain:** Mode 5 (= mode 3 with packed two-head probability evaluation) vs mode 3, GPU, two Strix Halo nodes over 2x USB4 Gen3, PerfOpt kernel, one binary, rank --ab pfmsarework (A=3, B=5; A B A B A B A prefills) in two processes: 8K TTFT B - bracketing A mean -0.129 s, 1K -0.109 s (n=6 each, intervals include 0, not slower); producers -2.9 %, attention -1 % in the kernel bench. Mode 3 vs the original: 8K B/A 0.963 / 0.957 / 0.972
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-pf-gather-tok"></a>

### Prefill MoE token-major gather

**Engine:** `minimax`. **Control:** `STRIX_MMX_PREFILL_GATHER_TOK`. **Default:** 1.

Read each token's input row once for all of its routed experts and the shared expert, instead of once per expert and matrix.

- **Preconditions:** v4 prefill MoE core (the default).
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win.
- **Fallback notice:** None needed: every row count uses the same kernel; =0 restores the per-pair gather.
- **Exactness evidence:** Exact: tile-major gate/up inputs, down inputs, split partials and outputs byte-identical to the per-pair gather (T 1 to 4096 rows and crafted row counts, single-split fusion off and on, real weights of two layers); prefill state, final rows and tokens identical on both ranks in every PAIR arm at 8K and 32K.
- **Off switch:** STRIX_MMX_PREFILL_GATHER_TOK=0
- **Measured gain:** Gather stage per 4096-row call 6.35 -> 4.14 ms (1024 rows: 1.48 -> 0.99 ms), real weights. Together with the q/k/v item below, two Strix Halo nodes over 2x USB4, fresh process per arm: 8K TTFT -2.3 to -4.3 %, 32K -4.3 to -5.3 %.
- **Implementation / evidence:** [kernel](../src/minimax/moe/mmx_prefill.hip), [resolver](../src/minimax/engine/mm_defaults.cpp), [bench tests](../tests/hip_mmx_moe_bench.hip)

<a id="mm-pf-dense-qkv"></a>

### Prefill q/k/v projections on the v4 core

**Engine:** `minimax`. **Control:** `STRIX_MM_PF_DENSE_QKV`. **Default:** 1.

Run the attention q, k and v projections of a prefill piece as one call on the engine's v4 WMMA core, which decodes each weight tile once per 128 rows and transforms the input once for all three.

- **Preconditions:** Prefill pieces of at least 1024 rows, EXL3 weights of 3, 4, 5, 6 or 8 bits (other widths and smaller pieces keep the library call).
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. The o projection stays on the library (slower at 1024 rows on the v4 core).
- **Fallback notice:** None: unsupported shapes silently keep the library call.
- **Exactness evidence:** Exact: q, k, v and o outputs byte-identical to the library GEMM for 9 to 4096 rows (split counts 1 to 32), real weights of two layers; prefill state, final rows and tokens identical on both ranks in every PAIR arm at 8K and 32K.
- **Off switch:** STRIX_MM_PF_DENSE_QKV=0
- **Measured gain:** q+k+v per piece 12.37 -> 7.78 ms at 4096 rows, 2.61 -> 1.96 ms at 1024 rows (k+v alone 5.47 -> 2.09 ms). Together with the token-major gather: 8K TTFT -2.3 to -4.3 %, 32K -4.3 to -5.3 %.
- **Implementation / evidence:** [dense call](../src/minimax/moe/mmx_prefill.hip), [engine site](../src/minimax/engine/engine.hip), [resolver](../src/minimax/engine/mm_defaults.cpp), [bench tests](../tests/hip_mmx_moe_bench.hip)

<a id="mm-pf-router-v2"></a>

### Prefill router, tiled scores

**Engine:** `minimax`. **Control:** `STRIX_MM_PF_ROUTER_ROWS`. **Default:** 32.

Compute the prefill router scores in tiles of 16 rows by 32 experts, so the router weights are read once per 16 rows instead of once per 4 or 8; the expert selection runs as a separate small kernel.

- **Preconditions:** Batch router on (the default) and prefill pieces of 1024 to 4096 rows (other piece sizes keep the 8-row form).
- **Automatic selection:** Resolver supplies 32 only when unset; explicit values win.
- **Fallback notice:** None: every row count uses the same kernels.
- **Exactness evidence:** Exact: routes and weights byte-identical to the fused and 4/8-row forms for every input kind (ties, zeros, denormals, huge values) on the tested boundary and tail row counts; prefill state, final rows and tokens identical on both ranks in every PAIR arm.
- **Off switch:** STRIX_MM_PF_ROUTER_ROWS=8 (previous default) or =4
- **Measured gain:** post-exchange + router per 4096-row piece 5.40 -> 4.82 ms; TTFT 1K -0.9 %, 8K -0.8 % (in-process A/B/A, two Strix Halo nodes).
- **Implementation / evidence:** [router kernels](../src/hip_minimax_moe.hip), [resolver](../src/minimax/engine/mm_defaults.cpp), [router tests](../tests/hip_minimax_moe_tests.hip)

<a id="mm-pf-msa-scratch"></a>

### Prefill MSA scratch for whole pieces

**Engine:** `minimax`. **Control:** `STRIX_MM_PF_MSA_SCRATCH_MB`. **Default:** 1152.

Size the prefill attention scratch so that every 16-row window of a 4096-row piece runs in one batch (512 MiB fitted about 124 windows, so each piece ran three batches, the last one nearly empty).

- **Preconditions:** At least 4 GiB of device memory left after the request; otherwise the former 512 MiB cap is kept with one notice.
- **Automatic selection:** Resolver supplies 1152 only when unset; explicit values (64 to 8192) win.
- **Fallback notice:** One stderr line when the headroom check keeps 512 MiB.
- **Exactness evidence:** Exact: windows are independent, so only the number of launches changes; prefill state, final rows and tokens identical on both ranks in every PAIR arm.
- **Off switch:** STRIX_MM_PF_MSA_SCRATCH_MB=512
- **Measured gain:** Part of the shipping set below (8K -2.4 %, 32K -3.0 %).
- **Implementation / evidence:** [engine](../src/minimax/engine/engine.hip), [resolver](../src/minimax/engine/mm_defaults.cpp)

<a id="mm-pf-dense-o"></a>

### Prefill o projection on the v4 core

**Engine:** `minimax`. **Control:** `STRIX_MM_PF_DENSE_O`. **Default:** 1.

Run the attention output projection of full 4096-row prefill pieces on the engine's v4 WMMA core, like q/k/v.

- **Preconditions:** Exactly 4096-row pieces and EXL3 weights of 3, 4, 5, 6 or 8 bits (otherwise the library call).
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win.
- **Fallback notice:** None: other pieces silently keep the library call.
- **Exactness evidence:** Exact: o outputs byte-identical to the library GEMM on the tested row counts (9 to 4096); prefill state, final rows and tokens identical on both ranks in every PAIR arm.
- **Off switch:** STRIX_MM_PF_DENSE_O=0
- **Measured gain:** o projection per 4096-row piece 7.89 -> 7.59 ms; part of the shipping set below.
- **Implementation / evidence:** [engine site](../src/minimax/engine/engine.hip), [dense call](../src/minimax/moe/mmx_prefill.hip)

<a id="mm-pf-table-ring"></a>
### Prefill MoE table ring

**Engine:** `minimax`. **Control:** `STRIX_MM_PF_TABLE_RING`. **Default:** 1 in `strix-minimax-rank`; 0 in `strix-minimax-serve`.

Upload each prefill MoE call's item tables from a ring of pinned slots with per-slot events, so the host does not wait for the previous upload before planning the next call.

- **Preconditions:** One engine per process driven by a single host thread (the ring is process-wide); qualified in the rank app only, so serve keeps it off.
- **Automatic selection:** The rank app supplies 1 when unset and `STRIX_MM_BASELINE` is not 1; explicit values win.
- **Fallback notice:** None.
- **Exactness evidence:** Exact: identical table bytes; prefill state, final rows and tokens identical on both ranks at 1K and 8K (in-process A/B/A and fresh-process B A B A).
- **Off switch:** STRIX_MM_PF_TABLE_RING=0
- **Measured gain:** 1K TTFT -0.5 %; 8K not slower (in-process +0.7/+0.9/-1.7 %, fresh-process B 14.30/14.33 s vs A 15.03/14.33 s).
- **Implementation / evidence:** [table ring](../src/minimax/moe/mmx_prefill.hip)

<a id="mm-pf-msa-mode6"></a>

### MSA prefill mode 6

**Engine:** `minimax`. **Control:** `STRIX_MM_PF_MSA_REWORK`. **Default:** 6.

Mode 5 evaluates two heads' probabilities in one packed exponential; mode 6 also computes each 16-key tile's two running-max corrections in the spare lanes of that same evaluation instead of in two separate scalar evaluations.

- **Preconditions:** Supported MSA prefill shapes (as mode 5).
- **Automatic selection:** Resolver supplies 6 only when unset; explicit values win.
- **Fallback notice:** As mode 5.
- **Exactness evidence:** Exact: the fused MSA unit suite compares every partial, score, index and output byte of every rework mode (over 10,000 cases, both PV paths, dense and sparse, scratch sizes, graph replay); prefill state and final rows identical on both ranks in an in-process 5-vs-6 PAIR gate.
- **Off switch:** STRIX_MM_PF_MSA_REWORK=5 (previous default), =3, =0 (original)
- **Measured gain:** 8K TTFT in-process A/B/A vs mode 5: -1.6 / 0.0 / 0.0 % (not slower); producer fixture within noise at small row counts.
- **Implementation / evidence:** [kernel](../src/minimax/attn/msa_fused.hip), [tests](../src/minimax/attn/msa_fused_tests.hip)

<a id="mm-rank-pfx"></a>

### GPU-gated prefill exchange in the rank app

**Engine:** `minimax` (`strix-minimax-rank`). **Control:** `STRIX_MM_PFX`. **Default:** on.

The rank app now switches on its prefill exchange module, whose copy kernels and stream waits move each prefill partial without blocking the host (previously constructed but off).

- **Preconditions:** Host-path USB4 transport (unset = on only there; other transports keep it constructed but off; not built with the GPU-direct transport). The serve app does not use the module yet: it would disable nested serving.
- **Automatic selection:** Unset = on; `load` = constructed and off; `off` = not constructed.
- **Fallback notice:** Startup line names the module state.
- **Exactness evidence:** Exact: prefill state, final rows and tokens identical on both ranks in-process (1K, 8K) and across fresh processes in any order.
- **Off switch:** STRIX_MM_PFX=load (previous default) or =off
- **Measured gain:** TTFT 1K -3.5 %, 8K -1.5 % (in-process A/B/A, two Strix Halo nodes).
- **Implementation / evidence:** [rank app](../apps/minimax_rank_main.cpp), [module](../src/minimax/pfx/pf_exchange.hpp)

<a id="mm-pf-set-20261102"></a>

### Prefill shipping set: router, MSA scratch, o projection, rank exchange

Measured together on two Strix Halo nodes, fresh process per arm, previous defaults vs the five rows above (mode 6 measured separately): 8K TTFT -2.4 % (14.41 -> 14.06 s), 32K -3.0 % (68.08 -> 66.01 s); prefill state, final rows and tokens identical to the published references on both ranks in every arm.

<a id="mm-moe-gu-interleave"></a>

### Decode MoE gate/up interleave

**Engine:** `minimax`. **Control:** `STRIX_MMX_MOE_GU_INTERLEAVE`. **Default:** 1 via the engine resolver; raw-library unset is 0.

At eight decode rows, schedule the gate and up projection tiles of each expert interleaved instead of as separate matrix planes; every dot, split and finish order is unchanged.

- **Preconditions:** Eight-row decode steps using the v4 gate/up launcher, with or without router/shared overlap.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win.
- **Fallback notice:** Other batch sizes keep the existing schedule; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Decode MoE measurement: bitwise (610 N/U/S cases, every row == alone), B8 gate/up interleave diverse U8-32 -1.5..-3 %, B1/B4 unchanged; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MMX_MOE_GU_INTERLEAVE=0
- **Measured gain:** Decode MoE measurement: bitwise (610 N/U/S cases, every row == alone), B8 gate/up interleave diverse U8-32 -1.5..-3 %, B1/B4 unchanged
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-moe-single"></a>

### Decode MoE singleton reader

**Engine:** `minimax`. **Control:** `STRIX_MMX_MOE_SINGLE`. **Default:** 1 via the engine resolver; raw-library unset is 0.

At eight decode rows with many experts used by a single row, read those experts through a one-row coalesced reader chosen on the device per step; the arithmetic contract is unchanged.

- **Preconditions:** Eight-row v4 decode steps, Depth=4, narrow W=4, K3 routed matrices, at least 25 distinct routed experts and 18 used by one row; with or without overlap.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win.
- **Fallback notice:** Depth=8/12, shared/reused slots, K5, wide columns, W=8 and ineligible N/U/S keep the existing reader; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Decode MoE measurement: bitwise (610 N/U/S cases, every row == alone), B8 U>=25 singleton reader -3 %, B1/B4 unchanged; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MMX_MOE_SINGLE=0
- **Measured gain:** Decode MoE measurement: bitwise (610 N/U/S cases, every row == alone), B8 U>=25 singleton reader -3 %, B1/B4 unchanged
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-moe-down-split"></a>

### Decode MoE down partials and ordered finish

**Engine:** `minimax`. **Control:** `STRIX_MMX_MOE_DOWN_SPLIT`. **Default:** 1 via the engine resolver; raw-library unset is 0.

At four decode rows, the v4 down launch writes only its split partials and one stream-ordered 48-CTA launch does the finish and fixed-order combine (+ shared) and publishes the exchange message, removing the per-tile last-arrival counters; the arithmetic contract is unchanged.

- **Preconditions:** Four-row v4 decode steps on the wide W=4 Depth=4 path (the resolver default WIDE=2).
- **Automatic selection:** Resolver supplies 1 only when unset; 1 selects the split only at the measured B4 wide W4 Depth4 regime (host-known, no synchronisation to read U/S); 2 forces it on every v4 down (diagnostic); explicit values win.
- **Fallback notice:** B1, B2, B3, B5-B8, narrow, W=8 and Depth=8/12 keep the fused last-arrival finish at 1; invalid explicit settings fail validation.
- **Exactness evidence:** hip_mmx_moe_bench --diverse --test: 610 N/U/S cases, IL/SINGLE x DOWN 0/1/2 output/workspace/routes bitwise, repeated calls, every row == alone; same partial scratch, ascending split sum, H128/scales, ascending-expert fmaf, shared add; repeat golden-logit/token/state gates for the chosen workload.
- **Off switch:** STRIX_MMX_MOE_DOWN_SPLIT=0
- **Measured gain:** Decode MoE measurement (five fresh processes OFF/ON/OFF/ON/OFF, 16 samples/cell) B/A at B4: U4/S0 0.986/0.986, U8/S0 0.993/0.989, U12/S8 0.991/0.997, U16/S16 0.986/1.000; B1 1.001/1.004 and B8 mixed stay off
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-attend-diet"></a>

### Decode attention 32 KB LDS footprint

**Engine:** `minimax`. **Control:** `STRIX_MM_ATTEND_DIET`. **Default:** 1 via the engine resolver; raw-library unset is 0.

On two single-row sparse decode sequences at 8K positions or more, the v3 attend launches with its actual 32 KB dynamic LDS (K/V tile unioned with the selection scratch) instead of the 43008-byte floor, so four 16-wave blocks share a WGP; kernel and arithmetic are unchanged.

- **Preconditions:** Decode steps with two sequences of one row each, both past 8191 positions (sparse), and a selection scratch that fits the 32 KB tile.
- **Automatic selection:** Resolver supplies 1 only when unset; the engine sets the per-call view flag only on route::attend_n2_sparse grids; when STRIX_MM_ATTEND4_DECODE selects v4 there, v4 takes precedence; explicit values win.
- **Fallback notice:** N1, N4, N8, verify windows, dense steps (below 2048) and 2K-8K positions keep the 42 KB footprint (1K N8 96.4 -> 103.6, 1K W8 77.9 -> 83.1, 8K N4 127.7 -> 134.4 us were slower).
- **Exactness evidence:** msa_fused_tests run_case: v2/v3 lds_diet at every head split bitwise vs the current kernels (attention, caches, selection, partials); allocation-only change.
- **Off switch:** STRIX_MM_ATTEND_DIET=0
- **Measured gain:** Same binary/process bench on current main kernels (msa_fused_tests, STRIX_MMATTN_BENCH_DIET=1, graph, cache rotation; v3-full / v3 32 KB diet / v4 one head per wave): sparse N2/W1 8K 74.4 / 69.9 / 71.0, 32K 133.5 / 130.7 / 131.4, 128K 343.1 / 342.1 / 342.6 us/layer; the diet is chosen first. Controls: 8K N4 v3-full 122.8 vs diet 129.3 (stays v3). Earlier t9: 8K 83.9 -> 76.2 us
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-attend4-decode"></a>

### Decode attention v4, one head per wave

**Engine:** `minimax`. **Control:** `STRIX_MM_ATTEND4_DECODE`. **Default:** 1 via the engine resolver; raw-library unset is 0.

On two single-row sparse decode sequences at 8K positions or more, decode attention launches the v4 attend (all 16 heads of a KV head per block, one head per wave sharing each K/V LDS read, dynamic LDS at its real size) instead of v3; arithmetic is unchanged.

- **Preconditions:** Decode steps with two sequences of one row each, both past 8191 positions (sparse).
- **Automatic selection:** Resolver supplies 1 only when unset; the engine selects v4 hpw1 per step only on route::attend_n2_sparse grids, before the LDS diet; explicit values win.
- **Fallback notice:** N1, N4, N8, verify windows, dense steps and 2K-8K positions keep v3 (1K N1 25.3 -> 28.5, 1K N8 92.3 -> 103.9, 8K N4 122.0 -> 133.6 us were slower with v4-h2); prefill keeps STRIX_MM_PF_ATTEND4.
- **Exactness evidence:** msa_fused_tests run_case: v4 heads_per_wave 1/2/4 bitwise vs the current kernels (attention, caches, selection, partials) on every decode case.
- **Off switch:** STRIX_MM_ATTEND4_DECODE=0
- **Measured gain:** Used on the same grids only when STRIX_MM_ATTEND_DIET=0 (the diet is faster there): same-binary bench v3-full / v4 one head per wave 8K 74.4 / 71.0, 32K 133.5 / 131.4, 128K 343.1 / 342.6 us/layer; earlier t12 8K 78.6 -> 75.2 us
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-pf-idx-tile"></a>

### MSA prefill index query tiles

**Engine:** `minimax`. **Control:** `STRIX_MM_PF_IDX_TILE`. **Default:** 1.

Reuse each loaded index-key tile across consecutive prefill query windows when scoring MSA blocks; the score arithmetic is unchanged.

- **Preconditions:** Prefill chunks larger than eight rows past the 2048-position dense prefix.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. `STRIX_MM_PF_IDX_ROWS` (16, 32 or 64; default 64) sets the query tile; 64 runs the Q64 macrotile kernel (one key-tile load per 64 queries).
- **Fallback notice:** Shorter chunks and the dense prefix keep the per-window index path; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Prefill index measurement: exact (state/final rows identical), index scoring 8K -6.7 %, 32K -6.4 %; TTFT 8K -1.9 %, 32K -0.6 %, 1K neutral; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_PF_IDX_TILE=0
- **Measured gain:** Prefill index measurement: exact (state/final rows identical), index scoring 8K -6.7 %, 32K -6.4 %; TTFT 8K -1.9 %, 32K -0.6 %, 1K neutral
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-pf-idx-rows"></a>

### MSA prefill index Q64 macrotiles

**Engine:** `minimax`. **Control:** `STRIX_MM_PF_IDX_ROWS`. **Default:** 64.

Score 64 prefill queries per loaded index-key tile in four 16-query subtiles; every dot, reduction, maximum and selection step is unchanged.

- **Preconditions:** Index query tiles on (`STRIX_MM_PF_IDX_TILE=1`, the default) and prefill chunks larger than eight rows past the 2048-position dense prefix.
- **Automatic selection:** Resolver supplies 64 only when unset; explicit values win (16, 32 or 64).
- **Fallback notice:** Invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Prefill index measurement: exact (state/final rows identical), Q64 vs Q32 index scoring 8K -35 %, 32K -28 %; TTFT 8K -1.5 %, 32K -1.8 %, 1K neutral; fused MSA unit tests byte-compare Q64 against the default and Q32 paths.
- **Off switch:** STRIX_MM_PF_IDX_ROWS=32
- **Measured gain:** Prefill index measurement: exact (state/final rows identical), Q64 vs Q32 index scoring 8K -35 %, 32K -28 %; TTFT 8K -1.5 %, 32K -1.8 %, 1K neutral
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-pf-moe-pipe"></a>

### MoE prefill pipeline

**Engine:** `minimax`. **Control:** `STRIX_MMX_PREFILL_PIPE`. **Default:** 1.

Pipeline compatible expert prefill work.

- **Preconditions:** Supported prefill expert core and ordered workspace ownership.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Prefill measurement: exact, 8K B/A 0.973 / 0.935 / 0.924; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MMX_PREFILL_PIPE=0
- **Measured gain:** Prefill measurement: exact, 8K B/A 0.973 / 0.935 / 0.924
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-pf-kv-tile"></a>

### Automatic KV tile

**Engine:** `minimax`. **Control:** `STRIX_MM_PF_KV_TILE`. **Default:** auto.

Choose the supported KV tile implementation.

- **Preconditions:** Linked trellis tile override hook and supported KV shape.
- **Automatic selection:** Resolver supplies auto only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Absent trellis lab tile hook prints `STRIX_MM_PF_KV_TILE: trellis lab tile override absent; switch is a no-op`; keep the normal KV path.
- **Exactness evidence:** Published same-identity qualification: Prefill measurement: exact, 8K B/A 0.960 / 0.949 / 1.017; needs the Trellis tile hook, else a logged no-op; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_PF_KV_TILE=off
- **Measured gain:** Prefill measurement: exact, 8K B/A 0.960 / 0.949 / 1.017; needs the Trellis tile hook, else a logged no-op
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-pf-tx-uncached"></a>

### Uncached prefill TX

**Engine:** `minimax`. **Control:** `STRIX_MM_PF_TX_UNCACHED`. **Default:** 1.

Avoid cache pollution in prefill exchange sends.

- **Preconditions:** Prefill module route and uncached TX allocation support.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unavailable uncached TX allocation prints `mmpfx: uncached TX slots unavailable ...; TX stays host-coherent`; retain the coherent buffer.
- **Exactness evidence:** Published same-identity qualification: Prefill exchange measurement: exact, exchange-send 0.23 -> 0.002 s GPU, ~1.3 % (prefill module route); repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_PF_TX_UNCACHED=0
- **Measured gain:** Prefill exchange measurement: exact, exchange-send 0.23 -> 0.002 s GPU, ~1.3 % (prefill module route)
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-dist-crc-parallel"></a>

### Parallel host CRC

**Engine:** `minimax`. **Control:** `STRIX_DIST_CRC_PARALLEL`. **Default:** 1.

Compute host transport CRC on workers.

- **Preconditions:** Host exchange route; preserve compatible settings across ranks.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Host transport measurement: ABA x3 TTFT 1K -12.8 %, decode -1 %, bits unchanged; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_DIST_CRC_PARALLEL=0
- **Measured gain:** Host transport measurement: ABA x3 TTFT 1K -12.8 %, decode -1 %, bits unchanged
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-dist-host-crc-overlap"></a>

### Host CRC overlap

**Engine:** `minimax`. **Control:** `STRIX_DIST_HOST_CRC_OVERLAP`. **Default:** 1.

Overlap host CRC and transport work.

- **Preconditions:** Host exchange route; integrity checks enabled.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Host transport measurement: ABA x3 TTFT 1K -12.8 %, decode -1 %, bits unchanged; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_DIST_HOST_CRC_OVERLAP=0
- **Measured gain:** Host transport measurement: ABA x3 TTFT 1K -12.8 %, decode -1 %, bits unchanged
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-dist-host-chunk"></a>

### Host transfer chunks

**Engine:** `minimax`. **Control:** `STRIX_DIST_HOST_CHUNK_BYTES`. **Default:** 1048576.

Send host transport in 1 MiB pieces.

- **Preconditions:** Host exchange route with chunk support.
- **Automatic selection:** Resolver supplies 1048576 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Host transport measurement: ABA x3 TTFT 1K -12.8 %, decode -1 %, bits unchanged; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_DIST_HOST_CHUNK_BYTES=0
- **Measured gain:** Host transport measurement: ABA x3 TTFT 1K -12.8 %, decode -1 %, bits unchanged
- **Implementation / evidence:** [38-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-routes"></a>

### Decode and prefill exchange routes

**Engine:** `minimax`. **Control:** `STRIX_MM_XFER; STRIX_MM_PFX; STRIX_MM_PF_INTERLEAVE`. **Default:** XFER on, PFX loaded, interleave on.

Use the host-path decode exchange module, load the prefill exchange module, and interleave prefill chunks over the selected exchange.

- **Preconditions:** Two compatible ranks on a host exchange route; module loads are skipped with GPU-direct transport.
- **Automatic selection:** Unset XFER loads/uses decode exchange. Unset PFX loads the module; explicit PFX=on installs it. Interleave uses the installed module or communicator; =comm forces the communicator.
- **Fallback notice:** GPU-direct keeps its communicator route; startup prints the module/route selection. Invalid route values fail validation.
- **Exactness evidence:** Same token IDs, K/V/index state and terminal rows in prefill qualification; retain protocol integrity gates.
- **Off switch:** STRIX_MM_XFER=off; STRIX_MM_PFX=off; STRIX_MM_PF_INTERLEAVE=off (one at a time)
- **Measured gain:** Interleaving: 8K TTFT 40.0 -> 35.4 s; 1K inert

<a id="mm-gu4"></a>

### Scalar gu4 expert addressing

**Engine:** `minimax`. **Control:** `STRIX_MINIMAX_GU4_SCALAR`. **Default:** ON (build time).

Use the scalar gu4 MoE expert-address reader.

- **Preconditions:** MiniMax build and supported expert layout.
- **Automatic selection:** CMake defaults the scalar reader on; the effective config reports its build state.
- **Fallback notice:** Build-time selection has no runtime fallback notice; use the previous reader by rebuilding.
- **Exactness evidence:** Same-layout expert reader qualification; verify model tokens/digests after changing the build.
- **Off switch:** STRIX_MINIMAX_GU4_SCALAR=OFF at CMake configure time
- **Measured gain:** No isolated measurement

<a id="minimax-grouped-prefill-async"></a>

### Grouped prefill early destination and late source fence

**Engine:** `minimax`. **Control:** `STRIX_MM_PF_ASYNC_START`, `STRIX_MM_PF_LATE_SRC_FENCE`, `STRIX_MM_PF_ASYNC_MIN_ROWS`. **Default:** ON for grouped calls whose every member has at least 2048 rows.

A grouped prefill call publishes its receive destination early and fences each member's source only before the next
producer overwrites it, so the copies overlap compute. Compute and scratch stay serialized; only copy timing changes.

- **Preconditions:** Two-rank grouped prefill; every member of the grouped call has at least `STRIX_MM_PF_ASYNC_MIN_ROWS` rows (default 2048; `0` removes the gate). Smaller groups take the ordinary path with byte-identical scheduling. The decision uses only the ordered member row counts and is identical on both ranks.
- **Exactness evidence:** Scheduling only: unchanged producer/post kernels, group rows, payload bytes, rank summation order, sequence/tags and chunk CRC. Complete token IDs identical in every measured arm, including resume, nested-512, nested-511 and a 7936-token grouped prompt on the async path (both ranks).
- **Off switch:** `STRIX_MM_PF_ASYNC_START=0` (global); `STRIX_MM_PF_LATE_SRC_FENCE=0` for the late fence; `STRIX_MM_BASELINE=1` unless explicitly overridden.
- **Measured gain:** Balanced 7940-token prefill (4096+3844): -2.3 to -2.6 % in both measurement orders. Unbalanced shapes (4096+443) measured neutral and take the ordinary path through the row gate.
- **Implementation / evidence:** [grouped prefill](minimax-grouped-prefill.md)

## GLM

```sh
strix-glm5-rank MODEL_DIR --world 2 --rank R --transport usb4-auto --prompt PROMPT.u32 --decode 64
```

Launch both ranks with the same model, prompt and compatible transport. `--print-effective-config` reports
the resolution before GPU/peer work. Defaults: `--identity r1b`, `--mtp 0`, `--prefill auto`,
`--decode-xch xfer`. R1b uses R1 prompt arithmetic and R1b generation; `--identity r1` restores R1 throughout.
`STRIX_GLM_IDENTITY` overrides the CLI identity; CLI prefill overrides `STRIX_GLM_PREFILL`;
`STRIX_GLM_DECODE_XCH` overrides CLI exchange. Explicit values win over baseline defaults.

R1b changes FP32 summation order and is **not bit-identical to R1**. Own-identity runs are stable, but the
32-prompt/8192-position comparison found 99.19% top-1 agreement and mean KL about 1.8e-3 nats; free runs
diverged on 28/32 prompts within 19-227 tokens. Measured R1b 41.93-41.96 versus R1 47.33-47.48 ms/token
is an identity comparison, not an exact R1 optimization. The faster R1b kernels separately measured R1/R8
-7.8/-5.4% with the same tokens within R1b. See [identity implementation](../src/glm/engine/glm_r1b.hip).

`--baseline` / `STRIX_GLM_BASELINE=1` restores R1, reference prefill, MTP loaded, comm decode exchange and
the listed off values unless explicitly overridden. `STRIX_GLM_MOE=fast` remains on: prepared reference
experts do not fit the real TP2 rank's GTT. Its per-item `ref` switch is not a runnable full-model baseline.
Use G3/G4, `--idcheck`, prefill logits/digests and the speculative oracle within the selected identity;
R1b-vs-R1 agreement is reported separately from those exactness gates.

<a id="glm-prefill"></a>

### Exact chunked prefill

**Engine:** `glm`. **Control:** `--prefill auto`. **Default:** auto.

Use chunked gemv prefill with the R1 arithmetic identity.

- **Preconditions:** Ordinary prefill, MTP/spec off, not r1b-all, G3-only/smoke/memory-plan/identity-check/reference-gate modes, with capacity for chunked path/workspace.
- **Automatic selection:** auto chooses gemv for normal generation; excluded diagnostic/spec modes choose ref. R1b default still prefills in R1.
- **Fallback notice:** At world 1, failed auto admission prints `prefill auto: exact chunked path not admitted (...) -> ref (same bits, slower)`. Explicit requests and world 2 failures propagate so peers cannot choose incompatible exchange patterns.
- **Exactness evidence:** Real-model logits are bitwise equal on both ranks to previous exact R1 prefill; 1K/8K/32K gates. This does not assert R1b decode equality to R1.
- **Off switch:** --prefill ref; STRIX_GLM_PREFILL=ref if no CLI prefill override
- **Measured gain:** 1K/8K/32K TTFT 4.268/34.70/144.75 s = 240/236/226 tok/s
- **Implementation / evidence:** [Resolver](../apps/glm_rank_defaults.hpp), [admission](../apps/glm_rank_main.cpp)

<a id="glm-identity"></a>

### R1b decode identity

**Engine:** `glm`. **Control:** `--identity r1b`. **Default:** ON.

Use R1 prompt arithmetic and the R1b generation identity.

- **Preconditions:** Admitted GLM model, R1b kernels and identity-specific reference.
- **Automatic selection:** Default identity is r1b unless CLI/environment chooses another; baseline uses r1.
- **Fallback notice:** No automatic substitution between identities; explicit unsupported settings retain validation.
- **Exactness evidence:** Stable within R1b; not bit-identical to R1. Teacher-forced top-1 agreement 8126/8192; mean KL ~1.8e-3 nats; free runs diverge on 28/32 prompts.
- **Off switch:** --identity r1 (unset STRIX_GLM_IDENTITY, which otherwise wins)
- **Measured gain:** 41.93-41.96 vs R1 47.33-47.48 ms/token (-11.5%)

<a id="glm-xfer"></a>

### Direct decode exchange

**Engine:** `glm`. **Control:** `--decode-xch xfer`. **Default:** ON.

Use the production Xfer decode exchange.

- **Preconditions:** Two-rank transport, compatible exchange buffers and topology.
- **Automatic selection:** Default xfer; baseline comm; STRIX_GLM_DECODE_XCH overrides CLI.
- **Fallback notice:** Invalid/incompatible transport fails; no independent per-rank protocol fallback.
- **Exactness evidence:** Same-identity rank token/state/logit gates and exchange integrity checks.
- **Off switch:** --decode-xch comm (unset overriding environment)
- **Measured gain:** Boundary bundle reduces exchange overhead; no isolated measurement

<a id="glm-kda-tree"></a>

### KDA two-barrier-tree column-quad forms

**Engine:** `glm`. **Control:** `STRIX_GLM_KDA_TREE; STRIX_GLM_KDA_LANES`. **Default:** on.

The automatic KDA lane form uses the two-barrier-tree column-quad kernels: 2ct at one row, cqst at 2..8 rows (same bits as 2c / cqs).

- **Preconditions:** Decode KDA steps of 1..8 independent rows; kSeq keeps 2 lanes.
- **Automatic selection:** Automatic unless STRIX_GLM_KDA_LANES names a form.
- **Fallback notice:** None; STRIX_GLM_KDA_TREE=0 restores 2c / cqs.
- **Exactness evidence:** KDA kernel gate 57/0 bitwise (2ct / cqst vs 2c / cqs); real GLM-5.3-Flash TP2 on the pair, default (2ct / cqst) and STRIX_GLM_KDA_TREE=0 arms alternating: R1b decode 64 tokens fnv 8256e1e0 on both ranks and R8 batch fnv 99838248 in every arm, spec SP1a PASS on both ranks.
- **Off switch:** STRIX_GLM_KDA_TREE=0
- **Measured gain:** kda.step us 2c -> 2ct / cqs -> cqst: R1 34.4 -> 33.9, R2 63.3 -> 62.8, R4 111.1 -> 109.1, R8 209.1 -> 208.0 (-0.5..-1.8 %); model decode A/B (fresh processes A B B A A B): R1 40.12 vs 40.13 ms/token, R8 112.85 vs 112.95 ms/step (not slower)

<a id="glm-hp-crc-parallel"></a>

### Parallel host CRC

**Engine:** `glm`. **Control:** `STRIX_DIST_CRC_PARALLEL`. **Default:** 1.

Compute host transport CRC using parallel workers while retaining integrity checks.

- **Preconditions:** Host-path exchange; compatible rank protocol.
- **Automatic selection:** Unset resolves to 1; the app exports it before runtime readers. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Measurement: hp host path, exact TTFT 1K -8.5 %, 8K -3.1 %, logits fnv matches reference 16/16; compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_DIST_CRC_PARALLEL=0
- **Measured gain:** hp host path, exact TTFT 1K -8.5 %, 8K -3.1 %, logits fnv matches reference 16/16
- **Implementation / evidence:** [24-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-hp-crc-overlap"></a>

### Host CRC overlap

**Engine:** `glm`. **Control:** `STRIX_DIST_HOST_CRC_OVERLAP`. **Default:** 1.

Overlap host CRC work with exchange transfers.

- **Preconditions:** Host-path exchange with integrity checks enabled.
- **Automatic selection:** Unset resolves to 1; the app exports it before runtime readers. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Measurement: hp host path, exact TTFT 1K -8.5 %, 8K -3.1 %, logits fnv matches reference 16/16; compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_DIST_HOST_CRC_OVERLAP=0
- **Measured gain:** hp host path, exact TTFT 1K -8.5 %, 8K -3.1 %, logits fnv matches reference 16/16
- **Implementation / evidence:** [24-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-hp-chunk-bytes"></a>

### Host transfer chunks

**Engine:** `glm`. **Control:** `STRIX_DIST_HOST_CHUNK_BYTES`. **Default:** 1048576.

Split host exchange payloads into 1 MiB chunks.

- **Preconditions:** Host-path exchange and chunk support.
- **Automatic selection:** Unset resolves to 1048576; the app exports it before runtime readers. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Measurement: hp host path, exact TTFT 1K -8.5 %, 8K -3.1 %, logits fnv matches reference 16/16; compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_DIST_HOST_CHUNK_BYTES=0
- **Measured gain:** hp host path, exact TTFT 1K -8.5 %, 8K -3.1 %, logits fnv matches reference 16/16
- **Implementation / evidence:** [24-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-hsa-no-fragment-allocator"></a>

### Allocator granules

**Engine:** `glm`. **Control:** `HSA_DISABLE_FRAGMENT_ALLOCATOR`. **Default:** 1.

Disable the HSA fragment allocator so production allocation granules fit GTT.

- **Preconditions:** Runtime allocator and GTT capacity for the production model.
- **Automatic selection:** Unset resolves to 1; the app exports it before runtime readers. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Measurement: production decode and prefill configuration; allocator granules fit GTT; compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** HSA_DISABLE_FRAGMENT_ALLOCATOR=0
- **Measured gain:** production decode and prefill configuration; allocator granules fit GTT
- **Implementation / evidence:** [24-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-r1b-dense-auto"></a>

### R1b dense scheduling

**Engine:** `glm`. **Control:** `STRIX_GLM_R1B_DENSE`. **Default:** 7.

Select dense variant 7 within the R1b generation identity.

- **Preconditions:** R1b generation identity and supported dense sites.
- **Automatic selection:** Unset resolves to 7; the engine default provides it; the resolver reports it. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Measurement: R1b-fast kernels -7.8 % R1 / -5.4 % R8, same tokens; compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_GLM_R1B_DENSE=2
- **Measured gain:** R1b-fast kernels -7.8 % R1 / -5.4 % R8, same tokens
- **Implementation / evidence:** [24-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-r1b-hc-v4"></a>

### R1b HC v4

**Engine:** `glm`. **Control:** `STRIX_GLM_R1B_HC`. **Default:** 4.

Use HC variant 4 within the R1b generation identity.

- **Preconditions:** R1b generation identity and compatible HC sites.
- **Automatic selection:** Unset resolves to 4; the engine default provides it; the resolver reports it. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Measurement: with r1b-dense-auto, R1b-fast kernels -7.8 % R1 / -5.4 % R8, same tokens; compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_GLM_R1B_HC=1
- **Measured gain:** with r1b-dense-auto, R1b-fast kernels -7.8 % R1 / -5.4 % R8, same tokens
- **Implementation / evidence:** [24-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-hc-pre-v7"></a>

### HC pre v7

**Engine:** `glm`. **Control:** `STRIX_GLM_HC_PRE_V`. **Default:** 7.

Use the qualified v7 pre-HC reduction/normalization schedule.

- **Preconditions:** Supported HC pre shapes and norm/weight geometry.
- **Automatic selection:** Unset resolves to 7; the engine default provides it; the resolver reports it. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Measurement: hc_pre 77.4-77.9 vs 102.1-102.8 us, gates 0-7 bitwise; tokens match reference; compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_GLM_HC_PRE_V=4
- **Measured gain:** hc_pre 77.4-77.9 vs 102.1-102.8 us, gates 0-7 bitwise; tokens match reference
- **Implementation / evidence:** [24-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-post-exchange-fast"></a>

### Fast post-exchange consumers

**Engine:** `glm`. **Control:** `STRIX_GLM_POST`. **Default:** fast.

Consume post-exchange values through fused fast kernels rather than separate reference consumers.

- **Preconditions:** Compatible two-rank post-exchange buffers.
- **Automatic selection:** Unset resolves to fast; the engine default provides it; the resolver reports it. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Measurement: fast post-exchange consumers match STRIX_GLM_POST=ref and reference (real model, 6 layers); compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_GLM_POST=ref
- **Measured gain:** fast post-exchange consumers match STRIX_GLM_POST=ref and reference (real model, 6 layers)
- **Implementation / evidence:** [24-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-attn-fast"></a>

### Fast attention

**Engine:** `glm`. **Control:** `STRIX_GLM_ATTN`. **Default:** fast.

Install FastAttn at the attention seam, preserving the selected identity and state contract.

- **Preconditions:** Supported attention geometry; reference form supports plain execution only.
- **Automatic selection:** Unset resolves to fast; the engine default provides it; the resolver reports it. Explicit values win.
- **Fallback notice:** Explicit ref is for plain execution only; speculative verify on the reference seam fails validation.
- **Exactness evidence:** Measurement: FastAttn matches STRIX_GLM_ATTN=ref (G3), Tier A 12-40x; compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_GLM_ATTN=ref
- **Measured gain:** FastAttn matches STRIX_GLM_ATTN=ref (G3), Tier A 12-40x
- **Implementation / evidence:** [24-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-attn-tierb-fused"></a>

### Fused attention Tier B

**Engine:** `glm`. **Control:** `STRIX_GLM_TIERB`. **Default:** fused.

Use fused Tier-B attention projections from raw weights instead of prepared library linears.

- **Preconditions:** TP2 attention, supported projection shapes and raw weights.
- **Automatic selection:** Unset resolves to fused; the engine default provides it; the resolver reports it. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Measurement: fused Tier B -2.46 ms/token vs strix_gemv, tokens match reference; compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_GLM_TIERB=lib
- **Measured gain:** fused Tier B -2.46 ms/token vs strix_gemv, tokens match reference
- **Implementation / evidence:** [24-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-xfer-boundary-fused"></a>

### Fused Xfer boundary

**Engine:** `glm`. **Control:** `STRIX_GLM_BOUNDARY`. **Default:** 1.

Fuse exchange reduction and its dependent boundary work.

- **Preconditions:** Compatible Xfer decode boundary.
- **Automatic selection:** Unset resolves to 1; the engine default provides it; the resolver reports it. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Measurement: Xfer reduce +1.41 -> fused boundary ~+1.10 ms/token over local, bitwise; compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_GLM_BOUNDARY=0
- **Measured gain:** Xfer reduce +1.41 -> fused boundary ~+1.10 ms/token over local, bitwise
- **Implementation / evidence:** [24-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-moe-fast"></a>

### Fast raw-expert MoE

**Engine:** `glm`. **Control:** `STRIX_GLM_MOE`. **Default:** fast.

Use raw-only expert storage and the bitwise FastMoe implementation so the TP2 model fits GTT.

- **Preconditions:** Real TP2 raw expert slices; prepared reference experts exceed GTT.
- **Automatic selection:** Unset resolves to fast; the engine default provides it; the resolver reports it. Explicit values win.
- **Fallback notice:** Baseline retains fast MoE; explicit ref may fail full-model TP2 memory admission. No silent numerical-identity substitution.
- **Exactness evidence:** Measurement: FastMoe matches DefaultMoe bitwise (real TP2 slices); raw-only experts (fits GTT); compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_GLM_MOE=ref (excluded from master baseline for capacity)
- **Measured gain:** FastMoe matches DefaultMoe bitwise (real TP2 slices); raw-only experts (fits GTT)
- **Implementation / evidence:** [24-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-max-rows-auto"></a>
### Memory-admitted prefill chunk rows

**Engine:** `glm`. **Control:** `--max-rows`, `STRIX_GLM_MAX_ROWS_AUTO`. **Default:** automatic 8192 or 4096.

- **Routes:** The 8192 default, when both ranks admit it, applies to prompt prefill in two-rank decode
  runs: production decode with MTP self-speculation and plain chunked prefill inside `--decode` runs.
  `--pf-ttft` and gate harnesses keep 4096 unless `--max-rows 8192` is given explicitly.
  Reference prefill (`--prefill ref`, automatic prefill under `r1b-all` or `--idcheck` with decode)
  uses fixed 4096 with source `auto-4096-ref`, without auto admission or a row-count agreement exchange.
- **Automatic selection:** After load planning and allocator calibration, before weight and row-buffer allocation,
  the app checks the same physical load, first-forward reserve, pinned staging, whole-tensor read and bounce
  terms as load admission. Headroom subtracts free CMA and file pages in CMA; the GTT limit and its margin
  also apply. Prefill path, workspace, install owners and communication growth are included, along with the
  device-free guard. RAM headroom retains the largest of 8 GiB, the load margin and the prefill margin.
  Missing telemetry or a refused 8192 candidate selects 4096 before rank agreement.
- **Rank agreement:** Normal two-rank Comm decode reuses the existing eligibility exchange and mismatch vote.
  A spare configuration bit carries auto mode; only auto row values are negated so the merge selects the
  minimum proposed count. Both auto ranks use 8192 only if both admit it. Mixed auto/fixed modes refuse on
  both ranks before any forward. Fixed modes preserve the existing payload bytes, message count and tags. After communication growth,
  a fresh admission check uses one auto-only minimum vote; a refusal at 8192 selects 4096 on both ranks.
- **Protocol limitation:** Prefill-only, reference gates and batch-only runs have no common startup configuration
  exchange. Two-rank Comm auto uses 4096 with source `auto-4096-no-handshake`, resolved independently
  from local flags before engine construction, without a control exchange. Explicit `--max-rows` must match on both ranks.
  Fixed-mode row mismatches in these runs cannot be checked at startup without changing their wire protocol;
  only normal decode has the existing equality check. External `--xch xfer` auto resolves to 4096 with source
  `auto-4096-xfer` and performs no row-count or host configuration exchange.
- **Communication memory:** Auto constructs the Communicator and CommExchange with 4096-row capacity and grows
  them only after agreement on 8192. A 4096 fallback shrinks both pools to exactly the off-switch communication capacity.
- **Preconditions:** Omitted `--max-rows`, outside baseline, with `STRIX_GLM_MAX_ROWS_AUTO` unset or enabled.
  Explicit row counts retain their value; baseline retains 1024 unless explicitly overridden. Existing load and
  prefill allocation guards still apply, including protection against memory pressure changing during load.
- **Inspection:** `--print-effective-config` reports the host-independent candidate and startup policy,
  without host measurements. Startup stderr logs admission headroom and the final `max-rows` value and source:
  `auto-8192`, `auto-4096-memory`, `auto-4096-no-handshake`, `auto-4096-ref`, `auto-4096-xfer`, `explicit` or `baseline`.
- **Exactness evidence:** Measured 8192-row chunks produce the same logits and tokens as 4096 on the exact
  prefill route; chunk size preserves its arithmetic.
- **Off switch:** `STRIX_GLM_MAX_ROWS_AUTO=0` restores fixed 4096; `--max-rows N` fixes an explicit size.
- **Measured gain:** Chunked-prefill route at 8192 rows: TTFT -4 % at 8K; 32K 116.3 to 109.9 s (-5.5 %);
  about 7.3 GiB additional GPU memory per node. Measuring this with `--pf-ttft` requires explicit `--max-rows 8192`.

<a id="glm-pf-exact-moe-auto"></a>
<a id="glm-pf-exact-moe-v3"></a>

### Exact prefill MoE auto

**Engine:** `glm`. **Control:** `STRIX_GLM_PF_EXACTMOE_V`. **Default:** auto.

Grouped expert work in exact chunked prefill; calls of 128 rows or more run routed and shared experts through the X3 core (v11: activations in scalar registers, LDS only for decoded weights; whole 64-row tiles at eight rows per wave and each group's remainder in 1/2/4-row-per-wave buckets), smaller calls keep the W2 grouped path (v3), whose tiles compute only the smallest row bucket (1/2/4/8/16) covering their rows. The expert plan (pairs, tiles, groups) is built on the GPU from the routed ids, so a MoE call needs no route download, host grouping or plan upload.

- **Preconditions:** Exact chunked prefill; auto takes v11 for MoE calls of at least 128 rows (STRIX_GLM_PF_EXACTMOE_X_MIN_ROWS) and v3 below. 10 is the X3 core with 32/64-row tiles per group, 7 the previous X2 core, 3 forces v3 for every call; STRIX_GLM_PF_EXACTMOE_V3_RCAP=0 makes v3 compute 16 rows per tile again. The device plan needs a grouped variant (3, 6-11) and 16-row work tiles; STRIX_GLM_PF_EXACTMOE_PLAN=host keeps the host plan.
- **Automatic selection:** Unset resolves to auto; the app sets it; the resolver reports it in `--print-effective-config`. Explicit values win; `--baseline` sets 1.
- **Fallback notice:** Below the row threshold v3 runs; invalid values fail. No silent numerical-identity substitution.
- **Exactness evidence:** Bitwise for all outputs: v7, v10 and v11 route routed and shared experts through the X2 / X3 cores with the v3 expert order, gather/scatter and fold (X3 keeps the X2 DOT2 chains, reduce4 and split order; only the operand source changes); v3 and v11 row buckets run the same statements for every live row; the device plan's pairs, tiles and groups are byte-equal to the host plan for every gated call, and invalid ids raise the engine's flag instead of throwing; MoE outputs, FP16 down inputs and final rows equal v3 for row tails, 2047/2048/2049 and 4096-row calls, concentrated expert groups, signed zero, FP16 subnormals, +-Inf and NaN payloads; on the real pair decode anchors, speculative oracle and 1K/8K/32K prefill logits equal the v3 path. Internal FP32 gate/up workspace NaN payloads may differ in the FP16 overflow case (+Inf meeting -Inf inside a DOT2 chain; not reachable with finite activations); the payload is discarded at the FP16 down input.
- **Off switch:** STRIX_GLM_PF_EXACTMOE_V=10, 7 (previous large-call cores) or 3; STRIX_GLM_PF_EXACTMOE_V3_RCAP=0; STRIX_GLM_PF_EXACTMOE_PLAN=host; 1 for the baseline
- **Measured gain:** 8K TTFT 25.24 -> 20.32 s (-19.5 %) and 1K 3.75 -> 3.18 s (-15 %) vs STRIX_GLM_PF_EXACTMOE_V=7 in the same run (two pairs each, 8192-row chunks); MoE call v7 -> v11: 128 rows 17.2 -> 10.1 ms, 512 rows 29.2 -> 16.9 ms, 4096 rows 127 -> 75 ms; v3 row buckets -18 to -22 % on calls of 33-100 rows
- **Implementation / evidence:** [resolver](../apps/glm_rank_defaults.hpp), [MoE](../src/glm/prefill/glm_pf_exact_moe.hip), [tests](../tests/hip_glm_pf_exact_moe_tests.hip)

<a id="glm-pf-exact-x"></a>

### Exact prefill X3 linear core

**Engine:** `glm`. **Control:** `STRIX_GLM_PF_EXACT_X`. **Default:** auto.

Register-blocked exact prefill linear: LDS-decoded weights reused over 64 rows, activations read with scalar loads, four outputs x eight rows per lane (X3; X2 = LDS activations over 32 rows with next-stage register prefetch).

- **Preconditions:** Exact chunked prefill linears; auto takes the X3 core with eight rows per wave for calls of at least 2048 rows (STRIX_GLM_PF_EXACT_X_MIN_ROWS) and the TM16 path below. 1, 2, 3 or 4 force X, X2, X3 (four rows per wave) or X3 (eight) for every call.
- **Automatic selection:** Unset resolves to auto; the app sets it; the resolver reports it in `--print-effective-config`. Explicit values win; `--baseline` sets 0.
- **Fallback notice:** Below the row threshold the TM16 path runs; invalid values fail. No silent numerical-identity substitution.
- **Exactness evidence:** Bitwise: X/X2/X3 keep the TM path's DOT2 chains, fold and split order (X3 reads the wave-uniform activation pairs with scalar loads; LDS holds only decoded weights); dense outputs byte-equal TM for row tails 1..32, 2047/2048/2049 and 4096-row calls, signed zero, FP16 subnormals, +-Inf and NaN payloads; on the real pair decode anchors, speculative oracle and 1K/8K/32K prefill logits equal the X=0 path.
- **Off switch:** STRIX_GLM_PF_EXACT_X=2 (previous X2 core) or 0
- **Measured gain:** X3: 8K TTFT 21.42 -> 18.37 s (-14.3 %) vs STRIX_GLM_PF_EXACT_X=2 in the same run (two pairs, MoE v10 in both arms); X2 earlier: 8K 29.59 -> 27.93 s (-5.6 %) vs X=0; logits bitwise on both ranks
- **Implementation / evidence:** [resolver](../apps/glm_rank_defaults.hpp), [X core](../src/glm/prefill/glm_pf_exact_x.hpp), [tests](../tests/hip_glm_pf_exact_tests.hip)

<a id="glm-pf-obar-v5"></a>

### Prefill MLA o-bar v7

**Engine:** `glm`. **Control:** `STRIX_GLM_PF_OBAR`. **Default:** 7.

Exact MLA probability-weighted latent sum (o-bar) with the same lane chains and 256-lane pairing tree as v5, placed as
4 heads x 4 latent dims per thread: the probabilities are wave-uniform and come through scalar registers, and the latent
is widened to f32 once per block.

- **Preconditions:** Exact chunked prefill, 32 MLA heads per rank (64 heads use v2).
- **Automatic selection:** Unset resolves to 7; the engine default provides it; the resolver reports it. Explicit values win.
- **Fallback notice:** 64 heads run v2 (same bits); invalid explicit combinations fail.
- **Exactness evidence:** Bitwise equal to v5 and to the decode kernel on the prefill gate (probabilities, o-bar, outputs; NaN/Inf/subnormal latents, list lengths 1..2051).
- **Off switch:** STRIX_GLM_PF_OBAR=5 (v5) or =2 (v2)
- **Measured gain:** 9.0-10.2 vs 16.6-17.3 us/row per DSA layer at chunk positions 4096..28672 (6.6 vs 11.3 at 0); v5 was 2.9x v2
- **Implementation / evidence:** [24-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-pf-mla-gemm-v2"></a>

### Prefill MLA absorb / output GEMM v2

**Engine:** `glm`. **Control:** `STRIX_GLM_PF_MLA_GEMM`. **Default:** 2.

The MLA query absorption and output projection GEMMs keep every output's chain (k ascending from +0) with a 128-row x
64-column tile, A staged transposed and the F16 weight widened to f32 once per tile, 8 x 4 outputs per thread.

- **Preconditions:** Exact chunked prefill (the legacy MLA pair keeps its own kernels).
- **Automatic selection:** Unset resolves to 2; the engine default provides it; the resolver reports it. Explicit values win.
- **Fallback notice:** None.
- **Exactness evidence:** Absorbed queries and outputs bitwise equal to v1 and to the decode kernel on the prefill gate.
- **Off switch:** STRIX_GLM_PF_MLA_GEMM=1
- **Measured gain:** absorb 6.1 -> 4.6 and output 4.1 -> 3.3 us/row per DSA layer (32 heads per rank)
- **Implementation / evidence:** [resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-pf-scores-v3"></a>

### Prefill MLA scores v6

**Engine:** `glm`. **Control:** `STRIX_GLM_PF_SCORES`. **Default:** 6.

Exact MLA score kernel (same bits as v5, v4, v3 and v2): one lane per two list entries with every head's chain in
registers, the absorbed query read through scalar registers, the entries' latent straight from memory. v6 issues v5's
inner step as hand-paired dual FMAs (the two entries of a lane share each query register; compact register layout, four
heads interleaved per block, generated by `tools/gen_glm_scores_vopd.py`). Heads 64 keep v5.

- **Preconditions:** Exact chunked prefill and supported score geometry.
- **Automatic selection:** Unset resolves to 6; the resolver reports it. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Bitwise equal to v3 on the kernel fixtures; complete same-identity tokens plus logits/state digests on the pair.
- **Off switch:** STRIX_GLM_PF_SCORES=5 (v5), =4 (v4), =3 (v3) or =2 (v2)
- **Measured gain:** v6 3.40 / 8.27 / 10.38 vs v5 4.57 / 9.27 / 11.39 us/row per DSA layer at chunk positions 0 / 4096 / 28672 (-26 / -11 / -9 %); v5 was 8.6-11.5 vs 14.0 for v4 at 4096..28672, v4 -8..-17 % vs v3, v3 -22 % vs v2. Rank 0 of a 32K pair prefill 2.94-2.99 -> 2.61-2.63 s in scores. Together with KDA scan v3 on the two-node pair: 32K TTFT 68.76 -> 67.72 s (-1.5 %, mean of both run orders), logits bitwise equal on both ranks
- **Implementation / evidence:** [resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-pf-kda-scan-v3"></a>

### Prefill KDA scan v3

**Engine:** `glm`. **Control:** `STRIX_GLM_PF_KDA_SCAN`. **Default:** 3.

The token-sequential KDA recurrence of a prefill chunk keeps every lane's chains and the 128-key pairing tree; the two
tree levels that cross lanes of a quad (q ^ 2, q ^ 1) read the partner's value through quad DPP moves instead of
`ds_bpermute` shuffles, which takes four LDS round trips off every token's serial path.

- **Preconditions:** Exact chunked prefill.
- **Automatic selection:** Unset resolves to 3; the resolver reports it. Explicit values win.
- **Fallback notice:** None.
- **Exactness evidence:** Raw scan outputs, outputs and states after every chunk bitwise equal to v1 and to the decode kernel on the prefill gate (real and synthetic weights, special rows).
- **Off switch:** STRIX_GLM_PF_KDA_SCAN=1
- **Measured gain:** 1.84-1.93 -> 1.23-1.28 us/row per KDA layer at chunk positions 0 / 4096 / 28672 (-33 %); rank 0 of an 8K pair prefill 0.534 -> 0.355 s in the scan, of a 32K one 2.15-2.22 -> 1.43-1.44 s; 8K TTFT 16.63 -> 16.34 s (-1.7 %, alone). Together with MLA scores v6: 32K TTFT 68.76 -> 67.72 s (-1.5 %, mean of both run orders), logits bitwise equal on both ranks
- **Implementation / evidence:** [resolver](../apps/glm_rank_defaults.hpp)

### Prefill pipeline and async exchange start

**Engine:** `glm`. **Control:** `STRIX_GLM_PF_PIPELINE`, `STRIX_DIST_ASYNC_START`. **Default:** 2 / 1.

Overlap two whole layers of the chunked prefill, and start the host exchange asynchronously so it hides behind compute
(only the pipeline uses the async start; other engines keep the ordinary start unless they set the variable).

- **Preconditions:** Exact chunked prefill on two ranks; full-host chunked exchange path.
- **Automatic selection:** The rank app sets both when unset; explicit values win.
- **Fallback notice:** Other exchange paths use the ordinary start.
- **Exactness evidence:** Prefill gate bitwise to the reference row-by-row prefill (token, logits, all state bytes); TTFT logits equal on both ranks at 1K/8K/32K.
- **Off switch:** STRIX_GLM_PF_PIPELINE=1, STRIX_DIST_ASYNC_START=0, or --baseline
- **Measured gain:** 8K about -4 %; the exchange is fully hidden (GPU idle per 2K prefill 0.97 to 0.09 s)
- **Implementation / evidence:** [resolver](../apps/glm_rank_defaults.hpp)

### Prefill chunk rows 4096

**Engine:** `glm`. **Control:** `--max-rows`. **Default:** 4096 fallback when the automatic 8192-row budget does not fit (baseline 1024).

Process prefill in 4096-row chunks.

- **Preconditions:** Exact chunked prefill.
- **Automatic selection:** See [memory-admitted chunk rows](#glm-max-rows-auto); `--max-rows 4096` or
  `STRIX_GLM_MAX_ROWS_AUTO=0` fixes this size; `--baseline` uses 1024.
- **Fallback notice:** None.
- **Exactness evidence:** Logits equal to the reference prefill on both ranks at 1K/8K/32K.
- **Off switch:** --max-rows 2048 / 1024, or --baseline
- **Measured gain:** with the pipeline on: 8K 35.1 s at 1024, 32.1 s at 2048, 30.1-30.8 s at 4096; 32K 136.7 to 128.4 s vs 2048
- **Implementation / evidence:** [resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-mtp-spec"></a>

### MTP self-speculation

**Engine:** `glm`. **Control:** `--mtp; --mtp-spec; STRIX_GLM_MTP_SPEC; STRIX_GLM_SPEC_XCH; STRIX_GLM_SPEC_KDA_TREE`. **Default:** on (Xfer verify exchange, KDA tree 2, adaptive K).

The model's own MTP layer drafts tokens that the target verifies in the normal `--decode` path; the prompt fills the MTP
cache in chunks during prefill. Committed tokens are the target's greedy tokens.

- **Preconditions:** Two-rank TP2 decode with the model's MTP layer; both ranks must agree on the configuration (checked at start).
- **Automatic selection:** The resolver item mtp-spec is on unless switched off; `--print-effective-config` shows `mtp-spec=1`.
- **Fallback notice:** A rank disagreement on the speculative configuration stops at start instead of decoding.
- **Exactness evidence:** Full token vectors and prompt logits equal --mtp 0 on 7 prompts, both ranks; MTP-fill prefill equals the row-by-row bootstrap; speculative runs exact in every arm with identical adaptive decisions on both ranks.
- **Off switch:** --mtp 0 (or --mtp-spec 0, STRIX_GLM_MTP_SPEC=0, --baseline); STRIX_GLM_SPEC_XCH=comm and STRIX_GLM_SPEC_KDA_TREE=1 restore the previous verify exchange and tree
- **Measured gain:** Decode chat / code / JSON 40.3-40.6 -> 26.98 / 22.46 / 24.98 ms/token (1.50x / 1.79x / 1.62x); cost: TTFT 8K +2.9 %, 32K +8.7 % (MTP cache fill)
- **Implementation / evidence:** [resolver](../apps/glm_rank_defaults.hpp), [spec runtime](../src/glm/spec/glm_spec_run.cpp)

`STRIX_GLM_HC_NORM=1` and `STRIX_GLM_MOE_PIPE=1` are included as opt-in experiments (default off, not yet gated).

<a id="glm-pf-indexer-v2"></a>

### Prefill indexer scores v3 and select v2

**Engine:** `glm`. **Control:** `STRIX_GLM_PF_IDX`, `STRIX_GLM_PF_SELECT`. **Default:** 3 / 2.

Scores v3: 8 rows share the pooled keys staged once per block, and each row's queries and head weights come through
scalar registers (same dot and head chains as v2). Select v2: the same top-512 lists (ties by ascending pool, pools
ascending, then the tail) with a parallel radix digit search and ballot compaction instead of a serial digit walk.

- **Preconditions:** Exact chunked prefill; only rows with more than 512 pools score and select.
- **Automatic selection:** Unset resolves to 3 / 2; the engine default provides them; the resolver reports them. Explicit values win.
- **Fallback notice:** None; unknown explicit values fall back to the default.
- **Exactness evidence:** Keys, scaled head weights, token lists and counts bitwise equal to scores v2 / select v1 and to the decode indexer on the prefill gate.
- **Off switch:** STRIX_GLM_PF_IDX=2 (v2) or =1 (v1); STRIX_GLM_PF_SELECT=1
- **Measured gain:** at chunk position 28672 (8192 pools): scores 5.3 vs 9.6, select 0.38 vs 0.99 us/row per DSA layer; at 4096: 1.17 vs 1.81 and 0.08 vs 0.50
- **Implementation / evidence:** [24-item resolver](../apps/glm_rank_defaults.hpp)

## Platform (kernel)

<a id="platform-perfopt"></a>

### GPU IOMMU PerfOpt

**Engine:** platform. **Control:** `amdgpu.iommu_perfopt` (kernel module parameter). **Default:** on: the lab nodes boot the PerfOpt kernel by default (permanent GRUB default on both nodes; `iommu_perfopt=-1`, the kernel logs "PerfOpt armed").

The integrated GPU reaches system memory without IOMMU translation; the IOMMU stays on and every other device,
including the USB4 controllers, keeps its translated domain. Trade-off: the GPU's own DMA is no longer contained
by the IOMMU.

- **Preconditions:** Kernel with AMD's PerfOpt patches (Linux 7.4 or the IOMMU tree) and the GPU in an identity domain; dedicated inference host without GPU unbind, VFIO, kexec/kdump or suspend.
- **Automatic selection:** On by default in a kernel that carries the patches; nothing in the engine selects it.
- **Fallback notice:** None at engine level; the kernel logs "PerfOpt armed" when active.
- **Exactness evidence:** Same kernel build, PerfOpt on vs amdgpu.iommu_perfopt=0, interleaved boots: tokens identical in every arm; GPU identity domain, USB4 NHIs still DMA-FQ, iommu_dma_protection=1, no AMD-Vi fault lines after two-node traffic.
- **Off switch:** `options amdgpu iommu_perfopt=0`, or boot the previous kernel
- **Measured gain:** Qwen 32K prefill +2.9 % / +3.1 % (two repeats, one node, same power, equal or lower GPU clock); DeepSeek two-node decode within +/-1 % with one node on PerfOpt, both-node run pending
- **Implementation / evidence:** [driver repo: PerfOpt](https://github.com/hyukudan/strix-halo-usb4-driver/blob/main/docs/PERFOPT.md)

<a id="qwen-moe-route-bf16"></a>

### MoE route output BF16 storage candidate

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_MOE_ROUTE_BF16=1` or
`--prefill-moe-route-bf16 on`. **Config key:** `prefill_moe_route_bf16`.
**Default:** OFF. **Status:** candidate; bit-changing; outside READY; needs qualification.

Rounds each DA16 FP32 route dot to BF16 with round-to-nearest-even, preserving NaNs
and infinities. The ordered merge widens each dot exactly to FP32 and retains the
existing slot-order FMA chain, nonfinite status checks, commit and output format.
Weights, activations and route weights retain their existing arithmetic.

- **Covered route:** full-row known-prefix fast FP16 MoE prefill, Q4_K/Q5_K gate/up,
  Q5_1/Q8_0 DA16 certificate down, effective tunes 80/82/88 (including XOR rows), ROUTE=14 and MC=1.
  The default vector merge/commit and scalar route (`STRIX_PREFILL_MOE_MERGE_VEC=0`)
  both widen BF16 in the admitted and device-side merge-zero/finish fallback branches.
  Native and YaRN long-context plain CLI routes are covered; this
  storage control adds no rotary site and leaves the YaRN rotary identity unchanged.
- **Declines:** other tunes, core/G3/P1, SMALL, non-DA16 F16X, GSQ, plain/decode and
  known-rows organizations, MTP, nested/serving, cache/session routes and CUDA.
  Unsupported geometry or weight formats are refused before execution. The ordinary
  serial suffix and subsequent decode retain their decode profile and FP32 route scratch.
- **Identity and inspection:** adds component bit 64 outside the default component set;
  the profile-pair hash changes. The storage descriptor is `fp16-v1+moeroute16`;
  OFF retains `fp16-v1` (`a0094634843e16ee`). `--print-effective-config` reports the
  requested environment, effective boolean, storage descriptor/hash and decline or
  suppression reason. HTTP capabilities explicitly report this candidate unsupported.
- **Memory:** logical route-dot row size and slot pitch halve. The arena allocation
  remains FP32-sized with its original per-row arena pitch; admission is unchanged.
- **Off switch:** `STRIX_QPF2K_MOE_ROUTE_BF16=0` (any value beginning with `0`),
  `--prefill-moe-route-bf16 off`, `--baseline`, `STRIX_BASELINE=1`, or
  `STRIX_QPF2K_DEFAULTS=0`. CLI selection overrides the candidate environment value;
  baseline and the master off switch suppress both.
- **Measured gain:** same build, plain CLI on one node, OFF vs ON in three alternating pairs:
  32K TTFT median **21.83 → 21.15 s** (per pair −0.78 / −0.68 / −0.66 s, mean **−3.2 %**),
  16K **10.38 → 10.11 s** (−0.19 / −0.27 / −0.27 s, mean **−2.3 %**); every pair faster,
  first generated token equal. The receipt reports the candidate active on the default route.
  262K YaRN (factor 2, serial route, 262,144-token prompt, `--profile default` vs `--profile fast`,
  two interleaved pairs, one run per arm): prefill **191.55 / 190.89 → 186.78 / 184.58 s**, mean
  **191.2 → 185.7 s (−2.9 %)**; decode 20.55-20.60 tok/s in both.
- **Qualification:** CPU option/identity/suppression tests and the native HIP fixture
  (scalar and vector merge, tunes 80/82/88, nonfinite status) pass. Model-fidelity
  qualification against its own references is pending; until it passes the candidate stays
  opt-in outside the `default` and `quality` profiles, and `--profile fast` applies it as a
  provisional row. YaRN model-fidelity and complete-workload qualification remain pending;
  host admission does not establish them.
- **Native fixture:** `ctest -R strix-hip-moe-route-bf16-tests --output-on-failure`
  covers both down formats, all three tunes, scalar/vector tails, top-k ten, padded tails, RNE ties, subnormals,
  infinities, NaNs, OFF byte equality and both merge/commit branches.

## HC low-band layout

**Default ON (mode 3)** through decoder qpf2k defaults.
OFF: `STRIX_QPF2K_HC_LOWBAND=0`; restart. Baseline/defaults-off and injected
GSQ suppression follow the defaults table. Banded transposed low_t changes only
addresses; projection stage/WMMA/injection order and the shared output rank chains
are unchanged. Mode 3 also omits low16 stores only when no consumer reads them.
The route guard admits projection form 4 and FP32 output v3 band 16; output WMMA,
check mode, normalized16, inverse and other output/projection routes retain mode 0.
At at most 512 padded rows only the unused-store omission applies. Numerical,
kernel-plan and cache identities are unchanged. Native local CI passed 132/132
cases with no stray stores, including eight T8224 cases with 512-row pool rounding.
At 32K, last-row logits match mode 0 bitwise with `hc_output_wmma` OFF. Exact,
bitwise-checked; no timing claim. Low-band is inactive on the default route.

## PLE staged-gather acceptance fix

Unconditional correctness fix; no OFF switch. Successful independent row-0
consumption permits acceptance of the complete ordered deferred tail. Every row
and head ID still matches, including row 0. Failed staging and invalid IDs retain
the inline copy work list. Receipt staged_rows counts the full accepted block.
Copy semantics, numerical identities, worker-error fallback and slot events are
unchanged. CPU acceptance/fallback regression coverage is provided by
`strix-ple-staged-gather-tests`; native local CI passed the PLE launcher and staged-gather
fixtures. The staged versus `STRIX_PLE_STAGE_PREFETCH=0` gathered-byte comparison
remains open and nonblocking; the fixtures do not establish that comparison.

## Exact bundle control and reservation contract

Baseline (`--baseline` or `STRIX_BASELINE=1`) and `STRIX_QPF2K_DEFAULTS=0`
suppress early setup, eager loading, automatic T16384, compact images and
index batching. Baseline also suppresses low-band layout. An existing
`HIP_ENABLE_DEFERRED_LOADING` value remains under user control. The exact-FP16
master controls arithmetic options; it does not disable exact organization items.
Low-band mode 3 is live only on the exact-FP16 output route with
`hc_output_wmma` OFF, not on the default output WMMA route.

Pool capacities round up to 512 rows, preserving the historical 8192-row floor.
Explicit T above 8192 must be a multiple of 32; T8224 is supported. All automatic
T routes price the simultaneous native pools in UMA and GTT admission, including
optional HC normalized16 (about 161 MiB at T8192), MoE A16/DA16 (140 MiB),
HC packs, QSA descriptors/tables/maxima, and pinned aliases. Explicit tiled routes
must supply `--known-prefix-host-headroom-bytes` covering the reported native
host and device bytes. Zero or otherwise insufficient headroom is refused, including
under `--baseline`: baseline does not restore the old explicit-route acceptance
because these pools are reserved regardless of bundle switches. These conservative
reservations can lower the admitted T even below the T16384 prompt threshold.
At 131,072 and 262,143 real-text tokens the default admits automatic T16384, and
with `STRIX_QPF2K_T16K=0` it admits T8192 as before, with no fallback.
Setup growth drains old scratch readers before replacement; admission prices
one additional largest-pool allocation for the replacement peak. Larger MTP
group owners are included in explicit-route pool sizing. Execution never resizes these pools. Optional QSA
stage allocation failure, compact-policy or capacity mismatch declines to the
per-group path before any stage reader is submitted. Tail owners retain the
shared high-water strides.

Eager loading can increase resident code-object memory; the amount depends on
the runtime and loaded kernels and is not a fixed workspace saving. Record it
with process startup memory and wall time. Early setup adds no image allocation.

Automatic index batching yields to explicit index rows OFF, index shape and
side-stream mask selection. With batching OFF, historical mask-bit scheduling
and shape controls apply. Zero-leading values consistently disable exact boolean
switches. T16384 and index batching are excluded from session numerical identity,
as are the other exact items; physical arena identities still describe storage.

`/v1/capabilities` and HTTP generation statistics report `prefill_exact_items`
separately from arithmetic identity: early setup, automatic T16384 and compact
images are excluded on HTTP; index batching and low-band report their policy
and route guard, eager loading reports the applied HIP policy, and the
unconditional PLE acceptance fix is disclosed. `--print-effective-config` also
reports `hc_lowband_policy` and `hc_lowband_route: "exact-fp16-output-only"`;
the resolved policy can be nonzero while inactive on the default output WMMA
route. It is not a kernel-launch witness.
`index_proj_wmma` stays opt-in outside READY and is excluded from MTP and nested
scheduling whenever its fast QSA component is selected.

The PLE fix accepts the ordered deferred tail after independent row-0
consumption. The previous path re-gathered the tail inline; both copy the same
bytes. Every staged row ID, including row 0, remains checked before acceptance.
Recorded native local-CI results and open, nonblocking gates are listed
in [bundle qualification](engine/prefill/hc-output-wmma-qualification.md#bundle-local-ci-evidence).
