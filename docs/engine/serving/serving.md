# Serving over HTTP

`strix-hip-decode-token --http-serve` loads the model once, qualifies the MTP predictor once and
then serves OpenAI-compatible and Anthropic-compatible HTTP requests until it receives SIGINT or
SIGTERM. Every request runs the same native path as a command-line request: tiled prompt prefill,
MTP speculative decode with target verification, greedy selection. Output tokens are those of the
command-line route for the same prompt and flags, up to the first stop token (the command line stops
only at `--eos-token`, by default `<|endoftext|>`; the server also stops at `<|im_end|>`).

The HTTP front end (`strix_serve`: JSON, tokenizer, chat template, HTTP server, request queue and the
API routes) is host-only code with no GPU dependency; its contracts are tested without a GPU.

## Starting the server

Serving runs on the tiled MTP route, so it takes the flags of a tiled MTP request (see
[prefix-state-cache.md](prefix-state-cache.md#mtp) and [mtp-native-cli.md](../speculative/mtp-native-cli.md)) plus:

| Flag | Default | Meaning |
|---|---|---|
| `--http-serve` | off | serve requests after the calibration replay instead of running the command-line request |
| `--http-bind ADDR` | `127.0.0.1` | listen address (numeric) |
| `--http-port N` | `8080` | listen port (`0`: any free port, printed on stderr) |
| `--http-model-id NAME` | `qwen3.8-flash-next` | model name reported by `/v1/models` and responses |
| `--http-api-key-file FILE` | none | require `Authorization: Bearer KEY` or `x-api-key: KEY` on the API routes |
| `--http-batch N` | AUTO | omitted: as many request slots as memory admits ([below](#batched-decoding)); `N`: at most N; `0`: one request at a time |
| `--http-sampling greedy\|reject\|sample` | `greedy` | requests with `temperature > 0`: run greedy and say so in the response, answer 400, or sample (sampling-v1, batched route only) |
| `--http-default-temperature X` | `0` (built-in) | Qwen HTTP default in [0, 2]; positive values require `--http-sampling sample` |
| `--http-default-top-k N` | `0` (built-in) | Qwen HTTP default non-negative integer, at most 4294967295 |
| `--http-default-top-p X` | `1` (built-in) | Qwen HTTP default in (0, 1] |
| `--http-default-min-p X` | `0` (built-in) | Qwen HTTP default in [0, 1) |
| `--http-default-max-tokens N` | `4096` | generation budget when a request gives none |
| `--http-max-queue N` | `65536` | waiting requests before new ones get 429; explicit small limits retain rejection |
| `--http-max-connections N` | `4096` | open connections before new ones get 503; independent of execution slot occupancy |
| `--http-responses on\|off` | `on` | the stateless `POST /v1/responses` route ([below](#responses-stateless)); `off`: 404 |
| `--http-capabilities on\|off` | `on` | `GET /v1/capabilities` and its `capabilities_id` in `/health`, `/v1/models`, `/api/status`; `off`: 404, no id |
| `--http-stream-receipts on\|off` | `on` | the applied-policy `strix` object at the start and end of every stream ([below](#applied-policy-and-keepalive)); `off` together with `--http-sse-keepalive 0 --http-sse-progress-interval 0`: the previous build's bytes |
| `--http-anthropic-thinking-budget enforce\|note` | `note` | Messages `thinking.budget_tokens`: `note` accepts it and states in `strix.note` that it is not enforced (the request keeps speculative decoding); `enforce` applies `thinking-budget-v1` (resolved host selection route) |
| `--http-thinking-budget on\|off` | `on` | makes the opt-in `thinking-budget-v1` policy available ([below](#thinking-budget)); `off`: explicit policy requests get 400 |
| `--http-selection-mtp on\|off` | `on` | policy selection uses admitted serial-equivalent MTP windows; `off` forces `serial-host` for JSON, budgets, scores and transforms |
| `--http-selection-fast on\|off` | `on` | argmax-first policy selection: rows whose unmasked argmax the constraint admits skip the mask walk (and, on `mtp-host`, the logits readback); `off` walks the full mask for every row |
| `--http-logprobs on\|off` | `on` | Qwen host logprobs; `off` refuses score requests |
| `--http-logit-transforms on\|off` | `on` | Qwen opt-in logit transforms; `off` refuses non-neutral transforms |
| `--http-constraints on\|off` | `on` | constrained JSON output ([below](#constrained-json-output)); `off`: such requests get 400 and capabilities say unavailable |
| `--http-sse-progress-interval SECONDS` | `1` | coalesced completed prefill progress as SSE comments; `0`: none; request opt-out: `"strix":{"progress":false}` |
| `--http-sse-keepalive SECONDS` | `10` | SSE comment `: keepalive` after this long without a write while a stream waits (queue, prefill, decode); `0`: none |

The token IDs on the command line are the **calibration prompt**: its first
`--mtp-calibration-rows R` pairs are replayed against the golden (`--mtp-golden`) to qualify the
predictor, exactly as for a command-line request. Requests do not need to share anything with it.
`--generate` is ignored while serving. The prefix state cache flags (`--prefix-cache-host-bytes`,
`--prefix-cache-every`, `--prefix-cache-dir`) apply across requests. Certified prefix
caching is ON by default only for the qualified single tiled-MTP HTTP runner
(`--http-batch 0` or auto capacity resolving to single), T=2048, default numerical
profile/schedule/tails and model assets, context up to 16384. Qualified results are
bitwise identical to cache-off. Explicit host bytes retain their existing behavior.
`--prefix-cache off` disables prefix caching; `--prefix-batch-cert off` and
`--session-cache off` disable their separate policies. In-flight prefix joining is
on by default wherever the cache is on and only acts on concurrent strict-L0
certified batch routes (inert on the single runner); `STRIX_PREFIX_CACHE_INFLIGHT=0`
or `--prefix-cache-inflight off` turns it off.

The prompt may run a separate prefill profile (`--prefill-numerical-profile`, see
[prefill-numerical-profile.md](../prefill/prefill-numerical-profile.md)), for example the fast prefill profile
`fast-prefill-f16wmma-v1` with the dense-A8 decode profile and `--known-prefix-prefill-rows 8192` (T
above 4096 needs `--known-prefix-host-metadata-budget-bytes` above the plan's metadata, about 286 MB
at T = 8192). Its complete tiled blocks run the prefill profile; the calibration rows, the serial
suffix and decoding run the decode profile, and decoding is exact from the state the blocks produced.
The calibration golden is that of the decode profile. The prefix state cache keys its checkpoints by
the pair, both executions and T.

Before any device work the process prints one `strix-build-info` and one `strix-startup-event`
JSON line on stderr: configured build identity and a read-only memory preflight (GTT/VRAM, host
memory, 2 MiB-equivalent free blocks, compaction counters, memlock, KFD clients), with warnings derived
only from known plan terms ([startup-preflight.md](../../operations/startup-preflight.md); default ON,
`STRIX_PREFLIGHT=0` / `STRIX_RELEASE_EVENTS=0` to switch off).
The process prints `strix serve: listening on ADDR:PORT` on stderr once it accepts requests, and a
one-line JSON receipt (`"mode":"native-mtp-http-serve"`, the number of served requests) on stdout
when it stops.

## Default prompt prefill

On the qualified native Qwen checkpoint and target, HTTP serving now resolves a known-prefix
T8192 owner before requests arrive. Keep the required MTP calibration prompt, golden, sidecar,
pins and window budgets; tiled-prefill organization flags and budgets can be omitted. Unlike
single-request auto prefill, the server cannot choose T from an incoming prompt at startup.
It preserves the resolved decode/prefill numerical profiles and explicit context capacity;
this does not enable automatic FP16 arithmetic. Request prompts must still fit that capacity.

Automatic candidates that would create an extra `STRIX_PROMPT_BATCH_ROWS` owner decline
with an actionable notice before allocation in both startup and inspection. Unset that
variable or supply explicit T and the existing group workspace/metadata budgets.
A value at or below the selected T creates no extra group owner.

The resolver compiles and sizes the primary arena and all configured tail owners, metadata,
model backing, state, serial scratch, runtime payload and at least one runnable MTP slot.
For automatic native serving, target backing means the page-aligned registered extents,
including coalesced gaps, using the same range planner as native registration. These
aliases consume physical UMA once and GTT once. Unregistered mapped file cache (the
CPU-read PLE table) is reclaimable and does not constitute a committed UMA allocation;
its native row staging and snapshots remain charged. This changes admission accounting,
not the mapping, warming policy or arithmetic. Inspection reports `target_mapped_file_bytes`,
`target_registered_backing_bytes` and `serve_committed_host_peak_bytes`. The generic
model residency reservation still describes the conservative full mapped backing.
The shared startup/inspection/runtime calculation uses verified sidecar/calibration asset sizes,
the sidecar residency planner (including copied backing and setup peak), native predictor pool,
target capture, target-window arenas/captures, deferred banks, selector and calibration payloads.
Native primary/tail PLE staging, projections, gates, snapshots, hashes and deferred GDN banks
are reserved separately from the arenas. Routed CSR uses the production layout for
fixed primary/tail allocations and conservatively bounds every retained partial row
count from 16 through T in steps of 8, per owner. Runtime retains distinct workspaces
for those counts; this bound does not assume which merged/grouped units execute.
Batch-certificate staging is a shared reserve derived from the target registry and predictor/held state; an explicit staging cap takes precedence.
MTP budget ceilings are limits, not simultaneous allocations; their sum is never charged.
Automatic T declines target `--residency copy`, selective `--model-copy-tier` and
`STRIX_MODEL_COPY_ALL` with a notice; preserve those requests and supply an explicit tiled route.
The default workspace budget is the compiled primary-plus-tail requirement: serving retains uniform
logits storage and all tail owners, so the CLI's compact/pruned 8 GiB limit does not apply. Default limits
are 1 GiB metadata, at most 120 GiB UMA bounded by observed available memory, and 4 GiB host
headroom. Explicit budgets win. Both available host memory and
free GTT must fit. Context or memory/plan rejection tries T4096, T2048, T1024, T256, T64 and T16;
the `Qwen serve-prefill:` startup line records each trial's exception text (or the failed memory
bound with required/limit bytes) and the selected T. Missing memory
observations decline automatic allocation. If no tiled owner is admitted, the server reports
the decline and requires an explicit valid tiled route before device work. Explicit T is never
resized. Keep `--http-serve` and the same MTP arguments when adding `--print-effective-config`:
the dry run uses the same CPU prefill admission and HTTP/cache route validation, reports
`serve_prefill_notice` in JSON, and exits nonzero on a route rejection. Omitting `--http-serve`
inspects a CLI route. No device work occurs during inspection, so it cannot predict allocation
failures or the final HTTP slot count (`http_slots.slots` remains null for a batched route).
Later HTTP slot admission refuses zero capacity, including explicit `--http-batch 0`, measures stack costs, reserves active peaks and
headroom, and rechecks available memory before lazy growth; a larger shared prefill arena can
reduce admitted slot capacity. The default context remains 8192; longer contexts require the
existing explicit context contract and do not acquire a new qualification claim.

For a configured prefix cache, omitted `--prefix-cache-block-rows` selects 1024 when it divides T.
This changes checkpoint granularity, not automatic cache-budget qualification (which remains
off). `--prefix-cache-block-rows 0` restores T-sized checkpoints; another valid explicit divisor
wins. `STRIX_BATCH_PROMPT_SMALL=1` (tail-size prompt units between decode cycles)
remains opt-in for HTTP batch runners and CLI cohorts: with several long prompts queued it
delayed first tokens (TTFT p50 +37 % at 4 and +60 % at 8 concurrent 8.5K prompts) and is not
part of this default.

`STRIX_SERVE_PREFILL=0` disables automatic T selection; supply the previous explicit tiled
configuration to serve with it. `--known-prefix-prefill-rows T` and its explicit budgets always
win, including with that environment switch set. `--baseline` / `STRIX_BASELINE=1` suppresses
the automatic T and the 1024-row checkpoint default. Inspect `--print-effective-config` for T,
`serve_automatic_prefill` and `prefix_cache_block_rows`.

GPU service, nesting and slots gates reported EXACT against sequential explicit T2048: complete
IDs and committed target state matched; service/nesting also checked prompt state, and slots
covered 1/2/4/8 clients. Every default startup selected automatic T8192. These gates do not
compare logits. Same binary, interleaved arms (ABBAAB), identical token IDs, automatic T8192
versus `STRIX_SERVE_PREFILL=0` with explicit T2048, one request at a time, three rounds per
arm: cold TTFT 21.95 versus 22.93 s at 17K (-4.3 %) and 11.08/11.11 versus 11.53/11.58 s for
two 8.5K prompts (-3.9 %, -4.0 %); every automatic arm was faster than every explicit arm.
With eight staggered 8.5K prompts queued: TTFT p50 65.3/65.4 versus 66.4/71.7 s, p95
100.0/99.8 versus 101.4/106.3 s, wave wall 103.8/103.6 versus 105.2/110.2 s; at four, TTFT p50
35.3/35.5 versus 34.4/34.7 s (within run-to-run spread). T8192 consumes capacity: admitted
slots were 29 versus 55 with the explicit T2048 configuration, a queueing tradeoff only above 29
concurrent requests. Pressure paths are covered by CPU tests. Keep receipts outside Git.
The exactness contract remains complete IDs/state at the same numerical identity, context
and frontier; FP16 prefill does not claim dense-A8 equality. Missing state diagnostics are not a pass.

For a cross-T reference, use the service gate's persisted sequential reference. Here `CMD` is
the executable, `PINNED_ARGS` is a shell array containing the same model, calibration IDs,
MTP assets/window budgets, numerical profile, context and optional cache host budget for every
run, with no HTTP or tiled-prefill flags. `OUT` must be outside the checkout. Use a long prompt
with multiple T8192 blocks plus a tail, and a distinct short prompt:

```bash
REF_PREFILL=(--known-prefix-window --known-prefix-full-rows --known-prefix-expert-union
  --known-window-router-rows --known-prefix-prefill-rows 2048
  --known-prefix-workspace-budget-bytes 8589934592
  --known-prefix-host-metadata-budget-bytes 1073741824
  --known-prefix-uma-budget-bytes 128849018880
  --known-prefix-host-headroom-bytes 4294967296)
python3 tests/batch_service_gpu_gate.py --arms alone \
  --short SHORT.ids --long LONG.ids --out "$OUT/t2048" -- \
  "$CMD" "${PINNED_ARGS[@]}" "${REF_PREFILL[@]}"
python3 tests/batch_service_gpu_gate.py --reference "$OUT/t2048/alone.json" \
  --short SHORT.ids --long LONG.ids --out "$OUT/default-service" --service-ms 1 -- \
  "$CMD" "${PINNED_ARGS[@]}"
python3 tests/batch_service_gpu_gate.py --scenario nest --reference "$OUT/t2048/alone.json" \
  --short SHORT.ids --long LONG.ids --out "$OUT/default-nest" -- \
  "$CMD" "${PINNED_ARGS[@]}"
python3 tests/http_slots_gpu_gate.py --prompt SHORT.ids --prompt LONG.ids \
  --out "$OUT/default-slots" --modes auto -- "$CMD" "${PINNED_ARGS[@]}"
```

Require startup to report automatic T8192 for the default arms; a smaller admitted tile is
fallback coverage, not a T8192 witness. Use enough
memory for the slots gate's mandatory lazy-growth witness. No logits equality claim follows
from these commands.

## Execution model

- One engine thread (the decoder's thread) runs requests one at a time in arrival order. HTTP
  connections run on their own threads; they exchange tokens with the engine through a ticket that
  the engine fills after every verify cycle. Streaming therefore delivers tokens per cycle (one to
  eight tokens).
- Each request starts from a fresh target state and session, a fresh predictor and no pending
  verify-window rows, so its output does not depend on earlier requests. With the prefix state cache
  the longest cached block boundary of its prompt is restored first; restored tokens are reported as
  `usage.prompt_tokens_details.cached_tokens` (`cache_read_input_tokens` on the Anthropic route).
- Generation stops at a stop token (`<|im_end|>`, `<|endoftext|>` and `--eos-token` if given), at the
  request budget, at the context capacity, at a stop string, or when the client disconnects (the
  engine stops after the current cycle).
- A failure inside the engine leaves the target state unknown: the failing request gets an error,
  `/health` answers 503 and the process stops with a non-zero exit status.
  Opportunistic prefix capture is read-only: a checkpoint that cannot fit is a
  `budget_skips` miss. Only a typed pinned-allocation failure (including allocator
  `std::bad_alloc`) with a successful capture-stream drain drops partial buffers
  and increments `capture_failures` without failing the request. Destinations stay
  in a process-lifetime quarantine if completion or the drain returns an error.
  Quarantined owners are never freed; one line records static count/byte totals
  before the error propagates and the engine fail-stops. Successful drains release
  partial buffers normally. Restore source leases use the same contract. Every HIP
  fence, copy, completion or drain error propagates, including OOM. Baseline explicit-cache capture and explicit
  session checkpoint operations retain their original failure contracts.
- Decoding is greedy unless the batched route samples (below). A request with `temperature > 0` is decoded greedy
  and the response carries `strix.note`; with `--http-sampling reject` it gets a 400 instead.

Each sampling filter resolves independently: a non-null request value, then the configured server default,
then the built-in value (`temperature: 0`, `top_k: 0`, `top_p: 1`, `min_p: 0`). The four default-filter flags
are available on the Qwen command. The native DeepSeek pair keeps these built-in defaults but
executes positive-temperature requests with target-only sampling; temperature zero remains greedy.
Temperature, top-p and min-p must be finite numbers in their advertised ranges, even during greedy decoding.
Top-k must be a non-negative integer in uint32 range. Seed must be an integer in [0, 2^63); strings,
fractions and negative seeds are refused with a 400 naming the field before submission.
At temperature 0, filters do not change greedy argmax; non-neutral filters are accepted with the note
`filters ignored at temperature 0 (greedy)`. Positive temperature on a greedy-only engine retains the
configured downgrade or rejection policy.

A sampled request without a seed receives an API-generated 63-bit seed. The engine uses that seed and
the policy receipt echoes it. Reusing it reproduces the run with the same model, numerical identity,
sampler and position-keyed uniforms. Greedy receipts omit the seed.

### DeepSeek pair: target-only sampling

The native two-rank server executes positive-temperature requests using the shared
`sampling-v1` transform (temperature, top-k, top-p, min-p and a per-request seed).
Rank 0 samples the gathered target scores; its selected tokens are the subsequent
inputs on both ranks. The first output token is sampled too. Prefix lookups leave
the device prefill's minimum tail to recompute its scores instead of reusing a
cached greedy next token.

Sampled requests use plain target batches, without DSpark or prompt-lookup drafts.
Greedy requests retain their existing speculative route; mixed traffic is split
into separate steps. The numerical profile and model assets are unchanged. The
final receipt reports `sampling_effective: sample` and
`sampling_route: target-only-host`. No sampling-speed equivalence to speculative
greedy decoding is claimed.

The heterogeneous pipeline remains greedy-only and rejects positive temperature.
Sampled named-session checkpoints are not supported yet; they return an error.
Ordinary prompt-prefix caching remains available. This does not add logit biases,
repetition penalties or thinking-budget enforcement.

Host contract tests: `strix-deepseek41-sampling-tests` and the existing
`strix-token-sampling-tests`. Before promoting a deployment, exercise the live
two-rank endpoint (use at least two slots for the mixed-traffic gate):

```sh
python3 tests/deepseek41_sampling_http_gate.py --url http://127.0.0.1:8080 \
  --out /path/to/private/qualification --require-mixed
```

This checks the published greedy token reference, top-k-one equivalence, seed
replay and variation, cached-prefix and short-prompt paths, streamed token
identity, mixed greedy/sampled traffic, invalid parameters and cancellation.
Assess model-level repetition and task quality separately with representative
long contexts; successful sampling plumbing does not establish a model-quality gain.

## Batched decoding

Without `--http-batch` the server is batched with as many request slots as memory admits (AUTO). The batched
route needs prefix windows, the draft chain and window rows `{2,4,8}`; when they are missing, or a single-only
feature is set (state import, copy windows), AUTO serves one request at a time and says why on startup.
`--http-batch N` caps the slots at N (an explicit N with incompatible options is an error) and `--http-batch 0`
selects the one-at-a-time route. AUTO never changes the numerical profile or adds MTP assets.

- Sizing: every automatic T candidate reserves the compiled primary/tail arenas once, plus each native
  owner's separate PLE staging, projections, gates, convolution snapshot, hashes, deferred GDN banks
  and fixed/retained-variant routed CSR payloads.
  Host, GTT and UMA checks include one runnable bridge's geometry, verified assets, cache budget and shared
  batch-certificate staging (an explicit staging cap takes precedence), while retaining headroom
  (`--known-prefix-host-headroom-bytes`, 4 GiB unless set). A failed minimum retries a smaller T before
  allocation. Runtime uses the same staging policy and reserves independent prefill-owner buffers for
  additional slots over the shared arenas. Lazy growth uses fresh observations after shared
  bridge residency, reserving fixed headroom and only outstanding cache/staging payloads;
  it does not charge the shared setup peak again. It probes one extra slot when at least two fit and uses the
  larger of the static estimate and observed MemAvailable/GTT deltas. MTP budget ceilings remain bounds,
  not simultaneous allocations. Startup prints `http slots N (auto: limited by ...)` or `(cap C: ...)`;
  there is no fixed slot ceiling.
- Startup builds `min(N, STRIX_HTTP_EAGER_SLOTS)` slots (default 4; `0` builds all N). The rest are built one at
  a time between engine steps as soon as every built slot is busy (`STRIX_HTTP_SPARE_SLOTS=0`: only once work is
  queued), so a request arriving during a long prompt finds a free slot, through the same code as the eager ones,
  after re-checking memory (growth stops for the server, with a notice, when a slot no longer fits).
  A build takes 1.4-2.9 s (measured) and delays that step: it is the longest decode gap of a server that grows;
  `STRIX_HTTP_EAGER_SLOTS` trades startup time for fewer builds under load.
- Requests beyond the built slots wait in the queue and are never rejected for busy slots: responses carry
  `X-Strix-Http-Slots` (capacity) and `X-Strix-Queue-Position` (waiting jobs ahead at submission), the final
  `strix` receipt `queue_ms`. The queue and connection limits default to 65536 / 4096.
- Measured (one node, dense serving numerics, T8192, 2026-10-27): AUTO resolved 25 slots (limited by GTT),
  started with 4 and grew to 12 under 16 clients; every response equalled the `--http-batch 0` response in
  full token IDs and every committed target-state component.

Each admitted slot owns what a single request owns: target state and session, launch
plan, PLE backend and prefetch state, tiled prefill backends (the prefill arenas are shared; one prompt runs at a
time), prefix state cache view (over the shared store), predictor, target producer and its 2/4/8 window owners on one
window stream shared by all slots. The calibration replay runs once.

- Between decode cycles the engine thread takes queued requests into free slots. A joining request starts exactly
  like a request of the one-at-a-time route: fresh state and session, fresh predictor, its prompt through the tiled
  prefill, its first token from the prompt's last row.
- Its prompt advances one unit at a time (a T-row block, a tail block or 16 serial rows) between the decode cycles of
  the other requests, so that a long prompt does not stop them: one prompt unit, then one decode cycle.
  `STRIX_BATCH_PROMPT_SHARE=X` gives the decode cycles at least X times the time of the prompt units (more decode
  progress while prompts wait, later full cohorts); `STRIX_BATCH_INTERLEAVE=0` runs each prompt whole at admission.
- Long prompts do not stop the other requests inside a block either. Native T-row blocks (and prompt groups)
  run on a prefill worker thread that can pause at a completed layer: then the engine thread runs one decode
  round of the decoding requests (cohort cycle as usual) and the block continues with the same T, kernels and
  rows. A round starts when `STRIX_BATCH_SERVICE_MS` (default 250, `0` = off) of completed GPU prefill time, or of
  wall time since the last decode round, has passed; `STRIX_BATCH_SERVICE_LOOKAHEAD` (default 2) bounds how many layers the host
  enqueues ahead of the GPU. The prefill stream is drained before the round and the decode streams before the
  block resumes; nothing else (admission, prompt finishing, cache publication) runs inside a pause.
- Short prompts first: between prompt units a prompt with at most `STRIX_BATCH_SHORT_FIRST_ROWS` (default 2048,
  `0` = oldest first) rows left runs before longer ones.
- Nested admission: at a pause, also on an otherwise idle server, a queued or waiting request with at most
  `STRIX_BATCH_NEST_ROWS` (default 2048, `0` = off) uncached rows is prefilled completely on the engine thread
  through its usual 1024/256/64/16 tail units, which use their own arenas (never the paused block's), with
  decode rounds between its units; it then decodes with the others and the long block resumes. One nested
  prompt per pause; T8192 with the default tails and the dense-A8 prefill only (other routes decline, counted
  in `nest_skip_reasons`).
- Cancellation (`STRIX_BATCH_CANCEL_POINTS`, default 1) is observed at those layer boundaries: the request's
  partial block is discarded (nothing is published to the prefix cache), the shared prefill storage is reset
  and the slot starts fresh. Groups and window cohorts finish their unit first.
- `STRIX_BATCH_SERVICE_MS=0 STRIX_BATCH_CANCEL_POINTS=0 STRIX_BATCH_NEST_ROWS=0` runs the previous synchronous
  path (no worker). Receipts (`batch_service`, per request and in the shutdown batch line): `service_rounds`,
  `service_ms_suspended`, `max_bundle_gpu_ms`, `nested_admits`, `nested_rows`, `nested_ms`, `nest_skip_reasons`,
  `cancel_points_cancelled`.
- Exactness: every request's generated IDs and committed target state are bitwise those of the request alone,
  with or without pauses, nesting and cancellation of other requests. Only the MTP predictor's own rows can
  differ with concurrency (draft batching), as before; they never decide a token.
- Measured (3 decoding streams, a 32K prompt, two 1192-token arrivals, default command without `--http-batch`,
  2026-10-27): decode gap p90 during the 32K prefill 14.3 s -> 0.54 s, short-request first token 42-52 s -> 2.3-3.8 s
  (both nested), 32K prompt first token 53.1 -> 63.5 s (the time given to the other requests), identical token IDs;
  the longest gap left (5.6 s) is the spare slot build when the 32K prompt took the last built slot. With 6 slots
  built at startup the longest gap was 2.8 s (12.3 s without these features).
- A cycle decodes every request past its prompt: with two or more it is a cohort cycle (cohort windows, D-039: each
  request drafts up to 8 / requests rows, at least 2, and requests are packed into 4- and 8-row windows whose
  weight-reading steps run once for all rows), with one the ordinary cycle. With five to eight requests,
  `STRIX_COHORT_ROWS=16 STRIX_COHORT_WIDE=all` runs the cycle as one group of up to 16 rows whose GDN/QSA
  projections, LM head and expert union run once for all rows (D-052,
  [multi-request cohort](../scheduling/multi-request-cohort.md#global-row-budget-and-wide-steps-opt-in)).
  The experimental `STRIX_MTP_BATCH_JOINT=1` replaces that fixed per-request cap
  with a joint cost-based choice and supports shared dense steps through 32
  rows; see [joint serving windows](../scheduling/joint-serving-windows.md). It is off by default.
- A request leaves its slot when it ends (stop token, budget, context, stop string, disconnect), without stopping the
  others; its slot is free for the next queued request.
- Greedy requests return exactly the tokens they return alone (and on the one-at-a-time route): every request keeps
  its own state and arithmetic, and the shared steps compute each row as the request's own window would.
- `--http-sampling sample`: a request with `temperature > 0` samples with its own policy (`temperature`, `top_k`,
  `top_p`, `min_p`) and `seed` (random when absent), with speculative decisions on its drafts (D-040; on the GPU as
  in [sampled-decoding.md](../decode/sampled-decoding.md)). Its tokens follow the target distribution exactly; they are not
  the same tokens as alone, because the drafts per cycle depend on how many requests share the cycle.
- Statistics per request are those of the one-at-a-time route; `prefill_ms` runs from admission to the first
  token (it includes the decode cycles interleaved with the prompt), `queue_ms` from submission to admission.
  The receipt at shutdown is `"mode":"native-mtp-batch"` with slot, step and cohort window counts.

## Endpoints

| Method and path | Purpose |
|---|---|
| `POST /v1/chat/completions` | chat; `stream` with server-sent events |
| `POST /v1/completions` | raw prompt (text or token IDs); `stream` supported |
| `POST /v1/messages` | Anthropic-compatible messages; `stream` with typed events |
| `POST /v1/responses` | OpenAI Responses format, stateless; `stream` with typed events |
| `GET /v1/models` | the served model (with `capabilities_id`) |
| `GET /v1/capabilities` | what this server executes: endpoints, sampling policy and modes, stream features, unavailable features |
| `POST /tokenize`, `POST /detokenize` | `{"text"}` ↔ `{"tokens"}` with the model tokenizer |
| `GET /health`, `GET /v1/health` | Qwen [service health record](../../operations/service-health.md), with build/assets, resolved settings and backend evidence; `STRIX_SERVE_HEALTH_RECORD=0` restores the legacy body (including `capabilities_id` when enabled). Failed/draining: 503 |
| `GET /metrics` | Prometheus text: request counters, token counters, queue depth, busy time |
| `GET /api/status` | JSON snapshot (`"schema":"strix-serve-status-v1"`): state, queue, totals, recent requests |

`/health`, `/metrics` and `/api/status` do not require the API key.

### Chat requests

- `messages` with roles `system`, `developer`, `user`, `assistant` and `tool`; string content or text
  parts. Images and video are rejected.
- `tools` (function tools) are rendered into the system turn the way the model was trained; assistant
  tool calls in the history are rendered from `tool_calls` (string or object `arguments`).
- Thinking: on by default (the template default). `chat_template_kwargs.enable_thinking: false`
  (or top-level `enable_thinking: false`) disables it; `reasoning_effort` (`low`, `medium`, `high`,
  `xhigh`) and `preserve_thinking` are passed to the template.
- The reply is split into `reasoning_content` (the thinking block), `content` and `tool_calls`.
  Parameter values of a tool call are typed by the tool's JSON schema (strings stay strings, other
  types are parsed as JSON); `finish_reason` is `tool_calls` when the reply contains calls.
- `tool_choice: "none"` renders no tools.
- Other fields: `max_tokens` / `max_completion_tokens`, `stop` (string or list), `stream`,
  `stream_options.include_usage`, `n` (only 1), `temperature`, `top_p`, `top_k`, `seed`.
- Generation fields the server does not execute yet are refused with a 400 naming the field, on every route:
  `response_format` other than text / `json_object` / `json_schema` ([constrained output](#constrained-json-output)), `parallel_tool_calls: false`, `required` or named
  `tool_choice` (Anthropic: `any` / `tool`, `disable_parallel_tool_use: true`), `strict` tools, legacy `functions` /
  `function_call`; on completions also `n` / `best_of` other than 1, a `suffix`; on
  messages `thinking.type` other than `enabled` / `disabled` and non-null `output_config` members. Neutral values are accepted.
  Enabled `thinking.budget_tokens` is accepted and stated in `strix.note` as not enforced, unless the server runs
  with `--http-anthropic-thinking-budget enforce`, which maps it to `thinking-budget-v1` ([thinking budget](#thinking-budget)).
  `GET /v1/capabilities` lists these features as unavailable. Logprobs and transforms are available only on executing Qwen routes ([below](#logprobs-and-logit-transforms)); other engines refuse them.

Streaming sends `reasoning_content` and `content` deltas as they are decoded (holding back only
bytes that may belong to a stop string, a `</think>` or `<tool_call>` marker, an incomplete UTF-8
sequence or trailing whitespace), tool calls when they are complete, a final chunk with
`finish_reason`, optionally a usage chunk, and `data: [DONE]`.
Named-session capture admission runs before streaming headers: a refused budget
returns the same HTTP 507 JSON error with code `session_budget` as a non-streaming
request. If capture admission fails later, completions and chat send one
`data: {"error":{"type":"server_error","message":"...","status":507,"code":"session_budget"}}`
event and close without a final finish reason or `[DONE]`. Messages sends
`event: error` with `{"type":"error","error":{"type":"api_error","message":"...","status":507,"code":"session_budget"}}`
and closes without `message_delta` or `message_stop`. These errors affect only the
request; the server continues serving.

### Thinking budget

`thinking-budget-v1` is an explicit, token-changing selection policy. Chat and Responses requests opt in with:

```json
"strix": {
  "thinking_policy": "thinking-budget-v1",
  "max_thinking_tokens": 2048,
  "answer_room_tokens": 1024
}
```

Thinking must be enabled. With effective output budget `B`, answer reservation `R`, and closure length `C`,
reasoning may commit at most `min(max_thinking_tokens, B - R - C)` tokens before forced closure starts.
`max_thinking_tokens` must be a positive integer. Optional `answer_room_tokens` is a nonnegative integer;
omitted, the answer gets what remains (`R = B - max_thinking_tokens - C`), and a budget that leaves no answer
room (`max_thinking_tokens + C >= B`) returns 400. Explicit reservations with `R + C > B` return 400 before submission. A zero remaining reasoning allowance closes immediately.
Unknown policies and unavailable execution return 400. Without these fields, generation is unchanged.

The executing Qwen route uses the shared host selector on its resolved selection route. It resolves
`</think>` from the loaded tokenizer, then encodes the template's following two newlines. Forced IDs become
the actual target frontier, pass through target execution, and consume output/context capacity. They carry
`forced_control` provenance in selection; no model-selected probability is claimed. This policy requires
effective greedy selection and rejects nonempty stop strings and active tools. It composes with JSON output:
the answer constraint starts only after the entire forced closure. Natural closure before the limit needs no
injection. EOS during reasoning remains incomplete with reason `reasoning_eos`.

Every policy response includes `strix.thinking` with `policy`, requested `max_thinking_tokens`, resolved
`answer_room_tokens`, `closure` (`natural` or `forced`), committed `reasoning_tokens`, and `forced_ids`.
`status` and an optional `reason` distinguish incomplete output. Stream terminal metadata includes these fields
even with `--http-stream-receipts off`. Reasoning and answer split on the committed end-thinking token ID.
`GET /v1/capabilities` advertises `thinking_budget` and `thinking_budget_policy` only for an executing route.

For `/v1/messages`, `thinking: {"type":"enabled","budget_tokens":2048}` selects the same policy and default
answer reservation when the server runs with `--http-anthropic-thinking-budget enforce`; the response states the
executed policy. By default (`note`) the budget is accepted, the request keeps speculative decoding and `strix.note`
states that the budget is not enforced. Enforcing it uses the resolved host selection route. `--http-thinking-budget off` disables explicit policy requests. Other engines retain that note
until they integrate the executing selection adapter.

CPU contract tests cover the production selection adapter and HTTP translation. Native GPU token/state,
first-token/cache routes, mixed-slot isolation, and complete-workload qualification remain separate gates;
compilation and CPU results do not establish those gates.

### Anthropic-compatible messages

`system` (string or text blocks), `messages` with text, `tool_use` and `tool_result` blocks, `tools`
with `input_schema`, `thinking` (`{"type":"enabled"}` or `{"type":"disabled"}`), `stop_sequences`,
`max_tokens`. The reply has `thinking`, `text` and `tool_use` blocks and `stop_reason`
`end_turn`, `max_tokens`, `stop_sequence` or `tool_use`. Streaming emits `message_start`,
`content_block_start`, `content_block_delta` (`thinking_delta`, `text_delta`, `input_json_delta`),
`content_block_stop`, `message_delta` and `message_stop`.

### Responses (stateless)

`POST /v1/responses` is translated onto the chat route: `instructions` becomes the leading system message, `input`
(a string or items) becomes the messages, function tools and `tool_choice` `auto`/`none` map to their chat forms,
`reasoning.effort` to `reasoning_effort`, `max_output_tokens` to `max_tokens`. Message items take `input_text` /
`output_text` parts; `function_call` items (their `arguments` string kept as given) join the preceding assistant turn,
`function_call_output` items become tool messages matched by `call_id`, and plaintext `reasoning` items are attached
to the following assistant message. The prompt token IDs equal those of the equivalent chat request (tested), so a
Responses request generates the same tokens as its chat form.

The route keeps no state: `previous_response_id`, `store: true`, `conversation` and `background` are refused with
400, as are features the server does not execute yet: `text.format` other than text / `json_object` / `json_schema`, `text.verbosity` other than
`medium`, `strict: true` tools, `required` or named `tool_choice`, non-function tools, image or file input,
encrypted-only reasoning, reasoning summaries, `include`, `truncation` other than `disabled`, `max_tool_calls`,
unknown or duplicate `call_id`s; the 400 names the field. A `null` member counts as absent. The reply is a response
object with `reasoning`, `message` and `function_call` output items, `status` `completed` or `incomplete`
(`max_output_tokens`, or `unfinished_tool_call` when a stop string or the end of generation cut a tool call; such a
call has `status: incomplete`), usage and `strix`. As on the chat route, text from a `<tool_call>` marker on is
treated as tool framing, also when the model writes the marker literally. Streaming sends typed events (`event:` line + JSON with the same `type`
and an increasing `sequence_number`): `response.created`, `response.in_progress`, `response.output_item.added`,
`response.content_part.added`, `response.reasoning_text.delta`/`.done`, `response.output_text.delta`/`.done`,
`response.function_call_arguments.delta`/`.done`, `response.content_part.done`, `response.output_item.done`, and
exactly one of `response.completed`, `response.incomplete`, `response.failed`. There is no `[DONE]`.

### Logprobs and logit transforms

Qwen chat, completions and Responses execute these greedy-only policies on the same
resolved host selection route as constrained JSON and thinking budgets. Other slots retain their
selected routes. Effective temperature greater than zero is refused with 400; the
capabilities document advertises availability and execution. DeepSeek and MiniMax
continue to refuse these requests. The two feature switches above default to on;
disabling a feature refuses requests requiring it.

Chat accepts `logprobs:true` and integer `top_logprobs` in [0,20]. Completions accepts
integer `logprobs` in [0,5], including zero to request the chosen-token score only.
Responses accepts `top_logprobs` in [0,20] and
`include:["message.output_text.logprobs"]`. A positive `top_logprobs` also requests
scores. Scores include the first selected token from the terminal prompt row and do
not change selection. They are temperature-one log-softmax over the eligible final
selection row, after transforms and any JSON mask. Top entries sort by descending
logprob, then ascending token ID. `strix.scoring` reports
`basis:"target-row-log-softmax"`, `transform` (`none` or `logit-transform-v1`) and
`constrained`. These scores describe that row, not the unmasked model distribution.

Chat emits `choices[].logprobs.content` records with `token`, `logprob`, `bytes` and
`top_logprobs`. Token text uses lossy UTF-8; bytes preserve the model token bytes.
Completions emits `tokens`, `token_logprobs`, `top_logprobs` and `text_offset`.
Responses emits `output_text.logprobs` and `response.output_text.delta.logprobs`.
Streams carry the same scores as non-stream replies. Thinking-budget forced control
records have `logprob:null`, an empty top list and `forced_control:true`; no model
likelihood is invented for them.

Non-neutral transforms are explicit token-changing request opt-ins. The order is:

1. For IDs seen in the prompt or committed reply, apply `repetition_penalty` rho to the raw logit:
   positive logits become z/rho; zero and negative logits become z*rho.
2. Subtract `presence_penalty` once for each ID seen in the committed reply.
3. Subtract `frequency_penalty` times that ID's committed reply count.
4. Add each `logit_bias` value, then apply any JSON mask and greedy selection.

Presence and frequency accept [-2,2], repetition accepts [0.1,10], and bias accepts
at most 1024 canonical decimal vocabulary token-ID string keys with values in [-100,100]. Reply counts
include reasoning and forced controls and advance only when the selected token
commits. Transform arithmetic is applied to the target float row. Responses label
sampling `greedy-logit-transform-v1`, set `unconstrained_greedy_equal:false` and report `strix.transform` with `presence`,
`frequency`, `repetition` and `bias_entries`. This identity does not claim exact
untransformed greedy output. Neutral values (0, 0, 1, empty bias) without a scoring
request retain the ordinary route. JSON constraints compose as transform, mask,
greedy; scoring uses the resulting eligible row.

CPU tests cover these contracts. Native GPU qualification still needs complete IDs,
first-token route coverage, logits/state digests, mixed-slot isolation and workload
cost measurements for the executing model/profile; a compile check is not that gate.

### Constrained JSON output

`response_format` (chat) or `text.format` (Responses) set to `{"type":"json_object"}` or to a JSON schema
(`{"type":"json_schema","json_schema":{"name","schema","strict"}}` in chat, `{"type":"json_schema","name","schema","strict"}`
in Responses) makes the answer a JSON value that the token selection itself keeps valid: at every answer position the
server picks the highest-logit token among those whose bytes keep the output a valid prefix (lowest id on ties), and
end of sequence only once the value is complete. Thinking, when on, runs unconstrained until the end-of-thinking token;
the constraint applies to the answer after it. The answer bytes are returned exactly (no tool parsing, no trimming).

The schema language is `strix-json-schema-v1`: `type` (also type arrays), `properties`, `required`, boolean
`additionalProperties`, `items`, `minItems` / `maxItems`, `minLength` / `maxLength` (Unicode scalars), `enum`, `const`,
`anyOf`, local acyclic `$defs` / `$ref`, and the annotations `title`, `description`, `$comment`. Any other keyword
(`minimum`, `pattern`, `format`, `oneOf`, ...) is refused with a 400 whose `param` is the keyword's full path (for
example `text.format.schema.properties.age.minimum`) and `code` `unsupported_schema_keyword`; `strict: false` does not
turn the refusal into silent ignoring. Output follows a documented generation language (standard JSON with at most two
whitespace characters between tokens, no duplicate keys, integers without fraction or exponent, `enum` / `const` in
their compact spelling). Compilation is bounded (schema size, nodes, depth, properties, enum values, work, memory);
limits are listed in `GET /v1/capabilities` under `constraints`.

Constraints run greedy only with host target selection on the resolved route described below. Constraints are refused with stop strings, with effective sampling
(`temperature > 0` on a sampling server) and with a non-empty tool offering. Every constrained response carries
`strix.constraint` (also with stream receipts off): `identity` (`json-object` or `strix-json-schema-v1`), `selection`,
`execution`, `phase`, `status` (`complete`, `incomplete`, `failed`), `accepting` and `reason` (`max_output_tokens`,
`context`, `reasoning_eos`, `cancelled`, `constraint_failure`, `execution_failure`). A reply cut by the budget is
reported incomplete, never repaired; Responses then has `status: incomplete`. `host_profile` reports the host
selection cost of the request (wall time, informational, differs between runs): `known_rows` selected
argmax-first (below), `rows` / `select_ms` selected on the full row (transforms, argmax, scores),
`mask_ms` building constraint masks and, on `mtp-host`, `window_rows` / `window_row_ms` for whole window
rows (logits readback and selection). Rows of rejected window suffixes are included.

### Policy selection routes

`--http-selection-mtp on` (default) admits `mtp-host` for closed R2/R4/R8
serial-derived owners whose per-row arithmetic preserves the complete serial target
FP32 logits and state. Both prefix commit and checkpoint/restore owners use this
admission. Tiled owners, GSQ owners, the legacy all-Q8 staged GDN pair, unknown
recipes and window-only inject/normalization or combine folds are excluded here.
These conservative exclusions fall back before drafting. The existing prefix gate
compares full-logit digests and committed state, including rejection depths and
owner changes; matching argmax IDs alone is insufficient.

Argmax-first selection (`--http-selection-fast on`, default) applies on every route. Masked greedy
picks the lowest-ID maximum over admitted IDs, so when a row's unmasked lowest-ID maximum is admitted
it is the masked result: the row needs only a membership check of that one token against the
constraint cursor (EOS only at a complete value, never the end-of-thinking ID), not the full mask
walk. On `mtp-host` the maximum comes from the window's GPU reduction (same finite first-max rule, both
zeros equal), so such rows also skip the logits readback and finite scan; on `serial-host` and serial
replays the host computes it from the row it read. Forced closure and reasoning-phase rows use it
directly. Rows with a nonfinite logit or a refused maximum, and requests with scores or transforms, are
selected on the full row; token IDs are identical either way. One difference to `--http-selection-fast
off` (every row walks the full mask): a full walk can stop a request on a mask-walk evaluation limit
reached by another token, which a membership check does not walk; all routes of one setting behave the same.

Every policy response reports `strix.selection_execution` (`mtp-host` or
`serial-host`) and `strix.selection_fallback` (null when admitted). Fallbacks are
`selection-mtp-off`, `no-window-owner` and `window-arithmetic-not-serial`, in that
order. The switch covers scoring-only and transform-only requests too. Feature
disable switches still refuse requests. Capabilities list routes available on the
loaded owners; the request receipt reports the resolved route.

The first token retains its terminal-prompt transaction. Each later MTP cycle uses
ordinary unconstrained proposals and selects only reachable target rows with the
same host selector, masks, transforms, forced controls and CPU scoring as serial.
Acceptance compares each draft with that policy-selected token. Prefix owners
commit the decided rows only; checkpoint owners restore and replay serially unless
the whole chunk matches without consuming EOS. Context/history/budget limits can
also cause transaction-aware serial verification within an `mtp-host` cycle.
Reconciliation completes before the retained state and every score publish, then
IDs emit. Rejected suffix rows publish nothing. Policy requests have no context,
frame, peer or wide-copy proposals and do not train ordinary shared calibration.

Exactness is relative to the same declared policy and numerical identity: complete
IDs, forced-control records and every chosen/top score must equal serial, with
bit-identical reachable full-logit rows and committed model state. CPU tests and
compilation do not establish GPU or complete-workload qualification. No MTP policy
speed claim is made.

### Applied policy and keepalive

Streaming replies may emit compact comments such as
`: progress {"phase":"prefill","prefilled_tokens":128,"cached_tokens":64,"total_tokens":256}`
followed by a blank line. Counts describe completed host-visible prefill units: computed tokens and restored
cached tokens separately; their sum never exceeds the prompt total. The phase becomes `decode` at the first
published token. Snapshots are coalesced, so short phases and intermediate counts may be skipped. No progress
is invented for engines that do not report it. `/v1/capabilities` advertises `stream.progress.interval_seconds`
and `engine_reports`; the applied stream receipt reports the request-effective interval. The interval defaults
to one second; `--http-sse-progress-interval 0` or `"strix":{"progress":false}` disables comments.
Progress comments are shared by chat, Messages, Responses and completions streams, never non-stream replies
or data events. They count as writes for keepalive. Reports use existing host counters at completed blocks,
serial rows and windows, including cache restore; they introduce no device synchronization or scheduling changes.
Progress-on/off GPU token/state equality remains a native qualification gate; CPU fake-engine tests cover wire equality.

With `--http-stream-receipts on` (the default), every response's `strix` object states the policy that ran: `sampling_requested` and `sampling_effective`
(`greedy` or `sample`, with `note` when a sampled request ran greedy), `filters` (each of `temperature`,
`top_k`, `top_p`, `min_p` has a resolved `value` and `source`: `request`, `server` or `builtin`),
`seed` for sampled runs, `max_tokens_requested` and
`max_tokens_effective`, and the stream settings. A stream carries it in its first event (chat: the role chunk;
messages: `message_start.message`; responses: `response.created`; completions: the first chunk) and its last,
whether or not `stream_options.include_usage` is set. `GET /v1/capabilities` (`"object":"strix.capabilities"`,
`"version":1`) lists what the server executes and marks every feature it does not as `false`; its
`capabilities_id` is a hash of the document. `sampling.defaults` reports each effective server default
with its `value` and `source` (`server` or `builtin`), and `sampling.seed_echo: true` advertises seed receipts.

While a stream waits for tokens (queued, during a long prefill, between decode steps) and nothing was written for
`--http-sse-keepalive` seconds, the server writes the SSE comment `: keepalive`, which clients ignore and proxies
count as activity. A request turns it off with `"strix": {"keepalive": false}`. Neither the receipts nor the
keepalive change scheduling or generated tokens (GPU-gated: token IDs equal with them on and off).

### Engine fields

Non-streamed responses (and the final usage chunk of a stream) carry a `strix` object: queue,
prefill and decode milliseconds, decode tokens per second, MTP cycles, proposed and accepted drafts.
`"return_token_ids": true` in the request adds `strix.token_ids`, the generated token IDs including
the stop token, for exactness checks against the command line.

## Status for the monitor panel

`/api/status` reports per request the fields of the live monitor's request record (`status`
`done|cancelled|error`, `stop_reason`, `prompt_tokens`, `reused_tokens`, `output_tokens`,
`prefill_tok_s`, `decode_tok_s`, `duration_ms`, `mtp_cycles`, `mtp_proposed`, `mtp_accepted`), so
the monitor page can show served requests without another format. The serving loop publishes the
same facts it already holds on the host; it adds no device synchronization.

## MiniMax serve (`strix-minimax-serve`)

The MiniMax TP2 server (rank 0 serves HTTP; rank 1 replays the control blocks) has its own API module,
`src/minimax/serve/minimax_api.*`, with the same routes plus:

DSpark mode defaults to `auto` (`STRIX_MM_SERVE_DSPARK`); `--dspark-mode off|auto|on` overrides
explicit environment, which overrides the resolver. Environment aliases `0|off` and `1|on` remain
supported. Invalid values fail preflight on both ranks, including inspection-only runs.
`STRIX_MM_BASELINE=1` supplies `0` when the item is unset; explicit environment or CLI still wins.
Use `STRIX_MM_SERVE_DSPARK=0` for a DSpark-only comparison. Baseline also changes other optimizations.
`--dspark DIR` supplies assets independently of mode. With no explicit path, auto/on discovers local
safetensors under `MODEL_DIR/dspark`; missing optional assets use plain. No assets are downloaded.
Explicit invalid assets fail. Both arms of an in-process comparison must load the same drafter;
report cold load time, rings, nested tap banks and peak memory separately from generation.

The frozen default prior is **structural-v1**: eligible C1 prompts with JSON/code markers
or at least 6551 tokens enter the legacy automatic controller. Ordinary prose of every
language stays plain without neural probes. Select `STRIX_MM_DSPARK_PRIOR=unqualified-v1`
or `--dspark-prior unqualified-v1` for conservative plain admission; CLI overrides env.
The [frozen feature definitions and thresholds](dspark-auto-qualification.md#frozen-structural-v1-features)
are host-only and evaluated on rank 0. Grouped prompts remain
`plain:grouped_taps_unqualified`. Auto excludes queued competitors, C2/C4 and competing
prefills, including surviving requests after burst drain. Raising
`STRIX_MM_SERVE_DSPARK_INFLIGHT` never enables automatic C2; zero disables admission.
Structural-v1 node qualification: Serve, two Strix Halo nodes over 2x USB4 Gen3, PerfOpt kernel, one load with mode switched per request (off / auto / auto / off, twice per class), 256-token greedy C1 chat requests: generated token IDs (ids_fnv) and text identical in every visit of every class; wall time auto vs off: JSON -34.6 % (13.57 -> 8.87 s, 95 % interval -4.99..-4.40 s), code -17.5 % (-2.67..-1.90 s), Spanish prose plain (no drafts) +0.2 % (-0.08..+0.13 s), English story plain -0.1 % (-0.04..+0.03 s). Long prompts were not measured on serve (prompts above max_rows stay plain). English story remains plain unless structural/length
features qualify; its historical marginal gain is not claimed as retained.

`--dspark-mode on` or `/admin/sched` boolean `dspark:true` retains the legacy controller and its
exploration behavior; it has no new per-class performance claim. Boolean false is immediate OFF;
`dspark_mode:"off"|"auto"|"on"` selects explicitly. Sending both fields is rejected even if they agree.
An epoch change retires existing contexts on both ranks; no stale ring resumes. Protocol v3 uses the
same 512-byte control block and a context retirement mask. OFF stops context appends at the returned
boundary even in an open prefill; tap disable waits for both active and parked prefills to close.
Target calls, original row shapes and split policies stay fixed. Runtime mode, prior, asset availability,
context count and concurrency eligibility are reported by `/admin/sched`; request timings report
`dspark_mode`, `dspark_reason`, `dspark_context`. Status snapshots can lag one returned boundary.

Rank `--spec ...:auto` uses the same frozen structural admission and the ordinary session as its plain
oracle. `auto:legacy` preserves the previous automatic controller; `fixed:K` and diagnostics remain
explicit. The serve-specific environment does not rewrite rank policies. Candidate Markov remains
applicable to promoted rank auto and explicit neural policies; serve still runs the full Markov chain.
Passive grouped trailing taps, the net-cost active/probe controller and serve candidate
Markov integration remain pending. See [DSpark qualification](dspark-auto-qualification.md) before claiming performance qualification.

The resolver defaults to `STRIX_MM_SERVE_SCHED=interleave`, `STRIX_MM_SERVE_SERVICE_MS=250`
and `STRIX_MM_SERVE_SHORT_FIRST=1`. GPU exactness and HTTP load measurements appear below.
`STRIX_MM_SERVE_SCHED=off` restores whole-prefill scheduling and FIFO admission;
`STRIX_MM_BASELINE=1` also selects that path unless explicitly overridden.

Serve preflight validates `STRIX_MM_SERVE_SCHED`, `STRIX_MM_SERVE_SERVICE_MS`,
`STRIX_MM_SERVE_GAP_MS`, `STRIX_MM_SERVE_SHORT_FIRST` and `STRIX_MM_SERVE_NEST` on both ranks,
before transport creation, device work or model loading, including inspection-only invocations.
Scheduler values are `off`/`0` or `interleave`/`1`; short-first and nest accept only `0` or `1`.
Service accepts finite milliseconds from 0 through UINT_MAX; gap accepts 1 through UINT_MAX.
Invalid values exit with a named configuration error. `--print-effective-config` includes resolved
`serve.sched`, `serve.service_ms`, `serve.gap_ms`, `serve.short_first` and `serve.nest` values;
100-ms and 1500-ms deadlines have distinct configuration output. This validation is scoped to serve.

Start both ranks with the same serve policy environment. A value rejected on one rank exits that rank
before transport creation; the peer then fails after its transport timeout (default 300 s).

Deadline-based admission and layer budgeting passed both-rank GPU exactness checks and were
measured under HTTP load together with nested admission; results and open costs appear below.

Rank 0 protects a decoder for a candidate only when that decoder has emitted its first token
strictly before the candidate's queue submission time. Equal timestamps do not qualify. Protection
uses the first emission, never the most recent emission or admission time. With no protected decoder
and no warranted nesting, admission follows exactly the OFF path: FIFO, one unlimited whole prefill,
then another admission before any decode cycle. A pure burst therefore keeps OFF's operation sequence,
row counts and batching, including when burst-mates have already emitted their first tokens.
No new environment switch is introduced; existing off switches retain their meanings.

Short-first may reorder only when the FIFO head has a protected decoder. Every later submission then
has that same protected decoder; unprotected burst prompts retain FIFO order. A short is an older waiting
child only if its submission strictly precedes both the parent's submission and the start of the parent's
prefill. Requiring both excludes burst-mates from justifying a split. Otherwise a short needs its own
protected decoder to nest, which permits shorts arriving during a long prefill beside pre-existing decoders.
Nesting also requires the existing bank, size, quota and deadline checks; an older waiting child can
warrant bounded admission only for a long prompt (over 512 rows). The effective request policy is retained
across resumes and decoder cancellation/slot reuse;
whole admissions retain OFF cancellation semantics. Cancelled tickets never grant protection, regardless
of their admission policy. A whole admission never polls or parks midway through its prefill.

`strix-minimax-lead-tests` runs the production rank-0 Lead loop with a fake clock and engine/control
sink (no GPU). Submissions become visible only at their due times, including during fake prefill/step
execution. It compares emitted control fields, checkpoints, slot generations, ordered rows and layer
budgets against OFF for 1, 4 and 8 equal-time descending-size bursts, before-first-emission arrivals,
cancellation before admission and during whole prefill, and cancelled-decoder slot retention with DSpark
disabled/enabled. Test-only counters compare queue selection locks, cancellation scans and older-child
lookahead work against OFF. Mixed traces check a short arriving during a bounded parent, a future short
that cannot overtake a reserved long, cancelled/completed protectors and slot reuse. Runtime-policy
traces apply the same shared policy/epoch transition as `/admin/sched`; they cover OFF-to-interleave
with a cancelled old-policy decoder, interleave-to-OFF during a bounded parent and a pending
reservation that stays ahead of a later short after switching OFF. A mixed trace also checks protector
completion during an open parent and cancellation of a bounded prefill at its returned boundary.
They do not exercise HTTP parsing, GPU arithmetic, native timing
or both-rank replay.
The GPU measurements below confirm the burst identity with OFF; host sequence equality alone does not
establish a universal aggregate-throughput guarantee for mixed loads.

With a protected decoder, the deadline policy below remains in force.

Before opening a selected fresh prompt, rank 0 checks elapsed time since the oldest decoder's last token plus
predicted prefill, the next complete decode cycle including the candidate row, and a 20 ms control margin.
The sum must fit the **minimum gap target over all live requests** (`STRIX_MM_SERVE_GAP_MS`, default 1500).
Protected whole prefills can run consecutively only while this deadline permits them. Otherwise rank 0 decodes immediately;
it never waits for more arrivals to enlarge a batch. A selected candidate is reserved until admission or cancellation;
its queue timing includes that reservation. Long candidates that cannot fit even after a cycle open as bounded tasks.

With decoders waiting, layer units spend their actual remaining gap budget, capped at 1300 ms, less elapsed wait,
the next full cycle and the control margin. The service target does not cap these units or override a depleted gap.
Predictions use the greater of the layer EMA and the last eight unit samples' layer-cost upper bound in the
same row bucket (scaled up for more rows, never down), plus the fixed-cost estimate. Complete cycle safety estimates survive policy epochs; cycle samples
include selection, drafting, verification, emission and the counter tail. Unmeasured cycle row counts use a scaled
estimate. These are conservative predictions, not proven bounds on future GPU or transport costs.

Unprotected admissions without warranted nesting run whole even when their prediction exceeds the
idle service target. For bounded tasks without decoders, unit size uses the service target (250 ms by
default); service 0 makes those resumes unlimited. With waiting decoders it still subtracts elapsed time,
the next cycle and margin from the gap. Zero layers means decode first. If one indivisible layer still
cannot fit just after decoding, the scheduler explicitly forces one layer to make progress. Such overruns
are unavoidable without changing the numerical unit. Gap values must be finite and at least 1; service values must be finite,
nonnegative and nonoverflowing. Profiling and FP32 prefill retain forced whole-prefill resumes and report
`profiling` or `fp32_prefill`; these modes cannot enforce decoder gaps.

All prefill row counts, chunk/group sizes, GEMM/split identities and reductions remain unchanged. Only whole-layer
pause boundaries and admission order change. Prefill cache pointer tables are reused for the same slot and immutable
layout across decode cycles; decode uses a separate table. A slot change retains the existing table copy and host
lifetime synchronization. Transport and CRC counters remain global across pauses and bank switches.

`STRIX_MM_SERVE_TIMELINE=1` enables rank-0 JSON lines (`event=mmserve-timeline`), default off and read once at startup.
Nested-candidate rejections emit `nest-reject` with reason `timing`, `no boundary`, `bank`, `quota`,
`fit`, `overtake` or `decode-needed`; candidate rows accompany eligibility rejections. With the timeline
off, rejection diagnostics read no clocks and allocate no records.
Records cover `admit`, `unit`, `step`, `idle` and `heartbeat`, with monotonic operation ID, epoch, steady-clock
`t_start_us`/`t_end_us` since scheduler start, rows, slot/request/position/token-index summaries, prefill chunk members,
layer range, completion age/slack and `whole`/`defer`/`deadline` reason. Admission records describe prefill begin separately from
unit completion; unit host-return time is not necessarily completed GPU time. `control_end_us` and `run_end_us`
mark control/setup completion and runner return (both default to operation end when inapplicable). Step intervals cover selection through
emission/counter tail. Records add no device synchronization and contain no prompt text. Up to 65536 records buffer
until idle or shutdown, with excess counted by `mmserve-timeline-drops`; flushing is outside active operations.
The disabled path constructs no records and performs no instrumentation clock reads or allocations.

When the FIFO head has a protected decoder, short-first chooses the smallest remaining queued prompt,
with arrival-order ties and at most eight later admissions overtaking any request. `STRIX_MM_SERVE_SHORT_FIRST=0` selects FIFO. `STRIX_MM_SERVE_NEST=1` (default ON) also permits one short prompt to nest at a returned long-prefill
layer boundary. `STRIX_MM_SERVE_NEST=0` or the strict boolean `/admin/sched` field `nest=false` disables new children.
Nested admission passed both-rank GPU exactness checks for bounded 512-token and whole 511-token children;
the mixed-load measurements below include nested admission. No slot-capacity expansion is performed.
A child keeps its original one-chunk shape, with 1..min(512,max_rows) rows and a different reserved slot.
Its separate activation bank retains the long's entire original group in place; scratch is reused sequentially.
Parking retires GPU work, restores any eight-row decode save into the parent bank and preserves its task/sink/timeline.
Unparking retires child work, rebinds the cache table and parent metadata, then takes a fresh eight-row save.
Transport sequences, CRC state and workspace ring cursors never roll back.

Rank 0 admits a whole child only if predicted prefill plus park/check/unpark, elapsed decoder wait, the next full cycle
including the child and at least 20 ms host margin fit the minimum live gap (capped at 1300 ms). Without decoders,
child service still must fit that budget. The measured switch estimate includes CHECK, control/setup and
first-token publication: elapsed nesting time minus child engine service counted exactly once. Cold estimates can defer admission. There is one nesting level, at most two
children per parent (including cancellations), and a completed parent unit between children. Cumulative nested wall
service is limited to two gap budgets. Queue aging still vetoes bypass of an ineligible aged request.
**Parent units retain the deadline policy's gap-budget sizing; there is no additional 250 ms arrival polling cap.** An arrival can
wait one long unit (predicted up to roughly 1.3 s), then its own service and control overhead. These are predictions,
not hard latency bounds. Unlimited profiling/FP32 prefill and optional prefill exchange are excluded.

Both ranks reserve the child bank before READY, after the optional drafter, with at least eight physical decode rows,
independent timeline events and optional taps. Reservation requires 4 GiB device and OS available-memory headroom;
a failure on either rank disables effective nesting with a reason. Banks stay reserved when policy is OFF.
Startup policy and `/admin/sched` report requested/effective `nest`, reason, rows, bank/tap bytes, quota, active/parked
IDs and a static runtime qualification label. Per-request timings and policy receipts identify nested requests and their parents.
Structured `mmserve-req` receipts include `task_id`, `parent_task_id`, `nest_count`, `park_wall_ms`,
`nest_service_ms` and `nest_switch_ms`, including partial parked totals when a parent is cancelled. IDs are assigned even with tracing OFF. The versioned control blocks
carry explicit task IDs, parent IDs, slot generations and both task checkpoints. Rank 1 replays them verbatim; replies
acknowledge prior completion, and CHECK precedes child publication. Either task can be cancelled by ID at a returned
boundary, with GPU retirement before slot release. Global tap changes wait for both tasks to close.
Cancellation cleanup in each queue take is limited to the entries present at its first selection; new arrivals
participate in the next take. Cancelled entries take precedence within that batch.
Cancelled interleaved decoders retire before a decode cycle. Rank 0 alone chooses admissions, budgets and cancellation;
rank 1 replays its control blocks without local timing or selection decisions.

Startup policy output, `/admin/sched` (with `--admin`) and reply `timings` include `sched`, `effective_sched`,
`service_ms`, `short_first`, `max_unit_layers` and `fallback_reason`. `max_unit_layers` is the largest requested
resume budget seen so far (per request in timings, since reset in policy); 4294967295 denotes unlimited,
including a forced-unlimited numerical mode. Each request keeps its admission-time scheduling mode, service target
and gap target through completion, including all open-prefill resumes. Its `short_first` records the admission
selection policy. Runtime changes to these settings apply to later admissions; reply timings report the policy
that request executed. DSpark controls remain runtime switchable. The runtime qualification label is static
and does not encode the measurements below.

`--resume-check PROMPT.u32:1/7/60 --resume-check-tokens 256 --check-only` runs on both ranks before serving.
It requires at least three slots and capacity for the prompt plus the complete requested continuation.
It compares complete token-ID vectors, final committed-state and continuation-logit digests, and prompt output rows,
including idle bounded resumes and a two-prompt short-first reorder. With a drafter loaded, only the legacy
long request with a prompt of at most 4096 tokens fills `Result::draft`; legacy background requests and
reorder/nested arms compare zero-initialized draft arrays. Either rank's
mismatch fails both. Run ordinary, grouped/tail and drafter variants with the same model and numerical settings.
Only `--resume-check-nested` selects the extended gate. It additionally runs a long parked at every supplied
layer in 1..59, a short child (whole by default), and the resumed long, beside an existing background decoder. For example, on both ranks:

```sh
strix-minimax-serve MODEL_DIR --rank R --slots 3 --capacity 8256 --resume-check LONG4608.u32:1/3/7/17/31/45/59 --resume-check-nested SHORT512.u32 --resume-check-tokens 128 --check-only
```

The entire explicit short fixture must contain complete u32 tokens, 1..min(512, max_rows) original rows,
and leave capacity for the continuation. Missing, empty, partial-token and oversized fixtures fail; no input is truncated.
Run boundary fixtures of 511 and 512 rows, and require a 513-row fixture to fail. Without the nested option,
only the unchanged legacy cases run, including `:60`-only, small-row structural and unlimited fallback checks.
Add `--resume-check-child-layers 7` (1..59) and at least 128 continuation tokens to exercise repeated bank-1
save/restore with background decode between child units. Explicit unsupported nested requests fail. New independent
standalone B1 oracles cover all three requests, all prompt output rows (including earlier groups), the first head
logits, every continuation logit and the final committed K/V/index-key bytes via exact-byte digests. The nested arm
uses real task-addressed control blocks for begin/park/nest/unit/CHECK/unpark, runs both with and without a background
decode before parking, checks the entire frozen parent activation group and valid taps, and compares subsequent
mixed decode layouts with the B1 oracles. Either rank's mismatch fails both; an unavailable bank or unlimited fallback
fails rather than silently skipping nesting. The measured GPU cases below passed. Digests are compact byte-equality
oracles, subject to hash collisions. Broader shape, cancellation/protocol adversarial, DSpark and memory/performance
gates remain required beyond these measured cases. The gate itself does not establish HTTP arrival latency,
cancellation latency or mixed-load performance; HTTP load measurements are reported separately below.

- `GET /api/status[?detail=1]`: `"schema": "strix-serve-status-v1"`, `"engine": "strix-minimax-serve"`, state,
  queue, lifetime totals (from the engine queue: requests, prompt, cached, imported, computed and generated
  tokens, busy seconds), slots, the process `instance` (`id`: random 16 hex chosen once at start; `start_time` and
  `started_at`: the same unix start time; always present; the key for counter
  baselines), `capabilities` (prefix cache and state import are `false` in this build: their counters are
  real zeros, and a consumer shows them as not supported; `draft` is true when a DSpark drafter is loaded), load facts and the recent-request ring (50 records,
  newest first). `detail=1` adds the latency histograms as JSON (`le` = finite bounds, `counts` = cumulative
  counts with the last one for +Inf).
- Per record: `id` (the accepted `X-Request-Id`, else the completion id), `request_id`, `completion_id`,
  route, arrival time, `status` (`done|cancelled|error|refused`), `stop_reason`, `stream`, prompt accounting
  (`prompt_tokens`, `cached_tokens`, `imported_tokens`, `computed_tokens` = prompt - cached - imported;
  `reused_tokens` is the legacy name of `cached_tokens`), `output_tokens` (stop token included), `queue_ms`,
  `prefill_ms`, `decode_ms`, `duration_ms`, `first_token_ms` (submit to the first published token, measured by
  the scheduler; null when no token was published), `max_gap_ms` (the scheduler's longest gap between two
  emitted tokens; null with fewer than two), draft counts when the engine drafted, rates. The decode rate is
  (output - 1) / decode seconds, because this engine's decode time starts at the first token.
- Reply `timings` (non-stream reply and the stream's final chunk): the scheduler's per-request measurements,
  `queue_ms`, `prefill_ms`, `decode_ms`, `decode_tok_s`, `completion_tokens`, `ttft_ms`, `max_gap_ms`,
  `stall_ms` (decode time stalled by prefill units), `prefill_units`, `cycles`, `drafted`, `accepted`, `ids_fnv`,
  `prompt_fnv` (FNV-1a, 16 hex) and `sampling`, for every request, armed or not.
- `GET /metrics`: metrics text with `# HELP`/`# TYPE` for every family. The `mm_*` value lines are
  unchanged (the queue's seven and the scheduler's step, prefill, draft, idle and decode-stall seconds, steps,
  step rows, prefill units, draft cycles, drafted and accepted tokens; `POST /admin/sched` with
  `reset_counters` zeroes the scheduler's, a counter reset); the `strix_*` counters and gauges use the shared names (`strix_requests_completed_total`,
  `strix_prompt_tokens_total`, `strix_generated_tokens_total`, `strix_queue_waiting`, ...), plus
  `strix_prompt_tokens_imported_total`, `strix_prompt_tokens_computed_total`, `strix_engine_info{engine,model,
  instance}`, `strix_engine_start_time_seconds`, `strix_instrumentation_armed` and, once armed, the
  histograms `strix_ttft_seconds` and `strix_decode_tokens_per_second`.
- Armed instrumentation: the first `GET /metrics` or `GET /api/status?detail=1` arms the process (one-way).
  Requests submitted before that cost one relaxed atomic load each and are not recorded in the ring or the
  histograms; lifetime totals are always available. Armed, a request adds one ring record and constant-time
  histogram updates at its end on its HTTP thread, and the scheduler copies one already-measured time into its
  statistics at completion; nothing on the device path changes.
- Request ids: an inbound `X-Request-Id` of 8-64 characters `[A-Za-z0-9._-]` (for example `<16 hex>.<attempt>`)
  is kept whole and echoed in the `X-Request-Id` of every response to that request; a generation request that
  reaches the queue without one echoes its completion id (requests rejected before that echo only an accepted
  inbound id); any other value is ignored.
- `"return_token_ids": true` adds `strix.token_ids` (every generated id, the stop token included, so its length
  is `usage.completion_tokens`) to the reply, and in a stream to the final chunk (the one with `finish_reason`)
  and to the usage chunk. `stream_options.include_usage` adds a chunk with empty `choices` and the `usage`
  after the final chunk, before `data: [DONE]`. These boolean options must be booleans (400 otherwise).
- `--admin`: `GET` / `POST /admin/sched` (scheduler policy and counters at runtime), served by the same API module
  through the app's hook.
- Tests: `strix-minimax-observe-tests` (ids, ring, status JSON, a metrics text validator, histograms) and
  `strix-minimax-api-tests` (the HTTP contracts against a scripted engine thread: echo rules, token ids,
  stream event order, engine errors, failed stream start, client disconnect, queue refusal, arming during a
  request, the scheduler's timings, extras, `mm_*` lines and `/admin/sched` hooks). `tools/minimax/serve_observe_check.py` checks a live server and writes the token-identity and
  overhead receipt across unarmed and armed sets.

### Measured deadline-based and nested serving policy

Measured on GPU with two Strix Halo nodes over 2x USB4 Gen3, PerfOpt kernel, MiniMax M3.1 Flash
EXL3 3.06 bpw TP2, with the protection rule above. One binary per comparison, built without a
compiler cache in a fresh build directory; policy ON (the default, no `STRIX_MM_SERVE_*` variable set)
and OFF (`STRIX_MM_SERVE_SCHED=off`) alternate inside each two-node turn, in both orders. All
comparisons below are policy ON vs OFF of the same binary.

Every `--resume-check` arm compares complete fixed-length token IDs, committed state, a digest of
all normalized prompt output rows (every chunk and grouped member), and logits of the first generated
token and every continuation token, for every compared request on both ranks. Legacy idle and
beside-decoder arms compare the long prompt and both background requests against the whole-prefill
oracle. With a drafter loaded, only the legacy long request with a prompt of at most 4096 tokens fills
`Result::draft`; legacy background requests and reorder/nested arms compare zero-initialized draft arrays.
Short-first reorder compares both requests
against FIFO standalone execution. Nested arms compare parent, child and background requests against
standalone B1 execution, and check parked activation/checkpoint preservation. Either rank's mismatch
fails both ranks. The legacy, reorder and nested arms use the same prompt-output/first-head capture.
All runs used complete 128-token continuations:

| Cases | Result on both ranks |
|---|---|
| Legacy: 1K prompt at 1/7/60-layer units idle and beside decoders, short-first reorder, 4608-token prompt | 11/11 IDENTICAL |
| Nested: 512-token short in 7-layer child units, 4608-token long parked at layers 7 and 31, beside a decoder | 10/10 IDENTICAL |
| Nested: whole 511-token short, long parked at layer 17 | 6/6 IDENTICAL |

Under HTTP load, every request's token IDs were identical with the policy ON and OFF.

Bursts (all requests submitted before any first token, 64 tokens each) take OFF's operation path:
every request is one whole prefill and both policies run the same number of decode cycles. Aggregate
tok/s, ON / OFF, two turns (ON first, then OFF first):

| Burst | Turn 1 ON / OFF | Turn 2 ON / OFF |
|---|---|---|
| Distinct prompts, C1 | 17.42 / 17.36 | 17.38 / 17.38 |
| Distinct prompts, C4 | 31.33 / 31.25 | 31.34 / 31.34 |
| Distinct prompts, C8 | 36.28 / 36.10 | 36.48 / 36.61 |
| Identical prompts, C4 | 36.65 / 36.40 | 36.59 / 36.58 |
| Identical prompts, C8 | 43.74 / 43.78 | 43.81 / 42.79 |

The total wall time of all burst sets differed by +0.05 % and -1.1 % (ON vs OFF). An earlier build of
the same rule measured twelve alternating arms with a mean difference of 0.0 %. Within-arm
differences are run-to-run noise: ON and OFF issue the same operations.

Mixed load started three decoding streams (384 tokens each), then a 4539-token prompt, with 183-token
prompts arriving 1 s and 3 s later. Three turns, ON / OFF:

| Metric | Turn 1 | Turn 2 | Turn 3 |
|---|---|---|---|
| Makespan, all six requests (s) | 36.70 / 36.60 | 35.69 / 35.67 | 35.74 / 35.61 |
| First short prompt TTFT (s) | 1.13 / 12.69 | 1.22 / 12.24 | 1.24 / 12.19 |
| Second short prompt TTFT (s) | 1.11 / 11.60 | 1.07 / 11.15 | 1.11 / 11.09 |
| Long prompt TTFT (s) | 15.87 / 12.76 | 15.13 / 12.34 | 15.21 / 12.28 |
| Decoders' p90 / max chunk gap (ms) | 52 / 2080 vs 52 / 15723 | 51 / 1929 vs 50 / 15290 | 51 / 1935 vs 50 / 15279 |

Mixed aggregate throughput is within 0.4 % of OFF (makespan +0.27 %, +0.06 %, +0.36 %). The second
short prompt is admitted as a nested prefill at a layer boundary of the long prompt. The long prompt's
TTFT is 19-24 % above OFF because its layer units yield to the decoders; removing that cost needs
decode steps that run concurrently with prefill units, which is work in progress. Use distinct
prompts when re-measuring bursts so identical lockstep rows do not inflate expert reuse.

## Tokenizer and chat template

The tokenizer is read from the model GGUF (byte-level BPE, 248320 tokens, its merges and its
pre-tokenizer split rule) and implemented natively; special tokens in a prompt string are parsed as
single tokens. The chat template is a native implementation of the template embedded in the GGUF
(system/developer merge, reasoning-effort instructions, tool block, tool-call and tool-response
rendering, thinking on/off generation prompt). `tools/serve/gen_chat_template_cases.py` renders the
embedded template on fixed cases; `strix-serve-frontend-tests` checks the native renderer against
them byte for byte, and the tokenizer against expected splits. `strix-tokenize MODEL.gguf
encode|plain|decode FILE` exposes the tokenizer on the command line; MODEL may also be a Hugging Face
`tokenizer.json` or a model directory that holds one (EXL3 models).

## Tests

- `strix-serve-frontend-tests`: JSON, tokenizer pre-split and UTF-8 streaming, chat template cases,
  assistant output parsing (with a model GGUF as second argument, also an encode/decode round trip).
- `strix-serve-api-tests`: the HTTP contracts against a scripted engine: chat (plain, streamed, tool
  calls, stop strings, thinking off, length), completions (text and token IDs), Anthropic messages
  (plain, streamed, tool use), errors, authentication, two concurrent clients, cancellation on
  client disconnect, status and metrics.
- On hardware: the same prompt through the command line and through the server gives the same
  token IDs, also when two clients send requests at the same time and when the prefix state cache
  restores part of the prompt.

## Limits

- `--http-batch 0`: one request at a time, greedy only; concurrent requests wait in the queue.
- Batched: prompt blocks of several requests may form a segmented group; one native prompt unit owns the shared
  prefill arena at a time. With more than four decoding requests a cycle forms several 8-row cohort windows (two
  rows per request); joint planning (`STRIX_MTP_BATCH_JOINT=1`) runs in groups of at most eight requests.
- YaRN mode (`--rope-yarn`) is CLI-only: the HTTP server refuses it (MTP is available on the explicit CLI route), and a request cannot
  change the rotary mode.
- Every built slot holds a full context of state memory and its own window owners; memory grows with the built
  slots. A nested prompt pauses the long prompt for its own prefill; boundary cancellation is not on the
  one-at-a-time route.
- Each request resets the adaptive min-p, depth or survival controller to its configured prior, retaining
  the slot's device owners. Previous requests' acceptance estimates and scheduler-dependent cycle costs
  do not set the new request's draft depth. `STRIX_MTP_REQUEST_RESET=0` retains the earlier policy history
  for controlled comparisons. Target verification continues to decide every token and committed state.
- The window controller switches (`STRIX_MTP_CONTROLLER=survival`, `STRIX_MTP_ACTIVE_ROWS=1`,
  `STRIX_MTP_DRAFT_VOCAB=2`, the window folds) apply to served requests as on the command line. The draft
  vocabulary is chosen per request from that request's prompt.
- `n` > 1, images and audio are not supported. Logprobs use the resolved host selection route described above.
- Plain HTTP/1.1 (keep-alive, chunked responses); put a TLS proxy in front for remote access.

## Independent-session file route (`--serve-prefix-u32le`)

A separate, older route serves a fixed list of token files in one process without HTTP.

Source implementation only. No native equivalence, concurrent execution or
throughput qualification is established by these interfaces.

The decoder accepts repeated `--serve-prefix-u32le FILE` arguments with
`--generate N`. Each file is a nonempty literal sequence of little-endian uint32
token IDs, bounded by the selected context and vocabulary. Ordinary positional
prefix tokens are excluded on this route. All files are parsed before any
request is submitted. The decoder owns one shared mapping and model residency;
weights are not cloned for a request.

Required reservation controls are `--serve-memory-budget-bytes N`,
`--serve-request-budget-bytes N` and `--serve-headroom-bytes N` (zero headroom
is an explicit choice). `--serve-concurrency 1|2|4` defaults to one. At most64
requests are retained. These are caller-declared virtual backing/request budgets,
not a physical-memory fit guarantee. The request floor checks state, scratch,
selector payload and token storage; allocator/driver/stream/metadata overhead
requires the declared additional budget/headroom. Completed handles keep their
reservation and owners charged until released.

Each accepted factory allocates its own state, scratch, PLE backend/prefetch,
stream and sticky D1 completion record, with a fresh allocation epoch. All
operator resources and arenas belong to that request. Its production backend
validates finite output and D1 completion before committing state. Prepared mode
is passed explicitly to that backend; overlap, islands, graph/integer, paired
prompt routes, prefix roundtrip and single-session RAW/reference/final-state or
catalog-inventory output are rejected rather than silently disabled.

Concurrency two/four means independent C1 sessions on scheduler workers and
owned streams, not cross-request tensor batching. One uses the inline path with
no batch-fill wait. The existing token cursor keeps EOS, inclusive output and
pending-last-token semantics; prefill chunks, FIFO/decode priority with aging,
queue timeout and cancellation act only at completed backend boundaries. The
public scheduler exposes cancellation/timeouts; this file-list CLI drains its
fixed submitted family and reports every terminal result.

Setup uses existing guarded allocations and global HIP synchronization, so
submitting another request can wait for GPU work already active. This can limit
observed concurrency. Two streams or workers do not prove physical overlap or a
speed gain. Host service intervals may overlap; they must not be summed into GPU
busy time. The receipt distinguishes queue/service/wall/TTFT and fixed log2
inter-token histograms; histogram quantiles would be bounds, not exact percentiles.

Optional telemetry uses one counter owner per request and binds it on the worker
thread for that request's serialized calls. No non-atomic counter is shared by
concurrent workers. Per-request reports cover factory/worker calls before owner
teardown; the enclosing main-thread report covers remaining setup/teardown.
The normal decoder route creates no scheduler or extra request storage.

### Explicit shared prefix cache

The optional shared pool requires all four positive limits:
`--serve-prefix-cache-entries N`, `--serve-prefix-cache-snapshot-bytes N`,
`--serve-prefix-cache-capture-peak-bytes N` and
`--serve-prefix-cache-max-tokens N`. The token bound cannot exceed the selected
context. Omitting these options preserves the existing no-cache route; partial,
duplicate and out-of-serving controls are rejected.

The pool's checked reservation includes retained snapshot payload, the declared
capture peak and bounded entry/token bookkeeping. It is charged to the scheduler
shared reservation before the pool or request factories are constructed, in
addition to the shared model and caller headroom. Snapshots retain independent
allocator owners rather than retaining a completed request's scratch/state/stream
outside its request reservation. Cache identity and complete-state restoration
remain the production backend's responsibility; an exact-prefix hit carries an
unconsumed terminal token, while a proper-prefix hit continues its known suffix.
The optional telemetry wrapper forwards both cache hooks unchanged.

Receipts report actual pool lookups, hits, restored logical inputs, capture
attempts, inserts, skips, failures and currently retained entries/bytes. These
are operation counters, not measured time savings or physical-memory traffic.
This integration remains source-only until final validation.

### Independent bounded PLE row caches

`--ple-cache-rows N --ple-cache-budget-bytes N` requires both positive explicit
limits and resolved planned PLE I/O. It reserves payload and private row tags
once per backend. Serving adds the pure requested amount to each request's
minimum reservation before factories allocate it. Normal setup records the same
reservation before construction. Each enabled backend retains an independent
source object that itself retains the immutable mapped model; there is no cycle
back to its request owner. Disabled caches preserve the existing borrowed source.
Counters report actual hits/misses/inserts/evictions/invalidations and retained
rows/bytes per backend, not measured I/O time savings. This cache is distinct
from the shared full-state serving prefix pool.

## Scheduling qualification

CPU: `strix-batch-service-handoff-tests`, `strix-batch-nested-recipe-tests`, `strix-http-slot-capacity-tests`,
`strix-mtp-window-controller-tests`, `strix-serve-api-tests`. GPU (pinned model and MTP arguments after `--`,
output outside the checkout; `STRIX_PROMPT_STATE_DIGEST=1 STRIX_PREFIX_CACHE_DIGEST=1` and
`STRIX_PREFIX_CACHE_DIGEST_COMPONENTS=1` add `prompt_state_digest`, `final_state_digest`, `final_state_position`
and per-component `final_state_components` to the responses):

```bash
# alone vs service off / cancel only / service at every layer; then a short request nested into an idle long one
python3 tests/batch_service_gpu_gate.py --short SHORT.ids --long LONG.ids --out "$OUT/service" --service-ms 1 -- CMD...
python3 tests/batch_service_gpu_gate.py --scenario nest --reference "$OUT/service/alone.json" \
  --short SHORT.ids --long LONG.ids --out "$OUT/nest" -- CMD...
# --http-batch 0 references vs one AUTO server at 1/2/4/8/16 clients (lazy growth witnessed)
python3 tests/http_slots_gpu_gate.py --prompt A.ids --prompt B.ids --out "$OUT/slots" --modes auto -- CMD...
```

The gates compare full token IDs, the final frontier and every committed target component and require
witnesses (`service_rounds`, `nested_admits`, `cancel_points_cancelled`, `lazy_slot_builds`). Prompt digests
materialize accepted GDN window rows before hashing, as do prompt-end exports before copying state.
Every serial prompt row also materializes pending rows before consuming recurrent/conv state, including
fallback after history exhaustion. Serial decode entry folds first too; accepted windows fold their predecessor.
`STRIX_FINAL_STATE_DIGEST=1` with `STRIX_PROMPT_STATE_DIGEST=1` registers state and reports the final committed
digest/frontier (and components with `STRIX_PREFIX_CACHE_DIGEST_COMPONENTS=1`) without observing prompt state.
It logs prompt execution receipts without a prompt digest, so observation-free continuation gates can require
the actual accepted/refused window boundary. The general slots gate retains its prompt-digest
exclusion pending GPU qualification across all prompt/predictor layouts; same-route checks still compare it.
The opt-in `tests/http_slots_gpu_gate.py --export-roundtrip` arm explicitly compares prompt digests for a
T+4 terminal-R4 prompt, every saved component, and a verified batch-export -> single-import continuation.
The `--window-state-arm digest-only`, `live-continuation`, and `history-exhaustion` arms isolate prompt
digest materialization, live batch continuation, and R8 acceptance followed by refused R4/serial fallback.
The last also compares against a live batch producer exporting at T+8 before the fallback.
See the gate docstring for separate short commands; missing cold-window/import witnesses fail the gate.

## DeepSeek V4.1 pair scheduler

Launch one `strix-deepseek41-serve MODEL.gguf --rank R --transport usb4auto`
per node with matching model, profile, context and assets. Resident slot capacity is
unchanged. With device prefill, `STRIX_DS41_SERVE_SCHED=interleave` remains the
default; `off` runs whole prefills and `defer` retains deferred admission.

`STRIX_DS41_SERVE_SERVICE_MS=250` is the default soft target for the next completed
service boundary while requests decode beside the prompt or another admitted prompt waits
for its prefill. Rank 0 predicts a positive number of finest resumable units and sends that
budget to rank 1. While no request decodes and no other admitted prompt waits, rank 0 sends
budget 0 instead: both ranks run the rest of the current piece at once (same bits; up to
16384 rows with `--prefill-rows auto`), so a request that arrives meanwhile, an abort or an
import waits for that piece. `STRIX_DS41_SERVE_IDLE_WHOLE=0` restores bounded units while idle.
The ordinary piece endpoints and cache snapshots stay unchanged. Arrivals, status
and cancellation are inspected between resumes. One indivisible unit, decode sweep,
state import or allocation can exceed the target; it is not a latency guarantee.
With no explicit legacy unit controls, zero restores the previous coarse interleave:
whole default pieces while idle, 256-row units beside decoders. It disables finest
resumable service, not interleaving or short-first ordering. Negative, nonfinite and
overflowing targets are rejected before device work.

Any explicit legacy unit control overrides the service target, including zero.
Positive `STRIX_DS41_SERVE_SCHED_UNIT_MS` predicts resumable units and takes precedence
over `STRIX_DS41_SERVE_SCHED_UNIT_BUDGET`; zero milliseconds preserves a positive
fixed budget. A positive fixed budget counts resumable units. Otherwise
`STRIX_DS41_SERVE_SCHED_UNIT_ROWS` controls row units, including while idle, using
the existing row partition rules (default 256). Short prompts below
`STRIX_DS41_SERVE_SCHED_MIN_ROWS` (default 64) retain the direct whole-prefill path.

`STRIX_DS41_SERVE_SHORT_FIRST=1` selects the shortest queued prompt at admission and
the shortest waiting device prefill. Ties retain arrival order; each stage permits
at most eight overtakes of older eligible work. Zero orders eligible work by submission
time, which can differ from the old running order after deferral, pool waiting or
imported-state replacement. Pool waiters retain their reservation retry order and parked states still match by prompt.
There is one active device prefill: a later short prompt can be admitted at a service
boundary, but waits for that task to finish or cancel. No nested prefill lane is enabled.

An HTTP abort cancels the active resumable task at the completed boundary on both
ranks without computing the remaining piece (bounded units; an idle whole piece completes first). The abandoned sequence cannot extend
until a fresh run or a validated complete state import restores it. Import is refused
while a task owns the instance. Rank 1 applies rank 0's budget and abort controls;
it does not select them from its own clock. Both ranks exchange and compare completion,
rows and within-piece unit position at each service boundary before further collectives;
a mismatch poisons both ranks.

`GET /v1/capabilities` includes `scheduler` with effective mode, service target,
effective unit target, budget source, short-first and nesting availability.
`GET /api/status` includes these under `prefill`, plus `max_unit_ms` (maximum measured
indivisible device unit), `cancellations` (active task aborts, including imported-state
replacement) and `fallback_reason`. Explicit legacy budgeting, disabled service and
non-interleave modes have distinct fallback reasons. Direct short prompts and coarse
row runs do not produce fine-unit timing. Cached, imported and computed prompt
accounting remain disjoint.

The DS prefix policy now defaults to bounded `agent` host retention with in-run snapshots
(reach 0), plain-decode sessions, and disk/before-tail OFF. `--cache-profile off` alone selects
the prefix baseline; `--no-prefix-cache` is a hard veto. See the
[DS-specific controls and resource contract](prefix-state-cache.md#deepseek-pair-agent-prefix-cache)
and [qualification protocol](ds41-prefix-cache-qualification.md). Native qualification (two Strix Halo nodes over 2x USB4 Gen3, PerfOpt kernel, production packs): prefix tool exactness on both ranks at 6000 and 8192 rows (max rows 4096 / 2048): miss, same-prompt, warm, in-run and byte-suffix resumes equal a fresh prefill (logits and state digest), 16 decoded tokens equal, 0 differences. Serve timing, a fresh process per arm, `--cache-profile off` vs the default in ABBA order over two turns (n = 4 pairs per size), generated text and token IDs identical in all 8 processes: cold 2K miss -0.14 s mean (pairs +0.09 / -0.46 / -0.13 / -0.08 s; not slower), an 8K prompt that shares its 2K prefix with the previous request -3.63 s (-24 %, 15.0 -> 11.2 s mean; every pair faster). The older qualification below predates the new policy.

The new service, cancellation and ordering behavior has CPU contract coverage. On both ranks,
with the scheduler on, prefill logits, decode windows, batch rows and greedy tokens equal the
reference; resumable prefill task gates at unit budgets 1, 3, 32 and 250 equal whole prefills; served
requests return identical ids with the scheduler on and off, including with the host prefix tier on
(`--prefix-cache-mb`, partial and whole prefix hits), and cancel-then-reuse and whole-request repeats are
equal. No mixed-load gain is claimed. The scheduler is on by
default with off switches: `STRIX_DS41_SERVE_SERVICE_MS=0` disables fine
service and its mid-piece cancellation boundaries, `STRIX_DS41_SERVE_SHORT_FIRST=0`
disables short-first, and `STRIX_DS41_SERVE_SCHED=off` disables interleaving altogether.
Qualification requires the both-rank resumable prefill GPU gate below and the
both-rank HTTP serve exactness gates for cache miss, partial prefix hit, whole hit
and cancel-then-reuse. Import and pending-task abort are covered by the CPU scheduler
tests and the unchanged import validation; the HTTP import arm is pending a matched
state/message fixture.
Qualification must compare complete token IDs, logits and committed-state digests against isolated requests with
the same numerical identity. Cover cache miss, partial/whole cache hits, imported-state
replacement, cancellation followed by slot reuse, piece tails and supported transports.
Measure long-alone/short-arrival, decode/long/short arrivals and saturated-slot
cancel/refill loads. Record service overruns, cancellation release, per-class TTFT,
long-request progress, throughput, memory and engine/SSE gap distributions separately.
A short request arriving during an active prefill waits for the single lane; bounded
polling does not establish preemption or short-prompt TTFT improvement.

Run the existing resumable GPU gate on both nodes simultaneously, setting `R` to
0 and 1 respectively and retaining the same qualified profile environment:

```sh
for K in 1 3 32; do
  build/strix-deepseek41-prefill assets/model.gguf --rank "$R" --transport usb4-auto \
    --prompt assets/prompt.ids --rows 8192 --task-rows 2048,2053,8192 \
    --task-levels whole,layer,phase,finest --task-units "$K" --interleave 256
done
build/strix-deepseek41-prefill assets/model.gguf --rank "$R" --transport usb4-auto \
  --prompt assets/prompt.ids --rows 8192 --task-levels finest --task-budget 250 --interleave 256
```

This gate checks its existing logits/state/background-ID contract, not HTTP
cancellation or complete HTTP continuations. For the HTTP campaign, launch the
server on both nodes, keep request JSON and outputs outside Git, and use a fresh
server for the idle-arrival arm. The long request must exceed the direct-prefill
threshold and last long enough for the first unit to be observed:

```sh
build/strix-deepseek41-serve assets/model.gguf --rank "$R" --transport usb4auto \
  --context 16384 --http-port 8080
```

On the client, use `BASE` for the authorized HTTP origin and `OUT` for a private
output directory. Request files specify greedy streaming generation with the same
model and limits in every arm:

```sh
curl -fsSN "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  --data-binary @assets/long.request.json > "$OUT/long.sse" &
LONG=$!
# Run this recipe with STRIX_DS41_SERVE_IDLE_WHOLE=0 on both ranks (an idle whole piece has no fine units).
# After /api/status reports prefill.max_unit_ms > 0, while the long prefill is active:
curl -fsSN "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  --data-binary @assets/short.request.json > "$OUT/short.sse" &
SHORT=$!
kill "$LONG"  # disconnect the active long request at the next boundary
wait "$LONG" || true
wait "$SHORT"
curl -fsS "$BASE/api/status" > "$OUT/status.json"
```

Repeat with decoding streams already active, then with all slots occupied and
queued refill requests. Repeat cancellation before the first unit, mid-piece and
at piece boundaries, followed by a miss, partial cache hit, whole cache hit and
imported-state replacement. Compare every surviving request with its isolated
complete-ID/logit/state reference. SSE text and this curl smoke scenario alone
cannot establish exactness. Change one off switch per A/B arm and restart both ranks.
## Request retention and cache statistics

The [HTTP server](serving.md) accepts the same optional storage policy on
`POST /v1/chat/completions`, `/v1/completions` and `/v1/messages`:

```json
{"state_cache":{"retain":false}}
```

Absent means server policy; `true` permits ordinary retention; `false` allows hits
but prevents new prefix captures, host promotion of disk hits and publication of a
named turn's new frontier. Existing parked/published parent bytes remain unchanged.
The policy travels with queued single and batch jobs. It does not change token
selection, numerical profiles or scheduler inputs. A present field must be an
object containing boolean `retain`, otherwise the API returns HTTP 400 with
`invalid_request_error` and a field-specific explanation.

Authenticated `GET /v1/cache/stats` is available when prefix retention or named
session management is enabled, including prefix-only configurations. Existing
`ram_bytes`, `ram_budget`, `parked`, `disk_bytes`, `hits`, `misses`, `refusals` and
existing `engine` members remain. The additive `engine.state_cache` object uses
`StateCacheSnapshotV1` (`schema: 1`):

- `published`, monotonic `sequence`, and `age_ms` computed by the HTTP reader;
- `prefix_enabled`, `session_enabled`, `pool_present` flags;
- lifetime `prefix` counters, including token credit, capture/skip and eviction reasons;
- `named` hits, misses, refusals, parked count and no-store skips;
- `resident`, `host`, `disk` tier byte/segment/checkpoint/budget gauges;
- a reserved `pool` object whose fields are all zero with `pool_present: false`.

Session counters are zero when that feature is off. Host physical payload gauges
include registered buffers leased after directory eviction. Disk occupancy is
sampled by storage operations, never scanned by HTTP readers. Published responses
represent the last completed generation or management boundary; startup exposes
zero counters and configured budgets until the first publication. HTTP readers
load an immutable response without entering the generation runner, synchronizing
HIP or traversing payloads. This endpoint adds no observation arming behavior and
makes no changes to `/metrics` or `/api/status`.

See [prefix retention policy](prefix-state-cache.md) for budget fitting, fair/LRU
controls, exact boundary rules and the current qualification limit. The certified MTP
prefix path consumes the incumbent cold plan and resumes its whole-prompt cursor.
It captures asynchronously only at certified eligible boundaries. After native residency,
automatic admission fits `B = min(4 GiB, A/16, A - reserve)`, rounded down to 4 KiB,
using conservative available memory A and protected commitments in reserve. Context,
slot capacity and lazy growth never shrink. Other automatic configurations decline
by name and execute exactly as cache-off. Batch routes, sessions, serial CLI, YaRN
and other profiles remain unqualified for automatic reuse; explicit diagnostic
contracts stay separate. Refused RAM candidates add `plan_misses`, without accepted-hit
token credit.

Startup `qwen cache policy`, `--print-effective-config` and `/v1/capabilities`
report `cache_policy` requested/effective/source/reason/phase, including declines
and OFF. `/v1/cache/stats` includes that policy only while storage is active.
Inspection is allocation-free and pending (`effective` and B null); runtime reports
final admission or decline. The late deciding phase is `post-residency-ready`;
early terminal refusals use `pre-bridge-feasibility`. Batch policy requests
`l0-where-eligible` but declines unqualified families without changing scheduling.
Named-session storage is separately unqualified and never implicitly selects fixed
MTP or tiled-suffix arithmetic. Policy-selected requests keep their existing cache
bypass. See [automatic budget details](prefix-state-cache.md#controls-and-fitted-http-budget).

The stats endpoint returns the last immutable sample without entering the runner
or store. Startup seeds all legacy session fields and disk budget before listening.
Every completed request (including unnamed turns and refused operations) and
management operation publishes a sample. While generation is blocked, reads remain
nonblocking with respect to the runner and `age_ms` increases; a read never refreshes
`sequence`. These are last-sample gauges, not a synchronous live-store observation.
Ledger byte and segment gauges are read together. Eviction reason counters have
their own scope and do not sum to legacy `evicted`; explicit deletion leaves that
legacy counter unchanged.


### Prefix joining and restart qualification

The [certified prefix tier](prefix-state-cache.md#certified-restart-and-in-flight-joining)
can persist unnamed prefix checkpoints across process restarts. This is distinct
from named sessions. In-flight joining is on by default wherever the cache is on,
acts only on concurrent strict L0 certified batch routes, and uses the same
certificate and cold continuation as ordinary hits.
`--prefix-cache-inflight off` / `STRIX_PREFIX_CACHE_INFLIGHT=0` disables waiting;
`--prefix-cache-disk off` / `STRIX_PREFIX_CACHE_DISK=0` disables prefix disk reads
and writes; `--prefix-cache-durable off` / `STRIX_PREFIX_CACHE_DURABLE=0` disables
publication syncs. Effective config and capabilities disclose the guarded policies.
Automatic prefix budgets and batch L1 remain unqualified.

The native gate below requires a pinned native server command with model, MTP
receipt/state/reference assets, numerical settings, context, tiled prefill and an
explicit `--prefix-cache-host-bytes` budget. Keep all assets and receipts private.
The producer/joiner fixtures must share several full-T blocks and diverge before
their final prompt rows; the short fixture is unrelated. All must fit the context.
For T=8192, this experiment requires nesting off; a finer cache grid does not
qualify tail-to-T equivalence.

```sh
python3 tests/prefix_cache_gpu_gate.py \
  --producer assets/producer.ids --joiner assets/joiner.ids \
  --short assets/short.ids --timing --out "$PRIVATE_RECEIPTS/prefix-joining" \
  -- build/strix-hip-decode-token MODEL.gguf [PINNED_MTP_AND_PREFILL_ARGS] \
  --prefix-cache-host-bytes 4294967296
```

The driver runs cold references, mixed-arrival off/on/off ABA three times, and a
persisted producer followed by a new-process joining turn. It requires an actual
joining reuse witness and restart restore credit. Equality covers all token IDs,
every emitted raw FP32 vocabulary row through a coverage-counted SHA-256 chain,
prompt terminal logits and state, and final committed target/predictor state,
positions and generations/lineage. Missing fields or tripwire evidence fail closed.
It does not exclude predictor components to obtain a pass.

Repeat with partial terminal blocks/tails and shared-prefix divergences, no-store,
producer cancellation and immediate slot reuse, capture-credit refusal and service
admissions. Compare against cold at the identical numerical identity. With `--timing`, a second ABA x3 matrix disables diagnostic readbacks, checks
complete IDs against the exactness reference, and requires no slower producer,
joiner or unrelated-short median TTFT and no lower complete-workload throughput.
Keep capacity and startup settings fixed across arms; retain the private receipts.
No timing or native exactness claim is established by the host tests in this change.

Qwen `hc_output_wmma` is default ON. `hc_output_wmma` passed a pre-declared confirmation held-out set; its NLL endpoint is measured against a rounding-equivalent control whose per-case movement is capped at +-0.02 nats/token (a bounded control allowance, not an unbiased estimate of the uncapped difference); the absolute NLL change vs the exact reference also passed (-0.37 % NLL/token, upper bound +0.38 %). The off switch restores the exact output path.
Controls are server-level only: `STRIX_PREFILL_HC_OUTPUT_WMMA=0`,
`--prefill-disable hc_output_wmma`, or config `prefill_disable = ["hc_output_wmma"]`;
master `STRIX_PREFILL_EXACT_FP16=1`, `--prefill-exact-fp16`, or config
`prefill_exact_fp16 = true`. Restart the server; request-level controls return HTTP 400.
Fast HC with this option is excluded from MTP and nested admission pending qualification;
ordinary MTP/HTTP profiles retain their existing arithmetic. OFF restoration compares logits
and seven state payloads to public main; compiled, lineage and cache identity bytes may differ.
