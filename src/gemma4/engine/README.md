# Gemma 4 generation coordinator

This host/device product path executes decode `g4-v1`. Prompt prefill identity
is reported from the backend prefill route as `prefill_identity` in effective
config and `/v1/capabilities`: `g4-wmma-pf-v2` under `default` on Q4_K_M GGUF
packs once its `default` receipt is named (until then `default` runs exact prefill
and the receipt lists `pending g4-wmma-pf-v2 (exact runs)`), `g4-wmma-pf-v1` with
`STRIX_G4_PREFILL_IDENTITY=v1`, `prefill-exact` for the exact many-row or eight-row
routes (other packs, `quality`, or `STRIX_G4_PREFILL_IDENTITY=exact`), and
`prefill-a8` for `a8-many-row`.
Broader complete-workload qualification remains separate. Model, table, tokenizer and
drafter hashes appear in runtime receipts; private prompts and receipts do not
belong in the repository.

`Backend` extends the common `Target` with a configuration, identity receipt and
fresh `StreamState` factory. State supplies `begin`, `frontier` and a canonical
digest. A device implementation can recognize its own `KvTxn` without changing
the coordinator or the common contract. Assets and backend outlive coordinators;
coordinators are single-owner objects, with cancellation between steps.
The optional admission overload receives prompt plus output capacity; device
streams allocate only that global KV capacity. Existing backends may use the
default overload, which delegates to their fresh-state factory.

A backend may also offer exact many-row prefill (`prefill_chunk_rows`,
`prefill_rows`). The device backend's exact route uses canonical attention and
exact linears: a stream with more than eight prompt rows left prefills one chunk
(ending on a multiple of 512 positions) on its own, and the drafter receives the
same rows in groups of at most eight. Committed KV, taps and hidden rows equal
the eight-row
`forward_rows` passes, so this exact route reports `prefill-exact`.
`STRIX_G4_PREFILL_EXACT_MANYROW=0` (recorded in `off_switches`, with
`prefill_route` in the backend identity) restores the eight-row passes.

`STRIX_G4_PREFILL_IDENTITY=a8` opts into `a8-many-row` prompt prefill only with
decode `STRIX_G4_IDENTITY=g4-a8-v1`; otherwise prefill-exact runs with a log line.
Prompt KV and logits equal the a8 decode of the same rows, byte for byte,
on 512-row chunks. `STRIX_G4_PREFILL_EXACT_MANYROW=0` keeps the eight-row passes.

Each step calls `forward_rows` once with at most eight total rows. Every live
stream reserves at least one row; remaining capacity goes to exact prompt chunks
or explicitly enabled verify windows. Prompt ingestion processes all but the
last prompt token, which becomes the first anchor. After a verify verdict,
exactly the emitted count of target inputs is committed. The last emitted token
remains unprocessed. Committed taps append only after publication. Backend,
commit or context failures terminate affected requests rather than resuming
inconsistent state. Completed slots can be reused with a new generation ID;
their previous result must be copied before submitting another request.

Greedy is the default and uses capped logits with low-ID ties. The optional
`splitmix64-cdf-v1` sampling policy uses a private counter per stream. For draw
number `d`, starting at one, mix `seed + d * 0x9e3779b97f4a7c15` modulo 2^64:
xor-shift 30, multiply `0xbf58476d1ce4e5b9`, xor-shift 27, multiply
`0x94d049bb133111eb`, xor-shift 31. The upper 53 bits times 2^-53 form the uniform.
Sort capped logits descending, breaking ties by token ID; retain top-k if nonzero,
compute FP32 `declared_exp((logit-max)/temperature)` and an ordered FP64 sum,
then retain the smallest prefix reaching top-p mass. Select the first CDF bin
strictly above the uniform times retained mass. Every sampled token consumes one
draw, even with top-k=1. Greedy consumes none. Sampling forces k=0 and is a
declared token-changing request policy, separate from target arithmetic.

`DraftSession` supplies C's full eight-row proposals and staged context API.
The coordinator implements C's device-cycle publication protocol around its
single shared target pass: verify, truncate, stage accepted context, target
commit, draft publish, then expose statistics and tokens. It uses the same
`verify_greedy` as `run_device_cycle`; invoking a separate single-stream target
pass would break batching. A stage failure discards the target overlay. A
publication failure terminates the request. Sessions and transactions stay
stream-local, including across cancellation and slot reuse.

Explicit `--dflash on --draft-k K` supports k=0..7 within the aggregate row and
output budget. `STRIX_G4_DFLASH=0` disables proposals and context allocation;
`STRIX_G4_DFLASH=on` forces automatic requests to the fixed requested k.
Sampling remains serial. Full-block reservation is required even for k=1.

`stream-survival-cost-v1` learns independently in each stream and length bin
(boundaries 512/2K/8K/32K). At depth d, only prefix-eligible comparisons count.
Evidence decays by 0.98 per cycle; a Beta(1,2) prior supplies one success and two
failures. Conditional survival uses the posterior mean minus one posterior
standard deviation. Expected output is `1 + sum(product(survival[0..d]))`,
including the correction/bonus. Complete cycle costs and causal verify costs
are measured separately for every executed R. No row-cost interpolation or
unrelated prompt measurements admit an unmeasured window. Candidate cycle
mean plus one standard error is compared to plain mean minus one standard
error; drafting is admitted only when its expected rate exceeds plain by 5%.
Measured context maintenance is removed from plain cost to compare against
the bare OFF route. At least two candidate measurements and four initial plain
cycles are needed. New length bins start without borrowed timing evidence.

Calibration is an explicit, bounded exception to gain admission: after four
plain cycles, up to six probes run every other cycle, initially k=2 (or the
smaller request cap). Later probes take a pair of samples at the least-observed k
every sixteen serial cycles or thirty-two admitted cycles. The adjacent repeat
provides a fresh variance estimate instead of admitting a single timing sample.
Every sixteenth cycle refreshes
plain cost. Receipts distinguish probes from gain-admitted windows and expose
the decayed eligible/accepted arrays and measured costs. Calibration can make
a short request slower; it is not evidence of a speedup. C2..C8 auto stays plain
until a whole-batch model is qualified. Timings of overlapping streams include
the shared step and must not be added as independent wall times. State hashing
runs at completion/cancellation outside decode timing and on explicit live
receipt inspection; session allocation and
weight upload have a separate setup timer. The legacy aggregate `CostModel`
API remains available for diagnostics; it does not override stream admission.

`run_chat` uses the native adapter and incremental parser. A positive thinking
budget below max_tokens requires enable_thinking. A length cutoff at that budget
uses the checked template continuation for a second job, with the same sampler
counter and remaining output allowance. A natural stop does not continue.
The continuation re-prefills a fresh state; it does not claim prefix-cache reuse.
Malformed/incomplete control or tool frames fail explicitly at that boundary.

`Api` uses the existing portable JSON, HTTP writer and generation queue. One
batcher thread owns the backend for chat jobs: it admits up to
`STRIX_G4_SERVE_SLOTS` (default 8) queued requests into one long-lived
`Coordinator`, steps them together (one target pass per step, prompt chunks of
at most `STRIX_G4_SERVE_PREFILL_PER_STEP` requests per step, default 1,
round-robin) and retires each on stop, length, cancellation or its thinking
budget continuation. Each request's ids equal the same request served alone:
rows of different streams never mix, and selection and sampling draws are per
request. Socket threads write their own SSE deltas, so a slow client never
stalls the batch. `STRIX_G4_SERVE_BATCH=0` restores one chat job at a time.
Each phase receipt carries `serve_batch` (slot, admission step, steps, mean and
widest batch, steps that shared a prompt chunk). The family handler supports `/health`, `/v1/models`,
`/v1/capabilities` and `/v1/chat/completions`, including SSE, object tool arguments
on input (OpenAI JSON strings are parsed), JSON-string tool arguments on the
wire, separate reasoning content, conversational mode and budget continuation.
Chat selection stops at EOS, turn end or the tool-response marker; a speculative
window truncates before committing any later input. Receipts list these stop IDs.
Unknown generation fields, custom stop strings, penalties, constraints, logprobs,
n other than one and forced tool selection are refused. Complete tool frames
are streamed only after validation. Disconnects cancel at scheduling boundaries.

Host-only example, with relative asset placeholders:

```sh
strix-gemma4-decode model/ --rope-tables tables/ --ids prompt.ids --generate 16
strix-gemma4-decode model/ --rope-tables tables/ --tokenizer model/tokenizer.json --chat request.json
strix-gemma4-decode model/ --rope-tables tables/ --ids prompt.ids --streams 4 --dflash off
strix-gemma4-decode model/ --rope-tables tables/ --ids prompt.ids --draft draft.gguf --draft-rope draft.g4dr --dflash on
strix-gemma4-serve model/ --rope-tables tables/ --tokenizer model/tokenizer.json
```

The decode CLI builds on Windows and Linux. The socket executable is Linux-only;
the portable handler is tested on both. `--fixture fixture.npz --generate 16`
checks generated IDs against a private `prompt_ids` / `gen_ids` fixture and exits
2 on mismatch. `--print-effective-config` loads/validates read-only assets and
reports settings without creating a stream or executing the target. Raw prompt
encoding inserts no BOS/EOS; chat rendering owns those boundaries.

`STRIX_GEMMA4_DEVICE_BACKEND=ON` requires Linux, HIP and the
`strix_gemma4_device` target. The engine adapter implements the factory in
`runtime.hpp` over `gpu::DeviceTarget` and `gpu::DeviceStream`; select it with
`--backend device`. Device inspection does not initialize HIP or upload weights.
Receipts include graph, fusion, shared-weight, grouped-norm and packed-chunk
switches plus the running kernel and PerfOpt parameter. DFlash uses a resident
device drafter with the device target and a host reference with the host target.
The engine shares one immutable `DraftModel` and the target's original resident
tied embedding/head bytes across stream sessions. Source identity, geometry,
format and stride must match; an unsupported resident view uses one packed copy.
This preserves Q8_0 and BF16 coefficients without decoding or re-encoding them.
Runtime preparation uploads the model and prepares one reusable stream before
request timing. A bounded pool returns completed/cancelled sessions across
coordinator lifetimes and slot reuse; acquisition clears only context rings and
request-local state. Execution workspace is retained, never checkpoint data.
Poisoned sessions are dropped. The scheduler serializes all device calls.

`STRIX_G4_DRAFT_MODEL=0` restores private model uploads per new session;
`STRIX_G4_DRAFT_REUSE=0` disables the session pool;
`STRIX_G4_DRAFT_HEAD_SHARE=0` restores the immutable packed head copy.
These exact fallbacks are recorded in effective configuration/receipts.
Inspection initializes no HIP state; DFlash OFF/sampling creates no request
session. Assets/factory and target must outlive all leases. Concurrent HTTP clients
decode together; the C2..C8 draft planner applies to them as in the engine.
Final HTTP `strix.generated_ids` includes every emitted raw token ID for
streaming and ordinary replies, including control/stop tokens.
Frozen draft tables must cover the admitted context and full draft reservation.
The Linux product target compiles
the existing family adapter and shared HTTP/JSON/queue sources unchanged,
without linking unrelated engine-core implementations.

`strix-gemma4-gate` is an explicit full-model CLI, never a ctest. It performs
three interleaved C1 (or `--streams 4`) auto/OFF rounds of at least 256 greedy tokens, comparing
every raw ID and the final canonical state digest. Chat inputs are rendered
normally, then natural stop IDs are deliberately cleared to test the complete
raw prefix. This does not claim serving fidelity after an end-of-turn marker.
Each round reports decode throughput, request wall time, setup/prefill cost,
window histogram, calibration probes and acceptance. `--ignore-eos` provides
the same raw-prefix selection behavior for decode IDs; chat serving retains its
normal EOS, tool-response and budget boundaries.
