# Prefix state cache

Agent requests resend long, mostly identical prompts: a system prompt, tool schemas and the
conversation so far. The certified prefix cache is ON by default for the qualified single
tiled-MTP HTTP family: T=2048, default numerical profile, schedule, tails and model assets,
and context up to 16384. This covers `--http-batch 0` and automatic capacity resolving to
the actual single runner; an explicit batch runner with one slot is a batch family.
Admission requires all three row bits: GPU equality, resource/capacity and cost
qualification. The MTP contract is restricted to the bound production draft Q4
arithmetic, pinned auto vocabulary bytes, context vocabulary 4096, prefix verifier
rows 2/4/8, draft minimum probability 0.7 and default n-gram mode. Changed draft
arithmetic, vocabulary, verifier geometry, copy modes or state-transfer modes
refuse automatic reuse with `mtp-contract-unqualified`.
A later request whose prompt starts with a certified cached prefix restores the complete
model state at a tiled boundary and prefills only the rest. On this qualified family,
complete generated token IDs, raw logits and canonical state digests are bitwise identical
to cache-off execution under the same configuration, as required by the qualification gates.
Other automatic configurations decline with a named reason and execute exactly as with
the cache off. Explicit `--prefix-cache-host-bytes N` retains its existing behavior.

It runs on the tiled known-prefix prefill
(`--known-prefix-window --known-prefix-full-rows --known-prefix-prefill-rows T`), with serial
greedy decode or with MTP decode (see [MTP](#mtp)). Every other route rejects the options.

## DeepSeek pair agent prefix cache

These are the DS-specific `strix-deepseek41-serve` controls; the generic options below belong to
other serving routes. The executable now resolves omitted cache flags and `--cache-profile agent`
to a bounded host candidate: 512 MiB **per rank**, 256-token blocks, an 8192 MiB reserve on each
node after predicted capture allocations, in-run snapshots with reach 0, and session snapshots
only for plain R=1 decode. Before-tail and disk persistence stay OFF. This policy is the default. Native qualification (two Strix Halo nodes over 2x USB4 Gen3, PerfOpt kernel, production packs): prefix tool exactness on both ranks at 6000 and 8192 rows (max rows 4096 / 2048): miss, same-prompt, warm, in-run and byte-suffix resumes equal a fresh prefill (logits and state digest), 16 decoded tokens equal, 0 differences. Serve timing, a fresh process per arm, `--cache-profile off` vs the default in ABBA order over two turns (n = 4 pairs per size), generated text and token IDs identical in all 8 processes: cold 2K miss -0.14 s mean (pairs +0.09 / -0.46 / -0.13 / -0.08 s; not slower), an 8K prompt that shares its 2K prefix with the previous request -3.63 s (-24 %, 15.0 -> 11.2 s mean; every pair faster).
No historical session speedup is a measurement of this policy.

Both ranks agree fresh capture admission before planning or copying. A refusal removes cache
snapshot points and cache-specific run splits, including with an explicit legacy planner.
In-run runs keep baseline endpoints, even when the configured row limit is not block aligned;
far stable-prefix points are retained inside runs, never silently replaced by splits at reach 0.
The final aligned snapshot retains the required slot-handoff export. Captures still allocate,
copy and widen replay cones; admission and in-run capture are not free on misses.

| Control | Meaning |
|---|---|
| `--cache-profile off` | Supply no profile settings; alone restores the prefix-cache baseline. Explicit cache options remain active. |
| `--no-prefix-cache` or `--prefix-cache-mb 0` | Disable the entire prefix bundle. |
| `--no-prefix-disk` | Disable disk even when a directory/limit was supplied. |
| `--no-prefix-before-tail` | Disable the optional edited-tail checkpoint. |
| `--no-prefix-session-snapshots` | Disable captures during decode. |
| `--no-prefix-inrun-snapshots` | Disable in-run captures; with retention enabled the explicit legacy piece planner may add splits. |
| `--prefix-cache-mb M`, `--prefix-reserve-mb M`, `--prefix-block B` | Explicit host ceiling, reserve and token block size. |
| `--prefix-inrun-reach N` | Explicit distance limit; 0 never adds reach splits. Nonzero values keep legacy reach behavior. |
| `--prefix-cache-dir DIR` | Explicit disk opt-in. No directory means no writer or startup scan. |
| `--prefix-cache-disk-mb M` | Explicit per-tier ceiling; **0 means unlimited per tier**, never disk OFF. Without a directory, this explicit opt-in resolves `$STRIX_DS41_PREFIX_DIR`, then `$HOME/.cache/strix/ds41-prefix`. |

Negative switches, including a numeric host zero, veto conflicting positive options independently
of argv order. Other explicit values survive profile fills. Rows prefill and hetero-listen receive
no implicit cache settings; explicit incompatible requests still fail before loading. Implicit
sessions never disable auto-DSpark. An explicit, non-vetoed session request still prevents automatic
DSpark selection; explicitly combining sessions and DSpark fails before model loading.

Host ceilings use nonblocking kernel leases shared by service instances of the same OS user on a
node, capped at 1024 MiB in total, with 64 MiB reservation granularity. No host memory is
preallocated. Explicit larger host ceilings are accepted but effective retention is capped at
1024 MiB; a failed lease disables new retention on both ranks. The host accounting includes
per-layer vector descriptors and allocator allowances; it is not an allocator/RSS hard limit.
The conservative transient prediction includes retained copies, whole-prefix exports, serialization
and snapshot rings. Unsupported host locking fails closed.

An explicit directory gets a 4096 MiB tier ceiling unless overridden. Its namespaces and service
processes share a 4096 MiB directory quota, a 128 MiB outstanding-payload limit and a 16384 MiB
filesystem free-space floor. Queue reservations precede serialization; payloads remain charged
through the active write. Files pending deletion and temporary files count physically until gone.
Quota/space/peer refusal keeps RAM when possible. Each namespace/rank directory has one exclusive
service owner; a second writer is refused, and no sibling namespace is evicted. Inactive writer
reservations are ignored after a crash. Disk reads retain all-or-nothing peer restore and do not
promote into RAM without collective admission. The quota covers one configured service root;
**separate roots and OS users are not an aggregate node-wide disk safeguard**. Keep all service
namespaces under one root. Cross-root coordination and allocator-level resident-byte accounting
remain future work. Disk stays explicit and unqualified.

The old bundle can be requested with `--cache-profile off --prefix-cache-mb 8192
--prefix-cache-dir DIR --prefix-cache-disk-mb 65536 --prefix-before-tail
--no-prefix-inrun-snapshots`, plus `--prefix-session-snapshots` only without DSpark.
Those requested ceilings are subject to the new host and directory safeguards; this does not
restore the old 8 GiB resident capacity or claim non-regression. Host-only defaults lose restart
hits, and before-tail OFF loses the special edited-tail checkpoint. Prompts below one block still
cannot produce a checkpoint, so short-chat benefit cannot be assumed.

Startup prints resolved policy and disk-zero interpretation. `GET /api/status` exposes the policy,
effective host ceiling, admission/bypass counters and predicted transient peak, plus disk queue
bytes/peak and resource refusals. `STRIX_DS41_PREFIX_DIAGNOSTICS=1` adds exported bytes, counts by
snapshot source, executed prefill row-layers and planned cache splits; it adds no target kernels.
Unset/0 disables these diagnostic counters. Compare diagnostic arms separately from production
timing; reported bytes are host export bytes, not all device or allocator traffic.

Exactness concerns continuation-relevant state, fresh suffix logits and complete generated IDs
under the same numerical identity. Recomputed top-k, inactive ring slots and invalid carry contents
are excluded from the existing state digest; this is not equality of every serialized byte.
Unknown session logits do not count as an equality result. The prefix tool's optional
`--byte-check` compares canonical continuation bytes and full freshly computed logits; whole hits
receive a fresh-suffix diagnostic. Session and in-run state-only comparisons are labeled explicitly. Namespace/model/layout/checksum,
token comparisons, unknown-next-token protection and both-rank disk intersection remain mandatory.
See the [qualification protocol](ds41-prefix-cache-qualification.md) before making performance or
current native exactness claims. Every exact gain that meets its not-slower rule is promoted
without a minimum percentage threshold.

## Options

| Option | Meaning |
|---|---|
| `--prefix-cache on\|off` | Requested policy defaults on; `off` disables prefix caching. Automatic admission is limited to the qualified family and late memory fit. |
| `--prefix-cache-host-bytes N` | Explicit byte budget of the host tier (pinned host memory); existing behavior is unchanged. Not required for qualified automatic admission. |
| `--prefix-cache-dir DIR` | Optional disk tier: checkpoints are written behind the request and found by later processes. |
| `--prefix-cache-disk-bytes N` | Optional disk budget (oldest checkpoints are deleted first; 0 = unbounded). |
| `--prefix-cache-every N` | Capture interval in tokens, a multiple of T (default: 4096 rounded up to a multiple of T). The last eligible block boundary is selected; incomplete or uncertified boundaries are skipped. Certified async boundaries use stable snapshots. |
| `--prefix-cache-staging-bytes N` | Device snapshot cap; 0 defaults to one T-block of registered rows plus fixed state. Environment: `STRIX_PREFIX_CACHE_STAGING_BYTES`. |
| `--prefix-cache-then-prompt FILE` | Repeatable. After the main request, run these prompts (token IDs separated by commas or whitespace) in the same process, each from a fresh session, through the same cache. Serial-decode route only. |
| `--known-prefix-tail-rows T2[,T3...]\|none` | Independent of the cache: after the last complete T-row block, run the remainder as complete blocks of the largest tail size that fits (descending sizes, 16..1024, multiples of 8, less than T), and only the rest as serial rows. Default (D-056): the sizes of 1024, 256, 64, 16 below T, whenever the tiled prefill has full rows and an output head; `none` keeps the remainder serial. |

Diagnostics: `STRIX_PREFIX_CACHE_LOOKUP=0` captures without ever restoring (every request is cold);
`STRIX_PREFIX_CACHE_DIGEST=1` adds a SHA-256 of the whole prompt state to the MTP receipt.

The receipt gains `prefix_state_cache` (serial route: identity, geometry, main request, follow-ups,
store counters) or `mtp_tiled_prompt.prefix_cache` (MTP route).

## What a checkpoint holds

A checkpoint at block boundary P = kÂ·T is the session state after the first P prompt tokens:

- per-position rows of every attention layer: KV rows and raw indexer keys, positions [0, P);
- the state overwritten in place: GDN recurrent matrices, GDN and PLE convolution histories, the
  n-gram history and the position history;
- the session counters, state generations and the arithmetic lineage (every block interval with its
  producer identities);
- on the MTP route, the predictor's cache rows of the pairs (h_i, t_{i+1}) for i < P-1 and the target's
  wide hidden row h_{P-1}.

Per-position rows never change once written, so they are stored once per block (a segment) and the
checkpoints of one prompt share their common segments. The rest (about 111 MiB for this model) is
stored per checkpoint. Lookup returns the longest cached boundary P < prompt length (at least one
prompt token always runs, to produce the first output's logits); restore copies the rows and the
fixed state into the fresh session and republishes the recorded counters and lineage.

## Keys and identity

`identity = SHA-256` of: model content identity, executable digest (disk tier only), state layout
identity (includes the context), runtime compatibility identity, decode execution and kernel-plan
identities, prefill (sibling) execution and kernel-plan identities, tiled prefill plan identities,
prefill and decode numerical profile identities and their pair identity, T, the tail sizes and the context size.
The canonical `prefix-execution-v5` descriptor includes the resolved prefill option mask,
option-set identity hash and identity name. The existing v2 cache encoder binds that descriptor
once through `effective_execution`; certificates and session compatibility inherit it.
On the MTP route it also covers the sidecar content, the MTP plan and native profile identities and
the draft-kernel switches (the predictor's cache rows are built with the draft kernels unless
`--mtp-draft-exact`).

The v4-to-v5 execution descriptor transition changes cache keys even with the option disabled.
Existing keys become cold misses; numerical state bytes may still match. There is no in-place
migration of incompatible identity metadata.

`digest_k = SHA-256(digest_{k-1} || tokens of block k)`, `digest_0 = identity`. A digest names one
identity and one exact token prefix; the stored tokens are compared as well. Why each field matters:

- the tiled prefill reproduces serial rows bit for bit within one numerical profile (D-034), but with
  a split pair (D-033) complete blocks run the prefill profile and the unpadded remainder the decode
  profile, so which rows run which arithmetic depends on T (and the tail sizes). Checkpoints are only taken at
  block boundaries and the suffix of a restored request keeps the same block alignment as the full
  prompt, so the row-to-profile assignment is the same as without the cache;
- the decode identities do not change the state at a block boundary, but are part of the key so that
  no two configurations whose outputs could differ ever share an entry;
- context size and layout fix the state geometry; the executable digest keeps a disk tier from
  serving a state that another build produced.

## Storage

Host tier: pinned host memory, fair retention by default or explicit LRU within the byte budget.
A segment stays alive until its last checkpoint or reader releases it. Disk hits promote into
the host tier unless the request suppresses retention.

Disk tier: `seg-<digest>.tmps` (one block's rows) and `ckpt-<digest>.tmpc` (fixed state, metadata
and the list of segment digests), written to a temporary name and renamed. Headers carry identity,
digests and a 64-bit payload checksum; segment tokens are re-hashed on load and must chain to the
requested digest. A file that fails any check is skipped (counted) and the next shorter prefix is
tried. Writes run on a background thread; the process waits for them before exiting.
On Linux, segment and checkpoint files use the same unbuffered state-file I/O as session.bin
(O_DIRECT staged writes and read-ahead restore; buffered fallback when the filesystem refuses
O_DIRECT); file bytes are unchanged. `STRIX_SESSION_DIRECT_IO=0` restores buffered I/O. See
[State-file I/O](session-cache.md#state-file-io-linux).

## MTP

With `--mtp` and the tiled prefill flags the prompt itself runs through the tiled prefill:

1. Calibration: the first R prompt pairs (`--mtp-calibration-rows R`, default 8) run serially through
   the MTP target producer and are replayed against a golden of exactly those pairs (dumped with
   `--mtp-golden-dump` under the same flags). This qualifies and activates the predictor as before.
2. The target state and the predictor are reset; the longest cached prefix is restored.
3. Complete T-row blocks (then complete tail blocks) run tiled. A step observer copies every
   row's wide hidden state (the MTP target tap) out of the block arena, and the predictor's cache
   rows are built from those copies. Eligible certified boundaries use asynchronous snapshots
   without changing the priming schedule (see [Certified MTP subset](#certified-mtp-subset)).
4. A block (T rows or a tail size) that ends exactly at the last prompt row also computes that
   row's head (terminal-row output policy). The first draft cycle starts from that row: its wide
   hidden state and FP32 logits mint the pending target ticket (`HipMtpTargetHidden::adopt_prefill_row`,
   same first-max selection as a serial row), so no prompt row runs serially.
5. Otherwise the remaining rows, at least the last prompt row, run serially through the MTP target
   producer, whose last ticket starts the first draft cycle. Decode is unchanged (adaptive windows).

`STRIX_MTP_TILED_START=0` restores the previous handoff (the last prompt row always serial; a prompt
of exactly kÂ·T tokens then runs its last T rows through the tail blocks and serial rows). The receipt's
`mtp_tiled_prompt.tiled_start` says which handoff ran. A split numerical-profile pair is rejected on
this route.

## Measured

See docs/decisions/decision-register.md (D-035).

## Prefix retention policy

The [prefix state cache](prefix-state-cache.md) retains exact, complete tiled
boundaries. Explicit serial and MTP cache configuration leaves the cold tiled plan unchanged:
store absence, occupancy, quota and retention do not select merge, balance or predictor
priming. A host-only enumerator supplies the same whole-prompt units and continuation
cursor with storage enabled or disabled. Automatic host budgets still wait for
native route qualification. No resident pool is implemented
by this policy slice, and CPU policy tests do not establish GPU output qualification.

### Certified serial subset

The serial tiled route uses the incumbent `KnownPromptChunkCursor` for execution and
for a CPU-only walk of the whole prompt. Cache presence never disables deferred,
merged or balanced scheduling. Each executed unit records absolute bounds, owner
width, T-owner status, balanced/merged class, deferred completion and head intent.
A checkpoint requires a contiguous prefix of ordinary, completed, solo full-T units
from position zero, matching the preflighted trace and ready generation/lineage
metadata. Balanced, merged, deferred, tail and serial prefixes decline capture.

The versioned full-T certificate records the serial route, resolved plan and profile
identities, each producer unit's class and expected state generations. Its namespace
also binds the canonical effective-execution signature, separately from MTP and batch
reuse. Lookup checks the certificate and complete checkpoint before any state copy.
The consumer cursor must accept the boundary in its original whole-prompt plan;
restore advances that cursor without backend calls and executes its original suffix.
It never constructs a new request from the remaining tokens.

Serial capture retains the synchronous, fenced snapshot path. Asynchronous serial
capture declines as `serial-async-capture-unsupported`; no unordered snapshot is
submitted. Certified serial checkpoints can use the explicitly configured prefix disk tier.
Old uncertified disk/RAM entries and segment grids smaller than T cannot authorize
reuse. Baseline retains the original legacy disk path.

Request receipts and published cache statistics expose named `certificate_declines`.
Reasons distinguish unsupported plans/boundaries/units, mismatched producer or state
metadata, missing certificates, unsupported whole-prompt continuation, disabled
lookup and absence of a certified checkpoint. Effective configuration lists the
protocol and refusal names. CPU schedule, continuation and copy-submission tests do
not establish native token/logit/state equality; automatic qualification stays gated.

### Certified MTP subset

The new MTP path uses a distinct full-T plan-family identity covering the existing
model/profile/layout identities plus resolved schedule, output policies, tiled-start
and priming flags. In-memory producer certificates record each full-T unit through P.
Segments share only within this family. Balanced units (including full-sized balanced
units), merged units and tails are conservatively excluded from certified prefixes.

Lookup leases candidates in decreasing P order. It checks tokens, producer proof,
metadata/lineage, section geometry and chain before any device write or accepted-hit
accounting. A rejected candidate increments `plan_misses`, then lookup tries a shorter
candidate. Restore resumes the cursor computed from the whole prompt, preserving its
cold suffix instead of planning again from the remaining length. Partial-copy failure
keeps the existing poison/quarantine behavior.

Capture requires an actual incumbent boundary, 0<P<n, P divisible by the
segment grid, certified ready target metadata and successful target completion
validation. It never creates a numerical unit or forces synchronous priming.
Batch certification is a separate experimental opt-in described below. Grouped/SMALL
producer prefixes, named-session suffixes and imported/exported state remain outside
the certified subset. Deferred target validations still pending at the boundary skip.

A process-wide slot admits at most one asynchronous snapshot descriptor. Admission
reserves physical host credits and allocates every pinned destination, device staging
range and event before submitting copies. No credit, an occupied slot, allocation
failure before submission, or an oversized candidate skips without waiting. The
plan enumerator must identify the next mutator as a block of the same T owner,
and its actual stream handle must equal the snapshot producer stream. Tail,
window, serial, group, handoff or unknown next owners skip the boundary; no producer
wait is added to save it. Only under this same-stream condition do target snapshot
reads precede the next block's writes by stream order. An incumbent partial block
of that same T owner is also safe by stream order.

Automatic pooled capture on the admitted synchronous single-runner path splits
submission at P. Target production is followed by the target snapshot on its
producer stream, target-ready, then target D2H on the download stream. These
target downloads are queued before the original predictor prime and held update.
After that prime/update, metadata, host fixed state and the producer certificate
are frozen and checked at the complete P/P-1 frontier. Predictor snapshots then
follow on the predictor stream, with predictor-ready and predictor D2H on the
same download stream. Target-only completion cannot publish a checkpoint.

The explicit/legacy combined path retains per-capture resources: host history and
metadata are copied at P, predictor submission follows priming and the incumbent
held copy, and downloads begin after both snapshot-ready events.
For both paths, the existing MTP worker submission copies predictor rows and held
hidden after priming and the incumbent held copy, then records its ready event.
The next predictor/held writer must use that same MTP stream. The next target block
uses the spare tap buffer; the incumbent priming drain joins the worker and completes
the MTP stream before tap reuse or a synchronous held write. Unknown predictor
ordering also refuses capture. `capture_skips_cross_stream` counts these refusals
before snapshot allocation/submission (no-store requests count only no-store skips).
One side D2H stream reads only staging. In the automatic split it waits on
target-ready before target D2H, then predictor-ready before predictor D2H; in the
combined path it waits on both ready events before either group's downloads.
A single final done event covers both groups. Done plus CPU commit/cancellation
handoff gates publication, resource retirement and reusable-slot admission.

The normal priming drain validates the P-1 frontier and commits the descriptor.
Service points query its D2H completion event; only committed, successfully completed
descriptors publish. Cancellation or failed validation discards them after completion.
Descriptor ownership survives prompt destruction and slot reset. Shutdown explicitly
drains outstanding D2H. HIP submission/completion/drain errors are fatal and retain
all descriptor backing in quarantine. Capture adds no device fence or host completion
wait on the request path; the incumbent producer/priming waits remain in place.
Snapshot copies, allocation and bookkeeping still have costs.

`--prefix-cache-staging-bytes N` (`STRIX_PREFIX_CACHE_STAGING_BYTES`, CLI wins)
bounds device staging. Zero, the default, uses one T-block of all registered rows
plus the aligned fixed registry. Live staging bytes charge only missing row slices
and fixed device bytes of an active descriptor. Automatic pooling separately retains the entire staging
cap across requests, including when live staging bytes are zero. Larger catch-up
candidates are refused. Pinned destinations and
private row leases remain inside the existing host allocation ledger. Stats expose
`staging_budget_bytes`, `staging_bytes`, `staging_peak_bytes`, `staging_cap_skips`,
`capture_busy_skips`, and `row_prefetches`; pool `move_staging_bytes` mirrors live
staging. Per-request capture counters report reaped work; async `capture_ms` is
descriptor elapsed time, including overlap, rather than a request stall measurement.
It sums all reaped descriptors, including row-only prefetches; dividing it by
`captured_checkpoints` does not measure the cost of a checkpoint. A descriptor
completed on the download stream may be reaped only after another native block.
This bounded snapshot workspace does not implement a resident KV pool.

When checkpoint cadence exceeds T, intervening ordinary full-T boundaries may stage
rows only. After normal commit plus D2H success, these segments remain private to the
request's certified row chain; they do not publish a checkpoint or enter disk storage.
The next selected checkpoint reuses these charged row leases. This preserves both
the checkpoint cadence and the one-block staging default. A delayed/busy descriptor
can prevent that preparation; missing slices exceeding the cap still skip.

Certified checkpoints use the existing v1 store with a tagged certificate envelope
in its opaque metadata section when a disk directory is explicitly configured.
Existing segment files and baseline capture/priming behavior remain unchanged.
No old uncertified RAM or disk checkpoint is accepted by certified lookup.

With schedule 12, pipeline 255, default cadence (4096 rounded up to T), tiled-start
and terminal heads, the host shim checks these opportunities. Counts assume timely
D2H completion, available credits and no eviction; they are not promised native
retention counts or GPU qualification evidence.

| Prompt tokens | T | Safe selected checkpoints | Private row preparations | Cross-stream/owner skips |
|---:|---:|---|---:|---:|
| 1280 | 2048 | 0 | 0 | 0 |
| 6000 | 2048 | 1 (4096, next partial T on same stream) | 1 | 0 |
| 8300 | 2048 | 1 (4096) | 2 | 1 (8192, next tail) |
| 10300 | 2048 | 2 (4096, 8192) | 2 | 1 (10240, next tail) |
| 10400 | 2048 | 2 (4096, 8192) | 2 | 1 (10240, next tail) |
| 8192 | 2048 | 2 (4096, 6144) | 1 | 0 |
| 32768 | 2048 | 8 (4096 multiples through 28672, then 30720) | 7 | 0 |
| 1280 / 6000 / 8192 | 8192 | 0 | 0 | 0 |
| 8300 | 8192 | 0 | 0 | 1 (8192) |
| 10300 / 10400 | 8192 | 1 (8192, next partial T on same stream) | 0 | 0 |
| 32768 | 8192 | 3 (8192, 16384, 24576) | 0 | 0 |

These are independent retaining cold requests, with the ladder 1024/256/64/16.
A 10300-token no-store request prepares/publishes zero rows/checkpoints; it may
still restore a compatible checkpoint retained by an earlier request.

With `STRIX_PREFIX_CACHE_DIGEST=1`, each prompt receipt emits
`terminal_logits_digest`: SHA-256 of the raw FP32 vocabulary row at its last
prompt position, or JSON null when unavailable (comparison gates must reject null).
It uses the actual tiled final logits, last accepted prompt-window ticket, or
serial ticket without changing route. Window hashes are retained before producer
scratch reuse. The existing `final_logits_digest` and `serial_logits_digest` fields
and their switches retain their meanings. Beside `state_digest`, `terminal_metadata`
reports committed tokens, next position, the canonical LE generations/lineage
digest (including validity and taint), and the predictor frontier (next source
position, P-1 at prompt completion). These diagnostics add readbacks only when
enabled and do not qualify automatic caching or replace authoritative full-row
and state comparisons.

For the production BF16 geometry, fixed device state is 116,400,128 bytes
(aligned fixed registry 116,400,640 bytes, including host histories/padding).
Selected T=2048 captures with the preceding rows prepared stage **177,741,824 B**;
the first row-only preparation stages **61,339,392 B**, later ones **61,341,696 B**.
For T=8192, the first selected capture stages **361,764,608 B**, later captures
**361,766,912 B**. Default caps are **177,742,336 B** and **361,767,424 B** respectively.
These are registry/formula calculations, not measured device peaks; other storage
geometries use their actual registered sizes. P=n remains excluded.

### Controls and fitted HTTP budget

`--prefix-cache on|off` controls retention (default on). `STRIX_PREFIX_CACHE=0|1`
(or `off|on`) supplies the environment default; CLI wins. Explicit
`--prefix-cache-host-bytes N` retains its existing meaning. Outside HTTP serving,
there is no automatic host budget.

Automatic admission is qualified only for the single tiled-MTP HTTP family described
above. Batch routes, named sessions, serial CLI, YaRN and other profiles remain
unqualified for automatic caching. Their explicit controls retain their existing
contracts; an ON request does not bypass qualification or enable numerical modes.

The automatic budget is decided after native residency and calibration, once target,
predictor, window and other native owners are ready. With fresh conservative host/UMA
availability A, the shared host budget is:

```text
B = align_down(min(4 GiB, floor(A / 16), saturating_sub(A, reserve)), 4096)
```

The reserve covers protected host headroom, remaining cache-off commitments (including
lazy slots), bounded metadata/uncertainty and device snapshot staging. Fit also checks
the shared UMA ceiling, device headroom and room for a complete checkpoint and its
working set. Unknown observations, unsafe fit or setup failure decline by name before
optional cache work. The cache never reduces context, slot capacity or lazy growth,
or changes the numerical profile, schedule or tails. Certified storage is RAM-only;
explicit disk configuration retains its existing contract.

The startup `qwen cache policy` record, `--print-effective-config`,
`GET /v1/capabilities` expose `cache_policy` with
`requested`, `effective`, `source`, `reason` and `phase`, plus family and budget details.
`GET /v1/cache/stats` includes `cache_policy` only while storage is active.
Declines and OFF are reported by startup, inspection and capabilities; they do not
add a stats route.
Read-only inspection exits before device work: automatic `effective` and B are null,
with `pending-post-residency` and phase `inspection`. Early terminal refusals use
`pre-bridge-feasibility`; the deciding fresh late fit uses `post-residency-ready`.
Runtime records report the final admission or named decline, such as
`fitted-host-budget`, `batch-scheduler-family-unqualified`, `no-reuse-route` or
`serial-capture-cost-unqualified`. A decline preserves cache-off execution.

`--prefix-cache-retention fair|lru` and `STRIX_PREFIX_CACHE_RETENTION=fair|lru`
select the host victim policy; fair is the default. `--baseline` or
`STRIX_BASELINE=1` disables the automatic budget and selects lru; explicit prefix
options (host bytes, directory, cadence and follow-up prompts) remain unchanged.
`--prefix-cache off` or `STRIX_PREFIX_CACHE=0|off` fully disables the cache, including
explicit prefix options. Automatic declines leave explicit options unchanged.
Outside baseline, effective configuration includes `prefix_cache.enabled`, `source`
(`auto`, `explicit`, `off`), `budget`, `retention` and `reason`.

### Fairness and lifetime

A caller-supplied conversation key takes precedence; unnamed checkpoints use their
first complete block digest. A checkpoint can acquire up to 16 conversation
memberships through validated hits. The fair directory is bounded to 256
checkpoints; a full directory evicts an
unleased entry by the fair order, or skips admission when all entries are leased.
LRU has no directory entry cap and preserves the legacy victim order. These keys
are retention groups, not
security or tenant boundaries.

Within a conversation, a branch uses the first block after the deepest digest shared
with another retained checkpoint. If that checkpoint ends at the shared ancestor,
its terminal digest identifies the provisional branch. Associations are recomputed
after removals. Fixed/private bytes are charged to their memberships; physical
segments shared by conversations are charged fractionally across those memberships.
The physical byte ledger counts a payload once.

For budget B, conversation cap is `min(B, max(one full checkpoint, floor(B/3)))`.
The branch cap is `min(conversation cap, max(one branch checkpoint,
floor(conversation cap/live branches)))`. A lone conversation may borrow otherwise
unused capacity. Eviction occurs only under byte-budget or fair directory pressure;
conversation/branch caps order victims and never force eviction with spare capacity.
Under pressure, competing conversations reclaim borrowing first. Over-cap branches,
then over-cap conversations, then other entries supply victims. Victim score is
`marginal bytes * (1 + age epochs) / (1 + successful reuse count)`; larger scores
leave first. Ties use older access epoch, then digest. One epoch is one capture
(including a duplicate insertion) or validated touch across **all conversations in
the store**. It is neither wall time nor a per-conversation turn count. Misses,
rejected eligibility, directory probes and segment lookups do not advance it.
Epochs are deterministic, without per-token clock reads. Marginal bytes are recomputed after every removal.

Idle conversation groups also supply rank-2 victims alongside over-cap
conversations. This requires **no validated reuse during the current retained
lifetime**, a newer private-prefix successor, and a newest retained insertion or
touch more than 32 epochs behind the newest group member. An incoming capture
contributes the next epoch. Age alone never establishes eligibility.

A successor must have begun its retained lifetime later than the target. Their
first blocks must share an exact, partial token prefix longer than the target's
overlap with every other retained group, excluding that proposed successor.
At least one other group is required to establish this private-history boundary.
Equal complete first blocks decline: a common multi-block system/tools header
does not establish supersession. The boundary can fall inside the first block;
requiring a shared complete first block would identify the same unnamed group.
This is a content relationship, not proof that an older branch will never return.

Group birth and recency use unsigned elapsed ages from a common reference across
epoch rollover. Touching an old group does not make it a newly born successor.
A validated hit protects the entire group even if that hit checkpoint leaves
before its siblings. Birth, reuse and prefix-order metadata disappear with the last
retained association. Weak root references add no payload leases. Exact lexical
neighbors and cached prefix overlaps avoid a pairwise group comparison in eviction;
an incoming group is merged in one linear scan at the configured block size.
An expired root or a group containing inconsistent first blocks conservatively
declines reclamation; inconsistent ordering stays declined until the store empties.
There is no history of departed groups, new capture, wait or per-token clock read.
Branch ranks, marginal-byte scores, leases and byte admission remain unchanged.
Idle-group reclamation is default ON for stores containing only certified prefix
checkpoints. A fair store using named-session parking keeps its complete original fair ranking,
including during its first park. Any resident or incoming uncertified checkpoint
also suppresses idle demotion. HTTP named-session parking continues to use its separate LRU store.
`STRIX_PREFIX_FAIR_RECLAIM_IDLE=0` restores the original fair victim order when the
application builds the store policy; unset or `1` selects idle-group reclamation.
The default ON rule requires a private-prefix successor, no reuse in the retained
lifetime, and more than 32 global epochs of recency separation; one epoch is one
capture or validated touch across all conversations. Pair-exclusive sub-block
boilerplate in other workloads can create a false successor. `/v1/cache/stats`
reports `evictions_superseded` for removals whose rank was raised by this rule
(excluding already over-cap conversations or higher-priority branch-cap victims).
The library takes an explicit policy and never reads the process environment;
changing the application control requires a server restart. This is exact
(retention choice only), with no change to payload arithmetic or certificates. The
switch preserves automatic cache routing, numerical identity and restore validation. LRU is
unaffected. GPU performance qualification of this retention revision is pending.

Removing a directory entry never invalidates a shared handle or frees another
checkpoint's segments. Physical host gauges include registered payloads kept alive
by restore or writer handles. `counters()` reads scalar counters and incremental
gauges without walking payloads; the older diagnostic `stats()` reconciliation
continues to expose its existing directory/weak-segment accounting.

### Request no-store and counters

`"state_cache":{"retain":false}` permits validated cache hits but skips eligible
prefix captures before digest work, fences or allocation. It also suppresses disk
promotion and named frontier publication. A named continuation restores copies of
its envelope into the execution lane; the published parent is not overwritten.
Cancellation does not invalidate that parent for a no-store request. The cache
object remains present, so the request policy does not change scheduling inputs.

Absent `state_cache` uses server policy. Present `state_cache` must be an object
with boolean `retain`; invalid shapes and types receive 400 `invalid_request_error`.

Lifetime counters add lookup, hit and restored token totals; completed capture
attempts, capture failures, no-store and budget skips; and evictions for space,
conversation cap,
branch cap, explicit deletion and invalid entries. A suppressed eligible operation
counts once, not once per component. Hits count validated target prefix tokens;
predictor rows do not add token credit. No-store hits can update recency/reuse.
Reason counters have their own scope and do not sum to legacy `evicted`:
explicit deletion increments `evictions_explicit` but does not increment `evicted`.
The [serving snapshot](serving.md) exposes these counters without payload scans on
HTTP readers. Resident/pool gauges remain zero.

Opportunistic capture checks the store's incremental-byte estimate before pinned
allocation. A move-only reservation holds capacity against retained bytes plus
outstanding reservations until insertion or failure. Allocation converts reserved
credits into payload receipts without double charging. Disk reads reserve the
candidate payload before allocating; capacity refusal skips to a shorter candidate
or misses (`budget_skips`). Returned disk payloads remain charged until their last
handle drops, including no-store reads and refused promotion. A checkpoint that
cannot fit after eviction is skipped (`budget_skips`).
Only `PrefixCacheAllocationFailure` (including `std::bad_alloc` from the pinned
allocator) can skip capture, and only after `hipStreamSynchronize` returns
`hipSuccess`. After a successful drain, partial destinations are discarded and
`capture_failures` increments. If completion or a failure drain
returns any error, all possibly referenced checkpoint, fixed-buffer and pending
segment owners move into a process-lifetime quarantine before propagation. These
owners are never freed, even during shutdown; static count/byte totals are logged
on one line per quarantine transfer. The HTTP engine fail-stops, bounding this
retention to the failed operation. Restore source leases follow the same contract.
Every HIP error from the fence, copy
submission, completion or drain propagates, including `hipErrorOutOfMemory`.
Baseline explicit-cache capture preserves the original fatal allocation error
`prefix state cache pinned host allocation failed`, LRU and legacy counters, with
no new prefix notice and no additive `prefix_cache` JSON object, whether selected
by `--baseline` or `STRIX_BASELINE=1`. Other effective-config receipts retain the
additive object and non-baseline HTTP retains its single prefix status notice.
Explicit named-session checkpoint operations retain their existing failure contracts and LRU parking.

Policy selection requests (constrained JSON, thinking budgets, logprobs and logit
transforms) bypass prefix restore and capture, including the legacy cache and RTX
import paths. Their response carries `strix.state_cache_refusal` with
`policy-selection-cache-unqualified` when a store or import registry is present.
The full prompt runs from position zero through the existing policy selector and
admitted MTP verification route; cache certificates do not yet qualify that
composition. Named sessions continue to refuse policy requests before submission.

The canonical effective-execution signature is version 4. It binds resolved rotary mode,
factor, original and effective maximum context, and the complete rotary identity (including
mathematical descriptor and coefficient-table SHA-256). Prefix namespaces and producer
certificates use this signature; session envelope compatibility uses the same versioned
rotary frame. Cross-rotary prefix lookups miss and session loads refuse with
`session compatibility mismatch`. YaRN with session save/restore or prefix state caching
is currently refused by name; this binding does not qualify those combinations.

### Experimental batch certification

The requested automatic policy is `l0-where-eligible`; effective batch reuse remains
unqualified and declines before optional capture, restore or staging work. Ordinary
scheduler defaults stay unchanged. Batch routes, including an explicit one-slot batch
runner, are off unless explicitly requested under their existing diagnostic contract;
that access does not qualify automatic reuse. `--prefix-batch-cert off` (or `0`) and
`STRIX_PREFIX_BATCH_CERT=0` disable certification. CLI wins over the environment;
`--baseline` suppresses automatic admission.

Explicit `--prefix-batch-cert l0` retains strict diagnostic L0 capture/restore;
`capture` collects without restoring, and `l1` currently also captures only. These
explicit modes require the existing resolved RAM budget. L1 and automatic batch
qualification remain false; no batch family is promoted by single-runner qualification.

`cache_policy.batch_certification` in inspection, startup and runtime metadata reports
requested `policy`, `effective`, `reason`, `l1_qualified` and `automatic_qualified`.
Pending inspection is distinct from a runtime decline. Legacy diagnostic mode fields
may also appear for explicitly configured certification; they do not imply automatic
qualification.

Under explicit diagnostic certification, `snapshot_preparations` counts snapshot preparations after the
no-store check; it is monotonic, so a no-store request must not increase it.
`GET /v1/cache/stats` exposes it in `prefix`, alongside the existing `staging_bytes`
gauge for live snapshot staging. Prompt receipts with `STRIX_PREFIX_CACHE_DIGEST=1`
include both fields in `capture_counters_global`; end-of-run cache receipts also
include `snapshot_preparations` and `staging_bytes` under the certification opt-in.
Stats are published snapshots: staging may still be in flight when observed.
These diagnostic additions do not appear with certification off; the existing
stats staging gauge remains available independently.

The batch RAM namespace includes the route and prompt-group, SMALL, window/cohort,
service duration/lookahead, prompt share, short-first threshold, interleave, shared
families, cancellation, nesting and deferred-headless settings. Slot numbers and
arrival order are excluded, allowing reuse across slots. `QFULLT2` certificates
bind the canonical v4 effective-execution signature, scheduler namespace, route
and each unit's execution class. Capture requires an observed,
complete trace of solo full-T units on that slot's own T backend/owner, matching
the whole-prompt replay. A restored request cannot publish a derived batch
checkpoint in this round because its omitted producer units were not executed
in that request. Single-route certificates and segments never mix with batch.

A group owner created by `STRIX_PROMPT_BATCH_ROWS > T` declines even when
`STRIX_PROMPT_BATCH=0`; the receipt names `group-owner:prompt-batch-rows`.
SMALL, a separate group owner, state import/export and resumed
sessions decline certification. Schedule bit 2 (part of the default
`STRIX_QPF2K_SCHED=14`) is inert on this route, so the plan, its identity and
certification use the schedule-12 equivalent. The next writer must be the same T owner/stream:
a full-T unit, or a partial T unit with nesting ruled out. The nesting check
conservatively treats every active 8192-row nesting configuration as possible,
including profiles/ladders that might themselves decline nesting. No producer
wait is added. A foreign-stream writer observed before the target snapshot event
completes permanently poisons that snapshot; `capture_skips_tripwire` counts these
incidents. The qualification gate requires zero. Configuration declines increment
`capture_skips_batch_config`.

Restore is always a miss during service-mode admissions. L0 additionally refuses
any suffix that could share tail units or windows under the active configuration,
and refuses possible nesting. It installs the preflighted whole-prompt cold
cursor. For broader diagnostic coverage, explicitly set `STRIX_PROMPT_BATCH=0
STRIX_PROMPT_WINDOW_COHORT=0 STRIX_BATCH_NEST_ROWS=0` in both comparison arms.
The cache does not alter these scheduler settings itself. Batch equivalence and
native exactness remain unqualified; this opt-in is not a production contract.

With the default prompt grouping and window cohorts this guard refuses nearly every
stored boundary on the batch route (whenever the remaining suffix ends in a grouped
tail-owner unit or window), so the batch route usually recomputes the prompt; the
single tiled-MTP route does not use this guard.

A stored boundary rejected by this suffix guard reports `batch-suffix-not-solo`
in the request's `cache_refusal` and the cache's `certificate_declines` counters.
Service-mode admissions report `prefix-restore-disabled`. Missing/incompatible
certificates, producer-plan mismatch, failed metadata preflight and unsupported
continuations have separate named counters; a lookup without a matching candidate
reports `prefix-no-compatible-checkpoint`. A completed certified snapshot remains
valid after its producer is cancelled; cancellation of a later unit does not
invalidate the owned snapshot. Reuse still requires a certified consumer suffix.
`tests/cancelled_prefix_retry_gpu_gate.py` checks a 4096 checkpoint surviving a
mid-prefill cancellation against a fresh request's complete IDs and state. Its
default diagnostic arms explicitly disable grouping and window cohorts in both
processes; `--scheduler-defaults` instead checks the named suffix refusal.
Run `--arm offdef` and `--arm fresh` in separate turns, then use
`--compare RETRY/offdef.json FRESH/fresh.json` offline. Each arm has a shared
monotonic deadline (default and maximum `--budget 400` seconds, including a
20-second cleanup reserve); an exhausted budget fails and one arm alone stays
unqualified. Saved receipts bind the executable, supplied asset files, command,
prompt identity, client and relevant environment. Comparison revalidates the raw
request-local `cached_tokens` against the matched prompt's `restored_position`,
requiring explicit zero for fresh and refusal arms. It requires complete token
and logit-row coverage, terminal metadata, zero snapshot tripwires and all canonical
final fields except `request_order`. No predictor fields or `mtp.*` components are
excluded: the measured retry executes after the optional decoder retires.
Offline fail-closed controls: `python3 tests/cancelled_prefix_retry_gate_tests.py`.

With batch certification enabled and `STRIX_PREFIX_CACHE_DIGEST=1`, prompt receipts include an execution trace,
submission order, omitted full-T units, continuation cursor and configuration
refusal reason. Window deferred rows are flushed before the prompt state hash.
`request final:` reports `request_order`, `generated_ids_count`,
`output_logits_rows`, `output_logits_chain`, `committed_tokens`, `next_position`,
`predictor_frontier`, `pending_token`, `generations_lineage_digest` and
`state_digest`, plus prompt length and FNV identity. The logits chain uses
`H0=SHA256(empty)` and `Hn=SHA256(H(n-1) || SHA256(raw FP32 vocabulary row n))`
over every emitted output row. Missing coverage fails closed. The final hash
reads the actual predictor frontier without replay or normalization. Diagnostic
readbacks/fences affect timing; performance runs must disable them.

Automatic sizing protects cache-off context, maximum slot capacity and lazy growth
first, then fits the host budget and process-wide K=1 staging cap in residual capacity.
Batch per-request logits/state equality, peer equivalence, race/lifetime and
complete-workload gates remain required before automatic batch reuse is qualified.

Automatically admitted certified MTP capture prepares its payload slab and one reusable staging slot,
download stream and event set at post-residency attachment. Automatic startup
policy records include `pool_setup_ms`; failure declines with
`capture-pool-allocation-failed`. Explicit and legacy paths retain per-capture
allocation and retirement; an explicit host budget creates no startup pool.
Both paths share payload framing. The automatic synchronous split queues target
D2H before predictor priming; explicit/legacy capture retains combined submission
and per-capture allocation as described above.

The pinned slab and its bounded metadata allowance together consume exactly B.
`host_bytes` counts that commitment once, including free ranges; the logical
reservation ledger tracks admission credits without adding the slab again.
`pool_bytes`, `pool_metadata_reserved`, `pool_free_bytes` and
`pool_staging_bytes` expose the retained payload, metadata allowance, available
ranges and retained device staging cap. `pool_staging_bytes` remains charged
between captures; `staging_bytes` and pool `move_staging_bytes` report only live
descriptor staging and return to zero on retirement. Eviction returns payload
ranges to the slab. Physical admission accounts for aligned payload sizes and
contiguous free extents within the slab sub-budget. Eligible unleased checkpoints
are reclaimed in the existing retention order with bounded, nonwaiting work
before an extent-pressure refusal. Automatic capture and retirement allocate or
free no HIP memory, streams or events.
Payload leases survive both DMA completion and commit/cancellation handoff, and
quarantined resources cannot be reused. Busy or exhausted storage declines
immediately and increments `capture_skipped_no_slot`; no admission waits or
capacity reductions are introduced. Hashing, component framing, native producer
order and predictor priming order are unchanged; automatic target snapshot/download submission moves ahead of
priming under the split dependencies above. Pooling alone does not establish
the complete-workload cost or native exactness qualification gates.


### Certified restart and in-flight joining

`--prefix-cache-disk on|off` / `STRIX_PREFIX_CACHE_DISK=0|1|off|on` controls the
explicit disk tier. `--prefix-cache-durable on|off` /
`STRIX_PREFIX_CACHE_DURABLE=0|1|off|on` controls durable publication. Both default
on with an explicitly configured `--prefix-cache-dir`; neither creates a directory
or budget by default. CLI wins over environment. On POSIX, files are flushed and
synced before rename, segment directory entries are synced before checkpoint
publication, and the directory is synced after publishing the checkpoint. Other
host platforms sync file contents only. Baseline preserves legacy disk behavior.

The existing v1 checkpoint header's metadata-envelope tag distinguishes legacy
metadata (0) from certificate plus metadata (1). The section checksum covers both;
segments, digest chains, paths and store are unchanged. A certified disk candidate
must pass the same certificate, token, complete-state and cold-cursor preflight as
RAM before hit credit or device copies. Legacy and certified reads never cross.
An invalid candidate misses or falls back to a shorter checkpoint. Restart uses
the same binary, model, runtime, capacity, execution signature and scheduler
namespace. Named session envelopes are a separate continuation protocol.

Restore retains main's reservation-backed whole-load path and immutable DMA
leases. This change adds neither an aligned second format nor deferred/streamed
payloads. Snapshot and restore failures keep the existing drain/quarantine rules.

`--prefix-cache-inflight on|off` / `STRIX_PREFIX_CACHE_INFLIGHT=0|1|off|on`
defaults on wherever the prefix cache is on: it follows the cache's own enablement
and never widens the cache's qualified routes. It acts only where concurrent
producers and joiners share a strict L0 certified batch route; on the single
runner (including the automatic default route) it is inert. `--prefix-cache off`
and baseline turn it off. It is armed only with an active cache and interleaved ordinary
admission, solo prompt groups (`STRIX_PROMPT_BATCH=0`), lookup enabled, matching
full-T cold certificates/namespaces and the existing suffix-solo/nesting guards.
Short-first eligible consumers, policy selection, service-mode admissions,
no-store producers and producers restored from a cache do not supply shared work.
Consumer no-store still permits a validated hit. No production qualification or
automatic budget is enabled by this policy default.

A joining prompt waits before lookup and before executing any units. It requests
an existing full-T capture boundary of the producer's cold plan, without splitting
or replanning blocks. Any cache grid that divides T, including the HTTP 1024-row
checkpoint default, contains those boundaries. Independent waiters can request different boundaries. After
the producer passes the boundary, the consumer performs ordinary certified lookup
and installs its own preflighted whole-prompt continuation cursor. Only an observed
valid snapshot can become a hit; missed capture falls back to an ordinary lookup
or cold execution. Cancellation and completion release waiters; an admission
generation prevents reused producer slots from extending an old wait. Waiting
consumers do not enter prompt groups, window cohorts or nested execution.

Effective config's `prefix_cache` and HTTP capabilities report the armed `inflight`,
`certified_disk`, `durable` policies and `inflight_policy` guards, plus
`inflight_requested`, `inflight_source` (`default`, `cli`, `env`, `baseline` or
`prefix-cache-off`) and `inflight_reason` (`armed`, `off-switch`, `single-slot-route`,
`prefix-cache-inactive`, `batch-cert-not-strict-l0`, ...); these are not a
promise of a hit on every request. The batch shutdown receipt reports
`inflight_waits`, `inflight_wait_tokens` and `inflight_restored_tokens`.

The exactness contract preserves the cold T/tail decomposition, owner/kernel and
profile identities, priming and committed target/predictor generations/lineage.
It does not infer that tail rows equal full-T rows from the segment size. The
canonical effective-execution signature and QFULLT2 batch namespace remain the
admission authority. The new operational switches do not change that signature.

Source and host tests do not qualify native execution or timing. Run the
[serving gate](serving.md#prefix-joining-and-restart-qualification) before claiming
native equality, a complete-workload gain or broader automatic admission.
