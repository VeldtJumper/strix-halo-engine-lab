# Committed QSA key service

The native MTP CLI, the single-request HTTP runner and explicitly opted-in HTTP routes can retain one
derived QSA key service per session. Twelve target entries serve every admitted R2/R4/R8 verifier and serial
fallback. A separate thirteenth entry serves the predictor, including batched
predictor dispatch. Each HTTP slot owns its own service.

`STRIX_QSA_MTP_POOL=0` restores the historical MTP key paths, admission bytes and
inspection image. Otherwise the service defaults on when the route and memory
admit it on the qualified CLI routes and the single-request HTTP runner in the table
below. HTTP batch slots, nested scheduling, sessions, disk prefix restore and state
import require `STRIX_QSA_MTP_POOL=1`.
Baseline, GSQ, defaults suppression, explicit FP32 pool storage, graphs,
hierarchical selection, wide-copy writers and explicit cohort requests retain
their existing paths. The explicit YaRN MTP CLI route can use the
retained table-backed service and rolling PLE by default for factor 2 through
524288 positions and factor 4 through 1048576 positions. Both `=0` off switches
remain authoritative. YaRN HTTP remains closed. The ordinary serial pool is a separate
policy. Failed service admission or allocation before prompt mutation retains the
historical native route and reports a decline reason. YaRN refuses that failure
before mutation; rerun with `STRIX_QSA_MTP_POOL=0` for historical consumers.

For capacity C and B = floor(C / 4), packed key payload is 13 * B * 128 * 2 bytes.
There is no persistent FP32 mirror or per-window service. One shared selector
scratch envelope reserves eight rows of max(ceil(B / 4), 16384) FP32 words. A
16 KiB host metadata envelope covers bounded entry, lease and event-handle state;
driver bookkeeping remains subject to the independent physical headroom reserve.
The reservation is charged once before window construction and per HTTP slot.

## Single-request HTTP runner

The service is on by default on the single-request HTTP runner (`--http-batch 0`,
or a configuration the batch runner cannot take). That runner serves one request at
a time from one service: every ordinary request starts a fresh epoch, and the
in-memory prefix-cache restore hook invalidates both domains before canonical state
is restored. Sessions, disk prefix restore (`--prefix-cache-dir`), RTX state import,
serving prefixes and YaRN HTTP keep the opt-in.

AUTO or capped `--http-batch N` sizes its slots without the service. If measured
startup capacity resolves to one slot, the server falls back to the single runner and
fits that slot again with the service charged. It keeps the service only if the slot
still fits; otherwise it runs the historical key paths. Startup prints
`qsa-mtp-pool single-fallback enabled=0|1 reserved_bytes=N (limit)`. An explicit
`--http-batch 1` keeps the batch runner. `--print-effective-config` reports the
`qsa_mtp_pool` object with `route` (`cli`, `http-single`, `http-batch`), `source`
(`default` or `STRIX_QSA_MTP_POOL=1`) and `effective`: `runtime-admission-required`
when the route admits the service, or `single-fallback-only` for AUTO/capped batch,
where only the one-slot fallback can take it. Each response's `qsa_mtp_pool_active`
reports the actual owner. Product profiles (`--profile default|fast|quality`) carry no
pool lever and do not change this decision.

Qualification, Strix Halo gfx1151, one binary run with the service default on and
with `STRIX_QSA_MTP_POOL=0`, servers restarted between arms. The configuration was dense-A8
(`hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1`) at T=8192 with R2/R4/R8 prefix windows, draft chain,
Q4 draft and context vocabulary, `--profile default`. For every request, token IDs, the prompt state digest, the
29 committed target state components and the full-vocabulary logits of every committed step were
identical. `STRIX_QSA_MTP_POOL_CHECK=1` compared all thirteen roles with zero score
differences. The requests covered:

- explicit `--http-batch 0`, with 10 requests: prompts of 2048, 2050 and 2052 tokens, constrained
  JSON, request n-gram drafts, early EOS, a temperature request on the greedy runner,
  and a client cancel followed by the next request;
- the automatically resolved single runner (row set 4,8), with 10 requests;
- prompts of 12K and 22K tokens;
- the in-memory prefix cache (`--prefix-cache-host-bytes`, T=2048), with 6 requests: checkpoint capture,
  three requests restored from position 6144 (an exact repeat and two shared-prefix requests), and an unrelated miss;
- the AUTO batch to single fallback. A separate process held 24-25 GiB of GTT so that measured capacity
  was one slot. Startup re-fitted the slot with the service charged
  (`single-fallback enabled=1 reserved_bytes=28442624` at 33,536 context) and served 2 requests
  identical to a pool-off one-slot fallback;
- `--profile fast`, with the same 10 requests as the explicit single runner. Every fast row is pending on
  this route, so it runs the default arithmetic.

`strix-hip-qsa-committed-pool-tests` passes natively and with `--yarn-factor 2` and `--yarn-factor 4`.

Timing used the same configuration at 131,072 context with a 6 GiB in-memory prefix cache and no
diagnostics. The prompt was 127,566 tokens of prose. A cold request was followed by two follow-up
turns restored from the cache, each generating 256 tokens greedily. The two arms ran ABBA, two
servers per arm, and produced identical tokens.

| Request | Decode, pool off | Decode, pool on | Change |
|---|---:|---:|---:|
| Cold (127.6K-token prefill, 176-183 s either way) | 39.5 tok/s | 52.0 tok/s | +32 % |
| Restored turn 2 | 36.5 tok/s | 47.7 tok/s | +31 % |
| Restored turn 3 | 46.7 tok/s | 63.9 tok/s | +37 % |

Prefill and restore time are unchanged. The gain comes from decode-side QSA scoring
over a long history. Shorter contexts were checked only for exactness, with diagnostics on, and carry no timing claim.

The adaptive window controller picks R2/R4/R8 from measured cycle times. A faster
cycle can therefore choose a different window, which changes the speculative rows
and the predictor KV but never a committed token or the target state. This happened once,
in an n-gram request. The existing HTTP gates exclude predictor state
for the same reason. Not covered by this evidence: the certified automatic T2048
prefix-cache family (its assets were not on the test node) and YaRN (closed on HTTP).

Canonical raw BF16 keys and KV remain authoritative. Each entry retains a
committed complete-block frontier and at most three provisional complete blocks.
A writer invalidates every block intersecting its first rewritten row before
publishing raw bytes. Accepted prefixes promote only complete, computed blocks.
Predictor proposals remain provisional until actual teacher/fast replay certifies
the cache rows. Restores, imports, replay, cancellation and prompt replacement
invalidate the appropriate entries. Derived keys are absent from snapshots. The prefix-cache restore/import hook
invalidates both domains before restoring canonical state. Fixed-session registry
restores and live boundaries start a fresh service epoch before state reuse;
ordinary request resets also start a fresh epoch. These boundaries drain prior
consumer events and preserve the admitted allocation.
Per-entry completion events order writers and readers across streams; leases
retain the service through queued consumers. Flushing deferred canonical state
retains certified keys.

Packing preserves the upper sixteen bits of the existing final BF16-rounded key;
integer widening restores exactly those FP32 bits before the existing products,
reduction tree and head accumulation. Selection and attention are unchanged.
`STRIX_QSA_MTP_POOL_CHECK=1` checks target and predictor score bytes against fresh
raw-key scoring, compares selected IDs and mutated scores, restores captured
outputs, and fails on any mismatch. Summary statistics report rebuilds,
invalidations, promotions, comparisons and differences.

`STRIX_MTP_LOGITS_DIGEST=1` captures actual physical window logits before slot
reuse and emits committed-step and terminal SHA-256 receipts. It also hashes the
actual final hidden ticket. Prompt/final state diagnostics require the component
registry (`STRIX_PROMPT_STATE_DIGEST=1`) and its reporting controls
(`STRIX_PREFIX_CACHE_DIGEST=1`, optionally `STRIX_FINAL_STATE_DIGEST=1`). These
checks synchronize and invalidate timings.

Host frontier/admission tests and device fixture sources accompany the service.
The default scope is limited to eager CLI rolling PLE, native MTP CLI, the
single-request HTTP runner, YaRN factor-2 MTP CLI through 524K, factor-4 MTP CLI
through 1M, and serial YaRN.
The 1M factor-4 prose gate matched all 256 tokens, every committed full-vocabulary
logit digest, prompt/final target state and predictor state across serial YaRN,
pooled MTP and pool-off MTP. The single-runner HTTP evidence is above; it does not
qualify batch slots, nested scheduling or session ownership. Host tests and builds
add no GPU qualification. Final-prefill seeding and selector algorithm changes are not
part of this service.

The shipped route policy is:

| Route | Rolling PLE | QSA MTP pool (`STRIX_QSA_MTP_POOL`) |
|---|---|---|
| Eligible eager tiled CLI | Default on | Native MTP CLI default on when admitted |
| Adaptive R2/R4/R8 + min-p | Leased CLI enabled | Enabled when admitted |
| HTTP single runner (incl. in-memory prefix cache) | Off (unsupported) | Default on; one-slot fallback only if the slot fits with it |
| HTTP batch slots, nested scheduling | Off (unsupported) | Opt-in: `STRIX_QSA_MTP_POOL=1`, per-slot owner |
| Session load/live/follow-up | Off | Opt-in: `STRIX_QSA_MTP_POOL=1`; fixed-session boundaries invalidate |
| Disk prefix restore, import, serving prefixes | Plain prompt-wide; MTP warm | Opt-in: `STRIX_QSA_MTP_POOL=1`; restore invalidates both domains |
| YaRN | Serial default on; MTP f2 through 524K and f4 through 1M default on | MTP f2 through 524K and f4 through 1M default on |
| Baseline / DEFAULTS=0 / GSQ | Suppressed | Suppressed |

Explicit opt-in never bypasses unsupported writers, rotary admission, capacity
or master off switches. A session save of the initial plain request can finish
its current rolling lease; subsequent loads/continuations use the warm route.
HTTP receipt counters for paid drafts and chosen window rows are emitted only
for an actual QSA MTP pool owner. With the pool off the historical receipt has no extra fields.

Automatic PLE setup failure, including unsupported direct I/O or a direct read
probe error, destroys provisional readers/leases and drops all rolling credit.
A fresh complete resident owner census checks total UMA, observed free memory,
workspace, metadata and GTT before warming the table or mutating prompt state.
If that envelope does not fit, execution refuses clearly before prompt mutation;
MTP is never silently disabled. Required reclaim refuses instead of substituting.
Post-admission read failures poison the source and exit nonzero with a clear
runtime error. There is no mid-run resident restart, even if tokens have not
been published: prompt/calibration may already have mutated target and predictor
state, and no pristine replay checkpoint or transactional restart owner exists.
Restart with a fresh invocation to obtain independent zero-credit admission.

The explicit YaRN CLI keeps the target's immutable table for all thirteen roles.
`STRIX_QSA_MTP_POOL_CHECK=1` reports per-role score comparison coverage with
`qsa-mtp-pool role=R checks=N scores=N differ=N`. Require all thirteen roles,
nonzero compared scores and zero differences; the aggregate alone cannot prove
that coverage. Common serial/MTP generation ledgers and target-state digests are
described in the [CLI contract](speculative/mtp-native-cli.md#explicit-yarn-mtp-cli).
These diagnostics and CPU tests do not extend the GPU qualification scope above.
