# MiniMax conversation session cache

CPU codec, transaction, common API, defaults and phase-admission checks pass. The two-node serve route `serve/usb4-auto` is GPU-qualified, so `strix-minimax-serve` admits sessions by default; other apps and transports stay off until qualified.

## Defaults and switches

The defaults resolver admits automatic session serving only for routes in the single `gpu_qualified_session_routes` table in `src/minimax/engine/mm_defaults.cpp`, after slot and memory checks. The table lists `serve/usb4-auto/fixed-width-one-session-v1`, the default `strix-minimax-serve` transport. Any other eligible plain invocation prints `session cache: off (route not yet GPU-qualified: X)`, where X identifies the app, transport and fixed-width-one session policy. `--session-cache on` or `STRIX_SESSION_CACHE=1` explicitly opts into qualification runs while retaining slot and budget checks; it does not qualify the route. `--session-cache off`, `STRIX_SESSION_CACHE=0` and the baseline switches (`--baseline`, `STRIX_MM_BASELINE=1`, `STRIX_BASELINE=1`) turn sessions off and take precedence. Finished named sessions park in host RAM LRU, with at most eight entries. Requests without `session` retain the existing continuous batching and adaptive DSpark scheduler. Named requests reserve their queue position, wait for existing work to drain, hold later ordinary arrivals behind them, and use the committed single-row/fixed-width-one DSpark session policy. HTTP mutation requests enqueue commands; all cache/device work runs on the engine thread on both ranks. List and stats read the last published host metadata without waiting for an idle boundary. Metadata refreshes after each session operation.

RAM defaults to `min(8 GiB, 25% of MemAvailable)` at startup, measured after the model load. Windows CPU resolution uses available physical memory. The startup precondition includes the conservative capture peak; each capture/restore checks current availability again. Automatic startup failures disable serving sessions with one `session cache: off (<reason>)` line. Explicit ON startup failures log a reason and exit non-zero. Both ranks agree admission before constructing the runtime and agree construction success before serving. A refusal selects the same reason on both ranks. Measured budgets appear only in runtime admission logs. Later admission failures return structured budget errors. A named request whose history plus new turn does not fit the context returns HTTP 400 with error code `context_length_exceeded` (the session stays parked and usable); generation for a named request stops at the end of the context exactly like a `max_tokens` limit, with finish reason `length` (a stop token wins when it is the last token).

Automatic serving parks in host RAM only unless the disk tier is requested; the directory has no per-server ownership, so one server per directory. Disk defaults to `$XDG_CACHE_HOME/strix-engine/sessions`, otherwise `~/.cache/strix-engine/sessions`, capped at `min(64 GiB, 25% of available filesystem space)` at startup. An unwritable directory or space below two configured-context checkpoints disables only disk. All payload generations, including orphans, count toward the cap. Rank 0 selects disk LRU victims; both ranks remove the same session. Successful disk restores refresh recency. A persistent `.session.lock` file serializes quota admission, eviction, payload writing and manifest publication using Linux `flock` (Windows CPU builds use `LockFileEx`). The lock also protects deletion and disk restore. Acquisition is nonblocking: contention refuses collectively as `session_budget`, avoiding crossed locks across rank-local directories. The lock file must not be removed while processes use the directory. Replacement admission charges existing generations plus the incoming payload, including its temporary write, before publication; orphan generations remain charged.

| Option | Meaning |
| --- | --- |
| `--session-cache on` / `STRIX_SESSION_CACHE=1` | Explicit qualification-run opt-in; preserves admission checks |
| `--session-cache off` / `STRIX_SESSION_CACHE=0` | Disable session serving and ignore the request field |
| `--baseline` / `STRIX_MM_BASELINE=1` / `STRIX_BASELINE=1` | Disable session serving through baseline resolution |
| `--session-disk on` / `STRIX_SESSION_DISK=1` | Add the disk tier to automatic serving (off by default; explicit save/load always use their directory) |
| `--session-ram-bytes N` / `STRIX_SESSION_RAM_BYTES` | RAM parking and transient peak cap |
| `--session-max-parked N` / `STRIX_SESSION_MAX_PARKED` | Maximum RAM entries |
| `--session-dir DIR` / `STRIX_SESSION_DIR` | Rank-local disk directory |
| `--session-disk-bytes N` / `STRIX_SESSION_DISK_BYTES` | Disk payload cap |
| `--print-effective-config` | Requested mode, host-independent effective policy and reason; no host measurements or filesystem writes |

`--print-effective-config` appends only `session_requested`, `session_cache` and `session_reason` to the ordinary engine configuration. `session_cache=pending` means policy permits a request but resource admission must run at startup; it is not a live runtime. Inspection never resolves home paths, queries available memory or disk space, or probes writability.

CLI values override environment values. Baseline always disables session serving, even with `STRIX_SESSION_CACHE=1` or `--session-cache on`. Ordinary serving continues and reports `session cache: off (baseline)`. Explicit ON initialization failures without a baseline override exit nonzero. Legacy `--session-host-budget` and `--session-cache-dir` remain aliases. One-shot save/load remain explicit commands; a budget alone does not invoke them.

## Common HTTP API

These routes exist only while session serving is enabled. Disabled routes return 404 and disabled serving ignores `session`, including malformed values. IDs match `[A-Za-z0-9_-]{1,128}`.

| Route | Body / response |
| --- | --- |
| `POST /v1/sessions/{id}/checkpoint` | `{"durability":"ram"}` or `{"durability":"disk"}` |
| `POST /v1/sessions/{id}/park` | Park committed state |
| `POST /v1/sessions/{id}/resume` | Resume from a tier available on both ranks |
| `GET /v1/sessions` | `sessions` array with `id`, `position`, `consumed`, `pending`, `bytes`, `state`, `durability` |
| `GET /v1/cache/stats` | `ram_bytes`, `ram_budget`, `parked`, `disk_bytes`, `hits`, `misses`, `refusals` |
| `DELETE /v1/sessions/{id}` | Remove RAM and durable files on both ranks |

Chat/completions accepts `"session":"conversation-A"`. A full token history extending stored consumed history plus its emitted pending token resumes from that frontier; another history is a normal miss. Generation streams through the existing ticket callback, preserves EOS boundaries and parks the final committed state. Named requests decode without the DSpark drafter by default: its prefill taps disable grouped prefill (a 6.9K-token first turn took 22 s instead of 13 s) and fixed-width-one drafting gave no decode gain on that conversation; ids are identical either way. `STRIX_MM_SESSION_DRAFT=1` restores fixed-width-one speculation for named requests. A named request is answered first and its state is parked right after, before the next request is admitted. Resume parks its host copy again so ordinary slot reuse cannot destroy it.

Errors are `{"error":{"code":"...","message":"..."}}`: unknown ID is 404 `session_not_found`, incompatible checkpoint is 409 `session_incompatible`, corrupt payload is 422 `session_corrupt`, admission refusal is 507 `session_budget`. Session mutation requests wait at most one second in the queue, then return 503 `unavailable` with a `Retry-After: 1` header. Expired queued commands are removed. Once the engine takes an operation, the request waits for its completion. Mutations require an idle boundary; under continuous ordinary load they may keep returning 503 busy until that boundary is reached. The `/v1/session-cache` endpoint is unsupported.

## One-shot commands

Rank positional arguments remain `MODEL RANK TRANSPORT OUT_DIR`; serve uses `MODEL --rank R --transport T`. Both ranks need identical executable/assets/numerical settings/geometry and at least two slots.

| Flag | Meaning |
| --- | --- |
| `--session-id ID` | ID, default `session` |
| `--session-prompt P.u32` | Initial little-endian uint32 prompt |
| `--session-generate N` | First generated turn, default 8 |
| `--session-save DIR` | Save after first turn, then continue live |
| `--session-load DIR` | Restore instead of initial prefill |
| `--session-suffix S.u32` | Consume emitted pending token, then suffix IDs |
| `--session-suffix-generate N` | Continuation count, default 64 |
| `--session-output PATH` | IDs `.u32`, every FP32 row `.f32`, final `.sha256` |
| `--session-control-output PATH` | Never-saved prompt/turn/suffix control |
| `--session-dspark` | Exact ring snapshot and fixed-width-one verification |

Save/load are mutually exclusive. Explicit one-shot commands resolve eligibility separately from automatic serving: route qualification, baseline, and serving/disk off switches do not prohibit an explicitly requested save/load. Slot, RAM peak, physical memory, parked-count and disk budget checks still apply. Disk space and writability are checked against the actual `--session-save DIR` or `--session-load DIR`, independently of the automatic cache directory. Existing `--dspark DIR` / `--dspark-exl3 DIR` select drafter assets.

## State, compatibility and admission

All 60 layers preserve BF16 K/V (256 elements each) and index keys (128). Target state costs exactly **76,800 B/token/rank**: 75 MiB per 1,024 tokens and 8.789 GiB at 122,880 tokens. DSpark adds 40 MiB rings, 32 KiB positions and 32 B counters, totaling about **8.83 GiB/rank** at 120K before envelope/history/logits. The default 8 GiB cap cannot admit it and declines cleanly.

The envelope includes consumed/accepted/drafter positions, emitted count, pending token and emitted state, greedy sampling, producer lineage, full consumed history and pending FP32 logits. Vocabulary shards have 100,096 and 99,968 entries. All five physical DSpark rings, the 4,096-entry absolute position map and fixed-policy counters are included; invalid ring slots serialize as zero. Capture requires committed context with no deferred taps. Reset and successful import discard staged taps for that slot. General sampling/RNG and arbitrary adaptive speculation schedules are outside this schema.

Admission reserves six payloads plus existing parked bytes and 256 MiB physical headroom. RAM LRU eviction precedes capture allocation. Restore admits actual file size before reading and refuses growth. Both ranks select a common tier. Validation precedes import into an unpublished alternate slot; a collective import barrier publishes it. Refusal leaves the previous slot selected. Parking does not free preallocated GPU capacity.

The format is `strix-session-v1` / `minimax-conversation-v1`, alongside unchanged legacy hand-off blobs. Scalars/lengths are little endian, section offsets contiguous, and every section plus the full envelope has SHA-256 integrity. Compatibility binds full-file model/draft/executable/shared-library hashes, numerical environment, geometry, dtype/layout, producer policy, GPU architecture and runtime identity. Pack shards referenced by `weight_map` and every drafter `.safetensors` file are included using shared loader resolution, including hidden paths and referenced subdirectories. The pack inventory consists exactly of its index and the shards resolved by its weight map; this loader reads no separate config/control files. The draft inventory consists exactly of the root `.safetensors` paths opened by DraftFile. Unindexed files, README files, dotfiles and unrelated symlinks are ignored before validation. Each loader records the device, inode, size, mtime and ctime from every descriptor it opens, including separate
header and index reads. Live compatibility uses these snapshots; resolved paths are diagnostic only. Reference
paths (including aliases and symlink/parent paths) remain part of the inventory. Persistent digest hits must match
the captured identity; misses verify the hashing descriptor before and after reading. After all compatibility
components are built, retained loader descriptors are rechecked with fstat and closed descriptors through their
original paths. Changes refuse explicit ON and turn automatic caching off with `model files changed during startup`.
Storage/budget switches and diagnostic/turn metadata are operational. Keep disk directories outside asset
directories. The existing persistent full-file hash cache is reused; first lookup may scan assets.

Each rank atomically writes and fsyncs `ID.GENERATION.rR.session`. Both durable receipts must agree on generation, compatibility, frontier/history and replicated bytes before group manifests publish last. Missing ranks, corrupt sections and mixed groups refuse before upload. Orphan payloads are safe and count toward disk eviction. Delete can remove corrupt checkpoints without importing them.

## CPU checks and phased qualification

```bash
cmake -S tests/minimax_session_cpu -B build-session-focused
cmake --build build-session-focused --config Release -j 4
ctest --test-dir build-session-focused -C Release --output-on-failure --timeout 60
```

Coverage includes layout/opaque bits, full-model mutation, two-rank refusal, operational compatibility, corruption/truncation/enlargement/missing ranks/mixed generations, memory peaks, RAM count/byte LRU, disk cap/LRU, replacement peaks, competing directory handles, Linux interprocess locking, API routes/errors (including peer budget status 507), default qualification refusal and explicit CLI eligibility, observer overlays and phase prerequisites. Process-group helper tests explicitly skip with exit 77 without Linux `flock`/`setsid`. Ordinary HTTP integration tests require the project's POSIX HTTP implementation.

Invoke **one phase per separate invocation**, keeping the same absolute `OUT` outside Git:

```bash
bash tools/minimax/session_cache_gate.sh prepare
# Final staging must precede this invocation:
bash tools/minimax/session_cache_gate.sh prewarm
# A new invocation:
bash tools/minimax/session_cache_gate.sh save-off
# A new invocation:
bash tools/minimax/session_cache_gate.sh restore-off
# Separate invocations for save-on and restore-on, respectively.
```

Set `BIN`, `MODEL`, `DSPARK`, `CPU_TEST`, `OUT`, `NODE1` and an absolute persistent `STRIX_HASH_CACHE`; optionally `BUDGET`. No invocation launches another phase. All engine phases require `prewarm.done`. Prewarm also requires the four final staged observer binaries listed below. It runs the rank app's `--session-hash-only` mode on both nodes before any transport, engine, runtime GPU query or drafter construction. It hashes the exact target/draft inventories, executable mappings, every staged binary and their `ldd` libraries through `model_content_hash`, then repeats the same pass and refuses unless every lookup hits the persistent cache. Each node records wall time, second-pass time, hit count and zero misses in `rR/prewarm.receipt`; `.done` publishes only after both receipts and cleanup succeed. Restage in a new OUT and repeat prewarm if any bound input changes. Every later engine phase writes per-node `.timing.json` receipts with hash seconds, cache hits/misses for each compatibility lookup, and checkpoint I/O seconds plus operation counts (zero when no session operation runs). Later phases refuse any cache miss, malformed hashing receipt, or missing expected compatibility lookup before publishing `.done`. Save/restore logs and `.timing` receipts also retain hash timing and checkpoint payload/manifest write/fsync, read I/O, and read/verify/agreement timing separately. Hash-only `--session-hash-file PATH` additionally warms supplied binary/library inputs; it does not attest model fidelity. Save A and never-saved L share a load. Restore B requires a completed save receipt and both saved digest witnesses, and runs in a later invocation. P has 8,192 tokens, the initial turn eight, S four and continuation 64. Both ranks must match A/L/B IDs, every FP32 byte and final digests.

For ordinary-request identity, prepare disposable baseline and candidate trees using the same `session_default_probe.py` overlay. For source exports without Git metadata, pass the source revision as the overlay tool's optional second argument. Set `BASELINE_BIN`, `DEFAULT_BIN`, `BASELINE_SERVE`, `DEFAULT_SERVE`, `BASELINE_REVISION`, `DEFAULT_REVISION`. Use the public source before session caching as baseline. Run each phase in its own invocation:

- `old-rank`, `off-rank`, `explicit-on-rank`;
- `old-serve-off`, `off-serve-off`, `explicit-on-serve-off` (ordinary DSpark disabled);
- `old-serve-on`, `off-serve-on`, `explicit-on-serve-on` (ordinary DSpark enabled).

Run `auto-serve-off` and `auto-serve-on` with the candidate server and no session switch in the environment: they require automatic admission (`session cache: on (runtime live)`) on both ranks, and `compare` checks their unnamed requests against OFF exactly like the explicit arms. `off-serve-default` and `auto-serve-default` run the server as a plain invocation resolves it (DSpark mode, drafter discovery, slots, capacity and max rows from the defaults) with sessions off and automatic; `compare` requires the same equality there. The `explicit-on-*` phases force `STRIX_SESSION_CACHE=1`; serve arms require a live runtime receipt on both ranks. Each serve phase runs both the scheduler-off arm and the default adaptive scheduler arm, with separate receipts. Finally run `compare`. It requires all prior receipts and checks OFF versus baseline, unnamed explicit and automatic ON versus OFF, IDs, all evaluated FP32 rows, raw final target state, emitted callback bytes and bidirectional control/prompt witnesses. The adaptive scheduler arms size DSpark windows from measured cycle costs, so their evaluated rows, wire bytes and the tail of the last state export differ between processes; for those arms `compare` checks the emitted IDs and the committed state (the first prompt + emitted - 1 rows of the last export, whole rows on both ranks) instead, and the scheduler-off arms keep the complete evaluated-row witness. Separate framed transport witnesses record every framed transport message after pairing, through admission, including message sizes. OFF must match the baseline for these complete startup witnesses; explicit ON retains its admission exchanges.

The sequence is `prepare -> prewarm -> save-off -> restore-off -> save-on -> restore-on -> auto-serve-off -> auto-serve-on -> old-rank -> off-rank -> explicit-on-rank -> old-serve-off -> off-serve-off -> explicit-on-serve-off -> old-serve-on -> off-serve-on -> explicit-on-serve-on -> off-serve-default -> auto-serve-default -> compare`. Each arrow requires a separate invocation.

Each phase, including CPU-only prewarm, has an absolute **690-second** deadline including cold loads, transfers and verified cleanup. Work stops at 660 seconds, reserving cleanup time. The outer watchdog fires at 680 seconds with five seconds of kill grace. Expected phase time prints before work. Process registration/cleanup is isolated per phase; completion receipts publish only after successful cleanup on both nodes. Missing prerequisites, timeouts, signals, absent end markers and cleanup failure are FAIL, never qualification.

Native compilation and all paired GPU/default-identity witnesses remain unqualified. CPU success and defaults do not establish native exactness or performance.
