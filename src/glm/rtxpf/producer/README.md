# Native full-model producer (draft)

`glm_native_service` is a persistent Windows CUDA worker for one sm_120 PRO.
It is explicit and unqualified; it changes no node engine defaults or auto route.
Build offline with MSVC, CUDA 13, CMake and Ninja using
`build.ps1 -BuildDir BUILD_DIR` in a fresh build directory. The standalone
configure disables compiler launchers/ccache and builds at idle CPU priority
with at most two jobs. The build
compiles and disassembles only; SASS stays in RAM and only small evidence JSON
is written. Run CPU verification separately:

```powershell
ctest --test-dir BUILD_DIR --output-on-failure -j 1
BUILD_DIR/glm_native_verified_io.exe BUILD_DIR
python -B tests/glm_native/test_qualification_window.py
python -B tests/glm_native/test_producer.py --build BUILD_DIR
# Without any sockets/network:
python -B tests/glm_native/test_producer.py --cache-only --build BUILD_DIR
python -B tests/glm_native/test_shards.py --build BUILD_DIR
python -B tests/glm_native/test_shards.py --model MODEL_DIR --build BUILD_DIR
python -B tests/glm_native/receiver_loopback.py BUILD_DIR/glm_native_receiver.exe BUILD_DIR
```

## Model and cache

All 45 trunk layers, layer 45 MTP, embeddings, final norm, both HC sites,
shared/dense FFNs and every routed expert must be available. An incomplete
shard set fails before publishing any cache. The cache inventory validates
the full native tensor schema, including a separate `mtp.safetensors` omitted
from a trunk-only shard index, mul1 markers, rates, types, scales, finite
plain weights and the supported configuration. Prefill needs no vocabulary head;
decode-capable caches (`cache.py --with-head`) add the root-level `lm_head` (see
"Single-session decode" below).

```powershell
python -B src/glm/rtxpf/producer/cache.py --model MODEL_DIR --cache CACHE_DIR
python -B src/glm/rtxpf/producer/cache.py --verify --cache CACHE_DIR
```

This decoder already consumes the original compressed K16/N16 layout.
Persistent packing aligns original compressed tensor bytes to 256 bytes;
it does not decode or requantize them. The key binds the content-hashed
source manifest, layout, decoder, H128 transform and tile versions. Blob
digests are established before weight allocation. The worker holds read-only
Win32 handles denying writes and deletion through its entire lifetime. A cold
verification runs one worker per logical core, hashing independent whole files
in parallel with SHA-NI when available and two aligned 8 MiB reads in flight.
Whole-file SHA-256 values and the manifest format remain unchanged.

`%LOCALAPPDATA%/Strix/glm-verified-hashes-v2` persists computed shard digests and
successful blob verifications. Its protected directory DACL grants access only to the current
user and SYSTEM. A random 32-byte per-user key is created once, persisted using
Windows DPAPI (`CryptProtectData`, current-user scope), and authenticates every
record with HMAC-SHA256. The authenticated v3 canonical record binds format version,
volume serial, 128-bit file ID, size, last-write, creation and change times,
the file's last-change USN (`FSCTL_READ_FILE_USN_DATA` on the held handle), the
volume's USN journal ID (`FSCTL_QUERY_USN_JOURNAL`), and the computed digest.
Cache mode also checks that digest against the pinned expected value.
Byte writes invalidate records even after all basic timestamps are restored;
recreating the journal also invalidates records. Old timestamp-only records miss.
The accelerator requires a readable active NTFS journal. Non-NTFS volumes,
inactive/unavailable journals and failed queries always hash the complete file
and publish no record. Missing/wrong MACs, metadata/digest changes, and key
mismatches force a complete hash. Unusable keys are repaired under an exclusive
interprocess lock file, with a re-check under that lock; the repairing attempt
hashes fully and publishes no record. Any directory, ACL, key load, RNG, DPAPI,
key persistence or publication failure disables the accelerator for that attempt:
ignore records, hash fully, publish nothing. Record-write failure retains the
successful cold verification. Content mismatches and real read failures still reject.
The trust boundary includes NTFS journal semantics and Windows user isolation:
same-user code execution, privileged journal/volume manipulation, an
administrator, and offline filesystem/DPAPI compromise are out of scope.
Original shards are not
retiled or reread on a validated cache hit. Treat a cache directory as immutable and
model-specific; use another directory for a different model. Pin the complete
`manifest.json` SHA-256 in the launch command so tensor metadata is also bound.
Cache creation and validation are startup work, outside warm request timings.
The manifest is read once through its held deny-write/delete handle. Its pinned
SHA is checked on exactly those bytes, and the resulting parsed object is passed
to Runtime and the integrity tool without another pathname open. Prewarm or audit the same persistent cache without any
CUDA linkage:

```powershell
BUILD_DIR/glm_native_integrity.exe CACHE_DIR MANIFEST_SHA256
```

The CPU helper verifies the pinned manifest and all listed blobs; it does not
replace the runtime's full schema/geometry checks. Python `cache.py --verify`
remains an independent full-hash audit and does not use the metadata cache.
The helper reports `journal_unavailable_blobs` alongside `verified_hash_hits`;
zero hits on a volume without a journal means full verification, not failure.

### Direct shards (no disk cache)

Use `--model MODEL_DIR` instead of `--cache` and the cache manifest argument.
The worker opens the complete shard set read-only, holds Windows handles that
refuse writes/deletes, validates configuration/schema/markers, and computes or
recalls each source digest through the same authenticated authority as cache mode.
A matching record supplies its authenticated digest without reading the shard
payload. A miss hashes the complete held handle with two aligned 8 MiB reads in
flight, in parallel across files, then publishes the computed digest. Unavailable
journals or authority failures force full hashing without publishing records.
Configuration, shard headers, transform markers and every weight read consume
those retained handles, including resident loads and repeated file-tier read-ahead.
Independent positional readers use aligned scratch for unaligned tensor ranges;
large windows retain two unbuffered overlapped 8 MiB requests. Replacing a model
directory junction after inventory cannot redirect subsequent reads. The index
JSON is not consumed by this native reader; it inventories all shard files.
CPU regressions replace an ancestor junction after authenticated recall and
between file-tier requests, and compare synthetic packed blobs against the
previous pathname reader byte for byte, including zero padding and chunk tails.
The handles remain held until worker teardown. It derives the same content model
identity and cache key as the cache builder. No persistent packed or decoded files are created.

For device/pinned layers it packs each layer in host memory at load; file layers
are packed from the original shards by the one asynchronous read-ahead worker.
The virtual blobs use exactly the cache's lexicographic tensor order and 256-byte
zero padding. K16/N16 payloads are already native: no transpose, quantization or
floating point weight transformation occurs. An 8 MiB transient buffer feeds
bounded compressed layer rings, with event-protected reuse and H2D/GPU overlap.
Finite plain weights are checked while streaming. File-tier source integrity is
bound by startup hashes and lifetime read locks, rather than rehashing shards on
every request. The cache remains the faster option for contiguous reads without
per-tensor placement or source validation on every restart.

`test_shards.py --build BUILD_DIR` runs small synthetic CPU source sets, checking
cold bytes read, restart recall without payload reads, timestamp-restored writes,
authenticated-record failures, and unchanged model identity/cache keys against
independent full hashing. With explicit `--model MODEL_DIR`, it hashes production
C++ virtual streams and the cache builder's production packing stream independently,
without creating a disk cache. Its
scope includes the complete base blob, two routed KDA and two routed DSA layers,
and the unique MTP layer. Source schema, layout sizes, canonical model identity
and cache key are checked too. CPU byte identity establishes compressed placement;
it does not qualify GPU arithmetic or model fidelity.

Health reports startup source hashing time, resident load time/bytes, file-tier
bytes/request and a host read-cost estimate extrapolated from resident loads.
Request timings include actual file read/placement time and bytes, staging wait,
graph binding time and total runtime. These host costs can overlap GPU work;
cold storage and actual admitted tiers determine the observed cost.

## Execution and residency

The driver embeds prefix `[0,N-1)`, executes each entire layer before the next,
then uses normalized final trunk hidden rows paired with input tokens `[1,N)`
to populate MTP. HC normalization/collapse/expansion, norms, routing and nonlinear
parts use FP32 arithmetic. The node's HC chains, 20 Sinkhorn iterations and
q/k/v-major convolution serialization are retained. Native projections,
KDA32, DSA64 and CUTLASS grouped MoE preserve their existing declared schedules.
Each layer completes before the next layer begins. Within a layer, attention,
HC, routing, shared/routed FFNs and MTP run in exact row tiles. KDA convolution
and B32 recurrence carry across tiles; DSA retains absolute positions and
complete pools. A tail contains only real rows, with no padded recurrence or
routing. State release occurs after the layer's final row tile.

Capacity accepts **131072 inputs**, including the node suffix token. Only the
four residual streams and final hidden rows span the prefix. All other activation,
dispatch and MTP scratch uses `--row-tile` (numeric values fix a multiple of 32
no larger than 32768). Default `auto`: in the tierstream decode service the tile
must not eat the expert arena: the largest tile <= 8192 (8192 = one piece at 8K,
4 at 32K; min(capacity, 8192), then halved, each a valid schedule) that the
whole-device plan admits AND whose activation + workspace growth over tile 2048
is at most 5% of the expert arena planned at 2048. Per prefill row about 0.8-0.9
MB grows with the tile (14 fp32 rows of 4096 activations, about 229 KB; the
routed CUTLASS layout's 8 routes of operands and scratch, about 557 KB, also
inside the MTP workspace), plus 5 x capacity residual rows that do not depend
on it; at the full 32768 capacity tile 32768 would add about 27 GiB over 2048,
so the full card defaults to 4096 and 24/32 to 2048 (representative parts).
The plan reports `row_tile_choice`, `row_tiles_tried` (each candidate's cost,
growth and arena share) and `row_tile_components` (residual / tile activations
and each workspace at 2048 / 8192 / 32768) and `row_tile_trade` (expert arena,
residents, pinned and file units at tiles 2048 / chosen / 8192); health reports
`row_tile`. The harness `--row-tile N` sets the startup allocation (e.g. 8192:
the whole 8K prompt in one piece and the full-card 32K arm at 8192 vs 2048, at the
arena cost the plan reports); without it the service's 5% rule applies. The
prefill-only service keeps 256 below 32K capacity and 2048 at or above. The
`ttft` decode command takes an optional `row_tile` (at most the allocated one)
and records the prefill state digest untimed; the full-card harness compares
8K at tile 8192 vs 2048 and 32K at the allocated tile vs 2048
(`tile_compare`: BITWISE-EQUAL, or DIFFERENT, which makes the larger tile a
FAST-profile lever rather than the default).
Per-request tile (exact; profile row `prefill_tile_crossover`, default 0 =
always the allocated tile since the one-load tile sweep had the allocated
8192 fastest at 8K/16K/32K;
`--prefill-tile-crossover N` re-enables the rule): the service still allocates the
largest tile the plan admits, and every prefill picks its tile inside it:
prefill rows (prompt tokens - 1) below the crossover run at min(4096,
allocated), at and above it at the allocated tile (`schedule.hpp`
`select_prefill_tile`). Tiles only change the row chunking; the graph
templates are keyed by first row and rows, so alternating tiles recapture.
Every receipt records the chosen tile (`row_tile`, `row_tile_choice` in the
session prefill, `session_row_tile` / fresh `row_tile` in qualify), and
health reports `prefill_tile_crossover`. `--tile-requests 4096,8192` (full
card) measures the crossover in ONE service load: the exactness gate, then
TTFT at 8K/16K/32K (`q8192`, `ttft-16384`, `ttft-32000`) at every tile via
the per-request `row_tile` (capped at the allocation) plus the auto arm; the
summary's `tile_requests` holds the per-length bitwise compare and a
`crossover_estimate`. The per-start `--tile-sweep` below runs its service
with the crossover off, so each start measures its own tile.
The `--decode-commands` protocol is parsed by `decode_command.hpp` (fields
per op: `op`, `prompt`, `steps`; `order` for bench, `export_dir` for qualify,
`row_tile` for ttft); `--check-decode-commands` runs that parser alone (CPU,
no model, no CUDA), and the suite feeds it every command the harness's
`decode_command()` builds for every run mode. Each session prefill reports its
own expert cache traffic (`prefill_expert_cache`: file reads/bytes, misses,
copies); TTFT records carry `prefill_file_reads`/`prefill_file_bytes`, and
the first decode token now counts only its own traffic.
Decode residency restore (placement only; exact): deferred, it runs before
the next decode token (or untimed after a TTFT prefill) and only for the
planned units the prefill displaced (slot-table readback; the still-resident
ones are held first so the copies can only evict prefill leftovers). TTFT
records carry `decode_restore` (wall ms, displaced/copied units and bytes,
cache evictions/write-backs), `prefill_cache` (the prefill's own evictions,
write-backs, make_resident units/bytes/ns) and `make_resident_ms_per_layer`.
Under a file tier the arena spares are sized from the plan (`spare_policy`:
288 - residents per layer, so a prefill layer's make_resident fits without
evicting decode residents; the trade vs the percent share and the minimum is
listed). `--ttft-only` (full card) runs one load without the gate: TTFT 8K
and 32K at the allocated tile, 2 repeats each, compared bitwise
(`ttft_repeats`). Workaround on older builds: `--restore-decode-residency 0`
or `STRIX_GLM_RTX_RESTORE_RESIDENCY=0` (only `0`/`off` turn it off).
`--tile-sweep 2048,4096,8192` (full card) lets the data pick: one service start
per tile (`--row-tile T`; `startup_s` per start, later starts recall the
unit-file record), the exactness gate (q1024 + q8192 + the untimed q1024
compare) once on the first tile, then per tile TTFT q8192 + ttft-32000 and the
decode bench (cold/warm) on one chat kind at 1K/8K, all with state/logits
digests, each tile under `tile-T/`. The summary's `tile_sweep.comparison`
says per item whether the digests are bitwise equal across tiles (TTFT: the
complete prefill state signature, i.e. both ranks' SHA-256 and bytes, the KDA
replay checkpoint, the frontier and the component layout, plus the first token;
a TTFT record without it is NO-REFERENCE; bench: tokens, logits and the final
state signature per arm, validated in judge_bench against the frontier the
request implies, prompt + steps - 1; comparators only compare stored validated
signatures and report NO-REFERENCE instead of raising) and lists
TTFT and tok/s per tile. The printed estimate covers every start and stop
(1790 s for the three tiles); the 4096 arm is dropped first when it does not fit
`--timeout`, and the run is refused if it still does not.

Design note (not implemented): borrowing expert-arena slots for the prefill
workspace. The arena is a cache, so slots lent to prefill and refilled after
are exact. Only the tile-dependent growth over a permanent baseline would be
borrowed: the per-row prefill buffers that decode (one row) does not need: the
routed CUTLASS layout's per-route operands, activations, contributions and
scratch (about 557 KB per row), the MTP per-row buffers around it, the KDA
per-row raw/qkv/f/g buffers, and the tile activations (X2, F, mix, ids, rates).
The decode-sized workspace and the 5 x capacity residual rows stay permanent.
Slots: growth / unit stride (12.06 MiB): about 135 slots for tile 4096, 410 for
8192 and 2290 for 32768 at capacity 32768 (representative sizes; the plan's
`row_tile_trade` gives the exact bytes). The CUTLASS/MTP layouts need ONE
contiguous workspace, so the loan must be a contiguous run of slots (the arena
is one device array: slot k is base + k x stride), which needs a library call:
borrow n contiguous slots (evict their units through the boundary rules:
write-back into free pinned blocks, or drop file-homed units; the slots then
read as absent), and return them as free spares; decode misses refill them on
demand. Refill cost: at most n units of H2D after prefill (410 x 12.6 MB = 5.2
GB, about 0.2 s at 25 GB/s, spread over the first decode tokens), plus the same
volume of write-backs at borrow time unless the evicted units have clean copies.
With prefill through the cache, the borrowed run must be excluded from
make_resident victims (a keep without units) and the arena must still hold one
layer kept: slots >= 288 + spares + borrowed; the full card (about 6260 slots)
has room for any tile; the 24/32 arena (about 500-800 slots) does not at 8192.
Open questions: contiguous-run allocation under fragmentation (a reserved run at
one end of the arena avoids it), and the write-back burst at borrow time.

DSA's bounded 64-query score scratch still scales linearly with pool
capacity. At 128K/tile 256, activation accounting is under 11 GiB, before compressed
weights/rings, attention state and the shared native workspace. Admission still
uses actual allocation sizes; this is not evidence of successful device allocation.
The narrowed performance claim is fewer decode passes from larger row tiles,
not decoded-window reuse across a layer. Expert decoding still repeats per row
tile with two bounded decoded windows; at
32K the automatic tile reduces the number of complete expert decode passes
from 128 to 16. No cross-tile reuse or measured speedup is claimed. This changes scheduling only: KDA B32 carry, DSA absolute
positions, chronological K chains, H128/RN16 boundaries, route-slot order and
MMA family remain fixed. M-dependent hot/cold dispatch still needs the bitwise
execution gate. `--row-tile 256` retains the previous scheduling arm.
`admission_model.py` estimates activation/workspace and residual file layers
without CUDA; its physical and commit inputs are post-load observations.
Confirm estimates against health's actual allocation counters.

CUDA Graph templates capture the complete per-layer row sequence, including
CUTLASS decode-stream joins. At most two executable templates per layer are held:
the fixed full row tile and an exact tail. Changed positions, pointers, identities
or tail sizes are rebound through capture and `cudaGraphExecUpdate`, reinstantiating
if topology changes. No graphs accumulate per prefix position. Binding overhead
is reported, and `--no-graphs` provides the ordinary enqueue control arm. Graph
allocation is checked against the device cap. Graph execution, tiled numerical
identity and 128K admission remain GPU qualification gates.

The 20/19/14 device expert layers are upper targets for 8K/32K/128K
respectively. Pinned layers have no fixed count cap and are preferred to files
whenever physical RAM, Windows commit and pinned registration all permit.
Admission charges actual fixed allocations, existing device use, the 99.86 GB
cap, a 2 GiB device reserve, available physical RAM and available Windows commit.
Pinned admission leaves `--host-floor-gib` (default 8 GiB) available in both
physical memory and Windows commit; `--host-floor-gib 16` restores the old floor. A failed optional allocation demotes
to host or file immediately, with no admission retry or wait for resources.
There are two compressed layer rings, two decoded 16-expert buffers inside the
shared native workspace, one ordered H2D stream, event-protected reuse and at
most one file-layer read-ahead future. Fixed rings that cannot fit cause startup
rejection. File-tier requests use the already-verified held handles with
unbuffered overlapped reads straight into 64 KiB-aligned registered rings,
without reopening or hashing. Immutable pinned descriptors are copied once at
startup. Ready/retired events permit read N+2, H2D N+1 and compute N to overlap;
host reuse waits for the last H2D read, device reuse for the last compute read.
`--max-device-layers N` and `--max-pinned-layers N` cap the respective expert
tier counts (0..43, defaults 43). Device admission also retains the capacity's
existing upper target; memory guards can demote further. All remaining layers
use the file tier. Caps, actual counts and the file-ring parity mask appear in
health and self-check reports. `--require-file-tier` requires at least two file
layers covering both ring parities and positive read bytes through each ring on
every request. Caps that guarantee any file layers also require positive file
reads. Requests reread their file layers, including consecutive requests when
only the last two layers are files. A failed coverage gate prevents sealing.
For a deterministic qualification arm at capacity 32768 use
`--max-device-layers 19 --max-pinned-layers 22 --require-file-tier`.
`--serial-stage` is the synchronization/control arm. Request bytes/read time
remain measured, and health reports hash hits, workers, floor and IO shape.
`GET /health` records actual admitted tiers and free-memory snapshots;
target counts are never reported as measured allocations.

## Service and wire

Set `CUDA_VISIBLE_DEVICES` to one **full** PRO UUID, then launch only during an
authorized device window. Or pass `--dgpu` instead of `--allow-gpu`; it is the
same explicit GPU request and sets `CUDA_VISIBLE_DEVICES` itself before any
CUDA call:

- `--dgpu auto`: the sm_120 RTX PRO with the most memory, from NVML (no CUDA
  context on any card; ties go to the lower UUID).
- `--dgpu <uuid>`: that card, which must be an sm_120 RTX PRO (UUIDs from
  `nvidia-smi -L`; hex case does not matter).
- `--dgpu off`: refused (this service has no CPU or integrated-GPU path).

`--dgpu` is refused when NVML cannot report the card's compute capability, and
when `CUDA_VISIBLE_DEVICES` is also set to anything but the same full UUID
(empty, an index, a list or a MIG id included). Without `--dgpu`, a card that
NVML reports as not an sm_120 RTX PRO is refused before the runtime starts; when
NVML cannot tell, the runtime's own device check refuses such a card after it
starts (as before `--dgpu` existed).

```powershell
BUILD_DIR/glm_native_service.exe --model MODEL_DIR --key-file import.key --capacity 8192 --row-tile auto --bind 127.0.0.1 --port 8097 --allow-gpu
# Optional faster persistent cache, after its disk budget is approved:
BUILD_DIR/glm_native_service.exe --cache CACHE_DIR --cache-manifest-sha256 MANIFEST_SHA256 --key-file import.key --capacity 8192 --row-tile auto --bind 127.0.0.1 --port 8097 --allow-gpu
```

`GET /health` returns capacity, identity, draft qualification and residency.
`POST /v1/prefill` accepts this JSON shape:

```json
{"tokens":[1,2,3],"handoff":{"rank0":"127.0.0.1:9000","rank1":"127.0.0.1:9001","key":"64_HEX_DIGITS","node_identity":"r1b"},"identity":"glm-pf-rtx-native-v1"}
```

`key` is the 32 raw shared key bytes encoded as hex, compared against the
startup key file before opening either receiver connection. The service does
not log keys, bodies or endpoints. HTTP does not encrypt the request. Protect
the service and use an appropriate private deployment link. One request owns
the runtime; another prefill receives 503. Health remains available concurrently.
`STRIX_GLM_RTX_PREFILL=0` refuses startup. Unknown fields are refused.

The sender uses wire v1 begin/frame/commit/abort exactly: fresh receiver
challenge, HMAC-SHA256 with sequence, payload SHA-256 and ordered complete-frame
digests, raw bytes only. Per-layer state is released as soon as attention state
is final; bounded sender queues overlap the rest of the producer. KDA heads
are split 32 per rank; convolution selects each TP slice separately within
q/k/v. DSA histories/pools are replicated to both ranks. Pooled tails are omitted.

Both sender queues must drain before either commit is sent. On failure,
unsealed transactions abort or disconnect. ACK 1 is only a **local seal**:
the HTTP receipt is not evidence of the receivers' collective commit, apply
or continuation. Obtain both node receipts. The producer never supplies the
node's final prompt row, head, MTP seed or a qualification receipt. Node binary
hashes and node phase timings must come from the composed execution bundle.

The receiver accepts only the exact base native identity. Alternate indexer
IDs cannot be substituted in live handoff. For an offline diagnostic export,
use the same executable with `--export-state PRIVATE_OUTPUT --prompt PROMPT.u32
--identity INDEX_ID` instead of `--key-file`; all DSA index IDs from `dsa.cuh`
are selectable and also apply to MTP. These files/receipts retain that ID.

## Pending gates

CPU protocol tests establish byte transport into CPU staging, including both
ranks, malformed traffic and abort. They establish no CUDA execution or node
continuation. Compilation and per-object SASS checks cover native MMA/copy
loops, KDA blocked recurrence and DSA/MTP kernels. Complete-model allocation,
restart latency, pressure/demotion, deadline stalls, numerical/model fidelity,
node suffix, speculative oracle and endpoint qualification remain pending.
No on-disk cache is currently available. The `--cache` overlapped-reader and
H2D-overlap execution qualification is a separate pending gate. Direct-shard
file-tier runs and CPU byte tests do not establish that cache-reader gate.
The service contains no synthetic-state success path or CPU arithmetic fallback.

## Scheduling and FP32 index execution gate

The tiled FP32 index candidate uses 8-row/32-column projections with shared
operands and two complete independent K chains per thread. Scores use two
queries and 16 pools, with eight accumulators per thread, ascending K=0..127,
and the unchanged ascending 32-head fold. No split-K or new rounding point is
introduced. `--legacy-fp32-index` selects the original scalar index kernels.
All candidates retain draft status until native bitwise and workload gates pass.

`glm_dsa_bench --weights LAYER7.native --input CAPTURED_4097.f32 --rows 4097
--iters 1 --index-variant fp32 --self-check-exact --allow-gpu` runs OLD/NEW
kernels on the same captured normalized layer input. It compares raw ordinary
and trellis projections, valid score windows/ordered keys, full output/state,
selected IDs and cardinalities with host memcmp and differing-element counts.
4097 rows exercise both sides of the 512-pool cutoff, odd row tails and partial
pools. The captured FP32 input is 67,125,248 bytes. This mode requires real
captured input, and never substitutes another index identity.

`glm_native_service --self-check-schedule --prompt PROMPT.u32 ...` runs the
internal 256-row scalar-index/serial-stage arm and the candidate using the
same resident real weights and token IDs. It compares every exported state
component and the complete final hidden buffer using memcmp and byte counts;
per-rank SHA-256 covers payloads in release order, without wire headers.
Repeated `--prompt` options reuse one startup. Both arms include actual read
bytes/time and total prefill time. Mismatch exits 3 with counts. No state files
are written; baseline state is retained in bounded host memory for the check.

The main `qualification_window.py` mode compares HTTP handoff results to
private, independently captured `a6d126c4` receiver SEALED state digests. It
starts the candidate with `--model MODEL_DIR` (direct shards, no cache), capacity
32768 and default numerical/scheduling flags. Two local CPU test receivers use
node identity `r1`. Each resident service handles consecutive pairs of
1024/8192/32768-token requests. A second service repeats the same plan with
`--max-device-layers 19 --max-pinned-layers 22 --require-file-tier`. Every rank,
request and repeat must match the independent reference; every forced-file
request must read both rings. Missing evidence is failure, never a reference pass.

Keep reference results, provenance, prompt pairs and the private argument file
outside Git. The input file supplies `model`, `prompt_dir`, `reference_results`,
`reference_provenance`, `reference_results_sha256`, `reference_provenance_sha256`,
`reference_service_sha256` and `reference_commit` (`a6d126c4`). Pin the exact
private results/provenance files and the reference executable identified there;
that reference used cache input, capacity 32768 and row tile 256. The launcher
checks prompt u32 bytes against JSON IDs and provenance hashes, and binds both
candidate binaries, the reference model/manifest identity and expected complete
per-rank hashes in its private evidence. Reference provenance is an operator
attestation to the independent executable/run; this launcher does not rebuild
or execute that historical binary.

```powershell
# Plan only: no process, sockets, CUDA imports or device discovery.
python -B src/glm/rtxpf/producer/qualification_window.py --build BUILD_DIR --inputs PRIVATE_INPUTS.json
# CPU-only validation of private pins, all prompt pairs and candidate binaries.
python -B src/glm/rtxpf/producer/qualification_window.py --build BUILD_DIR --inputs PRIVATE_INPUTS.json --check-inputs
```

Add `--allow-gpu` only in an authorized PRO window, with one full PRO UUID in
`CUDA_VISIBLE_DEVICES`. Expected wall budget is **90--150 minutes** for two cold
source-hash startups and twelve HTTP requests; this is an estimate, not a measured
candidate latency. The hard aggregate deadline defaults to **180 minutes**,
can be shortened with `--timeout SECONDS`, and cannot exceed 10800 seconds.
An independent supervisor process enforces the absolute deadline, including input
validation, executable copying/hashing, startup and execution. Before any worker
code runs it is assigned, suspended, to a Windows job with `KILL_ON_JOB_CLOSE`;
all descendants inherit that job without breakaway. At the deadline the supervisor
terminates the entire job even if the worker blocks in validation or HTTP IO.
Cleanup waits at most fifteen additional seconds in total; no process-name kills
are used. Timeout is INCOMPLETE. Each run copies service and receiver executables
into a unique private run directory, hashes the copies and holds read handles
denying writes and deletion (and a directory handle denying replacement) until
all requests complete. Only those copies are launched. Every HTTP receipt must
name exactly the pinned service digest in `producer_executable_sha256`; rebuilding
the original candidate cannot change the running qualification identity. Logs,
receipts and results stay in a new immutable directory under the build directory;
no state payload files are written and the temporary handoff key is removed.

`--diagnostic-self-check` adds the internal OLD/NEW check after the independent
HTTP passes, under the same aggregate deadline, with both file rings required.
It needs a third startup and additional compute and may exceed the budget.
Internal agreement is diagnostic only: it shares candidate code. The independent
SEALED references cover exported state; independent final-hidden-buffer evidence,
other capacities (including the normal 8192 default), targeted hot/cold dispatch,
interleaved timing medians, model fidelity and complete node continuation remain
separate pending gates. No performance or production qualification follows from
CPU checks or from a reference plan that has not run.
The harness prints the separate pending `--cache` overlapped-reader/H2D-overlap
gate in its plan, input-validation output and run verdict; neither direct-shard
phase qualifies it, and no cache is created by this harness.

## Single-session decode (draft identity `glm-dec-rtx-native-v1`)

Explicit and unqualified; the prefill producer identity, wire, health and
behaviour are unchanged when decode is not requested. Decode is enabled by
`--decode N --prompt PROMPT.u32` (one generation, receipt on stdout),
`--enable-generate` (HTTP `POST /v1/generate`, needs `--key-file`) or
`--decode-commands` (qualification harness protocol over stdin/stdout).
`--expert-tier reference|tierstream` selects the routed-expert backend.

**Session and token convention.** One pending token: for sequence S of length N
the trunk state holds rows `S[0..N-2]`, MTP holds pairs `(hidden[j],S[j+1])`,
`S[N-1]` is pending and both frontiers are `P=N-1` (the prefill export layout).
A step consumes the pending token at absolute position P through layers 0-44,
collapses HC, applies the final norm and head, chooses `y`, advances layer 45
with `(hidden[P],y)` and appends `y`. After t steps the session equals a fresh
prefill of `prompt+generated[:t]`. Layer 45 is maintained for qualification and
never chooses the token. `prompt_length + max_new_tokens <= capacity`, so a
fresh prefill of the final sequence is always admissible. Resident state per
capacity C: 34 KDA recurrence/convolution sets (152,633,344 B), 12 DSA
latent/key/gate/pool sets (`1664*C` B each), plus 170,459,136 B of KDA replay
checkpoints. Session prefill writes this state directly (no release/export);
the legacy prefill path keeps its shared release buffers and launch sequence.

**KDA exactness.** Production KDA is B32 block arithmetic with FP16 block
operands; one-row calls would change rounding. Each KDA layer keeps the state
and convolution history at the start of the open 32-row block plus its
normalized input rows. A step restores the checkpoint and replays `P%32+1` rows
through the unchanged production `enqueue_kda`, using only the newest output
row; the 32nd row promotes the block. `enqueue_kda` gained an optional
`KdaCheckpoint` (a state copy before the block and a convolution gather): reads
only, prefill arithmetic unchanged. Prefill blocks are absolute because row
tiles are multiples of 32.

**DSA / MoE / MTP.** DSA uses `enqueue_dsa(rows=1, first=P)` on the session
state (FP32 index identity). Dense layers 0-2 and shared experts use the same
CUTLASS calls with one row. `enqueue_mtp` has `phase` 1 (through routing, IDs
copied out) and 2 (FFN/finish on the same workspace) so only the selected
layer-45 experts are acquired; phase 0 is the unchanged prefill call.

**Vocabulary head.** Root-level `lm_head.{mul1,trellis,suh,svh}` (K6, I16
`[256,9680,96]`, marker 0x83dcd12d, 476,109,316 payload bytes) live in their own
`head.bin` blob so `base.bin` and every prefill tensor offset are unchanged. The
cache key gains `"head":"lm_head-root-k6-v1"`: prefill caches keep their key,
decode rejects a headless cache, and a decode-capable cache still serves
prefill. The head is decoded once at startup to FP16 `[4096,154880]`
(1,268,776,960 B; the compressed blob is freed), then projected with the H128
input/output transforms and RN16 boundaries and one ascending scalar FP32
`__fmaf_rn` chain per logit (the projection reference semantics, shared by
both qualification arms). Argmax orders logit descending then token ascending,
`+0 == -0`, and rejects the token if any logit is nonfinite.

**Routed experts.** Expert addresses come from an immutable per-(layer,expert)
extent catalog (`expert_source.hpp`), never `id*average_bytes`: packing is
lexicographic (`experts.1.` < `experts.10.` < `experts.100.` < `experts.2.`).
The product path is the shared tierstream expert cache (`expert_tier.hpp`):
GLM expert *units* are one opaque 64 KiB-strided unit per (layer, expert) in
numeric order carrying gate/up/down trellis, suh, svh and mul1 original bytes
(256-byte aligned, canonical order, zero padding), with a unit manifest binding
model identity, geometry, rates and per-unit SHA-256, and a one-part
`tierstream/desc-v1` descriptor accepted by the real tierstream parser: a
`sources` array, layers listed as gated ordinals 0..42 (trunk 3-44 -> 0-41,
MTP -> 42; model layer in `x-glm-layer`), format `x-glm-exl3-unit-v1`,
`offset + unit*stride`, stride == bytes. `glm_native_decode_host` compiles
tierstream `src/desc.cpp`/`src/json.cpp` read-only from `GLM_TIERSTREAM_SRC`
(a private `git archive` snapshot `../tierstream-f3fa6cf` of tierstream `f3fa6cf`,
its STAMP checked against the pin: configuration fails on another commit) and parses the generated descriptor; it reports SKIP
when that worktree is absent. The adapter
follows the DS tier site: `begin_token`, `acquire` (export IDs + wait),
`bind_selected_descriptors` (device kernel: guarded slot -> descriptors from
the unit offset table; abort/out-of-range -> guard, absent -> `absent_err`),
`release` (end after layer 45), `end_token`, `collect_stats`, `reset_cache`.
`--expert-tier tierstream --unit-file PRIMARY` is the product backend: a
concrete `TierCacheApi` over tierstream's device-gated C ABI (`tierstream_binding.cpp`,
tierstream built from the read-only `GLM_TIERSTREAM_SRC` worktree inside each build
directory: CUDA archive for the service, CPU archive for tests, linked against
this project's `cudart_static`). It opens after prefill residency: the VRAM
arena is what remains after base weights, admitted device expert layers,
workspaces, head and session (4 GiB reserve, `--tier-vram-gib` cap); the pinned
RAM tier comes from free host memory above `--host-floor-gib` (`--tier-ram-gib`
cap); every other unit is homed in the unit file (`dg_disk_tier`, read on demand
into `--tier-staging` pinned buffers; victims VRAM -> RAM -> drop). Placement
follows the DeepSeek tooling (`cuda_ds41_replica` `seed_order` / residency
file): one priority list of keys (gated ordinal x 288 + expert) gives both the
tier array (its first `resident` keys on the device, then pinned, then file) and
`ss_options.dg_seed` (which units take the device slots; the initial recency
order). The list is the residency file of the previous run (`ss_dg_residency`,
most recent first, written after every request (HTTP prefill, generation,
decode commands) to `<unit-file>.residency` in the
DS layout with the `glm-residency-v1` tag; `--expert-residency PATH|none`), then
experts by routing count from DS-format routes census files (`--expert-routes`;
`--routes-out FILE` records every decoded token's routes, the receipts carry them
as `routes`, and `glm_route_census.py RECEIPTS -o FILE` collects them), then the
no-profile fallback: expert-major round-robin across all 43 routed layers (every
layer holds the same share; the full card held layers 3..24 entirely and nothing
of 25..45 in key order). Boundary victims use tierstream's scored policy as DS
does (`--tier-policy auto` = SCORED_AUTO, online learning, no profile file;
`lru`, `scored`). Placement changes timing only, never outputs. Per token: `ts_begin`,
per routed layer export + wait + guarded-slot descriptor bind, `enqueue_end`
after layer 45, `ts_dg_end_rows(1)`; absent units or device timeouts fail the
token. The planner bills no per-layer dynamic slots: the arena's spares are one
global count (`dg_spare_slots`) already in our device budget. Misses are
retained only up to the spare count per token (a demand without a free slot is
served from the bank and not kept; the full card had 16 spares, evicted 16 per
token and fetched 110-171), so the default is 5% of the arena
(`--tier-spare-percent`, a second planner call with that count: residents give
way, the arena bytes stay) and at least `--tier-spare-slots` (16; an explicit
`--tier-spare-slots` alone is an exact count). For reference, the DeepSeek service runs
`--spares 96 --host-spares 48` (40 x 384 experts); GLM's 5% is 314 spares on the
full card, and `--tier-spare-slots 96 --tier-host-spares 48` reproduces DS exactly. Optional overlapped next-layer
prefetch (`--tier-predict 1`, off by default): after layer l's export and wait,
layer l+1's router runs on its own copy of layer l's MoE input (copied on the
main stream before the fork; the next layer overwrites the original) on a side
stream into its own ids/rates/flag buffers and `ss_dg_enqueue_predict(l)` hands those 8 ids to the
library; every export of layers 3..43 promises 8 predicted ids; the branch joins
before layer l+1's export. Layer 44 does not predict the MTP layer. The token's
math never reads the predicted buffers and the library only prefetches; the
untimed q1024 compare with `--tier-predict 1` is the exactness check; the real CPU library admits full 43 x 288 opens at production
stride (e.g. 3,353 residents) in the CPU tests, which also show the former
per-layer billing being refused. Startup authenticates the unit file before any
use: it first takes the deny-write/delete handle and checks the expected size on
it, reads `PRIMARY.units.json` once through its own held handle against the
operator-pinned `--unit-manifest-sha256` (printed by `glm_unit_file copy`; the
manifest binds model identity, geometry, rates and the whole-file SHA-256), then
authenticates the contents through the held handle: a matching authenticated
verification record (the HMAC/DPAPI/USN authority of `verified_io.hpp`, keyed by
file identity and digest; `copy` publishes it) is recalled, otherwise the whole
file is hashed. Without a USN journal on the unit file's volume (E: has none)
the unit file uses an mtime-stamped record instead: the engine's Windows file-hash stamp v3 (volume
serial, file index, size, last-write and creation times; ChangeTime excluded as in
the engine) read on the held deny-write handle, plus the manifest pin, under the same DPAPI-keyed MAC; recall needs all of them equal and
the file held deny-write from open. Health states the weaker guarantee ("no USN:
mtime-stamped record"): a same-user rewrite that restores every timestamp
between runs is not detected. Other files keep USN-only records. Only then is the library's per-fetch hashing off; the handle
stays held. The library opens its source through held
ancestor handles (bound open) and requires the authenticated identity
(`ss_options.source_identities`: volume serial, file index, size of our held
handle; any other file fails the open). The name it receives is also bound to
the authenticated object (defence in depth): the cache receives the held handle's final
path (`GetFinalPathNameByHandleW`; local drives only, no aliases or junctions),
every ancestor directory is held with `FILE_LIST_DIRECTORY` sharing only reads
(no write, rename, delete or reparse installation while held; reparse ancestors
are refused), a reparse-refusing name handle on the file must carry the
authenticated volume serial and FileId, and after the cache opens a serving
handle through the bound path is compared to that FileId. The bound path is
passed to the library as lossless UTF-8 of the native (backslash) form, which
must decode back to the identical wide path (tierstream decodes source paths as
UTF-8; an ANSI string would name a different file for non-ASCII names).
Alternate data streams are refused, both on the operator path and on the held
final path. The binding and the cache are owned together (`BoundCache`: binding
declared first, cache second), so the cache always closes before the binding is
released, also when the backend constructor unwinds after the cache was opened;
health reports the binding (`unit_file_binding`).
The CPU tests swap the operator's junction between authentication and the open:
the alias then names same-size impostor bytes, while the bound path still serves
the authenticated object and every ancestor rename or file replacement fails
until the binding is released. A non-ASCII regression binds `cafÃ©\units.bin` next
to a same-size impostor `café\units.bin`; tierstream's own public reader
(`ss_mfile_open` on the parsed descriptor's source path) reads the authenticated
bytes, while the cp1252 ANSI form of the same name opens the impostor. A lifetime
test injects a failure after cache install and checks that the bound ancestors are
still held while the cache closes.

**Unit-file identity.** The former residual (the library's own handles were
not identity-checked in earlier tierstream versions) is closed by the library's bound open with
the expected source identity; our held ancestor/name binding stays as defence
in depth. Health: `unit_file_expected_identity`, `tierstream_commit`.
**Threat boundary:**
same-user accidental or hostile renames, junction swaps and replacements on local
NTFS are in scope; DOS device remapping (`DefineDosDevice`, per-session drive
letters) and privileged attackers (administrator, backup/restore privilege, kernel,
raw volume writes) are out of scope.

**Prefill through the cache (one copy of every expert).** Tierstream mode
keeps NO prefill resident/pinned expert copies. With the tierstream
reader/resident API (`ss_dg_make_resident` between
tokens, kept units resolving through `ss_dg_slot` in any step, single-layer steps
via `rows_per_layer`), each routed layer runs: `make_resident(g, 0..287)` ->
one step whose only gated layer is g -> `bind_units` writes the 288 descriptors
from the slot table into the layer's descriptor array -> the UNCHANGED CUTLASS
prefill over every row chunk (the graphs read that array at launch) ->
`enqueue_end` / end -> `ss_dg_keep(g, NULL, 0)`. Prefill then reads the very unit
bytes decode reads, from VRAM slots (pinned and file units copied in by the
cache); no prefill rings are allocated. `SS_E_CAPACITY` (an arena below
288 + spares + guard) is refused: expert batches with masked dispatch are not
implemented. A CPU mock of the contract (decode_host) checks the protocol and
that every slot-bound descriptor addresses bytes identical to the source
tensors. Active with the pinned tierstream (f3fa6cf)
(`SS_DG_MAKE_RESIDENT`; first GPU run is GLM's: serial pinned -> device
copy-ins; transfers are transactional: a read failure leaves
every keep and slot consistent, a device failure poisons the context, which
retires the worker here). Each kept set is released right after its layer's
step, so decode tokens never meet a held (unexported) layer, and no call ever
replaces a live keep: the library's room rule for a disjoint replacement (old +
new + spares) never applies; each call needs 288 + the spare target, which the
startup refusal already checks. Receipts add `prefill_resident_*_total`. ANY unsuccessful residency call,
step or release (a capacity refusal included) retires the worker irreversibly:
later requests report the original error. The keepable requirement is the
library's (n + kept + spare target <= persistent slots, ring/bank slots and the
guard never kept): slots >= 288 + spares; the planner gets
min_resident = 288 and adds the spares itself, so dry plans and startup refuse before
authentication or any load. CUDA graph capture, update, replacement and
instantiation are measured (samples while the captured graph lives, the cap
enforced at each) and their net persistent memory is charged; every generation
receipt (CLI, HTTP) carries the final memory report and fails when a peak
exceeds a cap (exit 5 / HTTP 507). Schedules print their estimate with a 300 s
startup prior (full card 1600 s, whole-card 24/32 1760 s); receipts record the
measured `startup_s`. Precondition failures (residency inside an open
token) and any exception while a layer is held or its step is open retire the
worker with the original error. Graph results are sampled before any cleanup
(capture, update, instantiate; success or failure); a graph executable whose
release fails keeps its charge and retires the process. A generation that throws
still returns a FAILED receipt with the original error and the measured peaks,
plus `expert_cache`: the library's last token error (which flag of which gated
layer and token), `device_timeouts`, stalls, misses, free slots and `flush_ready`.
In `--decode-commands` mode the failing command writes that receipt to stderr as
one `FAILED-RECEIPT {...}` line before the worker exits.

**Steps, predictions and the decode timeline.** The binding keeps the library's
step rule (`StepPlan`, equal to tierstream's `DgSteps` over GLM's whole call order
in `glm_native_decode_host`): a step exports a gated layer when its rows name it
and it holds no prefill keep set. An export promises predicted ids, and
`--tier-predict 1` enqueues them, only when the step also exports the next layer,
so no prediction ever targets a layer the step skips or holds; an export of a
skipped or held layer, a mismatched prediction, and an all-layer decode token
while any layer still holds its keep set all fail loudly naming the layer.
`--decode-profile` (service and harness) records one CUDA event per enqueued
decode phase (embed, hyper-connection mixes, KDA with its open-block replay, DSA
indexer + sparse attention, router, cache export + device wait + bind, predicted
fork, routed MoE, shared expert, combine, head + argmax, the host gap at the
result synchronization, MTP) and puts per-layer and per-phase device/host
milliseconds into every decode step record (`timeline`) and per bench arm
(`timeline_totals`). Events only: tokens and digests are unchanged; timings
include the events. A failed token keeps its partial timeline in the failure
receipt (`expert_cache.decode_timeline_at_failure`), whose longest interval names
the layer and phase where the device stalled.

**Prefill workspace.** The CUTLASS (C1) MoE layout reserves no activation buffer
(its transform never reads one: the down input is `c1_down_transform` over `ad`; only
the legacy path keeps it), and the MTP layer runs its prefill in pieces of at most
2048 rows whatever the trunk tile (`mtp_row_tile`; per-row router, GEMM K order and
absolute DSA positions do not depend on the piece; the bitwise-equal
prefill state across tiles includes the MTP layer). The workspace is the maximum over operators
with MTP at its own tile: 3.444 / 3.624 / 5.637 GB at tiles 2048 / 4096 / 8192
(was 3.579 / 5.057 / 8.015 GB); the plan's `row_tile_components` list each part.
The prefill's shared/dense calls also use the persistent FP16 operands when
`persistent_dense_fp16` is admitted.

**Eager module loading.** The service sets `CUDA_MODULE_LOADING=EAGER` at the top
of `main`, before any CUDA call, unless the variable is set explicitly (the harness
also defaults it and records it in the receipt): with lazy loading, the first launch
of a decode kernel while the cache's wait kernel spins can stall until the wait
deadline. Health records the value and the driver's actual loading mode, and the
tierstream decode service refuses a lazy driver. `--print-effective-config` reports
the value (it runs no CUDA call).

**Slot census.** Health `tier_plan.slot_census_at_open`, every `--decode-profile`
step record (`slot_census`) and every failure receipt
(`expert_cache.slot_census_at_failure`) carry the library slot table's homes per
gated layer (units on a persistent slot - `guard_slot - 2 x max_ids` residents and
spares - on a bank alias slot, absent), the free persistent slots read from the
table (live, unlike `ts_dg_stats.free_slots`, which is folded at the last
boundary), and with `--decode-profile` each exported layer's decode misses
(its ids absent from the table when it exported; tierstream counts misses only at
`ss_end`). Read-only kernels; outputs unchanged.

**Few-row decode kernels (`decode_row_kernels`, on in every profile).** Decode runs one
row (two to four for routed and shared MoE, up to 32 for the KDA replay rows), so the
grouped CUTLASS GEMMs and the per-token FP16 decode of whole weight matrices are
replaced, for those row counts, by `row_mma.cuh`: one warp per 16-column strip decodes
each lane's eight weights straight from the packed tile (the tile is stored in
m16n8k16 B-fragment order) and runs one m16n8k16 chain per output in ascending K from
+0, with no split-K. That is how the pinned CUTLASS kernels (instruction 16x8x16, K
tiles of 32, `OpMultiplyAdd`, beta 0) accumulate each output, so the FP32 results are
the same bits. Sites: routed, shared and dense MoE (`Call::row_gemv`), the KDA qkv and
out projections (`KdaCall::row_kernels`), the DSA kva, qb and out projections
(`DsaCall::row_kernels`; qa and iq stay decoded for the FP32 index) and the MTP layer
(`MtpCall::row_proj`, `row_moe`). Two more kernels keep their arithmetic and change
only where operands come from: `hc_pre_staged` feeds the 24 serial mix chains from
shared memory, and `router_scores` + `router_select` run the router chains on 9 CTAs
and pick the same top 8 by rank (key descending, index ascending), then sum them in
index order. `--decode-row-kernels 0|1|<subset>` (`moe`, `hc`, `router`, `kda`, `dsa`
joined by `+`) or `STRIX_GLM_RTX_ROW_KERNELS` selects them; decode commands take a
`row_kernels` field for an in-process A/B. `bench/row_mma_test.cu`
(`glm_row_mma_test`, run by hand on a GPU) compares the strips with the CUTLASS paths
on random tiles of every rate. The hyper-connection mix and the router also run for
two to four rows (the verify step of self-speculation). Round two of the one-row
kernels is opt-in (`--decode-row-kernels 2`, or `r2` in a subset): FP32 index
projections for up to four rows in one ascending `fmaf` chain per output
(`projection_fp32_row`), dense FP16 `[n][k]` strips for the KDA ordinary projections
(`row_dense`), A operands loaded before each tile's weight decode, four-wide reads in
the hyper-connection mix, and a parallel expert compaction and prefix sum. The
values are the same bits, but in an A/B on one RTX PRO 6000 it was not faster than
round one, so round one stays the default.

**Self-speculation (`mtp_self_spec`, pending: `--mtp-self-spec 1`, tierstream cache only).**
Each decode step verifies `[pending, draft 1..3]` in one trunk pass of two to four rows
with the plain step's per-row kernels, so every row's arithmetic is the plain step's
arithmetic; the step emits the accepted drafts plus the next token, which are the
plain greedy tokens. Drafts come from the MTP layer (one per step) or, with
`lookup_drafts` (`--spec-lookup 1`, pending: exact but not faster on the measured
prompts), from the longest earlier match of the context's suffix. A verify never
crosses a KDA B32 block: KDA promotes at the block end and replays the open block
from its checkpoint, so rejected rows need no undo; DSA rewrites their pool rows
in place. A non-finite flag falls back to the plain step. `generate` and the decode
commands (`bench` with `"spec": true`) use it; `--mtp-self-spec 1` or
`STRIX_GLM_RTX_MTP_SPEC=1` turns it on, and it is off without the tierstream cache
or with `--routes-out` (both one row per layer). It is adaptive: the measured
milliseconds per emitted token (EMA) of verify cycles and of plain steps pick the
path for the next 32 steps (the first 8 verify cycles and 4 plain steps measure
both), then 4 steps probe the other one, so a context where
drafts are rarely accepted decodes plainly; `--mtp-self-spec always` verifies every
step.

**Predicted prefetch for several rows (`predicted_prefetch`, pending).** With
`--tier-predict 1` each routed layer runs the next layer's router on its own copy
of the current hidden rows (up to four, on a side stream) and hands the predicted
ids to tierstream, which prefetches them; outputs are unchanged. After 16 tokens
without a demand miss the step predicts nothing (a warm cache is not churned by
prefetches) until the next miss. Tokens and every
logits digest equal `--tier-predict 0` with plain decode and self-speculation; its
speed is not settled yet (it is pending).

**Exact decode levers (pending rows).** `skip_empty_expert_groups`
(`--skip-empty-expert-groups 0|1`, `STRIX_GLM_RTX_SKIP_EMPTY_GROUPS`): the routed
MoE call lists the experts holding routed rows first on the device
(`c1_schedule.hpp`, after the dispatch, in the scatter cursor's space) and launches
only ceil(min(288, rows x 8) / 16) expert groups (1 instead of 18 for a decode
row); groups keep their 0.. numbering, parity and `retired[g-2]` guard; every
expert's decode, GEMM problems and epilogue are unchanged, and the merge sums the
routes in the same order. `persistent_dense_fp16` (`--persistent-dense-fp16 0|1`,
`STRIX_GLM_RTX_PERSISTENT_DENSE_FP16`): the shared experts of layers 3..44 and the
dense FFNs of 0..2 are decoded once at startup by the C1 decode window into
persistent FP16 operands (~2.8 GiB, the plan part "persistent FP16 shared + dense
experts (decode)", charged before the cache opens so the arena shrinks by it); the
decode calls skip their decode window. When the whole-device plan cannot afford
it, startup drops it and health `tier_plan.persistent_dense_fp16` says why. Both
are pending in every profile (their builtin, off, runs) until a window shows equal
decode digests with and without them; `--allow-pending` or the switch runs them. An explicit `--tier-vram-gib` arena and the keepable minimum (288 residents; the
planner adds the spares) are checked in the startup plan and the dry plan,
before authentication or any load. Every run uses the reduced schedule (32K TTFT
only with `--full-card`); a schedule whose printed estimate exceeds `--timeout`
is refused before execution. Without the API the interim
path streams all 43 routed layers through two rings (~150 GB/request, safety
floor only); health `prefill_experts` says which.

**Expert batches (arena below one layer).** The CUTLASS MoE path skips routes
with ids >= 288 and never dereferences zero-row experts' descriptors; each
route's expert output lands in its own (row, slot) contribution, merged in a
fixed k = 0..7 order. So a layer can run in batches of WHOLE experts (only the
batch resident and bound, the other routes masked to 0xFFFFFFFF), over the same
rows and workspace, with `Call::combine=false` on all but the last batch: the
result is bitwise the one-call result. `glm_native_bench --run --path cutlass
--batch-check B` proves it on the PRO (full call repeated, batched == full,
`BATCH-EXACT`). The runtime path that uses it (attention and routing of all
row chunks first, then each batch over every chunk with the per-chunk
contributions kept) is not wired yet: an arena below 288 + spares + guard is
refused.

**Startup order and memory accounting.** With deferred hashing, the shard
schema check reads metadata only; the per-matrix payload markers are read after
admission and source authentication (plan mode never reads them). Without
`--vram-total-gib` the usable device (99.86 GB) is the VRAM cap, also with only
`--ram-total-gib`; `--full-card` defaults to 93.0 GiB (= 99.86e9 B). CUDA graph
instantiations are measured (device use before/after) and charged to the ledger
under their own component, checked against the effective cap, and every
allocation records a staged sample, so receipts carry allocation-time peaks. The
reference backend runs only inside the untimed q1024 compare: the service
refuses decode/generation/timing on it and the harness refuses standalone
reference runs. All planning goes through ONE
adapter file, `budget_adapter.cpp`, the only GLM code calling a planner/ledger
API (tierstream `ss_budget_plan` / `ss_budget_ledger`); GLM supplies
named component sizes and keeps no budget arithmetic of its own. `tier_budget`
(arena-only budgets) is a thin call into it whose results equal the earlier GLM
rules (decode_host oracle grid). Startup plans the whole budget from shard
headers and blob sizes BEFORE source hashing, unit-file authentication and every
load, so a bad budget fails in seconds naming the parts and the smallest caps:
- `--vram-total-gib N --ram-total-gib M` (service and harness): WHOLE-device
  caps, as if the card had N GiB. Fixed device parts: CUDA context (measured,
  includes other processes), trunk + dense + shared (`base.bin`), activations,
  prefill KDA/DSA state, workspaces, prefill expert rings, decode KV/state +
  replay, lm_head (decoded + transient compressed), comparison slots, unit tables,
  a 1 GiB runtime reserve and the library's ring/guard slots and tables; host:
  pinned rings/staging, the library's bounce/load ring, the process's measured
  private bytes and a 1 GiB growth margin. The expert arena and the pinned tier
  get what is left; at cache open the plan is redone from the measured use.
- Without them: today's behaviour (the device room under the 99.86 GB cap with a
  4 GiB reserve; host above `--host-floor-gib`). `--tier-vram-gib` /
  `--tier-ram-gib` stay as arena-only caps (granted exactly or refused).
- Every device and pinned `Buffer` is charged by component name to the ledger
  (an over-cap allocation fails naming it); the library charges its own arena,
  tables and pinned blocks through `ss_options.budget`. Every receipt carries
  `memory`: ledger peaks by component, device-wide `cudaMemGetInfo` peak samples
  (startup, cache open, every request; NVML per-process is not linked), our
  host commit and `within_caps`; the harness gates them. Under WDDM every device
  allocation is also charged to the process commit (eviction backing, not
  resident): the full card measured 175.6 GB private = 96.0 GB of our device
  allocations + 79.6 GB host (78.2 GB pinned + 0.9 GB pageable tracked, 0.4 GB
  residual) with an 80 GB working set. The host side is therefore private bytes
  minus our ATTRIBUTABLE device bytes (exact allocation sizes: our buffers, the
  cache arena and tables; never the measured context + other processes part or
  the device-wide CUDA graph estimates, which stay on the host side). The gate
  `host_peak_commit_bytes` = max(sampled peak over every observation, OS peak
  commit minus the peak attributable backing); the OS term catches untracked
  temporaries between samples (it assumes the commit peak coincides with the
  backing peak). State export downloads/slices are charged as host temporaries
  for their lifetime and sampled. Receipts also carry `host_peak_sampled_bytes` (measured
  samples only, kept apart from the gate history),
  `host_peak_os_bound_bytes`, `wddm_device_backing_peak_bytes`,
  `graph_device_estimate_bytes`,
  `process_private_bytes_raw`, `process_peak_commit_bytes_raw`,
  `wddm_device_backing_bytes`, `host_tracked_bytes` and
  `host_commit_residual_bytes`.
- Harness `--full-card`: `--vram-total-gib 93` and `--ram-total-gib` = host RAM
  - 16 GiB unless given; plan: qualify 1K/8K, the untimed q1024 reference
  compare, TTFT (service op `ttft`: cold cache, prefill + first token on the
  cached path) at 1K/8K/32K (`ttft-32000.u32`), decode cold/warm on two chat
  kinds at 1K/8K. Reference-backend arms are never timed.
- `glm_native_service --plan-budget` (harness `--plan-budget`) prints the
  planned split from shard headers only: no hashing, no unit-file read, no CUDA
  calls (it reports `cuda_driver_loaded`). It also checks the library view (one part, stride == unit stride, 43
gated layers in order). A failure after `begin_token` retires the cache: the
worker reports that original failure on every later request (restart required)
instead of reusing a cache token that never ended. Receipts count device hits,
misses split into resurrected (no copy) and covered (in-flight copy), copies by
source (RAM blocks = copies - file reads, from the library's copy/read counters)
and file reads/bytes. Receipt totals sum every integer counter exactly (number
text parsed to uint64, overflow refused); the harness keeps every aggregated
counter with a per-token rate. `reference` remains the uncached test backend;
`--reference-compare` keeps it beside tierstream for same-window comparisons. Guarded slot rules mirror `ss_dg_slot` exactly.

**Unit file tool** (`glm_unit_file`, host only):

```powershell
BUILD_DIR/glm_unit_file.exe plan    --model MODEL_DIR --out OUT_DIR/glm-units.bin [--stripe STRIPE_DIR:64]
BUILD_DIR/glm_unit_file.exe dry-run --model MODEL_DIR --out OUT_DIR/glm-units.bin [--stripe STRIPE_DIR:64]
BUILD_DIR/glm_unit_file.exe copy    --model MODEL_DIR --out OUT_DIR/glm-units.bin --confirm-bytes PRIMARY_BYTES
BUILD_DIR/glm_unit_file.exe verify  --model MODEL_DIR --out OUT_DIR/glm-units.bin
```

`plan`/`dry-run` read shard headers only and write nothing; they print the exact
primary bytes (12,384 units x unit stride), manifest/descriptor paths, free space
per target and, per `--stripe DIR:K` (or `DIR:even:K`), the exact chunk count and
bytes of a tierstream `blocks:K` stripe plus the `ts-stripe make` command and its
`tierstream/stripe-ranges-v1` JSON (striping itself is tierstream's tool, which
also writes the stripe manifest). `copy` authenticates the source identities,
requires `--confirm-bytes` equal to the planned primary bytes, reserves all three
outputs as `.partial` by exclusive creation (any existing final or partial
refuses; dry-run readiness lists all six paths), keeps them owned (deny
write/delete) while writing units in numeric order through the retained verified
shard handles, the manifest (per-unit and whole-file SHA-256) and descriptor,
verifies every unit and the whole-file digest through the owned handle, then
publishes each with a no-replace rename of that handle. (These guarantees are
Windows-only: the non-Windows fallback reserves with `fopen("wbx")` but publishes
by an existence check plus rename, which is not race-free.) It finally publishes the
authenticated verification record and prints the manifest digest to pin. `reference` is the uncached per-expert backend kept only
for exactness/tests: device layers use resident descriptors, pinned layers copy
only selected expert spans, file layers read only selected ranges through the
retained verified handles (coalesced 64 KiB-aligned reads, scales checked
finite), all into eight compact slots with rebased descriptors indexed by
original expert ID (others zero; zero-count experts are never dereferenced).
It reads routes back on the host and is therefore eager, with no fetched-expert cache.

**HTTP.** `POST /v1/generate` with `{"tokens":[...],"max_new_tokens":N,
"identity":"glm-dec-rtx-native-v1","stop_token_ids":[...],"key":"64_HEX"}`.
The body `key` authenticates generation independently of prefill handoff
fields (401 on failure). Greedy only: sampling fields are refused with a
field-specific 400; unknown fields 400; validation precedes any mutation; busy
(shared lock with prefill) 503; a device/IO failure returns 500 and resets the
session while weights stay loaded.

**Receipts** carry executable/model/cache identity, capacity, residency/decode
profile, generated IDs and stop reason, prefill, first-token and loop times,
tokens/s, per-step tiers and selected experts, and logical, H2D and file bytes
(with per-token averages, startup/prefill excluded). Qualification mode adds
logits/hidden digests per step.

**Profiles.** One switch, `--profile default|fast|quality` (`STRIX_GLM_RTX_PROFILE`),
selects every lever of the decode service; the resolver (`profile.hpp/.cpp`) mirrors
the DS RTX resolver's structure (a table row per lever: value per profile, ready /
runnable, exact | bit-changing, its options, source; CLI > environment > profile >
builtin). `default` (= no `--profile` = today) runs every exact lever: `expert_tier`
tierstream, `placement` (residency / routes / round-robin), `eviction_policy` auto,
`spare_share` 5, `row_tile` auto (the planner rule), `cuda_graphs`, `flush_ready`
(tierstream `dg_flush_ready` as the DS tier site: the cache's demand stream is
flushed after each layer's ready signal so WDDM submits the ready write at once;
submission only; `--tier-flush-ready 0|1`, `STRIX_GLM_RTX_FLUSH_READY`; recorded in
health `tier_plan.flush_ready`);
`predicted_prefetch` is pending in every profile until a PRO window proves it
exact. `fast` adds the pending bit-changing rows `prefill_tile_large` (decided by
the tile-sweep window) and `dsa_index_mma` (the MMA index variants are not wired
into decode sessions): none is qualified, so fast = default today. `quality` =
default (no higher-precision expert pack), stated in the receipt and on stderr.
A pending row never refuses: its builtin form runs, it is listed in the receipt
and on one stderr line; `--allow-pending` runs pending rows that are in this build
for benchmarks only (receipt `unqualified: <rows>`). `--print-effective-config`
prints the resolution (every row with value, status, effective setting, switch,
kind, source) and exits before any model read or CUDA call; the same object is
`profile` in the ready event and every receipt. The harness forwards `--profile`,
`--allow-pending` and `--print-effective-config`.

**Evidence.** Every service record goes through ONE strict validator per kind
(`evidence_qualify` / `_compare` / `_bench` / `_ttft`) that takes the PLANNED
request (prompt n, steps, arm labels, effective tile, startup allocation) and
returns canonical evidence: lowercase SHA-256 digests, per-rank byte counts > 0
(rank 0's equal to the sum of its component extents), the replay checkpoint,
frontier = the planned frontier (n - 1 after a prefill, n - 1 + steps after
decoding), non-empty ordered component extents, token IDs that are ints in
[0, 154880) of the planned length, every token bound to its step record from
index 0, the initial state for qualify, and for TTFT the reported tile equal to
the effective requested tile within a positive allocation. Any defect is
NO-REFERENCE with its reason. Every judge (incremental vs fresh, reference vs
tierstream, cold vs warm) and both tile comparisons work only on these objects.
Execution has one outer finalization: a case whose result cannot be built is
NO-REFERENCE, any other exception is recorded as `failure`, unfinished cases are
INCOMPLETE, the service is always stopped, and the summary is always printed
(non-zero exit; a failed summary builder, a comparison item that could not be
compared and a summarize() error are run failures too). Step positions
are evidence (n - 1 + j for every arm step from index 0, n - 1 + k for each
fresh next token). Residual: fields that no second record witnesses are checked
for form only: the qualify initial state's rank-1 bytes, replay checkpoint and
component layer/index numbers, and step 0's logits; the initial state's rank-0
bytes and component byte extents are checked only for internal consistency
(rank-0 bytes = the sum of the extents), with no independent witness.

**Exactness gate.** `decode_qualification_window.py` (job object, absolute
deadline <= 30 min, pinned service/receiver copies, owned-process cleanup)
starts one service with `--decode-commands` and the forced file tier
(`--max-device-layers 19 --max-pinned-layers 22 --require-file-tier`). Prompts
must have exactly 1024/8192 tokens (`q1024.u32`, `q8192.u32`, and
`chat-<kind>-<1024|8192>.u32` for all seven kinds); `--steps >= 3`. Per
qualification prompt it checks:
session prefill == unchanged prefill producer per-rank state; for t=1..steps the
incremental state equals a fresh prefill of `prompt+generated[:t]` (per-rank
SHA-256, component extents/order, KDA replay checkpoint digest); both arms then
consume the pending token with identical complete logits digests and greedy
token. The 1K incremental export is also verified by
`glm_native_receiver --snapshot FILE RANK P N` (receiver layout + payload hash,
header excluded, no live-import widening). Chat prompts run timing arms from one
restored session snapshot, cold/warm order interleaved; both arms must match
the requested labels and step count and agree on tokens, every complete logits
digest and the final state/replay digests (digest work excluded from the timed
loop) before timings are reported. Required coverage: consumed positions cross
a B32 boundary (residues 31, 0, 1), sparse DSA selection (>512 complete pools)
at 8K, nonzero file-tier expert reads during generation, and every planned chat
case. Exactness of completed cases and completion of the planned workload are
reported separately; the verdict is EXACT only when both hold, INCOMPLETE when
coverage or planned cases are missing, MISMATCH/NO-REFERENCE on failed
evidence. A failed gate stops qualification. With `--expert-tier tierstream` (default;
`--unit-file PRIMARY`, `--tier-ram-gib` default 32 so the file tier is used) the
same exactness gates apply; health must report the tierstream backend with a
three-tier plan, cold arms run after `reset_open`, warm arms after an untimed
warm-up pass, and each arm reports misses/token, device hits, resurrections,
covered misses, RAM copies, file reads and file/H2D bytes per token. The service
runs with `--reference-compare --unit-manifest-sha256 PIN`; run order is
qualification (q1024, q8192), then reference comparisons (q1024, q8192 and the
explanation-1K, coding-1K and explanation-8K chat cases: the identical
continuation under the reference tier and the tierstream cache must give
identical tokens, every complete logits digest and identical final state/replay
digests), then the timing arms. Budget: about 2 prefills plus 2 x steps tokens
per comparison (roughly 5-8 minutes for the five), inside the 30-minute deadline;
unfinished comparisons are INCOMPLETE.

```powershell
python -B src/glm/rtxpf/producer/decode_qualification_window.py --build BUILD_DIR --model MODEL_DIR --prompt-dir PRIVATE_PROMPTS
python -B src/glm/rtxpf/producer/decode_qualification_window.py --build BUILD_DIR --model MODEL_DIR --prompt-dir PRIVATE_PROMPTS --check-inputs
# authorized PRO window only:
python -B src/glm/rtxpf/producer/decode_qualification_window.py --build BUILD_DIR --model MODEL_DIR --prompt-dir PRIVATE_PROMPTS --allow-gpu --timeout 1800
```

**Deviations from the design note** (exact/safe choices): the head uses a
separate `head.bin` blob rather than joining `base.bin`; generation is
authenticated by a body `key`; per-rank digests are hashed in process (the
receiver snapshot independently re-checks the 1K export) to avoid writing
hundreds of MB per step; the gate additionally pins session prefill to the
unchanged prefill producer; by the merged A+B scope the uncached reader is a
reference backend only and the product path is the tierstream adapter, which
waits for the shared tierstream binding. The current tierstream device-gated
layout refuses units without a device or pinned home, so an NVMe demand tier
must come from that library. Native byte equality (row-count independence of
CUTLASS hot/cold tiles and the replayed KDA block) remains a GPU gate.

The decode CPU tests run as a CTest fixture pair: `producer_decode_host`
(FIXTURES_SETUP) publishes a fresh, nonce- and time-stamped receipt artifact to
`BUILD_DIR/ctest-fixtures/`, and `producer_decode_python` (FIXTURES_REQUIRED)
checks that publication and reads it through `cache_summary`.
