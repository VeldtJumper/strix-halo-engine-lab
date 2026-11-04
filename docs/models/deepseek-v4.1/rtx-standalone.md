# DeepSeek V4.1 Flash on one RTX PRO 6000

## What it is

`strix-cuda-ds41-replica` runs the full DeepSeek V4.1 Flash target and a
DSpark drafter on one RTX PRO 6000 (96 GB, Windows WDDM). The standalone
implementation is [cuda_ds41_replica.cu](../../../apps/cuda_ds41_replica.cu),
with the worker protocol in
[cuda_ds41_replica_serve.hpp](../../../apps/cuda_ds41_replica_serve.hpp).

This profile uses the Q2 GGUF: routed gate/up **IQ2_XXS**, routed down
**Q2_K**, and dense **Q8_0**. Its exactness claims apply to this identity;
equality with the two-node trellis profile is not claimed.

Experts occupy an exclusive VRAM + pinned-host cache managed by
[tierstream](https://github.com/hyukudan/tierstream)'s device-gated decode
mode. Missing and predicted experts move over PCIe at the MoE sites. Dense
weights and the drafter share the target GPU; the drafter shares the target's
embedding and output weights. The built-in drafter pruning keeps 64 experts
per stage. See tierstream's
[library reproduction guide](https://github.com/hyukudan/tierstream/blob/main/docs/reproduce.md)
for the library side.

## Build on Windows

Use an x64 MSVC developer PowerShell with CMake (3.24 or newer), Ninja, and
a CUDA toolkit supporting architecture 120. Build tierstream's CUDA static
library first, using a compatible toolchain.
This replica is built and tested against tierstream `main` at commit
`4c823b4` (#14); the three-tier placement, whole-device budgets and file-read
queue use its device-gated file tier and budget planner. The schedule and
budget CPU tests link its CPU library (`build-cpu`) when that build exists.

From the repository root, with a tierstream checkout at the relative
placeholder `TIERSTREAM`:

```powershell
cmake -S . -B build-rtx -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl `
  -DSTRIX_CUDA_DS41_ONLY=ON -DSTRIX_ENABLE_CUDA_DS41=ON `
  -DSTRIX_FAMILY_DEEPSEEK41=ON -DSTRIX_ENABLE_HIP_RUNTIME=OFF `
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRIX_TIERSTREAM_DIR=TIERSTREAM
cmake --build build-rtx --target strix-cuda-ds41-replica
```

[CMakeLists.txt](../../../CMakeLists.txt) defines these options. The CUDA-only
configuration enables the CUDA backend and requires the family enabled and
HIP disabled. `STRIX_TIERSTREAM_DIR` is a cache path to the library's source
tree: CMake imports `build-cuda/tierstream.lib` and its `include` directory,
then links the CUDA runtime/driver and Windows `avrt`/`winmm` libraries.
Leaving that variable empty omits the replica target. The executable links
`strix_ds41_cuda_tier`, `strix_ds41_cuda_prefill`, and `strix_ds41_cuda_dspark`.

## Run and persistent worker

This is an imported-state decode runner. Supply the admitted model, its
receipt binding SHA-256, size and modification time, an exported attention
state, the matching greedy reference, and tokenized chat sequences. These
assets are external. `CHATS.tsv` has a label, a tab, and comma-separated token
IDs per line; chat indices are zero-based. Each selected chat must contain
at least the requested prompt length. All filenames below are placeholders.

```powershell
.\build-rtx\strix-cuda-ds41-replica.exe MODEL.gguf `
  --receipt MODEL.receipt.json --import-state STATE.bin --ref GREEDY.log `
  --draft DRAFTER.gguf --draft-device 0 --arena-gib 80 `
  --phases exact0,pressure,exact8,wforce0,spec0,exact0,chats0,schats0,chats0 `
  --steps 64 --max-rows 6 --max-positions 4096 `
  --seqs-file CHATS.tsv --chats 0 --chat-prompt 96 --chat-steps 128 `
  --qual --json RESULTS.jsonl
```

The imported position plus the requested work and safety allowance must fit
`--max-positions`; increase it for a later imported state. Size the arena and
pinned-host budget for available memory; defaults are 74.8 GiB for a keep-64 drafter arena (76.5 for keep-32,
71.5 for a full drafter, 80 without a drafter),
68 GiB maximum pinned allocation, and a 6 GiB available-host-memory margin.

To load once and accept repeated requests:

```powershell
.\build-rtx\strix-cuda-ds41-replica.exe MODEL.gguf `
  --receipt MODEL.receipt.json --import-state STATE.bin `
  --draft DRAFTER.gguf --draft-device 0 --arena-gib 80 `
  --phases chats0,schats0,chats0 `
  --seqs-file CHATS.tsv --chats 0 --chat-prompt 96 --chat-steps 128 `
  --max-rows 6 --max-positions 4096 --serve SPOOL --qual
```

Wait for `SPOOL/ready.json`. Write a request as `SPOOL/queue/REQUEST.tmp`,
then rename it to `REQUEST.req`. The file contains one argument per line,
for example:

```text
--phases
chats0,schats0,chats0
--reset-cache
open
--json
REQUEST.jsonl
```

The worker moves it through `run/` to `done/` and writes `done/REQUEST.json`
with `rc`, elapsed time and any error. Requests restore launch-time runtime
options before applying their overrides. `--reset-cache warm` is the default;
`open` restores the expert cache's initial placement for fresh-load comparisons.
Model, arena, drafter pruning and decode kernel choices require a new worker.
A request containing only `--quit` stops it. This is a file-queue worker.

The model can also be read from two NVMe drives at once (a verified partial stripe on the second drive):
[rtx-two-drives.md](rtx-two-drives.md).

## Expert placement: VRAM, pinned RAM, NVMe

The routed experts (40 layers x 384, 9.95 MB each, 142.4 GiB) live in a
three-tier cascade: the VRAM arena, then pinned host RAM, then the model file
on NVMe (one drive, or a multi-drive stripe with `--weight-dirs`). The
replica sizes the tiers at load and prints the plan as a `tier_plan` JSON
event (GiB and experts per tier, the spare blocks, the clean-copy share, the
staging buffers):

| Flag | Meaning | Default |
|---|---|---|
| `--vram-expert-gib G\|auto` | VRAM arena for resident experts (same as `--arena-gib`; spares on top) | by drafter, see below |
| `--ram-expert-gib G\|auto` | Pinned RAM tier, spare blocks and staging included; `auto` = what the host can pin now, keeping `--host-margin-gib` free and within `--max-pinned-gib` | `auto` |
| file tier | Experts that fit neither tier stay in the model file and are read on demand into pinned staging buffers | on when needed; `--no-file-tier` / `STRIX_DS41_FILE_TIER=0` / `--baseline` refuse as before |
| `--ram-inclusive-gib G\|auto\|0` | Pinned blocks that device experts keep as clean copies after a promotion; evicting such an expert is a drop, not a device-to-host copy. Only RAM left after the pinned homes is used (`auto`: up to 16 GiB) | `0` (exclusive, as before) until measured |
| `--disk-prefetch` | Predictions may also read file-tier experts (a synchronous read on the library's service thread) | off: demand reads only |
| `--disk-staging N` | Pinned staging buffers of the file reads | 8 |
| `--disk-queue auto\|N` | File reads in flight at once (at most `--disk-staging`). `auto` = tierstream's clamp(3 x read locations, 4, 8): one drive 4, two 6, three or more 8. Measured at 24+32 on one drive: 4 in flight +20 % plain / +26 % DSpark vs one at a time, 16 = 4 (one drive saturates), same bytes | auto; `STRIX_DS41_DISK_QUEUE=auto\|N`; `--disk-queue 1` / `--baseline` = one at a time |

Which experts go where: the seed priority (the saved residency, then the
`--routes` usage order) fills the arena first, then the RAM tier; the least
used experts stay in the file. At run time a missing expert comes from the
next tier down, and victims move down one tier: a VRAM victim is written back
into a free pinned block; when no block is free, the least recently homed
pinned expert is dropped to its file (no copy, the file holds the same bytes),
and only with no pinned expert left is the VRAM victim itself dropped. A
clean copy (inclusive share) makes the VRAM victim's eviction a drop.

When the budgets cover every expert (today's PRO with 128 GB RAM), the plan,
the library options and every byte are today's: no file tier, no staging
buffers, the same pinned blocks. The only default behaviour change is that a
host that cannot pin every non-resident expert loads with a file tier instead
of refusing (`not enough available host memory for the pinned tier`). All
placements are exact: every expert byte the GPU reads is the file's byte, so
tokens and logits equal the two-tier run (the PRO identity scripts compare
chat hashes across budgets).

Expected throughput by budget. Estimates from a replay of the recorded routes
(held-out fold) through the three-tier cache model, calibrated on today's
measured plain decode (9.27 misses/token, 70 tok/s) and on the measured
random expert read rates (D: alone ~1 GB/s, D:+E: ~3.1 GB/s; three drives not
measured, assumed equal to two). DSpark is an upper bound (today's
controller/plain ratio). The measured columns are filled by the PRO runs.

| Budget (VRAM + RAM) | File tier | Misses/token (file reads) | Drives | Plain tok/s est. | DSpark tok/s est. (<=) | 96-token chat prompt s est. | Measured DSpark cold / warm tok/s |
|---|---|---|---|---|---|---|---|
| 96 GB PRO + 128 GB RAM (arena 74.8 GiB, today) | 0 | 9.3 (0) | any | 70 | 84 | 1.3 | 72-77 / 90-95 (mdec loads) |
| 96 GB PRO + 64 GB RAM (arena 74.8, RAM auto ~48) | 20 GiB | 9.3 (1.2) | D: | 38 | 45 | 3.2 | pending |
| | | | D:+E: | 55 | 66 | 1.9 | pending |
| 48 GiB VRAM + 64 GiB RAM | 31 GiB | 19.0 (2.4) | D: | 23 | 28 | 6.1 | pending |
| | | | D:+E: / D:+E:+C: | 38 | 46 | 3.3 | pending |
| 32 GiB VRAM + 32 GiB RAM | 79 GiB | 32.9 (12.6) | D: | 7 | 8 | 26 | pending |
| | | | D:+E: / D:+E:+C: | 16 | 19 | 10 | pending |
| 20 GiB VRAM (24 GB GPU) + 48 GiB RAM | 75 GiB | 54.6 (11.3) | D: | 7 | 8 | 23 | pending |
| | | | D:+E: / D:+E:+C: | 14 | 17 | 9 | pending |

File reads cost a whole expert per miss (three part reads, about 10 ms on a
single drive); a second drive roughly triples the random-read rate. Loads with
a file tier are shorter (fewer bytes read at load). The replica reports the
file tier's counters in a `tier3` event at the end of each request (file
reads and their host time, pinned drops, victim drops, clean copies kept and
dropped).

### Whole-device budgets: `--vram-total-gib` / `--ram-total-gib`

`--vram-expert-gib` / `--ram-expert-gib` size the expert tiers only: the CUDA
context, the dense trunk, the attention state, the drafter, the workspaces and
the graphs come on top (a "24 GiB" expert arena used ~46 GB of the card). A
total cap means the whole device, "as if the user had a 24 GiB GPU":

| Flag | Meaning | Default |
|---|---|---|
| `--vram-total-gib N\|none` (`STRIX_DS41_VRAM_TOTAL_GIB`) | Everything this process holds on the device: context + trunk + attention state for `--max-positions` + drafter + workspaces + arena + margin | none (today) |
| `--ram-total-gib M\|none` (`STRIX_DS41_RAM_TOTAL_GIB`) | The process's whole commit, pinned memory included | none (today) |
| `--vram-margin-gib G` / `--ram-margin-gib G` | Kept for allocations after the plan (graph capture, warmup, per-request buffers) | 1.5 / 1 |
| `--budget-receipt` | Measure and print the budget events without caps (implied by a cap) | off |
| `--budget-sample-ms N` | Period of the NVML peak sampler (0: stage samples only) | 100 |
| `--drafter auto\|on\|off` (`STRIX_DS41_DRAFTER`) | Whether `--draft` loads the DSpark drafter. `auto` = off when a file tier is certain and the expert reads have one read location, else on (below). Without the drafter its bytes go to the expert tiers and speculative requests run plain decode (event `drafter_off_plain`; the same tokens) | `auto`; `--baseline` = `on` |

At load the replica creates the context, allocates the trunk, the drafter, the
decode workspaces, and measures each part (`cudaMemGetInfo`, minus the device's
use before the process as NVML reports it). Then tierstream's budget planner
(`ss_budget_plan` in `strix_stream/ss.h`, shared with the other engines) gives the arena
what is left: `min(--vram-expert-gib, (N - fixed - margin - bank/site/graph
reserves) / expert)` slots, and the pinned tier `min(--ram-expert-gib or auto,
(M - process commit - margin - load staging) / expert)` blocks; the rest of
the experts stay in the model file. A cap only shrinks a tier: caps that cover
everything give today's placement. When the fixed part does not fit, the load
is refused before the arena is allocated, with the minimum (e.g. `the minimum
is --vram-total-gib 20.24 (21 rounded up)`).

The `budget` events: `stage:"plan"` (caps, measured components, the plan),
`stage:"load"` (measured per component, declared bytes, tiers, peaks),
`stage:"warmup"`, and `stage:"request"` at every request end (`stage:"run"`
without `--serve`). Each carries the device peak (stage samples + the NVML
sampler), the host peak (the process's resident memory: on Windows the private
working set, its maximum sampled by the replica; on Linux VmHWM; plus pinned
bytes if it does not show them) and `peak_le_cap` (tierstream's `ss_budget_ledger_observe` /
`ss_budget_ledger_within_caps`). The `plan` object is `ss_budget_plan_json`'s
(`resident`, `arena_slots`, `pinned`, `disk`, `host_cap_units`, `device_capped`,
`host_capped`, `min_vram_total_gib`, ...). Above a cap at load or after warmup the
worker refuses to serve; at a request end the request fails (rc 1).

Estimate (CPU, from the GGUF header and a
PRO load at 2304 positions): trunk on the device 9.41 GiB (8.79 GiB of Q2/Q8
file tensors), context + untracked ~2.8 GiB, attention state 17 MiB, drafter
keep-64 4.12 GiB: ~16.5 GiB fixed, so the minimum is ~20.2 GiB and a 24 GiB
card keeps ~5 GiB of arena (~440 resident experts besides the 96 spares).

### Drafter by budget: `--drafter auto|on|off`

DSpark pays when the experts it verifies are in VRAM or RAM; with a file tier
on one drive its windows read more experts than plain decode and lose.
Measured on the PRO (one NVMe drive, the same chat tokens in every arm):

| Budget | Plain, no drafter | DSpark | Plain, drafter loaded |
|---|---|---|---|
| 24 GiB + 32 GiB, one drive | 8.18-8.20 tok/s | 5.94-7.78 | 6.24-6.44 |
| 32 GiB + 32 GiB, one drive | 10.91 | 10.20 | |
| full budget (no file tier) | 26.7 | 105-108 | |
| 24 GiB + 32 GiB, expert stripes on 2 drives | 7.75-7.78 | 9.22-9.27 | |
| 48 GiB + 32 GiB, expert stripes on 2 drives | 31.5 | 20.9-21.6 | |
| 48 GiB + 32 GiB, one drive | 32.2 | 20.6-21.8 | |
| 64 GiB + 32 GiB, one drive | 52.2 | 47.4-51.8 | |

With `auto` at 24 GiB + 32 GiB the replica measured 8.05-8.13 tok/s (the drafter-free number), and in every row above
`auto` picks the faster or an equal arm: off on one drive with a file tier, on with stripes on two drives or without a
file tier. The exception is 48 GiB + 32 GiB with stripes on two drives, where `auto` keeps DSpark and it is about 32 %
slower; use `--drafter off` there until the rule is updated.

`auto` turns the drafter off exactly when a file tier is certain and the
expert reads have fewer than 2 read locations. A file tier is certain when both
sides are bounded (VRAM: `--vram-total-gib` or an explicit `--vram-expert-gib`;
RAM: `--ram-total-gib` or an explicit `--ram-expert-gib`) and the room for
experts is below every routed expert's bytes (layers x experts x one expert,
from the GGUF header). The room is decided before the load, so it is an
estimate that never undercounts: each cap minus its margin, minus the trunk's
GGUF bytes on the device (nothing on the host), each side at most its explicit
tier. Read locations = the model file plus each `--weight-dirs` /
`STRIX_WEIGHT_DIRS` directory holding `<model file>.stripe.json` (tierstream's
count). Doubtful cases keep the drafter. The startup event `drafter_select`
(`mode`, `loaded`, `reason`, `file_tier_expected`, `read_locations`,
`expert_bytes`, `room_bytes`) and `--print-config` (`drafter`,
`drafter_select`) report the decision. Without `--serve`, speculative phases
under `--drafter off` are refused; under `auto` they run their plain equivalent
(`schats<K>` -> `chats<K>`, `spec<K>` -> `exact<K>`) with the reason printed.

## Profiles: `--profile default|fast|quality`

One switch selects every lever (`--profile NAME` or `STRIX_DS41_PROFILE=NAME`; the option wins). The per-lever options
below stay as expert overrides. Precedence: an explicit option (the last of the lever's own options and `--baseline`) >
an explicit environment variable (`STRIX_DS41_RTX_BASELINE=1` over the lever's own variable) > the profile > the builtin.

| Profile | What runs |
|---|---|
| `default` (also no `--profile`) | Today's replica: every exact lever of the table below ON. `--print-config` is byte for byte the configuration without the switch. The layer-major exact prompt path is wanted here too but is pending (not in this build). |
| `fast` | `default` plus every further exact lever and every bit-changing lever that has passed its quality qualification. None has yet, so `fast` runs the `default` levers and lists the fast-only rows as pending. |
| `quality` | = `default`. DeepSeek on the RTX has no higher-precision expert pack, so there is nothing to trade for quality. |

| Lever | Kind | default | fast | quality | Status | Override |
|---|---|---|---|---|---|---|
| Lean decode state | exact | ON | ON | ON | ready | `--no-lean-state` |
| Q8 R1 GEMVs | exact | ON | ON | ON | ready | `--q8-r1-off`, `--q8-r1-pad` |
| dsrtxk decode kernels | exact | ON | ON | ON | ready | `--dsrtxk-off` |
| Fusions (hc, shared expert) | exact | ON | ON | ON | ready | `--fuse-off` |
| Decode forms (deferred records, router form 2, qa_kv S=1) | exact | ON | ON | ON | ready | `--records-inline`, `--router-pf0`, `--qakv-auto` |
| Auto eviction | exact | ON | ON | ON | ready | `--evict lru` |
| Controller window (built-in prior, seeded costs, chat scope) | exact | ON | ON | ON | ready | `--window`, `--controller-prior`, `--controller-costs`, `--controller-scope` |
| Drafter keep-64 | exact | ON | ON | ON | ready | `--draft-keep 0` |
| Drafter vocabulary auto | exact | ON | ON | ON | ready | `--draft-vocab full` |
| Deferred write-backs | exact | ON | ON | ON | ready | `--no-wb-defer` |
| CE pacing | exact | ON | ON | ON | ready | `--no-ce-pacing`, `STRIX_DS41_CE_PACING=0` |
| Write-back stream | exact | ON | ON | ON | ready | `--no-wb-stream`, `STRIX_DS41_WB_STREAM=0` |
| File tier fallback | exact | ON | ON | ON | ready | `--no-file-tier`, `STRIX_DS41_FILE_TIER=0` |
| Residency auto seed / save | exact | ON | ON | ON | ready | `--residency-seed none`, `--residency-save none` |
| Warmup graphs (serve) | exact | ON | ON | ON | ready | `--warmup-graphs none` |
| Engram RAM cache | exact | ON | ON | ON | ready | `--engram-cache-mb 0` |
| Layer-major exact prompt path | exact | pending | pending | pending | PRO-qualified exact (about -5 %); pending owner review, not in this build | - |
| Prefill service closed batches (batch-long) | exact | - | pending | - | pending: decision on the first member's TTFT | prefill service `--batch-max` |
| DS FP16 dense prefill | bit-changing | - | pending | - | pending: quality qualification | prefill service |

A pending row is never silently used. The builtin form runs, the row is listed under `pending` in the `profile` object
of `--print-config` and in the run's first `profile` event, and stderr names it. `--allow-pending` (or
`STRIX_DS41_PROFILE_ALLOW_PENDING=1`) runs a pending row whose implementation is in this build, for benchmarks only,
and stamps the receipt `"qualification": "unqualified: <rows>"`. No replica row is runnable that way today. The
`profile` object gives each lever's value, status (`profile`, `explicit-cli`, `explicit-env`, `baseline-cli`,
`baseline-env`, `pending` or `unqualified`), effective form, kind, switches and source commit. `--baseline` still
restores every reference form, and `--profile` is a load-time option that a worker request cannot change.

The prefill service (`strix-cuda-ds41-prefill-service`) does not take `--profile` yet. Its default already runs the
fast dense / fast MoE arm (`--exact` for the reference bits), so a `default` profile there would change today's service
behaviour. Its rows (batch-long, FP16 dense) are in the resolver's table (`strix/ds41_rtx_profile.hpp`) for the
service to use once that decision is made.

## Optimizations (default ON) and off switches

These options change scheduling, placement or draft proposals. Committed
outputs retain target-greedy exactness. Gains below describe prior qualification;
this defaults change is checked on CPU and does not establish new GPU results.

| Item | What | Default | Measured gain | Exact | Off switch |
|---|---|---|---|---|---|
| Decode kernels | Router, output projection, forks, expansion fusion, routed MoE, attention and normalization | ON | All-hit 12.62 to 10.50 ms/token; full loop 5.8-10.8% | Bitwise | `--dsrtxk-off` |
| Fusions | Hyperconnection and shared expert | ON | Included in decode qualification | Bitwise at supported shapes | `--fuse-off` |
| Records | Move records off the critical path | Deferred | Qualified; isolated gain not listed | Same outputs | `--records-inline` |
| Router | Form 2 | ON | R1 17.3 to 15.4 us; 8-layer device time -0.5% | Bitwise rows 1-8 | `--router-pf0` |
| QA/KV | Split 1 | ON | 8-layer device time -1.54 to -1.94% | Bitwise | `--qakv-auto` |
| Q8 R1 | Compact single-row kernel, no padding | ON | 8-layer device time -0.80 / -0.53% | Bitwise | `--q8-r1-off`; `--q8-r1-pad` selects the padded form |
| Drafter | Keep most-used experts per stage | 64 | Qualified; isolated gain not listed | Target-greedy | `--draft-keep 0` |
| Controller | Built-in survival prior and seeded costs, no exploration or cost learning | Warm, chat scope | Qualified; isolated gain not listed | Target-greedy | `--controller-prior none` / `--controller-costs none`; `--controller-scope run` restores run scope |
| Write-backs | Deferred scheduling | ON | Qualified; isolated gain not listed | Bitwise | `--no-wb-defer` |
| Engram | RAM record cache | 512 MiB | Avoids repeat storage stalls | Same bytes | `--engram-cache-mb 0` |
| Cold load | Tierstream pipeline | ON | Qualified; isolated gain not listed | Bitwise | `DG_LOAD_SERIAL=1` |
| Graph warmup | Capture common graphs before worker readiness | `0,8` in serve mode | First 16 steps 56.05 to 33.99 ms/step | Bitwise | `--warmup-graphs none` |
| Residency | Reuse valid prior placement, save on worker exit | Auto | Step 0 332.6 to 156.0 ms; first 16 steps -20% | Bitwise | `--residency-seed none` / `--residency-save none` |
| Lean state | Decode-only attention state: an allocation ledger and a split at the maximum decode position instead of full prefill buffers | ON | 32K state 2,486 to 59 MiB (+264 expert slots, plain decode +3.0 %); 128K now loads (state 195 MiB; without it the replica does not fit) | Bitwise: prefill states, greedy logits, DSpark windows, chat tokens, worker requests | `--no-lean-state` |
| Arena sizing | Size the expert arena from free VRAM minus a headroom | Off (fixed arena by drafter) | More expert slots when VRAM is free | Same outputs | opt-in `--arena-gib auto [--arena-headroom-gib G]` |
| Eviction | Public tierstream scored victims after verify windows; plain steps keep LRU | Auto | DSpark misses/token −14 %, median +6.4 % tok/s (7 chats); plain decode neutral | Bitwise (only residency changes) | `--evict lru` |
| CE pacing | Cap background write-back debt on the single copy engine at about one expert (demand and speculation stay first) | ON | 4 alternating loads: DSpark +2.7 % / plain -0.5 % / wall -1.1 % adjacent, neutral bracketed | Bitwise (only copy timing changes) | `--no-ce-pacing` / `STRIX_DS41_CE_PACING=0` |
| Write-back stream | Deferred write-backs go out on their own D2H stream (tierstream `dg_wb_stream`) so they overlap the next layer's demand H2D | ON | 4 alternating loads: DSpark plain -2.0 %, controller -6.4 % per commit; copy bench duplex 43.3 to 44.8 GB/s | Bitwise (same copies, order and destinations; only the stream changes) | `--no-wb-stream` / `STRIX_DS41_WB_STREAM=0` |
| File tier | Load with VRAM + pinned RAM + NVMe tiers when the host cannot pin every non-resident expert, instead of refusing | ON (only when needed) | Pending PRO runs (estimates above) | Bitwise (placement only) | `--no-file-tier` / `STRIX_DS41_FILE_TIER=0` |
| Clean copies | Inclusive share of the pinned tier (`--ram-inclusive-gib`) | Off | Pending PRO runs | Bitwise (placement only) | `--ram-inclusive-gib 0` |

`--baseline` restores all reference forms at that argument; later explicit
flags override their own settings. `STRIX_DS41_RTX_BASELINE=1` applies the
reference forms before all arguments. Baseline uses 96 host spares, run controller
scope, a 71.5 GiB arena with a drafter (80 without), full draft vocabulary,
and serial loading unless the caller already set `DG_LOAD_SERIAL`. It also
restores phase-inferred row capacity and 4 pressure spares.
`--dsrtxk`, `--fuse`, `--records-deferred`, `--router-pf`, `--qakv-s1`, and
`--q8-r1` can re-enable those features after baseline.

Baseline also turns the file tier off (the old refusal) and the clean-copy share to 0.

Normal defaults use 48 host spares, 8 pressure spares and 6 maximum rows.
An explicit `--arena-gib` wins over the drafter-dependent arena rule.
Vocabulary defaults to `auto:DIR` when `STRIX_DS41_DRAFT_VOCAB` names a directory,
otherwise when `ds41-draft-vocab` exists beside the drafter; absent a directory,
it uses `full` with a one-line notice. Explicit `--draft-vocab` wins.
The auto residency path is `MODEL.gguf.residency`, overridden by
`STRIX_DS41_RESIDENCY`. A missing seed quietly uses routes order; an invalid
auto seed reports a one-line notice and falls back. Explicit seed files retain
strict errors. Saves use a temporary file and rename. Unsupported Q8 R1 shapes
use the generic exact kernel with a one-time notice. Non-serve runs do not
warm up graphs or save residency unless explicitly requested.

`--print-config` prints one resolved JSON object and exits before any CUDA call,
fixture or model read. It can read small prior, pruning and residency files;
model and draft paths may be dummy paths. The JSON reports effective prior
counts and expert IDs so built-in and file-based configurations can be compared.
`strix-cuda-ds41-config-tests` is registered with CTest and requires no GPU or model.
Window expert prefetch remains opt-in through phases ending in `w`.

## Exactness check

Use an off-switch run as the reference: repeat the same command and fixtures
with `--baseline`. Compare complete per-chat token vectors
and state digests against the default run, with the same capacity and model.
Drafter pruning and controller choices can change acceptance, never the
committed target-greedy sequence.

The example's phase order exercises plain greedy (`exact0`), forced
write-back pressure (`pressure`), predicted-expert prefetch (`exact8`), forced
verify windows (`wforce0`), speculation (`spec0`), and plain greedy again.
Each phase reimports the same state. Pressure must exercise demand banks,
resurrections and mid-token write-back collection. Forced windows vary row
and kept-prefix lengths, insert wrong tokens after the retained prefix, and
commit only that prefix. Speculative windows commit only verified greedy
tokens. Both compare retained rows with the plain greedy reference and must
finish at the same position and state digest. Exact phases require at least
64 compared steps and reference coverage for every position.

Read the JSON lines on stdout or in `--json` output:

- `gate`: `pass: true`, `compared == matched == steps`, and `digest_same: true`.
  Pressure also requires nonzero `bank`, `resurrected`, and `collected_mid`.
- `canonical`: `same: true`, with matching exported state digest and position.
- `chat_tokens`: `plain_found: true`, `pass: true`, and `same_prefix == compared`.
  With equal lengths, each plain/speculative chat vector has `chat_steps + 1`
  tokens. The first plain chat supplies the canonical vector; the example
  includes plain chats before and after speculation.
- `chat_hash`: compare `len` and `fnv` across the reference and default runs.
  `--qual` additionally emits chat token vectors and window-row records.
- `spec_summary` / `summary`: throughput and timing, separate from gate verdicts.
  A worker request must also finish with `rc: 0` in its completion JSON.

## Measurements and variance

The published [README](../../../README.md) and
[performance record](../../performance.md) report **82.3 / 87.7 / 94.4 tok/s**
for code / English / Spanish on the same load. Quote code throughput across
loads as **82–95 tok/s**. Load-to-load variation is up to **~10%**: compare
options within one loaded process, keeping fixtures, capacity and cache-reset
policy fixed. Prompt lengths and generated-token counts are unspecified for
these latest decode measurements.

The same published record reports full loading **108.7 → 61.8 s**, persistent
worker requests **8–18 s/request** without reloading, and fresh-load cache reset
**3.4 s**. Loading and persistent-worker work are marked **in development**;
these are workload-specific measurements, separate from decode throughput.

## Conversation sessions

Named sessions with RAM parking are on by default for the qualified
`rtx-tp1-decode-rows` route. RAM eviction drops the session and returns
`session_not_found`/404 on its next suffix-only turn. Implicit continuation and
full-history matching use only this worker's RAM; a full-history miss recomputes.
Checkpoints are explicit save points. Durable reads require explicit
restore/resume with `--session-durability disk` by name or an explicit
`--session-path`, and return exactly that checkpoint even if RAM had advanced. One worker exclusively owns
each managed tier; competing workers receive `session_tier_busy`/409.
`--session-disk off` / `STRIX_SESSION_DISK=0` disables the managed tier.
Disable automatic sessions with
`--session-cache off`, `STRIX_SESSION_CACHE=0`, or `--baseline`
(`STRIX_DS41_RTX_BASELINE=1`). DSpark sessions require `--draft-vocab full`;
other vocabulary routes return `session_unsupported`/422 for DSpark sessions.
Target-only sessions accept any launch vocabulary.

The eight explicit-checkpoint qualification phases passed with 8,192-token P
and 64 generated rows: tokens, all logits and complete attention/drafter/frontier
state matched across live, parked and restarted continuations. The public-port
run measured save **210.32 / 255.52 s** and restore **110.42 / 115.18 s** for
target-only / DSpark. The new default-route eviction/restore phases require a
separate run against the rebuilt candidate. See [RTX session cache](rtx-session-cache.md)
for flags, spool operations, format, qualification scope and measured results.
