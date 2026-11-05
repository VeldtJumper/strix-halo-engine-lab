# DeepSeek V4.1 Flash: the USB4 pair, the RTX pipeline split and the standalone RTX

[English](README.md) · [Español](README.es.md) · [Main README](../../../README.md)

DeepSeek V4.1 Flash (1017 checkpoint) runs in three Production configurations: two Strix Halo nodes over USB4
(expert tensor parallel decode, expert parallel device prefill, DSpark speculative decode, batched decode, HTTP server),
the same pair with an RTX PRO 6000 running layers 0-21 over 1 GbE, and the whole model on one RTX PRO 6000 under
Windows. This page keeps every published measurement with its build and conditions, the launch commands and the
profiles. How the tables are read and what "exact" covers: [methodology](../../methodology.md). Build instructions:
[getting started](../../getting-started/README.md); the pair itself: [two nodes over USB4](../../pair.md).

<a id="deepseek-pair"></a>

## Run: the pair

**Weights:** [hyukudan/deepseek-v4.1-flash-gguf-mirror](https://huggingface.co/hyukudan/deepseek-v4.1-flash-gguf-mirror) (the GGUF),
[hyukudan/DeepSeek-V4.1-Flash-1017-strix-dense-U6](https://huggingface.co/hyukudan/DeepSeek-V4.1-Flash-1017-strix-dense-U6) (dense overlay) and
[hyukudan/DeepSeek-V4.1-Flash-1017-strix-trellis-EXL3](https://huggingface.co/hyukudan/DeepSeek-V4.1-Flash-1017-strix-trellis-EXL3)
(trellis packs; production is `ours-K4` plus the `ours-K5` head, as listed in `mixB.txt`). Directory layout:
[reproduce](../../reproduce.md).

**Hardware:** two Strix Halo nodes with 128 GB each, linked by two USB4 cables (one also works), with the
[strix-halo-usb4-driver](https://github.com/hyukudan/strix-halo-usb4-driver) module built for the running kernel.

From a fresh checkout, [`scripts/pair/ds41-pair`](../../pair-setup.md) sets up both nodes (preflight, USB4 driver,
build, verified weights, launch, exact smoke test). By hand: run one rank on each node with matching assets and
environment; set `R` to 0 or 1 locally:

```sh
./build/strix-deepseek41-serve MODEL.gguf --rank R --transport usb4auto
```

The resolver discovers production packs and the drafter, enables TP attention and device prefill,
and discovers matching offline-prepared expert packs. A missing prepared pack uses the original loader
with a notice; execution never prepares packs itself. Bounded agent prefix retention is on by default.
See [production defaults](production-defaults.md),
[one-time expert preparation](prepared-load.md),
[prefix caching](../../engine/serving/prefix-state-cache.md#deepseek-pair-agent-prefix-cache),
[long contexts](long-context.md) and
[hot-expert placement](hot-placement.md).

Every pair app (`rank`, `batch`, `prefill`, `serve`) accepts `--profile default|fast|quality`; run the
same profile on both ranks. `default` is the exact production configuration above and sets nothing. `fast`
adds the relaxed MoE epilogue (`STRIX_DS41_FAST_MOE=1`, bit-changing; its qualification is in progress) and lists
the still-pending FP16 sparse attention in the receipt and in one stderr line. `quality` has no pack for this model and runs `default` with a notice. An explicit environment
variable or option always wins; `--allow-pending` runs pending options whose code is in the build, for
benchmarks only, stamped `unqualified`. `--print-effective-config` lists the profile and every option.

**RTX + pair:** use the [pipeline stage launch contract](pipeline-stages.md).
The RTX runs layers 0–21; the pair runs layers 22–39 and the head. Preserve the declared composed
identity and its assets when comparing references.

<a id="deepseek-standalone-rtx"></a>

## Run: standalone RTX

This is an imported-state runner on Windows WDDM. The model receipt, exported attention state,
matching greedy reference, drafter and tokenized chat file are required external assets:

```powershell
.\build-rtx\strix-cuda-ds41-replica.exe MODEL.gguf `
  --receipt MODEL.receipt.json --import-state STATE.bin --ref GREEDY.log `
  --draft DRAFTER.gguf --draft-device 0 `
  --phases exact0,pressure,exact8,wforce0,spec0,exact0,chats0,schats0,chats0 `
  --steps 64 --max-rows 6 --max-positions 4096 `
  --seqs-file CHATS.tsv --chats 0 --chat-prompt 96 --chat-steps 128 `
  --qual --json RESULTS.jsonl
```

The plain options use the fast resolver, including an automatically sized arena. Imported position,
requested work and safety allowance must fit capacity. Add `--print-config` to inspect without CUDA.
For persistent `--serve SPOOL` execution and required asset preparation, follow the
[standalone command](rtx-standalone.md#run-and-persistent-worker).
Exact conversation sessions are documented in [RTX session cache](rtx-session-cache.md).

Tierstream supports verified partial stripes across multiple disks through `--weight-dirs`;
without a secondary directory it retains the model-file route. Stripe failure falls back to the model
file. Preparation, verification, read eligibility and startup hashing costs are in
[two-drive streaming](rtx-two-drives.md). This is storage configuration,
not a different numerical profile.

On a card or host too small for every expert (for example 24 GB of VRAM and 32 GB of RAM), the replica keeps the
fastest experts in VRAM, the next ones in pinned RAM and reads the rest from the model file (three-tier placement;
`--vram-total-gib` / `--ram-total-gib` size every tier from whole-device caps). File reads run several at a time by default
(`--disk-queue auto`: 4 on one drive, 6 on two, 8 on three or more; `--disk-queue 1` or `--baseline` reads one at a
time). Measured with 4 at a time on one RTX PRO 6000 with whole-device caps `--vram-total-gib 24 --ram-total-gib 32`, experts that fit neither tier read from one NVMe drive, 2 chats x 64 tokens after a 96-token prompt, warm, one binary,
6.55 / 6.75 tok/s plain / DSpark vs 5.45 / 5.37 one at a time (+20 % / +26 %), 16 at a time the same as 4, identical
tokens. Flags and tiers: [standalone RTX](rtx-standalone.md).

At such budgets the DSpark drafter usually costs more than it gives: its windows read more experts from the file than
plain decode. `--drafter auto` (the default) leaves it unloaded when experts will certainly be read from a single drive and
lets its 4.47 GB hold more experts; DSpark requests then run plain decode with the same tokens. On one RTX PRO 6000 with one
NVMe drive: 24+32 GiB 8.2 tok/s without the drafter vs 6-7.8 with DSpark, 48+32 32.2 vs 21.8, 64+32 52.2 vs 51.8; with
expert stripes on two drives DSpark wins at 24+32 (9.3 vs 7.8) and `auto` keeps it while the device expert room is below
18 GiB, as with no file tier (full card: DSpark 105-108 tok/s); at 48+32 on two drives DSpark is slower (about 21 vs 31.5
tok/s), so `auto` runs it without the drafter, and 32+32 on two drives (not measured there) also runs without it.
`--drafter on` / `off` force it.

<a id="how-fast"></a>

## Measured results: the pair

Sources: [published performance](../../performance.md) and [current pair defaults](production-defaults.md).

Pair-only rows: DeepSeek V4.1 Flash (1017 checkpoint), routed experts in the published 2-bit (Q2) GGUF, dense weights in
a 6.5-bit overlay with trellis-coded attention, shared-expert and head weights. 2 nodes, expert tensor parallel
(TP2) over USB4.

| Mode | Hardware | Prompt / context | Concurrency | Best | Typical | Build |
|---|---|---|---:|---:|---:|---|
| Greedy decode | 2 nodes | short decode benchmark, R1 | 1 | **28.333 ms/token** (35.3 tok/s; 59.5 % of calculated ceiling, 16.87 ms floor) | 2 interleaved pairs; tokens identical | `6a5604bf` |
| DSpark speculative decode | 2 nodes | 8 prompts per type, 256 tokens each | 1 | **70.6 tok/s** (code) | 41.9 (English) / 45.3 (Spanish); 50.0 over all 24 | `cab238d6` |
| Batched decode, no drafting | 2 nodes | N8 decode benchmark, all rows equal the reference | 8 | **70.24 ms/step** (113.9 tok/s aggregate, calculated; 66.2 % of calculated ceiling, 46.48 ms floor) | | `6a5604bf` |
| Device prefill (expert parallel) | 2 nodes | 1K benchmark prompt, kind unspecified | 1 | TTFT **1.954 s** | median of 4; logits bit-identical on both ranks | `6a5604bf` |
| Device prefill | 2 nodes | 8K benchmark prompt, kind unspecified | 1 | TTFT **9.757 s** | median of 3; logits bit-identical on both ranks | `6a5604bf` |
| Device prefill | 2 nodes | 16K | 1 | TTFT **20.40 s** (803 tok/s) | | `f9332f3e` |
| Device prefill | 2 nodes | 32K | 1 | TTFT **39.6 s** (828 tok/s), 2 runs | | `43086d05` + prefill defaults |
| Server, one client, chat | 2 nodes | chat prompt | 1 | **66.9 tok/s** (code) | 34.9 tok/s (prose) | release |
| Server, 8 chats × 128 tokens | 2 nodes | chat prompts | 1 / 2 / 4 / 8 | **41.4 / 54.0 / 59.7 / 67.9 tok/s** aggregate | | release |
| Server, 8-category short set | 2 nodes | 36-303-token prompts | 1 / 2 / 4 / 8 | | **39.88 / 61.66 / 66.53 / 74.25 tok/s** aggregate | `cab238d6` |
| Warmed server, 8-category short set | 2 nodes | same short set as above | 1 / 2 / 4 / 8 | **48.00 / 63.97 / 69.20 / 77.00 tok/s** aggregate | 64/64 IDs equal; C8 clean run (see below) | `6a5604bf` |
| Server TTFT | 2 nodes | 8K, warm server idle | 1 | **11.80 s** | **26.40 s** while 4 other streams decode; fresh-server first request 12.72 s | `cab238d6` |

**On by default** in the pair build (`6a5604bf`): shared-down fusion
(`STRIX_DS41_SHARED_DOWN_FUSED=0` to disable), split gate/up MoE prefill kernel "form 7"
(`STRIX_DS41_MOE_PREFILL=0`), exact per-site dense prefill selector (`STRIX_DS41_DENSE_SELECT=0`;
FP16 forms stay off), attention window fold / live chunks / f16 tile
(`STRIX_DS41_ATTN_WINDOW_FOLD=0` / `STRIX_DS41_ATTN_LIVE_CHUNKS=0` / `STRIX_DS41_ATTN_TILE_F16=0`),
and server warm-up + DSpark calibration (`STRIX_DS41_SERVE_WARM=0` / `STRIX_DS41_DSPARK_CALIB=0`).
Measured gains versus `69c6bdab`, with the same GPU turns, Q2 experts + 6.5-bit/trellis dense profile,
and two Strix Halo nodes using expert tensor parallel over USB4:

| Measurement | `69c6bdab` → `6a5604bf` | Change | Verification / conditions |
|---|---:|---:|---|
| Device prefill TTFT, 1K | 2.098 → **1.954 s** | −6.9 % | median of 4 |
| Device prefill TTFT, 8K | 11.090 → **9.757 s** | −12.0 % | median of 3 |
| Greedy decode, R1 | 28.873 → **28.333 ms/token** | −1.9 % | 2 interleaved pairs; tokens identical; 35.3 tok/s, 59.5 % of calculated ceiling (16.87 ms floor) |
| Six-row verify window, R6 | 54.01 → **52.70 ms/window** | −2.4 % | identical |
| Eight streams, no drafting, N8 | 71.84 → **70.24 ms/step** | −2.2 % | all rows equal the reference; 113.9 tok/s aggregate (calculated), 66.2 % of calculated ceiling (46.48 ms floor) |
| Warmed server, 8-category short set, C1 / C2 / C4 / C8 | 43.58 / 61.13 / 66.53 / 75.72 → **48.00 / 63.97 / 69.20 / 77.00 tok/s** aggregate | +9.9 / +3.0 / +2.9 / +1.7 % | token IDs 64/64 equal in every run |
| Streaming, single client | 54.67 → **58.77 tok/s** | +7.4 % | same warmed-server campaign |

Prefill uses benchmark prompts of unspecified kind; logits are bit-identical to the archived reference
on every run on both ranks. Prompt kind/context are unspecified for the short decode benchmarks.
Server: two runs per arm in both orders; one of the two new-build runs had a single prefill unit stall
of 3.8 s (cause under investigation), so C8 uses the clean run. The measured pair build supersedes
the earlier single-client serving regression: it serves faster than `69c6bdab`.

**On by default** (exact, same logits and generated ids): attention lean chunk
kernel (`STRIX_DS41_ATTN_LEAN=0` to disable), split prefill MoE kernels v2 (`STRIX_DS41_PREFILL_MOE_V2=0`), and
Engram lookahead between server prefill pieces (`STRIX_DS41_ENGRAM_LOOKAHEAD=0`). Server TTFT, one client, cache off,
cold English prompts, 2 nodes; the current-default column also includes the later default changes through PR #184:

| Prompt | Before (main #148) | These three levers | Current default (after #184) | Change vs before |
|---|---:|---:|---:|---:|
| 8,399 tokens | 10.44 s | 9.70 s (−7.1 %) | **8.29 s** | −20.6 % |
| 32,499 tokens | 36.85 s | 33.79 s (−8.3 %) | **29.77 s** | −19.2 % |
| 44,699 tokens | 53.62 s | 48.27 s (−10.0 %) | **42.72 s** | −20.3 % |

Logits and the 32 generated ids equal the published references at all three lengths. Per lever (same GPU turn or
separate turns, one load per arm): MoE v2 −5.9 / −6.9 / −8.0 %, Engram lookahead −3.3 % at 32,499 and −2.8 % at
44,699 tokens, attention lean −1.4 to −2.4 %.

Also **on by default**: the deferred decoder cone of non-final prefill pieces (`STRIX_DS41_DEFER_DECODER=0` to
disable). A long prompt's earlier pieces no longer run the decoder layers' cone; the last piece builds everything
decoding needs. Same logits and generated ids. Server TTFT, one client, cache off, cold English prompts, 2 nodes,
default profile, mean of two GPU turns per arm:

| Prompt | Lever off | Now | Change |
|---|---:|---:|---:|
| 8,399 tokens (one piece) | 8.30 s | **8.19 s** | - |
| 32,499 tokens (two pieces) | 29.90 s | **28.92 s** | −3.3 % |
| 44,699 tokens (three pieces) | 43.07 s | **40.51 s** | −5.9 % |

`--profile fast` turns on the relaxed-order MoE lever `STRIX_DS41_FAST_MOE=1`: **7.56 / 26.33 / 36.77 s** on the same
prompts, against 8.36 / 29.38 / 41.55 s for `default` in the same run (−9.6 / −10.4 / −11.5 %; build with the deferred
decoder cone, before the 32-row FAST down below). It changes logits; its quality qualification is in progress, and `default` is unchanged.
Within `--profile fast`, the relaxed MoE down projection now works on 32-row groups by default
(`STRIX_DS41_FAST_MOE_DOWN_ROWS=64` restores the previous groups; same FAST output bits). The 64-row form spilled
registers. In-app pair prefill, four alternating runs: 8.4K 7.863 → 7.533 s (−4.2 %), 32.5K 27.509 → 27.077 s
(−1.6 %); the down kernel alone is 13-25 % faster.
In `default`, dense prefill chunks of 896 rows or more now run o_b's two K-halves in one launch
(`STRIX_DS41_PREFILL_O_B_ONE=0` restores two launches; every non-NaN output bitwise equal). Server TTFT on the pair,
both run orders, same logits and generated ids: 8.4K 8.069 → 7.935 s (−1.7 %), 32.5K 27.41 → 27.18 s (−0.8 %).
The table above predates this change and was not re-measured with it.

<a id="deepseek-rtx-pipeline"></a>

## Measured results: RTX PRO + pair (pipeline split)

**DeepSeek with the RTX PRO, pipeline split** (same model files; the RTX runs layers 0-21 with CUDA, the two nodes run
layers 22-39 and the head):

| Mode | Hardware | Prompt / context | Concurrency | Best | Typical | Build |
|---|---|---|---:|---:|---:|---|
| Batched decode, no drafting | RTX PRO + 2 nodes | 8 benchmark prompts, 48 tokens; 2 groups in flight | 2 / 4 / 8 | **60.1 / 97.5 / 137.2 tok/s** aggregate | bitwise equal to pair-only reference | pair `6525d5e8`, CUDA `463c08ec` line |
| Batched DSpark decode | RTX PRO + 2 nodes | same benchmark, controller windows | 1 / 2 / 4 | **52.4 / 91.3 / 156.8 tok/s** wall aggregate | bitwise reference gates | same line |
| Server, 8-category short set | RTX PRO + 2 nodes | short prompts, DSpark; 2 groups in flight | 1 / 2 / 4 / 8 | **43.2 / 65.8 / 77.9 / 91.7 tok/s** aggregate | token IDs equal pair-only in 32/32 requests | pair `6525d5e8`, CUDA `463c08ec` |
| Prompt reading on the RTX PRO (exact profile) | RTX PRO | 2K / 8K / 32K | 1 | TTFT **3.64 / 4.39 / 12.20 s** | | release |

The batch/serving pipeline overlaps two groups of requests: the RTX processes one while the pair processes
the other (one group for a single request). These batch/serving rows use Q2 experts, the 6.5-bit dense overlay and the separately gated expert-TP
identity; they do not use the single-request trellis-composed profile.

The single-request RTX
split is bitwise equal to the pair without the trellis sites; with trellis sites it is a declared composed
identity, checked against the pair (KL 1.7e-3), not bitwise. RTX TTFT: warm resident service, model load
excluded, up to the exported attention state handed to the import endpoint. A faster CUDA prefill profile fails
the quality gate on long prompts and is not quoted.

<a id="deepseek-standalone-rtx-results"></a>

## Measured results: standalone RTX

**Standalone RTX DeepSeek — Production.** Full DeepSeek V4.1 Flash target on an
RTX PRO 6000 (96 GB, Windows WDDM): Q2 GGUF routed gate/up **IQ2_XXS**, routed down **Q2_K**, dense **Q8_0**.
Experts use an exclusive VRAM + pinned-host cache; the target and DSpark draft share the GPU. The draft keeps
the 64 most-used experts per stage. Deferred/bounded write-backs, the CUDA-graph instantiation fix and the
warm controller default are adopted; keep-64 pruning measured **+16 %** on code with a four-row window.

| Mode | Code | English | Spanish | Calculated ceiling share |
|---|---:|---:|---:|---|
| Exact DSpark decode, warm controller + deferred write-backs, same load | **82.3 tok/s** | **87.7 tok/s** | **94.4 tok/s** | **51–58 %** of the 6.18 ms/token all-hit floor |
| Exact DSpark decode, best runs across loads | **94–95 tok/s** | **97 tok/s** | **100–101 tok/s** | **58–63 %** of the same ceiling |

Quote code throughput as **82–95 tok/s**: loads vary by up to **~10 %**, so comparisons apply within one load.
Exact gates pass for this RTX profile; equality against the pair's trellis profile is not claimed.
Prompt lengths and generated-token counts are unspecified for these latest decode measurements.

| Loading / serving work | Measured result | Status |
|---|---|---|
| Pipelined expert loading | Full load **108.7 → 61.8 s**; expert fill **83.1 → 38.0 s**, **2.25 → 6.73 GB/s**, disk-bound | **Published, on by default** ([PR #52](https://github.com/hyukudan/strix-halo-engine-lab/pull/52)); bitwise-identical placement and outputs |
| Persistent loaded worker | **8–18 s per request** on test workloads without reload; fresh-load cache reset **3.4 s** | **Published** (`--serve`, PR #52); each request bitwise identical to a fresh process |
| Window prefetch | No repeatable gain in matched warm comparisons | Exact; default off |

**Cold start, same standalone RTX profile.** Persistent worker,
tested binary `1569bce5`; runtime write-back/flush switches preserve fresh-process tokens, logits and state
(6/6 offline comparisons). These are step timings, not chat throughput or TTFT:

| Configuration | Step 0 | Mean steps 0–15 | Mean steps 16–63 |
|---|---:|---:|---:|
| Fresh load | 332.6 ms | 56.05 ms | 28.15 ms |
| Fresh load with saved residency seed | 156.0 ms | 44.67 ms | 25.66 ms |
| Worker with graph pre-capture | 11.0 ms | 33.99 ms | 27.26 ms |
| Worker, warm expert cache | 10.9 ms | 11.63 ms | 11.49 ms |

Graph pre-capture adds **1.5 s before readiness**. Prompt kind/length are unspecified for this exact-phase
benchmark. The worker, residency seed and graph pre-capture are published with
[PR #52](https://github.com/hyukudan/strix-halo-engine-lab/pull/52): the adopted exact settings (keep-64, deferred
write-backs, warm controller, decode kernel bundle and fusions) are on by default with off switches, and a no-flag run
matched the explicit configuration (all gates 64/64, identical speculation traces, 21/21 chat tokens) on a reduced
pinned cache, so that gate does not give speed numbers. The measured **82–95 tok/s** code range remains the quoted result.
Build, run and defaults: [docs/models/deepseek-v4.1/rtx-standalone.md](rtx-standalone.md).

## Every shipped DeepSeek default

The changelog of every default-on DeepSeek lever (pair and RTX) with its measured gain, exactness evidence and off
switch is in [measured gains](../../measured-gains.md); the full catalog is
[docs/optimizations.md](../../optimizations.md#deepseek-two-node) and
[docs/optimizations.md#deepseek-rtx-pro-standalone](../../optimizations.md#deepseek-rtx-pro-standalone). Pair
production defaults: [production-defaults.md](production-defaults.md). Serving:
[serving](../../serving.md).
