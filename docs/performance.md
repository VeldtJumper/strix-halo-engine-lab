# Measured performance

For current benchmark commands, model requirements, reference verdicts and machine interfaces, see [Reproduce engine measurements](reproduce.md).

Every speed number quoted in the [README](../README.md) is listed here with its date, configuration and
exactness check. All numbers are measured in the lab; values are medians of 3 or more runs unless a row
says otherwise. The overnight snapshot states counts where recorded; unspecified counts must not be assumed to be medians.
Estimates and calculations are not listed as results.

The README shows, per mode, the **best** measured result and a **typical** one. Additional measured builds
are collected [below](#additional-measured-builds), with build IDs. The overnight snapshot and README
supersede older rows only for the named build and workload.

Hardware names used below:

- **Node:** one AMD Strix Halo mini PC (Ryzen AI Max+ 395, Radeon 8060S `gfx1151`, 128 GB unified memory), Linux,
  ROCm.
- **Pair:** two nodes joined by two direct USB4 cables (one 40 Gb/s, one 20 Gb/s link) using the
  [USB4 driver series](https://github.com/hyukudan/strix-halo-usb4-driver).
- **Workstation:** a PC with an NVIDIA RTX PRO 6000 Blackwell (96 GB), reaching the pair over 1 GbE.

## Overnight snapshot (2026-10-25/26)

All six engines are **Production**: Qwen, DeepSeek on the pair, standalone RTX, RTX + pair, MiniMax and GLM.
New optimizations require separate qualification and publication. Measurements below supersede historical
rows only for the named build and workload. Raw receipts and review correspondence remain outside Git.

| Engine / profile | Hardware and workload | Latest measured result | Verification / publication |
|---|---|---|---|
| DeepSeek V4.1 Flash 1017, Q2 experts + 6.5-bit/trellis dense | Two nodes expert TP2 over USB4, short R1 / six-row verify / N8 benchmarks, 2026-10-26 | **28.333 ms/token / 52.70 ms/window / 70.24 ms/step**; N8 **113.9 tok/s** aggregate (calculated); R1 / N8 **59.5 / 66.2 %** of calculated ceilings | `6a5604bf`; published pair default; tokens / verify window / batch rows equal the reference |
| DeepSeek standalone RTX, Q2 GGUF + Q8_0 dense | RTX PRO 6000 (96 GB, Windows WDDM), exclusive VRAM + pinned-host cache, DSpark keep-64, code / English / Spanish, 2026-10-25 | Same-load **82.3 / 87.7 / 94.4 tok/s**; **51–58 %** of the ceiling for the 6.18 ms all-hit floor; code **82–95 tok/s** across loads | Exact for this profile; prompt lengths/counts unspecified; default-binary integration gate pending |
| Qwen3.8-Flash-Next, UD-Q4_K_XL, dense A8 | One node, exact prefill sweep, 2026-10-26 | **3.00–240.62 s** across 4K–262,143 tokens; **40.5–45.3 %** of route-matched contract ceilings | `8b99cf58` shipped in PR #49; HC v3 band 16 shipped in #50; later measurements below |
| Qwen MTP decode, UD-Q4_K_XL, dense A8 | One node, code 1K / 8K, auto windows, 256 generated tokens; `702bc074`, 2026-10-25 | **59.59 / 54.00 tok/s**; 1K **69 %** of 86.8 tok/s byte ceiling | Recipe golden equal; prose 8K **49.58 tok/s** on `226bb691`, exact |
| DeepSeek pipeline, expert-TP identity | RTX PRO + pair over 1 GbE, two groups; pair `6525d5e8`, CUDA `463c08ec` line | Batch N2/4/8 **60.1 / 97.5 / 137.2 tok/s**; DSpark N1/2/4 **52.4 / 91.3 / 156.8 tok/s** wall aggregate (2026-10-24); server C1/2/4/8 **43.2 / 65.8 / 77.9 / 91.7 tok/s** (2026-10-25) | Batch: 8 prompts × 48 tokens, bitwise gates; server: 8-category short set, 32/32 IDs equal pair-only |
| MiniMax M3.1 Flash, EXL3 3.06 bpw | Two nodes TP2 over USB4, 1K / 8K greedy benchmark, 2026-10-26 | **44.57 / 46.10 ms/token**; fresh 8K prefill **17.731 s (462 tok/s)** | Exact archived-reference checks at 1K/8K; new defaults pending publication |
| GLM-5.3 Flash, EXL3 4.05 bpw, R1 | Two nodes TP2 over USB4, chat decode / R8 batch / exact 8K prefill | **45.29 ms/token (22.1 tok/s)**, **51 %** of 23.2 ms floor; batch **122.0 ms/step = 65.5 tok/s** aggregate (calculated); exact 8K prefill **17.77 s** with the current defaults (20.32 s after PR #183, earlier 34.70 s on `875ca821`) | R1 reference equal; latest integration `41d8cfdc`, pending publication |

Standalone RTX best runs across loads (2026-10-25): code **94–95**, English **97**, Spanish **100–101 tok/s**,
**58–63 %** of the 6.18 ms all-hit ceiling. Loads vary by up to **~10 %**; compare within one load.
Keep-64 draft pruning measured **+16 %** on code with a four-row window. Deferred/bounded write-backs,
CUDA-graph instantiation fixes and the warm controller were adopted in the measured line.

| RTX loading / serving work | Measured result | Date | Status / exactness |
|---|---|---|---|
| Pipelined expert loading | Full load **108.7 → 61.8 s**; expert fill **83.1 → 38.0 s**, **2.25 → 6.73 GB/s**, disk-bound | 2026-10-25 | Published, on by default (PR #52); placement and outputs bitwise identical |
| Persistent loaded worker | R1 requests **13.9–18.3 s**, R2 **7.9–8.2 s**, no reload; fresh-load cache reset **~3.4 s** | 2026-10-25 | Published (`--serve`, PR #52); fresh-process tokens/logits/state equal; request lengths unspecified |
| Window prefetch | No repeatable gain in matched warm comparisons | 2026-10-26 | Exact; default off |

**Cold start, same standalone RTX profile (2026-10-26).** Persistent worker,
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
[PR #52](https://github.com/hyukudan/strix-halo-engine-lab/pull/52) (2026-10-26): the adopted exact settings (keep-64, deferred
write-backs, warm controller, decode kernel bundle and fusions) are on by default with off switches, and a no-flag run
matched the explicit configuration (all gates 64/64, identical speculation traces, 21/21 chat tokens) on a reduced
pinned cache, so that gate does not give speed numbers. The measured **82–95 tok/s** code range remains the quoted result.
Build, run and defaults: [docs/models/deepseek-v4.1/rtx-standalone.md](../docs/models/deepseek-v4.1/rtx-standalone.md).

The newer RTX kernel set is exact but does not establish an additional chat-throughput gain:

| Kernel comparison, 2026-10-26 | Conditions | Measured effect | Status |
|---|---|---|---|
| Compact single-row Q8 kernel | RTX PRO, eight-layer smoke, two warm-up occurrences, two matched comparisons; `2fdf0eb2` | **−0.80 / −0.53 %** device time | Local default on; full-model gain not established |
| Deferred records + router form 2 + single-split query/KV projection | Full-model all-hit R1, two load pairs; `4b770fd8` | **10.182 / 10.189 → 10.137 / 10.163 ms**, mean **−0.35 %** | Exact gates; host variation prevents a chat-throughput conclusion |

### Qwen prefill and defaults

**Exact combined prefill (2026-10-26).** One Strix Halo node, Qwen3.8-Flash-Next UD-Q4_K_XL,
dense A8 profile; build `8b99cf58`, defaults without environment overrides. Same-node comparisons
against `4a44683b`, except the maximum-context same-build off comparison. Terminal FP32 logits and tokens
match the reference at every length; code-1K and prose-8K qualification receipts match the archived baseline.
Prompt kind is unspecified for the length sweep.

| Prompt tokens | Baseline TTFT | Default TTFT | Change | Default tok/s | Contract ceiling share |
|---|---:|---:|---:|---:|---:|
| 4,096 | 3.53 s | **3.00 s** | −15.0 % | 1,363 | 43.4 % of 3,139 tok/s |
| 17,700 | 14.97 s | **12.81 s** | −14.4 % | 1,382 | 44.7 % of 3,089 tok/s |
| 32,768 | 25.95 s | **23.49 s** | −9.5 % | 1,395 | 45.3 % of 3,078 tok/s |
| 65,536 | 55.37 s | **48.18 s** | −13.0 % | 1,360 | 44.9 % of 3,030 tok/s |
| 131,072 | 120.96 s | **105.84 s** | −12.5 % | 1,238 | 42.6 % of 2,907 tok/s |
| 262,143 | 291.82 s (same-build off) | **240.62 s** | −17.5 % | 1,089 | 40.5 % of 2,687 tok/s |

Two runs per arm at 4K / 17,700; 32K has one baseline and two default runs; longer rows are single runs.
Prompt tiles: 4,096 at 4K / 128K, 8,192 at 17,700 / 32K / 64K, 2,048 with 256/16-row tails at 262,143.
Run-to-run and node-to-node spread reaches ~5 %; single comparisons are indicative.

[PR #49](https://github.com/hyukudan/strix-halo-engine-lab/pull/49) ships combined prefill, GDN gates batching/core16,
and router selection/activation staging as defaults. [PR #50](https://github.com/hyukudan/strix-halo-engine-lab/pull/50)
adds HC output v3 with band 16 (`5011b842`): 32K TTFT **−0.52 s (−2.2 %)** in three alternating pairs,
4K **−54 ms (−1.8 %)** in two pairs; logits and tokens remain equal. A later single 32K run on that build
measured **22.03 s, 1,487 tok/s (48.3 % of the 3,078 tok/s contract ceiling)**.
`STRIX_QPF2K_DEFAULTS=0` restores the earlier prefill path; HC-specific off switches are
`STRIX_QPF2K_HC_OUTPUT3=0` / `STRIX_QPF2K_HC_OUTPUT3_BAND=0`.

**QSA stage batching — on by default (2026-10-26, [PR #51](https://github.com/hyukudan/strix-halo-engine-lab/pull/51)).**
Exact 32K TTFT **22.21 / 22.66 / 22.76 → 21.57 / 22.32 / 22.58 s** (three pairs, mean **−0.39 s, −1.7 %**);
on the published build, 32K in four ABBA pairs **−0.17 s** on average (every pair faster) and 64K **46.76 → 46.01 s (−1.6 %)**;
17,700 **12.54 → 12.46 s** (one pair). Tokens, terminal logits and the code-1K / prose-8K qualification receipts match.
`STRIX_QPF2K_QSA_MASK=0x57` restores the previous stages (default `0x2257`; `0x257` keeps everything except the attend occupancy variant).
These timings are separate comparisons, not a new full-length sweep.

Exact MTP draft attention and the router defaults are enabled in the public Qwen line.
Batched MTP drafting remains opt-in. GSQ below is a separate speed-over-quality profile.

### DeepSeek pair optimization status

**On by default since 2026-10-26** in the pair build (`6a5604bf`): shared-down fusion
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
of 3.8 s (cause under investigation), so C8 uses the clean run. The 2026-10-26 pair build supersedes
the earlier single-client serving regression: it serves faster than `69c6bdab`.

The shared gate/up bandwidth redesign remains off: on 2026-10-26, a cold-rotation native microbenchmark
(R1, 24 weight sets, three rounds) measured a co-producer chain at **74.54 µs** versus **73.91 µs** for the
existing declared pair. Device results match; host/device NaN-payload selection remains a separate contract
question. This is not an end-to-end speedup.

### MiniMax latest measurements

All rows: EXL3 trellis weights at 3.06 bpw, two Strix Halo nodes TP2 over USB4, greedy reference.
The engine is Production; the newer optimization sets below are **in development, pending publication**.

| Mode | Prompt / context | Measured result | Date | Build / exactness |
|---|---|---|---|---|
| Greedy decode | 1K / 8K benchmark prompts, 256 tokens | **44.57 / 46.10 ms/token** (22.44 / 21.69 tok/s) | 2026-10-26 | `8cc30ac0`; ranks agree, reference IDs equal |
| Fresh combined prefill | 8K benchmark prompt | **17.731 s**, 462 tok/s; baseline 21.874 s (**−19 %**); **36–39 %** of the calculated 6.3–7.0 s kernel floor | 2026-10-26 | `e0bad3d1`; both ranks' state/final logits equal archived production |
| Fresh combined prefill | 32K benchmark prompt | **91.508 s**, 358 tok/s; **28–32 %** of the calculated 26–29 s kernel floor | 2026-10-26 | `e0bad3d1`; new reference, not an archived-reference exactness comparison |
| Combined prefill, same process | 1K / 8K, three alternating pairs | **2.73–2.77 / 20.3–20.8 s**, **0.848–0.853× / 0.783–0.797×** baseline time | 2026-10-26 | `561de319`; state/final-row checks and tokens equal on both ranks |
| Batch, fresh process per arm | N=1 / N=8, 1K benchmark | **43.92 / 130.24 ms/step** (N8: **61.4 tok/s aggregate**, calculated) | 2026-10-26 | `8cc30ac0`; each row equals running alone; batch invariance passes |
| DSpark | JSON / Python / prose, 256 tokens | **54.4 / 34.1 / 24.1 tok/s** | 2026-10-26 | `8cc30ac0`; IDs equal greedy |

Prompt kinds are unspecified for the prefill and batch benchmarks. Fresh-prefill rows are single runs;
the 1K first-use result was **3.509 s**. The combined run later hit its time limit in another arm; only completed
measurements are quoted. New prefill defaults in `561de319` include tiled dense projections, batched attention/router
and exact-fit MoE row tiles. They preserve the production identity.

Exchange-boundary defaults in `8cc30ac0` are qualified at 1K/8K/16K (16K synthetic repetition): matched
replays save **4.20 / 2.82 / 3.09 ms/token**. In fresh-process batch A/B, N1/N8 fall from **48.22 / 134.03**
to **43.92 / 130.24 ms/step**; the earlier N8 slowdown was traced to GPU thermal/power throttling under
sustained load (about 96 °C, shader clock −20 % within ~100 s), not to the engine. The rows above are
cold-start numbers from fresh processes; under sustained serving, measured decode is about 5 % and prefill
about 15 % slower on this cooling (29 sequential requests: 44.9 → 47.2 ms/step). This does not justify a general serving speed claim. Further MoE/attention variants remain off.

MiniMax's eight-text quality gate passed on 2026-10-25: mean KL to the exact reference **0.1449** versus CPU
production arithmetic **0.1425**; top-1 agreement difference **−0.47 percentage points**, inside the
pre-registered **2-point** margin; next-token accuracy difference **+0.01 point**. Both independent TP2 runs
are byte-identical. Qualification is conditional on those eight texts; the older two-prompt failure below is historical.

### GLM latest measurements

EXL3 4.05 bpw, two Strix Halo nodes TP2 over USB4. Production engine; newer builds below are
**in development, pending publication**. R1b is the default decode identity; `--identity r1` restores R1.

| Mode | Prompt / context | Measured | Date | Build / exactness |
|---|---|---:|---|---|
| Greedy decode, R1 | 31-token chat prompt + 64 generated tokens | **45.29 ms/token = 22.1 tok/s**; **51 %** of calculated 23.2 ms floor | 2026-10-26 | `1836c99c`; tokens equal the reference |
| Batch decode, R1 | Eight independent rows, benchmark context unspecified | **122.0 ms/step** = 65.5 tok/s aggregate (calculated) | 2026-10-26 | `1836c99c`; tokens equal the earlier real-pair reference |
| Exact chunked prefill | 1K / 8K / 32K benchmark prompts, kind unspecified | TTFT **4.268 / 34.70 / 144.75 s** (240 / 236 / 226 tok/s) | 2026-10-26 | `875ca821`, integrated in `41d8cfdc`; both ranks' logits bitwise equal the previous exact prefill |
| Exact chunked prefill, X3 MoE core and row buckets (PR #183) | 1K / 8K benchmark prompts, 8192-row chunks | TTFT **3.18 / 20.32 s**; same run with `STRIX_GLM_PF_EXACTMOE_V=7`: 3.75 / 25.24 s (**−15 % / −19.5 %**) | — | PR #183; two pairs per arm; R1 bitwise (same DOT2 chains, reduce4 and split order) |
| Exact chunked prefill, X3 tail buckets + dense X3 + device MoE plan (PR #215) | 8K / 32K benchmark prompts, 8192-row chunks | TTFT **17.77 / 76.59 s** (461 / 428 tok/s); same run with the PR #183 defaults: 20.22 / 86.66 s (**−12.1 % / −11.6 %**) | — | one pair per arm; per-rank logits equal the public anchors at 8K and 32K (same DOT2 chains, reduce4 and split order) |
| Exact chunked prefill, attention kernels (MLA scores v5 / o-bar v7 / absorb-out GEMM v2, indexer scores v3 / select v2; measured on the PR #183 defaults, before PR #215) | 8K / 32K benchmark prompts, 8192-row chunks | 32K TTFT **−6.8 %** (mean of both run orders: 105.44 → 103.66 s and 110.31 → 97.66 s); rank-0 attention kernels 18.6 → 10.3 s; 8K **25.23 → 23.62 s (−6.4 %)** | — | logits bitwise equal the previous kernels on both ranks; measured before PR #183 |
| Exact chunked prefill, KDA scan v3 + MLA scores v6 (measured on f7e063b7e; pair prefill code unchanged up to fb66457e4) | 8K / 32K benchmark prompts, 8192-row chunks | 32K TTFT **−1.5 %** (mean of both run orders: 68.81 → 67.26 s and 68.71 → 68.17 s); rank-0 KDA scan 2.18 → 1.43 s, MLA scores 2.96 → 2.62 s; 8K KDA scan alone **16.63 → 16.34 s (−1.7 %)** | — | logits bitwise equal the previous kernels on both ranks; rank 0 under the kernel tracer |
| Exact chunked prefill, KDA scan v5 (on the PR #254 defaults) | 8K benchmark prompt, 8192-row chunks | 8K TTFT **16.60 vs 16.60 s** (mean of 8 arms, both orders: neutral); rank-0 scan 0.355 → 0.340 s; scan −3.7 % per layer | — | logits bitwise equal v3 on both ranks; rank 0 under the kernel tracer |
| R1b-fast, now the default decode identity | Same chat benchmark, 96-token alternating comparison | **41.93–41.96 ms/token**, vs R1 47.33–47.48 (**−11.5 %**, **55 %** of 23.2 ms floor) | 2026-10-26 | `8877de1d`; stable within each identity, not bitwise to R1 |
| R1b teacher-forced agreement | 32 prompts, 256 positions each; chat, code, JSON, math, reasoning, translation | **8,126/8,192 = 99.19 %** top-1; mean KL **~1.8e-3 nats** | 2026-10-26 | `4dad4c19`; free runs diverge on **28/32** prompts within 19–227 tokens |

The R≥3 buffer fix is verified on the real pair. The latest integration uses direct exchange for plain
forwards and exact MoE v3 for prefill; the speculative oracle also passes with drained exchange switching.
Prefill rows are single runs. They replace the earlier synthetic-only result.

R1b changes FP32 summation order and is the default decode identity (`--identity r1` restores R1); which identity is closer to float64 at full depth is
unresolved. HIP graphs are **not implemented**: the latest profile (`41d8cfdc`) keeps the GPU busy **96.3 %** of each forward (was 80.8 %; 909 launches instead of 1,679) and leaves only **2.10 ms idle per forward**
and a calculated recoverable gap of **≤1.8 ms** (a bound, not a measured speedup). Host progress across
exchange boundaries requires segmented capture; device kernels and rank skew take priority.

## Qwen3.8-Flash-Next

Model: unsloth UD-Q4_K_XL GGUF (111.33 GB) plus a locally distilled Q8_0 MTP draft sidecar (2.79 GB). Every row
uses MTP speculative decoding, greedy (temperature 0), thinking off, dense A8 numerical profile.

### One node

| What | Value | Date | Conditions | Exactness |
|---|---|---|---|---|
| Decode, code, 1K-token prompt | **59.9 tok/s** | 2026-10-22 | 256 tokens, MTP windows 2/4/8, production decode set | golden replay error 0 |
| Decode, prose, 1K-token prompt | **55.5 tok/s** | 2026-10-22 | same | same |
| Decode, prose, 8K-token prompt | 48.9 tok/s | 2026-10-22 | same | same |
| Decode, code, 32K-token prompt | 51.8 tok/s | 2026-10-22 | same | same |
| Prefill / time to first token, 1K | 916 tok/s / **1.13 s** | 2026-10-22 | fast prompt profile, 8192-row tiles | byte-identical to the serial route within the profile |
| Prefill / time to first token, 4K | 1,085 tok/s / 3.78 s | 2026-10-22 | same | same |
| Prefill / time to first token, 16K | 1,089 tok/s / 15.05 s | 2026-10-22 | same | same |
| Prefill / time to first token, 32K | 1,065 tok/s / 30.76 s | 2026-10-22 | same, `--unqualified-context` | same |

These are historical 2026-10-22 rows; the overnight snapshot gives the newer exact-prefill measurements.

### Long context (YaRN, one node)

Measured on one Strix Halo node, default route (code `a199c8e0` plus the bundle selection of #137), diverse
real prose (50 public-domain books in six languages), greedy decoding, automatic storage ladder (no storage
overrides). TTFT runs from process launch to the first token and includes model load.
The 262K prefill cell is from public main `669bcb83` (262,144-token prompt, prefill time without model load,
`--profile default` and `--profile fast` interleaved, two pairs: 179.07 s and 174.18 s); the 262K decode cells are from
build `7b2453fe`; the row's other cells are from the build above.

| Context (YaRN factor) | Exact gate (MTP == serial == pool off) | TTFT serial / MTP | Prefill, serial | Decode serial / MTP | MTP acceptance |
|---|---|---|---|---|---|
| 262,400 (2) | Full scope | 219 s / 224 s | **1,464 tok/s** `--profile default`; 1,505 tok/s `--profile fast` (`669bcb83`) | 20.55 / **59.5 tok/s** (2.9x, `7b2453fe`) | 75 % |
| 524,288 (2) | Target scope | 470 s / 508 s | 1,215 tok/s | 19.7 / **60.6 tok/s** (3.1x) | 85 % |
| 1,048,576 (4) | Full scope, plus the default-path receipt | 1,005 s / 1,096 s | 1,081 tok/s (MTP route about 970) | 18.7 / **55.9 tok/s** (3.0x) | 83 % |

These are exactness checks against serial YaRN, not native rotary; they do not establish general model
fidelity at these lengths. See the [Qwen long-context section](models/qwen3.8/README.md#qwen-long-context).

### Two replicas (one complete model per node, behind the router)

Setup: the [TensorMeld](https://github.com/hyukudan/tensormeld) router balancing active decodes; servers with
`--http-batch 8`, T 1024, a prefix-cache entry every 1024 tokens, per-request controller reset (D-057) and joint
windows (D-058); both servers restarted before each of 3 rounds, no warm-up; median of 3 rounds; 2026-10-22.
Token ids equal the single-node run in **570/570** responses. Evidence: [docs/decisions/decision-register.md D-060](decisions/decision-register.md#d-060-two-full-replicas-one-per-node-behind-the-router).

Two ways to count, both reported:

- **Sum of per-request decode:** completion tokens / decode time of each request, summed over the concurrent
  requests (what the hardware sustains while they all decode).
- **Wall aggregate:** completion tokens / wall time from release to the last response, including time to first
  token and queueing (what a client sees).

| Workload | Clients | Sum of per-request decode | Wall aggregate | TTFT p50 |
|---|---:|---:|---:|---:|
| Coding chat (~70-token prompt, 187-token answers) | 1 | **90.7** | 68.1 | 0.76 s |
| Coding chat | 2 | **181.5** | 136.2 | 0.76 s |
| Coding chat | 4 | **201.6** | 146.9 | 1.25 s |
| Coding chat | 8 | **255.0** | 166.7 | 2.37 s |
| Code, 1,024-token prompt, 256-token answers | 1 | 59.9 | 46.1 | 1.27 s |
| Code, 1,024-token prompt | 2 | 120.0 | 92.5 | 1.26 s |
| Code, 1,024-token prompt | 4 | 142.1 | 111.9 | 2.17 s |
| 8-category short set (36-303-token prompts, 82-200-token answers) | 1 | 58.2 | **42.8** | 0.89 s |
| 8-category short set | 2 | 116.6 | **85.8** | 0.85 s |
| 8-category short set | 4 | 142.3 | **102.9** | 1.32 s |
| 8-category short set | 6 | 159.4 | **110.6** | 1.91 s |
| 8-category short set | 8 | 177.5 | **120.4** | 2.55 s |

*(campaign)*: per-replica serving at 1 and 4 clients is part of the running campaign.

## DeepSeek V4.1 Flash

Model: DeepSeek V4.1 Flash (1017 checkpoint). Routed experts: the published 2-bit (Q2) GGUF. Dense weights: a
6.5-bit packed overlay with trellis-coded (EXL3/QTIP-style, [strix-trellis](https://github.com/hyukudan/strix-trellis))
shared-expert and head weights ("mix-B"). Production defaults: [production-defaults.md](models/deepseek-v4.1/production-defaults.md),
[docs/decisions/decision-register.md D-061 to D-064](decisions/decision-register.md#d-061-deepseek-v41-two-node-production-defaults).

Exactness on the pair: every gate run is bitwise. Greedy and DSpark windows equal the stored references on both
ranks, batch tokens equal the campaign reference, and the server's token ids equal the reference in 64/64 requests.

### The pair

| What | Value | Date | Conditions |
|---|---|---|---|
| Greedy decode, 1 stream, R1 | **28.333 ms/token = 35.3 tok/s**; 59.5 % of calculated ceiling (16.87 ms floor) | 2026-10-26 | `6a5604bf`; 2 interleaved pairs; tokens identical; prompt kind/context unspecified |
| DSpark speculative decode, 1 stream, 8 English / 8 Spanish / 8 code prompts × 256 tokens | **37.5 / 39.3 / 62.3 tok/s**, 46.4 over all 24 | 2026-10-22 | DSpark drafter on both ranks (P17.1) |
| Batched decode, 8 streams, no drafting, N8 | **70.24 ms/step = 113.9 tok/s** aggregate (calculated); 66.2 % of calculated ceiling (46.48 ms floor) | 2026-10-26 | `6a5604bf`; all rows equal the reference; prompt kind/context unspecified |
| Device prefill, 8K tokens | TTFT **9.757 s** | 2026-10-26 | `6a5604bf`; median of 3; benchmark prompts, kind unspecified; logits bit-identical on both ranks |
| Device prefill, 1K tokens | TTFT **1.954 s** | 2026-10-26 | `6a5604bf`; median of 4; benchmark prompts, kind unspecified; logits bit-identical on both ranks |
| Device prefill, 1K / 4K / 16K tokens | 458 / 640 / 758 tok/s | 2026-10-22 | P17.1 build, before fused staging and final KV by owner |
| Warmed server, 8-category short set, aggregate at 1 / 2 / 4 / 8 clients | **48.00 / 63.97 / 69.20 / 77.00 tok/s** | 2026-10-26 | `6a5604bf`; same short set; 64/64 token IDs equal; C8 clean run (see optimization status) |
| Server, 8-category short set, aggregate at 1 / 2 / 4 / 8 clients *(campaign)* | **38.4 / 52.7 / 59.3 / 66.8 tok/s** | 2026-10-22 | `strix-deepseek41-serve`, 8 slots, DSpark (drafting paused above 4 requests in flight), P17.4 |
| Server, 8 chats × 128 tokens at 1 / 2 / 4 / 8 clients | 41.4 / 54.0 / 59.7 / 67.9 tok/s | 2026-10-22 | same |
| Server, one client, prose / code chat, decode after the first token | 34.9 / **66.9 tok/s** (31.8 / 55.4 including time to first token; TTFT p50 0.50 s) | 2026-10-22 | P17.1 build |
| Server time to first token, 8K-token prompt, after 40 s idle *(campaign)* | **15.8 s** | 2026-10-22 | P17.5; 27.9 s while 4 other streams decode, longest decode stall 940 ms |

Measured effect of each production default (A/B inside the timing campaign, bitwise): segmented batch steps
N=8 +9.1 %; DSpark in the batch app +82 % at N=1, +53 % at N=2; DSpark in the server +62 / +52 / +19 % at
1 / 2 / 4 clients; eager small sends greedy +2.5 %; final KV by owner 8K prefill +1.8 % and idle 8K TTFT −9 %.

### The pair plus the RTX PRO 6000 (opt-in)

| What | Pair only | Pair + RTX PRO | Date | Conditions |
|---|---|---|---|---|
| DSpark decode, all 24 prompts | 46.4 tok/s | **61.8 tok/s (×1.33)** | 2026-10-22 | RTX runs layers [0, 22) with CUDA, the pair runs [22, 40) and the head; one hidden state crosses 1 GbE per step; P17.1 pair build |
| DSpark decode, English / Spanish / code | 37.5 / 39.3 / 62.3 | **50.3 / 54.2 / 81.1** | same | same |
| Greedy decode | 27.7 | 38.3 | same | pipeline fallback, one stream |

Exactness: without the trellis sites the split is bitwise equal to the pair-only run (61.7 tok/s, 24/24 prompts,
6,144/6,144 tokens). With them (the 61.8 row) the RTX layers sum the trellis shared expert in IEEE order: a
declared composed identity, checked against the pair (KL mean 1.7e-3, top-1 127/128), not bitwise. Evidence:
[pipeline-stages.md](models/deepseek-v4.1/pipeline-stages.md), [cuda-trellis-shexp.md](models/deepseek-v4.1/cuda-trellis-shexp.md),
[docs/decisions/decision-register.md D-064](decisions/decision-register.md#d-064-a-discrete-gpu-in-the-deepseek-v41-decode-rtx-split-opt-in).

**RTX PRO prompt reading, exact profile** (resident `strix-cuda-ds41-prefill-service`, 73.3 GB of experts pinned
in host RAM and the rest read from NVMe, warm, model load excluded, median of 3; the exported attention state's
SHA equals the exact reference; handoff measured to a local import endpoint), 2026-10-22:

| Prompt | Prefill time | Prefill tok/s | Time to first token including state export |
|---|---:|---:|---:|
| 2K | 3.39 s | 605 | **3.64 s** |
| 8K | 4.04 s | 2,029 | **4.39 s** |
| 32K | 11.55 s | 2,838 | **12.20 s** |

A faster CUDA prefill profile exists but fails the quality gate at long prompts; it is experimental and none of its
numbers are quoted as results.

**Hot-expert placement** (all dense weights and ~80 GB of the hottest experts on the RTX PRO, a 10 GiB expert
cache, the rest from `strix-deepseek41-expert-server` on the pair over 1 GbE): 40.1 tok/s greedy against 27.2
pair-only, 256/256 tokens equal; measured on an earlier pair build (2026-10-21). Evidence:
[hot-placement.md](models/deepseek-v4.1/hot-placement.md).

### Dense format quality

Trellis mix-B dense weights (all K4, head K5) against the 6.5-bit overlay, KL to the original-precision model over
103,908 tokens, computed in an emulation of the formats (not with the engine's kernels): mix-B 0.0446 at 2.27 GB per
token per node, 6.5-bit 0.0447 at 3.63 GB; paired ΔKL −0.0001.

<a id="after-this-release-pre-release-builds"></a>

## Additional measured builds

Historical builds measured through 2026-10-25; each hash identifies the measured build.
The overnight snapshot above records newer results without qualifying unmeasured combinations.
Same hardware, greedy, every run exact against its reference before its time is counted.

### DeepSeek V4.1 Flash, pair

| What | Value | Date | Build | Conditions and exactness |
|---|---|---|---|---|
| Greedy decode, 1 stream | **28.8 ms/token = 34.7 tok/s** (production: 34.8 ms/token) | 2026-10-25 | `940a4b85` + trellis library release candidate | interleaved with the previous candidate, 3 clean runs; every run exact against its reference on both ranks |
| Verify window of 6 rows | 54.77 ms/window (production 63.30) | 2026-10-25 | `940a4b85` | same confirmation; verify windows including rejected rows are bit-equal to one-row decode on both ranks |
| Batched decode, 8 streams | **73.93 ms/step = 108.2 tok/s** aggregate (production 85.64 ms/step) | 2026-10-24 | `cab238d6` | no drafting, 8 prompts × 48 tokens; 3 interleaved runs per arm; tokens identical |
| Device prefill, 1K | TTFT **2.18 s** (469 tok/s; production 2.21 s) | 2026-10-24 | `cab238d6` | interleaved A/B ×3; logits identical on both ranks and between builds |
| Device prefill, 8K | TTFT **10.78 s** (760 tok/s) | 2026-10-24 | `f9332f3e` | production 11.21 s in the same session; logits identical on both ranks and between builds |
| Device prefill, 16K | TTFT **20.40 s** (803 tok/s; production 21.36 s) | 2026-10-24 | `f9332f3e` | A/B ×3 in the same session; logits identical |
| Device prefill, 32K | TTFT **39.55 s** (828 tok/s), 2 runs | 2026-10-23 | `43086d05` with the three prefill defaults of the next release | defaults campaign; base 41.37 s; logits identical |
| DSpark speculative decode, 1 stream | **49.97 tok/s** over all 24; code 70.57 / English 41.94 / Spanish 45.33 (production 45.11 over all 24; code 62.62 / English 38.68 / Spanish 40.34) | 2026-10-24 | `cab238d6` | 8 English / 8 Spanish / 8 code prompts × 256 tokens; 3 interleaved runs per arm; every speculative run equals plain greedy on 24/24 prompts |
| Server time to first token, 8K prompt while 4 other streams decode | **26.40 s** (production 28.18 s) | 2026-10-24 | `cab238d6` | 3 interleaved runs per arm; ids identical idle vs loaded |

On `cab238d6` (2026-10-24), the short-set server wall aggregate at C1 / C2 / C4 / C8 is
**39.88 / 61.66 / 66.53 / 74.25 tok/s**. Warm idle 8K TTFT is **11.80 s**; a fresh server's first request is
**12.72 s**, versus **26.40 s** while four streams decode. Interleaved runs retain reference token IDs.

### Qwen3.8-Flash-Next

| What | Value | Date | Build | Conditions and exactness |
|---|---|---|---|---|
| TTFT one node, 1K / 4K / 16K / 32K | 1.14 / **3.72 / 14.46 / 29.60 s** (production build, same session: 1.15 / 3.83 / 15.24 / 31.24 s) | 2026-10-24 | `0a646b90` with the pipelined dense and reworked MoE prefill schedules (defaults of the next release) | fast prompt profile, fresh process per run, A/B ×4 in ABBA/BAAB order, median of 4; prefix state and logits identical to the release build |
| Copy windows, long copy-edit: document fix / patch / C++ rewrite | **5.451 / 6.102 / 6.449 ms/token = 183.5 / 163.9 / 155.1 tok/s** (default MTP: 10.76 / 10.87 / 10.87 ms/token) | 2026-10-24 | `e1b115a2` | production fast preset, `--mtp-ngram-draft=wide --mtp-copy-wide-rows 32`, A/B ×3, tokens identical; wide-window gate (tokens, full-vocabulary logits of every row and state equal to serial) PASS |
| Copy windows, 7 long copy-edit tasks (token-weighted) | **7.987 ms/token = 125.2 tok/s** (default MTP 11.086 = 90.2 tok/s) | 2026-10-24 | `dd65dbd5` | 1,114-2,047-token answers; A/B ×3; 78/78 samples identical to the default. JSON with an edit in every record: neutral (+0.2 %) |
| Copy windows, 6 coding-agent sessions | 16.708 ms/token = 59.9 tok/s (default 16.714) | 2026-10-24 | `dd65dbd5` | neutral (−0.04 %), tokens identical |
| Build `0117a140`, one node: TTFT 1K / 4K / 16K / 32K | **1.12 / 3.73 / 14.64 / 30.17 s** (release build, same session: 1.15 / 3.86 / 15.42 / 31.79 s; −2.3 / −3.4 / −5.1 / −5.1 %, ranges disjoint) | 2026-10-24 | `0117a140` | integration of the prefill, decode-stall and attention work with its new defaults, no extra settings; fast prompt profile, fresh process per run, interleaved A/B, median of 4; prefix state and logits identical to the release reference with the new defaults and with each of them turned off |
| Build `0117a140`, one node: decode 1K code / 1K prose / 8K prose / 32K code | **60.0 / 55.7** / 48.1 / 47.9 tok/s (release, same session: 59.8 / 55.5 / 48.1 / 48.2); prompt reading 3.7-4.5 % faster | 2026-10-24 | `0117a140` | 256 tokens, median of 4; 1K ranges disjoint; 8K and 32K vary run to run in both builds (32K ranges 44.5-52.6 and 44.3-52.6 tok/s, same tokens per cycle); tokens, state and final logits identical to the release |
| Build `0117a140`, two replicas: coding chat wall throughput at concurrency 1 / 2 / 4 / 8 | 68.2 / 136.4 / **146.8 / 158.1 tok/s** (release, same session: 67.7 / 136.2 / 141.3 / 152.7); TTFT p50 at concurrency 4: 1.28 s (1.37 s) | 2026-10-24 | `0117a140` | `--http-batch 8`, router, fresh servers per run with one excluded warm-up request, 3 runs per build in ABBAAB order, median; 90/90 responses token-identical to the single-node reference. At concurrency 8 the per-request TTFT median is 2.51 s against 2.46 s: it falls on the step between the second and third prompt each node reads in turn, and one run per build was slowed by queueing; the best runs of the two builds are the same |
| RTX PRO prefill offload, TTFT 1K / 4K / 8K / 16K / 32K | **1.36 / 1.92 / 2.94 / 4.95 / 8.99 s** (same server alone: 1.80 / 5.20 / 10.46 / 21.43 / 44.02 s) | 2026-10-24 | `70b93af3` | exact CUDA prefill (A8 profile) reproduces the node's prefill bit for bit (61 state components); KV and recurrent state streamed to the node per layer over 1 GbE with a lossless codec on 2 connections; 64 generated tokens identical to the node-only run at every length; one run per length |

### MiniMax M3.1 Flash

Production engine: EXL3 trellis weights at 3.06 bits per weight, two nodes, tensor parallel over USB4.
The following rows retain their historical build; the overnight snapshot records newer measurements.

| What | Value | Date | Build | Conditions and exactness |
|---|---|---|---|---|
| Greedy decode, 1K prompt | **48.44 ms/token = 20.64 tok/s**; chat prompts 20.66 tok/s | 2026-10-24 | `3ba49a3d` | all-defaults bench, 256 tokens ×3; ranks agree, token ids checked |
| Greedy decode, 8K prompt | **50.80 ms/token = 19.68 tok/s** | 2026-10-24 | `3ba49a3d` | all-defaults bench; ranks agree, token ids checked |
| Prefill / time to first token, 1K / 8K | **5.20 / 37.73 s** (217 tok/s) | 2026-10-24 | `3ba49a3d` | all-defaults bench; ranks agree, token ids checked |
| Batched decode, N = 1 / 2 / 4 / 8 | **20.75 / 30.52 / 42.09 / 57.59 tok/s** aggregate | 2026-10-24 | `3ba49a3d` | all-defaults bench; each row identical to running alone |
| DSpark decode, JSON table, 256 tokens | **52.03 tok/s** (×2.52) | 2026-10-24 | `3ba49a3d` | bundled K4 drafter; ids equal plain greedy on the same build (20.6 tok/s) |
| DSpark decode, Python code / prose, 256 tokens | Python **32.73 tok/s** (×1.59); English prose 22.68 tok/s (×1.10); on the wider set, DSpark is roughly at or below plain speed on prose (0.88-0.95×), and 1.8-2.2× on code and JSON | 2026-10-24; wider set 2026-10-25 | `3ba49a3d` | bundled K4 drafter; ids equal plain greedy in all three initial prompts and on the wider set of 16 prompts: 2 each in 6 languages plus code and JSON |
| Historical quality gate (superseded by the 2026-10-25 eight-text pass above) | prompt B: 7/7 pass; prompt A: 6/7, top-1 agreement with the exact reference 71.7 % against 73.8 % required | 2026-10-24 | pair build | top-1 against the real text within 1 point of the CPU reference; **did not pass this earlier gate** |

## The USB4 link

Measured with the [driver series](https://github.com/hyukudan/strix-halo-usb4-driver) (four host streams over the
two cables, 2026-10-22): about **37 Gb/s per direction** of common-window throughput with both directions busy
(the sum of the four per-stream rates is 39.8 Gb/s per direction; the two are different measures). A 20 KB
exchange takes 18.4 µs p50 back to back (raw), 35.5 µs through the engine's host path.
