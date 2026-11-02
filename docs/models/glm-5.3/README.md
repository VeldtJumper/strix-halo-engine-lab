# GLM-5.3 Flash: the USB4 pair and one RTX PRO 6000

[English](README.md) · [Español](README.es.md) · [Main README](../../../README.md)

GLM-5.3 Flash, EXL3 4.05 bpw. On two Strix Halo nodes (TP2 over USB4) it is Production: R1b decode identity by
default, exact MTP self-speculation by default, exact R1 chunked prefill. On one RTX PRO 6000 (Windows) it is an opt-in
draft: exact prefill and greedy decode with the routed experts streamed through the tierstream device-gated cache.
This page keeps every published measurement with its build and conditions and the launch commands. How the tables are
read and what "exact" covers: [methodology](../../methodology.md). Build instructions (trellis library included):
[getting started](../../getting-started/README.md); the pair itself: [two nodes over USB4](../../pair.md).

<a id="glm-pair"></a>

## Run: the pair

Run once per node with `R` set to 0 or 1 and the same tokenized prompt on both:

```sh
./build/strix-glm5-rank MODEL_DIR --world 2 --rank R --transport usb4-auto --prompt PROMPT.u32 --decode 64
```

Default R1b decode uses exact MTP self-speculation and R1 prompt prefill; `--mtp 0` runs plain decode
with the same tokens. `--identity r1` selects a different reference identity. Prefill automatically uses
8192 rows if both ranks admit them, otherwise 4096. See the [resolver](../../../apps/glm_rank_defaults.hpp),
[model families](../model-families.md) and [qualification/performance](../../performance.md).

## Run: one RTX PRO 6000

Windows only, opt-in: build with `pwsh src/glm/rtxpf/producer/build.ps1` against a tierstream source export, build
the unit file once with `glm_unit_file.exe`, then run `decode_qualification_window.py --full-card --allow-gpu`. Every
step, the pinned tierstream commit and the options (`--profile`, `--print-effective-config`, `--vram-total-gib`) are
in the [GLM RTX guide](../../../src/glm/rtxpf/README.md).

<a id="how-fast"></a>

## Measured results: the pair

Sources: [performance records](../../performance.md) and [current GLM defaults/evidence](../../optimizations.md#glm).

EXL3 4.05 bpw, two Strix Halo nodes TP2 over USB4. Production engine. R1b is the default decode identity; prompts use exact R1 chunked prefill;
decode uses exact MTP self-speculation by default (`--mtp 0` = plain decode).
`--identity r1` restores R1 throughout. The historical R1 measurements below keep their original identity.

Prompt prefill in two-rank decode runs defaults to 8192 rows when both ranks admit the calibrated load
and all prefill owners; otherwise both use 4096. This applies to production decode with MTP
self-speculation and plain chunked prefill inside `--decode` runs. Selection uses the load admission's RAM
and GTT headroom, staging, reserves, whole-tensor reads and bounce buffer, plus prefill and communication
growth and the device-free guard.
The largest of the load margin, prefill margin and 8 GiB is retained. Normal two-rank Comm decode reuses
its existing configuration exchange and mismatch vote for mode agreement and the minimum proposed rows.
Fixed modes preserve their payload bytes and tag order; mixed auto/fixed modes refuse on both ranks.
Auto communication pools start at 4096 and grow only after agreement on 8192. A fresh admission
check after growth uses one auto-only minimum vote and shrinks both pools on a 4096 fallback.
Two-rank Comm runs without that handshake (prefill-only, reference gates and batch-only) use 4096 in auto
mode, with source `auto-4096-no-handshake`, without a control exchange. Explicit row counts must match.
`--pf-ttft` and gate harnesses keep 4096 unless `--max-rows 8192` is given explicitly.
Fixed row mismatches there cannot be checked at startup without
changing the existing protocol. External `--xch xfer` auto uses 4096 without an agreement exchange.
Reference prefill (`--prefill ref`, automatic prefill under `r1b-all` or `--idcheck` with decode) uses
4096 in auto mode (`auto-4096-ref`) without a row-count agreement exchange.
The exact prefill route preserves its arithmetic across chunk sizes; measured TTFT on the chunked-prefill
route at 8192 rows improves 4 % at 8K and 116.3 to 109.9 s (5.5 %) at 32K, using about 7.3 GiB more
GPU memory per node.
`STRIX_GLM_MAX_ROWS_AUTO=0` restores fixed 4096; `--baseline` keeps 1024, and explicit `--max-rows` keeps its value.
`--print-effective-config` reports the host-independent candidate and startup policy; stderr reports
admission headroom and the final value with its source. See the [catalog](../../optimizations.md#glm-max-rows-auto).

| Mode | Prompt / context | Measured | Build / exactness |
|---|---|---:|---|
| Greedy decode, R1 | 31-token chat prompt + 64 generated tokens | **45.29 ms/token = 22.1 tok/s**; **51 %** of calculated 23.2 ms floor | `1836c99c`; tokens equal the reference |
| Batch decode, R1 | Eight independent rows, benchmark context unspecified | **122.0 ms/step = 65.5 tok/s aggregate** | `1836c99c`; tokens equal the earlier real-pair reference |
| Exact chunked prefill | 1K / 8K / 32K benchmark prompts, kind unspecified | TTFT **4.268 / 34.70 / 144.75 s** (240 / 236 / 226 tok/s) | `875ca821`, integrated in `41d8cfdc`; both ranks' logits bitwise equal the previous exact prefill |
| R1b-fast, now the default decode identity | Same chat benchmark, 96-token alternating comparison | **41.93–41.96 ms/token**, vs R1 47.33–47.48 (**−11.5 %**, **55 %** of 23.2 ms floor) | `8877de1d`; stable within each identity, not bitwise to R1 |
| MTP self-speculation (default) | Chat / code / JSON prompts, 64–128 generated tokens, rank 0 | **26.98 / 22.46 / 24.98 ms/token = 37.1 / 44.5 / 40.0 tok/s** (1.50× / 1.79× / 1.62× plain 40.3–40.6 ms/token); TTFT 8K **31.13 s** (+2.9 %), 32K **140.9 s** (+8.7 %: MTP cache fill of the extra layer) | Full token vectors and prompt logits equal `--mtp 0` on 7 prompts, both ranks; plain TTFT unchanged |
| Exact chunked prefill, X2 linear core (default) | 1K / 8K / 32K benchmark prompts | TTFT **3.78 / 27.93 / 120.26 s** (271 / 293 / 272 tok/s); same run with `STRIX_GLM_PF_EXACT_X=0`: 1K 3.84 s, 8K 29.59 s (**−5.6 %** at 8K) | logits bitwise equal the X=0 path on both ranks; decode anchors and speculative oracle unchanged |
| Exact chunked prefill, X2 linear core + MoE auto (default) | 1K / 8K / 32K benchmark prompts | TTFT **3.74 / 26.85 / 113.73 s** (274 / 305 / 288 tok/s); same run with `STRIX_GLM_PF_EXACTMOE_V=3`: 1K 3.76 s, 8K 28.17 s (**−4.7 %** at 8K) | outputs bitwise equal the v3 path on both ranks (internal FP32 NaN payloads may differ only in the FP16 overflow case); decode anchors and speculative oracle unchanged |
| Exact chunked prefill, X3 MoE core and row buckets (PR #183) | 1K / 8K benchmark prompts, 8192-row chunks | TTFT **3.18 / 20.32 s**; same run with `STRIX_GLM_PF_EXACTMOE_V=7`: 1K 3.75 s, 8K 25.24 s (**−15 % / −19.5 %**) | two pairs per arm; R1 bitwise (same DOT2 chains, reduce4 and split order) |
| Exact chunked prefill, X3 tail buckets + dense X3 + device MoE plan (PR #215) | 8K / 32K benchmark prompts, 8192-row chunks | TTFT **17.77 / 76.59 s** (461 / 428 tok/s); same run with the previous defaults (PR #183): 20.22 / 86.66 s (**−12.1 % / −11.6 %**) | one pair per arm; per-rank logits equal the public anchors at 8K and 32K in both arms (same DOT2 chains, reduce4 and split order) |
| Exact chunked prefill, attention kernels: MLA scores v5 / o-bar v7 / absorb-out GEMM v2, indexer scores v3 / select v2 (measured on the PR #183 defaults, before PR #215) | 8K / 32K benchmark prompts, 8192-row chunks | 32K TTFT **−6.8 %** (−7.2 s, mean of both run orders: 105.44 → 103.66 s previous kernels first, 110.31 → 97.66 s new kernels first); 32K attention kernels on rank 0 **18.6 → 10.3 s**; 8K **25.23 → 23.62 s (−6.4 %)**, two pairs, without the GEMM v2 | logits bitwise equal the previous kernels on both ranks; same binary, previous kernels via their switches; measured before the X3 MoE core (PR #183) |
| Exact chunked prefill, KDA scan v3 + MLA scores v6 (measured on f7e063b7e; pair prefill code unchanged up to fb66457e4) | 8K / 32K benchmark prompts, 8192-row chunks | 32K TTFT **−1.5 %** (−1.04 s, mean of both run orders: 68.81 → 67.26 s new kernels first, 68.71 → 68.17 s previous kernels first); 32K rank-0 kernels: KDA scan **2.18 → 1.43 s**, MLA scores **2.96 → 2.62 s**; 8K KDA scan alone 16.63 → 16.34 s (−1.7 %) | logits bitwise equal the previous kernels on both ranks; same binary, previous kernels via `STRIX_GLM_PF_KDA_SCAN=1` `STRIX_GLM_PF_SCORES=5`; rank 0 ran under the kernel tracer in every arm |
| R1b teacher-forced agreement | 32 prompts, 256 positions each; chat, code, JSON, math, reasoning, translation | **8,126/8,192 = 99.19 %** top-1; mean KL **~1.8e-3 nats** | `4dad4c19`; free runs diverge on **28/32** prompts within 19–227 tokens |

The R≥3 buffer fix is verified on the real pair. The latest integration uses direct exchange for plain
forwards and exact MoE v3 for prefill; the speculative oracle also passes with drained exchange switching.
Prefill rows are single runs. They replace the earlier synthetic-only result.

R1b changes FP32 summation order and is not bit-identical to R1; which identity is closer to float64 at full depth is
unresolved. HIP graphs are **not implemented**: the latest profile (`41d8cfdc`) keeps the GPU busy **96.3 %** of each forward (was 80.8 %; 909 launches instead of 1,679) and leaves only **2.10 ms idle per forward**
and a calculated recoverable gap of **≤1.8 ms** (a bound, not a measured speedup). Host progress across
exchange boundaries requires segmented capture; device kernels and rank skew take priority.

<a id="glm-rtx-results"></a>

## Measured results: one RTX PRO 6000

**GLM-5.3 Flash on one RTX PRO 6000 (draft, opt-in, Windows).** The trunk, dense layers and shared experts stay
on the card; the 43 x 288 routed experts are served by the [tierstream](https://github.com/hyukudan/tierstream)
device-gated cache, and prefill reads them through the same cache, so the card holds one copy of every expert.
Prefill and decode are exact (bitwise digests of logits and state; prefill state equal across row tiles 2048 / 4096 /
8192). Full card, tile 8192: TTFT **19.7 s** at 8K and **41.7 s** at 32K (exact; the decode-residency restore now runs
after the prefill, outside time-to-first-token, only when decode follows); decode **6.00 / 6.15 tok/s** warm at 1K / 8K (3.75 / 3.70 cold) with the few-row decode kernels on by
default, from 2.43 / 2.36 tok/s with them off in the same service load (full card; host RAM put 1,098 expert units on
the unit file, so per-token file misses are included; 24-token bench). Tokens, every logits digest and the final
session state are bitwise equal on and off, and the 1K/8K qualification passes with them on. The kernels decode each
weight tile straight into tensor-core fragments and keep the pinned GEMMs' accumulation order; turn them off with
`--decode-row-kernels 0` or `STRIX_GLM_RTX_ROW_KERNELS=0`.
With 24 GiB VRAM + 32 GiB host RAM (most experts streamed from the unit file on disk) it also runs exact: TTFT
**64.8 s** at 1K and **76.7 s** at 8K; decode **0.67 tok/s** warm (0.63 cold). Build, unit file and options: [GLM RTX guide](../../../src/glm/rtxpf/README.md).

## Every shipped GLM default

The changelog of every default-on GLM lever with its measured gain, exactness evidence and off switch is in
[measured gains](../../measured-gains.md); the full catalog is [docs/optimizations.md](../../optimizations.md#glm).
GLM RTX build, unit file and options: [GLM RTX guide](../../../src/glm/rtxpf/README.md); native RTX prefill notes:
[docs/models/glm/rtx-prefill-native.md](../glm/rtx-prefill-native.md).
