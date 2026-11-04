# MiniMax M3.1 Flash on the USB4 pair

[English](README.md) · [Español](README.es.md) · [Main README](../../../README.md)

MiniMax M3.1 Flash, EXL3 trellis weights at 3.06 bpw, on two Strix Halo nodes (TP2 over USB4): exact prefill and
decode, DSpark serving with automatic admission, MSA prefill mode 6, deadline-based and nested serving. Production;
the eight-text quality gate passed. This page keeps every published measurement with its build and conditions and the
launch command. How the tables are read and what "exact" covers: [methodology](../../methodology.md). Build
instructions (trellis library included): [getting started](../../getting-started/README.md); the pair itself:
[two nodes over USB4](../../pair.md).

<a id="minimax-pair"></a>

## Run

Run once per node with `R` set to 0 or 1 and matching EXL3 assets:

```sh
./build/strix-minimax-rank MODEL_DIR R usb4-auto OUTPUT_DIR
```

The [resolver](../../../src/minimax/engine/mm_defaults.cpp) enables the production decode/prefill/exchange set,
including MSA prefill mode 6. `strix-minimax-serve` uses structural DSpark auto admission with compatible
local assets; eligible uncontended requests may draft, while competing/grouped or unsupported requests
stay plain. See [admission and qualification](../../engine/serving/dspark-auto-qualification.md).
A named session request that does not fit the context returns HTTP 400 `context_length_exceeded` (the
session stays parked); generation stops at the end of the context with finish reason `length`. See
[session cache](../../minimax/session-cache.md).

<a id="how-fast"></a>

## Measured results

Sources: [performance records](../../performance.md) and [current optimization evidence](../../optimizations.md#minimax).

All rows: EXL3 trellis weights at 3.06 bpw, two Strix Halo nodes TP2 over USB4, greedy reference.
The engine is Production; the qualified optimization sets below are public and on by default.
The [grouped host prefill scheduling changes](../../minimax-grouped-prefill.md) default on for
grouped calls whose every member has at least 2048 rows. Balanced 7940-token prompts measured
2.3–2.6% lower prefill time with identical complete token IDs; unbalanced small tails use the ordinary path.

| Mode | Prompt / context | Measured result | Build / exactness |
|---|---|---|---|
| Greedy decode | 1K / 8K benchmark prompts, cold | **43.30 / 45.06 ms/token** (23.10 / 22.19 tok/s); 44.57 / 46.10 ms/token on `8cc30ac0` (256 tokens) | public tree of PR #59; ranks agree, reference IDs equal |
| Combined prefill | 8K benchmark prompt | **14.04 s**, 583 tok/s (alternating arms in one process, 13.85–14.29 s); **45–50 %** of the calculated 6.3–7.0 s kernel floor; 17.731 s on `e0bad3d1` (baseline 21.874 s) | #334 `c225cdaa2`; both ranks' state/final rows equal the references |
| Fresh combined prefill | 32K benchmark prompt | **63.48 s**, 516 tok/s (three runs per arm, both orders); **41–46 %** of the calculated 26–29 s kernel floor; 91.508 s on `e0bad3d1` | #334 `c225cdaa2`; result hash equal in all six runs, ranks agree |
| Combined prefill, same process | 1K / 8K, alternating arms | **2.49–2.52 / 13.85–14.29 s**; on `561de319`: 2.73–2.77 / 20.3–20.8 s, 0.848–0.853× / 0.783–0.797× baseline time | #334 build; state/final-row checks equal on both ranks |
| Batch, fresh process per arm | N=1 / N=8, 1K benchmark | **43.92 / 130.24 ms/step** (N8: **61.4 tok/s aggregate**, calculated) | `8cc30ac0`; each row equals running alone; batch invariance passes |
| DSpark | JSON / Python / prose | **59.96 / 36.93 / 27.05 tok/s**; 54.4 / 34.1 / 24.1 tok/s on `8cc30ac0` | PR #70 `fbd43423`; IDs equal greedy |
| Served, C1 / C2 / C4 / C8 | 8 distinct ~190-token prompts, 256 tokens | **22.0 / 32.2 / 44.5 / 55.7 tok/s** aggregate (repeat run 22.1 / 32.5 / 45.1 / 56.1) | main `e6ce78571`; every request's IDs equal the request alone |

Prompt kinds are unspecified for the prefill and batch benchmarks. Fresh-prefill rows are single runs;
the 1K first-use result was **3.509 s**. The combined run later hit its time limit in another arm; only completed
measurements are quoted. New prefill defaults in `561de319` include tiled dense projections, batched attention/router
and exact-fit MoE row tiles. They preserve the production identity.

Exchange-boundary defaults in `8cc30ac0` are qualified at 1K/8K/16K (16K synthetic repetition): matched
replays save **4.20 / 2.82 / 3.09 ms/token**. In fresh-process batch A/B, N1/N8 fall from **48.22 / 134.03**
to **43.92 / 130.24 ms/step**; the earlier N8 slowdown was traced to GPU thermal/power throttling under
sustained load (about 96 °C, shader clock −20 % within ~100 s), not to the engine. The rows above are
cold-start numbers from fresh processes; under sustained serving, measured decode is about 5 % and prefill
about 15 % slower on this cooling (29 sequential requests: 44.9 → 47.2 ms/step). This does not justify a general serving speed claim. Later selective MoE/attention defaults are listed in the [catalog](../../optimizations.md#minimax).

**Quality gate : passed** the pre-registered eight-text evaluation, including two independent,
byte-identical TP2 runs and all prompt/regime checks. Mean KL to the exact reference is **0.1449** versus
**0.1425** for the production CPU arithmetic. Top-1 agreement is **0.47 percentage points lower** than the CPU
reference, within the pre-registered 2-point margin; next-token accuracy differs by **+0.01 point**. This
qualification is conditional on those eight texts; it supersedes the earlier two-prompt failure.

## Every shipped MiniMax default

The changelog of every default-on MiniMax lever with its measured gain, exactness evidence and off switch is in
[measured gains](../../measured-gains.md); the full catalog is [docs/optimizations.md](../../optimizations.md#minimax).
Related notes: [grouped host prefill scheduling](../../minimax-grouped-prefill.md), [MSA prefill](../../minimax-msa.md),
[session cache](../../minimax/session-cache.md), [DSpark admission and qualification](../../engine/serving/dspark-auto-qualification.md).
Serving: [serving](../../serving.md).
