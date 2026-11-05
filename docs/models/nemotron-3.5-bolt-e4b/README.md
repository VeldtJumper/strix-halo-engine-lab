# Nemotron 3.5 Bolt E4B (gemma4 family) on one Strix Halo node

[English](README.md) · [Español](README.es.md) · [Main README](../../../README.md)

NVIDIA Nemotron 3.5 Bolt E4B is a post-trained Gemma 4 E4B (`model_type: gemma4_text`): 42 layers, hidden 2560,
GeGLU FFN 10240, 35 sliding-window layers (window 512, head dim 256) and 7 global layers (head dim 512, proportional
partial RoPE), per-layer embeddings (PLE), shared KV in the last 18 layers, logit softcap 30, tied head, vocabulary
262144, 64K context. The engine runs it natively as the `gemma4` family (`STRIX_FAMILY_GEMMA4=ON`, opt-in build).
This page keeps the profiles, assets, launch commands, every published measurement with its conditions, and the off
switches. How the tables are read and what "exact" covers: [methodology](../../methodology.md).

<a id="profiles"></a>

## Profiles

One switch selects everything: `--profile default|fast|quality` (default: `default`). An explicit option or
environment variable always wins over the profile. Every receipt records the profile, each lever's value and source,
and any pending row. A pending (not yet qualified) row is never applied: the profile runs every ready row, the pending
row keeps the exact behaviour, the receipt lists it with its declared target, and one stderr line says so. A profile
refuses only for a missing required asset (`quality` without `--exl3-arm`) or an explicit request for something this
build does not contain.

| Profile | Decode | Prefill | Guarantee |
|---|---|---|---|
| `default` | `g4-v1` (FP32 activations, Q4_K_M weights) + MTP drafter at C1 | Declared: `g4-wmma-pf-v2` on Q4_K_M GGUF packs (other pack types: `prefill-exact`); qualification against the `default` bars pending, `prefill-exact` runs until then and the receipt lists `pending g4-wmma-pf-v2 (exact runs)`. `STRIX_G4_PREFILL_IDENTITY=exact` keeps exact prefill | Decode stays exact: output bytes equal the declared serial path for every batch size (C4 == C1) and with the drafter on or off; the `g4-wmma-pf-v2` prompt prefill runs under `default` only once its receipt against the `default` bars is named (not bitwise vs exact prefill) |
| `fast` | `g4-a8-v1` (int8 activations, qualified) + MTP drafter; decode attention at depth `g4-dattn-f2` (int8 KV copies for the global layers): qualification pending, exact attention runs until then | `a8` prompt prefill: `g4-a8-v1` linears on 512-row chunks; prompt rows equal the a8 decode of the same rows, so the `g4-a8-v1` qualification covers them (not bitwise vs exact prefill; `STRIX_G4_PREFILL_IDENTITY=exact` restores exact prefill) | Named profile, qualified against frozen fidelity bands |
| `quality` | EXL3 pack (`--exl3-arm DIR`), identity chosen once per pack and session (`g4-exl3-v1`/`v3`) | `prefill-exact` | Exact within its EXL3 identity; higher fidelity than Q4_K_M at similar size |

The identity of the EXL3 route is fixed at startup from the pack and `--streams`, never per step, so a session's
output does not depend on load. `STRIX_G4_EXL3_IDENTITY=v1|v2|v3` forces it.

## Assets

- Model: the BF16 safetensors directory (tokenizer, chat template) and, for `default`/`fast`, the Q4_K_M GGUF.
- Rotary tables, generated once offline (see `src/gemma4/ref/README.md`):
  `python src/gemma4/ref/make_rope_tables.py rotary-tables --positions 65536 --workers 6`
- Drafter: an MTP assistant package in `<model dir>/drafter-mtp/` (config, weights, LICENSE, NOTICE, LINEAGE,
  MANIFEST) is found automatically; the published drafter is [hyukudan/Nemotron-3.5-Bolt-E4B-MTP-Drafter](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-MTP-Drafter). `--draft-mtp DIR` points elsewhere; `--draft off` or `STRIX_G4_DRAFT=0`
  unloads the drafter (prompt lookup, below, still drafts); `STRIX_G4_LOOKUP=0` turns prompt lookup off and
  `--dflash off` turns all drafting off.
- EXL3 packs (`quality`): one directory per pack arm, loaded with `--exl3-arm DIR` next to the BF16 model; the
  published pack is [hyukudan/Nemotron-3.5-Bolt-E4B-EXL3](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-EXL3) (M4).

## Build and run

Build with `-DSTRIX_FAMILY_GEMMA4=ON` (default OFF) and `ninja -C build strix-gemma4-serve`. Generate the rotary
tables once, then serve on one node:

```sh
python src/gemma4/ref/make_rope_tables.py rotary-tables --positions 65536 --workers 6
./build/strix-gemma4-serve MODEL_DIR --rope-tables rotary-tables --tokenizer MODEL_DIR/tokenizer.json --profile default
```

`--profile fast` selects int8 activations; `--profile quality` needs `--exl3-arm DIR`. An MTP drafter package in
`MODEL_DIR/drafter-mtp/` is used automatically (`--draft off` disables it). Download the published packs:

```sh
hf download hyukudan/Nemotron-3.5-Bolt-E4B-MTP-Drafter --local-dir MODEL_DIR/drafter-mtp
hf download hyukudan/Nemotron-3.5-Bolt-E4B-EXL3 --local-dir EXL3_DIR
./build/strix-gemma4-serve MODEL_DIR --rope-tables rotary-tables --tokenizer MODEL_DIR/tokenizer.json --profile quality --exl3-arm EXL3_DIR
```

Without the drafter folder, decode runs exactly as without a drafter. See the
[model guide](README.md).

With `MODEL/drafter-mtp/` present, single-request decode drafts automatically; output is identical to drafting off.
With more than one active request the drafter is admitted only where its measured cost pays off.
Single-request decode also drafts by prompt lookup (the continuation of the longest earlier match of the last tokens
in the prompt and output), verified token by token, so output is identical to drafting off. Beside the drafter a
per-request selector uses it only on a strong match (equal to the drafter alone within noise); without a drafter,
agent-style prompts that copy from their context gain most (C1 `default`: 115.6 vs 48.0 tok/s). `STRIX_G4_LOOKUP=0`
turns it off.
`--print-effective-config` (decode) prints the resolved profile and switches without loading a request.

<a id="how-fast"></a>

## Measured results

One Strix Halo node (gfx1151, 128 GB), official Q4_K_M GGUF for `default` and `fast`, EXL3 pack for `quality`;
256-token English chat decode, greedy. C4 and C8 are aggregate tok/s across concurrent requests. Unless a cell is
marked, every number in this section comes from one measuring session of one engine build, and each llama.cpp value
was measured in the same turn as the row it is compared with.

| Profile | C1 shipped ([MTP drafter](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-MTP-Drafter) + prompt lookup) | C1, drafting off | C4 | C8 | Prefill 512 / 2K / 8K |
|---|---:|---:|---:|---:|---:|
| `default` (`g4-v1`) | **91.9** | 55.8 | 169.8 | 237.9 | 532 / 535 / 509 exact¹ |
| `fast` (`g4-a8-v1`, qualified) | **99.5** | 54.0 | 206.5 | 340.6 | 551 / 538 / 509 exact; `a8` prefill: see below |
| `quality` ([EXL3 M4 pack](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-EXL3), `g4-exl3-v3`) | **96.4** | 50.2 | 184.8 | 303.0 | 794 / 815 / 746 |
| llama.cpp `74a4fb00`, same Q4_K_M | — | 55.5 | 193.3 | 279.9 | 2,077 / 2,062 / 1,881 |

C1 shipped is the configuration a user gets by default: the MTP drafter found in `MODEL_DIR/drafter-mtp/` plus prompt
lookup, measured as three interleaved rounds against drafting off with token ids and the committed state equal in
every round. C1 drafting off, C4 and C8 are plain decode (no drafting). The prefill cells are exact prefill except
where marked. ¹ `default` prefill identity `g4-wmma-pf-v2` on Q4_K_M (pending qualification; until then the exact
prefill runs, as measured here; `STRIX_G4_PREFILL_IDENTITY=exact` keeps it). Prefill per identity is in the WMMA
prefill table below.

Compared with the previously published table (same machine): `quality` C1 shipped 83.8 -> **96.4** tok/s, plain
decode 44.4 / 153.1 / 258.3 -> **50.2 / 184.8 / 303.0** (C1 / C4 / C8); exact prefill at 2K tokens 466 / 458 / 769
-> **535 / 538 / 815** tok/s (`default` / `fast` / `quality`); `default` and `fast` decode are unchanged
within run-to-run noise.

At one request the engine verifies only the drafted tokens the drafter is confident about (a cumulative-probability
window, threshold stored per drafter), which lifted C1 with the drafter from 85.1 / 92.2 / 72.1 to 92.9 / 100.6 / 83.8
tok/s (`default` / `fast` / `quality`, measured when it shipped); `STRIX_G4_MTP_CONF=0` restores the previous window
policy.

At long contexts each drafted depth costs more relative to a target pass, so the window follows the stream's own
measured costs: when a draft depth costs at least 0.04 of a one-row pass it drafts one depth at a time, extends while the
cumulative probability is >= 0.6 and verifies the prefix at >= 0.15. Decode at 16K / 32K tokens (`default`, 512 tokens,
mean of two runs per arm): +0.3 / +1.6 % code, +0.8 / +1.2 % English, +3.1 / +2.4 % Spanish, +0.6 / +3.5 % Japanese,
token ids unchanged; code and Japanese at 16K are within the run-to-run spread;
`STRIX_G4_MTP_CONF_LONG=0` keeps the short-context window at every length.

Decode passes return the device's greedy token per row and copy logits and hidden-state taps to the host only when a
request or drafter reads them: +1.9 / +4.6 / +5.9 % at C1 / C4 / C8 with the drafter off and +3.3 / +8.3 / +9.3 % with
the MTP drafter on (`default`, short prompts, same bytes); `STRIX_G4_DECODE_EXPORT=0` restores the full copies.

The MTP drafter's attention loads each target KV head once for both query heads that share it, and its cluster top-k
is one bitonic sort: GPU ms per draft depth 0.423 / 0.746 / 2.617 → 0.346 / 0.578 / 1.750 at 512 / 4K / 32K context
(q8_0 drafter, top-64, HIP graphs, one node), draft ids bit-identical; `STRIX_G4_MTP_KERNELS=legacy` restores the
previous kernels.

Single-request decode also drafts from the text itself (prompt lookup): it proposes the continuation of the longest
earlier match of the last tokens in the prompt and output, and the main model verifies every token, so token ids equal
drafting off. It pays off where the output copies from the context. Agent code-edit prompt (a 2.2K-token Python file
plus an edit request, `default`, 256 tokens, token ids equal to drafting off in every arm):

| Agent code-edit prompt, C1 | tok/s |
|---|---:|
| Drafting off | 48.0 |
| Prompt lookup alone (no drafter) | **115.6** |
| MTP drafter alone (`STRIX_G4_LOOKUP=0`) | 149.0 |
| Shipped: MTP drafter + prompt lookup | **149.1** |

Beside the MTP drafter a per-request selector drafts a lookup window only on a strong match and backs off on misses,
so it stays out of the way: equal to the drafter alone within noise (English chat 91.9 vs 91.9, agent edit
149.1 vs 149.0). Without a drafter it gains strongly; in the session that shipped it, a JSON tool prompt
(1.2K tokens) ran at 122.9 vs 49.3 tok/s, a 29.5K-token document at 35.0 vs 28.1 and code at 55.8 vs 51.9.
`STRIX_G4_LOOKUP=0` turns it off; `--dflash off` turns all drafting off.

The HTTP server batches concurrent chat requests: up to eight requests (`STRIX_G4_SERVE_SLOTS`) share one scheduler
and decode in the same target pass, new requests join between steps, and every request's token ids equal the same
request served alone (verified for `default`, `fast` and `quality` with eight to eleven overlapping requests: greedy
chat in four languages, a long document, seeded sampling, thinking-budget continuation, streaming and a client that
disconnects mid-stream). One request at a time is unchanged (C1 with the drafter, same ids and speed).
`STRIX_G4_SERVE_BATCH=0` restores one chat job at a time. When `--streams` is not given, the session-fixed EXL3 core
identity is chosen for the slot count (eight), so an M5 pack selects `g4-exl3-v3` instead of `v1`; M4 and M6 always
use `v3`.

| `default`, streaming, 128 tokens per request | 1 client | 4 clients | 8 clients |
|---|---:|---:|---:|
| Mixed prompts (chat, code, one ~1.3K-token document): batching on, aggregate tok/s | 72.6 | **76.7** | **76.1** |
| Same, one request at a time (`STRIX_G4_SERVE_BATCH=0`) | — | 65.4 | 66.9 |
| Same, time to first token p50, on / off | 0.1 s | **0.3** / 5.1 s | **0.6** / 13.8 s |
| llama.cpp `74a4fb00` server (`-np 8`, same Q4_K_M, same turn) | 52.5 | 123.0 | 227.1 |
| Short chat prompts, 256 tokens: batching on / llama.cpp server | — | — | **160.4** / 244.1 |

The table above was measured before the next two serving changes (prompt chunks while others decode and the prompt prefix cache).
Later, on main `d7069831d` (short chat prompts, 128 tokens, prefix cache off, `default`): C1 / C4 / C8
**92.1 / 143.8 / 187.4 tok/s** aggregate, time to first token p50 0.10 / 0.19 / 0.37 s.

While other requests decode, a prompt is processed in chunks of at most 64 rows (`STRIX_G4_SERVE_PREFILL_ROWS`,
`0` = whole 512-row chunks), so a long prompt pauses them for shorter stretches: with the mixed prompts above the
longest pause between two tokens of a request went from 1.05-1.12 s to 0.41-0.42 s and the 99th percentile from
0.44 / 1.08 s to 0.17 / 0.20 s (4 / 8 clients); throughput and token ids are unchanged. `fast` (`a8` prompt rows equal
its decode rows) shortens chunks the same way; with `STRIX_G4_PREFILL_IDENTITY=exact` under `fast` (exact prompt
processing beside the int8-activation decode) prompt chunks keep their full size, since where a chunk ends decides
which prompt rows a decode pass writes. A WMMA prompt route (`g4-wmma-pf-v2` under `default` once qualified, or `=v1` /
`=v2`) keeps full-size chunks the same way; `default` with exact prefill shortens them.

The server also keeps the committed state at the end of each prompt and of each finished request and reuses it
when a later prompt starts with exactly those tokens, typically the next turn of a conversation, which re-sends the
conversation so far: only the new rows are processed. Turn-2 time to first token (`default`, six conversations in
parallel): 4K-token document **0.20 s** vs 12.59 s without the cache, 1K-token document **0.19 s** vs 3.48 s, chat
0.39-0.51 s vs 1.31-1.66 s; both turns' token ids equal the cache off. Up to 16 states within 4 GB
(`STRIX_G4_SERVE_PREFIX_MB`); `STRIX_G4_SERVE_PREFIX=0` turns it off. `fast` reuses any state, as `default` with exact prefill does (its
`a8` prompt rows equal its decode rows); with `STRIX_G4_PREFILL_IDENTITY=exact` under `fast`, or a WMMA prompt route
(`g4-wmma-pf-v2` under `default` once qualified, or `=v1` / `=v2`), only states at a 512-row prompt boundary written by
prompt processing are reused, never an answer. Two different conversations that share a long
document but diverge after it do not share it yet.

**MTP drafter** - [hyukudan/Nemotron-3.5-Bolt-E4B-MTP-Drafter](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-MTP-Drafter). A small assistant model (Google's
Gemma-4-E4B-it MTP assistant, 78.8M parameters) proposes the next few tokens and the main model verifies them, so the
output is identical to decoding without it; only the speed changes. The speed-up comes from the MTP assistant
architecture and the engine's exact speculative decoding: C1 decode goes from 51.8 to 91.9 tok/s in English
(`default`, drafting off and on interleaved in the same turn, x1.78); [Google's untuned assistant](https://huggingface.co/google/gemma-4-E4B-it-assistant) is about as fast in
this engine (see the drafter comparison in this section). We lightly fine-tuned it on about 6.3M tokens that Nemotron 3.5 Bolt E4B
generated from 17,865 prompts in eight languages plus code, tool calls and reasoning; on a held-out set of 697 prompts
that adds about 1.5 % expected tokens per cycle (English 3.61 -> 3.65, Japanese 2.52 -> 2.60); with the confidence window it is
+1.4 % faster than the untuned assistant (geometric mean over English, Spanish, Japanese and code). Download it into `MODEL_DIR/drafter-mtp/`; the engine uses it automatically.

**EXL3 M4 pack (`quality` profile)** - [hyukudan/Nemotron-3.5-Bolt-E4B-EXL3](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-EXL3). An EXL3 (trellis-coded,
scaled-Hessian) quantization of the BF16 model: MLP at 4 bits, attention q/o at 5, k/v and the output head at 6,
embedding and per-layer tables kept in BF16 - 4.65 bits per weight across the quantized linears, a 9.7 GB download
(2.5 GB of quantized weights) that reads 2.70 GB of weights per token (about 14 % less than Q4_K_M). On a held-out
40-document, 8-language set its output is closer to the original BF16 model than the standard Q4_K_M: top-1 agreement
92.94 % vs 87.0 %, mean KL 0.0367 vs 0.141. Pick it (`--profile quality --exl3-arm DIR`) when you want output closer
to BF16; stay on `default` or `fast` with Q4_K_M for the highest speed. `quality` decodes at 96.4 tok/s shipped
(50.2 without drafting, token ids unchanged) and prefills at 815 tok/s at 2K tokens; its prefill and decode
are still being optimised.

**Compared with NVIDIA's official releases.** NVIDIA publishes the main model as [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16) and as one
quantization, the [Q4_K_M GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB), plus a [DFlash drafter](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash). Measured cells only; an empty cell was not
measured. Same machine, greedy, 256-token chat decode, tok/s.

| Model (engine profile) | Download GB | Weight bytes read per token | Top-1 vs BF16 | Mean KL vs BF16 | C1 shipped | C1 plain | C4 | C8 | Prefill 2K | Prefill 8K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Official BF16, reference ¹ | 16.06 | 9.29 GB | 100 % | 0 | 38.1 | 19.8 | 72.4 | 156.7 | 565 | 534 |
| Official Q4_K_M, `default` | 6.18 | 3.13 GB | 87.0 % | 0.141 | 91.9 | 55.8 | 169.8 | 237.9 | 535 | 509 |
| Official Q4_K_M, `fast` | 6.18 | 3.13 GB | 86.7 % | 0.150 | 99.5 | 54.0 | 206.5 | 340.6 | 538 | 509 |
| **EXL3 M4**, `quality` | 9.71 | **2.70 GB** | **92.94 %** | **0.0367** | 96.4 | 50.2 | 184.8 | 303.0 | 815 | 746 |
| llama.cpp `74a4fb00`, official Q4_K_M | 6.18 | 3.13 GB | | | | 55.5 | 193.3 | 279.9 | 2,062 | 1,881 |

Weight bytes per token: all linear layers plus one row of the embedding and per-layer tables (the EXL3 download is
larger because it carries those tables in BF16). Fidelity: the held-out 40-document, 8-language set against the
official BF16 logits; the `fast` row comes from the int8-activation evaluation, which measured `default` at
86.95 % / 0.144 side by side. C1 shipped = MTP drafter + prompt lookup; C4 and C8 run without drafting.
¹ BF16 row from an earlier session (MTP drafter before the confidence window, prefill before the current exact
prefill kernels); every other speed cell comes from the session of the tables above.

Drafters on the official Q4_K_M (`default`, C1, token ids identical to drafting off for every drafter; MTP drafters
with the confidence window at their own thresholds, 0.4 for ours and 0.6 for Google's). This table comes from an
earlier session than the tables above (before prompt lookup); the DFlash columns from an earlier session still:

| Prompt | Our MTP drafter: tokens / step | tok/s | Google assistant, untuned: tokens / step | tok/s | NVIDIA DFlash (360M): tokens / step | tok/s | Drafting off: tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| English | 2.39 | 93.8 | 2.59 | **97.6** | 2.21 | 69.0 | 52.1 |
| Spanish | 2.10 | **86.7** | 2.10 | 83.4 | 1.55 | 58.5 | 52.0 |
| Japanese | 2.13 | **85.9** | 2.19 | 85.0 | 1.56 | 59.9 | 52.0 |
| Code | 3.46 | **113.3** | 3.32 | 108.1 | 2.15 | 73.6 | 52.0 |

DFlash gets more tokens per step in English, but each of its draft steps costs about 10 ms on this machine against
about 1 ms for the MTP assistant, so its speed-up is lower (x1.12-1.41 vs x1.60-2.18).

Long-context decode (`default`, one request, Q4_K_M, tok/s at the given context depth; earlier session; global-layer
attention reads each K/V chunk once per pass, byte-identical, `STRIX_G4_ATTN_KV_SHARED=0` restores the previous kernel):

| Context | Before | Now | llama.cpp `74a4fb00`, same depth |
|---|---:|---:|---:|
| 8K | 33.9 | **38.7** | 49.4 |
| 32K | 9.8 | **26.7** | 42.6 |
| 64K | 5.7 | **18.4** | 36.2 |

Prefill (tok/s; exact prefill in every column):

| Prompt | `default` | `fast` | `quality` | llama.cpp |
|---|---:|---:|---:|---:|
| 512 | 532 | 551 | 794 | 2,077 |
| 2K | 535 | 538 | 815 | 2,062 |
| 8K | 509 | 509 | 746 | 1,881 |
| 16K | 462 | 465 | 638 | 1,689 |
| 32K | 397 | 399 | 533 | 1,376 |

Exact prefill uses FP32 arithmetic without contraction, so token ids and state are identical at every length;
`quality` runs the exact many-row EXL3 prefill. `STRIX_G4_PREFILL_EXACT_MANYROW=0` restores the eight-row passes with
the same bytes.

`default` prompt prefill: the declared identity is `g4-wmma-pf-v2` on Q4_K_M GGUF packs (other pack types keep exact
prefill). The `default` contract (prompt prefill `g4-wmma-pf-v2` + decode `g4-v1`) is pending qualification against
the `default` bars on a fresh held-out set; until its receipt is named, `default` runs exact prefill, the receipt lists
`pending g4-wmma-pf-v2 (exact runs)` and one stderr line says so. `STRIX_G4_PREFILL_IDENTITY=exact` is the off switch:
exact prefill, bit-exact with the serial path, at any time. `=v1` selects `g4-wmma-pf-v1`, whose contract with `g4-v1`
decode passed the same bars (teacher-forced prefill fidelity up to ~8.4K tokens in 8 languages within its bands;
Global-MMLU, 2,000 paired items in each of 8 languages: accuracy-loss upper bound 0.15-0.32 points per language, bar
0.5; receipt `0b98d054`). Prefill per identity (Q4_K_M, tok/s at 2K / 4K / 8K, interleaved session below): exact
526 / 509 / 481, `g4-wmma-pf-v1` 2,359 / 2,249 / 1,957.

`fast` prompt prefill: `a8` (`STRIX_G4_PREFILL_IDENTITY=a8`, valid only with the `g4-a8-v1` decode; with another decode
identity exact prefill runs and one stderr line says so). The prompt runs in 512-row chunks on the same `g4-a8-v1`
linears as decode (IU8 WMMA for long chunks, the decode kernels for up to 8 rows), canonical attention: every prompt
row's KV and logits equal the a8 decode of the same rows, byte for byte, so the `g4-a8-v1` qualification covers it
and served token ids equal those of the same prompts fed through decode rows (C1 and C4). It is not bitwise equal to
exact prefill. The server prints `gemma4 prefill route a8-many-row (g4-a8-v1 linears, 512-row chunks)` when it runs.
`STRIX_G4_PREFILL_IDENTITY=exact` turns it off. Q4_K_M, one node, prefill tok/s, the same binary for all three
routes, fresh process per run, both run orders, `g4-a8-v1` decode:

| Prompt | exact (many-row) | **`a8` (`fast`)** | `g4-wmma-pf-v1` + flash attention (not qualified) |
|---|---:|---:|---:|
| 512 | 542 | **1,871** | 2,756 |
| 2K | 516 | **1,733** | 2,656 |
| 8K | 477 | **1,498** | 2,591 |

Through the engine (8K-token prompt, one fresh process each, same binary): exact 16.19 s, **`a8` 5.13 s**, v1 + flash
2.97 s.

Other selectable prompt prefill routes (non-EXL3 packs; `quality` keeps exact prefill unless one is selected):
`STRIX_G4_PREFILL_IDENTITY=v1` runs `g4-wmma-pf-v1`: the linears run on the WMMA prefill GEMMs with FP16 operands over
weight panels decoded once at load; attention stays canonical; decode is unchanged. `=v2` runs `g4-wmma-pf-v2` (hi/lo
split activations), the declared `default` prompt prefill. Either route runs the prompt bulk: the last <= 8 prompt tokens and the anchor (the final prompt
position, whose logits pick the first token) run the decode arithmetic (`g4-v1` under `default`), as decode rows do.
Both change logits, so they are not exact. v1 prompt prefill for `fast`: qualification of the pair v1 prefill + a8 decode pending; `fast` runs `a8`
prefill until then. Q4_K_M, one node, prefill tok/s, interleaved runs in one session:

| Prompt | exact (`=exact`) | `g4-wmma-pf-v1` (`=v1`) |
|---|---:|---:|
| 2K | 526 | 2,359 |
| 4K | 509 | 2,249 |
| 8K | 481 | 1,957 |

The routes keep canonical attention, byte-identical to the decode attention. Its score grid runs row tiles first in groups of 8 and its D512 values run in 512-thread blocks that share one staged probability tile: attention 8K 1,333 → 1,265 ms (−5.3 %), 32K 16.83 → 15.91 s (−5.5 %), same bytes; off with `STRIX_G4_PREFILL_CANON_WIDE=0 STRIX_G4_PREFILL_CANON_GROUP=0` (`STRIX_G4_PREFILL_CANON_ROWS_MAJOR=0` restores key tiles first).

llama.cpp commands: C1 `llama-bench -m Q4_K_M.gguf -fa 1 -ngl 99 -n 128 -p 0 -r 3`; C4/C8
`llama-batched-bench -m Q4_K_M.gguf -fa on -ngl 99 -c 2048 -npp 64 -ntg 32 -npl 1,4,8`; prefill
`llama-bench -m Q4_K_M.gguf -fa 1 -ngl 99 -n 0 -r 2 -p 512,2048,8192` and `-r 1 -p 16384,32768`; the llama.cpp `74a4fb00` server with
`-m Q4_K_M.gguf -ngl 99 -fa on -c 16384 -np 8 --jinja`.

Numbers vary with node load (shared LPDDR5X bandwidth); compare runs from the same session only.

## Off switches

Every speed lever that keeps bytes identical is on by default with an `STRIX_G4_*=0` switch that restores the
previous exact path; the active switches appear under `off_switches` in each receipt. Module READMEs list them:
`src/gemma4/kernels`, `src/gemma4/prefill`, `src/gemma4/exl3`, `src/gemma4/eagle`, `src/gemma4/engine`. The
changelog of every default-on Nemotron lever with its measured gain is in [measured gains](../../measured-gains.md).
