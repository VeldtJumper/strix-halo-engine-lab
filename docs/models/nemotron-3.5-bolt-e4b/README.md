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
| `default` | `g4-v1` (FP32 activations, Q4_K_M weights) + MTP drafter at C1 | `prefill-exact` | Exact: output bytes equal the declared serial path for every batch size (C4 == C1) and with the drafter on or off |
| `fast` | `g4-a8-v1` (int8 activations, qualified) + MTP drafter | WMMA prefill (pending `g4-wmma-pf-v2`; exact prefill runs until then) | Named profile, qualified against frozen fidelity bands |
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
agent-style prompts that copy from their context gain most (C1 `default`: 114.9 vs 48.2 tok/s). `STRIX_G4_LOOKUP=0`
turns it off.
`--print-effective-config` (decode) prints the resolved profile and switches without loading a request.

<a id="how-fast"></a>

## Measured results

One Strix Halo node (gfx1151, 128 GB), official Q4_K_M GGUF for `default` and `fast`, EXL3 pack for `quality`;
256-token English chat decode, greedy. C4 and C8 are aggregate tok/s across concurrent requests. Each llama.cpp
value comes from the same measuring session as the row it is compared with.

| Profile | C1 with the [MTP drafter](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-MTP-Drafter) (shipped) | C1, drafter off | C4 | C8 |
|---|---:|---:|---:|---:|
| `default` (exact, `g4-v1`) | **92.9** | 56.0 | 169.3 | 238.5 |
| `fast` (`g4-a8-v1`, qualified) | **100.6** | 54.1 | 206.7 | 340.2 |
| `quality` ([EXL3 M4 pack](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-EXL3), `g4-exl3-v3`) | **83.8** | 44.4 | 153.1 | 258.3 |
| llama.cpp `74a4fb00`, same Q4_K_M | — | 55.5 | 193.8 | 283.4 |

With the drafter, token ids equal drafting off (verified for English, Spanish, Japanese and code at C1, and at C4).
At one request the engine verifies only the drafted tokens the drafter is confident about (a cumulative-probability
window, threshold stored per drafter), which lifted C1 with the drafter from 85.1 / 92.2 / 72.1 to 92.9 / 100.6 / 83.8
tok/s (`default` / `fast` / `quality`, same machine); `STRIX_G4_MTP_CONF=0` restores the previous window policy.

Single-request decode also drafts from the text itself (prompt lookup): it proposes the continuation of the longest
earlier match of the last tokens in the prompt and output, and the main model verifies every token, so token ids equal
drafting off. Without a drafter it gains strongly (C1, `default`, 256 tokens): agent edit prompt (2.2K tokens)
**114.9** tok/s vs 48.2 with drafting off, JSON tool prompt (1.2K) **122.9** vs 49.3, 29.5K-token document **35.0** vs
28.1, code **55.8** vs 51.9; English, Spanish and Japanese chat +1 to 2 %. Beside the MTP drafter a per-request
selector drafts a lookup window only on a strong match and backs off on misses, so it stays out of the way: equal to
the drafter alone within noise on chat, code, agent prompts and 32K (agent edit 146.8 vs 146.2, JSON tool 172.2 vs
171.6, code 112.6 vs 112.5, 29.5K 50.5 vs 50.4, English chat 92.8 vs 92.8). `STRIX_G4_LOOKUP=0` turns it off.

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
| Mixed prompts (chat, code, one ~1.3K-token document): batching on, aggregate tok/s | 70.1 | **87.9** | **73.1** |
| Same, one request at a time (`STRIX_G4_SERVE_BATCH=0`) | 70.0 | 62.8 | 63.0 |
| Same, time to first token p50, on / off | 0.10 / 0.10 s | **0.26** / 4.6 s | **0.60** / 13.8 s |
| llama.cpp `74a4fb00` server (`-np 8`, same Q4_K_M, same session) | 50.1 | 124.4 | 239.7 |
| Short chat prompts, 256 tokens: batching on / llama.cpp server | — | — | **172.4** / 245.9 |

The decode step with eight requests matches the engine's own C8 step. With long prompts in the mix, exact prompt
processing (about 435 tok/s) still pauses the other requests while a prompt chunk runs.

While other requests decode, a prompt is processed in chunks of at most 64 rows (`STRIX_G4_SERVE_PREFILL_ROWS`,
`0` = whole 512-row chunks), so a long prompt pauses them for shorter stretches: with the mixed prompts above the
longest gap between two tokens of a request went from 1.05-1.12 s to 0.41-0.42 s and the 99th percentile from
0.44 / 1.08 s to 0.17 / 0.20 s (4 / 8 clients); throughput and token ids are unchanged. With `fast`
(int8-activation decode beside the exact prompt processing) prompt chunks keep their full size, since where a chunk
ends decides which prompt rows a decode pass writes.

The server also keeps the committed state at the end of each prompt and of each finished request and reuses it
when a later prompt starts with exactly those tokens, typically the next turn of a conversation, which re-sends the
conversation so far: only the new rows are processed. Turn-2 time to first token (`default`, six conversations in
parallel): 4K-token document **0.20 s** vs 12.59 s without the cache, 1K-token document **0.19 s** vs 3.48 s, chat
0.39-0.51 s vs 1.31-1.66 s; both turns' token ids equal the cache off. Up to 16 states within 4 GB
(`STRIX_G4_SERVE_PREFIX_MB`); `STRIX_G4_SERVE_PREFIX=0` turns it off. With `fast`, whose int8-activation decode differs from the exact prompt
processing, only states at a 512-row prompt boundary written by prompt processing are reused, never an answer. Two different conversations that share a long
document but diverge after it do not share it yet.

**MTP drafter** - [hyukudan/Nemotron-3.5-Bolt-E4B-MTP-Drafter](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-MTP-Drafter). A small assistant model (Google's
Gemma-4-E4B-it MTP assistant, 78.8M parameters) proposes the next few tokens and the main model verifies them, so the
output is identical to decoding without it; only the speed changes. The speed-up comes from the MTP assistant
architecture and the engine's exact speculative decoding: C1 decode goes from 52.0 to 92.9 tok/s in English
(`default`, drafter off and on in the same session, x1.79); [Google's untuned assistant](https://huggingface.co/google/gemma-4-E4B-it-assistant) is about as fast in
this engine (see the comparison below). We lightly fine-tuned it on about 6.3M tokens that Nemotron 3.5 Bolt E4B
generated from 17,865 prompts in eight languages plus code, tool calls and reasoning; on a held-out set of 697 prompts
that adds about 1.5 % expected tokens per cycle (English 3.61 -> 3.65, Japanese 2.52 -> 2.60); with the confidence window it is faster
than the untuned assistant on Spanish, Japanese and code and slower on English (+1.4 % geometric mean over the four). Download it into `MODEL_DIR/drafter-mtp/`; the engine uses it automatically.

**EXL3 M4 pack (`quality` profile)** - [hyukudan/Nemotron-3.5-Bolt-E4B-EXL3](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-EXL3). An EXL3 (trellis-coded,
scaled-Hessian) quantization of the BF16 model: MLP at 4 bits, attention q/o at 5, k/v and the output head at 6,
embedding and per-layer tables kept in BF16 - 4.65 bits per weight across the quantized linears, a 9.7 GB download
(2.5 GB of quantized weights) that reads 2.70 GB of weights per token (about 14 % less than Q4_K_M). On a held-out
40-document, 8-language set its output is closer to the original BF16 model than the standard Q4_K_M: top-1 agreement
92.94 % vs 87.0 %, mean KL 0.0367 vs 0.141. Pick it (`--profile quality --exl3-arm DIR`) when you want output closer
to BF16; stay on `default` or `fast` with Q4_K_M for the highest speed. `quality` decodes at 83.8 tok/s with the
drafter (44.4 without, token ids unchanged); its prefill and decode are still being optimised.

**Compared with NVIDIA's official releases.** NVIDIA publishes the main model as [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16) and as one
quantization, the [Q4_K_M GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB), plus a [DFlash drafter](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash). Measured cells only; an empty cell was not
measured. Same machine, greedy, 256-token chat decode, tok/s.

| Model (engine profile) | Download GB | Weight bytes read per token | Top-1 vs BF16 | Mean KL vs BF16 | C1 with MTP drafter | C1 plain | C4 | C8 | Prefill 2K | Prefill 8K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Official BF16, reference | 16.06 | 9.29 GB | 100 % | 0 | 38.1 ¹ | 19.8 | 72.4 | 156.7 | 565 | 534 |
| Official Q4_K_M, `default` | 6.18 | 3.13 GB | 87.0 % | 0.141 | 92.9 | 56.0 | 169.3 | 238.5 | 466 | 429 |
| Official Q4_K_M, `fast` | 6.18 | 3.13 GB | 86.7 % | 0.150 | 100.6 | 54.1 | 206.7 | 340.2 | 458 | 414 |
| **EXL3 M4**, `quality` | 9.71 | **2.70 GB** | **92.94 %** | **0.0367** | 83.8 | 44.4 | 153.1 | 258.3 | 769 | 701 |
| llama.cpp `74a4fb00`, official Q4_K_M | 6.18 | 3.13 GB | | | | 55.5 | 193.8 | 283.4 | 2,052 | 1,873 |

Weight bytes per token: all linear layers plus one row of the embedding and per-layer tables (the EXL3 download is
larger because it carries those tables in BF16). Fidelity: the held-out 40-document, 8-language set against the
official BF16 logits; the `fast` row comes from the int8-activation evaluation, which measured `default` at
86.95 % / 0.144 side by side. C4 and C8 run without drafting. The EXL3 prefill cells come from a later session, in which
llama.cpp measured 2,053 / 1,875 tok/s. ¹ BF16 row measured before the confidence window; the Q4_K_M and EXL3
decode cells and the llama.cpp decode cells come from one later session.

Drafters on the official Q4_K_M (`default`, C1, token ids identical to drafting off for every drafter; MTP drafters
with the confidence window at their own thresholds, 0.4 for ours and 0.6 for Google's; DFlash from an earlier session):

| Prompt | Our MTP drafter: tokens / step | tok/s | Google assistant, untuned: tokens / step | tok/s | NVIDIA DFlash (360M): tokens / step | tok/s | Drafting off: tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| English | 2.39 | 93.8 | 2.59 | **97.6** | 2.21 | 69.0 | 52.1 |
| Spanish | 2.10 | **86.7** | 2.10 | 83.4 | 1.55 | 58.5 | 52.0 |
| Japanese | 2.13 | **85.9** | 2.19 | 85.0 | 1.56 | 59.9 | 52.0 |
| Code | 3.46 | **113.3** | 3.32 | 108.1 | 2.15 | 73.6 | 52.0 |

DFlash gets more tokens per step in English, but each of its draft steps costs about 10 ms on this machine against
about 1 ms for the MTP assistant, so its speed-up is lower (x1.12-1.41 vs x1.60-2.18).

Long-context decode (`default`, one request, Q4_K_M, tok/s at the given context depth; global-layer attention
reads each K/V chunk once per pass, byte-identical, `STRIX_G4_ATTN_KV_SHARED=0` restores the previous kernel):

| Context | Before | Now | llama.cpp `74a4fb00`, same depth |
|---|---:|---:|---:|
| 8K | 33.9 | **38.7** | 49.4 |
| 32K | 9.8 | **26.7** | 42.6 |
| 64K | 5.7 | **18.4** | 36.2 |

Prefill (tok/s; every profile uses exact prefill):

| Prompt | `default` | `fast` | `quality` | llama.cpp |
|---|---:|---:|---:|---:|
| 512 | 471 | 474 | 768 | 2,065 |
| 2K | 466 | 458 | 769 | 2,052 |
| 8K | 429 | 414 | 701 | 1,873 |
| 16K | 404 | 403 | 618 | 1,684 |
| 32K | 358 | 355 | 520 | 1,373 |

Exact prefill uses FP32 arithmetic without contraction; `quality` uses the EXL3 exact many-row prefill. The `quality`
column was measured in a later session, where llama.cpp gave 2,048 / 2,053 / 1,875 / 1,664 / 1,371 tok/s;
`STRIX_G4_PREFILL_EXACT_MANYROW=0` restores the eight-row passes with the same bytes. The `fast` profile's WMMA
prefill is pending qualification and is not applied.

llama.cpp commands: C1 `llama-bench -m Q4_K_M.gguf -fa 1 -ngl 99 -n 128 -p 0 -r 3`; C4/C8
`llama-batched-bench -m Q4_K_M.gguf -fa on -ngl 99 -c 2048 -npp 64 -ntg 32 -npl 1,4,8`; prefill
`llama-bench -m Q4_K_M.gguf -fa 1 -ngl 99 -n 0 -r 2 -p 512,2048,8192` and `-r 1 -p 8192,16384,32768`.

Numbers vary with node load (shared LPDDR5X bandwidth); compare runs from the same session only.

## Off switches

Every speed lever that keeps bytes identical is on by default with an `STRIX_G4_*=0` switch that restores the
previous exact path; the active switches appear under `off_switches` in each receipt. Module READMEs list them:
`src/gemma4/kernels`, `src/gemma4/prefill`, `src/gemma4/exl3`, `src/gemma4/eagle`, `src/gemma4/engine`. The
changelog of every default-on Nemotron lever with its measured gain is in [measured gains](../../measured-gains.md).
