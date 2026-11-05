# Qwen3.8-Flash-Next on one RTX PRO: `strix-cuda-qwen-serve`

`strix-cuda-qwen-serve` serves Qwen3.8-Flash-Next (UD-Q4_K_XL) from one RTX PRO 6000 (96 GB, Windows WDDM)
with the CUDA engine: exact prefill and exact greedy decode with MTP verify windows. It speaks the same APIs
as the node server: OpenAI chat completions and completions, Anthropic messages, SSE streaming, tool calls,
`/tokenize`, `/health` and `/api/status`, with the GGUF BPE tokenizer and the Qwen chat template.

Output is exact greedy: every request gets the token ids it gets when it runs alone, at any slot count and
with every default below.

## Build

Windows, Visual Studio 2022 and CUDA 13.0, `sm_120`:

```powershell
cmake -S . -B build-rtx -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRIX_CUDA_QWEN_ONLY=ON `
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-rtx --target strix-cuda-qwen-serve
```

`STRIX_CUDA_QWEN_ONLY=ON` builds only the CUDA Qwen targets; Strix Halo builds do not compile these files.

## Run

```powershell
.\build-rtx\strix-cuda-qwen-serve.exe --model Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00003.gguf `
  --mtp-sidecar MTP.gguf --draft-vocab DRAFT_VOCAB --tables TABLES_DIR --rope-table ROPE_TABLE `
  --bind 127.0.0.1 --port 8090
```

`--model`, `--tables` and `--rope-table` are required. Split GGUF parts are found from the first part.

| Flag | Default | Meaning |
|---|---|---|
| `--bind`, `--port` | `127.0.0.1`, `8090` | Listen address. |
| `--api-key-file F` | none | Require the key in `F` as a bearer token. |
| `--model-id ID` | `qwen3.8-flash-next` | Model name reported by the API. |
| `--max-positions N` | 32768 | Context cap per session. A prompt longer than the context returns `context_length_exceeded`; a session never spills to system memory. |
| `--mtp-depth N` | 7 | MTP draft depth per verify window. |
| `--min-p P` | 0.7 | The MTP draft chain stops at a draft token below this probability. Output ids do not depend on it. |
| `--http-slots N` | 4 | Concurrent sessions (continuous batching). `1` serves one request at a time; `STRIX_QRTX_COHORT=0` does the same. |
| `--slot-margin-mib N` | 2048 | A new slot is admitted only while this much VRAM stays free after its session. |
| `--admit-grace-ms N` | 2 | Wait for a burst of requests before the next step (`0` = off). |
| `--max-queue N` | 64 | Requests waiting for a slot; beyond it the server answers busy. |
| `--prefill-chunk N\|auto` | 1024 | Prefill rows of a new request per chunk, run between the decode steps of the other sessions. `auto` fits the chunk to the decode step time. |
| `--cohort-depth fair\|max` | `fair` | How MTP depth is shared when several sessions decode in one step. |
| `--block-rows N` | 1024 | Prefill block rows (multiple of 16, at least 1024). |

Every response carries the `strix.rtx` receipt: emitted ids, prefill and decode timings and slot occupancy.

## Engine defaults

All of these are exact (same token ids as the reference paths) and ON by default. Each has an off switch
that selects the reference path:

| Default ON | Off switch |
|---|---|
| Decode-shaped kernels for windows of up to 8 rows (dense A8, HC mix, router, routed MoE, QSA attention) | `STRIX_QRTX_DECODE_KERNELS=0` |
| CUDA graphs per verify window | `STRIX_QRTX_GRAPHS=0` |
| Graph descriptor tables | `STRIX_QRTX_GRAPH_TABLE=0` |
| Multi-segment QSA cores between graph segments | `STRIX_QRTX_MULTISEG=0` |
| Shared expert on a side stream | `STRIX_QRTX_SIDE_SHARED=0` |
| MTP drafts batched across sessions | `STRIX_QRTX_DRAFT_BATCH=0` |

Opt-in and OFF by default:

| Opt-in | Effect |
|---|---|
| `STRIX_QRTX_WINDOW_ROWS=16` | Verify windows of 9 to 16 rows. |
| `STRIX_QRTX_HEAD1=1` | One-row LM head per window row. |
| `STRIX_QRTX_DRAFT_SYNC=1` | Host sync after each draft step. |
| `STRIX_QRTX_PREFILL_OVERLAP=1` | Prefill chunks on a second stream beside the decode steps. |
| `STRIX_QRTX_PREFILL_POLL=1` | With overlap, decode steps continue until the chunk completes. |
| `STRIX_QRTX_ALLOW_SYSMEM=1` | Allow an allocation to spill to system memory (refused by default). |
| `STRIX_QRTX_NVTX=1` | NVTX profiling ranges. |

## Tests

`qwen_decode_e2e` (decode gate, rollback arms, draft invariance), `qwen_cohort_e2e` and `qwen_cohort_plan_test`
(multi-session equality and window packing), `qwen_qsa_multi_test`, `qwen_quant_host_test` and
`qwen_quant_device_test` (GGUF decoders against the host reference), and the `qwen_decode_bench` timing bench.
