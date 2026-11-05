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

## Prompt prefix cache

The server keeps exact checkpoints of the prompt state in blocks of 256 tokens. A request restores the longest
cached prefix of its token ids (at most n-2 tokens) and prefills only the rest. It matches token ids, not a
session id, so it also hits when an agent client resends the conversation without the previous turn's reasoning.
Checkpoints are taken at the last block boundary before the final prompt token and every 16,384 tokens, in pinned
host RAM, LRU-evicted; row segments are shared between checkpoints of the same prefix.

- **Default:** ON. **Off switch:** `--prefix-cache off` or `STRIX_QRTX_PREFIX_CACHE=0` (the previous code path).
- **Budgets:** `--prefix-cache-host-mb N` (default: the smaller of 8,192 MiB and 25 % of the available RAM at
  startup; pinned host memory is reserved in slabs of 32 segments as checkpoints are captured and stays reserved
  until shutdown) and `--prefix-cache-vram-mb N` (default 0). A capture that does not fit is skipped; a request
  never fails because of the cache.
- **Tuning:** `--prefix-cache-block T` (default 256), `--prefix-cache-stride S` (default 16384, 0 = only the last
  block boundary; a multiple of T).
- **Reporting:** `usage.prompt_tokens_details.cached_tokens` is the number of restored tokens. The `strix.rtx`
  receipt adds `prefix_cached`, `prefix_restore_ms` and `prefix_capture_ms`. `GET /v1/cache/stats` and the
  `prefix_cache` object of `/api/status` report lookups, hits, captures, bytes, evictions, capture time and the
  reserved host bytes.
- **Compatibility:** a checkpoint is used only by the same process and the same key: SHA-256 over the layout
  version, block size, model, MTP sidecar, draft vocabulary, rope and table files (path, size, mtime), context
  cap, block schedule, PLE placement and every engine environment variable that selects kernels. Nothing is
  written to disk.
- **Exactness:** restored state, 64 decoded ids and their logits are bit-identical to a fresh prefill
  (`qwen_prefix_cache_e2e`: checkpoint-continue, restore into another session, changed suffix, changed boundary
  token; sync and overlapped prefill). With MTP, a checkpoint at position P also records token P and is restored
  only when it matches. Over HTTP, 48-50K-token agent turns gave identical output with the cache on and off (8/8).
- **Cost to other slots:** a restore copies the saved rows on the decode stream, so the other sessions' decode
  pauses for the copy; the pause grows with the number of restored tokens.

## Tests

`qwen_decode_e2e` (decode gate, rollback arms, draft invariance), `qwen_cohort_e2e` and `qwen_cohort_plan_test`
(multi-session equality and window packing), `qwen_qsa_multi_test`, `qwen_quant_host_test` and
`qwen_quant_device_test` (GGUF decoders against the host reference), and the `qwen_decode_bench` timing bench. Prefix cache: `qwen_prefix_cache_host_test` (planner, identity,
store, slab pool, batched capture) and `qwen_prefix_cache_e2e`.
