# Serving over HTTP

[English](serving.md) · [Español](serving.es.md) · [Main README](../README.md)

Every engine ships a server. This page is the entry point: which server each engine has, what it does, and how to
expose it safely. The measured serving results stay on the model pages; the request contracts, options and gates are
in [docs/engine/serving/serving.md](engine/serving/serving.md).

| Engine | Server | Serving features | Measured results |
|---|---|---|---|
| Qwen3.8-Flash-Next | `strix-hip-decode-token --http-serve` | OpenAI/Anthropic-compatible chat, completions and stateless Responses routes; request slots sized from memory (`--http-batch N`); automatic tiled prefill (T8192 first, smaller tiles under memory pressure); decoding keeps streaming inside long prompt blocks and short requests nest into a paused long prompt; certified prefix state cache; constrained JSON, thinking budgets, logprobs, logit transforms, SSE progress; QSA MTP pool on the single runner; two replicas behind the TensorMeld router | [Qwen page](models/qwen3.8/README.md#how-fast) |
| DeepSeek V4.1 Flash pair | `strix-deepseek41-serve` (one per rank) | chat, completions and Responses routes; server warm-up and DSpark calibration; Engram lookahead between prefill pieces; bounded agent prefix retention; deferred decoder cone for long prompts; `--profile default\|fast` | [DeepSeek page](models/deepseek-v4.1/README.md#how-fast) |
| DeepSeek V4.1 Flash, standalone RTX | `strix-cuda-ds41-replica --serve SPOOL` (persistent file-queue worker) | each request bitwise identical to a fresh process; residency seed and graph pre-capture; session cache | [DeepSeek page](models/deepseek-v4.1/README.md#deepseek-standalone-rtx-results) |
| MiniMax M3.1 Flash | `strix-minimax-serve` | greedy decoding; structural DSpark auto admission; deadline resumes, short-first and nested admission; named sessions (`context_length_exceeded` on overflow) | [MiniMax page](models/minimax-m3.1/README.md#how-fast) |
| GLM-5.3 Flash pair | `strix-glm5-rank` runs | command-line decode with exact MTP self-speculation; serving is through the rank apps | [GLM page](models/glm-5.3/README.md#how-fast) |
| Nemotron 3.5 Bolt E4B | `strix-gemma4-serve` | continuous batching of up to eight chat requests (`STRIX_G4_SERVE_SLOTS`), every request's tokens equal the same request served alone; MTP drafter and prompt lookup at one request; `--profile default\|fast\|quality` | [Nemotron page](models/nemotron-3.5-bolt-e4b/README.md#how-fast) |

## Exposing the servers

Qwen and DeepSeek publish executed features at `GET /v1/capabilities`; each response's `strix` object
states requested versus effective sampling policy. On every server (Qwen, DeepSeek, MiniMax) unsupported
generation fields return 400 naming the field. MiniMax decodes greedy: a positive temperature is accepted and
reported as ignored in `timings.sampling_note`. Qwen supports constrained JSON exact to its constrained target, thinking budgets, logprobs,
logit transforms and SSE progress. Budgets/transforms are declared token-changing policies;
logprobs/progress preserve tokens. Policy selection uses admitted serial-equivalent MTP windows or
serial fallback; `--http-selection-mtp off` forces the latter. See
[chat request contracts](engine/serving/serving.md#chat-requests) and
[constrained output](engine/serving/serving.md#constrained-json-output).

The HTTP servers (`strix-hip-decode-token --http-serve` and `strix-deepseek41-serve`) listen on loopback unless
`--http-bind` says otherwise. They have no TLS of their own. Authentication is one optional shared key:
with `--http-api-key-file FILE` every API and admin route (completions, `/admin/drain`, `/admin/import-state`)
needs `Authorization: Bearer <key>` or `x-api-key: <key>`; without it the server accepts any client. `/health`,
`/metrics` and `/api/status` answer without the key. Bodies, queued requests and open connections are bounded
(`--http-max-queue`, `--http-max-connections`). The HTTP servers close a request that is not fully received
within 60 seconds (408; plus one second per MiB of body) and treat a client that accepts no data for 60 seconds as
gone, cancelling its generation. Ambiguous request framing (conflicting Content-Length, any Transfer-Encoding,
malformed request line or header names) is refused with 400/501. Request bodies being received or handled share a
256 MiB budget (503 with Retry-After when full), and one request body may hold at most 2,097,152 JSON values. Keep the servers on loopback or a private network and put
remote clients behind an authenticating front end such as the [TensorMeld](https://github.com/hyukudan/tensormeld)
router or its control channel; the two-node DeepSeek ranks exchange tensors over the USB4 link, and the RTX split
(pipeline stage TCP port) and the expert servers talk plain TCP: none of these carry authentication, so they must
stay on a direct cable or a private network.

## Further reading

- [Serving over HTTP](engine/serving/serving.md): request contracts, batched decoding, default prompt prefill,
  constrained output, measured scheduling policies.
- [Prefix state cache](engine/serving/prefix-state-cache.md) and
  [serving prefill and prefix-cache composition](engine/serving/serving-prefill-prefix-cache.md).
- [DeepSeek prefix cache qualification](engine/serving/ds41-prefix-cache-qualification.md),
  [DSpark serving auto admission](engine/serving/dspark-auto-qualification.md),
  [MiniMax session cache](minimax/session-cache.md), [DeepSeek RTX session cache](models/deepseek-v4.1/rtx-session-cache.md).
- [Several requests in one decode loop](engine/scheduling/multi-request-cohort.md),
  [sampled decoding](engine/decode/sampled-decoding.md), [release assets](operations/release-assets.md),
  [startup preflight](operations/startup-preflight.md), [service health](operations/service-health.md).
