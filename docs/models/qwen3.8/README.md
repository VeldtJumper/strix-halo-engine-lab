# Qwen3.8-Flash-Next on one Strix Halo node

[English](README.md) · [Español](README.es.md) · [Main README](../../../README.md)

Qwen3.8-Flash-Next, UD-Q4_K_XL GGUF (111 GB) plus a distilled Q8_0 MTP draft head (2.8 GB), on one Strix Halo node
(`gfx1151`, 128 GB): exact MTP speculative decoding, automatic tiled FP16 prefill, native context through 262K and
explicit YaRN through 1M, HTTP serving with batched slots and a prefix state cache, and two replicas behind a router.
This page keeps every published Qwen measurement with its build and conditions, the launch commands, the profiles and
the tuning switches. How the tables are read and what "exact" covers: [methodology](../../methodology.md). Build
instructions: [getting started](../../getting-started/README.md).

<a id="qwen-recommended-command"></a>

## Run

Recommended run: the plain command selects the production defaults, 256 tokens.

```sh
./build/strix-hip-decode-token Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf $(cat prompt.ids) --generate 256
```

Add `--print-effective-config` to inspect before HIP initialization. See [Defaults and off switches](../../methodology.md#defaults-and-off-switches) for baseline scope.

Eligible plain CLI prompts automatically select known-prefix tiled prefill (T8192 for long prompts,
T4096 at 4K), expert union, router rows and `fast-prefill-f16wmma-v1`. Context is sized automatically;
beyond 8192 the resolver adds `--unqualified-context`. Registered residency (`--residency register`)
is the default on UMA; automatic page locking warns and continues unlocked on failure, as before.
An explicit `STRIX_MODEL_REGISTER_MLOCK=1` refuses startup if locking fails; `STRIX_MODEL_REGISTER_MLOCK=0` disables locking, see [Registered page lock](../../operations/model-weight-residency.md#registered-page-lock). FP16 prefill has its own exact reference and is not bit-identical to dense-A8 serial prefill.
Plain native MTP retains its asset-dependent prefill contract. HTTP serving now automatically admits
T8192 (falling back through smaller tiles under memory pressure), with its existing arithmetic and
1024-row checkpoints for a configured prefix cache. This is separate from CLI FP16 auto prefill.
See [HTTP prefill](../../engine/serving/serving.md#default-prompt-prefill). See the
[auto route and off switches](../../getting-started/engine-reference.md#command-line).

`prompt.ids` holds the prompt as token IDs from the model's tokenizer; the engine prints a JSON receipt with
`generated_tokens` and timings. For the speeds above add the speculative options (MTP sidecar, verify windows
2/4/8, draft vocabulary): the complete command is in **[docs/getting-started/engine-reference.md → Command line](../../getting-started/engine-reference.md#command-line)**
and [docs/engine/speculative/mtp-native-cli.md](../../engine/speculative/mtp-native-cli.md). The HTTP server, batched slots (`--http-batch`) and the
prefix cache are in [docs/engine/serving/serving.md](../../engine/serving/serving.md); several requests in one process in
[docs/engine/scheduling/multi-request-cohort.md](../../engine/scheduling/multi-request-cohort.md); sampling in
[docs/engine/decode/sampled-decoding.md](../../engine/decode/sampled-decoding.md).

Qwen's [prefix-state cache](../../engine/serving/prefix-state-cache.md) restores complete attention/GDN
state at admitted boundaries and reads only the suffix. Configure `--prefix-cache-host-bytes N` and
optionally `--prefix-cache-dir DIR`; fair retention is default (`--prefix-cache-retention lru` restores
LRU). Fair retention also reclaims idle certified groups with a newer private-prefix successor by default.
The rule requires no reuse in the retained lifetime and more than 32 global epochs
of recency separation (one capture or validated touch across all conversations).
Pair-exclusive sub-block boilerplate in other workloads can create a false successor.
`/v1/cache/stats` counts these removals in `evictions_superseded`.
`STRIX_PREFIX_FAIR_RECLAIM_IDLE=0` restores its previous victim order. This is exact
(retention choice only); restore validation and numerical identity are unchanged.
Captures reserve memory before pinned copies. `"state_cache":{"retain":false}` permits reads
without storing new prefixes; `GET /v1/cache/stats` reports retention and eviction reasons.
`--prefix-cache off` / `STRIX_PREFIX_CACHE=0` disables it. Certified caching is on by default only for
single tiled-MTP HTTP (`--http-batch 0`, or automatic capacity resolving to the single runner) at T=2048 with
the default profile and assets and context up to 16,384; there the automatic budget after native residency is
`min(4 GiB, A/16, A - reserve)` of available memory A, and results are bitwise identical to cache-off. Every
other automatic configuration, including the plain serve command (which resolves T=8192), declines by name and
runs exactly as cache-off; explicit host bytes keep their behaviour. Startup, `--print-effective-config`,
`/v1/capabilities` and `/v1/cache/stats` report the requested and effective policy and its reason. In-flight
prefix joining (a request waits for the certified checkpoint another slot is producing instead of recomputing it)
is on by default wherever the cache is on and only acts on concurrent strict-L0 certified batch routes;
`STRIX_PREFIX_CACHE_INFLIGHT=0` / `--prefix-cache-inflight off` turns it off. MTP restore accepts only suffix plans equivalent to the uncached execution (on the batch route, prefix restore needs a solo suffix unit, so with prompt grouping it usually recomputes; refusals are named `batch-suffix-not-solo`);
checkpoint copies follow existing block boundaries.

<a id="qwen-profile"></a>

### `--profile default|fast|quality`

One switch selects a product profile for Qwen (CLI, `--http-serve` and the MTP runner):

- `default`: the exact default described above; it sets nothing.
- `fast`: currently applies BF16 MoE route storage (`STRIX_QPF2K_MOE_ROUTE_BF16=1`) where the route admits it,
  ahead of its qualification receipt, without `--allow-pending`. This is bit-changing and changes logits; qualification is
  in progress. The receipt stamps `qualification in progress: moe_route_bf16` and marks the row `provisional`;
  one stderr line discloses the bit change and qualification status. Elsewhere, including HTTP and MTP,
  the row stays `pending` with the decline reason, keeps FP32 route storage and is not exported or stamped as applied.
  Attention selection budget, FP16 chunked recurrence and FP16 output head remain pending and keep their builtin paths,
  with one notice each; the FP16 chunked recurrence row is runnable (`--allow-pending` runs it, stamped `unqualified`).
- `quality`: there is no quality pack for this model, so it runs `default` and says so in the receipt.

`--allow-pending` (or `STRIX_QWEN_PROFILE_ALLOW_PENDING=1`) runs genuinely pending options whose code is in
the build, for benchmarking only, and stamps them `unqualified: <rows>`. The three pending rows currently
have no runnable implementation, so this flag does not change fast's resolution. On YaRN, a row refused by
rotary admission stays pending with its reason and is not stamped as applied. An explicit environment variable
or CLI option always wins over the profile, including explicit off. `--print-effective-config` shows the selected
profile and the state of every option.

<a id="qwen-long-context"></a>

## Long context

Native capacity is 262,144 positions; prompt plus generation must fit. The resolver automatically
marks native contexts above the 8192-token fidelity qualification as unqualified. A capacity or
exactness pass does not establish task accuracy at that length.

YaRN is an opt-in rotary identity, CLI only. With the verified
[distill-g bundle recipe](../../../recipes/qwen-distill-g-yarn.mtp-bundle.json) and matching sidecar/vocabulary
beside the model, a bare YaRN command selects exact MTP automatically:

```sh
./build/strix-hip-decode-token MODEL.gguf --prompt-ids prompt.ids --rope-yarn 2 --context 524288 --generate 256
```

Use factor 4 with `--context 1048576` for 1M capacity. `--prompt-ids` avoids shell argument limits.
The QSA committed MTP pool and rolling PLE are default-on on admitted routes. Memory admission tries
the storage ladder before smaller prompt tiles and reports `storage_ladder_step` and `yarn_mtp_storage`.
A post-registration physical-margin refusal advances the ladder with allocated storage pinned.
The ladder ends with a tight storage set at 8,192, then 4,096, then 2,048 rows; a 1M MTP run on one node is admitted at 8,192 rows before
registration and settles at 2,048 rows with the tight set after it, without falling back to serial. On the MTP route the
estimate charges one dense FP16 image and no replacement PLE slab, only while the prompt runs; when the
`gdn_fp16_chunk` prefill option is enabled, its persistent buffers (489.7 MB at 8,192 rows) are charged.
A refusal reports the observed free memory and the shortfall.
A missing/invalid bundle before mutation falls back to serial YaRN with a notice; explicit `--mtp`
retains its pinned-asset contract. `STRIX_QWEN_YARN_MTP=0` selects serial YaRN.

Full 262K/1M and target-scope 524K MTP equality checks are published; these are relative to serial
YaRN, not native rotary, and do not qualify general 1M model fidelity or complete workloads.
Measured on one Strix Halo node, default route (code `a199c8e0` plus the bundle selection of #137), diverse
real prose (50 public-domain books in six languages), greedy decoding, automatic storage ladder (no storage
overrides). TTFT runs from process launch to the first token and includes model load.
The 262K prefill cell and the 262K serial decode cell are from the build with the GDN plain-launch fix (262,144-token
prompt, prefill time without model load, `--profile default`, one run on Linux 7.4: 175.18 s; 176.60 s on `17b9c57b8`,
within run-to-run noise); the 262K MTP decode cell is from build
`7b2453fe`; the row's other cells are from the build above.
The 1M row's serial cells are from public main `fdc39faba` on Linux 7.3; its MTP cells and exact gate are from the
MTP admission change (#288) on Linux 7.4. The two sets were measured on different kernels, so each cell states its
kernel and the row gives no serial-to-MTP ratio.

| Context (YaRN factor) | Exact gate (MTP == serial == pool off) | TTFT serial / MTP | Prefill, serial | Decode serial / MTP | MTP acceptance |
|---|---|---|---|---|---|
| 262,400 (2) | Full scope | 219 s / 224 s | **1,496 tok/s** `--profile default` (GDN plain-launch fix) | 21.74 (GDN plain-launch fix) / **59.5 tok/s** (`7b2453fe`) | 79 % (98/124) |
| 524,288 (2) | Target scope | 470 s / 508 s | 1,215 tok/s | 19.7 / **60.6 tok/s** (3.1x) | 85 % |
| 1,048,576 (4) | Full scope (MTP against serial with the two prefill paths below off) | 925 s (Linux 7.3) / 1,133 s (Linux 7.4) | 1,198 tok/s (Linux 7.3); MTP route 936 tok/s (Linux 7.4) | 19.6 (Linux 7.3) / **60.2 tok/s** (Linux 7.4) | 85 % (102/120) |

Since the HC FP16 output path became the default prefill arithmetic, serial YaRN prefill uses it, while the
YaRN MTP route keeps the previous HC output and index projection arithmetic. Each route is exact against its
own reference: the MTP route matches serial YaRN run with
`--prefill-disable hc_output_wmma --prefill-disable index_proj_wmma`. The 262K and 524K MTP == serial rows
above were measured before that change; the 1M row's gate is the comparison with those two paths off.

See [YaRN MTP and bundles](../../engine/speculative/mtp-native-cli.md#explicit-yarn-mtp-cli),
[pool/PLE route matrix](../../engine/qsa-committed-pool.md) and
[long-context fidelity checks](../../../tools/qualification/longctx_fidelity/README.md).

<a id="beyond-english"></a>

## Beyond English

- **Draft vocabulary packs per language (Qwen).** The draft head drafts from a subset of the vocabulary. The
  repository ships a 32k core plus 16k packs for German, Spanish, French, Italian, Portuguese, Russian, Japanese and
  Chinese in [`data/draft-vocab/`](../../../data/draft-vocab); `--mtp-draft-vocab auto:data/draft-vocab` picks the pack
  that matches the prompt. On a Spanish chat prompt this raised decode from 32.9 to 38.9 tok/s (+18 %). DeepSeek's
  DSpark uses the same idea (`--dspark-vocab auto:`). The pack only changes how many guesses are accepted, never the
  text.
- **Quality per language and category.** The harness reports KL and top-1 for multilingual text, Spanish prose,
  English prose, code, agent tasks, math and chat, each with a margin. A choice that hurts one category is rejected
  even when the average looks fine.

<a id="how-fast"></a>

## Measured results

Sources: [prefill/default measurements](../../performance.md#qwen-prefill-and-defaults) and [decode/serving results](../../performance.md).

All rows: Qwen3.8-Flash-Next, UD-Q4_K_XL GGUF (111 GB) plus a distilled Q8_0 MTP draft head (2.8 GB), MTP
speculative decoding with windows 2/4/8, dense A8 profile, thinking off, 256 generated tokens unless stated.

| Mode | Hardware | Prompt / context | Concurrency | Best | Typical | Build |
|---|---|---|---:|---:|---:|---|
| MTP decode | 1 node | 1K, code, auto windows | 1 | **62.44 tok/s** (72 % of 86.8 tok/s byte ceiling) | prose: 55.7 tok/s (`0117a140`) | main `85353311b` |
| MTP decode | 1 node | 8K | 1 | **54.00 tok/s** (code, auto windows) | prose: 49.58 tok/s (`226bb691`, exact draft attention) | code `702bc074`; prose `226bb691` |
| MTP decode | 1 node | 16K | 1 | **50.69 tok/s** (code, auto windows) | prose: 50.31 tok/s | main `85353311b` |
| MTP decode | 1 node | 32K | 1 | **47.9 tok/s** (code; observed range 44.5–52.6) | median of 4 | `0117a140` |
| MTP + copy windows (opt-in), long copy-edit | 1 node | Markdown document (~10 KB) to fix, 2,047-token answer | 1 | **183.5 tok/s** | | `e1b115a2` |
| MTP + copy windows (opt-in), long copy-edit | 1 node | Python file + diff to apply, 1,417-token answer | 1 | **163.9 tok/s** | | `e1b115a2` |
| MTP + copy windows (opt-in), long copy-edit | 1 node | C++ file to rewrite with renames, 1,711-token answer | 1 | **155.1 tok/s** | | `e1b115a2` |
| MTP + copy windows (opt-in), 7 copy-edit tasks | 1 node | 5-10 KB inputs, 1,114-2,047-token answers | 1 | | **125.2 tok/s** (90.2 without copy windows) | `dd65dbd5` |
| MTP + copy windows (opt-in), agent sessions | 1 node | 6 coding-agent sessions | 1 | | 59.9 tok/s (unchanged by copy windows) | `dd65dbd5` |

Copy windows (`--mtp-ngram-draft=wide --mtp-copy-wide-rows 32`) verify up to 32 rows when the answer copies text
from the prompt; tokens are identical to the default in every run. They help when the answer re-emits long spans
of the input (rewrites, patches, document fixes), do nothing on new text, and currently work for one request per
process (not yet with the batched server).

**Two replicas** (one complete model per node behind the router, `--http-batch 8`, servers restarted before each
of 3 rounds, no warm-up; token ids equal the single-node run in 570/570 responses). Newer coding-chat wall
rates use build `0117a140`, fresh servers with one excluded warm-up request, 3 ABBAAB rounds; 90/90
responses equal the single-node reference:

| Mode | Hardware | Prompt / context | Concurrency | Best: coding chat | Typical: 8-category short set | Build |
|---|---|---|---:|---:|---:|---|
| Server, two replicas | 2 nodes | ~70-token chat / 36-303-token prompts | 1 | **68.2 tok/s** wall | 42.8 tok/s wall, TTFT p50 0.89 s | `0117a140` coding; release short set |
| Server, two replicas | 2 nodes | same | 2 | **136.4 tok/s** wall | 85.8 wall, TTFT p50 0.85 s | `0117a140` coding; release short set |
| Server, two replicas | 2 nodes | same | 4 | **146.8 tok/s** wall | 102.9 wall, TTFT p50 1.32 s | `0117a140` coding; release short set |
| Server, two replicas | 2 nodes | same | 8 | **158.1 tok/s** wall | 120.4 wall, TTFT p50 2.55 s | `0117a140` coding; release short set |

*Wall* is all completion tokens over wall time including the first token and queueing. The earlier
coding-chat campaign measured **90.7 / 181.5 / 201.6 / 255.0 tok/s** summed per-request decode
at concurrency 1 / 2 / 4 / 8; summed decode excludes each request's prompt and queueing time.

<a id="qwen-prefill-offload"></a>

### Prompt reading, one node and with the RTX PRO

**Prompt reading, one node** (profiles and conditions below) and **with the RTX PRO** (exact CUDA
prefill on the RTX, attention and recurrent state streamed to the node over 1 GbE as each layer finishes, then the
node decodes; 64 generated tokens identical to the node-only run at every length; one run per length):

| Mode | Hardware | 1K | 4K | 8K | 16K | 32K | Build |
|---|---|---:|---:|---:|---:|---:|---|
| Prefill, TTFT | 1 node | **0.77 s** | **2.42 s** | | **9.47 s** | **19.13 s** | main `85353311b`, plain command |
| Prefill offload, TTFT | RTX PRO + 1 node | **1.36 s** | **1.92 s** | **2.94 s** | **4.95 s** | **8.99 s** | `70b93af3` |
| Same server without the RTX, TTFT | 1 node | 1.80 s | 5.20 s | 10.46 s | 21.43 s | 44.02 s | same server build |

The offload compares against the same production server it forwards to (2,048-row prompt tiles), which reads
prompts more slowly than the one-node prefill row above (automatic row tiles). At 32K the RTX path is 4.9× faster than
that server. With the offload enabled, the router sends prompts of 1K tokens or more to the RTX PRO; shorter ones
are read on the node.

**Serving from the RTX PRO alone.** `strix-cuda-qwen-serve` serves the model from one RTX PRO 6000 with exact
prefill and exact greedy MTP decode, the same APIs as the node server and continuous batching over 4 slots.
An exact prompt prefix cache is ON by default (`--prefix-cache off` disables it).
[Build, run and defaults](rtx-serve.md).

**Build `0117a140`** (measured against the release build in the same session, interleaved runs,
every run exact): TTFT 2.3 % (1K), 3.4 % (4K) and 5.1 % (16K, 32K) lower; decode equal or up to 0.4 % faster;
two replicas at concurrency 4 and 8: 3.9 % and 3.6 % more wall throughput, and a 6.9 % lower TTFT at
concurrency 4. Details in [docs/performance.md](../../performance.md#qwen38-flash-next-1).

### Exact prefill defaults and candidates

**Exact combined prefill.** One Strix Halo node, Qwen3.8-Flash-Next UD-Q4_K_XL,
dense-A8 decode with a separate fast FP16 prefill identity; build `8b99cf58`, defaults without environment overrides. Same-node comparisons
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
Re-measured on public main `85353311b` (plain command, one node): 4,096 **1,695 tok/s**, 17,700 **1,665**,
32,768 **1,713** (19.13 s), 131,072 **1,617** (81.08 s), 262,143 **1,509** (173.74 s) = 54.0 / 53.9 / 55.7 / 55.6 / 56.2 %
of the same ceilings; 65,536 was not re-measured.

[PR #49](https://github.com/hyukudan/strix-halo-engine-lab/pull/49) ships combined prefill, GDN gates batching/core16,
and router selection/activation staging as defaults. [PR #50](https://github.com/hyukudan/strix-halo-engine-lab/pull/50)
adds HC output v3 with band 16 (`5011b842`): 32K TTFT **−0.52 s (−2.2 %)** in three alternating pairs,
4K **−54 ms (−1.8 %)** in two pairs; logits and tokens remain equal. A later single 32K run on that build
measured **22.03 s, 1,487 tok/s (48.3 % of the 3,078 tok/s contract ceiling)**.
`STRIX_QPF2K_DEFAULTS=0` suppresses the prefill default set and declines automatic tiling; HC-specific off switches are
`STRIX_QPF2K_HC_OUTPUT3=0` / `STRIX_QPF2K_HC_OUTPUT3_BAND=0`.

**QSA stage batching — on by default, [PR #51](https://github.com/hyukudan/strix-halo-engine-lab/pull/51).**
Exact 32K TTFT **22.21 / 22.66 / 22.76 → 21.57 / 22.32 / 22.58 s** (three pairs, mean **−0.39 s, −1.7 %**);
on the published build, 32K in four ABBA pairs **−0.17 s** on average (every pair faster) and 64K **46.76 → 46.01 s (−1.6 %)**;
17,700 **12.54 → 12.46 s** (one pair). Tokens, terminal logits and the code-1K / prose-8K qualification receipts match.
`STRIX_QPF2K_QSA_MASK=0x57` restores the previous stages (default `0x2257`; `0x257` keeps everything except the attend occupancy variant).
These timings are separate comparisons, not a new full-length sweep.

The plain CLI now resolves to the measured production route. On one Strix Halo node, the fast FP16
prefill identity measured **19.13 s / 1,713 tok/s at 32K**, **81.08 s at 131,072 tokens**, and
**173.74 s / 1,509 tok/s at 262,143 tokens** (public main `85353311b`, automatic T16384 from 32K).
Digests match that identity's reference at 32K/64K/128K/262K; this is not dense-A8 serial bit equality
or a new long-context model-fidelity qualification. See the [catalog](../../optimizations.md#qwen-tiled).

Exact MTP draft attention and the router defaults are enabled in the public Qwen line.
Batched MTP drafting remains opt-in. The HTTP server sizes its request slots from memory (`--http-batch N` caps, `0` = one at a time), serves decoding requests at layer boundaries inside long prompt blocks (`STRIX_BATCH_SERVICE_MS=250`), runs short prompts first and nests them into a paused long prompt (`STRIX_BATCH_NEST_ROWS=2048`), and cancels at those boundaries; every request keeps the token IDs and committed state it has alone. With a 32K prompt arriving among three decoding streams (default command) the p90 decode gap went from 14.3 s to 0.54 s and a short request's first token from 42-52 s to 2.3-3.8 s ([serving](../../engine/serving/serving.md#batched-decoding)).

**MoE route BF16 storage — opt-in candidate, pending qualification.** `STRIX_QPF2K_MOE_ROUTE_BF16=1`
(or `--prefill-moe-route-bf16 on`) stores the per-route MoE dot products in BF16 and widens them exactly in the
ordered merge. It is bit-changing (prefill storage identity `fp16-v1+moeroute16`), default OFF, and
applied only by `--profile fast` as a provisional row until it passes qualification. Same build, plain CLI, three paired runs each: 32K TTFT
**21.83 → 21.15 s (−3.2 %)**, 16K **10.38 → 10.11 s (−2.3 %)**, every pair faster. At 262K (YaRN factor 2,
serial route, `--profile default` vs `--profile fast`, two interleaved pairs) prefill took **191.2 → 185.7 s
(−2.9 %, 1,371 → 1,412 tok/s)**, every fast run faster than every default run; decode is unchanged. See the
[catalog](../../optimizations.md#qwen-moe-route-bf16).

**Chunked GDN prefill on FP16 WMMA — opt-in candidate, pending qualification.** `STRIX_QPF2K_GDN_FP16_CHUNK=1`
(or `--gdn-fp16-chunk 1`) runs the GDN layers' conv, recurrence and gated norm as 32-row chunks on FP16 WMMA with FP32
accumulation and FP32 state (identity `qwen-gdn-fp16-chunk rev 2`; the input projection then writes its BF16-rounded
values as BF16 rows). It is bit-changing, default OFF; `--profile fast` lists it as pending until it passes fidelity
qualification (`--allow-pending` runs it, stamped `unqualified`). Same build, plain CLI, interleaved pairs: 4K
**2.42 → 2.36 s (−2.6 %)**, 32K **19.61 → 18.81 s (−4.1 %)**, 262K **177.9 → 170.0 s (−4.5 %, 1,471 → 1,539 tok/s; one pair)**; the first generated tokens match the
exact route. Unset or `=0` is the exact route. See the [catalog](../../optimizations.md#qwen-gdn-chunk).

GSQ below is a separate speed-over-quality profile.

<a id="gsq-opt-in-speed-over-quality-profile"></a>

### GSQ: opt-in speed-over-quality profile

GSQ HYB-Q2 replaces the quality recipe's routed experts with Q2_0 weights; it is **never the default**.
In weight-file scoring on the locked 64-prompt quality set (23,552 scored positions, 2K–32K contexts), top-1 agreement with the
Q8_0 reference is **76.6 %**, versus **88.1 %** for UD-Q4_K_XL: an **11.5 percentage-point** loss. This profile
does not pass the quality recipe's acceptance criterion. These quality scores do not rescore the engine's FP16 prefill path.

Measured on one Strix Halo node, Qwen3.8-Flash-Next, single request, 256 generated tokens,
`702bc074`; decode time excludes the prompt. Each run matches its own recipe's golden tokens.
The two recipes have different weights and can produce different answers.

| Code prompt | MTP window | GSQ HYB-Q2, tok/s | Quality recipe, tok/s | GSQ / quality |
|---|---|---:|---:|---:|
| 1K | auto | **73.18** | **59.59** | 1.23x |
| 1K | {2} route | **49.16** | **41.85** | 1.17x |
| 1K | forced R2 | **49.17** | **41.82** | 1.18x |
| 1K | forced R4 | **64.42** | **51.56** | 1.25x |
| 1K | forced R8 | **66.56** | **52.07** | 1.28x |
| 8K | auto | **65.96** | **54.00** | 1.22x |
| 8K | {2} route | **45.51** | **40.05** | 1.14x |
| 8K | forced R2 | **45.42** | **40.27** | 1.13x |
| 8K | forced R4 | **59.61** | **51.44** | 1.16x |
| 8K | forced R8 | **56.60** | **50.05** | 1.13x |

Auto uses the production MTP policy. The {2} route is the harness's "R1" MTP setting, not serial greedy decode.
MTP acceptance at auto: 1K 0.72 (GSQ) vs 0.66 (quality); 8K 0.66 vs 0.73. Prompt processing in the same runs
(warm): 1K 1.03 s vs 1.17 s; 8K 7.7-7.8 s vs 7.9 s. Against each recipe's byte ceiling for code 1K with MTP,
GSQ auto reaches about 64 % (113.8 tok/s ceiling) and the quality recipe about 69 % (86.8 tok/s ceiling).

### How a long prompt is read

**Reading a long prompt (Qwen tiled prefill).** The prompt is cut into admitted tiles (up to 8192 rows on the automatic route); inside a block each
weight is read once for all rows, and the recurrent (GDN) layers still update rows in order, so the result is
exact to the declared prefill reference. CLI fast FP16 and HTTP prefill retain separate arithmetic contracts.

## Exact startup and tiling defaults

Exact Qwen startup and tiling defaults: prompt-independent static setup precedes the
request timer; eager HIP code loading preserves a user-supplied HIP setting; automatic
T16384 starts at 32,768 prompt tokens with memory fallback to T8192; exact indexer
projection batching is independent of numerical options. Per-item OFF switches are
`STRIX_QPF2K_EARLY_SETUP=0`, `STRIX_EAGER_CODE_LOAD=0`, `STRIX_QPF2K_T16K=0` and
`STRIX_QPF2K_INDEX_BATCH=0`. Setup/loading move work into startup; they do not establish
lower process wall time. See the [catalog](../../optimizations.md#early-tiled-prefill-geometry-setup)
and [native bundle gates](../../engine/prefill/hc-output-wmma-qualification.md#bundle-native-gates).

<a id="bit-changing-prefill-options"></a>

## Bit-changing prefill options

Qwen's registry is [prefill_identity.hpp](../../../include/strix/prefill_identity.hpp). Each option is
**on by default once READY**, with an individual off switch and a master exact-FP16 switch.
HC low-band layout defaults to mode 3 through decoder qpf2k defaults;
`STRIX_QPF2K_HC_LOWBAND=0` restores the old layout. It is live only with `hc_output_wmma` OFF (the exact-FP16 output route),
not with the default output WMMA identity. Only the qpf2k projection
form 4 to FP32 output v3 band 16 uses it. HC output WMMA, check mode, normalized16,
inverse and other routes retain their old layout. Arithmetic and numerical identities
are unchanged; restart for this cached control. Native local CI passed 132/132 cases
without stray stores, including eight T8224 cases with 512-row pool rounding, and
32K last-row logits matched mode 0 bitwise with `hc_output_wmma` OFF. Exact,
bitwise-checked; no timing claim because low-band is inactive on the default route.
See [local CI](../../engine/prefill/hc-output-wmma-qualification.md#bundle-local-ci-evidence).

Exact [compact QSA output images](../../engine/prefill/compact-stage-images.md)
default ON for eligible full-T score-ring owners;
`STRIX_QPF2K_COMPACT_IMAGES=0` restores FP32 containers after restart.
Calculated memory saving is 96 MiB per T8192 owner. Native local CI passed the
host/HIP fixtures and 32K last-row bitwise logit comparison. Interleaved 32K A/B
was neutral within noise (OFF 19.95 s, n=2, versus default 20.09 s); no speed
gain claimed. Already rounded BF16 bits widen identically. Numerical,
rotary, kernel-plan and saved-state/cache numerical identities are unchanged.
The physical execution identity must change because it hashes the smaller arena
sizes, offsets and strides; it is not a numerical identity. Effective config
reports the requested policy; compiled organization reports admitted savings.

PLE staged-gather acceptance now tracks successful independent row-0 consumption
and accepts the complete ordered deferred tail only after comparing every staged
row ID, including row 0. This bug fix is unconditional, with no switch; copy
semantics, staging guards, worker-error fallback, slot events and identities stay
unchanged. Native local CI passed all 11 device/host fixtures, including the PLE
launcher and staged-gather tests. Baseline and `STRIX_QPF2K_DEFAULTS=0` each turn
every controlled exact organization item off. At 32K, last-row logits match bitwise between
default and each OFF arm (early setup, eager loading, T16384, index batching, compact
images). Three-round interleaved medians were 20.09 s default, 20.48 s early setup
OFF, 20.34 s T16384 OFF and 20.07 s eager loading OFF. Early setup and T16384 show
measured gains; eager loading is neutral within noise. These gains are not additive.
At 131,072 and 262,143 real-text tokens the default admits automatic T16384, and with
`STRIX_QPF2K_T16K=0` it admits T8192 as before, with no fallback. Open, nonblocking gates
are the seven state payloads and T16384 at 262K with pressure fallback; see
[local CI](../../engine/prefill/hc-output-wmma-qualification.md#bundle-local-ci-evidence).
The staged-gather acceptance fix is measured with one binary, staged versus the earlier acceptance rule: full logits
equal at 4K and 32K, tokens equal at 4K, 32K and 262K; prefill 4K -1.1 %, 32K -1.9 % (-0.40 s), and 262K (YaRN
factor 2, T8192 blocks) 198.19 -> 185.49 s (-6.4 %, mean of two pairs), with GPU idle time over that prompt down
from 10.2 s to 1.56 s.
Combined, with one binary on the plain CLI default route (alternating runs), the defaults above against every one of
their OFF switches: 4K 2.548 -> 2.397 s (-5.9 %), 32K 20.57 -> 19.05 s (-7.4 %; T16384 against T8192). The OFF arm
reproduces the earlier logits byte for byte, and generated tokens are equal in every arm.

The READY mask contains `hc_output_wmma`. The implemented candidate `index_proj_wmma` stays outside READY and is default OFF pending qualification. The default prefill identity remains
`fp16-v1+hcout16` (`prefill_identity_hash` ee2cd64684149069); `--prefill-exact-fp16` gives the plain
`fp16-v1` identity (a0094634843e16ee).

| Option | What it does | Gain | Changes bits vs FP16 identity | Measured quality | Switch off (CLI / config / API / env) |
|---|---|---|---|---|---|
| `hc_output_wmma` (on) | HC output: FP16 low values and Q8 coefficients, WMMA accumulation in ascending K16 order, same sigmoid and stream mix | Passed a pre-declared confirmation held-out set | yes | `hc_output_wmma` passed a pre-declared confirmation held-out set; its NLL endpoint is measured against a rounding-equivalent control whose per-case movement is capped at +-0.02 nats/token (a bounded control allowance, not an unbiased estimate of the uncapped difference); the absolute NLL change vs the exact reference also passed (-0.37 % NLL/token, upper bound +0.38 %). The off switch restores the exact output path. | `--prefill-disable hc_output_wmma` / `prefill_disable = ["hc_output_wmma"]` / server-level only / `STRIX_PREFILL_HC_OUTPUT_WMMA=0` |
| `index_proj_wmma` (off; candidate, outside READY) | BF16 indexer q/k WMMA with independent K16 panels and ordered FP32 panel addition; tag `idxproj16` | Pending local qualification | yes | Compiler, native execution, model fidelity and complete-workload gates pending | Opt in: `STRIX_PREFILL_INDEX_PROJ_WMMA=1`; off: `STRIX_PREFILL_INDEX_PROJ_WMMA=0`, `--prefill-disable index_proj_wmma`, config `prefill_disable = ["index_proj_wmma"]`; master exact switch also disables |

The prefill identity hash is FNV-1a 64 of the identity name. The enabled-set tag is not
executed-path evidence: OUT16 requires the fast HC component, the qualified wide shape
(hidden 2560, four streams, rank 320), QPF2K HC admission, FP32 normalized rows and
compatible output selection. Serial suffixes, incompatible normalization forms and
non-fast profiles can retain `hcout16` without executing OUT16. The once-per-process
stderr witness does not establish execution for every request.

Use `--prefill-exact-fp16`, `prefill_exact_fp16 = true`, or `STRIX_PREFILL_EXACT_FP16=1`
to switch all arithmetic options in this registry off. `--prefill-disable` accepts comma-separated names and can be repeated;
unknown or empty names are rejected. CLI/config restrictions combine with environment restrictions;
a config `false` or empty array does not re-enable an environment-disabled option.
These arithmetic controls preserve automatic route selection.

The decode-token app and its `--http-serve` mode share these startup controls.
`--config FILE` reads the arithmetic keys above and the exact startup booleans `qpf2k_early_setup` and `eager_code_load` as `key = value` assignments,
one per line, with lowercase booleans, single-line arrays of double-quoted names and `#` comments.
Unknown/duplicate keys and malformed values are rejected. For example:

```text
prefill_exact_fp16 = false
prefill_disable = ["hc_output_wmma"]
```

Receipts, `--print-effective-config`, HTTP response `strix` receipts and `GET /v1/capabilities`
report `prefill_identity`, `prefill_identity_hash` (16 lowercase hex digits), and
`prefill_options_active` (names in registry order). This identifies the option set; the existing
numerical profile pair still identifies the surrounding arithmetic. Every changed active set
must be re-qualified against its own reference. The set is declared on every route, but
`hc_output_wmma` executes only when prefill runs a fast prefill profile (stderr witness
`qpf2k HC out16 active`); on other routes, including `--baseline` and explicit dense-A8 routes,
it is inert. Equivalence never uses the declared set alone: prefix-state cache keys include the
effective prefill command line (both numerical profiles and `--baseline`), and session state
carries the compiled prefill execution identity, so routes with different arithmetic never compare equal.

**Server-level only:** restart the server to change these options. Compiled prefill plans and
prefix-state caches are shared across requests; the enabled set is bound into compiled execution,
lineage and persistent-cache keys. The process-wide override has no request-scoped synchronization.
A per-call getter alone cannot isolate concurrent or resumed requests. Request fields `prefill_exact_fp16` and `prefill_disable` are refused with HTTP 400.

OFF is required to preserve public-main logits and all seven state components across CLI, MTP, HTTP, nested, GSQ and baseline routes. Compiled execution, lineage and cache identity bytes may differ even when numerical bytes match; native restoration gates must compare numerical payloads separately. Native restoration against public main remains a required release gate.

The [native release gate commands](../../engine/prefill/hc-output-wmma-qualification.md) cover public-main/OFF restoration, ON repeatability, full state exports, MTP, HTTP and nested scheduling.

## Every shipped Qwen default

The changelog of every default-on Qwen lever with its measured gain, exactness evidence and off switch is in
[measured gains](../../measured-gains.md); the full catalog with preconditions and notices is
[docs/optimizations.md](../../optimizations.md#qwen). Serving (HTTP routes, batching, prefix cache, security):
[serving](../../serving.md).
