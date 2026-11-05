# DeepSeek-V4.1 two-node: production defaults (P17.2)

From the P17.1 timing campaign (3 interleaved rounds per arm, every arm's tokens checked against its identity's
reference). Code defaults change only what was measured as a gain and is exact; the production identity needs the
deployment's environment (pack paths), listed below.

## Code defaults (unset = on)

| setting | default | off | effect (P17.1 campaign) |
|---|---|---|---|
| `STRIX_DS41_PREPARED_LOAD` | discover exact matching per-node pack ([layout and preparation](prepared-load.md)) | `=0`; explicit directory overrides discovery | R1 tokens identical; default discovery vs `=0` whole-run wall -22.1 s mean (66.0 -> 43.8 s; explicit directory -20.6 s), fresh-process ABBA x2, two Strix Halo nodes, production identity, TP packed accepted on both ranks (72 GiB/node) |
| `STRIX_DS41_BATCH_SEGMENTS` | on (off by itself with `STRIX_DS41_BATCH_ROWS16=1`) | `=0` | batch N=8 +9 % |
| `STRIX_DS41_TP_DOWN` | on (the expert-TP down, width 1152 only) | `=0` | R6 window -3 %, batch N=8 +1.6 % |
| `STRIX_DS41_EXPERT_ROWS` | on: routed IQ2_XXS gate/up and Q2_K down hold every weight group of a wave in registers across the staged-row parts | `=0` | same bits (rank in-process ABA, both ranks: R1 tokens + logits slice, window tokens, attention state). EP decode (the serve identity): R1 -0.16 % [-0.060, -0.031 ms], R6 -1.43 % [-1.001, -0.524 ms], 4/4 faster; expert-TP decode: R1 +0.03 % (neutral), R6 -0.39 % (3/3 faster) |
| `STRIX_DS41_SERVE_SCHED` | `interleave` | `=off` | 8K prompt under load: decode stall <= 1 s instead of ~13 s (its own TTFT about doubles) |
| serve `--dspark-ranks` | 2 (a drafter on each rank) | `--dspark-ranks 1` | |
| serve `--dspark-max-inflight` | 4: plain steps while more requests are in flight | `0`: always draft | keeps C1..C4 (+19..+76 %) without the C8 TTFT cost |
| batch app `--draft-ranks` | 2 | `--draft-ranks 1` | |
| `STRIX_DIST_USB4_EAGER` | 65536 (P17.3) | `=0` | R1 decode +2.8 %, R6 / batch equal |
| `STRIX_DS41_LATE_HC_LAYERS` | `all` (P17.5): the attention site's hc mixes run during the o_b exchange (not with `STRIX_DS41_BATCH_PREFETCH`) | `=none` | R1 -0.2 ms, R6 -0.35 ms, batch (no prefetch) N=2/4/8 -1.5/-0.7/-0.6 ms/step; bitwise |
| `STRIX_DS41_PREFILL_FINAL_KV` | `all` (P17.5): row-split prefill owners send the final ring / compressed entries (+ ratio-2 carry) instead of FP32 K/V and compressor inputs | `=none` | 8K prefill +1.8 % (714.9 -> 729.4 tok/s); prefill --reference bitwise, decode continuation identical |
| `STRIX_DIST_STAGING` | `fused` (P17.4): the TP o_b product writes the pinned send buffer, the add reads the pinned receive buffer (host-path routes) | `=copy` | R1 -0.2..-0.4 ms/token, batch N=1..2 -0.25 ms/step, R6 / N=8 equal; bitwise in layouts 1 and 2bq |
| `STRIX_DS41_EXPERT_TP`, `EXPERT_SLOTS`, `ROUTES_FAST`, `F16_PAIR` | on (P16) | `=0` | |

Unchanged (off unless set): the tpserve prefill
(`STRIX_DS41_PREFILL_EXPERT_TP`, slower than the EP prefill), `STRIX_DS41_BATCH_ROWS16`.

## Agent prefix policy (default)

Serve now resolves no cache flags to `--cache-profile agent`: 512 MiB host per rank, block 256,
8192 MiB post-allocation reserve on both nodes, in-run snapshots with reach 0, and session
snapshots only without DSpark. Disk and before-tail stay OFF. Native qualification (two Strix Halo nodes over 2x USB4 Gen3, PerfOpt kernel, production packs): prefix tool exactness on both ranks at 6000 and 8192 rows (max rows 4096 / 2048): miss, same-prompt, warm, in-run and byte-suffix resumes equal a fresh prefill (logits and state digest), 16 decoded tokens equal, 0 differences. Serve timing, a fresh process per arm, `--cache-profile off` vs the default in ABBA order over two turns (n = 4 pairs per size), generated text and token IDs identical in all 8 processes: cold 2K miss -0.14 s mean (pairs +0.09 / -0.46 / -0.13 / -0.08 s; not slower), an 8K prompt that shares its 2K prefix with the previous request -3.63 s (-24 %, 15.0 -> 11.2 s mean; every pair faster). Short prompts below one block do not hit.

Use `--cache-profile off` alone for the prefix baseline, or `--no-prefix-cache` to veto the bundle
and all explicit prefix features. Component OFF controls, legacy settings, explicit disk-zero
semantics, effective host/service-directory caps and pending work are listed in
[the DS cache contract](../../engine/serving/prefix-state-cache.md#deepseek-pair-agent-prefix-cache).
The launcher still selects DSpark automatically; implicit sessions never suppress it.
Inspection by `--print-effective-config` reports launcher settings; the executable's startup
`prefix policy` line and `/api/status` report the resolved cache policy after manifest fills.

## Additional qualified two-node code defaults (unset = on, `=0` = off)

Qualified on 2026-10-23 with three interleaved ABA rounds, every run bitwise against its reference, and the
full-workload gate; confirmed in a combined measurement.

| setting | default | qualification effect |
|---|---|---|
| `STRIX_DIST_CRC_PARALLEL`, `STRIX_DIST_HOST_CRC_OVERLAP` | on | hp: R6 -1.37 %, N8 -1.77 %, TTFT 2K/8K/32K -1.15/-1.24/-0.89 % |
| `STRIX_DIST_HOST_CHUNK_BYTES` | 1048576 (`=0`: one piece) | part of hp |
| `STRIX_DS41_ENGRAM_PAYLOAD_CACHE` | on (`_MIB` caps the budget) | TTFT -0.66/-1.17/-1.63 %, decode equal |
| `STRIX_DS41_ENGRAM_F16_PF` | on for prefill Engram pieces of 32..8192 rows except 49..64 (full aligned F16 view; `=0` off, read per call) | 48 x 64 tile with next-slab register prefetch, bit-identical to enqueue_f16_rows; isolated projection 0.86x at 512-4096 rows (1024: 88.1 -> 75.8 ms), 0.87x at 8192, 0.87x / 0.94x at 128 / 256, 0.54x at 32 |
| `STRIX_DS41_ENGRAM_TP` | on for every DeviceForward / BatchForward step of up to 8 rows (decode or prompt; `=0` off, `=r1` single-row steps only, the previous default) | the two ranks split the Engram projection's output rows (same per-row arithmetic) and exchange the halves: R1 -2.1 %, verify windows R2/R4/R6 -1.5/-1.0/-1.0 %, R8 -0.6 %, same tokens/logits/state; needs two ranks and the F8 engram_kv (`STRIX_DS41_F16_AS_F8`), else the full projection on each rank |
| `STRIX_DS41_PREFILL_TILES` | off (`=1`: unsplit tiles for prefill runs of >= `STRIX_DS41_PREFILL_TILES_MIN_ROWS` rows, default 2048) | re-measured on the current kernels, tiles on vs off: fresh-process TTFT 2K/8K/32K +1.0/+3.4/+1.9 %, same-process A/B 4K/8K/16K +0.89/+0.66/+0.82 % (slower); the earlier -0.37/-1.26/-1.13 % predates the current split-K kernels. Same logits either way |
| `STRIX_DS41_ATTN_LEAN` | 3 (`=0` off, `=1` keeps the previous key dequantisation, `=2` the previous QK, softmax and PV) | packed-attention chunk kernels in fewer instructions, same values in the same order; serve TTFT 8,399 / 32,499 / 44,699 tokens -1.9 / -1.4 / -2.4 % (level 2) vs `=0`, logits and generated ids identical; level 3 vs 2: prefill window -7.0 % (32 heads) / -9.2 % (64 heads) per 8-row window, bytes equal |
| `STRIX_DS41_PREFILL_MOE_V2` | on (`=0` off; `STRIX_DS41_PREFILL_MOE_BAND`, default 4) | split IQ2_XXS gate/up and Q2_K down prefill kernels v2, same products and reduction tree; serve TTFT 8,399 / 32,499 / 44,699 tokens -5.9 / -6.9 / -8.0 % vs `=0`, logits and generated ids identical |
| `STRIX_DS41_PREFILL_MOE_V2_Q8` | on (`=0` keeps the v1 split kernels for Q8_0) | v2 split kernels for the Q8_0 shared expert and the Q8_0 x A8 16-row projections, same products and reduction tree; shared expert gate/up -20 / -20 / -32 %, down -26 / -33 / -40 % at 2048 / 4096 / 8192 rows, routed experts unchanged; every non-NaN result and NaN position identical |
| `STRIX_DS41_PREFILL_O_A8_FUSED` | on (`=0` keeps the separate quantization of o) | the exact attention combine writes o's A8 codes, deltas and sums; serve TTFT 8,399 / 32,499 tokens -1.3 / -1.2 % vs `=0`, logits and generated ids identical |
| `STRIX_DS41_PREFILL_ROUTER_V2` | on (`=0` keeps the v1 scores kernel) | prefill router scores on 32-row x 32-expert blocks with the next chunk loaded during the current one, the decode router's chains; router 20.71 -> 5.87 ms per 8192-row call; serve TTFT 8,399 / 32,499 tokens -1.9 / -1.5 % vs `=0`, logits and generated ids identical |
| `STRIX_DS41_DEFER_DECODER` | on (`=0` off) | a non-final server prefill piece skips its decoder cone (the next piece rebuilds what decoding needs; only the first decoder layer's compressed-KV / index entries of every row are kept); serve TTFT 32,499 / 44,699 tokens -3.3 / -5.9 % vs `=0`, logits and generated ids identical |
| `STRIX_DS41_ENGRAM_SPLIT_READ` | on (`=0` off) | a prefill piece reads its first Engram layer's rows first, so that layer no longer waits for the later Engram layers' rows; serve TTFT 8,399 / 32,499 / 44,699 tokens -5.6 / -3.4 / -1.4 % vs `=0`, logits and generated ids identical |
| `STRIX_DS41_ENGRAM_LOOKAHEAD` | on (`=0` off) | the server reads the next prefill piece's Engram rows into the payload cache while the current piece runs; serve TTFT 32,499 / 44,699 tokens -3.3 / -2.8 % vs `=0`, logits and generated ids identical |
| `STRIX_DIST_STAGED_READY_BYTES` | 262144 (`=0`: no readiness views) | needed by the two below |
| `STRIX_DS41_READY_LEAN`, `STRIX_DS41_MOE_READY` | on | Lean ready releases, MoE half sum through ready views |
| `STRIX_DS41_OB_READY_EPILOGUE`, `STRIX_DS41_READY_READS_ONLY` | follow `READY_LEAN` | without the lean form they are the slower previous epilogue |
| `STRIX_DS41_BATCH_PREFETCH` + `STRIX_DS41_PREFETCH_SIDE` | 24 MB, side stream on (`PREFETCH_ROW_MB` 3) | MALL read-ahead on a lowest-priority stream during the exchange waits; N2/N4/N8 -2.1/-4.9/-3.3 % on the Q8 sites |
| `STRIX_DS41_DECODE_PREFETCH` | 2 MB per row, o_b window | R6 -1.8 %, R1 0 on the Q8 sites |
| `STRIX_DS41_TRELLIS_OB_READY` | on | TP trellis o_b through the ready views (one copy+publish kernel into TX, add from RX; no SDMA D2H/H2D or events); with CRC_DEFER R1 -1.6 %, R6 -1.9 %, N8 -0.7 %, same bits |
| `STRIX_DIST_CRC_DEFER` | on | ready-view exchanges compute the TX CRC during the wire and the RX CRC after delivery, each checked through the next trailer (a step still fails on corruption; mixed modes fail at once) |
| `STRIX_DS41_IDX_DIRECT`, `STRIX_DS41_IDX_CAND_FILTER`, `STRIX_DS41_ATTN_SEG_HPW` | on | indexer skipped when the count is <= 512, candidate-only scoring for pool layers above 16K entries, shared heads-per-wave for N8 segments; short R1/R6/N8 -2.0/-2.7/-1.0 %, 128K -0.8/-1.9/-3.3 %, same bits |
| `STRIX_DS41_BATCH_SHARED` | `overlap` (`off`/`0`: before the routed experts; `split` unchanged) | BatchForward shared expert inside the half-sum exchange, N1/N2/N4/N8 -2.8/-3.2/-5.1/-5.2 % in qualification, same bits |

Exact decode launch and staging options (unset = on, `=0` = off each), in-process ABA on both ranks, every arm bitwise
(R1 tokens + logits slices, verify-window tokens, attention states; batch rows and states):

| setting | effect |
|---|---|
| `STRIX_DS41_TOUCH_BATCH` | a prefetch window's read-ahead touches in one launch (up to 16 spans) instead of one per span |
| `STRIX_DS41_TRELLIS_ROW_VIEWS` | multi-row steps pass strided rows straight to the dense-r1 trellis plan (no staging copy) |
| `STRIX_DS41_NORM_ILP` | norm_kernel gathers its loads before the first add / store |
| `STRIX_DS41_R1FUSE_ROUTES` | the router's top-k writes the expert routes (no separate build-routes launch) |
| `STRIX_DS41_COMPRESS_FAST` | compressor kernel in fewer instructions (same chains) |

All five together vs all off: R1 -0.95 % (26.94 -> 26.68 ms/token, 4/4 pairs faster), R6 -0.68 %, batch N8 decode -0.40 % / wall -1.16 %.
Alone vs off, R1: TOUCH_BATCH -0.31 %, TRELLIS_ROW_VIEWS -0.08 %, NORM_ILP -0.35 %, COMPRESS_FAST -0.34 % (R6 -0.59 %), R1FUSE_ROUTES -0.11 %; R6 not slower for any.

Attention-core qualification (in-process ABA x3 per cell, every arm bitwise on both
ranks: tokens, logits slices, attention states; prefill 32K / 400 rows logits + state digest equal):

| setting | default | effect |
|---|---|---|
| `STRIX_DS41_IDX_DIRECT` | on (`=0` off) | count <= 512: no indexer projections / scores / select. Short context R1 -2.0 %, R6 -2.7 %, N8 -1.0 % |
| `STRIX_DS41_IDX_CAND_FILTER` | on (`=0` off) | candidate users read and select only candidate blocks (> 16K entries). 128K R1 -0.8 %, R6 -1.9 % |
| `STRIX_DS41_ATTN_SEG_HPW` | on (`=0` off) | N8 segments share one heads-per-wave form (Hpw 4 instead of 1). With the above: N8 8K -1.1 %, 32K -1.4 %, 128K -3.3 % |

Still off (measured slower or neutral): `STRIX_DS41_PREFILL_SLABS` (8K +0.76 %), `STRIX_DS41_R1FUSE_QNORM_A8`, `STRIX_DS41_GRAPHS` (only 0.21 ms/step of GPU idle waits for the host), CMake `STRIX_EXPERIMENTAL_MALL_Q2_DOWN_NT`.
With the trellis o_b site (below) the o_b product is the trellis plan, so the Q8 o_b ready epilogue does not run there;
the MoE half-sum ready path does.
The read-ahead skips trellis sites (their own pack) and runs only in the Q8 o_b exchange windows, so with the trellis
attention set it has little left to read in decode; it is kept on because it is exact and never slower.

## Current two-node defaults (2026-10-26)

The qualified two-node revision `481aeb0f` keeps these additional defaults. Each was gated on the nodes;
this publication merge does not establish a new device qualification.

| setting | default | override or constraint |
|---|---|---|
| `STRIX_DS41_DENSE_SELECT` | on: exact per-site dense prefill selector | `=0` restores the configured dense forms |
| `STRIX_DS41_ATTN_WINDOW_FOLD`, `STRIX_DS41_ATTN_LIVE_CHUNKS`, `STRIX_DS41_ATTN_TILE_F16` | on: exact attention window folding, live chunk scheduling and half-precision tiles | `=0` disables each independently |
| `STRIX_DS41_ATTN_LDS_SWIZZLE` | on: the f32 attention tiles (4 heads per wave, f16 fallback tiles) permute their 16-byte LDS vectors; the tile choice is unchanged | `=0` off; rank ABA R6 n=5 exact (tokens, logits slice, window tokens, attention state, both ranks) -0.20 % (3/4 faster), R1 neutral; batch ABA N8 n=5 exact, decode median -0.04 % |
| `STRIX_DS41_MOE_PREFILL` | split MoE prefill form 7 | `=0` disables it |
| `STRIX_DS41_SERVE_WARM` | warm the serving paths before READY | `=0` disables it; the default logs and skips decode warm-up if the KV pool cannot hold the dummy slots; explicit `=1` still fails when they do not fit; split decode skips it |
| `STRIX_DS41_DSPARK_CALIB` | on with the warm-up setting outside split decode | explicit `=0` / `=1` overrides it |
| `STRIX_SERVE_OBSERVE_EXPIRY_S` | 60 seconds after an arming request | `=0` keeps observation armed until process exit; `/metrics` and `/api/status?detail=1` renew it |

Approximate BF16 attention and FP16 dense prefill remain opt-in. Their projection, layer range, row threshold,
A8, fast and split-workspace controls are part of the prefix-cache namespace, so incompatible states do not share
cache entries. The optional FP16 dense backend requires CMake `STRIX_DS41_PREFILL_FP16_LT=ON`.

## Default invocation = the measured production command (2026-10-26)

The rank, batch, prefill and serve apps resolve their production environment and flags themselves
(`include/strix/deepseek41_pair_defaults.hpp`), so the plain command

```
strix-deepseek41-serve MODEL.gguf --rank R --transport usb4auto      # rank / batch the same; prefill: --transport usb4-auto
```

runs what the 2026-10-26 qualification measured, when the binaries were built with the trellis library:

```
cmake ... -DSTRIX_DS41_TRELLIS=ON -DSTRIX_DS41_TRELLIS_LIBRARY=<trellis archive> \
  -DSTRIX_DS41_TRELLIS_INCLUDE_DIR=<dir with strix_trellis.h>    # STRIX_DS41_TRELLIS_API defaults to dense_r1
```

Without the library the build still works; the trellis packs are then skipped with one notice. Measured (TTFT 1K 2.098 -> 1.954 s, 8K 11.090 -> 9.757 s; R1 -1.9 %, R6 -2.4 %, N8 -2.2 %;
serve C1 43.58 -> 48.00 tok/s; every run exact):

| item | default | off |
|---|---|---|
| production identity: `STRIX_DS41_OVERLAY=<packs>/c-u6.gguf STRIX_DS41_CB4=1` | on when the pack is found | `STRIX_DS41_OVERLAY=0` |
| trellis mix-B at every dense site: `STRIX_DS41_TRELLIS=<packs>/ours-K4 STRIX_DS41_TRELLIS_MIX=head=<packs>/ours-K5 STRIX_DS41_TRELLIS_SITES=all STRIX_DS41_TRELLIS_DENSE_FULL=1 STRIX_DS41_TRELLIS_HEADOUT=1` (not the EP prefill app) | on when both packs are found and the build links the dense_r1 trellis library | `STRIX_DS41_TRELLIS=0` |
| rank / batch `--tp-attention --engram-direct` | on (TP attention with two ranks) | `--no-tp-attention`, `--engram-mapped` |
| prefill `STRIX_DS41_EXPERT_TP=0` (EP prefill) | on | an explicit value |
| serve `--peer-timeout 120 --prefill-path device --slots 8 --prefill-rows auto` (largest piece of 16384 / 8192 / 4096 rows both ranks admit after the weights and the drafter load (32768-row pieces measured slower than 16384: 8.01 / 28.30 / 39.94 vs 7.94 / 27.44 / 38.98 s at 8,399 / 32,499 / 44,699 tokens, same logits and tokens), else 2048 when it fits; when no piece fits both ranks refuse before READY with the numbers (pass `--prefill-rows N` to override); Pair, context 49152, fresh prompts: 8,399 tokens 14.57 -> 10.34 s, 32,499 tokens 60.8 -> 39.1 s (16384 vs 2048 rows), same logits and tokens) | on | the explicit flag (`--prefill-rows N`) |
| serve `--dspark <model dir>/*-DSpark-MXFP4.gguf --dspark-vocab auto:<packs>/vocab` | on when the draft (and the vocabulary's `core48k.bin` / `core32k.bin`) are found; off with `STRIX_DS41_BATCH_ROWS16=1` or `--prefix-session-snapshots` | `--no-dspark`, `--dspark-vocab full` |

`<packs>` is `STRIX_DS41_PACKS` when set, else `<model dir>/packs`, else the model's directory (`STRIX_DS41_PACKS=0`:
no packs). An argument on the command line wins over the default, a variable already in the environment is never changed,
and a missing precondition drops its item with one `deepseek41 defaults:` line on stderr. `--print-effective-config`
prints the resolved environment, arguments and these seven session lines before any device work:

```text
session cache off
session ram_bytes RAM_BYTES
session max_parked MAX_PARKED
session disk off
session disk_bytes DISK_BYTES
session directory CACHE_DIRECTORY
session reason OFF_REASON
```

All four apps print these session lines, including under `STRIX_DS41_BASELINE=1`.
Serve also reports `session cache: off (OFF_REASON)` on stderr when sessions are
not admitted, including a plain invocation. With the current empty qualification
table, the reason is `route not yet GPU-qualified: tp2-http-target` or
`route not yet GPU-qualified: tp2-http-batch-dspark`, according to the resolved
producer. Inspection includes the same message as a `notice session cache: off
(OFF_REASON)` line. Under the DeepSeek baseline the reason is `baseline switch`;
an explicit cache off switch reports `off switch`.

Master switch `STRIX_DS41_BASELINE=1`: nothing above is added, and the code defaults adopted on 2026-10-25/26 are set
to `=0` (`STRIX_DS41_SHARED_GROUPED`, `SHARED_DOWN_FUSED`, `MOE_PREFILL`, `DENSE_SELECT`, `ATTN_WINDOW_FOLD`,
`ATTN_LIVE_CHUNKS`, `ATTN_TILE_F16`, `SERVE_WARM`, `DSPARK_CALIB`) unless set explicitly: the plain command as it ran
before, without those items. The CPU test `strix-deepseek41-pair-defaults-tests` checks the resolved command against the receipt
and every off switch.

## Product profiles

`--profile default|fast|quality` (every pair app, same value on both ranks) resolves before the defaults on this page.
`default` changes nothing on this page. `fast` adds qualified bit-changing options only; today its two rows (FP16
sparse attention, relaxed MoE epilogue) are pending and not applied. `quality` = `default` (no quality pack). Rows,
precedence and receipts: [DeepSeek pair product profiles](../../optimizations.md#product-profiles-pair).

## Transport and USB4 layout (P17.4)

- Every app takes `--transport usb4gpu` (prefill: `usb4-gpu`); the serve manifest kind is `usb4_gpu_auto`.
  - With gpu_direct streams (layout 2bq: data0 host, data1 gpu_direct + gpu_poll=1, latency_qos_us -1) they build the device exchange. Route A is `65536:host,1048576:gpu:2,inf:mix:2:35`, with prepost.
  - Without gpu_direct streams (layout 1) they print "gpu-direct unavailable ... host path only" and run the host path. The bits are the same in both layouts.
- Serve quiesces both ranks when it goes idle after work, so no posted GPU receive outlives the driver's 30 s DMA deadline.
- 2bq vs layout 1, the same binary and kernel (P17.4):
  - faster: R6 window +2.9 %, prefill 8K +1.6 %, serve 8K TTFT -10 %, serve C2..C8 +4..7 %;
  - slower: R1 -1 % (2 host streams instead of 4);
  - equal: batch.
- 2b at latency_qos_us 0 is slower everywhere: do not use it.
- `STRIX_DS41_OB_SPLIT=K` (o_b exchange in K column chunks overlapping the products) is exact but slower on the host path (R6 +1.2 / +3 / +5 ms at K=2 / 4 / 8). It stays opt-in.

## Production identity (environment of both ranks)

```
STRIX_DS41_OVERLAY=<requant>/ov/c-u6.gguf STRIX_DS41_CB4=1
STRIX_DS41_TRELLIS=<packs>/ours-K4 STRIX_DS41_TRELLIS_MIX=head=<packs>/ours-K5
STRIX_DS41_TRELLIS_SITES=all STRIX_DS41_TRELLIS_DENSE_FULL=1 STRIX_DS41_TRELLIS_HEADOUT=1
```

Attention sites on trellis in the qualified two-node configuration: every mix-B dense class incl. q_a / kv / q_b / o_a / o_b
reads its K4 pack. Needs the build `STRIX_DS41_TRELLIS_API=dense_r1` (the CMake default) linked against a trellis library with the dense-r1 and grouped GEMV APIs
(grouped wo_a GEMV, dense-r1 q_a + kv pair, cold schedules). `STRIX_DS41_TRELLIS_OA_GROUP` (grouped o_a) and
`STRIX_DS41_TRELLIS_QAKV_PAIR` (q_a + kv in one launch) are on by default and bit-identical to their off forms (=0).
Measured (pair, ABA, same binary, only the env differs): R1 34.7 -> 31.15 ms/token (-10.2 %), R6 window 63.17 -> 60.59 ms
(-4.1 %), N8 85.38 -> 83.56 ms/step (-2.1 %), every run exact against its reference. The previous production set
`STRIX_DS41_TRELLIS_SITES=shexp,head` keeps its bits on this build. Quality: mix-B with all dense classes on
K4 is the configuration of the trellis quality report (full set KL 0.0446, paired vs c-u6 -0.0001).

The earlier trellis mix-B (shexp + head sites, K4 dense / K5 head) passed its quality gate
(full-set KL equal to the 6.5-bit format). Either trellis library (CMake `STRIX_DS41_TRELLIS_API` v12 or splitk04)
gives the same bits. Serve: `--prefill-path device --dspark <DSpark.gguf> --dspark-vocab auto:<vocab>`.

## HTTP service boundaries

Device serving now defaults to a 250 ms soft service target and stable short-first
admission/waiting-prefill selection. These additions still require both-rank GPU and
mixed-load qualification; the P17.1 figures above do not qualify them. See the
[server contract and legacy-budget precedence](../../engine/serving/serving.md#deepseek-v41-pair-scheduler).
While no request decodes and no other prompt waits for its prefill, the pending prefill runs the rest of its
piece at once instead of 250 ms units (same bits; a request that arrives meanwhile waits for that piece);
`STRIX_DS41_SERVE_IDLE_WHOLE=0` keeps the bounded units while idle.
Use `STRIX_DS41_SERVE_SERVICE_MS=0` for coarse row budgeting and
`STRIX_DS41_SERVE_SHORT_FIRST=0` for FIFO; restart both ranks when changing settings.

Conversation checkpoint/park/resume is a production [session-cache feature](session-cache.md).
Automatic admission requires a GPU-qualified route; the qualification table is
currently empty. Explicit rank commands run the qualification gate. Sessions
honour only `STRIX_DS41_BASELINE`; generic baseline settings retain their existing
DeepSeek behavior.
