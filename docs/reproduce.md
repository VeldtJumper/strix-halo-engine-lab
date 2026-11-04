# Reproduce engine measurements

The kit records token-identified workloads and produces issue-ready JSON and Markdown.
This guide describes the current kit, rather than asserting reproduction of every README campaign.

```bash
bash scripts/bench/bench.sh --help
bash scripts/bench/bench.sh list --json
bash scripts/bench/bench.sh list
bash scripts/bench/bench.sh check --engine qwen --only exactness --suite quick --json
bash scripts/bench/bench.sh run --engine qwen --only exactness --suite quick --json
bash scripts/bench/bench.sh run --engine qwen --suite quick --dry-run
```

## Requirements

Use Bash 4+, Python 3.10+ (`python3` by default), Git, GNU `timeout` and `sha256sum`.
Set `BENCH_PYTHON=python` if that is the installed command. The core client uses the standard
library; GLM/MiniMax also require Python `tokenizers` and the model's local `tokenizer.json`.
Fetching needs `curl`, `hf` or `huggingface-cli`; Qwen merging needs `llama-gguf-split`.
Dry-run validates arguments and prompt files without executing GPU binaries, downloads or SSH.

Qwen needs one Linux Strix Halo (Ryzen AI Max+ 395, Radeon 8060S `gfx1151`, 128 GB unified
memory). DeepSeek, GLM and MiniMax pairs need two such nodes. Install working ROCm
(the documented setup uses core-10.1), with `rocminfo`, `hipconfig --version` and readable,
writable `/dev/kfd`. Reserve memory for model, workspace, drafts and driver staging; use fast
storage and avoid paging. ROCm release and HIP versions are recorded separately; no particular
release number is enforced.

Pairs require the [project USB4 driver](https://github.com/hyukudan/strix-halo-usb4-driver),
loaded `thunderbolt_stream`, accessible `/dev/tbstream*`, `setsid`, and SSH `BatchMode=yes`
access to a peer with Bash and Python. The reported setup uses two direct cables, 40 and
20 Gb/s; device count does not prove independent links. The kit changes no permissions,
modules, clocks, power modes or governors.

Standalone `deepseek-rtx` needs CUDA, `nvidia-smi`, substantial host memory/storage and an NVIDIA
GPU. The README hardware is an RTX PRO 6000 Blackwell, 96 GB, Windows WDDM. Use Git Bash/MSYS,
native Python and the CUDA/MSVC toolchain. Replica receipt admission is Windows-only in this
checkout: Linux does not read model size/mtime, so `check` rejects it despite separate CUDA
prefill support.

Commands use repository-relative example directories. `node-b` and `user@node-b` are synthetic
SSH placeholders. Relative asset/output paths resolve from the repository root even when the
entry point is invoked elsewhere.

## Build each engine

Install CMake >=3.24, Ninja, a C++20 compiler and ROCm development files. The OpenSSL option
needs OpenSSL development files. Set `CMAKE_HIP_COMPILER` to the installed ROCm compiler if
CMake cannot discover it. Common HIP configuration and Qwen targets:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DSTRIX_ENABLE_HIP_RUNTIME=ON -DSTRIX_USE_OPENSSL_SHA256=ON \
  -DSTRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME=ON \
  -DSTRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE=ON -DBUILD_TESTING=ON
ninja -C build strix-tokenize strix-hip-decode-token
```

Configure both pair nodes identically and use identical executable bytes. Build the adopted
[strix-trellis](https://github.com/hyukudan/strix-trellis) headers/archive first, retaining
provenance and notices. DeepSeek requires dense-r1 and grouped GEMV APIs. `TRELLIS` below is
a relative dependency directory; the archive must implement these APIs, not merely have the
example filename.

```bash
cmake -S . -B build \
  -DSTRIX_FAMILY_DEEPSEEK41=ON -DSTRIX_DS41_TRELLIS=ON \
  -DSTRIX_DS41_TRELLIS_API=dense_r1 \
  -DSTRIX_DS41_TRELLIS_LIBRARY=TRELLIS/build-qualification-v03/libstrix_trellis.a \
  -DSTRIX_DS41_TRELLIS_INCLUDE_DIR=TRELLIS/include
ninja -C build strix-tokenize strix-deepseek41-rank strix-deepseek41-dspark-rank \
  strix-deepseek41-prefill strix-deepseek41-serve
```

CMake accepts `v12`, `tp03`, `splitk04`, `dense_r1`; this kit requires **dense_r1**,
matching CMake's current default `dense_r1`. A conflicting cache value fails precheck; missing fields produce
warnings, not proof of build identity.

GLM and MiniMax have real targets. Their trellis directory options expect
`include/strix_trellis.h` and `build-qualification-v03/libstrix_trellis.a` inside the supplied
tree. Missing files can leave the engine targets absent. Reconfigure the common HIP build:

```bash
cmake -S . -B build -DSTRIX_FAMILY_GLM5=ON -DSTRIX_GLM5_TRELLIS_DIR=TRELLIS
ninja -C build strix-glm5-rank
cmake -S . -B build -DSTRIX_FAMILY_MINIMAX=ON -DSTRIX_MINIMAX_TRELLIS_DIR=TRELLIS
ninja -C build strix-minimax-rank strix-minimax-serve
```

For RTX first build tierstream's CUDA static library at `TIERSTREAM/build-cuda/tierstream`
with the platform static-library suffix. In an x64 MSVC developer environment, with CUDA
supporting architecture 120 (Bash syntax):

```bash
cmake -S . -B build-rtx -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl \
  -DSTRIX_CUDA_DS41_ONLY=ON -DSTRIX_ENABLE_CUDA_DS41=ON \
  -DSTRIX_FAMILY_DEEPSEEK41=ON -DSTRIX_ENABLE_HIP_RUNTIME=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRIX_TIERSTREAM_DIR=TIERSTREAM
ninja -C build-rtx strix-cuda-ds41-replica strix-cuda-ds41-prefill
cmake -S . -B build-host -G Ninja -DCMAKE_CXX_COMPILER=cl
ninja -C build-host strix-tokenize
cp build-host/strix-tokenize.exe build-rtx/
```

CUDA-only configuration returns before creating the tokenizer. Build/copy it separately.
For Linux omit `-DCMAKE_CXX_COMPILER=cl` and `.exe`, retaining the replica admission blocker.
See [standalone preparation](models/deepseek-v4.1/rtx-standalone.md) for tierstream and receipts.
CPU tests do not qualify native builds.

## Models and provenance

All mirrors below are under `huggingface.co/hyukudan`. Revisions identify upstream conversions
recorded by the fetcher, not a mirror `main` pin. Retain original licenses, conversion notices
and manifests. Unpinned sizes must be read from actual manifests/files; bpw is not a byte size.

| Engine | Mirror repository | Upstream source + revision | Files / layout | Size | License notes |
|---|---|---|---|---|---|
| Qwen | [Qwen3.8-Flash-Next-GGUF-mirror](https://huggingface.co/hyukudan/Qwen3.8-Flash-Next-GGUF-mirror) | `unsloth/Qwen3.8-Flash-Next-GGUF@38bb39ee` | `UD-Q4_K_XL/*.gguf`, merged as `Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf`; public base `MTP/*.gguf` | Target about 111.33 GB; default distilled sidecar 2,786,568,256 bytes | Qwen Community License; base MTP is not distill-g. |
| Qwen GSQ | [Qwen3.8-Flash-Next-GSQ-hybrid-Q2](https://huggingface.co/hyukudan/Qwen3.8-Flash-Next-GSQ-hybrid-Q2) | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF@ed59f920` plus project composition recipe | `hybrid-q2.gguf`; same distilled draft/vocabulary prerequisites | Unpinned | Qwen Community License; separate numerical/quality identity. |
| Qwen YaRN MTP bundle | Project recipe in this repository | Expanded distill-g pins and explicit 1M budgets | [`recipes/qwen-distill-g-yarn.mtp-bundle.json`](../recipes/qwen-distill-g-yarn.mtp-bundle.json); copy beside the GGUF as `<model>.mtp-bundle.json` | JSON only; sidecar and the `draft-vocab/` directory supplied separately | Preserves draft provenance; calibration caches locally on first use. |
| DeepSeek pair/RTX | [deepseek-v4.1-flash-gguf-mirror](https://huggingface.co/hyukudan/deepseek-v4.1-flash-gguf-mirror) | `kyuz0/deepseek-v4.1-flash-gguf@20408f69` | `DeepSeek-V4.1-Flash-1017-Q2.gguf`, `DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf` | 365,713,686,528 + 7,966,294,016 bytes | MIT; preserve notices. |
| DeepSeek pair overlay | [DeepSeek-V4.1-Flash-1017-strix-dense-U6](https://huggingface.co/hyukudan/DeepSeek-V4.1-Flash-1017-strix-dense-U6) | Project conversion in same repository; immutable revision not recorded by fetcher | `c-u6.gguf`, `c-u6.json` | Unpinned | DeepSeek MIT; retain conversion provenance. |
| DeepSeek pair packs | [DeepSeek-V4.1-Flash-1017-strix-trellis-EXL3](https://huggingface.co/hyukudan/DeepSeek-V4.1-Flash-1017-strix-trellis-EXL3) | Project packs in same repository; immutable revision not recorded by fetcher | `ours-K4/*`, `ours-K5/*`, `mixB.txt`, `aux/vocab/*.bin`, `aux/draft-keep-stats.jsonl`; copy vocabulary to `vocab/` for pairs | Unpinned | DeepSeek MIT; preserve adopted pack/code notices separately. |
| GLM | [GLM-5.3-Flash-exl3-mirror](https://huggingface.co/hyukudan/GLM-5.3-Flash-exl3-mirror) | `turboderp/GLM-5.3-Flash-exl3@2a30229e`, 4.05 bpw | Under `glm/`: `*.safetensors`, `*.json`, optional `*.txt`, `*.jinja`, `*.model`, including tokenizer | Exact bytes unpinned (4.05 bpw) | MIT; retain model/conversion notices. |
| MiniMax | [MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw-mirror](https://huggingface.co/hyukudan/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw-mirror) | `Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw@c33e7a66` | Under `minimax/`: same EXL3 patterns plus `dspark/*` | Exact bytes unpinned (3.06 bpw) | MiniMax Community License, non-commercial by default; attribution **Built with MiniMax M3.1-Flash**. Review commercial/distribution permission. |
| MiniMax EXL3 draft | same MiniMax mirror | Project packs: the `dspark/` draft re-quantized to EXL3 4-bit (fc included) with `src/minimax/dspark/ref/quant_draft.py --K 4 --fc` | Under `minimax-drafts/`: `dspark-exl3-K4fc-sh/` (production, scaled-Hessian metric) and `dspark-exl3-K4fc/` (previous pack); each `config.json`, `model-dspark-exl3.safetensors`, `model.safetensors.index.json` | 1,134,088,202 bytes per weight file; sha256 `341142bb9d2b9cbe35ccdd5c32f9b21ca374cdc268f988e5ac843becc03f941d` (K4fc-sh) and `29b86c0e9df79eefce92146f498ca720a7e0703301cb44a720a07ab1566872d2` (K4fc), all files in the mirror's `SHA256SUMS` | Same MiniMax Community License and attribution as the MiniMax row. |
| MiniMax draft provenance | [MiniMax-M3.1-Flash-DFlash2-mirror](https://huggingface.co/hyukudan/MiniMax-M3.1-Flash-DFlash2-mirror) | `incoai/MiniMax-M3.1-Flash-DFlash2@ac07a7c1` | Draft source; fetcher selects `dspark/*` from main MiniMax mirror rather than this separate mirror | Unpinned | Retain MiniMax and draft conversion terms. |

The DeepSeek pair layout under `--model-dir models` is:

```text
models/
  DeepSeek-V4.1-Flash-1017-Q2.gguf
  DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf
  c-u6.gguf
  ours-K4/
  ours-K5/
  vocab/core48k.bin
  vocab/core32k.bin
```

The kit sets `STRIX_DS41_PACKS` to that model directory on each rank. The resolver otherwise
searches the model's `packs/` subdirectory and then its directory. The draft stays next to the
model. The fetcher retains `aux/vocab/` with its manifest and copies verified vocabulary bytes
to `vocab/`. RTX uses the same target/draft filenames, plus `ds41-draft-vocab/` next to the
draft; its fetch recipe copies verified vocabulary there. RTX also requires the deployment
receipt and state below. Qwen continues to use its separate `draft-vocab/` directory.

For explicit CPU bundle integrity checks, see [release asset verification](operations/release-assets.md).

Qwen additionally requires `mtp-Qwen3.8-Flash-Next-shared-distillG-Q8_0.gguf` and `draft-vocab/`.
The fetcher explicitly does not supply the locally distilled sidecar; never rename base MTP
as distill-g. Even `plain` precheck requires these assets. RTX additionally needs deployment
`MODEL.receipt.json` and exported `STATE.bin`, not downloaded or shipped. The receipt binds
model hash, size and modification time; state must be a valid attention-state export.

Mirrors may still be uploading. Missing manifest groups, download failures and hash mismatch
fail closed; diagnostics identify upstream fallback. No automatic source switch occurs.
Fetch the listed upstream revision manually if necessary, arrange the same layout and verify
hashes, preserving attribution. Project-specific overlays/drafts need their publisher; raw
checkpoints are not replacements for converted assets.

## Fetching and hashing

```bash
bash scripts/bench/fetch-models.sh --engine qwen --model-dir models --dry-run
bash scripts/bench/fetch-models.sh --engine deepseek --model-dir models
bash scripts/bench/fetch-models.sh --engine glm --model-dir models
bash scripts/bench/fetch-models.sh --engine minimax --model-dir models
bash scripts/bench/fetch-models.sh --engine qwen --profile gsq --model-dir models
bash scripts/bench/bench.sh hash-models --engine deepseek --model-dir models
```

The MiniMax EXL3 draft pack is not fetched by the script. Fetch it once and point rank or
serve at it with `--dspark-exl3` (the bench kit reads `BENCH_MINIMAX_DRAFT_EXL3`):

```bash
hf download hyukudan/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw-mirror \n  --include 'minimax-drafts/dspark-exl3-K4fc-sh/*' --local-dir models
BENCH_MINIMAX_DRAFT_EXL3=models/minimax-drafts/dspark-exl3-K4fc-sh \n  bash scripts/bench/bench.sh run --engine minimax --peer user@node-b --suite quick
```

Fetch options: required `--engine` (five names and pair aliases), required `--model-dir`,
`--profile` (default `default`; `gsq` changes Qwen selection), `--dry-run` (off), `-h/--help`.
It downloads each mirror's `SHA256SUMS` at the revision pinned in `scripts/bench/lib/fetch_models.py`, selects manifest members, retains LICENSE/README
files and verifies selected bytes. Qwen merging checks a pinned hash. Byte-identical in-tree
vocabulary files may be reused. Manifest integrity is not independent publisher authentication.

`hash-models` fully hashes assets without GPU execution and emits
`{schema_version:1, engine, models_verified:true, model_files:[...]}`. File records have
`role`, basename `name`, `bytes`, `sha256`, `published_sha256` (null if unpinned). Known pins
in `kit.py` are enforced. Directories are recursive, excluding hidden components. First scans
can read hundreds of GB per node.

The private cache is `.bench-sha256-cache.json` in the model directory, falling back to the
XDG cache's `strix-bench` directory. Keys contain resolved path, bytes, nanosecond mtime and
inode. Default `--verify-models cached` reuses unchanged keys; `full` rereads; `skip` emits
null measured hashes and `models_verified:false`. Skip cannot qualify content identity;
Qwen MTP still obtains a real target hash for admission. Caches contain private paths and
must never be attached. Cached verification trusts metadata; use full after uncertain changes.
Pair inventories are hashed concurrently.

## CLI parameters and examples

Actions: `list` checks local availability; `check` checks prerequisites and pair identity;
`run` executes; `hash-models` hashes assets; `report DIR` validates `DIR/results.json` and
rewrites `DIR/results.md`. `--help` prints help. There is no implicit selected engine.

| Parameter | Default and meaning |
|---|---|
| `--engine E` | Required for check/run/hash-models: `qwen`, `deepseek`, `glm`, `minimax`, `deepseek-rtx`, `nemotron`, `all`. Aliases: `deepseek-pair`, `glm-pair`, `minimax-pair`. All dispatches sequentially. |
| `--suite quick\|full` | `full`; explicit parameters override suite defaults. Neither suite enables serve. |
| `--contexts LIST` | Quick `1k`; full Qwen `1k,4k,8k,16k,32k`, DeepSeek pair `1k,8k,16k,32k`, others `1k,8k,32k`. Positive integers, comma separated; `k` = 1024. |
| `--gen-tokens N` | Quick 128; full GLM 64, RTX 128, others 256. Minimum 2. |
| `--repeat N`, `--repeats N` | Quick 1, full 3; positive measured repeat count. |
| `--warmup N` | Quick 0, full 1; nonnegative discarded repeats. |
| `--concurrency LIST` | `1,4,8`; each 1..8, HTTP client workers and pair server slots. |
| `--prompts kit\|FILE` | `kit`: eight original prompts. JSON array or object with `prompts` array, rows `id`, `category`, `text`; or one nonempty UTF-8 text line per prompt. IDs unique `[A-Za-z0-9_-]+`. |
| `--profile P` | Qwen/DeepSeek/MiniMax/RTX/Nemotron `default`; GLM **r1b**. Allowlists below. |
| `--only LIST` | `decode,prefill,exactness`; subset of these plus `serve`. Either decode or exactness schedules decode; omitting exactness clears its final array. |
| `--serve` | Off; adds serve to only. GLM/RTX record a skip. |
| `--bin-dir DIR` | `BENCH_BIN_DIR` or `build`; all selected-engine binaries required, including for reduced work. |
| `--model-dir DIR` | `BENCH_MODEL_DIR` or `models`; all profile assets required. |
| `--peer HOST` | None; explicit SSH destination required for pairs. |
| `--peer-repo DIR` | `BENCH_PEER_REPO` or local repository root, on peer. |
| `--peer-bin-dir DIR` | `BENCH_PEER_BIN_DIR` or local binary directory. |
| `--peer-model-dir DIR` | `BENCH_PEER_MODEL_DIR` or local model directory. |
| `--timeout SECONDS` | `BENCH_TIMEOUT` or 3600; positive whole-run deadline including hashing, tokenization, HTTP readiness and teardown. |
| `--verify-models full\|cached\|skip` | `cached`; integrity caveats above. |
| `--out DIR` | New `bench-results/ENGINE-UTC_TIMESTAMP`; existing directory refused. All uses per-engine children under explicit out. |
| `--json`, `--md` | Off; mutually exclusive for ordinary processing. Supported document endpoints below. |
| `--dry-run` | Off; preview only; no result document even with json/md. |
| `--write-refs` | Off; candidate file in output, never edits shipped refs. |
| `--help` | Prints help and exits. |

Qwen profiles: `default`, `plain`, `baseline`, `gsq`, `mtp-k1` through `mtp-k8`. Default/GSQ use depth 7;
depth-specific profiles select corresponding depth/window limits. MTP also runs plain decode
and per-prompt/context calibration. DeepSeek: default runs greedy/spec, `greedy` only greedy,
`dspark` only spec, `baseline` greedy with `STRIX_DS41_BASELINE=1`. MiniMax has the same allowlist, but dspark still schedules greedy sessions
as well as specs; `baseline` uses `STRIX_MM_BASELINE=1` and greedy sessions. GLM: `r1b`, `r1`,
`mtp` (r1b identity, native `--spec-run 7`), `baseline` (`STRIX_GLM_BASELINE=1`).
Qwen `baseline` runs plain decode/prefill with `--baseline`; RTX: `default` or `baseline`
(the replica receives `--baseline` after measurement arguments). `BENCH_PORT` defaults to 18880; `BENCH_SSH_TIMEOUT` to 60 seconds for bounded
SSH operations (connection timeout 10 seconds). These are environment controls, not CLI flags.
Inherited `STRIX_`, `HIP_`, `HSA_`, `ROCR_` overrides are rejected. Pair apps resolve their
published production defaults. The kit supplies pack locations, measurement controls, rank/transport
and selected profile off switches. Qwen records `--print-effective-config` on rank 0 before
every launch, including plain decode, MTP calibration/decode, prefill and HTTP serve. MiniMax rank resolves its draft directory to `MODEL_DIR/dspark`; serve still needs
`--dspark MODEL_DIR/dspark`. GLM r1b is the plain default; r1 adds `--identity r1`, MTP adds
`--spec 1 --mtp 1 --spec-run 7`. Baseline switches turn off the resolver's adopted items,
while explicit measurement capacities and TTFT modes still apply.

```bash
bash scripts/bench/bench.sh run --engine qwen --suite quick
bash scripts/bench/bench.sh run --engine deepseek --peer user@node-b --suite quick
bash scripts/bench/bench.sh run --engine glm --peer user@node-b --profile r1 --suite quick
bash scripts/bench/bench.sh run --engine minimax --peer user@node-b --suite quick
bash scripts/bench/bench.sh run --engine deepseek-rtx --bin-dir build-rtx --suite quick
bash scripts/bench/bench.sh run --engine nemotron --profile default --suite quick --only exactness,serve
# One-context quick check; serve sweep; available engines
bash scripts/bench/bench.sh run --engine qwen --contexts 1k --only prefill --repeat 1 --warmup 0
bash scripts/bench/bench.sh run --engine minimax --peer user@node-b --only serve --concurrency 1,4,8
bash scripts/bench/bench.sh run --engine all --peer user@node-b --suite quick --out bench-results/all --json
bash scripts/bench/bench.sh report bench-results/example
```

All uses shared model/binary directories and profile override; leave profile unset for
engine defaults. Availability checks presence, not hardware/admission. Present incompatible
engines can fail; pairs without peer skip. All skipped is successful dispatch, not inference.

## Workloads, loads and wall time

See [prompt format](../scripts/bench/prompts/README.md). Eight original prompts cover English
and Spanish chat, code, JSON, math, prose, summary and SQL. Raw completions use no inferred
chat template. GGUF uses `strix-tokenize`; EXL3 uses local tokenizers without automatic special
tokens. Context inputs repeat token IDs and truncate exactly; prefill seed is the first
selected prompt. Source bytes and constructed token lists are hashed.

Qwen starts a fresh process per plain sample, calibration, spec sample and prefill sample;
production prefill uses the plain published command:

```bash
strix-hip-decode-token MODEL.gguf $(cat prompt.ids) --generate 1
```

The kit adds no prefill environment or organization flags. The resolver chooses T and context,
including automatic longer-context acknowledgement; no explicit context is required for production
long prompts. Both inspection and each measured/warmup run must print
`Qwen auto-prefill: automatic T... context=...`. The kit records T, context and the full admitted
notice in each prefill measurement's `auto_prefill` list. A declined line
`Qwen auto-prefill declined: <reason>; using serial prefill` fails closed with the reason;
a missing admission also fails. Diagnostic variables such as `STRIX_PREFILL_LOGITS_DIGEST`,
`STRIX_RTX_EXPORT_STATE`, `STRIX_QRTX_STEP_DUMP` and `STRIX_PROMPT_BATCH_ROWS` are never injected.
FP16 prefill has its own numerical identity; inspect the resolved profile pair and any arithmetic
fallback. MTP and serve retain explicit tiled commands and do not receive automatic fast prefill.
`--profile baseline` adds `--baseline`, skips MTP and measures a separate historical prefill
population without requiring auto admission; long baseline prompts receive explicit context.
GSQ production prefill cannot pass the auto admission gate on this revision; use decode-only GSQ.

DeepSeek runs `strix-deepseek41-dspark-rank` for greedy and DSpark decode:
one load per mode, with every context-expanded prompt, warmup and repeat in a prompt file.
Native generation-loop rates exclude prefill and HTTP overhead
(`timing_scope: native`); full token hashes, counts and plain/spec equality are recorded.
The first output token comes from prefill, so the native rate times `generate - 1` steps.
README greedy ms/token rows are converted to tok/s, and DSpark tok/s rows are compared in
`results.md`, with the workload-differs note. Before decode the kit calls
`strix-deepseek41-rank MODEL.gguf --rank R --transport usb4auto --print-effective-config`
on both nodes. Only its resolved environment is transferred to the native decode app;
`--tp-attention --engram-direct` are explicit production flags accepted by that app's parser
(omitted for baseline). Decode also specifies `--draft split --vocab auto:MODEL_DIR/vocab`.
The optional `--serve` sweep uses the published serve app and its own resolver, records
`timing_scope: http-end-to-end`, and compares only with README serve rows. The kit then runs
one dedicated EP prefill load per context using
`--ttft-timing --ttft-warmup 0` and explicit repeats. GLM batches raw short decode prompts
independently of context sweep, then prefill in another load (default gemv TTFT). GLM MTP
batches the prompt list through the current SPV1 parser. Its native equality/FNV receipts lack
full token IDs: timings are recorded, kit exactness is explicitly skipped and candidate reference
writing is refused. MiniMax
puts sessions/specs into one load, sessions first. RTX packs eight 96-token inputs/repeats
into TSV, with `chats0,schats0,chats0` in one replica load, then one exact prefill load per
context (1024-row passes, auto budgeting, hashes).

| Engine, default profile and decode/prefill/exactness | Quick loads | Full loads | Rough planning wall time quick / full |
|---|---:|---:|---|
| Qwen | 25 | 380 | 15-60 min / 3-12 h |
| DeepSeek pair | 3 paired launches | 6 paired launches | 5-20 min / 1-4 h |
| GLM r1b (r1 same count) | 2 paired launches | 2 paired launches | 5-30 min / 15-90 min |
| MiniMax | 1 paired launch | 1 paired launch | 5-30 min / 1-4 h |
| RTX | 2 | 4 | 5-20 min / 20-90 min |

DeepSeek quick uses exactly 3 paired model loads per rank: greedy decode, DSpark decode and
1K prefill, plus 6 host-only config-print calls (3 on each node). Plan 5-20 minutes; config
printing performs no device work and adds no model loads. Optional serve adds one load.

These broad scheduling estimates assume cached hashes, fast storage, admitted hardware and
no paging. They are not measured kit receipts or qualification evidence. First hashing,
cold loads, context processing and throttling add time. Increase the default 3600-second
timeout for full runs, especially Qwen. Actual `wall_s`/`model_loads` are recorded. Loads count
local inference/server launches, excluding tokenizers; each pair launch loads both nodes.

For P prompts, C contexts and R=warmup+repeat, Qwen MTP loads are `P*C*(2*R+1)+C*R`,
plain `P*C*R+C*R`; DeepSeek default `2+C`, single mode `1+C`; GLM 2; MiniMax 1; RTX `1+C`.
Removing work removes launches. Serve adds one persistent load for Qwen/DeepSeek/MiniMax,
none for GLM/RTX. Default C1/C4/C8 sweeps send eight requests per concurrency per round:
24 quick; 96 full including 24 discarded warmups. Budget minutes to tens of minutes plus
startup depending on tokens/queueing. Serve-only loads are 1/1 for supported engines in
quick/full and 0/0 with skip for GLM/RTX.

HTTP waits for a listener receipt and `/health`, then uses `/v1/completions`, token-ID inputs,
temperature 0, streaming and usage. Aggregate rate divides completion tokens by whole batch
wall time including prefill/queueing. TTFT is first nonempty content. Usage must match IDs;
`http-c*.json` records hashes/timings. Native decode/TTFT timings exclude load. Optional serve
rates include prefill and HTTP overhead (`timing_scope: http-end-to-end`); whole-run wall
includes load. Discarded warmups still execute and load according to each engine's plan.

## Expected numbers and exactness

These are all rows of [expected.json](../scripts/bench/expected.json), not new kit measurements.

| Engine | Prompt / mode | Expected | Tolerance |
|---|---|---:|---:|
| qwen | code / spec | 59.59 tok/s | 10% |
| qwen | prose / spec | 55.7 tok/s | 10% |
| qwen | prefill-1024 / ttft | 1.12 s | 15% |
| qwen | prefill-32768 / ttft | 22.95 s | 15% |
| deepseek-pair | code / greedy | 34.5 tok/s | 10% |
| deepseek-pair | code / spec | 70.6 tok/s | 10% |
| deepseek-pair | prose / spec | 41.9 tok/s | 10% |
| deepseek-pair | prefill-1024 / ttft | 2.18 s | 15% |
| deepseek-pair | prefill-8192 / ttft | 10.78 s | 15% |
| deepseek-pair | prefill-32768 / ttft | 39.6 s | 15% |
| deepseek-pair | http-c1 / aggregate-wall | 39.88 tok/s | 10% |
| deepseek-pair | http-c4 / aggregate-wall | 66.53 tok/s | 10% |
| deepseek-pair | http-c8 / aggregate-wall | 74.25 tok/s | 10% |
| deepseek-rtx | code / spec | 82.3 tok/s | 10% |
| deepseek-rtx | prose / spec | 87.7 tok/s | 10% |

All expected rows have `workload_match:false`:
unavailable original inputs, raw completions, lengths/counts prevent identical-workload claims.
Reports use PASS/WARN for numerical tolerance and append `(workload differs/unspecified)`
when workload match is false; a numerical PASS does not establish reproduction. Measured prompt IDs
with a context suffix (`code-ctx1024`) are compared with the README row of the bare prompt (`code`).
DeepSeek native decode compares against README decode rows; optional HTTP serving compares
only against README serve rows.
A row with `prompt_id` `*` is a README row measured on an unspecified benchmark prompt and applies to
every kit prompt of that mode (an exact prompt row wins); a row with a `profile` applies only to runs
of that `--profile` (GLM's R1 and R1b rows). MiniMax and GLM rows cover greedy decode, 1K (and GLM
8K/32K) prefill and MiniMax DSpark per content type; DSpark speed depends strongly on the content,
so DSpark rows on kit prompts are expected to differ. No standalone RTX TTFT baselines are shipped. Do not extrapolate
missing values from ceilings or other profiles.

Shipped refs are unmeasured/null. **NO-REFERENCE** means no published reference for this reference
key and cell, not an exactness pass. Unstable output/digests across repeats and plain/spec
inequality inside one run produce MISMATCH even without a reference. Omitting exactness via
`--only` clears its array; `complete` status alone is not qualification.

Each run records two fingerprints:

- `identity_sha256`: everything about the run (profile, build options, numerical environment, kit
  module hashes, binary and model hashes, ROCm/HIP, kernel, prompt set, resolved parameters). It
  tells two results apart; it is not used to look up references.
- `reference_sha256`: only what decides the output bits: engine, profile, the resolved numerical
  environment (which for the DeepSeek, GLM and MiniMax apps includes their build-time numerics: a
  binary built without a required library drops the item and the run fails), model file hashes and
  the prompt-set hash. Binary hashes, build options, compiler, kernel, ROCm, paths, timeouts and output options are
  not part of it, so a self-built binary of the same commit and options is compared against the
  published references. A MISMATCH with matching `reference_sha256` therefore means your build or
  platform produced different bits: report it with both results files.

References are looked up by `reference_sha256` plus `profile:prompt_id:mode:generate_length`,
comparing the prompt hash, the full output token hash, the token count and the engine's own digests
when it prints them. Token SHA256 hashes compact ASCII JSON, e.g. `[1,2,3]`, no spaces or newline.

```bash
bash scripts/bench/bench.sh run --engine qwen --suite full --timeout 86400 \
  --write-refs --out bench-results/candidate
```

Maintainers receive `candidate-refs.json`, never edits to shipped refs. Failed/mismatching
runs cannot write candidates. Review identity, stability, plain/spec equality, receipts,
fidelity and independent reproduction before publishing real fingerprint entries. Qwen's
self-generated calibration goldens are not independent references. Numerical proposals remain
drafts until compiler, native execution, model fidelity and complete-workload gates pass;
theoretical throughput and external results do not establish local qualification.

### Protocol fields

Every new `results.json` carries these additive fields; older results without them still validate.

- **Cache reuse.** HTTP runs read `usage.prompt_tokens_details.cached_tokens` from every response.
  Each `http-cN` measurement records `cache_state` (`cold`: every request reported 0; `reused`:
  any request reported more; `unverified`: the engine did not report it) and `cached_tokens_max`.
  `cache_policy` in the result compares the largest count with `BENCH_ALLOWED_CACHED_TOKENS`
  (default 0, the shared prefix a suite may reuse) and sets `nonconforming` when it is exceeded.
  `BENCH_DECLARED_CACHE_REUSE=1` declares reuse as intended. The default policy flags the run;
  `BENCH_CACHE_POLICY=refuse` writes the result with `status: failed` and exits 1. Never compare a
  `reused` arm with a `cold` arm.
- **Engine counts and rates.** Each HTTP request records the engine's `completion_tokens` (usage)
  beside `client_token_count` (the received ID vector; they must agree), plus the engine's
  `strix` `queue_ms`/`prefill_ms`/`decode_ms` when the reply carries them. Engine-timed rows
  (`engine-decode` tok/s per request, `engine-ttft` = queue + prefill) are added next to the
  client-clock rows (`rate_source: client-clock` vs `engine`). CLI modes already take rates from
  the engine receipt, never from the process clock.
- **Not-run list.** `not_run` is always present: a list of `{item, reason}` for everything
  this run did not measure (skips, exactness not selected, HTTP replies without engine timings
  or cache counts, and items added with `kit.py not-run RUN_DIR ITEM REASON`). An empty list is a
  claim that nothing was left out.
- **ABBA k-of-n pairs.** Compare two arms from separate runs in ABBA process order:

  ```bash
  python3 scripts/bench/lib/kit.py pairs pairs.json A=a1/results.json B=b1/results.json \
    B=b2/results.json A=a2/results.json
  ```

  Consecutive runs form pairs (AB, BA, ...). For each measurement the output lists the per-pair
  relative deltas `(B - A) / A`, their median, `k_of_n` (pairs where B is better: higher tok/s,
  lower seconds) and `sign_agreement`. Every receipt is validated and must have `status:
  complete`. Arms are refused when they differ in engine, prompt set, models, workload or capacity
  parameters (generation length, contexts, concurrency, slots), per-measurement `workload` (prompt
  identities and lengths, including the constituent prompts of `http-cN` rows, and requested
  generation length; never taken from the optional exactness rows), or cache state (`cache_state` and `cached_tokens_max` must be identical;
  declaring reuse never makes a cold arm comparable with a reused one). Cache-nonconforming runs
  are refused unless reuse is declared. A published row in
  `expected.json` may carry `pairs_faster` and `pairs`; the report then shows `(ABBA k/n)`.
- **Dispatch census.** `tools/dispatch_census.py` reads a `rocprofv3 --kernel-trace` file (CSV
  `*_kernel_trace.csv` or JSON `*_results.json`; integer nanoseconds, never rounded through
  floating point) and reports launches, GPU busy time and host gap share for the whole trace and
  per window. Launches belong to the window they start in; busy time is the union of dispatch
  intervals clipped to each window (work carried across a boundary counts where it runs), so the
  gap share stays in [0, 1], and the largest gap includes idle time at both window edges (an empty
  window is all gap). A window starts at each
  dispatch of a boundary kernel (`--window-kernel REGEX`) or at each matching roctx marker
  (`--markers marker_api_trace.csv --marker-regex REGEX`). `--tokens N` (the engine-reported
  token count) adds per-token figures; top kernels by count and time are listed. Attach it to a run:

  ```bash
  python3 scripts/bench/lib/kit.py census bench-results/run TRACE.csv --window-kernel 'marker' --tokens 128
  ```

  This writes `dispatch-census.json` and a `dispatch_census` field into `results.json`.

## Machine interface

Stdout is the document channel; diagnostics use stderr. `list --json` returns exactly
`{"schema_version":1,"engines":[EngineAvailability,...]}`, ordered qwen, deepseek, glm,
minimax, deepseek-rtx. Each record has exactly `engine`, boolean `available`, string arrays
`reasons`, `profiles`, `binaries`, `assets`. Example record inside that array:

```json
{"engine":"glm","available":true,"reasons":[],"profiles":["r1b","r1","mtp","baseline"],"binaries":["strix-glm5-rank"],"assets":["glm"]}
```

`check --engine E --json` returns this envelope (synthetic hardware example):

```json
{"schema_version":1,"engine":"glm","ok":false,"problems":["pair verification failed"],"warnings":[],"hardware":{"cpu_model":null,"gpu_names":[],"gfx_arch":[],"ram_bytes":null,"kernel":"example","os":"Linux","rocm_version":null,"hip_version":null,"driver_module_version":null,"usb4_links":[]},"pair_ok":false}
```

Hardware keys above are the allowlist; USB4 entries contain only `rx_speed`. `pair_ok` is
present and true for single-node checks (no pair required). `ok` combines local/pair checks;
problems/warnings are strings. DeepSeek cache uncertainty appears in warnings. Parse JSON
on exit 1 too; early argument/helper failures may emit only a diagnostic, not a JSON envelope.

`run --engine E --json` emits the full `results.json` object, not a path wrapper. A minimal
synthetic example shows schema-required fields; real runs populate them and add fields below:

```json
{"schema_version":1,"kit_version":"2.0.0","repo_commit":"0000000000000000000000000000000000000000","engine":"qwen","identity":{},"identity_sha256":"0000000000000000000000000000000000000000000000000000000000000000","status":"complete","hardware":{},"prompts":[],"measurements":[],"exactness":[]}
```

`run --md` emits the Markdown file itself, beginning `# Engine benchmark`, with engine/status,
identity, prompt provenance, load/wall totals, measurement and exactness tables; for example:

```markdown
# Engine benchmark
```

The exact single-run contract is [results.schema.json](../scripts/bench/results.schema.json)
(schema version 1; additional properties permitted) plus the stdlib validator:

| Object | Emitted fields / shape |
|---|---|
| Result | Required fields above; also `tracked_dirty`, `timestamp_utc`, `suite`, `peer_hardware`, `peer_inventory`, `params`, `prompt_set`, `models_verified`, `wall_s`, `model_loads`, `skipped`, `logs`, `not_run`; optional `cache_policy`, `dispatch_census` (see Protocol fields) |
| Identity | `profile`, `build_options`, `environment`, `command_profile_sha256`, `kit_source_sha256s`, `binary_sha256s`, `model_files`, `rocm_version`, `hip_version`, `kernel`, optional `peer_kernel`, `parameters`, `prompt_set` |
| Binary | `name`, `sha256` |
| Model | `role`, `name`, `bytes`, nullable `sha256`, nullable `published_sha256` |
| Params / identity.parameters | Public `engine`, `suite`; integer arrays `contexts`, `concurrency`; integers `gen_tokens`, `repeat`, `warmup`, `timeout`; strings `profile`, `prompts` (`kit`/`file`), `verify_models`; string array `only`; booleans `serve`, `json`, `md`, `write_refs`, `dry_run`, `peer`; path markers `bin_dir`, `model_dir`, `peer_bin_dir`, `peer_model_dir`, `peer_repo` (`default`/`configured`), `out` (`run-directory`) |
| Prompt set | `name` (`kit`/`file`), exact source-byte `sha256` |
| Prompt | `id`, `tokens`, token-list `sha256`, `construction` |
| Measurement | `prompt_id`, `mode`, `unit` (`s`/`tok/s`), `repeat_count`, numeric array `samples`, `median`, `min`, `max`, `workload` (`prompts`: list of `id`/`sha256`/`tokens`, `generate_length`; HTTP rows list their constituent prompts); HTTP rows also `timing_scope`, `rate_source`, `cache_state`, `cached_tokens_max` |
| Exactness | `prompt_id`, `mode`, `generate_length`, `actual_tokens`, `prompt_sha256`, `token_ids_sha256`, nullable `engine_digest`, boolean `repeat_stable`, `verdict` (`EXACT`/`MISMATCH`/`NO-REFERENCE`); spec with plain peer also has `plain_tokens_equal` |
| Peer inventory | Null or `repo_commit`, `tracked_dirty`, `binary_sha256s`, `model_files`, `models_verified`, `build_options`, `hardware`; no connection details |

`all --json` returns `{"schema_version":1,"engines":[...]}`. Elements are full successful
or `status:"mismatch"` results (internal pair engine names include `-pair`), skip records such as
`{"engine":"glm","skipped":true,"reasons":["pair requires --peer HOST"]}`, or failed-child
records `{"engine":"qwen","returncode":1,"dry_run":false}`. `all --md` concatenates child
Markdown; skips go to stderr. There is no aggregate result file. `all --dry-run --json`
uses preview records with `returncode:0,dry_run:true`. Single-engine dry-run emits no result
JSON. Only list-json, check-json and run-json/md have these supported document contracts.

Exit 0 means successful dispatch/check/run/report (including all skipped). Validation,
prerequisite, parsing and pair failures normally return 1; native failures may propagate.
GNU timeout normally returns 124, or 137 after forced KILL; handled interrupts use 130/143.
`run` returns 1 when any exactness verdict is MISMATCH (`status:"mismatch"`), after
`results.json`/`results.md` are written and the requested JSON/Markdown is printed, so
`bench.sh run ... && publish` stops; `all` then returns 1. NO-REFERENCE and EXACT verdicts keep
exit 0, so measure-only and `--write-refs` candidate runs still succeed. Inspect
status, models_verified, exactness and skips as well as exit status. Initialized failures
may retain a partial failed result; early failures may leave no document. `report DIR` and
`python3 scripts/bench/lib/kit.py validate DIR/results.json` validate before use.

## Pair orchestration and teardown

Inventories must match commit, binary hashes, model records, safe build options and ROCm/HIP.
Matching filenames is insufficient: copy binaries or use reproducible builds. Rank 1 receives
translated peer paths/rank and the same location/profile controls. Before any measured launch,
the kit calls each planned resolver on both ranks through the existing peer execution path with
`--print-effective-config`, keeping the measured arguments except for DeepSeek native decode,
which probes the rank app with the model, rank and `usb4auto` transport. No rank listener or device work is
started by these probes. The configs must match after replacing known model/binary/run roots
and the `--rank` value; other settings and arguments must agree. Comparison precedes privacy
redaction so distinct settings cannot collapse to the same redacted value. Any production
`defaults:` fallback notice fails the run with the notice text (the MiniMax baseline banner is
informational). Missing or unparseable config output also fails closed.

`effective-LABEL-rank0.log`, `effective-LABEL-rank1.log` contain sanitized resolver output;
`effective-LABEL.json` preserves normalized settings/arguments/environment. Results include
these records in `resolved_configs`; DeepSeek decode records both `rank0` and `rank1`, plus
`env_source: "strix-deepseek41-rank --print-effective-config"`. The per-launch
resolved environments populate both `identity.environment` and `reference_key.environment`,
keyed by launch label. No per-engine production environment guesses are used; the one kit setting
on top is `STRIX_DS41_DSPARK_WINDOW=controller` for the DeepSeek decode app (the README DSpark rows
were measured with it; the app's own default is `fixed:5`), recorded in both environments. DeepSeek rank (decode environment)
and serve resolve overlay, CB4 and trellis mix-B; EP prefill resolves overlay, CB4 and
`STRIX_DS41_EXPERT_TP=0` and receives no trellis environment. The kit supplies only
`STRIX_DS41_PACKS` and, for baseline, its master switch.

RTX's replica exposes `--print-config` (JSON), not `--print-effective-config`; the kit records
that resolved configuration before decode. Its defaults select keep64, warm controller,
74.8 GiB arena with a drafter, 48 host spares and the adopted kernels/fusions. The kit no
longer forces an 80 GiB arena or repeats draft-device/row defaults. The dedicated CUDA prefill
app exposes neither config flag and has no defaults resolver; its explicit measurement argv
remains recorded by the kit source/parameter identity. Prefill-only RTX therefore has no
resolver environment record. RTX `--profile baseline` applies to the replica; the separate
prefill app has no baseline switch. Qwen parses resolver JSON, retains environment provenance
and the resolved profile pair, and puts each launch's environment into identity/reference keys.
Qwen inspection reads model metadata/content and host admission data but exits before HIP work.

Only deterministic `.ids`, `.u32`, `.tsv` inputs are uploaded to a unique peer `.bench-runs/`
directory. Each batch starts rank 1 under setsid with recorded process-group PID/log, then
rank 0 under the shared deadline. Completion, failure, timeout and Ctrl-C trigger TERM/KILL
of this run's recorded group, check remaining members for `/dev/kfd`, and retrieve/sanitize
rank-1 logs. No process-name kill is used. Remote tokens/PID/log remain for diagnosis.
SSH cleanup failure reports unverified teardown; the remote timeout still bounds lifetime.
The outer timeout includes setup/teardown plus forced-kill grace. Forced termination/network
failure may prevent final report delivery; inspect retained evidence privately.

## Long-context retrieval and reasoning checks (Qwen)

`tools/qualification/longctx_fidelity/` builds deterministic prompts whose answers are computed by the
generator itself: single-needle and multi-key retrieval, variable tracking, multi-hop chains and
aggregation, at an exact consumed-token depth up to 262,143 with the evidence at a chosen fraction of the
prompt. Construction uses the model tokenizer (`strix-tokenize`) and the server's non-thinking chat
rendering; the plain CLI runs each case with a clean environment and the default resolution, and the
scorer compares the answer (up to `<|im_end|>`) with the ground truth. Generation, planning and the tests
run on a CPU (`python3 -m unittest discover -s tools/qualification/longctx_fidelity/tests`); only `run_cli`
executes the model. Its README has the commands, the receipt format and the limits.

A first pilot (single needle, one seed, 19 cases at 8K, 32K, 128K, 262,112 and 262,143 tokens, the last
one answering with a single token) returned the exact value in every case, and a repeated case produced
identical token IDs. A second pilot of the reasoning and multi-key families at 8K and 32K (with short
controls) is in the kit README. One or two seeds are a pilot, not a qualification: long context stays
labelled unqualified until the full matrix, the reasoning families and continuation scoring pass.

## Nemotron 3.5 Bolt E4B

One node. Build the gemma4 family (`-DSTRIX_FAMILY_GEMMA4=ON`) and the targets `strix-gemma4-serve` and
`strix-gemma4-bench`; `check` refuses a cache that does not enable the family. Profiles are `default`
(`g4-v1`), `fast` (`g4-a8-v1`) and `quality` (EXL3 pack); each is its own population, and Q4_K_M and EXL3
references never mix. Prompts need Python `tokenizers`. Assets live under `MODEL_DIR/nemotron/`:

| Path | Profiles | Content |
|---|---|---|
| `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf` | default, fast | Q4_K_M GGUF |
| `drafter-mtp/` | default, fast | Optional MTP drafter, found next to the GGUF |
| `bf16/` | quality | BF16 model directory; an optional `bf16/drafter-mtp/` is the quality drafter |
| `exl3/` | quality | EXL3 pack directory (`--exl3-arm`) |
| `rotary-tables/` | all | Frozen rotary tables (see the model README) |
| `tokenizer.json` | all | Model tokenizer |

A run starts one server per arm with the plain production command for the profile: drafting off
(`--draft off --dflash off`: no drafter and no prompt lookup) and the shipped configuration (MTP drafter discovered, prompt lookup on). The C1 round of each arm sends every
prompt once as a chat request and reads the engine receipt from the final SSE chunk: complete generated
IDs, the committed state digest, the resolved profile and the acceptance counters. Exactness compares every
drafter row with drafting off, IDs **and** state digest. The drafter arm needs a draft witness on every
request (MTP drafter loaded, `mtp_confidence_resolved`, acceptance counters) and a positive control: at least
one prompt that drafted, verified and accepted tokens. A missing witness fails the run; a prompt that
legitimately did not draft is labelled `declined`, never counted as drafted. With `serve`, C4/C8 rounds run on
the shipped arm and must reproduce its C1 IDs and state. HTTP rows are `http-c1-draft-off`, `http-c1`,
`http-c4`, `http-c8`. The server runs one chat job at a time, so C4/C8 measure queueing, not batched decode.

For `default` and `fast`, exactness also runs `strix-gemma4-bench --verify-rows all --context 64`
(rows 1..8 host == serial == batched) and compares the R8 committed KV digest with the published pin for that
identity in `refs/nemotron.json`. The pin applies only to the pinned GGUF; any other file is NO-REFERENCE.
`quality` has no published R8 pin. Pending profile rows are never applied: each receipt must list them with
one stderr notice, and an unqualified row fails the run. The resolved configuration comes from the measured
receipts, so a run loads the model three times (verify, two arms) or twice for `quality`. Prefill is not
measured by this adapter. Without `drafter-mtp/` the drafter arm is skipped and recorded as such.

## Known limitations

- Standalone RTX Linux receipt admission is blocked regardless of CUDA prefill support.
- GLM MTP SPV1 receipts report native equality and FNV hashes, without full token vectors.
  Kit token exactness and reference candidates remain unavailable for MTP. Prompt lists are accepted.
- The GLM pair HTTP server (`strix-glm5-serve`, see the [GLM page](models/glm-5.3/README.md#glm-serve)) is not driven
  by the kit yet; GLM RTX exposes a file queue, not this HTTP interface.
- Default Qwen distill-g and some project assets still need separate publication/supply.
  Full-profile prerequisites remain required for prefill-only/plain-only runs.
- List/all presence checks are not profile-aware for GSQ. Check does not establish native
  execution or fidelity; skip verification cannot establish model content.
- README pipeline/RTX-plus-pair, CUDA offload, replicas/router, batched microbenchmarks,
  copy-window rewrites and unpublished campaigns are outside the kit. GLM defaults to r1b;
  README R1 needs explicit r1. GLM/MiniMax have no shipped expected rows.
- Historical prompts differ; Qwen long context is explicitly unqualified (retrieval pilot only, see
  [long-context checks](#long-context-retrieval-and-reasoning-checks-qwen)). README RTX
  load-to-load variance is about 10%. Calibration/CPU fixtures are not independent references.

## Report results

Attach **results.json and results.md** to a GitHub issue in this repository. JSON records
version/commit/dirtiness, identity/hashes, resolved controls, prompts, allowlisted hardware,
samples/statistics, exactness, wall time, load counts and skips. Markdown renders this with
contextual expected values and warnings. Keep result directories and comparative artifacts
outside Git.

The two reports omit connection destinations, usernames, addresses, home/drive paths,
serials and private environment dumps. Paths use basenames or default/configured markers.
Do not attach private caches, binding receipts, SSH configuration, model/state files, source
prompts or whole output directories. Custom prompt IDs and engine error text can be user
supplied: use public identifiers and inspect attachments. Auxiliary logs are sanitized but
need separate inspection before optional sharing. Describe thermal/power differences without
identifying private infrastructure.

```bash
bash scripts/bench/tests/run.sh                         # Linux
BENCH_PYTHON=python TMPDIR=/c/bench-tmp bash scripts/bench/tests/run.sh   # Windows Git Bash
```

On Git Bash, point `TMPDIR` at a drive path: Git Bash rewrites `/tmp/...` arguments for native
Python, which breaks the path comparisons of the effective-config tests.

CPU verification covers shell syntax, five offline plans, pair quoting/mapping, platform
admission, effective-config equality/mismatch/dropped-item shims, receipts, schema/Markdown,
references, cache, parameters, machine JSON, fetch previews,
HTTP test server and forced local teardown. ShellCheck runs if installed. This verifies wrapper
behavior, not hardware performance, remote driver teardown or numerical qualification.
