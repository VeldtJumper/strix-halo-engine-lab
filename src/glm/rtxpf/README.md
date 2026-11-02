# GLM-5.3-Flash on one RTX card (GLM RTX)

GLM RTX runs GLM-5.3-Flash prefill and greedy decode on a single 96 GB RTX PRO
6000 class GPU (sm_120). The trunk, dense layers and shared experts stay on the
device. The routed experts (43 MoE layers x 288 experts) are served by the
[tierstream](#dependencies) device-gated expert cache. Each expert is one
"unit" in a unit file, and the cache spreads the units over three tiers:
device arena, pinned host RAM and the unit file on disk.

Prefill and decode are exact. Each prompt and continuation is computed with one
fixed kernel and accumulation schedule, and the harness checks bitwise digests
of logits and state. Prefill reads the experts through the same cache as decode,
so the device holds only one copy of every expert.

Status: draft, Windows only. It is opt-in and changes no other engine defaults
or routes.

## Dependencies

- Windows 10/11, Visual Studio 2022 (MSVC, x64), CUDA 13.0, CMake >= 3.24,
  Ninja, Python 3.11+ and PowerShell 7 (`pwsh`).
- [tierstream](https://github.com/hyukudan/tierstream) at commit
  [`5fc3441b600e50e17f9bcdad35f0c12a0d188875`](https://github.com/hyukudan/tierstream/commit/5fc3441b600e50e17f9bcdad35f0c12a0d188875).
  The build takes it as a plain source export with a `STAMP` file naming that
  commit (configuration refuses any other commit):

  ```powershell
  git clone https://github.com/hyukudan/tierstream.git tierstream-src
  mkdir TIERSTREAM_DIR
  git -C tierstream-src archive --format=tar 5fc3441b600e50e17f9bcdad35f0c12a0d188875 | tar -x -C TIERSTREAM_DIR
  Set-Content TIERSTREAM_DIR/STAMP 5fc3441b600e50e17f9bcdad35f0c12a0d188875
  ```

- CUTLASS 4.2.1 headers. CMake downloads the pinned release and checks its
  SHA-256. For an offline build, pass a local checkout of tag `v4.2.1` with
  `-CutlassDir` (CMake: `-DGLM_CUTLASS_DIR=...`).

## Build

```powershell
pwsh src/glm/rtxpf/producer/build.ps1 -BuildDir BUILD_DIR -Fresh -Parallel 2 `
    -TierstreamSource TIERSTREAM_DIR [-CutlassDir CUTLASS_DIR]
```

The build compiles at idle CPU priority, then checks the emitted SASS. It never
launches a CUDA process. To run the CPU tests (no GPU needed):

```powershell
ctest --test-dir BUILD_DIR --output-on-failure -j 1
python -B tests/glm_native/test_decode.py --build BUILD_DIR
```

## Unit file

The expert cache reads its file tier from one unit file, built once from the
model's safetensors shards (`MODEL_DIR`):

```powershell
BUILD_DIR/glm_unit_file.exe dry-run --model MODEL_DIR --out OUT_DIR/glm-units.bin
BUILD_DIR/glm_unit_file.exe copy    --model MODEL_DIR --out OUT_DIR/glm-units.bin --confirm-bytes PRIMARY_BYTES
BUILD_DIR/glm_unit_file.exe verify  --model MODEL_DIR --out OUT_DIR/glm-units.bin
```

- `dry-run` reads only the shard headers and writes nothing. It prints the exact
  output size (`PRIMARY_BYTES`: 12,384 units x the unit stride), the output
  paths and the free space. Add `--stripe STRIPE_DIR:K` (odd K-chunk runs) or
  `--stripe STRIPE_DIR:even:K` to plan stripe copies on other drives; the printed
  commands create them with tierstream's own tool (`ts-stripe make`). Pass the
  stripe directories at run time with `--weight-dirs` (see below): file-tier
  reads then use every drive. Each chunk is checked against the stripe manifest,
  and any bad chunk is read from the primary instead.
- `copy` authenticates the source shards and refuses unless `--confirm-bytes`
  equals the planned size. It writes the units, a manifest with per-unit and
  whole-file SHA-256, and the tierstream descriptor. It verifies everything and
  publishes the files without replacing existing ones. Finally it prints the
  manifest digest; pin that value with `--unit-manifest-sha256`.
- `verify` checks an existing unit file against its manifest.

## Running

`glm_native_service.exe` is a persistent worker that reads commands on stdin.
`decode_qualification_window.py` starts it, runs a fixed qualification plan
and writes a JSON receipt. CUDA execution always requires `--allow-gpu`.

```powershell
python -B src/glm/rtxpf/producer/decode_qualification_window.py --build BUILD_DIR `
    --model MODEL_DIR --prompt-dir PROMPT_DIR --unit-file OUT_DIR/glm-units.bin `
    --unit-manifest-sha256 MANIFEST_SHA256 --full-card --allow-gpu
```

Main options:

| Option | Meaning |
| --- | --- |
| `--full-card` | Whole-card mode: one 96 GB card, with host RAM minus a margin. Runs the 1K/8K qualification, TTFT at 1K/8K/32K, and cold/warm decode. The planner sizes the expert arena from the remaining memory. |
| `--vram-total-gib G`, `--ram-total-gib G` | Whole-device caps for smaller configurations. The shared budget planner splits experts across device, pinned RAM and file to fit them, and refuses plans that do not fit. |
| `--profile default\|fast\|quality` | Selects every performance option at once. `default` turns on all exact options. `fast` adds options that change bits only once they have passed qualification (none yet). `quality` is the same as `default`. Also `STRIX_GLM_RTX_PROFILE`. |
| `--print-effective-config` | Prints each option's value, source and status for the chosen profile, then exits without loading the model. |
| `--allow-pending` | Also runs profile rows that are still pending (benchmarks only; receipts stay unqualified). |
| `--row-tile N` | Prefill row tile (a multiple of 32). By default the planner picks the largest tile up to 8192 whose extra activation memory stays within 5% of the expert arena. |
| `--tile-sweep 2048,4096,8192` | Full card: one start per tile, with a cross-tile check that the digests are bitwise equal. |
| `--tier-predict 1` | Prefetches the next layer's experts while the current layer runs (pending). |
| `--weight-dirs "DIR;DIR"`, `--drive-readers N` | Multi-disk reads: stripe copies of the unit file on other drives (see Unit file); `N` reader threads per drive (1..32, default 4). Off by default (the primary only). Same bytes. |
| `--tier-disk-queue N`, `--tier-staging N` | File reads in flight in three-tier opens (1..64, at most the staging buffer count; default 1 = one at a time). Each reader's bounce buffer is billed to pinned host memory. Same bytes. |
| `--tier-clean-evict 1` | Three-tier opens: a device victim with no free pinned block goes back to the unit file with no device-to-host copy and displaces no pinned unit. Off by default. Same bytes. |
| `--prefill-overlap 0` | Turns off the prefill overlap, which is on by default: during a prefill through the cache, the next layer's experts are made resident while the current layer's kernels run (two layers held at once; when only the spare slots are in the way the prefill lowers the spare target and restores it before decode; off when the expert arena cannot keep two layers). Same bytes. |
| `--disk-prefetch 1` | With `--tier-predict 1`: predicted experts homed on the unit file are read ahead too (pending). |
| `--routes-out FILE --routes-prefill 1`, `--expert-routes FILE` | Placement census: record every prefill row's expert routes from your own prompts, then seed the cache's placement with them on the next start (placement only). |
| `--ab-arm`, `--census KINDS`, `--compare-arms DIR_A DIR_B` | Harness modes: one arm of a main-vs-build comparison (TTFT + decode, no exactness gate, every digest listed); a prefill-only routes census on chat prompts outside the decode bench; a CPU check that two arms' tokens, logits and states are bitwise equal, with speed ratios. |
| `--plan-budget` | Prints the planned memory split from shard headers only (no CUDA, no unit-file read). |

## Measured (draft, one RTX PRO 6000, full card)

- Exactness: every qualification, comparison and timing case is EXACT, and
  prefill state is bitwise equal across row tiles 2048, 4096 and 8192.
- Time to first token: 8K prompt 17.7-22.7 s, 32K prompt 41-55 s, depending on
  the row tile (best: 17.7 s at 8K and 41.2 s at 32K, with tiles 4096 and 8192).
- Decode with the few-row decode kernels (`decode_row_kernels`, on by default):
  6.0 tokens/s warm at 1K and 6.2 tokens/s at 8K (3.7 cold), from 2.4 tokens/s
  with them off, measured in one service load on a full-card open whose plan put
  1,098 expert units on the unit file (host RAM was short), so per-token file
  misses are still in these numbers. Tokens, every logits digest and the final
  session state are bitwise equal with the kernels on and off, and the 1K/8K
  qualification passes with them on. Per token, the routed MoE went from 137 to
  12 ms, KDA from 53 to 21 ms, the hyper-connection mixes from 56 to 11 ms and the
  router from 21 to 1 ms. Turn them off with `--decode-row-kernels 0` or
  `STRIX_GLM_RTX_ROW_KERNELS=0`.
- Smaller memory budgets run exact too. With 24 GiB VRAM + 32 GiB RAM
  (`--vram-total-gib 24 --ram-total-gib 32`; most experts are read from the unit
  file): TTFT 60.3 s at 1K and 62.4 s at 8K with the prefill overlap (63.4 /
  73.7 s with `--prefill-overlap 0`), decode 0.67 tokens/s warm.
- Tested on an RTX PRO 6000 Blackwell Workstation Edition, driver 617.14,
  CUDA 13.0 (Windows 11).
