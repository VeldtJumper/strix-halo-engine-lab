# Using a discrete GPU (optional)

[English](discrete-gpu.md) · [Español](discrete-gpu.es.md) · [Main README](../README.md)

Every engine runs on Strix Halo alone; that is the default and nothing on this page changes it. An NVIDIA discrete GPU
adds separate, opt-in routes. Each route is its own build and its own command, with its own declared identity and
gates. If you do not build or start it, the Strix Halo defaults are unchanged.

## What was tested

| | Tested |
|---|---|
| Card | RTX PRO 6000 Blackwell Workstation Edition, 96 GB (`sm_120`) |
| OS / driver | Windows 11 (WDDM), NVIDIA driver 617.14 |
| Toolkit | CUDA 13.0, Visual Studio 2022 (MSVC x64), CMake 3.24 or newer, Ninja |
| Smaller budgets | DeepSeek standalone and GLM run exact with `--vram-total-gib 24 --ram-total-gib 32`; DeepSeek also with 32 + 32 |
| Not tested | Other cards, Linux CUDA hosts, budgets below the ones listed |

The DeepSeek standalone planner refuses budgets below about 18.8 GiB of VRAM at `--max-positions 2560` and prints the
minimum it needs.

## Routes at a glance

| Route | What runs where | Measured | When it does not help |
|---|---|---|---|
| DeepSeek V4.1 Flash, standalone RTX | Whole model on the card: Q2 routed experts, Q8_0 dense, VRAM + pinned-host expert cache, target and DSpark draft on the same GPU | Full card: 82.3 / 87.7 / 94.4 tok/s (code / English / Spanish). 24 GiB VRAM + 32 GiB RAM, one NVMe, 96-token prompt: 6.55 tok/s plain, 6.75 tok/s with DSpark, with 4 file reads in flight (5.45 tok/s with 1) | With 24 GiB VRAM + 32 GiB RAM most experts are read from disk: 6.55 tok/s against 82-94 tok/s on the full card |
| DeepSeek V4.1 Flash, RTX + pair | RTX runs layers 0-21 with CUDA; the two Strix Halo nodes run layers 22-39 and the head, over 1 GbE | Batched decode 60.1 / 97.5 / 137.2 tok/s at 2 / 4 / 8 rows; prompt reading on the RTX 3.64 / 4.39 / 12.20 s at 2K / 8K / 32K | Hot-expert placement degrades DSpark windows of 4 or more |
| Qwen3.8-Flash-Next, prefill offload | RTX reads the prompt (exact CUDA prefill) and streams attention and recurrent state to the node layer by layer; the node decodes | TTFT 1.36 / 1.92 / 2.94 / 4.95 / 8.99 s at 1K / 4K / 8K / 16K / 32K, vs 1.80 / 5.20 / 10.46 / 21.43 / 44.02 s on the same server without the RTX | Below 1K tokens the router keeps prompts on the node; at 1K the one-node prefill (1.12 s) is faster than the offload (1.36 s) |
| GLM-5.3 Flash, one RTX (draft) | Trunk, dense layers and shared experts on the card; routed experts streamed through the [tierstream](https://github.com/hyukudan/tierstream) cache across VRAM, pinned RAM and disk | Full card: prefill overlap cuts TTFT to 0.56× at 8K and 0.66× at 32K; decode 6.0 / 6.2 tok/s warm at 1K / 8K. 24 GiB VRAM + 32 GiB RAM: TTFT 60.3 s at 1K, 62.4 s at 8K; decode 0.67 tok/s warm | Small budgets read most experts from disk, so decode is slow |

Every row is exact against its declared reference: token IDs (and, where listed, logits and state) are equal to the
reference run. Details and conditions: [DeepSeek standalone](models/deepseek-v4.1/README.md#deepseek-standalone-rtx-results),
[DeepSeek RTX + pair](models/deepseek-v4.1/README.md#deepseek-rtx-pipeline),
[Qwen prefill offload](models/qwen3.8/README.md#qwen-prefill-offload),
[GLM on one RTX](models/glm-5.3/README.md#glm-rtx-results).

## Pin the right GPU

When the machine has more than one NVIDIA GPU, select the card by UUID, never by index (indices can differ between
`nvidia-smi` and CUDA):

```powershell
nvidia-smi -L                      # lists each GPU with its UUID
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
$env:CUDA_VISIBLE_DEVICES = "GPU-<uuid of the card to use>"
```

Set it in the same shell before every command on this page, including configuration prints and dry runs.

## DeepSeek V4.1 Flash, standalone RTX

Build tierstream's CUDA library first, then, from the repository root with a tierstream checkout at `TIERSTREAM`:

```powershell
cmake -S . -B build-rtx -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl `
  -DSTRIX_CUDA_DS41_ONLY=ON -DSTRIX_ENABLE_CUDA_DS41=ON `
  -DSTRIX_FAMILY_DEEPSEEK41=ON -DSTRIX_ENABLE_HIP_RUNTIME=OFF `
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRIX_TIERSTREAM_DIR=TIERSTREAM
cmake --build build-rtx --target strix-cuda-ds41-replica
```

Run, the persistent worker, smaller budgets and every option: [standalone RTX guide](models/deepseek-v4.1/rtx-standalone.md).

## DeepSeek V4.1 Flash, RTX + pair

Needs the two-node pair, a CUDA build (`STRIX_ENABLE_CUDA_DS41`) on the GPU machine and a 1-2.5 GbE link between them.
The pipeline TCP port has no authentication: keep it on a private network. Stage launch contract:
[pipeline stages](models/deepseek-v4.1/pipeline-stages.md).

## Qwen3.8-Flash-Next, prefill offload

The node side uses `--rtx-state-import` with `--http-serve` (without `--http-batch`); with the flag off nothing
changes. Service build and protocol: [Qwen CUDA prefill service](engine/serving/qwen-cuda-prefill.md) and
[RTX state import](rtx-state-import.md).

## GLM-5.3 Flash on one RTX (draft)

Build, then the measured run (exactness and speed; an HTTP server for this route is not tested end to end yet):

```powershell
pwsh src/glm/rtxpf/producer/build.ps1 -BuildDir BUILD_DIR -Fresh -Parallel 2 `
    -TierstreamSource TIERSTREAM_DIR [-CutlassDir CUTLASS_DIR]
python -B src/glm/rtxpf/producer/decode_qualification_window.py --build BUILD_DIR `
    --model MODEL_DIR --prompt-dir PROMPT_DIR --unit-file OUT_DIR/glm-units.bin `
    --unit-manifest-sha256 MANIFEST_SHA256 --full-card --allow-gpu
```

For a smaller card replace `--full-card` with `--vram-total-gib 24 --ram-total-gib 32` (or your own caps). The unit
file, every option and the measured rows: [GLM RTX guide](../src/glm/rtxpf/README.md).

The GLM service (`glm_native_service`) also takes `--dgpu auto|<uuid>`: it checks through NVML, before any CUDA call,
that the card is an `sm_120` RTX PRO (`auto` picks the eligible card with the most memory) and pins it; `--dgpu off` is
refused (this route has no CPU path). The measured-run harness above selects the card with `CUDA_VISIBLE_DEVICES` as
described in [Pin the right GPU](#pin-the-right-gpu).

## Troubleshooting

- **A plan does not fit.** The DeepSeek and GLM planners refuse before loading and print the memory they need; lower the
  context (`--max-positions`) or raise the caps. GLM: `--plan-budget` prints the split with no CUDA.
- **Check the configuration first.** DeepSeek standalone: `--print-config` prints the resolved configuration and
  exits before any CUDA call. GLM: `--print-effective-config` prints each option, its value and source without loading
  the model.
- **The wrong card is used.** Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` and the UUID as above.
- **Speeds are lower than the table.** Other programs using the same GPU (screen recording, capture or other compute
  jobs) lower every number; the measurements above were taken with the card otherwise idle.
- **To stay on the Strix Halo defaults,** simply do not build or start these routes. Each route also has its own off
  switches in its guide.
