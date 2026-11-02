# Getting started: requirements, build and run

[English](README.md) · [Español](README.es.md) · [Main README](../../README.md)

The [main README](../../README.md#quick-start) shows the shortest path: build on one Linux Strix Halo node and run Qwen with
the production defaults. This page has the requirements for every engine, the complete build and where each engine's
launch command lives. All CMake options, the full command line and the environment switches are in the
[engine reference](engine-reference.md); benchmark commands, model provenance and machine
interfaces are in [reproduce](../reproduce.md).

<a id="what-you-need"></a>

## What you need

| | Qwen3.8-Flash-Next | DeepSeek V4.1 Flash |
|---|---|---|
| **Machines** | one Strix Halo (`gfx1151`) with **128 GB** | two Strix Halo with 128 GB each, two direct USB4 cables, the [USB4 driver series](https://github.com/hyukudan/strix-halo-usb4-driver); optionally a CUDA PC with an RTX PRO 6000 |
| **OS / GPU stack** | Linux, ROCm/HIP (tested with ROCm core 10.1 under `/opt/rocm/core-10.1`) | same; CUDA for the optional RTX parts |
| **Model files** | UD-Q4_K_XL GGUF (~111 GB) + Q8_0 MTP sidecar | the 2-bit expert GGUF, the 6.5-bit dense overlay, the trellis packs and the DSpark drafter ([docs/models/deepseek-v4.1/production-defaults.md](../models/deepseek-v4.1/production-defaults.md#production-identity-environment-of-both-ranks)) |
| **Build tools** | CMake >= 3.24, Ninja, a C++20 compiler, optionally OpenSSL for fast hashing | same |

MiniMax needs its admitted EXL3 3.06-bpw directory and compatible local draft assets for DSpark;
GLM needs its EXL3 4.05-bpw directory and native MTP assets. Both use two Linux Strix Halo ranks
with the USB4 driver and trellis library. Standalone RTX needs Windows WDDM, CUDA, the admitted
Q2/Q8 model and receipt/state/reference assets. Model manifests, conversion provenance and required
notices are listed in [models and provenance](../reproduce.md#models-and-provenance).

Kernel/IOMMU settings belong to the machine owner. Keep USB4 controllers translated; never disable
IOMMU for speed. Record the running kernel and `amdgpu.iommu_perfopt` in timing receipts and compare
PerfOpt on/off only on the same kernel.

The command-line tools take token IDs; the HTTP servers carry the models' own tokenizers and chat templates.

<a id="build"></a>

## Build

```sh
export PATH=/opt/rocm/core-10.1/bin:$PATH
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/bin/amdclang++ \
  -DSTRIX_ENABLE_HIP_RUNTIME=ON \
  -DSTRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME=ON \
  -DSTRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE=ON \
  -DSTRIX_USE_OPENSSL_SHA256=ON \
  -DBUILD_TESTING=ON
ninja -C build strix-hip-decode-token                         # Qwen
ninja -C build strix-deepseek41-rank strix-deepseek41-serve   # DeepSeek (family module, STRIX_FAMILY_DEEPSEEK41=ON by default)
```

MiniMax and GLM also need the adopted trellis headers and archive. After the common HIP configuration,
use a relative `TRELLIS` checkout containing `include/strix_trellis.h` and
`build-qualification-v03/libstrix_trellis.a`:

```sh
cmake -S . -B build -DSTRIX_FAMILY_MINIMAX=ON -DSTRIX_MINIMAX_TRELLIS_DIR=TRELLIS
ninja -C build strix-minimax-rank strix-minimax-serve
cmake -S . -B build -DSTRIX_FAMILY_GLM5=ON -DSTRIX_GLM5_TRELLIS_DIR=TRELLIS
ninja -C build strix-glm5-rank
```

DeepSeek production trellis API/library settings and the complete per-engine build instructions are in
[Build each engine](../reproduce.md#build-each-engine). Configure and build both ranks identically.
MiniMax and GLM family targets require these build options even though their execution defaults are fast.

The CUDA parts for the RTX PRO build separately with `-DSTRIX_ENABLE_CUDA_DS41=ON`. `ctest --test-dir build` runs
the host and device tests (many device tests need the GPU and the model files). All CMake options are listed in
[docs/getting-started/engine-reference.md](engine-reference.md#build-options).

<a id="run"></a>

## Run

Start with the plain command for the selected engine: production optimizations are on by default when their model,
shape, resource and reference guards pass, and `--print-effective-config` (standalone RTX: `--print-config`) shows the
resolved configuration before any device work ([defaults and off switches](../methodology.md#defaults-and-off-switches)).
Each engine's launch contract, profiles and long-context options are on its page:

- [Qwen3.8-Flash-Next](../models/qwen3.8/README.md#qwen-recommended-command): one node; `--profile`,
  [long context and YaRN](../models/qwen3.8/README.md#qwen-long-context), prefix cache, HTTP server.
- [DeepSeek V4.1 Flash pair](../models/deepseek-v4.1/README.md#deepseek-pair) and
  [standalone RTX](../models/deepseek-v4.1/README.md#deepseek-standalone-rtx); RTX + pair:
  [pipeline stages](../models/deepseek-v4.1/pipeline-stages.md).
- [MiniMax M3.1 Flash pair](../models/minimax-m3.1/README.md#minimax-pair).
- [GLM-5.3 Flash pair](../models/glm-5.3/README.md#glm-pair) and [GLM on one RTX PRO 6000](../../src/glm/rtxpf/README.md).
- [Nemotron 3.5 Bolt E4B](../models/nemotron-3.5-bolt-e4b/README.md#build-and-run) (opt-in build).
- Exposing the servers, authentication and limits: [serving](../serving.md#exposing-the-servers).
