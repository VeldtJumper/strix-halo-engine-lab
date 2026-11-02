# MiniMax H3 (audio + video) on Strix Halo

[English](README.md) · [Español](README.es.md) · [Main README](../../../README.md)

MiniMax H3 is an audio + video diffusion transformer: 50 dense blocks (hidden 5376, 56 heads x 128, SwiGLU 14336) with
full non-causal 3D attention over the packed sequence `[text | cond | audio | video]`, a 50-layer Qwen3-VL-32B text
tower as text encoder, a 36-layer ViT3D video VAE and an audio VAE. The engine runs the whole clip on one Strix Halo
node or on the USB4 pair. It is an **opt-in build** (`-DSTRIX_FAMILY_H3=ON`); with the option off nothing is configured
or built. How the tables are read and what "exact" covers: [methodology](../../methodology.md).

## Exactness

- **Dense path = reference.** Every kernel is deterministic and batch-invariant: the reduction order does not depend
  on the row count or on which node computes a row. The two-node clip is **byte-identical** to the one-node clip
  (latents and all frames, checked by sha256).
- Each package has a plain C++ host reference and host tests that build and run without a GPU; GPU tests are separate
  executables (CTest label `gpu`).
- Speed levers that change bits (fast-class) each carry their own identity, an off switch and a quality gate against
  the dense reference.

## Packages

| Package | Directory | What it is |
|---|---|---|
| scaffold | `cmake/h3_00_common.cmake`, `src/h3/README.md` | `STRIX_FAMILY_H3` option, shared target `strix_h3_common` |
| gemm | `src/h3/gemm` | W8A8 tensorwise + rotated int8 projections and the activation quantiser, fused SwiGLU; 37-45 TOPS end to end at 19K-38K rows |
| attn | `src/h3/attn` | dense non-causal flash attention (head dim 128), split-KV partials + canonical merge; ~33-35 TF/s |
| te | `src/h3/te`, `tools/h3/te` | tokenizer, text and vision towers from a te pack (nvfp4 codes, unswizzled block scales, sha256 manifest) |
| vae | `src/h3/vae` | 36-layer video decoder, audio decoder, video encoder; two-node decode split with the same frames |

The clip runner and the two-node schedule follow in later changes; the times below were measured with them on the
same kernels.

## Build and test (host only)

```sh
cmake -S . -B build-h3 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRIX_FAMILY_H3=ON -DBUILD_TESTING=ON
cmake --build build-h3
ctest --test-dir build-h3 -R h3 -LE "gpu|qualification"
```

The data-dependent tests report SKIPPED without their fixtures or weights; the tests labelled `qualification` fail
without them on purpose (a skip never counts as a pass there). GPU builds (gfx1151) add `-DSTRIX_ENABLE_HIP_RUNTIME=ON`;
the GPU tests are registered in CTest with `-DSTRIX_H3_GPU_CTEST=ON` where noted in each package's CMake file.

<a id="h3-results"></a>

## Measured results

Clip: text + audio to video, 1344x768, 124 frames (5.17 s at 24 fps) with audio, 8 sampling steps, seed 42, 37,734
tokens in the packed sequence. Times are GPU work with a warm page cache; weights: int8 rotated DiT, nvfp4 text
encoder pack, fp16 VAEs.

| Configuration | Text encoder | DiT, 8 steps | Video VAE | Total | Exactness |
|---|---:|---:|---:|---:|---|
| Dense, 1 node | 10 s (load 9.7 s, encode 0.74 s) | **860 s** (105.5-113.2 s/step) | 87 s (incl. load) | **~16 min** | reference |
| Dense, 2 nodes over USB4 | 10 s | **435 s** (52-55 s/step, 1.98x) | **40.6 s** (+5.3 s load) | **~8 min** | byte-identical to 1 node |

Audio VAE: 0.3-0.6 s per clip. Per run, each rank also loads the DiT weights (14-17 s, in parallel on the pair).

Kernel-level numbers (gfx1151, one node):

| Kernel | Result |
|---|---|
| int8 projections, quantise + GEMM, 19K rows | qkv 45.2, out 38.4, fc1 44.6, fc1+SwiGLU 44.9, fc2 38.6 TOPS |
| int8 projections, 38K rows | qkv 43.3, out 39.2, fc1 42.6, fc1+SwiGLU 42.9, fc2 37.2 TOPS |
| attention, 2-node per-block shape | 34.8 TF/s (16-wave schedule, default) |
| attention, fused 37910 x 37910 | 32.9 TF/s |
| text encoder forward, 300 / 2500 / 5000 tokens | 1.42 / 9.56 / 22.3 s (dense class) |
| video encoder, 17-frame 768x1344 clip | 11.8 s |

## Text encoder pack

The text encoder reads a te pack produced by `tools/h3/te/te_pack.py` from the nvfp4 text-encoder checkpoint:

```sh
python tools/h3/te/te_pack.py --src <checkpoint.safetensors> --out <pack dir> --verify all
```

The pack keeps the 4-bit codes exactly and stores the e4m3 block scales row-major; the loader checks `te-pack.json`
against the engine geometry and the sha256 of every file it opens (`STRIX_H3_TE_VERIFY=0` skips only the hashes).

## Switches

| Variable | Effect |
|---|---|
| `STRIX_H3_GEMM_QUANT_FAST=0/1` | force one quantiser variant (identical bits; default picks per K) |
| `STRIX_H3_ATTN_WAVES=8` | 8-wave attention schedule (identical bits; default 16) |
| `STRIX_H3_TE_SPLIT=0` | fast-class text encoder `h3-te-lm-fast-f16-v1` (default dense `h3-te-lm-dense-split-v1`) |
| `STRIX_H3_TE_VERIFY=0` | skip the pack sha256 check at load |
