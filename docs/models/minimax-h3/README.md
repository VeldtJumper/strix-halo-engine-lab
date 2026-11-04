# MiniMax H3 (audio + video) on the USB4 pair

[English](README.md) · [Español](README.es.md) · [Main README](../../../README.md)

**A 5 s video clip with stereo audio in 405.5 s (6.8 min) on two Strix Halo nodes.**

Text to video + audio, 1344x768, 124 frames (5.17 s at 24 fps) and 2-channel 32 kHz audio, 6 sampling steps (Euler,
shift 12 / 3), seed 42, prompt *"A red fox trots through fresh snow in a quiet birch forest at dawn, soft light, its
breath visible."* (the runner's default prompt), 37,734 tokens in the packed sequence.

MiniMax H3 is an audio + video diffusion transformer: 50 dense blocks (hidden 5376, 56 heads x 128, SwiGLU 14336) with
full non-causal 3D attention over the packed sequence `[text | cond | audio | video]`, a 50-layer Qwen3-VL-32B text
tower as text encoder, a video VAE and an audio VAE. It is an opt-in build (`-DSTRIX_FAMILY_H3=ON`); with the option
off nothing is configured or built. How the tables are read and what "exact" covers: [methodology](../../methodology.md).

<a id="h3-result"></a>

## The 405.5 s clip

Configuration: two nodes over USB4 (sequence-parallel K/V exchange), the 6-step step-distilled LoRA
`h3-lora-fasth3-6step-v1`, sparse attention `h3-attn-sparse-thr-v1` at tau 0.75 (the first 3 calls and layers 0-1 and
39-42 dense, the other 44 layers sparse from call 4), the 24-layer fast video VAE `h3-vae-fast24-v1`, on the exact
int8 W8A8 GEMM and the 16-wave flash attention.

**Exactness:** for this identity, one node, the two-node schedule run on one node (world 1) and the two nodes give the
same bits (latents, frames and audio, every step's velocity). The step LoRA, sparse attention and the fast VAE are
FAST-class: each has its own identity line in the run record and its own switch; without them the run is the dense
reference.

| Stage | Time |
|---|---:|
| Text encoder | 0.8 s (+ 5.0 s load) |
| DiT, 6 steps | **300.2 s**: steps 1-3 (dense window) 59.2 / 54.3 / 55.0 s, steps 4-6 (sparse) 46.3 / 42.8 / 42.2 s |
| Video VAE (fast, 24 layers) | 42.4 s (+ 12.1 s load) |
| Audio VAE | 0.3 s |
| Process start, weight upload, one-time content hashing of new asset files | ~45 s |
| **Total** | **405.5 s** |

Dense reference of the same clip (8 steps, 36-layer video VAE): ~16 min on one node (DiT 860 s, video VAE 87 s, text
encoder 10 s) and ~8 min on the pair (52-55 s per step, video VAE 40.6 s), the pair byte-identical to one node.

<a id="h3-install"></a>

## Install and run

1. **Build** on each node (ROCm, gfx1151):

   ```sh
   cmake -S . -B build-h3 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRIX_FAMILY_H3=ON -DSTRIX_ENABLE_HIP_RUNTIME=ON \
     -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DSTRIX_HIP_ARCHITECTURES=gfx1151 -DBUILD_TESTING=OFF
   cmake --build build-h3 --target strix-h3-clip
   ```

   Both nodes must run the same binary (the ranks compare binary and asset ids at start).
2. **Model files**, in one directory per node (`H3_MODELS`):

   | File | Used by |
   |---|---|
   | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | DiT (int8 rotated) |
   | `te-pack/` (52 files + `te-pack.json`) | text encoder pack, written by `tools/h3/te/te_pack.py` from the nvfp4 text-encoder checkpoint |
   | `fasth3_dense_comfy_6step.safetensors` | 6-step LoRA |
   | `fast_h3_vae_noncausal_24l.safetensors` | fast video VAE |
   | `minimax_h3_video_vae_fp16.safetensors` | 36-layer video VAE (dense reference) |
   | `minimax_h3_audio_vae_fp32.safetensors` | audio VAE |

   plus the Qwen tokenizer files (`vocab.json`, `merges.txt`, `tokenizer_config.json`) in `H3_TOKENIZER_DIR`.
3. **One command per clip** - start rank 1 on the second node, then rank 0 on the first (same arguments; rank 0
   writes the outputs):

   ```sh
   export H3_MODELS=/path/to/models H3_TOKENIZER_DIR=/path/to/tokenizer HSA_DISABLE_FRAGMENT_ALLOCATOR=1
   ARGS="--width 1344 --height 768 --length 124 --seed 42 --lora h3-lora-fasth3-6step-v1 \
     --sparse 0.75 --sparse-dense-layers 39-42 --sparse-dense-calls 3 --fast-vae"
   ./build-h3/strix-h3-clip --world 2 --rank 1 --transport usb4auto $ARGS             # node 2
   ./build-h3/strix-h3-clip --world 2 --rank 0 --transport usb4auto --out clips/fox $ARGS   # node 1
   ```

   `--prompt "..."` sets the prompt. `tools/h3/block/clip_pair.sh` runs both ranks from node 1 over ssh
   (`H3_BLOCK_DIR`, `H3_PEER`, `NAME`, `ARGS`). One node: `./build-h3/strix-h3-clip --out clips/fox $ARGS`.
   Outputs: `frames_rgb24.raw` (all frames), four sampled PNG frames + `preview_strip.png`, `audio.wav`, `meta.json`
   (stage times), `hashes.txt` and `run_identity.txt`.

**Rollback:** each FAST-class lever has its own flag - drop `--lora` (dense 8-step sampling), `--sparse` (dense
attention), `--fast-vae` (36-layer VAE); with none of them the run is the dense reference. Building without
`-DSTRIX_FAMILY_H3=ON` removes the family entirely.

## Packages

| Package | Directory | What it is |
|---|---|---|
| fixtures | `src/h3/common`, `tools/h3/fixtures` | fixture format h3fx v1, C++ and python readers |
| gemm | `src/h3/gemm` | int8 W8A8 tensorwise + rotated projections, activation quantiser, fused SwiGLU (37-45 TOPS) |
| attn | `src/h3/attn` | dense non-causal flash attention (head dim 128), split-KV partials + canonical merge (~33-35 TF/s) |
| te | `src/h3/te`, `tools/h3/te` | tokenizer, text and vision towers from the te pack (nvfp4 codes, sha256 manifest) |
| vae | `src/h3/vae` | video decoder (36 layers and fast 24 layers), audio decoder, video encoder; two-node decode |
| levers | `src/h3/levers` | FAST-class: step LoRAs, sparse attention, first-block cache |
| block | `src/h3/block`, `tools/h3/block` | DiT host reference, device forward, samplers, clip runner `strix-h3-clip` |
| dist | `src/h3/dist` | two-node sequence-parallel schedule over USB4 |

Host tests (no GPU): `cmake -S . -B build-h3 -DSTRIX_FAMILY_H3=ON -DBUILD_TESTING=ON && cmake --build build-h3 &&
ctest --test-dir build-h3 -R h3 -LE "gpu|qualification"`. Data-dependent cases report SKIPPED without their fixtures.

## Switches

| Switch | Effect |
|---|---|
| `--lora h3-lora-fasth3-6step-v1` | 6-step step-distilled LoRA (FAST-class); without it: dense 8-step sampling |
| `--sparse 0.75`, `--sparse-dense-calls 3`, `--sparse-dense-layers 39-42` | sparse attention (FAST-class); without it: dense attention |
| `--fast-vae` | 24-layer fast video VAE (FAST-class); without it: 36-layer VAE |
| `--fbc 0.08` | first-block cache (FAST-class) |
| `STRIX_H3_ATTN_WAVES=8` | 8-wave attention schedule (same bits; default 16) |
| `STRIX_H3_GEMM_QUANT_FAST=0/1` | force one quantiser variant (same bits; default per K) |
| `STRIX_H3_TE_SPLIT=0` | text encoder fast class `h3-te-lm-fast-f16-v1` (default dense) |
| `STRIX_H3_TE_VERIFY=0` | skip the te pack sha256 check at load |
