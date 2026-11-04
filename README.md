# Strix Halo engine lab

**Fast, exact LLM inference written from scratch for AMD Strix Halo: one mini PC, two linked over USB4, and a
discrete GPU when there is one.**

[English](README.md) · [Español](README.es.md)

This C++/HIP inference engine targets **AMD Strix Halo** (Ryzen AI Max+ 395, Radeon 8060S `gfx1151`, 128 GB unified
memory) under Linux with ROCm. The kernels, scheduling, speculative decoding, GGUF reader, memory planner and the
two-node communicator are our own code; the engine does not wrap another inference runtime. Six configurations are
**Production** and two are opt-in (see [Models](#models)); each has its own launch command, verification gates and server.

**Exactness means token-identical to the declared reference path.** Speculative decoding and concurrency preserve that
contract; changing weights, rotary semantics, numerical identity or request policy changes the reference. Every speed
in this README was measured in the lab with greedy decoding (temperature 0) on the build named in the row; these are
workload-specific results, not speed guarantees. How we measure, what "exact" covers and how an optimization becomes a
default: [methodology](docs/methodology.md).

<a id="models"></a>

## Models

| Model | Runs on | Status | Highlights |
|---|---|---|---|
| **Qwen3.8-Flash-Next** (UD-Q4_K_XL, 111 GB) | 1 node; 2 replicas behind a router | **Production** | Exact MTP with a distilled draft head; native context through 262K and explicit YaRN through 1M; automatic HTTP tiled prefill; server with as many request slots as memory admits (built on demand); decode keeps streaming during long prompts and short requests are admitted inside them; prefix state cache; [opt-in GSQ profile](docs/models/qwen3.8/README.md#gsq-opt-in-speed-over-quality-profile) |
| **DeepSeek V4.1 Flash** (1017; 2-bit experts, 6.5-bit + trellis dense) | 2 nodes over USB4 | **Production** | Expert tensor parallel decode, expert parallel device prefill, DSpark speculative decode, batched decode, server; prepared expert discovery and bounded prefix retention; bitwise gates |
| DeepSeek V4.1 Flash + RTX PRO 6000 | RTX PRO (layers 0-21) + 2 nodes (22-39) | **Production** (D-064) | Pipeline split over 1 GbE, CUDA prefill service, hot-expert placement |
| DeepSeek V4.1 Flash, standalone RTX | RTX PRO 6000 (96 GB, Windows WDDM) | **Production** | Q2 GGUF experts, Q8_0 dense; exclusive VRAM + pinned-host cache; on-GPU DSpark with warm controller selected as the speculative default |
| MiniMax M3.1 Flash (EXL3, 3.06 bpw) | 2 nodes TP2 over USB4 | **Production** | Eight-text quality gate passed; exact prefill/decode; DSpark serving auto admission and MSA mode 6; prefill/exchange defaults on |
| GLM-5.3 Flash (EXL3, 4.05 bpw) | 2 nodes TP2 over USB4 | **Production** | R1b decode identity by default; exact MTP self-speculation by default; exact R1 chunked prefill; `--identity r1` restores R1; R≥3 fix verified on the pair; HTTP server `strix-glm5-serve` (OpenAI chat/completions + SSE, continuous batching) |
| GLM-5.3 Flash on one RTX PRO 6000 (draft) | RTX PRO 6000 (96 GB, Windows) | **Opt-in** | Exact prefill and greedy decode; routed experts streamed through the [tierstream](https://github.com/hyukudan/tierstream) device-gated cache (device, pinned RAM, unit file); `--profile`; [guide](src/glm/rtxpf/README.md) |
| Nemotron 3.5 Bolt E4B (Q4_K_M; EXL3 pack for `quality`) | 1 node | **Opt-in build** (`STRIX_FAMILY_GEMMA4=ON`) | `--profile default\|fast\|quality`; exact decode at every batch size; exact MTP drafter at C1; continuous-batching server |
| **MiniMax H3** (audio + video; int8 DiT, nvfp4 text encoder pack) | 1 node; 2 nodes over USB4 | **Opt-in build** (`STRIX_FAMILY_H3=ON`) | 5 s 1344x768 clip + stereo audio in **405.5 s** on the pair; two nodes byte-identical to one node; dense reference and FAST-class levers (6-step LoRA, sparse attention, fast VAE) each with its own switch; [page](docs/models/minimax-h3/README.md) |

The core knows no model: each family is a module with its own contract, plan, kernels and gates
([docs/models/model-families.md](docs/models/model-families.md)). Automatic routes admit model assets under their documented contracts; provenance and licenses are recorded in
[docs/reference/sources-ledger.md](docs/reference/sources-ledger.md).

<a id="hardware"></a>

## Hardware

- **Node:** one AMD Strix Halo mini PC, Ryzen AI Max+ 395 with the Radeon 8060S iGPU (`gfx1151`), 128 GB of
  unified memory, Linux, ROCm.
- **Pair:** two nodes joined by two direct USB4 cables (one 40 Gb/s, one 20 Gb/s link), driven by our
  [USB4 driver series](https://github.com/hyukudan/strix-halo-usb4-driver): about 37 Gb/s per direction with both
  directions busy, 18.4 µs for a 20 KB exchange.
- **Optional discrete GPU:** an RTX PRO 6000 (96 GB, Windows WDDM), used for standalone DeepSeek, Qwen prompt offload
  or GLM on one card; the DeepSeek pipeline split reaches the pair over 1 GbE.

## Headline results

Optimised defaults first: each row is the engine's shipped configuration unless the mode says otherwise, with the
measured gain against our previous default as recorded in that row's own campaign. **tok/s** is decode speed after the
prompt has been read, for one request unless the concurrency says otherwise; **TTFT** is time to first token; C*n* is
*n* concurrent requests. Gains are workload-specific and must not be added. The model pages keep every row with its
conditions, repetition counts, exactness evidence and off switches.

| Model | Hardware | Mode and workload | Result | Gain vs our previous default | Build |
|---|---|---|---:|---|---|
| **Qwen3.8-Flash-Next** | 1 node | MTP decode, 1K / 8K code prompts, auto windows, 256 tokens | **62.44 / 54.00 tok/s** (prose 55.7 / 49.58) | 1K code 59.59 → 62.44 tok/s with grouped QSA score (+3.0 to +5.7 %) and dispatch fusion (−0.3 to −3.2 % decode time) | 1K code main `85353311b`; 8K code `702bc074`; prose `0117a140` / `226bb691` |
| Qwen3.8-Flash-Next | 1 node | Plain decode, 1K / 32K / 262K (YaRN) | **41.59 / 43.92 / 45.91 ms/step** (262K: 21.8 tok/s) | 43.95 / 46.75 / 49.46 ms/step → −5.4 / −6.1 / −7.2 %, tokens identical | main, one binary |
| Qwen3.8-Flash-Next | 1 node | Exact prefill TTFT, 4K / 32K, plain CLI default route | **2.397 / 19.05 s** | 2.548 / 20.57 s with every default lever off → −5.9 / −7.4 % (one binary, alternating runs; 32K T16384 vs T8192) | main, one binary |
| Qwen3.8-Flash-Next | 1 node | Exact prefill, plain command, 1K / 4K / 32K / 128K / 262,143 tokens | **1,337 / 1,695 / 1,713 / 1,617 / 1,509 tok/s** (32K 19.13 s; 54–56 % of the contract ceilings) | 1,363 / 1,395 / 1,238 / 1,089 tok/s at 4K / 32K / 128K / 262,143 on `8b99cf58` | main `85353311b`, one binary |
| Qwen3.8-Flash-Next | 1 node | 262,144-token prefill (YaRN factor 2), serial route | **1,484 tok/s** `default` (one run) | 1,464 tok/s on `669bcb83` | main `17b9c57b8` |
| Qwen3.8-Flash-Next | 1 node | MTP decode at 262K / 524K / 1M context (YaRN) | **59.5 / 60.6 / 60.2 tok/s** | serial YaRN decode 21.82 / 19.7 / 19.6 tok/s; at 1M serial ran on Linux 7.3 and MTP on Linux 7.4, so no ratio | `7b2453fe` (262K MTP), `17b9c57b8` (262K serial), `a199c8e0` (524K), #288 `3b8b9ac0f` (1M) |
| Qwen3.8-Flash-Next | 2 nodes, one replica each | HTTP coding chats, C1 / C8, wall throughput | **68.2 / 158.1 tok/s** | +3.6 % wall at C8 and −6.9 % TTFT at C4 vs the release build, same session | `0117a140` |
| Qwen3.8-Flash-Next | 1 node | HTTP single runner, 127.6K-token conversation, cold decode | **52.0 tok/s** | 39.5 tok/s before the QSA MTP pool; restored turns +31 % / +37 % | main |
| Qwen3.8-Flash-Next + RTX PRO | 1 node + RTX PRO | Prefill offload, TTFT at 32K (8K) | **8.99 s** (2.94 s) | 44.02 s (10.46 s) on the same server without the RTX, 4.9× at 32K | `70b93af3` |
| **DeepSeek V4.1 Flash** | 2 nodes over USB4 | DSpark speculative decode, code / English / Spanish, 256 tokens | **70.6** / 41.9 / 45.3 tok/s | cooperative draft on both ranks: agent edit 83.8 vs 80.4 tok/s (+4.3 %), tool JSON 93.5 vs 89.0 (+5.0 %), prose 36.1 vs 34.1 (+5.7 %) | `cab238d6`; cooperative draft on main |
| DeepSeek V4.1 Flash | 2 nodes | Batched decode, 8 rows, no drafting | **70.24 ms/step** (113.9 tok/s aggregate, calculated) | 71.84 ms/step on `69c6bdab` → −2.2 %; routed expert rows in registers: R6 −1.43 % | `6a5604bf` |
| DeepSeek V4.1 Flash | 2 nodes | Server TTFT, one client, cache off, 8,399 / 32,499 / 44,699 tokens, `default` | **8.19 / 28.92 / 40.51 s** | 10.44 / 36.85 / 53.62 s on main #148 → 8.29 / 29.77 / 42.72 s (−20.6 / −19.2 / −20.3 %); deferred decoder cone then −3.3 % / −5.9 % at 32.5K / 44.7K; HC fusion −1.1 / −0.9 % | deferred-cone build, mean of two turns |
| DeepSeek V4.1 Flash | 2 nodes | Same prompts, `--profile fast` (bit-changing) | **7.56 / 26.33 / 36.77 s** | 8.36 / 29.38 / 41.55 s for `default` in the same run → −9.6 / −10.4 / −11.5 % | deferred-cone build |
| DeepSeek V4.1 Flash + RTX PRO | RTX PRO + 2 nodes | Pipeline split, batched decode N2 / N4 / N8; server C8 | **60.1 / 97.5 / 137.2 tok/s** aggregate; **91.7 tok/s** | bitwise equal to the pair-only reference (no earlier default on this route) | pair `6525d5e8`, CUDA `463c08ec` |
| DeepSeek V4.1 Flash, standalone | RTX PRO 6000 (Windows) | On-GPU DSpark decode, code / English / Spanish, same load | **82.3 / 87.7 / 94.4 tok/s** (best across loads 94–95 / 97 / 100–101) | keep-64 draft pruning +16 % on code; decode/cache kernel set all-hit 12.62 → 10.50 ms/token | measured line of PR #52 |
| **MiniMax M3.1 Flash** | 2 nodes | Greedy decode, 1K / 8K, 256 tokens | **44.57 / 46.10 ms/token** (22.44 / 21.69 tok/s) | exchange-boundary defaults: N1 48.22 → 43.92 ms/step; later MoE grid + o_proj warm levers −1.2 % and −1.3 % at 8K | `8cc30ac0` |
| MiniMax M3.1 Flash | 2 nodes | DSpark decode, JSON / Python / prose | **54.4 / 34.1 / 24.1 tok/s** | serving auto admission: JSON −34.6 %, code −17.5 % wall time, prose stays plain | `8cc30ac0` |
| MiniMax M3.1 Flash | 2 nodes | Fresh combined prefill, 8K / 32K | **17.731 s** (462 tok/s) / **91.508 s** (358 tok/s) | 8K baseline 21.874 s → −19 %; later gather + q/k/v levers 8K −2.3 to −4.3 %, 32K −4.3 to −5.3 % | `e0bad3d1` |
| **GLM-5.3 Flash** | 2 nodes | MTP self-speculation decode, chat / code / JSON | **37.1 / 44.5 / 40.0 tok/s** | 1.50× / 1.79× / 1.62× plain R1b decode (40.3–40.6 ms/token); R1b identity −11.5 % vs R1 | default identity |
| GLM-5.3 Flash | 2 nodes | Exact chunked prefill TTFT, 1K / 8K / 32K, 8192-row chunks | **2.83 / 15.15 / 61.96 s** (541 / 529 tok/s at 8K / 32K), all defaults | 3.18 / 17.77 / 76.59 s → 2.83 / 15.15 / 61.96 s (−11.0 / −14.7 / −19.1 %), logits bitwise; last step MoE remainder tiles: 8K −1.3 % (alternating pairs), 32K −1.2 % (both run orders) | #300 `32ec66bd3` |
| GLM-5.3 Flash | 2 nodes | Served (`strix-glm5-serve`), concurrent streams C1 / C2 / C4 / C6 / C8, steady-state decode; 8 built-in short chat prompts in the GLM chat template, 22–59 prompt tokens (mean ~33), up to 128 greedy tokens each, barrier start, cache off, the server has no prompt cache | **C1 40.0 / C2 53.0 (G2 off 37.8) / C4 59.9 (53.6) / C6 62.5 (62.4) / C8 69.5 (69.4) tok/s aggregate** | every stream == the same request alone (exact); C1 through MTP (25.30 tok/s plain); batched MTP verify (G2) +40 / +12 % at C2 / C4, hands over to batched plain steps from C6 | current |
| GLM-5.3 Flash on one RTX PRO 6000 | RTX PRO (Windows) | Exact prefill TTFT, 8K / 32K, full card, tile 8192 | **11.4 / 30.5 s** | first request, mean of two loads; prefill clean eviction and the tile rule's 8192 at full card; GPU-compute bound at 32K | current |
| GLM-5.3 Flash on one RTX PRO 6000 | RTX PRO (Windows) | Decode, 1K / 8K, warm (cold) | **13.25 / 13.52 tok/s** (3.63 / 4.09) | full card, 24-token bench, ABBA means of one load; every compare bitwise equal | current |
| GLM-5.3 Flash on one RTX PRO 6000 | 24 GiB VRAM + 32 GiB host RAM | TTFT 1K / 8K; decode 1K warm (cold) | **65.6 / 57.5 s**; **0.77 tok/s** (0.75) | file reads in flight auto by default at this budget (76 % of experts on the unit file); TTFT one load, decode mean of two loads, 24-token bench; single loads vary up to 2x | current |
| **Nemotron 3.5 Bolt E4B** | 1 node | `default`, English chat, 256 tokens: C1 shipped (MTP drafter + prompt lookup) / C1 plain / C4 / C8 | **91.9** / 55.8 / 169.8 / 237.9 tok/s | prompt lookup on an agent code-edit prompt: 48.0 → 115.6 tok/s without a drafter, **149.1** shipped; chat decode unchanged within noise | `f32aab24` |
| Nemotron 3.5 Bolt E4B | 1 node | `fast` (int8 activations, qualified), same columns | **99.5** / 54.0 / 206.5 / **340.6** tok/s | exact prefill 2K 458 → 538 tok/s; decode unchanged within noise | `f32aab24` |
| Nemotron 3.5 Bolt E4B | 1 node | `quality` (EXL3 M4 pack), same columns; top-1 agreement with BF16 | **96.4** / 50.2 / 184.8 / 303.0 tok/s; **92.94 %** (Q4_K_M 87.0 %) | C1 shipped 83.8 → 96.4; plain C1 / C4 / C8 44.4 / 153.1 / 258.3 → 50.2 / 184.8 / 303.0 | `f32aab24` |
| Nemotron 3.5 Bolt E4B | 1 node | Selectable prompt prefill route `g4-wmma-pf-v1` (`STRIX_G4_PREFILL_IDENTITY=v1`; FP16 WMMA linears on the prompt bulk, the last <= 8 prompt tokens and the anchor on the decode arithmetic; not exact), 2K / 4K / 8K | **2,359 / 2,249 / 1,957** tok/s | exact prefill 526 / 509 / 481, `g4-wmma-pf-v2` (`=v2`) 1,537 / 1,478 / 1,349 in the same session; every profile keeps exact prefill by default | current |
| Nemotron 3.5 Bolt E4B | 1 node | Exact prefill, 2K / 8K / 32K: `default` and `quality` | **535 / 509 / 397** and **815 / 746 / 533 tok/s** | 466 / 429 / 358 and 769 / 701 / 520 tok/s before the fused gate/up tile, linear tiles, 24-bit and shared Q6_K decode, EXL3 LDS pairs | `f32aab24` |
| Nemotron 3.5 Bolt E4B | 1 node | HTTP server, `default`, 8 clients, mixed prompts (chat, code, one ~1.3K-token document), prefix cache off, 128 tokens: aggregate / TTFT p50 | **76.1 tok/s / 0.6 s** | one request at a time: 66.9 tok/s / 13.8 s | `f32aab24` |
| **MiniMax H3** | 2 nodes over USB4 | Text to video + audio, 1344x768, 124 frames (5.17 s) + stereo audio, seed 42: 6-step LoRA, sparse attention tau 0.75, fast VAE | **405.5 s** (DiT 300.2 s, video VAE 42.4 s) | dense reference 8 steps: ~16 min on 1 node, ~8 min on the pair | h3 block `9d9d523b` |
| llama.cpp `74a4fb00`, same machine, same Q4_K_M | 1 node | Reference: C1 plain / C4 / C8; prefill 2K / 8K / 32K; server at 8 clients | 55.5 / 193.3 / 279.9 tok/s; 2,062 / 1,881 / 1,376 tok/s; 227.1 tok/s | measured in the same session as the Nemotron rows | `74a4fb00` |

Model pages: [Qwen](docs/models/qwen3.8/README.md) · [DeepSeek](docs/models/deepseek-v4.1/README.md) ·
[MiniMax](docs/models/minimax-m3.1/README.md) · [GLM](docs/models/glm-5.3/README.md) ·
[Nemotron](docs/models/nemotron-3.5-bolt-e4b/README.md) · [MiniMax H3](docs/models/minimax-h3/README.md). The changelog of every shipped default with its measured
gain and off switch is [measured gains](docs/measured-gains.md); the dated records are in
[performance](docs/performance.md).

## Quick start

This minimal example is for **one Linux Strix Halo node** with the required ROCm toolchain, model GGUF and a tokenizer-produced `prompt.ids` file. It runs **Qwen generation with the production defaults**, including automatic fast prefill for eligible prompts.

```bash
git clone https://github.com/hyukudan/strix-halo-engine-lab.git
cd strix-halo-engine-lab
export PATH=/opt/rocm/core-10.1/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/bin/amdclang++ \
  -DSTRIX_ENABLE_HIP_RUNTIME=ON \
  -DSTRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME=ON \
  -DSTRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE=ON \
  -DSTRIX_USE_OPENSSL_SHA256=ON -DBUILD_TESTING=ON
ninja -C build strix-hip-decode-token
ctest --test-dir build --output-on-failure
```

Every HIP target builds for the AMD GPU targets in `STRIX_HIP_ARCHITECTURES` (default `gfx1151`); to add an
architecture, pass for example `-DSTRIX_HIP_ARCHITECTURES="gfx1151;gfx1160"`. The runtime still decides which kernel
sets a given architecture may run.

With the matching Qwen model and token IDs available in your working directory:

```bash
./build/strix-hip-decode-token Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf $(cat prompt.ids) --generate 256
```

Add `--print-effective-config` to inspect before HIP initialization. See [Defaults and off switches](docs/methodology.md#defaults-and-off-switches) for baseline scope.

The command produces a JSON receipt with token IDs and timing. Actual model files and prompt tokens are **not included**. For model assets, optional MTP/DSpark acceleration, serving and two-node transport, continue with [requirements](docs/getting-started/README.md#what-you-need), [build](docs/getting-started/README.md#build), [run](docs/getting-started/README.md#run), and the [full engine reference](docs/getting-started/engine-reference.md). Some CTest cases require the GPU and model assets.

Requirements, the complete build for every engine and each engine's launch command:
[getting started](docs/getting-started/README.md). Optional NVIDIA GPU routes: [using a discrete GPU](#discrete-gpu).

## Quick setup for the DeepSeek pair

Two Strix Halo machines with 128 GB each and two USB4 cables between them: one script checks each node, builds the
USB4 driver modules and the engine, fetches and verifies the weights, starts both ranks and runs an exact smoke test.
Run on both nodes (`--rank 1` and the other address on the second one):

```bash
scripts/pair/ds41-pair init --rank 0 --peer user@node-b --model-dir /path/to/models
scripts/pair/ds41-pair check && scripts/pair/ds41-pair driver && scripts/pair/ds41-pair build && scripts/pair/ds41-pair fetch
```

then `scripts/pair/ds41-pair run --both` and `scripts/pair/ds41-pair smoke` on the rank 0 node. Root steps are printed,
never run silently. Every subcommand: [docs/pair-setup.md](docs/pair-setup.md).

## Profiles: `default`, `fast`, `quality`

Qwen (CLI, `--http-serve` and the MTP runner), every DeepSeek pair app and Nemotron take `--profile default|fast|quality`;
run the same profile on both ranks of a pair. `default` is the exact production configuration and sets nothing. An
explicit option or environment variable always wins over the profile, including explicit off.
`--print-effective-config` (standalone RTX: `--print-config`) shows the selected profile and the state of every option
before any device work. `--allow-pending` runs pending options whose code is in the build, for benchmarking only,
stamped `unqualified`.

| Engine | `fast` | `quality` |
|---|---|---|
| Qwen3.8-Flash-Next | BF16 MoE route storage (`STRIX_QPF2K_MOE_ROUTE_BF16=1`) where the route admits it: bit-changing, qualification in progress, the receipt marks the row `provisional`. 262K prefill 1,505 vs 1,464 tok/s (`669bcb83`), 32K TTFT 21.83 → 21.15 s (−3.2 %). Chunked GDN FP16 prefill stays pending (`--allow-pending` runs it: 32K −4.1 %, 262K −4.5 %). | No quality pack for this model: runs `default` and says so in the receipt. |
| DeepSeek V4.1 Flash pair | Relaxed-order MoE (`STRIX_DS41_FAST_MOE=1`, down projection on 32-row groups): bit-changing, qualification in progress. Server TTFT 7.56 / 26.33 / 36.77 s vs 8.36 / 29.38 / 41.55 s `default` in the same run. | No pack: runs `default` with a notice. |
| Nemotron 3.5 Bolt E4B | Int8 activations (`g4-a8-v1`, qualified against frozen fidelity bands): C1 99.5 tok/s shipped, C8 340.6 tok/s; top-1 vs BF16 86.7 %. Prompt prefill stays exact; v1 prompt prefill for `fast`: qualification of the pair v1 prefill + a8 decode pending. | EXL3 M4 pack (`--profile quality --exl3-arm DIR`): top-1 agreement with BF16 92.94 % vs 87.0 % for Q4_K_M, mean KL 0.0367 vs 0.141; 96.4 tok/s at C1 shipped. |

Which Qwen profile to use: start with the plain command (dense-A8 decode and eligible fast FP16 CLI prefill, each exact
to its own reference). Use the FP32-dense profile when you need that arithmetic; the two pick different tokens where
two candidates are nearly tied (2.8 % of positions), and each is deterministic and exact with respect to itself.
Per-engine profile details: [Qwen](docs/models/qwen3.8/README.md#qwen-profile),
[DeepSeek pair](docs/models/deepseek-v4.1/README.md#deepseek-pair),
[Nemotron](docs/models/nemotron-3.5-bolt-e4b/README.md#profiles); GLM on one RTX has its own `--profile` in its
[guide](src/glm/rtxpf/README.md).

<a id="discrete-gpu"></a>

## Using a discrete GPU (optional)

Everything above runs on Strix Halo alone. An RTX PRO 6000 (96 GB) adds four opt-in routes; none of them changes the
single-node or pair defaults, and each has its own declared identity and gates. Tested on Windows 11, NVIDIA driver
617.14, CUDA 13.0; select the card by UUID (`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=GPU-<uuid>`).
Requirements, one build or launch per route, what runs where, when a route does not help and troubleshooting:
[using a discrete GPU](docs/discrete-gpu.md).

- **DeepSeek V4.1 Flash, standalone RTX (Production, Windows WDDM).** The whole model on the card: Q2 GGUF routed
  experts, Q8_0 dense, an exclusive VRAM + pinned-host expert cache, the target and the DSpark draft on the same GPU.
  82.3 / 87.7 / 94.4 tok/s on code / English / Spanish at the same load. Persistent file-queue worker (`--serve`),
  two-drive expert striping. [Results](docs/models/deepseek-v4.1/README.md#deepseek-standalone-rtx-results) ·
  [build, run and defaults](docs/models/deepseek-v4.1/rtx-standalone.md).
- **DeepSeek V4.1 Flash, RTX + pair (Production, D-064).** The RTX runs layers 0-21 with CUDA, the two nodes run layers
  22-39 and the head, over 1 GbE: 60.1 / 97.5 / 137.2 tok/s aggregate at 2 / 4 / 8 rows; exact prompt reading on the
  RTX 3.64 / 4.39 / 12.20 s at 2K / 8K / 32K. [Results](docs/models/deepseek-v4.1/README.md#deepseek-rtx-pipeline) ·
  [pipeline stage launch contract](docs/models/deepseek-v4.1/pipeline-stages.md).
- **Qwen3.8-Flash-Next, prefill offload.** Exact CUDA prefill on the RTX with the attention and recurrent state
  streamed to the node as each layer finishes; the node decodes. 32K TTFT 8.99 s against 44.02 s on the same server
  without the RTX; 64 generated tokens identical at every length.
  [Results](docs/models/qwen3.8/README.md#qwen-prefill-offload).
- **GLM-5.3 Flash on one RTX PRO 6000 (draft, opt-in, Windows).** Trunk, dense layers and shared experts on the card;
  the 43 × 288 routed experts streamed through the [tierstream](https://github.com/hyukudan/tierstream) device-gated
  cache across VRAM, pinned RAM and disk. Exact prefill and decode; TTFT 11.4 / 30.5 s at 8K / 32K on the full card,
  and it also runs exact with 24 GiB VRAM + 32 GiB host RAM.
  [Results](docs/models/glm-5.3/README.md#glm-rtx-results) · [guide](src/glm/rtxpf/README.md).

## MiniMax H3 audio + video (opt-in build)

**A 5 s 1344x768 video clip with stereo audio in 405.5 s on the USB4 pair.** Text to video + audio, 124 frames,
seed 42, 6-step LoRA, sparse attention, fast VAE; one node, the two-node schedule on one node and the two nodes give
the same bits for that identity. Build with `-DSTRIX_FAMILY_H3=ON` (nothing is built without it); each speed lever has
its own flag, and without them the run is the dense reference.
[Install, model files, one command per clip and rollback](docs/models/minimax-h3/README.md#h3-install) ·
[stage breakdown](docs/models/minimax-h3/README.md#h3-result).

## Documentation

- **Getting started:** [requirements, build and run](docs/getting-started/README.md) ·
  [engine reference](docs/getting-started/engine-reference.md) (build options, full CLI, environment switches) ·
  [benchmark kit and reproduction](docs/reproduce.md).
- **Models:** [Qwen3.8-Flash-Next](docs/models/qwen3.8/README.md) · [DeepSeek V4.1 Flash](docs/models/deepseek-v4.1/README.md) ·
  [MiniMax M3.1 Flash](docs/models/minimax-m3.1/README.md) · [GLM-5.3 Flash](docs/models/glm-5.3/README.md) ·
  [Nemotron 3.5 Bolt E4B](docs/models/nemotron-3.5-bolt-e4b/README.md) · [model families](docs/models/model-families.md).
- **[Methodology](docs/methodology.md):** how the tables are read, exactness, model fidelity, defaults and off switches,
  design decisions.
- **[Two nodes over USB4](docs/pair.md):** the pair, what it needs, how DeepSeek, MiniMax and GLM run on it;
  [setting up the DeepSeek pair](docs/pair-setup.md) with one script.
- **[Serving](docs/serving.md):** HTTP servers, request contracts, batching, prefix caches, exposure and security.
- **[Measured gains](docs/measured-gains.md):** every shipped default with its measured gain, exactness evidence and off
  switch · [optimization catalog](docs/optimizations.md) ([JSON](docs/optimizations.json)) ·
  [performance records](docs/performance.md).
- **Internals:** [architecture guide](docs/architecture/README.md) · [decision register](docs/decisions/decision-register.md) ·
  [implementation status](docs/reference/implementation-status.md) · [qualification ledger](docs/reference/qualification-ledger.md) ·
  [sources and provenance](docs/reference/sources-ledger.md) · [full documentation index](docs/README.md) ·
  [AGENTS.md](AGENTS.md) for coding agents.

<a id="related-repositories"></a>

## Related repositories

- [strix-halo-usb4-driver](https://github.com/hyukudan/strix-halo-usb4-driver): the Linux patch series that turns
  the Strix Halo USB4 controllers into the direct link between the two nodes.
- [strix-trellis](https://github.com/hyukudan/strix-trellis): the trellis-quantization (EXL3/QTIP-style) library
  used for DeepSeek's dense weights and MiniMax inference.
- [TensorMeld](https://github.com/hyukudan/tensormeld): placement and routing across nodes and GPUs; its router
  serves the Qwen replicas.
- [tierstream](https://github.com/hyukudan/tierstream): tiered expert streaming across VRAM, pinned host memory
  and NVMe used by the standalone DeepSeek RTX route.

<a id="license"></a>

## License

The engine source is licensed under the **[Apache License 2.0](LICENSE)**, except the Linux kernel
module under `lab/direct_gpu_stream/kernel/`, which is **GPL-2.0-only** as Linux kernel code must be (its license
texts are kept in that directory). Code adopted from other projects keeps its original license and notices. Model
weights are distributed under their own licenses. **Built with MiniMax M3.1-Flash**; its model license
is non-commercial by default. Preserve model and conversion notices and review the applicable terms. Third-party references used for formats and model semantics are
listed with their licenses in [docs/reference/sources-ledger.md](docs/reference/sources-ledger.md).
