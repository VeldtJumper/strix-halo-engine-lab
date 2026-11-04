# Documentation index

One line per document. The project overview is the top-level [README](../README.md)
([Español](../README.es.md)); the normative architecture guide starts at
[architecture/README.md](architecture/README.md). Documents marked *Historical* record
work as it stood on their date; documents marked **Superseded** name their replacement.

## Getting started

- [Getting started: requirements, build and run](getting-started/README.md) ([Español](getting-started/README.es.md)): Requirements per engine, the complete build, where each engine's launch command lives.
- [Using a discrete GPU (optional)](discrete-gpu.md) ([Español](discrete-gpu.es.md)): Tested card, driver and CUDA, what runs where for each NVIDIA route, measured gains, when a route does not help, pinning the GPU, troubleshooting.
- [Engine reference](getting-started/engine-reference.md): Decode configuration, build options, full CLI, prefill, draft vocab, env switches, repo layout, docs map.

## Models

- [Qwen3.8-Flash-Next](models/qwen3.8/README.md) ([Español](models/qwen3.8/README.es.md)): Results, launch commands, profiles, long context, GSQ, bit-changing prefill options and tuning switches on one node.
- [DeepSeek V4.1 Flash](models/deepseek-v4.1/README.md) ([Español](models/deepseek-v4.1/README.es.md)): Results, launch commands and profiles for the USB4 pair, the RTX pipeline split and the standalone RTX.
- [MiniMax M3.1 Flash](models/minimax-m3.1/README.md) ([Español](models/minimax-m3.1/README.es.md)): Results, quality gate and launch command on the USB4 pair.
- [GLM-5.3 Flash](models/glm-5.3/README.md) ([Español](models/glm-5.3/README.es.md)): Results and launch commands for the USB4 pair and for one RTX PRO 6000.
- [Nemotron 3.5 Bolt E4B](models/nemotron-3.5-bolt-e4b/README.md) ([Español](models/nemotron-3.5-bolt-e4b/README.es.md)): Profiles, assets, build and run, results and off switches on one node.
- [MiniMax H3](models/minimax-h3/README.md) ([Español](models/minimax-h3/README.es.md)): Opt-in audio + video build: the 405.5 s pair clip with its stage breakdown, install, model files, one command per clip, rollback, packages and switches.
- [Model families](models/model-families.md): Model-family registry in the core and how to add a family; deepseek41 as worked example.
- [deepseek41 decode attention (CSA2)](models/deepseek-v4.1/attention.md): deepseek41 CSA2 decode attention on gfx1151: shapes, indexer, RoPE, declared arithmetic, host references.
- [Sixteen-row BatchForward steps](models/deepseek-v4.1/batch-rows16.md): STRIX_DS41_BATCH_ROWS16: up to sixteen rows per BatchForward step.
- [deepseek41 CUDA attention (sm_120)](models/deepseek-v4.1/cuda-attention.md): deepseek41 CSA2 attention on CUDA sm_120, bitwise equal to host reference.
- [CUDA prefill with bounded passes](models/deepseek-v4.1/cuda-bounded-pass-prefill.md): Experimental CUDA prefill with bounded passes (--pass-rows) for million-token contexts.
- [Exact CUDA decode launch fusions](models/deepseek-v4.1/cuda-decode-fusions.md): Exact CUDA decode launch fusions --fuse-hc and --fuse-shared for decode and pipeline FRONT.
- [deepseek41 CUDA fast prefill attention](models/deepseek-v4.1/cuda-fast-attention.md): Non-bitwise fast CUDA prefill attention/indexer (--fast-attention) using mma.sync FP16/FP8.
- [deepseek41 on CUDA: numeric identity of the layer core](models/deepseek-v4.1/cuda-numeric-identity.md): deepseek41 CUDA layer-core kernels: how ds41-exact-v1 bitwise identity is kept on sm_120.
- [CUDA prefill MoE pilot](models/deepseek-v4.1/cuda-prefill-moe-pilot.md): Draft CUDA prefill fast-MoE options: wide tiles and signed-LUT IQ2 decode, plus route capture.
- [CUDA trellis shared expert (split decode FRONT on the pair's mix-B identity)](models/deepseek-v4.1/cuda-trellis-shexp.md): CUDA trellis shared expert for split-decode FRONT on mix-B identity (--trellis-shexp).
- [deepseek41 DSpark drafter](models/deepseek-v4.1/dspark-drafter.md): Host reference, device kernels, greedy verify protocol and window choice.
- [deepseek41 expert server and parts-host (strix/ds41-expert-rpc-v1)](models/deepseek-v4.1/expert-server.md): deepseek41 expert server and parts-host RPC protocol (ds41-expert-rpc-v1): roles, wire, backends.
- [ds41-expert-tp2-i1152-v1: routed experts split by the intermediate dimension over two ranks](models/deepseek-v4.1/expert-tp2-i1152-v1.md): Numerical identity splitting routed experts by intermediate dim over two ranks; two-rank default since P16.
- [deepseek41 weight and cache formats (decode)](models/deepseek-v4.1/formats.md): Decoders, A8 activation contract, routed-expert row products.
- [deepseek41 hot-expert placement (reader and layer split)](models/deepseek-v4.1/hot-placement.md): Hot-expert placement file format (hot-expert-placement-v1) and local/remote row split.
- [deepseek41: layer sites outside attention and the MoE](models/deepseek-v4.1/layer.md): deepseek41 non-attention/non-MoE layer sites and ds41-exact-v1 reduction orders, HC, Engram.
- [deepseek41: long context (128K-512K) on two nodes](models/deepseek-v4.1/long-context.md): deepseek41 long context (128K-512K) on two nodes: attention state size, memory, shared KV budget.
- [DeepSeek V4.1 Flash: model contract (host only)](models/deepseek-v4.1/model-contract.md): Host-only admission contract for DeepSeek V4.1 Flash GGUF and DSpark draft: config, Engram, inventory, pins.
- [deepseek41 MoE layer (decode)](models/deepseek-v4.1/moe.md): deepseek41 MoE layer decode: window routes, router, shared and routed experts, combine order.
- [deepseek41 pipeline stages (`strix/ds41-pipeline-v1`) and `--describe-json`](models/deepseek-v4.1/pipeline-stages.md): deepseek41 FRONT/TAIL pipeline stages (ds41-pipeline-v1) protocol, command lines and --describe-json.
- [deepseek41 prefill: routed experts and exact suffix replay](models/deepseek-v4.1/prefill.md): WMMA routed experts over prompt blocks, bit-equal to decode, and exact suffix replay.
- [Prepared rank-local expert loading (v1)](models/deepseek-v4.1/prepared-load.md): Opt-in prepared rank-local routed-expert loading (STRIX_DS41_PREPARED_LOAD) and strix-ds41-prepare-load.
- [DeepSeek-V4.1 two-node: production defaults (P17.2)](models/deepseek-v4.1/production-defaults.md): DeepSeek-V4.1 two-node production defaults (P17.2) from the P17.1 timing campaign.
- [Experimental state transfer v2](models/deepseek-v4.1/state-transfer-v2.md): Experimental state transfer v2 receiver stub and proposed HTTP import contract.
- [Numerical identity ds41-trellis-k5-v1 (QTIP-style trellis dense decode sites)](models/deepseek-v4.1/trellis-k5-v1.md): QTIP-style trellis dense decode sites, mixA/mixB identities.
- [Trellis library adapters](models/deepseek-v4.1/trellis-library-api.md): STRIX_DS41_TRELLIS_API CMake selection of trellis library adapters (v12, tp03, splitk04).
- [Optional single-session latency kernels](models/qwen3.8/latency-kernels.md): Optional STRIX_QKERN_* single-session latency kernel switches for Qwen decode.
- [Qwen attention, draft and recurrent dispatch experiments](models/qwen3.8/persistence-kernels.md): STRIX_QPERSIST_* attention, draft and recurrent persistent dispatch experiments for Qwen.
- [Shared down R8 organization (draft)](models/qwen3.8/shared-down-r8.md): Draft STRIX_QWEN_SHARED_DOWN_NARROW shared-expert down projection organization for R8 windows.


## Performance

- [Measured performance](performance.md): All measured speed numbers with date, config and exactness check, per model and for the USB4 link.
- [Methodology](methodology.md) ([Español](methodology.es.md)): How the tables are read, exactness per engine, model fidelity, defaults and off switches, design decisions.
- [Measured gains](measured-gains.md) ([Español](measured-gains.es.md)): Changelog of every shipped default with its measured gain, exactness evidence and off switch.
- [Two nodes over USB4](pair.md) ([Español](pair.es.md)): The pair, what it needs, how DeepSeek, MiniMax and GLM run on it, links to the transport internals.
- [Serving over HTTP](serving.md) ([Español](serving.es.md)): Which server each engine has, serving features, exposing the servers safely.

## Architecture (normative)

- [01 — Target Platform](architecture/01-target-platform.md): Normative: Strix Halo gfx1151 on Linux/ROCm target, platform assumptions, consequences, non-goals.
- [02 — Memory and Residency](architecture/02-memory-and-residency.md): Normative unified-memory budget: memory classes, admission, host reserve, file-backed, registered, huge pages, pressure.
- [03 — Model Loading and Weight Layout](architecture/03-model-loading-and-weight-layout.md): Normative loader stages, model manifest, tensor binding, storage/internal cache, quantized weights, placement, partitioning.
- [04 — Execution Model and Graphs](architecture/04-execution-and-graphs.md): Normative static phase-specific executor: buffers, execution identity, eager failure containment, graph capture, sync.
- [05 — Kernel Design](architecture/05-kernel-design.md): Normative kernel registry, specialization, fusion, matrix ops, MoE, recurrent and attention kernels.
- [06 — Quantization and Numerical Rules](architecture/06-quantization-and-numerics.md): Normative numerical contract: validation layers, plan identity, accumulation, row invariance, ties, KV quant, golden refs.
- [07 — KV, Recurrent State and Snapshots](architecture/07-state-kv-and-recurrence.md): Normative per-request state: KV layout, recurrent state, snapshots, prefix reuse, eviction.
- [08 — Scheduling and Concurrency](architecture/08-scheduling-and-concurrency.md): Normative C1 fast path, small-batch, compute/transport schedulers, prefill vs decode, fairness, backpressure.
- [09 — Speculative Decoding](architecture/09-speculative-decoding.md): Normative speculative decoding contract: state, draft depth, greedy acceptance, sampling, verification, commit, metrics.
- [10 — Benchmarking and Qualification](architecture/10-benchmarking-and-qualification.md): Normative benchmarking/qualification: environment fingerprint, benchmark classes, transport matrix, correctness gates, A/B discipline.
- [11 — Runtime Observability and Recovery](architecture/11-runtime-observability-and-recovery.md): Normative telemetry, structured logs, failure containment, dirty termination, admin boundary, runtime profiles.
- [12 — Multi-Node and USB4 Interconnect](architecture/12-multi-node-and-usb4.md): Normative multi-node over USB4: transport abstraction/API, dual-link strategies, partitioning, framing, qualification gates.
- [13 — Platform Discovery and Capability Negotiation](architecture/13-platform-discovery.md): Normative platform discovery: GPU/UMA/USB4 inventory, DMA capability, capability levels, peer topology, JSON contract.
- [14 — Implementation Roadmap](architecture/14-implementation-roadmap.md): Implementation roadmap in slices A0/A1/A2 with deferred work and integrated optimization sequence. *Historical.*
- [15 — Optimization Strategy](architecture/15-optimization-strategy.md): Normative optimization strategy: cost per committed token, low-row and MoE-aware verify, adaptive speculation, distributed hierarchy.
- [16 — Direct GPU Transport over USB4](architecture/16-direct-gpu-transport.md): Normative direct GPU DMA-BUF transport over USB4: pools, terminal commit, RX priming, dual-link, qualification, promotion.
- [Strix Halo Architecture Guide](architecture/README.md): Index of the normative architecture guide: decision language, priorities, document list, speculative path, draft investigations.

## Engine: kernels

- [Batched routed-expert activation](engine/kernels/batched-moe-activation.md): --batched-moe-activation: one 2D SiLU kernel replaces per-slot routed-expert activation submissions.
- [Dense Q6P32 weights (`--dense-q6p-pack`)](engine/kernels/dense-q6p32-weights.md): Draft Q6P32 6.5-bit dense weight format and bit-exact kernels (--dense-q6p-pack).
- [GDN recurrent LDS layout diagnostic](engine/kernels/gdn-recurrent-layout.md): GDN recurrent-state LDS layout diagnostic (stride32/33/transposed) and --gdn-recurrent-layout option.
- [Persistent HC mix for verify-window rows](engine/kernels/hc-persistent-window-rows.md): STRIX_HC_PERSISTENT: one persistent launch per HC step of a verify window.
- [Experimental HC projection prefetch](engine/kernels/hc-projection-prefetch.md): Experimental --prefetch-hc-projection-next-tile prefetch of next HC projection tile.
- [Experimental Q4K A8G32 native gate/up path](engine/kernels/q4k-a8g32-native-gate-up.md): Experimental opt-in Q4_K A8G32 integer gate/up decode path (--q4k-a8g32-gate-up): arithmetic, ownership, qualification.
- [Standalone Q4_K gate/up A8G32 probe](engine/kernels/q4k-a8g32-probe.md): Contract, arithmetic, status words, build and ISA gates. *Historical.*
- [Experimental exact grouped Q5_1 down probe](engine/kernels/q51-grouped-down-probe.md): Standalone exact grouped Q5_1 down probe: workgroup layout, shapes, host tests, build target. *Historical.*
- [Opt-in exact grouped Q5_1 down runtime](engine/kernels/q51-grouped-down-runtime.md): Opt-in exact grouped Q5_1 down-projection runtime (--grouped-exact-q51-down): contract, identity, qualification status.
- [Slot-parallel Q5_1 down projection](engine/kernels/q51-slot-parallel.md): Experimental slot-parallel Q5_1 down projection with ordered weighted merge.
- [Experimental Q8 projection organizations](engine/kernels/q8-projection-organization.md): Experimental Q8 LM-head and GDN input projection organizations (prefetch, wave8).
- [Selected-attention GQA reuse diagnostic](engine/kernels/qsa-gqa-reuse.md): QSA GQA reuse: 2/4 query heads per workgroup sharing KV (--qsa-gqa-heads-per-block).
- [Hierarchical QSA selection diagnostic](engine/kernels/qsa-hierarchical-selection.md): Hierarchical QSA top-512 block selection (--qsa-hierarchical-selection) diagnostic and integration.
- [Experimental staged router logits](engine/kernels/staged-router-logits.md): Experimental staged router logits organization (--staged-moe-router-logits).
- [Experimental virtual Q8 GDN input projection](engine/kernels/virtual-q8-gdn-input.md): Experimental --virtual-q8-gdn-input one-wave Q8 GDN input projection.
- [Experimental prefetch8 Q8 GDN input variant](engine/kernels/virtual-q8-gdn-prefetch.md): Experimental --prefetch-q8-gdn-input prefetch8 variant of virtual Q8 GDN input.
- [Verify-window step folds](engine/kernels/window-step-folds.md): Opt-in verify-window step folds (inject/norm, combine/inject, commit/combine) via env switches.

## Engine: decode

- [Int8 attention KV cache (kv-int8-tok-v1)](engine/decode/attention-kv-int8.md): Opt-in int8 per-token-scaled QSA attention KV cache (kv-int8-tok-v1): storage layout, admission, validation.
- [Deferred eager MoE status](engine/decode/deferred-moe-status.md): Deferred eager MoE completion-status policy; included in adopted automatic v3 profile.
- [Requested generation output](engine/decode/generation-output.md): --logprob-token: requested token logprobs during serial --generate via bounded ledger.
- [Experimental scalar greedy output](engine/decode/gpu-greedy-output.md): Experimental --gpu-greedy-token: on-GPU argmax over full LM-head output with validated reply.
- [Automatic decoder optimization profile](engine/decode/optimization-profile.md): Automatic content-admitted decoder optimization profile (auto v3): selection rules, qualified adoption, limits.
- [Experimental prepared eager submission](engine/decode/prepared-eager-execution.md): Experimental --prepared-eager immutable prepared submission of the compiled serial execution.
- [Sampled decoding with speculative drafts](engine/decode/sampled-decoding.md): Sampled decoding (temperature/top-k/top-p) with speculative MTP drafts, sampling-v1 identity.

## Engine: prefill

- [Known input windows](engine/prefill/known-input-windows.md): Known-input windows: layer-major traversal of 2/4/8 known inputs (--known-prefix-window).
- [Explicit known-prefix prefill](engine/prefill/known-prefix-prefill.md): Early source-only note on --known-prefix-prefill-rows T (16..2048) step-major prefill.
- [Prefill numerical profile](engine/prefill/prefill-numerical-profile.md): --prefill-numerical-profile: separate prefill arithmetic from decode numerical profile; admitted pairs.
- [Prompt groups: several requests' prompt blocks in one prefill run](engine/prefill/prompt-batch.md): STRIX_PROMPT_BATCH: several requests' tiled prompt blocks in one prefill run.
- [Teacher-forced scoring over the tiled prefill](engine/prefill/teacher-forced-tiled-prefill.md): Teacher-forced scoring (--teacher-forced-score) over the tiled known-prefix prefill.
- [Step-major tiled prefill](engine/prefill/tiled-prefill.md): Step-major tiled prefill route (--known-prefix-prefill-rows T, 16..8192) organization and identities.

## Engine: speculative decoding

- [Decode defaults (D-055)](engine/speculative/decode-defaults.md): MTP draft profiles, min-p and window switches chosen from final campaign.
- [Persisted MTP window-cost tables](engine/speculative/mtp-cost-tables.md): Window-cost tables that give the Qwen min-p MTP controller a bootstrap prior across processes (key, OFF).
- [Depth-aware MTP policy (draft)](engine/speculative/mtp-depth-controller.md): Draft depth-aware MTP controller (STRIX_MTP_CONTROLLER=depth) using measured costs and acceptance.
- [Independent MTP forward golden file](engine/speculative/mtp-forward-golden-file.md): Binary format and reader for independent MTP forward golden reference files.
- [Native MTP fixture runner](engine/speculative/mtp-model-qualification.md): Native MTP fixture runner and target-input export for independent reference producers.
- [Explicit native MTP CLI](engine/speculative/mtp-native-cli.md): Explicit native MTP CLI options (--mtp, golden pins, tolerances, --mtp-metadata-only).
- [Native MTP source implementation](engine/speculative/mtp-native-runtime.md): Sidecar assets, arithmetic, draft/verify runtime and state.
- [MTP sidecar source and explicit shared tensors](engine/speculative/mtp-sidecar-assets.md): MTP sidecar loader profile and explicit shared target embedding/output tensors.
- [Observed prompt lookup depth](engine/speculative/prompt-lookup-adaptive.md): --prompt-lookup-adaptive: bounded depth policy (0-2) for prompt lookup by observed cost.
- [Prompt lookup](engine/speculative/prompt-lookup.md): Explicit occupied-KV snapshot format for the lookup route.
- [Multi-row verify window: design and status](engine/speculative/verify-window-plan.md): Multi-row verify window design and status: profiling motivation, design, result, kernel work packages.

## Engine: scheduling

- [Joint serving windows (experimental)](engine/scheduling/joint-serving-windows.md): STRIX_MTP_BATCH_JOINT: joint R2/R4/R8 window scheduling across batched HTTP requests.
- [Conservative MoE fork/join arena](engine/scheduling/moe-fork-join-arena.md): Conservative MoE fork/join arena planner for shared/routed overlap in C1/R1 calendar.
- [Experimental shared-expert fork/join](engine/scheduling/moe-shared-overlap.md): Experimental shared-expert fork/join on a sibling stream (--moe-shared-overlap).
- [Observed MoE token schedule](engine/scheduling/moe-token-cost-policy.md): --moe-token-cost-policy: observed per-token choice between serial and fork/join MoE schedules.
- [Several requests in one decode loop (cohort windows)](engine/scheduling/multi-request-cohort.md): Several independent requests in one decode loop for aggregate throughput.

## Engine: serving

- [Greedy serving](engine/serving/greedy-serving.md): Greedy serving options: shared-projection cohorts and known-input prefill chunks with budgets.
- [Prefix state cache](engine/serving/prefix-state-cache.md): Certified prefix reuse defaults on for the qualified single tiled-MTP HTTP family; late memory admission, OFF controls and explicit host-budget contract.
- [Serving prefill and prefix-cache composition](engine/serving/serving-prefill-prefix-cache.md): Composition of serving prefix cache with native prefill producer (exact-length lookup).
- [Serving over HTTP](engine/serving/serving.md): HTTP serving (--http-serve): OpenAI/Anthropic-compatible API, queue, batching, options.

## Engine: transport, USB4 and multi-node

- [Two-rank communicator](engine/transport/communicator.md): Two-rank distributed communicator: memory/TCP/USB4 transports, automatic USB4 pairing, exchange protocol.
- [Native known-input sequence component](engine/transport/distributed-known-sequence.md): HipDistributedKnownSequence: two-peer known-input sequence over native transport with stage-owned weights/state.
- [Stage-owned storage inventory](engine/transport/distributed-owner-inventory.md): Stage-owned storage inventory API for distributed stage resources and token participants.
- [Coherent staged readiness](engine/transport/staged-readiness.md): STRIX_DIST_STAGED_READY_BYTES: coherent staged readiness host path in communicator.
- [Cooperative controller recovery and safe domain retirement](engine/transport/usb4-controller-recovery.md): USB4 driver patch 0009: cooperative controller recovery, domain retirement, controller groups, USB4NET coexistence.
- [Integrated USB4 multi-slot engine: build and test entrypoint](engine/transport/usb4-gpu-queue-integration.md): Entrypoint for integrated USB4 multi-slot GPU queue (strix-gpu-queue-visibility): contract, recovery, build, two-host run.
- [GPU visibility probes — compatibility route and current multi-slot route](engine/transport/usb4-gpu-visibility-probes.md): Original PR #27/#28 C1 compatibility routes, shared probe contract, build/tests. **Superseded** by [Integrated USB4 multi-slot engine: build and test entrypoint](engine/transport/usb4-gpu-queue-integration.md).

## Operations

- [Complete suite evidence](operations/ctest-evidence.md): ctest_evidence.py: turns an existing CTest JUnit report plus frozen expected names into a suite receipt.
- [Device queue trace](operations/device-queue-trace.md): --device-queue-trace: HIP event timing of eager dispatcher main and shared-expert side queues.
- [Execution observability](operations/execution-observability.md): Offline runtime inventory tool and host dispatcher telemetry for attributing execution cost.
- [Validation-only final generation state](operations/final-generation-state-validation.md): --validation-final-state: export final session state allocations after an eager generation for validation.
- [UMA/GTT budget — working draft](operations/memory-budget.md): UMA/GTT working-draft budget for 128 GB: GTT sizing, host reserve, admission categories, admin UI.
- [Selective model weight backing](operations/model-weight-residency.md): Opt-in selective model weight backing (--model-copy-tier) into owned HIP allocations.
- [Persisted offline tuning preparation](operations/offline-tuning.md): --offline-tuning-cache: persisted versioned offline tuning records and their keys.
- [Base runtime owner inventory](operations/runtime-owner-inventory.md): --execution-inventory: base runtime owner inventory of model views and allocator pools.
- [Retained UMA owner inventory](operations/uma-owner-inventory.md): build_uma_owner_inventory: retained UMA allocation lease inventory and alias accounting.

## Decisions and evidence

- [Decision register](decisions/decision-register.md): Accepted D-001..D-064, pending P-xxx, rejected, measured-and-rejected; long dated sections per decision.
- [Evidence register](decisions/evidence-register.md): Evidence register E-001..E-056 (observed/published/inferred/hypothesis/pending/withdrawn) backing spec and decisions.

## Reference

- [Generation request timing](reference/generation-request-timing.md): Definition of generation_request_elapsed_ms and output tokens/s fields in the generation receipt.
- [GGUF reader A0.2](reference/gguf-reader.md): GGUF v3 reader (slice A0.2): pinned spec sources, implemented scope, deliberate limits, security.
- [Implementation status](reference/implementation-status.md): Current DeepSeek/Qwen summaries (2026-10-23, 2026-10-17) then slice-by-slice A0.1..A2.2 history.
- [Current qualification ledger](reference/qualification-ledger.md): Current qualification ledger (2026-10-16): per-area implementations, checks and required evidence.
- [Source and provenance ledger](reference/sources-ledger.md): Source/provenance ledger: pinned upstream revisions, licenses and contracts per component; approval rule.

## Research

- [Native runtime investigation scope](research/README.md): Short stub: native runtime investigation scope and publication boundary for research.
- [Experimental selected dense projections](research/dense-trellis-g1.md): Experimental Hadamard/trellis K5 dense projections (dense-a8-trellis-g1) for selected Qwen projections. *Historical.*
- [Design acceptance actions](research/history/2026-09-28-design-acceptance.md): Design-review acceptance requirements recorded 2026-09-28 plus spec v1 draft integration notes (D-015..D-020). *Historical.*
- [Strix Halo optimization review — 2026-09-29](research/history/2026-09-29-optimization-review.md): Dated research review of optimization mechanisms, UMA, dual USB4 and partitioning, with proposed amendments. *Historical.*
- [Platform qualification plan](research/history/platform-review.md): Short stub: platform qualification principles; defers to docs/architecture and implementation ledger. **Superseded** by [13 — Platform Discovery and Capability Negotiation](architecture/13-platform-discovery.md).
- [Technical specification v0.4 — inference engine for Strix Halo](research/history/spec-draft-v0.4.md): Technical specification v0.4 research draft: objectives, principles, architecture, phases A0-E, decision gates. **Superseded** by [Strix Halo Architecture Guide](architecture/README.md).
- [Phase specific packed weight and integer execution proposal](research/proposals/2026-10-11-phase-specific-low-precision.md): Draft proposal (LPE-xxx) for phase-specific packed weights and integer IU8/IU4 execution.
- [Execution optimization gates](research/proposals/2026-10-12-execution-optimization-gates.md): Draft proposal (OPT-xxx): gates for launches, graph lifetime, caches, prefill fairness, numerics, recovery.
- [Shared-expert row reuse diagnostic](research/shared-expert-row-diagnostic.md): Standalone shared-expert R2/R4/R8 row-reuse diagnostic component.
- [Dated implementation notes](research/history/README.md): 35 notes from 2026-10-11 to 2026-10-15, indexed by date.
