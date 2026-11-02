// src/minimax/engine/engine.hip:2056
{"decode-layout", 'a', "model+slot+position+prefix", "msa_fused_tests.hip:run_case fused vs current; rows == alone hardware gate for 1..8 rows", ""},
// src/minimax/engine/engine.hip:1456
{"msa-dense", 'a', "model+position+prefix; 128-key blocks", "msa_fused_tests.hip:run_case dense v3/v4 and n4-mixed/n8-mixed; prefill_batch_cases", ""},
// src/minimax/attn/index_proj.hip:341
{"index-proj", 'a', "index projection FP32 chain", "index_proj_tests.hip; split/ring/LDS bitwise gates", ""},
// src/minimax/engine/engine.hip:1372
{"index-fallback", 'a', "index projection FP32 chain", "mm_kernels.hpp dense_t16 contract; engine index fallback contract; index_proj_tests.hip", ""},
// src/minimax/moe/mmx_moe.hip:1128
{"decode-wide", 'a', "EXL3 vector shape+bits+S", "mm_defaults.cpp moe_wide; rows == alone hardware gate; mmx_moe.hip wide-column contract", ""},
// src/minimax/moe/mmx_moe.hip: make_args / launch_experts4_grid
{"decode-moe-interleave", 'a', "EXL3 matrix+fixed split/k/reduction chain; v4 N8 grid", "hip_mmx_moe_bench.hip --diverse --test four-switch output/workspace/routes and rows==alone gate", ""},
// src/minimax/moe/mmx_moe.hip: single_stream_for / singleton3_core
{"decode-moe-singleton", 'a', "EXL3 matrix+fixed split/k/reduction chain; N/U/S+slot occupancy+K3+narrow W4+Depth4", "hip_mmx_moe_bench.hip --diverse --test reader counters+resources; --diverse-fallback --test wide/K5/W8/Depth8+12", ""},
// src/minimax/moe/mmx_moe.hip: launch_experts4_grid / down4_combine_kernel
{"decode-moe-down-split", 'a', "EXL3 matrix+fixed split/k/reduction chain; same partial scratch, ascending split sum, expert fmaf, shared add", "hip_mmx_moe_bench.hip --diverse --test DOWN_SPLIT=0/1/2 output/workspace/routes, repeated calls and rows==alone gate", ""},
// src/minimax/moe/mmx_moe.hip:1195
{"decode-moe-core", 'a', "EXL3 vector shape+bits+S; router row", "hip-mmx-moe-bench exact-form gates; mmx_moe.hpp arithmetic contract; hip_minimax_g4.hip bitwise gate", ""},
// src/minimax/moe/mmx_moe.hip:1071
{"decode-moe-depth", 'a', "fixed dot2/split history", "mmx_moe.hip launch_experts/launch_experts3/launch_experts4; bench bitwise form gates", ""},
// src/minimax/moe/mmx_moe.hip:2404
{"decode-linears", 'a', "EXL3 vector shape+bits+S", "mmx_moe.hip linear schedule; mm_defaults.cpp linear_v11 bitwise gate; ids==greedy is secondary", ""},
// src/minimax/moe/mmx_moe.hip:2206
{"linear-split", 'a', "input+output+vector policy version", "mmx_trellis_device.hpp:vector_splits and declared finish order", ""},
// src/minimax/moe/mmx_moe.hip:2188
{"linear-resources", 'a', "job descriptors+fixed arithmetic family", "plan_linears; enqueue_linears_impl validation; strict source receipt", ""},
// src/minimax/moe/mmx_prefill.hip:785
{"prefill-tile", 'a', "matrix+fixed splits+WMMA chain", "mmx_prefill_core.hpp header contract; mmx_prefill.hip TILE comment", ""},
// src/minimax/moe/mmx_prefill.hip:639
{"prefill-mt", 'a', "matrix+fixed splits+per-row WMMA chain", "mm_defaults.cpp pf_mtfit bitwise gate; core contract", ""},
// src/minimax/moe/mmx_prefill.hip:573
{"prefill-core", 'a', "matrix+fixed splits+per-row WMMA chain", "mm_defaults.cpp pf_core_v4; every split partial bitwise gate", ""},
// src/minimax/moe/mmx_prefill.hip:710
{"prefill-pipe", 'a', "matrix+fixed splits+WMMA chain", "mm_defaults.cpp pf_moe_pipe bitwise gate; source core8 comments; nondefault combinations need same-partial gate", ""},
// src/minimax/moe/mmx_prefill.hip: gather_transform_token_kernel
{"prefill-gather-tok", 'a', "per-(pair, matrix) input transform statements; tile-major pair layout", "hip-mmx-moe-bench test_prefill_gather: raw tiled xt/at, partials and output bitwise (T 1..4096, crafted, gufuse 0/1); mm_defaults.cpp pf_gather_tok", ""},
// src/minimax/moe/mmx_prefill.hip: enqueue_prefill_dense; src/minimax/engine/engine.hip: dense_qkv
{"prefill-dense-qkv", 'a', "EXL3 matrix+library gemm_splits+per-row WMMA chain+ordered split finish", "hip-mmx-moe-bench test_prefill_dense == strix_gemm bitwise (q/k/v/o, q+k+v, T 9..4096); bits 3/4/5/6/8 only, else the library call; mm_defaults.cpp pf_dense_qkv", ""},
// src/minimax/engine/engine.hip: enqueue_post (STRIX_MM_PF_ROUTER_ROWS=32 -> enqueue_router_tiled)
{"prefill-router-tiled", 'a', "router lane chain (j = lane mod 32 ascending) + xor butterfly + ordered top-4 scan", "hip-minimax-moe-tests router_batch_tests: tiled 32 vs fused/4/8 bytes for every kind and tail; opt-in only (default 8)", ""},
// src/minimax/moe/mmx_prefill.hip:647
{"prefill-fuse", 'a', "single-split WMMA+epilogue contract", "prefill gufuse intermediate/producer debug gate in hip-mmx-moe-bench; default OFF; combinations outside gate remain unqualified", ""},
// src/minimax/moe/mmx_prefill.hip:416
{"prefill-table", 'a', "identical item table bytes", "prefill table ring event ownership; default OFF; A/B pftab gate in rank app", ""},
// src/minimax/attn/msa_fused.hip:1714
{"msa-attend", 'a', "MSA block128/chunk/FP32 combine contract", "msa_fused_tests.hip:run_case versions/head counts/graph replay", ""},
// src/minimax/engine/engine.hip: route::attend_n2_sparse; src/minimax/attn/msa_fused.hip: launch_attend
{"decode-attend-grid", 'a', "MSA block128/chunk/FP32 combine contract; seqs+rows_per_seq+dense+min position", "msa_fused_tests.hip:run_case v3 lds_diet and v4 heads_per_wave 1/2/4 bitwise (engine: v4 hpw1, else diet); bench_cell arms (STRIX_MMATTN_BENCH_DIET)", ""},
// src/minimax/attn/msa_fused.hip:1693
{"msa-grid", 'a', "MSA query+block identity", "msa_fused_tests.hip fused grid3; index tests; same_out score bytes", ""},
// src/minimax/attn/msa_fused.hip:1934
{"msa-scratch", 'a', "fixed unit rows+absolute positions+MSA contract", "msa_fused_tests.hip:prefill_batch_cases and every-batch producer/selection/output bytes", ""},
// src/minimax/engine/engine.hip:1727
{"prefill-schedule", 'a', "immutable ordered chunk list (max_rows fixed)", "serve --resume-check with layer budgets 1/7/60 bitwise gate; engine state+final rows contract; no max_rows equivalence claim", ""},
// src/minimax/engine/engine.hip:1087
{"post-path", 'a', "rank order0,1+post arithmetic", "mm_defaults.cpp xbound bitwise gate; rank xb_check N/verify; mm_kernels post contract", ""},
// src/minimax/engine/engine.hip:1066
{"router-batch", 'a', "router row FP32 chain+expert tie order", "mm_defaults.cpp pf_router_batch bitwise gate; enqueue_router_rows byte contract", ""},
// src/minimax/engine/engine.hip:1670
{"head-path", 'a', "vocab partition+GEMV+lower-id tie", "xfer selftest/argmax_exchange; rows == alone hardware gate; hip-minimax-g4", ""},
// src/minimax/engine/engine.hip:214
{"workspace", 'a', "descriptor+required capacity", "engine Lib::mm; DraftLinears::lib_mm; enqueue_prefill_moe workspace guard", ""},
// src/minimax/engine/engine.hip:579
{"staging", 'a', "same per-row operations and rank order", "hip_minimax_g4.hip bitwise gate; shared-first source sorted expert loop", ""},
// src/minimax/xfer/minimax_xfer.hip:161
{"transport", 'a', "wire tags+rank order+payload bytes", "xfer selftest (xfer-report); mm_defaults.cpp transport hp exact; pfx CRC/tag validation", ""},
// src/minimax/rtxpf/fp32_identity.hpp:73
{"rtx-tile", 'a', "F32 identity version+matrix+input order", "fp32_identity_device.inc f32_core_body; hip-mm-fp32id-tests; mm_fp32_identity_tests.cu", ""},
// src/minimax/rtxpf/fp32_identity.hpp: resolve_core_version; cuda/fp32_identity.cu
{"rtx-core", 'a', "FP32 identity+matrix+input order; v3/v3.1 decode", "mm_fp32_identity_tests.cu default/env/explicit and all 65536 mul1 states vs host; full-model gate at 1K/8K code prompts, both ranks", ""},
// src/minimax/rtxpf/cuda/mm_rtx_prefill_main.cu:635
{"io-cache", 'a', "model tensor content+dtype+rank slices", "weight loader sizes/format validation; actual dependency notices preserved", ""},
// src/minimax/dspark/hip_minimax_dspark.hip:755
{"draft-native-linears", 'a', "drafter model dtype+matrix+reduction contract", "hip_minimax_dspark_tests.hip linears/attention bitwise gates; mm_defaults.cpp draft linears contract", ""},
// src/minimax/dspark/markov_cand.hip:472
{"markov-topc", 'a', "vocabulary+C+candidate score contract", "hip_markov_cand_tests.hip payload v1/v2 vs CPU; mm_defaults.cpp markov_topc_v2", ""},
// src/minimax/engine/engine.hip:436
{"operation-plan", 'b', "model+operation kind+arithmetic profile+ordered units", "apps/minimax_rank_main.cpp teacher-force comparison explicitly rounding not bitwise; not a gate between identities", ""},
// src/minimax/attn/msa_fused.hip:1780
{"context-plan", 'b', "model+absolute position+prefix tokens+MSA block/chunk contract", "MSA host/reference bitwise tests at boundary and long context", ""},
// src/minimax/dspark/hip_minimax_dspark.hip:990
{"draft-context", 'b', "draft model+absolute position+context window+chunk64", "mm_defaults.cpp dspark_attn bitwise gate at positions 300/1000/4095; no claim all profiles match", ""},
// src/minimax/engine/draft_lin.hip:245
{"draft-model", 'b', "target model hash+draft asset hashes+dtype+head index", "DSpark reference and target acceptance verification; explicit --dspark-exl3/--dspark-head", ""},
// apps/minimax_rank_main.cpp:488
{"draft-markov", 'c', "BLOCKED draft-history identity until request fixes method+C+H+vocabulary+seed", "UNCOVERED candidate/full arithmetic-history equality; target greedy verification is a separate gate", "Markov candidate pruning before rank-dot correction/argmax (full vocabulary vs C/H subset)"},
// src/minimax/serve/minimax_lead.hpp: admit / effective request policy
{"draft-policy", 'a', "target prefix+per-row vector arithmetic", "speculative tokens == greedy tested chats; rows == alone hardware gate for 1..8 rows; not universal workload qualification", ""},
// apps/minimax_rank_main.cpp:292
{"import-plan", 'b', "target model+arithmetic profile+original chunk list+position/cut+payload digest", "rtxpf_state_compare.py / rank import comparisons; UNCOVERED cross-profile handoff, explicit blocked gate", ""},
// src/minimax/engine/engine.hip:569
{"model-layout", 'b', "model tensor hashes+TP2 partition+rank order", "G3/G4 model gates; pack validation; F32 model fidelity remains draft pending all gates", ""},
// src/minimax/moe/mmx_prefill.hip:476
{"prefill-routed-split", 'c', "BLOCKED until matrix+arithmetic-version fixed S (no expert occupancy)", "UNCOVERED: known failed identity; existing core/mtfit receipts hold S fixed", "L3 routed gate/up WMMA k-tile24 reset vs continuation (rows1->9,S16->8)"},
// src/minimax/moe/mmx_prefill.hip:477
{"prefill-shared-split", 'c', "BLOCKED until shared matrix+arithmetic-version fixed S", "UNCOVERED: known failed max_rows identity", "L3 shared gate/up WMMA k-tile24 reset vs continuation (rows1->9,S16->8)"},
// src/minimax/engine/engine.hip:227
{"library-prefill", 'c', "BLOCKED until library exposes frozen full matrix arithmetic tuple", "UNCOVERED across row counts; bitwise gate compares fixed row count only", "L0 q_proj strix_gemm core (first internal divergent FP op unavailable without dependency receipt)"},
// src/minimax/engine/engine.hip:1734
{"prefill-unit-cuts", 'c', "BLOCKED until original request prompt+absolute chunk list+profile immutable", "UNCOVERED across max_rows/cache-cut boundaries; server deliberately has no prefix cache", "L0 q_proj GEMM after first changed unit cut (before L3 MoE)"},
// src/minimax/engine/engine.hip:658
{"arithmetic-profile", 'c', "BLOCKED until request stores target+draft arithmetic profile and rejects mismatch", "UNCOVERED profile equality; FP32 path is an opt-in distinct draft identity", "L0 q_proj input transform RN16 materialization vs FP32 transform; draft unchunked PV vs chunk partials"},
// src/minimax/dspark/hip_minimax_dspark.hip:949
{"draft-append", 'c', "BLOCKED until draft append absolute-position plan fixed or pin vector per row", "UNCOVERED append32 vs8 intermediate/cache byte identity", "draft append FC GEMM32 vs GEMV8, then K/V projected context (first library core op opaque)"},
// src/minimax/moe/mmx_prefill.hip:881
{"diagnostic-cores", 'c', "BLOCKED in production; diagnostic-only plan", "UNCOVERED for production by design; expected-wrong-output timing APIs", "first skipped A/weight load or attention compute in diagnostic producer"},
// src/minimax/rtxpf/cuda/mm_rtx_kernels.cu:601
{"rtx-attention", 'c', "BLOCKED handoff until MSA cache/output bytes match exact producer plan", "UNCOVERED full native/CUDA MSA model fidelity for all positions/cuts", "RTX MSA index score or attention QK/PV legacy vs shared (first op needs CUDA byte gate)"},
// src/minimax/engine/engine.hip: park_task / resume_parked; src/minimax/serve/minimax_lead.hpp: nested_candidate
{"prefill-nest", 'a', "model+profile+original ordered chunk/group rows+positions+slot generation; sequential bank ownership", "GPU covered on both ranks, check-only resume-check: 512-token child in 7-layer units with 4608-token parent parked at layers 7/31 beside a decoder, 10/10 IDENTICAL; whole 511-token child with parent parked at layer 17, 6/6 IDENTICAL; complete 128-token continuations, all prompt outputs, committed state, first/continuation logits, frozen groups and standalone B1/mixed decode; HTTP token IDs identical policy ON/OFF; timing evidence in serving documentation", ""},
// Grouped communicator prefill: scheduling only; unchanged numerical and wire plan.
{"prefill-async-start", 'a', "original group rows+producer/post kernels+rank sum order+wire sequence/tags/chunks/CRC", "Measured two-node TP2 IDs identical in every arm; UNCOVERED full cross-binary prompt-output/state/logit and complete-workload gates pending; hip-dist-hostpath-tests covers API lifetimes only", ""},
{"prefill-late-src-fence", 'a', "same main-stream compute/scratch order+per-member source and destination lifetime", "Measured combined balanced 4096+3844 lone-prefill -2.3/-2.5 %, 4096+443 neutral and gated off; UNCOVERED full pause/nest/cancel/bank-reuse exactness pending; hip-dist-hostpath-tests covers immediate/late fences", ""},
// Per-unit deterministic row admission; ordinary start/event schedule on rejection.
{"prefill-async-min-rows", 'a', "all ordered member rows >= STRIX_MM_PF_ASYNC_MIN_ROWS (2048 default, 0 unconditional); global async off wins", "CPU minimax-defaults-tests covers boundaries, member order, zero and global off; measured TP2 token IDs identical; UNCOVERED full prompt-output/state/logit and complete-workload lifetime gates pending", ""},
// src/minimax/dspark/minimax_auto.hpp; apps/minimax_rank_main.cpp; src/minimax/serve/minimax_lead.hpp
{"draft-auto", 'a', "target identity+unchanged OFF prefill plan+greedy verify; frozen structural-v1 or unqualified-v1", "CPU features/admission and Lead traces: prose == OFF, structural C1 uses legacy controller, C2 retires; GPU tokens/logits/committed-state and paired json/code/long/story/prose-es receipts pending", ""},
// src/minimax/session/runtime.cpp; src/minimax/serve/minimax_lead.hpp
{"session-producer", 'b', "model+numerical environment+geometry+single-row suffix+fixed-width-one verification", "session_cache_gate.sh A/L/B IDs, every FP32 row and complete-state digest on both ranks; UNCOVERED native and complete-workload qualification", ""},
// src/minimax/session/session_cache.cpp; include/strix/file_snapshot.hpp
{"session-restore", 'b', "loader-fd asset identities+executable+runtime+schema+frontier+rank-group digest", "minimax_session_cache_tests.cpp envelope/transaction/retarget negatives; session_cache_gate.sh fresh restore and never-saved control; UNCOVERED GPU state equality", ""},
// src/minimax/moe/mmx_moe.hip: moe_grid_slots / launch_experts4_grid
{"decode-moe-tight-grid", 'a', "v4 routed grid.y = min(32, 4N)+1; map_slot rejects y > count before work/counters", "rank --xb-check cfg g vs -: decode logits bitwise N1/N4/N8 + verify window; mm_defaults.cpp moe_tight_grid", ""},
// src/minimax/engine/engine.hip: xb_warm_o
{"decode-warm-o", 'a', "cache state only (side-stream reads, sink store); no consumer of its data", "rank --xb-check cfg o vs -: decode logits bitwise N1/N4/N8 + verify window; --ab xbwarmo matched replay; mm_defaults.cpp xb_warm_o", ""},
// src/minimax/serve/minimax_lead.hpp: lookup_start / lookup_commit / cycle prompt lookup
{"serve-prompt-lookup", 'a', "draft source only (rank 0 chooses window rows from strix_lookup); greedy verify decides every token", "minimax_lead_tests prompt lookup emits the tokens of lookup off; serve A/B ids_fnv per workload with STRIX_MM_LOOKUP=0", ""},
