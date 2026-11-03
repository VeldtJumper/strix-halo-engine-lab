# Qwen host YaRN table contract (opt-in)

The plain serial CLI accepts opt-in static YaRN with explicit
`--rope-yarn 2|4` (or `STRIX_QWEN_ROPE_YARN=2|4`) and explicit `--context C`,
where `262144 < C <= 262144*factor`. Equal CLI/environment values agree;
conflicting non-off factors refuse. CLI `--rope-yarn off` overrides the environment.
`--baseline` or `STRIX_BASELINE=1` suppresses YaRN before YaRN validation and
reports the suppressed request in effective config and stderr. Native capacity
validation still applies, so baseline cannot admit oversized contexts. `off` selects native. This mode changes
outputs at every position, including position zero, and is not exact to native.
On gfx1151 the rotary device fixtures (every rotary site, factors 2 and 4, table
lifetime) match the host table byte for byte and native outputs are unchanged.
Measured single-needle retrieval with the plain command (synthetic filler, one seed):

| Factor | Context | Prompt tokens | Evidence at token | Result | Wall incl. load |
|---:|---:|---:|---:|---|---:|
| 2 | 300,000 | 8,192 | 4,502 | exact (also multi-hop and aggregation) | 11 s |
| 4 | 600,000 | 8,192 | 4,502 | exact | 12 s |
| 2 | 300,000 | 299,000 | 284,051 | exact | 277 s |
| 2 | 393,216 | 393,000 | 373,345 | exact | 355 s |
| 2 | 520,192 | 520,000 | 493,999 | exact | 476 s |

A single-needle pass is a capability check, not an accuracy rate. Model-fidelity and complete-workload qualification are
not done; 1,048,576 tokens is not demonstrated: prefill time grows faster than linearly
(476 s at 520,000 tokens). These retrieval checks do not qualify a complete 1M workload. Admission depends on the PLE pages
a prompt touches: the measured runs used synthetic filler, and diverse text at
the same length touches more pages and can be refused.

```sh
strix-hip-decode-token MODEL.gguf --prompt-ids prompt.ids \
  --rope-yarn 2 --context 300000 --generate 256 --print-effective-config
```

`--prompt-ids` reads whitespace-separated decimal token IDs without shell
argument-size limits. Remove `--print-effective-config` to execute an admitted
request. Inspection exits before device work and prints the resolved rotary
identity/table digest plus an `admission` block in bytes. Native/off inspection
now always includes a `rotary` status with `rope_mode: native`. This is an
inspection schema change; native identity serialization and hash bytes remain
empty and unchanged. Qwen HTTP `/v1/capabilities` declares native rotary at
server scope and YaRN unsupported (`CLI-only`). Both HTTP factories publish
that same contract. The table is uploaded once and retained by all launcher readers.

The eager plain serial route and explicitly pinned MTP route are
admitted subject to their separate resource/asset gates. HTTP, teacher scoring,
graph capture, alternative KV storage, unaudited execution environment controls,
cache/replay/recording and explicit
low-level execution routes refuse with named reasons. No request field may
change process-wide mode. Prompt plus generation must fit the requested
capacity; occupied positions are strictly below capacity. Native CLI parsing
continues to enforce `C<=262144` before resolver or inspection, including with
`--unqualified-context`. The standalone native memory estimator retains its
historical zero-context and above-model sizing behavior (for example,
`strix-memory-plan --tokens 524288` returns JSON). Its capacity bound applies
only when an explicit YaRN descriptor/table is present.

The explicit MTP route is a capacity-bound opt-in. The following
1M f4 candidate changes storage scheduling only: H1024, 256 MiB source cap,
256 score-ring rows, production PIPE255 (unset), Q4 retained and M1 enabled.
It preserves the pinned profile pair, T8192/tails, depth 7, calibration 16,
R2/R4/R8 windows and all immutable asset/policy pins. It is a CPU-admitted
candidate; compiler success and modeled spare do not establish device exactness,
physical pressure, model fidelity or complete-workload qualification.

Use relative assets (`MODEL=assets/MODEL.gguf`, `SIDECAR=assets/SIDECAR.gguf`,
`GOLDEN=assets/C1048576-f4.golden`, `DRAFT_VOCAB=assets/draft-vocab.u32`,
`PROMPT_1048320_IDS=assets/prompt-1048320.ids`). Fill the remaining provenance,
policy and error-bound variables from the independently verified asset bundle;
the v3 golden must match capacity, rotary table and profile identities.

```bash
unset STRIX_QWEN_YARN_HOST_RESERVE STRIX_PREFILL_PIPE # production PIPE defaults to 255
STRIX_PLE_RECLAIMABLE=1 STRIX_PLE_ROLLING=1 \
STRIX_PLE_ROLLING_POSITIONS=1024 STRIX_PLE_ROLLING_CAP_BYTES=268435456 \
STRIX_QSA_SCORE_RING_ROWS=256 STRIX_QSA_MTP_POOL=1 \
STRIX_MTP_DRAFT_Q4_ALL=1 \
strix-hip-decode-token "$MODEL" --prompt-ids "$PROMPT_1048320_IDS" \
  --rope-yarn 4 --context 1048576 --generate 256 --mtp \
  --mtp-sidecar "$SIDECAR" --mtp-sidecar-logical-name "$SIDECAR_LOGICAL_NAME" \
  --mtp-sidecar-sha256 "$SIDECAR_SHA256" --mtp-target-sha256 "$TARGET_SHA256" \
  --mtp-target-content "$TARGET_CONTENT" \
  --mtp-source-repository "$SOURCE_REPOSITORY" --mtp-source-revision "$SOURCE_REVISION" \
  --mtp-source-path "$SOURCE_PATH" --mtp-converter-sha256 "$CONVERTER_SHA256" \
  --mtp-norm-converter-sha256 "$NORM_CONVERTER_SHA256" --mtp-converter-revision "$CONVERTER_REVISION" \
  --mtp-conversion effective-norm-embedding-first-v1 \
  --mtp-normalized-rounding bf16-rne --mtp-projection-rounding bf16-rne \
  --mtp-sidecar-residency register --mtp-sidecar-bytes 2786568256 \
  --mtp-header-bytes 33554432 --mtp-sidecar-retained-bytes 3221225472 \
  --mtp-sidecar-setup-bytes 4294967296 --mtp-golden "$GOLDEN" --mtp-golden-sha256 "$GOLDEN_SHA256" \
  --mtp-golden-max-bytes 6442450944 --mtp-replay-host-bytes 6442450944 \
  --mtp-runtime-bytes 2418528231 --mtp-target-capture-bytes 16777216 \
  --mtp-policy-id "$POLICY_ID" --mtp-max-hidden-error "$MAX_HIDDEN_ERROR" \
  --mtp-max-logit-error "$MAX_LOGIT_ERROR" --mtp-spec-min-p "$SPEC_MIN_P" \
  --mtp-calibration-rows 16 --mtp-draft-chain --mtp-draft-vocab "$DRAFT_VOCAB" \
  --mtp-draft-context-vocab 4096 --mtp-draft-q4 --mtp-candidate-depth 7 \
  --mtp-target-window-rows 8 --mtp-target-window-rows-set 2,4,8 \
  --mtp-window-prefix --mtp-window-hyper-mix --mtp-window-qsa-rows \
  --mtp-window-expert-union --mtp-window-expert-union-bytes 65536 \
  --mtp-window-arena-bytes 2147483648 --mtp-window-capture-bytes 67108864 \
  --mtp-window-checkpoint-bytes 8589934592 --mtp-window-total-bytes 17179869184 \
  --mtp-window-history-intervals 8192
```

The production estimator's synthetic metadata fixture uses target backing
111334654400 B, sidecar 2786568256 B, Q8-convertible payload 2745937920 B,
32768 draft IDs plus 4096 context IDs, and P1048320/C1048576/G256. The complete
rolling owner is charged in every applicable phase; verified credit is counted
once. Against the conservative usable ceiling 127473287168 B, the spare is:

| Phase | Spare B | Spare decimal GB |
|---|---:|---:|
| Setup | 5689874725 | 5.689875 |
| Calibration | 2362357721 | 2.362358 |
| Prompt | 552951069 | 0.552951 |
| Decode | 5546445949 | 5.546446 |
| Drain | 5546445949 | 5.546446 |

The CPU fixture asserts admission at T8192 and at least 500000000 B spare per
phase. It injects success-verified credit for sizing; runtime must verify the
actual backing, fresh drop and physical/GTT/owner observations. MemAvailable
is never increased by credit. Required full-token/logit/state and MTP cache/held
hidden comparisons precede timing claims. Serial retains its independently
resolved storage defaults. This candidate does not make YaRN or f4 MTP automatic.

Environment classification applies equally to automatic defaults and explicit
values, including off switches and masks. It classifies the complete resolved
environment before any admission estimate or verified reclaim callback. A new
or unknown item refuses by name until its code paths have been audited; matching
a default value is not evidence of rotary coverage. Nonrotary controls remain
subject to their normal kernel/shape/resource checks.

| Environment item | Performs RoPE? | Covered site / operation |
|---|---|---|
| `STRIX_QPF2K_SCHED` | No | Must resolve to `12`; owner inventory and tail pruning cover only this schedule |
| `STRIX_QPF2K_DENSE` | No | Dense projections |
| `STRIX_QPF2K_GDN` | No | Gated recurrence |
| `STRIX_QPF2K_GDN_GATES_BATCH` | No | Batched recurrence gates |
| `STRIX_QPF2K_GDN_CORE16` | No | Recurrence core |
| `STRIX_QPF2K_HC` | No | Hyper-connection mix |
| `STRIX_QPF2K_HC_MASK` | No | Hyper-connection selection |
| `STRIX_QPF2K_HC_OUTPUT3` | No | Hyper-connection output |
| `STRIX_QPF2K_HC_OUTPUT3_BAND` | No | Hyper-connection output bands |
| `STRIX_QPF2K_MOE_DA16` | No | Expert activation/down |
| `STRIX_QPF2K_MOE_ROUTE` | No | Expert routing |
| `STRIX_QPF2K_MOE_DEFAULTED` | No | Two-bit MoE provenance mask; both defaults produce `3` |
| `STRIX_QPF2K_PRE_INJECTED` | No | Startup provenance of the injected defaults across an image restart; never a request (the environment sweep skips it) |
| `STRIX_PLE_RECLAIMABLE` | No | Page reclaim |
| `STRIX_STARTUP_MARKERS` | No | Optional monotonic startup diagnostics (`0`/`1`) |
| `STRIX_INTPREFILL_DENSE_ROWS` | No | Projection row grouping |
| `STRIX_PREFILL_PIPE` | No | Pipeline scheduling, row dispatch and embedding copies |
| `STRIX_QSA_ATTEND_BATCH` | No | Selection/attention on already rotated inputs |
| `STRIX_QSA_SCORE_PARK` | No | Scoring already rotated pooled keys |
| `STRIX_QSA_INDEX_ROWS` | No | Index projection batching |
| `STRIX_LM_HEAD_COHORT_LEAN` | No | Head projection; HTTP remains excluded |
| `STRIX_LM_HEAD_ROWS_LEAN` | No | Head projection |
| `STRIX_DISPATCH_FUSE` | No | Dispatch organization; the YaRN MTP route keeps it off |
| `STRIX_QPF2K_SCORE_ROWS` | No | Stage-batch scorer row tiling over already rotated pooled keys |
| `STRIX_PREFILL_PLE_BATCH` | No | PLE projection batching |
| `STRIX_QPF2K_SHARED_PRE` | No | Shared-expert projection |
| `STRIX_QPF2K_MOE_MC` | No | Expert merge/commit |
| `STRIX_QPF2K_MOE_HDR` | No | Expert dispatch headers |
| `STRIX_QPF2K_MOE_XOR64` | No | Expert LDS row layout |
| `STRIX_QPF2K_QSA` | Yes | B3 sites A-F below |
| `STRIX_QPF2K_QSA_MASK` | Yes | B3 sites A-F below, including grouped/batched alternatives |
| `STRIX_QSA_NORM_GROUP` | Yes | B3 sites B, C, E below |
| `STRIX_QPF2K_DEFAULTS`, `STRIX_PREFILL_F16` | No | Injection policy / dense operand profile |
| `STRIX_QWEN_ROPE_YARN` | Selects identity | Must agree with the resolved factor |

Other explicit controls are classified by one table in
`apps/qwen_yarn_admission.inc` (`yarn_environment_rules`). Classification
admits a setting; it never changes what YaRN executes by default. A refusal
lists every offending variable in one notice, each refused or out-of-range
row with its reason, for example
`YaRN execution environment not supported: STRIX_HCW_OUTPUT (changes HC output arithmetic), STRIX_UNKNOWN`.
An unlisted variable keeps refusing by name, whatever its value.

| Class | Environment item | Accepted | Reason |
|---|---|---|---|
| OFF switch | `STRIX_QSA_UNION_MERGE`, `STRIX_QSA_SCORE_ROWS64`, `STRIX_PREFILL_MOE_MERGE_VEC` | `0`, `1` | Code-default exact items; OFF restores the range union, 32-row scorer and scalar MoE merge over already rotated keys |
| OFF switch | `STRIX_QPF2K_WMMA_PIN`, `STRIX_QPF2K_HC_UPCOPY`, `STRIX_PREFILL_ROUTER_SELECT` | `0`, `1` | Dense register pins, HC up-weight copy and router selection; no rotary work |
| OFF switch | `STRIX_ROUTER_STAGE_ACT` | `0` | Unset is stage mode 2; `1` is another split and stays unclassified |
| OFF switch | `STRIX_QSA_WINDOW_ATTN_V2` | `0`, `1` | Verify-window score over already rotated queries and keys |
| OFF switch | `STRIX_QPF2K_GDN_CHUNK16` | `0`, `1` | GDN core chunk length (unset auto by call length); linear-attention layers only, no rotary work |
| OFF switch | `STRIX_PREFIX_CACHE` | `0` | Caching itself remains refused by the route guard |
| Diagnostic | `STRIX_PREFILL_TIMING`, `STRIX_QSA_FAST_STATS`, `STRIX_QPF2K_DENSE_TRACE`, `STRIX_QPF2K_HC_VERBOSE`, `STRIX_PREFLIGHT`, `STRIX_STARTUP_MARKERS`, `STRIX_RELEASE_EVENTS` | `0`, `1` | Host clocks, stderr lines and host counters only |
| Refused | `STRIX_HCW_OUTPUT`, `STRIX_HCW_NORM16`, `STRIX_QSA_FAST_SPLITS` | none | Change HC output/norm or QSA split arithmetic |
| Refused | `STRIX_HCW_NORMFREE`, `STRIX_QSA_FAST_TILE`, `STRIX_QPF2K_GDN_PARTS` | none | Alternative routes not qualified on YaRN |
| Refused | `STRIX_QPF2K_QSA_TRACE`, `STRIX_QPF2K_MOE_P1STATS`, `STRIX_QPF2K_GDN_CHECK`, `STRIX_GDN_PREFILL_CHECK`, `STRIX_QPF2K_HC_CHECK` | none | Device traces, counters and checks not audited for YaRN |
| Refused | `STRIX_PLE_WARM`, `STRIX_PLE_WINDOW_WARM`, `STRIX_MODEL_REGISTER_MLOCK` | none | Page residency not audited for YaRN admission |

The table also carries the previously admitted nonrotary controls (bundle
scheduling, operational policies, the chunked GDN prefill route, decode pool,
compact memory and the closed
single-request MTP/PLE controls) with unchanged values. `STRIX_PREFILL_TIMING=1`
reports host/enqueue spans, not device stage times. Use the OFF arms for A/B
measurement while holding the numerical profile pair fixed.

The B3-covered rotary sites are listed below. Paths with a table source use
`with_qwen_rotary`; grouped/wave consumers read the table rows it constructs.
File:line citations refer to the implementation inventory, not model qualification.

- A: `src/hip_indexed_attention_launcher.hip:5768` and `:5773`,
  `norm_rope_rows_kernel` (fallback direct Q/K/index-query normalization).
- B: `src/hip_indexed_attention_launcher.hip:5809`,
  `qsa_index::rope_table_kernel` (group coefficient construction).
- C: `src/hip_indexed_attention_launcher.hip:5820` and `:5824`,
  `qsa_flash::norm_rope_wave_kernel` / `qsa_index::norm_rope_group_kernel`
  (consume the constructed coefficients).
- D: `src/hip_indexed_attention_launcher.hip:5885`,
  `block_pool_kernel` (raw index-key normalization/rotation).
- E: `src/hip_qsa_prefill_batch.inc:193`, `:199`, `:203`, `:216`,
  batched coefficient construction, wave/group consumers and pooled index keys.
- F: `src/hip_indexed_attention_launcher.hip:5993` and `:6015`,
  `block_score_rows_kernel` (unpooled index-key normalization/rotation).

YaRN requires effective `STRIX_QPF2K_SCHED=12` before reclaim or estimation.
A defaulted schedule resolves to `12` on YaRN (the native default is `14`, which
adds bit 2). Other schedule values, including an explicit `14` and unset with
default injection disabled, refuse by name; native scheduling is unchanged. The
inventory covers merged primary CSR and the pruned tail owners for schedule 12.

`STRIX_STARTUP_MARKERS=1` emits stderr lines of the form
`strix-startup-marker phase=PHASE event=EVENT monotonic_ms=MS chunk=N`.
The monotonic origin is initialized before main; paired `begin`/`end` events
cover preflight reset, first HIP guard, rotary allocation/generation/upload,
plan compilation, registration, PLE hash/drop and credit verification.
Generation/upload are measured separately per bounded rotary chunk (`N` is
its first row); registration uses the file byte offset. `first_prefill_tile`
uses `submit`, and `first_token` uses `ready`, each once per process. Marker
output is serialized, contains no paths or tokens, and does not synchronize
HIP. Unset/`0` emits no marker bytes. Diagnostic timings are host call spans,
not device kernel durations or a claim of recovered startup time.

Pre-device PLE setup verifies authority, direct-reader probes and the page drop
without starting reader workers, prompt hints, a rolling pipeline or staging.
The preflight owner is released before HIP setup. Runtime performs a fresh
post-registration drop/credit verification before activating prompt work;
failed verification still earns no credit and follows the existing refusal or
zero-credit admission policy.

Execution order is environment classification, host admission, table upload,
then execution. `--print-effective-config` stops after host admission and emits
its complete owner/limit block, including a memory refusal when applicable;
inspection never uploads the table. Passing classification does not bypass
memory admission or establish model fidelity.

Admission compiles T=8192,4096,2048 in one numerical lineage. It reports target
KV/raw index, recurrent and convolution state, host state/shadow/alignment,
model backing, rotary table, shared and slab pool bounds, launcher scratch,
compiled workspace/tails, metadata, PLE staging and runtime payload. The additional
internal host reserve defaults to zero; `STRIX_QWEN_YARN_HOST_RESERVE=1` restores
the previous 8 GiB charge in both admission and runtime (only `0` and `1` are accepted).
The independent 4 GiB physical margin remains binding. Both reserve modes run
a host MemAvailable watchdog, sampling `/proc/meminfo` every 100 ms. Below 2 GiB
(strictly), or if the observation becomes unreadable, it latches a stop. The next
completed tile, calibration row or decode cycle boundary raises an error; no
further generated output is published. In-flight work drains during owner cleanup.
The process exits nonzero with `stop_reason: yarn-memory-watchdog` and a
`yarn_memory_watchdog` receipt containing the threshold, minimum, trip observation,
sample count and accumulated read time. Successful execution includes the same
receipt with `tripped: false`; inspection does not start a watchdog. Polling cannot
guarantee survival of an abrupt allocation or another process exhausting memory
between observations. The shared prefill pool is counted once across layers; window/slab routes
are excluded. Launcher and staging owners use conservative payload bounds. Required bytes must fit the observed total UMA
ceiling, incremental MemAvailable and GTT headroom; MemAvailable is never
increased by PLE credit. Only a successful page drop verified with mincore
earns credit. Inspection can report a provisional zero-credit refusal; it does
not run the pre-device verification callback. After registration,
credit and available headroom are checked again before state allocation.
A refusal lists owners, verified credit, limits and the smallest T tried.
There is no context shrink, KV8 substitution or native fallback.

The immutable descriptor preserves checkpoint original maximum 262144 and
distinguishes it from effective table extent 524288 or 1048576. Requested context
is a separate admission input. Both main and index heads use 64 rotary
coordinates (32 frequency pairs). Sections are 11/11/10, interleaved on the
frequency axis; text positions coincide on all three axes. Coordinate pairing
is `(i,i+32)`. Nonrotary tails and V are unchanged.

For pair `i`, the named `qwen-yarn-hf64-v1` mathematics are:

```text
c(beta) = 64*ln(262144/(2*pi*beta))/(2*ln(10000000))
low = max(0,floor(c(32))) = 14
high = min(63,ceil(c(1))) = 22
r = clamp((i-low)/(high-low),0,1)
u = 10000000**(-2*i/64)
w = u*(1-r) + (u/factor)*r
A = 1 + 0.1*ln(factor)
cs[p][i] = (RNE_f32(A*cos(double(p)*w)), RNE_f32(A*sin(double(p)*w)))
```

Truncation of correction bounds is enabled. The generator uses C++ double
`std::log`, `std::pow`, `std::cos`, `std::sin` from the host libm, without fast
math or contraction, and requires round-to-nearest-even. Transcendental
rounding ties can vary by libm; cross-platform bit identity is not claimed.
Retain the compiler/libm provenance alongside the artifact and its SHA256.
The Python oracle checks independent algebra with decimal-base powers and
binary32 conversion. Neither host implementation establishes model fidelity
or device qualification.

`generate_qwen_yarn_rows` delivers `[first,end)` in chunks of at most 4096
rows (default 256), with positions strictly below the effective maximum.
Each row has 32 interleaved binary32 cosine/sine pairs (256 bytes). A full
factor-2 table is 128 MiB; factor 4 is 256 MiB. The writer makes two bounded
passes, first hashing, then writing. It holds no full-table copy. Readers
validate without publishing unchecked rows. Native generation emits nothing,
including no header, and retains the existing in-kernel FP32 arithmetic.

The table is a standalone format; raw native table files are not accepted as
this format. All multibyte integers and floating point bit patterns below are
little-endian:

| Offset | Field |
|---|---|
| 0 | 8-byte magic `STYARN01` |
| 8 | u32 schema = 1 |
| 12 | u32 endian marker = 0x01020304 |
| 16 | u32 canonical descriptor byte length D |
| 20 | u32 first position |
| 24 | u32 row count |
| 28 | u64 payload byte count = row count * 256 |
| 36 | D descriptor bytes |
| 36+D | 32 raw SHA256 bytes of payload |
| 68+D | binary32 interleaved cosine/sine rows |

The canonical descriptor starts with NUL-terminated UTF-8 strings
`STRIX-ROTARY-V1`, `yarn`, `qwen-yarn-hf64-v1`,
`fp64-libm-to-f32-rne`. Next are u64 fields: factor, original maximum,
effective maximum, main rotary dimension, index rotary dimension, theta bits,
three section counts, interleaved boolean, beta-fast bits, beta-slow bits,
truncate boolean, attention-factor bits. Double fields use IEEE binary64 bits.
Bench reference receipts cross-check rotary payloads against effective argv,
requested arguments, environment and resolver reason. A YaRN signal requires
a complete non-native payload with an agreeing factor and table digest;
deleting the whole rotary object or replacing it with native is refused.
Legacy receipts may omit rotary only when no non-native signal is present.

Native identity serialization is an empty payload: appending it changes no
existing compiled execution, session layout or MTP identity bytes. Later
runtime integration must bind the table digest as well as this mathematical
descriptor to owners, caches and compatibility checks before execution.

Validation rejects metadata/endianness mismatches, incorrect ranges, nonfinite
coefficients, wrong hashes, truncation and trailing data. SHA256 is an integrity
check, not authentication or evidence of numerical qualification.

The HIP runtime has an internal test seam: `HipQwenRotaryTable` generates the
full table with bounded host staging, uploads each chunk once, and exposes a
read-only device pointer, extent and payload SHA256. Its owner must outlive all
stream and captured-graph readers. `ModelContract::qwen_rotary_table` carries
a shared owning handle; null selects the original native kernels. Host admission
binds the same coefficient digest before the device allocation.

An empty compile-time source pack retains the native kernel arguments and
FP32 expressions. Table instances read `float2 cs[position][32]`, including A.
Direct Q/K and index-query normalization, dynamic graph positions, pooled
raw-index keys, known-row kernels, grouped/batched prefill table construction,
wave/group consumers and MTP normalization/scoring share these coefficients.
Raw index keys remain pre-normalization and pre-rotation. Pool reuse refuses a
different table owner, and an MTP batch requires the same table handle.

Scalar positions refuse before enqueue; device-table instances additionally
trap on an invalid position before reading coefficients or writing their
outputs. Grouped table construction copies the selected full-table rows.
Dynamic graph capacity must fit the table. Cross-owner identities bind the resolved table payload; callers must not mutate a live
configuration or release its table while a graph retains it.

`strix-hip-qwen-yarn-rotary-tests` is a device fixture target, labeled `gpu` in
CTest. It exercises actual kernels against a coefficient/BF16 CPU reference
and frozen project-owned native implementations. Building it does not execute
it. Device exactness and model qualification require separate authorized runs.
The compiler inspection helper opens no device:

```sh
python3 scripts/qualification/qwen_rotary_isa.py \
  --object BUILD/CMakeFiles/strix_hip_runtime.dir/src/hip_indexed_attention_launcher.hip.o \
  --out PRIVATE_OUTPUT
```

Run against both revisions with identical toolchain/options. Keep the extracted
code objects, disassembly, metadata and per-symbol SHA256 manifest outside Git.
Compare native symbol bytes and VGPR/scratch counts; whole `.text` hashes change
when new table specializations are added. ISA equality does not replace device
token/logit/state qualification.
