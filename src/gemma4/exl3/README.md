# EXL3 target weights

`g4-exl3-v1` is an explicitly selected numerical identity.
Its signed input/output scales, normalized H128 transforms, FP16 boundaries,
rotated FP16 codebook coefficients and chunk16 FP32 dot order are declared in
`format.hpp`. Other target operations and authoritative FP32 KV retain the
`g4-v1` contract. It does not imply equality with another weight format.
`g4-exl3-wmma-pf-v1` additionally replaces core dots with increasing K16
WMMA instructions; it requires separate arithmetic and fidelity qualification.

The streaming assembler is `tools/gemma4/exl3_assemble.py`:

```sh
python tools/gemma4/exl3_assemble.py --source QUANT_DIR --output ARM_DIR --arm M4 --model-card BF16_DIR/README.md
```

Inputs are immutable per-K safetensors and indices; the output is a contiguous
`weights.bin` plus `manifest.json`. Memory consists of metadata and a 4 MiB
copy buffer. Existing output packs are refused. A manifest is published only
after the payload is complete. A partial file after interruption requires a
fresh output directory. Arm recipes include M4, M5, M6, U5 and two predefined
variants; `--class-k CLASS=K` explicitly creates a different class map.
The original model card is copied unchanged and the applicable model license
is retained alongside it, with both file hashes recorded in the manifest.

Byte counts include scale vectors and BF16 linear exemptions. The codebook
is algorithmic and stores zero table bytes. Resident BF16 gather tables are
reported separately from streamed linears; per-token gathers are included in
`weight_bytes_per_token`. KV/activation/norm traffic is not a weight-bandwidth
measurement. Packs and comparative evaluation receipts stay outside Git.

`load_arm(pack, bf16)` validates adopted tensor hashes, layout and scale
finiteness and binds norms, scalars, embeddings and PLE table from the original
BF16 safetensors. Embedding gathers keep their original BF16 boundary. The
quantized head has separate storage and is never used as a gather table.
The returned manifest records excluded/replaced source names; unexpected text
tensors outside the explicit shared-KV compatibility whitelist are refused.

Product decode selects this route explicitly with
`--model BF16_DIR --exl3-arm ARM_DIR --rope-tables TABLE_DIR`.
`STRIX_G4_SHARED_WEIGHTS=0` restores independent row core passes;
`STRIX_G4_GRAPHS=0` and `STRIX_G4_FUSION=0` retain their existing exact
fallbacks. `STRIX_G4_EXL3_OUTPUT_TILE=0` uses one output/wave instead of two,
retaining each dot's order. `STRIX_G4_PREFILL_PREPARE=0` decodes panels within the WMMA pass.
`STRIX_G4_EXL3_COLUMN_CACHE=0` restores per-coefficient ring loads; the default
caches the K or K+1 packed words needed by a column before its sixteen decodes.
Coefficient bits and row accumulation order are unchanged.
The default decode core is the lane-tile route: each lane keeps the declared
chunk16 ownership (chunks c = l mod 32, e ascending, one exact FP16 product per
add, unchanged butterfly) but reads an aligned window of its own tile with
16-byte loads and decodes each coefficient with one FP16 FMA, which equals the
declared FP32-then-FP16 codebook for all states. `STRIX_G4_EXL3_LANE_TILE=0`
restores the column-cache route above; `STRIX_G4_EXL3_LANE_GROUPS=4|8` forces
half or whole tile windows. Linears sharing one input (q/k/v, gate/up) share
their transform launches; `STRIX_G4_EXL3_GROUP=0` issues them separately.
For up to two rows the lane windows are staged through LDS (cooperative
coalesced loads; `STRIX_G4_EXL3_STAGED=0` loads them per lane), and one block
owns 128 outputs: it forms the transformed FP16 input rows in LDS and applies
the output H128 transform and signed scale after the unchanged butterfly
(`STRIX_G4_EXL3_FUSED_INPUT=0` and `STRIX_G4_EXL3_FUSED_OUTPUT=0` restore the
separate launches). Grouped members then run as one launch
(`STRIX_G4_EXL3_GROUP_LAUNCH=0` launches each member separately). All of these
reuse the same transform and core arithmetic, so bytes are unchanged.
`strix-gemma4-exl3-bench` (node only) checks every codebook state and route
equality at the family shapes, then times the routes with DRAM-cold weights.
No arm becomes an automatic default through this import path.

Prompt chunks of more than eight rows use `device_linear_rows`: the same
input transform, a many-row core, and the same output transform. A workgroup
decodes each lane's coefficients once and applies them to 16 (v1) or 32
(v2/v3) rows whose transformed FP16 inputs are staged in LDS; every output
keeps its identity's order (v1 lane chunks and butterfly, v2/v3 split ranges,
quad reduction and increasing split sum), so the bytes equal `device_linear`
over eight-row slices. Under v2/v3 the prefill FFN uses `device_ffn_rows`:
the gate and up cores, then one launch that forms both output transforms,
`gelu_tanh(gate) * up` and the down input transform in registers (the same
operations as the separate output launches, `gelu_product` and the input
transform), then the down core; `STRIX_G4_EXL3_FFN_FUSE=0` runs the separate
linears. `strix-gemma4-exl3-rows-tests` (ctest per identity)
compares them by memcmp; `STRIX_G4_PREFILL_EXACT_MANYROW=0` keeps the slices.

`g4-exl3-wmma-pf-v1` prompts run on the prefill GEMMs over FP16 panels decoded
once at load (`PreparedLinear`): the pack's input transform writes the GEMMs'
K16 input panels, the GEMMs run unchanged, then the pack's output transform.
Every output keeps one increasing K16 WMMA chain on the same FP16 operands, so
the bytes equal the pack's own 64x64 WMMA kernel (`STRIX_G4_EXL3_PREFILL_GEMM=0`
restores it). The FFN keeps raw gate/up GEMM values and forms the middle stage
above into down's input panels in one launch (`STRIX_G4_PREFILL_FUSE_FFN=0` runs
the separate linears). `strix-gemma4-exl3-panels-tests` compares both by memcmp.
Decode cores (rows <= 8) keep raw trellis words of several chunks in flight per
wave (eight for 4-bit tiles at one or two rows, four elsewhere) and align each
chunk's window only when it is consumed; chunks are consumed in the same order
with the same arithmetic, so the bytes are unchanged
(`STRIX_G4_EXL3_COOP_DEPTH=1|2|4|8` forces one depth; 8 falls back to the
largest dividing 4, 2 or 1). Tile words and activation pairs are loaded from a
wave-uniform scalar base plus a 32-bit lane offset (`STRIX_G4_EXL3_COOP_SADDR=0`
restores per-lane 64-bit addresses; the opt-in fused core keeps them), and the
codebook product `s * 0x83dcd12d` is formed (up to four rows) from two full-rate
16-bit multiplies with the byte sum by `v_sad_u8` instead of the quarter-rate
32-bit multiply and the byte dot product (`STRIX_G4_EXL3_HASH16=0`, also implied
by `STRIX_G4_EXL3_COOP_SADDR=0`; same product and sum for all 16-bit states, and
the operator bench compares all 65,536 codebook values). Input transforms and
the FFN middle stage issue all their activation and scale loads before the first
FP32 rounding boundary (`STRIX_G4_EXL3_INPUT_PRELOAD=0` restores the interleaved
order; same values and boundaries).

The engine selects this identity with `STRIX_G4_EXL3_PREFILL=wmma` (profile row
`exl3_prefill`, pending in QUALITY until its qualification receipt; the panels
add about 2 bytes per weight of device memory).

The separate `strix-gemma4-exl3-check` CLI runs full-model gates, bounded
speed trials (`--mode speed`) or teacher-forced prefill (`--mode quality` on
`g4-exl3-wmma-pf-v1`, `--mode quality-exact` on the engine's exact prefill).
It accepts `--model ARM_DIR --bf16 BF16_DIR --rope TABLE_DIR`, or a GGUF
file for same-corpus qualification. Quality uses `--teacher TEACHER_DIR`,
with resumable `--skip N --limit N` parent ranges. It validates ordinary
teacher hashes, uses all rows for top-1 and next-token NLL, and full-vocabulary
FP64 KL only where full reference rows exist. FP16 teacher logits are
self-normalized. `tools/gemma4/exl3_quality.py` rejects incomplete or mixed
identity receipts and retains every language gate alongside pooled metrics.
Record kernel and IOMMU PerfOpt state with timing receipts; compare complete
tokens/state first. Full-model commands are explicit CLIs, never host ctest.

`--mode speed-all --model ARM_ROOT --bf16 BF16_DIR --q4 MODEL.gguf --rope TABLE_DIR`
retains five packed targets on the native node and runs three interleaved
rounds (reverse order on round two). Prefill borrows the target's device views;
only one arm's FP16 panels is live at a time. Initialization/panel setup is
reported outside timed runs. Decode samples include forward execution, KV
commit and greedy next-token selection. This is a node-only campaign, never a PC check.
Qualification CLIs require canonical attention so their prefill identity cannot
silently include an unrelated arithmetic sub-profile.

The codec/transform semantics are adapted from strix-trellis, revision
`5acf0905a3586e719081abf735152298c320827c` (Apache-2.0). The adaptation adds
the declared chunk16 core reduction, family views, strict failure handling and
family integration. Original attribution and license are in `notices/`.
No source code from the format-defining project is incorporated here.
Synthetic generated fixtures contain no checkpoint data and require neither
Python nor a model at C++ test runtime. All release and full-workload gates
remain required; synthetic byte checks alone do not qualify an arm/default.
