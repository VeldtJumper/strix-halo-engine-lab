# H3 reference fixtures: format h3fx v1

Writer: `tools/h3/fixtures/h3fx/fxio.py` (`FixtureWriter`); the capture scripts that run the reference model
are not part of this repository. Reader: header-only C++
`src/h3/common/fixture.hpp` (`strix::h3::fixture::Manifest::load(dir)`, `.read(name)`, sha256
verified on read; test `strix-h3-fixture-reader-tests [set_dir ...]`) and python `h3fx.fxio.read_set`.

## Layout

```
<H3FX_ROOT>/
  FORMAT.md                this file
  QUANT-SPEC.md            activation quantiser chain (int8 tensorwise + convrot 256), verified bitwise
  inventory/<ckpt>.json    checkpoint inventories (name, dtype, shape, absolute byte offsets, quant configs)
  <set>/manifest.json      one fixture set
  <set>/<name>.bin         one tensor per file; <name> may contain '/' (sub-directories)
  te/ vae/ ...             sets written by other packages with their own formats (not h3fx)
```

## manifest.json

```json
{"format": "h3fx", "version": 1, "set": "<set>", "meta": {...free-form...},
 "tensors": [{"name": "f0/blocks/out_00", "file": "f0/blocks/out_00.bin", "dtype": "bf16",
              "shape": [400, 5376], "nbytes": 4300800, "sha256": "<hex of the .bin bytes>", "desc": "..."}]}
```

- `.bin` = raw little-endian, C order (last dim fastest), no header, `nbytes = prod(shape) * size(dtype)`.
- dtypes: `bf16` (raw u16 bits), `f16`, `f32`, `f64`, `i8`, `u8`, `i16`, `i32`, `i64`, `bool` (u8 0/1).
- `meta` carries the capture config, seeds, RNG recipe, layout segments, mod segments, timings.
- Self-consistency: `python -m h3fx.compare SET_A SET_B` (same names + sha256) prints a digest per set.

## Sets

| set | content |
|---|---|
| `quant` | quantiser reference: K=5376 (block 0 qkv weight rows 0..63) and K=14336 (block 0 fc2 rows 0..63), families normal / heavy / edge / real (captured block 0 activations): x, x_rot, row_absmax, row_scale, row_scale_bf16, q, w_q, w_scale, acc (i32), out (bf16) + fp32-staged variant |
| `dit_t2va_small` | 256x256, 5 frames: video latent [1,24,2,16,16] (128 rows), audio [1,32,2,8] (16 rows), 256 synthetic text rows: S = 400 |
| `dit_fl2va_small` | same + 1 first-frame keyframe (64 cond rows), text rows 12..77 carry tag 0 (vision span): S = 464 |
| `dit_t2va_mid` | 512x512, 17 frames (22 frames after snapping: latent [1,24,7,32,32] -> 1792 video rows, audio 37 x 2 = 74 rows, S = 2122): first forward stopped after block 0 (block 0 detail + out_00 + layout) |

### DiT set names (prefix `fN/` = N-th forward call of the sampling run, f0 = step 0, f1 = step 1)

| name | dtype / shape | meaning |
|---|---|---|
| `in/text_states` | bf16 [1,L,5120] | synthetic Qwen-like layer-50 states (recipe in meta.text_synthetic) |
| `in/text_token_tags` | i64 [L] | 1 = text, 0 = vision (adaLN modality tag) |
| `in/keyframe0_latent` | f32 [1,24,1,h,w] | fl2va only, synthetic normalized keyframe latent |
| `text/condition_proj_out`, `text/refiner_block{0,1}_out`, `text/refined` | bf16 [L,5376] | text path (runs once per sampling run) |
| `noise/video`, `noise/audio` | f32 | initial noise (RNG recipe in meta.rng) |
| `noise/cond_video_rows0` | f32 [n_cond,96] | fl2va: the cond-row augmentation noise |
| `sampler/sigmas_simple20`, `sampler/sigmas_simple8` | f32 | simple scheduler, shift 12 (video sigma) |
| `sampler/step{i}_x_{video,audio}` | f32 | sampler state entering step i (audio is CARRIED: x_audio * sigma_v/sigma_a scale, see model.forward) |
| `sampler/step{i}_denoised_{video,audio}` | f32 | x0 prediction of step i |
| `sampler/final_{video,audio}` | f32 | after the run (step 0 = Euler, step 1 = res_multistep 2nd order) |
| `layout/position_ids` | f64 [S,3] | (t,h,w) per packed row |
| `layout/{img,audio}_pos`, `layout/{img,audio}_update` | i64 / bool | packed-row indices of video-like / audio-like rows and which are denoised targets |
| `layout/rope_angles` | f32 [S,96] | `rope_freqs()`: [t16 h16 w16 | same] angles |
| `fN/rope/table` | bf16 [1,S,1,48,2,2] | rotation matrices [[c,-s],[s,c]] cast to the compute dtype (as consumed) |
| `fN/adaln/t_emb` | f32 [n_t,8] (f0 only) | interpolated curve rows for the sorted unique timesteps |
| `fN/b0/adaln_out{0..5}` | f32 [3*n_t,5376] | block 0 shift/scale/gate msa, shift/scale/gate mlp; row = t_index*3 + tag (tag 0 video, 1 text, 2 audio) |
| `fN/model_in/*`, `fN/model_out/*` | | DiT forward() inputs/outputs as the sampler sees them (timestep = sigma*1000, audio carried) |
| `fN/stream_in/*`, `fN/stream_out/*` | | `_forward` inputs/outputs (audio un-carried; outputs are -velocity) |
| `fN/blocks/in_00`, `fN/blocks/out_{00..49}` | bf16 [S,5376] | residual stream before block 0 and after every block |
| `fN/b0/norm1_in/out`, `attn_in/out`, `qkv_proj_in/out`, `out_proj_in/out`, `norm2_in/out`, `mlp_in/out`, `fc1_out` | bf16 | block 0 sub-ops (f0 only). `attn_in` = modulated norm1 output; `qkv_proj_out` = raw qkv before q/k norm |
| `fN/attn/b{00,01,24,49}_{q,k,v}` | bf16 [1,56,S,128] | q,k after per-head RMSNorm + partial split-half RoPE; v raw (f0 only) |
| `fN/attn/bXX_out` | bf16 [1,S,7168] | attention output before out_proj (CPU default backend: chunked sub-quadratic exact-softmax attention, bf16) |
| `fN/final/h_in`, `fN/final/{video,audio}_rows`, `fN/final/adaln_{shift,scale}` | | final layer input and per-row head outputs (fp32) |

Row order of the packed sequence (both small sets): `[text | (cond) | audio (ch0 t0..T-1, ch1 t0..T-1) | video (t,h,w)]`;
the exact segment table is `meta.layout.segments` ([start, stop, kind]) and the adaLN segment rows
`meta.mod_segments[fwd]` ([start, stop, mod_row]).

## Important caveats

- Compute = CPU, bf16 compute dtype forced (as on the reference GPU); attention = the CPU default
  chunked sub-quadratic backend (exact softmax, not the GPU flash kernel), int8 GEMMs = the portable library code path (the fused GPU quantiser is binary-only, see
  QUANT-SPEC.md). Integer accumulators are exact (computed through fp64, bit-identical to int32).
- Text states are SYNTHETIC (no CPU text encoder for the nvfp4 checkpoint); keyframe latent is synthetic.
- Every set was produced twice and the two runs compare equal (sha256 of every tensor).
