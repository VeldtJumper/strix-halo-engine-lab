# MiniMax H3 family (opt-in)

Audio+video diffusion transformer: 50 dense blocks (hidden 5376, 56 heads x 128, SwiGLU 14336),
full non-causal 3D attention over the packed sequence `[text | cond | audio | video]`.

Build with `-DSTRIX_FAMILY_H3=ON`. Nothing here is built by default.

Each package lives in its own directory and adds one `cmake/h3_<NN>_<pkg>.cmake` (included in name order).
`cmake/h3_00_common.cmake` defines the shared interface target `strix_h3_common`.

| dir | package |
|---|---|
| `gemm/` | W8A8 tensorwise + convrot GEMM and the activation quantiser |
| `attn/` | dense non-causal flash attention, head dim 128, split-KV partials and canonical merge |
| `block/` | block glue, weight loader, single-node forward, sampler (later change) |
| `dist/` | two-node sequence-parallel K/V exchange schedule (later change) |
| `te/` | text encoder |
| `vae/` | video and audio decoders |
| `levers/` | opt-in fast-class levers (later change) |

Exactness classes: the dense path is the reference and must be bit-identical between one and two
nodes. Every quality lever is fast-class: its own identity, an off switch, and a quality gate
against the dense reference.
