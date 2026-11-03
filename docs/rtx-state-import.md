# Externally produced prompt state: `--rtx-state-import`

Opt-in. With the flag off nothing changes: no route, no registry, no extra receipt field, same request path.

A prefill service elsewhere (the RTX PRO 6000 CUDA port, or another Strix node for testing) computes the state of
a prompt prefix `prompt[0, p)` and streams it to a Qwen `--http-serve` node. A later request whose prompt starts
with those `p` tokens and has at least one more token starts from that state. The node then runs the rest of the
prompt, at least the last row, and decodes.

## The state is a prefix-cache checkpoint at any position p

`apps/hip_prefix_state_cache.hpp` registers the components (`HipPrefixStateCacheSession`). An imported state is
exactly what a checkpoint holds, at any position `1 <= p <= prompt - 1`. It is not limited to block boundaries,
because under the single A8 profile a tiled block equals the same rows run serially, bit for bit.

- **Row components**, in this order, each with rows `[0, p - shift)`:
  - the 12 attention KV slices `kv.L<layer>`: 2048 B/row, shift 0;
  - the 12 raw index slices `index.L<layer>`: 256 B/row, shift 0;
  - on the MTP route, `mtp.kv` (2048 B/row) and `mtp.index` (256 B/row), both with shift 1.
- **Fixed components**, each at a 256-aligned offset of the fixed blob:
  - `recurrent_matrix`, `gdn_convolution` and `ple_convolution`;
  - `ngram_history` and `position_history` (host words);
  - on MTP, `mtp.held`: the wide row h_{p-1}, 20480 B.
- **Canonical payload**: every row component's rows in component order, then the fixed blob. The padding is
  zero and is never sent.

## Restore (request thread, `begin_request`)

The restore runs before the checkpoint lookup. It applies when a parked import's tokens are a prefix of the prompt
and `p <= prompt - 1`. Steps:

1. Asynchronous host-to-device copies per component, one stream sync. A failure poisons the session.
2. `next_position = committed_tokens = p`.
3. Set the state generations. For each flag f, `generations[f] = fresh[f] + p * (number of steps of the compiled
   serial plan whose writes_state contains f)`, and `valid = fresh | union of the steps' writes`. Every prompt row
   runs each step once, whether tiled or serial; a headless block skips only `lm_head`, which writes no state.
   - The export file records the producing session's real counters. `tools/rtx_state_import.py check` compares
     them with this formula.
   - `ckpt` compares a T-boundary checkpoint's metadata with it.
4. Build the lineage with `prepare_prefill_chunk_lineage`: one `prefill_chunk` interval `[0, p)` that carries the
   importer's nonzero producer ids (numerical, reduction, state_update).
5. Run the MTP restore hook (`restore_cache_frontier(p - 1)`).

Other rules:

- An import is consumed once unless it was begun with `"keep": true`.
- After an import, no prefix-cache checkpoint is captured for that request. State derived from foreign bytes must
  not enter the store under the HIP identity.
- Without `--prefix-cache-*`, the flag creates the same component registry over **no store** (a null store).
  Capture, lookup and the disk tier are then impossible by construction. That is simpler and safer than a second
  "capture off" switch: `STRIX_PREFIX_CACHE_LOOKUP=0` disables only the lookup.
- `--rtx-state-import` requires `--http-serve` without `--http-batch`, because batch slots get no registry
  in v1.

## CLI file pair: `--rtx-state-import-file NAME.json`

A fresh single-request CLI can consume the same prefix without an HTTP server:

```sh
python tools/rtx_state_import.py file-pair --state prefix.rtxs --file-pair prefix.json
strix-hip-decode-token MODEL.gguf --prompt-ids full-prompt.ids --context 524288 --rope-yarn 2 --rtx-state-import-file prefix.json --generate 256
```

`--prompt-ids` is the full prompt. Its first `position` tokens must equal the
imported tokens, and it must contain at least one suffix token. The node prefills
that suffix and decodes the requested generation. Native rotary is also admitted
with matching assets and capacity. Use the existing YaRN MTP bundle or explicit
`--mtp` asset options to select MTP; the existing admission notice and serial
fallback apply when the bundle cannot admit MTP. The pair must match the selected
consumer component layout, including predictor components when MTP is selected.

`NAME.json` is exactly an HTTP begin object plus `payload_bytes` and
`payload_sha256` (64 hexadecimal characters). It carries `tokens`, `position`,
`producer` (`numerical`, `reduction`, `state_update`), `layout_identity`,
`geometry_identity`, `encoding`, and any producer `context_tokens` and `rotary`
identity. Identities use the same string representation as HTTP. The writer
shares the HTTP begin-body builder and preserves producer capacity and rotary.

`NAME.bin`, beside the JSON, is the unencoded canonical payload: each row
component in layout order, then the fixed blob, including zero alignment padding.
`payload_bytes` and `payload_sha256` cover this entire binary file, including
padding. The begin `encoding` retains the HTTP negotiation semantics; file bytes
are always canonical bytes, with no chunk frames or segment headers.

The reader streams through a 4 MiB ordinary-memory buffer into the registry's
staging allocation, without a second payload-sized pinned allocation. It uses
the HTTP begin parser, `RtxStateStaging` admission, segment coverage and commit
checks; capacity and rotary mismatches are refused by those same checks. A bad
hash, length or token prefix fails the command without a cold-prompt fallback.
The ordinary HTTP restore then sets counters and lineage and invokes the MTP
frontier hook. Imported state never enters the local prefix cache.

HTTP-only options, session/replay/cache routes and graph execution are refused.
These CPU admission checks do not establish GPU continuation fidelity or qualify
an externally produced numerical identity.

Measured between two Strix Halo nodes (same build, YaRN factor 2, MTP route on both sides): a
state exported at position 253,952 of a 262,144-token prompt (7.7 GB) was imported by a fresh CLI, which
prefilled the remaining 8,192 tokens and decoded 256: generated tokens, the full committed-logit
ledger and the final state equal the full-prompt run, in 135.5 s against 333.5 s for the full prompt.
The same check at 32K is exact too.

## HTTP (connection threads; host memory only, so imports stream while another request decodes)

The connection is HTTP/1.1 keep-alive, with Content-Length bodies of up to 256 MiB when the flag is on. Every u64
identity in JSON is a decimal or `0x` hex **string**.

| route | body / reply |
|---|---|
| `GET /admin/rtx-import/layout` | `format` = `strix-qwen-rtx-state-v1`; `rows` [{id (index), name, bytes_per_row, shift}]; `fixed` [{id, name, bytes, offset}]; `row_bytes`, `fixed_bytes`, `layout_identity`, `geometry_identity`, `generations_per_row`, `generations_base`, `valid_union`, `valid_base`, `max_context`, `context_tokens`, `rotary`, `model_content`, `identity_extension` (MTP), `cache_identity`, `sha256_provider` |
| `POST /admin/rtx-import/begin` | {tokens, position, producer {numerical, reduction, state_update}, layout_identity, geometry_identity?, rotary?, context_tokens?, keep?} -> {id, payload_bytes, ...}. Allocates pinned host memory. At most 2 imports in flight. |
| `POST /admin/rtx-import/chunks?id=` | a sequence of segments, each a 16-byte header {u32 kind (0 rows, 1 fixed), u32 component id, u32 first_row \| offset_lo, u32 rows \| offset_hi}, then a u64 byte count, then the bytes (LE). Each chunk is all or none: every segment is bounds-checked and must not overlap written bytes or another segment. |
| `POST /admin/rtx-import/chunk?id=&component=&first_row=` / `?id=&fixed=&offset=` | One component, given by name or index. |
| every chunk | `X-Chunk-Sha256` (the body's SHA-256), verified on receipt. Optional `X-Chunk-Seq` gives the order; without it, the receive order. |
| `POST /admin/rtx-import/commit?id=` | {chunks, bytes (segment data bytes), manifest_sha256 = SHA-256 of the 32-byte chunk digests in sequence order, canonical_sha256?}. Checks every byte was written exactly once, the counts and the manifest, then parks the import. `canonical_sha256: true` also hashes the canonical payload (a diagnostic, off the critical path). |
| `GET /admin/rtx-import/status`, `POST /admin/rtx-import/abort?id=` | |

The layout's `context_tokens` (also exposed as `max_context`) is the consumer route's
capacity; import positions must satisfy `1 <= position < context_tokens`. `rotary`
uses the model's resolved rotary descriptor JSON: `rope_mode`, `factor`,
`original_max` and the existing `table_sha256` identify the table. Native mode has
no table. A supplied begin `rotary` or `context_tokens` must match the consumer.
YaRN geometry additionally binds capacity and the resolved rotary identity; begin
requires either that geometry identity or both explicit rotary and capacity.
Native geometry retains its original byte stream for existing producers. State
files record rotary and capacity, and the upload tool forwards those producer facts.
These are admission and layout contracts, not native execution or fidelity qualification.

## Chunk encoding (opt-in): `bp-v1`

The 1 GbE link is the hand-off bottleneck (~117 MB/s), so the chunks may travel compressed. It is lossless and exact
by construction: the node reproduces the plain chunk body byte for byte, then runs every check above on it.

- **Negotiation.** `layout` lists `codecs: ["bp-v1"]`. `begin` may carry `"encoding": "bp-v1"` (or `"identity"`);
  an unknown codec is a 400. The reply echoes `encoding`. A chunk POST with `X-Chunk-Encoding: bp-v1` needs an
  import begun with that encoding (400 otherwise). Chunks without the header are plain, as before, even inside an
  encoded import. Without `encoding` and the header, nothing differs from the plain protocol.
- **What is encoded.** The whole wire body of one `chunks` (or `chunk`) POST: the encoded body is a bp-v1 frame
  whose decoded bytes are exactly the plain body. The segment headers are unchanged and describe the decoded bytes.
- **Integrity.** `X-Chunk-Sha256` is the SHA-256 of the **wire** body (the frame). The manifest is built from those
  digests in sequence order. Commit `bytes` stays the segment data bytes (decoded). Each frame block also carries
  the XXH64 of its plain bytes, which the decoder checks. A codec defect therefore fails the chunk (400) and can
  never hand over different bytes. The `canonical_sha256` diagnostic still hashes the reconstructed payload.
- **Frame** (`include/strix/bp_codec.hpp`):
  - The frame is `"BPV1"`, `u32 block_count`, `u64 plain_bytes`, then per block `{u32 plain, u32 encoded,
    u64 xxh64}`, then the blocks.
  - A block holds the bytes of one region, at most 1 MiB, with a width w of 1 (segment headers), 2 (BF16) or 4 (FP32).
  - The block is split into w byte planes, and each plane is stored one of three ways: raw; as a constant byte; or
    canonical Huffman with lengths of at most 11 bits, 4 interleaved LSB-first streams and a 128-byte table of
    length nibbles. A plane is coded only when that saves at least 1.5 %. The trailing `plain % w` bytes are raw.
  - The width only affects the ratio. Any width decodes exactly.
- **Node.** Decoding runs on a shared worker pool (`--rtx-state-import-decode-threads N`, default 4; the connection
  thread decodes too) into a per-connection buffer, then the usual segment parse, bounds and coverage checks run.
  Decoded chunks are limited to 256 MiB. The chunk reply adds `wire_bytes`, `decoded_bytes` and `decode_ms`; the
  commit reply adds `wire_bytes` and `encoding`.
- **Producer** (`src/cuda/qwen_handoff.cpp`, service flags):
  - `--handoff-codec bp-v1|identity` (default identity) chooses the encoding. A request can override it with
    `"handoff_codec"`. The producer falls back to plain chunks when the node does not list the codec.
  - `--handoff-encode-threads N` (4) sets the encoder workers.
  - `--handoff-connections N` (1) sets how many chunk POSTs are in flight, each on its own keep-alive connection.
    The node orders the digests by `X-Chunk-Seq`.
  - `--handoff-chunk-mib N` (32) sets the chunk size.
  - A packer thread builds, encodes and hashes the next chunk while the posters send the previous ones. Pieces
    larger than a chunk (the 113 MB recurrent matrices) are split, so their chunks pipeline.
  - The element width per component is fixed: `recurrent_matrix` and the host words are FP32 (4); the rest are
    BF16 (2).
- **Measured** on a real 1031-row state (`tests/bp_codec_tests.cpp --sample`): KV 0.681, index 0.674, MTP rows
  0.681, conv histories 0.689, `recurrent_matrix` 0.854; the whole payload is 0.814. At 32K the payload is about
  89 % KV, so about 0.70.
  - Throughput: one thread encodes about 1.0 GB/s and decodes 1.5 GB/s on the PC (i7-13700K); the node's Ryzen AI
    Max+ 395 does 1.3-1.45 and 2.0-2.9 GB/s. With 8 threads the PC reaches 3.0 / 7.4 GB/s and the node 4.2 / 11.3.
  - Over the real 1 GbE link, CPU only (`tests/cuda/qwen_handoff_codec_test.cpp --send` to
    `strix-rtx-state-import-tests --serve-file`), the 147 MB sample took 1.34-1.38 s plain and 1.04 s with
    bp-v1 + 2 connections. The wire stays at about 115 MB/s and the node's payload is exact.

When an import was used, the request's `strix` object carries `rtx_import: {id, position, bytes, chunks,
restore_ms, verify_ms, sha_ok, manifest_sha256}`. `cached_tokens` counts the imported positions.

## Producer for testing: `STRIX_RTX_EXPORT_STATE=<file>` (+ `STRIX_RTX_EXPORT_POSITION=<p>`)

This works on the tiled MTP route, as a CLI run or a server. The env alone creates the registry; no flag is
needed. After a prompt unit (block, serial row or prompt window) ends at the selected position p, the engine writes the
file. The default p is the end of every prompt. On batched HTTP, accepted GDN window rows are materialized in
live recurrent/conv state before this copy, including an explicitly selected intermediate window boundary.
Prompt receipts and HTTP prompt digests likewise materialize before hashing when diagnostics are requested.
Without an export at that boundary or a requested digest, the window stays deferred until its next consumer.
The standalone commit uses the same ordered recurrence and BF16 history shifts as the next window and clears
the shared pending rows, so continuation cannot apply them twice.

The opt-in terminal-R4 batch-export -> single-import gate is `tests/http_slots_gpu_gate.py --export-roundtrip`;
its docstring gives the exact command and required model/prompt setup. It compares saved component bytes,
including recurrent/conv and predictor state, and then continuation IDs and every committed target component.
Missing R4 or verified-import witnesses fail qualification.

The file is `STRXRTX1`, a u64 LE header length, a header JSON, then the canonical payload. The header JSON holds the
tokens, the position, the producer, the layout, the sha256 and the session counters.

`tools/rtx_state_import.py upload` sends the file and then the completion. The AMD -> AMD exactness test starts a server with `--rtx-state-import` and `STRIX_RTX_EXPORT_STATE` /
`STRIX_RTX_EXPORT_POSITION` at p = N-1 to export the state of the prompt prefix, uploads it into a fresh server
(`tools/rtx_state_import.py upload`) and requests the full prompt: the token ids must equal the plain full-prompt
request on both servers and on a second import.
