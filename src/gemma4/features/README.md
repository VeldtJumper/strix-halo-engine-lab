# Disposable branch features

`features::Exporter` returns the existing `g4-v1` target's post-final-norm
FP32 row (2560 values for the production model). It does not pool, rescale or
normalize the row a second time. Optional code logits are the target's capped
logits at the read position, in caller token-ID order, including duplicates.
There is no probability calibration or decision head in this API.

The prefix is fully processed and committed. Each question is a separate,
unpublished `KvTxn`: reserve the whole branch, forward chunks of at most eight
rows, copy requested results, discard. All producer rows remain in the overlay
across chunks; sliding consumers in layers 24..41 resolve producer 22 and global
consumers resolve producer 23 through that same transaction. Branch tokens never
enter canonical history. Branches in a request group run sequentially. A group
is therefore equal to separate branch calls, without a claim of shared-pass
batch acceleration. `chunk_rows: 1` provides the serial fallback. With the
default `chunk_rows: 8` and a backend that offers exact many-row prefill, the
prefix is prefilled in 512-position chunks and each branch of 9..512 tokens in
one many-row pass (hidden rows from the same final norm, code logits from the
same head); the bytes equal the eight-row passes.

Receipts contain complete frozen identities, absolute and branch-relative row
positions, processed row counts, zero committed branch rows, and canonical
logical state digests before and after the group. A changed digest is an error.
The route hash covers tensor names, original formats, shapes and byte strides;
the weights and RoPE hashes use the loader's existing content identities.
The tokenizer hash covers the supplied tokenizer asset (empty for raw IDs).
The template hash covers the LF-normalized native chat renderer and adapter source/header
hashes in that order. These are implementation source identities, not hashes
of the official template asset. Only `prefill-exact` / FP32 cache is currently
admitted. A different prefill identity is refused.

## C++

Link `strix_gemma4_engine`, which exports the feature module together with its
backend and family adapter dependencies.

Construct `Exporter(backend, freeze(manifest, tokenizer_hash))` with model and
RoPE identities already frozen. `export_features(request)` either ingests
`request.prefix` once or reuses `request.stream_handle`. A retained handle
belongs to that exporter and process, and cannot change identity. The default
cache limit is eight prefixes; a full cache refuses admission. `release(handle)`
frees it. `release_prefix: true` on a handle request releases it after exporting.
Retained device prefixes reserve the model context capacity for later branches;
transient prefixes allocate only their admitted prefix plus longest branch.

`export_features(request, committed_stream)` borrows an existing stream; its
explicit stream argument is authoritative and prefix/handle/cache flags in the
request are ignored. The caller guarantees matching identity and exclusive
backend/stream ownership. It can continue ordinary decode on this stream after
the call. Model/backend lifetimes follow the existing engine contract.

`RowPlan::export_hidden` defaults to false. `ForwardResult::hidden` stays empty
when no active row requests it; otherwise it uses output-row indexing, with
unrequested/padding entries zero. This adds no device arithmetic, allocation or
transfer to ordinary decode. `KvTxn::reserve_rows(n)` is an optional additive
multi-pass contract, called once before the first forward. Unreserved
transactions retain their single-pass validation. Failed transactions cannot
append or commit; destroying them releases the stream lease.

## JSON, CLI and HTTP

Inspect the exact identity before creating a request:

```sh
strix-gemma4-features --model MODEL.gguf --rope-tables TABLES --tokenizer tokenizer.json --backend device --print-effective-config
```

Each JSONL input line (or HTTP body) has this shape. Copy the entire identity
object from inspection, including all hashes and identity fields:

```json
{
  "identity": {
    "weights_hash": "WEIGHTS_HASH", "route_hash": "ROUTE_HASH",
    "tokenizer_hash": "TOKENIZER_HASH", "template_hash": "TEMPLATE_HASH",
    "rope_hash": "ROPE_HASH", "identity": "g4-v1",
    "prefill_identity": "prefill-exact", "cache_dtype": "F32"
  },
  "prefix_ids": [2, 100, 101],
  "branches": [
    {"id":"q1","token_ids":[102,103,104,105],"read_position":3,"option_close_positions":[1,2]}
  ],
  "code_ids": [106, 107],
  "chunk_rows": 8,
  "retain_prefix": false
}
```

Positions are zero-based token indices **inside the branch**, including the
instruction/options/delimiter tokens supplied by the caller. They must refer
to processed input tokens; there is no implicit sampling or extra read token.
Option positions preserve caller order and may repeat. Read may occur before
the last branch row; later causal rows do not change its result. Empty prefixes
are supported, empty branches are refused. Input IDs must be in vocabulary and
prefix plus branch must fit context. Unknown fields are refused.

Use exactly one prefix source: `prefix_ids`, `messages` with optional
`chat_template_kwargs`/`tools` rendered by the family adapter, or `stream_handle`
returned by an earlier `retain_prefix: true` group. Messages require tokenizer
assets. Branches always contain explicit token IDs.

```sh
strix-gemma4-features --model MODEL.gguf --rope-tables TABLES --backend device --input requests.jsonl --receipts receipts.jsonl --npz-dir features
strix-gemma4-features --model MODEL.gguf --rope-tables TABLES --backend device --input requests.jsonl --receipts receipts.jsonl --npz-dir features --resume
```

Without `--npz-dir`, JSONL includes FP32 values represented as round-trippable
JSON numbers. With it, each record names `features-N.npz`; JSONL retains row
positions, identities and isolation evidence. Arrays are little-endian FP32:
`branch_B_read` `[1,H]`, `branch_B_options` `[option_count,H]`, and
`branch_B_code_logits` `[1,code_count]`. B follows request branch order.
Empty option/code arrays keep their declared dimensions. Arrays are written
through a temporary file before the journal record is flushed.

Resume validates exact input-line hashes and frozen identity, skips completed
records, and repairs a torn trailing journal line. A completed archive lacking
its journal row is regenerated. Resume requires independent prefix requests;
process-local retained handles cannot survive a restart. Ordinary runs can use
handles across JSONL lines. Completed output archives are otherwise not replaced.

`POST /v1/gemma4/features` on `strix-gemma4-serve` accepts the same request and
returns features and receipt inline. `GET /v1/capabilities` exposes its frozen
identity as `feature_exporter_identity`: the route of this endpoint, which always
runs prefill-exact rows. The served chat prompt route is the top-level `prefill`
(and `dflash.prefill_route`), which may differ. Earlier builds named this
capabilities entry `features`; clients reading that key must read
`feature_exporter_identity` instead. HTTP feature calls serialize backend execution with ordinary
chat generation. Handles are process-local opaque references, not credentials;
the service's deployment access control applies to this endpoint too.

## Qualification

The lightweight host/device feature gate uses a synthetic 42-layer model with
real producer aliases 22/23 and small geometry. It checks byte equality against
ordinary forward, serial chunks, grouped versus separate branches, repeated
calls, unchanged continuation after extraction, failure cleanup, handles,
strict JSON and NPZ round trips. Device execution is explicitly requested with
`strix-gemma4-features-tests --device`; it belongs inside a reserved GPU turn.
An optional tokenizer-asset test exercises the HTTP handler and chat rendering.
Full-model throughput is measured separately through the CLI, never ctest.
These gates do not establish model fidelity or qualify another numerical profile.
