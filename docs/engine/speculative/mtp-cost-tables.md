# Persisted MTP window-cost tables

The Qwen MTP min-p window controller chooses, every cycle, between serial decode and the R2 / R4 / R8 verify
windows by expected tokens per measured millisecond. A new request used to start with no cost estimates: each window
size was first tried, then timed. Window-cost tables keep the measured costs between processes and give the next
request a starting point.

## What is stored

One small text file per key. Entries are named `r1` (serial, ms per emitted token) and `r2`, `r4`, `r8` (ms per
verify cycle); each holds the median of at least three measured cycles and its sample count. The file repeats every key
field and carries a format version, a generation, an epoch and a checksum.

Key fields (a table is used only when all of them match):

| Field | Content |
|---|---|
| engine | `qwen-mtp-minp` |
| binary | SHA-256 of the running executable |
| device | HIP device name, architecture, compute units, memory, clock, HIP runtime and driver versions |
| model | target model content identity |
| policy | requested route (tiled, HTTP, batched, `--http-batch` value or `auto`, cohort size), kernel plan, execution and numerical profile identities, optimization profile, the effective CLI without prompts and outputs, and the `STRIX_` / `HIP_` / `HSA_` / `AMD_` / `ROCR_` / `GPU_` environment (except `STRIX_COST_TABLE*` and `STRIX_CYCLE_LOG`) |
| bucket | context bucket: 0 below 8192 positions, then one per doubling |

The automatic HTTP slot capacity is not keyed: it follows free memory at startup and differs between runs of the
same command.

## How the controller uses it

- The prior is bootstrap-only. It is kept apart from the measured samples and adds no sample weight: a window size uses
  its table cost only until it has a timed sample of its own in this request, and every remaining prior expires after
  4 cycles. Warm-up, acceptance calibration and re-probing are unchanged.
- Used only when every window owner passes the existing policy-selection (serial-equivalence) guard; never for sampled,
  cohort, depth/survival-controller or timing-independent sessions.
- Drift check: on each load a rotating 15 % of the entries (at least one) is re-timed instead of seeded; a re-timed
  entry more than 15 % away from the table withdraws the whole table (`STRIX_COST_TABLE_RECHECK`,
  `STRIX_COST_TABLE_DRIFT`).
- Writes happen at the next request and at teardown: locked merge, atomic replace, never throwing. A missing, corrupt,
  oversized or mismatched file is ignored (cold start) and replaced by the next store.
- One stderr line per loaded bucket: `cost-table: qwen-mtp-minp bucket B loaded|missing|corrupt|version|mismatch
  entries=N recheck=rX file=<name>`. `STRIX_CYCLE_LOG=1` prints one line per window (rows, emitted tokens, ms, cost in
  use, own samples, live priors).

## Where the tables live and how to switch them off

On by default. Tables live in `STRIX_COST_TABLE_DIR` when it is set, else under the user cache directory:
`$XDG_CACHE_HOME/strix/cost-tables`, else `~/.cache/strix/cost-tables` (Windows: `%LOCALAPPDATA%/strix/cost-tables`).
The directory is created on the first store; if it cannot be written, the run simply has no table (cold start, as
before). A directory inside a source checkout (any ancestor holding `.git`) is refused. Deleting the directory resets
every table.

OFF: `STRIX_COST_TABLE=0`, or `STRIX_COST_TABLE_DIR=off` (also `0` or empty).
The historical baseline (`--baseline`, `STRIX_BASELINE=1`, `STRIX_QPF2K_DEFAULTS=0`) also turns the default directory
off; an explicit `STRIX_COST_TABLE_DIR` still applies there. The first MTP verifier of a process logs
`cost-table: directory source=env|default|off|baseline|refused|unavailable [dir=...]`.

## Measurement

gfx1151, Qwen3.8-Flash-Next with the distillG MTP sidecar, min-p 0.7, rows 2/4/8, HTTP server, 8 short prompts x 2
passes, 100 tokens, 6 paired cold/warm rounds (one pair per turn, order alternating, prompt order rotated, frozen
starting table, per-window log in every arm): decode rate over the first 32 tokens +3.3 % (95 requests), first 64
+0.5 %, whole request +0.15 % (95 % CI -0.19 to +0.49 %, warm faster in 5 of 6 rounds). Token ids, per-position
committed full-logit SHA-256 and final-state SHA-256 equal the no-table run (CLI agent/en2/es2 and the server set,
including requests whose window schedule changed).
