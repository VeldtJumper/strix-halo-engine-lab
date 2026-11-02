# Joint serving windows (opt-in)

`STRIX_MTP_BATCH_JOINT=1` enables joint window scheduling on the native HTTP
batch route. It is off by default. A single decoding request continues to use
the ordinary request controller and its existing execution path.

The batch controller searches the requests' R2/R4/R8 choices together under
`STRIX_COHORT_ROWS=16|32` (32 if unset). It measures a whole cohort topology,
rather than attributing its wall time to any one request's window owner. Each
slot has independent acceptance history, cleared at admission. Initial bounded
probes and periodic exploration prevent a first-use cost from permanently
excluding deeper windows. The controller can skip drafting for a request whose
expected accepted tokens do not repay the draft cost.

An undrafted request normally uses an R2 owner, verifies its pending target token
and commits exactly one row. A larger owner can pad a valid cohort. Padding is rolled back by the
existing prefix commit. Window admission still checks the request's own token
budget, context capacity and history capacity. Context and budget tails use the
existing fallback. Six-row plans are excluded because they cannot use a four-
or eight-row primary and would otherwise execute two independent chunks.

With `STRIX_COHORT_WIDE=all`, cohorts of 9–32 rows also share the HC projections,
router and shared expert, in addition to the existing GDN/QSA projections, head
and routed expert union. Every request retains its own recurrent state, KV,
capture, token selection and prefix commit. Padding is never published. These
paths retain the selected arithmetic expression for every row; native identity
qualification is required before promotion.

Diagnostics (off unless explicitly enabled):

- `STRIX_MTP_BATCH_LOG=1`: cohort geometry, emitted tokens, wall time and steady
  clock boundaries. Joint cycles also report draft and target phase times.
- `STRIX_COHORT_WIDE_LOG=2`: every cohort's wide, chunk-shared and own step counts;
  `=1` retains the sparse log.
- `STRIX_WINDOW_ROUTE_LOG=1`: per-session routes, including shared cohort steps.
  This synchronizes and must not be used for throughput measurements.

Shared-execution witness (host counters, no synchronization, always counted):
grouping requests does not prove that they shared work, since a group's chunks
may verify alone and a batched draft may decline. With `STRIX_MTP_BATCH_LOG=1`
every `batch cycle` line ends with the cycle's `chunks` (budget/joint chunks),
`shared_windows` / `shared_sessions` / `shared_rows` / `max_members` (windows
of two or more sessions on one primary), `draft_batches` /
`draft_batch_sessions` / `draft_launches` / `draft_shared_launches` /
`draft_shared_rows` / `draft_max_rows` (`STRIX_MTP_DRAFT_BATCH=1` launch chains
and launches of two or more rows), `joint_fallbacks` (context/budget tails that
ran the ordinary cohort cycle) and the reasons a session drafted alone
(`draft_declines=`: `sampled`, `exact`, `device_draft`, `keep`, `stopped`,
`depth`, `busy`, `broken`, `unsupported_branch`, `incompatible_owner`, `alone`,
`capacity`, `ab_off`, `inactive`) or verified alone (`verify_declines=`:
`no_rows`, `wide`, `padding`, `alone`, `no_owner`). Each HTTP batch request's
`strix.shared_execution` holds the same per request (`verify_shared_cycles`,
`draft_shared_steps`, maxima and declines). `tests/serve_slots_gate.py` qualifies
the joint, budget and draft-batch arms only when this witness shows shared
windows (joint/budget: in a cycle that ran chunks) and shared draft launches.

Logical weight sharing is not a hardware DRAM-byte measurement. Expert kernels
may tile assignments, caches may reread weights, and state/KV traffic grows with
the independent requests. End-to-end qualification must measure C1 and multiple
concurrent requests with diagnostics disabled and compare each request's tokens,
full committed vocabulary rows and authoritative state against a single run.

Identity validation covers greedy requests with a 1,024-token prompt and the
selected dense A8 profile, min-p 0.7, independent slots at concurrency 1/2/3/4,
64/128 output tokens and cohorts through 32 rows. The kernel qualification and
final scheduling check total 31 requests, 3,328 complete FP32 vocabulary rows and
217 canonical state components identical to C1. Occupied KV/index bytes and all
fixed state components are compared; unused cache tails and scratch are outside
the committed-state contract. The final C3 exploration check pads R2+R2+R2 to
R4+R2+R2 when admission allows it. Context/budget tails retain their smaller valid
chunks when they cannot be padded.

## Joint plan rule

When joint windows are enabled, `STRIX_MTP_JOINT_PLAN` defaults to
`calibrated-c3`: use the existing `calibrated` acceptance estimate for groups
with at least three members, and the `production` estimate for one or two.
The count is the number of members in the joint plan group at `plan()` time,
latched before drafting for that plan's `choose()`. It is independent of server
slot capacity, other active groups, physical row count and retained draft count.
Selection is deterministic given the plan group, controller history and costs.
The ordinary single-request route remains unchanged.

Calibration changes only the post-draft predictor acceptance estimate: each
probability bin uses the slot's observed acceptance, shrunk toward the
controller-wide bin. Context rows, plan-time priors, exploration and target
verification retain their existing rules. Explicit `calibrated` applies at all
group sizes; `production` (or `0`) restores the previous rule at all sizes.
Explicit `pooled`, `cap4` and `cap2` rules retain their existing meanings and
can be combined with a calibrated rule. `--baseline` / `STRIX_BASELINE=1`
defaults to production; an explicit plan rule overrides that default.

In-process A/B can select `production`, `calibrated` or `calibrated-c3` through
`STRIX_MTP_JOINT_RULE_FILE`. The file is read at each admission, using the same
parser; missing/empty files use the startup environment/default rule. Existing
`wave=N`, `snapshot` and `restore` controls retain controller snapshot/replay
behavior. Admission logs include `cal_min_members` (3 for the gated rule).

Measured calibrated versus production on the same public-main binary with
in-process controller snapshot/restore: C4 median wall aggregate throughput
+2.3–3.1 % (prose1k/prose8k/code1k); C2 −1.0 to −1.4 %; C8 median +0.55 % over
16 bracketed pairs (code1k and prose1k), 11/16 positive. Complete token IDs were
identical in those measurements; no new logits/state digest claim is made.
The C2 loss is consistent with coupled R4-to-R2 trims: the six-row R2+R4
topology is excluded because it rereads weights. The gated default preserves
the production rule there. These measurements support the threshold policy;
they do not establish a speedup for every workload or a measured C3 gain.
CPU controller tests cover parsing, baseline/explicit precedence and the switch
at group sizes 1/2/3/4/8, including a return below the threshold.
