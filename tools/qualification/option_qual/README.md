# Option qualification kit

Some engine options change the bits of the result (a different reduction order, 16-bit operands, a
blocked algorithm). They cannot pass an exactness gate against the reference identity, so they need a
quality qualification instead. This kit measures how far such an option moves the model's next-token
distribution and real-text prediction, against margins that are declared before the decisive data is
seen. It writes one receipt per option and per composition of options.

Exactness gates are not replaced. Within one declared identity, a fast kernel must still reproduce its
simple reference bit for bit. This kit only qualifies a new identity against the old one.

Python 3.10+ and numpy. CPU-only parts (manifest, plan, analysis, tests) run anywhere; the runner
executes the engine's plain decode command and therefore runs on a GPU node, inside one GPU turn.

## What is measured

Every arm (the reference included) runs the same prompt and is then teacher-forced on the same
recorded continuation: the real next tokens of the document, or the real assistant turn of a
conversation. The engine's measurement hook writes the full FP32 logits of every token selection
(`STRIX_BITCALC_LOGITS_OUT`, forced tokens from `STRIX_BITCALC_FORCE_TOKENS`). At each shared position,
with reference probabilities p and option probabilities q over the full vocabulary (float64):

| metric | meaning |
|---|---|
| overlap = sum min(p, q) = 1 - TV | the largest probability that both models pick the same token at this history, under the best coupling of the two draws; not greedy equality |
| KL(p || q), KL(q || p) | both directions are kept; they weigh different tails |
| top-1 agreement | equal argmax (lowest ID on ties); the reference's top-2 log-probability gap marks near ties |
| NLL of the real token | -log q(label) - (-log p(label)): the real-text prediction change |

Averages over recorded histories do not say what happens on histories that were not scored, and
teacher-forced next-token metrics are not task success.

## Data rules

* A **parent** document or conversation bundle is the statistical unit. A source item belongs to one
  parent only. All lengths of a parent are nested prefixes of the same token stream, so they stay in
  one cluster.
* The manifest (`manifest.py build`) freezes the token IDs of every case with SHA-256 hashes, the
  source hashes and licences, and the split. Within each stratum (for example en / es / code / chat),
  parents are ordered by `sha256(salt, parent)` and the first `--search-per-stratum` become SEARCH; the
  rest are HOLDOUT. The salt is fixed when the manifest is built, before any result exists.
* SEARCH is for pilots, debugging and margin calibration: look at it as much as needed. HOLDOUT is
  analysed once, under a frozen plan. If an option changes after a HOLDOUT look, its next qualification
  needs new HOLDOUT data. Every HOLDOUT runner invocation is appended to `holdout-ledger.jsonl`.
* Chat parents are rendered in the chat template, every assistant message with the empty reasoning
  block that precedes a served non-thinking answer; content with special-token or reasoning markup is
  rejected. A case's prompt ends exactly where an assistant answer begins (the last such start at or
  before the nominal length and not before 75 % of it), and its continuation stays inside that answer
  (never the end marker).

## Statistics and verdicts

The plan (`plan.py`) names the reference, an optional control (an arm whose change is accepted as
ordinary rounding noise, for example a pure FP32 reordering), the options, compositions, lengths and
margins. Default endpoints, each a mean over parents of the per-parent difference:

| id | quantity | default margin |
|---|---|---|
| E1 | NLL change vs reference (nats/token) | 0.005 (about +0.5 % perplexity) |
| E2 | overlap loss of the option minus that of the control | max(0.002, 0.5 x control mean on SEARCH) |
| E3 | KL(reference || option) minus that of the control | max(5e-4, 0.5 x control mean on SEARCH) |
| E4 | top-1 disagreement minus that of the control | 0.005 |
| E5 | KL(reference || option) | 0.04 (fast-parity) |
| E6 | top-1 disagreement with the reference | 0.03 (fast-parity) |
| E7 | NLL change of the option minus that of the control (nats/token) | 0.005 |

These form margin sets (`plan init --tier`). `exact-class` (the default) is meant for an individual
option: with a control it decides on E7, E2, E3 and E4 (the guard on E7), because a pure rounding change
already moves the NLL of chat text at 8K/32K, so the absolute change E1 cannot tell options apart; without
a control it uses E1. `reference-nll` (E1 alone, with its guard) is meant as the secondary set that
reports the absolute NLL change beside such a verdict. `fast-parity` bounds a composed fast mode in absolute terms: E1 below 0.0129
(+1.3 % perplexity), E5 below 0.04 and E6 below 3 %, with the same per-stratum guard on E1; it needs no
control. `--secondary-tier` evaluates a second set on the same runs and reports it beside the verdict
(for example a fast mode decided by `fast-parity` and also shown against `exact-class`); it never
changes the verdict.

Two estimator options can be declared with `plan init` (they change how an endpoint is estimated, not
its margin). `--e7-control-clip K`: E7 per case is the option's NLL change minus the control's clipped to
+-K nats/token, then averaged per parent; a case where the control itself derails (a rounding change can
move a whole chat document by tenths of a nat) can then neither excuse nor penalise the option by more
than K. `--stratum-weights equal`: every endpoint is the mean of the stratum means, so a design that puts
more parents into a noisy stratum does not give that stratum more weight; the bootstrap weights each
resampled parent by 1/(S n_s) and the t bound uses the stratified standard error (Satterthwaite degrees
of freedom). A stratum with fewer than two usable parents leaves such an endpoint without a bound,
which makes the verdict INCOMPLETE, never FAIL. The control means that `plan calibrate` uses are means
over parents (not over strata), whatever the plan's weighting; a plan that wants other margins declares
them as fixed numbers. The clipped E7 is its own declared estimand (option minus a control whose per-case
movement is capped in both directions), not an unbiased estimate of the unclipped E7, and receipts say so
in the endpoint label.

Long contexts (every length 131072 and more) use `exact-class-long` / `fast-parity-long` instead, and a
plan never mixes long and shorter lengths. There any rounding change moves a single document's NLL by
tenths of a nat in both directions, so E1 gets a wide margin (0.05) and the decision rests on the
full-distribution endpoints; such a receipt reads "rounding-sensitive regime; non-inferior to a
rounding-equivalent control", not a per-document NLL guarantee. Every margin must be a positive finite
number; a calibrated margin must follow from its rule and the control mean recorded by `plan calibrate`,
and freeze, HOLDOUT runs and HOLDOUT analysis all apply that same check.

`plan calibrate` turns the rule margins into numbers from a clean SEARCH analysis (complete, finite
reference and control coverage, no global problems) and binds the plan to that SEARCH's reference and
control definitions, binary and model receipt: a HOLDOUT run with anything else is INVALID. `plan freeze`
stamps the plan. A guard needs its endpoint in the plan, and a stratum needs at least two parents. The runner refuses HOLDOUT with an unfrozen plan and stamps the plan's SHA-256 into every run;
the analysis refuses runs stamped with another plan, manifest or binary.

For each endpoint the one-sided upper bound at level `1 - alpha / k` (k = options + compositions that
are qualified together, or the whole declared `--family` when the options of one release are
qualified in separate campaigns on separate binaries) is the larger of a stratum-preserving parent bootstrap quantile (the same
resamples for every endpoint) and a Student t bound on parent means. An option passes when every
endpoint's bound is below its margin, the per-stratum point estimate of the guarded NLL endpoint (E7
for exact-class with a control, E1 otherwise; the plan stores which) stays below twice its margin, and
all hard gates hold: every planned run complete, no non-finite logits, identical forced
histories, identical prefill route lines across arms (or, for an arm that intentionally takes another
route, lines matching the route the plan declares for it with `--route ARM=REGEX`), and bitwise-identical repeats of the reference
where the plan asks for them. A bound needs more parents than strata (residual degrees
of freedom); otherwise the option is INCOMPLETE. Every in-scope parent must be complete for the option
and the control;
problems of the control or of the reference concern every option. Results must carry their provenance:
each runner output directory is validated on its own (its run-meta, run records, copied arms file,
done markers and reductions), each reduction must name the logits hashes of a validated complete run (whose
row and forced-step counts equal the continuation) and of the reference run of the same directory, its
metric arrays must have the hash its done marker recorded, retained logits files must still match, the model receipt must be the one the manifest was
tokenized with, and directories combined in one analysis must share one identity and hold disjoint
cases (give each node its own `--cases`); a case found in two directories is INVALID, and its logged labels must equal the
manifest continuation (the runner also checks the forced-step count, consecutive steps and positions,
so an arm that silently ignores teacher forcing can never complete). Verdicts: PASS, FAIL, INCOMPLETE (missing data, too few parents,
unresolved margin), INVALID (plan, manifest or binary mismatch, broken history). A SEARCH analysis
prefixes its verdicts with `SEARCH-` and is never a qualification.

## Running

From the repository root (placeholders in capitals):

```sh
python3 -m tools.qualification.option_qual.manifest build --sources SOURCES.jsonl --out MANIFEST \
  --tokenizer bin/strix-tokenize --model MODEL.gguf --model-receipt RECEIPT \
  --lengths 8192,32768 --cont 256 --stop-ids EOS_IDS --salt SALT --search-per-stratum 4
python3 -m tools.qualification.option_qual.plan init --manifest MANIFEST --split search \
  --reference REF --control CTRL --options OPT_A,OPT_B --composition ALL=OPT_A+OPT_B \
  --lengths 8192,32768 --repeat-cases CASE_ID --out SEARCH_PLAN.json
# inside a GPU turn on the node (prompt IDs are passed as arguments: raise the stack limit first)
ulimit -s 65536
python3 -m tools.qualification.option_qual.run --manifest MANIFEST --plan SEARCH_PLAN.json \
  --arms ARMS.json --binary bin/strix-hip-decode-token --model MODEL.gguf --model-receipt RECEIPT \
  --out RUNS_SEARCH --soft-seconds 430
python3 -m tools.qualification.option_qual.analyze --manifest MANIFEST --plan SEARCH_PLAN.json \
  --runs RUNS_SEARCH --receipts RECEIPTS_SEARCH
python3 -m tools.qualification.option_qual.plan init ... --split holdout --out HOLDOUT_PLAN.json
python3 -m tools.qualification.option_qual.plan calibrate HOLDOUT_PLAN.json \
  --search-summary RECEIPTS_SEARCH/summary.json --out HOLDOUT_PLAN.json
python3 -m tools.qualification.option_qual.plan freeze HOLDOUT_PLAN.json
```

`ARMS.json` maps arm names to extra environment variables and arguments, for example
`{"arms": {"REF": {"env": {}}, "OPT_A": {"env": {"OPTION_SWITCH": "1"}, "argv": []}}}`. The runner
accepts optional `"model": "OPTION.gguf"` and `"model_receipt": "OPTION.receipt"` paths together
in each arm for `--replay` only; direct runs refuse any arm model override. Each replay capture,
including the reference repeat, must name the model SHA-256 from that arm's receipt. Omit both to use
`--model` and `--model-receipt`. The binary stays identical across arms; the manifest's tokenizer receipt
remains tied to the run's `--model-receipt`. Run metadata, holdout ledgers and receipts record
`arm_model_receipt_sha256` only when the arms file has overrides. Each run record binds its arm's
receipt. The arms-file hash in run metadata covers these overrides; successful records in such directories require replay capture
provenance, including independent reference and repeat captures. Historical direct captures must be
reimported into a new output directory; adding `--replay` cannot validate old records. Plans do not
freeze an arms-file hash. The runner starts each arm from a clean environment (`PATH`, `HOME`, `LD_LIBRARY_PATH` only, every inherited
engine variable dropped) and sets the measurement hook variables itself. It is resumable: run it again
in the next turn until it exits 0 (2 means the soft budget stopped it; 1 means a run failed: the other
arms and cases still run, and a failing pair is retried up to `--max-attempts` times, then counts
against its option). An output directory is bound to its plan, manifest, arms file, binary and model
receipt; anything else needs a new `--out`. Each run gets its own execution cap (1.5 x, and at least
60 s above, the longest completed run of that length or `--predict`), never more than the soft budget
less `--reserve-seconds`, the observed post-processing time and the time the invocation itself spent
before its first run (fixed from then on, at least 10 s); a run is
started only if its whole cap and post-processing still fit. If no invocation could fit a run that has to
execute, the runner exits 1 with the numbers to change (a run crowded out only by one invocation's
slow start-up is a budget stop); cases that only need a
reduction or are finished are never held back by this. A timeout counts as a failure only when the
run had its full cap, otherwise it is a budget stop. Checking and hashing a finished run's logits
happens in a child process stopped at the deadline (a budget stop: the unchecked file is discarded
and the run is repeated, without counting a failure); re-checking retained logits on resume is bounded
the same way. Engine output fields are checked before they
are written as evidence; a malformed value is kept only as a diagnostic and the run counts as a
failed attempt. Reductions run in child processes
(at most one queued behind the GPU, its time reserved at admission); nothing started by an invocation
outlives `--soft-seconds`: a run or reduction still going then is stopped, and the next invocation
reduces a finished run's kept logits instead of running it again. Every saved reduction is bound by
the hashes of both its files, which are checked before it is loaded; a damaged or missing one is
produced again on resume. Damaged lines of the diagnostic journals (effective-config, recovery,
holdout ledger) are kept and skipped; the receipt reports them, and capture is simply repeated.
At most one reduction is queued behind the GPU. Each arm's `--print-effective-config` output is captured
once per output directory (at most 90 s per invocation; `--no-effective-config` skips it); every later
invocation records which capture it relies on. Arm names contain no `.`, `/` or `@`, and a plan needs at
least one gating endpoint. Logit files (about 1 MB per
position) are deleted as soon as they are reduced; `--keep-logits` keeps them. `reduce.py` also works
on a pair of teacher spools (`--format spool`).

### Run witness

An engine can decline a lever or fall back to another route at run time (for example when memory is short), and
the run then produces another arm's numbers without any error. A plan can declare, per arm, what every run of that
arm must show (`plan.py init --witness FILE`, frozen with the plan):

```json
{"OPT": {"stdout": {"lever_engaged": true, "storage_identity": "fp16-v1+lever"},
         "stderr_forbid": ["declined", "insufficient-memory"],
         "stderr_require": ["reason=qualified-prompt"]},
 "REF": {"stdout": {"lever_engaged": false}}}
```

`stdout` keys are compared with the run's JSON result as serialized JSON values (`1`, `1.0`, `true` and `"1"`
all differ); `stderr_forbid` texts must not and `stderr_require` patterns must occur in the run's stderr (decoded
as UTF-8, no newline translation). A validated run that misses its witness is not evidence: it counts as a failed
attempt (`witness: ...` in its run record) and is repeated up to `--max-attempts`. Witness misses never count
toward FAIL: a pair is FAIL only when its ordinary failures alone reach the limit, otherwise it is INCOMPLETE; this
is read from the run journal, so a marker restored after an interruption is classified the same way. Every run
record binds the sha256 of its saved stdout/stderr logs; the analysis checks the witness of every kept run (the
reference included, before any of its data is used) on exactly those bytes. A kept run whose logs are missing,
changed or lack the witness, or whose record does not bind its logs (an older kit), is INVALID. Plans without a
witness are unchanged.

## Receipts

`RECEIPTS/<option>.receipt.json` holds the identities (kit version and commit, manifest, frozen plan
and time, arms file, binary and model receipt hashes), the gates and problems, the endpoint table
(estimate, bootstrap and t bounds, margin, pass), the guard, and a descriptive report: parent means of
every metric, perplexity change in percent, lower-tail overlap quantiles, maximum KL, agreement away
from near ties, worst and best parent, and per-stratum and per-length tables. `<option>.receipt.md`
is a short human summary; `summary.json` / `summary.md` cover all options and carry the control means
used by `plan calibrate`.

## Limits

* Few parents give wide bounds; add parents rather than positions, because positions inside a
  parent are strongly correlated.
* Long contexts (131K and beyond) need their own plan scope: their cost forces fewer parents, and an
  option that matters only there must be qualified there.
* A pass bounds the average change on this corpus. Rare failure modes, multi-step task success and
  free-running divergence need their own checks.
