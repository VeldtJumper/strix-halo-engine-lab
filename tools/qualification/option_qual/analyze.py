"""Gates, endpoint bounds and verdicts from runner output; writes one receipt per option/composition.

usage:
  python3 -m tools.qualification.option_qual.analyze --manifest DIR --plan PLAN.json --runs OUT[,OUT2...] \
      --receipts DIR

Parents are the statistical units: each parent's scored positions (all in-scope lengths) are pooled
into one mean per arm, and parents weigh equally. Every option and composition of the plan is bounded
simultaneously: one-sided level 1 - alpha/k, k = options + compositions (or the declared family).

Each runner output directory is a self-contained provenance unit: its run-meta, run records, done
markers and reductions must agree with each other and with the plan. Several directories (for example
one per node) may be combined only if they hold disjoint cases and share one identity.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np

from .common import captured_text
from .common import (VERSION, JournalError, witness_problem, decode_log, diagnostic_records, atomic, canonical, file_sha, kit_commit, read_id_text, read_journal,
                     read_journal_lines, sha)
from . import manifest as manifest_mod
from . import plan as plan_mod
from .receipt import render_md, render_summary_md
from .reduce import load as load_red, load_bytes as load_red_bytes
from .run import REPEAT_SUFFIX
from . import evidence
from .stats import resample_indices, upper_bound

NEAR_TIE = 0.05
ORDER = {"PASS": 0, "INCOMPLETE": 1, "FAIL": 2, "INVALID": 3}  # an observed failure outranks missing data
IDENTITY = ("plan_sha256", "manifest_sha256", "split", "arms_sha256", "binary_sha256", "model_receipt_sha256",
            "arm_model_receipt_sha256")


def _worst(*verdicts):
    return max(verdicts, key=ORDER.__getitem__)


def _journal(path, problems):
    """Records of a journal; an incomplete tail or malformed record becomes a problem, never a crash."""
    try:
        records, tail = read_journal(path)
    except JournalError as e:
        problems.append(f"{path.parent.name}: {e}")
        return []
    if tail is not None:
        problems.append(f"{path.parent.name}: {path.name} ends with an incomplete record "
                        "(run the runner again to quarantine it)")
    return records


class RunDir:
    """One runner output directory, validated on its own."""

    def __init__(self, path, required):
        self.path = Path(path)
        self.problems = []
        meta = _journal(self.path / "run-meta.json", self.problems)
        self.identity = None
        self.arms = None
        arms_path = self.path / "arms.json"
        arms_bytes = (arms_path.read_bytes()
                      if meta and isinstance(meta[0], dict) and arms_path.exists() else None)
        if meta and isinstance(meta[0], dict) and arms_bytes is not None and sha(arms_bytes) == meta[0].get("arms_sha256"):
            self.arms = json.loads(captured_text(arms_bytes))["arms"]
        elif meta:
            self.problems.append(f"{self.path.name}: arms.json copy missing or not the stamped arms file")
        if not meta:
            self.problems.append(f"{self.path.name}: run-meta.json missing (no provenance)")
        else:
            try:
                ids = set()
                for m in meta:
                    receipts = evidence.normalize_arm_receipts(
                        m, required, allow_legacy=self.arms is not None and not evidence.has_model_overrides(self.arms))
                    ids.add(canonical(dict({k: m.get(k) for k in IDENTITY}, arm_model_receipt_sha256=receipts)))
                if len(ids) != 1 or None in json.loads(next(iter(ids))).values():
                    raise ValueError("invocations disagree or lack identity fields")
                self.identity = json.loads(next(iter(ids)))
            except (ValueError, TypeError, AttributeError) as e:
                self.problems.append(f"{self.path.name}: {e}")
        if not (self.path / "runs.jsonl").exists():
            self.problems.append(f"{self.path.name}: runs.jsonl missing (no provenance)")
        self.runs_bytes = None
        try:
            ledger_path = self.path / "runs.jsonl"
            self.runs_bytes = ledger_path.read_bytes() if ledger_path.exists() else b""
            lines, tail, raws = read_journal_lines(self.path / "runs.jsonl", data=self.runs_bytes)
        except OSError as e:
            lines, tail, raws = [], None, []
            self.problems.append(f"{self.path.name}: runs.jsonl unreadable ({e})")
        if tail is not None:
            self.problems.append(f"{self.path.name}: runs.jsonl ends with an incomplete record "
                                 "(run the runner again to quarantine it)")
        resets = _journal(self.path / "resets.jsonl", self.problems)
        self.require_replay = self.arms is not None and evidence.has_model_overrides(self.arms)
        self.ledger = evidence.Ledger(lines, resets, self.identity or {}, raws,
                                      require_replay=self.require_replay)
        if self.ledger.bad_resets:
            self.problems.append(f"{self.path.name}: resets.jsonl holds resets not bound to this run journal "
                                 "(run the runner again to repair)")
        for pos, case, why in self.ledger.uncovered_bad():
            self.problems.append(f"{self.path.name}: runs.jsonl line {pos + 1} ({case or 'unknown case'}): {why}"
                                 " (run the runner again to reset the case)")
        # A diagnostic: damaged lines are skipped and reported, never a problem with the evidence.
        records, self.effective_damaged = diagnostic_records(self.path / "effective-config.jsonl")
        self.effective = {r["arm"]: r.get("sha256") for r in records if isinstance(r.get("arm"), str)}

    def holds(self, case_id, labels):
        done = self.path / "done"
        return any((done / f"{case_id}.{lab}").exists() for lab in labels)

    def bound(self, case, label, sha, cont_tokens, mark, plan_sha):
        """The run record the marker names, if the shared evidence rules accept it; else None."""
        if not self.identity or not self.ledger.result_bound(case, label, sha, cont_tokens, mark, plan_sha):
            return None
        return self.ledger.by_id(mark.get("run_id"), counting_case=case)

    def marker(self, case, label):
        path = self.path / "done" / f"{case}.{label}"
        if not path.exists():
            return None
        try:
            mark = json.loads(path.read_text())
            if isinstance(mark, dict) and "status" in mark:
                return mark
        except ValueError:
            pass
        return {"status": "UNREADABLE"}

    def failure_bound(self, case, label, mark, ref_sha, plan_sha):
        """A FAILED/INVALID marker is the newest outcome of this pair (shared rules in evidence.py)."""
        return bool(self.identity) and self.ledger.outcome_bound(mark, case, label, ref_sha, plan_sha)

    def failure_count(self, case, label):
        return len(self.ledger.failed(case, label))

    def witness_failure_count(self, case, label):
        return len(self.ledger.witness_failed(case, label))

    def reduction(self, case, label):
        stem = self.path / "red" / f"{case}.{label}"
        return load_red(stem) if Path(str(stem) + ".json").exists() else None

    def verified(self, case, label, npz_sha, json_sha):
        """The reduction bundle, loaded only if both files exist and carry the recorded hashes."""
        stem = self.path / "red" / f"{case}.{label}"
        npz, js = Path(str(stem) + ".npz"), Path(str(stem) + ".json")
        if not (npz_sha and json_sha and npz.exists() and js.exists()):
            return None
        try:
            npz_bytes, json_bytes = npz.read_bytes(), js.read_bytes()
            if sha(npz_bytes) != npz_sha or sha(json_bytes) != json_sha:
                return None
            return load_red_bytes(npz_bytes, json_bytes)
        except (OSError, ValueError, KeyError):
            return None

    def logits(self, case, label):
        path = self.path / "logits" / f"{case}.{label}.bin"
        return path if path.exists() else None


def analyze(manifest_dir, plan_path, run_dirs, *, _captured=None):
    plan, plan_digest = plan_mod.load(plan_path)
    manifest, manifest_digest = manifest_mod.load(manifest_dir)
    if plan["manifest_sha256"] != manifest_digest:
        raise ValueError("plan was made for another manifest")
    dirs = [RunDir(d, plan_mod.arms_needed(plan)) for d in run_dirs]
    if _captured is not None:
        # Private verification channel; never changes public summaries or receipt bytes.
        _captured.update(plan_sha256=plan_digest,
                         ledgers={d.path.resolve(): sha(d.runs_bytes) if d.runs_bytes is not None else None
                                  for d in dirs})
    ref, ctrl = plan["reference"], plan.get("control")
    qualified = plan["options"] + [c["name"] for c in plan["compositions"]]
    arms = ([ctrl] if ctrl else []) + qualified
    cases = [c for c in manifest["cases"] if c["split"] == plan["split"] and c["nominal_length"] in plan["lengths"]]
    problems = defaultdict(list)          # option name or "*" -> [(verdict, text)]

    # --- global gates: identity, provenance, plan, calibration source
    for d in dirs:
        problems["*"].extend(("INVALID", p) for p in d.problems)
    identities = {canonical(d.identity) for d in dirs if d.identity}
    identity = json.loads(next(iter(identities))) if len(identities) == 1 else None
    if len(identities) > 1:
        problems["*"].append(("INVALID", "run directories have different identities (plan/binary/arms/receipt)"))
    if identity:
        if identity["plan_sha256"] != plan_digest:
            problems["*"].append(("INVALID", "runs stamped with another plan sha"))
        if identity["manifest_sha256"] != manifest_digest or identity["split"] != plan["split"]:
            problems["*"].append(("INVALID", "runs made on another manifest or split"))
        tok_receipt = manifest["tokenizer"].get("model_receipt_sha256")
        if not manifest.get("fake_tokenizer") and tok_receipt != identity["model_receipt_sha256"]:
            problems["*"].append(("INVALID", "model receipt differs from the manifest's tokenizer receipt"))
    if plan["split"] == "holdout":
        unresolved = plan_mod.unresolved(plan)
        if unresolved:
            problems["*"].append(("INVALID", f"holdout analysis needs a frozen plan with resolved margins: {unresolved}"))
    plan_problem = plan_mod.check(plan)
    if plan_problem:
        problems["*"].append(("INVALID", plan_problem))
    arm_defs = next((d.arms for d in dirs if d.arms), None)
    source = plan.get("calibration_source")
    if source is not None and identity and not plan_problem:
        try:
            summary = source.get("search_summary", {})
            source_meta = dict(summary, model_receipt_sha256=source["model_receipt_sha256"])
            calibrated = evidence.normalize_arm_receipts(source_meta, [n for n in (ref, ctrl) if n])
            if any(identity["arm_model_receipt_sha256"][n] != digest for n, digest in calibrated.items()):
                problems["*"].append(("INVALID", "effective reference/control model receipts differ from calibrated SEARCH receipts"))
        except ValueError as e:
            problems["*"].append(("INVALID", f"calibration source: {e}"))
        if (identity["binary_sha256"], identity["model_receipt_sha256"]) != \
                (source["binary_sha256"], source["model_receipt_sha256"]) or arm_defs is None or \
                arm_defs.get(ref) != source["reference_def"] or arm_defs.get(ctrl) != source["control_def"]:
            problems["*"].append(("INVALID", "runs do not use the binary/model/reference/control the margins "
                                             "were calibrated with"))

    # --- each case must come from exactly one directory
    all_labels = [ref] + arms + [ref + REPEAT_SUFFIX]
    owner = {}
    for c in cases:
        holders = [d for d in dirs if d.holds(c["case_id"], all_labels)]
        if len(holders) > 1:
            problems["*"].append(("INVALID", f"{c['case_id']}: results in {len(holders)} run directories"))
        owner[c["case_id"]] = holders[0] if len(holders) == 1 else None

    data = defaultdict(lambda: defaultdict(list))   # arm -> parent -> [arrays per case]
    case_order = defaultdict(lambda: defaultdict(list))   # arm -> parent -> [case ids, in data order]
    self_data = defaultdict(list)
    lengths_of = defaultdict(lambda: defaultdict(list))  # arm -> length -> [arrays]
    route_lines = set()
    declared = plan.get("routes", {})
    for c in cases:
        cid = c["case_id"]
        d = owner[cid]
        if d is None:
            problems["*"].append(("INCOMPLETE", f"{cid}: no results"))
            continue
        cont = read_id_text(Path(manifest_dir) / c["cont_file"])
        ref_mark = d.marker(cid, ref)
        if ref_mark and ref_mark.get("status") in ("UNREADABLE",) or (
                ref_mark and ref_mark.get("status") == "FAILED"
                and not d.failure_bound(cid, ref, ref_mark, None, plan_digest)):
            problems["*"].append(("INVALID", f"{cid}: reference marker unreadable or not bound to a run record"))
            continue
        if not ref_mark or ref_mark.get("status") != "COMPLETE" or "self_red_sha256" not in ref_mark:
            problems["*"].append(("INCOMPLETE", f"{cid}: reference missing"))
            continue
        ref_red = d.verified(cid, f"{ref}.self", ref_mark.get("self_red_sha256"), ref_mark.get("self_redj_sha256"))
        if ref_red is None:
            problems["*"].append(("INVALID", f"{cid}: reference reduction missing, damaged or not the recorded one"))
            continue
        ref_sha = ref_mark.get("sha256")
        raw = d.logits(cid, ref)
        ref_rec = d.bound(cid, ref, ref_sha, c["cont_tokens"], ref_mark, plan_digest)
        if (ref_mark.get("plan_sha256") != plan_digest or ref_rec is None
                or ref_red[1]["ref_sha256"] != ref_sha or ref_red[0]["label"].tolist() != cont
                or (raw is not None and file_sha(raw) != ref_sha)):
            problems["*"].append(("INVALID", f"{cid}: reference result not bound to a validated run"))
            continue
        why = kept_witness(d.path, cid, ref, ref_rec, plan.get("witness", {}).get(ref))
        if why:
            problems["*"].append(("INVALID", f"{cid}: {ref} kept run lacks its witness: {why}"))
            continue
        if int(ref_red[0]["nonfinite"].sum()):
            problems["*"].append(("FAIL", f"{cid}: non-finite reference logits"))
            continue
        self_data[c["parent_id"]].append(ref_red[0])
        routes = {ref: ref_rec["route"]}
        route_lines.update(routes[ref] or [])
        labels = arms + ([ref + REPEAT_SUFFIX] if cid in plan.get("ref_repeat_cases", []) else [])
        for lab in labels:
            mark = d.marker(cid, lab)
            # Problems of the control or of the reference repeat concern every option.
            key = lab if lab in qualified else "*"
            if mark is None:
                problems[key].append(("INCOMPLETE", f"{cid}: {lab} missing"))
                continue
            if mark["status"] == "UNREADABLE" or (mark["status"] in ("FAILED", "INVALID")
                                                   and not d.failure_bound(cid, lab, mark, ref_sha, plan_digest)):
                problems[key].append(("INVALID", f"{cid}: {lab} marker unreadable or not bound to a run record"))
                continue
            if mark["status"] == "INVALID":
                problems[key].append(("INVALID", f"{cid}: {lab}: {mark.get('error')}"))
                continue
            if mark["status"] == "FAILED":
                # A witness miss (the engine did not run what the arm declares: lever declined, route fell
                # back) is missing evidence, never a failed option: it never counts toward FAIL. Classified
                # from the journal's run records, not from the marker's text.
                misses = d.witness_failure_count(cid, lab) if plan.get("witness") else 0
                final = d.failure_count(cid, lab) - misses >= mark.get("max_attempts", 2)
                problems[key].append(("FAIL" if final else "INCOMPLETE",
                                      f"{cid}: {lab} run failed ({mark.get('attempts', 1)} attempts"
                                      + (f", {misses} witness misses)" if misses else ")")))
                continue
            if mark["status"] != "COMPLETE":
                problems[key].append(("INCOMPLETE", f"{cid}: {lab} {mark['status']}"))
                continue
            red = d.verified(cid, lab, mark.get("red_sha256"), mark.get("redj_sha256"))
            if red is None:
                problems[key].append(("INVALID", f"{cid}: {lab} reduction missing, damaged or not the recorded one"))
                continue
            arrays, info = red
            run_label = lab
            raw = d.logits(cid, lab)
            if (mark.get("plan_sha256") != plan_digest or mark.get("ref_sha256") != ref_sha
                    or info["ref_sha256"] != ref_sha or info["cand_sha256"] != mark.get("cand_sha256")
                    or d.bound(cid, run_label, info["cand_sha256"], c["cont_tokens"], mark, plan_digest) is None
                    or (raw is not None and file_sha(raw) != info["cand_sha256"])
                    or arrays["label"].tolist() != cont):
                problems[key].append(("INVALID", f"{cid}: {lab} result not bound to a validated run and reference"))
                continue
            rec = d.bound(cid, run_label, info["cand_sha256"], c["cont_tokens"], mark, plan_digest)
            routes[lab] = rec["route"]
            why = kept_witness(d.path, cid, lab, rec, plan.get("witness", {}).get(
                lab[:-len(REPEAT_SUFFIX)] if lab.endswith(REPEAT_SUFFIX) else lab))
            if why:
                problems[key].append(("INVALID", f"{cid}: {lab} kept run lacks its witness: {why}"))
                continue
            if lab.endswith(REPEAT_SUFFIX):
                # A replayed repeat must be another capture of another engine invocation than the accepted
                # reference (equal logits bytes are expected; equal capture identity is not a repeat).
                rc_ref, rc_rep = (ref_rec or {}).get("replay_capture"), (rec or {}).get("replay_capture")
                if d.require_replay or rc_ref is not None or rc_rep is not None:
                    if not (evidence.replay_provenance(ref_rec) and evidence.replay_provenance(rec)
                            and rc_ref["capture_id"] != rc_rep["capture_id"]
                            and rc_ref["collection_id"] != rc_rep["collection_id"]):
                        problems["*"].append(("INVALID", f"{cid}: reference repeat is not an independent capture"))
                        continue
                # Identity of the repeat is decided by the two validated logits files themselves.
                if info["cand_sha256"] != ref_sha or not mark.get("bitwise_identical"):
                    problems["*"].append(("FAIL", f"{cid}: reference repeat not bitwise identical"))
                continue
            if int(arrays["nonfinite_r"].sum()) or int(arrays["nonfinite_c"].sum()):
                problems[key].append(("FAIL", f"{cid}: {lab} non-finite logits"))
                continue
            data[lab][c["parent_id"]].append(arrays)
            case_order[lab][c["parent_id"]].append(cid)
            lengths_of[lab][c["nominal_length"]].append(arrays)
        for lab, route in routes.items():
            arm = lab[:-len(REPEAT_SUFFIX)] if lab.endswith(REPEAT_SUFFIX) else lab
            if arm in declared:
                ok = bool(route) and all(re.search(declared[arm], line) for line in route)
            else:
                ok = route == routes[ref]
            if not ok:
                why = "does not match its declared route" if arm in declared else "differs from reference"
                problems[lab if lab in qualified else "*"].append(("FAIL", f"{cid}: {lab} route {why}: {route}"))

    parents = {p["parent_id"]: p for p in manifest["parents"]}
    in_scope = sorted({c["parent_id"] for c in cases})
    per_stratum = {s: 0 for s in sorted({p["stratum"] for p in manifest["parents"]})}
    for pid in in_scope:
        per_stratum[parents[pid]["stratum"]] += 1
    for s_name, n in per_stratum.items():
        if n < plan["min_parents_per_stratum"]:
            problems["*"].append(("INCOMPLETE", f"stratum {s_name}: {n} parents in scope < "
                                                f"{plan['min_parents_per_stratum']}"))

    # --- per-parent means: a parent enters an option's statistics when the reference, the control
    # and that option are complete for every in-scope case of the parent.
    n_cases = defaultdict(int)
    for c in cases:
        n_cases[c["parent_id"]] += 1

    def complete(arm, pid):
        return len(data[arm][pid]) == n_cases[pid] and len(self_data[pid]) == n_cases[pid]

    metric_fns = {"dnll": lambda x: x["nll_c"] - x["nll_r"], "loss": lambda x: 1.0 - x["overlap"],
                  "kl_rc": lambda x: x["kl_rc"], "kl_cr": lambda x: x["kl_cr"],
                  "disagree": lambda x: 1.0 - x["agree"], "nll_c": lambda x: x["nll_c"],
                  "nll_r": lambda x: x["nll_r"]}

    def means(arm, pids):
        out = {m: np.array([float(np.concatenate([f(x) for x in data[arm][p]]).mean()) for p in pids])
               for m, f in metric_fns.items()}
        # Per-case NLL changes (for a clipped-control E7), with the case ids they belong to.
        out["_case_dnll"] = [(list(case_order[arm][p]), np.array([float(np.mean(x["nll_c"] - x["nll_r"]))
                                                                  for x in data[arm][p]])) for p in pids]
        return out

    k = max(len(set(plan.get("family", [])) | set(qualified)), 1)
    level = 1.0 - plan["alpha"] / k
    results = {}
    for opt in qualified:
        verdict, notes = "PASS", []
        for v, text in problems["*"] + problems[opt]:
            verdict = _worst(verdict, v)
            notes.append(f"{v}: {text}")
        usable = [p for p in in_scope if complete(opt, p) and (not ctrl or complete(ctrl, p))]
        strata = [parents[p]["stratum"] for p in usable]
        if len(usable) < len(in_scope):
            # A pass must rest on every planned parent; dropped parents are never silent.
            verdict = _worst(verdict, "INCOMPLETE")
            notes.append(f"INCOMPLETE: {len(in_scope) - len(usable)} of {len(in_scope)} parents lack complete data")
        endpoints, guards, report, secondary = {}, {}, None, None
        if len(usable) <= len(set(strata)):
            verdict = _worst(verdict, "INCOMPLETE")
            notes.append("INCOMPLETE: not more complete parents than strata (no residual degrees of freedom)")
        # Hard gates and coverage are shared by every margin set; each set adds only its own outcome.
        shared, shared_notes = verdict, list(notes)
        if len(usable) > len(set(strata)):
            pp = {opt: means(opt, usable), **({ctrl: means(ctrl, usable)} if ctrl else {})}
            idx = resample_indices(strata, plan["bootstrap"]["B"], plan["bootstrap"]["seed"])
            for name, specs, guard_specs in plan_mod.margin_sets(plan):
                outcome, set_notes, ends, grds = evaluate(specs, guard_specs, opt, ctrl, pp, strata, level, idx,
                                                          plan.get("stratum_weights"))
                if name == "primary":
                    endpoints, guards = ends, grds
                    verdict = _worst(shared, outcome)
                    notes = shared_notes + set_notes
                else:
                    # Reported beside the verdict, never changing it: the shared gates plus its own outcome.
                    secondary = {"tier": plan["secondary"]["tier"],
                                 "statement": plan_mod.TIER_STATEMENTS.get(plan["secondary"]["tier"]),
                                 "verdict": _worst(shared, outcome),
                                 "notes": shared_notes + set_notes, "endpoints": ends, "guards": grds}
            report = describe(opt, usable, strata, pp[opt], data, lengths_of)
        if plan.get("secondary") and secondary is None:
            secondary = {"tier": plan["secondary"]["tier"],
                         "statement": plan_mod.TIER_STATEMENTS.get(plan["secondary"]["tier"]),
                         "verdict": shared, "notes": shared_notes,
                         "endpoints": {}, "guards": {}}
        results[opt] = {"verdict": verdict, "notes": notes, "endpoints": endpoints, "guards": guards,
                        "tier": plan.get("tier", "exact-class"),
                        "statement": plan_mod.TIER_STATEMENTS.get(plan.get("tier", "exact-class")),
                        "secondary": secondary,
                        "members": next((c["members"] for c in plan["compositions"] if c["name"] == opt), None),
                        "parents_used": len(usable),
                        "parents_used_per_stratum": {x: strata.count(x) for x in sorted(set(strata))},
                        "report": report}

    # --- control summary: the input of calibration, only from complete, clean coverage
    control_report = control_means = None
    ctrl_parents = [p for p in in_scope if ctrl and complete(ctrl, p)]
    if ctrl and ctrl_parents == in_scope and len(in_scope) > len({parents[p]["stratum"] for p in in_scope}) \
            and not problems["*"]:
        cm = means(ctrl, ctrl_parents)
        control_report = describe(ctrl, ctrl_parents, [parents[p]["stratum"] for p in ctrl_parents], cm, data,
                                  lengths_of)
        control_means = {m: float(cm[m].mean()) for m in ("loss", "kl_rc", "kl_cr", "disagree", "dnll")}
    ref_parents = [p for p in in_scope if self_data[p]]
    ref_nll = (float(np.mean([np.concatenate([x["nll"] for x in self_data[p]]).mean() for p in ref_parents]))
               if ref_parents else None)
    ref_nll = ref_nll if ref_nll is not None and np.isfinite(ref_nll) else None
    effective = {}
    for d in dirs:
        effective.update(d.effective)
    effective_damaged = sum(d.effective_damaged for d in dirs)
    context = {"kit": VERSION, "kit_commit": kit_commit(), "split": plan["split"],
               "qualification": plan["split"] == "holdout", "plan_sha256": plan_digest,
               "plan_frozen_utc": plan.get("frozen_utc"), "manifest_sha256": manifest_digest,
               "lengths": plan["lengths"], "alpha": plan["alpha"], "k": k, "level": level,
               "reference": ref, "control": ctrl,
               "reference_def": arm_defs.get(ref) if arm_defs else None,
               "control_def": arm_defs.get(ctrl) if arm_defs and ctrl else None,
               "binary_sha256": [identity["binary_sha256"]] if identity else [],
               "arms_sha256": [identity["arms_sha256"]] if identity else [],
               "model_receipt_sha256": [identity["model_receipt_sha256"]] if identity else [],
               "run_dirs": len(dirs), "parents_in_scope": len(in_scope),
               "parents_per_stratum": dict(sorted(per_stratum.items())), "reference_mean_nll": ref_nll,
               "route_lines": sorted(route_lines), "effective_config": effective,
               "stratum_weights": plan.get("stratum_weights"),
               "effective_config_damaged_lines": effective_damaged,
               "global_problems": [f"{v}: {t}" for v, t in problems["*"]]}
    if arm_defs and evidence.has_model_overrides(arm_defs):
        context["arm_model_receipt_sha256"] = identity["arm_model_receipt_sha256"] if identity else {}
        if (plan["split"] == "search" and control_means is not None
                and not any(problems.values())
                and all(complete(arm, pid) for arm in arms for pid in in_scope)):
            context["replay_policy"] = plan_mod.REPLAY_POLICY
    return context, results, control_report, control_means


def kept_witness(run_dir, cid, lab, rec, spec):
    """Why the kept run ``rec`` of (case, label) does not carry its arm's witness (None when it does or none is
    declared), re-checked on its saved stdout/stderr logs: the exact bytes the run record binds, decoded as at run
    time. A run record without log hashes (older kit) cannot show a witness."""
    if not spec:
        return None
    if not isinstance(rec.get("stdout_sha256"), str) or not isinstance(rec.get("stderr_sha256"), str):
        return "run record does not bind its logs"
    stem = Path(run_dir) / "log" / f"{cid}.{lab}"
    try:
        out = (stem.parent / (stem.name + ".stdout")).read_bytes()
        err = (stem.parent / (stem.name + ".stderr")).read_bytes()
    except OSError as e:
        return f"logs unreadable ({type(e).__name__})"
    if sha(out) != rec["stdout_sha256"] or sha(err) != rec["stderr_sha256"]:
        return "logs are not the ones the run recorded"
    from .run import parse_result
    return witness_problem(spec, parse_result(decode_log(out)), decode_log(err))


def evaluate(specs, guard_specs, opt, ctrl, pp, strata, level, idx, weights=None):
    """One margin set: (verdict, notes, endpoints, guards) for an option's per-parent means."""
    verdict, notes, endpoints, guards = "PASS", [], {}, {}
    for e, spec in specs.items():
        d = endpoint_values(e, opt, ctrl, pp, spec)
        bound = upper_bound(d, strata, level, idx, weights)
        try:
            m = plan_mod.margin(spec)
        except ValueError:
            m = None
        available = bool(np.isfinite(bound["ucb"]))
        ok = m is not None and available and bound["ucb"] < m
        label = plan_mod.ENDPOINTS[e]["label"]
        if "control_clip" in spec:
            label += f", control clipped to +-{spec['control_clip']:g} per case"
        endpoints[e] = dict(bound, margin=m, passed=bool(ok), label=label)
        if m is None:
            verdict = _worst(verdict, "INCOMPLETE")
            notes.append(f"INCOMPLETE: {e} margin unresolved (calibrate on SEARCH)")
        elif not available:
            # No bound exists (e.g. a stratum with fewer than two usable parents under equal stratum
            # weights): missing evidence, never a failed test.
            verdict = _worst(verdict, "INCOMPLETE")
            notes.append(f"INCOMPLETE: {e} has no upper bound (a stratum lacks two usable parents)")
        elif not ok:
            verdict = _worst(verdict, "FAIL")
    for g, spec in guard_specs.items():
        e = spec["endpoint"]
        if e not in specs or endpoints[e]["margin"] is None:
            verdict = _worst(verdict, "INCOMPLETE")
            notes.append(f"INCOMPLETE: guard {g} has no resolved {e} margin")
            continue
        d = endpoint_values(e, opt, ctrl, pp, specs[e])
        limit = spec["max_stratum_point"] * endpoints[e]["margin"]
        points = {s: float(np.mean([d[i] for i in range(len(strata)) if strata[i] == s]))
                  for s in sorted(set(strata))}
        guards[g] = {"endpoint": e, "stratum_points": points, "limit": limit,
                     "passed": all(v < limit for v in points.values())}
        if not guards[g]["passed"]:
            verdict = _worst(verdict, "FAIL")
    return verdict, notes, endpoints, guards


def endpoint_values(e, opt, ctrl, per_parent, margin_spec=None):
    spec = plan_mod.ENDPOINTS[e]
    clip = (margin_spec or {}).get("control_clip")
    if clip is not None:
        # E7 per parent: mean over its cases of (option dNLL - control dNLL clipped to +-clip).
        out = []
        for (o_ids, o), (c_ids, c) in zip(per_parent[opt]["_case_dnll"], per_parent[ctrl]["_case_dnll"]):
            if o_ids != c_ids:
                raise ValueError("option and control cases do not pair up")
            out.append(float(np.mean(o - np.clip(c, -clip, clip))))
        return np.array(out)
    v = per_parent[opt][spec["metric"]]
    return v - per_parent[ctrl][spec["metric"]] if spec["kind"] == "vs_control" else v


def describe(arm, usable, strata, pp, data, lengths_of):
    allrows = [x for p in usable for x in data[arm][p]]
    cat = {k: np.concatenate([x[k] for x in allrows]) for k in ("overlap", "kl_rc", "kl_cr", "agree", "gap_r")}
    clear = cat["gap_r"] >= NEAR_TIE
    by_stratum = {}
    for s in sorted(set(strata)):
        sel = [i for i, x in enumerate(strata) if x == s]
        by_stratum[s] = {m: float(pp[m][sel].mean()) for m in ("dnll", "loss", "kl_rc", "disagree")}
    by_length = {}
    for length, rows in sorted(lengths_of[arm].items()):
        d = np.concatenate([x["nll_c"] - x["nll_r"] for x in rows])
        by_length[str(length)] = {"positions": int(d.size), "dnll": float(d.mean()),
                                  "loss": float(np.concatenate([1 - x["overlap"] for x in rows]).mean()),
                                  "kl_rc": float(np.concatenate([x["kl_rc"] for x in rows]).mean())}
    order = np.argsort(pp["dnll"])
    return {"positions": int(cat["overlap"].size), "parents": len(usable),
            "mean_parent": {m: float(v.mean()) for m, v in pp.items() if not m.startswith("_")},
            "ppl_change_percent": float(np.expm1(pp["dnll"].mean()) * 100.0),
            "overlap_position_mean": float(cat["overlap"].mean()),
            "overlap_q01": float(np.quantile(cat["overlap"], 0.01)),
            "overlap_q001": float(np.quantile(cat["overlap"], 0.001)),
            "kl_rc_max": float(cat["kl_rc"].max()), "kl_cr_mean": float(cat["kl_cr"].mean()),
            "agree_position": float(cat["agree"].mean()),
            "agree_clear_positions": float(cat["agree"][clear].mean()) if clear.any() else None,
            "near_tie_fraction": float(1 - clear.mean()),
            "worst_parent": {"parent": usable[order[-1]], "dnll": float(pp["dnll"][order[-1]])},
            "best_parent": {"parent": usable[order[0]], "dnll": float(pp["dnll"][order[0]])},
            "by_stratum": by_stratum, "by_length": by_length}


def finite(value):
    """JSON-safe copy: non-finite floats become strings ('inf', '-inf', 'nan') so a receipt always writes."""
    if isinstance(value, float) and not np.isfinite(value):
        return "nan" if np.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return value


def build_summary(context, results, control_report, control_means):
    """Build the canonical summary in memory, without writing receipts or mutating inputs."""
    context, results = finite(context), finite(results)
    replay_policy = context.pop("replay_policy", None)
    prefix = "" if context["qualification"] else "SEARCH-"
    rows = [{"option": name, "verdict": prefix + r["verdict"]} for name, r in results.items()]
    summary = dict(context, verdicts=rows, control=context["control"], control_means=finite(control_means),
                   control_report=finite(control_report))
    if replay_policy is not None:
        summary["replay_policy"] = replay_policy
    summary["self_sha256"] = sha(canonical(summary))
    return summary


def write(receipts, context, results, control_report, control_means):
    summary = build_summary(context, results, control_report, control_means)
    context, results = finite(context), finite(results)
    context.pop("replay_policy", None)
    out = Path(receipts)
    out.mkdir(parents=True, exist_ok=True)
    prefix = "" if context["qualification"] else "SEARCH-"
    for name, r in results.items():
        receipt = dict(context, option=name, verdict=prefix + r["verdict"], **{k: v for k, v in r.items()
                                                                               if k != "verdict"})
        if r.get("secondary"):
            receipt["secondary"] = dict(r["secondary"], verdict=prefix + r["secondary"]["verdict"])
        atomic(out / f"{name}.receipt.json", canonical(receipt))
        atomic(out / f"{name}.receipt.md", render_md(receipt).encode())
    atomic(out / "summary.json", canonical(summary))
    atomic(out / "summary.md", render_summary_md(summary, results).encode())
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--receipts", required=True)
    a = p.parse_args(argv)
    context, results, control_report, control_means = analyze(a.manifest, a.plan, a.runs.split(","))
    summary = write(a.receipts, context, results, control_report, control_means)
    print(json.dumps({"verdicts": summary["verdicts"], "control_means": control_means}, indent=1))


if __name__ == "__main__":
    main()
