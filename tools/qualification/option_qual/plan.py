"""Qualification plan: what is compared, on which split, against which declared margins.

usage:
  python3 -m tools.qualification.option_qual.plan init --manifest DIR --split search|holdout \
      --reference REF --control CTRL --options A,B --composition ALL=A+B --lengths 8192,32768 --out PLAN.json
  python3 -m tools.qualification.option_qual.plan calibrate PLAN.json --search-summary SUMMARY.json --out NEW.json
  python3 -m tools.qualification.option_qual.plan freeze PLAN.json
  python3 -m tools.qualification.option_qual.plan show PLAN.json

Margins are part of the plan. A rule margin (``max(abs, frac x control mean on SEARCH)``) must be
resolved by ``calibrate`` from SEARCH results before a HOLDOUT plan can be frozen. A frozen plan is
immutable: its sha256 is stamped into every run, and analysis refuses runs stamped with another sha.
"""
import argparse
import copy
import re
import datetime
import json
import math
from pathlib import Path

from .common import captured_text
from .common import VERSION, atomic, canonical, sha
from . import manifest as manifest_mod

# Endpoint quantities, per parent: mean over that parent's scored positions in scope.
#   kind "vs_ref":     option metric (needs no control)
#   kind "vs_control": option metric minus control metric
ENDPOINTS = {
    "E1": {"label": "real-text NLL change vs reference (nats/token)", "metric": "dnll", "kind": "vs_ref"},
    "E2": {"label": "overlap loss excess over control", "metric": "loss", "kind": "vs_control"},
    "E3": {"label": "KL(reference||option) excess over control", "metric": "kl_rc", "kind": "vs_control"},
    "E4": {"label": "top-1 disagreement excess over control", "metric": "disagree", "kind": "vs_control"},
    "E5": {"label": "KL(reference||option) (nats/token)", "metric": "kl_rc", "kind": "vs_ref"},
    "E6": {"label": "top-1 disagreement with the reference", "metric": "disagree", "kind": "vs_ref"},
    "E7": {"label": "real-text NLL change excess over control (nats/token)", "metric": "dnll", "kind": "vs_control"},
}
# Margin sets. "exact-class" is the default for an individual option. "fast-parity" bounds a composed
# fast mode in absolute terms (NLL change, KL and top-1 disagreement against the reference itself).
TIERS = {
    # With a control, exact-class decides the NLL change against the rounding-equivalent control (E7):
    # a pure rounding change already moves chat NLL at 8K/32K, so the absolute change (E1, reported by
    # "reference-nll") cannot tell options apart. Without a control it falls back to E1.
    "exact-class": {
        "E7": {"fixed": 0.005},
        "E1": {"fixed": 0.005},
        "E2": {"rule": {"abs": 0.002, "frac": 0.5, "of": "loss"}},
        "E3": {"rule": {"abs": 5e-4, "frac": 0.5, "of": "kl_rc"}},
        "E4": {"fixed": 0.005},
    },
    "fast-parity": {
        "E1": {"fixed": 0.0129},
        "E5": {"fixed": 0.0400},
        "E6": {"fixed": 0.03},
    },
    # Long contexts (every length >= LONG_MIN): any rounding change moves single-document NLL both ways,
    # so the NLL margin is wide and the decision rests on the full-distribution endpoints.
    "reference-nll": {
        "E1": {"fixed": 0.005},
    },
    "exact-class-long": {
        "E1": {"fixed": 0.05},
        "E2": {"rule": {"abs": 0.002, "frac": 0.5, "of": "loss"}},
        "E3": {"rule": {"abs": 5e-4, "frac": 0.5, "of": "kl_rc"}},
        "E4": {"fixed": 0.005},
    },
    "fast-parity-long": {
        "E1": {"fixed": 0.05},
        "E5": {"fixed": 0.0400},
        "E6": {"fixed": 0.03},
    },
}
LONG_MIN = 131072
TIER_STATEMENTS = {
    "reference-nll": ("absolute NLL change vs the reference; at 8K/32K chat text is rounding-sensitive "
                      "(a pure rounding control moves it too)"),
    "exact-class-long": "rounding-sensitive regime; non-inferior to a rounding-equivalent control",
    "fast-parity-long": "rounding-sensitive regime; non-inferior to a rounding-equivalent control",
}
# A summary may attest this policy only after validated, complete replay evidence.
REPLAY_POLICY = "verified-arm-model-replay-v1"
# Only records issued after verification of the actual captured buffers authorize margins.
VERIFICATION_POLICY = "single-capture-arm-model-replay-v1"
RULE_METRICS = ("loss", "kl_rc", "kl_cr", "disagree", "dnll")
DEFAULT_MARGINS = {**TIERS["exact-class"], **{e: m for e, m in TIERS["fast-parity"].items() if e != "E1"}}


def _tier(name, endpoints, control):
    """Endpoints (with margins) and guards of a margin set; ``endpoints`` narrows an exact-class plan."""
    if name not in TIERS:
        raise ValueError(f"unknown tier {name}")
    if name.startswith("exact-class"):
        nll = "E7" if name == "exact-class" else "E1"
        chosen = list(endpoints or ([nll, "E2", "E3", "E4"] if control else ["E1"]))
        specs = {}
        for e in chosen:
            if e not in ENDPOINTS:
                raise ValueError(f"unknown endpoint {e}")
            specs[e] = copy.deepcopy(TIERS[name].get(e, DEFAULT_MARGINS[e]))
    else:
        if endpoints:
            raise ValueError(f"tier {name} fixes its own endpoints")
        specs = copy.deepcopy(TIERS[name])
    for e in specs:
        if ENDPOINTS[e]["kind"] == "vs_control" and not control:
            raise ValueError(f"{e} needs a control arm")
    nll = "E7" if "E7" in specs else ("E1" if "E1" in specs else None)
    guards = {"G1": {"endpoint": nll, "max_stratum_point": 2.0, "unit": "x margin"}} if nll else {}
    return specs, guards


def init(manifest_dir, split, reference, control, options, compositions, lengths, alpha=0.05,
         bootstrap=20000, seed=20261027, min_parents=None, repeat_cases=(), endpoints=None, family=None,
         routes=None, tier="exact-class", secondary_tier=None, e7_control_clip=None, stratum_weights=None,
         witness=None):
    """``family``: names of every option/composition of the release family when they are qualified in
    separate campaigns (separate binaries); the simultaneous level then uses the whole family.
    ``routes``: {arm: regular expression} for arms that intentionally take another route than the
    reference; each of that arm's route lines must match it. Other arms must match the reference.
    ``tier``: the margin set that decides the verdict; ``secondary_tier``: another margin set evaluated on
    the same runs and reported beside it, never deciding the verdict. ``e7_control_clip``: E7 per case is
    the option's NLL change minus the control's clipped to +-this value (nats/token), so one case where
    the control itself derails can neither excuse nor penalise the option by more than that.
    ``stratum_weights="equal"``: every endpoint is the mean of the stratum means."""
    if split not in manifest_mod.SPLITS:
        raise ValueError("split must be search or holdout")
    _, digest = manifest_mod.load(manifest_dir)
    specs, guards = _tier(tier, endpoints, control)
    secondary = None
    if secondary_tier:
        if secondary_tier == tier:
            raise ValueError("the secondary tier must differ from the deciding tier")
        s_specs, s_guards = _tier(secondary_tier, None, control)
        secondary = {"tier": secondary_tier, "endpoints": s_specs, "guards": s_guards}
    if e7_control_clip is not None:
        sets = [specs] + ([secondary["endpoints"]] if secondary else [])
        if not any("E7" in x for x in sets):
            raise ValueError("--e7-control-clip needs E7 in a margin set")
        for x in sets:
            if "E7" in x:
                x["E7"]["control_clip"] = e7_control_clip
    names = [reference] + ([control] if control else []) + list(options) + [c["name"] for c in compositions]
    if len(set(names)) != len(names):
        raise ValueError("arm names must be distinct")
    if any(not n or "." in n or "/" in n or "@" in n for n in names):
        raise ValueError("arm names must be nonempty and contain no '.', '/' or '@'")
    for c in compositions:
        if not set(c["members"]) <= set(options) or len(c["members"]) < 2:
            raise ValueError(f"composition {c['name']} must combine at least two declared options")
    qualified = list(options) + [c["name"] for c in compositions]
    routes = dict(routes or {})
    for arm, pattern in routes.items():
        if arm not in names:
            raise ValueError(f"route declared for unknown arm {arm}")
        re.compile(pattern)
    family = sorted(set(family or []) | set(qualified))
    if min_parents is not None and min_parents < 2:
        raise ValueError("at least two parents per stratum are needed for a bound")
    plan = {"schema": 1, "kit": VERSION, "manifest_sha256": digest, "split": split, "family": family, "routes": routes,
            "lengths": sorted(int(x) for x in lengths), "reference": reference, "control": control or None,
            "options": list(options), "compositions": list(compositions), "ref_repeat_cases": list(repeat_cases),
            "alpha": alpha, "bootstrap": {"B": bootstrap, "seed": seed},
            "min_parents_per_stratum": min_parents if min_parents is not None else (12 if split == "holdout" else 4),
            "tier": tier, "endpoints": specs, "guards": guards,
            **({"secondary": secondary} if secondary else {}),
            **({"stratum_weights": stratum_weights} if stratum_weights else {}),
            **({"witness": witness} if witness is not None else {}),
            "frozen": False}
    problem = check(plan)
    if problem:
        raise ValueError(problem)
    return plan


def margin_sets(plan):
    """(name, endpoints, guards) of the deciding margin set and of the secondary one, if any."""
    sets = [("primary", plan["endpoints"], plan.get("guards", {}))]
    if plan.get("secondary"):
        sets.append(("secondary", plan["secondary"]["endpoints"], plan["secondary"].get("guards", {})))
    return sets


def _number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def margin(spec):
    if "fixed" not in spec:
        raise ValueError("margin rule not resolved; run plan calibrate on SEARCH results")
    if not (_number(spec["fixed"]) and spec["fixed"] > 0):
        raise ValueError(f"margin {spec['fixed']!r} is not a positive finite number")
    return float(spec["fixed"])


def _spec_problem(e, spec):
    """Every part of a margin that is present must be well formed (a resolved rule stays checked)."""
    if not isinstance(spec, dict) or not ("fixed" in spec or "rule" in spec):
        return f"{e}: a margin is {{'fixed': number}} or {{'rule': ...}}"
    if "fixed" in spec and not (_number(spec["fixed"]) and spec["fixed"] > 0):
        return f"{e}: margin {spec['fixed']!r} is not a positive finite number"
    if "rule" in spec:
        r = spec["rule"]
        if not (isinstance(r, dict) and _number(r.get("abs")) and r["abs"] > 0 and _number(r.get("frac"))
                and r["frac"] >= 0 and r.get("of") in RULE_METRICS):
            return f"{e}: malformed margin rule {r!r}"
    if "calibration" in spec and not (isinstance(spec["calibration"], dict)
                                      and _number(spec["calibration"].get("control_mean"))):
        return f"{e}: malformed calibration record"
    if "control_clip" in spec and (e != "E7" or not (_number(spec["control_clip"]) and spec["control_clip"] > 0)):
        return f"{e}: control_clip is a positive finite number and applies to E7 only"
    return None


# Label semantics apply only at the paths owned by each document schema.
_LABEL_MAPS = {
    "summary": {"parents_per_stratum": None, "effective_config": None},
    "source": {},
    "plan": {"routes": None, "witness": "witness", "endpoints": None, "guards": None},
    "margin_set": {"endpoints": None, "guards": None},
    "result": {"parents_used_per_stratum": None, "endpoints": None, "guards": "guard"},
    "secondary_result": {"endpoints": None, "guards": "guard"},
    "report": {"by_stratum": None, "by_length": None},
    "guard": {"stratum_points": None},
}
_SCHEMA_CHILDREN = {
    "summary": {"control_report": "report"},
    "source": {"search_summary": "summary"},
    "plan": {"calibration_source": "source", "secondary": "margin_set"},
    "result": {"report": "report", "secondary": "secondary_result"},
}


def has_model_overrides(value, schema="summary"):
    """Inspect provenance fields using the caller's schema; unknown wrappers stay generic."""
    def walk(value, state):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"model", "model_receipt", "arm_model_receipt_sha256"}:
                    return True
                if key in {"env", "argv"} or (state == "witness" and key == "stdout"):
                    # Expected stdout is arbitrary result data, including nested objects.
                    continue
                maps = _LABEL_MAPS.get(state, {})
                if key in maps:
                    if isinstance(child, dict):
                        if any(walk(v, maps[key]) for v in child.values()):
                            return True
                        continue
                    # Malformed label maps have no label exemption.
                if walk(child, _SCHEMA_CHILDREN.get(state, {}).get(key)):
                    return True
            return False
        if isinstance(value, list):
            return any(walk(v, None) for v in value)
        return False
    return walk(value, schema)


def _replay_policy_problem(summary, schema="summary"):
    if has_model_overrides(summary, schema) and summary.get("replay_policy") != REPLAY_POLICY:
        return ("override-bearing SEARCH summary lacks the verified replay policy attestation; "
                "regenerate from validated replay evidence")
    return None


def _verification_problem(source):
    if not has_model_overrides(source, "source"):
        return None
    # Whole-plan rewriting and rehashing is outside this boundary: the frozen plan SHA is published
    # before data collection. This record binds the verification performed during calibration.
    v = source.get("replay_verification")
    digest = lambda x: isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x) is not None
    if not isinstance(v, dict) or v.get("policy") != VERIFICATION_POLICY or not digest(v.get("search_plan_sha256")):
        return "override-bearing calibration source lacks well-formed replay_verification; recalibrate with evidence"
    ledgers = v.get("runs_ledger_sha256")
    if not isinstance(ledgers, dict) or not ledgers or not all(
            isinstance(k, str) and k.endswith("/runs.jsonl") and not k.startswith("/")
            and "\\" not in k and all(p not in {"", ".", ".."} for p in k.split("/")) and digest(h)
            for k, h in ledgers.items()):
        return "calibration source has malformed replay_verification runs ledger"
    body = source.get("search_summary", source)
    if body.get("plan_sha256") != v["search_plan_sha256"]:
        return "calibration source replay_verification SEARCH plan does not match"
    return None


def _verify_replay(summary, evidence):
    """Evidence is {manifest: directory, search_plan: path, runs: [directory, ...]}."""
    from . import analyze
    import os
    if not isinstance(evidence, dict) or not all(evidence.get(k) for k in ("manifest", "search_plan", "runs")):
        raise ValueError("override-bearing SEARCH calibration requires replay evidence (manifest, search_plan, runs)")
    dirs = [Path(d).resolve() for d in evidence["runs"]]
    if not dirs or len(set(dirs)) != len(dirs):
        raise ValueError("replay evidence requires distinct run directories")
    root = Path(os.path.commonpath([str(d.parent) for d in dirs]))
    def ledgers():
        return {d.relative_to(root).as_posix() + "/runs.jsonl": sha((d / "runs.jsonl").read_bytes()) for d in dirs}
    before = ledgers()
    _, plan_sha = load(evidence["search_plan"])
    captured = {}
    rebuilt = analyze.build_summary(*analyze.analyze(evidence["manifest"], evidence["search_plan"], dirs,
                                                  _captured=captured))
    consumed = {d.relative_to(root).as_posix() + "/runs.jsonl": captured["ledgers"][d] for d in dirs}
    body = lambda s: {k: v for k, v in s.items() if k != "self_sha256"}
    if canonical(body(rebuilt)) != canonical(body(summary)):
        raise ValueError("SEARCH summary differs from current replay evidence analysis")
    if rebuilt.get("replay_policy") != REPLAY_POLICY or rebuilt.get("plan_sha256") != plan_sha:
        raise ValueError("SEARCH replay evidence is not verified")
    after = ledgers()
    _, after_plan_sha = load(evidence["search_plan"])
    if (before != consumed or consumed != after or captured["plan_sha256"] != plan_sha
            or after_plan_sha != plan_sha):
        raise ValueError("SEARCH replay evidence changed during verification")
    return {"policy": VERIFICATION_POLICY, "search_plan_sha256": plan_sha, "runs_ledger_sha256": consumed}


def _legacy_source_problem(source):
    problem = _replay_policy_problem(source, "source") or _verification_problem(source)
    if problem:
        return problem
    for key in ("binary_sha256", "model_receipt_sha256", "search_summary_sha256"):
        if not (isinstance(source.get(key), str) and source[key]):
            return f"calibration source lacks {key}"
    for key in ("reference_def", "control_def"):
        if not isinstance(source.get(key), dict):
            return f"calibration source lacks {key}"
    return None


def _scope_problem(plan):
    """The SEARCH summary a plan was calibrated from must cover this very plan: manifest, split, lengths,
    reference and control (the checks calibrate made, repeated whenever the plan is used)."""
    body = plan["calibration_source"].get("search_summary")
    if body is None:
        return None          # earlier source form (legacy SEARCH only): nothing embedded to compare
    if body.get("split") != "search":
        return "calibration source: the summary is not a SEARCH summary"
    if body.get("manifest_sha256") != plan.get("manifest_sha256"):
        return "calibration source: the SEARCH summary comes from another manifest"
    if sorted(body.get("lengths", [])) != sorted(plan.get("lengths", [])):
        return "calibration source: the SEARCH summary covers other lengths than this plan"
    if body.get("reference") != plan.get("reference") or body.get("control") != plan.get("control"):
        return "calibration source: the SEARCH summary used another reference or control arm"
    return None


def _source_problem(source, legacy=False):
    """The calibration source carries the whole SEARCH summary it was made from; every bound value is
    taken from that body, whose canonical SHA-256 must be the recorded one. ``legacy``: a tierless
    SEARCH plan may keep the earlier form without the embedded summary."""
    if not isinstance(source, dict):
        return "calibration source is not a record"
    if legacy and "search_summary" not in source:
        return _legacy_source_problem(source)
    body = source.get("search_summary")
    if not isinstance(body, dict) or not (isinstance(source.get("search_summary_sha256"), str)
                                          and source["search_summary_sha256"]):
        return "calibration source lacks its SEARCH summary"
    if sha(canonical(body)) != source["search_summary_sha256"]:
        return "calibration source: the SEARCH summary does not match its recorded SHA-256"
    problem = _replay_policy_problem(body) or _verification_problem(source)
    if problem:
        return "calibration source: " + problem
    derived = {"binary_sha256": (body.get("binary_sha256") or [None])[0],
               "model_receipt_sha256": (body.get("model_receipt_sha256") or [None])[0],
               "reference_def": body.get("reference_def"), "control_def": body.get("control_def")}
    for key, value in derived.items():
        if value is None or source.get(key) != value:
            return f"calibration source: {key} is not the SEARCH summary's"
    if not isinstance(body.get("control_means"), dict):
        return "calibration source: the SEARCH summary has no control means"
    return None


def resolved_problem(plan):
    """Why a plan's margins cannot decide anything yet (None when every margin of every set is a finite
    number and every calibrated margin is bound to the plan's calibration source and agrees with it).
    One rule for freeze, HOLDOUT admission and HOLDOUT analysis."""
    problem = check(plan)
    if problem:
        return problem
    source = plan.get("calibration_source")
    for _, specs, _ in margin_sets(plan):
        for e, spec in specs.items():
            if "fixed" not in spec:
                return f"{e} margin not resolved (run plan calibrate on SEARCH results)"
            if "rule" in spec or "calibration" in spec:
                if source is None:
                    return f"{e} margin is calibrated but the plan has no calibration source"
                cal = spec.get("calibration")
                if not cal or "rule" not in spec:
                    return f"{e} margin lacks its rule or calibration record"
                if cal.get("search_summary_sha256") != source["search_summary_sha256"]:
                    return f"{e} margin was calibrated from another SEARCH summary"
                r = spec["rule"]
                if cal.get("metric") != r["of"]:
                    return f"{e} rule metric {r['of']!r} is not the metric it was calibrated on ({cal.get('metric')!r})"
                means = source["search_summary"]["control_means"]
                if not (_number(means.get(r["of"])) and float(means[r["of"]]) == float(cal["control_mean"])):
                    return f"{e} calibration mean does not match the SEARCH summary's control mean of {r['of']}"
                expected = max(float(r["abs"]), float(r["frac"]) * float(cal["control_mean"]))
                if not math.isclose(float(spec["fixed"]), expected, rel_tol=1e-12, abs_tol=0.0):
                    return f"{e} margin {spec['fixed']} does not follow from its rule and calibration"
    return None


def unresolved(plan):
    """Why a plan cannot decide a HOLDOUT (None when frozen and resolved_problem finds nothing)."""
    problem = resolved_problem(plan)
    if problem:
        return problem
    if not plan.get("frozen"):
        return "plan not frozen"
    return None


def calibrate(plan, search_summary, evidence=None):
    """Resolve rule margins from the control's SEARCH means (``summary.json`` of a SEARCH analysis)."""
    if plan.get("frozen"):
        raise ValueError("cannot calibrate a frozen plan")
    problem = check(plan)
    if problem:
        raise ValueError(problem)
    if search_summary.get("split") != "search":
        raise ValueError("calibration must use SEARCH results")
    problem = _replay_policy_problem(search_summary)
    if problem:
        raise ValueError(problem)
    means = search_summary.get("control_means")
    if not means:
        raise ValueError("SEARCH summary has no control means")
    if search_summary.get("manifest_sha256") != plan["manifest_sha256"]:
        raise ValueError("SEARCH summary comes from another manifest")
    if search_summary.get("control") != plan.get("control") or search_summary.get("reference") != plan["reference"]:
        raise ValueError("SEARCH summary used another reference or control arm")
    for key in ("binary_sha256", "model_receipt_sha256"):
        if len(search_summary.get(key, [])) != 1:
            raise ValueError(f"SEARCH summary must come from exactly one {key}")
    if search_summary.get("reference_def") is None or search_summary.get("control_def") is None:
        raise ValueError("SEARCH summary lacks the reference/control arm definitions")
    if sorted(search_summary.get("lengths", [])) != sorted(plan["lengths"]):
        raise ValueError("SEARCH summary covers other lengths")
    if search_summary.get("global_problems"):
        raise ValueError("SEARCH summary has unresolved global problems")
    body = {k: v for k, v in search_summary.items() if k != "self_sha256"}
    if search_summary.get("self_sha256") != sha(canonical(body)):
        raise ValueError("SEARCH summary was modified after it was written")
    verification = _verify_replay(search_summary, evidence) if has_model_overrides(search_summary) else None
    new = copy.deepcopy(plan)
    # The holdout must run the very reference and control (binary, model, arm definitions) the margins
    # were calibrated with; analysis enforces this binding.
    new["calibration_source"] = {"binary_sha256": search_summary["binary_sha256"][0],
                                 "model_receipt_sha256": search_summary["model_receipt_sha256"][0],
                                 "reference_def": search_summary["reference_def"],
                                 "control_def": search_summary["control_def"],
                                 "search_summary_sha256": search_summary["self_sha256"],
                                 "search_summary": body}
    if verification is not None:
        new["calibration_source"]["replay_verification"] = verification
    for e, spec in [(e, s) for _, specs, _ in margin_sets(new) for e, s in specs.items()]:
        if "rule" in spec:
            r = spec["rule"]
            if not _number(means.get(r["of"])):
                raise ValueError(f"SEARCH summary has no finite control mean for {r['of']}")
            value = max(float(r["abs"]), float(r["frac"]) * float(means[r["of"]]))
            spec["fixed"] = value
            spec["calibration"] = {"metric": r["of"], "control_mean": float(means[r["of"]]), "search_summary_sha256":
                                   search_summary.get("self_sha256"), "search_manifest_sha256":
                                   search_summary.get("manifest_sha256")}
    return new


def _witness_problem(witness, names):
    """Structure of a plan's per-arm run witness (see README, 'Run witness')."""
    if not isinstance(witness, dict) or not witness:
        return "witness must be a non-empty object {arm: spec}"
    for arm, spec in witness.items():
        if arm not in names:
            return f"witness for unknown arm {arm!r}"
        if not isinstance(spec, dict) or not spec or set(spec) - {"stdout", "stderr_forbid", "stderr_require"}:
            return f"witness of {arm}: a non-empty object with stdout / stderr_forbid / stderr_require"
        if "stdout" in spec and (not isinstance(spec["stdout"], dict) or not spec["stdout"]
                                 or not all(isinstance(k, str) and k for k in spec["stdout"])):
            return f"witness of {arm}: stdout must be a non-empty object of result keys"
        for key in ("stderr_forbid", "stderr_require"):
            if key in spec and (not isinstance(spec[key], list) or not spec[key]
                                or not all(isinstance(x, str) and x for x in spec[key])):
                return f"witness of {arm}: {key} must be a non-empty list of strings"
        for pattern in spec.get("stderr_require", []):
            try:
                re.compile(pattern)
            except (re.error, OverflowError, RecursionError, ValueError) as e:
                return f"witness of {arm}: bad pattern {pattern!r}: {type(e).__name__}: {e}"
    return None


def check(plan):
    """Structural problems of a plan (None when sound)."""
    if not plan.get("endpoints"):
        return "a plan needs at least one gating endpoint"
    names = [plan["reference"]] + ([plan["control"]] if plan.get("control") else []) + plan["options"] + \
        [c["name"] for c in plan["compositions"]]
    if any(not n or "." in n or "/" in n or "@" in n for n in names):
        return "arm names must be nonempty and contain no '.', '/' or '@'"
    for _, specs, guards in margin_sets(plan):
        if not specs:
            return "a margin set needs at least one endpoint"
        for e, spec in specs.items():
            if e not in ENDPOINTS:
                return f"unknown endpoint {e}"
            problem = _spec_problem(e, spec)
            if problem:
                return problem
        for g, spec in guards.items():
            if spec.get("endpoint") not in specs:
                return f"guard {g} refers to endpoint {spec.get('endpoint')} that the plan does not evaluate"
            if not (_number(spec.get("max_stratum_point")) and spec["max_stratum_point"] > 0):
                return f"guard {g}: multiplier {spec.get('max_stratum_point')!r} is not a positive finite number"
    if plan["min_parents_per_stratum"] < 2:
        return "at least two parents per stratum are needed for a bound"
    if "witness" in plan:
        problem = _witness_problem(plan["witness"], names)
        if problem:
            return problem
    if any(ENDPOINTS[e]["kind"] == "vs_control" for _, specs, _ in margin_sets(plan) for e in specs) \
            and not plan.get("control"):
        return "an endpoint needs a control arm"
    # A SEARCH plan written before tiers existed (no "tier" key) is kept as it was: its primary set is not
    # held to the length rule and its calibration source may have the earlier form. Never for HOLDOUT.
    legacy = "tier" not in plan and plan.get("split") == "search"
    if "stratum_weights" in plan and plan["stratum_weights"] != "equal":
        return f"stratum_weights {plan['stratum_weights']!r} is not 'equal'"
    if "calibration_source" in plan:
        problem = _source_problem(plan["calibration_source"], legacy)
        if problem:
            return problem
        problem = _scope_problem(plan)
        if problem:
            return problem
    tiers = [plan.get("tier", "exact-class")] + ([plan["secondary"].get("tier")] if plan.get("secondary") else [])
    for t in tiers:
        if t not in TIERS:
            return f"unknown tier {t!r}"
    lengths = plan.get("lengths", [])
    long_lengths = [x for x in lengths if x >= LONG_MIN]
    checked = tiers[1:] if legacy else tiers
    if checked and long_lengths and len(long_lengths) != len(lengths):
        return f"a plan is either long ({LONG_MIN} tokens and more) or not; split the lengths into two plans"
    for t in checked:
        if bool(long_lengths) != t.endswith("-long"):
            return (f"tier {t} does not fit lengths {lengths}: lengths of {LONG_MIN} and more need a *-long tier, "
                    "and *-long tiers need only such lengths")
    return None


def freeze(plan, now=None):
    if plan.get("frozen"):
        raise ValueError("plan already frozen")
    problem = check(plan)
    if problem:
        raise ValueError(problem)
    problem = resolved_problem(plan)
    if problem:
        raise ValueError(problem)
    new = copy.deepcopy(plan)
    new["frozen"] = True
    new["frozen_utc"] = (now or datetime.datetime.now(datetime.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return new


def plan_sha(plan):
    return sha(canonical(plan))


def load(path):
    # Preserve the canonical plan identity, derived solely from this one parsed buffer.
    plan = json.loads(captured_text(Path(path).read_bytes(), encoding="utf-8"))
    return plan, plan_sha(plan)


def save(path, plan):
    atomic(path, canonical(plan))
    return plan_sha(plan)


def require_runnable(plan, manifest_digest):
    if plan["manifest_sha256"] != manifest_digest:
        raise ValueError("plan was made for another manifest")
    problem = check(plan)
    if problem:
        raise ValueError(problem)
    if plan["split"] == "holdout":
        why = unresolved(plan)
        if why:
            raise ValueError(f"HOLDOUT runs need a frozen plan with numeric margins ({why})")


def arms_needed(plan):
    return ([plan["reference"]] + ([plan["control"]] if plan.get("control") else []) + plan["options"]
            + [c["name"] for c in plan["compositions"]])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--manifest", required=True)
    i.add_argument("--split", required=True, choices=manifest_mod.SPLITS)
    i.add_argument("--reference", required=True)
    i.add_argument("--control", default="")
    i.add_argument("--options", required=True)
    i.add_argument("--composition", action="append", default=[], help="NAME=OPT1+OPT2[+...]")
    i.add_argument("--lengths", required=True)
    i.add_argument("--endpoints", default="", help="exact-class only: a subset of E1-E7")
    i.add_argument("--tier", default="exact-class", choices=sorted(TIERS), help="margin set deciding the verdict")
    i.add_argument("--secondary-tier", default="", choices=[""] + sorted(TIERS),
                   help="margin set reported beside the verdict on the same runs")
    i.add_argument("--e7-control-clip", type=float, default=None,
                   help="E7 per case: option NLL change minus the control's clipped to +-this (nats/token)")
    i.add_argument("--stratum-weights", default="", choices=["", "equal"],
                   help="equal: every endpoint is the mean of the stratum means")
    i.add_argument("--alpha", type=float, default=0.05)
    i.add_argument("--bootstrap", type=int, default=20000)
    i.add_argument("--seed", type=int, default=20261027)
    i.add_argument("--min-parents-per-stratum", type=int)
    i.add_argument("--repeat-cases", default="")
    i.add_argument("--family", default="", help="all options/compositions of the release family (comma list)")
    i.add_argument("--route", action="append", default=[], help="ARM=REGEX: expected route lines of an arm")
    i.add_argument("--witness", default="", help="JSON file {arm: {stdout, stderr_forbid, stderr_require}}")
    i.add_argument("--out", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("plan")
    c.add_argument("--search-summary", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--search-runs")
    c.add_argument("--search-plan")
    c.add_argument("--manifest")
    f = sub.add_parser("freeze")
    f.add_argument("plan")
    s = sub.add_parser("show")
    s.add_argument("plan")
    a = p.parse_args(argv)
    if a.cmd == "init":
        witness = json.loads(Path(a.witness).read_text(encoding="utf-8")) if a.witness else None
        if a.witness and witness is None:
            # a given witness file is never read as "no witness"
            raise ValueError(f"witness file {a.witness} holds null; a witness is an object {{arm: spec}}")
        comps = []
        for item in a.composition:
            name, _, members = item.partition("=")
            comps.append({"name": name, "members": [m for m in members.split("+") if m]})
        plan = init(a.manifest, a.split, a.reference, a.control, [o for o in a.options.split(",") if o], comps,
                    [int(x) for x in a.lengths.split(",") if x], a.alpha, a.bootstrap, a.seed,
                    a.min_parents_per_stratum, [x for x in a.repeat_cases.split(",") if x],
                    [x for x in a.endpoints.split(",") if x] or None, [x for x in a.family.split(",") if x],
                    dict(r.split("=", 1) for r in a.route), a.tier, a.secondary_tier or None,
                    a.e7_control_clip, a.stratum_weights or None,
                    witness)
        print(save(a.out, plan))
    elif a.cmd == "calibrate":
        plan, _ = load(a.plan)
        summary = json.loads(Path(a.search_summary).read_text())
        print(save(a.out, calibrate(plan, summary, {"manifest": a.manifest,
                                                  "search_plan": a.search_plan,
                                                  "runs": a.search_runs.split(",") if a.search_runs else None})))
    elif a.cmd == "freeze":
        plan, _ = load(a.plan)
        print(save(a.plan, freeze(plan)))
    else:
        plan, digest = load(a.plan)
        print(json.dumps({"plan_sha256": digest, **plan}, indent=1))


if __name__ == "__main__":
    main()
