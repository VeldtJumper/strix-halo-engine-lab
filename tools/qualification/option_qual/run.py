"""Run the planned arms on the manifest cases (inside one GPU turn) and reduce them on the spot.

usage (from the repository root on the node, prompt IDs are passed as arguments, so the stack limit
must allow long argv, e.g. ``ulimit -s 65536``):
  python3 -m tools.qualification.option_qual.run --manifest DIR --plan PLAN.json --arms ARMS.json \
      --binary bin/strix-hip-decode-token --model MODEL.gguf --model-receipt RECEIPT --out OUT \
      --soft-seconds 430 [--cases ID,...] [--predict 8192=40,32768=70] [--keep-logits]

ARMS.json: {"arms": {NAME: {"env": {VAR: VALUE}, "argv": [extra arguments]}}}. An arm may also
set both "model" and "model_receipt" paths in --replay mode only, overriding the run defaults. Every arm, the
reference included, is teacher-forced on the case's recorded continuation; the logits of every token
selection are reduced against the reference in a separate process while the next arm runs. Nothing
outlives the soft deadline: a run or reduction still going then is stopped, and the next invocation
resumes (a finished run whose reduction was stopped is reduced, not run again). Exit 0 = everything
selected is finished, 2 = stopped by the soft budget (run again), 1 = a run failed or a result is
invalid (other cases continue; a failing pair is retried up to --max-attempts times, then counts
against its arm).
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import uuid

from .common import captured_text
from .common import (VERSION, atomic, canonical, decode_log, diagnostic_records, file_sha, kit_commit, read_id_text,
                     witness_problem, read_journal, read_journal_lines, sha)
from . import manifest as manifest_mod
from . import plan as plan_mod
from .records import STEP_HEADER
from .reduce import EXIT_HISTORY
from . import evidence

REPEAT_SUFFIX = "@repeat"
KEEP_ENV = ("PATH", "HOME", "LD_LIBRARY_PATH", "SYSTEMROOT")
DROP_PREFIX = ("STRIX_", "HIP_", "HSA_")
HOOK_ENV = ("STRIX_BITCALC_LOGITS_OUT", "STRIX_BITCALC_FORCE_TOKENS")
CONFIG_BUDGET = 90.0      # seconds per invocation for --print-effective-config captures
MAX_CAP_FRACTION = 0.9    # cap with no timing evidence yet: this share of the cap limit
START_HEADROOM = 10.0     # least start-up allowance: preflight and repair before an invocation's first run
STARTUP_MARGIN = 5.0      # seconds added to the start-up an invocation measured at its first engine admission


class BudgetStop(Exception):
    """The deadline stopped bounded work (e.g. hashing retained logits); nothing is judged, resume later."""

ROOT = Path(__file__).resolve().parents[3]     # repository root: reductions run as `python -m` from here


def clean_env(arm_env, base=None):
    base = os.environ if base is None else base
    env = {k: base[k] for k in KEEP_ENV if k in base}
    for k, v in arm_env.items():
        if not isinstance(v, str) or not k or "=" in k:
            raise ValueError(f"bad arm environment entry {k!r}")
        env[k] = v
    return env


def load_arms(path, needed, *, data=None):
    arms = json.loads(captured_text(Path(path).read_bytes() if data is None else data))["arms"]
    for name in needed:
        if name not in arms:
            raise ValueError(f"arms file lacks arm {name}")
        spec = arms[name]
        if set(spec) - {"env", "argv", "model", "model_receipt"}:
            raise ValueError(f"arm {name}: only env, argv, model and model_receipt are allowed")
        if ("model" in spec) != ("model_receipt" in spec):
            raise ValueError(f"arm {name}: model and model_receipt must be specified together")
        if set(spec.get("env", {})) & set(HOOK_ENV):
            raise ValueError(f"arm {name}: the measurement hook variables are set by the runner")
    return arms


def count_step_rows(path):
    size = Path(path).stat().st_size
    if size == 0:
        return 0
    if size < STEP_HEADER.size:
        raise ValueError("logits file shorter than one record header")
    with open(path, "rb") as f:
        vocab = STEP_HEADER.unpack(f.read(STEP_HEADER.size))[5]
    if not vocab:
        raise ValueError("logits record with an empty vocabulary")
    record = STEP_HEADER.size + 4 * vocab
    if size % record:
        raise ValueError("logits file is not a whole number of records")
    return size // record


def parse_result(stdout):
    try:
        from tools.qualification.longctx_fidelity.run_cli import parse_output
        return parse_output(stdout)
    except (ValueError, KeyError, ImportError):
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except ValueError:
                    pass
    return None


def check_forced(path, cont):
    """The logits file must follow the forced continuation: labels, consecutive steps and positions."""
    rows = list(step_headers(path))
    if [r[2] for r in rows] != list(cont):
        return "logged labels differ from the continuation (teacher forcing not applied?)"
    if [r[0] for r in rows] != list(range(len(rows))):
        return "steps are not 0..n-1"
    if any(b[1] != a[1] + 1 for a, b in zip(rows, rows[1:])):
        return "positions are not consecutive"
    return None


def step_headers(path):
    """(step, position, label) of every record, reading headers only."""
    with open(path, "rb") as f:
        while True:
            head = f.read(STEP_HEADER.size)
            if not head:
                return
            if len(head) != STEP_HEADER.size:
                raise ValueError("truncated step header")
            magic, step, position, argmax, emitted, vocab = STEP_HEADER.unpack(head)
            yield step, position, emitted
            f.seek(4 * vocab, 1)


def parse_predict(text):
    """'8192=40,32768=70' per nominal length; a bare number applies to every length (key 0)."""
    out = {}
    for item in (x.strip() for x in (text or "").split(",")):
        if not item:
            continue
        length, sep, seconds = item.partition("=")
        try:
            out[int(length) if sep else 0] = float(seconds if sep else length)
        except ValueError:
            raise ValueError(f"--predict entry {item!r}: expected LENGTH=SECONDS or SECONDS") from None
    return out


JOURNALS = ("run-meta.json", "runs.jsonl", "effective-config.jsonl", "holdout-ledger.jsonl", "resets.jsonl",
            "recovery.jsonl")
# Bad lines are kept and judged by their reader, never edited away: the run journal by the evidence
# ledger; the diagnostic journals are skipped where damaged (they never decide a result).
TOLERANT = ("runs.jsonl", "effective-config.jsonl", "holdout-ledger.jsonl", "recovery.jsonl")


def _lines(path):
    return read_journal(path)[0]


def recover_journals(out):
    """An interrupted append leaves an incomplete final record: move it aside with a recovery record
    (anything it described is produced again, since no marker can point to a lost run record)."""
    for name in JOURNALS:
        path = Path(out) / name
        tail = read_journal_lines(path)[1] if name in TOLERANT else read_journal(path)[1]
        if tail is None:
            continue
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if name in TOLERANT:
            # Append-only: the unterminated last line is kept and terminated. If it parses it is an
            # ordinary record; if not, the ledger treats it as a malformed line and resets its case.
            with open(path, "ab") as f:
                f.write(b"\n")
            with open(Path(out) / "recovery.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"journal": name, "terminated_tail_bytes": len(tail), "utc": stamp},
                                   sort_keys=True) + "\n")
            continue
        aside = Path(out) / f"{name}.incomplete-{stamp}"
        aside.write_bytes(tail)
        data = path.read_bytes()
        atomic(path, data[:len(data) - len(tail)])
        with open(Path(out) / "recovery.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"journal": name, "quarantined": aside.name, "bytes": len(tail),
                                "utc": stamp}, sort_keys=True) + "\n")


def bundle_shas(stem):
    """(npz sha, json sha) of a saved reduction, or None when either file is missing."""
    npz, js = Path(str(stem) + ".npz"), Path(str(stem) + ".json")
    return (file_sha(npz), file_sha(js)) if npz.exists() and js.exists() else None


def drop_bundle(stem):
    for suffix in (".npz", ".json", ".tmp.npz"):
        path = Path(str(stem) + suffix)
        if path.exists():
            path.unlink()


class Reductions:
    """Reductions in child processes, so a deadline can stop them; results are recorded by the parent."""

    def __init__(self, runner):
        self.r = runner
        self.jobs = []

    def submit(self, kind, cid, lab, ref_bin, cand, meta):
        stem = self.r.out / "red" / (f"{cid}.{lab}" if kind == "pair" else f"{cid}.{lab}.self")
        drop_bundle(stem)
        argv = [sys.executable, "-m", "tools.qualification.option_qual.reduce", str(ref_bin)]
        argv += [str(cand)] if kind == "pair" else ["--self"]
        argv += ["--format", "steps", "--out", str(stem)]
        err = open(self.r.out / "log" / f"reduce.{cid}.{lab}.{kind}.stderr", "wb")
        proc = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=err)
        self.jobs.append({"proc": proc, "err": err, "kind": kind, "cid": cid, "lab": lab, "stem": stem,
                          "cand": cand, "meta": meta, "start": self.r.now()})

    def wait(self, keep, deadline):
        """Wait until at most ``keep`` reductions run. False = the deadline stopped them (resumable)."""
        for job in [j for j in self.jobs if j["proc"].poll() is not None]:
            self.finish(job)
        while len(self.jobs) > keep:
            job = self.jobs[0]
            try:
                job["proc"].wait(timeout=max(0.0, deadline - self.r.now()))
            except subprocess.TimeoutExpired:
                self.stop_all()
                return False
            self.finish(job)
        return True

    def stop_all(self):
        for job in self.jobs:
            job["proc"].kill()
            job["proc"].wait()
            job["err"].close()
            drop_bundle(job["stem"])       # the run's logits stay; the next invocation reduces them
        self.jobs = []

    def finish(self, job):
        self.jobs.remove(job)
        job["err"].close()
        r, cid, lab, meta = self.r, job["cid"], job["lab"], job["meta"]
        r.reduce_cost = max(r.reduce_cost, r.now() - job["start"] + 2.0)
        rc = job["proc"].returncode
        shas = bundle_shas(job["stem"])
        if job["kind"] == "self":
            mark = r.done(cid, lab) or {}
            info = json.loads(Path(str(job["stem"]) + ".json").read_text()) if shas else {}
            if rc == 0 and shas and info.get("ref_sha256") == mark.get("sha256"):
                mark.update(self_red_sha256=shas[0], self_redj_sha256=shas[1])
                r.mark(cid, lab, mark)
            else:
                drop_bundle(job["stem"])
                r.fail(cid, lab, f"reference reduction exit {rc}", run_id=mark.get("run_id"))
            return
        cand = job["cand"]
        info = json.loads(Path(str(job["stem"]) + ".json").read_text()) if shas else {}
        if rc == 0 and shas and info.get("ref_sha256") == meta["ref_sha256"] \
                and info.get("cand_sha256") == meta["cand_sha256"]:
            r.mark(cid, lab, dict(meta, status="COMPLETE", red_sha256=shas[0], redj_sha256=shas[1],
                                  bitwise_identical=meta["cand_sha256"] == meta["ref_sha256"]))
        elif rc == EXIT_HISTORY:
            drop_bundle(job["stem"])
            r.invalid(cid, lab, "the run does not follow the reference's forced history", meta["ref_sha256"],
                      meta.get("run_id"))
        else:
            drop_bundle(job["stem"])
            r.fail(cid, lab, f"reduction exit {rc}", run_id=meta.get("run_id"))
        if not r.a.keep_logits and cand.exists():
            cand.unlink()


class Runner:
    def __init__(self, a, now=time.monotonic):
        self.a, self.now = a, now
        self.t0 = now()
        self.deadline = self.t0 + a.soft_seconds
        self.out = Path(a.out)
        for d in ("logits", "red", "done", "log"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        recover_journals(self.out)
        self.plan, self.plan_sha = plan_mod.load(a.plan)
        self.manifest, self.manifest_sha = manifest_mod.load(a.manifest)
        plan_mod.require_runnable(self.plan, self.manifest_sha)
        if self.manifest.get("fake_tokenizer") and not a.allow_fake_for_tests:
            raise ValueError("manifest was built with a fake tokenizer")
        receipt_bytes = Path(a.model_receipt).read_bytes()     # hashed and parsed from the same buffer
        receipt_sha = sha(receipt_bytes)
        if not self.manifest.get("fake_tokenizer") and \
                self.manifest["tokenizer"].get("model_receipt_sha256") != receipt_sha:
            raise ValueError("model receipt differs from the receipt the manifest was tokenized with")
        lengths = set(self.plan["lengths"])
        wanted = set(a.cases.split(",")) if a.cases else None
        self.cases = [c for c in self.manifest["cases"] if c["split"] == self.plan["split"]
                      and c["nominal_length"] in lengths and (wanted is None or c["case_id"] in wanted)]
        if wanted is not None and len(self.cases) != len(wanted):
            raise ValueError("unknown case ID or case outside the plan's split/lengths")
        manifest_mod.verify(a.manifest, case_ids=[c["case_id"] for c in self.cases])
        self.arm_names = plan_mod.arms_needed(self.plan)
        arms_bytes = Path(a.arms).read_bytes()
        self.arms = load_arms(a.arms, self.arm_names, data=arms_bytes)
        self.require_replay = evidence.has_model_overrides(self.arms)
        if self.require_replay and not a.replay:
            raise ValueError("per-arm model overrides require --replay; direct runs cannot verify arm model provenance")
        self.binary = str(Path(a.binary).resolve())
        self.meta = {"kit": VERSION, "kit_commit": kit_commit(), "plan_sha256": self.plan_sha,
                     "manifest_sha256": self.manifest_sha, "arms_sha256": sha(arms_bytes),
                     "binary_sha256": file_sha(self.binary), "model_receipt_sha256": receipt_sha,
                     "split": self.plan["split"]}
        self.models, self.model_shas, receipt_shas = {}, {}, {}
        for name in self.arm_names:
            spec = self.arms[name]
            raw = Path(spec["model_receipt"]).read_bytes() if "model_receipt" in spec else receipt_bytes
            fields = raw.decode("utf-8", "replace").split()
            self.models[name] = str(Path(spec.get("model", a.model)).resolve())
            self.model_shas[name] = fields[0] if fields else None
            receipt_shas[name] = sha(raw)
        self.meta["arm_model_receipt_sha256"] = receipt_shas
        self.predict = parse_predict(a.predict)
        self.repeats = set(self.plan.get("ref_repeat_cases", []))
        self.launcher = [sys.executable] if a.python_binary_for_tests else []
        self.replay = Path(a.replay).resolve() if a.replay else None
        self.replay_labels = dict(x.split("=", 1) for x in a.replay_label)
        if self.replay is not None:
            if not self.replay.is_dir():
                raise ValueError(f"--replay {a.replay} is not a directory")
            journal = self.replay / "runs.jsonl"
            if not journal.exists():
                raise ValueError("--replay needs the engine's runs.jsonl (one committed line per capture)")
            raw = journal.read_bytes()           # one snapshot: every pair is matched against these bytes
            self.meta["replay_dir"] = str(self.replay)
            self.meta["replay_runs_sha256"] = sha(raw)
            if not all(self.model_shas.values()):
                raise ValueError("--replay needs a model receipt that names the model (empty receipt)")
            self.replay_entries = {}
            for line in raw.decode("utf-8", "replace").splitlines():
                try:
                    e = json.loads(line)
                except ValueError:
                    continue                     # an incomplete last line is not a capture
                if isinstance(e, dict) and e.get("status") == "complete" and isinstance(e.get("capture_id"), str):
                    self.replay_entries.setdefault((e.get("case_id"), e.get("label")), []).append(e)
        self.bad = 0        # FAILED or INVALID outcomes of this invocation
        self.invocation = uuid.uuid4().hex[:16]     # unique; never derived from a recoverable journal
        self.reduce_cost = 10.0     # seconds; raised to the slowest reduction seen
        self.reductions = Reductions(self)
        self.run_seq = 0

    # ---------------------------------------------------------------- bookkeeping
    def append(self, name, record):
        with open(self.out / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def preflight(self):
        for first in _lines(self.out / "run-meta.json"):
            old_receipts = evidence.normalize_arm_receipts(
                first, self.arm_names, allow_legacy=not evidence.has_model_overrides(self.arms))
            for key in ("plan_sha256", "manifest_sha256", "split", "arms_sha256", "binary_sha256",
                        "model_receipt_sha256", "arm_model_receipt_sha256"):
                old = old_receipts if key == "arm_model_receipt_sha256" else first.get(key)
                if old != self.meta[key]:
                    raise ValueError(f"output directory was started with another {key}; use a new --out")
        for r in read_journal_lines(self.out / "runs.jsonl")[0]:
            if self.require_replay and evidence.run_ok(r, self.meta) and not evidence.replay_provenance(r):
                raise ValueError("successful run lacks replay provenance; reimport historical captures into a new --out")
            # Run records carry the identity too, so a lost metadata journal cannot hide a mismatch.
            expected = evidence.arm_identity(self.meta, r.get("arm")) if isinstance(r, dict) else {}
            for key in evidence.RUN_IDENTITY:
                if isinstance(r, dict) and isinstance(r.get(key), str) and r[key] != expected[key] and \
                        evidence.run_record(r):
                    raise ValueError(f"output directory holds runs with another {key}; use a new --out")
        copy = self.out / "arms.json"
        if not copy.exists():
            shutil.copyfile(self.a.arms, copy)
        if file_sha(copy) != self.meta["arms_sha256"]:
            raise ValueError("output directory holds another arms file")
        longest = max(self.cases, key=lambda c: c["prompt_tokens"], default=None)
        if longest is not None and self.replay is None:
            from tools.qualification.longctx_fidelity.run_cli import argv_check
            ids = read_id_text(Path(self.a.manifest) / longest["prompt_file"])
            for name in self.arm_names:
                argv_check(self.launcher + [self.binary, self.models[name], *map(str, ids), "--generate",
                                            str(longest["cont_tokens"])], clean_env({}))
        stamp = dict(self.meta, started_utc=datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"), cases=len(self.cases), invocation=self.invocation)
        if not self.require_replay:
            stamp.pop("arm_model_receipt_sha256", None)
        self.append("run-meta.json", stamp)
        if self.plan["split"] == "holdout":
            self.append("holdout-ledger.jsonl", stamp)
        self.reset_bad()

    def done(self, case, arm):
        path = self.out / "done" / f"{case}.{arm}"
        if not path.exists():
            return None
        try:
            mark = json.loads(path.read_text())
            if isinstance(mark, dict) and "status" in mark:
                return mark
        except ValueError:
            pass
        self.quarantine(path, "unreadable done marker")
        return None

    def quarantine(self, path, why):
        """Move an untrustworthy file aside (never delete evidence) and record why."""
        qdir = self.out / "quarantine"
        qdir.mkdir(exist_ok=True)
        target = qdir / f"{path.name}.{uuid.uuid4().hex[:8]}"
        os.replace(path, target)
        self.append("recovery.jsonl", {"quarantined": str(target.relative_to(self.out)), "why": why,
                                       "invocation": self.invocation})

    def status(self, case, arm):
        return (self.done(case, arm) or {}).get("status")

    def ledger(self):
        lines, _, raws = read_journal_lines(self.out / "runs.jsonl")
        return evidence.Ledger(lines, _lines(self.out / "resets.jsonl"), self.meta, raws, require_replay=self.require_replay)

    def reset_bad(self):
        """Malformed or contradictory journal lines are never edited away: the case they belong to (all
        cases when unattributable) is reset - its earlier lines stop counting, its files go to
        quarantine and it is produced again."""
        led = self.ledger()
        if led.bad_resets:
            # A reset journal that does not belong to this journal prefix/identity is set aside; the
            # bad lines it claimed to cover are reset again below with properly bound resets.
            self.quarantine(self.out / "resets.jsonl", "reset journal not bound to this run journal")
            led = self.ledger()
        bad = led.uncovered_bad()
        if not bad:
            return
        scopes = {case or "*" for _, case, _ in bad}
        scopes = {"*"} if "*" in scopes else scopes
        _, _, raws = read_journal_lines(self.out / "runs.jsonl")
        for scope in sorted(scopes):
            self.reconcile_failures(led, scope)
            led = self.ledger()
            _, _, raws = read_journal_lines(self.out / "runs.jsonl")
            self.append("resets.jsonl", {"case_id": scope, "upto": len(raws),
                                         "prefix_sha256": evidence.prefix_sha(raws, len(raws)),
                                         "invocation": self.invocation,
                                         **{k: self.meta[k] for k in evidence.RUN_IDENTITY},
                                         "why": sorted({why for _, case, why in bad if (case or "*") == scope
                                                        or scope == "*"})})
            for sub in ("done", "red", "logits"):
                for path in sorted((self.out / sub).iterdir()):
                    if scope == "*" or path.name.startswith(scope + "."):
                        self.quarantine(path, f"case reset ({scope})")

    def reconcile_failures(self, led, scope):
        """Before a reset sets markers aside: a FAILED marker naming a complete run of its pair whose
        failure is not in the journal (e.g. its event line was lost) is written back as an event, so a
        reset never erases an attempt."""
        done = self.out / "done"
        for path in sorted(done.iterdir()):
            if scope != "*" and not path.name.startswith(scope + "."):
                continue
            cid, _, label = path.name.rpartition(".")
            mark = self.done(cid, label)
            if not mark or mark.get("status") != "FAILED":
                continue
            rec = led.by_id(mark.get("run_id"))
            if rec is not None and (rec["case_id"], rec["arm"]) == (cid, label) and rec["complete"] is True \
                    and mark["run_id"] not in led.failed(cid, label):
                self.run_seq += 1
                self.append("runs.jsonl", {"event": "reduction_failed", "run_id": f"{self.invocation}-e{self.run_seq}",
                                           "of_run": mark["run_id"], "case_id": cid, "arm": label,
                                           "error": "reconstructed from its failure marker before a reset",
                                           "reconstructed": True,
                                           **{k: evidence.arm_identity(self.meta, label)[k] for k in evidence.RUN_IDENTITY}})

    def attempts(self, case, arm):
        """Failures of a pair, rebuilt from the run journal (evidence.Ledger.failed), never from a marker."""
        return len(self.ledger().failed(case, arm))

    def terminal(self, case, arm):
        return self.status(case, arm) == "FAILED" and self.attempts(case, arm) >= self.a.max_attempts

    def finished(self, case, arm):
        """Nothing left to run for this pair: reduced, broken history, or failed for good."""
        return self.status(case, arm) in ("COMPLETE", "INVALID") or self.terminal(case, arm)

    def mark(self, case, arm, status):
        atomic(self.out / "done" / f"{case}.{arm}", canonical(dict(status, plan_sha256=self.plan_sha)))

    def unmark(self, case, arm):
        path = self.out / "done" / f"{case}.{arm}"
        if path.exists():
            path.unlink()

    def fail(self, case, arm, why, rec=None, run_id=None):
        """A failure is bound to the run record it concerns (the failed run, or the run whose
        reduction failed)."""
        run_id = rec["run_id"] if rec else run_id
        if rec is None and run_id:
            # A reduction failure of a complete run: recorded in the run journal itself.
            self.run_seq += 1
            self.append("runs.jsonl", {"event": "reduction_failed", "run_id": f"{self.invocation}-e{self.run_seq}",
                                       "of_run": run_id, "case_id": case, "arm": arm, "error": why,
                                       **{k: evidence.arm_identity(self.meta, arm)[k] for k in evidence.RUN_IDENTITY}})
        self.mark(case, arm, {"status": "FAILED", "attempts": self.attempts(case, arm),
                              "max_attempts": self.a.max_attempts, "error": why, "run_id": run_id})
        self.bad += 1

    def invalid(self, case, arm, why, ref_sha, run_id):
        self.mark(case, arm, {"status": "INVALID", "error": why, "ref_sha256": ref_sha, "run_id": run_id})
        self.bad += 1

    def walls(self, length, arm=None):
        return self.ledger().walls(length, arm)

    def predicted(self, length):
        return max([self.predict.get(length, self.predict.get(0, 0.0))] + self.walls(length))

    def remaining(self):
        return self.deadline - self.now()

    def cap(self, length):
        """Execution cap of one run: the larger of --cap-factor x and --cap-floor seconds above the longest
        completed run at this length (or the --predict value); with no evidence yet, the most any run
        may get: a fixed fraction of the soft budget less the reserve, which fits an invocation that
        has just started. A cap never depends on what ran earlier in the invocation."""
        limit = self.cap_limit()
        p = self.predicted(length)
        if p <= 0:
            return MAX_CAP_FRACTION * limit
        return min(limit, max(self.a.cap_factor * p, p + self.a.cap_floor))

    def cap_limit(self):
        """The longest execution a fresh invocation can admit: soft budget less reserve, post-processing
        and the start-up headroom an invocation spends before its first run."""
        return self.a.soft_seconds - self.a.reserve_seconds - self.post_cost() - self.startup_allowance()

    def startup_allowance(self):
        """Time this invocation spent before its first engine admission (preflight, repair, waiting for
        reductions, any effective-config capture) plus the reductions still pending then and a small
        margin; at least START_HEADROOM. Frozen at that first admission, so every run of the invocation
        gets the same cap and the first one always fits when it is feasible at all."""
        if getattr(self, "_startup", None) is None:
            return max(START_HEADROOM, self.now() - self.t0 + len(self.reductions.jobs) * self.reduce_cost
                       + STARTUP_MARGIN)
        return self._startup

    def feasible(self, length):
        return self.cap_limit() > 0 and self.predicted(length) <= self.cap_limit()

    def post_cost(self):
        """Time to budget after a run for validation and hashing (the longest seen, at least 10 s)."""
        return max([10.0] + self.ledger().post())

    def affordable(self, case):
        """A run starts only if its whole cap and its post-processing still fit, so the invocation budget
        never shortens it. A run that no invocation could fit is an error with an actionable message; one
        that only this invocation's start-up crowds out is a budget stop."""
        length = case["nominal_length"]
        if getattr(self, "_startup", None) is None:
            self._startup = self.startup_allowance()       # frozen at the first engine admission
        p = self.predicted(length)
        least = self.a.soft_seconds - self.a.reserve_seconds - self.post_cost() - START_HEADROOM
        why = (f"length {length}: predicted run {p:.0f} s does not fit --soft-seconds {self.a.soft_seconds:.0f} less "
               f"reserve {self.a.reserve_seconds:.0f} s, post-processing {self.post_cost():.0f} s and start-up")
        if p > least or least <= 0:
            raise ValueError(f"{why} {START_HEADROOM:.0f} s; raise the budget or lower --predict / --reserve-seconds")
        if not self.feasible(length):
            # Only this invocation's own start-up left too little: a budget stop (the next invocation
            # measures its own start-up); the message says what to look at if it keeps happening.
            print(f"{why} {self.startup_allowance():.0f} s of this invocation; stopping (if every invocation "
                  "starts this slowly, raise the budget)", flush=True)
            return False
        pending = len(self.reductions.jobs)
        return self.remaining() >= (self.cap(case["nominal_length"]) + self.post_cost() + self.a.reserve_seconds
                                    + pending * self.reduce_cost)

    def timeout_counts(self, rec):
        """A timeout at the run's own cap is the arm's failure; a run that could not get its whole cap
        (only possible if admission was bypassed) is a budget stop and stays resumable."""
        return not rec.get("shortened", True)

    # ---------------------------------------------------------------- execution
    def effective_configs(self):
        if not self.cases:
            return
        first = self.cases[0]
        ids = read_id_text(Path(self.a.manifest) / first["prompt_file"])
        attempted = {}
        # Captures before a damaged line are not trusted (captured again); those after it are reused.
        for r in diagnostic_records(self.out / "effective-config.jsonl", after_damage=True)[0]:
            if r.get("attempted") and isinstance(r.get("arm"), str):
                attempted.setdefault(r["arm"], r)
        # Capture only when every run of this invocation would still get its predicted time afterwards.
        longest = max((self.predicted(c["nominal_length"]) for c in self.cases), default=0.0)
        room = longest > 0 and self.remaining() - CONFIG_BUDGET >= longest + self.a.reserve_seconds
        deadline = self.now() + CONFIG_BUDGET
        for name in self.arm_names:
            path = self.out / "log" / f"effective-config.{name}.txt"
            if name in attempted:
                # One attempt per output directory (binary and arms are fixed for it).
                self.append("effective-config.jsonl", {"arm": name, "invocation": self.invocation,
                                                       "reused_from": attempted[name].get("invocation"),
                                                       "rc": attempted[name].get("rc"),
                                                       "sha256": attempted[name].get("sha256")})
                continue
            budget = deadline - self.now()
            if not room or budget <= 1:
                self.append("effective-config.jsonl", {"arm": name, "invocation": self.invocation, "rc": None,
                                                       "deferred": "no prediction yet" if longest <= 0 else
                                                       "not enough budget in this invocation"})
                continue
            spec = self.arms[name]
            argv = self.launcher + [self.binary, self.models[name], *map(str, ids), "--generate",
                                    str(first["cont_tokens"]), *spec.get("argv", []), "--print-effective-config"]
            try:
                r = subprocess.run(argv, env=clean_env(spec.get("env", {})), capture_output=True, timeout=budget)
                atomic(path, r.stdout + b"\n--- stderr ---\n" + r.stderr)
                self.append("effective-config.jsonl", {"arm": name, "invocation": self.invocation, "attempted": True,
                                                       "rc": r.returncode, "sha256": file_sha(path)})
            except (OSError, subprocess.SubprocessError) as e:
                self.append("effective-config.jsonl", {"arm": name, "invocation": self.invocation, "attempted": True,
                                                       "rc": None, "error": f"{type(e).__name__}: {e}"})

    def run_arm(self, case, arm_label, arm_name, ids, cont_file):
        """One engine run. Returns (record, path or None); the path exists only for a complete run."""
        spec = self.arms[arm_name]
        bin_path = self.out / "logits" / f"{case['case_id']}.{arm_label}.bin"
        partial = bin_path.with_name(bin_path.name + ".partial")
        for stale in (bin_path, partial):      # the engine writes PATH.partial and renames it on success
            if stale.exists() or stale.is_symlink():
                stale.unlink()
        env = clean_env(spec.get("env", {}))
        env["STRIX_BITCALC_LOGITS_OUT"] = str(bin_path)
        env["STRIX_BITCALC_FORCE_TOKENS"] = str(cont_file)
        argv = self.launcher + [self.binary, self.models[arm_name], *map(str, ids), "--generate", str(case["cont_tokens"]),
                                *spec.get("argv", [])]
        cap = self.cap(case["nominal_length"])
        room = (self.remaining() - self.a.reserve_seconds - self.post_cost()
                - len(self.reductions.jobs) * self.reduce_cost)
        allowed = max(1.0, min(cap, room))
        shortened = allowed < cap
        self.run_seq += 1
        run_id = f"{self.invocation}-{self.run_seq}"
        start = self.now()
        record = {"case_id": case["case_id"], "arm": arm_label, "nominal_length": case["nominal_length"],
                  "plan_sha256": self.plan_sha, "binary_sha256": self.meta["binary_sha256"],
                  "arms_sha256": self.meta["arms_sha256"], "model_receipt_sha256": self.meta["arm_model_receipt_sha256"][arm_name],
                  "argv_tail": [*argv[len(self.launcher) + 2 + len(ids):]],
                  "env": {k: v for k, v in env.items() if k.startswith(DROP_PREFIX)}, "prompt_tokens": len(ids),
                  "allowed_seconds": round(allowed, 1), "cap_seconds": round(cap, 1), "shortened": shortened,
                  "invocation": self.invocation,
                  "run_id": run_id}
        stem = self.out / "log" / f"{case['case_id']}.{arm_label}"
        rc, timed_out, stdout, stderr, rows, forced, problem, digest = None, False, b"", b"", 0, None, None, None
        exec_seconds = None
        interrupted = False
        try:
            if self.replay is not None:
                self.replay_capture = None
                rc, stdout, stderr, why = self.replay_run(case, arm_label, arm_name, bin_path, cont_file)
                if why is not None:          # no engine output for this pair yet: not run, never a failure
                    interrupted, record["replay"] = True, why
                elif self.replay_capture is not None:
                    record["replay_capture"] = self.replay_capture
            else:
                try:
                    r = subprocess.run(argv, env=env, capture_output=True, timeout=allowed)
                    rc, stdout, stderr = r.returncode, r.stdout, r.stderr
                except subprocess.TimeoutExpired as e:
                    timed_out, stdout, stderr = True, e.stdout or b"", e.stderr or b""
            exec_seconds = round(self.now() - start, 3)
            atomic(str(stem) + ".stdout", stdout)
            atomic(str(stem) + ".stderr", stderr)
            result = parse_result(decode_log(stdout))
            raw_forced = (result or {}).get("bitcalc_forced_steps") if isinstance(result, dict) else None
            # Engine output is normalised before it becomes evidence: a non-integer count is kept only
            # as a diagnostic, so the run record always stays well-formed (and its failure counts).
            forced = raw_forced if isinstance(raw_forced, int) and not isinstance(raw_forced, bool) else None
            if raw_forced is not None and forced is None:
                record["forced_steps_raw"] = repr(raw_forced)[:200]
            if rc == 0 and not timed_out:
                if not isinstance(result, dict):
                    problem = "no JSON result"
                elif not bin_path.exists():
                    problem = "no logits file"
                elif forced != case["cont_tokens"]:
                    problem = f"forced steps {repr(raw_forced)[:80]} != continuation {case['cont_tokens']}"
                else:
                    checked = self.validate(bin_path, cont_file, case["cont_tokens"])
                    if checked is None:
                        interrupted = True
                    else:
                        rows, problem, digest = int(checked["rows"]), checked["problem"], checked["sha256"]
                        if self.replay is not None and digest is not None and self.replay_capture is not None \
                                and digest != self.replay_capture["bin_sha256"]:
                            problem, digest = "replay: validated logits differ from the committed capture", None
                        why = witness_problem(self.plan.get("witness", {}).get(arm_name), result,
                                              decode_log(stderr))
                        if digest is not None and why:
                            problem, digest = f"witness: {why}", None
            elif not timed_out:
                problem = f"exit code {rc}"
        except (OSError, ValueError, struct.error) as e:
            problem = f"{type(e).__name__}: {e}"
        finally:
            complete = digest is not None
            if not complete and bin_path.exists():
                bin_path.unlink()      # never leave a partial or unverified file behind
            if partial.exists() or partial.is_symlink():
                partial.unlink()
        wall = round(self.now() - start, 3)
        record.update(rc=rc, timeout=timed_out, wall_seconds=wall,
                      exec_seconds=wall if exec_seconds is None else min(exec_seconds, wall), rows=rows,
                      route=[l.strip() for l in stderr.decode("utf-8", "replace").splitlines() if "auto-prefill" in l],
                      forced_steps=forced, problem=problem, complete=complete, logits_sha256=digest,
                      interrupted=interrupted, stdout_sha256=sha(stdout), stderr_sha256=sha(stderr))
        self.append("runs.jsonl", record)
        return record, (bin_path if complete else None)

    def replay_run(self, case, arm_label, arm_name, bin_path, cont_file):
        """(rc, stdout, stderr, missing) of one pair taken from the --replay directory. The result must name this
        case and the sha256 of its prompt and continuation files, else the run is a failed attempt (rc 3); a pair
        without output is reported as missing (not run)."""
        cid = case["case_id"]
        label = self.replay_label(arm_label)
        entries = self.replay_entries.get((cid, label), [])
        if not entries:
            return None, b"", b"", f"no committed capture for {cid}.{label}"
        bad = lambda why: (3, b"", f"replay: {why}\n".encode(), None)
        if len(entries) != 1:
            return bad(f"{len(entries)} committed captures for {cid}.{label}")
        e = entries[0]
        # The capture names this case, the frozen manifest inputs and the binary the kit hashed.
        if e.get("prompt_sha256") != case["prompt_sha256"] or e.get("cont_sha256") != case["cont_sha256"]:
            return bad("capture inputs are not this case's frozen prompt / continuation")
        if e.get("binary_sha256") != self.meta["binary_sha256"]:
            return bad("capture was made by another binary")
        if not isinstance(e.get("model_sha256"), str) or not e["model_sha256"] or e["model_sha256"] != self.model_shas[arm_name]:
            return bad("capture names another model (model_sha256 differs from the model receipt)")
        if not e.get("capture_id") or not e.get("collection_id"):
            return bad("capture lacks capture_id / collection_id")
        if arm_label.endswith(REPEAT_SUFFIX):
            # The reference repeat must be an independent capture (equal logits bytes are expected).
            mark = self.done(cid, self.plan["reference"]) or {}
            kept = self.ledger().by_id(mark.get("run_id")) if mark.get("run_id") else None
            cap = (kept or {}).get("replay_capture") or {}
            if not evidence.replay_provenance(kept) or cap["capture_id"] == e["capture_id"] \
                    or cap["collection_id"] == e["collection_id"]:
                return bad("reference repeat is not an independent capture of the retained reference run")
        try:
            stdout = (self.replay / f"{cid}.{label}.json").read_bytes()
            err = self.replay / f"{cid}.{label}.stderr"
            stderr = err.read_bytes() if err.exists() else b""
            shutil.copyfile(self.replay / f"{cid}.{label}.bin", bin_path)
        except OSError as x:
            return bad(f"capture files unreadable ({type(x).__name__})")
        # stderr feeds the route gate and any stderr witness, so it is committed like the other files
        # (an engine without stderr commits the digest of the empty file and writes none).
        if not isinstance(e.get("stderr_sha256"), str):
            bin_path.unlink()
            return bad("capture commits no stderr digest")
        if sha(stdout) != e.get("json_sha256") or file_sha(bin_path) != e.get("bin_sha256") or \
                sha(stderr) != e["stderr_sha256"]:
            bin_path.unlink()
            return bad("capture bytes differ from the committed journal entry")
        result = parse_result(decode_log(stdout))
        want = {"case_id": cid, "prompt_sha256": case["prompt_sha256"], "cont_sha256": case["cont_sha256"]}
        if not isinstance(result, dict) or any(result.get(k) != v for k, v in want.items()):
            bin_path.unlink()
            return bad("result does not name this case / prompt / continuation")
        if file_sha(cont_file) != case["cont_sha256"]:
            bin_path.unlink()
            return bad("continuation file changed since the manifest was frozen")
        self.replay_capture = {"capture_id": e["capture_id"], "collection_id": e["collection_id"], "label": label,
                               "bin_sha256": e["bin_sha256"]}
        return 0, stdout, stderr, None

    def replay_label(self, arm_label):
        """The engine's label of a kit label: an explicit mapping, else the same name (the repeat is never folded
        into the reference)."""
        return self.replay_labels.get(arm_label, arm_label)

    def logits_sha(self, path):
        """Hash of a retained logits file, computed in a child stopped at the deadline (BudgetStop)."""
        budget = self.remaining() - self.a.reserve_seconds / 2
        if budget <= 1:
            raise BudgetStop(str(path))
        argv = [sys.executable, "-m", "tools.qualification.option_qual.validate", "--sha", str(path)]
        try:
            r = subprocess.run(argv, cwd=ROOT, capture_output=True, timeout=budget)
        except subprocess.TimeoutExpired:
            raise BudgetStop(str(path))
        try:
            out = json.loads(r.stdout.decode("utf-8", "replace").strip().splitlines()[-1])
        except (ValueError, IndexError):
            out = None
        if r.returncode != 0 or not isinstance(out, dict):
            raise OSError(f"hashing {path.name} failed (exit {r.returncode})")
        return out["sha256"]

    def validate(self, path, cont_file, rows):
        """Validation and hashing in a child process bounded by the deadline; None = stopped by it."""
        budget = self.remaining() - self.a.reserve_seconds / 2
        if budget <= 1:
            return None
        argv = [sys.executable, "-m", "tools.qualification.option_qual.validate", str(path), str(cont_file), str(rows)]
        try:
            r = subprocess.run(argv, cwd=ROOT, capture_output=True, timeout=budget)
        except subprocess.TimeoutExpired:
            return None
        try:
            out = json.loads(r.stdout.decode("utf-8", "replace").strip().splitlines()[-1])
            if r.returncode != 0 or not isinstance(out, dict):
                raise ValueError("bad validator output")
            return out
        except (ValueError, IndexError):
            return {"rows": 0, "problem": f"validator failed (exit {r.returncode})", "sha256": None}

    def labels(self, cid):
        ref = self.plan["reference"]
        return [(a, a) for a in self.arm_names[1:]] + ([(ref + REPEAT_SUFFIX, ref)] if cid in self.repeats else [])

    def forget_dependents(self, cid):
        """Results reduced against an earlier reference file must be produced again."""
        for lab, _ in self.labels(cid):
            self.unmark(cid, lab)
            drop_bundle(self.out / "red" / f"{cid}.{lab}")

    def consistent(self, case, label, mark, sha_key, ref_sha=None):
        """The same rules the analysis applies (evidence.Ledger)."""
        led = self.ledger()
        if mark.get("status") in ("FAILED", "INVALID"):
            return led.outcome_bound(mark, case["case_id"], label, ref_sha, self.plan_sha)
        return led.result_bound(case["case_id"], label, mark.get(sha_key), case["cont_tokens"], mark, self.plan_sha)

    def restore_terminal(self, case, label):
        """A pair whose recorded failures reached the limit stays failed - even if its marker was lost or
        quarantined, or a reset voided its runs - unless a counting run succeeded after them."""
        cid = case["case_id"]
        if self.status(cid, label) in ("COMPLETE", "INVALID", "FAILED") or \
                self.attempts(cid, label) < self.a.max_attempts:
            return
        led = self.ledger()
        failed, newest = led.failed(cid, label), led.newest(cid, label)
        if newest is None or newest in failed:
            last = newest if newest is not None else failed[-1]
            self.mark(cid, label, {"status": "FAILED", "attempts": len(failed),
                                   "max_attempts": self.a.max_attempts, "error": "restored from the run journal",
                                   "run_id": last, "restored": True})

    def at_ceiling(self, case, label):
        """Checked before scheduling any run: an exhausted pair is never run again."""
        self.restore_terminal(case, label)
        return self.terminal(case["case_id"], label)

    def reconcile_marker(self, cid, label, mark, led):
        """Before a FAILED marker is set aside: if it names a complete run of its pair whose failure the
        journal does not hold, the failure is written back as an event (no attempt is ever lost)."""
        if mark.get("status") != "FAILED":
            return
        rec = led.by_id(mark.get("run_id"))
        if rec is not None and (rec["case_id"], rec["arm"]) == (cid, label) and rec["complete"] is True \
                and mark["run_id"] not in led.failed(cid, label):
            self.run_seq += 1
            self.append("runs.jsonl", {"event": "reduction_failed", "run_id": f"{self.invocation}-e{self.run_seq}",
                                       "of_run": mark["run_id"], "case_id": cid, "arm": label,
                                       "error": "reconstructed from its failure marker", "reconstructed": True,
                                       **{k: evidence.arm_identity(self.meta, label)[k] for k in evidence.RUN_IDENTITY}})

    def settle_recorded_failures(self, case):
        """A marker left on a run that the journal already records as failed (e.g. interrupted between
        journaling the failure and updating the marker) becomes FAILED with the journal's count."""
        cid = case["case_id"]
        led = self.ledger()
        for label in [self.plan["reference"]] + [l for l, _ in self.labels(cid)]:
            mark = self.done(cid, label) or {}
            if mark.get("status") in ("RAN", "COMPLETE") and mark.get("run_id") in led.failed(cid, label):
                failed = led.failed(cid, label)
                self.mark(cid, label, {"status": "FAILED", "attempts": len(failed),
                                       "max_attempts": self.a.max_attempts, "run_id": mark["run_id"],
                                       "error": "failure recorded in the journal; marker updated on resume"})

    def repair(self, case):
        """Validate everything already on disk for this case before trusting it on resume."""
        cid = case["case_id"]
        ref = self.plan["reference"]
        self.settle_recorded_failures(case)
        mark = self.done(cid, ref) or {}
        if mark.get("status") == "FAILED" and not self.consistent(case, ref, mark, "sha256"):
            self.reconcile_marker(cid, ref, mark, self.ledger())
            self.quarantine(self.out / "done" / f"{cid}.{ref}", "failure marker not bound to its run record")
            mark = {}
        self.restore_terminal(case, ref)
        mark = self.done(cid, ref) or {}
        if mark.get("status") == "COMPLETE" and not self.consistent(case, ref, mark, "sha256"):
            # The run record behind this reference is missing (e.g. a quarantined journal tail).
            self.mark(cid, ref, {"status": "STALE", "attempts": mark.get("attempts", 0),
                                 "error": "no complete run record for the recorded reference"})
            drop_bundle(self.out / "red" / f"{cid}.{ref}.self")
            self.forget_dependents(cid)
            mark = self.done(cid, ref)
        ref_bin = self.out / "logits" / f"{cid}.{ref}.bin"
        if ref_bin.exists() and (mark.get("status") != "COMPLETE" or self.logits_sha(ref_bin) != mark.get("sha256")):
            ref_bin.unlink()           # an unverified reference file is never used
            if mark.get("status") == "COMPLETE":
                # A retained reference that contradicts its record: everything of the case is produced again
                # (the attempt count survives; dependents are re-reduced against the new reference).
                self.mark(cid, ref, {"status": "STALE", "attempts": mark.get("attempts", 0),
                                     "error": "retained reference logits contradict the recorded hash"})
                drop_bundle(self.out / "red" / f"{cid}.{ref}.self")
                self.forget_dependents(cid)
                mark = self.done(cid, ref)
        if mark.get("status") == "COMPLETE" and "self_red_sha256" in mark:
            if bundle_shas(self.out / "red" / f"{cid}.{ref}.self") != (mark["self_red_sha256"],
                                                                         mark.get("self_redj_sha256")):
                for key in ("self_red_sha256", "self_redj_sha256"):
                    mark.pop(key, None)
                mark.pop("plan_sha256", None)
                self.mark(cid, ref, mark)
        for lab, _ in self.labels(cid):
            m = self.done(cid, lab) or {}
            cand = self.out / "logits" / f"{cid}.{lab}.bin"
            if m.get("status") in ("COMPLETE", "RAN", "FAILED", "INVALID") and \
                    not self.consistent(case, lab, m, "cand_sha256", mark.get("sha256")):
                self.reconcile_marker(cid, lab, m, self.ledger())
                self.quarantine(self.out / "done" / f"{cid}.{lab}", "marker not bound to its run record")
                drop_bundle(self.out / "red" / f"{cid}.{lab}")
                m = {}
            if not m and mark.get("status") == "COMPLETE":
                self.restore_terminal(case, lab)
                m = self.done(cid, lab) or {}
            if m.get("status") == "COMPLETE" and (
                    m.get("ref_sha256") != mark.get("sha256") or
                    bundle_shas(self.out / "red" / f"{cid}.{lab}") != (m.get("red_sha256"), m.get("redj_sha256"))
                    or (cand.exists() and self.logits_sha(cand) != m.get("cand_sha256"))):
                self.unmark(cid, lab)
                drop_bundle(self.out / "red" / f"{cid}.{lab}")
            elif m.get("status") == "RAN" and (not cand.exists() or self.logits_sha(cand) != m.get("cand_sha256")
                                                 or m.get("ref_sha256") != mark.get("sha256")):
                self.unmark(cid, lab)
            if self.status(cid, lab) != "RAN" and cand.exists() and self.status(cid, lab) != "COMPLETE":
                cand.unlink()

    def self_ok(self, cid):
        mark = self.done(cid, self.plan["reference"]) or {}
        return mark.get("status") == "COMPLETE" and "self_red_sha256" in mark

    def run_case(self, case):
        """'done', 'budget' or 'failed' (a pair failed or is invalid; other cases continue)."""
        cid = case["case_id"]
        ref = self.plan["reference"]
        labels = self.labels(cid)
        try:
            self.repair(case)
        except BudgetStop:
            return "budget"

        def outcome():
            states = [self.status(cid, lab) for lab, _ in labels]
            if self.status(cid, ref) == "FAILED" or "FAILED" in states or "INVALID" in states:
                return "failed"
            return "done"

        if self.terminal(cid, ref):
            return "failed"
        if self.self_ok(cid) and all(self.finished(cid, lab) for lab, _ in labels):
            return outcome()
        root = Path(self.a.manifest)
        ids = read_id_text(root / case["prompt_file"])
        cont_file = (root / case["cont_file"]).resolve()
        ref_bin = self.out / "logits" / f"{cid}.{ref}.bin"
        stopped = False
        if not ref_bin.exists():      # (repair removed any unverified one)
            if self.at_ceiling(case, ref):
                return "failed"
            if not self.affordable(case):
                return "budget"
            prev = self.done(cid, ref) or {}
            old = prev.get("sha256")
            rec, path = self.run_arm(case, ref, ref, ids, cont_file)
            if path is None:
                if rec["interrupted"] or (rec["timeout"] and not self.timeout_counts(rec)):
                    return "budget"
                self.fail(cid, ref, rec["problem"] or "timeout at the run's cap", rec)
                return "failed"
            if old and old != rec["logits_sha256"]:
                self.forget_dependents(cid)
            # A reference that failed before (e.g. its reduction) keeps its attempt count across re-runs.
            self.mark(cid, ref, {"status": "COMPLETE", "route": rec["route"], "sha256": rec["logits_sha256"],
                                 "rows": rec["rows"], "attempts": prev.get("attempts", 0), "run_id": rec["run_id"]})
        ref_sha = self.done(cid, ref)["sha256"]
        if not self.self_ok(cid):
            self.reductions.submit("self", cid, ref, ref_bin, None, None)
        for lab, name in labels:
            if self.finished(cid, lab):
                continue
            if not self.reductions.wait(1, self.deadline):      # at most one reduction behind the GPU
                stopped = True
                break
            if self.status(cid, ref) != "COMPLETE":
                break                                           # the reference reduction failed
            m = self.done(cid, lab) or {}
            cand = self.out / "logits" / f"{cid}.{lab}.bin"
            if m.get("status") == "RAN":                        # finished run, reduction still owed
                self.reductions.submit("pair", cid, lab, ref_bin, cand, {k: m.get(k) for k in
                                                                         ("route", "ref_sha256", "cand_sha256",
                                                                          "attempts", "run_id")})
                continue
            if self.at_ceiling(case, lab):
                continue
            if not self.affordable(case):
                stopped = True
                break
            rec, cand = self.run_arm(case, lab, name, ids, cont_file)
            if cand is None:
                if rec["interrupted"] or (rec["timeout"] and not self.timeout_counts(rec)):
                    stopped = True
                    break
                self.fail(cid, lab, rec["problem"] or "timeout at the run's cap", rec)
                continue
            meta = {"route": rec["route"], "ref_sha256": ref_sha, "cand_sha256": rec["logits_sha256"],
                    "attempts": (self.done(cid, lab) or {}).get("attempts", 0), "run_id": rec["run_id"]}
            self.mark(cid, lab, dict(meta, status="RAN"))
            self.reductions.submit("pair", cid, lab, ref_bin, cand, meta)
        if not self.reductions.wait(0, self.deadline):
            stopped = True
        if stopped:
            return "budget"
        if self.status(cid, ref) != "COMPLETE" or not self.self_ok(cid):
            return "failed" if self.status(cid, ref) == "FAILED" else "budget"
        if not all(self.finished(cid, lab) for lab, _ in labels):
            return "failed"
        if not self.a.keep_logits and ref_bin.exists():
            ref_bin.unlink()
        return outcome()

    def run(self):
        failed = waiting = False
        captured = not self.a.effective_config or self.replay is not None
        try:
            self.preflight()
            for case in self.cases:
                missing = self.replay_missing(case)
                if missing:
                    # Replay: this case has no engine output yet; it is not run (and never a failure).
                    waiting = True
                    print(f"{case['case_id']}: no engine output yet ({', '.join(missing)})", flush=True)
                    continue
                state = self.run_case(case)
                if not captured and state != "budget":
                    self.effective_configs()     # after the first case: it never shortens a first run
                    captured = True
                if state == "budget":
                    print(f"soft budget: stopped at {case['case_id']}", flush=True)
                    return 2
                if state == "failed":
                    failed = True
                    print(f"{case['case_id']}: a run failed or a result is invalid; see runs.jsonl and done/",
                          flush=True)
                    continue
                print(f"{case['case_id']}: complete ({self.now() - self.t0:.0f} s)", flush=True)
        finally:
            self.reductions.stop_all()
        bad = any(self.status(c["case_id"], lab) in ("FAILED", "INVALID")
                  for c in self.cases for lab in [self.plan["reference"]] + [l for l, _ in self.labels(c["case_id"])])
        return 1 if failed or self.bad or bad else (2 if waiting else 0)

    def replay_missing(self, case):
        """Engine labels of this case's still-needed pairs that have no output in the --replay directory."""
        if self.replay is None:
            return []
        cid, ref = case["case_id"], self.plan["reference"]
        settled = ("COMPLETE", "RAN", "FAILED", "INVALID")
        need = []
        if not self.finished(cid, ref) and self.status(cid, ref) not in settled and \
                not (self.out / "logits" / f"{cid}.{ref}.bin").exists():
            need.append(ref)
        need += [lab for lab, _ in self.labels(cid) if self.status(cid, lab) not in settled]
        return [self.replay_label(lab) for lab in need
                if not self.replay_entries.get((cid, self.replay_label(lab)))]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--arms", required=True)
    p.add_argument("--binary", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model-receipt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--soft-seconds", type=float, default=430.0,
                   help="nothing started by this invocation outlives this many seconds")
    p.add_argument("--reserve-seconds", type=float, default=20.0,
                   help="kept free at the end of the budget for bookkeeping")
    p.add_argument("--cases", default="")
    p.add_argument("--predict", default="",
                   help="LENGTH=SECONDS,... (or one SECONDS for every length): worst expected wall time of one run")
    p.add_argument("--keep-logits", action="store_true")
    p.add_argument("--max-attempts", type=int, default=2, help="runs of one failing (case, arm) before giving up")
    p.add_argument("--cap-factor", type=float, default=1.5,
                   help="execution cap of a run = this x the longest completed run at its length (or --predict)")
    p.add_argument("--cap-floor", type=float, default=60.0,
                   help="the execution cap is at least this many seconds above the predicted run time")
    p.add_argument("--no-effective-config", dest="effective_config", action="store_false",
                   help="do not record each arm's --print-effective-config output")
    p.add_argument("--replay", default="",
                   help="DIR of a many-case engine run: each (case, arm) is taken from DIR/<case>.<label>.json "
                        "(the engine's result, used as the run's stdout; it must name the case and the sha256 of "
                        "its prompt and continuation files) and DIR/<case>.<label>.bin (its logits) instead of "
                        "starting the engine; --binary is still the engine that produced them (hashed as usual)")
    p.add_argument("--replay-label", action="append", default=[],
                   help="ARM=LABEL: the engine's label of a plan arm in --replay (default: the arm name; the "
                        "reference repeat arm is REF" + REPEAT_SUFFIX + ")")
    p.add_argument("--allow-fake-for-tests", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--python-binary-for-tests", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None):
    try:
        return Runner(parse_args(argv)).run()
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
