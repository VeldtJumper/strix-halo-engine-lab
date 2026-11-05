import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.qualification.option_qual import analyze, common, evidence, manifest, plan, run
from tools.qualification.longctx_fidelity.tok import FakeTokenizer
from tools.qualification.option_qual.tests.test_core import _sources

# A stand-in for the decode command: deterministic logits that depend on the recent history, plus
# optional noise (FAKE_NOISE) for an "option" arm; honours the measurement hook variables.
FAKE = r'''
import json, math, os, random, struct, sys
args = sys.argv[1:]
if "--print-effective-config" in args:
    print(json.dumps({"fake": True, "noise": os.environ.get("FAKE_NOISE", "0")})); sys.exit(0)
g = args.index("--generate")
ids = [int(x) for x in args[1:g]]
n = int(args[g + 1])
forced = [int(x) for x in open(os.environ["STRIX_BITCALC_FORCE_TOKENS"]).read().replace(",", " ").split()]
if os.environ.get("FAKE_TRUNCATE") or (os.environ.get("FAKE_TRUNCATE_FILE")
                                       and os.path.exists(os.environ["FAKE_TRUNCATE_FILE"])):
    n = n // 2
if os.environ.get("FAKE_ONE_BYTE"):
    open(os.environ["STRIX_BITCALC_LOGITS_OUT"], "wb").write(b"x")
    print(json.dumps({"generated_tokens": [], "bitcalc_forced_steps": n}))
    sys.exit(0)
if os.environ.get("FAKE_IGNORE_FORCE"):
    forced = []
noise = float(os.environ.get("FAKE_NOISE", "0"))
if open(args[0], "rb").read() == b"wrong historical model":
    noise += 0.8 if os.path.basename(args[0]) == "REF.gguf" else 1.5
if os.environ.get("FAKE_NOISE_FILE") and os.path.exists(os.environ["FAKE_NOISE_FILE"]):
    noise = float(open(os.environ["FAKE_NOISE_FILE"]).read())
vocab = 256
hist = list(ids)
final = os.environ["STRIX_BITCALC_LOGITS_OUT"]
if os.path.lexists(final) or os.path.lexists(final + ".partial"):
    sys.stderr.write("capture destination is not fresh\n"); sys.exit(5)
out = open(final + ".partial", "xb")
emitted = []
for step in range(n):
    key = (hist[-1] * 31 + hist[-2] * 7 + len(hist)) % 100003
    r = random.Random(key)
    z = [2.0 * r.gauss(0, 1) for _ in range(vocab)]
    z[(hist[-1] + 1) % vocab] += 4.0
    if noise:
        r2 = random.Random(key + 7919)
        z = [v + noise * r2.gauss(0, 1) for v in z]
    if os.environ.get("FAKE_NAN") or (os.environ.get("FAKE_NAN_REF") and step == 0):
        z[3] = math.nan
    top = max(range(vocab), key=lambda i: (z[i], -i))
    tok = forced[step] if step < len(forced) else top
    out.write(struct.pack("<6I", 0x42434C47, step, len(hist) - 1, top, tok, vocab) + struct.pack("<%df" % vocab, *z))
    emitted.append(tok)
    hist.append(tok)
out.close()
os.rename(final + ".partial", final)
sys.stderr.write("Qwen auto-prefill: automatic T8 context=%d; reason=test%s\n" % (len(ids), os.environ.get("FAKE_ROUTE", "")))
steps = n if os.environ.get("FAKE_CLAIM_FORCED") else min(n, len(forced))
if os.environ.get("FAKE_FORCED_TEXT_FILE") and os.path.exists(os.environ["FAKE_FORCED_TEXT_FILE"]):
    steps = str(steps)
print(json.dumps({"generated_tokens": emitted, "bitcalc_forced_steps": steps,
                  "lever": os.environ.get("FAKE_LEVER_ON") == "1" or bool(os.environ.get("FAKE_LEVER_FILE") and
                           os.path.exists(os.environ["FAKE_LEVER_FILE"]))}))
'''


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.dir = Path(self.tmp.name)
        src = _sources(d / "src", parents=6)
        manifest.build(src, d / "m", FakeTokenizer(), [256, 512], cont=24, salt="e2e", search_per_stratum=2)
        (d / "fake.py").write_text(FAKE)
        (d / "model.gguf").write_bytes(b"model")
        (d / "receipt").write_text(common.file_sha(d / "model.gguf") + "  model.gguf")
        arms = {"REF": {}, "CTRL": {"env": {"FAKE_NOISE": "0.02"}}, "A": {"env": {"FAKE_NOISE": "0.005"}},
                "B": {"env": {"FAKE_NOISE": "1.5"}}, "AB": {"env": {"FAKE_NOISE": "1.5"}}}
        (d / "arms.json").write_text(json.dumps({"arms": arms}))

    def tearDown(self):
        self.tmp.cleanup()

    def _plan(self, split, frozen=True, name="plan.json"):
        pl = plan.init(self.dir / "m", split, "REF", "CTRL", ["A", "B"], [{"name": "AB", "members": ["A", "B"]}],
                       [256, 512], bootstrap=500, min_parents=2,
                       repeat_cases=["en-00.L256"] if split == "search" else ["en-02.L256", "en-05.L256"])
        pl["endpoints"]["E2"] = {"fixed": 0.01}
        pl["endpoints"]["E3"] = {"fixed": 0.01}
        pl["endpoints"]["E4"] = {"fixed": 0.05}
        if frozen:
            pl = plan.freeze(pl)
        plan.save(self.dir / name, pl)
        return self.dir / name

    def _run(self, plan_path, out, extra=()):
        d = self.dir
        return run.main(["--manifest", str(d / "m"), "--plan", str(plan_path), "--arms", str(d / "arms.json"),
                         "--binary", str(d / "fake.py"), "--model", str(d / "model.gguf"), "--model-receipt",
                         str(d / "receipt"), "--out", str(out), "--soft-seconds", "600", "--allow-fake-for-tests",
                         "--python-binary-for-tests", *extra])

    def _capture_fixture(self, live, target, models):
        """Package fake captures with producer provenance, independently of kit run metadata."""
        import shutil
        target.mkdir()
        cases = {c["case_id"]: c for c in manifest.load(self.dir / "m")[0]["cases"]}
        entries = []
        for n, out in enumerate(sorted((live / "log").glob("*.stdout"))):
            cid, label = out.name[:-len(".stdout")].rsplit(".", 1)
            c = cases[cid]
            result = run.parse_result(out.read_text())
            result.update(case_id=cid, prompt_sha256=c["prompt_sha256"], cont_sha256=c["cont_sha256"])
            (target / f"{cid}.{label}.json").write_text(json.dumps(result))
            for folder, suffix in (("log", "stderr"), ("logits", "bin")):
                shutil.copyfile(live / folder / f"{cid}.{label}.{suffix}", target / f"{cid}.{label}.{suffix}")
            entries.append(dict(status="complete", capture_id=f"cap-{n}",
                                collection_id="repeat" if label.endswith("@repeat") else "main",
                                case_id=cid, label=label, binary_sha256=common.file_sha(self.dir / "fake.py"),
                                model_sha256=models[label.removesuffix("@repeat")],
                                prompt_sha256=c["prompt_sha256"], cont_sha256=c["cont_sha256"],
                                **{suffix + "_sha256": common.file_sha(target / f"{cid}.{label}.{suffix}")
                                   for suffix in ("json", "bin", "stderr")}))
        (target / "runs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))

    def test_per_arm_model_capture_and_receipts(self):
        d = self.dir
        p, live = self._plan("holdout"), d / "live"
        self.assertEqual(self._run(p, live, ["--keep-logits"]), 0)
        arms = json.loads((d / "arms.json").read_text())
        models = {name: common.file_sha(d / "model.gguf") for name in arms["arms"]}
        receipts = {name: common.file_sha(d / "receipt") for name in arms["arms"]}
        for name in ("REF", "A"):
            model, receipt = d / f"{name}.gguf", d / f"{name}.receipt"
            model.write_bytes(name.encode())
            receipt.write_text(common.file_sha(model) + "  " + model.name)
            arms["arms"][name].update(model=str(model), model_receipt=str(receipt))
            models[name], receipts[name] = common.file_sha(model), common.file_sha(receipt)
        self.assertEqual(len(set(models.values())), 3)
        (d / "arms.json").write_text(json.dumps(arms))
        replay, out = d / "captures", d / "rep"
        self._capture_fixture(live, replay, models)
        self.assertEqual(self._run(p, out, ["--replay", str(replay)]), 0)
        self.assertEqual(self._run(p, out, ["--replay", str(replay)]), 0)
        ctx, results, ctrl, means = analyze.analyze(d / "m", p, [out])
        self.assertEqual(ctx["global_problems"], [])
        self.assertTrue(ctx["qualification"])
        self.assertEqual(results["A"]["verdict"], "PASS", results["A"]["notes"])
        self.assertEqual(results["B"]["verdict"], "FAIL")
        for rec in common.read_journal(out / "runs.jsonl")[0]:
            self.assertEqual(rec["model_receipt_sha256"], receipts[rec["arm"].removesuffix("@repeat")])
        analyze.write(d / "receipts", ctx, results, ctrl, means)
        js = json.loads((d / "receipts" / "A.receipt.json").read_text())
        md = (d / "receipts" / "A.receipt.md").read_text()
        self.assertEqual(js["arm_model_receipt_sha256"], receipts)
        for name in ("REF", "CTRL", "A"):
            self.assertIn(f"`{name}` `{receipts[name]}`", md)
        meta_path = out / "run-meta.json"
        original_meta = meta_path.read_text()
        legacy_meta = [json.loads(line) for line in original_meta.splitlines()]
        for meta in legacy_meta:
            del meta["arm_model_receipt_sha256"]
        meta_path.write_text("".join(json.dumps(meta) + "\n" for meta in legacy_meta))
        invalid_ctx, invalid_results, _, _ = analyze.analyze(d / "m", p, [out])
        self.assertEqual(invalid_results["A"]["verdict"], "INVALID")
        self.assertTrue(any("provenance missing for model overrides" in x for x in invalid_ctx["global_problems"]))
        self.assertEqual(self._run(p, out, ["--replay", str(replay)]), 1)
        meta_path.write_text(original_meta)
        # Every other model must be refused for every arm, including independent repeats.
        args = run.parse_args(["--manifest", str(d / "m"), "--plan", str(p), "--arms", str(d / "arms.json"),
                               "--binary", str(d / "fake.py"), "--model", str(d / "model.gguf"),
                               "--model-receipt", str(d / "receipt"), "--out", str(out),
                               "--replay", str(replay), "--allow-fake-for-tests"])
        runner = run.Runner(args)
        for label in [*models, "REF@repeat"]:
            name = label.removesuffix("@repeat")
            case = next(c for c in runner.cases if (c["case_id"], label) in runner.replay_entries)
            entry = runner.replay_entries[(case["case_id"], label)][0]
            for wrong in set(models.values()) - {models[name]}:
                with self.subTest(label=label, wrong=wrong):
                    entry["model_sha256"] = wrong
                    rc, _, stderr, missing = runner.replay_run(case, label, name, d / "unused.bin", d / "unused.ids")
                    self.assertEqual(rc, 3)
                    self.assertIsNone(missing)
                    self.assertIn(b"model_sha256 differs", stderr)
            entry["model_sha256"] = models[name]

    @staticmethod
    def _historical_module(commit, name):
        """Execute the recorded project implementation as an artifact compatibility oracle."""
        import subprocess
        import types
        mod = types.ModuleType("tools.qualification.option_qual._golden_" + name)
        mod.__package__ = "tools.qualification.option_qual"
        mod.__file__ = str(Path(run.__file__).with_name(name + ".py"))
        try:
            source = subprocess.check_output(
                ["git", "show", f"{commit}:tools/qualification/option_qual/{name}.py"],
                cwd=Path(run.__file__).parent, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            raise unittest.SkipTest(f"revision {commit} is not in this checkout (compatibility oracle unavailable)")
        exec(compile(source, f"{commit}/{name}.py", "exec"), mod.__dict__)
        return mod

    def test_historical_direct_overrides_are_never_reused(self):
        from unittest import mock
        d = self.dir
        p = self._plan("search", frozen=False)
        original = json.loads((d / "arms.json").read_text())
        old = self._historical_module("e2f0d9b95", "run")
        empty = d / "empty-replay"
        empty.mkdir()
        (empty / "runs.jsonl").write_bytes(b"")
        for kind in ("COMPLETE", "RAN", "REF-CTRL"):
            with self.subTest(kind=kind):
                arms = json.loads(json.dumps(original))
                for name in (("REF", "CTRL") if kind == "REF-CTRL" else ("A",)):
                    receipt = d / f"{name}.receipt"
                    receipt.write_text(("c" if name == "REF" else "d") * 64 + "  other.gguf")
                    arms["arms"][name].update(model=str(d / "model.gguf"), model_receipt=str(receipt))
                (d / "arms.json").write_text(json.dumps(arms))
                out = d / kind
                with mock.patch.object(run, "Runner", old.Runner):
                    self.assertEqual(self._run(p, out, ["--keep-logits"]), 0)
                if kind == "RAN":
                    mark_path = next((out / "done").glob("*.A"))
                    mark = json.loads(mark_path.read_text())
                    mark["status"] = "RAN"
                    mark.pop("red_sha256")
                    mark.pop("redj_sha256")
                    mark_path.write_bytes(common.canonical(mark))
                    for suffix in ("npz", "json"):
                        (out / "red" / (mark_path.name + "." + suffix)).unlink()
                ctx, res, _, means = analyze.analyze(d / "m", p, [out])
                self.assertEqual(res["A"]["verdict"], "INVALID")
                self.assertTrue(any("lacks replay provenance" in x for x in ctx["global_problems"]))
                self.assertIsNone(means)
                summary = analyze.write(d / (kind + "-receipts"), ctx, res, None, means)
                self.assertNotIn("replay_policy", summary)
                hp = plan.init(d / "m", "holdout", "REF", "CTRL", ["A"], [], [256, 512])
                with self.assertRaises(ValueError):
                    plan.calibrate(hp, summary)
                before = {str(q.relative_to(out)): q.read_bytes() for q in out.rglob("*") if q.is_file()}
                self.assertEqual(self._run(p, out, ["--replay", str(empty)]), 1)
                self.assertEqual(before, {str(q.relative_to(out)): q.read_bytes()
                                          for q in out.rglob("*") if q.is_file()})
                # Losing metadata cannot turn historical direct evidence into replay evidence.
                (out / "run-meta.json").unlink()
                self.assertEqual(self._run(p, out, ["--replay", str(empty)]), 1)

    def test_historical_search_summary_requires_replay_attestation(self):
        from unittest import mock
        d = self.dir
        sp = self._plan("search", frozen=False)
        original_arms = json.loads((d / "arms.json").read_text())
        original_arms["arms"]["A"]["env"]["FAKE_NOISE"] = "0.1"
        (d / "arms.json").write_text(json.dumps(original_arms))
        search_live = d / "search-live"
        self.assertEqual(self._run(sp, search_live, ["--keep-logits"]), 0)
        hp_live = self._plan("holdout", name="capture-plan.json")
        live = d / "holdout-live"
        self.assertEqual(self._run(hp_live, live, ["--keep-logits"]), 0)
        arms = json.loads((d / "arms.json").read_text())
        for name in ("REF", "CTRL"):
            model, receipt = d / f"{name}.gguf", d / f"{name}.receipt"
            model.write_bytes(b"wrong historical model")
            # The receipts identify the correct model; the old direct runner uses wrong bytes.
            receipt.write_text(common.file_sha(d / "model.gguf") + "  " + model.name)
            arms["arms"][name].update(model=str(model), model_receipt=str(receipt))
        (d / "arms.json").write_text(json.dumps(arms))
        old_run = self._historical_module("e2f0d9b95", "run")
        old_analyze = self._historical_module("9344a7aea", "analyze")
        old_plan = self._historical_module("9344a7aea", "plan")
        out = d / "historical-search"
        with mock.patch.object(run, "Runner", old_run.Runner):
            self.assertEqual(self._run(sp, out, ["--keep-logits"]), 0)
        ctx, res, ctrl, means = old_analyze.analyze(d / "m", sp, [out])
        self.assertEqual(ctx["global_problems"], [])
        summary = old_analyze.write(d / "historical-receipts", ctx, res, ctrl, means)
        original = common.canonical(summary)
        self.assertNotIn("replay_policy", summary)
        hp = plan.init(d / "m", "holdout", "REF", "CTRL", ["A", "B"],
                       [{"name": "AB", "members": ["A", "B"]}], [256, 512], bootstrap=500, min_parents=2)
        # Isolate the model-dependent E2/E3 calibration regression from fixed NLL/top-1 gates.
        hp["endpoints"] = {e: hp["endpoints"][e] for e in ("E2", "E3")}
        hp["guards"] = {}
        frozen = old_plan.freeze(old_plan.calibrate(hp, summary))
        self.assertIsNone(old_plan.check(frozen))
        with self.assertRaisesRegex(ValueError, "replay policy attestation"):
            plan.calibrate(hp, summary)
        self.assertIn("replay policy attestation", plan.check(frozen))
        path = d / "historically-frozen.json"
        plan.save(path, frozen)
        self.assertEqual(self._run(path, d / "refused"), 1)
        captures, replay = d / "holdout-captures", d / "historically-admitted"
        self._capture_fixture(live, captures, {n: common.file_sha(d / "model.gguf") for n in arms["arms"]})
        # Recreate a HOLDOUT admitted by the historical plan checker, with correct replay models.
        with mock.patch.object(plan, "require_runnable", old_plan.require_runnable):
            self.assertEqual(self._run(path, replay, ["--replay", str(captures)]), 0)
        ctx, res, _, means = analyze.analyze(d / "m", path, [replay])
        self.assertEqual(res["A"]["verdict"], "INVALID")
        self.assertTrue(any("replay policy attestation" in p for p in ctx["global_problems"]))
        self.assertIsNone(means)
        self.assertTrue(all("replay policy attestation" in p for p in ctx["global_problems"]), ctx)
        self.assertEqual(common.canonical(summary), original)
        forged = dict(summary, replay_policy=plan.REPLAY_POLICY)
        forged["self_sha256"] = common.sha(common.canonical({k: v for k, v in forged.items() if k != "self_sha256"}))
        forged_plan = old_plan.calibrate(hp, forged)
        self.assertIn("replay_verification", plan.check(forged_plan))
        with self.assertRaisesRegex(ValueError, "replay_verification"):
            plan.freeze(forged_plan)
        search_captures = d / "search-captures"
        self._capture_fixture(search_live, search_captures,
                              {n: common.file_sha(d / "model.gguf") for n in arms["arms"]})
        search_replay = d / "verified-search"
        self.assertEqual(self._run(sp, search_replay, ["--replay", str(search_captures)]), 0)
        ev = {"manifest": d / "m", "search_plan": sp, "runs": [search_replay]}
        with self.assertRaisesRegex(ValueError, "differs from current replay evidence"):
            plan.calibrate(hp, forged, ev)
        forged_path = d / "forged-frozen.json"
        plan.save(forged_path, old_plan.freeze(forged_plan))
        self.assertEqual(self._run(forged_path, d / "forged-refused", ["--replay", str(captures)]), 1)
        admitted = d / "forged-admitted"
        with mock.patch.object(plan, "require_runnable", old_plan.require_runnable):
            self.assertEqual(self._run(forged_path, admitted, ["--replay", str(captures)]), 0)
        with mock.patch.object(old_analyze, "plan_mod", old_plan):
            old_ctx, old_res, _, _ = old_analyze.analyze(d / "m", forged_path, [admitted])
        self.assertEqual(old_ctx["global_problems"], [])
        self.assertEqual(old_res["A"]["verdict"], "PASS")
        before = {q.relative_to(admitted): q.read_bytes() for q in admitted.rglob("*") if q.is_file()}
        self.assertEqual(self._run(forged_path, admitted, ["--replay", str(captures)]), 1)
        self.assertEqual(before, {q.relative_to(admitted): q.read_bytes() for q in admitted.rglob("*") if q.is_file()})
        invalid_ctx, invalid_res, _, _ = analyze.analyze(d / "m", forged_path, [admitted])
        self.assertEqual(invalid_res["A"]["verdict"], "INVALID")
        self.assertTrue(any("replay_verification" in p for p in invalid_ctx["global_problems"]))
        verified = analyze.build_summary(*analyze.analyze(d / "m", sp, [search_replay]))
        calibrated = plan.calibrate(hp, verified, ev)
        self.assertGreater(forged_plan["endpoints"]["E2"]["fixed"], calibrated["endpoints"]["E2"]["fixed"])
        self.assertGreater(forged_plan["endpoints"]["E3"]["fixed"], calibrated["endpoints"]["E3"]["fixed"])
        # The same correct HOLDOUT captures fail with verified, model-faithful SEARCH margins.
        plan.save(path, plan.freeze(calibrated))
        verified_holdout = d / "verified-holdout"
        self.assertEqual(self._run(path, verified_holdout, ["--replay", str(captures)]), 0)
        ctx, res, _, _ = analyze.analyze(d / "m", path, [verified_holdout])
        self.assertEqual(ctx["global_problems"], [])
        self.assertEqual(res["A"]["verdict"], "FAIL")


    def test_calibration_effective_reference_and_control_receipts(self):
        d = self.dir
        # Produce actual captures for both splits before enabling model overrides.
        sp, slive = self._plan("search", frozen=False), d / "search-live"
        hp, hlive = self._plan("holdout", name="uncalibrated.json"), d / "holdout-live"
        self.assertEqual(self._run(sp, slive, ["--keep-logits"]), 0)
        self.assertEqual(self._run(hp, hlive, ["--keep-logits"]), 0)
        arms = json.loads((d / "arms.json").read_text())
        models = {n: common.file_sha(d / "model.gguf") for n in arms["arms"]}
        for name in ("REF", "CTRL"):
            receipt = d / f"{name}.receipt"
            receipt.write_bytes((d / "receipt").read_bytes())
            arms["arms"][name].update(model=str(d / "model.gguf"), model_receipt=str(receipt))
        (d / "arms.json").write_text(json.dumps(arms))
        sr, hr = d / "search-captures", d / "holdout-captures"
        self._capture_fixture(slive, sr, models)
        self._capture_fixture(hlive, hr, models)
        sout = d / "search-replay"
        self.assertEqual(self._run(sp, sout, ["--replay", str(sr)]), 0)
        ctx, res, ctrl, means = analyze.analyze(d / "m", sp, [sout])
        self.assertEqual(ctx["global_problems"], [])
        self.assertEqual(res["A"]["verdict"], "PASS")
        self.assertIsNotNone(means)
        summary = analyze.write(d / "search-receipts", ctx, res, ctrl, means)
        self.assertEqual(summary["replay_policy"], plan.REPLAY_POLICY)
        body = {k: v for k, v in summary.items() if k != "self_sha256"}
        self.assertEqual(summary["self_sha256"], common.sha(common.canonical(body)))
        self.assertNotIn("replay_policy", json.loads((d / "search-receipts" / "A.receipt.json").read_text()))
        hp = plan.init(d / "m", "holdout", "REF", "CTRL", ["A", "B"],
                       [{"name": "AB", "members": ["A", "B"]}], [256, 512], bootstrap=500, min_parents=2)
        for policy in (None, "verified-arm-model-replay-v0", 1, {}):
            with self.subTest(policy=policy):
                bad = dict(summary, replay_policy=policy)
                bad["self_sha256"] = common.sha(common.canonical({k: v for k, v in bad.items() if k != "self_sha256"}))
                with self.assertRaisesRegex(ValueError, "replay policy attestation"):
                    plan.calibrate(hp, bad)
        # Preserve the emission boundary for all six labels, including the independent repeat.
        snapshot = {q: q.read_bytes() for q in sout.rglob("*") if q.is_file()}
        records = common.read_journal(sout / "runs.jsonl")[0]
        cid = "en-00.L256"
        def restore():
            for q, data in snapshot.items():
                q.write_bytes(data)
        def check_withheld():
            damaged = analyze.build_summary(*analyze.analyze(d / "m", sp, [sout]))
            self.assertNotIn("replay_policy", damaged)
            with self.assertRaises(ValueError):
                plan.calibrate(hp, damaged, {"manifest": d / "m", "search_plan": sp, "runs": [sout]})
        try:
            for label in ("REF", "CTRL", "A", "B", "AB", "REF@repeat"):
                for defect in ("missing", "partial", "FAILED", "INVALID", "record", "replay", "receipt"):
                    with self.subTest(label=label, defect=defect):
                        restore()
                        marker = sout / "done" / f"{cid}.{label}"
                        if defect == "missing":
                            marker.unlink()
                        elif defect == "partial":
                            marker.write_bytes(b'{"status":')
                        elif defect in ("FAILED", "INVALID"):
                            mark = json.loads(marker.read_text())
                            mark["status"] = defect
                            marker.write_bytes(common.canonical(mark))
                        else:
                            edited = copy.deepcopy(records)
                            rec = next(r for r in edited if r.get("case_id") == cid and r.get("arm") == label)
                            if defect == "record":
                                del rec["rows"]
                            elif defect == "replay":
                                del rec["replay_capture"]
                            else:
                                rec["model_receipt_sha256"] = "0" * 64
                            (sout / "runs.jsonl").write_bytes(b"".join(common.canonical(r) + b"\n" for r in edited))
                        check_withheld()
            for defect in ("partial-ledger", "missing-meta", "repeat-capture", "repeat-collection", "nonfinite"):
                with self.subTest(defect=defect):
                    restore()
                    if defect == "partial-ledger":
                        with (sout / "runs.jsonl").open("ab") as f:
                            f.write(b'{"case_id":')
                    elif defect == "missing-meta":
                        (sout / "run-meta.json").unlink()
                    elif defect.startswith("repeat-"):
                        edited = copy.deepcopy(records)
                        ref = next(r for r in edited if r.get("case_id") == cid and r.get("arm") == "REF")
                        rep = next(r for r in edited if r.get("case_id") == cid and r.get("arm") == "REF@repeat")
                        key = "capture_id" if defect == "repeat-capture" else "collection_id"
                        rep["replay_capture"][key] = ref["replay_capture"][key]
                        (sout / "runs.jsonl").write_bytes(b"".join(common.canonical(r) + b"\n" for r in edited))
                    else:
                        red = sout / "red" / f"{cid}.A.npz"
                        with np.load(red) as loaded:
                            arrays = dict(loaded)
                        arrays["nonfinite_c"][:] = 1
                        np.savez(red, **arrays)
                        marker = sout / "done" / f"{cid}.A"
                        mark = json.loads(marker.read_text())
                        mark["red_sha256"] = common.file_sha(red)
                        marker.write_bytes(common.canonical(mark))
                    check_withheld()
        finally:
            restore()
        path = d / "calibrated.json"
        ev = {"manifest": d / "m", "search_plan": sp, "runs": [sout]}
        with self.assertRaisesRegex(ValueError, "requires replay evidence"):
            plan.calibrate(hp, summary)
        calibrated = plan.calibrate(hp, summary, ev)
        verification = calibrated["calibration_source"]["replay_verification"]
        self.assertEqual(verification, {"policy": plan.VERIFICATION_POLICY,
                         "search_plan_sha256": plan.load(sp)[1],
                         "runs_ledger_sha256": {"search-replay/runs.jsonl": common.file_sha(sout / "runs.jsonl")}})
        # Migration objects are issued by the actual earlier verifier and analyzer.
        from unittest import mock
        old_plan = self._historical_module("ded24908b", "plan")
        old_analyze = self._historical_module("ded24908b", "analyze")
        old_analyze.plan_mod = old_plan
        with mock.patch("tools.qualification.option_qual.analyze", old_analyze):
            old_summary = old_analyze.build_summary(*old_analyze.analyze(d / "m", sp, [sout]))
            old_calibrated = old_plan.calibrate(hp, old_summary, ev)
        old_frozen = old_plan.freeze(old_calibrated)
        self.assertEqual(old_calibrated["calibration_source"]["replay_verification"]["policy"], plan.REPLAY_POLICY)
        migration_path, admitted = d / "migration.json", d / "migration-runs"
        old_plan.save(migration_path, old_frozen)
        with mock.patch.object(run, "plan_mod", old_plan):
            self.assertEqual(self._run(migration_path, admitted, ["--replay", str(hr)]), 0)
        retained = {q.relative_to(admitted): q.read_bytes() for q in admitted.rglob("*") if q.is_file()}
        for historical in (old_calibrated, old_frozen):
            with self.subTest(frozen=historical["frozen"]):
                self.assertIn("replay_verification", plan.check(historical))
                with self.assertRaises(ValueError):
                    plan.freeze(historical)
                source = historical["calibration_source"]
                legacy = {k: v for k, v in source.items() if k != "search_summary"}
                legacy["replay_policy"] = plan.REPLAY_POLICY
                self.assertIn("replay_verification", plan._source_problem(legacy, legacy=True))
                plan.save(migration_path, historical)
                self.assertEqual(self._run(migration_path, d / "migration-fresh", ["--replay", str(hr)]), 1)
                self.assertEqual(self._run(migration_path, admitted, ["--replay", str(hr)]), 1)
                self.assertEqual(retained, {q.relative_to(admitted): q.read_bytes()
                                            for q in admitted.rglob("*") if q.is_file()})
                mc, mr, _, _ = analyze.analyze(d / "m", migration_path, [admitted])
                self.assertEqual(mr["A"]["verdict"], "INVALID")
                self.assertTrue(any("replay_verification" in p for p in mc["global_problems"]))
        self.assertIsNone(plan.check(plan.calibrate(hp, old_summary, ev)))
        # A lacks replay provenance; analysis sees valid B, then the file returns to A.
        # Inject all three reads synchronously: before, RunDir's parse, after.
        from unittest import mock
        ledger_path = sout / "runs.jsonl"
        ledger_b = ledger_path.read_bytes()
        ledger_a = b"".join(common.canonical({k: v for k, v in r.items() if k != "replay_capture"}) + b"\n"
                            for r in records)
        original_read = Path.read_bytes
        observed = []
        def race_read(path):
            if path == ledger_path:
                data = (ledger_a, ledger_b, ledger_a)[len(observed)]
                observed.append(data)
                return data
            return original_read(path)
        with mock.patch.object(Path, "read_bytes", race_read):
            with self.assertRaisesRegex(ValueError, "changed during verification"):
                plan.calibrate(hp, summary, ev)
        self.assertEqual(observed, [ledger_a, ledger_b, ledger_a])
        whitespace_bytes = ledger_b + b" \n  "
        with mock.patch.object(Path, "read_bytes", lambda p: whitespace_bytes if p == ledger_path else original_read(p)):
            verified_whitespace = plan.calibrate(hp, summary, ev)["calibration_source"]["replay_verification"]
        self.assertEqual(verified_whitespace["runs_ledger_sha256"],
                         {"search-replay/runs.jsonl": common.sha(whitespace_bytes)})
        # Keep every byte, including whitespace and an incomplete tail, in the parsed digest.
        tail_bytes = ledger_b + b'  \n{"partial":'
        with mock.patch.object(Path, "read_bytes", lambda p: tail_bytes if p == ledger_path else original_read(p)):
            rd = analyze.RunDir(sout, plan.arms_needed(plan.load(sp)[0]))
        self.assertEqual(rd.runs_bytes, tail_bytes)
        self.assertTrue(any("incomplete record" in p for p in rd.problems))
        unverified = copy.deepcopy(calibrated)
        del unverified["calibration_source"]["replay_verification"]
        with self.assertRaisesRegex(ValueError, "replay_verification"):
            plan.freeze(unverified)
        plan.save(path, plan.freeze(calibrated))
        for changed in (None, "REF", "CTRL"):
            with self.subTest(changed=changed):
                if changed:
                    (d / f"{changed}.receipt").write_bytes((d / "receipt").read_bytes() + b"\n")
                out = d / f"calibrated-{changed}"
                self.assertEqual(self._run(path, out, ["--replay", str(hr)]), 0)
                context, results, _, means = analyze.analyze(d / "m", path, [out])
                if changed:
                    self.assertEqual(results["A"]["verdict"], "INVALID")
                    self.assertTrue(any("calibrated SEARCH receipts" in x for x in context["global_problems"]))
                    self.assertIsNone(means)
                    (d / f"{changed}.receipt").write_bytes((d / "receipt").read_bytes())
                else:
                    self.assertEqual(context["global_problems"], [])
                    self.assertEqual(results["A"]["verdict"], "PASS", results["A"]["notes"])
                    self.assertTrue(context["qualification"])
                    self.assertIsNotNone(means)

    def test_no_override_artifacts_match_original_runner(self):
        import datetime
        import shutil
        import types
        from contextlib import ExitStack
        from unittest import mock
        d = self.dir
        arms_path = d / "arms.json"
        arms = json.loads(arms_path.read_text())
        arms["arms"]["CTRL"]["env"]["NON_ASCII"] = "caf\u00e9"
        arms_path.write_bytes(json.dumps(arms, ensure_ascii=False).encode("utf-8"))
        p, live = self._plan("holdout"), d / "golden-live"
        self.assertEqual(self._run(p, live, ["--keep-logits"]), 0)
        captures = d / "golden-captures"
        self._capture_fixture(live, captures, {n: common.file_sha(d / "model.gguf")
                                             for n in ("REF", "CTRL", "A", "B", "AB")})
        baseline_plan = self._historical_module("cd13ad369", "plan")
        baseline_analyze = self._historical_module("cd13ad369", "analyze")
        ss = baseline_analyze.write(d / "baseline-search-receipts",
                                    *baseline_analyze.analyze(d / "m", p, [live]))
        ss["split"] = "search"
        ss["self_sha256"] = common.sha(common.canonical({k: v for k, v in ss.items() if k != "self_sha256"}))
        hp = plan.init(d / "m", "holdout", "REF", "CTRL", ["A", "B"],
                       [{"name": "AB", "members": ["A", "B"]}], [256, 512], bootstrap=500, min_parents=2,
                       repeat_cases=["en-02.L256", "en-05.L256"])
        baseline_plan.save(p, baseline_plan.freeze(baseline_plan.calibrate(hp, ss)))
        baseline_run = self._historical_module("cd13ad369", "run")
        baseline_analyze = self._historical_module("cd13ad369", "analyze")
        baseline_evidence = self._historical_module("cd13ad369", "evidence")
        baseline_receipt = self._historical_module("cd13ad369", "receipt")
        baseline_run.evidence = baseline_analyze.evidence = baseline_evidence
        baseline_analyze.render_md = baseline_receipt.render_md
        baseline_analyze.render_summary_md = baseline_receipt.render_summary_md
        class FixedDate(datetime.datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 1, 1, tzinfo=tz)
        for replay in (False, True):
            expected = None
            retained = None
            for phase, rm, am in (("fresh", baseline_run, baseline_analyze), ("fresh", run, analyze),
                                 ("resume", baseline_run, baseline_analyze), ("resume", run, analyze)):
                if phase == "resume":
                    for name, data in retained.items():
                        q = d / name
                        q.parent.mkdir(parents=True, exist_ok=True)
                        q.write_bytes(data)
                    if rm is baseline_run:
                        expected = None
                out, receipts = d / "golden-output", d / "golden-receipts"
                with ExitStack() as stack:
                    for mod in (rm, am):
                        stack.enter_context(mock.patch.object(mod, "kit_commit", lambda: "fixed-commit"))
                    stack.enter_context(mock.patch.object(rm.datetime, "datetime", FixedDate))
                    stack.enter_context(mock.patch.object(rm.uuid, "uuid4", lambda: types.SimpleNamespace(hex="a" * 32)))
                    runner_class = rm.Runner
                    stack.enter_context(mock.patch.object(run, "Runner", lambda a: runner_class(a, now=lambda: 0.)))
                    extra = ["--no-effective-config", "--keep-logits"]
                    if replay:
                        extra += ["--replay", str(captures)]
                    self.assertEqual(self._run(p, out, extra), 0)
                    ctx, res, ctrl, means = am.analyze(d / "m", p, [out])
                    self.assertEqual(ctx["global_problems"], [])
                    self.assertEqual(res["A"]["verdict"], "PASS")
                    am.write(receipts, ctx, res, ctrl, means)
                summary = json.loads((receipts / "summary.json").read_text())
                self.assertFalse(plan.has_model_overrides(summary))
                baseline_plan = self._historical_module("cd13ad369", "plan")
                for split in ("search", "holdout"):
                    pp = plan.init(d / "m", split, "REF", "CTRL", ["A", "B"],
                                   [{"name": "AB", "members": ["A", "B"]}], [256, 512])
                    self.assertFalse(plan.has_model_overrides(pp, "plan"))
                    ss = dict(summary, split="search")
                    ss["self_sha256"] = common.sha(common.canonical({k: v for k, v in ss.items() if k != "self_sha256"}))
                    current = plan.freeze(plan.calibrate(pp, ss), now=FixedDate(2026, 1, 1))
                    previous = baseline_plan.freeze(baseline_plan.calibrate(pp, ss), now=FixedDate(2026, 1, 1))
                    self.assertFalse(plan.has_model_overrides(current, "plan"))
                    self.assertFalse(plan.has_model_overrides(previous, "plan"))
                    self.assertEqual(common.canonical(current), common.canonical(previous))
                artifacts = {str(q.relative_to(d)): q.read_bytes()
                             for root in (out, receipts) for q in root.rglob("*") if q.is_file()}
                # Include every emitted artifact, especially both metadata journals and the repeat.
                self.assertIn("golden-output/run-meta.json", {k.replace("\\", "/") for k in artifacts})
                self.assertTrue(list((out / "done").glob("*.REF@repeat")))
                if phase == "fresh" and rm is baseline_run:
                    retained = artifacts
                self.assertNotIn("replay_policy", json.loads((receipts / "summary.json").read_text()))
                if expected is None:
                    expected = artifacts
                else:
                    self.assertEqual(set(expected), set(artifacts))
                    for name in expected:
                        with self.subTest(replay=replay, phase=phase, artifact=name):
                            self.assertEqual(expected[name], artifacts[name])
                shutil.rmtree(out)
                shutil.rmtree(receipts)

    def test_reserved_labels_match_original_admission(self):
        import datetime
        d = self.dir
        src = _sources(d / "named-src", strata=("model",), parents=6)
        manifest.build(src, d / "named-m", FakeTokenizer(), [256], cont=24, salt="names", search_per_stratum=2)
        # Use the normal runner helper with the named manifest.
        import shutil
        shutil.rmtree(d / "m")
        shutil.copytree(d / "named-m", d / "m")
        baseline = self._historical_module("cd13ad369", "plan")
        for arm in ("model", "model_receipt"):
            with self.subTest(arm=arm):
                arms = {"REF": {}, "CTRL": {"env": {"FAKE_NOISE": "0.02"}}, arm: {}}
                (d / "arms.json").write_text(json.dumps({"arms": arms}))
                sp = plan.init(d / "m", "search", "REF", "CTRL", [arm], [], [256],
                               bootstrap=100, min_parents=2, routes={arm: "Qwen"})
                path = d / (arm + ".json")
                plan.save(path, sp)
                out = d / (arm + "-runs")
                self.assertEqual(self._run(path, out), 0)
                ctx, res, ctrl, means = analyze.analyze(d / "m", path, [out])
                self.assertEqual(ctx["global_problems"], [])
                self.assertIn("model", ctx["parents_per_stratum"])
                self.assertIn("model", ctrl["by_stratum"])
                self.assertIn(arm, ctx["effective_config"])
                summary = analyze.build_summary(ctx, res, ctrl, means)
                hp = plan.init(d / "m", "holdout", "REF", "CTRL", [arm], [], [256],
                               bootstrap=100, min_parents=2, routes={arm: "Qwen"})
                for deferred in (False, True):
                    with self.subTest(deferred=deferred):
                        ss = copy.deepcopy(summary)
                        if deferred:
                            ss["effective_config"][arm] = None
                        ss["self_sha256"] = common.sha(common.canonical({k: v for k, v in ss.items()
                                                                                       if k != "self_sha256"}))
                        now = datetime.datetime(2026, 1, 1)
                        current = plan.freeze(plan.calibrate(hp, ss), now=now)
                        frozen = baseline.freeze(baseline.calibrate(hp, ss), now=now)
                        self.assertEqual(common.canonical(current), common.canonical(frozen))
                        for pp in (current, frozen):
                            self.assertFalse(plan.has_model_overrides(pp, "plan"))
                            self.assertIsNone(plan.check(pp))
                            self.assertIsNone(plan.unresolved(pp))
                            plan.require_runnable(pp, manifest.load(d / "m")[1])

    def test_captured_reader_encoding_matches_original(self):
        from tools.qualification.option_qual import reduce
        old_run = self._historical_module("cd13ad369", "run")
        old_analyze = self._historical_module("cd13ad369", "analyze")
        old_plan = self._historical_module("cd13ad369", "plan")
        old_reduce = self._historical_module("cd13ad369", "reduce")
        d = self.dir / "encoding"
        d.mkdir()
        path = d / "arms.json"
        stem = d / "reduction"
        reduce.save(stem, {"x": np.array([1.])}, {})
        document = '{"arms": {"A": {"env": {"VALUE": "caf\u00e9"}}}}\r\n'
        def outcome(read):
            try:
                return ("accepted", read())
            except (ValueError, UnicodeError) as exc:
                return ("refused", type(exc).__name__)
        buffers = [document.encode(enc) for enc in
                   ("cp1252", "utf-8", "utf-8-sig", "utf-16", "utf-32")]
        for data in buffers:
            with self.subTest(data=data[:12]):
                path.write_bytes(data)
                meta = {k: "a" * 64 for k in old_analyze.IDENTITY}
                meta["arms_sha256"] = common.sha(data)
                (d / "run-meta.json").write_bytes(common.canonical(meta) + b"\n")
                Path(str(stem) + ".json").write_bytes(data)
                for previous, current in (
                    (lambda: old_run.load_arms(path, ["A"]), lambda: run.load_arms(path, ["A"])),
                    (lambda: old_analyze.RunDir(d).arms, lambda: analyze.RunDir(d, ["A"]).arms),
                    (lambda: old_plan.load(path), lambda: plan.load(path)),
                    (lambda: old_reduce.load(stem)[1], lambda: reduce.load(stem)[1]),
                ):
                    self.assertEqual(outcome(previous), outcome(current))
                self.assertEqual(outcome(lambda: old_run.load_arms(path, ["A"])),
                                 outcome(lambda: run.load_arms(path, ["A"], data=data)))

    def test_label_maps_inspect_values_and_unknown_wrappers(self):
        # Every actual label map inspects its values; unknown placements stay generic.
        for schema, fields in plan._LABEL_MAPS.items():
            for field in fields:
                for label in ("model", "model_receipt", "arm_model_receipt_sha256"):
                    with self.subTest(schema=schema, field=field, label=label):
                        self.assertFalse(plan.has_model_overrides({field: {label: None}}, schema))
                        self.assertTrue(plan.has_model_overrides(
                            {field: {label: {"wrapper": [{"model": "x"}]}}}, schema))
                        self.assertTrue(plan.has_model_overrides({"unknown": {field: {label: "x"}}}, schema))
        self.assertTrue(plan.has_model_overrides({"by_stratum": {"model": "unverified.gguf"}}))
        pp = plan.load(self._plan("holdout"))[0]
        pp["witness"] = {"A": {"stdout": {"model": {"model_receipt": "expected data"}}}}
        self.assertIsNone(plan.check(pp))
        self.assertFalse(plan.has_model_overrides(pp, "plan"))
        self.assertTrue(plan.has_model_overrides({"unknown": pp}, "plan"))

    def test_reduction_hash_load_single_capture(self):
        from unittest import mock
        from tools.qualification.option_qual import reduce
        d = self.dir
        (d / "red").mkdir()
        stem = d / "red" / "case.CTRL"
        npz, js = Path(str(stem) + ".npz"), Path(str(stem) + ".json")
        reduce.save(stem, {"overlap": np.array([1.0])}, {"rows": 1})
        captured = {p: p.read_bytes() for p in (npz, js)}
        original_read = Path.read_bytes
        reads = []
        def swap_read(path):
            data = original_read(path)
            if path in captured:
                reads.append(path)
                # Swap immediately after the verified bytes are read, before parsing.
                path.write_bytes(b"invalid replacement")
            return data
        rd = object.__new__(analyze.RunDir)
        rd.path = d
        with mock.patch.object(Path, "read_bytes", swap_read):
            arrays, info = rd.verified("case", "CTRL", common.sha(captured[npz]), common.sha(captured[js]))
        self.assertEqual(reads, [npz, js])
        np.testing.assert_array_equal(arrays["overlap"], [1.0])
        self.assertEqual(info, {"rows": 1})
        self.assertIsNone(rd.verified("case", "CTRL", common.sha(captured[npz]), common.sha(captured[js])))

    def test_recursive_override_source_placements(self):
        d = self.dir
        sp = self._plan("search", frozen=False)
        out = d / "search"
        self.assertEqual(self._run(sp, out), 0)
        summary = analyze.build_summary(*analyze.analyze(d / "m", sp, [out]))
        hp = plan.init(d / "m", "holdout", "REF", "CTRL", ["A", "B"],
                       [{"name": "AB", "members": ["A", "B"]}], [256, 512])
        baseline_plan = self._historical_module("cd13ad369", "plan")
        self.assertFalse(plan.has_model_overrides(summary))
        self.assertEqual(common.canonical(plan.calibrate(hp, summary)),
                         common.canonical(baseline_plan.calibrate(hp, summary)))
        # Arms parsing and SEARCH-plan identity use their captured bytes even if the
        # backing file changes immediately after the read. Public identity stays canonical.
        from unittest import mock
        inputs = {sp: sp.read_bytes(), out / "arms.json": (out / "arms.json").read_bytes()}
        reads = []
        original_read = Path.read_bytes
        def swap_inputs(path):
            data = original_read(path)
            if path in inputs:
                reads.append(path)
                path.write_bytes(b"invalid replacement")
            return data
        try:
            with mock.patch.object(Path, "read_bytes", swap_inputs):
                rebuilt = analyze.build_summary(*analyze.analyze(d / "m", sp, [out]))
            self.assertEqual(rebuilt, summary)
            self.assertCountEqual(reads, list(inputs))
        finally:
            for path, data in inputs.items():
                path.write_bytes(data)
        placements = [
            {"by_stratum": {"model": "unverified.gguf"}},
            {"model": "x"}, {"model_receipt": "x"},
            {"reference_def": {"wrapper": {"model": "x"}}},
            {"control_def": {"wrapper": {"model_receipt": "x"}}},
            {"wrapper": {"arm_model_receipt_sha256": {}}},
            {"wrapper": [0, {"nested": [0, {"model": "x"}]}]},
        ]
        for placement in placements:
            with self.subTest(placement=placement):
                bad = copy.deepcopy(summary)
                bad.update(placement)
                bad["replay_policy"] = plan.REPLAY_POLICY
                bad["self_sha256"] = common.sha(common.canonical({k: v for k, v in bad.items() if k != "self_sha256"}))
                with self.assertRaisesRegex(ValueError, "requires replay evidence"):
                    plan.calibrate(hp, bad)
                source = baseline_plan.calibrate(hp, bad)["calibration_source"]
                self.assertIn("replay_verification", plan._source_problem(source))
                embedded = dict(hp, calibration_source=source)
                self.assertIn("replay_verification", plan.check(embedded))
                with self.assertRaisesRegex(ValueError, "replay_verification"):
                    plan.freeze(embedded)
                legacy = {k: v for k, v in source.items() if k != "search_summary"}
                legacy.update(placement)
                legacy["replay_policy"] = plan.REPLAY_POLICY
                self.assertIn("replay_verification", plan._legacy_source_problem(legacy))
                self.assertIn("replay_verification", plan._source_problem(legacy, legacy=True))
                lp = dict(embedded, split="search", calibration_source=legacy)
                del lp["tier"]
                self.assertIn("replay_verification", plan.check(lp))
                with self.assertRaisesRegex(ValueError, "replay_verification"):
                    plan.freeze(lp)
        payload = {"model": "model_receipt", "nested": [{"arm_model_receipt_sha256": "x"}]}
        accepted = copy.deepcopy(summary)
        accepted["reference_def"] = {"env": payload, "argv": [payload, "model"]}
        accepted["self_sha256"] = common.sha(common.canonical({k: v for k, v in accepted.items() if k != "self_sha256"}))
        self.assertFalse(plan.has_model_overrides(accepted))
        calibrated = plan.calibrate(hp, accepted)
        self.assertIsNone(plan._source_problem(calibrated["calibration_source"]))
        self.assertIsNone(plan.check(plan.freeze(calibrated)))
        legacy = {k: v for k, v in calibrated["calibration_source"].items() if k != "search_summary"}
        self.assertIsNone(plan._legacy_source_problem(legacy))
        # The old CLI remains sufficient and irrelevant evidence flags are never read.
        summary_path, dest = d / "summary.json", d / "cli.json"
        summary_path.write_bytes(common.canonical(summary))
        for extra in ([], ["--search-runs", "missing", "--search-plan", "missing", "--manifest", "missing"]):
            plan.main(["calibrate", str(self._plan("holdout", frozen=False, name="cli-plan.json")),
                       "--search-summary", str(summary_path), "--out", str(dest), *extra])
            self.assertNotIn("replay_verification", plan.load(dest)[0]["calibration_source"])

    def test_direct_model_overrides_refused_at_start(self):
        import contextlib, io
        d = self.dir
        p = self._plan("search")
        original = json.loads((d / "arms.json").read_text())
        for name in original["arms"]:
            arms = json.loads(json.dumps(original))
            arms["arms"][name].update(model=str(d / "model.gguf"), model_receipt=str(d / "missing.receipt"))
            (d / "arms.json").write_text(json.dumps(arms))
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(self._run(p, d / name), 1)
            self.assertIn("per-arm model overrides require --replay", err.getvalue())
            self.assertFalse((d / name / "runs.jsonl").exists())

    def test_legacy_mixed_resume_and_malformed_maps(self):
        import shutil
        d = self.dir
        p, live = self._plan("search"), d / "live"
        self.assertEqual(self._run(p, live), 0)
        original = (live / "run-meta.json").read_text()
        # One untouched legacy-shaped directory and one resumed all-default directory,
        # with disjoint cases and a common plan/arms/binary/default receipt.
        legacy, resumed = d / "legacy", d / "resumed"
        shutil.copytree(live, legacy)
        shutil.copytree(live, resumed)
        metas = [json.loads(line) for line in original.splitlines()]
        for m in metas:
            m.pop("arm_model_receipt_sha256", None)
        (legacy / "run-meta.json").write_text("".join(json.dumps(m) + "\n" for m in metas))
        explicit = dict(metas[0], arm_model_receipt_sha256={
            n: common.file_sha(d / "receipt") for n in ("REF", "CTRL", "A", "B", "AB")})
        (resumed / "run-meta.json").write_text(json.dumps(explicit) + "\n")
        cases = sorted({r["case_id"] for r in common.read_journal(live / "runs.jsonl")[0]})
        for directory, keep in ((legacy, set(cases[::2])), (resumed, set(cases[1::2]))):
            records = [r for r in common.read_journal(directory / "runs.jsonl")[0] if r["case_id"] in keep]
            (directory / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            for mark in (directory / "done").iterdir():
                if mark.name.rsplit(".", 1)[0] not in keep:
                    mark.unlink()
        ctx, results, ctrl, means = analyze.analyze(d / "m", p, [legacy, resumed])
        self.assertEqual(ctx["global_problems"], [])
        self.assertEqual(results["A"]["verdict"], "PASS")
        analyze.write(d / "receipts", ctx, results, ctrl, means)
        self.assertNotIn("arm_model_receipt_sha256", json.loads((d / "receipts" / "A.receipt.json").read_text()))
        self.assertNotIn("Model receipt sha256:", (d / "receipts" / "A.receipt.md").read_text())
        # Default-only resume omits the map; explicit all-default maps remain accepted.
        self.assertEqual(self._run(p, legacy, ["--cases", ",".join(cases[::2])]), 0)
        self.assertEqual(analyze.analyze(d / "m", p, [legacy, resumed])[0]["global_problems"], [])
        for bad in (None, {}, "oops", {"REF": "a" * 64},
                    {n: None for n in ("REF", "CTRL", "A", "B", "AB")},
                    {n: "z" * 64 for n in ("REF", "CTRL", "A", "B", "AB")}):
            with self.subTest(bad=bad):
                m = json.loads(original.splitlines()[0])
                m["arm_model_receipt_sha256"] = bad
                (live / "run-meta.json").write_text(json.dumps(m) + "\n")
                ctx, results, _, _ = analyze.analyze(d / "m", p, [live])
                self.assertEqual(results["A"]["verdict"], "INVALID")
                self.assertTrue(any("arm_model_receipt_sha256 provenance" in x for x in ctx["global_problems"]))
                self.assertEqual(self._run(p, live), 1)

    def test_replay_of_a_many_case_engine_run(self):
        """--replay takes each (case, arm) from an engine's output directory and gives the verdicts of a live run."""
        import hashlib, shutil
        p = self._plan("holdout")
        live, R = self.dir / "live", self.dir / "replay-src"
        self.assertEqual(self._run(p, live, ["--keep-logits"]), 0)
        R.mkdir()
        cases = {c["case_id"]: c for c in json.loads((self.dir / "m" / "manifest.json").read_text())["cases"]}

        def fsha(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        binary = fsha(self.dir / "fake.py")
        model = (self.dir / "receipt").read_text().split()[0]
        journal = []
        for n, out in enumerate(sorted((live / "log").glob("*.stdout"))):
            cid, lab = out.name[:-len(".stdout")].rsplit(".", 1)
            res = json.loads([l for l in out.read_text().splitlines() if l.startswith("{")][-1])
            c = cases[cid]
            res.update(case_id=cid, prompt_sha256=c["prompt_sha256"], cont_sha256=c["cont_sha256"])
            (R / f"{cid}.{lab}.json").write_text(json.dumps(res))
            shutil.copyfile(live / "log" / f"{cid}.{lab}.stderr", R / f"{cid}.{lab}.stderr")
            shutil.copyfile(live / "logits" / f"{cid}.{lab}.bin", R / f"{cid}.{lab}.bin")
            journal.append(dict(status="complete", capture_id=f"cap-{n}",
                                collection_id="col-repeat" if lab.endswith("@repeat") else "col-main",
                                case_id=cid, label=lab, binary_sha256=binary, model_sha256=model,
                                prompt_sha256=c["prompt_sha256"],
                                cont_sha256=c["cont_sha256"], json_sha256=fsha(R / f"{cid}.{lab}.json"),
                                bin_sha256=fsha(R / f"{cid}.{lab}.bin"), stderr_sha256=fsha(R / f"{cid}.{lab}.stderr")))
        (R / "runs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in journal))
        rep = self.dir / "rep"
        self.assertEqual(self._run(p, rep, ["--replay", str(R)]), 0)
        _, want, _, _ = analyze.analyze(self.dir / "m", p, [live])
        _, got, _, _ = analyze.analyze(self.dir / "m", p, [rep])
        for opt in ("A", "B", "AB"):
            self.assertEqual(got[opt]["verdict"], want[opt]["verdict"], (opt, got[opt]["notes"]))
            self.assertEqual(json.dumps(got[opt].get("endpoints"), sort_keys=True, default=str),
                             json.dumps(want[opt].get("endpoints"), sort_keys=True, default=str))
        meta = json.loads((rep / "run-meta.json").read_text().splitlines()[0])
        self.assertEqual(meta["replay_dir"], str(R.resolve()))
        victim = sorted(R.glob("*.A.json"))[0]
        vcase = victim.name[:-len(".A.json")]

        def broken(name, mutate, extra=()):
            src, out = self.dir / f"replay-{name}", self.dir / f"rep-{name}"
            shutil.copytree(R, src)
            mutate(src)
            rc = self._run(p, out, ["--replay", str(src), *extra])
            _, res, _, _ = analyze.analyze(self.dir / "m", p, [out]) if (out / "runs.jsonl").exists() else (0, None, 0, 0)
            return rc, res
        # a result that names another case is a failed attempt, never evidence
        def other_case(src):
            f = src / victim.name; r = json.loads(f.read_text()); r["case_id"] = "someone-else"; f.write_text(json.dumps(r))
        rc, res = broken("case", other_case)
        self.assertEqual(rc, 1); self.assertNotEqual(res["A"]["verdict"], "PASS")
        # the option's logits replaced by the reference's: bytes differ from the committed entry
        rc, res = broken("swap", lambda src: shutil.copyfile(src / f"{vcase}.REF.bin", src / f"{vcase}.A.bin"))
        self.assertEqual(rc, 1); self.assertNotEqual(res["A"]["verdict"], "PASS")
        # the repeat folded onto the reference capture is not an independent run
        rc, res = broken("alias", lambda src: None, ["--replay-label", "REF@repeat=REF"])
        self.assertEqual(rc, 1); self.assertNotEqual(res["A"]["verdict"], "PASS")
        # a capture made by another binary
        def other_binary(src):
            lines = [json.loads(x) for x in (src / "runs.jsonl").read_text().splitlines()]
            for e in lines:
                if e["case_id"] == vcase and e["label"] == "A":
                    e["binary_sha256"] = "0" * 64
            (src / "runs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in lines))
        rc, res = broken("binary", other_binary)
        self.assertEqual(rc, 1); self.assertNotEqual(res["A"]["verdict"], "PASS")
        # a capture that names another model
        def other_model(src):
            lines = [json.loads(x) for x in (src / "runs.jsonl").read_text().splitlines()]
            for e in lines:
                if e["case_id"] == vcase and e["label"] == "A":
                    e["model_sha256"] = "f" * 64
            (src / "runs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in lines))
        rc, res = broken("model", other_model)
        self.assertEqual(rc, 1); self.assertNotEqual(res["A"]["verdict"], "PASS")
        # malformed producer model identities: missing, null, empty, not a string
        for n, value in enumerate((None, "", 7, "MISSING")):
            def bad_model(src, value=value):
                lines = [json.loads(x) for x in (src / "runs.jsonl").read_text().splitlines()]
                for e in lines:
                    if e["case_id"] == vcase and e["label"] == "A":
                        if value == "MISSING":
                            del e["model_sha256"]
                        else:
                            e["model_sha256"] = value
                (src / "runs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in lines))
            rc, res = broken(f"model{n}", bad_model)
            self.assertEqual(rc, 1, value); self.assertNotEqual(res["A"]["verdict"], "PASS", value)
        # a receipt that names no model: refused before any capture is imported
        for n, text in enumerate(("", "   \n\t")):
            receipt, out = self.dir / f"receipt-empty-{n}", self.dir / f"rep-empty-{n}"
            receipt.write_text(text)
            rc = run.main(["--manifest", str(self.dir / "m"), "--plan", str(p), "--arms", str(self.dir / "arms.json"),
                           "--binary", str(self.dir / "fake.py"), "--model", str(self.dir / "model.gguf"),
                           "--model-receipt", str(receipt), "--out", str(out), "--soft-seconds", "600",
                           "--allow-fake-for-tests", "--python-binary-for-tests", "--replay", str(R)])
            self.assertEqual(rc, 1)
            self.assertFalse((out / "runs.jsonl").exists() and (out / "runs.jsonl").read_text().strip())
        # no producer journal at all: refused before anything is imported
        rc, _ = broken("nojournal", lambda src: (src / "runs.jsonl").unlink())
        self.assertEqual(rc, 1)
        # a pair without engine output is not run: resumable (exit 2), INCOMPLETE, never FAIL
        R3, rep3 = self.dir / "replay-gap", self.dir / "rep3"
        shutil.copytree(R, R3)
        lines = [x for x in (R3 / "runs.jsonl").read_text().splitlines()
                 if not (json.loads(x)["case_id"] == vcase and json.loads(x)["label"] == "A")]
        (R3 / "runs.jsonl").write_text("".join(x + "\n" for x in lines))     # not committed yet
        self.assertEqual(self._run(p, rep3, ["--replay", str(R3)]), 2)
        _, gap, _, _ = analyze.analyze(self.dir / "m", p, [rep3])
        self.assertEqual(gap["A"]["verdict"], "INCOMPLETE", gap["A"]["notes"])

    def test_holdout_verdicts(self):
        p = self._plan("holdout")
        out = self.dir / "out"
        self.assertEqual(self._run(p, out), 0)
        self.assertFalse(list((out / "logits").glob("*.bin")))          # logits removed after reduction
        self.assertTrue((out / "holdout-ledger.jsonl").exists())
        ctx, results, _, means = analyze.analyze(self.dir / "m", p, [out])
        self.assertTrue(ctx["qualification"])
        self.assertEqual(results["A"]["verdict"], "PASS", results["A"])
        self.assertEqual(results["B"]["verdict"], "FAIL")
        self.assertEqual(results["AB"]["verdict"], "FAIL")
        self.assertEqual(ctx["k"], 3)
        s = analyze.write(self.dir / "rc", ctx, results, None, means)
        self.assertEqual(json.loads((self.dir / "rc" / "A.receipt.json").read_text())["verdict"], "PASS")
        self.assertIn("A: PASS", (self.dir / "rc" / "A.receipt.md").read_text())
        self.assertEqual(s["verdicts"][0]["option"], "A")
        # rerun is a no-op (resumable)
        self.assertEqual(self._run(p, out), 0)

        # a run stamped with another plan sha makes everything INVALID
        runs = out / "runs.jsonl"
        lines = runs.read_text().splitlines()
        rec = json.loads(lines[0])
        rec["plan_sha256"] = "0" * 64
        runs.write_text("\n".join([json.dumps(rec)] + lines[1:]) + "\n")
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "INVALID")
        runs.write_text("\n".join(lines) + "\n")

        # a missing reduction makes the option INCOMPLETE
        victim = next((out / "done").glob("*.A"))
        victim.unlink()
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "INCOMPLETE")

    def test_run_witness_misses_are_incomplete_never_fail(self):
        d = self.dir
        arms = {"REF": {}, "CTRL": {"env": {"FAKE_NOISE": "0.02"}},
                "A": {"env": {"FAKE_NOISE": "0.005", "FAKE_LEVER_ON": "1"}},   # lever engaged
                "B": {"env": {"FAKE_NOISE": "0.005"}},                         # lever silently declined
                "AB": {"env": {"FAKE_NOISE": "1.5"}}}
        (d / "arms-w.json").write_text(json.dumps({"arms": arms}))
        witness = {"A": {"stdout": {"lever": True}, "stderr_require": ["reason=test"]},
                   "B": {"stdout": {"lever": True}}, "REF": {"stdout": {"lever": False}}}
        pl = plan.init(d / "m", "holdout", "REF", "CTRL", ["A", "B"], [{"name": "AB", "members": ["A", "B"]}],
                       [256, 512], bootstrap=500, min_parents=2, repeat_cases=["en-02.L256", "en-05.L256"],
                       witness=witness)
        for e, m in (("E2", 0.01), ("E3", 0.01), ("E4", 0.05)):
            pl["endpoints"][e] = {"fixed": m}
        pl = plan.freeze(pl)
        plan.save(d / "plan-w.json", pl)
        out = d / "out-w"
        argv = ["--manifest", str(d / "m"), "--plan", str(d / "plan-w.json"), "--arms", str(d / "arms-w.json"),
                "--binary", str(d / "fake.py"), "--model", str(d / "model.gguf"), "--model-receipt",
                str(d / "receipt"), "--out", str(out), "--soft-seconds", "600", "--allow-fake-for-tests",
                "--python-binary-for-tests"]
        self.assertEqual(run.main(argv), 1)          # B's runs fail their witness (a failed attempt each)
        self.assertEqual(run.main(argv), 1)
        recs = [json.loads(x) for x in (out / "runs.jsonl").read_text().splitlines() if '"event"' not in x]
        b = [r for r in recs if r["arm"] == "B"]
        self.assertTrue(b and all((r["problem"] or "").startswith("witness: result lever = false") for r in b), b[:1])
        self.assertTrue(all(r["complete"] for r in recs if r["arm"] == "A"))
        _, results, _, _ = analyze.analyze(d / "m", d / "plan-w.json", [out])
        self.assertEqual(results["A"]["verdict"], "PASS", results["A"]["notes"])
        self.assertEqual(results["B"]["verdict"], "INCOMPLETE", results["B"]["notes"])
        self.assertTrue(any("witness" in n for n in results["B"]["notes"]))
        # a lost terminal marker restored from the journal stays INCOMPLETE (classified from the run records)
        for m in (out / "done").glob("*.B"):
            m.unlink()
        run.main(argv)
        self.assertTrue(any("restored" in m.read_text() for m in (out / "done").glob("*.B")))
        _, results, _, _ = analyze.analyze(d / "m", d / "plan-w.json", [out])
        self.assertEqual(results["B"]["verdict"], "INCOMPLETE", results["B"]["notes"])
        self.assertEqual(results["A"]["verdict"], "PASS", results["A"]["notes"])
        # the primary reference's logs are checked too: replaced bytes (even still carrying the witness) are INVALID
        r_case = next(x.name for x in (out / "done").glob("*.A"))[:-2]
        ref_log = out / "log" / f"{r_case}.REF.stdout"
        kept = ref_log.read_bytes()
        ref_log.write_bytes(kept + b" ")
        _, results, _, _ = analyze.analyze(d / "m", d / "plan-w.json", [out])
        self.assertEqual(results["A"]["verdict"], "INVALID", results["A"]["notes"])
        ref_log.write_bytes(kept)
        # a kept candidate run whose logs were replaced is INVALID, not evidence - even if they show the witness
        a_err = out / "log" / f"{r_case}.A.stderr"
        a_err.write_bytes(a_err.read_bytes() + b"reason=test\n")
        _, results, _, _ = analyze.analyze(d / "m", d / "plan-w.json", [out])
        self.assertEqual(results["A"]["verdict"], "INVALID")

    def test_witness_misses_never_count_toward_fail(self):
        d = self.dir
        lever, trunc = d / "lever-on", d / "truncate"
        arms = {"REF": {}, "CTRL": {"env": {"FAKE_NOISE": "0.02"}},
                "A": {"env": {"FAKE_NOISE": "0.005", "FAKE_LEVER_FILE": str(lever),
                              "FAKE_TRUNCATE_FILE": str(trunc)}}}
        (d / "arms-m.json").write_text(json.dumps({"arms": arms}))
        pl = plan.init(d / "m", "holdout", "REF", "CTRL", ["A"], [], [256], bootstrap=500, min_parents=2,
                       witness={"A": {"stdout": {"lever": True}}})
        for e, m in (("E2", 0.01), ("E3", 0.01), ("E4", 0.05)):
            pl["endpoints"][e] = {"fixed": m}
        plan.save(d / "plan-m.json", plan.freeze(pl))
        out = d / "out-m"
        argv = ["--manifest", str(d / "m"), "--plan", str(d / "plan-m.json"), "--arms", str(d / "arms-m.json"),
                "--binary", str(d / "fake.py"), "--model", str(d / "model.gguf"), "--model-receipt",
                str(d / "receipt"), "--out", str(out), "--soft-seconds", "600", "--allow-fake-for-tests",
                "--python-binary-for-tests"]
        self.assertEqual(run.main(argv), 1)          # attempt 1 of every A run: witness miss
        lever.write_text("1")
        trunc.write_text("1")
        self.assertEqual(run.main(argv), 1)          # attempt 2: an ordinary failure; the attempts are used up
        recs = [json.loads(x) for x in (out / "runs.jsonl").read_text().splitlines() if '"event"' not in x]
        probs = [r["problem"] or "" for r in recs if r["arm"] == "A"]
        self.assertTrue(any(x.startswith("witness:") for x in probs) and any(x.startswith("forced") for x in probs))
        _, results, _, _ = analyze.analyze(d / "m", d / "plan-m.json", [out])
        self.assertEqual(results["A"]["verdict"], "INCOMPLETE", results["A"]["notes"])

    def test_witness_plan_validation(self):
        base = dict(split="holdout", reference="REF", control="CTRL", options=["A"], compositions=[],
                    lengths=[256], bootstrap=500, min_parents=2)
        bad = [{"Z": {"stdout": {"x": 1}}}, {"A": {}}, {"A": {"stdout": {}}}, {"A": {"stderr_forbid": ["", "x"]}},
               {"A": {"stderr_require": ["(unclosed"]}}, {"A": {"other": 1}}, {}, [], False, 0,
               {"A": {"stderr_require": ["a{4294967296}"]}}]
        for w in bad:
            with self.assertRaises(ValueError, msg=w):
                plan.init(self.dir / "m", **base, witness=w)
        self.assertNotIn("witness", plan.init(self.dir / "m", **base))     # legacy plans unchanged
        # CLI: a given witness file is always validated (null included); omitting the flag gives no witness
        cli = ["init", "--manifest", str(self.dir / "m"), "--split", "holdout", "--reference", "REF", "--control",
               "CTRL", "--options", "A", "--lengths", "256", "--bootstrap", "500", "--min-parents-per-stratum", "2"]
        for doc in ("null", "{}", "false"):
            (self.dir / "w.json").write_text(doc)
            with self.assertRaises(ValueError, msg=doc):
                plan.main(cli + ["--witness", str(self.dir / "w.json"), "--out", str(self.dir / "pw.json")])
            self.assertFalse((self.dir / "pw.json").exists())
        plan.main(cli + ["--out", str(self.dir / "pn.json")])
        self.assertNotIn("witness", json.loads((self.dir / "pn.json").read_text()))
        self.assertIn("witness", plan.init(self.dir / "m", **base, witness={"A": {"stdout": {"lever": True}}}))

    def test_provenance_control_and_forcing_gates(self):
        p = self._plan("holdout")
        out = self.dir / "g"
        self.assertEqual(self._run(p, out), 0)
        # control results missing for one stratum: no option may pass on the remaining parents
        for marker in (out / "done").glob("code-*.CTRL"):
            marker.unlink()
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertNotEqual(results["A"]["verdict"], "PASS")
        # results without provenance are invalid
        out2 = self.dir / "g2"
        self.assertEqual(self._run(p, out2), 0)
        (out2 / "runs.jsonl").unlink()
        (out2 / "run-meta.json").unlink()
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out2])
        self.assertEqual(results["A"]["verdict"], "INVALID")
        # an arm that ignores teacher forcing (even when it claims otherwise) never completes
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["A"]["env"].update(FAKE_IGNORE_FORCE="1", FAKE_CLAIM_FORCED="1")
        (self.dir / "arms.json").write_text(json.dumps(arms))
        out3 = self.dir / "g3"
        self.assertEqual(self._run(p, out3), 1)
        self.assertEqual(self._run(p, out3, ["--max-attempts", "2"]), 1)
        rec = [json.loads(l) for l in (out3 / "runs.jsonl").read_text().splitlines() if '"arm": "A"' in l]
        self.assertTrue(rec and all(r["problem"] and not r["complete"] for r in rec))
        self.assertFalse(list((out3 / "logits").glob("*.A.bin")))
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out3])
        self.assertEqual(results["A"]["verdict"], "FAIL")
        self.assertEqual(results["B"]["verdict"], "FAIL")       # other options were still run and judged
        # resuming with a changed arms file is refused
        self.assertEqual(self._run(p, out), 1)

    def test_declared_routes(self):
        pl = plan.init(self.dir / "m", "search", "REF", "CTRL", ["A", "B"], [], [256], bootstrap=200, min_parents=2,
                       routes={"A": "T8 context", "B": "native-route"}, endpoints=["E1"])
        plan.save(self.dir / "r.json", pl)
        out = self.dir / "r"
        self.assertEqual(self._run(self.dir / "r.json", out), 0)
        _, results, _, _ = analyze.analyze(self.dir / "m", self.dir / "r.json", [out])
        self.assertFalse(any("route" in n for n in results["A"]["notes"]))
        self.assertTrue(any("declared route" in n for n in results["B"]["notes"]))
        self.assertEqual(results["B"]["verdict"], "FAIL")

    def test_merging_and_borrowed_provenance_are_invalid(self):
        p = self._plan("holdout")
        a, b = self.dir / "ma", self.dir / "mb"
        self.assertEqual(self._run(p, a), 0)
        self.assertEqual(self._run(p, b), 0)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [a])
        self.assertEqual(results["A"]["verdict"], "PASS")
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [a, b])     # the same cases twice
        self.assertEqual(results["A"]["verdict"], "INVALID")
        # provenance of one directory cannot be supplied by another
        (a / "run-meta.json").unlink()
        meta_only = self.dir / "meta-only"
        meta_only.mkdir()
        (meta_only / "run-meta.json").write_text((b / "run-meta.json").read_text())
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [a, meta_only])
        self.assertEqual(results["A"]["verdict"], "INVALID")
        # a reduction not bound to the candidate file of a validated run is invalid
        red = next((b / "red").glob("*.A.json"))
        info = json.loads(red.read_text())
        info["cand_sha256"] = "0" * 64
        red.write_text(json.dumps(info))
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [b])
        self.assertEqual(results["A"]["verdict"], "INVALID")

    def test_swapped_arrays_short_files_and_repeat_resume(self):
        p = self._plan("holdout")
        out = self.dir / "sw"
        self.assertEqual(self._run(p, out), 0)
        # metric arrays of one arm swapped in for another: never a valid result
        for a_npz in (out / "red").glob("*.A.npz"):
            target = Path(str(a_npz).replace(".A.npz", ".B.npz"))
            target.write_bytes(a_npz.read_bytes())
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["B"]["verdict"], "INVALID")
        # an INVALID reference repeat keeps the exit code at 1 on resume
        rep = next((out / "done").glob("*.REF@repeat"))
        mark = json.loads(rep.read_text())
        rep.write_text(json.dumps(dict(mark, status="INVALID", error="injected")))
        self.assertEqual(self._run(p, out), 1)
        self.assertEqual(len((out / "run-meta.json").read_text().splitlines()), 2)   # the resume really ran
        # a one-byte logits file is a recorded failure of that pair only
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["B"]["env"]["FAKE_ONE_BYTE"] = "1"
        (self.dir / "arms.json").write_text(json.dumps(arms))
        out2 = self.dir / "ob"
        self.assertEqual(self._run(p, out2), 1)
        marks = {q.name: json.loads(q.read_text()) for q in (out2 / "done").iterdir()}
        self.assertTrue(marks and all(v["status"] == "FAILED" for k, v in marks.items() if k.endswith(".B")))
        self.assertTrue(all(v["status"] == "COMPLETE" for k, v in marks.items() if k.endswith(".AB")))

    def test_route_from_run_records_and_retained_reference_repair(self):
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["A"]["env"]["FAKE_ROUTE"] = "other"
        (self.dir / "arms.json").write_text(json.dumps(arms))
        p = self._plan("holdout")
        out = self.dir / "rt"
        self.assertEqual(self._run(p, out, ["--keep-logits"]), 0)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "FAIL")
        # editing the route of a done marker, or of the run record, or dropping it: never a PASS
        ref_route = json.loads(next((out / "done").glob("*.REF")).read_text())["route"]
        saved = {m: m.read_text() for m in (out / "done").glob("*.A")}
        for marker in saved:
            marker.write_text(json.dumps(dict(json.loads(marker.read_text()), route=ref_route)))
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertIn(results["A"]["verdict"], ("FAIL", "INVALID"))
        for marker, text in saved.items():
            marker.write_text(text)
        runs = out / "runs.jsonl"
        original = runs.read_text()
        recs = [json.loads(l) for l in original.splitlines()]
        runs.write_text("".join(json.dumps(dict(r, route=ref_route) if r["arm"] == "A" else r) + "\n" for r in recs))
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertIn(results["A"]["verdict"], ("FAIL", "INVALID"))
        runs.write_text("".join(json.dumps({k: v for k, v in r.items() if k != "route"}) + "\n" for r in recs))
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "INVALID")
        # an interrupted append: analysis reports it, the runner quarantines it and the campaign resumes
        runs.write_text(original + '{"case_id": "x", "ar')
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "INVALID")
        self._run(p, out, ["--keep-logits"])
        self.assertIn("terminated_tail_bytes", (out / "recovery.jsonl").read_text())   # kept, never cut
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "FAIL")
        # a retained reference file that contradicts its record makes the case run again
        runs_before = len((out / "runs.jsonl").read_text().splitlines())
        victim = sorted((out / "logits").glob("*.REF.bin"))[0]
        data = bytearray(victim.read_bytes())
        data[-1] ^= 1
        victim.write_bytes(bytes(data))
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["B"]["verdict"], "INVALID")
        self._run(p, out, ["--keep-logits"])
        self.assertGreater(len((out / "runs.jsonl").read_text().splitlines()), runs_before)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["B"]["verdict"], "FAIL")      # valid again, same verdict as before

    def test_metadata_recovery_keeps_ids_unique(self):
        p = self._plan("holdout")
        out = self.dir / "md"
        self.assertEqual(self._run(p, out), 0)
        # an interrupted first metadata append: recovered, resumable, no crash
        meta = out / "run-meta.json"
        whole = meta.read_text()
        meta.write_text(whole[:len(whole) // 2])
        self.assertEqual(self._run(p, out), 0)
        # a lost metadata append while run records survive; another arm must be produced again
        meta.write_text(meta.read_text()[:-5])
        victim = next((out / "done").glob("*.A"))
        victim.unlink()
        self.assertEqual(self._run(p, out), 0)
        ids = [json.loads(l)["run_id"] for l in (out / "runs.jsonl").read_text().splitlines()]
        self.assertEqual(len(ids), len(set(ids)))
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "PASS", results["A"]["notes"][:3])

    def test_damaged_or_foreign_markers_recover(self):
        p = self._plan("holdout")
        out = self.dir / "mk"
        self.assertEqual(self._run(p, out), 0)
        markers = sorted((out / "done").glob("*.A"))
        # a half-transferred marker: analysis reports it, resume quarantines it and redoes the pair
        text = markers[0].read_text()
        markers[0].write_text(text[:len(text) // 2])
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "INVALID")
        self.assertEqual(self._run(p, out), 0)
        self.assertTrue(list((out / "quarantine").iterdir()))
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "PASS")
        # a failure marker copied from another case: never honoured
        foreign = dict(json.loads(markers[1].read_text()), status="FAILED", attempts=2, max_attempts=2)
        markers[0].write_text(json.dumps(foreign))
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "INVALID")
        self.assertEqual(self._run(p, out), 0)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "PASS")

    def test_resume_applies_the_analysis_rules(self):
        p = self._plan("holdout")
        out = self.dir / "rr"
        self.assertEqual(self._run(p, out), 0)
        # a completion marker with another plan sha, or its run record turned into a failed run:
        # analysis rejects it, and resume repairs it instead of trusting it
        marker = sorted((out / "done").glob("*.A"))[0]
        marker.write_text(json.dumps(dict(json.loads(marker.read_text()), plan_sha256="0" * 64)))
        runs = out / "runs.jsonl"
        recs = [json.loads(l) for l in runs.read_text().splitlines()]
        target = json.loads(sorted((out / "done").glob("*.A"))[1].read_text())["run_id"]
        runs.write_text("".join(json.dumps(dict(r, rc=1) if r["run_id"] == target else r) + "\n" for r in recs))
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "INVALID")
        before = len(runs.read_text().splitlines())
        self.assertEqual(self._run(p, out), 0)
        self.assertGreater(len(runs.read_text().splitlines()), before)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "PASS")
        # a stale INVALID marker (made against another reference) is not honoured
        stale = sorted((out / "done").glob("*.A"))[2]
        good = json.loads(stale.read_text())
        stale.write_text(json.dumps(dict(good, status="INVALID", ref_sha256="1" * 64, error="old")))
        self.assertEqual(self._run(p, out), 0)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "PASS")

    def test_damaged_failure_marker_keeps_the_retry_count(self):
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["B"]["env"]["FAKE_TRUNCATE"] = "1"
        (self.dir / "arms.json").write_text(json.dumps(arms))
        p = self._plan("holdout")
        out = self.dir / "fc"
        case = "en-02.L256"
        self.assertEqual(self._run(p, out, ["--cases", case]), 1)                 # B fails once
        marker = out / "done" / f"{case}.B"
        text = marker.read_text()
        marker.write_text(text[:len(text) // 2])                               # damaged in transfer
        self.assertEqual(self._run(p, out, ["--cases", case]), 1)                 # second failure
        b_runs = lambda: sum(1 for l in (out / "runs.jsonl").read_text().splitlines()
                             if json.loads(l)["arm"] == "B")
        self.assertEqual(json.loads(marker.read_text())["attempts"], 2)
        n = b_runs()
        self.assertEqual(self._run(p, out, ["--cases", case]), 1)                 # terminal: no third run
        self.assertEqual(b_runs(), n)

    def test_stale_success_and_journal_repair(self):
        knob = self.dir / "noise.txt"
        knob.write_text("0.005")
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["A"] = {"env": {"FAKE_NOISE_FILE": str(knob)}}
        (self.dir / "arms.json").write_text(json.dumps(arms))
        p = self._plan("holdout")
        out = self.dir / "st"
        self.assertEqual(self._run(p, out), 0)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "PASS")
        # keep one case's successful A artifacts, then redo that case with much worse output
        cases = sorted(m.name[:-2] for m in (out / "done").glob("*.A"))
        files = {}
        for case in cases:
            files[case] = {q: q.read_bytes() for q in [out / "done" / f"{case}.A"] +
                           [out / "red" / f"{case}.A{x}" for x in (".npz", ".json")]}
            (out / "done" / f"{case}.A").unlink()
        knob.write_text("1.5")
        self.assertEqual(self._run(p, out), 0)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "FAIL")
        for saved in files.values():                       # restore the older, favourable artifacts
            for q, data in saved.items():
                q.write_bytes(data)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertNotEqual(results["A"]["verdict"], "PASS")
        self.assertEqual(self._run(p, out), 0)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "FAIL")
        # a malformed historical run record: quarantined on resume, the campaign heals
        runs = out / "runs.jsonl"
        recs = [json.loads(l) for l in runs.read_text().splitlines()]
        target = json.loads(next((out / "done").glob("*.B")).read_text())["run_id"]
        runs.write_text("".join(json.dumps({k: v for k, v in r.items() if not (r["run_id"] == target and k == "rc")})
                                + "\n" for r in recs))
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["B"]["verdict"], "INVALID")
        self.assertEqual(self._run(p, out), 0)
        self.assertTrue((out / "resets.jsonl").exists())              # the case was reset, journal untouched
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["B"]["verdict"], "FAIL")
        self.assertFalse(any("INVALID" in n for n in results["B"]["notes"]))
        # restoring the old favourable A artifacts while damaging the newer record never resurrects them
        for saved in files.values():
            for q, data in saved.items():
                q.write_bytes(data)
        recs = [json.loads(l) for l in runs.read_text().splitlines()]
        newest_a = [r["run_id"] for r in recs if r.get("arm") == "A"][-1]
        runs.write_text("".join(json.dumps(dict(r, rc="x") if r.get("run_id") == newest_a else r) + "\n"
                                for r in recs))
        for _ in range(2):
            self._run(p, out)
            _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
            self.assertNotEqual(results["A"]["verdict"], "PASS")
        # a run record without its timing field cannot crash a resume
        recs = [json.loads(l) for l in runs.read_text().splitlines()]
        first_ok = next(r["run_id"] for r in recs if r.get("complete") and "event" not in r)
        runs.write_text("".join(json.dumps({k: v for k, v in r.items() if not (r.get("run_id") == first_ok
                                                                                and k == "wall_seconds")})
                                + "\n" for r in recs))
        self.assertIn(self._run(p, out), (0, 1))

    def test_unterminated_tail_foreign_resets_and_huge_numbers(self):
        knob = self.dir / "noise.txt"
        knob.write_text("0.005")
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["A"] = {"env": {"FAKE_NOISE_FILE": str(knob)}}
        (self.dir / "arms.json").write_text(json.dumps(arms))
        p = self._plan("holdout")
        out = self.dir / "tl"
        case = "en-02.L256"
        self.assertEqual(self._run(p, out, ["--cases", case]), 0)
        saved = {q: q.read_bytes() for q in [out / "done" / f"{case}.A", out / "red" / f"{case}.A.npz",
                                            out / "red" / f"{case}.A.json"]}
        (out / "done" / f"{case}.A").unlink()
        knob.write_text("1.5")
        self.assertEqual(self._run(p, out, ["--cases", case]), 0)          # newer, much worse A run
        for q, data in saved.items():                                      # older favourable files back
            q.write_bytes(data)
        runs = out / "runs.jsonl"
        data = runs.read_bytes()
        newest_a = [json.loads(l) for l in data.decode().splitlines() if json.loads(l).get("arm") == "A"][-1]
        if data.decode().splitlines()[-1] != json.dumps(newest_a, sort_keys=True):
            runs.write_bytes(data + (json.dumps(newest_a, sort_keys=True)).encode())   # newest A unterminated
        else:
            runs.write_bytes(data[:-1])
        for _ in range(2):
            self._run(p, out, ["--cases", case])
            _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
            self.assertNotEqual(results["A"]["verdict"], "PASS")
            ids = [json.loads(l)["run_id"] for l in runs.read_text().splitlines() if l.strip().startswith("{")]
            self.assertIn(newest_a["run_id"], ids)                         # the journal never loses a line
        # a reset journal from another campaign is not honoured and gets repaired
        resets = out / "resets.jsonl"
        bogus = {"case_id": "*", "upto": 10_000, "line_sha256": "0" * 64, "invocation": "x"}
        with open(resets, "a") as f:
            f.write(json.dumps(bogus) + "\n")
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "INVALID")
        self._run(p, out, ["--cases", case])
        self.assertTrue(list((out / "quarantine").glob("resets.jsonl.*")))
        # an absurd number never crashes the validators
        recs = [l for l in runs.read_text().splitlines() if l.strip()]
        first = json.loads(recs[0])
        first["wall_seconds"] = 10 ** 400
        runs.write_text("\n".join([json.dumps(first)] + recs[1:]) + "\n")
        analyze.analyze(self.dir / "m", p, [out])
        self.assertIn(self._run(p, out, ["--cases", case]), (0, 1))

    def test_dangling_events_reset_and_prefix_bound_resets(self):
        p = self._plan("holdout")
        out = self.dir / "ev"
        case = "en-02.L256"
        self.assertEqual(self._run(p, out, ["--cases", case]), 0)
        runs = out / "runs.jsonl"
        some = json.loads(runs.read_text().splitlines()[0])
        event = {"event": "reduction_failed", "run_id": "forged-e1", "of_run": "no-such-run", "case_id": case,
                 "arm": "A", "error": "forged", **{k: some[k] for k in ("plan_sha256", "binary_sha256",
                                                                          "arms_sha256", "model_receipt_sha256")}}
        with open(runs, "a") as f:
            f.write(json.dumps(event) + "\n")
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "INVALID")              # dangling event: never ignored
        self.assertEqual(self._run(p, out, ["--cases", case]), 0)        # its case is reset and redone
        self.assertTrue((out / "resets.jsonl").exists())
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertFalse(any("INVALID" in n for n in results["A"]["notes"]))
        # changing any line covered by a reset makes that reset invalid (prefix-bound, not endpoint-bound)
        lines = runs.read_text().splitlines()
        first = json.loads(lines[0])
        first["wall_seconds"] = first["wall_seconds"] + 1.0
        runs.write_text("\n".join([json.dumps(first, sort_keys=True)] + lines[1:]) + "\n")
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "INVALID")

    def test_retry_ceiling_survives_a_case_reset(self):
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["B"]["env"]["FAKE_TRUNCATE"] = "1"
        (self.dir / "arms.json").write_text(json.dumps(arms))
        p = self._plan("holdout")
        out = self.dir / "rc"
        case = "en-02.L256"
        def distinct_b():
            ids = set()
            for line in (out / "runs.jsonl").read_text().splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("arm") == "B" and "event" not in rec:
                    ids.add(rec["run_id"])
            return len(ids)
        self._run(p, out, ["--cases", case])
        self._run(p, out, ["--cases", case])
        self.assertEqual(distinct_b(), 2)                                   # limit reached
        with open(out / "runs.jsonl", "a") as f:                            # malformed line of this case
            f.write('{"case_id": "%s", "broken": ' % case + "\n")
        for _ in range(2):
            self._run(p, out, ["--cases", case])                            # reset, then resumes
        self.assertTrue((out / "resets.jsonl").exists())
        self.assertEqual(distinct_b(), 2)                                   # never a third attempt
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertNotEqual(results["B"]["verdict"], "PASS")

    def test_failure_recorded_before_its_marker_and_event_lineage(self):
        p = self._plan("holdout")
        out = self.dir / "fl"
        case = "en-02.L256"
        self.assertEqual(self._run(p, out, ["--cases", case]), 0)
        runs = out / "runs.jsonl"
        mark = json.loads((out / "done" / f"{case}.A").read_text())
        some = json.loads(runs.read_text().splitlines()[0])
        ident = {k: some[k] for k in ("plan_sha256", "binary_sha256", "arms_sha256", "model_receipt_sha256")}
        # a failure journaled for A's run, interrupted before the marker changed
        with open(runs, "a") as f:
            f.write(json.dumps({"event": "reduction_failed", "run_id": "x-e1", "of_run": mark["run_id"],
                                "case_id": case, "arm": "A", "error": "interrupted", **ident}) + "\n")
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertNotEqual(results["A"]["verdict"], "PASS")              # a failed run is never a result
        self.assertEqual(self._run(p, out, ["--cases", case]), 0)         # marker settled, one retry, success
        a_runs = [json.loads(l) for l in runs.read_text().splitlines() if l.startswith("{")]
        self.assertEqual(sum(1 for r in a_runs if r.get("arm") == "A" and "event" not in r), 2)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        # only this case ran (others are simply missing): nothing about it is invalid or failed
        self.assertFalse([n for n in results["A"]["notes"] if case in n])
        # an event naming an older run of its pair (not the latest preceding one) resets the case
        older = [r for r in a_runs if r.get("arm") == "A" and "event" not in r][0]["run_id"]
        with open(runs, "a") as f:
            f.write(json.dumps({"event": "reduction_failed", "run_id": "x-e2", "of_run": older,
                                "case_id": case, "arm": "A", "error": "stale target", **ident}) + "\n")
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "INVALID")
        self._run(p, out, ["--cases", case])
        self.assertTrue((out / "resets.jsonl").exists())

    def test_malformed_engine_count_is_a_counted_failure(self):
        knob = self.dir / "text.flag"
        knob.write_text("1")
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["A"]["env"]["FAKE_FORCED_TEXT_FILE"] = str(knob)
        (self.dir / "arms.json").write_text(json.dumps(arms))
        p = self._plan("holdout")
        out = self.dir / "mf"
        case = "en-02.L256"
        for _ in range(3):
            self._run(p, out, ["--cases", case])
        runs = [json.loads(l) for l in (out / "runs.jsonl").read_text().splitlines()]
        a_runs = [r for r in runs if r.get("arm") == "A" and "event" not in r]
        self.assertEqual(len(a_runs), 2)                                   # the ceiling holds
        for r in a_runs:
            self.assertTrue(evidence.run_record(r))                        # schema-valid evidence
            self.assertIsNone(r["forced_steps"])
            self.assertEqual(r["forced_steps_raw"], repr(str(24)))
        self.assertFalse((out / "resets.jsonl").exists())
        knob.unlink()                                                      # the condition goes away
        self._run(p, out, ["--cases", case])
        runs = [json.loads(l) for l in (out / "runs.jsonl").read_text().splitlines()]
        self.assertEqual(sum(1 for r in runs if r.get("arm") == "A" and "event" not in r), 2)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertNotEqual(results["A"]["verdict"], "PASS")

    def test_interrupted_validation_is_a_budget_stop(self):
        from unittest import mock
        p = self._plan("holdout")
        out = self.dir / "iv"
        case = "en-02.L256"
        with mock.patch.object(run.Runner, "validate", lambda self, *a: None):
            self.assertEqual(self._run(p, out, ["--cases", case]), 2)
        runs = [json.loads(l) for l in (out / "runs.jsonl").read_text().splitlines()]
        self.assertTrue(runs and all(r["interrupted"] and not r["complete"] for r in runs))
        self.assertFalse(list((out / "logits").glob("*.bin")))            # no unvalidated file kept
        self.assertEqual(self._run(p, out, ["--cases", case]), 0)
        led = evidence.Ledger([json.loads(l) for l in (out / "runs.jsonl").read_text().splitlines()], [],
                              {k: runs[0][k] for k in evidence.RUN_IDENTITY})
        self.assertEqual(led.failed(case, runs[0]["arm"]), [])

    def test_hashing_retained_logits_stops_at_the_deadline(self):
        from unittest import mock
        p = self._plan("holdout")
        out = self.dir / "hs"
        case = "en-02.L256"
        self.assertEqual(self._run(p, out, ["--cases", case, "--keep-logits"]), 0)
        self.assertTrue(list((out / "logits").glob("*.bin")))
        marks = {q.name: q.read_bytes() for q in (out / "done").iterdir()}

        def stop(self, path):
            raise run.BudgetStop(str(path))
        with mock.patch.object(run.Runner, "logits_sha", stop):
            self.assertEqual(self._run(p, out, ["--cases", case, "--keep-logits"]), 2)
        self.assertEqual({q.name: q.read_bytes() for q in (out / "done").iterdir()}, marks)   # nothing judged
        self.assertFalse((out / "resets.jsonl").exists())
        self.assertEqual(self._run(p, out, ["--cases", case, "--keep-logits"]), 0)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertFalse([n for n in results["A"]["notes"] if case in n])

    def test_damaged_diagnostic_journals_never_block_a_resume(self):
        p = self._plan("holdout")
        out = self.dir / "dd"
        case = "en-02.L256"
        self.assertEqual(self._run(p, out, ["--cases", case]), 0)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        before = (out / "runs.jsonl").read_text()
        verdicts = {k: v["verdict"] for k, v in results.items()}
        for name in ("effective-config.jsonl", "recovery.jsonl", "holdout-ledger.jsonl"):
            path = out / name
            old = path.read_text() if path.exists() else ""
            path.write_text('{"arm": "A", "attem' + old + '{"arm": "B", "rc": 0}\n{"partial')
        for extra in ([], ["--no-effective-config"]):
            self.assertEqual(self._run(p, out, ["--cases", case, *extra]), 0)
        self.assertEqual((out / "runs.jsonl").read_text(), before)          # nothing re-run
        context, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual({k: v["verdict"] for k, v in results.items()}, verdicts)
        self.assertGreater(context["effective_config_damaged_lines"], 0)
        # the damage is a boundary: captures written after it are reused, not repeated forever
        self.assertEqual(self._run(p, out, ["--cases", case]), 0)
        recs = common.diagnostic_records(out / "effective-config.jsonl", after_damage=True)[0]
        last = {r["arm"]: r for r in recs if isinstance(r.get("arm"), str)}
        self.assertTrue(last and all("reused_from" in r for r in last.values()))
        # a torn last line alone is reported too
        (out / "effective-config.jsonl").write_bytes(b'{"partial')
        context, _, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(context["effective_config_damaged_lines"], 1)

    def test_fast_parity_decides_and_exact_class_is_reported(self):
        pl = plan.init(self.dir / "m", "holdout", "REF", "CTRL", ["A", "B"], [], [256, 512], bootstrap=500,
                       min_parents=2, repeat_cases=["en-02.L256"], tier="fast-parity", secondary_tier="exact-class")
        pl["secondary"]["endpoints"]["E2"] = {"fixed": 0.01}
        pl["secondary"]["endpoints"]["E3"] = {"fixed": 0.01}
        pl = plan.freeze(pl)
        p = self.dir / "fast.json"
        plan.save(p, pl)
        out = self.dir / "fp"
        self.assertEqual(self._run(p, out), 0)
        context, results, ctrl_report, ctrl_means = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(sorted(results["A"]["endpoints"]), ["E1", "E5", "E6"])
        self.assertEqual(results["A"]["secondary"]["tier"], "exact-class")
        self.assertEqual(sorted(results["A"]["secondary"]["endpoints"]), ["E2", "E3", "E4", "E7"])
        self.assertEqual(results["B"]["verdict"], "FAIL")                    # large noise fails parity too
        receipts = self.dir / "fr"
        analyze.write(receipts, context, results, ctrl_report, ctrl_means)
        md = (receipts / "A.receipt.md").read_text()
        self.assertIn("Deciding margins: fast-parity", md)
        rec_a = json.loads((receipts / "A.receipt.json").read_text())
        self.assertIsNone(rec_a["statement"])                                 # short tiers carry none
        self.assertIn("statement", rec_a["secondary"])
        self.assertIn("Also reported (does not decide the verdict): exact-class", md)
        rec = json.loads((receipts / "A.receipt.json").read_text())
        self.assertEqual(rec["verdict"], results["A"]["verdict"])
        self.assertEqual(rec["secondary"]["verdict"], results["A"]["secondary"]["verdict"])

    def test_secondary_shares_the_hard_gates(self):
        pl = plan.init(self.dir / "m", "holdout", "REF", "CTRL", ["A", "B"], [], [256, 512], bootstrap=500,
                       min_parents=2, repeat_cases=["en-02.L256"], tier="fast-parity", secondary_tier="exact-class")
        for spec in pl["secondary"]["endpoints"].values():
            spec.pop("rule", None)
            spec.setdefault("fixed", 0.01)
        p = self.dir / "sg.json"
        plan.save(p, plan.freeze(pl))
        out = self.dir / "sg"
        # partial coverage: neither set may PASS
        some = sorted(c["case_id"] for c in manifest.load(self.dir / "m")[0]["cases"] if c["split"] == "holdout")[:4]
        self._run(p, out, ["--cases", ",".join(some)])
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        for opt in ("A", "B"):
            self.assertNotEqual(results[opt]["secondary"]["verdict"], "PASS")
            self.assertNotEqual(results[opt]["verdict"], "PASS")
        # a shared hard-gate failure (undeclared route) reaches the secondary verdict too
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["A"]["env"]["FAKE_ROUTE"] = "other"
        (self.dir / "arms.json").write_text(json.dumps(arms))
        out2 = self.dir / "sg2"
        self._run(p, out2)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out2])
        self.assertEqual(results["A"]["verdict"], "FAIL")
        self.assertEqual(results["A"]["secondary"]["verdict"], "FAIL")
        self.assertTrue(any("route" in n for n in results["A"]["secondary"]["notes"]))

    def test_exact_class_decides_on_nll_excess_over_control(self):
        pl = plan.init(self.dir / "m", "holdout", "REF", "CTRL", ["A", "B"], [], [256, 512], bootstrap=500,
                       min_parents=2, repeat_cases=["en-02.L256"], secondary_tier="reference-nll")
        self.assertEqual(sorted(pl["endpoints"]), ["E2", "E3", "E4", "E7"])
        self.assertEqual(pl["guards"]["G1"]["endpoint"], "E7")
        self.assertEqual(pl["secondary"]["guards"]["G1"]["endpoint"], "E1")
        pl["endpoints"]["E2"] = {"fixed": 0.01}
        pl["endpoints"]["E3"] = {"fixed": 0.01}
        p = self.dir / "e7.json"
        plan.save(p, plan.freeze(pl))
        out = self.dir / "e7"
        self.assertEqual(self._run(p, out), 0)
        _, results, ctrl_report, _ = analyze.analyze(self.dir / "m", p, [out])
        e7 = results["A"]["endpoints"]["E7"]["estimate"]
        e1 = results["A"]["secondary"]["endpoints"]["E1"]["estimate"]
        # E7 is the option's NLL change minus the control's, per parent (same parents in both)
        self.assertAlmostEqual(e7, e1 - ctrl_report["mean_parent"]["dnll"], places=12)
        self.assertNotAlmostEqual(e7, e1, places=12)
        self.assertIn("rounding-sensitive", results["A"]["secondary"]["statement"])
        self.assertIsNone(results["A"]["statement"])

    def test_clipped_control_e7_with_equal_weights(self):
        import glob as _g
        from collections import defaultdict

        def run_plan(name, clip):
            pl = plan.init(self.dir / "m", "holdout", "REF", "CTRL", ["A", "B"], [], [256, 512], bootstrap=300,
                           min_parents=2, repeat_cases=["en-02.L256"], e7_control_clip=clip, stratum_weights="equal")
            pl["endpoints"]["E2"] = {"fixed": 0.01}
            pl["endpoints"]["E3"] = {"fixed": 0.01}
            p = self.dir / f"{name}.json"
            plan.save(p, plan.freeze(pl))
            out = self.dir / name
            self.assertEqual(self._run(p, out), 0)
            return analyze.analyze(self.dir / "m", p, [out])[1], out

        def manual(out, clip):
            """Mean of stratum means of per-parent means of per-case (A dNLL - clip(CTRL dNLL))."""
            per = defaultdict(list)
            for f in sorted(_g.glob(str(out / "red" / "*.A.npz"))):
                case = Path(f).name[: -len(".A.npz")]
                a, c = np.load(f), np.load(f.replace(".A.npz", ".CTRL.npz"))
                cd = float(np.mean(c["nll_c"] - c["nll_r"]))
                per[case.rsplit(".L", 1)[0]].append(float(np.mean(a["nll_c"] - a["nll_r"]))
                                                    - (cd if clip is None else float(np.clip(cd, -clip, clip))))
            by_s = defaultdict(list)
            for parent, vals in per.items():
                by_s[parent.split("-")[0]].append(float(np.mean(vals)))
            return float(np.mean([np.mean(x) for x in by_s.values()]))

        big, out_big = run_plan("clipbig", 1e9)
        e7 = big["A"]["endpoints"]["E7"]
        self.assertEqual(e7["weights"], "equal")
        self.assertIn("control clipped", e7["label"])
        self.assertAlmostEqual(e7["estimate"], manual(out_big, None), places=12)   # a clip that never binds
        small, out_small = run_plan("clipsmall", 0.001)
        self.assertAlmostEqual(small["A"]["endpoints"]["E7"]["estimate"], manual(out_small, 0.001), places=12)

    def test_stale_partial_capture_is_cleared_before_a_run(self):
        p = self._plan("holdout")
        out = self.dir / "sp"
        case = "en-02.L256"
        (out / "logits").mkdir(parents=True)
        for arm in ("REF", "CTRL", "A", "B"):
            (out / "logits" / f"{case}.{arm}.bin.partial").write_bytes(b"left by a killed run")
        self.assertEqual(self._run(p, out, ["--cases", case]), 0)
        self.assertFalse(list((out / "logits").glob("*.partial")))

    def test_feasibility_checked_only_for_engine_work(self):
        p = self._plan("holdout")
        out = self.dir / "fz"
        case = "en-02.L256"
        self.assertEqual(self._run(p, out, ["--cases", case]), 0)
        # a finished case resumes under an engine prediction no invocation could fit
        self.assertEqual(self._run(p, out, ["--cases", case, "--predict", "5000"]), 0)
        # engine work that can never fit is an error, not a budget stop
        other = next(c["case_id"] for c in manifest.load(self.dir / "m")[0]["cases"]
                     if c["split"] == "holdout" and c["nominal_length"] == 256 and c["case_id"] != case)
        before = (out / "runs.jsonl").read_text()
        self.assertEqual(self._run(p, out, ["--cases", other]), 0)          # the selection itself is valid
        self.assertEqual(self._run(p, out, ["--cases", other, "--predict", "5000"]), 0)   # now finished
        out2 = self.dir / "fz2"
        self.assertEqual(self._run(p, out2, ["--cases", other, "--predict", "5000"]), 1)
        self.assertFalse((out2 / "runs.jsonl").exists() and (out2 / "runs.jsonl").read_text().strip())
        self.assertTrue(before)

    def test_duplicated_or_lost_outcomes_do_not_change_the_retry_count(self):
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["B"]["env"]["FAKE_TRUNCATE"] = "1"
        (self.dir / "arms.json").write_text(json.dumps(arms))
        p = self._plan("holdout")
        out = self.dir / "oc"
        case = "en-02.L256"
        b_runs = lambda: sum(1 for l in (out / "runs.jsonl").read_text().splitlines()
                             if json.loads(l)["arm"] == "B")
        self.assertEqual(self._run(p, out, ["--cases", case]), 1)        # one failure
        runs = out / "runs.jsonl"
        failed_line = next(l for l in runs.read_text().splitlines() if json.loads(l)["arm"] == "B")
        with open(runs, "a") as f:
            f.write((failed_line + "\n") * 2)                            # identical duplicated lines
        self.assertEqual(self._run(p, out, ["--cases", case]), 1)        # still gets its second attempt
        distinct_b = lambda: len({json.loads(l)["run_id"] for l in runs.read_text().splitlines()
                                  if json.loads(l)["arm"] == "B"})
        self.assertEqual(distinct_b(), 2)
        self.assertEqual(self._run(p, out, ["--cases", case]), 1)        # limit reached: no third run
        self.assertEqual(distinct_b(), 2)

    def test_nonfinite_reference_receipt_is_written(self):
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["REF"] = {"env": {"FAKE_NAN_REF": "1"}}
        (self.dir / "arms.json").write_text(json.dumps(arms))
        p = self._plan("holdout")
        out = self.dir / "nr"
        self._run(p, out)
        ctx, results, ctrl, means = analyze.analyze(self.dir / "m", p, [out])
        self.assertIn(results["A"]["verdict"], ("FAIL", "INVALID"))
        analyze.write(self.dir / "nrr", ctx, results, ctrl, means)
        self.assertTrue((self.dir / "nrr" / "A.receipt.json").exists())

    def test_failures_do_not_block_and_calibration_binding(self):
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["B"]["env"]["FAKE_TRUNCATE"] = "1"
        (self.dir / "arms.json").write_text(json.dumps(arms))
        sp = self._plan("search", frozen=False, name="s.json")
        out = self.dir / "fb"
        self.assertEqual(self._run(sp, out), 1)                 # B fails, everything else runs
        marks = {p.name: json.loads(p.read_text()) for p in (out / "done").iterdir()}
        self.assertTrue(all(v["status"] == "FAILED" for k, v in marks.items() if k.endswith(".B")))
        self.assertTrue(all(v["status"] == "COMPLETE" for k, v in marks.items() if k.endswith(".A")))
        ctx, results, ctrl, means = analyze.analyze(self.dir / "m", sp, [out])
        self.assertEqual(results["A"]["verdict"], "PASS")
        self.assertNotEqual(results["B"]["verdict"], "PASS")
        s = analyze.write(self.dir / "rs", ctx, results, ctrl, means)
        self.assertIsNotNone(s["control_means"])
        hp = plan.init(self.dir / "m", "holdout", "REF", "CTRL", ["A", "B"], [{"name": "AB", "members": ["A", "B"]}],
                       [256, 512], bootstrap=500, min_parents=2)
        hp = plan.freeze(plan.calibrate(hp, s))
        plan.save(self.dir / "h.json", hp)
        # the holdout must use the control the margins were calibrated with
        arms["arms"]["CTRL"]["env"]["FAKE_NOISE"] = "0.5"
        (self.dir / "arms.json").write_text(json.dumps(arms))
        out2 = self.dir / "fh"
        self._run(self.dir / "h.json", out2)
        _, results, _, _ = analyze.analyze(self.dir / "m", self.dir / "h.json", [out2])
        self.assertEqual(results["A"]["verdict"], "INVALID")
        self.assertTrue(any("calibrated" in n for n in results["A"]["notes"]))

    def test_holdout_refuses_unfrozen_plan(self):
        p = self._plan("holdout", frozen=False, name="open.json")
        self.assertEqual(self._run(p, self.dir / "out2"), 1)
        self.assertFalse((self.dir / "out2" / "runs.jsonl").exists())

    def test_search_calibration_and_nonfinite(self):
        p = self._plan("search", frozen=False, name="search.json")
        out = self.dir / "s"
        self.assertEqual(self._run(p, out), 0)
        ctx, results, ctrl, means = analyze.analyze(self.dir / "m", p, [out])
        self.assertFalse(ctx["qualification"])
        s = analyze.write(self.dir / "src", ctx, results, ctrl, means)
        self.assertTrue(json.loads((self.dir / "src" / "A.receipt.json").read_text())["verdict"].startswith("SEARCH-"))
        hp = plan.init(self.dir / "m", "holdout", "REF", "CTRL", ["A"], [], [256, 512])
        cal = plan.calibrate(hp, s)
        self.assertGreaterEqual(cal["endpoints"]["E2"]["fixed"], 0.002)
        self.assertAlmostEqual(cal["endpoints"]["E2"]["calibration"]["control_mean"], means["loss"])
        # non-finite logits in an arm fail that arm
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["A"]["env"]["FAKE_NAN"] = "1"
        (self.dir / "arms.json").write_text(json.dumps(arms))
        out2 = self.dir / "s2"
        m, _ = manifest.load(self.dir / "m")
        search = [c["case_id"] for c in m["cases"] if c["split"] == "search"]
        self.assertEqual(self._run(p, out2, ["--cases", ",".join(search)]), 0)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out2])
        self.assertEqual(results["A"]["verdict"], "FAIL")
        self.assertTrue(any("non-finite" in n for n in results["A"]["notes"]))


if __name__ == "__main__":
    unittest.main()
