import json
import copy
import uuid
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.qualification.option_qual import common, manifest, plan
from tools.qualification.option_qual.metrics import compare, log_softmax, self_metrics
from tools.qualification.option_qual.records import read_spool, read_steps, write_spool, write_steps
from tools.qualification.option_qual.reduce import HistoryMismatch, reduce_pair, reduce_self
from tools.qualification.option_qual.stats import resample_indices, t_quantile, upper_bound
from tools.qualification.longctx_fidelity.tok import FakeTokenizer


class MetricTests(unittest.TestCase):
    def test_identical_rows(self):
        z = np.array([0.5, -1.0, 2.0, 0.0], np.float32)
        m = compare(z, z, 2)
        self.assertAlmostEqual(m["overlap"], 1.0, places=12)
        self.assertEqual((m["kl_rc"], m["kl_cr"], m["agree"]), (0.0, 0.0, 1))
        self.assertAlmostEqual(m["nll_r"], m["nll_c"])

    def test_hand_example(self):
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.2, 0.3, 0.5])
        m = compare(np.log(p), np.log(q), 1)
        self.assertAlmostEqual(m["overlap"], 0.7, places=12)
        self.assertAlmostEqual(m["overlap"], 1 - 0.5 * np.abs(p - q).sum(), places=12)
        self.assertAlmostEqual(m["kl_rc"], float((p * np.log(p / q)).sum()), places=12)
        self.assertAlmostEqual(m["kl_cr"], float((q * np.log(q / p)).sum()), places=12)
        self.assertEqual((m["top1_r"], m["top1_c"], m["agree"]), (0, 2, 0))
        self.assertAlmostEqual(m["nll_c"], -math.log(0.3), places=12)
        self.assertAlmostEqual(m["gap_r"], math.log(0.5 / 0.3), places=12)

    def test_kl_direction_and_zero_mass(self):
        z_r = np.array([0.0, -800.0, 0.0])
        z_c = np.array([0.0, 0.0, 0.0])
        m = compare(z_r, z_c, 0)
        self.assertGreater(m["kl_cr"], m["kl_rc"])
        self.assertTrue(np.isfinite(m["kl_cr"]))

    def test_nonfinite_and_ties(self):
        m = compare(np.array([1.0, np.nan]), np.array([1.0, 0.0]), 0)
        self.assertEqual((m["nonfinite_r"], m["nonfinite_c"], m["agree"]), (1, 0, 0))
        self.assertTrue(math.isnan(m["overlap"]))
        self.assertEqual(compare(np.array([1.0, 1.0]), np.array([1.0, 1.0]), 1)["top1_r"], 0)
        self.assertEqual(self_metrics(np.array([0.0, 3.0]), 1)["is_label"], 1)

    def test_log_softmax_stable(self):
        lp = log_softmax(np.array([1000.0, 1000.0]))
        self.assertAlmostEqual(float(np.exp(lp).sum()), 1.0)


def _rows(n, vocab, rng, noise=0.0, base=None, labels=None):
    base = rng.normal(size=(n, vocab)) if base is None else base
    labels = rng.integers(0, vocab, n) if labels is None else labels
    rows = []
    for i in range(n):
        z = (base[i] + noise * rng.normal(size=vocab)).astype(np.float32)
        rows.append((i, 100 + i, int(np.argmax(z)), int(labels[i]), z))
    return rows, base, labels


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_steps_roundtrip_and_truncation(self):
        rng = np.random.default_rng(1)
        rows, _, _ = _rows(3, 7, rng)
        path = self.dir / "a.bin"
        write_steps(path, rows)
        back = list(read_steps(path))
        self.assertEqual([(r.step, r.position, r.label) for r in back], [(0, 100, rows[0][3]), (1, 101, rows[1][3]),
                                                                         (2, 102, rows[2][3])])
        self.assertTrue(np.array_equal(back[1].logits, rows[1][4]))
        data = path.read_bytes()
        path.write_bytes(data[:-3])
        with self.assertRaises(ValueError):
            list(read_steps(path))

    def test_spool_roundtrip_and_footer(self):
        rng = np.random.default_rng(2)
        inputs = [3, 1, 4, 1, 5]
        logits = [rng.normal(size=6) for _ in range(4)]
        path = self.dir / "s.bin"
        write_spool(path, inputs, logits)
        back = list(read_spool(path))
        self.assertEqual([r.label for r in back], inputs[1:])
        path.write_bytes(path.read_bytes()[:-16])
        with self.assertRaises(ValueError):
            list(read_spool(path))

    def test_reduce_pair(self):
        rng = np.random.default_rng(3)
        rows, base, labels = _rows(5, 11, rng)
        write_steps(self.dir / "r.bin", rows)
        write_steps(self.dir / "r2.bin", rows)
        arrays, info = reduce_pair(self.dir / "r.bin", self.dir / "r2.bin")
        self.assertTrue(info["bitwise_identical"])
        self.assertEqual(info["rows"], 5)
        self.assertTrue(np.allclose(arrays["overlap"], 1.0))
        noisy, _, _ = _rows(5, 11, rng, 0.1, base, labels)
        write_steps(self.dir / "c.bin", noisy)
        arrays, info = reduce_pair(self.dir / "r.bin", self.dir / "c.bin")
        self.assertFalse(info["bitwise_identical"])
        self.assertTrue((arrays["overlap"] < 1).all())
        bad = [(s, p, a, (l + 1) % 11, z) for s, p, a, l, z in noisy]
        write_steps(self.dir / "bad.bin", bad)
        with self.assertRaises(HistoryMismatch):
            reduce_pair(self.dir / "r.bin", self.dir / "bad.bin")
        write_steps(self.dir / "short.bin", noisy[:4])
        with self.assertRaises(HistoryMismatch):
            reduce_pair(self.dir / "r.bin", self.dir / "short.bin")
        self.assertEqual(reduce_self(self.dir / "r.bin")[1]["rows"], 5)


class StatsTests(unittest.TestCase):
    def test_t_quantiles(self):
        table = {(1, 0.95): 6.313752, (5, 0.95): 2.015048, (30, 0.95): 1.697261, (100, 0.95): 1.660234,
                 (5, 0.99): 3.364930, (30, 0.99): 2.457262, (30, 0.9875): 2.359566, (10, 0.975): 2.228139}
        for (df, p), v in table.items():
            self.assertAlmostEqual(t_quantile(p, df), v, places=4, msg=(df, p))
        self.assertAlmostEqual(t_quantile(0.05, 7), -t_quantile(0.95, 7), places=9)

    def test_bootstrap_keeps_strata_and_seed(self):
        strata = ["a"] * 3 + ["b"] * 5
        i1 = resample_indices(strata, 50, 7)
        i2 = resample_indices(strata, 50, 7)
        self.assertTrue(np.array_equal(i1, i2))
        for row in i1:
            self.assertEqual(sorted(strata[j] for j in row), sorted(strata))

    def test_no_residual_degrees_of_freedom(self):
        strata = ["a", "b"]
        b = upper_bound([0.0, 0.0], strata, 0.95, resample_indices(strata, 50, 1))
        self.assertEqual(b["df"], 0)
        self.assertEqual(b["ucb"], float("inf"))

    def test_bound_rarely_passes_at_margin(self):
        # True mean difference equals the margin: a valid upper bound must stay below it rarely.
        rng = np.random.default_rng(11)
        strata = ["a"] * 12 + ["b"] * 12
        idx = resample_indices(strata, 2000, 3)
        passes = 0
        for _ in range(200):
            v = 0.005 + rng.standard_t(4, size=24) * 0.01
            passes += upper_bound(v, strata, 0.95, idx)["ucb"] < 0.005
        self.assertLess(passes, 20)


def _sources(directory, strata=("en", "code"), parents=6, chat=False):
    directory = Path(directory)
    (directory / "items").mkdir(parents=True, exist_ok=True)
    lines = []
    for s in strata:
        for k in range(parents):
            pid = f"{s}-{k:02d}"
            for j in range(2):
                sid = f"{pid}-{j}"
                text = " ".join(f"{s}{k}w{j}x{n}" for n in range(40))
                (directory / "items" / f"{sid}.txt").write_text(text, encoding="utf-8")
                lines.append({"source_id": sid, "parent_id": pid, "stratum": s, "lang": s, "kind": "text",
                              "license": "test", "origin": "synthetic", "path": f"items/{sid}.txt"})
    if chat:
        for k in range(parents):
            pid = f"chat-{k:02d}"
            msgs = []
            for t in range(12):
                msgs.append({"role": "user", "content": f"q {k} {t} " * 2})
                msgs.append({"role": "assistant", "content": f"answer {k} {t} " * 4})
            (directory / "items" / f"{pid}.json").write_text(json.dumps({"messages": msgs}), encoding="utf-8")
            lines.append({"source_id": pid, "parent_id": pid, "stratum": "chat", "lang": "en", "kind": "chat",
                          "license": "test", "origin": "synthetic", "path": f"items/{pid}.json"})
    path = directory / "sources.jsonl"
    path.write_text("".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")
    return path


class ArmsFileTests(unittest.TestCase):
    def test_execution_time_is_timing_evidence(self):
        from tools.qualification.option_qual.evidence import Ledger
        ident = {k: "x" for k in ("plan_sha256", "binary_sha256", "arms_sha256", "model_receipt_sha256")}
        rec = dict(ident, case_id="c.L8", arm="A", run_id="r1", route=[], rc=0, timeout=False, complete=True,
                   shortened=False, rows=4, forced_steps=4, problem=None, logits_sha256="f", nominal_length=8,
                   wall_seconds=12.0, exec_seconds=9.0)
        led = Ledger([rec], [], ident, [b"x"])
        self.assertEqual(led.walls(8), [9.0])
        self.assertEqual(led.post(), [3.0])
        self.assertFalse(Ledger([dict(rec, exec_seconds=13.0)], [], ident, [b"x"]).records)
        # an interrupted validation is a budget stop: its timing never becomes a post-processing need
        cut = dict(rec, run_id="r2", complete=False, logits_sha256=None, interrupted=True, wall_seconds=80.0)
        self.assertEqual(Ledger([rec, cut], [], ident, [b"x", b"y"]).post(), [3.0])

    def test_cap_never_below_an_observed_run_that_fits(self):
        from types import SimpleNamespace
        from tools.qualification.option_qual.run import Runner
        r = Runner.__new__(Runner)
        r.a = SimpleNamespace(soft_seconds=430.0, reserve_seconds=20.0, cap_factor=1.5, cap_floor=60.0)
        r._startup = 10.0
        r.predicted = lambda length: 350.0
        r.post_cost = lambda: 50.0
        self.assertEqual(r.cap_limit(), 350.0)                 # start-up headroom left inside the soft budget
        self.assertGreaterEqual(r.cap(8192), 350.0)
        self.assertLessEqual(r.cap(8192), r.cap_limit())
        # a saturated cap is still admissible after start-up has used part of the budget
        r.predicted = lambda length: 300.0
        r.post_cost = lambda: 10.0
        self.assertLessEqual(r.cap(8192) + 10.0 + 20.0, 430.0 - 5.0)
        r.predicted = lambda length: 395.0
        self.assertFalse(r.feasible(8192))

    def test_startup_allowance_is_this_invocations_own(self):
        from types import SimpleNamespace
        import tempfile
        from tools.qualification.option_qual.run import Runner

        def runner(d, started, predict, pending=0):
            clock = [started]
            r = Runner.__new__(Runner)
            r.a = SimpleNamespace(soft_seconds=430.0, reserve_seconds=20.0, cap_factor=1.5, cap_floor=60.0)
            r.out, r.now, r.t0 = Path(d), (lambda: clock[0]), 0.0
            r.deadline = 430.0
            r.invocation = uuid.uuid4().hex
            r.reductions = SimpleNamespace(jobs=[object()] * pending)
            r.reduce_cost = 10.0
            r.predicted = lambda length: predict
            r.post_cost = lambda: 10.0
            return r

        case = {"nominal_length": 8192}
        with tempfile.TemporaryDirectory() as d:
            # 120 s of mandatory repair before the first run: the cap adapts at once, the run is admitted
            r = runner(d, 120.0, 250.0)
            self.assertTrue(r.affordable(case))
            self.assertGreaterEqual(r.cap(8192), 250.0)
            # a one-off 90 s capture before the first admission: admitted now, and the next invocation
            # (fast start-up) gets the full cap again
            r = runner(d, 91.5, 300.0)
            self.assertTrue(r.affordable(case))
            r = runner(d, 1.5, 300.0)
            self.assertTrue(r.affordable(case))
            self.assertEqual(r.cap(8192), 390.0)
            # pending reductions are part of the first admission's allowance
            self.assertTrue(runner(d, 2.0, 300.0, pending=1).affordable(case))
        with tempfile.TemporaryDirectory() as d:
            # start-up alone crowds the run out: budget stops, an error after STARTUP_STOPS in a row
            for _ in range(4):                     # only ever a budget stop: the next invocation measures again
                self.assertFalse(runner(d, 100.0, 350.0).affordable(case))
            self.assertTrue(runner(d, 2.0, 350.0).affordable(case))        # a fast start-up still runs
            with self.assertRaises(ValueError):                             # no invocation could fit this
                runner(d, 1.0, 395.0).affordable(case)

    def test_predict_forms(self):
        from tools.qualification.option_qual.run import parse_predict
        self.assertEqual(parse_predict("8192=40,32768=70"), {8192: 40.0, 32768: 70.0})
        self.assertEqual(parse_predict("81"), {0: 81.0})
        with self.assertRaises(ValueError):
            parse_predict("8192=x")

    def test_ledger_binds_arm_receipts_and_failure_events(self):
        from tools.qualification.option_qual.evidence import Ledger, run_ok
        ident = {k: "default" for k in ("plan_sha256", "binary_sha256", "arms_sha256", "model_receipt_sha256")}
        ident["arm_model_receipt_sha256"] = {"REF": "ref-receipt", "A": "option-receipt"}
        rec = dict({k: ident[k] for k in ("plan_sha256", "binary_sha256", "arms_sha256")},
                   model_receipt_sha256="option-receipt", case_id="c.L8", arm="A", run_id="r1", route=[],
                   rc=0, timeout=False, complete=True, shortened=False, rows=4, forced_steps=4, problem=None,
                   logits_sha256="f", nominal_length=8, wall_seconds=1.0)
        event = dict({k: rec[k] for k in ident if k != "arm_model_receipt_sha256"},
                     event="reduction_failed", case_id="c.L8", arm="A", run_id="e1", of_run="r1", error="failed")
        led = Ledger([rec, event], [], ident)
        self.assertFalse(led.bad)
        self.assertEqual(led.failed("c.L8", "A"), ["r1"])
        self.assertTrue(run_ok(rec, ident))
        wrong = dict(rec, model_receipt_sha256="default")
        self.assertFalse(run_ok(wrong, ident))
        self.assertTrue(Ledger([wrong], [], ident).bad)
        repeat = dict(rec, arm="REF@repeat", model_receipt_sha256="ref-receipt")
        self.assertTrue(run_ok(repeat, ident))

    def test_replay_provenance_is_required_by_the_ledger_policy(self):
        from tools.qualification.option_qual.evidence import Ledger, RUN_IDENTITY, replay_provenance
        ident = {k: "identity" for k in RUN_IDENTITY}
        rec = dict(ident, case_id="c.L8", arm="REF", run_id="r1", route=[], rc=0, timeout=False,
                   complete=True, shortened=False, rows=4, forced_steps=4, problem=None,
                   logits_sha256="a" * 64, nominal_length=8, wall_seconds=1.)
        cap = dict(capture_id="capture", collection_id="collection", label="REF", bin_sha256="a" * 64)
        good = dict(rec, replay_capture=cap)
        self.assertTrue(replay_provenance(good))
        self.assertFalse(Ledger([good], [], ident, require_replay=True).bad)
        invalid = [None, [], {}, "capture"]
        for key in cap:
            for value in (None, "", 1):
                invalid.append(dict(cap, **{key: value}))
        invalid.extend([dict(cap, bin_sha256="b" * 64), dict(cap, bin_sha256="not-a-digest")])
        for value in invalid:
            with self.subTest(capture=value):
                bad = dict(rec, replay_capture=value)
                self.assertFalse(replay_provenance(bad))
                self.assertTrue(Ledger([bad], [], ident, require_replay=True).bad)
                self.assertFalse(Ledger([bad], [], ident).bad)
        self.assertFalse(replay_provenance(rec))
        self.assertTrue(Ledger([rec], [], ident, require_replay=True).bad)
        # Failed attempts do not need a successful capture and retain their retry accounting.
        failed = dict(rec, complete=False, rc=3, problem="capture rejected")
        led = Ledger([failed], [], ident, require_replay=True)
        self.assertFalse(led.bad)
        self.assertEqual(led.failed("c.L8", "REF"), ["r1"])

    def test_per_arm_model_pair(self):
        from tools.qualification.option_qual.run import load_arms
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "arms.json"
            spec = {"model": "option.gguf", "model_receipt": "option.receipt"}
            path.write_text(json.dumps({"arms": {"REF": {}, "X": spec}}))
            self.assertEqual(load_arms(path, ["REF", "X"])["X"], spec)
            for key in ("model", "model_receipt"):
                path.write_text(json.dumps({"arms": {"X": {key: spec[key]}}}))
                with self.assertRaisesRegex(ValueError, "specified together"):
                    load_arms(path, ["X"])
            path.write_text(json.dumps({"arms": {"X": dict(spec, unknown=True)}}))
            with self.assertRaises(ValueError):
                load_arms(path, ["X"])

    def test_option_switches_allowed_hook_variables_refused(self):
        from tools.qualification.option_qual.run import load_arms
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "arms.json"
            path.write_text(json.dumps({"arms": {"REF": {}, "X": {"env": {"STRIX_EXAMPLE_OPTION": "1"}}}}))
            self.assertIn("X", load_arms(path, ["REF", "X"]))
            path.write_text(json.dumps({"arms": {"REF": {"env": {"STRIX_BITCALC_LOGITS_OUT": "f"}}}}))
            with self.assertRaises(ValueError):
                load_arms(path, ["REF"])


class ManifestPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_build_deterministic_split_nested(self):
        src = _sources(self.dir / "src", chat=True)
        m1, d1 = manifest.build(src, self.dir / "m1", FakeTokenizer(), [256, 512], cont=32, cont_min=20,
                                salt="s1", search_per_stratum=2)
        m2, d2 = manifest.build(src, self.dir / "m2", FakeTokenizer(), [256, 512], cont=32, cont_min=20,
                                salt="s1", search_per_stratum=2)
        self.assertEqual(d1, d2)
        self.assertTrue(m1["fake_tokenizer"])
        by = {}
        for p in m1["parents"]:
            by.setdefault(p["stratum"], []).append(p["split"])
        for s, splits in by.items():
            self.assertEqual(splits.count("search"), 2, s)
        cases = {c["case_id"]: c for c in m1["cases"]}
        a = manifest.read_id_text(self.dir / "m1" / cases["en-00.L256"]["prompt_file"]) \
            if hasattr(manifest, "read_id_text") else None
        from tools.qualification.option_qual.common import read_id_text
        short = read_id_text(self.dir / "m1" / cases["en-00.L256"]["prompt_file"])
        long = read_id_text(self.dir / "m1" / cases["en-00.L512"]["prompt_file"])
        self.assertEqual(long[:256], short)
        self.assertEqual(read_id_text(self.dir / "m1" / cases["en-00.L256"]["cont_file"]), long[256:288])
        # chat cuts: prompt ends where an assistant content starts; continuation stays inside it
        tok = FakeTokenizer()
        header = tok.encode("<|im_start|>assistant\n" + manifest.ASSISTANT_HEAD)
        for c in m1["cases"]:
            if c["kind"] != "chat":
                continue
            p = read_id_text(self.dir / "m1" / c["prompt_file"])
            q = read_id_text(self.dir / "m1" / c["cont_file"])
            self.assertEqual(p[-len(header):], header)
            self.assertNotIn("<|im_end|>", tok.decode(q))
            self.assertLessEqual(c["prompt_tokens"], c["nominal_length"])
        # salt changes the split; adding a parent to another stratum does not move existing ones
        _, d3 = manifest.build(src, self.dir / "m3", FakeTokenizer(), [256], cont=32, salt="s2", search_per_stratum=2)
        self.assertNotEqual(d1, d3)
        manifest.verify(self.dir / "m1", self.dir / "src")
        f = self.dir / "m1" / cases["code-01.L256"]["cont_file"]
        f.write_text(f.read_text().replace(",", ", ", 1))
        with self.assertRaises(ValueError):
            manifest.verify(self.dir / "m1")

    def test_split_stable_when_other_stratum_grows(self):
        p1 = {f"a{i}": {"stratum": "a"} for i in range(6)}
        p2 = dict(p1, **{f"b{i}": {"stratum": "b"} for i in range(9)})
        s1 = manifest.assign_splits(p1, "salt", 2)
        s2 = manifest.assign_splits(p2, "salt", 2)
        self.assertEqual({k: s2[k] for k in s1}, s1)

    def test_chat_content_markers_rejected(self):
        for bad in ("x <|im_end|> y", "a <|endoftext|>", "<think>x</think>", "   "):
            with self.assertRaises(ValueError):
                manifest.render_chat([{"role": "user", "content": bad}])
        self.assertIn("assistant\n<think>\n\n</think>\n\nok<|im_end|>",
                      manifest.render_chat([{"role": "assistant", "content": " ok "}]))

    def test_margin_tiers(self):
        src = _sources(self.dir / "src")
        manifest.build(src, self.dir / "m", FakeTokenizer(), [256], cont=32, salt="s", search_per_stratum=2)
        fast = plan.init(self.dir / "m", "holdout", "REF", None, ["FAST"], [], [256], tier="fast-parity")
        self.assertEqual(fast["tier"], "fast-parity")
        self.assertEqual({e: s["fixed"] for e, s in fast["endpoints"].items()},
                         {"E1": 0.0129, "E5": 0.04, "E6": 0.03})
        self.assertEqual(fast["guards"]["G1"]["endpoint"], "E1")
        plan.require_runnable(plan.freeze(fast), fast["manifest_sha256"])      # needs no control
        with self.assertRaises(ValueError):
            plan.init(self.dir / "m", "holdout", "REF", None, ["FAST"], [], [256], tier="fast-parity",
                      endpoints=["E1"])                                       # the tier fixes its endpoints
        no_ctrl = plan.init(self.dir / "m", "holdout", "REF", None, ["FAST"], [], [256], tier="fast-parity",
                            secondary_tier="exact-class")
        self.assertEqual(list(no_ctrl["secondary"]["endpoints"]), ["E1"])     # E2-E4 need a control
        with self.assertRaises(ValueError):
            plan.init(self.dir / "m", "holdout", "REF", None, ["FAST"], [], [256], endpoints=["E2"])
        with self.assertRaises(ValueError):
            plan.init(self.dir / "m", "holdout", "REF", "CTRL", ["FAST"], [], [256], tier="nope")
        both = plan.init(self.dir / "m", "holdout", "REF", "CTRL", ["FAST"], [], [256], tier="fast-parity",
                         secondary_tier="exact-class")
        self.assertEqual(both["secondary"]["tier"], "exact-class")
        with self.assertRaises(ValueError):
            plan.freeze(both)                       # the secondary set's rule margins are unresolved
        body = {"split": "search", "control_means": {"loss": 0.01, "kl_rc": 1e-4}, "control": "CTRL",
                "reference": "REF", "manifest_sha256": both["manifest_sha256"], "lengths": [256],
                "global_problems": [], "binary_sha256": ["b" * 64], "model_receipt_sha256": ["m" * 64],
                "reference_def": {}, "control_def": {"env": {"X": "1"}}}
        cal = plan.calibrate(both, dict(body, self_sha256=common.sha(common.canonical(body))))
        self.assertAlmostEqual(cal["secondary"]["endpoints"]["E2"]["fixed"], 0.005)
        self.assertEqual(cal["endpoints"]["E5"]["fixed"], 0.04)                # fixed margins untouched
        plan.require_runnable(plan.freeze(cal), cal["manifest_sha256"])
        # an exact-class plan may also narrow to the absolute endpoints
        abs_only = plan.init(self.dir / "m", "search", "REF", None, ["A"], [], [256], endpoints=["E1", "E5", "E6"])
        self.assertEqual(sorted(abs_only["endpoints"]), ["E1", "E5", "E6"])

    def test_margins_must_be_finite_and_resolved(self):
        src = _sources(self.dir / "src")
        manifest.build(src, self.dir / "m", FakeTokenizer(), [256], cont=32, salt="s", search_per_stratum=2)
        base = plan.init(self.dir / "m", "holdout", "REF", "CTRL", ["FAST"], [], [256], tier="fast-parity",
                         secondary_tier="exact-class")
        for bad in ("inf", "nan", float("inf"), True, -0.01, 0):
            pl = copy.deepcopy(base)
            pl["endpoints"]["E5"] = {"fixed": bad}
            self.assertIsNotNone(plan.check(pl))
            with self.assertRaises(ValueError):
                plan.freeze(pl)
        pl = copy.deepcopy(base)
        pl["secondary"]["guards"]["G1"]["max_stratum_point"] = float("inf")
        self.assertIsNotNone(plan.check(pl))
        pl = copy.deepcopy(base)
        pl["secondary"]["endpoints"]["E2"] = {"rule": {"abs": "x", "frac": 0.5, "of": "loss"}}
        self.assertIsNotNone(plan.check(pl))
        # a frozen plan edited back to an unresolved margin is refused for HOLDOUT
        pl = copy.deepcopy(base)
        for spec in pl["secondary"]["endpoints"].values():
            spec.pop("rule", None)
            spec["fixed"] = 0.01
        fr = plan.freeze(pl)
        plan.require_runnable(fr, fr["manifest_sha256"])
        fr["secondary"]["endpoints"]["E2"] = {"rule": {"abs": 0.002, "frac": 0.5, "of": "loss"}}
        with self.assertRaises(ValueError):
            plan.require_runnable(fr, fr["manifest_sha256"])
        self.assertIsNotNone(plan.unresolved(fr))

    def test_calibration_binding_and_long_tiers(self):
        src = _sources(self.dir / "src")
        manifest.build(src, self.dir / "m", FakeTokenizer(), [256], cont=32, salt="s", search_per_stratum=2)
        base = plan.init(self.dir / "m", "holdout", "REF", "CTRL", ["FAST"], [], [256], tier="fast-parity",
                         secondary_tier="exact-class")
        body = {"split": "search", "control_means": {"loss": 0.01, "kl_rc": 1e-4}, "control": "CTRL",
                "reference": "REF", "manifest_sha256": base["manifest_sha256"], "lengths": [256],
                "global_problems": [], "binary_sha256": ["b" * 64], "model_receipt_sha256": ["m" * 64],
                "reference_def": {}, "control_def": {"env": {"X": "1"}}}
        cal = plan.calibrate(base, dict(body, self_sha256=common.sha(common.canonical(body))))
        plan.require_runnable(plan.freeze(cal), cal["manifest_sha256"])

        def refused(pl):
            self.assertIsNotNone(plan.resolved_problem(pl))
            with self.assertRaises(ValueError):
                plan.freeze(pl)
            fr = dict(pl, frozen=True)
            with self.assertRaises(ValueError):
                plan.require_runnable(fr, fr["manifest_sha256"])

        for source in ({}, {"binary_sha256": ""}, None):
            pl = copy.deepcopy(cal)
            if source is None:
                del pl["calibration_source"]
            else:
                pl["calibration_source"] = source
            refused(pl)
        pl = copy.deepcopy(cal)                      # resolved margin, garbage retained rule
        pl["secondary"]["endpoints"]["E2"]["rule"] = {"abs": "nan", "frac": True, "of": "UNKNOWN"}
        refused(pl)
        pl = copy.deepcopy(cal)                      # margin edited away from its rule and calibration
        pl["secondary"]["endpoints"]["E2"]["fixed"] = 0.5
        refused(pl)
        pl = copy.deepcopy(cal)                      # calibrated from another SEARCH summary
        pl["secondary"]["endpoints"]["E3"]["calibration"]["search_summary_sha256"] = "0" * 64
        refused(pl)
        pl = copy.deepcopy(cal)                      # rule + fixed without a calibration record
        del pl["secondary"]["endpoints"]["E2"]["calibration"]
        refused(pl)
        pl = copy.deepcopy(cal)                      # rule metric edited after calibration
        pl["secondary"]["endpoints"]["E2"]["rule"]["of"] = "dnll"
        refused(pl)
        pl = copy.deepcopy(cal)                      # recorded mean no longer the SEARCH control mean
        pl["secondary"]["endpoints"]["E2"]["calibration"]["control_mean"] = 0.2
        pl["secondary"]["endpoints"]["E2"]["fixed"] = 0.1
        refused(pl)
        self.assertEqual(cal["secondary"]["endpoints"]["E2"]["calibration"]["metric"], "loss")
        self.assertEqual(cal["calibration_source"]["search_summary"]["control_means"]["loss"], 0.01)
        # the embedded SEARCH summary's mean, the records and the margins edited together
        pl = copy.deepcopy(cal)
        pl["calibration_source"]["search_summary"]["control_means"]["loss"] = 0.4
        for specs in (pl["secondary"]["endpoints"],):
            specs["E2"]["calibration"]["control_mean"] = 0.4
            specs["E2"]["fixed"] = 0.2
        refused(pl)
        pl = copy.deepcopy(cal)                      # a bound identity that is not the summary's
        pl["calibration_source"]["binary_sha256"] = "c" * 64
        refused(pl)
        pl = copy.deepcopy(cal)                      # scope edited after calibration
        pl["lengths"] = [512]
        refused(pl)
        pl = copy.deepcopy(cal)
        pl["control"] = "CTRL2"
        refused(pl)
        # a calibrated SEARCH plan in the earlier form (no tier, no embedded summary) stays usable
        legacy = copy.deepcopy(cal)
        legacy["split"] = "search"
        del legacy["tier"]
        del legacy["calibration_source"]["search_summary"]
        legacy.pop("secondary")
        self.assertIsNone(plan.check(legacy))
        self.assertIsNotNone(plan.check(dict(legacy, split="holdout")))   # never for HOLDOUT
        # the legacy exemption covers the primary set only; explicit secondary tiers are checked
        sec_long = copy.deepcopy(legacy)
        sec_long["secondary"] = {"tier": "fast-parity-long", "endpoints": {"E1": {"fixed": 0.05}}, "guards": {}}
        self.assertIsNotNone(plan.check(sec_long))
        # a long SEARCH plan written before tiers existed (no "tier" key) stays valid unchanged;
        # the same without a tier as HOLDOUT is refused
        legacy = plan.init(self.dir / "m", "search", "REF", "CTRL", ["X"], [], [131072, 261888],
                           tier="exact-class-long")
        legacy = {k: v for k, v in legacy.items() if k != "tier"}
        legacy["endpoints"]["E1"] = {"fixed": 0.005}
        self.assertIsNone(plan.check(legacy))
        self.assertIsNotNone(plan.check(dict(legacy, split="holdout")))
        # long contexts need a *-long tier, and a plan is either long or not
        for tier in ("exact-class", "fast-parity"):
            with self.assertRaises(ValueError):
                plan.init(self.dir / "m", "search", "REF", "CTRL", ["X"], [], [131072], tier=tier)
        with self.assertRaises(ValueError):
            plan.init(self.dir / "m", "search", "REF", "CTRL", ["X"], [], [256], tier="exact-class-long")
        with self.assertRaises(ValueError):
            plan.init(self.dir / "m", "search", "REF", "CTRL", ["X"], [], [32768, 131072], tier="exact-class-long")
        lg = plan.init(self.dir / "m", "search", "REF", "CTRL", ["X"], [], [131072, 261888], tier="exact-class-long",
                       secondary_tier="fast-parity-long")
        self.assertEqual(lg["endpoints"]["E1"]["fixed"], 0.05)
        self.assertEqual(sorted(lg["endpoints"]), ["E1", "E2", "E3", "E4"])
        self.assertEqual(lg["secondary"]["endpoints"]["E1"]["fixed"], 0.05)
        self.assertIn("rounding-sensitive regime", plan.TIER_STATEMENTS["exact-class-long"])
        self.assertIn("rounding-sensitive regime", plan.TIER_STATEMENTS["fast-parity-long"])
        self.assertNotIn("exact-class", plan.TIER_STATEMENTS)

    def test_equal_stratum_weights_and_e7_clip_validation(self):
        from tools.qualification.option_qual.stats import resample_indices, upper_bound
        rng = np.random.default_rng(3)
        v = rng.normal(0.0, 0.01, 12)
        st = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
        idx = resample_indices(st, 2000, 7)
        plain, eq = upper_bound(v, st, 0.975, idx), upper_bound(v, st, 0.975, idx, "equal")
        self.assertAlmostEqual(plain["estimate"], eq["estimate"], places=14)      # equal counts: same estimand
        self.assertAlmostEqual(plain["ucb_bootstrap"], eq["ucb_bootstrap"], places=12)
        st2 = ["a"] * 8 + ["b"] * 2 + ["c"] * 2
        v2 = np.array([1.0] * 8 + [0.0] * 2 + [0.0] * 2) + rng.normal(0, 1e-3, 12)
        idx2 = resample_indices(st2, 2000, 7)
        eq2 = upper_bound(v2, st2, 0.975, idx2, "equal")
        means = [v2[:8].mean(), v2[8:10].mean(), v2[10:].mean()]
        self.assertAlmostEqual(eq2["estimate"], float(np.mean(means)), places=12)  # mean of stratum means
        self.assertLess(abs(eq2["estimate"] - 1 / 3), 0.01)
        self.assertGreater(eq2["ucb"], eq2["estimate"])
        one = upper_bound(np.array([0.1, 0.2, 0.3]), ["a", "a", "b"], 0.975,
                          resample_indices(["a", "a", "b"], 100, 1), "equal")
        self.assertEqual(one["ucb"], float("inf"))                                  # a single-parent stratum
        with self.assertRaises(ValueError):
            upper_bound(v, st, 0.975, idx, "parents")
        src = _sources(self.dir / "src")
        manifest.build(src, self.dir / "m", FakeTokenizer(), [256], cont=32, salt="s", search_per_stratum=2)
        pl = plan.init(self.dir / "m", "holdout", "REF", "CTRL", ["A"], [], [256], secondary_tier="reference-nll",
                       e7_control_clip=0.02, stratum_weights="equal")
        self.assertEqual(pl["endpoints"]["E7"]["control_clip"], 0.02)
        self.assertEqual(pl["stratum_weights"], "equal")
        self.assertNotIn("control_clip", pl["secondary"]["endpoints"]["E1"])
        for bad in (0, -0.1, "0.02", True, float("nan")):
            p2 = copy.deepcopy(pl)
            p2["endpoints"]["E7"]["control_clip"] = bad
            self.assertIsNotNone(plan.check(p2))
        p2 = copy.deepcopy(pl)
        p2["endpoints"]["E4"]["control_clip"] = 0.02                                 # E7 only
        self.assertIsNotNone(plan.check(p2))
        self.assertIsNotNone(plan.check(dict(pl, stratum_weights="parents")))
        with self.assertRaises(ValueError):                                         # no E7 anywhere
            plan.init(self.dir / "m", "holdout", "REF", None, ["A"], [], [256], e7_control_clip=0.02)

    def test_unavailable_bound_is_incomplete_not_fail(self):
        from tools.qualification.option_qual.analyze import evaluate
        from tools.qualification.option_qual.stats import resample_indices

        def pp_of(n):
            z = np.zeros(n)
            return {m: z for m in ("dnll", "loss", "kl_rc", "kl_cr", "disagree", "nll_c", "nll_r")}
        specs = {"E7": {"fixed": 0.005}}
        guards = {"G1": {"endpoint": "E7", "max_stratum_point": 2.0, "unit": "x margin"}}
        for strata, want in ((["a", "b", "b", "b"], "INCOMPLETE"), (["a", "a", "b", "b"], "PASS")):
            idx = resample_indices(strata, 200, 1)
            pp = {"A": pp_of(len(strata)), "C": pp_of(len(strata))}
            verdict, notes, ends, _ = evaluate(specs, guards, "A", "C", pp, strata, 0.9875, idx, "equal")
            self.assertEqual(verdict, want, notes)
            if want == "INCOMPLETE":
                self.assertTrue(any("no upper bound" in n for n in notes))
                self.assertFalse(ends["E7"]["passed"])

    def test_plan_lifecycle(self):
        src = _sources(self.dir / "src")
        manifest.build(src, self.dir / "m", FakeTokenizer(), [256], cont=32, salt="s", search_per_stratum=2)
        pl = plan.init(self.dir / "m", "holdout", "REF", "CTRL", ["A", "B"], [{"name": "AB", "members": ["A", "B"]}],
                       [256])
        with self.assertRaises(ValueError):
            plan.freeze(pl)          # rule margins unresolved
        with self.assertRaises(ValueError):
            plan.require_runnable(pl, pl["manifest_sha256"])
        def summary(**over):
            body = dict({"split": "search", "control_means": {"loss": 0.01, "kl_rc": 1e-4}, "control": "CTRL",
                         "reference": "REF", "manifest_sha256": pl["manifest_sha256"], "lengths": [256],
                         "global_problems": [], "binary_sha256": ["b" * 64], "model_receipt_sha256": ["m" * 64],
                         "reference_def": {}, "control_def": {"env": {"X": "1"}}}, **over)
            return dict(body, self_sha256=common.sha(common.canonical(body)))
        cal = plan.calibrate(pl, summary())
        self.assertEqual(cal["calibration_source"]["control_def"], {"env": {"X": "1"}})
        for bad in (summary(control="X"), summary(reference="R2"), summary(binary_sha256=["a", "b"]),
                    summary(control_def=None), summary(lengths=[512]), summary(global_problems=["x"]),
                    summary(manifest_sha256="0" * 64), dict(summary(), control_means={"loss": 1, "kl_rc": 1})):
            with self.assertRaises(ValueError):
                plan.calibrate(pl, bad)
        self.assertAlmostEqual(cal["endpoints"]["E2"]["fixed"], 0.005)
        self.assertAlmostEqual(cal["endpoints"]["E3"]["fixed"], 5e-4)
        with self.assertRaises(ValueError):
            plan.calibrate(pl, summary(split="holdout"))
        import datetime
        t = datetime.datetime(2026, 1, 2, tzinfo=datetime.timezone.utc)
        fr = plan.freeze(cal, t)
        self.assertEqual(plan.plan_sha(fr), plan.plan_sha(plan.freeze(cal, t)))
        plan.require_runnable(fr, fr["manifest_sha256"])
        with self.assertRaises(ValueError):
            plan.calibrate(fr, summary())
        with self.assertRaises(ValueError):
            plan.init(self.dir / "m", "search", "REF", "", ["A"], [], [256], endpoints=["E2"])
        with self.assertRaises(ValueError):
            plan.init(self.dir / "m", "search", "REF.v1", "CTRL", ["A"], [], [256])
        empty = plan.init(self.dir / "m", "search", "REF", "CTRL", ["A"], [], [256])
        empty["endpoints"], empty["guards"] = {}, {}
        self.assertIn("endpoint", plan.check(empty))
        with self.assertRaises(ValueError):
            plan.freeze(empty)
        e2 = plan.init(self.dir / "m", "search", "REF", "CTRL", ["A"], [], [256], endpoints=["E2"])
        self.assertEqual(e2["guards"], {})
        e2["guards"] = {"G1": {"endpoint": "E1", "max_stratum_point": 2.0}}
        self.assertIn("guard", plan.check(e2))
        with self.assertRaises(ValueError):
            plan.init(self.dir / "m", "search", "REF", "CTRL", ["A"], [], [256], min_parents=1)
        fam = plan.init(self.dir / "m", "search", "REF", "CTRL", ["A"], [], [256], family=["A", "B", "C"])
        self.assertEqual(fam["family"], ["A", "B", "C"])
        with self.assertRaises(ValueError):
            plan.init(self.dir / "m", "search", "REF", "CTRL", ["A"], [], [256], routes={"Z": "x"})


if __name__ == "__main__":
    unittest.main()
