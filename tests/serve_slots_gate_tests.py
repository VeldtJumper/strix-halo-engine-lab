#!/usr/bin/env python3
"""CPU tests for tests/serve_slots_gate.py against a spawned fake engine server (this file with --fake-server).

The fake enforces each engine's request contract (MiniMax string prompts, Nemotron chat field whitelist, token-ID
prompts for the shared API), runs a step loop that batches active requests like the real schedulers, and exposes the
same server-side evidence (Qwen 'batch cycle sessions=N' log, DeepSeek /api/status batch counters, MiniMax /metrics
mm_* counters, Nemotron serve_batch receipts). FAKE_FAULT (comma list) injects one defect per adversarial test;
'witness' makes the Qwen fake emit the shared-execution witness (cycle line suffix + strix.shared_execution), and
'witness-separate' / 'witness-draft-declined' / 'witness-fallback' emit it for grouped-but-separate execution.
"""
import collections
import hashlib
import http.server
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest

HERE = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location("serve_slots_gate", HERE.parent / "serve_slots_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

NEMOTRON_FIELDS = {"model", "messages", "tools", "tool_choice", "stream", "stream_options", "max_tokens",
                   "max_completion_tokens", "temperature", "top_p", "top_k", "seed", "enable_thinking",
                   "preserve_thinking", "conversational_mode", "thinking_budget", "chat_template_kwargs", "dflash", "n"}


def fnv64(ids):
    h = 14695981039346656037
    for b in struct.pack("<%dI" % len(ids), *ids):
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return "%016x" % h


# The C++ witness suffix (include/strix/mtp_shared_execution_witness.hpp mtp_shared_execution_log; the same string
# is asserted by tests/mtp_shared_execution_witness_tests.cpp).
CXX_WITNESS_SUFFIX = (" shared_windows=1 shared_sessions=2 shared_rows=8 max_members=2 draft_batches=1 draft_batch_sessions=2"
                      " draft_launches=3 draft_shared_launches=2 draft_shared_rows=4 draft_max_rows=2 joint_fallbacks=0"
                      " draft_declines=sampled:1 verify_declines=none")


def fake_witness(n, faults, chunked, draft):
    """One cycle of n live sessions: (log suffix, verify shared, draft shared) as the engine would report it."""
    shared = n >= 2 and not faults & {"witness-separate", "witness-fallback"}
    drafted = draft and n >= 2 and "witness-draft-declined" not in faults
    chunks = 1 if chunked and n >= 2 and "witness-fallback" not in faults else 0
    w = n if shared else 0
    d = n if drafted else 0
    draft_declines = f"sampled:{n}" if draft and not drafted else "none"
    verify_declines = f"no_owner:{n}" if n >= 2 and not shared else "none"
    suffix = (f" chunks={chunks} shared_windows={int(shared)} shared_sessions={w} shared_rows={2 * w} max_members={w}"
              f" draft_batches={int(drafted)} draft_batch_sessions={d} draft_launches={2 * int(drafted)}"
              f" draft_shared_launches={2 * int(drafted)} draft_shared_rows={2 * d} draft_max_rows={d}"
              f" joint_fallbacks={int('witness-fallback' in faults and n >= 2)}"
              f" draft_declines={draft_declines} verify_declines={verify_declines}")
    return suffix, shared, drafted


# ---------------------------------------------------------------- fake engine server

def fake_server(argv):
    flags = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            flags[argv[i]] = argv[i + 1]
            i += 2
        else:
            flags[argv[i]] = True
            i += 1
    flavor = flags["--flavor"]
    port = int(flags.get("--port") or flags["--http-port"])
    faults = set(filter(None, os.environ.get("FAKE_FAULT", "").split(",")))
    if flavor == "qwen":
        slots = 8 if "--http-batch" not in flags else max(1, int(flags["--http-batch"]))
    elif flavor in ("deepseek", "minimax"):
        slots = int(flags.get("--slots", "4"))
    else:
        slots = 1 if os.environ.get("STRIX_G4_SERVE_BATCH") == "0" else int(os.environ.get("STRIX_G4_SERVE_SLOTS") or 8)
    batched = slots > 1
    admit_cap = 1 if faults & {"serial-only", "delayed-receipt"} else slots
    streams = int(flags["--streams"]) if "--streams" in flags else slots
    log_batch = flavor == "qwen" and os.environ.get("STRIX_MTP_BATCH_LOG")
    witness = flavor == "qwen" and any(f.startswith("witness") for f in faults)
    chunked = os.environ.get("STRIX_MTP_BATCH_JOINT", "").startswith("1") or os.environ.get("STRIX_COHORT_ROWS") in ("16", "32")
    draft_batch = os.environ.get("STRIX_MTP_DRAFT_BATCH", "").startswith("1")
    drafter = flavor == "deepseek" and "--dspark" in flags
    cond = threading.Condition()
    waiting, active = collections.deque(), []
    counters = dict(steps=0, decode_rows=0, drafted=0, admitted=0, armed=False, armed_steps=0, windows=0,
                    dspark_steps=0)

    class Req:
        def __init__(self, seed, n):
            self.seed, self.n, self.ids, self.width, self.done = seed, n, [], 0, threading.Event()
            self.shared = dict(cohort_cycles=0, verify_shared_cycles=0, verify_shared_rows=0, verify_max_members=0,
                               draft_steps=0, draft_shared_steps=0, draft_max_rows=0, draft_declines={},
                               verify_declines={})

    def step_loop():
        while True:
            with cond:
                cap = 1 if "serial-after-3" in faults and counters["admitted"] >= 3 else admit_cap
                if "mixed-prefill" in faults:
                    cap = 1  # one decoder per step, each step also carrying another request's prompt rows
                while len(active) < cap and waiting:
                    active.append(waiting.popleft())
                    counters["admitted"] += 1
                if not active:
                    cond.wait(0.05)
                    continue
                batch = list(active)
                counters["steps"] += 1
                # with a drafter every decoder carries two draft rows (rows alone prove nothing)
                counters["decode_rows"] += len(batch) * (3 if drafter else 1)
                if drafter and counters["armed"]:
                    counters["armed_steps"] += 1
                    counters["windows"] += len(batch)
                    if "mixed-prefill" not in faults or counters["armed_steps"] % 2:
                        counters["dspark_steps"] += 1
                suffix, shared, drafted = fake_witness(len(batch), faults, chunked, draft_batch)
                for r in batch:
                    r.ids.append((r.seed + 7 * len(r.ids)) % 50000)
                    r.width = max(r.width, len(batch))
                    if len(batch) >= 2:
                        w = r.shared
                        w["cohort_cycles"] += 1
                        w["verify_shared_cycles"] += int(shared)
                        w["verify_shared_rows"] += 2 * int(shared)
                        w["verify_max_members"] = max(w["verify_max_members"], len(batch) if shared else 0)
                        w["draft_steps"] += 2
                        w["draft_shared_steps"] += 2 * int(drafted)
                        w["draft_max_rows"] = max(w["draft_max_rows"], len(batch) if drafted else 0)
                        if not shared:
                            w["verify_declines"]["no_owner"] = w["verify_declines"].get("no_owner", 0) + 1
                        if draft_batch and not drafted:
                            w["draft_declines"]["sampled"] = w["draft_declines"].get("sampled", 0) + 1
                    if len(r.ids) == r.n:
                        active.remove(r)
                        r.done.set()
            if log_batch:
                n = len(batch)
                groups = 0 if n < 2 or "qwen-solo-groups" in faults else 1
                print(f"batch cycle sessions={n} groups={groups} rows={4 * groups} solo={n - 2 * groups} emitted={n}"
                      + (suffix if witness else ""), file=sys.stderr, flush=True)
                if "joint" in faults and n >= 2:
                    print(f"joint cycle sessions={n} rows=2,2 drafts=1 emitted={n}", file=sys.stderr, flush=True)
            time.sleep(0.02)

    threading.Thread(target=step_loop, daemon=True).start()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):
            pass

        def reply(self, code, obj, content="application/json"):
            data = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", content)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self.reply(200, {"status": "ok"})
            if self.path.startswith("/api/status") and flavor == "deepseek" and "no-status" not in faults:
                with cond:
                    if "detail=1" in self.path:
                        counters["armed"] = True
                    status = {"engine": "fake", "batch": {"steps": counters["steps"], "decode_rows": counters["decode_rows"]}}
                    if drafter and counters["armed_steps"]:  # published only after an armed step
                        status["dspark"] = {"decode_steps": counters["dspark_steps"], "window_rows": [0, counters["windows"]]}
                    return self.reply(200, status)
            if self.path == "/metrics" and flavor == "minimax":
                with cond:
                    text = (f"mm_queued 0\nmm_steps {counters['steps']}\nmm_step_rows {counters['decode_rows']}\n"
                            f"mm_drafted {counters['drafted']}\n")
                return self.reply(200, text.encode(), "text/plain")
            self.reply(404, {"error": {"message": "unknown endpoint"}})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize" and flavor == "minimax":
                return self.reply(200, {"tokens": list(body["text"].encode("utf-8"))})
            if flavor == "nemotron":
                if self.path != "/v1/chat/completions":
                    return self.reply(404, {"error": {"message": "unknown endpoint"}})
                unknown = set(body) - NEMOTRON_FIELDS
                if unknown:
                    return self.reply(400, {"error": {"message": "unsupported field: " + sorted(unknown)[0]}})
                prompt_ids = list(body["messages"][0]["content"].encode("utf-8"))
            else:
                if self.path != "/v1/completions":
                    return self.reply(404, {"error": {"message": "no route"}})
                prompt = body.get("prompt")
                if flavor == "minimax":
                    if not isinstance(prompt, str):
                        return self.reply(400, {"error": {"message": "prompt (string) required"}})
                    prompt_ids = list(prompt.encode("utf-8"))
                elif isinstance(prompt, list):
                    prompt_ids = prompt
                else:
                    return self.reply(400, {"error": {"message": "fake: send token IDs"}})
            if "http500" in faults and batched:
                return self.reply(500, {"error": {"message": "boom"}})
            n = body["max_tokens"]
            req = Req(sum(prompt_ids) * 31 + len(prompt_ids), n - 1 if "truncated" in faults else n)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("X-Strix-Http-Slots", str(slots))
            self.send_header("X-Strix-Queue-Position", "0")
            self.end_headers()
            with cond:
                waiting.append(req)
                cond.notify_all()
            sent = 0
            while True:
                finished = req.done.wait(0.005)
                while sent < len(req.ids):
                    sent += 1
                    choice = {"index": 0, "finish_reason": None}
                    choice.update({"delta": {"content": "x"}} if flavor == "nemotron" else {"text": "x"})
                    self.wfile.write(b"data: " + json.dumps({"choices": [choice]}).encode() + b"\n\n")
                    self.wfile.flush()
                if finished and sent == len(req.ids):
                    break
            if "delayed-receipt" in faults:
                time.sleep(0.3)  # serial engine, late receipts: client clocks would see "overlap"
            ids = list(req.ids)
            if "batch-ids" in faults and req.width > 1:
                ids[-1] += 1
            if "batch-ids-p3" in faults and batched and len(prompt_ids) == 2:
                ids[-1] += 1
            digest = hashlib.sha256(json.dumps([len(prompt_ids), ids]).encode()).hexdigest()
            if "batch-state" in faults and req.width > 1:
                digest = hashlib.sha256(b"perturbed" + digest.encode()).hexdigest()
            prompt_tokens = len(prompt_ids) + (1 if "wrong-prompt-tokens" in faults and batched else 0)
            usage = {"prompt_tokens": prompt_tokens, "completion_tokens": len(ids) + (1 if "usage-count" in faults else 0),
                     "total_tokens": prompt_tokens + len(ids)}
            finish = None if "no-finish" in faults else "length"
            final = {"choices": [{"index": 0, "finish_reason": finish}]}
            if flavor == "qwen":
                components = {"recurrent_matrix": digest, "gdn_convolution": digest[::-1],
                              "kv.L3": hashlib.sha256(b"kv" + digest.encode()).hexdigest(),
                              "index.L3": hashlib.sha256(b"ix" + digest.encode()).hexdigest(),
                              "mtp.kv": hashlib.sha256(str(time.monotonic()).encode()).hexdigest()}
                if "no-kv" in faults:
                    components.pop("kv.L3")
                    components.pop("index.L3")
                if "null-component" in faults:
                    components["recurrent_matrix"] = None
                final.update(usage=usage, strix={"token_ids": ids, "prompt_state_digest": "p" + str(time.monotonic()),
                                                 "final_state_position": len(prompt_ids) + n,
                                                 "final_state_components": components})
                if witness and "witness-no-receipt" not in faults:
                    final["strix"]["shared_execution"] = req.shared
            elif flavor == "deepseek":
                final.update(usage=usage, strix={"token_ids": ids})
            elif flavor == "minimax":
                wrong = "wrong-fnv" in faults and batched
                final.update(strix={"token_ids": ids}, timings={
                    "prompt_fnv": fnv64(prompt_ids + ([1] if wrong else [])), "ids_fnv": fnv64(ids)})
            else:
                phase = {"state_digest": "xyz" if "bad-hex" in faults else digest, "frontier": len(prompt_ids) + len(ids),
                         "decode_identity": ("q4k-b" if "identity-batch" in faults and batched else "q4k") + f"-s{streams}",
                         "prefill_identity": "prefill-exact", "sampling_identity": "splitmix64-cdf-v1", "row_cap": 8,
                         "streams": len(active) + 1}
                if "no-frontier" in faults:
                    phase.pop("frontier")
                if batched and "no-serve-batch" not in faults:
                    phase["serve_batch"] = {"mode": "continuous", "slots": slots, "max_batch": req.width}
                final.update(strix={"phases": [phase], "generated_ids": ids})
            self.wfile.write(b"data: " + json.dumps(final).encode() + b"\n\n")
            if flavor in ("minimax", "nemotron"):  # separate usage chunk (MiniMax repeats the IDs)
                u = {"choices": [], "usage": usage}
                if flavor == "minimax":
                    u["strix"] = {"token_ids": ids}
                self.wfile.write(b"data: " + json.dumps(u).encode() + b"\n\n")
            if "no-done" not in faults:
                self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    exit_code = [0]

    def stop(*_):
        if flavor == "deepseek" and "no-stopped" not in faults:
            print('{"event":"STOPPED","rank":0}', flush=True)
        exit_code[0] = 3 if "crash-exit" in faults else 0
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGBREAK if os.name == "nt" else signal.SIGINT, stop)
    marker = {"deepseek": '{"event":"READY","rank":0}', "minimax": "READY rank 0 http://127.0.0.1",
              "nemotron": "Gemma 4 service ready"}.get(flavor)
    if marker:
        print(marker, flush=True)
    server.serve_forever()
    server.server_close()
    return exit_code[0]


# ---------------------------------------------------------------- tests

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Done:
    def __init__(self, returncode, stdout, stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.ids = []
        for i, toks in enumerate(([1, 2, 3, 4], [9, 8, 7], [100, 200])):
            p = self.dir / f"p{i}.ids"
            p.write_text(" ".join(map(str, toks)))
            self.ids.append(str(p))
        self.texts = []
        for i, t in enumerate(("Hello there.", "Write a function.", "Hola.")):
            p = self.dir / f"t{i}.txt"
            p.write_text(t)
            self.texts.append(str(p))
        os.environ.pop("FAKE_FAULT", None)
        self.real_ssh = gate.ssh

    def tearDown(self):
        gate.ssh = self.real_ssh
        os.environ.pop("FAKE_FAULT", None)
        self.tmp.cleanup()

    def run_gate(self, engine, prompts, *extra, flavor=None, fault=None, command_extra=(), clients="1,2,4", out=None):
        if fault:
            os.environ["FAKE_FAULT"] = fault
        else:
            os.environ.pop("FAKE_FAULT", None)
        out = out or self.dir / ("out-" + engine + "-" + str(fault) + "-" + str(time.monotonic_ns()))
        argv = ["--engine", engine, "--out", str(out), "--port", str(free_port()), "--startup-timeout", "30",
                "--generate", "4", "--clients", clients]
        for p in prompts:
            argv += ["--prompt", p]
        argv += list(extra) + ["--", sys.executable, str(HERE), "--fake-server", "--flavor", flavor or engine,
                               *command_extra]
        code = gate.main(argv)
        summary = json.loads((out / "summary.json").read_text())
        return code, summary

    def nemotron(self, *extra, fault=None, streams=True, **kw):
        return self.run_gate("nemotron", self.texts, *extra, fault=fault,
                             command_extra=("--streams", "8") if streams else (), **kw)

    # -- positive paths

    def test_deepseek_tokens_only_exact_with_server_witness(self):
        code, s = self.run_gate("deepseek", self.ids, "--modes", "2,4", command_extra=("--rank", "0"))
        self.assertEqual((code, s["verdict"], s["coverage"]), (0, "EXACT", "tokens-only"), s["message"])
        self.assertTrue(any(r["witness"] for r in s["rounds"]))
        self.assertTrue(all(r["engine_digest"] is None for r in s["rows"]))
        self.assertEqual(len(s["rows"]), 3)

    def test_nemotron_equal_but_unqualified(self):
        code, s = self.nemotron("--modes", "auto,2")
        self.assertEqual((code, s["coverage"], s["verdict"]), (5, "tokens+state", "UNQUALIFIED"), s["message"])
        self.assertIn("forward_rows", s["message"])
        self.assertEqual(s["servers"]["reference"]["env"]["STRIX_G4_SERVE_BATCH"], "0")

    def test_qwen_generic_excludes_predictor_rows(self):
        code, s = self.run_gate("qwen", self.ids, "--generic", "--modes", "auto,2", "--reference-repeat", "2")
        self.assertEqual((code, s["verdict"]), (0, "EXACT"), s["message"])

    def test_glm_not_run(self):
        code, s = self.run_gate("glm", self.ids)
        self.assertEqual((code, s["verdict"], s["coverage"]), (3, "NOT-RUN", "not-run"))

    # -- finding 1: MiniMax takes a string prompt; the fake refuses token arrays like the real API

    def test_f1_minimax_string_prompt_contract(self):
        code, s = self.run_gate("minimax-pair", self.texts, "--modes", "auto,4", flavor="minimax")
        self.assertEqual((code, s["engine"], s["verdict"]), (0, "minimax", "EXACT"), s["message"])
        self.assertTrue(all(f["prompt_fnv"] for f in s["prompt_facts"]))

    # -- finding 2: overlap must be server-side evidence

    def test_f2_serial_engine_with_late_receipts_is_not_exact(self):
        code, s = self.run_gate("deepseek", self.ids, "--modes", "4", fault="delayed-receipt")
        self.assertEqual((code, s["verdict"]), (2, "NO-WITNESS"), s["message"])

    def test_f2_unreadable_evidence_is_unqualified(self):
        code, s = self.run_gate("deepseek", self.ids, "--modes", "4", fault="no-status")
        self.assertEqual((code, s["verdict"]), (5, "UNQUALIFIED"), s["message"])

    # -- finding 3: no EXACT without a concurrent workload

    def test_f3_single_client_or_single_slot_is_unqualified(self):
        code, s = self.run_gate("deepseek", self.ids, "--modes", "2,4", clients="1")
        self.assertEqual((code, s["verdict"]), (5, "UNQUALIFIED"), s["message"])
        code, s = self.run_gate("deepseek", self.ids, "--modes", "1")
        self.assertEqual((code, s["verdict"]), (5, "UNQUALIFIED"), s["message"])
        code, s = self.run_gate("deepseek", self.ids, "--modes", "4", "--allow-no-witness")
        self.assertEqual((code, s["verdict"]), (5, "UNQUALIFIED"), s["message"])
        self.assertNotIn("rows", s)

    # -- finding 4: every prompt runs in the candidate workloads

    def test_f4_prompt_beyond_client_count_is_checked(self):
        code, s = self.run_gate("deepseek", self.ids, "--modes", "4", clients="2", fault="batch-ids-p3")
        self.assertEqual((code, s["verdict"]), (1, "MISMATCH"), s["message"])
        self.assertIn("prompt 3", s["message"])
        self.assertEqual(s["rows"][0]["prompt_id"], "http-slots-p3")
        code, s = self.run_gate("deepseek", self.ids, "--modes", "4", clients="2")
        self.assertEqual(code, 0, s["message"])
        self.assertEqual([r["prompts"] for r in s["rounds"]], [[1, 2], [3, 1]])
        self.assertEqual(len(s["rows"]), 3)

    # -- finding 5: stream completion and request binding

    def test_f5_truncated_or_unterminated_streams(self):
        for fault in ("truncated", "no-done", "no-finish", "usage-count"):
            code, s = self.run_gate("deepseek", self.ids, "--modes", "4", fault=fault)
            self.assertEqual((code, s["verdict"]), (4, "NO-REFERENCE"), fault + ": " + s["message"])

    def test_f5_receipt_bound_to_another_request(self):
        code, s = self.run_gate("minimax", self.texts, "--modes", "4", fault="wrong-fnv")
        self.assertEqual((code, s["verdict"]), (1, "MISMATCH"), s["message"])
        self.assertIn("prompt_fnv", s["message"])
        code, s = self.run_gate("deepseek", self.ids, "--modes", "4", fault="wrong-prompt-tokens")
        self.assertEqual((code, s["verdict"]), (1, "MISMATCH"), s["message"])

    def test_f5_batch_token_and_state_mismatch(self):
        code, s = self.run_gate("minimax", self.texts, "--modes", "4", fault="batch-ids")
        self.assertEqual((code, s["verdict"]), (1, "MISMATCH"), s["message"])
        code, s = self.nemotron("--modes", "4", fault="batch-state")
        self.assertEqual((code, s["verdict"]), (1, "MISMATCH"), s["message"])
        self.assertIn("state digests differ", s["message"])

    # -- finding 6: strict state schemas

    def test_f6_malformed_state_is_no_reference(self):
        code, s = self.run_gate("qwen", self.ids, "--generic", "--modes", "2", fault="null-component")
        self.assertEqual((code, s["verdict"]), (4, "NO-REFERENCE"), s["message"])
        for fault in ("bad-hex", "no-frontier"):
            code, s = self.nemotron("--modes", "2", fault=fault)
            self.assertEqual((code, s["verdict"]), (4, "NO-REFERENCE"), fault + ": " + s["message"])

    # -- finding 7: Nemotron streams pinned, identity verified

    def test_f7_nemotron_streams_pinned_and_identity_checked(self):
        code, s = self.nemotron("--modes", "2", streams=False)
        self.assertEqual((code, s["verdict"]), (4, "NO-REFERENCE"), s["message"])
        self.assertIn("--streams", s["message"])
        code, s = self.nemotron("--modes", "2", fault="identity-batch")
        self.assertEqual((code, s["verdict"]), (4, "NO-REFERENCE"), s["message"])
        self.assertIn("identity", s["message"])

    # -- finding 8: shutdown, rank 1, stale evidence

    def test_f8_unclean_rank0_shutdown_is_error(self):
        for fault in ("crash-exit", "no-stopped"):
            code, s = self.run_gate("deepseek", self.ids, "--modes", "4", fault=fault)
            self.assertEqual((code, s["verdict"]), (6, "ERROR"), fault + ": " + s["message"])

    def test_f8_rank1_launch_and_exit_enforced(self):
        calls = []

        def fake_ssh(stop_reply):
            def run(args, script):
                calls.append(script)
                return Done(0, "STARTED\n") if "setsid" in script else Done(0, stop_reply)
            return run

        common = ("--modes", "4", "--peer", "peer", "--peer-dir", "/r")
        gate.ssh = fake_ssh("RC=0\n")
        code, s = self.run_gate("deepseek", self.ids, *common, command_extra=("--rank", "0"))
        self.assertEqual(code, 0, s["message"])
        self.assertEqual(s["servers"]["slots-4"]["rank1_exit_code"], 0)
        for reply in ("RC=1\n", "KILLED\n", "INT-RC=0\n"):
            gate.ssh = fake_ssh(reply)
            code, s = self.run_gate("deepseek", self.ids, *common, command_extra=("--rank", "0"))
            self.assertEqual((code, s["verdict"]), (6, "ERROR"), reply + s["message"])
        gate.ssh = lambda args, script: Done(255, "", "ssh: connect failed") if "setsid" in script else Done(0, "NO-PID\n")
        calls.clear()
        code, s = self.run_gate("deepseek", self.ids, *common, command_extra=("--rank", "0"))
        self.assertEqual((code, s["verdict"]), (4, "NO-REFERENCE"), s["message"])

    def test_f8_stale_summary_replaced_on_uncaught_failure(self):
        out = self.dir / "stale"
        out.mkdir()
        (out / "summary.json").write_text(json.dumps({"verdict": "EXACT"}))
        code, s = self.run_gate("deepseek", self.ids, "--modes", "4", fault="http500", out=out)
        self.assertEqual((code, s["verdict"]), (6, "ERROR"), s["message"])

    # -- round 2 (sol-wp26-r2): one adversarial test per item

    def test_r2_1_qwen_witness_requires_shared_cohort(self):
        code, s = self.run_gate("qwen", self.ids, "--generic", "--modes", "4", fault="qwen-solo-groups")
        self.assertEqual((code, s["verdict"]), (2, "NO-WITNESS"), s["message"])
        code, s = self.run_gate("qwen", self.ids, "--generic", "--modes", "4", fault="joint")
        self.assertEqual((code, s["verdict"]), (5, "UNQUALIFIED"), s["message"])
        os.environ["STRIX_COHORT_ROWS"] = "16"
        try:
            code, s = self.run_gate("qwen", self.ids, "--generic", "--modes", "4")
        finally:
            os.environ.pop("STRIX_COHORT_ROWS")
        self.assertEqual((code, s["verdict"]), (5, "UNQUALIFIED"), s["message"])
        self.assertIn("shared-execution witness", s["message"])

    # -- shared-execution witness (joint / budget / draft-batch arms)

    def qwen_env(self, fault, **env):
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            return self.run_gate("qwen", self.ids, "--generic", "--modes", "4", fault=fault)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_witness_parse_matches_engine_format(self):
        line = "batch cycle sessions=2 groups=1 rows=8 solo=0 emitted=4 ms=1.5 begin_ns=1 end_ns=2 chunks=1" + CXX_WITNESS_SUFFIX
        (c,) = gate.parse_batch_cycles("noise\n" + line + "\n")
        self.assertEqual((c["sessions"], c["chunks"], c["shared_windows"], c["max_members"]), (2, 1, 1, 2))
        self.assertEqual((c["draft_shared_launches"], c["draft_max_rows"]), (2, 2))
        self.assertEqual((c["draft_declines"], c["verify_declines"]), ({"sampled": 1}, {}))
        arms = gate.qwen_arms({"STRIX_MTP_BATCH_JOINT": "1", "STRIX_MTP_DRAFT_BATCH": "1"})
        d = gate.shared_execution_detail(arms, [c], None)
        self.assertEqual((d["shared"], d["verify_shared"], d["draft_shared"]), (True, True, True))
        # grouped but executed separately: no shared window / chunks only from the ordinary fallback
        for key, value in (("shared_windows", 0), ("chunks", 0), ("draft_shared_launches", 0)):
            d = gate.shared_execution_detail(arms, [dict(c, **{key: value})], None)
            self.assertFalse(d["shared"], key)
        # receipts must show every request sharing
        receipt = dict(verify_shared_cycles=1, draft_shared_steps=1)
        self.assertTrue(gate.shared_execution_detail(arms, [c], [receipt, receipt])["shared"])
        self.assertFalse(gate.shared_execution_detail(arms, [c], [receipt, dict(receipt, draft_shared_steps=0)])["shared"])
        # joint without cycle lines cannot prove the path; the default cohort path can rest on receipts
        self.assertIsNone(gate.shared_execution_detail(arms, [], [receipt, receipt])["shared"])
        self.assertTrue(gate.shared_execution_detail(gate.qwen_arms({}), [], [receipt, receipt])["shared"])

    def test_witness_qualifies_joint_and_budget_paths(self):
        code, s = self.qwen_env("witness,joint", STRIX_MTP_BATCH_JOINT="1")
        self.assertEqual((code, s["verdict"]), (0, "EXACT"), s["message"])
        detail = next(r["witness_detail"] for r in s["rounds"] if r["pair_round"])
        self.assertEqual(detail["arms"], ["joint"])
        self.assertGreaterEqual(detail["shared_windows"], 1)
        self.assertGreaterEqual(detail["chunks"], 1)
        self.assertEqual(len(detail["receipts"]), 2)
        code, s = self.qwen_env("witness", STRIX_COHORT_ROWS="16")
        self.assertEqual((code, s["verdict"]), (0, "EXACT"), s["message"])

    def test_witness_grouped_but_separate_is_no_witness(self):
        for fault in ("witness-separate", "witness-fallback"):
            code, s = self.qwen_env(fault, STRIX_MTP_BATCH_JOINT="1")
            self.assertEqual((code, s["verdict"]), (2, "NO-WITNESS"), fault + ": " + s["message"])
        detail = next(r["witness_detail"] for r in s["rounds"] if r["pair_round"])
        self.assertGreaterEqual(detail["joint_fallbacks"], 1)
        self.assertIn("no_owner", detail["verify_declines"])

    def test_witness_draft_batch_arm(self):
        code, s = self.qwen_env("witness", STRIX_MTP_DRAFT_BATCH="1", STRIX_MTP_BATCH_JOINT="1")
        self.assertEqual((code, s["verdict"]), (0, "EXACT"), s["message"])
        detail = next(r["witness_detail"] for r in s["rounds"] if r["pair_round"])
        self.assertEqual(detail["arms"], ["draft_batch", "joint"])
        self.assertTrue(detail["draft_shared"])
        self.assertGreaterEqual(detail["draft_max_rows"], 2)
        code, s = self.qwen_env("witness-draft-declined", STRIX_MTP_DRAFT_BATCH="1")
        self.assertEqual((code, s["verdict"]), (2, "NO-WITNESS"), s["message"])
        detail = next(r["witness_detail"] for r in s["rounds"] if r["pair_round"])
        self.assertEqual((detail["verify_shared"], detail["draft_shared"]), (True, False))
        self.assertIn("sampled", detail["draft_declines"])
        self.assertIn("request.sampled", detail["draft_declines"])

    def test_witness_cycle_lines_alone_suffice(self):
        code, s = self.qwen_env("witness,witness-no-receipt", STRIX_MTP_BATCH_JOINT="1")
        self.assertEqual((code, s["verdict"]), (0, "EXACT"), s["message"])
        self.assertIsNone(next(r["witness_detail"] for r in s["rounds"] if r["pair_round"])["receipts"])

    def test_r2_2_nemotron_live_width_is_not_a_witness(self):
        code, s = self.nemotron("--modes", "4")
        self.assertEqual((code, s["verdict"]), (5, "UNQUALIFIED"), s["message"])
        self.assertNotIn("rows", s)

    def test_r2_3_each_prompt_needs_its_own_witnessed_round(self):
        code, s = self.run_gate("deepseek", self.ids, "--modes", "4", clients="2", fault="serial-after-3")
        self.assertEqual((code, s["verdict"]), (5, "UNQUALIFIED"), s["message"])
        self.assertIn("[3]", s["message"])
        self.assertEqual([r["witness"] for r in s["rounds"]], [True, False])

    def test_r2_4_legacy_qwen_route_is_labelled(self):
        real = gate.subprocess.call
        try:
            for rc, verdict in ((0, "LEGACY-PASS"), (1, "LEGACY-FAIL")):
                gate.subprocess.call = lambda argv, rc=rc: rc
                out = self.dir / ("legacy-" + str(rc))
                code = gate.main(["--engine", "qwen", "--prompt", self.ids[0], "--out", str(out), "--", "bin", "M"])
                s = json.loads((out / "summary.json").read_text())
                self.assertEqual((code, s["verdict"]), (rc, verdict))
                self.assertIn("stream completion not validated", s["coverage"])
                self.assertIn("--generic", s["message"])
                self.assertNotIn("rows", s)
        finally:
            gate.subprocess.call = real

    def test_r2_5_qwen_full_component_manifest(self):
        code, s = self.run_gate("qwen", self.ids, "--generic", "--modes", "4", fault="no-kv")
        self.assertEqual((code, s["verdict"]), (4, "NO-REFERENCE"), s["message"])
        self.assertIn("kv.L", s["message"])

    def test_r2_6_dspark_windows_in_mixed_prefill_steps(self):
        # clients 2,2: the second cell has window baselines even without priming (old false EXACT path)
        code, s = self.run_gate("deepseek", self.ids, "--modes", "4", clients="2,2", fault="mixed-prefill",
                                command_extra=("--dspark", "d.gguf"))
        self.assertEqual((code, s["verdict"]), (2, "NO-WITNESS"), s["message"])

    def test_r2_7_dspark_counters_primed_before_first_round(self):
        code, s = self.run_gate("deepseek", self.ids, "--modes", "4", clients="2", command_extra=("--dspark", "d.gguf"))
        self.assertEqual((code, s["verdict"]), (0, "EXACT"), s["message"])
        self.assertIn("windows", s["rounds"][0]["counters"][0])

    # -- helpers

    def test_rows_match_kit_exactness_shape(self):
        code, s = self.run_gate("deepseek", self.ids, "--modes", "2")
        self.assertEqual(code, 0, s["message"])
        for row in s["rows"]:
            for key in ("prompt_sha256", "token_ids_sha256"):
                self.assertRegex(row[key], "^[0-9a-f]{64}$")
            self.assertEqual(row["verdict"], "EXACT")

    def test_forbidden_flags_and_distinct_prompts(self):
        code, s = self.run_gate("deepseek", self.ids, "--modes", "2", command_extra=("--slots", "2"))
        self.assertEqual(code, 4)
        code, s = self.run_gate("minimax", self.texts, "--modes", "16")
        self.assertEqual(code, 4)
        code, s = self.run_gate("deepseek", [self.ids[0], self.ids[0]], "--modes", "2")
        self.assertEqual(code, 4)
        self.assertIn("distinct", s["message"])

    def test_fnv_matches_engine_definition(self):
        self.assertEqual(gate.fnv64([]), "cbf29ce484222325")
        self.assertEqual(gate.fnv64([1, 2, 3]), fnv64([1, 2, 3]))

    def test_legacy_qwen_argv_is_forwarded_unchanged(self):
        raw = ["--engine", "qwen", "--prompt", "A.ids", "--out", "/x", "--modes", "auto", "--export-block-rows",
               "8192", "--", "bin", "MODEL", "--generate", "4"]
        argv = gate.legacy_qwen_argv(None, raw)
        self.assertEqual(argv[1], str(gate.LEGACY_QWEN_GATE))
        self.assertEqual(argv[2:], raw[2:])
        self.assertTrue(gate.LEGACY_QWEN_GATE.is_file())

    def test_peer_command_and_remote_scripts(self):
        class A:
            peer_command = None
            command = ["env", "STRIX_DS41_PACKS=/p", "bin/strix-deepseek41-serve", "M.gguf", "--rank", "0",
                       "--transport", "usb4auto"]
            port = 18080
        argv = gate.peer_command(A, gate.ADAPTERS["deepseek"], "4")
        self.assertEqual(argv[argv.index("--rank") + 1], "1")
        self.assertEqual(argv[-2:], ["--slots", "4"])
        script = gate.remote_start_script("/d/run dir", "slots-4", argv)
        self.assertEqual(shlex.split(script)[:2], ["cd", "/d/run dir"])
        self.assertIn("STARTED", script)
        stop = gate.remote_stop_script("/d/run dir", "slots-4")
        self.assertIn("'/d/run dir/slots-4'", stop)
        self.assertIn("kill -INT", stop)
        A.command = ["bin", "M.gguf"]
        with self.assertRaises(gate.GateError):
            gate.peer_command(A, gate.ADAPTERS["deepseek"], "4")

    @unittest.skipIf(os.name == "nt" or not shutil.which("setsid"), "needs bash + setsid")
    def test_remote_scripts_run_locally(self):
        d = self.dir / "peer"
        d.mkdir()
        start = gate.remote_start_script(str(d), "x", [sys.executable, "-c", "import sys; sys.exit(3)"])
        out = subprocess.run(["bash", "-c", start], capture_output=True, text=True, timeout=30)
        self.assertIn("STARTED", out.stdout)
        stop = subprocess.run(["bash", "-c", gate.remote_stop_script(str(d), "x", grace=10)],
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(stop.stdout.strip(), "RC=3")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--fake-server"]:
        sys.exit(fake_server(sys.argv[2:]))
    unittest.main()
