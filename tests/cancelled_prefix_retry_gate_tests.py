#!/usr/bin/env python3
"""Offline fail-closed controls for cancelled-prefix per-arm receipts; no model/GPU."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import cancelled_prefix_retry_gpu_gate as gate

D = "a" * 64
E = "b" * 64
PROMPT = [1, 2, 3, 4]


def record(arm="offdef", refusal=False):
    cached = 0 if arm == "fresh" or refusal else 2
    order = 3 if arm == "offdef" else 1
    n, fnv = gate.prompt_key(PROMPT)
    handoff = dict(request_order=order, prompt_tokens=n, prompt_fnv=fnv, restored_position=cached,
                   state_digest=D, terminal_logits_digest=D, capture_counters_global=dict(tripwire=0),
                   terminal_metadata=dict(committed_tokens=n, next_position=n, predictor_frontier=n-1,
                                          generations_lineage_digest=D),
                   cache_refusal="batch-suffix-not-solo" if refusal and arm == "offdef" else "")
    final = dict(request_order=order, prompt_tokens=n, prompt_fnv=fnv, generated_ids_count=2,
                 output_logits_rows=2, output_logits_chain=D, state_digest=D,
                 generations_lineage_digest=D, committed_tokens=6, next_position=6,
                 predictor_frontier=5, pending_token=7)
    retry = dict(strix=dict(token_ids=[5, 6], prompt_state_digest=D, final_state_position=6,
                           final_state_components={"target.rows": D, "mtp.held": D}),
                 usage=dict(prompt_tokens_details=dict(cached_tokens=cached)))
    sig, cached, reason = gate.exact_signature(retry, PROMPT, [handoff], [final])
    binding = dict(schema=1, prompt=gate.prompt_key(PROMPT), scheduler_defaults=refusal, expect_restored=2,
                   command=["engine", "model"], files={"engine": D, "model": D}, client=D)
    return dict(binding=binding, result=dict(arm=arm, signature=sig, cached_tokens=cached,
                                            retry_cache_refusal=reason, elapsed_s=10),
                batch=dict(mode="native-mtp-batch"), retry=retry, prompt_ids=PROMPT,
                prompts=[handoff], finals=[final])


def resign(row):
    sig, cached, reason = gate.exact_signature(row["retry"], row["prompt_ids"], row["prompts"], row["finals"])
    row["result"].update(signature=sig, cached_tokens=cached, retry_cache_refusal=reason)


class Receipts(unittest.TestCase):
    def compare(self, rows):
        with redirect_stdout(io.StringIO()):
            gate.compare(rows)

    def test_positive_restore_and_refusal(self):
        self.compare([record(), record("fresh")])
        self.compare([record(refusal=True), record("fresh", refusal=True)])

    def test_request_local_counts_required(self):
        for arm in ("offdef", "fresh"):
            for value in (None, -1, True, "0", 0.0, "absent"):
                with self.subTest(arm=arm, value=value):
                    rows = [record(), record("fresh")]
                    row = next(r for r in rows if r["result"]["arm"] == arm)
                    details = row["retry"]["usage"]["prompt_tokens_details"]
                    if value == "absent":
                        details.pop("cached_tokens")
                    else:
                        details["cached_tokens"] = value
                    with self.assertRaises(RuntimeError):
                        self.compare(rows)

    def test_restore_and_refusal_contradictions(self):
        rows = [record(), record("fresh")]
        rows[0]["prompts"][0]["restored_position"] = 0
        with self.assertRaisesRegex(RuntimeError, "contradictory restore"):
            self.compare(rows)
        for value in (None, -1, True, "absent"):
            rows = [record(), record("fresh")]
            if value == "absent":
                rows[1]["prompts"][0].pop("restored_position")
            else:
                rows[1]["prompts"][0]["restored_position"] = value
            with self.assertRaises(RuntimeError):
                self.compare(rows)
        rows = [record(refusal=True), record("fresh", refusal=True)]
        rows[0]["prompts"][0]["cache_refusal"] = "prefix-no-compatible-checkpoint"
        resign(rows[0])
        with self.assertRaisesRegex(RuntimeError, "batch-suffix-not-solo"):
            self.compare(rows)

    def test_missing_metadata_and_row_coverage(self):
        for field in ("terminal_metadata", "terminal_logits_digest", "state_digest", "capture_counters_global"):
            rows = [record(), record("fresh")]
            rows[0]["prompts"][0].pop(field)
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                self.compare(rows)
        for field in ("predictor_frontier", "pending_token", "generations_lineage_digest", "state_digest",
                      "committed_tokens", "next_position", "output_logits_rows", "generated_ids_count", "output_logits_chain"):
            rows = [record(), record("fresh")]
            rows[0]["finals"][0].pop(field)
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                self.compare(rows)
        rows = [record(), record("fresh")]
        rows[0]["finals"][0]["output_logits_rows"] = 1
        with self.assertRaisesRegex(RuntimeError, "row coverage"):
            self.compare(rows)

    def test_canonical_mismatches(self):
        for section, field, value in (("finals", "predictor_frontier", 4),
                                      ("finals", "generations_lineage_digest", E),
                                      ("finals", "state_digest", E)):
            rows = [record(), record("fresh")]
            rows[0][section][0][field] = value
            resign(rows[0])
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "MISMATCH"):
                self.compare(rows)
        for field, value in (("predictor_frontier", 2), ("generations_lineage_digest", E)):
            rows = [record(), record("fresh")]
            rows[0]["prompts"][0]["terminal_metadata"][field] = value
            resign(rows[0])
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "MISMATCH"):
                self.compare(rows)
        rows = [record(), record("fresh")]
        rows[0]["retry"]["strix"]["final_state_components"]["mtp.held"] = E
        resign(rows[0])
        with self.assertRaisesRegex(RuntimeError, "MISMATCH"):
            self.compare(rows)

    def test_identity_order_tripwire_and_binding(self):
        for field, value in (("request_order", 8), ("prompt_fnv", "0000000000000000"),
                             ("capture_counters_global", dict(tripwire=1))):
            rows = [record(), record("fresh")]
            rows[0]["prompts"][0][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                self.compare(rows)
        rows = [record(), record("fresh")]
        rows[0]["binding"]["files"]["engine"] = E
        with self.assertRaisesRegex(RuntimeError, "configuration-bound"):
            self.compare(rows)
        rows = [record(), record("fresh")]
        rows[0]["result"]["elapsed_s"] = 420
        with self.assertRaisesRegex(RuntimeError, "turn cap"):
            self.compare(rows)
        with self.assertRaises(RuntimeError):
            self.compare([record()])

    def test_reviewer_negative_reproductions_cli(self):
        cases = []
        rows = [record(), record("fresh")]
        for row in rows:
            row["retry"]["usage"]["prompt_tokens_details"].pop("cached_tokens")
            row["prompts"][0]["restored_position"] = 0
            row["totals"] = dict(cached_tokens=4096 if row["result"]["arm"] == "offdef" else 0)
        cases.append(("aggregate-fallback", rows, "request-local"))
        rows = [record(), record("fresh")]
        rows[0]["prompts"][0]["terminal_metadata"].update(predictor_frontier=2, generations_lineage_digest=E)
        rows[0]["finals"][0].update(predictor_frontier=4, generations_lineage_digest=E, state_digest=E)
        rows[0]["retry"]["strix"]["final_state_components"]["mtp.held"] = E
        resign(rows[0])
        cases.append(("lineage-frontier-final-mtp", rows, "MISMATCH"))
        with tempfile.TemporaryDirectory() as directory:
            for name, rows, error in cases:
                paths = [Path(directory) / (arm + ".json") for arm in ("offdef", "fresh")]
                for path, row in zip(paths, rows):
                    path.write_text(json.dumps(row))
                run = subprocess.run([sys.executable, gate.__file__, "--compare", *map(str, paths)],
                                     capture_output=True, text=True, timeout=5)
                self.assertNotEqual(run.returncode, 0, name)
                self.assertNotIn("EXACT:", run.stdout, name)
                self.assertIn(error, run.stderr, name)
                print(f"negative reproduction {name}: FAIL/unqualified (expected)")


class Deadline(unittest.TestCase):
    def test_interruptible_reads_headers_completion_cancel_and_status(self):
        for phase in ("headers", "completion", "cancel", "status"):
            with self.subTest(phase=phase):
                class Handler(socketserver.BaseRequestHandler):
                    def handle(self):
                        self.request.recv(65536)
                        if phase != "headers":
                            self.request.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                        time.sleep(1)
                class Server(socketserver.ThreadingTCPServer):
                    daemon_threads = True
                with Server(("127.0.0.1", 0), Handler) as server:
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    port = server.server_address[1]
                    started = time.monotonic()
                    try:
                        with self.assertRaises((RuntimeError, OSError, ValueError, gate.http.client.HTTPException)):
                            end = time.monotonic() + 0.15
                            if phase == "status":
                                gate.status(port, end)
                            elif phase == "cancel":
                                gate.cancel_mid_prefill(port, PROMPT, 2, 2, end)
                            else:
                                gate.completion(port, PROMPT, 2, end)
                        self.assertLess(time.monotonic() - started, 0.8)
                    finally:
                        server.shutdown()
                thread.join()


if __name__ == "__main__":
    unittest.main(verbosity=2)
