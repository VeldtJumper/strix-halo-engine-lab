"""Task suite (scripts/bench/tasks): tasks.json is what build.py builds, and run_tasks.py against a fake SSE chat
server: (N-1)/(t_last-t_first), ids digest from the strix receipt, spec_stats passthrough, compare EQUAL/MISMATCH."""
import http.server
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
import unittest

TASKS = pathlib.Path(__file__).resolve().parents[1] / "tasks"
spec = importlib.util.spec_from_file_location("run_tasks", TASKS / "run_tasks.py")
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

IDS = list(range(100, 140))
STATS = {"passes": 16, "emitted": 40, "tokens_per_pass": 2.5,
         "sources": {"serial": {"passes": 1}, "lookup": {"passes": 1}, "model": {"passes": 0}}}


class Handler(http.server.BaseHTTPRequestHandler):
    ids = IDS
    requests = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path != "/v1/models":
            self.send_response(404)
            self.end_headers()
            return
        msg = b'{"object":"list","data":[{"id":"fake-model","object":"model"}]}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(msg)))
        self.end_headers()
        self.wfile.write(msg)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.requests.append(body)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(len(Handler.ids)):  # one chunk per token, ~15 ms apart
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": f"w{i} "}}]}).encode() + b"\n\n")
            self.wfile.flush()
            time.sleep(0.015)
        final = {"choices": [{"delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 9, "completion_tokens": len(Handler.ids)},
                 "strix": {"token_ids": Handler.ids, "spec_stats": STATS, "decode_ms": 100.0}}
        self.wfile.write(b"data: " + json.dumps(final).encode() + b"\n\ndata: [DONE]\n\n")


class TaskSuite(unittest.TestCase):
    def test_tasks_file_is_current(self):
        subprocess.run([sys.executable, str(TASKS / "build.py"), "--check"], check=True)
        doc = json.loads((TASKS / "tasks.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["mode"], "chat")
        self.assertFalse(doc["thinking"])
        ids = [t["id"] for t in doc["tasks"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("code-edit", ids)
        self.assertTrue(all(t["max_tokens"] in (4096, 8192) for t in doc["tasks"]))

    def test_client_and_compare(self):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with tempfile.TemporaryDirectory() as d:
                rc = rt.main(["--url", url, "--out", d + "/a", "--label", "off", "--tasks", "sql,prose-en", "--warmup", "1"])
                self.assertEqual(rc, 0)
                doc = json.loads(pathlib.Path(d, "a", "tasks-results.json").read_text())
                self.assertEqual([r["id"] for r in doc["results"]], ["sql", "prose-en"])
                r = doc["results"][0]
                self.assertEqual(r["tokens"], 40)
                self.assertTrue(r["thinking_off"])
                self.assertEqual(r["ids_sha256"], rt.id_hash(IDS))
                self.assertEqual(r["spec_stats"]["tokens_per_pass"], 2.5)
                self.assertEqual(r["finish_reason"], "stop")
                # 40 chunks ~15 ms apart: 39 tokens over >= 0.585 s
                self.assertGreater(r["decode_tok_s"], 25)
                self.assertLess(r["decode_tok_s"], 67)
                body = Handler.requests[-1]
                self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
                self.assertEqual(body["temperature"], 0)
                self.assertEqual(body["model"], "fake-model")  # from GET /v1/models
                self.assertTrue(body["stream"])
                self.assertEqual(len(Handler.requests), 3)  # warm-up + 2 tasks
                rc = rt.main(["--url", url, "--out", d + "/b", "--label", "on", "--tasks", "sql,prose-en", "--warmup", "0"])
                self.assertEqual(rt.compare(d + "/a/tasks-results.json", d + "/b/tasks-results.json"), 0)
                self.assertEqual(rt.table([d + "/a/tasks-results.json", d + "/b/tasks-results.json"]), 0)
                Handler.ids = IDS[:5] + [7] + IDS[6:]
                rt.main(["--url", url, "--out", d + "/c", "--label", "bad", "--tasks", "sql", "--warmup", "0"])
                self.assertEqual(rt.compare(d + "/a/tasks-results.json", d + "/c/tasks-results.json"), 1)
        finally:
            Handler.ids = IDS
            server.shutdown()


class StrictHandler(Handler):
    """A server that rejects unknown fields (400) and returns the ids as generated_ids (gemma4 serve style)."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if "return_token_ids" in body:
            msg = b'{"error":{"message":"unsupported field: return_token_ids"}}'
            self.send_response(400)
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        self.send_response(200)
        self.end_headers()
        for i in range(40):
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": f"w{i} "}}]}).encode() + b"\n\n")
            self.wfile.flush()
            time.sleep(0.015)
        final = {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"completion_tokens": 40},
                 "strix": {"generated_ids": IDS}}
        self.wfile.write(b"data: " + json.dumps(final).encode() + b"\n\ndata: [DONE]\n\n")


class StrictServer(unittest.TestCase):
    def test_optional_field_dropped(self):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StrictHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as d:
                rc = rt.main(["--url", f"http://127.0.0.1:{server.server_address[1]}", "--out", d, "--tasks", "sql",
                              "--warmup", "0"])
                self.assertEqual(rc, 0)
                r = json.loads(pathlib.Path(d, "tasks-results.json").read_text())["results"][0]
                self.assertEqual(r["ids_sha256"], rt.id_hash(IDS))
        finally:
            rt._dropped.clear()
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
