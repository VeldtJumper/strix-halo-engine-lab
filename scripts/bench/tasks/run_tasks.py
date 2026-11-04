#!/usr/bin/env python3
"""Single-stream task suite client: runs tasks.json against one running serve endpoint (any engine).

Per task one /v1/chat/completions request: stream, temperature 0, thinking off (chat_template_kwargs), max_tokens
from the task, C1 (one request at a time). The client timestamps every SSE content chunk:

    decode tok/s = (N - 1) / (t_last - t_first)     N = completion tokens (usage), load and prefill excluded
    TTFT         = t_first - t_request

From the final chunk's `strix` receipt it keeps token_ids (-> ids_sha256), spec_stats (the shared
`strix-spec-stats` JSON: passes, tokens/pass, source mix, acceptance by depth, rows), decode_ms/prefill_ms and the
engine identity fields when present. Missing fields are recorded as absent, never invented.

usage: run_tasks.py --url http://127.0.0.1:18880 --out DIR [--label NAME] [--tasks a,b] [--warmup 1]
                    [--repeats 1] [--tasks-file FILE] [--server-pid PID] [--timeout 1800]
       run_tasks.py compare BASE.json CANDIDATE.json   (exactness + speed ratio per task; exit 1 on a mismatch)
       run_tasks.py table RUN.json...                 (one row per run: tok/s and tokens/pass per task)
Writes DIR/tasks-results.json and DIR/tasks-results.md; exit 1 when a request fails.
"""
import argparse
import hashlib
import json
import pathlib
import platform
import statistics
import sys
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
SCHEMA = "strix-bench-task-results/1"


def id_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def peak_rss_kib(pid):
    """VmHWM of a local server process (Linux); None when unavailable."""
    try:
        for line in pathlib.Path(f"/proc/{int(pid)}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def drafting_stats(receipt):
    """The shared spec_stats JSON, or the same fields normalised from an engine's own receipt (marked 'source')."""
    if isinstance(receipt.get("spec_stats"), dict):
        return receipt["spec_stats"]
    phases = receipt.get("phases")
    if isinstance(phases, list) and phases and isinstance(phases[0], dict) and isinstance(phases[0].get("spec_stats"), dict):
        return phases[0]["spec_stats"]  # gemma4: the drafting session reports into its phase receipt
    acc = phases[0].get("acceptance") if isinstance(phases, list) and phases and isinstance(phases[0], dict) else None
    if isinstance(acc, dict) and isinstance(acc.get("cycles"), int) and acc["cycles"] > 0:  # gemma4 serve
        out = {"source": "gemma4-acceptance", "passes": acc["cycles"], "emitted": acc.get("emitted"),
               "tokens_per_pass": acc["emitted"] / acc["cycles"] if isinstance(acc.get("emitted"), int) else None,
               "depth_offered": acc.get("depth_eligible"), "depth_accepted": acc.get("depth_accepted"),
               "proposed": acc.get("attempted"), "accepted": acc.get("accepted")}
        lookup = phases[0].get("prompt_lookup")
        if isinstance(lookup, dict):
            out["lookup"] = {k: lookup.get(k) for k in ("cycles", "proposed", "accepted")}
        return out
    if isinstance(receipt.get("mtp_cycles"), int) and receipt["mtp_cycles"] > 0:  # shared serve front-end (MTP)
        return {"source": "serve-mtp", "passes": receipt["mtp_cycles"], "proposed": receipt.get("mtp_proposed"),
                "accepted": receipt.get("mtp_accepted"),
                "lookup": {"proposed": receipt.get("ngram_rows"), "accepted": receipt.get("ngram_accepted")},
                "rows": receipt.get("mtp_chosen_window_rows_histogram")}
    return None


OPTIONAL_FIELDS = {"return_token_ids"}  # dropped for servers that reject them (their receipt carries the ids)
_dropped = set()


def open_stream(url, body, timeout):
    """POST the request; a server that rejects an optional field (400 naming it) is asked again without it."""
    while True:
        req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            field = next((f for f in OPTIONAL_FIELDS if f in body and f in detail), None)
            if e.code != 400 or field is None:
                raise ValueError(f"HTTP {e.code}: {detail}") from None
            _dropped.add(field)
            body = {k: v for k, v in body.items() if k != field}


def served_model(url):
    """The first model id the server lists, or None (the request then names no model)."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=10) as r:
            data = json.loads(r.read()).get("data") or []
        return data[0]["id"] if data and isinstance(data[0].get("id"), str) else None
    except (OSError, ValueError, KeyError):
        return None


def chat(url, task, timeout, model):
    body = {"messages": [{"role": "user", "content": task["text"]}], "max_tokens": int(task["max_tokens"]),
            "temperature": 0, "stream": True, "return_token_ids": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    if model:
        body["model"] = model
    for field in _dropped:
        body.pop(field, None)
    start = time.perf_counter()
    first = last = None
    chunks = reasoning = 0
    text = []
    usage = receipt = finish = None
    with open_stream(url, body, timeout) as response:
        for raw in response:
            raw = raw.strip()
            if not raw.startswith(b"data:") or raw == b"data: [DONE]":
                continue
            obj = json.loads(raw[5:])
            if "error" in obj:
                raise ValueError(f"server error: {obj['error']}")
            now = time.perf_counter()
            for choice in obj.get("choices", []):
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content")
                if piece:
                    first = now if first is None else first
                    last = now
                    chunks += 1
                    reasoning += bool(delta.get("reasoning_content"))
                    text.append(piece)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
            if obj.get("usage"):
                usage = obj["usage"]
            if obj.get("strix"):
                receipt = obj["strix"]
    if first is None or not usage or not isinstance(usage.get("completion_tokens"), int):
        raise ValueError("SSE response lacks content or completion usage")
    n = usage["completion_tokens"]
    out = "".join(text)
    # A rate needs a decode span the client clock can resolve: >= 32 tokens over >= 0.5 s, in >= 4 chunks.
    timed = n >= 32 and last - first >= 0.5 and chunks >= 4
    row = {"tokens": n, "prompt_tokens": usage.get("prompt_tokens"), "finish_reason": finish,
           "ttft_s": first - start, "decode_s": last - first, "chunks": chunks,
           "decode_tok_s": (n - 1) / (last - first) if timed else None,
           "untimed_reason": None if timed else "decode span too short for the client clock",
           # thinking off must be honoured, not only requested
           "thinking_off": reasoning == 0 and "<think>" not in out,
           "text_sha256": hashlib.sha256(out.encode()).hexdigest(), "text_chars": len(out)}
    if isinstance(receipt, dict):
        ids = receipt.get("token_ids", receipt.get("generated_ids"))
        if isinstance(ids, list):
            row["ids_sha256"] = id_hash(ids)
            row["ids_count"] = len(ids)
        for key in ("decode_ms", "prefill_ms", "queue_ms"):
            if isinstance(receipt.get(key), (int, float)):
                row["engine_" + key] = receipt[key]
        stats = drafting_stats(receipt)
        if stats:
            if stats.get("tokens_per_pass") is None and stats.get("passes"):
                stats["tokens_per_pass"] = n / stats["passes"]  # completion tokens per target pass
            row["spec_stats"] = stats
        for key in ("profile", "identity", "build", "model"):
            if key in receipt:
                row["engine_" + key] = receipt[key]
    return row, out


def summary(rows):
    rates = [r["decode_tok_s"] for r in rows if r.get("decode_tok_s")]
    head = dict(rows[0])
    head["repeats"] = len(rows)
    head["decode_tok_s"] = statistics.median(rates) if rates else None
    head["decode_tok_s_all"] = rates
    head["ttft_s"] = statistics.median(r["ttft_s"] for r in rows)
    head["repeat_stable"] = len({r.get("ids_sha256") or r["text_sha256"] for r in rows}) == 1
    return head


def markdown(doc):
    lines = [f"# Task suite: {doc['label']}", "", f"Endpoint {doc['url']}; {doc['host']['node']} kernel "
             f"{doc['host']['kernel']}; tasks sha256 {doc['tasks_sha256'][:12]}; C1 greedy, thinking off.", "",
             "| task | N | finish | decode tok/s (N-1)/t | TTFT s | tokens/pass | lookup / model / serial passes | ids |",
             "|---|---:|---|---:|---:|---:|---|---|"]
    for r in doc["results"]:
        s = r.get("spec_stats") or {}
        src = s.get("sources") or {}
        mix = " / ".join(str((src.get(k) or {}).get("passes", "-")) for k in ("lookup", "model", "serial")) if src else "-"
        tpp = f"{s['tokens_per_pass']:.2f}" if isinstance(s.get("tokens_per_pass"), (int, float)) else "-"
        rate = f"{r['decode_tok_s']:.1f}" if r.get("decode_tok_s") else "-"
        ids = (r.get("ids_sha256") or "text:" + r["text_sha256"])[:12]
        lines.append(f"| {r['id']} | {r['tokens']} | {r['finish_reason']} | {rate} | {r['ttft_s']:.2f} | {tpp} | {mix} | {ids} |")
    return "\n".join(lines) + "\n"


def write(out, doc, pid):
    if pid:
        doc["server_peak_rss_kib"] = peak_rss_kib(pid)
    (out / "tasks-results.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    (out / "tasks-results.md").write_text(markdown(doc), encoding="utf-8")


def compare(base_path, cand_path):
    base, cand = (json.loads(pathlib.Path(p).read_text(encoding="utf-8")) for p in (base_path, cand_path))
    if base["tasks_sha256"] != cand["tasks_sha256"]:
        sys.exit("different task files")
    by_id = {r["id"]: r for r in base["results"]}
    rc = 0
    print(f"| task | {base['label']} tok/s | {cand['label']} tok/s | ratio | ids |")
    print("|---|---:|---:|---:|---|")
    for r in cand["results"]:
        b = by_id.get(r["id"])
        if not b:
            continue
        key = "ids_sha256" if b.get("ids_sha256") and r.get("ids_sha256") else "text_sha256"
        same = b[key] == r[key]
        rc |= not same
        ratio = r["decode_tok_s"] / b["decode_tok_s"] if r.get("decode_tok_s") and b.get("decode_tok_s") else None
        fmt = lambda v: f"{v:.1f}" if v else "-"
        print(f"| {r['id']} | {fmt(b.get('decode_tok_s'))} | {fmt(r.get('decode_tok_s'))} | "
              f"{ratio:.3f} | {'EQUAL' if same else 'MISMATCH'} ({key.split('_')[0]}) |" if ratio else
              f"| {r['id']} | - | - | - | {'EQUAL' if same else 'MISMATCH'} |")
    return rc


def table(paths):
    """One row per run (label), one column per task: decode tok/s and tokens per pass when reported."""
    docs = [json.loads(pathlib.Path(p).read_text(encoding="utf-8")) for p in paths]
    if len({d["tasks_sha256"] for d in docs}) != 1:
        sys.exit("runs use different task files")
    ids = []
    for d in docs:
        ids += [r["id"] for r in d["results"] if r["id"] not in ids]
    print("| run | " + " | ".join(ids) + " |")
    print("|---|" + "---:|" * len(ids))
    for d in docs:
        by_id = {r["id"]: r for r in d["results"]}
        cells = []
        for i in ids:
            r = by_id.get(i)
            if not r or not r.get("decode_tok_s"):
                cells.append("-")
                continue
            tpp = (r.get("spec_stats") or {}).get("tokens_per_pass")
            cells.append(f"{r['decode_tok_s']:.1f}" + (f" ({tpp:.2f}/pass)" if isinstance(tpp, (int, float)) else ""))
        print(f"| {d['label']} | " + " | ".join(cells) + " |")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["compare"]:
        if len(argv) != 3:
            sys.exit("usage: run_tasks.py compare BASE.json CANDIDATE.json")
        return compare(argv[1], argv[2])
    if argv[:1] == ["table"]:
        if len(argv) < 2:
            sys.exit("usage: run_tasks.py table RUN.json...")
        return table(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="engine")
    ap.add_argument("--model", default="auto", help="model id; auto = the first id of GET /v1/models (else none sent)")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--tasks-file", default=str(HERE / "tasks.json"))
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--server-pid", type=int)
    a = ap.parse_args(argv)
    raw = pathlib.Path(a.tasks_file).read_bytes()
    suite = json.loads(raw)
    tasks = suite["tasks"]
    if a.tasks:
        wanted = a.tasks.split(",")
        unknown = set(wanted) - {t["id"] for t in tasks}
        if unknown:
            sys.exit(f"unknown task(s): {', '.join(sorted(unknown))}")
        tasks = [t for t in tasks if t["id"] in wanted]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"schema": SCHEMA, "label": a.label, "url": a.url, "tasks_sha256": hashlib.sha256(raw).hexdigest(),
           "protocol": {"concurrency": 1, "temperature": 0, "thinking": False, "warmup": a.warmup, "repeats": a.repeats,
                        "rate": "(N-1)/(t_last_chunk - t_first_chunk), client clock"},
           "host": {"node": platform.node(), "kernel": platform.release(), "system": platform.system()},
           "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": [], "failures": []}
    if a.model == "auto":
        a.model = served_model(a.url)
    doc["model"] = a.model
    for i in range(a.warmup):  # warm-up: the first task cut to 64 tokens, excluded
        try:
            chat(a.url, dict(tasks[0], max_tokens=64), a.timeout, a.model)
        except (OSError, ValueError) as e:
            # a request the server rejects would fail every task: stop here, keep the reason
            doc["failures"].append({"id": tasks[0]["id"], "warmup": i, "error": str(e)})
            write(out, doc, a.server_pid)
            print(f"warm-up failed: {e}", file=sys.stderr)
            return 1
    for task in tasks:
        rows = []
        for rep in range(a.repeats):
            try:
                row, text = chat(a.url, task, a.timeout, a.model)
            except (OSError, ValueError) as e:
                doc["failures"].append({"id": task["id"], "repeat": rep, "error": str(e)})
                continue
            (out / f"{task['id']}-{rep}.txt").write_text(text, encoding="utf-8")
            rows.append(row)
        if rows:
            doc["results"].append({"id": task["id"], "category": task["category"], "max_tokens": task["max_tokens"],
                                   **summary(rows)})
        write(out, doc, a.server_pid)  # after every task: a turn cut at its limit keeps what finished
    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write(out, doc, a.server_pid)
    print(markdown(doc))
    return 1 if doc["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
