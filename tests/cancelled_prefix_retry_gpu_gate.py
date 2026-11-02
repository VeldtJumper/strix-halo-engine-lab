#!/usr/bin/env python3
"""Cancel a certified 4096 checkpoint, then compare retry against a fresh request.

Uses explicit diagnostic L0 on the batch runner. Solo suffix scheduling is required
for a restore witness; --scheduler-defaults exercises the named refusal instead.
Run one --arm per turn (<=400 s including cleanup), then --compare offline.
No predictor field is excluded: measured retry runs after the optional decoder retires.
Receipts and supplied prompt/model assets must stay outside the checkout.
"""
import argparse
from contextlib import contextmanager
import hashlib
import concurrent.futures
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_service_gpu_gate import require, tokens
from prefix_cache_gpu_gate import prompt_key  # noqa: E402


def remaining(deadline):
    seconds = deadline - time.monotonic()
    require(seconds > 0, "FAIL/unqualified: turn budget exhausted")
    return seconds


@contextmanager
def response_for(port, method, path, body, deadline):
    """Interrupt the socket even when a peer trickles bytes or stalls SSE/header reads."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=remaining(deadline))
    response = None
    def interrupt():
        sock = connection.sock
        if sock is None and response is not None and response.fp is not None:
            sock = getattr(response.fp.raw, "_sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
    timer = threading.Timer(remaining(deadline), interrupt)
    timer.daemon = True
    timer.start()
    try:
        connection.request(method, path, body, {"Content-Type": "application/json"})
        remaining(deadline)
        if connection.sock is not None:
            connection.sock.settimeout(remaining(deadline))
        response = connection.getresponse()
        require(response.status == 200, f"HTTP {response.status}: {path}")
        yield response, interrupt
        remaining(deadline)
    finally:
        timer.cancel()
        interrupt()
        if response is not None:
            response.close()
        connection.close()


def status(port, deadline):
    with response_for(port, "GET", "/api/status", None, deadline) as (response, _):
        return json.loads(response.read())


def completion(port, prompt, count, deadline, first=None):
    result, arrivals = None, []
    body = json.dumps(dict(prompt=prompt, max_tokens=count, temperature=0,
                           stream=True, return_token_ids=True))
    with response_for(port, "POST", "/v1/completions", body, deadline) as (response, _):
        for line in response:
            remaining(deadline)
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            require("error" not in chunk, f"completion error: {chunk.get('error')}")
            if "strix" in chunk:
                result = chunk
                final_at = time.monotonic()
            else:
                arrivals.append(time.monotonic())
                if first:
                    first.set()
    require(result is not None, "NO-REFERENCE: missing final streaming receipt")
    result.update(completed_at=time.monotonic(), first_at=arrivals[0] if arrivals else final_at,
                  emission_gaps_ms=[1000 * (b - a) for a, b in zip(arrivals, arrivals[1:])])
    return result


def cancel_mid_prefill(port, prompt, cancel_at, generate, deadline, fallback_delay=0.0):
    body = json.dumps(dict(prompt=prompt, max_tokens=generate, temperature=0, stream=True))
    seen, reached, fallback = None, False, threading.Event()
    with response_for(port, "POST", "/v1/completions", body, deadline) as (response, interrupt):
        def close_fallback():
            fallback.set()
            interrupt()
        timer = threading.Timer(fallback_delay, close_fallback) if fallback_delay else None
        if timer:
            timer.daemon = True
            timer.start()
        try:
            while True:
                try:
                    line = response.readline()
                except (OSError, ValueError, http.client.HTTPException):
                    line = b""
                remaining(deadline)
                if not line:
                    break
                if line.startswith(b": progress "):
                    seen = json.loads(line[len(b": progress "):])
                    if seen.get("phase") == "prefill" and seen.get("prefilled_tokens", 0) >= cancel_at:
                        reached = True
                        break
                elif line.startswith(b"data: "):
                    text = line[6:].decode().strip()
                    chunk = json.loads(text) if text.startswith("{") else {}
                    require("error" not in chunk, "cancel request failed: " + text[:500])
                    require(not any(c.get("text") or c.get("finish_reason") for c in chunk.get("choices", [])),
                            "NO-WITNESS: token arrived before cancel point")
        finally:
            if timer:
                timer.cancel()
            interrupt()
    require(reached or fallback.is_set(), "NO-WITNESS: stream ended before cancel point")
    return seen if reached else dict(fallback_delay_s=fallback_delay, last_progress=seen)


def nonnegative(value):
    return type(value) is int and value >= 0


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def exact_signature(retry, prompt, prompts, finals):
    """No predictor exclusions: retry runs after all peers retire, including mtp.*."""
    key = prompt_key(prompt)
    candidates = [r for r in finals if (r.get("prompt_tokens"), r.get("prompt_fnv")) == key]
    require(candidates, "NO-REFERENCE: missing retry final")
    # Cancelled producer can have the same prompt identity; the completed retry has the latest order.
    require(all(nonnegative(r.get("request_order")) for r in candidates), "missing request_order")
    final = max(candidates, key=lambda r: r["request_order"])
    order = final["request_order"]
    require(sum(r["request_order"] == order for r in candidates) == 1, "ambiguous retry final")
    matched = [r for r in prompts if r.get("request_order") == order and
               (r.get("prompt_tokens"), r.get("prompt_fnv")) == key]
    require(len(matched) == 1, "NO-REFERENCE: missing/ambiguous retry prompt identity/order")
    handoff = matched[0]
    require(prompts and all(type(r.get("capture_counters_global", {}).get("tripwire")) is int and
                           r["capture_counters_global"]["tripwire"] == 0 for r in prompts),
            "NO-REFERENCE: missing/nonzero snapshot tripwire")
    require("diagnostic_error" not in final, "incomplete final diagnostics")
    ids = retry.get("strix", {}).get("token_ids")
    require(isinstance(ids, list) and ids and all(nonnegative(i) for i in ids), "missing complete IDs")
    require(nonnegative(final.get("generated_ids_count")) and nonnegative(final.get("output_logits_rows")) and
            len(ids) == final["generated_ids_count"] == final["output_logits_rows"], "incomplete raw-logit row coverage")
    for field in ("output_logits_chain", "state_digest", "generations_lineage_digest"):
        require(digest(final.get(field)), f"NO-REFERENCE: missing final {field}")
    for field in ("committed_tokens", "next_position", "predictor_frontier", "pending_token"):
        require(nonnegative(final.get(field)), f"NO-REFERENCE: missing final {field}")
    require(final["committed_tokens"] == final["next_position"], "uncommitted final frontier")
    for field in ("terminal_logits_digest", "state_digest"):
        require(digest(handoff.get(field)), f"NO-REFERENCE: missing prompt {field}")
    metadata = handoff.get("terminal_metadata")
    require(isinstance(metadata, dict) and digest(metadata.get("generations_lineage_digest")),
            "NO-REFERENCE: missing terminal metadata/lineage")
    require(all(nonnegative(metadata.get(k)) for k in ("committed_tokens", "next_position", "predictor_frontier")),
            "NO-REFERENCE: missing terminal frontier")
    require(metadata["committed_tokens"] == metadata["next_position"] == len(prompt), "invalid prompt frontier")
    cached = retry.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens")
    restored = handoff.get("restored_position")
    require(nonnegative(cached) and nonnegative(restored), "NO-WITNESS: request-local cached_tokens/restored_position required")
    require(cached == restored and cached <= len(prompt), "contradictory restore witnesses")
    state = retry["strix"]
    require(digest(state.get("prompt_state_digest")) and state["prompt_state_digest"] == handoff["state_digest"],
            "missing/contradictory prompt state")
    require(state.get("final_state_position") == final["next_position"], "contradictory final position")
    components = state.get("final_state_components")
    require(isinstance(components, dict) and components and any(k.startswith("mtp.") for k in components) and
            all(digest(v) for v in components.values()), "NO-REFERENCE: missing complete canonical components (including mtp.*)")
    return dict(ids=ids, prompt={k: handoff[k] for k in
                ("terminal_logits_digest", "state_digest", "terminal_metadata")},
                final={k: v for k, v in final.items() if k != "request_order"}, components=components), cached, handoff.get("cache_refusal")


def file_hash(path, deadline):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        while data := source.read(4 * 1024 * 1024):
            remaining(deadline)
            h.update(data)
    return h.hexdigest()


def configuration(args, env, deadline):
    # Bind the executable, explicitly supplied asset files/directories, client, prompt IDs and environment.
    files = {}
    for value in args.command:
        path = Path(value.removeprefix("auto:"))
        if path.is_file():
            files[str(path.resolve())] = file_hash(path, deadline)
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    files[str(child.resolve())] = file_hash(child, deadline)
    client = Path(__file__)
    for module in (client, client.with_name("batch_service_gpu_gate.py"), client.with_name("prefix_cache_gpu_gate.py")):
        files[str(module.resolve())] = file_hash(module, deadline)
    return dict(schema=1, command=args.command, files=files, client=file_hash(client, deadline),
                prompt=prompt_key(tokens(args.prompt)), generate=args.generate,
                decoder_prompt=prompt_key(tokens(args.decoder_prompt)) if args.decoder_prompt else None,
                decoder_generate=args.decoder_generate, cancel_at=args.cancel_at,
                cancel_fallback_delay=args.cancel_fallback_delay, scheduler_defaults=args.scheduler_defaults,
                expect_restored=args.expect_restored,
                environment={k: v for k, v in env.items() if k.startswith(("STRIX_", "HIP_", "HSA_", "ROCR_")) or
                             k in ("LD_LIBRARY_PATH", "PATH", "XDG_CACHE_HOME")})


def run_arm(args, name):
    env = os.environ.copy()
    env.update(STRIX_BATCH_CANCEL_POINTS="0", STRIX_BATCH_SERVICE_MS="0", STRIX_BATCH_NEST_ROWS="0",
               STRIX_FINAL_STATE_DIGEST="1", STRIX_PROMPT_STATE_DIGEST="1", STRIX_PREFIX_CACHE_DIGEST="1", STRIX_PREFIX_CACHE_DIGEST_COMPONENTS="1",
               STRIX_MTP_BATCH_LOG="a", STRIX_KEEP_CANCELLED_PREFILL="0")
    if not args.scheduler_defaults:
        env.update(STRIX_PROMPT_BATCH="0", STRIX_PROMPT_WINDOW_COHORT="0")
    log_path = args.out / (name + ".log")
    base = list(args.command)
    if "--prefix-cache-every" in base:
        at = base.index("--prefix-cache-every")
        del base[at:at + 2]
    command = base + ["--http-serve", "--http-batch", "2", "--http-bind", "127.0.0.1",
                              "--http-port", str(args.port)]
    prompt = tokens(args.prompt)
    started = time.monotonic()
    turn_end = started + args.budget
    work_end = turn_end - 20  # Graceful shutdown gets 15 s, kill/reap gets the remaining 5 s.
    binding = configuration(args, env, work_end)
    result = dict(arm=name, prompt_tokens=len(prompt))
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    def request_end():
        return min(work_end, time.monotonic() + args.request_timeout)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            deadline = min(work_end, time.monotonic() + args.startup_timeout)
            while True:
                require(process.poll() is None, f"{name}: server exited; inspect {log_path.name}")
                try:
                    with response_for(args.port, "GET", "/health", None, min(deadline, time.monotonic() + 1)):
                        break
                except OSError:
                    require(time.monotonic() < deadline, "server startup timed out")
                    time.sleep(0.25)
            result["startup_s"] = time.monotonic() - started
            decoder = None
            if name != "fresh" and args.decoder_prompt:
                first = threading.Event()
                decoder = pool.submit(completion, args.port, tokens(args.decoder_prompt), args.decoder_generate, request_end(), first)
                require(first.wait(min(args.request_timeout, remaining(work_end))), "decoder emitted no first token")
            if name != "fresh":
                result["cancel_sent_at"] = time.monotonic()
                seen = cancel_mid_prefill(args.port, prompt, args.cancel_at, args.generate, request_end(),
                                          args.cancel_fallback_delay)
                result["cancel_progress"] = seen
                result["cancel_closed_at"] = time.monotonic()
                if decoder:
                    d = decoder.result(timeout=min(args.request_timeout, remaining(work_end)))
                    gaps = d["emission_gaps_ms"]
                    if d["completed_at"] <= result["cancel_closed_at"] + 0.5:
                        print(f"note: {name}: decoder finished before the cancel was retired; raise --decoder-generate")
                    result["decoder"] = dict(tokens=len(d["strix"]["token_ids"]), gaps=len(gaps),
                                             gap_max_ms=max(gaps) if gaps else None,
                                             gap_p50_ms=statistics.median(gaps) if gaps else None,
                                             tok_s=len(d["strix"]["token_ids"]) / (d["completed_at"] - d["first_at"]))
                pool.shutdown(wait=True)
                deadline = request_end()
                while True:
                    queue = status(args.port, deadline).get("queue", {})
                    if queue.get("cancelled", 0) >= 1 and not queue.get("running", 0) and not queue.get("queued", 0):
                        break
                    require(time.monotonic() < deadline, "cancelled request was not retired")
                    time.sleep(0.05)
            sent = time.monotonic()
            retry = completion(args.port, prompt, args.generate, request_end())
            result["ttft_s"] = retry["first_at"] - sent
            result["total_s"] = retry["completed_at"] - sent
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=max(0, min(15, turn_end - time.monotonic() - 5)))
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=remaining(turn_end))
            pool.shutdown(wait=False, cancel_futures=True)
        remaining(turn_end)
        require(process.returncode == 0, f"{name}: unclean shutdown; inspect {log_path.name}")
    receipts = []
    prompt_receipts, final_receipts = [], []
    for line in log_path.read_text().splitlines():
        if line.startswith("tiled prompt: "):
            prompt_receipts.append(json.loads(line[len("tiled prompt: "):]))
        if line.startswith("request final: "):
            final_receipts.append(json.loads(line[len("request final: "):]))
        try:
            row = json.loads(line)
            if row.get("mode") == "native-mtp-batch":
                receipts.append(row)
        except (ValueError, AttributeError):
            pass
    require(receipts, "missing final batch receipt")
    require(prompt_receipts and final_receipts, "NO-REFERENCE: missing retry prompt/final diagnostics")
    result["signature"], result["cached_tokens"], result["retry_cache_refusal"] = exact_signature(
        retry, prompt, prompt_receipts, final_receipts)
    batch = receipts[-1]
    result["elapsed_s"] = time.monotonic() - started
    remaining(turn_end)
    record = dict(binding=binding, result=result, batch=batch, retry=retry, prompt_ids=prompt,
                  prompts=prompt_receipts, finals=final_receipts)
    path = args.out / (name + ".json")
    require(not path.exists(), "refusing to overwrite arm receipt")
    path.write_text(json.dumps(record, indent=2))
    print(f"{name}: ttft {result['ttft_s']:.3f} s cached {result['cached_tokens']}"
          f" refusal {result['retry_cache_refusal']} cancel progress {result.get('cancel_progress')}"
          f" decoder {result.get('decoder')}")
    return record


def compare(records):
    require(len(records) == 2 and {r["result"]["arm"] for r in records} == {"offdef", "fresh"},
            "need one offdef and one fresh arm")
    require(records[0]["binding"] == records[1]["binding"], "configuration-bound receipts differ")
    binding = records[0]["binding"]
    results = {}
    for record in records:
        result = record["result"]
        # Revalidate saved raw evidence, rather than trusting a serialized signature.
        key = tuple(binding["prompt"])
        ids = record["retry"]["strix"]["token_ids"]
        require(ids and 0 < result["elapsed_s"] < 420, "arm exceeded turn cap")
        prompt = record["prompt_ids"]
        require(prompt_key(prompt) == key, "receipt prompt identity differs")
        sig, cached, refusal = exact_signature(record["retry"], prompt, record["prompts"], record["finals"])
        require(sig == result["signature"] and cached == result["cached_tokens"] and
                refusal == result["retry_cache_refusal"], "contradictory saved witnesses")
        results[result["arm"]] = dict(signature=sig, cached_tokens=cached, retry_cache_refusal=refusal)
    retry, fresh = results["offdef"], results["fresh"]
    require(retry["signature"] == fresh["signature"], "MISMATCH: complete canonical retry state differs")
    require(fresh["cached_tokens"] == 0, "fresh reference must explicitly restore zero")
    expected = 0 if binding["scheduler_defaults"] else binding["expect_restored"]
    require(retry["cached_tokens"] == expected, "retry restore witness differs")
    if binding["scheduler_defaults"]:
        require(retry["retry_cache_refusal"] == "batch-suffix-not-solo", "missing batch-suffix-not-solo refusal")
    print("EXACT: complete IDs/logits/metadata/canonical state (including mtp.*); restore/refusal witnessed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-defaults", action="store_true",
                        help="retain grouping/cohort defaults; require a named zero-restore refusal")
    parser.add_argument("--expect-restored", type=int, default=4096)
    parser.add_argument("--compare", nargs=2, type=Path, help="offline comparison of two per-arm JSON receipts")
    parser.add_argument("--prompt", help="prompt token IDs (text or JSON), several T blocks long")
    parser.add_argument("--out", type=Path, help="fresh private output directory outside checkout")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--generate", type=int, default=16)
    parser.add_argument("--cancel-at", type=int, help="prefilled tokens before closing the connection (default 60%%)")
    parser.add_argument("--decoder-prompt", help="optional prompt IDs of a request that decodes through the cancel")
    parser.add_argument("--decoder-generate", type=int, default=256)
    parser.add_argument("--cancel-fallback-delay", type=float, default=0,
                        help="seconds after sending to cancel when the server sends no progress comments (0: require them)")
    parser.add_argument("--arm", choices=("offdef", "fresh"), help="one GPU turn; never qualifies alone")
    parser.add_argument("--budget", type=float, default=400, help="whole arm cap including cleanup; 20 < budget <= 400")
    parser.add_argument("--startup-timeout", type=float, default=400)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- server executable plus pinned model/MTP/cache "
                        "args (e.g. --known-prefix-prefill-rows 1024 --prefix-cache-host-bytes N); omit HTTP flags")
    args = parser.parse_args()
    if args.compare:
        compare([json.loads(p.read_text()) for p in args.compare])
        return
    if args.command[:1] == ["--"]:
        args.command.pop(0)
    require(args.command and args.arm and args.prompt and args.out, "--arm, --prompt, --out and server command required")
    require(20 < args.budget <= 400 and args.request_timeout > 0 and args.startup_timeout > 0,
            "invalid deadline: arm budget must be <=400 s, reserving cleanup inside 420 s")
    checkout = Path(__file__).resolve().parents[1]
    require(not args.out.resolve().is_relative_to(checkout), "receipts must stay outside Git checkout")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.cancel_at is None:
        args.cancel_at = len(tokens(args.prompt)) * 6 // 10
    require(not (args.out / (args.arm + ".json")).exists(), "arm receipt already exists")
    run_arm(args, args.arm)
    print("UNQUALIFIED: arm recorded; compare offdef/fresh receipts offline with --compare")


if __name__ == "__main__":
    main()
