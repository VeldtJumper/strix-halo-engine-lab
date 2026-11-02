#!/usr/bin/env python3
r"""Shared HTTP slots exactness gate: concurrent clients vs the engine's serial server route.

An engine adapter states how to start the server (flags and environment per slots mode), the
request contract (token-ID or text prompt, route), how to read the generated token IDs, the
request binding and the SHA-256 digests of committed state from the receipt, and which
server-side counter proves that two requests shared a target step. The reference is the same
command on its serial route (Qwen --http-batch 0, DeepSeek/MiniMax --slots 1, Nemotron
STRIX_G4_SERVE_BATCH=0), one request at a time. Every slots mode then gets one warm request
(counter baselines), then each client count in barrier-started rounds that cover every prompt.
Every request's token IDs and state digests must equal the reference for its prompt. A prompt
is covered only by a two-request round whose own counters prove the two requests shared a
step (counters cannot name the requests of a wider round); EXACT needs every prompt covered.

Coverage per engine (declared by the adapter, reported in summary.json):
  qwen      tokens+state  ids: strix.token_ids; state: final frontier + the full committed target
                          manifest (kv.L*, index.L*, recurrent/conv, ...; mtp.* excluded), 64-hex,
                          key set equal to the reference; witness: the shared-execution witness
                          when the build has it ('batch cycle ... shared_windows=W ...' with
                          STRIX_MTP_BATCH_LOG=1, plus strix.shared_execution per request): a
                          sessions=2 cycle with a shared verification window (joint/budget paths:
                          in a cycle that ran chunks), and with STRIX_MTP_DRAFT_BATCH=1 one with a
                          batched draft launch of two or more rows; every request's receipt must
                          show a shared window (and a shared draft step). Older builds: 'batch
                          cycle sessions=2 groups>=1' on the default 8-row cohort path;
                          budget/joint cohort paths -> UNQUALIFIED (no shared-chunk counter)
  nemotron  tokens+state  text prompt on /v1/chat/completions; ids: strix.generated_ids; state:
                          per-phase state_digest (64-hex) + integer frontier; identity fields
                          equal across arms; --streams must be pinned; witness: none exists
                          (max_batch counts live requests) -> UNQUALIFIED until the engine
                          reports shared forward passes per request
  deepseek  tokens-only   ids: strix.token_ids; witness: /api/status batch.decode_rows vs
                          batch.steps (no drafter) or DSpark decoder windows vs batch.steps
  minimax   tokens-only   text prompt (the API takes a string); ids: strix.token_ids; binding:
                          timings.prompt_fnv / ids_fnv (FNV-1a over u32 ids, /tokenize);
                          witness: /metrics mm_step_rows - mm_drafted > mm_steps
  glm       not-run       no HTTP server; batch equality is proven in process only

Every stream must end with one finish_reason (stop|length), usage (prompt_tokens equal to the
prompt / reference, completion_tokens == len(ids)) and data: [DONE].

Qwen without --generic delegates to tests/http_slots_gpu_gate.py with the same arguments
(backward compatible). That route does not validate stream completion and writes no kit
rows; summary.json says so (LEGACY-PASS/LEGACY-FAIL). Recommended: --generic.

Verdicts (exit code): EXACT 0, MISMATCH 1, NO-WITNESS 2 (two-request rounds ran, counters
readable, no shared step), NOT-RUN 3 (no HTTP server), NO-REFERENCE 4 (receipt, stream or
schema evidence missing/malformed, setup refused), UNQUALIFIED 5 (no two-request round, no
proving counter, a prompt not covered, or --allow-no-witness), ERROR 6 (unexpected failure,
incl. unclean shutdown of either rank). summary.json is removed at start and always rewritten.

Examples (output directories outside the checkout; commands without HTTP/slots/port flags):
  python3 tests/serve_slots_gate.py --engine qwen --generic --prompt A.ids --prompt B.ids \
    --out /tmp/slots-qwen -- build/strix-hip-decode-token MODEL.gguf PINNED_ARGS
  python3 tests/serve_slots_gate.py --engine nemotron --prompt chat.txt --prompt code.txt \
    --out /tmp/slots-e4b -- build/strix-gemma4-serve MODEL --rope-tables TABLES \
    --tokenizer MODEL/tokenizer.json --profile default --streams 8
  # pair engines: rank 0 here, rank 1 started over ssh with the same serve flags per mode
  python3 tests/serve_slots_gate.py --engine deepseek --prompt A.ids --prompt B.ids \
    --out /tmp/slots-ds --peer PEER --peer-dir /path/on/peer -- \
    env STRIX_DS41_PACKS=PACKS build/strix-deepseek41-serve MODEL.gguf --rank 0 --transport usb4auto
"""
import argparse
import concurrent.futures
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LEGACY_QWEN_GATE = ROOT / "tests" / "http_slots_gpu_gate.py"
EXIT = {"EXACT": 0, "MISMATCH": 1, "NO-WITNESS": 2, "NOT-RUN": 3, "NO-REFERENCE": 4, "UNQUALIFIED": 5, "ERROR": 6}
HEX64 = re.compile(r"[0-9a-f]{64}")


class GateError(RuntimeError):
    def __init__(self, verdict, message, prompt=None):
        super().__init__(verdict + ": " + message)
        self.verdict = verdict
        self.prompt = prompt


def require(value, verdict, message, prompt=None):
    if not value:
        raise GateError(verdict, message, prompt)


def is_int(value):
    return type(value) is int


def id_hash(ids):
    # Same canonical form as the bench kit: ASCII JSON integer array without spaces.
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest()


def state_hash(state):
    return None if state is None else hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def fnv64(ids):
    # FNV-1a 64 over little-endian u32 IDs (src/minimax/serve/minimax_lead.hpp fnv()).
    h = 14695981039346656037
    for b in struct.pack("<%dI" % len(ids), *ids):
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return "%016x" % h


def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_key(value, key)
            if found is not None:
                return found
    return None


def http_get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
        return r.read().decode("utf-8", "replace")


def http_post(port, path, obj):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(obj).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as r:
        return json.loads(r.read())


def has_flag(command, flag):
    return any(a.split("=", 1)[0] == flag for a in command)


# ---------------------------------------------------------------- adapters

class Adapter:
    name = ""
    http = True
    no_http_reason = ""
    pair = False
    prompt_kind = "ids"          # ids: token ID file; text: UTF-8 text file (one prompt per file)
    path = "/v1/completions"
    coverage = "tokens-only"     # or tokens+state
    reference_mode = "1"
    default_modes = "auto,2,4,8"
    default_generate = 128
    max_slots = 8
    ready_marker = None          # text the server log must contain before /health counts
    stop_marker = None           # text the log must contain after a clean stop
    exit_ok = (0,)               # accepted exit codes after SIGINT (negative = killed by that signal)
    rank1_exit_ok = (0,)
    forbidden = ()
    env_base = {}

    def serve_args(self, port, mode):
        raise NotImplementedError

    def env(self, mode):
        return dict(self.env_base)

    def check_command(self, command):
        flags = [a for a in command if a.split("=", 1)[0] in self.forbidden]
        require(not flags, "NO-REFERENCE", "command must omit the flags the gate sets: " + " ".join(flags))

    def body(self, prompt, count):
        return dict(prompt=prompt, max_tokens=count, temperature=0, stream=True, return_token_ids=True,
                    stream_options=dict(include_usage=True))

    def token_ids(self, strix):
        return strix.get("token_ids") if isinstance(strix, dict) else None

    def state(self, strix):
        return None

    def identity(self, strix):
        return None

    def prepare_prompt(self, port, prompt):
        """Per-prompt facts the binding checks need (called once on the reference server)."""
        return dict(prompt_tokens=len(prompt) if isinstance(prompt, list) else None)

    def bind(self, response, facts):
        """Engine-specific request binding beyond usage; raises GateError."""

    def extract(self, response, facts=None, count=None):
        strix = response["strix"]
        ids = self.token_ids(strix)
        require(isinstance(ids, list) and ids and all(is_int(x) and x >= 0 for x in ids), "NO-REFERENCE",
                self.name + ": receipt has no generated token IDs")
        usage = response["usage"]
        require(isinstance(usage, dict) and is_int(usage.get("completion_tokens")) and is_int(usage.get("prompt_tokens")),
                "NO-REFERENCE", self.name + ": usage lacks integer prompt/completion token counts")
        require(usage["completion_tokens"] == len(ids), "NO-REFERENCE",
                f"{self.name}: usage.completion_tokens {usage['completion_tokens']} != {len(ids)} receipt IDs")
        finish = response["finish_reason"]
        require(finish in ("stop", "length"), "NO-REFERENCE", f"{self.name}: finish_reason {finish!r}")
        if count is not None:
            require(len(ids) <= count and (finish != "length" or len(ids) == count), "NO-REFERENCE",
                    f"{self.name}: {len(ids)} IDs inconsistent with finish {finish} and max_tokens {count}")
        if facts is not None:
            if facts.get("prompt_tokens") is not None:
                require(usage["prompt_tokens"] == facts["prompt_tokens"], "MISMATCH",
                        f"{self.name}: usage.prompt_tokens {usage['prompt_tokens']} != prompt ({facts['prompt_tokens']}): "
                        "the receipt belongs to another request")
            self.bind(response, facts)
        state = self.state(strix) if self.coverage == "tokens+state" else None
        return dict(token_ids=ids, state=state, prompt_tokens=usage["prompt_tokens"], identity=self.identity(strix))

    def mode_slots(self, mode):
        return None if mode == "auto" else int(mode)

    # Server-side evidence that the two requests of a round shared a target step. probe() snapshots
    # counters before/after one round; witness() returns True/False, or None when the engine exposes no
    # counter that proves it (the verdict is then UNQUALIFIED). ctx: command, env of the server.
    witness_gap = None           # why witness() cannot prove shared execution (reported)
    last_detail = None           # witness() may leave the counts it judged here (reported per round)
    def probe(self, port):
        return None

    def witness(self, before, after, responses, log_delta, ctx):
        return None


class Qwen(Adapter):
    name = "qwen"
    coverage = "tokens+state"
    reference_mode = "0"
    default_modes = "auto,1,2,4,8,16"
    default_generate = 256
    max_slots = None
    # Same observation environment as http_slots_gpu_gate.server().
    env_base = dict(STRIX_PROMPT_STATE_DIGEST="1", STRIX_PREFIX_CACHE_DIGEST="1",
                    STRIX_PREFIX_CACHE_DIGEST_COMPONENTS="1", STRIX_FINAL_STATE_DIGEST="0",
                    STRIX_MTP_BATCH_LOG="1", STRIX_HTTP_EAGER_SLOTS="4")
    forbidden = ("--http-serve", "--http-port", "--http-batch")
    witness_gap = ("budget/joint cohort paths count groups whose chunks may run alone; this build predates the "
                   "shared-execution witness ('batch cycle ... shared_windows=' and strix.shared_execution)")

    def serve_args(self, port, mode):
        return ["--http-serve", "--http-port", str(port)] + ([] if mode == "auto" else ["--http-batch", mode])

    def env(self, mode):
        env = dict(self.env_base)
        env.update({k: None for k in ("STRIX_RTX_EXPORT_STATE", "STRIX_RTX_EXPORT_POSITION")})
        return env

    def state(self, strix):
        position = strix.get("final_state_position")
        require(is_int(position) and position > 0, "NO-REFERENCE", "qwen: final_state_position must be a positive integer")
        components = strix.get("final_state_components")
        require(isinstance(components, dict), "NO-REFERENCE", "qwen: final_state_components missing")
        target = {k: v for k, v in components.items() if not k.startswith("mtp.")}
        require({"recurrent_matrix", "gdn_convolution"} <= target.keys(), "NO-REFERENCE",
                "qwen: missing recurrent/conv committed components")
        bad = sorted(k for k, v in components.items() if not (isinstance(v, str) and HEX64.fullmatch(v)))
        require(not bad, "NO-REFERENCE", "qwen: component digests not 64-hex: " + ",".join(bad))
        # Registry (apps/hip_prefix_state_cache.hpp): per-layer kv.L*/index.L* rows plus fixed components.
        for prefix in ("kv.L", "index.L"):
            require(any(k.startswith(prefix) for k in target), "NO-REFERENCE",
                    f"qwen: component manifest lacks {prefix}* rows")
        return dict(final_state_position=position, components=target)

    def manifest(self, state):
        return sorted(state["components"]) if state else None

    def mode_slots(self, mode):
        return None if mode == "auto" else max(1, int(mode))

    def witness(self, before, after, responses, log_delta, ctx):
        # Shared execution only; the full committed-logit comparison is the TODO(full-logit gate) in run_gate.check.
        env = ctx.get("env", {})
        arms = qwen_arms(env, log_delta)
        cycles = parse_batch_cycles(log_delta)
        witnessed = [c for c in cycles if "shared_windows" in c]
        receipts = [find_key(r.get("strix"), "shared_execution") for r in responses]
        receipts = receipts if receipts and all(isinstance(x, dict) for x in receipts) else None
        if witnessed or receipts:
            self.last_detail = shared_execution_detail(arms, witnessed, receipts)
            return self.last_detail["shared"]
        # Builds without the shared-execution witness. 'batch cycle sessions=N groups=G ...': on the default 8-row
        # cohort path every counted group verifies >= 2 sessions on one shared primary (verify_cohort), so G >= 1
        # with N == 2 is shared target execution of both requests. The budget path (STRIX_COHORT_ROWS=16|32) and
        # the joint path (STRIX_MTP_BATCH_JOINT=1, 'joint cycle' lines) count groups whose chunks may all run
        # alone: unprovable. (The draft batch is not judged without the witness.)
        if arms["joint"] or arms["budget"]:
            return None
        return any(c.get("sessions") == 2 and c.get("groups", 0) >= 1 for c in cycles)


QWEN_DECLINE = re.compile(r"([a-z_]+):(\d+)")


def qwen_arms(env, log_delta=""):
    """The Qwen batched-serving arms a server runs: joint verification, a 16/32-row budget, the draft batch."""
    return dict(joint=str(env.get("STRIX_MTP_BATCH_JOINT") or "").startswith("1") or "joint cycle sessions=" in log_delta,
                budget=str(env.get("STRIX_COHORT_ROWS") or "").strip() in ("16", "32"),
                draft_batch=str(env.get("STRIX_MTP_DRAFT_BATCH") or "").startswith("1"))


def parse_batch_cycles(log):
    """'batch cycle ...' lines (src/hip_mtp_greedy_runtime.hip; the witness suffix is
    include/strix/mtp_shared_execution_witness.hpp mtp_shared_execution_log) -> one dict per cycle: integers, and
    draft_declines / verify_declines as {reason: count} ('none' -> {})."""
    out = []
    for line in re.findall(r"^.*?batch cycle (sessions=.*)$", log, re.M):
        cycle = {}
        for key, value in re.findall(r"(\w+)=(\S+)", line):
            if key in ("draft_declines", "verify_declines"):
                cycle[key] = {k: int(n) for k, n in QWEN_DECLINE.findall(value)}
            elif re.fullmatch(r"\d+", value):
                cycle[key] = int(value)
        out.append(cycle)
    return out


def shared_execution_detail(arms, cycles, receipts):
    """Judges one two-request round from its witnessed 'batch cycle' lines and the requests' strix.shared_execution.

    Shared verification: a sessions=2 cycle with a shared window of two or more sessions (shared_windows >= 1); on
    the joint/budget paths in a cycle that ran budget chunks (chunks >= 1: not the ordinary-cohort fallback), and
    every receipt with verify_shared_cycles >= 1. Draft batch (STRIX_MTP_DRAFT_BATCH=1): a sessions=2 cycle with a
    batched draft launch of two or more rows (draft_shared_launches >= 1), and every receipt with
    draft_shared_steps >= 1. Grouped-but-separate execution (chunks alone, declined drafts) is not shared."""
    pair = [c for c in cycles if c.get("sessions") == 2]
    chunked = arms["joint"] or arms["budget"]

    def total(key):
        return sum(c.get(key, 0) for c in cycles)

    def declines(key):
        merged = {}
        for c in cycles:
            for reason, n in c.get(key, {}).items():
                merged[reason] = merged.get(reason, 0) + n
        for r in receipts or ():
            for reason, n in (r.get(key) or {}).items():
                merged["request." + reason] = merged.get("request." + reason, 0) + n
        return merged

    verify_log = any(c.get("shared_windows", 0) >= 1 and (c.get("chunks", 0) >= 1 or not chunked) for c in pair) if cycles else None
    draft_log = any(c.get("draft_shared_launches", 0) >= 1 for c in pair) if cycles else None
    verify_receipts = all(int(r.get("verify_shared_cycles") or 0) >= 1 for r in receipts) if receipts else None
    draft_receipts = all(int(r.get("draft_shared_steps") or 0) >= 1 for r in receipts) if receipts else None

    def judge(log, receipt, log_required):
        if log is None and (log_required or receipt is None):
            return None  # the cycle lines are missing (STRIX_MTP_BATCH_LOG off): nothing proves the path
        return all(x for x in (log, receipt) if x is not None)

    verify = judge(verify_log, verify_receipts, chunked)
    draft = judge(draft_log, draft_receipts, False) if arms["draft_batch"] else True
    shared = None if verify is None or draft is None else verify and draft
    return dict(arms=sorted(k for k, on in arms.items() if on), shared=shared, verify_shared=verify,
                draft_shared=draft if arms["draft_batch"] else None,
                cycles=len(cycles), pair_cycles=len(pair), chunks=total("chunks"),
                shared_windows=total("shared_windows"), shared_sessions=total("shared_sessions"),
                shared_rows=total("shared_rows"), max_members=max([c.get("max_members", 0) for c in cycles] or [0]),
                draft_batches=total("draft_batches"), draft_batch_sessions=total("draft_batch_sessions"),
                draft_launches=total("draft_launches"), draft_shared_launches=total("draft_shared_launches"),
                draft_max_rows=max([c.get("draft_max_rows", 0) for c in cycles] or [0]),
                joint_fallbacks=total("joint_fallbacks"),
                draft_declines=declines("draft_declines"), verify_declines=declines("verify_declines"),
                receipts=[{k: r.get(k) for k in ("cohort_cycles", "verify_shared_cycles", "verify_max_members",
                                                 "draft_steps", "draft_shared_steps", "draft_max_rows")}
                          for r in receipts] if receipts else None)


class DeepSeek(Adapter):
    name = "deepseek"
    pair = True
    ready_marker = '"READY"'
    stop_marker = '"STOPPED"'    # rank 0 drains, stops rank 1, then reports the clean stop
    max_slots = 16  # above 8 the server itself requires STRIX_DS41_BATCH_ROWS16=1
    forbidden = ("--http-port", "--http-bind", "--slots")

    def serve_args(self, port, mode):
        return ["--http-bind", "127.0.0.1", "--http-port", str(port)] + ([] if mode == "auto" else ["--slots", mode])

    def probe(self, port):
        try:
            status = json.loads(http_get(port, "/api/status?detail=1"))
        except (OSError, ValueError):
            return None
        batch = find_key(status, "batch")
        if not (isinstance(batch, dict) and is_int(batch.get("steps")) and is_int(batch.get("decode_rows"))):
            return None
        snap = dict(steps=batch["steps"], decode_rows=batch["decode_rows"])
        dspark = find_key(status, "dspark")
        if isinstance(dspark, dict) and is_int(dspark.get("decode_steps")) and isinstance(dspark.get("window_rows"), list):
            snap.update(dspark_steps=dspark["decode_steps"], windows=sum(int(n) for n in dspark["window_rows"]))
        return snap

    def witness(self, before, after, responses, log_delta, ctx):
        if before is None or after is None:
            return None
        command = ctx.get("command", [])
        toggles = [a.split("=", 1)[0] for a in command if a.split("=", 1)[0] in ("--dspark", "--no-dspark")]
        drafter = bool(toggles) and toggles[-1] == "--dspark"
        steps = after["steps"] - before["steps"]
        if "windows" in after and "windows" in before:
            # window_rows counts one entry per decoding request in EVERY speculator step, mixed-prefill steps
            # included (hip_deepseek41_batch_dspark.hip); batch.steps counts every forward step of the server, a
            # superset of those steps. More decoder windows than steps = a step with two decoding requests.
            return after["windows"] - before["windows"] > steps
        if drafter or "windows" in after or "windows" in before:
            return None  # draft rows inflate decode_rows; without both window snapshots nothing proves two requests
        return after["decode_rows"] - before["decode_rows"] > steps


class MiniMax(Adapter):
    name = "minimax"
    pair = True
    prompt_kind = "text"         # src/minimax/serve/minimax_api.cpp: "prompt (string) required"
    ready_marker = "READY rank 0"
    forbidden = ("--port", "--slots", "--bind")

    def serve_args(self, port, mode):
        return ["--bind", "127.0.0.1", "--port", str(port)] + ([] if mode == "auto" else ["--slots", mode])

    def prepare_prompt(self, port, prompt):
        tokens = http_post(port, "/tokenize", dict(text=prompt)).get("tokens")
        require(isinstance(tokens, list) and tokens and all(is_int(t) for t in tokens), "NO-REFERENCE",
                "minimax: /tokenize returned no tokens")
        return dict(prompt_tokens=len(tokens), prompt_fnv=fnv64(tokens))

    def bind(self, response, facts):
        timings = response.get("timings")
        require(isinstance(timings, dict), "NO-REFERENCE", "minimax: final chunk lacks timings")
        require(timings.get("prompt_fnv") == facts["prompt_fnv"], "MISMATCH",
                "minimax: timings.prompt_fnv differs from the prompt's IDs: the receipt belongs to another request")
        require(timings.get("ids_fnv") == fnv64(self.token_ids(response["strix"])), "MISMATCH",
                "minimax: timings.ids_fnv differs from the receipt token IDs")

    def probe(self, port):
        try:
            text = http_get(port, "/metrics")
        except OSError:
            return None
        values = dict(re.findall(r"^(mm_steps|mm_step_rows|mm_drafted) (\d+)$", text, re.M))
        return {k: int(v) for k, v in values.items()} if len(values) == 3 else None

    def witness(self, before, after, responses, log_delta, ctx):
        if before is None or after is None:
            return None
        d = {k: after[k] - before[k] for k in after}
        # a cycle has one row per running request plus its draft rows (minimax_lead.hpp)
        return d["mm_step_rows"] - d["mm_drafted"] > d["mm_steps"]


class Nemotron(Adapter):
    name = "nemotron"
    prompt_kind = "text"
    path = "/v1/chat/completions"
    coverage = "tokens+state"
    reference_mode = "serial"
    ready_marker = "service ready"
    exit_ok = (0, -int(signal.SIGINT), 130)  # no SIGINT handler: default termination by SIGINT is the clean stop
    forbidden = ("--port", "--bind")
    identity_keys = ("decode_identity", "prefill_identity", "sampling_identity", "row_cap")

    def check_command(self, command):
        Adapter.check_command(self, command)
        # --streams defaults to the serve slot count (apps/gemma4_serve_main.cpp), which differs between the
        # serial reference and the batched arms and can select another EXL3 numerical identity.
        require(has_flag(command, "--streams"), "NO-REFERENCE", "nemotron: pin --streams N in the command")

    def serve_args(self, port, mode):
        return ["--bind", "127.0.0.1", "--port", str(port)]

    def env(self, mode):
        if mode == self.reference_mode:
            return dict(STRIX_G4_SERVE_BATCH="0", STRIX_G4_SERVE_SLOTS=None)
        return dict(STRIX_G4_SERVE_BATCH="1", STRIX_G4_SERVE_SLOTS=None if mode == "auto" else mode)

    def mode_slots(self, mode):
        return 1 if mode == self.reference_mode else Adapter.mode_slots(self, mode)

    def body(self, prompt, count):
        return dict(messages=[dict(role="user", content=prompt)], max_tokens=count, temperature=0, stream=True,
                    stream_options=dict(include_usage=True))

    def token_ids(self, strix):
        return strix.get("generated_ids") if isinstance(strix, dict) else None

    def phases(self, strix):
        phases = strix.get("phases")
        require(isinstance(phases, list) and phases and all(isinstance(p, dict) for p in phases), "NO-REFERENCE",
                "nemotron: receipt has no phases")
        return phases

    def state(self, strix):
        out = []
        for p in self.phases(strix):
            require(isinstance(p.get("state_digest"), str) and HEX64.fullmatch(p["state_digest"]), "NO-REFERENCE",
                    "nemotron: phase state_digest is not 64-hex")
            require(is_int(p.get("frontier")) and p["frontier"] >= 0, "NO-REFERENCE",
                    "nemotron: phase frontier must be a non-negative integer")
            out.append(dict(state_digest=p["state_digest"], frontier=p["frontier"]))
        return out

    def identity(self, strix):
        ident = []
        for p in self.phases(strix):
            missing = [k for k in self.identity_keys if p.get(k) is None]
            require(not missing, "NO-REFERENCE", "nemotron: receipt identity lacks " + ",".join(missing))
            ident.append({k: p[k] for k in self.identity_keys})
        return ident

    witness_gap = ("serve_batch.max_batch and the receipt 'streams' record live requests (api.cpp live.size(), "
                   "coordinator peak_streams), also for streams that prefill a chunk alone; a per-request count of "
                   "forward_rows passes shared with another request's rows is needed")

    def witness(self, before, after, responses, log_delta, ctx):
        return None


class Glm(Adapter):
    name = "glm"
    http = False
    pair = True
    no_http_reason = ("GLM pair has no HTTP server (strix-glm5-rank is batch-only); batch equality is proven "
                      "in process (tools/glm/gate_d_digest.py), not over HTTP")

    def serve_args(self, port, mode):
        raise GateError("NOT-RUN", self.no_http_reason)


ADAPTERS = {a.name: a for a in (Qwen(), DeepSeek(), MiniMax(), Nemotron(), Glm())}
ALIASES = {"deepseek-pair": "deepseek", "minimax-pair": "minimax", "glm-pair": "glm", "nemotron-e4b": "nemotron"}


def adapter_for(name):
    name = ALIASES.get(name, name)
    if name not in ADAPTERS:
        raise ValueError("unknown engine: " + name)
    return ADAPTERS[name]


# ---------------------------------------------------------------- prompts and requests

def load_prompt(adapter, path):
    text = Path(path).read_text(encoding="utf-8")
    if adapter.prompt_kind == "text":
        require(text.strip(), "NO-REFERENCE", "empty prompt file: " + str(path))
        return text
    text = text.strip()
    try:
        ids = json.loads(text) if text.startswith("[") else [int(x) for x in text.replace(",", " ").split()]
    except ValueError:
        ids = None
    require(isinstance(ids, list) and ids and all(is_int(x) and x >= 0 for x in ids), "NO-REFERENCE",
            "bad token ID file: " + str(path))
    return ids


def prompt_hash(prompt):
    if isinstance(prompt, str):
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return id_hash(prompt)


def completion(adapter, port, prompt, count, timeout=1800):
    """One streamed greedy request; validates the stream's terminal structure and returns its receipt parts."""
    body = json.dumps(adapter.body(prompt, count)).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}{adapter.path}", data=body,
                                     headers={"Content-Type": "application/json"})
    finishes, usages, receipts, timings = [], [], [], []
    done = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        headers = {k.lower(): v for k, v in response.headers.items()}
        for line in response:
            if not line.startswith(b"data:"):
                continue
            require(not done, "NO-REFERENCE", "data after [DONE]")
            if line.strip() == b"data: [DONE]":
                done = True
                continue
            chunk = json.loads(line[5:])
            require("error" not in chunk, "MISMATCH", "server error: " + json.dumps(chunk.get("error")))
            for choice in chunk.get("choices") or []:
                if choice.get("finish_reason") is not None:
                    finishes.append(choice["finish_reason"])
            if chunk.get("usage") is not None:
                usages.append(chunk["usage"])
            if chunk.get("timings") is not None:
                timings.append(chunk["timings"])
            if isinstance(chunk.get("strix"), dict) and adapter.token_ids(chunk["strix"]) is not None:
                receipts.append(chunk["strix"])
    require(done, "NO-REFERENCE", "stream ended without data: [DONE]")
    require(len(finishes) == 1, "NO-REFERENCE", f"expected one finish_reason, got {len(finishes)}")
    require(usages and all(u == usages[0] for u in usages), "NO-REFERENCE", "missing or inconsistent usage")
    require(receipts, "NO-REFERENCE", "missing final receipt with token IDs")
    ids = [adapter.token_ids(r) for r in receipts]
    require(all(i == ids[0] for i in ids), "NO-REFERENCE", "receipt chunks disagree on token IDs")
    return dict(strix=receipts[-1], usage=usages[0], finish_reason=finishes[0], headers=headers,
                timings=timings[-1] if timings else None)


def port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


# ---------------------------------------------------------------- servers

def merged_env(base, overrides):
    env = dict(base)
    for k, v in overrides.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    return env


def peer_command(args, adapter, mode):
    """Rank-1 argv: --peer-command if given, else the local command with --rank 0 replaced by --rank 1."""
    if args.peer_command:
        base = shlex.split(args.peer_command)
    else:
        base = list(args.command)
        positions = [i for i, a in enumerate(base[:-1]) if a == "--rank" and base[i + 1] == "0"]
        require(len(positions) == 1, "NO-REFERENCE", "pair command needs exactly one '--rank 0' (or --peer-command)")
        base[positions[0] + 1] = "1"
    env = adapter.env(mode)
    prefix = ["env"] + ["-u" + k for k, v in env.items() if v is None] + [f"{k}={v}" for k, v in env.items() if v is not None]
    return prefix + base + adapter.serve_args(args.port, mode)


def remote_start_script(peer_dir, name, argv):
    # The child's pid goes to NAME.pid, its exit status to NAME.rc once it ends (the wrapper waits for it).
    launcher = '"$@" & c=$!; echo "$c" > "$1.pid"; wait "$c"; echo "$?" > "$1.rc"'
    launcher = 'n="$1"; shift; ' + launcher.replace('"$1.', '"$n.')
    return (f"cd {shlex.quote(peer_dir)} && rm -f {shlex.quote(name + '.pid')} {shlex.quote(name + '.rc')} && "
            f"{{ setsid bash -c {shlex.quote(launcher)} rank1 {shlex.quote(name)} "
            + " ".join(shlex.quote(a) for a in argv) + f" > {shlex.quote(name + '.rank1.log')} 2>&1 < /dev/null & }}; "
            f"for i in $(seq 50); do [ -s {shlex.quote(name + '.pid')} ] && echo STARTED && exit 0; sleep 0.2; done; "
            "echo NO-PID; exit 1")


def remote_stop_script(peer_dir, name, grace=120):
    base = shlex.quote(peer_dir.rstrip("/") + "/" + name)
    # Only the process this gate started (its own pid file). Rank 0's clean stop normally ends rank 1:
    # wait for its exit status first, then INT, then KILL (reported, never accepted).
    return (f"b={base}; p=$(cat \"$b.pid\" 2>/dev/null) || {{ echo NO-PID; exit 0; }}; "
            f"for i in $(seq {grace}); do [ -s \"$b.rc\" ] && echo RC=$(cat \"$b.rc\") && exit 0; sleep 1; done; "
            "kill -INT \"$p\" 2>/dev/null; "
            f"for i in $(seq 60); do [ -s \"$b.rc\" ] && echo INT-RC=$(cat \"$b.rc\") && exit 0; sleep 1; done; "
            "kill -KILL \"$p\" 2>/dev/null; sleep 1; echo KILLED")


def ssh(args, script):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "--", args.peer,
                           "bash -c " + shlex.quote(script)], capture_output=True, text=True, timeout=600)


def stop_local(process):
    if process.poll() is not None:
        return False
    if os.name == "nt":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=180)
        return False
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        return True


@contextmanager
def server(args, adapter, name, mode, record):
    require(port_free(args.port), "NO-REFERENCE", f"port {args.port} already serves; refusing a stale server")
    env = merged_env(os.environ, adapter.env(mode))
    command = args.command + adapter.serve_args(args.port, mode)
    log_path = args.out / (name + ".log")
    entry = record[name] = dict(command_tail=adapter.serve_args(args.port, mode), env=adapter.env(mode))
    peer_attempted = False
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    failure = None
    with log_path.open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        try:
            if adapter.pair and args.peer:
                peer_attempted = True  # from here on the stop path runs, even if ssh fails or times out
                done = ssh(args, remote_start_script(args.peer_dir, name, peer_command(args, adapter, mode)))
                entry["rank1_launch"] = dict(returncode=done.returncode, stdout=done.stdout.strip()[-200:])
                require(done.returncode == 0 and "STARTED" in done.stdout, "NO-REFERENCE",
                        "rank 1 launch failed: " + (done.stderr or done.stdout).strip()[-300:])
            deadline = time.monotonic() + args.startup_timeout
            while True:
                require(process.poll() is None, "NO-REFERENCE", "server exited: " + name)
                ready = adapter.ready_marker is None or adapter.ready_marker in log_path.read_text(errors="replace")
                if ready:
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=1):
                            break
                    except OSError:
                        pass
                require(time.monotonic() < deadline, "NO-REFERENCE", "startup timeout: " + name)
                time.sleep(0.25)
            yield log_path
        finally:
            killed = stop_local(process)
            entry.update(exit_code=process.returncode, killed=killed)
            if killed or process.returncode not in adapter.exit_ok:
                failure = f"rank 0 unclean shutdown: {name} exit {process.returncode}{' (killed)' if killed else ''}"
            elif adapter.stop_marker and adapter.stop_marker not in log_path.read_text(errors="replace"):
                failure = f"rank 0 stop marker {adapter.stop_marker} missing: {name}"
            if peer_attempted:
                try:
                    stopped = ssh(args, remote_stop_script(args.peer_dir, name))
                    out = stopped.stdout.strip()
                except (OSError, subprocess.SubprocessError) as error:
                    stopped, out = None, "ssh failed: " + str(error)
                entry["rank1_stop"] = out[-200:]
                entry["rank1_log"] = args.peer_dir.rstrip("/") + "/" + name + ".rank1.log"
                rc = re.search(r"^(?:INT-)?RC=(-?\d+)$", out, re.M)
                if stopped is None or stopped.returncode != 0 or rc is None or out.startswith("INT-"):
                    failure = failure or f"rank 1 did not stop cleanly: {name}: {out[-120:]}"
                elif int(rc.group(1)) not in adapter.rank1_exit_ok:
                    failure = failure or f"rank 1 exit {rc.group(1)}: {name}"
                entry["rank1_exit_code"] = int(rc.group(1)) if rc else None
            deadline = time.monotonic() + 60
            while not port_free(args.port) and time.monotonic() < deadline:
                time.sleep(0.5)
            if not port_free(args.port):
                failure = failure or f"port {args.port} still serving after {name} stopped"
            entry["clean"] = failure is None
    if failure:
        raise GateError("ERROR", failure)


# ---------------------------------------------------------------- gate

def run_gate(args, adapter, prompts, record):
    """Fills `record` (the summary) as it goes, so a failing run keeps its cells and server records."""
    servers = record["servers"] = {}
    with server(args, adapter, "reference", adapter.reference_mode, servers):
        facts = [adapter.prepare_prompt(args.port, p) for p in prompts]
        reference = []
        for index, prompt in enumerate(prompts):
            runs = [adapter.extract(completion(adapter, args.port, prompt, args.generate), facts[index], args.generate)
                    for _ in range(args.reference_repeat)]
            require(all(r == runs[0] for r in runs), "MISMATCH", "serial reference is not repeat-stable", index)
            reference.append(runs[0])
    for index, ref in enumerate(reference):
        if facts[index].get("prompt_tokens") is None:
            facts[index]["prompt_tokens"] = ref["prompt_tokens"]
    record["prompt_facts"] = facts
    record["reference_token_sha256"] = [id_hash(r["token_ids"]) for r in reference]
    (args.out / "reference.json").write_text(json.dumps(reference, indent=2))
    clients_list = [int(c) for c in args.clients.split(",")]
    rounds_log, covered = [], set()
    record["rounds"] = rounds_log
    pair_rounds = unreadable = witnessed = 0

    def check(where, index, result):
        got = adapter.extract(result, facts[index], args.generate)
        want = reference[index]
        require(got["identity"] == want["identity"], "NO-REFERENCE",
                where + ": engine identity differs from the reference arm", index)
        if want["state"] is not None and hasattr(adapter, "manifest"):
            require(adapter.manifest(got["state"]) == adapter.manifest(want["state"]), "NO-REFERENCE",
                    where + ": committed component manifest differs from the reference", index)
        require(got["token_ids"] == want["token_ids"], "MISMATCH",
                where + ": token IDs differ from the serial reference", index)
        require(got["state"] == want["state"], "MISMATCH", where + ": committed state digests differ", index)
        # TODO(full-logit gate): promoting the batched Qwen serving defaults (draft batch + joint verification)
        # also needs every request's full committed FP32 logits equal to its C1 reference. That comparison is the
        # qualification kit's extension of this gate and plugs in here, per request, next to the IDs and state.

    for mode in args.modes.split(","):
        name = "slots-" + mode
        with server(args, adapter, name, mode, servers) as log_path:
            ctx = dict(command=args.command, env=merged_env(os.environ, adapter.env(mode)))
            # Arm observation, then one warm request alone so every counter (DSpark windows are published only
            # after an armed step) has a baseline before the first measured round.
            adapter.probe(args.port)
            check(name + " warm-up", 0, completion(adapter, args.port, prompts[0], args.generate))
            for clients in clients_list:
                # Barrier-started rounds of `clients` requests until every prompt has run in this cell.
                rounds = -(-len(prompts) // clients)
                assigned = [(r * clients + i) % len(prompts) for r in range(rounds) for i in range(clients)]
                cell = []
                for r in range(rounds):
                    batch = assigned[r * clients:(r + 1) * clients]
                    barrier = threading.Barrier(clients)

                    def request(index):
                        barrier.wait()
                        return completion(adapter, args.port, prompts[index], args.generate)

                    log_mark = log_path.stat().st_size
                    before = adapter.probe(args.port)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as pool:
                        results = list(pool.map(request, batch))
                    after = adapter.probe(args.port)
                    with log_path.open("rb") as f:
                        f.seek(log_mark)
                        log_delta = f.read().decode("utf-8", "replace")
                    cell += results
                    for slot, (index, result) in enumerate(zip(batch, results)):
                        check(f"{name} clients {clients} round {r + 1} request {slot} (prompt {index + 1})", index, result)
                    slots = adapter.mode_slots(mode)
                    header = results[0]["headers"].get("x-strix-http-slots")
                    if header is not None and header.isdigit():
                        slots = int(header)
                    # Counters cannot say WHICH requests shared a step; with exactly two requests in the round a
                    # shared step involves both, so only two-request rounds can cover their prompts.
                    pair = len(batch) == 2 and slots != 1
                    adapter.last_detail = None
                    witness = adapter.witness(before, after, results, log_delta, ctx) if pair else None
                    detail = adapter.last_detail if pair else None
                    if detail:
                        print(f"WITNESS {adapter.name} {name} clients {clients} round {r + 1}: shared={detail['shared']} "
                              f"arms={','.join(detail['arms']) or 'cohort'} shared_windows={detail['shared_windows']} "
                              f"max_members={detail['max_members']} chunks={detail['chunks']} "
                              f"draft_shared_launches={detail['draft_shared_launches']} "
                              f"draft_max_rows={detail['draft_max_rows']}", flush=True)
                    if pair:
                        pair_rounds += 1
                        unreadable += witness is None
                        if witness:
                            witnessed += 1
                            covered.update(batch)
                    rounds_log.append(dict(mode=mode, clients=clients, round=r + 1, prompts=[i + 1 for i in batch],
                                           slots=slots, pair_round=pair, witness=witness, counters=[before, after],
                                           witness_detail=detail))
                (args.out / f"{name}-clients-{clients}.json").write_text(json.dumps(cell, indent=2))
                print(f"EQUAL {adapter.name} {name} clients {clients}", flush=True)
    record["rows"] = [dict(prompt_id=f"http-slots-p{i + 1}", mode="http-slots", generate_length=args.generate,
                           verdict="EXACT", repeat_stable=True, prompt_sha256=prompt_hash(p),
                           token_ids_sha256=id_hash(ref["token_ids"]), engine_digest=state_hash(ref["state"]),
                           actual_tokens=len(ref["token_ids"]), coverage=adapter.coverage, engine=adapter.name)
                      for i, (p, ref) in enumerate(zip(prompts, reference)) if i in covered]
    if pair_rounds == 0:
        raise GateError("UNQUALIFIED", "no two-request round ran on a mode with more than one slot (include 2 in --clients)")
    if witnessed == 0 and unreadable:
        raise GateError("UNQUALIFIED", f"{adapter.name}: no server counter proves shared execution"
                        + (f" ({adapter.witness_gap})" if adapter.witness_gap else ""))
    if witnessed == 0:
        raise GateError("NO-WITNESS", "no two-request round shared a server step")
    missing = [i + 1 for i in range(len(prompts)) if i not in covered]
    require(not missing, "UNQUALIFIED", f"prompts {missing} never ran in a two-request round with a shared-step witness")
    if args.allow_no_witness:
        raise GateError("UNQUALIFIED", "--allow-no-witness given: equality recorded, coverage not qualified")


def legacy_qwen_argv(args, raw):
    """Qwen without --generic: forward the caller's options to http_slots_gpu_gate.py unchanged."""
    out = [sys.executable, str(LEGACY_QWEN_GATE)]
    skip = {"--engine", "--generic", "--allow-no-witness", "--reference-repeat", "--peer", "--peer-dir", "--peer-command"}
    i = 0
    while i < len(raw):
        a = raw[i]
        if a == "--":
            out += raw[i:]
            break
        key = a.split("=", 1)[0]
        if key in skip:
            i += 1 if ("=" in a or key in ("--generic", "--allow-no-witness")) else 2
            continue
        out.append(a)
        i += 1
    return out


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", required=True, help="qwen|deepseek|minimax|glm|nemotron (aliases: *-pair)")
    parser.add_argument("--generic", action="store_true", help="qwen: use the shared adapter path, not the Qwen gate")
    parser.add_argument("--prompt", action="append", default=[], help="token ID file (text file for nemotron/minimax); repeat")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--generate", type=int)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--modes", help="comma list of slots modes (auto or N); default per engine")
    parser.add_argument("--clients", default="1,2,4,8,16")
    parser.add_argument("--reference-repeat", type=int, default=1, help="serial runs per prompt (repeat stability)")
    parser.add_argument("--allow-no-witness", action="store_true", help="record equality; verdict stays UNQUALIFIED")
    parser.add_argument("--peer", help="pair engines: ssh host for rank 1")
    parser.add_argument("--peer-dir", help="pair engines: existing directory on the peer for rank-1 logs/pid files")
    parser.add_argument("--peer-command", help="pair engines: rank-1 argv (shell words); default: local command with --rank 1")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- server executable and pinned arguments, no HTTP/slots flags")
    # Legacy Qwen options pass through to http_slots_gpu_gate.py untouched.
    parser.add_argument("--export-roundtrip", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--window-state-arm", help=argparse.SUPPRESS)
    parser.add_argument("--export-block-rows", help=argparse.SUPPRESS)
    parser.add_argument("--suffix-token", help=argparse.SUPPRESS)
    args = parser.parse_args(raw)
    adapter = adapter_for(args.engine)
    if args.out.resolve().is_relative_to(ROOT):
        parser.error("evidence must stay outside Git")
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "summary.json"
    summary_path.unlink(missing_ok=True)  # never leave an earlier verdict behind
    if adapter.name == "qwen" and not args.generic:
        # Backward compatible route: the Qwen gate's own checks. It does not validate stream completion
        # (finish_reason, usage, [DONE]) nor emit kit rows; --generic is the recommended route.
        code = subprocess.call(legacy_qwen_argv(args, raw))
        summary_path.write_text(json.dumps(dict(
            engine="qwen", route="legacy gate (tests/http_slots_gpu_gate.py)",
            coverage="legacy gate, stream completion not validated", verdict="LEGACY-PASS" if code == 0 else "LEGACY-FAIL",
            exit_code=code, message="legacy gate, stream completion not validated; use --generic for the shared "
            "hardened checks and bench-kit rows"), indent=2))
        return code
    if args.export_roundtrip or args.window_state_arm:
        parser.error("--export-roundtrip/--window-state-arm are Qwen gate arms (omit --generic)")
    if args.command[:1] == ["--"]:
        args.command.pop(0)
    args.modes = args.modes or adapter.default_modes
    args.generate = args.generate or adapter.default_generate
    summary = dict(engine=adapter.name, coverage=adapter.coverage if adapter.http else "not-run",
                   reference_mode=adapter.reference_mode, modes=args.modes, clients=args.clients,
                   generate=args.generate, prompts=len(args.prompt), verdict="ERROR", message="running")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    prompts = []
    try:
        require(adapter.http, "NOT-RUN", adapter.no_http_reason)
        require(args.command, "NO-REFERENCE", "server command required")
        require(args.prompt, "NO-REFERENCE", "at least one --prompt")
        require(args.generate >= 2, "NO-REFERENCE", "--generate must be at least 2")
        require(args.reference_repeat >= 1, "NO-REFERENCE", "--reference-repeat must be at least 1")
        adapter.check_command(args.command)
        for mode in args.modes.split(","):
            require(mode == "auto" or mode.isdigit(), "NO-REFERENCE", "mode must be auto or N: " + mode)
            require(mode == "auto" or adapter.max_slots is None or 1 <= int(mode) <= adapter.max_slots,
                    "NO-REFERENCE", f"{adapter.name} slots must be 1..{adapter.max_slots}: {mode}")
        require(all(c.isdigit() and int(c) >= 1 for c in args.clients.split(",")), "NO-REFERENCE", "bad --clients")
        if adapter.pair:
            require(bool(args.peer) == bool(args.peer_dir), "NO-REFERENCE", "--peer and --peer-dir go together")
            if not args.peer:
                print(f"note: {adapter.name} is a pair engine and no --peer was given; rank 1 must already follow "
                      "each server start (the gate restarts rank 0 per mode); rank-1 exits are not checked", flush=True)
                summary["rank1"] = "external (not checked)"
        prompts = [load_prompt(adapter, p) for p in args.prompt]
        require(len({prompt_hash(p) for p in prompts}) == len(prompts), "NO-REFERENCE",
                "prompts must be distinct (request binding is checked per prompt)")
        run_gate(args, adapter, prompts, summary)
        summary["verdict"] = "EXACT"
        detail = ("token IDs and committed state digests" if adapter.coverage == "tokens+state"
                  else "token IDs only (engine exposes no per-request state digest)")
        message = f"EXACT: {adapter.name}: every request equals the serial reference: {detail}"
    except GateError as error:
        summary["verdict"] = error.verdict
        message = str(error)
        if error.verdict == "MISMATCH" and error.prompt is not None and prompts:
            summary["rows"] = [dict(prompt_id=f"http-slots-p{error.prompt + 1}", mode="http-slots",
                                    generate_length=args.generate, verdict="MISMATCH", repeat_stable=False,
                                    prompt_sha256=prompt_hash(prompts[error.prompt]), engine=adapter.name,
                                    coverage=adapter.coverage,
                                    token_ids_sha256=(summary.get("reference_token_sha256") or [None] * len(prompts))[error.prompt])]
        elif error.verdict != "EXACT":
            summary.pop("rows", None)
    except Exception as error:  # noqa: BLE001 - any other failure must still replace the summary
        summary["verdict"] = "ERROR"
        summary.pop("rows", None)
        message = f"ERROR: {type(error).__name__}: {error}"
    summary["message"] = message
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(message, flush=True)
    return EXIT[summary["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
