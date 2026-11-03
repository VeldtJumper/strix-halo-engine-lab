#!/usr/bin/env python3
"""Client of the Qwen server's externally produced prompt state import (--rtx-state-import, docs/rtx-state-import.md).

  upload    send a strix-qwen-rtx-state-v1 file (STRIX_RTX_EXPORT_STATE) to a server, then optionally run
            /v1/completions on a prompt that starts with the state's tokens; prints tokens and timings as JSON
  file-pair write NAME.json/NAME.bin for --rtx-state-import-file from an exported state
  complete  /v1/completions only (the baseline without an import)
  layout    print GET /admin/rtx-import/layout
  check     compare a state file's recorded session generations with a server's layout formula
  ckpt      decode the metadata of a prefix-cache checkpoint file (ckpt-*.tmpc) and, with --layout, compare its
            generations with the formula (the T-boundary check)

Only the standard library. Protocol: begin (JSON), chunks (multi-segment bodies, X-Chunk-Sha256 + X-Chunk-Seq,
one keep-alive connection), commit {chunks, bytes, manifest_sha256}.
"""
import argparse
import hashlib
import http.client
import json
import struct
import sys
import time
import urllib.parse

MAGIC = b"STRXRTX1"


def read_tokens(path):
    with open(path, encoding="utf-8") as f:
        text = f.read().replace(",", " ")
    return [int(t) for t in text.split()]


class Server:
    def __init__(self, url, api_key=None, timeout=600):
        u = urllib.parse.urlparse(url)
        self.host, self.port = u.hostname, u.port or 80
        self.timeout = timeout
        self.headers = {"Connection": "keep-alive"}
        if api_key:
            self.headers["Authorization"] = "Bearer " + api_key
        self.conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)

    def request(self, method, path, body=None, headers=None, expect=200):
        h = dict(self.headers)
        if headers:
            h.update(headers)
        if body is not None and "Content-Type" not in h:
            h["Content-Type"] = "application/json" if isinstance(body, str) else "application/octet-stream"
        data = body.encode() if isinstance(body, str) else body
        for attempt in (0, 1):
            try:
                self.conn.request(method, path, body=data, headers=h)
                r = self.conn.getresponse()
                payload = r.read()
                break
            except (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError):
                if attempt:
                    raise
                self.conn.close()
                self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        if expect is not None and r.status != expect:
            raise RuntimeError(f"{method} {path}: HTTP {r.status} {payload[:2000].decode(errors='replace')}")
        return r.status, json.loads(payload) if payload else None


def read_state_header(path):
    with open(path, "rb") as f:
        if f.read(8) != MAGIC:
            raise SystemExit(f"{path}: not a strix-qwen-rtx-state-v1 file")
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 16 + n


def formula(layout, position):
    per_row = [int(x) for x in layout["generations_per_row"]]
    base = [int(x) for x in layout["generations_base"]]
    return [b + position * r for b, r in zip(base, per_row)], int(layout["valid_base"]) | int(layout["valid_union"])


def check_generations(layout, header):
    expected, valid = formula(layout, int(header["position"]))
    got = [int(x) for x in header["session"]["generations"]]
    return {"expected": expected, "exported": got, "valid_expected": valid, "valid_exported": int(header["session"]["valid"]),
            "match": expected == got and valid == int(header["session"]["valid"])}


def row_offsets(layout, position):
    out, at = [], 0
    for r in layout["rows"]:
        rows = max(0, position - int(r["shift"]))
        out.append((at, rows, int(r["bytes_per_row"])))
        at += rows * int(r["bytes_per_row"])
    return out, at


def segment(kind, comp, a, b, data):
    return struct.pack("<IIIIQ", kind, comp, a, b, len(data)) + data


def state_begin_body(header, keep=False):
    begin_request = {
        "tokens": header["tokens"], "position": int(header["position"]), "keep": keep,
        "producer": {k: str(int(v, 0)) for k, v in header["producer"].items()},
        "layout_identity": str(int(header["layout_identity"], 0)), "geometry_identity": str(int(header["geometry_identity"], 0))}
    for key in ("rotary", "context_tokens"):
        if key in header:
            begin_request[key] = header[key]
    begin_request["encoding"] = "identity"
    return begin_request


def write_file_pair(source, destination, chunk_bytes=4 << 20):
    """Write the HTTP begin body and its canonical, unencoded payload beside it."""
    from pathlib import Path
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    header, offset = read_state_header(source)
    metadata = Path(destination)
    binary = metadata.with_suffix(".bin")
    if metadata.suffix != ".json" or Path(source).resolve() in (metadata.resolve(), binary.resolve()):
        raise ValueError("destination must be a distinct NAME.json/NAME.bin pair")
    body = state_begin_body(header)
    count, digest = 0, hashlib.sha256()
    with open(source, "rb") as src, open(binary, "wb") as dst:
        src.seek(offset)
        while True:
            data = src.read(chunk_bytes)
            if not data:
                break
            digest.update(data)
            dst.write(data)
            count += len(data)
    if count != int(header["payload_bytes"]) or digest.hexdigest() != header["sha256"]:
        binary.unlink()
        raise ValueError("source payload size or SHA256 mismatch")
    body.update(payload_bytes=count, payload_sha256=digest.hexdigest())
    metadata.write_text(json.dumps(body, indent=2) + "\n", encoding="ascii", newline="\n")
    return {"payload_bytes": count, "payload_sha256": digest.hexdigest()}


def upload(server, path, chunk_bytes, keep=False, canonical=False, check_only=False):
    header, offset = read_state_header(path)
    _, layout = server.request("GET", "/admin/rtx-import/layout")
    p = int(header["position"])
    problems = []
    if layout["format"] != header["format"]:
        problems.append("format")
    for key in ("layout_identity", "geometry_identity"):
        if int(layout[key], 0) != int(header[key], 0):
            problems.append(key)
    for key in ("model_content", "identity_extension"):
        if layout.get(key) and header.get(key) and layout[key] != header[key]:
            problems.append(key)
    if [(r["bytes_per_row"], r["shift"]) for r in layout["rows"]] != [(r["bytes_per_row"], r["shift"]) for r in header["rows"]]:
        problems.append("rows")
    if "context_tokens" in header and int(header["context_tokens"]) != int(layout.get("context_tokens", layout["max_context"])):
        problems.append("context_tokens")
    if "rotary" in header:
        def rotary_key(value):
            return (value.get("rope_mode", "native"), value.get("factor", 0),
                    value.get("original_max", 262144), value.get("table_sha256"))
        if rotary_key(header["rotary"]) != rotary_key(layout.get("rotary", {})):
            problems.append("rotary")
    if problems:
        raise SystemExit("the state file does not target this server: " + ", ".join(problems))
    generations = check_generations(layout, header)
    report = {"position": p, "payload_bytes": int(header["payload_bytes"]), "generations": generations}
    if check_only:
        return report, header
    t0 = time.time()
    begin_request = state_begin_body(header, keep)
    _, begin = server.request("POST", "/admin/rtx-import/begin", json.dumps(begin_request))
    ident = begin["id"]
    t1 = time.time()
    offsets, rows_end = row_offsets(layout, p)
    row_bytes = int(layout["row_bytes"])
    digests, seq, data_bytes, sent = [], 0, 0, 0
    with open(path, "rb") as f:
        def read(at, n):
            f.seek(offset + at)
            return f.read(n)

        def post(body):
            nonlocal seq, sent
            d = hashlib.sha256(body).digest()
            server.request("POST", f"/admin/rtx-import/chunks?id={ident}", body,
                           {"X-Chunk-Sha256": d.hex(), "X-Chunk-Seq": str(seq), "Content-Type": "application/octet-stream"})
            digests.append(d)
            seq += 1
            sent += len(body)

        block = max(1, chunk_bytes // row_bytes)
        for first in range(0, p, block):
            body = bytearray()
            for c, (at, rows, bpr) in enumerate(offsets):
                if first >= rows:
                    continue
                n = min(block, rows - first)
                body += segment(0, c, first, n, read(at + first * bpr, n * bpr))
                data_bytes += n * bpr
            if body:
                post(bytes(body))
        body = bytearray()
        for i, fx in enumerate(layout["fixed"]):
            size = int(fx["bytes"])
            for o in range(0, size, chunk_bytes):
                n = min(chunk_bytes, size - o)
                if body and len(body) + n > chunk_bytes:
                    post(bytes(body))
                    body = bytearray()
                body += segment(1, i, o & 0xffffffff, o >> 32, read(rows_end + int(fx["offset"]) + o, n))
                data_bytes += n
        if body:
            post(bytes(body))
    t2 = time.time()
    commit = {"chunks": len(digests), "bytes": data_bytes, "manifest_sha256": hashlib.sha256(b"".join(digests)).hexdigest()}
    if canonical:
        commit["canonical_sha256"] = True
    _, committed = server.request("POST", f"/admin/rtx-import/commit?id={ident}", json.dumps(commit))
    t3 = time.time()
    report.update({"id": ident, "chunks": len(digests), "data_bytes": data_bytes, "http_body_bytes": sent,
                   "begin_s": t1 - t0, "upload_s": t2 - t1, "commit_s": t3 - t2,
                   "upload_MBps": data_bytes / max(t2 - t1, 1e-9) / 1e6, "commit": committed})
    if canonical:
        report["canonical_matches_file"] = committed.get("canonical_sha256") == header["sha256"]
    return report, header


def complete(server, prompt, max_tokens):
    t0 = time.time()
    _, out = server.request("POST", "/v1/completions", json.dumps({
        "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "return_token_ids": True}), expect=200)
    strix = out.get("strix", {})
    return {"token_ids": strix.get("token_ids"), "text": out["choices"][0]["text"], "finish": out["choices"][0]["finish_reason"],
            "usage": out.get("usage"), "strix": strix, "wall_s": time.time() - t0}


def decode_ckpt(path):
    with open(path, "rb") as f:
        raw = f.read()
    position, block_rows, envelope, segments, fixed_bytes, metadata_bytes = struct.unpack_from("<QIIQQQ", raw, 72)
    meta_at = 128 + 32 * segments
    m = struct.unpack_from(f"<{metadata_bytes // 8}Q", raw, meta_at)
    certificate = []
    if envelope == 1:
        if not m or not 0 < m[0] < len(m):
            raise SystemExit("malformed prefix producer certificate")
        certificate, m = list(m[1:1 + m[0]]), m[1 + m[0]:]
    elif envelope:
        raise SystemExit("unknown prefix metadata envelope")
    if m[0] != 0x31544d5043585054:
        raise SystemExit("not a prefix-state checkpoint metadata block")
    out = {"position": position, "block_rows": block_rows, "segments": segments, "fixed_bytes": fixed_bytes,
           "producer_certificate": certificate,
           "next_position": m[1], "committed_tokens": m[2], "layout_identity": hex(m[3]), "row_bytes": m[4], "fixed_bytes_meta": m[5],
           "execution_identity": hex(m[6]), "runtime_compatibility_identity": hex(m[7]), "valid": m[8], "generations": list(m[9:15])}
    count = m[17]
    out["lineage"] = [list(m[18 + 6 * i: 24 + 6 * i]) for i in range(count)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["upload", "complete", "layout", "check", "ckpt", "file-pair"])
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--api-key")
    ap.add_argument("--state", help="strix-qwen-rtx-state-v1 file")
    ap.add_argument("--prompt-tokens", help="token ids (comma/space separated) of the completion prompt")
    ap.add_argument("--max-tokens", type=int, default=0, help="0: upload only")
    ap.add_argument("--chunk-mib", type=int, default=16)
    ap.add_argument("--keep", action="store_true", help="the parked state survives the request")
    ap.add_argument("--canonical", action="store_true", help="diagnostic: the server also hashes the canonical payload")
    ap.add_argument("--ckpt", help="ckpt-*.tmpc file (command ckpt)")
    ap.add_argument("--file-pair", help="destination NAME.json for command file-pair")
    ap.add_argument("--out", help="also write the JSON result here")
    a = ap.parse_args()
    server = Server(a.server, a.api_key)  # connects lazily
    result = {}
    if a.command == "file-pair":
        result = write_file_pair(a.state, a.file_pair, a.chunk_mib << 20)
    elif a.command == "layout":
        result = server.request("GET", "/admin/rtx-import/layout")[1]
    elif a.command == "check":
        result = upload(server, a.state, a.chunk_mib << 20, check_only=True)[0]
    elif a.command == "ckpt":
        result = decode_ckpt(a.ckpt)
        if a.server:
            try:
                layout = server.request("GET", "/admin/rtx-import/layout")[1]
                expected, valid = formula(layout, result["position"])
                result["formula"] = {"expected": expected, "valid_expected": valid,
                                     "match": expected == result["generations"] and valid == result["valid"]}
            except Exception as e:  # the server may run without --rtx-state-import
                result["formula_error"] = str(e)
    elif a.command == "upload":
        report, header = upload(server, a.state, a.chunk_mib << 20, a.keep, a.canonical)
        result["import"] = report
        if a.max_tokens:
            prompt = read_tokens(a.prompt_tokens) if a.prompt_tokens else header["tokens"]
            if prompt[: len(header["tokens"])] != header["tokens"] or len(prompt) <= len(header["tokens"]):
                raise SystemExit("the prompt must start with the state's tokens and be longer")
            result["completion"] = complete(server, prompt, a.max_tokens)
            result["import_used"] = "rtx_import" in result["completion"]["strix"]
    elif a.command == "complete":
        result["completion"] = complete(server, read_tokens(a.prompt_tokens), a.max_tokens)
    text = json.dumps(result, indent=1)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    if a.command == "upload" and a.max_tokens and not result.get("import_used"):
        sys.exit(3)


if __name__ == "__main__":
    main()
