"""Shared helpers: canonical JSON, hashes, atomic writes, ID files."""
import io
import json
import re
import subprocess
from pathlib import Path

from tools.qualification.longctx_fidelity.common import atomic, canonical, file_sha, sha

__all__ = ["VERSION", "atomic", "canonical", "file_sha", "sha", "read_id_text", "write_id_text", "kit_commit",
           "JournalError", "read_journal"]

VERSION = "option-qual-v1"


class JournalError(ValueError):
    pass


def read_journal(path):
    """Records of a JSON-lines journal and its incomplete final append (bytes after the last newline, or
    None). Every complete line must parse; a malformed interior record is an error, never skipped."""
    path = Path(path)
    if not path.exists():
        return [], None
    data = path.read_bytes()
    body, sep, tail = data.rpartition(b"\n")
    if not sep:
        body, tail = b"", data
    records = []
    for n, line in enumerate(body.split(b"\n") if body else [], 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            raise JournalError(f"{path.name}: malformed record at line {n}") from None
    return records, (tail if tail.strip() else None)


def read_journal_lines(path, *, data=None):
    """Like read_journal, but tolerant: returns (lines, tail, raws). A line that does not parse is
    returned as its raw text (the run journal keeps every line; the evidence ledger decides what a bad
    line means); ``raws`` are the exact bytes of every line, for binding resets to a journal prefix.
    ``data`` lets callers retain and hash the complete buffer parsed here, including its tail."""
    path = Path(path)
    if data is None:
        if not path.exists():
            return [], None, []
        data = path.read_bytes()
    body, sep, tail = data.rpartition(b"\n")
    if not sep:
        body, tail = b"", data
    lines, raws = [], []
    for line in (body.split(b"\n") if body else []):
        if not line.strip():
            continue
        raws.append(line)
        try:
            lines.append(json.loads(line))
        except ValueError:
            lines.append(line.decode("utf-8", "replace"))
    return lines, (tail if tail.strip() else None), raws


def diagnostic_records(path, after_damage=False):
    """Records of a diagnostic journal and the number of damaged lines, an incomplete last line included
    (skipped, never an error). With after_damage, only the records after the last damaged line: damage
    is a durable boundary, so what was written again after it is used again."""
    lines, tail, _ = read_journal_lines(path)
    records, damaged = [], 0
    for r in lines:
        if isinstance(r, dict):
            records.append(r)
        else:
            damaged += 1
            if after_damage:
                records = []
    return records, damaged + (tail is not None)


def decode_log(data):
    """Text of captured engine output, decoded the same way at run time and when the analysis re-checks the saved
    bytes (no newline translation)."""
    return data.decode("utf-8", "replace")


def witness_problem(spec, result, stderr_text):
    """Why a run does not carry its arm's witness (None when it does, or when no witness is declared).

    ``spec`` = {"stdout": {key: value}, "stderr_forbid": [text], "stderr_require": [regex]}: every listed key of the
    run's JSON result equals its value (JSON-typed: 1, true and "1" differ), no forbidden text and every required
    pattern occurs in the run's stderr. A miss means the engine did not run what the arm declares (for example a lever
    declined, or the route fell back), so the run is not evidence for that arm."""
    if not spec:
        return None

    def j(v):
        return json.dumps(v, sort_keys=True, separators=(",", ":"))   # JSON-typed: 1, true and "1" differ
    for key, want in sorted(spec.get("stdout", {}).items()):
        got = result.get(key, None) if isinstance(result, dict) else None
        if not isinstance(result, dict) or key not in result or j(got) != j(want):
            return f"result {key} = {'absent' if not isinstance(result, dict) or key not in result else j(got)}, " \
                   f"declared {j(want)}"
    for text in spec.get("stderr_forbid", []):
        if text in stderr_text:
            return f"stderr contains {text!r}"
    for pattern in spec.get("stderr_require", []):
        if not re.search(pattern, stderr_text):
            return f"stderr lacks {pattern!r}"
    return None


def write_id_text(path, ids):
    atomic(path, (",".join(str(int(i)) for i in ids) + "\n").encode("ascii"))


def read_id_text(path):
    text = Path(path).read_text(encoding="ascii").strip()
    return [int(x) for x in text.replace("\n", ",").split(",") if x.strip()] if text else []


def kit_commit():
    """Commit of the checkout this package runs from; for an exported copy, the commit recorded in a
    ``COMMIT`` file next to this module at export time; else None."""
    stamp = Path(__file__).resolve().parent / "COMMIT"
    if stamp.exists():
        return stamp.read_text().strip() or None
    try:
        result = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10)
        return result.stdout.strip() or None if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def captured_text(data, encoding=None):
    """Decode captured bytes with Path.read_text semantics, including universal newlines."""
    with io.TextIOWrapper(io.BytesIO(data), encoding=encoding) as stream:
        return stream.read()
