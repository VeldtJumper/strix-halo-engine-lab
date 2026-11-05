"""Validate one engine logits file against its forced continuation, in a child process.

usage: python3 -m tools.qualification.option_qual.validate LOGITS CONTINUATION_FILE ROWS
       python3 -m tools.qualification.option_qual.validate --sha FILE

Prints {"rows", "problem", "sha256"} (or {"sha256"}) as JSON. The runner starts this as a separate process so that a
slow validation or hash can be stopped at the invocation deadline (a budget stop, never a failure).
"""
import json
from pathlib import Path
import struct
import sys

import hashlib

from .common import file_sha, read_id_text
from .records import STEP_HEADER


def _step_heads(data):
    """(rows, [(step, position, label)]) of an in-memory step-record file (same rules as run.count_step_rows)."""
    if not data:
        return 0, []
    if len(data) < STEP_HEADER.size:
        raise ValueError("logits file shorter than one record header")
    vocab = STEP_HEADER.unpack_from(data, 0)[5]
    if not vocab:
        raise ValueError("logits record with an empty vocabulary")
    record = STEP_HEADER.size + 4 * vocab
    if len(data) % record:
        raise ValueError("logits file is not a whole number of records")
    heads = []
    for off in range(0, len(data), record):
        _, step, position, _, emitted, v = STEP_HEADER.unpack_from(data, off)
        if v != vocab:
            raise ValueError("vocabulary changes between records")
        heads.append((step, position, emitted))
    return len(heads), heads


def _forced_problem(heads, cont):
    """Same rules as run.check_forced, on the headers of the bytes in memory."""
    if [h[2] for h in heads] != list(cont):
        return "logged labels differ from the continuation (teacher forcing not applied?)"
    if [h[0] for h in heads] != list(range(len(heads))):
        return "steps are not 0..n-1"
    if any(b[1] != a[1] + 1 for a, b in zip(heads, heads[1:])):
        return "positions are not consecutive"
    return None


def validate(path, cont_file, expected):
    """Row count, forced continuation and sha256 of ONE read of the file (the digest names the bytes checked)."""
    rows, problem, digest = 0, None, None
    try:
        data = Path(path).read_bytes()
        rows, heads = _step_heads(data)
        if rows != expected:
            problem = f"{rows} logged rows != continuation {expected}"
        else:
            problem = _forced_problem(heads, read_id_text(cont_file))
        if problem is None:
            digest = hashlib.sha256(data).hexdigest()
    except (OSError, ValueError, struct.error) as e:
        problem = f"{type(e).__name__}: {e}"
    return {"rows": rows, "problem": problem, "sha256": digest}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "--sha":
        print(json.dumps({"sha256": file_sha(argv[1]) if Path(argv[1]).is_file() else None}))
        return 0
    if len(argv) != 3:
        print(__doc__.splitlines()[2], file=sys.stderr)
        return 2
    print(json.dumps(validate(argv[0], argv[1], int(argv[2]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
