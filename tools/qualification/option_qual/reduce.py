"""Reduce a reference and a candidate logit file to per-position metrics.

usage: python3 -m tools.qualification.option_qual.reduce REF CAND --format steps|spool --out STEM
       (exit 3 = the files describe different forced histories)
       python3 -m tools.qualification.option_qual.reduce REF --self --format steps --out STEM

Both files must describe the same forced history: equal row count and, row by row, equal step,
position, label and vocabulary. Any difference raises; nothing is skipped. Writes STEM.npz (one
array per metric) and STEM.json (row count, vocabulary, file hashes, bitwise identity flag).
"""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

from .common import captured_text
from .common import atomic, canonical, file_sha
from .metrics import FIELDS, SELF_FIELDS, compare, self_metrics
from .records import reader

INT_FIELDS = {"agree", "top1_r", "top1_c", "nonfinite_r", "nonfinite_c", "top1", "is_label", "nonfinite",
              "top1_r_is_label", "top1_c_is_label"}


EXIT_HISTORY = 3   # CLI exit code: the two files do not describe the same forced history


class HistoryMismatch(ValueError):
    pass


def _arrays(rows, fields):
    return {k: np.asarray([r[k] for r in rows], dtype=np.int64 if k in INT_FIELDS else np.float64) for k in fields}


def reduce_pair(ref_path, cand_path, fmt="steps"):
    read = reader(fmt)
    rows, labels, identical, vocab = [], [], True, None
    h_ref, h_cand = hashlib.sha256(), hashlib.sha256()     # digests of the bytes actually reduced
    ref_iter, cand_iter = read(ref_path, h_ref), read(cand_path, h_cand)
    while True:
        a = next(ref_iter, None)
        b = next(cand_iter, None)
        if a is None and b is None:
            break
        if a is None or b is None:
            raise HistoryMismatch(f"row count differs after {len(rows)} rows")
        if (a.step, a.position, a.label, a.logits.size) != (b.step, b.position, b.label, b.logits.size):
            raise HistoryMismatch(f"row {len(rows)}: step/position/label/vocabulary differ "
                                  f"{(a.step, a.position, a.label, a.logits.size)} vs "
                                  f"{(b.step, b.position, b.label, b.logits.size)}")
        vocab = a.logits.size
        identical = identical and a.argmax == b.argmax and a.logits.tobytes() == b.logits.tobytes()
        rows.append(compare(a.logits, b.logits, a.label))
        labels.append(a.label)
    if not rows:
        raise HistoryMismatch("no rows")
    arrays = _arrays(rows, FIELDS)
    arrays["label"] = np.asarray(labels, dtype=np.int64)
    info = {"rows": len(rows), "vocab": vocab, "format": fmt, "ref_sha256": h_ref.hexdigest(),
            "cand_sha256": h_cand.hexdigest(), "bitwise_identical": bool(identical)}
    return arrays, info


def reduce_self(ref_path, fmt="steps"):
    rows, labels, vocab = [], [], None
    h_ref = hashlib.sha256()
    for r in reader(fmt)(ref_path, h_ref):
        rows.append(self_metrics(r.logits, r.label))
        labels.append(r.label)
        vocab = r.logits.size
    if not rows:
        raise HistoryMismatch("no rows")
    arrays = _arrays(rows, SELF_FIELDS)
    arrays["label"] = np.asarray(labels, dtype=np.int64)
    return arrays, {"rows": len(rows), "vocab": vocab, "format": fmt, "ref_sha256": h_ref.hexdigest()}


def save(stem, arrays, info):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(stem) + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(str(stem) + ".npz")
    atomic(str(stem) + ".json", canonical(info))


def load_bytes(npz_bytes, json_bytes):
    """Parse exactly the captured reduction buffers (also used by hash verification)."""
    with np.load(io.BytesIO(npz_bytes)) as data:
        arrays = {k: data[k] for k in data.files}
    return arrays, json.loads(captured_text(json_bytes))


def load(stem):
    stem = Path(stem)
    return load_bytes(Path(str(stem) + ".npz").read_bytes(), Path(str(stem) + ".json").read_bytes())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ref")
    p.add_argument("cand", nargs="?")
    p.add_argument("--self", action="store_true", dest="self_only")
    p.add_argument("--format", default="steps", choices=["steps", "spool", "bclg", "tmllog"],
                   help="steps (alias bclg) or spool (alias tmllog)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    a.format = {"bclg": "steps", "tmllog": "spool"}.get(a.format, a.format)
    if a.self_only == bool(a.cand):
        p.error("give CAND, or --self without CAND")
    try:
        arrays, info = reduce_self(a.ref, a.format) if a.self_only else reduce_pair(a.ref, a.cand, a.format)
    except HistoryMismatch as e:
        print(f"history mismatch: {e}", file=sys.stderr)
        return EXIT_HISTORY
    save(a.out, arrays, info)
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
