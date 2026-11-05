"""Streaming readers for full-vocabulary logit records.

Two little-endian formats, FP32 logits in ascending vocabulary order:

* step stream (``STRIX_BITCALC_LOGITS_OUT``): per token selection a header of six uint32
  ``magic, step, position, argmax, emitted, vocab`` followed by ``vocab`` floats. With forced
  tokens, ``emitted`` is the forced token, which is the label the row is scored against.
* teacher spool (``--teacher-logits-file``): ``TMLLOG01``, u64 vocab, u64 rows, u64 input count,
  u32 input IDs; per row u32 input, u64 position, ``vocab`` floats; footer ``TMLEOF01``, u64 rows.
  Row ``r`` is scored against input ``r + 1``.

Readers never load a whole file and reject truncated or incomplete files.
"""
from dataclasses import dataclass
import struct

import numpy as np

STEP_MAGIC = 0x42434C47
STEP_HEADER = struct.Struct("<6I")
SPOOL_MAGIC, SPOOL_FOOTER = b"TMLLOG01", b"TMLEOF01"


@dataclass
class Row:
    step: int
    position: int
    argmax: int
    label: int
    logits: np.ndarray  # float32, read-only view of the file bytes


def _exact(f, n, what):
    data = f.read(n)
    if len(data) != n:
        raise ValueError(f"truncated {what}")
    return data


class _Hashing:
    """A file whose read() also feeds every returned byte to a hasher (the digest of the bytes actually read)."""
    def __init__(self, f, hasher):
        self.f, self.h = f, hasher

    def read(self, n=-1):
        data = self.f.read(n)
        self.h.update(data)
        return data


def read_steps(path, hasher=None):
    with open(path, "rb") as raw:
        f = _Hashing(raw, hasher) if hasher is not None else raw
        while True:
            head = f.read(STEP_HEADER.size)
            if not head:
                return
            if len(head) != STEP_HEADER.size:
                raise ValueError("truncated step header")
            magic, step, position, argmax, emitted, vocab = STEP_HEADER.unpack(head)
            if magic != STEP_MAGIC:
                raise ValueError("bad step-record magic")
            if not vocab or argmax >= vocab or emitted >= vocab:
                raise ValueError("step record IDs outside vocabulary")
            logits = np.frombuffer(_exact(f, 4 * vocab, "step logits"), dtype="<f4")
            yield Row(step, position, argmax, emitted, logits)


def read_spool(path, hasher=None):
    with open(path, "rb") as raw:
        f = _Hashing(raw, hasher) if hasher is not None else raw
        if f.read(8) != SPOOL_MAGIC:
            raise ValueError("bad spool magic")
        vocab, rows, count = struct.unpack("<QQQ", _exact(f, 24, "spool header"))
        if not vocab or count < 2 or rows >= count:
            raise ValueError("inconsistent spool dimensions")
        inputs = np.frombuffer(_exact(f, 4 * count, "spool inputs"), dtype="<u4")
        if inputs.size and int(inputs.max()) >= vocab:
            raise ValueError("spool input outside vocabulary")
        row_head = struct.Struct("<IQ")
        for r in range(rows):
            token, position = row_head.unpack(_exact(f, row_head.size, "spool row header"))
            if position != r or token != int(inputs[r]):
                raise ValueError("spool row order or input mismatch")
            logits = np.frombuffer(_exact(f, 4 * vocab, "spool logits"), dtype="<f4")
            yield Row(r, position, -1, int(inputs[r + 1]), logits)
        if f.read(8) != SPOOL_FOOTER or struct.unpack("<Q", _exact(f, 8, "spool footer"))[0] != rows:
            raise ValueError("spool footer missing or wrong (incomplete file)")
        if f.read(1):
            raise ValueError("trailing bytes after spool footer")


READERS = {"steps": read_steps, "spool": read_spool}


def reader(fmt):
    try:
        return READERS[fmt]
    except KeyError:
        raise ValueError(f"unknown record format {fmt!r}; expected one of {sorted(READERS)}") from None


def write_steps(path, rows):
    """Test/fixture writer: rows = iterable of (step, position, argmax, emitted, logits)."""
    with open(path, "ab") as f:
        for step, position, argmax, emitted, logits in rows:
            z = np.asarray(logits, dtype="<f4")
            f.write(STEP_HEADER.pack(STEP_MAGIC, step, position, argmax, emitted, z.size))
            f.write(z.tobytes())


def write_spool(path, inputs, rows_logits):
    """Test/fixture writer: rows_logits[r] are the logits of row r (len(inputs) - 1 rows)."""
    inputs = list(inputs)
    rows_logits = [np.asarray(z, dtype="<f4") for z in rows_logits]
    vocab = rows_logits[0].size
    with open(path, "wb") as f:
        f.write(SPOOL_MAGIC + struct.pack("<QQQ", vocab, len(rows_logits), len(inputs)))
        f.write(np.asarray(inputs, dtype="<u4").tobytes())
        for r, z in enumerate(rows_logits):
            f.write(struct.pack("<IQ", inputs[r], r) + z.tobytes())
        f.write(SPOOL_FOOTER + struct.pack("<Q", len(rows_logits)))
