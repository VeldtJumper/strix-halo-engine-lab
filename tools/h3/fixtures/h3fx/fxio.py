"""Fixture writer/reader: one raw little-endian .bin per tensor + manifest.json (format h3fx v1)."""

import hashlib
import json
import os
import time

import numpy as np
import torch

FORMAT = "h3fx"
VERSION = 1

# torch dtype -> (manifest name, numpy view dtype used for the raw bytes)
_DT = {
    torch.bfloat16: ("bf16", np.uint16),
    torch.float16: ("f16", np.float16),
    torch.float32: ("f32", np.float32),
    torch.float64: ("f64", np.float64),
    torch.int8: ("i8", np.int8),
    torch.uint8: ("u8", np.uint8),
    torch.int16: ("i16", np.int16),
    torch.int32: ("i32", np.int32),
    torch.int64: ("i64", np.int64),
    torch.bool: ("bool", np.uint8),
}
_NP_BY_NAME = {v[0]: v[1] for v in _DT.values()}
_TORCH_BY_NAME = {v[0]: k for k, v in _DT.items()}


def tensor_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().to("cpu").contiguous()
    name, npdt = _DT[t.dtype]
    if t.dtype == torch.bfloat16:
        arr = t.view(torch.int16).numpy().view(np.uint16)
    elif t.dtype == torch.bool:
        arr = t.to(torch.uint8).numpy()
    else:
        arr = t.numpy()
    arr = arr.astype(np.dtype(npdt).newbyteorder("<"), copy=False)
    return arr.tobytes(order="C")


class FixtureWriter:
    """Writes tensors under root/<set>/<name>.bin and the manifest at close()."""

    def __init__(self, root, set_name, meta=None):
        self.dir = os.path.join(root, set_name)
        os.makedirs(self.dir, exist_ok=True)
        self.set_name = set_name
        self.meta = dict(meta or {})
        self.entries = {}
        self.t0 = time.time()

    def add(self, name, tensor, desc=""):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        if name in self.entries:
            raise KeyError(f"duplicate fixture tensor {name}")
        data = tensor_bytes(tensor)
        rel = name + ".bin"
        path = os.path.join(self.dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        self.entries[name] = {
            "name": name,
            "file": rel,
            "dtype": _DT[tensor.dtype][0],
            "shape": list(tensor.shape),
            "nbytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "desc": desc,
        }
        return self.entries[name]["sha256"]

    def close(self, extra_meta=None):
        if extra_meta:
            self.meta.update(extra_meta)
        man = {
            "format": FORMAT,
            "version": VERSION,
            "set": self.set_name,
            "meta": self.meta,
            "tensors": [self.entries[k] for k in self.entries],
        }
        with open(os.path.join(self.dir, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(man, f, indent=1, sort_keys=False)
            f.write("\n")
        return man


def load_manifest(set_dir):
    with open(os.path.join(set_dir, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def read_tensor(set_dir, entry, verify=True):
    with open(os.path.join(set_dir, entry["file"]), "rb") as f:
        data = f.read()
    if verify and hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"sha256 mismatch for {entry['name']}")
    npdt = np.dtype(_NP_BY_NAME[entry["dtype"]]).newbyteorder("<")
    arr = np.frombuffer(data, dtype=npdt).reshape(entry["shape"]).copy()
    t = torch.from_numpy(arr.astype(npdt.newbyteorder("="), copy=False))
    if entry["dtype"] == "bf16":
        t = t.view(torch.int16).view(torch.bfloat16)
    elif entry["dtype"] == "bool":
        t = t.to(torch.bool)
    return t


def read_set(set_dir, verify=True):
    man = load_manifest(set_dir)
    return {e["name"]: read_tensor(set_dir, e, verify) for e in man["tensors"]}, man
