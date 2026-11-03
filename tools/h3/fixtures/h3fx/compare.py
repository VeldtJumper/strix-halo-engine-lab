"""Self-consistency: compare the per-tensor sha256 of two fixture sets (run A vs re-run B).

Usage: python -m h3fx.compare SET_A SET_B   -> exit 0 iff the same names, shapes and sha256.
Also prints a set digest = sha256 over the sorted "name sha256" lines.
"""

import hashlib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from h3fx.fxio import load_manifest  # noqa: E402


def digest(man):
    lines = sorted(f"{e['name']} {e['sha256']}" for e in man["tensors"])
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def main():
    a, b = load_manifest(sys.argv[1]), load_manifest(sys.argv[2])
    ea = {e["name"]: e for e in a["tensors"]}
    eb = {e["name"]: e for e in b["tensors"]}
    diff = [n for n in sorted(set(ea) | set(eb))
            if n not in ea or n not in eb or ea[n]["sha256"] != eb[n]["sha256"] or ea[n]["shape"] != eb[n]["shape"]]
    print(f"A {sys.argv[1]}: {len(ea)} tensors digest {digest(a)}")
    print(f"B {sys.argv[2]}: {len(eb)} tensors digest {digest(b)}")
    if diff:
        print(f"MISMATCH {len(diff)} tensors, first: {diff[:10]}")
        return 1
    print("IDENTICAL (all sha256 equal)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
