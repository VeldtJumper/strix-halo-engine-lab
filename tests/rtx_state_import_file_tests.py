#!/usr/bin/env python3
"""CPU checks for the shared HTTP/CLI state-pair writer."""
import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("rtx_import", Path(__file__).resolve().parents[1] / "tools/rtx_state_import.py")
rtx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rtx)


class FilePairTests(unittest.TestCase):
    def test_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.rtxs"
            destination = Path(directory) / "state.json"
            payload = bytes(range(256)) * 17
            header = {"tokens": [1, 2], "position": 2,
                      "producer": {"numerical": "0x11", "reduction": "0x22", "state_update": "0x33"},
                      "layout_identity": "0x44", "geometry_identity": "0x55", "context_tokens": 524288,
                      "rotary": {"rope_mode": "yarn", "factor": 2, "original_max": 262144, "table_sha256": "1" * 64},
                      "payload_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}

            def save(data):
                metadata = json.dumps(header).encode("ascii")
                source.write_bytes(rtx.MAGIC + struct.pack("<Q", len(metadata)) + metadata + data)

            save(payload)
            result = rtx.write_file_pair(source, destination, chunk_bytes=31)
            expected = rtx.state_begin_body(header)
            expected.update(payload_bytes=len(payload), payload_sha256=header["sha256"])
            self.assertEqual(json.loads(destination.read_text()), expected)
            self.assertEqual(destination.with_suffix(".bin").read_bytes(), payload)
            self.assertEqual(result["payload_sha256"], header["sha256"])
            self.assertNotIn(b"\r", destination.read_bytes())
            save(payload[:-1] + b"x")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                rtx.write_file_pair(source, destination, chunk_bytes=31)
            self.assertFalse(destination.with_suffix(".bin").exists())
            save(payload[:-1])
            with self.assertRaises(ValueError):
                rtx.write_file_pair(source, destination)
            with self.assertRaises(ValueError):
                rtx.write_file_pair(source, source)


if __name__ == "__main__":
    unittest.main()
