"""GGUF reader tests on synthetic files; no model artifacts required."""
import struct
import tempfile
import unittest
from pathlib import Path

from research.sovereign_learning.s0_trainability.gguf_reader import GGUFFile
from research.sovereign_learning.s0_trainability.ptq1_codec import (
    decode_matrix, encode_ptq1_block)


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def build_gguf(rows: list[bytes], *, alignment: int = 32) -> bytes:
    kv = _s("general.alignment") + struct.pack("<II", 4, alignment)
    kv += _s("prism.hadamard.sign_widths") + struct.pack("<IIQ", 9, 5, 1) + struct.pack("<i", 256)
    info = _s("w") + struct.pack("<I", 2) + struct.pack("<QQ", 256, len(rows)) + struct.pack("<IQ", 143, 0)
    head = b"GGUF" + struct.pack("<IQQ", 3, 1, 2) + kv + info
    head += bytes(-len(head) % alignment)
    return head + b"".join(rows)


class ReaderTests(unittest.TestCase):
    def _write(self, blob: bytes) -> Path:
        tmp = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        tmp.write(blob)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        return Path(tmp.name)

    def test_locates_rows_and_metadata(self):
        r0 = encode_ptq1_block([1, 0, -1, 0] * 32, 0.5) + encode_ptq1_block([-1] * 128, 2.0)
        r1 = encode_ptq1_block([0] * 128, 1.0) + encode_ptq1_block([1] * 128, 0.25)
        with GGUFFile(self._write(build_gguf([r0, r1]))) as g:
            self.assertEqual(g.metadata["prism.hadamard.sign_widths"], [256])
            info = g.tensors["w"]
            self.assertEqual((info.row_length, info.rows, info.nbytes), (256, 2, 112))
            self.assertEqual(bytes(g.payload("w", first_row=1, rows=1)), r1)
            q, s = decode_matrix(bytes(g.payload("w")), rows=2, cols=256)
            self.assertEqual(q[0][:4], [1, 0, -1, 0])
            self.assertEqual(s, [[0.5, 2.0], [1.0, 0.25]])

    def test_rejects_truncated_payload_and_bad_magic(self):
        blob = build_gguf([encode_ptq1_block([0] * 128, 1.0) * 2])
        with self.assertRaises(ValueError):
            GGUFFile(self._write(blob[:-1]))
        with self.assertRaises(ValueError):
            GGUFFile(self._write(b"GGUX" + blob[4:]))

    def test_row_range_checked(self):
        blob = build_gguf([encode_ptq1_block([0] * 128, 1.0) * 2])
        with GGUFFile(self._write(blob)) as g:
            with self.assertRaises(ValueError):
                g.payload("w", first_row=1, rows=1)


if __name__ == "__main__":
    unittest.main()
