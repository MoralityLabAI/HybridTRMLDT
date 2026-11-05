"""Bounded CPU tests; no model artifacts or GPU required."""
import math
import random
from fractions import Fraction
import struct
import unittest
from itertools import product

from research.sovereign_learning.s0_trainability.ptq1_codec import (
    GROUP, GGUF_PQ2_TYPE, GGUF_PTQ1_TYPE, PTQ1_BLOCK_BYTES,
    PQ2_BLOCK_BYTES, encode_ptq1_block, decode_ptq1_block,
    encode_pq2_block, decode_pq2_block, decode_matrix,
)


class CodecTests(unittest.TestCase):
    def test_format_contract(self):
        self.assertEqual((GROUP, PTQ1_BLOCK_BYTES, PQ2_BLOCK_BYTES), (128,28,34))
        self.assertEqual((GGUF_PTQ1_TYPE,GGUF_PQ2_TYPE), (143,142))

    def test_c_layout_known_vectors(self):
        self.assertEqual(encode_ptq1_block([-1]*128,1),bytes(26)+bytes.fromhex("003c"))
        self.assertEqual(encode_ptq1_block([0]*128,1),bytes([128]*24+[127,127])+bytes.fromhex("003c"))
        self.assertEqual(encode_ptq1_block([1]*128,1),bytes([255]*24+[253,253])+bytes.fromhex("003c"))

    def test_each_five_trit_pattern_in_two_stages(self):
        for offset,width in ((0,16),(80,8)):
            for digits in product((-1,0,1),repeat=5):
                q=[0]*128
                for n,t in enumerate(digits):
                    q[offset+n*width]=t
                blob=encode_ptq1_block(q,0.75)
                self.assertEqual(len(blob),28)
                self.assertEqual(decode_ptq1_block(blob),(tuple(q),0.75))

    def test_each_four_trit_pattern_in_qh(self):
        for digits in product((-1,0,1),repeat=4):
            q=[0]*128
            for n,t in enumerate(digits):
                q[120+n*2]=t
            self.assertEqual(decode_ptq1_block(encode_ptq1_block(q,1))[0],tuple(q))

    def test_pq2_scale_first_known_encode(self):
        expected = bytes.fromhex("00 38 02") + bytes(31)
        self.assertEqual(encode_pq2_block([1] + [-1] * 127, 0.5), expected)

    def test_pq2_scale_first_known_decode(self):
        independent_block = bytes.fromhex("00 38 02") + bytes(31)
        self.assertEqual(decode_pq2_block(independent_block),
                         (tuple([1] + [-1] * 127), 0.5))

    def test_non_integer_trits_rejected_without_truncation(self):
        invalid = (True, False, 0.5, -0.5, 1.5, 1.0, float("nan"),
                   float("inf"), float("-inf"), Fraction(1, 2), Fraction(1, 1))
        for encoder in (encode_ptq1_block, encode_pq2_block):
            for value in invalid:
                with self.subTest(encoder=encoder.__name__, value=repr(value)):
                    with self.assertRaises(ValueError):
                        encoder([value] + [0] * 127, 0.5)

    def test_integer_index_scalars_supported(self):
        class IndexScalar:
            def __init__(self, value):
                self.value = value

            def __index__(self):
                return self.value

        trits = [IndexScalar(1), IndexScalar(0), IndexScalar(-1)] + [0] * 125
        expected = tuple([1, 0, -1] + [0] * 125)
        for encoder, decoder in ((encode_ptq1_block, decode_ptq1_block),
                                 (encode_pq2_block, decode_pq2_block)):
            with self.subTest(encoder=encoder.__name__):
                self.assertEqual(decoder(encoder(trits, 0.5)), (expected, 0.5))

    def test_random_cross_format_identity(self):
        rng=random.Random(20261028)
        for _ in range(512):
            q=[rng.randrange(-1,2) for _ in range(GROUP)]
            scale=rng.randrange(1,2048)/1024
            self.assertEqual(decode_ptq1_block(encode_ptq1_block(q,scale)),
                             decode_pq2_block(encode_pq2_block(q,scale)))

    def test_matrix_rows_and_groups(self):
        rng=random.Random(19)
        q=[[rng.randrange(-1,2) for _ in range(256)] for _ in range(2)]
        s=[[0.5,1],[1.5,2]]
        for fmt,encoder in (("ptq1",encode_ptq1_block),("pq2",encode_pq2_block)):
            payload=b"".join(encoder(q[i][g*128:(g+1)*128],s[i][g])
                              for i in range(2) for g in range(2))
            self.assertEqual(decode_matrix(payload,rows=2,cols=256,fmt=fmt),(q,s))

    def test_invalid_shapes_and_values_fail_closed(self):
        with self.assertRaises(ValueError):decode_ptq1_block(bytes(27))
        with self.assertRaises(ValueError):decode_pq2_block(bytes(33))
        with self.assertRaises(ValueError):encode_ptq1_block([0]*127,1)
        with self.assertRaises(ValueError):encode_pq2_block([2]+[0]*127,1)
        with self.assertRaises(ValueError):encode_ptq1_block([0]*128,math.inf)
        with self.assertRaises(ValueError):decode_matrix(bytes(28),rows=1,cols=256)
        with self.assertRaises(ValueError):decode_matrix(bytes(28),rows=1,cols=128,fmt="q1")
        invalid=bytearray(encode_pq2_block([0]*128,1))
        invalid[2]=3  # First code byte follows the two-byte FP16 scale.
        with self.assertRaisesRegex(ValueError, r"\+2"):
            decode_pq2_block(bytes(invalid))
        nan_pq2_scale=struct.pack("<e",float("nan"))+bytes(32)
        with self.assertRaises(ValueError):decode_pq2_block(nan_pq2_scale)
        nan_scale=encode_ptq1_block([0]*128,1)[:-2]+struct.pack("<e",float("nan"))
        with self.assertRaises(ValueError):decode_ptq1_block(nan_scale)


if __name__ == "__main__":
    unittest.main()
