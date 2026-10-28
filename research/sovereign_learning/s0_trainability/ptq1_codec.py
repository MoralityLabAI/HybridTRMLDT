"""CPU reference for PrismML's PTQ1_0 and PQ2_0 blocks, not a GGUF parser.

PrismML-Eng/llama.cpp ggml/src/ggml-quants.c, prism commit
c96baf4abe24d11a50f326e40b6223b687449d63.
PTQ1_0: 24 qs, 2 qh, FP16 scale, 28 bytes / 128 weights.
Hadamard and GGUF metadata are separate from the block codec.
"""
from __future__ import annotations
import math
import operator
import struct
from typing import Sequence

GROUP = 128
PTQ1_BLOCK_BYTES = 28
PQ2_BLOCK_BYTES = 34
GGUF_PTQ1_TYPE = 143
GGUF_PQ2_TYPE = 142


def _check_trits(trits: Sequence[int]) -> tuple[int, ...]:
    if len(trits) != GROUP:
        raise ValueError(f"expected {GROUP} ternary values, got {len(trits)}")
    values = []
    for value in trits:
        if isinstance(value, bool):
            raise ValueError("boolean values are not trits")
        try:
            trit = operator.index(value)
        except TypeError as exc:
            raise ValueError("trits require integer-index scalar values") from exc
        if trit not in (-1, 0, 1):
            raise ValueError("trits must be -1, 0, +1")
        values.append(trit)
    return tuple(values)


def _half(scale: float) -> bytes:
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and non-negative")
    try:
        return struct.pack("<e", scale)
    except (OverflowError, struct.error) as exc:
        raise ValueError("scale cannot be represented as FP16") from exc


def _byte_to_digit(code: int, place: int) -> int:
    return ((((code * (3 ** place)) & 0xff) * 3) >> 8) - 1


def encode_ptq1_block(trits: Sequence[int], scale: float) -> bytes:
    """Serialize previously chosen trits, without doing float quantization."""
    t = _check_trits(trits)
    qs = bytearray(24)
    pos = 0
    # The fork's 32/16/8 stage loop processes 24 bytes as 16 then 8.
    for start, width in ((0, 16), (16, 8)):
        for m in range(width):
            value = 0
            for n in range(5):
                value = 3 * value + t[pos + m + n * width] + 1
            qs[start + m] = (value * 256 + 242) // 243
        pos += 5 * width
    assert pos == 120
    qh = bytearray(2)
    for h in range(2):
        value = 0
        for n in range(4):
            value = 3 * value + t[120 + h + n * 2] + 1
        qh[h] = (value * 3 * 256 + 242) // 243
    return bytes(qs + qh) + _half(scale)


def decode_ptq1_block(block: bytes) -> tuple[tuple[int, ...], float]:
    if len(block) != PTQ1_BLOCK_BYTES:
        raise ValueError(f"PTQ1_0 block must have {PTQ1_BLOCK_BYTES} bytes")
    qs, qh = block[:24], block[24:26]
    scale = struct.unpack("<e", block[26:])[0]
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("invalid FP16 scale")
    values = []
    for start, width in ((0, 16), (16, 8)):
        for n in range(5):
            for m in range(width):
                values.append(_byte_to_digit(qs[start + m], n))
    for n in range(4):
        for h in range(2):
            values.append(_byte_to_digit(qh[h], n))
    if len(values) != GROUP or any(t not in (-1, 0, 1) for t in values):
        raise ValueError("invalid PTQ1_0 ternary codes")
    return tuple(values), scale


def encode_pq2_block(trits: Sequence[int], scale: float) -> bytes:
    """Encode the ternary subset of PQ2_0: FP16 scale before packed codes."""
    t = _check_trits(trits)
    qs = bytearray(32)
    for j, v in enumerate(t):
        qs[j // 4] |= (v + 1) << (2 * (j % 4))
    return _half(scale) + bytes(qs)


def decode_pq2_block(block: bytes) -> tuple[tuple[int, ...], float]:
    """Decode ternary PQ2_0; PrismML's 0b11 (+2) is deliberately rejected."""
    if len(block) != PQ2_BLOCK_BYTES:
        raise ValueError(f"PQ2_0 block must have {PQ2_BLOCK_BYTES} bytes")
    scale = struct.unpack("<e", block[:2])[0]
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("invalid FP16 scale")
    codes = [(block[2 + j // 4] >> (2 * (j % 4))) & 3 for j in range(GROUP)]
    if any(c == 3 for c in codes):
        raise ValueError("PQ2_0 contains +2 code, outside the ternary contract")
    return tuple(c - 1 for c in codes), scale


def decode_matrix(payload: bytes, *, rows: int, cols: int,
                  fmt: str = "ptq1") -> tuple[list[list[int]], list[list[float]]]:
    """Decode a bounded, already-located GGUF tensor payload.

    Does not parse GGUF offsets, strides, rotations, or model sharding.
    """
    if rows <= 0 or cols <= 0 or cols % GROUP:
        raise ValueError("positive dimensions and multiple-of-128 width required")
    if fmt == "ptq1":
        block_size, decoder = PTQ1_BLOCK_BYTES, decode_ptq1_block
    elif fmt == "pq2":
        block_size, decoder = PQ2_BLOCK_BYTES, decode_pq2_block
    else:
        raise ValueError(f"unsupported format: {fmt}")
    groups = cols // GROUP
    if len(payload) != rows * groups * block_size:
        raise ValueError("payload length disagrees with dimensions and format")
    q_rows, scale_rows = [], []
    pos = 0
    for _ in range(rows):
        row, scales = [], []
        for _ in range(groups):
            trits, scale = decoder(payload[pos:pos + block_size])
            row.extend(trits)
            scales.append(scale)
            pos += block_size
        q_rows.append(row)
        scale_rows.append(scales)
    return q_rows, scale_rows
